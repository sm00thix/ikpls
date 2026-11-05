"""
This script demonstrates local PLS for many unknown samples with the NumPy
implementations of IKPLS, with the settings of our IKPLS 6.3.0 benchmark: 100,000 local
models, each fitted on its sample's 200 nearest neighbours, with its number of
components (1 to 30) chosen by 5-fold cross-validation. It needs no JAX.
``local_pls_jax.py`` runs the same local PLS with JAX and gives the same results.

Local PLS predicts each unknown sample with its own PLS model, fitted on the unknown's
nearest neighbours in a large library of spectra with known reference values (the
"pool"). Here, each of the pool's 100,000 spectra in turn plays the unknown, and its
model is fitted on its 200 nearest OTHER spectra, so its prediction can be compared with
its own reference value. The script spreads these 100,000 small models over the CPU
cores with joblib:

  1. Find each spectrum's 200 nearest other spectra, with FAISS if it is installed
     (``pip install faiss-cpu``), else with scikit-learn.
  2. Split each spectrum's neighbours into 5 folds by Venetian blinds over y.
  3. Cross-validate every neighbourhood with 1 to 30 components and choose the number
     of components with the lowest validation MSE:
     ``ikpls.fast_cross_validation.numpy.PLS.cross_validate``.
  4. Refit every neighbourhood with its number of components and predict its spectrum:
     ``ikpls.numpy.PLS.fit`` and ``predict``.

Set ``K = 2800`` for all wavelengths or ``K = 100`` for every 28th of them, and
``DISTANCE_WEIGHTED = True`` to weight each model's neighbours by their distance.

Note: requires ikpls 6.3.0 or later; earlier versions print a line for every
cross-validation. Creating the spectra takes about 7 GB of memory. Reduce ``N_POOL`` for
a quicker run.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import os
import tempfile
import time
from typing import Optional

import joblib
import numpy as np
from joblib import Parallel, delayed

from ikpls.fast_cross_validation.numpy import PLS as FastCVPLS
from ikpls.numpy import PLS as NpPLS


def make_spectra(n: int, K: int = 2800, seed: int = 42):
    """
    Create a synthetic dataset of NIR-like spectra X (n, K) and a property y (n, 1).
    """
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(0.0, 1.0, K)
    centres = rng.normal(size=(12, 4))
    C = centres[rng.integers(12, size=n)] + 0.6 * rng.normal(size=(n, 4))
    factors = [C[:, i] for i in range(4)]
    factors += [C[:, i] * C[:, j] for i in range(4) for j in range(i, 4)]
    factors += [np.tanh(C[:, i]) for i in range(4)]
    S = np.stack(factors, axis=1)
    S = (S - S.mean(axis=0)) / S.std(axis=0)
    S = np.concatenate([S, rng.normal(size=(n, 22))], axis=1)
    S *= 1.0 / (1.0 + np.arange(S.shape[1])) ** 0.6  # Decaying factor variances.
    loadings = np.zeros((S.shape[1], K))
    for i in range(S.shape[1]):
        n_bands = rng.integers(2, 6)
        centre = rng.uniform(0.05, 0.95, n_bands)[:, None]
        width = rng.uniform(0.01, 0.08, n_bands)[:, None]
        amplitude = rng.normal(size=n_bands)[:, None]
        band = amplitude * np.exp(-0.5 * ((wavelengths - centre) / width) ** 2)
        loadings[i] = band.sum(axis=0) / np.sqrt(np.mean(band.sum(axis=0) ** 2))
    X = (S @ loadings) * 0.1
    offset, slope = rng.normal(0.0, 0.05, (n, 1)), rng.normal(0.0, 0.05, (n, 1))
    X += offset + slope * (wavelengths - 0.5)
    X += 1.0 + 0.5 * wavelengths  # The common baseline.
    for start in range(0, n, 5000):  # The noise in chunks, to save memory.
        X[start : start + 5000] += rng.normal(0.0, 5e-3, (min(5000, n - start), K))
    c0, c1, c2, c3 = C.T
    y = c0 + 0.5 * c1**2 + 0.4 * c0 * c2 + 0.3 * np.tanh(2 * c3)
    y = (y - y.mean()) / y.std() * 2.0 + 12.0 + rng.normal(0.0, 0.5, n)
    return X, y[:, None]


def nearest_neighbours(X: np.ndarray, k: int):
    """
    Each row's `k` nearest OTHER rows of `X` by exact Euclidean distance, nearest
    first: their indices (n, k), their distances (n, k) and the library used. With
    FAISS if it is installed, else with scikit-learn.
    """
    try:
        import faiss
    except ImportError:
        from sklearn.neighbors import NearestNeighbors

        library = "scikit-learn"
        distances, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    else:
        library = "FAISS"
        # FAISS searches in float32. Centring first leaves the distances unchanged but
        # reduces their float32 round-off.
        X32 = np.ascontiguousarray(X - X.mean(axis=0), dtype=np.float32)
        index = faiss.IndexFlatL2(X.shape[1])  # Exact (brute-force) search.
        index.add(X32)
        squared_distances, idx = index.search(X32, k + 1)
        distances = np.sqrt(np.maximum(squared_distances, 0))
    # Drop each row itself: normally its own nearest neighbour, but if an exact
    # duplicate pushed it out of the k + 1 nearest, drop the farthest instead.
    is_self = idx == np.arange(len(X))[:, None]
    is_self[~is_self.any(axis=1), -1] = True
    shape = (len(X), k)
    return idx[~is_self].reshape(shape), distances[~is_self].reshape(shape), library


def venetian_blinds(Y_nb: np.ndarray, n_folds: int) -> np.ndarray:
    """
    Venetian-blinds folds over y of every neighbourhood: sorted by y, the r-th smallest
    neighbour goes to fold r mod `n_folds`. `Y_nb` (P, N) holds the y-values of each
    neighbourhood's N neighbours. Returns the fold of every neighbour, (P, N). These
    are the same folds as in local_pls_jax.py.
    """
    order = np.argsort(Y_nb, axis=1, kind="stable")
    folds = np.empty_like(order)
    np.put_along_axis(folds, order, np.arange(Y_nb.shape[1]) % n_folds, axis=1)
    return folds


def mse(Y_true: np.ndarray, Y_pred: np.ndarray, weights=None) -> np.ndarray:
    """
    Per-component, per-target MSE of one fold's validation set, weighted by the
    validation samples' weights if given. `Y_true` has shape (N_val, M); `Y_pred` has
    shape (A, N_val, M); `weights` has shape (N_val,). Returns (A, M).
    """
    return np.average((Y_true - Y_pred) ** 2, axis=-2, weights=weights)


def tricube_weights(distances: np.ndarray) -> np.ndarray:
    """
    Tricube weights (P, N) of each neighbourhood's neighbours, from their distances
    (P, N): 1 at distance 0, decreasing to near 0 at the farthest neighbour. The
    bandwidth is 10% wider than the farthest neighbour, so every weight is positive.
    """
    bandwidth = 1.1 * distances.max(axis=1, keepdims=True)
    return (1 - (distances / bandwidth) ** 3) ** 3


def local_pls_chunk(
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    ids: np.ndarray,
    neighbours: np.ndarray,
    folds: np.ndarray,
    weights: Optional[np.ndarray],
    A: int,
    algorithm: int,
    pls_kwargs: dict,
):
    """
    Steps 3 and 4 for the local models of the spectra `ids` (B,), one after the other:
    cross-validate each one's neighbourhood, choose its number of components, refit on
    the whole neighbourhood with that many components and predict its spectrum.
    `neighbours`, `folds` and `weights` have shape (B, N_NEIGHBOURS); `weights` is None
    for unweighted local PLS. Returns the predictions (B,) and the chosen numbers of
    components (B,).
    """
    cv_pls = FastCVPLS(algorithm=algorithm, **pls_kwargs)
    fit_pls = NpPLS(algorithm=algorithm, **pls_kwargs)
    preds = np.empty(len(ids))
    n_components = np.empty(len(ids), dtype=int)
    for i, (q, nb_idx) in enumerate(zip(ids, neighbours)):
        X_nb, Y_nb = X_pool[nb_idx], Y_pool[nb_idx]  # (N_NEIGHBOURS, K), (.., 1)
        w = None if weights is None else weights[i]
        # The chunks already run in parallel, so each cross-validation runs in this
        # process (n_jobs=1) and prints nothing (verbose=0).
        fold_mse = cv_pls.cross_validate(
            X_nb, Y_nb, A, folds[i], mse, w, n_jobs=1, verbose=0
        )  # {fold: (A, 1)}
        best = int(np.argmin(np.mean(list(fold_mse.values()), axis=0)[:, 0])) + 1
        fit_pls.fit(X_nb, Y_nb, best, w)
        preds[i] = fit_pls.predict(X_pool[q : q + 1], n_components=best)[0, 0]
        n_components[i] = best
    return preds, n_components


if __name__ == "__main__":
    N_POOL = 100_000  # Spectra with known reference values = local models.
    K = 100  # Wavelengths: 2800, or every (2800 // K)-th of them, e.g. K = 100.
    N_NEIGHBOURS = 200  # Neighbours per spectrum, i.e. samples per local model.
    A = 30  # Maximum number of components.
    N_FOLDS = 5  # Folds per local cross-validation.
    DISTANCE_WEIGHTED = False  # True: weight the neighbours by their distance.
    # IKPLS Algorithm #1 is the faster choice when the training sets (here 160 samples
    # per fold) have fewer samples than wavelengths, and Algorithm #2 otherwise. Both
    # give the same results up to floating-point round-off.
    ALGORITHM = 1 if K > N_NEIGHBOURS - N_NEIGHBOURS // N_FOLDS else 2
    # Worker processes and BLAS threads per worker, the fastest on a 16-core CPU with
    # 32 threads: one single-threaded worker per core at K = 100, and half as many
    # workers with 4 threads each at K = 2800.
    cores = joblib.cpu_count(only_physical_cores=True)
    N_JOBS = cores if K <= 100 else max(cores // 2, 1)
    THREADS = 1 if K <= 100 else max(joblib.cpu_count() // N_JOBS, 1)
    # How many chunks of local models to spread over the workers: many small chunks per
    # worker, so that the workers finish at about the same time.
    N_CHUNKS = 32 * N_JOBS

    if 2800 % K != 0:
        raise ValueError("K must divide 2800.")

    X_pool, Y_pool = make_spectra(N_POOL)
    X_pool = np.ascontiguousarray(X_pool[:, :: 2800 // K])
    pls_kwargs = dict(center_X=True, center_Y=True, scale_X=False, scale_Y=False)

    # ----- 1. + 2. Neighbour search and folds (and distance weights). -----
    t0 = time.perf_counter()
    neighbours, distances, search_library = nearest_neighbours(X_pool, N_NEIGHBOURS)
    search_time = time.perf_counter() - t0
    folds = venetian_blinds(Y_pool[neighbours, 0], N_FOLDS)  # (N_POOL, N_NEIGHBOURS)
    weights = tricube_weights(distances) if DISTANCE_WEIGHTED else None

    # ----- 3. + 4. Local PLS. -----
    def run_local_pls(X_pool, ids):
        chunks = np.array_split(ids, N_CHUNKS)
        results = Parallel(n_jobs=N_JOBS)(
            delayed(local_pls_chunk)(
                X_pool,
                Y_pool,
                c,
                neighbours[c],
                folds[c],
                None if weights is None else weights[c],
                A,
                ALGORITHM,
                pls_kwargs,
            )
            for c in chunks
        )
        preds, n_components = zip(*results)
        return np.concatenate(preds), np.concatenate(n_components)

    with (
        tempfile.TemporaryDirectory() as tmp,
        joblib.parallel_config("loky", inner_max_num_threads=THREADS),
    ):
        # The workers read the pool from a memory-mapped file, instead of joblib
        # copying it to a file for every call.
        t0 = time.perf_counter()
        np.save(os.path.join(tmp, "X_pool.npy"), X_pool)
        X_pool_mmap = np.load(os.path.join(tmp, "X_pool.npy"), mmap_mode="r")
        run_local_pls(X_pool_mmap, np.arange(N_CHUNKS))  # Starts the workers.
        startup_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        local_pred, local_A = run_local_pls(X_pool_mmap, np.arange(N_POOL))
        local_time = time.perf_counter() - t0
        del X_pool_mmap

    def report(label, value):
        print(f"  {label:<46}: {value}")

    weighted = ", distance-weighted" if DISTANCE_WEIGHTED else ""
    print(
        f"{N_POOL:,} local PLS models in {N_JOBS} processes x {THREADS} threads: each "
        f"spectrum's {N_NEIGHBOURS} nearest neighbours among {N_POOL:,} spectra x {K} "
        f"wavelengths{weighted}, {N_FOLDS}-fold Venetian blinds over y, 1 to {A} "
        f"components, IKPLS Algorithm #{ALGORITHM}:"
    )
    report(f"neighbour search ({search_library})", f"{search_time:7.2f} s")
    report("local PLS: memory-map the pool, start workers", f"{startup_time:7.2f} s")
    report(f"local PLS: all {N_POOL:,} models", f"{local_time:7.2f} s")
    report(
        "RMSE (the reference values have noise SD 0.5)",
        f"{np.sqrt(np.mean((local_pred - Y_pool[:, 0]) ** 2)):.4f}",
    )
    report(
        "components chosen per local model",
        f"min {local_A.min()}, median {int(np.median(local_A))}, max {local_A.max()}",
    )
