"""
This script demonstrates local PLS for many unknown samples at once with the JAX
implementations of IKPLS, with the settings of our IKPLS 6.3.0 benchmark: 100,000 local
models, each fitted on its sample's 200 nearest neighbours, with its number of
components (1 to 30) chosen by 5-fold cross-validation. ``local_pls_numpy.py`` runs the
same local PLS with NumPy only.

Local PLS predicts each unknown sample with its own PLS model, fitted on the unknown's
nearest neighbours in a large library of spectra with known reference values (the
"pool"). Here, each of the pool's 100,000 spectra in turn plays the unknown, and its
model is fitted on its 200 nearest OTHER spectra, so its prediction can be compared with
its own reference value. The script runs these 100,000 small models in compiled
programs with ``jax.vmap``:

  1. Find each spectrum's 200 nearest other spectra, with FAISS if it is installed
     (``pip install faiss-cpu``), else with scikit-learn.
  2. Split each spectrum's neighbours into 5 folds by Venetian blinds over y.
  3. Cross-validate every neighbourhood with 1 to 30 components and choose the number
     of components with the lowest validation MSE:
     ``ikpls.fast_cross_validation.jax.PLS.stateless_cross_validate``.
  4. Refit every neighbourhood with its number of components and predict its spectrum:
     ``ikpls.jax.PLS.stateless_fit`` and ``stateless_predict``, with the models grouped
     by their number of components, which fixes the shapes of the fit.

Set ``K = 2800`` for all wavelengths or ``K = 100`` for every 28th of them, and
``DISTANCE_WEIGHTED = True`` to weight each model's neighbours by their distance.

On a CPU, ``local_pls_numpy.py`` is faster: this script runs in one process, where
``jax.vmap`` over many small models uses the cores poorly.

Note: requires ``ikpls[jax]`` 6.3.0 or later. Creating the spectra takes about 7 GB of
memory, and at K = 2800, a batch of 256 local models about 10 GB on the GPU: lower
``BATCH_SIZE`` on a smaller GPU. Reduce ``N_POOL`` for a quicker run.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ikpls.fast_cross_validation.jax import PLS as FastCVPLS
from ikpls.jax import PLS as JaxPLS

# Allow JAX to use 64-bit floating point precision.
jax.config.update("jax_enable_x64", True)


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
    neighbourhood's N neighbours. Returns the validation positions (0..N-1 within the
    neighbourhood) of every neighbourhood's folds, (P, n_folds, N // n_folds).
    """
    P, N = Y_nb.shape
    order = np.argsort(Y_nb, axis=1, kind="stable")
    return order.reshape(P, N // n_folds, n_folds).transpose(0, 2, 1)


def mse(Y_true: jnp.ndarray, Y_pred: jnp.ndarray, weights=None) -> jnp.ndarray:
    """
    Per-component, per-target MSE of one fold's validation set, weighted by the
    validation samples' weights if given. Must be JAX-traceable. `Y_true` has shape
    (N_val, M); `Y_pred` has shape (A, N_val, M); `weights` has shape (N_val,). Returns
    (A, M).
    """
    return jnp.average((Y_true - Y_pred) ** 2, axis=-2, weights=weights)


def tricube_weights(distances: np.ndarray) -> np.ndarray:
    """
    Tricube weights (P, N) of each neighbourhood's neighbours, from their distances
    (P, N): 1 at distance 0, decreasing to near 0 at the farthest neighbour. The
    bandwidth is 10% wider than the farthest neighbour, so every weight is positive.
    """
    bandwidth = 1.1 * distances.max(axis=1, keepdims=True)
    return (1 - (distances / bandwidth) ** 3) ** 3


def batches(ids: np.ndarray, size: int):
    """
    `ids` in batches of `size`, the last one padded by repeating its last id, so that
    all batches have the same shape and run in the same compiled program. Yields
    (the batch's ids, the padded batch).
    """
    for start in range(0, len(ids), size):
        batch = ids[start : start + size]
        yield batch, np.pad(batch, (0, size - len(batch)), mode="edge")


if __name__ == "__main__":
    N_POOL = 100_000  # Spectra with known reference values = local models.
    K = 100  # Wavelengths: 2800, or every (2800 // K)-th of them, e.g. K = 100.
    N_NEIGHBOURS = 200  # Neighbours per spectrum, i.e. samples per local model.
    A = 30  # Maximum number of components.
    N_FOLDS = 5  # Folds per local cross-validation (N_NEIGHBOURS must be divisible).
    DISTANCE_WEIGHTED = False  # True: weight the neighbours by their distance.
    # IKPLS Algorithm #1 is the faster choice when the training sets (here 160 samples
    # per fold) have fewer samples than wavelengths, and Algorithm #2 otherwise. Both
    # give the same results up to floating-point round-off.
    ALGORITHM = 1 if K > N_NEIGHBOURS - N_NEIGHBOURS // N_FOLDS else 2
    # Local models per program call. On the GPU, the fastest that fit in 24 GB; lower
    # them on a smaller GPU.
    if jax.default_backend() == "gpu":
        BATCH_SIZE = 512 if K <= 100 else 256
    else:
        BATCH_SIZE = 32 if K <= 100 else 1
    if 2800 % K != 0 or N_NEIGHBOURS % N_FOLDS != 0:
        raise ValueError("K must divide 2800, and N_FOLDS must divide N_NEIGHBOURS.")

    X_pool, Y_pool = make_spectra(N_POOL)
    X_pool = np.ascontiguousarray(X_pool[:, :: 2800 // K])

    # ----- 1. + 2. Neighbour search and folds (and distance weights). -----
    t0 = time.perf_counter()
    neighbours, distances, search_library = nearest_neighbours(X_pool, N_NEIGHBOURS)
    search_time = time.perf_counter() - t0
    val_idx = venetian_blinds(Y_pool[neighbours, 0], N_FOLDS)  # (N_POOL, 5, 40)
    weights = tricube_weights(distances) if DISTANCE_WEIGHTED else None

    pls_kwargs = dict(center_X=True, center_Y=True, scale_X=False, scale_Y=False)
    cv_pls = FastCVPLS(algorithm=ALGORITHM, **pls_kwargs)
    fit_pls = JaxPLS(algorithm=ALGORITHM, **pls_kwargs)

    # Traced folds and weights cannot be checked inside the compiled programs, and
    # invalid ones would silently give wrong or NaN results, so check them all here.
    cv_pls.validate_folds(val_idx, N_NEIGHBOURS, weights)

    # ----- 3. + 4. Local PLS: the compiled programs. -----
    # The pool, neighbours, folds and weights are passed as arguments, not closed over,
    # so they are not baked into the compiled programs as constants. Each program
    # gathers its batch's neighbourhoods from the pool on the device.
    @jax.jit
    def choose_components(X, Y, neighbours, weights, val_idx, ids):
        """
        Step 3 for the local models of the spectra `ids` (B,): cross-validate each
        one's neighbourhood and return its number of components (B,).
        """

        def one(nb, w, vi):
            fold_mse = cv_pls.stateless_cross_validate(X[nb], Y[nb], A, vi, mse, w)
            return jnp.argmin(fold_mse.mean(axis=0)[:, 0]) + 1  # (5, A, 1) -> 1..A

        w = None if weights is None else weights[ids]
        return jax.vmap(one)(neighbours[ids], w, val_idx[ids])

    @partial(jax.jit, static_argnames="n_components")
    def refit_predict(X, Y, neighbours, weights, ids, n_components):
        """
        Step 4 for the local models of the spectra `ids` (B,), which all chose
        `n_components`: fit each one on its whole neighbourhood and predict its
        spectrum (B,).
        """

        def one(nb, w, x):
            # stateless_fit returns B, the other fitted matrices and, last, X_mean,
            # Y_mean, X_std and Y_std.
            matrices = fit_pls.stateless_fit(X[nb], Y[nb], n_components, w)
            X_mean, Y_mean, X_std, Y_std = matrices[-4:]
            return fit_pls.stateless_predict(
                x[None], matrices[0], n_components, X_mean, X_std, Y_mean, Y_std
            )[0, 0]

        w = None if weights is None else weights[ids]
        return jax.vmap(one)(neighbours[ids], w, X[ids])

    def compile_programs():
        """
        Compiles every program on one batch: step 3, and step 4 for every number of
        components.
        """
        ids = jnp.asarray(np.arange(BATCH_SIZE) % N_POOL)
        jax.block_until_ready(choose_components(*pool, val_idx_dev, ids))
        for a in range(1, A + 1):
            jax.block_until_ready(refit_predict(*pool, ids, n_components=a))

    def run_local_pls():
        """
        Steps 3 and 4 for every spectrum: (predictions (N_POOL,), numbers of components
        (N_POOL,)). JAX dispatches asynchronously, so all batches of a step are queued
        before the first result is copied back.
        """
        chosen = [
            (batch, choose_components(*pool, val_idx_dev, jnp.asarray(padded)))
            for batch, padded in batches(np.arange(N_POOL), BATCH_SIZE)
        ]
        n_components = np.empty(N_POOL, dtype=int)
        for batch, a in chosen:
            n_components[batch] = np.asarray(a)[: len(batch)]
        refits = [
            (batch, refit_predict(*pool, jnp.asarray(padded), n_components=int(a)))
            for a in np.unique(n_components)
            for batch, padded in batches(np.flatnonzero(n_components == a), BATCH_SIZE)
        ]
        predictions = np.empty(N_POOL)
        for batch, pred in refits:
            predictions[batch] = np.asarray(pred)[: len(batch)]
        return predictions, n_components

    t0 = time.perf_counter()
    pool = jax.device_put((X_pool, Y_pool, neighbours, weights))
    val_idx_dev = jax.device_put(val_idx)
    compile_programs()
    compile_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    local_pred, local_A = run_local_pls()
    local_time = time.perf_counter() - t0

    def report(label, value):
        print(f"  {label:<46}: {value}")

    weighted = ", distance-weighted" if DISTANCE_WEIGHTED else ""
    print(
        f"{N_POOL:,} local PLS models on {jax.devices()[0].device_kind}: each "
        f"spectrum's {N_NEIGHBOURS} nearest neighbours among {N_POOL:,} spectra x {K} "
        f"wavelengths{weighted}, {N_FOLDS}-fold Venetian blinds over y, 1 to {A} "
        f"components, IKPLS Algorithm #{ALGORITHM}:"
    )
    report(f"neighbour search ({search_library})", f"{search_time:7.2f} s")
    report("local PLS: copy to the device and compile", f"{compile_time:7.2f} s")
    report(f"local PLS: all {N_POOL:,} models", f"{local_time:7.2f} s")
    report(
        "RMSE (the reference values have noise SD 0.5)",
        f"{np.sqrt(np.mean((local_pred - Y_pool[:, 0]) ** 2)):.4f}",
    )
    report(
        "components chosen per local model",
        f"min {local_A.min()}, median {int(np.median(local_A))}, max {local_A.max()}",
    )
