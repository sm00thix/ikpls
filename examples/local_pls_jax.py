"""
This script demonstrates distance-weighted local PLS for MANY unknown samples at once
with the JAX implementations of IKPLS.

Local PLS predicts each unknown sample with its own PLS model, fitted on the unknown's
nearest neighbours in a large library of spectra with known reference values (the
"pool"). The neighbours are often weighted by their distance to the unknown, and each
model's number of components is chosen by cross-validation on its neighbours. With many
unknowns, that is many small, independent models, which this script cross-validates,
fits and predicts in compiled programs with ``jax.vmap``:

  1. Find each unknown's ``N_NEIGHBOURS`` nearest pool samples with scikit-learn's
     ``NearestNeighbors``, and weight them by their distance to the unknown.
  2. Choose each unknown's number of components by weighted 10-fold cross-validation
     of its neighbourhood, with its own random folds: ``jax.vmap`` over
     ``ikpls.fast_cross_validation.jax.PLS.stateless_cross_validate``, with one weight
     vector per unknown. The neighbourhoods are gathered from the pool on the device.
  3. Fit each neighbourhood with ``ikpls.jax.PLS.stateless_fit`` (weighted) and predict
     its unknown with its own number of components.

Traced inputs cannot be checked inside the compiled program, so
``validate_folds`` checks the folds and weights of all unknowns on the host
beforehand. Invalid traced folds or weights would otherwise silently give wrong or NaN
results.

The synthetic spectra are mixtures of three constituents, and the predicted property
depends nonlinearly on their concentrations. A single global PLS model can only fit a
linear function of the spectra, while local models only need to be linear within each
neighbourhood. The script compares global PLS, unweighted local PLS and
distance-weighted local PLS.

Note: requires ``ikpls[jax]``. Reduce ``N_POOL`` and ``N_UNKNOWN`` for a quicker run.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ikpls.fast_cross_validation.jax import PLS as FastCVPLS
from ikpls.jax import PLS as JaxPLS
from ikpls.numpy import PLS as NpPLS

# Allow JAX to use 64-bit floating point precision.
jax.config.update("jax_enable_x64", True)


def make_spectra(rng: np.random.Generator, n: int, K: int):
    """
    Spectra (n, K) of mixtures of three constituents with overlapping absorption bands,
    a random baseline offset and noise, and a property (n, 1) that depends
    nonlinearly on the constituents' concentrations.
    """
    concentrations = rng.uniform(size=(n, 3))
    grid = np.linspace(0.0, 1.0, K)
    bands = np.stack(
        [np.exp(-0.5 * ((grid - mu) / 0.12) ** 2) for mu in (0.3, 0.5, 0.7)]
    )
    X = (
        concentrations @ bands
        + rng.uniform(-0.05, 0.05, size=(n, 1))
        + 0.002 * rng.normal(size=(n, K))
    )
    c0, c1, c2 = concentrations.T
    y = np.sin(2 * np.pi * c0) + 4 * (c1 - 0.5) ** 2 + 2 * c0 * c2
    return X, (y + 0.02 * rng.normal(size=n))[:, None]


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
    Tricube weights (B, N_NEIGHBOURS) of each unknown's neighbours, from their distances
    (B, N_NEIGHBOURS): 1 at distance 0, decreasing to near 0 at the farthest neighbour.
    The bandwidth is 10% wider than the farthest neighbour, so every weight is
    positive.
    """
    bandwidth = 1.1 * distances.max(axis=1, keepdims=True)
    return (1 - (distances / bandwidth) ** 3) ** 3


if __name__ == "__main__":
    N_POOL = 20000  # Spectra with known reference values.
    N_UNKNOWN = 2000  # Unknown samples to predict.
    K = 100  # Wavelength channels.
    N_NEIGHBOURS = 200  # Neighbours per unknown, i.e. samples per local model.
    A = 20  # Maximum number of components.
    N_FOLDS = 10  # Folds per local cross-validation (N_NEIGHBOURS must be divisible).
    BATCH_SIZE = 500  # How many unknowns to vmap at once (bounds device memory).

    rng = np.random.default_rng(seed=42)
    X_pool, Y_pool = make_spectra(rng, N_POOL, K)
    X_unknown, Y_unknown = make_spectra(rng, N_UNKNOWN, K)

    # ----- Global PLS: one model on the whole pool, A chosen by 10-fold fast CV. -----
    pls_kwargs = dict(center_X=True, center_Y=True, scale_X=False, scale_Y=False)
    t0 = time.perf_counter()
    pool_folds = np.arange(N_POOL) % N_FOLDS
    global_cv = FastCVPLS(algorithm=2, **pls_kwargs).cross_validate(
        X_pool, Y_pool, A, pool_folds, mse, show_progress=False
    )  # {fold: (A, 1)}
    global_A = int(np.argmin(np.mean(list(global_cv.values()), axis=0))) + 1
    global_pls = NpPLS(algorithm=2, **pls_kwargs).fit(X_pool, Y_pool, global_A)
    global_pred = global_pls.predict(X_unknown, n_components=global_A)
    global_time = time.perf_counter() - t0

    # ----- 1. Neighbour search and distance weights. -----
    t0 = time.perf_counter()
    distances, neighbours = (
        NearestNeighbors(n_neighbors=N_NEIGHBOURS).fit(X_pool).kneighbors(X_unknown)
    )
    weights = tricube_weights(distances)
    search_time = time.perf_counter() - t0
    # A random, equal-size 10-fold split of every unknown's neighbourhood: row f of
    # `val_idx[i]` holds the validation positions (0..N_NEIGHBOURS-1 within the
    # neighbourhood) of unknown i's fold f -> shape (N_UNKNOWN, N_FOLDS, N_VAL).
    val_idx = np.argsort(rng.random((N_UNKNOWN, N_NEIGHBOURS)), axis=1).reshape(
        N_UNKNOWN, N_FOLDS, N_NEIGHBOURS // N_FOLDS
    )

    # Algorithm #2 is the faster choice here because each fold trains on 180 > K
    # samples (Algorithm #1 is faster when K > N_train). Both algorithms yield the same
    # results up to floating-point round-off.
    cv_pls = FastCVPLS(algorithm=2, **pls_kwargs)
    fit_pls = JaxPLS(algorithm=2, **pls_kwargs)

    # Checks every unknown's folds and weights at once, before they are traced.
    cv_pls.validate_folds(val_idx, N_NEIGHBOURS, weights)

    def local_pls_one(nb_idx, w, val_idx_i, x_unknown, X_pool, Y_pool):
        """
        Local PLS for ONE unknown: cross-validate its neighbourhood, choose the number
        of components, fit on the whole neighbourhood and predict the unknown. `w` is
        None for unweighted local PLS.
        """
        X_nb, Y_nb = X_pool[nb_idx], Y_pool[nb_idx]  # (N_NEIGHBOURS, K), (.., 1)
        fold_mse = cv_pls.stateless_cross_validate(X_nb, Y_nb, A, val_idx_i, mse, w)
        best = jnp.argmin(fold_mse.mean(axis=0)[:, 0])  # (N_FOLDS, A, 1) -> 0-based
        # stateless_fit returns, for Algorithm #2,
        # (B, W, P, Q, R, max_stable_components, X_mean, Y_mean, X_std, Y_std).
        out = fit_pls.stateless_fit(X_nb, Y_nb, A, w)
        X_mean, Y_mean, X_std, Y_std = out[-4:]
        y_all = fit_pls.stateless_predict(
            x_unknown[None], out[0], None, X_mean, X_std, Y_mean, Y_std
        )  # (A, 1, 1): a prediction for every number of components
        return y_all[best, 0], best + 1

    # jit(vmap(...)) compiles one program for a whole batch of unknowns. The pool is
    # passed as an argument, not vmapped (in_axes=None), so it is not baked into the
    # compiled program as a constant.
    batched = jax.jit(jax.vmap(local_pls_one, in_axes=(0, 0, 0, 0, None, None)))
    X_pool_dev, Y_pool_dev = jnp.asarray(X_pool), jnp.asarray(Y_pool)

    def run_local_pls(weights):
        preds, n_components = [], []
        for start in range(0, N_UNKNOWN, BATCH_SIZE):
            sl = slice(start, start + BATCH_SIZE)
            pred, a = batched(
                jnp.asarray(neighbours[sl]),
                None if weights is None else jnp.asarray(weights[sl]),
                jnp.asarray(val_idx[sl]),
                jnp.asarray(X_unknown[sl]),
                X_pool_dev,
                Y_pool_dev,
            )
            preds.append(pred)
            n_components.append(a)
        # JAX dispatches asynchronously; block so the timings below are meaningful.
        return jax.block_until_ready(
            (
                np.asarray(jnp.concatenate(preds)),
                np.asarray(jnp.concatenate(n_components)),
            )
        )

    # ----- 2. + 3. Local CV, fit and predict for all unknowns. -----
    # The first call includes JIT compilation (cold); the second reuses it (warm).
    t0 = time.perf_counter()
    run_local_pls(weights)
    local_cold_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    weighted_pred, weighted_A = run_local_pls(weights)
    local_warm_time = time.perf_counter() - t0
    unweighted_pred, unweighted_A = run_local_pls(None)

    def rmse(pred):
        return float(np.sqrt(np.mean((pred - Y_unknown) ** 2)))

    def report(label, value):
        print(f"  {label:<46}: {value}")

    print(
        f"Pool of {N_POOL} spectra x {K} wavelengths, {N_UNKNOWN} unknowns, "
        f"{N_NEIGHBOURS} neighbours each, {N_FOLDS}-fold CV, up to A={A} components:"
    )
    report("global PLS (CV on the pool, fit, predict)", f"{global_time:7.2f} s")
    report("neighbour search (scikit-learn)", f"{search_time:7.2f} s")
    report("local PLS, all unknowns (cold, incl. compile)", f"{local_cold_time:7.2f} s")
    report("local PLS, all unknowns (warm)", f"{local_warm_time:7.2f} s")
    print("RMSE on the unknowns (the reference values have noise SD 0.02):")
    report(f"global PLS (A={global_A} by CV)", f"{rmse(global_pred):.4f}")
    report("local PLS, unweighted", f"{rmse(unweighted_pred):.4f}")
    report("local PLS, distance-weighted", f"{rmse(weighted_pred):.4f}")
    report(
        "components chosen per unknown (weighted)",
        f"min {weighted_A.min()}, median {int(np.median(weighted_A))}, "
        f"max {weighted_A.max()}",
    )
