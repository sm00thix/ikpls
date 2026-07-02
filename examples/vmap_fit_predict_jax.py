"""
This script demonstrates how to use ``jax.vmap`` to fit AND predict with the JAX
implementation of IKPLS on MANY independent datasets at once -- here, 10,000 separate
datasets of 200 samples and 110 wavelength channels -- far faster than a Python loop.

This is the right tool when you have many SEPARATE small models to fit (e.g. one model
per spatial region/pixel, per experimental condition, or a data-variation sweep). For a
SINGLE dataset evaluated with k-fold or leave-one-out cross-validation, use
``cross_validate`` or the fast cross-validation classes
(``ikpls.fast_cross_validation``) instead.

Two things make this work:
  1. Use the STATELESS API (``stateless_fit`` / ``stateless_predict``). The stateful
     ``fit``/``predict`` store results on the instance and are not ``jax.vmap``-friendly.
  2. The JAX fit emits no host-side callbacks, so it is directly ``jax.vmap``-able (no
     flags or special construction required).

Peak device memory is kept bounded by processing the datasets in ``jax.vmap``'d chunks
of ``BATCH_SIZE``.

Note: requires the ``ikpls[jax]`` package. Reduce ``N_DATASETS`` for a quicker run.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from ikpls.jax import PLS as JaxPLS
from ikpls.numpy import PLS as NpPLS

# Allow JAX to use 64-bit floating point precision.
jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":
    N_DATASETS = 10000  # Number of independent datasets to fit and predict.
    N = 200  # Samples per dataset.
    K = 110  # Features (e.g. wavelength channels) per dataset.
    M = 1  # Targets per dataset.
    A = 30  # Number of components (latent variables).
    BATCH_SIZE = 2000  # How many datasets to vmap at once (bounds device memory).

    # Algorithm #2 is the faster choice here because N > K (Algorithm #1 is faster when
    # K > N). Both algorithms yield identical results.
    kwargs = dict(center_X=True, center_Y=True, scale_X=True, scale_Y=True, ddof=0)

    rng = np.random.default_rng(seed=42)
    X = rng.uniform(size=(N_DATASETS, N, K))
    Y = rng.uniform(size=(N_DATASETS, N, M))

    # ----- Baseline: a plain Python loop over the datasets with the NumPy backend. -----
    np_pls = NpPLS(algorithm=2, **kwargs)
    np_preds = np.empty((N_DATASETS, A, N, M))
    t0 = time.perf_counter()
    for i in range(N_DATASETS):
        np_pls.fit(X[i], Y[i], A)
        # predict returns shape (A, N, M): a prediction for every number of components
        # from 1 up to and including A.
        np_preds[i] = np_pls.predict(X[i])
    numpy_time = time.perf_counter() - t0

    # ----- JAX + jax.vmap over the datasets. -----
    pls = JaxPLS(algorithm=2, **kwargs)

    def fit_predict(X_i, Y_i):
        """Fit one dataset and predict on it, using the (vmap-able) stateless API."""
        # stateless_fit returns, for Algorithm #2,
        # (B, W, P, Q, R, max_stable_components, X_mean, Y_mean, X_std, Y_std):
        # B is the first element and the four preprocessing statistics are the last four.
        out = pls.stateless_fit(X_i, Y_i, A)
        B = out[0]
        X_mean, Y_mean, X_std, Y_std = out[-4:]
        return pls.stateless_predict(
            X_i, B, X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std
        )

    # jit(vmap(...)) compiles a single fit+predict once and runs it across a whole chunk
    # of datasets in parallel.
    batched_fit_predict = jax.jit(jax.vmap(fit_predict))

    def run_vmap():
        preds = []
        for start in range(0, N_DATASETS, BATCH_SIZE):
            X_chunk = jnp.asarray(X[start : start + BATCH_SIZE])
            Y_chunk = jnp.asarray(Y[start : start + BATCH_SIZE])
            preds.append(batched_fit_predict(X_chunk, Y_chunk))
        out = jnp.concatenate(preds, axis=0)  # (N_DATASETS, A, N, M)
        # JAX dispatches asynchronously; block so the timings below are meaningful.
        return jax.block_until_ready(out)

    # The first call includes JIT compilation (cold); the second reuses the cache (warm).
    t0 = time.perf_counter()
    jax_preds = run_vmap()
    jax_cold_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    jax_preds = run_vmap()
    jax_warm_time = time.perf_counter() - t0

    # The vmapped JAX predictions match the NumPy loop (same algorithm); the difference
    # is only floating-point round-off.
    max_abs_diff = np.max(np.abs(np.asarray(jax_preds) - np_preds))

    print(
        f"{N_DATASETS} datasets of {N}x{K} with {M} target(s), A={A} components, "
        f"Algorithm #2:"
    )
    print(f"  NumPy loop           : {numpy_time:8.3f} s")
    print(f"  JAX vmap (cold)      : {jax_cold_time:8.3f} s  (includes compilation)")
    print(f"  JAX vmap (warm)      : {jax_warm_time:8.3f} s")
    print(
        f"  cold speedup vs loop : {numpy_time / jax_cold_time:7.1f}x"
        "  (one-shot, including compilation)"
    )
    print(f"  warm speedup vs loop : {numpy_time / jax_warm_time:7.1f}x")
    print(f"  max |JAX - NumPy| over all predictions: {max_abs_diff:.2e}")
