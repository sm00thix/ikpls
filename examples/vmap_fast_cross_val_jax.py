"""
This script demonstrates how to combine the JAX fast cross-validation with ``jax.vmap``
to cross-validate MANY independent datasets at once -- here, 2,000 separate datasets of
200 samples and 110 wavelength channels, each with its own random 10-fold split -- far
faster than a Python loop over ``cross_validate``.

This is the right tool when you have many SEPARATE small models to cross-validate, e.g.
one local PLS model per unknown sample (built on its nearest neighbours in a large
spectral library), per spatial region/pixel, or per experimental condition. For a
SINGLE dataset, use ``cross_validate`` instead (it accepts arbitrary, size-imbalanced
folds and returns a dict).

Two things make this work:
  1. Use the STATELESS API (``stateless_cross_validate``). It takes the validation
     indices of equally sized folds as a 2-D array and returns the per-fold metrics
     stacked over the folds, stores nothing on the instance, and is therefore
     ``jax.vmap``-able over a leading dataset axis. The stateful ``cross_validate``
     (host-side fold bookkeeping, dict result) is not.
  2. The per-fold computation (cvmatrix rank-updates + the Improved Kernel PLS
     ``lax.scan`` fit) emits no host-side callbacks, so nested ``jax.vmap`` (datasets x
     folds) compiles to a single program under ``jax.jit``.

Peak device memory is kept bounded by processing the datasets in ``jax.vmap``'d chunks
of ``BATCH_SIZE``. Per dataset in a chunk, Algorithm #1 gathers every fold's training
rows (roughly ``1.3-2.3 * n_folds * N_train * K`` values), so use a smaller
``BATCH_SIZE`` for wide ``X``; Algorithm #2 holds the fold-wise kernel matrices (roughly
``2-4 * n_folds * K * K`` values). See the Notes of ``stateless_cross_validate``.

Note: requires ``ikpls[jax]``. Reduce ``N_DATASETS`` for a quicker run (keep it at least
``N_LOOP``).

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from ikpls.fast_cross_validation.jax import PLS

# Allow JAX to use 64-bit floating point precision.
jax.config.update("jax_enable_x64", True)


def mse_for_each_target(Y_true: jnp.ndarray, Y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Per-component, per-target MSE of one fold's validation set. Must be JAX-traceable.
    `Y_true` has shape (N_val, M); `Y_pred` has shape (A, N_val, M). Returns (A, M).
    """
    return jnp.mean((Y_true - Y_pred) ** 2, axis=-2)


if __name__ == "__main__":
    N_DATASETS = 2000  # Number of independent datasets to cross-validate.
    N = 200  # Samples per dataset.
    K = 110  # Features (e.g. wavelength channels) per dataset.
    M = 1  # Targets per dataset.
    A = 20  # Number of components (latent variables).
    N_FOLDS = 10  # Folds per dataset (equal size: N must be divisible by N_FOLDS).
    BATCH_SIZE = 500  # How many datasets to vmap at once (bounds device memory).
    N_LOOP = 20  # Datasets to time with a plain loop over `cross_validate`.

    rng = np.random.default_rng(seed=42)
    # Low-rank spectra-like data: 8 latent factors with smooth loadings plus noise, and
    # a target that depends on a random number (1..L) of the factors per dataset. The
    # CV-optimal number of components is then about L (the number of latent factors in
    # X), well below A.
    L = 8
    scores = rng.normal(size=(N_DATASETS, N, L)) / np.sqrt(np.arange(1, L + 1))
    grid = np.linspace(0.0, 1.0, K)
    loadings = np.stack(
        [np.exp(-0.5 * ((grid - c) / 0.08) ** 2) for c in rng.uniform(0.1, 0.9, L)]
    )
    X = scores @ loadings + 0.01 * rng.normal(size=(N_DATASETS, N, K))
    # Each dataset's target depends on a different number (1..L) of the factors.
    n_relevant = rng.integers(1, L + 1, size=N_DATASETS)
    coef = rng.normal(size=(N_DATASETS, L, M)) * (
        np.arange(L)[None, :, None] < n_relevant[:, None, None]
    )
    Y = np.einsum("bnl,blm->bnm", scores, coef) + 0.1 * rng.normal(
        size=(N_DATASETS, N, M)
    )
    # A random, equal-size 10-fold split for every dataset: row f of `val_idx[i]` holds
    # the validation indices of dataset i's fold f -> shape (N_DATASETS, N_FOLDS, N/N_FOLDS).
    val_idx = np.argsort(rng.random((N_DATASETS, N)), axis=1).reshape(
        N_DATASETS, N_FOLDS, N // N_FOLDS
    )

    # Algorithm #2 is the faster choice here because N_train > K (Algorithm #1 is faster
    # when K > N_train). Both algorithms yield the same results up to floating-point
    # round-off.
    pls = PLS(algorithm=2, center_X=True, center_Y=True, scale_X=False, scale_Y=False)
    # Inside jax.vmap, each dataset's folds are traced and cannot be checked, so check
    # the folds of all datasets at once beforehand.
    pls.validate_folds(val_idx, N)

    # ----- Baseline: a plain Python loop over datasets with `cross_validate`. -----
    # `cross_validate` takes fold LABELS (one per sample); build them from the indices.
    def labels_from_val_idx(vi):
        labels = np.empty(N, dtype=int)
        for f in range(N_FOLDS):
            labels[vi[f]] = f
        return labels

    loop_mse = np.empty((N_LOOP, N_FOLDS, A, M))
    t0 = time.perf_counter()
    for i in range(N_LOOP):
        metrics = pls.cross_validate(
            X[i],
            Y[i],
            A,
            labels_from_val_idx(val_idx[i]),
            mse_for_each_target,
            show_progress=False,
        )
        loop_mse[i] = np.stack([np.asarray(metrics[f]) for f in range(N_FOLDS)])
    loop_time = time.perf_counter() - t0

    # ----- JAX + jax.vmap over the datasets. -----
    def cv_one_dataset(X_i, Y_i, val_idx_i):
        """10-fold fast CV of ONE dataset -> (N_FOLDS, A, M) fold-wise MSE."""
        return pls.stateless_cross_validate(X_i, Y_i, A, val_idx_i, mse_for_each_target)

    # jit(vmap(...)) compiles a single program that cross-validates a whole chunk of
    # datasets (all their folds) in parallel on the CPU/GPU/TPU.
    batched_cv = jax.jit(jax.vmap(cv_one_dataset))

    def run_vmap():
        out = []
        for start in range(0, N_DATASETS, BATCH_SIZE):
            sl = slice(start, start + BATCH_SIZE)
            out.append(
                batched_cv(
                    jnp.asarray(X[sl]), jnp.asarray(Y[sl]), jnp.asarray(val_idx[sl])
                )
            )
        out = jnp.concatenate(out, axis=0)  # (N_DATASETS, N_FOLDS, A, M)
        # JAX dispatches asynchronously; block so the timings below are meaningful.
        return jax.block_until_ready(out)

    # The first call includes JIT compilation (cold); the second reuses the cache (warm).
    t0 = time.perf_counter()
    fold_mse = run_vmap()
    vmap_cold_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    fold_mse = run_vmap()
    vmap_warm_time = time.perf_counter() - t0

    # Model selection per dataset: average the fold-wise MSE, pick the best A (1-indexed).
    mean_mse = fold_mse.mean(axis=1)  # (N_DATASETS, A, M)
    best_num_components = np.asarray(
        jnp.argmin(mean_mse, axis=1) + 1
    )  # (N_DATASETS, M)

    # The vmapped results match the loop over `cross_validate` (same per-fold function);
    # any difference is floating-point round-off from XLA fusing the ops differently.
    max_abs_diff = np.max(np.abs(np.asarray(fold_mse[:N_LOOP]) - loop_mse))

    print(
        f"{N_DATASETS} datasets of {N}x{K} with {M} target(s), {N_FOLDS}-fold CV, "
        f"A={A} components, Algorithm #2:"
    )
    print(
        f"  loop over cross_validate ({N_LOOP} datasets): {loop_time:8.3f} s  "
        f"-> {loop_time / N_LOOP * N_DATASETS:8.1f} s projected for all {N_DATASETS}"
    )
    print(f"  JAX vmap (cold, incl. compilation)        : {vmap_cold_time:8.3f} s")
    print(f"  JAX vmap (warm)                           : {vmap_warm_time:8.3f} s")
    print(
        "  warm speedup vs projected loop            : "
        f"{loop_time / N_LOOP * N_DATASETS / vmap_warm_time:7.1f}x"
    )
    print(f"  max |vmap - loop| over the first {N_LOOP} datasets: {max_abs_diff:.2e}")
    print(
        "  best number of components, first 10 datasets: "
        f"{best_num_components[:10, 0].tolist()}"
    )
