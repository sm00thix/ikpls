"""
This file contains an example of the fast JAX cross-validation in IKPLS
(`ikpls.fast_cross_validation.jax_ikpls.PLS`). It mirrors `fast_cross_val_numpy.py`,
but the per-fold training matrices are computed by rank-update with the `cvmatrix` JAX
backend and the folds are batched together with `jax.vmap`, running on CPU/GPU/TPU.

Two things differ from the NumPy fast cross-validation:
  - The `metric_function` must be JAX-traceable (it is evaluated on device, inside the
    vmapped per-fold computation), so use `jax.numpy` in it.
  - There is no `n_jobs`; instead a `batch_size` argument caps how many folds are
    vmapped together to bound peak memory. `None` vmaps all folds of a given
    validation-set size at once.

Both Improved Kernel PLS Algorithm #1 and #2 are supported and give identical results.
Algorithm #2 (XTX-based) is efficient for tall `X`; Algorithm #1 (X-based) avoids
forming XTX and is efficient for wide `X` (but materializes per-fold training rows, so
prefer a modest `batch_size` there).

Note: requires `ikpls[jax]` and `cvmatrix[jax]`.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import jax
import jax.numpy as jnp
import numpy as np

from ikpls.fast_cross_validation.jax_ikpls import PLS

# Allow JAX to use 64-bit floating point precision.
jax.config.update("jax_enable_x64", True)


def mse_for_each_target(Y_true: jnp.ndarray, Y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the mean squared error for each number of components and each target.

    Must be JAX-traceable: it runs on device inside jax.vmap over the folds.
    `Y_true` has shape (N_val, M); `Y_pred` has shape (A, N_val, M). Returns the per-
    component, per-target MSE of shape (A, M).
    """
    return jnp.mean((Y_true - Y_pred) ** 2, axis=-2)


if __name__ == "__main__":
    N = 100  # Number of samples.
    K = 50  # Number of features.
    M = 10  # Number of targets.
    A = 20  # Number of latent variables (PLS components).
    splits = np.arange(N) % 5  # Assign each sample to one of 5 splits.

    X = np.random.uniform(size=(N, K))
    Y = np.random.uniform(size=(N, M))

    # Centering and scaling are computed over the training splits only, per fold, to
    # avoid data leakage. ddof is the delta degrees of freedom for the standard
    # deviation (0 = biased, 1 = unbiased). Algorithm #1 and #2 give identical results.
    jax_fast_cv = PLS(
        algorithm=1,
        center_X=True,
        center_Y=True,
        scale_X=True,
        scale_Y=True,
        ddof=0,
    )

    # `metrics` maps each unique value in `splits` to the result of `mse_for_each_target`
    # for that fold's validation set. `batch_size` bounds how many folds are vmapped
    # together (None = all folds of a given size at once).
    metrics = jax_fast_cv.cross_validate(
        X=X,
        Y=Y,
        A=A,
        folds=splits,
        metric_function=mse_for_each_target,
        weights=None,
        batch_size=None,
        show_progress=True,
    )

    unique_splits = np.unique(splits)

    # Stack the per-fold (A, M) MSE arrays -> (n_splits, A, M).
    mse_per_split = np.asarray([np.asarray(metrics[s]) for s in unique_splits])

    # Lowest MSE per target per split -> (n_splits, M); and the best number of
    # components (1-indexed) per target per split -> (n_splits, M).
    best_mse_per_split = np.min(mse_per_split, axis=1)
    best_num_components_per_split = np.argmin(mse_per_split, axis=1) + 1

    print("Lowest validation MSE per split (rows) and target (cols):")
    print(best_mse_per_split)
    print("Best number of components per split (rows) and target (cols):")
    print(best_num_components_per_split)
