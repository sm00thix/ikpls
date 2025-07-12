"""
This file contains an example implementation of cross-validation using the JAX
implementations of IKPLS. It demonstrates how to perform cross-validation with
preprocessing, metric computation, and evaluation.

The code includes the following functions:
- `cross_val_preprocessing`: A function to apply preprocessing to each split of the
    cross-validation.
- `mse_per_component_and_best_components`: A function to compute the mean squared error
    per component and the best number of components.

To run the cross-validation, execute the file.

Note: The code assumes the availability of the `ikpls` package and its dependencies.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

# For this example, we will use IKPLS Algorithm #1.
# The interface for IKPLS Algorithm #2 is identical.
from ikpls.jax_ikpls_alg_1 import PLS

# Allow JAX to use 64-bit floating point precision.
jax.config.update("jax_enable_x64", True)


def cross_val_preprocessing(
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    Y_val: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Apply preprocessing to each split of the cross-validation. Here, we just use the
    identity function. This function will be applied before any potential centering and
    scaling The internals of .cross_validate() in JAX are JIT compiled. That includes
    the preprocessing function.
    """
    print("Preprocessing function will be JIT compiled...")
    return X_train, Y_train, X_val, Y_val


def mse_per_component_and_best_components(
    Y_true: jnp.ndarray,
    Y_pred: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the mean squared error per component and the best number of components.
    The internals of .cross_validate() in JAX are JIT compiled. That includes the
    metric function.
    """
    print("Metric function will be JIT compiled...")
    # Y_true has shape (N_val, M), Y_pred has shape (A, N_val, M).
    e = Y_true - Y_pred  # Shape (A, N_val, M)
    se = e**2  # Shape (A, N_val, M)
    mse = jnp.mean(se, axis=-2)  # Shape (A, M)
    best_num_components = jnp.argmin(mse, axis=0) + 1  # Shape (M,)
    return (mse, best_num_components)


if __name__ == "__main__":
    # NOTE: Every time a training or validation split has a different size from the
    # previously encountered one, recompilation will occur. This is because the JIT
    # compiler must generate a new function for each unique input shape.
    # Thus, if all splits have the same shape, JIT compilation happens only once.

    N = 100  # Number of samples.
    K = 50  # Number of features.
    M = 10  # Number of targets.
    A = 20  # Number of latent variables (PLS components).
    folds = np.arange(100) % 5  # Randomly assign each sample to one of 5 splits.

    X = np.random.uniform(size=(N, K))
    Y = np.random.uniform(size=(N, M))

    # For this example, we will use IKPLS Algorithm #1.
    # The interface for IKPLS Algorithm #2 is identical.
    # Centering and scaling are computed over the training splits
    # only to avoid data leakage from the validation splits.
    # ddof is the delta degrees of freedom for the standard deviation.
    # ddof=0 is the biased estimator, ddof=1 is the unbiased estimator.
    center_X = center_Y = scale_X = scale_Y = True
    ddof = 0
    jax_pls_alg_1 = PLS(
        center_X=center_X,
        center_Y=center_Y,
        scale_X=scale_X,
        scale_Y=scale_Y,
        ddof=ddof,
        verbose=True,
    )

    metric_names = ["mse", "best_num_components"]
    metric_values_dict = jax_pls_alg_1.cross_validate(
        X,
        Y,
        A,
        folds=folds,
        preprocessing_function=cross_val_preprocessing,
        metric_function=mse_per_component_and_best_components,
        metric_names=metric_names,
        weights=None,
    )

    """
    list of length 5 where each element is an array of shape (A, M) = (20, 10)
    corresponding to the mse output of mse_per_component_and_best_components for each
    split.
    """
    mse_for_each_split = metric_values_dict["mse"]

    # shape (n_splits, A, M) = (5, 20, 10)
    mse_for_each_split = np.array(mse_for_each_split)

    """
    list of length 5 where each element is an array of shape (M,) = (10,)
    corresponding to the best_num_components output of
    mse_per_component_and_best_components for each split.
    """
    best_num_components_for_each_split = metric_values_dict["best_num_components"]

    # shape (n_splits, M) = (5, 10)
    best_num_components_for_each_split = np.array(best_num_components_for_each_split)

    """
    # The -1 in the index is due to the fact that mse_for_each_split is 0-indexed but
    the number of components go from 1 to A. This could also have been implemented
    using jax.numpy in mse_per_component_and_best_components directly as part of the
    metric function.
    """
    # (n_splits, M) shape (5, 10)
    best_mse_for_each_split = np.amin(mse_for_each_split, axis=-2)

    # shape (n_splits, M) = (5, 10)
    equivalent_best_mse_for_each_split = np.array(
        [
            mse_for_each_split[
                i, best_num_components_for_each_split[i] - 1, np.arange(M)
            ]
            for i in range(len(best_num_components_for_each_split))
        ]
    )
    (best_mse_for_each_split == equivalent_best_mse_for_each_split).all()  # True
