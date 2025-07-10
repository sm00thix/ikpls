"""
This file contains an example implementation of cross-validation using the NumPy
implementations of IKPLS. It demonstrates how to perform cross-validation with
sklearn's `cross_validate` column-wise centering and scaling. It also demonstrates
metric computation and evaluation.

The code includes the following functions:
- `cv_splitter`: A function to generate indices to split data into training and
    validation sets.
- `mse_for_each_target`: A function to compute the mean squared error for each target
    and the number of components that achieves the lowest MSE for each target.

To run the cross-validation, execute the file.

Note: The code assumes the availability of the `ikpls` package and its dependencies.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import numpy as np
from sklearn.model_selection import cross_validate

from ikpls.numpy_ikpls import PLS


def cv_splitter(folds: np.ndarray):
    """
    Generate indices to split data into training and validation sets.
    """
    uniq_folds = np.unique(folds)
    for split in uniq_folds:
        train_idxs = np.nonzero(folds != split)[0]
        val_idxs = np.nonzero(folds == split)[0]
        yield train_idxs, val_idxs


def mse_for_each_target(estimator, X: np.ndarray, Y_true: np.ndarray, **kwargs) -> dict:
    """
    Compute the mean squared error for each target and the number of components that
    achieves the lowest MSE for each target.
    """
    # Y_true has shape (N, M)
    Y_pred = estimator.predict(X, **kwargs)  # Shape (A, N, M)
    e = Y_true - Y_pred  # Shape (A, N, M)
    se = e**2  # Shape (A, N, M)
    mse = np.mean(se, axis=-2)  # Compute the mean over samples. Shape (A, M).

    # The number of components that minimizes the MSE for each target. Shape (M,).
    row_idxs = np.argmin(mse, axis=0)

    # The lowest MSE for each target. Shape (M,).
    lowest_mses = mse[row_idxs, np.arange(mse.shape[1])]

    # Indices are 0-indexed but number of components is 1-indexed.
    num_components = row_idxs + 1

    # List of names for the lowest MSE values.
    mse_names = [f"lowest_mse_target_{i}" for i in range(lowest_mses.shape[0])]

    # List of names for the number of components that
    # achieves the lowest MSE for each target.
    num_components_names = [
        f"num_components_lowest_mse_target_{i}" for i in range(lowest_mses.shape[0])
    ]
    all_names = mse_names + num_components_names  # List of all names.
    all_values = np.concatenate((lowest_mses, num_components))  # Array of all values.
    return dict(zip(all_names, all_values))


if __name__ == "__main__":
    N = 100  # Number of samples.
    K = 50  # Number of features.
    M = 10  # Number of targets.
    A = 20  # Number of latent variables (PLS components).
    folds = np.random.randint(
        0, 5, size=N
    )  # Randomly assign each sample to one of 5 folds.
    number_of_folds = np.unique(folds).shape[0]

    X = np.random.uniform(size=(N, K))
    Y = np.random.uniform(size=(N, M))

    # For this example, we will use IKPLS Algorithm #1.
    # The interface for IKPLS Algorithm #2 is identical.
    # Centering and scaling are enabled by default and computed over the
    # training folds only to avoid data leakage from the validation folds.
    np_pls_alg_1 = PLS(algorithm=1)
    params = {"A": A}
    np_pls_alg_1_results = cross_validate(
        np_pls_alg_1,
        X,
        Y,
        cv=cv_splitter(folds),
        scoring=mse_for_each_target,
        params=params,
        return_estimator=False,
        n_jobs=-1,
    )

    # Shape (M, folds) = (10, number_of_folds).
    # Lowest MSE for each target for each split.
    lowest_val_mses = np.array(
        [np_pls_alg_1_results[f"test_lowest_mse_target_{i}"] for i in range(M)]
    )

    # Shape (M, folds) = (10, number_of_folds).
    # Number of components that achieves the lowest MSE for each target for each split.
    best_num_components = np.array(
        [
            np_pls_alg_1_results[f"test_num_components_lowest_mse_target_{i}"]
            for i in range(M)
        ]
    )
