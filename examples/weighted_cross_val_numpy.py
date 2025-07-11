"""
This file contains an example implementation of weighted cross-validation using IKPLS.
It demonstrates how to perform weighted cross-validation with column-wise centering and
scaling. It also demonstrates metric computation and evaluation.

The code includes the following functions:
- `wmse_for_each_target`: A function to compute the weigthed mean squared error for each
    target and the number of components that achieves the lowest MSE for each target.

To run the cross-validation, execute the file.

Note: The code assumes the availability of the `ikpls` package and its dependencies.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import numpy as np

from ikpls.numpy_ikpls import PLS


def cross_val_preprocessing(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    train_weights: np.ndarray,
    val_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply preprocessing to each split of the cross-validation. Here, we just use the
    identity function. This function will be applied before any potential centering and
    scaling.
    """
    return X_train, Y_train, X_val, Y_val


def wmse_for_each_target(
    Y_true: np.ndarray, Y_pred: np.ndarray, val_weights: np.ndarray
):
    """
    We can return anything we want. Here, we compute the mean squared error for each
    target and the number of components that achieves the lowest MSE for each target.
    """
    # Y_true has shape (N, M)
    # Y_pred has shape (A, N, M)
    e = Y_true - Y_pred  # Shape (A, N, M)
    se = e**2  # Shape (A, N, M)

    # Compute the weighted mean over samples. Shape (A, M).
    wmse = np.average(se, axis=-2, weights=val_weights)

    # The number of components that minimizes the MSE for each target. Shape (M,).
    row_idxs = np.argmin(wmse, axis=0)

    # The lowest MSE for each target. Shape (M,).
    lowest_wmses = wmse[row_idxs, np.arange(wmse.shape[1])]

    # Indices are 0-indexed but number of components is 1-indexed.
    num_components = row_idxs + 1

    # List of names for the lowest MSE values.
    wmse_names = [f"lowest_mse_target_{i}" for i in range(lowest_wmses.shape[0])]

    # List of names for the number of components that
    # achieves the lowest MSE for each target.
    num_components_names = [
        f"num_components_lowest_mse_target_{i}" for i in range(lowest_wmses.shape[0])
    ]
    all_names = wmse_names + num_components_names  # List of all names.
    all_values = np.concatenate((lowest_wmses, num_components))  # Array of all values.
    return dict(zip(all_names, all_values))


if __name__ == "__main__":
    N = 100  # Number of samples.
    K = 50  # Number of features.
    M = 10  # Number of targets.
    A = 20  # Number of latent variables (PLS components).
    splits = np.random.randint(
        0, 5, size=N
    )  # Randomly assign each sample to one of 5 splits.
    number_of_splits = np.unique(splits).shape[0]

    X = np.random.uniform(size=(N, K))
    Y = np.random.uniform(size=(N, M))
    weights = np.random.uniform(low=0, high=2, size=(N,))

    # For this example, we will use IKPLS Algorithm #1.
    # The interface for IKPLS Algorithm #2 is identical.
    # Centering and scaling are computed over the training splits
    # only to avoid data leakage from the validation splits.
    # ddof is the delta degrees of freedom for the standard deviation.
    # ddof=0 is the biased estimator, ddof=1 is the unbiased estimator.
    algorithm = 1
    center_X = center_Y = scale_X = scale_Y = True
    ddof = 0
    np_pls_alg_1 = PLS(
        algorithm=algorithm,
        center_X=center_X,
        center_Y=center_Y,
        scale_X=scale_X,
        scale_Y=scale_Y,
        ddof=ddof,
    )
    np_pls_alg_1_cv_wmses = np_pls_alg_1.cross_validate(
        X=X,
        Y=Y,
        A=A,
        folds=splits,
        preprocessing_function=None,
        metric_function=wmse_for_each_target,
        weights=weights,
        n_jobs=-1,
        verbose=10,
    )

    # np_pls_alg_1_fast_cv_results is a dict with keys corresponding to the names of
    # unique validation splits. The values are the results of the metric function that
    # is passed to the cross_validate method - in this case, mse_for_each_target.

    unique_splits = np.unique(splits)

    # Shape (M, splits) = (10, number_of_splits).
    # Lowest MSE for each target for each split.
    lowest_val_wmses = np.asarray(
        [
            [
                np_pls_alg_1_cv_wmses[split][f"lowest_mse_target_{i}"]
                for split in unique_splits
            ]
            for i in range(M)
        ]
    )

    # Shape (M, splits) = (10, number_of_splits).
    # Number of components that achieves the lowest MSE for each target for each split.
    best_num_components = np.asarray(
        [
            [
                np_pls_alg_1_cv_wmses[split][f"num_components_lowest_mse_target_{i}"]
                for split in unique_splits
            ]
            for i in range(M)
        ]
    )
