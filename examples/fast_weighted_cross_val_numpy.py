"""
This file contains an example implementation of fast cross-validation using IKPLS.
It demonstrates how to perform weighted cross-validation using the fast
cross-validation algorithm with column-wise weighted centering and scaling. It also
demonstrates weighted metric computation and evaluation.

The code includes the following functions:
- `wmse_for_each_target`: A function to compute the mean squared error for each target
    and the number of components that achieves the lowest MSE for each target.

To run the cross-validation, execute the file.

Note: The code assumes the availability of the `ikpls` package and its dependencies.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import numpy as np

from ikpls.fast_cross_validation.numpy_ikpls import PLS


def wmse_for_each_target(Y_true, Y_pred, weights):
    """
    We can return anything we want. Here, we compute the mean squared error for each
    target and the number of components that achieves the lowest MSE for each target.
    """
    # Y_true has shape (N_val, M)
    # Y_pred has shape (A, N_val, M)
    # weights has shape (N_val,)
    e = Y_true - Y_pred  # Shape (A, N_val, M)
    se = e**2  # Shape (A, N_val, M)

    # Compute the mean over samples. Shape (A, M).
    wmse = np.average(se, axis=-2, weights=weights)

    # The number of components that minimizes the WMSE for each target. Shape (M,).
    row_idxs = np.argmin(wmse, axis=0)

    # The lowest MSE for each target. Shape (M,).
    lowest_wmses = wmse[row_idxs, np.arange(wmse.shape[1])]

    # Indices are 0-indexed but number of components is 1-indexed.
    num_components = row_idxs + 1

    # List of names for the lowest MSE values.
    wmse_names = [f"lowest_wmse_target_{i}" for i in range(lowest_wmses.shape[0])]

    # List of names for the number of components that
    # achieves the lowest MSE for each target.
    num_components_names = [
        f"num_components_lowest_wmse_target_{i}" for i in range(lowest_wmses.shape[0])
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
    weights = np.random.uniform(size=N)  # Random weights for each sample.

    # For this example, we will use IKPLS Algorithm #1.
    # The interface for IKPLS Algorithm #2 is identical.
    # Centering and scaling are computed over the training splits
    # only to avoid data leakage from the validation splits.
    # ddof is the delta degrees of freedom for the standard deviation.
    # ddof=0 is the biased estimator, ddof=1 is the unbiased estimator.
    algorithm = 1
    center_X = center_Y = scale_X = scale_Y = True
    ddof = 0
    np_pls_alg_1_fast_cv = PLS(
        algorithm=algorithm,
        center_X=center_X,
        center_Y=center_Y,
        scale_X=scale_X,
        scale_Y=scale_Y,
        ddof=ddof,
    )
    np_pls_alg_1_fast_cv_results = np_pls_alg_1_fast_cv.cross_validate(
        X=X,
        Y=Y,
        A=A,
        folds=splits,
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
    # Lowest WMSE for each target for each split.
    lowest_val_wmses = np.asarray(
        [
            [
                np_pls_alg_1_fast_cv_results[split][f"lowest_wmse_target_{i}"]
                for split in unique_splits
            ]
            for i in range(M)
        ]
    )

    # Shape (M, splits) = (10, number_of_splits).
    # Number of components that achieves the lowest WMSE for each target for each split.
    best_num_components = np.asarray(
        [
            [
                np_pls_alg_1_fast_cv_results[split][
                    f"num_components_lowest_wmse_target_{i}"
                ]
                for split in unique_splits
            ]
            for i in range(M)
        ]
    )
