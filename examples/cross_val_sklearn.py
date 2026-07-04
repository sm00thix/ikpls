"""
This file contains an example of cross-validating the NumPy IKPLS implementation
with scikit-learn's ``cross_validate``, using the scikit-learn-conformant wrapper
``ikpls.sklearn.PLS``. It demonstrates column-wise centering and scaling
(computed over the training splits only, to avoid data leakage), and per-component
metric computation via the wrapper's ``predict_all_components`` method.

The wrapper exposes a standard scikit-learn estimator interface (``n_components`` is
a constructor parameter and ``fit(X, y)`` / ``predict(X)`` follow the sklearn
contract), so it drops straight into ``cross_validate``. The all-components feature
of IKPLS -- predicting with every number of components ``1..A`` in a single call --
remains available through ``predict_all_components`` and is used here to pick, per
target and per split, the number of components that minimizes the validation MSE.

The code includes the following functions:
- ``cv_splitter``: generate train/validation index pairs for the splits.
- ``mse_for_each_target``: compute, per target, the lowest validation MSE over all
    component counts and the number of components that achieves it.

To run the cross-validation, execute the file.

Note: The code assumes the availability of the ``ikpls`` package and its dependencies.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import numpy as np
from sklearn.model_selection import cross_validate

from ikpls.sklearn import PLS


def cv_splitter(folds: np.ndarray):
    """
    Generate indices to split data into training and validation sets.
    """
    for split in np.unique(folds):
        train_idxs = np.nonzero(folds != split)[0]
        val_idxs = np.nonzero(folds == split)[0]
        yield train_idxs, val_idxs


def mse_for_each_target(estimator, X: np.ndarray, Y_true: np.ndarray) -> dict:
    """
    Compute the mean squared error for each target and the number of components
    that achieves the lowest MSE for each target.
    """
    # predict_all_components returns predictions for every number of components
    # 1..A in a single call. Shape (A, N_val, M).
    Y_pred = estimator.predict_all_components(X)
    e = Y_true - Y_pred  # Shape (A, N_val, M).
    mse = np.mean(e**2, axis=-2)  # Mean over samples. Shape (A, M).

    # Lowest MSE per target and the number of components achieving it.
    row_idxs = np.argmin(mse, axis=0)  # Shape (M,).
    lowest_mses = mse[row_idxs, np.arange(mse.shape[1])]  # Shape (M,).
    num_components = row_idxs + 1  # Components are 1-indexed.

    mse_names = [f"lowest_mse_target_{i}" for i in range(lowest_mses.shape[0])]
    num_components_names = [
        f"num_components_lowest_mse_target_{i}" for i in range(lowest_mses.shape[0])
    ]
    all_names = mse_names + num_components_names
    all_values = np.concatenate((lowest_mses, num_components))
    return dict(zip(all_names, all_values))


if __name__ == "__main__":
    N = 100  # Number of samples.
    K = 50  # Number of features.
    M = 10  # Number of targets.
    A = 20  # Number of latent variables (PLS components).
    folds = np.random.randint(0, 5, size=N)  # Assign each sample to one of 5 folds.

    X = np.random.uniform(size=(N, K))
    Y = np.random.uniform(size=(N, M))

    # For this example, we use IKPLS Algorithm #1. Algorithm #2 is identical in
    # interface. Centering and scaling are computed over the training splits only
    # (handled by PLS.fit). ddof is the delta degrees of freedom for the
    # standard deviation: ddof=0 is the biased estimator, ddof=1 the unbiased one.
    pls = PLS(
        n_components=A,
        algorithm=1,
        center_X=True,
        center_Y=True,
        scale_X=True,
        scale_Y=True,
        ddof=0,
    )

    # No params={"A": A} needed -- n_components is a constructor parameter.
    results = cross_validate(
        pls,
        X,
        Y,
        cv=cv_splitter(folds),
        scoring=mse_for_each_target,
        return_estimator=False,
        n_jobs=-1,
    )

    # Shape (M, folds). Lowest MSE for each target for each split.
    lowest_val_mses = np.array(
        [results[f"test_lowest_mse_target_{i}"] for i in range(M)]
    )

    # Shape (M, folds). Number of components achieving the lowest MSE per target.
    best_num_components = np.array(
        [results[f"test_num_components_lowest_mse_target_{i}"] for i in range(M)]
    )

    print("Lowest validation MSE per target per split:\n", lowest_val_mses)
    print("Best number of components per target per split:\n", best_num_components)
