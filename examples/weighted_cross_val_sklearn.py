"""
This file contains an example of cross-validating the NumPy IKPLS implementation
with scikit-learn's ``cross_validate``, using the scikit-learn-conformant wrapper
``ikpls.sklearn.PLS``, with sample-weighted PLS.

``cross_validate`` performs the per-fold fits, and the per-sample weights are
forwarded to each fold's ``fit`` simply by passing them as
``params={"sample_weight": ...}``: scikit-learn slices them to each training split
automatically. With scikit-learn's default configuration (metadata routing
disabled) this is all that is needed -- no ``set_config`` / ``set_fit_request``.

The validation metric is *also* weighted. scikit-learn does not route metadata
(such as ``sample_weight``) to a custom scoring callable -- it only forwards
``params`` to ``fit``, and routes metadata to ``make_scorer``-style scorers or the
estimator's own ``score`` -- so a custom ``scoring`` callback would silently be
called without the weights. Instead, we ask ``cross_validate`` for the fitted
per-fold estimators and the validation indices (``return_estimator=True`` and
``return_indices=True``) and score each fold ourselves, using its validation-set
weights. This also keeps the all-components feature of IKPLS -- predicting with
every number of components ``1..A`` in a single call via ``predict_all_components``
-- which is used to pick, per target and per split, the number of components that
minimizes the weighted validation MSE.

The code includes the following functions:
- ``cv_splitter``: generate train/validation index pairs for the splits.
- ``weighted_mse_for_each_target``: compute, per target, the lowest weighted
    validation MSE over all component counts and the number of components achieving
    it, given the validation-fold sample weights.

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


def weighted_mse_for_each_target(
    estimator, X_val: np.ndarray, Y_val: np.ndarray, sample_weight: np.ndarray
) -> dict:
    """
    Compute, per target, the lowest *weighted* validation MSE over all component
    counts and the number of components that achieves it.

    Unlike a scikit-learn scoring callable -- to which scikit-learn does not route
    metadata such as ``sample_weight`` -- this receives the validation-fold weights
    explicitly and uses them, so the validation metric is weighted consistently with
    the (weighted) fit.
    """
    # predict_all_components returns predictions for every number of components
    # 1..A in a single call. Shape (A, N_val, M).
    Y_pred = estimator.predict_all_components(X_val)
    se = (Y_val - Y_pred) ** 2  # Shape (A, N_val, M).
    mse = np.average(se, axis=-2, weights=sample_weight)  # Weighted over samples. (A, M).

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
    sample_weight = np.random.uniform(size=(N,))  # Per-sample weights.

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

    # Passing params={"sample_weight": ...} is all that is needed to weight the fit:
    # scikit-learn slices sample_weight to each training split and forwards it to
    # PLS.fit (no metadata-routing setup required). return_estimator and
    # return_indices give us the fitted per-fold models and the validation indices,
    # so we can compute a *weighted* validation metric ourselves below -- scikit-learn
    # does not route sample_weight to a custom scoring callable, and cross_validate's
    # own default score (unweighted R^2) is left unused here.
    cv_results = cross_validate(
        pls,
        X,
        Y,
        cv=cv_splitter(folds),
        params={"sample_weight": sample_weight},
        return_estimator=True,
        return_indices=True,
        n_jobs=-1,
    )

    # Weighted per-component scoring, one fold at a time, using each fold's
    # validation-set sample weights.
    fold_scores = [
        weighted_mse_for_each_target(
            estimator, X[val_idxs], Y[val_idxs], sample_weight[val_idxs]
        )
        for estimator, val_idxs in zip(
            cv_results["estimator"], cv_results["indices"]["test"]
        )
    ]

    # Shape (M, folds). Lowest weighted validation MSE for each target per split.
    lowest_val_mses = np.array(
        [[fs[f"lowest_mse_target_{i}"] for fs in fold_scores] for i in range(M)]
    )

    # Shape (M, folds). Number of components achieving the lowest MSE per target.
    best_num_components = np.array(
        [
            [fs[f"num_components_lowest_mse_target_{i}"] for fs in fold_scores]
            for i in range(M)
        ]
    )

    print("Lowest weighted validation MSE per target per split:\n", lowest_val_mses)
    print("Best number of components per target per split:\n", best_num_components)
