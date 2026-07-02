"""
This script demonstrates the usage of the NumPy implementation of IKPLS for fitting and
predicting on data. The script generates random input data and fits the IKPLS model to
the data. It then demonstrates how to make predictions using the fitted model. The
internal model parameters can also be accessed for further analysis.

Note: The code assumes the availability of the `ikpls` package and its dependencies.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import numpy as np

from ikpls.numpy import PLS

if __name__ == "__main__":
    N = 100  # Number of samples.
    K = 50  # Number of features.
    M = 10  # Number of targets.
    A = 20  # Number of latent variables (PLS components).

    # Using float64 is important for numerical stability.
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
    np_ikpls_alg_1 = PLS(
        center_X=center_X,
        center_Y=center_Y,
        scale_X=scale_X,
        scale_Y=scale_Y,
        ddof=ddof,
    )
    np_ikpls_alg_1.fit(X, Y, A)

    # Has shape (A, N, M) = (20, 100, 10). Contains a prediction for all possible
    # numbers of components up to and including A.
    y_pred = np_ikpls_alg_1.predict(X)

    # Has shape (N, M) = (100, 10).
    y_pred_20_components = np_ikpls_alg_1.predict(X, n_components=20)
    print("All-components predictions y_pred, shape:", y_pred.shape)
    print("Predictions with 20 components, shape:", y_pred_20_components.shape)
    print(
        "predict(X, n_components=20) equals y_pred[19]:",
        (y_pred_20_components == y_pred[19]).all(),
    )

    # The internal model parameters can be accessed as follows:

    # Regression coefficients tensor of shape (A, K, M) = (20, 50, 10).
    print("Regression coefficients B, shape:", np_ikpls_alg_1.B.shape)

    # X weights matrix of shape (K, A) = (50, 20).
    print("X weights W, shape:", np_ikpls_alg_1.W.shape)

    # X loadings matrix of shape (K, A) = (50, 20).
    print("X loadings P, shape:", np_ikpls_alg_1.P.shape)

    # Y loadings matrix of shape (M, A) = (10, 20).
    print("Y loadings Q, shape:", np_ikpls_alg_1.Q.shape)

    # X rotations matrix of shape (K, A) = (50, 20).
    print("X rotations R, shape:", np_ikpls_alg_1.R.shape)

    # X scores matrix of shape (N, A) = (100, 20).
    # This is only computed for IKPLS Algorithm #1.
    print("X scores T, shape:", np_ikpls_alg_1.T.shape)
