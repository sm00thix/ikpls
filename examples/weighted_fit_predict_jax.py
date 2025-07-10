"""
This script demonstrates the usage of the JAX implementation of IKPLS for fitting and
predicting on data. The script generates random input data and fits the IKPLS model to
the data. It then demonstrates how to make predictions using the fitted model. The
internal model parameters can also be accessed for further analysis.

Note: The script assumes that the 'ikpls' package is installed and accessible.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import numpy as np

from ikpls.jax_ikpls_alg_1 import PLS

if __name__ == "__main__":
    N = 100  # Number of samples.
    K = 50  # Number of features.
    M = 10  # Number of targets.
    A = 20  # Number of latent variables (PLS components).

    X = np.random.uniform(size=(N, K))
    Y = np.random.uniform(size=(N, M))
    weights = np.random.uniform(low=0, high=2, size=(N,))

    jax_ikpls_alg_1 = PLS()
    jax_ikpls_alg_1.fit(X, Y, A, weights)

    # Has shape (A, N, M) = (20, 100, 10). Contains a prediction for all
    # possible numbers of components up to and including A.
    y_pred = jax_ikpls_alg_1.predict(X)

    # Has shape (N, M) = (100, 10).
    y_pred_20_components = jax_ikpls_alg_1.predict(X, n_components=20)

    # True. Exact equality might not hold due to numerical differences.
    np.allclose(y_pred_20_components, y_pred[19], atol=0, rtol=1e14)

    # The internal model parameters can be accessed as follows:

    # Regression coefficients tensor of shape (A, K, M) = (20, 50, 10).
    jax_ikpls_alg_1.B

    # X weights matrix of shape (K, A) = (50, 20).
    jax_ikpls_alg_1.W

    # X loadings matrix of shape (K, A) = (50, 20).
    jax_ikpls_alg_1.P

    # Y loadings matrix of shape (M, A) = (10, 20).
    jax_ikpls_alg_1.Q

    # X rotations matrix of shape (K, A) = (50, 20).
    jax_ikpls_alg_1.R

    # X scores matrix of shape (N, A) = (100, 20).
    # This is only computed for IKPLS Algorithm #1.
    jax_ikpls_alg_1.T
