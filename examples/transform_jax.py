"""
This script demonstrates the usage of the JAX implementation of IKPLS for fitting on
data and transforming data to score space and back to original space. The script
generates random input data and fits the IKPLS model to the data. It then demonstrates
how to make transformations to score space using the fitted model. The internal model
parameters can also be accessed for further analysis.

Note: The code assumes the availability of the `ikpls[jax]` package and its
dependencies.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import jax
import numpy as np

from ikpls.jax_ikpls_alg_1 import PLS

# Allow JAX to use 64-bit floating point precision.
jax.config.update("jax_enable_x64", True)

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
    jax_ikpls_alg_1 = PLS(
        center_X=center_X,
        center_Y=center_Y,
        scale_X=scale_X,
        scale_Y=scale_Y,
        ddof=ddof,
    )

    # These two lines are equivalent to calling jax_ikpls_alg_1.fit_transform(X=X, Y=Y, A=A) or jax_ikpls_alg_1.fit(X=X, Y=Y, A=A).transform(X=X, Y=Y)
    jax_ikpls_alg_1.fit(X=X, Y=Y, A=A)
    X_scores_using_all_components, Y_scores_using_all_components = (
        jax_ikpls_alg_1.transform(X=X, Y=Y, n_components=None)
    )  # Shapes are (N, A) = (100, 20).

    also_X_scores_using_all_components = jax_ikpls_alg_1.transform(
        X=X
    )  # Also works for X only.
    also_Y_scores_using_all_components = jax_ikpls_alg_1.transform(
        Y=Y
    )  # Also works for Y only.

    X_scores_using_five_components, Y_scores_using_five_components = (
        jax_ikpls_alg_1.transform(X=X, Y=Y, n_components=5)
    )  # Shaoes are (N, 5).
    X_scores_using_five_components_only = jax_ikpls_alg_1.transform(
        X=X, n_components=5
    )  # Also works for X only.
    Y_scores_using_five_components_only = jax_ikpls_alg_1.transform(
        Y=Y, n_components=5
    )  # Also works for Y only.

    # Reconstruct X and Y to shapes (N, K) and (N, M) using all available components
    X_reconstructed_using_all_components, Y_reconstructed_using_all_components = (
        jax_ikpls_alg_1.inverse_transform(
            T=X_scores_using_all_components, U=Y_scores_using_all_components
        )
    )
    X_reconstructed_using_all_components = jax_ikpls_alg_1.inverse_transform(
        T=X_scores_using_all_components
    )  # Also works for X only
    Y_reconstructed_using_all_components = jax_ikpls_alg_1.inverse_transform(
        U=Y_scores_using_all_components
    )  # Also works for Y only

    # Reconstruct X and Y to shapes (N, K) and (N, M) using only five components
    X_reconstructed_using_five_components, Y_reconstructed_using_five_components = (
        jax_ikpls_alg_1.inverse_transform(
            T=X_scores_using_five_components, U=Y_scores_using_five_components
        )
    )

    # Reconstruct X and Y to shapes (N, K) and (N, M) using all components for X and only 5 for Y
    X_reconstructed_using_all_components, Y_reconstructed_using_five_components = (
        jax_ikpls_alg_1.inverse_transform(
            T=X_scores_using_all_components, U=Y_scores_using_five_components
        )
    )  # We can use different numbers of components for X scores and Y scores
