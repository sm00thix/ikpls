"""
This file demonstrates how to propagate gradients through the IKPLS model using JAX.
It shows how to compute the gradient of the mean squared error between the true and
predicted values with respect to the weights of a convolution filter that is applied
to the X values before a subsequent PLS fit.

The code includes the following functions:
- `apply_1d_convolution`: A function to apply a 1D convolution filter to a matrix.
- `mean_squared_error`: A function to compute the mean squared error between the true
    and predicted values.
- `convolve_fit_mse`: A function to differentiate the mean squared error with respect
    to the weights of the convolution filter.

Note: The code assumes the availability of the `ikpls[jax]` package and its
dependencies.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from typing import Callable, Union

import jax
import numpy as np
from jax import numpy as jnp

from ikpls.jax_ikpls_alg_1 import PLS as JAX_Alg_1

# Allow JAX to use 64-bit floating point precision.
jax.config.update("jax_enable_x64", True)


@jax.jit
def apply_1d_convolution(X: jnp.ndarray, conv_filter: jnp.ndarray) -> jnp.ndarray:
    """
    Perform a 1D convolution on the rows of the matrix X using the convolution filter.
    """
    # X is a matrix of shape (N, K) = (100, 100)
    # conv_filter is a vector of shape (filter_size,) = (7,)

    # Shape (N, K - filter_size + 1) = (100, 94)
    convolved_rows = jax.vmap(lambda row: jnp.convolve(row, conv_filter, "valid"))(X)
    return convolved_rows


@jax.jit
def weighted_mean_squared_error(
    Y_true: jnp.ndarray, Y_pred: jnp.ndarray, weights: jnp.ndarray
) -> float:
    """
    Compute the weighted mean squared error between the true and predicted values.
    """
    # Y_true is a matrix of shape (N, M) = (100, 10)
    # Y_pred is a matrix of shape (N, M) = (100, 10) or (A, N, M) = (20, 100, 10)
    # Y_pred is a
    e = Y_true - Y_pred  # Shape (N, M) or (A, N, M)
    se = e**2  # Shape (N, M) or (A, N, M)
    mse = jnp.mean(se, axis=(-2, -1))  # Shape () or (A,)
    return mse


# Function to differentiate.
def convolve_fit_mse(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    weights: jnp.ndarray,
    pls_alg: JAX_Alg_1,
    A: int,
    n_components: Union[int, None] = None,
) -> Callable[[jnp.ndarray], float]:
    """
    Return a function that computes the mean squared error between the true and
    predicted values using the given convolution filter. The function is differentiated
    with respect to the weights of the convolution filter.
    """

    @jax.jit
    def helper(conv_filter):
        """
        Compute the mean squared error between the true and predicted values using the
        given convolution filter.
        """
        filtered_X = apply_1d_convolution(X, conv_filter)

        # We must use stateless_fit() because we are using JAX's autodiff.
        matrices = pls_alg.stateless_fit(filtered_X, Y, A, weights)
        B = matrices[0]  # Extract the regression matrix

        # Predict the values.
        Y_pred = pls_alg.stateless_predict(filtered_X, B, n_components)
        mse_loss = weighted_mean_squared_error(Y, Y_pred, weights)
        return mse_loss

    return helper


if __name__ == "__main__":
    N = 100
    K = 100
    M = 10
    A = 20

    # Generate random data.
    jnp_X = jnp.array(np.random.uniform(size=(N, K)), dtype=jnp.float64)
    jnp_Y = jnp.array(np.random.uniform(size=(N, M)), dtype=jnp.float64)
    jnp_w = jnp.array(np.random.uniform(size=(N,)), dtype=jnp.float64)

    filter_size = 7  # Filter size for convolution
    conv_filter = jnp.array(np.random.rand(filter_size))  # Random filter

    # We will use IKPLS Algorithm #1 for this example.
    # The interface for IKPLS Algorithm #2 is identical.
    diff_pls_alg_1 = JAX_Alg_1(differentiable=True, verbose=True)

    # Compute values and gradients for the conv_filter using mean squared error as the
    # loss function and IKPLS Algorithm #1 as the PLS algorithm with exactly 20
    # components. The gradient is computed using backwards mode differentiation.

    # Compute the gradient of the mean_squared_error of Y_true and Y_pred (with 20
    # components) with respect to the weights of the convolution filter.
    grad_fun = jax.grad(
        convolve_fit_mse(jnp_X, jnp_Y, jnp_w, diff_pls_alg_1, A=A, n_components=A),
        argnums=0,
    )
    grad_alg_1 = grad_fun(conv_filter)

    # Compute gradients (Jacobian matrix) for the conv_filter using mean squared error
    # as the loss function and IKPLS Algorithm #1 as the PLS algorithm with all number
    # of components from 1 to 20. The gradients are computed using forwards mode
    # differentiation. We could also use backwards mode differentiation (jacrev). But
    # forward mode differentiation is faster when the Jacobian is tall.

    # Compute the gradient of the mean_squared_error of Y_true and Y_pred (from 1 to 20
    # components) with respect to the weights of the convolution filter.
    jac_fun = jax.jacfwd(
        convolve_fit_mse(jnp_X, jnp_Y, jnp_w, diff_pls_alg_1, A=A, n_components=None),
        argnums=0,
    )
    jac_alg_1 = jac_fun(conv_filter)
    np.allclose(jac_alg_1[A], grad_alg_1, atol=0)  # True
