"""
Contains the PLS Class which implements partial least-squares regression using Improved
Kernel PLS Algorithm #1 by Dayal and MacGregor:
https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

The class is implemented using JAX for end-to-end differentiability. Additionally, JAX
allows CPU, GPU, and TPU execution.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

from ikpls.jax_ikpls_base import PLSBase


class PLS(PLSBase):
    """
    Implements partial least-squares regression using Improved Kernel PLS Algorithm #1
    by Dayal and MacGregor:
    https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

    Parameters
    ----------
    center_X : bool, default=True
        Whether to center `X` before fitting by subtracting its row of
        column-wise means from each row.

    center_Y : bool, default=True
        Whether to center `Y` before fitting by subtracting its row of
        column-wise means from each row.

    scale_X : bool, default=True
        Whether to scale `X` before fitting by dividing each row with the row of `X`'s
        column-wise standard deviations.

    scale_Y : bool, default=True
        Whether to scale `Y` before fitting by dividing each row with the row of `Y`'s
        column-wise standard deviations.

    ddof : int, default=1
        The delta degrees of freedom to use when computing the sample standard
        deviation. A value of 0 corresponds to the biased estimate of the sample
        standard deviation, while a value of 1 corresponds to Bessel's correction for
        the sample standard deviation.

    copy : bool, optional, default=True
        Whether to copy `X` and `Y` in fit before potentially applying centering and
        scaling. If True, then the data is copied before fitting. If False, and `dtype`
        matches the type of `X` and `Y`, then centering and scaling is done inplace,
        modifying both arrays.

    dtype : DTypeLike, optional, default=jnp.float64
        The float datatype to use in computation of the PLS algorithm. Using a lower
        precision than float64 will yield significantly worse results when using an
        increasing number of components due to propagation of numerical errors.

    differentiable: bool, optional, default=False
        Whether to make the implementation end-to-end differentiable. The
        differentiable version is slightly slower. Results among the two versions are
        identical. If this is True, `fit` and `stateless_fit` will not issue a warning
        if the residual goes below machine epsilon, and `max_stable_components` will
        not be set.

    verbose : bool, optional, default=False
        If True, each sub-function will print when it will be JIT compiled. This can be
        useful to track if recompilation is triggered due to passing inputs with
        different shapes.

    Notes
    -----
    Any centering and scaling is undone before returning predictions with `fit` to
    ensure that predictions are on the original scale. If both centering and scaling
    are True, then the data is first centered and then scaled.
    """

    def __init__(
        self,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        ddof: int = 1,
        copy: bool = True,
        dtype: DTypeLike = jnp.float64,
        differentiable: bool = False,
        verbose: bool = False,
    ) -> None:
        self.name = "Improved Kernel PLS Algorithm #1"
        super().__init__(
            center_X=center_X,
            center_Y=center_Y,
            scale_X=scale_X,
            scale_Y=scale_Y,
            ddof=ddof,
            copy=copy,
            dtype=dtype,
            differentiable=differentiable,
            verbose=verbose,
        )
        self.name += " #1"
        self.T = None

    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def _get_initial_matrices(
        self, A: int, K: int, M: int, N: int
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Initialize the matrices and arrays needed for the PLS algorithm. This method is
        part of the PLS fitting process.

        Parameters
        ----------
        A : int
            Number of components in the PLS model.

        K : int
            Number of predictor variables.

        M : int
            Number of response variables.

        N : int
            Number of samples.

        Returns
        -------
        B : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        W : Array of shape (A, K)
            PLS weights matrix for X.

        P : Array of shape (A, K)
            PLS loadings matrix for X.

        Q : Array of shape (A, M)
            PLS Loadings matrix for Y.

        R : Array of shape (A, K)
            PLS weights matrix to compute scores T directly from original X.

        T : Array of shape (A, N)
            PLS scores matrix of X.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_get_initial_matrices for {self.name} will be JIT compiled...")
        B, W, P, Q, R = super()._get_initial_matrices(A, K, M)
        T = jnp.empty(shape=(A, N), dtype=self.dtype)
        return B, W, P, Q, R, T

    @partial(jax.jit, static_argnums=(0,))
    def _get_sqrt_WX(self, X: jax.Array, weights: jax.Array) -> jax.Array:
        """
        Compute the product of the square root of the weights matrix and the predictor
        matrix.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        weights : Array of shape (N,)
            Weights for each observation.

        Returns
        -------
        sqrt_WX : Array of shape (N, K)
            Product of the square root of the weights matrix and the predictor matrix.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_get_sqrt_WX for {self.name} will be JIT compiled...")
        weights = jnp.expand_dims(weights, axis=1)
        sqrt_weights_mat = jnp.sqrt(weights)
        return sqrt_weights_mat * X

    @partial(jax.jit, static_argnums=(0,))
    def _step_1(
        self, X: jax.Array, Y: jax.Array, weights: Optional[jax.Array]
    ) -> jax.Array:
        """
        Perform the first step of Improved Kernel PLS Algorithm #1.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M)
            Response variables.

        weights : Array of shape (N,) or None
            Weights for each observation.

        Returns
        -------
        XTY : Array of shape (K, M)
            Intermediate result used in the PLS algorithm.
        """
        if weights is not None:
            if X.shape[1] < Y.shape[1]:
                X = self._compute_AW(X, weights)
            else:
                Y = self._compute_AW(Y, weights)

        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_step_1 for {self.name} will be JIT compiled...")
        return self._compute_initial_XTY(X.T, Y)

    @partial(jax.jit, static_argnums=(0,))
    def _step_4(self, X: jax.Array, XTY: jax.Array, r: jax.Array):
        """
        Perform the fourth step of Improved Kernel PLS Algorithm #1.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        XTY : Array of shape (K, M)
            Intermediate result used in the PLS algorithm.

        r : Array of shape (K, 1)
            Intermediate result used in the PLS algorithm.

        Returns
        -------
        tTt : float
            Intermediate result used in the PLS algorithm.

        p : Array of shape (K, 1)
            Intermediate result used in the PLS algorithm.

        q : Array of shape (M, 1)
            Intermediate result used in the PLS algorithm.

        t : Array of shape (N, 1)
            Intermediate result used in the PLS algorithm.

        See Also
        --------
        _step_1 : Computes the initial intermediate result in the PLS algorithm.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_step_4 for {self.name} will be JIT compiled...")
        t = X @ r
        tT = t.T
        tTt = tT @ t
        p = (tT @ X).T / tTt
        q = (r.T @ XTY).T / tTt
        return tTt, p, q, t

    @partial(jax.jit, static_argnums=(0, 1, 5, 6))
    def _main_loop_body(
        self,
        A: int,
        i: int,
        X: jax.Array,
        XTY: jax.Array,
        M: int,
        K: int,
        P: jax.Array,
        R: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Execute the main loop body of Improved Kernel PLS Algorithm #1. This function
        performs various steps of the PLS algorithm for each component.

        Parameters
        ----------
        A : int
            Number of components in the PLS model.

        i : int
            Current component index.

        X : Array of shape (N, K)
            Predictor variables.

        XTY : Array of shape (K, M)
            Intermediate result used in the PLS algorithm.

        M : int
            Number of response variables.

        K : int
            Number of predictor variables.

        P : Array of shape (K, A)
            PLS loadings matrix for X.

        R : Array of shape (K, A)
            PLS weights matrix to compute scores T directly from original X.

        Returns
        -------
        XTY : Array of shape (K, M)
            Updated intermediate result used in the PLS algorithm.

        w : Array of shape (K, 1)
            Updated intermediate result used in the PLS algorithm.

        p : Array of shape (K, 1)
            Updated intermediate result used in the PLS algorithm.

        q : Array of shape (M, 1)
            Updated intermediate result used in the PLS algorithm.

        r : Array of shape (K, 1)
            Updated intermediate result used in the PLS algorithm.

        t : Array of shape (N, 1)
            Updated intermediate result used in the PLS algorithm.


        Warns
        -----
        UserWarning.
            If at any point during iteration over the number of components `A`, the
            residual goes below machine epsilon.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_main_loop_body for {self.name} will be JIT compiled...")
        # step 2
        w, norm = self._step_2(XTY, M, K)
        if not self.differentiable:
            self._weight_warning_callback(i, norm)
        # step 3
        if self.differentiable:
            r = self._step_3(A, w, P, R)
        else:
            r = self._step_3(i, w, P, R)
        # step 4
        tTt, p, q, t = self._step_4(X, XTY, r)
        # step 5
        XTY = self._step_5(XTY, p, q, tTt)
        return XTY, w, p, q, r, t

    def fit(
        self, X: ArrayLike, Y: ArrayLike, A: int, weights: Optional[ArrayLike] = None
    ) -> None:
        """
        Fits Improved Kernel PLS Algorithm #1 on `X` and `Y` using `A` components.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M) or (N,)
            Response variables.

        A : int
            Number of components in the PLS model.

        weights : Array of shape (N,) or None, optional, default=None
            Weights for each observation. If None, then all observations are weighted
            equally.

        Attributes
        ----------
        A: int
            Number of components in the PLS model.

        max_stable_components : int
            Maximum number of components that can be used without the residual going
            below machine epsilon. This is not set if `differentiable` is True.

        B : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        W : Array of shape (K, A)
            PLS weights matrix for X.

        P : Array of shape (K, A)
            PLS loadings matrix for X.

        Q : Array of shape (M, A)
            PLS Loadings matrix for Y.

        R : Array of shape (K, A)
            PLS weights matrix to compute scores T directly from original X.

        T : Array of shape (N, A)
            PLS scores matrix of X.

        X_mean : Array of shape (1, K) or None
            Mean of X. If centering is not performed, this is None. If weights are
            used, then this is the weighted mean.

        Y_mean : Array of shape (1, M) or None
            Mean of Y. If centering is not performed, this is None. If weights are
            used, then this is the weighted mean.

        X_std : Array of shape (1, K) or None
            Sample standard deviation of X. If scaling is not performed, this is None.
            If weights are used, then this is the weighted standard deviation.

        Y_std : Array of shape (1, M) or None
            Sample standard deviation of Y. If scaling is not performed, this is None.
            If weights are used, then this is the weighted standard deviation.

        Returns
        -------
        None.

        Raises
        ------
        ValueError
            If `weights` are provided and not all weights are non-negative.

        Warns
        -----
        UserWarning.
            If at any point during iteration over the number of components `A`, the
            residual goes below machine epsilon.

        See Also
        --------
        stateless_fit : Performs the same operation but returns the output matrices
        instead of storing them in the class instance. stateless_fit does not raise an
        error if `weights` are provided and not all weights are non-negative.
        """
        if weights is not None:
            if jnp.any(weights < 0):
                raise ValueError("Weights must be non-negative.")
        self.A = A
        if not self.differentiable:
            self.max_stable_components = A
        self.B, W, P, Q, R, T, self.X_mean, self.Y_mean, self.X_std, self.Y_std = (
            self.stateless_fit(
                X,
                Y,
                A,
                weights,
            )
        )
        self.W = W.T
        self.P = P.T
        self.Q = Q.T
        self.R = R.T
        self.T = T.T

    @partial(jax.jit, static_argnums=(0, 3))
    def stateless_fit(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        A: int,
        weights: Optional[ArrayLike] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Fits Improved Kernel PLS Algorithm #1 on `X` and `Y` using `A` components.
        Returns the internal matrices instead of storing them in the class instance.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables. Its dtype will be converted to float64 for reliable
            results.

        Y : Array of shape (N, M) or (N,)
            Response variables. Its dtype will be converted to float64 for reliable
            results.

        A : int
            Number of components in the PLS model.

        weights : Array of shape (N,) or None, optional, default=None
            Weights for each observation. If None, then all observations are weighted
            equally.

        Returns
        -------
        B : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        W : Array of shape (A, K)
            PLS weights matrix for X.

        P : Array of shape (A, K)
            PLS loadings matrix for X.

        Q : Array of shape (A, M)
            PLS Loadings matrix for Y.

        R : Array of shape (A, K)
            PLS weights matrix to compute scores T directly from original X.

        T : Array of shape (A, N)
            PLS scores matrix of X.

        X_mean : Array of shape (1, K) or None
            Mean of X. If centering is not performed, this is None. If weights are
            used, then this is the weighted mean.

        Y_mean : Array of shape (1, M) or None
            Mean of Y. If centering is not performed, this is None. If weights are
            used, then this is the weighted mean.

        X_std : Array of shape (1, K) or None
            Sample standard deviation of X. If scaling is not performed, this is None.
            If weights are used, then this is the weighted standard deviation.

        Y_std : Array of shape (1, M) or None
            Sample standard deviation of Y. If scaling is not performed, this is None.
            If weights are used, then this is the weighted standard deviation.

        Raises
        ------
        ValueError
            If `weights` are provided and not all weights are non-negative.

        Warns
        -----
        UserWarning.
            If at any point during iteration over the number of components `A`, the
            residual goes below machine epsilon.

        See Also
        --------
        fit : Performs the same operation but stores the output matrices in the class
        instance instead of returning them.

        Notes
        -----
        For optimization purposes, the internal representation of all matrices
        (except B) is transposed from the usual representation.
        """

        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"stateless_fit for {self.name} will be JIT compiled...")

        X, Y, weights = self._initialize_input_matrices(X, Y, weights)
        X, Y, X_mean, Y_mean, X_std, Y_std = self._center_scale_input_matrices(
            X, Y, weights
        )

        # Get shapes
        N, K = X.shape
        M = Y.shape[1]

        # Initialize matrices
        B, W, P, Q, R, T = self._get_initial_matrices(A, K, M, N)

        # step 1
        XTY = self._step_1(X, Y, weights)

        if weights is not None:
            X = self._get_sqrt_WX(X, weights)

        # steps 2-6
        for i in range(A):
            XTY, w, p, q, r, t = self._main_loop_body(A, i, X, XTY, M, K, P, R)
            W = W.at[i].set(w.squeeze())
            P = P.at[i].set(p.squeeze())
            Q = Q.at[i].set(q.squeeze())
            R = R.at[i].set(r.squeeze())
            T = T.at[i].set(t.squeeze())
            b = self._compute_regression_coefficients(B[i - 1], r, q)
            B = B.at[i].set(b)

        return B, W, P, Q, R, T, X_mean, Y_mean, X_std, Y_std
