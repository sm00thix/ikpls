"""
This file contains an abstract class for partial least-squares regression using
Improved Kernel PLS by Dayal and MacGregor.

Implementations of concrete classes exist for both Improved Kernel PLS Algorithm #1
and Improved Kernel PLS Algorithm #2.

For more details, refer to the paper:
"Improved Kernel Partial Least Squares Regression" by Dayal and MacGregor.

This file also contains the _R_Y_Mapping class which is a concrete Mapping to store the
PLS weights matrix to compute scores U directly from original Y for different numbers
of components.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import abc
import warnings
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Optional, Self, Tuple, Union

import jax
import jax.experimental
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax.typing import ArrayLike, DTypeLike
from sklearn.exceptions import NotFittedError
from tqdm import tqdm


class _R_Y_Mapping(Mapping):
    """A read-only Mapping that computes values lazily on first access."""

    def __init__(self, QT: jax.Array) -> None:
        self.QT = QT
        self.eps = jnp.finfo(QT.dtype).eps
        self._valid_keys = set(range(1, QT.shape[0] + 1))
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._valid_keys:
            raise KeyError(
                f"Invalid number of components: {key}. Valid numbers of components are 1 to {self.QT.shape[0]}."
            )
        if key not in self._cache:
            self._cache[key] = jla.pinv(self.QT[:key], rcond=self.eps)
        return self._cache[key]

    def __iter__(self):
        return iter(self._valid_keys)

    def __len__(self):
        return len(self._valid_keys)

    def __contains__(self, key):
        return key in self._valid_keys

    def __repr__(self):
        return f"_R_Y_Mapping(Cached R_Y for {len(self._cache)}/{len(self._valid_keys)} n_components)"


class PLSBase(abc.ABC):
    """
    Implements an abstract class for partial least-squares regression using Improved
    Kernel PLS by Dayal and MacGregor:
    https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

    Implementations of concrete classes exist for both Improved Kernel PLS Algorithm #1
    and Improved Kernel PLS Algorithm #2.

    Parameters
    ----------
    center_X : bool, default=True
        Whether to center the predictor variables (X) before fitting. If True, then the
        mean of the training data is subtracted from the predictor variables.

    center_Y : bool, default=True
        Whether to center the response variables (Y) before fitting. If True, then the
        mean of the training data is subtracted from the response variables.

    scale_X : bool, default=True
        Whether to scale the predictor variables (X) before fitting by dividing each
        row with their row of column-wise standard deviatons.

    scale_Y : bool, default=True
        Whether to scale the predictor variables (Y) before fitting by dividing each
        row with their row of column-wise standard deviatons.

    ddof : int, default=1
        The delta degrees of freedom to use when computing the sample standard
        deviation. A value of 0 corresponds to the biased estimate of the sample
        standard deviation, while a value of 1 corresponds to Bessel's correction for
        the sample standard deviation.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before potentially applying centering and
        scaling. If True, then the data is copied before fitting. If False, and `dtype`
        matches the type of `X` and `Y`, then centering and scaling is done inplace,
        modifying both arrays.

    dtype : DTypeLike, default=jnp.float64
        The float datatype to use in computation of the PLS algorithm. This should be
        jax.numpy.float32 or jax.numpy.float64. Using a lower precision than float64
        will yield significantly worse results when using an increasing number of
        components due to propagation of numerical errors.

    differentiable: bool, default=False
        Whether to make the implementation end-to-end differentiable. The
        differentiable version is slightly slower. Results among the two versions are
        identical. If this is True, `fit` and `stateless_fit` will not issue a warning
        if the residual goes below machine epsilon, and `max_stable_components` will
        not be set.

    verbose : bool, default=False
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
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.ddof = ddof
        self.copy = copy
        self.dtype = dtype
        self.eps = jnp.finfo(self.dtype).eps
        self.differentiable = differentiable
        self.verbose = verbose
        self.name = "Improved Kernel PLS Algorithm"
        self.A = None
        self.B = None
        self.W = None
        self.P = None
        self.Q = None
        self.R = None
        self.R_Y = None
        self.T = None
        self.X_mean = None
        self.Y_mean = None
        self.X_std = None
        self.Y_std = None
        self.max_stable_components = None
        self.fitted_ = False
        if self.dtype == jnp.float64:
            jax.config.update("jax_enable_x64", True)

    def _weight_warning(self, i: DTypeLike):
        """
        Display a warning message if the weight is close to zero.

        Parameters
        ----------
        i : int
            The current component number in the PLS algorithm.

        Warns
        -----
        UserWarning.
            If the weight norm is below machine epsilon, a warning message is
            displayed.

        Notes
        -----
        This method issues a warning if the weight becomes close to zero during the PLS
        algorithm. It provides a hint about potential instability in results with a
        higher number of components.
        """
        warnings.warn(
            message=f"Weight is close to zero. Results with A = {i + 1} "
            "component(s) or higher may be unstable.",
            category=UserWarning,
        )
        if self.max_stable_components in (None, self.A):
            self.max_stable_components = int(i)

    @partial(jax.jit, static_argnums=0)
    def _weight_warning_callback(self, i: int, norm: DTypeLike):
        close_to_zero = jnp.isclose(norm, 0, atol=jnp.finfo(self.dtype).eps, rtol=0)
        return jax.lax.cond(
            close_to_zero,
            lambda num_components: jax.experimental.io_callback(
                self._weight_warning, None, num_components, ordered=True
            ),
            lambda _num_components: None,
            i,
        )

    @partial(jax.jit, static_argnums=0)
    def _compute_regression_coefficients(
        self, b_last: jax.Array, r: jax.Array, q: jax.Array
    ) -> jax.Array:
        """
        Compute the regression coefficients in the PLS algorithm.

        Parameters
        ----------
        b_last : Array of shape (K, M)
            The previous regression coefficient matrix.

        r : Array of shape (K, 1)
            The orthogonal weight vector for the current component.

        q : Array of shape (M, 1)
            The loadings vector for the response variables.

        Returns
        -------
        b : Array of shape (K, M)
            The updated regression coefficient matrix for the current component.

        Notes
        -----
        This method computes the regression coefficients matrix for the current
        component in the PLS algorithm, incorporating the orthogonal weight vector and
        loadings vector.
        """
        b = b_last + r @ q.T
        return b

    @partial(jax.jit, static_argnums=0)
    def _convert_input_to_array(self, arr: ArrayLike) -> jax.Array:
        """
        Converts input array to numpy array of type self.dtype. Inserts a second axis
        if the input array is 1-dimensional.

        Parameters
        ----------
        arr : ArrayLike
            Input array.

        Returns
        -------
        array_converted : Array of shape of input array
            Converted input array.
        """
        arr = jnp.asarray(arr, dtype=self.dtype)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    @partial(jax.jit, static_argnums=0)
    def _initialize_input_matrices(
        self, X: ArrayLike, Y: ArrayLike, weights: Optional[ArrayLike] = None
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        """
        Initialize the input matrices used in the PLS algorithm.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables matrix.

        Y : Array of shape (N, M) or (N,)
            Response variables matrix.

        weights : Array of shape (N,) or None, optional, default=None
            Weights for each observation. If None, then all observations are weighted
            equally.

        Returns
        -------
        X : Array of shape (N, K)
            Predictor variables matrix, converted to the specified dtype.

        Y : Array of shape (N, M)
            Response variables matrix, converted to the specified dtype.

        weights : Array of shape (N,) or None
            Weights for each observation, converted to the specified dtype. If None,
            then all observations are weighted equally.
        """
        X = self._convert_input_to_array(X)
        Y = self._convert_input_to_array(Y)

        if weights is not None:
            weights = jnp.asarray(weights, dtype=self.dtype)
            weights = jnp.ravel(weights)
        return X, Y, weights

    @partial(jax.jit, static_argnums=0)
    def _get_mean(self, A: ArrayLike, weights: Optional[ArrayLike] = None):
        """
        Get the mean of the a matrix.

        Parameters
        ----------
        A : Array of shape (N, K) or (N, M)
            Predictor variables matrix or response variables matrix.

        weights : Array of shape (N,) or None, optional, default=None
            Weights for each observation. If None, then all observations are weighted
            equally.

        Returns
        -------
        A_mean : Array of shape (1, K) or (1, M)
            Mean of the predictor variables or response variables.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"get_means for {self.name} will be JIT compiled...")

        A_mean = jnp.average(A, axis=0, weights=weights, keepdims=True)
        A_mean = jnp.asarray(A_mean, dtype=self.dtype)
        return A_mean

    @partial(jax.jit, static_argnums=0)
    def _get_std(
        self,
        A: ArrayLike,
        mean: ArrayLike,
        weights: Optional[ArrayLike],
        scale_dof: DTypeLike,
        avg_non_zero_weights: Optional[DTypeLike],
    ) -> jax.Array:
        """
        Get the standard deviation of a matrix.

        Parameters
        ----------
        A : Array of shape (N, K) or (N, M)
            Predictor variables matrix or response variables matrix.

        mean : Array of shape (1, K) or (1, M)
            Mean of the predictor variables or response variables. This is the value
            from which the squared deviations are calculated.

        weights : Array of shape (N,) or None
            Weights for each observation. If None, then all observations are weighted
            equally.

        scale_dof : int
            The degrees of freedom to use in the standard deviation calculation

        avg_non_zero_weights : float or None
            The average of the non-zero weights in the weights vector. Only used if
            weights are provided.

        Returns
        -------
        A_std : Array of shape (1, K) or (1, M)
            Sample standard deviation of the predictor variables or response variables.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"get_stds for {self.name} will be JIT compiled...")

        if weights is None:
            A_std = jnp.sqrt(
                jnp.sum((A - mean) ** 2, axis=0, keepdims=True) / scale_dof
            )
        else:
            A_std = jnp.sqrt(
                jnp.sum(
                    jnp.expand_dims(weights, axis=1) * (A - mean) ** 2,
                    axis=0,
                    keepdims=True,
                )
                / (scale_dof * avg_non_zero_weights)
            )
        A_std = jnp.where(A_std <= self.eps, 1, A_std)
        return A_std

    @partial(jax.jit, static_argnums=0)
    def _center_scale_input_matrices(
        self,
        X: jax.Array,
        Y: jax.Array,
        weights: Optional[jax.Array],
    ):
        """
        Preprocess the input matrices based on the centering and scaling parameters.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables matrix.

        Y : Array of shape (N, M)
            Response variables matrix.

        weights : Array of shape (N,) or None
            Weights for each observation. If None, then all observations are weighted
            equally.

        Returns
        -------
        X : Array of shape (N, K)
            Potentially centered and potentially scaled variables matrix.

        Y : Array of shape (N, M)
            Potentially centered and potentially scaled response variables matrix.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_preprocess_input_matrices for {self.name} will be JIT compiled...")

        if (self.center_X or self.scale_X) and self.copy:
            X = X.copy()

        if (self.center_Y or self.scale_Y) and self.copy:
            Y = Y.copy()

        if self.center_X:
            X_mean = self._get_mean(X, weights)
            X = X - X_mean
        else:
            X_mean = None

        if self.center_Y:
            Y_mean = self._get_mean(Y, weights)
            Y = Y - Y_mean
        else:
            Y_mean = None

        if self.scale_X or self.scale_Y:
            if weights is not None:
                num_nonzero_weights = jnp.asarray(
                    jnp.count_nonzero(weights), dtype=self.dtype
                )
                avg_non_zero_weights = jnp.sum(weights) / num_nonzero_weights
                scale_dof = num_nonzero_weights - self.ddof
            else:
                avg_non_zero_weights = None
                scale_dof = X.shape[0] - self.ddof

        if self.scale_X:
            new_X_mean = 0 if self.center_X else self._get_mean(X, weights)
            X_std = self._get_std(
                X, new_X_mean, weights, scale_dof, avg_non_zero_weights
            )
            X = X / X_std
        else:
            X_std = None

        if self.scale_Y:
            new_Y_mean = 0 if self.center_Y else self._get_mean(Y, weights)
            Y_std = self._get_std(
                Y, new_Y_mean, weights, scale_dof, avg_non_zero_weights
            )
            Y = Y / Y_std
        else:
            Y_std = None

        return X, Y, X_mean, Y_mean, X_std, Y_std

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def _get_initial_matrices(
        self, A: int, K: int, M: int
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Initialize the matrices used in the PLS algorithm.

        Parameters
        ----------
        A : int
            Number of components in the PLS model.

        K : int
            Number of predictor variables.

        M : int
            Number of response variables.

        Returns
        -------
        A tuple of initialized matrices:
        B : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        W : Array of shape (A, K)
            PLS weights matrix for X.

        P : Array of shape (A, K)
            PLS loadings matrix for X.

        Q : Array of shape (A, M)
            PLS Loadings matrix for Y.

        R : Array of shape (A, K)
            PLS weights matrix to compute scores T directly from the original X.

        Notes
        -----
        This abstract method is responsible for initializing various matrices used in
        the PLS algorithm, including regression coefficients, weights, loadings, and
        orthogonal weights.
        """
        B = jnp.zeros(shape=(A, K, M), dtype=self.dtype)
        W = jnp.zeros(shape=(A, K), dtype=self.dtype)
        P = jnp.zeros(shape=(A, K), dtype=self.dtype)
        Q = jnp.zeros(shape=(A, M), dtype=self.dtype)
        R = jnp.zeros(shape=(A, K), dtype=self.dtype)
        return B, W, P, Q, R

    @partial(jax.jit, static_argnums=0)
    def _compute_XT(self, X: jax.Array) -> jax.Array:
        """
        Compute the transposed predictor variable matrix.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables matrix.

        Returns
        -------
        XT : Array of shape (K, N)
            Transposed predictor variables matrix.

        Notes
        -----
        This method calculates the transposed predictor variables matrix from the
        original predictor variables matrix.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_compute_XT for {self.name} will be JIT compiled...")
        return X.T

    @partial(jax.jit, static_argnums=0)
    def _compute_AW(self, A: jax.Array, weights: jax.Array) -> jax.Array:
        """
        Get the product of A and the weights matrix.

        Parameters
        ----------
        A : Array of shape (N, K) or (N, M)
            Predictor variables.

        weights : Array of shape (N,)
            Weights for each observation.

        Returns
        -------
        AW : Array of shape (N, K) or (N, M)
            Product of A and the weights matrix.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_get_AW for {self.name} will be JIT compiled...")
        return A * jnp.expand_dims(weights, axis=1)

    @partial(jax.jit, static_argnums=0)
    def _compute_initial_XTY(self, XT: jax.Array, Y: jax.Array) -> jax.Array:
        """
        Compute the initial cross-covariance matrix of the predictor variables and the
        response variables.

        Parameters
        ----------
        XT : Array of shape (K, N)
            Transposed predictor variables matrix.

        Y : Array of shape (N, M)
            Response variables matrix.

        Returns
        -------
        XTY : Array of shape (K, M)
            Initial cross-covariance matrix of the predictor variables and the response
            variables.

        Notes
        -----
        This method calculates the initial cross-covariance matrix of the predictor
        variables and the response variables.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_compute_initial_XTY for {self.name} will be JIT compiled...")
        return XT @ Y

    @partial(jax.jit, static_argnums=0)
    def _compute_XTX(self, XT: jax.Array, X: jax.Array) -> jax.Array:
        """
        Compute the product of the transposed predictor variables matrix and the
        predictor variables matrix.

        Parameters
        ----------
        XT : Array of shape (K, N)
            Transposed predictor variables matrix.

        X : Array of shape (N, K)
            Predictor variables matrix.

        Returns
        -------
        XTX : Array of shape (K, K)
            Product of transposed predictor variables and predictor variables.

        Notes
        -----
        This method calculates the product of the transposed predictor variables matrix
        and the predictor variables matrix.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_compute_XTX for {self.name} will be JIT compiled...")
        return XT @ X

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _step_1(self, X: jax.Array, Y: jax.Array, weights: Optional[jax.Array]):
        """
        Abstract method representing the first step in the PLS algorithm. This step
        should be implemented in concrete PLS classes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method represents the first step of the PLS algorithm and should be
        implemented in concrete PLS classes.
        """

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _step_2(
        self, XTY: jax.Array, M: int, K: int
    ) -> (
        Tuple[jax.Array, DTypeLike] | Tuple[jax.Array, DTypeLike, jax.Array, jax.Array]
    ):
        """
        The second step of the PLS algorithm. Computes the next weight vector and the
        associated norm.

        Parameters
        ----------
        XTY : Array of shape (K, M)
            The cross-covariance matrix of the predictor variables and the response
            variables.

        M : int
            Number of response variables.

        K : int
            Number of predictor variables.

        Returns
        -------
        w : Array of shape (K, 1)
            The next weight vector for the PLS algorithm.

        norm : float
            The l2 norm of the weight vector `w`.

        Notes
        -----
        This method computes the next weight vector `w` for the PLS algorithm and its
        associated norm.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_step_2 for {self.name} will be JIT compiled...")
        if M == 1:
            norm = jla.norm(XTY)
            w = XTY / norm
        elif M < K:
            XTYTXTY = XTY.T @ XTY
            eig_vals, eig_vecs = jla.eigh(XTYTXTY)
            q = eig_vecs[:, -1:]
            q = q.reshape(-1, 1)
            w = XTY @ q
            norm = jla.norm(w)
            w = w / norm
            return w, norm, q, eig_vals[-1]
        else:
            XTYYTX = XTY @ XTY.T
            eig_vals, eig_vecs = jla.eigh(XTYYTX)
            w = eig_vecs[:, -1:]
            norm = eig_vals[-1]
        return w, norm

    def _step_3_base(
        self, i: int, w: jax.Array, P: jax.Array, R: jax.Array
    ) -> jax.Array:
        """
        The third step of the PLS algorithm. Computes the orthogonal weight vectors.

        Parameters
        ----------
        i : int
            The current component number in the PLS algorithm.

        w : Array of shape (K, 1)
            The current weight vector.

        P : Array of shape (A, K)
            The loadings matrix for the predictor variables.

        R : Array of shape (A, K)
            The weights matrix to compute scores `T` directly from the original
            predictor variables.

        Returns
        -------
        r : Array of shape (K, 1)
            The orthogonal weight vector for the current component.

        Notes
        -----
        This method computes the orthogonal weight vector `r` for the current component
        in the PLS algorithm. It is a key step for calculating the loadings and weights
        matrices.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_step_3 for {self.name} will be JIT compiled...")
        r = jnp.copy(w)
        r, P, w, R = jax.lax.fori_loop(0, i, self._step_3_body, (r, P, w, R))
        return r

    def _step_3(self, i: int, w: jax.Array, P: jax.Array, R: jax.Array) -> jax.Array:
        """
        This is an API to the third step of the PLS algorithm. Computes the orthogonal
        weight vectors.

        Parameters
        ----------
        i : int
            The current component number in the PLS algorithm.

        w : Array of shape (K, 1)
            The current weight vector.

        P : Array of shape (A, K)
            The loadings matrix for the predictor variables.

        R : Array of shape (A, K)
            The weights matrix to compute scores `T` directly from the original
            predictor variables.

        Returns
        -------
        r : Array of shape (K, 1)
            The orthogonal weight vector for the current component.

        Notes
        -----
        This method compiles _step_3_base which in turn computes the orthogonal weight
        vector `r` for the current component in the PLS algorithm. It is a key step for
        calculating the loadings and weights matrices.

        See Also
        --------
        _step_3_base : The third step of the PLS algorithm.
        """
        if self.differentiable:
            jax.jit(self._step_3_base, static_argnums=(0, 1))
            return self._step_3_base(i, w, P, R)
        jax.jit(self._step_3_base, static_argnums=0)
        return self._step_3_base(i, w, P, R)

    @partial(jax.jit, static_argnums=0)
    def _step_3_body(
        self, j: int, carry: Tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        The body of the third step of the PLS algorithm. Iteratively computes
        orthogonal weight vectors.

        Parameters
        ----------
        j : int
            The current iteration index.

        carry : Tuple of arrays
            A tuple containing weight vectors and matrices used in the PLS algorithm.

        Returns
        -------
        carry : Tuple of arrays
            Updated weight vectors and matrices used in the PLS algorithm.

        Notes
        -----
        This method is the body of the third step of the PLS algorithm and iteratively
        computes orthogonal weight vectors used in the PLS algorithm.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_step_3_body for {self.name} will be JIT compiled...")
        r, P, w, R = carry
        r = r - P[j].reshape(-1, 1).T @ w * R[j].reshape(-1, 1)
        return r, P, w, R

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _step_4(self):
        """
        Abstract method representing the fourth step in the PLS algorithm. This step
        should be implemented in concrete PLS classes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method represents the fourth step of the PLS algorithm and should be
        implemented in concrete PLS classes.
        """

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def _step_4_compute_q(
        self,
        r: jax.Array,
        XTY: jax.Array,
        tTt: jax.Array,
        M: int,
        K: int,
        # step_2_res: Tuple[jax.Array, DTypeLike]
        # | Tuple[jax.Array, DTypeLike, jax.Array],
        norm: DTypeLike,
        q: Optional[jax.Array] = None,
        largest_eigval: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Selects the appropriate branch function for step 4 based on the number of
        response and predictor variables.

        Parameters
        ----------
        M : int
            Number of response variables.

        K : int
            Number of predictor variables.

        Returns
        -------
        branch_function : Callable[..., jax.Array]
            The selected branch function for step 4.

        Notes
        -----
        This method selects the appropriate branch function for step 4 of the PLS
        algorithm based on the number of response and predictor variables.
        """
        # return self._step_4_compute_q_branch_3(r=r, XTY=XTY, tTt=tTt)
        if M == 1:
            return self._step_4_compute_q_branch_1(norm=norm, tTt=tTt)
        elif M < K:
            return self._step_4_compute_q_branch_2(
                q=q, largest_eigval=largest_eigval, tTt=tTt
            )
        else:
            return self._step_4_compute_q_branch_3(r=r, XTY=XTY, tTt=tTt)

    @partial(jax.jit, static_argnums=0)
    def _step_4_compute_q_branch_1(self, norm: jax.Array, tTt: jax.Array) -> jax.Array:
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_step_4_helper_branch_1 for {self.name} will be JIT compiled...")
        return norm / tTt

    @partial(jax.jit, static_argnums=0)
    def _step_4_compute_q_branch_2(
        self, q: jax.Array, largest_eigval: jax.Array, tTt: jax.Array
    ) -> jax.Array:
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_step_4_helper_branch_2 for {self.name} will be JIT compiled...")
        return q * jnp.sqrt(largest_eigval) / tTt

    @partial(jax.jit, static_argnums=0)
    def _step_4_compute_q_branch_3(
        self, r: jax.Array, XTY: jax.Array, tTt: jax.Array
    ) -> jax.Array:
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_step_4_helper_branch_3 for {self.name} will be JIT compiled...")
        return (r.T @ XTY).T / tTt

    @partial(jax.jit, static_argnums=0)
    def _step_5(
        self, XTY: jax.Array, p: jax.Array, q: jax.Array, tTt: jax.Array
    ) -> jax.Array:
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_step_5 for {self.name} will be JIT compiled...")
        return XTY - (p @ q.T) * tTt

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _main_loop_body(self):
        """
        Abstract method representing the main loop body in the PLS algorithm. This
        method should be implemented in concrete PLS classes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method represents the main loop body of the PLS algorithm and should be
        implemented in concrete PLS classes.
        """

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0, 3))
    def stateless_fit(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        A: int,
        weights: Optional[ArrayLike] = None,
    ) -> Union[
        Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ]:
        """
        Fits Improved Kernel PLS Algorithm #1 on `X` and `Y` using `A` components.
        Returns the internal matrices instead of storing them in the class instance.

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

        R_Y : Mapping[int, Array]
            Mapping from number of components to PLS weights matrix to compute scores U
            directly from original Y. Keys range from 1 to A. Values are arrays of
            shape (M, n_components) where n_components is the key. Values are computed
            lazily and cached upon first access. See Notes for more information.

        T : Array of shape (A, N)
            PLS scores matrix of X. Only Returned for Improved Kernel PLS Algorithm #1.

        X_mean : Array of shape (1, K) or None
            Mean of the predictor variables `center_X` is True, otherwise None.

        Y_mean : Array of shape (1, M) or None
            Mean of the response variables `center_Y` is True, otherwise None.

        X_std : Array of shape (1, K) or None
            Sample standard deviation of the predictor variables `scale_X` is True,
            otherwise None.

        Y_std : Array of shape (1, M) or None
            Sample standard deviation of the response variables `scale_Y` is True,
            otherwise None.

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

    @abc.abstractmethod
    def fit(
        self, X: ArrayLike, Y: ArrayLike, A: int, weights: Optional[ArrayLike] = None
    ) -> Self:
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
        A : int
            Number of components in the PLS model.

        max_stable_components : int
            Maximum number of components that can be used without the residual going
            below machine epsilon.

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

        R_Y : Mapping[int, Array]
            Mapping from number of components to PLS weights matrix to compute scores U
            directly from original Y. Keys range from 1 to A. Values are arrays of
            shape (M, n_components) where n_components is the key. Values are computed
            lazily and cached upon first access. See Notes for more information.

        T : Array of shape (N, A)
            PLS scores matrix of X. Only assigned for Improved Kernel PLS Algorithm #1.
            IMPORTANT: If weights are provided, these are NOT the scores of X but
            instead weighted scores. In this case, scores can be computerd using
            transform.

        Returns
        -------
        self : PLS
            Fitted model.

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
        instead of storing them in the class instance.
        """
        self.fitted_ = True
        return self

    @partial(jax.jit, static_argnums=(0, 3))
    def stateless_predict(
        self,
        X: ArrayLike,
        B: jax.Array,
        n_components: Optional[int] = None,
        X_mean: Optional[jax.Array] = None,
        X_std: Optional[jax.Array] = None,
        Y_mean: Optional[jax.Array] = None,
        Y_std: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Predicts with Improved Kernel PLS Algorithm #1 on `X` with `B` using
        `n_components` components. If `n_components` is None, then predictions are
        returned for all number of components.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        B : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        n_components : int or None, optional
            Number of components in the PLS model. If None, then all number of
            components are used.

        X_mean : Array of shape (1, K) or None, optional, default=None
            Mean of the predictor variables. If None, then no mean is subtracted from
            `X`.

        X_std : Array of shape (1, K) or None, optional, default=None
            Sample standard deviation of the predictor variables. If None, then no
            scaling is applied to `X`.

        Y_mean : Array of shape (1, M) or None, optional, default=None
            Mean of the response variables. If None, then no mean is subtracted from
            `Y`.

        Y_std : Array of shape (1, M) or None, optional, default=None
            Sample standard deviation of the response variables. If None, then no
            scaling is applied to `Y`.

        Returns
        -------
        Y_pred : Array of shape (N, M) or (A, N, M)
            If `n_components` is an int, then an array of shape (N, M) with the
            predictions for that specific number of components is used. If
            `n_components` is None, returns a prediction for each number of components
            up to `A`.

        See Also
        --------
        predict : Performs the same operation but uses the class instances of `B`,
        `X_mean`, `X_std`, `Y_mean`, and `Y_std` instead of the ones passed as
        arguments.
        """
        X = jnp.asarray(X, dtype=self.dtype)
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"stateless_predict for {self.name} will be JIT compiled...")

        if X_mean is not None:
            X = X - X_mean
        if X_std is not None:
            X = X / X_std

        if n_components is None:
            Y_pred = X @ B
        else:
            Y_pred = X @ B[n_components - 1]

        if Y_std is not None:
            Y_pred = Y_pred * Y_std
        if Y_mean is not None:
            Y_pred = Y_pred + Y_mean
        return Y_pred

    @partial(jax.jit, static_argnums=(0, 2))
    def predict(self, X: ArrayLike, n_components: Optional[int] = None) -> jax.Array:
        """
        Predicts with Improved Kernel PLS Algorithm #1 on `X` with `B` using
        `n_components` components. If `n_components` is None, then predictions are
        returned for all number of components.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        n_components : int or None, optional
            Number of components in the PLS model. If None, then all number of
            components are used.

        Returns
        -------
        Y_pred : Array of shape (N, M) or (A, N, M)
            If `n_components` is an int, then an array of shape (N, M) with the
            predictions for that specific number of components is used. If
            `n_components` is None, returns a prediction for each number of components
            up to `A`.

        See Also
        --------
        stateless_predict : Performs the same operation but uses inputs `B`, `X_mean`,
        `X_std`, `Y_mean`, and `Y_std` instead of the ones stored in the class
        instance.
        """
        if not self.fitted_:
            raise NotFittedError(
                "This model is not fitted yet, call 'fit' with approriate arguments before 'predict'."
            )
        return self.stateless_predict(
            X, self.B, n_components, self.X_mean, self.X_std, self.Y_mean, self.Y_std
        )

    def transform(
        self,
        X: Optional[ArrayLike] = None,
        Y: Optional[ArrayLike] = None,
        n_components: Optional[int] = None,
        copy: bool = True,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array], None]:
        """
        Transforms `X` and `Y` to their respective scores using `n_components` components.
        If `n_components` is None, then scores for all components up to `A` are returned.

        Parameters
        ----------
        X : Array of shape (N, K) or None, optional, default=None
            Predictor variables.
        Y : Array of shape (N, M) or None, optional, default=None
            Response variables.
        n_components : int or None, optional, default=None.
            Number of components in the PLS model. If None, then scores for all
            components up to `A` are returned.

        copy : bool, default=True
            Whether to copy `X` and `Y` before potentially applying centering and
            scaling. If True, then the data is copied beforehand. If False, and `dtype`
            matches the type of `X` and `Y`, then centering and scaling is done inplace,
            modifying both arrays.

        Returns
        -------
        T : Array of shape (N, n_components) or (N, A) or None
            X scores of `X`. If `n_components` is an int, then the scores up to
            `n_components` is used. If `n_components` is None, returns scores for all
            components up to `A`.
        U : Array of shape (N, n_components) or (N, A) or None
            Y scores of `Y`. If `n_components` is an int, then the scores up to
            `n_components` is used. If `n_components` is None, returns scores for all
            components up to `A`.

        Raises
        ------
        NotFittedError
            If the model has not been fitted before calling `transform()`.

        See Also
        --------
        fit_transform : Fits the model and returns the scores of `X` and `Y`.
        inverse_transform : Reconstructs `X` and `Y` from their respective scores.

        Notes
        -----
        If multiple calls to `transform` are made with Y provided and a previously seen
        value of `n_components`, then self.R_Y[n_components] is only computed once and
        stored for future use.

        Any centering and scaling of `X`and `Y` is carried out before computation of
        the scores and is undone afterwards.
        """
        if not self.fitted_:
            raise NotFittedError(
                "This model is not fitted yet, call 'fit' with approriate arguments before 'transform'."
            )
        if n_components is None:
            n_components = self.A
        if X is not None:
            X = self._convert_input_to_array(X)
            if (self.center_X or self.scale_X) and copy:
                X = X.copy()
            if self.center_X:
                X -= self.X_mean
            if self.scale_X:
                X /= self.X_std
            T = X @ self.R[:, :n_components]
        if Y is not None:
            Y = self._convert_input_to_array(Y)
            if (self.center_Y or self.scale_Y) and copy:
                Y = Y.copy()
            if self.center_Y:
                Y -= self.Y_mean
            if self.scale_Y:
                Y /= self.Y_std
            U = Y @ self.R_Y[n_components]
        if X is not None and Y is not None:
            return T, U
        if X is not None:
            return T
        if Y is not None:
            return U
        return None

    def fit_transform(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        A: int,
        weights: Optional[ArrayLike] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Fits Improved Kernel PLS Algorithm #1 on `X` and `Y` using `A` components and
        returns T and U which are the scores of `X` and `Y`, respectively.

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

        Returns
        -------
        T : Array of shape (N, A)
            PLS scores matrix of X.

        U : Array of shape (N, A)
            PLS scores matrix of Y.

        See Also
        --------
        transform : Transforms `X` and `Y` to their respective scores.
        inverse_transform : Reconstructs `X` and `Y` from their respective scores.
        """
        self.fit(X=X, Y=Y, A=A, weights=weights)
        if self.T is not None and weights is None:
            return jnp.copy(self.T), self.transform(Y=Y)
        return self.transform(X=X, Y=Y)

    def inverse_transform(
        self,
        T: Optional[ArrayLike] = None,
        U: Optional[ArrayLike] = None,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array], None]:
        """
        Reconstructs `X` and `Y` from their respective scores.

        Parameters
        ----------
        T : Array of shape (N, n_X_components) or (N, A) or None, optional, default=None
            Scores of predictor variables.
        U : Array of shape (N, n_Y_components) or (N, A) or None, optional, default=None
            Scores of response variables.

        Returns
        -------
        X_reconstructed : Array of shape (N, K)
            If `T` is not None, returns the reconstructed `X`.
        Y_reconstructed : Array of shape (N, M)
            If `U` is not None, returns the reconstructed `Y`.

        Raises
        ------
        NotFittedError
            If the model has not been fitted before calling `inverse_transform()`.

        See Also
        --------
        transform : Transforms `X` and `Y` to their respective scores.
        fit_transform : Fits the model and returns the scores of `X` and `Y`.
        """
        if not self.fitted_:
            raise NotFittedError(
                "This model is not fitted yet, call 'fit' with approriate arguments before 'inverse_transform'."
            )
        X_reconstructed = None
        Y_reconstructed = None

        if T is not None:
            T = self._convert_input_to_array(T)
            X_components = T.shape[1]
            X_reconstructed = T @ self.P[:, :X_components].T
            if self.scale_X:
                X_reconstructed = X_reconstructed * self.X_std
            if self.center_X:
                X_reconstructed = X_reconstructed + self.X_mean

        if U is not None:
            U = self._convert_input_to_array(U)
            Y_components = U.shape[1]
            Y_reconstructed = U @ self.Q[:, :Y_components].T
            if self.scale_Y:
                Y_reconstructed = Y_reconstructed * self.Y_std
            if self.center_Y:
                Y_reconstructed = Y_reconstructed + self.Y_mean

        if X_reconstructed is not None and Y_reconstructed is not None:
            return X_reconstructed, Y_reconstructed
        elif X_reconstructed is not None:
            return X_reconstructed
        elif Y_reconstructed is not None:
            return Y_reconstructed
        else:
            return None

    @partial(jax.jit, static_argnums=(0, 3, 8))
    def stateless_fit_predict_eval(
        self,
        X_train: ArrayLike,
        Y_train: ArrayLike,
        A: int,
        weights_train: Optional[ArrayLike],
        X_val: ArrayLike,
        Y_val: ArrayLike,
        weights_val: Optional[ArrayLike],
        metric_function: Union[
            Callable[
                [
                    jax.Array,
                    jax.Array,
                    jax.Array,
                ],
                Any,
            ],
            Callable[
                [
                    jax.Array,
                    jax.Array,
                ],
                Any,
            ],
        ],
    ) -> Any:
        """
        Computes `B` with `stateless_fit`. Then computes `Y_pred` with
        `stateless_predict`. `Y_pred` is an array of shape (A, N, M). Then evaluates
        and returns the result of `metric_function(Y_val, Y_pred)`.

        Parameters
        ----------
        X_train : Array of shape (N_train, K)
            Predictor variables.

        Y_train : Array of shape (N_train, M) or (N_train,)
            Response variables.

        A : int
            Number of components in the PLS model.

        weights_train : Array of shape (N_train,) or None
            Weights for each observation. If None, then all observations are weighted
            equally.

        X_val : Array of shape (N_val, K)
            Predictor variables.

        Y_val : Array of shape (N_val, M) or (N_val,)
            Response variables.

        weights_val : Array of shape (N_val,) or None
            Weights for each observation. If None, then all observations are weighted
            equally.

        metric_function : Callable receiving arrays `Y_val` (N_val, M), `Y_pred`\
        (A, N_val, M), and, if `weights` is not None, also, `weights_val` (N_val,),\
        and returning Any.
            Computes a metric based on true values `Y_val` and predicted values
            `Y_pred`. `Y_pred` contains a prediction for all `A` components.

        Returns
        -------
        metric_function(Y_val, Y_pred, weights_val) : Any.

        See Also
        --------
        stateless_fit : Fits on `X_train` and `Y_train` using `A` components while
        optionally performing centering and scaling. Then returns the internal matrices
        instead of storing them in the class instance.

        stateless_predict : Computes `Y_pred` given predictor variables `X` and
        regression tensor `B` and optionally `A` components.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"stateless_fit_predict_eval for {self.name} will be JIT compiled...")

        X_train, Y_train, weights_train = self._initialize_input_matrices(
            X=X_train, Y=Y_train, weights=weights_train
        )
        X_val, Y_val, weights_val = self._initialize_input_matrices(
            X=X_val, Y=Y_val, weights=weights_val
        )

        matrices = self.stateless_fit(
            X=X_train,
            Y=Y_train,
            A=A,
            weights=weights_train,
        )
        B = matrices[0]
        X_mean, Y_mean, X_std, Y_std = matrices[-4:]
        Y_pred = self.stateless_predict(
            X_val, B, X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std
        )
        if weights_val is None:
            return metric_function(Y_val, Y_pred)
        return metric_function(Y_val, Y_pred, weights_val)

    def cross_validate(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        A: int,
        folds: ArrayLike,
        metric_function: Union[
            Callable[
                [
                    jax.Array,
                    jax.Array,
                    jax.Array,
                ],
                Any,
            ],
            Callable[
                [
                    jax.Array,
                    jax.Array,
                ],
                Any,
            ],
        ],
        metric_names: list[str],
        preprocessing_function: Union[
            None,
            Union[
                Callable[
                    [
                        jax.Array,
                        jax.Array,
                        jax.Array,
                        jax.Array,
                        jax.Array,
                        jax.Array,
                    ],
                    Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ],
                Callable[
                    [
                        jax.Array,
                        jax.Array,
                        jax.Array,
                        jax.Array,
                    ],
                    Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ],
            ],
        ] = None,
        weights: Optional[ArrayLike] = None,
        show_progress=True,
    ) -> dict[str, Any]:
        """
        Performs cross-validation for the Partial Least-Squares (PLS) model on given
        data. `preprocessing_function` will be applied before any potential centering
        and scaling as determined by `self.center_X`, `self.center_Y`, `self.scale_X`,
        and `self.scale_Y`. Any such potential centering and scaling is applied for
        each split using training set statistics to avoid data leakage from the
        validation set.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M) or (N,)
            Response variables.

        A : int
            Number of components in the PLS model.

        folds : Array of Int of shape (N,)
            An array defining cross-validation splits. Each unique Int in `folds`
            corresponds to a different fold.

        metric_function : Callable receiving arrays `Y_val` (N_val, M), `Y_pred`\
        (A, N_val, M), and, if `weights` is not None, also, `weights_val` (N_val,),\
        and returning Any.
            Computes a metric based on true values `Y_val` and predicted values
            `Y_pred`. `Y_pred` contains a prediction for all `A` components.

        metric_names : list of str
            A list of names for the metrics used for evaluation.

        preprocessing_function : Callable or None, optional, default=None,
            A function that preprocesses the training and validation data for each
            fold. It should return preprocessed arrays for `X_train`, `Y_train`,
            `X_val`, and `Y_val`. If Callable, it should receive arrays `X_train`,
            `Y_train`, `X_val`, `Y_val`, and, if `weights` is not None, also
            `weights_train`, and `weights_val`, and returning a Tuple of preprocessed
            `X_train`, `Y_train`, `X_val`, and `Y_val`.

        show_progress : bool, default=True
            If True, displays a progress bar for the cross-validation.

        weights : Array of shape (N,) or None, optional, default=None
            Weights for each observation. If None, then all observations are weighted
            equally.

        Attributes
        ----------
        A : int
            Number of components in the PLS model.

        Returns
        -------
        metrics : dict[str, Any]
            A dictionary containing evaluation metrics for each metric specified in
            `metric_names`. The keys are metric names, and the values are lists of
            metric values for each cross-validation fold.

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
        _inner_cv : Performs cross-validation for a single fold and computes evaluation
        metrics.

        _update_metric_value_lists : Updates lists of metric values for each metric and
        fold.

        _finalize_metric_values : Organizes and finalizes the metric values into a
        dictionary for the specified metric names.

        stateless_fit_predict_eval : Fits the PLS model, makes predictions, and
        evaluates metrics for a given fold.

        Notes
        -----
        This method is used to perform cross-validation on the PLS model with different
        data splits and evaluate its performance using user-defined metrics.
        """
        X = jnp.asarray(X, dtype=self.dtype)
        Y = jnp.asarray(Y, dtype=self.dtype)
        if weights is not None:
            weights = jnp.asarray(weights, dtype=self.dtype)
            if jnp.any(weights < 0):
                raise ValueError("Weights must be non-negative.")
        self.A = A
        folds = jnp.asarray(folds, dtype=jnp.int64)
        metric_value_lists = [[] for _ in metric_names]
        unique_splits = jnp.unique(folds)
        for split in tqdm(unique_splits, disable=not show_progress):
            train_idxs = jnp.nonzero(folds != split)[0]
            val_idxs = jnp.nonzero(folds == split)[0]
            metric_values = self._inner_cross_validate(
                X=X,
                Y=Y,
                train_idxs=train_idxs,
                val_idxs=val_idxs,
                A=A,
                weights=weights,
                preprocessing_function=preprocessing_function,
                metric_function=metric_function,
            )
            metric_value_lists = self._update_metric_value_lists(
                metric_value_lists, metric_names, metric_values
            )
        return self._finalize_metric_values(metric_value_lists, metric_names)

    @partial(jax.jit, static_argnums=(0, 5, 7, 8))
    def _inner_cross_validate(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        train_idxs: jax.Array,
        val_idxs: jax.Array,
        A: int,
        weights: Optional[ArrayLike],
        preprocessing_function: Union[
            None,
            Union[
                Callable[
                    [
                        jax.Array,
                        jax.Array,
                        jax.Array,
                        jax.Array,
                        jax.Array,
                        jax.Array,
                    ],
                    Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ],
                Callable[
                    [
                        jax.Array,
                        jax.Array,
                        jax.Array,
                        jax.Array,
                    ],
                    Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ],
            ],
        ],
        metric_function: Union[
            Callable[
                [
                    jax.Array,
                    jax.Array,
                    jax.Array,
                ],
                Any,
            ],
            Callable[
                [
                    jax.Array,
                    jax.Array,
                ],
                Any,
            ],
        ],
    ):
        """
        Performs cross-validation for a single fold of the data and computes evaluation
        metrics.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M)
            Response variables.

        train_idxs : Array of shape (N_train,)
            Indices of data points in the training set.

        val_idxs : Array of shape (N_val,)
            Indices of data points in the validation set.

        A : int
            Number of components in the PLS model.

        weights : Array of shape (N,) or None
            Weights for each observation. If None, then all observations are weighted
            equally.

        preprocessing_function : Callable or None, optional, default=None,
            A function that preprocesses the training and validation data for each
            fold. It should return preprocessed arrays for `X_train`, `Y_train`,
            `X_val`, and `Y_val`. If Callable, it should receive arrays `X_train`,
            `Y_train`, `X_val`, `Y_val`, and, if `weights` is not None, also
            `weights_train`, and `weights_val`, and returning a Tuple of preprocessed
            `X_train`, `Y_train`, `X_val`, and `Y_val`.

        metric_function : Callable receiving arrays `Y_val` (N_val, M), `Y_pred`\
        (A, N_val, M), and, if `weights` is not None, also, `weights_val` (N_val,),\
        and returning Any.
            Computes a metric based on true values `Y_val` and predicted values
            `Y_pred`. `Y_pred` contains a prediction for all `A` components.

        Returns
        -------
        metric_values : Any
            metric values based on the true and predicted values for a single fold.

        Notes
        -----
        This method performs cross-validation for a single fold of the data, including
        preprocessing, fitting, predicting, and evaluating the PLS model.
        """
        if self.verbose and not jax.config.values["jax_disable_jit"]:
            print(f"_inner_cv for {self.name} will be JIT compiled...")

        X_train = jnp.take(X, train_idxs, axis=0)
        Y_train = jnp.take(Y, train_idxs, axis=0)

        X_val = jnp.take(X, val_idxs, axis=0)
        Y_val = jnp.take(Y, val_idxs, axis=0)

        if weights is not None:
            weights_train = jnp.take(weights, train_idxs)
            weights_val = jnp.take(weights, val_idxs)
        else:
            weights_train = None
            weights_val = None

        if preprocessing_function is not None:
            if weights is None:
                X_train, Y_train, X_val, Y_val = preprocessing_function(
                    X_train, Y_train, X_val, Y_val
                )
            else:
                X_train, Y_train, X_val, Y_val = preprocessing_function(
                    X_train, Y_train, X_val, Y_val, weights_train, weights_val
                )
        metric_values = self.stateless_fit_predict_eval(
            X_train=X_train,
            Y_train=Y_train,
            A=A,
            weights_train=weights_train,
            X_val=X_val,
            Y_val=Y_val,
            weights_val=weights_val,
            metric_function=metric_function,
        )
        return metric_values

    def _update_metric_value_lists(
        self,
        metric_value_lists: list[list[Any]],
        metric_names: list[str],
        metric_values: Any,
    ):
        """
        Updates lists of metric values for each metric and fold during
        cross-validation.

        Parameters
        ----------
        metric_value_lists : list of list of Any
            Lists of metric values for each metric and fold.

        metric_values : list of Any
            Metric values for a single fold.

        Returns
        -------
        metric_value_lists : list of list of Any
            Updated lists of metric values for each metric and fold.

        Notes
        -----
        This method updates the lists of metric values for each metric and fold during
        cross-validation.
        """
        if len(metric_names) == 1:
            metric_value_lists[0].append(metric_values)
        else:
            for i in range(len(metric_names)):
                metric_value_lists[i].append(metric_values[i])
        return metric_value_lists

    def _finalize_metric_values(
        self, metrics_results: list[list[Any]], metric_names: list[str]
    ):
        """
        Organizes and finalizes the metric values into a dictionary for the specified
        metric names.

        Parameters
        ----------
        metrics_results : list of list of Any
            Lists of metric values for each metric and fold.

        metric_names : list of str
            A list of names for the metrics used for evaluation.

        Returns
        -------
        metrics : dict[str, list[Any]]
            A dictionary containing evaluation metrics for each metric specified in
            `metric_names`. The keys are metric names, and the values are lists of
            metric values for each cross-validation fold.

        Notes
        -----
        This method organizes and finalizes the metric values into a dictionary for the
        specified metric names, making it easy to analyze the cross-validation results.
        """
        metrics = {}
        for name, lst_of_metric_value_for_each_split in zip(
            metric_names, metrics_results
        ):
            metrics[name] = lst_of_metric_value_for_each_split
        return metrics
