"""
This file contains the PLS Class which implements partial least-squares regression
using Improved Kernel PLS by Dayal and MacGregor:
https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

The PLS class subclasses scikit-learn's BaseEstimator to ensure compatibility with e.g.
scikit-learn's cross_validate. It is written using NumPy.

This file also contains the _R_Y_Mapping class which is a concrete Mapping to store the
PLS weights matrix to compute scores U directly from original Y for different numbers
of components.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import warnings
from collections.abc import Callable, Hashable, Mapping
from typing import Any, Iterable, Optional, Self, Tuple, Union

import joblib
import numpy as np
import numpy.linalg as la
import numpy.typing as npt
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class _R_Y_Mapping(Mapping):
    """A read-only Mapping that computes values lazily on first access."""

    def __init__(self, Q: npt.NDArray[np.floating]) -> None:
        self.Q = Q
        self.eps = np.finfo(Q.dtype).eps
        self._valid_keys = set(range(1, Q.shape[1] + 1))
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._valid_keys:
            raise KeyError(
                f"Invalid number of components: {key}. Valid numbers of components are 1 to {self.Q.shape[1]}."
            )
        if key not in self._cache:
            self._cache[key] = la.pinv(self.Q[:, :key].T, rcond=self.eps)
        return self._cache[key]

    def __iter__(self):
        return iter(self._valid_keys)

    def __len__(self):
        return len(self._valid_keys)

    def __contains__(self, key):
        return key in self._valid_keys

    def __repr__(self):
        return f"_R_Y_Mapping(Cached R_Y for {len(self._cache)}/{len(self._valid_keys)} n_components)"


class PLS(BaseEstimator):
    """
    Implements partial least-squares regression using Improved Kernel PLS by Dayal and
    MacGregor:
    https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

    Parameters
    ----------
    algorithm : int, default=1
        Whether to use Improved Kernel PLS Algorithm #1 or #2.

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

    copy : bool, default=True
            Whether to copy `X` and `Y` in before potentially applying centering and
            scaling. If True, then the data is copied before fitting. If False, and `dtype`
            matches the type of `X` and `Y`, then centering and scaling is done inplace,
            modifying both arrays.

    dtype : type[np.floating], default=numpy.float64
        The float datatype to use in computation of the PLS algorithm. This should be
        numpy.float32 or numpy.float64. Using a lower precision than float64
        will yield significantly worse results when using an increasing number of
        components due to propagation of numerical errors.

    Raises
    ------
    ValueError
        If `algorithm` is not 1 or 2.

    Notes
    -----
    Any centering and scaling is undone before returning predictions to ensure that
    predictions are on the original scale. If both centering and scaling are True, then
    the data is first centered and then scaled.
    """

    def __init__(
        self,
        algorithm: int = 1,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        ddof: int = 1,
        copy: bool = True,
        dtype: type[np.floating] = np.float64,
    ) -> None:
        self.algorithm = algorithm
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.ddof = ddof
        self.copy = copy
        self.dtype = dtype
        self.eps = np.finfo(dtype).eps
        self.name = f"Improved Kernel PLS Algorithm #{algorithm}"
        if self.algorithm not in [1, 2]:
            raise ValueError(
                f"Invalid algorithm: {self.algorithm}. Algorithm must be 1 or 2."
            )
        self.A = None
        self.N = None
        self.K = None
        self.M = None
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

    def _weight_warning(self, i: int) -> None:
        """
        Warns the user that the weight is close to zero.

        Parameters
        ----------
        i : int
            Number of components.

        Returns
        -------
        None.
        """
        warnings.warn(
            message=f"Weight is close to zero. Results with A = {i + 1} "
            "component(s) or higher may be unstable.",
            category=UserWarning,
        )
        self.max_stable_components = i

    def _convert_input_to_array(
        self, arr: Optional[npt.ArrayLike]
    ) -> Optional[npt.NDArray[np.floating]]:
        """
        Converts input array to numpy array of type self.dtype. Inserts a second axis
        if the input array is 1-dimensional.

        Parameters
        ----------
        arr : ArrayLike or None
            Input array.

        Returns
        -------
        array_converted : Array of shape of input array or None
            Converted input array.
        """
        if arr is not None:
            arr = np.asarray(arr, dtype=self.dtype)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
        return arr

    def _copy_arrays(
        self,
        copy_X: bool,
        copy_Y: bool,
        X: Optional[npt.NDArray[np.floating]] = None,
        Y: Optional[npt.NDArray[np.floating]] = None,
    ) -> Tuple[Optional[npt.NDArray[np.floating]], Optional[npt.NDArray[np.floating]]]:
        """
        Copies `X and `Y` so that downstream centering and scaling do not modify the
        input directly but instead work on copies thereof.

        Parameters
        ----------
        copy_X: bool
            Whether to copy `X`.

        copy_Y : bool
            Whether to copy `Y`.

        X : Array of shape (N, K) or None
            Predictor variables.

        Y : Array of shape (N, M) or None
            Response variables.

        Returns
        -------
        X_copy : Array of shape (N, K) or None
            A copy of `X` if it is provided and the following conditions are met:
            `copy_X` is True, and `X` will be either centered or scaled.

        Y_copy : Array of shape (N, M) or None
            A copy of `Y` if it is provided and the following conditions are met:
            `copy_Y` is True, and `Y` will be either centered or scaled.
        """
        if copy_X and X is not None and (self.center_X or self.scale_X):
            X = X.copy()
        if copy_Y and Y is not None and (self.center_Y or self.scale_Y):
            Y = Y.copy()
        return X, Y

    def _get_input(
        self,
        copy: bool,
        X: Optional[npt.ArrayLike] = None,
        Y: Optional[npt.ArrayLike] = None,
        weights: Optional[npt.ArrayLike] = None,
    ) -> Tuple[
        Optional[npt.ArrayLike], Optional[npt.ArrayLike], Optional[npt.ArrayLike]
    ]:
        """
        Applies `self._convert_input_to_array` on `X`, `Y`, and `weights` and then
        applies `self._copy_arrays` to `X` and `Y`. See those two functions for a
        description of parameters and return values.

        Returns
        -------
        X, Y, weights
            Converted inputs for safe downstream processing.
        """

        new_X = self._convert_input_to_array(arr=X)
        new_Y = self._convert_input_to_array(arr=Y)
        copy_X = copy and (np.may_share_memory(new_X, X))
        copy_Y = copy and (np.may_share_memory(new_Y, Y))
        weights = self._convert_input_to_array(arr=weights)
        if weights is not None:
            self._check_nonnegative_weights(weights=weights)
        new_X, new_Y = self._copy_arrays(copy_X=copy_X, copy_Y=copy_Y, X=new_X, Y=new_Y)
        return new_X, new_Y, weights

    def _center_scale_X_Y(
        self,
        X: Optional[npt.NDArray[np.floating]] = None,
        Y: Optional[npt.NDArray[np.floating]] = None,
    ) -> Tuple[Optional[npt.NDArray[np.floating]], Optional[npt.NDArray[np.floating]]]:
        """
        Centers and scales the input array based on the model's parameters.

        Parameters
        ----------
        X : Array of shape (N, K) or None
            Predictor variables.

        Y : Array of shape (N, M) or None
            Response variables.

        Returns
        -------
        X_centered_scaled : Array of shape (N, K) or None
            Potentially centered and scaled `X`.

        Y_centered_scaled : Array of shape (N, M) or None
            Potentially centered and scaled `Y`.
        """
        if X is not None:
            if self.center_X:
                X -= self.X_mean if self.X_mean is not None else X
            if self.scale_X:
                X /= self.X_std if self.X_std is not None else X
        if Y is not None:
            if self.center_Y:
                Y -= self.Y_mean if self.Y_mean is not None else Y
            if self.scale_Y:
                Y /= self.Y_std if self.Y_std is not None else Y
        return (X, Y)

    def _un_center_scale_X_Y(
        self,
        X: Optional[npt.NDArray[np.floating]] = None,
        Y: Optional[npt.NDArray[np.floating]] = None,
    ) -> Tuple[Optional[npt.NDArray[np.floating]], Optional[npt.NDArray[np.floating]]]:
        """
        Restores `X` and `Y`to their original scale and location by undoing any
        centering and scaling.

        Parameters
        ----------
        X : Array of shape (N, K) or None
            Predictor variables.

        Y : Array of shape (N, M) or None
            Response variables.

        Returns
        -------
        X_restored : Array of shape (N, K) or None
            `X` with any centering and scaling undone.

        Y_restored : Array of shape (N, M) or None
            `Y` with any centering and scaling undone.
        """
        if X is not None:
            if self.scale_X:
                X *= self.X_std
            if self.center_X:
                X += self.X_mean
        if Y is not None:
            if self.scale_Y:
                Y *= self.Y_std
            if self.center_Y:
                Y += self.Y_mean
        return (X, Y)

    def _check_nonnegative_weights(self, weights: npt.NDArray[np.floating]) -> None:
        """
        Checks that weights are non-negative.

        Parameters
        ----------
        weights : Array of shape (N, 1)
            Sample weights used to fit PLS.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If weights are not all non-negative.
        """
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    def _check_num_nonzero_weights_greater_than_ddof(
        self, num_nonzero_weights: np.intp
    ) -> None:
        """
        Checks that the number of nonzero weights are greater than `self.ddof` if
        either of `self.scale_X` or `self.scale_Y` is True so we must compute the
        corresponding weighted standard deviation.

        Parameters
        ----------
        num_nonzero_weights : int
            The number of nonzero weights."

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the number of nonzero weights is not greater than `self.ddof`
        """
        if self.scale_X or self.scale_Y:
            if not num_nonzero_weights > self.ddof:
                raise ValueError(
                    f"The number of nonzero weights: {num_nonzero_weights} must be greater than ddof: {self.ddof}"
                )

    def _compute_mean_and_std(
        self,
        X: npt.NDArray[np.floating],
        Y: npt.NDArray[np.floating],
        weights: Optional[npt.NDArray[np.floating]],
    ) -> None:
        """
        Computes the weighted means and standard deviations for `X` and `Y`.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M):
            Response variables.

        weights : Array of shape (N, 1) or None
            Sample weights.

        Attributes
        ----------
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
        None

        Raises
        ------
        ValueError
            If `weights` are provided and scaling of `X` or `Y` is required and the
            number of non-negative weights is less or equal to `self.ddof`.
        """
        if not (self.center_X or self.center_Y or self.scale_X or self.scale_Y):
            return
        if weights is not None:
            flattened_weights = weights.flatten()
            if self.scale_X or self.scale_Y:
                num_non_zero_weights = np.count_nonzero(weights)
                self._check_num_nonzero_weights_greater_than_ddof(
                    num_nonzero_weights=num_non_zero_weights
                )
                scale_dof = num_non_zero_weights - self.ddof
                avg_non_zero_weights = np.sum(weights) / num_non_zero_weights
        else:
            flattened_weights = None

        if self.center_X or self.scale_X:
            self.X_mean = np.asarray(
                np.average(X, axis=0, weights=flattened_weights, keepdims=True),
                dtype=self.dtype,
            )

        if self.center_Y or self.scale_Y:
            self.Y_mean = np.asarray(
                np.average(Y, axis=0, weights=flattened_weights, keepdims=True),
                dtype=self.dtype,
            )

        if self.scale_X:
            if weights is None:
                self.X_std = X.std(
                    axis=0,
                    ddof=self.ddof,
                    dtype=self.dtype,
                    keepdims=True,
                    mean=self.X_mean,
                )
            else:
                self.X_std = np.sqrt(
                    np.sum(weights * (X - self.X_mean) ** 2, axis=0, keepdims=True)
                    / (scale_dof * avg_non_zero_weights)
                )
            self.X_std[self.X_std <= self.eps] = 1

        if self.scale_Y:
            if weights is None:
                self.Y_std = Y.std(
                    axis=0,
                    ddof=self.ddof,
                    dtype=self.dtype,
                    keepdims=True,
                    mean=self.Y_mean,
                )
            else:
                self.Y_std = np.sqrt(
                    np.sum(weights * (Y - self.Y_mean) ** 2, axis=0, keepdims=True)
                    / (scale_dof * avg_non_zero_weights)
                )
            self.Y_std[self.Y_std <= self.eps] = 1

    def fit(
        self,
        X: npt.ArrayLike,
        Y: npt.ArrayLike,
        A: int,
        weights: Optional[npt.ArrayLike] = None,
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
        self : PLS
            Fitted model.

        Raises
        ------
        ValueError
            If `weights` are provided and not all weights are non-negative.

        ValueError
            If `weights` are provided, and `scale_X` or `scale_Y` is True, and the
            number of non-zero weights is not greater than `ddof`.

        Warns
        -----
        UserWarning.
            If at any point during iteration over the number of components `A`, the
            residual goes below machine precision for np.float64.

        See Also
        --------
        transform : Transforms `X` and `Y` to their respective scores.
        fit_transform : Fits the model and returns the scores of `X` and `Y`.

        Notes
        -----
        `R_Y` is provided for convenience only as it is not required to derive `B`.
        Therefore, every value in `R_Y` is computed lazily and only actually evaluated
        when accessed by its key for the first time after a call to `fit` - either by
        the user or because it is needed by `transform`. After a value is computed, it
        is cached for fast future retrieval. R_Y is implemented as a concrete Mapping.
        """
        self.fitted_ = True
        X, Y, weights = self._get_input(copy=self.copy, X=X, Y=Y, weights=weights)
        self._compute_mean_and_std(X=X, Y=Y, weights=weights)
        X, Y = self._center_scale_X_Y(X=X, Y=Y)

        N, K = X.shape
        M = Y.shape[1]

        self.B = np.zeros(shape=(A, K, M), dtype=self.dtype)
        W = np.zeros(shape=(A, K), dtype=self.dtype)
        P = np.zeros(shape=(A, K), dtype=self.dtype)
        Q = np.zeros(shape=(A, M), dtype=self.dtype)
        R = np.zeros(shape=(A, K), dtype=self.dtype)
        self.W = W.T
        self.P = P.T
        self.Q = Q.T
        self.R = R.T
        self.R_Y = _R_Y_Mapping(self.Q)
        if self.algorithm == 1:
            T = np.zeros(shape=(A, N), dtype=self.dtype)
            self.T = T.T
        self.A = A
        self.max_stable_components = A
        self.N = N
        self.K = K
        self.M = M

        # Step 1
        if weights is not None:
            if self.algorithm == 2 or K < M:
                XTW = (X * weights).T
                XTY = XTW @ Y
            else:
                XTY = X.T @ (Y * weights)
        else:
            XTY = X.T @ Y

        # Used for algorithm #2
        if self.algorithm == 2:
            if weights is not None:
                XTX = XTW @ X
            else:
                XTX = X.T @ X
        else:
            if weights is not None:
                X = np.sqrt(weights) * X

        for i in range(A):
            # Step 2
            if M == 1:
                norm = la.norm(XTY, ord=2)
                if np.isclose(norm, 0, atol=self.eps, rtol=0):
                    self._weight_warning(i)
                    break
                w = XTY / norm
            elif M < K:
                XTYTXTY = XTY.T @ XTY
                eig_vals, eig_vecs = la.eigh(XTYTXTY)
                q = eig_vecs[:, -1:]
                w = XTY @ q
                norm = la.norm(w, ord=2)
                if np.isclose(norm, 0, atol=self.eps, rtol=0):
                    self._weight_warning(i)
                    break
                w = w / norm
            else:
                XTYYTX = XTY @ XTY.T
                eig_vals, eig_vecs = la.eigh(XTYYTX)
                norm = eig_vals[-1]
                if np.isclose(norm, 0, atol=self.eps, rtol=0):
                    self._weight_warning(i)
                    break
                w = eig_vecs[:, -1:]
            W[i] = w.squeeze()

            # Step 3
            r = np.copy(w)
            for j in range(i):
                r = r - P[j].reshape(-1, 1).T @ w * R[j].reshape(-1, 1)
            R[i] = r.squeeze()

            # Step 4
            if self.algorithm == 1:
                t = X @ r
                T[i] = t.squeeze()
                tTt = t.T @ t
                p = (t.T @ X).T / tTt
            else:
                rXTX = r.T @ XTX
                tTt = rXTX @ r
                p = rXTX.T / tTt
            if M == 1:
                q = norm / tTt
            elif M < K:
                q = q * np.sqrt(eig_vals[-1]) / tTt
            else:
                q = (r.T @ XTY).T / tTt
            P[i] = p.squeeze()
            Q[i] = q.squeeze()

            # Step 5
            XTY = XTY - (p @ q.T) * tTt

            # Compute regression coefficients
            self.B[i] = self.B[i - 1] + r @ q.T

        return self

    def predict(
        self, X: npt.ArrayLike, n_components: Optional[int] = None
    ) -> npt.NDArray[np.floating]:
        """
        Predicts on `X` with `B` using `n_components` components. If `n_components` is
        None, then predictions are returned for each individual number of components.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        n_components : int or None, optional, default=None.
            Number of components in the PLS model. If None, then each individual number
            of components is used.

        Returns
        -------
        Y_pred : Array of shape (N, M) or (A, N, M)
            If `n_components` is an int, then an array of shape (N, M) with the
            predictions for that specific number of components is used. If
            `n_components` is None, returns a prediction for each number of components
            up to `A`.

        Raises
        ------
        NotFittedError
            If the model has not been fitted before calling `predict()`.
        """
        if not self.fitted_:
            raise NotFittedError(
                "This model is not fitted yet, call 'fit' with approriate arguments before 'predict'."
            )
        X, _, _ = self._get_input(copy=True, X=X)
        X, _ = self._center_scale_X_Y(X=X)

        if n_components is None:
            Y_pred = X @ self.B
        else:
            Y_pred = X @ self.B[n_components - 1]

        _, Y_pred = self._un_center_scale_X_Y(Y=Y_pred)
        return Y_pred

    def transform(
        self,
        X: Optional[npt.ArrayLike] = None,
        Y: Optional[npt.ArrayLike] = None,
        n_components: Optional[int] = None,
    ) -> Union[npt.ArrayLike, Tuple[npt.ArrayLike, npt.ArrayLike], None]:
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
        X, Y, _ = self._get_input(copy=True, X=X, Y=Y)
        X, Y = self._center_scale_X_Y(X=X, Y=Y)
        if X is not None:
            T = X @ self.R[:, :n_components]
        if Y is not None:
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
        X: npt.ArrayLike,
        Y: npt.ArrayLike,
        A: int,
        weights: Optional[npt.ArrayLike] = None,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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
        if self.algorithm == 1 and weights is None:
            return np.copy(self.T), self.transform(Y=Y)
        return self.transform(X=X, Y=Y)

    def inverse_transform(
        self,
        T: Optional[npt.ArrayLike] = None,
        U: Optional[npt.ArrayLike] = None,
    ) -> Union[npt.ArrayLike, Tuple[npt.ArrayLike, npt.ArrayLike], None]:
        """
        Reconstructs `X` and `Y` from their respective scores.

        Parameters
        ----------
        T : Array of shape (N, n_X_components) or (N, A) or None, optional, default=None
            Scores of predictor variables (X).
        U : Array of shape (N, n_Y_components) or (N, A) or None, optional, default=None
            Scores of response variables (Y).

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
        T = self._convert_input_to_array(arr=T)
        U = self._convert_input_to_array(arr=U)

        if T is not None:
            X_components = T.shape[1]
            X_reconstructed = T @ self.P[:, :X_components].T

        if U is not None:
            Y_components = U.shape[1]
            Y_reconstructed = U @ self.Q[:, :Y_components].T

        X_reconstructed, Y_reconstructed = self._un_center_scale_X_Y(
            X=X_reconstructed, Y=Y_reconstructed
        )

        if X_reconstructed is not None and Y_reconstructed is not None:
            return X_reconstructed, Y_reconstructed
        elif X_reconstructed is not None:
            return X_reconstructed
        elif Y_reconstructed is not None:
            return Y_reconstructed
        else:
            return None

    def cross_validate(
        self,
        X: npt.ArrayLike,
        Y: npt.ArrayLike,
        A: int,
        folds: Iterable[Hashable],
        metric_function: Union[
            Callable[
                [
                    npt.NDArray[np.floating],
                    npt.NDArray[np.floating],
                    npt.NDArray[np.floating],
                ],
                Any,
            ],
            Callable[
                [
                    npt.NDArray[np.floating],
                    npt.NDArray[np.floating],
                ],
                Any,
            ],
        ],
        preprocessing_function: Union[
            None,
            Union[
                Callable[
                    [
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                    ],
                    Tuple[
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                    ],
                ],
                Callable[
                    [
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                    ],
                    Tuple[
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                        npt.NDArray[np.floating],
                    ],
                ],
            ],
        ] = None,
        weights: Optional[npt.ArrayLike] = None,
        n_jobs: int = -1,
        verbose: int = 10,
    ) -> dict[Hashable, Any]:
        """
        Performs cross-validation for the Partial Least-Squares (PLS) model on given
        data. `preprocessing_function` will be applied before any potential centering
        and scaling as determined by `self.center_X`, `self.center_Y`, `self.scale_X`,
        and `self.scale_Y`. Any such potential centering and scaling is applied for
        each split using training set statistics to avoid data leakage from the
        validation set. If `weights` are provided, then these will be provided in
        `preprocessing_function` and `metric_function` and used for any centering and
        scaling in the cross-validation procedure.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M) or (N,)
            Response variables.

        A : int
            Number of components in the PLS model.

        folds : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `folds` corresponds to a different fold.

        metric_function : Callable receiving arrays `Y_val` (N_val, M), `Y_pred`\
        (A, N_val, M), and, if `weights` is not None, also, `weights_val` (N_val,),\
        and returning Any.
            Computes a metric based on true values `Y_val` and predicted values
            `Y_pred`. `Y_pred` contains a prediction for all `A` components.

        preprocessing_function : Callable or None, optional, default=None,
            A function that preprocesses the training and validation data for each
            fold. It should return preprocessed arrays for `X_train`, `Y_train`,
            `X_val`, and `Y_val`. If Callable, it should receive arrays `X_train`,
            `Y_train`, `X_val`, `Y_val`, and, if `weights` is not None, also
            `weights_train`, and `weights_val`, and returning a Tuple of preprocessed
            `X_train`, `Y_train`, `X_val`, and `Y_val`.

        weights : Array of shape (N,) or None, optional, default=None
            Weights for each observation. If None, then all observations are weighted
            equally.

        n_jobs : int, optional default=-1
            Number of parallel jobs to use. A value of -1 will use the minimum of all
            available cores and the number of unique values in `folds`.

        verbose : int, optional default=10
            Controls verbosity of parallel jobs.

        Returns
        -------
        metrics : dict of Hashable to Any
            A dictionary mapping each unique value in `folds` to the result of
            evaluating `metric_function` on the validation set corresponding to that
            value.

        Raises
        ------
        ValueError
            If `weights` are provided and not all weights are non-negative.

        Notes
        -----
        This method is used to perform cross-validation on the PLS model with different
        data splits and evaluate its performance using user-defined metrics.
        """
        X, Y, weights = self._get_input(copy=False, X=X, Y=Y, weights=weights)

        if weights is not None:
            self._check_nonnegative_weights(weights=weights)
            flattened_weights = weights.squeeze()
        else:
            flattened_weights = None

        folds_dict = self._init_folds_dict(folds)
        num_splits = len(folds_dict)
        all_indices = np.arange(X.shape[0])

        if n_jobs == -1:
            n_jobs = min(num_splits, joblib.cpu_count())
        else:
            n_jobs = min(num_splits, n_jobs)

        def worker(val_indices: npt.NDArray[np.int_]) -> Any:
            train_indices = np.setdiff1d(all_indices, val_indices, assume_unique=True)
            X_train = X[train_indices]
            Y_train = Y[train_indices]
            X_val = X[val_indices]
            Y_val = Y[val_indices]
            if flattened_weights is not None:
                weights_train = flattened_weights[train_indices]
                weights_val = flattened_weights[val_indices]
            else:
                weights_train = None
                weights_val = None

            if preprocessing_function is not None:
                if flattened_weights is not None:
                    X_train, Y_train, X_val, Y_val = preprocessing_function(
                        X_train, Y_train, X_val, Y_val, weights_train, weights_val
                    )
                else:
                    X_train, Y_train, X_val, Y_val = preprocessing_function(
                        X_train, Y_train, X_val, Y_val
                    )

            self.fit(X_train, Y_train, A, weights_train)
            Y_pred = self.predict(X_val, n_components=None)
            if flattened_weights is not None:
                return metric_function(Y_val, Y_pred, weights_val)
            return metric_function(Y_val, Y_pred)

        metrics_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(worker)(val_indices) for val_indices in folds_dict.values()
        )
        metrics_dict = dict(zip(folds_dict, metrics_list))
        return metrics_dict

    def _init_folds_dict(
        self, folds: Iterable[Hashable]
    ) -> dict[Hashable, npt.NDArray[np.int_]]:
        """
        Generates a list of validation indices for each fold in `folds`.

        Parameters
        ----------
        folds : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `folds` corresponds to a different fold.

        Returns
        -------
        index_dict : dict of Hashable to Array
            A dictionary mapping each unique value in `folds` to an array of
            validation indices.
        """
        index_dict = {}
        for i, fold in enumerate(folds):
            try:
                index_dict[fold].append(i)
            except KeyError:
                index_dict[fold] = [i]
        for fold in index_dict:
            index_dict[fold] = np.asarray(index_dict[fold], dtype=int)
        return index_dict
