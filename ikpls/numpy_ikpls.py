"""
Contains the PLS Class which implements partial least-squares regression using Improved
Kernel PLS by Dayal and MacGregor:
https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

The PLS class subclasses scikit-learn's BaseEstimator to ensure compatibility with e.g.
scikit-learn's cross_validate. It is written using NumPy.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import warnings
from collections.abc import Callable, Hashable
from typing import Any, Iterable, Optional, Tuple, Union

import joblib
import numpy as np
import numpy.linalg as la
import numpy.typing as npt
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator


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
        Whether to copy `X` and `Y` in fit before potentially applying centering and
        scaling. If True, then the data is copied before fitting. If False, and `dtype`
        matches the type of `X` and `Y`, then centering and scaling is done inplace,
        modifying both arrays.

    dtype : numpy.float, default=numpy.float64
        The float datatype to use in computation of the PLS algorithm. Using a lower
        precision than float64 will yield significantly worse results when using an
        increasing number of components due to propagation of numerical errors.

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
        dtype: np.floating = np.float64,
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
        self.T = None
        self.X_mean = None
        self.Y_mean = None
        self.X_std = None
        self.Y_std = None
        self.max_stable_components = None

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

    def fit(
        self,
        X: npt.ArrayLike,
        Y: npt.ArrayLike,
        A: int,
        weights: Optional[npt.ArrayLike] = None,
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

        T : Array of shape (N, A)
            PLS scores matrix of X. Only assigned for Improved Kernel PLS Algorithm #1.

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
            residual goes below machine precision for np.float64.
        """
        X = np.asarray(X, dtype=self.dtype)
        Y = np.asarray(Y, dtype=self.dtype)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        N, K = X.shape
        M = Y.shape[1]

        if weights is not None:
            weights = np.asarray(weights, dtype=self.dtype)
            weights = np.reshape(weights, (-1, 1))
            flattened_weights = weights.flatten()
            if not np.all(weights >= 0):
                raise ValueError("Weights must be non-negative.")
            if self.scale_X or self.scale_Y:
                num_non_zero_weights = np.asarray(
                    np.count_nonzero(weights), dtype=self.dtype
                )
                scale_dof = num_non_zero_weights - self.ddof
                avg_non_zero_weights = np.sum(weights) / num_non_zero_weights
        else:
            flattened_weights = None

        if (self.center_X or self.scale_X) and self.copy:
            X = X.copy()

        if (self.center_Y or self.scale_Y) and self.copy:
            Y = Y.copy()

        if self.center_X or self.scale_X:
            self.X_mean = np.asarray(
                np.average(X, axis=0, weights=flattened_weights, keepdims=True),
                dtype=self.dtype,
            )
        if self.center_X:
            X -= self.X_mean

        if self.center_Y or self.scale_Y:
            self.Y_mean = np.asarray(
                np.average(Y, axis=0, weights=flattened_weights, keepdims=True),
                dtype=self.dtype,
            )
        if self.center_Y:
            Y -= self.Y_mean

        if self.scale_X:
            new_X_mean = 0 if self.center_X else self.X_mean
            if weights is None:
                self.X_std = X.std(
                    axis=0,
                    ddof=self.ddof,
                    dtype=self.dtype,
                    keepdims=True,
                    mean=new_X_mean,
                )
            else:
                self.X_std = np.sqrt(
                    np.sum(weights * (X - new_X_mean) ** 2, axis=0, keepdims=True)
                    / (scale_dof * avg_non_zero_weights)
                )
            self.X_std[np.abs(self.X_std) <= self.eps] = 1
            X /= self.X_std

        if self.scale_Y:
            new_Y_mean = 0 if self.center_Y else self.Y_mean
            if weights is None:
                self.Y_std = Y.std(
                    axis=0,
                    ddof=self.ddof,
                    dtype=self.dtype,
                    keepdims=True,
                    mean=new_Y_mean,
                )
            else:
                self.Y_std = np.sqrt(
                    np.sum(weights * (Y - new_Y_mean) ** 2, axis=0, keepdims=True)
                    / (scale_dof * avg_non_zero_weights)
                )
            self.Y_std[np.abs(self.Y_std) <= self.eps] = 1
            Y /= self.Y_std

        self.B = np.zeros(shape=(A, K, M), dtype=self.dtype)
        W = np.zeros(shape=(A, K), dtype=self.dtype)
        P = np.zeros(shape=(A, K), dtype=self.dtype)
        Q = np.zeros(shape=(A, M), dtype=self.dtype)
        R = np.zeros(shape=(A, K), dtype=self.dtype)
        self.W = W.T
        self.P = P.T
        self.Q = Q.T
        self.R = R.T
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
            else:
                if M < K:
                    XTYTXTY = XTY.T @ XTY
                    eig_vals, eig_vecs = la.eigh(XTYTXTY)
                    q = eig_vecs[:, -1:]
                    w = XTY @ q
                    norm = la.norm(w)
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
            q = (r.T @ XTY).T / tTt
            P[i] = p.squeeze()
            Q[i] = q.squeeze()

            # Step 5
            XTY = XTY - (p @ q.T) * tTt

            # Compute regression coefficients
            self.B[i] = self.B[i - 1] + r @ q.T

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
        """
        X = np.asarray(X, dtype=self.dtype)
        if self.center_X:
            X = X - self.X_mean
        if self.scale_X:
            X = X / self.X_std

        if n_components is None:
            Y_pred = X @ self.B
        else:
            Y_pred = X @ self.B[n_components - 1]

        if self.scale_Y:
            Y_pred = Y_pred * self.Y_std
        if self.center_Y:
            Y_pred = Y_pred + self.Y_mean
        return Y_pred

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
    ) -> dict[str, Any]:
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

        metric_function : Callable receiving arrays `Y_val`, `Y_pred`, and, if
        `weights` is not None, also, `weights_val`, and returning Any.
            Computes a metric based on true values `Y_val` and predicted values
            `Y_pred`. `Y_pred` contains a prediction for all `A` components.

        preprocessing_function : Callable or None, optional, default=None,
        If Callable, it should receive arrays `X_train`, `Y_train`, `X_val`, `Y_val`,
        and, if `weights` is not None, also `weights_train`, and `weights_val`, and
        returning a Tuple of preprocessed `X_train`, `Y_train`, `X_val`, and `Y_val`.
            A function that preprocesses the training and validation data for each
            fold. It should return preprocessed arrays for `X_train`, `Y_train`,
            `X_val`, and `Y_val`.

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
        X = np.asarray(X, dtype=self.dtype)
        Y = np.asarray(Y, dtype=self.dtype)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if weights is not None:
            weights = np.asarray(weights, dtype=self.dtype)
            weights = weights.squeeze()
            if np.any(weights < 0):
                raise ValueError("Weights must be non-negative.")

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
            if weights is not None:
                weights_train = weights[train_indices]
                weights_val = weights[val_indices]
            else:
                weights_train = None
                weights_val = None

            if preprocessing_function is not None:
                if weights is not None:
                    X_train, Y_train, X_val, Y_val = preprocessing_function(
                        X_train, Y_train, X_val, Y_val, weights_train, weights_val
                    )
                else:
                    X_train, Y_train, X_val, Y_val = preprocessing_function(
                        X_train, Y_train, X_val, Y_val
                    )

            self.fit(X_train, Y_train, A, weights_train)
            Y_pred = self.predict(X_val, n_components=None)
            if weights is not None:
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
        for i, num in enumerate(folds):
            try:
                index_dict[num].append(i)
            except KeyError:
                index_dict[num] = [i]
        for key in index_dict:
            index_dict[key] = np.asarray(index_dict[key], dtype=int)
        return index_dict
