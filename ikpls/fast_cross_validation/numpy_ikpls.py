"""
Contains the PLS class which implements fast cross-validation with partial
least-squares regression using Improved Kernel PLS by Dayal and MacGregor:
https://arxiv.org/abs/2401.13185
https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

The implementation is written using NumPy and allows for parallelization of the
cross-validation process using joblib.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import warnings
from collections.abc import Callable, Hashable
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import numpy.linalg as la
import numpy.typing as npt
from cvmatrix.cvmatrix import CVMatrix
from cvmatrix.partitioner import Partitioner
from joblib import Parallel, delayed


class PLS:
    """
    Implements fast cross-validation with partial least-squares regression using
    Improved Kernel PLS by Dayal and MacGregor:
    https://arxiv.org/abs/2401.13185
    https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

    Parameters
    ----------
    algorithm : int, default=1
        Whether to use Improved Kernel PLS Algorithm #1 or #2. Generally, Algorithm #1
        is faster if `X` has less rows than columns, while Algorithm #2 is faster if
        `X` has more rows than columns.

    center_X : bool, optional default=True
        Whether to center `X` before fitting by subtracting its row of
        column-wise means from each row. The row of column-wise means is computed on
        the training set for each fold to avoid data leakage.

    center_Y : bool, optional default=True
        Whether to center `Y` before fitting by subtracting its row of
        column-wise means from each row. The row of column-wise means is computed on
        the training set for each fold to avoid data leakage.

    scale_X : bool, optional default=True
        Whether to scale `X` before fitting by dividing each row with the row of `X`'s
        column-wise standard deviations. The row of column-wise standard deviations is
        computed on the training set for each fold to avoid data leakage.

    scale_Y : bool, optional default=True
        Whether to scale `Y` before fitting by dividing each row with the row of `X`'s
        column-wise standard deviations. The row of column-wise standard deviations is
        computed on the training set for each fold to avoid data leakage.

    ddof : int, default=1
        The delta degrees of freedom to use when computing the sample standard
        deviation. A value of 0 corresponds to the biased estimate of the sample
        standard deviation, while a value of 1 corresponds to Bessel's correction for
        the sample standard deviation.

    dtype : numpy.float, default=numpy.float64
        The float datatype to use in computation of the PLS algorithm. Using a lower
        precision than float64 will yield significantly worse results when using an
        increasing number of components due to propagation of numerical errors.

    copy : bool, default=True
        Whether to copy `X`, `Y`, and `weights` when cross-validating. If True or the
        data is not already arrays of the specified `dtype`, then the data is copied.
        Otherwise, the data is not copied and changes to `X`, `Y`, and `weights`
        outside may affect the results of the cross-validation.

    Raises
    ------
    ValueError
        If `algorithm` is not 1 or 2.

    Notes
    -----
    Any centering and scaling is undone before returning predictions to ensure that
    predictions are on the original scale. If both centering and scaling are True, then
    the data is first centered and then scaled.

    Setting either of `center_X`, `center_Y`, `scale_X`, or `scale_Y` to True, while
    using multiple jobs, will increase the memory consumption as each job will then
    have to keep its own copy of :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
    :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` with its specific centering and scaling.
    """

    def __init__(
        self,
        algorithm: int = 1,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        ddof: int = 1,
        dtype: np.floating = np.float64,
        copy: bool = True,
    ) -> None:
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.algorithm = algorithm
        self.ddof = ddof
        self.dtype = dtype
        self.copy = copy
        self.eps = np.finfo(dtype).eps
        self.name = f"Improved Kernel PLS Algorithm #{algorithm}"
        if self.algorithm not in [1, 2]:
            raise ValueError(
                f"Invalid algorithm: {self.algorithm}. Algorithm must be 1 or 2."
            )
        self.sqrt_weights = None  # Used for algorithm 1
        self.A = None
        self.N = None
        self.K = None
        self.M = None
        self.cvm = None
        if self.algorithm == 1:
            self.all_indices = None

    def _weight_warning(self, i: int) -> None:
        """
        Raises a warning if the weight is close to zero.

        Parameters
        ----------
        norm : float
            The norm of the weight vector.

        i : int
            The current number of components.
        """
        warnings.warn(
            message=f"Weight is close to zero. Results with A = {i + 1} "
            "component(s) or higher may be unstable.",
            category=UserWarning,
        )

    def _stateless_fit(
        self,
        validation_indices: npt.NDArray[np.int_],
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        Optional[npt.NDArray[np.floating]],
        Optional[npt.NDArray[np.floating]],
        Optional[npt.NDArray[np.floating]],
        Optional[npt.NDArray[np.floating]],
        Optional[npt.NDArray[np.floating]],
    ]:
        """
        Fits Improved Kernel PLS Algorithm #1 on `X` and `Y` using `A` components.

        Parameters
        ----------
        validation_indices : Array of shape (N_val,)
            Integer array defining indices into X and Y corresponding to validation
            samples.

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

        T : None or Array of shape (A, N_train)
            PLS scores matrix of X. Only Returned for Improved Kernel PLS Algorithm #1.
            If `self.algorithm` is 2, then this will be None.

        training_X_mean : None or Array of shape (1, K)
            Mean row of training X. Will be None if `self.center_X`, `self.center_Y`
            and `self.scale_X` are all False.

        training_Y_mean : None or Array of shape (1, M)
            Mean row of training Y. Will be None if `self.center_X`, `self.center_Y`
            and `self.scale_Y` are all False.

        training_X_std : Array of shape (1, K)
            Sample standard deviation row of training X. Will be None if `self.scale_X`
            is False.

        training_Y_std : Array of shape (1, M)
            Sample standard deviation row of training Y. Will be None if `self.scale_Y`
            is False.

        Warns
        -----
        UserWarning.
            If at any point during iteration over the number of components `A`, the
            residual goes below machine epsilon.
        """

        B = np.zeros(shape=(self.A, self.K, self.M), dtype=self.dtype)
        WT = np.zeros(shape=(self.A, self.K), dtype=self.dtype)
        PT = np.zeros(shape=(self.A, self.K), dtype=self.dtype)
        QT = np.zeros(shape=(self.A, self.M), dtype=self.dtype)
        RT = np.zeros(shape=(self.A, self.K), dtype=self.dtype)
        W = WT.T
        P = PT.T
        Q = QT.T
        R = RT.T

        if self.algorithm == 1:
            validation_size = validation_indices.size
            TT = np.zeros(shape=(self.A, self.N - validation_size), dtype=self.dtype)
            T = TT.T
        else:
            T = None

        # If algorithm is 1, extract training set X
        if self.algorithm == 1:
            training_indices = np.setdiff1d(
                self.all_indices, validation_indices, assume_unique=True
            )
            training_X = self.cvm.X[training_indices]
            training_sqrt_w = (
                self.sqrt_weights[training_indices]
                if self.sqrt_weights is not None
                else None
            )
            result = self.cvm.training_XTY(validation_indices)
            training_XTY = result[0]
            training_X_mean, training_X_std, training_Y_mean, training_Y_std = result[1]
            if self.center_X:
                training_X = training_X - training_X_mean
            if self.scale_X:
                training_X = training_X / training_X_std
            if training_sqrt_w is not None:
                training_X = training_X * training_sqrt_w

        else:
            result = self.cvm.training_XTX_XTY(validation_indices)
            training_XTX, training_XTY = result[0]
            training_X_mean, training_X_std, training_Y_mean, training_Y_std = result[1]

        # Execute Improved Kernel PLS steps 2-5
        for i in range(self.A):
            # Step 2
            if self.M == 1:
                norm = la.norm(training_XTY, ord=2)
                if np.isclose(norm, 0, atol=self.eps, rtol=0):
                    self._weight_warning(i)
                    break
                w = training_XTY / norm
            else:
                if self.M < self.K:
                    training_XTYTtraining_XTY = training_XTY.T @ training_XTY
                    eig_vals, eig_vecs = la.eigh(training_XTYTtraining_XTY)
                    q = eig_vecs[:, -1:]
                    w = training_XTY @ q
                    norm = la.norm(w)
                    if np.isclose(norm, 0, atol=self.eps, rtol=0):
                        self._weight_warning(i)
                        break
                    w = w / norm
                else:
                    training_XTYYTX = training_XTY @ training_XTY.T
                    eig_vals, eig_vecs = la.eigh(training_XTYYTX)
                    norm = eig_vals[-1]
                    if np.isclose(norm, 0, atol=self.eps, rtol=0):
                        self._weight_warning(i)
                        break
                    w = eig_vecs[:, -1:]
            WT[i] = w.squeeze()

            # Step 3
            r = np.copy(w)
            for j in range(i):
                r = r - PT[j].reshape(-1, 1).T @ w * RT[j].reshape(-1, 1)
            RT[i] = r.squeeze()

            # Step 4
            if self.algorithm == 1:
                t = training_X @ r
                TT[i] = t.squeeze()
                tTt = t.T @ t
                p = (t.T @ training_X).T / tTt
            elif self.algorithm == 2:
                rtraining_XTX = r.T @ training_XTX
                tTt = rtraining_XTX @ r
                p = rtraining_XTX.T / tTt
            q = (r.T @ training_XTY).T / tTt
            PT[i] = p.squeeze()
            QT[i] = q.squeeze()

            # Step 5
            training_XTY = training_XTY - (p @ q.T) * tTt

            # Compute regression coefficients
            B[i] = B[i - 1] + r @ q.T

        # Return PLS matrices and training set statistics
        return (
            B,
            W,
            P,
            Q,
            R,
            T,
            training_X_mean,
            training_Y_mean,
            training_X_std,
            training_Y_std,
        )

    def _stateless_predict(
        self,
        indices: npt.NDArray[np.int_],
        B: npt.NDArray[np.floating],
        training_X_mean: Optional[npt.NDArray[np.floating]],
        training_Y_mean: Optional[npt.NDArray[np.floating]],
        training_X_std: Optional[npt.NDArray[np.floating]],
        training_Y_std: Optional[npt.NDArray[np.floating]],
        n_components: Optional[int] = None,
    ) -> npt.NDArray[np.floating]:
        """
        Predicts with Improved Kernel PLS Algorithm #1 on `X` with `B` using
        `n_components` components. If `n_components` is None, then predictions are
        returned for all number of components.

        Parameters
        ----------
        indices : Array of shape (N_val,)
            Integer array defining indices into X and Y corresponding to samples on
            which to predict.

        B : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        training_X_mean : None or Array of shape (1, K)
            Mean row of training X.

        training_Y_mean : None or Array of shape (1, M)
            Mean row of training Y.

        training_X_std : None or Array of shape (1, K)
            Sample standard deviation row of training X.

        training_Y_std : None or Array of shape (1, M)
            Sample standard deviation row of training Y.

        n_components : int or None, optional
            Number of components in the PLS model. If None, then all number of
            components are used.

        Returns
        -------
        Y_pred : Array of shape (N_pred, M) or (A, N_pred, M)
            If `n_components` is an int, then an array of shape (N_pred, M) with the
            predictions for that specific number of components is used. If
            `n_components` is None, returns a prediction for each number of components
            up to `A`.

        Notes
        -----
        """

        predictor_variables = self.cvm.X[indices]
        # Apply the potential training set centering and scaling
        if self.center_X:
            predictor_variables = predictor_variables - training_X_mean
        if self.scale_X:
            predictor_variables = predictor_variables / training_X_std
        if n_components is None:
            Y_pred = predictor_variables @ B
        else:
            Y_pred = predictor_variables @ B[n_components - 1]
        # Multiply by the potential training set scale and add the potential training
        # set bias
        if self.scale_Y:
            Y_pred = Y_pred * training_Y_std
        if self.center_Y:
            Y_pred = Y_pred + training_Y_mean
        return Y_pred

    def _stateless_fit_predict_eval(
        self,
        validation_indices: npt.NDArray[np.int_],
        metric_function: Callable[
            [
                npt.NDArray[np.floating],
                npt.NDArray[np.floating],
                Optional[npt.NDArray[np.floating]],
            ],
            Any,
        ],
    ) -> Any:
        """
        Fits Improved Kernel PLS Algorithm #1 or #2 on `X` or `XTX`, `XTY` and `Y`
        using `A` components, predicts on `X` and evaluates predictions using
        `metric_function`. The fit is performed on the training set defined by
        all samples not in `validation_indices`. The prediction is performed on the
        validation set defined by all samples in `validation_indices`.

        Parameters
        ----------
        validation_indices : Array of shape (N_val,)
            Integer array defining indices into X and Y corresponding to validation
            samples.

        metric_function : Callable receiving arrays `Y_true` and `Y_pred`, and, if
        `self.weights` is not None, also `weights`, and returning Any.

        Returns
        -------
        metric : Any
            The result of evaluating `metric_function` on the validation set.
        """
        matrices = self._stateless_fit(validation_indices)
        B = matrices[0]
        training_X_mean = matrices[-4]
        training_Y_mean = matrices[-3]
        training_X_std = matrices[-2]
        training_Y_std = matrices[-1]
        Y_pred = self._stateless_predict(
            validation_indices,
            B,
            training_X_mean,
            training_Y_mean,
            training_X_std,
            training_Y_std,
        )
        Y_true = self.cvm.Y[validation_indices]
        if self.cvm.weights is None:
            return metric_function(Y_true, Y_pred)
        weights = self.cvm.weights[validation_indices].flatten()
        return metric_function(Y_true, Y_pred, weights)

    def cross_validate(
        self,
        X: npt.ArrayLike,
        Y: npt.ArrayLike,
        A: int,
        folds: Iterable[Hashable],
        metric_function: Callable[[npt.ArrayLike, npt.ArrayLike], Any],
        weights: Optional[npt.ArrayLike] = None,
        n_jobs=-1,
        verbose=10,
    ) -> dict[Hashable, Any]:
        """
        Cross-validates the PLS model using `folds` splits on `X` and `Y` with
        `n_components` components evaluating results with `metric_function`.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M) or (N,)
            Target variables.

        A : int
            Number of components in the PLS model.

        folds : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `folds` corresponds to a different fold.

        metric_function : Callable receiving arrays `Y_val`, `Y_pred`, and, if
        `weights` is not None, also, `weights_val`, and returning Any.
            Computes a metric based on true values `Y_val` and predicted values
            `Y_pred`. `Y_pred` contains a prediction for all `A` components.

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
        The order of cross-validation folds is determined by the order of the unique
        values in `folds`. The keys and values of `metrics` will be sorted in the
        same order.
        """

        X = np.asarray(X, dtype=self.dtype)
        Y = np.asarray(Y, dtype=self.dtype)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if weights is not None:
            weights = np.asarray(weights, dtype=self.dtype)
            if weights.ndim == 1:
                weights = weights.reshape(-1, 1)
            if self.algorithm == 1:
                self.sqrt_weights = np.sqrt(weights)
        else:
            self.sqrt_weights = None

        self.cvm = CVMatrix(
            center_X=self.center_X,
            center_Y=self.center_Y,
            scale_X=self.scale_X,
            scale_Y=self.scale_Y,
            ddof=self.ddof,
            dtype=self.dtype,
            copy=False,
        )

        # It is important that Partitioner is not made an attribute of this
        # class, as it will then be pickled and sent to each worker in parallel jobs
        # which will cause immense performance degradation.
        p = Partitioner(folds)
        num_splits = len(p.folds_dict)

        if n_jobs == -1:
            n_jobs = min(joblib.cpu_count(), num_splits)

        print(
            f"Cross-validating Improved Kernel PLS Algorithm {self.algorithm} with {A}"
            f" components on {num_splits} unique splits using {n_jobs} "
            f"parallel processes."
        )

        self.cvm.fit(X, Y, weights)
        self.A = A
        self.N, self.K = self.cvm.X.shape
        self.M = self.cvm.Y.shape[1]

        if self.algorithm == 1:
            self.all_indices = np.arange(self.cvm.N, dtype=int)
            if weights is not None:
                self.sqrt_weights = np.sqrt(self.cvm.weights)

        def worker(
            validation_indices: npt.NDArray[np.int_],
            metric_function: Callable[[npt.ArrayLike, npt.ArrayLike], Any],
        ) -> Any:
            return self._stateless_fit_predict_eval(validation_indices, metric_function)

        metrics_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(worker)(p.get_validation_indices(fold), metric_function)
            for fold in p.folds_dict
        )
        metrics_dict = dict(zip(p.folds_dict, metrics_list))
        return metrics_dict
