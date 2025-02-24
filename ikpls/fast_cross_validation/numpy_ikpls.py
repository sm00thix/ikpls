"""
Contains the PLS class which implements fast cross-validation with partial
least-squares regression using Improved Kernel PLS by Dayal and MacGregor:
https://arxiv.org/abs/2401.13185
https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

The implementation is written using NumPy and allows for parallelization of the
cross-validation process using joblib.

Author: Ole-Christian Galbo Engstrøm
E-mail: ole.e@di.ku.dk
"""

import warnings
from collections.abc import Callable, Hashable
from typing import Any, Iterable, Union

import joblib
import numpy as np
import numpy.linalg as la
import numpy.typing as npt
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
        Whether to use Improved Kernel PLS Algorithm #1 or #2.

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
        column-wise standard deviations. Bessel's correction for the unbiased estimate
        of the sample standard deviation is used. The row of column-wise standard
        deviations is computed on the training set for each fold to avoid data leakage.

    scale_Y : bool, optional default=True
        Whether to scale `Y` before fitting by dividing each row with the row of `X`'s
        column-wise standard deviations. Bessel's correction for the unbiased estimate
        of the sample standard deviation is used. The row of column-wise standard
        deviations is computed on the training set for each fold to avoid data leakage.

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

    Setting either of `center_X`, `center_Y`, `scale_X`, or `scale_Y` to True, while
    using multiple jobs, will increase the memory consumption as each job will then
    have to keep its own copy of :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
    :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` with its specific centering and scaling.
    """

    def __init__(
        self,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        algorithm: int = 1,
        dtype: np.floating = np.float64,
    ) -> None:
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.algorithm = algorithm
        self.dtype = dtype
        self.eps = np.finfo(dtype).eps
        self.name = f"Improved Kernel PLS Algorithm #{algorithm}"
        if self.algorithm not in [1, 2]:
            raise ValueError(
                f"Invalid algorithm: {self.algorithm}. Algorithm must be 1 or 2."
            )
        self.X = None
        self.Y = None
        self.A = None
        self.N = None
        self.K = None
        self.M = None
        self.X_mean = None
        self.Y_mean = None
        self.sum_X = None
        self.sum_Y = None
        self.sum_sq_X = None
        self.sum_sq_Y = None
        self.XTX = None
        self.XTY = None
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
    ) -> Union[
        tuple[
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
        ],
        tuple[
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
        ],
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

        T : Array of shape (A, N_train)
            PLS scores matrix of X. Only Returned for Improved Kernel PLS Algorithm #1.

        training_X_mean : Array of shape (1, K)
            Mean row of training X. Will be an array of zeros if `self.center` is
            False.

        training_Y_mean : Array of shape (1, M)
            Mean row of training Y. Will be an array of zeros if `self.center` is
            False.

        training_X_std : Array of shape (1, K)
            Sample standard deviation row of training X. Will be an array of ones if
            `self.scale` is False. Any zero standard deviations will be replaced with
            ones.

        training_Y_std : Array of shape (1, M)
            Sample standard deviation row of training Y. Will be an array of ones if
            `self.scale` is False. Any zero standard deviations will be replaced with
            ones.

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

        validation_size = validation_indices.size
        if self.algorithm == 1:
            TT = np.zeros(shape=(self.A, self.N - validation_size), dtype=self.dtype)
            T = TT.T

        # Extract training X and Y
        validation_X = self.X[validation_indices]
        validation_Y = self.Y[validation_indices]

        if self.center_X or self.center_Y or self.scale_X or self.scale_Y:
            training_size = self.N - validation_size
            N_total_over_N_train = self.N / training_size
            N_val_over_N_train = validation_size / training_size

        # Compute the training set means
        if self.center_X or self.center_Y or self.scale_X:
            training_X_mean = (
                N_total_over_N_train * self.X_mean
                - N_val_over_N_train * np.mean(validation_X, axis=0, keepdims=True)
            )
        if self.center_X or self.center_Y or self.scale_Y:
            training_Y_mean = (
                N_total_over_N_train * self.Y_mean
                - N_val_over_N_train * np.mean(validation_Y, axis=0, keepdims=True)
            )

        # Compute the training set standard deviations for X
        if self.scale_X:
            train_sum_X = self.sum_X - np.expand_dims(
                np.einsum("ij -> j", validation_X), axis=0
            )
            train_sum_sq_X = self.sum_sq_X - np.expand_dims(
                np.einsum("ij,ij -> j", validation_X, validation_X), axis=0
            )
            training_X_std = np.sqrt(
                1
                / (training_size - 1)
                * (
                    -2 * training_X_mean * train_sum_X
                    + training_size
                    * np.einsum("ij,ij -> ij", training_X_mean, training_X_mean)
                    + train_sum_sq_X
                )
            )
            training_X_std[np.abs(training_X_std) <= self.eps] = 1

        # Compute the training set standard deviations for Y
        if self.scale_Y:
            train_sum_Y = self.sum_Y - np.expand_dims(
                np.einsum("ij -> j", validation_Y), axis=0
            )
            train_sum_sq_Y = self.sum_sq_Y - np.expand_dims(
                np.einsum("ij,ij -> j", validation_Y, validation_Y), axis=0
            )
            training_Y_std = np.sqrt(
                1
                / (training_size - 1)
                * (
                    -2 * training_Y_mean * train_sum_Y
                    + training_size
                    * np.einsum("ij,ij -> ij", training_Y_mean, training_Y_mean)
                    + train_sum_sq_Y
                )
            )
            training_Y_std[np.abs(training_Y_std) <= self.eps] = 1

        # Subtract the validation set's contribution from the total XTY
        training_XTY = self.XTY - validation_X.T @ validation_Y

        # Apply the training set centering
        if self.center_X or self.center_Y:
            # Apply the training set centering
            training_XTY = training_XTY - training_size * (
                training_X_mean.T @ training_Y_mean
            )

        # Apply the training set scaling
        if self.scale_X and self.scale_Y:
            divisor = training_X_std.T @ training_Y_std
        elif self.scale_X:
            divisor = training_X_std.T
        elif self.scale_Y:
            divisor = training_Y_std
        if self.scale_X or self.scale_Y:
            training_XTY = training_XTY / divisor

        # If algorithm is 1, extract training set X
        if self.algorithm == 1:
            training_indices = np.setdiff1d(
                self.all_indices, validation_indices, assume_unique=True
            )
            training_X = self.X[training_indices]
            if self.center_X:
                # Apply the training set centering
                training_X = training_X - training_X_mean
            if self.scale_X:
                # Apply the training set scaling
                training_X = training_X / training_X_std

        # If algorithm is 2, derive training set XTX from total XTX and validation XTX
        else:
            training_XTX = self.XTX - validation_X.T @ validation_X
            if self.center_X:
                # Apply the training set centering
                training_XTX = training_XTX - training_size * (
                    training_X_mean.T @ training_X_mean
                )
            if self.scale_X:
                # Apply the training set scaling
                training_XTX = training_XTX / (training_X_std.T @ training_X_std)

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

        # Use additive and multiplicative identities for means and standard deviations
        # for centering and scaling if they are not used
        if not self.center_X:
            training_X_mean = np.zeros((1, self.K))
        if not self.center_Y:
            training_Y_mean = np.zeros((1, self.M))
        if not self.scale_X:
            training_X_std = np.ones((1, self.K))
        if not self.scale_Y:
            training_Y_std = np.ones((1, self.M))

        # Return PLS matrices and training set statistics
        if self.algorithm == 1:
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
        return (
            B,
            W,
            P,
            Q,
            R,
            training_X_mean,
            training_Y_mean,
            training_X_std,
            training_Y_std,
        )

    def _stateless_predict(
        self,
        indices: npt.NDArray[np.int_],
        B: npt.NDArray[np.floating],
        training_X_mean: npt.NDArray[np.floating],
        training_Y_mean: npt.NDArray[np.floating],
        training_X_std: npt.NDArray[np.floating],
        training_Y_std: npt.NDArray[np.floating],
        n_components: Union[None, int] = None,
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

        training_X_mean : Array of shape (1, K)
            Mean row of training X. If self.center_X is False, then this should be an
            array of zeros.

        training_Y_mean : Array of shape (1, M)
            Mean row of training Y. If self.center_Y is False, then this should be an
            array of zeros.

        training_X_std : Array of shape (1, K)
            Sample standard deviation row of training X. If self.scale_X is False, then
            this should be an array of ones. Any zero standard deviations should be
            replaced with ones.

        training_Y_std : Array of shape (1, M)
            Sample standard deviation row of training Y. If self.scale_Y is False, then
            this should be an array of ones. Any zero standard deviations should be
            replaced with ones.

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

        predictor_variables = self.X[indices]
        # Apply the potential training set centering and scaling
        predictor_variables = (predictor_variables - training_X_mean) / training_X_std
        if n_components is None:
            Y_pred = predictor_variables @ B
        else:
            Y_pred = predictor_variables @ B[n_components - 1]
        # Multiply by the potential training set scale and add the potential training
        # set bias
        return Y_pred * training_Y_std + training_Y_mean

    def _stateless_fit_predict_eval(
        self,
        validation_indices: npt.NDArray[np.int_],
        metric_function: Callable[
            [npt.NDArray[np.floating], npt.NDArray[np.floating]], Any
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

        metric_function : Callable receiving arrays `Y_true` and `Y_pred` and returning
        Any

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
        return metric_function(self.Y[validation_indices], Y_pred)

    def _init_folds_dict(
        self, cv_splits: Iterable[Hashable]
    ) -> dict[Hashable, npt.NDArray[np.int_]]:
        """
        Generates a list of validation indices for each fold in `cv_splits`.

        Parameters
        ----------
        cv_splits : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `cv_splits` corresponds to a different fold.

        Returns
        -------
        index_dict : dict of Hashable to Array
            A dictionary mapping each unique value in `cv_splits` to an array of
            validation indices.
        """
        index_dict = {}
        for i, num in enumerate(cv_splits):
            try:
                index_dict[num].append(i)
            except KeyError:
                index_dict[num] = [i]
        for key in index_dict:
            index_dict[key] = np.asarray(index_dict[key], dtype=int)
        return index_dict

    def cross_validate(
        self,
        X: npt.ArrayLike,
        Y: npt.ArrayLike,
        A: int,
        cv_splits: Iterable[Hashable],
        metric_function: Callable[[npt.ArrayLike, npt.ArrayLike], Any],
        n_jobs=-1,
        verbose=10,
    ) -> dict[Hashable, Any]:
        """
        Cross-validates the PLS model using `cv_splits` splits on `X` and `Y` with
        `n_components` components evaluating results with `metric_function`.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M) or (N,)
            Target variables.

        A : int
            Number of components in the PLS model.

        cv_splits : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `cv_splits` corresponds to a different fold.

        metric_function : callable
            A callable receiving arrays `Y_true` of shape (N_val, M) and `Y_pred` of
            shape (A, N_val, M) and returning Any. Computes a metric based on true
            values `Y_true` and predicted values `Y_pred`. `Y_pred` contains a
            prediction for all `A` components.

        n_jobs : int, optional default=-1
            Number of parallel jobs to use. A value of -1 will use the minimum of all
            available cores and the number of unique values in `cv_splits`.

        verbose : int, optional default=10
            Controls verbosity of parallel jobs.

        Returns
        -------
        metrics : dict of Hashable to Any
            A dictionary mapping each unique value in `cv_splits` to the result of
            evaluating `metric_function` on the validation set corresponding to that
            value.

        Notes
        -----
        The order of cross-validation folds is determined by the order of the unique
        values in `cv_splits`. The keys and values of `metrics` will be sorted in the
        same order.
        """

        self.X = np.asarray(X, dtype=self.dtype)
        self.Y = np.asarray(Y, dtype=self.dtype)
        if self.Y.ndim == 1:
            self.Y = self.Y.reshape(-1, 1)
        self.A = A
        self.N, self.K = X.shape
        self.M = self.Y.shape[1]
        folds_dict = self._init_folds_dict(cv_splits)
        num_splits = len(folds_dict)

        if self.algorithm == 1:
            self.all_indices = np.arange(self.N, dtype=int)

        if n_jobs == -1:
            n_jobs = min(joblib.cpu_count(), num_splits)

        print(
            f"Cross-validating Improved Kernel PLS Algorithm {self.algorithm} with {A}"
            f" components on {num_splits} unique splits using {n_jobs} "
            f"parallel processes."
        )

        # We can compute these once for the entire dataset and subtract the
        # validation parts during cross-validation.
        if self.center_X or self.center_Y or self.scale_X:
            self.X_mean = np.mean(self.X, axis=0, keepdims=True)
        if self.center_X or self.center_Y or self.scale_Y:
            self.Y_mean = np.mean(self.Y, axis=0, keepdims=True)
        if self.scale_X:
            self.sum_X = np.expand_dims(np.einsum("ij->j", self.X), axis=0)
            self.sum_sq_X = np.expand_dims(
                np.einsum("ij,ij->j", self.X, self.X), axis=0
            )
        if self.scale_Y:
            self.sum_Y = np.expand_dims(np.einsum("ij->j", self.Y), axis=0)
            self.sum_sq_Y = np.expand_dims(
                np.einsum("ij,ij->j", self.Y, self.Y), axis=0
            )

        if verbose > 0:
            print("Computing total XTY...")
        self.XTY = self.X.T @ self.Y
        if verbose > 0:
            print("Done!")
        if self.algorithm == 2:
            if verbose > 0:
                print("Computing total XTX...")
            self.XTX = self.X.T @ self.X
            if verbose > 0:
                print("Done!")

        def worker(
            validation_indices: npt.NDArray[np.int_],
            metric_function: Callable[[npt.ArrayLike, npt.ArrayLike], Any],
        ) -> Any:
            return self._stateless_fit_predict_eval(validation_indices, metric_function)

        metrics_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(worker)(validation_indices, metric_function)
            for validation_indices in folds_dict.values()
        )
        metrics_dict = dict(zip(folds_dict.keys(), metrics_list))
        return metrics_dict
