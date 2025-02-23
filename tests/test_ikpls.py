"""
This file contains tests for the ikpls package. Some of the tests are taken directly
from the test-suite for the scikit-learn PLSRegression class. Most of the tests are
written for this work specifically.

The tests are designed to check the consistency of the PLS algorithm implementations
across different shapes of input data. The tests also check for consistency between
the IKPLS implementations in the ikpls package and the scikit-learn NIPALS
implementation.

Author: Ole-Christian Galbo Engstrøm
E-mail: ole.e@di.ku.dk
"""

from collections.abc import Callable
from itertools import product
from typing import List, Optional, Tuple, Union

import jax
import numpy as np
import numpy.typing as npt
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose
from sklearn.cross_decomposition import PLSRegression as SkPLS
from sklearn.datasets import load_linnerud
from sklearn.model_selection import cross_validate

from ikpls.fast_cross_validation.numpy_ikpls import PLS as FastCVPLS
from ikpls.jax_ikpls_alg_1 import PLS as JAX_Alg_1
from ikpls.jax_ikpls_alg_2 import PLS as JAX_Alg_2
from ikpls.numpy_ikpls import PLS as NpPLS

from . import load_data

# Allow JAX to use 64-bit floating point precision.
jax.config.update("jax_enable_x64", True)

# Set JAX default device to CPU.
jax.default_device(jax.devices("cpu")[0])


# Warning raised due to MathJax in some docstrins in the FastCVPLS class.
@pytest.mark.filterwarnings(
    "ignore", category=DeprecationWarning, match="Invalid escape sequence"
)
@pytest.mark.filterwarnings("ignore")
class TestClass:
    """
    Class for testing the IKPLS implementation.

    This class contains methods for testing the IKPLS implementation.

    Attributes
    ----------
    csv : DataFrame
        The CSV data containing target values.

    raw_spectra : npt.NDArray[np.float64]
        The raw spectral data.
    """

    csv = load_data.load_csv()
    raw_spectra = load_data.load_spectra()
    seed = 42
    rng = np.random.default_rng(seed)

    def load_X(self) -> npt.NDArray[np.float64]:
        """
        Description
        -----------
        Load the raw spectral data.

        Returns
        -------
        npt.NDArray[np.float64]
            The raw spectral data.
        """
        return np.copy(self.raw_spectra)

    def load_Y(self, values: list[str]) -> npt.NDArray[np.float64]:
        """
        Description
        -----------
        Load target values based on the specified column names.

        Parameters
        ----------
        values : list[str]
            List of column names to extract target values from the CSV data.

        Returns
        -------
        npt.NDArray[np.float64]
            Target values as a NumPy array.
        """
        target_values = self.csv[values].to_numpy()
        return target_values

    def load_weights(self, kind: Union[None, float]) -> npt.NDArray[np.float64]:
        """
        Description
        -----------
        Load weights for the spectral data.

        Parameters
        ----------
        kind : Union[None, float]
            The kind of weights to generate. If None, random weights are generated.
            Otherwise, constant weights are generated based on the specified value.

        Returns
        -------
        npt.NDArray[np.float64]
            Weights for the spectral data.
        """
        if kind is None:
            weights = self.rng.uniform(low=0, high=3, size=self.raw_spectra.shape[0])
            # reset the rng
            self.rng = np.random.default_rng(self.seed)
        else:
            weights = np.full(self.raw_spectra.shape[0], kind, dtype=np.float64)
        return weights

    def get_model_preds(
        self,
        X: np.ndarray,
        models: List[Union[NpPLS, JAX_Alg_1, JAX_Alg_2]],
        last_only=False,
    ):
        """
        Description
        -----------
        Get predictions from PLS models.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        models : List of NpPLS, JAX_Alg_1, JAX_Alg_2 objects.
            List of PLS models.

        last_only : bool
            Whether to return predictions for the last component only.

        Returns
        -------
        List[np.ndarray], List[np.ndarray]
            Predictions and regression matrices.
        """
        preds = []
        for model in models:
            if last_only and model.max_stable_components is None:
                n_components = model.A
            else:
                n_components = model.max_stable_components if last_only else None
            preds.append(model.predict(X, n_components=n_components))
        Bs = [m.B for m in models]
        if last_only:
            Bs = [
                (
                    B[model.max_stable_components - 1]
                    if model.max_stable_components is not None
                    else B[-1]
                )
                for B, model in zip(Bs, models)
            ]
        return preds, Bs

    def fit_predict_wls(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        weights: np.ndarray,
        center_X: bool,
        center_Y: bool,
        scale_X: bool,
        scale_Y: bool,
    ):
        """
        Description
        -----------
        Fit and predict using ordinary least squares regression.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Y : np.ndarray
            Target values.

        center_X : bool
            Center the input data.

        center_Y : bool
            Center the target values.

        scale_X : bool
            Scale the input data.

        scale_Y : bool
            Scale the target values.

        Returns
        -------
        np.ndarray, np.ndarray
            Predicted target values and regression matrix
        """
        weights_matrix = np.diag(np.sqrt(weights))
        if center_X:
            X_mean = np.average(X, axis=0, weights=weights)
            X = X - X_mean
        if center_Y:
            Y_mean = np.average(Y, axis=0, weights=weights)
            Y = Y - Y_mean
        if scale_X or scale_Y:
            num_nonzero_weights = np.asarray(
                np.count_nonzero(weights), dtype=np.float64
            )
            avg_non_zero_weights = np.sum(weights) / num_nonzero_weights
            scale_dof = num_nonzero_weights - 1
        if scale_X:
            new_X_mean = 0 if center_X else np.average(X, axis=0, weights=weights)
            X_std = np.sqrt(
                np.sum(
                    np.reshape(weights, (-1, 1)) * (X - new_X_mean) ** 2,
                    axis=0,
                    keepdims=True,
                )
                / (scale_dof * avg_non_zero_weights)
            )
            X_std[np.abs(X_std) <= np.finfo(np.float64).eps] = 1
            X = X / X_std
        if scale_Y:
            new_Y_mean = 0 if center_Y else np.average(Y, axis=0, weights=weights)
            Y_std = np.sqrt(
                np.sum(
                    np.reshape(weights, (-1, 1)) * (Y - new_Y_mean) ** 2,
                    axis=0,
                    keepdims=True,
                )
                / (scale_dof * avg_non_zero_weights)
            )
            Y_std[np.abs(Y_std) <= np.finfo(np.float64).eps] = 1
            Y = Y / Y_std
        XW = weights_matrix @ X
        YW = weights_matrix @ Y
        wls_B = np.linalg.lstsq(XW, YW)[0]
        wls_pred = X @ wls_B
        wls_pred = wls_pred
        if scale_Y:
            wls_pred *= Y_std
        if center_Y:
            wls_pred += Y_mean
        return wls_pred, wls_B

    def fit_predict_ols(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        center_X: bool,
        center_Y: bool,
        scale_X: bool,
        scale_Y: bool,
    ):
        """
        Description
        -----------
        Fit and predict using ordinary least squares regression.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Y : np.ndarray
            Target values.

        center_X : bool
            Center the input data.

        center_Y : bool
            Center the target values.

        scale_X : bool
            Scale the input data.

        scale_Y : bool
            Scale the target values.

        Returns
        -------
        np.ndarray, np.ndarray
            Predicted target values and regression matrix
        """
        if center_X:
            X_mean = np.mean(X, axis=0)
            X = X - X_mean
        if center_Y:
            Y_mean = np.mean(Y, axis=0)
            Y = Y - Y_mean
        if scale_X:
            X_std = np.std(X, axis=0, ddof=1)
            X_std[X_std <= np.finfo(np.float64).eps] = 1
            X = X / X_std
        if scale_Y:
            Y_std = np.std(Y, axis=0, ddof=1)
            Y_std[Y_std <= np.finfo(np.float64).eps] = 1
            Y = Y / Y_std
        ols_B = np.linalg.lstsq(X, Y)[0]
        ols_pred = X @ ols_B
        if scale_Y:
            ols_pred *= Y_std
        if center_Y:
            ols_pred += Y_mean
        return ols_pred, ols_B

    def get_models(
        self,
        center_X: bool,
        center_Y: bool,
        scale_X: bool,
        scale_Y: bool,
        fast_cv: bool,
    ):
        """
        Description
        -----------
        Get PLS models with different preprocessing settings.

        Parameters
        ----------
        center_X : bool
            Center the input data.

        center_Y : bool
            Center the target values.

        scale_X : bool
            Scale the input data.

        scale_Y : bool
            Scale the target values.

        fast_cv : bool
            Whether to return the fast cross-validation PLS model as well.

        Returns
        -------
        Tuple[NpPLS, NpPLS, JAX_Alg_1, JAX_Alg_2, JAX_Alg_1, JAX_Alg_2] or
        Tuple[NpPLS, NpPLS, JAX_Alg_1, JAX_Alg_2, JAX_Alg_1, JAX_Alg_2, FastCVPLS, FastCVPLS]
            PLS models with different preprocessing settings.
        """

        np_pls_alg_1 = NpPLS(
            algorithm=1,
            center_X=center_X,
            center_Y=center_Y,
            scale_X=scale_X,
            scale_Y=scale_Y,
        )
        np_pls_alg_2 = NpPLS(
            algorithm=2,
            center_X=center_X,
            center_Y=center_Y,
            scale_X=scale_X,
            scale_Y=scale_Y,
        )
        jax_pls_alg_1 = JAX_Alg_1(
            center_X=center_X,
            center_Y=center_Y,
            scale_X=scale_X,
            scale_Y=scale_Y,
            differentiable=False,
            verbose=True,
        )
        jax_pls_alg_2 = JAX_Alg_2(
            center_X=center_X,
            center_Y=center_Y,
            scale_X=scale_X,
            scale_Y=scale_Y,
            differentiable=False,
            verbose=True,
        )
        diff_jax_pls_alg_1 = JAX_Alg_1(
            center_X=center_X,
            center_Y=center_Y,
            scale_X=scale_X,
            scale_Y=scale_Y,
            differentiable=True,
            verbose=True,
        )
        diff_jax_pls_alg_2 = JAX_Alg_2(
            center_X=center_X,
            center_Y=center_Y,
            scale_X=scale_X,
            scale_Y=scale_Y,
            differentiable=True,
            verbose=True,
        )

        if fast_cv:
            fast_cv_pls_alg_1 = FastCVPLS(
                algorithm=1,
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
            )
            fast_cv_pls_alg_2 = FastCVPLS(
                algorithm=2,
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
            )
            return (
                np_pls_alg_1,
                np_pls_alg_2,
                jax_pls_alg_1,
                jax_pls_alg_2,
                diff_jax_pls_alg_1,
                diff_jax_pls_alg_2,
                fast_cv_pls_alg_1,
                fast_cv_pls_alg_2,
            )

        return (
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
            diff_jax_pls_alg_1,
            diff_jax_pls_alg_2,
        )

    def fit_models(
        self,
        X: npt.NDArray,
        Y: npt.NDArray,
        n_components: int,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        weights: Optional[npt.NDArray] = None,
        return_sk_pls: Optional[bool] = True,
    ) -> Union[
        Tuple[
            SkPLS, npt.NDArray, NpPLS, NpPLS, JAX_Alg_1, JAX_Alg_2, JAX_Alg_1, JAX_Alg_2
        ],
        Tuple[NpPLS, NpPLS, JAX_Alg_1, JAX_Alg_2, JAX_Alg_1, JAX_Alg_2],
    ]:
        """
        Description
        -----------

        Fit various PLS models using different implementations and preprocessing.

        Parameters
        ----------
        X : NDArray[float]
            Input data (spectra).

        Y : NDArray[float]
            Target values.

        n_components : int
            Number of PLS components.

        weights : NDArray[float], optional
            Weights for the input data

        Returns
        -------
        tuple
            A tuple containing PLS models and their respective regression matrices.
        """
        jnp_X = jnp.asarray(X)
        jnp_Y = jnp.asarray(Y)

        (
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
            diff_jax_pls_alg_1,
            diff_jax_pls_alg_2,
        ) = self.get_models(
            center_X=center_X,
            center_Y=center_Y,
            scale_X=scale_X,
            scale_Y=scale_Y,
            fast_cv=False,
        )

        np_pls_alg_1.fit(X=X, Y=Y, A=n_components, weights=weights)
        np_pls_alg_2.fit(X=X, Y=Y, A=n_components, weights=weights)
        jax_pls_alg_1.fit(X=jnp_X, Y=jnp_Y, A=n_components, weights=weights)
        jax_pls_alg_2.fit(X=jnp_X, Y=jnp_Y, A=n_components, weights=weights)
        diff_jax_pls_alg_1.fit(X=jnp_X, Y=jnp_Y, A=n_components, weights=weights)
        diff_jax_pls_alg_2.fit(X=jnp_X, Y=jnp_Y, A=n_components, weights=weights)

        if weights is None and return_sk_pls:
            sk_pls = SkPLS(n_components=n_components)
            sk_pls.fit(X=X, y=Y)
            # Reconstruct SkPLS regression matrix for all components
            sk_B = np.empty(np_pls_alg_1.B.shape)
            for i in range(sk_B.shape[0]):
                sk_B_at_component_i = np.dot(
                    sk_pls.x_rotations_[..., : i + 1],
                    sk_pls.y_loadings_[..., : i + 1].T,
                )
                sk_B[i] = sk_B_at_component_i
            return (
                sk_pls,
                sk_B,
                np_pls_alg_1,
                np_pls_alg_2,
                jax_pls_alg_1,
                jax_pls_alg_2,
                diff_jax_pls_alg_1,
                diff_jax_pls_alg_2,
            )
        return (
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
            diff_jax_pls_alg_1,
            diff_jax_pls_alg_2,
        )

    @staticmethod
    def jax_rmse_per_component(
        Y_true: jnp.ndarray, Y_pred: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        """
        Description
        -----------
        Compute the root mean squared error (RMSE) for each component using JAX.

        Parameters
        ----------
        Y_true : Array of shape (N, M) or (N,)
            True target values.

        Y_pred : Array of shape (A, N, M)
            Predicted target values.

        Returns
        -------
        Array of shape (A, M)
            RMSE for each component.
        """
        if Y_true.ndim == 1:
            Y_true = Y_true.reshape(-1, 1)
        e = Y_true - Y_pred
        se = e**2
        mse = jnp.mean(se, axis=-2)
        rmse = jnp.sqrt(mse)
        return rmse

    @staticmethod
    def rmse_per_component(Y_true: npt.NDArray, Y_pred: npt.NDArray) -> npt.NDArray:
        """
        Description
        -----------
        Compute the root mean squared error (RMSE) for each component.

        Parameters
        ----------
        Y_true : Array of shape (N, M) or (N,)
            True target values.

        Y_pred : Array of shape (A, N, M)
            Predicted target values.

        Returns
        -------
        Array of shape (A, M)
            RMSE for each component.
        """
        if Y_true.ndim == 1:
            Y_true = Y_true.reshape(-1, 1)
        e = Y_true - Y_pred
        se = e**2
        mse = np.mean(se, axis=-2)
        rmse = np.sqrt(mse)
        return rmse

    @staticmethod
    def cv_splitter(splits: npt.NDArray):
        """
        Description
        -----------
        Generate train and validation indices for cross-validation.

        Parameters
        ----------
        splits : Array of shape (N,)
            Split indices.

        Yields
        ------
        Tuple[npt.NDArray, npt.NDArray]
            Train and validation indices.
        """
        uniq_splits = np.unique(splits)
        for split in uniq_splits:
            train_idxs = np.nonzero(splits != split)[0]
            val_idxs = np.nonzero(splits == split)[0]
            yield train_idxs, val_idxs

    @staticmethod
    def snv(
        X_train: npt.NDArray, Y_train, X_val, Y_val, *args, **kwargs
    ) -> npt.NDArray:
        """
        Description
        -----------
        Perform standard normal variate (SNV) scaling on the input data using JAX.

        Parameters
        ----------
        X_train : Array of shape (N_train, K)
            Training data (spectra).

        Y_train : Array of shape (N_train, M)
            Training target values.

        X_val : Array of shape (N_val, K)
            Validation data (spectra).

        Y_val : Array of shape (N_val, M)
            Validation target values.

        *args
            Additional arguments. Unused.

        **kwargs
            Additional keyword arguments. Unused.

        Returns
        -------
        X_snv : Array of shape (N_train, K)
            SNV-scaled training data.

        Y_train : Array of shape (N_train, M)
            Unchanged training target values.

        X_val : Array of shape (N_val, K)
            SNV-scaled validation data.

        Y_val : Array of shape (N_val, M)
            Unchanged validation target values.
        """
        X_mean = np.mean(X_train, axis=1, keepdims=True)
        X_std = np.std(X_train, axis=1, ddof=1, mean=X_mean, keepdims=True)
        X_std[X_std <= np.finfo(np.float64).eps] = 1
        X_snv = (X_train - X_mean) / X_std
        return (
            X_snv,
            Y_train,
            X_val,
            Y_val,
        )

    @staticmethod
    def jax_snv(
        X_train: npt.NDArray, Y_train, X_val, Y_val, *args, **kwargs
    ) -> jnp.ndarray:
        """
        Description
        -----------
        Perform standard normal variate (SNV) scaling on the input data using JAX.

        Parameters
        ----------
        X_train : Array of shape (N_train, K)
            Training data (spectra).

        Y_train : Array of shape (N_train, M)
            Training target values.

        X_val : Array of shape (N_val, K)
            Validation data (spectra).

        Y_val : Array of shape (N_val, M)
            Validation target values.

        *args
            Additional arguments. Unused.

        **kwargs
            Additional keyword arguments. Unused.

        Returns
        -------
        X_snv : Array of shape (N_train, K)
            SNV-scaled training data.

        Y_train : Array of shape (N_train, M)
            Unchanged training target values.

        X_val : Array of shape (N_val, K)
            SNV-scaled validation data.

        Y_val : Array of shape (N_val, M)
            Unchanged validation target values.
        """
        X_mean = jnp.mean(X_train, axis=1, keepdims=True)
        X_std = jnp.std(X_train, axis=1, ddof=1, keepdims=True)
        X_std = jnp.where(X_std <= np.finfo(np.float64).eps, 1, X_std)
        X_snv = (X_train - X_mean) / X_std
        return X_snv, Y_train, X_val, Y_val

    @staticmethod
    def pred_metric(Y_val, Y_pred, *args, **kwargs):
        """
        Description
        -----------
        A metric function that simply returns the predicted values.
        """
        return Y_pred

    def assert_matrix_orthogonal(
        self, M: npt.NDArray, atol: float, rtol: float
    ) -> None:
        """
        Description
        -----------

        Check if a matrix is orthogonal.

        Parameters
        ----------
        M : NDArray[float]
            Matrix to check for orthogonality.

        atol : float
            Absolute tolerance for checking orthogonality.

        rtol : float
            Relative tolerance for checking orthogonality.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the matrix is not orthogonal within the specified tolerances.
        """
        MTM = np.dot(M.T, M)
        assert_allclose(MTM, np.diag(np.diag(MTM)), atol=atol, rtol=rtol)

    def check_regression_matrices(
        self,
        sk_B: npt.NDArray,
        np_pls_alg_1: NpPLS,
        np_pls_alg_2: NpPLS,
        jax_pls_alg_1: NpPLS,
        jax_pls_alg_2: NpPLS,
        diff_jax_pls_alg_1: JAX_Alg_1,
        diff_jax_pls_alg_2: JAX_Alg_2,
        atol: float,
        rtol: float,
        n_good_components: int = -1,
    ):
        """
        Description
        -----------

        Check regression matrices of different PLS models for consistency.

        Parameters
        ----------
        sk_B : NDArray[float]
            Sklearn PLS regression matrix.

        np_pls_alg_1
            Numpy-based PLS model using algorithm 1.

        np_pls_alg_2
            Numpy-based PLS model using algorithm 2.

        jax_pls_alg_1
            JAX-based PLS model using algorithm 1.

        jax_pls_alg_2
            JAX-based PLS model using algorithm 2.

        diff_jax_pls_alg_1
            JAX-based PLS model using algorithm 1 with differentiation.

        diff_jax_pls_alg_2
            JAX-based PLS model using algorithm 2 with differentiation.

        atol : float
            Absolute tolerance for checking equality.

        rtol : float
            Relative tolerance for checking equality.

        n_good_components : int, optional
            Number of components to check, or -1 to use all possible number of
            components.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the regression matrices are not consistent.
        """
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A
        assert_allclose(
            np_pls_alg_1.B[:n_good_components],
            sk_B[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np_pls_alg_2.B[:n_good_components],
            sk_B[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(jax_pls_alg_1.B)[:n_good_components],
            sk_B[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(jax_pls_alg_2.B)[:n_good_components],
            sk_B[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(diff_jax_pls_alg_1.B)[:n_good_components],
            sk_B[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(diff_jax_pls_alg_2.B)[:n_good_components],
            sk_B[:n_good_components],
            atol=atol,
            rtol=rtol,
        )

    def check_predictions(
        self,
        sk_pls: SkPLS,
        sk_B: npt.NDArray,
        np_pls_alg_1: NpPLS,
        np_pls_alg_2: NpPLS,
        jax_pls_alg_1: JAX_Alg_1,
        jax_pls_alg_2: JAX_Alg_2,
        diff_jax_pls_alg_1: JAX_Alg_1,
        diff_jax_pls_alg_2: JAX_Alg_2,
        X: npt.NDArray,
        atol: float,
        rtol: float,
        n_good_components: int = -1,
    ) -> None:
        """
        Description
        -----------
        Check predictions of different PLS models for consistency.

        Parameters
        ----------
        sk_pls : SkPLS
            Sklearn PLS model.

        sk_B : NDArray[float]
            Sklearn PLS regression matrix.

        np_pls_alg_1
            Numpy-based PLS model using algorithm 1.

        np_pls_alg_2
            Numpy-based PLS model using algorithm 2.

        jax_pls_alg_1
            JAX-based PLS model using algorithm 1.

        jax_pls_alg_2
            JAX-based PLS model using algorithm 2.

        diff_jax_pls_alg_1
            JAX-based PLS model using algorithm 1 with differentiation.

        diff_jax_pls_alg_2
            JAX-based PLS model using algorithm 2 with differentiation.

        X : NDArray[float]
            Input data (spectra) for making predictions.

        atol : float
            Absolute tolerance for checking equality.

        rtol : float
            Relative tolerance for checking equality.

        n_good_components : int, optional
            Number of components to check, or -1 to use all possible number of
            components.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the predictions are not consistent.
        """
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A
        sk_all_preds = (X - sk_pls._x_mean) / sk_pls._x_std @ (
            sk_B * sk_pls._y_std
        ) + sk_pls._y_mean
        assert_allclose(
            np_pls_alg_1.predict(X)[:n_good_components],
            sk_all_preds[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np_pls_alg_2.predict(X)[:n_good_components],
            sk_all_preds[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(jax_pls_alg_1.predict(X))[:n_good_components],
            sk_all_preds[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(jax_pls_alg_2.predict(X))[:n_good_components],
            sk_all_preds[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(diff_jax_pls_alg_1.predict(X))[:n_good_components],
            sk_all_preds[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(diff_jax_pls_alg_2.predict(X))[:n_good_components],
            sk_all_preds[:n_good_components],
            atol=atol,
            rtol=rtol,
        )

        # Check predictions using the largest good number of components.
        sk_final_pred = sk_all_preds[n_good_components - 1]
        assert_allclose(
            np_pls_alg_1.predict(X, n_components=n_good_components),
            sk_final_pred,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np_pls_alg_2.predict(X, n_components=n_good_components),
            sk_final_pred,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(jax_pls_alg_1.predict(X, n_components=n_good_components)),
            sk_final_pred,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(jax_pls_alg_2.predict(X, n_components=n_good_components)),
            sk_final_pred,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(diff_jax_pls_alg_1.predict(X, n_components=n_good_components)),
            sk_final_pred,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(diff_jax_pls_alg_2.predict(X, n_components=n_good_components)),
            sk_final_pred,
            atol=atol,
            rtol=rtol,
        )

    def check_orthogonality_properties(
        self,
        np_pls_alg_1: NpPLS,
        np_pls_alg_2: NpPLS,
        jax_pls_alg_1: JAX_Alg_1,
        jax_pls_alg_2: JAX_Alg_2,
        diff_jax_pls_alg_1: JAX_Alg_1,
        diff_jax_pls_alg_2: JAX_Alg_2,
        atol: float,
        rtol: float,
        n_good_components: int = -1,
    ) -> None:
        """
        Check orthogonality properties of PLS algorithm results.

        Parameters
        ----------
        np_pls_alg_1
            Numpy-based PLS model using algorithm 1.

        np_pls_alg_2
            Numpy-based PLS model using algorithm 2.

        jax_pls_alg_1
            JAX-based PLS model using algorithm 1.

        jax_pls_alg_2
            JAX-based PLS model using algorithm 2.

        diff_jax_pls_alg_1
            JAX-based PLS model using algorithm 1 with differentiation.

        diff_jax_pls_alg_2
            JAX-based PLS model using algorithm 2 with differentiation.

        atol : float
            Absolute tolerance for checking equality.

        rtol : float
            Relative tolerance for checking equality.

        n_good_components : int, optional
            Number of components to check, or -1 to use all possible number of
            components.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If either of the X weights or X scores are not orthogonal.
        """
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A
        # X weights should be orthogonal
        self.assert_matrix_orthogonal(
            np_pls_alg_1.W[..., :n_good_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            np_pls_alg_2.W[..., :n_good_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            np.array(jax_pls_alg_1.W)[..., :n_good_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            np.array(jax_pls_alg_2.W)[..., :n_good_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            np.array(diff_jax_pls_alg_1.W)[..., :n_good_components],
            atol=atol,
            rtol=rtol,
        )
        self.assert_matrix_orthogonal(
            np.array(diff_jax_pls_alg_2.W)[..., :n_good_components],
            atol=atol,
            rtol=rtol,
        )

        # X scores (only computed by algorithm 1) should be orthogonal
        self.assert_matrix_orthogonal(
            np_pls_alg_1.T[..., :n_good_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            np.array(jax_pls_alg_1.T)[..., :n_good_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            np.array(diff_jax_pls_alg_1.T)[..., :n_good_components],
            atol=atol,
            rtol=rtol,
        )

    def check_equality_properties(
        self,
        np_pls_alg_1: NpPLS,
        jax_pls_alg_1: JAX_Alg_1,
        diff_jax_pls_alg_1: JAX_Alg_1,
        X: npt.NDArray,
        atol: float,
        rtol: float,
        n_good_components: int = -1,
    ) -> None:
        """
        Check equality properties of PLS algorithm results.

        Parameters
        ----------
        np_pls_alg_1
            Numpy-based PLS model using algorithm 1.

        jax_pls_alg_1
            JAX-based PLS model using algorithm 1.

        diff_jax_pls_alg_1
            JAX-based PLS model using algorithm 1 with differentiation.

        X : ndarray
            Original input matrix.

        atol : float
            Absolute tolerance for comparing matrix values.

        rtol : float
            Relative tolerance for comparing matrix values.

        n_good_components : int, optional
            Number of components to check, or -1 to use all possible number of
            components.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the equality properties are not satisfied.
        """
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A

        # Assume that models are fitted on centered and scaled data
        X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True, ddof=1)

        # X can be reconstructed by multiplying X scores (T) and the transpose of X
        # loadings (P)
        assert_allclose(
            np.dot(
                np_pls_alg_1.T[..., :n_good_components],
                np_pls_alg_1.P[..., :n_good_components].T,
            ),
            X,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.dot(
                np.array(jax_pls_alg_1.T[..., :n_good_components]),
                np.array(jax_pls_alg_1.P[..., :n_good_components]).T,
            ),
            X,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.dot(
                np.array(diff_jax_pls_alg_1.T[..., :n_good_components]),
                np.array(diff_jax_pls_alg_1.P[..., :n_good_components]).T,
            ),
            X,
            atol=atol,
            rtol=rtol,
        )

        # X multiplied by X rotations (R) should be equal to X scores (T)
        assert_allclose(
            np.dot(X, np_pls_alg_1.R[..., :n_good_components]),
            np_pls_alg_1.T[..., :n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.dot(X, np.array(jax_pls_alg_1.R[..., :n_good_components])),
            np.array(jax_pls_alg_1.T[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.dot(X, np.array(diff_jax_pls_alg_1.R[..., :n_good_components])),
            np.array(diff_jax_pls_alg_1.T[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )

    def check_cpu_gpu_equality(
        self,
        np_pls_alg_1: NpPLS,
        np_pls_alg_2: NpPLS,
        jax_pls_alg_1: JAX_Alg_1,
        jax_pls_alg_2: JAX_Alg_2,
        diff_jax_pls_alg_1: JAX_Alg_1,
        diff_jax_pls_alg_2: JAX_Alg_2,
        n_good_components: int = -1,
    ) -> None:
        """
        Check equality properties between CPU and GPU implementations of PLS algorithm.

        Parameters
        ----------
        np_pls_alg_1
            Numpy-based PLS model using algorithm 1.

        np_pls_alg_2
            Numpy-based PLS model using algorithm 2.

        jax_pls_alg_1
            JAX-based PLS model using algorithm 1.

        jax_pls_alg_2
            JAX-based PLS model using algorithm 2.

        diff_jax_pls_alg_1
            JAX-based PLS model using algorithm 1 with differentiation.

        diff_jax_pls_alg_2
            JAX-based PLS model using algorithm 2 with differentiation.

        n_good_components : int, optional
            Number of components to check, or -1 to use all possible number of
            components.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the internal matrices are not consistent across NumPy and JAX
            implementations.
        """
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A

        try:
            atol = np.finfo(np_pls_alg_1.dtype).eps
        except AttributeError:
            atol = 0
        rtol = 1e-4
        # Regression matrices
        assert_allclose(
            np_pls_alg_1.B[:n_good_components],
            np.array(jax_pls_alg_1.B[:n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np_pls_alg_1.B[:n_good_components],
            np.array(diff_jax_pls_alg_1.B[:n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np_pls_alg_2.B[:n_good_components],
            np.array(jax_pls_alg_2.B[:n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np_pls_alg_2.B[:n_good_components],
            np.array(diff_jax_pls_alg_2.B[:n_good_components]),
            atol=atol,
            rtol=rtol,
        )

        # X weights
        assert_allclose(
            np.abs(np_pls_alg_1.W[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_1.W[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_1.W[..., :n_good_components]),
            np.abs(np.array(diff_jax_pls_alg_1.W[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.W[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_2.W[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.W[..., :n_good_components]),
            np.abs(np.array(diff_jax_pls_alg_2.W[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )

        # X loadings
        assert_allclose(
            np.abs(np_pls_alg_1.P[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_1.P[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_1.P[..., :n_good_components]),
            np.abs(np.array(diff_jax_pls_alg_1.P[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.P[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_2.P[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.P[..., :n_good_components]),
            np.abs(np.array(diff_jax_pls_alg_2.P[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )

        # Y loadings
        assert_allclose(
            np.abs(np_pls_alg_1.Q[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_1.Q[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_1.Q[..., :n_good_components]),
            np.abs(np.array(diff_jax_pls_alg_1.Q[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.Q[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_2.Q[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.Q[..., :n_good_components]),
            np.abs(np.array(diff_jax_pls_alg_2.Q[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )

        # X rotations
        assert_allclose(
            np.abs(np_pls_alg_1.R[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_1.R[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_1.R[..., :n_good_components]),
            np.abs(np.array(diff_jax_pls_alg_1.R[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.R[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_2.R[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.R[..., :n_good_components]),
            np.abs(np.array(diff_jax_pls_alg_2.R[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )

        # X scores - only computed by Algorithm #1
        assert_allclose(
            np.abs(np_pls_alg_1.T[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_1.T[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_1.T[..., :n_good_components]),
            np.abs(np.array(diff_jax_pls_alg_1.T[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )

    def test_pls_1(self) -> None:
        """
        Description
        -----------
        Test PLS1 algorithm.

        This method performs testing of the PLS1 algorithm using various models and
        checks for equality, orthogonality, regression matrices, and predictions. It
        also validates the algorithm's numerical stability for the "Protein" dataset.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        X = self.load_X()
        Y = self.load_Y(["Protein"])
        n_components = 25
        assert Y.shape[1] == 1
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
            diff_jax_pls_alg_1,
            diff_jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)

        self.check_cpu_gpu_equality(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
        )

        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            X=X,
            atol=1e-1,
            rtol=1e-5,
        )
        self.check_orthogonality_properties(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            atol=1e-1,
            rtol=0,
        )

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            atol=3e-8,
            rtol=2e-4,
        )

        self.check_predictions(
            sk_pls=sk_pls,
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            X=X,
            atol=1e-8,
            rtol=1e-5,
        )  # PLS1 is very numerically stable for protein.

        # Remove the singleton dimension and check that the predictions are consistent.
        Y = Y.squeeze()
        assert Y.ndim == 1
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
            diff_jax_pls_alg_1,
            diff_jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)

        self.check_cpu_gpu_equality(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
        )

        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            X=X,
            atol=1e-1,
            rtol=1e-5,
        )
        self.check_orthogonality_properties(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            atol=1e-1,
            rtol=0,
        )

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            atol=3e-8,
            rtol=2e-4,
        )

        self.check_predictions(
            sk_pls=sk_pls,
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            X=X,
            atol=1e-8,
            rtol=1e-5,
        )  # PLS1 is very numerically stable for protein.

    def test_pls_2_m_less_k(self):
        """
        Description
        -----------
        Test PLS2 algorithm when the number of targets is less than the number of
        features (M < K).

        This method tests the PLS2 algorithm under the scenario where the number of
        target variables (M) is less than the number of features (K) in the dataset. It
        performs various tests on different PLS2 models, checks for equality,
        orthogonality, regression matrices, and predictions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
                "Wheat_H5",
                "Wheat_Halland",
                "Wheat_Oland",
                "Wheat_Spelt",
                "Moisture",
                "Protein",
            ]
        )
        assert Y.shape[1] > 1
        assert Y.shape[1] < X.shape[1]
        n_components = 25
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
            diff_jax_pls_alg_1,
            diff_jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)

        self.check_cpu_gpu_equality(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
        )

        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            X=X,
            atol=1e-1,
            rtol=1e-5,
        )
        self.check_orthogonality_properties(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            atol=1e-1,
            rtol=0,
        )

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            atol=0.06,
            rtol=0,
        )
        self.check_predictions(
            sk_pls=sk_pls,
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            X=X,
            atol=1.3e-2,
            rtol=0,
        )  # PLS2 is not as numerically stable as PLS1.

    def test_pls_2_m_eq_k(self) -> None:
        """
        Description
        -----------
        Test PLS2 algorithm where the number of targets is equal to the number of
        features (M = K).

        This method tests the PLS2 algorithm under the scenario where the number of
        target variables (M) is equal to the number of features (K) in the dataset. It
        performs various tests on different PLS2 models, checks for equality,
        orthogonality, regression matrices, and predictions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        X = self.load_X()
        X = X[..., :10]
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
                "Wheat_H5",
                "Wheat_Halland",
                "Wheat_Oland",
                "Wheat_Spelt",
                "Moisture",
                "Protein",
            ]
        )
        assert Y.shape[1] > 1
        assert Y.shape[1] == X.shape[1]
        n_components = 10
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
            diff_jax_pls_alg_1,
            diff_jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)

        self.check_cpu_gpu_equality(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
        )

        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            X=X,
            atol=1e-1,
            rtol=1e-5,
        )
        self.check_orthogonality_properties(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            atol=1e-1,
            rtol=0,
        )

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            atol=1e-8,
            rtol=0.1,
        )
        self.check_predictions(
            sk_pls=sk_pls,
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            X=X,
            atol=2.1e-3,
            rtol=0,
        )  # PLS2 is not as numerically stable as PLS1.

    def test_pls_2_m_greater_k(self) -> None:
        """
        Description
        -----------

        Test PLS2 algorithm where the number of targets is greater than the number of
        features (M > K).

        This method tests the PLS2 algorithm under the scenario where the number of
        target variables (M) is greater than the number of features (K) in the dataset.
        It performs various tests on different PLS2 models, checks for equality,
        orthogonality, regression matrices, and predictions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        X = self.load_X()
        X = X[..., :9]
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
                "Wheat_H5",
                "Wheat_Halland",
                "Wheat_Oland",
                "Wheat_Spelt",
                "Moisture",
                "Protein",
            ]
        )
        assert Y.shape[1] > 1
        assert Y.shape[1] > X.shape[1]
        n_components = 9
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
            diff_jax_pls_alg_1,
            diff_jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)

        self.check_cpu_gpu_equality(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
        )

        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            X=X,
            atol=1e-1,
            rtol=1e-5,
        )
        self.check_orthogonality_properties(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            atol=1e-1,
            rtol=0,
        )

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            atol=1e-8,
            rtol=2e-2,
        )
        self.check_predictions(
            sk_pls=sk_pls,
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            X=X,
            atol=2e-3,
            rtol=0,
        )  # PLS2 is not as numerically stable as PLS1.

    def test_sanity_check_pls_regression(self) -> None:
        """
        Description
        -----------
        Taken from SkLearn's test suite and modified to include own algorithms. Test
        the PLS regression algorithm with a sanity check.

        This method performs a sanity check on the PLS regression algorithm. It loads
        the Linnerud dataset, fits the PLS regression models using various algorithms,
        and checks for equality between implemenetations' regression matrices,
        predictions, and for PLS properties. It also compares the results to expected
        values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        d = load_linnerud()
        X = d.data  # Shape = (20,3)
        Y = d.target  # Shape = (20,3)
        n_components = X.shape[1]  # 3
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
            diff_jax_pls_alg_1,
            diff_jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)

        self.check_cpu_gpu_equality(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
        )

        # Check for orthogonal X weights.
        self.assert_matrix_orthogonal(sk_pls.x_weights_, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(np_pls_alg_1.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(np_pls_alg_2.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(jax_pls_alg_1.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(jax_pls_alg_2.W, atol=1e-8, rtol=0)

        # Check for orthogonal X scores - not computed by Algorithm #2.
        self.assert_matrix_orthogonal(sk_pls.x_scores_, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(np_pls_alg_1.T, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(jax_pls_alg_1.T, atol=1e-8, rtol=0)

        # Check invariants.
        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            X=X,
            atol=1e-8,
            rtol=1e-5,
        )

        expected_x_weights = np.array(
            [
                [-0.61330704, -0.00443647, 0.78983213],
                [-0.74697144, -0.32172099, -0.58183269],
                [-0.25668686, 0.94682413, -0.19399983],
            ]
        )

        expected_x_loadings = np.array(
            [
                [-0.61470416, -0.24574278, 0.78983213],
                [-0.65625755, -0.14396183, -0.58183269],
                [-0.51733059, 1.00609417, -0.19399983],
            ]
        )

        expected_y_loadings = np.array(
            [
                [+0.32456184, 0.29892183, 0.20316322],
                [+0.42439636, 0.61970543, 0.19320542],
                [-0.13143144, -0.26348971, -0.17092916],
            ]
        )

        # Check for expected X weights
        assert_allclose(
            np.abs(sk_pls.x_weights_), np.abs(expected_x_weights), atol=1e-8, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_1.W), np.abs(expected_x_weights), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_2.W), np.abs(expected_x_weights), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_1.W), np.abs(expected_x_weights), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_2.W), np.abs(expected_x_weights), atol=2e-6, rtol=0
        )

        # Check for expected X loadings
        assert_allclose(
            np.abs(sk_pls.x_loadings_), np.abs(expected_x_loadings), atol=1e-8, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_1.P), np.abs(expected_x_loadings), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_2.P), np.abs(expected_x_loadings), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_1.P), np.abs(expected_x_loadings), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_2.P), np.abs(expected_x_loadings), atol=2e-6, rtol=0
        )

        # Check for expected Y loadings
        assert_allclose(
            np.abs(sk_pls.y_loadings_), np.abs(expected_y_loadings), atol=1e-8, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_1.Q), np.abs(expected_y_loadings), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_2.Q), np.abs(expected_y_loadings), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_1.Q), np.abs(expected_y_loadings), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_2.Q), np.abs(expected_y_loadings), atol=2e-6, rtol=0
        )

        # Check that sign flip is consistent and exact across loadings and weights
        sk_x_loadings_sign_flip = np.sign(sk_pls.x_loadings_ / expected_x_loadings)
        sk_x_weights_sign_flip = np.sign(sk_pls.x_weights_ / expected_x_weights)
        sk_y_loadings_sign_flip = np.sign(sk_pls.y_loadings_ / expected_y_loadings)
        assert_allclose(sk_x_loadings_sign_flip, sk_x_weights_sign_flip, atol=0, rtol=0)
        assert_allclose(
            sk_x_loadings_sign_flip, sk_y_loadings_sign_flip, atol=0, rtol=0
        )

        np_alg_1_x_loadings_sign_flip = np.sign(np_pls_alg_1.P / expected_x_loadings)
        np_alg_1_x_weights_sign_flip = np.sign(np_pls_alg_1.W / expected_x_weights)
        np_alg_1_y_loadings_sign_flip = np.sign(np_pls_alg_1.Q / expected_y_loadings)
        assert_allclose(
            np_alg_1_x_loadings_sign_flip, np_alg_1_x_weights_sign_flip, atol=0, rtol=0
        )
        assert_allclose(
            np_alg_1_x_loadings_sign_flip, np_alg_1_y_loadings_sign_flip, atol=0, rtol=0
        )

        np_alg_2_x_loadings_sign_flip = np.sign(np_pls_alg_2.P / expected_x_loadings)
        np_alg_2_x_weights_sign_flip = np.sign(np_pls_alg_2.W / expected_x_weights)
        np_alg_2_y_loadings_sign_flip = np.sign(np_pls_alg_2.Q / expected_y_loadings)
        assert_allclose(
            np_alg_2_x_loadings_sign_flip, np_alg_2_x_weights_sign_flip, atol=0, rtol=0
        )
        assert_allclose(
            np_alg_2_x_loadings_sign_flip, np_alg_2_y_loadings_sign_flip, atol=0, rtol=0
        )

        jax_alg_1_x_loadings_sign_flip = np.sign(jax_pls_alg_1.P / expected_x_loadings)
        jax_alg_1_x_weights_sign_flip = np.sign(jax_pls_alg_1.W / expected_x_weights)
        jax_alg_1_y_loadings_sign_flip = np.sign(jax_pls_alg_1.Q / expected_y_loadings)
        assert_allclose(
            jax_alg_1_x_loadings_sign_flip,
            jax_alg_1_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert_allclose(
            jax_alg_1_x_loadings_sign_flip,
            jax_alg_1_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

        jax_alg_2_x_loadings_sign_flip = np.sign(jax_pls_alg_2.P / expected_x_loadings)
        jax_alg_2_x_weights_sign_flip = np.sign(jax_pls_alg_2.W / expected_x_weights)
        jax_alg_2_y_loadings_sign_flip = np.sign(jax_pls_alg_2.Q / expected_y_loadings)
        assert_allclose(
            jax_alg_2_x_loadings_sign_flip,
            jax_alg_2_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert_allclose(
            jax_alg_2_x_loadings_sign_flip,
            jax_alg_2_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

    def test_sanity_check_pls_regression_constant_column_Y(
        self,
    ) -> (
        None
    ):  # Taken from SkLearn's test suite and modified to include own algorithms.
        """
        Description
        -----------

        Test the PLS regression algorithm with a sanity check and a constant column in
        Y.

        This method performs a sanity check on the PLS regression algorithm using a
        dataset that includes a constant column in the target (Y) data. It loads the
        Linnerud dataset, sets the first column of Y to a constant, and fits the PLS
        regression models using various algorithms. The test checks for equality
        between implemenetations' regression matrices, predictions, and for PLS
        properties. It also compares the results to expected values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        d = load_linnerud()
        X = d.data  # Shape = (20,3)
        Y = d.target  # Shape = (20,3)
        Y[:, 0] = 1  # Set the first column to a constant
        n_components = X.shape[1]  # 3
        (
            sk_pls,
            _sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
            diff_jax_pls_alg_1,
            diff_jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)

        self.check_cpu_gpu_equality(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
        )

        expected_x_weights = np.array(
            [
                [-0.6273573, 0.007081799, 0.7786994],
                [-0.7493417, -0.277612681, -0.6011807],
                [-0.2119194, 0.960666981, -0.1794690],
            ]
        )

        expected_x_loadings = np.array(
            [
                [-0.6273512, -0.22464538, 0.7786994],
                [-0.6643156, -0.09871193, -0.6011807],
                [-0.5125877, 1.01407380, -0.1794690],
            ]
        )

        expected_y_loadings = np.array(
            [
                [0.0000000, 0.0000000, 0.0000000],
                [0.4357300, 0.5828479, 0.2174802],
                [-0.1353739, -0.2486423, -0.1810386],
            ]
        )

        # Check for expected X weights
        assert_allclose(
            np.abs(sk_pls.x_weights_), np.abs(expected_x_weights), atol=5e-8, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_1.W), np.abs(expected_x_weights), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_2.W), np.abs(expected_x_weights), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_1.W), np.abs(expected_x_weights), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_2.W), np.abs(expected_x_weights), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(diff_jax_pls_alg_1.W), np.abs(expected_x_weights), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(diff_jax_pls_alg_2.W), np.abs(expected_x_weights), atol=3e-6, rtol=0
        )

        # Check for expected X loadings
        assert_allclose(
            np.abs(sk_pls.x_loadings_), np.abs(expected_x_loadings), atol=5e-8, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_1.P), np.abs(expected_x_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_2.P), np.abs(expected_x_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_1.P), np.abs(expected_x_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_2.P), np.abs(expected_x_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(diff_jax_pls_alg_1.P), np.abs(expected_x_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(diff_jax_pls_alg_2.P), np.abs(expected_x_loadings), atol=3e-6, rtol=0
        )

        # Check for expected Y loadings
        assert_allclose(
            np.abs(sk_pls.y_loadings_), np.abs(expected_y_loadings), atol=5e-8, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_1.Q), np.abs(expected_y_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_2.Q), np.abs(expected_y_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_1.Q), np.abs(expected_y_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_2.Q), np.abs(expected_y_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(diff_jax_pls_alg_1.Q), np.abs(expected_y_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(diff_jax_pls_alg_2.Q), np.abs(expected_y_loadings), atol=3e-6, rtol=0
        )

        # Check for orthogonal X weights.
        self.assert_matrix_orthogonal(sk_pls.x_weights_, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(np_pls_alg_1.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(np_pls_alg_2.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(jax_pls_alg_1.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(jax_pls_alg_2.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(diff_jax_pls_alg_1.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(diff_jax_pls_alg_2.W, atol=1e-8, rtol=0)

        # Check for orthogonal X scores - not computed by Algorithm #2.
        self.assert_matrix_orthogonal(sk_pls.x_scores_, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(np_pls_alg_1.T, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(jax_pls_alg_1.T, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(diff_jax_pls_alg_1.T, atol=1e-8, rtol=0)

        # Check that sign flip is consistent and exact across loadings and weights.
        # Ignore the first column of Y which will be a column of zeros (due to mean
        # centering of its constant value).
        sk_x_loadings_sign_flip = np.sign(sk_pls.x_loadings_ / expected_x_loadings)
        sk_x_weights_sign_flip = np.sign(sk_pls.x_weights_ / expected_x_weights)
        sk_y_loadings_sign_flip = np.sign(
            sk_pls.y_loadings_[1:] / expected_y_loadings[1:]
        )
        assert_allclose(sk_x_loadings_sign_flip, sk_x_weights_sign_flip, atol=0, rtol=0)
        assert_allclose(
            sk_x_loadings_sign_flip[1:], sk_y_loadings_sign_flip, atol=0, rtol=0
        )

        np_alg_1_x_loadings_sign_flip = np.sign(np_pls_alg_1.P / expected_x_loadings)
        np_alg_1_x_weights_sign_flip = np.sign(np_pls_alg_1.W / expected_x_weights)
        np_alg_1_y_loadings_sign_flip = np.sign(
            np_pls_alg_1.Q[1:] / expected_y_loadings[1:]
        )
        assert_allclose(
            np_alg_1_x_loadings_sign_flip, np_alg_1_x_weights_sign_flip, atol=0, rtol=0
        )
        assert_allclose(
            np_alg_1_x_loadings_sign_flip[1:],
            np_alg_1_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

        np_alg_2_x_loadings_sign_flip = np.sign(np_pls_alg_2.P / expected_x_loadings)
        np_alg_2_x_weights_sign_flip = np.sign(np_pls_alg_2.W / expected_x_weights)
        np_alg_2_y_loadings_sign_flip = np.sign(
            np_pls_alg_2.Q[1:] / expected_y_loadings[1:]
        )
        assert_allclose(
            np_alg_2_x_loadings_sign_flip, np_alg_2_x_weights_sign_flip, atol=0, rtol=0
        )
        assert_allclose(
            np_alg_2_x_loadings_sign_flip[1:],
            np_alg_2_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

        jax_alg_1_x_loadings_sign_flip = np.sign(jax_pls_alg_1.P / expected_x_loadings)
        jax_alg_1_x_weights_sign_flip = np.sign(jax_pls_alg_1.W / expected_x_weights)
        jax_alg_1_y_loadings_sign_flip = np.sign(
            jax_pls_alg_1.Q[1:] / expected_y_loadings[1:]
        )
        assert_allclose(
            jax_alg_1_x_loadings_sign_flip,
            jax_alg_1_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert_allclose(
            jax_alg_1_x_loadings_sign_flip[1:],
            jax_alg_1_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

        diff_jax_alg_1_x_loadings_sign_flip = np.sign(
            diff_jax_pls_alg_1.P / expected_x_loadings
        )
        diff_jax_alg_1_x_weights_sign_flip = np.sign(
            diff_jax_pls_alg_1.W / expected_x_weights
        )
        diff_jax_alg_1_y_loadings_sign_flip = np.sign(
            diff_jax_pls_alg_1.Q[1:] / expected_y_loadings[1:]
        )
        assert_allclose(
            diff_jax_alg_1_x_loadings_sign_flip,
            diff_jax_alg_1_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert_allclose(
            diff_jax_alg_1_x_loadings_sign_flip[1:],
            diff_jax_alg_1_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

        jax_alg_2_x_loadings_sign_flip = np.sign(jax_pls_alg_2.P / expected_x_loadings)
        jax_alg_2_x_weights_sign_flip = np.sign(jax_pls_alg_2.W / expected_x_weights)
        jax_alg_2_y_loadings_sign_flip = np.sign(
            jax_pls_alg_2.Q[1:] / expected_y_loadings[1:]
        )
        assert_allclose(
            jax_alg_2_x_loadings_sign_flip,
            jax_alg_2_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert_allclose(
            jax_alg_2_x_loadings_sign_flip[1:],
            jax_alg_2_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

        diff_jax_alg_2_x_loadings_sign_flip = np.sign(
            diff_jax_pls_alg_2.P / expected_x_loadings
        )
        diff_jax_alg_2_x_weights_sign_flip = np.sign(
            diff_jax_pls_alg_2.W / expected_x_weights
        )
        diff_jax_alg_2_y_loadings_sign_flip = np.sign(
            diff_jax_pls_alg_2.Q[1:] / expected_y_loadings[1:]
        )
        assert_allclose(
            diff_jax_alg_2_x_loadings_sign_flip,
            diff_jax_alg_2_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert_allclose(
            diff_jax_alg_2_x_loadings_sign_flip[1:],
            diff_jax_alg_2_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

    def _helper_check_pls_constant_y(
        self,
        pls_model: Union[SkPLS, NpPLS, JAX_Alg_1, JAX_Alg_2, FastCVPLS],
        X: npt.NDArray,
        Y: npt.NDArray,
        n_components: int,
        cv_splits: Optional[npt.NDArray] = None,
    ) -> None:
        """
        Description
        -----------
        Check that every call to PLS fit correctly raises a warning when Y is constant.

        This method checks that every call to the PLS fit method raises a warning when
        the target data (Y) is constant. It fits the PLS regression models using
        various algorithms and checks for warnings related to weights being close to
        zero.

        Parameters
        ----------
        pls_model : Union[SkPLS, NpPLS, JAX_Alg_1, JAX_Alg_2, FastCVPLS]
            The PLS regression model to test.
        X : numpy.ndarray
            The predictor variables.
        Y : numpy.ndarray
            The target variables.
        n_components : int
            The number of components to extract.

        Returns
        -------
        None
        """
        if isinstance(pls_model, SkPLS):
            msg = "y residual is constant at iteration"
            with pytest.warns(UserWarning, match=msg) as record:
                for _ in range(2):
                    pls_model.fit(X=X, y=Y)
                    assert_allclose(pls_model.x_rotations_, 0)
                assert len(record) == 2

        elif isinstance(pls_model, FastCVPLS):
            msg = "Weight is close to zero."
            with pytest.warns(UserWarning, match=msg) as record:
                pls_model.cross_validate(
                    X=X,
                    Y=Y,
                    A=n_components,
                    cv_splits=cv_splits,
                    metric_function=lambda x, y: 0,
                    n_jobs=1,
                )
                assert len(record) == 2
        elif isinstance(pls_model, (NpPLS, JAX_Alg_1, JAX_Alg_2)):
            msg = "Weight is close to zero."
            with pytest.warns(UserWarning, match=msg) as record:
                for _ in range(2):
                    pls_model.fit(X=X, Y=Y, A=n_components)
                    if isinstance(pls_model, NpPLS):
                        assert_allclose(pls_model.R, 0)
                    assert pls_model.A > pls_model.max_stable_components
                assert len(record) >= 2
        else:
            raise ValueError("Invalid PLS model.")

    def check_pls_constant_y(
        self, X: npt.NDArray, Y: npt.NDArray
    ) -> (
        None
    ):  # Taken from SkLearn's test suite and modified to include own algorithms.
        """
        Description
        -----------
        Check PLS regression behavior when Y is constant.

        This method checks the behavior of PLS regression when the target data (Y) is
        constant. It first pre-processes the input data by centering and scaling it.
        Then, it fits PLS regression models using different algorithms, including the
        sklearn, NumPy-based, and JAX-based implementations. It checks for warnings
        related weights being close to zero during the fitting process.

        Parameters
        ----------
        X : numpy.ndarray
            The predictor variables.
        Y : numpy.ndarray
            The target variables.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the warnings for weights being close to zero are not raised for all
            implementations.
        """
        n_components = 2
        sk_pls = SkPLS(n_components=n_components)  # Do not rescale again.
        np_pls_alg_1 = NpPLS(algorithm=1)
        np_pls_alg_2 = NpPLS(algorithm=2)
        jax_pls_alg_1 = JAX_Alg_1(differentiable=False, verbose=True)
        jax_pls_alg_2 = JAX_Alg_2(differentiable=False, verbose=True)
        fast_cv_alg_1 = FastCVPLS(algorithm=1)
        fast_cv_alg_2 = FastCVPLS(algorithm=2)
        cv_splits = np.zeros(shape=(X.shape[0],), dtype=int)
        cv_splits[: X.shape[0] // 2] = 1

        self._helper_check_pls_constant_y(
            pls_model=sk_pls, X=X, Y=Y, n_components=n_components
        )
        self._helper_check_pls_constant_y(
            pls_model=np_pls_alg_1, X=X, Y=Y, n_components=n_components
        )
        self._helper_check_pls_constant_y(
            pls_model=np_pls_alg_2, X=X, Y=Y, n_components=n_components
        )
        self._helper_check_pls_constant_y(
            pls_model=jax_pls_alg_1, X=X, Y=Y, n_components=n_components
        )
        self._helper_check_pls_constant_y(
            pls_model=jax_pls_alg_2, X=X, Y=Y, n_components=n_components
        )
        self._helper_check_pls_constant_y(
            pls_model=fast_cv_alg_1,
            X=X,
            Y=Y,
            n_components=n_components,
            cv_splits=cv_splits,
        )
        self._helper_check_pls_constant_y(
            pls_model=fast_cv_alg_2,
            X=X,
            Y=Y,
            n_components=n_components,
            cv_splits=cv_splits,
        )

    def test_pls_1_constant_y(self):
        """
        Description
        -----------
        Test PLS regression when Y is constant with single target variable.

        This test generates random predictor variables (X) and a target variable (Y)
        where Y is a constant array with a single column. It ensures that Y has only
        one column and calls the 'check_pls_constant_y' method to validate the behavior
        of PLS regression in this scenario.

        Returns
        -------
        None
        """
        rng = np.random.RandomState(42)
        X = rng.rand(100, 3)
        Y = np.zeros(shape=(100, 1))
        assert Y.shape[1] == 1
        self.check_pls_constant_y(X, Y)

    def test_pls_2_m_less_k_constant_y(self):
        """
        Description
        -----------
        Test PLS regression when Y is constant with m < k target variables.

        This test generates random predictor variables (X) and a target variable (Y)
        where Y is a constant array with fewer columns (m) than the number of columns
        in X (k). It ensures that Y has more than one column but less than the number
        of columns in X and calls the 'check_pls_constant_y' method to validate the
        behavior of PLS regression in this scenario.

        Returns
        -------
        None
        """
        rng = np.random.RandomState(42)
        X = rng.rand(100, 3)
        Y = np.zeros(shape=(100, 2))
        assert Y.shape[1] > 1
        assert Y.shape[1] < X.shape[1]
        self.check_pls_constant_y(X, Y)

    def test_pls_2_m_eq_k_constant_y(self):
        """
        Description
        -----------
        Test PLS regression when Y is constant with m = k target variables.

        This test generates random predictor variables (X) and a target variable (Y)
        where Y is a constant array with the same number of columns (m) as the number
        of columns in X (k). It ensures that Y has more than one column and the same
        number of columns as X, and calls the 'check_pls_constant_y' method to validate
        the behavior of PLS regression in this scenario.

        Returns
        -------
        None
        """
        rng = np.random.RandomState(42)
        X = rng.rand(100, 3)
        Y = np.zeros(shape=(100, 3))
        assert Y.shape[1] > 1
        assert Y.shape[1] == X.shape[1]
        self.check_pls_constant_y(X, Y)

    def test_pls_2_m_greater_k_constant_y(self):
        """
        Description
        -----------
        Test PLS regression when Y is constant with m > k target variables.

        This test generates random predictor variables (X) and a target variable (Y)
        where Y is a constant array with more columns (M) than the number of columns in
        X (K). It ensures that Y has more than one column and more columns than X, and
        calls the 'check_pls_constant_y' method to validate the behavior of PLS
        regression in this scenario.

        Returns
        -------
        None
        """
        rng = np.random.RandomState(42)
        X = rng.rand(100, 3)
        Y = np.zeros(shape=(100, 4))
        assert Y.shape[1] > 1
        assert Y.shape[1] > X.shape[1]
        self.check_pls_constant_y(X, Y)

    def check_gradient_pls(
        self,
        X: npt.NDArray,
        Y: npt.NDArray,
        num_components: int,
        filter_size: int,
        val_atol: float,
        val_rtol: float,
        grad_atol: float,
        grad_rtol: float,
    ) -> None:
        """
        Description
        -----------
        This method tests the gradient propagation for differentiable JAX PLS. It
        convolves the input spectra with a filter, computes the gradients of the RMSE
        loss with respect to the parameters of the preprocessing filter, and verifies
        the correctness of gradient values and numerical stability.

        Parameters:
        X : numpy.ndarray:
            The input predictor variables.

        Y : numpy.ndarray
            The target variables.

        num_components : int
            The number of PLS components.

        filter_size (int):
            The size of the convolution filter.

        val_atol : float
            Absolute tolerance for value comparisons.

        val_rtol : float
            Relative tolerance for value comparisons.

        grad_atol : float
            Absolute tolerance for gradient comparisons.

        grad_rtol : float
            Relative tolerance for gradient comparisons.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the gradients are not computed, if the gradients are not equal across
            Improved Kernel PLS Algorithm #1 and #2, or if the output values are not
            consistent across all JAX implementations.
        """
        jnp_X = jnp.array(X, dtype=jnp.float64)
        jnp_Y = jnp.array(Y, dtype=jnp.float64)

        diff_pls_alg_1 = JAX_Alg_1(differentiable=True, verbose=True)
        diff_pls_alg_2 = JAX_Alg_2(differentiable=True, verbose=True)

        uniform_filter = jnp.ones(filter_size) / filter_size

        # Preprocessing convolution filter for which we will obtain the gradients.
        @jax.jit
        def apply_1d_convolution(
            matrix: jnp.ndarray, conv_filter: jnp.ndarray
        ) -> jnp.ndarray:
            convolved_rows = jax.vmap(
                lambda row: jnp.convolve(row, conv_filter, "valid")
            )(matrix)
            return convolved_rows

        # Loss function which we want to minimize.
        @jax.jit
        def rmse(Y_true: jnp.ndarray, Y_pred: jnp.ndarray) -> float:
            e = Y_true - Y_pred
            se = e**2
            mse = jnp.mean(se)
            rmse = jnp.sqrt(mse)
            return rmse

        # Function to differentiate.
        def preprocess_fit_rmse(
            X: jnp.ndarray, Y: jnp.ndarray, pls_alg, A: int
        ) -> Callable[[jnp.ndarray], float]:
            @jax.jit
            def helper(conv_filter):
                filtered_X = apply_1d_convolution(X, conv_filter)
                matrices = pls_alg.stateless_fit(filtered_X, Y, A)
                B = matrices[0]
                Y_pred = pls_alg.stateless_predict(filtered_X, B, A)
                rmse_loss = rmse(Y, Y_pred)
                return jnp.squeeze(rmse_loss)

            return helper

        # Compute values and gradients for algorithm #1
        grad_fun = jax.value_and_grad(
            preprocess_fit_rmse(jnp_X, jnp_Y, diff_pls_alg_1, num_components), argnums=0
        )
        output_val_diff_alg_1, grad_alg_1 = grad_fun(uniform_filter)

        # Compute the gradient and output value for a single number of components
        grad_fun = jax.value_and_grad(
            preprocess_fit_rmse(jnp_X, jnp_Y, diff_pls_alg_2, num_components), argnums=0
        )
        output_val_diff_alg_2, grad_alg_2 = grad_fun(uniform_filter)

        # Check that outputs and gradients of algorithm 1 and 2 are identical
        assert_allclose(
            np.array(grad_alg_1), np.array(grad_alg_2), atol=grad_atol, rtol=grad_rtol
        )
        assert_allclose(
            np.array(output_val_diff_alg_1),
            np.array(output_val_diff_alg_2),
            atol=val_atol,
            rtol=val_rtol,
        )

        # Check that no NaNs are encountered in the gradient
        assert jnp.all(~jnp.isnan(grad_alg_1))
        assert jnp.all(~jnp.isnan(grad_alg_2))

        # Check that the gradient actually flows all the way through
        zeros = jnp.zeros(filter_size, dtype=jnp.float64)
        assert jnp.any(jnp.not_equal(grad_alg_1, zeros))
        assert jnp.any(jnp.not_equal(grad_alg_2, zeros))

        # Check that we can not differentiate the JAX implementations using
        # differentiable=False
        pls_alg_1 = JAX_Alg_1(differentiable=False, verbose=True)
        pls_alg_2 = JAX_Alg_2(differentiable=False, verbose=True)
        msg = (
            "Reverse-mode differentiation does not work for lax.while_loop or "
            "lax.fori_loop with dynamic start/stop values"
        )
        with pytest.raises(ValueError, match=msg):
            grad_fun = jax.value_and_grad(
                preprocess_fit_rmse(jnp_X, jnp_Y, pls_alg_1, num_components), argnums=0
            )
            grad_fun(uniform_filter)

        with pytest.raises(ValueError, match=msg):
            grad_fun = jax.value_and_grad(
                preprocess_fit_rmse(jnp_X, jnp_Y, pls_alg_2, num_components), argnums=0
            )
            grad_fun(uniform_filter)

        # For good measure, let's assure ourselves that the results are equivalent
        # across differentiable and non differentiable versions:
        output_val_alg_1 = preprocess_fit_rmse(jnp_X, jnp_Y, pls_alg_1, num_components)(
            uniform_filter
        )
        output_val_alg_2 = preprocess_fit_rmse(jnp_X, jnp_Y, pls_alg_2, num_components)(
            uniform_filter
        )
        assert_allclose(output_val_alg_1, output_val_diff_alg_1, atol=0, rtol=1e-8)
        assert_allclose(output_val_alg_2, output_val_diff_alg_2, atol=0, rtol=1e-8)

    def test_gradient_pls_1(self):
        """
        Description
        -----------
        This test loads input predictor variables and a target variable with a single
        column and calls the 'check_gradient_pls' method to validate the gradient
        propagation for differentiable JAX PLS.

        Returns:
        None
        """
        X = self.load_X()
        Y = self.load_Y(["Protein"])
        num_components = 25
        filter_size = 7
        assert Y.shape[1] == 1
        self.check_gradient_pls(
            X=X,
            Y=Y,
            num_components=num_components,
            filter_size=filter_size,
            val_atol=0,
            val_rtol=1e-5,
            grad_atol=0,
            grad_rtol=1e-5,
        )

    def test_gradient_pls_2_m_less_k(self):
        """
        Description
        -----------
        This test loads input predictor variables and multiple target variables with
        M < K, and calls the 'check_gradient_pls' method to validate the gradient
        propagation for differentiable JAX PLS.

        Returns:
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
                "Wheat_H5",
                "Wheat_Halland",
                "Wheat_Oland",
                "Wheat_Spelt",
                "Moisture",
                "Protein",
            ]
        )
        num_components = 25
        filter_size = 7
        assert Y.shape[1] > 1

        # The output of the convolution preprocessing is what is actually fed as input
        # to the PLS algorithms.
        assert Y.shape[1] < X.shape[1] - filter_size + 1
        self.check_gradient_pls(
            X=X,
            Y=Y,
            num_components=num_components,
            filter_size=filter_size,
            val_atol=0,
            val_rtol=1e-5,
            grad_atol=0,
            grad_rtol=1e-5,
        )

    def test_gradient_pls_2_m_eq_k(self):
        """
        Description
        -----------
        This test loads input predictor variables and multiple target variables with
        M = K, and calls the 'check_gradient_pls' method to validate the gradient
        propagation for differentiable JAX PLS.

        Returns:
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
                "Wheat_H5",
                "Wheat_Halland",
                "Wheat_Oland",
                "Wheat_Spelt",
                "Moisture",
                "Protein",
            ]
        )
        filter_size = 7
        X = X[..., : 10 + filter_size - 1]
        num_components = 10
        assert Y.shape[1] > 1

        # The output of the convolution preprocessing is what is actually fed as input
        # to the PLS algorithms.
        assert Y.shape[1] == X.shape[1] - filter_size + 1
        self.check_gradient_pls(
            X=X,
            Y=Y,
            num_components=num_components,
            filter_size=filter_size,
            val_atol=0,
            val_rtol=1e-5,
            grad_atol=0,
            grad_rtol=1e-5,
        )

    def test_gradient_pls_2_m_greater_k(self):
        """
        Description
        -----------
        This test loads input predictor variables and multiple target variables with
        M > K, and calls the 'check_gradient_pls' method to validate the gradient
        propagation for differentiable JAX PLS.

        Returns:
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
                "Wheat_H5",
                "Wheat_Halland",
                "Wheat_Oland",
                "Wheat_Spelt",
                "Moisture",
                "Protein",
            ]
        )
        filter_size = 7
        X = X[..., : 9 + filter_size - 1]
        num_components = 9
        assert Y.shape[1] > 1

        # The output of the convolution preprocessing is what is actually fed as input
        # to the PLS algorithms.
        assert Y.shape[1] > X.shape[1] - filter_size + 1
        self.check_gradient_pls(
            X=X,
            Y=Y,
            num_components=num_components,
            filter_size=filter_size,
            val_atol=0,
            val_rtol=1e-5,
            grad_atol=0,
            grad_rtol=1e-5,
        )

    def check_cross_val_pls(
        self,
        X: npt.NDArray,
        Y: npt.NDArray,
        splits: npt.NDArray,
        atol: float,
        rtol: float,
    ) -> None:
        """
        Description
        -----------
        This method tests the ability to perform cross-validation to obtain the root
        mean square error (RMSE) and the best number of components for each target
        variable and each split.

        Parameters:
        X : numpy.ndarray
            The input predictor variables.
        Y : numpy.ndarray
            The target variables.
        splits : numpy.ndarray
            Split indices for cross-validation.

        atol : float
            Absolute tolerance for value comparisons.

        rtol : float
            Relative tolerance for value comparisons.

        Returns:
        None

        Raises
        ------
        AssertionError
            If the best number of components found by cross validation with is not
            exactly equal across each different PLS implementation.

            If the output RMSEs for the best number of components are not equal down to
            the specified tolerance across each different PLS implementation.
        """

        try:
            M = Y.shape[1]
        except IndexError:
            M = 1

        n_components = X.shape[1]

        sk_pls = SkPLS(n_components=n_components)
        jax_pls_alg_1 = JAX_Alg_1(differentiable=False, verbose=True)
        jax_pls_alg_2 = JAX_Alg_2(differentiable=False, verbose=True)
        diff_jax_pls_alg_1 = JAX_Alg_1(differentiable=True, verbose=True)
        diff_jax_pls_alg_2 = JAX_Alg_2(differentiable=True, verbose=True)

        jnp_splits = jnp.array(splits)

        # Calibrate SkPLS
        sk_results = cross_validate(
            sk_pls, X, Y, cv=self.cv_splitter(splits), return_estimator=True, n_jobs=-1
        )
        sk_models = sk_results["estimator"]

        # Extract regression matrices for SkPLS for all possible number of components
        # and make a prediction with the regression matrices at all possible number of
        # components.
        sk_Bs = np.empty((len(sk_models), n_components, X.shape[1], M))
        sk_preds = np.empty((len(sk_models), n_components, X.shape[0], M))
        for i, sk_model in enumerate(sk_models):
            for j in range(sk_Bs.shape[1]):
                sk_B_at_component_j = np.dot(
                    sk_model.x_rotations_[..., : j + 1],
                    sk_model.y_loadings_[..., : j + 1].T,
                )
                sk_Bs[i, j] = sk_B_at_component_j
            sk_pred = (X - sk_model._x_mean) / sk_model._x_std @ (
                sk_Bs[i] * sk_model._y_std
            ) + sk_model._y_mean
            sk_preds[i] = sk_pred

            # Sanity check. SkPLS also uses the maximum number of components in its
            # predict method.
            assert_allclose(
                sk_pred[-1],
                sk_models[i].predict(X).reshape(X.shape[0], M),
                atol=0,
                rtol=1e-7,
            )

        # Compute RMSE on the validation predictions
        sk_pls_rmses = np.empty((len(sk_models), n_components, M))
        for i in range(len(sk_models)):
            val_idxs = val_idxs = np.nonzero(splits == i)[0]
            Y_true = Y[val_idxs]
            Y_pred = sk_preds[i, :, val_idxs, ...].swapaxes(0, 1)
            val_rmses = self.rmse_per_component(Y_true, Y_pred)
            sk_pls_rmses[i] = val_rmses

        # Calibrate NumPy PLS
        np_pls_alg_1 = NpPLS(algorithm=1)
        np_pls_alg_2 = NpPLS(algorithm=2)
        np_pls_alg_1_rmses = np_pls_alg_1.cross_validate(
            X, Y, n_components, splits.flatten(), self.rmse_per_component
        )
        np_pls_alg_2_rmses = np_pls_alg_2.cross_validate(
            X, Y, n_components, splits.flatten(), self.rmse_per_component
        )

        # Calibrate FastCV NumPy PLS
        fast_cv_np_pls_alg_1 = FastCVPLS(algorithm=1)
        fast_cv_np_pls_alg_2 = FastCVPLS(algorithm=2)
        fast_cv_np_pls_alg_1_results = fast_cv_np_pls_alg_1.cross_validate(
            X, Y, n_components, splits.flatten(), self.rmse_per_component
        )
        fast_cv_np_pls_alg_2_results = fast_cv_np_pls_alg_2.cross_validate(
            X, Y, n_components, splits.flatten(), self.rmse_per_component
        )

        # Convert the results from dict to list for easier comparison
        fast_cv_np_pls_alg_1_results = [
            np.asarray(value) for value in fast_cv_np_pls_alg_1_results.values()
        ]
        fast_cv_np_pls_alg_2_results = [
            np.asarray(value) for value in fast_cv_np_pls_alg_2_results.values()
        ]

        # Sort fast cv results according to the unique splits for comparison with the
        # other algorithms
        unique_splits, sort_indices = np.unique(splits, return_index=True)
        unique_splits = unique_splits.astype(int)
        fast_cv_order = np.argsort(sort_indices)
        other_alg_order = np.argsort(fast_cv_order)
        fast_cv_np_pls_alg_1_results = np.array(fast_cv_np_pls_alg_1_results)[
            other_alg_order
        ]
        fast_cv_np_pls_alg_2_results = np.array(fast_cv_np_pls_alg_2_results)[
            other_alg_order
        ]

        # Calibrate JAX PLS
        jax_pls_alg_1_results = jax_pls_alg_1.cross_validate(
            X,
            Y,
            n_components,
            jnp_splits,
            self.jax_rmse_per_component,
            ["RMSE"],
        )
        diff_jax_pls_alg_1_results = diff_jax_pls_alg_1.cross_validate(
            X,
            Y,
            n_components,
            jnp_splits,
            self.jax_rmse_per_component,
            ["RMSE"],
        )
        jax_pls_alg_2_results = jax_pls_alg_2.cross_validate(
            X,
            Y,
            n_components,
            jnp_splits,
            self.jax_rmse_per_component,
            ["RMSE"],
        )
        diff_jax_pls_alg_2_results = diff_jax_pls_alg_2.cross_validate(
            X,
            Y,
            n_components,
            jnp_splits,
            self.jax_rmse_per_component,
            ["RMSE"],
        )

        # Get the best number of components in terms of minimizing validation RMSE for
        # each split is equal among all algorithms
        unique_splits = np.unique(splits).astype(int)
        sk_best_num_components = [
            [np.argmin(sk_pls_rmses[split][..., i]) for split in unique_splits]
            for i in range(M)
        ]
        np_pls_alg_1_best_num_components = [
            [np.argmin(np_pls_alg_1_rmses[split][..., i]) for split in unique_splits]
            for i in range(M)
        ]
        np_pls_alg_2_best_num_components = [
            [np.argmin(np_pls_alg_2_rmses[split][..., i]) for split in unique_splits]
            for i in range(M)
        ]
        fast_cv_np_pls_alg_1_best_num_components = [
            [
                np.argmin(fast_cv_np_pls_alg_1_results[split][..., i])
                for split in unique_splits
            ]
            for i in range(M)
        ]
        fast_cv_np_pls_alg_2_best_num_components = [
            [
                np.argmin(fast_cv_np_pls_alg_2_results[split][..., i])
                for split in unique_splits
            ]
            for i in range(M)
        ]
        jax_pls_alg_1_best_num_components = [
            [
                np.argmin(jax_pls_alg_1_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(M)
        ]
        jax_pls_alg_2_best_num_components = [
            [
                np.argmin(jax_pls_alg_2_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(M)
        ]
        diff_jax_pls_alg_1_best_num_components = [
            [
                np.argmin(diff_jax_pls_alg_1_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(M)
        ]
        diff_jax_pls_alg_2_best_num_components = [
            [
                np.argmin(diff_jax_pls_alg_2_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(M)
        ]

        # Check that the RMSE achieved by the best number of components is similar
        sk_best_rmses = [
            [
                sk_pls_rmses[split][sk_best_num_components[i][split], i]
                for split in unique_splits
            ]
            for i in range(M)
        ]
        np_pls_alg_1_best_rmses = [
            [
                np_pls_alg_1_rmses[split][np_pls_alg_1_best_num_components[i][split], i]
                for split in unique_splits
            ]
            for i in range(M)
        ]
        np_pls_alg_2_best_rmses = [
            [
                np_pls_alg_2_rmses[split][np_pls_alg_2_best_num_components[i][split], i]
                for split in unique_splits
            ]
            for i in range(M)
        ]
        fast_cv_np_pls_alg_1_best_rmses = [
            [
                fast_cv_np_pls_alg_1_results[split][
                    fast_cv_np_pls_alg_1_best_num_components[i][split], i
                ]
                for split in unique_splits
            ]
            for i in range(M)
        ]
        fast_cv_np_pls_alg_2_best_rmses = [
            [
                fast_cv_np_pls_alg_2_results[split][
                    fast_cv_np_pls_alg_2_best_num_components[i][split], i
                ]
                for split in unique_splits
            ]
            for i in range(M)
        ]
        jax_pls_alg_1_best_rmses = [
            [
                jax_pls_alg_1_results["RMSE"][split][
                    jax_pls_alg_1_best_num_components[i][split], i
                ]
                for split in unique_splits
            ]
            for i in range(M)
        ]
        jax_pls_alg_2_best_rmses = [
            [
                jax_pls_alg_2_results["RMSE"][split][
                    jax_pls_alg_2_best_num_components[i][split], i
                ]
                for split in unique_splits
            ]
            for i in range(M)
        ]
        diff_jax_pls_alg_1_best_rmses = [
            [
                diff_jax_pls_alg_1_results["RMSE"][split][
                    diff_jax_pls_alg_1_best_num_components[i][split], i
                ]
                for split in unique_splits
            ]
            for i in range(M)
        ]
        diff_jax_pls_alg_2_best_rmses = [
            [
                diff_jax_pls_alg_2_results["RMSE"][split][
                    diff_jax_pls_alg_2_best_num_components[i][split], i
                ]
                for split in unique_splits
            ]
            for i in range(M)
        ]

        assert_allclose(np_pls_alg_1_best_rmses, sk_best_rmses, atol=atol, rtol=rtol)
        assert_allclose(np_pls_alg_2_best_rmses, sk_best_rmses, atol=atol, rtol=rtol)
        assert_allclose(
            fast_cv_np_pls_alg_1_best_rmses, sk_best_rmses, atol=atol, rtol=rtol
        )
        assert_allclose(
            fast_cv_np_pls_alg_2_best_rmses, sk_best_rmses, atol=atol, rtol=rtol
        )
        assert_allclose(jax_pls_alg_1_best_rmses, sk_best_rmses, atol=atol, rtol=rtol)
        assert_allclose(jax_pls_alg_2_best_rmses, sk_best_rmses, atol=atol, rtol=rtol)
        assert_allclose(
            diff_jax_pls_alg_1_best_rmses, sk_best_rmses, atol=atol, rtol=rtol
        )
        assert_allclose(
            diff_jax_pls_alg_2_best_rmses, sk_best_rmses, atol=atol, rtol=rtol
        )

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def test_cross_val_pls_1(self):
        """
        Description
        -----------
        This test loads input predictor variables, a single target variable, and split
        indices for cross-validation. It then calls the 'check_cross_val_pls' method to
        validate the cross-validation results, specifically for a single target
        variable.

        Returns:
        None
        """
        X = self.load_X()
        X = X[..., :3]
        Y = self.load_Y(["Protein"])
        splits = self.load_Y(["split"])  # Contains 3 splits of different sizes
        assert Y.shape[1] == 1
        self.check_cross_val_pls(X, Y, splits, atol=0, rtol=1e-5)

        # Remove the singleton dimension and check that the predictions are consistent.
        Y = Y.squeeze()
        assert Y.ndim == 1
        self.check_cross_val_pls(X, Y, splits, atol=0, rtol=1e-5)

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def test_cross_val_pls_2_m_less_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is less than K), and split indices for cross-validation. It then calls the
        'check_cross_val_pls' method to validate the cross-validation results for this
        scenario.

        Returns:
        None
        """
        X = self.load_X()
        X = X[..., :3]
        Y = self.load_Y(
            [
                "Moisture",
                "Protein",
            ]
        )
        splits = self.load_Y(["split"])  # Contains 3 splits of different sizes
        assert Y.shape[1] > 1
        assert Y.shape[1] < X.shape[1]
        self.check_cross_val_pls(X, Y, splits, atol=0, rtol=2e-4)

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def test_cross_val_pls_2_m_eq_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is equal to K), and split indices for cross-validation. It then calls the
        'check_cross_val_pls' method to validate the cross-validation results for this
        scenario.

        Returns:
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Moisture",
                "Protein",
            ]
        )
        splits = self.load_Y(["split"])  # Contains 3 splits of different sizes
        X = X[..., :2]
        assert Y.shape[1] > 1
        assert Y.shape[1] == X.shape[1]
        self.check_cross_val_pls(X, Y, splits, atol=0, rtol=3.4e-5)

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def test_cross_val_pls_2_m_greater_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is greater than K), and split indices for cross-validation. It then calls the
        'check_cross_val_pls' method to validate the cross-validation results for this
        scenario.

        Returns:
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
            ]
        )
        splits = self.load_Y(["split"])  # Contains 3 splits of different sizes
        X = X[..., :3]
        assert Y.shape[1] > 1
        assert Y.shape[1] > X.shape[1]
        self.check_cross_val_pls(X, Y, splits, atol=0, rtol=3e-5)

    def check_fast_cross_val_pls(self, X, Y, splits, center, scale, atol, rtol):
        """
        Description
        -----------
        This method tests the ability to perform cross-validation to obtain the root
        mean square error (RMSE) and the best number of components for each target
        variable and each split. It tests the fast cross-validation algorithm against
        the ordinary cross-validation algorithm.

        Parameters:
        X : numpy.ndarray
            The input predictor variables.
        Y : numpy.ndarray
            The target variables.
        splits : numpy.ndarray
            Split indices for cross-validation.

        center : bool
            Whether or not to center the data before performing PLS.

        scale : bool
            Whether or not to scale the data before performing PLS.

        atol : float
            Absolute tolerance for value comparisons.

        rtol : float
            Relative tolerance for value comparisons.

        Returns:
        None

        Raises
        ------
        AssertionError
            If the best number of components found by cross validation with is not
            exactly equal across each different PLS implementation.

            If the output RMSEs for the best number of components are not equal down to
            the specified tolerance across each different PLS implementation.
        """

        try:
            M = Y.shape[1]
        except IndexError:
            M = 1

        np_pls_alg_1 = NpPLS(
            algorithm=1, center_X=center, center_Y=center, scale_X=scale, scale_Y=scale
        )
        np_pls_alg_2 = NpPLS(
            algorithm=2, center_X=center, center_Y=center, scale_X=scale, scale_Y=scale
        )
        fast_cv_np_pls_alg_1 = FastCVPLS(
            algorithm=1, center_X=center, center_Y=center, scale_X=scale, scale_Y=scale
        )
        fast_cv_np_pls_alg_2 = FastCVPLS(
            algorithm=2, center_X=center, center_Y=center, scale_X=scale, scale_Y=scale
        )

        n_components = X.shape[1]

        np_pls_alg_1_rmses = np_pls_alg_1.cross_validate(
            X, Y, n_components, splits.flatten(), self.rmse_per_component
        )

        np_pls_alg_2_rmses = np_pls_alg_2.cross_validate(
            X, Y, n_components, splits.flatten(), self.rmse_per_component
        )

        # Compute RMSE on the validation predictions using the fast cross-validation
        # algorithm
        fast_cv_np_pls_alg_1_results = fast_cv_np_pls_alg_1.cross_validate(
            X=X,
            Y=Y,
            A=n_components,
            cv_splits=splits.flatten(),
            metric_function=self.rmse_per_component,
            n_jobs=-1,
            verbose=0,
        )
        fast_cv_np_pls_alg_2_results = fast_cv_np_pls_alg_2.cross_validate(
            X=X,
            Y=Y,
            A=n_components,
            cv_splits=splits.flatten(),
            metric_function=self.rmse_per_component,
            n_jobs=-1,
            verbose=0,
        )

        # Convert the results from dict to list for easier comparison
        fast_cv_np_pls_alg_1_results = [
            np.asarray(value) for value in fast_cv_np_pls_alg_1_results.values()
        ]
        fast_cv_np_pls_alg_2_results = [
            np.asarray(value) for value in fast_cv_np_pls_alg_2_results.values()
        ]

        # Sort fast cv results according to the unique splits for comparison with the
        # other algorithms
        unique_splits, sort_indices = np.unique(splits, return_index=True)
        unique_splits = unique_splits.astype(int)
        fast_cv_order = np.argsort(sort_indices)
        other_alg_order = np.argsort(fast_cv_order)
        fast_cv_np_pls_alg_1_results = np.array(fast_cv_np_pls_alg_1_results)[
            other_alg_order
        ]
        fast_cv_np_pls_alg_2_results = np.array(fast_cv_np_pls_alg_2_results)[
            other_alg_order
        ]

        # Check that best number of components in terms of minimizing validation RMSE
        # for each split is equal among all algorithms
        np_pls_alg_1_best_num_components = [
            [np.argmin(np_pls_alg_1_rmses[split][..., i]) for split in unique_splits]
            for i in range(M)
        ]
        np_pls_alg_2_best_num_components = [
            [np.argmin(np_pls_alg_2_rmses[split][..., i]) for split in unique_splits]
            for i in range(M)
        ]
        fast_cv_np_pls_alg_1_best_num_components = [
            [
                np.argmin(np.array(fast_cv_np_pls_alg_1_results)[split][..., i])
                for split in unique_splits
            ]
            for i in range(M)
        ]
        fast_cv_np_pls_alg_2_best_num_components = [
            [
                np.argmin(np.array(fast_cv_np_pls_alg_2_results)[split][..., i])
                for split in unique_splits
            ]
            for i in range(M)
        ]

        assert np_pls_alg_1_best_num_components == np_pls_alg_2_best_num_components
        assert (
            np_pls_alg_1_best_num_components == fast_cv_np_pls_alg_1_best_num_components
        )
        assert (
            np_pls_alg_2_best_num_components == fast_cv_np_pls_alg_2_best_num_components
        )

        # Check that the RMSE achieved is similar
        np_pls_alg_1_best_rmses = [
            [np.amin(np_pls_alg_1_rmses[split][..., i]) for split in unique_splits]
            for i in range(M)
        ]
        np_pls_alg_2_best_rmses = [
            [np.amin(np_pls_alg_2_rmses[split][..., i]) for split in unique_splits]
            for i in range(M)
        ]
        fast_cv_np_pls_alg_1_best_rmses = [
            [
                np.amin(np.array(fast_cv_np_pls_alg_1_results)[split][..., i])
                for split in unique_splits
            ]
            for i in range(M)
        ]
        fast_cv_np_pls_alg_2_best_rmses = [
            [
                np.amin(np.array(fast_cv_np_pls_alg_2_results)[split][..., i])
                for split in unique_splits
            ]
            for i in range(M)
        ]

        assert_allclose(
            np_pls_alg_1_best_rmses, np_pls_alg_2_best_rmses, atol=atol, rtol=rtol
        )
        assert_allclose(
            np_pls_alg_1_best_rmses,
            fast_cv_np_pls_alg_1_best_rmses,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np_pls_alg_2_best_rmses,
            fast_cv_np_pls_alg_2_best_rmses,
            atol=atol,
            rtol=rtol,
        )

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def test_fast_cross_val_pls_1(self):
        """
        Description
        -----------
        This test loads input predictor variables, a single target variable, and split
        indices for cross-validation. It then calls the 'check_fast_cross_val_pls'
        method to validate the cross-validation results, specifically for a single
        target variable.

        Returns:
        None
        """
        X = self.load_X()
        X = X[..., :3]
        Y = self.load_Y(["Protein"])
        splits = self.load_Y(["split"])
        assert Y.shape[1] == 1
        self.check_fast_cross_val_pls(
            X, Y, splits, center=False, scale=False, atol=0, rtol=1e-7
        )
        self.check_fast_cross_val_pls(
            X, Y, splits, center=True, scale=False, atol=0, rtol=1e-7
        )
        self.check_fast_cross_val_pls(
            X, Y, splits, center=True, scale=True, atol=0, rtol=1e-7
        )

        # Remove the singleton dimension and check that the predictions are consistent.
        Y = Y.squeeze()
        assert Y.ndim == 1
        self.check_fast_cross_val_pls(
            X, Y, splits, center=False, scale=False, atol=0, rtol=1e-7
        )

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def test_fast_cross_val_pls_2_m_less_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is less than K), and split indices for cross-validation. It then calls the
        'check_fast_cross_val_pls' method to validate the cross-validation results for
        this scenario.

        Returns:
        None
        """
        X = self.load_X()
        X = X[..., :3]
        Y = self.load_Y(
            [
                "Moisture",
                "Protein",
            ]
        )
        splits = self.load_Y(["split"])
        assert Y.shape[1] > 1
        assert Y.shape[1] < X.shape[1]
        self.check_fast_cross_val_pls(
            X, Y, splits, center=False, scale=False, atol=0, rtol=1e-6
        )
        self.check_fast_cross_val_pls(
            X, Y, splits, center=True, scale=False, atol=0, rtol=1e-6
        )
        self.check_fast_cross_val_pls(
            X, Y, splits, center=True, scale=True, atol=0, rtol=1e-6
        )

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def test_fast_cross_val_pls_2_m_eq_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is equal to K), and split indices for cross-validation. It then calls the
        'check_fast_cross_val_pls' method to validate the cross-validation results for
        this scenario.

        Returns:
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Moisture",
                "Protein",
            ]
        )
        splits = self.load_Y(["split"])
        X = X[..., :2]
        assert Y.shape[1] > 1
        assert Y.shape[1] == X.shape[1]
        self.check_fast_cross_val_pls(
            X, Y, splits, center=False, scale=False, atol=0, rtol=2e-8
        )
        self.check_fast_cross_val_pls(
            X, Y, splits, center=True, scale=False, atol=0, rtol=2e-8
        )
        self.check_fast_cross_val_pls(
            X, Y, splits, center=True, scale=True, atol=0, rtol=2e-8
        )

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def test_fast_cross_val_pls_2_m_greater_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is greater than K), and split indices for cross-validation. It then calls the
        'check_fast_cross_val_pls' method to validate the cross-validation results for
        this scenario.

        Returns:
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
            ]
        )
        splits = self.load_Y(["split"])
        X = X[..., :3]
        assert Y.shape[1] > 1
        assert Y.shape[1] > X.shape[1]
        self.check_fast_cross_val_pls(
            X, Y, splits, center=False, scale=False, atol=0, rtol=1e-8
        )
        self.check_fast_cross_val_pls(
            X, Y, splits, center=True, scale=False, atol=0, rtol=1e-8
        )
        self.check_fast_cross_val_pls(
            X, Y, splits, center=True, scale=True, atol=0, rtol=1e-8
        )

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def check_center_scale_combinations(self, X, Y, splits, atol, rtol):
        """
        Description
        -----------
        This method tests the ability to perform cross-validation to obtain the root
        mean square error (RMSE) and the best number of components for each target
        variable and each split. It tests the fast cross-validation algorithm against
        the ordinary cross-validation algorithm for all possible combinations of
        centering and scaling.

        Parameters:
        X : numpy.ndarray
            The input predictor variables.
        Y : numpy.ndarray
            The target variables.
        splits : numpy.ndarray
            Split indices for cross-validation.

        atol : float
            Absolute tolerance for value comparisons.

        rtol : float
            Relative tolerance for value comparisons.

        Returns:
        None

        Raises
        ------
        AssertionError
            If the best number of components found by cross validation with is not
            exactly equal across each different PLS implementation.

            If the output RMSEs for the best number of components are not equal down to
            the specified tolerance across each different PLS implementation.
        """

        center_Xs = [False, True]
        center_Ys = [False, True]
        scale_Xs = [False, True]
        scale_Ys = [False, True]

        center_scale_combinations = product(center_Xs, center_Ys, scale_Xs, scale_Ys)

        try:
            M = Y.shape[1]
        except IndexError:
            M = 1

        jnp_splits = jnp.array(splits)
        # Sort fast cv results according to the unique splits for comparison with
        # the other algorithms
        unique_splits, sort_indices = np.unique(splits, return_index=True)
        unique_splits = unique_splits.astype(int)
        fast_cv_order = np.argsort(sort_indices)
        other_alg_order = np.argsort(fast_cv_order)

        largest_split = np.max(
            [np.count_nonzero(splits == split) for split in unique_splits]
        )
        n_components = min(X.shape[1], largest_split)

        for center_X, center_Y, scale_X, scale_Y in center_scale_combinations:
            (
                np_pls_alg_1,
                np_pls_alg_2,
                jax_pls_alg_1,
                jax_pls_alg_2,
                diff_jax_pls_alg_1,
                diff_jax_pls_alg_2,
                fast_cv_np_pls_alg_1,
                fast_cv_np_pls_alg_2,
            ) = self.get_models(
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
                fast_cv=True,
            )

            np_pls_alg_1_rmses = np_pls_alg_1.cross_validate(
                X, Y, n_components, splits, self.rmse_per_component
            )
            np_pls_alg_2_rmses = np_pls_alg_2.cross_validate(
                X, Y, n_components, splits, self.rmse_per_component
            )

            # Compute RMSE on the validation predictions using the fast cross-validation
            # algorithm
            fast_cv_np_pls_alg_1_results = fast_cv_np_pls_alg_1.cross_validate(
                X=X,
                Y=Y,
                A=n_components,
                cv_splits=splits,
                metric_function=self.rmse_per_component,
                n_jobs=-1,
                verbose=0,
            )
            fast_cv_np_pls_alg_2_results = fast_cv_np_pls_alg_2.cross_validate(
                X=X,
                Y=Y,
                A=n_components,
                cv_splits=splits,
                metric_function=self.rmse_per_component,
                n_jobs=-1,
                verbose=0,
            )

            # Convert the results from dict to list for easier comparison
            fast_cv_np_pls_alg_1_results = [
                np.asarray(value) for value in fast_cv_np_pls_alg_1_results.values()
            ]
            fast_cv_np_pls_alg_2_results = [
                np.asarray(value) for value in fast_cv_np_pls_alg_2_results.values()
            ]

            fast_cv_np_pls_alg_1_results = np.array(fast_cv_np_pls_alg_1_results)[
                other_alg_order
            ]
            fast_cv_np_pls_alg_2_results = np.array(fast_cv_np_pls_alg_2_results)[
                other_alg_order
            ]

            # Calibrate JAX PLS
            jax_pls_alg_1_results = jax_pls_alg_1.cross_validate(
                X,
                Y,
                n_components,
                jnp_splits,
                self.jax_rmse_per_component,
                ["RMSE"],
            )
            diff_jax_pls_alg_1_results = diff_jax_pls_alg_1.cross_validate(
                X,
                Y,
                n_components,
                jnp_splits,
                self.jax_rmse_per_component,
                ["RMSE"],
            )
            jax_pls_alg_2_results = jax_pls_alg_2.cross_validate(
                X,
                Y,
                n_components,
                jnp_splits,
                self.jax_rmse_per_component,
                ["RMSE"],
            )
            diff_jax_pls_alg_2_results = diff_jax_pls_alg_2.cross_validate(
                X,
                Y,
                n_components,
                jnp_splits,
                self.jax_rmse_per_component,
                ["RMSE"],
            )

            # Get the best number of components in terms of minimizing validation RMSE
            # for each split is equal among all algorithms
            unique_splits = np.unique(splits).astype(int)
            np_pls_alg_1_best_num_components = [
                [
                    np.argmin(np_pls_alg_1_rmses[split][..., i])
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            np_pls_alg_2_best_num_components = [
                [
                    np.argmin(np_pls_alg_2_rmses[split][..., i])
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            fast_cv_np_pls_alg_1_best_num_components = [
                [
                    np.argmin(fast_cv_np_pls_alg_1_results[split][..., i])
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            fast_cv_np_pls_alg_2_best_num_components = [
                [
                    np.argmin(fast_cv_np_pls_alg_2_results[split][..., i])
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            jax_pls_alg_1_best_num_components = [
                [
                    np.argmin(jax_pls_alg_1_results["RMSE"][split][..., i])
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            jax_pls_alg_2_best_num_components = [
                [
                    np.argmin(jax_pls_alg_2_results["RMSE"][split][..., i])
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            diff_jax_pls_alg_1_best_num_components = [
                [
                    np.argmin(diff_jax_pls_alg_1_results["RMSE"][split][..., i])
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            diff_jax_pls_alg_2_best_num_components = [
                [
                    np.argmin(diff_jax_pls_alg_2_results["RMSE"][split][..., i])
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            np_pls_alg_1_best_rmses = [
                [
                    np_pls_alg_1_rmses[split][
                        np_pls_alg_1_best_num_components[i][split], i
                    ]
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            np_pls_alg_2_best_rmses = [
                [
                    np_pls_alg_2_rmses[split][
                        np_pls_alg_2_best_num_components[i][split], i
                    ]
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            fast_cv_np_pls_alg_1_best_rmses = [
                [
                    fast_cv_np_pls_alg_1_results[split][
                        fast_cv_np_pls_alg_1_best_num_components[i][split], i
                    ]
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            fast_cv_np_pls_alg_2_best_rmses = [
                [
                    fast_cv_np_pls_alg_2_results[split][
                        fast_cv_np_pls_alg_2_best_num_components[i][split], i
                    ]
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            jax_pls_alg_1_best_rmses = [
                [
                    jax_pls_alg_1_results["RMSE"][split][
                        jax_pls_alg_1_best_num_components[i][split], i
                    ]
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            jax_pls_alg_2_best_rmses = [
                [
                    jax_pls_alg_2_results["RMSE"][split][
                        jax_pls_alg_2_best_num_components[i][split], i
                    ]
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            diff_jax_pls_alg_1_best_rmses = [
                [
                    diff_jax_pls_alg_1_results["RMSE"][split][
                        diff_jax_pls_alg_1_best_num_components[i][split], i
                    ]
                    for split in unique_splits
                ]
                for i in range(M)
            ]
            diff_jax_pls_alg_2_best_rmses = [
                [
                    diff_jax_pls_alg_2_results["RMSE"][split][
                        diff_jax_pls_alg_2_best_num_components[i][split], i
                    ]
                    for split in unique_splits
                ]
                for i in range(M)
            ]

            err_msg = f"Center_X: {center_X}, Center_Y: {center_Y}, Scale_X: {scale_X}, Scale_Y: {scale_Y}"
            assert_allclose(
                np_pls_alg_2_best_rmses,
                np_pls_alg_1_best_rmses,
                atol=atol,
                rtol=rtol,
                err_msg=err_msg,
            )
            assert_allclose(
                fast_cv_np_pls_alg_1_best_rmses,
                np_pls_alg_1_best_rmses,
                atol=atol,
                rtol=rtol,
                err_msg=err_msg,
            )
            assert_allclose(
                fast_cv_np_pls_alg_2_best_rmses,
                np_pls_alg_1_best_rmses,
                atol=atol,
                rtol=rtol,
                err_msg=err_msg,
            )
            assert_allclose(
                jax_pls_alg_1_best_rmses,
                np_pls_alg_1_best_rmses,
                atol=atol,
                rtol=rtol,
                err_msg=err_msg,
            )
            assert_allclose(
                jax_pls_alg_2_best_rmses,
                np_pls_alg_1_best_rmses,
                atol=atol,
                rtol=rtol,
                err_msg=err_msg,
            )
            assert_allclose(
                diff_jax_pls_alg_1_best_rmses,
                np_pls_alg_1_best_rmses,
                atol=atol,
                rtol=rtol,
                err_msg=err_msg,
            )
            assert_allclose(
                diff_jax_pls_alg_2_best_rmses,
                np_pls_alg_1_best_rmses,
                atol=atol,
                rtol=rtol,
                err_msg=err_msg,
            )

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def test_center_scale_combinations_pls_1(self):
        """
        Description
        -----------
        This test loads input predictor variables, a single target variable, and split
        indices for cross-validation. It then calls the
        `check_center_scale_combinations` method to validate the cross-validation
        results for all possible combinations of centering and scaling.

        Returns:
        None
        """
        X = self.load_X()
        X = X[..., :3]  # Decrease the amount of features in the interest of time.
        Y = self.load_Y(["Protein"])
        splits = (
            self.load_Y(["split"]).flatten().astype(np.int64)
        )  # Contains 3 splits of different sizes
        # Decrease the amount of samples in the interest of time.
        X = X[::50]
        Y = Y[::50]
        splits = splits[::50]
        assert Y.shape[1] == 1
        self.check_center_scale_combinations(X, Y, splits, atol=0, rtol=1e-8)

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    # We are probably fitting too many components here. But we do not care.
    @pytest.mark.filterwarnings(
        "ignore", category=UserWarning, match="Weight is close to zero"
    )
    def test_center_scale_combinations_pls_1_n_less_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, a single target variable, and split
        indices for cross-validation. It uses a quadratic X matrix. It then calls the
        `check_center_scale_combinations` method to validate the cross-validation
        results for all possible combinations of centering and scaling.

        Returns:
        None
        """
        X = self.load_X()
        Y = self.load_Y(["Protein"])
        splits = (
            self.load_Y(["split"]).flatten().astype(np.int64)
        )  # Contains 3 splits of different sizes
        step_size_row = 1000
        X = X[::step_size_row]
        Y = Y[::step_size_row]
        splits = splits[::step_size_row]
        assert Y.shape[1] == 1
        assert X.shape[0] < X.shape[1]
        self.check_center_scale_combinations(X, Y, splits, atol=0, rtol=0.15)

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    # We are probably fitting too many components here. But we do not care.
    @pytest.mark.filterwarnings(
        "ignore", category=UserWarning, match="Weight is close to zero"
    )
    def test_center_scale_combinations_pls_1_n_eq_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, a single target variable, and split
        indices for cross-validation. It uses a quadratic X matrix. It then calls the
        `check_center_scale_combinations` method to validate the cross-validation
        results for all possible combinations of centering and scaling.

        Returns:
        None
        """
        X = self.load_X()
        Y = self.load_Y(["Protein"])
        splits = (
            self.load_Y(["split"]).flatten().astype(np.int64)
        )  # Contains 3 splits of different sizes
        min_x_dim = min(X.shape)
        min_x_dim = min(min_x_dim, 50)
        step_size_row = X.shape[0] // min_x_dim
        step_size_col = X.shape[1] // min_x_dim
        X = X[
            : step_size_row * min_x_dim : step_size_row,
            : step_size_col * min_x_dim : step_size_col,
        ]
        Y = Y[: step_size_row * min_x_dim : step_size_row]
        splits = splits[: step_size_row * min_x_dim : step_size_row]
        assert Y.shape[1] == 1
        assert X.shape[0] == X.shape[1]
        self.check_center_scale_combinations(X, Y, splits, atol=0, rtol=0.2)

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def test_center_scale_combinations_pls_2_m_less_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is less than K), and split indices for cross-validation. It then calls the
        `check_center_scale_combinations` method to validate the cross-validation
        results for all possible combinations of centering and scaling.

        Returns:
        None
        """
        X = self.load_X()
        X = X[..., :3]  # Decrease the amount of features in the interest of time.
        Y = self.load_Y(
            [
                "Moisture",
                "Protein",
            ]
        )
        splits = (
            self.load_Y(["split"]).flatten().astype(np.int64)
        )  # Contains 3 splits of different sizes
        # Decrease the amount of samples in the interest of time.
        X = X[::50]
        Y = Y[::50]
        splits = splits[::50]
        assert Y.shape[1] > 1
        assert Y.shape[1] < X.shape[1]
        self.check_center_scale_combinations(X, Y, splits, atol=0, rtol=1e-7)

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def test_center_scale_combinations_pls_2_m_eq_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is equal to K), and split indices for cross-validation. It then calls the
        `check_center_scale_combinations` method to validate the cross-validation
        results for all possible combinations of centering and scaling.

        Returns:
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Moisture",
                "Protein",
            ]
        )
        X = X[..., :2]
        splits = (
            self.load_Y(["split"]).flatten().astype(np.int64)
        )  # Contains 3 splits of different sizes
        # Decrease the amount of samples in the interest of time.
        X = X[::50]
        Y = Y[::50]
        splits = splits[::50]
        assert Y.shape[1] > 1
        assert Y.shape[1] == X.shape[1]
        self.check_center_scale_combinations(X, Y, splits, atol=0, rtol=3e-8)

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def test_center_scale_combinations_pls_2_m_greater_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is greater than K), and split indices for cross-validation. It then calls the
        `check_center_scale_combinations` method to validate the cross-validation
        results for all possible combinations of centering and scaling.

        Returns:
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
            ]
        )
        X = X[..., :3]
        splits = (
            self.load_Y(["split"]).flatten().astype(np.int64)
        )  # Contains 3 splits of different sizes
        # Decrease the amount of samples in the interest of time.
        X = X[::50]
        Y = Y[::50]
        splits = splits[::50]
        assert Y.shape[1] > 1
        assert Y.shape[1] > X.shape[1]
        self.check_center_scale_combinations(X, Y, splits, atol=0, rtol=1e-8)

    def check_wpls_max_components_eq_to_wls(
        self,
        pls_preds,
        wls_pred,
        pls_Bs,
        wls_B,
        pred_atol,
        pred_rtol,
        B_atol,
        B_rtol,
        **kwargs,
    ):
        """
        Description
        -----------
        This method checks whether weighted PLS with the maximum number of components yields the
        same results as weighted least squares.

        Parameters
        ----------
        X : numpy.ndarray
            The input predictor variables.
        Y : numpy.ndarray
            The target variables.
        pls_preds : list[numpy.ndarray]
            The predictions from the weighted PLS algorithms.
        wls_pred : numpy.ndarray
            The predictions from the weighted least squares
        atol : float
            Absolute tolerance for value comparisons.
        rtol : float
            Relative tolerance for value comparisons.
        **kwargs : dict
            Additional keyword arguments. Will be printed in the error message if the
            test fails.

        Returns
        -------
        None
        """
        err_msg = f"Additional keyword arguments: {kwargs}"
        for i, pred in enumerate(pls_preds):
            local_err_msg = f"Model: {i}, {err_msg}"
            assert_allclose(
                pred,
                wls_pred,
                atol=pred_atol,
                rtol=pred_rtol,
                err_msg=local_err_msg,
            )
        for B in pls_Bs:
            assert_allclose(B, wls_B, atol=B_atol, rtol=B_rtol, err_msg=err_msg)

    def check_wpls_constant_weights(
        self,
        X,
        Y,
        pred_atol,
        pred_rtol,
        B_atol,
        B_rtol,
        wls_pred_atol,
        wls_pred_rtol,
        wls_B_atol,
        wls_B_rtol,
    ):
        center_Xs = [False, True]
        center_Ys = [False, True]
        scale_Xs = [False, True]
        scale_Ys = [False, True]
        center_scale_combinations = product(center_Xs, center_Ys, scale_Xs, scale_Ys)
        n_components = min(X.shape[0], X.shape[1])
        for center_X, center_Y, scale_X, scale_Y in center_scale_combinations:
            weights_one = np.ones(X.shape[0])
            weights_point_five = np.full(X.shape[0], 0.5)
            weights_one_pls_models = self.fit_models(
                X=X,
                Y=Y,
                n_components=n_components,
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
                weights=weights_one,
            )
            weights_point_five_pls_models = self.fit_models(
                X=X,
                Y=Y,
                n_components=n_components,
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
                weights=weights_point_five,
            )
            no_weights_pls_models = self.fit_models(
                X=X,
                Y=Y,
                n_components=n_components,
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
                weights=None,
                return_sk_pls=False,
            )
            weights_one_preds, weights_one_Bs = self.get_model_preds(
                X, weights_one_pls_models, last_only=True
            )
            weights_point_five_preds, weights_point_five_Bs = self.get_model_preds(
                X, weights_point_five_pls_models, last_only=True
            )
            no_weights_preds, no_weights_Bs = self.get_model_preds(
                X, no_weights_pls_models, last_only=True
            )
            weights_one_wls_pred, weights_one_wls_B = self.fit_predict_wls(
                X=X,
                Y=Y,
                weights=weights_one,
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
            )
            weights_point_five_wls_pred, weights_point_five_wls_B = (
                self.fit_predict_wls(
                    X=X,
                    Y=Y,
                    weights=weights_point_five,
                    center_X=center_X,
                    center_Y=center_Y,
                    scale_X=scale_X,
                    scale_Y=scale_Y,
                )
            )
            ols_pred, ols_B = self.fit_predict_ols(
                X=X,
                Y=Y,
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
            )

            # Check that all are equal
            err_msg = f"Center_X: {center_X}, Center_Y: {center_Y}, Scale_X: {scale_X}, Scale_Y: {scale_Y}"
            assert_allclose(
                weights_one_wls_pred,
                ols_pred,
                atol=pred_atol,
                rtol=pred_rtol,
                err_msg=err_msg,
            )
            assert_allclose(
                weights_one_wls_B,
                ols_B,
                atol=pred_atol,
                rtol=pred_rtol,
                err_msg=err_msg,
            )
            assert_allclose(
                weights_point_five_wls_pred,
                ols_pred,
                atol=pred_atol,
                rtol=pred_rtol,
                err_msg=err_msg,
            )
            assert_allclose(
                weights_point_five_wls_B,
                ols_B,
                atol=pred_atol,
                rtol=pred_rtol,
                err_msg=err_msg,
            )

            for pred, B in zip(weights_one_preds, weights_one_Bs):
                assert_allclose(
                    pred, ols_pred, atol=pred_atol, rtol=pred_rtol, err_msg=err_msg
                )
                assert_allclose(B, ols_B, atol=B_atol, rtol=B_rtol, err_msg=err_msg)
            for pred, B in zip(weights_point_five_preds, weights_point_five_Bs):
                assert_allclose(
                    pred, ols_pred, atol=pred_atol, rtol=pred_rtol, err_msg=err_msg
                )
                assert_allclose(B, ols_B, atol=B_atol, rtol=B_rtol, err_msg=err_msg)
            for pred, B in zip(no_weights_preds, no_weights_Bs):
                assert_allclose(
                    pred, ols_pred, atol=pred_atol, rtol=pred_rtol, err_msg=err_msg
                )
                assert_allclose(B, ols_B, atol=B_atol, rtol=B_rtol, err_msg=err_msg)

            self.check_wpls_max_components_eq_to_wls(
                pls_preds=weights_one_preds,
                wls_pred=weights_one_wls_pred,
                pls_Bs=weights_one_Bs,
                wls_B=weights_one_wls_B,
                pred_atol=wls_pred_atol,
                pred_rtol=wls_pred_rtol,
                B_atol=wls_B_atol,
                B_rtol=wls_B_rtol,
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
            )

    def check_wpls_zero_weights(
        self,
        X,
        Y,
        weights,
        pred_atol,
        pred_rtol,
        B_atol,
        B_rtol,
        wls_pred_atol,
        wls_pred_rtol,
        wls_B_atol,
        wls_B_rtol,
    ):
        center_Xs = [False, True]
        center_Ys = [False, True]
        scale_Xs = [False, True]
        scale_Ys = [False, True]
        center_scale_combinations = product(center_Xs, center_Ys, scale_Xs, scale_Ys)
        # Set every 10th weight to zero
        weights_zero = weights.copy()
        weights_zero[::10] = 0

        reduced_X = X[weights_zero != 0]
        reduced_Y = Y[weights_zero != 0]
        reduced_weights = weights[weights_zero != 0]
        n_components = min(X.shape[0], X.shape[1])
        for center_X, center_Y, scale_X, scale_Y in center_scale_combinations:
            pls_models_reduced_input = self.fit_models(
                X=reduced_X,
                Y=reduced_Y,
                n_components=n_components,
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
                weights=reduced_weights,
            )
            pls_preds_reduced_input, pls_Bs_reduced_input = self.get_model_preds(
                X, pls_models_reduced_input
            )
            pls_models = self.fit_models(
                X=X,
                Y=Y,
                n_components=n_components,
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
                weights=weights_zero,
            )
            pls_preds, pls_Bs = self.get_model_preds(X, pls_models)

            # Check that the predictions are equal
            for i, (pred_reduced_input, pred) in enumerate(
                zip(pls_preds_reduced_input, pls_preds)
            ):
                err_msg = f"Model {i}, Center_X: {center_X}, Center_Y: {center_Y}, Scale_X: {scale_X}, Scale_Y: {scale_Y}"
                assert_allclose(
                    pred_reduced_input,
                    pred,
                    atol=pred_atol,
                    rtol=pred_rtol,
                    err_msg=err_msg,
                )

            # Check that the B matrices are equal
            for B_reduced_input, B in zip(pls_Bs_reduced_input, pls_Bs):
                assert_allclose(
                    B_reduced_input,
                    B,
                    atol=B_atol,
                    rtol=B_rtol,
                    err_msg=err_msg,
                )

            last_pls_preds, last_pls_Bs = self.get_model_preds(
                X, pls_models, last_only=True
            )
            wls_pred, wls_B = self.fit_predict_wls(
                X=X,
                Y=Y,
                weights=weights_zero,
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
            )
            self.check_wpls_max_components_eq_to_wls(
                pls_preds=last_pls_preds,
                wls_pred=wls_pred,
                pls_Bs=last_pls_Bs,
                wls_B=wls_B,
                pred_atol=wls_pred_atol,
                pred_rtol=wls_pred_rtol,
                B_atol=wls_B_atol,
                B_rtol=wls_B_rtol,
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
            )

    def test_wpls_1(self):
        """
        Description
        -----------
        This test loads input predictor variables, a single target variable, and weights
        for the weighted PLS algorithm. It then calls the `check_wpls_zero_weights` and
        `check_wpls_constant_weights` methods, both of which call
        `check_wpls_max_components_eq_to_wls`. All three methods validate the results
        of the weighted PLS algorithms.

        Returns
        -------
        None
        """
        X = self.load_X()
        Y = self.load_Y(["Protein"])
        random_weights = self.load_weights(None)
        # Decrease the amount of samples and features in the interest of time.
        step = 100
        cutoff = 10
        X = X[::step, :cutoff]
        Y = Y[::step]
        random_weights = random_weights[::step]

        assert Y.shape[1] == 1

        self.check_wpls_zero_weights(
            X,
            Y,
            random_weights,
            pred_atol=1e-5,
            pred_rtol=1e-5,
            B_atol=5e-3,
            B_rtol=5e-3,
            wls_pred_atol=5e-3,
            wls_pred_rtol=5e-4,
            wls_B_atol=0.85,
            wls_B_rtol=0.02,
        )

        self.check_wpls_constant_weights(
            X,
            Y,
            pred_atol=5e-3,
            pred_rtol=5e-4,
            B_atol=0.85,
            B_rtol=0.02,
            wls_pred_atol=5e-3,
            wls_pred_rtol=5e-4,
            wls_B_atol=0.85,
            wls_B_rtol=0.02,
        )

    # JAX will issue a warning if os.fork() is called as JAX is incompatible with
    # multi-threaded code. os.fork() is called by the  other cross-validation
    # algorithms. However, there is no interaction between the JAX and the other
    # algorithms, so we can safely ignore this warning.
    @pytest.mark.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="os.fork() was called. os.fork() is"
        " incompatible with multithreaded code, and JAX is"
        " multithreaded, so this will likely lead to a"
        " deadlock.",
    )
    def test_wpls_2_m_less_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is less than K), and weights for the weighted PLS algorithm. It then calls the
        `check_wpls_zero_weights` and `check_wpls_constant_weights` methods, both of
        which call `check_wpls_max_components_eq_to_wls`. All three methods validate
        the results of the weighted PLS algorithms.

        Returns
        -------
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Moisture",
                "Protein",
            ]
        )
        random_weights = self.load_weights(None)
        step = 100
        cutoff = 10
        X = X[::step, :cutoff]
        Y = Y[::step]
        random_weights = random_weights[::step]

        assert Y.shape[1] > 1
        assert Y.shape[1] < X.shape[1]

        self.check_wpls_zero_weights(
            X,
            Y,
            random_weights,
            pred_atol=1e-3,
            pred_rtol=5e-5,
            B_atol=5e-3,
            B_rtol=5e-3,
            wls_pred_atol=5e-3,
            wls_pred_rtol=5e-4,
            wls_B_atol=0.85,
            wls_B_rtol=0.02,
        )

        self.check_wpls_constant_weights(
            X,
            Y,
            pred_atol=5e-3,
            pred_rtol=5e-4,
            B_atol=0.85,
            B_rtol=0.02,
            wls_pred_atol=5e-3,
            wls_pred_rtol=5e-4,
            wls_B_atol=0.85,
            wls_B_rtol=0.02,
        )

    def test_wpls_2_m_eq_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is equal to K), and weights for the weighted PLS algorithm. It then calls the
        `check_wpls_zero_weights` and `check_wpls_constant_weights` methods, both of
        which call `check_wpls_max_components_eq_to_wls`. All three methods validate
        the results of the weighted PLS algorithms.

        Returns
        -------
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Moisture",
                "Protein",
            ]
        )
        random_weights = self.load_weights(None)
        # Decrease the amount of samples in the interest of time.
        step = 100
        X = X[::step, :2]
        Y = Y[::step]
        random_weights = random_weights[::step]

        assert Y.shape[1] > 1
        assert Y.shape[1] == X.shape[1]

        self.check_wpls_zero_weights(
            X,
            Y,
            random_weights,
            pred_atol=1e-5,
            pred_rtol=1e-5,
            B_atol=5e-3,
            B_rtol=5e-3,
            wls_pred_atol=5e-3,
            wls_pred_rtol=5e-4,
            wls_B_atol=0.85,
            wls_B_rtol=0.02,
        )

        self.check_wpls_constant_weights(
            X,
            Y,
            pred_atol=5e-3,
            pred_rtol=5e-4,
            B_atol=0.85,
            B_rtol=0.02,
            wls_pred_atol=5e-3,
            wls_pred_rtol=5e-4,
            wls_B_atol=0.85,
            wls_B_rtol=0.02,
        )

    def test_wpls_2_m_greater_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is greater than K), and weights for the weighted PLS algorithm. It then calls
        the `check_wpls_zero_weights` and `check_wpls_constant_weights` methods, both
        of which call `check_wpls_max_components_eq_to_wls`. All three methods validate
        the results of the weighted PLS algorithms.

        Returns
        -------
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
            ]
        )
        random_weights = self.load_weights(None)
        # Decrease the amount of samples in the interest of time
        step = 100
        cutoff = 3
        X = X[::step, :cutoff]
        Y = Y[::step]
        random_weights = random_weights[::step]

        assert Y.shape[1] > 1
        assert Y.shape[1] > X.shape[1]

        self.check_wpls_zero_weights(
            X,
            Y,
            random_weights,
            pred_atol=1e-5,
            pred_rtol=1e-5,
            B_atol=5e-3,
            B_rtol=5e-3,
            wls_pred_atol=5e-3,
            wls_pred_rtol=5e-4,
            wls_B_atol=0.85,
            wls_B_rtol=0.02,
        )

        self.check_wpls_constant_weights(
            X,
            Y,
            pred_atol=5e-3,
            pred_rtol=5e-4,
            B_atol=0.85,
            B_rtol=0.02,
            wls_pred_atol=5e-3,
            wls_pred_rtol=5e-4,
            wls_B_atol=0.85,
            wls_B_rtol=0.02,
        )

    def check_weighted_cross_val_with_preprocessing(
        self, X, Y, weights, splits, atol, rtol
    ):
        center_Xs = [False, True]
        center_Ys = [False, True]
        scale_Xs = [False, True]
        scale_Ys = [False, True]
        center_scale_combinations = product(center_Xs, center_Ys, scale_Xs, scale_Ys)
        n_components = min(X.shape[0], X.shape[1])
        unique_splits = np.unique(splits)
        for center_X, center_Y, scale_X, scale_Y in center_scale_combinations:
            models = self.get_models(
                center_X=center_X,
                center_Y=center_Y,
                scale_X=scale_X,
                scale_Y=scale_Y,
                fast_cv=False,
            )

            for model in models:
                if isinstance(model, (JAX_Alg_1, JAX_Alg_2)):
                    results = model.cross_validate(
                        X=X,
                        Y=Y,
                        A=n_components,
                        cv_splits=splits,
                        preprocessing_function=self.jax_snv,
                        metric_function=self.pred_metric,
                        metric_names=["Preds"],
                        weights=weights,
                    )
                    cv_preds = [results["Preds"][split] for split in unique_splits]
                elif isinstance(model, NpPLS):
                    results = model.cross_validate(
                        X=X,
                        Y=Y,
                        A=n_components,
                        cv_splits=splits,
                        preprocessing_function=self.snv,
                        metric_function=self.pred_metric,
                        weights=weights,
                    )
                    cv_preds = [results[split] for split in unique_splits]
                else:
                    raise ValueError(f"Model {model} not recognized.")

                for i, split in enumerate(unique_splits):
                    X_train = X[splits != split]
                    Y_train = Y[splits != split]
                    X_val = X[splits == split]
                    Y_val = Y[splits == split]
                    X_train, Y_train, X_val, Y_val = self.snv(
                        X_train, Y_train, X_val, Y_val
                    )
                    weights_train = weights[splits != split]
                    model.fit(X_train, Y_train, n_components, weights_train)
                    direct_preds = model.predict(X_val)
                    max_stable_components = model.max_stable_components
                    # SNV removes the mean, so we are losing one component
                    if (
                        max_stable_components is None
                        or max_stable_components == model.A
                    ):
                        max_stable_components = model.A - 1

                    err_msg = (
                        f"Model {model}, split_idx: {i}, Center_X: {center_X},"
                        f"Center_Y: {center_Y}, Scale_X: {scale_X}, Scale_Y: {scale_Y}"
                    )

                    assert_allclose(
                        cv_preds[i][:max_stable_components],
                        direct_preds[:max_stable_components],
                        atol=atol,
                        rtol=rtol,
                        err_msg=err_msg,
                    )

    # We are probably fitting too many components here. But we do not care.
    @pytest.mark.filterwarnings(
        "ignore", category=UserWarning, match="Weight is close to zero"
    )
    def test_weighted_cross_val_with_preprocessing_pls_1(self):
        """
        Description
        -----------
        This test loads input predictor variables, a single target variable, weights, and
        split indices for cross-validation. It then calls the
        `check_weighted_cross_val_with_preprocessing` method to validate the
        cross-validation results for the PLS algorithms with preprocessing.

        Returns
        -------
        None
        """
        X = self.load_X()
        Y = self.load_Y(["Protein"])
        splits = self.load_Y(["split"]).squeeze().astype(np.int64)
        # Check that split contains integers from 0 to n_splits - 1. This is not
        # necessary for the cross validation methods but it is necessary for the check
        # we are performing.
        assert np.all(np.unique(splits) == np.arange(splits.max() + 1))
        random_weights = self.load_weights(None)
        # Decrease the amount of samples and features in the interest of time.
        step = 100
        cutoff = 10
        X = X[::step, :cutoff]
        Y = Y[::step]
        splits = splits[::step]
        random_weights = random_weights[::step]
        assert Y.shape[1] == 1
        self.check_weighted_cross_val_with_preprocessing(
            X, Y, random_weights, splits, atol=1e-8, rtol=1e-7
        )

    # We are probably fitting too many components here. But we do not care.
    @pytest.mark.filterwarnings(
        "ignore", category=UserWarning, match="Weight is close to zero"
    )
    def test_weighted_cross_val_with_preprocessing_pls_2_m_less_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is less than K), weights, and split indices for cross-validation. It then calls
        the `check_weighted_cross_val_with_preprocessing` method to validate the
        cross-validation results for the PLS algorithms with preprocessing.

        Returns
        -------
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Moisture",
                "Protein",
            ]
        )
        splits = self.load_Y(["split"]).squeeze().astype(np.int64)
        # Check that split contains integers from 0 to n_splits - 1. This is not
        # necessary for the cross validation methods but it is necessary for the check
        # we are performing.
        assert np.all(np.unique(splits) == np.arange(splits.max() + 1))
        random_weights = self.load_weights(None)
        # Decrease the amount of samples and features in the interest of time.
        step = 100
        cutoff = 10
        X = X[::step, :cutoff]
        Y = Y[::step]
        splits = splits[::step]
        random_weights = random_weights[::step]
        assert Y.shape[1] > 1
        assert Y.shape[1] < X.shape[1]
        self.check_weighted_cross_val_with_preprocessing(
            X, Y, random_weights, splits, atol=1e-7, rtol=1e-7
        )

    # We are probably fitting too many components here. But we do not care.
    @pytest.mark.filterwarnings(
        "ignore", category=UserWarning, match="Weight is close to zero"
    )
    def test_weighted_cross_val_with_preprocessing_pls_2_m_eq_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is equal to K), weights, and split indices for cross-validation. It then calls
        the `check_weighted_cross_val_with_preprocessing` method to validate the
        cross-validation results for the PLS algorithms with preprocessing.

        Returns
        -------
        None
        """
        X = self.load_X()
        Y = self.load_Y(
            [
                "Moisture",
                "Protein",
            ]
        )
        splits = self.load_Y(["split"]).squeeze().astype(np.int64)
        # Check that split contains integers from 0 to n_splits - 1. This is not
        # necessary for the cross validation methods but it is necessary for the check
        # we are performing.
        assert np.all(np.unique(splits) == np.arange(splits.max() + 1))
        random_weights = self.load_weights(None)
        # Decrease the amount of samples and features in the interest of time.
        step = 100
        cutoff = 2
        X = X[::step, :cutoff]
        Y = Y[::step]
        splits = splits[::step]
        random_weights = random_weights[::step]
        assert Y.shape[1] > 1
        assert Y.shape[1] == X.shape[1]
        self.check_weighted_cross_val_with_preprocessing(
            X, Y, random_weights, splits, atol=1e-7, rtol=1e-7
        )

    @pytest.mark.filterwarnings(
        "ignore", category=UserWarning, match="Weight is close to zero"
    )
    def test_weighted_cross_val_with_preprocessing_pls_2_m_greater_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M
        is greater than K), weights, and split indices for cross-validation. It then
        calls the `check_weighted_cross_val_with_preprocessing` method to validate the
        cross-validation results for the PLS algorithms with preprocessing.

        Returns
        -------
        None
        """

        X = self.load_X()
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
            ]
        )
        splits = self.load_Y(["split"]).squeeze().astype(np.int64)
        # Check that split contains integers from 0 to n_splits - 1. This is not
        # necessary for the cross validation methods but it is necessary for the check
        # we are performing.
        assert np.all(np.unique(splits) == np.arange(splits.max() + 1))
        random_weights = self.load_weights(None)
        # Decrease the amount of samples and features in the interest of time.
        step = 100
        cutoff = 3
        X = X[::step, :cutoff]
        Y = Y[::step]
        splits = splits[::step]
        random_weights = random_weights[::step]
        assert Y.shape[1] > 1
        assert Y.shape[1] > X.shape[1]
        self.check_weighted_cross_val_with_preprocessing(
            X, Y, random_weights, splits, atol=1e-7, rtol=1e-7
        )
