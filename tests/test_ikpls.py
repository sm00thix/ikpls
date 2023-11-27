import pytest
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose
import jax
from jax import numpy as jnp
from typing import Tuple, Callable, Union
from sklearn.cross_decomposition import PLSRegression as SkPLS
from ikpls.jax_ikpls_alg_1 import PLS as JAX_Alg_1
from ikpls.jax_ikpls_alg_2 import PLS as JAX_Alg_2
from ikpls.numpy_ikpls import PLS as NpPLS
from . import load_data


class TestClass:
    csv = load_data.load_csv()
    raw_spectra = load_data.load_spectra()

    def load_X(self) -> npt.NDArray[np.float_]:
        """
        Description
        -----------
        Load the raw spectral data.

        Returns
        -------
        npt.NDArray[np.float_]
            The raw spectral data.
        """
        return np.copy(self.raw_spectra)

    def load_Y(self, values: list[str]) -> npt.NDArray[np.float_]:
        """
        Description
        -----------
        Load target values based on the specified column names.

        Parameters
        ----------
        `values` : list[str]
            List of column names to extract target values from the CSV data.

        Returns
        -------
        NDArray[float]
            Target values as a NumPy array.
        """
        target_values = self.csv[values].to_numpy()
        return target_values

    def fit_models(
        self, X: npt.NDArray, Y: npt.NDArray, n_components: int
    ) -> Tuple[
        SkPLS, npt.NDArray, NpPLS, NpPLS, JAX_Alg_1, JAX_Alg_2, JAX_Alg_1, JAX_Alg_2
    ]:
        """
        Description
        -----------

        Fit various PLS models using different implementations and preprocessing.

        Parameters
        ----------
        `X` : NDArray[float]
            Input data (spectra).

        `Y` : NDArray[float]
            Target values.

        `n_components` : int
            Number of PLS components.

        Returns
        -------
        tuple
            A tuple containing PLS models and their respective regression matrices.
        """
        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
        jnp_X = jnp.array(X)
        jnp_Y = jnp.array(Y)

        sk_pls = SkPLS(
            n_components=n_components, scale=False, copy=True
        )  # Do not rescale again.
        np_pls_alg_1 = NpPLS(algorithm=1)
        np_pls_alg_2 = NpPLS(algorithm=2)
        jax_pls_alg_1 = JAX_Alg_1(reverse_differentiable=False, verbose=True)
        jax_pls_alg_2 = JAX_Alg_2(reverse_differentiable=False, verbose=True)
        diff_jax_pls_alg_1 = JAX_Alg_1(reverse_differentiable=True, verbose=True)
        diff_jax_pls_alg_2 = JAX_Alg_2(reverse_differentiable=True, verbose=True)

        sk_pls.fit(X=X, Y=Y)
        np_pls_alg_1.fit(X=X, Y=Y, A=n_components)
        np_pls_alg_2.fit(X=X, Y=Y, A=n_components)
        jax_pls_alg_1.fit(X=jnp_X, Y=jnp_Y, A=n_components)
        jax_pls_alg_2.fit(X=jnp_X, Y=jnp_Y, A=n_components)
        diff_jax_pls_alg_1.fit(X=jnp_X, Y=jnp_Y, A=n_components)
        diff_jax_pls_alg_2.fit(X=jnp_X, Y=jnp_Y, A=n_components)

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

    def assert_matrix_orthogonal(
        self, M: npt.NDArray, atol: float, rtol: float
    ) -> None:
        """
        Description
        -----------

        Check if a matrix is orthogonal.

        Parameters
        ----------
        `M` : NDArray[float]
            Matrix to check for orthogonality.

        `atol` : float
            Absolute tolerance for checking orthogonality.

        `rtol` : float
            Relative tolerance for checking orthogonality.

        Returns
        -------
        `None`

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
        `sk_B` : NDArray[float]
            Sklearn PLS regression matrix.

        `np_pls_alg_1`
            Numpy-based PLS model using algorithm 1.

        `np_pls_alg_2`
            Numpy-based PLS model using algorithm 2.

        `jax_pls_alg_1`
            JAX-based PLS model using algorithm 1.

        `jax_pls_alg_2`
            JAX-based PLS model using algorithm 2.

        `diff_jax_pls_alg_1`
            JAX-based PLS model using algorithm 1 with reverse differentiation.

        `diff_jax_pls_alg_2`
            JAX-based PLS model using algorithm 2 with reverse differentiation.

        `atol` : float
            Absolute tolerance for checking equality.

        `rtol` : float
            Relative tolerance for checking equality.

        `n_good_components` : int, optional
            Number of components to check, or -1 to use all possible number of components.

        Returns
        -------
        `None`

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
        `sk_B` : NDArray[float]
            Sklearn PLS regression matrix.

        `np_pls_alg_1`
            Numpy-based PLS model using algorithm 1.

        `np_pls_alg_2`
            Numpy-based PLS model using algorithm 2.

        `jax_pls_alg_1`
            JAX-based PLS model using algorithm 1.

        `jax_pls_alg_2`
            JAX-based PLS model using algorithm 2.

        `diff_jax_pls_alg_1`
            JAX-based PLS model using algorithm 1 with reverse differentiation.

        `diff_jax_pls_alg_2`
            JAX-based PLS model using algorithm 2 with reverse differentiation.

        `X` : NDArray[float]
            Input data (spectra) for making predictions.

        `atol` : float
            Absolute tolerance for checking equality.

        `rtol` : float
            Relative tolerance for checking equality.

        `n_good_components` : int, optional
            Number of components to check, or -1 to use all possible number of components.

        Returns
        -------
        `None`

        Raises
        ------
        AssertionError
            If the predictions are not consistent.
        """
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A
        sk_all_preds = X @ sk_B
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
        `np_pls_alg_1`
            Numpy-based PLS model using algorithm 1.

        `np_pls_alg_2`
            Numpy-based PLS model using algorithm 2.

        `jax_pls_alg_1`
            JAX-based PLS model using algorithm 1.

        `jax_pls_alg_2`
            JAX-based PLS model using algorithm 2.

        `diff_jax_pls_alg_1`
            JAX-based PLS model using algorithm 1 with reverse differentiation.

        `diff_jax_pls_alg_2`
            JAX-based PLS model using algorithm 2 with reverse differentiation.

        `atol` : float
            Absolute tolerance for checking equality.

        `rtol` : float
            Relative tolerance for checking equality.

        `n_good_components` : int, optional
            Number of components to check, or -1 to use all possible number of components.

        Returns
        -------
        `None`

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
        `np_pls_alg_1`
            Numpy-based PLS model using algorithm 1.

        `jax_pls_alg_1`
            JAX-based PLS model using algorithm 1.

        `diff_jax_pls_alg_1`
            JAX-based PLS model using algorithm 1 with reverse differentiation.

        `X` : ndarray
            Original input matrix.

        `atol` : float
            Absolute tolerance for comparing matrix values.

        `rtol` : float
            Relative tolerance for comparing matrix values.

        `n_good_components` : int, optional
            Number of components to check, or -1 to use all possible number of components.

        Returns
        -------
        `None`

        Raises
        ------
        AssertionError
            If the equality properties are not satisfied.
        """
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A

        # X can be reconstructed by multiplying X scores (T) and the transpose of X loadings (P)
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
        `np_pls_alg_1`
            Numpy-based PLS model using algorithm 1.

        `np_pls_alg_2`
            Numpy-based PLS model using algorithm 2.

        `jax_pls_alg_1`
            JAX-based PLS model using algorithm 1.

        `jax_pls_alg_2`
            JAX-based PLS model using algorithm 2.

        `diff_jax_pls_alg_1`
            JAX-based PLS model using algorithm 1 with reverse differentiation.

        `diff_jax_pls_alg_2`
            JAX-based PLS model using algorithm 2 with reverse differentiation.

        `n_good_components` : int, optional
            Number of components to check, or -1 to use all possible number of components.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the internal matrices are not consistent across NumPy and JAX implementations.
        """
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A

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

        This method performs testing of the PLS1 algorithm using various models and checks for equality,
        orthogonality, regression matrices, and predictions. It also validates the algorithm's numerical
        stability for the "Protein" dataset.

        Parameters
        ----------
        None

        Returns
        -------
        `None`
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
            atol=1e-8,
            rtol=6e-5,
        )

        self.check_predictions(
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
        Test PLS2 algorithm when the number of targets is less than the number of features (M < K).

        This method tests the PLS2 algorithm under the scenario where the number of target variables (M)
        is less than the number of features (K) in the dataset. It performs various tests on different
        PLS2 models, checks for equality, orthogonality, regression matrices, and predictions.

        Parameters
        ----------
        None

        Returns
        -------
        `None`
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
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            diff_jax_pls_alg_1=diff_jax_pls_alg_1,
            diff_jax_pls_alg_2=diff_jax_pls_alg_2,
            X=X,
            atol=1e-2,
            rtol=0,
        )  # PLS2 is not as numerically stable as PLS1.

    def test_pls_2_m_eq_k(self) -> None:
        """
        Description
        -----------
        Test PLS2 algorithm where the number of targets is equal to the number of features (M = K).

        This method tests the PLS2 algorithm under the scenario where the number of target variables (M)
        is equal to the number of features (K) in the dataset. It performs various tests on different
        PLS2 models, checks for equality, orthogonality, regression matrices, and predictions.

        Parameters
        ----------
        None

        Returns
        -------
        `None`
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

    def test_pls_2_m_greater_k(self) -> None:
        """
        Description
        -----------

        Test PLS2 algorithm where the number of targets is greater than the number of features (M > K).

        This method tests the PLS2 algorithm under the scenario where the number of target variables (M)
        is greater than the number of features (K) in the dataset. It performs various tests on different
        PLS2 models, checks for equality, orthogonality, regression matrices, and predictions.

        Parameters
        ----------
        None

        Returns
        -------
        `None`
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

    def test_sanity_check_pls_regression(
        self,
    ) -> (
        None
    ):  # Taken from SkLearn's test suite and modified to include own algorithms.
        """
        Description
        -----------
        Test the PLS regression algorithm with a sanity check.

        This method performs a sanity check on the PLS regression algorithm. It loads the Linnerud dataset,
        fits the PLS regression models using various algorithms, and checks for equality between implemenetations'
        regression matrices, predictions, and for PLS properties. It also compares the results to expected values.

        Parameters
        ----------
        None

        Returns
        -------
        `None`
        """
        from sklearn.datasets import load_linnerud

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

        Test the PLS regression algorithm with a sanity check and a constant column in Y.

        This method performs a sanity check on the PLS regression algorithm using a dataset that includes a constant
        column in the target (Y) data. It loads the Linnerud dataset, sets the first column of Y to a constant, and fits
        the PLS regression models using various algorithms. The test checks for equality between implemenetations' regression
        matrices, predictions, and for PLS properties. It also compares the results to expected values.

        Parameters
        ----------
        None

        Returns
        -------
        `None`
        """
        from sklearn.datasets import load_linnerud

        d = load_linnerud()
        X = d.data  # Shape = (20,3)
        Y = d.target  # Shape = (20,3)
        Y[:, 0] = 1  # Set the first column to a constant
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

        # Check that sign flip is consistent and exact across loadings and weights. Ignore the first column of Y which will be a column of zeros (due to mean centering of its constant value).
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

    def check_pls_constant_y(
        self, X: npt.NDArray, Y: npt.NDArray
    ) -> (
        None
    ):  # Taken from SkLearn's test suite and modified to include own algorithms.
        """
        Description
        -----------
        Check PLS regression behavior when Y is constant.

        This method checks the behavior of PLS regression when the target data (Y) is constant. It first pre-processes the input
        data by centering and scaling it. Then, it fits PLS regression models using different algorithms, including the sklearn,
        NumPy-based, and JAX-based implementations. It checks for warnings related weights being close to zero during the fitting process.

        Parameters
        ----------
        `X` : numpy.ndarray
            The predictor variables.
        `Y` : numpy.ndarray
            The target variables.

        Returns
        -------
        `None`

        Raises
        ------
        AssertionError
            If the warnings for weights being close to zero are not raised for all implementations.
        """
        ## Taken from self.fit_models to check each individual algorithm for early stopping.
        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
        jnp_X = jnp.array(X)
        jnp_Y = jnp.array(Y)
        n_components = 2
        sk_pls = SkPLS(n_components=n_components, scale=False)  # Do not rescale again.
        np_pls_alg_1 = NpPLS(algorithm=1)
        np_pls_alg_2 = NpPLS(algorithm=2)
        jax_pls_alg_1 = JAX_Alg_1(reverse_differentiable=False, verbose=True)
        jax_pls_alg_2 = JAX_Alg_2(reverse_differentiable=False, verbose=True)
        diff_jax_pls_alg_1 = JAX_Alg_1(reverse_differentiable=True, verbose=True)
        diff_jax_pls_alg_2 = JAX_Alg_2(reverse_differentiable=True, verbose=True)

        sk_msg = "Y residual is constant at iteration"
        with pytest.warns(UserWarning, match=sk_msg):
            sk_pls.fit(X=X, Y=Y)
            assert_allclose(sk_pls.x_rotations_, 0)

        msg = "Weight is close to zero."
        with pytest.warns(UserWarning, match=msg):
            np_pls_alg_1.fit(X=X, Y=Y, A=n_components)
            assert_allclose(np_pls_alg_1.R, 0)
        with pytest.warns(UserWarning, match=msg):
            np_pls_alg_2.fit(X=X, Y=Y, A=n_components)
            assert_allclose(np_pls_alg_2.R, 0)
        with pytest.warns(UserWarning, match=msg):
            jax_pls_alg_1.fit(X=jnp_X, Y=jnp_Y, A=n_components)
        with pytest.warns(UserWarning, match=msg):
            jax_pls_alg_2.fit(X=jnp_X, Y=jnp_Y, A=n_components)
        with pytest.warns(UserWarning, match=msg):
            diff_jax_pls_alg_1.fit(X=jnp_X, Y=jnp_Y, A=n_components)
        with pytest.warns(UserWarning, match=msg):
            diff_jax_pls_alg_2.fit(X=jnp_X, Y=jnp_Y, A=n_components)

    def test_pls_1_constant_y(self):
        """
        Description
        -----------
        Test PLS regression when Y is constant with single target variable.

        This test generates random predictor variables (X) and a target variable (Y) where Y is a constant array with a single
        column. It ensures that Y has only one column and calls the 'check_pls_constant_y' method to validate the behavior of
        PLS regression in this scenario.

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

        This test generates random predictor variables (X) and a target variable (Y) where Y is a constant array with fewer
        columns (m) than the number of columns in X (k). It ensures that Y has more than one column but less than the number
        of columns in X and calls the 'check_pls_constant_y' method to validate the behavior of PLS regression in this scenario.

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

        This test generates random predictor variables (X) and a target variable (Y) where Y is a constant array with the same
        number of columns (m) as the number of columns in X (k). It ensures that Y has more than one column and the same number
        of columns as X, and calls the 'check_pls_constant_y' method to validate the behavior of PLS regression in this scenario.

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

        This test generates random predictor variables (X) and a target variable (Y) where Y is a constant array with more
        columns (M) than the number of columns in X (K). It ensures that Y has more than one column and more columns than X,
        and calls the 'check_pls_constant_y' method to validate the behavior of PLS regression in this scenario.

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
        This method tests the gradient propagation for reverse-mode differentiable JAX PLS. It convolves the input spectra
        with a filter, computes the gradients of the RMSE loss with respect to the parameters of the preprocessing filter,
        and verifies the correctness of gradient values and numerical stability.

        Parameters:
        `X` : numpy.ndarray:
            The input predictor variables.

        `Y` : numpy.ndarray
            The target variables.

        `num_components` : int
            The number of PLS components.

        `filter_size` (int):
            The size of the convolution filter.

        `val_atol` : float
            Absolute tolerance for value comparisons.

        `val_rtol` : float
            Relative tolerance for value comparisons.

        `grad_atol` : float
            Absolute tolerance for gradient comparisons.

        `grad_rtol` : float
            Relative tolerance for gradient comparisons.

        Returns
        -------
        `None`

        Raises
        ------
        AssertionError
            If the gradients are not computed, if the gradients are not equal across Improved Kernel PLS Algorithm #1 and #2, or if the output values are not consistent across all JAX implementations.
        """
        # Taken from self.fit_models to check each individual algorithm for early stopping.
        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std

        jnp_X = jnp.array(X, dtype=jnp.float64)
        jnp_Y = jnp.array(Y, dtype=jnp.float64)

        diff_pls_alg_1 = JAX_Alg_1(reverse_differentiable=True, verbose=True)
        diff_pls_alg_2 = JAX_Alg_2(reverse_differentiable=True, verbose=True)

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

        # Check that we can not differentiate the JAX implementations using reverse-mode differentiation without setting the parameter reverse_differentiable=True
        pls_alg_1 = JAX_Alg_1(reverse_differentiable=False, verbose=True)
        pls_alg_2 = JAX_Alg_2(reverse_differentiable=False, verbose=True)
        msg = "Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop with dynamic start/stop values."
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

        # For good measure, let's assure ourselves that the results are equivalent across reverse differentiable and non reverse differentiable versions:
        output_val_alg_1 = preprocess_fit_rmse(jnp_X, jnp_Y, pls_alg_1, num_components)(
            uniform_filter
        )
        output_val_alg_2 = preprocess_fit_rmse(jnp_X, jnp_Y, pls_alg_2, num_components)(
            uniform_filter
        )
        assert_allclose(output_val_alg_1, output_val_diff_alg_1, atol=0, rtol=1e-14)
        assert_allclose(output_val_alg_2, output_val_diff_alg_2, atol=0, rtol=1e-14)

    def test_gradient_pls_1(self):
        """
        Description
        -----------
        This test loads input predictor variables and a target variable with a single column and calls the 'check_gradient_pls'
        method to validate the gradient propagation for reverse-mode differentiable JAX PLS.

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
        This test loads input predictor variables and multiple target variables with M < K, and calls the 'check_gradient_pls'
        method to validate the gradient propagation for reverse-mode differentiable JAX PLS.

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
        assert (
            Y.shape[1] < X.shape[1] - filter_size + 1
        )  # The output of the convolution preprocessing is what is actually fed as input to the PLS algorithms.
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
        This test loads input predictor variables and multiple target variables with M = K, and calls the 'check_gradient_pls'
        method to validate the gradient propagation for reverse-mode differentiable JAX PLS.

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
        assert (
            Y.shape[1] == X.shape[1] - filter_size + 1
        )  # The output of the convolution preprocessing is what is actually fed as input to the PLS algorithms.
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
        This test loads input predictor variables and multiple target variables with M > K, and calls the 'check_gradient_pls'
        method to validate the gradient propagation for reverse-mode differentiable JAX PLS.

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
        assert (
            Y.shape[1] > X.shape[1] - filter_size + 1
        )  # The output of the convolution preprocessing is what is actually fed as input to the PLS algorithms.
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
        This method tests the ability to perform cross-validation to obtain the root mean square error (RMSE) and the best number
        of components for each target variable and each split.

        Parameters:
        `X` : numpy.ndarray
            The input predictor variables.
        `Y` : numpy.ndarray
            The target variables.
        `splits` : numpy.ndarray
            Split indices for cross-validation.

        `atol` : float
            Absolute tolerance for value comparisons.

        `rtol` : float
            Relative tolerance for value comparisons.

        Returns:
        `None`

        Raises
        ------
        AssertionError
            If the best number of components found by cross validation with is not exactly equal across each different PLS implementation.

            If the output RMSEs for the best number of components are not equal down to the specified tolerance across each different PLS implementation.
        """
        from sklearn.model_selection import cross_validate

        def cross_val_preprocessing(
            X_train: jnp.ndarray,
            Y_train: jnp.ndarray,
            X_val: jnp.ndarray,
            Y_val: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            x_mean = X_train.mean(axis=0, keepdims=True)
            X_train -= x_mean
            X_val -= x_mean
            y_mean = Y_train.mean(axis=0, keepdims=True)
            Y_train -= y_mean
            Y_val -= y_mean
            return X_train, Y_train, X_val, Y_val

        n_components = X.shape[1]

        sk_pls = SkPLS(n_components=n_components, scale=False)
        jax_pls_alg_1 = JAX_Alg_1(reverse_differentiable=False, verbose=True)
        jax_pls_alg_2 = JAX_Alg_2(reverse_differentiable=False, verbose=True)
        diff_jax_pls_alg_1 = JAX_Alg_1(reverse_differentiable=True, verbose=True)
        diff_jax_pls_alg_2 = JAX_Alg_2(reverse_differentiable=True, verbose=True)

        def cv_splitter(splits: npt.NDArray):
            uniq_splits = np.unique(splits)
            for split in uniq_splits:
                train_idxs = np.nonzero(splits != split)[0]
                val_idxs = np.nonzero(splits == split)[0]
                yield train_idxs, val_idxs

        def rmse_per_component(Y_true: npt.NDArray, Y_pred: npt.NDArray) -> npt.NDArray:
            e = Y_true - Y_pred
            se = e**2
            mse = np.mean(se, axis=-2)
            rmse = np.sqrt(mse)
            return rmse

        def jax_rmse_per_component(
            Y_true: jnp.ndarray, Y_pred: jnp.ndarray
        ) -> jnp.ndarray:
            e = Y_true - Y_pred
            se = e**2
            mse = jnp.mean(se, axis=-2)
            rmse = jnp.sqrt(mse)
            return rmse

        jnp_splits = jnp.array(splits)

        # Calibrate SkPLS
        sk_results = cross_validate(
            sk_pls, X, Y, cv=cv_splitter(splits), return_estimator=True, n_jobs=-1
        )
        sk_models = sk_results["estimator"]
        # Extract regression matrices for SkPLS for all possible number of components and make a prediction with the regression matrices at all possible number of components.
        sk_Bs = np.empty((len(sk_models), n_components, X.shape[1], Y.shape[1]))
        sk_preds = np.empty((len(sk_models), n_components, X.shape[0], Y.shape[1]))
        for i, sk_model in enumerate(sk_models):
            for j in range(sk_Bs.shape[1]):
                sk_B_at_component_j = np.dot(
                    sk_model.x_rotations_[..., : j + 1],
                    sk_model.y_loadings_[..., : j + 1].T,
                )
                sk_Bs[i, j] = sk_B_at_component_j
            sk_pred = (X - sk_models[i]._x_mean) / sk_models[i]._x_std @ sk_Bs[
                i
            ] + sk_models[i].intercept_
            sk_preds[i] = sk_pred
            assert_allclose(
                sk_pred[-1], sk_models[i].predict(X), atol=0, rtol=1e-13
            )  # Sanity check. SkPLS also uses the maximum number of components in its predict method.

        # Compute RMSE on the validation predictions
        sk_pls_rmses = np.empty((len(sk_models), n_components, Y.shape[1]))
        for i in range(len(sk_models)):
            val_idxs = val_idxs = np.nonzero(splits == i)[0]
            Y_true = Y[val_idxs]
            Y_pred = sk_preds[i, :, val_idxs, ...].swapaxes(0, 1)
            val_rmses = rmse_per_component(Y_true, Y_pred)
            sk_pls_rmses[i] = val_rmses

        # Calibrate NumPy PLS
        # Since SkLearn's cross_validate does not allow for a preprocessing function to be executed on each split, we have to get a bit creative.
        # This is because SkLearn's PLS implementation always mean centers X and Y based on the training data and uses these mean values in its predictions.
        # We subclass the original implementation and modify it to mimic the SkLearn implementation's behavior so that we can accurately compare results.
        class NpPLSWithPreprocessing(NpPLS):
            def __init__(
                self, algorithm: int = 1, dtype: np.float_ = np.float64
            ) -> None:
                super().__init__(algorithm, dtype)

            def fit(self, X: npt.ArrayLike, Y: npt.ArrayLike, A: int) -> None:
                self.X_mean = np.mean(X, axis=0)
                self.Y_mean = np.mean(Y, axis=0)
                X -= self.X_mean
                Y -= self.Y_mean
                return super().fit(X, Y, A)

            def predict(
                self, X: npt.ArrayLike, A: Union[None, int] = None
            ) -> npt.NDArray[np.float_]:
                return super().predict(X - self.X_mean, A) + self.Y_mean

        np_pls_alg_1 = NpPLSWithPreprocessing(algorithm=1)
        np_pls_alg_2 = NpPLSWithPreprocessing(algorithm=2)

        fit_params = {"A": n_components}
        np_pls_alg_1_results = cross_validate(
            np_pls_alg_1,
            X,
            Y,
            cv=cv_splitter(splits),
            scoring=lambda *args, **kwargs: 0,
            fit_params=fit_params,
            return_estimator=True,
            n_jobs=-1,
        )
        np_pls_alg_1_models = np_pls_alg_1_results["estimator"]
        np_pls_alg_2_results = cross_validate(
            np_pls_alg_2,
            X,
            Y,
            cv=cv_splitter(splits),
            scoring=lambda *args, **kwargs: 0,
            fit_params=fit_params,
            return_estimator=True,
            n_jobs=-1,
        )
        np_pls_alg_2_models = np_pls_alg_2_results["estimator"]

        # Compute RMSE on the validation predictions
        np_pls_alg_1_rmses = np.empty(
            (len(np_pls_alg_1_models), n_components, Y.shape[1])
        )
        np_pls_alg_2_rmses = np.empty(
            (len(np_pls_alg_2_models), n_components, Y.shape[1])
        )
        for i in range(len(np_pls_alg_1_models)):
            val_idxs = val_idxs = np.nonzero(splits == i)[0]
            Y_true = Y[val_idxs]
            Y_pred_alg_1 = np_pls_alg_1_models[i].predict(X[val_idxs])
            Y_pred_alg_2 = np_pls_alg_2_models[i].predict(X[val_idxs])
            val_rmses_alg_1 = rmse_per_component(Y_true, Y_pred_alg_1)
            val_rmses_alg_2 = rmse_per_component(Y_true, Y_pred_alg_2)
            np_pls_alg_1_rmses[i] = val_rmses_alg_1
            np_pls_alg_2_rmses[i] = val_rmses_alg_2

        # Calibrate JAX PLS
        jax_pls_alg_1_results = jax_pls_alg_1.cv(
            X,
            Y,
            n_components,
            jnp_splits,
            cross_val_preprocessing,
            jax_rmse_per_component,
            ["RMSE"],
        )
        diff_jax_pls_alg_1_results = diff_jax_pls_alg_1.cv(
            X,
            Y,
            n_components,
            jnp_splits,
            cross_val_preprocessing,
            jax_rmse_per_component,
            ["RMSE"],
        )
        jax_pls_alg_2_results = jax_pls_alg_2.cv(
            X,
            Y,
            n_components,
            jnp_splits,
            cross_val_preprocessing,
            jax_rmse_per_component,
            ["RMSE"],
        )
        diff_jax_pls_alg_2_results = diff_jax_pls_alg_2.cv(
            X,
            Y,
            n_components,
            jnp_splits,
            cross_val_preprocessing,
            jax_rmse_per_component,
            ["RMSE"],
        )

        # Check that best number of components in terms of minimizing validation RMSE for each split is equal among all algorithms
        unique_splits = np.unique(splits).astype(int)
        sk_best_num_components = [
            [np.argmin(sk_pls_rmses[split][..., i]) for split in unique_splits]
            for i in range(Y.shape[1])
        ]
        np_pls_alg_1_best_num_components = [
            [np.argmin(np_pls_alg_1_rmses[split][..., i]) for split in unique_splits]
            for i in range(Y.shape[1])
        ]
        np_pls_alg_2_best_num_components = [
            [np.argmin(np_pls_alg_2_rmses[split][..., i]) for split in unique_splits]
            for i in range(Y.shape[1])
        ]
        jax_pls_alg_1_best_num_components = [
            [
                np.argmin(jax_pls_alg_1_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(Y.shape[1])
        ]
        jax_pls_alg_2_best_num_components = [
            [
                np.argmin(jax_pls_alg_2_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(Y.shape[1])
        ]
        diff_jax_pls_alg_1_best_num_components = [
            [
                np.argmin(diff_jax_pls_alg_1_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(Y.shape[1])
        ]
        diff_jax_pls_alg_2_best_num_components = [
            [
                np.argmin(diff_jax_pls_alg_2_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(Y.shape[1])
        ]

        assert sk_best_num_components == np_pls_alg_1_best_num_components
        assert sk_best_num_components == np_pls_alg_2_best_num_components
        assert sk_best_num_components == jax_pls_alg_1_best_num_components
        assert sk_best_num_components == jax_pls_alg_2_best_num_components
        assert sk_best_num_components == diff_jax_pls_alg_1_best_num_components
        assert sk_best_num_components == diff_jax_pls_alg_2_best_num_components

        # Check that the RMSE achieved is similar
        sk_best_rmses = [
            [np.amin(sk_pls_rmses[split][..., i]) for split in unique_splits]
            for i in range(Y.shape[1])
        ]
        np_pls_alg_1_best_rmses = [
            [np.amin(np_pls_alg_1_rmses[split][..., i]) for split in unique_splits]
            for i in range(Y.shape[1])
        ]
        np_pls_alg_2_best_rmses = [
            [np.amin(np_pls_alg_2_rmses[split][..., i]) for split in unique_splits]
            for i in range(Y.shape[1])
        ]
        jax_pls_alg_1_best_rmses = [
            [
                np.amin(jax_pls_alg_1_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(Y.shape[1])
        ]
        jax_pls_alg_2_best_rmses = [
            [
                np.amin(jax_pls_alg_2_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(Y.shape[1])
        ]
        diff_jax_pls_alg_1_best_rmses = [
            [
                np.amin(diff_jax_pls_alg_1_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(Y.shape[1])
        ]
        diff_jax_pls_alg_2_best_rmses = [
            [
                np.amin(diff_jax_pls_alg_2_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(Y.shape[1])
        ]

        assert_allclose(np_pls_alg_1_best_rmses, sk_best_rmses, atol=atol, rtol=rtol)
        assert_allclose(np_pls_alg_2_best_rmses, sk_best_rmses, atol=atol, rtol=rtol)
        assert_allclose(jax_pls_alg_1_best_rmses, sk_best_rmses, atol=atol, rtol=rtol)
        assert_allclose(jax_pls_alg_2_best_rmses, sk_best_rmses, atol=atol, rtol=rtol)
        assert_allclose(
            diff_jax_pls_alg_1_best_rmses, sk_best_rmses, atol=atol, rtol=rtol
        )
        assert_allclose(
            diff_jax_pls_alg_2_best_rmses, sk_best_rmses, atol=atol, rtol=rtol
        )

    def test_cross_val_pls_1(self):
        """
        Description
        -----------
        This test loads input predictor variables, a single target variable, and split indices for cross-validation. It then calls
        the 'check_cross_val_pls' method to validate the cross-validation results, specifically for a single target variable.

        Returns:
        None
        """
        X = self.load_X()
        Y = self.load_Y(["Protein"])
        splits = self.load_Y(["split"])  # Contains 3 splits of differfent sizes
        assert Y.shape[1] == 1
        self.check_cross_val_pls(X, Y, splits, atol=0, rtol=1e-5)

    def test_cross_val_pls_2_m_less_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M is less than K), and split indices for
        cross-validation. It then calls the 'check_cross_val_pls' method to validate the cross-validation results for this scenario.

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
        splits = self.load_Y(["split"])  # Contains 3 splits of differfent sizes
        assert Y.shape[1] > 1
        assert Y.shape[1] < X.shape[1]
        self.check_cross_val_pls(X, Y, splits, atol=0, rtol=2e-4)

    def test_cross_val_pls_2_m_eq_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M is equal to K), and split indices for
        cross-validation. It then calls the 'check_cross_val_pls' method to validate the cross-validation results for this scenario.

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
        splits = self.load_Y(["split"])  # Contains 3 splits of differfent sizes
        X = X[..., :10]
        assert Y.shape[1] > 1
        assert Y.shape[1] == X.shape[1]
        self.check_cross_val_pls(X, Y, splits, atol=0, rtol=1e-5)

    def test_cross_val_pls_2_m_greater_k(self):
        """
        Description
        -----------
        This test loads input predictor variables, multiple target variables (where M is greater than K), and split indices for
        cross-validation. It then calls the 'check_cross_val_pls' method to validate the cross-validation results for this scenario.

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
        splits = self.load_Y(["split"])  # Contains 3 splits of differfent sizes
        X = X[..., :9]
        assert Y.shape[1] > 1
        assert Y.shape[1] > X.shape[1]
        self.check_cross_val_pls(X, Y, splits, atol=0, rtol=3e-5)
