"""
Shared Improved Kernel PLS inner loop (steps 2-5 plus the regression-coefficient
update) by Dayal and MacGregor:
https://arxiv.org/abs/2401.13185
https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

This is the numerical core of Improved Kernel PLS, factored out so that both the
standard fit (`ikpls.numpy_ikpls`) and the fast cross-validation fit
(`ikpls.fast_cross_validation.numpy_ikpls`) share a single implementation rather than
duplicating it. It is pure NumPy (no scikit-learn dependency) and is called once per
fit/fold, so it adds no per-component overhead compared to the previous inline loops.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from typing import Callable, Optional, Tuple

import numpy as np
import numpy.linalg as la
import numpy.typing as npt


def improved_kernel_pls_inner_loop(
    algorithm: int,
    A: int,
    K: int,
    M: int,
    XTY: npt.NDArray[np.floating],
    X: Optional[npt.NDArray[np.floating]] = None,
    XTX: Optional[npt.NDArray[np.floating]] = None,
    dtype: type = np.float64,
    eps: float | np.floating = np.finfo(np.float64).eps,
    weight_warning: Optional[Callable[[int], None]] = None,
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    Optional[npt.NDArray[np.floating]],
]:
    r"""
    Runs Improved Kernel PLS steps 2-5 for `A` components.

    For Improved Kernel PLS Algorithm #1, pass the (already centered/scaled/weighted)
    training `X`. For Algorithm #2, pass `XTX`. In both cases pass the corresponding
    `XTY`. `XTY` is not modified in place (it is deflated via reassignment to a local).

    Parameters
    ----------
    algorithm : int
        Whether to use Improved Kernel PLS Algorithm #1 (uses `X`) or #2 (uses `XTX`).

    A : int
        Number of components.

    K : int
        Number of predictor variables (features).

    M : int
        Number of response variables (targets).

    XTY : Array of shape (K, M)
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` (training set, with any centering,
        scaling, and weighting already applied).

    X : None or Array of shape (N_train, K)
        Training predictor variables (with any centering, scaling, and weighting already
        applied). Required for Algorithm #1; ignored for Algorithm #2.

    XTX : None or Array of shape (K, K)
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` (training set, with any centering,
        scaling, and weighting already applied). Required for Algorithm #2; ignored for
        Algorithm #1.

    dtype : type[np.floating], default=numpy.float64
        The float datatype to use for the allocated output arrays.

    eps : float, default=numpy.finfo(numpy.float64).eps
        Machine epsilon used to detect a residual that has gone to (near) zero.

    weight_warning : None or Callable[[int], None], optional
        Called with the current component index `i` if the residual goes below `eps`
        (after which iteration stops). If None, no warning is emitted.

    Returns
    -------
    B : Array of shape (A, K, M)
        PLS regression coefficients tensor.

    W : Array of shape (K, A)
        PLS weights matrix for X.

    P : Array of shape (K, A)
        PLS loadings matrix for X.

    Q : Array of shape (M, A)
        PLS loadings matrix for Y.

    R : Array of shape (K, A)
        PLS weights matrix to compute scores T directly from original X.

    T : None or Array of shape (N_train, A)
        PLS scores matrix of X. Only returned for Algorithm #1; None for Algorithm #2.
    """
    B = np.zeros(shape=(A, K, M), dtype=dtype)
    WT = np.zeros(shape=(A, K), dtype=dtype)
    PT = np.zeros(shape=(A, K), dtype=dtype)
    QT = np.zeros(shape=(A, M), dtype=dtype)
    RT = np.zeros(shape=(A, K), dtype=dtype)
    W = WT.T
    P = PT.T
    Q = QT.T
    R = RT.T
    if algorithm == 1:
        TT = np.zeros(shape=(A, X.shape[0]), dtype=dtype)
        T = TT.T
    else:
        T = None

    for i in range(A):
        # Step 2
        if M == 1:
            # `XTY` is a (K, 1) column vector; its Euclidean (Frobenius) norm equals its
            # matrix 2-norm but avoids the SVD that `ord=2` triggers on a 2-D array.
            norm = la.norm(XTY)
            if np.isclose(norm, 0, atol=eps, rtol=0):
                if weight_warning is not None:
                    weight_warning(i)
                break
            w = XTY / norm
        elif M < K:
            XTYTXTY = XTY.T @ XTY
            eig_vals, eig_vecs = la.eigh(XTYTXTY)
            q = eig_vecs[:, -1:]
            w = XTY @ q
            # `w` is a (K, 1) column vector, so its Euclidean (Frobenius) norm equals its
            # matrix 2-norm but avoids the SVD that `ord=2` triggers on a 2-D array.
            norm = la.norm(w)
            if np.isclose(norm, 0, atol=eps, rtol=0):
                if weight_warning is not None:
                    weight_warning(i)
                break
            w = w / norm
        else:
            XTYYTX = XTY @ XTY.T
            eig_vals, eig_vecs = la.eigh(XTYYTX)
            norm = eig_vals[-1]
            if np.isclose(norm, 0, atol=eps, rtol=0):
                if weight_warning is not None:
                    weight_warning(i)
                break
            w = eig_vecs[:, -1:]
        WT[i] = w.squeeze()

        # Step 3
        r = w - RT[:i].T @ (PT[:i] @ w)
        RT[i] = r.squeeze()

        # Step 4
        if algorithm == 1:
            t = X @ r
            TT[i] = t.squeeze()
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
        PT[i] = p.squeeze()
        QT[i] = q.squeeze()

        # Step 5
        XTY = XTY - (p @ q.T) * tTt

        # Compute regression coefficients
        B[i] = B[i - 1] + r @ q.T

    return B, W, P, Q, R, T
