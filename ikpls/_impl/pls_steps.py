"""
Shared Improved Kernel PLS inner loop (steps 2-5 plus the regression-coefficient
update) by Dayal and MacGregor:
https://arxiv.org/abs/2401.13185
https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

This is the numerical core of Improved Kernel PLS, factored out so that both the
standard fit (`ikpls.numpy`) and the fast cross-validation fit
(`ikpls.fast_cross_validation.numpy`) share a single implementation rather than
duplicating it. It is pure NumPy (no scikit-learn dependency) and is called once per
fit/fold, so it adds no per-component overhead compared to the previous inline loops.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import numpy.linalg as la
import numpy.typing as npt


def _stability_warning(i: int, cause: str) -> None:
    """Warn that component ``i`` (0-indexed) and beyond are numerically unstable.

    Emitted by :func:`improved_kernel_pls_inner_loop` when iteration stops early.
    ``cause`` is ``"weight"`` (an X-weight norm underflowed to ~0, so the X-Y
    cross-covariance is exhausted) or ``"score"`` (the score norm ``t^T t``
    collapsed relative to the largest component's -- a null-space direction past the
    numerical rank of X). Shared by the standard (``ikpls.numpy``) and fast
    cross-validation (``ikpls.fast_cross_validation.numpy``) NumPy fits.
    """
    detail = (
        "X weight is close to zero."
        if cause == "weight"
        else "X score norm is close to zero (component past the numerical rank of X)."
    )
    warnings.warn(
        message=f"{detail} Predictions with A = {i + 1} component(s) or higher are "
        f"identical to predictions with A = {i} component(s).",
        category=UserWarning,
    )


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
) -> Tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    Optional[npt.NDArray[np.floating]],
    int,
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
        Machine epsilon used to detect a residual/score norm that has gone to
        (near) zero.

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

    max_stable_components : int
        Number of leading, numerically stable components actually computed. Equals
        `A` unless iteration stopped early -- when an X-weight norm underflows below
        `eps` (X-Y cross-covariance exhausted; a "weight" warning is emitted) or a
        component's score norm ``t^T t`` collapses relative to the largest score norm
        seen (a null-space component past the numerical rank of X; a "score"
        warning). Rows at or beyond this index are left as zeros.
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

    # Number of leading, numerically stable components. Reduced to the current index
    # if iteration stops early (weight underflow or score-norm collapse).
    max_stable_components = A
    # Largest score norm ||t||^2 seen so far, for the RELATIVE null-space threshold.
    max_tTt = 0.0

    for i in range(A):
        # Step 2: X-weight w. If its (pre-normalization) norm underflows to ~0 the
        # X-Y cross-covariance is exhausted; normalizing would divide by ~0 and yield
        # NaN weights, so stop before writing anything for this component.
        if M == 1:
            # `XTY` is a (K, 1) column vector; its Euclidean (Frobenius) norm equals its
            # matrix 2-norm but avoids the SVD that `ord=2` triggers on a 2-D array.
            norm = la.norm(XTY)
            if np.isclose(norm, 0, atol=eps, rtol=0):
                _stability_warning(i, "weight")
                max_stable_components = i
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
                _stability_warning(i, "weight")
                max_stable_components = i
                break
            w = w / norm
        else:
            XTYYTX = XTY @ XTY.T
            eig_vals, eig_vecs = la.eigh(XTYYTX)
            norm = eig_vals[-1]
            if np.isclose(norm, 0, atol=eps, rtol=0):
                _stability_warning(i, "weight")
                max_stable_components = i
                break
            w = eig_vecs[:, -1:]

        # Step 3: rotation r mapping the original X to this component's score.
        r = w - RT[:i].T @ (PT[:i] @ w)

        # Step 4 (score norm first): tTt = ||t||^2, the value used to deflate XTY and
        # to normalize the loadings. When n_components exceeds the numerical rank of
        # X, r lies in X's null space and tTt collapses to ~0 relative to the earlier
        # components even though the weight norm above did not underflow. Such a
        # component carries no variance and its loadings p = (...) / tTt would blow
        # up, so detect it with a threshold RELATIVE to the largest score norm (an
        # absolute eps threshold would wrongly flag legitimately small-magnitude
        # data) and stop before dividing by tTt or writing the component. The weight
        # norm does not catch this: deflating XTY by (p q^T) tTt barely changes XTY
        # when tTt ~ 0.
        if algorithm == 1:
            t = X @ r
            tTt = t.T @ t
        else:
            rXTX = r.T @ XTX
            tTt = rXTX @ r
        tTt_val = tTt.item()
        max_tTt = max(max_tTt, tTt_val)
        if tTt_val <= eps * max_tTt:
            _stability_warning(i, "score")
            max_stable_components = i
            break

        # Step 4 (loadings): tTt is safely above the relative floor.
        if algorithm == 1:
            p = (t.T @ X).T / tTt
        else:
            p = rXTX.T / tTt
        if M == 1:
            q = norm / tTt
        elif M < K:
            q = q * np.sqrt(eig_vals[-1]) / tTt
        else:
            q = (r.T @ XTY).T / tTt

        # Commit component i (all stability guards passed). Writing here rather than
        # inline above guarantees rows >= max_stable_components stay zero.
        WT[i] = w.squeeze()
        RT[i] = r.squeeze()
        if algorithm == 1:
            TT[i] = t.squeeze()
        PT[i] = p.squeeze()
        QT[i] = q.squeeze()

        # Step 5: deflate XTY.
        XTY = XTY - (p @ q.T) * tTt

        # Compute regression coefficients
        B[i] = B[i - 1] + r @ q.T

    # Components beyond the last stable one are null-space directions that add no
    # predictive information, so carry the last stable regression coefficient forward.
    # This makes predictions with more than ``max_stable_components`` components
    # identical to predictions with exactly ``max_stable_components`` (rather than
    # degrading to the intercept-only prediction that all-zero coefficients would give).
    # When ``max_stable_components == 0`` there is no stable component and B stays zero
    # (intercept-only predictions).
    if 0 < max_stable_components < A:
        B[max_stable_components:] = B[max_stable_components - 1]

    return B, W, P, Q, R, T, max_stable_components
