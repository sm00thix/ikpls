"""
Contains the prediction shared by the NumPy and JAX implementations:
``((X - X_mean) / X_std) @ B * Y_std + Y_mean``, computed either on X as written or,
when that is cheaper, as ``X @ B' + b0`` with the centering and scaling applied to the
regression coefficients B instead of to X.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import math
import operator
from collections.abc import Callable
from typing import Any, Optional

# Folding the centering and scaling into B saves the elementwise passes over X
# (n_rows * K numbers) and adds about three operations per number of B (prod(B.shape))
# plus their dispatch. By operation count it pays off from about 2x; requiring
# _MIN_SAVED_RATIO times leaves a margin for the dispatch of the extra small operations,
# whose cost depends on the hardware (where the two forms cost about the same, choosing
# either costs little).
_MIN_SAVED_RATIO = 8


def fold_into_coefficients_is_cheaper(n_rows: int, B_shape: tuple) -> bool:
    """
    Whether a prediction for `n_rows` samples with regression coefficients of shape
    `B_shape` -- (K, M) for one number of components, (A, K, M) for all -- is faster
    with the centering and scaling applied to B than to X: when X (``n_rows * K``
    numbers) has at least `_MIN_SAVED_RATIO` times as many numbers as B
    (``prod(B_shape)``), i.e. for at least 8 * M samples with one number of components
    and 8 * A * M with all; e.g. not for one sample, or for the few validation samples
    of a fold with all components.
    """
    return n_rows * B_shape[-2] >= _MIN_SAVED_RATIO * math.prod(B_shape)


def _fold(
    X: Any,
    B: Any,
    X_mean: Optional[Any],
    X_std: Optional[Any],
    Y_mean: Optional[Any],
    Y_std: Optional[Any],
) -> bool:
    """
    Whether to predict as ``X @ B' + b0``: if that is cheaper
    (`fold_into_coefficients_is_cheaper`) and the shapes are concrete (not symbolic, as
    with `jax.export`) and those that `fold_into_coefficients` requires: X (N, K),
    X_mean and X_std (1, K), and Y_mean and Y_std (1, M). Other inputs are predicted
    from centred X, which broadcasts as before. The cheap tests come first, as this
    runs in every prediction.
    """
    X_shape, B_shape = getattr(X, "shape", ()), getattr(B, "shape", ())
    if len(X_shape) != 2 or len(B_shape) not in (2, 3):
        return False
    # Concrete shapes multiply to an int; symbolic ones do not.
    if not isinstance(math.prod(X_shape) * math.prod(B_shape), int):
        return False
    if not fold_into_coefficients_is_cheaper(X_shape[0], B_shape):
        return False
    K, M = B_shape[-2:]
    for stat, shape in ((X_mean, K), (X_std, K), (Y_mean, M), (Y_std, M)):
        # A statistic of another shape, or a float, is not folded.
        if stat is not None and getattr(stat, "shape", None) != (1, shape):
            return False
    return True


def fold_into_coefficients(
    B: Any,
    X_mean: Optional[Any],
    X_std: Optional[Any],
    Y_mean: Optional[Any],
    Y_std: Optional[Any],
    matmul: Callable[[Any, Any], Any] = operator.matmul,
) -> tuple:
    """
    ``(B', b0)`` with ``((X - X_mean) / X_std) @ B * Y_std + Y_mean == X @ B' + b0`` for
    every X: ``B' = B / X_std.T * Y_std`` and ``b0 = Y_mean - X_mean @ B'``. A statistic
    that is None is not applied; `b0` is None if it is zero. `B` is (K, M) or
    (A, K, M), and `b0` then (1, M) or (A, 1, M). Uses only arithmetic operators and
    `matmul`, so it works on NumPy and JAX arrays alike.
    """
    if X_std is not None:
        B = B / X_std.T
    if Y_std is not None:
        B = B * Y_std
    b0 = None
    if X_mean is not None:
        b0 = -matmul(X_mean, B)
    if Y_mean is not None:
        b0 = Y_mean if b0 is None else Y_mean + b0
    return B, b0


def predict(
    X: Any,
    B: Any,
    X_mean: Optional[Any],
    X_std: Optional[Any],
    Y_mean: Optional[Any],
    Y_std: Optional[Any],
    matmul: Callable[[Any, Any], Any] = operator.matmul,
    in_place: bool = False,
    overwrite_X: bool = False,
) -> Any:
    """
    ``((X - X_mean) / X_std) @ B * Y_std + Y_mean`` for X (N, K) and B (K, M) -> (N, M)
    or (A, K, M) -> (A, N, M); a statistic that is None is not applied. When X has at
    least 8 times as many numbers as B (`fold_into_coefficients_is_cheaper`) and the
    shapes allow it (`_fold`), it is computed as ``X @ B' + b0``
    (`fold_into_coefficients`), which touches X only in the matrix product. That
    equals the centred form up to rounding, but the rounding error grows with how far
    the columns of X lie from zero relative to their spread, as X @ B' and X_mean @ B'
    then nearly cancel. `matmul` computes the products ``X @ B``.
    Without `in_place`, the operands' types promote. With `in_place` (NumPy arrays of
    the dtype of B), a copy of X (or, with `overwrite_X`, X itself) is centred and
    scaled in place and the prediction is unscaled and uncentred in place, as in
    earlier versions, so that besides the prediction at most one array the size of X is
    allocated, and every result is stored in the dtype of X (B' is computed with the
    statistics, then cast). X is not modified unless `overwrite_X`.
    """
    if _fold(X, B, X_mean, X_std, Y_mean, Y_std):
        return _predict_folded(X, B, X_mean, X_std, Y_mean, Y_std, matmul, in_place)
    if in_place:
        return _predict_centred_in_place(
            X, B, X_mean, X_std, Y_mean, Y_std, matmul, overwrite_X
        )
    return _predict_centred(X, B, X_mean, X_std, Y_mean, Y_std, matmul)


def _predict_folded(X, B, X_mean, X_std, Y_mean, Y_std, matmul, in_place):
    """`predict` as ``X @ B' + b0``."""
    B, b0 = fold_into_coefficients(B, X_mean, X_std, Y_mean, Y_std, matmul)
    if not in_place:
        Y_pred = matmul(X, B)
        return Y_pred if b0 is None else Y_pred + b0
    Y_pred = matmul(X, B.astype(X.dtype, copy=False))
    if b0 is not None:
        Y_pred += b0
    return Y_pred


def _predict_centred(X, B, X_mean, X_std, Y_mean, Y_std, matmul):
    """`predict` on centred and scaled X, out of place."""
    if X_mean is not None:
        X = X - X_mean
    if X_std is not None:
        X = X / X_std
    Y_pred = matmul(X, B)
    if Y_std is not None:
        Y_pred = Y_pred * Y_std
    if Y_mean is not None:
        Y_pred = Y_pred + Y_mean
    return Y_pred


def _predict_centred_in_place(X, B, X_mean, X_std, Y_mean, Y_std, matmul, overwrite_X):
    """`predict` on centred and scaled X, in place: on a copy of X, or with
    `overwrite_X` on X itself."""
    if (X_mean is not None or X_std is not None) and not overwrite_X:
        X = X.copy()
    if X_mean is not None:
        X -= X_mean
    if X_std is not None:
        X /= X_std
    Y_pred = matmul(X, B)
    if Y_std is not None:
        Y_pred *= Y_std
    if Y_mean is not None:
        Y_pred += Y_mean
    return Y_pred
