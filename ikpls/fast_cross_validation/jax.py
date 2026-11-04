"""
Contains the PLS class which implements fast cross-validation with partial
least-squares regression using Improved Kernel PLS by Dayal and MacGregor:
https://arxiv.org/abs/2401.13185
https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

This is the JAX counterpart of ``ikpls.fast_cross_validation.numpy``. Instead of
parallelizing folds across CPU processes with joblib, it batches the folds with
``jax.vmap`` so the per-fold fits run together on a CPU/GPU/TPU. The per-fold training
:math:`\\mathbf{X}^{\\mathbf{T}}\\mathbf{W}\\mathbf{X}` and/or
:math:`\\mathbf{X}^{\\mathbf{T}}\\mathbf{W}\\mathbf{Y}` (centered/scaled/weighted using
training-set statistics) are obtained by rank-update from the dataset-wide matrices using
``cvmatrix`` with its JAX backend (``cvmatrix[jax]``).

Two entry points are provided:

* ``cross_validate``: the stateful counterpart of
  ``ikpls.fast_cross_validation.numpy.PLS.cross_validate`` for a SINGLE dataset. It
  accepts arbitrary (possibly size-imbalanced) fold labels and returns a dict mapping
  each fold to its metric. Its per-fold computation is compiled per combination of
  input shapes, settings and `A` for the most recently used `metric_function`, and
  later calls with the same function reuse it.
* ``stateless_cross_validate``: a purely functional variant that takes the validation
  indices of equally sized folds as a 2-D array and returns the metrics stacked over the
  folds. It is ``jax.jit``- and ``jax.vmap``-able, so MANY independent datasets (e.g. one
  local model per sample, per pixel, or per experimental condition), each with its own
  folds and, optionally, its own sample weights, can be cross-validated at once by
  ``jax.vmap``-ing it over a stack of datasets -- the fast cross-validation analogue of
  ``ikpls.jax.PLS.stateless_fit``. Like ``stateless_fit`` it is ``jax.jit``-compiled
  internally, so eager calls are fast too.

Both Improved Kernel PLS Algorithm #1 and #2 are supported, mirroring
``ikpls.fast_cross_validation.numpy``:

* Algorithm #2 fits each fold directly from the downdated
  :math:`\\mathbf{X}^{\\mathbf{T}}\\mathbf{W}\\mathbf{X}` (K, K) and
  :math:`\\mathbf{X}^{\\mathbf{T}}\\mathbf{W}\\mathbf{Y}` (K, M). It is efficient when
  ``X`` has more rows than columns.
* Algorithm #1 gathers each fold's training ``X`` (centered/scaled/weighted using the
  downdated training statistics) and the downdated
  :math:`\\mathbf{X}^{\\mathbf{T}}\\mathbf{W}\\mathbf{Y}`. It avoids forming
  :math:`\\mathbf{X}^{\\mathbf{T}}\\mathbf{W}\\mathbf{X}` and is efficient when ``X`` has
  more columns than rows (but materializes per-fold training rows, so use ``batch_size``
  to bound memory).

Algorithm #1 and #2 produce identical regression coefficients.

Note that the JAX implementations do not emit any underflow warning, in any path
(single ``fit`` or cross-validation). Degenerate folds (training sets with too few
non-zero weights to compute centering/scaling statistics) are still rejected up-front
with a ``ValueError`` before the vmapped computation (by ``stateless_cross_validate``
only when its `validation_indices` and `sample_weight` are concrete; see its docstring).

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import functools
import operator
from collections import defaultdict
from collections.abc import Callable, Hashable
from typing import Any, Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from cvmatrix.cvmatrix import CVMatrix
from cvmatrix.partitioner import Partitioner
from jax.typing import ArrayLike, DTypeLike
from tqdm import tqdm

from ikpls._impl.compile_cache import _CompiledCache, constructor_settings
from ikpls._impl.jax_alg_1 import PLS as _JaxAlg1
from ikpls._impl.jax_alg_2 import PLS as _JaxAlg2
from ikpls._impl.prediction import predict as _predict


def _new_fitter(algorithm: int, dtype: DTypeLike) -> Any:
    """The internal fitter: fits B from training matrices that are already centered and
    scaled, so it applies no centering or scaling."""
    fitter_cls = _JaxAlg1 if algorithm == 1 else _JaxAlg2
    return fitter_cls(
        center_X=False, center_Y=False, scale_X=False, scale_Y=False, dtype=dtype
    )


@functools.lru_cache(maxsize=None)
def _shared_fitter(algorithm: int, dtype_type: type, dtype: DTypeLike) -> Any:
    """One internal fitter per algorithm and dtype, shared by all instances: JAX's
    caches keep every fitter the fold computations were traced with, so a fitter per
    instance would keep one per instance. It holds nothing but these settings."""
    return _new_fitter(algorithm, dtype)


def _build_stateless_body(
    pls: Any, metric_function: Callable[..., Any]
) -> Callable[..., Any]:
    """The ``stateless_cross_validate`` body, ``jax.jit``-compiled per input
    shapes/dtypes, `A` and whether `sample_weight` is None."""

    def body(X, Y, validation_indices, sample_weight, A):
        return pls._stateless_body(
            X, Y, validation_indices, A, metric_function, sample_weight
        )

    return jax.jit(body, static_argnames=("A",))


class _FittedCVMatrix:
    """
    A fitted ``CVMatrix`` as a JAX pytree, so that ``cross_validate`` can pass it to a
    cached, ``jax.jit``-compiled function as an argument instead of closing over it:
    its array attributes are the leaves, and its other attributes (settings, shapes,
    the array module) are static auxiliary data, part of the compilation cache key.
    """

    def __init__(self, cvm: CVMatrix) -> None:
        self.cvm = cvm


def _flatten_fitted_cvm(fitted: _FittedCVMatrix):
    attrs = vars(fitted.cvm)
    names = tuple(k for k, v in attrs.items() if isinstance(v, (jax.Array, np.ndarray)))
    static = tuple((k, v) for k, v in attrs.items() if k not in names)
    return [attrs[k] for k in names], (type(fitted.cvm), names, static)


def _unflatten_fitted_cvm(aux, leaves) -> _FittedCVMatrix:
    cvm_type, names, static = aux
    cvm = object.__new__(cvm_type)  # keeps a subclass from an overridden _make_cvm
    vars(cvm).update(static)
    vars(cvm).update(zip(names, leaves))
    return _FittedCVMatrix(cvm)


jax.tree_util.register_pytree_node(
    _FittedCVMatrix, _flatten_fitted_cvm, _unflatten_fitted_cvm
)


def _build_cv_folds(
    pls: Any, metric_function: Callable[..., Any]
) -> Callable[..., Any]:
    """The per-fold computation of ``cross_validate``: the fold function
    ``jax.vmap``-ed over a batch of folds, ``jax.jit``-compiled per fitted
    ``CVMatrix`` structure, input shapes and `A`."""

    def body(fitted, validation_indices, training_indices, A):
        fold_fn = pls._make_fold_fn(fitted.cvm, A, metric_function)
        if training_indices is None:
            return jax.vmap(fold_fn)(validation_indices)
        return jax.vmap(fold_fn)(validation_indices, training_indices)

    return jax.jit(body, static_argnames=("A",))


# Separate caches for weighted and unweighted calls, whose metric functions take
# different arguments, so alternating between the two does not recompile.
_stateless_bodies = {w: _CompiledCache(_build_stateless_body) for w in (False, True)}
_cv_folds = {w: _CompiledCache(_build_cv_folds) for w in (False, True)}


def _as_array(a: Any) -> Union[np.ndarray, jax.Array]:
    return a if isinstance(a, (jax.Array, np.ndarray)) else np.asarray(a)


def _dataset_label(index: Iterable[Any]) -> str:
    index = tuple(int(i) for i in index)
    return str(index[0] if len(index) == 1 else index)


def _fold_location(bad: np.ndarray) -> str:
    """Names the first True of `bad` (..., n_folds): its fold, and its dataset if any."""
    *dataset, fold = np.argwhere(bad)[0]
    if not dataset:
        return f"Fold {int(fold)}"
    return f"Dataset {_dataset_label(dataset)}, fold {int(fold)}"


def _validation_index_array(
    validation_indices: ArrayLike, concrete: bool
) -> Union[np.ndarray, jax.Array]:
    """
    `validation_indices` checked to be a 2-D integer array: as a NumPy array when
    `concrete`, otherwise the tracer as given.
    """
    ragged_msg = (
        "`validation_indices` must be a rectangular (n_folds, N_val) integer "
        "array: all folds must have the same size. Use `cross_validate` for "
        "size-imbalanced folds."
    )
    if concrete:
        try:
            vi = np.asarray(validation_indices)
        except (ValueError, TypeError) as err:
            raise ValueError(ragged_msg) from err
        if vi.dtype == object:
            raise ValueError(
                f"{ragged_msg} Got an array of dtype object (ragged rows, or "
                "Python objects that are not integers)."
            )
    else:
        vi = validation_indices
    if vi.ndim != 2:
        raise ValueError(
            "`validation_indices` must be a 2-D array of shape (n_folds, N_val); "
            f"got shape {vi.shape}."
        )
    if not jnp.issubdtype(vi.dtype, jnp.integer):
        raise ValueError(
            f"`validation_indices` must be an integer array; got dtype {vi.dtype}."
        )
    return vi


class PLS:
    r"""
    Implements fast cross-validation with partial least-squares regression using
    Improved Kernel PLS by Dayal and MacGregor:
    https://arxiv.org/abs/2401.13185
    https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

    The per-fold training kernel matrices are computed by rank-update with ``cvmatrix``
    (JAX backend) and the folds are batched with ``jax.vmap``.

    Parameters
    ----------
    algorithm : int, default=1
        Whether to use Improved Kernel PLS Algorithm #1 or #2. Generally, Algorithm #1
        is faster if `X` has fewer rows than columns, while Algorithm #2 is faster if
        `X` has more rows than columns. Both yield identical regression coefficients.

    center_X : bool, default=True
        Whether to center `X` before fitting by subtracting its row of column-wise
        means from each row. The row of column-wise means is computed on the training
        set for each fold to avoid data leakage.

    center_Y : bool, default=True
        Whether to center `Y`. See `center_X`.

    scale_X : bool, default=True
        Whether to scale `X` before fitting by dividing each row with the row of `X`'s
        column-wise standard deviations, computed on the training set for each fold.

    scale_Y : bool, default=True
        Whether to scale `Y`. See `scale_X`.

    ddof : int, default=0
        The delta degrees of freedom to use when computing the sample standard
        deviation. A value of 0 corresponds to the biased estimate of the sample
        standard deviation, while a value of 1 corresponds to Bessel's correction.

    copy : bool, default=True
        Whether to copy `X`, `Y`, and `sample_weight` when cross-validating.

    dtype : DTypeLike, default=jnp.float64
        The float datatype to use. Using a lower precision than float64 will yield
        significantly worse results with an increasing number of components due to
        propagation of numerical errors.

    Raises
    ------
    ValueError
        If `algorithm` is not 1 or 2.

    Notes
    -----
    Any centering and scaling is undone before returning predictions so that
    predictions are on the original scale. If both centering and scaling are True, then
    the data is first centered and then scaled.

    ``cross_validate`` cross-validates a single dataset (any fold labels, dict result).
    ``stateless_cross_validate`` is the ``jax.jit``/``jax.vmap``-able functional
    variant (equal-size folds given as a 2-D index array, stacked result); ``jax.vmap``
    it over a stack of datasets to cross-validate many independent models at once.

    See Also
    --------
    ikpls.fast_cross_validation.numpy.PLS :
        The NumPy/joblib counterpart. Its ``cross_validate`` shares the same return type
        (a dict mapping each fold to its metric) and ``metric_function`` contract, but
        the performance knobs differ: this class takes ``batch_size`` and
        ``show_progress`` (vmap-based), whereas the NumPy class takes ``n_jobs`` and
        ``verbose`` (joblib-based).
    """

    def __init__(
        self,
        algorithm: int = 1,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        ddof: int = 0,
        copy: bool = True,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        if algorithm not in (1, 2):
            raise ValueError(
                f"Invalid algorithm: {algorithm}. Algorithm must be 1 or 2."
            )
        self.algorithm = algorithm
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.ddof = ddof
        self.copy = copy
        self.dtype = dtype
        self.name = f"Improved Kernel PLS Algorithm #{algorithm} (JAX fast CV)"
        # Internal fitter: fits B from the (already centered/scaled) training matrices via
        # the algorithm's lax.scan (the JAX fit emits no host-side callbacks, so it runs
        # under jax.vmap). Centering/scaling are handled by cvmatrix, so the fitter applies
        # none.
        self._fitter = self._make_fitter()
        self.A = None
        self.N = None
        self.K = None
        self.M = None
        self._cvm = None  # The CVMatrix of the most recent `cross_validate` call.

    def _make_cvm(self) -> CVMatrix:
        """Construct an (unfitted) ``cvmatrix.CVMatrix`` with this instance's settings.
        Algorithm #1 never uses the dataset-wide X^T W X, so it is not formed."""
        return CVMatrix(
            center_X=self.center_X,
            center_Y=self.center_Y,
            scale_X=self.scale_X,
            scale_Y=self.scale_Y,
            ddof=self.ddof,
            dtype=self.dtype,
            copy=self.copy,
            backend="jax",
            compute_XTX=self.algorithm != 1,
            compute_XTY=True,
        )

    @staticmethod
    def _training_indices(val_idx: jax.Array, N: int) -> jax.Array:
        """
        The training indices of a fold: the ascending complement of ``val_idx`` in
        ``range(N)``. Trace-safe (the output size ``N - N_val`` is static), so it runs
        inside ``jax.jit``/``jax.vmap`` and never materializes an (n_folds, N_train)
        index array on the host.
        """
        # A single stable sort of the validation mask puts the training positions first,
        # in ascending order (cheaper to compile than `jnp.flatnonzero(size=...)`, which
        # lowers to two cumulative sums).
        is_val = jnp.zeros(N, dtype=bool).at[val_idx].set(True)
        return jnp.argsort(is_val, stable=True)[: N - val_idx.shape[0]]

    def _make_fitter(self):
        try:
            return _shared_fitter(self.algorithm, type(self.dtype), self.dtype)
        except TypeError:  # an unhashable dtype
            return _new_fitter(self.algorithm, self.dtype)

    def _fit_B(
        self,
        operand: jax.Array,
        XTY: jax.Array,
        A: int,
        K: int,
        M: int,
        sample_weight: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute the (A, K, M) regression-coefficients tensor from a fold's training
        matrices via the algorithm's ``lax.scan`` fit. ``operand`` is the training ``X``
        (N_train, K) for Algorithm #1 or the training ``XTX`` (K, K) for Algorithm #2;
        ``XTY`` is the training ``XTY`` (K, M). ``sample_weight`` holds the training
        weights (N_train,) for a weighted Algorithm #1 fit and is None otherwise.
        Reverse-mode-differentiable and ``jax.vmap``-safe.

        As in ``stateless_fit`` and the NumPy implementation, the coefficients of the
        last numerically stable component are carried forward: from the first
        component whose weight norm underflows or whose score norm collapses relative
        to the largest one so far (e.g. beyond the numerical rank of the training
        ``X``), every later component repeats the previous coefficients, and if the
        first component is unstable, all coefficients are zero. ``stateless_fit``
        applies the same criteria after its loop; here they are tracked inside the
        loop, which adds fewer operations to compile, as ``cross_validate`` compiles
        this for every new shape.
        """
        f = self._fitter
        if type(f) is not (_JaxAlg1 if self.algorithm == 1 else _JaxAlg2) or (
            f.dtype != self.dtype
        ):
            f = self._make_fitter()  # `algorithm` or `dtype` changed after construction
        init = (
            (
                XTY,
                jnp.zeros((A, K), dtype=self.dtype),
                jnp.zeros((A, K), dtype=self.dtype),
                jnp.zeros((K, M), dtype=self.dtype),
            ),
            jnp.zeros((), dtype=self.dtype),  # running max of the score norms
            jnp.ones((), dtype=bool),  # whether all components so far are stable
        )

        weight_args = () if sample_weight is None else (sample_weight,)
        eps = f.eps

        def body(carry, i):
            fit_carry, max_tTt, stable = carry
            B_prev = fit_carry[-1]
            fit_carry, outs = f._scan_body(operand, M, K, A, fit_carry, i, *weight_args)
            # Both algorithms' scan bodies output the score norm ``tTt``, the weight
            # norm and the regression coefficients ``B`` last (Algorithm #1: w, p, q,
            # r, t, tTt, norm, B; Algorithm #2: w, p, q, r, tTt, norm, B).
            tTt, norm, B_i = outs[-3:]
            max_tTt = jnp.maximum(max_tTt, tTt)
            stable = stable & ~((jnp.abs(norm) <= eps) | (tTt <= eps * max_tTt))
            B_i = jnp.where(stable, B_i, B_prev)
            return (fit_carry[:-1] + (B_i,), max_tTt, stable), B_i

        _, B = jax.lax.scan(body, init, jnp.arange(A))
        return B

    def _predict_on_val(
        self,
        cvm: CVMatrix,
        val_idx: jax.Array,
        B: jax.Array,
        x_mean: Optional[jax.Array],
        x_std: Optional[jax.Array],
        y_mean: Optional[jax.Array],
        y_std: Optional[jax.Array],
    ) -> jax.Array:
        """Predict on the held-out rows for all components, undoing train preprocessing."""
        return _predict(  # (N_val, K) raw, unweighted rows @ (A, K, M) -> (A, N_val, M)
            cvm.X[val_idx],
            B,
            x_mean if self.center_X else None,
            x_std if self.scale_X else None,
            y_mean if self.center_Y else None,
            y_std if self.scale_Y else None,
        )  # the broadcasting matmul: here faster and leaner than `_jax_matmul`

    def _fold_predict_alg1(
        self,
        cvm: CVMatrix,
        val_idx: jax.Array,
        train_idx: Optional[jax.Array],
        A: int,
        K: int,
        M: int,
    ) -> jax.Array:
        """Algorithm #1 per-fold fit + predict (gathers the training rows of ``X``, and
        of the weights for a weighted fit). ``train_idx`` is derived on device from
        ``val_idx`` when it is None."""
        xty_t, (x_mean, x_std, y_mean, y_std) = cvm.training_XTY(val_idx)
        if train_idx is None:
            train_idx = self._training_indices(val_idx, cvm.N)
        x_train = cvm.X[train_idx]  # (N_train, K)
        if self.center_X:
            x_train = x_train - x_mean
        if self.scale_X:
            x_train = x_train / x_std
        w_train = None if cvm.weights is None else cvm.weights[train_idx].ravel()
        b = self._fit_B(x_train, xty_t, A, K, M, w_train)
        return self._predict_on_val(cvm, val_idx, b, x_mean, x_std, y_mean, y_std)

    def _fold_predict_alg2(
        self, cvm: CVMatrix, val_idx: jax.Array, A: int, K: int, M: int
    ) -> jax.Array:
        """Algorithm #2 per-fold fit + predict (uses the downdated ``XTX``/``XTY``)."""
        (xtx_t, xty_t), (x_mean, x_std, y_mean, y_std) = cvm.training_XTX_XTY(val_idx)
        b = self._fit_B(xtx_t, xty_t, A, K, M)
        return self._predict_on_val(cvm, val_idx, b, x_mean, x_std, y_mean, y_std)

    def _make_fold_fn(
        self, cvm: CVMatrix, A: int, metric_function: Callable[..., Any]
    ) -> Callable[[jax.Array], Any]:
        """
        Build the per-fold function ``val_idx (N_val,) -> metric`` for a fitted
        ``cvm``: fit on the complement of ``val_idx``, predict the held-out rows for all
        `A` components, and evaluate ``metric_function`` on device. Shared by
        ``cross_validate`` and ``stateless_cross_validate`` so both compute the same
        per-fold results (up to floating-point round-off from XLA fusing the ops
        differently under different jit boundaries); ``jax.vmap`` it over a stack of
        validation-index rows. For Algorithm #1 it optionally also takes the fold's
        training indices; ``cross_validate`` computes them on the host, which compiles
        faster than deriving them on device.
        """
        K, M = cvm.K, cvm.M
        has_weights = cvm.weights is not None

        def evaluate(val_idx, y_pred):
            # On-device metric: gather the held-out targets (and sample_weight) and apply
            # the JAX-traceable `metric_function` inside the vmapped/jitted computation, so
            # no per-fold host round-trip is needed and only the metric (not the full
            # prediction tensor) is transferred back.
            y_true = cvm.Y[val_idx]
            if has_weights:
                return metric_function(y_true, y_pred, cvm.weights[val_idx].ravel())
            return metric_function(y_true, y_pred)

        if self.algorithm == 1:

            def fold_fn(val_idx, train_idx=None):
                y_pred = self._fold_predict_alg1(cvm, val_idx, train_idx, A, K, M)
                return evaluate(val_idx, y_pred)

        else:

            def fold_fn(val_idx):
                y_pred = self._fold_predict_alg2(cvm, val_idx, A, K, M)
                return evaluate(val_idx, y_pred)

        return fold_fn

    def _check_cv_folds(
        self,
        N: int,
        fold_items: Iterable[Tuple[Hashable, np.ndarray]],
        sample_weight: Optional[ArrayLike],
    ) -> None:
        """
        Host-side pre-flight rejection of degenerate folds for `cross_validate`. With
        weights, the cvmatrix JAX backend skips its per-fold ``ValueError`` checks while
        the folds are traced by ``jax.vmap``, so they are performed here instead,
        mirroring the numpy backend's behavior. ``fold_items`` yields
        ``(fold, validation_indices)`` pairs.
        """
        needs_stats = self.center_X or self.center_Y or self.scale_X or self.scale_Y
        if not needs_stats:
            return
        needs_std = self.scale_X or self.scale_Y
        if sample_weight is None:
            total_nonzero = N
        else:
            # Count non-zeros on the dtype-cast weights actually used on-device, so a
            # weight that underflows to 0 under ``self.dtype`` is rejected just as the
            # numpy backend (which validates the already-cast sample_weight) would.
            w = np.asarray(sample_weight, dtype=self.dtype).ravel()
            total_nonzero = int(np.count_nonzero(w))
        for fold, vi in fold_items:
            if sample_weight is None:
                nnz_train = N - int(vi.shape[0])
            else:
                nnz_train = total_nonzero - int(np.count_nonzero(w[vi]))
            if nnz_train == 0:
                raise ValueError(
                    f"Fold {fold!r}: the training set has no non-zero weights, so "
                    "centering/scaling statistics cannot be computed."
                )
            if needs_std and nnz_train <= self.ddof:
                raise ValueError(
                    f"Fold {fold!r}: the training set has too few non-zero weights "
                    f"(={nnz_train}) relative to ddof (={self.ddof}) to compute the "
                    "sample standard deviation."
                )

    def _check_x64(self) -> None:
        if np.dtype(self.dtype).itemsize >= 8 and not jax.config.jax_enable_x64:
            raise ValueError(
                f"dtype {np.dtype(self.dtype).name} requires jax_enable_x64=True at "
                "call time (enable it before tracing); cvmatrix would otherwise toggle "
                "the flag mid-trace."
            )

    def _check_stateless_shapes(
        self, N: int, N_Y: int, vi_shape: Tuple[int, int]
    ) -> None:
        """Shape-only checks of ``stateless_cross_validate``, valid under tracing too."""
        if N_Y != N:
            raise ValueError(
                f"`X` and `Y` must have the same number of rows; got {N} and {N_Y}."
            )
        n_folds, n_val = vi_shape
        if n_folds == 0 or n_val == 0:
            raise ValueError(
                "`validation_indices` must hold at least one fold with at least one "
                f"validation index; got shape {vi_shape}."
            )
        if n_val >= N:
            raise ValueError(
                f"Each fold's validation set (N_val={n_val}) must be smaller than the "
                f"number of samples (N={N}) so that its training set is non-empty."
            )
        # Every training set has N - N_val samples (with weights, at most that many
        # non-zero ones).
        if (self.scale_X or self.scale_Y) and N - n_val <= self.ddof:
            raise ValueError(
                f"Each fold's training set has N - N_val = {N - n_val} samples, too few "
                f"to compute the standard deviation for scaling with ddof={self.ddof}."
            )

    def _check_stateless_weights(
        self, sample_weight: Union[np.ndarray, jax.Array], N: int
    ) -> None:
        """
        The shape of `sample_weight` is checked also when it is traced, its values only
        when it is concrete: traced weights are the caller's to validate.
        """
        if sample_weight.shape not in ((N,), (N, 1)):
            raise ValueError(
                f"`sample_weight` must have shape (N,) or (N, 1) with N={N}; got "
                f"{sample_weight.shape}."
            )
        if isinstance(sample_weight, jax.core.Tracer):
            return
        if np.any(np.asarray(sample_weight, dtype=self.dtype) < 0):
            raise ValueError("Weights must be non-negative.")

    def _check_concrete_folds(
        self,
        vi: np.ndarray,
        N: int,
        sample_weight: Optional[Union[np.ndarray, jax.Array]],
    ) -> None:
        """
        Host-side rejection of out-of-range, duplicated and degenerate folds. Traced
        indices cannot be checked; the caller is responsible for them, as for
        cvmatrix's own deferred checks. So are the training sets' non-zero weights when
        the weights are traced.
        """
        self._check_index_values(vi, N)
        if sample_weight is not None and not isinstance(sample_weight, jax.core.Tracer):
            w = np.asarray(sample_weight, dtype=self.dtype).reshape(-1)
            self._check_training_weights(vi, w)

    @staticmethod
    def _check_index_values(vi: np.ndarray, N: int) -> None:
        """Concrete indices (..., n_folds, N_val): in range(N), distinct within a fold."""
        bad = np.any((vi < 0) | (vi >= N), axis=-1)
        if np.any(bad):
            raise ValueError(
                f"{_fold_location(bad)}: `validation_indices` contains indices outside "
                f"range(N) with N={N}."
            )
        vi_sorted = np.sort(vi, axis=-1)
        bad = np.any(vi_sorted[..., 1:] == vi_sorted[..., :-1], axis=-1)
        if np.any(bad):
            raise ValueError(
                f"{_fold_location(bad)}: each fold's validation indices must be "
                "distinct."
            )

    def _check_training_weights(self, vi: np.ndarray, w: np.ndarray) -> None:
        """
        Each fold's training set needs a non-zero weight for centering or scaling, and
        more than `ddof` of them for scaling. `vi` (..., n_folds, N_val) holds concrete,
        valid validation indices and `w` (..., N) the weights cast to `dtype`, with
        broadcastable leading axes.
        """
        if not (self.center_X or self.center_Y or self.scale_X or self.scale_Y):
            return
        nonzero = w != 0
        batch = np.broadcast_shapes(vi.shape[:-2], w.shape[:-1])
        nonzero_val = np.take_along_axis(
            np.broadcast_to(nonzero[..., None, :], batch + (vi.shape[-2], w.shape[-1])),
            np.broadcast_to(vi, batch + vi.shape[-2:]),
            axis=-1,
        ).sum(axis=-1)
        nnz_train = nonzero.sum(axis=-1)[..., None] - nonzero_val  # (..., n_folds)
        bad = nnz_train == 0
        if np.any(bad):
            raise ValueError(
                f"{_fold_location(bad)}: the training set has no non-zero weights, so "
                "centering/scaling statistics cannot be computed."
            )
        bad = (self.scale_X or self.scale_Y) & (nnz_train <= self.ddof)
        if np.any(bad):
            raise ValueError(
                f"{_fold_location(bad)}: the training set has too few non-zero weights "
                f"(={nnz_train[tuple(np.argwhere(bad)[0])]}) relative to ddof "
                f"(={self.ddof}) to compute the sample standard deviation."
            )

    def validate_folds(
        self,
        validation_indices: ArrayLike,
        N: int,
        sample_weight: Optional[ArrayLike] = None,
    ) -> None:
        """
        Checks the concrete validation indices and sample weights of one dataset or a
        stack of datasets on the host, before `stateless_cross_validate` traces them.
        `stateless_cross_validate` runs these checks itself on concrete inputs, but it
        cannot check traced ones, e.g. per-dataset folds and weights under
        ``jax.vmap``, and invalid traced inputs silently give wrong or NaN results. The
        checks are vectorized over the datasets and use this instance's `center_X`,
        `center_Y`, `scale_X`, `scale_Y`, `ddof` and `dtype`.

        Parameters
        ----------
        validation_indices : Integer array of shape (..., n_folds, N_val)
            The validation indices of each dataset's folds, as passed to
            `stateless_cross_validate`. Leading axes, e.g. (B,), index the datasets.

        N : int
            The number of samples in each dataset.

        sample_weight : Array of shape (..., N) or (..., N, 1), or None, optional, \
default=None
            The weights of each dataset's samples. Leading axes must broadcast with
            those of `validation_indices`, e.g. (N,) for weights shared by all
            datasets.

        Raises
        ------
        ValueError
            If `validation_indices` is not an integer array of at least two dimensions
            with ``0 < N_val < N``; if an index is outside ``range(N)`` or repeated
            within a fold; if each fold's training set of ``N - N_val`` samples is too
            small to compute the standard deviation with this `ddof`; if
            `sample_weight` does not have `N` weights per dataset, its leading axes do
            not broadcast with those of `validation_indices`, or it contains negative
            values after the cast to `dtype`; or if a fold's training set has no
            non-zero weights when centering or scaling, or not more than `ddof` of them
            when scaling. The checks run in this order, and the message of the first
            failing check names its first failing dataset and fold, or, for negative
            weights, the index into `sample_weight`.
        """
        vi = np.asarray(validation_indices)
        if vi.ndim < 2 or not np.issubdtype(vi.dtype, np.integer):
            raise ValueError(
                "`validation_indices` must be an integer array of shape "
                f"(..., n_folds, N_val); got shape {vi.shape} and dtype {vi.dtype}."
            )
        self._check_stateless_shapes(N, N, vi.shape[-2:])
        self._check_index_values(vi, N)
        if sample_weight is not None:
            self._check_training_weights(
                vi, self._weights_per_dataset(sample_weight, N, vi)
            )

    def _weights_per_dataset(
        self, sample_weight: ArrayLike, N: int, vi: np.ndarray
    ) -> np.ndarray:
        """`sample_weight` as checked, non-negative (..., N) weights cast to `dtype`."""
        w = np.asarray(sample_weight, dtype=self.dtype)
        if w.ndim >= 2 and w.shape[-2:] == (N, 1):
            w = w[..., 0]
        if w.ndim == 0 or w.shape[-1] != N:
            raise ValueError(
                f"`sample_weight` must have shape (..., N) or (..., N, 1) with N={N}; "
                f"got {w.shape}."
            )
        try:
            np.broadcast_shapes(vi.shape[:-2], w.shape[:-1])
        except ValueError:
            raise ValueError(
                f"The leading axes of `sample_weight` {w.shape} and "
                f"`validation_indices` {vi.shape} must broadcast."
            ) from None
        negative = np.any(w < 0, axis=-1)
        if negative.ndim and np.any(negative):
            index = ", ".join(str(int(i)) for i in np.argwhere(negative)[0])
            raise ValueError(
                f"Weights must be non-negative, but `sample_weight[{index}]` has a "
                "negative weight."
            )
        if np.any(negative):
            raise ValueError("Weights must be non-negative.")
        return w

    def stateless_cross_validate(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        A: int,
        validation_indices: ArrayLike,
        metric_function: Callable[..., Any],
        sample_weight: Optional[ArrayLike] = None,
    ) -> Any:
        r"""
        Cross-validates the PLS model with `A` components on a single dataset given the
        validation indices of its equally sized folds, and returns the per-fold metrics
        stacked over the folds. This is the functional, ``jax.jit``/``jax.vmap``-able
        counterpart of `cross_validate` (the fast cross-validation analogue of
        ``ikpls.jax.PLS.stateless_fit``): nothing is stored on the instance and the
        result is a pytree of arrays rather than a dict. Calls are ``jax.jit``-compiled
        internally, once per settings, input shapes, `A` and whether `sample_weight` is
        given, for the most recently used `metric_function` (see `metric_function`),
        so eager calls run at compiled speed.

        Its main use is cross-validating MANY independent datasets at once: ``jax.vmap``
        it over a leading dataset axis of `X` (B, N, K), `Y` (B, N, M),
        `validation_indices` (B, n_folds, N_val) and, optionally, `sample_weight`
        (B, N) and wrap the result in ``jax.jit`` so a single compiled program fits all
        ``B * n_folds`` models together on the CPU/GPU/TPU. A plain Python loop over
        `cross_validate` would instead handle one dataset at a time, with a host
        round-trip per dataset. In local PLS, for example, each dataset holds the
        nearest neighbours of one unknown sample, weighted by their distance to it.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M) or (N,)
            Target variables.

        A : int
            Number of components in the PLS model.

        validation_indices : Integer array of shape (n_folds, N_val)
            Row ``i`` holds the indices (into the `N` samples) of fold ``i``'s validation
            set; its training set is the complement. All folds must have the same
            validation-set size ``0 < N_val < N`` (fixed shapes are what make the
            computation ``jax.vmap``-able). Balanced k-fold labels ``folds`` (N,) convert
            with ``np.stack([np.flatnonzero(folds == f) for f in np.unique(folds)])``.
            Range, uniqueness and degenerate-fold checks run host-side when the indices
            are concrete: a NumPy array or list, or a concrete JAX array, including one
            closed over inside the caller's ``jax.jit``. When the indices are traced (an
            argument of the caller's ``jax.jit`` or ``jax.vmap``, e.g. per-dataset
            indices) they cannot be checked: an out-of-range or duplicated index then
            silently corrupts that fold, so check traced indices beforehand with
            `validate_folds`.

        metric_function : Callable
            Computes a metric from the true ``Y_val`` (N_val, M) and the predicted
            ``Y_pred`` (A, N_val, M) -- and, if ``sample_weight`` is not None, also
            ``weights_val`` (N_val,) -- returning any pytree of arrays. ``Y_pred``
            contains a prediction for all `A` components. **Must be JAX-traceable**
            (``jax.numpy``-based): it is evaluated on device inside ``jax.vmap`` over
            the folds. The internally compiled program is kept for the most recently
            used `metric_function` per settings (separately for weighted and
            unweighted calls), matched by identity (the same method of the same object
            counts as the same). So define it once rather than per call: another
            object -- a lambda created per call, or a new instance of a callable
            class, even an equal one -- compiles anew and replaces the program, as
            does alternating between two functions. To evaluate several metrics,
            return them all from one function. It is traced when first compiled, so
            what it reads then -- values it closes over, globals, or its own
            attributes -- is fixed for later calls with the same function. The
            function, and what it closes over, stays alive until another function
            replaces it or its settings are evicted: programs are kept for the 64
            most recently used settings.

        sample_weight : Array of shape (N,) or (N, 1), or None, optional, default=None
            Weights for each observation. If None, all observations are weighted
            equally. Must be non-negative, and each fold's training set needs a non-zero
            weight when centering or scaling, and more than `ddof` of them when
            scaling. Concrete weights, including ones closed over inside the caller's
            ``jax.jit``, are checked host-side (the training sets when
            `validation_indices` is concrete, too). Traced weights, e.g. one weight
            vector per dataset under ``jax.vmap``, cannot be checked: invalid ones
            silently give wrong or NaN results, so check them beforehand with
            `validate_folds`.

        Returns
        -------
        metrics : Any
            The pytree returned by `metric_function`, with every leaf stacked over the
            folds along a new leading axis of length ``n_folds`` (fold ``i`` at index
            ``i``, in the row order of `validation_indices`). Under an outer
            ``jax.vmap`` over datasets, a further leading axis of length ``B`` is added.

        Raises
        ------
        ValueError
            If `X` and `Y` have different numbers of rows; if `validation_indices` is
            not a rectangular 2-D integer array with ``0 < N_val < N``; if (when
            concrete) it contains out-of-range or, within a row, duplicate indices; if
            `sample_weight` has the wrong shape or (when concrete) contains negative
            values; if the training-set size ``N - N_val`` is too small for the
            requested scaling with this `ddof` (checked even under tracing, since it
            depends on shapes only); if a weighted fold's training set has too few
            non-zero weights (checked when `validation_indices` and `sample_weight` are
            concrete); or if ``dtype`` is 64-bit while ``jax_enable_x64`` is off.

        Notes
        -----
        Per-fold results agree with `cross_validate` up to floating-point round-off (the
        same per-fold function is used; XLA may fuse operations differently across jit
        boundaries). As in `ikpls.jax.PLS` and the NumPy fast cross-validation, and
        with the same stability criteria, predictions with more components than a
        fold's numerically stable ones (e.g. beyond the numerical rank of its training
        `X`) equal those with its last stable component. A component right at the
        stability threshold may be classified differently under different round-off.
        With Algorithm #2, whose score norms come from the downdated ``X^T W X``, the
        first component past the numerical rank often lies close to the threshold, so
        either implementation may admit one round-off component there, and their
        results for more components may then differ.

        If all datasets share one split, close over a single concrete (n_folds, N_val)
        `validation_indices` and ``jax.vmap`` over `X` and `Y` only; the fold validity
        checks then run host-side, also inside the caller's ``jax.jit``.

        The internally compiled program is built from a new instance constructed from
        this instance's constructor arguments, read from the attributes of the same
        names (the scikit-learn convention). A subclass must follow that convention,
        as other instance state is not part of the compiled program. A subclass
        constructor's ``**kwargs`` stand for its base class's constructor arguments.
        If the constructor arguments cannot be read or hashed, or a new instance
        constructed from them does not reproduce them, the call runs un-jitted.

        Memory per dataset vmapped together, in elements (float64: 8 bytes each), as a
        rough guide: Algorithm #1 materializes the training rows of every fold, about
        ``1.3-2.3 * n_folds * (N - N_val) * K``; Algorithm #2 holds the fold-wise
        ``X_val^T X_val`` products and the downdated training ``X^T X`` at once, about
        ``2-4 * n_folds * K * K``. Both also stack per-component outputs, roughly
        ``2-3 * n_folds * A * K`` more. To size the outer batch, measure your shapes, e.g.
        ``jax.jit(f).lower(*args).compile().memory_analysis()``.

        Examples
        --------
        Cross-validate 1,000 independent datasets of 200 samples and 110 features with
        10-fold CV each, on the default device:

        >>> import jax, jax.numpy as jnp, numpy as np
        >>> from ikpls.fast_cross_validation.jax import PLS
        >>> jax.config.update("jax_enable_x64", True)
        >>> B, N, K, M, A, n_folds = 1000, 200, 110, 1, 30, 10
        >>> rng = np.random.default_rng(0)
        >>> X = rng.uniform(size=(B, N, K)); Y = rng.uniform(size=(B, N, M))
        >>> # Random equal-size folds per dataset: (B, n_folds, N // n_folds).
        >>> val_idx = np.stack([rng.permutation(N).reshape(n_folds, -1) for _ in range(B)])
        >>> mse = lambda Y_true, Y_pred: jnp.mean((Y_true - Y_pred) ** 2, axis=-2)  # (A, M)
        >>> pls = PLS(algorithm=2, center_X=True, center_Y=True, scale_X=False, scale_Y=False)
        >>> cv = jax.jit(jax.vmap(lambda x, y, vi: pls.stateless_cross_validate(x, y, A, vi, mse)))
        >>> fold_mse = cv(X, Y, val_idx)  # (B, n_folds, A, M)
        >>> best_A = jnp.argmin(fold_mse.mean(axis=1), axis=1) + 1  # (B, M)

        With one weight vector per dataset, check the concrete weights and folds before
        they are traced, and use a metric that takes the validation weights:

        >>> W = rng.uniform(0.5, 1.5, size=(B, N))
        >>> pls.validate_folds(val_idx, N, W)
        >>> wmse = lambda Y_true, Y_pred, w: jnp.average(
        ...     (Y_true - Y_pred) ** 2, axis=-2, weights=w
        ... )  # (A, M)
        >>> cv_w = jax.jit(jax.vmap(
        ...     lambda x, y, vi, w: pls.stateless_cross_validate(x, y, A, vi, wmse, w)
        ... ))
        >>> fold_wmse = cv_w(X, Y, val_idx, W)  # (B, n_folds, A, M)
        """
        self._check_x64()
        A = operator.index(A)  # a static Python int, also when given as a 0-d array
        X = _as_array(X)
        Y = _as_array(Y)
        # Decide concreteness on the argument as given: inside the caller's jax.jit,
        # jnp.asarray would stage even a closed-over NumPy array into a tracer. The
        # same holds for `sample_weight`, whose checks also skip only actual tracers.
        concrete = not isinstance(validation_indices, jax.core.Tracer)
        vi = _validation_index_array(validation_indices, concrete)
        N = np.shape(X)[0]
        self._check_stateless_shapes(N, np.shape(Y)[0], vi.shape)
        if sample_weight is not None:
            sample_weight = _as_array(sample_weight)
            self._check_stateless_weights(sample_weight, N)
        if concrete:
            self._check_concrete_folds(vi, N, sample_weight)
        validation_indices = jnp.asarray(validation_indices)
        settings = self._settings()
        compiled = None
        if settings is not None:
            cache = _stateless_bodies[sample_weight is not None]
            compiled = cache.get(type(self), settings, metric_function)
        if compiled is not None:
            return compiled(X, Y, validation_indices, sample_weight, A=A)
        return self._stateless_body(
            X, Y, validation_indices, A, metric_function, sample_weight
        )

    def _settings(self) -> Optional[tuple]:
        """The constructor settings keying the compiled programs; see
        ``constructor_settings``."""
        return constructor_settings(self)

    def _vmapped_fold_fn(
        self, cvm: CVMatrix, A: int, metric_function: Callable[..., Any]
    ) -> Callable[..., Any]:
        """
        The jitted per-fold function of `cross_validate`, ``jax.vmap``-ed over a batch
        of folds, for a fitted ``cvm``. With hashable settings, it is the cached
        compiled program, reused across calls with the same shapes, settings,
        `metric_function` and `A`, which takes the fitted matrices as arguments;
        otherwise, a new ``jax.jit`` that closes over ``cvm``.
        """
        settings = self._settings()
        compiled = None
        if settings is not None:
            cache = _cv_folds[cvm.weights is not None]
            compiled = cache.get(type(self), settings, metric_function)
        if compiled is None:
            return jax.jit(jax.vmap(self._make_fold_fn(cvm, A, metric_function)))
        fitted = _FittedCVMatrix(cvm)

        def vmapped(validation_indices, training_indices=None):
            return compiled(fitted, validation_indices, training_indices, A=A)

        return vmapped

    def _stateless_body(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        validation_indices: jax.Array,
        A: int,
        metric_function: Callable[..., Any],
        sample_weight: Optional[ArrayLike] = None,
    ) -> Any:
        """Fit the cvmatrix globals and vmap the per-fold function over the folds."""
        cvm = self._make_cvm()
        cvm.fit(X, Y, sample_weight)  # builds the dataset-wide matrices/statistics
        fold_fn = self._make_fold_fn(cvm, A, metric_function)
        return jax.vmap(fold_fn)(validation_indices)

    def cross_validate(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        A: int,
        folds: Iterable[Hashable],
        metric_function: Callable[..., Any],
        sample_weight: Optional[ArrayLike] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> dict[Hashable, Any]:
        r"""
        Cross-validates the PLS model using `folds` splits on `X` and `Y` with `A`
        components, evaluating results with `metric_function`.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M) or (N,)
            Target variables.

        A : int
            Number of components in the PLS model.

        folds : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in `folds`
            corresponds to a different fold.

        metric_function : Callable
            Computes a metric from the true ``Y_val`` (N_val, M) and the predicted
            ``Y_pred`` (A, N_val, M) -- and, if ``sample_weight`` is not None, also
            ``weights_val`` (N_val,) -- returning any value. ``Y_pred`` contains a
            prediction for all `A` components. **Must be JAX-traceable**
            (``jax.numpy``-based): it is evaluated on device inside ``jax.vmap`` over
            the folds, so only the metric (not the prediction tensor) is transferred
            back. As for `stateless_cross_validate`, the compiled program is kept
            for the most recently used `metric_function` per settings (separately
            for weighted and unweighted calls), matched by identity (the same method
            of the same object counts as the same). So define it once rather than
            per call: another object -- a lambda created per call, or a new instance
            of a callable class, even an equal one -- compiles anew and replaces the
            program, as does alternating between two functions. To evaluate several
            metrics, return them all from one function. It is traced when first
            compiled, so what it reads then -- values it closes over, globals, or its
            own attributes -- is fixed for later calls with the same function. The
            function, and what it closes over, stays alive until another function
            replaces it or its settings are evicted: programs are kept for the 64
            most recently used settings.

        sample_weight : Array of shape (N,) or None, optional, default=None
            Weights for each observation. If None, all observations are weighted
            equally. Must be non-negative.

        batch_size : int or None, optional, default=None
            Number of folds processed together in a single ``jax.vmap`` call. ``None``
            processes all folds of a given validation-set size at once. Use a smaller
            value to bound peak memory when there are many folds and/or large `K`
            (especially for Algorithm #1, which materializes per-fold training rows).

        show_progress : bool, default=True
            Whether to display a progress bar over the vmapped fold batches.

        Returns
        -------
        metrics : dict of Hashable to Any
            A dictionary mapping each unique value in `folds` to the result of
            evaluating `metric_function` on the corresponding validation set.

        Raises
        ------
        ValueError
            If `sample_weight` are provided and not all weights are non-negative, or if a
            fold's training set has too few non-zero weights to compute the requested
            centering/scaling statistics.

        Notes
        -----
        Folds are grouped by validation-set size so that ``jax.vmap`` sees fixed shapes;
        leave-one-out and balanced k-fold use a single group, while size-imbalanced
        folds use one group per distinct size.

        The per-fold computation is compiled per combination of input shapes,
        settings, `A` and batch shape for the most recently used `metric_function`
        (see `metric_function`), and later calls with the same function reuse it,
        also with new data. To cross-validate many
        independent datasets at once, ``jax.vmap`` `stateless_cross_validate` over a
        stack of datasets instead of looping over this method, which handles one
        dataset per call. As for `stateless_cross_validate`, the compiled program is
        built from a new instance constructed from this instance's constructor
        arguments, read from the attributes of the same names (the scikit-learn
        convention). A subclass must follow that convention, as other instance state
        is not part of the compiled program. A subclass constructor's ``**kwargs``
        stand for its base class's constructor arguments. If the constructor
        arguments cannot be read or hashed, or a new instance constructed from them
        does not reproduce them, each call compiles anew.

        See Also
        --------
        stateless_cross_validate :
            The functional, ``jax.jit``/``jax.vmap``-able variant for equal-size folds.
        """
        cvm = self._make_cvm()
        cvm.fit(X, Y, sample_weight)  # validates non-negative weights; builds globals
        self._cvm = cvm
        self.A = A
        self.N, self.K = cvm.X.shape
        self.M = cvm.Y.shape[1]

        p = Partitioner(folds)
        N = cvm.N  # not self.N: another call on this instance may rebind it
        self._check_cv_folds(N, p.folds_dict.items(), sample_weight)

        # Group folds by validation-set size so vmap sees fixed shapes.
        buckets: dict[int, list] = defaultdict(list)
        for fold in p.folds_dict:
            vi = p.get_validation_indices(fold)
            buckets[int(vi.shape[0])].append((fold, vi))

        vmapped = self._vmapped_fold_fn(cvm, A, metric_function)
        all_idx = np.arange(N, dtype=int)

        n_batches = sum(
            int(np.ceil(len(items) / (batch_size or len(items))))
            for items in buckets.values()
        )
        pbar = tqdm(total=n_batches, disable=not show_progress)

        metrics: dict[Hashable, Any] = {}
        for items in buckets.values():
            val_stack = jnp.asarray(
                np.stack([vi for _, vi in items])
            )  # (n_folds, size)
            if self.algorithm == 1:
                # Per-fold training indices = complement of the validation indices.
                train_stack = jnp.asarray(
                    np.stack([np.setdiff1d(all_idx, vi) for _, vi in items])
                )
            bs = batch_size or len(items)
            for s in range(0, len(items), bs):
                chunk = items[s : s + bs]
                if self.algorithm == 1:
                    batched = vmapped(val_stack[s : s + bs], train_stack[s : s + bs])
                else:
                    batched = vmapped(val_stack[s : s + bs])
                # `metric_function` may return any pytree (array, tuple, dict, ...); vmap
                # stacks each leaf over the folds. Move to host, then slice out each
                # fold's metric, preserving the pytree structure.
                batched = jax.tree_util.tree_map(np.asarray, batched)
                for j, (fold, _vi) in enumerate(chunk):
                    metrics[fold] = jax.tree_util.tree_map(
                        lambda leaf, j=j: leaf[j], batched
                    )
                pbar.update(1)
        pbar.close()
        return metrics
