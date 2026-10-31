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
  each fold to its metric.
* ``stateless_cross_validate``: a purely functional variant that takes the validation
  indices of equally sized folds as a 2-D array and returns the metrics stacked over the
  folds. It is ``jax.jit``- and ``jax.vmap``-able, so MANY independent datasets (e.g. one
  local model per sample, per pixel, or per experimental condition) can be
  cross-validated at once by ``jax.vmap``-ing it over a stack of datasets -- the fast
  cross-validation analogue of ``ikpls.jax.PLS.stateless_fit``. Like ``stateless_fit``
  it is ``jax.jit``-compiled internally (unweighted calls with a hashable
  `metric_function`), so eager calls are fast too.

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
only when its `validation_indices` are concrete; see its docstring).

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import functools
import inspect
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

from ikpls._impl.jax_alg_1 import PLS as _JaxAlg1
from ikpls._impl.jax_alg_2 import PLS as _JaxAlg2

_TRACED_WEIGHTS_MSG = (
    "`sample_weight` must be None under jax.jit tracing (as a traced argument or a "
    "concrete array closed over under jax.jit): the cvmatrix backend validates the "
    "weights host-side. Call eagerly, use a plain jax.vmap with one concrete shared "
    "weight vector, or use `cross_validate`."
)


@functools.lru_cache(maxsize=64)
def _compiled_stateless_body(
    cls: type, settings: tuple, metric_function: Callable[..., Any]
) -> Callable[..., Any]:
    """
    The unweighted ``stateless_cross_validate`` body for one (class, constructor
    settings, `metric_function`), ``jax.jit``-compiled per input shapes/dtypes and `A`.
    It is keyed by the settings rather than by the instance, so instances stay
    picklable, copies use their own settings, and changing an instance's settings
    after a call takes effect. Only the 64 most recently used combinations are kept,
    so the metric functions (and what they capture) are not retained indefinitely.
    """
    pls = cls(**dict(settings))

    def body(X, Y, validation_indices, A):
        return pls._stateless_body(X, Y, validation_indices, A, metric_function)

    return jax.jit(body, static_argnames=("A",))


def _is_hashable(obj: Any) -> bool:
    try:
        hash(obj)
    except TypeError:
        return False
    return True


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
        fitter_cls = _JaxAlg1 if self.algorithm == 1 else _JaxAlg2
        return fitter_cls(
            center_X=False,
            center_Y=False,
            scale_X=False,
            scale_Y=False,
            dtype=self.dtype,
        )

    def _fit_B(
        self, operand: jax.Array, XTY: jax.Array, A: int, K: int, M: int
    ) -> jax.Array:
        """
        Compute the (A, K, M) regression-coefficients tensor from a fold's training
        matrices via the algorithm's ``lax.scan`` fit. ``operand`` is the training ``X``
        (N_train, K) for Algorithm #1 or the training ``XTX`` (K, K) for Algorithm #2;
        ``XTY`` is the training ``XTY`` (K, M). Reverse-mode-differentiable and
        ``jax.vmap``-safe.
        """
        f = self._fitter
        if type(f) is not (_JaxAlg1 if self.algorithm == 1 else _JaxAlg2) or (
            f.dtype != self.dtype
        ):
            f = self._make_fitter()  # `algorithm` or `dtype` changed after construction
        init = (
            XTY,
            jnp.zeros((A, K), dtype=self.dtype),
            jnp.zeros((A, K), dtype=self.dtype),
            jnp.zeros((K, M), dtype=self.dtype),
        )

        def body(carry, i):
            return f._scan_body(operand, M, K, A, carry, i)

        # Both algorithms' scan bodies stack the regression coefficients ``B`` as the
        # last output element (Algorithm #1: W, P, Q, R, T, norm, B; Algorithm #2:
        # W, P, Q, R, norm, B), so outs[-1] is B.
        _, outs = jax.lax.scan(body, init, jnp.arange(A))
        return outs[-1]

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
        x_val = cvm.X[val_idx]  # (N_val, K) raw, unweighted
        if self.center_X:
            x_val = x_val - x_mean
        if self.scale_X:
            x_val = x_val / x_std
        y_pred = x_val @ B  # (N_val, K) @ (A, K, M) -> (A, N_val, M)
        if self.scale_Y:
            y_pred = y_pred * y_std
        if self.center_Y:
            y_pred = y_pred + y_mean
        return y_pred

    def _fold_predict_alg1(
        self,
        cvm: CVMatrix,
        sqrt_weights: Optional[jax.Array],
        val_idx: jax.Array,
        train_idx: Optional[jax.Array],
        A: int,
        K: int,
        M: int,
    ) -> jax.Array:
        """Algorithm #1 per-fold fit + predict (gathers the training rows of ``X``).
        ``train_idx`` is derived on device from ``val_idx`` when it is None."""
        xty_t, (x_mean, x_std, y_mean, y_std) = cvm.training_XTY(val_idx)
        if train_idx is None:
            train_idx = self._training_indices(val_idx, cvm.N)
        x_train = cvm.X[train_idx]  # (N_train, K)
        if self.center_X:
            x_train = x_train - x_mean
        if self.scale_X:
            x_train = x_train / x_std
        if sqrt_weights is not None:
            x_train = x_train * sqrt_weights[train_idx]
        b = self._fit_B(x_train, xty_t, A, K, M)
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
        faster than deriving them on device (it compiles anew on every call).
        """
        K, M = cvm.K, cvm.M
        has_weights = cvm.weights is not None
        # Algorithm #1 sqrt-weights the gathered training rows (mirrors a weighted fit).
        sqrt_weights = (
            jnp.sqrt(cvm.weights) if (self.algorithm == 1 and has_weights) else None
        )

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
                y_pred = self._fold_predict_alg1(
                    cvm, sqrt_weights, val_idx, train_idx, A, K, M
                )
                return evaluate(val_idx, y_pred)

        else:

            def fold_fn(val_idx):
                y_pred = self._fold_predict_alg2(cvm, val_idx, A, K, M)
                return evaluate(val_idx, y_pred)

        return fold_fn

    def _validate_folds(
        self,
        N: int,
        fold_items: Iterable[Tuple[Hashable, np.ndarray]],
        sample_weight: Optional[np.ndarray],
    ) -> None:
        """
        Host-side pre-flight rejection of degenerate folds. The cvmatrix JAX backend
        omits the per-fold ``ValueError`` checks while traced by ``jax.vmap``, so they
        are performed here instead, mirroring the numpy backend's behavior.
        ``fold_items`` yields ``(fold, validation_indices)`` pairs.
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

    def _check_stateless_weights(self, sample_weight: ArrayLike, N: int) -> None:
        w = np.asarray(sample_weight, dtype=self.dtype).ravel()
        if w.shape[0] != N:
            raise ValueError(
                f"`sample_weight` must have shape (N,) = ({N},); got {w.shape}."
            )
        if np.any(w < 0):
            raise ValueError("Weights must be non-negative.")

    def _check_concrete_folds(
        self, vi: np.ndarray, N: int, sample_weight: Optional[ArrayLike]
    ) -> None:
        """
        Host-side rejection of out-of-range, duplicated and degenerate folds. Traced
        indices cannot be checked; the caller is responsible for them, as for
        cvmatrix's own deferred checks.
        """
        if vi.min() < 0 or vi.max() >= N:
            raise ValueError(
                f"`validation_indices` contains indices outside range(N) with N={N}."
            )
        vi_sorted = np.sort(vi, axis=1)
        if np.any(vi_sorted[:, 1:] == vi_sorted[:, :-1]):
            raise ValueError(
                "Each row of `validation_indices` must hold distinct indices."
            )
        self._validate_folds(N, enumerate(vi), sample_weight)

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
        result is a pytree of arrays rather than a dict. Unweighted calls with a hashable
        `metric_function` are ``jax.jit``-compiled internally (one compilation per
        settings, input shapes, `A` and `metric_function`; the 64 most recently used
        settings/`metric_function` combinations are kept compiled), so eager calls run
        at compiled speed; other calls run un-jitted.

        Its main use is cross-validating MANY independent datasets at once: ``jax.vmap``
        it over a leading dataset axis of `X` (B, N, K), `Y` (B, N, M) and
        `validation_indices` (B, n_folds, N_val) and wrap the result in ``jax.jit`` so a
        single compiled program fits all ``B * n_folds`` models together on the
        CPU/GPU/TPU. A plain Python loop over `cross_validate` would instead pay a host
        round-trip and a recompilation per dataset.

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
            silently corrupts that fold, so validate traced indices beforehand.

        metric_function : Callable
            Computes a metric from the true ``Y_val`` (N_val, M) and the predicted
            ``Y_pred`` (A, N_val, M) -- and, if ``sample_weight`` is not None, also
            ``weights_val`` (N_val,) -- returning any pytree of arrays. ``Y_pred``
            contains a prediction for all `A` components. **Must be JAX-traceable**
            (``jax.numpy``-based): it is evaluated on device inside ``jax.vmap`` over
            the folds. It is part of the internal compilation cache key, so define it
            once rather than per call (a new function object, e.g. a lambda created
            per call, recompiles), and values it closes over are captured when it is
            first compiled. The cache holds the 64 most recently used combinations of
            settings and `metric_function`, and with them the functions and what
            they capture. An unhashable callable is not compiled internally.

        sample_weight : Array of shape (N,) or None, optional, default=None
            Weights for each observation. If None, all observations are weighted
            equally. Must be non-negative. Supported when called eagerly and under a
            plain ``jax.vmap`` with one concrete weight vector shared by all datasets
            (closed over, not vmapped). Not supported under ``jax.jit`` or as a traced
            (vmapped) argument: the cvmatrix backend validates the weights host-side,
            which requires concrete values; such calls raise a ``ValueError``.

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
            `sample_weight` is traced (a tracer, or a concrete array staged under
            ``jax.jit``), has the wrong length or contains negative values; if the
            training-set size ``N - N_val`` is too small for the requested scaling with
            this `ddof` (checked even under tracing, since it depends on shapes only); if
            a weighted fold's training set has too few non-zero weights (checked when
            `validation_indices` is concrete); or if ``dtype`` is 64-bit while
            ``jax_enable_x64`` is off.

        Notes
        -----
        Per-fold results agree with `cross_validate` up to floating-point round-off (the
        same per-fold function is used; XLA may fuse operations differently across jit
        boundaries). Components beyond the numerical rank of `X` are unstable in any
        implementation and may differ more between the two entry points.

        If all datasets share one split, close over a single concrete (n_folds, N_val)
        `validation_indices` and ``jax.vmap`` over `X` and `Y` only; the fold validity
        checks then run host-side, also inside the caller's ``jax.jit``.

        The internally compiled program is built from a new instance constructed from
        this instance's constructor arguments, read from the attributes of the same
        names (the scikit-learn convention). A subclass must follow that convention,
        as other instance state is not part of the compiled program; if its
        constructor arguments cannot be read or hashed, the call runs un-jitted.

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
        """
        if isinstance(sample_weight, jax.core.Tracer):
            raise ValueError(_TRACED_WEIGHTS_MSG)
        self._check_x64()
        A = operator.index(A)  # a static Python int, also when given as a 0-d array
        if not isinstance(X, (jax.Array, np.ndarray)):
            X = np.asarray(X)
        if not isinstance(Y, (jax.Array, np.ndarray)):
            Y = np.asarray(Y)
        # Decide concreteness on the argument as given: inside the caller's jax.jit,
        # jnp.asarray would stage even a closed-over NumPy array into a tracer.
        concrete = not isinstance(validation_indices, jax.core.Tracer)
        vi = _validation_index_array(validation_indices, concrete)
        N = np.shape(X)[0]
        self._check_stateless_shapes(N, np.shape(Y)[0], vi.shape)
        if sample_weight is not None:
            self._check_stateless_weights(sample_weight, N)
        if concrete:
            self._check_concrete_folds(vi, N, sample_weight)
        validation_indices = jnp.asarray(validation_indices)
        settings = self._settings()
        if (
            sample_weight is None
            and settings is not None
            and _is_hashable(metric_function)
        ):
            compiled = _compiled_stateless_body(type(self), settings, metric_function)
            return compiled(X, Y, validation_indices, A=A)
        # Weighted calls run un-jitted: cvmatrix validates the weights host-side, which
        # needs concrete values (so they cannot run under a caller's jax.jit either).
        # So do calls whose settings or metric cannot key the compiled body.
        return self._stateless_body(
            X, Y, validation_indices, A, metric_function, sample_weight
        )

    def _settings(self) -> Optional[tuple]:
        """
        The constructor arguments, read from the attributes of the same names (the
        scikit-learn convention, also for subclasses), as a hashable key for the
        compiled body; None if they cannot key it.
        """
        try:
            params = inspect.signature(type(self).__init__).parameters.values()
            settings = tuple(
                (p.name, getattr(self, p.name))
                for p in params
                if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            )
            hash(settings)
        except (AttributeError, TypeError, ValueError):
            return None
        return settings

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
        try:
            cvm.fit(X, Y, sample_weight)  # builds the dataset-wide matrices/statistics
        except jax.errors.TracerBoolConversionError as err:
            # A concrete `sample_weight` closed over under a caller's jax.jit is staged
            # into a tracer, which cvmatrix's host-side non-negativity check cannot
            # inspect.
            raise ValueError(_TRACED_WEIGHTS_MSG) from err
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
            back.

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

        To cross-validate many independent datasets at once, ``jax.vmap``
        `stateless_cross_validate` over a stack of datasets instead of looping over
        this method (each call here re-traces and re-compiles its fold function).

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
        self._validate_folds(N, p.folds_dict.items(), sample_weight)

        # Group folds by validation-set size so vmap sees fixed shapes.
        buckets: dict[int, list] = defaultdict(list)
        for fold in p.folds_dict:
            vi = p.get_validation_indices(fold)
            buckets[int(vi.shape[0])].append((fold, vi))

        vmapped = jax.jit(jax.vmap(self._make_fold_fn(cvm, A, metric_function)))
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
