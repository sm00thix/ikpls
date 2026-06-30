"""
Contains the PLS class which implements fast cross-validation with partial
least-squares regression using Improved Kernel PLS by Dayal and MacGregor:
https://arxiv.org/abs/2401.13185
https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

This is the JAX counterpart of ``ikpls.fast_cross_validation.numpy_ikpls``. Instead of
parallelizing folds across CPU processes with joblib, it batches the folds with
``jax.vmap`` so the per-fold fits run together on a CPU/GPU/TPU. The per-fold training
:math:`\\mathbf{X}^{\\mathbf{T}}\\mathbf{W}\\mathbf{X}` and/or
:math:`\\mathbf{X}^{\\mathbf{T}}\\mathbf{W}\\mathbf{Y}` (centered/scaled/weighted using
training-set statistics) are obtained by rank-update from the dataset-wide matrices using
``cvmatrix`` with its JAX backend (``cvmatrix[jax]``).

Both Improved Kernel PLS Algorithm #1 and #2 are supported, mirroring
``ikpls.fast_cross_validation.numpy_ikpls``:

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

Note that, unlike a single ``fit``, this fast cross-validation does not emit the
"weight is close to zero" underflow warning: the warning relies on an ordered host
callback that is incompatible with ``jax.vmap``. Degenerate folds (training sets with
too few non-zero weights to compute centering/scaling statistics) are instead rejected
up-front with a ``ValueError`` before the vmapped computation.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from collections import defaultdict
from collections.abc import Callable, Hashable
from typing import Any, Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from cvmatrix.cvmatrix import CVMatrix
from cvmatrix.partitioner import Partitioner
from jax.typing import ArrayLike, DTypeLike
from tqdm import tqdm

from ikpls.jax_ikpls_alg_1 import PLS as _JaxAlg1
from ikpls.jax_ikpls_alg_2 import PLS as _JaxAlg2


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

    ddof : int, default=1
        The delta degrees of freedom to use when computing the sample standard
        deviation. A value of 0 corresponds to the biased estimate of the sample
        standard deviation, while a value of 1 corresponds to Bessel's correction.

    copy : bool, default=True
        Whether to copy `X`, `Y`, and `weights` when cross-validating.

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

    See Also
    --------
    ikpls.fast_cross_validation.numpy_ikpls.PLS :
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
        ddof: int = 1,
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
        # the algorithm's lax.scan. It runs under jax.vmap, where the ordered io_callback
        # underflow warning is unsupported, so we suppress it (`_warn_underflow=False`);
        # centering/scaling are handled by cvmatrix, so the fitter applies none.
        fitter_cls = _JaxAlg1 if algorithm == 1 else _JaxAlg2
        self._fitter = fitter_cls(
            center_X=False,
            center_Y=False,
            scale_X=False,
            scale_Y=False,
            dtype=dtype,
        )
        self._fitter._warn_underflow = False
        self.A = None
        self.N = None
        self.K = None
        self.M = None
        self._cvm = None
        self._sqrt_weights = None  # (N, 1) sqrt of weights, used by Algorithm #1

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
        init = (
            XTY,
            jnp.zeros((A, K), dtype=self.dtype),
            jnp.zeros((A, K), dtype=self.dtype),
            jnp.zeros((K, M), dtype=self.dtype),
        )

        def body(carry, i):
            return f._scan_body(operand, M, K, A, carry, i)

        # Both algorithms' scan bodies stack the regression coefficients ``B`` as the
        # last output element (Algorithm #1: W, P, Q, R, T, B; Algorithm #2: W, P, Q, R, B).
        _, outs = jax.lax.scan(body, init, jnp.arange(A))
        return outs[-1]

    def _predict_on_val(
        self,
        val_idx: ArrayLike,
        B: jax.Array,
        x_mean: Optional[jax.Array],
        x_std: Optional[jax.Array],
        y_mean: Optional[jax.Array],
        y_std: Optional[jax.Array],
    ) -> jax.Array:
        """Predict on the held-out rows for all components, undoing train preprocessing."""
        x_val = self._cvm.X[val_idx]  # (N_val, K) raw, unweighted
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
        self, val_idx: ArrayLike, train_idx: ArrayLike, A: int, K: int, M: int
    ) -> jax.Array:
        """Algorithm #1 per-fold fit + predict (gathers the training rows of ``X``)."""
        cvm = self._cvm
        xty_t, (x_mean, x_std, y_mean, y_std) = cvm.training_XTY(val_idx)
        x_train = cvm.X[train_idx]  # (N_train, K)
        if self.center_X:
            x_train = x_train - x_mean
        if self.scale_X:
            x_train = x_train / x_std
        if self._sqrt_weights is not None:
            x_train = x_train * self._sqrt_weights[train_idx]
        b = self._fit_B(x_train, xty_t, A, K, M)
        return self._predict_on_val(val_idx, b, x_mean, x_std, y_mean, y_std)

    def _fold_predict_alg2(
        self, val_idx: ArrayLike, A: int, K: int, M: int
    ) -> jax.Array:
        """Algorithm #2 per-fold fit + predict (uses the downdated ``XTX``/``XTY``)."""
        cvm = self._cvm
        (xtx_t, xty_t), (x_mean, x_std, y_mean, y_std) = cvm.training_XTX_XTY(val_idx)
        b = self._fit_B(xtx_t, xty_t, A, K, M)
        return self._predict_on_val(val_idx, b, x_mean, x_std, y_mean, y_std)

    def _validate_folds(self, p: Partitioner, weights: Optional[np.ndarray]) -> None:
        """
        Host-side pre-flight rejection of degenerate folds. The cvmatrix JAX backend
        omits the per-fold ``ValueError`` checks while traced by ``jax.vmap``, so they
        are performed here instead, mirroring the numpy backend's behavior.
        """
        needs_stats = self.center_X or self.center_Y or self.scale_X or self.scale_Y
        if not needs_stats:
            return
        needs_std = self.scale_X or self.scale_Y
        if weights is None:
            total_nonzero = self.N
        else:
            # Count non-zeros on the dtype-cast weights actually used on-device, so a
            # weight that underflows to 0 under ``self.dtype`` is rejected just as the
            # numpy backend (which validates the already-cast weights) would.
            w = np.asarray(weights, dtype=self.dtype).ravel()
            total_nonzero = int(np.count_nonzero(w))
        for fold, vi in p.folds_dict.items():
            if weights is None:
                nnz_train = self.N - int(vi.shape[0])
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

    def cross_validate(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        A: int,
        folds: Iterable[Hashable],
        metric_function: Callable[..., Any],
        weights: Optional[ArrayLike] = None,
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

        metric_function : Callable receiving `Y_val` (N_val, M), `Y_pred`\
        (A, N_val, M), and, if `weights` is not None, also `weights_val` (N_val,),\
        and returning Any.
            Computes a metric from the true `Y_val` and predicted `Y_pred`. `Y_pred`
            contains a prediction for all `A` components. **Must be JAX-traceable**
            (``jax.numpy``-based): it is evaluated on device inside ``jax.vmap`` over the
            folds, so only the metric (not the prediction tensor) is transferred back.

        weights : Array of shape (N,) or None, optional, default=None
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
            If `weights` are provided and not all weights are non-negative, or if a
            fold's training set has too few non-zero weights to compute the requested
            centering/scaling statistics.

        Notes
        -----
        Folds are grouped by validation-set size so that ``jax.vmap`` sees fixed shapes;
        leave-one-out and balanced k-fold use a single group, while size-imbalanced
        folds use one group per distinct size.
        """
        self._cvm = CVMatrix(
            center_X=self.center_X,
            center_Y=self.center_Y,
            scale_X=self.scale_X,
            scale_Y=self.scale_Y,
            ddof=self.ddof,
            dtype=self.dtype,
            copy=self.copy,
            backend="jax",
        )
        self._cvm.fit(X, Y, weights)  # validates non-negative weights; builds globals
        self.A = A
        self.N, self.K = self._cvm.X.shape
        self.M = self._cvm.Y.shape[1]
        # Algorithm #1 sqrt-weights the gathered training rows (mirrors a weighted fit).
        if self.algorithm == 1 and self._cvm.weights is not None:
            self._sqrt_weights = jnp.sqrt(self._cvm.weights)
        else:
            self._sqrt_weights = None

        p = Partitioner(folds)
        self._validate_folds(p, weights)
        all_idx = np.arange(self.N, dtype=int)
        has_weights = self._cvm.weights is not None

        # Group folds by validation-set size so vmap sees fixed shapes.
        buckets: dict[int, list] = defaultdict(list)
        for fold in p.folds_dict:
            vi = p.get_validation_indices(fold)
            buckets[int(vi.shape[0])].append((fold, vi))

        def evaluate(val_idx, y_pred):
            # On-device metric: gather the held-out targets (and weights) and apply the
            # JAX-traceable `metric_function` inside the vmapped/jitted computation, so no
            # per-fold host round-trip is needed and only the metric (not the full
            # prediction tensor) is transferred back.
            y_true = self._cvm.Y[val_idx]
            if has_weights:
                return metric_function(
                    y_true, y_pred, self._cvm.weights[val_idx].ravel()
                )
            return metric_function(y_true, y_pred)

        if self.algorithm == 1:

            def fold_fn(val_indices, train_indices):
                y_pred = self._fold_predict_alg1(
                    val_indices, train_indices, A, self.K, self.M
                )
                return evaluate(val_indices, y_pred)
        else:

            def fold_fn(val_indices):
                y_pred = self._fold_predict_alg2(val_indices, A, self.K, self.M)
                return evaluate(val_indices, y_pred)

        vmapped = jax.jit(jax.vmap(fold_fn))

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
