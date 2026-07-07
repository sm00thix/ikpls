"""
This file contains the ``PLS`` class: a thin, scikit-learn-conformant
wrapper around ikpls's NumPy ``PLS`` (``ikpls.numpy.PLS``), mirroring
``sklearn.cross_decomposition.PLSRegression``.

The wrapper performs ONLY input validation and plumbing; all PLS math is
delegated unchanged to the inner ``PLS``, so the native fast API
(``PLS.fit(X, Y, A)`` / ``PLS.predict`` returning all component counts /
``PLS.cross_validate``) is untouched. ``PLS`` exists to make the NumPy
IKPLS usable inside the scikit-learn ecosystem (``Pipeline``, ``cross_validate``,
``GridSearchCV``).

``PLS`` is conformant both as a *regressor* (``predict`` / ``score``) and
as a dimensionality-reducing *transformer* (``transform`` / ``inverse_transform``,
each supporting both ``X`` and ``Y``), and exposes the standard ``PLSRegression``
fitted attributes (``x_weights_``, ``y_weights_``, ``x_loadings_``,
``y_loadings_``, ``x_rotations_``, ``y_rotations_``, ``coef_``, ``intercept_``).
The vectorized
all-components prediction (shape ``(A, N, M)``) remains available via
``predict_all_components``.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from numbers import Integral
from typing import Any, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.metrics import r2_score
from sklearn.utils import Tags
from sklearn.utils._param_validation import Interval, Options
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from ikpls.numpy import PLS as _NumpyPLS

# Array-like data arguments to the public methods (``X`` / ``y`` /
# ``sample_weight``) are typed ``Any`` rather than ``npt.ArrayLike`` on purpose.
# Each public method is a validation boundary: it delegates to scikit-learn's
# ``validate_data`` / ``check_array``, which raise the sklearn-conformant errors
# for unsupported input (e.g. sparse matrices rejected because
# ``input_tags.sparse`` is False). The test suite runs with typeguard
# (``--typeguard-packages=ikpls``); a strict ``npt.ArrayLike`` annotation would
# make typeguard raise a ``TypeCheckError`` on such adversarial input *before* the
# body runs, pre-empting that validation and breaking ``check_estimator``. The
# honest array-like shapes are documented in each method's docstring instead.
_ArrayLikeInput = Any

# Return type of the score-producing methods: the X-scores alone, or the
# (X-scores, Y-scores) tuple when Y is supplied.
_Scores = Union[
    npt.NDArray[np.floating],
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
]


class PLS(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    RegressorMixin,
    MultiOutputMixin,
    BaseEstimator,
):
    """scikit-learn-conformant transformer + regressor wrapping ikpls's PLS.

    Mixin order mirrors sklearn's own ``_PLS`` base
    (``ClassNamePrefixFeaturesOutMixin, TransformerMixin, RegressorMixin,
    MultiOutputMixin, BaseEstimator``):

    * ``ClassNamePrefixFeaturesOutMixin`` provides ``get_feature_names_out``
      from the fitted ``_n_features_out`` attribute (output names like
      ``pls0 .. pls{A-1}``).
    * ``TransformerMixin`` provides ``fit_transform`` (fit then transform) and
      sets the ``transformer_tags``.
    * ``RegressorMixin`` provides the regressor tags; ``score`` is overridden
      below for an explicit signature.
    * ``MultiOutputMixin`` marks the estimator as natively multi-output
      (``target_tags.multi_output = True``), matching ``PLSRegression``: PLS
      handles a 2D ``Y`` with multiple targets directly.
    * ``BaseEstimator`` is last, as required.

    Parameters mirror ``ikpls.numpy.PLS`` plus ``n_components`` (which
    maps to the inner ``A``). The constructor stores parameters verbatim and
    performs no validation, as required by the sklearn estimator contract.

    Parameters
    ----------
    n_components : int, default=1
        Number of PLS components (latent variables). Maps to the inner ``A``.

    algorithm : int, default=1
        Improved Kernel PLS algorithm to use, either 1 or 2.

    center_X, center_Y, scale_X, scale_Y : bool, default=True
        Whether to (training-set) center / scale the predictors / responses.

    ddof : int, default=0
        Delta degrees of freedom for the (weighted) standard deviation used in
        scaling.

    copy : bool, default=True
        Whether to copy the input arrays.

    dtype : type, default=numpy.float64
        Floating point dtype used for the computations.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The PLS weights for ``X`` (inner ``PLS.W``).

    y_weights_ : ndarray of shape (n_targets, n_components)
        The PLS weights for ``Y`` (inner ``PLS.C``). In Improved Kernel PLS these
        equal ``y_loadings_``, matching ``sklearn.cross_decomposition.PLSRegression``
        (whose ``y_weights_`` equals its ``y_loadings_``).

    x_loadings_ : ndarray of shape (n_features, n_components)
        The PLS loadings for ``X`` (inner ``PLS.P``).

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The PLS loadings for ``Y`` (inner ``PLS.Q``).

    x_rotations_ : ndarray of shape (n_features, n_components)
        The rotations mapping ``X`` directly to its scores (inner ``PLS.R``).

    y_rotations_ : ndarray of shape (n_targets, n_components)
        The rotations mapping ``Y`` directly to its scores (inner ``PLS.R_Y``).
        Exposed as a lazy property: it requires a pseudo-inverse that is computed
        (and cached) on first access rather than during :meth:`fit`.

    coef_ : ndarray of shape (n_targets, n_features)
        Regression coefficients, matching ``sklearn.cross_decomposition.
        PLSRegression.coef_``. As in scikit-learn, ``predict`` effectively centers
        ``X``, so ``predict(X) == (X - X_mean) @ coef_.T + intercept_`` (not
        ``X @ coef_.T + intercept_``).

    intercept_ : ndarray of shape (n_targets,)
        Intercept term. Following ``sklearn.cross_decomposition.PLSRegression``,
        this is the (weighted) mean of ``Y`` used for centering (zeros when
        ``center_Y=False``).

    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Notes
    -----
    Improved Kernel PLS does not form a separate Y-weights matrix in its inner
    loop (it computes the Y *loadings* ``y_loadings_`` directly). Following
    ``sklearn.cross_decomposition.PLSRegression`` -- whose ``y_weights_`` equals
    its ``y_loadings_`` -- ``y_weights_`` is exposed as an alias of
    ``y_loadings_`` (inner ``PLS.C``, which equals inner ``PLS.Q``).

    For the fast native API -- fitting once and predicting with every number of
    components ``1..A`` in a single call (shape ``(A, N, M)``), and the built-in
    fast ``cross_validate`` -- use :class:`ikpls.numpy.PLS` directly. This
    wrapper exposes the all-components prediction via
    :meth:`predict_all_components`.

    ``X`` scores and ``Y`` scores can be obtained via the `transform` method.
    """

    # scikit-learn parameter constraints, validated at fit by _validate_params.
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "algorithm": [Options(Integral, {1, 2})],
        "center_X": ["boolean"],
        "center_Y": ["boolean"],
        "scale_X": ["boolean"],
        "scale_Y": ["boolean"],
        "ddof": [Interval(Integral, 0, None, closed="left")],
        "copy": ["boolean"],
        "dtype": "no_validation",
    }

    def __init__(
        self,
        # The constructor parameters are intentionally left unannotated: the
        # scikit-learn contract requires __init__ to store them verbatim and defer
        # validation to fit (via _parameter_constraints / _validate_params).
        # Annotating them would make typeguard (which the test suite runs via
        # --typeguard-packages=ikpls) reject non-conforming values at construction
        # instead of the clean fit-time InvalidParameterError. The expected
        # types/values are documented in the Parameters section above.
        n_components=1,
        algorithm=1,
        center_X=True,
        center_Y=True,
        scale_X=True,
        scale_Y=True,
        ddof=0,
        copy=True,
        dtype=np.float64,
    ) -> None:
        self.n_components = n_components
        self.algorithm = algorithm
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.ddof = ddof
        self.copy = copy
        self.dtype = dtype

    def fit(
        self,
        X: _ArrayLikeInput,
        y: Optional[_ArrayLikeInput],
        sample_weight: Optional[_ArrayLikeInput] = None,
    ) -> "PLS":
        """Fit the PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor variables.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Response variables.

        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights passed to the inner PLS.

        Returns
        -------
        self : object
            The fitted estimator.
        """
        # Validate all constructor parameters against _parameter_constraints
        # first, so a bad value raises a clean InvalidParameterError before any
        # data work (sklearn-style fail-fast).
        self._validate_params()

        # y is required to fit (supervised), even though the transformer side of the
        # API does not require it (target_tags.required is False). Raise a clear
        # message: with required=False, validate_data would otherwise route the
        # fit-only kwargs below into check_array and fail with an opaque TypeError.
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} is a supervised estimator: fit requires "
                "y, but y is None."
            )

        # Validate + record n_features_in_. reset=True because fit defines it.
        # multi_output=True lets (N, M) y through; 1D y stays 1D. ensure_min_samples=2
        # because PLS needs variance to fit (a single sample makes the scaled std
        # degenerate); validate_data then raises scikit-learn's standard
        # too-few-samples message, which the estimator checks accept.
        X, y = validate_data(
            self,
            X,
            y,
            reset=True,
            multi_output=True,
            y_numeric=True,
            dtype="numeric",
            ensure_min_samples=2,
        )

        # n_components cannot exceed the rank bound, matching scikit-learn's
        # PLSRegression, which requires 1 <= n_components <= min(n_samples,
        # n_features). Checked here (after validate_data) because the bound needs
        # the validated shapes.
        rank_upper_bound = min(X.shape[0], X.shape[1])
        if self.n_components > rank_upper_bound:
            raise ValueError(
                f"n_components={self.n_components} must be <= "
                f"min(n_samples, n_features)={rank_upper_bound}."
            )

        # A (N, 1) column-vector y is first-class here (multi-output estimator with
        # M = 1, like PLSRegression) -- no DataConversionWarning; predictions simply
        # keep the 2D shape. Whether the user gave 1D y is recorded AFTER inner.fit
        # succeeds (below) so a failed refit cannot leave _y_1d inconsistent with
        # the surviving fitted state.
        y_1d = y.ndim == 1

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=self.dtype)

        inner = _NumpyPLS(
            algorithm=self.algorithm,
            center_X=self.center_X,
            center_Y=self.center_Y,
            scale_X=self.scale_X,
            scale_Y=self.scale_Y,
            ddof=self.ddof,
            copy=self.copy,
            dtype=self.dtype,
        )
        # The inner PLS reshapes a 1D y to a 2D column (and casts to self.dtype).
        inner.fit(X, y, A=self.n_components, sample_weight=sample_weight)

        self.inner_ = inner
        # Remember whether the user gave 1D y so predict can match its shape.
        self._y_1d = y_1d
        self.n_components_ = self.n_components
        # Number of output features of transform() == number of components.
        # Consumed by ClassNamePrefixFeaturesOutMixin.get_feature_names_out.
        self._n_features_out = self.n_components_

        # --- Fitted attributes mirroring sklearn.cross_decomposition.PLSRegression.
        # These are the inner PLS matrices; their shapes already follow sklearn's
        # (n_features/n_targets, n_components) convention, so no recomputation is
        # needed. (y_rotations_ is exposed as a lazy property; y_weights_ == Q.)
        # ``np.array`` (not ``np.asarray``) forces a copy so each exposed attribute
        # is independent of the inner model and of each other -- matching sklearn's
        # mutation isolation (e.g. y_weights_ and y_loadings_ are equal but distinct
        # arrays). The copies are tiny ((K, A)/(M, A)) and one-off per fit.
        A = self.n_components_
        K = X.shape[1]
        M = 1 if self._y_1d else y.shape[1]
        self.x_weights_ = np.array(inner.W, dtype=self.dtype)  # (K, A)
        self.y_weights_ = np.array(inner.C, dtype=self.dtype)  # (M, A) == Q
        self.x_loadings_ = np.array(inner.P, dtype=self.dtype)  # (K, A)
        self.y_loadings_ = np.array(inner.Q, dtype=self.dtype)  # (M, A)
        self.x_rotations_ = np.array(inner.R, dtype=self.dtype)  # (K, A)
        # y_rotations_ is intentionally NOT materialized here: it requires a
        # pseudo-inverse (inner.R_Y[A]) that most callers never touch, so it is
        # exposed as a lazy property below and computed on first access instead.

        # Regression coefficients and intercept follow scikit-learn's PLSRegression
        # convention exactly: coef_ (M, K) is the original-space coefficient and
        # intercept_ is the Y-mean. As in sklearn, predict effectively centers X, so
        # predict(X) == (X - X_mean) @ coef_.T + intercept_ (NOT X @ coef_.T +
        # intercept_). Bp is the preprocessed-space coefficient for A components; the
        # inner predict computes ((X - X_mean)/X_std) @ Bp * Y_std + Y_mean.
        Bp = np.asarray(inner.B[A - 1], dtype=self.dtype)  # (K, M)
        inv_x_std = (
            1.0 / np.asarray(inner.X_std, dtype=self.dtype).ravel()
            if (self.scale_X and inner.X_std is not None)
            else np.ones(K, dtype=self.dtype)
        )
        y_std = (
            np.asarray(inner.Y_std, dtype=self.dtype).ravel()
            if (self.scale_Y and inner.Y_std is not None)
            else np.ones(M, dtype=self.dtype)
        )
        # ``np.array`` (copy) so intercept_ does not alias inner_.Y_mean, which
        # predict/inverse_transform read for un-centering (mutation isolation).
        y_mean = (
            np.array(inner.Y_mean, dtype=self.dtype).ravel()
            if (self.center_Y and inner.Y_mean is not None)
            else np.zeros(M, dtype=self.dtype)
        )
        coef_original_space = (inv_x_std[:, None] * Bp) * y_std[None, :]  # (K, M)
        self.coef_ = coef_original_space.T  # (M, K), sklearn convention
        self.intercept_ = y_mean  # (M,) == scikit-learn PLSRegression's intercept_
        return self

    def fit_transform(
        self, X: _ArrayLikeInput, y: Optional[_ArrayLikeInput] = None, **fit_params
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Fit the model, then return the training scores.

        The default ``TransformerMixin.fit_transform`` calls ``transform(X)`` and
        thus returns only the X-scores. This override instead fits and returns
        ``transform(X, y)`` -- i.e. the ``(x_scores, y_scores)`` tuple, matching
        ``ikpls.numpy.PLS.fit_transform`` and
        ``sklearn.cross_decomposition.PLSRegression.fit_transform``. It goes
        through :meth:`fit` (so a subclass overriding ``fit`` still runs) and
        then :meth:`transform`, so ``fit_transform`` stays consistent with
        ``fit(...).transform(...)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor variables.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Response variables. Required (PLS is supervised).

        **fit_params : dict
            Extra keyword arguments forwarded to :meth:`fit` (e.g.
            ``sample_weight``).

        Returns
        -------
        (x_scores, y_scores) : tuple of ndarray
            The training X-scores and Y-scores, each of shape
            ``(n_samples, n_components)``.
        """
        return self.fit(X, y, **fit_params).transform(X, y)

    @property
    def y_rotations_(self) -> npt.NDArray[np.floating]:
        """The rotations mapping ``Y`` directly to its scores.

        Computed lazily on first access, because it requires a pseudo-inverse
        (the inner ``PLS.R_Y[n_components_]``) that ``fit`` would otherwise pay
        for on every call even though most callers never read it. The inner
        model caches the pseudo-inverse, so repeated access only re-copies a
        small array.

        Returns
        -------
        ndarray of shape (n_targets, n_components)

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet. As ``NotFittedError``
            subclasses ``AttributeError``, ``hasattr(est, "y_rotations_")`` is
            ``False`` before fitting, matching the behaviour of a regular fitted
            attribute.
        """
        check_is_fitted(self)
        # ``np.array`` (copy), not ``np.asarray``: R_Y[n] returns the inner model's
        # cached array, which ``transform`` also reads. Exposing a copy keeps the
        # attribute independent of the inner model, matching the eager attrs above.
        return np.array(self.inner_.R_Y[self.n_components_], dtype=self.dtype)

    def predict(self, X: _ArrayLikeInput) -> npt.NDArray[np.floating]:
        """Predict using ``n_components`` components.

        Returns an array of shape ``(n_samples,)`` if ``y`` was 1D at fit time,
        else ``(n_samples, n_targets)``.
        """
        check_is_fitted(self)
        # reset=False: verify n_features_in_ consistency, reject bad input.
        X = validate_data(self, X, reset=False, dtype="numeric")
        preds = self.inner_.predict(X, n_components=self.n_components_)
        # preds is (N, M). Match the y shape the user trained with.
        if self._y_1d:
            return preds.ravel()
        return preds

    def predict_all_components(self, X: _ArrayLikeInput) -> npt.NDArray[np.floating]:
        """Fast all-components prediction: returns shape ``(A, N, M)``.

        Preserves ikpls's vectorized feature of predicting with every number
        of components ``1..A`` in a single call.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="numeric")
        return self.inner_.predict(X, n_components=None)

    def transform(
        self, X: _ArrayLikeInput, y: Optional[_ArrayLikeInput] = None
    ) -> _Scores:
        """Project ``X`` (and optionally ``Y``) onto the latent component space.

        Returns the X-scores ``T`` of shape ``(n_samples, n_components)``. If
        ``y`` is provided, also returns the Y-scores and yields
        ``(x_scores, y_scores)``, matching scikit-learn's
        ``PLSRegression.transform(X, y)``. Delegates to the inner
        ``PLS.transform``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor variables to project.

        y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
            Response variables to project. If given, Y-scores are also returned.

        Returns
        -------
        x_scores : ndarray of shape (n_samples, n_components)
            Returned when ``y`` is None.
        (x_scores, y_scores) : tuple of ndarray
            Returned when ``y`` is given; ``y_scores`` has shape
            ``(n_samples, n_components)``.
        """
        check_is_fitted(self)
        # reset=False: enforce the n_features_in_ established at fit.
        X = validate_data(self, X, reset=False, dtype="numeric")
        if y is None:
            return self.inner_.transform(X=X, n_components=self.n_components_)
        # The inner transform coerces a 1D y to a 2D column and casts to self.dtype
        # (via _convert_input_to_array), so only validation is needed here.
        y = check_array(y, dtype="numeric", ensure_2d=False)
        return self.inner_.transform(X=X, Y=y, n_components=self.n_components_)

    def inverse_transform(
        self, X: _ArrayLikeInput, y: Optional[_ArrayLikeInput] = None
    ) -> _Scores:
        """Map scores back to the original space.

        ``X`` is the X-scores (as returned by :meth:`transform`); the predictors
        are reconstructed via the X-loadings. If ``y`` (the Y-scores) is given,
        the responses are also reconstructed and ``(X_hat, Y_hat)`` is returned,
        matching scikit-learn's ``PLSRegression.inverse_transform(X, y)``. The
        round trip is exact only when ``n_components`` equals ``n_features`` (for
        ``X``) / ``n_targets`` (for ``Y``); the sklearn transformer test-suite
        does not require an exact round trip for dimensionality-reducing
        transformers, so exposing this is safe.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            X-scores produced by :meth:`transform`.

        y : array-like of shape (n_samples, n_components), optional
            Y-scores produced by :meth:`transform`.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features_in_)
            Reconstructed predictors. Returned alone when ``y`` is None.
        (X_original, Y_original) : tuple of ndarray
            Returned when ``y`` is given; ``Y_original`` has shape
            ``(n_samples, n_targets)``.
        """
        check_is_fitted(self)
        # Scores live in component space, not feature space, so we do NOT validate
        # against n_features_in_. But like PLSRegression's inverse_transform, the
        # scores are check_array-validated: a 1D score vector is REJECTED (sklearn's
        # 'Expected 2D array' message) rather than silently reinterpreted as
        # n_components one-component samples, and non-finite scores are rejected.
        # check_array also casts to self.dtype, so the inner delegate's own
        # 2D/dtype coercion is a no-op afterwards.
        X = check_array(X, dtype=self.dtype, input_name="X")
        if y is None:
            return self.inner_.inverse_transform(T=X)
        y = check_array(y, dtype=self.dtype, input_name="y")
        return self.inner_.inverse_transform(T=X, U=y)

    def score(
        self,
        X: _ArrayLikeInput,
        y: _ArrayLikeInput,
        sample_weight: Optional[_ArrayLikeInput] = None,
    ) -> float:
        """Coefficient of determination R^2 of the prediction.

        Predicts with ``n_components`` components and scores against ``y`` with
        scikit-learn's ``r2_score`` (``multioutput='uniform_average'``), matching
        the regressor ``score`` contract, with optional per-sample
        ``sample_weight``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test predictors.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True responses for ``X``.

        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights.

        Returns
        -------
        score : float
            The R^2 score.
        """
        check_is_fitted(self)
        return r2_score(y, self.predict(X), sample_weight=sample_weight)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        # PLS is a poor generic regressor on arbitrary data; sklearn's own
        # PLSRegression sets the same flag so the regressor checks do not
        # demand a good R^2 on the synthetic check datasets.
        tags.regressor_tags.poor_score = True
        # y is required at fit (supervised), but is NOT required by transform.
        tags.target_tags.required = False
        return tags
