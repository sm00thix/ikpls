"""
scikit-learn API conformance tests for ``ikpls.sklearn.PLS``.

The main test verifies that ``SklearnPLS`` passes scikit-learn's estimator check
suite. The only expected failures are the two transformer-consistency checks:
``fit_transform`` returns the ``(X-scores, Y-scores)`` tuple (like sklearn's own
``PLSRegression``), which scikit-learn special-cases only for its hard-coded
``CROSS_DECOMPOSITION`` class names -- and ours is named ``PLS`` (see
``_EXPECTED_FAILED_CHECKS``). With the repo-wide default ``ddof=0`` the weighted
standard deviation is row-repetition equivalent, so the frequency-weight
``sample_weight``-equivalence check passes.

The remaining tests confirm the performance-preserving guarantees: the
all-components ``(A, N, M)`` prediction stays bit-identical to the inner ``PLS``,
and ``n_components`` is validated with a clean ``ValueError``.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from itertools import product

import numpy as np
import pytest
from sklearn.metrics import r2_score
from sklearn.utils._testing import assert_allclose
from sklearn.utils.estimator_checks import check_estimator

from ikpls.numpy import PLS as NpPLS
from ikpls.sklearn import PLS as SklearnPLS


# The transformer-consistency checks compare fit_transform(X, y) against
# transform(X). PLS.fit_transform returns the (X-scores, Y-scores) tuple --
# matching sklearn.cross_decomposition.PLSRegression.fit_transform and
# ikpls.numpy.PLS.fit_transform -- but scikit-learn only compares tuple-to-tuple
# for a hard-coded set of class NAMES (CROSS_DECOMPOSITION = "PLSRegression",
# "PLSCanonical", "CCA", "PLSSVD"). For any other name it compares the tuple to
# the X-only transform(X) and fails. Our class is named "PLS", so these are
# expected failures, NOT a real conformance defect: sklearn's own PLSRegression
# has identical tuple behaviour and only passes by virtue of its name.
_EXPECTED_FAILED_CHECKS = {
    "check_transformer_general": (
        "PLS.fit_transform returns the (X-scores, Y-scores) tuple like sklearn's "
        "PLSRegression; sklearn only special-cases tuple-returning fit_transform "
        "for its hard-coded CROSS_DECOMPOSITION class names, and 'PLS' is not one."
    ),
    "check_transformer_data_not_an_array": (
        "Same cause as check_transformer_general: the tuple-returning "
        "fit_transform is only special-cased by sklearn for its hard-coded "
        "CROSS_DECOMPOSITION class names."
    ),
}


def test_sklearn_pls_passes_check_estimator():
    """SklearnPLS conforms to the scikit-learn estimator API.

    ``check_estimator`` runs the full suite; the only expected failures are the
    two transformer-consistency checks (see ``_EXPECTED_FAILED_CHECKS``), which
    fail solely because ``fit_transform`` returns the ``(X-scores, Y-scores)``
    tuple and scikit-learn special-cases tuple ``fit_transform`` only for its
    hard-coded ``CROSS_DECOMPOSITION`` class names. With the default ``ddof=0``
    the weighted ``sample_weight``-equivalence check passes.
    """
    check_estimator(SklearnPLS(), expected_failed_checks=_EXPECTED_FAILED_CHECKS)


@pytest.mark.parametrize("algorithm", [1, 2])
@pytest.mark.parametrize("n_components", [1, 3, 5])
def test_fit_transform_returns_xy_scores_tuple(algorithm, n_components):
    """fit_transform returns the (X-scores, Y-scores) tuple -- matching
    ikpls.numpy.PLS.fit_transform and sklearn's PLSRegression -- rather than
    TransformerMixin's X-only default, and is consistent with
    fit(...).transform(...)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8))
    Y = rng.standard_normal((40, 3))

    out = SklearnPLS(n_components=n_components, algorithm=algorithm).fit_transform(X, Y)
    assert isinstance(out, tuple) and len(out) == 2
    T, U = out
    assert T.shape == (40, n_components)
    assert U.shape == (40, n_components)

    # Consistent with fit followed by transform (the two agree to ~1e-15).
    T2, U2 = (
        SklearnPLS(n_components=n_components, algorithm=algorithm)
        .fit(X, Y)
        .transform(X, Y)
    )
    assert_allclose(T, T2)
    assert_allclose(U, U2)


def test_multi_output_tag_matches_plsregression():
    """PLS supports multiple targets, so it is tagged multi-output like PLSRegression."""
    from sklearn.cross_decomposition import PLSRegression

    tags = SklearnPLS().__sklearn_tags__()
    assert tags.target_tags.multi_output is True
    assert (
        tags.target_tags.multi_output
        == PLSRegression().__sklearn_tags__().target_tags.multi_output
    )


@pytest.mark.parametrize("algorithm", [1, 2])
@pytest.mark.parametrize("n_components", [1, 5])
def test_predict_all_components_bit_identical_to_inner(algorithm, n_components):
    """The (A, N, M) fast path survives the wrapper bit-for-bit."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 12))
    Y = rng.standard_normal((40, 3))

    wrapper = SklearnPLS(
        n_components=n_components,
        algorithm=algorithm,
        center_X=True,
        center_Y=True,
        scale_X=True,
        scale_Y=True,
        ddof=1,
    ).fit(X, Y)
    inner = NpPLS(
        algorithm=algorithm,
        center_X=True,
        center_Y=True,
        scale_X=True,
        scale_Y=True,
        ddof=1,
    ).fit(X, Y, A=n_components)

    np.testing.assert_array_equal(
        wrapper.predict_all_components(X), inner.predict(X, n_components=None)
    )
    np.testing.assert_array_equal(
        wrapper.predict(X), inner.predict(X, n_components=n_components)
    )


# ``True`` is intentionally omitted: ``Interval(Integral, ...)`` accepts bool
# (a ``True`` is an ``Integral`` equal to 1), matching scikit-learn's own
# ``PLSRegression``, which likewise accepts ``n_components=True``. The other
# values are rejected by ``_validate_params`` with an ``InvalidParameterError``
# (a ``ValueError``).
@pytest.mark.parametrize("bad", [0, -1, 1.5, "2", None])
def test_n_components_validation(bad):
    """Invalid n_components raises a clean ValueError (not a deep IndexError)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    y = rng.standard_normal(10)
    with pytest.raises(ValueError):
        SklearnPLS(n_components=bad).fit(X, y)


def test_fitted_attribute_shapes():
    """The exposed PLSRegression-style fitted attributes have sklearn shapes."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 10))
    Y = rng.standard_normal((40, 3))
    A = 4
    est = SklearnPLS(n_components=A).fit(X, Y)
    assert est.x_weights_.shape == (10, A)
    assert est.y_weights_.shape == (3, A)
    assert est.x_loadings_.shape == (10, A)
    assert est.y_loadings_.shape == (3, A)
    assert est.x_rotations_.shape == (10, A)
    assert est.y_rotations_.shape == (3, A)
    assert est.coef_.shape == (3, 10)
    assert est.intercept_.shape == (3,)
    assert est.n_features_in_ == 10


@pytest.mark.parametrize("m", [1, 3])
@pytest.mark.parametrize("scale", [True, False])
def test_y_weights_equals_y_loadings(m, scale):
    """The exposed ``y_weights_`` is exactly ``y_loadings_`` (its value, not just
    its shape), matching scikit-learn's PLSRegression convention.

    This is a value check that catches a wrong inner attribute being assigned
    (e.g. ``x_weights_``, which has a different shape) or a dtype/reshape mishap,
    and holds for any data and any center/scale configuration.
    """
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 6))
    Y = rng.standard_normal((50, m))
    est = SklearnPLS(n_components=3, scale_X=scale, scale_Y=scale).fit(X, Y)
    assert_allclose(est.y_weights_, est.y_loadings_, atol=0, rtol=0)


def test_y_weights_match_sklearn_on_linnerud():
    """The wrapper's ``y_weights_`` reproduces scikit-learn PLSRegression's
    ``y_weights_`` on the Linnerud sanity-check dataset.

    Random, structureless data is avoided on purpose: Improved Kernel PLS and
    PLSRegression pick different (equally valid) directions for components past
    the true rank, so their weights only agree on data with real latent
    structure -- exactly the Linnerud case used by the sanity-check tests.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.datasets import load_linnerud

    d = load_linnerud()
    X = d.data
    Y = d.target
    A = X.shape[1]
    est = SklearnPLS(n_components=A).fit(X, Y)
    skl = PLSRegression(n_components=A).fit(X, Y)
    assert_allclose(
        np.abs(est.y_weights_), np.abs(skl.y_weights_), atol=1e-6, rtol=1e-5
    )


def test_lazy_attributes_absent_before_fit():
    """``y_weights_`` and the lazy ``y_rotations_`` property honor the fitted-attr
    contract: absent (``hasattr`` is False) before fit, present after.

    ``NotFittedError`` subclasses ``AttributeError``, so accessing the lazy
    ``y_rotations_`` before fit makes ``hasattr`` return False, matching an
    ordinary fitted attribute such as the eager ``y_weights_``.
    """
    est = SklearnPLS(n_components=2)
    assert not hasattr(est, "y_weights_")
    assert not hasattr(est, "y_rotations_")

    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 5))
    Y = rng.standard_normal((20, 2))
    est.fit(X, Y)
    assert hasattr(est, "y_weights_")
    assert hasattr(est, "y_rotations_")


@pytest.mark.parametrize("algorithm", [1, 2])
# All 16 center/scale combinations. Note the mixed cases (e.g. center_X=False with
# scale_X=True) where inner_.X_mean is computed but must be treated as 0 in the
# reconstruction (predict does not subtract it when center_X is False).
@pytest.mark.parametrize(
    "center_X,center_Y,scale_X,scale_Y",
    list(product([True, False], repeat=4)),
)
@pytest.mark.parametrize("m", [1, 3])
def test_coef_intercept_reproduce_predict(
    algorithm, center_X, center_Y, scale_X, scale_Y, m
):
    """coef_/intercept_ are the affine map predict applies (all center/scale combos).

    Following scikit-learn's PLSRegression convention, predict centers X, so the
    reconstruction is ``(X - X_mean) @ coef_.T + intercept_`` (intercept_ is the
    Y-mean), not ``X @ coef_.T + intercept_``.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8))
    Y = rng.standard_normal((50, m))
    est = SklearnPLS(
        n_components=4,
        algorithm=algorithm,
        center_X=center_X,
        center_Y=center_Y,
        scale_X=scale_X,
        scale_Y=scale_Y,
        ddof=1,
    ).fit(X, Y)
    x_mean = (
        np.asarray(est.inner_.X_mean).ravel()
        if (center_X and est.inner_.X_mean is not None)
        else np.zeros(X.shape[1])
    )
    manual = (X - x_mean) @ est.coef_.T + est.intercept_  # (N, m)
    pred = est.predict(X)
    pred_2d = pred.reshape(-1, 1) if pred.ndim == 1 else pred
    assert_allclose(manual, pred_2d, atol=1e-9, rtol=1e-7)
    assert est.coef_.shape == (m, 8)
    assert est.intercept_.shape == (m,)


def test_dual_transform_and_inverse_transform():
    """transform/inverse_transform handle both X and Y, matching PLSRegression."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6))
    Y = rng.standard_normal((40, 2))
    A = 3
    est = SklearnPLS(n_components=A).fit(X, Y)

    T = est.transform(X)
    assert T.shape == (40, A)
    T2, U2 = est.transform(X, Y)
    assert T2.shape == (40, A) and U2.shape == (40, A)
    assert_allclose(T, T2, atol=1e-12)

    X_rec = est.inverse_transform(T)
    assert X_rec.shape == (40, 6)
    X_rec2, Y_rec2 = est.inverse_transform(T2, U2)
    assert X_rec2.shape == (40, 6) and Y_rec2.shape == (40, 2)
    assert_allclose(X_rec, X_rec2, atol=1e-12)


def test_inverse_transform_roundtrip_full_rank():
    """At n_components == n_features, transform/inverse_transform round-trips X."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 5))
    Y = rng.standard_normal((30, 2))
    est = SklearnPLS(n_components=5, algorithm=1).fit(X, Y)  # A == n_features
    X_rec = est.inverse_transform(est.transform(X))
    assert_allclose(X_rec, X, atol=1e-8, rtol=1e-6)


def test_score_matches_r2_with_sample_weight():
    """score(X, y, sample_weight) equals sklearn r2_score on predict(X)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6))
    y = rng.standard_normal(40)
    est = SklearnPLS(n_components=3).fit(X, y)
    sw = rng.uniform(0.5, 2.0, size=40)
    assert_allclose(est.score(X, y), r2_score(y, est.predict(X)))
    assert_allclose(
        est.score(X, y, sample_weight=sw),
        r2_score(y, est.predict(X), sample_weight=sw),
    )


def test_get_params_set_params_clone_round_trip():
    """get_params exposes every constructor param; set_params and clone round-trip.

    Guards the param<->attribute-name contract (including the class-object ``dtype``
    param) and that ``clone`` yields an equivalent, unfitted estimator.
    """
    from sklearn.base import clone

    non_defaults = dict(
        n_components=3,
        algorithm=2,
        center_X=False,
        scale_Y=False,
        ddof=1,
        dtype=np.float32,
    )
    est = SklearnPLS(**non_defaults)
    params = est.get_params()
    for key, value in non_defaults.items():
        assert params[key] == value

    round_tripped = SklearnPLS().set_params(**params)
    assert round_tripped.get_params() == params

    cloned = clone(est)
    assert cloned.get_params() == params
    assert not hasattr(cloned, "inner_")  # clone is unfitted


def test_refit_overwrites_state_and_is_deterministic():
    """Refitting overwrites fitted state (no stale attrs), and refitting on identical
    data is deterministic."""
    rng = np.random.default_rng(0)
    X1 = rng.standard_normal((40, 10))
    Y1 = rng.standard_normal((40, 2))
    X2 = rng.standard_normal((30, 6))
    Y2 = rng.standard_normal((30, 3))

    est = SklearnPLS(n_components=3).fit(X1, Y1)
    assert est.n_features_in_ == 10
    assert est.coef_.shape == (2, 10)
    est.fit(X2, Y2)
    assert est.n_features_in_ == 6
    assert est.coef_.shape == (3, 6)

    first = SklearnPLS(n_components=3).fit(X2, Y2).coef_
    second = SklearnPLS(n_components=3).fit(X2, Y2).coef_
    np.testing.assert_array_equal(first, second)


def test_failed_refit_leaves_previous_fit_intact():
    """A refit that raises must not corrupt the surviving fitted state.

    scikit-learn defines no contract for estimator state after a failed fit (and
    check_estimator never exercises fit-success -> fit-failure -> keep using), but
    this estimator guarantees the stronger property: a failed refit leaves the
    estimator behaving exactly as the previous successful fit. Regression test for
    the 1D-y bookkeeping, which used to be recorded before the inner fit could
    fail: a failed 1D-y refit then made predict silently ravel the surviving
    multi-target predictions to the wrong shape.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 5))
    Y = rng.standard_normal((20, 3))
    y_1d = rng.standard_normal(20)

    est = SklearnPLS(n_components=2).fit(X, Y)
    inner_before = est.inner_
    preds_before = est.predict(X).copy()
    coef_before = est.coef_.copy()

    # Negative sample weights make the inner fit raise (after the flag used to
    # be set), aborting the refit partway through.
    with pytest.raises(ValueError):
        est.fit(X, y_1d, sample_weight=-np.ones(20))

    # The estimator must behave exactly as the previous successful fit: same
    # inner model, same (N, 3) prediction shape (no 1D ravel), same values.
    assert est.inner_ is inner_before
    preds_after = est.predict(X)
    assert preds_after.shape == (20, 3)
    np.testing.assert_array_equal(preds_after, preds_before)
    np.testing.assert_array_equal(est.coef_, coef_before)


def test_fit_returns_self():
    """fit returns the estimator instance (sklearn contract)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 5))
    y = rng.standard_normal(20)
    est = SklearnPLS(n_components=2)
    assert est.fit(X, y) is est


def test_feature_names_in_from_dataframe():
    """A pandas DataFrame input records feature_names_in_ (env-independent check)."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 5))
    Y = rng.standard_normal((30, 2))
    cols = [f"f{i}" for i in range(5)]
    est = SklearnPLS(n_components=3).fit(pd.DataFrame(X, columns=cols), Y)
    np.testing.assert_array_equal(
        est.feature_names_in_, np.asarray(cols, dtype=object)
    )
    assert est.n_features_in_ == 5


def test_set_output_pandas_transform():
    """set_output(transform='pandas') makes transform(X) return a named DataFrame."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 5))
    Y = rng.standard_normal((30, 2))
    A = 3
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    est = SklearnPLS(n_components=A).set_output(transform="pandas").fit(df, Y)
    out = est.transform(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == [f"pls{i}" for i in range(A)]


def test_univariate_1d_y_equivalence():
    """A 1D y yields the same coef_ as an equivalent (n, 1) column y."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6))
    y = rng.standard_normal(40)
    est_1d = SklearnPLS(n_components=3).fit(X, y)
    est_2d = SklearnPLS(n_components=3).fit(X, y.reshape(-1, 1))
    assert_allclose(est_1d.coef_, est_2d.coef_, atol=1e-12, rtol=1e-9)
    assert est_1d.coef_.shape == est_2d.coef_.shape == (1, 6)


def test_1d_y_predict_shape_and_voting_regressor():
    """A 1D y gives a 1D predict, and the wrapper works inside a VotingRegressor."""
    from sklearn.ensemble import VotingRegressor
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 5))
    coef = rng.standard_normal(5)
    y = X @ coef
    est = SklearnPLS(n_components=3).fit(X, y)
    pred = est.predict(X)
    assert pred.ndim == 1 and pred.shape == (40,)

    voting = VotingRegressor(
        [("pls", SklearnPLS(n_components=3)), ("lr", LinearRegression())]
    ).fit(X, y)
    assert voting.predict(X).shape == (40,)


def test_scaling_coef_recovers_true_coefficient():
    """With scaling, coef_ recovers the true generating coefficient (both X- and
    Y-std applied), reproducing scikit-learn's scaling_coef ground-truth check."""
    rng = np.random.default_rng(0)
    n, p = 500, 5
    X = rng.standard_normal((n, p)) * 10.0
    coef = rng.uniform(size=(3, p))
    Y = X @ coef.T
    # ddof does not affect this test: it only rescales the standardized space by a
    # single factor sqrt(n / (n - ddof)) that is common to every X and Y column, and
    # that factor cancels when coef_ is mapped back to the original space. So the
    # recovered coef_ is identical for any ddof; ddof=1 here mirrors scikit-learn's
    # PLSRegression scaling convention.
    est = SklearnPLS(n_components=p, scale_X=True, scale_Y=True, ddof=1).fit(X, Y)
    assert_allclose(est.coef_, coef, atol=1e-6, rtol=1e-4)
    assert_allclose(est.predict(X), Y, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("shape,bad_n_components", [((10, 5), 6), ((20, 64), 21)])
def test_n_components_upper_bound_raises(shape, bad_n_components):
    """n_components exceeding min(n_samples, n_features) raises ValueError, matching
    scikit-learn's PLSRegression (which bounds n_components by min(n_samples,
    n_features))."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal(shape)
    y = rng.standard_normal(shape[0])
    with pytest.raises(ValueError):
        SklearnPLS(n_components=bad_n_components).fit(X, y)


def test_exposed_attributes_are_independent_copies():
    """Mutating an exposed fitted attribute in place must not corrupt the inner
    model. The eager attributes are explicit copies; this pins the two attributes
    derived from live inner buffers -- ``intercept_`` (else aliases
    ``inner_.Y_mean``, read by predict for un-centering) and ``y_rotations_`` (else
    aliases the inner ``R_Y`` cache that ``transform`` also reads) -- which regress
    to views whenever ``np.asarray(..., dtype=...)`` is a dtype no-op."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6))
    Y = rng.standard_normal((40, 3))
    est = SklearnPLS(
        n_components=3, center_X=True, center_Y=True, scale_X=True, scale_Y=True
    ).fit(X, Y)

    baseline = est.predict(X).copy()
    rot_before = est.y_rotations_.copy()

    # Corrupt the exposed attributes in place.
    est.intercept_[...] = 1e9
    est.y_rotations_[...] = 1e9

    # Predictions and a fresh y_rotations_ access must be exactly unaffected.
    assert_allclose(est.predict(X), baseline, atol=0, rtol=0)
    assert_allclose(est.y_rotations_, rot_before, atol=0, rtol=0)
