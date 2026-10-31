"""
Tests that run without JAX to verify the package works for numpy-only installations.

These tests are specifically designed to run when JAX is NOT installed (i.e., when the
package is installed with `pip install ikpls` instead of `pip install ikpls[jax]`).

Author: Ole-Christian Galbo Engstrøm
E-mail: ole.e@di.ku.dk
"""

import numpy as np
import pytest


def test_package_imports_without_jax():
    """Verify the package can be imported without JAX installed."""
    import ikpls

    assert hasattr(ikpls, "numpy")
    assert hasattr(ikpls, "fast_cross_validation")


def test_jax_modules_not_loaded():
    """Verify JAX algorithms are NOT available when JAX is not installed."""
    import ikpls

    assert not hasattr(ikpls, "jax"), (
        "ikpls.jax should not be loaded without JAX installed"
    )


def test_jax_direct_import_fails():
    """Verify importing JAX modules directly raises ImportError."""
    with pytest.raises(ImportError):
        from ikpls import jax  # noqa: F401


def test_fast_cv_jax_not_loaded():
    """Verify the fast cross-validation JAX backend is NOT available without JAX."""
    import ikpls.fast_cross_validation

    assert not hasattr(ikpls.fast_cross_validation, "jax"), (
        "ikpls.fast_cross_validation.jax should not be loaded without JAX installed"
    )


def test_fast_cv_jax_direct_import_fails():
    """Verify importing the fast cross-validation JAX backend raises ImportError."""
    with pytest.raises(ImportError):
        from ikpls.fast_cross_validation import jax  # noqa: F401


def test_numpy_pls_alg_1_fit():
    """Verify NumPy PLS Algorithm 1 can fit without JAX."""
    from ikpls.numpy import PLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    pls = PLS(algorithm=1)
    pls.fit(X, Y, A=5)

    assert pls.B is not None
    assert pls.B.shape == (5, 10, 2)


def test_numpy_pls_alg_2_fit():
    """Verify NumPy PLS Algorithm 2 can fit without JAX."""
    from ikpls.numpy import PLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    pls = PLS(algorithm=2)
    pls.fit(X, Y, A=5)

    assert pls.B is not None
    assert pls.B.shape == (5, 10, 2)


def test_numpy_pls_predict():
    """Verify NumPy PLS can predict without JAX."""
    from ikpls.numpy import PLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    pls = PLS(algorithm=1)
    pls.fit(X, Y, A=5)
    predictions = pls.predict(X)

    assert predictions.shape == (5, 100, 2)


def test_numpy_pls_single_component_predict():
    """Verify NumPy PLS can predict with a specific number of components."""
    from ikpls.numpy import PLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    pls = PLS(algorithm=1)
    pls.fit(X, Y, A=5)
    predictions = pls.predict(X, n_components=3)

    assert predictions.shape == (100, 2)


def test_fast_cv_numpy_pls_alg_1():
    """Verify Fast CV NumPy PLS Algorithm 1 works without JAX."""
    from ikpls.fast_cross_validation.numpy import PLS as FastCVPLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    # Create 5-fold split indices
    folds = np.repeat(np.arange(5), 20)

    def rmse(Y_true, Y_pred):
        return np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))

    pls = FastCVPLS(algorithm=1)
    results = pls.cross_validate(X, Y, A=5, folds=folds, metric_function=rmse)

    assert results is not None
    assert len(results) == 5  # 5 folds


def test_fast_cv_numpy_pls_alg_2():
    """Verify Fast CV NumPy PLS Algorithm 2 works without JAX."""
    from ikpls.fast_cross_validation.numpy import PLS as FastCVPLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    # Create 5-fold split indices
    folds = np.repeat(np.arange(5), 20)

    def rmse(Y_true, Y_pred):
        return np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))

    pls = FastCVPLS(algorithm=2)
    results = pls.cross_validate(X, Y, A=5, folds=folds, metric_function=rmse)

    assert results is not None
    assert len(results) == 5  # 5 folds


def test_numpy_pls_cross_validate():
    """Verify NumPy PLS cross-validation works without JAX."""
    from ikpls.numpy import PLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    # Create 5-fold split indices
    folds = np.repeat(np.arange(5), 20)

    def rmse(Y_true, Y_pred):
        return np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))

    pls = PLS(algorithm=1)
    results = pls.cross_validate(X, Y, A=5, folds=folds, metric_function=rmse)

    assert results is not None
    assert len(results) == 5  # 5 folds


def _mse_per_component(Y_true, Y_pred, sample_weight=None):
    if sample_weight is None:
        return np.mean((Y_true - Y_pred) ** 2, axis=-2)
    sq = sample_weight[:, None] * (Y_true - Y_pred) ** 2
    return np.sum(sq, axis=-2) / np.sum(sample_weight)


@pytest.mark.parametrize("algorithm", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_numpy_cross_validate_thread_safe(algorithm, weighted):
    """`ikpls.numpy.PLS.cross_validate` must give the same per-fold metrics when its
    folds run concurrently in one process (joblib's threading backend) as when run
    serially, and must not fit the instance itself. A barrier after `fit` makes every
    fold finish fitting before any fold predicts, so folds sharing one model would get
    wrong metrics deterministically rather than only when a race happens to occur."""
    import threading

    from joblib import parallel_config
    from sklearn.utils._testing import assert_allclose

    from ikpls.numpy import PLS

    n_folds = 12
    barrier = threading.Barrier(n_folds, timeout=60)

    class BarrierPLS(PLS):
        def fit(self, *args, **kwargs):
            fitted = super().fit(*args, **kwargs)
            barrier.wait()
            return fitted

    rng = np.random.default_rng(8)
    N, K, M, A = 240, 30, 2, 8
    X = rng.standard_normal((N, K))
    Y = X[:, :5] @ rng.standard_normal((5, M)) + 0.1 * rng.standard_normal((N, M))
    folds = np.arange(N) % n_folds
    sample_weight = rng.uniform(0.5, 2.0, size=N) if weighted else None
    serial = PLS(algorithm=algorithm).cross_validate(
        X, Y, A, folds, _mse_per_component, sample_weight=sample_weight,
        n_jobs=1, verbose=0,
    )
    shared = BarrierPLS(algorithm=algorithm)
    with parallel_config(backend="threading"):
        threaded = shared.cross_validate(
            X, Y, A, folds, _mse_per_component, sample_weight=sample_weight,
            n_jobs=n_folds, verbose=0,
        )
    for f in range(n_folds):
        assert_allclose(threaded[f], serial[f], rtol=1e-12, atol=1e-14)
    assert not shared.fitted_ and shared.B is None


def test_fast_cv_numpy_alg_1_does_not_form_xtx():
    """Algorithm #1 fast cross-validation never uses the dataset-wide X^T W X, so it is
    not formed (cvmatrix's compute_XTX=False); Algorithm #2 needs and forms it."""
    from ikpls.fast_cross_validation.numpy import PLS as FastCVPLS

    rng = np.random.default_rng(7)
    N, K, M, A = 30, 50, 2, 4
    X = rng.standard_normal((N, K))
    Y = rng.standard_normal((N, M))
    folds = np.arange(N) % 5
    fcv_1 = FastCVPLS(algorithm=1)
    fcv_1.cross_validate(X, Y, A, folds, _mse_per_component, n_jobs=1, verbose=0)
    assert fcv_1.cvm.XTX is None
    fcv_2 = FastCVPLS(algorithm=2)
    fcv_2.cross_validate(X, Y, A, folds, _mse_per_component, n_jobs=1, verbose=0)
    assert fcv_2.cvm.XTX is not None and fcv_2.cvm.XTX.shape == (K, K)



def _small_cv_problem(seed=12, N=60, K=8, M=2, n_folds=4):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1, 5, size=(N, K)) * rng.uniform(0.1, 10, size=K)
    Y = rng.standard_normal((N, M))
    return X, Y, np.arange(N) % n_folds


def test_numpy_cross_validate_reused_instance():
    """`ikpls.numpy.PLS.cross_validate` keeps one fold model of its last call outside the
    instance (to reuse its memory): any setting changed between calls, constructor
    argument or not, must take effect, also with the threading backend; at most one
    model is kept per instance; pickling and copying the instance are unaffected."""
    import copy
    import pickle

    from joblib import parallel_config
    from sklearn.utils._testing import assert_allclose

    import ikpls.numpy as ikpls_numpy
    from ikpls.numpy import PLS

    X, Y, folds = _small_cv_problem()
    A = 4

    def cv(pls, **kwargs):
        return pls.cross_validate(X, Y, A, folds, _mse_per_component, verbose=0, **kwargs)

    pls = PLS(algorithm=1)  # scales X and Y by default
    scaled = cv(pls, n_jobs=1)
    pls.scale_X = pls.scale_Y = False  # a constructor setting
    unscaled = cv(pls, n_jobs=1)
    reference = cv(PLS(algorithm=1, scale_X=False, scale_Y=False), n_jobs=1)
    pls.eps = 1e-2  # not a constructor setting
    big_eps = cv(pls, n_jobs=1)
    reference_eps = PLS(algorithm=1, scale_X=False, scale_Y=False)
    reference_eps.eps = 1e-2
    reference_big_eps = cv(reference_eps, n_jobs=1)
    with parallel_config(backend="threading"):
        threaded = [cv(pls, n_jobs=4) for _ in range(2)]  # the second call reuses
    for f in range(4):
        assert not np.allclose(scaled[f], unscaled[f])
        assert_allclose(unscaled[f], reference[f], rtol=1e-12, atol=1e-14)
        assert_allclose(big_eps[f], reference_big_eps[f], rtol=1e-12, atol=1e-14)
        for t in threaded:
            assert_allclose(t[f], reference_big_eps[f], rtol=1e-12, atol=1e-14)
    assert not pls.fitted_
    kept = ikpls_numpy._CV_FOLD_MODELS.get(pls)
    assert kept is not None and kept is not pls
    unpickled = pickle.loads(pickle.dumps(pls))  # nosec B301
    for clone in (unpickled, copy.copy(pls), copy.deepcopy(pls)):
        assert vars(clone).keys() == vars(pls).keys() and not clone.fitted_
        assert clone.eps == 1e-2 and clone.scale_X is False


def test_numpy_cross_validate_leaves_a_fitted_instance_untouched():
    """An instance the user fitted keeps its fitted arrays (identity and values)
    through `cross_validate`, with the sequential and the threading backend."""
    from joblib import parallel_config

    from ikpls.numpy import PLS

    X, Y, folds = _small_cv_problem()
    pls = PLS(algorithm=2).fit(X, Y, 3)
    before = {k: v for k, v in vars(pls).items() if isinstance(v, np.ndarray)}
    copies = {k: v.copy() for k, v in before.items()}
    pls.cross_validate(X, Y, 3, folds, _mse_per_component, n_jobs=1, verbose=0)
    with parallel_config(backend="threading"):
        pls.cross_validate(X, Y, 3, folds, _mse_per_component, n_jobs=4, verbose=0)
    for k, v in before.items():
        assert getattr(pls, k) is v
        np.testing.assert_array_equal(v, copies[k])


def _make_shrunk_pls_class():
    from ikpls.numpy import PLS

    class _ShrunkPLS(PLS):
        """A slotted subclass (module level, so the standard pickle can find it)."""

        __slots__ = ("shrink",)

        def __init__(self, shrink, **kwargs):
            super().__init__(**kwargs)
            self.shrink = shrink

        def predict(self, X, n_components=None):
            return self.shrink * super().predict(X, n_components)

    _ShrunkPLS.__qualname__ = "_ShrunkPLS"
    return _ShrunkPLS


_ShrunkPLS = _make_shrunk_pls_class()


def test_numpy_cross_validate_slotted_subclass():
    """Subclasses with `__slots__` keep their slot values through pickling and copying,
    and the fold copies made by `cross_validate` carry them too."""
    import copy
    import pickle

    from ikpls.numpy import PLS

    X, Y, folds = _small_cv_problem()
    pls = _ShrunkPLS(0.5, algorithm=2)
    unpickled = pickle.loads(pickle.dumps(pls))  # nosec B301
    for clone in (unpickled, copy.copy(pls), copy.deepcopy(pls)):
        assert clone.shrink == 0.5
    out = pls.cross_validate(X, Y, 3, folds, _mse_per_component, n_jobs=1, verbose=0)
    ref = PLS(algorithm=2).cross_validate(
        X, Y, 3, folds,
        lambda Y_true, Y_pred, w=None: _mse_per_component(Y_true, 0.5 * Y_pred),
        n_jobs=1, verbose=0,
    )
    for f in range(4):
        np.testing.assert_allclose(out[f], ref[f], rtol=1e-12, atol=1e-14)


def test_numpy_cross_validate_ships_no_fitted_models_to_workers():
    """With a process-based backend, the instance is pickled to the workers without
    the fold model kept from a previous in-process call."""
    from ikpls.numpy import PLS

    pickled_fitted = []

    class Spy(PLS):
        def __reduce_ex__(self, protocol):
            pickled_fitted.append(bool(self.fitted_))
            return super().__reduce_ex__(protocol)

    X, Y, folds = _small_cv_problem(n_folds=8)
    pls = Spy(algorithm=2)
    pls.cross_validate(X, Y, 3, folds, _mse_per_component, n_jobs=1, verbose=0)
    pickled_fitted.clear()
    pls.cross_validate(X, Y, 3, folds, _mse_per_component, n_jobs=2, verbose=0)
    assert pickled_fitted and not any(pickled_fitted)
