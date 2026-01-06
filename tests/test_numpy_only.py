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

    assert hasattr(ikpls, "numpy_ikpls")
    assert hasattr(ikpls, "fast_cross_validation")


def test_jax_modules_not_loaded():
    """Verify JAX algorithms are NOT available when JAX is not installed."""
    import ikpls

    assert not hasattr(ikpls, "jax_ikpls_alg_1"), (
        "jax_ikpls_alg_1 should not be loaded without JAX installed"
    )
    assert not hasattr(ikpls, "jax_ikpls_alg_2"), (
        "jax_ikpls_alg_2 should not be loaded without JAX installed"
    )


def test_jax_direct_import_fails():
    """Verify importing JAX modules directly raises ImportError."""
    with pytest.raises(ImportError):
        from ikpls import jax_ikpls_alg_1  # noqa: F401

    with pytest.raises(ImportError):
        from ikpls import jax_ikpls_alg_2  # noqa: F401


def test_numpy_pls_alg_1_fit():
    """Verify NumPy PLS Algorithm 1 can fit without JAX."""
    from ikpls.numpy_ikpls import PLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    pls = PLS(algorithm=1)
    pls.fit(X, Y, A=5)

    assert pls.B is not None
    assert pls.B.shape == (5, 10, 2)


def test_numpy_pls_alg_2_fit():
    """Verify NumPy PLS Algorithm 2 can fit without JAX."""
    from ikpls.numpy_ikpls import PLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    pls = PLS(algorithm=2)
    pls.fit(X, Y, A=5)

    assert pls.B is not None
    assert pls.B.shape == (5, 10, 2)


def test_numpy_pls_predict():
    """Verify NumPy PLS can predict without JAX."""
    from ikpls.numpy_ikpls import PLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    pls = PLS(algorithm=1)
    pls.fit(X, Y, A=5)
    predictions = pls.predict(X)

    assert predictions.shape == (5, 100, 2)


def test_numpy_pls_single_component_predict():
    """Verify NumPy PLS can predict with a specific number of components."""
    from ikpls.numpy_ikpls import PLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    pls = PLS(algorithm=1)
    pls.fit(X, Y, A=5)
    predictions = pls.predict(X, n_components=3)

    assert predictions.shape == (100, 2)


def test_fast_cv_numpy_pls_alg_1():
    """Verify Fast CV NumPy PLS Algorithm 1 works without JAX."""
    from ikpls.fast_cross_validation.numpy_ikpls import PLS as FastCVPLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    pls = FastCVPLS(algorithm=1)
    pls.fit(X, Y, A=5)

    assert pls.B is not None
    assert pls.B.shape == (5, 10, 2)


def test_fast_cv_numpy_pls_alg_2():
    """Verify Fast CV NumPy PLS Algorithm 2 works without JAX."""
    from ikpls.fast_cross_validation.numpy_ikpls import PLS as FastCVPLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    pls = FastCVPLS(algorithm=2)
    pls.fit(X, Y, A=5)

    assert pls.B is not None
    assert pls.B.shape == (5, 10, 2)


def test_numpy_pls_cross_validate():
    """Verify NumPy PLS cross-validation works without JAX."""
    from ikpls.numpy_ikpls import PLS

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10))
    Y = rng.standard_normal((100, 2))

    # Create 5-fold split indices
    splits = np.repeat(np.arange(5), 20)

    def rmse(Y_true, Y_pred):
        return np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))

    pls = PLS(algorithm=1)
    results = pls.cross_validate(X, Y, A=5, cv_splits=splits, metric_function=rmse)

    assert results is not None
    assert len(results) == 5  # 5 folds
