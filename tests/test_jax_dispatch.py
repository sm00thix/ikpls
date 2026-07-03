"""
Tests for the ``ikpls.jax.PLS`` dispatcher.

``ikpls.jax.PLS(algorithm=1|2)`` is a factory that returns the algorithm-specific
implementation from ``ikpls._impl`` while remaining ``isinstance`` /
``issubclass``-compatible with ``ikpls.jax.PLS`` and carrying the full public API
on the class itself (so it is documentable). These tests guard that contract,
including that constructing via the dispatcher is numerically identical to using
the private implementation directly.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import numpy as np
import pytest

from ikpls._impl.jax_alg_1 import PLS as _Alg1
from ikpls._impl.jax_alg_2 import PLS as _Alg2
from ikpls.jax import PLS

PUBLIC_API = [
    "fit",
    "predict",
    "transform",
    "fit_transform",
    "inverse_transform",
    "stateless_fit",
    "stateless_predict",
    "cross_validate",
]


def test_public_api_present_on_dispatcher():
    """All public methods live on ikpls.jax.PLS itself, so autodoc documents them."""
    missing = [m for m in PUBLIC_API if not hasattr(PLS, m)]
    assert not missing, f"ikpls.jax.PLS is missing public methods: {missing}"


@pytest.mark.parametrize("algorithm, impl", [(1, _Alg1), (2, _Alg2)])
def test_dispatch_isinstance_and_algorithm(algorithm, impl):
    pls = PLS(algorithm=algorithm)
    assert isinstance(pls, PLS)  # via ABC registration
    assert issubclass(type(pls), PLS)
    assert isinstance(pls, impl)  # returns the algorithm-specific implementation
    assert pls.algorithm == algorithm


@pytest.mark.parametrize("bad", [0, 3, -1])
def test_bad_algorithm_raises_valueerror(bad):
    with pytest.raises(ValueError):
        PLS(algorithm=bad)


@pytest.mark.parametrize("algorithm, impl", [(1, _Alg1), (2, _Alg2)])
def test_dispatcher_matches_impl_numerically(algorithm, impl):
    """PLS(algorithm=k) is bit-identical to using the private impl directly.

    The kwargs deliberately use NON-default values so that a dispatcher that
    silently dropped ``**kwargs`` could not pass (with defaults, dropping them
    would be invisible)."""
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.standard_normal((30, 6)))
    Y = jnp.asarray(rng.standard_normal((30, 2)))
    kwargs = dict(center_X=False, center_Y=True, scale_X=True, scale_Y=False, ddof=1)

    via_dispatcher = PLS(algorithm=algorithm, **kwargs)
    via_dispatcher.fit(X, Y, 3)
    direct = impl(**kwargs)
    direct.fit(X, Y, 3)

    # The non-default kwargs must actually land on the returned implementation
    # (guards the dispatcher's "**kwargs forwarded verbatim" contract).
    for name, value in kwargs.items():
        assert getattr(via_dispatcher, name) == value, name

    np.testing.assert_array_equal(
        np.asarray(via_dispatcher.B), np.asarray(direct.B)
    )
