__version__ = "4.0.1"
__all__ = ["fast_cross_validation", "numpy_ikpls"]

from . import fast_cross_validation, numpy_ikpls

try:
    from . import jax_ikpls_alg_1, jax_ikpls_alg_2

    __all__ += ["jax_ikpls_alg_1", "jax_ikpls_alg_2"]
except ImportError:
    # JAX is an optional dependency. Install with: pip install ikpls[jax]
    pass
