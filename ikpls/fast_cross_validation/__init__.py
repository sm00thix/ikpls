__all__ = ["numpy_ikpls"]

from . import numpy_ikpls

try:
    from . import jax_ikpls

    __all__ += ["jax_ikpls"]
except ImportError:
    # JAX is an optional dependency. Install with: pip install ikpls[jax]
    pass
