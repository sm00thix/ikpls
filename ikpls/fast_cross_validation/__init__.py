"""Fast cross-validation backends, imported lazily (PEP 562).

``numpy`` is always available; ``jax`` is optional (``pip install ikpls[jax]``)
and imported only on first access, so ``ikpls.fast_cross_validation.numpy`` never
imports JAX. When JAX is not installed, ``ikpls.fast_cross_validation.jax`` raises
``AttributeError`` (so ``hasattr`` is ``False``) and importing it raises
``ImportError``.
"""

import importlib

# ``jax`` is intentionally omitted from ``__all__`` so that ``import *`` never
# requires the optional JAX dependency; it stays accessible as ``.jax``.
__all__ = ["numpy"]

_LAZY_SUBMODULES = ("jax", "numpy")
_OPTIONAL_SUBMODULES = frozenset({"jax"})


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        try:
            return importlib.import_module(f"{__name__}.{name}")
        except ImportError:
            if name in _OPTIONAL_SUBMODULES:
                # Report a missing optional backend as an absent attribute so that
                # ``hasattr(..., name)`` is False (JAX is optional).
                raise AttributeError(
                    f"module {__name__!r} has no attribute {name!r}; the optional "
                    f"'{name}' backend requires 'pip install ikpls[{name}]'"
                ) from None
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
