"""IKPLS: Improved Kernel Partial Least Squares.

Public submodules are imported lazily (PEP 562), so importing one backend does
not import the others. In particular, ``import ikpls`` and ``import ikpls.numpy``
never import JAX, even when JAX is installed.

Submodules
----------
numpy
    NumPy PLS (always available).
sklearn
    scikit-learn-conformant wrapper (always available).
fast_cross_validation
    Fast cross-validation (NumPy always available; JAX optional).
jax
    JAX PLS (optional; ``pip install ikpls[jax]``).

The optional ``jax`` submodule is imported only on first access. When JAX is not
installed, ``ikpls.jax`` raises ``AttributeError`` (so ``hasattr(ikpls, "jax")``
is ``False``) and ``from ikpls import jax`` raises ``ImportError``.
"""

import importlib

__version__ = "6.1.2"

# ``jax`` is intentionally omitted from ``__all__`` so that ``from ikpls import *``
# never requires the optional JAX dependency; it stays accessible as ``ikpls.jax``.
__all__ = ["fast_cross_validation", "numpy", "sklearn"]

# All public submodules, loaded lazily on first attribute access.
_LAZY_SUBMODULES = ("fast_cross_validation", "jax", "numpy", "sklearn")
# Optional submodules whose absence is reported as a missing attribute.
_OPTIONAL_SUBMODULES = frozenset({"jax"})


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        try:
            return importlib.import_module(f"{__name__}.{name}")
        except ImportError as err:
            if name in _OPTIONAL_SUBMODULES:
                # Report a missing optional backend as an absent attribute so that
                # ``hasattr(ikpls, name)`` is False (JAX is optional). The original
                # ImportError is chained as __cause__ so that a real failure inside
                # an INSTALLED-but-broken JAX stack remains diagnosable from the
                # traceback instead of being masked as "not installed".
                raise AttributeError(
                    f"module {__name__!r} has no attribute {name!r}; the optional "
                    f"'{name}' backend requires 'pip install ikpls[{name}]'"
                ) from err
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
