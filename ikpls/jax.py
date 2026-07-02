"""
This file exposes the JAX Improved Kernel PLS implementation as a single ``PLS``
class parameterized by ``algorithm`` (1 or 2), mirroring ``ikpls.numpy.PLS``. It
dispatches to the algorithm-specific implementations in ``ikpls._impl``; those
classes are private and should be used through this entry point.

Requires the ``jax`` extra (``pip install ikpls[jax]``).

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from ikpls._impl.jax_alg_1 import PLS as _PLSAlgorithm1
from ikpls._impl.jax_alg_2 import PLS as _PLSAlgorithm2
from ikpls._impl.jax_base import PLSBase


class PLS(PLSBase):
    """JAX Improved Kernel PLS (Algorithm #1 or #2).

    ``ikpls.jax.PLS(algorithm=1)`` and ``ikpls.jax.PLS(algorithm=2)`` construct
    the Algorithm #1 and #2 implementations, respectively. Both share the same
    public API (``fit``, ``predict``, ``transform``, ``fit_transform``,
    ``inverse_transform``, ``cross_validate``, and the ``stateless_*`` methods).
    Algorithm #1 is faster when the number of features K exceeds the number of
    samples N; Algorithm #2 is faster when N > K. Both yield identical results.

    ``ikpls.jax.PLS`` subclasses the shared JAX PLS base, so it carries and
    documents the full public API listed above. Constructing it is a dispatcher
    that returns an instance of the selected algorithm-specific implementation
    (registered as a virtual subclass of this class), so:

    * ``isinstance(model, ikpls.jax.PLS)`` (and ``issubclass``) is ``True``;
    * ``model.algorithm`` is ``1`` or ``2``;
    * ``model.stateless_fit`` / ``jax.vmap`` operate directly on the
      implementation with no wrapping indirection or performance cost;
    * the algorithm-specific classes stay private in ``ikpls._impl``.

    Parameters
    ----------
    algorithm : int, default=1
        Which Improved Kernel PLS algorithm to use (1 or 2).

    **kwargs
        Forwarded verbatim to the algorithm-specific implementation
        (``center_X``, ``center_Y``, ``scale_X``, ``scale_Y``, ``ddof``,
        ``dtype``, ``verbose``, ...). See the implementation for the full
        signature.
    """

    def __new__(cls, algorithm: int = 1, *args, **kwargs):
        try:
            impl_cls = {1: _PLSAlgorithm1, 2: _PLSAlgorithm2}[algorithm]
        except KeyError:
            raise ValueError(
                f"algorithm must be 1 or 2, but was {algorithm!r}."
            ) from None
        # impl_cls subclasses PLSBase but not this PLS (it is a sibling under
        # PLSBase, registered as a virtual subclass), so the returned instance
        # fails type.__call__'s C-level real-subtype check and PLS.__init__ is
        # never invoked on it -- no double initialization. The instance is fully
        # constructed here.
        instance = impl_cls(*args, **kwargs)
        instance.algorithm = algorithm
        return instance


# Register the implementations as virtual subclasses so isinstance / issubclass
# against ikpls.jax.PLS hold, without making them real subclasses (which would
# trigger a second __init__ via type.__call__).
PLS.register(_PLSAlgorithm1)
PLS.register(_PLSAlgorithm2)
