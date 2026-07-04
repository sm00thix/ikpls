"""
This file contains the _R_Y_Mapping class: a read-only, lazily-computed Mapping
that stores the PLS weights used to compute the Y-scores U directly from the
original (centered and scaled) Y for any number of components.

A single implementation serves both the NumPy and JAX PLS classes via a
``backend`` argument, mirroring the backend-parametrized design used in the
cvmatrix package. The mapping is only ever accessed host-side (from ``transform``
and from the scikit-learn wrapper's ``fit``), never from inside a jitted or
vmapped region, so the backend branch has no effect on any hot path.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

from collections.abc import Mapping

import numpy as np
import numpy.linalg as la


class _R_Y_Mapping(Mapping):
    """A read-only Mapping that computes values lazily on first access.

    For ``key`` components the value is the pseudo-inverse of ``Q[:, :key].T``,
    computed once on first access and cached for future retrieval.

    Parameters
    ----------
    Q : array of shape (M, A)
        The Y-loadings matrix, where M is the number of targets and A is the
        (maximum) number of components.

    backend : {"numpy", "jax"}, default="numpy"
        Which array backend to use for the pseudo-inverse. ``"jax"`` is imported
        lazily so that a NumPy-only installation never requires JAX.
    """

    def __init__(self, Q, backend: str = "numpy") -> None:
        if backend == "numpy":
            finfo = np.finfo
            self._pinv = la.pinv
        elif backend == "jax":
            import jax.numpy as jnp
            import jax.numpy.linalg as jla

            finfo = jnp.finfo
            self._pinv = jla.pinv
        else:
            raise ValueError(
                f"backend must be 'numpy' or 'jax', but was {backend!r}."
            )
        self.Q = Q
        self.backend = backend
        self.eps = finfo(Q.dtype).eps
        self._valid_keys = set(range(1, Q.shape[1] + 1))
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._valid_keys:
            raise KeyError(
                f"Invalid number of components: {key}. Valid numbers of components are 1 to {self.Q.shape[1]}."
            )
        if key not in self._cache:
            self._cache[key] = self._pinv(self.Q[:, :key].T, rcond=self.eps)
        return self._cache[key]

    def __iter__(self):
        return iter(self._valid_keys)

    def __len__(self):
        return len(self._valid_keys)

    def __contains__(self, key):
        return key in self._valid_keys

    def __repr__(self):
        return f"_R_Y_Mapping(Cached R_Y for {len(self._cache)}/{len(self._valid_keys)} n_components)"
