"""
Contains the cache of compiled programs shared by the JAX cross-validation methods
(``ikpls.jax.PLS.cross_validate`` and the fast cross-validation methods in
``ikpls.fast_cross_validation.jax``), so that repeated calls, e.g. one per local model,
reuse one compiled program instead of compiling anew.

Author: Ole-Christian Galbo Engstrøm
E-mail: ocge@foss.dk
"""

import inspect
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Optional

_MAX_CACHED_SETTINGS = 64


def _same_function(
    a: Optional[Callable[..., Any]], b: Optional[Callable[..., Any]]
) -> bool:
    """
    Whether `a` and `b` are the same function: the same object, or the same method of
    the same object (a bound method is a new object on every access). Equality is not
    used: an equal object may compute something else (e.g. a mutated dataclass), and
    comparing could stage operations on tracers of a finished trace.
    """
    if a is b:
        return True
    return (
        inspect.ismethod(a)
        and inspect.ismethod(b)
        and a.__self__ is b.__self__
        and a.__func__ is b.__func__
    )


def _constructor_names(cls: type) -> list:
    """The names of the constructor arguments of `cls`, following a subclass
    constructor's ``**kwargs`` to its base classes' constructor arguments."""
    names: list = []
    for klass in cls.__mro__:
        if "__init__" not in vars(klass):
            continue
        params = inspect.signature(vars(klass)["__init__"]).parameters.values()
        names += [p.name for p in params if _is_named(p) and p.name not in names]
        if all(p.kind != p.VAR_KEYWORD for p in params):
            break
    return names


def _is_named(p: inspect.Parameter) -> bool:
    return p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)


def _typed(value: Any) -> tuple:
    """`value` with its type, also for the items of a tuple, so that equal values of
    different types key different programs."""
    if isinstance(value, tuple):
        return (type(value), tuple(_typed(v) for v in value))
    return (type(value), value)


def constructor_settings(obj: Any) -> Optional[tuple]:
    """
    The constructor arguments of `obj`, read from the attributes of the same names (the
    scikit-learn convention, also for subclasses), as a hashable key for its compiled
    programs; None if they cannot key them. Each value is keyed with its type (also
    inside tuples), as equal values of different types (e.g. 1 and ``np.float64(1.0)``)
    can compute differently.
    A subclass constructor's ``**kwargs`` stand for its base class's constructor
    arguments.
    """
    try:
        settings = tuple(
            (name, _typed(value), value)
            for name, value in (
                (name, getattr(obj, name)) for name in _constructor_names(type(obj))
            )
        )
        hash(settings)
    except (AttributeError, TypeError, ValueError):
        return None
    return settings


class _CompiledCache:
    """
    Compiled programs reused across calls: per (class, constructor settings), the
    program that ``build`` makes for a new instance with those settings and the most
    recently used user functions (`metric_function`, and `preprocessing_function` where
    there is one). Other functions replace it (one function can compute several
    metrics), so a function created anew per call compiles anew and is kept only until
    the next call with the same settings. The instance is reused for new functions, and
    also after its settings were evicted for as long as JAX's caches, which keep every
    instance its jitted methods were traced for, keep it alive. Keying by the
    settings rather than by the instance keeps instances picklable, lets copies use
    their own settings, and makes a change of an instance's settings take effect. The
    programs of the `_MAX_CACHED_SETTINGS` most recently used settings are kept. Each
    program is ``jax.jit``-compiled per input shapes/dtypes and the static arguments.
    """

    def __init__(self, build: Callable[..., Callable[..., Any]]) -> None:
        self._build = build
        self._entries: OrderedDict = OrderedDict()  # key -> (functions, program, pls)
        self._instances: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._lock = threading.Lock()

    def get(
        self, cls: type, settings: tuple, *functions: Optional[Callable[..., Any]]
    ) -> Optional[Callable[..., Any]]:
        """The program, or None if a new instance of `cls` constructed from
        `settings` does not reproduce them (it would compute something else)."""
        key = (cls, settings)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and all(
                _same_function(a, b) for a, b in zip(entry[0], functions)
            ):
                self._entries.move_to_end(key)
                return entry[1]
            pls = entry[2] if entry is not None else self._instances.get(key)
        if pls is None:
            # Built outside the lock: the constructor of a subclass is user code.
            pls = self._new_instance(cls, settings)
            if pls is None:
                return None
        program = self._build(pls, *functions)
        with self._lock:
            self._instances[key] = pls
            self._entries[key] = (functions, program, pls)
            self._entries.move_to_end(key)
            while len(self._entries) > _MAX_CACHED_SETTINGS:
                self._entries.popitem(last=False)
        return program

    @staticmethod
    def _new_instance(cls: type, settings: tuple) -> Any:
        """A new instance of `cls` constructed from `settings`, or None if it does
        not reproduce them."""
        try:
            pls = cls(**{name: value for name, _, value in settings})
            reproduced = pls._settings() == settings
        except (TypeError, ValueError):
            return None
        return pls if reproduced else None
