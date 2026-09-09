"""Alias so ``pip install expanderkit`` and ``import expanderkit`` agree.

The implementation package is ``ramanujan`` — the name predates the repository
and is what existing experiment code imports, so it stays. This module makes
``expanderkit`` a second name for the SAME module objects, not a copy of them.

Why that distinction matters: if ``expanderkit.layer`` were a separate module
object from ``ramanujan.layer``, then ``SparseLinear`` would be two different
classes, ``isinstance`` would fail across the two spellings, and a registry
built under one name would not be the registry seen under the other. So every
submodule is imported once and registered under both names in ``sys.modules``,
rather than relying on ``__path__`` to re-import it later under a new name.

``ramanujan.layer`` needs torch and is skipped when torch is absent — the alias
for it appears only if the real module can be imported, which is the same
condition ``ramanujan.layer`` itself is under.
"""

from __future__ import annotations

import importlib as _importlib
import pkgutil as _pkgutil
import sys as _sys

import ramanujan as _ramanujan
from ramanujan import *  # noqa: F401,F403
from ramanujan import __all__  # noqa: F401

__version__ = getattr(_ramanujan, "__version__", "0.2.0")

# Import every submodule once so it can be aliased. Anything that fails to
# import is skipped and named in __skipped__ rather than propagating.
__skipped__: dict[str, str] = {}
for _info in _pkgutil.iter_modules(_ramanujan.__path__):
    _full = f"ramanujan.{_info.name}"
    if _full in _sys.modules:
        continue
    try:
        _importlib.import_module(_full)
    except Exception as _exc:  # noqa: BLE001 - a bad sibling must not be fatal
        __skipped__[_info.name] = f"{type(_exc).__name__}: {_exc}"

# Alias SUBMODULES only. Mapping the top-level name as well would put the
# ramanujan module object itself at sys.modules["expanderkit"], replacing this
# one and discarding everything defined here (__skipped__, __version__). What
# has to be shared is the submodules and therefore the classes inside them;
# whether the two top-level module objects are identical is nobody's business.
# Registering in sys.modules makes "from expanderkit.masks import X" work, but
# the import system normally also binds the submodule as an attribute of its
# parent package -- which it will not do here, because nothing is actually
# imported under the expanderkit name. Bind it explicitly so attribute access
# ("expanderkit.masks") behaves the same as the from-import.
for _name in list(_sys.modules):
    if _name.startswith("ramanujan."):
        _short = _name[len("ramanujan."):]
        _sys.modules["expanderkit." + _short] = _sys.modules[_name]
        if "." not in _short:
            globals()[_short] = _sys.modules[_name]

# Shared __path__ so a submodule added later is still importable under either
# name; the loop above is what guarantees identity for everything present now.
__path__ = _ramanujan.__path__

del _importlib, _pkgutil, _sys, _ramanujan