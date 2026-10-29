# SPDX-License-Identifier: MIT
"""Deprecated alias of :mod:`singlet.s1` (the per-sample format used to be called "v2").

DUAL (transition from the old name; removed in a later cleanup): ``import singlet.v2`` and ``from singlet.v2 import cellcall`` keep
working and give the very same module objects as ``singlet.s1`` / ``singlet.s1.cellcall``.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import singlet.s1 as _store  # noqa: F401  (imported for its side effect: the package must exist before the alias hook)
from singlet.s1 import *  # noqa: F401,F403
from singlet.s1 import FORMAT  # noqa: F401

warnings.warn(
    "singlet.v2 is the old name of singlet.s1 and will be removed; import singlet.s1 instead",
    DeprecationWarning,
    stacklevel=2,
)

_PREFIX = __name__ + "."
_SAVED_SPECS: dict = {}


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """``singlet.v2.<sub>`` -> the module object ``singlet.s1.<sub>`` (no copy)."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(_PREFIX):
            try:
                if importlib.util.find_spec("singlet.s1." + fullname[len(_PREFIX) :]) is None:
                    return None
            except (ImportError, ValueError):
                return None
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        mod = importlib.import_module("singlet.s1." + spec.name[len(_PREFIX) :])
        _SAVED_SPECS[mod.__name__] = (mod, mod.__spec__)
        return mod

    def exec_module(self, module):
        mod, spec = _SAVED_SPECS.pop(module.__name__, (None, None))
        if (
            mod is not None
        ):  # the import machinery overwrote __spec__ with the alias spec: put the real one back
            mod.__spec__ = spec


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.append(_AliasFinder())
