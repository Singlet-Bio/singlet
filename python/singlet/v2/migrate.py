# SPDX-License-Identifier: MIT
"""Deprecated: `python -m singlet.v2.migrate` is now `python -m singlet.store.migrate`."""
import sys

from singlet.store import migrate as _m

if __name__ == "__main__":
    sys.exit(_m.main())
sys.modules[__name__] = _m
