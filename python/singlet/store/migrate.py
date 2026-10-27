# SPDX-License-Identifier: MIT
"""Deprecated: `python -m singlet.store.migrate` is now `python -m singlet.s1.migrate`."""

import sys

from singlet.s1 import migrate as _m

if __name__ == "__main__":
    sys.exit(_m.main())
sys.modules[__name__] = _m
