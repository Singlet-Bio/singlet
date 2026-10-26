# SPDX-License-Identifier: MIT
"""Deprecated: `python -m singlet.v2.packs_build` is now `python -m singlet.store.packs_build`."""

import sys

from singlet.store import packs_build as _m

if __name__ == "__main__":
    sys.exit(_m.main())
sys.modules[__name__] = _m
