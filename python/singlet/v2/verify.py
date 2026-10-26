# SPDX-License-Identifier: MIT
"""Deprecated: `python -m singlet.v2.verify` is now `python -m singlet.store.verify`."""

import sys

from singlet.store import verify as _m

if __name__ == "__main__":
    sys.exit(_m.main())
sys.modules[__name__] = _m
