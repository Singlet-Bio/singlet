"""
Module entry point — makes ``python -m singlify ...`` work.

This is a fallback for users who don't have the ``singlify`` script on
their PATH. The ``[project.scripts]`` entry in ``pyproject.toml``
provides the preferred ``singlify info ...`` invocation when the
package is installed normally; this module-form is a portable backup
that always works whether the package is editable-installed, wheel-
installed, or running straight from the source tree.

Usage::

    python -m singlify info path/to/sample.1pz
    python -m singlify verify path/to/sample.1pz
    python -m singlify convert path/to/sample/ -o sample.h5ad
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
