# SPDX-License-Identifier: MIT
"""
singlet.gpu._coreutil — shared helper for importing the compiled ``_core`` extension.

``singlet.gpu._core`` is the pybind11 / CUDA extension built by CMake.  It is
absent on nodes without a CUDA toolchain (e.g. login nodes, doc builders).
Every GPU wrapper module needs the same "import ``_core``, fail loudly with
install guidance if missing, optionally check that a specific kernel binding
is present" logic — ``require_core`` consolidates it in one place.
"""

_INSTALL_HINT = (
    "The C++ extension must be compiled on a CUDA-capable node before use.  "
    "Run: pip install -e python/ on a GPU node."
)


def require_core(*attrs):
    """
    Import and return the compiled ``singlet.gpu._core`` extension.

    Parameters
    ----------
    *attrs : str
        Optional attribute path to validate on ``_core`` (e.g.
        ``require_core("fate", "cospar")`` checks ``_core.fate.cospar``
        exists).  Each level is verified in turn.

    Returns
    -------
    module / object
        With no ``attrs``, returns the ``_core`` module itself.  With one or
        more ``attrs``, returns the object at the *parent* of the last named
        attribute — i.e. ``require_core("fate", "cospar")`` returns
        ``_core.fate`` and ``require_core("run_daesc")`` returns ``_core``.

    Raises
    ------
    ImportError
        If ``singlet.gpu._core`` cannot be imported, or if any element of
        ``attrs`` is missing from the extension.
    """
    try:
        import singlet.gpu._core as _core
    except ImportError as e:
        raise ImportError(
            f"singlet.gpu._core could not be imported ({e}).  {_INSTALL_HINT}"
        ) from e

    obj = _core
    parent = _core
    walked = []
    for name in attrs:
        if not hasattr(obj, name):
            path = "_core" + "".join(f".{w}" for w in walked) + f".{name}"
            raise ImportError(
                f"{path} is not available — the compiled binding is missing.  "
                f"{_INSTALL_HINT}"
            )
        parent = obj
        obj = getattr(obj, name)
        walked.append(name)

    return parent if attrs else _core
