# SPDX-License-Identifier: MIT
"""System information for bug reports (like pandas.show_versions)."""

from __future__ import annotations

import platform


def show_versions() -> str:
    """Print version information useful for bug reports.

    Returns
    -------
    str
        Formatted version info string (also printed to stdout).

    Examples
    --------
    >>> singlet.show_versions()  # doctest: +SKIP
    singlet: 2.0.0
    Python: 3.12.0
    Platform: Linux-6.1.0-x86_64
    ...
    """
    import singlet

    lines = [
        f"singlet: {singlet.__version__}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        "",
        "Core dependencies:",
    ]

    deps = [
        "numpy",
        "scipy",
        "pandas",
        "anndata",
    ]
    optional = [
        ("sparsepress", "sparsepress"),
        ("torch", "PyTorch"),
        ("cupy", "CuPy (GPU)"),
        ("zarr", "zarr"),
        ("tiledbsoma", "TileDB-SOMA"),
        ("h5py", "h5py"),
        ("requests", "requests"),
        ("hypothesis", "hypothesis"),
    ]

    for pkg in deps:
        lines.append(f"  {pkg}: {_get_version(pkg)}")

    lines.append("")
    lines.append("Optional:")
    for mod_name, label in optional:
        ver = _get_version(mod_name)
        if ver != "not installed":
            lines.append(f"  {label}: {ver}")

    # GPU availability
    lines.append("")
    try:
        import cupy  # type: ignore[import-not-found]

        lines.append(f"GPU: {cupy.cuda.runtime.getDeviceCount()} device(s)")
    except Exception:
        lines.append("GPU: not available")

    result = "\n".join(lines)
    print(result)
    return result


def _get_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "installed (unknown version)")
    except ImportError:
        return "not installed"
