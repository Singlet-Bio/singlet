# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath("../python"))

project = "singlet"
author = "Zach DeBruine"
copyright = "2026, Zach DeBruine"

try:
    release = importlib.metadata.version("singlet")
except importlib.metadata.PackageNotFoundError:
    release = "1.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# singlet's public API is intentionally implemented in private submodules
# (singlet/_foo.py) and re-exported from singlet/__init__.py. autosummary
# needs to follow those re-exports and generate stub pages for imported
# members, not just ones defined directly in the module being documented.
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "imported-members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
# Group members in source order (the categorized order singlet/__init__.py
# already imports them in) rather than resorting alphabetically.
autodoc_member_order = "bysource"

# GPU (`singlet.gpu`), PyTorch (`singlet.torch`) and MCP (`singlet.mcp`)
# are opt-in extras (`pip install singlet[gpu|torch|mcp]`) and are not
# imported by `import singlet` itself, so the docs build's base install
# doesn't have them. Individual analysis functions in the base package also
# lazily import third-party algorithm packages (leidenalg, umap-learn,
# harmonypy, phate, palantir, ...) only when called, with a pure-Python or
# scipy fallback otherwise -- so none of those need mocking here either.
#
# Deliberately NOT using autodoc_mock_imports for torch/cupy/mcp/etc: Sphinx's
# mock objects respond to attribute access by returning further mocks instead
# of raising, which breaks packages that probe for optional deps with a real
# try/except ImportError (anndata does this for cupy) and that register
# singledispatch implementations against the probed class
# (`anndata/utils.py`'s `@asarray.register(CupyArray)` calls
# `functools.singledispatch.register`, which -- because a Sphinx mock object
# is not an instance of `type` -- falls into the type-hint-inspection code
# path and recurses through the mock's `__getattr__` indefinitely, hanging
# the build). Leaving these unmocked lets `import cupy` fail with a genuine
# ModuleNotFoundError, which anndata already handles correctly by defining
# safe dummy classes. `singlet` itself never imports torch/cupy/mcp/zarr/
# tiledbsoma unconditionally (only inside TYPE_CHECKING or lazily inside
# function bodies), so the base docs install doesn't need any of them mocked.
autodoc_mock_imports = []

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "singlet"
html_show_sourcelink = False
html_theme_options = {
    "source_repository": "https://github.com/Singlet-Bio/singlet",
    "source_branch": "main",
    "source_directory": "docs-python/",
}
