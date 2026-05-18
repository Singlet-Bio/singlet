# SPDX-License-Identifier: MIT
"""
singlet.gpu.io — I/O helpers for .1pz pipeline outputs.

Public API
----------
read_pz_to_anndata(pz_dir, *, modality, keep_host_pinned) -> AnnData
write_anndata_to_pz(adata, pz_dir, *, modality)           -> None
load_donor_assignments(tsv_path, ...)                      -> pd.Series

Both loader functions live in :mod:`singlet.gpu.io.loader`.
The donor helper lives in :mod:`singlet.gpu.io.donor`.
All are re-exported here for the convenience import
``from singlet.gpu.io import read_pz_to_anndata``.
"""

from .loader import read_pz_to_anndata, write_anndata_to_pz
from .donor import load_donor_assignments

# Re-export the C++ load_pz primitive so callers can do
# `singlet.gpu.io.load_pz(path)` without reaching into _core directly.
# Also exported at the package root via singlet.gpu.__init__.
try:
    from singlet.gpu._core import load_pz
except ImportError:
    load_pz = None  # _core not built (e.g. tooling env)

__all__ = ["read_pz_to_anndata", "write_anndata_to_pz",
           "load_donor_assignments", "load_pz"]
