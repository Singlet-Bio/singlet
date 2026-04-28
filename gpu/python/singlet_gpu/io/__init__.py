# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.io — I/O helpers for .1pz pipeline outputs.

Public API
----------
read_pz_to_anndata(pz_dir, *, modality, keep_host_pinned) -> AnnData
write_anndata_to_pz(adata, pz_dir, *, modality)           -> None
load_donor_assignments(tsv_path, ...)                      -> pd.Series

Both loader functions live in :mod:`singlet_gpu.io.loader`.
The donor helper lives in :mod:`singlet_gpu.io.donor`.
All are re-exported here for the convenience import
``from singlet_gpu.io import read_pz_to_anndata``.
"""

from singlet_gpu.io.loader import read_pz_to_anndata, write_anndata_to_pz
from singlet_gpu.io.donor import load_donor_assignments

__all__ = ["read_pz_to_anndata", "write_anndata_to_pz", "load_donor_assignments"]
