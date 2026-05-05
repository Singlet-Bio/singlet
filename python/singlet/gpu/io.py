# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.io — high-level I/O helpers for .1pz pipeline outputs.

Functions
---------
read_anndata(pz_path, *, modality="exon") -> AnnData
    Load a .1pz file (or directory of .1pz files) and return an AnnData
    whose X matrix is a cupy.sparse.csr_matrix view — zero device→host copy.

to_anndata(device_csc, metadata) -> AnnData
    Wrap an already-loaded DeviceCsc + Metadata into an AnnData.

Both functions populate ``adata.uns['singlify']`` with the embedded GEO
metadata dict (gsm_id, gse_id, protocol, organism, …).

Requirements
------------
anndata ≥ 0.10, cupy-cuda12x ≥ 13.0.  Neither is a hard import at module
level so that tools importing singlet_gpu on non-GPU nodes still work.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Modality → canonical filename stem mapping.
# Mirrors the singlify output artifact table in CLAUDE.md §Full Output Artifacts.
# ---------------------------------------------------------------------------
_MODALITY_TO_STEM = {
    "exon":    "exon_counts",
    "intron":  "intron_counts",
    "gene":    "gene_counts",
    "sj":      "splice_junctions",
    "snp_ad":  "snp_ad",
    "snp_dp":  "snp_dp",
    "mt":      "mt_alleles",
    "adt":     "adt",
    "fragments": "fragments",
}

# Modality → subdirectory (singlify v2 schema moves donor/nonhost into subdirs)
_MODALITY_TO_SUBDIR = {
    "snp_ad": "donor",
    "snp_dp": "donor",
}


def _resolve_path(pz_path: Union[str, Path], modality: str) -> str:
    """
    Resolve a .1pz path from either a direct file path or a sample directory.

    If *pz_path* is a file, it is used directly.
    If it is a directory, we look for ``{stem}.1pz`` matching *modality*.
    Falls back to subdirectory layout (singlify v2 schema).
    """
    p = Path(pz_path)
    if p.is_file():
        return str(p)
    if p.is_dir():
        stem = _MODALITY_TO_STEM.get(modality)
        if stem is None:
            raise ValueError(
                f"Unknown modality '{modality}'. "
                f"Recognised values: {sorted(_MODALITY_TO_STEM)}"
            )
        candidate = p / f"{stem}.1pz"
        if not candidate.exists():
            subdir = _MODALITY_TO_SUBDIR.get(modality)
            if subdir:
                candidate = p / subdir / f"{stem}.1pz"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Expected {stem}.1pz for modality='{modality}' "
                f"but file not found in {p}"
            )
        return str(candidate)
    raise FileNotFoundError(f"Path does not exist: {pz_path}")


def to_anndata(device_csc, metadata):
    """
    Wrap a :class:`~singlet.gpu.DeviceCsc` and :class:`~singlet.gpu.Metadata`
    into an AnnData whose ``X`` is a zero-copy ``cupy.sparse.csr_matrix`` view.

    Parameters
    ----------
    device_csc : singlet.gpu.DeviceCsc
        Device CSC matrix (rows = genes, cols = cells).
    metadata : singlet.gpu.Metadata
        Embedded GEO metadata from the .1pz file.

    Returns
    -------
    anndata.AnnData
        ``adata.X``  — cupy.sparse.csr_matrix (cells × genes), zero-copy.
        ``adata.obs`` — DataFrame indexed by cell barcodes (from metadata.colnames).
        ``adata.var`` — DataFrame indexed by gene names (from metadata.rownames).
        ``adata.uns['singlify']`` — dict of all GEO + provenance fields.
    """
    try:
        import anndata as ad
        import pandas as pd
        import cupy as cp
        # cupy >= 14 removed `cupy.sparse`; the new home is `cupyx.scipy.sparse`
        # which exposes the same csc_matrix / csr_matrix API.
        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp         # cupy < 14 fallback
    except ImportError as e:
        raise ImportError(
            "to_anndata requires anndata, cupy, and pandas. "
            f"Install with: pip install 'singlet[gpu]'\n"
            f"Original error: {e}"
        ) from e

    rows = device_csc.rows
    cols = device_csc.cols

    # Build zero-copy cupy views.
    # WHY .T (transpose): .1pz stores the matrix as (genes × cells) CSC.
    # AnnData convention is (cells × genes) with X in CSR layout.
    # cupy >= 14 dtype-strictness: cp.asarray() no longer accepts a plain dict;
    # the object must expose __cuda_array_interface__ as an *attribute*.
    # Wrap each dict in a minimal shim so both cupy 13 and 14 work.  Zero copy.
    class _CaiView:  # lightweight shim — not on the hot path
        def __init__(self, d): self.__cuda_array_interface__ = d

    cu_data    = cp.asarray(_CaiView(device_csc.data_view))
    cu_indices = cp.asarray(_CaiView(device_csc.indices_view))
    cu_indptr  = cp.asarray(_CaiView(device_csc.indptr_view))

    # (genes × cells) csc → (cells × genes) csr via transpose (no data copy).
    genes_x_cells_csc = csp.csc_matrix(
        (cu_data, cu_indices, cu_indptr), shape=(rows, cols)
    )
    cells_x_genes_csr = genes_x_cells_csc.T.tocsr()

    # obs (cells) and var (genes) DataFrames.
    obs_names = list(metadata.colnames) if metadata.colnames else [str(i) for i in range(cols)]
    var_names = list(metadata.rownames) if metadata.rownames else [str(i) for i in range(rows)]

    obs = pd.DataFrame(index=obs_names)
    var = pd.DataFrame(index=var_names)

    # uns['singlify'] — GEO + provenance dict.
    singlify_meta = dict(metadata.to_dict())
    # Also include rownames and colnames counts for convenience.
    singlify_meta["n_genes"] = rows
    singlify_meta["n_cells"] = cols

    adata = ad.AnnData(
        X=cells_x_genes_csr,
        obs=obs,
        var=var,
        uns={"singlify": singlify_meta},
    )

    # Keep a reference to the DeviceCsc alive via obsm so Python's GC
    # does not collect the device allocation while adata.X is in use.
    # We store the DeviceCsc object in a private slot — AnnData tolerates
    # arbitrary Python objects in uns.
    adata.uns["_singlet_gpu_device_ref"] = device_csc

    return adata


def read_anndata(
    pz_path: Union[str, Path],
    *,
    modality: str = "exon",
    keep_host_pinned: bool = False,
) -> "anndata.AnnData":
    """
    Load a ``.1pz`` file (or sample directory) and return an AnnData.

    ``adata.X`` is a ``cupy.sparse.csr_matrix`` view over the device buffer —
    no data is copied to the host.

    Parameters
    ----------
    pz_path : str or Path
        Either a direct path to a ``.1pz`` file, or a sample directory
        containing ``{modality_stem}.1pz`` (e.g. ``exon_counts.1pz``).
    modality : str, default "exon"
        Which modality to load when *pz_path* is a directory.
        One of: ``"exon"``, ``"intron"``, ``"gene"``, ``"sj"``,
        ``"snp_ad"``, ``"snp_dp"``, ``"mt"``, ``"adt"``, ``"fragments"``.
    keep_host_pinned : bool, default False
        Passed through to :func:`~singlet.gpu.load_pz`.  When ``True``,
        retains pinned host copies — useful for SVD adapters.

    Returns
    -------
    anndata.AnnData
        ``adata.X``  — cupy.sparse.csr_matrix (cells × genes), zero-copy device view.
        ``adata.obs`` — cell barcodes from metadata (if embedded).
        ``adata.var`` — gene names from metadata (if embedded).
        ``adata.uns['singlify']`` — GEO + provenance dict.

    Examples
    --------
    Load by file path::

        adata = singlet.gpu.io.read_anndata("path/to/exon_counts.1pz")

    Load by sample directory (auto-detects exon_counts.1pz)::

        adata = singlet.gpu.io.read_anndata("quant/scrna/GSE127/GSE127918/GSM4037629/")

    Load unspliced counts::

        adata_unspliced = singlet.gpu.io.read_anndata(sample_dir, modality="intron")

    Access metadata::

        print(adata.uns['singlify']['gsm_id'])   # "GSM4037629"
    """
    # Import here (not at module top) so the module can be imported on login nodes.
    try:
        from singlet.gpu import load_pz as _load_pz
    except (ImportError, TypeError):
        raise ImportError(
            "singlet.gpu._core is not available. "
            "The C++ extension must be compiled on a CUDA-capable node. "
            "Install with: pip install singlet[gpu]"
        )

    path = _resolve_path(pz_path, modality)
    result = _load_pz(path, keep_host_pinned=keep_host_pinned)

    return to_anndata(result.mat, result.meta)


__all__ = ["read_anndata", "to_anndata"]
