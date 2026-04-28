# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.io — high-level I/O helpers for .1pz pipeline outputs.

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


def _resolve_path(pz_path: Union[str, Path], modality: str) -> str:
    """
    Resolve a .1pz path from either a direct file path or a sample directory.

    If *pz_path* is a file, it is used directly.
    If it is a directory, we look for ``{stem}.1pz`` matching *modality*.
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
            raise FileNotFoundError(
                f"Expected {candidate} for modality='{modality}' "
                f"but file not found in {p}"
            )
        return str(candidate)
    raise FileNotFoundError(f"Path does not exist: {pz_path}")


def to_anndata(device_csc, metadata):
    """
    Wrap a :class:`~singlet_gpu.DeviceCsc` and :class:`~singlet_gpu.Metadata`
    into an AnnData whose ``X`` is a zero-copy ``cupy.sparse.csr_matrix`` view.

    Parameters
    ----------
    device_csc : singlet_gpu.DeviceCsc
        Device CSC matrix (rows = genes, cols = cells).
    metadata : singlet_gpu.Metadata
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
        import cupy.sparse as csp
        import cupy as cp
    except ImportError as e:
        raise ImportError(
            "to_anndata requires anndata, cupy, and pandas. "
            f"Install with: pip install 'singlet-gpu[anndata]' cupy-cuda12x\n"
            f"Original error: {e}"
        ) from e

    rows = device_csc.rows
    cols = device_csc.cols

    # Build zero-copy cupy views.
    # The cupy.ndarray constructors accept an __cuda_array_interface__ dict
    # directly as the first argument (cupy v9+).
    # We construct a csc_matrix from the CSC layout, then convert to csr
    # because AnnData conventionally stores X as (cells × genes) csr.
    #
    # WHY .T (transpose): .1pz stores the matrix as (genes × cells) CSC.
    # AnnData convention is (cells × genes) with X in CSR layout.
    # cupy.sparse.csc_matrix.T returns a csr_matrix without copying data.

    data_view    = device_csc.data_view     # dict with __cuda_array_interface__
    indices_view = device_csc.indices_view
    indptr_view  = device_csc.indptr_view

    # cupy.asarray reads __cuda_array_interface__ — zero copy.
    cu_data    = cp.asarray(data_view)
    cu_indices = cp.asarray(indices_view)
    cu_indptr  = cp.asarray(indptr_view)

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
        Passed through to :func:`~singlet_gpu.load_pz`.  When ``True``,
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

        adata = singlet_gpu.io.read_anndata("path/to/exon_counts.1pz")

    Load by sample directory (auto-detects exon_counts.1pz)::

        adata = singlet_gpu.io.read_anndata("quant/scrna/GSE127/GSE127918/GSM4037629/")

    Load unspliced counts::

        adata_unspliced = singlet_gpu.io.read_anndata(sample_dir, modality="intron")

    Access metadata::

        print(adata.uns['singlify']['gsm_id'])   # "GSM4037629"
    """
    # Import here (not at module top) so the module can be imported on login nodes.
    try:
        from singlet_gpu import load_pz as _load_pz
    except (ImportError, TypeError):
        raise ImportError(
            "singlet_gpu._core is not available. "
            "The C++ extension must be compiled on a CUDA-capable node. "
            "Run: pip install -e singlet-gpu/python/ on a GPU node."
        )

    path = _resolve_path(pz_path, modality)
    result = _load_pz(path, keep_host_pinned=keep_host_pinned)

    return to_anndata(result.mat, result.meta)


__all__ = ["read_anndata", "to_anndata"]
