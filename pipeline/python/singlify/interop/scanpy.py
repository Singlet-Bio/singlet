"""
Scanpy convenience helpers for singlify pipeline outputs.

Scanpy operates on :class:`anndata.AnnData`, and :func:`singlify.interop.anndata.read_anndata`
already produces a fully-populated AnnData — so "Scanpy interop" is mostly a
matter of making the AnnData scvelo-ready and providing one-step shortcuts
for the common workflows.

Entry points::

    import singlify as sl
    import scanpy as sc

    adata = sl.interop.scanpy.read(path)          # alias for read_anndata + scvelo-ready
    sl.interop.scanpy.quick_qc(adata)              # filter cells/genes using the preloaded obs columns
    sl.interop.scanpy.ensure_scvelo_layers(adata)  # guarantees layers["spliced"] / ["unspliced"]

None of the heavy lifting happens here — everything delegates to scanpy or
to the data already populated by :func:`read_anndata`.
"""

from __future__ import annotations

import os
from typing import Optional, Union

PathLike = Union[str, bytes, os.PathLike]


def read(
    path: PathLike,
    *,
    primary_assay: str = "spliced",
    scvelo_ready: bool = True,
) -> "anndata.AnnData":  # noqa: F821
    """Read a singlify pipeline directory into an AnnData, scvelo-ready.

    Thin shim over :func:`singlify.interop.anndata.read_anndata` that
    optionally guarantees ``layers["spliced"]`` and ``layers["unspliced"]``
    both exist (scvelo's minimum contract for velocity).
    """
    from .anndata import read_anndata

    adata = read_anndata(path, primary_assay=primary_assay)
    if scvelo_ready:
        ensure_scvelo_layers(adata)
    return adata


def ensure_scvelo_layers(adata: "anndata.AnnData") -> None:  # noqa: F821
    """Ensure ``adata.layers`` contains both ``spliced`` and ``unspliced``.

    scvelo expects both as dedicated layers. :func:`read_anndata` puts
    ``spliced`` into ``.X`` and only mirrors the others into ``.layers``
    for storage efficiency. This helper creates a ``layers["spliced"]``
    entry that aliases ``.X`` so scvelo finds it.

    If neither ``X`` nor ``layers`` contains spliced/unspliced data, this
    is a no-op — the caller probably didn't load a gene-counts-capable
    pipeline output.

    Does not copy — both ``layers["spliced"]`` and ``adata.X`` will reference
    the same underlying sparse buffer.
    """
    primary = adata.uns.get("singlify_primary_assay", "")
    if primary == "spliced" and "spliced" not in adata.layers:
        adata.layers["spliced"] = adata.X
    # unspliced is already in layers by read_anndata when present
    if "unspliced" not in adata.layers:
        # Nothing to do — the pipeline output lacked unspliced counts
        pass


def quick_qc(
    adata: "anndata.AnnData",  # noqa: F821
    *,
    min_genes: int = 200,
    min_cells: int = 3,
    max_mt_pct: float = 20.0,
    min_umis: int = 500,
    inplace: bool = True,
) -> Optional["anndata.AnnData"]:  # noqa: F821
    """One-step QC filter using singlify's pre-loaded per-cell metrics.

    Leverages the columns populated by :func:`read_anndata` from
    ``cell_qc_metrics.tsv`` (``total_umis``, ``total_genes``, ``mt_pct``):

    - Keep cells with at least ``min_umis`` UMIs AND ``min_genes`` genes
      AND ``mt_pct`` below ``max_mt_pct``.
    - Keep genes expressed in at least ``min_cells`` cells.

    Falls back to scanpy's own ``calculate_qc_metrics`` if the sidecar
    columns are not present (e.g. the pipeline output predates
    cell_qc_metrics.tsv).

    Parameters
    ----------
    adata
        Output of :func:`read_anndata`.
    min_genes, min_cells, max_mt_pct, min_umis
        Standard thresholds.
    inplace
        When True, modify ``adata`` and return ``None``. When False, return
        a new filtered AnnData. Default True (matches scanpy convention).
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "singlify.interop.scanpy.quick_qc requires scanpy. "
            "Install with `pip install scanpy`."
        ) from exc
    import numpy as np

    ad = adata if inplace else adata.copy()

    umi_col = "cell_qc_metrics__total_umis"
    gene_col = "cell_qc_metrics__total_genes"
    mt_col = "cell_qc_metrics__mt_pct"

    if umi_col in ad.obs and gene_col in ad.obs and mt_col in ad.obs:
        # Use the pipeline's pre-computed metrics
        umi = np.nan_to_num(ad.obs[umi_col].to_numpy(), nan=0.0)
        genes = np.nan_to_num(ad.obs[gene_col].to_numpy(), nan=0.0)
        mt = np.nan_to_num(ad.obs[mt_col].to_numpy(), nan=100.0)
        cell_mask = (umi >= min_umis) & (genes >= min_genes) & (mt <= max_mt_pct)
        ad._inplace_subset_obs(cell_mask)
    else:
        # Sidecar not present — fall back to scanpy's native metrics
        sc.pp.filter_cells(ad, min_genes=min_genes)
        sc.pp.calculate_qc_metrics(ad, percent_top=None, log1p=False, inplace=True)
        cell_mask = (
            (ad.obs["total_counts"].to_numpy() >= min_umis)
            & (ad.obs.get("pct_counts_mt", 0).to_numpy() <= max_mt_pct)
            if "pct_counts_mt" in ad.obs
            else (ad.obs["total_counts"].to_numpy() >= min_umis)
        )
        ad._inplace_subset_obs(cell_mask)

    sc.pp.filter_genes(ad, min_cells=min_cells)

    return None if inplace else ad


def normalize_log(
    adata: "anndata.AnnData",  # noqa: F821
    *,
    target_sum: float = 1e4,
    inplace: bool = True,
) -> Optional["anndata.AnnData"]:  # noqa: F821
    """Standard depth-normalize + log1p shortcut.

    This is literally ``sc.pp.normalize_total`` + ``sc.pp.log1p`` — it only
    exists so the user can write a single-line preprocessing step against
    a singlify output. Not "interop" in the loose sense, but it makes the
    quickstart workflow clean.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "singlify.interop.scanpy.normalize_log requires scanpy. "
            "Install with `pip install scanpy`."
        ) from exc

    ad = adata if inplace else adata.copy()
    sc.pp.normalize_total(ad, target_sum=target_sum)
    sc.pp.log1p(ad)
    return None if inplace else ad
