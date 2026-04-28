"""
AnnData adapter for singlify pipeline outputs.

Public entry points::

    from singlify.interop.anndata import read_anndata

    adata = read_anndata("quant/scrna/GSE174/GSE174399/GSM5293863")

Layout of the returned :class:`anndata.AnnData`:

- ``adata.X`` — the primary per-gene count matrix (spliced by default;
  fall back to gene_counts then gene_counts_em if spliced is unavailable).
  Oriented as ``cells × genes`` per AnnData convention. dtype matches the
  source ``.1pz`` (uint8/uint16/uint32 via ``vt_code``).
- ``adata.layers["spliced"]`` / ``["unspliced"]`` / ``["ambiguous"]`` —
  the velocity trio, if present. Each layer is ``cells × genes``.
- ``adata.layers["gene_counts_em"]`` — EM-rescued multi-mapper counts,
  if present.
- ``adata.obs_names`` — cell barcodes from the matrix's colnames.
- ``adata.var_names`` — gene IDs from the matrix's rownames.
- ``adata.var["feature_id"]`` — same as var_names (kept for explicit access).
- ``adata.uns["singlify"]`` — the ``user_kv`` dict from the pipeline
  (``gsm_id``, ``gse_id``, ``organism``, ``protocol``, ``singlify_version``,
  ``pipeline_date``, ``srr_ids``, ``read_count``).
- ``adata.obsm["exon_counts"]`` — per-exon matrix (cells × exons) if
  ``exon_counts.1pz`` is present. Has a different feature axis from
  ``adata.X``, so it lives in ``.obsm``.
- ``adata.obsm["intron_counts"]`` — per-intron matrix (cells × introns).
- ``adata.obsm["sj_counts"]`` — per-junction matrix.
- ``adata.obsm["mt_heteroplasmy"]`` — per-mt-site matrix.
- ``adata.obsm["vdj_gene_usage"]`` — per-VDJ-gene matrix.

Per-cell sidecars (``cell_qc_metrics.tsv``, ``cell_cycle_scores.tsv``,
``doublet_scores.tsv``, ``read_stats.tsv``, ``saturation_curve.tsv``) are
loaded into ``adata.obs`` when present.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Dict, Optional, Union

from ..io._reader import PipelineDirectory, read_dir

PathLike = Union[str, bytes, os.PathLike]


def _load_cell_sidecar_tsv(
    dir_path: pathlib.Path, filename: str, barcodes: list
) -> Dict[str, Any]:
    """Load a per-cell TSV, aligning its rows to the matrix's barcode order.

    Returns a dict of column name -> aligned 1D array (NaN for missing
    barcodes). Returns an empty dict if the file is not present.
    """
    import numpy as np
    import pandas as pd

    p = dir_path / filename
    if not p.is_file():
        return {}
    try:
        df = pd.read_csv(p, sep="\t")
    except Exception:
        return {}
    if df.empty:
        return {}

    # Detect the barcode column. Common names used by singlify sidecars.
    bc_col = None
    for c in ("barcode", "cell_barcode", "cell", "cell_id", "CB"):
        if c in df.columns:
            bc_col = c
            break
    if bc_col is None:
        # Fall back to the first column if it looks stringish
        first = df.columns[0]
        if df[first].dtype == object:
            bc_col = first
    if bc_col is None:
        return {}

    df = df.set_index(bc_col)
    # Reindex to the matrix barcode order (produces NaN for missing barcodes).
    df = df.reindex(barcodes)

    # Coerce object columns to a writable dtype. After reindex, columns that
    # were originally bool (e.g. doublet_scores.is_doublet) become object
    # dtype if any barcode is missing — h5py can't write those, so cast them
    # back to bool/int with sane defaults for missing rows.
    out: Dict[str, Any] = {}
    for col in df.columns:
        ser = df[col]
        if ser.dtype == object:
            # Try numeric first
            coerced = pd.to_numeric(ser, errors="coerce")
            if coerced.notna().any():
                out[col] = coerced.to_numpy()
            else:
                # Fall back to string; replace NaN with empty string so h5py
                # can write a vlen str dataset.
                out[col] = ser.astype("string").fillna("").to_numpy()
        else:
            out[col] = ser.to_numpy()
    return out


def read_anndata(
    path: PathLike,
    *,
    primary_assay: str = "spliced",
) -> "anndata.AnnData":  # noqa: F821  (forward ref — import is lazy)
    """Read a singlify pipeline output directory into an :class:`AnnData`.

    Parameters
    ----------
    path
        Directory containing the pipeline's ``.1pz`` + sidecar files.
    primary_assay
        Which per-gene matrix to use for ``adata.X``. One of:
        ``"spliced"`` (default), ``"gene_counts"``, ``"gene_counts_em"``.
        If the requested assay is not present, falls back in order
        ``spliced → gene_counts → gene_counts_em``.

    Returns
    -------
    anndata.AnnData
        A fully-populated AnnData — cells × genes, with the velocity trio
        in ``layers``, per-feature matrices in ``obsm``, per-cell sidecars
        in ``obs``, and the GEO context in ``uns["singlify"]``.

    Raises
    ------
    ImportError
        If the ``anndata`` package is not installed. Install with
        ``pip install anndata``.
    ValueError
        If the directory contains no usable per-gene count matrix.
    """
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "singlify.interop.anndata requires the anndata package. "
            "Install with `pip install anndata`."
        ) from exc

    import pandas as pd

    dd = read_dir(path)
    dir_path = dd.path

    # Pick the primary per-gene assay
    per_gene_candidates = [primary_assay, "spliced", "gene_counts", "gene_counts_em"]
    chosen: Optional[str] = None
    for name in per_gene_candidates:
        if name in dd:
            chosen = name
            break
    if chosen is None:
        raise ValueError(
            f"no per-gene count matrix in {dir_path} — "
            f"none of {per_gene_candidates} found"
        )

    X = dd[chosen].T.tocsr()  # cells × genes for AnnData
    n_cells, n_genes = X.shape

    # Use the metadata from the chosen matrix — read_dir's meta uses the
    # first-found user_kv block and every matrix embeds the same one.
    rownames = dd.metadata.rownames
    colnames = dd.metadata.colnames
    if not rownames or len(rownames) != n_genes:
        # Fall back to freshly-loaded metadata from the chosen file
        from ..io._reader import read_matrix as _read_matrix
        _, m = _read_matrix(dir_path / f"{chosen}.1pz")
        rownames = m.rownames
        colnames = m.colnames

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.Index(colnames, name="barcode")),
        var=pd.DataFrame(
            {"feature_id": rownames},
            index=pd.Index(rownames, name="feature_id"),
        ),
    )

    # Velocity trio layers (must match X's (cells × genes) shape).
    # If a per-gene matrix is missing but its per-feature counterpart is
    # present (e.g., spliced absent but exon_counts present), reconstruct
    # the per-gene matrix on-the-fly using aggregate_features_to_gene.
    # This makes the AnnData layers complete even when the pipeline drops
    # spliced.1pz / unspliced.1pz to save NFS storage.
    derive_from = {
        "spliced": ("exon_counts", "exon_counts"),
        "unspliced": ("intron_counts", "intron_counts"),
    }
    for layer_name in ("spliced", "unspliced", "ambiguous", "gene_counts_em"):
        if layer_name == chosen:
            continue
        if layer_name in dd:
            mat = dd[layer_name]
            if mat.shape == dd[chosen].shape:
                adata.layers[layer_name] = mat.T.tocsr()
        elif layer_name in derive_from:
            # Try to reconstruct from the per-feature matrix
            source_key, _ = derive_from[layer_name]
            if source_key in dd:
                from ..io._reader import aggregate_features_to_gene
                src_rownames = dd.matrix_rownames.get(source_key, [])
                if src_rownames:
                    aggregated, _ = aggregate_features_to_gene(
                        dd[source_key], src_rownames
                    )
                    if aggregated.shape == dd[chosen].shape:
                        adata.layers[layer_name] = aggregated.T.tocsr()
                        adata.uns.setdefault(
                            "singlify_derived_layers", []
                        ).append(f"{layer_name} (from {source_key})")

    # Per-feature matrices (different feature axis from genes) → obsm
    for obsm_name in (
        "exon_counts", "intron_counts", "sj_counts",
        "splice_psi", "mt_heteroplasmy", "vdj_gene_usage",
    ):
        if obsm_name in dd:
            mat = dd[obsm_name]
            if mat.shape[1] == n_cells:
                adata.obsm[obsm_name] = mat.T.tocsr()

    # Per-cell sidecar TSVs → obs columns
    for sidecar in (
        "cell_qc_metrics.tsv",
        "cell_cycle_scores.tsv",
        "doublet_scores.tsv",
        "read_stats.tsv",
        "ambient_contamination.tsv",
    ):
        cols = _load_cell_sidecar_tsv(dir_path, sidecar, list(colnames))
        for col_name, values in cols.items():
            obs_col = f"{sidecar.replace('.tsv','')}__{col_name}"
            adata.obs[obs_col] = values

    # GEO context + provenance → uns
    adata.uns["singlify"] = dict(dd.metadata.user_kv)
    adata.uns["singlify_source_path"] = str(dir_path)
    adata.uns["singlify_primary_assay"] = chosen

    return adata
