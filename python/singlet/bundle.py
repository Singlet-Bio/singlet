# SPDX-License-Identifier: MIT
"""singlet.bundle — Per-GSE .singlet bundle format (ZIP64).

A .singlet file is a ZIP64 archive containing ALL processed GSMs for one GEO
Series (GSE). It is the primary distribution unit for the Singlet atlas.

Bundle layout::

    <GSE>.singlet          (ZIP64 archive)
    ├── manifest.json      schema_version, gse_id, gsm list, per-file checksums,
    │                      reference build, created_at, singlet/pipeline versions
    ├── study_meta.json    series title/abstract + per-GSM obs metadata
    │                      (tissue/donor/disease/sex/protocol/n_cells/qc_flag)
    ├── feature_vocab.json gene ↔ Ensembl map + reference build (shared across GSMs)
    └── samples/
        └── <GSM>/
            ├── exon_counts.1pz      (stored uncompressed — already zstd inside)
            ├── intron_counts.1pz    (stored uncompressed)
            ├── sj_counts.1pz        (stored uncompressed, if present)
            ├── splice_psi.1pz       (stored uncompressed, if present)
            ├── vdj_gene_usage.1pz   (stored uncompressed, if present)
            ├── mt_heteroplasmy.1pz  (stored uncompressed, if present)
            ├── summary.json         (DEFLATED)
            ├── pileup_stats.json    (DEFLATED)
            └── provenance.json      (DEFLATED)

Design decisions:
- .1pz files stored STORED (no extra compression); they contain zstd frames.
- JSON sidecar files stored DEFLATED (good ratio, tiny overhead).
- feature_vocab.json is shared across all GSMs in the bundle; callers must not
  assume gene order is the same as the pipeline's raw row order — use
  feature_vocab.json as the canonical var axis.
- manifest.json includes SHA-256 checksums of all member files for integrity.
- study_meta.json is populated from: processing catalog (organism/protocol/
  sample_characteristics/series_title/summary), summary.json (n_cells,
  reference_build, protocol_name), and the catalog's qc_flag column.

Public API
----------
pack_gse(gse_id, results_dir, catalog_path, out_path)
    Build a .singlet bundle from pipeline results.

SingletBundle.open(path)
    Open a bundle for reading.

SingletBundle.to_anndata(layer="gene_counts")
    Concatenate all GSMs into one AnnData.

SingletBundle.to_h5ad(path)
    Write concatenated AnnData to .h5ad.

SingletBundle.to_parquet(path)
    Write obs DataFrame (metadata only, no counts) to Parquet.

CLI (``singlet-pack``)::

    python -m singlet.bundle pack --gse GSE122083 \\
        --results /path/to/results --catalog /path/to/catalog.parquet \\
        --out /path/to/out/GSE122083.singlet

Notes on large GSEs
-------------------
For GSEs with hundreds of GSMs and millions of cells, the bundle can exceed
10–50 GB. The packager streams each file without buffering the whole archive in
memory (``zipfile.ZipFile`` in Python writes incrementally). The reader loads
count matrices on demand (lazy). For very large bundles (>50 GSMs), prefer
to_h5ad() which streams each GSM and concatenates with AnnData's incremental
concat rather than loading all in RAM simultaneously.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np

__all__ = [
    "pack_gse",
    "SingletBundle",
    "Modality",
    "MODALITIES",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"
_BUNDLE_EXTENSION = ".singlet"

# Files inside each samples/<GSM>/ that are always included when present
_PZ_FILES = [
    "exon_counts.1pz",
    "intron_counts.1pz",
    "sj_counts.1pz",
    "splice_psi.1pz",
    "vdj_gene_usage.1pz",
    "mt_heteroplasmy.1pz",
]

_JSON_SIDECARS = [
    "summary.json",
    "pileup_stats.json",
    "provenance.json",
]

# TSV sidecars — DEFLATED (small, text)
_TSV_SIDECARS = [
    "cell_calls.tsv",
]


# ---------------------------------------------------------------------------
# Modality registry
# ---------------------------------------------------------------------------
# Every non-gene-expression output the pipeline can emit for a sample, keyed by
# the short name used in :meth:`SingletBundle.modalities` and
# :meth:`SingletBundle.read`. ``member`` lists candidate archive paths inside
# ``samples/<GSM>/`` in priority order (layouts changed across pipeline
# versions). ``kind`` selects the reader: ``matrix`` → .1pz sparse matrix,
# ``table`` → TSV, ``json`` → JSON, ``text`` → raw bytes decoded as UTF-8.


class Modality(NamedTuple):
    """One addressable per-sample output inside a ``.singlet`` bundle."""

    name: str
    kind: str
    members: tuple
    description: str


MODALITIES: Dict[str, Modality] = {
    m.name: m
    for m in (
        # ── Counts ────────────────────────────────────────────────────────
        Modality("exon_counts", "matrix", ("exon_counts.1pz",),
                 "Per-exon-feature UMI counts (the spliced half of the raw matrix)."),
        Modality("intron_counts", "matrix", ("intron_counts.1pz",),
                 "Per-intron-feature UMI counts (the unspliced half of the raw matrix)."),
        Modality("cell_calls", "table", ("cell_calls.tsv",),
                 "Barcode, is_cell and the cell-calling statistics per barcode."),
        Modality("cell_qc", "table", ("cell_qc_metrics.tsv",),
                 "Per-cell QC metrics (UMIs, genes, mitochondrial fraction, …)."),
        # ── Splicing ──────────────────────────────────────────────────────
        Modality("junctions", "matrix", ("sj_counts.1pz",),
                 "Per-cell splice-junction counts."),
        Modality("splice_psi", "matrix", ("splice_psi.1pz",),
                 "Per-cell percent-spliced-in (PSI) per splice event."),
        Modality("splice_events", "table", ("splice_events.tsv",),
                 "Splice-event annotation for the rows of splice_psi."),
        # ── Mitochondrial genome ──────────────────────────────────────────
        Modality("mt_heteroplasmy", "matrix", ("mt_heteroplasmy.1pz",),
                 "Per-cell mitochondrial heteroplasmy (VAF) per chrM variant site."),
        Modality("mt_variants", "table", ("mt_variants.tsv", "mt/mt_summary.tsv"),
                 "Called chrM variants with depth, allele counts and annotation."),
        Modality("mt_events", "matrix", ("mt/mt_events.1pz",),
                 "Per-cell chrM allele-support matrix used for lineage tracing."),
        # ── Donor / genotype ──────────────────────────────────────────────
        Modality("donor_assignments", "table",
                 ("donor/donor_assignments.tsv", "donor_assignments.tsv"),
                 "Genotype-free donor demultiplexing: barcode → donor, with doublet calls."),
        Modality("donor_snp_ad", "matrix", ("donor/snp_ad.1pz",),
                 "Per-cell alternate-allele depth over the SNP panel."),
        Modality("donor_snp_dp", "matrix", ("donor/snp_dp.1pz",),
                 "Per-cell total read depth over the SNP panel (denominator for snp_ad)."),
        Modality("ase_counts", "table", ("ase_counts.tsv",),
                 "Allele-specific expression counts per gene."),
        Modality("ancestry_call", "json", ("ancestry_call.json",),
                 "Continental ancestry estimate from the SNP panel."),
        Modality("sex_call", "json", ("sex_call.json",),
                 "Genetic sex call from chrX/chrY expression and coverage."),
        # ── Non-host ──────────────────────────────────────────────────────
        Modality("nonhost_species", "table",
                 ("nonhost/nonhost_em_abundance.tsv", "nonhost_em_abundance.tsv"),
                 "Per-taxon non-host (microbial/viral) abundance after EM re-assignment."),
        Modality("nonhost_summary", "json",
                 ("nonhost/nonhost_summary.json", "nonhost_summary.json"),
                 "Non-host classification summary: reads classified, top taxa, database."),
        # ── Immune repertoire ─────────────────────────────────────────────
        Modality("vdj_gene_usage", "matrix", ("vdj_gene_usage.1pz",),
                 "Per-cell V(D)J segment usage counts."),
        # ── Per-cell annotations ──────────────────────────────────────────
        Modality("doublet_scores", "table", ("doublet_scores.tsv",),
                 "Per-cell doublet score and call."),
        Modality("cell_cycle_scores", "table", ("cell_cycle_scores.tsv",),
                 "Per-cell S/G2M scores and phase assignment."),
        Modality("ambient_contamination", "table", ("ambient_contamination.tsv",),
                 "Per-cell ambient-RNA contamination fraction."),
        Modality("ambient_profile", "table", ("ambient_profile.tsv",),
                 "Ambient-RNA expression profile estimated from empty droplets."),
        # ── Sample-level QC ───────────────────────────────────────────────
        Modality("summary", "json", ("summary.json",),
                 "Sample-level metrics: cells called, mapping rate, medians, reference build."),
        Modality("pileup_stats", "json", ("pileup_stats.json",),
                 "Alignment/pileup statistics for the sample."),
        Modality("provenance", "json", ("provenance.json",),
                 "Pipeline version, command line, reference checksums."),
        Modality("saturation_curve", "table", ("saturation_curve.tsv",),
                 "Sequencing-saturation curve (downsampled read depth vs genes detected)."),
        Modality("star_log", "text", ("star_Log.final.out",),
                 "STAR final alignment log for the sample."),
    )
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_kind(member: str) -> str:
    """Reader kind for a literal archive member that is not in MODALITIES."""
    if member.endswith(".1pz"):
        return "matrix"
    if member.endswith(".json"):
        return "json"
    if member.endswith((".tsv", ".csv")):
        return "table"
    return "text"


def _union_feature_axis(exon, intron):
    """Stack exon and intron AnnData on a shared cell axis with both layers.

    Feature rows are disjoint between the two matrices (exons vs introns), so
    the union axis is a concatenation. ``X`` is the sum, which equals each
    layer on its own half of the axis.
    """
    import anndata as ad
    import numpy as _np
    import pandas as pd
    from scipy.sparse import csr_matrix, hstack

    present = [a for a in (exon, intron) if a is not None]
    if not present:
        raise ValueError("neither exon nor intron counts are available")
    barcodes = list(map(str, present[0].obs_names))
    for a in present[1:]:
        keep = set(map(str, a.obs_names))
        barcodes = [bc for bc in barcodes if bc in keep]
    if not barcodes:
        raise ValueError("exon and intron matrices share no barcodes")

    blocks, block_kinds, kinds, var_names = [], [], [], []
    for a, kind in ((exon, "exon"), (intron, "intron")):
        if a is None:
            continue
        row = {str(bc): i for i, bc in enumerate(a.obs_names)}
        idx = _np.array([row[bc] for bc in barcodes], dtype=_np.int64)
        blocks.append(csr_matrix(a.X)[idx, :])
        block_kinds.append(kind)
        kinds.extend([kind] * a.n_vars)
        var_names.extend(map(str, a.var_names))

    n_obs = len(barcodes)
    dtype = blocks[0].dtype
    X = hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0].tocsr()

    def _layer(which: str):
        # Each block occupies a contiguous column range, so a layer is that
        # block padded with all-zero columns. Building it this way avoids a
        # masked assignment into a CSR matrix, which is quadratic.
        parts = [
            blk if k == which else csr_matrix((n_obs, blk.shape[1]), dtype=dtype)
            for blk, k in zip(blocks, block_kinds)
        ]
        return hstack(parts, format="csr") if len(parts) > 1 else parts[0].tocsr()

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.Index(barcodes)),
        var=pd.DataFrame(
            {
                "feature_kind": kinds,
                "gene_id": [v.split("_")[0] for v in var_names],
            },
            index=pd.Index(var_names),
        ),
    )
    adata.layers["spliced"] = _layer("exon")
    adata.layers["unspliced"] = _layer("intron")
    return adata


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_summary(out_dir: Path) -> dict:
    p = out_dir / "summary.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _load_pileup_stats(out_dir: Path) -> dict:
    p = out_dir / "pileup_stats.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _get_called_barcodes(out_dir: Path) -> List[str]:
    """Return list of cell barcodes from cell_calls.tsv (is_cell == True)."""
    import pandas as pd

    p = out_dir / "cell_calls.tsv"
    if not p.exists():
        # Fall back: all barcodes from auto_barcodes.tsv
        ab = out_dir / "auto_barcodes.tsv"
        if ab.exists():
            return list(pd.read_csv(ab, sep="\t", header=None)[0])
        return []
    df = pd.read_csv(p, sep="\t")
    if len(df) == 0:
        return []
    if "is_cell" in df.columns:
        # Use .loc with explicit bool cast to avoid pandas object-dtype Series
        # being misinterpreted as a column selector on empty DataFrames.
        mask = df["is_cell"].astype(bool)
        return list(df.loc[mask, "barcode"])
    return list(df["barcode"])


def _build_feature_vocab(out_dir: Path) -> dict:
    """Build feature_vocab from gene_expression.tsv.

    Returns a dict with:
      - reference_build: str
      - genes: list of {gene_id, gene_name}
    """
    import pandas as pd

    gene_expr_path = out_dir / "gene_expression.tsv"
    if not gene_expr_path.exists():
        raise FileNotFoundError(f"gene_expression.tsv not found in {out_dir}")

    gdf = pd.read_csv(gene_expr_path, sep="\t")
    summary = _load_summary(out_dir)
    reference_build = summary.get("reference_build", "unknown")

    genes = []
    for _, row in gdf.iterrows():
        genes.append({
            "gene_id": str(row["gene_id"]),
            "gene_name": str(row["gene_name"]),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "reference_build": reference_build,
        "n_genes": len(genes),
        "genes": genes,
    }


def _aggregate_gene_counts(out_dir: Path, gene_order: List[str], called_bcs: List[str]):
    """Return (cells x genes) sparse matrix of gene-level UMI counts.

    Sums exon + intron counts across all exon/intron intervals per gene.
    Returns a scipy.sparse.csc_matrix of shape (n_cells, n_genes).
    """
    from scipy.sparse import coo_matrix, csc_matrix
    from singlet._io import _read_pz_native

    gene_id_to_row = {g: i for i, g in enumerate(gene_order)}
    n_genes = len(gene_order)
    n_cells = len(called_bcs)

    def _agg_pz(pz_path: Path) -> Optional["csc_matrix"]:
        if not pz_path.exists():
            return None
        mat, rownames, all_bcs, _ = _read_pz_native(str(pz_path))
        bc_to_col = {bc: i for i, bc in enumerate(all_bcs)}
        cell_col_idx = np.array(
            [bc_to_col[bc] for bc in called_bcs if bc in bc_to_col], dtype=np.int32
        )
        if len(cell_col_idx) == 0:
            return csc_matrix((n_genes, n_cells), dtype=np.int32)
        mat_cells = mat[:, cell_col_idx]

        gene_ids_per_feature = [r.split("_")[0] for r in rownames]
        feat_to_gene = np.array(
            [gene_id_to_row.get(g, -1) for g in gene_ids_per_feature], dtype=np.int32
        )
        valid = feat_to_gene >= 0
        if not valid.any():
            return csc_matrix((n_genes, n_cells), dtype=np.int32)

        rows_v = feat_to_gene[valid]
        indicator = coo_matrix(
            (np.ones(valid.sum(), dtype=np.float32), (rows_v, np.arange(valid.sum()))),
            shape=(n_genes, valid.sum()),
        ).tocsr()
        agg = indicator.dot(mat_cells[valid, :])  # (n_genes, n_cells)
        return agg.T.tocsc().astype(np.int32)  # (n_cells, n_genes)

    exon = _agg_pz(out_dir / "exon_counts.1pz")
    intron = _agg_pz(out_dir / "intron_counts.1pz")

    if exon is None and intron is None:
        return csc_matrix((n_cells, n_genes), dtype=np.int32)
    if exon is None:
        return intron
    if intron is None:
        return exon
    return (exon + intron).tocsc()


# ---------------------------------------------------------------------------
# Packager
# ---------------------------------------------------------------------------


def pack_gse(
    gse_id: str,
    results_dir: Union[str, Path],
    catalog_path: Union[str, Path],
    out_path: Union[str, Path],
    *,
    verbose: bool = True,
    metadata_enriched_path: Optional[Union[str, Path]] = None,
    publications_enriched_path: Optional[Union[str, Path]] = None,
    series_meta_path: Optional[Union[str, Path]] = None,
    _catalog_df=None,
) -> Path:
    """Build a per-GSE .singlet bundle from pipeline results.

    Gathers all done GSMs for *gse_id* (those with
    ``results_dir/<GSM>/out/pileup_stats.json``), assembles the bundle per
    the .singlet spec (ZIP64), computes SHA-256 checksums, writes
    ``<out_path>``.

    Parameters
    ----------
    gse_id
        GEO Series accession (e.g. ``"GSE122083"``).
    results_dir
        Root directory of pipeline results; each child is a GSM directory.
    catalog_path
        Path to ``processing_catalog.parquet`` (maps gsm_id → gse_id + metadata).
    out_path
        Output path for the .singlet bundle. Typically ``<dir>/<gse_id>.singlet``.
    verbose
        Print progress lines to stdout.
    metadata_enriched_path
        Optional path to ``metadata_enriched.parquet`` (per-GSM enriched fields
        such as meta_tissue, meta_cell_type, meta_disease, meta_sex, meta_age).
        Merged opportunistically; absence or partial writes are handled silently.
    publications_enriched_path
        Optional path to ``publications_enriched.parquet`` (per-GSE pubs).
        Merged opportunistically; absence or partial writes are handled silently.
    series_meta_path
        Optional path to write a standalone ``series_meta.json`` sidecar
        (GSE metadata + per-GSM list + cells + status) that can be served
        separately from the heavy bundle.  When *None*, no sidecar is written.

    Returns
    -------
    Path
        Resolved path to the written bundle.

    Raises
    ------
    ValueError
        If no done GSMs are found for *gse_id*.
    """
    import pandas as pd

    results_dir = Path(results_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load catalog ---------------------------------------------------
    if _catalog_df is not None:
        catalog = _catalog_df
    else:
        if verbose:
            print(f"[pack_gse] Loading catalog from {catalog_path}")
        catalog = pd.read_parquet(catalog_path)

    # Filter to GSE
    gse_rows = catalog[catalog["gse_id"] == gse_id].copy()
    if gse_rows.empty:
        raise ValueError(f"GSE {gse_id!r} not found in catalog")

    # Find done GSMs (have pileup_stats.json)
    def _is_done(gsm: str) -> bool:
        return (results_dir / gsm / "out" / "pileup_stats.json").exists()

    gse_rows["_done"] = gse_rows["gsm_id"].apply(_is_done)
    done_rows = gse_rows[gse_rows["_done"]]
    if done_rows.empty:
        raise ValueError(f"No done GSMs found for {gse_id} under {results_dir}")

    gsm_ids = list(done_rows["gsm_id"])
    if verbose:
        print(f"[pack_gse] {gse_id}: {len(gsm_ids)} done GSMs: {gsm_ids}")

    # ---- Opportunistically load enriched metadata -----------------------
    gsm_enriched, publications = _load_enriched_meta(
        gse_id=gse_id,
        gsm_ids=gsm_ids,
        metadata_enriched_path=metadata_enriched_path,
        publications_enriched_path=publications_enriched_path,
    )
    if verbose and (gsm_enriched or publications):
        print(f"[pack_gse] Enriched metadata: {len(gsm_enriched)} GSMs enriched, "
              f"{len(publications)} publications")

    # ---- Build feature_vocab from first GSM ----------------------------
    first_out = results_dir / gsm_ids[0] / "out"
    feature_vocab = _build_feature_vocab(first_out)
    gene_order = [g["gene_id"] for g in feature_vocab["genes"]]
    feature_vocab_bytes = json.dumps(feature_vocab, indent=2).encode()
    if verbose:
        print(f"[pack_gse] feature_vocab: {feature_vocab['n_genes']} genes, "
              f"ref={feature_vocab['reference_build']}")

    # ---- Build study_meta -----------------------------------------------
    # Series-level fields from first GSE row
    first_row = gse_rows.iloc[0]
    series_title = str(first_row.get("series_title", "")) if pd.notna(first_row.get("series_title")) else ""
    series_summary = str(first_row.get("summary", "")) if pd.notna(first_row.get("summary")) else ""
    series_overall_design = str(first_row.get("overall_design", "")) if pd.notna(first_row.get("overall_design")) else ""

    # Per-GSM metadata: catalog base fields + opportunistic enrichment overlay
    per_gsm_meta: Dict[str, dict] = {}
    for _, row in done_rows.iterrows():
        gsm = row["gsm_id"]
        out_dir = results_dir / gsm / "out"
        summ = _load_summary(out_dir)
        called_bcs = _get_called_barcodes(out_dir)
        base: Dict[str, Any] = {
            "gsm_id": gsm,
            "organism": str(row.get("organism", "")) if pd.notna(row.get("organism")) else "",
            "protocol": str(row.get("protocol_inferred", "")) if pd.notna(row.get("protocol_inferred")) else "",
            "sample_source": str(row.get("sample_source", "")) if pd.notna(row.get("sample_source")) else "",
            "sample_characteristics": str(row.get("sample_characteristics", "")) if pd.notna(row.get("sample_characteristics")) else "",
            "qc_flag": str(row.get("qc_flag", "")) if pd.notna(row.get("qc_flag")) else "",
            "processing_status": str(row.get("processing_status", "")) if pd.notna(row.get("processing_status")) else "",
            "reference_build": summ.get("reference_build", feature_vocab["reference_build"]),
            "protocol_name": summ.get("protocol_name", ""),
            "n_cells": summ.get("n_cells_called", len(called_bcs)),
            "median_umi_per_cell": summ.get("median_umi_per_cell", None),
            "median_genes_per_cell": summ.get("median_genes_per_cell", None),
            "mapping_rate": summ.get("mapping_rate", None),
        }
        # Overlay enriched metadata (additive: keys not in base, or explicit
        # enriched overrides like meta_tissue that are new fields)
        enriched = gsm_enriched.get(gsm, {})
        if enriched:
            base["enriched"] = enriched
        per_gsm_meta[gsm] = base

    study_meta = {
        "schema_version": SCHEMA_VERSION,
        "gse_id": gse_id,
        "series_title": series_title,
        "series_summary": series_summary,
        "series_overall_design": series_overall_design,
        "publications": publications,
        "enriched_meta_available": len(gsm_enriched) > 0,
        "gsm_meta": per_gsm_meta,
    }
    study_meta_bytes = json.dumps(study_meta, indent=2).encode()

    # ---- Write series_meta.json sidecar (standalone, served separately) ---
    if series_meta_path is not None:
        series_meta_path = Path(series_meta_path)
        series_meta_path.parent.mkdir(parents=True, exist_ok=True)
        # Compact version: include gsm list, n_cells, statuses, enriched fields
        total_cells = sum(m.get("n_cells", 0) or 0 for m in per_gsm_meta.values())
        series_meta_out = {
            "schema_version": SCHEMA_VERSION,
            "gse_id": gse_id,
            "series_title": series_title,
            "series_summary": series_summary,
            "series_overall_design": series_overall_design,
            "publications": publications,
            "enriched_meta_available": len(gsm_enriched) > 0,
            "n_gsms": len(gsm_ids),
            "n_cells_total": total_cells,
            "reference_build": feature_vocab["reference_build"],
            "gsm_meta": per_gsm_meta,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(series_meta_path, "w") as f:
            json.dump(series_meta_out, f, indent=2)
        if verbose:
            print(f"[pack_gse] series_meta.json written → {series_meta_path}")

    # ---- Write ZIP64 bundle --------------------------------------------
    checksums: Dict[str, str] = {}
    included_files: Dict[str, list] = {}

    if verbose:
        print(f"[pack_gse] Writing bundle → {out_path}")

    with zipfile.ZipFile(str(out_path), "w", allowZip64=True) as zf:

        # feature_vocab.json  (DEFLATED)
        checksums["feature_vocab.json"] = _sha256_bytes(feature_vocab_bytes)
        zf.writestr(
            zipfile.ZipInfo("feature_vocab.json"),
            feature_vocab_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
        )

        # Per-GSM files
        for gsm in gsm_ids:
            out_dir = results_dir / gsm / "out"
            gsm_files = []

            # Pack the ENTIRE out/ recursively so the bundle carries the full
            # rich pipeline output (donor/SNP/ASE, mt_variants, nonhost/, extended
            # QC, splice/ambient, STAR outputs) — not just the curated core.
            # .1pz files are STORED (already zstd); everything else DEFLATED.
            # Dotfiles, raw reads (.1fq) and bulky disposable read-dumps are skipped.
            if out_dir.is_dir():
                for fpath in sorted(out_dir.rglob("*")):
                    if not fpath.is_file():
                        continue
                    rel = fpath.relative_to(out_dir)
                    if any(part.startswith(".") for part in rel.parts):
                        continue
                    # Trim raw count matrices — redundant with the .1pz codecs
                    # (the count matrix is reconstructable from exon/intron_counts.1pz).
                    if rel.parts[0] in {"star_Solo.out", "filtered_feature_bc_matrix"}:
                        continue
                    if fpath.suffix == ".1fq" or fpath.name in {
                        "star_Unmapped.out.mate1",
                        "star_Unmapped.out.mate2",
                        "nonhost_reads.fq",
                    }:
                        continue
                    arc_name = f"samples/{gsm}/{rel.as_posix()}"
                    with open(fpath, "rb") as fh:
                        data = fh.read()
                    checksums[arc_name] = _sha256_bytes(data)
                    gsm_files.append(rel.as_posix())
                    zi = zipfile.ZipInfo(arc_name)
                    zi.compress_type = (
                        zipfile.ZIP_STORED
                        if fpath.suffix == ".1pz"
                        else zipfile.ZIP_DEFLATED
                    )
                    zf.writestr(zi, data)

            included_files[gsm] = gsm_files
            if verbose:
                print(f"  [pack_gse] {gsm}: {len(gsm_files)} files packed")

        # study_meta.json (DEFLATED)
        checksums["study_meta.json"] = _sha256_bytes(study_meta_bytes)
        zf.writestr(
            zipfile.ZipInfo("study_meta.json"),
            study_meta_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
        )

        # manifest.json (DEFLATED) — written last so checksums are complete
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "gse_id": gse_id,
            "gsm_ids": gsm_ids,
            "reference_build": feature_vocab["reference_build"],
            "n_gsms": len(gsm_ids),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "singlet_version": _singlet_version(),
            "included_files": included_files,
            "checksums": checksums,
        }
        manifest_bytes = json.dumps(manifest, indent=2).encode()
        zf.writestr(
            zipfile.ZipInfo("manifest.json"),
            manifest_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
        )

    bundle_size_mb = out_path.stat().st_size / 1e6
    if verbose:
        print(f"[pack_gse] Done: {out_path}  ({bundle_size_mb:.1f} MB)")

    return out_path


def _load_enriched_meta(
    gse_id: str,
    gsm_ids: List[str],
    metadata_enriched_path: Optional[Union[str, Path]],
    publications_enriched_path: Optional[Union[str, Path]],
) -> tuple:
    """Opportunistically load enriched metadata for one GSE.

    Reads ``metadata_enriched.parquet`` (per-GSM fields: meta_tissue,
    meta_cell_type, meta_disease, meta_sex, meta_age, etc.) and
    ``publications_enriched.parquet`` (per-GSE pubs) if they exist and
    contain rows for this GSE/GSMs.  Falls back gracefully when files are
    absent, unreadable, or only partially written by a concurrent job.

    Returns
    -------
    (gsm_enriched: dict[str, dict], publications: list[dict])
        *gsm_enriched* maps gsm_id → enriched field dict (may be empty for
        GSMs not yet enriched).  *publications* is a list of pub dicts (may
        be empty).
    """
    import pandas as pd

    gsm_enriched: Dict[str, Dict[str, Any]] = {}
    publications: List[Dict[str, Any]] = []

    # --- per-GSM enriched metadata ---
    if metadata_enriched_path is not None:
        meta_path = Path(metadata_enriched_path)
        if meta_path.exists():
            try:
                meta_df = pd.read_parquet(meta_path)
                # Filter to our GSMs
                if "gsm_id" in meta_df.columns:
                    meta_df = meta_df[meta_df["gsm_id"].isin(gsm_ids)]
                    for _, row in meta_df.iterrows():
                        gsm = row["gsm_id"]
                        entry: Dict[str, Any] = {}
                        for col in meta_df.columns:
                            if col == "gsm_id":
                                continue
                            val = row[col]
                            # Skip NaN/None
                            try:
                                import math
                                if val is None or (isinstance(val, float) and math.isnan(val)):
                                    continue
                            except (TypeError, ValueError):
                                pass
                            entry[col] = val
                        if entry:
                            gsm_enriched[gsm] = entry
            except Exception:
                # File may be partially written by concurrent job — skip silently
                pass

    # --- per-GSE publications ---
    if publications_enriched_path is not None:
        pubs_path = Path(publications_enriched_path)
        if pubs_path.exists():
            try:
                pubs_df = pd.read_parquet(pubs_path)
                if "gse_id" in pubs_df.columns:
                    pubs_df = pubs_df[pubs_df["gse_id"] == gse_id]
                    publications = pubs_df.drop(columns=["gse_id"], errors="ignore").to_dict(orient="records")
            except Exception:
                pass

    return gsm_enriched, publications


def _singlet_version() -> str:
    try:
        from singlet._versions import __version__
        return __version__
    except Exception:
        try:
            import importlib.metadata
            return importlib.metadata.version("singlet")
        except Exception:
            return "unknown"


# ---------------------------------------------------------------------------
# Reader / converter
# ---------------------------------------------------------------------------


class SingletBundle:
    """Reader for a per-GSE .singlet bundle.

    Open with :meth:`open` (class method) or directly via the constructor.

    Attributes
    ----------
    path : Path
        Path to the .singlet bundle file.
    manifest : dict
        Parsed ``manifest.json``.
    study_meta : dict
        Parsed ``study_meta.json``.
    feature_vocab : dict
        Parsed ``feature_vocab.json``.
    gsm_ids : list[str]
        Ordered list of GSM accessions in the bundle.

    Examples
    --------
    >>> bundle = SingletBundle.open("GSE122083.singlet")  # doctest: +SKIP
    >>> adata = bundle.to_anndata()                        # doctest: +SKIP
    >>> adata.shape                                        # doctest: +SKIP
    (6974, 38606)
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._zf: Optional[zipfile.ZipFile] = None
        self._manifest: Optional[dict] = None
        self._study_meta: Optional[dict] = None
        self._feature_vocab: Optional[dict] = None

    @classmethod
    def open(cls, path: Union[str, Path]) -> "SingletBundle":
        """Open a .singlet bundle for reading.

        Parameters
        ----------
        path
            Path to the ``.singlet`` file.

        Returns
        -------
        SingletBundle
        """
        return cls(path)

    # ---- Low-level ZIP access ------------------------------------------

    def _zip(self) -> zipfile.ZipFile:
        if self._zf is None or not self._zf.fp:
            self._zf = zipfile.ZipFile(str(self.path), "r")
        return self._zf

    def _read_json(self, arc_name: str) -> dict:
        with self._zip().open(arc_name) as fh:
            return json.load(fh)

    def close(self) -> None:
        """Close the underlying ZIP file handle."""
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    def __enter__(self) -> "SingletBundle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- Properties ----------------------------------------------------

    @property
    def manifest(self) -> dict:
        """Parsed ``manifest.json``."""
        if self._manifest is None:
            self._manifest = self._read_json("manifest.json")
        return self._manifest

    @property
    def study_meta(self) -> dict:
        """Parsed ``study_meta.json``."""
        if self._study_meta is None:
            self._study_meta = self._read_json("study_meta.json")
        return self._study_meta

    @property
    def feature_vocab(self) -> dict:
        """Parsed ``feature_vocab.json``."""
        if self._feature_vocab is None:
            self._feature_vocab = self._read_json("feature_vocab.json")
        return self._feature_vocab

    @property
    def gsm_ids(self) -> List[str]:
        """Ordered list of GSM accessions in the bundle."""
        return self.manifest["gsm_ids"]

    # ---- Modality access -----------------------------------------------

    def list_files(self, gsm: Optional[str] = None) -> List[str]:
        """List archive members, optionally restricted to one sample.

        Parameters
        ----------
        gsm
            When given, return the paths *relative to* ``samples/<gsm>/``
            (e.g. ``"exon_counts.1pz"``, ``"donor/donor_assignments.tsv"``).
            When ``None``, return every member path in the archive.

        Returns
        -------
        list of str
        """
        names = self._zip().namelist()
        if gsm is None:
            return names
        prefix = f"samples/{gsm}/"
        return [n[len(prefix):] for n in names if n.startswith(prefix) and not n.endswith("/")]

    def _resolve_member(self, gsm: str, name: str) -> Optional[str]:
        """Map a modality name (or a literal member path) to an archive member."""
        available = set(self.list_files(gsm))
        candidates = MODALITIES[name].members if name in MODALITIES else (name,)
        for cand in candidates:
            if cand in available:
                return f"samples/{gsm}/{cand}"
        return None

    def has(self, name: str, gsm: Optional[str] = None) -> bool:
        """Whether a modality (or literal file) is present for a sample.

        With ``gsm=None`` the check passes if *any* sample has it.
        """
        targets = [gsm] if gsm is not None else self.gsm_ids
        return any(self._resolve_member(g, name) is not None for g in targets)

    def modalities(self, gsm: Optional[str] = None) -> Dict[str, str]:
        """Modalities available for a sample, as ``{name: description}``.

        Parameters
        ----------
        gsm
            Sample to inspect. Defaults to the first sample in the bundle,
            which is representative because every sample of a study is
            processed with the same pipeline invocation.

        Returns
        -------
        dict
            Ordered as in :data:`MODALITIES`. Use :meth:`read` to load any of
            the keys.

        Examples
        --------
        >>> bundle.modalities()                       # doctest: +SKIP
        {'exon_counts': 'Per-exon-feature UMI counts…', …}
        """
        target = gsm if gsm is not None else (self.gsm_ids[0] if self.gsm_ids else None)
        if target is None:
            return {}
        return {
            name: mod.description
            for name, mod in MODALITIES.items()
            if self._resolve_member(target, name) is not None
        }

    def read(self, name: str, gsm: str):
        """Read one modality for one sample.

        Parameters
        ----------
        name
            A key of :data:`MODALITIES` (e.g. ``"mt_heteroplasmy"``,
            ``"donor_assignments"``, ``"nonhost_species"``) or a literal file
            path inside ``samples/<gsm>/``.
        gsm
            GEO sample accession.

        Returns
        -------
        anndata.AnnData
            For ``matrix`` modalities: cells × features, with
            ``var_names`` = feature ids and ``obs_names`` = barcodes.
        pandas.DataFrame
            For ``table`` modalities.
        dict
            For ``json`` modalities.
        str
            For ``text`` modalities.

        Raises
        ------
        KeyError
            If the modality is not present for this sample. Check first with
            :meth:`modalities` or :meth:`has`.
        """
        member = self._resolve_member(gsm, name)
        if member is None:
            raise KeyError(
                f"{name!r} is not present for {gsm} in {self.path.name}. "
                f"Available: {sorted(self.modalities(gsm))}"
            )
        kind = MODALITIES[name].kind if name in MODALITIES else _infer_kind(member)
        with self._zip().open(member) as fh:
            data = fh.read()

        if kind == "json":
            return json.loads(data)
        if kind == "text":
            return data.decode("utf-8", errors="replace")
        if kind == "table":
            import io as _io

            import pandas as pd

            return pd.read_csv(_io.BytesIO(data), sep="\t")
        if kind != "matrix":
            raise ValueError(f"Unknown modality kind {kind!r} for {name!r}")

        import tempfile

        import anndata as ad
        import pandas as pd

        from singlet._io import _read_pz_native

        with tempfile.NamedTemporaryFile(suffix=".1pz", delete=False) as tf:
            tf.write(data)
            tmp_path = tf.name
        try:
            mat, rownames, colnames, user_kv = _read_pz_native(tmp_path)
        finally:
            os.unlink(tmp_path)

        # .1pz stores features × cells; AnnData wants cells × features.
        adata = ad.AnnData(X=mat.T.tocsr())
        if rownames:
            adata.var_names = pd.Index(rownames)
        if colnames:
            adata.obs_names = pd.Index(colnames)
        adata.obs["gsm_id"] = gsm
        if user_kv:
            adata.uns.update(user_kv)
        adata.uns["modality"] = name
        return adata

    def raw_counts(self, gsm: str, *, gene_level: bool = True, cells: str = "called"):
        """The raw count matrix for one sample, with spliced/unspliced layers.

        This is the matrix a conventional (non-USA) pipeline would hand you:
        ``X`` is exonic + intronic counts, and the two halves are kept as
        layers so nothing is lost.

        Parameters
        ----------
        gsm
            GEO sample accession.
        gene_level
            ``True`` (default) sums the per-feature counts onto the bundle's
            canonical gene axis, so the matrix is comparable across samples
            and studies. ``False`` keeps the pipeline's native per-feature
            rows (individual exons and introns), which is what you want for
            feature-resolved work.
        cells
            ``"called"`` (default) keeps only barcodes the pipeline called as
            cells. ``"all"`` keeps every barcode in the matrix, including
            empty droplets — the equivalent of a ``raw_feature_bc_matrix``.
            Only meaningful with ``gene_level=False``; the gene-level path is
            always restricted to called cells.

        Returns
        -------
        anndata.AnnData
            Cells × features. ``layers["spliced"]`` = exonic,
            ``layers["unspliced"]`` = intronic, ``X`` = their sum.

        Examples
        --------
        >>> adata = bundle.raw_counts("GSM5293863")          # doctest: +SKIP
        >>> adata.X.sum() == (adata.layers["spliced"].sum()
        ...                   + adata.layers["unspliced"].sum())  # doctest: +SKIP
        True
        """
        import anndata as ad
        import pandas as pd

        if cells not in ("called", "all"):
            raise ValueError("cells must be 'called' or 'all'")

        if gene_level:
            mat, barcodes = self._load_gsm_gene_counts(gsm)
            if mat.shape[0] == 0:
                raise ValueError(f"{gsm} has no called cells in {self.path.name}")
            gene_ids = [g["gene_id"] for g in self.feature_vocab["genes"]]
            gene_names = [g["gene_name"] for g in self.feature_vocab["genes"]]
            adata = ad.AnnData(
                X=mat.tocsr(),
                obs=pd.DataFrame(index=pd.Index(barcodes)),
                var=pd.DataFrame({"gene_name": gene_names}, index=pd.Index(gene_ids)),
            )
            for layer, kind in (("spliced", "exon"), ("unspliced", "intron")):
                part = self._gene_level_part(gsm, kind, barcodes)
                if part is not None:
                    adata.layers[layer] = part
        else:
            exon = self.read("exon_counts", gsm) if self.has("exon_counts", gsm) else None
            intron = self.read("intron_counts", gsm) if self.has("intron_counts", gsm) else None
            if exon is None and intron is None:
                raise KeyError(f"{gsm} has no count matrices in {self.path.name}")
            adata = _union_feature_axis(exon, intron)
            if cells == "called":
                called = self._called_barcodes(gsm)
                if called is not None:
                    keep = [bc for bc in adata.obs_names if bc in called]
                    if not keep:
                        raise ValueError(f"{gsm} has no called cells in {self.path.name}")
                    adata = adata[keep].copy()

        adata.obs["gsm_id"] = gsm
        adata.var_names_make_unique()
        adata.uns["reference_build"] = self.feature_vocab.get("reference_build", "")
        return adata

    def _called_barcodes(self, gsm: str) -> Optional[set]:
        """Barcodes the pipeline called as cells, or None if not recorded."""
        if not self.has("cell_calls", gsm):
            return None
        calls = self.read("cell_calls", gsm)
        col = next((c for c in ("barcode", "cb", "cell_barcode") if c in calls.columns), None)
        if col is None:
            return None
        return set(calls[col].astype(str))

    def _gene_level_part(self, gsm: str, kind: str, barcodes: List[str]):
        """Gene-level exon-only or intron-only matrix aligned to ``barcodes``."""
        fname = "exon_counts.1pz" if kind == "exon" else "intron_counts.1pz"
        if fname not in set(self.list_files(gsm)):
            return None
        from scipy.sparse import coo_matrix

        adata = self.read(fname, gsm)
        gene_order = [g["gene_id"] for g in self.feature_vocab["genes"]]
        gene_row = {g: i for i, g in enumerate(gene_order)}
        feat_gene = np.array(
            [gene_row.get(str(f).split("_")[0], -1) for f in adata.var_names], dtype=np.int64
        )
        valid = feat_gene >= 0
        projector = coo_matrix(
            (
                np.ones(int(valid.sum()), dtype=np.float32),
                (np.flatnonzero(valid), feat_gene[valid]),
            ),
            shape=(adata.n_vars, len(gene_order)),
        ).tocsr()
        bc_row = {bc: i for i, bc in enumerate(adata.obs_names)}
        idx = np.array([bc_row[bc] for bc in barcodes if bc in bc_row], dtype=np.int64)
        if len(idx) != len(barcodes):
            return None
        return (adata.X[idx, :] @ projector).astype(np.int32).tocsr()

    # ---- Convenience accessors -----------------------------------------

    def mt_variants(self, gsm: str):
        """Per-cell mitochondrial heteroplasmy (cells × chrM variant sites)."""
        return self.read("mt_heteroplasmy", gsm)

    def donors(self, gsm: str):
        """Donor demultiplexing assignments: barcode → donor, with doublet calls."""
        return self.read("donor_assignments", gsm)

    def nonhost(self, gsm: str):
        """Non-host (microbial/viral) abundance per taxon for one sample."""
        return self.read("nonhost_species", gsm)

    def junctions(self, gsm: str):
        """Per-cell splice-junction counts (cells × junctions)."""
        return self.read("junctions", gsm)

    def splice_psi(self, gsm: str):
        """Per-cell percent-spliced-in per splice event (cells × events)."""
        return self.read("splice_psi", gsm)

    def vdj(self, gsm: str):
        """Per-cell V(D)J segment usage (cells × segments)."""
        return self.read("vdj_gene_usage", gsm)

    def qc(self, gsm: Optional[str] = None) -> Dict[str, Any]:
        """Sample-level QC from ``summary.json``, keyed by GSM when ``gsm`` is None."""
        if gsm is not None:
            return self.read("summary", gsm)
        out: Dict[str, Any] = {}
        for g in self.gsm_ids:
            try:
                out[g] = self.read("summary", g)
            except KeyError:
                continue
        return out

    # ---- Data loading --------------------------------------------------

    def _extract_pz(self, gsm: str, fname: str) -> bytes:
        """Extract raw bytes of a .1pz file from the bundle."""
        arc_name = f"samples/{gsm}/{fname}"
        with self._zip().open(arc_name) as fh:
            return fh.read()

    def _load_gsm_gene_counts(self, gsm: str) -> tuple:
        """Load gene-level counts for one GSM.

        Returns
        -------
        (csc_matrix (n_cells, n_genes), barcodes: list[str])
        """
        import tempfile
        from singlet._io import _read_pz_native
        from scipy.sparse import coo_matrix, csc_matrix

        gene_order = [g["gene_id"] for g in self.feature_vocab["genes"]]
        gene_id_to_row = {g: i for i, g in enumerate(gene_order)}
        n_genes = len(gene_order)

        def _agg_from_arc(fname: str):
            """Extract .1pz to tmp, read, aggregate by gene."""
            try:
                raw = self._extract_pz(gsm, fname)
            except KeyError:
                return None, None, None

            with tempfile.NamedTemporaryFile(suffix=".1pz", delete=False) as tf:
                tf.write(raw)
                tmp_path = tf.name
            try:
                mat, rownames, all_bcs, _ = _read_pz_native(tmp_path)
            finally:
                os.unlink(tmp_path)
            return mat, rownames, all_bcs

        exon_mat, exon_rows, all_bcs = _agg_from_arc("exon_counts.1pz")
        if exon_mat is None:
            return csc_matrix((0, n_genes), dtype=np.int32), []

        # Determine called cell barcodes from cell_calls.tsv in the bundle.
        # If cell_calls.tsv is absent, fall back to all barcodes in the .1pz.
        called_bcs: List[str]
        try:
            import io as _io
            import pandas as _pd
            cc_arc = f"samples/{gsm}/cell_calls.tsv"
            with self._zip().open(cc_arc) as fh:
                cc_df = _pd.read_csv(_io.BytesIO(fh.read()), sep="\t")
            if "is_cell" in cc_df.columns:
                called_bcs = list(cc_df[cc_df["is_cell"]]["barcode"])
            else:
                called_bcs = list(cc_df["barcode"])
        except KeyError:
            # Older bundles without cell_calls.tsv — use all barcodes
            called_bcs = list(all_bcs)

        n_cells = len(called_bcs)

        # Build column index: map called_bcs → column indices in the .1pz
        bc_to_col = {bc: i for i, bc in enumerate(all_bcs)}
        cell_col_idx = np.array(
            [bc_to_col[bc] for bc in called_bcs if bc in bc_to_col], dtype=np.int32
        )
        # Some called_bcs might not be in the .1pz (edge case); re-derive
        matched_bcs = [bc for bc in called_bcs if bc in bc_to_col]
        n_cells = len(matched_bcs)
        if n_cells == 0:
            return csc_matrix((0, n_genes), dtype=np.int32), []

        def _agg_pz_mat(mat, rownames, pz_all_bcs):
            """Aggregate feature-level matrix to gene-level, filtered to called cells."""
            if mat is None:
                return None
            if pz_all_bcs is not None:
                # Build column index for this pz file's barcode order
                _bc_to_col = {bc: i for i, bc in enumerate(pz_all_bcs)}
                _cell_idx = np.array(
                    [_bc_to_col[bc] for bc in matched_bcs if bc in _bc_to_col],
                    dtype=np.int32,
                )
                if len(_cell_idx) == 0:
                    return csc_matrix((n_genes, n_cells), dtype=np.int32)
                mat_cells = mat[:, _cell_idx]
            else:
                # all_bcs ordering shared; use precomputed cell_col_idx
                mat_cells = mat[:, cell_col_idx]

            gene_ids_per_feature = [r.split("_")[0] for r in rownames]
            feat_to_gene = np.array(
                [gene_id_to_row.get(g, -1) for g in gene_ids_per_feature],
                dtype=np.int32,
            )
            valid = feat_to_gene >= 0
            if not valid.any():
                return csc_matrix((n_genes, n_cells), dtype=np.int32)
            rows_v = feat_to_gene[valid]
            indicator = coo_matrix(
                (np.ones(valid.sum(), dtype=np.float32),
                 (rows_v, np.arange(valid.sum()))),
                shape=(n_genes, valid.sum()),
            ).tocsr()
            return indicator.dot(mat_cells[valid, :]).astype(np.int32)

        exon_gene = _agg_pz_mat(exon_mat, exon_rows, all_bcs)

        # intron
        intron_mat, intron_rows, intron_all_bcs = _agg_from_arc("intron_counts.1pz")
        intron_gene = _agg_pz_mat(intron_mat, intron_rows, intron_all_bcs)

        if exon_gene is None and intron_gene is None:
            gene_mat = csc_matrix((n_genes, n_cells), dtype=np.int32)
        elif intron_gene is None:
            gene_mat = exon_gene
        elif exon_gene is None:
            gene_mat = intron_gene
        else:
            gene_mat = (exon_gene + intron_gene).astype(np.int32)

        # Transpose to (n_cells, n_genes) CSC
        return gene_mat.T.tocsc(), matched_bcs

    def to_anndata(
        self,
        layer: str = "gene_counts",
        *,
        layers: bool = True,
        verbose: bool = True,
    ) -> "anndata.AnnData":
        """Concatenate all GSMs into one AnnData.

        Parameters
        ----------
        layer : str
            Which count layer to assemble. Currently only ``"gene_counts"``
            (exon + intron sum) is supported.
        layers : bool
            Also attach ``spliced`` (exonic) and ``unspliced`` (intronic)
            layers alongside ``X``. Set to ``False`` to halve peak memory when
            you only need total counts.
        verbose : bool
            Print per-GSM progress.

        Returns
        -------
        anndata.AnnData
            - ``X`` : gene_counts (cells × genes), CSR sparse, int32.
            - ``layers["spliced"]`` / ``layers["unspliced"]`` : the exonic and
              intronic halves of ``X`` (only when ``layers=True`` and both
              matrices are present for every sample).
            - ``obs`` : index = ``<GSM>_<barcode>``; columns include
              ``gsm_id``, ``organism``, ``protocol``, ``sample_source``,
              ``sample_characteristics``, ``qc_flag``, ``reference_build``,
              ``n_cells_sample``.
            - ``var`` : index = Ensembl gene IDs; ``gene_name`` column.
            - ``uns`` : ``study_meta``, ``manifest``.
        """
        import anndata as ad
        import pandas as pd
        from scipy.sparse import vstack

        feat_vocab = self.feature_vocab
        gene_ids = [g["gene_id"] for g in feat_vocab["genes"]]
        gene_names = [g["gene_name"] for g in feat_vocab["genes"]]
        gsm_meta_map = self.study_meta.get("gsm_meta", {})

        obs_frames = []
        mats = []
        spliced_mats: List[Any] = []
        unspliced_mats: List[Any] = []

        for gsm in self.gsm_ids:
            if verbose:
                print(f"  [SingletBundle.to_anndata] Loading {gsm} ...")
            mat, barcodes = self._load_gsm_gene_counts(gsm)
            if mat.shape[0] == 0:
                if verbose:
                    print(f"    {gsm}: 0 cells, skipping")
                continue

            n_cells = mat.shape[0]
            obs_index = [f"{gsm}_{bc}" for bc in barcodes]
            gsm_info = gsm_meta_map.get(gsm, {})

            obs_df = pd.DataFrame(
                {
                    "gsm_id": [gsm] * n_cells,
                    "organism": [gsm_info.get("organism", "")] * n_cells,
                    "protocol": [gsm_info.get("protocol", "")] * n_cells,
                    "protocol_name": [gsm_info.get("protocol_name", "")] * n_cells,
                    "sample_source": [gsm_info.get("sample_source", "")] * n_cells,
                    "sample_characteristics": [gsm_info.get("sample_characteristics", "")] * n_cells,
                    "qc_flag": [gsm_info.get("qc_flag", "")] * n_cells,
                    "reference_build": [gsm_info.get("reference_build", "")] * n_cells,
                    "n_cells_sample": [gsm_info.get("n_cells", n_cells)] * n_cells,
                },
                index=obs_index,
            )
            obs_frames.append(obs_df)
            mats.append(mat)
            if layers:
                spliced_mats.append(self._gene_level_part(gsm, "exon", barcodes))
                unspliced_mats.append(self._gene_level_part(gsm, "intron", barcodes))
            if verbose:
                print(f"    {gsm}: {n_cells} cells, {mat.shape[1]} genes, "
                      f"nnz={mat.nnz}")

        if not mats:
            raise RuntimeError(f"No cell data loaded for bundle {self.path}")

        X = vstack(mats, format="csr")
        obs = pd.concat(obs_frames, axis=0)

        var = pd.DataFrame(
            {"gene_name": gene_names},
            index=gene_ids,
        )
        var.index.name = "gene_id"

        adata = ad.AnnData(X=X, obs=obs, var=var)
        if layers:
            for name, parts in (("spliced", spliced_mats), ("unspliced", unspliced_mats)):
                if parts and all(p is not None for p in parts):
                    adata.layers[name] = vstack(parts, format="csr")
                elif verbose:
                    print(f"  [SingletBundle.to_anndata] {name} layer unavailable "
                          f"for some samples — skipped")
        adata.uns["study_meta"] = self.study_meta
        adata.uns["manifest"] = self.manifest
        adata.uns["singlet_bundle_path"] = str(self.path)

        if verbose:
            print(f"  [SingletBundle.to_anndata] Done: {adata.shape}")

        return adata

    def to_h5ad(
        self,
        path: Union[str, Path],
        *,
        compression: str = "gzip",
        verbose: bool = True,
    ) -> Path:
        """Write concatenated AnnData to HDF5 (.h5ad).

        Parameters
        ----------
        path
            Output .h5ad path.
        compression
            HDF5 compression. Default ``"gzip"``.
        verbose
            Print progress.

        Returns
        -------
        Path
            Resolved output path.
        """
        path = Path(path)
        adata = self.to_anndata(verbose=verbose)
        adata.write_h5ad(path, compression=compression)
        if verbose:
            size_mb = path.stat().st_size / 1e6
            print(f"  [SingletBundle.to_h5ad] Wrote {path}  ({size_mb:.1f} MB)")
        return path

    def to_parquet(
        self,
        path: Union[str, Path],
        *,
        verbose: bool = True,
    ) -> Path:
        """Write obs metadata (no counts) to Parquet.

        Parameters
        ----------
        path
            Output .parquet path.
        verbose
            Print progress.

        Returns
        -------
        Path
        """
        import pandas as pd

        path = Path(path)
        gsm_meta_map = self.study_meta.get("gsm_meta", {})
        rows = []
        for gsm in self.gsm_ids:
            info = gsm_meta_map.get(gsm, {})
            rows.append({
                "gsm_id": gsm,
                "gse_id": self.manifest["gse_id"],
                "organism": info.get("organism", ""),
                "protocol": info.get("protocol", ""),
                "protocol_name": info.get("protocol_name", ""),
                "sample_source": info.get("sample_source", ""),
                "sample_characteristics": info.get("sample_characteristics", ""),
                "qc_flag": info.get("qc_flag", ""),
                "reference_build": info.get("reference_build", ""),
                "n_cells": info.get("n_cells", None),
                "median_umi_per_cell": info.get("median_umi_per_cell", None),
                "median_genes_per_cell": info.get("median_genes_per_cell", None),
                "mapping_rate": info.get("mapping_rate", None),
                "series_title": self.study_meta.get("series_title", ""),
            })
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        if verbose:
            print(f"  [SingletBundle.to_parquet] Wrote {path} ({len(df)} rows)")
        return path

    def __repr__(self) -> str:
        try:
            n = len(self.gsm_ids)
            gse = self.manifest.get("gse_id", "?")
            return f"SingletBundle({gse!r}, {n} GSMs, path={str(self.path)!r})"
        except Exception:
            return f"SingletBundle({str(self.path)!r})"


# ---------------------------------------------------------------------------
# CLI: singlet-pack
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry-point for ``python -m singlet.bundle``."""
    parser = argparse.ArgumentParser(
        prog="singlet-pack",
        description="Build a per-GSE .singlet bundle from pipeline results.",
    )
    sub = parser.add_subparsers(dest="cmd")

    pack_p = sub.add_parser("pack", help="Pack a GSE into a .singlet bundle.")
    pack_p.add_argument("--gse", required=True, help="GSE accession (e.g. GSE122083)")
    pack_p.add_argument(
        "--results",
        required=True,
        help="Root results directory (contains <GSM>/out/... subdirs)",
    )
    pack_p.add_argument(
        "--catalog",
        required=True,
        help="Path to processing_catalog.parquet",
    )
    pack_p.add_argument(
        "--out",
        required=True,
        help="Output path for the .singlet bundle",
    )
    pack_p.add_argument("--quiet", action="store_true")
    pack_p.add_argument(
        "--metadata-enriched",
        default=None,
        help="Optional path to metadata_enriched.parquet (per-GSM enriched fields); "
             "merged opportunistically if present.",
    )
    pack_p.add_argument(
        "--publications-enriched",
        default=None,
        help="Optional path to publications_enriched.parquet (per-GSE pubs); "
             "merged opportunistically if present.",
    )
    pack_p.add_argument(
        "--series-meta",
        default=None,
        help="Optional path to write a standalone series_meta.json sidecar.",
    )

    args = parser.parse_args(argv)

    if args.cmd == "pack":
        pack_gse(
            gse_id=args.gse,
            results_dir=args.results,
            catalog_path=args.catalog,
            out_path=args.out,
            verbose=not args.quiet,
            metadata_enriched_path=args.metadata_enriched,
            publications_enriched_path=args.publications_enriched,
            series_meta_path=args.series_meta,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
