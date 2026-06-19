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
from typing import Any, Dict, List, Optional, Union

import numpy as np

__all__ = [
    "pack_gse",
    "SingletBundle",
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
# Internal helpers
# ---------------------------------------------------------------------------


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

            # .1pz files — STORED (already zstd)
            for fname in _PZ_FILES:
                fpath = out_dir / fname
                if not fpath.exists():
                    continue
                arc_name = f"samples/{gsm}/{fname}"
                chk = _sha256_file(fpath)
                checksums[arc_name] = chk
                gsm_files.append(fname)
                zi = zipfile.ZipInfo(arc_name)
                zi.compress_type = zipfile.ZIP_STORED
                with open(fpath, "rb") as fh:
                    zf.writestr(zi, fh.read())
                if verbose:
                    size_mb = fpath.stat().st_size / 1e6
                    print(f"  [pack_gse] {gsm}/{fname}  ({size_mb:.1f} MB, STORED)")

            # JSON + TSV sidecars — DEFLATED
            for fname in _JSON_SIDECARS + _TSV_SIDECARS:
                fpath = out_dir / fname
                if not fpath.exists():
                    continue
                arc_name = f"samples/{gsm}/{fname}"
                with open(fpath, "rb") as fh:
                    data = fh.read()
                checksums[arc_name] = _sha256_bytes(data)
                gsm_files.append(fname)
                zf.writestr(
                    zipfile.ZipInfo(arc_name),
                    data,
                    compress_type=zipfile.ZIP_DEFLATED,
                )

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
        verbose: bool = True,
    ) -> "anndata.AnnData":
        """Concatenate all GSMs into one AnnData.

        Parameters
        ----------
        layer : str
            Which count layer to assemble. Currently only ``"gene_counts"``
            (exon + intron sum) is supported.
        verbose : bool
            Print per-GSM progress.

        Returns
        -------
        anndata.AnnData
            - ``X`` : gene_counts (cells × genes), CSR sparse, int32.
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
