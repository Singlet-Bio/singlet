#!/usr/bin/env python3
"""Migrate v5 pipeline outputs to the Cellarium dataset layout.

Source:  pipeline/quant/<GSE>/<GSM>/counts.spz  (flat, USA-concatenated)
Target:  dataset/<GSE>/<GSM>/rna/{spliced,unspliced,ambiguous}.spz
         dataset/<GSE>/<GSM>/cells.parquet
         dataset/<GSE>/<GSM>/manifest.json
         dataset/<GSE>/<GSM>/rna/features.parquet
         dataset/<GSE>/<GSM>/kraken2/{cell_taxa.parquet,summary.json}

Usage:
    python migrate_v5_to_cellarium.py --audit
    python migrate_v5_to_cellarium.py --gsm GSM1234567
    python migrate_v5_to_cellarium.py --all --threads 16
    python migrate_v5_to_cellarium.py --all --threads 16 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

import singlepress as sprs

# ── Paths ──────────────────────────────────────────────────────────────
PIPELINE_ROOT = Path("/mnt/projects/debruinz_project/cellarium/pipeline/quant")
DATASET_ROOT = Path("/mnt/projects/debruinz_project/cellarium/dataset")
CATALOG_ROOT = Path("/mnt/projects/debruinz_project/cellarium/catalog")
MIGRATION_LOG = DATASET_ROOT / "migration_log.json"

# Known USA total-feature counts -> (organism, n_genes)
KNOWN_USA_GENES = {
    115818: ("Homo sapiens", 38606),
    171540: ("Mus musculus", 57180),
    97560: ("Danio rerio", 32520),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("migrate")


@dataclass
class SampleInfo:
    gsm_id: str
    gse_id: str
    src_dir: Path
    dst_dir: Path
    manifest: dict = field(default_factory=dict)
    spz_rows: int = 0
    spz_cols: int = 0
    is_usa: bool = False
    is_transposed: bool = False
    n_genes: int = 0
    organism: str = ""
    status: str = "pending"
    error: str = ""


def detect_orientation(info_dict: dict) -> tuple[bool, bool, int]:
    """Detect USA structure and orientation from spz info.

    Returns (is_usa, is_transposed, n_genes).
    """
    rows, cols = info_dict["rows"], info_dict["cols"]
    if rows in KNOWN_USA_GENES:
        return True, False, KNOWN_USA_GENES[rows][1]
    if cols in KNOWN_USA_GENES:
        return True, True, KNOWN_USA_GENES[cols][1]
    if rows % 3 == 0 and rows > 30000:
        return True, False, rows // 3
    if cols % 3 == 0 and cols > 30000:
        return True, True, cols // 3
    return False, False, max(rows, cols)


def discover_samples() -> list[SampleInfo]:
    """Scan pipeline/quant/ for all GSE/GSM dirs with counts.spz."""
    samples = []
    for gse in sorted(os.listdir(PIPELINE_ROOT)):
        gse_path = PIPELINE_ROOT / gse
        if not gse_path.is_dir() or not gse.startswith("GSE"):
            continue
        for gsm in os.listdir(gse_path):
            gsm_path = gse_path / gsm
            if not gsm_path.is_dir() or not gsm.startswith("GSM"):
                continue
            if not (gsm_path / "counts.spz").exists():
                continue
            samples.append(SampleInfo(
                gsm_id=gsm, gse_id=gse,
                src_dir=gsm_path, dst_dir=DATASET_ROOT / gse / gsm,
            ))
    return samples


def load_manifest(src_dir: Path) -> dict:
    mf_path = src_dir / "sample_manifest.json"
    if mf_path.exists():
        with open(mf_path) as f:
            return json.load(f)
    return {}


def audit_samples(samples: list[SampleInfo]) -> dict:
    """Audit: detect duplicates, orientation stats, organism breakdown."""
    logger.info(f"Auditing {len(samples)} samples...")
    gsm_to_gse: dict[str, list[str]] = {}
    for s in samples:
        gsm_to_gse.setdefault(s.gsm_id, []).append(s.gse_id)
    duplicates = {k: v for k, v in gsm_to_gse.items() if len(v) > 1}

    usa_std = usa_trans = non_usa = 0
    organisms: dict[str, int] = {}
    protocols: dict[str, int] = {}
    check_n = min(len(samples), 500)
    step = max(1, len(samples) // check_n)

    for i in range(0, len(samples), step):
        s = samples[i]
        try:
            info = sprs.info(str(s.src_dir / "counts.spz"))
            is_usa, is_trans, ng = detect_orientation(info)
            s.is_usa, s.is_transposed, s.n_genes = is_usa, is_trans, ng
        except Exception as e:
            s.error = str(e)
        mf = load_manifest(s.src_dir)
        s.organism = mf.get("organism", "unknown")
        proto = mf.get("protocol_detected", "unknown")
        if s.is_usa and not s.is_transposed:
            usa_std += 1
        elif s.is_usa and s.is_transposed:
            usa_trans += 1
        else:
            non_usa += 1
        organisms[s.organism] = organisms.get(s.organism, 0) + 1
        protocols[proto] = protocols.get(proto, 0) + 1

    already = sum(1 for s in samples
                  if s.dst_dir.exists() and (s.dst_dir / "manifest.json").exists())
    return {
        "total_samples": len(samples),
        "unique_gsm_ids": len(gsm_to_gse),
        "duplicate_gsm_ids": len(duplicates),
        "duplicates_sample": dict(sorted(duplicates.items())[:30]),
        "checked_orientation": check_n,
        "usa_standard": usa_std,
        "usa_transposed": usa_trans,
        "non_usa": non_usa,
        "organisms": dict(sorted(organisms.items(), key=lambda x: -x[1])),
        "protocols": dict(sorted(protocols.items(), key=lambda x: -x[1])),
        "already_migrated": already,
    }


def deduplicate_samples(samples: list[SampleInfo]) -> list[SampleInfo]:
    """Resolve duplicate GSMs: keep lowest GSE number (subseries), drop others.

    GEO convention: superseries have higher GSE numbers and aggregate
    subseries. We prefer the original subseries posting.
    """
    gsm_best: dict[str, SampleInfo] = {}
    for s in samples:
        if s.gsm_id not in gsm_best:
            gsm_best[s.gsm_id] = s
        else:
            existing = gsm_best[s.gsm_id]
            # Keep the one with the lower GSE number (original subseries)
            if int(s.gse_id[3:]) < int(existing.gse_id[3:]):
                gsm_best[s.gsm_id] = s
    deduped = list(gsm_best.values())
    n_removed = len(samples) - len(deduped)
    if n_removed > 0:
        logger.info(f"Deduplication: removed {n_removed} duplicate GSM entries")
    return deduped


# ── Core migration per GSM ─────────────────────────────────────────────

def migrate_one(sample: SampleInfo, *, dry_run: bool = False) -> SampleInfo:
    """Migrate a single GSM from v5 layout to cellarium layout.

    Read-only on source. Writes to DATASET_ROOT.
    """
    src = sample.src_dir
    dst = sample.dst_dir

    # Skip if already migrated
    if (dst / "manifest.json").exists():
        sample.status = "skipped"
        return sample

    try:
        # 1. Read source counts.spz
        spz_path = src / "counts.spz"
        info = sprs.info(str(spz_path))
        is_usa, is_trans, n_genes = detect_orientation(info)
        sample.is_usa = is_usa
        sample.is_transposed = is_trans
        sample.n_genes = n_genes

        if not is_usa:
            sample.status = "skipped"
            sample.error = f"non-USA structure rows={info['rows']} cols={info['cols']}"
            return sample

        if dry_run:
            sample.status = "dry_run"
            return sample

        # Read the full matrix
        mat = sprs.read(str(spz_path))  # returns sparse CSC

        # Ensure genes×cells orientation
        if is_trans:
            mat = mat.T.tocsc()

        total_features = mat.shape[0]
        n_cells = mat.shape[1]

        # 2. Split USA layers (rows: [0..n_genes), [n_genes..2*n_genes), [2*n_genes..3*n_genes))
        spliced = mat[:n_genes, :]
        unspliced = mat[n_genes:2*n_genes, :]
        ambiguous = mat[2*n_genes:3*n_genes, :]

        # 3. Extract gene names from info or feature_metadata
        gene_names = _get_gene_names(src, n_genes, total_features)
        cell_barcodes = _get_cell_barcodes(src, n_cells)

        # 4. Create destination directories
        rna_dir = dst / "rna"
        rna_dir.mkdir(parents=True, exist_ok=True)

        # 5. Write split .spz files
        for name, layer_mat in [("spliced", spliced), ("unspliced", unspliced), ("ambiguous", ambiguous)]:
            _write_layer_spz(rna_dir / f"{name}.spz", layer_mat, gene_names, cell_barcodes)

        # 6. Build and write cells.parquet
        _write_cells_parquet(dst, cell_barcodes, spliced, unspliced, ambiguous, src)

        # 7. Write rna/features.parquet
        _write_features_parquet(rna_dir, gene_names, src)

        # 8. Migrate manifest
        _write_manifest(dst, src, n_cells, n_genes, spliced, unspliced, ambiguous)

        # 9. Migrate kraken2
        _migrate_kraken2(dst, src)

        sample.status = "migrated"
        logger.info(f"[{sample.gsm_id}] migrated: {n_genes} genes × {n_cells} cells")

    except Exception as e:
        sample.status = "error"
        sample.error = str(e)
        logger.error(f"[{sample.gsm_id}] error: {e}")

    return sample


# ── Helper functions ────────────────────────────────────────────────────

def _get_gene_names(src_dir: Path, n_genes: int, total_features: int) -> list[str]:
    """Extract gene names from feature_metadata.parquet or spz rownames."""
    feat_path = src_dir / "feature_metadata.parquet"
    if feat_path.exists():
        df = pd.read_parquet(feat_path)
        if "gene_name" in df.columns:
            names = df["gene_name"].tolist()
            if len(names) == total_features:
                return names[:n_genes]
            elif len(names) == n_genes:
                return names
    return [f"gene_{i}" for i in range(n_genes)]


def _get_cell_barcodes(src_dir: Path, n_cells: int) -> list[str]:
    """Extract cell barcodes from cell_metadata.parquet or spz colnames."""
    cell_path = src_dir / "cell_metadata.parquet"
    if cell_path.exists():
        df = pd.read_parquet(cell_path)
        if "barcode" in df.columns and len(df) == n_cells:
            return df["barcode"].tolist()
    return [f"cell_{i}" for i in range(n_cells)]


def _write_layer_spz(
    path: Path, mat: sp.spmatrix, gene_names: list[str], cell_barcodes: list[str]
) -> None:
    """Write one USA layer as a .spz file (genes x cells, CSC)."""
    mat_csc = mat.tocsc() if not sp.isspmatrix_csc(mat) else mat
    sprs.write(
        str(path), mat_csc,
        rownames=gene_names, colnames=cell_barcodes, verbose=0,
    )


def _write_cells_parquet(
    dst_dir: Path, barcodes: list[str],
    spliced: sp.spmatrix, unspliced: sp.spmatrix, ambiguous: sp.spmatrix,
    src_dir: Path,
) -> None:
    """Build and write cells.parquet with per-cell QC stats."""
    s_umis = np.asarray(spliced.sum(axis=0)).ravel().astype(np.uint32)
    u_umis = np.asarray(unspliced.sum(axis=0)).ravel().astype(np.uint32)
    a_umis = np.asarray(ambiguous.sum(axis=0)).ravel().astype(np.uint32)
    total_umis = s_umis + u_umis + a_umis
    n_genes_arr = np.asarray((spliced > 0).sum(axis=0)).ravel().astype(np.uint16)

    mf = load_manifest(src_dir)
    frac_unspliced = np.where(
        total_umis > 0, u_umis / total_umis, 0.0
    ).astype(np.float32)

    cells = pd.DataFrame({
        "barcode": barcodes,
        "cell_idx": np.arange(len(barcodes), dtype=np.uint32),
        "gsm_id": mf.get("gsm_id", ""),
        "gse_id": mf.get("gse_id", ""),
        "organism": mf.get("organism", ""),
        "protocol": mf.get("protocol_detected", ""),
        "chemistry": mf.get("chemistry_used", ""),
        "is_nuclear": False,
        "rna_total_umis": total_umis,
        "rna_n_genes": n_genes_arr,
        "rna_spliced_umis": s_umis,
        "rna_unspliced_umis": u_umis,
        "rna_ambiguous_umis": a_umis,
        "rna_frac_unspliced": frac_unspliced,
        "has_rna": True,
        "has_atac": False,
        "has_adt": False,
        "has_spatial": False,
        "has_guides": False,
        "has_kraken2": False,
    })

    # Enrich with kraken2 per-cell data if available
    kraken_path = src_dir / "kraken2_cell_taxa.parquet"
    if kraken_path.exists():
        try:
            taxa = pd.read_parquet(kraken_path)
            if "barcode" in taxa.columns and "umi_count" in taxa.columns:
                agg = taxa.groupby("barcode").agg(
                    kraken2_n_nonhost_umis=("umi_count", "sum"),
                ).reset_index()
                agg["kraken2_n_nonhost_umis"] = agg["kraken2_n_nonhost_umis"].astype(np.uint16)
                cells = cells.merge(agg, on="barcode", how="left")
                cells["kraken2_n_nonhost_umis"] = cells["kraken2_n_nonhost_umis"].fillna(0).astype(np.uint16)
                cells["kraken2_frac_nonhost"] = np.where(
                    cells["rna_total_umis"] > 0,
                    cells["kraken2_n_nonhost_umis"] / cells["rna_total_umis"],
                    0.0,
                ).astype(np.float32)
                cells["has_kraken2"] = True
        except Exception as e:
            logger.warning(f"Kraken2 enrichment failed: {e}")

    cells.to_parquet(
        str(dst_dir / "cells.parquet"), compression="zstd", index=False,
    )


def _write_features_parquet(
    rna_dir: Path, gene_names: list[str], src_dir: Path
) -> None:
    """Write rna/features.parquet from gene names."""
    feat_src = src_dir / "feature_metadata.parquet"
    if feat_src.exists():
        df = pd.read_parquet(feat_src)
        if "gene_name" in df.columns:
            n_genes = len(gene_names)
            if len(df) >= n_genes:
                df = df.iloc[:n_genes].copy()
            df.to_parquet(
                str(rna_dir / "features.parquet"),
                compression="zstd", index=False,
            )
            return
    pd.DataFrame({"gene_name": gene_names}).to_parquet(
        str(rna_dir / "features.parquet"), compression="zstd", index=False,
    )


def _write_manifest(
    dst_dir: Path, src_dir: Path,
    n_cells: int, n_genes: int,
    spliced: sp.spmatrix, unspliced: sp.spmatrix, ambiguous: sp.spmatrix,
) -> None:
    """Write manifest.json in the new schema."""
    old_mf = load_manifest(src_dir)
    new_mf = {
        "gsm_id": old_mf.get("gsm_id", ""),
        "gse_id": old_mf.get("gse_id", ""),
        "organism": old_mf.get("organism", ""),
        "taxon_id": old_mf.get("taxon_id", 0),
        "protocol_detected": old_mf.get("protocol_detected", ""),
        "chemistry_used": old_mf.get("chemistry_used", ""),
        "is_nuclear": False,
        "pipeline_version": "cellarium-v7.0-migrated",
        "processing_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_pipeline_version": old_mf.get("pipeline_version", ""),
        "rna": {
            "n_cells": n_cells,
            "n_genes": n_genes,
            "mapping_rate": old_mf.get("mapping_rate", 0.0),
            "median_genes_per_cell": old_mf.get("median_genes_per_cell", 0),
            "median_umis_per_cell": old_mf.get("median_counts_per_cell", 0),
            "frac_unspliced": old_mf.get("frac_unspliced", 0.0),
            "qc_status": old_mf.get("qc_status", ""),
            "spliced_spz_bytes": (dst_dir / "rna" / "spliced.spz").stat().st_size,
            "unspliced_spz_bytes": (dst_dir / "rna" / "unspliced.spz").stat().st_size,
            "ambiguous_spz_bytes": (dst_dir / "rna" / "ambiguous.spz").stat().st_size,
        },
        "kraken2": None,
        "atac": None, "adt": None, "spatial": None, "guides": None,
    }
    if old_mf.get("kraken2_frac_nonhost") is not None:
        new_mf["kraken2"] = {
            "ran": True,
            "frac_nonhost": old_mf.get("kraken2_frac_nonhost", 0.0),
            "cells_with_nonhost": old_mf.get("kraken2_cells_with_nonhost", 0),
            "total_nonhost_umis": old_mf.get("kraken2_total_nonhost_umis", 0),
        }
    with open(dst_dir / "manifest.json", "w") as f:
        json.dump(new_mf, f, indent=2)


def _migrate_kraken2(dst_dir: Path, src_dir: Path) -> None:
    """Copy kraken2 files to the new layout."""
    kr_src = src_dir / "kraken2_cell_taxa.parquet"
    kr_summary = src_dir / "kraken2_summary.json"
    if kr_src.exists():
        kr_dst = dst_dir / "kraken2"
        kr_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(kr_src), str(kr_dst / "cell_taxa.parquet"))
        if kr_summary.exists():
            shutil.copy2(str(kr_summary), str(kr_dst / "summary.json"))


# ── Batch runner ────────────────────────────────────────────────────────

def load_migration_log() -> dict:
    """Load migration progress log."""
    if MIGRATION_LOG.exists():
        with open(MIGRATION_LOG) as f:
            return json.load(f)
    return {"migrated": [], "errors": [], "skipped": []}


def save_migration_log(log: dict) -> None:
    """Save migration progress log."""
    MIGRATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(MIGRATION_LOG, "w") as f:
        json.dump(log, f, indent=2)


def run_batch(
    samples: list[SampleInfo],
    threads: int = 1,
    dry_run: bool = False,
    resume: bool = False,
) -> None:
    """Run migration on all samples with optional parallelism."""
    log = load_migration_log() if resume else {"migrated": [], "errors": [], "skipped": []}
    done_set = set(log["migrated"]) | set(log["skipped"])
    pending = [s for s in samples if s.gsm_id not in done_set]

    logger.info(f"Migration batch: {len(pending)} pending, {len(done_set)} already done")

    completed = 0
    errors = 0
    t0 = time.time()

    if threads <= 1:
        for s in pending:
            result = migrate_one(s, dry_run=dry_run)
            _update_log(log, result)
            completed += 1
            if result.status == "error":
                errors += 1
            if completed % 100 == 0:
                save_migration_log(log)
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {completed}/{len(pending)} ({rate:.1f}/s) errors={errors}")
    else:
        # Use multiprocessing.Pool with maxtasksperchild=1 for crash isolation.
        # Each worker processes exactly one sample then exits, so a C++ segfault
        # in sparsepress only loses that one sample (pool auto-replaces workers).
        import multiprocessing as mp
        with mp.Pool(processes=threads, maxtasksperchild=1) as pool:
            worker = partial(migrate_one, dry_run=dry_run)
            results_iter = pool.imap_unordered(worker, pending, chunksize=1)
            pending_iter = iter(pending)
            while completed < len(pending):
                try:
                    result = next(results_iter)
                except StopIteration:
                    break
                except Exception as exc:
                    # Worker crash — log error and continue
                    errors += 1
                    completed += 1
                    logger.error(f"Worker crashed on sample: {exc}")
                    continue
                _update_log(log, result)
                completed += 1
                if result.status == "error":
                    errors += 1
                if completed % 100 == 0:
                    save_migration_log(log)
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info(f"Progress: {completed}/{len(pending)} ({rate:.1f}/s) errors={errors}")

    save_migration_log(log)
    elapsed = time.time() - t0
    logger.info(
        f"Done: {completed} processed in {elapsed:.0f}s "
        f"(migrated={len(log['migrated'])}, errors={len(log['errors'])}, "
        f"skipped={len(log['skipped'])})"
    )


def _update_log(log: dict, sample: SampleInfo) -> None:
    """Update migration log with one sample result."""
    key = f"{sample.gse_id}/{sample.gsm_id}"
    if sample.status == "migrated":
        log["migrated"].append(key)
    elif sample.status == "error":
        log["errors"].append({"key": key, "error": sample.error})
    elif sample.status in ("skipped", "dry_run"):
        log["skipped"].append(key)


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Migrate v5 pipeline → Cellarium dataset layout")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit", action="store_true", help="Audit samples (no changes)")
    group.add_argument("--gsm", type=str, help="Migrate a single GSM")
    group.add_argument("--all", action="store_true", help="Migrate all completed samples")
    parser.add_argument("--threads", type=int, default=1, help="Parallel threads (default: 1)")
    parser.add_argument("--resume", action="store_true", help="Resume from migration log")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no writes)")
    args = parser.parse_args()

    logger.info("Discovering samples...")
    samples = discover_samples()
    logger.info(f"Found {len(samples)} samples with counts.spz")

    if args.audit:
        report = audit_samples(samples)
        print(json.dumps(report, indent=2))
        return

    # Deduplicate before migration
    samples = deduplicate_samples(samples)

    if args.gsm:
        target = [s for s in samples if s.gsm_id == args.gsm]
        if not target:
            logger.error(f"GSM {args.gsm} not found")
            sys.exit(1)
        result = migrate_one(target[0], dry_run=args.dry_run)
        print(json.dumps({"gsm": result.gsm_id, "status": result.status, "error": result.error}))
        return

    if args.all:
        run_batch(samples, threads=args.threads, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
