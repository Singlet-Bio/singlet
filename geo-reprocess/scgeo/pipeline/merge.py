"""Merge per-GSM .1pz files into per-GSE consolidated counts.1pz.

Implements Phase 2 (Per-GSE Consolidation) and Phase 3 (Metadata
Standardization) of the catalog v1.0 MASTER_PLAN.

For each GSE directory:
  1. Discover all GSM subdirs with counts.1pz
  2. Read per-GSM matrices + sidecar metadata
  3. Verify gene reference consistency across GSMs
  4. hstack matrices into a single per-GSE counts.1pz
  5. Embed obs/var/uns metadata into the merged file
  6. Write study_metadata.json and provenance.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as ss

# Known USA gene reference counts per species
KNOWN_REFS = {
    115818: "USA_115818_human",
    171540: "USA_171540_mouse",
    97560: "USA_97560_zebrafish",
}

CATALOG_VERSION = "1.0"


@dataclass
class MergeReport:
    """Result of merging GSMs within a GSE (or species group)."""

    gse_id: str
    organism: str = "unknown"
    status: str = "unknown"
    n_gsms: int = 0
    n_cells: int = 0
    n_genes: int = 0
    nnz: int = 0
    merge_method: str = ""
    gsm_ids: list[str] = field(default_factory=list)
    output_dir: str = ""
    error: str = ""

    def ok(self) -> bool:
        return self.status == "ok"


def discover_gsms(gse_dir: Path) -> list[dict]:
    """Find all GSM subdirectories with counts.1pz and sidecar files."""
    gsms = []
    for entry in sorted(gse_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("GSM"):
            continue
        pz_path = entry / "counts.1pz"
        if not pz_path.exists():
            continue
        manifest_path = entry / "sample_manifest.json"
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        gsms.append({
            "gsm_id": entry.name,
            "dir": entry,
            "pz_path": pz_path,
            "manifest": manifest,
            "has_cell_meta": (entry / "cell_metadata.parquet").exists(),
            "has_feature_meta": (entry / "feature_metadata.parquet").exists(),
            "has_kraken2": (entry / "kraken2_cell_taxa.parquet").exists(),
        })
    return gsms


def group_by_species(gsms: list[dict]) -> dict[str, list[dict]]:
    """Group GSMs by organism for multi-species handling."""
    groups: dict[str, list[dict]] = {}
    for gsm in gsms:
        organism = gsm["manifest"].get("organism", "unknown")
        groups.setdefault(organism, []).append(gsm)
    return groups


def _read_feature_names(gsm_dir: Path) -> list[str] | None:
    """Read gene names from feature_metadata.parquet."""
    path = gsm_dir / "feature_metadata.parquet"
    if not path.exists():
        return None
    schema = pq.read_schema(path)
    col_names = [f.name for f in schema]
    if "gene_name" in col_names:
        df = pq.read_table(path, columns=["gene_name"]).to_pandas()
        return df["gene_name"].tolist()
    elif "gene_id" in col_names:
        df = pq.read_table(path, columns=["gene_id"]).to_pandas()
        return df["gene_id"].tolist()
    return None


def _read_cell_barcodes(gsm_dir: Path) -> list[str] | None:
    """Read cell barcodes from cell_metadata.parquet."""
    path = gsm_dir / "cell_metadata.parquet"
    if not path.exists():
        return None
    df = pq.read_table(path, columns=["barcode"]).to_pandas()
    return df["barcode"].tolist()


def merge_species_group(
    gse_id: str,
    gsms: list[dict],
    output_dir: Path,
    descriptions: dict[str, Any] | None = None,
    num_threads: int = 4,
) -> MergeReport:
    """Merge all GSMs for a single species group into counts.1pz.

    Parameters
    ----------
    gse_id : str
        GEO series accession.
    gsms : list[dict]
        GSM info dicts from discover_gsms().
    output_dir : Path
        Directory to write merged files.
    descriptions : dict, optional
        GSE-level descriptions keyed by gse_id.
    num_threads : int
        Threads for singlepress read/write.

    Returns
    -------
    MergeReport
        Merge status and statistics.
    """
    import singlepress

    descriptions = descriptions or {}
    report = MergeReport(
        gse_id=gse_id,
        n_gsms=len(gsms),
        gsm_ids=[g["gsm_id"] for g in gsms],
    )

    matrices = []
    all_colnames = []
    all_obs_rows = []
    provenance_gsms = {}
    gene_names = None
    expected_nrows = None

    for gsm in gsms:
        gsm_id = gsm["gsm_id"]

        mat = singlepress.read_1pz_int(str(gsm["pz_path"]), num_threads=num_threads)
        nrows, ncols = mat.shape

        if expected_nrows is None:
            expected_nrows = nrows
        elif nrows != expected_nrows:
            report.status = "error"
            report.error = (
                f"Row count mismatch: {gsm_id} has {nrows} rows, "
                f"expected {expected_nrows}"
            )
            return report

        if gene_names is None and gsm["has_feature_meta"]:
            gene_names = _read_feature_names(gsm["dir"])
            if gene_names is not None and len(gene_names) != nrows:
                gene_names = None

        if gsm["has_cell_meta"]:
            barcodes = _read_cell_barcodes(gsm["dir"])
        else:
            barcodes = None

        if barcodes is not None and len(barcodes) == ncols:
            prefixed = [f"{gsm_id}.{b}" for b in barcodes]
        else:
            prefixed = [f"{gsm_id}.cell_{i}" for i in range(ncols)]

        colsums = singlepress.colsums_1pz(str(gsm["pz_path"]))

        organism = gsm["manifest"].get("organism", "unknown")
        protocol = gsm["manifest"].get("protocol_catalog", "unknown")
        for i in range(ncols):
            all_obs_rows.append({
                "barcode": prefixed[i],
                "gsm_id": gsm_id,
                "organism": organism,
                "total_counts": int(colsums[i]),
            })

        all_colnames.extend(prefixed)
        matrices.append(mat)

        provenance_gsms[gsm_id] = {
            "pipeline_version": gsm["manifest"].get("pipeline_version", "unknown"),
            "n_cells": ncols,
            "n_features": nrows,
            "organism": organism,
            "protocol": protocol,
            "qc_status": gsm["manifest"].get("qc_status", "unknown"),
        }

    # Merge or copy single matrix
    if len(matrices) == 1:
        merged = matrices[0]
        merge_method = "single"
    else:
        merged = ss.hstack(matrices, format="csc")
        merge_method = "hstack"

    total_cells = merged.shape[1]
    total_genes = merged.shape[0]

    # Build var DataFrame
    var_df = None
    if gene_names is not None:
        ref_label = KNOWN_REFS.get(total_genes, f"custom_{total_genes}")
        var_df = pd.DataFrame({
            "gene_name": gene_names,
            "reference": [ref_label] * total_genes,
        })

    # Build obs DataFrame
    obs_df = pd.DataFrame(all_obs_rows)

    # Build uns metadata
    organisms = list({g["manifest"].get("organism", "unknown") for g in gsms})
    protocols = list({g["manifest"].get("protocol_catalog", "unknown") for g in gsms})
    desc = descriptions.get(gse_id, {})

    uns = {
        "gse_id": gse_id,
        "organism": "|".join(organisms),
        "protocol": "|".join(protocols),
        "n_samples": str(len(gsms)),
        "n_cells": str(total_cells),
        "n_genes": str(total_genes),
        "catalog_version": CATALOG_VERSION,
        "merge_method": merge_method,
    }
    if desc.get("title"):
        uns["title"] = desc["title"]
    if desc.get("pubmed_ids"):
        uns["pubmed_ids"] = desc["pubmed_ids"]

    # Write merged counts.1pz
    output_dir.mkdir(parents=True, exist_ok=True)
    singlepress.write_1pz(
        str(output_dir / "counts.1pz"),
        merged,
        rownames=gene_names,
        colnames=all_colnames,
        obs=obs_df,
        var=var_df,
        uns=uns,
        num_threads=8,
        level=3,
    )

    # Write sidecar metadata
    obs_df.to_parquet(str(output_dir / "metadata.parquet"), index=False)
    if var_df is not None:
        var_df.to_parquet(str(output_dir / "feature_metadata.parquet"), index=False)

    # Write study_metadata.json
    study_meta = {
        "gse_id": gse_id,
        "title": desc.get("title", ""),
        "summary": desc.get("summary", ""),
        "organism": organisms,
        "n_samples": len(gsms),
        "n_cells": total_cells,
        "n_genes": total_genes,
        "gsm_ids": [g["gsm_id"] for g in gsms],
        "protocol": protocols[0] if len(protocols) == 1 else protocols,
        "reference": KNOWN_REFS.get(total_genes, f"custom_{total_genes}"),
        "pubmed_ids": desc.get("pubmed_ids", ""),
        "submission_date": desc.get("submission_date", ""),
        "license": "public_domain",
        "catalog_version": CATALOG_VERSION,
    }
    with open(output_dir / "study_metadata.json", "w") as f:
        json.dump(study_meta, f, indent=2)

    # Write provenance.json
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "merge_method": merge_method,
        "source_gsms": provenance_gsms,
        "merged_shape": [total_genes, total_cells],
        "merged_nnz": int(merged.nnz),
        "catalog_version": CATALOG_VERSION,
    }
    with open(output_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    report.status = "ok"
    report.n_cells = total_cells
    report.n_genes = total_genes
    report.nnz = int(merged.nnz)
    report.merge_method = merge_method
    report.output_dir = str(output_dir)
    return report


def merge_gse(
    gse_dir: Path,
    descriptions: dict[str, Any] | None = None,
    num_threads: int = 4,
) -> list[MergeReport]:
    """Merge all GSMs within a GSE into per-species counts.1pz.

    Multi-species GSEs produce per-species subdirectories.

    Parameters
    ----------
    gse_dir : Path
        Path to the GSE directory containing GSM subdirs.
    descriptions : dict, optional
        GSE-level descriptions keyed by gse_id.
    num_threads : int
        Threads for singlepress read/write.

    Returns
    -------
    list[MergeReport]
        One report per species group.
    """
    gse_id = gse_dir.name
    gsms = discover_gsms(gse_dir)
    if not gsms:
        return [MergeReport(gse_id=gse_id, status="skip", error="No GSMs with counts.1pz")]

    species_groups = group_by_species(gsms)
    is_multi_species = len(species_groups) > 1

    reports = []
    for organism, group_gsms in species_groups.items():
        if is_multi_species:
            safe_name = organism.replace(" ", "_")
            output_dir = gse_dir / safe_name
        else:
            output_dir = gse_dir

        report = merge_species_group(
            gse_id, group_gsms, output_dir, descriptions, num_threads,
        )
        report.organism = organism
        reports.append(report)

    return reports
