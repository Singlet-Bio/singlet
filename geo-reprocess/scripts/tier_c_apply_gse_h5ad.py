#!/usr/bin/env python3
"""Tier C: Map and apply GEO GSE-level h5ad metadata to per-GSM files.

Generalizable script for GSE-level h5ad files where:
- obs index has sample-prefixed barcodes (BARCODE-SAMPLEID)
- A 'sample' column (or similar) identifies samples
- Need to map sample IDs to GSM IDs via barcode overlap

Usage:
    python tier_c_apply_gse_h5ad.py <h5ad_path> <gse_id> <dataset_dir> <catalog_path> [--sample-col COLNAME] [--source LABEL]
"""
import sys
import json
import anndata as ad
import pandas as pd
import numpy as np
import os
import re

def detect_sample_column(obs, n_gsms=None):
    """Auto-detect the column that identifies samples."""
    candidates = ["sample", "samplename", "sample_id", "library_id", "orig.ident",
                   "batch", "donor_id", "sample_name", "Sample", "sample_labels"]
    # If n_gsms known, prefer column with matching nunique (±30%)
    if n_gsms and n_gsms > 1:
        for c in candidates:
            if c in obs.columns:
                nu = obs[c].nunique()
                if 0.7 * n_gsms <= nu <= 1.3 * n_gsms:
                    return c
        # Still try exact candidate match
        for c in candidates:
            if c in obs.columns:
                nu = obs[c].nunique()
                if nu >= 2:
                    return c
    else:
        for c in candidates:
            if c in obs.columns:
                return c
    # Fallback: column with moderate unique count
    for c in obs.columns:
        n = obs[c].nunique()
        if 2 <= n < 500 and obs[c].dtype.name in ("category", "object"):
            if n_gsms and 0.5 * n_gsms <= n <= 2.0 * n_gsms:
                return c
    for c in obs.columns:
        n = obs[c].nunique()
        if 2 <= n < 500 and obs[c].dtype.name in ("category", "object"):
            return c
    return None

def detect_barcode_format(obs):
    """Detect how barcodes are structured in the obs index."""
    sample = str(obs.index[0])
    # Pattern: BARCODE-SAMPLEID or SAMPLEID_BARCODE
    if re.match(r"[ACGTN]{12,18}-", sample):
        return "barcode-suffix"
    elif re.match(r".+_[ACGTN]{12,18}", sample):
        return "prefix-barcode"
    elif re.match(r"[ACGTN]{12,18}$", sample):
        return "bare"
    # Check for colon-separated: PREFIX:BARCODE-N
    elif ":" in sample and re.search(r"[ACGTN]{12,18}", sample):
        return "colon-barcode"
    # Fallback: any ACGTN sequence in the index
    elif re.search(r"[ACGTN]{12,18}", sample):
        return "regex-extract"
    return "unknown"

def extract_bare_barcode(obs_index, fmt):
    """Extract bare barcode from obs index based on detected format."""
    idx = obs_index.astype(str)
    if fmt == "barcode-suffix":
        return idx.str.split("-").str[0]
    elif fmt == "prefix-barcode":
        # Split on _ and take last part, then strip non-ACGTN suffix
        parts = idx.str.split("_").str[-1]
        return parts.str.extract(r"([ACGTN]{12,18})", expand=False).fillna(parts)
    elif fmt == "bare":
        return idx
    elif fmt == "colon-barcode":
        # PREFIX:BARCODE-N → extract BARCODE
        after_colon = idx.str.split(":").str[-1]
        return after_colon.str.split("-").str[0]
    elif fmt == "regex-extract":
        # Extract first ACGTN run of 12-18bp
        return idx.str.extract(r"([ACGTN]{12,18})", expand=False).fillna(idx)
    else:
        # Try multiple strategies
        for strategy in ["barcode-suffix", "prefix-barcode", "colon-barcode", "regex-extract"]:
            result = extract_bare_barcode(obs_index, strategy)
            if result.str.match(r"^[ACGTN]{12,18}$").mean() > 0.5:
                return result
        return idx

def main():
    h5ad_path = sys.argv[1]
    gse_id = sys.argv[2]
    dataset_dir = sys.argv[3]
    catalog_path = sys.argv[4]

    sample_col = None
    source_label = "gse_h5ad"
    for i, a in enumerate(sys.argv):
        if a == "--sample-col" and i + 1 < len(sys.argv):
            sample_col = sys.argv[i + 1]
        if a == "--source" and i + 1 < len(sys.argv):
            source_label = sys.argv[i + 1]

    # 1. Load obs
    print(f"Loading {os.path.basename(h5ad_path)}...")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    print(f"Cells: {len(obs)}, Columns: {len(obs.columns)}")

    # Count GSMs available
    gse_dir = os.path.join(dataset_dir, gse_id)
    if os.path.isdir(gse_dir):
        gsm_dirs = sorted([d for d in os.listdir(gse_dir) if d.startswith("GSM")])
    else:
        print(f"ERROR: {gse_dir} not found")
        sys.exit(1)
    n_gsms = len(gsm_dirs)

    # Handle existing 'barcode' column in obs — rename to avoid collision
    if "barcode" in obs.columns:
        obs = obs.rename(columns={"barcode": "author_barcode"})

    # 2. Detect sample column
    single_gsm_mode = (n_gsms == 1)
    if single_gsm_mode:
        print(f"Single-GSM mode: applying all metadata to {gsm_dirs[0]}")
        sample_col = None
        n_samples = 1
    else:
        if sample_col is None:
            sample_col = detect_sample_column(obs, n_gsms=n_gsms)
        if sample_col is None:
            print("ERROR: Could not detect sample column. Use --sample-col")
            sys.exit(1)
        n_samples = obs[sample_col].nunique()
        print(f"Sample column: '{sample_col}' ({n_samples} unique)")

    # 3. Detect and extract bare barcodes
    bc_fmt = detect_barcode_format(obs)
    print(f"Barcode format: {bc_fmt}")
    bare_bcs = extract_bare_barcode(obs.index.to_series(), bc_fmt)
    bc_match_rate = bare_bcs.str.match(r"^[ACGTN]{12,18}$").mean()
    print(f"Barcode extraction: {bc_match_rate:.1%} match ACGTN pattern")
    if bc_match_rate < 0.3:
        print("WARNING: Low barcode extraction rate. Index may not contain barcodes.")
    obs["bare_bc"] = bare_bcs.values

    # 4. Identify annotation columns (exclude QC/numeric/barcode)
    skip_set = {"n_genes", "n_counts", "n_molecules", "total_counts", "pct_",
                "percent_", "doublet", "mito", "ribo", "bare_bc", "score",
                "ncount", "nfeature", "log1p_"}
    if sample_col:
        skip_set.add(sample_col)
    meta_cols = []
    for c in obs.columns:
        cl = c.lower()
        if c in skip_set or any(p in cl for p in skip_set):
            continue
        if obs[c].dtype.name in ("float64", "float32", "int64", "int32"):
            # Keep if low cardinality (likely categorical encoded as int)
            if obs[c].nunique() > 200:
                continue
        meta_cols.append(c)
    print(f"Annotation columns ({len(meta_cols)}): {meta_cols}")

    # 5. Build per-sample barcode index
    if single_gsm_mode:
        # All cells go to the single GSM
        sg = obs[["bare_bc"] + meta_cols].copy()
        sg = sg.drop_duplicates(subset="bare_bc", keep="first")
        sg = sg.set_index("bare_bc")
        sample_groups = {"__all__": sg}
    else:
        sample_groups = {}
        for s in obs[sample_col].unique():
            sg = obs.loc[obs[sample_col] == s, ["bare_bc"] + meta_cols].copy()
            sg = sg.drop_duplicates(subset="bare_bc", keep="first")
            sg = sg.set_index("bare_bc")
            sample_groups[s] = sg

    print(f"\nMapping {len(gsm_dirs)} GSMs to {n_samples} samples...")

    mapping = {}
    for gsm in gsm_dirs:
        cells_path = os.path.join(gse_dir, gsm, "cells.parquet")
        if not os.path.exists(cells_path):
            continue
        our_cells = pd.read_parquet(cells_path, columns=["barcode"])
        our_bcs = set(our_cells["barcode"].astype(str).tolist())
        best_s, best_n = None, 0
        for s, sg in sample_groups.items():
            n = len(our_bcs & set(sg.index))
            if n > best_n:
                best_n = n
                best_s = s
        if best_n > 0:
            mapping[gsm] = dict(sample=best_s, overlap=best_n, n_ours=len(our_bcs))

    print(f"Mapped: {len(mapping)}/{len(gsm_dirs)}")

    # 7. Apply metadata
    results = []
    for gsm, m in sorted(mapping.items()):
        sg = sample_groups[m["sample"]]
        our_cells = pd.read_parquet(os.path.join(gse_dir, gsm, "cells.parquet"), columns=["barcode"])
        our_bcs = set(our_cells["barcode"].astype(str).tolist())

        aligned = sg.reindex(sorted(our_bcs))
        n_matched = aligned[meta_cols[0]].notna().sum() if meta_cols else 0
        match_rate = n_matched / len(our_bcs) if our_bcs else 0

        # Merge into author_metadata
        am_path = os.path.join(gse_dir, gsm, "author_metadata.parquet")
        new_cols = []
        if os.path.exists(am_path):
            existing = pd.read_parquet(am_path)
            if "barcode" in existing.columns:
                existing = existing.set_index("barcode")
            for col in meta_cols:
                if col not in existing.columns:
                    existing[col] = aligned[col].reindex(existing.index)
                    new_cols.append(col)
            out = existing.reset_index()
            if out.columns.duplicated().any():
                out = out.loc[:, ~out.columns.duplicated()]
            out.to_parquet(am_path, index=False)
        else:
            aligned.index.name = "barcode"
            aligned.reset_index().to_parquet(am_path, index=False)
            new_cols = meta_cols

        results.append(dict(gsm=gsm, sample=m["sample"], n_matched=int(n_matched),
                           n_ours=len(our_bcs), rate=round(match_rate, 4),
                           new_cols=len(new_cols)))

    rdf = pd.DataFrame(results)
    print(f"\nResults:")
    print(f"  Processed: {len(rdf)}")
    if len(rdf) > 0 and "rate" in rdf.columns:
        print(f"  Match rates: mean={rdf['rate'].mean():.3f}, median={rdf['rate'].median():.3f}")
        print(f"  >90%: {(rdf['rate'] > 0.9).sum()}, >50%: {(rdf['rate'] > 0.5).sum()}, >0%: {(rdf['rate'] > 0).sum()}")
    else:
        print("  No GSMs were matched.")

    # 8. Update catalog (only if rates improve)
    cat = pd.read_parquet(catalog_path)
    updated = 0
    for _, r in rdf.iterrows():
        mask = (cat["gse_id"] == gse_id) & (cat["gsm_id"] == r["gsm"])
        if mask.sum() == 0:
            continue
        idx = cat[mask].index[0]
        existing_rate = cat.loc[idx, "stage2a_match_rate"]
        if pd.notna(existing_rate) and existing_rate >= r["rate"]:
            continue  # Skip if existing rate is already better
        cat.loc[idx, "stage2a_status"] = "done"
        cat.loc[idx, "stage2a_format"] = "h5ad"
        cat.loc[idx, "stage2a_source"] = f"tierc:{source_label}"
        cat.loc[idx, "stage2a_n_cols"] = r["new_cols"]
        cat.loc[idx, "stage2a_match_rate"] = r["rate"]
        cat.loc[idx, "updated_at"] = pd.Timestamp.utcnow().isoformat()
        updated += 1

    cat.to_parquet(catalog_path, index=False)
    print(f"\nCatalog updated: {updated} GSMs")

    # Save results
    out_path = os.path.join(os.path.dirname(h5ad_path), f"{gse_id}_tier_c_results.csv")
    rdf.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
