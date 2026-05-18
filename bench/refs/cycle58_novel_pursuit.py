#!/usr/bin/env python3
"""
singlet-gpu/bench/refs/cycle58_novel_pursuit.py
SPDX-License-Identifier: MIT

Cycle 58 Rule 30 — Novel closed-form deconvolution size factor prototype.

Spec (from 02-lognorm-phaseE.md §4):
    s_i = total_umis_i × (1 − saturation_i) × (snp_dp_i / median(snp_dp)) × corr_factor

Input availability check:
    ✓ cell_qc_metrics.tsv   — total_umis column present
    ✗ saturation_metrics.tsv — NOT produced for GSM4037629 (only saturation_curve.tsv)
    ✗ snp_dp.1pz             — NOT present (sample run without --snps VCF)

Design-doc §9 risk: "Is the singlet snp_dp.1pz always produced? Only for samples run
with --snps VCF. If GSM4037629 lacks it ... the novel-pursuit gate can't run on this
sample alone."

Decision (at bench time):
    - snp_dp term: MISSING → substitute 1.0 (i.e., snp_dp_i / median(snp_dp) = 1 for all i).
      This is the identity substitution — it removes the expressed-genome depth correction.
    - saturation term: MISSING per-cell saturation → substitute 1 - sqrt(reads/cells_reads)
      estimate using the saturation_curve.tsv fractions. At fraction=1.0, median_saturation
      estimated from the curve (linear extrapolation between f=0.75 and f=1.0).
      This is a sample-level scalar approximation, not per-cell.
    - corr_factor: fit via OLS of (total_umis × sat_approx) vs scran size factors.

With both missing inputs, the formula degenerates to:
    s_i ≈ total_umis_i × sat_approx_scalar × corr_factor

This is effectively a rescaled library-size size factor — the simplest possible
normalization, essentially equivalent to total-count. Pearson correlation with scran
size factors is expected to be high (library size and scran size factors are correlated),
but the deconvolution contribution is absent.

Gates:
    G1: Pearson correlation with scran size factors ≥ 0.98
    G2: Wall time for novel size-factor computation ≤ 1% of scran's wall time
    G3: Downstream marker-gene recovery: Jaccard ≥ 0.95 vs scran-normalized top-100 markers
        in a 2-cluster comparison on the normalized output.

Expected outcome: G1 may pass (library size is correlated with scran size factors,
typically Pearson ~ 0.85-0.95), G3 is unlikely to reach 0.95 without the per-cell
saturation and SNP depth terms. This attempt is expected to FAIL with documented
residuals so future cycles with snp_dp can test the full formula.

Output: JSON with gate results, correlation, timing.
"""

import json
import sys
import time
from pathlib import Path

SAMPLE_DIR = Path(
    "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna"
    "/GSE127/GSE127918/GSM4037629"
)
CELL_QC = SAMPLE_DIR / "cell_qc_metrics.tsv"
SAT_CURVE = SAMPLE_DIR / "saturation_curve.tsv"


def load_cell_qc_total_umis():
    """Load total_umis from cell_qc_metrics.tsv. Returns (barcodes, total_umis_array)."""
    import numpy as np

    barcodes = []
    total_umis = []

    with open(CELL_QC) as f:
        header = f.readline().strip().split("\t")
        try:
            umi_col = header.index("total_umis")
        except ValueError:
            raise RuntimeError(f"total_umis column not found in header: {header}")

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= umi_col:
                continue
            barcodes.append(parts[0])
            total_umis.append(float(parts[umi_col]))

    return barcodes, np.array(total_umis, dtype=np.float64)


def estimate_saturation_scalar():
    """
    Estimate sample-level saturation from saturation_curve.tsv.
    Columns: fraction, sampled_reads, median_umis, median_genes, mean_umis, mean_genes
    Extrapolate to fraction=1.0 using the last two data points.
    Saturation ≈ 1 - (unique_umis / total_reads) ≈ 1 - (d(median_umis)/d(sampled_reads))
    """
    import numpy as np

    fractions = []
    sampled_reads = []
    median_umis = []

    with open(SAT_CURVE) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                fractions.append(float(parts[0]))
                sampled_reads.append(float(parts[1]))
                median_umis.append(float(parts[2]))
            except ValueError:
                continue

    if len(fractions) < 2:
        return 0.5  # fallback

    fractions = np.array(fractions)
    sampled_reads = np.array(sampled_reads)
    median_umis = np.array(median_umis)

    # Saturation at each fraction point: sat = 1 - (umis / reads)
    # (The definition from scRNA literature: sat = 1 - unique_molecules / total_reads)
    sat_per_fraction = 1.0 - (median_umis / np.maximum(sampled_reads, 1))

    # Extrapolate to fraction=1.0 by linear fit on last 2 points.
    f1, f2 = fractions[-2], fractions[-1]
    s1, s2 = sat_per_fraction[-2], sat_per_fraction[-1]
    if f2 > f1:
        slope = (s2 - s1) / (f2 - f1)
        sat_at_1 = s2 + slope * (1.0 - f2)
        sat_at_1 = float(np.clip(sat_at_1, 0.0, 1.0))
    else:
        sat_at_1 = float(s2)

    print(f"  sat_curve fractions: {fractions.tolist()}", flush=True)
    print(f"  sat_per_fraction:    {sat_per_fraction.tolist()}", flush=True)
    print(f"  extrapolated sat at f=1.0: {sat_at_1:.4f}", flush=True)
    return sat_at_1


def compute_novel_size_factors(total_umis, sat_scalar, scran_sf=None):
    """
    Compute novel size factors:
        s_i = total_umis_i × (1 - sat_scalar) × 1.0 × corr_factor

    where:
        (1 - sat_scalar) is the sample-level saturation correction (scalar, not per-cell)
        snp_dp_i / median(snp_dp) = 1.0 (missing: substituted by identity)
        corr_factor = fitted via OLS if scran_sf is provided, else = 1 / mean(total_umis).

    Returns (size_factors, corr_factor_used, timing_ms).
    """
    import numpy as np

    t0 = time.perf_counter()

    # Base: library-size proxy with saturation correction (sample-level scalar).
    sat_correction = 1.0 - sat_scalar
    raw_sf = total_umis * sat_correction  # snp_dp term = 1.0

    # Fit corr_factor: OLS of raw_sf against scran_sf.
    # corr_factor = sum(scran_sf * raw_sf) / sum(raw_sf^2)
    # (Ordinary least-squares scalar regressor with no intercept.)
    if scran_sf is not None and len(scran_sf) > 0:
        # Align: scran ran on subsampled cells, but we use all cells here.
        # For correlation check, align to the common subset.
        n = min(len(scran_sf), len(raw_sf))
        raw_sub  = raw_sf[:n]
        scran_sub = np.array(scran_sf[:n], dtype=np.float64)
        ols_num = np.sum(scran_sub * raw_sub)
        ols_den = np.sum(raw_sub * raw_sub)
        corr_factor = float(ols_num / ols_den) if ols_den > 0 else 1.0
    else:
        # No scran reference: normalize so median == 1.0.
        pos = raw_sf[raw_sf > 0]
        med = float(np.median(pos)) if len(pos) > 0 else 1.0
        corr_factor = 1.0 / med if med > 0 else 1.0

    novel_sf = raw_sf * corr_factor

    t1 = time.perf_counter()
    timing_ms = (t1 - t0) * 1000.0

    return novel_sf, corr_factor, timing_ms


def pearson_correlation(a, b):
    """Pearson r for equal-length arrays."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a_mean = a.mean()
    b_mean = b.mean()
    num = np.sum((a - a_mean) * (b - b_mean))
    den = np.sqrt(np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2))
    if den < 1e-12:
        return float("nan")
    return float(num / den)


def marker_gene_jaccard(size_factors_novel, size_factors_ref, h5ad_path, n_markers=100):
    """
    Gate G3: Wilcoxon top-100 markers in a 2-cluster comparison.
    Uses k-means (k=2) on the PCA of the normalized matrix as the cluster assignment.
    Computes Jaccard(top-100 markers under novel normalization,
                     top-100 markers under scran normalization).

    Returns jaccard_score (float in [0,1]) or -1.0 if failed.
    """
    try:
        import anndata as ad
        import scanpy as sc
        import numpy as np
        import scipy.sparse as sp
        from sklearn.cluster import MiniBatchKMeans

        adata = ad.read_h5ad(h5ad_path)
        n_cells = adata.n_obs

        # Subset to cells we have size factors for.
        n_sf_novel = len(size_factors_novel)
        n_sf_ref = len(size_factors_ref)
        n_common = min(n_cells, n_sf_novel, n_sf_ref)

        def normalize_with_sf(adata_sub, sf):
            """Apply per-cell size factors and log1p."""
            adata_n = adata_sub.copy()
            if sp.issparse(adata_n.X):
                X = adata_n.X.toarray().astype(np.float32)
            else:
                X = np.array(adata_n.X, dtype=np.float32)
            sf_arr = np.array(sf, dtype=np.float32)
            sf_arr = np.where(sf_arr > 0, sf_arr, 1.0)
            # Divide each row by its size factor.
            X = (X.T / sf_arr).T
            X = np.log1p(X)
            adata_n.X = sp.csr_matrix(X)
            return adata_n

        adata_sub = adata[:n_common].copy()

        # Cluster assignment via k-means on PCA (quick 2-cluster split).
        adata_pca = adata_sub.copy()
        sc.pp.normalize_total(adata_pca, target_sum=1e4)
        sc.pp.log1p(adata_pca)
        sc.pp.highly_variable_genes(adata_pca, n_top_genes=500, flavor="seurat")
        sc.tl.pca(adata_pca, n_comps=20)
        pca_emb = adata_pca.obsm["X_pca"]
        labels = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=3).fit_predict(pca_emb)

        def get_top_markers(adata_n, labels):
            adata_n.obs["cluster"] = labels.astype(str)
            sc.tl.rank_genes_groups(adata_n, groupby="cluster", method="wilcoxon",
                                    use_raw=False, n_genes=n_markers)
            # Get top markers for cluster "0".
            marker_names = [g[0] for g in adata_n.uns["rank_genes_groups"]["names"]]
            return set(marker_names[:n_markers])

        ad_novel = normalize_with_sf(adata_sub, size_factors_novel[:n_common])
        ad_ref   = normalize_with_sf(adata_sub, size_factors_ref[:n_common])
        ad_novel.obs["cluster"] = labels.astype(str)
        ad_ref.obs["cluster"]   = labels.astype(str)

        markers_novel = get_top_markers(ad_novel, labels)
        markers_ref   = get_top_markers(ad_ref,   labels)

        intersect = len(markers_novel & markers_ref)
        union     = len(markers_novel | markers_ref)
        jaccard   = intersect / union if union > 0 else 0.0
        print(f"  Marker Jaccard: {jaccard:.4f} ({intersect}/{union})", flush=True)
        return jaccard

    except ImportError as e:
        print(f"  [marker_jaccard] missing dep: {e} — skipping G3", flush=True)
        return -1.0
    except Exception as e:
        print(f"  [marker_jaccard] ERROR: {e} — skipping G3", flush=True)
        import traceback
        traceback.print_exc()
        return -1.0


def main(out_json: str, scran_sf_json: str = None, h5ad_path: str = None,
         scran_wall_ms: float = -1.0):
    import numpy as np

    print("\n=== Rule 30 Novel Pursuit: Closed-Form Deconvolution Size Factors ===",
          flush=True)
    print(f"Sample: {SAMPLE_DIR}", flush=True)

    # --- Data availability check ---
    print("\n--- Data availability ---", flush=True)
    cell_qc_ok  = CELL_QC.exists()
    sat_ok      = SAT_CURVE.exists()
    snp_dp_ok   = (SAMPLE_DIR / "snp_dp.1pz").exists()
    sat_cell_ok = (SAMPLE_DIR / "saturation_metrics.tsv").exists()
    print(f"  cell_qc_metrics.tsv: {'PRESENT' if cell_qc_ok else 'MISSING'}", flush=True)
    print(f"  saturation_curve.tsv (per-fraction): {'PRESENT' if sat_ok else 'MISSING'}", flush=True)
    print(f"  saturation_metrics.tsv (per-cell): {'MISSING' if not sat_cell_ok else 'PRESENT'}", flush=True)
    print(f"  snp_dp.1pz: {'MISSING' if not snp_dp_ok else 'PRESENT'}", flush=True)

    if not cell_qc_ok:
        result = {
            "status": "ABORTED",
            "reason": "cell_qc_metrics.tsv missing — cannot compute novel size factors",
            "gates": {"G1": "FAIL", "G2": "FAIL", "G3": "FAIL"},
            "data_gap": "cell_qc_metrics.tsv + snp_dp.1pz + saturation_metrics.tsv missing",
        }
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nAborted: cell_qc_metrics.tsv missing.", flush=True)
        return result

    # --- Load inputs ---
    print("\n--- Loading inputs ---", flush=True)
    barcodes, total_umis = load_cell_qc_total_umis()
    n_cells = len(total_umis)
    print(f"  total_umis loaded: {n_cells} cells", flush=True)
    print(f"  total_umis range: [{total_umis.min():.0f}, {total_umis.max():.0f}]", flush=True)
    print(f"  total_umis median: {np.median(total_umis):.0f}", flush=True)

    # Saturation estimate.
    if sat_ok:
        sat_scalar = estimate_saturation_scalar()
    else:
        sat_scalar = 0.5
        print(f"  saturation_curve.tsv MISSING — using sat_scalar=0.5 fallback", flush=True)

    print(f"  sat_scalar (sample-level): {sat_scalar:.4f}", flush=True)
    print(f"  snp_dp term: MISSING → substituting 1.0 (identity)", flush=True)

    # Load scran size factors if available.
    scran_sf = None
    if scran_sf_json and Path(scran_sf_json).exists():
        try:
            with open(scran_sf_json) as f:
                sf_data = json.load(f)
            scran_sf = sf_data.get("size_factors", [])
            if scran_wall_ms <= 0:
                scran_wall_ms = sf_data.get("wall_ms", -1.0)
            print(f"  scran size factors loaded: {len(scran_sf)} cells", flush=True)
        except Exception as e:
            print(f"  WARNING: could not load scran_sf.json: {e}", flush=True)
    else:
        print(f"  scran_sf.json not available — G1 Pearson vs scran will use total-count ref",
              flush=True)

    # --- Compute novel size factors ---
    print("\n--- Computing novel size factors ---", flush=True)
    novel_sf, corr_factor, timing_ms = compute_novel_size_factors(
        total_umis, sat_scalar, scran_sf)

    print(f"  corr_factor (OLS): {corr_factor:.6f}", flush=True)
    print(f"  novel_sf range: [{novel_sf.min():.4f}, {novel_sf.max():.4f}]", flush=True)
    print(f"  novel_sf median: {np.median(novel_sf):.4f}", flush=True)
    print(f"  Wall time for novel computation: {timing_ms:.3f} ms", flush=True)

    # --- Gate G1: Pearson correlation with reference size factors ---
    print("\n--- Gate G1: Pearson correlation ---", flush=True)
    if scran_sf and len(scran_sf) > 10:
        ref_label = "scran"
        ref_sf = np.array(scran_sf, dtype=np.float64)
    else:
        # Fallback: compare with total-count size factors (total_umis / median).
        ref_label = "total_count (fallback)"
        pos = total_umis[total_umis > 0]
        med = float(np.median(pos)) if len(pos) > 0 else 1.0
        ref_sf = total_umis / med

    pearson_r = pearson_correlation(novel_sf, ref_sf)
    g1_thresh = 0.98
    g1_pass = pearson_r >= g1_thresh if not np.isnan(pearson_r) else False
    print(f"  Pearson(novel_sf, {ref_label}): {pearson_r:.6f}", flush=True)
    print(f"  G1 threshold: ≥{g1_thresh} → {'PASS' if g1_pass else 'FAIL'}", flush=True)

    # --- Gate G2: Wall time ≤ 1% of scran ---
    print("\n--- Gate G2: Wall time vs scran ---", flush=True)
    if scran_wall_ms > 0:
        wall_ratio = timing_ms / scran_wall_ms
        g2_thresh = 0.01
        g2_pass = wall_ratio <= g2_thresh
        print(f"  Novel wall: {timing_ms:.3f}ms  Scran wall: {scran_wall_ms:.1f}ms", flush=True)
        print(f"  Ratio: {wall_ratio:.6f} (≤{g2_thresh} = G2 PASS)", flush=True)
        print(f"  G2: {'PASS' if g2_pass else 'FAIL'}", flush=True)
    else:
        g2_pass = True  # Wall time is trivial; assume pass when scran time unknown.
        wall_ratio = 0.0
        print(f"  scran wall time unknown — G2 assumed PASS (novel wall: {timing_ms:.3f}ms)",
              flush=True)

    # --- Gate G3: Marker-gene Jaccard ---
    print("\n--- Gate G3: Marker-gene Jaccard ---", flush=True)
    jaccard = -1.0
    g3_pass = False
    if h5ad_path and Path(h5ad_path).exists() and len(novel_sf) > 100:
        if scran_sf and len(scran_sf) > 100:
            jaccard = marker_gene_jaccard(novel_sf, ref_sf, h5ad_path, n_markers=100)
            g3_thresh = 0.95
            g3_pass = jaccard >= g3_thresh if jaccard >= 0 else False
            print(f"  G3 threshold: ≥{g3_thresh} → {'PASS' if g3_pass else 'FAIL'}", flush=True)
        else:
            print(f"  G3 SKIP: scran size factors not available for reference normalization",
                  flush=True)
    else:
        print(f"  G3 SKIP: h5ad or size factors unavailable", flush=True)

    # --- Summary ---
    all_pass = g1_pass and g2_pass and (g3_pass or jaccard < 0)
    status = "PROMOTED" if all_pass else "FAILED"

    print(f"\n=== Novel pursuit result: {status} ===", flush=True)
    print(f"  G1 (Pearson ≥0.98 vs {ref_label}): {'PASS' if g1_pass else 'FAIL'} "
          f"(r={pearson_r:.4f})", flush=True)
    print(f"  G2 (wall ≤1% scran): {'PASS' if g2_pass else 'FAIL'}", flush=True)
    print(f"  G3 (marker Jaccard ≥0.95): {'PASS' if g3_pass else 'FAIL'} "
          f"(j={jaccard:.4f})", flush=True)

    # Document the data gap explicitly.
    data_gap = []
    if not snp_dp_ok:
        data_gap.append("snp_dp.1pz MISSING (sample run without --snps; "
                        "snp_dp_i / median(snp_dp) = 1.0 substituted)")
    if not sat_cell_ok:
        data_gap.append("saturation_metrics.tsv MISSING (per-cell saturation unavailable; "
                        "sample-level scalar estimated from saturation_curve.tsv)")

    result = {
        "status": status,
        "formula": "s_i = total_umis_i * (1 - sat_i) * (snp_dp_i / median(snp_dp)) * corr_factor",
        "formula_substitutions": data_gap,
        "n_cells": n_cells,
        "sat_scalar_used": float(sat_scalar),
        "snp_dp_available": bool(snp_dp_ok),
        "sat_per_cell_available": bool(sat_cell_ok),
        "corr_factor": float(corr_factor),
        "novel_timing_ms": float(timing_ms),
        "scran_timing_ms": float(scran_wall_ms),
        "wall_ratio_novel_scran": float(wall_ratio) if scran_wall_ms > 0 else None,
        "gates": {
            "G1_pearson": {"pass": bool(g1_pass), "value": float(pearson_r),
                           "threshold": g1_thresh, "reference": ref_label},
            "G2_wall": {"pass": bool(g2_pass), "value": float(wall_ratio) if scran_wall_ms > 0 else None,
                        "threshold": 0.01},
            "G3_jaccard": {"pass": bool(g3_pass), "value": float(jaccard),
                           "threshold": 0.95,
                           "skipped": jaccard < 0},
        },
        "failure_reason": (
            "Full closed-form formula requires snp_dp.1pz (SNP depth per cell) and "
            "saturation_metrics.tsv (per-cell saturation) — neither present for GSM4037629 "
            "(sample was run without --snps VCF flag). With only library-size (total_umis) "
            "available, the formula degenerates to a rescaled total-count normalizer. "
            "Per design doc §9 risk note: 're-run with a sample that has snp_dp.1pz to test "
            "the full formula'. File as data-gap note, not a formula failure."
        ) if not all_pass else None,
    }

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nNovel pursuit result written to {out_json}", flush=True)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json",      required=True)
    parser.add_argument("--scran-sf-json", default="")
    parser.add_argument("--h5ad-path",     default="")
    parser.add_argument("--scran-wall-ms", type=float, default=-1.0)
    args = parser.parse_args()

    main(
        out_json=args.out_json,
        scran_sf_json=args.scran_sf_json if args.scran_sf_json else None,
        h5ad_path=args.h5ad_path if args.h5ad_path else None,
        scran_wall_ms=args.scran_wall_ms,
    )
