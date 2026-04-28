#!/usr/bin/env python3
"""Compare singlet-pileup SNP output against cellSNP-lite baseline.

Usage:
    python compare_cellsnp.py <cellsnp_dir> <pileup_dir>

Reads cellSNP-lite AD/DP matrices and singlet-pileup snp_ad/snp_dp matrices,
aligns them by SNP position and barcode, and reports concordance statistics.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def load_cellsnp(cellsnp_dir: Path):
    """Load cellSNP-lite AD and DP matrices."""
    ad_path = cellsnp_dir / "cellSNP.tag.AD.mtx"
    dp_path = cellsnp_dir / "cellSNP.tag.DP.mtx"
    bc_path = cellsnp_dir / "cellSNP.samples.tsv"

    ad = sio.mmread(str(ad_path)).tocsc()
    dp = sio.mmread(str(dp_path)).tocsc()
    barcodes = bc_path.read_text().strip().split("\n")

    # SNP positions from VCF
    vcf_path = cellsnp_dir / "cellSNP.base.vcf.gz"
    import gzip
    snp_positions = []
    with gzip.open(str(vcf_path), "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            chrom, pos, _, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            snp_positions.append(f"{chrom}:{pos}:{ref}>{alt}")

    return ad, dp, barcodes, snp_positions


def load_pileup(pileup_dir: Path):
    """Load singlet-pileup AD and DP matrices."""
    ad = sio.mmread(str(pileup_dir / "snp_ad.mtx")).tocsc()
    dp = sio.mmread(str(pileup_dir / "snp_dp.mtx")).tocsc()
    barcodes = (pileup_dir / "snp_ad_barcodes.tsv").read_text().strip().split("\n")
    features = (pileup_dir / "snp_ad_features.tsv").read_text().strip().split("\n")
    return ad, dp, barcodes, features


def main():
    parser = argparse.ArgumentParser(description="Compare pileup vs cellSNP-lite")
    parser.add_argument("cellsnp_dir", type=Path)
    parser.add_argument("pileup_dir", type=Path)
    args = parser.parse_args()

    print("Loading cellSNP-lite output...")
    cs_ad, cs_dp, cs_bc, cs_snps = load_cellsnp(args.cellsnp_dir)
    print(f"  cellSNP: {cs_ad.shape[0]} SNPs × {cs_ad.shape[1]} cells, AD nnz={cs_ad.nnz}, DP nnz={cs_dp.nnz}")

    print("Loading singlet-pileup output...")
    sp_ad, sp_dp, sp_bc, sp_snps = load_pileup(args.pileup_dir)
    print(f"  pileup:  {sp_ad.shape[0]} SNPs × {sp_ad.shape[1]} cells, AD nnz={sp_ad.nnz}, DP nnz={sp_dp.nnz}")

    # Align barcodes
    cs_bc_set = set(cs_bc)
    sp_bc_set = set(sp_bc)
    common_bc = sorted(cs_bc_set & sp_bc_set)
    print(f"\nBarcodes: cellSNP={len(cs_bc)}, pileup={len(sp_bc)}, overlap={len(common_bc)}")

    # Align SNP positions (normalize chromosome names: chr1 ↔ 1)
    def normalize_snp(s):
        """Normalize chr prefix: chr1:pos:R>A → 1:pos:R>A"""
        if s.startswith("chr"):
            chrom, rest = s.split(":", 1)
            bare = chrom[3:]
            if bare == "M":
                bare = "MT"
            return bare + ":" + rest
        return s

    # Index both by normalized name
    cs_norm = {normalize_snp(s): i for i, s in enumerate(cs_snps)}
    sp_norm = {normalize_snp(s): i for i, s in enumerate(sp_snps)}
    common_norm = sorted(set(cs_norm.keys()) & set(sp_norm.keys()))
    print(f"SNPs: cellSNP={len(cs_snps)}, pileup={len(sp_snps)}, overlap={len(common_norm)} (after chr-prefix normalization)")

    if not common_bc or not common_norm:
        print("ERROR: No overlap! Check coordinate systems and barcode formats.")
        sys.exit(1)

    # Build index maps
    cs_bc_idx = {bc: i for i, bc in enumerate(cs_bc)}
    sp_bc_idx = {bc: i for i, bc in enumerate(sp_bc)}

    # Extract common submatrices
    cs_bc_sel = [cs_bc_idx[bc] for bc in common_bc]
    sp_bc_sel = [sp_bc_idx[bc] for bc in common_bc]
    cs_snp_sel = [cs_norm[s] for s in common_norm]
    sp_snp_sel = [sp_norm[s] for s in common_norm]

    cs_ad_sub = cs_ad[cs_snp_sel, :][:, cs_bc_sel].toarray()
    sp_ad_sub = sp_ad[sp_snp_sel, :][:, sp_bc_sel].toarray()
    cs_dp_sub = cs_dp[cs_snp_sel, :][:, cs_bc_sel].toarray()
    sp_dp_sub = sp_dp[sp_snp_sel, :][:, sp_bc_sel].toarray()

    # Compare DP (total depth)
    dp_match = np.sum(cs_dp_sub == sp_dp_sub)
    dp_total = cs_dp_sub.size
    dp_corr = np.corrcoef(cs_dp_sub.ravel(), sp_dp_sub.ravel())[0, 1]
    print(f"\nDP comparison ({len(common_norm)} SNPs × {len(common_bc)} cells):")
    print(f"  Exact match: {dp_match}/{dp_total} ({100*dp_match/dp_total:.1f}%)")
    print(f"  Pearson r: {dp_corr:.6f}")
    print(f"  MAE: {np.mean(np.abs(cs_dp_sub.astype(float) - sp_dp_sub.astype(float))):.4f}")

    # Compare AD (alt allele depth)
    ad_match = np.sum(cs_ad_sub == sp_ad_sub)
    ad_total = cs_ad_sub.size
    ad_corr = np.corrcoef(cs_ad_sub.ravel(), sp_ad_sub.ravel())[0, 1] if np.any(cs_ad_sub) else 0
    print(f"\nAD comparison:")
    print(f"  Exact match: {ad_match}/{ad_total} ({100*ad_match/ad_total:.1f}%)")
    print(f"  Pearson r: {ad_corr:.6f}")
    print(f"  MAE: {np.mean(np.abs(cs_ad_sub.astype(float) - sp_ad_sub.astype(float))):.4f}")

    # Summary
    print(f"\n{'='*60}")
    if dp_corr > 0.99 and ad_corr > 0.99:
        print("PASS: Excellent concordance with cellSNP-lite baseline")
    elif dp_corr > 0.95:
        print("WARN: Good concordance but some differences (check MAPQ/BaseQ filters)")
    else:
        print("FAIL: Poor concordance — investigate alignment/filter differences")


if __name__ == "__main__":
    main()
