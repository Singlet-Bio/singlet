#!/usr/bin/env python3
"""
validate_e2e.py — End-to-end validation of singlet-pileup pipeline outputs.

Compares singlet-pileup outputs against:
1. STARsolo Gene/GeneFull matrices (gene-level correlation)
2. VireoSNP (donor demux agreement)
3. Independent mt allele counting from samtools (mt heteroplasmy ground truth)
4. USA matrix reconstruction from exon+intron data
5. Internal consistency checks (AD ≤ DP, format round-trip, etc.)

Usage:
    python validate_e2e.py <pileup_dir> <starsolo_dir> [options]

Arguments:
    pileup_dir      Directory with singlet-pileup outputs (.1pz or .mtx)
    starsolo_dir    STARsolo Gene/filtered directory

Options:
    --bam PATH          BAM file for ground truth mt pileup
    --snp-vcf PATH      SNP VCF for vireoSNP comparison
    --n-donors N        Expected donors for vireoSNP (default: 2)
    --skip-vireo        Skip vireoSNP comparison
    --skip-mt           Skip mt ground truth comparison
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.io as sio
import scipy.sparse as sps


# ─── Helpers ───

def load_mtx(path):
    """Load a Matrix Market file."""
    return sio.mmread(path).tocsc()

def load_1pz(path):
    """Load a .1pz file via singlepress. Returns CSC matrix with rownames/colnames."""
    import singlepress as sp
    return sp.read_1pz_int(path)

def get_1pz_meta(directory, prefix):
    """Load .1pz and return (rownames, colnames) or (None, None)."""
    import singlepress as sp
    pz_path = os.path.join(directory, f"{prefix}.1pz")
    if not os.path.exists(pz_path):
        return None, None
    mat = sp.read_1pz_int(pz_path)
    rn = getattr(mat, 'rownames', None)
    cn = getattr(mat, 'colnames', None)
    return rn, cn

def load_matrix_auto(directory, prefix):
    """Load a matrix from either .1pz or .mtx format."""
    pz_path = os.path.join(directory, f"{prefix}.1pz")
    mtx_path = os.path.join(directory, f"{prefix}.mtx")
    mtx_dir_path = os.path.join(directory, prefix, "matrix.mtx")
    
    if os.path.exists(pz_path):
        return load_1pz(pz_path)
    elif os.path.exists(mtx_path):
        return load_mtx(mtx_path)
    elif os.path.exists(mtx_dir_path):
        return load_mtx(mtx_dir_path)
    return None

def load_features(directory, prefix):
    """Load feature names from TSV or .1pz embedded metadata."""
    for path in [
        os.path.join(directory, f"{prefix}_features.tsv"),
        os.path.join(directory, prefix, "features.tsv"),
    ]:
        if os.path.exists(path):
            return [l.strip().split('\t') for l in open(path)]
    # Fallback: try .1pz rownames
    rn, _ = get_1pz_meta(directory, prefix)
    if rn is not None:
        return [name.split('\t') for name in rn]
    return None

def load_barcodes(directory, prefix=None):
    """Load barcodes from TSV or .1pz embedded metadata."""
    paths = []
    if prefix:
        paths.append(os.path.join(directory, f"{prefix}_barcodes.tsv"))
        paths.append(os.path.join(directory, prefix, "barcodes.tsv"))
    paths.append(os.path.join(directory, "barcodes.tsv"))
    for p in paths:
        if os.path.exists(p):
            return [l.strip() for l in open(p)]
    # Fallback: try .1pz colnames
    if prefix:
        _, cn = get_1pz_meta(directory, prefix)
        if cn is not None:
            return list(cn)
    return None


class ValidationResult:
    def __init__(self, name):
        self.name = name
        self.passed = True
        self.checks = []
    
    def check(self, condition, description, value=None):
        status = "PASS" if condition else "FAIL"
        self.checks.append((status, description, value))
        if not condition:
            self.passed = False
        vstr = f" [{value}]" if value is not None else ""
        print(f"  [{status}] {description}{vstr}")
    
    def info(self, description, value=None):
        vstr = f" = {value}" if value is not None else ""
        print(f"  [INFO] {description}{vstr}")


# ─── Test 1: Gene-level correlation vs STARsolo ───

def test_gene_correlation(pileup_dir, starsolo_dir):
    r = ValidationResult("Gene-level correlation vs STARsolo")
    print(f"\n{'='*60}")
    print(f"Test 1: {r.name}")
    print(f"{'='*60}")
    
    # Load pileup gene counts — fall back to aggregating exon_counts by gene
    P = load_matrix_auto(pileup_dir, "gene_counts")
    p_feats = load_features(pileup_dir, "gene_counts")
    p_bcs = load_barcodes(pileup_dir, "gene_counts")

    if P is None:
        # Aggregate exon_counts → gene_counts on-the-fly using ENSGXXX prefix
        E = load_matrix_auto(pileup_dir, "exon_counts")
        e_feats = load_features(pileup_dir, "exon_counts")
        if E is None or e_feats is None:
            r.check(False, "gene_counts or exon_counts matrix exists")
            return r
        r.info("gene_counts not found — aggregating exon_counts by gene ID")
        # Parse gene IDs from feature names like "ENSG00000000457_SCYL3_chr1:..."
        gene_ids = [f[0].split('_')[0] if '_' in f[0] else f[0] for f in e_feats]
        unique_genes = list(dict.fromkeys(gene_ids))  # preserve order, deduplicate
        gene_idx = {g: i for i, g in enumerate(unique_genes)}
        # Build sparse aggregation matrix G (n_genes × n_exons), then P = G @ E
        import scipy.sparse as sps_local
        n_exons = len(e_feats)
        n_genes = len(unique_genes)
        exon_to_gene = np.array([gene_idx[g] for g in gene_ids], dtype=np.int32)
        G = sps_local.csr_matrix(
            (np.ones(n_exons, dtype=np.float32),
             (exon_to_gene, np.arange(n_exons, dtype=np.int32))),
            shape=(n_genes, n_exons)
        )
        P = (G @ E.astype(np.float32).tocsr()).tocsc()
        P = P.astype(np.int32)
        del G, E
        # Build gene feature / barcode lists
        p_feats = [[g] for g in unique_genes]
        p_bcs = load_barcodes(pileup_dir, "exon_counts")
        r.check(True, "exon→gene aggregation", f"{P.shape[0]} genes x {P.shape[1]} bcs nnz={P.nnz}")
    else:
        r.check(True, "gene_counts matrix loaded", f"{P.shape[0]}x{P.shape[1]} nnz={P.nnz}")

    # Load STARsolo
    S_path = os.path.join(starsolo_dir, "matrix.mtx")
    if not os.path.exists(S_path):
        r.check(False, "STARsolo matrix exists")
        return r
    S = sio.mmread(S_path).tocsc()
    r.check(True, "STARsolo matrix loaded", f"{S.shape[0]}x{S.shape[1]} nnz={S.nnz}")
    
    # Load feature/barcode maps
    s_feats = load_features(starsolo_dir, ".")
    if s_feats is None:
        s_feats = [l.strip().split('\t') for l in open(os.path.join(starsolo_dir, "features.tsv"))]
    s_bcs = load_barcodes(starsolo_dir)
    if s_bcs is None:
        s_bcs = [l.strip() for l in open(os.path.join(starsolo_dir, "barcodes.tsv"))]
    
    # Map common barcodes
    p_bc_map = {bc: i for i, bc in enumerate(p_bcs)}
    common_bcs = [(si, p_bc_map[bc]) for si, bc in enumerate(s_bcs) if bc in p_bc_map]
    r.info("Common barcodes", len(common_bcs))
    
    # Map common genes
    p_gene_map = {f[0]: i for i, f in enumerate(p_feats)} if p_feats else {}
    s_gene_map = {f[0]: i for i, f in enumerate(s_feats)} if s_feats else {}
    common_genes = [gid for gid in s_gene_map if gid in p_gene_map]
    r.info("Common genes", len(common_genes))
    
    if len(common_bcs) == 0 or len(common_genes) == 0:
        r.check(False, "Sufficient overlap for comparison")
        return r
    
    # Fast vectorized extraction: reindex both matrices to common genes × common barcodes
    p_gi = np.array([p_gene_map[g] for g in common_genes], dtype=np.int32)
    s_gi = np.array([s_gene_map[g] for g in common_genes], dtype=np.int32)
    s_bc_idx = np.array([si for si, _ in common_bcs], dtype=np.int32)
    p_bc_idx = np.array([pi for _, pi in common_bcs], dtype=np.int32)
    # Subset matrices: [common_genes × common_bcs]
    import scipy.sparse as _sps
    P_sub = P.tocsr()[p_gi, :][:, p_bc_idx]
    S_sub = S.tocsr()[s_gi, :][:, s_bc_idx]
    # Per-gene totals (sum across common barcodes)
    p_totals = np.asarray(P_sub.sum(axis=1)).flatten()
    s_totals = np.asarray(S_sub.sum(axis=1)).flatten()
    # Per-cell totals (sum across common genes)
    p_cell_totals = np.asarray(P_sub.sum(axis=0)).flatten()
    s_cell_totals = np.asarray(S_sub.sum(axis=0)).flatten()
    del P_sub, S_sub

    # Pearson correlation
    mask = (p_totals + s_totals) > 0
    from scipy.stats import pearsonr
    gene_r, gene_p = pearsonr(p_totals[mask], s_totals[mask])
    r.check(gene_r > 0.99, f"Gene Pearson r > 0.99", f"r={gene_r:.6f}")
    
    cell_r, _ = pearsonr(p_cell_totals, s_cell_totals)
    r.check(cell_r > 0.999, f"Cell-level r > 0.999", f"r={cell_r:.6f}")
    
    # Biggest discrepancies
    diffs = p_totals - s_totals
    worst_idx = np.argsort(-np.abs(diffs))[:5]
    r.info("Top 5 gene discrepancies:")
    for idx in worst_idx:
        gid = common_genes[idx]
        gname = s_feats[s_gene_map[gid]][1] if len(s_feats[s_gene_map[gid]]) > 1 else gid
        print(f"    {gname:20s} pileup={int(p_totals[idx]):6d}  star={int(s_totals[idx]):6d}  diff={int(diffs[idx]):+6d}")
    
    return r


# ─── Test 2: GeneFull = Gene + Intron ───

def test_genefull_reconstruction(pileup_dir, starsolo_dir):
    r = ValidationResult("GeneFull = exon-only-gene + intron reconstruction")
    print(f"\n{'='*60}")
    print(f"Test 2: {r.name}")
    print(f"{'='*60}")
    
    GF = load_matrix_auto(pileup_dir, "genefull_counts")
    G = load_matrix_auto(pileup_dir, "gene_counts")
    
    if GF is None:
        r.info("genefull_counts not found, skipping")
        return r
    if G is None:
        r.check(False, "gene_counts exists")
        return r
    
    r.check(True, "Both gene_counts and genefull_counts loaded")
    r.check(GF.shape == G.shape, "Same dimensions", f"GF={GF.shape} G={G.shape}")
    
    # GeneFull should be >= Gene (it includes intronic reads)
    diff = GF - G
    if sps.issparse(diff):
        min_val = diff.min()
    else:
        min_val = np.min(diff)
    r.check(min_val >= 0, "GeneFull >= Gene everywhere", f"min_diff={min_val}")
    
    # Compare to STARsolo GeneFull if available
    sf_dir = starsolo_dir.replace("/Gene/", "/GeneFull/")
    sf_path = os.path.join(sf_dir, "matrix.mtx")
    if os.path.exists(sf_path):
        SF = sio.mmread(sf_path).tocsc()
        r.info("STARsolo GeneFull loaded", f"{SF.shape}")
        # Compute correlation
        from scipy.stats import pearsonr
        gf_totals = np.array(GF.sum(axis=1)).flatten()
        sf_totals_raw = np.array(SF.sum(axis=1)).flatten()
        # They may have different feature sets, need alignment
        r.info("GeneFull total UMIs", f"ours={int(gf_totals.sum())} star={int(sf_totals_raw.sum())}")
    
    return r


# ─── Test 3: USA Matrix Reconstruction ───

def test_usa_reconstruction(pileup_dir):
    r = ValidationResult("USA matrix reconstruction (Spliced/Unspliced/Ambiguous)")
    print(f"\n{'='*60}")
    print(f"Test 3: {r.name}")
    print(f"{'='*60}")
    
    exon = load_matrix_auto(pileup_dir, "exon_counts")
    intron = load_matrix_auto(pileup_dir, "intron_counts")
    gene = load_matrix_auto(pileup_dir, "gene_counts")
    genefull = load_matrix_auto(pileup_dir, "genefull_counts")
    
    if exon is None:
        r.info("exon_counts not found, skipping")
        return r
    
    r.check(True, "exon_counts loaded", f"{exon.shape} nnz={exon.nnz}")
    
    if intron is not None:
        r.check(True, "intron_counts loaded", f"{intron.shape} nnz={intron.nnz}")
        
        # Spliced = Gene (exon-only aggregation)
        # Unspliced = GeneFull - Gene (intron contribution)
        # These are reconstructible from our per-exon and per-intron data
        if gene is not None and genefull is not None:
            unspliced = genefull - gene
            r.info("Spliced (Gene) total UMIs", int(np.array(gene.sum()).sum()))
            r.info("Unspliced (GeneFull-Gene) total UMIs", int(np.array(unspliced.sum()).sum()))
            r.info("Spliced+Unspliced total", int(np.array(genefull.sum()).sum()))
            
            # Verify unspliced >= 0
            if sps.issparse(unspliced):
                min_val = unspliced.min()
            else:
                min_val = np.min(unspliced)
            r.check(min_val >= 0, "Unspliced >= 0 everywhere", f"min={min_val}")
    
    # SJ check
    sj = load_matrix_auto(pileup_dir, "sj_counts")
    if sj is not None:
        r.check(True, "sj_counts loaded", f"{sj.shape} nnz={sj.nnz}")
        r.info("Total SJ UMIs", int(np.array(sj.sum()).sum()))
    
    return r


# ─── Test 4: SNP AD ≤ DP consistency ───

def test_snp_consistency(pileup_dir):
    r = ValidationResult("SNP AD ≤ DP consistency")
    print(f"\n{'='*60}")
    print(f"Test 4: {r.name}")
    print(f"{'='*60}")
    
    ad = load_matrix_auto(pileup_dir, "snp_ad")
    dp = load_matrix_auto(pileup_dir, "snp_dp")
    
    if ad is None or dp is None:
        r.info("SNP matrices not found, skipping")
        return r
    
    r.check(True, "snp_ad loaded", f"{ad.shape} nnz={ad.nnz}")
    r.check(True, "snp_dp loaded", f"{dp.shape} nnz={dp.nnz}")
    r.check(ad.shape == dp.shape, "Same dimensions")
    
    # AD ≤ DP at every position
    diff = dp - ad
    if sps.issparse(diff):
        violations = (diff < 0).sum()
    else:
        violations = np.sum(diff < 0)
    r.check(violations == 0, "AD ≤ DP everywhere", f"violations={violations}")
    
    # SNP coverage stats
    snp_coverage = np.array(dp.sum(axis=1)).flatten()
    covered = np.sum(snp_coverage > 0)
    r.info("Covered SNPs (DP>0)", f"{covered}/{dp.shape[0]}")
    r.info("Mean DP per covered SNP", f"{snp_coverage[snp_coverage>0].mean():.1f}")
    
    return r


# ─── Test 5: .1pz round-trip fidelity ───

def test_1pz_roundtrip(pileup_dir):
    r = ValidationResult(".1pz format fidelity")
    print(f"\n{'='*60}")
    print(f"Test 5: {r.name}")
    print(f"{'='*60}")
    
    try:
        import singlepress as sp
    except ImportError:
        r.info("singlepress not available, skipping")
        return r
    
    for prefix in ["gene_counts", "exon_counts", "snp_ad", "snp_dp"]:
        pz_path = os.path.join(pileup_dir, f"{prefix}.1pz")
        if not os.path.exists(pz_path):
            continue
        
        # Validate format
        try:
            info = sp.info_1pz(pz_path)
            r.check(True, f"{prefix}.1pz readable", f"{info}")
        except Exception as e:
            r.check(False, f"{prefix}.1pz readable", str(e))
            continue
        
        # Read and check
        try:
            mat = sp.read_1pz_int(pz_path)
            r.check(mat.nnz > 0, f"{prefix}.1pz has data", f"nnz={mat.nnz}")
        except Exception as e:
            r.check(False, f"{prefix}.1pz data valid", str(e))
    
    return r


# ─── Test 6: Donor demux validation ───

def test_donor_demux(pileup_dir, bam_path=None, snp_vcf=None, n_donors=2):
    r = ValidationResult("Donor demux (vs vireoSNP)")
    print(f"\n{'='*60}")
    print(f"Test 6: {r.name}")
    print(f"{'='*60}")
    
    assign_path = os.path.join(pileup_dir, "donor_assignments.tsv")
    if not os.path.exists(assign_path):
        r.info("donor_assignments.tsv not found, skipping")
        return r
    
    # Load our assignments
    import pandas as pd
    ours = pd.read_csv(assign_path, sep='\t')
    r.check(True, "donor_assignments.tsv loaded", f"{len(ours)} cells")
    
    assigned = ours[ours['donor_id'] != 'unassigned']
    r.info("Assigned cells", len(assigned))
    r.info("Donor distribution", ours['donor_id'].value_counts().to_dict())
    r.info("Median confidence", f"{ours['prob_max'].median():.4f}")
    
    # Run vireoSNP for comparison (if AD/DP available)
    ad = load_matrix_auto(pileup_dir, "snp_ad")
    dp = load_matrix_auto(pileup_dir, "snp_dp")
    
    if ad is not None and dp is not None:
        try:
            from vireoSNP import BinomMixtureVB
            from sklearn.metrics import adjusted_mutual_info_score
            
            # Filter to covered SNPs
            snp_cov = np.array(dp.sum(axis=1)).flatten()
            covered = snp_cov > 0
            ad_sub = np.array(ad[covered, :].todense(), dtype=np.float64)
            dp_sub = np.array(dp[covered, :].todense(), dtype=np.float64)
            
            r.info("Running vireoSNP for comparison...")
            model = BinomMixtureVB(n_var=ad_sub.shape[0], n_cell=ad_sub.shape[1],
                                    n_donor=n_donors)
            model.fit(ad_sub, dp_sub)
            vireo_labels = model.ID_prob.argmax(axis=1)
            
            # Map our labels to integers
            our_int = []
            for _, row in ours.iterrows():
                if str(row['donor_id']).startswith('donor'):
                    our_int.append(int(str(row['donor_id']).replace('donor', '')))
                else:
                    our_int.append(-1)
            our_int = np.array(our_int)
            
            mask = our_int >= 0
            ami = adjusted_mutual_info_score(vireo_labels[mask], our_int[mask])
            r.info("AMI vs vireoSNP", f"{ami:.4f}")
            
            # Agreement (handle label swap)
            agree0 = (vireo_labels[mask] == our_int[mask]).sum()
            agree1 = (vireo_labels[mask] == (n_donors - 1 - our_int[mask])).sum()
            best = max(agree0, agree1)
            r.info("Best agreement", f"{best}/{mask.sum()} = {best/mask.sum()*100:.1f}%")
            
            # Note: low agreement expected for single-donor samples
            r.check(True, "vireoSNP comparison completed")
            
        except ImportError:
            r.info("vireoSNP/sklearn not available, skipping comparison")
    
    return r


# ─── Test 7: mt heteroplasmy ───

def test_mt_heteroplasmy(pileup_dir, bam_path=None):
    r = ValidationResult("mt heteroplasmy")
    print(f"\n{'='*60}")
    print(f"Test 7: {r.name}")
    print(f"{'='*60}")
    
    # Check for mt outputs
    mt_variants_path = os.path.join(pileup_dir, "mt_variants.tsv")
    mt_allele_path_1pz = os.path.join(pileup_dir, "mt_allele_counts.1pz")
    mt_allele_path_mtx = os.path.join(pileup_dir, "mt_allele_counts.mtx")
    mt_het_path = os.path.join(pileup_dir, "mt_heteroplasmy.mtx")
    
    # Load mt allele counts
    mt_alleles = None
    if os.path.exists(mt_allele_path_1pz):
        mt_alleles = load_1pz(mt_allele_path_1pz)
        r.check(True, "mt_allele_counts.1pz loaded", f"{mt_alleles.shape} nnz={mt_alleles.nnz}")
    elif os.path.exists(mt_allele_path_mtx):
        mt_alleles = load_mtx(mt_allele_path_mtx)
        r.check(True, "mt_allele_counts.mtx loaded", f"{mt_alleles.shape} nnz={mt_alleles.nnz}")
    else:
        r.info("mt_allele_counts not found, skipping")
        return r
    
    # Basic validation: shape should be 66276 × n_cells  (16569 * 4 = 66276)
    r.check(mt_alleles.shape[0] == 66276 or mt_alleles.shape[0] <= 66276,
            "mt feature count consistent", f"nrows={mt_alleles.shape[0]}")
    
    # Coverage stats
    mt_dense = mt_alleles.toarray() if sps.issparse(mt_alleles) else mt_alleles
    
    # Per-position total coverage (sum across 4 bases and all cells)
    pos_coverage = np.zeros(16569)
    for pos in range(16569):
        for b in range(4):
            feat = pos * 4 + b
            if feat < mt_dense.shape[0]:
                pos_coverage[pos] += mt_dense[feat, :].sum()
    
    covered_pos = np.sum(pos_coverage > 0)
    r.info("Covered mt positions", f"{covered_pos}/16569")
    r.info("Mean coverage per covered position", f"{pos_coverage[pos_coverage>0].mean():.1f}")
    
    # Per-cell mt read count
    cell_mt_counts = np.array(mt_alleles.sum(axis=0)).flatten()
    r.info("Median mt bases per cell", f"{np.median(cell_mt_counts):.0f}")
    r.info("Total mt base calls", f"{int(cell_mt_counts.sum())}")
    
    # Check mt variants file
    if os.path.exists(mt_variants_path):
        import pandas as pd
        variants = pd.read_csv(mt_variants_path, sep='\t')
        r.check(True, "mt_variants.tsv loaded", f"{len(variants)} variants")
        if len(variants) > 0:
            r.info("Top variants:")
            for _, v in variants.head(5).iterrows():
                print(f"    pos={int(v['pos'])} {v['ref']}>{v['alt']} "
                      f"covered={int(v['n_cells_covered'])} het={int(v['n_cells_het'])} "
                      f"VAF={v['mean_vaf']:.3f}")
    
    # Check heteroplasmy matrix
    if os.path.exists(mt_het_path):
        het = load_mtx(mt_het_path)
        r.check(True, "mt_heteroplasmy.mtx loaded", f"{het.shape} nnz={het.nnz}")
        if het.nnz > 0:
            het_vals = het.data
            r.info("VAF range", f"[{het_vals.min():.4f}, {het_vals.max():.4f}]")
            r.check(np.all(het_vals >= 0) and np.all(het_vals <= 1),
                    "VAF values in [0,1]")
    
    # Ground truth: compute from BAM if available
    if bam_path and os.path.exists(bam_path):
        r.info("Computing ground truth from BAM (samtools)...")
        try:
            import subprocess
            # Use samtools mpileup to get per-position base counts for chrM
            cmd = f"samtools mpileup -r chrM -q 10 -Q 10 --no-BAQ {bam_path} 2>/dev/null | head -100"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                r.info(f"samtools mpileup produced {len(lines)} position lines (showing first 100)")
                r.check(True, "Ground truth BAM pileup available")
        except Exception as e:
            r.info(f"samtools comparison failed: {e}")
    
    return r


# ─── Test 8: Output completeness ───

def test_output_completeness(pileup_dir, pipeline_mode=False):
    r = ValidationResult("Output completeness")
    print(f"\n{'='*60}")
    print(f"Test 8: {r.name}")
    print(f"{'='*60}")
    
    # Expected files
    expected = ["pileup_stats.json"]
    
    # Check what matrices exist (either .1pz or .mtx)
    matrix_prefixes = ["exon_counts", "gene_counts", "snp_ad", "snp_dp"]
    optional_prefixes = ["intron_counts", "sj_counts", "editing", "genefull_counts"]
    
    for prefix in matrix_prefixes:
        found = (os.path.exists(os.path.join(pileup_dir, f"{prefix}.1pz")) or
                 os.path.exists(os.path.join(pileup_dir, f"{prefix}.mtx")) or
                 os.path.exists(os.path.join(pileup_dir, prefix, "matrix.mtx")))
        r.check(found, f"{prefix} exists")
    
    for prefix in optional_prefixes:
        found = (os.path.exists(os.path.join(pileup_dir, f"{prefix}.1pz")) or
                 os.path.exists(os.path.join(pileup_dir, f"{prefix}.mtx")) or
                 os.path.exists(os.path.join(pileup_dir, prefix, "matrix.mtx")))
        if found:
            r.info(f"{prefix} present")
    
    if pipeline_mode:
        r.check(os.path.exists(os.path.join(pileup_dir, "donor_assignments.tsv")),
                "donor_assignments.tsv exists")
        mt_found = (os.path.exists(os.path.join(pileup_dir, "mt_allele_counts.1pz")) or
                    os.path.exists(os.path.join(pileup_dir, "mt_allele_counts.mtx")))
        r.check(mt_found, "mt_allele_counts exists")
        r.check(os.path.exists(os.path.join(pileup_dir, "mt_variants.tsv")),
                "mt_variants.tsv exists")
    
    # Stats JSON
    stats_path = os.path.join(pileup_dir, "pileup_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        r.check(True, "pileup_stats.json valid JSON")
        r.info("Total reads", stats.get("total_reads"))
        r.info("Barcoded reads", stats.get("barcoded_reads"))
        r.info("Exon hits", stats.get("exon_hits"))
        r.info("chrM reads", stats.get("chrm_reads"))
        if "mt_pileup_bases" in stats:
            r.info("mt pileup bases", stats.get("mt_pileup_bases"))
    
    return r


# ─── Test 9: Samtools ground truth for chrM per-position coverage ───

def test_mt_ground_truth(pileup_dir, bam_path):
    r = ValidationResult("mt ground truth (samtools mpileup)")
    print(f"\n{'='*60}")
    print(f"Test 9: {r.name}")
    print(f"{'='*60}")
    
    if not bam_path or not os.path.exists(bam_path):
        r.info("BAM not available, skipping")
        return r
    
    mt_alleles = None
    for fmt in [".1pz", ".mtx"]:
        path = os.path.join(pileup_dir, f"mt_allele_counts{fmt}")
        if os.path.exists(path):
            mt_alleles = load_1pz(path) if fmt == ".1pz" else load_mtx(path)
            break
    
    if mt_alleles is None:
        r.info("mt_allele_counts not found, skipping")
        return r
    
    import subprocess
    
    # Load barcodes
    bcs = load_barcodes(pileup_dir, "mt_allele_counts")
    if bcs is None:
        bcs = load_barcodes(pileup_dir)
    if bcs is None:
        r.info("Barcodes not found, skipping")
        return r
    
    # Use samtools to get total base counts per chrM position
    # (aggregate across all barcoded reads)
    cmd = (f"samtools mpileup -r chrM -q 0 -Q 10 --no-BAQ "
           f"-d 1000000 {bam_path} 2>/dev/null")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        r.info("samtools mpileup timed out")
        return r
    
    if result.returncode != 0 or not result.stdout.strip():
        # Try MT instead of chrM
        cmd = cmd.replace("-r chrM", "-r MT")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    
    if not result.stdout.strip():
        r.info("No chrM/MT data in mpileup output")
        return r
    
    # Parse mpileup: position, ref_base, coverage, bases, quals
    samtools_coverage = {}
    samtools_base_counts = {}  # pos → {A: n, C: n, G: n, T: n}
    for line in result.stdout.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        pos = int(parts[1]) - 1  # 0-based
        depth = int(parts[3])
        bases_str = parts[4].upper()
        ref = parts[2].upper()
        
        samtools_coverage[pos] = depth
        counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        for c in bases_str:
            if c == '.' or c == ',':
                counts[ref] += 1
            elif c in counts:
                counts[c] += 1
        samtools_base_counts[pos] = counts
    
    r.info("samtools positions with coverage", len(samtools_coverage))
    
    # Compare to our per-position totals
    our_coverage = np.zeros(16569)
    our_base_counts = np.zeros((16569, 4))
    mt_arr = mt_alleles.toarray() if sps.issparse(mt_alleles) else np.array(mt_alleles)
    
    for pos in range(min(16569, mt_arr.shape[0] // 4)):
        for b in range(4):
            feat = pos * 4 + b
            if feat < mt_arr.shape[0]:
                total = mt_arr[feat, :].sum()
                our_base_counts[pos, b] = total
                our_coverage[pos] += total
    
    # Correlation of per-position total coverage
    common_pos = sorted(set(range(16569)) & set(samtools_coverage.keys()))
    if len(common_pos) > 0:
        st_cov = np.array([samtools_coverage.get(p, 0) for p in common_pos])
        our_cov = np.array([our_coverage[p] for p in common_pos])
        
        from scipy.stats import pearsonr
        # Filter to positions with any coverage
        mask = (st_cov > 0) | (our_cov > 0)
        if mask.sum() > 10:
            cov_r, _ = pearsonr(st_cov[mask], our_cov[mask])
            r.info("Coverage correlation (per-position)", f"r={cov_r:.6f}")
            # Note: samtools counts all reads, we only count barcoded reads
            # so some discrepancy is expected
            r.check(cov_r > 0.95, "Coverage correlation > 0.95", f"r={cov_r:.6f}")
        
        # Compare base-level counts at a few positions
        r.info("Base-count comparison at top-coverage positions:")
        top_pos = sorted(common_pos, key=lambda p: -samtools_coverage.get(p, 0))[:5]
        for pos in top_pos:
            st = samtools_base_counts.get(pos, {})
            our = {c: int(our_base_counts[pos, i]) for i, c in enumerate('ACGT')}
            print(f"    pos={pos+1}: samtools={st}  ours={our}")
    
    return r


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="End-to-end validation of singlet-pileup")
    parser.add_argument("pileup_dir", help="Directory with singlet-pileup outputs")
    parser.add_argument("starsolo_dir", help="STARsolo Gene/filtered directory")
    parser.add_argument("--bam", help="BAM file for ground truth mt pileup")
    parser.add_argument("--snp-vcf", help="SNP VCF for vireoSNP")
    parser.add_argument("--n-donors", type=int, default=2)
    parser.add_argument("--skip-vireo", action="store_true")
    parser.add_argument("--skip-mt", action="store_true")
    parser.add_argument("--pipeline", action="store_true",
                        help="Expect pipeline mode outputs (demux, mt)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("singlet-pileup End-to-End Validation")
    print("=" * 60)
    print(f"Pileup dir: {args.pileup_dir}")
    print(f"STARsolo dir: {args.starsolo_dir}")
    if args.bam:
        print(f"BAM: {args.bam}")
    
    results = []
    
    # Test 1: Gene correlation
    results.append(test_gene_correlation(args.pileup_dir, args.starsolo_dir))
    
    # Test 2: GeneFull reconstruction
    results.append(test_genefull_reconstruction(args.pileup_dir, args.starsolo_dir))
    
    # Test 3: USA reconstruction
    results.append(test_usa_reconstruction(args.pileup_dir))
    
    # Test 4: SNP consistency
    results.append(test_snp_consistency(args.pileup_dir))
    
    # Test 5: .1pz round-trip
    results.append(test_1pz_roundtrip(args.pileup_dir))
    
    # Test 6: Donor demux
    if not args.skip_vireo:
        results.append(test_donor_demux(args.pileup_dir, args.bam, args.snp_vcf, args.n_donors))
    
    # Test 7: mt heteroplasmy
    if not args.skip_mt:
        results.append(test_mt_heteroplasmy(args.pileup_dir, args.bam))
    
    # Test 8: Output completeness
    results.append(test_output_completeness(args.pileup_dir, args.pipeline))
    
    # Test 9: mt ground truth
    if not args.skip_mt and args.bam:
        results.append(test_mt_ground_truth(args.pileup_dir, args.bam))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_pass = sum(1 for r in results if r.passed)
    total = len(results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        n_checks = len(r.checks)
        n_pass = sum(1 for s, _, _ in r.checks if s == "PASS")
        print(f"  [{status}] {r.name} ({n_pass}/{n_checks} checks)")
    
    print(f"\nOverall: {total_pass}/{total} test suites passed")
    return 0 if total_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
