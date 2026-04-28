#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/e2e_scanpy_reference.py
#
# End-to-end scanpy reference pipeline for cycle 49a integration correctness.
# Runs on GSM4037629 (11,560 cells) and writes binary outputs that the C++
# harness reads via numpy.load / npy binary readers.
#
# Stages produced:
#   1. stage1_shape.npy        — [n_cells, n_genes] int32 (from raw counts)
#   2. stage2_size_factors.npy — float32[n_cells]  (sc.pp.normalize_total size factors)
#   3. stage3_hvg_mask.npy     — bool[n_genes] (top-2000 HVGs, seurat_v3 flavor)
#   4. stage3_hvg_indices.npy  — int32[2000]   (sorted HVG gene indices)
#   5. stage4_pca_varexp.npy   — float32[30]   (explained variance ratio per PC)
#   6. stage5_knn_indices.npy  — int32[n_cells, 15]
#   7. stage5_knn_distances.npy— float32[n_cells, 15]
#   8. stage6_leiden_labels.npy— int32[n_cells]
#   9. stage7_umap_coords.npy  — float32[n_cells, 2]
#  10. stage8_de_lfc.npy       — float32[n_genes] (log2-FC for top-2 cluster comparison)
#  11. stage8_de_geneorder.npy — int32[100]       (top-100 gene indices by abs lfc)
#  12. stage9_marker_scores.npy— float32[n_cells, n_gene_sets]  (score_genes output)
#  13. stage9_geneset_labels.npy— byte string list written as JSON
#
# Usage (called as subprocess by integration_e2e_correctness.cpp):
#   python e2e_scanpy_reference.py \
#       --pz-path   <path/to/exon_counts.1pz>  \
#       --out-dir   <output_dir>               \
#       [--seed     <int>]                     \   default 42
#       [--n-hvg    <int>]                     \   default 2000
#       [--n-pcs    <int>]                     \   default 30
#       [--k-nn     <int>]                     \   default 15
#       [--leiden-res <float>]                 \   default 1.0
#       [--n-de-top  <int>]                    \   default 100
#
# Dependencies: scanpy, anndata, numpy, scipy, leidenalg, umap-learn
# The script does NOT import singlify — it uses scanpy's built-in 10x reader
# path after decoding the .1pz via singlify.io (if available) or by reading
# the pre-written CSC binary produced by the C++ harness (fallback path).
#
# Fallback path (no singlify.io): the C++ harness writes a CSCC binary at
# <out_dir>/raw_counts.bin before calling this script. This script reads it.
#
# The script exits 0 on success, non-zero on any failure.

import argparse
import json
import os
import sys
import struct
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Scanpy E2E reference pipeline")
    p.add_argument("--pz-path",   required=False, default="",
                   help="Path to exon_counts.1pz (used if singlify.io available)")
    p.add_argument("--csc-bin",   required=False, default="",
                   help="Pre-written CSCC binary from C++ (fallback if no singlify.io)")
    p.add_argument("--out-dir",   required=True,
                   help="Directory to write output .npy files")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--n-hvg",      type=int, default=2000)
    p.add_argument("--n-pcs",      type=int, default=30)
    p.add_argument("--k-nn",       type=int, default=15)
    p.add_argument("--leiden-res", type=float, default=1.0)
    p.add_argument("--n-de-top",   type=int, default=100)
    return p.parse_args()

# ---------------------------------------------------------------------------
# CSC binary reader (matches dump_csc.h format:
#   uint32 magic=0x43535343  uint32 n_rows  uint32 n_cols  uint64 nnz
#   float[nnz] values  int32[n_cols+1] indptr  int32[nnz] indices)
# ---------------------------------------------------------------------------

def read_csc_bin(path):
    with open(path, "rb") as f:
        magic, n_rows, n_cols = struct.unpack("<III", f.read(12))
        nnz, = struct.unpack("<Q", f.read(8))
        if magic != 0x43535343:
            raise ValueError(f"Bad magic in {path}: {magic:#010x}")
        values  = np.frombuffer(f.read(nnz * 4),       dtype=np.float32).copy()
        indptr  = np.frombuffer(f.read((n_cols+1) * 4), dtype=np.int32 ).copy()
        indices = np.frombuffer(f.read(nnz * 4),        dtype=np.int32 ).copy()
    import scipy.sparse as sp
    return sp.csc_matrix((values, indices, indptr), shape=(n_rows, n_cols))

# ---------------------------------------------------------------------------
# Load data into AnnData
# ---------------------------------------------------------------------------

def load_anndata(args):
    """
    Try to load via singlify.io; fall back to --csc-bin.
    Returns AnnData with raw integer counts in .X (CSC float32).
    """
    # Primary path: singlify.io Python wrapper
    if args.pz_path:
        try:
            import singlify.io as sio
            ad = sio.read_anndata(args.pz_path)
            # Ensure integer (or float) counts in .X
            import scipy.sparse as sp
            if not sp.issparse(ad.X):
                import scipy.sparse as sp2
                ad.X = sp2.csc_matrix(ad.X)
            print(f"[e2e_ref] Loaded via singlify.io: {ad.shape}", flush=True)
            return ad
        except Exception as e:
            print(f"[e2e_ref] singlify.io unavailable ({e}); "
                  "trying --csc-bin fallback", flush=True)

    # Fallback: pre-written CSC binary
    if args.csc_bin and os.path.exists(args.csc_bin):
        import anndata
        mat = read_csc_bin(args.csc_bin)
        ad = anndata.AnnData(X=mat.astype(np.float32))
        ad.obs_names = [f"cell_{i}" for i in range(ad.n_obs)]
        ad.var_names = [f"gene_{i}" for i in range(ad.n_vars)]
        print(f"[e2e_ref] Loaded via csc_bin: {ad.shape}", flush=True)
        return ad

    raise RuntimeError("No data source available: supply --pz-path (singlify.io) "
                       "or --csc-bin (pre-written CSCC binary).")

# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def save_npy(path, arr):
    np.save(path, arr)
    print(f"[e2e_ref] Wrote {os.path.basename(path)}: {arr.shape} {arr.dtype}",
          flush=True)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import scanpy as sc
    import scipy.sparse as sp

    # -----------------------------------------------------------------------
    # Stage 1 — Load raw counts
    # -----------------------------------------------------------------------
    print("[e2e_ref] Stage 1: loading data", flush=True)
    ad = load_anndata(args)
    n_cells, n_genes = ad.shape
    save_npy(os.path.join(args.out_dir, "stage1_shape.npy"),
             np.array([n_cells, n_genes], dtype=np.int32))
    print(f"[e2e_ref] Shape: {n_cells} cells × {n_genes} genes", flush=True)

    # -----------------------------------------------------------------------
    # Stage 2 — Log-normalize (total-count + log1p)
    # -----------------------------------------------------------------------
    print("[e2e_ref] Stage 2: log-normalize", flush=True)
    # sc.pp.normalize_total returns size factors if inplace=True + key_added.
    # Compute size factors explicitly as sum-per-cell / median(sum-per-cell).
    # This matches singlet-gpu's total-count normalization: s_i = sum_i / target_sum.
    raw_sums = np.asarray(ad.X.sum(axis=1)).ravel()  # n_cells
    target_sum = np.median(raw_sums)
    if target_sum == 0.0:
        target_sum = 1.0
    size_factors = raw_sums / target_sum          # what we divide each cell by
    save_npy(os.path.join(args.out_dir, "stage2_size_factors.npy"),
             size_factors.astype(np.float32))

    sc.pp.normalize_total(ad, target_sum=target_sum)
    sc.pp.log1p(ad)

    # -----------------------------------------------------------------------
    # Stage 3 — HVG (seurat_v3 flavor on log-normalized counts, top n-hvg)
    # -----------------------------------------------------------------------
    print(f"[e2e_ref] Stage 3: HVG top {args.n_hvg}", flush=True)
    sc.pp.highly_variable_genes(ad, n_top_genes=args.n_hvg, flavor="seurat_v3")
    hvg_mask = ad.var["highly_variable"].values.astype(bool)
    hvg_indices = np.where(hvg_mask)[0].astype(np.int32)
    save_npy(os.path.join(args.out_dir, "stage3_hvg_mask.npy"),    hvg_mask)
    save_npy(os.path.join(args.out_dir, "stage3_hvg_indices.npy"), hvg_indices)
    print(f"[e2e_ref] HVGs selected: {hvg_mask.sum()}", flush=True)

    # -----------------------------------------------------------------------
    # Stage 4 — PCA (randomized SVD on HVG subset, n_comps=n-pcs)
    # -----------------------------------------------------------------------
    print(f"[e2e_ref] Stage 4: PCA k={args.n_pcs}", flush=True)
    ad_hvg = ad[:, hvg_mask].copy()
    sc.pp.scale(ad_hvg, max_value=10)
    sc.tl.pca(ad_hvg, n_comps=args.n_pcs, svd_solver="randomized",
              random_state=args.seed)
    var_exp = ad_hvg.uns["pca"]["variance_ratio"].astype(np.float32)
    save_npy(os.path.join(args.out_dir, "stage4_pca_varexp.npy"), var_exp)
    # Reattach PCA to main AnnData for downstream use
    ad.obsm["X_pca"] = ad_hvg.obsm["X_pca"]

    # -----------------------------------------------------------------------
    # Stage 5 — kNN (k=k-nn on PCA embedding)
    # -----------------------------------------------------------------------
    print(f"[e2e_ref] Stage 5: kNN k={args.k_nn}", flush=True)
    sc.pp.neighbors(ad, n_neighbors=args.k_nn + 1, n_pcs=args.n_pcs,
                    use_rep="X_pca", random_state=args.seed)
    # Extract indices + distances from connectivity structure.
    # sc.pp.neighbors stores distances in adata.obsp["distances"].
    dist_mat = ad.obsp["distances"]   # CSR, n_cells × n_cells
    knn_indices   = np.zeros((n_cells, args.k_nn), dtype=np.int32)
    knn_distances = np.zeros((n_cells, args.k_nn), dtype=np.float32)
    dist_csr = dist_mat.tocsr()
    for i in range(n_cells):
        start, end = dist_csr.indptr[i], dist_csr.indptr[i+1]
        row_idx  = dist_csr.indices[start:end]
        row_dist = dist_csr.data[start:end]
        order = np.argsort(row_dist)[:args.k_nn]
        actual_k = len(order)
        knn_indices[i,   :actual_k] = row_idx[order].astype(np.int32)
        knn_distances[i, :actual_k] = row_dist[order].astype(np.float32)
    save_npy(os.path.join(args.out_dir, "stage5_knn_indices.npy"),   knn_indices)
    save_npy(os.path.join(args.out_dir, "stage5_knn_distances.npy"), knn_distances)

    # -----------------------------------------------------------------------
    # Stage 6 — Leiden clustering
    # -----------------------------------------------------------------------
    print(f"[e2e_ref] Stage 6: Leiden resolution={args.leiden_res}", flush=True)
    sc.tl.leiden(ad, resolution=args.leiden_res, random_state=args.seed,
                 key_added="leiden")
    leiden_labels = ad.obs["leiden"].astype(int).values.astype(np.int32)
    save_npy(os.path.join(args.out_dir, "stage6_leiden_labels.npy"), leiden_labels)
    n_clusters = int(leiden_labels.max()) + 1
    print(f"[e2e_ref] Leiden clusters: {n_clusters}", flush=True)

    # -----------------------------------------------------------------------
    # Stage 7 — UMAP
    # -----------------------------------------------------------------------
    print("[e2e_ref] Stage 7: UMAP", flush=True)
    sc.tl.umap(ad, random_state=args.seed)
    umap_coords = ad.obsm["X_umap"].astype(np.float32)
    save_npy(os.path.join(args.out_dir, "stage7_umap_coords.npy"), umap_coords)

    # -----------------------------------------------------------------------
    # Stage 8 — Wilcoxon DE between top-2 clusters by size
    # -----------------------------------------------------------------------
    print("[e2e_ref] Stage 8: Wilcoxon DE", flush=True)
    cluster_sizes = np.bincount(leiden_labels)
    top2 = np.argsort(cluster_sizes)[-2:]
    clust_a, clust_b = int(top2[1]), int(top2[0])  # largest, second-largest
    mask_a = leiden_labels == clust_a
    mask_b = leiden_labels == clust_b
    ad_sub = ad[mask_a | mask_b].copy()
    ad_sub.obs["group"] = np.where(
        (leiden_labels[mask_a | mask_b]) == clust_a, "A", "B"
    )
    sc.tl.rank_genes_groups(ad_sub, groupby="group", method="wilcoxon",
                            reference="B", key_added="de_wilcoxon",
                            pts=False)
    result = ad_sub.uns["de_wilcoxon"]
    # Gene names for group "A" — get log-fold-changes
    names_A  = np.array(result["names"]["A"],       dtype=str)
    lfc_A    = np.array(result["logfoldchanges"]["A"], dtype=np.float32)
    # Map back to global gene indices
    gene_name_to_idx = {g: i for i, g in enumerate(ad.var_names)}
    de_gene_indices = np.array([gene_name_to_idx.get(g, -1) for g in names_A],
                                dtype=np.int32)
    # Save full lfc array (indexed by global gene index)
    lfc_full = np.zeros(n_genes, dtype=np.float32)
    for gi, lfc in zip(de_gene_indices, lfc_A):
        if gi >= 0:
            lfc_full[gi] = lfc
    save_npy(os.path.join(args.out_dir, "stage8_de_lfc.npy"), lfc_full)
    # Top-100 gene indices by |lfc| among those that have non-zero lfc
    nonzero_mask = np.abs(lfc_full) > 0
    nonzero_idx  = np.where(nonzero_mask)[0]
    top100_local = np.argsort(np.abs(lfc_full[nonzero_idx]))[-args.n_de_top:][::-1]
    top100_genes = nonzero_idx[top100_local].astype(np.int32)
    save_npy(os.path.join(args.out_dir, "stage8_de_geneorder.npy"), top100_genes)
    # Save cluster pair for C++ harness
    save_npy(os.path.join(args.out_dir, "stage8_de_clusters.npy"),
             np.array([clust_a, clust_b], dtype=np.int32))
    print(f"[e2e_ref] DE between clusters {clust_a} vs {clust_b}", flush=True)

    # -----------------------------------------------------------------------
    # Stage 9 — Marker scoring (sc.tl.score_genes)
    # -----------------------------------------------------------------------
    print("[e2e_ref] Stage 9: marker scoring", flush=True)
    # Use the top-100 DE genes split into 4 gene sets of 25 each as our
    # gene set library (gives a deterministic, meaningful reference).
    n_sets    = 4
    set_size  = args.n_de_top // n_sets
    gene_sets = {}
    valid_var_names = list(ad.var_names)
    for s in range(n_sets):
        set_genes = []
        for gi in top100_genes[s * set_size : (s+1) * set_size]:
            if 0 <= gi < len(valid_var_names):
                set_genes.append(valid_var_names[gi])
        if set_genes:
            gene_sets[f"set_{s}"] = set_genes

    n_scored_sets = len(gene_sets)
    scores_matrix = np.zeros((n_cells, n_scored_sets), dtype=np.float32)
    set_names_list = []
    for si, (sname, sgenes) in enumerate(gene_sets.items()):
        sc.tl.score_genes(ad, gene_list=sgenes, score_name=f"_sg_{si}",
                          random_state=args.seed)
        scores_matrix[:, si] = ad.obs[f"_sg_{si}"].values.astype(np.float32)
        set_names_list.append(sname)

    save_npy(os.path.join(args.out_dir, "stage9_marker_scores.npy"), scores_matrix)
    # Write set names as JSON
    json_path = os.path.join(args.out_dir, "stage9_geneset_labels.json")
    with open(json_path, "w") as f:
        json.dump(set_names_list, f)
    print(f"[e2e_ref] Marker scores: {scores_matrix.shape}", flush=True)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("[e2e_ref] All stages complete.", flush=True)
    print(f"[e2e_ref] Outputs in: {args.out_dir}", flush=True)

    # Write a completion sentinel so C++ can verify the script finished.
    with open(os.path.join(args.out_dir, "done"), "w") as f:
        f.write("ok\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
