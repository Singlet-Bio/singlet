#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/wgcna_py_reference.py
#
# Reference implementation for network/hdwgcna correctness harness (cycle 46).
#
# Dispatch: called by network_hdwgcna_correctness.cpp via subprocess.
#
# Exit codes:
#   0  — WGCNA/hdWGCNA R ran successfully; reference written to --out_dir
#   2  — R/WGCNA not available; pure-Python scipy fallback ran; reference written
#   1  — unexpected error
#
# Primary path: R WGCNA classic (Langfelder & Horvath 2008) via Rscript.
#   Requires: R >= 4.0, WGCNA (Bioconductor), hdWGCNA optional for the full
#   hdWGCNA path. The R sub-script is tests/refs/hdwgcna_r_reference.R.
#
# Fallback path (always present): pure-Python scipy hierarchy.
#   Steps match the algorithm spec in state/designs/46-hdwgcna.md:
#   1. Pearson correlation matrix via numpy.
#   2. Soft power: |r|^beta, beta=6.
#   3. Topological overlap matrix (TOM) via the WGCNA formula.
#   4. Hierarchical clustering: scipy.cluster.hierarchy.linkage(1-TOM, 'average').
#   5. Dynamic tree cutting: simple height threshold cut.
#   6. Eigengene: first PC of expression submatrix (scipy.sparse.linalg.svds, k=1).
#   7. Hub gene kME: Pearson(gene, eigengene).
#
# Input files (from --data_dir, written by C++ test):
#   expression.bin   — [uint32_t n_cells][uint32_t n_genes]
#                      [float × n_cells × n_genes]  (row-major, dense)
#   soft_power.bin   — [uint32_t beta]  (soft threshold power, default 6)
#
# Output files (to --out_dir):
#   py_module_labels.bin    — [uint32_t n_genes][int32_t × n_genes]  (0=unassigned)
#   py_tom.bin              — [uint32_t n_genes][float × n_genes × n_genes] (row-major)
#   py_eigengenes.bin       — [uint32_t n_cells][uint32_t n_modules]
#                             [float × n_cells × n_modules] (row-major)
#   py_hub_kme.bin          — [uint32_t n_hub_entries]
#                             [int32_t module][int32_t gene][float kme] × n_hub_entries
#
# Usage:
#   python3 wgcna_py_reference.py --task tiny_synthetic \
#       --data_dir /tmp/... --out_dir /tmp/... [--use_r]

import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
from scipy.cluster import hierarchy as sch
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="hdWGCNA / WGCNA Python reference")
    p.add_argument("--task", default="tiny_synthetic",
                   help="task tag (used only for logging)")
    p.add_argument("--data_dir", required=True,
                   help="directory containing expression.bin, soft_power.bin")
    p.add_argument("--out_dir", required=True,
                   help="directory to write reference output files")
    p.add_argument("--use_r", action="store_true",
                   help="try to delegate to hdwgcna_r_reference.R first")
    p.add_argument("--r_script",
                   default=os.path.join(os.path.dirname(__file__),
                                        "hdwgcna_r_reference.R"),
                   help="path to the R reference script")
    p.add_argument("--height_cut", type=float, default=0.9,
                   help="dendrogram cut height for dynamic tree cutting (fallback)")
    p.add_argument("--min_module_size", type=int, default=5,
                   help="minimum genes per module")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Binary I/O helpers
# ---------------------------------------------------------------------------

def read_expression(data_dir):
    path = os.path.join(data_dir, "expression.bin")
    with open(path, "rb") as f:
        n_cells, n_genes = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(n_cells * n_genes * 4), dtype=np.float32)
    return data.reshape(n_cells, n_genes).copy()  # cells × genes


def read_soft_power(data_dir):
    path = os.path.join(data_dir, "soft_power.bin")
    with open(path, "rb") as f:
        (beta,) = struct.unpack("<I", f.read(4))
    return int(beta)


def write_module_labels(out_dir, labels):
    """labels: int32 array of length n_genes (0=unassigned)."""
    n = len(labels)
    path = os.path.join(out_dir, "py_module_labels.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(np.asarray(labels, dtype=np.int32).tobytes())


def write_tom(out_dir, tom):
    """tom: float32 n_genes × n_genes row-major."""
    n = tom.shape[0]
    path = os.path.join(out_dir, "py_tom.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(tom.astype(np.float32).tobytes())


def write_eigengenes(out_dir, eigengenes):
    """eigengenes: float32 n_cells × n_modules row-major."""
    n_cells, n_mods = eigengenes.shape
    path = os.path.join(out_dir, "py_eigengenes.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n_cells, n_mods))
        f.write(eigengenes.astype(np.float32).tobytes())


def write_hub_kme(out_dir, hub_entries):
    """hub_entries: list of (module_idx, gene_idx, kme_float)."""
    n = len(hub_entries)
    path = os.path.join(out_dir, "py_hub_kme.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        for (mod, gene, kme) in hub_entries:
            f.write(struct.pack("<iif", mod, gene, kme))


# ---------------------------------------------------------------------------
# Core WGCNA algorithm (scipy fallback)
# ---------------------------------------------------------------------------

def compute_pearson_matrix(expr):
    """expr: cells × genes float32. Returns genes × genes float32 correlation."""
    # Center per gene
    g = expr.T  # genes × cells
    g_centered = g - g.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(g_centered, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    g_norm = g_centered / norms
    corr = g_norm @ g_norm.T  # genes × genes
    corr = np.clip(corr, -1.0, 1.0)
    return corr.astype(np.float32)


def compute_tom(corr, beta):
    """Compute Topological Overlap Matrix from |r|^beta adjacency."""
    adj = np.abs(corr) ** beta  # soft power adjacency

    # Degree (row sum excluding diagonal)
    n = adj.shape[0]
    k = adj.sum(axis=1) - np.diag(adj)  # length-n vector

    # TOM numerator: adj @ adj + adj[i,j]
    # TOM[i,j] = (L[i,j] + adj[i,j]) / (min(k_i, k_j) + 1 - adj[i,j])
    L = adj @ adj  # n×n: sum_k(adj[i,k]*adj[j,k])
    L_diag = np.diag(L)

    num = L + adj  # n×n

    ki = k[:, np.newaxis]  # n×1
    kj = k[np.newaxis, :]  # 1×n
    denom = np.minimum(ki, kj) + 1.0 - adj  # n×n
    denom = np.where(denom < 1e-10, 1e-10, denom)

    tom = num / denom
    tom = np.clip(tom, 0.0, 1.0)
    np.fill_diagonal(tom, 1.0)
    return tom.astype(np.float32)


def dynamic_tree_cut(dist_mat, height_cut, min_size):
    """
    Simple height-threshold dynamic tree cut.
    Returns: int32 array of module labels (1-based; 0 = unassigned/grey).
    """
    n = dist_mat.shape[0]
    # Convert condensed distance matrix for scipy linkage
    # dist_mat is full symmetric; extract upper triangle
    from scipy.spatial.distance import squareform
    condensed = squareform(dist_mat, checks=False)
    Z = sch.linkage(condensed, method="average")

    # Cut at height_cut
    flat_labels = sch.fcluster(Z, t=height_cut, criterion="distance")  # 1-based

    # Relabel small modules as 0 (grey)
    from collections import Counter
    counts = Counter(flat_labels)
    label_map = {}
    mod_id = 1
    for lab, cnt in sorted(counts.items()):
        if cnt >= min_size:
            label_map[lab] = mod_id
            mod_id += 1
        else:
            label_map[lab] = 0

    result = np.array([label_map[l] for l in flat_labels], dtype=np.int32)
    return result


def compute_eigengenes(expr, module_labels):
    """
    First PC of expression submatrix for each non-zero module.
    expr: cells × genes
    Returns eigengenes: cells × n_modules (float32), ordered by module id 1..max.
    """
    from scipy.sparse.linalg import svds

    unique_mods = sorted(m for m in np.unique(module_labels) if m > 0)
    n_cells = expr.shape[0]
    eigengenes = np.zeros((n_cells, len(unique_mods)), dtype=np.float32)

    for col_i, mod in enumerate(unique_mods):
        gene_mask = (module_labels == mod)
        sub = expr[:, gene_mask].astype(np.float64)  # cells × mod_genes
        # Mean-center columns
        sub -= sub.mean(axis=0, keepdims=True)
        if sub.shape[1] == 1:
            # Only one gene: eigengene = that gene's expression (normalized)
            vec = sub[:, 0]
            norm = np.linalg.norm(vec)
            vec = vec / (norm if norm > 1e-10 else 1.0)
            eigengenes[:, col_i] = vec.astype(np.float32)
        else:
            try:
                _, _, Vt = svds(sub, k=1)
                pc1 = (sub @ Vt[0]).astype(np.float32)
                # Orient so mean > 0 (match R convention)
                if pc1.mean() < 0:
                    pc1 = -pc1
                eigengenes[:, col_i] = pc1
            except Exception:
                eigengenes[:, col_i] = 0.0

    return eigengenes


def compute_hub_kme(expr, module_labels, eigengenes, top_n=10):
    """
    Compute kME (module eigengene correlation) for every gene in its module.
    Returns list of (module_idx_1based, gene_idx, kme) for top_n genes per module.
    """
    unique_mods = sorted(m for m in np.unique(module_labels) if m > 0)
    hub_entries = []

    for col_i, mod in enumerate(unique_mods):
        gene_indices = np.where(module_labels == mod)[0]
        eigen = eigengenes[:, col_i].astype(np.float64)
        kme_vals = []
        for g_idx in gene_indices:
            gene_expr = expr[:, g_idx].astype(np.float64)
            if np.std(gene_expr) < 1e-10 or np.std(eigen) < 1e-10:
                r = 0.0
            else:
                r, _ = pearsonr(gene_expr, eigen)
            kme_vals.append((g_idx, float(r)))
        # Sort by kME descending, keep top_n
        kme_vals.sort(key=lambda x: -x[1])
        for (g_idx, kme) in kme_vals[:top_n]:
            hub_entries.append((int(mod), int(g_idx), float(kme)))

    return hub_entries


# ---------------------------------------------------------------------------
# R delegation path
# ---------------------------------------------------------------------------

def try_r_path(args, data_dir, out_dir):
    """
    Attempt to run hdwgcna_r_reference.R. Returns True if succeeded (exit 0),
    False if R/WGCNA not available (exit 2).
    """
    if not os.path.isfile(args.r_script):
        print(f"[wgcna_py_reference] R script not found: {args.r_script}",
              file=sys.stderr)
        return False

    cmd = [
        "Rscript", args.r_script,
        "--data_dir", data_dir,
        "--out_dir", out_dir,
        "--task", args.task,
        "--height_cut", str(args.height_cut),
        "--min_module_size", str(args.min_module_size),
    ]
    try:
        ret = subprocess.run(cmd, timeout=300)
        if ret.returncode == 0:
            return True
        if ret.returncode == 2:
            print("[wgcna_py_reference] R WGCNA not installed; using scipy fallback.",
                  file=sys.stderr)
            return False
        print(f"[wgcna_py_reference] Rscript exited {ret.returncode}; "
              "using scipy fallback.", file=sys.stderr)
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[wgcna_py_reference] Rscript invocation failed ({e}); "
              "using scipy fallback.", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Try R path first if requested
    if args.use_r:
        ok = try_r_path(args, args.data_dir, args.out_dir)
        if ok:
            sys.exit(0)

    # Pure-Python scipy fallback
    print("[wgcna_py_reference] Running scipy fallback.", file=sys.stderr)

    expr = read_expression(args.data_dir)          # cells × genes float32
    beta = read_soft_power(args.data_dir)          # int

    corr = compute_pearson_matrix(expr)            # genes × genes
    tom = compute_tom(corr, beta)                  # genes × genes
    dist_mat = (1.0 - tom).astype(np.float32)
    np.fill_diagonal(dist_mat, 0.0)
    np.clip(dist_mat, 0.0, None, out=dist_mat)

    labels = dynamic_tree_cut(dist_mat, args.height_cut, args.min_module_size)
    eigengenes = compute_eigengenes(expr, labels)  # cells × n_mods
    hub_entries = compute_hub_kme(expr, labels, eigengenes, top_n=10)

    write_module_labels(args.out_dir, labels)
    write_tom(args.out_dir, tom)
    write_eigengenes(args.out_dir, eigengenes)
    write_hub_kme(args.out_dir, hub_entries)

    n_modules = int((labels > 0).any() and labels.max() or 0)
    print(f"[wgcna_py_reference] Done. n_genes={len(labels)}, "
          f"n_modules={n_modules}, beta={beta}", file=sys.stderr)
    sys.exit(2)   # exit 2 = scipy fallback was used (valid reference)


if __name__ == "__main__":
    main()
