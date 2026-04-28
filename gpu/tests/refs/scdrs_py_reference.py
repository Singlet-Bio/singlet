#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/scdrs_py_reference.py
#
# Reference driver for scDRS per-cell disease relevance scoring.
# Primary reference: scDRS Python package (Zhang et al. 2022, Nature Genetics).
# Fallback: pure-Python weighted sum + control-set null matching (always runs).
#
# Algorithm:
#   1. Load log-normalized expression (n_cells × n_genes) from NPY/NPZ.
#   2. Load disease gene sets with weights from TSV.
#   3. Z-score normalize genes across cells.
#   4. Compute raw_score per (cell, disease) as weighted sum of z-scores.
#   5. Build matched control gene sets (matched on mean expression + variance).
#   6. Compute control scores → empirical null per cell.
#   7. Compute normalized score (z vs null) and Monte Carlo p-value.
#   8. Write outputs: raw_score.npy, norm_score.npy, pvalue.npy
#
# Environment:
#   Python >= 3.9
#   pip install scdrs numpy scipy anndata
#   Fallback requires only: numpy, scipy
#
# Usage:
#   python scdrs_py_reference.py \
#       --expr_npy    <expr.npy>            # n_cells × n_genes float32 log-norm expr
#       --gene_names  <gene_names.txt>      # one gene name per line, n_genes lines
#       --diseases    <diseases_mini.tsv>   # columns: disease_name gene_idx weight
#       --n_ctrl      <int>                 # number of control sets (default 1000)
#       --seed        <int>                 # default 42
#       --out_dir     <dir>                 # writes raw_score.npy, norm_score.npy, pvalue.npy
#       [--force_fallback]                  # skip scdrs even if installed
#
# Output:
#   out_dir/raw_score.npy   — float32 (n_cells × n_diseases)
#   out_dir/norm_score.npy  — float32 (n_cells × n_diseases)
#   out_dir/pvalue.npy      — float32 (n_cells × n_diseases)
#   out_dir/backend.txt     — "scdrs" or "fallback"
#
# Exit codes:
#   0 — success
#   1 — argument or I/O error
#   2 — no backend available (numpy missing; should never happen)

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="scDRS Python reference driver")
    p.add_argument("--expr_npy",   required=True, help="n_cells × n_genes float32 NPY")
    p.add_argument("--gene_names", required=True, help="Gene names TXT (one per line)")
    p.add_argument("--diseases",   required=True, help="Disease gene-set TSV")
    p.add_argument("--n_ctrl",     type=int, default=1000, help="Number of control sets")
    p.add_argument("--seed",       type=int, default=42,   help="RNG seed")
    p.add_argument("--out_dir",    required=True, help="Output directory")
    p.add_argument("--force_fallback", action="store_true",
                   help="Skip scdrs even if installed; use pure-Python fallback")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_expr(path: str) -> np.ndarray:
    """Load expression matrix from .npy (n_cells × n_genes, float32)."""
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"expr_npy must be 2-D, got shape {arr.shape}")
    return arr.astype(np.float32)


def load_gene_names(path: str) -> list[str]:
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def load_diseases(path: str, gene_names: list[str]) -> dict[str, dict[int, float]]:
    """
    Parse disease TSV with columns: disease_name  gene_idx  weight
    gene_idx is a 0-based integer index into gene_names.
    Returns {disease_name: {gene_idx: weight}}.
    """
    diseases: dict[str, dict[int, float]] = {}
    n_genes = len(gene_names)
    with open(path) as fh:
        header = fh.readline().strip().split("\t")
        di = header.index("disease_name")
        gi = header.index("gene_idx")
        wi = header.index("weight")
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            dname = parts[di]
            gidx  = int(parts[gi])
            w     = float(parts[wi])
            if gidx < 0 or gidx >= n_genes:
                continue
            diseases.setdefault(dname, {})[gidx] = w
    return diseases


# ---------------------------------------------------------------------------
# Pure-Python fallback implementation (always available; no scdrs dep)
# ---------------------------------------------------------------------------

def _gene_zscore(expr: np.ndarray) -> np.ndarray:
    """Z-score normalize across cells per gene: result shape = n_cells × n_genes."""
    mu  = expr.mean(axis=0, keepdims=True)   # 1 × n_genes
    sig = expr.std(axis=0, keepdims=True)    # 1 × n_genes
    sig = np.where(sig < 1e-8, 1.0, sig)
    return (expr - mu) / sig                 # n_cells × n_genes


def _build_weight_matrix(diseases: dict[str, dict[int, float]],
                          n_genes: int) -> tuple[np.ndarray, list[str]]:
    """
    Returns:
      W       — float32 (n_genes × n_diseases), column-normalised to unit sum
      dnames  — list of disease names in column order
    """
    dnames = sorted(diseases.keys())
    W = np.zeros((n_genes, len(dnames)), dtype=np.float32)
    for j, dname in enumerate(dnames):
        for gidx, w in diseases[dname].items():
            W[gidx, j] = w
        col_sum = W[:, j].sum()
        if col_sum > 0:
            W[:, j] /= col_sum
    return W, dnames


def _build_control_sets(diseases: dict[str, dict[int, float]],
                         expr: np.ndarray,
                         n_ctrl: int,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Build matched control gene sets.
    For each disease, sample n_ctrl control sets with matched (mean, var) bins.
    Returns ctrl_W: float32 (n_genes × n_ctrl), weights summing to ~1 per column.

    Strategy (design doc §"Control set matching"):
      - Bin genes into 20 bins by (mean_expr + var_expr).
      - For each gene in the disease set, sample a replacement from the same bin.
    """
    n_cells, n_genes = expr.shape
    # Gene statistics across cells.
    gene_mean = expr.mean(axis=0)   # n_genes
    gene_var  = expr.var(axis=0)    # n_genes
    score     = gene_mean + gene_var
    n_bins    = 20
    pcts      = np.percentile(score, np.linspace(0, 100, n_bins + 1))
    pcts[0]  -= 1e-6
    bin_ids   = np.digitize(score, pcts[1:])  # 0-based bin per gene (0 .. n_bins-1)
    bin_ids   = np.clip(bin_ids, 0, n_bins - 1)

    # Precompute gene lists per bin.
    bins: dict[int, list[int]] = {}
    for gidx in range(n_genes):
        b = int(bin_ids[gidx])
        bins.setdefault(b, []).append(gidx)

    # Aggregate disease gene sets (union of all disease gene indices).
    all_disease_genes: set[int] = set()
    for gs in diseases.values():
        all_disease_genes.update(gs.keys())
    disease_list = sorted(all_disease_genes)
    if not disease_list:
        return np.zeros((n_genes, n_ctrl), dtype=np.float32)

    ctrl_W = np.zeros((n_genes, n_ctrl), dtype=np.float32)
    n_disease = len(disease_list)
    w_per_gene = 1.0 / n_disease

    for k in range(n_ctrl):
        # For each disease gene, sample a replacement from the same bin.
        sampled: list[int] = []
        for gidx in disease_list:
            b = int(bin_ids[gidx])
            candidates = bins.get(b, [])
            if len(candidates) <= 1:
                sampled.append(gidx)  # degenerate: keep original
            else:
                # Exclude the gene itself if possible.
                pool = [g for g in candidates if g != gidx]
                if not pool:
                    pool = candidates
                sampled.append(int(rng.choice(pool)))
        for gidx in sampled:
            ctrl_W[gidx, k] += w_per_gene
    return ctrl_W


def run_fallback(expr: np.ndarray,
                 gene_names: list[str],
                 diseases: dict[str, dict[int, float]],
                 n_ctrl: int,
                 seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure-Python scDRS-style scoring.
    Returns: (raw_score, norm_score, pvalue) each shape (n_cells, n_diseases).
    """
    rng = np.random.default_rng(seed)
    n_cells, n_genes = expr.shape

    # 1. Z-score normalize (n_cells × n_genes).
    z = _gene_zscore(expr)

    # 2. Disease weight matrix (n_genes × n_diseases).
    W_dis, dnames = _build_weight_matrix(diseases, n_genes)
    n_dis = len(dnames)

    # 3. Raw scores: (n_cells × n_diseases) = z @ W_dis.
    raw_score = z @ W_dis   # float32 (n_cells × n_diseases)

    # 4. Control gene sets (n_genes × n_ctrl).
    W_ctrl = _build_control_sets(diseases, expr, n_ctrl, rng)

    # 5. Control scores: (n_cells × n_ctrl) = z @ W_ctrl.
    ctrl_score = z @ W_ctrl   # float32

    # 6. Normalized score per (cell, disease):
    #    null_mean[c] = mean over k of ctrl_score[c, k]
    #    null_std[c]  = std  over k of ctrl_score[c, k]
    null_mean = ctrl_score.mean(axis=1, keepdims=True)  # n_cells × 1
    null_std  = ctrl_score.std(axis=1,  keepdims=True)  # n_cells × 1
    null_std  = np.where(null_std < 1e-8, 1.0, null_std)
    norm_score = (raw_score - null_mean) / null_std     # n_cells × n_diseases

    # 7. Monte Carlo p-value: fraction of controls with score >= observed.
    #    pvalue[c, d] = (# k : ctrl_score[c,k] >= raw_score[c,d]) / n_ctrl + 1e-7
    # Vectorised: (n_cells, 1, n_ctrl) vs (n_cells, n_diseases, 1)
    raw_exp   = raw_score[:, :, np.newaxis]    # n_cells × n_dis × 1
    ctrl_exp  = ctrl_score[:, np.newaxis, :]   # n_cells × 1     × n_ctrl
    pvalue    = (ctrl_exp >= raw_exp).mean(axis=2).astype(np.float32)  # n_cells × n_dis
    pvalue    = np.clip(pvalue, 1e-7, 1.0)

    return raw_score, norm_score, pvalue


def run_scdrs(expr: np.ndarray,
              gene_names: list[str],
              diseases: dict[str, dict[int, float]],
              n_ctrl: int,
              seed: int,
              out_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Try to use the scdrs Python package.
    Falls back to pure-Python if import fails.
    Returns: (raw_score, norm_score, pvalue) each shape (n_cells, n_diseases).
    """
    try:
        import anndata as ad
        import scdrs

        n_cells, n_genes = expr.shape
        # Build AnnData object.
        adata = ad.AnnData(X=expr.copy())
        adata.var_names = [str(g) for g in gene_names]
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

        # Build gs_dict: {disease_name: (gene_list, weight_list)}
        dnames = sorted(diseases.keys())
        gs_dict: dict[str, tuple[list[str], list[float]]] = {}
        for dname in dnames:
            gs = diseases[dname]
            glist = [gene_names[gidx] for gidx in sorted(gs.keys())]
            wlist = [gs[gidx] for gidx in sorted(gs.keys())]
            gs_dict[dname] = (glist, wlist)

        # Pre-compute gene stats required by scdrs.
        scdrs.preprocess(adata, n_mean_bin=20, n_var_bin=20, copy=False)

        raw_list  = []
        norm_list = []
        pval_list = []
        for dname in dnames:
            glist, wlist = gs_dict[dname]
            # scdrs.score_cell returns a DataFrame indexed by cell.
            df = scdrs.score_cell(
                adata, gene_list=glist, gene_weight=wlist,
                n_ctrl=n_ctrl, ctrl_match_key="ctrl_norm_mean",
                weight_opt="vs", random_seed=seed,
            )
            raw_list.append(df["raw_score"].values.astype(np.float32))
            norm_list.append(df["norm_score"].values.astype(np.float32))
            pval_list.append(df["pval"].values.astype(np.float32))

        raw_score  = np.column_stack(raw_list)
        norm_score = np.column_stack(norm_list)
        pvalue     = np.column_stack(pval_list)
        return raw_score, norm_score, pvalue

    except ImportError:
        return None, None, None   # caller will use fallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load inputs.
    try:
        expr       = load_expr(args.expr_npy)
        gene_names = load_gene_names(args.gene_names)
        diseases   = load_diseases(args.diseases, gene_names)
    except Exception as exc:
        print(f"[scdrs_py_reference] I/O error: {exc}", file=sys.stderr)
        sys.exit(1)

    if expr.shape[1] != len(gene_names):
        print(f"[scdrs_py_reference] expr columns ({expr.shape[1]}) != "
              f"gene_names length ({len(gene_names)})", file=sys.stderr)
        sys.exit(1)

    if not diseases:
        print("[scdrs_py_reference] No diseases loaded from TSV.", file=sys.stderr)
        sys.exit(1)

    backend = "fallback"
    raw_score = norm_score = pvalue = None

    if not args.force_fallback:
        raw_score, norm_score, pvalue = run_scdrs(
            expr, gene_names, diseases, args.n_ctrl, args.seed, args.out_dir)
        if raw_score is not None:
            backend = "scdrs"

    if raw_score is None:
        raw_score, norm_score, pvalue = run_fallback(
            expr, gene_names, diseases, args.n_ctrl, args.seed)
        backend = "fallback"

    # Write outputs.
    np.save(os.path.join(args.out_dir, "raw_score.npy"),  raw_score)
    np.save(os.path.join(args.out_dir, "norm_score.npy"), norm_score)
    np.save(os.path.join(args.out_dir, "pvalue.npy"),     pvalue)
    with open(os.path.join(args.out_dir, "backend.txt"), "w") as fh:
        fh.write(backend + "\n")

    dnames = sorted(diseases.keys())
    print(f"[scdrs_py_reference] backend={backend} "
          f"n_cells={expr.shape[0]} n_genes={expr.shape[1]} "
          f"n_diseases={len(dnames)} n_ctrl={args.n_ctrl}")
    print(f"[scdrs_py_reference] raw_score shape={raw_score.shape} "
          f"finite={np.all(np.isfinite(raw_score))}")


if __name__ == "__main__":
    main()
