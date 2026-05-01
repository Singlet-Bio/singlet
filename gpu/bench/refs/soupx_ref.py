#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/soupx_ref.py
#
# CYCLE-180 Phase E — manual scipy/numpy SoupX CPU baseline.
#
# GPU kernel (qc::soupx): 5 GPU passes —
#   Pass 1: per-droplet UMI sum t[j] (warp/col from CSC);
#   Pass 2: ambient scatter (atomic/nnz for empty cols, t[j] <= lower) + normalize → pi[g];
#   Pass 3: top-ambient-gene mask (one D2H of pi, host partial_sort + threshold, H2D mask);
#   Pass 4: per-cell rho_c (warp/col, top_amb_mask weighted numerator / denom_pi);
#   Pass 5: dense corrected output (cudaMemset + nnz-overwrite;
#            max(0, x[g,c] - rho_c[c] * t[c] * pi[g]) stored only for nnz entries;
#            algebraic identity: max(0, 0 - ...) = 0 handled by cudaMemset).
# Reference: Young MD, Behjati S (2020) GigaScience 9:giaa151.
#
# CPU implementation: manual numpy/scipy — mirrors the algorithmic structure.
#   Pass 1 — UMI totals:      t = np.asarray(X.sum(axis=0)).ravel()
#   Pass 2 — ambient profile: raw = X[:, t <= lower].sum(axis=1); pi = raw / raw.sum()
#   Pass 3 — top-ambient mask: argsort/argpartition on pi; top 10% genes → mask
#   Pass 4 — per-cell rho_c:  num = X[mask, :].sum(axis=0) (sparse slice);
#                               rho_c = clip(num / (t * denom_pi), min_rho, max_rho)
#   Pass 5 — sparse correction: for each stored nnz entry,
#                                corrected = max(0, x - rho_c[c] * t[c] * pi[g])
#                               (sparse output, no dense intermediate on CPU)
#
# SOTA structure: all passes are vectorized scipy/numpy (CSC slice, sum, argsort,
#   sparse-indexing, clip). No Python inner loops. python_overhead_multiplier = 1×.
# Memory: CPU outputs a sparse correction matrix (same nnz as input ~3%).
#         GPU outputs a DENSE m×n corrected matrix (114 MB at 10k, 342 MB at 30k).
#         memory_bandwidth_advantage = 1× (CPU stays sparse; GPU HBM advantage
#         on dense write is offset by having to write more data).
# GPU compute: light-medium (5 passes; no cuRAND, no BLAS, no Cholesky).
# §J.7 prediction: 10-30× (class 3, same as wsum/decoupler family;
#   lower bound because GPU materializes dense output CPU avoids).
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#   Matrix synthesis is untimed. Only passes 1-5 (the soupx pipeline) are timed.
#
# Sanity-check printed to stderr (after first timed run):
#   n_cells, n_empty, rho_c_mean, ambient_top5_genes, correction_min/max
#   (expect rho_c_mean in [0.05, 0.5] for Zipf-Poisson synthetic data)
#
# Scales:
#   10k: n_droplets=10000, n_genes=3000, lower=100
#   30k: n_droplets=30000, n_genes=3000, lower=100
#
# Usage: python3 soupx_ref.py

import sys
import time

import numpy as np
import scipy.sparse as sp

WARMUP_ITERS = 2
TIMED_ITERS  = 5

SCALES = [
    ("10k", 10000, 3000, 100),
    ("30k", 30000, 3000, 100),
]

TOP_AMBIENT_FRAC = 0.10
MIN_RHO          = 0.0
MAX_RHO          = 0.9


# ---------------------------------------------------------------------------
# make_synth_10x_csc — mirrors the C++ make_synth_10x_csc exactly.
# Returns a scipy CSC matrix (n_genes × n_droplets).
# ---------------------------------------------------------------------------
def make_synth_10x_csc(n_genes: int, n_droplets: int,
                        lower: int, frac_empty: float = 0.8,
                        seed: int = 42) -> sp.csc_matrix:
    """Synthetic raw 10X-style CSC.

    Zipf ambient: pi[g] ∝ 1/(g+1), normalized.
    Empty droplets (~frac_empty): Poisson UMI mean = lower*0.3.
    Cell droplets: Poisson UMI mean = 200 (shifted above lower).
    """
    rng = np.random.default_rng(seed=seed)

    pi = np.array([1.0 / (g + 1) for g in range(n_genes)], dtype=np.float64)
    pi /= pi.sum()

    n_empty = int(frac_empty * n_droplets)
    n_cand  = n_droplets - n_empty

    umi_empty = rng.poisson(lam=lower * 0.3, size=n_empty).clip(1)
    umi_cand  = rng.poisson(lam=200, size=n_cand) + lower + 1

    umi_totals = np.concatenate([umi_empty, umi_cand]).astype(np.int32)

    col_ptrs = [0]
    row_indices_list = []
    values_list = []
    for j in range(n_droplets):
        t_j = int(umi_totals[j])
        counts = rng.multinomial(t_j, pi)
        nonzero_genes = np.where(counts > 0)[0]
        row_indices_list.append(nonzero_genes)
        values_list.append(counts[nonzero_genes].astype(np.float32))
        col_ptrs.append(col_ptrs[-1] + len(nonzero_genes))

    row_indices = np.concatenate(row_indices_list).astype(np.int32)
    values      = np.concatenate(values_list).astype(np.float32)
    indptr      = np.array(col_ptrs, dtype=np.int32)

    return sp.csc_matrix((values, row_indices, indptr),
                         shape=(n_genes, n_droplets))


# ---------------------------------------------------------------------------
# soupx_numpy — manual 5-pass SoupX equivalent (ambient RNA correction).
# ---------------------------------------------------------------------------
def soupx_numpy(X: sp.csc_matrix, lower: int,
                top_ambient_frac: float = TOP_AMBIENT_FRAC,
                min_rho: float = MIN_RHO,
                max_rho: float = MAX_RHO) -> dict:
    """Manual numpy/scipy SoupX (Young & Behjati 2020) for CPU baseline.

    Args:
        X               — (n_genes, n_droplets) scipy CSC
        lower           — UMI threshold separating empty from cells
        top_ambient_frac — fraction of genes treated as ambient markers
        min_rho/max_rho — per-cell contamination fraction bounds

    Returns dict with keys:
        rho_c       — (n_droplets,) float32 per-cell contamination fraction
        ambient_pi  — (n_genes,) float32 normalized ambient profile
        corrected_vals — corrected values for stored nnz entries (sparse)
        n_cells     — int: number of droplets above lower threshold
        n_empty     — int: number of empty droplets
    """
    n_genes, n_droplets = X.shape

    # Pass 1: UMI totals per droplet (vectorized CSC sum).
    t = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)

    # Pass 2: ambient profile from empty droplets (t[j] <= lower).
    is_empty = t <= lower
    n_empty  = int(is_empty.sum())
    n_cells  = n_droplets - n_empty

    ambient_raw = np.asarray(X[:, is_empty].sum(axis=1)).ravel().astype(np.float64)
    total_ambient = ambient_raw.sum()
    if total_ambient <= 0:
        raise ValueError("soupx: no empty droplets found (lower=" + str(lower) + ").")
    ambient_pi = (ambient_raw / total_ambient).astype(np.float32)

    # Pass 3: top-ambient-gene mask (host-side argsort/argpartition).
    n_top = max(1, int(n_genes * top_ambient_frac))
    # argpartition gives the top-n_top indices (partial sort, vectorized).
    top_idx  = np.argpartition(ambient_pi, -n_top)[-n_top:]
    top_mask = np.zeros(n_genes, dtype=bool)
    top_mask[top_idx] = True
    denom_pi = float(ambient_pi[top_mask].sum())

    # Pass 4: per-cell rho_c — numerator = sum of top-ambient-gene counts per cell.
    # Efficient: slice sparse X to top_mask rows, sum over genes per cell.
    X_top    = X[top_mask, :]  # sparse (n_top × n_droplets)
    num      = np.asarray(X_top.sum(axis=0)).ravel().astype(np.float32)
    # rho_c[c] = num[c] / (t[c] * denom_pi), clipped; 0 for empty droplets.
    denom    = t * denom_pi
    rho_c    = np.where(
        (denom > 0) & (~is_empty),
        np.clip(num / np.where(denom > 0, denom, 1.0), min_rho, max_rho),
        0.0,
    ).astype(np.float32)

    # Pass 5: sparse correction — overwrite stored nnz entries.
    # corrected[g,c] = max(0, x[g,c] - rho_c[c] * t[c] * pi[g])
    # Implicit zeros stay zero (no dense materialization needed on CPU).
    X_coo    = X.tocoo()
    g_idx    = X_coo.row
    c_idx    = X_coo.col
    x_vals   = X_coo.data.astype(np.float32)
    sub      = rho_c[c_idx] * t[c_idx] * ambient_pi[g_idx]
    corrected_vals = np.maximum(0.0, x_vals - sub)

    return {
        "rho_c":           rho_c,
        "ambient_pi":      ambient_pi,
        "corrected_vals":  corrected_vals,
        "n_cells":         n_cells,
        "n_empty":         n_empty,
    }


# ---------------------------------------------------------------------------
# time_soupx — 2 warmup + 5 timed iterations.
# Returns (wall_ms_median, last_result_dict).
# ---------------------------------------------------------------------------
def time_soupx(X: sp.csc_matrix, lower: int) -> tuple:
    """Benchmark soupx_numpy. Returns (wall_ms_median, result_dict)."""
    result = None
    wall_times = []

    for run in range(WARMUP_ITERS + TIMED_ITERS):
        t0 = time.perf_counter()
        result = soupx_numpy(X, lower=lower)
        t1 = time.perf_counter()
        if run >= WARMUP_ITERS:
            wall_times.append((t1 - t0) * 1000.0)

    wall_times.sort()
    n_t = len(wall_times)
    median_ms = (wall_times[n_t // 2] if n_t % 2 == 1
                 else 0.5 * (wall_times[n_t // 2 - 1] + wall_times[n_t // 2]))

    return median_ms, result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("scale,n_droplets,n_genes,lower,wall_ms", flush=True)

    for scale_name, n_droplets, n_genes, lower in SCALES:
        print(
            f"[ref] Building synthetic {scale_name}: {n_genes} genes × "
            f"{n_droplets} droplets, lower={lower}...",
            file=sys.stderr, flush=True,
        )

        X = make_synth_10x_csc(
            n_genes=n_genes, n_droplets=n_droplets,
            lower=lower, frac_empty=0.8, seed=42)

        print(
            f"[ref] {scale_name}: X.shape={X.shape}, nnz={X.nnz}. "
            f"Timing soupx (5-pass numpy)...",
            file=sys.stderr, flush=True,
        )

        try:
            wall_ms, res = time_soupx(X, lower=lower)

            rho_mean = float(res["rho_c"][res["rho_c"] > 0].mean()) \
                       if res["rho_c"].any() else 0.0
            top5 = int(np.argpartition(res["ambient_pi"], -5)[-5:].max())
            corr_min = float(res["corrected_vals"].min())
            corr_max = float(res["corrected_vals"].max())

            print(
                f"[ref] {scale_name}: n_cells={res['n_cells']}, "
                f"n_empty={res['n_empty']}, "
                f"rho_c_mean={rho_mean:.4f}, "
                f"top5_ambient_gene_idx<={top5}, "
                f"corrected_range=[{corr_min:.2f}, {corr_max:.2f}]",
                file=sys.stderr, flush=True,
            )
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}", file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_droplets},{n_genes},{lower},{wall_ms:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
