#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/empty_drops_ref.py
#
# CYCLE-179 Phase E — manual scipy/numpy emptyDrops CPU baseline.
#
# GPU kernel (qc::empty_drops): 6 GPU passes — UMI column-sum (warp/col),
#   mark-empty + ambient-scatter (atomic, one thread/nnz), normalize-ambient,
#   observed LL (warp/cand), MC p-values (one block/cand, cuRAND Philox4x32,
#   smem cumulative CDF + binary-search categorical sampling, 256 threads/block),
#   BH FDR (host-side sort+scan at one D2H boundary), scatter is_cell.
#   See include/singlet-gpu/qc/empty_drops.h.
# Citation: Lun ATL et al. (2019) Genome Biology 20:63.
#
# CPU implementation: manual numpy/scipy — mirrors the algorithmic structure
#   but uses Python loops for the MC sampling per candidate.
#   Pass 1 — UMI totals:     t = X.sum(axis=0)   (numpy/scipy CSC)
#   Pass 2 — ambient:        pi = X[:, t<=lower].sum(axis=1); pi /= pi.sum()
#   Pass 3 — observed LL:    ll_obs[j] = X[:,j].toarray().ravel() @ log(pi)
#   Pass 4 — MC p-values:    per-candidate Python loop: rng.multinomial(t_j, pi)
#                             × niters → compare synth_ll to ll_obs → hit count
#   Pass 5 — BH FDR:         numpy argsort + BH scan (vectorized)
#   Pass 6 — is_cell:        fdr < fdr_thresh
#
# SOTA structure: per-candidate per-iteration Python loop with
#   numpy rng.multinomial(t_j, pi) per call. The multinomial draw itself
#   is vectorized C (fast per call) but is invoked inside a Python for-loop
#   → python_overhead_multiplier = 10-100× (Python loop overhead per MC iter
#   per candidate).
# §J.7 prediction: python_overhead_multiplier = 10-100×,
#   memory_bandwidth_advantage = 1× (no dense intermediates > 100 MB),
#   GPU compute = medium-heavy (cuRAND + smem CDF + binary search O(log m)).
#   Predicted speedup: 50-200× (class 2-3).
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#   Only the full emptyDrops inference pipeline (passes 1-6) is timed.
#   Matrix synthesis is untimed.
#
# Scales:
#   10k: n_droplets=10000, n_genes=3000, lower=100, niters=10000
#   30k: n_droplets=30000, n_genes=3000, lower=100, niters=10000
#
# Sanity-check printed to stderr:
#   n_candidates, n_cells_called, pval_range
#   (expect n_cells ~ 20% of droplets with FDR<0.001 for synthetic data)
#
# Usage: python3 empty_drops_ref.py

import sys
import time

import numpy as np
import scipy.sparse as sp

WARMUP_ITERS = 2
TIMED_ITERS  = 5

SCALES = [
    ("10k", 10000, 3000, 100, 10000),
    ("30k", 30000, 3000, 100, 10000),
]

FDR_THRESH = 0.001


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
    Candidate droplets: Poisson UMI mean = 200 (shifted above lower).
    """
    rng = np.random.default_rng(seed=seed)

    # Zipf ambient profile.
    pi = np.array([1.0 / (g + 1) for g in range(n_genes)], dtype=np.float64)
    pi /= pi.sum()

    n_empty = int(frac_empty * n_droplets)
    n_cand  = n_droplets - n_empty

    # UMI totals.
    umi_empty = rng.poisson(lam=lower * 0.3, size=n_empty).clip(1)
    umi_cand  = rng.poisson(lam=200, size=n_cand) + lower + 1

    umi_totals = np.concatenate([umi_empty, umi_cand]).astype(np.int32)

    # Build CSC col-by-col via multinomial sampling.
    col_ptrs = [0]
    row_indices_list = []
    values_list = []
    for j in range(n_droplets):
        t_j = int(umi_totals[j])
        # Sample multinomial counts over n_genes.
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
# empty_drops_numpy — manual 6-pass emptyDrops equivalent.
# ---------------------------------------------------------------------------
def empty_drops_numpy(X: sp.csc_matrix, lower: int, niters: int,
                      fdr_thresh: float = FDR_THRESH, seed: int = 42):
    """Manual numpy/scipy emptyDrops (Lun 2019) for CPU baseline.

    Args:
        X           — (n_genes, n_droplets) scipy CSC
        lower       — UMI threshold separating empty from candidates
        niters      — MC iterations per candidate
        fdr_thresh  — BH FDR threshold for cell calling

    Returns:
        pvalue      — (n_droplets,) float32; 1.0 for empty/skipped
        fdr         — (n_droplets,) float32; BH q-values
        is_cell     — (n_droplets,) bool
        n_candidates— int
    """
    rng = np.random.default_rng(seed=seed)
    n_genes, n_droplets = X.shape

    # Pass 1: UMI totals per droplet.
    t = np.asarray(X.sum(axis=0)).ravel()  # (n_droplets,)

    # Pass 2: ambient profile from empty droplets.
    is_empty = t <= lower
    ambient_raw = np.asarray(X[:, is_empty].sum(axis=1)).ravel().astype(np.float64)
    total_ambient = ambient_raw.sum()
    if total_ambient <= 0:
        raise ValueError("empty_drops: no empty droplets detected.")
    pi = (ambient_raw / total_ambient).astype(np.float64)
    log_pi = np.log(np.maximum(pi, 1e-30))

    # Build candidate list: UMI > lower and UMI <= 100000.
    candidates = np.where((t > lower) & (t <= 100000))[0]
    n_cand = len(candidates)

    pvalue  = np.ones(n_droplets, dtype=np.float32)
    fdr_out = np.ones(n_droplets, dtype=np.float32)
    is_cell = np.zeros(n_droplets, dtype=bool)

    if n_cand == 0:
        return pvalue, fdr_out, is_cell, n_cand

    # Pass 3: observed LL per candidate.
    ll_obs = np.empty(n_cand, dtype=np.float64)
    for ci, j in enumerate(candidates):
        col_data = X.getcol(j)
        genes    = col_data.indices
        counts   = col_data.data.astype(np.float64)
        ll_obs[ci] = float(np.sum(counts * log_pi[genes]))

    # Pass 4: MC p-values — Python loop per candidate, per iter.
    hit_count = np.zeros(n_cand, dtype=np.int32)
    for ci, j in enumerate(candidates):
        t_j    = int(t[j])
        obs_ll = ll_obs[ci]
        hits   = 0
        for _ in range(niters):
            # Multinomial draw of t_j UMIs under ambient pi.
            synth_counts = rng.multinomial(t_j, pi)
            synth_ll     = float(np.sum(synth_counts * log_pi))
            if synth_ll <= obs_ll:
                hits += 1
        hit_count[ci] = hits

    cand_pvals = (1.0 + hit_count.astype(np.float64)) / float(niters + 1)
    pvalue[candidates] = cand_pvals.astype(np.float32)

    # Pass 5: BH FDR (vectorized argsort + scan).
    ord_idx = np.argsort(cand_pvals)
    fdr_cand = np.empty(n_cand, dtype=np.float32)
    rmin = 1.0
    for rank in range(n_cand - 1, -1, -1):
        ci = ord_idx[rank]
        q  = min(1.0, float(cand_pvals[ci]) * n_cand / (rank + 1))
        rmin = min(q, rmin)
        fdr_cand[ci] = float(rmin)

    fdr_out[candidates] = fdr_cand

    # Pass 6: is_cell scatter.
    is_cell[candidates] = fdr_cand < fdr_thresh

    return pvalue, fdr_out, is_cell, n_cand


# ---------------------------------------------------------------------------
# time_empty_drops — 2 warmup + 5 timed iterations.
# Returns median wall time in ms.
# ---------------------------------------------------------------------------
def time_empty_drops(X: sp.csc_matrix, lower: int, niters: int,
                     fdr_thresh: float = FDR_THRESH) -> tuple:
    """Benchmark empty_drops_numpy.

    Returns:
        (wall_ms_median, pvalue, fdr, is_cell, n_candidates)
    """
    pvalue, fdr, is_cell, n_cand = None, None, None, 0

    for run in range(WARMUP_ITERS + TIMED_ITERS):
        t0 = time.perf_counter()
        pvalue, fdr, is_cell, n_cand = empty_drops_numpy(
            X, lower=lower, niters=niters, fdr_thresh=fdr_thresh)
        t1 = time.perf_counter()
        if run == 0:
            wall_times = []
        if run >= WARMUP_ITERS:
            wall_times.append((t1 - t0) * 1000.0)

    wall_times.sort()
    n_t = len(wall_times)
    median_ms = (wall_times[n_t // 2] if n_t % 2 == 1
                 else 0.5 * (wall_times[n_t // 2 - 1] + wall_times[n_t // 2]))

    return median_ms, pvalue, fdr, is_cell, n_cand


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("scale,n_droplets,n_genes,lower,niters,wall_ms", flush=True)

    for scale_name, n_droplets, n_genes, lower, niters in SCALES:
        print(
            f"[ref] Building synthetic {scale_name}: {n_genes} genes × "
            f"{n_droplets} droplets, lower={lower}, niters={niters}...",
            file=sys.stderr, flush=True,
        )

        X = make_synth_10x_csc(
            n_genes=n_genes, n_droplets=n_droplets,
            lower=lower, frac_empty=0.8, seed=42)

        print(
            f"[ref] {scale_name}: X.shape={X.shape}, nnz={X.nnz}. "
            f"Timing emptyDrops MC (niters={niters})...",
            file=sys.stderr, flush=True,
        )

        try:
            wall_ms, pvalue, fdr, is_cell, n_cand = time_empty_drops(
                X, lower=lower, niters=niters, fdr_thresh=FDR_THRESH)

            n_cells = int(is_cell.sum())
            cand_pvals = pvalue[pvalue < 1.0]
            pval_min   = float(cand_pvals.min()) if len(cand_pvals) else float('nan')
            pval_max   = float(cand_pvals.max()) if len(cand_pvals) else float('nan')

            print(
                f"[ref] {scale_name}: n_candidates={n_cand}, "
                f"n_cells_called={n_cells}, "
                f"pval_range=[{pval_min:.4f}, {pval_max:.4f}]",
                file=sys.stderr, flush=True,
            )
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}", file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_droplets},{n_genes},{lower},{niters},{wall_ms:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
