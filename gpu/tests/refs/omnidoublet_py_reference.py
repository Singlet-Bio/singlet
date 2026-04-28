#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/omnidoublet_py_reference.py
#
# Reference implementation driver for cycle 39 OmniDoublet correctness harness.
#
# Hierarchy of reference implementations (tried in order):
#   1. OmniDoublet Python (pip install omnidoublet) — preferred, multimodal
#   2. Scrublet Python   (pip install scrublet)     — RNA-only fallback
#   3. Pure-Python Scrublet-style implementation embedded below — always present
#
# If (1) is present: runs full multimodal OmniDoublet using both rna_counts and
#   adt_counts.  Scores include ADT signal.
# If (1) absent but (2) present: runs Scrublet on rna_counts only.  Scores will
#   be lower-quality but still provide a meaningful regression target for
#   Omnidoublet_TinySynthetic_VsPython (AUC ≥ 0.85 against planted doublets).
# If both absent: runs the embedded pure-Python Scrublet-style algorithm so the
#   test always has a reference even in a minimal Python environment (numpy +
#   scipy + scikit-learn only).
#
# Usage (called by qc_omnidoublet_correctness.cpp via subprocess):
#   python3 omnidoublet_py_reference.py \
#       --rna-counts  <rna_counts.bin>   \
#       --adt-counts  <adt_counts.bin>   \
#       --out         <results.npz>      \
#       --n-synth-frac <F>               \
#       --k           <K>                \
#       --n-pcs       <N>                \
#       --expected-doublet-rate <R>      \
#       --seed        <S>
#
# Exit codes:
#   0  success — results.npz written
#   1  any unexpected error → test FAIL
#   (exit 2 is NOT used: we always have the pure-Python fallback)
#
# Output npz keys:
#   doublet_scores          float64[n_cells]  — per-cell doublet score in [0,1]
#   doublet_calls           int32[n_cells]    — binary call (0/1)
#   simulated_doublet_scores float64[n_sim]   — scores on simulated cells
#   threshold               float64 scalar    — FDR threshold applied
#   n_cells                 int32 scalar
#   n_sim                   int32 scalar
#   reference_name          bytes             — which backend was used
#
# Input .bin format (dump_csc.h "CSCC" binary, little-endian, no padding):
#   uint32  magic     = 0x43535343 ("CSCC")
#   uint32  n_rows    (genes or ADT tags)
#   uint32  n_cols    (cells)
#   uint64  nnz
#   float32[nnz]      values
#   int32[n_cols+1]   indptr
#   int32[nnz]        indices

import sys
import argparse
import struct
import math

import numpy as np
import scipy.sparse


# ---------------------------------------------------------------------------
# CSC binary reader (dump_csc.h "CSCC" format)
# ---------------------------------------------------------------------------

def read_csc_bin(path: str) -> scipy.sparse.csc_matrix:
    """Read dump_csc.h binary; return scipy CSC (features × cells)."""
    with open(path, 'rb') as f:
        magic, n_rows, n_cols = struct.unpack('<III', f.read(12))
        if magic != 0x43535343:
            raise ValueError(f"Bad magic in {path}: 0x{magic:08X}")
        (nnz,) = struct.unpack('<Q', f.read(8))
        values  = np.frombuffer(f.read(nnz * 4),          dtype=np.float32).copy()
        indptr  = np.frombuffer(f.read((n_cols + 1) * 4), dtype=np.int32).copy()
        indices = np.frombuffer(f.read(nnz * 4),          dtype=np.int32).copy()
    return scipy.sparse.csc_matrix(
        (values, indices, indptr), shape=(n_rows, n_cols)
    )


# ---------------------------------------------------------------------------
# Attempt to import OmniDoublet and Scrublet
# ---------------------------------------------------------------------------

def _try_omnidoublet():
    try:
        import omnidoublet  # noqa: F401
        return omnidoublet
    except ImportError:
        return None


def _try_scrublet():
    try:
        import scrublet  # noqa: F401
        return scrublet
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Backend 1 — OmniDoublet (multimodal)
# ---------------------------------------------------------------------------

def run_omnidoublet(rna_csc, adt_csc, n_synth_frac, k, n_pcs,
                   expected_rate, seed, omnidoublet_mod):
    """Run OmniDoublet Python reference; returns (scores, calls, sim_scores, threshold)."""
    n_genes, n_cells = rna_csc.shape

    # OmniDoublet expects cells × features (transposed from our genes × cells).
    rna_cx = rna_csc.T.tocsr().astype(np.float32)   # cells × genes
    adt_cx = adt_csc.T.tocsr().astype(np.float32)   # cells × tags

    n_sim = max(1, int(round(n_synth_frac * n_cells * 2)))  # 2× per design doc

    od = omnidoublet_mod.OmniDoublet(
        rna_counts=rna_cx,
        adt_counts=adt_cx,
        n_neighbors=k,
        expected_doublet_rate=expected_rate,
        sim_doublet_ratio=float(n_sim) / max(1, n_cells),
        n_prin_comps=min(n_pcs, min(n_genes, n_cells) - 1),
        random_state=seed,
    )

    scores, calls = od.run_doublet_detection()
    scores = np.asarray(scores, dtype=np.float64)
    calls  = np.asarray(calls,  dtype=np.int32)

    sim_scores = np.asarray(getattr(od, 'simulated_doublet_scores_', []), dtype=np.float64)
    threshold  = float(getattr(od, 'threshold_', 0.5))

    return scores, calls, sim_scores, threshold, "omnidoublet"


# ---------------------------------------------------------------------------
# Backend 2 — Scrublet (RNA-only fallback)
# ---------------------------------------------------------------------------

def run_scrublet(rna_csc, n_synth_frac, k, n_pcs, expected_rate, seed, scrublet_mod):
    """Run Scrublet on RNA only; returns (scores, calls, sim_scores, threshold)."""
    n_genes, n_cells = rna_csc.shape
    counts_cx = rna_csc.T.tocsr().astype(np.float32)  # cells × genes

    n_sim = max(1, int(round(n_synth_frac * n_cells * 2)))

    scrub = scrublet_mod.Scrublet(
        counts_matrix=counts_cx,
        n_neighbors=k,
        expected_doublet_rate=expected_rate,
        sim_doublet_ratio=float(n_sim) / max(1, n_cells),
        random_state=seed,
    )

    try:
        scores, calls = scrub.scrub_doublets(
            min_counts=1,
            min_cells=1,
            min_gene_variability_pctl=0,
            n_prin_comps=min(n_pcs, min(n_genes, n_cells) - 1),
            verbose=False,
        )
    except Exception as e:
        print(f"[omnidoublet_py_reference] scrublet retry with n_prin_comps=5: {e}",
              file=sys.stderr)
        scores, calls = scrub.scrub_doublets(
            min_counts=1, min_cells=1, min_gene_variability_pctl=0,
            n_prin_comps=min(5, min(n_genes, n_cells) - 1), verbose=False,
        )

    scores = np.asarray(scores, dtype=np.float64)
    calls  = np.asarray(calls,  dtype=np.int32)
    sim_scores = np.asarray(getattr(scrub, 'doublet_scores_sim_', []), dtype=np.float64)
    threshold  = float(getattr(scrub, 'threshold_', 0.5))

    return scores, calls, sim_scores, threshold, "scrublet"


# ---------------------------------------------------------------------------
# Backend 3 — Pure-Python Scrublet-style (always present)
#
# Algorithm (mirrors Scrublet core, RNA-only):
#  1. Simulate n_sim synthetic doublets by summing random pairs.
#  2. Log-normalize real + simulated (log1p of library-size-normalized counts).
#  3. PCA via SVD on the real cells, project simulated cells.
#  4. For each real cell compute fraction of its k-NN that are simulated.
#  5. Score = fraction.  Threshold at expected_rate quantile of simulated scores.
# ---------------------------------------------------------------------------

def _log_normalize(mat_cx: scipy.sparse.csc_matrix) -> np.ndarray:
    """Return dense log1p-normalized cells × features, median-scaled."""
    mat_csr = mat_cx.tocsr().astype(np.float64)
    row_sums = np.asarray(mat_csr.sum(axis=1)).ravel()
    median_sum = float(np.median(row_sums[row_sums > 0])) if np.any(row_sums > 0) else 1.0
    # Scale each cell to median library size, then log1p.
    scale = np.where(row_sums > 0, median_sum / row_sums, 1.0)
    # Faster: scale rows via a sparse diagonal multiply.
    from scipy.sparse import diags
    scaled = diags(scale) @ mat_csr
    return np.log1p(scaled.toarray())


def _pca(data: np.ndarray, n_pcs: int, rng: np.random.Generator):
    """Randomized PCA via numpy SVD (sklearn not required)."""
    n, p = data.shape
    k = min(n_pcs, n - 1, p - 1)
    if k <= 0:
        return data[:, :1] * 0.0
    # Center.
    mean = data.mean(axis=0)
    centered = data - mean
    # Randomized SVD via one-pass sketch (fast for moderate k).
    # Power iteration once for accuracy.
    Omega = rng.standard_normal((p, k))
    Y = centered @ Omega
    for _ in range(2):  # 2 power iterations
        Y = centered @ (centered.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ centered
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    return centered @ Vt[:k].T  # n × k


def run_pure_python(rna_csc, n_synth_frac, k, n_pcs, expected_rate, seed):
    """Pure-Python Scrublet-style doublet scoring (RNA-only)."""
    rng = np.random.default_rng(seed)
    n_genes, n_cells = rna_csc.shape
    rna_cx = rna_csc.T  # cells × genes (sparse)

    # 1. Simulate doublets.
    n_sim = max(10, int(round(n_synth_frac * n_cells * 2)))
    i_idx = rng.integers(0, n_cells, size=n_sim)
    j_idx = rng.integers(0, n_cells, size=n_sim)
    # Ensure i != j (best-effort; tiny datasets may have few unique pairs).
    mask = (i_idx == j_idx)
    j_idx[mask] = (j_idx[mask] + 1) % n_cells

    rna_csr = rna_cx.tocsr().astype(np.float32)
    sim_rows = rna_csr[i_idx] + rna_csr[j_idx]           # n_sim × n_genes sparse
    sim_dense = sim_rows.toarray()                        # n_sim × n_genes dense

    real_dense = rna_csr.toarray()                        # n_cells × n_genes

    # 2. Log-normalize.
    def _lognorm_dense(X: np.ndarray) -> np.ndarray:
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        median_sum = float(np.median(X.sum(axis=1)))
        if median_sum <= 0:
            median_sum = 1.0
        return np.log1p(X / row_sums * median_sum)

    real_ln  = _lognorm_dense(real_dense)
    sim_ln   = _lognorm_dense(sim_dense)

    # 3. PCA on real cells; project simulated.
    n_pcs_eff = min(n_pcs, n_cells - 1, n_genes - 1)
    if n_pcs_eff <= 0:
        n_pcs_eff = 1
    pca_real = _pca(real_ln,  n_pcs_eff, rng)           # n_cells × k
    # Project simulated onto the same principal axes.
    mean_real = real_ln.mean(axis=0)
    centered_sim = sim_ln - mean_real
    # Recompute Vt from the real PCA (stored implicitly — redo fast).
    Omega = np.random.default_rng(seed ^ 0xDEAD).standard_normal((n_genes, n_pcs_eff))
    Y = (real_ln - mean_real) @ Omega
    for _ in range(2):
        Y = (real_ln - mean_real) @ ((real_ln - mean_real).T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ (real_ln - mean_real)
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    Vt_k = Vt[:n_pcs_eff]
    pca_real = (real_ln - mean_real) @ Vt_k.T     # n_cells × k
    pca_sim  = centered_sim @ Vt_k.T              # n_sim  × k

    # 4. Concatenate and build kNN: for each real cell find k-NN in all cells.
    all_embed = np.vstack([pca_real, pca_sim])     # (n_cells + n_sim) × k
    is_sim = np.array([False] * n_cells + [True] * n_sim)

    k_eff = min(k, n_cells + n_sim - 1)

    # Batched brute-force kNN (fine for tiny inputs).
    doublet_scores = np.zeros(n_cells, dtype=np.float64)
    for ci in range(n_cells):
        dists = np.sum((all_embed - all_embed[ci]) ** 2, axis=1)
        dists[ci] = np.inf  # exclude self
        nn_idx = np.argpartition(dists, k_eff)[:k_eff]
        doublet_scores[ci] = is_sim[nn_idx].mean()

    # Simulated cell scores.
    sim_scores = np.zeros(n_sim, dtype=np.float64)
    for si in range(n_sim):
        idx = n_cells + si
        dists = np.sum((all_embed - all_embed[idx]) ** 2, axis=1)
        dists[idx] = np.inf
        nn_idx = np.argpartition(dists, k_eff)[:k_eff]
        sim_scores[si] = is_sim[nn_idx].mean()

    # 5. Threshold at (1 - expected_rate) quantile of simulated scores.
    threshold = float(np.percentile(sim_scores, (1.0 - expected_rate) * 100.0))
    calls = (doublet_scores >= threshold).astype(np.int32)

    return doublet_scores, calls, sim_scores, threshold, "pure_python"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OmniDoublet / Scrublet reference for singlet-gpu cycle 39")
    parser.add_argument('--rna-counts',   required=True,
                        help='RNA count matrix .bin (dump_csc format, genes × cells)')
    parser.add_argument('--adt-counts',   required=True,
                        help='ADT count matrix .bin (dump_csc format, tags × cells)')
    parser.add_argument('--out',          required=True,
                        help='output .npz path')
    parser.add_argument('--n-synth-frac', type=float, default=1.0,
                        help='n_sim = n_synth_frac × 2 × n_cells (default 1.0 → n_sim=2×n_cells)')
    parser.add_argument('--k',            type=int,   default=50,
                        help='kNN neighbors (default 50)')
    parser.add_argument('--n-pcs',        type=int,   default=30,
                        help='joint PCA components (default 30)')
    parser.add_argument('--expected-doublet-rate', type=float, default=0.10,
                        help='expected doublet rate for thresholding (default 0.10)')
    parser.add_argument('--seed',         type=int,   default=0,
                        help='random seed')
    args = parser.parse_args()

    try:
        rna_csc = read_csc_bin(args.rna_counts)
        adt_csc = read_csc_bin(args.adt_counts)

        n_genes, n_cells = rna_csc.shape
        n_tags,  n_cells2 = adt_csc.shape
        if n_cells != n_cells2:
            raise ValueError(f"RNA has {n_cells} cells but ADT has {n_cells2}")

        # Try backends in order of preference.
        od_mod = _try_omnidoublet()
        sc_mod = _try_scrublet()

        if od_mod is not None:
            print("[omnidoublet_py_reference] Using OmniDoublet Python backend",
                  file=sys.stderr)
            scores, calls, sim_scores, threshold, name = run_omnidoublet(
                rna_csc, adt_csc,
                n_synth_frac=args.n_synth_frac,
                k=args.k,
                n_pcs=args.n_pcs,
                expected_rate=args.expected_doublet_rate,
                seed=args.seed,
                omnidoublet_mod=od_mod,
            )
        elif sc_mod is not None:
            print("[omnidoublet_py_reference] OmniDoublet absent — using Scrublet (RNA-only)",
                  file=sys.stderr)
            scores, calls, sim_scores, threshold, name = run_scrublet(
                rna_csc,
                n_synth_frac=args.n_synth_frac,
                k=args.k,
                n_pcs=args.n_pcs,
                expected_rate=args.expected_doublet_rate,
                seed=args.seed,
                scrublet_mod=sc_mod,
            )
        else:
            print("[omnidoublet_py_reference] Neither OmniDoublet nor Scrublet installed — "
                  "using pure-Python fallback",
                  file=sys.stderr)
            scores, calls, sim_scores, threshold, name = run_pure_python(
                rna_csc,
                n_synth_frac=args.n_synth_frac,
                k=args.k,
                n_pcs=args.n_pcs,
                expected_rate=args.expected_doublet_rate,
                seed=args.seed,
            )

        # Ensure sim_scores is non-empty (some backends may return empty arrays).
        if len(sim_scores) == 0:
            sim_scores = np.array([threshold], dtype=np.float64)

        np.savez(
            args.out,
            doublet_scores          = scores.astype(np.float64),
            doublet_calls           = calls.astype(np.int32),
            simulated_doublet_scores= sim_scores.astype(np.float64),
            threshold               = np.array([threshold], dtype=np.float64),
            n_cells                 = np.array([n_cells],  dtype=np.int32),
            n_sim                   = np.array([len(sim_scores)], dtype=np.int32),
            reference_name          = np.array([name.encode('ascii')]),
        )

        n_doublets = int(calls.sum())
        call_rate  = n_doublets / max(1, n_cells)
        print(f"[omnidoublet_py_reference] {name}: "
              f"n_cells={n_cells} n_tags={n_tags} "
              f"n_sim={len(sim_scores)} n_doublets={n_doublets} "
              f"call_rate={call_rate:.3f} threshold={threshold:.4f}",
              file=sys.stderr)

    except SystemExit:
        raise
    except Exception as e:
        print(f"[omnidoublet_py_reference] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
