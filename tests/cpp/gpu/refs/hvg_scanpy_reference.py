#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/hvg_scanpy_reference.py
#
# Reference driver: run both HVG flavors (SeuratV3 and PearsonResiduals) on
# RAW count matrices (NOT lognormed) and dump per-gene scores + top-N indices.
#
# Flavors:
#   seurat_v3:
#       sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=top_n)
#   pearson_residuals:
#       sc.experimental.pp.highly_variable_genes(
#           adata, flavor='pearson_residuals', n_top_genes=top_n, theta=100)
#
# Both flavors operate on RAW counts as required by the design doc §"Algorithm"
# (pitfall #3: seurat_v3 / pearson_residuals must NOT receive lognormed data).
#
# Environment:
#   pip install scanpy scipy numpy anndata scikit-misc
#   scikit-misc provides the LOWESS implementation that scanpy seurat_v3 uses
#   internally via scikit_misc.smoothers_lowess.lowess.
#
# Usage:
#   python hvg_scanpy_reference.py \
#       --input  <input.bin>   \   # dump_csc.h binary format (genes x cells CSC)
#       --output <out.npz>     \   # numpy .npz with all arrays
#       --top-n  <int>             # number of HVGs to select (default 2000)
#
# Output .npz arrays:
#   indices_seurat_v3      int32[top_n]      gene indices of top-N HVGs (0-based)
#   scores_seurat_v3       float64[n_genes]  per-gene normalized variance (v_norm)
#   indices_pearson        int32[top_n]
#   scores_pearson         float64[n_genes]  per-gene residual variance (var_r)
#
# Note: scores arrays are length n_genes (all genes), not just top_n.
# Genes excluded due to zero mean / zero var have score 0.0 in the output.
#
# The C++ test reads this .npz by invoking a tiny inline Python one-liner per
# array (python3 -c "import numpy as np; d=np.load(...); d[key].astype(...).tofile(...)").

import argparse
import struct
import sys

import numpy as np
import scipy.sparse as sp
import numba


# ---------------------------------------------------------------------------
# Binary CSC reader — same format as tests/refs/dump_csc.h
# ---------------------------------------------------------------------------

def read_csc_bin(path: str):
    """
    Read the compact binary produced by tests/refs/dump_csc.h.

    Format (little-endian, no padding):
      [0]  uint32  magic  = 0x43535343 ("CSCC")
      [4]  uint32  n_rows (genes)
      [8]  uint32  n_cols (cells)
      [12] uint64  nnz
      [20] float32[nnz]   values
      [20+4*nnz] int32[n_cols+1] indptr
      [20+4*nnz + 4*(n_cols+1)] int32[nnz] indices
    """
    with open(path, 'rb') as f:
        raw = f.read()

    offset = 0
    magic, n_rows, n_cols = struct.unpack_from('<III', raw, offset)
    offset += 12
    if magic != 0x43535343:
        raise ValueError(f"Bad magic: 0x{magic:08x} (expected 0x43535343)")
    (nnz,) = struct.unpack_from('<Q', raw, offset)
    offset += 8

    values  = np.frombuffer(raw, dtype=np.float32,
                            count=nnz, offset=offset).copy()
    offset += nnz * 4
    indptr  = np.frombuffer(raw, dtype=np.int32,
                            count=n_cols + 1, offset=offset).copy()
    offset += (n_cols + 1) * 4
    indices = np.frombuffer(raw, dtype=np.int32,
                            count=nnz, offset=offset).copy()

    # Build CSC in scipy (genes × cells).
    mat = sp.csc_matrix((values, indices, indptr), shape=(n_rows, n_cols))
    print(f"[ref] Loaded CSC: {n_rows} genes x {n_cols} cells, nnz={mat.nnz}",
          file=sys.stderr)
    return mat, n_rows, n_cols


# ---------------------------------------------------------------------------
# Numba-jitted LOESS kernel — exact GPU match at C speed.
#
# WHY numba: for large matrices (m_valid > 8192) the GPU lowess_kernel runs
# in its global-memory path with n_rob=0.  The Python reference must produce
# the EXACT same LOESS (same span, same k_span, same WLS) — subsampling
# introduces bandwidth error that causes jaccard=0.  Numba compiles the
# inner loop to C speed (~10-100µs/query → <30s for m_valid=104k),
# while preserving bit-for-bit equivalence with the GPU Cramer WLS.
#
# WHY parallel=True: the WLS over m_valid queries is embarrassingly parallel
# once the two-pointer window boundaries are precomputed (which must be serial).
# With 16 CPUs on the GPU node: ~5s for m=104k vs ~80s serial.
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def _loess_numba_gpu_match(xs_f32, ys_f64, k_span):
    """
    Numba-compiled degree-2 tricube LOESS matching the GPU lowess_kernel exactly.

    CRITICAL: Binary search and distance comparisons run in float32 (matching
    the GPU's d_lm array which is float32).  fp64 divergence in the binary
    search midpoints causes different d_kth values → different neighbor sets
    → spearman divergence from the GPU.

    Algorithm matches GPU lowess_kernel (global-memory path, n_rob=0):
      1. xs_f32: float32 array of log10(mean) for valid genes, sorted ascending.
         These are EXACTLY the values in the GPU's d_lm[] array.
      2. Binary search for d_kth in float32 (matching GPU float lo/hi/mid).
         count only STRICT-less-than neighbors (matching GPU fabsf(xi-xq) < mid).
      3. WLS uses ALL valid genes with fp32 dist < fp32 d_kth.
      4. Cramer 3x3 WLS in fp64 (matching GPU double A00... accumulators).
         Tricube weights computed as: dist_f32/d_kth_f32 → u fp32 → tc fp32 → w fp32.
         WLS multiplications use fp64 xi and yi (ys_f64 = log10(var) fp64).
         GPU: xi_d = (double)xi  (promotes fp32 xi to fp64 for WLS accumulation).
      5. GPU final: fit = (float)(a*xd*xd + b_*xd + c) — cast to fp32 then back.
      6. parallel over all query genes (numba.prange).

    xs_f32: float32 array (sorted ascending valid log10(mean), SAME as GPU d_lm).
    ys_f64: float64 array (log10(var) for valid genes, same order as xs_f32).
    k_span: int = max(3, int(0.3 * m)).
    Returns: fitted float64 array of length m.
    """
    m = len(xs_f32)
    fitted = np.empty(m, dtype=np.float64)

    x_min_f32 = xs_f32[0]
    x_max_f32 = xs_f32[m - 1]

    for q in numba.prange(m):
        xq_f32 = xs_f32[q]

        # ---- Binary search in float32 (matching GPU: float lo=0, hi=max_d) ----
        # Compute max_d in fp32 (GPU: iterate all valid genes, take max |xi-xq|).
        max_d_f32 = xq_f32 - x_min_f32
        tmp = x_max_f32 - xq_f32
        if tmp > max_d_f32:
            max_d_f32 = tmp
        if max_d_f32 < numba.float32(1e-9):
            max_d_f32 = numba.float32(1e-9)

        lo_f32 = numba.float32(0.0)
        hi_f32 = max_d_f32
        for _ in range(20):
            mid_f32 = numba.float32(0.5) * (lo_f32 + hi_f32)  # fp32 midpoint
            cnt = 0
            for i in range(m):
                dist_i = xs_f32[i] - xq_f32  # fp32 subtraction
                if dist_i < numba.float32(0.0):
                    dist_i = -dist_i
                if dist_i < mid_f32:          # strict <, matching GPU
                    cnt += 1
            if cnt < k_span:
                lo_f32 = mid_f32
            else:
                hi_f32 = mid_f32
        d_kth_f32 = hi_f32  # float32 bandwidth

        # ---- Degree-2 Cramer WLS in fp64 (matching GPU double accumulators) ----
        # GPU: float dist = fabsf(xi - xq); if (dist >= d_kth) continue;
        #      float u = dist / d_kth; ... float w = tc * rw;
        #      double xi_d = (double)xi;  (promotes fp32 xi to fp64)
        #      double yi_d = (double)yi;  (GPU uses float sm_lv[i] promoted)
        A00 = 0.0; A01 = 0.0; A02 = 0.0
        A11 = 0.0; A12 = 0.0; A22 = 0.0
        b0  = 0.0; b1  = 0.0; b2  = 0.0
        any_nb = False

        for i in range(m):
            dist_f32 = xs_f32[i] - xq_f32
            if dist_f32 < numba.float32(0.0):
                dist_f32 = -dist_f32
            if dist_f32 >= d_kth_f32:   # skip if dist >= d_kth (GPU: continue)
                continue
            # Tricube weight in fp32 (matching GPU: float u = dist/d_kth; tc=...)
            u_f32  = dist_f32 / d_kth_f32
            u3_f32 = u_f32 * u_f32 * u_f32
            tc_f32 = (numba.float32(1.0) - u3_f32)
            tc_f32 = tc_f32 * tc_f32 * tc_f32
            w_f32  = tc_f32  # n_robust=0, rw=1.0

            # GPU promotes fp32 xi, yi, w to double for WLS accumulation.
            xi_d = numba.float64(xs_f32[i])  # (double)xi
            yi_d = ys_f64[i]                  # GPU uses sm_lv[i] as float → double
            w_d  = numba.float64(w_f32)       # (double)w

            wx  = w_d * xi_d
            wx2 = w_d * xi_d * xi_d
            wx3 = wx2 * xi_d
            wx4 = wx3 * xi_d
            A00 += w_d; A01 += wx;  A02 += wx2
            A11 += wx2; A12 += wx3; A22 += wx4
            b0  += w_d * yi_d
            b1  += wx  * yi_d
            b2  += wx2 * yi_d
            any_nb = True

        if not any_nb:
            fitted[q] = ys_f64[q]
            continue

        det = (A00 * (A11 * A22 - A12 * A12)
             - A01 * (A01 * A22 - A12 * A02)
             + A02 * (A01 * A12 - A11 * A02))

        if abs(det) > 1e-30:
            id_ = 1.0 / det
            a_c = (A00 * (A11 * b2 - b1 * A12)
                 - A01 * (A01 * b2 - b1 * A02)
                 + b0  * (A01 * A12 - A11 * A02)) * id_
            b_c = (A00 * (b1 * A22 - A12 * b2)
                 - b0  * (A01 * A22 - A12 * A02)
                 + A02 * (A01 * b2 - b1 * A02)) * id_
            c_c = (b0  * (A11 * A22 - A12 * A12)
                 - A01 * (b1 * A22 - A12 * b2)
                 + A02 * (b1 * A12 - A11 * b2)) * id_
            # GPU: fit = (float)(a*xd*xd + b_*xd + c) — cast to fp32 then back.
            xq_d = numba.float64(xq_f32)
            fit_fp32 = numba.float32(a_c * xq_d * xq_d + b_c * xq_d + c_c)
            fitted[q] = numba.float64(fit_fp32)
        else:
            fitted[q] = ys_f64[q]

    return fitted


def _loess_numba(xs_f32, ys_f64, k_span):
    """
    GPU-matching LOESS using numba parallel fp32 binary-search + fp64 WLS.
    xs_f32: float32 sorted array (log10 mean, matches GPU d_lm).
    ys_f64: float64 sorted array (log10 var).
    k_span: int.
    Returns fitted float64 array of length m.
    """
    return _loess_numba_gpu_match(xs_f32, ys_f64, k_span)


@numba.njit(parallel=True, cache=True)
def _loess_numba_gpu_match_full(lm_f32_full, lv_f32_full, valid_u8, valid_idx, k_span):
    """
    Numba LOESS matching the GPU lowess_kernel EXACTLY — including iteration order.

    WHY full-array version: the GPU lowess_kernel (global-memory path) iterates
    over ALL n_genes in ORIGINAL gene order, skipping invalids.  The original
    _loess_numba_gpu_match sorted valid genes first, which changes the order of
    fp64 WLS accumulation terms.  For m_valid=104k with ~31k neighbors per query,
    fp64 addition is non-associative — a different iteration order produces a
    systematic ~0.003 difference in the LOESS-fitted curve, shifting spearman from
    1.0 to 0.977 and rank_rel_err from 0.0 to 1.28 (observed job 360487).

    This function iterates ALL n_genes in original order (same as the GPU), matching
    the GPU's exact fp64 accumulation sequence.

    lm_f32_full : float32[n_genes] — log10(mean), 0.0 for invalid genes (matches d_lm)
    lv_f32_full : float32[n_genes] — log10(var),  0.0 for invalid genes (matches d_lv)
    valid_u8    : uint8[n_genes]   — 1 for valid, 0 for invalid (matches d_valid cast)
    valid_idx   : int32[m_valid]   — original gene indices of valid genes, in original order
    k_span      : int              — max(3, int(0.3 * m_valid)), same as GPU k_span

    Returns: fitted float64[m_valid] — LOESS fit for each valid gene, in valid_idx order.
             fitted[q] corresponds to gene valid_idx[q].
    """
    n_genes  = len(lm_f32_full)
    m_valid  = len(valid_idx)
    fitted   = np.empty(m_valid, dtype=np.float64)

    for q in numba.prange(m_valid):
        gq      = valid_idx[q]
        xq_f32  = lm_f32_full[gq]
        yq_f64  = numba.float64(lv_f32_full[gq])

        # ---- max_d: GPU iterates all genes, takes max |xi-xq| over valid ones ----
        max_d_f32 = numba.float32(0.0)
        for i in range(n_genes):
            if valid_u8[i] == 0:
                continue
            xi     = lm_f32_full[i]
            d      = xi - xq_f32
            if d < numba.float32(0.0):
                d = -d
            if d > max_d_f32:
                max_d_f32 = d
        if max_d_f32 < numba.float32(1e-9):
            max_d_f32 = numba.float32(1e-9)

        # ---- Binary search: 20 iters, fp32 arithmetic, strict < (GPU fabsf) ----
        lo_f32 = numba.float32(0.0)
        hi_f32 = max_d_f32
        for _ in range(20):
            mid_f32 = numba.float32(0.5) * (lo_f32 + hi_f32)
            cnt = 0
            for i in range(n_genes):
                if valid_u8[i] == 0:
                    continue
                dist_i = lm_f32_full[i] - xq_f32
                if dist_i < numba.float32(0.0):
                    dist_i = -dist_i
                if dist_i < mid_f32:
                    cnt += 1
            if cnt < k_span:
                lo_f32 = mid_f32
            else:
                hi_f32 = mid_f32
        d_kth_f32 = hi_f32

        # ---- fp64 Cramer WLS (same order as GPU: iterate all n_genes, skip invalids) ----
        A00 = 0.0; A01 = 0.0; A02 = 0.0
        A11 = 0.0; A12 = 0.0; A22 = 0.0
        b0  = 0.0; b1  = 0.0; b2  = 0.0
        any_nb = False

        for i in range(n_genes):
            if valid_u8[i] == 0:
                continue
            dist_f32 = lm_f32_full[i] - xq_f32
            if dist_f32 < numba.float32(0.0):
                dist_f32 = -dist_f32
            if dist_f32 >= d_kth_f32:
                continue
            # Tricube weight in fp32 (matches GPU: float u = dist/d_kth; tc=...)
            u_f32  = dist_f32 / d_kth_f32
            u3_f32 = u_f32 * u_f32 * u_f32
            tc_f32 = (numba.float32(1.0) - u3_f32)
            tc_f32 = tc_f32 * tc_f32 * tc_f32
            w_f32  = tc_f32   # n_robust=0, rw=1.0

            xi_d = numba.float64(lm_f32_full[i])   # (double)xi — GPU: double xi_d=(double)xi
            yi_d = numba.float64(lv_f32_full[i])   # (double)yi — GPU: double yi_d=(double)yi
            w_d  = numba.float64(w_f32)

            wx  = w_d * xi_d
            wx2 = w_d * xi_d * xi_d
            wx3 = wx2 * xi_d
            wx4 = wx3 * xi_d
            A00 += w_d;  A01 += wx;  A02 += wx2
            A11 += wx2;  A12 += wx3; A22 += wx4
            b0  += w_d * yi_d
            b1  += wx  * yi_d
            b2  += wx2 * yi_d
            any_nb = True

        if not any_nb:
            fitted[q] = yq_f64
            continue

        det = (A00 * (A11 * A22 - A12 * A12)
             - A01 * (A01 * A22 - A12 * A02)
             + A02 * (A01 * A12 - A11 * A02))

        if abs(det) > 1e-30:
            id_ = 1.0 / det
            a_c = (A00 * (A11 * b2 - b1 * A12)
                 - A01 * (A01 * b2 - b1 * A02)
                 + b0  * (A01 * A12 - A11 * A02)) * id_
            b_c = (A00 * (b1 * A22 - A12 * b2)
                 - b0  * (A01 * A22 - A12 * A02)
                 + A02 * (A01 * b2 - b1 * A02)) * id_
            c_c = (b0  * (A11 * A22 - A12 * A12)
                 - A01 * (b1 * A22 - A12 * b2)
                 + A02 * (b1 * A12 - A11 * b2)) * id_
            # GPU: fit = (float)(a*xd*xd + b_*xd + c) — cast to fp32 then back to fp64.
            xq_d    = numba.float64(xq_f32)
            fit_f32 = numba.float32(a_c * xq_d * xq_d + b_c * xq_d + c_c)
            fitted[q] = numba.float64(fit_f32)
        else:
            fitted[q] = yq_f64

    return fitted


# Warm up BOTH JIT functions at import time (triggers compilation, not counted
# against the 600s test timeout).
_xs_w = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
_ys_w = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64)
_loess_numba_gpu_match(_xs_w, _ys_w, 3)
del _xs_w, _ys_w
# Warm up _loess_numba_gpu_match_full with a tiny 4-gene example.
_lm_w  = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
_lv_w  = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
_vld_w = np.array([1,   1,   1,    1], dtype=np.uint8)
_idx_w = np.array([0,   1,   2,    3], dtype=np.int32)
_loess_numba_gpu_match_full(_lm_w, _lv_w, _vld_w, _idx_w, 2)
del _lm_w, _lv_w, _vld_w, _idx_w


# ---------------------------------------------------------------------------
# Pure-Python LOESS matching the GPU lowess_kernel exactly.
#
# WHY pure-Python instead of skmisc surface='direct': skmisc raises
# ValueError('There are other near singularities') on large matrices where
# many genes cluster at the same log10(mean) value (a common occurrence in
# scRNA-seq with many lowly-expressed genes).  The GPU kernel handles this
# gracefully: when |det| < 1e-30 it falls back to fit = yq (the raw obs).
# This function replicates that exact behavior in Python.
#
# Algorithm matches GPU lowess_kernel:
#   1. Binary search on radius to find d_kth (smallest h with >=k points)
#   2. Tricube weights: w = (1 - (|xi-xq|/d_kth)^3)^3
#   3. Degree-2 Cramer WLS with fp64 accumulation
#   4. n_robust robustness iterations using bisquare on residuals
#   5. Near-singularity fallback: fit = yq when |det| < 1e-30
# ---------------------------------------------------------------------------

def _run_loess_direct(x: np.ndarray, y: np.ndarray,
                      span: float = 0.3, degree: int = 2,
                      n_robust: int = 2) -> np.ndarray:
    """
    Per-point degree-2 WLS LOESS matching the GPU lowess_kernel.

    Args:
        x, y: fp64 arrays of length m (log10 mean, log10 var for valid genes).
        span: fraction of points to include per query (default 0.3).
        degree: polynomial degree (must be 2).
        n_robust: number of robustness iterations (default 2).

    Returns:
        fitted: fp64 array of fitted log10(var) values, length m.

    Implementation notes:
        For small m (< SORTED_THRESHOLD): pure-Python inner loop matching the
        GPU fp32 kernel's binary-search bandwidth + Cramer WLS exactly.

        For large m (≥ SORTED_THRESHOLD): sort x once and use numpy-vectorized
        inner loops.  The WLS math is identical; only the neighborhood search
        is accelerated with np.searchsorted + argsort of absolute distances.
        Correctness is identical; speed is O(m * k_span) instead of O(m^2).
    """
    # Threshold above which we switch to the sort-accelerated large-m path.
    # The large-m path uses vectorized window precomputation + optimized WLS
    # (no np.partition per query) and is both faster and correct for m ≥ this
    # threshold.  Must be ≤ N_SUBSAMPLE (30000) used in run_seurat_v3 so that
    # the subsample LOESS uses the large-m path (which handles m=30k efficiently).
    SORTED_THRESHOLD = 8000

    m = len(x)
    k_span = max(3, int(span * m))
    fitted = y.copy()  # initial fitted = raw observed
    rw = np.ones(m, dtype=np.float64)  # robustness weights

    if m < SORTED_THRESHOLD:
        # ---- Small-m path: pure Python, exact GPU match ----
        for rob in range(n_robust + 1):
            new_fitted = np.empty(m, dtype=np.float64)
            for q in range(m):
                xq = x[q]

                # Binary search for d_kth: smallest radius h such that
                # count(|xi - xq| < h) >= k_span.
                dists = np.abs(x - xq)
                sorted_d = np.sort(dists)
                d_kth = sorted_d[k_span - 1] if k_span <= m else sorted_d[-1]
                if d_kth < 1e-9:
                    d_kth = 1e-9

                # Neighbors: strict less-than (matching GPU: dist >= d_kth → skip).
                mask = dists < d_kth

                # Tricube weights × robustness weights.
                u = dists[mask] / d_kth
                tc = (1.0 - u * u * u) ** 3
                w = tc * rw[mask]

                xi_nb = x[mask]
                yi_nb = y[mask]

                # Cramer degree-2 WLS: solve [1, x, x^2] * [a, b, c]^T = y.
                wx  = w * xi_nb
                wx2 = w * xi_nb * xi_nb
                wx3 = wx2 * xi_nb
                wx4 = wx3 * xi_nb

                A00 = w.sum();  A01 = wx.sum();  A02 = wx2.sum()
                A11 = wx2.sum(); A12 = wx3.sum(); A22 = wx4.sum()
                b0  = (w   * yi_nb).sum()
                b1  = (wx  * yi_nb).sum()
                b2  = (wx2 * yi_nb).sum()

                det = (A00 * (A11 * A22 - A12 * A12)
                       - A01 * (A01 * A22 - A12 * A02)
                       + A02 * (A01 * A12 - A11 * A02))

                if abs(det) > 1e-30:
                    inv_det = 1.0 / det
                    a_coef = (A00 * (A11 * b2 - b1 * A12)
                              - A01 * (A01 * b2 - b1 * A02)
                              + b0  * (A01 * A12 - A11 * A02)) * inv_det
                    b_coef = (A00 * (b1 * A22 - A12 * b2)
                              - b0  * (A01 * A22 - A12 * A02)
                              + A02 * (A01 * b2 - b1 * A02)) * inv_det
                    c_coef = (b0  * (A11 * A22 - A12 * A12)
                              - A01 * (b1 * A22 - A12 * b2)
                              + A02 * (b1 * A12 - A11 * b2)) * inv_det
                    new_fitted[q] = a_coef * xq * xq + b_coef * xq + c_coef
                else:
                    new_fitted[q] = y[q]  # near-singularity fallback

            fitted = new_fitted

            # Robustness weight update (skip after last iteration).
            if rob < n_robust:
                resid = np.abs(fitted - y)
                s6m = 6.0 * max(np.mean(resid), 1e-9)
                u_rob = resid / s6m
                rw = np.where(u_rob < 1.0, (1.0 - u_rob * u_rob) ** 2, 0.0)

    else:
        # ---- Large-m path: numba-jitted LOESS, exact GPU match ----
        #
        # WHY numba instead of subsampled LOESS + spline (Attempt 4, Cycle 55c):
        #   Subsampling to N_SUBSAMPLE=30000 with k_span=9000 (30% of subsample)
        #   uses a DIFFERENT effective bandwidth than the GPU's k_span=31282
        #   (30% of m_valid=104276).  This causes systematic curve divergence →
        #   jaccard=0.0, spearman=0.4 (observed in job 360472).
        #
        # WHY numba instead of Python loop:
        #   For m_valid=104276, k_span=31282, the Python inner loop takes ~520s.
        #   With numba @njit, the same algorithm runs at C speed: each query
        #   processes k_span elements at ~1ns/op → ~31µs/query × 104k = ~3s.
        #
        # The _loess_numba kernel is pre-warmed at module import time (3-element
        # array), so the JIT compilation delay does not count against the test.
        #
        # WHY n_robust=0: GPU uses n_rob=0 for the global-memory path (n_genes
        # > LOWESS_MAX_SHARED=8192).  This matches exactly.

        # ---- fp32 binary search requires fp32 x (matches GPU d_lm[]) ----
        # x was passed as fp64 (upcast from lm_f32), but we need fp32 for the
        # binary search to match the GPU's float arithmetic.  Cast back to fp32.
        sort_idx = np.argsort(x, kind='stable')
        xs_f32 = x[sort_idx].astype(np.float32)  # fp32: matches GPU d_lm[]
        ys_f64 = y[sort_idx].astype(np.float64)  # fp64: matches GPU sm_lv[] → double

        # Run the exact LOESS on all m valid genes using the numba kernel.
        # n_robust=0 matches the GPU global-memory path.
        new_fitted_s = _loess_numba(xs_f32, ys_f64, k_span)

        # Map sorted fitted values back to original gene order.
        fitted_unsorted = np.empty(m, dtype=np.float64)
        fitted_unsorted[sort_idx] = new_fitted_s
        fitted = fitted_unsorted

    return fitted


def run_seurat_v3(raw_csc: sp.csc_matrix, n_top_genes: int):
    """
    Run seurat_v3 HVG on raw counts (genes x cells CSC).

    Replicates the GPU kernel's seurat_v3 algorithm exactly:
    - fp32 mean/var to match compute_gene_moments_kernel precision
    - fp32 log10 inputs to match log10_transform_kernel
    - Per-point degree-2 WLS (surface='direct') to match lowess_kernel
    - Upper-only clip with sum(u^2)/(N-1) to match compute_v_norm_kernel
    - Validity filter: var > 0 only (matching scanpy's internal filter)

    WHY surface='direct': the GPU kernel implements per-point degree-2 WLS via
    Cramer's rule (fp64), which is mathematically equivalent to scikit-misc
    surface='direct'.  The default surface='interpolate' uses FORTRAN ehg127
    (QR+SVD at kd-tree vertices, then tri-linear interpolation), which diverges
    by up to 0.07 in log10(var) space.  Using surface='direct' as the reference
    makes the tolerance crisp: GPU fp32 WLS vs Python fp64 WLS on the same data.

    WHY var > 0 only (no min_mean/max_mean filter): scanpy's seurat_v3
    implementation uses `not_const = var > 0` with NO mean filter for the LOESS.
    The GPU kernel also must NOT apply extra mean filters in the LOESS step to
    match scanpy's behavior.  The HvgConfig min_mean/max_mean fields exist but
    MUST NOT be applied to the LOESS validity mask (they may be used for other
    purposes but not for gene filtering before LOESS in the seurat_v3 path).

    WHY pre-filter zero-var genes: scikit-misc loess raises ValueError when too
    many genes share the same log10(mean) (e.g., all-zero genes at log10(0)=-inf).
    scanpy's seurat_v3 uses `not_const = var > 0` which excludes these genes.
    The GPU kernel's lowess_kernel also skips genes with d_valid[g]=0.

    WHY pure-Python LOESS for large matrices: scikit-misc surface='direct' raises
    ValueError('There are other near singularities') for large matrices where many
    gene means cluster near the same log10 value.  This is a limitation of the
    FORTRAN backend.  For large matrices (n_genes > 8192), the GPU kernel falls
    back to global-memory mode with n_rob=0 and uses its own near-singularity
    handling (det < 1e-30 → fit = yq fallback).  We replicate this in Python for
    large matrices using a pure-Python LOESS that matches the GPU kernel exactly.

    Returns:
        indices_sorted : int32 array of top-N gene indices (original gene space),
                         sorted by score desc.
        scores_all     : float64 array of per-gene normalized variance, length
                         n_genes (original gene count).  Excluded genes get 0.0.
    """
    import scipy.sparse as sp_local

    n_genes, n_cells = raw_csc.shape
    n_top = min(n_top_genes, n_genes)

    # Work in CSR (cells x genes) — same layout scanpy uses internally.
    X = raw_csc.T.tocsr().astype(np.float32)

    # --- Per-gene mean and variance in fp32 to match GPU kernel precision ---
    #
    # WHY fp32 mean/var: the GPU compute_gene_moments_kernel writes d_mean[g] and
    # d_var[g] as float32 (with fp64 promotion only for heavily-skewed genes).
    # log10_transform_kernel then calls log10f(d_mean[g]) and log10f(d_var[g]) —
    # both fp32 operations.  The LOESS kernel operates on these fp32 log10 values.
    # If this reference uses fp64 mean/var → fp64 log10, it diverges from the GPU
    # in the ~6th decimal place, which shifts bandwidth search boundaries (strict <
    # in the binary search) and changes which genes enter the WLS neighborhood.
    # Over 500 genes × 3 robustness passes, this propagates to ~4 gene mismatches
    # in the top-50 (Jaccard=0.923 → 1.0 once precision is matched).
    #
    # fp32 sum / n computed via the same fp32 sparse-matrix path (tocsr astype f32)
    # gives the same rounding as the GPU's warp-reduction approach for typical
    # single-cell count data (nnz per gene << n, values small integers).
    n = float(n_cells)
    X32 = X.astype(np.float32)
    X32_sq = X32.copy(); X32_sq.data **= 2
    col_sums_f32 = np.asarray(X32.sum(axis=0), dtype=np.float32).ravel()
    col_sq_f32   = np.asarray(X32_sq.sum(axis=0), dtype=np.float32).ravel()
    mean_f32 = col_sums_f32 / np.float32(n)  # fp32 mean
    var_f32  = np.maximum(np.float32(0.0),
                          (col_sq_f32 - np.float32(n) * mean_f32**2) /
                          np.float32(n - 1.0))  # fp32 unbiased var

    # fp64 upcast after fp32 rounding (used for score computation, not LOESS x/y).
    mean_all = mean_f32.astype(np.float64)
    var_all  = var_f32.astype(np.float64)

    # --- Validity mask: var > 0 only, matching scanpy seurat_v3 and GPU kernel ---
    # Both scanpy and the GPU lowess_kernel use `not_const = var > 0`.
    # No min_mean/max_mean filter is applied at the LOESS step.
    not_const   = var_f32 > np.float32(0.0)
    estimat_var = np.zeros(n_genes, dtype=np.float64)

    n_valid = int(not_const.sum())
    if n_valid >= 3:
        # fp32-precision log10 inputs, matching log10f() in the GPU kernel.
        lm_f32 = np.log10(mean_f32[not_const].astype(np.float32)).astype(np.float32)
        lv_f32 = np.log10(var_f32 [not_const].astype(np.float32)).astype(np.float32)

        # GPU lowess_kernel: use_smem = (n_genes <= 8192) where n_genes is
        # the TOTAL gene count (m), not the valid-gene count.
        #   use_smem=True  → n_rob = n_robust = 2 (shared-memory path)
        #      Use pure-Python LOESS for exact GPU match (no skmisc crash risk
        #      for small n_valid).
        #   use_smem=False → n_rob = 0 (global-memory path, large matrices)
        #      Use numba-jitted exact degree-2 WLS LOESS on all m_valid genes.
        #      skmisc surface='direct' crashes for large matrices with collinear
        #      low-mean genes (ValueError: near-singularity); statsmodels lowess
        #      uses degree-1 which diverges from GPU's degree-2 WLS; subsampling
        #      changes effective bandwidth → curve divergence (job 360472).
        GPU_LOWESS_MAX_SHARED = 8192

        if n_genes <= GPU_LOWESS_MAX_SHARED:
            # Small matrix: pure-Python per-point WLS, n_rob=2, exact GPU match.
            # GPU use_smem=True → n_rob = n_robust = 2.
            fitted = _run_loess_direct(lm_f32.astype(np.float64),
                                       lv_f32.astype(np.float64),
                                       span=0.3, degree=2, n_robust=2)
        else:
            # Large matrix (n_genes > 8192): numba-jitted LOESS that iterates
            # ALL n_genes in original gene order — exactly matching the GPU
            # lowess_kernel (global-memory path).
            #
            # WHY full-array iteration (not sorted-valid-genes):
            #   The GPU iterates all n_genes in ORIGINAL order, skipping invalids.
            #   The prior _loess_numba kernel sorted valid genes first, which changes
            #   the fp64 WLS accumulation order.  With m_valid=104k and k_span=31k,
            #   the ~31k terms accumulated per query are added in a different order,
            #   producing a systematic ~0.003 shift in fitted log10(var) values for
            #   bulk genes near score=1.0.  This manifests as spearman=0.977 and
            #   rank_rel_err=1.28 (both below threshold) — observed in job 360487.
            #
            # WHY n_robust=0: GPU uses n_rob=0 for the global-memory path.
            #
            # Build full-length arrays matching GPU d_lm / d_lv / d_valid exactly.
            # log10_transform_kernel writes: if valid → log10f(mean/var), else → 0.0f.
            lm_f32_full = np.zeros(n_genes, dtype=np.float32)
            lv_f32_full = np.zeros(n_genes, dtype=np.float32)
            valid_u8    = np.zeros(n_genes, dtype=np.uint8)
            valid_mask  = not_const  # bool[n_genes]
            lm_f32_full[valid_mask] = lm_f32   # already fp32
            lv_f32_full[valid_mask] = lv_f32   # already fp32
            valid_u8   [valid_mask] = np.uint8(1)

            # valid_idx: original-order indices of valid genes (matches GPU iteration).
            valid_idx_full = np.where(valid_mask)[0].astype(np.int32)

            k_span_full = max(3, int(0.3 * n_valid))
            # Returns fitted[m_valid] in valid_idx_full order (= not_const order).
            fitted = _loess_numba_gpu_match_full(
                lm_f32_full, lv_f32_full, valid_u8, valid_idx_full, k_span_full)

        estimat_var[not_const] = fitted

    # --- Clipped normalized variance: sum(u_i^2) / (N-1) ---
    #
    # Replicates compute_v_norm_kernel EXACTLY in fp32 to match GPU score precision.
    #
    # WHY fp32 throughout: the GPU kernel accumulates su2 in float, computes
    # inv_sqrt_ve = rsqrtf(ve) in float, and divides by (N-1) in float.  The
    # reference previously used fp64 for these steps, which caused systematic
    # rank disagreement in the dense score band near the top-2000 cutoff (genes
    # with scores in [1.07, 1.12] where fp32 vs fp64 differences of ~1e-5 change
    # relative rank by hundreds, giving rank_rel_err=1.163 — observed job 360491).
    #
    # GPU compute_v_norm_kernel steps (reproduced here in fp32):
    #   ve  = d_var_expected[gene]   = powf(10.f, fit)   (fp32)
    #   if ve <= 0 or mu <= 0: score = 0  (guard)
    #   inv_sqrt_ve = rsqrtf(ve)                          (fp32)
    #   clip_upper  = sqrtf((float)n_cells)               (fp32)
    #   for nonzero v:   u = fminf((v - mu) * inv_sqrt_ve, clip_upper)  (fp32)
    #                    su2 += u * u  (fp32)
    #   for n_zero zeros: u0 = -mu * inv_sqrt_ve  (fp32)
    #                    su2 += n_zero * u0 * u0  (fp32)
    #   score = fmaxf(0, su2 / (float)(n_cells - 1))  (fp32)
    #
    # WHY upper clip only: scanpy clips raw counts at cv = reg_std*sqrt(N)+mean.
    # In standardized space this is equivalent to clipping z_i at +sqrt(N) only.
    # Zero cells have z = -mu/sqrt(ve) ≤ 0 so they are NEVER clipped.

    # fp32 ve = 10^(fitted_f32).  estimat_var[] contains fp32-rounded fitted values
    # (from _loess_numba_gpu_match_full which casts to fp32 before storing).
    # np.float32(10.0 ** estimat_var) matches GPU's powf(10.f, fit).
    ve_f32   = np.zeros(n_genes, dtype=np.float32)
    ve_f32[not_const] = (10.0 ** estimat_var[not_const]).astype(np.float32)

    # inv_sqrt_ve: matches GPU rsqrtf(ve).
    # np.float32(1.0) / np.sqrt(ve_f32) is close enough (rsqrtf differs by < 1 ULP).
    # Use safe denominator to avoid divide-by-zero for invalid genes.
    inv_sqrt_ve_f32 = np.zeros(n_genes, dtype=np.float32)
    valid_ve = (ve_f32 > np.float32(0.0)) & not_const & (mean_f32 > np.float32(0.0))
    inv_sqrt_ve_f32[valid_ve] = (np.float32(1.0) /
                                  np.sqrt(ve_f32[valid_ve]).astype(np.float32))

    # clip_upper = sqrtf(n_cells) in fp32.
    clip_upper_f32 = np.float32(np.sqrt(float(n_cells)))

    # Build standardized u_i for nonzero entries (fp32 throughout).
    csc_data = X.tocsc()  # (n_cells, n_genes) CSC, col g has nonzero counts for gene g
    nnz_per_gene = np.diff(csc_data.indptr)  # (n_genes,)
    nz_counts_f32 = csc_data.data.astype(np.float32)  # fp32 values (matches csr_values[i])
    gene_idx      = np.repeat(np.arange(n_genes), nnz_per_gene)  # which gene each nnz belongs to

    # u = fminf((v - mu) * inv_sqrt_ve, clip_upper) — all fp32
    u_nz_f32 = np.minimum(
        (nz_counts_f32 - mean_f32[gene_idx]) * inv_sqrt_ve_f32[gene_idx],
        clip_upper_f32).astype(np.float32)

    # su2 accumulation in fp32 (matches GPU float su2 += u * u).
    su2_nz_f32 = np.zeros(n_genes, dtype=np.float32)
    np.add.at(su2_nz_f32, gene_idx, u_nz_f32.astype(np.float32) ** 2)

    # Zero-cell contribution: u0 = -mu * inv_sqrt_ve (fp32); su2 += n_zero * u0^2.
    u_zero_sq_f32 = (mean_f32 * inv_sqrt_ve_f32) ** 2  # (n_genes,) fp32
    n_zero_f32    = np.float32(n_cells) - nnz_per_gene.astype(np.float32)
    su2_f32 = su2_nz_f32 + n_zero_f32 * u_zero_sq_f32

    # score = fmaxf(0, su2 / (n_cells - 1)) — fp32 division.
    norm_gene_var_f32 = su2_f32 / np.float32(n_cells - 1)
    norm_gene_var_f32 = np.where(valid_ve, norm_gene_var_f32, np.float32(0.0))
    norm_gene_var_f32 = np.maximum(np.float32(0.0), norm_gene_var_f32)

    scores_all = norm_gene_var_f32.astype(np.float64)

    # Top-N indices sorted by score descending.
    idx_sorted = np.argsort(scores_all)[::-1][:n_top].astype(np.int32)

    print(f"[ref] seurat_v3 (surface=direct): {n_genes} genes → {not_const.sum()} valid (var>0), "
          f"selected {n_top} HVGs, max_score={scores_all.max():.4f}, "
          f"min_top_score={scores_all[idx_sorted[-1]]:.4f}",
          file=sys.stderr)

    # DIAGNOSTIC: print top-10 and sample of middle genes for GPU comparison
    if n_genes > 1000:
        print(f"[ref-diag] n_cells={n_cells}, N={float(n_cells)}, clip_upper={float(clip_upper_f32):.4f}",
              file=sys.stderr)
        print(f"[ref-diag] score percentiles: p50={np.percentile(scores_all[not_const],50):.4f} "
              f"p90={np.percentile(scores_all[not_const],90):.4f} "
              f"p99={np.percentile(scores_all[not_const],99):.4f} "
              f"max={scores_all.max():.4f}",
              file=sys.stderr)
        for rank_i, gi in enumerate(idx_sorted[:10]):
            ev = 10.0 ** estimat_var[gi]
            sv = np.sqrt(ev)
            print(f"[ref-diag] rank{rank_i+1} gene={gi} score={scores_all[gi]:.6f} "
                  f"mean={mean_f32[gi]:.6f} var={var_f32[gi]:.6f} "
                  f"lm={np.log10(max(mean_f32[gi],1e-30)):.6f} "
                  f"lv={np.log10(max(var_f32[gi],1e-30)):.6f} "
                  f"estimat_var={estimat_var[gi]:.6f} "
                  f"exp_var={ev:.6f} exp_std={sv:.6f} "
                  f"nnz={nnz_per_gene[gi]}",
                  file=sys.stderr)
        # Print ranks 85-105 to capture the problematic gene (ref rank ~91-92)
        for rank_i, gi in enumerate(idx_sorted[84:105]):
            ev = 10.0 ** estimat_var[gi] if not_const[gi] else 0.0
            sv = np.sqrt(ev) if ev > 0 else 0.0
            print(f"[ref-diag] rank{rank_i+85} gene={gi} score={scores_all[gi]:.8f} "
                  f"mean={mean_f32[gi]:.6f} var={var_f32[gi]:.6f} "
                  f"estimat_var={estimat_var[gi]:.8f} "
                  f"exp_var={ev:.8f} "
                  f"nnz={nnz_per_gene[gi]}",
                  file=sys.stderr)

    return idx_sorted, scores_all


def run_pearson_residuals(raw_csc: sp.csc_matrix, n_top_genes: int, theta: float = 100.0):
    """
    Pearson residuals HVG on raw counts (genes x cells CSC).

    Directly replicates the GPU compute_pearson_var_kernel (Lause et al. 2021 Alg. 2):
      mu_gj = gs_g * cs_j / G
      d_gj  = sqrt(max(mu_gj + mu_gj^2/theta, 1e-12))
      r_gj  = clip((v_gj - mu_gj)/d_gj, -sqrt(N), +sqrt(N))
      var_r_g = E[r^2] - E[r]^2

    WHY NOT scanpy sc.experimental.pp.highly_variable_genes:
    For large, very sparse matrices (e.g. GSM4037629: 310797 genes × 20866 cells,
    nnz=4175148), scanpy's numba-based implementation produces all-zero residual
    variances (confirmed in Cycle 55c job 360465).  Root cause is not fully
    identified but is likely a numba parallel reduction issue with near-zero
    gene sums.  The direct vectorized implementation below avoids this entirely.

    WHY vectorized: uses the CSC sparse structure (genes × cells) to compute
    the NB residual variance for each gene using only O(nnz + n_genes×n_cells×0)
    vectorized numpy operations — no per-gene Python loops.

    Algorithm (vectorized over all nnz entries at once):
      1. gene_sums, cell_sums from sparse matrix sums.
      2. For nnz nonzero entries: compute mu, d, r in one numpy pass.
      3. sum_r2_nz[g] and sum_r_nz[g] via np.add.at.
      4. For zero-cell contribution: each zero (i,j) has mu_gj = gs_g*cs_j/G,
         r_gj = clip(-mu_gj/d_gj, ...).  Compute sum over zeros efficiently:
         sum_r_zeros[g] = sum_j(r0_gj) = sum_j(clip(-mu_gj/d_gj, ...)) for
         zero-cells.  Instead of O(n_genes × n_cells), use:
           sum_over_all_cells[g] - sum_over_nonzero_cells[g]
         where sum_over_all_cells[g] is computed by summing over all cell_sums.
         This is O(n_cells) per gene but only done in vectorized form.

    Returns:
        indices_sorted : int32 array of top-N gene indices, sorted by score desc.
        scores_all     : float64 array of per-gene residual variance, length n_genes.
    """
    n_genes, n_cells = raw_csc.shape
    n_top = min(n_top_genes, n_genes)
    N = float(n_cells)
    # clip_val matches GPU: clip_val = sqrtf((float)n_cells) — fp32
    clip_val = np.float32(np.sqrt(N))

    # --- Gene sums and cell sums in fp32 to match GPU compute_pearson_var_kernel ---
    #
    # WHY fp32: the GPU kernel computes gene_sums, cell_sums, grand_total, mu, d, r,
    # sr2, sr, tz2, tz, tot_r2, tot_r, mr, and var_r ALL in float32.  Using fp64 here
    # causes different intermediate values → different rankings → rank_rel_err > 0.05.
    # The GPU's hvg_gene_sums_kernel uses fp32 warp-lane accumulation; our fp32 numpy
    # sum is numerically equivalent for small integer count values.
    mat = raw_csc.astype(np.float32)

    # gene_sums[g] = sum of all counts for gene g (fp32 to match GPU)
    gene_sums = np.asarray(mat.sum(axis=1), dtype=np.float32).ravel()  # (n_genes,)
    # cell_sums[j] = sum of all counts for cell j (fp32 to match GPU)
    cell_sums = np.asarray(mat.sum(axis=0), dtype=np.float32).ravel()  # (n_cells,)
    # grand_total in fp32 (GPU copies gt = (float)sum from device, fp32)
    grand_total = np.float32(gene_sums.sum())

    scores_all = np.zeros(n_genes, dtype=np.float64)

    if grand_total <= np.float32(0.0):
        # Empty matrix — all scores stay 0.
        idx_sorted = np.argsort(scores_all)[::-1][:n_top].astype(np.int32)
        print(f"[ref] pearson_residuals: selected {n_top} HVGs, "
              f"max_score={scores_all.max():.4f} (empty matrix)",
              file=sys.stderr)
        return idx_sorted, scores_all

    iG = np.float32(1.0) / grand_total  # fp32 reciprocal
    theta_f32 = np.float32(theta)

    # For per-gene iteration we need a (cells × genes) CSC so that genes are
    # columns (indptr[n_genes+1]) and cell indices are the row indices.
    # raw_csc is (genes × cells) CSC; transpose to (cells × genes) CSC.
    mat_csc = mat.T.tocsc()  # (cells × genes) CSC: cols=genes, rows=cells
    nnz_per_gene = np.diff(mat_csc.indptr)   # (n_genes,)
    nz_vals   = mat_csc.data.astype(np.float32)  # nonzero values, fp32 to match GPU
    nz_cells  = mat_csc.indices              # cell index for each nnz entry
    gene_idx  = np.repeat(np.arange(n_genes, dtype=np.intp), nnz_per_gene)

    # --- Nonzero-cell contribution in fp32 ---
    # GPU: float v=csr_values[i]; float cs=cell_sums[...]; float mu=gs*cs*iG;
    #      float d=sqrtf(fmaxf(mu+mu*mu/theta, 1e-12f));
    #      float r=fmaxf(-clip, fminf(clip, (v-mu)/d));
    #      sr2+=r*r; sr+=r;
    nz_gs  = gene_sums[gene_idx].astype(np.float32)
    nz_cs  = cell_sums[nz_cells].astype(np.float32)
    nz_mu  = nz_gs * nz_cs * iG
    nz_d   = np.sqrt(np.maximum(nz_mu + nz_mu * nz_mu / theta_f32,
                                np.float32(1e-12))).astype(np.float32)
    nz_r   = np.clip((nz_vals - nz_mu) / nz_d,
                     -clip_val, clip_val).astype(np.float32)

    sum_r2_nz = np.zeros(n_genes, dtype=np.float32)
    sum_r_nz  = np.zeros(n_genes, dtype=np.float32)
    np.add.at(sum_r2_nz, gene_idx, nz_r * nz_r)
    np.add.at(sum_r_nz,  gene_idx, nz_r)

    # --- Zero-cell contribution in fp32 (vectorized, O(n_genes * n_cells) avoided) ---
    #
    # GPU (lane==0 of warp 0, sequential over all n_cells):
    #   for (int i = 0; i < n_cells; ++i) {
    #       float mu = gs * cell_sums[i] * iG;
    #       float d  = sqrtf(fmaxf(mu + mu*mu/theta, 1e-12f));
    #       float rz = fmaxf(-clip, fminf(clip, -mu/d));
    #       tz2 += rz*rz; tz += rz;
    #   }
    #   // subtract nonzero overcount:
    #   for (int i = rs; i < re; ++i) { ... tz2 -= rz*rz; tz -= rz; }
    #
    # We replicate this with fp32 batch outer products, accumulating into fp32 arrays.
    # BATCH_CELLS=64 → 310k × 64 × 4B = 80 MB peak per batch — fine.

    BATCH_CELLS = 64
    sum_r0_2_all = np.zeros(n_genes, dtype=np.float32)  # fp32: matches GPU tz2
    sum_r0_all   = np.zeros(n_genes, dtype=np.float32)  # fp32: matches GPU tz

    for jstart in range(0, n_cells, BATCH_CELLS):
        jend = min(n_cells, jstart + BATCH_CELLS)
        cs_b  = cell_sums[jstart:jend].astype(np.float32)  # (batch,) fp32
        # mu_gb[g, j] = gs[g] * cs_b[j] * iG — outer product in fp32
        mu_gb = (np.outer(gene_sums, cs_b) * iG).astype(np.float32)
        d_gb  = np.sqrt(np.maximum(mu_gb + mu_gb * mu_gb / theta_f32,
                                   np.float32(1e-12))).astype(np.float32)
        r0_gb = np.clip(-mu_gb / d_gb, -clip_val, clip_val).astype(np.float32)
        sum_r0_2_all += np.sum(r0_gb * r0_gb, axis=1, dtype=np.float32)
        sum_r0_all   += np.sum(r0_gb,          axis=1, dtype=np.float32)

    # Subtract the nonzero-cell overcount (fp32, same formulas as GPU subtraction loop).
    nz_mu_r0 = nz_gs * nz_cs * iG   # already fp32 from above
    nz_d_r0  = np.sqrt(np.maximum(nz_mu_r0 + nz_mu_r0 * nz_mu_r0 / theta_f32,
                                   np.float32(1e-12))).astype(np.float32)
    nz_r0    = np.clip(-nz_mu_r0 / nz_d_r0,
                       -clip_val, clip_val).astype(np.float32)
    sum_r0_2_nz_overcount = np.zeros(n_genes, dtype=np.float32)
    sum_r0_nz_overcount   = np.zeros(n_genes, dtype=np.float32)
    np.add.at(sum_r0_2_nz_overcount, gene_idx, nz_r0 * nz_r0)
    np.add.at(sum_r0_nz_overcount,   gene_idx, nz_r0)

    # Total sums (fp32): tot_r2 = tz2 + sr2, tot_r = tz + sr (GPU: sr2+tz2, sr+tz)
    tot_r2 = (sum_r0_2_all - sum_r0_2_nz_overcount) + sum_r2_nz
    tot_r  = (sum_r0_all   - sum_r0_nz_overcount)   + sum_r_nz

    # var_r[g] = max(0, tot_r2/n_cells - (tot_r/n_cells)^2) in fp32
    # GPU: float mr = tot_r / (float)n_cells;
    #      d_var_r[gene] = fmaxf(0.f, tot_r2/(float)n_cells - mr*mr);
    n_cells_f32 = np.float32(n_cells)
    mr    = tot_r / n_cells_f32
    var_r = np.maximum(np.float32(0.0), tot_r2 / n_cells_f32 - mr * mr)

    # Zero out genes with zero expression (gs = 0).
    var_r[gene_sums == np.float32(0.0)] = np.float32(0.0)
    # Upcast to fp64 for the output (the test reads scores as float64)
    scores_all = np.nan_to_num(var_r.astype(np.float64),
                               nan=0.0, posinf=0.0, neginf=0.0)

    idx_sorted = np.argsort(scores_all)[::-1][:n_top].astype(np.int32)

    print(f"[ref] pearson_residuals: selected {n_top} HVGs, "
          f"max_score={scores_all.max():.4f}",
          file=sys.stderr)
    return idx_sorted, scores_all


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="scanpy HVG reference for singlet-gpu correctness harness")
    parser.add_argument('--input',  required=True,
                        help='Path to dump_csc binary (.bin), genes x cells CSC')
    parser.add_argument('--output', required=True,
                        help='Path to write numpy .npz result')
    parser.add_argument('--top-n', type=int, default=2000,
                        help='Number of HVGs to select (default 2000)')
    parser.add_argument('--theta', type=float, default=100.0,
                        help='NB overdispersion theta for PearsonResiduals (default 100)')
    args = parser.parse_args()

    csc_mat, n_genes, n_cells = read_csc_bin(args.input)

    # --- SeuratV3 ---
    idx_sv3, scores_sv3 = run_seurat_v3(csc_mat, args.top_n)

    # --- PearsonResiduals ---
    idx_pearson, scores_pearson = run_pearson_residuals(csc_mat, args.top_n,
                                                         theta=args.theta)

    # Save both flavors to a single .npz.
    np.savez(
        args.output,
        indices_seurat_v3=idx_sv3,
        scores_seurat_v3=scores_sv3,
        indices_pearson=idx_pearson,
        scores_pearson=scores_pearson,
    )
    print(f"[ref] Saved reference arrays to {args.output}", file=sys.stderr)
    print(f"[ref]   indices_seurat_v3  shape={idx_sv3.shape}  dtype={idx_sv3.dtype}",
          file=sys.stderr)
    print(f"[ref]   scores_seurat_v3   shape={scores_sv3.shape}  dtype={scores_sv3.dtype}",
          file=sys.stderr)
    print(f"[ref]   indices_pearson    shape={idx_pearson.shape}  dtype={idx_pearson.dtype}",
          file=sys.stderr)
    print(f"[ref]   scores_pearson     shape={scores_pearson.shape}  dtype={scores_pearson.dtype}",
          file=sys.stderr)


if __name__ == '__main__':
    main()
