#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/refs/dpt_ref.py
#
# CYCLE-161 Phase E — scanpy CPU baseline for embed/dpt.
#
# §J.6 AT-RISK audit: dpt uses the same dense cuSOLVER Ssyevd kernel as
# diffmap. Hypothesis: GPU dpt at n=10k will be 5-20× slower than scanpy's
# sparse ARPACK-based sc.tl.dpt. This script provides the scanpy CPU baseline.
#
# Generates synthetic dense PCA embeddings (n_cells x n_pcs) matching the GPU
# bench, then times only sc.tl.dpt() after untimed sc.pp.neighbors and
# sc.tl.diffmap (DPT requires diffmap eigenvectors in adata.obsm['X_diffmap']).
#
# Protocol:
#   - sc.pp.neighbors(adata, n_neighbors=10, use_rep='X') — untimed.
#   - sc.tl.diffmap(adata, n_comps=15)                    — untimed warmup.
#   - adata.uns['iroot'] = 0                              — root cell.
#   - sc.tl.dpt(adata) — 2 warmup + 5 timed iterations.
#   - Wall time covers only sc.tl.dpt (not neighbors or diffmap steps).
#
# NOTE on sc.tl.dpt with synthetic random data:
#   sc.tl.dpt requires adata.uns['iroot'] (root cell index) and uses the
#   diffmap eigenvectors from adata.obsm['X_diffmap']. With random synthetic
#   PCA input and iroot=0, the call succeeds but pseudotime is uniformly
#   distributed (as expected for random data). This is intentional — we are
#   measuring wall time only, not correctness.
#
# Scales:
#   - 10k: n_cells=10000, n_pcs=50. DPT ARPACK is tractable (~1-10s).
#   - 30k: n_cells=30000, n_pcs=50. Scanpy ARPACK should be tractable
#     (~15-30s for 30k); included to provide upper-bound reference even if
#     GPU crashes at 30k (same cuSOLVER pattern as diffmap).
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_pcs,k,n_eigenvecs,wall_ms
#
# Usage: python3 dpt_ref.py

import sys
import time
import copy

import numpy as np

WARMUP_ITERS  = 2
TIMED_ITERS   = 5
N_NEIGHBORS   = 10
N_EIGENVECS   = 15   # DPT n_comps for diffmap (n_eigenvecs in GPU bench)
ROOT_CELL     = 0

SCALES = [
    ("10k",  10000, 50),
    ("30k",  30000, 50),  # scanpy ARPACK tractable; GPU 30k skipped (crash risk)
]


def make_synthetic_embedding(n_cells: int, n_pcs: int, seed: int = 42) -> np.ndarray:
    """Build a synthetic dense PCA embedding (n_cells x n_pcs, float32).

    Uses np.random.default_rng(seed).standard_normal to match a standardized
    PCA coordinate distribution. Same seed and shape as the GPU bench driver.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_cells, n_pcs)).astype(np.float32)


def build_adata(embedding: np.ndarray):
    """Wrap a numpy embedding array in an AnnData object with X = embedding."""
    import anndata as ad
    adata = ad.AnnData(X=embedding)
    return adata


def run_neighbors_and_diffmap(adata, n_neighbors: int = N_NEIGHBORS,
                               n_comps: int = N_EIGENVECS):
    """Build kNN + diffmap (both untimed — prerequisites for sc.tl.dpt).

    sc.pp.neighbors with use_rep='X' builds the graph from adata.X.
    sc.tl.diffmap populates adata.obsm['X_diffmap'] required by sc.tl.dpt.
    adata.uns['iroot'] = ROOT_CELL sets the root cell index.
    n_neighbors=10, n_comps=15 match GPU bench config.
    """
    import scanpy as sc
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.diffmap(adata, n_comps=n_comps)
    adata.uns['iroot'] = ROOT_CELL


def time_dpt(adata) -> float:
    """2 warmup + 5 timed iters of sc.tl.dpt. Returns median wall_ms.

    A fresh copy of adata is made per iteration so scanpy cannot reuse stale
    state between runs. The neighbors graph and diffmap eigenvectors are
    already embedded in adata.uns and adata.obsm.

    NOTE: sc.tl.dpt modifies adata in-place (adds adata.obs['dpt_pseudotime']).
    copy.copy() is a shallow copy; for AnnData this copies the obs/obsm/uns
    references but dpt writes only to obs which is replaced — sufficient for
    independent timing.
    """
    import scanpy as sc
    import anndata as ad

    def _run(ad_in):
        # Use AnnData copy to avoid state leakage between timed runs.
        a = ad_in.copy()
        sc.tl.dpt(a)

    # Warmup iterations (populate JIT/ARPACK caches, pre-allocate buffers).
    for _ in range(WARMUP_ITERS):
        try:
            _run(adata)
        except Exception as exc:
            print(f"[ref] warmup error (ignored): {exc}", file=sys.stderr, flush=True)

    wall_times = []
    for i in range(TIMED_ITERS):
        t0 = time.perf_counter()
        _run(adata)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        wall_times.append(elapsed_ms)
        print(f"[ref] timed iter {i+1}/{TIMED_ITERS}: {elapsed_ms:.1f} ms",
              file=sys.stderr, flush=True)

    wall_times.sort()
    n = len(wall_times)
    median_ms = (wall_times[n // 2] if n % 2 == 1
                 else 0.5 * (wall_times[n // 2 - 1] + wall_times[n // 2]))
    return median_ms


def main():
    import importlib.util
    for pkg in ("anndata", "scanpy"):
        if importlib.util.find_spec(pkg) is None:
            print(f"ERROR: {pkg} not installed", file=sys.stderr)
            sys.exit(1)

    import scanpy as sc
    # Silence scanpy verbose output to keep stdout clean (CSV only).
    sc.settings.verbosity = 0

    print("scale,n_cells,n_pcs,k,n_eigenvecs,wall_ms", flush=True)

    for scale_name, n_cells, n_pcs in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_cells} cells x {n_pcs} PCs "
              f"(dense, float32)...", file=sys.stderr, flush=True)

        embedding = make_synthetic_embedding(n_cells, n_pcs, seed=42)
        adata = build_adata(embedding)

        print(f"[ref] Building kNN graph n_neighbors={N_NEIGHBORS} + diffmap "
              f"n_comps={N_EIGENVECS} for {scale_name} (untimed)...",
              file=sys.stderr, flush=True)
        try:
            run_neighbors_and_diffmap(adata, n_neighbors=N_NEIGHBORS,
                                       n_comps=N_EIGENVECS)
        except Exception as exc:
            print(f"[ref] ERROR building neighbors/diffmap for {scale_name}: {exc}",
                  file=sys.stderr, flush=True)
            print(f"{scale_name},{n_cells},{n_pcs},{N_NEIGHBORS},{N_EIGENVECS},-1.0",
                  flush=True)
            continue

        print(f"[ref] Timing sc.tl.dpt iroot={ROOT_CELL} for {scale_name} "
              f"({WARMUP_ITERS}w + {TIMED_ITERS}t)...", file=sys.stderr, flush=True)

        try:
            wall_ms = time_dpt(adata)
        except Exception as exc:
            print(f"[ref] ERROR timing dpt for {scale_name}: {exc}",
                  file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_pcs},{N_NEIGHBORS},{N_EIGENVECS},{wall_ms:.1f}",
              flush=True)


if __name__ == "__main__":
    main()
