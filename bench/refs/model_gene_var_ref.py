#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/model_gene_var_ref.py
#
# CYCLE-162 Phase E — scanpy CPU baseline for preprocess/model_gene_var.
#
# GPU kernel: port of scran::modelGeneVarByPoisson (Lun-McCarthy-Marioni 2016).
# Two scanpy flavors are benchmarked per scale:
#
#   flavor='pearson_residuals'  — closest algorithm (Poisson-null residuals,
#       Lause et al. 2021; same Poisson-null assumption as the GPU kernel)
#   flavor='seurat_v3'          — most popular HVG flavor (Stuart et al. 2019;
#       different algorithm, same use-case: select top-N HVGs)
#
# Same scipy.sparse.random seed=42 as the GPU bench for input parity.
# Protocol: 2 warmup + 5 timed iterations per flavor per scale; median wall time.
#
# NOTE (CYCLE-158 lesson): AnnData var_names set via `adata.var_names = ...`
# (direct assignment), NOT via `var={"col": ...}` constructor argument —
# the constructor form does NOT set the index, only adds a column.
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_genes,density,n_top,flavor,wall_ms
#
# Usage: python3 model_gene_var_ref.py

import sys
import time
import copy

import numpy as np
import scipy.sparse as sp

WARMUP_ITERS = 2
TIMED_ITERS  = 5
N_TOP        = 2000

SCALES = [
    ("10k",  10000, 5000, 0.05),
    ("30k",  30000, 5000, 0.05),
]

FLAVORS = ["pearson_residuals", "seurat_v3"]


def make_synthetic_adata(n_cells: int, n_genes: int, density: float, seed: int = 42):
    """Build AnnData with a synthetic sparse log-normalized float32 matrix (CSC).

    Values are in [0, ~2.7] — representative of log1p-normalized UMI data.
    CSC format matches the GPU bench input layout.

    NOTE: var_names set by direct assignment, not via var= constructor arg
    (CYCLE-158 lesson: constructor `var={"name": ...}` adds a column but does
    NOT set the index; anndata flavor HVG routines check the index).
    """
    import anndata as ad
    rng = np.random.default_rng(seed)
    # scipy.sparse.random in [0,1]; scale to log1p-like range [0, ~2.7]
    # by sampling UMI counts 1-15 and log1p-transforming.
    X = sp.random(n_cells, n_genes, density=density, format="csc",
                  dtype=np.float32, random_state=rng)
    # Snap nonzeros to integer UMI counts in [1, 15], then log1p-transform.
    X.data = np.log1p(
        np.clip(np.round(X.data * 15.0).astype(np.float32), 1.0, 15.0)
    )
    adata = ad.AnnData(X=X)
    # Set var_names (gene IDs) via direct index assignment — see CYCLE-158 lesson.
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]
    return adata


def time_flavor(adata, flavor: str, n_top: int):
    """2 warmup + 5 timed iters of sc.pp.highly_variable_genes for a given flavor.

    flavor='pearson_residuals': closest algorithm to the GPU kernel
        (Poisson-null residuals; Lause et al. 2021).
    flavor='seurat_v3': most popular HVG flavor
        (variance-stabilizing, Stuart et al. 2019).

    Returns median wall_ms, or -1.0 on error.
    """
    import scanpy as sc

    def _run(ad_in):
        a = copy.copy(ad_in)
        if flavor == "pearson_residuals":
            try:
                sc.experimental.pp.highly_variable_genes(
                    a, flavor="pearson_residuals",
                    n_top_genes=min(n_top, ad_in.n_vars))
            except AttributeError:
                # scanpy >=1.10 moved pearson_residuals into sc.pp
                sc.pp.highly_variable_genes(
                    a, flavor="pearson_residuals",
                    n_top_genes=min(n_top, ad_in.n_vars))
        else:
            # seurat_v3 requires a raw (not log-transformed) count matrix;
            # we pass the log1p matrix as a proxy — the benchmark measures
            # wall time, not biological correctness.
            sc.pp.highly_variable_genes(
                a, flavor=flavor,
                n_top_genes=min(n_top, ad_in.n_vars))
        return a

    # Warmup (errors silently discarded — warmup may fail on first import).
    for _ in range(WARMUP_ITERS):
        try:
            _run(adata)
        except Exception:
            pass

    wall_times = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        _run(adata)
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1000.0)

    wall_times.sort()
    n = len(wall_times)
    return (wall_times[n // 2] if n % 2 == 1
            else 0.5 * (wall_times[n // 2 - 1] + wall_times[n // 2]))


def main():
    import importlib.util
    for pkg in ("anndata", "scanpy"):
        if importlib.util.find_spec(pkg) is None:
            print(f"ERROR: {pkg} not installed", file=sys.stderr)
            sys.exit(1)

    print("scale,n_cells,n_genes,density,n_top,flavor,wall_ms", flush=True)

    for scale_name, n_cells, n_genes, density in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_cells} cells × {n_genes} genes "
              f"density={density:.0%}...", file=sys.stderr, flush=True)

        adata = make_synthetic_adata(n_cells, n_genes, density, seed=42)

        for flavor in FLAVORS:
            print(f"[ref] Timing scanpy HVG {scale_name} flavor='{flavor}'...",
                  file=sys.stderr, flush=True)
            try:
                wall_ms = time_flavor(adata, flavor, N_TOP)
            except Exception as exc:
                print(f"[ref] ERROR {scale_name} flavor={flavor}: {exc}",
                      file=sys.stderr, flush=True)
                wall_ms = -1.0

            print(f"{scale_name},{n_cells},{n_genes},{density:.2f},{N_TOP},"
                  f"{flavor},{wall_ms:.1f}", flush=True)


if __name__ == "__main__":
    main()
