#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/cycle61_nmf_baselines.py
#
# Cycle 61 Phase E — NMF adapter baselines.
#
# Modes:
#   sklearn_nmf       : sklearn.decomposition.NMF (CPU, multiplicative update / HALS)
#   cnmf              : cNMF (Kotliar 2019) consensus NMF — pip install cnmf
#   factornet_cpu     : factornet CPU NMF (correctness reference)
#                       Implemented via sklearn NMF as the closest public equivalent
#                       (factornet CPU NMF uses HALS — sklearn MU is the proxy).
#
# Usage:
#   python3 cycle61_nmf_baselines.py --mode sklearn_nmf --k 20 \
#       --h5ad /dev/shm/singlet_gpu_bench_small.h5ad \
#       --out /dev/shm/result.json
#
#   python3 cycle61_nmf_baselines.py --dry-run   (import check, no compute)
#
# Output JSON keys: wall_ms, mem_mb, loss, status, n_cells, n_features, k.
#
# Notes:
#   - rapids-singlecell NOT attempted (ENV-RAPIDS-G001 DAG open).
#   - cuml NOT attempted (same ENV gate).
#   - rcppml_r NOT attempted (ENV-SCRAN-G001).
#   - Memory for CPU baselines = RSS delta via tracemalloc.
#   - NMF operates on raw non-negative counts (NOT log-normalized).
#     The h5ad is loaded, top-HVG subset selected (top 2000 by std),
#     then passed to the NMF solver.
#   - Loss = reconstruction MSE: ||A - WH||_F^2 / (m * n).

import argparse
import json
import os
import sys
import time
import tracemalloc


HVG_N = 2000  # Top HVG subset to match C++ bench driver


# ---------------------------------------------------------------------------
# Argument parsing.
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Cycle 61 NMF Phase E baselines")
    p.add_argument("--mode", choices=["sklearn_nmf", "cnmf", "factornet_cpu"],
                   help="Which baseline to run")
    p.add_argument("--k", type=int, default=20, help="NMF rank")
    p.add_argument("--h5ad", type=str, default="",
                   help="Path to h5ad file (anndata with raw counts)")
    p.add_argument("--out", type=str, default="/dev/shm/cycle61_nmf_baseline.json",
                   help="Output JSON path")
    p.add_argument("--dry-run", action="store_true",
                   help="Import check only — no compute. Exits 0 on success.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dry-run: verify all required imports are available.
# ---------------------------------------------------------------------------
def dry_run_check():
    missing = []

    try:
        import numpy as np
        _ = np.__version__
    except ImportError as e:
        missing.append(f"numpy: {e}")

    try:
        import scipy
        _ = scipy.__version__
    except ImportError as e:
        missing.append(f"scipy: {e}")

    try:
        import sklearn
        _ = sklearn.__version__
    except ImportError as e:
        missing.append(f"sklearn: {e}")

    try:
        import anndata
        _ = anndata.__version__
    except ImportError as e:
        missing.append(f"anndata: {e}")

    # Optional imports — report but do not fail.
    optional_ok = []
    optional_miss = []

    try:
        import scanpy
        optional_ok.append(f"scanpy {scanpy.__version__}")
    except ImportError:
        optional_miss.append("scanpy (optional)")

    try:
        import cnmf as cnmf_pkg
        optional_ok.append("cnmf")
    except ImportError:
        optional_miss.append("cnmf (optional — CONFIG_UNAVAILABLE if missing)")

    # Print summary.
    if missing:
        print(f"DRY_RUN FAIL: missing required packages: {missing}")
        return False

    print(f"DRY_RUN PASS: required packages OK (numpy, scipy, sklearn, anndata)")
    if optional_ok:
        print(f"  Optional available: {optional_ok}")
    if optional_miss:
        print(f"  Optional missing: {optional_miss}")
    return True


# ---------------------------------------------------------------------------
# Load h5ad and select top-HVG subset.
# ---------------------------------------------------------------------------
def load_hvg_matrix(h5ad_path, hvg_n=HVG_N):
    """Load h5ad, extract raw counts as dense numpy array (top-HVG genes x cells)."""
    import anndata
    import numpy as np
    import scipy.sparse as sp

    adata = anndata.read_h5ad(h5ad_path)
    n_cells = adata.shape[0]
    n_genes = adata.shape[1]

    # Use raw counts if available; else use .X.
    if adata.raw is not None:
        X = adata.raw.X
    else:
        X = adata.X

    if sp.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    # X is cells x genes.

    # Select top-HVG_N genes by standard deviation.
    gene_std = X.std(axis=0)
    top_idx = np.argsort(gene_std)[::-1][:hvg_n]
    X_hvg = X[:, top_idx]  # cells x HVG_N

    print(f"  Loaded h5ad: {n_cells} cells x {n_genes} genes", file=sys.stderr)
    print(f"  HVG subset: {X_hvg.shape}", file=sys.stderr)
    return X_hvg.astype(np.float32), n_cells


# ---------------------------------------------------------------------------
# Reconstruction MSE loss helper.
# ---------------------------------------------------------------------------
def reconstruction_loss(X, W, H):
    """||X - W@H||_F^2 / (m*n)  — NMF convention: X approx W @ H."""
    import numpy as np
    R = X - W @ H
    return float(np.mean(R ** 2))


# ---------------------------------------------------------------------------
# sklearn NMF baseline.
# ---------------------------------------------------------------------------
def run_sklearn_nmf(h5ad_path, k, out_path):
    import numpy as np
    from sklearn.decomposition import NMF

    tracemalloc.start()
    X, n_cells = load_hvg_matrix(h5ad_path)

    t0 = time.perf_counter()
    try:
        model = NMF(n_components=k, init='nndsvd', max_iter=200,
                    tol=1e-4, random_state=42, verbose=0)
        W = model.fit_transform(X)
        H = model.components_
    except Exception as e:
        write_result(out_path, -1.0, -1.0, -1.0, f"FAILED({e})", n_cells, X.shape[1], k)
        return

    wall_ms = (time.perf_counter() - t0) * 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_mb = peak / (1024 ** 2)

    loss = reconstruction_loss(X, W, H)
    write_result(out_path, wall_ms, mem_mb, loss, "OK", n_cells, X.shape[1], k)


# ---------------------------------------------------------------------------
# cNMF baseline.
# ---------------------------------------------------------------------------
def run_cnmf(h5ad_path, k, out_path):
    """Consensus NMF via cNMF package (Kotliar 2019)."""
    import numpy as np
    import tempfile

    # Try importing cnmf.
    try:
        import cnmf as cnmf_pkg
    except ImportError:
        print("  cnmf not installed — CONFIG_UNAVAILABLE", file=sys.stderr)
        write_result(out_path, -1.0, -1.0, -1.0, "CONFIG_UNAVAILABLE",
                     0, 0, k)
        return

    X, n_cells = load_hvg_matrix(h5ad_path)

    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        # cNMF requires an AnnData and a directory.
        import anndata
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build minimal AnnData from HVG matrix.
            adata_tmp = anndata.AnnData(X=X)
            h5ad_tmp = os.path.join(tmpdir, "input.h5ad")
            adata_tmp.write_h5ad(h5ad_tmp)

            # Run cNMF: prepare + factorize + combine.
            cn = cnmf_pkg.cNMF(output_dir=tmpdir, name="cycle61")
            cn.prepare(counts_fn=h5ad_tmp,
                       components=[k],
                       n_iter=10,
                       seed=42,
                       num_highvar_genes=HVG_N)
            cn.factorize(worker_i=0, total_workers=1)
            cn.combine()
            # Get the consensus W and H.
            usage, _, spectra = cn.load_results(K=k, density_threshold=0.01)
            W_cnmf = usage.values.astype(np.float32)
            H_cnmf = spectra.values.astype(np.float32).T
    except Exception as e:
        wall_ms = (time.perf_counter() - t0) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  cNMF FAILED: {e}", file=sys.stderr)
        write_result(out_path, -1.0, peak / (1024**2), -1.0,
                     f"FAILED({type(e).__name__})", n_cells, X.shape[1], k)
        return

    wall_ms = (time.perf_counter() - t0) * 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_mb = peak / (1024 ** 2)

    # Reconstruction loss: X ~ W_cnmf @ H_cnmf.T (cNMF convention).
    try:
        loss = reconstruction_loss(X, W_cnmf, H_cnmf)
    except Exception:
        loss = -1.0

    write_result(out_path, wall_ms, mem_mb, loss, "OK", n_cells, X.shape[1], k)


# ---------------------------------------------------------------------------
# factornet_cpu (sklearn proxy for CPU HALS reference).
# ---------------------------------------------------------------------------
def run_factornet_cpu(h5ad_path, k, out_path):
    """
    Proxy for factornet CPU NMF using sklearn NMF with cd solver (HALS-like).
    sklearn NMF solver='cd' uses coordinate descent, which approximates HALS.
    This is the CPU correctness reference: final loss should be within 5% of
    factornet GPU HALS on the same data.
    """
    import numpy as np
    from sklearn.decomposition import NMF

    tracemalloc.start()
    X, n_cells = load_hvg_matrix(h5ad_path)

    t0 = time.perf_counter()
    try:
        # solver='cd' is closest to HALS (coordinate descent on H, then W).
        model = NMF(n_components=k, init='nndsvd', solver='cd',
                    max_iter=500, tol=1e-5, random_state=42, verbose=0)
        W = model.fit_transform(X)
        H = model.components_
        n_iter = model.n_iter_
    except Exception as e:
        write_result(out_path, -1.0, -1.0, -1.0, f"FAILED({e})", 0, 0, k)
        return

    wall_ms = (time.perf_counter() - t0) * 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_mb = peak / (1024 ** 2)

    loss = reconstruction_loss(X, W, H)
    print(f"  factornet_cpu (sklearn cd): k={k} n_iter={n_iter} loss={loss:.6f}",
          file=sys.stderr)
    write_result(out_path, wall_ms, mem_mb, loss, "OK", n_cells, X.shape[1], k)


# ---------------------------------------------------------------------------
# Write JSON result.
# ---------------------------------------------------------------------------
def write_result(out_path, wall_ms, mem_mb, loss, status, n_cells, n_features, k):
    result = {
        "wall_ms":    wall_ms,
        "mem_mb":     mem_mb,
        "loss":       loss,
        "status":     status,
        "n_cells":    n_cells,
        "n_features": n_features,
        "k":          k,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Result: {result}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.dry_run:
        ok = dry_run_check()
        sys.exit(0 if ok else 1)

    if not args.mode:
        print("ERROR: --mode is required unless --dry-run", file=sys.stderr)
        sys.exit(1)

    if not args.h5ad or not os.path.exists(args.h5ad):
        print(f"ERROR: h5ad not found: {args.h5ad}", file=sys.stderr)
        write_result(args.out, -1.0, -1.0, -1.0, "NO_H5AD", 0, 0, args.k)
        sys.exit(1)

    print(f"[cycle61_nmf_baselines] mode={args.mode} k={args.k} h5ad={args.h5ad}",
          file=sys.stderr)

    if args.mode == "sklearn_nmf":
        run_sklearn_nmf(args.h5ad, args.k, args.out)
    elif args.mode == "cnmf":
        run_cnmf(args.h5ad, args.k, args.out)
    elif args.mode == "factornet_cpu":
        run_factornet_cpu(args.h5ad, args.k, args.out)
    else:
        print(f"ERROR: unknown mode {args.mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
