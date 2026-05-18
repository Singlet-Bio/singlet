#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/harmony_harmonypy_reference.py
#
# Reference driver: run harmonypy.run_harmony on a dense fp32 embedding and
# write the corrected embedding + final objective to .npy / .json files.
#
# Environment: pip install harmonypy numpy
#
# Usage:
#   python harmony_harmonypy_reference.py \
#       --embedding   <embedding.npy>      \   # float32 (N, d)
#       --batch-labels <batch_labels.npy>  \   # int32 (N,) in [0, n_batches)
#       --out-embedding <corrected.npy>    \   # float32 (N, d)
#       --out-meta      <meta.json>        \   # {"n_iters": int, "final_obj": float}
#       [--n-clusters <int>]               \   # default: 20
#       [--theta <float>]                  \   # default: 2.0
#       [--max-iter <int>]                 \   # default: 10
#       [--random-state <int>]             \   # default: 42
#
# The C++ test calls this as a subprocess then reads corrected.npy and meta.json.

import argparse
import json
import sys
import numpy as np


def run_harmony_reference(
    embedding: np.ndarray,
    batch_labels: np.ndarray,
    n_clusters: int = 20,
    theta: float = 2.0,
    max_iter: int = 10,
    random_state: int = 42,
):
    """Run harmonypy and return (corrected_embedding, n_iters, final_obj)."""
    try:
        import harmonypy
    except ImportError:
        print("[ref] ERROR: harmonypy not installed. pip install harmonypy",
              file=sys.stderr)
        raise

    N, d = embedding.shape
    n_batches = int(batch_labels.max()) + 1
    print(f"[ref] embedding: {N} x {d}  n_batches={n_batches}  "
          f"n_clusters={n_clusters}  theta={theta}  max_iter={max_iter}",
          file=sys.stderr)

    # harmonypy expects a pandas DataFrame for meta_data.
    import pandas as pd
    meta = pd.DataFrame({"batch": batch_labels.astype(str)})

    ho = harmonypy.run_harmony(
        embedding.astype(np.float32),
        meta,
        "batch",
        nclust=n_clusters,
        theta=theta,
        max_iter_harmony=max_iter,
        random_state=random_state,
        verbose=False,
    )

    corrected = np.array(ho.Z_corr.T, dtype=np.float32)  # (N, d) — Z_corr is (d, N)
    n_iters = int(ho.objective_harmony.shape[0]) - 1  # rows = iters + 1 initial
    final_obj = float(ho.objective_harmony[-1])

    print(f"[ref] corrected shape: {corrected.shape}  "
          f"n_iters={n_iters}  final_obj={final_obj:.6f}", file=sys.stderr)
    return corrected, n_iters, final_obj


def main():
    parser = argparse.ArgumentParser(
        description="harmonypy reference driver for singlet-gpu correctness test")
    parser.add_argument("--embedding",    required=True,
                        help="float32 embedding .npy, shape (N, d)")
    parser.add_argument("--batch-labels", required=True,
                        help="int32 batch labels .npy, shape (N,), values in [0, B)")
    parser.add_argument("--out-embedding", required=True,
                        help="Path for corrected embedding .npy, shape (N, d)")
    parser.add_argument("--out-meta",      required=True,
                        help="Path for JSON with n_iters, final_obj")
    parser.add_argument("--n-clusters", type=int, default=20)
    parser.add_argument("--theta",      type=float, default=2.0)
    parser.add_argument("--max-iter",   type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    embedding    = np.load(args.embedding).astype(np.float32)
    batch_labels = np.load(args.batch_labels).astype(np.int32)

    corrected, n_iters, final_obj = run_harmony_reference(
        embedding, batch_labels,
        n_clusters=args.n_clusters,
        theta=args.theta,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    np.save(args.out_embedding, corrected)
    with open(args.out_meta, "w") as f:
        json.dump({"n_iters": n_iters, "final_obj": final_obj}, f, indent=2)
    print(f"[ref] Saved corrected embedding to {args.out_embedding}", file=sys.stderr)
    print(f"[ref] Saved meta to {args.out_meta}", file=sys.stderr)


if __name__ == "__main__":
    main()
