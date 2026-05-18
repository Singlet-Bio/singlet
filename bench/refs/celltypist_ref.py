#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/celltypist_ref.py
#
# CYCLE-176 Phase E — sklearn LogisticRegression CPU baseline for anno/celltypist.
#
# GPU kernel (anno::celltypist_predict): cuBLAS Sgemm (L = W^T . Z) +
#   bias-add kernel + strided softmax-argmax per cell.
#   See include/singlet-gpu/anno/celltypist.h.
# Citation: Dominguez Conde C et al. (2022) Science 376:eabl5197.
#
# CPU implementation: sklearn.linear_model.LogisticRegression.predict_proba(X).
#   sklearn IS installed (CYCLE-172 confirmed, v1.6.1).
#   sklearn predict_proba hot path = one BLAS DGEMM (coef_ @ X.T + intercept_)
#   + scipy.special.softmax.  No Python inner loop.
#   SOTA structure: BLAS-backed matmul.
#   §J.7 prediction: class 3 or BLAS-tight (5-30×).  Could be 100-200× if
#   Python/scipy overhead dominates at this small DGEMM size (n=50, m=20).
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#   Only predict_proba(X) is timed — matches GPU bench protocol (data synthesis
#   and weight generation are untimed).
#
# Scales:
#   10k: n_cells=10000, n_features=50 PCs, n_classes=20
#   30k: n_cells=30000, n_features=50 PCs, n_classes=20
#
# Weights: synthesized random (no training).  Bypass sklearn fit() by directly
#   assigning coef_, intercept_, and classes_ — standard pattern for inference
#   benchmarking without a labelled dataset.
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_features,n_classes,wall_ms
#
# Sanity-check columns printed to stderr:
#   n_unique_predicted_classes (expect 20/20 with random weights + 10k cells)
#
# Usage: python3 celltypist_ref.py

import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

WARMUP_ITERS = 2
TIMED_ITERS  = 5

SCALES = [
    ("10k", 10000, 50, 20),
    ("30k", 30000, 50, 20),
]


def make_synth_X(n_cells: int, n_features: int, seed: int = 42) -> np.ndarray:
    """Synthetic expression matrix (n_cells x n_features, float32).

    Gaussian values matching the GPU bench synthetic intent (values in [-1,1]
    range after Gaussian draw — same statistical character).
    """
    rng = np.random.default_rng(seed=seed)
    return rng.standard_normal((n_cells, n_features)).astype(np.float32)


def make_fitted_lr(n_features: int, n_classes: int, seed: int = 42) -> LogisticRegression:
    """Synthesize a fitted LogisticRegression without training.

    Assigns random coef_ (n_classes x n_features) and intercept_ (n_classes)
    directly.  This is the standard inference-only benchmarking pattern.
    sklearn predict_proba() only uses coef_, intercept_, and classes_ —
    no solver state is needed.
    """
    rng = np.random.default_rng(seed=seed)
    lr = LogisticRegression(multi_class="multinomial", max_iter=1, solver="lbfgs")
    lr.classes_    = np.arange(n_classes, dtype=np.intp)
    lr.coef_       = rng.standard_normal((n_classes, n_features))   # float64
    lr.intercept_  = rng.standard_normal(n_classes)                 # float64
    return lr


def time_predict_proba(lr: LogisticRegression, X: np.ndarray) -> float:
    """2 warmup + 5 timed iterations of predict_proba(X).

    Returns median wall time in ms.
    """
    for _ in range(WARMUP_ITERS):
        _ = lr.predict_proba(X)

    wall_times = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        _ = lr.predict_proba(X)
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1000.0)

    wall_times.sort()
    n_t = len(wall_times)
    return (wall_times[n_t // 2] if n_t % 2 == 1
            else 0.5 * (wall_times[n_t // 2 - 1] + wall_times[n_t // 2]))


def main():
    print("scale,n_cells,n_features,n_classes,wall_ms", flush=True)

    for scale_name, n_cells, n_features, n_classes in SCALES:
        print(
            f"[ref] Building synthetic {scale_name}: {n_features} features × "
            f"{n_cells} cells, n_classes={n_classes}...",
            file=sys.stderr, flush=True,
        )

        X  = make_synth_X(n_cells, n_features, seed=42)
        lr = make_fitted_lr(n_features, n_classes, seed=42)

        print(
            f"[ref] {scale_name}: X.shape={X.shape}, dtype={X.dtype}. "
            f"Timing LogisticRegression.predict_proba...",
            file=sys.stderr, flush=True,
        )

        try:
            wall_ms = time_predict_proba(lr, X)
            proba   = lr.predict_proba(X)
            pred    = np.argmax(proba, axis=1)
            n_unique = len(np.unique(pred))
            print(
                f"[ref] {scale_name}: n_unique_predicted_classes={n_unique}/{n_classes}",
                file=sys.stderr, flush=True,
            )
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}", file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_features},{n_classes},{wall_ms:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
