#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/symphony_ref.py
#
# CYCLE-178 Phase E — manual NumPy Symphony CPU baseline for anno/symphony.
#
# GPU kernel (anno::symphony_predict): 5 sequential device passes — standardize,
#   PCA project (Sgemm), per-cell distance to centroids (sumsq + Sgemm cross +
#   combine), soft assignment (1/d, L1-normalize), label transfer (Sgemm) + argmax.
#   See include/singlet-gpu/anno/symphony.h.
# Citation: Kang JB, Nathan A, Weinand K et al. (2021) Nat Commun 12:5890.
#
# CPU implementation: manual NumPy — no Python Symphony package dependency.
#   All 5 passes vectorized with numpy broadcasting / np.dot:
#     Pass 1 — standardize:   Z_std = (Z - mu) / np.maximum(sigma, 1e-3)
#     Pass 2 — PCA project:   X_pca = Z_std @ W_pca       (n_cells x k_pcs)
#     Pass 3 — distances:     D[c,k] = ||X[c]||² + ||C[k]||² - 2·(X @ C.T)[c,k]
#                              broadcasted in one vectorized expression
#     Pass 4 — soft assign:   s[c,k] = 1/max(D[c,k], eps); row-normalize (L1)
#     Pass 5 — label transfer: proba = s @ P_label; pred = argmax(proba, axis=1)
#
#   SOTA structure: vectorized numpy — tight C inner loops via BLAS (np.dot = Sgemm)
#     for PCA project and label transfer; distance is a single broadcasting
#     expression with no Python inner loop.
#   §J.7 prediction: python_overhead_multiplier = 1× (single tight vectorized call),
#     memory_bandwidth_advantage = 1× (30k × 50 intermediates < 10 MB),
#     GPU compute = medium/heavy (5 passes vs celltypist 3).
#     Predict 30-100× class (similar to celltypist 50× but slightly lower because
#     GPU has 2 more passes and numpy SOTA here is also well-vectorized).
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#   Only the 5-pass inference (standardize → label transfer) is timed.
#   Data synthesis and weight generation are untimed.
#
# Scales:
#   10k: n_cells=10000, n_features=50, k_pcs=50, n_centroids=20, n_classes=20
#   30k: n_cells=30000, n_features=50, k_pcs=50, n_centroids=20, n_classes=20
#
# Sanity-check printed to stderr:
#   mean_confidence, n_unique_pred_classes (expect close to n_classes with random weights)
#
# Usage: python3 symphony_ref.py

import sys
import time

import numpy as np

WARMUP_ITERS = 2
TIMED_ITERS  = 5
EPS_SIGMA    = 1e-3
EPS_DIST     = 1e-6

SCALES = [
    ("10k", 10000, 50, 50, 20, 20),
    ("30k", 30000, 50, 50, 20, 20),
]


def make_synth_data(n_cells: int, n_features: int, k_pcs: int,
                    n_centroids: int, n_classes: int,
                    seed: int = 42):
    """Synthesize all inputs for one scale (untimed).

    Returns:
        Z        — (n_cells, n_features) float32  query expression
        W_pca    — (n_features, k_pcs)  float32  PCA loadings
        C_ref    — (n_centroids, k_pcs) float32  reference centroids
        P_label  — (n_centroids, n_classes) float32  row-normalized label distribution
        mu       — (n_features,) float32  reference mean
        sigma    — (n_features,) float32  reference std (positive)
    """
    rng = np.random.default_rng(seed=seed)
    Z       = rng.standard_normal((n_cells, n_features)).astype(np.float32)
    W_pca   = rng.standard_normal((n_features, k_pcs)).astype(np.float32)
    C_ref   = rng.standard_normal((n_centroids, k_pcs)).astype(np.float32)
    mu      = rng.standard_normal(n_features).astype(np.float32)
    sigma   = (np.abs(rng.standard_normal(n_features)) + 0.01).astype(np.float32)

    # P_label: row-normalized uniform-positive (n_centroids x n_classes).
    raw     = rng.random((n_centroids, n_classes)).astype(np.float32) + 1e-4
    P_label = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)

    return Z, W_pca, C_ref, P_label, mu, sigma


def symphony_predict_numpy(Z, W_pca, C_ref, P_label, mu, sigma):
    """Manual NumPy implementation of Symphony centroid-projection label transfer.

    Mirrors the 5 GPU passes in anno/symphony.h exactly (same math, same order).

    Args:
        Z       — (n_cells, n_features) float32
        W_pca   — (n_features, k_pcs) float32
        C_ref   — (n_centroids, k_pcs) float32
        P_label — (n_centroids, n_classes) float32  rows sum to 1
        mu      — (n_features,) float32
        sigma   — (n_features,) float32

    Returns:
        pred_class  — (n_cells,) int32
        confidence  — (n_cells,) float32  max label probability
    """
    # Pass 1: standardize.
    Z_std = (Z - mu) / np.maximum(sigma, EPS_SIGMA)              # (n_cells, n_features)

    # Pass 2: PCA project.
    X_pca = Z_std @ W_pca                                         # (n_cells, k_pcs)

    # Pass 3: distances.
    X_sq   = (X_pca ** 2).sum(axis=1)                            # (n_cells,)
    C_sq   = (C_ref ** 2).sum(axis=1)                            # (n_centroids,)
    cross  = X_pca @ C_ref.T                                     # (n_cells, n_centroids)
    D      = X_sq[:, None] + C_sq[None, :] - 2.0 * cross        # (n_cells, n_centroids)
    D      = np.maximum(D, 0.0)                                   # clamp to [0, inf)

    # Pass 4: soft assignment + L1 normalize.
    s_inv  = 1.0 / np.maximum(D, EPS_DIST)                       # (n_cells, n_centroids)
    row_sum = s_inv.sum(axis=1, keepdims=True)
    s_soft  = s_inv / np.maximum(row_sum, 1e-12)                  # (n_cells, n_centroids)

    # Pass 5: label transfer + argmax.
    proba      = s_soft @ P_label                                  # (n_cells, n_classes)
    pred_class = np.argmax(proba, axis=1).astype(np.int32)
    confidence = proba[np.arange(len(pred_class)), pred_class].astype(np.float32)

    return pred_class, confidence


def time_symphony(Z, W_pca, C_ref, P_label, mu, sigma) -> float:
    """2 warmup + 5 timed iterations of symphony_predict_numpy.

    Returns median wall time in ms.
    """
    for _ in range(WARMUP_ITERS):
        _ = symphony_predict_numpy(Z, W_pca, C_ref, P_label, mu, sigma)

    wall_times = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        _ = symphony_predict_numpy(Z, W_pca, C_ref, P_label, mu, sigma)
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1000.0)

    wall_times.sort()
    n_t = len(wall_times)
    return (wall_times[n_t // 2] if n_t % 2 == 1
            else 0.5 * (wall_times[n_t // 2 - 1] + wall_times[n_t // 2]))


def main():
    print("scale,n_cells,n_features,k_pcs,n_centroids,n_classes,wall_ms", flush=True)

    for scale_name, n_cells, n_features, k_pcs, n_centroids, n_classes in SCALES:
        print(
            f"[ref] Building synthetic {scale_name}: {n_features} features × "
            f"{n_cells} cells, k_pcs={k_pcs}, n_centroids={n_centroids}, "
            f"n_classes={n_classes}...",
            file=sys.stderr, flush=True,
        )

        Z, W_pca, C_ref, P_label, mu, sigma = make_synth_data(
            n_cells, n_features, k_pcs, n_centroids, n_classes, seed=42)

        print(
            f"[ref] {scale_name}: Z.shape={Z.shape}, dtype={Z.dtype}. "
            f"Timing NumPy Symphony (5 passes)...",
            file=sys.stderr, flush=True,
        )

        try:
            wall_ms = time_symphony(Z, W_pca, C_ref, P_label, mu, sigma)
            pred, conf = symphony_predict_numpy(Z, W_pca, C_ref, P_label, mu, sigma)
            n_unique = len(np.unique(pred))
            mean_conf = float(conf.mean())
            print(
                f"[ref] {scale_name}: n_unique_pred_classes={n_unique}/{n_classes}, "
                f"mean_confidence={mean_conf:.4f}",
                file=sys.stderr, flush=True,
            )
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}", file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_features},{k_pcs},{n_centroids},{n_classes},{wall_ms:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
