// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (Harmony adapted) + cycle 8 KNN (BBKNN)
//
// integrate/types.h — Shared type definitions for the integration module.
//
// Used by harmony.h and bbknn.h; include this first.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/graph/knn.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace singlet_gpu {
namespace integrate {

// ─── Harmony types ────────────────────────────────────────────────────────────

struct HarmonyConfig {
    int      n_clusters      = 20;    // soft k-means K
    int      max_iter        = 10;    // max correction iterations
    float    tol             = 1e-4f; // convergence: max |R_new - R_old| < tol
    float    lambda          = 1.0f;  // ridge penalty on centroid calculation
    int      kmeans_init_iter = 10;   // k-means++ init iterations
    uint64_t seed            = 0;     // fixed seed → deterministic init
    bool     deterministic   = false; // true → segmented scan for centroid accum
};

// Returned by harmony(). corrected is n*d row-major (same shape as input Z).
struct HarmonyResult {
    core::DeviceMemory<float> corrected;   // n × d in-place corrected embedding
    int   n_iters_used;
    float final_delta;                     // max |R_new - R_old| at last iter
};

// ─── BBKNN types ──────────────────────────────────────────────────────────────

struct BbknnConfig {
    int k_within          = 3;       // neighbors per batch; total k = k_within * n_batches
    int approx_threshold  = 100000;  // n above which to use HNSW backend
    graph::DistanceMetric metric = graph::DistanceMetric::L2;
};

}  // namespace integrate
}  // namespace singlet_gpu
