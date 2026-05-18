// SPDX-License-Identifier: MIT
// singlet/gpu/reduce/nmf/graph.h
//
// FactorGraph multi-modal NMF — DEFERRED INDEFINITELY as of CYCLE-105.
//
// The _bind_kernels.hpp nmf_graph_factorize binding is gated behind
// SINGLET_GPU_BUILD_NMF_GRAPH (not set by default). This stub header exists
// so existing #includes compile without error.
//
// DO NOT implement this file until the graph NMF design is revisited.
// See state/roadmap.md for the deferred status.

#pragma once

// Intentionally empty. All symbols in this header are deferred.
// If code reaches here expecting FactorGraph symbols, it must define
// SINGLET_GPU_BUILD_NMF_GRAPH and provide its own implementation.
