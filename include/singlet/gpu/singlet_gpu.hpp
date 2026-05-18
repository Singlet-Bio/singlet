// SPDX-License-Identifier: MIT
// singlet/gpu/singlet::gpu.hpp — public API umbrella.
//
// This header is the FROZEN public API surface. Everything you can include
// from `<singlet/gpu/singlet::gpu.hpp>` is API-stable across MINOR versions
// per state/release-policy.md. Internal headers under per-module subdirs
// (preprocess/, de/, graph/, ...) are NOT frozen — including them is at
// your own risk for API churn.
//
// Adding a public symbol: append to state/public-api.md, add the export
// here, bump MINOR at next release tag.
// Removing a public symbol: mark `[[deprecated(...)]]` for one MINOR,
// then remove and bump MAJOR.
//
// See state/public-api.md for the full contract.

#pragma once

#include <singlet/gpu/version.h>

// ── Core types ─────────────────────────────────────────────────────────────
//
// `core::DeviceCSC`     — device CSC sparse matrix (genes × cells, fp32).
// `core::DeviceDense`   — device dense matrix (fp32).
// `core::DeviceMemory`  — RAII device buffer template.
// `core::GPUContext`    — owns cuBLAS / cuSPARSE / cuSOLVER handles + streams.
// `core::PinnedBuffer`  — RAII pinned host buffer.
// `core::Metadata`      — typed GEO sidecar: gsm_id, gse_id, organism, protocol,
//                         modality, srr_ids, read_count, geo_title, rownames, colnames, ...
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/core/handles.h>

// ── I/O ────────────────────────────────────────────────────────────────────
//
// `io::load_pz(path, stream, keep_host_pinned)` — zero-copy .1pz → DeviceCSC.
// `io::PzDeviceMatrix` — the return type (DeviceCSC + Metadata + stream + pinned staging).
// `io::PzChunkIterator` — streaming iterator for matrices that exceed device memory.
//
// Reference: docs/api/io_load_pz.md
#include <singlet/gpu/io/pz_device_loader.h>

// ── preprocess ─────────────────────────────────────────────────────────────
// log_normalize, compute_deconv_size_factors — docs/api/preprocess_log_normalize.md
// select_hvg, deviance_feature_selection — docs/api/preprocess_select_hvg.md
// scale, regress_out — docs/api/preprocess_scale.md
#include <singlet/gpu/preprocess/lognorm.h>
#include <singlet/gpu/preprocess/deconv_size_factors.h>
#include <singlet/gpu/preprocess/hvg.h>
#include <singlet/gpu/preprocess/scale.h>

// ── reduce (SVD + NMF — native GPU kernels) ────────────────────────────────
// reduce::svd::auto_select / deflation / randomized — docs/api/reduce_svd.md
// reduce::nmf::fit — docs/api/reduce_nmf.md
#include <singlet/gpu/reduce/svd/auto_select.h>
#include <singlet/gpu/reduce/nmf/fit.h>

// ── qc ─────────────────────────────────────────────────────────────────────
// calculate_qc_metrics, filter_cells, filter_genes — docs/api/qc_metrics.md
// doublet_score — docs/api/qc_metrics.md (Scrublet-equivalent)
#include <singlet/gpu/qc/metrics.h>
#include <singlet/gpu/qc/doublet_score.h>

// ── graph ──────────────────────────────────────────────────────────────────
// compute_knn (Exact + CAGRA) + build_snn — docs/api/graph_knn.md
#include <singlet/gpu/graph/knn.h>
#include <singlet/gpu/graph/snn.h>

// ── de (differential expression) ───────────────────────────────────────────
// wilcoxon_de + ttest_de — docs/api/de_wilcoxon_ttest.md
#include <singlet/gpu/de/wilcoxon.h>
#include <singlet/gpu/de/ttest.h>

// ── Public namespace re-exports for ergonomics ────────────────────────────
//
// Lets users write `singlet::gpu::DeviceCSC` instead of
// `singlet::gpu::core::DeviceCSC` for the most common types. The submodule
// namespaces remain available for full disambiguation.

namespace singlet::gpu {

// Core types
using core::DeviceCSC;
using core::DeviceDense;
using core::DeviceMemory;
using core::GPUContext;
using core::PinnedBuffer;
using core::Metadata;

// I/O
using io::PzDeviceMatrix;
using io::load_pz;

// Preprocess
using preprocess::log_normalize;
using preprocess::compute_deconv_size_factors;
using preprocess::select_hvg;
using preprocess::deviance_feature_selection;
using preprocess::scale;
using preprocess::regress_out;

// Reduce
using reduce::svd::auto_select;
// reduce::nmf::fit lives under a deeper namespace; users qualify as
// singlet::gpu::reduce::nmf::fit(...) — top-level alias would collide with
// other potential `fit` symbols. Same convention applied to svd::deflation /
// svd::randomized: call them via the full path when you need a specific backend.

// QC
using qc::calculate_qc_metrics;
using qc::filter_cells;
using qc::filter_genes;
using qc::doublet_score;

// Graph
using graph::compute_knn;
using graph::compute_snn;

// Differential expression
using de::wilcoxon_de;
using de::ttest_de;

}  // namespace singlet::gpu

// ── Pending public exports (still ahead of `released` state) ──────────────
//
// As features transition to `released` (per state/release-policy.md), append
// the umbrella export here AND a row to state/public-api.md. Remaining:
//
//   embed::umap (feature 10 — todo, blocked on cuVS),
//   graph::leiden (feature 9 — todo, blocked on cuGraph),
//   integrate::harmony / integrate::bbknn (feature 14 — todo),
//   gsea::fgsea / gsea::aucell (feature 12 — todo),
//   anno::* (feature 13 — todo),
//   de::donor_pseudobulk (feature 11 sub-variant — frontier-pending),
//   models::scvi / models::scanvi / models::totalvi (feature 15 — P2 todo),
//   fate::velocity / fate::pseudotime (feature 16 — P2 todo),
//   streaming::PzShardIterator / streaming::run_pipeline (feature 17 — partial).
