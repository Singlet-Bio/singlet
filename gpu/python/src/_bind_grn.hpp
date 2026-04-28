// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/python/src/_bind_grn.hpp
//
// pybind11 binding for singlet_gpu::grn::run_granie (cycle 36).
//
// Architecture:
//   - PyGranieResult holds the C++ GranieResult's host-side data.
//     Edge list is returned as a dict-per-edge list; tf_activity as a numpy array.
//   - Input marshalling:
//       gex         : PyDeviceCsc (already on device)
//       peaks       : PyDeviceCsc (already on device)
//       tf_motif_h  : Python dict with keys row_ptr, col_indices, values, n_rows, n_cols
//       peak_gene_pairs : list of (peak_idx, gene_idx) tuples
//   - GranieConfig is assembled from Python kwargs.
//   - GPUContext is created internally from the optional stream handle.
//
// Result marshalling:
//   Returns a dict with:
//     "edges"            : list of dicts {tf_idx, gene_idx, combined_score, p_value, fdr}
//     "tf_activity"      : numpy float32 array (n_tfs × n_cells, row-major)
//     "n_edges"          : int
//     "n_peak_gene_links": int
//     "n_tfs"            : int
//     "n_cells"          : int
//     "community_labels" : list of int (empty if run_leiden=False)
//     "n_communities"    : int

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <singlet-gpu/grn/granie.h>
#include <singlet-gpu/core/handles.h>

#include "_bind_loader.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cstdint>

namespace py = pybind11;

namespace singlet_gpu {
namespace python {

// ---------------------------------------------------------------------------
// py_run_granie
//
// Parameters:
//   gex_obj        : DeviceCsc  — GEX matrix (cells × genes, CSC)
//   peaks_obj      : DeviceCsc  — ATAC peaks matrix (cells × peaks, CSC)
//   tf_motif_dict  : dict with keys:
//                      "row_ptr"     — list[int], length n_tfs + 1
//                      "col_indices" — list[int], length nnz
//                      "values"      — list[float], length nnz
//                      "n_rows"      — int (n_tfs)
//                      "n_cols"      — int (n_peaks)
//   peak_gene_pairs_list : list of 2-element sequences [peak_idx, gene_idx]
//   min_peak_mean  : float
//   min_peak_var   : float
//   min_abs_r      : float
//   peak_gene_fdr  : float
//   tf_target_fdr  : float
//   min_combined_score : float
//   run_leiden     : bool
//   leiden_resolution  : float
//   leiden_max_iter    : int
//   chunk_size_tfs : int
//   stream         : optional int (cudaStream_t handle as Python int, or None)
//   seed           : int
// ---------------------------------------------------------------------------
inline py::dict py_run_granie(
    py::object    gex_obj,
    py::object    peaks_obj,
    py::dict      tf_motif_dict,
    py::list      peak_gene_pairs_list,
    float         min_peak_mean       = 0.01f,
    float         min_peak_var        = 0.01f,
    float         min_abs_r           = 0.3f,
    float         peak_gene_fdr       = 0.05f,
    float         tf_target_fdr       = 0.05f,
    float         min_combined_score  = 0.0f,
    bool          run_leiden          = false,
    float         leiden_resolution   = 1.0f,
    int           leiden_max_iter     = 100,
    int           chunk_size_tfs      = 100,
    py::object    stream              = py::none(),
    uint64_t      seed                = 0)
{
    // ── Unwrap DeviceCsc inputs ───────────────────────────────────────────────
    auto& gex_py   = gex_obj.cast<PyDeviceCsc&>();
    auto& peaks_py = peaks_obj.cast<PyDeviceCsc&>();
    const core::DeviceCSC& gex_csc   = *gex_py.csc;
    const core::DeviceCSC& peaks_csc = *peaks_py.csc;

    // ── Resolve CUDA stream ───────────────────────────────────────────────────
    cudaStream_t cu_stream = nullptr;
    if (!stream.is_none()) {
        cu_stream = reinterpret_cast<cudaStream_t>(
            static_cast<uintptr_t>(stream.cast<py::int_>()));
    }

    // ── Build HostCSR from Python dict ────────────────────────────────────────
    grn::HostCSR tf_motif;
    tf_motif.n_rows = tf_motif_dict["n_rows"].cast<int>();
    tf_motif.n_cols = tf_motif_dict["n_cols"].cast<int>();
    tf_motif.row_ptr     = tf_motif_dict["row_ptr"].cast<std::vector<int>>();
    tf_motif.col_indices = tf_motif_dict["col_indices"].cast<std::vector<int>>();
    tf_motif.values      = tf_motif_dict["values"].cast<std::vector<float>>();
    tf_motif.nnz         = static_cast<int>(tf_motif.col_indices.size());

    // ── Build peak_gene_pairs list ────────────────────────────────────────────
    std::vector<int2> peak_gene_pairs;
    peak_gene_pairs.reserve(peak_gene_pairs_list.size());
    for (auto item : peak_gene_pairs_list) {
        auto seq = item.cast<py::sequence>();
        int2 p;
        p.x = seq[0].cast<int>();  // peak_idx
        p.y = seq[1].cast<int>();  // gene_idx
        peak_gene_pairs.push_back(p);
    }

    // ── Build GranieConfig ────────────────────────────────────────────────────
    grn::GranieConfig cfg;
    cfg.min_peak_mean      = min_peak_mean;
    cfg.min_peak_var       = min_peak_var;
    cfg.min_abs_r          = min_abs_r;
    cfg.peak_gene_fdr      = peak_gene_fdr;
    cfg.tf_target_fdr      = tf_target_fdr;
    cfg.min_combined_score = min_combined_score;
    cfg.run_leiden         = run_leiden;
    cfg.leiden_resolution  = leiden_resolution;
    cfg.leiden_max_iter    = leiden_max_iter;
    cfg.chunk_size_tfs     = chunk_size_tfs;
    cfg.seed               = seed;

    // ── Build GPUContext from stream ──────────────────────────────────────────
    core::GPUContext ctx;
    ctx.stream = cu_stream;

    // ── Run kernel ────────────────────────────────────────────────────────────
    grn::GranieResult result = grn::run_granie(
        gex_csc, peaks_csc, tf_motif, peak_gene_pairs, ctx, cfg);

    // ── Marshal results to Python dict ────────────────────────────────────────
    py::list edges_list;
    for (const auto& e : result.edges) {
        py::dict ed;
        ed["tf_idx"]         = e.tf_idx;
        ed["gene_idx"]       = e.gene_idx;
        ed["combined_score"] = e.combined_score;
        ed["p_value"]        = e.p_value;
        ed["fdr"]            = e.fdr;
        edges_list.append(ed);
    }

    // tf_activity_host: n_tfs × n_cells, row-major → numpy array
    py::array_t<float> tf_act_arr;
    if (!result.tf_activity_host.empty()) {
        tf_act_arr = py::array_t<float>(
            {static_cast<py::ssize_t>(result.n_tfs),
             static_cast<py::ssize_t>(result.n_cells)},
            result.tf_activity_host.data());
    } else {
        tf_act_arr = py::array_t<float>({0L, 0L});
    }

    py::list community_list;
    for (int c : result.community_labels) community_list.append(c);

    py::dict out;
    out["edges"]             = edges_list;
    out["tf_activity"]       = tf_act_arr;
    out["n_edges"]           = result.n_edges;
    out["n_peak_gene_links"] = result.n_peak_gene_links;
    out["n_tfs"]             = result.n_tfs;
    out["n_cells"]           = result.n_cells;
    out["community_labels"]  = community_list;
    out["n_communities"]     = result.n_communities;
    return out;
}

// ---------------------------------------------------------------------------
// bind_grn — register all grn bindings on module m
// ---------------------------------------------------------------------------
inline void bind_grn(py::module_& m) {
    m.def("run_granie",
        &py_run_granie,
        py::arg("gex"),
        py::arg("peaks"),
        py::arg("tf_motif"),
        py::arg("peak_gene_pairs"),
        py::arg("min_peak_mean")      = 0.01f,
        py::arg("min_peak_var")       = 0.01f,
        py::arg("min_abs_r")          = 0.3f,
        py::arg("peak_gene_fdr")      = 0.05f,
        py::arg("tf_target_fdr")      = 0.05f,
        py::arg("min_combined_score") = 0.0f,
        py::arg("run_leiden")         = false,
        py::arg("leiden_resolution")  = 1.0f,
        py::arg("leiden_max_iter")    = 100,
        py::arg("chunk_size_tfs")     = 100,
        py::arg("stream")             = py::none(),
        py::arg("seed")               = 0,
        R"doc(
GPU-native GRaNIE gene regulatory network inference (cycle 36).

Runs the full GRaNIE pipeline: peak filtering → peak–gene correlation →
TF activity scoring → TF–target edge scoring → BH FDR → optional Leiden.

Parameters
----------
gex : DeviceCsc
    GEX matrix (cells × genes, CSC, float32).  Rows = cells, cols = genes.
peaks : DeviceCsc
    ATAC peaks matrix (cells × peaks, CSC, float32).  Must have same row count as gex.
tf_motif : dict
    TF motif-in-peak sparse matrix (CSR format).
    Keys: ``n_rows`` (n_tfs), ``n_cols`` (n_peaks), ``row_ptr``, ``col_indices``, ``values``.
peak_gene_pairs : list of [peak_idx, gene_idx]
    Cis-window peak–gene candidate pairs (pre-computed from genomic coordinates).
min_peak_mean : float, default 0.01
min_peak_var : float, default 0.01
min_abs_r : float, default 0.3
    |Pearson r| threshold for peak–gene links.
peak_gene_fdr : float, default 0.05
tf_target_fdr : float, default 0.05
min_combined_score : float, default 0.0
run_leiden : bool, default False
leiden_resolution : float, default 1.0
leiden_max_iter : int, default 100
chunk_size_tfs : int, default 100
    TFs processed per scoring chunk (OOC control).
stream : int or None
    CUDA stream handle (as Python int).  None uses the default stream.
seed : int, default 0

Returns
-------
dict
    "edges"            — list of dicts {tf_idx, gene_idx, combined_score, p_value, fdr}
    "tf_activity"      — numpy float32 array (n_tfs × n_cells, row-major)
    "n_edges"          — int
    "n_peak_gene_links"— int
    "n_tfs", "n_cells" — int
    "community_labels" — list[int] (empty if run_leiden=False)
    "n_communities"    — int
)doc");
}

} // namespace python
} // namespace singlet_gpu
