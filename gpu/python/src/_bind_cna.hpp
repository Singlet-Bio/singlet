// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/python/src/_bind_cna.hpp
//
// pybind11 binding for singlet_gpu::cna::detect_cna (cycle 35, Numbat).
//
// Architecture:
//   CnaResult holds DeviceMemory<uint8_t> cna_states, DeviceMemory<float> hmm_posteriors,
//   DeviceMemory<int> clone_labels — all device-resident.
//   Segment boundaries are host-side std::vector<SegmentBoundary>.
//
// Input marshalling:
//   expr       : PyDeviceCsc (genes × cells, CSC)
//   chr        : list[int] per-gene chromosome (1-based, 1-22, 23=X, 24=Y, 25=MT)
//   start_bp   : list[int] per-gene genomic start position
//   ref        : optional numpy float32 [n_genes] reference expression, or None
//                (None → kernel computes mean across all cells as reference)
//
// Result marshalling:
//   Returns a dict with:
//     "cna_states"         : __cuda_array_interface__ dict (uint8, n_cells × n_segs)
//     "hmm_posteriors"     : __cuda_array_interface__ dict (float32, n_cells × n_segs × 3)
//     "clone_labels"       : __cuda_array_interface__ dict (int32, n_cells)
//     "n_clones"           : int
//     "leiden_modularity"  : float
//     "n_total_segments"   : int
//     "segments"           : list of dicts {chr, gene_start, gene_end, seg_idx}
//     "_result_owner"      : Python object keeping device memory alive

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <singlet-gpu/cna/numbat.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/core/memory.h>

#include "_bind_loader.hpp"
#include "_cupy_interop.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cstdint>

namespace py = pybind11;

namespace singlet_gpu {
namespace python {

// ---------------------------------------------------------------------------
// PyCnaResult — holds CnaResult device buffers with Python lifetime management.
// ---------------------------------------------------------------------------
struct PyCnaResult {
    core::DeviceMemory<uint8_t> cna_states;
    core::DeviceMemory<float>   hmm_posteriors;
    core::DeviceMemory<int>     clone_labels;
    int   n_total_segments  = 0;
    int   n_cells           = 0;
    int   n_clones          = 0;
    float leiden_modularity = 0.f;
    std::vector<cna::SegmentBoundary> segments;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// py_detect_cna
// ---------------------------------------------------------------------------
inline py::dict py_detect_cna(
    py::object    expr_obj,
    py::list      chr_list,
    py::list      start_bp_list,
    py::object    ref_obj,              // numpy float32 [n_genes] or None
    int           smooth_half_win   = 5,
    int           min_genes_per_chr = 10,
    float         loss_mean         = -1.0f,
    float         loss_sigma        =  1.0f,
    float         neutral_mean      =  0.0f,
    float         neutral_sigma     =  0.5f,
    float         gain_mean         =  0.5f,
    float         gain_sigma        =  1.0f,
    float         trans_prob        =  0.01f,
    int           segments_per_chr  =  50,
    bool          include_sex_chrs  =  false,
    int           knn_k             =  15,
    py::object    stream            =  py::none(),
    uint64_t      seed              =  0)
{
    // ── Unwrap DeviceCsc ──────────────────────────────────────────────────────
    const core::DeviceCSC& expr_csc = *expr_obj.cast<PyDeviceCsc&>().csc;

    // ── Build GeneAnnotation ──────────────────────────────────────────────────
    cna::GeneAnnotation annot;
    annot.chr     = chr_list.cast<std::vector<int>>();
    auto sbp_list = start_bp_list.cast<std::vector<uint32_t>>();
    annot.start_bp = sbp_list;

    // ── Build reference expression (optional) ─────────────────────────────────
    std::vector<float> ref_vec;
    const float* ref_ptr = nullptr;
    if (!ref_obj.is_none()) {
        auto arr = ref_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        auto buf = arr.request();
        const float* raw = static_cast<const float*>(buf.ptr);
        ref_vec.assign(raw, raw + buf.size);
        ref_ptr = ref_vec.data();
    }

    // ── Resolve CUDA stream ───────────────────────────────────────────────────
    cudaStream_t cu_stream = nullptr;
    if (!stream.is_none())
        cu_stream = reinterpret_cast<cudaStream_t>(
            static_cast<uintptr_t>(stream.cast<py::int_>()));

    // ── Build config ──────────────────────────────────────────────────────────
    cna::CnaConfig cfg;
    cfg.smooth_half_win   = smooth_half_win;
    cfg.min_genes_per_chr = min_genes_per_chr;
    cfg.loss_mean         = loss_mean;
    cfg.loss_sigma        = loss_sigma;
    cfg.neutral_mean      = neutral_mean;
    cfg.neutral_sigma     = neutral_sigma;
    cfg.gain_mean         = gain_mean;
    cfg.gain_sigma        = gain_sigma;
    cfg.trans_prob        = trans_prob;
    cfg.segments_per_chr  = segments_per_chr;
    cfg.include_sex_chrs  = include_sex_chrs;
    cfg.knn_k             = knn_k;
    cfg.seed              = seed;

    // ── Run kernel ────────────────────────────────────────────────────────────
    cna::CnaResult cpp_result = cna::detect_cna(expr_csc, annot, ref_ptr, cfg, cu_stream);

    // ── Package into shared_ptr for Python lifetime management ────────────────
    auto py_result = std::make_shared<PyCnaResult>();
    py_result->cna_states       = std::move(cpp_result.cna_states);
    py_result->hmm_posteriors   = std::move(cpp_result.hmm_posteriors);
    py_result->clone_labels     = std::move(cpp_result.clone_labels);
    py_result->n_total_segments = cpp_result.n_total_segments;
    py_result->n_cells          = expr_csc.cols;
    py_result->n_clones         = cpp_result.n_clones;
    py_result->leiden_modularity = cpp_result.leiden_modularity;
    py_result->segments         = std::move(cpp_result.segments);
    py_result->stream           = cu_stream;

    // ── Build Python owner object ─────────────────────────────────────────────
    // Wrap shared_ptr as a py::object so device memory stays alive.
    py::object owner = py::cast(py_result, py::return_value_policy::take_ownership);

    const int n_cells = py_result->n_cells;
    const int n_segs  = py_result->n_total_segments;

    // ── cna_states view: [n_cells × n_segs] uint8 ────────────────────────────
    py::dict cna_view;
    cna_view["typestr"]  = "|u1";
    cna_view["version"]  = 3;
    cna_view["shape"]    = py::make_tuple(n_cells, n_segs);
    cna_view["data"]     = py::make_tuple(
        reinterpret_cast<uintptr_t>(py_result->cna_states.get()), true);
    cna_view["stream"]   = reinterpret_cast<uintptr_t>(cu_stream);
    cna_view["_owner"]   = owner;

    // ── hmm_posteriors view: [n_cells × n_segs × 3] float32 ──────────────────
    py::dict post_view;
    post_view["typestr"] = "<f4";
    post_view["version"] = 3;
    post_view["shape"]   = py::make_tuple(n_cells, n_segs, cna::HMM_NSTATES);
    post_view["data"]    = py::make_tuple(
        reinterpret_cast<uintptr_t>(py_result->hmm_posteriors.get()), true);
    post_view["stream"]  = reinterpret_cast<uintptr_t>(cu_stream);
    post_view["_owner"]  = owner;

    // ── clone_labels view: [n_cells] int32 ────────────────────────────────────
    py::dict clone_view;
    clone_view["typestr"] = "<i4";
    clone_view["version"] = 3;
    clone_view["shape"]   = py::make_tuple(n_cells);
    clone_view["data"]    = py::make_tuple(
        reinterpret_cast<uintptr_t>(py_result->clone_labels.get()), true);
    clone_view["stream"]  = reinterpret_cast<uintptr_t>(cu_stream);
    clone_view["_owner"]  = owner;

    // ── Segment boundary list ─────────────────────────────────────────────────
    py::list seg_list;
    for (const auto& s : py_result->segments) {
        py::dict sd;
        sd["chr"]        = s.chr;
        sd["gene_start"] = s.gene_start;
        sd["gene_end"]   = s.gene_end;
        sd["seg_idx"]    = s.seg_idx;
        seg_list.append(sd);
    }

    // ── Assemble output dict ──────────────────────────────────────────────────
    py::dict out;
    out["cna_states"]        = cna_view;
    out["hmm_posteriors"]    = post_view;
    out["clone_labels"]      = clone_view;
    out["n_clones"]          = py_result->n_clones;
    out["leiden_modularity"] = py_result->leiden_modularity;
    out["n_total_segments"]  = py_result->n_total_segments;
    out["segments"]          = seg_list;
    out["_result_owner"]     = owner;
    return out;
}

// ---------------------------------------------------------------------------
// bind_cna
// ---------------------------------------------------------------------------
inline void bind_cna(py::module_& m) {
    // Register PyCnaResult so the shared_ptr can be held via py::cast.
    py::class_<PyCnaResult, std::shared_ptr<PyCnaResult>>(m, "_CnaResultOwner",
        "Internal: keeps CnaResult device buffers alive for cupy views.")
    .def("__repr__", [](const PyCnaResult& r) {
        return "<_CnaResultOwner n_cells=" + std::to_string(r.n_cells)
             + " n_segs=" + std::to_string(r.n_total_segments)
             + " n_clones=" + std::to_string(r.n_clones) + ">";
    });

    m.def("detect_cna",
        &py_detect_cna,
        py::arg("expr"),
        py::arg("chr"),
        py::arg("start_bp"),
        py::arg("ref")                = py::none(),
        py::arg("smooth_half_win")    = 5,
        py::arg("min_genes_per_chr")  = 10,
        py::arg("loss_mean")          = -1.0f,
        py::arg("loss_sigma")         =  1.0f,
        py::arg("neutral_mean")       =  0.0f,
        py::arg("neutral_sigma")      =  0.5f,
        py::arg("gain_mean")          =  0.5f,
        py::arg("gain_sigma")         =  1.0f,
        py::arg("trans_prob")         =  0.01f,
        py::arg("segments_per_chr")   =  50,
        py::arg("include_sex_chrs")   =  false,
        py::arg("knn_k")              =  15,
        py::arg("stream")             = py::none(),
        py::arg("seed")               =  0,
        R"doc(
GPU-native Numbat CNA detection (cycle 35).

Detects copy-number alterations at single-cell resolution via log-ratio
smoothing + HMM forward-backward + Leiden clone clustering.

Parameters
----------
expr : DeviceCsc
    Gene expression matrix (genes × cells, CSC, float32).
chr : list[int]
    Per-gene chromosome assignment (1-based: 1-22, 23=X, 24=Y, 25=MT).
    Length must equal expr.rows.
start_bp : list[int]
    Per-gene genomic start position (bp).  Used for chromosome-local ordering.
ref : ndarray float32 [n_genes], optional
    Reference expression profile (e.g. normal cells mean).
    None → kernel computes mean across all cells as reference.
smooth_half_win : int, default 5
    Rolling-window half-width; full window = 2×smooth_half_win+1 (≤15).
min_genes_per_chr : int, default 10
loss_mean, loss_sigma, neutral_mean, neutral_sigma, gain_mean, gain_sigma : float
    Gaussian HMM emission parameters per CNA state.
trans_prob : float, default 0.01
    HMM off-diagonal transition probability per segment boundary.
segments_per_chr : int, default 50
include_sex_chrs : bool, default False
knn_k : int, default 15
    kNN k for clone-clustering Leiden input graph.
stream : int or None
seed : int, default 0

Returns
-------
dict
    "cna_states"        — __cuda_array_interface__ dict (uint8, n_cells × n_segs)
                          Values: 0=loss, 1=neutral, 2=gain.
    "hmm_posteriors"    — __cuda_array_interface__ dict (float32, n_cells × n_segs × 3)
    "clone_labels"      — __cuda_array_interface__ dict (int32, n_cells)
    "n_clones"          — int
    "leiden_modularity" — float
    "n_total_segments"  — int
    "segments"          — list of dicts {chr, gene_start, gene_end, seg_idx}
    "_result_owner"     — internal lifetime anchor (keep alive with cupy views)
)doc");
}

} // namespace python
} // namespace singlet_gpu
