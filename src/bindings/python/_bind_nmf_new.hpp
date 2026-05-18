// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_nmf_new.hpp
//
// Cycle 52a — CSI-GEP pybind11 binding (reduce/nmf/csi_gep.h, cycle 28).
//
// Exposes:
//   csi_gep(mat, ...) -> CsiGepResult
//
// CsiGepResult holds:
//   chosen_k                  — auto-selected rank
//   consensus_programs_view   — [chosen_k × n_genes] row-major device float32
//   program_usage_view        — [n_cells × chosen_k] device float32
//   reproducibility_curve     — host list (float, len k_range)
//   k_used                    — host list (int, mirrors k_range)
//
// The NmfConfig (forwarded to inner fit calls) is exposed as a flat parameter
// list to avoid a nested Config object on the Python side.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <singlet/gpu/reduce/nmf/csi_gep.h>
#include <singlet/gpu/reduce/nmf/types.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>

#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cstdint>

namespace py = pybind11;

namespace singlet::gpu {
namespace python {

// ---------------------------------------------------------------------------
// PyCsiGepResult
// ---------------------------------------------------------------------------
struct PyCsiGepResult {
    int chosen_k = 0;
    singlet::gpu::core::DeviceMemory<float> consensus_programs;  // chosen_k × n_genes
    singlet::gpu::core::DeviceMemory<float> program_usage;       // n_cells × chosen_k
    std::vector<float> reproducibility_curve;
    std::vector<int>   k_used;
    int n_genes  = 0;
    int n_cells  = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// Helper: resolve cudaStream_t from Python object (None → nullptr)
// ---------------------------------------------------------------------------
inline cudaStream_t resolve_stream_nmf2(py::object stream_obj) {
    if (stream_obj.is_none()) return nullptr;
    return reinterpret_cast<cudaStream_t>(py::cast<uintptr_t>(stream_obj));
}

// ---------------------------------------------------------------------------
// bind_nmf_new — called from PYBIND11_MODULE
// ---------------------------------------------------------------------------
inline void bind_nmf_new(py::module_& m) {

    // ── CsiGepResult ──────────────────────────────────────────────────────
    py::class_<PyCsiGepResult, std::shared_ptr<PyCsiGepResult>>(
        m, "CsiGepResult",
        "Result of csi_gep.  Consensus gene expression programs + cell usages.")
    .def_property_readonly("chosen_k",
        [](const PyCsiGepResult& r){ return r.chosen_k; },
        "Auto-selected rank (elbow on reproducibility curve).")
    .def_property_readonly("n_genes",
        [](const PyCsiGepResult& r){ return r.n_genes; })
    .def_property_readonly("n_cells",
        [](const PyCsiGepResult& r){ return r.n_cells; })
    .def_property_readonly("reproducibility_curve",
        [](const PyCsiGepResult& r){ return r.reproducibility_curve; },
        "Mean max-Jaccard reproducibility score per k in k_range.")
    .def_property_readonly("k_used",
        [](const PyCsiGepResult& r){ return r.k_used; },
        "k values evaluated (mirrors k_range config).")
    .def_property_readonly("consensus_programs_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyCsiGepResult&>();
            return make_view_object<float>(
                r.consensus_programs.get(),
                r.consensus_programs.size(),
                r.stream, self_obj);
        }, "[chosen_k × n_genes] float32 row-major — __cuda_array_interface__.")
    .def_property_readonly("program_usage_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyCsiGepResult&>();
            return make_view_object<float>(
                r.program_usage.get(),
                r.program_usage.size(),
                r.stream, self_obj);
        }, "[n_cells × chosen_k] float32 NNLS coefficients — __cuda_array_interface__.")
    .def("__repr__",
        [](const PyCsiGepResult& r){
            return "<CsiGepResult chosen_k=" + std::to_string(r.chosen_k)
                 + " n_genes=" + std::to_string(r.n_genes)
                 + " n_cells=" + std::to_string(r.n_cells) + ">";
        });

    // ── csi_gep function ──────────────────────────────────────────────────
    m.def("csi_gep",
        [](const PyDeviceCsc& mat,
           py::list           k_range_list,
           int     n_runs,
           float   subsample_frac,
           int     top_n_genes_jaccard,
           float   elbow_gap_threshold,
           int     max_kmeans_iters,
           float   kmeans_tol,
           // NmfConfig inner params
           int     nmf_max_iter,
           float   nmf_tol,
           py::object stream,
           uint64_t seed) -> py::object
        {
            cudaStream_t s = resolve_stream_nmf2(stream);
            const singlet::gpu::core::DeviceCSC& csc = *mat.csc;

            singlet::gpu::reduce::nmf::CsiGepConfig cfg;
            cfg.k_range.clear();
            for (auto& k : k_range_list)
                cfg.k_range.push_back(py::cast<int>(k));
            cfg.n_runs              = n_runs;
            cfg.subsample_frac      = subsample_frac;
            cfg.top_n_genes_jaccard = top_n_genes_jaccard;
            cfg.elbow_gap_threshold = elbow_gap_threshold;
            cfg.seed                = seed;
            cfg.max_kmeans_iters    = max_kmeans_iters;
            cfg.kmeans_tol          = kmeans_tol;
            cfg.nmf_cfg.max_iter    = nmf_max_iter;
            cfg.nmf_cfg.tol         = nmf_tol;
            // Override inner NMF seed per-run by convention (seed + run_index).
            cfg.nmf_cfg.seed        = seed;

            auto r = singlet::gpu::reduce::nmf::csi_gep(csc, cfg, s);

            auto py_r = std::make_shared<PyCsiGepResult>();
            py_r->chosen_k              = r.chosen_k;
            py_r->consensus_programs    = std::move(r.consensus_programs);
            py_r->program_usage         = std::move(r.program_usage);
            py_r->reproducibility_curve = std::move(r.reproducibility_curve);
            py_r->k_used                = std::move(r.k_used);
            py_r->n_genes               = (int)csc.nrows;
            py_r->n_cells               = (int)csc.ncols;
            py_r->stream                = s;
            return py::cast(py_r);
        },
        py::arg("mat"),
        py::arg("k_range")              = py::list(),
        py::arg("n_runs")               = 100,
        py::arg("subsample_frac")       = 0.8f,
        py::arg("top_n_genes_jaccard")  = 30,
        py::arg("elbow_gap_threshold")  = 0.05f,
        py::arg("max_kmeans_iters")     = 100,
        py::arg("kmeans_tol")           = 1e-4f,
        py::arg("nmf_max_iter")         = 200,
        py::arg("nmf_tol")              = 1e-4f,
        py::arg("stream")               = py::none(),
        py::arg("seed")                 = 0ULL,
        R"doc(
        Consensus NMF gene expression programs via CSI-GEP (cycle 28).

        Parameters
        ----------
        mat : DeviceCsc
            genes × cells matrix.  Must have host-pinned copies (pz_device_loader sets these).
        k_range : list[int]
            Ranks to evaluate.  Default [5, 10, 15, 20, 25, 30].
        n_runs : int
            Number of independent NMF runs per k.
        subsample_frac : float
            Fraction of cells per run (Philox-subsampled).
        top_n_genes_jaccard : int
            Top-N genes used to compute per-program Jaccard reproducibility.
        elbow_gap_threshold : float
            Gap threshold on reproducibility curve for elbow detection.
        stream : int or None
        seed : int

        Returns
        -------
        CsiGepResult
        )doc");
}

}  // namespace python
}  // namespace singlet::gpu
