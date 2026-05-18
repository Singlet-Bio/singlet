// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_score_genes.hpp
//
// Cycle 186 — pybind11 binding for enrich::score_genes.
//
// Wraps singlet::gpu::enrich::score_genes (cycle 129, Phase E benchmark:
// 213-493× speedup vs scanpy.tl.score_genes, CYCLE-158).
//
// Design:
//   - Exposes PyScoreGenesResult with a scores_view __cuda_array_interface__
//     property [n_cells × n_sets, col-major float32].
//   - score_genes() Python binding accepts gene_sets as vector<vector<int>>
//     (caller builds the index lookup host-side in the Python wrapper).
//   - Gene name → index lookup is intentionally NOT done here: that requires
//     var_names, which lives in Python-land.  The Python wrapper resolves names
//     to indices, drops unknown genes (with warnings), and passes int lists here.
//   - Rule 26: thin marshalling only — no algorithmic logic.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <singlet/gpu/enrich/score_genes.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"    // PyDeviceCsc

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstdint>

namespace py = pybind11;

namespace singlet::gpu {
namespace python {

// ---------------------------------------------------------------------------
// PyScoreGenesResult
// ---------------------------------------------------------------------------
struct PyScoreGenesResult {
    singlet::gpu::core::DeviceMemory<float> scores;  // n_cells × n_sets, col-major
    int          n_cells = 0;
    int          n_sets  = 0;
    cudaStream_t stream  = nullptr;
};

// ---------------------------------------------------------------------------
// Helper: resolve cudaStream_t from Python object (None → nullptr)
// ---------------------------------------------------------------------------
inline cudaStream_t resolve_stream_sg(py::object stream_obj) {
    if (stream_obj.is_none()) return nullptr;
    return reinterpret_cast<cudaStream_t>(py::cast<uintptr_t>(stream_obj));
}

// ---------------------------------------------------------------------------
// bind_score_genes — called from PYBIND11_MODULE
// ---------------------------------------------------------------------------
inline void bind_score_genes(py::module_& m) {

    // ── ScoreGenesResult ─────────────────────────────────────────────────────
    py::class_<PyScoreGenesResult, std::shared_ptr<PyScoreGenesResult>>(
        m, "ScoreGenesResult",
        "Result of score_genes.  scores device buffer [n_cells × n_sets] col-major float32.")
    .def_property_readonly("n_cells",
        [](const PyScoreGenesResult& r){ return r.n_cells; })
    .def_property_readonly("n_sets",
        [](const PyScoreGenesResult& r){ return r.n_sets; })
    .def_property_readonly("scores_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyScoreGenesResult&>();
            const std::size_t n = static_cast<std::size_t>(r.n_cells)
                                * static_cast<std::size_t>(r.n_sets);
            return make_view_object<float>(r.scores.get(), n, r.stream, self_obj);
        },
        "[n_cells × n_sets] float32 col-major — __cuda_array_interface__.")
    .def("__repr__",
        [](const PyScoreGenesResult& r){
            return "<ScoreGenesResult n_cells=" + std::to_string(r.n_cells)
                 + " n_sets=" + std::to_string(r.n_sets) + ">";
        });

    // ── score_genes ──────────────────────────────────────────────────────────
    m.def("score_genes",
        [](py::object    mat_obj,
           py::list      gene_sets_py,
           int           ctrl_size,
           int           n_bins,
           uint64_t      seed,
           bool          use_set_size_for_ctrl,
           py::object    stream) -> py::object
        {
            auto& py_csc = mat_obj.cast<PyDeviceCsc&>();

            // Construct a non-owning PzDeviceMatrix view from PyDeviceCsc.
            // Mirrors make_pz_view in _bind_kernels.hpp without importing that
            // header (avoids ODR issues — _bind_kernels.hpp defines many statics
            // and is already included transitively through _singlet_gpu_core.cpp).
            singlet::gpu::io::PzDeviceMatrix pz;
            pz.mat = singlet::gpu::core::DeviceCSC::from_device_ptrs(
                static_cast<int>(py_csc.csc->rows),
                static_cast<int>(py_csc.csc->cols),
                static_cast<int>(py_csc.csc->nnz),
                py_csc.csc->col_ptr.data(),
                py_csc.csc->row_indices.get(),
                py_csc.csc->values.data());
            pz.producer_stream = py_csc.producer_stream;
            pz.host_retained   = false;

            cudaStream_t s = resolve_stream_sg(stream);

            // Convert Python list-of-lists to vector<vector<int>>.
            // Empty inner lists are legal here — the Python wrapper filters
            // them before calling and emits NaN for those sets.
            std::vector<std::vector<int>> gene_sets;
            gene_sets.reserve(static_cast<std::size_t>(py::len(gene_sets_py)));
            for (auto& set_obj : gene_sets_py) {
                py::list inner = py::cast<py::list>(set_obj);
                std::vector<int> indices;
                indices.reserve(static_cast<std::size_t>(py::len(inner)));
                for (auto& idx : inner) {
                    indices.push_back(py::cast<int>(idx));
                }
                gene_sets.push_back(std::move(indices));
            }

            singlet::gpu::enrich::ScoreGenesConfig cfg;
            cfg.ctrl_size             = ctrl_size;
            cfg.n_bins                = n_bins;
            cfg.seed                  = seed;
            cfg.use_set_size_for_ctrl = use_set_size_for_ctrl;
            cfg.deterministic         = true;

            auto r = singlet::gpu::enrich::score_genes(pz, gene_sets, cfg, s);

            auto py_r = std::make_shared<PyScoreGenesResult>();
            py_r->scores  = std::move(r.scores);
            py_r->n_cells = r.n_cells;
            py_r->n_sets  = r.n_sets;
            py_r->stream  = s;
            return py::cast(py_r);
        },
        py::arg("mat"),
        py::arg("gene_sets"),
        py::arg("ctrl_size")             = 50,
        py::arg("n_bins")                = 25,
        py::arg("seed")                  = 0ULL,
        py::arg("use_set_size_for_ctrl") = true,
        py::arg("stream")                = py::none(),
        R"doc(
        Per-cell gene-set scoring (cycle 129, Satija et al. 2015 / Seurat AddModuleScore).

        Parameters
        ----------
        mat : DeviceCsc
            genes × cells expression matrix (log-normalized recommended).
        gene_sets : list[list[int]]
            One inner list per gene set.  Each element is a 0-based gene index
            into the rows of ``mat``.  Build via Python wrapper's name→index lookup.
        ctrl_size : int
            Number of control genes to sample per set (default 50, scanpy default).
        n_bins : int
            Expression-mean bins for control sampling (default 25, scanpy default).
        seed : int
            RNG seed for deterministic control sampling (std::mt19937 host-side).
        use_set_size_for_ctrl : bool
            When True, effective ctrl_size = max(ctrl_size, |S|) — scanpy default.
        stream : int or None

        Returns
        -------
        ScoreGenesResult
            scores [n_cells × n_sets] float32 col-major on device.
            Sync the stream before reading via scores_view.
        )doc");
}

}  // namespace python
}  // namespace singlet::gpu
