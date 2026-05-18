// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_enrich.hpp
//
// Cycle 52a — Enrichment pybind11 bindings.
// Covers:
//   ssgsea   (enrich/ssgsea.h,  cycle 44)
//   progeny  (enrich/progeny.h, cycle 44)
//
// ssGSEA: per-cell KS enrichment, distinct from the ranked-list fgsea.h binding.
// PROGENy: weighted-sum pathway activity (SpMM + optional z-score normalization).
//
// Both expose simple run functions that return device-resident score matrices.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <singlet/gpu/enrich/ssgsea.h>
#include <singlet/gpu/enrich/progeny.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>

#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

namespace py = pybind11;

namespace singlet::gpu {
namespace python {

// ---------------------------------------------------------------------------
// PySsGseaResult
// ---------------------------------------------------------------------------
struct PySsGseaResult {
    singlet::gpu::core::DeviceMemory<float> scores;  // [n_sets × n_cells] col-major
    int   n_sets  = 0;
    int   n_cells = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyProgenyResult
// ---------------------------------------------------------------------------
struct PyProgenyResult {
    singlet::gpu::core::DeviceMemory<float> activity;  // [n_cells × n_pathways] row-major
    int   n_cells    = 0;
    int   n_pathways = 0;
    std::vector<std::string> pathway_names;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// Helper: resolve cudaStream_t from Python object (None → nullptr)
// ---------------------------------------------------------------------------
inline cudaStream_t resolve_stream_enrich(py::object stream_obj) {
    if (stream_obj.is_none()) return nullptr;
    return reinterpret_cast<cudaStream_t>(py::cast<uintptr_t>(stream_obj));
}

// ---------------------------------------------------------------------------
// bind_enrich — called from PYBIND11_MODULE
// ---------------------------------------------------------------------------
inline void bind_enrich(py::module_& m) {

    // ── SsGseaResult ──────────────────────────────────────────────────────
    py::class_<PySsGseaResult, std::shared_ptr<PySsGseaResult>>(
        m, "SsGseaResult",
        "Result of ssgsea.  scores device buffer [n_sets × n_cells] col-major.")
    .def_property_readonly("n_sets",
        [](const PySsGseaResult& r){ return r.n_sets; })
    .def_property_readonly("n_cells",
        [](const PySsGseaResult& r){ return r.n_cells; })
    .def_property_readonly("scores_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PySsGseaResult&>();
            return make_view_object<float>(r.scores.get(), r.scores.size(),
                                           r.stream, self_obj);
        }, "[n_sets × n_cells] float32 col-major — __cuda_array_interface__.")
    .def("__repr__",
        [](const PySsGseaResult& r){
            return "<SsGseaResult n_sets=" + std::to_string(r.n_sets)
                 + " n_cells=" + std::to_string(r.n_cells) + ">";
        });

    // ── ProgenyResult ──────────────────────────────────────────────────────
    py::class_<PyProgenyResult, std::shared_ptr<PyProgenyResult>>(
        m, "ProgenyResult",
        "Result of progeny.  activity device buffer [n_cells × n_pathways] row-major.")
    .def_property_readonly("n_cells",
        [](const PyProgenyResult& r){ return r.n_cells; })
    .def_property_readonly("n_pathways",
        [](const PyProgenyResult& r){ return r.n_pathways; })
    .def_property_readonly("pathway_names",
        [](const PyProgenyResult& r){ return r.pathway_names; })
    .def_property_readonly("activity_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyProgenyResult&>();
            return make_view_object<float>(r.activity.get(), r.activity.size(),
                                           r.stream, self_obj);
        }, "[n_cells × n_pathways] float32 row-major — __cuda_array_interface__.")
    .def("__repr__",
        [](const PyProgenyResult& r){
            return "<ProgenyResult n_cells=" + std::to_string(r.n_cells)
                 + " n_pathways=" + std::to_string(r.n_pathways) + ">";
        });

    // ── ssgsea ────────────────────────────────────────────────────────────
    m.def("ssgsea",
        [](const PyDeviceCsc& mat,
           py::list           gene_set_names,
           py::list           gene_set_offsets,
           py::list           gene_set_indices,
           float   alpha,
           int     cell_tile,
           int     set_chunk,
           bool    normalize,
           py::object stream,
           uint64_t seed) -> py::object
        {
            cudaStream_t s = resolve_stream_enrich(stream);
            const singlet::gpu::core::DeviceCSC& csc = *mat.csc;

            // Build SsGseaGeneSets from Python lists.
            singlet::gpu::enrich::SsGseaGeneSets gsets;
            for (auto& name : gene_set_names)
                gsets.set_names.push_back(py::cast<std::string>(name));
            for (auto& off : gene_set_offsets)
                gsets.set_offsets.push_back(py::cast<int>(off));
            for (auto& idx : gene_set_indices)
                gsets.gene_idx.push_back(py::cast<int>(idx));

            singlet::gpu::enrich::SsGseaConfig cfg;
            cfg.alpha     = alpha;
            cfg.cell_tile = cell_tile;
            cfg.set_chunk = set_chunk;
            cfg.normalize = normalize;
            cfg.seed      = seed;

            auto r = singlet::gpu::enrich::ssgsea(csc, gsets, cfg, s);

            auto py_r = std::make_shared<PySsGseaResult>();
            py_r->scores  = std::move(r.scores);
            py_r->n_sets  = r.n_sets;
            py_r->n_cells = r.n_cells;
            py_r->stream  = s;
            return py::cast(py_r);
        },
        py::arg("mat"),
        py::arg("gene_set_names"),
        py::arg("gene_set_offsets"),
        py::arg("gene_set_indices"),
        py::arg("alpha")      = 1.0f,
        py::arg("cell_tile")  = 8192,
        py::arg("set_chunk")  = 500,
        py::arg("normalize")  = true,
        py::arg("stream")     = py::none(),
        py::arg("seed")       = 0ULL,
        R"doc(
        Per-cell single-sample GSEA enrichment (cycle 44, ssGSEA kernel).

        Parameters
        ----------
        mat : DeviceCsc
            genes × cells expression matrix.
        gene_set_names : list[str]
            Name for each gene set.
        gene_set_offsets : list[int]
            CSR-style row offsets into gene_set_indices (length n_sets + 1).
        gene_set_indices : list[int]
            0-based gene indices (total_nnz).
        alpha : float
            Rank exponent (1.0 = standard ssGSEA, 0.75 = common alternative).
        normalize : bool
            Divide ES by n_genes to make alpha-independent.
        stream : int or None
        seed : int
            Reserved (ssGSEA is deterministic via secondary sort key).

        Returns
        -------
        SsGseaResult
            scores [n_sets × n_cells] col-major float32.
        )doc");

    // ── progeny ───────────────────────────────────────────────────────────
    m.def("progeny",
        [](const PyDeviceCsc& mat,
           const std::string& weights_tsv,
           py::list           gene_names_list,
           bool   normalize,
           float  norm_eps,
           int    cell_batch,
           py::object stream,
           uint64_t /*seed*/) -> py::object
        {
            cudaStream_t s = resolve_stream_enrich(stream);
            const singlet::gpu::core::DeviceCSC& csc = *mat.csc;

            // Build gene_names and name-to-idx map from Python list.
            std::vector<std::string> gene_names;
            std::unordered_map<std::string, int> name_to_idx;
            for (auto& g : gene_names_list) {
                std::string gn = py::cast<std::string>(g);
                name_to_idx[gn] = (int)gene_names.size();
                gene_names.push_back(gn);
            }

            singlet::gpu::enrich::PROGENyWeights weights =
                singlet::gpu::enrich::load_progeny_weights_tsv(
                    weights_tsv, gene_names, name_to_idx);

            // Upload weights to GPU (mutates weights.d_weights + gpu_ready).
            singlet::gpu::enrich::upload_progeny_weights(weights, s);

            // cuSPARSE handle — PROGENy uses cuSPARSE SpMM internally.
            cusparseHandle_t cusparse_h;
            cusparseCreate(&cusparse_h);
            cusparseSetStream(cusparse_h, s);

            singlet::gpu::enrich::PROGENyConfig cfg;
            cfg.normalize  = normalize;
            cfg.norm_eps   = norm_eps;
            cfg.cell_batch = cell_batch;

            auto r = singlet::gpu::enrich::progeny(csc, weights, cfg, s, cusparse_h);
            cusparseDestroy(cusparse_h);

            auto py_r = std::make_shared<PyProgenyResult>();
            py_r->activity      = std::move(r.activity);
            py_r->n_cells       = r.n_cells;
            py_r->n_pathways    = r.n_pathways;
            py_r->pathway_names = std::move(r.pathway_names);
            py_r->stream        = s;
            return py::cast(py_r);
        },
        py::arg("mat"),
        py::arg("weights_tsv"),
        py::arg("gene_names"),
        py::arg("normalize")  = true,
        py::arg("norm_eps")   = 1e-6f,
        py::arg("cell_batch") = 0,
        py::arg("stream")     = py::none(),
        py::arg("seed")       = 0ULL,
        R"doc(
        PROGENy pathway activity scoring (cycle 44, progeny kernel).

        Parameters
        ----------
        mat : DeviceCsc
            genes × cells log-normalized expression.
        weights_tsv : str
            Path to PROGENy weight TSV (columns: gene, pathway, weight).
            Use the Holland et al. 2020 human top-100 file.
        gene_names : list[str]
            Gene names corresponding to rows of ``mat``.
        normalize : bool
            Apply per-pathway z-score normalization across cells.
        cell_batch : int
            0 = process all cells at once (preferred).
        stream : int or None
        seed : int
            Unused (PROGENy is deterministic).

        Returns
        -------
        ProgenyResult
            activity [n_cells × n_pathways] float32 row-major.
        )doc");
}

}  // namespace python
}  // namespace singlet::gpu
