// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_disease.hpp
//
// pybind11 bindings for disease-score kernels:
//   - scdrs  (cycle 50) singlet::gpu::disease::run_scdrs
//
// Device result buffers exposed zero-copy via __cuda_array_interface__.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <singlet/gpu/disease/scdrs.h>
#include <singlet/gpu/core/types.h>

#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// PyScDrsResult
// ---------------------------------------------------------------------------
struct PyScDrsResult {
    singlet::gpu::core::DeviceMemory<float> raw_score;         // [n_cells × n_diseases]
    singlet::gpu::core::DeviceMemory<float> normalized_score;  // [n_cells × n_diseases]
    singlet::gpu::core::DeviceMemory<float> p_value;           // [n_cells × n_diseases]
    singlet::gpu::core::DeviceMemory<float> fdr;               // [n_cells × n_diseases]

    int n_cells    = 0;
    int n_diseases = 0;

    cudaStream_t stream = nullptr;
};

namespace singlet::gpu {
namespace python {

inline void bind_disease(py::module_& m) {

    // --- PyScDrsResult -----------------------------------------------------
    py::class_<PyScDrsResult, std::shared_ptr<PyScDrsResult>>(m, "ScDrsResult",
        "Result of scdrs().  All arrays [n_cells × n_diseases] on device.")
        .def_property_readonly("n_cells",
            [](const PyScDrsResult& r) { return r.n_cells; })
        .def_property_readonly("n_diseases",
            [](const PyScDrsResult& r) { return r.n_diseases; })
        .def_property_readonly("raw_score_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyScDrsResult&>();
                return make_view_object<float>(
                    r.raw_score.get(),
                    static_cast<std::size_t>(r.n_cells) * r.n_diseases,
                    r.stream, self_obj);
            },
            "Weighted z-score [n_cells × n_diseases] — __cuda_array_interface__.")
        .def_property_readonly("normalized_score_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyScDrsResult&>();
                return make_view_object<float>(
                    r.normalized_score.get(),
                    static_cast<std::size_t>(r.n_cells) * r.n_diseases,
                    r.stream, self_obj);
            },
            "Normalised z-score vs null [n_cells × n_diseases] — __cuda_array_interface__.")
        .def_property_readonly("p_value_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyScDrsResult&>();
                return make_view_object<float>(
                    r.p_value.get(),
                    static_cast<std::size_t>(r.n_cells) * r.n_diseases,
                    r.stream, self_obj);
            },
            "Monte Carlo p-value [n_cells × n_diseases] — __cuda_array_interface__.")
        .def_property_readonly("fdr_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyScDrsResult&>();
                return make_view_object<float>(
                    r.fdr.get(),
                    static_cast<std::size_t>(r.n_cells) * r.n_diseases,
                    r.stream, self_obj);
            },
            "BH-adjusted FDR [n_cells × n_diseases] — __cuda_array_interface__.")
        .def("__repr__",
            [](const PyScDrsResult& r) {
                return "<ScDrsResult n_cells=" + std::to_string(r.n_cells)
                     + " n_diseases=" + std::to_string(r.n_diseases) + ">";
            });

    // --- scdrs() ------------------------------------------------------------
    m.def("scdrs",
        [](py::object mat_obj,
           py::list   gene_set_names,
           py::list   gene_set_gene_lists,
           py::list   gene_set_weights_list,
           py::array_t<int32_t> gene_name_index,   // gene index for each name string
           int      n_controls,
           int      n_bins,
           int      disease_chunk,
           py::object stream_obj,
           uint64_t   seed) -> std::shared_ptr<PyScDrsResult>
        {
            auto& py_csc = mat_obj.cast<PyDeviceCsc&>();

            cudaStream_t stream = py_csc.producer_stream;
            if (!stream_obj.is_none()) {
                stream = reinterpret_cast<cudaStream_t>(
                    stream_obj.cast<uintptr_t>());
            }

            singlet::gpu::disease::ScDrsConfig cfg;
            cfg.n_controls    = n_controls;
            cfg.n_bins        = n_bins;
            cfg.seed          = seed;
            cfg.disease_chunk = disease_chunk;

            // Build DiseaseGeneSet vector from Python lists.
            // gene_set_gene_lists[i] is a list of gene-name strings.
            // gene_set_weights_list[i] is a list of floats (or empty → weight=1.0).
            // gene_name_index maps gene-name hashes to matrix row indices (not used
            // at binding level — the C++ kernel receives row-index lists directly).
            int n_diseases = (int)gene_set_names.size();
            std::vector<singlet::gpu::disease::DiseaseGeneSet> gene_sets;
            gene_sets.reserve(n_diseases);

            for (int d = 0; d < n_diseases; ++d) {
                singlet::gpu::disease::DiseaseGeneSet gs;
                gs.name = gene_set_names[d].cast<std::string>();

                py::list genes   = gene_set_gene_lists[d].cast<py::list>();
                py::list weights = gene_set_weights_list[d].cast<py::list>();
                bool has_weights = (weights.size() == genes.size());

                for (int g = 0; g < (int)genes.size(); ++g) {
                    std::string gname = genes[g].cast<std::string>();
                    float w = has_weights ? weights[g].cast<float>() : 1.0f;
                    gs.genes[gname] = w;
                }
                gene_sets.push_back(std::move(gs));
            }

            singlet::gpu::io::PzDeviceMatrix pz;
            pz.mat = singlet::gpu::core::DeviceCSC::from_device_ptrs(
                (int)py_csc.csc->rows, (int)py_csc.csc->cols, (int)py_csc.csc->nnz,
                py_csc.csc->col_ptr.data(),
                py_csc.csc->row_indices.data(),
                py_csc.csc->values.data());
            pz.producer_stream = stream;
            pz.host_retained   = false;

            // Row names from the PzDeviceMatrix metadata (gene names).
            const std::vector<std::string>& row_names = py_csc.meta.rownames;

            singlet::gpu::disease::ScDrsResult r =
                singlet::gpu::disease::run_scdrs(
                    pz, row_names, gene_sets, cfg, stream);

            auto out = std::make_shared<PyScDrsResult>();
            out->raw_score        = std::move(r.raw_score);
            out->normalized_score = std::move(r.normalized_score);
            out->p_value          = std::move(r.p_value);
            out->fdr              = std::move(r.fdr);
            out->n_cells          = r.n_cells;
            out->n_diseases       = r.n_diseases;
            out->stream           = stream;
            return out;
        },
        py::arg("mat"),
        py::arg("gene_set_names"),
        py::arg("gene_set_gene_lists"),
        py::arg("gene_set_weights_list")  = py::list(),
        py::arg("gene_name_index")        = py::array_t<int32_t>(),
        py::arg("n_controls")             = 1000,
        py::arg("n_bins")                 = 20,
        py::arg("disease_chunk")          = 5,
        py::arg("stream")                 = py::none(),
        py::arg("seed")                   = 0ULL,
        R"doc(
        GPU-native scDRS cell-level disease relevance scoring (cycle-50 kernel).

        Parameters
        ----------
        mat : DeviceCsc
            Expression matrix (genes × cells) on device.  Should be
            normalised and z-scored by the C++ kernel internally.
        gene_set_names : list of str
            Name of each disease/trait gene set.
        gene_set_gene_lists : list of list of str
            Per-disease list of gene names.
        gene_set_weights_list : list of list of float or empty list
            Per-disease per-gene MAGMA weights.  If empty, all weights = 1.0.
        n_controls : int, default 1000
            Number of matched control gene sets per disease.
        n_bins : int, default 20
            Bins per axis for (mean, var) gene matching.
        disease_chunk : int, default 5
            Diseases processed per OOC pass (0 = all at once).
        stream : int or None
            CUDA stream pointer.
        seed : int, default 0
            Seed for host-side control set sampling.

        Returns
        -------
        ScDrsResult
        )doc");
}

inline void bind_disease_module(py::module_& parent) {
    auto m = parent.def_submodule("disease",
        "GPU-native disease-score kernels: scdrs.");
    bind_disease(m);
}

} // namespace python
} // namespace singlet::gpu
