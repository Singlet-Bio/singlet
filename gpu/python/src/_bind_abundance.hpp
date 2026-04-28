// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/python/src/_bind_abundance.hpp
//
// pybind11 bindings for differential-abundance kernels:
//   - milo  (cycle 47) singlet_gpu::abundance::run_milo
//
// Device buffers exposed zero-copy via __cuda_array_interface__.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <singlet-gpu/abundance/milo.h>
#include <singlet-gpu/core/types.h>

#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// PyMiloResult
// ---------------------------------------------------------------------------
struct PyMiloResult {
    singlet_gpu::core::DeviceMemory<float> log_fc;          // [n_nh]
    singlet_gpu::core::DeviceMemory<float> se;              // [n_nh]
    singlet_gpu::core::DeviceMemory<float> p_value;         // [n_nh]
    singlet_gpu::core::DeviceMemory<float> fdr;             // [n_nh]
    singlet_gpu::core::DeviceMemory<int>   nh_index_cells;  // [n_nh]
    singlet_gpu::core::DeviceMemory<int>   nh_valid;        // [n_nh]
    singlet_gpu::core::DeviceMemory<float> nh_counts;       // [n_nh × n_donors]

    int n_nh     = 0;
    int n_donors = 0;

    cudaStream_t stream = nullptr;
};

namespace singlet_gpu {
namespace python {

inline void bind_abundance(py::module_& m) {

    // --- PyMiloResult ------------------------------------------------------
    py::class_<PyMiloResult, std::shared_ptr<PyMiloResult>>(m, "MiloResult",
        "Result of milo().  Device arrays exposed via __cuda_array_interface__.")
        .def_property_readonly("n_nh",
            [](const PyMiloResult& r) { return r.n_nh; })
        .def_property_readonly("n_donors",
            [](const PyMiloResult& r) { return r.n_donors; })
        .def_property_readonly("log_fc_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyMiloResult&>();
                return make_view_object<float>(
                    r.log_fc.get(), r.n_nh, r.stream, self_obj);
            },
            "Log fold-change per neighborhood (n_nh,) — __cuda_array_interface__.")
        .def_property_readonly("se_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyMiloResult&>();
                return make_view_object<float>(
                    r.se.get(), r.n_nh, r.stream, self_obj);
            },
            "Standard error of LFC per neighborhood — __cuda_array_interface__.")
        .def_property_readonly("p_value_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyMiloResult&>();
                return make_view_object<float>(
                    r.p_value.get(), r.n_nh, r.stream, self_obj);
            },
            "Wald test p-values (n_nh,) — __cuda_array_interface__.")
        .def_property_readonly("fdr_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyMiloResult&>();
                return make_view_object<float>(
                    r.fdr.get(), r.n_nh, r.stream, self_obj);
            },
            "Graph-weighted BH FDR (n_nh,) — __cuda_array_interface__.")
        .def_property_readonly("nh_index_cells_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyMiloResult&>();
                return make_view_object<int>(
                    r.nh_index_cells.get(), r.n_nh, r.stream, self_obj);
            },
            "Index cell for each neighborhood (n_nh,) — __cuda_array_interface__.")
        .def_property_readonly("nh_valid_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyMiloResult&>();
                return make_view_object<int>(
                    r.nh_valid.get(), r.n_nh, r.stream, self_obj);
            },
            "Validity flag per neighborhood (1=included in GLM) — __cuda_array_interface__.")
        .def_property_readonly("nh_counts_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyMiloResult&>();
                return make_view_object<float>(
                    r.nh_counts.get(),
                    static_cast<std::size_t>(r.n_nh) * r.n_donors,
                    r.stream, self_obj);
            },
            "Donor cell counts per neighborhood [n_nh × n_donors] — __cuda_array_interface__.")
        .def("__repr__",
            [](const PyMiloResult& r) {
                return "<MiloResult n_nh=" + std::to_string(r.n_nh)
                     + " n_donors=" + std::to_string(r.n_donors) + ">";
            });

    // --- milo() -------------------------------------------------------------
    m.def("milo",
        [](py::array_t<float>   embedding,
           py::array_t<int32_t> condition,
           py::array_t<int32_t> donor_id,
           int      k,
           int      n_nh_target,
           int      max_irls_iters,
           float    irls_tol,
           float    min_cells_fraction,
           int      min_donors_per_nh,
           bool     deterministic,
           py::object stream_obj,
           uint64_t seed) -> std::shared_ptr<PyMiloResult>
        {
            cudaStream_t stream = nullptr;
            if (!stream_obj.is_none()) {
                stream = reinterpret_cast<cudaStream_t>(
                    stream_obj.cast<uintptr_t>());
            }

            singlet_gpu::abundance::MiloConfig cfg;
            cfg.k                   = k;
            cfg.n_nh_target         = n_nh_target;
            cfg.max_irls_iters      = max_irls_iters;
            cfg.irls_tol            = irls_tol;
            cfg.min_cells_fraction  = min_cells_fraction;
            cfg.min_donors_per_nh   = min_donors_per_nh;
            cfg.seed                = seed;
            cfg.deterministic       = deterministic;

            auto emb  = embedding.unchecked<2>();   // (n_cells, n_dims)
            auto cond = condition.unchecked<1>();   // (n_cells,) 0/1
            auto don  = donor_id.unchecked<1>();    // (n_cells,)

            int n_cells  = (int)emb.shape(0);
            int n_dims   = (int)emb.shape(1);
            int n_donors = *std::max_element(don.data(0), don.data(0) + n_cells) + 1;

            singlet_gpu::abundance::MiloResult r =
                singlet_gpu::abundance::run_milo(
                    emb.data(0, 0), n_cells, n_dims,
                    cond.data(0),
                    don.data(0), n_donors,
                    cfg, stream);

            auto out = std::make_shared<PyMiloResult>();
            out->log_fc         = std::move(r.log_fc);
            out->se             = std::move(r.se);
            out->p_value        = std::move(r.p_value);
            out->fdr            = std::move(r.fdr);
            out->nh_index_cells = std::move(r.nh_index_cells);
            out->nh_valid       = std::move(r.nh_valid);
            out->nh_counts      = std::move(r.nh_counts);
            out->n_nh           = r.n_nh;
            out->n_donors       = r.n_donors;
            out->stream         = stream;
            return out;
        },
        py::arg("embedding"),
        py::arg("condition"),
        py::arg("donor_id"),
        py::arg("k")                  = 30,
        py::arg("n_nh_target")        = 2000,
        py::arg("max_irls_iters")     = 20,
        py::arg("irls_tol")           = 1e-4f,
        py::arg("min_cells_fraction") = 0.2f,
        py::arg("min_donors_per_nh")  = 3,
        py::arg("deterministic")      = true,
        py::arg("stream")             = py::none(),
        py::arg("seed")               = 42ULL,
        R"doc(
        GPU-native Milo differential-abundance testing (cycle-47 kernel).

        Parameters
        ----------
        embedding : np.ndarray float32, shape (n_cells, n_dims)
            Low-dimensional cell embedding (e.g. PCA).
        condition : np.ndarray int32, shape (n_cells,)
            Binary condition label (0 / 1).
        donor_id : np.ndarray int32, shape (n_cells,)
            Donor identifier (0-indexed integers).
        k : int, default 30
            kNN neighborhood size for graph construction.
        n_nh_target : int, default 2000
            Target number of neighborhoods to sample.
        stream : int or None
            CUDA stream pointer.
        seed : int, default 42
            Philox seed for neighborhood sampling.

        Returns
        -------
        MiloResult
        )doc");
}

inline void bind_abundance_module(py::module_& parent) {
    auto m = parent.def_submodule("abundance",
        "GPU-native differential-abundance kernels: milo.");
    bind_abundance(m);
}

} // namespace python
} // namespace singlet_gpu
