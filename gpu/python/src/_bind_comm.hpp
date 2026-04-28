// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/python/src/_bind_comm.hpp
//
// pybind11 bindings for cell-communication kernels:
//   - cellchat  (cycle 37) singlet_gpu::comm::run_cellchat
//
// Pattern mirrors _bind_kernels.hpp / _bind_results.hpp.
// Device result buffers exposed zero-copy via __cuda_array_interface__.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <singlet-gpu/comm/cellchat.h>
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
// PyCellChatResult
// ---------------------------------------------------------------------------
struct PyCellChatResult {
    singlet_gpu::core::DeviceMemory<float> comm_prob;       // [n_lr × n_types²]
    singlet_gpu::core::DeviceMemory<float> p_values;        // [n_lr × n_types²]
    singlet_gpu::core::DeviceMemory<float> pathway_activity;// [n_pathways × n_types²]

    int n_types    = 0;
    int n_lr       = 0;
    int n_pathways = 0;

    cudaStream_t stream = nullptr;
};

namespace singlet_gpu {
namespace python {

inline void bind_comm(py::module_& m) {

    // --- PyCellChatResult --------------------------------------------------
    py::class_<PyCellChatResult, std::shared_ptr<PyCellChatResult>>(m, "CellChatResult",
        "Result of cellchat().  Device arrays exposed via __cuda_array_interface__.")
        .def_property_readonly("n_types",
            [](const PyCellChatResult& r) { return r.n_types; })
        .def_property_readonly("n_lr",
            [](const PyCellChatResult& r) { return r.n_lr; })
        .def_property_readonly("n_pathways",
            [](const PyCellChatResult& r) { return r.n_pathways; })
        .def_property_readonly("comm_prob_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyCellChatResult&>();
                return make_view_object<float>(
                    r.comm_prob.get(),
                    static_cast<std::size_t>(r.n_lr) * r.n_types * r.n_types,
                    r.stream, self_obj);
            },
            "Communication probability [n_lr × n_types × n_types] — __cuda_array_interface__.")
        .def_property_readonly("p_values_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyCellChatResult&>();
                return make_view_object<float>(
                    r.p_values.get(),
                    static_cast<std::size_t>(r.n_lr) * r.n_types * r.n_types,
                    r.stream, self_obj);
            },
            "Permutation p-values [n_lr × n_types²] — __cuda_array_interface__.")
        .def_property_readonly("pathway_activity_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyCellChatResult&>();
                return make_view_object<float>(
                    r.pathway_activity.get(),
                    static_cast<std::size_t>(r.n_pathways) * r.n_types * r.n_types,
                    r.stream, self_obj);
            },
            "Pathway activity [n_pathways × n_types²] — __cuda_array_interface__.")
        .def("__repr__",
            [](const PyCellChatResult& r) {
                return "<CellChatResult n_lr=" + std::to_string(r.n_lr)
                     + " n_types=" + std::to_string(r.n_types)
                     + " n_pathways=" + std::to_string(r.n_pathways) + ">";
            });

    // --- cellchat() ---------------------------------------------------------
    m.def("cellchat",
        [](py::object mat_obj,
           py::array_t<int32_t> cell_type,
           py::array_t<int32_t> lr_receptor,
           py::array_t<int32_t> lr_ligand,
           py::array_t<int32_t> pathway_id,
           int      n_pathways,
           int      n_permutations,
           float    hill_K,
           bool     deterministic,
           py::object stream_obj,
           uint64_t seed) -> std::shared_ptr<PyCellChatResult>
        {
            auto& py_csc = mat_obj.cast<PyDeviceCsc&>();

            cudaStream_t stream = py_csc.producer_stream;
            if (!stream_obj.is_none()) {
                stream = reinterpret_cast<cudaStream_t>(
                    stream_obj.cast<uintptr_t>());
            }

            singlet_gpu::comm::CellChatConfig cfg;
            cfg.n_permutations = n_permutations;
            cfg.hill_K         = hill_K;
            cfg.deterministic  = deterministic;
            cfg.seed           = seed;

            singlet_gpu::io::PzDeviceMatrix pz;
            pz.mat = singlet_gpu::core::DeviceCSC::from_device_ptrs(
                (int)py_csc.csc->rows, (int)py_csc.csc->cols, (int)py_csc.csc->nnz,
                py_csc.csc->col_ptr.data(),
                py_csc.csc->row_indices.data(),
                py_csc.csc->values.data());
            pz.producer_stream = stream;
            pz.host_retained   = false;

            auto ct   = cell_type.unchecked<1>();
            auto lrr  = lr_receptor.unchecked<1>();
            auto lrl  = lr_ligand.unchecked<1>();
            auto pwid = pathway_id.unchecked<1>();

            int n_types = *std::max_element(ct.data(0), ct.data(0) + ct.shape(0)) + 1;
            int n_lr    = (int)lrr.shape(0);

            singlet_gpu::comm::CellChatResult r =
                singlet_gpu::comm::run_cellchat(
                    pz,
                    ct.data(0), (int)py_csc.csc->cols,
                    n_types,
                    lrl.data(0), lrr.data(0), n_lr,
                    pwid.data(0), n_pathways,
                    cfg, stream);

            auto out = std::make_shared<PyCellChatResult>();
            out->comm_prob        = std::move(r.comm_prob);
            out->p_values         = std::move(r.p_values);
            out->pathway_activity = std::move(r.pathway_activity);
            out->n_types          = r.n_types;
            out->n_lr             = r.n_lr;
            out->n_pathways       = r.n_pathways;
            out->stream           = stream;
            return out;
        },
        py::arg("mat"),
        py::arg("cell_type"),
        py::arg("lr_receptor"),
        py::arg("lr_ligand"),
        py::arg("pathway_id"),
        py::arg("n_pathways")      = 0,
        py::arg("n_permutations")  = 1000,
        py::arg("hill_K")          = 0.5f,
        py::arg("deterministic")   = false,
        py::arg("stream")          = py::none(),
        py::arg("seed")            = 0ULL,
        R"doc(
        GPU-native CellChat ligand-receptor communication scoring (cycle-37 kernel).

        Parameters
        ----------
        mat : DeviceCsc
            Expression matrix (genes × cells) on device.
        cell_type : np.ndarray int32, shape (n_cells,)
            Integer cell-type label (0-indexed).
        lr_receptor : np.ndarray int32, shape (n_lr,)
            Gene index of the receptor for each LR pair.
        lr_ligand : np.ndarray int32, shape (n_lr,)
            Gene index of the ligand for each LR pair.
        pathway_id : np.ndarray int32, shape (n_lr,)
            Pathway index (0-indexed) for each LR pair.
        n_pathways : int
            Total number of pathways (0 = infer from pathway_id.max()+1).
        n_permutations : int, default 1000
            Permutations for p-value estimation.
        hill_K : float, default 0.5
            Hill function half-saturation constant.
        deterministic : bool, default False
            Use deterministic (scan-based) count path.
        stream : int or None
            CUDA stream pointer.
        seed : int, default 0
            Philox seed for permutation sampling.

        Returns
        -------
        CellChatResult
        )doc");
}

inline void bind_comm_module(py::module_& parent) {
    auto m = parent.def_submodule("comm",
        "GPU-native cell communication kernels: cellchat.");
    bind_comm(m);
}

} // namespace python
} // namespace singlet_gpu
