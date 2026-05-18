// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_network.hpp
//
// pybind11 bindings for gene-regulatory network kernels:
//   - hdwgcna  (cycle 46) singlet::gpu::network::run_hdwgcna
//
// Host-side result vectors returned as numpy arrays.
// Device buffers (if any) exposed via __cuda_array_interface__.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <singlet/gpu/network/hdwgcna.h>
#include <singlet/gpu/core/types.h>

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
// PyHdwgcnaResult
// ---------------------------------------------------------------------------
struct PyHdwgcnaResult {
    std::vector<int>   module_assignment;  // n_genes (host)
    std::vector<float> eigengenes;         // n_modules × n_cells row-major (host)
    std::vector<int>   hub_genes;          // n_modules × n_hub_genes row-major (host)
    std::vector<float> kme;               // n_genes (host)

    int n_modules   = 0;
    int n_cells     = 0;
    int n_genes     = 0;
    int n_hub_genes = 0;

    cudaStream_t stream = nullptr;
};

namespace singlet::gpu {
namespace python {

inline void bind_network(py::module_& m) {

    // --- PyHdwgcnaResult ---------------------------------------------------
    py::class_<PyHdwgcnaResult, std::shared_ptr<PyHdwgcnaResult>>(m, "HdwgcnaResult",
        "Result of hdwgcna().  All arrays host-side numpy.")
        .def_property_readonly("n_modules",
            [](const PyHdwgcnaResult& r) { return r.n_modules; })
        .def_property_readonly("n_cells",
            [](const PyHdwgcnaResult& r) { return r.n_cells; })
        .def_property_readonly("n_genes",
            [](const PyHdwgcnaResult& r) { return r.n_genes; })
        .def_property_readonly("module_assignment",
            [](const PyHdwgcnaResult& r) {
                return py::array_t<int32_t>(
                    {r.n_genes}, r.module_assignment.data());
            },
            "Per-gene module label (0 = grey/unassigned), numpy int32.")
        .def_property_readonly("eigengenes",
            [](const PyHdwgcnaResult& r) {
                return py::array_t<float>(
                    {r.n_modules, r.n_cells}, r.eigengenes.data());
            },
            "Module eigengene matrix (n_modules × n_cells), numpy float32.")
        .def_property_readonly("hub_genes",
            [](const PyHdwgcnaResult& r) {
                return py::array_t<int32_t>(
                    {r.n_modules, r.n_hub_genes}, r.hub_genes.data());
            },
            "Hub gene indices (n_modules × n_hub_genes), numpy int32.")
        .def_property_readonly("kme",
            [](const PyHdwgcnaResult& r) {
                return py::array_t<float>({r.n_genes}, r.kme.data());
            },
            "Module eigengene correlation per gene (n_genes,), numpy float32.")
        .def("__repr__",
            [](const PyHdwgcnaResult& r) {
                return "<HdwgcnaResult n_genes=" + std::to_string(r.n_genes)
                     + " n_modules=" + std::to_string(r.n_modules)
                     + " n_cells=" + std::to_string(r.n_cells) + ">";
            });

    // --- hdwgcna() ----------------------------------------------------------
    m.def("hdwgcna",
        [](py::object mat_obj,
           float    soft_power,
           int      min_module_size,
           float    height_cut_fraction,
           int      n_hub_genes,
           bool     deterministic,
           int      ooc_chunk_size,
           py::object stream_obj,
           uint64_t seed) -> std::shared_ptr<PyHdwgcnaResult>
        {
            auto& py_csc = mat_obj.cast<PyDeviceCsc&>();

            cudaStream_t stream = py_csc.producer_stream;
            if (!stream_obj.is_none()) {
                stream = reinterpret_cast<cudaStream_t>(
                    stream_obj.cast<uintptr_t>());
            }

            singlet::gpu::network::HdwgcnaConfig cfg;
            cfg.soft_power          = soft_power;
            cfg.min_module_size     = min_module_size;
            cfg.height_cut_fraction = height_cut_fraction;
            cfg.n_hub_genes         = n_hub_genes;
            cfg.deterministic       = deterministic;
            cfg.ooc_chunk_size      = ooc_chunk_size;
            cfg.seed                = seed;

            singlet::gpu::io::PzDeviceMatrix pz;
            pz.mat = singlet::gpu::core::DeviceCSC::from_device_ptrs(
                (int)py_csc.csc->rows, (int)py_csc.csc->cols, (int)py_csc.csc->nnz,
                py_csc.csc->col_ptr.data(),
                py_csc.csc->row_indices.data(),
                py_csc.csc->values.data());
            pz.producer_stream = stream;
            pz.host_retained   = false;

            singlet::gpu::network::HdwgcnaResult r =
                singlet::gpu::network::run_hdwgcna(pz, cfg, stream);

            auto out = std::make_shared<PyHdwgcnaResult>();
            out->module_assignment = std::move(r.module_assignment);
            out->eigengenes        = std::move(r.eigengenes);
            out->hub_genes         = std::move(r.hub_genes);
            out->kme               = std::move(r.kme);
            out->n_modules         = r.n_modules;
            out->n_cells           = r.n_cells;
            out->n_genes           = (int)out->module_assignment.size();
            out->n_hub_genes       = r.n_hub_genes;
            out->stream            = stream;
            return out;
        },
        py::arg("mat"),
        py::arg("soft_power")          = 6.0f,
        py::arg("min_module_size")     = 30,
        py::arg("height_cut_fraction") = 0.25f,
        py::arg("n_hub_genes")         = 10,
        py::arg("deterministic")       = false,
        py::arg("ooc_chunk_size")      = 0,
        py::arg("stream")              = py::none(),
        py::arg("seed")                = 42ULL,
        R"doc(
        GPU-native hdWGCNA co-expression network analysis (cycle-46 kernel).

        Parameters
        ----------
        mat : DeviceCsc
            Expression matrix (genes × cells) on device.  Should be
            normalised / log-transformed before calling this function.
        soft_power : float, default 6.0
            WGCNA soft-thresholding power beta (6 for unsigned, 12 for signed).
        min_module_size : int, default 30
            Minimum number of genes to form a module.
        height_cut_fraction : float, default 0.25
            Dendrogram height cut threshold as fraction of max linkage height.
        n_hub_genes : int, default 10
            Top-N hub genes returned per module.
        deterministic : bool, default False
            Pass CUBLAS_ATOMICS_NOT_ALLOWED for deterministic SYRK/GEMM.
        ooc_chunk_size : int, default 0
            Chunk size for OOC correlation; 0 = auto from free device memory.
        stream : int or None
            CUDA stream pointer.
        seed : int, default 42
            Seed for randomised SVD used in eigengene computation.

        Returns
        -------
        HdwgcnaResult
        )doc");
}

inline void bind_network_module(py::module_& parent) {
    auto m = parent.def_submodule("network",
        "GPU-native co-expression network kernels: hdwgcna.");
    bind_network(m);
}

} // namespace python
} // namespace singlet::gpu
