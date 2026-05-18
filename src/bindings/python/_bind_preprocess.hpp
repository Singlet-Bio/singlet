// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_preprocess.hpp
//
// CYCLE-103 — pybind11 bindings for:
//   preprocess::scale       (scale.h:221, DeviceCSC overload) → DeviceDense
//   preprocess::regress_out (scale.h:279)                    → void (in-place)
//
// PyDenseResult wraps DeviceDense (DenseMatrixGPU<float>) via a single
// DeviceMemory<float> buffer and exposes it via __cuda_array_interface__ as a
// 2-D array (n_genes × n_cells, row-major).
//
// scale:
//   Accepts a DeviceCsc and two device float32 arrays (mean, std) that the
//   caller pre-computed (e.g. via calculate_qc_metrics or HVG kernel).
//   Returns a PyDenseResult.
//
// regress_out:
//   Accepts a PyDenseResult (mutates in-place) and a device col-major design
//   matrix C [n_cells × p].  Returns None.
//   Raises ValueError if p > 32 (C++ hard limit; surfaces cleanly here).

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <singlet/gpu/preprocess/scale.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>

#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <memory>

namespace py = pybind11;

namespace singlet::gpu {
namespace python {

// ---------------------------------------------------------------------------
// PyDenseResult — owns the DeviceDense returned by scale().
//
// WHY own by DeviceMemory<float> rather than DeviceDense directly:
//   DeviceDense (= DenseMatrixGPU<float>) is a move-only type.  Wrapping it
//   in a std::shared_ptr<PyDenseResult> is straightforward — just move the
//   entire DeviceDense inside.  We alias the .data field for the view.
// ---------------------------------------------------------------------------
struct PyDenseResult {
    singlet::gpu::core::DeviceDense dense;  // owns n_genes*n_cells floats
    cudaStream_t stream = nullptr;

    int n_genes() const { return dense.rows; }
    int n_cells() const { return dense.cols; }
};

// ---------------------------------------------------------------------------
// Helper: resolve stream
// ---------------------------------------------------------------------------
inline cudaStream_t resolve_stream_pp(py::object stream_obj) {
    if (stream_obj.is_none()) return nullptr;
    return reinterpret_cast<cudaStream_t>(py::cast<uintptr_t>(stream_obj));
}

// ---------------------------------------------------------------------------
// bind_preprocess — called from PYBIND11_MODULE(_core, m)
// ---------------------------------------------------------------------------
inline void bind_preprocess(py::module_& m) {

    // ── DenseResult Python class ─────────────────────────────────────────
    py::class_<PyDenseResult, std::shared_ptr<PyDenseResult>>(
        m, "DenseResult",
        "Scaled dense expression matrix (n_genes × n_cells, row-major float32).")
    .def_property_readonly("n_genes",
        [](const PyDenseResult& r){ return r.n_genes(); })
    .def_property_readonly("n_cells",
        [](const PyDenseResult& r){ return r.n_cells(); })
    .def_property_readonly("data_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyDenseResult&>();
            // Expose as 1-D flat buffer; Python wrapper reshapes to (n_genes, n_cells).
            return make_view_object<float>(
                r.dense.data.get(),
                r.dense.data.size(),
                r.stream, self_obj);
        }, "[n_genes * n_cells] float32 row-major buffer — __cuda_array_interface__.")
    .def("__repr__",
        [](const PyDenseResult& r){
            return "<DenseResult n_genes=" + std::to_string(r.n_genes())
                 + " n_cells=" + std::to_string(r.n_cells()) + ">";
        });

    // ── scale function ────────────────────────────────────────────────────
    m.def("scale",
        [](const PyDeviceCsc& mat,
           py::object mean_obj,
           py::object std_obj,
           float max_value,
           bool  zero_center,
           bool  unit_variance,
           py::object stream_obj) -> py::object
        {
            cudaStream_t s = resolve_stream_pp(stream_obj);

            // Unwrap mean / std device arrays via __cuda_array_interface__.
            auto mean_iface = mean_obj.attr("__cuda_array_interface__").cast<py::dict>();
            auto mean_ptr = reinterpret_cast<const float*>(
                py::cast<uintptr_t>(mean_iface["data"].cast<py::tuple>()[0]));

            auto std_iface = std_obj.attr("__cuda_array_interface__").cast<py::dict>();
            auto std_ptr = reinterpret_cast<const float*>(
                py::cast<uintptr_t>(std_iface["data"].cast<py::tuple>()[0]));

            singlet::gpu::preprocess::ScaleConfig cfg;
            cfg.max_value     = max_value;
            cfg.zero_center   = zero_center;
            cfg.unit_variance = unit_variance;

            auto dense = singlet::gpu::preprocess::scale(*mat.csc, mean_ptr, std_ptr, cfg, s);

            auto py_r = std::make_shared<PyDenseResult>();
            py_r->dense  = std::move(dense);
            py_r->stream = s;
            return py::cast(py_r);
        },
        py::arg("mat"),
        py::arg("mean"),
        py::arg("std"),
        py::arg("max_value")     = 10.0f,
        py::arg("zero_center")   = true,
        py::arg("unit_variance") = true,
        py::arg("stream")        = py::none(),
        R"doc(
        Z-score scale a sparse HVG matrix to a dense float32 matrix (cycle-103).

        Converts the sparse genes × cells CSC matrix to a dense (n_genes × n_cells)
        row-major matrix by applying: scaled = clip((x - mean) / std, -max, +max).

        Pre-computed per-gene mean and std must be provided as device float32 arrays
        (e.g. from QcResult.gene_mean_view + gene_var_view: std = sqrt(var)).

        Parameters
        ----------
        mat : DeviceCsc
            Typically the HVG-subsetted matrix.
        mean : cuda array (float32, n_genes)
            Per-gene mean.  Ignored (set to 0) when zero_center=False.
        std : cuda array (float32, n_genes)
            Per-gene standard deviation.  Ignored when unit_variance=False.
        max_value : float, default 10.0
            Clip scaled values to [-max_value, max_value].
        zero_center : bool, default True
            Subtract per-gene mean.
        unit_variance : bool, default True
            Divide by per-gene std.
        stream : int or None

        Returns
        -------
        DenseResult
            Holds the dense matrix; access via ``data_view`` (n_genes*n_cells flat).
        )doc");

    // ── regress_out function ──────────────────────────────────────────────
    m.def("regress_out",
        [](py::object dense_obj,
           py::object C_obj,
           int        p,
           py::object stream_obj) -> void
        {
            // Surface p > 32 as a clear Python ValueError before entering C++.
            if (p <= 0 || p > 32)
                throw py::value_error(
                    "regress_out: p must be in [1, 32]; got p=" +
                    std::to_string(p) + ".  Pass fewer covariate columns.");

            cudaStream_t s = resolve_stream_pp(stream_obj);

            // Unwrap DenseResult (mutable in-place).
            auto& py_r = dense_obj.cast<PyDenseResult&>();
            float* X = py_r.dense.data.get();
            int n_genes = py_r.n_genes();
            int n_cells = py_r.n_cells();

            // Unwrap C design matrix [n_cells × p] col-major via __cuda_array_interface__.
            auto C_iface = C_obj.attr("__cuda_array_interface__").cast<py::dict>();
            auto C_ptr = reinterpret_cast<const float*>(
                py::cast<uintptr_t>(C_iface["data"].cast<py::tuple>()[0]));

            // Validate shape: expect n_cells × p (or flat n_cells*p buffer).
            auto C_shape = C_iface["shape"].cast<py::tuple>();
            if ((int)py::len(C_shape) >= 1) {
                auto rows_c = py::cast<int>(C_shape[0]);
                if (rows_c != n_cells)
                    throw std::runtime_error(
                        "regress_out: C.shape[0]=" + std::to_string(rows_c) +
                        " != n_cells=" + std::to_string(n_cells));
            }

            singlet::gpu::preprocess::regress_out(X, n_genes, n_cells, C_ptr, p, s);
        },
        py::arg("dense"),
        py::arg("C"),
        py::arg("p"),
        py::arg("stream") = py::none(),
        R"doc(
        In-place batch OLS residualization of a dense expression matrix (cycle-103).

        Removes the linear effect of p covariates from X using QR factorization
        (cuSOLVER geqrf + cuBLAS GEMM, fully device-side).  X is modified in-place.

        Parameters
        ----------
        dense : DenseResult
            The row-major n_genes × n_cells float32 matrix returned by scale().
            Modified in place.
        C : cuda array (float32, n_cells × p, col-major)
            Design matrix of covariates (e.g. total_counts, pct_mt, batch labels).
        p : int
            Number of covariate columns.  Must be in [1, 32].
        stream : int or None

        Raises
        ------
        ValueError
            If p is outside [1, 32] (C++ hard limit for register-budget QR).

        Returns
        -------
        None  (X is modified in-place inside dense)
        )doc");
}

}  // namespace python
}  // namespace singlet::gpu
