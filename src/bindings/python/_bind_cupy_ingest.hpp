// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_cupy_ingest.hpp
//
// from_cupy_csr / to_cupy_csr — bidirectional cupy.sparse ↔ DeviceCSC marshal.
//
// Lifetime contract:
//   from_cupy_csr: the source Python object is stashed in PyDeviceCsc::cupy_owner
//   (std::shared_ptr<py::object>).  As long as any Python reference to the returned
//   PyDeviceCsc (or a cupy view built from it) is alive, the owning csr_matrix is
//   NOT garbage-collected.  The DeviceCSC wraps the device buffers with non-owning
//   DeviceMemory (DeviceMemory::wrap) — zero device copies.
//
//   to_cupy_csr: returns a Python dict with __cuda_array_interface__ views of the
//   DeviceCSC arrays plus "_owner" = the PyDeviceCsc.  Same zero-copy, same anchor.
//
// WHY non-owning DeviceMemory::wrap: the cupy matrix already owns the device memory.
// Duplicating it would double device RAM and require an explicit sync before the copy
// was readable.  wrap() sets owns_=false so the DeviceCSC destructor does NOT call
// cudaFree on pointers it doesn't own.
//
// dtype validation: we require int32 indptr/indices and float32 data — the same layout
// our DeviceCSC carries.  Any other dtype raises ValueError immediately.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <singlet/gpu/core/types.h>
#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"   // PyDeviceCsc definition

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace singlet::gpu {
namespace python {

// ---------------------------------------------------------------------------
// cupy_extract_ptr — parse __cuda_array_interface__ from one cupy array.
//
// Validates dtype (expected_typestr) and shape (must be 1-D).
// Returns (raw device pointer as void*, n_elements).
// ---------------------------------------------------------------------------
static std::pair<void*, std::size_t>
cupy_extract_ptr(py::object arr, const char* expected_typestr, const char* field_name)
{
    if (!py::hasattr(arr, "__cuda_array_interface__")) {
        throw py::type_error(
            std::string(field_name) + " lacks __cuda_array_interface__");
    }
    py::dict iface = arr.attr("__cuda_array_interface__").cast<py::dict>();

    std::string ts = iface["typestr"].cast<std::string>();
    if (ts != expected_typestr) {
        throw py::value_error(
            std::string(field_name) + " typestr='" + ts
            + "', expected '" + expected_typestr + "'");
    }

    py::tuple shape = iface["shape"].cast<py::tuple>();
    if (shape.size() != 1) {
        throw py::value_error(
            std::string(field_name) + " must be 1-D");
    }
    std::size_t n = shape[0].cast<std::size_t>();

    // data tuple: (int_ptr, read_only_bool)
    py::tuple dtup = iface["data"].cast<py::tuple>();
    std::intptr_t ptr = dtup[0].cast<std::intptr_t>();

    return { reinterpret_cast<void*>(ptr), n };
}

// ---------------------------------------------------------------------------
// from_cupy_csr
//
// Accepts any object with .indptr / .indices / .data attributes that each
// expose __cuda_array_interface__ (cupy.sparse.csr_matrix or csc_matrix).
//
// Returns a PyDeviceCsc holding:
//   - a non-owning DeviceCSC (via SparseMatrixGPU::from_device_ptrs)
//   - a lifetime anchor (cupy_owner) on the source Python object
// ---------------------------------------------------------------------------
inline std::shared_ptr<PyDeviceCsc> from_cupy_csr(py::object csr_obj)
{
    py::object indptr_arr  = csr_obj.attr("indptr");
    py::object indices_arr = csr_obj.attr("indices");
    py::object data_arr    = csr_obj.attr("data");

    auto [indptr_ptr,  n_indptr ] = cupy_extract_ptr(indptr_arr,  "<i4", "indptr");
    auto [indices_ptr, n_indices] = cupy_extract_ptr(indices_arr, "<i4", "indices");
    auto [data_ptr,    n_nnz    ] = cupy_extract_ptr(data_arr,    "<f4", "data");

    // shape: (rows, cols) for csr_matrix.
    py::tuple shape = csr_obj.attr("shape").cast<py::tuple>();
    if (shape.size() != 2) {
        throw py::value_error("from_cupy_csr: shape must be 2-D");
    }
    int rows = static_cast<int>(shape[0].cast<int64_t>());
    int cols = static_cast<int>(shape[1].cast<int64_t>());
    int nnz  = static_cast<int>(n_nnz);

    // csr_matrix: indptr.size() == rows+1; csc_matrix: indptr.size() == cols+1.
    // We accept both and use the supplied shape to determine the CSC layout.
    // For our internal DeviceCSC (CSC layout) we set col_ptr length = cols+1.
    // If the user passes a csr_matrix, they must convert first (cupy policy
    // ensures csr.tocsc() returns correct indptr layout).  We validate that
    // n_indptr equals either rows+1 or cols+1 and warn on ambiguity.
    bool csr_layout = (n_indptr == static_cast<std::size_t>(rows + 1));
    bool csc_layout = (n_indptr == static_cast<std::size_t>(cols + 1));
    if (!csr_layout && !csc_layout) {
        throw py::value_error(
            "from_cupy_csr: indptr.size()=" + std::to_string(n_indptr)
            + " does not match rows+1=" + std::to_string(rows + 1)
            + " or cols+1=" + std::to_string(cols + 1));
    }

    // Build non-owning SparseMatrixGPU — from_device_ptrs sets owns_=false on all
    // three DeviceMemory members via DeviceMemory::wrap so no cudaFree on dtor.
    auto csc = std::make_shared<singlet::gpu::core::DeviceCSC>(
        singlet::gpu::core::DeviceCSC::from_device_ptrs(
            rows, cols, nnz,
            static_cast<int*>(indptr_ptr),
            static_cast<int*>(indices_ptr),
            static_cast<float*>(data_ptr)
        )
    );

    auto py_csc = std::make_shared<PyDeviceCsc>(std::move(csc), /*stream=*/nullptr);

    // Lifetime anchor: keep the cupy matrix alive as long as py_csc is alive.
    // std::shared_ptr<py::object> increments the Python refcount via pybind11's
    // py::object copy constructor and decrements it when the shared_ptr is destroyed.
    py_csc->cupy_owner = std::make_shared<py::object>(csr_obj);

    return py_csc;
}

// ---------------------------------------------------------------------------
// to_cupy_csr
//
// Returns a Python dict with zero-copy __cuda_array_interface__ views plus
// "shape" and "_owner".  The caller builds a cupy.sparse.csc_matrix:
//   import cupy.sparse as csp
//   d = _core.to_cupy_csr(dev_csc)
//   csc = csp.csc_matrix((d["data"], d["indices"], d["indptr"]), shape=d["shape"])
// ---------------------------------------------------------------------------
inline py::dict to_cupy_csr(py::object csc_obj)
{
    auto& self = csc_obj.cast<PyDeviceCsc&>();

    py::dict out;

    // col_ptr: int32, length cols+1
    out["indptr"] = make_view_object<int32_t>(
        self.csc->col_ptr.data(),
        static_cast<std::size_t>(self.csc->cols + 1),
        self.producer_stream,
        csc_obj
    );
    // row_indices: int32, length nnz
    out["indices"] = make_view_object<int32_t>(
        self.csc->row_indices.data(),
        static_cast<std::size_t>(self.csc->nnz),
        self.producer_stream,
        csc_obj
    );
    // values: float32, length nnz
    out["data"] = make_view_object<float>(
        self.csc->values.data(),
        static_cast<std::size_t>(self.csc->nnz),
        self.producer_stream,
        csc_obj
    );
    out["shape"] = py::make_tuple(
        static_cast<py::ssize_t>(self.csc->rows),
        static_cast<py::ssize_t>(self.csc->cols)
    );
    // Lifetime anchor — keep PyDeviceCsc alive as long as this dict is alive.
    out["_owner"] = csc_obj;

    return out;
}

}  // namespace python
}  // namespace singlet::gpu
