// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/python/src/_cupy_interop.hpp
//
// __cuda_array_interface__ v3 protocol helpers.
//
// Returns a Python dict describing a device pointer so that cupy, numba, and
// pytorch can construct zero-copy views without a host round-trip.
//
// Protocol spec: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
//   version:  3  (int)
//   shape:    (N,)  tuple of ints
//   typestr:  dtype string, e.g. "<f4", "<i4"
//   data:     (int_pointer, read_only_bool)  tuple
//   stream:   int CUDA stream handle (or None for default stream)
//   strides:  None for C-contiguous
//
// WHY this approach: pybind11 does not ship __cuda_array_interface__ built-in.
// We implement it as a Python @property on each binding that returns a fresh
// dict every time.  cupy recognizes the dict and wraps the device pointer
// in a cupy.ndarray WITHOUT copying data.
//
// Lifetime note: the Python DeviceCsc object that owns the shared_ptr<DeviceCSC>
// must stay alive as long as any cupy view built from these pointers is in use.
// In io.py we achieve this by attaching the DeviceCsc as a Python attribute on
// every cupy view, which causes Python's refcount to keep it alive.

#pragma once

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// typestr helper — maps a C++ type to numpy typestr "<Xn".
// Only the types that appear in singlet-gpu CSC arrays are needed.
// ---------------------------------------------------------------------------
template<typename T>
inline const char* cuda_typestr() {
    if constexpr (std::is_same_v<T, float>)    return "<f4";
    if constexpr (std::is_same_v<T, int32_t>)  return "<i4";
    if constexpr (std::is_same_v<T, uint32_t>) return "<u4";
    if constexpr (std::is_same_v<T, int64_t>)  return "<i8";
    if constexpr (std::is_same_v<T, uint8_t>)  return "|u1";   // cycle 20: QC mask
    throw std::runtime_error("cuda_typestr: unsupported type");
}

// ---------------------------------------------------------------------------
// make_cuda_array_interface
//
// device_ptr: raw CUDA device pointer (cast to void*).
// n_elements: number of T elements.
// stream:     CUDA stream the data was last written on.
//             Passed as int to the dict (cupy uses it for stream ordering).
//             Use 0 for default stream / synchronize-before-use.
// read_only:  whether to tell cupy this buffer is read-only.
//
// Returns: a Python dict compatible with __cuda_array_interface__ v3.
// ---------------------------------------------------------------------------
template<typename T>
inline py::dict make_cuda_array_interface(
    const T* device_ptr,
    std::size_t n_elements,
    cudaStream_t stream = nullptr,
    bool read_only = false)
{
    py::dict iface;

    // version
    iface["version"] = 3;

    // shape: 1-D array of n_elements
    iface["shape"] = py::make_tuple(static_cast<py::ssize_t>(n_elements));

    // typestr: numpy-style dtype descriptor
    iface["typestr"] = cuda_typestr<T>();

    // data: (pointer_as_int, read_only_flag)
    // The pointer must be cast to intptr_t — cupy expects a Python int.
    iface["data"] = py::make_tuple(
        reinterpret_cast<std::intptr_t>(device_ptr),
        read_only
    );

    // stream: cupy v13+ uses this to insert a dependency on the producer stream
    // before consuming the data. 0 = legacy default stream (sync on all).
    // We pass the stream handle as a Python int (or None if null, meaning the
    // caller should sync manually).
    if (stream != nullptr) {
        iface["stream"] = reinterpret_cast<std::intptr_t>(stream);
    } else {
        iface["stream"] = py::int_(1);  // 1 = cudaStreamLegacy (PTDS default)
    }

    // strides: None → C-contiguous (1-D flat array, stride = sizeof(T))
    iface["strides"] = py::none();

    return iface;
}

// ---------------------------------------------------------------------------
// cuda_array_interface_property
//
// Helper that, given a device pointer + size + stream, builds the interface
// dict and attaches it as the __cuda_array_interface__ attribute on a Python
// object.  Called inside @property definitions in the bindings.
// ---------------------------------------------------------------------------
template<typename T>
inline py::object make_view_object(
    const T* device_ptr,
    std::size_t n_elements,
    cudaStream_t stream,
    py::object owner)           // the DeviceCsc Python object — kept alive
{
    // CYCLE-225 (CYCLE-193-FOLLOWUP): return a `types.SimpleNamespace` with
    // `__cuda_array_interface__` exposed as an OBJECT ATTRIBUTE (not a dict
    // key) plus `_owner` holding the source PyDeviceCsc.
    //
    // WHY this matters:
    //  1. cupy >= 14 is dtype-strict: `cp.asarray(bare_cai_dict)` raises
    //     "Unsupported dtype object".  An object with `__cuda_array_interface__`
    //     as an attribute IS accepted (the protocol's intended form).
    //     Eliminates the per-call-site `_CaiView` shim used 7+ times across
    //     this session (CYCLE-189/193/194/195/202/222 and downstream).
    //  2. Lifetime anchoring: cupy.asarray() sets the resulting ndarray's
    //     `.base` to the source object — so as long as the cupy view is alive,
    //     the SimpleNamespace stays alive, which holds `_owner = py_csc`,
    //     keeping the PyDeviceCsc alive.  Resolves CYCLE-193-FOLLOWUP /
    //     test_lifetime_safety AssertionError.
    py::dict iface = make_cuda_array_interface<T>(device_ptr, n_elements, stream);

    py::object types_mod = py::module_::import("types");
    py::object ns = types_mod.attr("SimpleNamespace")();
    py::setattr(ns, "__cuda_array_interface__", iface);
    py::setattr(ns, "_owner", owner);   // anchor source DeviceCsc

    return ns;
}
