// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_loader.hpp
//
// pybind11 bindings for singlet::gpu::io::load_pz and the PzDeviceMatrix
// result struct.
//
// Design:
//   - load_pz(path, keep_host_pinned=False) → PzDeviceMatrix Python object.
//   - PzDeviceMatrix holds a DeviceCsc (shared_ptr) + Metadata.
//   - The PzDeviceMatrix Python object is the only object the user normally
//     needs; DeviceCsc is also exposed for advanced use (direct cupy wrapping).
//
// Thread safety: pybind11 holds the GIL during all call frames here.  The
// cudaMemcpyAsync calls inside load_pz are non-blocking w.r.t. the CPU, but
// the stream is synchronised implicitly when the user first accesses device
// data via cupy (cupy inserts cudaStreamSynchronize internally when
// stream != nullptr and stream != default-stream).

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/streaming/pz_data_loader.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>

#include "_cupy_interop.hpp"
#include "_bind_metadata.hpp"

#include <memory>
#include <string>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Python DeviceCsc wrapper
//
// Wraps singlet::gpu::core::DeviceCSC via std::shared_ptr so Python's GC
// correctly tracks the device allocation lifetime.
//
// WHY shared_ptr: DeviceCSC is a non-copyable factornet type whose destructor
// calls cudaFree.  shared_ptr lets multiple Python objects (e.g. cupy views)
// hold a reference while the device memory stays alive.
// ---------------------------------------------------------------------------
struct PyDeviceCsc {
    std::shared_ptr<singlet::gpu::core::DeviceCSC> csc;
    cudaStream_t producer_stream = nullptr;  // stream the async copy was issued on

    // Lifetime anchor for from_cupy_csr: when DeviceCSC wraps a non-owning view
    // of a cupy matrix's device buffers, we store the source Python object here
    // so Python's GC cannot collect it before this PyDeviceCsc is destroyed.
    // Null for DeviceCSC objects created by load_pz (those own their memory).
    std::shared_ptr<pybind11::object> cupy_owner;

    explicit PyDeviceCsc(std::shared_ptr<singlet::gpu::core::DeviceCSC> p,
                         cudaStream_t s = nullptr)
        : csc(std::move(p)), producer_stream(s) {}
};

// ---------------------------------------------------------------------------
// bind_device_csc — registers DeviceCsc in the module.
// ---------------------------------------------------------------------------
inline void bind_device_csc(py::module_& m) {

    py::class_<PyDeviceCsc, std::shared_ptr<PyDeviceCsc>>(m, "DeviceCsc",
        R"doc(
        A CSC sparse matrix resident on the GPU device.

        Device memory is managed by factornet's RAII pool (DeviceMemory<T>)
        and is freed when the last Python reference is dropped.

        To build a zero-copy cupy view::

            import cupy.sparse as csp
            mat = csp.csr_matrix(
                (m.data_view, m.indices_view, m.indptr_view),
                shape=(m.rows, m.cols)
            )

        To get a scipy host copy::

            csr_host = m.to_host()
        )doc")

    .def_property_readonly("rows",
        [](const PyDeviceCsc& self) { return static_cast<py::ssize_t>(self.csc->rows); },
        "Number of rows (features/genes).")

    .def_property_readonly("cols",
        [](const PyDeviceCsc& self) { return static_cast<py::ssize_t>(self.csc->cols); },
        "Number of columns (cells).")

    .def_property_readonly("nnz",
        [](const PyDeviceCsc& self) { return static_cast<py::ssize_t>(self.csc->nnz); },
        "Number of non-zero entries.")

    .def_property_readonly("dtype",
        [](const PyDeviceCsc&) { return py::str("float32"); },
        "Element dtype — always float32 in singlet-gpu 0.1.x.")

    .def_property_readonly("device",
        [](const PyDeviceCsc& self) -> py::object {
            int dev = 0;
            // factornet DeviceMemory stores the device id; fall back to current.
            // The csc itself doesn't expose device directly, so we ask CUDA.
            (void)cudaGetDevice(&dev);
            return py::str("cuda:" + std::to_string(dev));
        },
        "CUDA device string, e.g. 'cuda:0'.")

    // ------------------------------------------------------------------
    // Zero-copy view properties — return a dict with __cuda_array_interface__
    // so cupy.ndarray() can wrap the device pointer without copying.
    //
    // Each property also stores a reference to `self` (the PyDeviceCsc) in
    // the returned dict under the key "_owner".  This ensures the device
    // allocation stays alive as long as any cupy view built from the dict
    // is alive (Python's GC follows the _owner reference chain).
    // ------------------------------------------------------------------
    .def_property_readonly("data_view",
        [](py::object self_obj) {
            auto& self = self_obj.cast<PyDeviceCsc&>();
            // values: float32 array of length nnz
            return make_view_object<float>(
                self.csc->values.get(),
                static_cast<std::size_t>(self.csc->nnz),
                self.producer_stream,
                self_obj
            );
        },
        "Device float32 data array (nnz,) — __cuda_array_interface__ dict.")

    .def_property_readonly("indices_view",
        [](py::object self_obj) {
            auto& self = self_obj.cast<PyDeviceCsc&>();
            // row indices: int32 array of length nnz
            return make_view_object<int32_t>(
                self.csc->row_indices.get(),
                static_cast<std::size_t>(self.csc->nnz),
                self.producer_stream,
                self_obj
            );
        },
        "Device int32 row-index array (nnz,) — __cuda_array_interface__ dict.")

    .def_property_readonly("indptr_view",
        [](py::object self_obj) {
            auto& self = self_obj.cast<PyDeviceCsc&>();
            // column pointer: int32 array of length cols+1
            return make_view_object<int32_t>(
                self.csc->col_ptr.get(),
                static_cast<std::size_t>(self.csc->cols + 1),
                self.producer_stream,
                self_obj
            );
        },
        "Device int32 column-pointer array (cols+1,) — __cuda_array_interface__ dict.")

    // ------------------------------------------------------------------
    // to_host(): explicit device → host copy returning scipy.sparse.csr_matrix.
    // This is the ONLY method that moves data to the host.  No implicit copy
    // ever occurs in the default load path.
    // ------------------------------------------------------------------
    .def("to_host",
        [](const PyDeviceCsc& self) -> py::object {
            // Sync the producer stream so the async copy has completed.
            if (self.producer_stream) {
                cudaError_t err = cudaStreamSynchronize(self.producer_stream);
                if (err != cudaSuccess)
                    throw std::runtime_error(
                        std::string("cudaStreamSynchronize: ") + cudaGetErrorString(err));
            }

            std::size_t nnz  = static_cast<std::size_t>(self.csc->nnz);
            std::size_t cols = static_cast<std::size_t>(self.csc->cols);

            // Allocate host buffers.
            std::vector<float>   h_values(nnz);
            std::vector<int32_t> h_indices(nnz);
            std::vector<int32_t> h_indptr(cols + 1);

            cudaError_t e1 = cudaMemcpy(h_values.data(),  self.csc->values.get(),
                                        nnz  * sizeof(float),   cudaMemcpyDeviceToHost);
            cudaError_t e2 = cudaMemcpy(h_indices.data(), self.csc->row_indices.get(),
                                        nnz  * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaError_t e3 = cudaMemcpy(h_indptr.data(),  self.csc->col_ptr.get(),
                                        (cols+1) * sizeof(int32_t), cudaMemcpyDeviceToHost);

            for (auto e : {e1, e2, e3}) {
                if (e != cudaSuccess)
                    throw std::runtime_error(
                        std::string("to_host cudaMemcpy: ") + cudaGetErrorString(e));
            }

            // Build scipy.sparse.csc_matrix then convert to csr for standard usage.
            py::module_ scipy_sparse = py::module_::import("scipy.sparse");
            py::object np = py::module_::import("numpy");

            auto data_arr = np.attr("array")(h_values,  py::arg("dtype") = "float32");
            auto idx_arr  = np.attr("array")(h_indices, py::arg("dtype") = "int32");
            auto ptr_arr  = np.attr("array")(h_indptr,  py::arg("dtype") = "int32");

            py::object csc = scipy_sparse.attr("csc_matrix")(
                py::make_tuple(data_arr, idx_arr, ptr_arr),
                py::arg("shape") = py::make_tuple(
                    static_cast<py::ssize_t>(self.csc->rows),
                    static_cast<py::ssize_t>(cols)
                )
            );
            return csc.attr("tocsr")();
        },
        R"doc(
        Synchronise the producer stream and copy the device matrix to host.

        Returns
        -------
        scipy.sparse.csr_matrix
            Host sparse matrix, float32.  This is the ONLY method that copies
            data from device to host — the default load path is zero-copy.
        )doc")

    .def("__repr__",
        [](const PyDeviceCsc& self) {
            return "<DeviceCsc rows=" + std::to_string(self.csc->rows) +
                   " cols=" + std::to_string(self.csc->cols) +
                   " nnz=" + std::to_string(self.csc->nnz) +
                   " dtype=float32"
                   " — call .to_host() for a numpy/scipy view>";
        });
}

// ---------------------------------------------------------------------------
// PyPzDeviceMatrix — Python-visible result of load_pz().
//
// Combines a shared_ptr<PyDeviceCsc> (so cupy views can outlive this object)
// with a Metadata value.
// ---------------------------------------------------------------------------
struct PyPzDeviceMatrix {
    std::shared_ptr<PyDeviceCsc> device_csc;
    singlet::gpu::core::Metadata  meta;
};

// bind_pz_device_matrix — registers PzDeviceMatrix and load_pz in the module.
inline void bind_pz_device_matrix(py::module_& m) {

    py::class_<PyPzDeviceMatrix, std::shared_ptr<PyPzDeviceMatrix>>(m, "PzDeviceMatrix",
        R"doc(
        Result of :func:`load_pz`: a device CSC matrix plus embedded metadata.

        Attributes
        ----------
        mat : DeviceCsc
            The sparse matrix on the GPU.
        meta : Metadata
            GEO metadata and provenance from the .1pz TLV block.
        )doc")

    .def_property_readonly("mat",
        [](const PyPzDeviceMatrix& self) -> std::shared_ptr<PyDeviceCsc> {
            return self.device_csc;
        },
        "The GPU CSC sparse matrix.")

    .def_property_readonly("meta",
        [](const PyPzDeviceMatrix& self) -> const singlet::gpu::core::Metadata& {
            return self.meta;
        },
        py::return_value_policy::reference_internal,
        "Embedded GEO metadata.")

    // Convenience pass-throughs so users can write m.rows instead of m.mat.rows.
    .def_property_readonly("rows",
        [](const PyPzDeviceMatrix& s) { return s.device_csc->csc->rows; })
    .def_property_readonly("cols",
        [](const PyPzDeviceMatrix& s) { return s.device_csc->csc->cols; })
    .def_property_readonly("nnz",
        [](const PyPzDeviceMatrix& s) { return s.device_csc->csc->nnz; })

    .def("__repr__",
        [](const PyPzDeviceMatrix& self) {
            return "<PzDeviceMatrix gsm='" + self.meta.gsm_id + "'"
                   " rows=" + std::to_string(self.device_csc->csc->rows) +
                   " cols=" + std::to_string(self.device_csc->csc->cols) +
                   " nnz="  + std::to_string(self.device_csc->csc->nnz) + ">";
        });

    // ------------------------------------------------------------------
    // load_pz — module-level function, registered here for locality.
    //
    // WHY no stream argument in the Python API: cupy users don't manage
    // CUDA streams directly.  We create a high-priority stream inside the
    // C++ call, store it in PzDeviceMatrix, and expose it implicitly via
    // __cuda_array_interface__'s `stream` key.  Advanced users who need
    // stream control use the C++ API directly.
    // ------------------------------------------------------------------
    m.def("load_pz",
        [](const std::string& path, bool keep_host_pinned) -> std::shared_ptr<PyPzDeviceMatrix> {

            // Release GIL during file I/O + decompression (pure C++, thread-safe).
            py::gil_scoped_release release;

            singlet::gpu::io::PzDeviceMatrix result =
                singlet::gpu::io::load_pz(path, /*stream=*/nullptr, keep_host_pinned);

            // Wrap DeviceCSC in a shared_ptr so cupy views can share ownership.
            auto dev_csc = std::make_shared<PyDeviceCsc>(
                std::make_shared<singlet::gpu::core::DeviceCSC>(std::move(result.mat)),
                result.producer_stream
            );

            auto out = std::make_shared<PyPzDeviceMatrix>();
            out->device_csc = std::move(dev_csc);
            out->meta        = std::move(result.meta);
            return out;
        },
        py::arg("path"),
        py::arg("keep_host_pinned") = false,
        R"doc(
        Load a .1pz file from *path* and return a :class:`PzDeviceMatrix`.

        The .1pz file is decompressed on the CPU (GIL released during I/O).
        Three async ``cudaMemcpyAsync`` calls stage the CSC arrays onto the
        device.  No host data is retained after the copies are in flight unless
        *keep_host_pinned* is ``True``.

        Parameters
        ----------
        path : str
            Absolute path to a ``.1pz`` file.
        keep_host_pinned : bool, default False
            When ``True``, retains pinned host copies of ``indptr``,
            ``indices``, and ``values`` inside the returned object.  Useful
            for SVD adapters that require host pointers.  Doubles memory use.

        Returns
        -------
        PzDeviceMatrix
            Device matrix + embedded metadata.  Device memory is freed when
            the last Python reference to this object (or any cupy view derived
            from it) is garbage-collected.
        )doc");
}

// CYCLE-275: PzDataLoader binding — exposes streaming/pz_data_loader.h's
// `singlet::gpu::io::PzDataLoader` so Python can construct one and pass it to
// `_core.nmf_chunked` (and future streaming kernels).  Single-path ctor only;
// multi-file streaming is filed as CYCLE-7-MULTI-INPUT-NMF in followups.
inline void bind_pz_data_loader(py::module_& m) {
    py::class_<singlet::gpu::io::PzDataLoader,
               std::shared_ptr<singlet::gpu::io::PzDataLoader>>(m, "PzDataLoader",
        "Streaming loader for a single .1pz dataset (forward + transpose chunks).")
    .def(py::init<const std::string&, uint32_t>(),
         py::arg("path"),
         py::arg("chunk_cols") = singlet::gpu::io::PzDataLoader::DEFAULT_CHUNK_COLS)
    .def_property_readonly("rows",
        [](const singlet::gpu::io::PzDataLoader& self) { return self.rows(); },
        "Number of rows (genes).")
    .def_property_readonly("cols",
        [](const singlet::gpu::io::PzDataLoader& self) { return self.cols(); },
        "Number of cols (cells).")
    .def_property_readonly("nnz",
        [](const singlet::gpu::io::PzDataLoader& self) { return self.nnz(); },
        "Total non-zero entries.")
    .def_property_readonly("path",
        [](const singlet::gpu::io::PzDataLoader& self) { return self.path(); },
        "Source .1pz path.")
    .def_property_readonly("num_forward_chunks",
        [](const singlet::gpu::io::PzDataLoader& self) { return self.num_forward_chunks(); })
    .def_property_readonly("num_transpose_chunks",
        [](const singlet::gpu::io::PzDataLoader& self) { return self.num_transpose_chunks(); })
    .def("__repr__",
        [](const singlet::gpu::io::PzDataLoader& self) {
            return "<PzDataLoader rows=" + std::to_string(self.rows())
                 + " cols=" + std::to_string(self.cols())
                 + " nnz=" + std::to_string(self.nnz()) + ">";
        });
}
