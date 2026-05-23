// SPDX-License-Identifier: MIT
/**
 * @file _pz.cpp
 * @brief pybind11 bindings for the singlet .1pz (TP1Z / VOCSC) codec.
 *
 * Exposes three functions to Python:
 *   read_1pz(path)  -> dict with CSC arrays + metadata
 *   write_1pz(...)  -> bool
 *   info_1pz(path)  -> dict with header fields only (no decompression)
 *
 * Build via:  pip install .   (setup.py wires this via Pybind11Extension)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <singlet/pileup/pz_reader.h>  // also pulls in pz_writer.h

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>

namespace py = pybind11;

// ============================================================================
// read_1pz: path -> dict
// ============================================================================
static py::dict py_read_1pz(const std::string& path)
{
    singlet::pz::ReadResult r;
    try {
        r = singlet::pz::read_1pz(path);
    } catch (const singlet::pz::PZReadError& e) {
        throw std::runtime_error(e.what());
    }

    // indptr: uint32 -> int32 (values fit: nnz <= UINT32_MAX and callers expect int32)
    py::array_t<int32_t> indptr(static_cast<py::ssize_t>(r.n + 1));
    {
        auto buf = indptr.mutable_unchecked<1>();
        for (uint32_t j = 0; j <= r.n; ++j)
            buf(j) = static_cast<int32_t>(r.indptr[j]);
    }

    // indices: uint32 -> int32
    py::array_t<int32_t> indices(static_cast<py::ssize_t>(r.nnz));
    {
        auto buf = indices.mutable_unchecked<1>();
        for (uint64_t k = 0; k < r.nnz; ++k)
            buf(k) = static_cast<int32_t>(r.indices[k]);
    }

    // data: keep as uint32
    py::array_t<uint32_t> data(static_cast<py::ssize_t>(r.nnz));
    {
        auto buf = data.mutable_unchecked<1>();
        for (uint64_t k = 0; k < r.nnz; ++k)
            buf(k) = r.data[k];
    }

    py::dict result;
    result["m"]        = static_cast<int64_t>(r.m);
    result["n"]        = static_cast<int64_t>(r.n);
    result["nnz"]      = static_cast<int64_t>(r.nnz);
    result["vt_code"]  = static_cast<int>(r.vt_code);
    result["indptr"]   = indptr;
    result["indices"]  = indices;
    result["data"]     = data;
    result["rownames"] = r.rownames;
    result["colnames"] = r.colnames;
    result["user_kv"]  = r.user_kv;
    return result;
}

// ============================================================================
// write_1pz — template dispatcher called once the ValT is resolved
// ============================================================================
template<typename ValT>
static bool dispatch_write(
    const std::string& path,
    const std::vector<int32_t>& indptr_vec,
    const std::vector<int32_t>& indices_vec,
    const std::vector<ValT>& data_vec,
    uint32_t m, uint32_t n,
    const std::vector<std::string>& rownames_vec,
    const std::vector<std::string>& colnames_vec,
    int zstd_level, int chunk_cols, int threads,
    const std::map<std::string, std::string>& user_meta_map)
{
    return singlet::pz::write_1pz<ValT>(
        path, m, n,
        indptr_vec, indices_vec, data_vec,
        rownames_vec, colnames_vec,
        zstd_level, chunk_cols, threads,
        user_meta_map);
}

// ============================================================================
// py_write_1pz
// ============================================================================
static bool py_write_1pz(
    const std::string& path,
    py::array indptr_arr,
    py::array indices_arr,
    py::array data_arr,
    int m,
    int n,
    py::object rownames_obj,
    py::object colnames_obj,
    int zstd_level,
    int chunk_cols,
    int threads,
    py::object user_meta_obj)
{
    // Force-cast indptr/indices to int32
    auto indptr_i32  = py::array_t<int32_t,
        py::array::c_style | py::array::forcecast>(indptr_arr);
    auto indices_i32 = py::array_t<int32_t,
        py::array::c_style | py::array::forcecast>(indices_arr);

    auto p_buf = indptr_i32.request();
    auto i_buf = indices_i32.request();
    if (p_buf.ndim != 1 || i_buf.ndim != 1)
        throw std::runtime_error("indptr and indices must be 1-D arrays");

    std::vector<int32_t> indptr_vec(
        static_cast<int32_t*>(p_buf.ptr),
        static_cast<int32_t*>(p_buf.ptr) + p_buf.shape[0]);
    std::vector<int32_t> indices_vec(
        static_cast<int32_t*>(i_buf.ptr),
        static_cast<int32_t*>(i_buf.ptr) + i_buf.shape[0]);

    // rownames / colnames
    std::vector<std::string> rownames_vec, colnames_vec;
    if (!rownames_obj.is_none())
        rownames_vec = rownames_obj.cast<std::vector<std::string>>();
    if (!colnames_obj.is_none())
        colnames_vec = colnames_obj.cast<std::vector<std::string>>();

    // user_meta
    std::map<std::string, std::string> user_meta_map;
    if (!user_meta_obj.is_none())
        user_meta_map = user_meta_obj.cast<std::map<std::string, std::string>>();

    // Inspect dtype of data
    auto d_buf = data_arr.request();
    if (d_buf.ndim != 1)
        throw std::runtime_error("data must be a 1-D array");

    const char kind    = data_arr.dtype().kind();
    const int  itemsz  = static_cast<int>(data_arr.dtype().itemsize());
    const py::ssize_t nnz = d_buf.shape[0];

    if (kind == 'f')
        throw py::value_error("write_1pz does not accept float data; "
                              "pass integer counts (uint8/uint16/uint32 or int32)");

    if (kind == 'u') {
        // Unsigned integer: dispatch on width
        if (itemsz == 1) {
            auto arr = py::array_t<uint8_t,
                py::array::c_style | py::array::forcecast>(data_arr);
            auto buf = arr.request();
            std::vector<uint8_t> dv(static_cast<uint8_t*>(buf.ptr),
                                    static_cast<uint8_t*>(buf.ptr) + nnz);
            return dispatch_write(path, indptr_vec, indices_vec, dv,
                                  static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                                  rownames_vec, colnames_vec,
                                  zstd_level, chunk_cols, threads, user_meta_map);
        } else if (itemsz == 2) {
            auto arr = py::array_t<uint16_t,
                py::array::c_style | py::array::forcecast>(data_arr);
            auto buf = arr.request();
            std::vector<uint16_t> dv(static_cast<uint16_t*>(buf.ptr),
                                     static_cast<uint16_t*>(buf.ptr) + nnz);
            return dispatch_write(path, indptr_vec, indices_vec, dv,
                                  static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                                  rownames_vec, colnames_vec,
                                  zstd_level, chunk_cols, threads, user_meta_map);
        } else {
            // uint32 or uint64 → cast to uint32
            auto arr = py::array_t<uint32_t,
                py::array::c_style | py::array::forcecast>(data_arr);
            auto buf = arr.request();
            std::vector<uint32_t> dv(static_cast<uint32_t*>(buf.ptr),
                                     static_cast<uint32_t*>(buf.ptr) + nnz);
            return dispatch_write(path, indptr_vec, indices_vec, dv,
                                  static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                                  rownames_vec, colnames_vec,
                                  zstd_level, chunk_cols, threads, user_meta_map);
        }
    } else if (kind == 'i' || kind == 'u') {
        // Signed integer: cast to uint32, require non-negative
        auto arr = py::array_t<int64_t,
            py::array::c_style | py::array::forcecast>(data_arr);
        auto buf = arr.request();
        const int64_t* src = static_cast<const int64_t*>(buf.ptr);
        std::vector<uint32_t> dv(static_cast<size_t>(nnz));
        for (py::ssize_t k = 0; k < nnz; ++k) {
            if (src[k] < 0)
                throw py::value_error("write_1pz: data contains negative values");
            dv[k] = static_cast<uint32_t>(src[k]);
        }
        return dispatch_write(path, indptr_vec, indices_vec, dv,
                              static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                              rownames_vec, colnames_vec,
                              zstd_level, chunk_cols, threads, user_meta_map);
    } else {
        // signed int dtypes ('i')
        auto arr = py::array_t<int32_t,
            py::array::c_style | py::array::forcecast>(data_arr);
        auto buf = arr.request();
        const int32_t* src = static_cast<const int32_t*>(buf.ptr);
        std::vector<uint32_t> dv(static_cast<size_t>(nnz));
        for (py::ssize_t k = 0; k < nnz; ++k) {
            if (src[k] < 0)
                throw py::value_error("write_1pz: data contains negative values");
            dv[k] = static_cast<uint32_t>(src[k]);
        }
        return dispatch_write(path, indptr_vec, indices_vec, dv,
                              static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                              rownames_vec, colnames_vec,
                              zstd_level, chunk_cols, threads, user_meta_map);
    }
}

// ============================================================================
// info_1pz: path -> dict  (reads only the first 96-byte header)
// ============================================================================
static py::dict py_info_1pz(const std::string& path)
{
    std::ifstream fin(path, std::ios::binary);
    if (!fin)
        throw std::runtime_error("info_1pz: cannot open " + path);

    singlet::pz::PZHeader hdr{};
    fin.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!fin)
        throw std::runtime_error("info_1pz: file too short or read error: " + path);
    fin.close();

    if (hdr.magic != singlet::pz::TP1_MAGIC)
        throw std::runtime_error("info_1pz: bad magic (not a .1pz file): " + path);
    if (hdr.version != 1 && hdr.version != 3 && hdr.version != 4)
        throw std::runtime_error("info_1pz: unsupported version " +
                                 std::to_string(hdr.version) + ": " + path);

    py::dict result;
    result["magic"]       = std::string("TP1Z");
    result["version"]     = static_cast<int>(hdr.version);
    result["m"]           = static_cast<int64_t>(hdr.m);
    result["n"]           = static_cast<int64_t>(hdr.n);
    result["nnz"]         = static_cast<int64_t>(hdr.nnz);
    result["vt_code"]     = static_cast<int>(hdr.vt_code);
    result["num_chunks"]  = static_cast<int64_t>(hdr.num_chunks);
    result["chunk_cols"]  = static_cast<int64_t>(hdr.chunk_cols);
    result["flags"]       = static_cast<int>(hdr.flags);
    result["ptr_width"]   = static_cast<int>(hdr.ptr_width);
    result["codec_level"] = static_cast<int>(hdr.codec_level);
    return result;
}

// ============================================================================
// Module definition
// ============================================================================
PYBIND11_MODULE(_pz, m) {
    m.doc() = "singlet._pz: pybind11 bindings for the .1pz (TP1Z/VOCSC) codec.\n"
              "Functions: read_1pz, write_1pz, info_1pz";

    m.def("read_1pz", &py_read_1pz,
          py::arg("path"),
          "Read a .1pz file.\n"
          "Returns dict with keys: m, n, nnz, vt_code, indptr (int32), "
          "indices (int32), data (uint32), rownames, colnames, user_kv.");

    m.def("write_1pz", &py_write_1pz,
          py::arg("path"),
          py::arg("indptr"),
          py::arg("indices"),
          py::arg("data"),
          py::arg("m"),
          py::arg("n"),
          py::kw_only(),
          py::arg("rownames")    = py::none(),
          py::arg("colnames")    = py::none(),
          py::arg("zstd_level")  = 3,
          py::arg("chunk_cols")  = 1024,
          py::arg("threads")     = 4,
          py::arg("user_meta")   = py::none(),
          "Write a CSC sparse matrix to a .1pz file.\n"
          "data must be an integer dtype (uint8/uint16/uint32 or int32/int64).\n"
          "Returns True on success.");

    m.def("info_1pz", &py_info_1pz,
          py::arg("path"),
          "Read header fields from a .1pz file without decompression.\n"
          "Returns dict with: magic, version, m, n, nnz, vt_code, "
          "num_chunks, chunk_cols, flags, ptr_width, codec_level.");
}
