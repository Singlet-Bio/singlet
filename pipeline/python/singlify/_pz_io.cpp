// singlify._pz_io — thin pybind11 wrapper around singlet::pz::read_1pz.
//
// Architecture: the entire .1pz decode pipeline lives in C++
// (include/singlet-pileup/pz_reader.h, reusing helpers from pz_writer.h).
// This module's only job is to:
//
//   1. Accept a path string from Python.
//   2. Call read_1pz().
//   3. Copy the resulting indptr/indices/data/rownames/colnames/user_kv into
//      numpy arrays and Python containers.
//
// Zero "meat" in Python — all format parsing, decompression, bit-plane /
// bitmap / byte-split inverse transforms, permutation unmap, varint column
// decoding, and per-column re-sorting happen in C++.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <stdexcept>
#include <string>

// pz_reader.h transitively includes pz_writer.h. The writer's
// write_1pz_csc convenience wrapper references SparseAccumulator<ValT>,
// for which pz_writer.h ships a forward declaration so this binding does
// not need to pull in the full sparse_accumulator.h header.
#include "singlet-pileup/pz_reader.h"

namespace py = pybind11;
namespace pz = singlet::pz;

// Convert a ReadResult into a Python dict that scipy.sparse.csc_matrix can
// consume directly:
//
//     d = singlify._pz_io.read_1pz("gene_counts.1pz")
//     import scipy.sparse as sp
//     mat = sp.csc_matrix((d["data"], d["indices"], d["indptr"]), shape=d["shape"])
//     rownames = d["rownames"]
//     colnames = d["colnames"]
//     user_kv = d["user_kv"]
//
static py::dict read_1pz_py(const std::string& path, bool verify_crc) {
    pz::ReadResult r;
    try {
        (void)verify_crc;  // CRC is always verified inside slurp_and_validate
        r = pz::read_1pz(path);
    } catch (const pz::PZReadError& e) {
        throw py::value_error(std::string("pz_reader: ") + e.what());
    } catch (const std::exception& e) {
        throw py::value_error(std::string("pz_reader: ") + e.what());
    }

    py::dict out;
    out["shape"] = py::make_tuple(r.m, r.n);
    out["nnz"] = r.nnz;
    out["vt_code"] = r.vt_code;

    // indptr: length n+1, int64 (scipy's CSC accepts int32 or int64; we use
    // int64 to avoid overflow on very large matrices, cheap for this size).
    {
        py::array_t<int64_t> arr(static_cast<py::ssize_t>(r.indptr.size()));
        auto v = arr.mutable_unchecked<1>();
        for (size_t i = 0; i < r.indptr.size(); ++i) {
            v(i) = static_cast<int64_t>(r.indptr[i]);
        }
        out["indptr"] = arr;
    }

    // indices: length nnz, int64 for scipy compatibility
    {
        py::array_t<int64_t> arr(static_cast<py::ssize_t>(r.indices.size()));
        auto v = arr.mutable_unchecked<1>();
        for (size_t i = 0; i < r.indices.size(); ++i) {
            v(i) = static_cast<int64_t>(r.indices[i]);
        }
        out["indices"] = arr;
    }

    // data: length nnz. ReadResult always stores uint32 internally, but we
    // cast to the narrowest type the writer's vt_code hint certifies,
    // matching the writer's lossless-narrow encoding policy. This keeps
    // the AnnData / scipy matrix memory-efficient on disk and in RAM.
    if (r.vt_code == 1) {
        py::array_t<uint8_t> arr(static_cast<py::ssize_t>(r.data.size()));
        auto v = arr.mutable_unchecked<1>();
        for (size_t i = 0; i < r.data.size(); ++i) {
            v(i) = static_cast<uint8_t>(r.data[i]);
        }
        out["data"] = arr;
    } else if (r.vt_code == 2) {
        py::array_t<uint16_t> arr(static_cast<py::ssize_t>(r.data.size()));
        auto v = arr.mutable_unchecked<1>();
        for (size_t i = 0; i < r.data.size(); ++i) {
            v(i) = static_cast<uint16_t>(r.data[i]);
        }
        out["data"] = arr;
    } else {
        // vt_code == 3 (uint32) or any future code — use uint32
        py::array_t<uint32_t> arr(static_cast<py::ssize_t>(r.data.size()));
        auto v = arr.mutable_unchecked<1>();
        for (size_t i = 0; i < r.data.size(); ++i) {
            v(i) = r.data[i];
        }
        out["data"] = arr;
    }

    // String lists
    out["rownames"] = r.rownames;
    out["colnames"] = r.colnames;

    // User key-value metadata (GEO context from the singlify pipeline)
    py::dict kv;
    for (const auto& p : r.user_kv) {
        kv[py::str(p.first)] = py::str(p.second);
    }
    out["user_kv"] = kv;

    return out;
}

PYBIND11_MODULE(_pz_io, m) {
    m.doc() =
        "Thin pybind11 wrapper around singlet::pz::read_1pz. Decodes .1pz "
        "pipeline outputs into Python dicts containing numpy arrays "
        "(indptr / indices / data), string lists (rownames, colnames), and "
        "a user_kv dict. All format parsing and VOCSC decompression happens "
        "in the C++ reader (include/singlet-pileup/pz_reader.h).";

    m.def("read_1pz", &read_1pz_py,
          py::arg("path"),
          py::arg("verify_crc") = true,
          R"pbdoc(
Read a .1pz file and return its contents as a dict.

The dict contains:
    shape      : (m, n) tuple
    nnz        : total non-zeros (int)
    vt_code    : writer's value-type hint (1=uint8, 2=uint16, 3=uint32)
    indptr     : int64 numpy array of length n+1
    indices    : int64 numpy array of length nnz
    data       : numpy array of length nnz, dtype matching vt_code
    rownames   : list[str] of feature names
    colnames   : list[str] of cell barcodes
    user_kv    : dict[str, str] of embedded pipeline metadata
                 (gsm_id, gse_id, organism, protocol, singlify_version,
                 pipeline_date, etc. — the GEO context from singlify).

The result is shaped to drop directly into scipy.sparse.csc_matrix:

    d = read_1pz("gene_counts.1pz")
    mat = scipy.sparse.csc_matrix(
        (d["data"], d["indices"], d["indptr"]), shape=d["shape"])

Parameters
----------
path : str
    Path to the .1pz file.
verify_crc : bool, default True
    Kept for API forward compatibility. The C++ reader always verifies the
    file CRC32 as part of slurp_and_validate(), so this flag currently has
    no effect — passing False will not skip the check.

Raises
------
ValueError
    If the file is missing, truncated, has the wrong magic, fails the
    CRC32 check, or cannot be decoded.
)pbdoc");
}
