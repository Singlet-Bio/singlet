// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_r_to_eigen.hpp
//
// Marshalling helpers: Matrix::dgCMatrix (R) ↔ singlet_gpu::core::HostCsc (C++).
//
// WHY pinned host CSC: cudaMemcpyAsync requires the source buffer to be
// page-locked (pinned).  Pageable R vectors would force a silent synchronous
// copy inside the CUDA driver.  We allocate pinned host CSC once, copy R's
// slot vectors into it, then hand it to the kernel.
//
// Host-copy cost (documented R limitation):
//   R has no native GPU sparse type (no cupy, no CuDF).  Every kernel call
//   incurs one O(nnz) host copy on input AND one on output.  For a typical
//   scRNA sample (10k cells, 30M nnz) this is ~100–200 ms of PCIe bandwidth.
//   Users who require full device residency between calls should use the
//   Python wrapper (singlet_gpu Python package, cupy zero-copy path).
//
// dgCMatrix slot layout (0-indexed CSC, standard Matrix::dgCMatrix):
//   @i   — integer vector of length nnz: row indices
//   @p   — integer vector of length n_cols+1: column pointers
//   @x   — numeric vector of length nnz: values (double/float)
//   @Dim — integer vector of length 2: [n_rows, n_cols]
//
// Note: R's @x is "numeric" (double), but singlet-gpu kernels work in fp32.
//   We saturating-cast double → float on copy (values > 2^24 lose precision;
//   typical count matrices never exceed 2^16).

#pragma once

#include <Rcpp.h>

// singlet-gpu headers (header-only; no nvcc needed for this include unit)
#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/io/pz_device_loader.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// HostCsc — plain struct for pinned host CSC that mirrors the layout
// singlet-gpu kernels expect.  We do NOT use core::PinnedBuffer directly here
// because PinnedBuffer requires CUDA to be present at include time (cudaMallocHost).
// This struct uses std::vector<float> for the values and will be given to
// singlet-gpu's pinned path after conversion — see r_sparse_to_host_csc below.
// ---------------------------------------------------------------------------
struct HostCsc {
    std::vector<int32_t> indptr;   // length n_cols + 1
    std::vector<int32_t> indices;  // length nnz
    std::vector<float>   values;   // length nnz  (fp32 cast from R double)
    int32_t n_rows = 0;
    int32_t n_cols = 0;
    int64_t nnz    = 0;
};

// ---------------------------------------------------------------------------
// r_sparse_to_host_csc
//
// Extract the @i, @p, @x, @Dim slots from a Matrix::dgCMatrix R S4 object
// and copy into a HostCsc (pageable host memory).
//
// Caller checks: input must be class "dgCMatrix".  We validate @Dim[0] > 0
// and @Dim[1] > 0 but do not re-validate the structural integrity of @i/@p
// (that is the Matrix package's responsibility).
//
// Values: R @x slot is "numeric" (double).  We saturating-cast to float.
//   Values > 2^24 lose mantissa precision — a warning is emitted to R's
//   message stream if any such value is found (rare in count matrices).
// ---------------------------------------------------------------------------
inline HostCsc r_sparse_to_host_csc(const Rcpp::S4& dgc) {
    // Validate class
    if (!dgc.is("dgCMatrix")) {
        Rcpp::stop("r_sparse_to_host_csc: input must be a Matrix::dgCMatrix");
    }

    Rcpp::IntegerVector r_i   = dgc.slot("i");
    Rcpp::IntegerVector r_p   = dgc.slot("p");
    Rcpp::NumericVector r_x   = dgc.slot("x");
    Rcpp::IntegerVector r_dim = dgc.slot("Dim");

    if (r_dim.size() != 2) {
        Rcpp::stop("r_sparse_to_host_csc: @Dim must have length 2");
    }

    const int32_t n_rows = static_cast<int32_t>(r_dim[0]);
    const int32_t n_cols = static_cast<int32_t>(r_dim[1]);
    const int64_t nnz    = static_cast<int64_t>(r_x.size());

    if (n_rows <= 0 || n_cols <= 0) {
        Rcpp::stop("r_sparse_to_host_csc: matrix must have positive dimensions");
    }
    if (static_cast<int64_t>(r_i.size()) != nnz) {
        Rcpp::stop("r_sparse_to_host_csc: @i and @x length mismatch");
    }
    if (static_cast<int64_t>(r_p.size()) != static_cast<int64_t>(n_cols + 1)) {
        Rcpp::stop("r_sparse_to_host_csc: @p length must be n_cols+1");
    }

    HostCsc out;
    out.n_rows = n_rows;
    out.n_cols = n_cols;
    out.nnz    = nnz;

    // Copy indptr (@p) — already int32
    out.indptr.resize(static_cast<std::size_t>(n_cols + 1));
    for (int32_t j = 0; j <= n_cols; ++j) {
        out.indptr[static_cast<std::size_t>(j)] = r_p[j];
    }

    // Copy indices (@i) — already int32
    out.indices.resize(static_cast<std::size_t>(nnz));
    for (int64_t k = 0; k < nnz; ++k) {
        out.indices[static_cast<std::size_t>(k)] = r_i[static_cast<R_xlen_t>(k)];
    }

    // Copy values (@x): double → float with saturation check
    out.values.resize(static_cast<std::size_t>(nnz));
    bool warned = false;
    static constexpr double kMaxExact = 16777216.0;  // 2^24
    for (int64_t k = 0; k < nnz; ++k) {
        double v = r_x[static_cast<R_xlen_t>(k)];
        if (!warned && (v > kMaxExact || v < -kMaxExact)) {
            Rcpp::warning(
                "r_sparse_to_host_csc: value %g exceeds fp32 exact range (2^24); "
                "mantissa precision lost on cast to float32", v);
            warned = true;
        }
        out.values[static_cast<std::size_t>(k)] = static_cast<float>(v);
    }

    return out;
}

// ---------------------------------------------------------------------------
// host_csc_to_r_sparse
//
// Construct a Matrix::dgCMatrix from raw C++ CSC data.
// Used by kernels that return a modified sparse matrix (e.g. lognorm_cpp).
//
// Parameters:
//   indptr  — int32 column pointers, length n_cols+1
//   indices — int32 row indices, length nnz
//   values  — float32 values, length nnz
//   n_rows, n_cols — matrix dimensions
//
// The returned S4 object holds a fresh copy; there is no aliasing.
// ---------------------------------------------------------------------------
inline Rcpp::S4 host_csc_to_r_sparse(
        const int32_t* indptr,
        const int32_t* indices,
        const float*   values,
        int32_t        n_rows,
        int32_t        n_cols)
{
    const int64_t nnz = static_cast<int64_t>(indptr[n_cols]);

    // Build R vectors
    Rcpp::IntegerVector r_p(static_cast<R_xlen_t>(n_cols + 1));
    Rcpp::IntegerVector r_i(static_cast<R_xlen_t>(nnz));
    Rcpp::NumericVector r_x(static_cast<R_xlen_t>(nnz));

    for (int32_t j = 0; j <= n_cols; ++j) {
        r_p[j] = indptr[static_cast<std::size_t>(j)];
    }
    for (int64_t k = 0; k < nnz; ++k) {
        r_i[static_cast<R_xlen_t>(k)] = indices[static_cast<std::size_t>(k)];
        r_x[static_cast<R_xlen_t>(k)] = static_cast<double>(
                values[static_cast<std::size_t>(k)]);
    }

    // Construct Matrix::dgCMatrix via new()
    Rcpp::S4 dgc("dgCMatrix");
    dgc.slot("i")   = r_i;
    dgc.slot("p")   = r_p;
    dgc.slot("x")   = r_x;
    dgc.slot("Dim") = Rcpp::IntegerVector::create(n_rows, n_cols);

    return dgc;
}

// ---------------------------------------------------------------------------
// host_csc_to_r_sparse (overload for std::vector arguments)
// ---------------------------------------------------------------------------
inline Rcpp::S4 host_csc_to_r_sparse(
        const std::vector<int32_t>& indptr,
        const std::vector<int32_t>& indices,
        const std::vector<float>&   values,
        int32_t n_rows,
        int32_t n_cols)
{
    return host_csc_to_r_sparse(
        indptr.data(), indices.data(), values.data(), n_rows, n_cols);
}

}  // namespace singlet_gpu_r
