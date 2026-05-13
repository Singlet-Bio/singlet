// singlet R package — Rcpp binding for pz_reader.h.
//
// This is a *thin* wrapper. All decode work (format parsing, VOCSC chunk
// decompression, bit-plane / bitmap / byte-split inverse transforms,
// permutation unmap, varint column decoding) happens inside the header-only
// C++ reader at ../inst/include/singlet/pz_reader.h. The R binding
// just marshals the returned ReadResult into R-native objects:
//
//   * a dgCMatrix (Matrix package sparse column-compressed)
//   * attributes for rownames / colnames / user_kv / vt_code
//
// The pz_reader.h file is a bit-exact copy of the one used by the Python
// wrapper. It is staged into `inst/include/singlet/` at package build time
// so R CMD INSTALL can find it via the -I flag in Makevars.

#include <Rcpp.h>

#include <exception>
#include <string>

#include "singlet/pz_reader.h"

namespace pz = singlet::pz;

//' Read a .1pz file.
//'
//' @description Low-level binding that returns the raw decoded payload as
//'   an R list. Prefer the high-level [read_1pz()] wrapper in `R/read.R`,
//'   which attaches this output into a `Matrix::dgCMatrix` with named
//'   rows and columns.
//'
//' @param path Filesystem path to the ``.1pz`` file.
//' @return A list with the following elements:
//' \describe{
//'   \item{shape}{Integer vector of length 2: (nrows, ncols).}
//'   \item{nnz}{Total number of non-zero entries.}
//'   \item{vt_code}{Writer's value-type hint (1 = uint8, 2 = uint16, 3 = uint32).}
//'   \item{indptr}{Integer vector of length ncols + 1.}
//'   \item{indices}{Integer vector of length nnz — row indices.}
//'   \item{data}{Numeric (double) vector of length nnz — values, promoted
//'     to double so that R can store them without loss.}
//'   \item{rownames}{Character vector of feature names.}
//'   \item{colnames}{Character vector of cell barcodes.}
//'   \item{user_kv}{Named character vector of embedded pipeline metadata
//'     (gsm_id, gse_id, organism, protocol, singlet_version,
//'     pipeline_date, etc.).}
//' }
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List pz_read_1pz_raw(std::string path) {
    pz::ReadResult r;
    try {
        r = pz::read_1pz(path);
    } catch (const pz::PZReadError& e) {
        Rcpp::stop("pz_reader: %s", e.what());
    } catch (const std::exception& e) {
        Rcpp::stop("pz_reader: %s", e.what());
    }

    // indptr
    Rcpp::IntegerVector indptr(r.indptr.size());
    for (size_t i = 0; i < r.indptr.size(); ++i) {
        indptr[i] = static_cast<int>(r.indptr[i]);
    }

    // indices
    Rcpp::IntegerVector indices(r.indices.size());
    for (size_t i = 0; i < r.indices.size(); ++i) {
        indices[i] = static_cast<int>(r.indices[i]);
    }

    // data: promote to double for R (dgCMatrix stores doubles; R has no
    // native integer sparse matrix in Matrix). Callers that want narrower
    // storage can cast downstream.
    Rcpp::NumericVector data(r.data.size());
    for (size_t i = 0; i < r.data.size(); ++i) {
        data[i] = static_cast<double>(r.data[i]);
    }

    Rcpp::CharacterVector rownames(r.rownames.size());
    for (size_t i = 0; i < r.rownames.size(); ++i) {
        rownames[i] = r.rownames[i];
    }

    Rcpp::CharacterVector colnames(r.colnames.size());
    for (size_t i = 0; i < r.colnames.size(); ++i) {
        colnames[i] = r.colnames[i];
    }

    // user_kv as a named character vector
    Rcpp::CharacterVector user_kv(r.user_kv.size());
    Rcpp::CharacterVector user_kv_names(r.user_kv.size());
    {
        size_t i = 0;
        for (const auto& p : r.user_kv) {
            user_kv_names[i] = p.first;
            user_kv[i] = p.second;
            ++i;
        }
        user_kv.names() = user_kv_names;
    }

    return Rcpp::List::create(
        Rcpp::Named("shape")    = Rcpp::IntegerVector::create(
            static_cast<int>(r.m), static_cast<int>(r.n)),
        Rcpp::Named("nnz")      = static_cast<double>(r.nnz),
        Rcpp::Named("vt_code")  = static_cast<int>(r.vt_code),
        Rcpp::Named("indptr")   = indptr,
        Rcpp::Named("indices")  = indices,
        Rcpp::Named("data")     = data,
        Rcpp::Named("rownames") = rownames,
        Rcpp::Named("colnames") = colnames,
        Rcpp::Named("user_kv")  = user_kv
    );
}
