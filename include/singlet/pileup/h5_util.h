// SPDX-License-Identifier: MIT
// pileup/h5_util.h — Shared HDF5 C-API boilerplate for the matrix writers
//
// Consolidates dataset / dataspace / attribute creation helpers that were
// duplicated verbatim between h5ad_writer.h and loom_writer.h.
//
// Header-only, uses the HDF5 C API directly — no HighFive, no H5Cpp.

#pragma once

#include <hdf5.h>
#include <cstdint>
#include <string>
#include <vector>

namespace singlet::pileup {
namespace h5 {

// Create a variable-length UTF-8 C-string HDF5 type. Caller must H5Tclose().
inline hid_t make_vlen_str_type() {
    hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, H5T_VARIABLE);
    H5Tset_strpad(t, H5T_STR_NULLTERM);
    H5Tset_cset(t, H5T_CSET_UTF8);
    return t;
}

// Write a scalar VL-string attribute named `name` on `loc`.
inline bool write_str_attr(hid_t loc, const char* name, const char* value) {
    hid_t stype = make_vlen_str_type();
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr  = H5Acreate2(loc, name, stype, space, H5P_DEFAULT, H5P_DEFAULT);
    bool ok = (attr >= 0);
    if (ok) {
        // VL string write: H5Awrite expects a pointer to const char*.
        ok = (H5Awrite(attr, stype, &value) >= 0);
        H5Aclose(attr);
    }
    H5Sclose(space);
    H5Tclose(stype);
    return ok;
}

// Write a 1-D VL-string dataset named `name` under `loc`.
// An empty vector produces a valid 0-length dataset.
inline bool write_vlen_str_dataset(hid_t loc, const char* name,
                                   const std::vector<std::string>& values) {
    hid_t stype = make_vlen_str_type();
    hsize_t n   = static_cast<hsize_t>(values.size());
    hid_t space = H5Screate_simple(1, &n, nullptr);
    hid_t dset  = H5Dcreate2(loc, name, stype, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    bool ok = (dset >= 0);
    if (ok) {
        if (n > 0) {
            std::vector<const char*> ptrs;
            ptrs.reserve(n);
            for (const auto& s : values) ptrs.push_back(s.c_str());
            ok = (H5Dwrite(dset, stype, H5S_ALL, H5S_ALL,
                           H5P_DEFAULT, ptrs.data()) >= 0);
        }
        H5Dclose(dset);
    }
    H5Sclose(space);
    H5Tclose(stype);
    return ok;
}

// Write a 1-D numeric dataset named `name` under `grp`.
//   file_type: HDF5 type stored in the file (may differ for upcasting)
//   mem_type : HDF5 type of the in-memory buffer
// A count of 0 or a null buffer produces a valid empty dataset.
inline bool write_numeric_dataset(hid_t grp, const char* name,
                                  hsize_t count, hid_t file_type,
                                  hid_t mem_type, const void* buf) {
    hid_t space = H5Screate_simple(1, &count, nullptr);
    hid_t dset  = H5Dcreate2(grp, name, file_type, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    bool ok = (dset >= 0);
    if (ok) {
        if (count > 0 && buf != nullptr) {
            ok = (H5Dwrite(dset, mem_type, H5S_ALL, H5S_ALL,
                           H5P_DEFAULT, buf) >= 0);
        }
        H5Dclose(dset);
    }
    H5Sclose(space);
    return ok;
}

}  // namespace h5
}  // namespace singlet::pileup
