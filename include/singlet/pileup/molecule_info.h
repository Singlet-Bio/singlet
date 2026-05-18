// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: molecule_info.h
// Cell Ranger-compatible molecule_info.h5 output.
// Uses HDF5 C API directly — header-only, no HighFive or H5Cpp dependency.
//
// CELL RANGER MOLECULE_INFO.H5 LAYOUT (v3):
//   /
//   ├── barcode            int64[n_mol]    2-bit-encoded cell barcodes
//   ├── barcode_idx        int64[n_mol]    index into /barcodes list
//   ├── count              int32[n_mol]    UMI count (1 for raw records)
//   ├── feature_idx        int32[n_mol]    gene/feature index
//   ├── gem_group          int32[n_mol]    GEM group (always 1)
//   ├── library_idx        int32[n_mol]    library index (always 0)
//   ├── umi                int64[n_mol]    2-bit-encoded UMI
//   ├── umi_type           int32[n_mol]    0=polyT, 1=random
//   ├── metrics_json       scalar string   pipeline metrics JSON
//   ├── barcodes           string[n_bc]    barcode sequence strings (unique, sorted)
//   └── feature_ref/
//       ├── id             string[n_feat]  Ensembl gene IDs
//       ├── name           string[n_feat]  gene names / symbols
//       └── feature_type   string[n_feat]  "Gene Expression" for all scRNA
//
// SORT ORDER:  records sorted by barcode (2-bit uint64) before writing,
//              then by feature_idx within the same barcode (stable sort).
//
// USAGE:
//   MoleculeInfoWriter w;
//   w.open({filepath, 16, 12});
//   for (auto& mol : molecules) w.add_molecule(mol);
//   w.finalize(gene_names, gene_ids, barcodes, metrics_json);
//
// Compatible with:
//   cellranger mat2csv, kb-python, scvi-tools MoleculeInfoWriter,
//   STARsolo --soloFeatures GeneFull --outFormat CellRanger2

#include <hdf5.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace singlet {

// ── Records ───────────────────────────────────────────────────────────────

/// One per-UMI molecule record (in-memory, before HDF5 flush).
struct MoleculeRecord {
    uint64_t barcode     = 0;   ///< 2-bit-encoded barcode
    int32_t  feature_idx = 0;   ///< Gene / feature index
    uint64_t umi         = 0;   ///< 2-bit-encoded UMI
    int32_t  count       = 1;   ///< UMI count (1 for raw per-molecule)
};

// ── Config ────────────────────────────────────────────────────────────────

struct MoleculeInfoConfig {
    std::string filepath;
    uint32_t    cb_len  = 16;  ///< Barcode length (bp)
    uint32_t    umi_len = 12;  ///< UMI length (bp)
};

// ── Implementation helpers ────────────────────────────────────────────────

namespace mol_info_detail {

/// Create a variable-length UTF-8 C-string HDF5 type. Caller must H5Tclose.
inline hid_t make_vl_str_type() {
    hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, H5T_VARIABLE);
    H5Tset_cset(t, H5T_CSET_UTF8);
    return t;
}

/// Write a 1-D int32 dataset under `parent` named `name`.
inline bool write_int32_dataset(hid_t parent, const char* name,
                                 const std::vector<int32_t>& data) {
    hsize_t dims = data.size();
    hid_t space  = H5Screate_simple(1, &dims, nullptr);
    hid_t ds     = H5Dcreate2(parent, name, H5T_NATIVE_INT32, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    bool ok = (ds >= 0);
    if (ok && !data.empty())
        ok = (H5Dwrite(ds, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       data.data()) >= 0);
    if (ds >= 0)  H5Dclose(ds);
    H5Sclose(space);
    return ok;
}

/// Write a 1-D int64 dataset under `parent` named `name`.
inline bool write_int64_dataset(hid_t parent, const char* name,
                                 const std::vector<int64_t>& data) {
    hsize_t dims = data.size();
    hid_t space  = H5Screate_simple(1, &dims, nullptr);
    hid_t ds     = H5Dcreate2(parent, name, H5T_NATIVE_INT64, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    bool ok = (ds >= 0);
    if (ok && !data.empty())
        ok = (H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       data.data()) >= 0);
    if (ds >= 0)  H5Dclose(ds);
    H5Sclose(space);
    return ok;
}

/// Write a 1-D variable-length string dataset.
inline bool write_str_dataset(hid_t parent, const char* name,
                               const std::vector<std::string>& strs) {
    hsize_t n    = strs.size();
    hid_t space  = H5Screate_simple(1, &n, nullptr);
    hid_t vl_str = make_vl_str_type();
    hid_t ds     = H5Dcreate2(parent, name, vl_str, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    bool ok = (ds >= 0);
    if (ok && !strs.empty()) {
        std::vector<const char*> ptrs;
        ptrs.reserve(strs.size());
        for (const auto& s : strs) ptrs.push_back(s.c_str());
        ok = (H5Dwrite(ds, vl_str, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       ptrs.data()) >= 0);
    }
    if (ds >= 0) H5Dclose(ds);
    H5Tclose(vl_str);
    H5Sclose(space);
    return ok;
}

/// Write a scalar VL-string dataset (single value).
inline bool write_scalar_str(hid_t parent, const char* name,
                              const std::string& value) {
    hid_t space  = H5Screate(H5S_SCALAR);
    hid_t vl_str = make_vl_str_type();
    hid_t ds     = H5Dcreate2(parent, name, vl_str, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    bool ok = (ds >= 0);
    if (ok) {
        const char* p = value.c_str();
        ok = (H5Dwrite(ds, vl_str, H5S_ALL, H5S_ALL, H5P_DEFAULT, &p) >= 0);
    }
    if (ds >= 0) H5Dclose(ds);
    H5Tclose(vl_str);
    H5Sclose(space);
    return ok;
}

} // namespace mol_info_detail

// ── Writer class ──────────────────────────────────────────────────────────

class MoleculeInfoWriter {
public:
    MoleculeInfoWriter() = default;
    ~MoleculeInfoWriter() { close(); }

    /// Initialise writer; does not create the HDF5 file yet.
    bool open(const MoleculeInfoConfig& cfg) {
        cfg_       = cfg;
        molecules_.clear();
        return true;  // HDF5 file is created in finalize()
    }

    /// Accumulate one molecule record.
    void add_molecule(const MoleculeRecord& rec) {
        molecules_.push_back(rec);
    }

    /// Sort accumulated records by barcode (then feature_idx), write HDF5 file.
    /// `gene_names`, `gene_ids`, `barcodes` MUST be non-empty.
    /// `metrics_json` defaults to "{}".
    bool finalize(const std::vector<std::string>& gene_names,
                  const std::vector<std::string>& gene_ids,
                  const std::vector<std::string>& barcodes,
                  const std::string& metrics_json = "{}") {
        // Sort by barcode, then feature_idx (stable)
        std::stable_sort(molecules_.begin(), molecules_.end(),
            [](const MoleculeRecord& a, const MoleculeRecord& b) {
                if (a.barcode != b.barcode) return a.barcode < b.barcode;
                return a.feature_idx < b.feature_idx;
            });

        // Build barcode look-up: barcode_str → index into `barcodes`
        std::unordered_map<std::string, int64_t> bc_index;
        bc_index.reserve(barcodes.size());
        for (int64_t i = 0; i < (int64_t)barcodes.size(); ++i)
            bc_index[barcodes[i]] = i;

        const size_t n = molecules_.size();

        // Parallel arrays to write
        std::vector<int64_t> v_barcode(n), v_barcode_idx(n), v_umi(n);
        std::vector<int32_t> v_count(n), v_feature_idx(n),
                             v_gem_group(n, 1), v_library_idx(n, 0),
                             v_umi_type(n, 0);

        for (size_t i = 0; i < n; ++i) {
            const auto& m   = molecules_[i];
            v_barcode[i]     = static_cast<int64_t>(m.barcode);
            v_umi[i]         = static_cast<int64_t>(m.umi);
            v_count[i]       = m.count;
            v_feature_idx[i] = m.feature_idx;
            // Resolve barcode_idx: decode to string → look up in barcodes list.
            // If not found (e.g. barcode not in the provided list), default to -1.
            std::string bc_seq;
            if (cfg_.cb_len <= 32) {
                // Use bus encoding for decode (same 2-bit scheme)
                static const char B[4] = {'A','C','G','T'};
                bc_seq.resize(cfg_.cb_len);
                for (uint32_t j = 0; j < cfg_.cb_len; ++j) {
                    uint32_t bits = static_cast<uint32_t>(
                        (m.barcode >> (62 - 2*j)) & 0x3ULL);
                    bc_seq[j] = B[bits];
                }
            }
            auto it = bc_index.find(bc_seq);
            v_barcode_idx[i] = (it != bc_index.end()) ? it->second : -1;
        }

        // Create / truncate HDF5 file
        hid_t fid = H5Fcreate(cfg_.filepath.c_str(), H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);
        if (fid < 0) {
            std::cerr << "[molecule_info] cannot create '" << cfg_.filepath << "'\n";
            return false;
        }

        using namespace mol_info_detail;
        bool ok = true;

        ok &= write_int64_dataset(fid, "barcode",      v_barcode);
        ok &= write_int64_dataset(fid, "barcode_idx",  v_barcode_idx);
        ok &= write_int32_dataset(fid, "count",        v_count);
        ok &= write_int32_dataset(fid, "feature_idx",  v_feature_idx);
        ok &= write_int32_dataset(fid, "gem_group",    v_gem_group);
        ok &= write_int32_dataset(fid, "library_idx",  v_library_idx);
        ok &= write_int64_dataset(fid, "umi",          v_umi);
        ok &= write_int32_dataset(fid, "umi_type",     v_umi_type);
        ok &= write_scalar_str  (fid, "metrics_json",  metrics_json);
        ok &= write_str_dataset (fid, "barcodes",      barcodes);

        // feature_ref group
        hid_t feat_grp = H5Gcreate2(fid, "feature_ref",
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (feat_grp >= 0) {
            ok &= write_str_dataset(feat_grp, "name", gene_names);
            ok &= write_str_dataset(feat_grp, "id",   gene_ids);

            // feature_type: "Gene Expression" for every gene
            std::vector<std::string> feat_types(gene_names.size(), "Gene Expression");
            ok &= write_str_dataset(feat_grp, "feature_type", feat_types);

            H5Gclose(feat_grp);
        } else {
            ok = false;
        }

        H5Fclose(fid);
        return ok;
    }

    void close() { molecules_.clear(); }

    uint64_t molecules_written() const { return molecules_.size(); }

private:
    MoleculeInfoConfig        cfg_;
    std::vector<MoleculeRecord> molecules_;
};

// ── Read-back helpers (for tests) ─────────────────────────────────────────

namespace mol_info_detail {

/// Read a complete 1-D int64 dataset. Returns empty on error.
inline std::vector<int64_t> read_int64_ds(hid_t loc, const char* name) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) return {};
    hid_t sp  = H5Dget_space(ds);
    hsize_t n = 0;
    H5Sget_simple_extent_dims(sp, &n, nullptr);
    std::vector<int64_t> v(n);
    if (n > 0)
        H5Dread(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Sclose(sp);
    H5Dclose(ds);
    return v;
}

/// Read a complete 1-D int32 dataset. Returns empty on error.
inline std::vector<int32_t> read_int32_ds(hid_t loc, const char* name) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) return {};
    hid_t sp  = H5Dget_space(ds);
    hsize_t n = 0;
    H5Sget_simple_extent_dims(sp, &n, nullptr);
    std::vector<int32_t> v(n);
    if (n > 0)
        H5Dread(ds, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Sclose(sp);
    H5Dclose(ds);
    return v;
}

/// Read a complete 1-D VL-string dataset.
inline std::vector<std::string> read_str_ds(hid_t loc, const char* name) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) return {};
    hid_t sp   = H5Dget_space(ds);
    hsize_t n  = 0;
    H5Sget_simple_extent_dims(sp, &n, nullptr);
    hid_t vl_t = make_vl_str_type();
    std::vector<char*> ptrs(n, nullptr);
    if (n > 0)
        H5Dread(ds, vl_t, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptrs.data());
    std::vector<std::string> result;
    result.reserve(n);
    for (auto* p : ptrs) result.emplace_back(p ? p : "");
    H5Treclaim(vl_t, sp, H5P_DEFAULT, ptrs.data());
    H5Tclose(vl_t);
    H5Sclose(sp);
    H5Dclose(ds);
    return result;
}

/// Read a scalar VL-string dataset. Returns empty string on error.
inline std::string read_scalar_str_ds(hid_t loc, const char* name) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) return "";
    hid_t sp   = H5Dget_space(ds);
    hid_t vl_t = make_vl_str_type();
    char* p    = nullptr;
    H5Dread(ds, vl_t, H5S_ALL, H5S_ALL, H5P_DEFAULT, &p);
    std::string result = p ? p : "";
    H5Treclaim(vl_t, sp, H5P_DEFAULT, &p);
    H5Tclose(vl_t);
    H5Sclose(sp);
    H5Dclose(ds);
    return result;
}

} // namespace mol_info_detail

} // namespace singlet
