// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: te_counter.h  — T-L2-4
// Per-cell × TE-family sparse UMI counter writing te_counts.1pz
// per DROPLET_OUTPUT_SCHEMA.md §6.5 (schema version 1.1).
//
// Thread model: one TeCounter per worker thread; call merge() to fold
// shards before write().
//
// Output metadata (all required keys per §6.5):
//   schema_version=1.1, matrix_kind=te_counts, value_dtype=uint16,
//   te_family_source=Dfam_3.8, te_family_table_md5=<hex>,
//   k_mer_size=21, min_family_kmer_fraction=0.5,
//   resolved_at_layer=L2_te_classifier, track=B

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "sparse_accumulator.h"
#include "pz_writer.h"
#include "te_classifier.h"

namespace singlet {

// Required metadata keys per §6.5
struct TeCounterMeta {
    std::string protocol_id;          // e.g. "22"
    std::string species;              // "human" | "mouse"
    std::string te_family_table_hash; // hex SHA-256 of family_table.tsv
    std::string sample_id;
    std::string singlet_version;
    std::string reference_build;
    // Derived constants (fixed for L2)
    std::string te_family_table_source = "Dfam_3.8";
    std::string min_family_kmer_fraction = "0.5";
    std::string kmer_size = "21";
    std::string em_seed = "0xC0FFEE";
    std::string cascade_layer = "L2";
};

// ============================================================================
// TeCounter
// ============================================================================

class TeCounter {
   public:
    TeCounter() = default;

    void set_n_families(uint32_t n) {
        n_families_ = n;
        acc_.set_n_features(n);
    }
    void set_n_cells(uint32_t n) {
        n_cells_ = n;
    }
    void set_barcodes(const std::vector<std::string>& barcodes) {
        acc_.set_barcodes(barcodes);
        n_cells_ = static_cast<uint32_t>(barcodes.size());
    }

    /// Record one resolved TE UMI.
    /// family_idx: 0-based index into TeFamilySketchDB.families
    /// cell_idx:   0-based dense barcode index
    void add(uint32_t family_idx, uint32_t cell_idx) {
        if (family_idx >= n_families_) return;
        if (cell_idx   >= n_cells_)   return;
        acc_.increment(family_idx, cell_idx, static_cast<uint32_t>(1));
        ++total_umis_;
    }

    /// Merge another shard into this counter.  Single-threaded; caller synchronises.
    void merge(TeCounter& other) {
        acc_.merge_from(other.acc_);
        total_umis_ += other.total_umis_;
    }

    uint64_t total_umis() const { return total_umis_; }

    /// Write te_counts.1pz to `path`.
    /// family_names: display names for the TE family axis (row names).
    /// cell_barcodes: barcode strings for the cell axis (col names).
    /// meta: required metadata block per §6.5.
    bool write(const std::string& path,
               const std::vector<std::string>& family_names,
               const std::vector<std::string>& cell_barcodes,
               const TeCounterMeta& meta,
               int threads = 4) const
    {
        if (family_names.size() != n_families_)
            throw std::runtime_error("te_counter: family_names size mismatch");
        if (cell_barcodes.size() != n_cells_)
            throw std::runtime_error("te_counter: cell_barcodes size mismatch");

        auto csc = acc_.to_csc();

        // Build required metadata per §6.5
        std::map<std::string, std::string> user_meta;
        user_meta["schema_version"]           = "1.1";
        user_meta["matrix_kind"]              = "te_counts";
        user_meta["value_dtype"]              = "uint16";
        user_meta["te_family_source"]         = meta.te_family_table_source;
        user_meta["te_family_table_md5"]      = meta.te_family_table_hash;
        user_meta["k_mer_size"]               = meta.kmer_size;
        user_meta["min_family_kmer_fraction"] = meta.min_family_kmer_fraction;
        user_meta["resolved_at_layer"]        = "L2_te_classifier";
        user_meta["track"]                    = "B";
        user_meta["protocol_id"]              = meta.protocol_id;
        user_meta["species"]                  = meta.species;
        user_meta["sample_id"]                = meta.sample_id;
        user_meta["singlet_version"]         = meta.singlet_version;
        user_meta["reference_build"]          = meta.reference_build;
        user_meta["em_seed"]                  = meta.em_seed;
        user_meta["cascade_layer"]            = meta.cascade_layer;
        user_meta["n_features"]               = std::to_string(n_families_);
        user_meta["n_cells"]                  = std::to_string(n_cells_);

        return pz::write_1pz<uint32_t>(
            path,
            csc.nrows, csc.ncols,
            csc.indptr, csc.indices, csc.data,
            family_names, cell_barcodes,
            /*zstd_level=*/3, /*chunk_cols=*/1024, threads, user_meta);
    }

   private:
    uint32_t n_families_ = 0;
    uint32_t n_cells_    = 0;
    uint64_t total_umis_ = 0;
    SparseAccumulator<uint32_t> acc_;
};

}  // namespace singlet
