#pragma once
// singlet-pileup: adt_counter.h
// T2: CITE-seq / ADT UMI deduplication and counting.
//
// Accumulates (barcode_idx, tag_idx, umi) observations from matched ADT reads
// and produces a deduplicated tags × cells count matrix.
//
// UMI dedup strategy: per-(cell, tag) exact UMI set membership.  Each unique
// UMI string encountered for a given (cell, tag) pair is counted once.  This
// is equivalent to "exact dedup" and is appropriate for the short ADT
// read-length regime where sequencing errors are rare relative to GEX reads.
//
// Thread model: create one AdtCounter per worker thread; call merge() to fold
// shard results into a single counter before finalize().
//
// Output: SparseAccumulator<uint32_t> in (tag × cell) = (row × col) layout,
// matching the convention used by pz_writer for GEX output.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sparse_accumulator.h"

namespace singlet {

class AdtCounter {
   public:
    AdtCounter() = default;

    // ── Configuration (set before any add() calls) ────────────────────────

    void set_num_tags(size_t n) { n_tags_ = n; }
    void set_num_cells(size_t n) { n_cells_ = n; }

    // ── Observation accumulation ──────────────────────────────────────────

    /// Record one ADT read.  bc_idx and tag_idx are 0-based dense indices.
    /// umi is the raw UMI string (not packed; 10–12 bp typical).
    /// Duplicate (bc_idx, tag_idx, umi) observations are silently dropped.
    void add(uint32_t bc_idx, int tag_idx, const std::string& umi) {
        if (tag_idx < 0) return;  // no-match / ambiguous
        ++total_reads_;

        // Key: pack (bc_idx, tag_idx) into 64-bit for the per-key UMI set map
        uint64_t key = (static_cast<uint64_t>(bc_idx) << 20) ^
                       static_cast<uint64_t>(static_cast<uint32_t>(tag_idx));
        auto& umi_set = obs_[key];
        if (umi_set.insert(umi).second) {
            ++total_umis_;
        }
    }

    // ── Merge (for multi-threaded use) ────────────────────────────────────

    /// Fold another shard into this counter.  Typically called on the primary
    /// counter with each worker's local counter.  Thread-unsafe — caller must
    /// synchronise (e.g. hold a mutex or merge sequentially after joining).
    void merge(const AdtCounter& other) {
        total_reads_ += other.total_reads_;
        for (const auto& [key, umi_set] : other.obs_) {
            auto& dst = obs_[key];
            for (const auto& u : umi_set) {
                if (dst.insert(u).second) ++total_umis_;
            }
        }
    }

    // ── Finalization ──────────────────────────────────────────────────────

    /// Build the SparseAccumulator from accumulated UMI sets.
    /// Must be called exactly once, after all add() / merge() calls.
    void finalize() {
        acc_.set_n_features(static_cast<uint32_t>(n_tags_));
        std::vector<std::string> dummy_barcodes(n_cells_);
        acc_.set_barcodes(dummy_barcodes);  // just sets n_barcodes_

        for (const auto& [key, umi_set] : obs_) {
            uint32_t bc_idx = static_cast<uint32_t>(key >> 20);
            uint32_t tag_idx = static_cast<uint32_t>(key & 0xFFFFF);
            uint32_t cnt = static_cast<uint32_t>(umi_set.size());
            dedup_umis_ += cnt;
            acc_.increment(tag_idx, bc_idx, cnt);
        }
        finalized_ = true;
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    const SparseAccumulator<uint32_t>& counts() const { return acc_; }

    size_t total_reads() const { return total_reads_; }
    size_t total_umis() const { return total_umis_; }
    size_t dedup_umis() const { return dedup_umis_; }
    bool finalized() const { return finalized_; }

   private:
    size_t n_tags_ = 0;
    size_t n_cells_ = 0;

    // obs_: map from packed (bc_idx, tag_idx) key → set of distinct UMI strings
    std::unordered_map<uint64_t, std::unordered_set<std::string>> obs_;

    SparseAccumulator<uint32_t> acc_;

    size_t total_reads_ = 0;
    size_t total_umis_ = 0;  // unique (cell, tag, umi) tuples observed
    size_t dedup_umis_ = 0;  // same as total_umis_ post-finalize (for clarity)
    bool finalized_ = false;
};

}  // namespace singlet
