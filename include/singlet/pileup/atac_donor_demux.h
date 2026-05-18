// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: atac_donor_demux.h
// A4 — ATAC-seq donor demultiplexing via fragment-based SNP genotyping.
//
// Algorithm: same VB Beta-Binomial mixture as donor_demux.h (reused directly),
// but allele counts are accumulated from ATAC fragment/read observations at
// known SNP positions rather than from RNA BAM pileup records.
//
// Pipeline:
//   1. Caller supplies a list of AtacSnpObservation records (one per
//      read-overlapping-SNP event): barcode, SNP index, ref/alt call.
//   2. AtacDonorDemux builds sparse per-barcode AD/DP matrices.
//   3. Runs singlet::run_demux() → same DemuxResult as scRNA donor demux.
//
// Key differences from scRNA demux:
//   - Input: ATAC fragment observations (no UMI dedup needed, already deduped
//     at fragment level). Fragments carry sequence via underlying reads.
//   - Fragment overlap exposes both Tn5 cut sites, giving higher per-SNP depth
//     in accessible chromatin regions.
//   - Same maximum-likelihood VB assignment logic.
//
// Usage:
//   AtacDonorDemux demux;
//   demux.set_config(cfg);        // optional
//   demux.add_snp_observation(bc_idx, snp_idx, is_alt); // called per read
//   DemuxResult result = demux.run(n_snps, n_barcodes);

#include <cstdint>
#include <fstream>  // required by donor_demux.h write_donor_assignments
#include <unordered_map>
#include <vector>

#include "donor_demux.h"  // run_demux, DemuxResult, DonorDemuxConfig

namespace singlet {

// ============================================================================
// Per-read SNP observation (fragment overlap → sequence → allele)
// ============================================================================
struct AtacSnpObservation {
    uint32_t barcode_idx;  // index into barcode list
    int32_t snp_idx;       // index into reference SNP list (0-based)
    uint8_t is_alt;        // 1 = alt allele observed, 0 = ref allele
};

// ============================================================================
// AtacDonorDemux
// Accumulates AtacSnpObservations → runs VB donor assignment.
// ============================================================================
class AtacDonorDemux {
   public:
    AtacDonorDemux() = default;

    void set_config(const DonorDemuxConfig& cfg) { cfg_ = cfg; }

    // ------------------------------------------------------------------
    // Accumulate one allele observation from an ATAC read.
    // Call once per read that overlaps a SNP position.
    // ------------------------------------------------------------------
    void add_snp_observation(uint32_t barcode_idx, int32_t snp_idx, uint8_t is_alt) {
        obs_.push_back({barcode_idx, snp_idx, is_alt});
    }

    // ------------------------------------------------------------------
    // Batch add observations (convenience wrapper).
    // ------------------------------------------------------------------
    void add_observations(const std::vector<AtacSnpObservation>& obs) {
        obs_.insert(obs_.end(), obs.begin(), obs.end());
    }

    void clear() { obs_.clear(); }

    // ------------------------------------------------------------------
    // Run donor demultiplexing.
    //
    // n_snps     : total number of SNPs in the reference panel (rows)
    // n_barcodes : total number of barcodes / cells (columns)
    //
    // Returns DemuxResult (same as run_demux in donor_demux.h).
    // ------------------------------------------------------------------
    DemuxResult run(uint32_t n_snps, uint32_t n_barcodes) const {
        // Build per-barcode per-SNP sparse tallies.
        // Key: (barcode_idx << 32) | snp_idx → {ad, dp}
        struct Tally {
            uint32_t ad = 0;
            uint32_t dp = 0;
        };
        std::unordered_map<uint64_t, Tally> tally;
        tally.reserve(obs_.size());

        for (const auto& o : obs_) {
            if (o.snp_idx < 0 || static_cast<uint32_t>(o.snp_idx) >= n_snps) continue;
            if (o.barcode_idx >= n_barcodes) continue;
            uint64_t key = (static_cast<uint64_t>(o.barcode_idx) << 32) |
                           static_cast<uint32_t>(o.snp_idx);
            auto& t = tally[key];
            t.dp++;
            if (o.is_alt) t.ad++;
        }

        // Build CSC matrices in (barcode, SNP) format.
        // donor_demux uses: indptr[barcode], indices[snp], data[value]
        // — i.e. cells are columns, SNPs are rows.

        // Collect (barcode, snp) pairs sorted by (barcode, snp)
        struct Entry {
            uint32_t bc;
            uint32_t snp;
            uint8_t ad;
            uint8_t dp;
        };
        std::vector<Entry> entries;
        entries.reserve(tally.size());
        for (const auto& [key, t] : tally) {
            uint32_t bc = static_cast<uint32_t>(key >> 32);
            uint32_t snp = static_cast<uint32_t>(key & 0xFFFFFFFF);
            uint8_t ad = static_cast<uint8_t>(std::min(t.ad, 255u));
            uint8_t dp = static_cast<uint8_t>(std::min(t.dp, 255u));
            entries.push_back({bc, snp, ad, dp});
        }

        // Sort by barcode (column), then snp (row) — CSC order
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            return a.bc != b.bc ? a.bc < b.bc : a.snp < b.snp;
        });

        // Build CSC indptr/indices/data for AD and DP
        std::vector<int32_t> ad_indptr(n_barcodes + 1, 0);
        std::vector<int32_t> ad_indices;
        std::vector<uint8_t> ad_data;
        std::vector<int32_t> dp_indptr(n_barcodes + 1, 0);
        std::vector<int32_t> dp_indices;
        std::vector<uint8_t> dp_data;

        for (const auto& e : entries) {
            ad_indices.push_back(static_cast<int32_t>(e.snp));
            ad_data.push_back(e.ad);
            dp_indices.push_back(static_cast<int32_t>(e.snp));
            dp_data.push_back(e.dp);
            ad_indptr[e.bc + 1]++;
            dp_indptr[e.bc + 1]++;
        }

        // Convert counts to prefix sums
        for (uint32_t i = 1; i <= n_barcodes; ++i) {
            ad_indptr[i] += ad_indptr[i - 1];
            dp_indptr[i] += dp_indptr[i - 1];
        }

        const int32_t* aip = ad_indptr.data();
        const int32_t* aii = ad_indices.empty() ? nullptr : ad_indices.data();
        const uint8_t* aid = ad_data.empty() ? nullptr : ad_data.data();
        const int32_t* dip = dp_indptr.data();
        const int32_t* dii = dp_indices.empty() ? nullptr : dp_indices.data();
        const uint8_t* did = dp_data.empty() ? nullptr : dp_data.data();

        return run_demux(
            n_snps, n_barcodes,
            aip, aii, aid, static_cast<uint64_t>(ad_data.size()),
            dip, dii, did, static_cast<uint64_t>(dp_data.size()),
            cfg_);
    }

    size_t observation_count() const { return obs_.size(); }

   private:
    std::vector<AtacSnpObservation> obs_;
    DonorDemuxConfig cfg_;
};

}  // namespace singlet
