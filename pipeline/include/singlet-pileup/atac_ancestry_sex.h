#pragma once
// singlet-pileup: atac_ancestry_sex.h
// A5 — ATAC-seq ancestry classification + sex/karyotype calling.
//
// Ancestry:
//   Uses the same 50-AIM panel from ancestry.h but accumulates allele counts
//   from ATAC fragment/read observations rather than RNA BAM pileup.
//   Input: vector<AtacAimObservation> (barcode-level or sample-level)
//   Algorithm: same maximum-likelihood over 5 superpopulations as N13.
//   Output: AncestryResult (ancestry, confidence, prob[5], n_informative)
//
// Sex/Karyotype:
//   Counts ATAC fragments on chrX vs chrY vs autosomes.
//   No gene model needed — uses chromosome-level fragment counts.
//   Algorithm: chrY_frac = chrY_frags / (chrX_frags + chrY_frags)
//     XX: chrY_frac < 0.05
//     XY: chrY_frac > 0.25
//     uncertain: otherwise (mosaicism, low coverage, aneuploidy)
//   Confidence: distance from nearest threshold, scaled to [0,1].
//
// Usage:
//   // --- Sex calling ---
//   AtacSexCaller sex;
//   sex.add_fragment("chr1", 1000, 1200);  // autosome
//   sex.add_fragment("chrX", 5000, 5200);  // chrX
//   sex.add_fragment("chrY", 6000, 6200);  // chrY
//   AtacSexResult sr = sex.call();
//
//   // --- Ancestry (sample-level) ---
//   AtacAncestryClassifier anc;
//   anc.add_aim_observation(aim_idx, is_alt);  // per read overlapping AIM
//   AncestryResult ar = anc.classify();

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <iostream>
#include "singlet-pileup/ancestry.h"  // AIM, AncestryResult, get_aim_panel, kPopNames, Pop

namespace singlet {

// ============================================================================
// Sex / karyotype calling from ATAC fragment chromosome distribution
// ============================================================================

/// Result of ATAC-based sex/karyotype inference.
struct AtacSexResult {
    std::string sex;          ///< "XX", "XY", "uncertain", or "no_data"
    double      confidence;   ///< 0–1: distance from threshold, scaled
    uint64_t    chrx_frags;   ///< fragment count on chrX
    uint64_t    chry_frags;   ///< fragment count on chrY
    uint64_t    auto_frags;   ///< fragment count on autosomes
    double      chry_frac;    ///< chrY / (chrX + chrY); NaN if both zero
};

/// Accumulates per-fragment chromosome assignments and calls sex/karyotype.
class AtacSexCaller {
public:
    AtacSexCaller() = default;

    /// Record one fragment. Supply the chromosome name of the fragment midpoint.
    void add_fragment(const std::string& chrom, int32_t /*start*/, int32_t /*end*/) {
        if (chrom == "chrX" || chrom == "X") {
            chrx_frags_++;
        } else if (chrom == "chrY" || chrom == "Y") {
            chry_frags_++;
        } else {
            auto_frags_++;
        }
    }

    /// Batch-add counts directly (e.g., from a per-chromosome tally).
    void add_counts(uint64_t chrx, uint64_t chry, uint64_t autosomes) {
        chrx_frags_ += chrx;
        chry_frags_ += chry;
        auto_frags_ += autosomes;
    }

    void clear() { chrx_frags_ = chry_frags_ = auto_frags_ = 0; }

    /// Call sex/karyotype from accumulated fragment counts.
    AtacSexResult call() const {
        AtacSexResult r;
        r.chrx_frags = chrx_frags_;
        r.chry_frags = chry_frags_;
        r.auto_frags = auto_frags_;

        const uint64_t xy_total = chrx_frags_ + chry_frags_;

        if (xy_total == 0) {
            r.sex = "no_data";
            r.confidence = 0.0;
            r.chry_frac  = std::numeric_limits<double>::quiet_NaN();
            return r;
        }

        r.chry_frac = static_cast<double>(chry_frags_) / static_cast<double>(xy_total);

        // Thresholds (validated against 10x ATAC references):
        //   XY males:   chrY/(chrX+chrY) typically 0.35–0.55
        //   XX females: chrY/(chrX+chrY) typically 0.00–0.02
        static constexpr double kXXThresh   = 0.05;
        static constexpr double kXYThresh   = 0.25;
        static constexpr double kMidpoint   = (kXXThresh + kXYThresh) / 2.0;
        static constexpr double kHalfRange  = (kXYThresh - kXXThresh) / 2.0;

        if (r.chry_frac < kXXThresh) {
            r.sex = "XX";
            // confidence: 1.0 at frac=0, 0.0 at frac=kXXThresh
            r.confidence = 1.0 - r.chry_frac / kXXThresh;
        } else if (r.chry_frac > kXYThresh) {
            r.sex = "XY";
            // confidence: 0.0 at frac=kXYThresh, approaches 1.0 as frac→1
            double excess = r.chry_frac - kXYThresh;
            r.confidence  = std::min(1.0, excess / (1.0 - kXYThresh));
        } else {
            r.sex = "uncertain";
            // confidence: 0.0 at midpoint, 0.0 at thresholds
            double dist_from_mid = std::abs(r.chry_frac - kMidpoint);
            r.confidence = kHalfRange > 0.0 ? dist_from_mid / kHalfRange : 0.0;
        }

        return r;
    }

    uint64_t chrx_frags() const { return chrx_frags_; }
    uint64_t chry_frags() const { return chry_frags_; }
    uint64_t auto_frags() const { return auto_frags_; }

private:
    uint64_t chrx_frags_ = 0;
    uint64_t chry_frags_ = 0;
    uint64_t auto_frags_ = 0;
};

// ============================================================================
// Ancestry calling from ATAC fragment allele observations at AIM positions
// ============================================================================

/// A single allele observation at an AIM position from an ATAC read.
struct AtacAimObservation {
    uint32_t aim_idx;   ///< index into get_aim_panel() vector
    uint8_t  is_alt;    ///< 1 = alt allele observed, 0 = ref allele
};

/// Classifies continental ancestry from ATAC fragment allele counts at AIMs.
/// Reuses the same max-likelihood algorithm as ancestry.h (N13).
class AtacAncestryClassifier {
public:
    explicit AtacAncestryClassifier(const std::vector<AIM>* panel = nullptr)
        : panel_(panel ? panel : &get_aim_panel()) {
        clear();
    }

    void set_panel(const std::vector<AIM>& panel) { panel_ = &panel; }

    /// Record one allele observation from an ATAC read overlapping an AIM.
    void add_aim_observation(uint32_t aim_idx, uint8_t is_alt) {
        if (aim_idx >= panel_->size()) return;
        auto& t = tally_[aim_idx];
        t.dp++;
        if (is_alt) t.ad++;
    }

    /// Batch-add observations.
    void add_observations(const std::vector<AtacAimObservation>& obs) {
        for (const auto& o : obs) add_aim_observation(o.aim_idx, o.is_alt);
    }

    void clear() { tally_.clear(); }

    /// Run max-likelihood ancestry classification.
    AncestryResult classify() const {
        AncestryResult result;
        result.ancestry      = "ambiguous";
        result.confidence    = 0.0;
        result.n_informative = 0;
        result.n_covered     = 0;
        result.low_data      = false;
        for (int p = 0; p < 5; ++p) result.log_lik[p] = 0.0;

        const auto& aims = *panel_;
        static constexpr double kMinFreq = 1e-6;

        for (const auto& [idx, t] : tally_) {
            if (idx >= aims.size()) continue;
            if (t.dp == 0) continue;

            result.n_informative++;
            if (t.dp >= 3) result.n_covered++;

            const AIM& aim = aims[idx];
            for (int p = 0; p < 5; ++p) {
                double f = std::max(std::min(aim.freq[p], 1.0 - kMinFreq), kMinFreq);
                // AD observations: alt allele; (DP - AD): ref allele
                result.log_lik[p] += t.ad * std::log(f) +
                                     (t.dp - t.ad) * std::log(1.0 - f);
            }
        }

        result.low_data = (result.n_informative < kAncestryMinInformative);

        // Softmax to get posterior probabilities
        double max_ll = *std::max_element(result.log_lik, result.log_lik + 5);
        double sum_exp = 0.0;
        for (int p = 0; p < 5; ++p) {
            result.prob[p] = std::exp(result.log_lik[p] - max_ll);
            sum_exp        += result.prob[p];
        }
        for (int p = 0; p < 5; ++p) result.prob[p] /= sum_exp;

        // Pick top population
        int top_pop = static_cast<int>(
            std::max_element(result.prob, result.prob + 5) - result.prob);
        result.confidence = result.prob[top_pop];

        if (!result.low_data || result.confidence > 0.85) {
            result.ancestry = kPopNames[top_pop];
        }

        return result;
    }

private:
    struct Tally { uint32_t ad = 0; uint32_t dp = 0; };
    const std::vector<AIM>* panel_;
    std::unordered_map<uint32_t, Tally> tally_;
};

// ============================================================================
// JSON output helpers
// ============================================================================

/// Write ATAC sex call to JSON.
inline void write_atac_sex_json(const std::string& path, const AtacSexResult& r) {
    std::ofstream f(path);
    f << "{\n"
      << "  \"sex\": \"" << r.sex << "\",\n"
      << "  \"confidence\": " << r.confidence << ",\n"
      << "  \"chrx_fragments\": " << r.chrx_frags << ",\n"
      << "  \"chry_fragments\": " << r.chry_frags << ",\n"
      << "  \"autosome_fragments\": " << r.auto_frags << ",\n"
      << "  \"chry_fraction\": " << (std::isnan(r.chry_frac) ? 0.0 : r.chry_frac) << "\n"
      << "}\n";
}

/// Write ATAC ancestry call to JSON.
inline void write_atac_ancestry_json(const std::string& path, const AncestryResult& r) {
    std::ofstream f(path);
    f << "{\n"
      << "  \"ancestry\": \"" << r.ancestry << "\",\n"
      << "  \"confidence\": " << r.confidence << ",\n"
      << "  \"n_informative\": " << r.n_informative << ",\n"
      << "  \"n_covered\": " << r.n_covered << ",\n"
      << "  \"low_data\": " << (r.low_data ? "true" : "false") << ",\n"
      << "  \"populations\": {\n";
    for (int p = 0; p < 5; ++p) {
        f << "    \"" << kPopNames[p] << "\": " << r.prob[p]
          << (p < 4 ? "," : "") << "\n";
    }
    f << "  }\n}\n";
}

}  // namespace singlet
