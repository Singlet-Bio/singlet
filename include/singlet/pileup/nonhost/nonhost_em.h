// SPDX-License-Identifier: MIT
// nonhost_em.h — EM abundance deconvolution for nonhost species (NONHOST-EM)
//
// Converts per-read MinSketch multi-hit scores into species-level relative
// abundance estimates using Expectation-Maximization (same approach as Bracken
// for Kraken2 and Sylph for sketches).
//
// Input:  vector<NonHostScreener::MultiHit>  (per-read soft assignments)
// Output: vector<SpeciesAbundance>           (EM-converged relative abundances)
//
// Algorithm — standard EM for categorical mixtures:
//   θ_k  = relative abundance of species k  (sum to 1 over all nonhost reads)
//   E:     r_ik = θ_k * h_ik / Σ_j θ_j * h_ij   (h_ik = hit_rate of read i vs species k; 0 if no hit)
//   M:     θ_k  = (Σ_i r_ik) / N_nonhost          (N_nonhost = reads with ≥1 hit)
//   Stop:  max|θ_new − θ_old| < tolerance OR iter ≥ max_iter
//
// Wiring:
//   Called from singlet.cpp after STAR unmapped reads are classified with
//   NonHostScreener::classify_multi_batch().  em_deconvolve() replaces the
//   single-best-hit counting with statistically rigorous per-species abundances.
//   Outputs: nonhost_em_abundance.tsv + updated nonhost_summary.json.
//
// Pipeline insertion point: after classify_multi_batch(), before file export.
// CLI flag: none (automatic when --nonhost-db is provided).

#pragma once

#include "nonhost_screener.h"  // MultiHit, NonHostCategory
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace singlet {
namespace nonhost {

// ── Output type ──────────────────────────────────────────────────────────────

// Confidence level based on per-species mean_hit_rate after EM:
//   HIGH   ≥ 0.30 — strong match (substantial fraction of species minimizers recovered)
//   MEDIUM ≥ 0.10 — partial match (some species minimizers; plausibly real)
//   LOW    < 0.10 — likely noise from spurious k-mer matches
enum class ConfidenceLevel { LOW, MEDIUM, HIGH };

inline const char* confidence_str(ConfidenceLevel c) noexcept {
    switch (c) {
        case ConfidenceLevel::HIGH:   return "HIGH";
        case ConfidenceLevel::MEDIUM: return "MEDIUM";
        default:                      return "LOW";
    }
}

struct SpeciesAbundance {
    uint32_t        species_id         = 0;
    NonHostCategory kingdom            = NonHostCategory::UNKNOWN;
    float           relative_abundance = 0.0f;  // fraction of all nonhost reads
    uint32_t        n_reads_assigned   = 0;      // expected reads assigned by EM
    float           mean_hit_rate      = 0.0f;   // mean hit_rate weighted by EM posterior
    ConfidenceLevel confidence         = ConfidenceLevel::LOW;
};

// ── EM deconvolution ─────────────────────────────────────────────────────────

// Run EM deconvolution on a batch of multi-hit read assignments.
// Returns per-species abundances sorted descending by relative_abundance.
// Species with relative_abundance < min_abundance are filtered from the output.
//
// multi_hits       : per-read output of classify_multi_batch()
// kingdom_map      : species_id → NonHostCategory  (build from multi_hits: viral_hits → VIRAL,
//                    microbial_hits → BACTERIAL).  Used only to populate SpeciesAbundance.kingdom.
// max_iter         : EM iteration limit (default 500 — complex multiviral samples need more iters)
// tolerance        : convergence: stop when max|θ_new − θ_old| < tolerance (default 1e-6)
// min_abundance    : per-species abundance floor for output inclusion (default 1e-4 = 0.01%)
// min_prefilter_hr : per-hit rate threshold — hits below this are discarded before EM (default 0.05)
//                    Prevents low-specificity k-mer noise from entering the EM.
// min_report_hr    : per-species mean hit-rate threshold for output inclusion (default 0.30)
//                    Only high-confidence species (hit_rate ≥ 0.30) are reported; lower values
//                    are virtually always spurious k-mer noise; omitted from output.
// out_iterations   : if non-null, *out_iterations is set to the number of EM iterations performed
// min_abs_reads    : absolute minimum reads-assigned floor for output inclusion (default 0 = disabled).
//                    Set to max(500, 0.005 * total_unmapped) at the call site to prevent false
//                    positives on samples where nonhost reads are <<0.5% of total unmapped reads.
//                    A species must satisfy BOTH min_abundance AND min_abs_reads to be reported.
//
// Reads with no hits in ANY kingdom (viral_hits and microbial_hits both empty) are excluded
// from the EM and do not count toward N_nonhost.
inline std::vector<SpeciesAbundance> em_deconvolve(
        const std::vector<NonHostScreener::MultiHit>& multi_hits,
        const std::unordered_map<uint32_t, NonHostCategory>& kingdom_map,
        int      max_iter         = 500,
        float    tolerance        = 1e-6f,
        float    min_abundance    = 1e-4f,
        float    min_prefilter_hr = 0.05f,
        float    min_report_hr    = 0.30f,
        int*     out_iterations   = nullptr,
        uint32_t min_abs_reads    = 0) {

    if (multi_hits.empty()) {
        if (out_iterations) *out_iterations = 0;
        return {};
    }

    // ── Step 1: Group the reads by their hit pattern ──────────────────────────
    // A read is the set of (species, hit_rate) it hits after the pre-filter, with
    // viral and microbial rates merged by max. Reads with the same pattern contribute
    // identical EM terms, so the EM runs over the distinct patterns with an integer
    // multiplicity: the terms are summed per pattern in a canonical order (patterns
    // sorted, species ascending) in double precision, which makes the result a function
    // of the multiset of reads only, not of their order, of hash-table iteration or of
    // the number of threads that produced them.

    using Pattern = std::vector<std::pair<uint32_t, float>>;   // (species_id, rate), species ascending
    std::map<Pattern, uint64_t> pattern_count;

    for (const auto& mh : multi_hits) {
        if (mh.viral_hits.empty() && mh.microbial_hits.empty()) continue;

        // If a species appears in both viral and microbial lists (edge case), take the maximum.
        std::map<uint32_t, float> merged;
        for (const auto& [sid, rate] : mh.viral_hits)
            merged[sid] = std::max(merged[sid], rate);   // default-initialises to 0.0
        for (const auto& [sid, rate] : mh.microbial_hits)
            merged[sid] = std::max(merged[sid], rate);

        // Pre-filter: hits with hit_rate < min_prefilter_hr are almost always spurious
        // random minimizer collisions and must not enter the EM.
        Pattern pat;
        pat.reserve(merged.size());
        for (const auto& [sid, rate] : merged) {
            if (rate <= 0.0f) continue;
            if (min_prefilter_hr > 0.0f && rate < min_prefilter_hr) continue;
            pat.emplace_back(sid, rate);
        }
        if (!pat.empty()) ++pattern_count[std::move(pat)];
    }

    // species index = rank of the species id
    std::vector<uint32_t> idx_to_species;   // ascending species ids
    {
        std::set<uint32_t> ids;
        for (const auto& [pat, cnt] : pattern_count)
            for (const auto& [sid, rate] : pat) ids.insert(sid);
        idx_to_species.assign(ids.begin(), ids.end());
    }
    struct Group { std::vector<std::pair<uint32_t, float>> hits; double count; };   // (species index, rate)
    std::vector<Group> groups;
    groups.reserve(pattern_count.size());
    uint64_t n_reads_total = 0;
    for (const auto& [pat, cnt] : pattern_count) {
        Group g;
        g.count = static_cast<double>(cnt);
        n_reads_total += cnt;
        for (const auto& [sid, rate] : pat) {
            const auto it = std::lower_bound(idx_to_species.begin(), idx_to_species.end(), sid);
            g.hits.emplace_back(static_cast<uint32_t>(it - idx_to_species.begin()), rate);
        }
        groups.push_back(std::move(g));
    }

    // Canonical order (R8): reads by content, species by id
    std::sort(read_hits.begin(), read_hits.end());
    {
        for (const auto& rs : read_hits)
            for (const auto& [sid, h] : rs) { (void)h; species_to_idx.emplace(sid, 0u); }
        idx_to_species.reserve(species_to_idx.size());
        for (const auto& [sid, k] : species_to_idx) { (void)k; idx_to_species.push_back(sid); }
        std::sort(idx_to_species.begin(), idx_to_species.end());
        for (uint32_t k = 0; k < idx_to_species.size(); ++k) species_to_idx[idx_to_species[k]] = k;
        for (auto& rs : read_hits)
            for (auto& [sid, h] : rs) { (void)h; sid = species_to_idx[sid]; }
    }

    const uint32_t K = static_cast<uint32_t>(idx_to_species.size());
    const double   N = static_cast<double>(n_reads_total);   // number of nonhost reads

    if (K == 0 || N <= 0.0) {
        if (out_iterations) *out_iterations = 0;
        return {};
    }

    // ── Precompute per-species prior hit-rate for Dirichlet regularization ───
    // prior_hr[k] = unweighted mean hit_rate for species k across all reads where
    // it has a hit.  Used in M-step to apply a small hit-rate-proportional pseudo-count
    // that down-weights low-specificity species (random k-mer noise) relative to
    // high-specificity species (real viral hits with broad minimizer coverage).
    constexpr double kRegAlpha = 0.1;  // small Dirichlet pseudo-count weight
    std::vector<double> prior_hr(K, 0.0);
    {
        std::vector<double> prior_count(K, 0.0);
        for (const auto& g : groups) {
            for (const auto& [k, h] : g.hits) {
                prior_hr[k] += g.count * static_cast<double>(h);
                prior_count[k] += g.count;
            }
        }
        for (uint32_t k = 0; k < K; ++k)
            prior_hr[k] = (prior_count[k] > 0.0) ? prior_hr[k] / prior_count[k] : 0.0;
    }

    // ── Step 2: Initialise θ uniformly ──────────────────────────────────────
    std::vector<double> theta(K, 1.0 / static_cast<double>(K));
    std::vector<double> theta_new(K);

    // ── Step 3: EM iteration ─────────────────────────────────────────────────
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        std::fill(theta_new.begin(), theta_new.end(), 0.0);

        // E-step: for each pattern accumulate weighted responsibilities (× multiplicity)
        for (const auto& g : groups) {
            // Denominator: Σ_j θ_j * h_ij
            double denom = 0.0;
            for (const auto& [k, h] : g.hits)
                denom += theta[k] * static_cast<double>(h);
            if (denom <= 0.0) continue;  // numerical safety

            // Accumulate: θ_new[k] += count * θ_k * h_ik / denom
            for (const auto& [k, h] : g.hits)
                theta_new[k] += g.count * (theta[k] * static_cast<double>(h) / denom);
        }

        // M-step: apply hit-rate-weighted Dirichlet regularization and normalise.
        // theta[k] = (E[k] + alpha * prior_hr[k]) / (N + alpha * Σ prior_hr)
        // Species with high prior_hr (broad minimizer coverage) receive a larger
        // pseudo-count, penalising low-specificity noise relative to real viruses.
        double total = N;  // Σ_k theta_new[k] = N after E-step
        for (uint32_t k = 0; k < K; ++k) {
            theta_new[k] += kRegAlpha * prior_hr[k];
            total        += kRegAlpha * prior_hr[k];
        }
        if (total > 0.0) {
            for (uint32_t k = 0; k < K; ++k)
                theta_new[k] /= total;
        }

        // Convergence check: max |θ_new − θ_old|
        double max_delta = 0.0;
        for (uint32_t k = 0; k < K; ++k)
            max_delta = std::max(max_delta, std::abs(theta_new[k] - theta[k]));

        theta.swap(theta_new);

        if (max_delta < static_cast<double>(tolerance)) {
            ++iter;   // report the iteration just completed
            break;
        }
    }
    if (out_iterations) *out_iterations = iter;

    // ── Step 4: Compute per-species stats ────────────────────────────────────
    // mean_hit_rate[k] = Σ_i (h_ik * w_ik) / Σ_i w_ik   (posterior-weighted mean)
    std::vector<double> hit_rate_sum(K, 0.0);
    std::vector<double> weight_sum  (K, 0.0);

    for (const auto& g : groups) {
        double denom = 0.0;
        for (const auto& [k, h] : g.hits)
            denom += theta[k] * static_cast<double>(h);
        if (denom <= 0.0) continue;

        for (const auto& [k, h] : g.hits) {
            const double w = g.count * (theta[k] * static_cast<double>(h) / denom);
            hit_rate_sum[k] += static_cast<double>(h) * w;
            weight_sum[k]   += w;
        }
    }

    // ── Step 5: Build output, filter, and sort ───────────────────────────────
    std::vector<SpeciesAbundance> result;
    result.reserve(K);

    for (uint32_t k = 0; k < K; ++k) {
        if (theta[k] < static_cast<double>(min_abundance)) continue;

        const float mean_hr = (weight_sum[k] > 0.0)
                                  ? static_cast<float>(hit_rate_sum[k] / weight_sum[k])
                                  : 0.0f;

        // Post-filter: skip species whose posterior mean hit-rate is below the
        // reporting threshold.  Only HIGH-confidence species (≥0.30) are reported;
        // lower values are virtually always spurious k-mer noise.
        if (mean_hr < min_report_hr) continue;

        // Absolute read count gate: skip species with fewer than min_abs_reads expected reads.
        // This prevents false positives when nonhost reads are a tiny fraction of total unmapped
        // (e.g. a clean human PBMC with 10K total nonhost hits out of 2.6M unmapped reads).
        const uint32_t n_reads_est = static_cast<uint32_t>(std::llround(theta[k] * N));
        if (min_abs_reads > 0 && n_reads_est < min_abs_reads) continue;

        SpeciesAbundance sa;
        sa.species_id         = idx_to_species[k];
        sa.relative_abundance = static_cast<float>(theta[k]);
        sa.n_reads_assigned   = n_reads_est;
        sa.mean_hit_rate      = mean_hr;

        // Confidence level based on mean_hit_rate
        if      (mean_hr >= 0.30f) sa.confidence = ConfidenceLevel::HIGH;
        else if (mean_hr >= 0.10f) sa.confidence = ConfidenceLevel::MEDIUM;
        else                        sa.confidence = ConfidenceLevel::LOW;

        // Kingdom lookup from caller-supplied map
        auto kit = kingdom_map.find(sa.species_id);
        sa.kingdom = (kit != kingdom_map.end()) ? kit->second : NonHostCategory::UNKNOWN;

        result.push_back(sa);
    }

    // Sort descending by relative_abundance, ties by species_id (R8)
    std::sort(result.begin(), result.end(),
              [](const SpeciesAbundance& a, const SpeciesAbundance& b) {
                  if (a.relative_abundance != b.relative_abundance)
                      return a.relative_abundance > b.relative_abundance;
                  return a.species_id < b.species_id;
              });

    return result;
}

// ── Kingdom map builder ───────────────────────────────────────────────────────

// Build a species_id → NonHostCategory map from a batch of MultiHit results.
// All species_ids found in viral_hits → VIRAL.
// species_ids in microbial_hits → FUNGAL if their id is in fungal_species, else BACTERIAL.
// Call this before em_deconvolve() when no external map is available.
inline std::unordered_map<uint32_t, NonHostCategory> build_kingdom_map(
        const std::vector<NonHostScreener::MultiHit>& multi_hits,
        const std::unordered_set<uint32_t>& fungal_species = {}) {
    std::unordered_map<uint32_t, NonHostCategory> kmap;
    for (const auto& mh : multi_hits) {
        for (const auto& [sid, rate] : mh.viral_hits)
            kmap.emplace(sid, NonHostCategory::VIRAL);     // emplace: first insertion wins
        for (const auto& [sid, rate] : mh.microbial_hits) {
            NonHostCategory cat = fungal_species.count(sid)
                                  ? NonHostCategory::FUNGAL
                                  : NonHostCategory::BACTERIAL;
            kmap.emplace(sid, cat);
        }
    }
    return kmap;
}

} // namespace nonhost
} // namespace singlet
