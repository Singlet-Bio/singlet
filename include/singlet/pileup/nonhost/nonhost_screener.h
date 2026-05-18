// SPDX-License-Identifier: MIT
// nonhost_screener.h — Per-read viral/bacterial/fungal classification
//
// Part of NONHOST-SYLPH-PORT (DAG task). Classifies unmapped reads against
// two MinSketchIndex databases (viral and microbial) using hit-rate thresholding.
//
// Host-subtraction (NONHOST-HOST-SUBTRACT):
//   classify_multi_filtered() and classify_multi_batch_filtered() accept a
//   HostKmerFilter and exclude host-genome minimizers before querying the viral
//   and microbial indexes.  Reads whose non-host minimizer count is below
//   kMinNonhostMins (default 5) are treated as UNKNOWN (host reads that happen
//   to be unmapped — e.g., reads from repetitive/centromeric regions).
//
// Wiring (NONHOST-UNMAPPED-CAPTURE — separate DAG task):
//   singlet pipeline captures unmapped reads from the STAR phase after
//   alignment (reads with SAM flag 0x4 in the STARsolo-tagged BAM).
//   These sequences are passed to NonHostScreener::classify_batch().
//   Cell barcodes for unmapped reads are recovered from the CB tag in the BAM.
//   The output flows to NONHOST-SCREENER → NONHOST-MAP-VIRAL / NONHOST-MINIMAP2.
//
// Category assignment:
//   VIRAL      — top hit found in viral_idx above min_hit_rate
//   BACTERIAL  — top hit found in microbial_idx above min_hit_rate, and
//                species_id is NOT in the fungal_species set
//   FUNGAL     — top hit found in microbial_idx above min_hit_rate, and
//                species_id IS in the fungal_species set
//   UNKNOWN    — no index has a hit above min_hit_rate, or read is too short / all-N
//
// FUNGAL wiring:
//   Caller populates fungal_species (unordered_set<uint32_t>) from the fungal
//   MinSketchIndex entries, then passes it to the NonHostScreener constructor.
//   Species in microbial_idx whose IDs appear in fungal_species are assigned
//   FUNGAL; all others remain BACTERIAL.  This allows a single merged microbial
//   index (bacterial + fungal concatenated) while still distinguishing kingdoms.

#pragma once

#include "min_sketch.h"
#include "host_kmer_filter.h"
#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#  if defined(__GLIBCXX__)
#    include <parallel/algorithm>
#    define NONHOST_HAS_GNU_PARALLEL 1
#  endif
#endif

namespace singlet {
namespace nonhost {

enum class NonHostCategory : uint8_t {
    UNKNOWN    = 0,
    VIRAL      = 1,
    BACTERIAL  = 2,
    FUNGAL     = 3,
};

struct NonHostHit {
    uint32_t        species_id = 0;
    uint32_t        hit_count  = 0;
    float           hit_rate   = 0.0f;
    NonHostCategory category   = NonHostCategory::UNKNOWN;
};

class NonHostScreener {
public:
    // viral_idx      : MinSketchIndex built from viral reference sequences
    // microbial_idx  : MinSketchIndex built from bacterial+fungal references
    // min_hit_rate   : minimum fraction of query minimizers that must match for
    //                  a positive call (default 0.05 = 5%)
    // fungal_species : optional set of species_ids from the fungal index.
    //                  When provided, microbial hits whose species_id is in this
    //                  set are assigned FUNGAL instead of BACTERIAL.
    NonHostScreener(const MinSketchIndex& viral_idx,
                    const MinSketchIndex& microbial_idx,
                    float min_hit_rate = 0.05f,
                    std::unordered_set<uint32_t> fungal_species = {})
        : viral_(viral_idx), microbial_(microbial_idx),
          min_hit_rate_(min_hit_rate),
          fungal_species_(std::move(fungal_species))
    {}

    // Returns FUNGAL if species_id is in the fungal set, else BACTERIAL.
    NonHostCategory microbial_category(uint32_t species_id) const {
        return fungal_species_.count(species_id) ? NonHostCategory::FUNGAL
                                                  : NonHostCategory::BACTERIAL;
    }

    // Classify a single read sequence against both indexes.
    // Returns NonHostHit with category=UNKNOWN if no index hits exceed min_hit_rate,
    // or if the read is too short to form any minimizers (len < k+w-1).
    NonHostHit classify(const std::string& read_seq) const {
        // Extract minimizers once; share between viral and microbial lookups
        auto mins = extract_minimizers(read_seq, viral_.k(), viral_.w());
        if (mins.empty()) return {};                       // too short or all-N
        const uint32_t n_possible = static_cast<uint32_t>(mins.size());

        // ── Check viral index ────────────────────────────────────────────────
        auto vh = viral_.query_minimizers(mins);
        if (!vh.empty()) {
            auto best = std::max_element(vh.begin(), vh.end(),
                [](const auto& a, const auto& b){ return a.second < b.second; });
            float rate = static_cast<float>(best->second) / static_cast<float>(n_possible);
            if (rate >= min_hit_rate_)
                return {best->first, best->second, rate, NonHostCategory::VIRAL};
        }

        // ── Check microbial index ────────────────────────────────────────────
        // FUNGAL vs BACTERIAL is resolved by checking fungal_species_: if the
        // best-hit species_id is in that set, it is FUNGAL; otherwise BACTERIAL.
        auto mh = microbial_.query_minimizers(mins);
        if (!mh.empty()) {
            auto best = std::max_element(mh.begin(), mh.end(),
                [](const auto& a, const auto& b){ return a.second < b.second; });
            float rate = static_cast<float>(best->second) / static_cast<float>(n_possible);
            if (rate >= min_hit_rate_)
                return {best->first, best->second, rate, microbial_category(best->first)};
        }

        return {};  // UNKNOWN
    }

    // ── Multi-hit soft assignment (for EM deconvolution) ──────────────────────

    // Per-read result with ALL species above the soft threshold (min_hit_rate * mult).
    // Used as input to em_deconvolve() for statistically rigorous abundance estimation.
    struct MultiHit {
        std::string read_id;         // optional, empty if not tracking
        NonHostCategory top_category = NonHostCategory::UNKNOWN;  // kingdom of best hit
        std::vector<std::pair<uint32_t, float>> viral_hits;       // {species_id, hit_rate}
        std::vector<std::pair<uint32_t, float>> microbial_hits;   // {species_id, hit_rate}
    };

    // Classify a single read against both indexes with a lower (soft) threshold.
    // Returns ALL viral and microbial hits whose hit_rate ≥ min_hit_rate_ * soft_threshold_multiplier.
    // top_category reflects the kingdom with the single best-rate hit.
    // Returns an empty MultiHit (UNKNOWN) if the read is too short or produces no qualifying hits.
    MultiHit classify_multi(const std::string& read_seq,
                             float soft_threshold_multiplier = 0.25f) const {
        auto mins = extract_minimizers(read_seq, viral_.k(), viral_.w());
        if (mins.empty()) return {};
        const uint32_t n_possible = static_cast<uint32_t>(mins.size());
        const float soft_thresh   = min_hit_rate_ * soft_threshold_multiplier;

        MultiHit result;
        float best_viral_rate     = 0.0f;
        float best_microbial_rate = 0.0f;

        // ── Viral hits ───────────────────────────────────────────────────────
        auto vh = viral_.query_minimizers(mins);
        for (const auto& [sid, cnt] : vh) {
            float rate = static_cast<float>(cnt) / static_cast<float>(n_possible);
            if (rate >= soft_thresh) {
                result.viral_hits.emplace_back(sid, rate);
                if (rate > best_viral_rate) best_viral_rate = rate;
            }
        }

        // ── Microbial hits ───────────────────────────────────────────────────
        auto mh = microbial_.query_minimizers(mins);
        for (const auto& [sid, cnt] : mh) {
            float rate = static_cast<float>(cnt) / static_cast<float>(n_possible);
            if (rate >= soft_thresh) {
                result.microbial_hits.emplace_back(sid, rate);
                if (rate > best_microbial_rate) best_microbial_rate = rate;
            }
        }

        // top_category = kingdom with the best single hit rate
        if (best_viral_rate == 0.0f && best_microbial_rate == 0.0f) {
            result.top_category = NonHostCategory::UNKNOWN;
        } else if (best_viral_rate >= best_microbial_rate) {
            result.top_category = NonHostCategory::VIRAL;
        } else {
            // Find the best-rate microbial species to determine BACTERIAL vs FUNGAL
            uint32_t best_micro_sid = 0;
            float    best_micro_r   = 0.0f;
            for (const auto& [sid, r] : result.microbial_hits)
                if (r > best_micro_r) { best_micro_r = r; best_micro_sid = sid; }
            result.top_category = microbial_category(best_micro_sid);
        }

        return result;
    }

    // ── Batch classification using sort-merge scan (FAST path) ───────────────
    //
    // The classify_batch / classify_multi_batch / classify_multi_batch_filtered
    // methods all use the same batch algorithm:
    //
    //   Phase A (parallel):  Extract minimizers from each read.
    //                        For filtered: apply HostKmerFilter per-read.
    //   Phase B (parallel):  Sort the flat (hash, read_idx) list using an
    //                        OMP-parallel presort + GNU parallel sort.
    //   Phase C (sequential): Linear merge-scan of sorted minimizers vs sorted
    //                         DB entries — O(n_mins + DB_size) sequential DRAM
    //                         reads instead of O(n_mins × log DB_size) random.
    //   Phase D (parallel):  Convert per-read hit maps to result objects.
    //
    // Speedup vs per-read binary search:
    //   9.1M reads × 30 mins × log2(1.06B) ≈ 820M random DRAM lookups → ~1600s
    //   batch sort-merge: ~3s sort + ~1s linear scan × 2 DBs → ~15s total
    //   Effective speedup: ~100× for 9.1M-read batches.

    // ── Internal helpers ──────────────────────────────────────────────────────

    // Parallel sort of a flat (hash, read_idx) list using GNU parallel sort if
    // available, falling back to OMP presort+merge, then std::sort.
    static void parallel_sort_flat(
            std::vector<std::pair<uint64_t, uint32_t>>& flat,
            int n_threads) {
        if (flat.empty()) return;
        auto cmp = [](const std::pair<uint64_t,uint32_t>& a,
                      const std::pair<uint64_t,uint32_t>& b) {
            return a.first < b.first;
        };
#ifdef NONHOST_HAS_GNU_PARALLEL
        const int nt = std::max(1, n_threads);
        __gnu_parallel::sort(flat.begin(), flat.end(), cmp,
                             __gnu_parallel::parallel_tag(nt));
#elif defined(_OPENMP)
        // OMP presort: sort N chunks in parallel, then inplace_merge pairwise
        const int nt = std::max(1, n_threads);
        const size_t chunk = (flat.size() + nt - 1) / nt;
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int t = 0; t < nt; ++t) {
            const size_t lo = static_cast<size_t>(t) * chunk;
            const size_t hi = std::min(lo + chunk, flat.size());
            if (lo < hi) std::sort(flat.begin() + lo, flat.begin() + hi, cmp);
        }
        // Pairwise merge until one sorted run
        for (size_t step = chunk; step < flat.size(); step *= 2) {
            #pragma omp parallel for schedule(static) num_threads(nt)
            for (size_t lo = 0; lo < flat.size(); lo += 2 * step) {
                const size_t mid = std::min(lo + step, flat.size());
                const size_t hi  = std::min(lo + 2 * step, flat.size());
                if (mid < hi)
                    std::inplace_merge(flat.begin() + lo,
                                       flat.begin() + mid,
                                       flat.begin() + hi, cmp);
            }
        }
#else
        std::sort(flat.begin(), flat.end(), cmp);
#endif
    }

    // Build per-read MultiHit from batch-query hit maps (Phase D helper).
    // viral_hits[ri]    : {species_id → hit_count} from viral DB
    // microbial_hits[ri]: {species_id → hit_count} from microbial DB
    // n_possible        : number of minimizers for this read
    MultiHit hits_to_multi_hit(
            const std::unordered_map<uint32_t, uint32_t>& viral_h,
            const std::unordered_map<uint32_t, uint32_t>& microbial_h,
            uint32_t n_possible,
            float soft_threshold_multiplier) const {
        if (n_possible == 0) return {};
        const float soft_thresh = min_hit_rate_ * soft_threshold_multiplier;
        MultiHit result;
        float best_viral_rate     = 0.0f;
        float best_microbial_rate = 0.0f;

        for (const auto& [sid, cnt] : viral_h) {
            float rate = static_cast<float>(cnt) / static_cast<float>(n_possible);
            if (rate >= soft_thresh) {
                result.viral_hits.emplace_back(sid, rate);
                if (rate > best_viral_rate) best_viral_rate = rate;
            }
        }
        for (const auto& [sid, cnt] : microbial_h) {
            float rate = static_cast<float>(cnt) / static_cast<float>(n_possible);
            if (rate >= soft_thresh) {
                result.microbial_hits.emplace_back(sid, rate);
                if (rate > best_microbial_rate) best_microbial_rate = rate;
            }
        }
        if (best_viral_rate == 0.0f && best_microbial_rate == 0.0f) {
            result.top_category = NonHostCategory::UNKNOWN;
        } else if (best_viral_rate >= best_microbial_rate) {
            result.top_category = NonHostCategory::VIRAL;
        } else {
            uint32_t best_micro_sid = 0; float best_micro_r = 0.0f;
            for (const auto& [sid, r] : result.microbial_hits)
                if (r > best_micro_r) { best_micro_r = r; best_micro_sid = sid; }
            result.top_category = microbial_category(best_micro_sid);
        }
        return result;
    }

public:
    // ── Public batch API ──────────────────────────────────────────────────────

    // Classify a batch of reads with soft thresholding using sort-merge batch query.
    // n_threads: number of threads to use (0 = use omp_get_max_threads()).
    // Returns results in the same order as the input reads.
    std::vector<MultiHit> classify_multi_batch(
            const std::vector<std::string>& reads,
            float soft_threshold_multiplier = 0.25f,
            int n_threads = 0) const {

        const size_t n_reads = reads.size();
        if (n_reads == 0) return {};
#ifdef _OPENMP
        const int nt = std::max(1, n_threads > 0 ? n_threads : omp_get_max_threads());
#else
        const int nt = 1;
#endif

        // Phase A: extract minimizers from each read in parallel
        std::vector<std::vector<uint64_t>> read_mins(n_reads);
        size_t total_mins = 0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64) num_threads(nt)
#endif
        for (int ri = 0; ri < static_cast<int>(n_reads); ++ri)
            read_mins[ri] = extract_minimizers(reads[ri], viral_.k(), viral_.w());
        for (const auto& m : read_mins) total_mins += m.size();

        // Phase B: build flat (hash, read_idx) list and sort in parallel
        std::vector<std::pair<uint64_t, uint32_t>> flat;
        flat.reserve(total_mins);
        for (uint32_t ri = 0; ri < static_cast<uint32_t>(n_reads); ++ri)
            for (uint64_t h : read_mins[ri])
                flat.push_back({h, ri});
        parallel_sort_flat(flat, nt);

        // Phase C: batch merge-scan against both DBs (sequential, cache-friendly)
        std::vector<std::unordered_map<uint32_t, uint32_t>> vh(n_reads), mh(n_reads);
        viral_.batch_query_presorted(flat, vh);
        microbial_.batch_query_presorted(flat, mh);

        // Phase D: convert raw hit maps to MultiHit results in parallel
        std::vector<MultiHit> results(n_reads);
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64) num_threads(nt)
#endif
        for (int ri = 0; ri < static_cast<int>(n_reads); ++ri) {
            const uint32_t np = static_cast<uint32_t>(read_mins[ri].size());
            if (np == 0) continue;
            results[ri] = hits_to_multi_hit(vh[ri], mh[ri], np, soft_threshold_multiplier);
        }
        return results;
    }

    // Classify a batch of reads with host-subtraction using sort-merge batch query.
    // n_threads: number of threads to use (0 = use omp_get_max_threads()).
    // Returns results in the same order as the input reads.
    std::vector<MultiHit> classify_multi_batch_filtered(
            const std::vector<std::string>& reads,
            const HostKmerFilter&           host_filter,
            float soft_threshold_multiplier = 0.25f,
            int n_threads = 0) const {

        const size_t n_reads = reads.size();
        if (n_reads == 0) return {};
#ifdef _OPENMP
        const int nt = std::max(1, n_threads > 0 ? n_threads : omp_get_max_threads());
#else
        const int nt = 1;
#endif

        // Phase A: extract + filter minimizers per read (parallel)
        // Reads with < kMinNonhostMins non-host minimizers stay empty (treated UNKNOWN).
        std::vector<std::vector<uint64_t>> read_mins(n_reads);  // filtered per-read
        size_t total_mins = 0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64) num_threads(nt)
#endif
        for (int ri = 0; ri < static_cast<int>(n_reads); ++ri) {
            auto all = extract_minimizers(reads[ri], viral_.k(), viral_.w());
            std::vector<uint64_t> kept;
            kept.reserve(all.size());
            for (uint64_t h : all)
                if (!host_filter.contains(h))
                    kept.push_back(h);
            if (kept.size() >= kMinNonhostMins)
                read_mins[ri] = std::move(kept);
        }
        for (const auto& m : read_mins) total_mins += m.size();

        // Phase B: build flat list and sort in parallel
        std::vector<std::pair<uint64_t, uint32_t>> flat;
        flat.reserve(total_mins);
        for (uint32_t ri = 0; ri < static_cast<uint32_t>(n_reads); ++ri)
            for (uint64_t h : read_mins[ri])
                flat.push_back({h, ri});
        parallel_sort_flat(flat, nt);

        // Phase C: batch merge-scan against both DBs
        std::vector<std::unordered_map<uint32_t, uint32_t>> vh(n_reads), mh(n_reads);
        viral_.batch_query_presorted(flat, vh);
        microbial_.batch_query_presorted(flat, mh);

        // Phase D: convert hit maps to MultiHit results in parallel
        std::vector<MultiHit> results(n_reads);
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64) num_threads(nt)
#endif
        for (int ri = 0; ri < static_cast<int>(n_reads); ++ri) {
            const uint32_t np = static_cast<uint32_t>(read_mins[ri].size());
            if (np == 0) continue;
            // Use n_nonhost (filtered count) as denominator — same as classify_multi_filtered
            results[ri] = hits_to_multi_hit(vh[ri], mh[ri], np, soft_threshold_multiplier);
        }
        return results;
    }



    // Minimum number of non-host minimizers required to call a read nonhost.
    // Reads with fewer non-host minimizers are returned as UNKNOWN (probably
    // a host read from a repetitive or centromeric region that escaped STAR).
    static constexpr uint32_t kMinNonhostMins = 5u;

    // Classify a single read against both indexes, excluding minimizers that
    // appear in the host genome Bloom filter.
    //
    // The hit_rate for each species is computed as:
    //   viral_hits_nonhost / n_nonhost_mins
    // where n_nonhost_mins = total minimizers NOT in the host filter.
    //
    // If n_nonhost_mins < kMinNonhostMins, returns UNKNOWN immediately —
    // the read does not have enough non-host signal to make a reliable call.
    MultiHit classify_multi_filtered(
            const std::string&    read_seq,
            const HostKmerFilter& host_filter,
            float soft_threshold_multiplier = 0.25f) const {

        // Extract all minimizers (same (k,w) as the indexes)
        auto all_mins = extract_minimizers(read_seq, viral_.k(), viral_.w());
        if (all_mins.empty()) return {};

        // Partition into host-shared and non-host minimizers
        std::vector<uint64_t> nonhost_mins;
        nonhost_mins.reserve(all_mins.size());
        for (uint64_t h : all_mins)
            if (!host_filter.contains(h))
                nonhost_mins.push_back(h);

        // If too few non-host minimizers, treat as host read
        if (nonhost_mins.size() < kMinNonhostMins) return {};

        const uint32_t n_nonhost = static_cast<uint32_t>(nonhost_mins.size());
        const float soft_thresh  = min_hit_rate_ * soft_threshold_multiplier;

        MultiHit result;
        float best_viral_rate     = 0.0f;
        float best_microbial_rate = 0.0f;

        // Query viral index with non-host minimizers only
        auto vh = viral_.query_minimizers(nonhost_mins);
        for (const auto& [sid, cnt] : vh) {
            float rate = static_cast<float>(cnt) / static_cast<float>(n_nonhost);
            if (rate >= soft_thresh) {
                result.viral_hits.emplace_back(sid, rate);
                if (rate > best_viral_rate) best_viral_rate = rate;
            }
        }

        // Query microbial index with non-host minimizers only
        auto mh = microbial_.query_minimizers(nonhost_mins);
        for (const auto& [sid, cnt] : mh) {
            float rate = static_cast<float>(cnt) / static_cast<float>(n_nonhost);
            if (rate >= soft_thresh) {
                result.microbial_hits.emplace_back(sid, rate);
                if (rate > best_microbial_rate) best_microbial_rate = rate;
            }
        }

        if (best_viral_rate == 0.0f && best_microbial_rate == 0.0f) {
            result.top_category = NonHostCategory::UNKNOWN;
        } else if (best_viral_rate >= best_microbial_rate) {
            result.top_category = NonHostCategory::VIRAL;
        } else {
            // Find the best-rate microbial species to determine BACTERIAL vs FUNGAL
            uint32_t best_micro_sid = 0;
            float    best_micro_r   = 0.0f;
            for (const auto& [sid, r] : result.microbial_hits)
                if (r > best_micro_r) { best_micro_r = r; best_micro_sid = sid; }
            result.top_category = microbial_category(best_micro_sid);
        }

        return result;
    }

    float min_hit_rate() const { return min_hit_rate_; }

    // Access the fungal species set (for external kingdom map construction).
    const std::unordered_set<uint32_t>& fungal_species() const { return fungal_species_; }

    // Classify a batch of reads, one per element (OpenMP parallel, per-read binary search).
    // Prefer classify_multi_batch() for large batches — it uses the faster sort-merge path.
    // n_threads: number of threads to use (0 = use omp_get_max_threads()).
    std::vector<NonHostHit> classify_batch(const std::vector<std::string>& reads,
                                            int n_threads = 0) const {
        std::vector<NonHostHit> results(reads.size());
#ifdef _OPENMP
        const int nt = std::max(1, n_threads > 0 ? n_threads : omp_get_max_threads());
        #pragma omp parallel for schedule(dynamic, 32) num_threads(nt)
#endif
        for (int i = 0; i < static_cast<int>(reads.size()); ++i)
            results[static_cast<size_t>(i)] = classify(reads[static_cast<size_t>(i)]);
        return results;
    }

private:
    const MinSketchIndex&        viral_;
    const MinSketchIndex&        microbial_;
    float                        min_hit_rate_;
    std::unordered_set<uint32_t> fungal_species_;  // species_ids from fungal index
};

} // namespace nonhost
} // namespace singlet
