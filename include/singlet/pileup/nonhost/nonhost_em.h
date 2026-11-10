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
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <vector>

namespace singlet {
namespace nonhost {

// ── Species keys ─────────────────────────────────────────────────────────────
// The viral and microbial indices number their species independently, so one id can name a virus in one index and a
// microbe in the other (about 1% of candidate rows in pbmc1k_v3). Inside the engine a species is keyed by index plus id:
// a viral key is the viral id, a microbial key is the microbial id with bit 31 set. tag_microbial_keys() applies this to
// the screen's output once, so the EM, the per-cell matrix and the confirmed reads count a virus and a microbe that share
// an id as two species. Outputs write raw_species_id(key), the id in its own index, which is unchanged (D-100).
constexpr uint32_t kMicrobialKeyBit = 0x80000000u;
inline constexpr bool     is_microbial_key(uint32_t key) noexcept { return (key & kMicrobialKeyBit) != 0; }
inline constexpr uint32_t raw_species_id(uint32_t key)   noexcept { return key & ~kMicrobialKeyBit; }
inline constexpr uint32_t microbial_key(uint32_t sid)    noexcept { return sid | kMicrobialKeyBit; }

// Sets bit 31 on every microbial hit id. Returns false (and changes nothing) if an id already uses bit 31.
inline bool tag_microbial_keys(std::vector<NonHostScreener::MultiHit>& multi_hits) {
    for (const auto& mh : multi_hits)
        for (const auto& [sid, rate] : mh.microbial_hits)
            if (is_microbial_key(sid)) return false;
    for (auto& mh : multi_hits)
        for (auto& h : mh.microbial_hits) h.first = microbial_key(h.first);
    return true;
}

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
// One EM fit (steps 1 to 4 below), which every report of the same reads shares: the iterations depend only on the reads, the
// pre-filter, the iteration cap and the tolerance, never on the report thresholds (min_abundance, min_report_hr, min_abs_reads).
// The E-step and the per-species statistics run on nthreads threads and are bit-identical to the serial loop at any count: each
// pattern's denominator is its own sum in the pattern's species order, and each species adds its terms in pattern order through
// an inverted list, which is the order the serial loop adds them in.
struct EmFit {
    std::vector<uint32_t> idx_to_species;   // ascending species ids (species index = rank)
    std::vector<double>   theta;            // relative abundance per species index
    std::vector<double>   hit_rate_sum, weight_sum;
    double                N = 0.0;          // reads that entered the EM
    int                   iterations = 0;
    size_t                n_groups = 0;      // distinct hit patterns (diagnostics)
    double                sec_group = 0, sec_invert = 0, sec_iterate = 0, sec_stats = 0;   // wall seconds of the steps (diagnostics only)
};

namespace em_detail {
// fn(begin, end) over [0, n) in contiguous ranges: cuts (nthreads + 1 entries) or equal splits
template <class Fn>
inline void par_ranges(size_t n, int nthreads, const std::vector<size_t>* cuts, Fn&& fn) {
    const int nt = std::max(1, nthreads);
    if (nt == 1 || n < 2) { fn(size_t(0), n); return; }
    std::vector<std::thread> ts;
    ts.reserve(nt);
    for (int t = 0; t < nt; ++t) {
        const size_t b = cuts ? (*cuts)[t] : n * t / nt, e = cuts ? (*cuts)[t + 1] : n * (t + 1) / nt;
        if (b < e) ts.emplace_back([&fn, b, e]() { fn(b, e); });
    }
    for (auto& th : ts) th.join();
}
}  // namespace em_detail

inline EmFit em_fit(
        const std::vector<NonHostScreener::MultiHit>& multi_hits,
        int      max_iter         = 500,
        float    tolerance        = 1e-6f,
        float    min_prefilter_hr = 0.05f,
        int      nthreads         = 1) {
    EmFit fit;
    if (multi_hits.empty()) return fit;
    using em_clock = std::chrono::steady_clock;
    auto em_t0 = em_clock::now();
    auto em_lap = [&]() { const auto t1 = em_clock::now(); const double d = std::chrono::duration<double>(t1 - em_t0).count(); em_t0 = t1; return d; };

    // ── Step 1: Group the reads by their hit pattern ──────────────────────────
    // A read is the set of (species, hit_rate) it hits after the pre-filter, with
    // viral and microbial rates merged by max. Reads with the same pattern contribute
    // identical EM terms, so the EM runs over the distinct patterns with an integer
    // multiplicity: the terms are summed per pattern in a canonical order (patterns
    // sorted, species ascending) in double precision, which makes the result a function
    // of the multiset of reads only, not of their order, of hash-table iteration or of
    // the number of threads that produced them.

    using Pattern = std::vector<std::pair<uint32_t, float>>;   // (species_id, rate), species ascending
    // Each thread counts the patterns of a contiguous range of reads in a hash map and sorts its distinct patterns; the sorted lists are
    // merged pairwise (equal patterns add their counts). The result is the sorted list of distinct patterns with their counts, the
    // content of the std::map<Pattern, uint64_t> the serial loop built, whatever the thread count (counts are integers, and the
    // order of patterns is the lexicographic order of (species, rate) pairs).
    struct PatHash {
        size_t operator()(const Pattern& p) const noexcept {
            uint64_t h = 1469598103934665603ULL;
            for (const auto& [sid, rate] : p) {
                uint32_t rb;
                std::memcpy(&rb, &rate, sizeof rb);
                h ^= sid; h *= 1099511628211ULL;
                h ^= rb;  h *= 1099511628211ULL;
            }
            return static_cast<size_t>(h ^ (h >> 29));
        }
    };
    using PatCount = std::vector<std::pair<Pattern, uint64_t>>;
    const size_t nmh = multi_hits.size();
    const int nt1 = (nmh >= 200000) ? std::max(1, nthreads) : 1;
    std::vector<PatCount> parts(nt1);
    auto build_part = [&](int t) {
        const size_t rb = nmh * static_cast<size_t>(t) / nt1, re = nmh * static_cast<size_t>(t + 1) / nt1;
        std::unordered_map<Pattern, uint64_t, PatHash> local;
        std::vector<std::pair<uint32_t, float>> tmp;
        for (size_t i = rb; i < re; ++i) {
            const auto& mh = multi_hits[i];
            if (mh.viral_hits.empty() && mh.microbial_hits.empty()) continue;
            // A key in both lists takes the maximum rate (0.0 floor). After tag_microbial_keys() the lists share no key, so this only
            // matters for untagged input (the unit tests), where an id shared by both indices is one species.
            tmp.clear();
            for (const auto& [sid, rate] : mh.viral_hits) tmp.emplace_back(sid, std::max(0.0f, rate));
            for (const auto& [sid, rate] : mh.microbial_hits) tmp.emplace_back(sid, std::max(0.0f, rate));
            std::sort(tmp.begin(), tmp.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            // Pre-filter: hits with hit_rate < min_prefilter_hr are almost always spurious
            // random minimizer collisions and must not enter the EM.
            Pattern pat;
            pat.reserve(tmp.size());
            for (size_t x = 0; x < tmp.size();) {
                float m = 0.0f;
                size_t z = x;
                for (; z < tmp.size() && tmp[z].first == tmp[x].first; ++z) m = std::max(m, tmp[z].second);
                const uint32_t sid = tmp[x].first;
                x = z;
                if (m <= 0.0f) continue;
                if (min_prefilter_hr > 0.0f && m < min_prefilter_hr) continue;
                pat.emplace_back(sid, m);
            }
            if (!pat.empty()) ++local[std::move(pat)];
        }
        PatCount pc;
        pc.reserve(local.size());
        while (!local.empty()) {
            auto nh = local.extract(local.begin());
            pc.emplace_back(std::move(nh.key()), nh.mapped());
        }
        std::sort(pc.begin(), pc.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        parts[t] = std::move(pc);
    };
    if (nt1 == 1) {
        build_part(0);
    } else {
        std::vector<std::thread> ts;
        for (int t = 0; t < nt1; ++t) ts.emplace_back(build_part, t);
        for (auto& th : ts) th.join();
    }
    auto merge2 = [](PatCount&& x, PatCount&& y) {
        PatCount out;
        out.reserve(x.size() + y.size());
        size_t i = 0, j = 0;
        while (i < x.size() && j < y.size()) {
            if (x[i].first < y[j].first) out.push_back(std::move(x[i++]));
            else if (y[j].first < x[i].first) out.push_back(std::move(y[j++]));
            else { out.push_back(std::move(x[i])); out.back().second += y[j].second; ++i; ++j; }
        }
        while (i < x.size()) out.push_back(std::move(x[i++]));
        while (j < y.size()) out.push_back(std::move(y[j++]));
        return out;
    };
    while (parts.size() > 1) {
        std::vector<PatCount> next((parts.size() + 1) / 2);
        std::vector<std::thread> ts;
        for (size_t q = 0; q < next.size(); ++q) {
            if (2 * q + 1 < parts.size()) ts.emplace_back([&, q]() { next[q] = merge2(std::move(parts[2 * q]), std::move(parts[2 * q + 1])); });
            else next[q] = std::move(parts[2 * q]);
        }
        for (auto& th : ts) th.join();
        parts = std::move(next);
    }
    const PatCount pattern_count = parts.empty() ? PatCount{} : std::move(parts[0]);

    // species index = rank of the species id
    std::vector<uint32_t>& idx_to_species = fit.idx_to_species;   // ascending species ids
    {
        std::unordered_set<uint32_t> ids;
        for (const auto& [pat, cnt] : pattern_count)
            for (const auto& [sid, rate] : pat) ids.insert(sid);
        idx_to_species.assign(ids.begin(), ids.end());
        std::sort(idx_to_species.begin(), idx_to_species.end());
    }
    struct Group { std::vector<std::pair<uint32_t, float>> hits; double count; };   // (species index, rate)
    std::vector<Group> groups(pattern_count.size());
    uint64_t n_reads_total = 0;
    for (const auto& [pat, cnt] : pattern_count) n_reads_total += cnt;
    em_detail::par_ranges(pattern_count.size(), (pattern_count.size() >= 20000) ? std::max(1, nthreads) : 1, nullptr, [&](size_t b, size_t e) {
        for (size_t gi = b; gi < e; ++gi) {
            Group& g = groups[gi];
            g.count = static_cast<double>(pattern_count[gi].second);
            g.hits.reserve(pattern_count[gi].first.size());
            for (const auto& [sid, rate] : pattern_count[gi].first) {
                const auto it = std::lower_bound(idx_to_species.begin(), idx_to_species.end(), sid);
                g.hits.emplace_back(static_cast<uint32_t>(it - idx_to_species.begin()), rate);
            }
        }
    });

    const uint32_t K = static_cast<uint32_t>(idx_to_species.size());
    const double   N = static_cast<double>(n_reads_total);   // number of nonhost reads
    if (K == 0 || N <= 0.0) return fit;
    fit.N = N;

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

    fit.sec_group = em_lap();
    fit.n_groups = groups.size();
    // Inverted lists: per species index its (pattern, rate) terms in pattern order; species ranges cut by term count
    const size_t G = groups.size();
    std::vector<size_t> inv_off(K + 1, 0);
    for (const auto& g : groups) for (const auto& kh : g.hits) ++inv_off[kh.first + 1];
    for (uint32_t k = 0; k < K; ++k) inv_off[k + 1] += inv_off[k];
    std::vector<std::pair<uint32_t, float>> inv(inv_off[K]);
    {
        std::vector<size_t> pos(inv_off.begin(), inv_off.end() - 1);
        for (size_t gi = 0; gi < G; ++gi)
            for (const auto& kh : groups[gi].hits) inv[pos[kh.first]++] = {static_cast<uint32_t>(gi), kh.second};
    }
    const int nt = (G >= 20000) ? std::max(1, nthreads) : 1;   // small inputs (the unit tests) stay serial
    std::vector<size_t> kcut(nt + 1, K);
    kcut[0] = 0;
    {
        uint32_t k = 0;
        for (int t = 1; t < nt; ++t) {
            const size_t want = inv_off[K] * t / nt;
            while (k < K && inv_off[k] < want) ++k;
            kcut[t] = k;
        }
    }
    // The patterns' terms and counts as flat arrays (the same terms in the same order as groups[gi].hits): the iterations read them
    // contiguously instead of through each pattern's own vector. The sums are the same operations in the same order.
    std::vector<size_t> gh_off(G + 1, 0);
    for (size_t gi = 0; gi < G; ++gi) gh_off[gi + 1] = gh_off[gi] + groups[gi].hits.size();
    std::vector<uint32_t> gh_k(gh_off[G]);
    std::vector<float> gh_r(gh_off[G]);
    std::vector<double> gcnt(G);
    em_detail::par_ranges(G, nt, nullptr, [&](size_t b, size_t e) {
        for (size_t gi = b; gi < e; ++gi) {
            size_t q = gh_off[gi];
            for (const auto& kh : groups[gi].hits) { gh_k[q] = kh.first; gh_r[q] = kh.second; ++q; }
            gcnt[gi] = groups[gi].count;
        }
    });
    std::vector<double> denom(G);
    auto compute_denoms = [&](const std::vector<double>& th) {
        em_detail::par_ranges(G, nt, nullptr, [&](size_t b, size_t e) {
            for (size_t gi = b; gi < e; ++gi) {
                double d = 0.0;
                for (size_t q = gh_off[gi], qe = gh_off[gi + 1]; q < qe; ++q) d += th[gh_k[q]] * static_cast<double>(gh_r[q]);
                denom[gi] = d;
            }
        });
    };

    // ── Step 2: Initialise θ uniformly ──────────────────────────────────────
    std::vector<double>& theta = fit.theta;
    theta.assign(K, 1.0 / static_cast<double>(K));
    std::vector<double> theta_new(K);

    // ── Step 3: EM iteration ─────────────────────────────────────────────────
    fit.sec_invert = em_lap();
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        // E-step: θ_new[k] = Σ over patterns (in order) of count * θ_k * h_ik / denom; patterns with denom <= 0 are skipped
        compute_denoms(theta);
        em_detail::par_ranges(K, nt, &kcut, [&](size_t kb, size_t ke) {
            for (size_t k = kb; k < ke; ++k) {
                double acc = 0.0;
                for (size_t q = inv_off[k]; q < inv_off[k + 1]; ++q) {
                    const double d = denom[inv[q].first];
                    if (d <= 0.0) continue;  // numerical safety
                    acc += gcnt[inv[q].first] * (theta[k] * static_cast<double>(inv[q].second) / d);
                }
                theta_new[k] = acc;
            }
        });

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
    fit.iterations = iter;
    fit.sec_iterate = em_lap();

    // ── Step 4: Compute per-species stats ────────────────────────────────────
    // mean_hit_rate[k] = Σ_i (h_ik * w_ik) / Σ_i w_ik   (posterior-weighted mean)
    fit.hit_rate_sum.assign(K, 0.0);
    fit.weight_sum.assign(K, 0.0);
    compute_denoms(theta);
    em_detail::par_ranges(K, nt, &kcut, [&](size_t kb, size_t ke) {
        for (size_t k = kb; k < ke; ++k) {
            double hs = 0.0, ws = 0.0;
            for (size_t q = inv_off[k]; q < inv_off[k + 1]; ++q) {
                const double d = denom[inv[q].first];
                if (d <= 0.0) continue;
                const double h = static_cast<double>(inv[q].second);
                const double w = groups[inv[q].first].count * (theta[k] * h / d);
                hs += h * w;
                ws += w;
            }
            fit.hit_rate_sum[k] = hs;
            fit.weight_sum[k]   = ws;
        }
    });
    fit.sec_stats = em_lap();
    return fit;
}

// ── Step 5: the report of one fit: filter and sort ───────────────────────────
inline std::vector<SpeciesAbundance> em_report(
        const EmFit& fit,
        const std::unordered_map<uint32_t, NonHostCategory>& kingdom_map,
        float    min_abundance    = 1e-4f,
        float    min_report_hr    = 0.30f,
        uint32_t min_abs_reads    = 0) {
    std::vector<SpeciesAbundance> result;
    const uint32_t K = static_cast<uint32_t>(fit.theta.size());
    if (K == 0) return result;
    const std::vector<double>& theta = fit.theta;
    const std::vector<double>& hit_rate_sum = fit.hit_rate_sum;
    const std::vector<double>& weight_sum = fit.weight_sum;
    const std::vector<uint32_t>& idx_to_species = fit.idx_to_species;
    const double N = fit.N;
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

    // Sort descending by relative_abundance
    std::sort(result.begin(), result.end(),
              [](const SpeciesAbundance& a, const SpeciesAbundance& b) {
                  if (a.relative_abundance != b.relative_abundance)
                      return a.relative_abundance > b.relative_abundance;
                  return a.species_id < b.species_id;
              });

    return result;
}

inline std::vector<SpeciesAbundance> em_deconvolve(
        const std::vector<NonHostScreener::MultiHit>& multi_hits,
        const std::unordered_map<uint32_t, NonHostCategory>& kingdom_map,
        int      max_iter         = 500,
        float    tolerance        = 1e-6f,
        float    min_abundance    = 1e-4f,
        float    min_prefilter_hr = 0.05f,
        float    min_report_hr    = 0.30f,
        int*     out_iterations   = nullptr,
        uint32_t min_abs_reads    = 0,
        int      nthreads         = 1) {
    const EmFit fit = em_fit(multi_hits, max_iter, tolerance, min_prefilter_hr, nthreads);
    if (out_iterations) *out_iterations = fit.iterations;
    return em_report(fit, kingdom_map, min_abundance, min_report_hr, min_abs_reads);
}

// ── Kingdom map builder ───────────────────────────────────────────────────────

// Build a species key → NonHostCategory map from a batch of MultiHit results.
// Keys in viral_hits → VIRAL. Keys in microbial_hits only → FUNGAL if their raw id is in fungal_species, else BACTERIAL.
// Call this before em_deconvolve() when no external map is available.
//
// With tagged input (tag_microbial_keys) viral and microbial keys are disjoint, so each key has one kingdom. With untagged
// input an id shared by both indices is VIRAL whatever the read order (as species_name_lookup resolved the viral name first),
// which keeps the map a function of the set of reads and not of their order.
inline std::unordered_map<uint32_t, NonHostCategory> build_kingdom_map(
        const std::vector<NonHostScreener::MultiHit>& multi_hits,
        const std::unordered_set<uint32_t>& fungal_species = {}) {
    std::unordered_map<uint32_t, NonHostCategory> kmap;
    for (const auto& mh : multi_hits)
        for (const auto& [sid, rate] : mh.viral_hits)
            kmap[sid] = NonHostCategory::VIRAL;
    for (const auto& mh : multi_hits)
        for (const auto& [sid, rate] : mh.microbial_hits)
            kmap.emplace(sid, fungal_species.count(raw_species_id(sid)) ? NonHostCategory::FUNGAL
                                                                        : NonHostCategory::BACTERIAL);   // never replaces VIRAL
    return kmap;
}

} // namespace nonhost
} // namespace singlet
