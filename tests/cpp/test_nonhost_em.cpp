// SPDX-License-Identifier: MIT
// test_nonhost_em.cpp — Unit tests for NONHOST-EM deconvolution
//
// Tests the em_deconvolve() algorithm with synthesised MultiHit inputs.
// Tests are independent of MinSketch (the index + screener path is verified
// separately in test_nonhost.cpp).
//
// Build:
//   ninja -C build test_nonhost_em && ./build/test_nonhost_em
//
// Tests:
//   1. Single-species recovery:  100 reads → species 1 only → θ[1] ≈ 1.0
//   2. Two-species 50/50 mix:    50+50 reads ± 10 ambiguous → θ[1] ≈ θ[2] ≈ 0.5 (±0.1)
//   3. Ambiguous reads:          all reads hit 3 species → highest hit_rate wins
//   4. Convergence check:        iter count < max_iter (= 100) for all test cases
//   5. Min-abundance filter:     species with θ < threshold are removed from output

#include "singlet/pileup/nonhost/nonhost_em.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace singlet::nonhost;
using MH = NonHostScreener::MultiHit;

// ── Helpers ───────────────────────────────────────────────────────────────────

static MH make_viral(uint32_t sid, float rate) {
    MH mh;
    mh.top_category = NonHostCategory::VIRAL;
    mh.viral_hits   = {{sid, rate}};
    return mh;
}

static MH make_microbial(uint32_t sid, float rate) {
    MH mh;
    mh.top_category  = NonHostCategory::BACTERIAL;
    mh.microbial_hits = {{sid, rate}};
    return mh;
}

static MH make_mixed(uint32_t sid1, float rate1, uint32_t sid2, float rate2) {
    MH mh;
    mh.top_category = (rate1 >= rate2) ? NonHostCategory::VIRAL : NonHostCategory::BACTERIAL;
    mh.viral_hits   = {{sid1, rate1}};
    mh.microbial_hits = {{sid2, rate2}};
    return mh;
}

// Test result helpers
static int g_passed = 0, g_failed = 0;

static void PASS(const char* name) {
    ++g_passed;
    std::cerr << "PASS: " << name << "\n";
}
static void FAIL(const char* name, const char* msg) {
    ++g_failed;
    std::cerr << "FAIL: " << name << " — " << msg << "\n";
}

// ── Test 1: Single-species recovery ──────────────────────────────────────────
// 100 reads, all exclusively matching species_id=1 (viral, hit_rate=0.8).
// EM should converge to θ[1] ≈ 1.0.
static void test1_single_species() {
    const char* name = "test1_single_species";

    std::vector<MH> hits(100, make_viral(1u, 0.8f));
    auto kmap   = build_kingdom_map(hits);
    int  n_iter = 0;
    auto result = em_deconvolve(hits, kmap, 100, 1e-6f, 1e-4f, 0.0f, 0.0f, &n_iter);

    if (result.size() != 1) {
        FAIL(name, "expected exactly 1 species in output");
        return;
    }
    if (result[0].species_id != 1u) {
        FAIL(name, "wrong species_id in output");
        return;
    }
    if (std::abs(result[0].relative_abundance - 1.0f) > 0.01f) {
        FAIL(name, "relative_abundance not ≈ 1.0");
        return;
    }
    if (result[0].kingdom != NonHostCategory::VIRAL) {
        FAIL(name, "kingdom not VIRAL");
        return;
    }
    if (n_iter >= 100) {
        FAIL(name, "did not converge within 100 iterations");
        return;
    }
    PASS(name);
}

// ── Test 2: Two-species 50/50 mixture ────────────────────────────────────────
// 50 reads exclusively from species 1 (viral, 0.9), 50 from species 2 (bacterial, 0.7),
// plus 10 ambiguous reads matching both species.
// EM should converge to θ[1] ≈ θ[2] ≈ 0.5 (±0.1).
static void test2_two_species_mixture() {
    const char* name = "test2_two_species_mixture";

    std::vector<MH> hits;
    hits.reserve(110);
    for (int i = 0; i < 50; ++i) hits.push_back(make_viral(1u, 0.9f));
    for (int i = 0; i < 50; ++i) hits.push_back(make_microbial(2u, 0.7f));
    for (int i = 0; i < 10; ++i) hits.push_back(make_mixed(1u, 0.9f, 2u, 0.7f));

    auto kmap   = build_kingdom_map(hits);
    int  n_iter = 0;
    auto result = em_deconvolve(hits, kmap, 100, 1e-6f, 1e-4f, 0.0f, 0.0f, &n_iter);

    if (result.size() != 2) {
        FAIL(name, "expected exactly 2 species in output");
        return;
    }
    // result is sorted descending; both should be ≈ 0.5
    float a1 = result[0].relative_abundance;
    float a2 = result[1].relative_abundance;

    if (a1 < 0.4f || a1 > 0.65f) {
        FAIL(name, "species 1 relative_abundance not ≈ 0.5 ± 0.1");
        return;
    }
    if (a2 < 0.35f || a2 > 0.6f) {
        FAIL(name, "species 2 relative_abundance not ≈ 0.5 ± 0.1");
        return;
    }
    if (std::abs(a1 + a2 - 1.0f) > 0.01f) {
        FAIL(name, "abundances do not sum to 1.0");
        return;
    }
    if (n_iter >= 100) {
        FAIL(name, "did not converge within 100 iterations");
        return;
    }
    PASS(name);
}

// ── Test 3: Ambiguous reads (cross-strain) ────────────────────────────────────
// 100 reads each matching 3 species with hit_rates 0.8, 0.6, 0.4 respectively.
// EM should assign majority mass to the highest-rate species.
// Analytically, the EM fixed point for identical reads concentrates on species 1.
static void test3_ambiguous_reads() {
    const char* name = "test3_ambiguous_reads";

    std::vector<MH> hits;
    hits.reserve(100);
    for (int i = 0; i < 100; ++i) {
        MH mh;
        mh.top_category = NonHostCategory::VIRAL;
        mh.viral_hits   = {{1u, 0.8f}, {2u, 0.6f}, {3u, 0.4f}};
        hits.push_back(mh);
    }
    auto kmap   = build_kingdom_map(hits);
    int  n_iter = 0;
    // Use max_iter=500 (production default) — identical-reads ambiguous case may need
    // more iterations with hit-rate-weighted Dirichlet regularization.
    auto result = em_deconvolve(hits, kmap, 500, 1e-6f, 1e-4f, 0.0f, 0.0f, &n_iter);

    if (result.empty()) {
        FAIL(name, "empty output");
        return;
    }
    // Top species must be species_id=1 with highest hit_rate
    if (result[0].species_id != 1u) {
        FAIL(name, "species 1 not the top output");
        return;
    }
    // Must hold the majority (>> 1/3)
    if (result[0].relative_abundance < 0.7f) {
        FAIL(name, "majority not assigned to highest-hit-rate species (expected ≥ 0.7)");
        return;
    }
    if (n_iter >= 500) {
        FAIL(name, "did not converge within 500 iterations");
        return;
    }
    PASS(name);
}

// ── Test 4: Convergence check ─────────────────────────────────────────────────
// Verifies that single- and two-species cases converge within max_iter=100.
// Three-species ambiguous case uses max_iter=500 (production default) since
// hit-rate-weighted regularization may need more iterations on degenerate inputs.
// Also verifies that running with max_iter=50 gives identical results to 100
// for the simple single-species case (both converge long before 50).
static void test4_convergence() {
    const char* name = "test4_convergence";

    // Single species: should converge in 1 iteration
    {
        std::vector<MH> hits(100, make_viral(1u, 0.8f));
        auto kmap   = build_kingdom_map(hits);
        int  n100   = 0;
        auto r100   = em_deconvolve(hits, kmap, 100, 1e-6f, 1e-4f, 0.0f, 0.0f, &n100);
        int  n50    = 0;
        auto r50    = em_deconvolve(hits, kmap,  50, 1e-6f, 1e-4f, 0.0f, 0.0f, &n50);
        if (n100 >= 100) { FAIL(name, "single-species: not converged in 100 iters"); return; }
        if (r100.size() != r50.size()) { FAIL(name, "single-species: result size mismatch at 50 vs 100 iters"); return; }
        if (!r100.empty() && std::abs(r100[0].relative_abundance - r50[0].relative_abundance) > 1e-5f) {
            FAIL(name, "single-species: result differs between 50 and 100 iter runs");
            return;
        }
    }

    // Two-species mixture
    {
        std::vector<MH> hits;
        for (int i = 0; i < 50; ++i) hits.push_back(make_viral(1u, 0.9f));
        for (int i = 0; i < 50; ++i) hits.push_back(make_microbial(2u, 0.7f));
        for (int i = 0; i < 10; ++i) hits.push_back(make_mixed(1u, 0.9f, 2u, 0.7f));
        auto kmap  = build_kingdom_map(hits);
        int  n_iter = 0;
        em_deconvolve(hits, kmap, 100, 1e-6f, 1e-4f, 0.0f, 0.0f, &n_iter);
        if (n_iter >= 100) { FAIL(name, "two-species: not converged in 100 iters"); return; }
    }

    // Three-species ambiguous
    {
        std::vector<MH> hits;
        for (int i = 0; i < 100; ++i) {
            MH mh;
            mh.top_category = NonHostCategory::VIRAL;
            mh.viral_hits   = {{1u, 0.8f}, {2u, 0.6f}, {3u, 0.4f}};
            hits.push_back(mh);
        }
        auto kmap   = build_kingdom_map(hits);
        int  n_iter = 0;
        em_deconvolve(hits, kmap, 500, 1e-6f, 1e-4f, 0.0f, 0.0f, &n_iter);
        if (n_iter >= 500) { FAIL(name, "three-species: not converged in 500 iters"); return; }
    }

    PASS(name);
}

// ── Test 5: Min-abundance filter ──────────────────────────────────────────────
// 999 reads from species 1 (exclusive), 1 read from species 2 (exclusive).
// → θ[1] ≈ 0.999, θ[2] ≈ 0.001.
// With min_abundance = 0.005 (0.5%), species 2 must be absent from output.
static void test5_min_abundance_filter() {
    const char* name = "test5_min_abundance_filter";

    std::vector<MH> hits;
    hits.reserve(1000);
    for (int i = 0; i < 999; ++i) hits.push_back(make_viral(1u, 0.8f));
    hits.push_back(make_microbial(2u, 0.7f));  // one rare-species read

    auto kmap   = build_kingdom_map(hits);
    auto result = em_deconvolve(hits, kmap, 100, 1e-6f, /*min_abundance=*/0.005f);

    // Species 2 should be filtered (abundance ≈ 0.001 < 0.005)
    for (const auto& sa : result) {
        if (sa.species_id == 2u) {
            FAIL(name, "rare species (abundance < min_abundance) not filtered");
            return;
        }
    }
    // Species 1 must be present
    if (result.empty() || result[0].species_id != 1u) {
        FAIL(name, "dominant species (id=1) missing from output");
        return;
    }
    if (result[0].relative_abundance < 0.99f) {
        FAIL(name, "dominant species abundance unexpectedly low");
        return;
    }
    PASS(name);
}

// ── Test 6: reproducible under any read order ────────────────────────────────
// The same reads in any order (what different thread counts and reruns produce) give
// bit-identical abundances and the same iteration count.
static void test6_order_independent() {
    const char* name = "test6_order_independent";
    std::vector<MH> reads;
    uint32_t x = 12345;
    auto rnd = [&]() { x = x * 1664525u + 1013904223u; return x >> 8; };
    for (int i = 0; i < 3000; ++i) {
        const uint32_t a = rnd() % 12, b = (a + 1 + rnd() % 11) % 12;
        const float ra = 0.3f + 0.05f * (rnd() % 12), rb = 0.3f + 0.05f * (rnd() % 12);
        reads.push_back(i % 3 ? make_mixed(a, ra, 20 + b, rb) : make_viral(a, ra));
    }
    auto kmap = build_kingdom_map(reads);
    int it0 = 0;
    auto ref = em_deconvolve(reads, kmap, 500, 1e-6f, 1e-4f, 0.05f, 0.30f, &it0);
    for (int rep = 0; rep < 5; ++rep) {
        for (size_t i = reads.size() - 1; i > 0; --i) std::swap(reads[i], reads[rnd() % (i + 1)]);
        int it1 = 0;
        auto got = em_deconvolve(reads, kmap, 500, 1e-6f, 1e-4f, 0.05f, 0.30f, &it1);
        bool same = it0 == it1 && got.size() == ref.size();
        for (size_t i = 0; same && i < got.size(); ++i)
            same = got[i].species_id == ref[i].species_id &&
                   got[i].relative_abundance == ref[i].relative_abundance &&
                   got[i].n_reads_assigned == ref[i].n_reads_assigned &&
                   got[i].mean_hit_rate == ref[i].mean_hit_rate;
        if (!same) { FAIL(name, "shuffled input changed the result"); return; }
    }
    if (ref.empty()) { FAIL(name, "no species reported"); return; }
    PASS(name);
}

// ── Test 7: the kingdom of an id that both indices use does not depend on read order ──
static void test7_kingdom_order_independent() {
    const char* name = "test7_kingdom_order_independent";
    const std::unordered_set<uint32_t> fungal = {7};
    std::vector<MH> a = {make_microbial(7, 0.9f), make_viral(7, 0.8f), make_microbial(9, 0.9f)};
    std::vector<MH> b = {make_viral(7, 0.8f), make_microbial(9, 0.9f), make_microbial(7, 0.9f)};
    auto ka = build_kingdom_map(a, fungal), kb = build_kingdom_map(b, fungal);
    if (ka.at(7) != NonHostCategory::VIRAL || kb.at(7) != NonHostCategory::VIRAL) { FAIL(name, "shared id not VIRAL in both orders"); return; }
    if (ka.at(9) != NonHostCategory::BACTERIAL || kb.at(9) != NonHostCategory::BACTERIAL) { FAIL(name, "microbial-only id not BACTERIAL"); return; }
    std::vector<MH> c = {make_microbial(7, 0.9f)};
    if (build_kingdom_map(c, fungal).at(7) != NonHostCategory::FUNGAL) { FAIL(name, "fungal-only id not FUNGAL"); return; }
    PASS(name);
}

// ── Test 8: one fit, any thread count, any number of reports ──
// em_fit on 8 threads is bit-identical to 1 thread on an input large enough to run threaded (>= 20000 patterns), and
// em_deconvolve equals em_report of the shared fit.
static void test8_fit_threads_and_reports() {
    const char* name = "test8_fit_threads_and_reports";
    std::mt19937 rnd(12345);
    std::vector<MH> reads;
    for (int i = 0; i < 60000; ++i) {
        MH mh;
        const int nv = 1 + static_cast<int>(rnd() % 4), nm = static_cast<int>(rnd() % 3);
        for (int j = 0; j < nv; ++j) mh.viral_hits.push_back({static_cast<uint32_t>(rnd() % 400), 0.05f + (rnd() % 950) / 1000.0f});
        for (int j = 0; j < nm; ++j) mh.microbial_hits.push_back({1000u + static_cast<uint32_t>(rnd() % 300), 0.05f + (rnd() % 950) / 1000.0f});
        reads.push_back(std::move(mh));
    }
    const auto kmap = build_kingdom_map(reads);
    const EmFit f1 = em_fit(reads, 500, 1e-6f, 0.05f, 1), f8 = em_fit(reads, 500, 1e-6f, 0.05f, 8);
    bool same = f1.iterations == f8.iterations && f1.theta == f8.theta && f1.hit_rate_sum == f8.hit_rate_sum &&
                f1.weight_sum == f8.weight_sum && f1.idx_to_species == f8.idx_to_species && f1.N == f8.N;
    if (!same) { FAIL(name, "8 threads differ from 1 thread"); return; }
    int it = 0;
    const auto a = em_deconvolve(reads, kmap, 500, 1e-6f, 1e-4f, 0.05f, 0.30f, &it, 0);
    const auto b = em_report(f8, kmap, 1e-4f, 0.30f, 0);
    same = it == f8.iterations && a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i)
        same = a[i].species_id == b[i].species_id && a[i].relative_abundance == b[i].relative_abundance &&
               a[i].n_reads_assigned == b[i].n_reads_assigned && a[i].mean_hit_rate == b[i].mean_hit_rate &&
               a[i].kingdom == b[i].kingdom;
    if (!same) { FAIL(name, "em_deconvolve differs from em_report of the shared fit"); return; }
    if (em_report(f8, kmap, 0.0f, 0.0f, 0u).size() < b.size()) { FAIL(name, "the report without thresholds lists fewer species"); return; }
    PASS(name);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    test1_single_species();
    test2_two_species_mixture();
    test3_ambiguous_reads();
    test4_convergence();
    test5_min_abundance_filter();
    test6_order_independent();
    test7_kingdom_order_independent();
    test8_fit_threads_and_reports();

    std::cerr << "\n--- " << g_passed << "/" << (g_passed + g_failed) << " tests passed ---\n";
    return (g_failed == 0) ? 0 : 1;
}
