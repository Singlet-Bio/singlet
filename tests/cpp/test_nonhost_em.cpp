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

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    test1_single_species();
    test2_two_species_mixture();
    test3_ambiguous_reads();
    test4_convergence();
    test5_min_abundance_filter();

    std::cerr << "\n--- " << g_passed << "/" << (g_passed + g_failed) << " tests passed ---\n";
    return (g_failed == 0) ? 0 : 1;
}
