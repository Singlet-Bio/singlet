// test_nonhost_aligner.cpp — Unit tests for nonhost_aligner.h
//
// Tests SpeciesRefIndex, longest_chain(), and NonHostAligner end-to-end.
// All tests are self-contained with no external files (FASTA data written to /tmp).
//
// Build:
//   ninja -C build test_nonhost_aligner && ./build/test_nonhost_aligner
//
// Tests:
//   1. single_genome_align   : reads from a single-contig reference → aligned_reads > 0
//   2. multi_contig          : reads from 2-contig reference → both contigs contribute coverage
//   3. no_match              : poly-N reads against ACGT reference → aligned_reads == 0
//   4. abundance_filter      : species at 0.0005 abundance → skipped; species at 0.01 → aligned
//   5. chain_gap             : seeds within 500 bp → chained; seeds > 500 bp apart → not chained

#include "singlet/pileup/nonhost/nonhost_aligner.h"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace singlet::nonhost;

// ── Test infrastructure ───────────────────────────────────────────────────────

static int g_passed = 0, g_failed = 0;

static void PASS(const std::string& name) {
    ++g_passed;
    std::cerr << "PASS: " << name << "\n";
}
static void FAIL(const std::string& name, const char* msg) {
    ++g_failed;
    std::cerr << "FAIL: " << name << " — " << msg << "\n";
}
#define CHECK(name, cond, msg) do { if (cond) PASS(name); else FAIL(name, msg); } while (0)

// ── FASTA helpers ─────────────────────────────────────────────────────────────

// Write a simple 2-record FASTA to path.
static void write_fasta(const std::string& path,
                        const std::vector<std::pair<std::string,std::string>>& recs) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream f(path);
    for (const auto& [hdr, seq] : recs)
        f << ">" << hdr << "\n" << seq << "\n";
}

// Generate a pseudo-random DNA sequence of length `n` (deterministic LCG).
// Produces diverse k-mers so that the minimizer index has many distinct entries.
static std::string make_ref_seq(size_t n, uint64_t seed = 12345678901234567ULL) {
    static const char bases[] = "ACGT";
    std::string s;
    s.reserve(n);
    uint64_t state = seed;
    for (size_t i = 0; i < n; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        s.push_back(bases[(state >> 32) & 3]);
    }
    return s;
}

// Extract a substring of `source` starting at `pos`, length `len`.
// Used to produce reads that are guaranteed to match the reference.
static std::string make_read(const std::string& source, size_t pos, size_t len) {
    if (pos + len > source.size()) len = source.size() - pos;
    return source.substr(pos, len);
}

// ── Test 1: single_genome_align ───────────────────────────────────────────────
// One 800-bp contig; 15 reads of 120 bp each extracted from the reference.
// All reads should produce ≥3 chained seeds → aligned_reads == 15.

static void test_single_genome_align() {
    const std::string test_name = "single_genome_align";

    // Build reference and write to /tmp
    std::string ref_seq = make_ref_seq(800);
    const std::string fna_root = "/tmp/nonhost_aligner_test1/nonhost/viral_genomes";
    write_fasta(fna_root + "/species_7.fna", {{"contig1", ref_seq}});

    // Produce 15 reads spaced across the reference (120 bp each)
    std::vector<std::string> reads;
    for (int i = 0; i < 15; ++i)
        reads.push_back(make_read(ref_seq, static_cast<size_t>(i * 40), 120));

    // Build EM abundances: species 7 (VIRAL) at abundance 0.05
    std::vector<SpeciesAbundance> em = {
        {7, NonHostCategory::VIRAL, 0.05f, 15, 0.8f}
    };

    NonHostAligner aligner("/tmp/nonhost_aligner_test1");
    auto results = aligner.align(reads, em);

    CHECK(test_name, !results.empty(), "expected ≥1 result");
    if (!results.empty()) {
        CHECK(test_name, results[0].aligned_reads > 0,
              "expected aligned_reads > 0 for reads from reference");
        CHECK(test_name, results[0].mapping_rate > 0.0f,
              "expected mapping_rate > 0");
    }
}

// ── Test 2: multi_contig ──────────────────────────────────────────────────────
// Reference FASTA has 2 contigs (300 bp each); reads drawn from both.
// Coverage should reflect seeds from both contigs; aligned_reads > 0.

static void test_multi_contig() {
    const std::string test_name = "multi_contig";

    std::string ref1 = make_ref_seq(300);
    // Second contig: different seed so its minimizers are distinct from contig1
    std::string ref2 = make_ref_seq(300, 99887766554433221ULL);

    const std::string fna_root = "/tmp/nonhost_aligner_test2/nonhost/bacterial_genomes";
    write_fasta(fna_root + "/species_3.fna", {{"c1", ref1}, {"c2", ref2}});

    // 10 reads from contig1, 10 from contig2
    std::vector<std::string> reads;
    for (int i = 0; i < 10; ++i)
        reads.push_back(make_read(ref1, static_cast<size_t>(i * 20), 100));
    for (int i = 0; i < 10; ++i)
        reads.push_back(make_read(ref2, static_cast<size_t>(i * 20), 100));

    std::vector<SpeciesAbundance> em = {
        {3, NonHostCategory::BACTERIAL, 0.10f, 20, 0.7f}
    };

    NonHostAligner aligner("/tmp/nonhost_aligner_test2");
    auto results = aligner.align(reads, em);

    CHECK(test_name, !results.empty(), "expected 1 result");
    if (!results.empty()) {
        CHECK(test_name, results[0].aligned_reads > 0,
              "expected aligned_reads > 0 for reads from 2-contig reference");
        CHECK(test_name, results[0].coverage > 0.0f,
              "expected coverage > 0");
    }
}

// ── Test 3: no_match ──────────────────────────────────────────────────────────
// Reference: ACGT repeat (valid minimizers).
// Reads: poly-N sequences (all N → INVALID_KMER, no minimizers extracted).
// Expected: aligned_reads == 0.

static void test_no_match() {
    const std::string test_name = "no_match";

    std::string ref_seq = make_ref_seq(500);
    const std::string fna_root = "/tmp/nonhost_aligner_test3/nonhost/viral_genomes";
    write_fasta(fna_root + "/species_1.fna", {{"ref", ref_seq}});

    // Reads that are all-N — extract_minimizers returns empty for all-N
    std::vector<std::string> reads(10, std::string(100, 'N'));

    std::vector<SpeciesAbundance> em = {
        {1, NonHostCategory::VIRAL, 0.05f, 10, 0.5f}
    };

    NonHostAligner aligner("/tmp/nonhost_aligner_test3");
    auto results = aligner.align(reads, em);

    CHECK(test_name, !results.empty(), "expected 1 result (FASTA exists)");
    if (!results.empty()) {
        CHECK(test_name, results[0].aligned_reads == 0,
              "expected aligned_reads == 0 for all-N reads");
        CHECK(test_name, results[0].mapping_rate == 0.0f,
              "expected mapping_rate == 0");
    }
}

// ── Test 4: abundance_filter ──────────────────────────────────────────────────
// Two species: one at 0.0005 (below 0.001 threshold), one at 0.01 (above).
// Only the species above threshold should produce a result.

static void test_abundance_filter() {
    const std::string test_name = "abundance_filter";

    std::string ref_seq = make_ref_seq(400);
    const std::string fna_root = "/tmp/nonhost_aligner_test4/nonhost/viral_genomes";
    // Write FASTAs for both species
    write_fasta(fna_root + "/species_10.fna", {{"ref10", ref_seq}});
    write_fasta(fna_root + "/species_20.fna", {{"ref20", ref_seq}});

    // Reads drawn from the reference sequence
    std::vector<std::string> reads;
    for (int i = 0; i < 8; ++i)
        reads.push_back(make_read(ref_seq, static_cast<size_t>(i * 30), 100));

    // species 10: abundance 0.0005 → below threshold → must be skipped
    // species 20: abundance 0.01   → above threshold → must be aligned
    std::vector<SpeciesAbundance> em = {
        {20, NonHostCategory::VIRAL, 0.01f,   8, 0.6f},
        {10, NonHostCategory::VIRAL, 0.0005f, 0, 0.0f},
    };

    NonHostAligner aligner("/tmp/nonhost_aligner_test4");
    auto results = aligner.align(reads, em);

    CHECK(test_name, results.size() == 1,
          "expected exactly 1 result (species 10 filtered by abundance)");
    if (!results.empty()) {
        CHECK(test_name, results[0].species_id == 20,
              "expected species_id == 20 (the one above threshold)");
    }
}

// ── Test 5: chain_gap ─────────────────────────────────────────────────────────
// Tests longest_chain() directly and via the aligner:
//   a) sorted positions [10, 20, 30, 40, 50]  → gap=10   → chain=5 (≥3 → aligned)
//   b) sorted positions [10, 600, 1200]        → gap=590+ → chain=1 (< 3 → not aligned)

static void test_chain_gap() {
    const std::string test_name = "chain_gap";

    // Part A: tight chain — 5 seeds within 10 bp each → longest chain = 5
    {
        std::vector<uint32_t> close_seeds = {10, 20, 30, 40, 50};
        int chain = longest_chain(close_seeds, 500);
        CHECK(test_name, chain >= 3,
              "close seeds (gap=10) should form chain ≥ 3");
    }

    // Part B: distant seeds — 3 seeds >500 bp apart → longest chain = 1
    {
        std::vector<uint32_t> far_seeds = {10, 600, 1200};
        int chain = longest_chain(far_seeds, 500);
        CHECK(test_name, chain < 3,
              "far seeds (gap=590+) should NOT form chain ≥ 3");
    }

    // Part C: via aligner — build a 2000-bp reference; create a read whose
    // minimizers will land near position 0 only (extract from a short prefix).
    // A 31-bp read (just above K+W-1=31, the minimum for any minimizer) should
    // produce too few seeds (<3) to trigger alignment.
    {
        std::string ref_seq = make_ref_seq(2000);
        const std::string fna_root = "/tmp/nonhost_aligner_test5/nonhost/viral_genomes";
        write_fasta(fna_root + "/species_5.fna", {{"ref", ref_seq}});

        // Very short read: length = K+W-1 = 31 → exactly 1 minimizer → chain=1 < 3
        std::vector<std::string> reads_short = {make_read(ref_seq, 0, 31)};

        std::vector<SpeciesAbundance> em = {
            {5, NonHostCategory::VIRAL, 0.05f, 1, 0.5f}
        };

        NonHostAligner aligner("/tmp/nonhost_aligner_test5");
        auto results = aligner.align(reads_short, em);

        CHECK(test_name, !results.empty(), "expected 1 result for known FASTA");
        if (!results.empty()) {
            CHECK(test_name, results[0].aligned_reads == 0,
                  "31-bp read (1 minimizer) should not satisfy chain ≥ 3");
        }

        // Now a long read (150 bp) from the SAME reference should be aligned
        std::vector<std::string> reads_long = {make_read(ref_seq, 0, 150)};
        auto results2 = aligner.align(reads_long, em);
        CHECK(test_name, !results2.empty() && results2[0].aligned_reads > 0,
              "150-bp read should produce aligned_reads > 0");
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cerr << "=== test_nonhost_aligner ===\n";

    test_single_genome_align();
    test_multi_contig();
    test_no_match();
    test_abundance_filter();
    test_chain_gap();

    std::cerr << "\nResults: " << g_passed << " passed, " << g_failed << " failed\n";
    return (g_failed == 0) ? 0 : 1;
}
