// test_nonhost_unmapped_capture.cpp — Tests for NONHOST-UNMAPPED-CAPTURE task
//
// Tests the NonHostScreener pipeline on FASTQ-like reads:
//   - Half the reads match a viral sequence in the test index
//   - Half are random (no match)
//   - Verify viral_hits ≈ 5 for a batch of 10 reads
//   - Verify nonhost_summary.json structure is written correctly
//
// This test exercises the classify_batch() path that is wired into the
// singlify pipeline after STAR alignment for unmapped read screening.
//
// Run via ctest:
//   ctest -R nonhost_unmapped_capture --output-on-failure

#include "singlet/pileup/nonhost/min_sketch.h"
#include "singlet/pileup/nonhost/nonhost_screener.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace singlet::nonhost;

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string make_seq(const char* pattern, int length) {
    std::string s;
    s.reserve(static_cast<size_t>(length));
    const int plen = static_cast<int>(std::strlen(pattern));
    for (int i = 0; i < length; ++i) s += pattern[i % plen];
    return s;
}

// Write a tiny FASTQ file with the given sequences.
static void write_fastq(const char* path, const std::vector<std::string>& reads) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error(std::string("cannot write FASTQ: ") + path);
    for (size_t i = 0; i < reads.size(); ++i) {
        f << "@read_" << i << "\n"
          << reads[i] << "\n"
          << "+\n"
          << std::string(reads[i].size(), 'I') << "\n";  // fake quality scores
    }
}

// Parse a FASTQ file and return the sequences.
static std::vector<std::string> read_fastq(const char* path) {
    std::vector<std::string> seqs;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {  // @name
        if (line.empty() || line[0] != '@') continue;
        std::string seq;
        std::getline(f, seq);        // sequence
        std::string plus, qual;
        std::getline(f, plus);       // +
        std::getline(f, qual);       // quality
        if (!seq.empty()) seqs.push_back(seq);
    }
    return seqs;
}

// Write nonhost_summary.json from classification results (mirrors singlify pipeline logic).
static void write_nonhost_summary_json(
        const char* json_path,
        size_t total_reads,
        size_t screened_reads,
        size_t viral_hits,
        size_t bacterial_hits,
        size_t fungal_hits,
        size_t unknown_hits) {
    double vf = screened_reads > 0 ? static_cast<double>(viral_hits) / screened_reads : 0.0;
    double bf = screened_reads > 0 ? static_cast<double>(bacterial_hits) / screened_reads : 0.0;
    double ff = screened_reads > 0 ? static_cast<double>(fungal_hits) / screened_reads : 0.0;

    std::ofstream f(json_path);
    f << "{\n"
      << "  \"total_unmapped_reads\": " << total_reads << ",\n"
      << "  \"screened_reads\": " << screened_reads << ",\n"
      << "  \"viral_hits\": " << viral_hits << ",\n"
      << "  \"bacterial_hits\": " << bacterial_hits << ",\n"
      << "  \"fungal_hits\": " << fungal_hits << ",\n"
      << "  \"unknown_hits\": " << unknown_hits << ",\n"
      << "  \"viral_fraction\": " << vf << ",\n"
      << "  \"bacterial_fraction\": " << bf << ",\n"
      << "  \"fungal_fraction\": " << ff << "\n"
      << "}\n";
}

int main() {
    int passed = 0, failed = 0;

    auto PASS = [&](const char* name) {
        ++passed;
        std::cerr << "PASS: " << name << "\n";
    };
    auto FAIL = [&](const char* name, const char* msg) {
        ++failed;
        std::cerr << "FAIL: " << name << " — " << msg << "\n";
    };

    // Reference viral sequence (long enough for multiple minimizer windows).
    // This will be embedded in both the index reference and 5 of the 10 reads.
    std::string viral_ref = make_seq("CGCGCGCGCGCGCGCG", 50)
                          + make_seq("ACGTACGTACGTACGT", 80)  // distinct fingerprint
                          + make_seq("CGCGCGCGCGCGCGCG", 50);

    // "Random" sequence — does NOT share minimizers with viral_ref
    // (uses a completely different repeating pattern)
    std::string random_seq = make_seq("TTTAAGGGCCCAAATTT", 180);

    // ── Test 1: classify_batch detects 5 viral hits, 5 unknowns ─────────────
    {
        // Build a minimal viral index from viral_ref
        MinSketchIndex viral_idx = MinSketchIndex::build_from_sequences(
            {{viral_ref, 42u}}, 21, 11, 50000u);
        MinSketchIndex micro_idx = MinSketchIndex::build_from_sequences({}, 21, 11);
        NonHostScreener screener(viral_idx, micro_idx, 0.01f);

        // 10 reads: indices 0-4 are viral, 5-9 are random
        std::vector<std::string> reads;
        reads.reserve(10);
        for (int i = 0; i < 5; ++i) reads.push_back(viral_ref);
        for (int i = 0; i < 5; ++i) reads.push_back(random_seq);

        auto results = screener.classify_batch(reads);

        if (results.size() != 10) {
            FAIL("test1_batch_10_reads", "classify_batch returned wrong number of results");
            goto test2;
        }

        {
            size_t viral_hits = 0, unknown_hits = 0;
            for (const auto& h : results) {
                if (h.category == NonHostCategory::VIRAL) ++viral_hits;
                else if (h.category == NonHostCategory::UNKNOWN) ++unknown_hits;
            }
            if (viral_hits < 4 || viral_hits > 5) {  // allow ±1 for borderline minimizers
                char buf[80];
                std::snprintf(buf, sizeof(buf), "expected ~5 viral hits, got %zu", viral_hits);
                FAIL("test1_batch_10_reads", buf);
            } else if (unknown_hits < 4) {
                char buf[80];
                std::snprintf(buf, sizeof(buf), "expected ~5 unknown, got %zu", unknown_hits);
                FAIL("test1_batch_10_reads", buf);
            } else {
                PASS("test1_batch_10_reads");
            }
        }
    }

    test2:
    // ── Test 2: FASTQ read/write + NonHostScreener + write nonhost_summary.json ──
    // End-to-end simulation of what singlify does after STAR completes.
    {
        const char* fastq_path = "/tmp/test_nonhost_unmapped.fq";
        const char* json_path  = "/tmp/test_nonhost_summary.json";

        MinSketchIndex viral_idx = MinSketchIndex::build_from_sequences(
            {{viral_ref, 99u}}, 21, 11, 50000u);
        MinSketchIndex micro_idx = MinSketchIndex::build_from_sequences({}, 21, 11);
        NonHostScreener screener(viral_idx, micro_idx, 0.01f);

        // Write fake FASTQ with 6 viral + 4 random reads
        std::vector<std::string> fake_reads;
        for (int i = 0; i < 6; ++i) fake_reads.push_back(viral_ref);
        for (int i = 0; i < 4; ++i) fake_reads.push_back(random_seq);

        try {
            write_fastq(fastq_path, fake_reads);
        } catch (const std::exception& e) {
            FAIL("test2_fastq_pipeline", e.what());
            goto test3;
        }

        {
            // Read the FASTQ back (simulates singlify reading Unmapped.out.mate1)
            auto read_seqs = read_fastq(fastq_path);
            if (read_seqs.size() != 10) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "FASTQ round-trip: expected 10 reads, got %zu", read_seqs.size());
                FAIL("test2_fastq_pipeline", buf);
                goto cleanup_t2;
            }

            // Classify
            auto hits = screener.classify_batch(read_seqs);

            size_t viral_hits = 0, bacterial_hits = 0, fungal_hits = 0, unknown_hits = 0;
            for (const auto& h : hits) {
                switch (h.category) {
                    case NonHostCategory::VIRAL:      ++viral_hits; break;
                    case NonHostCategory::BACTERIAL:  ++bacterial_hits; break;
                    case NonHostCategory::FUNGAL:     ++fungal_hits; break;
                    default:                           ++unknown_hits; break;
                }
            }

            // Write nonhost_summary.json
            write_nonhost_summary_json(json_path, read_seqs.size(), read_seqs.size(),
                                       viral_hits, bacterial_hits, fungal_hits, unknown_hits);

            // Verify JSON was written and has expected fragment
            std::ifstream jsf(json_path);
            if (!jsf) {
                FAIL("test2_fastq_pipeline", "nonhost_summary.json not written");
                goto cleanup_t2;
            }
            std::string json_content((std::istreambuf_iterator<char>(jsf)),
                                      std::istreambuf_iterator<char>());
            if (json_content.find("\"viral_hits\"") == std::string::npos) {
                FAIL("test2_fastq_pipeline", "json missing viral_hits key");
                goto cleanup_t2;
            }
            if (json_content.find("\"viral_fraction\"") == std::string::npos) {
                FAIL("test2_fastq_pipeline", "json missing viral_fraction key");
                goto cleanup_t2;
            }
            if (json_content.find("\"total_unmapped_reads\": 10") == std::string::npos) {
                FAIL("test2_fastq_pipeline", "json has wrong total_unmapped_reads");
                goto cleanup_t2;
            }

            // Expect at least 4 of the 6 viral reads to be detected
            if (viral_hits < 4) {
                char buf[80];
                std::snprintf(buf, sizeof(buf),
                              "expected >= 4 viral hits, got %zu (fraction=%.3f)",
                              viral_hits, static_cast<double>(viral_hits) / 10.0);
                FAIL("test2_fastq_pipeline", buf);
            } else {
                PASS("test2_fastq_pipeline");
            }
        }

        cleanup_t2:
        std::remove(fastq_path);
        std::remove(json_path);
    }

    test3:
    // ── Test 3: Reads shorter than k+w-1 produce UNKNOWN, not a crash ────────
    {
        MinSketchIndex viral_idx = MinSketchIndex::build_from_sequences(
            {{viral_ref, 1u}}, 21, 11, 50000u);
        MinSketchIndex micro_idx = MinSketchIndex::build_from_sequences({}, 21, 11);
        NonHostScreener screener(viral_idx, micro_idx, 0.01f);

        // 10bp read — too short for k=21: must return UNKNOWN without crashing
        std::string short_read = "ACGTACGTAC";
        auto hit = screener.classify(short_read);
        if (hit.category != NonHostCategory::UNKNOWN) {
            char buf[80];
            std::snprintf(buf, sizeof(buf),
                          "short read should be UNKNOWN, got category=%d",
                          static_cast<int>(hit.category));
            FAIL("test3_short_reads", buf);
        } else {
            PASS("test3_short_reads");
        }
    }

    // ── Test 4: Large batch (100 reads) — parallel vs sequential agreement ───
    {
        MinSketchIndex viral_idx = MinSketchIndex::build_from_sequences(
            {{viral_ref, 7u}}, 21, 11, 50000u);
        MinSketchIndex micro_idx = MinSketchIndex::build_from_sequences({}, 21, 11);
        NonHostScreener screener(viral_idx, micro_idx, 0.01f);

        std::vector<std::string> big_batch;
        big_batch.reserve(100);
        for (int i = 0; i < 50; ++i) big_batch.push_back(viral_ref);
        for (int i = 0; i < 50; ++i) big_batch.push_back(random_seq);

        // Sequential
        std::vector<NonHostHit> seq_results;
        seq_results.reserve(100);
        for (const auto& r : big_batch)
            seq_results.push_back(screener.classify(r));

        // Batch (may use OpenMP internally)
        auto batch_results = screener.classify_batch(big_batch);

        bool all_match = true;
        for (size_t i = 0; i < 100; ++i) {
            if (seq_results[i].category != batch_results[i].category ||
                seq_results[i].species_id != batch_results[i].species_id) {
                all_match = false;
                break;
            }
        }

        if (!all_match)
            FAIL("test4_batch_vs_sequential", "batch and sequential results differ");
        else
            PASS("test4_batch_vs_sequential");
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    std::cerr << "\n" << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
