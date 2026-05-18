// SPDX-License-Identifier: MIT
// test_nonhost_db_build.cpp — Tests for NONHOST-VIRALDB/MICROBIALDB task
//
// Tests the build-nonhost-db workflow: building a MinSketchIndex from a
// synthetic FASTA file, saving it as .snhskidx, loading it, and querying
// an input sequence to verify a hit comes back.
//
// Covers Tasks 1 (NONHOST-VIRALDB) and 2 (NONHOST-MICROBIALDB):
//   - VIRAL    kingdom: build from viral-like sequences
//   - BACTERIAL kingdom: build from bacterial-like sequences
//   - FUNGAL    kingdom: build from fungal-like sequences
//
// Run via ctest:
//   ctest -R nonhost_db_build --output-on-failure

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

static std::string make_seq(const char* pattern, int length = 300) {
    std::string s;
    s.reserve(static_cast<size_t>(length));
    const int plen = static_cast<int>(std::strlen(pattern));
    for (int i = 0; i < length; ++i) s += pattern[i % plen];
    return s;
}

// Write a tiny 3-sequence FASTA to a temp file.
// Returns the path of the created file.
static std::string write_fasta_file(const char* path,
                                    const std::vector<std::pair<std::string, std::string>>& seqs) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error(std::string("cannot write FASTA: ") + path);
    for (const auto& [name, seq] : seqs) {
        f << ">" << name << "\n";
        // Write in 60-char lines
        for (size_t i = 0; i < seq.size(); i += 60)
            f << seq.substr(i, 60) << "\n";
    }
    return path;
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

    // ── Test 1: Build index from FASTA file, save, load, query ──────────────
    // Simulates NONHOST-VIRALDB workflow: build from viral FASTA, save to
    // .snhskidx, load, query the same input sequence to verify a hit.
    {
        const char* fasta_path = "/tmp/test_nonhost_viral.fna";
        const char* idx_path   = "/tmp/test_nonhost_viral.snhskidx";

        // Three synthetic viral-like sequences
        std::string seq1 = make_seq("ACGTACGTGCTAGCTAG", 500);
        std::string seq2 = make_seq("TAGCTAGCGATCGATCG", 500);
        std::string seq3 = make_seq("GCATGCATCGATCGATG", 500);

        try {
            write_fasta_file(fasta_path, {
                {"viral_seq_1 species_1", seq1},
                {"viral_seq_2 species_1", seq2},
                {"viral_seq_3 species_2", seq3},
            });
        } catch (const std::exception& e) {
            FAIL("test1_viral_build", e.what());
            goto test2;
        }

        try {
            // Step 1: Build from FASTA using the streaming build path
            MinSketchIndex idx(21, 11);
            std::ifstream fasta(fasta_path);
            std::string line, seq;
            uint32_t species_id = 0;
            uint32_t seq_count = 0;
            while (std::getline(fasta, line)) {
                if (line.empty()) continue;
                if (line[0] == '>') {
                    if (!seq.empty()) {
                        idx.add_sequence(seq, species_id, 50000u);
                        seq.clear();
                        ++seq_count;
                    }
                    ++species_id;
                } else {
                    seq += line;
                }
            }
            if (!seq.empty()) { idx.add_sequence(seq, species_id, 50000u); ++seq_count; }
            idx.finalize();

            if (idx.size() == 0) {
                FAIL("test1_viral_build", "index empty after build from FASTA file");
                goto cleanup_test1;
            }
            if (seq_count != 3) {
                char buf[64]; std::snprintf(buf, sizeof(buf), "expected 3 seqs, got %u", seq_count);
                FAIL("test1_viral_build", buf);
                goto cleanup_test1;
            }

            // Step 2: Save
            idx.save(idx_path);

            // Step 3: Load
            MinSketchIndex loaded = MinSketchIndex::load(idx_path);
            if (loaded.size() != idx.size()) {
                char buf[80];
                std::snprintf(buf, sizeof(buf), "load: size mismatch %zu vs %zu",
                              loaded.size(), idx.size());
                FAIL("test1_viral_build", buf);
                goto cleanup_test1;
            }

            // Step 4: Query seq1 (species_id=1) — must get a hit
            auto hits = loaded.query(seq1);
            if (hits.empty() || hits.find(1u) == hits.end() || hits.at(1u) == 0) {
                FAIL("test1_viral_build", "no hit for viral seq1 after load");
                goto cleanup_test1;
            }

            // Step 5: Query seq3 (species_id=3) — must get a hit
            hits = loaded.query(seq3);
            if (hits.empty() || hits.find(3u) == hits.end() || hits.at(3u) == 0) {
                FAIL("test1_viral_build", "no hit for viral seq3 after load");
                goto cleanup_test1;
            }

            PASS("test1_viral_build");

        } catch (const std::exception& e) {
            FAIL("test1_viral_build", e.what());
        }

        cleanup_test1:
        std::remove(fasta_path);
        std::remove(idx_path);
    }

    test2:
    // ── Test 2: BACTERIAL kingdom — same pattern ─────────────────────────────
    // Confirms BACTERIAL pathway builds a valid index and supports classification.
    {
        const char* idx_path = "/tmp/test_nonhost_bacterial.snhskidx";

        std::string bact_seq = make_seq("GCGCGCATGCTAGCTA", 500);

        try {
            MinSketchIndex bact_idx = MinSketchIndex::build_from_sequences(
                {{bact_seq, 100u}}, 21, 11, 50000u);
            bact_idx.save(idx_path);

            MinSketchIndex loaded_bact = MinSketchIndex::load(idx_path);

            // Build an empty viral index to pair with the bacterial one
            MinSketchIndex empty_viral = MinSketchIndex::build_from_sequences({}, 21, 11);
            NonHostScreener screener(empty_viral, loaded_bact, 0.01f);

            auto hit = screener.classify(bact_seq);
            if (hit.category != NonHostCategory::BACTERIAL) {
                char buf[80];
                std::snprintf(buf, sizeof(buf), "expected BACTERIAL, got category=%d",
                              static_cast<int>(hit.category));
                FAIL("test2_bacterial_build", buf);
            } else if (hit.hit_rate <= 0.0f) {
                FAIL("test2_bacterial_build", "hit_rate must be > 0 for matched bacterial seq");
            } else if (hit.species_id != 100u) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "expected species_id=100, got %u", hit.species_id);
                FAIL("test2_bacterial_build", buf);
            } else {
                PASS("test2_bacterial_build");
            }
        } catch (const std::exception& e) {
            FAIL("test2_bacterial_build", e.what());
        }
        std::remove(idx_path);
    }

    // ── Test 3: FUNGAL kingdom — same pattern ────────────────────────────────
    // Fungal entries use the microbial index (same as bacterial in NonHostScreener).
    // classification returns BACTERIAL (current limitation — FUNGAL metadata
    // requires per-species lookup not yet implemented, per nonhost_screener.h comment).
    {
        std::string fungal_seq = make_seq("TGCATGCAGCGCGCAT", 500);

        try {
            MinSketchIndex fungal_idx = MinSketchIndex::build_from_sequences(
                {{fungal_seq, 200u}}, 21, 11, 50000u);

            // When loaded as microbial index, NonHostScreener returns BACTERIAL
            // (FUNGAL/BACTERIAL distinction requires species metadata — see nonhost_screener.h)
            MinSketchIndex empty_viral = MinSketchIndex::build_from_sequences({}, 21, 11);
            NonHostScreener screener(empty_viral, fungal_idx, 0.01f);

            auto hit = screener.classify(fungal_seq);
            // BACTERIAL is the expected result when no species metadata distinguishes
            if (hit.category != NonHostCategory::BACTERIAL &&
                hit.category != NonHostCategory::FUNGAL) {
                char buf[80];
                std::snprintf(buf, sizeof(buf),
                              "expected BACTERIAL or FUNGAL, got category=%d",
                              static_cast<int>(hit.category));
                FAIL("test3_fungal_build", buf);
            } else if (hit.hit_rate <= 0.0f) {
                FAIL("test3_fungal_build", "hit_rate must be > 0 for matched fungal seq");
            } else {
                PASS("test3_fungal_build");
            }
        } catch (const std::exception& e) {
            FAIL("test3_fungal_build", e.what());
        }
    }

    // ── Test 4: Merge two per-kingdom indexes ────────────────────────────────
    // Confirm MinSketchIndex::merge works correctly across kingdoms.
    {
        std::string vs = make_seq("ACGTACGTCGATCGAT", 300);
        std::string bs = make_seq("TTTAAACCCGGGTTTT", 300);

        try {
            auto viral_idx = MinSketchIndex::build_from_sequences({{vs, 1u}}, 21, 11);
            auto bact_idx  = MinSketchIndex::build_from_sequences({{bs, 2u}}, 21, 11);
            auto merged    = MinSketchIndex::merge(viral_idx, bact_idx);

            if (merged.size() != viral_idx.size() + bact_idx.size()) {
                char buf[80];
                std::snprintf(buf, sizeof(buf),
                              "merge size %zu != %zu + %zu",
                              merged.size(), viral_idx.size(), bact_idx.size());
                FAIL("test4_merge", buf);
            } else {
                // Both species must be queryable from merged index
                auto hv = merged.query(vs);
                auto hb = merged.query(bs);
                if (hv.count(1u) == 0 || hb.count(2u) == 0)
                    FAIL("test4_merge", "merged index missing species hits");
                else
                    PASS("test4_merge");
            }
        } catch (const std::exception& e) {
            FAIL("test4_merge", e.what());
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    std::cerr << "\n" << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
