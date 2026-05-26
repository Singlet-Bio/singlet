// test_nonhost_sylph.cpp — B-G7-2: NonhostSylphScreener unit tests
//
// Tests:
//   1. Empty index → no hits
//   2. Spiked read → hit with containment > 0
//   3. Batch API matches per-read API
//   4. abundance field is 0.0 at classify time
//   5. DB with two species → correct species returned

#include "singlet/pileup/nonhost/nonhost_sylph.h"
#include "singlet/pileup/nonhost/min_sketch.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using namespace singlet::nonhost;

// Build a tiny MinSketchIndex with one known sequence for species 0
static MinSketchIndex make_tiny_index(int k = 21, int w = 11) {
    MinSketchIndex idx(k, w);
    // Known viral-like sequence (60 bp)
    const std::string seq = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG";
    idx.add_sequence(seq, /*species_id=*/0, /*max_size=*/100000);
    idx.set_species_name(0, "mock_virus_A");
    idx.finalize();
    return idx;
}

static MinSketchIndex make_two_species_index(int k = 21, int w = 11) {
    MinSketchIndex idx(k, w);
    const std::string seq_a = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG";
    const std::string seq_b = "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT";
    idx.add_sequence(seq_a, 0, 100000);
    idx.add_sequence(seq_b, 1, 100000);
    idx.set_species_name(0, "mock_virus_A");
    idx.set_species_name(1, "mock_bacteria_B");
    idx.finalize();
    return idx;
}

static int n_pass = 0;
static int n_fail = 0;

#define EXPECT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); ++n_fail; } \
    else         { printf("PASS: %s\n", msg); ++n_pass; } \
} while(0)

int main() {
    // ── Test 1: empty index → no hits ────────────────────────────────────────
    {
        MinSketchIndex empty_viral(21, 11); empty_viral.finalize();
        MinSketchIndex empty_micr(21, 11);  empty_micr.finalize();
        NonhostSketchDB db(empty_viral, empty_micr, 0.05f);
        NonhostSylphScreener s(db);
        auto hits = s.screen("ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG");
        EXPECT(hits.empty(), "T1: empty index → no hits");
    }

    // ── Test 2: spiked read → hit with containment > 0 ───────────────────────
    {
        auto viral = make_tiny_index();
        MinSketchIndex empty_micr(21, 11); empty_micr.finalize();
        NonhostSketchDB db(viral, empty_micr, 0.0f);  // threshold=0 to ensure hit
        NonhostSylphScreener s(db);
        const std::string read = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG";
        auto hits = s.screen(read);
        EXPECT(!hits.empty(), "T2a: spiked read → at least one hit");
        if (!hits.empty()) {
            EXPECT(hits[0].species_taxid == 0, "T2b: correct species_taxid=0");
            EXPECT(hits[0].containment > 0.0, "T2c: containment > 0");
            EXPECT(hits[0].abundance == 0.0,  "T2d: abundance=0.0 at classify time");
        }
    }

    // ── Test 3: batch API matches per-read API ────────────────────────────────
    {
        auto viral = make_tiny_index();
        MinSketchIndex empty_micr(21, 11); empty_micr.finalize();
        NonhostSketchDB db(viral, empty_micr, 0.0f);
        NonhostSylphScreener s(db);

        std::vector<std::string> reads = {
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",  // should hit
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  // no hit
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",  // should hit
        };

        auto batch = s.screen_batch(reads, 1);
        EXPECT(batch.size() == reads.size(), "T3a: batch size matches input");

        for (size_t i = 0; i < reads.size(); ++i) {
            auto single = s.screen(reads[i]);
            EXPECT(batch[i].size() == single.size(),
                   (std::string("T3b: batch[") + std::to_string(i) + "] size matches single").c_str());
        }
    }

    // ── Test 4: two-species index → correct species identified ───────────────
    {
        auto idx = make_two_species_index();
        MinSketchIndex empty_v(21, 11); empty_v.finalize();
        // Use idx as microbial DB
        NonhostSketchDB db(empty_v, idx, 0.0f);
        NonhostSylphScreener s(db);

        const std::string read_a = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG";
        const std::string read_b = "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT";

        auto hits_a = s.screen(read_a);
        auto hits_b = s.screen(read_b);

        // At minimum, the hit with highest containment should be species 0 / 1
        bool found_a = false, found_b = false;
        for (const auto& h : hits_a) if (h.species_taxid == 0 && h.containment > 0.0) found_a = true;
        for (const auto& h : hits_b) if (h.species_taxid == 1 && h.containment > 0.0) found_b = true;

        EXPECT(found_a, "T4a: read_a hits species 0");
        EXPECT(found_b, "T4b: read_b hits species 1");
    }

    printf("\n--- Results: %d PASS, %d FAIL ---\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
