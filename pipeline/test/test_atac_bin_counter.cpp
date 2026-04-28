// test_atac_bin_counter.cpp — unit tests for atac_bin_counter.h

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

#include "singlet-pileup/atac_bin_counter.h"

// Helper: look up value in CSC matrix at (row, col)
static uint16_t csc_get(const singlet::SparseAccumulator<uint16_t>::CSCMatrix& m,
                        int32_t row, uint32_t col) {
    if (col >= m.ncols) return 0;
    for (int32_t i = m.indptr[col]; i < m.indptr[col + 1]; ++i) {
        if (m.indices[static_cast<size_t>(i)] == row) {
            return m.data[static_cast<size_t>(i)];
        }
    }
    return 0;
}

// ── Test 1: Single fragment maps to correct single bin ──────────────────────
static void test_single_fragment() {
    singlet::ATACBinCounter counter;
    counter.set_bin_width(500);
    counter.set_chromosome_sizes({{"chr1", 5000}});  // 10 bins: 0-499,500-999,...

    singlet::ATACFragment frag{};
    frag.chrom_idx = 0;
    frag.start = 600;  // 600/500 = bin 1 (500-999)
    frag.end = 750;    // (750-1)/500 = bin 1 — single bin
    frag.barcode_idx = 0;

    counter.count_one(frag);
    auto m = counter.to_csc(2);

    // bin 1 (offset 1), cell 0 should have count 1
    assert(csc_get(m, 1, 0) == 1 && "single fragment must land in bin 1");
    // bin 0 must be empty
    assert(csc_get(m, 0, 0) == 0 && "bin 0 must be empty");
    std::puts("PASS test_single_fragment");
}

// ── Test 2: Fragment spanning bin boundary increments both bins ──────────────
static void test_fragment_spanning_boundary() {
    singlet::ATACBinCounter counter;
    counter.set_bin_width(500);
    counter.set_chromosome_sizes({{"chr1", 5000}});

    singlet::ATACFragment frag{};
    frag.chrom_idx = 0;
    frag.start = 450;  // 450/500 = bin 0 (0-499)
    frag.end = 550;    // (550-1)/500 = bin 1 (500-999)
    frag.barcode_idx = 0;

    counter.count_one(frag);
    auto m = counter.to_csc(2);

    assert(csc_get(m, 0, 0) == 1 && "bin 0 must be incremented");
    assert(csc_get(m, 1, 0) == 1 && "bin 1 must be incremented");
    std::puts("PASS test_fragment_spanning_boundary");
}

// ── Test 3: Multiple cells accumulate independently ──────────────────────────
static void test_multiple_cells() {
    singlet::ATACBinCounter counter;
    counter.set_bin_width(500);
    counter.set_chromosome_sizes({{"chr1", 5000}});

    // Cell 0: two fragments in bin 0
    counter.count_one({0, 10, 200, 0});
    counter.count_one({0, 50, 300, 0});
    // Cell 1: one fragment in bin 1
    counter.count_one({0, 600, 700, 1});
    // Cell 2: one fragment in bin 0
    counter.count_one({0, 100, 200, 2});

    auto m = counter.to_csc(3);

    assert(csc_get(m, 0, 0) == 2 && "cell 0 bin 0 should be 2");
    assert(csc_get(m, 1, 0) == 0 && "cell 0 bin 1 should be 0");
    assert(csc_get(m, 0, 1) == 0 && "cell 1 bin 0 should be 0");
    assert(csc_get(m, 1, 1) == 1 && "cell 1 bin 1 should be 1");
    assert(csc_get(m, 0, 2) == 1 && "cell 2 bin 0 should be 1");
    std::puts("PASS test_multiple_cells");
}

// ── Test 4: bin_features() generates correct labels ──────────────────────────
static void test_bin_features_labels() {
    singlet::ATACBinCounter counter;
    counter.set_bin_width(500);
    // chr1 of length 1200 → 3 bins: 0-500, 500-1000, 1000-1200
    counter.set_chromosome_sizes({{"chr1", 1200}, {"chr2", 600}});

    auto feats = counter.bin_features();
    // chr1: bins 0-500, 500-1000, 1000-1200
    // chr2: bins 0-500, 500-600
    assert(feats.size() == 5 && "expected 5 bins");

    assert(feats[0] == "chr1:0-500\tchr1:0-500\tPeaks" && "bin 0 label");
    assert(feats[1] == "chr1:500-1000\tchr1:500-1000\tPeaks" && "bin 1 label");
    assert(feats[2] == "chr1:1000-1200\tchr1:1000-1200\tPeaks" && "bin 2 label (clipped)");
    assert(feats[3] == "chr2:0-500\tchr2:0-500\tPeaks" && "chr2 bin 0 label");
    assert(feats[4] == "chr2:500-600\tchr2:500-600\tPeaks" && "chr2 bin 1 label");
    std::puts("PASS test_bin_features_labels");
}

// ── Test 5: Fragment on second chromosome uses correct global offset ──────────
static void test_second_chromosome_offset() {
    singlet::ATACBinCounter counter;
    counter.set_bin_width(500);
    // chr1: 1000 bp → 2 bins (offsets 0,1)
    // chr2: 1000 bp → 2 bins (offsets 2,3)
    counter.set_chromosome_sizes({{"chr1", 1000}, {"chr2", 1000}});

    // Fragment on chr2 bin 0 (global bin index 2)
    counter.count_one({1, 100, 200, 0});

    auto m = counter.to_csc(1);
    assert(csc_get(m, 2, 0) == 1 && "chr2 bin 0 should map to global bin 2");
    assert(csc_get(m, 0, 0) == 0 && "chr1 bins should be empty");
    std::puts("PASS test_second_chromosome_offset");
}

int main() {
    test_single_fragment();
    test_fragment_spanning_boundary();
    test_multiple_cells();
    test_bin_features_labels();
    test_second_chromosome_offset();
    std::puts("All ATACBinCounter tests passed.");
    return 0;
}
