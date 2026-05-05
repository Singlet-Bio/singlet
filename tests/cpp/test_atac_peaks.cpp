#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "singlet/pileup/atac_peaks.h"

using namespace singlet;

static void test_poisson_pvalue() {
    // P(X >= 100 | lambda=10) must be extremely small
    float pv = poisson_pvalue(100, 10.0f);
    (void)pv;
    assert(pv < 1e-30f && "poisson_pvalue(100,10) should be < 1e-30");

    // P(X >= 0 | lambda=5) == 1
    float pv0 = poisson_pvalue(0, 5.0f);
    (void)pv0;
    assert(pv0 == 1.0f && "poisson_pvalue(0,5) should be 1");

    // P(X >= 1 | lambda=0) == 0
    float pv1 = poisson_pvalue(1, 0.0f);
    (void)pv1;
    assert(pv1 == 0.0f && "poisson_pvalue(1,0) should be 0");

    // P(X >= 5 | lambda=5) ~= 0.56 (reasonable range)
    float pv5 = poisson_pvalue(5, 5.0f);
    (void)pv5;
    assert(pv5 > 0.3f && pv5 < 0.8f && "poisson_pvalue(5,5) out of expected range");

    printf("PASS: poisson_pvalue\n");
}

static void test_peak_calling() {
    // 20 bins on chr1 at 500bp resolution
    // Bins 8,9,10 have high counts (100,150,120), rest have low counts (5-10)
    const int NBINS = 20;
    std::vector<uint32_t> counts(NBINS);
    std::vector<std::string> chroms(NBINS, "chr1");
    std::vector<uint32_t> starts(NBINS);

    for (int i = 0; i < NBINS; ++i) {
        starts[i] = (uint32_t)(i * 500);
        counts[i] = (i % 3 == 0) ? 7 : 5;  // background 5-7
    }
    counts[8] = 100;
    counts[9] = 150;
    counts[10] = 120;

    PeakCallParams params;
    params.bin_size = 500;
    params.background_window = 5000;  // 10 bins each side
    params.pvalue_threshold = 1e-3f;
    params.min_peak_width = 200;
    params.merge_distance = 500;

    auto peaks = call_peaks(counts, chroms, starts, params);

    printf("Peaks called: %zu\n", peaks.size());
    for (auto& pk : peaks) {
        printf("  %s:%u-%u score=%.2f fe=%.2f signal=%u\n",
               pk.chrom.c_str(), pk.start, pk.end,
               pk.score, pk.fold_enrichment, pk.signal);
    }

    assert(!peaks.empty() && "Expected at least one peak");

    // There should be exactly 1 merged peak covering the 3 high-count bins
    assert(peaks.size() == 1 && "Expected exactly 1 merged peak");

    const auto& pk = peaks[0];
    assert(pk.chrom == "chr1" && "Peak chrom should be chr1");
    // Should contain bin index 8 (start=4000) through bin 10 (start=5000, end=5500)
    assert(pk.start <= 4000 && "Peak start should be <= 4000");
    assert(pk.end >= 5500 && "Peak end should be >= 5500");
    assert(pk.score > 0.0f && "Peak score should be > 0");
    assert(pk.fold_enrichment > 3.0f && "fold_enrichment should be > 3");

    printf("PASS: peak_calling (1 peak, score=%.2f, fe=%.2f)\n",
           pk.score, pk.fold_enrichment);
}

static void test_bed_output() {
    ATACPeak pk;
    pk.chrom = "chr1";
    pk.start = 4000;
    pk.end = 5500;
    pk.score = 15.3f;
    pk.signal = 370;
    pk.fold_enrichment = 18.5f;
    std::vector<ATACPeak> peaks = {pk};

    std::string path = "/tmp/test_atac_peaks_output.bed";
    bool ok = write_peaks_bed(path, peaks);
    (void)ok;
    assert(ok && "write_peaks_bed should succeed");

    FILE* f = fopen(path.c_str(), "r");
    assert(f && "BED file should exist");
    char chrom[64];
    uint32_t s, e;
    char name[64];
    float score;
    char strand[4];
    int n = fscanf(f, "%63s %u %u %63s %f %3s", chrom, &s, &e, name, &score, strand);
    (void)n;
    fclose(f);

    assert(n == 6 && "BED6: should parse 6 fields");
    assert(std::string(chrom) == "chr1");
    assert(s == 4000);
    assert(e == 5500);
    assert(std::string(strand) == ".");
    assert(score > 15.0f);

    std::remove(path.c_str());
    printf("PASS: bed_output (BED6 format verified)\n");
}

static void test_multi_chrom() {
    // Peaks on two chromosomes should stay separate
    std::vector<uint32_t> counts = {5, 5, 100, 150, 5, 5, 5, 200, 5, 5};
    std::vector<std::string> chroms = {"chr1", "chr1", "chr1", "chr1", "chr1",
                                       "chr2", "chr2", "chr2", "chr2", "chr2"};
    std::vector<uint32_t> starts;
    for (int i = 0; i < 5; ++i) starts.push_back((uint32_t)(i * 500));
    for (int i = 0; i < 5; ++i) starts.push_back((uint32_t)(i * 500));

    PeakCallParams params;
    params.pvalue_threshold = 1e-2f;
    params.min_peak_width = 200;
    params.merge_distance = 500;

    auto peaks = call_peaks(counts, chroms, starts, params);
    printf("Multi-chrom peaks: %zu\n", peaks.size());
    // Expect at least one peak per chromosome
    bool has_chr1 = false, has_chr2 = false;
    for (auto& pk : peaks) {
        if (pk.chrom == "chr1") has_chr1 = true;
        (void)has_chr1;
        if (pk.chrom == "chr2") has_chr2 = true;
        (void)has_chr2;
    }
    assert(has_chr1 && "Expected peak on chr1");
    assert(has_chr2 && "Expected peak on chr2");
    printf("PASS: multi_chrom\n");
}

int main() {
    test_poisson_pvalue();
    test_peak_calling();
    test_bed_output();
    test_multi_chrom();
    printf("All atac_peaks tests passed.\n");
    return 0;
}
