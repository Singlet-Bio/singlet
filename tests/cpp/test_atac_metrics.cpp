// SPDX-License-Identifier: MIT
// test_atac_metrics.cpp — unit tests for atac_metrics.h and call_cells_mixture (atac_cell_caller.h).
// Hand-built annotation and fragments; every expected value is computed by hand in the comments.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "singlet/pileup/atac_cell_caller.h"
#include "singlet/pileup/atac_metrics.h"

namespace {

using singlet::ATACFragment;

bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

void test_metrics() {
    // chrom 0: TSS at 10000, gene 0 = [9000, 12000); chrom 1 = chrM.
    singlet::AtacAnnotation ann;
    ann.tss = {{10000}, {}};
    ann.genes = {{0, 9000, 12000, 0}};
    ann.gene_names = {"G0_A"};
    singlet::AtacPeak pk;
    pk.chrom_idx = 0;
    pk.start = 9950;
    pk.end = 10050;
    const std::vector<singlet::AtacPeak> peaks = {pk};
    const std::vector<bool> is_mito = {false, true};

    // Sorted by (chrom, start). Barcode 0 unless noted.
    //  F [3000,3200)   len 200 mono; far from TSS and gene
    //  A [8050,8120)   len 70 NFR; cut 8050 is 1950 from TSS (flank), cut 8119 is 1881 (neither); outside gene
    //  G [9960,10040)  len 80 NFR; TSS window, both cuts (d=40,39) in TSS +-50 and in the peak; in gene
    //  B [9990,10100)  len 110 NFR, 3 read pairs; TSS window; cut 9990 (d=10) TSS and in peak, cut 10099 (d=99) neither
    //  C [11950,12250) len 300 neither NFR nor mono; cut 11950 is 1950 from TSS (flank); overlaps gene (11950 < 12000)
    //  E barcode 1 [500000,500100) len 100 NFR, far away
    //  D chrM [100,200) 2 read pairs, barcode 0
    std::vector<ATACFragment> fr = {
        {0, 3000, 3200, 0, 1},   {0, 8050, 8120, 0, 1},    {0, 9960, 10040, 0, 1}, {0, 9990, 10100, 0, 3},
        {0, 11950, 12250, 0, 1}, {0, 500000, 500100, 1, 1}, {1, 100, 200, 0, 2},
    };
    std::vector<int32_t> col_of = {0, -1};
    singlet::AtacTriplets peak_mat, gene_mat;
    auto m = singlet::compute_atac_metrics(fr, 2, is_mito, ann, peaks, &col_of, &peak_mat, &gene_mat);
    assert(m.size() == 2);

    const auto& a = m[0];
    assert(a.read_pairs == 1 + 1 + 1 + 3 + 1 + 2);  // 9
    assert(a.fragments == 5);
    assert(a.mito_fragments == 1);
    assert(a.tss_fragments == 2);                     // G, B
    assert(a.tss_insertions == 3);                    // G both cuts, B 9990
    assert(a.flank_insertions == 2);                  // A 8050, C 11950
    assert(a.nfr_fragments == 3);                     // A, G, B
    assert(a.mono_fragments == 1);                    // F
    assert(a.peak_region_fragments == 2);             // G, B
    assert(a.peak_region_cutsites == 3);              // G 2, B 1
    assert(a.gene_activity_fragments == 3);           // G, B, C
    assert(near(a.tss_enrichment(), (3.0 / 101.0) / (2.0 / 200.0)));  // 2.970297
    assert(near(a.frip(), 2.0 / 5.0));
    assert(near(a.nucleosome_signal(), 1.0 / 3.0));

    const auto& b = m[1];
    assert(b.read_pairs == 1 && b.fragments == 1 && b.mito_fragments == 0);
    assert(b.tss_fragments == 0 && b.tss_insertions == 0 && b.flank_insertions == 0);
    assert(b.nfr_fragments == 1 && b.mono_fragments == 0);
    assert(b.peak_region_fragments == 0 && b.peak_region_cutsites == 0 && b.gene_activity_fragments == 0);
    assert(near(b.tss_enrichment(), 0.0));  // 0 insertions over the 0.2 flank floor
    assert(near(b.frip(), 0.0));
    assert(near(b.nucleosome_signal(), 0.0));  // no mono fragments

    // Sparse matrices: only barcode 0 has a column. Peak 0: 3 cut sites; gene 0: 3 fragments.
    std::vector<int32_t> indptr, indices;
    std::vector<uint32_t> data;
    peak_mat.to_csc(1, indptr, indices, data);
    assert((indptr == std::vector<int32_t>{0, 1}));
    assert((indices == std::vector<int32_t>{0}));
    assert((data == std::vector<uint32_t>{3}));
    gene_mat.to_csc(1, indptr, indices, data);
    assert((indptr == std::vector<int32_t>{0, 1}));
    assert((indices == std::vector<int32_t>{0}));
    assert((data == std::vector<uint32_t>{3}));

    // TSV
    const std::string path = "test_atac_metrics_tmp.tsv";
    const std::vector<std::string> barcodes = {"AAAC", "TTTG"};
    assert(singlet::write_atac_metrics_tsv(path, m, barcodes, {1, 0}, {1, 0}));
    {
        std::ifstream in(path);
        assert(in);
        std::string line;
        std::vector<std::string> lines;
        while (std::getline(in, line)) lines.push_back(line);
        assert(lines.size() == 3);  // header + 2 rows
        for (const auto& l : lines) {
            size_t tabs = 0;
            for (char ch : l) tabs += (ch == '\t');
            assert(tabs == 11);  // 12 columns
        }
        assert(lines[0].rfind("barcode\tis_cell\t", 0) == 0);
        assert(lines[1].rfind("TTTG\t0\t1\t1\t0\t0\t0\t", 0) == 0);  // order = {1, 0}; duplicates = 1 - 1 = 0
        // barcode 0: is_cell 1, read_pairs 9, fragments 5, duplicate = 9 - (5 + 1) = 3, mito 1, tss_fragments 2
        assert(lines[2].rfind("AAAC\t1\t9\t5\t3\t1\t2\t", 0) == 0);
    }
    std::remove(path.c_str());
    std::printf("  metrics: PASS\n");
}

// Deterministic normal generator: LCG + Box-Muller.
struct Gauss {
    uint64_t s;
    explicit Gauss(uint64_t seed) : s(seed) {}
    double u() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return ((s >> 11) + 0.5) / 9007199254740992.0;  // (0, 1)
    }
    double z() { return std::sqrt(-2.0 * std::log(u())) * std::cos(6.283185307179586 * u()); }
};

void check_calls(const std::vector<uint32_t>& atac, const std::vector<uint8_t>& is_cell,
                 const singlet::MixtureCellCall& r) {
    assert(is_cell.size() == atac.size());
    uint32_t n = 0;
    for (size_t i = 0; i < atac.size(); ++i) {
        n += is_cell[i];
        if (atac[i] > 1000) assert(is_cell[i] == 1);
        if (atac[i] < 100) assert(is_cell[i] == 0);
    }
    assert(n == r.n_cells);
    assert(n >= 290 && n <= 310);
}

void test_mixture() {
    const size_t n_noise = 2000, n_cell = 300;
    Gauss g(2024);
    std::vector<uint32_t> atac, gex;
    for (size_t i = 0; i < n_noise; ++i) {
        atac.push_back(static_cast<uint32_t>(std::pow(10.0, std::log10(30.0) + 0.25 * g.z())));
        gex.push_back(static_cast<uint32_t>(std::pow(10.0, std::log10(20.0) + 0.25 * g.z())));
    }
    for (size_t i = 0; i < n_cell; ++i) {
        atac.push_back(static_cast<uint32_t>(std::pow(10.0, std::log10(5000.0) + 0.2 * g.z())));
        gex.push_back(static_cast<uint32_t>(std::pow(10.0, std::log10(3000.0) + 0.2 * g.z())));
    }

    std::vector<uint8_t> is_cell;
    auto r1 = singlet::call_cells_mixture({atac}, is_cell);
    assert(r1.dims == 1);
    assert(r1.mean[1][0] > r1.mean[0][0]);
    assert(r1.weight_cell > 0.10 && r1.weight_cell < 0.17);  // 300 / ~2250 candidates
    check_calls(atac, is_cell, r1);
    std::printf("  mixture_1d: PASS (%u cells)\n", r1.n_cells);

    auto r2 = singlet::call_cells_mixture({gex, atac}, is_cell);
    assert(r2.dims == 2);
    assert(r2.mean[1][0] > r2.mean[0][0] && r2.mean[1][1] > r2.mean[0][1]);
    check_calls(atac, is_cell, r2);
    std::printf("  mixture_2d: PASS (%u cells)\n", r2.n_cells);

    // Degenerate input: too few candidates -> no cells, no crash.
    auto r3 = singlet::call_cells_mixture({std::vector<uint32_t>{1, 2, 3}}, is_cell);
    assert(r3.n_cells == 0 && is_cell.size() == 3);
}

}  // namespace

int main() {
    std::printf("test_atac_metrics\n");
    test_metrics();
    test_mixture();
    std::printf("All tests passed.\n");
    return 0;
}
