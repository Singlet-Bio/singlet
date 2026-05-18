// SPDX-License-Identifier: MIT
// G-METRICS + G-PERMITLIST: unit tests for metrics_summary.h and forced/expect cells
// Run via ctest or standalone:
//   g++ -std=c++17 -O2 -I../include -o /tmp/test_ms test/test_metrics_summary.cpp && /tmp/test_ms

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/export.h"
#include "singlet/pileup/metrics_summary.h"
#include "singlet/pileup/pileup_engine.h"
#include "singlet/pileup/sparse_accumulator.h"

using namespace singlet;

// ── Helper: build a CSC matrix from (barcode, gene, count) entries ──────────
static SparseAccumulator<uint16_t>::CSCMatrix make_test_csc(
    uint32_t n_genes, uint32_t n_barcodes,
    const std::vector<std::tuple<uint32_t, uint32_t, uint16_t>>& entries) {
    SparseAccumulator<uint16_t> acc;
    std::vector<std::string> bcs(n_barcodes, "X");
    acc.set_barcodes(bcs);
    acc.set_n_features(n_genes);
    for (auto& [gene, barcode, cnt] : entries)
        acc.increment(gene, barcode, cnt);
    return acc.to_csc();
}

// ── Helper: read CSV into map ────────────────────────────────────────────────
static std::map<std::string, std::string> read_csv(const std::string& path) {
    std::map<std::string, std::string> m;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        auto comma = line.find(',');
        if (comma == std::string::npos) continue;
        m[line.substr(0, comma)] = line.substr(comma + 1);
    }
    return m;
}

int main() {
    int passed = 0, failed = 0;
    auto PASS = [&](const char* name) { ++passed; std::cerr << "PASS: " << name << "\n"; };
    auto FAIL = [&](const char* name, const char* msg) {
        ++failed;
        std::cerr << "FAIL: " << name << " — " << msg << "\n";
    };

    // ── 1. fmt_int: basic formatting ─────────────────────────────────────────
    {
        if (fmt_int(0) == "0")
            PASS("fmt_int(0)");
        else
            FAIL("fmt_int(0)", fmt_int(0).c_str());
    }
    {
        if (fmt_int(1000) == "1,000")
            PASS("fmt_int(1000)");
        else
            FAIL("fmt_int(1000)", fmt_int(1000).c_str());
    }
    {
        if (fmt_int(1234567) == "1,234,567")
            PASS("fmt_int(1234567)");
        else
            FAIL("fmt_int(1234567)", fmt_int(1234567).c_str());
    }

    // ── 2. fmt_pct ────────────────────────────────────────────────────────────
    {
        std::string s = fmt_pct(0.732);
        if (s == "73.2%")
            PASS("fmt_pct(0.732)");
        else
            FAIL("fmt_pct(0.732)", s.c_str());
    }

    // ── 3. median_of: odd size ────────────────────────────────────────────────
    {
        std::vector<uint32_t> v = {5, 3, 1, 4, 2};
        double m = median_of(v);
        if (m == 3.0)
            PASS("median_of odd");
        else
            FAIL("median_of odd", std::to_string(m).c_str());
    }

    // ── 4. median_of: even size ───────────────────────────────────────────────
    {
        std::vector<uint32_t> v = {4, 2, 1, 3};
        double m = median_of(v);
        if (m == 2.5)
            PASS("median_of even");
        else
            FAIL("median_of even", std::to_string(m).c_str());
    }

    // ── 5. median_of: empty ───────────────────────────────────────────────────
    {
        std::vector<uint32_t> v;
        if (median_of(v) == 0.0)
            PASS("median_of empty=0");
        else
            FAIL("median_of empty", "should be 0");
    }

    // ── 6. write_metrics_summary_csv: basic smoke test ───────────────────────
    {
        // 4 barcodes, 3 genes, 2 called cells
        // barcode 0: gene0=5, gene1=3  → 8 UMI, 2 genes
        // barcode 1: gene0=10, gene2=2 → 12 UMI, 2 genes
        // barcode 2: gene1=1            → 1 UMI, 1 gene (ambient)
        // barcode 3: gene2=1            → 1 UMI, 1 gene (ambient)
        auto csc = make_test_csc(3, 4, {{0, 0, 5}, {1, 0, 3}, {0, 1, 10}, {2, 1, 2}, {1, 2, 1}, {2, 3, 1}});

        PileupStats st;
        st.total_reads = 1000;
        st.mapped_reads = 900;
        st.barcoded_reads = 850;
        st.exon_hits = 800;
        st.intron_hits = 50;
        st.umi_unique = 21;    // 8+12+1 = actual UMI counts
        st.umi_duplicate = 7;  // 7 duplicate copies

        std::vector<uint32_t> cell_indices = {0, 1};  // called cells
        std::vector<uint64_t> bc_totals = {8, 12, 1, 1};

        // Minimal gene model with no hierarchy — use "no genemodel" path
        GeneModel gm;

        std::string tmp_csv = "/tmp/test_metrics_summary.csv";
        write_metrics_summary_csv(tmp_csv, st, cell_indices, bc_totals, csc, gm);

        // Verify file was written
        std::ifstream f(tmp_csv);
        bool file_ok = f.good();
        f.close();
        if (file_ok)
            PASS("write_metrics_summary_csv creates file");
        else {
            FAIL("write_metrics_summary_csv creates file", "file not found");
            goto next1;
        }

        {
            auto m = read_csv(tmp_csv);

            // n_cells = 2
            if (m["Estimated Number of Cells"] == "2")
                PASS("n_cells=2");
            else
                FAIL("n_cells=2", m["Estimated Number of Cells"].c_str());

            // total reads = 1,000
            if (m["Number of Reads"] == "1,000")
                PASS("total_reads formatted");
            else
                FAIL("total_reads formatted", m["Number of Reads"].c_str());

            // Q30 fields = N/A
            if (m["Q30 Bases in Barcode"] == "N/A")
                PASS("Q30 Barcode=N/A");
            else
                FAIL("Q30 Barcode=N/A", m["Q30 Bases in Barcode"].c_str());

            // Valid barcodes = 85.0% (850/1000)
            if (m["Valid Barcodes"] == "85.0%")
                PASS("valid_barcodes=85.0%");
            else
                FAIL("valid_barcodes=85.0%", m["Valid Barcodes"].c_str());

            // Sequencing Saturation = 7/(21+7) = 25.0%
            if (m["Sequencing Saturation"] == "25.0%")
                PASS("saturation=25%");
            else
                FAIL("saturation=25%", m["Sequencing Saturation"].c_str());

            // Reads Mapped to Genome = 90.0%
            if (m["Reads Mapped to Genome"] == "90.0%")
                PASS("mapped=90%");
            else
                FAIL("mapped=90%", m["Reads Mapped to Genome"].c_str());

            // Fraction Reads in Cells = (8+12)/1000 = 2.0%
            if (m["Fraction Reads in Cells"] == "2.0%")
                PASS("frac_in_cells=2%");
            else
                FAIL("frac_in_cells=2%", m["Fraction Reads in Cells"].c_str());

            // Median UMI = median of {8, 12} = 10
            if (m["Median UMI Counts per Cell"] == "10")
                PASS("median_umi=10");
            else
                FAIL("median_umi=10", m["Median UMI Counts per Cell"].c_str());
        }
    next1:;
        std::remove(tmp_csv.c_str());
    }

    // ── 7. write_metrics_summary_csv: zero cells ────────────────────────────
    {
        PileupStats st;
        st.total_reads = 500;
        GeneModel gm;
        SparseAccumulator<uint16_t>::CSCMatrix empty_csc;
        empty_csc.ncols = 0;
        empty_csc.nrows = 0;
        std::vector<uint32_t> cell_indices;
        std::vector<uint64_t> bc_totals;

        std::string tmp_csv = "/tmp/test_metrics_summary_empty.csv";
        write_metrics_summary_csv(tmp_csv, st, cell_indices, bc_totals, empty_csc, gm);
        auto m = read_csv(tmp_csv);
        if (m["Estimated Number of Cells"] == "0")
            PASS("zero cells handled");
        else
            FAIL("zero cells handled", m["Estimated Number of Cells"].c_str());
        std::remove(tmp_csv.c_str());
    }

    // ── 8. G-PERMITLIST: ExportConfig forced_cells field exists ──────────────
    {
        ExportConfig cfg;
        cfg.forced_cells = 100;
        cfg.expect_cells = 5000;
        if (cfg.forced_cells == 100 && cfg.expect_cells == 5000)
            PASS("ExportConfig forced_cells/expect_cells fields");
        else
            FAIL("ExportConfig forced_cells/expect_cells fields", "field values wrong");
    }

    // ── 9. G-PERMITLIST: forced_cells top-N logic ────────────────────────────
    {
        // Simulate forced-cells sort: 5 barcodes, want top-3
        std::vector<uint64_t> bc_totals = {5, 100, 3, 50, 20};
        std::vector<uint32_t> order(bc_totals.size());
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
                  [&](uint32_t a, uint32_t b) { return bc_totals[a] > bc_totals[b]; });
        uint32_t n = 3;
        std::vector<uint32_t> top3(order.begin(), order.begin() + n);
        // Expected order: barcode 1 (100) > barcode 3 (50) > barcode 4 (20)
        if (top3[0] == 1 && top3[1] == 3 && top3[2] == 4)
            PASS("forced_cells top-3 ordering");
        else
            FAIL("forced_cells top-3 ordering",
                 ("got " + std::to_string(top3[0]) + "," +
                  std::to_string(top3[1]) + "," + std::to_string(top3[2]))
                     .c_str());
    }

    // ── 10. G-PERMITLIST: forced_cells clamp to available barcodes ───────────
    {
        std::vector<uint64_t> bc_totals = {10, 5, 3};
        std::vector<uint32_t> order(bc_totals.size());
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
                  [&](uint32_t a, uint32_t b) { return bc_totals[a] > bc_totals[b]; });
        uint32_t requested = 1000;
        uint32_t n = std::min(requested, static_cast<uint32_t>(order.size()));
        if (n == 3)
            PASS("forced_cells clamp to available");
        else
            FAIL("forced_cells clamp", std::to_string(n).c_str());
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    std::cerr << "\n"
              << passed << "/" << (passed + failed) << " tests passed\n";
    return failed > 0 ? 1 : 0;
}
