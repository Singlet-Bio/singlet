// SPDX-License-Identifier: MIT
// test/test_tpm_fpkm.cpp
// Unit tests for tpm_fpkm.h (G-TPM feature).
// Tests effective-length computation, TPM sum invariant, FPKM formula,
// edge cases (zero counts, single gene, all-zero), and TSV output format.
// Does NOT require real GTF parsing — uses synthetic GeneModel via temp GTF.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/gene_model.h"
#include "singlet/pileup/tpm_fpkm.h"

// ── Helpers ──────────────────────────────────────────────────────────────

[[maybe_unused]] static bool approx_eq(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) <= tol;
}

// Write a minimal GTF with genes + exons, return path.
// Each entry: gene_id, gene_name, exon_start (1-based), exon_end (1-based inclusive)
struct ExonEntry {
    std::string gid, gname;
    int start, end;
};

static std::string write_gtf(const std::string& path,
                             const std::vector<ExonEntry>& exons) {
    std::ofstream f(path);
    // All genes on chr1, strand +
    // First pass: emit gene records (union of exon extents)
    std::unordered_map<std::string, std::pair<int, int>> gene_bounds;
    for (auto& e : exons) {
        auto it = gene_bounds.find(e.gid);
        if (it == gene_bounds.end())
            gene_bounds[e.gid] = {e.start, e.end};
        else {
            it->second.first = std::min(it->second.first, e.start);
            it->second.second = std::max(it->second.second, e.end);
        }
    }
    // Emit gene records (need to know gene_name per gid)
    std::unordered_map<std::string, std::string> gid_to_name;
    for (auto& e : exons) gid_to_name[e.gid] = e.gname;

    for (auto& [gid, bounds] : gene_bounds) {
        f << "chr1\ttest\tgene\t" << bounds.first << "\t" << bounds.second
          << "\t.\t+\t.\tgene_id \"" << gid << "\"; gene_name \""
          << gid_to_name[gid] << "\";\n";
    }
    for (auto& e : exons) {
        f << "chr1\ttest\texon\t" << e.start << "\t" << e.end
          << "\t.\t+\t.\tgene_id \"" << e.gid << "\"; gene_name \""
          << e.gname << "\";\n";
    }
    return path;
}

// ── Tests ─────────────────────────────────────────────────────────────────

// Test 1: effective gene length from GTF exon union
static void test_effective_lengths() {
    std::string gtf = "/tmp/test_tpm_gtf1.gtf";
    // Gene A: one exon 1..1000 → length 1000 bp (GTF half-open after parse: 0..1000)
    // Gene B: two exons 1..500 and 601..1000 → 500 + 400 = 900 bp
    write_gtf(gtf, {
                       {"ENSG001", "GeneA", 1, 1000},
                       {"ENSG002", "GeneB", 1, 500},
                       {"ENSG002", "GeneB", 601, 1000},
                   });

    singlet::GeneModel gm;
    bool ok = gm.load_gtf(gtf);
    (void)ok;
    assert(ok && "GTF load failed");
    assert(gm.n_genes() == 2 && "Expected 2 genes");

    auto lengths = singlet::compute_gene_lengths(gm);
    assert(lengths.size() == 2);

    // Find GeneA and GeneB
    uint32_t la = 0, lb = 0;
    for (auto& l : lengths) {
        if (l.gene_id == "ENSG001") la = l.effective_length;
        if (l.gene_id == "ENSG002") lb = l.effective_length;
    }
    assert(la == 1000 && "GeneA effective length should be 1000");
    assert(lb == 900 && "GeneB effective length should be 900 (two exons)");
    std::cerr << "  test_effective_lengths: PASS (A=" << la << " B=" << lb << ")\n";
    std::remove(gtf.c_str());
}

// Test 2: TPM sums to 1e6 (within floating-point tolerance)
static void test_tpm_sum_invariant() {
    // Use raw vectors directly — no GeneModel needed
    std::vector<singlet::GeneLength> lengths = {
        {"G1", "Gene1", 1000},
        {"G2", "Gene2", 500},
        {"G3", "Gene3", 2000},
        {"G4", "Gene4", 750},
        {"G5", "Gene5", 300},
    };
    std::vector<int32_t> counts = {100, 50, 300, 20, 80};

    auto tpm = singlet::compute_tpm(counts, lengths);
    assert(tpm.size() == 5);

    double sum = 0;
    for (double v : tpm) {
        assert(v >= 0.0 && "TPM must be non-negative");
        sum += v;
    }
    assert(approx_eq(sum, 1.0e6, 1e-3) && "TPM must sum to 1e6");
    std::cerr << "  test_tpm_sum_invariant: PASS (sum=" << sum << ")\n";
}

// Test 3: FPKM formula vs manual calculation
static void test_fpkm_formula() {
    // Single gene: count=100, length=1000, total_mapped=1e6
    // FPKM = (100 × 1e9) / (1000 × 1e6) = 100
    std::vector<singlet::GeneLength> lengths = {{"G1", "Gene1", 1000}};
    std::vector<int32_t> counts = {100};
    uint64_t total_mapped = 1000000ULL;

    auto fpkm = singlet::compute_fpkm(counts, lengths, total_mapped);
    assert(fpkm.size() == 1);
    assert(approx_eq(fpkm[0], 100.0, 1e-9) && "FPKM formula mismatch");
    std::cerr << "  test_fpkm_formula: PASS (FPKM=" << fpkm[0] << ")\n";
}

// Test 4: zero counts → TPM and FPKM are zero
static void test_zero_counts() {
    std::vector<singlet::GeneLength> lengths = {
        {"G1", "Gene1", 500},
        {"G2", "Gene2", 1000},
    };
    std::vector<int32_t> counts = {0, 0};

    auto tpm = singlet::compute_tpm(counts, lengths);
    auto fpkm = singlet::compute_fpkm(counts, lengths, 1000000ULL);

    double tpm_sum = 0;
    for (double v : tpm) {
        assert(v == 0.0);
        tpm_sum += v;
    }
    for (double v : fpkm) {
        (void)v;
        assert(v == 0.0);
    }
    assert(tpm_sum == 0.0 && "All-zero counts → TPM sum should be 0");
    std::cerr << "  test_zero_counts: PASS\n";
}

// Test 5: zero total_mapped → FPKM all zero (no divide-by-zero crash)
static void test_zero_total_mapped() {
    std::vector<singlet::GeneLength> lengths = {{"G1", "Gene1", 1000}};
    std::vector<int32_t> counts = {50};

    auto fpkm = singlet::compute_fpkm(counts, lengths, 0ULL);
    assert(fpkm.size() == 1);
    assert(fpkm[0] == 0.0 && "zero total_mapped → FPKM should be 0");
    std::cerr << "  test_zero_total_mapped: PASS\n";
}

// Test 6: zero-length exon clamped to 1 (no divide-by-zero in TPM)
static void test_zero_length_clamped() {
    // Manually construct GeneLength with effective_length = 0 → should already
    // be clamped to 1 by compute_gene_lengths(). Here test the compute functions
    // directly with length=1 (the clamped value).
    std::vector<singlet::GeneLength> lengths = {{"G1", "Gene1", 1}};
    std::vector<int32_t> counts = {10};

    auto tpm = singlet::compute_tpm(counts, lengths);
    auto fpkm = singlet::compute_fpkm(counts, lengths, 1000ULL);

    assert(std::isfinite(tpm[0]) && "TPM should be finite for length=1");
    assert(std::isfinite(fpkm[0]) && "FPKM should be finite for length=1");
    assert(tpm[0] > 0 && fpkm[0] > 0 && "Should be positive for nonzero count");
    std::cerr << "  test_zero_length_clamped: PASS\n";
}

// Test 7: TSV output format — header present, correct column count
static void test_tsv_output() {
    std::string path = "/tmp/test_tpm_out.tsv";
    std::vector<singlet::GeneLength> lengths = {
        {"ENSG001", "GeneA", 1000},
        {"ENSG002", "GeneB", 500},
    };
    std::vector<int32_t> counts = {200, 100};
    auto tpm = singlet::compute_tpm(counts, lengths);
    auto fpkm = singlet::compute_fpkm(counts, lengths, 500000ULL);

    bool ok = singlet::write_gene_expression_tsv(path, lengths, counts, tpm, fpkm);
    (void)ok;
    assert(ok && "TSV write failed");

    std::ifstream f(path);
    assert(f.is_open() && "TSV file not found");

    // Check header
    std::string header;
    std::getline(f, header);
    assert(header == "gene_id\tgene_name\teffective_length\tcount\tTPM\tFPKM" && "Header mismatch");

    // Check each data row has 6 tab-separated columns
    std::string line;
    int nrows = 0;
    while (std::getline(f, line)) {
        int ntabs = 0;
        for (char c : line)
            if (c == '\t') ntabs++;
        assert(ntabs == 5 && "Each data row must have 6 fields (5 tabs)");
        nrows++;
    }
    assert(nrows == 2 && "Expected 2 data rows");
    std::cerr << "  test_tsv_output: PASS\n";
    std::remove(path.c_str());
}

// Test 8: single gene TPM == 1e6 (all reads on one gene)
static void test_single_gene_tpm() {
    std::vector<singlet::GeneLength> lengths = {{"G1", "Gene1", 2000}};
    std::vector<int32_t> counts = {999};

    auto tpm = singlet::compute_tpm(counts, lengths);
    assert(approx_eq(tpm[0], 1.0e6, 1e-6) && "Single gene TPM should equal 1e6");
    std::cerr << "  test_single_gene_tpm: PASS\n";
}

// ── Main ──────────────────────────────────────────────────────────────────

int main() {
    std::cerr << "[test_tpm_fpkm] Running tests...\n";
    test_effective_lengths();
    test_tpm_sum_invariant();
    test_fpkm_formula();
    test_zero_counts();
    test_zero_total_mapped();
    test_zero_length_clamped();
    test_tsv_output();
    test_single_gene_tpm();
    std::cerr << "[test_tpm_fpkm] All tests PASSED\n";
    return 0;
}
