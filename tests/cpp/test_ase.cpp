#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/ase.h"
#include "singlet/pileup/sparse_accumulator.h"

using namespace singlet::ase;
using singlet::SparseAccumulator;

// Alias for the CSCMatrix type used in ASE
using CSCMatrix = SparseAccumulator<uint8_t>::CSCMatrix;

// Helper: Create an empty CSCMatrix
CSCMatrix create_empty_csc() {
    CSCMatrix m;
    m.nrows = 0;
    m.ncols = 0;
    m.indptr.resize(1, 0);
    return m;
}

// Helper: Create a CSCMatrix with data at specific (row, col) positions
// Entries are provided as lists of (row, col, value) triplets
CSCMatrix create_csc_from_entries(uint32_t nrows, uint32_t ncols,
                                   const std::vector<uint32_t>& rows,
                                   const std::vector<uint32_t>& cols,
                                   const std::vector<uint8_t>& vals) {
    assert(rows.size() == cols.size() && cols.size() == vals.size());

    CSCMatrix m;
    m.nrows = nrows;
    m.ncols = ncols;

    // Count entries per column
    std::vector<int32_t> col_count(ncols, 0);
    for (uint32_t c : cols) {
        col_count[c]++;
    }

    // Build indptr (column pointers)
    m.indptr.resize(ncols + 1, 0);
    for (uint32_t c = 0; c < ncols; ++c) {
        m.indptr[c + 1] = m.indptr[c] + col_count[c];
    }

    // Build indices and data by column
    std::vector<int32_t> col_pos(m.indptr.begin(), m.indptr.begin() + ncols);
    m.indices.resize(rows.size());
    m.data.resize(vals.size());

    for (size_t i = 0; i < rows.size(); ++i) {
        uint32_t col = cols[i];
        int32_t slot = col_pos[col]++;
        m.indices[slot] = static_cast<int32_t>(rows[i]);
        m.data[slot] = vals[i];
    }

    return m;
}

// Helper: Create a CSC matrix with a single column (single cell)
CSCMatrix create_csc_single_col(uint32_t nrows,
                                 const std::vector<uint32_t>& row_indices,
                                 const std::vector<uint8_t>& values) {
    assert(row_indices.size() == values.size());

    CSCMatrix m;
    m.nrows = nrows;
    m.ncols = 1;
    m.indptr = {0, static_cast<int32_t>(row_indices.size())};
    m.indices.assign(row_indices.begin(), row_indices.end());
    m.data.assign(values.begin(), values.end());

    return m;
}

// Test 1: compute_ase with empty matrices → empty result
void test_empty_matrices() {
    CSCMatrix empty_ad = create_empty_csc();
    CSCMatrix empty_dp = create_empty_csc();

    std::vector<ASEEntry> entries = compute_ase(empty_ad, empty_dp, 10);

    assert(entries.empty() && "Expected empty result for empty matrices");
    std::cout << "✓ test_empty_matrices: empty result\n";
}

// Test 2: compute_ase with one het site (dp=10, ad=5) → one entry with
// allelic_ratio=0.5
void test_single_het_site() {
    // Single SNP (nrows=1), single cell (ncols=1)
    // SNP 0 in cell 0: DP=10, AD=5
    CSCMatrix ad_csc = create_csc_single_col(1, {0}, {5});
    CSCMatrix dp_csc = create_csc_single_col(1, {0}, {10});

    std::vector<ASEEntry> entries = compute_ase(ad_csc, dp_csc, 10);

    assert(entries.size() == 1 && "Expected 1 ASE entry");
    const ASEEntry& e = entries[0];
    assert(e.cell_idx == 0 && "Expected cell_idx=0");
    assert(e.snp_idx == 0 && "Expected snp_idx=0");
    assert(e.ref_count == 5 && "Expected ref_count=5");
    assert(e.alt_count == 5 && "Expected alt_count=5");
    assert(std::abs(e.allelic_ratio - 0.5f) < 0.001f &&
           "Expected allelic_ratio≈0.5");
    std::cout << "✓ test_single_het_site: het site detected (ratio="
              << e.allelic_ratio << ")\n";
}

// Test 3: compute_ase with homozygous ref site (dp=10, ad=0) → filtered out
void test_homozygous_ref_filtered() {
    // SNP 0: DP=10, AD=0 (homozygous ref)
    CSCMatrix ad_csc = create_csc_single_col(1, {}, {});
    CSCMatrix dp_csc = create_csc_single_col(1, {0}, {10});

    std::vector<ASEEntry> entries = compute_ase(ad_csc, dp_csc, 10);

    assert(entries.empty() && "Expected homozygous ref to be filtered out");
    std::cout << "✓ test_homozygous_ref_filtered: correctly filtered\n";
}

// Test 4: compute_ase with homozygous alt site (dp=10, ad=10) → filtered out
void test_homozygous_alt_filtered() {
    // SNP 0: DP=10, AD=10 (homozygous alt, ref=0)
    CSCMatrix ad_csc = create_csc_single_col(1, {0}, {10});
    CSCMatrix dp_csc = create_csc_single_col(1, {0}, {10});

    std::vector<ASEEntry> entries = compute_ase(ad_csc, dp_csc, 10);

    assert(entries.empty() && "Expected homozygous alt (ref=0) to be filtered out");
    std::cout << "✓ test_homozygous_alt_filtered: correctly filtered\n";
}

// Test 5: compute_ase with low depth (dp=2, ad=1, min_depth=5) → filtered out
void test_low_depth_filtered() {
    // SNP 0: DP=2, AD=1 (would be het but depth too low)
    CSCMatrix ad_csc = create_csc_single_col(1, {0}, {1});
    CSCMatrix dp_csc = create_csc_single_col(1, {0}, {2});

    std::vector<ASEEntry> entries = compute_ase(ad_csc, dp_csc, 5);

    assert(entries.empty() && "Expected low-depth site to be filtered out");
    std::cout << "✓ test_low_depth_filtered: low depth correctly filtered\n";
}

// Test 6: compute_ase with multiple cells and sites
void test_multiple_cells_and_sites() {
    // 3 SNPs, 2 cells
    // Cell 0: SNP0(het:dp=10,ad=3), SNP1(het:dp=20,ad=7), SNP2(skip:dp=2)
    // Cell 1: SNP0(skip:dp doesn't exist), SNP1(het:dp=15,ad=6), SNP2(het:dp=12,ad=4)

    std::vector<uint32_t> rows = {
        0, 1, 2,        // Cell 0: SNP 0, SNP 1, SNP 2
        1, 2            // Cell 1: SNP 1, SNP 2
    };
    std::vector<uint32_t> cols = {
        0, 0, 0,        // Cell 0
        1, 1            // Cell 1
    };
    std::vector<uint8_t> ad_vals = {3, 7, 2, 6, 4};
    std::vector<uint8_t> dp_vals = {10, 20, 2, 15, 12};

    CSCMatrix ad_csc = create_csc_from_entries(3, 2, rows, cols, ad_vals);
    CSCMatrix dp_csc = create_csc_from_entries(3, 2, rows, cols, dp_vals);

    std::vector<ASEEntry> entries = compute_ase(ad_csc, dp_csc, 10);

    // Expected het entries (DP >= 10, AD >= 1, ref >= 1):
    // Cell 0, SNP 0: DP=10, AD=3 (ref=7) ✓
    // Cell 0, SNP 1: DP=20, AD=7 (ref=13) ✓
    // Cell 0, SNP 2: DP=2 < 10 ✗ (filtered out)
    // Cell 1, SNP 1: DP=15, AD=6 (ref=9) ✓
    // Cell 1, SNP 2: DP=12, AD=4 (ref=8) ✓
    // Total: 4 het entries

    assert(entries.size() == 4 && "Expected 4 het entries");

    // Check first entry (cell 0, snp 0)
    assert(entries[0].cell_idx == 0 && entries[0].snp_idx == 0 &&
           entries[0].alt_count == 3 && entries[0].ref_count == 7);

    // Check second entry (cell 0, snp 1)
    assert(entries[1].cell_idx == 0 && entries[1].snp_idx == 1 &&
           entries[1].alt_count == 7 && entries[1].ref_count == 13);

    // Check third entry (cell 1, snp 1)
    assert(entries[2].cell_idx == 1 && entries[2].snp_idx == 1 &&
           entries[2].alt_count == 6 && entries[2].ref_count == 9);

    // Check fourth entry (cell 1, snp 2)
    assert(entries[3].cell_idx == 1 && entries[3].snp_idx == 2 &&
           entries[3].alt_count == 4 && entries[3].ref_count == 8);

    std::cout << "✓ test_multiple_cells_and_sites: " << entries.size()
              << " het entries detected\n";
}

// Test 7: write_ase_tsv creates file with correct header
void test_write_ase_tsv() {
    std::vector<ASEEntry> entries;

    // Create a few test entries
    entries.push_back({0, 0, 5, 5, 0.5f});
    entries.push_back({0, 1, 7, 3, 0.3f});
    entries.push_back({1, 0, 6, 4, 0.4f});

    std::vector<std::string> barcodes = {"AAAAA", "BBBBB"};
    std::vector<std::string> snp_names = {"SNP0", "SNP1"};

    std::string path = "/tmp/test_ase_output.tsv";
    write_ase_tsv(path, entries, barcodes, snp_names);

    // Read and verify
    std::ifstream f(path);
    assert(f.good() && "TSV file created");

    std::string line;

    // Check header
    std::getline(f, line);
    assert(line == "barcode\tsnp_id\tref_count\talt_count\tallelic_ratio" &&
           "Incorrect header");

    // Check first data line
    std::getline(f, line);
    assert(line.find("AAAAA") != std::string::npos && "Missing barcode");
    assert(line.find("SNP0") != std::string::npos && "Missing SNP ID");
    assert(line.find("5") != std::string::npos && "Missing ref_count");
    assert(line.find("0.5000") != std::string::npos && "Missing allelic_ratio");

    // Check second data line
    std::getline(f, line);
    assert(line.find("AAAAA") != std::string::npos);
    assert(line.find("SNP1") != std::string::npos);
    assert(line.find("7") != std::string::npos);
    assert(line.find("0.3000") != std::string::npos);

    // Check third data line
    std::getline(f, line);
    assert(line.find("BBBBB") != std::string::npos);
    assert(line.find("SNP0") != std::string::npos);

    f.close();

    std::cout << "✓ test_write_ase_tsv: valid TSV file with correct header\n";
}

// Test 8: Varying allelic ratios
void test_varying_allelic_ratios() {
    // Test het sites with different AD/DP combinations
    std::vector<uint32_t> snp_indices = {0, 1, 2, 3};
    std::vector<uint8_t> ad_vals = {1, 5, 8, 9};     // AD values
    std::vector<uint8_t> dp_vals = {10, 10, 10, 10}; // DP=10 for all

    CSCMatrix ad_csc = create_csc_single_col(4, snp_indices, ad_vals);
    CSCMatrix dp_csc = create_csc_single_col(4, snp_indices, dp_vals);

    std::vector<ASEEntry> entries = compute_ase(ad_csc, dp_csc, 10);

    // All 4 should be het sites (AD >= 1, ref >= 1)
    assert(entries.size() == 4 && "Expected 4 het entries");

    // Check ratios: AD/DP
    assert(std::abs(entries[0].allelic_ratio - 0.1f) < 0.01f &&
           "Expected ratio≈0.1");
    assert(std::abs(entries[1].allelic_ratio - 0.5f) < 0.01f &&
           "Expected ratio≈0.5");
    assert(std::abs(entries[2].allelic_ratio - 0.8f) < 0.01f &&
           "Expected ratio≈0.8");
    assert(std::abs(entries[3].allelic_ratio - 0.9f) < 0.01f &&
           "Expected ratio≈0.9");

    std::cout << "✓ test_varying_allelic_ratios: all ratios correct\n";
}

int main() {
    std::cout << "Running ASE tests...\n\n";

    test_empty_matrices();
    test_single_het_site();
    test_homozygous_ref_filtered();
    test_homozygous_alt_filtered();
    test_low_depth_filtered();
    test_multiple_cells_and_sites();
    test_write_ase_tsv();
    test_varying_allelic_ratios();

    std::cout << "\n✓ All ASE tests passed!\n";
    return 0;
}
