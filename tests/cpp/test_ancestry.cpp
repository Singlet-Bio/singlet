// SPDX-License-Identifier: MIT
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/ancestry.h"

using namespace singlet;

// Mock CSC matrix for testing
struct MockCSC8 {
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<uint8_t> data;
    uint32_t nrows;
    uint32_t ncols;
};

// Helper: Create an empty CSC matrix
MockCSC8 create_empty_csc(uint32_t nrows) {
    MockCSC8 m;
    m.nrows = nrows;
    m.ncols = 0;
    m.indptr.resize(1, 0);
    return m;
}

// Helper: Create a CSC matrix with data at specific (row, col) positions
// This builds CSC format with ncols columns
MockCSC8 create_csc_from_entries(uint32_t nrows, uint32_t ncols,
                                  const std::vector<uint32_t>& rows,
                                  const std::vector<uint32_t>& cols,
                                  const std::vector<uint8_t>& vals) {
    assert(rows.size() == cols.size() && cols.size() == vals.size());

    MockCSC8 m;
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

// Simpler helper: build CSC with a single column (single barcode)
MockCSC8 create_csc_single_col(uint32_t nrows,
                                const std::vector<uint32_t>& row_indices,
                                const std::vector<uint8_t>& values) {
    assert(row_indices.size() == values.size());

    MockCSC8 m;
    m.nrows = nrows;
    m.ncols = 1;
    m.indptr = {0, static_cast<int32_t>(row_indices.size())};
    m.indices.assign(row_indices.begin(), row_indices.end());
    m.data.assign(values.begin(), values.end());

    return m;
}

// Test 1: get_aim_panel returns AIMs
void test_aim_panel_size() {
    const auto& aims = get_aim_panel();
    assert(aims.size() == 51 && "Expected 51 AIMs in panel");
    std::cout << "✓ test_aim_panel_size: " << aims.size() << " AIMs\n";
}

// Test 2: AIMs have valid freq arrays (all > 0, < 1)
void test_aim_frequencies() {
    const auto& aims = get_aim_panel();
    for (size_t i = 0; i < aims.size(); ++i) {
        const AIM& aim = aims[i];
        (void)aim;
        for (int p = 0; p < 5; ++p) {
            assert(aim.freq[p] > 0.0 && aim.freq[p] < 1.0 &&
                   "AIM frequency out of range");
        }
    }
    std::cout << "✓ test_aim_frequencies: all valid\n";
}

// Test 3: classify_ancestry with empty matrices → low_data=true
void test_empty_matrices() {
    const auto& aims = get_aim_panel();
    MockCSC8 empty_ad = create_empty_csc(50);
    MockCSC8 empty_dp = create_empty_csc(50);

    AncestryResult result = classify_ancestry(empty_ad, empty_dp, 0, aims);

    assert(result.low_data == true && "Expected low_data=true for empty matrices");
    assert(result.ancestry == "ambiguous" && "Expected ambiguous ancestry");
    std::cout << "✓ test_empty_matrices: correctly handled\n";
}

// Test 4: classify_ancestry with strong EUR signal
void test_eur_signal() {
    const auto& full_aims = get_aim_panel();

    // Use first 10 AIMs
    std::vector<AIM> test_aims(full_aims.begin(), full_aims.begin() + 10);

    // Create CSC with 1 column (1 barcode), 10 rows (10 AIMs)
    std::vector<uint32_t> row_indices;
    std::vector<uint8_t> ad_vals, dp_vals;

    for (uint32_t i = 0; i < 10; ++i) {
        double eur_freq = test_aims[i].freq[static_cast<int>(Pop::EUR)];
        if (eur_freq > 0.7) {
            // Strong EUR signal: set AD to freq * DP
            uint8_t dp = 10;
            uint8_t ad = static_cast<uint8_t>(eur_freq * dp);
            row_indices.push_back(i);
            ad_vals.push_back(ad);
            dp_vals.push_back(dp);
        } else {
            // Even for low-freq AIMs, add some coverage
            row_indices.push_back(i);
            ad_vals.push_back(1);
            dp_vals.push_back(10);
        }
    }

    MockCSC8 ad_csc = create_csc_single_col(10, row_indices, ad_vals);
    MockCSC8 dp_csc = create_csc_single_col(10, row_indices, dp_vals);

    AncestryResult result = classify_ancestry(ad_csc, dp_csc, 0, test_aims);

    // Check: should identify EUR ancestry
    assert(!result.low_data && "Expected sufficient data");
    assert(result.ancestry == "EUR" && "Expected EUR ancestry with strong signal");
    assert(result.confidence > 0.5 && "Expected high confidence");
    std::cout << "✓ test_eur_signal: EUR ancestry detected (confidence="
              << result.confidence << ")\n";
}

// Test 5: classify_ancestry with strong AFR signal
void test_afr_signal() {
    const auto& full_aims = get_aim_panel();

    // Use first 10 AIMs
    std::vector<AIM> test_aims(full_aims.begin(), full_aims.begin() + 10);

    std::vector<uint32_t> row_indices;
    std::vector<uint8_t> ad_vals, dp_vals;

    for (uint32_t i = 0; i < 10; ++i) {
        double afr_freq = test_aims[i].freq[static_cast<int>(Pop::AFR)];
        if (afr_freq > 0.4) {
            uint8_t dp = 10;
            uint8_t ad = static_cast<uint8_t>(afr_freq * dp);
            row_indices.push_back(i);
            ad_vals.push_back(ad);
            dp_vals.push_back(dp);
        } else {
            row_indices.push_back(i);
            ad_vals.push_back(1);
            dp_vals.push_back(10);
        }
    }

    MockCSC8 ad_csc = create_csc_single_col(10, row_indices, ad_vals);
    MockCSC8 dp_csc = create_csc_single_col(10, row_indices, dp_vals);

    AncestryResult result = classify_ancestry(ad_csc, dp_csc, 0, test_aims);

    // Check: should identify AFR ancestry
    assert(!result.low_data && "Expected sufficient data");
    assert(result.ancestry == "AFR" && "Expected AFR ancestry with strong signal");
    std::cout << "✓ test_afr_signal: AFR ancestry detected (confidence="
              << result.confidence << ")\n";
}

// Test 6: classify_ancestry with fewer than 5 informative AIMs → low_data=true
void test_insufficient_informative_aims() {
    const auto& full_aims = get_aim_panel();
    std::vector<AIM> test_aims(full_aims.begin(), full_aims.begin() + 10);

    // Only 3 AIMs with coverage (< 5 required)
    std::vector<uint32_t> row_indices = {0, 1, 2};
    std::vector<uint8_t> ad_vals = {1, 1, 1};
    std::vector<uint8_t> dp_vals = {5, 5, 5};

    MockCSC8 ad_csc = create_csc_single_col(10, row_indices, ad_vals);
    MockCSC8 dp_csc = create_csc_single_col(10, row_indices, dp_vals);

    AncestryResult result = classify_ancestry(ad_csc, dp_csc, 0, test_aims);

    assert(result.low_data == true && "Expected low_data=true with < 5 informative AIMs");
    assert(result.ancestry == "ambiguous" && "Expected ambiguous ancestry");
    std::cout << "✓ test_insufficient_informative_aims: correctly flagged low_data\n";
}

// Test 7: write_ancestry_json creates valid JSON
void test_write_ancestry_json() {
    const auto& aims = get_aim_panel();
    
    AncestryResult result;
    result.ancestry = "EUR";
    result.confidence = 0.95;
    result.n_informative = 45;
    result.n_covered = 40;
    result.low_data = false;
    for (int p = 0; p < 5; ++p) {
        result.log_lik[p] = -100.0 + p * 10.0;
        result.prob[p] = 0.1 + p * 0.05;
    }

    std::string path = "/tmp/test_ancestry_output.json";
    write_ancestry_json(path, result, aims);

    // Read and verify basic structure
    std::ifstream f(path);
    assert(f.good() && "JSON file created");
    
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    f.close();

    // Check for expected keys
    assert(content.find("\"ancestry\"") != std::string::npos && "Missing ancestry key");
    assert(content.find("\"EUR\"") != std::string::npos && "Missing EUR value");
    assert(content.find("\"populations\"") != std::string::npos && "Missing populations key");
    
    std::cout << "✓ test_write_ancestry_json: valid JSON file created\n";
}

// Test 8: AncestryResult probabilities sum to ~1.0
void test_probability_sum() {
    const auto& full_aims = get_aim_panel();
    std::vector<AIM> test_aims(full_aims.begin(), full_aims.begin() + 10);

    std::vector<uint32_t> row_indices;
    std::vector<uint8_t> ad_vals, dp_vals;

    for (uint32_t i = 0; i < 10; ++i) {
        row_indices.push_back(i);
        ad_vals.push_back(3);
        dp_vals.push_back(10);
    }

    MockCSC8 ad_csc = create_csc_single_col(10, row_indices, ad_vals);
    MockCSC8 dp_csc = create_csc_single_col(10, row_indices, dp_vals);

    AncestryResult result = classify_ancestry(ad_csc, dp_csc, 0, test_aims);

    if (!result.low_data) {
        double prob_sum = 0.0;
        for (int p = 0; p < 5; ++p) {
            prob_sum += result.prob[p];
        }
        // Should sum to ~1.0 (within floating point tolerance)
        assert(std::abs(prob_sum - 1.0) < 0.01 &&
               "Probabilities don't sum to 1.0");
        std::cout << "✓ test_probability_sum: probabilities sum to " << prob_sum << "\n";
    }
}

int main() {
    std::cout << "Running ancestry tests...\n\n";

    test_aim_panel_size();
    test_aim_frequencies();
    test_empty_matrices();
    test_eur_signal();
    test_afr_signal();
    test_insufficient_informative_aims();
    test_write_ancestry_json();
    test_probability_sum();

    std::cout << "\n✓ All ancestry tests passed!\n";
    return 0;
}
