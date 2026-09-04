// SPDX-License-Identifier: MIT
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <zlib.h>

#include "mtx_writer.h"

// Helper function to read gzipped file content
std::string read_gzip_content(const std::string& path) {
    gzFile gz = gzopen(path.c_str(), "rb");
    if (!gz) {
        return "";
    }
    
    std::string content;
    char buf[4096];
    int n;
    while ((n = gzread(gz, buf, sizeof(buf))) > 0) {
        content.append(buf, n);
    }
    gzclose(gz);
    return content;
}

// Helper function to count lines in content
int count_lines(const std::string& content) {
    int count = 0;
    for (size_t i = 0; i < content.size(); ++i) {
        if (content[i] == '\n') {
            count++;
        }
    }
    return count;
}

void test_write_mtx_integer() {
    std::cout << "TEST: write_mtx with integer data\n";
    
    // Create a 3x2 CSC matrix:
    // col 0: row 0 = 5, row 2 = 3
    // col 1: row 1 = 7
    uint32_t n_rows = 3;
    uint32_t n_cols = 2;
    std::vector<int32_t> indptr = {0, 2, 3};  // col pointers
    std::vector<int32_t> indices = {0, 2, 1};  // row indices
    std::vector<int64_t> data = {5, 3, 7};     // values
    
    std::string path = "/tmp/test_mtx_writer/test_int.mtx.gz";
    bool result = singlet::write_mtx(path, n_rows, n_cols, indptr, indices, data);
    (void)result;
    
    assert(result && "write_mtx should return true");
    
    // Read back the gzipped content
    std::string content = read_gzip_content(path);
    assert(!content.empty() && "File should not be empty");
    
    // Check header line contains "integer"
    assert(content.find("%%MatrixMarket matrix coordinate integer general") != std::string::npos &&
           "Header should contain 'integer'");
    
    // Check dimensions line
    assert(content.find("3 2 3") != std::string::npos &&
           "Should have dimensions 3 2 3");
    
    // Check matrix entries (1-indexed)
    // row 0, col 0 = 5 -> "1 1 5"
    // row 2, col 0 = 3 -> "3 1 3"
    // row 1, col 1 = 7 -> "2 2 7"
    assert(content.find("1 1 5") != std::string::npos && "Entry (1,1)=5");
    assert(content.find("3 1 3") != std::string::npos && "Entry (3,1)=3");
    assert(content.find("2 2 7") != std::string::npos && "Entry (2,2)=7");
    
    std::cout << "  ✓ write_mtx integer test passed\n";
}

void test_write_mtx_float() {
    std::cout << "TEST: write_mtx with float data\n";
    
    uint32_t n_rows = 2;
    uint32_t n_cols = 2;
    std::vector<int32_t> indptr = {0, 1, 2};
    std::vector<int32_t> indices = {0, 1};
    std::vector<float> data = {1.5f, 2.5f};
    
    std::string path = "/tmp/test_mtx_writer/test_float.mtx.gz";
    bool result = singlet::write_mtx(path, n_rows, n_cols, indptr, indices, data);
    (void)result;
    
    assert(result && "write_mtx should return true");
    
    std::string content = read_gzip_content(path);
    assert(!content.empty() && "File should not be empty");
    
    // Check header line contains "real"
    assert(content.find("%%MatrixMarket matrix coordinate real general") != std::string::npos &&
           "Header should contain 'real' for float");
    
    // Check dimensions
    assert(content.find("2 2 2") != std::string::npos &&
           "Should have dimensions 2 2 2");
    
    std::cout << "  ✓ write_mtx float test passed\n";
}

void test_write_mtx_double() {
    std::cout << "TEST: write_mtx with double data\n";
    
    uint32_t n_rows = 2;
    uint32_t n_cols = 2;
    std::vector<int32_t> indptr = {0, 1, 2};
    std::vector<int32_t> indices = {0, 1};
    std::vector<double> data = {3.14159, 2.71828};
    
    std::string path = "/tmp/test_mtx_writer/test_double.mtx.gz";
    bool result = singlet::write_mtx(path, n_rows, n_cols, indptr, indices, data);
    (void)result;
    
    assert(result && "write_mtx should return true");
    
    std::string content = read_gzip_content(path);
    assert(!content.empty() && "File should not be empty");
    
    // Check header line contains "real"
    assert(content.find("%%MatrixMarket matrix coordinate real general") != std::string::npos &&
           "Header should contain 'real' for double");
    
    std::cout << "  ✓ write_mtx double test passed\n";
}

void test_write_mtx_invalid_path() {
    std::cout << "TEST: write_mtx with invalid path\n";
    
    uint32_t n_rows = 1;
    uint32_t n_cols = 1;
    std::vector<int32_t> indptr = {0, 1};
    std::vector<int32_t> indices = {0};
    std::vector<int32_t> data = {42};
    
    // Try to write to a path that doesn't exist
    std::string invalid_path = "/nonexistent/path/that/should/fail/matrix.mtx.gz";
    bool result = singlet::write_mtx(invalid_path, n_rows, n_cols, indptr, indices, data);
    (void)result;
    
    assert(!result && "write_mtx should return false for invalid path");
    
    std::cout << "  ✓ write_mtx invalid path test passed\n";
}

void test_write_mtx_empty_matrix() {
    std::cout << "TEST: write_mtx with empty matrix (0 nnz)\n";
    
    uint32_t n_rows = 5;
    uint32_t n_cols = 3;
    std::vector<int32_t> indptr = {0, 0, 0, 0};  // All columns empty
    std::vector<int32_t> indices;                 // No entries
    std::vector<int32_t> data;                    // No data
    
    std::string path = "/tmp/test_mtx_writer/test_empty.mtx.gz";
    bool result = singlet::write_mtx(path, n_rows, n_cols, indptr, indices, data);
    (void)result;
    
    assert(result && "write_mtx should return true even for empty matrix");
    
    std::string content = read_gzip_content(path);
    assert(!content.empty() && "File should not be empty (should have header)");
    
    // Check header and dimensions line
    assert(content.find("%%MatrixMarket matrix coordinate integer general") != std::string::npos &&
           "Header should be present");
    assert(content.find("5 3 0") != std::string::npos &&
           "Should have dimensions 5 3 0");
    
    std::cout << "  ✓ write_mtx empty matrix test passed\n";
}

void test_write_names() {
    std::cout << "TEST: write_names\n";
    
    std::vector<std::string> names = {"ENSG00000000003", "ENSG00000000005", "ENSG00000000419"};
    std::string path = "/tmp/test_mtx_writer/test_names.tsv.gz";
    
    bool result = singlet::write_names(path, names);
    (void)result;
    assert(result && "write_names should return true");
    
    std::string content = read_gzip_content(path);
    assert(!content.empty() && "File should not be empty");
    
    // Check line count
    int line_count = count_lines(content);
    (void)line_count;
    assert(line_count == 3 && "Should have exactly 3 lines");
    
    // Check content
    assert(content.find("ENSG00000000003") != std::string::npos);
    assert(content.find("ENSG00000000005") != std::string::npos);
    assert(content.find("ENSG00000000419") != std::string::npos);
    
    std::cout << "  ✓ write_names test passed\n";
}

void test_write_names_empty() {
    std::cout << "TEST: write_names with empty list\n";
    
    std::vector<std::string> names;
    std::string path = "/tmp/test_mtx_writer/test_names_empty.tsv.gz";
    
    bool result = singlet::write_names(path, names);
    (void)result;
    assert(result && "write_names should return true even for empty list");
    
    std::string content = read_gzip_content(path);
    // Empty file is OK
    int line_count = count_lines(content);
    (void)line_count;
    assert(line_count == 0 && "Should have no lines for empty list");
    
    std::cout << "  ✓ write_names empty test passed\n";
}

void test_write_features_10x() {
    std::cout << "TEST: write_features_10x\n";
    
    std::vector<std::string> gene_ids = {"ENSG00000000003", "ENSG00000000005", "ENSG00000000419"};
    std::vector<std::string> gene_names = {"TSPAN6", "TNMD", "DPM1"};
    std::string feature_type = "Gene Expression";
    
    std::string path = "/tmp/test_mtx_writer/test_features.tsv.gz";
    bool result = singlet::write_features_10x(path, gene_ids, gene_names, feature_type);
    (void)result;
    
    assert(result && "write_features_10x should return true");
    
    std::string content = read_gzip_content(path);
    assert(!content.empty() && "File should not be empty");
    
    // Check line count (one per gene)
    int line_count = count_lines(content);
    (void)line_count;
    assert(line_count == 3 && "Should have exactly 3 lines");
    
    // Check format: gene_id\tgene_name\tfeature_type
    // Split by lines and check each one
    std::istringstream iss(content);
    std::string line;
    int idx = 0;
    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        
        // Count tabs
        int tab_count = 0;
        for (char c : line) {
            if (c == '\t') tab_count++;
        }
        assert(tab_count == 2 && "Each line should have 2 tabs (3 columns)");
        
        // Check that line contains expected gene_id
        assert(line.find(gene_ids[idx]) != std::string::npos);
        assert(line.find(gene_names[idx]) != std::string::npos);
        assert(line.find(feature_type) != std::string::npos);
        
        idx++;
    }
    
    std::cout << "  ✓ write_features_10x test passed\n";
}

void test_write_features_10x_with_fallback_names() {
    std::cout << "TEST: write_features_10x with fallback to gene_ids for names\n";
    
    std::vector<std::string> gene_ids = {"ENSG00000000003", "ENSG00000000005"};
    std::vector<std::string> gene_names = {"TSPAN6"};  // Only one name (should use gene_id for second)
    std::string feature_type = "Gene Expression";
    
    std::string path = "/tmp/test_mtx_writer/test_features_fallback.tsv.gz";
    bool result = singlet::write_features_10x(path, gene_ids, gene_names, feature_type);
    (void)result;
    
    assert(result && "write_features_10x should return true");
    
    std::string content = read_gzip_content(path);
    assert(!content.empty() && "File should not be empty");
    
    // Check that first line has the given name
    assert(content.find("ENSG00000000003\tTSPAN6\tGene Expression") != std::string::npos);
    
    // Check that second line falls back to gene_id (since no name provided)
    assert(content.find("ENSG00000000005\tENSG00000000005\tGene Expression") != std::string::npos);
    
    std::cout << "  ✓ write_features_10x fallback test passed\n";
}

void test_write_features_10x_default_type() {
    std::cout << "TEST: write_features_10x with default feature_type\n";
    
    std::vector<std::string> gene_ids = {"ENSG00000000003"};
    std::vector<std::string> gene_names = {"TSPAN6"};
    
    std::string path = "/tmp/test_mtx_writer/test_features_default.tsv.gz";
    bool result = singlet::write_features_10x(path, gene_ids, gene_names);  // No feature_type
    (void)result;
    
    assert(result && "write_features_10x should return true");
    
    std::string content = read_gzip_content(path);
    assert(!content.empty() && "File should not be empty");
    
    // Check that default "Gene Expression" is used
    assert(content.find("Gene Expression") != std::string::npos &&
           "Should use default 'Gene Expression' as feature type");
    
    std::cout << "  ✓ write_features_10x default type test passed\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  MTX Writer Unit Tests\n";
    std::cout << "========================================\n\n";

    // Every test writes into this directory; it does not exist on a fresh CI
    // runner, and write_mtx() (correctly) refuses to create parent dirs.
    std::filesystem::create_directories("/tmp/test_mtx_writer");

    try {
        test_write_mtx_integer();
        test_write_mtx_float();
        test_write_mtx_double();
        test_write_mtx_invalid_path();
        test_write_mtx_empty_matrix();
        test_write_names();
        test_write_names_empty();
        test_write_features_10x();
        test_write_features_10x_with_fallback_names();
        test_write_features_10x_default_type();
        
        std::cout << "\n========================================\n";
        std::cout << "  ✓ All tests passed!\n";
        std::cout << "========================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
