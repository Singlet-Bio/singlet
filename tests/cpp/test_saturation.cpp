// Tests for singlet/pileup/saturation.h
#include "singlet/pileup/saturation.h"
#include "singlet/pileup/umi_dedup.h"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace singlet;

// Helper: read TSV file and return lines as vector of vectors
std::vector<std::vector<std::string>> read_tsv(const std::string& path) {
    std::vector<std::vector<std::string>> result;
    std::ifstream f(path);
    if (!f) return result;
    
    std::string line;
    while (std::getline(f, line)) {
        std::vector<std::string> row;
        std::istringstream iss(line);
        std::string cell;
        while (std::getline(iss, cell, '\t')) {
            row.push_back(cell);
        }
        result.push_back(row);
    }
    return result;
}

int main() {
    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: compute_saturation with empty store → empty result
    // ─────────────────────────────────────────────────────────────────────────
    {
        DirectionalUmiStore store;
        auto result = compute_saturation(store, 6);
        assert(result.empty());
        std::cout << "Test 1 PASS: compute_saturation empty store\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: compute_saturation single cell, single UMI recorded 5 times
    // saturation = 1 - (unique_umis / total_reads) = 1 - 1/5 = 0.8
    // ─────────────────────────────────────────────────────────────────────────
    {
        DirectionalUmiStore store;
        uint64_t umi = umi_pack_2bit("ACGTAC", 6);
        
        // Record same UMI 5 times (barcode 0, gene 0, exon 0)
        for (int i = 0; i < 5; ++i) {
            store.record(0, 0, 0, umi);
        }
        
        auto result = compute_saturation(store, 6);
        assert(result.size() == 1);
        assert(result[0].barcode_idx == 0);
        assert(result[0].total_reads == 5);
        assert(result[0].unique_umis == 1);
        
        // saturation = 1 - 1/5 = 0.8
        double expected_sat = 1.0 - 1.0 / 5.0;
        (void)expected_sat;
        assert(std::abs(result[0].saturation - expected_sat) < 1e-6);
        
        // reads_per_umi = 5 / 1 = 5.0
        assert(std::abs(result[0].reads_per_umi - 5.0) < 1e-6);
        
        std::cout << "Test 2 PASS: single cell single UMI, saturation=0.8\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: compute_saturation with two cells → sorted by barcode_idx
    // ─────────────────────────────────────────────────────────────────────────
    {
        DirectionalUmiStore store;
        uint64_t umi1 = umi_pack_2bit("ACGTAC", 6);
        uint64_t umi2 = umi_pack_2bit("TGCATG", 6);
        
        // Barcode 1: record umi1 twice
        store.record(1, 0, 0, umi1);
        store.record(1, 0, 0, umi1);
        
        // Barcode 0: record umi2 three times
        store.record(0, 0, 0, umi2);
        store.record(0, 0, 0, umi2);
        store.record(0, 0, 0, umi2);
        
        auto result = compute_saturation(store, 6);
        assert(result.size() == 2);
        
        // Should be sorted by barcode_idx
        assert(result[0].barcode_idx == 0);
        assert(result[1].barcode_idx == 1);
        
        // Barcode 0: 3 reads, 1 unique UMI, sat = 1 - 1/3 ≈ 0.667
        assert(result[0].total_reads == 3);
        assert(result[0].unique_umis == 1);
        assert(std::abs(result[0].saturation - (1.0 - 1.0/3.0)) < 1e-6);
        
        // Barcode 1: 2 reads, 1 unique UMI, sat = 1 - 1/2 = 0.5
        assert(result[1].total_reads == 2);
        assert(result[1].unique_umis == 1);
        assert(std::abs(result[1].saturation - 0.5) < 1e-6);
        
        std::cout << "Test 3 PASS: two cells sorted by barcode_idx\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4: median_saturation with odd number of cells (3 cells)
    // ─────────────────────────────────────────────────────────────────────────
    {
        std::vector<CellSaturation> cells;
        cells.push_back({0, 10, 2, 0.8, 5.0});   // sat=0.8
        cells.push_back({1, 10, 5, 0.5, 2.0});   // sat=0.5
        cells.push_back({2, 10, 9, 0.1, 1.111});  // sat=0.1
        
        double med = median_saturation(cells);
        (void)med;
        // Sorted: [0.1, 0.5, 0.8] → median = 0.5
        assert(std::abs(med - 0.5) < 1e-6);
        std::cout << "Test 4 PASS: median_saturation odd number (3 cells)\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 5: median_saturation with even number of cells (4 cells)
    // ─────────────────────────────────────────────────────────────────────────
    {
        std::vector<CellSaturation> cells;
        cells.push_back({0, 10, 2, 0.8, 5.0});    // sat=0.8
        cells.push_back({1, 10, 5, 0.5, 2.0});    // sat=0.5
        cells.push_back({2, 10, 9, 0.1, 1.111});   // sat=0.1
        cells.push_back({3, 10, 3, 0.7, 3.333});   // sat=0.7
        
        double med = median_saturation(cells);
        (void)med;
        // Sorted: [0.1, 0.5, 0.7, 0.8] → median = (0.5 + 0.7) / 2 = 0.6
        assert(std::abs(med - 0.6) < 1e-6);
        std::cout << "Test 5 PASS: median_saturation even number (4 cells)\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 6: median_saturation with empty vector → 0.0
    // ─────────────────────────────────────────────────────────────────────────
    {
        std::vector<CellSaturation> cells;
        double med = median_saturation(cells);
        (void)med;
        assert(med == 0.0);
        std::cout << "Test 6 PASS: median_saturation empty vector returns 0.0\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 7: median_saturation with cells that have zero total_reads
    // Should skip cells with total_reads == 0
    // ─────────────────────────────────────────────────────────────────────────
    {
        std::vector<CellSaturation> cells;
        cells.push_back({0, 0, 0, 0.0, 0.0});      // total_reads=0, skip
        cells.push_back({1, 10, 2, 0.8, 5.0});     // sat=0.8
        cells.push_back({2, 10, 5, 0.5, 2.0});     // sat=0.5
        cells.push_back({3, 0, 0, 0.0, 0.0});      // total_reads=0, skip
        
        double med = median_saturation(cells);
        (void)med;
        // Filtered: [0.5, 0.8] → median = (0.5 + 0.8) / 2 = 0.65
        assert(std::abs(med - 0.65) < 1e-6);
        std::cout << "Test 7 PASS: median_saturation filters zero-read cells\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 8: write_saturation_tsv: write to file and verify contents
    // ─────────────────────────────────────────────────────────────────────────
    {
        std::vector<CellSaturation> cells;
        cells.push_back({0, 100, 20, 0.8, 5.0});
        cells.push_back({1, 50, 25, 0.5, 2.0});
        
        std::vector<std::string> barcodes = {"ACGTACGT", "TGCATGCA"};
        
        std::string tmpfile = "/tmp/test_sat_output.tsv";
        write_saturation_tsv(tmpfile, cells, barcodes);
        
        // Read back and verify
        auto lines = read_tsv(tmpfile);
        assert(lines.size() == 3);  // header + 2 data rows
        
        // Check header
        assert(lines[0].size() == 5);
        assert(lines[0][0] == "barcode");
        assert(lines[0][1] == "total_reads");
        assert(lines[0][2] == "unique_umis");
        assert(lines[0][3] == "saturation");
        assert(lines[0][4] == "reads_per_umi");
        
        // Check first data row
        assert(lines[1][0] == "ACGTACGT");
        assert(lines[1][1] == "100");
        assert(lines[1][2] == "20");
        
        // Check second data row
        assert(lines[2][0] == "TGCATGCA");
        assert(lines[2][1] == "50");
        assert(lines[2][2] == "25");
        
        std::cout << "Test 8 PASS: write_saturation_tsv file format\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 9: compute_saturation with multiple genes per cell
    // ─────────────────────────────────────────────────────────────────────────
    {
        DirectionalUmiStore store;
        uint64_t umi_gene0 = umi_pack_2bit("ACGTAC", 6);
        uint64_t umi_gene1 = umi_pack_2bit("TGCATG", 6);
        
        // Barcode 0, gene 0: 4 reads, 1 UMI
        for (int i = 0; i < 4; ++i) {
            store.record(0, 0, 0, umi_gene0);
        }
        
        // Barcode 0, gene 1: 6 reads, 1 UMI
        for (int i = 0; i < 6; ++i) {
            store.record(0, 1, 1, umi_gene1);
        }
        
        auto result = compute_saturation(store, 6);
        assert(result.size() == 1);
        assert(result[0].barcode_idx == 0);
        
        // Total: 4 + 6 = 10 reads, 2 unique UMIs
        assert(result[0].total_reads == 10);
        assert(result[0].unique_umis == 2);
        
        // saturation = 1 - 2/10 = 0.8
        assert(std::abs(result[0].saturation - 0.8) < 1e-6);
        assert(std::abs(result[0].reads_per_umi - 5.0) < 1e-6);
        
        std::cout << "Test 9 PASS: multiple genes per cell\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 10: compute_saturation preserves cell order and computes correctly
    // ─────────────────────────────────────────────────────────────────────────
    {
        DirectionalUmiStore store;
        uint64_t umi = umi_pack_2bit("AAAA", 4);
        
        // Add cells in non-sorted order (5, 2, 8)
        store.record(5, 0, 0, umi);
        store.record(5, 0, 0, umi);
        
        store.record(2, 0, 0, umi);
        
        store.record(8, 0, 0, umi);
        store.record(8, 0, 0, umi);
        store.record(8, 0, 0, umi);
        
        auto result = compute_saturation(store, 4);
        assert(result.size() == 3);
        
        // Should be sorted: 2, 5, 8
        assert(result[0].barcode_idx == 2);
        assert(result[1].barcode_idx == 5);
        assert(result[2].barcode_idx == 8);
        
        // Verify values
        assert(result[0].total_reads == 1);
        assert(result[1].total_reads == 2);
        assert(result[2].total_reads == 3);
        
        std::cout << "Test 10 PASS: cells sorted by barcode_idx\n";
    }

    std::cout << "\n✓ All saturation tests passed!\n";
    return 0;
}
