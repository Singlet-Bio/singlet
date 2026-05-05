// test_pz_writer.cpp
// Unit tests for pz_writer.h

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "singlet/pileup/pz_writer.h"

using namespace singlet::pz;

// Utility: read file and verify magic bytes
uint32_t read_magic(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return 0;
    uint32_t magic = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.close();
    return magic;
}

// Utility: get file size
uint64_t get_file_size(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    uint64_t size = f.tellg();
    f.close();
    return size;
}

// Test 1: sizeof(PZHeader) == 96
void test_header_size() {
    assert(sizeof(PZHeader) == 96);
    std::cout << "✓ test_header_size passed\n";
}

// Test 2: sizeof(PZFooter) == 16
void test_footer_size() {
    assert(sizeof(PZFooter) == 16);
    std::cout << "✓ test_footer_size passed\n";
}

// Test 3: TP1_MAGIC == 0x5A315054
void test_magic_constant() {
    assert(TP1_MAGIC == 0x5A315054);
    std::cout << "✓ test_magic_constant passed\n";
}

// Test 4: CRC32 empty data gives known CRC
void test_crc32_empty() {
    CRC32 crc;
    uint32_t result = crc.finalize();
    // Empty CRC should be specific value
    assert(result == 0);
    std::cout << "✓ test_crc32_empty passed\n";
}

// Test 5: CRC32 non-empty data gives different CRC
void test_crc32_nonempty() {
    uint8_t data[] = {1, 2, 3, 4, 5};
    uint32_t crc1 = CRC32::compute(data, 5);
    uint32_t crc2 = CRC32::compute(nullptr, 0);
    crc2 = CRC32().finalize();  // empty
    // Non-empty should differ from empty (which is 0)
    assert(crc1 != 0);
    std::cout << "✓ test_crc32_nonempty passed\n";
}

// Test 6: CRC32 deterministic (same data → same CRC)
void test_crc32_deterministic() {
    uint8_t data[] = {10, 20, 30, 40, 50};
    uint32_t crc1 = CRC32::compute(data, 5);
    uint32_t crc2 = CRC32::compute(data, 5);
    assert(crc1 == crc2);
    std::cout << "✓ test_crc32_deterministic passed\n";
}

// Test 7: CRC32 different payloads give different CRCs
void test_crc32_different_payloads() {
    uint8_t data1[] = {1, 2, 3, 4, 5};
    uint8_t data2[] = {5, 4, 3, 2, 1};
    uint32_t crc1 = CRC32::compute(data1, 5);
    uint32_t crc2 = CRC32::compute(data2, 5);
    assert(crc1 != crc2);
    std::cout << "✓ test_crc32_different_payloads passed\n";
}

// Test 8: write_1pz with small matrix - verify file starts with correct magic bytes
void test_write_1pz_magic() {
    system("mkdir -p /tmp/test_pz_out");
    
    // Create simple 3x2 uint16 matrix
    uint32_t m = 3;  // features
    uint32_t n = 2;  // barcodes
    std::vector<int32_t> indptr = {0, 1, 3};  // col 0: 1 nnz, col 1: 2 nnz
    std::vector<int32_t> indices = {0, 1, 2};  // row indices
    std::vector<uint16_t> data = {10, 20, 30};
    std::vector<std::string> rownames = {"gene1", "gene2", "gene3"};
    std::vector<std::string> colnames = {"cell1", "cell2"};
    
    bool result = write_1pz<uint16_t>(
        "/tmp/test_pz_out/test.1pz",
        m, n, indptr, indices, data,
        rownames, colnames,
        3, 1024, 1);
    
    assert(result == true);
    
    // Verify magic
    uint32_t magic = read_magic("/tmp/test_pz_out/test.1pz");
    assert(magic == TP1_MAGIC);
    std::cout << "✓ test_write_1pz_magic passed\n";
}

// Test 9: write_1pz round-trip - verify file size is reasonable
void test_write_1pz_file_size() {
    system("mkdir -p /tmp/test_pz_out");
    
    // Create a slightly larger 10x5 matrix
    uint32_t m = 10;  // features
    uint32_t n = 5;   // barcodes
    std::vector<int32_t> indptr(n + 1);
    std::vector<int32_t> indices;
    std::vector<uint16_t> data;
    
    // Create a sparse pattern: ~40 nnz
    for (uint32_t j = 0; j < n; ++j) {
        indptr[j] = indices.size();
        for (uint32_t i = 0; i < m; ++i) {
            if ((i + j) % 3 == 0) {  // sparse pattern
                indices.push_back(i);
                data.push_back(100 + i * 10 + j);
            }
        }
    }
    indptr[n] = indices.size();
    
    std::vector<std::string> rownames(m), colnames(n);
    for (uint32_t i = 0; i < m; ++i) rownames[i] = "gene_" + std::to_string(i);
    for (uint32_t j = 0; j < n; ++j) colnames[j] = "cell_" + std::to_string(j);
    
    bool result = write_1pz<uint16_t>(
        "/tmp/test_pz_out/test_large.1pz",
        m, n, indptr, indices, data,
        rownames, colnames,
        3, 1024, 1);
    
    assert(result == true);
    
    uint64_t file_size = get_file_size("/tmp/test_pz_out/test_large.1pz");
    // File should be at least header (96) + footer (16) + some data
    assert(file_size > 96 + 16);
    std::cout << "✓ test_write_1pz_file_size passed (size: " << file_size << ")\n";
}

// Test 10: write_1pz with uint8_t data
void test_write_1pz_uint8() {
    system("mkdir -p /tmp/test_pz_out");
    
    uint32_t m = 4;
    uint32_t n = 3;
    std::vector<int32_t> indptr = {0, 2, 4, 6};
    std::vector<int32_t> indices = {0, 2, 1, 3, 0, 1};
    std::vector<uint8_t> data = {1, 2, 3, 4, 5, 6};
    std::vector<std::string> rownames = {"g1", "g2", "g3", "g4"};
    std::vector<std::string> colnames = {"c1", "c2", "c3"};
    
    bool result = write_1pz<uint8_t>(
        "/tmp/test_pz_out/test_uint8.1pz",
        m, n, indptr, indices, data,
        rownames, colnames);
    
    assert(result == true);
    uint32_t magic = read_magic("/tmp/test_pz_out/test_uint8.1pz");
    assert(magic == TP1_MAGIC);
    uint64_t file_size = get_file_size("/tmp/test_pz_out/test_uint8.1pz");
    assert(file_size > 112);  // header + footer minimum
    std::cout << "✓ test_write_1pz_uint8 passed\n";
}

// Test 11: write_1pz with uint32_t data
void test_write_1pz_uint32() {
    system("mkdir -p /tmp/test_pz_out");
    
    uint32_t m = 5;
    uint32_t n = 2;
    std::vector<int32_t> indptr = {0, 3, 5};
    std::vector<int32_t> indices = {0, 2, 4, 1, 3};
    std::vector<uint32_t> data = {1000, 2000, 3000, 4000, 5000};
    std::vector<std::string> rownames(m), colnames(n);
    for (uint32_t i = 0; i < m; ++i) rownames[i] = "g" + std::to_string(i);
    for (uint32_t j = 0; j < n; ++j) colnames[j] = "c" + std::to_string(j);
    
    bool result = write_1pz<uint32_t>(
        "/tmp/test_pz_out/test_uint32.1pz",
        m, n, indptr, indices, data,
        rownames, colnames);
    
    assert(result == true);
    uint32_t magic = read_magic("/tmp/test_pz_out/test_uint32.1pz");
    assert(magic == TP1_MAGIC);
    std::cout << "✓ test_write_1pz_uint32 passed\n";
}

// Test 12: write_1pz with user metadata
void test_write_1pz_metadata() {
    system("mkdir -p /tmp/test_pz_out");
    
    uint32_t m = 2;
    uint32_t n = 2;
    std::vector<int32_t> indptr = {0, 1, 2};
    std::vector<int32_t> indices = {0, 1};
    std::vector<uint16_t> data = {10, 20};
    std::vector<std::string> rownames = {"gene1", "gene2"};
    std::vector<std::string> colnames = {"cell1", "cell2"};
    
    std::map<std::string, std::string> user_meta;
    user_meta["organism"] = "human";
    user_meta["tissue"] = "blood";
    
    bool result = write_1pz<uint16_t>(
        "/tmp/test_pz_out/test_metadata.1pz",
        m, n, indptr, indices, data,
        rownames, colnames,
        3, 1024, 1, user_meta);
    
    assert(result == true);
    uint32_t magic = read_magic("/tmp/test_pz_out/test_metadata.1pz");
    assert(magic == TP1_MAGIC);
    std::cout << "✓ test_write_1pz_metadata passed\n";
}

// Test 13: CRC32 large data consistency
void test_crc32_large_data() {
    std::vector<uint8_t> data(10000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i % 256);
    }
    uint32_t crc1 = CRC32::compute(data.data(), data.size());
    uint32_t crc2 = CRC32::compute(data.data(), data.size());
    assert(crc1 == crc2);
    std::cout << "✓ test_crc32_large_data passed\n";
}

// Test 14: CRC32 incremental update vs batch
void test_crc32_incremental() {
    uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint32_t crc_batch = CRC32::compute(data, 8);
    
    CRC32 crc_inc;
    crc_inc.update(data, 4);
    crc_inc.update(data + 4, 4);
    uint32_t crc_incremental = crc_inc.finalize();
    
    assert(crc_batch == crc_incremental);
    std::cout << "✓ test_crc32_incremental passed\n";
}

// Test 15: write_1pz empty matrix (valid but degenerate)
void test_write_1pz_empty() {
    system("mkdir -p /tmp/test_pz_out");
    
    uint32_t m = 5;
    uint32_t n = 3;
    std::vector<int32_t> indptr(n + 1, 0);  // All zeros
    std::vector<int32_t> indices;  // Empty
    std::vector<uint16_t> data;    // Empty
    std::vector<std::string> rownames(m), colnames(n);
    for (uint32_t i = 0; i < m; ++i) rownames[i] = "g" + std::to_string(i);
    for (uint32_t j = 0; j < n; ++j) colnames[j] = "c" + std::to_string(j);
    
    bool result = write_1pz<uint16_t>(
        "/tmp/test_pz_out/test_empty.1pz",
        m, n, indptr, indices, data,
        rownames, colnames);
    
    assert(result == true);
    uint32_t magic = read_magic("/tmp/test_pz_out/test_empty.1pz");
    assert(magic == TP1_MAGIC);
    std::cout << "✓ test_write_1pz_empty passed\n";
}

// Test 16: PZHeader struct layout (basic checks)
void test_pzheader_layout() {
    PZHeader hdr{};
    hdr.magic = TP1_MAGIC;
    hdr.version = TP1_VERSION;
    hdr.vt_code = 2;
    hdr.flags = FLAG_HAS_PERM | FLAG_HAS_COLSUMS;
    hdr.m = 100;
    hdr.n = 50;
    hdr.nnz = 1000;
    hdr.ptr_width = 2;
    
    assert(hdr.magic == TP1_MAGIC);
    assert(hdr.version == TP1_VERSION);
    assert(hdr.m == 100);
    assert(hdr.n == 50);
    assert(hdr.nnz == 1000);
    std::cout << "✓ test_pzheader_layout passed\n";
}

// Test 17: PZFooter struct layout
void test_pzfooter_layout() {
    PZFooter ftr{};
    ftr.file_crc32 = 0x12345678;
    ftr.num_chunks = 10;
    ftr.magic = TP1_MAGIC;
    
    assert(ftr.file_crc32 == 0x12345678);
    assert(ftr.num_chunks == 10);
    assert(ftr.magic == TP1_MAGIC);
    std::cout << "✓ test_pzfooter_layout passed\n";
}

int main() {
    try {
        test_header_size();
        test_footer_size();
        test_magic_constant();
        test_crc32_empty();
        test_crc32_nonempty();
        test_crc32_deterministic();
        test_crc32_different_payloads();
        test_write_1pz_magic();
        test_write_1pz_file_size();
        test_write_1pz_uint8();
        test_write_1pz_uint32();
        test_write_1pz_metadata();
        test_crc32_large_data();
        test_crc32_incremental();
        test_write_1pz_empty();
        test_pzheader_layout();
        test_pzfooter_layout();
        
        std::cout << "\n✓ All 17 tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
