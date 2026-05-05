// test_cascade_stats_writer.cpp
// Unit tests for cascade_stats_writer.h

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "singlet/pileup/cascade_stats_writer.h"

using namespace singlet;

// Utility: read entire file into string
std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Test 1: Default-constructed CascadeStats has reasonable defaults
void test_defaults() {
    CascadeStats cs;
    assert(cs.L1_txome.reads_in == 0);
    assert(cs.L1_txome.resolved == 0);
    assert(cs.L1_txome.passthrough == 0);
    assert(cs.L2_te.reads_in == 0);
    assert(cs.L3_star.mapped == 0);
    assert(cs.L4_nonhost.unmappable == 0);
    assert(cs.em_seed == 0xC0FFEE);
    assert(cs.cascade_enabled == true);
    assert(cs.deterministic == true);
    std::cout << "✓ test_defaults passed\n";
}

// Test 2: write_cascade_stats creates file at out_prefix/cascade_stats.json
void test_file_creation() {
    system("mkdir -p /tmp/test_cascade_out");
    
    CascadeStats cs;
    cs.L1_txome.reads_in = 1000;
    bool result = write_cascade_stats("/tmp/test_cascade_out", cs);
    assert(result == true);
    
    // Verify file exists
    std::string path = "/tmp/test_cascade_out/cascade_stats.json";
    std::ifstream f(path);
    assert(f.good());
    f.close();
    std::cout << "✓ test_file_creation passed\n";
}

// Test 3: Output contains expected JSON keys
void test_json_keys() {
    system("mkdir -p /tmp/test_cascade_out");
    
    CascadeStats cs;
    write_cascade_stats("/tmp/test_cascade_out", cs);
    
    std::string content = read_file("/tmp/test_cascade_out/cascade_stats.json");
    assert(content.find("\"schema_version\"") != std::string::npos);
    assert(content.find("\"layers\"") != std::string::npos);
    assert(content.find("\"L1_txome\"") != std::string::npos);
    assert(content.find("\"L2_te\"") != std::string::npos);
    assert(content.find("\"L3_star\"") != std::string::npos);
    assert(content.find("\"L4_nonhost\"") != std::string::npos);
    assert(content.find("\"cascade_enabled\"") != std::string::npos);
    assert(content.find("\"deterministic\"") != std::string::npos);
    assert(content.find("\"em_seed\"") != std::string::npos);
    assert(content.find("\"timing_seconds\"") != std::string::npos);
    assert(content.find("\"peak_rss_gb_per_layer\"") != std::string::npos);
    std::cout << "✓ test_json_keys passed\n";
}

// Test 4: Layer read counts are correct in output
void test_layer_counts() {
    system("mkdir -p /tmp/test_cascade_out");
    
    CascadeStats cs;
    cs.L1_txome.reads_in = 5000;
    cs.L1_txome.resolved = 3000;
    cs.L1_txome.passthrough = 2000;
    cs.L2_te.reads_in = 2000;
    cs.L2_te.resolved = 1500;
    cs.L2_te.passthrough = 500;
    cs.L3_star.reads_in = 500;
    cs.L3_star.mapped = 400;
    cs.L3_star.unmapped = 100;
    cs.L4_nonhost.reads_in = 100;
    cs.L4_nonhost.resolved = 50;
    cs.L4_nonhost.unmappable = 50;
    
    write_cascade_stats("/tmp/test_cascade_out", cs);
    std::string content = read_file("/tmp/test_cascade_out/cascade_stats.json");
    
    assert(content.find("\"reads_in\": 5000") != std::string::npos);
    assert(content.find("\"resolved\": 3000") != std::string::npos);
    assert(content.find("\"passthrough\": 2000") != std::string::npos);
    assert(content.find("\"mapped\": 400") != std::string::npos);
    assert(content.find("\"unmapped\": 100") != std::string::npos);
    assert(content.find("\"unmappable\": 50") != std::string::npos);
    std::cout << "✓ test_layer_counts passed\n";
}

// Test 5: cascade_enabled/deterministic flags serialize correctly
void test_flags() {
    system("mkdir -p /tmp/test_cascade_out");
    
    CascadeStats cs1;
    cs1.cascade_enabled = true;
    cs1.deterministic = true;
    write_cascade_stats("/tmp/test_cascade_out", cs1);
    std::string content1 = read_file("/tmp/test_cascade_out/cascade_stats.json");
    assert(content1.find("\"cascade_enabled\": true") != std::string::npos);
    assert(content1.find("\"deterministic\": true") != std::string::npos);
    
    CascadeStats cs2;
    cs2.cascade_enabled = false;
    cs2.deterministic = false;
    write_cascade_stats("/tmp/test_cascade_out", cs2);
    std::string content2 = read_file("/tmp/test_cascade_out/cascade_stats.json");
    assert(content2.find("\"cascade_enabled\": false") != std::string::npos);
    assert(content2.find("\"deterministic\": false") != std::string::npos);
    std::cout << "✓ test_flags passed\n";
}

// Test 6: em_seed is serialized as hex string
void test_em_seed_hex() {
    system("mkdir -p /tmp/test_cascade_out");
    
    CascadeStats cs;
    cs.em_seed = 0xC0FFEE;
    write_cascade_stats("/tmp/test_cascade_out", cs);
    std::string content = read_file("/tmp/test_cascade_out/cascade_stats.json");
    
    // Should be serialized as "0xC0FFEE"
    assert(content.find("\"0xC0FFEE\"") != std::string::npos);
    
    // Test with different seed
    cs.em_seed = 0xDEADBEEF;
    write_cascade_stats("/tmp/test_cascade_out", cs);
    content = read_file("/tmp/test_cascade_out/cascade_stats.json");
    assert(content.find("\"0xDEADBEEF\"") != std::string::npos);
    std::cout << "✓ test_em_seed_hex passed\n";
}

// Test 7: write_cascade_stats returns false for invalid path
void test_invalid_path() {
    CascadeStats cs;
    // Use a directory that doesn't exist and can't be created
    bool result = write_cascade_stats("/nonexistent/deeply/nested/path/that/does/not/exist", cs);
    assert(result == false);
    std::cout << "✓ test_invalid_path passed\n";
}

// Test 8: CascadeLayerStats defaults
void test_layer_stats_defaults() {
    CascadeLayerStats cls;
    assert(cls.reads_in == 0);
    assert(cls.resolved == 0);
    assert(cls.passthrough == 0);
    assert(cls.mapped == 0);
    assert(cls.unmapped == 0);
    assert(cls.unmappable == 0);
    assert(cls.wall_seconds == 0.0);
    assert(cls.peak_rss_gb == 0.0);
    std::cout << "✓ test_layer_stats_defaults passed\n";
}

// Test 9: Timing and memory metrics serialization
void test_timing_metrics() {
    system("mkdir -p /tmp/test_cascade_out");
    
    CascadeStats cs;
    cs.L1_txome.wall_seconds = 1.5;
    cs.L2_te.wall_seconds = 2.3;
    cs.L3_star.wall_seconds = 5.7;
    cs.L4_nonhost.wall_seconds = 0.8;
    cs.L1_txome.peak_rss_gb = 2.5;
    cs.L2_te.peak_rss_gb = 3.1;
    cs.L3_star.peak_rss_gb = 4.2;
    cs.L4_nonhost.peak_rss_gb = 1.8;
    
    write_cascade_stats("/tmp/test_cascade_out", cs);
    std::string content = read_file("/tmp/test_cascade_out/cascade_stats.json");
    
    assert(content.find("\"timing_seconds\"") != std::string::npos);
    assert(content.find("\"peak_rss_gb_per_layer\"") != std::string::npos);
    // Check that values are present (simple presence check, not exact parsing)
    assert(content.find("1.5") != std::string::npos);
    assert(content.find("2.3") != std::string::npos);
    std::cout << "✓ test_timing_metrics passed\n";
}

// Test 10: Valid JSON structure (basic parsing)
void test_json_structure() {
    system("mkdir -p /tmp/test_cascade_out");
    
    CascadeStats cs;
    cs.L1_txome.reads_in = 100;
    write_cascade_stats("/tmp/test_cascade_out", cs);
    std::string content = read_file("/tmp/test_cascade_out/cascade_stats.json");
    
    // Check for balanced braces
    int open = 0, close = 0;
    for (char c : content) {
        if (c == '{') open++;
        if (c == '}') close++;
    }
    assert(open == close);
    
    // Check it starts with { and has closing }
    assert(content[0] == '{');
    // Find last non-whitespace character
    size_t last = content.find_last_not_of(" \t\n\r");
    assert(last != std::string::npos);
    assert(content[last] == '}');
    std::cout << "✓ test_json_structure passed\n";
}

int main() {
    try {
        test_defaults();
        test_file_creation();
        test_json_keys();
        test_layer_counts();
        test_flags();
        test_em_seed_hex();
        test_invalid_path();
        test_layer_stats_defaults();
        test_timing_metrics();
        test_json_structure();
        
        std::cout << "\n✓ All 10 tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
