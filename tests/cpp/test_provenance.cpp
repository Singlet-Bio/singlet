#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "singlet/pileup/provenance.h"

using namespace singlet;

// Helper: check if file exists
[[maybe_unused]] static bool file_exists(const std::string& path) {
    struct stat buffer;
    return stat(path.c_str(), &buffer) == 0;
}

// Helper: read entire file into string
static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Helper: check if string contains substring
[[maybe_unused]] static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// Test 1: Empty input_file → no file written
void test_empty_input_file() {
    std::string out_dir = "/tmp/test_prov_1";
    system(("mkdir -p " + out_dir).c_str());
    
    ProvenanceConfig prov;
    prov.input_file = "";  // Empty = no-op
    
    write_provenance_json(out_dir, prov, 10, 5, 1000);
    
    std::string prov_file = out_dir + "/provenance.json";
    assert(!file_exists(prov_file) && "provenance.json should NOT be written when input_file is empty");
    
    std::cout << "✓ Test 1: Empty input_file → no file written\n";
}

// Test 2: Valid config → file written with expected JSON keys
void test_valid_config_writes_file() {
    std::string out_dir = "/tmp/test_prov_2";
    system(("mkdir -p " + out_dir).c_str());
    
    ProvenanceConfig prov;
    prov.input_file = "/data/reads.fq";
    prov.singlify_version = "1.0.0";
    prov.genome_dir = "/ref/GRCh38-2024-A";
    prov.gtf_path = "/ref/GRCh38-2024-A/genes.gtf";
    prov.input_reads = 50000000;
    prov.whitelist_name = "3M-feb-2018";
    
    write_provenance_json(out_dir, prov, 60000, 5000, 100000000);
    
    std::string prov_file = out_dir + "/provenance.json";
    assert(file_exists(prov_file) && "provenance.json should be written");
    
    std::string content = read_file(prov_file);
    assert(!content.empty() && "provenance.json should not be empty");
    
    // Check for expected JSON keys
    assert(contains(content, "\"singlify_version\"") && "Missing singlify_version key");
    assert(contains(content, "\"input\"") && "Missing input block key");
    assert(contains(content, "\"genome\"") && "Missing genome key");
    assert(contains(content, "\"output\"") && "Missing output key");
    assert(contains(content, "\"schema_version\"") && "Missing schema_version key");
    assert(contains(content, "\"timestamp\"") && "Missing timestamp key");
    assert(contains(content, "\"command_line\"") && "Missing command_line key");
    
    std::cout << "✓ Test 2: Valid config writes file with expected JSON keys\n";
}

// Test 3: Verify output JSON contains correct n_cells and total_umis values
void test_output_values() {
    std::string out_dir = "/tmp/test_prov_3";
    system(("mkdir -p " + out_dir).c_str());
    
    ProvenanceConfig prov;
    prov.input_file = "/data/reads.fq";
    
    uint32_t n_cells = 7823;
    uint64_t total_umis = 987654321;
    
    write_provenance_json(out_dir, prov, 50000, n_cells, total_umis);
    
    std::string content = read_file(out_dir + "/provenance.json");
    
    // Check for the specific cell count
    assert(contains(content, "\"cells\": 7823") && "n_cells value not correct in JSON");
    
    // Check for the specific UMI count
    assert(contains(content, "\"total_umis\": 987654321") && "total_umis value not correct in JSON");
    
    std::cout << "✓ Test 3: Output JSON contains correct n_cells and total_umis values\n";
}

// Test 4: Verify timestamp field exists and looks like ISO 8601
void test_timestamp_iso8601() {
    std::string out_dir = "/tmp/test_prov_4";
    system(("mkdir -p " + out_dir).c_str());
    
    ProvenanceConfig prov;
    prov.input_file = "/data/reads.fq";
    
    write_provenance_json(out_dir, prov, 10, 5, 1000);
    
    std::string content = read_file(out_dir + "/provenance.json");
    
    // Check for ISO 8601 timestamp pattern: "YYYY-MM-DDTHH:MM:SSZ"
    assert(contains(content, "\"timestamp\": \"") && "Missing timestamp field");
    
    // Simple pattern check: should contain T and Z for ISO 8601
    size_t ts_pos = content.find("\"timestamp\": \"");
    if (ts_pos != std::string::npos) {
        std::string ts_section = content.substr(ts_pos + 14, 30);  // Extract ~30 chars
        assert(ts_section.find('T') != std::string::npos && "Timestamp missing 'T' (not ISO 8601)");
        assert(ts_section.find('Z') != std::string::npos && "Timestamp missing 'Z' (not ISO 8601)");
        // Check that it looks like a valid date format
        assert(ts_section.find('-') != std::string::npos && "Timestamp missing '-' (invalid format)");
        assert(ts_section.find(':') != std::string::npos && "Timestamp missing ':' (invalid format)");
    }
    
    std::cout << "✓ Test 4: Timestamp field exists and looks like ISO 8601\n";
}

// Test 5: Verify command_line serialization
void test_command_line_serialization() {
    std::string out_dir = "/tmp/test_prov_5";
    system(("mkdir -p " + out_dir).c_str());
    
    ProvenanceConfig prov;
    prov.input_file = "/data/reads.fq";
    prov.command_line = {"singlify", "--arg1", "val1", "positional_arg"};
    
    write_provenance_json(out_dir, prov, 10, 5, 1000);
    
    std::string content = read_file(out_dir + "/provenance.json");
    
    // Check for command_line array with our values
    assert(contains(content, "\"command_line\"") && "Missing command_line field");
    assert(contains(content, "singlify") && "Missing command name");
    assert(contains(content, "--arg1") && "Missing --arg1 argument");
    assert(contains(content, "val1") && "Missing val1 value");
    assert(contains(content, "positional_arg") && "Missing positional_arg");
    
    std::cout << "✓ Test 5: Command_line serialization works correctly\n";
}

// Test 6: Default ProvenanceConfig has reasonable values
void test_default_config_values() {
    ProvenanceConfig prov;
    
    // Check default version
    assert(prov.singlify_version == "0.3.0" && "Default version should be 0.3.0");
    
    // Check default boolean values
    assert(prov.umi_dedup == true && "Default umi_dedup should be true");
    assert(prov.umi_dedup_directional == false && "Default umi_dedup_directional should be false");
    assert(prov.pipeline == false && "Default pipeline should be false");
    assert(prov.cascade_enabled == false && "Default cascade_enabled should be false");
    
    // Check default cascade_mode
    assert(prov.cascade_mode == "off" && "Default cascade_mode should be 'off'");
    assert(prov.te_classify_mode == "off" && "Default te_classify_mode should be 'off'");
    
    // Check default numeric values
    assert(prov.input_reads == 0 && "Default input_reads should be 0");
    assert(prov.threads == 0 && "Default threads should be 0");
    assert(prov.wall_seconds == 0.0 && "Default wall_seconds should be 0.0");
    
    // Check empty strings
    assert(prov.input_file.empty() && "Default input_file should be empty");
    assert(prov.singlify_git_sha == "unknown" && "Default git_sha should be 'unknown'");
    
    std::cout << "✓ Test 6: Default ProvenanceConfig has reasonable values\n";
}

// Test 7: Verify schema_version and singlify_version in output
void test_version_fields() {
    std::string out_dir = "/tmp/test_prov_7";
    system(("mkdir -p " + out_dir).c_str());
    
    ProvenanceConfig prov;
    prov.input_file = "/data/reads.fq";
    prov.singlify_version = "2.0.0";
    prov.singlify_git_sha = "abc123def456";
    
    write_provenance_json(out_dir, prov, 10, 5, 1000);
    
    std::string content = read_file(out_dir + "/provenance.json");
    
    assert(contains(content, "\"schema_version\": \"1.0\"") && "Schema version should be 1.0");
    assert(contains(content, "\"singlify_version\": \"2.0.0\"") && "Singlify version should match provided");
    assert(contains(content, "\"singlify_git_sha\": \"abc123def456\"") && "Git SHA should match provided");
    
    std::cout << "✓ Test 7: Version fields correctly populated\n";
}

// Test 8: Verify JSON structure contains all top-level blocks
void test_json_structure() {
    std::string out_dir = "/tmp/test_prov_8";
    system(("mkdir -p " + out_dir).c_str());
    
    ProvenanceConfig prov;
    prov.input_file = "/data/reads.fq";
    prov.genome_dir = "/ref/GRCh38";
    prov.gtf_path = "/ref/genes.gtf";
    prov.whitelist_name = "3M-feb-2018";
    
    write_provenance_json(out_dir, prov, 100, 50, 50000);
    
    std::string content = read_file(out_dir + "/provenance.json");
    
    // Check for all major JSON blocks
    assert(contains(content, "\"input\"") && "Missing input block");
    assert(contains(content, "\"references\"") && "Missing references block");
    assert(contains(content, "\"parameters\"") && "Missing parameters block");
    assert(contains(content, "\"host\"") && "Missing host block");
    assert(contains(content, "\"env\"") && "Missing env block");
    assert(contains(content, "\"timings\"") && "Missing timings block");
    assert(contains(content, "\"cascade\"") && "Missing cascade block");
    assert(contains(content, "\"output_schema_version\"") && "Missing output_schema_version field");
    
    std::cout << "✓ Test 8: JSON structure contains all expected blocks\n";
}

int main() {
    std::cout << "Running provenance tests...\n";
    
    try {
        test_empty_input_file();
        test_valid_config_writes_file();
        test_output_values();
        test_timestamp_iso8601();
        test_command_line_serialization();
        test_default_config_values();
        test_version_fields();
        test_json_structure();
        
        std::cout << "\n✅ All provenance tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed: " << e.what() << "\n";
        return 1;
    }
}
