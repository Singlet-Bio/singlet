// test_fragments_writer.cpp — unit tests for fragments_writer.h
// Tests: file creation, bgzf validity, data correctness, sort, barcode suffix,
// empty input, multiple chromosomes, fragment count field, sort function.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet-pileup/fragments_writer.h"

// ---------------------------------------------------------------------------
// Test framework
// ---------------------------------------------------------------------------
static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (cond) {                                                              \
            std::cout << "  PASS: " << (msg) << "\n";                           \
            ++n_pass;                                                            \
        } else {                                                                 \
            std::cout << "  FAIL: " << (msg) << " [line " << __LINE__ << "]\n"; \
            ++n_fail;                                                            \
        }                                                                        \
    } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string bgzf_read_all(const std::string& path) {
    BGZF* fp = bgzf_open(path.c_str(), "r");
    if (!fp) return {};
    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = bgzf_read(fp, buf, sizeof(buf))) > 0)
        out.append(buf, static_cast<size_t>(n));
    bgzf_close(fp);
    return out;
}

static std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> lines;
    std::istringstream ss(s);
    std::string line;
    while (std::getline(ss, line))
        if (!line.empty()) lines.push_back(line);
    return lines;
}

static std::vector<std::string> split_tab(const std::string& s) {
    std::vector<std::string> fields;
    std::istringstream ss(s);
    std::string f;
    while (std::getline(ss, f, '\t')) fields.push_back(f);
    return fields;
}

static void rm(const std::string& path) { std::remove(path.c_str()); }

// ---------------------------------------------------------------------------
// T1: File creation + bgzf magic bytes
// ---------------------------------------------------------------------------
static void test_file_creation_bgzf_valid() {
    std::cout << "\n=== T1: File creation + bgzf validity ===\n";
    const std::string path = "/tmp/test_fw_t1.tsv.gz";

    singlet::FragmentsWriteConfig cfg;
    cfg.filepath = path;
    cfg.compress = true;
    cfg.index    = false;
    cfg.append_barcode_suffix = true;

    singlet::FragmentsWriter w;
    bool opened = w.open(cfg);
    CHECK(opened, "bgzf file opened for writing");  // 1

    singlet::Fragment f{"chr1", 1000, 1400, "AACCGGTTAACCGGTT", 1};
    bool wrote = w.write_fragment(f);
    CHECK(wrote, "write_fragment returns true");  // 2
    w.close();

    std::FILE* fp = std::fopen(path.c_str(), "rb");
    CHECK(fp != nullptr, "output file created");  // 3
    if (fp) {
        uint8_t magic[2] = {};
        size_t n = std::fread(magic, 1, 2, fp);
        CHECK(n == 2, "read 2 magic bytes");  // 4
        CHECK(magic[0] == 0x1f && magic[1] == 0x8b, "bgzf magic bytes 1f 8b");  // 5
        std::fclose(fp);
    }
    rm(path);
}

// ---------------------------------------------------------------------------
// T2: Fragment data correctness (read-back)
// ---------------------------------------------------------------------------
static void test_data_correctness() {
    std::cout << "\n=== T2: Data correctness ===\n";
    const std::string path = "/tmp/test_fw_t2.tsv.gz";

    singlet::FragmentsWriteConfig cfg;
    cfg.filepath = path;
    cfg.compress = true;
    cfg.index    = false;
    cfg.append_barcode_suffix = false;

    singlet::FragmentsWriter w;
    w.open(cfg);
    singlet::Fragment f{"chr1", 10025, 10425, "AAACGAAAGCGTATGG-1", 2};
    w.write_fragment(f);
    w.close();

    std::string content = bgzf_read_all(path);
    auto lines = split_lines(content);
    CHECK(lines.size() == 1, "one fragment line");  // 6
    if (!lines.empty()) {
        auto fields = split_tab(lines[0]);
        CHECK(fields.size() == 5,   "5 tab fields");            // 7
        CHECK(fields[0] == "chr1",  "chrom=chr1");              // 8
        CHECK(fields[1] == "10025", "start=10025");             // 9
        CHECK(fields[2] == "10425", "end=10425");               // 10
        CHECK(fields[3] == "AAACGAAAGCGTATGG-1", "barcode");   // 11
        CHECK(fields[4] == "2",     "count=2");                 // 12
    }
    rm(path);
}

// ---------------------------------------------------------------------------
// T3: Barcode "-1" suffix
// ---------------------------------------------------------------------------
static void test_barcode_suffix() {
    std::cout << "\n=== T3: Barcode suffix ===\n";
    const std::string path = "/tmp/test_fw_t3.tsv.gz";

    singlet::FragmentsWriteConfig cfg;
    cfg.filepath = path;
    cfg.compress = true;
    cfg.index    = false;
    cfg.append_barcode_suffix = true;

    singlet::FragmentsWriter w;
    w.open(cfg);
    singlet::Fragment f{"chr2", 5000, 5500, "ACTGACTGACTGACTG", 1};
    w.write_fragment(f);
    w.close();

    auto lines = split_lines(bgzf_read_all(path));
    CHECK(!lines.empty(), "fragment written");  // 13
    if (!lines.empty()) {
        auto fields = split_tab(lines[0]);
        CHECK(fields.size() >= 4, "at least 4 fields");  // 14
        CHECK(fields.size() >= 4 && fields[3] == "ACTGACTGACTGACTG-1",
              "barcode has -1 suffix");  // 15
    }
    rm(path);
}

// ---------------------------------------------------------------------------
// T4: Empty fragment list — no crash
// ---------------------------------------------------------------------------
static void test_empty_fragments() {
    std::cout << "\n=== T4: Empty fragments ===\n";
    const std::string path = "/tmp/test_fw_t4.tsv.gz";

    singlet::FragmentsWriteConfig cfg;
    cfg.filepath = path;
    cfg.compress = true;
    cfg.index    = false;
    cfg.append_barcode_suffix = true;

    singlet::FragmentsWriter w;
    w.open(cfg);
    bool ok = w.write_fragments({});
    CHECK(ok,  "write_fragments({}) no crash");  // 16
    w.close();

    CHECK(w.fragments_written() == 0, "0 fragments written");  // 17
    std::string content = bgzf_read_all(path);
    CHECK(content.empty(), "empty decompressed content");  // 18
    rm(path);
}

// ---------------------------------------------------------------------------
// T5: Multiple chromosomes
// ---------------------------------------------------------------------------
static void test_multiple_chromosomes() {
    std::cout << "\n=== T5: Multiple chromosomes ===\n";
    const std::string path = "/tmp/test_fw_t5.tsv.gz";

    singlet::FragmentsWriteConfig cfg;
    cfg.filepath = path;
    cfg.compress = true;
    cfg.index    = false;
    cfg.append_barcode_suffix = false;

    std::vector<singlet::Fragment> frags = {
        {"chr1",  100,  400,  "AACCGGTTAACCGGTT-1", 1},
        {"chr2",  200,  600,  "TTGGCCAATTGGCCAA-1", 2},
        {"chr10", 5000, 5800, "GGTTAACCGGTTAACC-1", 3},
    };
    singlet::FragmentsWriter w;
    w.open(cfg);
    w.write_fragments(frags);
    w.close();

    CHECK(w.fragments_written() == 3, "fragments_written==3");  // 19
    auto lines = split_lines(bgzf_read_all(path));
    CHECK(lines.size() == 3, "3 lines");  // 20
    if (lines.size() >= 3) {
        CHECK(split_tab(lines[0])[0] == "chr1",  "line0 chr1");   // 21
        CHECK(split_tab(lines[1])[0] == "chr2",  "line1 chr2");   // 22
        CHECK(split_tab(lines[2])[0] == "chr10", "line2 chr10");  // 23
    }
    rm(path);
}

// ---------------------------------------------------------------------------
// T6: Fragment count (duplicate_count) field
// ---------------------------------------------------------------------------
static void test_fragment_count_field() {
    std::cout << "\n=== T6: Fragment count field ===\n";
    const std::string path = "/tmp/test_fw_t6.tsv.gz";

    singlet::FragmentsWriteConfig cfg;
    cfg.filepath = path;
    cfg.compress = true;
    cfg.index    = false;
    cfg.append_barcode_suffix = false;

    std::vector<singlet::Fragment> frags = {
        {"chr1", 1000, 1200, "BC1-1",  5},
        {"chr1", 2000, 2400, "BC2-1", 10},
        {"chr1", 3000, 3800, "BC3-1",  1},
    };
    singlet::FragmentsWriter w;
    w.open(cfg);
    w.write_fragments(frags);
    w.close();

    auto lines = split_lines(bgzf_read_all(path));
    CHECK(lines.size() == 3, "3 lines");  // 24
    if (lines.size() >= 3) {
        CHECK(split_tab(lines[0])[4] == "5",  "count=5");   // 25
        CHECK(split_tab(lines[1])[4] == "10", "count=10");  // 26
        CHECK(split_tab(lines[2])[4] == "1",  "count=1");   // 27
    }
    rm(path);
}

// ---------------------------------------------------------------------------
// T7: sort_fragments correctness
// ---------------------------------------------------------------------------
static void test_sort_function() {
    std::cout << "\n=== T7: sort_fragments ===\n";
    std::vector<singlet::Fragment> frags = {
        {"chr2",  500,  700,  "BC3-1", 1},
        {"chr1",  200,  400,  "BC1-1", 1},
        {"chr1",  100,  300,  "BC2-1", 1},
        {"chr10", 1000, 1500, "BC4-1", 1},
    };
    singlet::sort_fragments(frags);

    CHECK(frags[0].chrom == "chr1"  && frags[0].start == 100,  "chr1:100 first");   // 28
    CHECK(frags[1].chrom == "chr1"  && frags[1].start == 200,  "chr1:200 second");  // 29
    CHECK(frags[2].chrom == "chr10" && frags[2].start == 1000, "chr10:1000 third"); // 30
    CHECK(frags[3].chrom == "chr2"  && frags[3].start == 500,  "chr2:500 fourth");  // 31
}

// ---------------------------------------------------------------------------
// T8: Sorted input preserved through write
// ---------------------------------------------------------------------------
static void test_coordinate_sort_write() {
    std::cout << "\n=== T8: Coordinate sort preserved ===\n";
    const std::string path = "/tmp/test_fw_t8.tsv.gz";

    singlet::FragmentsWriteConfig cfg;
    cfg.filepath = path;
    cfg.compress = true;
    cfg.index    = false;
    cfg.append_barcode_suffix = false;

    std::vector<singlet::Fragment> frags = {
        {"chr1", 100, 300, "ABC-1", 1},
        {"chr1", 200, 400, "DEF-1", 2},
        {"chr1", 300, 500, "GHI-1", 1},
    };
    singlet::sort_fragments(frags);

    singlet::FragmentsWriter w;
    w.open(cfg);
    w.write_fragments(frags);
    w.close();

    auto lines = split_lines(bgzf_read_all(path));
    CHECK(lines.size() == 3, "3 lines after sort+write");  // 32
    bool sorted = true;
    int64_t prev = -1;
    for (const auto& line : lines) {
        auto fields = split_tab(line);
        if (fields.size() < 2) { sorted = false; break; }
        int64_t s = std::stoll(fields[1]);
        if (s < prev) { sorted = false; break; }
        prev = s;
    }
    CHECK(sorted, "start positions non-decreasing");  // 33
    rm(path);
}

// ---------------------------------------------------------------------------
// T9: fragments_written() counter
// ---------------------------------------------------------------------------
static void test_fragments_written_counter() {
    std::cout << "\n=== T9: fragments_written() counter ===\n";
    const std::string path = "/tmp/test_fw_t9.tsv.gz";

    singlet::FragmentsWriteConfig cfg;
    cfg.filepath = path;
    cfg.compress = true;
    cfg.index    = false;
    cfg.append_barcode_suffix = false;

    singlet::FragmentsWriter w;
    w.open(cfg);
    for (int i = 0; i < 7; ++i) {
        singlet::Fragment f{"chr1",
                            static_cast<int64_t>(i * 100),
                            static_cast<int64_t>(i * 100 + 200),
                            "BC-1", 1};
        w.write_fragment(f);
    }
    CHECK(w.fragments_written() == 7, "counter==7 before close");  // 34
    w.close();
    CHECK(w.fragments_written() == 7, "counter==7 after close");   // 35
    rm(path);
}

// ---------------------------------------------------------------------------
// T10: Open failure — graceful
// ---------------------------------------------------------------------------
static void test_open_failure() {
    std::cout << "\n=== T10: Open failure graceful ===\n";
    singlet::FragmentsWriter w;
    singlet::FragmentsWriteConfig cfg;
    cfg.filepath = "/nonexistent_dir/nope.tsv.gz";
    cfg.compress = true;
    cfg.index    = false;

    bool ok = w.open(cfg);
    CHECK(!ok, "open returns false for bad path");  // 36

    singlet::Fragment f{"chr1", 0, 100, "BC", 1};
    bool wrote = w.write_fragment(f);
    CHECK(!wrote, "write_fragment false when not open");  // 37
}

int main() {
    test_file_creation_bgzf_valid();
    test_data_correctness();
    test_barcode_suffix();
    test_empty_fragments();
    test_multiple_chromosomes();
    test_fragment_count_field();
    test_sort_function();
    test_coordinate_sort_write();
    test_fragments_written_counter();
    test_open_failure();

    std::cout << "\n──────────────────────────\n";
    std::cout << "PASSED: " << n_pass << "\n";
    std::cout << "FAILED: " << n_fail << "\n";
    return n_fail ? 1 : 0;
}
