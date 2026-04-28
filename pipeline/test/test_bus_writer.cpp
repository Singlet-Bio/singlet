// test/test_bus_writer.cpp
// Unit tests for G-BUS: BUS format export (bus_writer.h)
// Run: ./test_bus_writer
// No external dependencies — only libc.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "singlet-pileup/bus_writer.h"

using namespace singlet;
namespace fs = std::filesystem;

// ── Counters ─────────────────────────────────────────────────────────────────

static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (cond) {                                                               \
            std::cout << "  PASS: " << (msg) << "\n";                            \
            ++n_pass;                                                             \
        } else {                                                                  \
            std::cout << "  FAIL: " << (msg) << " [line " << __LINE__ << "]\n";  \
            ++n_fail;                                                             \
        }                                                                         \
    } while (0)

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string tmp_path(const char* name) {
    return std::string("/tmp/") + name;
}

// ── Test suite ────────────────────────────────────────────────────────────────

// T1: record struct size
static void test_record_size() {
    std::cout << "\n=== T1: Record size ===\n";
    CHECK(sizeof(BusRecord) == 32, "sizeof(BusRecord) == 32");
}

// T2: encode/decode round-trip — barcode
static void test_encode_decode_barcode() {
    std::cout << "\n=== T2: encode/decode barcode round-trip ===\n";

    // "ACGT" → should encode with distinct bits per base
    uint64_t enc = encode_bus_barcode("ACGT");
    std::string dec = decode_bus_barcode(enc, 4);
    CHECK(dec == "ACGT", "ACGT round-trip");

    // All-A barcode → encoded as 0
    uint64_t enc_a = encode_bus_barcode("AAAA");
    CHECK(enc_a == 0ULL, "AAAA encodes to 0");

    // All-T → all bits set (high 8 bits for 4-mer left-aligned)
    uint64_t enc_t = encode_bus_barcode("TTTT");
    CHECK(enc_t == 0xFF00000000000000ULL, "TTTT encodes to 0xFF00...00");

    // Individual bases
    CHECK((encode_bus_barcode("A") >> 62) == 0, "A → bit pair 0b00");
    CHECK((encode_bus_barcode("C") >> 62) == 1, "C → bit pair 0b01");
    CHECK((encode_bus_barcode("G") >> 62) == 2, "G → bit pair 0b10");
    CHECK((encode_bus_barcode("T") >> 62) == 3, "T → bit pair 0b11");

    // 16-mer 10x barcode round-trip
    std::string bc16 = "ACGTACGTACGTACGT";
    std::string dec16 = decode_bus_barcode(encode_bus_barcode(bc16), 16);
    CHECK(dec16 == bc16, "16-mer 10x barcode round-trip");

    // Lowercase input
    uint64_t enc_lower = encode_bus_barcode("acgt");
    CHECK(enc_lower == enc, "lowercase == uppercase encoding");
}

// T3: encode/decode — UMI (same encoding function, shorter seq)
static void test_encode_decode_umi() {
    std::cout << "\n=== T3: encode/decode UMI round-trip ===\n";

    std::string umi12 = "ACGTACGTACGT";
    uint64_t enc = encode_bus_seq(umi12);
    std::string dec = decode_bus_seq(enc, 12);
    CHECK(dec == umi12, "12-mer UMI round-trip");

    // Verify left-alignment: first nt occupies bits 63-62
    uint64_t enc_c = encode_bus_seq("C");
    CHECK((enc_c >> 62) == 1, "first base C in bits 63-62");
}

// T4: header-only (empty) BUS file
static void test_empty_file() {
    std::cout << "\n=== T4: Empty BUS file (header only) ===\n";
    const std::string path = tmp_path("test_bus_empty.bus");

    BusWriteConfig cfg;
    cfg.filepath    = path;
    cfg.cb_len      = 16;
    cfg.umi_len     = 12;
    cfg.header_text = "";

    {
        BusWriter w;
        bool opened = w.open(cfg);
        CHECK(opened, "open() returns true");
        CHECK(w.records_written() == 0, "records_written() == 0 before close");
        w.close();
    }

    // Verify file exists and has correct size:
    //   4 (magic) + 4 (version) + 4 (cb_len) + 4 (umi_len) + 4 (hlen) = 20 bytes
    auto sz = fs::file_size(path);
    CHECK(sz == 20, "empty BUS file is exactly 20 bytes");

    fs::remove(path);
}

// T5: magic bytes and header fields
static void test_header_verification() {
    std::cout << "\n=== T5: Header magic + fields ===\n";
    const std::string path = tmp_path("test_bus_hdr.bus");

    BusWriteConfig cfg;
    cfg.filepath    = path;
    cfg.cb_len      = 16;
    cfg.umi_len     = 10;
    cfg.header_text = "singlify test";

    {
        BusWriter w;
        w.open(cfg);
        w.close();
    }

    BusHeader hdr;
    bool ok = read_bus_header(path, hdr);
    CHECK(ok, "read_bus_header returns true");
    CHECK(hdr.magic[0]=='B' && hdr.magic[1]=='U' && hdr.magic[2]=='S' && hdr.magic[3]=='\0',
          "magic == 'BUS\\0'");
    CHECK(hdr.version  == 1,  "version == 1");
    CHECK(hdr.cb_len   == 16, "cb_len == 16");
    CHECK(hdr.umi_len  == 10, "umi_len == 10");
    CHECK(hdr.header_text == "singlify test", "header_text round-trip");
    CHECK(hdr.is_valid(), "is_valid() returns true");

    fs::remove(path);
}

// T6: write multiple records and read them back
static void test_multiple_records() {
    std::cout << "\n=== T6: Multiple records write + read-back ===\n";
    const std::string path = tmp_path("test_bus_recs.bus");

    // Build 5 records with known values
    std::vector<BusRecord> recs;
    for (int i = 0; i < 5; ++i) {
        BusRecord r;
        r.barcode = encode_bus_barcode("ACGT") + i;
        r.umi     = encode_bus_seq(std::string(12, static_cast<char>('A' + i % 4)));
        r.ec      = i * 10;
        r.count   = i + 1;
        r.flags   = 0;
        recs.push_back(r);
    }

    {
        BusWriteConfig cfg;
        cfg.filepath = path;
        cfg.cb_len   = 16;
        cfg.umi_len  = 12;
        BusWriter w;
        bool ok = w.open(cfg);
        CHECK(ok, "open() for multi-record write");
        bool wrote = w.write_records(recs);
        CHECK(wrote, "write_records() returns true");
        CHECK(w.records_written() == 5, "records_written() == 5");
    }

    auto read_back = read_bus_records(path);
    CHECK(read_back.size() == 5, "read back 5 records");

    for (int i = 0; i < 5 && i < (int)read_back.size(); ++i) {
        CHECK(read_back[i].ec    == recs[i].ec,    "ec round-trip record " + std::to_string(i));
        CHECK(read_back[i].count == recs[i].count, "count round-trip record " + std::to_string(i));
        CHECK(read_back[i].flags == 0,             "flags == 0 record " + std::to_string(i));
    }

    fs::remove(path);
}

// T7: file size formula check
static void test_file_size() {
    std::cout << "\n=== T7: File size formula ===\n";
    const std::string path = tmp_path("test_bus_size.bus");

    BusWriteConfig cfg;
    cfg.filepath    = path;
    cfg.cb_len      = 16;
    cfg.umi_len     = 12;
    cfg.header_text = "hdr";  // 3 bytes

    BusRecord r;
    r.barcode = 0; r.umi = 0; r.ec = 0; r.count = 1; r.flags = 0;

    {
        BusWriter w;
        w.open(cfg);
        w.write_record(r);
        w.write_record(r);
        w.write_record(r);  // 3 records
    }

    // Expected: 4+4+4+4+4+3 (header) + 3*32 (records) = 23 + 96 = 119
    auto sz = fs::file_size(path);
    size_t expected = 4 + 4 + 4 + 4 + 4 + 3 + 3 * 32;  // 115... wait: header text = 3 not NUL-terminated
    CHECK(sz == expected, "file size = header(" + std::to_string(4+4+4+4+4+3) +
          ") + 3*32 = " + std::to_string(expected));

    fs::remove(path);
}

// T8: write_record returns false for unopened writer
static void test_write_without_open() {
    std::cout << "\n=== T8: write_record on unopened writer ===\n";
    BusWriter w;
    BusRecord r{};
    bool ok = w.write_record(r);
    CHECK(!ok, "write_record returns false when fp_ is null");
}

// T9: barcode sort order verification
static void test_sorted_output() {
    std::cout << "\n=== T9: Sorted barcode output ===\n";
    const std::string path = tmp_path("test_bus_sorted.bus");

    // Records pre-sorted by barcode value (ascending)
    std::vector<BusRecord> recs;
    for (int i = 4; i >= 0; --i) {
        BusRecord r;
        r.barcode = static_cast<uint64_t>(i) << 56;  // distinct values
        r.umi     = 0;
        r.ec      = i;
        r.count   = 1;
        r.flags   = 0;
        recs.push_back(r);
    }
    // Sort to barcode order before writing (caller's responsibility)
    std::sort(recs.begin(), recs.end(),
        [](const BusRecord& a, const BusRecord& b){ return a.barcode < b.barcode; });

    {
        BusWriteConfig cfg;
        cfg.filepath = path; cfg.cb_len = 16; cfg.umi_len = 12;
        BusWriter w;
        w.open(cfg);
        w.write_records(recs);
    }

    auto rb = read_bus_records(path);
    bool sorted = true;
    for (size_t i = 1; i < rb.size(); ++i)
        if (rb[i].barcode < rb[i-1].barcode) { sorted = false; break; }
    CHECK(sorted, "records read back in barcode-sorted order");
    CHECK(rb.size() == 5, "all 5 sorted records present");

    fs::remove(path);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_record_size();
    test_encode_decode_barcode();
    test_encode_decode_umi();
    test_empty_file();
    test_header_verification();
    test_multiple_records();
    test_file_size();
    test_write_without_open();
    test_sorted_output();

    std::cout << "\n──────────────────────────\n";
    std::cout << "PASSED: " << n_pass << "\n";
    std::cout << "FAILED: " << n_fail << "\n";
    if (n_fail > 0) {
        std::cerr << "[test_bus_writer] " << n_fail << " assertion(s) FAILED\n";
        return 1;
    }
    return 0;
}
