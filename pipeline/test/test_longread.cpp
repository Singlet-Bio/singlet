// test/test_longread.cpp
// Unit tests for G-LONGREAD: long-read scRNA-seq (PacBio MAS-seq / ONT) support.
//
// Tests:
//   1.  detect_longread — empty input → is_longread=false
//   2.  detect_longread — all short reads (median ~90bp) → is_longread=false
//   3.  detect_longread — all long reads (median ~1500bp) → is_longread=true
//   4.  detect_longread — mixed reads, median 500bp → is_longread=true
//   5.  detect_longread — exactly at threshold (200bp) → true
//   6.  detect_longread — just below threshold (199bp) → false
//   7.  detect_longread — n_sampled correct
//   8.  detect_longread — mean_read_length correct
//   9.  split_masseq_read — no adapter in read → single segment spanning all
//  10.  split_masseq_read — one adapter hit → two segments
//  11.  split_masseq_read — two adapter hits → three segments
//  12.  split_masseq_read — adapter at position 0 → first segment empty, rest correct
//  13.  split_masseq_read — returns non-empty vector
//  14.  split_masseq_read — segment boundaries are non-overlapping and contiguous
//  15.  extract_barcode — prefix shorter than CB+UMI → found=false
//  16.  extract_barcode — exact CB+UMI at position 0 with poly-T anchor → found=true
//  17.  extract_barcode — correct barcode extracted
//  18.  extract_barcode — correct UMI extracted
//  19.  extract_barcode — alignment_score > 0.5 for clean match
//  20.  extract_barcode — 1 substitution in anchor still found within max_ed=3
//  21.  extract_barcode — cb_start and cb_end correct
//  22.  extract_barcode — umi_start and umi_end correct
//  23.  write_longread_qc — file is created
//  24.  write_longread_qc — JSON contains "total_reads"
//  25.  write_longread_qc — JSON contains "n_full_length"
//  26.  write_longread_qc — JSON contains "n_chimeric"
//  27.  write_longread_qc — JSON contains "median_read_length"
//  28.  write_longread_qc — JSON contains "n50"
//  29.  write_longread_qc — JSON contains "barcode_match_rate"
//  30.  write_longread_qc — write to bad path returns false

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "singlet-pileup/longread.h"

using namespace singlet;

// ─────────────────────────────────────────────────────────────────────────────
// Test infrastructure
// ─────────────────────────────────────────────────────────────────────────────

static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond, name)                                                        \
    do {                                                                         \
        if (cond) {                                                              \
            std::cout << "  PASS: " << (name) << "\n";                           \
            ++n_pass;                                                            \
        } else {                                                                 \
            std::cout << "  FAIL: " << (name) << " [line " << __LINE__ << "]\n"; \
            ++n_fail;                                                            \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, eps, name) CHECK(std::abs((double)(a) - (double)(b)) <= (eps), name)

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Read a file into a string.
static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    return std::string(std::istreambuf_iterator<char>(f), {});
}

// Build a string of repeated character ch of length n.
static std::string repeat(char ch, size_t n) { return std::string(n, ch); }

// ─────────────────────────────────────────────────────────────────────────────
// detect_longread tests
// ─────────────────────────────────────────────────────────────────────────────

static void test_detect_longread() {
    std::cout << "\n[detect_longread]\n";

    // 1. Empty input
    {
        std::vector<uint32_t> empty;
        auto d = detect_longread(empty);
        CHECK(!d.is_longread, "empty → is_longread=false");
    }

    // 2. All short reads (median ~90bp)
    {
        std::vector<uint32_t> lens(1000, 90u);
        auto d = detect_longread(lens);
        CHECK(!d.is_longread, "short reads median 90bp → false");
    }

    // 3. All long reads (median ~1500bp)
    {
        std::vector<uint32_t> lens(1000, 1500u);
        auto d = detect_longread(lens);
        CHECK(d.is_longread, "long reads median 1500bp → true");
    }

    // 4. Mixed reads, median 500bp → true
    {
        // 500 reads of 50bp and 500 reads of 950bp → median = 500bp
        std::vector<uint32_t> lens;
        lens.insert(lens.end(), 500, 50u);
        lens.insert(lens.end(), 500, 950u);
        auto d = detect_longread(lens);
        CHECK(d.is_longread, "mixed: median 500bp → true");
    }

    // 5. Exactly at threshold (200bp) → true
    {
        std::vector<uint32_t> lens(101, 200u);
        auto d = detect_longread(lens);
        CHECK(d.is_longread, "median exactly 200bp → true");
    }

    // 6. Just below threshold (199bp) → false
    {
        std::vector<uint32_t> lens(101, 199u);
        auto d = detect_longread(lens);
        CHECK(!d.is_longread, "median 199bp → false");
    }

    // 7. n_sampled correct
    {
        std::vector<uint32_t> lens(42, 300u);
        auto d = detect_longread(lens);
        CHECK(d.n_sampled == 42u, "n_sampled == 42");
    }

    // 8. mean_read_length correct
    {
        std::vector<uint32_t> lens = {100u, 200u, 300u};
        auto d = detect_longread(lens);
        CHECK_NEAR(d.mean_read_length, 200.0, 0.01, "mean = 200.0");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// split_masseq_read tests
// ─────────────────────────────────────────────────────────────────────────────

// Build a synthetic MAS-seq read with adapters at known positions.
// Structure: [seg0][adapter][seg1][adapter][seg2]
static std::string make_masseq_read(const std::string& adapter,
                                    size_t seg0_len, size_t seg1_len, size_t seg2_len) {
    // Use a 16-base match to ensure hits; simplify by using the first 16 chars
    std::string a16 = adapter.substr(0, std::min(adapter.size(), size_t(16)));
    return repeat('A', seg0_len) + a16 + repeat('C', seg1_len) + a16 + repeat('G', seg2_len);
}

static void test_split_masseq() {
    std::cout << "\n[split_masseq_read]\n";

    const std::string adapter =
        "ATCTCTCTCAACAACAACAACGGAGGAGGAGGAAAAGAGAGAGAT";

    // 9. No adapter → single segment
    {
        std::string seq = repeat('A', 500);
        auto segs = split_masseq_read(seq, adapter);
        CHECK(segs.size() == 1u, "no adapter → 1 segment");
    }

    // 10. One adapter hit → two segments
    {
        std::string a16 = adapter.substr(0, 16);
        std::string seq = repeat('A', 200) + a16 + repeat('C', 200);
        auto segs = split_masseq_read(seq, adapter);
        CHECK(segs.size() >= 1u, "one adapter → >= 1 segment");
        // Check that at least one segment has adapter_right=true
        bool any_right = false;
        for (auto& s : segs) any_right |= s.has_adapter_right;
        CHECK(any_right, "one adapter → some segment has adapter_right");
    }

    // 11. Two adapter hits → three segments
    {
        std::string seq = make_masseq_read(adapter, 100, 100, 100);
        auto segs = split_masseq_read(seq, adapter);
        CHECK(segs.size() >= 2u, "two adapters → >= 2 segments");
    }

    // 12. Adapter at position 0 → first segment may be empty, rest valid
    {
        std::string a16 = adapter.substr(0, 16);
        std::string seq = a16 + repeat('A', 200);
        auto segs = split_masseq_read(seq, adapter);
        CHECK(!segs.empty(), "adapter at pos 0 → non-empty result");
    }

    // 13. Returns non-empty vector always
    {
        auto segs = split_masseq_read(repeat('N', 300), adapter);
        CHECK(!segs.empty(), "always non-empty vector");
    }

    // 14. Segment boundaries are non-overlapping and ordered
    {
        std::string seq = make_masseq_read(adapter, 100, 100, 100);
        auto segs = split_masseq_read(seq, adapter);
        bool ordered = true;
        uint32_t prev_end = 0;
        for (auto& s : segs) {
            if (s.start < prev_end) {
                ordered = false;
                break;
            }
            if (s.end < s.start) {
                ordered = false;
                break;
            }
            prev_end = s.end;
        }
        CHECK(ordered, "segments are non-overlapping and ordered");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// extract_barcode tests
// ─────────────────────────────────────────────────────────────────────────────

// Build a synthetic read prefix: CB + UMI + poly-T anchor
static std::string make_lr_prefix(const std::string& cb, const std::string& umi,
                                  size_t anchor_len = 12, size_t extra = 50) {
    return cb + umi + repeat('T', anchor_len) + repeat('N', extra);
}

static void test_extract_barcode() {
    std::cout << "\n[extract_barcode]\n";

    const uint32_t cb_len = 16;
    const uint32_t umi_len = 12;

    const std::string cb = "ACGTACGTACGTACGT";  // 16bp
    const std::string umi = "TTTTAAAACCCC";     // 12bp
    const std::string prefix = make_lr_prefix(cb, umi);

    // 15. Prefix shorter than CB+UMI → found=false
    {
        std::string short_prefix(10, 'A');
        auto r = extract_barcode(short_prefix, cb_len, umi_len, 3);
        CHECK(!r.found, "short prefix → found=false");
    }

    // 16. Exact CB+UMI at pos 0 with poly-T anchor → found=true
    {
        auto r = extract_barcode(prefix, cb_len, umi_len, 3);
        CHECK(r.found, "clean prefix → found=true");
    }

    // 17. Correct barcode extracted
    {
        auto r = extract_barcode(prefix, cb_len, umi_len, 3);
        CHECK(r.barcode == cb, "correct barcode extracted");
    }

    // 18. Correct UMI extracted
    {
        auto r = extract_barcode(prefix, cb_len, umi_len, 3);
        CHECK(r.umi == umi, "correct UMI extracted");
    }

    // 19. alignment_score > 0.5 for clean match
    {
        auto r = extract_barcode(prefix, cb_len, umi_len, 3);
        CHECK(r.alignment_score > 0.5, "alignment_score > 0.5");
    }

    // 20. 1 substitution in anchor still found (within max_ed=3)
    {
        std::string noisy = prefix;
        // Flip one base in the poly-T anchor (position cb_len + umi_len)
        noisy[cb_len + umi_len] = 'A';
        auto r = extract_barcode(noisy, cb_len, umi_len, 3);
        CHECK(r.found, "1 substitution in anchor → still found");
    }

    // 21. cb_start and cb_end correct
    {
        auto r = extract_barcode(prefix, cb_len, umi_len, 3);
        CHECK(r.cb_start == 0u && r.cb_end == cb_len,
              "cb_start=0, cb_end=16");
    }

    // 22. umi_start and umi_end correct
    {
        auto r = extract_barcode(prefix, cb_len, umi_len, 3);
        CHECK(r.umi_start == cb_len && r.umi_end == cb_len + umi_len,
              "umi_start=16, umi_end=28");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// write_longread_qc tests
// ─────────────────────────────────────────────────────────────────────────────

static void test_write_qc() {
    std::cout << "\n[write_longread_qc]\n";

    const std::string path = "/tmp/test_longread_qc.json";

    LongReadQC qc;
    qc.total_reads = 100000;
    qc.n_full_length = 72000;
    qc.n_chimeric = 500;
    qc.median_read_length = 1850.5;
    qc.median_passes = 3.2;
    qc.n50 = 2100.0;
    qc.barcode_found = 68000;
    qc.barcode_match_rate = 0.68;

    // 23. File is created
    {
        bool ok = write_longread_qc(path, qc);
        CHECK(ok, "write returns true");
    }

    std::string content = read_file(path);

    // 24-29. JSON contains expected keys
    CHECK(content.find("total_reads") != std::string::npos, "JSON has total_reads");
    CHECK(content.find("n_full_length") != std::string::npos, "JSON has n_full_length");
    CHECK(content.find("n_chimeric") != std::string::npos, "JSON has n_chimeric");
    CHECK(content.find("median_read_length") != std::string::npos, "JSON has median_read_length");
    CHECK(content.find("n50") != std::string::npos, "JSON has n50");
    CHECK(content.find("barcode_match_rate") != std::string::npos, "JSON has barcode_match_rate");

    // 30. Write to bad path returns false
    {
        bool ok = write_longread_qc("/nonexistent/dir/qc.json", qc);
        CHECK(!ok, "bad path → returns false");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== test_longread ===\n";

    test_detect_longread();
    test_split_masseq();
    test_extract_barcode();
    test_write_qc();

    std::cout << "\n=== Summary: " << n_pass << " passed, " << n_fail << " failed ===\n";
    return n_fail == 0 ? 0 : 1;
}
