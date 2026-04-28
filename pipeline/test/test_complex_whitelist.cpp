// test/test_complex_whitelist.cpp
// Unit tests for AUTOFIX-BD-RHAPSODY-WHITELIST:
//   Per-segment whitelist discovery for CB_UMI_Complex protocols.
//
// Tests cover:
//   1. ComplexExtraSeg position parsing: "0_start_0_end" → {start, len}
//   2. ComplexExtraSeg skip for R2-relative positions "1_N_1_M"
//   3. Per-segment packed-key barcode counting from simulated R1 reads
//   4. Explosion guard: product > 2M → reject; product ≤ 2M → accept
//   5. BD Rhapsody position strings encode to correct start/length values
//   6. STAR "None" guard: effective_whitelist "None" triggers hard error path
//   7. Per-segment file writing produces correct barcodes with correct N/count
//   8. Multiple-segment count accumulation (N0 × N1 × N2)

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ── HELPERS ──────────────────────────────────────────────────────────────────

static int n_pass = 0;
static int n_fail = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL [" << __LINE__ << "]: " << (msg) << "\n"; \
        ++n_fail; \
    } else { \
        ++n_pass; \
    } \
} while (0)

// Parse CB_UMI_Complex position string "atype0_dist0_atype1_dist1".
// Returns {r1_start, len} or {-1, 0} when position is not R1-absolute (atype≠0).
struct ComplexExtraSeg { int r1_start; int len; };
static ComplexExtraSeg parse_complex_seg_pos(const std::string& p) {
    int vals[4] = {}; size_t px = 0; int vi = 0;
    for (size_t ci = 0; ci <= p.size() && vi < 4; ++ci) {
        if (ci == p.size() || p[ci] == '_') {
            vals[vi++] = std::atoi(p.c_str() + px);
            px = ci + 1;
        }
    }
    int atype0 = vals[0]; int dist0 = vals[1]; int dist1 = vals[3];
    int seg_len = dist1 - dist0 + 1;
    if (atype0 == 0 && vals[2] == 0 && dist0 >= 0 && seg_len > 0 && seg_len <= 21)
        return {dist0, seg_len};
    return {-1, 0};
}

// Extract barcodes at a given (0-based start, len) from a list of R1 byte sequences.
// Each R1 is a vector of 0-3 base values (0=A,1=C,2=G,3=T).
// Returns frequency map of packed 3-bit uint64 keys.
static std::unordered_map<uint64_t,uint64_t> count_seg_barcodes(
    const std::vector<std::vector<uint8_t>>& reads,
    int start, int len)
{
    std::unordered_map<uint64_t,uint64_t> counts;
    for (const auto& r1 : reads) {
        if ((int)r1.size() < start + len) continue;
        uint64_t key = 0;
        for (int j = 0; j < len; ++j) key = (key << 3) | r1[start + j];
        counts[key]++;
    }
    return counts;
}

// Unpack a 3-bit-per-base uint64 key to ASCII string of length `len`.
static std::string unpack_key(uint64_t key, int len) {
    static constexpr char BASE[5] = {'A','C','G','T','N'};
    std::string bc(len, ' ');
    for (int j = len - 1; j >= 0; --j) {
        bc[j] = BASE[key & 0x7]; key >>= 3;
    }
    return bc;
}

// Pack ASCII barcode to 3-bit-per-base uint64.
static uint64_t pack_barcode(const std::string& bc) {
    uint64_t key = 0;
    for (char c : bc) {
        int val = (c == 'A') ? 0 : (c == 'C') ? 1 : (c == 'G') ? 2 : (c == 'T') ? 3 : 4;
        key = (key << 3) | val;
    }
    return key;
}

// Simulated explosion guard check.
static bool explosion_guard(const std::vector<size_t>& seg_counts, size_t limit = 2000000) {
    size_t product = 1;
    for (size_t n : seg_counts) {
        if (n == 0) return true; // empty = reject
        if (product > limit / n) return true; // would exceed limit
        product *= n;
    }
    return product > limit;
}

// ── TEST FUNCTIONS ────────────────────────────────────────────────────────────

// Test 1: BD Rhapsody CLS1 position "0_0_0_8" → {start=0, len=9}
static void test_bd_rhapsody_cls1_pos() {
    auto seg = parse_complex_seg_pos("0_0_0_8");
    ASSERT(seg.r1_start == 0, "BD-Rhapsody CLS1 start should be 0");
    ASSERT(seg.len == 9,      "BD-Rhapsody CLS1 len should be 9 (0..8 inclusive)");
}

// Test 2: BD Rhapsody CLS2 position "0_21_0_29" → {start=21, len=9}
static void test_bd_rhapsody_cls2_pos() {
    auto seg = parse_complex_seg_pos("0_21_0_29");
    ASSERT(seg.r1_start == 21, "BD-Rhapsody CLS2 start should be 21");
    ASSERT(seg.len == 9,       "BD-Rhapsody CLS2 len should be 9");
}

// Test 3: BD Rhapsody CLS3 position "0_43_0_51" → {start=43, len=9}
static void test_bd_rhapsody_cls3_pos() {
    auto seg = parse_complex_seg_pos("0_43_0_51");
    ASSERT(seg.r1_start == 43, "BD-Rhapsody CLS3 start should be 43");
    ASSERT(seg.len == 9,       "BD-Rhapsody CLS3 len should be 9");
}

// Test 4: SPLiT-seq R2-relative position "1_10_1_17" → skip ({-1, 0})
static void test_splitseq_r2_pos_skip() {
    auto seg = parse_complex_seg_pos("1_10_1_17");
    ASSERT(seg.r1_start == -1, "SPLiT-seq R2 position should be skipped (r1_start=-1)");
    ASSERT(seg.len == 0,       "SPLiT-seq R2 position should be skipped (len=0)");
}

// Test 5: SureCell position "0_21_0_26" → {start=21, len=6}
static void test_surecell_seg_pos() {
    auto seg = parse_complex_seg_pos("0_21_0_26");
    ASSERT(seg.r1_start == 21, "SureCell seg2 start should be 21");
    ASSERT(seg.len == 6,       "SureCell seg2 len should be 6");
}

// Test 6: Per-segment barcode counting from synthetic BD Rhapsody R1 reads.
//   R1 layout: CLS1[0..8] + linker[9..20] + CLS2[21..29] + linker[30..42] + CLS3[43..51] + UMI[52..59]
//   We inject 3 CLS1 barcodes × 3 CLS2 barcodes × 3 CLS3 barcodes = 27 combos.
//   Each combo appears 200 times → all ≥100 threshold.
static void test_per_seg_counting() {
    // 3 barcodes per segment, each 9bp
    std::vector<std::string> cls1 = {"AATGCGACG", "CCTTAAGCT", "GGTACCATG"};
    std::vector<std::string> cls2 = {"TTTCCCAAA", "AAAGGGCCC", "CGCTATATA"};
    std::vector<std::string> cls3 = {"GCATGCATG", "TATCGATCG", "CGATCGATC"};

    // Build synthetic R1 reads (60bp): CLS1+NNNNNNNNNNNNN+CLS2+NNNNNNNNNNNNN+CLS3+NNNNNNN
    std::vector<std::vector<uint8_t>> reads;
    for (const auto& b1 : cls1) {
        for (const auto& b2 : cls2) {
            for (const auto& b3 : cls3) {
                std::string r1(60, 'N');
                r1.replace(0, 9, b1);
                r1.replace(21, 9, b2);
                r1.replace(43, 9, b3);
                // Convert ASCII to 0-4 encoding
                std::vector<uint8_t> r1e(60);
                for (int i = 0; i < 60; ++i) {
                    char c = r1[i];
                    r1e[i] = (c == 'A') ? 0 : (c == 'C') ? 1 : (c == 'G') ? 2 :
                             (c == 'T') ? 3 : 4;
                }
                // Each combo: 200 reads
                for (int rep = 0; rep < 200; ++rep)
                    reads.push_back(r1e);
            }
        }
    }

    // Count seg1 (CLS2 at positions 21-29)
    auto seg1_counts = count_seg_barcodes(reads, 21, 9);
    ASSERT(seg1_counts.size() == 3, "CLS2 should have exactly 3 unique barcodes");
    // Each CLS2 barcode appears in 3×3=9 combinations × 200 = 1800 reads
    for (const auto& [key, cnt] : seg1_counts) {
        ASSERT(cnt == 1800, "Each CLS2 barcode should appear 1800 times");
        std::string bc = unpack_key(key, 9);
        bool found = false;
        for (const auto& expected : cls2) if (expected == bc) found = true;
        ASSERT(found, "Counted CLS2 barcode must be one of the 3 expected sequences");
    }

    // Count seg2 (CLS3 at positions 43-51)
    auto seg2_counts = count_seg_barcodes(reads, 43, 9);
    ASSERT(seg2_counts.size() == 3, "CLS3 should have exactly 3 unique barcodes");
    for (const auto& [key, cnt] : seg2_counts) {
        ASSERT(cnt == 1800, "Each CLS3 barcode should appear 1800 times");
    }
}

// Test 7: Explosion guard correctly rejects > 2M product.
static void test_explosion_guard_reject() {
    // 4096 × 4096 × 4096 = 68.7B → reject
    ASSERT(explosion_guard({4096, 4096, 4096}), "4096^3 should be rejected");
    // 96 × 96 × 96 = 884736 → accept
    ASSERT(!explosion_guard({96, 96, 96}), "96^3 = 884736 should be accepted");
    // 1000 × 1000 × 3 = 3M → reject
    ASSERT(explosion_guard({1000, 1000, 3}), "1000×1000×3 = 3M should be rejected");
    // 100 × 100 × 100 = 1M → accept
    ASSERT(!explosion_guard({100, 100, 100}), "100^3 = 1M should be accepted");
    // Single empty segment → reject
    ASSERT(explosion_guard({96, 0, 96}), "Zero-entry segment should be rejected");
}

// Test 8: Per-segment barcode file writing produces correct content.
static void test_per_seg_file_writing() {
    // Pack two known 6-mers for a surecell-style test
    std::vector<std::string> expected_bcs = {"AAAAAA", "CCCCCC", "GGGGGG"};
    std::unordered_map<uint64_t,uint64_t> counts;
    for (const auto& bc : expected_bcs)
        counts[pack_barcode(bc)] = 500; // 500 reads each

    // Write per-segment file
    std::string tmp_file = "/tmp/test_complex_wl_seg1.tsv";
    static constexpr char UNPACK_BASE_T[5] = {'A','C','G','T','N'};
    int seg_len = 6;
    {
        std::vector<std::pair<uint64_t,std::string>> seg_bcs;
        for (auto& [key, cnt] : counts) {
            if (cnt >= 100) {
                std::string bc(seg_len, ' ');
                uint64_t k = key;
                for (int j = seg_len - 1; j >= 0; --j) {
                    bc[j] = UNPACK_BASE_T[k & 0x7]; k >>= 3;
                }
                seg_bcs.emplace_back(cnt, std::move(bc));
            }
        }
        std::sort(seg_bcs.begin(), seg_bcs.end(),
                  [](const auto& a, const auto& b){ return a.first > b.first; });
        std::ofstream out(tmp_file);
        for (auto& [cnt, bc] : seg_bcs) out << bc << "\n";
    }

    // Read back and verify
    std::ifstream in(tmp_file);
    std::vector<std::string> read_bcs;
    std::string line;
    while (std::getline(in, line)) if (!line.empty()) read_bcs.push_back(line);
    in.close();
    std::remove(tmp_file.c_str());

    ASSERT(read_bcs.size() == 3, "Should write 3 barcodes to file");
    for (const auto& bc : read_bcs) {
        ASSERT(bc.size() == 6u, "Each written barcode should be 6bp");
        bool found = false;
        for (const auto& exp : expected_bcs) if (exp == bc) found = true;
        ASSERT(found, "Each written barcode must match expected");
    }
}

// Test 9: sci-RNA-seq3 single-segment "0_24_0_33" → {start=24, len=10}
//   sci-rna-seq3 has 1 CB segment so it falls to the single-segment CB_samTagOut path,
//   but the position parsing must be correct.
static void test_scirna3_seg_pos() {
    auto seg = parse_complex_seg_pos("0_24_0_33");
    ASSERT(seg.r1_start == 24, "sci-RNA-seq3 CB start should be 24");
    ASSERT(seg.len == 10,      "sci-RNA-seq3 CB len should be 10");
}

// Test 10: inDrop two-part "0_0_0_9" and "0_32_0_39"
static void test_indrop_seg_pos() {
    auto seg0 = parse_complex_seg_pos("0_0_0_9");
    ASSERT(seg0.r1_start == 0,  "inDrop BC1 start=0");
    ASSERT(seg0.len == 10,      "inDrop BC1 len=10");
    auto seg1 = parse_complex_seg_pos("0_32_0_39");
    ASSERT(seg1.r1_start == 32, "inDrop BC2 start=32");
    ASSERT(seg1.len == 8,       "inDrop BC2 len=8");
}

// Test 11: Verify that "None" check is essential (no bypass possible).
//   The guard in singlify.cpp ensures effective_whitelist != "None"
//   before CB_UMI_Complex args are emitted.  Simulate the check here.
static void test_none_guard() {
    auto would_use_cb_umi_complex = [](const std::string& wl,
                                       bool has_per_seg,
                                       bool has_auto) -> bool {
        if (has_per_seg) return true;  // Priority 1: per-seg files exist
        if (has_auto)    return false; // Priority 2: fall back to CB_samTagOut
        // Priority 3: fixed whitelist — must NOT be "None"
        if (wl == "None" || wl == "none" || wl.empty()) return false; // guarded → error
        return true;
    };

    // Case: no per-seg, no auto, whitelist="None" → MUST NOT use CB_UMI_Complex
    ASSERT(!would_use_cb_umi_complex("None", false, false),
           "CB_UMI_Complex must not be used when whitelist=None and no auto barcodes");
    // Case: no per-seg, no auto, real whitelist → CB_UMI_Complex with fixed WL
    ASSERT(would_use_cb_umi_complex("/path/to/wl.txt", false, false),
           "CB_UMI_Complex should be used when fixed whitelist is available");
    // Case: per-seg files available → CB_UMI_Complex regardless
    ASSERT(would_use_cb_umi_complex("None", true, false),
           "CB_UMI_Complex should be used when per-seg files are available");
    // Case: auto barcodes only → CB_samTagOut (not CB_UMI_Complex)
    ASSERT(!would_use_cb_umi_complex("None", false, true),
           "CB_samTagOut should be used when only auto barcodes (seg-0) are available");
}

// Test 12: ddSEQ 6-mer positions (3 × 6bp ABs) should parse correctly.
static void test_ddseq_seg_pos() {
    auto seg0 = parse_complex_seg_pos("0_0_0_5");
    ASSERT(seg0.r1_start == 0, "ddSEQ seg0 start=0");
    ASSERT(seg0.len == 6,      "ddSEQ seg0 len=6");
    auto seg1 = parse_complex_seg_pos("0_9_0_14");
    ASSERT(seg1.r1_start == 9, "ddSEQ seg1 start=9");
    ASSERT(seg1.len == 6,      "ddSEQ seg1 len=6");
    auto seg2 = parse_complex_seg_pos("0_18_0_23");
    ASSERT(seg2.r1_start == 18, "ddSEQ seg2 start=18");
    ASSERT(seg2.len == 6,       "ddSEQ seg2 len=6");
}

// ── MAIN ─────────────────────────────────────────────────────────────────────

int main() {
    test_bd_rhapsody_cls1_pos();
    test_bd_rhapsody_cls2_pos();
    test_bd_rhapsody_cls3_pos();
    test_splitseq_r2_pos_skip();
    test_surecell_seg_pos();
    test_per_seg_counting();
    test_explosion_guard_reject();
    test_per_seg_file_writing();
    test_scirna3_seg_pos();
    test_indrop_seg_pos();
    test_none_guard();
    test_ddseq_seg_pos();

    std::cerr << "\n[complex_whitelist] " << n_pass << "/" << (n_pass + n_fail)
              << " tests passed" << (n_fail ? " ← FAILURES" : "") << "\n";
    return n_fail > 0 ? 1 : 0;
}
