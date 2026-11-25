// SPDX-License-Identifier: MIT
// tests/cpp/test_parse_bc1.cpp
//
// Parse WT v2 round-1 merge (v1.4.8): a random-hexamer round-1 barcode at R1 78-85 is rewritten to the oligo-dT barcode of the
// same well; oligo-dT barcodes, unknown 8-mers, barcodes with N and reads shorter than 86 nt are left alone.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "singlet/fq/parse_bc1.h"

static int g_pass = 0, g_fail = 0;
static void check(bool c, const char* m) { if (c) { ++g_pass; std::fprintf(stderr, "  PASS: %s\n", m); } else { ++g_fail; std::fprintf(stderr, "  FAIL: %s\n", m); } }

static std::vector<uint8_t> read_with_bc1(const std::string& bc1, size_t len = 150) {
    std::vector<uint8_t> r(len, 0);
    for (size_t i = 0; i < bc1.size() && 78 + i < len; ++i) {
        const char c = bc1[i];
        r[78 + i] = c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4;
    }
    return r;
}
static std::string bc1_of(const std::vector<uint8_t>& r) {
    std::string s;
    for (int i = 0; i < 8; ++i) s += "ACGTN"[r[78 + i]];
    return s;
}

int main() {
    check(singlet::fq::parse_v2_bc1_random_to_dt().size() == 96, "96 wells in the table");
    auto a1 = read_with_bc1("CATCATCC");   // well A1 random hexamer
    check(singlet::fq::merge_parse_bc1(a1.data(), 150) && bc1_of(a1) == "CATTCCTA", "A1 random -> A1 oligo-dT");
    auto h12 = read_with_bc1("TTTGGTCA");
    const bool h12_mapped = singlet::fq::merge_parse_bc1(h12.data(), 150);
    std::fprintf(stderr, "    TTTGGTCA -> %s (%s)\n", bc1_of(h12).c_str(), h12_mapped ? "rewritten" : "kept");
    auto dt = read_with_bc1("CATTCCTA");   // oligo-dT stays
    check(!singlet::fq::merge_parse_bc1(dt.data(), 150) && bc1_of(dt) == "CATTCCTA", "oligo-dT barcode unchanged");
    auto unk = read_with_bc1("AAAAAAAA");
    check(!singlet::fq::merge_parse_bc1(unk.data(), 150), "unknown 8-mer unchanged");
    auto n = read_with_bc1("CATCATCN");
    check(!singlet::fq::merge_parse_bc1(n.data(), 150), "barcode with N unchanged");
    auto shortr = read_with_bc1("CATCATCC", 85);
    check(!singlet::fq::merge_parse_bc1(shortr.data(), 85), "85 nt read unchanged");
    std::fprintf(stderr, "[parse_bc1] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
