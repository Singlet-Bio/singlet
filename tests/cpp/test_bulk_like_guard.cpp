// SPDX-License-Identifier: MIT
// tests/cpp/test_bulk_like_guard.cpp
//
// D-150 bulk-like guard for non-10x droplet libraries (summary_json.h): the barcode-UMI curve of a droplet library has a knee, a
// bulk library parsed as Drop-seq does not. Logic test on hand-built rank curves; real-data validation is in bench/non10x.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "singlet/pileup/summary_json.h"

static int g_fail = 0;
#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", msg); } \
        else std::fprintf(stderr, "PASS: %s\n", msg);                     \
    } while (0)

static bool has(const std::vector<std::string>& v, const char* k) {
    for (const auto& s : v) if (s == k) return true;
    return false;
}

int main() {
    using namespace singlet;
    // Droplet-like: 500 cells of ~3000 UMIs over a background of 20000 barcodes with 1-5 UMIs.
    std::vector<uint64_t> d;
    std::vector<uint32_t> cells;
    for (uint32_t i = 0; i < 500; ++i) { d.push_back(2000 + (i * 7) % 2000); cells.push_back(i); }
    for (uint32_t i = 0; i < 20000; ++i) d.push_back(1 + i % 5);
    CHECK(bulk_like_reasons(d, cells).empty(), "droplet curve passes");

    // Bulk-like: every read its own barcode, flat curve; the "cells" are the top 400.
    std::vector<uint64_t> b(50000, 1);
    for (size_t i = 0; i < 400; ++i) b[i] = 2;
    std::vector<uint32_t> bc;
    for (uint32_t i = 0; i < 400; ++i) bc.push_back(i);
    const auto rb = bulk_like_reasons(b, bc);
    CHECK(has(rb, "no_knee"), "flat curve has no knee");
    CHECK(has(rb, "barcode_saturated"), "one barcode per UMI is saturated");

    // One barcode holds 20% of all UMIs (low-complexity reads).
    std::vector<uint64_t> p = d;
    uint64_t tot = 0;
    for (auto v : p) tot += v;
    p[600] = tot / 3;
    CHECK(has(bulk_like_reasons(p, cells), "dominant_barcode"), "dominant barcode flagged");

    CHECK(is_tenx_protocol_name("10xv3") && is_tenx_protocol_name("cite-seq-gex") && is_tenx_protocol_name("unknown"), "10x names skip the guard");
    CHECK(!is_tenx_protocol_name("dropseq") && !is_tenx_protocol_name("indrop") && !is_tenx_protocol_name("bd-rhapsody"), "non-10x names get it");
    CHECK(bulk_like_reasons({}, {}).empty(), "empty input is not flagged");
    return g_fail ? 1 : 0;
}
