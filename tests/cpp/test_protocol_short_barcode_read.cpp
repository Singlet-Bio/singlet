// SPDX-License-Identifier: MIT
// tests/cpp/test_protocol_short_barcode_read.cpp
//
// Regression: GEO uploads of 10x v3 libraries often carry a barcode read trimmed to 26-27 bp (16 bp barcode + 10-11 bp UMI). The
// protocol table fixes R1 at 28 bp for v3, and a barcode read shorter than that was never a candidate for v3, so auto-detection fell to
// a no-whitelist protocol (celseq2: 6 bp barcodes) and the run produced garbage counts without a message (PBMC-E pbmc1, R1 = 27 bp).
//
// A whitelist protocol is now also a candidate for a barcode read 1-4 bp short (>= 8 UMI bases left), so the whitelist hit rate decides.
// And protocol_wl_floor_error() makes the encoder stop when an auto-detected whitelist protocol finds almost none of its barcodes.
//
//  1. truncated 27-nt v3 R1: detected as 10x-3p-v3, whitelist rate high, no floor error
//  2. untruncated 28-nt v3 R1 (control): unchanged
//  3. 26-nt v2 R1 (control): still v2
//  4. a real celseq2 R1 (12 nt, random barcodes): not taken for a 10x chemistry
//  5. 27-nt R1 with barcodes that are in no whitelist: no confident whitelist protocol
//  6. the floor: a low-rate auto-detected whitelist protocol is an error; forced / no-whitelist / high-rate ones are not

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include "singlet/fq/protocol.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg)                                                                                  \
    do {                                                                                                  \
        if (cond) { ++g_pass; std::fprintf(stderr, "  PASS: %s\n", msg); }                                 \
        else { ++g_fail; std::fprintf(stderr, "  FAIL: %s\n", msg); }                                     \
    } while (0)

struct Spot {
    std::vector<uint8_t> r1_seq, r2_seq;
    uint16_t r1_len = 0, r2_len = 0;
};

static std::string write_whitelists(std::vector<std::string>& v3, std::vector<std::string>& v2) {
    char tmpl[] = "/tmp/test_short_bc_XXXXXX";
    std::string dir = mkdtemp(tmpl);
    std::mt19937_64 rng(7);
    auto rand_bc = [&](int len) {
        static const char nuc[] = "ACGT";
        std::string s;
        for (int i = 0; i < len; ++i) s += nuc[rng() % 4];
        return s;
    };
    {   // "3M" list (v3) and "737K" list (v2): disjoint random 16-mers
        std::ofstream f3(dir + "/3M-february-2018.txt"), f2(dir + "/737K-august-2016.txt");
        for (int i = 0; i < 20000; ++i) { v3.push_back(rand_bc(16)); f3 << v3.back() << "\n"; }
        for (int i = 0; i < 8000; ++i) { v2.push_back(rand_bc(16)); f2 << v2.back() << "\n"; }
    }
    return dir;
}

static std::vector<Spot> make_spots(const std::vector<std::string>& wl, int r1_len, int umi_len, bool use_wl, int n = 600) {
    std::mt19937_64 rng(11);
    std::vector<Spot> spots(n);
    for (auto& s : spots) {
        for (int i = 0; i < r1_len; ++i) {
            if (i < 16 && use_wl) s.r1_seq.push_back(singlet::fq::nuc::ascii_to_num(wl[rng() % wl.size()][i]));
            else s.r1_seq.push_back(static_cast<uint8_t>(rng() % 4));
        }
        (void)umi_len;
        for (int i = 0; i < 90; ++i) s.r2_seq.push_back(static_cast<uint8_t>(rng() % 4));
        s.r1_len = static_cast<uint16_t>(r1_len);
        s.r2_len = 90;
    }
    return spots;
}

int main() {
    std::vector<std::string> v3, v2;
    const std::string dir = write_whitelists(v3, v2);
    const std::vector<std::string> dirs{dir};

    std::fprintf(stderr, "[short-bc] 27-nt v3 R1\n");
    {
        auto spots = make_spots(v3, 27, 11, true);
        auto r = singlet::fq::detect_protocol(spots, 27, 90, dirs);
        CHECK(r.tag == "10x-3p-v3", "27-nt R1 with v3 barcodes is detected as 10x-3p-v3");
        CHECK(r.wl_match_rate > 0.5, "whitelist hit rate is high");
        CHECK(singlet::fq::protocol_wl_floor_error(r).empty(), "no floor error");
    }
    std::fprintf(stderr, "[short-bc] 28-nt v3 R1 (control)\n");
    {
        auto spots = make_spots(v3, 28, 12, true);
        auto r = singlet::fq::detect_protocol(spots, 28, 90, dirs);
        CHECK(r.tag == "10x-3p-v3", "28-nt R1 is still 10x-3p-v3");
    }
    std::fprintf(stderr, "[short-bc] 26-nt v2 R1 (control)\n");
    {
        auto spots = make_spots(v2, 26, 10, true);
        auto r = singlet::fq::detect_protocol(spots, 26, 90, dirs);
        CHECK(r.tag == "10x-3p-v2", "26-nt R1 with v2 barcodes is still 10x-3p-v2");
    }
    std::fprintf(stderr, "[short-bc] celseq2 R1\n");
    {
        auto spots = make_spots(v3, 12, 6, false);
        auto r = singlet::fq::detect_protocol(spots, 12, 90, dirs);
        CHECK(r.tag.rfind("10x", 0) != 0, "a 12-nt random R1 is not taken for a 10x chemistry");
        CHECK(singlet::fq::protocol_wl_floor_error(r).empty(), "and gives no floor error");
    }
    std::fprintf(stderr, "[short-bc] 27-nt R1, barcodes in no whitelist\n");
    {
        auto spots = make_spots(v3, 27, 11, false);
        auto r = singlet::fq::detect_protocol(spots, 27, 90, dirs);
        const bool confident_wl = r.tag.rfind("10x", 0) == 0 && r.confidence >= singlet::fq::Confidence::MEDIUM;
        CHECK(!confident_wl || !singlet::fq::protocol_wl_floor_error(r).empty(),
              "random barcodes never give a confident 10x protocol without a floor error");
    }
    std::fprintf(stderr, "[short-bc] the whitelist floor\n");
    {
        singlet::fq::ProtocolCandidate c{};
        c.tag = "10x-3p-v3";
        c.confidence = singlet::fq::Confidence::HIGH;
        c.wl_match_rate = 0.001;
        CHECK(!singlet::fq::protocol_wl_floor_error(c).empty(), "0.1% hit rate on an auto-detected 10x protocol is an error");
        c.wl_match_rate = 0.6;
        CHECK(singlet::fq::protocol_wl_floor_error(c).empty(), "60% is fine");
        c.wl_match_rate = 0.001;
        c.confidence = singlet::fq::Confidence::FORCE;
        CHECK(singlet::fq::protocol_wl_floor_error(c).empty(), "a forced protocol is the user's call");
        c.confidence = singlet::fq::Confidence::HIGH;
        c.tag = "celseq2";
        CHECK(singlet::fq::protocol_wl_floor_error(c).empty(), "a protocol without a whitelist has no floor");
    }

    std::string cmd = "rm -rf " + dir;
    if (std::system(cmd.c_str()) != 0) { /* best effort */ }
    std::fprintf(stderr, "short_barcode_read: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
