// SPDX-License-Identifier: MIT
// tests/cpp/test_protocol_tenx_guard.cpp
//
// Regression (v1.4.3, protocol-detect panel 2026-11-23): two 10x libraries were rejected by the pre-gate as non-10x chemistries.
// A 33 nt v2 Read 1 (barcode read 7 nt longer than the kit's) was never a 10x candidate and went to Drop-seq (GSM4953093), and a
// 150 nt v2 Read 1 with the list in about half the reads went to Microwell-seq (GSM8548562). The 10x whitelist guard in
// detect_protocol replaces a non-10x winner when a 10x list holds at least 25 % of barcode reads exactly (40 % before 1.4.7).
//
//  1. 33 nt R1, v2 barcodes in every read: 10x-3p-v2
//  2. 150 nt R1, v2 barcodes in half the reads: a 10x v2 protocol
//  3. 26 nt v2 R1 (control): 10x-3p-v2 as before
//  4. 20 nt R1 of random bases (Drop-seq-like) and 12 nt celseq2 R1: not a 10x protocol
//  5. 33 nt R1 with listed barcodes in 20 % of reads (under the guard rate): the guard does not fire

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
    char tmpl[] = "/tmp/test_tenx_guard_XXXXXX";
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
    // Every other whitelist the protocol table names (arc-gex, visium, ...) gets its own unrelated random list, as in a real
    // installation. A protocol whose list is missing is scored without a whitelist and can win on geometry alone, which is not the
    // situation under test.
    for (const auto& spec : singlet::fq::known_protocols()) {
        if (spec.whitelist_file.empty() || spec.bc_len < 14) continue;
        const std::string path = dir + "/" + spec.whitelist_file;
        if (access(path.c_str(), F_OK) == 0) continue;
        std::ofstream f(path);
        for (int i = 0; i < 5000; ++i) f << rand_bc(spec.bc_len) << "\n";
    }
    return dir;
}

static std::vector<Spot> make_spots(const std::vector<std::string>& wl, int r1_len, int umi_len, bool use_wl, int n = 600, double frac = 1.0) {
    std::mt19937_64 rng(11);
    std::vector<Spot> spots(n);
    for (auto& s : spots) {
        const std::string& bc = wl[rng() % wl.size()];   // one whitelist barcode per spot
        const bool keep = (static_cast<double>(rng() % 1000) / 1000.0) < frac;   // share of spots that carry a listed barcode
        for (int i = 0; i < r1_len; ++i) {
            if (i < 16 && use_wl && keep) s.r1_seq.push_back(singlet::fq::nuc::ascii_to_num(bc[static_cast<size_t>(i)]));
            else s.r1_seq.push_back(static_cast<uint8_t>(rng() % 4));
        }
        (void)umi_len;
        for (int i = 0; i < 90; ++i) s.r2_seq.push_back(static_cast<uint8_t>(rng() % 4));
        s.r1_len = static_cast<uint16_t>(r1_len);
        s.r2_len = 90;
    }
    return spots;
}

static void show(const char* what, const singlet::fq::ProtocolCandidate& r) {
    std::fprintf(stderr, "    [%s] tag=%s wl_rate=%.3f score=%.3f conf=%d\n", what, r.tag.c_str(), r.wl_match_rate, r.score,
                 static_cast<int>(r.confidence));
}

int main() {
    std::vector<std::string> v3, v2;
    const std::string dir = write_whitelists(v3, v2);
    const std::vector<std::string> dirs{dir};
    auto is10x = [](const std::string& t) { return t.rfind("10x-", 0) == 0; };

    std::fprintf(stderr, "[guard] 33-nt v2 R1\n");
    {
        auto spots = make_spots(v2, 33, 10, true);
        auto r = singlet::fq::detect_protocol(spots, 33, 90, dirs); show("33", r);
        CHECK(r.tag == "10x-3p-v2", "33-nt R1 with v2 barcodes is 10x-3p-v2");
        CHECK(r.confidence >= singlet::fq::Confidence::MEDIUM, "with at least medium confidence (the pre-gate passes it)");
    }
    std::fprintf(stderr, "[guard] 150-nt R1, v2 barcodes in half the reads\n");
    {
        auto spots = make_spots(v2, 150, 10, true, 600, 0.5);
        auto r = singlet::fq::detect_protocol(spots, 150, 90, dirs); show("150", r);
        CHECK(r.tag == "10x-3p-v2" || r.tag == "10x-5p-v2", "150-nt R1 with v2 barcodes in half the reads is a 10x v2 protocol");
    }
    std::fprintf(stderr, "[guard] 26-nt v2 R1 (control)\n");
    {
        auto spots = make_spots(v2, 26, 10, true);
        auto r = singlet::fq::detect_protocol(spots, 26, 90, dirs); show("26", r);
        CHECK(r.tag == "10x-3p-v2", "26-nt v2 R1 is still 10x-3p-v2");
    }
    std::fprintf(stderr, "[guard] random 20-nt and 12-nt R1\n");
    {
        auto s20 = make_spots(v3, 20, 8, false);
        auto r20 = singlet::fq::detect_protocol(s20, 20, 90, dirs); show("20", r20);
        CHECK(!is10x(r20.tag), "random 20-nt R1 is not a 10x protocol");
        auto s12 = make_spots(v3, 12, 6, false);
        auto r12 = singlet::fq::detect_protocol(s12, 12, 90, dirs); show("12", r12);
        CHECK(!is10x(r12.tag), "random 12-nt R1 is not a 10x protocol");
    }
    std::fprintf(stderr, "[guard] 33-nt R1, listed barcodes in 20 %% of reads\n");
    {
        auto spots = make_spots(v2, 33, 10, true, 600, 0.2);
        auto r = singlet::fq::detect_protocol(spots, 33, 90, dirs); show("33@20%", r);
        CHECK(!(r.tag == "10x-3p-v2" && r.wl_match_rate < 0.25 && r.wl_match_rate > 0.0 && r.score == r.wl_match_rate),
              "the guard does not fire under its rate");
    }
    std::fprintf(stderr, "[guard] single 33-nt segment with a variable cDNA read (concat mode)\n");
    {
        auto spots = make_spots(v2, 33, 10, true);
        for (size_t i = 0; i < spots.size(); ++i) {   // cDNA read of 50-89 nt: the archive reports no constant R2 length
            spots[i].r2_len = static_cast<uint16_t>(50 + i % 40);
            spots[i].r2_seq.resize(spots[i].r2_len);
        }
        auto r = singlet::fq::detect_protocol(spots, 33, 0, dirs); show("33/0", r);
        CHECK(r.tag == "10x-3p-v2", "33-nt R1 with no constant R2 length is 10x-3p-v2");
    }
    std::fprintf(stderr, "[guard] two 150-nt reads, v2 barcodes in R2 in half the reads\n");
    {
        auto spots = make_spots(v2, 150, 10, true, 600, 0.5);
        std::mt19937_64 rng(5);
        for (auto& s : spots) {   // R1 becomes a 150-nt cDNA read, R2 the barcode read
            std::vector<uint8_t> cdna(150);
            for (auto& b : cdna) b = static_cast<uint8_t>(rng() % 4);
            s.r2_seq = s.r1_seq;
            s.r1_seq = cdna;
            s.r2_len = 150;
        }
        auto r = singlet::fq::detect_protocol(spots, 150, 150, dirs); show("150/150 R2", r);
        CHECK(r.tag == "10x-3p-v2" || r.tag == "10x-5p-v2", "barcodes in R2 of a 150/150 pair give a 10x v2 protocol");
        CHECK(r.reads_swapped, "and set the swap flag so the encoder reads the barcode from R2");
    }
    std::fprintf(stderr, "[guard] two 150-nt reads, v2 barcodes in R2 in 38 %% of reads (SRR30855730)\n");
    {
        auto spots = make_spots(v2, 150, 10, true, 1000, 0.38);
        std::mt19937_64 rng(9);
        for (auto& s : spots) {   // R1 becomes a 150-nt cDNA read, R2 the barcode read
            std::vector<uint8_t> cdna(150);
            for (auto& b : cdna) b = static_cast<uint8_t>(rng() % 4);
            s.r2_seq = s.r1_seq;
            s.r1_seq = cdna;
            s.r2_len = 150;
        }
        auto r = singlet::fq::detect_protocol(spots, 150, 150, dirs); show("150/150 R2 38%", r);
        CHECK(r.tag == "10x-3p-v2" || r.tag == "10x-5p-v2", "v2 barcodes in 38 % of R2 reads give a 10x v2 protocol");
    }
    std::fprintf(stderr, "[guard] two random 150-nt reads\n");
    {
        auto spots = make_spots(v2, 150, 10, false);
        for (auto& s : spots) { s.r2_seq.resize(150); s.r2_len = 150; }
        auto r = singlet::fq::detect_protocol(spots, 150, 150, dirs); show("150/150 random", r);
        CHECK(!is10x(r.tag) || r.wl_match_rate < 0.4, "two random 150-nt reads are not rescued as 10x");
        CHECK(!r.reads_swapped, "and are not swapped");
    }
    std::fprintf(stderr, "[guard] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
