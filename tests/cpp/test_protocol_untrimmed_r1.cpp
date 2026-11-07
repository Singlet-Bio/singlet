// SPDX-License-Identifier: MIT
// tests/cpp/test_protocol_untrimmed_r1.cpp
//
// Regression: GEO uploads of 10x libraries whose barcode read was deposited UNTRIMMED. GSM7064234 (SRR23617320): 3' v3, spot = [I1 8 nt][R1 151 nt][R2 151 nt],
// R1 = CB16 + UMI12 + 123 further bases, and 93.2% of the first 16 nt are in 3M-february-2018. The decode gate called it splitseq
// (non10x_protocol_low_conf_no_wl_no_10x_geometry) because R1 is 151 nt and not 28.
//
// A barcode read LONGER than the protocol r1_len whose first bc_len bases hit the protocol whitelist (exact match) in >= 50% of the probe spots is
// that protocol, with R1 trimmed to r1_len at decode (the encoder trims when detected.r1_length < observed R1 at MEDIUM+ confidence).
//
//  1. 151-nt R1 with v3 barcodes (R2 151): 10x-3p-v3, confidence >= MEDIUM, r1_length 28, whitelist rate high
//  2. the same with v2 barcodes: 10x-3p-v2, r1_length 26
//  3. 151-nt R1 with 60% / 30% whitelist hits: the threshold is 50%
//  4. 151-nt R1 with barcodes in no whitelist: no confident 10x protocol
//  5. a normal 28-nt v3 R1 and a 91-nt cDNA R2 (control): unchanged
//  6. (only when S14_R1_FASTQ, S14_R2_FASTQ, S14_WL_DIR are set) the real first reads of SRR23617320: 10x-3p-v3

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
    char tmpl[] = "/tmp/test_untrimmed_XXXXXX";
    std::string dir = mkdtemp(tmpl);
    std::mt19937_64 rng(7);
    auto rand_bc = [&](int len) {
        static const char nuc[] = "ACGT";
        std::string s;
        for (int i = 0; i < len; ++i) s += nuc[rng() % 4];
        return s;
    };
    {
        std::ofstream f3(dir + "/3M-february-2018.txt"), f2(dir + "/737K-august-2016.txt");
        for (int i = 0; i < 20000; ++i) { v3.push_back(rand_bc(16)); f3 << v3.back() << "\n"; }
        for (int i = 0; i < 8000; ++i) { v2.push_back(rand_bc(16)); f2 << v2.back() << "\n"; }
    }
    for (const auto& spec : singlet::fq::known_protocols()) {   // every other list the table names: unrelated random barcodes, as in a real install
        if (spec.whitelist_file.empty() || spec.bc_len < 14) continue;
        const std::string path = dir + "/" + spec.whitelist_file;
        if (access(path.c_str(), F_OK) == 0) continue;
        std::ofstream f(path);
        for (int i = 0; i < 5000; ++i) f << rand_bc(spec.bc_len) << "\n";
    }
    return dir;
}

// r1_len-nt R1: a whitelist barcode in the first 16 nt for wl_frac of the spots (random 16-mer otherwise), random bases after; R2 random, r2_len nt
static std::vector<Spot> make_spots(const std::vector<std::string>& wl, int r1_len, int r2_len, double wl_frac, int n = 1000) {
    std::mt19937_64 rng(11);
    std::vector<Spot> spots(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) {
        auto& s = spots[static_cast<size_t>(k)];
        const bool hit = (rng() % 10000) < static_cast<uint64_t>(wl_frac * 10000);
        const std::string& bc = wl[rng() % wl.size()];
        for (int i = 0; i < r1_len; ++i) {
            if (i < 16 && hit) s.r1_seq.push_back(singlet::fq::nuc::ascii_to_num(bc[static_cast<size_t>(i)]));
            else s.r1_seq.push_back(static_cast<uint8_t>(rng() % 4));
        }
        for (int i = 0; i < r2_len; ++i) s.r2_seq.push_back(static_cast<uint8_t>(rng() % 4));
        s.r1_len = static_cast<uint16_t>(r1_len);
        s.r2_len = static_cast<uint16_t>(r2_len);
    }
    return spots;
}

static void show(const char* what, const singlet::fq::ProtocolCandidate& r) {
    std::fprintf(stderr, "    [%s] tag=%s r1_length=%u wl_rate=%.3f score=%.3f conf=%d\n", what, r.tag.c_str(), r.r1_length, r.wl_match_rate,
                 r.score, static_cast<int>(r.confidence));
}

static bool confident_tenx(const singlet::fq::ProtocolCandidate& r) {
    return r.tag.rfind("10x", 0) == 0 && r.confidence >= singlet::fq::Confidence::MEDIUM;
}

static std::vector<Spot> read_fastq_pair(const std::string& f1, const std::string& f2, size_t n) {
    std::ifstream a(f1), b(f2);
    std::vector<Spot> spots;
    std::string l, seq1, seq2;
    auto conv = [](const std::string& s, std::vector<uint8_t>& v) {
        for (char c : s) v.push_back(singlet::fq::nuc::ascii_to_num(c));
    };
    while (spots.size() < n && std::getline(a, l)) {
        Spot s;
        if (!std::getline(a, seq1)) break;
        std::getline(a, l); std::getline(a, l);
        std::getline(b, l);
        if (!std::getline(b, seq2)) break;
        std::getline(b, l); std::getline(b, l);
        conv(seq1, s.r1_seq); conv(seq2, s.r2_seq);
        s.r1_len = static_cast<uint16_t>(seq1.size());
        s.r2_len = static_cast<uint16_t>(seq2.size());
        spots.push_back(std::move(s));
    }
    return spots;
}

int main() {
    std::vector<std::string> v3, v2;
    const std::string dir = write_whitelists(v3, v2);
    const std::vector<std::string> dirs{dir};

    std::fprintf(stderr, "[untrimmed] 151-nt v3 R1, 151-nt R2\n");
    {
        auto spots = make_spots(v3, 151, 151, 0.93);
        auto r = singlet::fq::detect_protocol(spots, 151, 151, dirs); show("v3 151", r);
        CHECK(r.tag == "10x-3p-v3", "untrimmed v3 R1 is 10x-3p-v3");
        CHECK(r.confidence >= singlet::fq::Confidence::MEDIUM, "at MEDIUM or better");
        CHECK(r.r1_length == 28, "r1_length 28: the encoder trims R1 to the protocol length");
        CHECK(r.wl_match_rate > 0.8, "whitelist hit rate is high");
    }
    std::fprintf(stderr, "[untrimmed] 151-nt v2 R1\n");
    {
        auto spots = make_spots(v2, 151, 151, 0.9);
        auto r = singlet::fq::detect_protocol(spots, 151, 151, dirs); show("v2 151", r);
        CHECK(r.tag == "10x-3p-v2", "untrimmed v2 R1 is 10x-3p-v2");
        CHECK(r.confidence >= singlet::fq::Confidence::MEDIUM && r.r1_length == 26, "MEDIUM+, r1_length 26");
    }
    std::fprintf(stderr, "[untrimmed] 50 percent threshold\n");
    {
        auto hi = make_spots(v3, 151, 151, 0.60);
        auto r = singlet::fq::detect_protocol(hi, 151, 151, dirs); show("60%", r);
        CHECK(r.tag == "10x-3p-v3" && r.confidence >= singlet::fq::Confidence::MEDIUM, "60% whitelist hits: 10x-3p-v3");
        auto lo = make_spots(v3, 151, 151, 0.30);
        auto q = singlet::fq::detect_protocol(lo, 151, 151, dirs); show("30%", q);
        CHECK(!(q.tag == "10x-3p-v3" && q.confidence >= singlet::fq::Confidence::HIGH), "30% is not promoted to HIGH by the untrimmed-R1 rule");
    }
    std::fprintf(stderr, "[untrimmed] 151-nt R1, barcodes in no whitelist\n");
    {
        auto spots = make_spots(v3, 151, 151, 0.0);
        auto r = singlet::fq::detect_protocol(spots, 151, 151, dirs); show("random", r);
        CHECK(!confident_tenx(r), "random barcodes never give a confident 10x protocol");
    }
    std::fprintf(stderr, "[untrimmed] 28-nt v3 R1 (control)\n");
    {
        auto spots = make_spots(v3, 28, 91, 0.95);
        auto r = singlet::fq::detect_protocol(spots, 28, 91, dirs); show("28", r);
        CHECK(r.tag == "10x-3p-v3" && r.r1_length == 28, "unchanged");
    }
    if (const char* f1 = std::getenv("S14_R1_FASTQ")) {
        const char* f2 = std::getenv("S14_R2_FASTQ");
        const char* wl = std::getenv("S14_WL_DIR");
        std::fprintf(stderr, "[untrimmed] real reads %s\n", f1);
        if (f2 && wl) {
            auto spots = read_fastq_pair(f1, f2, 5000);
            const std::vector<std::string> real{wl};
            CHECK(!spots.empty(), "read the probe spots");
            if (!spots.empty()) {
                auto r = singlet::fq::detect_protocol(spots, spots[0].r1_len, spots[0].r2_len, real); show("real", r);
                CHECK(r.tag == "10x-3p-v3" && r.confidence >= singlet::fq::Confidence::MEDIUM && r.r1_length == 28,
                      "real SRR23617320 reads: 10x-3p-v3, MEDIUM+, r1_length 28");
            }
        } else {
            CHECK(false, "S14_R1_FASTQ needs S14_R2_FASTQ and S14_WL_DIR");
        }
    }
    std::fprintf(stderr, "\nuntrimmed-r1: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
