// SPDX-License-Identifier: MIT
// tests/cpp/test_protocol_route.cpp
//
// route_evidence(): a non-10x call runs under its own spec only when the probe reads carry evidence beyond read length.
//
//  1. a Drop-seq-like library (20-nt R1, 12-nt barcodes repeated across reads) is confirmed
//  2. random 20-nt reads of the same length are not (no barcode repeats)
//  3. Drop-seq geometry at the wrong read length is not confirmed even when the barcodes repeat
//  4. CEL-Seq2 and MARS-seq (6-7 nt barcodes, no linker, no list) have no measurable evidence and are never confirmed
//  5. a BD Rhapsody read with its CLS1/CLS2 linker at the fixed offset is confirmed by the linker alone

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

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

static std::vector<uint8_t> num(const std::string& s) {
    std::vector<uint8_t> v;
    for (char c : s) v.push_back(singlet::fq::nuc::ascii_to_num(c));
    return v;
}

// n_bc barcodes at [0, bcl); random elsewhere. n_bc == 0 gives a fresh random barcode on every read.
static std::vector<Spot> make_spots(int r1_len, int bcl, int n_bc, int n = 3000, const std::string& linker = "", int linker_at = 0) {
    std::mt19937_64 rng(11);
    std::vector<std::vector<uint8_t>> bcs(static_cast<size_t>(n_bc > 0 ? n_bc : 1));
    for (auto& b : bcs) for (int i = 0; i < bcl; ++i) b.push_back(static_cast<uint8_t>(rng() % 4));
    const auto lk = num(linker);
    std::vector<Spot> spots(static_cast<size_t>(n));
    for (auto& s : spots) {
        const auto& bc = bcs[rng() % bcs.size()];
        for (int i = 0; i < r1_len; ++i) {
            uint8_t b = static_cast<uint8_t>(rng() % 4);
            if (i < bcl && n_bc > 0) b = bc[static_cast<size_t>(i)];
            if (!lk.empty() && i >= linker_at && i < linker_at + static_cast<int>(lk.size())) b = lk[static_cast<size_t>(i - linker_at)];
            s.r1_seq.push_back(b);
        }
        for (int i = 0; i < 90; ++i) s.r2_seq.push_back(static_cast<uint8_t>(rng() % 4));
        s.r1_len = static_cast<uint16_t>(r1_len);
        s.r2_len = 90;
    }
    return spots;
}

static singlet::fq::ProtocolCandidate called(const char* tag) {
    singlet::fq::ProtocolCandidate c{};
    c.tag = tag;
    c.confidence = singlet::fq::Confidence::MEDIUM;
    c.wl_match_rate = 0.0;
    return c;
}

int main() {
    using singlet::fq::route_evidence;
    const std::vector<std::string> no_dirs;
    std::fprintf(stderr, "[route] evidence\n");
    {
        auto ev = route_evidence(called("dropseq"), make_spots(20, 12, 400), no_dirs);
        CHECK(ev.ok && ev.bc_reuse > 0.9 && ev.exact_len > 0.99, "Drop-seq geometry with repeated 12-nt barcodes is confirmed");
    }
    {
        auto ev = route_evidence(called("dropseq"), make_spots(20, 12, 0), no_dirs);
        CHECK(!ev.ok && ev.bc_reuse < 0.05 && !ev.why.empty(), "random 20-nt reads are not confirmed, and the evidence is named");
    }
    {
        auto ev = route_evidence(called("dropseq"), make_spots(30, 12, 400), no_dirs);
        CHECK(!ev.ok && ev.exact_len < 0.1, "repeated barcodes at 30 nt do not confirm a spec that has neither linker nor list");
    }
    {
        auto ev = route_evidence(called("celseq2"), make_spots(12, 6, 96), no_dirs);
        CHECK(!ev.ok && ev.bc_reuse < 0.0 && ev.linker_rate < 0.0, "CEL-Seq2 (6-nt barcode) has no measurable evidence");
        auto ev2 = route_evidence(called("marsseq2"), make_spots(15, 7, 96), no_dirs);
        CHECK(!ev2.ok, "MARS-seq (7-nt barcode) is not confirmed");
    }
    {
        // BD Rhapsody: CLS1 (9) + ACTGGCCTGCGA at 9, 60-nt read; barcodes also repeat, but the linker alone suffices
        auto ev = route_evidence(called("bd-rhapsody"), make_spots(60, 9, 0, 3000, "ACTGGCCTGCGA", 9), no_dirs);
        CHECK(ev.ok && ev.linker_rate > 0.95 && ev.bc_reuse < 0.1, "BD Rhapsody linker at its offset confirms the call without barcode reuse");
    }
    std::fprintf(stderr, "[route] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
