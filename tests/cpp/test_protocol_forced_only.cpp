// SPDX-License-Identifier: MIT
// tests/cpp/test_protocol_forced_only.cpp
//
// Forced-only protocol specs (CandidateSpec::forced_only) are reachable by name through --protocol and are never auto-detection
// candidates, so adding one cannot change the call on a library that already processes. Covers the specs added for the benchmark
// corpus input routes: celseq2-u6b8 (Tian 2019 and Mereu 2020 CEL-Seq2), scrbseq and icell8 (Mereu 2020).
//
//  1. each tag (and its aliases) resolves, with the measured geometry and a 3' assay type
//  2. ids are unique across the table
//  3. reads of exactly the forced-only R1 length never auto-detect as a forced-only protocol
//  4. a 28-nt R1 (10x v3 length, +3 over icell8) is unaffected

#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
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

// Barcode read: a small set of barcodes at [bc0, bc0+bcl), random elsewhere (UMI); random 90-nt cDNA.
static std::vector<Spot> make_spots(int r1_len, int bc0, int bcl, int n_bc, int n = 600) {
    std::mt19937_64 rng(5);
    std::vector<std::vector<uint8_t>> bcs(static_cast<size_t>(n_bc));
    for (auto& b : bcs) for (int i = 0; i < bcl; ++i) b.push_back(static_cast<uint8_t>(rng() % 4));
    std::vector<Spot> spots(static_cast<size_t>(n));
    for (auto& s : spots) {
        const auto& bc = bcs[rng() % bcs.size()];
        for (int i = 0; i < r1_len; ++i)
            s.r1_seq.push_back(i >= bc0 && i < bc0 + bcl ? bc[static_cast<size_t>(i - bc0)] : static_cast<uint8_t>(rng() % 4));
        for (int i = 0; i < 90; ++i) s.r2_seq.push_back(static_cast<uint8_t>(rng() % 4));
        s.r1_len = static_cast<uint16_t>(r1_len);
        s.r2_len = 90;
    }
    return spots;
}

int main() {
    using singlet::fq::find_protocol_spec;
    struct Want { const char* tag; std::vector<const char*> aliases; int id, r1, bc0, bcl, umi0, umil; };
    const std::vector<Want> wants = {
        {"celseq2-u6b8", {"CELSeq2-U6B8", "celseq2_tian", "celseq2-mereu"}, 26, 14, 6, 8, 0, 6},
        {"scrbseq", {"SCRB-seq", "scrb"}, 27, 16, 0, 6, 6, 10},
        {"icell8", {"ICELL8", "icell8-3p"}, 28, 25, 0, 11, 11, 14},
    };
    std::fprintf(stderr, "[forced-only] lookup and geometry\n");
    for (const auto& w : wants) {
        const auto* s = find_protocol_spec(w.tag);
        CHECK(s != nullptr, w.tag);
        if (!s) continue;
        CHECK(s->forced_only, "is forced-only");
        CHECK(s->protocol_id == w.id && s->r1_len == w.r1 && s->bc_offset == w.bc0 && s->bc_len == w.bcl &&
                  s->umi_offset == w.umi0 && s->umi_len == w.umil, "geometry as measured");
        CHECK(s->whitelist_file.empty(), "no whitelist");
        CHECK(singlet::fq::is_3prime_protocol(s->tag), "3' protocol");
        CHECK(singlet::fq::protocol_tag_to_assay_type(s->tag) == singlet::fq::AssayType::SC_RNA_3PRIME, "3' assay type");
        for (const char* a : w.aliases) CHECK(find_protocol_spec(a) == s, a);
    }
    std::fprintf(stderr, "[forced-only] ids unique\n");
    {
        std::set<int> ids; bool dup = false;
        for (const auto& k : singlet::fq::known_protocols()) dup |= !ids.insert(k.protocol_id).second;
        CHECK(!dup, "protocol ids are unique");
    }
    std::fprintf(stderr, "[forced-only] never auto-detected\n");
    const std::vector<std::string> no_dirs;
    for (const auto& w : wants) {
        auto spots = make_spots(w.r1, w.bc0, w.bcl, 96);
        auto r = singlet::fq::detect_protocol(spots, static_cast<uint16_t>(w.r1), 90, no_dirs);
        std::fprintf(stderr, "    R1 %d -> %s\n", w.r1, r.tag.c_str());
        const auto* got = find_protocol_spec(r.tag);
        CHECK(!got || !got->forced_only, "auto-detection does not return a forced-only spec");
    }
    {
        auto spots = make_spots(28, 0, 16, 500);
        auto r = singlet::fq::detect_protocol(spots, 28, 90, no_dirs);
        std::fprintf(stderr, "    R1 28 -> %s\n", r.tag.c_str());
        CHECK(r.tag != "icell8", "a 28-nt R1 is not taken for icell8");
    }
    std::fprintf(stderr, "[forced-only] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
