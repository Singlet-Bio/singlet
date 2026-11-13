// SPDX-License-Identifier: MIT
// inDrop W1-anchored barcode extraction (include/singlet/fq/indrop_anchor.h) and 3' adapter trimming (adapter_trim.h) on synthetic reads
// built from the shipped barcode lists.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "singlet/fq/adapter_trim.h"
#include "singlet/fq/indrop_anchor.h"

#ifndef WHITELIST_DIR
#define WHITELIST_DIR "../python/singlet/data/whitelists"
#endif

static int g_pass = 0, g_fail = 0;
#define CHECK(name, cond)                                                                  \
    do {                                                                                   \
        if (cond) ++g_pass;                                                                \
        else { ++g_fail; std::cerr << "FAIL: " << name << " at line " << __LINE__ << "\n"; } \
    } while (0)

using singlet::fq::AdapterSet;
using singlet::fq::IndropAnchor;

static std::vector<std::string> read_list(const std::string& p) {
    std::vector<std::string> v;
    std::ifstream f(p);
    std::string l;
    while (std::getline(f, l)) {
        while (!l.empty() && (l.back() == '\r' || l.back() == ' ')) l.pop_back();
        if (!l.empty()) v.push_back(l);
    }
    return v;
}
static std::vector<uint8_t> num(const std::string& s) { return AdapterSet::from_string(s); }
static std::string str(const uint8_t* b, int n) {
    std::string s;
    for (int i = 0; i < n; ++i) s.push_back("ACGTN"[b[i]]);
    return s;
}

int main() {
    const std::string W1 = "GAGTGATTGCTTGTGACGCCTT";
    const auto bc1 = read_list(std::string(WHITELIST_DIR) + "/indrop_bc1.txt");
    const auto bc2 = read_list(std::string(WHITELIST_DIR) + "/indrop_bc2.txt");
    CHECK("bc1 list has 384 entries", bc1.size() == 384);
    CHECK("bc2 list has 384 entries", bc2.size() == 384);
    int len_count[13] = {};
    for (const auto& b : bc1) ++len_count[b.size()];
    CHECK("bc1 holds 96 barcodes of each length 8-11", len_count[8] == 96 && len_count[9] == 96 && len_count[10] == 96 && len_count[11] == 96);

    IndropAnchor a;
    CHECK("anchor loads", a.load({WHITELIST_DIR}));
    std::mt19937 rng(7);
    std::string pad = "TTTTTTTTTTTTTTTT";
    uint8_t out[IndropAnchor::kCanonLen];
    int n_ok = 0, n_tot = 0, n_len_ok = 0;
    for (int it = 0; it < 4000; ++it) {
        const std::string& b1 = bc1[rng() % bc1.size()];
        const std::string& b2 = bc2[rng() % bc2.size()];
        std::string umi;
        for (int i = 0; i < 6; ++i) umi.push_back("ACGT"[rng() % 4]);
        std::string read = b1 + W1 + b2 + umi + pad;
        read.resize(56, 'A');
        auto r = num(read);
        ++n_tot;
        const int rc = a.canonicalize(r.data(), static_cast<uint16_t>(r.size()), out);
        if (rc == IndropAnchor::kOk) {
            ++n_ok;
            // BC1 sits left-aligned in the first 11 slots, BC2 follows, then the UMI
            const std::string got1 = str(out, static_cast<int>(b1.size()));
            const std::string got2 = str(out + IndropAnchor::kBc2Start, 8);
            const std::string gotu = str(out + IndropAnchor::kUmiStart, 6);
            if (got1 == b1 && got2 == b2 && gotu == umi) ++n_len_ok;
        }
    }
    CHECK("all clean reads resolve", n_ok == n_tot);
    CHECK("all clean reads give the same barcodes and UMI", n_len_ok == n_tot);

    // one substitution in BC1 and BC2 is corrected, two in BC1 are not, W1 with 3 mismatches still anchors, 4 does not
    {
        const std::string b1 = bc1[5], b2 = bc2[9];
        auto build = [&](std::string x1, std::string x2, std::string w1) {
            std::string read = x1 + w1 + x2 + "ACGTAC" + pad;
            read.resize(56, 'A');
            return num(read);
        };
        auto sub = [](std::string s, size_t i) { s[i] = (s[i] == 'A') ? 'C' : 'A'; return s; };
        auto r = build(sub(b1, 2), sub(b2, 3), W1);
        CHECK("one substitution per barcode is corrected", a.canonicalize(r.data(), 56, out) == IndropAnchor::kOk && str(out, static_cast<int>(b1.size())) == b1 && str(out + 11, 8) == b2);
        r = build(sub(sub(b1, 1), 4), b2, W1);
        CHECK("two substitutions in BC1 are not corrected", a.canonicalize(r.data(), 56, out) == IndropAnchor::kBc1);
        std::string w3 = sub(sub(sub(W1, 0), 7), 15);
        r = build(b1, b2, w3);
        CHECK("W1 with three mismatches still anchors", a.canonicalize(r.data(), 56, out) == IndropAnchor::kOk);
        std::string w4 = sub(sub(sub(sub(W1, 0), 7), 15), 19);
        r = build(b1, b2, w4);
        CHECK("W1 with four mismatches does not", a.canonicalize(r.data(), 56, out) == IndropAnchor::kNoW1);
        auto shortr = num(std::string(40, 'A'));
        CHECK("a read too short for BC2 and UMI fails", a.canonicalize(shortr.data(), 40, out) == IndropAnchor::kShort);
        bool all_n = true;
        for (int i = 0; i < IndropAnchor::kCanonLen; ++i) all_n &= (out[i] == 4);
        CHECK("a failed read is N in every slot", all_n);
    }

    // adapter trimming
    {
        AdapterSet ad;
        ad.seqs = {AdapterSet::from_string("AGATCGGAAGAGC"), AdapterSet::from_string("AAGGCGTCACAAGCAATCACTC")};
        std::string insert = "GGTGGTGTCTACGTTAGC";
        auto r = num(insert + "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGTAGATCGGAAGAGCACACG");
        CHECK("full adapter is cut at its start", ad.cut(r.data(), static_cast<uint16_t>(r.size())) == insert.size());
        r = num(insert + "AAGGCGTCACAAGCAATCACTCAAT");
        CHECK("reverse complement of W1 is cut", ad.cut(r.data(), static_cast<uint16_t>(r.size())) == insert.size());
        r = num(std::string("ACGTTGCAAGTCCATTGGACCGATTTAGCCAAGTCGATGCCATGATTG") + "AGATCGGA");
        CHECK("a partial adapter at the read end is cut", ad.cut(r.data(), static_cast<uint16_t>(r.size())) == 48);
        r = num("ACGTTGCAAGTCCATTGGACCGATTTAGCCAAGTCGATGCCATGATTGACGTAGCT");
        CHECK("a read without adapter is kept whole", ad.cut(r.data(), static_cast<uint16_t>(r.size())) == r.size());
    }
    std::cout << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
