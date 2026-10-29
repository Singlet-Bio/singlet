// Unit test for order-free-rules R9 (UmiFeatureCollapse, umi3_*, SnpConsensus with directional collapse).
// Build: g++ -O2 -std=c++17 -Iinclude tests/test_umi_r9.cpp -o test_umi_r9 && ./test_umi_r9
#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include "singlet/pileup/umi_dedup.h"

using singlet::UmiFeatureCollapse;
using singlet::SnpConsensus;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

struct Read { uint32_t bc, f; std::string umi; };
using Emitted = std::map<std::pair<uint32_t, uint32_t>, uint32_t>;

static Emitted collapse(const std::vector<Read>& reads, bool directional) {
    UmiFeatureCollapse c;
    for (const auto& r : reads) c.add(r.bc, r.f, r.umi.empty() ? nullptr : r.umi.c_str(), static_cast<int>(r.umi.size()));
    Emitted out;
    c.finalize([&](uint32_t bc, uint32_t f, uint32_t n) { out[{bc, f}] += n; }, directional);
    return out;
}

static std::vector<Read> rep(uint32_t bc, uint32_t f, const std::string& u, int n) { return std::vector<Read>(n, Read{bc, f, u}); }
static void append(std::vector<Read>& a, const std::vector<Read>& b) { a.insert(a.end(), b.begin(), b.end()); }

struct Acc {
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> m;
    void increment(uint32_t f, uint32_t bc, uint32_t d = 1) { m[{f, bc}] += d; }
};
struct St { uint64_t snp_molecules = 0, snp_molecules_ambiguous = 0, snp_molecules_other = 0; };

int main() {
    using singlet::umi_pack3; using singlet::umi3_one_mismatch;
    // packing and mismatch test
    CHECK(umi3_one_mismatch(umi_pack3("AAAAAAAAAA", 10), umi_pack3("AAAAAAAAAC", 10)));
    CHECK(umi3_one_mismatch(umi_pack3("AAAAAAAAAA", 10), umi_pack3("CAAAAAAAAA", 10)));
    CHECK(!umi3_one_mismatch(umi_pack3("AAAAAAAAAA", 10), umi_pack3("AAAAAAAACC", 10)));
    CHECK(!umi3_one_mismatch(umi_pack3("AAAAAAAAAA", 10), umi_pack3("AAAAAAAAAA", 10)));
    CHECK(!umi3_one_mismatch(umi_pack3("AAAAAAAAAA", 10), umi_pack3("AAAAAAAAAAA", 11)));   // length differs
    CHECK(umi3_one_mismatch(umi_pack3("AAAAAAAAAA", 10), umi_pack3("AAAAANAAAA", 10)));     // N is a base class
    CHECK(umi3_one_mismatch(umi_pack3("TTTTTTTTTTTTTTTTTTTTT", 21), umi_pack3("TTTTTTTTTTTTTTTTTTTTG", 21)));

    // directional absorption: 10 vs 1 merges, 2 vs 2 does not, 3 vs 2 merges (3 >= 2*2-1)
    {
        std::vector<Read> r = rep(0, 5, "AAAAAAAAAA", 10); append(r, rep(0, 5, "AAAAAAAAAC", 1));
        CHECK((collapse(r, true)[{0, 5}] == 1));
        CHECK((collapse(r, false)[{0, 5}] == 2));
    }
    {
        std::vector<Read> r = rep(0, 5, "AAAAAAAAAA", 2); append(r, rep(0, 5, "AAAAAAAAAC", 2));
        CHECK((collapse(r, true)[{0, 5}] == 2));
    }
    {
        std::vector<Read> r = rep(0, 5, "AAAAAAAAAA", 3); append(r, rep(0, 5, "AAAAAAAAAC", 2));
        CHECK((collapse(r, true)[{0, 5}] == 1));
    }
    // chain A(10) -> B(4) -> C(1): A absorbs B, B absorbs C: one molecule; groups never mix
    {
        std::vector<Read> r = rep(1, 7, "ACGTACGTAC", 10); append(r, rep(1, 7, "ACGTACGTAA", 4)); append(r, rep(1, 7, "ACGTACGTTA", 1));
        append(r, rep(2, 7, "ACGTACGTAA", 1)); append(r, rep(1, 8, "ACGTACGTAA", 1));
        auto e = collapse(r, true);
        CHECK((e[{1, 7}] == 1)); CHECK((e[{2, 7}] == 1)); CHECK((e[{1, 8}] == 1));
    }
    // reads without a UMI are one molecule each; long UMIs exact
    {
        std::vector<Read> r = rep(3, 1, "", 4); append(r, rep(3, 1, std::string(24, 'A'), 3));
        append(r, rep(3, 1, std::string(23, 'A') + "C", 1));
        CHECK((collapse(r, true)[{3, 1}] == 6));
    }
    // order independence: random data, 20 permutations, identical output
    {
        std::mt19937 rng(7);
        const char* bases = "ACGT";
        std::vector<Read> r;
        for (int k = 0; k < 4000; ++k) {
            std::string u(10, 'A');
            for (auto& ch : u) ch = bases[rng() % 4];
            const int depth = 1 + static_cast<int>(rng() % 12);
            append(r, rep(rng() % 5, rng() % 7, u, depth));
            if (rng() % 3 == 0) { std::string e = u; e[rng() % 10] = bases[rng() % 4]; append(r, rep(r.back().bc, r.back().f, e, 1)); }
        }
        const Emitted ref = collapse(r, true);
        bool same = true;
        for (int t = 0; t < 20; ++t) { std::shuffle(r.begin(), r.end(), rng); same = same && collapse(r, true) == ref; }
        CHECK(same);
    }
    // SNP: UMI X has 3 REF reads, its 1-mismatch neighbour 1 ALT read -> one molecule, REF (DP 1, AD 0)
    {
        SnpConsensus s;
        for (int i = 0; i < 3; ++i) s.add(0, 9, "GGGGGGGGGG", 10, SnpConsensus::kRef);
        s.add(0, 9, "GGGGGGGGGA", 10, SnpConsensus::kAlt);
        Acc ad, dp; St st;
        s.finalize(ad, dp, st, true);
        CHECK((dp.m[{9, 0}] == 1)); CHECK((ad.m[{9, 0}] == 0)); CHECK(st.snp_molecules == 1);
        Acc ad2, dp2; St st2;
        s.finalize(ad2, dp2, st2, false);
        CHECK((dp2.m[{9, 0}] == 2)); CHECK((ad2.m[{9, 0}] == 1));
    }
    // SNP: merged tie becomes ambiguous (2 REF + 2 ALT, 2 >= 2*2-1 fails -> stays 2 molecules, each decided)
    {
        SnpConsensus s;
        s.add(1, 2, "CCCCCCCCCC", 10, SnpConsensus::kRef); s.add(1, 2, "CCCCCCCCCC", 10, SnpConsensus::kRef);
        s.add(1, 2, "CCCCCCCCCA", 10, SnpConsensus::kAlt); s.add(1, 2, "CCCCCCCCCA", 10, SnpConsensus::kAlt);
        Acc ad, dp; St st;
        s.finalize(ad, dp, st, true);
        CHECK((dp.m[{2, 1}] == 2)); CHECK((ad.m[{2, 1}] == 1));
    }
    // R9 per (cell, gene): the gene's molecule store decides which evidence UMIs are one molecule
    {
        using singlet::DirectionalUmiStore; using singlet::umi_pack_2bit;
        DirectionalUmiStore genes;
        auto rec = [&](uint32_t bc, uint32_t g, const char* u, int n) {
            for (int i = 0; i < n; ++i) genes.record_read(bc, g, umi_pack_2bit(u, 10), 0, UINT32_MAX, DirectionalUmiStore::kReadExonOnly);
        };
        rec(0, 4, "ACGTACGTAC", 20); rec(0, 4, "ACGTACGTAA", 18);   // two templates (20 < 2*18-1)
        rec(0, 5, "TTTTGGGGCC", 20); rec(0, 5, "TTTTGGGGCA", 2);    // parent and error copy
        auto gr = [&](uint32_t bc, uint32_t g, uint64_t u3, uint64_t& r) {
            int len = 0; const uint64_t p2 = singlet::umi3_to_2bit(u3, &len);
            return genes.gene_root(bc, g, p2, len, true, r);
        };
        // feature evidence: template pair 3 vs 1 read (per feature would merge), error pair 2 vs 2 (would not)
        UmiFeatureCollapse c;
        for (int i = 0; i < 3; ++i) c.add(0, 1, "ACGTACGTAC", 10, 4);
        c.add(0, 1, "ACGTACGTAA", 10, 4);
        for (int i = 0; i < 2; ++i) { c.add(0, 2, "TTTTGGGGCC", 10, 5); c.add(0, 2, "TTTTGGGGCA", 10, 5); }
        c.add(0, 3, "GGGGGGGGGG", 10); c.add(0, 3, "GGGGGGGGGA", 10);   // no gene: per feature, merges (1 >= 1)
        Emitted pf, pg;
        c.finalize([&](uint32_t bc, uint32_t f, uint32_t n) { pf[{bc, f}] += n; }, true);
        c.finalize([&](uint32_t bc, uint32_t f, uint32_t n) { pg[{bc, f}] += n; }, true, gr);
        CHECK((pf[{0, 1}] == 1)); CHECK((pg[{0, 1}] == 2));
        CHECK((pf[{0, 2}] == 2)); CHECK((pg[{0, 2}] == 1));
        CHECK((pf[{0, 3}] == 1)); CHECK((pg[{0, 3}] == 1));
        // the same through SnpConsensus: the template pair stays two molecules (REF and ALT)
        SnpConsensus s;
        for (int i = 0; i < 3; ++i) s.add(0, 9, "ACGTACGTAC", 10, SnpConsensus::kRef, 4);
        s.add(0, 9, "ACGTACGTAA", 10, SnpConsensus::kAlt, 4);
        Acc ad, dp; St st;
        s.finalize(ad, dp, st, true, gr);
        CHECK((dp.m[{9, 0}] == 2)); CHECK((ad.m[{9, 0}] == 1));
        // order independence with genes: shuffled adds, identical output
        std::mt19937 rng(11);
        std::vector<Read> r = rep(0, 1, "ACGTACGTAC", 3); append(r, rep(0, 1, "ACGTACGTAA", 1));
        append(r, rep(0, 2, "TTTTGGGGCC", 2)); append(r, rep(0, 2, "TTTTGGGGCA", 2));
        Emitted ref;
        for (int t = 0; t < 10; ++t) {
            std::shuffle(r.begin(), r.end(), rng);
            UmiFeatureCollapse d;
            for (const auto& x : r) d.add(x.bc, x.f, x.umi.c_str(), 10, x.f == 1 ? 4u : 5u);
            Emitted e;
            d.finalize([&](uint32_t bc, uint32_t f, uint32_t n) { e[{bc, f}] += n; }, true, gr);
            if (t == 0) ref = e; else CHECK(e == ref);
        }
    }
    std::printf(failures ? "R9 tests: %d FAILED\n" : "R9 tests: all passed%.0d\n", failures);
    return failures ? 1 : 0;
}
