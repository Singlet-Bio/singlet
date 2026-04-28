// N6: Unit tests for directional UMI deduplication
// Run standalone: compile + execute without CMake
// g++ -std=c++17 -O2 -o /tmp/test_dd src/test_directional_dedup.cpp && /tmp/test_dd

#include "../include/singlet-pileup/umi_dedup.h"
#include <cassert>
#include <iostream>

using namespace singlet;

int main() {
    int passed = 0, failed = 0;
    auto PASS = [&](const char* name) { ++passed; std::cerr << "PASS: " << name << "\n"; };
    auto FAIL = [&](const char* name, const char* msg) {
        ++failed; std::cerr << "FAIL: " << name << " — " << msg << "\n";
    };

    // ── 1. umi_pack_2bit: A=0 C=1 G=2 T=3, little-endian ──
    {
        const char* umi = "ACGT";  // 4-mer
        uint64_t p = umi_pack_2bit(umi, 4);
        // A@bits[1:0]=0, C@bits[3:2]=1, G@bits[5:4]=2, T@bits[7:6]=3
        // Expected: 0b11100100 = 0xE4
        if ((p & 0xFF) == 0xE4) PASS("umi_pack_2bit ACGT");
        else FAIL("umi_pack_2bit ACGT", "wrong encoding");
    }

    // ── 2. umi_hamming_2bit: identical → 0 ──
    {
        uint64_t a = umi_pack_2bit("ACGTACGT", 8);
        if (umi_hamming_2bit(a, a, 8) == 0) PASS("hamming(same)=0");
        else FAIL("hamming(same)=0", "should be 0");
    }

    // ── 3. umi_hamming_2bit: 1 substitution → 1 ──
    {
        uint64_t a = umi_pack_2bit("ACGTACGT", 8);
        uint64_t b = umi_pack_2bit("ACGTACGC", 8);  // T→C at pos 7
        if (umi_hamming_2bit(a, b, 8) == 1) PASS("hamming(1 sub)=1");
        else FAIL("hamming(1 sub)=1", "should be 1");
    }

    // ── 4. umi_hamming_2bit: 2 substitutions → 2 ──
    {
        uint64_t a = umi_pack_2bit("ACGTACGT", 8);
        uint64_t b = umi_pack_2bit("TCGTACGC", 8);  // A→T at pos 0, T→C at pos 7
        if (umi_hamming_2bit(a, b, 8) == 2) PASS("hamming(2 subs)=2");
        else FAIL("hamming(2 subs)=2", "should be 2");
    }

    // ── 5. directional_dedup: single UMI → 1 component ──
    {
        std::vector<std::pair<uint64_t, uint32_t>> umis = {
            {umi_pack_2bit("ACGTACGT", 8), 10}
        };
        auto r = directional_dedup(umis, 8);
        if (r == 1) PASS("directional: single UMI→1");
        else FAIL("directional: single UMI→1", "should be 1");
    }

    // ── 6. directional_dedup: equal-count hamming-1 UMIs NOT merged ──
    {
        // count=5, count=5: 5 >= 2*5-1=9? NO → no edge → 2 components
        uint64_t a = umi_pack_2bit("ACGTACGT", 8);
        uint64_t b = umi_pack_2bit("ACGTACGC", 8);
        std::vector<std::pair<uint64_t, uint32_t>> umis = {{a,5},{b,5}};
        auto r = directional_dedup(umis, 8);
        if (r == 2) PASS("equal-count hamming-1 NOT merged");
        else FAIL("equal-count hamming-1 NOT merged", ("got " + std::to_string(r)).c_str());
    }

    // ── 7. directional_dedup: high absorbs low at hamming=1 ──
    {
        // count_a=10, count_b=1: 10 >= 2*1-1=1 YES → merged → 1 component
        uint64_t a = umi_pack_2bit("ACGTACGT", 8);
        uint64_t b = umi_pack_2bit("ACGTACGC", 8);
        std::vector<std::pair<uint64_t, uint32_t>> umis = {{a,10},{b,1}};
        auto r = directional_dedup(umis, 8);
        if (r == 1) PASS("10×1 hamming-1 merged→1");
        else FAIL("10×1 hamming-1 merged→1", "should be 1");
    }

    // ── 8. directional_dedup: chain a(100)→b(10)→c(2) collapses to 1 ──
    {
        uint64_t a = umi_pack_2bit("ACGTACGT", 8);
        uint64_t b = umi_pack_2bit("ACGTACGC", 8);  // hamming(a,b)=1
        uint64_t c = umi_pack_2bit("ACGTACCC", 8);  // hamming(b,c)=1, hamming(a,c)=2
        // a→b: 100>=2*10-1=19 YES; b→c: 10>=2*2-1=3 YES; a+b+c → 1 component
        std::vector<std::pair<uint64_t, uint32_t>> umis = {{a,100},{b,10},{c,2}};
        auto r = directional_dedup(umis, 8);
        if (r == 1) PASS("chain a(100)→b(10)→c(2)=1");
        else FAIL("chain a(100)→b(10)→c(2)=1", "should be 1");
    }

    // ── 9. directional_dedup: two independent clusters → 2 ──
    {
        uint64_t a  = umi_pack_2bit("AAAAAAAA", 8); // count=50
        uint64_t a2 = umi_pack_2bit("AAAAAAAC", 8); // count=5, hamming(a,a2)=1
        uint64_t d  = umi_pack_2bit("TTTTTTTT", 8); // count=40
        uint64_t e  = umi_pack_2bit("TTTTTTGT", 8); // count=4, hamming(d,e)=1
        // a absorbs a2 (50>=9 YES), d absorbs e (40>=7 YES), clusters don't share edge
        std::vector<std::pair<uint64_t, uint32_t>> umis = {{a,50},{d,40},{a2,5},{e,4}};
        std::sort(umis.begin(), umis.end(), [](auto& x, auto& y){ return x.second > y.second; });
        auto r = directional_dedup(umis, 8);
        if (r == 2) PASS("two clusters→2");
        else FAIL("two clusters→2", ("got " + std::to_string(r)).c_str());
    }

    // ── 10. DirectionalUmiStore: accumulate + finalize ──
    {
        // Fake SparseAccumulator-like struct for testing
        struct FakeAcc {
            std::vector<std::pair<uint32_t,uint32_t>> entries;
            void increment(uint32_t feat, uint32_t bc, uint16_t /*delta*/ = 1) {
                entries.push_back({feat, bc});
            }
        } acc;

        DirectionalUmiStore store;
        uint64_t umi_a = umi_pack_2bit("ACGTACGT", 8);  // count=10
        uint64_t umi_b = umi_pack_2bit("ACGTACGC", 8);  // count=1, hamming=1 → merged

        // Cell 0, gene 5, exon 7: add umi_a 10 times and umi_b 1 time
        for (int i = 0; i < 10; i++) store.record(0, 5, 7, umi_a);
        store.record(0, 5, 7, umi_b);

        // After directional dedup: 1 component (exon 7, bc 0)
        uint64_t deduped = store.finalize(acc, 8);
        if (deduped == 1 && acc.entries.size() == 1 &&
            acc.entries[0].first == 7 && acc.entries[0].second == 0)
            PASS("DirectionalUmiStore finalize");
        else FAIL("DirectionalUmiStore finalize",
                  ("deduped=" + std::to_string(deduped) +
                   " entries=" + std::to_string(acc.entries.size())).c_str());
    }

    std::cerr << "\n" << passed << "/" << (passed+failed) << " tests passed";
    if (failed == 0) std::cerr << " — ALL OK\n";
    else std::cerr << " — " << failed << " FAILED\n";
    return (failed > 0) ? 1 : 0;
}
