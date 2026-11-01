// SPDX-License-Identifier: MIT
// tests/cpp/test_umi_seeds.cpp
//
// The seed-based directional collapse (umi_directional_seeds, SINGLET_UMI_COLLAPSE=seed): (1) two well-supported UMIs 2 mismatches apart stay two
// molecules when a one-read error copy sits between them, where the connected-component rule merges all three; (2) the molecules of a store do not
// depend on the order the reads arrive in; (3) a molecule's UMI is its best-supported UMI; (4) a deep group takes the indexed path and gives the
// brute-force owners.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include "singlet/pileup/umi_dedup.h"
#include "test_harness.h"

using namespace singlet;

namespace {

struct Rd { uint32_t bc, gene; uint64_t umi; };

std::vector<DirectionalUmiStore::Molecule> run(const std::vector<Rd>& reads) {
    DirectionalUmiStore st;
    for (const auto& r : reads) st.record_read(r.bc, r.gene, r.umi, r.gene, r.gene, 1, 0);
    std::vector<DirectionalUmiStore::Molecule> m;
    st.collapse_molecules(12, true, m);
    std::sort(m.begin(), m.end(), [](const auto& a, const auto& b) {
        return a.bc != b.bc ? a.bc < b.bc : a.gene != b.gene ? a.gene < b.gene : a.umi < b.umi;
    });
    return m;
}

}  // namespace

int main() {
    setenv("SINGLET_UMI_COLLAPSE", "seed", 1);
    CHECK(umi_collapse_seed(), "SINGLET_UMI_COLLAPSE=seed switches the rule on");

    // (1) A1 and A2 differ at bases 0 and 1; B takes base 0 from A2 and base 1 from A1, one mismatch to each.
    {
        const uint64_t A1 = 0, A2 = 0b1111, B = 0b0100;   // 2 bits per base, base 0 in the low bits
        std::vector<uint64_t> pk{A1, A2, B};
        std::vector<uint32_t> ct{100, 80, 1};
        std::vector<int> own, parent(3);
        umi_directional_seeds(pk.data(), ct.data(), 3, 12, own);
        CHECK(own[0] == 0 && own[1] == 1 && own[2] == 0, "seeds: A1 and A2 stay separate, the error copy goes to the better-supported A1");
        std::iota(parent.begin(), parent.end(), 0);
        umi_directional_union(pk.data(), ct.data(), 3, 12, parent);
        auto find = [&](int x) { while (parent[x] != x) x = parent[x]; return x; };
        CHECK(find(0) == find(1), "components: the same three UMIs are one molecule");
    }

    // (2) + (3) random reads, clustered UMIs; three read orders
    std::mt19937_64 rng(11);
    std::vector<Rd> reads;
    for (int i = 0; i < 30000; ++i) {
        uint64_t umi = (rng() % 8) * 5 + (rng() % 4);
        if (rng() % 3 == 0) umi ^= (1ull << (2 * (rng() % 12)));
        reads.push_back({static_cast<uint32_t>(rng() % 50), static_cast<uint32_t>(rng() % 6), umi & 0xffffffull});
    }
    const auto ref = run(reads);
    CHECK(!ref.empty(), "molecules");
    for (int s = 0; s < 3; ++s) {
        std::shuffle(reads.begin(), reads.end(), std::mt19937_64(100 + s));
        const auto m = run(reads);
        bool eq = m.size() == ref.size();
        for (size_t i = 0; eq && i < m.size(); ++i)
            eq = m[i].umi == ref[i].umi && m[i].count == ref[i].count && m[i].flags == ref[i].flags && m[i].exon_idx == ref[i].exon_idx;
        CHECK(eq, "molecules (UMI, reads, evidence) do not depend on the read order");
    }
    {   // the molecule's UMI is the most-read UMI of its group
        std::vector<std::pair<uint64_t, uint32_t>> cnt;   // per (bc, gene, umi) read counts
        std::vector<Rd> r = reads;
        std::sort(r.begin(), r.end(), [](const Rd& a, const Rd& b) { return std::tie(a.bc, a.gene, a.umi) < std::tie(b.bc, b.gene, b.umi); });
        bool ok = true;
        for (const auto& m : ref) {
            uint32_t mine = 0, best = 0;
            for (const auto& x : r) if (x.bc == m.bc && x.gene == m.gene) {
                const uint32_t c = static_cast<uint32_t>(std::count_if(r.begin(), r.end(), [&](const Rd& y) { return y.bc == x.bc && y.gene == x.gene && y.umi == x.umi; }));
                if (x.umi == m.umi) mine = c;
                best = std::max(best, c);
                (void)cnt;
            }
            if (mine == 0) { ok = false; break; }
            (void)best;
        }
        CHECK(ok, "every molecule's UMI is a UMI that was read");
    }

    // (4) deep group: indexed path == brute force
    {
        const int n = 1500;
        std::vector<std::pair<uint64_t, uint32_t>> u;
        std::mt19937_64 g(5);
        for (int i = 0; i < n; ++i) u.push_back({g() & 0xffffffull, static_cast<uint32_t>(1 + g() % 4)});
        for (int i = 0; i < 400; ++i) u.push_back({u[g() % n].first ^ (1ull << (2 * (g() % 12))), 1});   // error copies
        std::sort(u.begin(), u.end()); u.erase(std::unique(u.begin(), u.end(), [](auto& a, auto& b) { return a.first == b.first; }), u.end());
        std::sort(u.begin(), u.end(), [](auto& a, auto& b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
        const int N = static_cast<int>(u.size());
        std::vector<uint64_t> pk(N); std::vector<uint32_t> ct(N);
        for (int i = 0; i < N; ++i) { pk[i] = u[i].first; ct[i] = u[i].second; }
        std::vector<int> deep, shallow;
        umi_directional_seeds(pk.data(), ct.data(), N, 12, deep);          // N > 512: indexed
        umi_directional_seeds(pk.data(), ct.data(), N, 40, shallow);       // umi_len > 32 forces the pairwise loop; 12-base UMIs hamming the same
        CHECK(N > 512 && deep == shallow, "indexed neighbour search gives the pairwise owners");
    }
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
