// SPDX-License-Identifier: MIT
// tests/cpp/test_umi_store_scale.cpp
//
// The UMI store keeps each (barcode, gene) group sorted by packed UMI and finds
// directional edges through a neighbour index for big groups. Checks that both give
// exactly the molecule counts of the original pairwise algorithm, including on groups
// large enough to take the indexed path, and that read order does not matter.

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

#include "singlet/pileup/umi_dedup.h"
#include "test_harness.h"

using singlet::DirectionalUmiStore;

namespace {

struct Acc {
    std::map<std::pair<uint32_t, uint32_t>, int> n;
    void increment(uint32_t f, uint32_t bc) { n[{f, bc}]++; }
};

// The original O(n^2) algorithm, verbatim semantics.
uint32_t reference_components(std::vector<std::pair<uint64_t, uint32_t>> v, int len) {
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first; });
    const int n = static_cast<int>(v.size());
    std::vector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;
    auto find = [&](int x) { while (parent[x] != x) x = parent[x] = parent[parent[x]]; return x; };
    for (int i = 0; i < n - 1; ++i)
        for (int j = i + 1; j < n; ++j)
            if (singlet::umi_hamming_2bit(v[i].first, v[j].first, len) == 1 &&
                v[i].second >= 2 * v[j].second - 1) {
                int a = find(i), b = find(j);
                if (a != b) parent[b] = a;
            }
    uint32_t r = 0;
    for (int i = 0; i < n; ++i) r += find(i) == i;
    return r;
}

void test_equivalence() {
    std::cout << "\n=== indexed neighbour search == pairwise ===\n";
    std::mt19937_64 rng(7);
    for (int len : {10, 12, 16}) {
        for (int n : {50, 600, 3000}) {
            // a few heavy UMIs with sequencing-error neighbours plus random noise
            std::map<uint64_t, uint32_t> m;
            const uint64_t mask = (1ULL << (2 * len)) - 1;
            for (int k = 0; k < n / 10 + 1; ++k) {
                uint64_t u = rng() & mask;
                m[u] += 20 + rng() % 200;
                for (int e = 0; e < 8; ++e) m[u ^ (((rng() % 3) + 1) << (2 * (rng() % len)))] += 1 + rng() % 3;
            }
            while (static_cast<int>(m.size()) < n) m[rng() & mask] += 1 + rng() % 2;
            std::vector<std::pair<uint64_t, uint32_t>> v(m.begin(), m.end());
            std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
            const uint32_t want = reference_components(v, len);
            CHECK(singlet::directional_dedup(v, len) == want,
                  "directional_dedup matches pairwise (len " + std::to_string(len) + ", n " +
                  std::to_string(v.size()) + ")");

            DirectionalUmiStore s;
            std::vector<uint64_t> reads;
            for (auto& [u, c] : v) for (uint32_t k = 0; k < c; ++k) reads.push_back(u);
            std::shuffle(reads.begin(), reads.end(), rng);
            for (uint64_t u : reads) s.record_read(1, 2, u, 5, UINT32_MAX, DirectionalUmiStore::kReadExonOnly);
            Acc ex, in;
            auto st = s.finalize_molecules(ex, in, [](uint32_t, uint32_t) {}, len, true);
            CHECK(st.total() == want, "finalize_molecules molecule count matches pairwise");
            std::sort(reads.begin(), reads.end());
            DirectionalUmiStore s2;
            for (uint64_t u : reads) s2.record_read(1, 2, u, 5, UINT32_MAX, DirectionalUmiStore::kReadExonOnly);
            Acc ex2, in2;
            s2.finalize_molecules(ex2, in2, [](uint32_t, uint32_t) {}, len, true);
            CHECK(ex.n == ex2.n, "read order does not change the counts");
        }
    }
}

}  // namespace

int main() {
    test_equivalence();
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
