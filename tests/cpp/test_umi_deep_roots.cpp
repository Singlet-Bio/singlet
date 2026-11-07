// SPDX-License-Identifier: MIT
// tests/cpp/test_umi_deep_roots.cpp
//
// umi3_directional_roots() (R9 junction / V(D)J / SNP molecules) and DirectionalUmiStore::gene_root() used to test every
// pair of a (barcode, feature) group; deep groups now find the same edges through a neighbour index. The union-find root
// of a component depends on the order edges are joined, so this checks the roots themselves (not only the component count)
// against the original pair loops, on groups on both sides of the 512-UMI switch.

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "singlet/pileup/umi_dedup.h"
#include "test_harness.h"

using namespace singlet;

namespace {

struct E { uint64_t umi; uint32_t n; };

// The original pair loop, verbatim.
std::vector<uint32_t> old_roots3(const std::vector<E>& v) {
    const size_t n = v.size();
    std::vector<uint32_t> root(n);
    for (size_t i = 0; i < n; ++i) root[i] = static_cast<uint32_t>(i);
    auto find = [&](uint32_t x) { while (root[x] != x) { root[x] = root[root[x]]; x = root[x]; } return x; };
    for (size_t i = 0; i + 1 < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if (v[i].n >= 2 * v[j].n - 1 && umi3_one_mismatch(v[i].umi, v[j].umi)) {
                const uint32_t ri = find(static_cast<uint32_t>(i)), rj = find(static_cast<uint32_t>(j));
                if (ri != rj) root[rj] = ri;
            }
    for (size_t i = 0; i < n; ++i) root[i] = find(static_cast<uint32_t>(i));
    return root;
}

std::vector<E> make_group(std::mt19937_64& rng, size_t n, int len, int alphabet, bool clustered) {
    std::vector<E> v;
    std::vector<uint64_t> seen;
    auto rnd = [&]() {
        uint64_t p = 1;
        for (int i = 0; i < len; ++i) p = (p << 3) | (rng() % alphabet);
        return p;
    };
    uint64_t base = rnd();
    while (v.size() < n) {
        uint64_t u = rnd();
        if (clustered && (rng() & 1)) {   // a one-base error of an earlier UMI: many real edges
            u = v.empty() ? base : v[rng() % v.size()].umi;
            const int p = static_cast<int>(rng() % len);
            u ^= (((u >> (3 * p)) & 7ULL) ^ (rng() % alphabet)) << (3 * p);
        }
        if (std::find(seen.begin(), seen.end(), u) != seen.end()) continue;
        seen.push_back(u);
        v.push_back({u, 1u + static_cast<uint32_t>(rng() % 40) * static_cast<uint32_t>(rng() % 3 == 0 ? 8 : 1)});
    }
    std::sort(v.begin(), v.end(), [](const E& a, const E& b) { return a.n != b.n ? a.n > b.n : a.umi < b.umi; });
    return v;
}

void test_roots3() {
    std::mt19937_64 rng(7);
    for (size_t n : {10u, 300u, 512u, 513u, 900u, 2500u}) {
        for (int len : {10, 12}) {
            for (bool clustered : {false, true}) {
                auto v = make_group(rng, n, len, 6, clustered);
                std::vector<uint32_t> got;
                umi3_directional_roots(v.size(), [&](size_t i) { return v[i].umi; }, [&](size_t i) { return v[i].n; }, got);
                CHECK(got == old_roots3(v), "umi3_directional_roots roots identical to the pair loop (n=" + std::to_string(n) +
                                            ", len=" + std::to_string(len) + (clustered ? ", clustered)" : ", random)"));
            }
        }
    }
}

// gene_root through the public store: record_read() of every UMI, then compare each UMI's root to a pair-loop reference.
void test_gene_root() {
    std::mt19937_64 rng(11);
    for (size_t n : {40u, 513u, 1500u}) {
        const int len = 10;
        std::vector<E> v;
        std::vector<uint64_t> seen;
        while (v.size() < n) {
            uint64_t p2 = 0;
            for (int i = 0; i < len; ++i) p2 |= (rng() % 4) << (2 * i);
            if (rng() & 1 && !v.empty()) {
                p2 = v[rng() % v.size()].umi;
                p2 ^= (1ULL + rng() % 3) << (2 * (rng() % len));
            }
            if (std::find(seen.begin(), seen.end(), p2) != seen.end()) continue;
            seen.push_back(p2);
            v.push_back({p2, 1u + static_cast<uint32_t>(rng() % 30)});
        }
        DirectionalUmiStore st;
        for (const auto& e : v)
            for (uint32_t k = 0; k < e.n; ++k) st.record_read(1, 2, e.umi, 0, UINT32_MAX, DirectionalUmiStore::kReadExonOnly);
        auto s = v;
        std::sort(s.begin(), s.end(), [](const E& a, const E& b) { return a.n != b.n ? a.n > b.n : a.umi < b.umi; });
        std::vector<uint32_t> parent(s.size());
        for (size_t i = 0; i < s.size(); ++i) parent[i] = static_cast<uint32_t>(i);
        auto find = [&](uint32_t x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
        for (size_t i = 0; i + 1 < s.size(); ++i)
            for (size_t j = i + 1; j < s.size(); ++j)
                if (umi_hamming_2bit(s[i].umi, s[j].umi, len) == 1 && s[i].n >= 2 * s[j].n - 1) {
                    uint32_t ri = find(static_cast<uint32_t>(i)), rj = find(static_cast<uint32_t>(j));
                    if (ri != rj) parent[rj] = ri;
                }
        bool same = true;
        for (size_t i = 0; i < s.size(); ++i) {
            uint64_t r = 0;
            if (!st.gene_root(1, 2, s[i].umi, len, true, r) || r != s[find(static_cast<uint32_t>(i))].umi) same = false;
        }
        CHECK(same, "gene_root roots identical to the pair loop (n=" + std::to_string(n) + ")");
    }
}

}  // namespace

int main() {
    test_roots3();
    test_gene_root();
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
