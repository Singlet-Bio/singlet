// SPDX-License-Identifier: MIT
// tests/cpp/test_umi_consume.cpp
//
// DirectionalUmiStore::consume_molecules() (the end-of-stream finalize hands each (barcode, gene) group back as soon
// as it is collapsed) must give the molecules collapse_molecules() gives, in the same order, and must keep the
// per-cell saturation statistics that the export still reads once the groups are gone.

#include <cstdint>
#include <random>
#include <vector>

#include "singlet/pileup/umi_dedup.h"
#include "test_harness.h"

using namespace singlet;

namespace {

void fill(DirectionalUmiStore& st) {
    std::mt19937_64 rng(7);
    for (int i = 0; i < 40000; ++i) {
        const uint32_t bc = static_cast<uint32_t>(rng() % 300), gene = static_cast<uint32_t>(rng() % 40);
        uint64_t umi = (rng() % 6) * 5 + (rng() % 4);            // clustered, so 1-mismatch neighbours occur
        if (rng() % 3 == 0) umi ^= (1u << (2 * (rng() % 12)));
        st.record_read(bc, gene, umi & 0xffffffu, gene, gene, static_cast<uint8_t>(1 + rng() % 3), 0);
    }
}

bool same(const DirectionalUmiStore::Molecule& a, const DirectionalUmiStore::Molecule& b) {
    return a.umi == b.umi && a.bc == b.bc && a.gene == b.gene && a.exon_idx == b.exon_idx &&
           a.intron_idx == b.intron_idx && a.count == b.count && a.flags == b.flags && a.diag == b.diag;
}

}  // namespace

int main() {
    for (bool directional : {true, false}) {
        DirectionalUmiStore a, b;
        fill(a);
        fill(b);
        std::vector<DirectionalUmiStore::Molecule> ma, mb;
        a.collapse_molecules(12, directional, ma);
        const auto sa = a.per_cell_molecule_stats(12);
        b.consume_molecules(12, directional, mb);
        const auto sb = b.per_cell_molecule_stats(12);
        bool eq = ma.size() == mb.size();
        for (size_t i = 0; eq && i < ma.size(); ++i) eq = same(ma[i], mb[i]);
        CHECK(eq && !ma.empty(), "consume_molecules gives collapse_molecules' molecules in order");
        CHECK(b.n_molecule_groups() == 0, "groups are gone after consume");
        CHECK(!b.molecules_empty(), "a consumed store still reports molecules (saturation reads the cached stats)");
        bool st_eq = sa.size() == sb.size();
        for (const auto& [bc, s] : sa) {
            auto it = sb.find(bc);
            if (it == sb.end() || it->second.total_reads != s.total_reads || it->second.unique_umis != s.unique_umis) st_eq = false;
        }
        CHECK(st_eq, "per-cell saturation stats survive consume");
    }
    DirectionalUmiStore empty;
    std::vector<DirectionalUmiStore::Molecule> m;
    empty.consume_molecules(12, true, m);
    CHECK(empty.molecules_empty() && m.empty(), "consuming an empty store keeps it empty");
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
