// SPDX-License-Identifier: MIT
// tests/cpp/test_umi_staging.cpp
//
// DirectionalUmiStore staging (set_staging(), the serial stream loop): record_read() fills a flat table and seal() moves it
// into the groups. The sealed store must be the store record_read() builds without staging: the same groups with the same
// merged entries, in the same iteration order, so collapse_molecules() gives the same molecules in the same order. Checked
// with an explicit seal, a lazy one (the first reader seals), and a seal in the middle of the reads.

#include <cstdint>
#include <random>
#include <vector>

#include "singlet/pileup/umi_dedup.h"
#include "test_harness.h"

using namespace singlet;

namespace {

// Reads [from, to) of one fixed random sequence: clustered UMIs, varying exon/intron indices, flags and diagnostic bits.
void fill(DirectionalUmiStore& st, int from, int to, uint32_t n_bc, uint32_t n_gene) {
    std::mt19937_64 rng(11);
    for (int i = 0; i < to; ++i) {
        const uint32_t bc = static_cast<uint32_t>(rng() % n_bc), gene = static_cast<uint32_t>(rng() % n_gene);
        uint64_t umi = (rng() % 6) * 5 + (rng() % 4);
        if (rng() % 3 == 0) umi ^= (1u << (2 * (rng() % 12)));
        const uint32_t exon = static_cast<uint32_t>(gene * 10 + rng() % 7);
        const uint32_t intron = (rng() % 4 == 0) ? UINT32_MAX : static_cast<uint32_t>(gene * 10 + rng() % 5);
        const uint8_t flags = static_cast<uint8_t>(1 + rng() % 7), diag = static_cast<uint8_t>(rng() % 64);
        if (i >= from) st.record_read(bc, gene, umi & 0xffffffu, exon, intron, flags, diag);
    }
}

bool same(const DirectionalUmiStore::Molecule& a, const DirectionalUmiStore::Molecule& b) {
    return a.umi == b.umi && a.bc == b.bc && a.gene == b.gene && a.exon_idx == b.exon_idx &&
           a.intron_idx == b.intron_idx && a.count == b.count && a.flags == b.flags && a.diag == b.diag;
}

bool same_store(const DirectionalUmiStore& a, const DirectionalUmiStore& b, bool directional) {
    std::vector<DirectionalUmiStore::Molecule> ma, mb;
    a.collapse_molecules(12, directional, ma);
    b.collapse_molecules(12, directional, mb);
    if (ma.empty() || ma.size() != mb.size() || a.n_molecule_groups() != b.n_molecule_groups()) return false;
    for (size_t i = 0; i < ma.size(); ++i)
        if (!same(ma[i], mb[i])) return false;
    std::vector<uint64_t> ka, kb;
    a.append_umi_keys(ka);
    b.append_umi_keys(kb);
    return ka == kb;   // unsorted: the groups' iteration order and each group's entry order
}

}  // namespace

int main() {
    for (bool directional : {true, false}) {
        for (int big = 0; big < 2; ++big) {
            const int reads = big ? 600000 : 40000;
            const uint32_t n_bc = big ? 4000 : 300, n_gene = big ? 60 : 40;
            DirectionalUmiStore plain, sealed, lazy, mid;
            fill(plain, 0, reads, n_bc, n_gene);

            sealed.set_staging(true);
            fill(sealed, 0, reads, n_bc, n_gene);
            sealed.set_staging(false);
            CHECK(same_store(plain, sealed, directional), "staged then sealed: the plain store, molecules and order included");

            lazy.set_staging(true);
            fill(lazy, 0, reads, n_bc, n_gene);
            CHECK(same_store(plain, lazy, directional), "staged, sealed by the first reader: the plain store");

            mid.set_staging(true);
            fill(mid, 0, reads / 2, n_bc, n_gene);
            CHECK(mid.n_molecule_groups() > 0, "a reader in the middle seals what is staged so far");
            fill(mid, reads / 2, reads, n_bc, n_gene);
            mid.set_staging(false);
            CHECK(same_store(plain, mid, directional), "sealed twice, merging into groups that had entries: the plain store");

            const auto sp = plain.per_cell_molecule_stats(12), sl = lazy.per_cell_molecule_stats(12);
            bool st_eq = sp.size() == sl.size();
            for (const auto& [bc, s] : sp) {
                auto it = sl.find(bc);
                if (it == sl.end() || it->second.total_reads != s.total_reads || it->second.unique_umis != s.unique_umis) st_eq = false;
            }
            CHECK(st_eq, "per-cell saturation stats of a staged store equal the plain ones");
        }
    }
    DirectionalUmiStore empty;
    empty.set_staging(true);
    empty.set_staging(false);
    CHECK(empty.molecules_empty() && empty.n_molecule_groups() == 0, "staging nothing leaves the store empty");
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
