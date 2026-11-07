// SPDX-License-Identifier: MIT
// test_csc_add_signed: the uint16 cap of the pre-USA exon / intron split (export.h). csc_add_signed (sparse_accumulator.h) at and
// across 65535 for + and -, SparseAccumulator saturation, and a molecule-level fixture (DirectionalUmiStore::record_read, no BAM)
// whose mixed and unspliced counts pass 65535 in one (cell, intron): no value wraps; the one inexact case (a saturated intron
// entry) is asserted as documented at csc_add_signed.
#include "singlet/pileup/sparse_accumulator.h"
#include "singlet/pileup/umi_dedup.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {
int n_pass = 0, n_fail = 0;
void CHECK(bool ok, const std::string& what) {
    (ok ? n_pass : n_fail)++;
    std::cout << (ok ? "  PASS " : "  FAIL ") << what << "\n";
}
using Acc = singlet::SparseAccumulator<uint16_t>;
using CSC = Acc::CSCMatrix;

// one column, rows 0..n-1 of `vals` (0 = absent)
CSC col(const std::vector<int64_t>& vals) {
    Acc a; a.set_barcodes(std::vector<std::string>(1)); a.set_n_features(static_cast<uint32_t>(vals.size()));
    for (size_t r = 0; r < vals.size(); ++r)
        for (int64_t v = vals[r]; v > 0; v -= 60000) a.increment(static_cast<uint32_t>(r), 0, static_cast<uint16_t>(std::min<int64_t>(v, 60000)));
    return a.to_csc();
}
// value at (row, col 0), 0 if absent
int64_t at(const CSC& m, int32_t row) {
    for (int32_t k = m.indptr[0]; k < m.indptr[1]; ++k) if (m.indices[k] == row) return m.data[k];
    return 0;
}
bool no_zeros(const CSC& m) { for (auto v : m.data) if (v == 0) return false; return true; }

void test_accumulator() {
    std::cout << "=== SparseAccumulator saturation ===\n";
    Acc a; a.set_barcodes(std::vector<std::string>(1)); a.set_n_features(2);
    for (int i = 0; i < 70000; ++i) a.increment(0, 0);
    a.increment(1, 0, 65535); a.increment(1, 0, 65535);
    const CSC m = a.to_csc();
    CHECK(at(m, 0) == 65535, "70000 increments of 1 -> 65535");
    CHECK(at(m, 1) == 65535, "65535 + 65535 -> 65535 (no wrap)");
    Acc b; b.set_barcodes(std::vector<std::string>(1)); b.set_n_features(1); b.set_compact_limit(1000);
    for (int i = 0; i < 70000; ++i) b.increment(0, 0);
    CHECK(at(b.to_csc(), 0) == 65535, "with compaction: 70000 -> 65535");
}

void test_add_signed() {
    std::cout << "=== csc_add_signed ===\n";
    // rows:           0      1      2     3  4  5     6
    const CSC a = col({65534, 65535, 40000, 3, 7, 0,    9});
    const CSC b = col({1,     5,     30000, 5, 7, 11,   0});
    const CSC s = singlet::csc_add_signed(a, b, +1), d = singlet::csc_add_signed(a, b, -1);
    CHECK(at(s, 0) == 65535, "+: 65534 + 1 = 65535");
    CHECK(at(s, 1) == 65535, "+: 65535 + 5 = 65535 (clamped, no wrap)");
    CHECK(at(s, 2) == 65535, "+: 40000 + 30000 (70000 from two accumulators) = 65535");
    CHECK(at(s, 5) == 11 && at(s, 6) == 9, "+: an entry absent from one side is kept");
    CHECK(at(d, 0) == 65533 && at(d, 1) == 65530 && at(d, 2) == 10000, "-: 65534-1, 65535-5, 40000-30000");
    CHECK(at(d, 3) == 0, "-: 3 - 5 = 0 (clamped at 0)");
    CHECK(at(d, 4) == 0, "-: equal entries cancel");
    CHECK(at(d, 5) == 0 && at(d, 6) == 9, "-: absent in a -> 0, absent in b -> a");
    CHECK(no_zeros(s) && no_zeros(d), "zeros are dropped");
    CHECK(d.indices.size() == 4, "-: 4 stored entries (rows 0, 1, 2, 6)");
    // clamp(clamp(x) + clamp(y)) == clamp(x + y): the sum through two saturated accumulators equals one accumulator of x + y
    const CSC one = col({70000 + 70000});
    CHECK(at(singlet::csc_add_signed(col({70000}), col({70000}), +1), 0) == at(one, 0), "+: clamp(clamp(x) + clamp(y)) == clamp(x + y)");
}

// One cell, one gene, exon 0 / intron 0: n_mixed molecules with an exon-only read and an intron read (USA: unspliced; old labels:
// ambiguous, in the exon matrix), n_unspl molecules with intron reads only. Exact UMI collapse, distinct UMIs.
struct Split { int64_t exon, intron, amb, mixed_exon, mixed_intron, legacy_exon, legacy_intron; };
Split molecules(uint32_t n_mixed, uint32_t n_unspl) {
    using D = singlet::DirectionalUmiStore;
    D store;
    uint64_t umi = 0;
    for (uint32_t i = 0; i < n_mixed; ++i, ++umi) {
        store.record_read(0, 0, umi, 0, UINT32_MAX, D::kReadExonOnly | D::kReadTouchesExon);
        store.record_read(0, 0, umi, UINT32_MAX, 0, D::kReadIntron);
    }
    for (uint32_t i = 0; i < n_unspl; ++i, ++umi) store.record_read(0, 0, umi, UINT32_MAX, 0, D::kReadIntron);
    auto mk = [](uint32_t nf) { Acc a; a.set_barcodes(std::vector<std::string>(1)); a.set_n_features(nf); return a; };
    Acc ex = mk(1), in = mk(1), amb = mk(1), mx = mk(1), mi = mk(1);
    store.finalize_molecules(ex, in, [&](uint32_t bc, uint32_t g) { amb.increment(g, bc); }, 10, false, D::NoReads(),
                             [&](const D::Molecule& m) { mx.increment(m.exon_idx, m.bc); mi.increment(m.intron_idx, m.bc); });
    const CSC e = ex.to_csc(), i = in.to_csc(), a = amb.to_csc(), me = mx.to_csc(), mn = mi.to_csc();
    return {at(e, 0), at(i, 0), at(a, 0), at(me, 0), at(mn, 0),
            at(singlet::csc_add_signed(e, me, +1), 0), at(singlet::csc_add_signed(i, mn, -1), 0)};
}

void test_molecules() {
    std::cout << "=== molecule level ===\n";
    const Split u = molecules(30000, 2000);   // below the cap: exact
    CHECK(u.exon == 0 && u.intron == 32000 && u.mixed_exon == 30000 && u.mixed_intron == 30000, "below cap: USA intron 32000, mixed 30000");
    CHECK(u.legacy_exon == 30000 && u.legacy_intron == 2000, "below cap: legacy exon 30000, intron 2000 (the old labels, exact)");
    const Split s = molecules(70000, 2000);   // mixed 70000 and unspliced 72000 in one (cell, intron)
    CHECK(s.intron == 65535 && s.mixed_exon == 65535 && s.mixed_intron == 65535, "past cap: USA intron and mixed saturate at 65535, no wrap");
    CHECK(s.legacy_exon == 65535, "past cap: legacy exon = clamp(70000) = 65535 (exact: the old exon matrix holds the same)");
    CHECK(s.legacy_intron == 0, "past cap: legacy intron = 65535 - 65535 = 0 (the documented inexact case: old = 2000; undercount, no wrap)");
    const Split t = molecules(60000, 10000);  // unspliced 70000 saturates, mixed does not
    CHECK(t.intron == 65535 && t.mixed_intron == 60000 && t.legacy_intron == 5535,
          "intron saturated, mixed not: legacy intron 65535 - 60000 = 5535 (old = 10000; undercount, no wrap)");
    CHECK(t.legacy_exon == 60000, "intron saturated, mixed not: legacy exon exact (60000)");
}
}  // namespace

int main() {
    test_accumulator();
    test_add_signed();
    test_molecules();
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
