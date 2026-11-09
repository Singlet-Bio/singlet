// Storage rule and file format of the all-site member (allsite_store.h): the entries a cell keeps against its donor's genotype, and the
// binary sparse file with one byte range per cell.

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "singlet/pileup/allsite_store.h"

namespace A = singlet::allsite;
static int n_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cout << "FAIL " << msg << "\n"; ++n_fail; } } while (0)

int main() {
    // keep_entry(g, donors_differ, ad, dp): homozygous donor keeps only the other allele, heterozygous (or uncalled) keeps all coverage,
    // sites where the donors differ keep every covered cell, a cell without a donor is read against the reference
    CHECK(!A::keep_entry(0, false, 0, 7), "hom-ref donor, concordant coverage dropped");
    CHECK(A::keep_entry(0, false, 1, 7), "hom-ref donor, ALT molecule kept");
    CHECK(!A::keep_entry(2, false, 7, 7), "hom-alt donor, concordant coverage dropped");
    CHECK(A::keep_entry(2, false, 6, 7), "hom-alt donor, REF molecule kept");
    CHECK(A::keep_entry(1, false, 0, 3) && A::keep_entry(1, false, 3, 3), "het donor keeps every covered site");
    CHECK(A::keep_entry(-1, false, 0, 1), "no donor call keeps coverage");
    CHECK(A::keep_entry(0, true, 0, 5) && A::keep_entry(2, true, 5, 5), "sites where the donors differ keep every covered cell");
    CHECK(!A::keep_entry(A::kRef, false, 0, 9) && A::keep_entry(A::kRef, false, 2, 9), "reference cell keeps ALT only");
    CHECK(!A::keep_entry(1, true, 0, 0), "no coverage, no entry");

    // file format: 4 cells, one empty, row gaps beyond 2^21, counts up to 65535
    A::DeltaStore s; s.n_sites = 3000000; s.K = 2;
    s.off = {0, 3, 3, 5, 6};
    s.row = {0, 5, 2999999, 17, 1000000, 42};
    s.ad = {0, 65535, 3, 1, 0, 200};
    s.dp = {1, 65535, 9, 1, 65535, 300};
    const std::string p = "tmp_allele_delta.bin";
    const uint64_t bytes = A::write_store(p, s);
    A::DeltaStore b;
    CHECK(bytes > 64 && A::read_store(p, b), "written and read");
    CHECK(b.off == s.off && b.row == s.row && b.ad == s.ad && b.dp == s.dp && b.n_sites == s.n_sites && b.K == 2, "round trip equals the input");
    std::remove(p.c_str());
    std::cout << (n_fail ? "FAILED" : "all passed") << "\n";
    return n_fail ? 1 : 0;
}
