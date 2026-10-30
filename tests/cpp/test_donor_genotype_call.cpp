// R12 genotype call for the donor VCF (call_donor_genotype): fixed-error binomial posterior on aggregated (ad, dp).
// Unit test of the rule's anchor cases; the Kang evidence is in bench/snp_falsealt/README.md.

#include <cstring>
#include <iostream>

#include "singlet/pileup/donor_demux.h"

using singlet::call_donor_genotype;

static int n_fail = 0;
#define EXPECT(ad, dp, want) \
    do { \
        const char* got = call_donor_genotype((ad), (dp)); \
        if (std::strcmp(got, (want)) != 0) { \
            std::cout << "FAIL ad=" << (ad) << " dp=" << (dp) << " got " << got << " want " << (want) << "\n"; \
            ++n_fail; \
        } \
    } while (0)

int main() {
    EXPECT(0, 0, "./.");     // no data
    EXPECT(0, 10, "0/0");
    EXPECT(1, 10, "0/0");    // one alt molecule in ten: the old 0.1 cut called this 0/1
    EXPECT(1, 20, "0/0");
    EXPECT(2, 10, "0/1");    // two of ten is already 0/1 under err 0.01
    EXPECT(5, 10, "0/1");
    EXPECT(10, 10, "1/1");
    EXPECT(9, 10, "1/1");    // symmetric: one ref molecule in ten
    EXPECT(1, 1, "./.");     // too little to decide
    EXPECT(0, 3, "./.");
    EXPECT(0, 5, "0/0");
    EXPECT(12, 10, "1/1");   // ad > dp is clamped to dp
    if (n_fail == 0) std::cout << "all passed\n";
    return n_fail ? 1 : 0;
}
