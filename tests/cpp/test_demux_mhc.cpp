// SPDX-License-Identifier: MIT
// snp_in_mhc: the extended MHC of GRCh38 is left out of the auto-K search.
#include <cstdio>
#include "singlet/pileup/donor_demux.h"

using namespace singlet;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    CHECK(snp_in_mhc("chr6:32000000:A>G"));
    CHECK(snp_in_mhc("chr6:28510120:C>T"));
    CHECK(snp_in_mhc("chr6:33480577:C>T"));
    CHECK(!snp_in_mhc("chr6:28510119:C>T"));
    CHECK(!snp_in_mhc("chr6:33480578:C>T"));
    CHECK(!snp_in_mhc("chr1:32000000:A>G"));
    CHECK(!snp_in_mhc("chr16:32000000:A>G"));
    CHECK(snp_in_mhc("GRCh38___chr6:30000000:G>A"));          // barnyard panel
    CHECK(!snp_in_mhc("GRCm39___chr6:30000000:G>A"));         // mouse chr6 is not the MHC
    CHECK(!snp_in_mhc(""));
    CHECK(!snp_in_mhc("chr6"));
    std::printf(fails ? "test_demux_mhc: FAILED\n" : "test_demux_mhc: ok\n");
    return fails ? 1 : 0;
}
