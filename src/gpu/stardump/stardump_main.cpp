// stardump: the vendored STAR (prod/campaign) with ReadAlign_maxMappableLength2strands.cpp and ReadAlign_oneRead.cpp
// replaced by the dumping copies.
// Usage: STARDUMP_PREFIX=/tmp/x/seeds STARDUMP_M5=/tmp/x/m5 SINGLET_STAR_SEQCACHE_MB=0 stardump <STAR arguments...>
// (either prefix may be omitted; without both it behaves as STAR)
#include "dump_format.h"

namespace stardump_m5 { void closeAllExtern(); }
int star_main_impl(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    int rc = star_main_impl(argc, argv);
    stardump::closeAll();
    stardump_m5::closeAllExtern();
    return rc;
}
