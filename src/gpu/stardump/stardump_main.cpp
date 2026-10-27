// stardump: the vendored STAR (prod/campaign) with ReadAlign_maxMappableLength2strands.cpp replaced by the dumping copy.
// Usage: STARDUMP_PREFIX=/tmp/x/seeds SINGLET_STAR_SEQCACHE_MB=0 stardump <STAR arguments...>
#include "dump_format.h"

int star_main_impl(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    int rc = star_main_impl(argc, argv);
    stardump::closeAll();
    return rc;
}
