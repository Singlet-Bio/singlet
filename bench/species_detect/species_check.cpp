// SPDX-License-Identifier: MIT
// bench/species_detect/species_check.cpp: runs species_detect::detect on .1fq files (with $SINGLET_SPECIES_FILTERS) and prints one line per file.
// human_reads / mouse_reads are the reads with a species-specific k-mer call; minor is the smaller share (mixture rule, MIX_* in species_detect.h).
#include <cstdio>
#include "singlet/pileup/species_detect.h"

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        auto r = species_detect::detect(argv[i], /*verbose=*/false, "");
        std::printf("%s\tspecies=%s\tgenome=%s\tconfidence=%.4f\tmethod=%s\tconfident=%d\thits=%d/%d\treads=%d\thuman_reads=%d\tmouse_reads=%d\tminor=%.4f\n",
                    argv[i], r.species.c_str(), r.genome_tag.c_str(), static_cast<double>(r.confidence), r.method.c_str(), r.confident() ? 1 : 0,
                    r.kmers_hit, r.kmers_tested, r.reads_sampled, r.reads_human, r.reads_mouse, static_cast<double>(r.minor_species_fraction()));
    }
    return 0;
}
