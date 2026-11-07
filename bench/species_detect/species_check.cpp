// SPDX-License-Identifier: MIT
// bench/species_detect/species_check.cpp: runs species_detect::detect on .1fq files (with $SINGLET_SPECIES_FILTERS) and prints one line per file.
#include <cstdio>
#include "singlet/pileup/species_detect.h"

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        auto r = species_detect::detect(argv[i], /*verbose=*/false, "");
        std::printf("%s\tspecies=%s\tgenome=%s\tconfidence=%.4f\tmethod=%s\tconfident=%d\thits=%d/%d\treads=%d\n", argv[i], r.species.c_str(), r.genome_tag.c_str(),
                    static_cast<double>(r.confidence), r.method.c_str(), r.confident() ? 1 : 0, r.kmers_hit, r.kmers_tested, r.reads_sampled);
    }
    return 0;
}
