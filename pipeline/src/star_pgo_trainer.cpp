// star_pgo_trainer.cpp — Standalone STAR training binary for PGO data collection.
//
// This binary calls star_main_impl() directly in the main process (no fork),
// avoiding the fork()+OpenMP+"-fprofile-generate" double-free crash that blocks
// PGO training on the integrated singlify binary.
//
// Usage:
//   star_pgo_trainer --genomeDir <dir> --readFilesIn <R2> <R1> \
//     --readFilesCommand zcat --soloType CB_UMI_Simple --soloCBwhitelist <wl> \
//     --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 \
//     --outSAMtype BAM Unsorted --outBAMcompression 0 \
//     --readMapNumber 5000000 --runThreadN 8

#include "star_api.h"
#include <cstdlib>

int main(int argc, char* argv[]) {
    return star_main_impl(argc, argv);
}
