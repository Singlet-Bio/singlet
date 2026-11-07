// star_api.h — Declaration for the bundled STAR aligner entry point.
//
// STAR source is vendored in src/star/ (MIT license, see src/star/LICENSE.STAR)
// and compiled as an OBJECT library into the singlet pipeline binary. The pipeline forks
// a child process and calls star_main_impl() directly rather than exec-ing an
// external binary.  Because STAR runs in a forked child, its internal exit()
// calls terminate only the child, leaving the parent pileup process unaffected.
//
// Usage in the singlet pipeline (child side of fork):
//   int rc = star_main_impl(args.size(), const_cast<char**>(args.data()));
//   _exit(rc);
#pragma once
extern "C++" int star_main_impl(int argc, char *argv[]);

// In-process read source (set in the forked child before star_main_impl()).
//
// When g_starReadSource is non-null, ReadAlignChunk::processChunks takes its reads from it instead of the
// readFilesIn streams (which are then opened but never read; pass /dev/null). fill() is called by one thread at a
// time (under STAR's mutexInRead) and appends whole reads to chunkIn[mate] in exactly the bytes STAR's FASTQ branch
// would have written for the same reads:
//     @<name> <iReadAll> N <readFilesIndex>\n<seq>\n+\n<qual>\n          for every mate
// It keeps going while both sizeBytes[] are below sizeLimit and iReadAll != readMapNumber (STAR's own loop condition,
// so chunk boundaries are the same as from a file), incrementing iReadAll per read, and returns with sizeBytes
// unchanged at the end of the reads.
#include <cstdint>
struct StarReadSource {
    virtual ~StarReadSource() = default;
    virtual void fill(char** chunkIn, uint64_t* sizeBytes, uint64_t sizeLimit, uint64_t& iReadAll,
                      uint64_t readMapNumber, uint32_t readFilesIndex) = 0;
};
extern StarReadSource* g_starReadSource;
