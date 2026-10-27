#ifndef H_StarProfile
#define H_StarProfile
// Optional per-phase cycle counters for the mapping loop (build with -DSTAR_PROFILE); summed over
// threads and written to Log.out after mapping. Compiled out otherwise.
#ifdef STAR_PROFILE
#include <x86intrin.h>
#include <atomic>
enum {PROF_chunkRead, PROF_chunkMutex, PROF_chunkSort, PROF_readLoad, PROF_map, PROF_output, PROF_chunkWrite,
      PROF_seed, PROF_windows, PROF_stitch, PROF_N};
extern std::atomic<unsigned long long> g_prof[PROF_N];
//mapping cycles and read counts by outcome: 0 unique, 1 multi, 2 too many loci, 3 too short, 4 other unmapped, 5 cache hit
extern std::atomic<unsigned long long> g_profOutcome[6][2];
//unique STAR-mapped reads by shape: {perfect, 1 mismatch, other} x {unspliced, spliced} for one-gene concordant,
//annotated-junction, indel-free, non-chrM alignments; 6 = everything else unique
extern std::atomic<unsigned long long> g_profClass[7][2];
//per class: seed, windows, stitch cycles; MMP queries, SA binary searches
extern std::atomic<unsigned long long> g_profClassDetail[7][5];
//per outcome: seed, windows, stitch cycles; MMP queries, SA searches, windows before stitching, (unused),
//window create / flank / assign cycles, SA loci of anchors / of all seeds
extern std::atomic<unsigned long long> g_profOutcomeDetail[6][12];
#define PROF_T0(v) unsigned long long v = __rdtsc()
#define PROF_ADD(i, v) g_prof[i] += __rdtsc() - (v)
#else
#define PROF_T0(v)
#define PROF_ADD(i, v)
#endif
#endif
