#ifndef H_StarProfile
#define H_StarProfile
// Optional per-phase cycle counters for the mapping loop (build with -DSTAR_PROFILE); summed over
// threads and written to Log.out after mapping. Compiled out otherwise.
#ifdef STAR_PROFILE
#include <x86intrin.h>
#include <atomic>
enum {PROF_chunkRead, PROF_chunkMutex, PROF_chunkSort, PROF_readLoad, PROF_map, PROF_output, PROF_chunkWrite, PROF_N};
extern std::atomic<unsigned long long> g_prof[PROF_N];
#define PROF_T0(v) unsigned long long v = __rdtsc()
#define PROF_ADD(i, v) g_prof[i] += __rdtsc() - (v)
#else
#define PROF_T0(v)
#define PROF_ADD(i, v)
#endif
#endif
