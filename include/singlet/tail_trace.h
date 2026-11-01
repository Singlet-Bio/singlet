// E16 (docs/gpu-engine/fable/E16-tail.md): wall-clock marks of the level-5 tail (SINGLET_ENGINE_GPU_E2E=5, --engine gpu, SINGLET_GPU_PILEUP=on),
// from the end of the prefill's mapping to the end of the GPU pileup. One stderr line per mark, "[e16] <epoch s.ms> <pid> <text>": STAR's child
// and the pipeline write to the same stderr, so the job's log orders both processes on one clock (CLOCK_REALTIME). Nothing is printed at other
// levels; SINGLET_E16_TRACE=0 turns the marks off at level 5 too. Plain C++ (STAR's sources, the pipeline and the CUDA files include it).
#pragma once
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <unistd.h>

namespace singlet_tail {
inline bool on() {
    static const bool v = [] {
        const char* e = std::getenv("SINGLET_ENGINE_GPU_E2E");
        const char* t = std::getenv("SINGLET_E16_TRACE");
        return e != nullptr && e[0] == '5' && !(t != nullptr && t[0] == '0');
    }();
    return v;
}
inline double now() { timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9; }
inline void mark(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
inline void mark(const char* fmt, ...) {
    if (!on()) return;
    char b[900];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "[e16] %.3f %d %s\n", now(), (int) getpid(), b);
}
// STAR's mapping threads (ReadAlignChunk::processChunks): per thread its start, first chunk read, last chunk mapped and end (epoch s), and
// its chunk and read counts; STAR.cpp prints them after the join
constexpr int kMaxThr = 256;
inline double g_thr[kMaxThr][4];
inline unsigned long long g_thrN[kMaxThr][2];
} // namespace singlet_tail
