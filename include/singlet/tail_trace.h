// E16 (docs/gpu-engine/fable/E16-tail.md): wall-clock marks of the level-5 tail (SINGLET_ENGINE_GPU_E2E=5, --engine gpu, SINGLET_GPU_PILEUP=on),
// from the end of the prefill's mapping to the end of the GPU pileup. One stderr line per mark, "[e16] <epoch s.ms> <pid> <text>": STAR's child
// and the pipeline write to the same stderr, so the job's log orders both processes on one clock (CLOCK_REALTIME). Nothing is printed at other
// levels; SINGLET_E16_TRACE=0 turns the marks off at level 5 too. Plain C++ (STAR's sources, the pipeline and the CUDA files include it).
// Also E16's switch (cuts()): the tail's changes are made at level 5 only, and SINGLET_E16=0 turns them off.
#pragma once
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// E16's cuts of the level-5 tail: on at level 5 unless SINGLET_E16=0 (then the tail runs as E11 left it; the gate's baseline switch)
inline bool cuts() {
    static const bool v = [] {
        const char* e = std::getenv("SINGLET_ENGINE_GPU_E2E");
        const char* c = std::getenv("SINGLET_E16");
        return e != nullptr && e[0] == '5' && !(c != nullptr && c[0] == '0');
    }();
    return v;
}
// one cut by name, for measuring them one at a time: SINGLET_E16_OFF="hdr,chunk,..." turns the named ones off (names in E16-tail.md)
inline bool cut(const char* name) {
    if (!cuts()) return false;
    const char* off = std::getenv("SINGLET_E16_OFF");
    if (off == nullptr) return true;
    const size_t n = std::strlen(name);
    for (const char* p = off; *p;) {
        const char* q = p; while (*q && *q != ',') q++;
        if ((size_t) (q - p) == n && std::strncmp(p, name, n) == 0) return false;
        p = *q ? q + 1 : q;
    }
    return true;
}
inline double now() { timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9; }
inline void mark(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
inline void mark(const char* fmt, ...) {
    if (!on()) return;
    char b[4096];
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
