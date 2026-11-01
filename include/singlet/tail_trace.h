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
#include <dirent.h>
#include <string>
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
// the processes holding the write end of the pipe with inode ino (/proc/<pid>/fd links "pipe:[ino]" whose fdinfo flags say O_WRONLY):
// " pid (comm) fd" per holder, " none" if no process does (E16: STAR's BAM pipe after STAR closed it)
inline std::string pipeWriters(unsigned long ino) {
    std::string r; char want[64]; std::snprintf(want, sizeof want, "pipe:[%lu]", ino);
    DIR* d = opendir("/proc"); if (d == nullptr) return " ?";
    while (dirent* e = readdir(d)) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        char p[64]; std::snprintf(p, sizeof p, "/proc/%.20s/fd", e->d_name);
        DIR* f = opendir(p); if (f == nullptr) continue;
        while (dirent* x = readdir(f)) {
            if (x->d_name[0] == '.') continue;
            char l[128], t[128]; std::snprintf(l, sizeof l, "%s/%.20s", p, x->d_name);
            const ssize_t n = readlink(l, t, sizeof t - 1); if (n <= 0) continue; t[n] = 0;
            if (std::strcmp(t, want) != 0) continue;
            char fi[128]; std::snprintf(fi, sizeof fi, "/proc/%.20s/fdinfo/%.20s", e->d_name, x->d_name);
            unsigned flags = 0; char line[256];
            if (FILE* g = std::fopen(fi, "r")) { while (std::fgets(line, sizeof line, g)) if (std::sscanf(line, "flags: %o", &flags) == 1) break; std::fclose(g); }
            if ((flags & 3) != 1) continue;   // the read end
            char cp[64], comm[64] = "?"; std::snprintf(cp, sizeof cp, "/proc/%.20s/comm", e->d_name);
            if (FILE* g = std::fopen(cp, "r")) { if (std::fgets(comm, sizeof comm, g)) comm[std::strcspn(comm, "\n")] = 0; std::fclose(g); }
            r += std::string(" ") + e->d_name + " (" + comm + ") fd " + x->d_name;
        }
        closedir(f);
    }
    closedir(d);
    return r.empty() ? std::string(" none") : r;
}
// STAR's mapping threads (ReadAlignChunk::processChunks): per thread its start, first chunk read, last chunk mapped, end, chunk loop left and
// BAM buffer flushed (epoch s), and its chunk and read counts; STAR.cpp prints them after the join
constexpr int kMaxThr = 256;
inline double g_thr[kMaxThr][6];
inline unsigned long long g_thrN[kMaxThr][2];
} // namespace singlet_tail
