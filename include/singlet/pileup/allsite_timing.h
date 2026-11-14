// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: allsite_timing.h. SINGLET_ALLSITE_TIMING=1: one stderr line per sub-step of the all-site pass, "[allsite-step] <name>:
// wall X s, cpu Y s, threads N" (cpu = thread CPU time summed over the workers of the step). Off by default; it reads clocks only, so the
// outputs do not depend on it.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <time.h>

namespace singlet { namespace allsite { namespace astm {

inline bool on() {
    static const bool v = [] { const char* e = std::getenv("SINGLET_ALLSITE_TIMING"); return e && *e && *e != '0'; }();
    return v;
}
inline double thread_cpu() {
    timespec ts; if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) return 0;
    return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}
inline double wall() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// CPU seconds summed over threads (a worker adds its own thread's CPU time through Scope)
struct Acc {
    std::atomic<uint64_t> ns{0};
    void add(double s) { ns.fetch_add(static_cast<uint64_t>(s * 1e9)); }
    double sec() const { return 1e-9 * static_cast<double>(ns.load()); }
};
struct Scope {   // the CPU time of the current thread from construction to destruction goes to *a
    Acc* a; double c0;
    explicit Scope(Acc* acc) : a(on() ? acc : nullptr), c0(a ? thread_cpu() : 0) {}
    ~Scope() { if (a) a->add(thread_cpu() - c0); }
    Scope(const Scope&) = delete; Scope& operator=(const Scope&) = delete;
};
inline void line(const char* name, double wall_s, double cpu_s, unsigned threads) {
    if (!on()) return;
    char b[256]; std::snprintf(b, sizeof b, "[allsite-step] %s: wall %.3f s, cpu %.3f s, threads %u\n", name, wall_s, cpu_s, threads);
    std::cerr << b << std::flush;
}

struct Timed {   // prints the step's line when it goes out of scope (wall since construction, cpu summed in *acc)
    const char* name; double w0; Acc* acc; unsigned th;
    Timed(const char* n, Acc* a, unsigned t) : name(n), w0(on() ? wall() : 0), acc(a), th(t) {}
    ~Timed() { if (on()) line(name, wall() - w0, acc ? acc->sec() : 0, th); }
    Timed(const Timed&) = delete; Timed& operator=(const Timed&) = delete;
};

}}}  // namespace singlet::allsite::astm
