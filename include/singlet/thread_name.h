// SPDX-License-Identifier: MIT
// thread_name.h — names for the threads of a run, so per-thread CPU time can be read by role (top -H, /proc/<pid>/task/*/comm,
// bench/gpu/profile/threadsampler.py; perf is unavailable on some clusters). Names change no output.
//
// On Linux a new thread starts with its creator's name, so naming the thread that starts a phase also names every pool that phase creates:
//   singlet::set_thread_name("sg-demux");          the calling thread, until renamed (at most 15 characters)
//   singlet::ThreadNameScope s("gpu-hostlane");    the calling thread for the rest of the scope, then its previous name
//   singlet::name_omp_pool("sg-nonhost");          also the threads of OpenMP's default pool, which outlive the phase that created them.
//                                                  Call it only after STAR's child is forked: libgomp's pool does not survive a fork.
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <utility>
#if defined(__linux__)
#include <pthread.h>
#endif
#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace singlet {

inline void set_thread_name(const char* name) {
#if defined(__linux__)
    char b[16];
    std::strncpy(b, name, 15);
    b[15] = 0;
    pthread_setname_np(pthread_self(), b);
#else
    (void) name;
#endif
}

struct ThreadNameScope {
    char prev[16] = {0};
    explicit ThreadNameScope(const char* name) {
#if defined(__linux__)
        if (pthread_getname_np(pthread_self(), prev, sizeof prev) != 0) prev[0] = 0;
#endif
        set_thread_name(name);
    }
    ~ThreadNameScope() { if (prev[0]) set_thread_name(prev); }
    ThreadNameScope(const ThreadNameScope&) = delete;
    ThreadNameScope& operator=(const ThreadNameScope&) = delete;
};

inline void name_omp_pool(const char* name) {
    set_thread_name(name);
#if defined(_OPENMP)
#pragma omp parallel
    set_thread_name(name);
#endif
}

// SINGLET_EXPORT_TIMING=1: one stderr line per timed job (wall and the job thread's own CPU; threads the job starts are not
// counted), so the export phase's writers can be costed one by one. Off by default; it changes no output.
inline bool export_timing_on() {
    static const bool on = [] { const char* v = std::getenv("SINGLET_EXPORT_TIMING"); return v && v[0] == '1'; }();
    return on;
}
inline double thread_cpu_s() {
#if defined(__linux__) || defined(__APPLE__)
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#else
    return 0.0;
#endif
}
struct ExportTimer {
    const char* file; int line; bool on; std::chrono::steady_clock::time_point w0; double c0 = 0.0;
    ExportTimer(const char* f, int l) : file(f), line(l), on(export_timing_on()) {
        if (on) { w0 = std::chrono::steady_clock::now(); c0 = thread_cpu_s(); }
    }
    ~ExportTimer() {
        if (!on) return;
        const double w = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
        std::fprintf(stderr, "[export-timing] %s:%d wall %.2f s, thread cpu %.2f s\n", file, line, w, thread_cpu_s() - c0);
    }
    ExportTimer(const ExportTimer&) = delete;
    ExportTimer& operator=(const ExportTimer&) = delete;
};
template <class F>
auto timed_job(const char* file, int line, F f) {
    return [file, line, f = std::move(f)]() mutable { ExportTimer t(file, line); f(); };
}

// The export phases under SINGLET_EXPORT_TIMING=1: wall and process CPU (all threads) from one next() to the following one.
inline double process_cpu_s() {
#if defined(__linux__) || defined(__APPLE__)
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_stime.tv_sec + (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1e-6;
#else
    return 0.0;
#endif
}
struct PhaseClock {
    const char* cur = nullptr; std::chrono::steady_clock::time_point w0; double c0 = 0.0;
    void flush() {
        if (!cur) return;
        const double w = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
        std::fprintf(stderr, "[export-timing] phase \"%s\": wall %.2f s, process cpu %.2f s\n", cur, w, process_cpu_s() - c0);
        cur = nullptr;
    }
    void next(const char* name) {
        if (!export_timing_on()) return;
        flush();
        cur = name; w0 = std::chrono::steady_clock::now(); c0 = process_cpu_s();
    }
    ~PhaseClock() { flush(); }
};

}  // namespace singlet

// write_threads.emplace_back(lambda) -> SINGLET_TIMED(write_threads, lambda): the same thread, timed under SINGLET_EXPORT_TIMING=1
#define SINGLET_TIMED(vec, ...) (vec).emplace_back(::singlet::timed_job("export.h", __LINE__, __VA_ARGS__))
