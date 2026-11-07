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

#include <cstring>
#if defined(__linux__)
#include <pthread.h>
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

}  // namespace singlet
