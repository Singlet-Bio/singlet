// SPDX-License-Identifier: MIT
// role_cpu.cpp: SINGLET_ROLE_CPU=1 accounts each thread's CPU time to its name (its role, include/singlet/thread_name.h) when it exits, and
// prints one stderr line per process at exit: the exited threads by name, the threads still alive by name (from /proc/self/task), and the
// process total from getrusage for comparison. A 1 s /proc sampler misses short-lived threads (about 38% of a run's CPU); this sees them all.
// It works by defining pthread_create in the executable, which the dynamic linker binds for every library (libstdc++'s std::thread, libgomp,
// the CUDA runtime); the definition forwards to glibc's. Off (the default) it is one extra call per thread start. It changes no output.
#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dirent.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {
using create_fn = int (*)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
create_fn real_create() {
    static create_fn f = (create_fn) dlsym(RTLD_NEXT, "pthread_create");
    return f;
}
bool role_on() {
    static const bool on = [] { const char* v = std::getenv("SINGLET_ROLE_CPU"); return v != nullptr && v[0] == '1'; }();
    return on;
}
struct Role { char name[16]; double cpu; long n; };
constexpr int kRoles = 96;
Role g_roles[kRoles];
int g_nroles = 0;
std::mutex g_mu;
void add_role(const char* name, double cpu, long n) {
    std::lock_guard<std::mutex> g(g_mu);
    for (int i = 0; i < g_nroles; i++)
        if (std::strncmp(g_roles[i].name, name, 16) == 0) { g_roles[i].cpu += cpu; g_roles[i].n += n; return; }
    if (g_nroles < kRoles) { std::strncpy(g_roles[g_nroles].name, name, 15); g_roles[g_nroles].name[15] = 0; g_roles[g_nroles].cpu = cpu; g_roles[g_nroles].n = n; g_nroles++; }
}
double thread_cpu() { timespec ts{}; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
struct ExitRecorder {
    bool armed = false;
    ~ExitRecorder() {
        if (!armed) return;
        char nm[16] = {0};
        if (pthread_getname_np(pthread_self(), nm, sizeof nm) != 0) std::strcpy(nm, "?");
        add_role(nm, thread_cpu(), 1);
    }
};
thread_local ExitRecorder t_rec;
struct Start { void* (*fn)(void*); void* arg; };
void* trampoline(void* p) {
    Start s = *(Start*) p;
    std::free(p);
    t_rec.armed = true;   // its destructor runs when the thread returns or calls pthread_exit
    return s.fn(s.arg);
}
void report() {
    std::lock_guard<std::mutex> g(g_mu);
    const double tick = (double) sysconf(_SC_CLK_TCK);
    // threads alive now (the OpenMP pool, the main thread): utime + stime from /proc/self/task/<tid>/stat, by comm
    Role live[kRoles]; int nl = 0;
    if (DIR* d = opendir("/proc/self/task")) {
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            char path[96], buf[1024];
            std::snprintf(path, sizeof path, "/proc/self/task/%s/stat", e->d_name);
            FILE* f = std::fopen(path, "r");
            if (!f) continue;
            const size_t n = std::fread(buf, 1, sizeof buf - 1, f); std::fclose(f); buf[n] = 0;
            char* l = std::strchr(buf, '('); char* r = std::strrchr(buf, ')');
            if (!l || !r || r < l) continue;
            char nm[16] = {0}; std::strncpy(nm, l + 1, (size_t) std::min<long>(15, r - l - 1));
            unsigned long ut = 0, st = 0;
            // fields after ")": state(3) ... utime(14) stime(15)
            char* p = r + 2; int field = 3;
            while (*p && field < 14) { if (*p == ' ') field++; p++; }
            if (std::sscanf(p, "%lu %lu", &ut, &st) != 2) continue;
            const double c = (ut + st) / tick;
            int i = 0; for (; i < nl; i++) if (std::strncmp(live[i].name, nm, 16) == 0) break;
            if (i == nl && nl < kRoles) { std::memcpy(live[nl].name, nm, 16); live[nl].cpu = 0; live[nl].n = 0; nl++; }
            if (i < kRoles) { live[i].cpu += c; live[i].n++; }
        }
        closedir(d);
    }
    rusage ru{}; getrusage(RUSAGE_SELF, &ru);
    const double tot = ru.ru_utime.tv_sec + ru.ru_stime.tv_sec + (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1e-6;
    double sum = 0;
    for (int i = 0; i < g_nroles; i++) sum += g_roles[i].cpu;
    for (int i = 0; i < nl; i++) sum += live[i].cpu;
    std::fprintf(stderr, "[role-cpu] pid %d process %.2f s, accounted %.2f s; exited:", (int) getpid(), tot, sum);
    for (int i = 0; i < g_nroles; i++) std::fprintf(stderr, " %s=%.2f/%ld", g_roles[i].name, g_roles[i].cpu, g_roles[i].n);
    std::fprintf(stderr, "; live:");
    for (int i = 0; i < nl; i++) std::fprintf(stderr, " %s=%.2f/%ld", live[i].name, live[i].cpu, live[i].n);
    std::fprintf(stderr, "\n");
}
void child_reset() { g_nroles = 0; new (&g_mu) std::mutex(); }   // a forked child starts its own table (STAR's child, the pool's runs)
struct Init {
    Init() {
        if (!role_on()) return;
        pthread_atfork(nullptr, nullptr, child_reset);
        std::atexit(report);
    }
} g_init;
}  // namespace

extern "C" int pthread_create(pthread_t* t, const pthread_attr_t* a, void* (*fn)(void*), void* arg) noexcept {
    create_fn real = real_create();
    if (!role_on()) return real(t, a, fn, arg);
    Start* s = (Start*) std::malloc(sizeof(Start));
    if (!s) return real(t, a, fn, arg);
    s->fn = fn; s->arg = arg;
    const int rc = real(t, a, trampoline, s);
    if (rc != 0) std::free(s);
    return rc;
}
#endif
