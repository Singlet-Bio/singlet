// SPDX-License-Identifier: MIT
// role_cpu.cpp: SINGLET_ROLE_CPU=1 accounts each thread's CPU time to its name (its role, include/singlet/thread_name.h) when it exits, and
// prints one stderr line per process at exit: the exited threads by name, the threads still alive by name (from /proc/self/task), and the
// process total from getrusage for comparison. A 1 s /proc sampler misses short-lived threads (about 38% of a run's CPU); this sees them all.
// It works by defining pthread_create in the executable, which the dynamic linker binds for every library (libstdc++'s std::thread, libgomp,
// the CUDA runtime); the definition forwards to glibc's. Off (the default) it is one extra call per thread start. It changes no output.
// pthread_setname_np is defined the same way: when a thread renames itself (a phase of the main thread, say), the CPU it used since its
// last rename is charged to the old name, so a long-lived thread's time splits by the phases it ran instead of landing on its last name.
#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dirent.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/syscall.h>
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
double thread_cpu() { timespec ts{}; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
// CPU already charged at renames, per thread: the exit recorder and the live scan count only what came after
thread_local double t_mark = 0;
struct Charged { long tid; double cpu; };
constexpr int kCharged = 4096;
Charged g_charged[kCharged];
int g_ncharged = 0;
long my_tid() { return (long) syscall(SYS_gettid); }
void note_charged(long tid, double c) {   // under g_mu
    for (int i = 0; i < g_ncharged; i++) if (g_charged[i].tid == tid) { g_charged[i].cpu += c; return; }
    if (g_ncharged < kCharged) g_charged[g_ncharged++] = Charged{tid, c};
}
double charged_of(long tid) { for (int i = 0; i < g_ncharged; i++) if (g_charged[i].tid == tid) return g_charged[i].cpu; return 0; }
void forget_charged(long tid) { for (int i = 0; i < g_ncharged; i++) if (g_charged[i].tid == tid) { g_charged[i] = g_charged[--g_ncharged]; return; } }
void add_role_locked(const char* name, double cpu, long n) {
    for (int i = 0; i < g_nroles; i++)
        if (std::strncmp(g_roles[i].name, name, 16) == 0) { g_roles[i].cpu += cpu; g_roles[i].n += n; return; }
    if (g_nroles < kRoles) { std::strncpy(g_roles[g_nroles].name, name, 15); g_roles[g_nroles].name[15] = 0; g_roles[g_nroles].cpu = cpu; g_roles[g_nroles].n = n; g_nroles++; }
}
// the calling thread's CPU since its last mark, to its current name (n: 1 at its exit, 0 at a rename)
void charge_self(long n) {
    char nm[16] = {0};
    if (pthread_getname_np(pthread_self(), nm, sizeof nm) != 0) std::strcpy(nm, "?");
    const double c = thread_cpu(), d = c - t_mark;
    t_mark = c;
    std::lock_guard<std::mutex> g(g_mu);
    add_role_locked(nm, d > 0 ? d : 0, n);
    if (n) forget_charged(my_tid()); else note_charged(my_tid(), d > 0 ? d : 0);
}
struct ExitRecorder {
    bool armed = false;
    ~ExitRecorder() {
        if (!armed) return;
        charge_self(1);
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
    charge_self(0);   // the reporting thread (the main thread, STAR child's main) up to now, by its current name; skipped in the scan below
    const long self = my_tid();
    std::lock_guard<std::mutex> g(g_mu);
    const double tick = (double) sysconf(_SC_CLK_TCK);
    // threads alive now (the OpenMP pool, the main thread): utime + stime from /proc/self/task/<tid>/stat, by comm
    Role live[kRoles]; int nl = 0;
    if (DIR* d = opendir("/proc/self/task")) {
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            const long tid = std::atol(e->d_name);
            if (tid == self) continue;
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
            const double c = std::max(0.0, (ut + st) / tick - charged_of(tid));   // less what its renames charged already
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
void child_reset() { g_nroles = 0; g_ncharged = 0; t_mark = 0; new (&g_mu) std::mutex(); }   // a forked child starts its own table (STAR's child, the pool's runs)
struct Init {
    Init() {
        if (!role_on()) return;
        pthread_atfork(nullptr, nullptr, child_reset);
        std::atexit(report);
    }
} g_init;
}  // namespace

// a process that ends with _exit (STAR's child) skips atexit: it calls this first. Nothing without SINGLET_ROLE_CPU=1.
extern "C" void singlet_role_cpu_report() { if (role_on()) report(); }

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

using setname_fn = int (*)(pthread_t, const char*);
extern "C" int pthread_setname_np(pthread_t t, const char* name) noexcept {
    static setname_fn real = (setname_fn) dlsym(RTLD_NEXT, "pthread_setname_np");
    if (role_on() && pthread_equal(t, pthread_self())) charge_self(0);
    return real(t, name);
}
#else
extern "C" void singlet_role_cpu_report() {}
#endif
