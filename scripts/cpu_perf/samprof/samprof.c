// LD_PRELOAD sampling profiler for the singlet binary (perf is blocked on Clipper: perf_event_paranoid=4).
// ITIMER_PROF fires every SAMPROF_MS of *process* CPU time, so the signal lands on whichever thread is running and samples are
// proportional to CPU time across all threads. The handler records the thread name and up to 32 return addresses; at exit they are
// written to $SAMPROF_OUT/samples.bin with /proc/self/maps as maps.txt. Resolve with resolve.py. Active only in processes whose
// executable name starts with "singlet", so /usr/bin/time, bash and sam-dump are left alone. No source change to singlet.
#define _GNU_SOURCE
#include <execinfo.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/time.h>
#include <unistd.h>

#define MAXF 32
typedef struct { char name[16]; uint32_t n; uint32_t pad; uint64_t pc[MAXF]; } sample_t;
static sample_t *ring; static size_t cap; static atomic_size_t nxt; static int active;

static void handler(int sig, siginfo_t *si, void *uc) {
    (void)sig; (void)si; (void)uc;
    size_t i = atomic_fetch_add(&nxt, 1);
    if (i >= cap) return;
    sample_t *s = &ring[i];
    prctl(PR_GET_NAME, (unsigned long)s->name, 0, 0, 0);
    void *fr[MAXF + 2];
    int n = backtrace(fr, MAXF + 2);
    int skip = n > 2 ? 2 : 0;               // handler + signal trampoline
    s->n = (uint32_t)(n - skip);
    for (int k = 0; k < n - skip; ++k) s->pc[k] = (uint64_t)fr[k + skip];
}

__attribute__((constructor)) static void start(void) {
    char exe[512]; ssize_t l = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (l <= 0) return;
    exe[l] = 0;
    const char *b = strrchr(exe, '/'); b = b ? b + 1 : exe;
    if (strncmp(b, "singlet", 7) != 0 || !getenv("SAMPROF_OUT")) return;
    cap = (size_t)4 << 20; ring = (sample_t *)calloc(cap, sizeof(sample_t));
    if (!ring) return;
    void *w[4]; backtrace(w, 4);              // load libgcc_s now, not inside the handler
    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = handler; sa.sa_flags = SA_SIGINFO | SA_RESTART; sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, 0);
    const char *ms = getenv("SAMPROF_MS"); int m = ms ? atoi(ms) : 5; if (m < 1) m = 5;
    struct itimerval it = {{m / 1000, (m % 1000) * 1000}, {m / 1000, (m % 1000) * 1000}};
    setitimer(ITIMER_PROF, &it, 0); active = 1;
}

__attribute__((destructor)) static void stop(void) {
    if (!active) return;
    struct itimerval z; memset(&z, 0, sizeof z); setitimer(ITIMER_PROF, &z, 0); active = 0;
    size_t n = atomic_load(&nxt); if (n > cap) n = cap;
    const char *dir = getenv("SAMPROF_OUT"); char p[1024];
    snprintf(p, sizeof p, "%s/samples.bin", dir);
    FILE *f = fopen(p, "wb");
    if (f) { fwrite(&n, sizeof n, 1, f); fwrite(ring, sizeof(sample_t), n, f); fclose(f); }
    snprintf(p, sizeof p, "%s/maps.txt", dir);
    f = fopen(p, "w");
    FILE *m = fopen("/proc/self/maps", "r");
    if (f && m) { char ln[1024]; while (fgets(ln, sizeof ln, m)) fputs(ln, f); }
    if (f) fclose(f);
    if (m) fclose(m);
    fprintf(stderr, "[samprof] %zu samples\n", n);
}
