// LD_PRELOAD sampling profiler for the singlet binary (perf is blocked on Clipper: perf_event_paranoid=4).
// Every thread gets its own CLOCK_THREAD_CPUTIME_ID timer (armed at thread start through a pthread_create wrapper) that raises SIGPROF on
// that thread every SAMPROF_MS of its own CPU time, so each sample is exactly SAMPROF_MS of CPU and all threads are covered. (A single
// process-wide ITIMER_PROF delivers to the main thread only and misses the workers.) The handler records the thread name and up to 32 return addresses; at exit they are
// written to $SAMPROF_OUT/samples.bin with /proc/self/maps as maps.txt. Resolve with resolve.py. Active only in processes whose
// executable name starts with "singlet", so /usr/bin/time, bash and sam-dump are left alone. No source change to singlet.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
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
static sample_t *ring; static size_t cap; static atomic_size_t nxt; static int active; static int period_ms = 10;

static __thread uintptr_t stk_hi;
static void arm_thread(void) {
    { pthread_attr_t at; void *addr; size_t sz; if (pthread_getattr_np(pthread_self(), &at) == 0) { if (pthread_attr_getstack(&at, &addr, &sz) == 0) stk_hi = (uintptr_t)addr + sz; pthread_attr_destroy(&at); } }
    struct sigevent sev; memset(&sev, 0, sizeof sev);
    sev.sigev_notify = SIGEV_THREAD_ID; sev.sigev_signo = SIGPROF; sev._sigev_un._tid = (int)syscall(SYS_gettid);
    timer_t t;
    if (timer_create(CLOCK_THREAD_CPUTIME_ID, &sev, &t) != 0) return;
    struct itimerspec its; its.it_value.tv_sec = its.it_interval.tv_sec = period_ms / 1000;
    its.it_value.tv_nsec = its.it_interval.tv_nsec = (long)(period_ms % 1000) * 1000000L;
    timer_settime(t, 0, &its, 0);
}

typedef struct { void *(*fn)(void *); void *arg; } tramp_t;
static void *tramp(void *p) { tramp_t t = *(tramp_t *)p; free(p); arm_thread(); return t.fn(t.arg); }
int pthread_create(pthread_t *th, const pthread_attr_t *a, void *(*fn)(void *), void *arg) {
    static int (*real)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
    if (!real) real = (int (*)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *))dlsym(RTLD_NEXT, "pthread_create");
    if (!active) return real(th, a, fn, arg);
    tramp_t *t = (tramp_t *)malloc(sizeof *t); t->fn = fn; t->arg = arg;
    return real(th, a, tramp, t);
}

// Frame-pointer walk (no libgcc unwinder: it takes loader locks and crashed STAR's workers inside the handler). The binary is built with
// -fno-omit-frame-pointer; frames inside libc/libm without a frame pointer end the walk early, the bounds checks keep it safe.
static void handler(int sig, siginfo_t *si, void *ucv) {
    (void)sig; (void)si;
    size_t i = atomic_fetch_add(&nxt, 1);
    if (i >= cap) return;
    ucontext_t *uc = (ucontext_t *)ucv;
    sample_t *s = &ring[i];
    prctl(PR_GET_NAME, (unsigned long)s->name, 0, 0, 0);
    uintptr_t sp = (uintptr_t)uc->uc_mcontext.gregs[REG_RSP], fp = (uintptr_t)uc->uc_mcontext.gregs[REG_RBP];
    uint32_t n = 0;
    s->pc[n++] = (uint64_t)uc->uc_mcontext.gregs[REG_RIP];
    while (n < MAXF && stk_hi && fp >= sp && (fp & 7) == 0 && fp + 16 <= stk_hi) {
        uintptr_t next = ((uintptr_t *)fp)[0], ret = ((uintptr_t *)fp)[1];
        if (!ret) break;
        s->pc[n++] = ret;
        if (next <= fp) break;
        fp = next;
    }
    s->n = n;
}

__attribute__((constructor)) static void start(void) {
    char exe[512]; ssize_t l = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (l <= 0) return;
    exe[l] = 0;
    const char *b = strrchr(exe, '/'); b = b ? b + 1 : exe;
    if (strncmp(b, "singlet", 7) != 0 || !getenv("SAMPROF_OUT")) return;
    cap = (size_t)4 << 20; ring = (sample_t *)calloc(cap, sizeof(sample_t));
    if (!ring) return;
    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = handler; sa.sa_flags = SA_SIGINFO | SA_RESTART; sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, 0);
    const char *ms = getenv("SAMPROF_MS"); period_ms = ms ? atoi(ms) : 10; if (period_ms < 1) period_ms = 10;
    active = 1; arm_thread();
}

__attribute__((destructor)) static void stop(void) {
    if (!active) return;
    signal(SIGPROF, SIG_IGN); active = 0;
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
