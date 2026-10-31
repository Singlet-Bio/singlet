// LD_PRELOAD sampling profiler for the singlet binary (perf is blocked on Clipper: perf_event_paranoid=4).
// Every thread gets its own CLOCK_THREAD_CPUTIME_ID timer (armed at thread start through a pthread_create wrapper) that raises SIGPROF on
// that thread every SAMPROF_MS of its own CPU time, so each sample is exactly SAMPROF_MS of CPU and all threads are covered. (A single
// process-wide ITIMER_PROF delivers to the main thread only and misses the workers.) The handler records the thread name and up to 32 return addresses; at exit they are
// written to $SAMPROF_OUT/samples.<pid>.bin with /proc/self/maps as maps.<pid>.txt (the STAR child is a fork of singlet, so one file pair per process). Resolve with resolve.py. Active only in processes whose
// executable name starts with "singlet", so /usr/bin/time, bash and sam-dump are left alone. No source change to singlet.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
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
static uint64_t *hdr; static sample_t *ring; static size_t cap; static atomic_size_t nxt; static int active; static int period_ms = 10;

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
    *(volatile uint64_t *)hdr = atomic_load(&nxt) < cap ? atomic_load(&nxt) : cap;
}

// The ring is a MAP_SHARED mmap of samples.<pid>.bin (header: u64 count), so nothing has to run at exit: the STAR child is a fork of
// singlet that does not run destructors. A forked child opens its own file pair (pthread_atfork) and re-arms its main thread.
static void write_maps(const char *dir) {
    char p[1024]; snprintf(p, sizeof p, "%s/maps.%d.txt", dir, (int)getpid());
    FILE *f = fopen(p, "w"), *m = fopen("/proc/self/maps", "r");
    if (f && m) { char ln[1024]; while (fgets(ln, sizeof ln, m)) fputs(ln, f); }
    if (f) fclose(f);
    if (m) fclose(m);
}

static int open_ring(void) {
    const char *dir = getenv("SAMPROF_OUT"); char p[1024];
    snprintf(p, sizeof p, "%s/samples.%d.bin", dir, (int)getpid());
    int fd = open(p, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 0;
    size_t bytes = 8 + cap * sizeof(sample_t);
    if (ftruncate(fd, (off_t)bytes) != 0) { close(fd); return 0; }
    void *base = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return 0;
    atomic_store(&nxt, 0);
    hdr = (uint64_t *)base; *hdr = 0; ring = (sample_t *)(hdr + 1);
    write_maps(dir);
    return 1;
}

static void child_init(void) { if (active && open_ring()) arm_thread(); }

__attribute__((constructor)) static void start(void) {
    char exe[512]; ssize_t l = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (l <= 0) return;
    exe[l] = 0;
    const char *b = strrchr(exe, '/'); b = b ? b + 1 : exe;
    if (strncmp(b, "singlet", 7) != 0 || !getenv("SAMPROF_OUT")) return;
    cap = (size_t)1500000;
    if (!open_ring()) return;
    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = handler; sa.sa_flags = SA_SIGINFO | SA_RESTART; sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, 0);
    const char *ms = getenv("SAMPROF_MS"); period_ms = ms ? atoi(ms) : 10; if (period_ms < 1) period_ms = 10;
    active = 1; pthread_atfork(0, 0, child_init); arm_thread();
}
