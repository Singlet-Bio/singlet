#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
// nsys + forked children: flush CUPTI at _exit (the STAR child ends with _exit) and every 2 s from a thread (the pool's non-host server
// is forked and ends by SIGTERM); threads do not survive fork, so the child restarts it (pthread_atfork).
typedef int (*fl_t)(unsigned int);
static fl_t fl_fn;
static fl_t fl_find(void) {
  if (fl_fn) return fl_fn;
  char line[1024], path[900] = ""; FILE* f = fopen("/proc/self/maps", "r");
  if (f) { while (fgets(line, sizeof line, f)) { char* p = strchr(line, '/'); if (p && strstr(p, "libcupti.so")) { strncpy(path, p, 899); char* n = strchr(path, '\n'); if (n) *n = 0; break; } } fclose(f); }
  if (!path[0]) return 0;
  void* h = dlopen(path, RTLD_NOW | RTLD_NOLOAD); if (!h) return 0;
  fl_fn = (fl_t)dlsym(h, "cuptiActivityFlushAll"); return fl_fn;
}
static void fl_flush(int wait) { fl_t fn = fl_find(); if (fn) { fn(1); if (wait) usleep(3000000); } }
static void* fl_loop(void* a) { (void)a; for (;;) { sleep(2); fl_flush(0); } return 0; }
static void fl_start(void) { pthread_t t; if (pthread_create(&t, 0, fl_loop, 0) == 0) pthread_detach(t); }
__attribute__((constructor)) static void fl_init(void) { fl_start(); pthread_atfork(0, 0, fl_start); }
void _exit(int s) { static void (*real)(int); fl_flush(1); real = (void (*)(int))dlsym(RTLD_NEXT, "_exit"); real(s); for (;;) {} }
void _Exit(int s) { _exit(s); }
