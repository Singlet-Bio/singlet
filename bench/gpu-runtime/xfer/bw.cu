// xfer bandwidth probe: pinned vs pageable, staged D2H into a fresh/prefaulted memfd, cudaHostRegister cost of a memfd mapping
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
typedef unsigned long long u64;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
static void par(int nt, u64 n, void (*f)(u64, u64, void*), void* a) {
    std::vector<std::thread> th; for (int t = 0; t < nt; t++) th.emplace_back([=] { f(n * t / nt, n * (t + 1) / nt, a); }); for (auto& x : th) x.join();
}
struct CP { uint8_t* d; const uint8_t* s; };
static void cpf(u64 a, u64 b, void* p) { CP* c = (CP*) p; memcpy(c->d + a, c->s + a, b - a); }
static void touch(u64 a, u64 b, void* p) { uint8_t* x = (uint8_t*) p; for (u64 q = a & ~4095ull; q < b; q += 4096) x[q] = 1; }
static uint8_t* mkfd(u64 n, int* fdo) {
    int fd = (int) syscall(319, "bw", 1u); if (fd < 0 || ftruncate(fd, n)) { printf("memfd fail\n"); exit(1); }
    uint8_t* p = (uint8_t*) mmap(0, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); if (p == MAP_FAILED) { printf("mmap fail\n"); exit(1); } *fdo = fd; return p;
}
// staged D2H: dev -> 2 x 64 MB pinned -> dst, nt threads for the memcpy
static double staged(uint8_t* dst, const uint8_t* dsrc, u64 n, int nt, uint8_t* buf[2], cudaStream_t s, cudaEvent_t ev[2]) {
    const u64 CH = 64ull << 20, nc = (n + CH - 1) / CH; double t0 = now();
    auto issue = [&](u64 c) { u64 o = c * CH, m = std::min(CH, n - o); CK(cudaMemcpyAsync(buf[c & 1], dsrc + o, m, cudaMemcpyDeviceToHost, s)); CK(cudaEventRecord(ev[c & 1], s)); };
    issue(0);
    for (u64 c = 0; c < nc; c++) {
        if (c + 1 < nc) issue(c + 1);
        CK(cudaEventSynchronize(ev[c & 1]));
        u64 o = c * CH, m = std::min(CH, n - o); CP cp{dst + o, buf[c & 1]}; par(nt, m, cpf, &cp);
    }
    return now() - t0;
}
int main(int argc, char** argv) {
    const u64 N = (u64) (argc > 1 ? atoi(argv[1]) : 8) << 30;
    CK(cudaSetDevice(0)); uint8_t* d; CK(cudaMalloc(&d, N)); CK(cudaMemset(d, 7, N));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); cudaEvent_t ev[2]; uint8_t* buf[2];
    for (int b = 0; b < 2; b++) { CK(cudaHostAlloc((void**) &buf[b], 64 << 20, 0)); CK(cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming)); }
    double t;
    // pinned
    uint8_t* hp; t = now(); CK(cudaHostAlloc((void**) &hp, N, 0)); printf("cudaHostAlloc %llu GB: %.2f s (fresh, faults inside)\n", N >> 30, now() - t);
    for (int r = 0; r < 2; r++) {
        t = now(); CK(cudaMemcpy(hp, d, N, cudaMemcpyDeviceToHost)); double a = now() - t;
        t = now(); CK(cudaMemcpy(d, hp, N, cudaMemcpyHostToDevice)); double b = now() - t;
        printf("PINNED   D2H %.1f GB/s  H2D %.1f GB/s\n", N / 1e9 / a, N / 1e9 / b);
    }
    CK(cudaFreeHost(hp));
    // pageable (malloc, prefaulted)
    uint8_t* hg = (uint8_t*) malloc(N); par(16, N, touch, hg);
    for (int r = 0; r < 2; r++) {
        t = now(); CK(cudaMemcpy(hg, d, N, cudaMemcpyDeviceToHost)); double a = now() - t;
        t = now(); CK(cudaMemcpy(d, hg, N, cudaMemcpyHostToDevice)); double b = now() - t;
        printf("PAGEABLE D2H %.1f GB/s  H2D %.1f GB/s\n", N / 1e9 / a, N / 1e9 / b);
    }
    // staged into a fresh memfd, nt threads; second pass prefaulted
    for (int nt : {4, 8, 16}) {
        int fd; uint8_t* m = mkfd(N, &fd);
        double a = staged(m, d, N, nt, buf, s, ev), b = staged(m, d, N, nt, buf, s, ev);
        printf("STAGED nt=%2d into memfd: fresh %.1f GB/s, prefaulted %.1f GB/s\n", nt, N / 1e9 / a, N / 1e9 / b);
        munmap(m, N); close(fd);
    }
    { // staged into malloc fresh
        uint8_t* m = (uint8_t*) malloc(N); double a = staged(m, d, N, 8, buf, s, ev); double b = staged(m, d, N, 8, buf, s, ev);
        printf("STAGED nt= 8 into malloc: fresh %.1f GB/s, prefaulted %.1f GB/s\n", N / 1e9 / a, N / 1e9 / b); free(m);
    }
    { // populate cost, then register cost, then direct D2H
        int fd; uint8_t* m = mkfd(N, &fd);
        t = now(); par(16, N, touch, m); double pf = now() - t;
        t = now(); cudaError_t e = cudaHostRegister(m, N, cudaHostRegisterDefault); double rg = now() - t;
        printf("memfd %llu GB: parallel fault-in (16 thr) %.2f s; cudaHostRegister %.2f s (%s)\n", N >> 30, pf, rg, cudaGetErrorString(e));
        if (e == cudaSuccess) {
            for (int r = 0; r < 2; r++) { t = now(); CK(cudaMemcpy(m, d, N, cudaMemcpyDeviceToHost)); double a = now() - t; printf("direct D2H into registered memfd: %.1f GB/s\n", N / 1e9 / a); }
            t = now(); CK(cudaHostUnregister(m)); printf("cudaHostUnregister %.2f s\n", now() - t);
        } else cudaGetLastError();
        // register fresh (unfaulted) memfd
        uint8_t* m2; int fd2; m2 = mkfd(N, &fd2);
        t = now(); e = cudaHostRegister(m2, N, cudaHostRegisterDefault); rg = now() - t; printf("cudaHostRegister fresh memfd: %.2f s (%s)\n", rg, cudaGetErrorString(e));
        if (e == cudaSuccess) { t = now(); CK(cudaMemcpy(m2, d, N, cudaMemcpyDeviceToHost)); printf("direct D2H into fresh registered memfd: %.1f GB/s\n", N / 1e9 / (now() - t)); cudaHostUnregister(m2); }
        else cudaGetLastError();
    }
    return 0;
}
