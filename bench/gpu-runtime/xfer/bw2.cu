// xfer probe 2: destination pre-population rates (MADV_POPULATE_WRITE, fallocate) and zero-copy random reads of pinned host memory
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
typedef unsigned long long u64;
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
static uint8_t* mkfd(u64 n, int* fdo) {
    int fd = (int) syscall(319, "bw", 1u); if (fd < 0 || ftruncate(fd, n)) { printf("memfd fail\n"); exit(1); }
    uint8_t* p = (uint8_t*) mmap(0, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); if (p == MAP_FAILED) { printf("mmap fail\n"); exit(1); } *fdo = fd; return p;
}
template <class F> static void par(int nt, u64 n, F f) { std::vector<std::thread> th; for (int t = 0; t < nt; t++) th.emplace_back([=] { f(n * t / nt, n * (t + 1) / nt); }); for (auto& x : th) x.join(); }
__global__ void k_rand(const uint4* e, u64 nlines, u64 q, unsigned long long* sink, int sorted) {
    u64 acc = 0;
    for (u64 i = blockIdx.x * (u64) blockDim.x + threadIdx.x; i < q; i += (u64) gridDim.x * blockDim.x) {
        u64 x = i * 0x9E3779B97F4A7C15ull; x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
        u64 line = sorted ? (i * (nlines / q)) : (x % nlines);   // sorted: increasing addresses, one line per probe
        uint4 v = e[line * 4];   // 64 B line, first 16 B
        acc += v.x + v.y;
    }
    if (acc == 0x123456789ull) *sink = acc;
}
int main(int argc, char** argv) {
    const u64 N = (u64) (argc > 1 ? atoi(argv[1]) : 8) << 30; double t;
    for (const char* f : {"/sys/kernel/mm/transparent_hugepage/shmem_enabled", "/sys/kernel/mm/transparent_hugepage/enabled"}) { FILE* fp = fopen(f, "r"); char b[200] = ""; if (fp) { if (fgets(b, 200, fp)) printf("%s: %s", f, b); fclose(fp); } }
    for (int nt : {1, 4, 8, 16}) {
        int fd; uint8_t* m = mkfd(N, &fd);
        t = now(); par(nt, N >> 12, [&](u64 a, u64 b) { if (b > a && madvise(m + (a << 12), (b - a) << 12, MADV_POPULATE_WRITE)) { static int once; if (!once++) perror("MADV_POPULATE_WRITE"); } });
        printf("MADV_POPULATE_WRITE nt=%2d: %.1f GB/s\n", nt, N / 1e9 / (now() - t));
        munmap(m, N); close(fd);
    }
    for (int nt : {1, 4, 8}) {
        int fd; uint8_t* m = mkfd(N, &fd);
        t = now(); par(nt, N >> 12, [&](u64 a, u64 b) { if (b > a && fallocate(fd, 0, a << 12, (b - a) << 12)) { static int once; if (!once++) perror("fallocate"); } });
        double a1 = now() - t;
        t = now(); par(16, N >> 12, [&](u64 a, u64 b) { if (b > a) madvise(m + (a << 12), (b - a) << 12, MADV_POPULATE_WRITE); });
        printf("fallocate nt=%2d: %.1f GB/s, then map (populate 16 thr) %.1f GB/s\n", nt, N / 1e9 / a1, N / 1e9 / (now() - t));
        munmap(m, N); close(fd);
    }
    { // huge pages on memfd mapping
        int fd; uint8_t* m = mkfd(N, &fd); int r = madvise(m, N, MADV_HUGEPAGE);
        t = now(); par(16, N >> 12, [&](u64 a, u64 b) { if (b > a) madvise(m + (a << 12), (b - a) << 12, MADV_POPULATE_WRITE); });
        printf("MADV_HUGEPAGE rc=%d, populate 16 thr: %.1f GB/s\n", r, N / 1e9 / (now() - t)); munmap(m, N); close(fd);
    }
    // zero-copy random line reads from pinned mapped host memory (16 B of each 64 B line; 16 GB, as an index)
    const u64 HB = 16ull << 30; uint8_t* h; CK(cudaHostAlloc((void**) &h, HB, cudaHostAllocMapped)); par(16, HB >> 12, [&](u64 a, u64 b) { for (u64 i = a; i < b; i++) h[i << 12] = 1; });
    uint8_t* dh; CK(cudaHostGetDevicePointer((void**) &dh, h, 0)); unsigned long long* sink; CK(cudaMalloc(&sink, 8));
    for (u64 Q : {50000000ull, 200000000ull}) for (int sorted = 0; sorted < 2; sorted++) {
        k_rand<<<4096, 256>>>((const uint4*) dh, HB / 64, 1000000, sink, sorted); CK(cudaDeviceSynchronize());
        t = now(); k_rand<<<1 << 16, 256>>>((const uint4*) dh, HB / 64, Q, sink, sorted); CK(cudaDeviceSynchronize()); double s = now() - t;
        printf("zero-copy %s line reads: %llu M in %.2f s = %.0f M reads/s (%.1f GB/s of 64 B lines)\n", sorted ? "sorted" : "random", Q / 1000000, s, Q / 1e6 / s, Q * 64.0 / 1e9 / s);
    }
    return 0;
}
