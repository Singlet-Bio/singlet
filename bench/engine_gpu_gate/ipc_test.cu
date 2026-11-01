// SPDX-License-Identifier: MIT
// CUDA IPC feasibility for resident GPU indexes: `ipc_test server SOCK GB` allocates GB gigabytes on the device, fills it with a known pattern, exports the
// handle over a unix socket and stays up; `ipc_test client SOCK` (also from a forked child that initialises CUDA after the fork, as the STAR prefill child does)
// opens the handle, checks the pattern on the device, and times a device-side read of the whole buffer.
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)
__global__ void fill(uint64_t* p, uint64_t n) { uint64_t i = blockIdx.x * (uint64_t) blockDim.x + threadIdx.x; if (i < n) p[i] = i * 0x9E3779B97F4A7C15ULL; }
__global__ void check(const uint64_t* p, uint64_t n, unsigned long long* bad, unsigned long long* sum) {
    uint64_t s = 0, b = 0;
    for (uint64_t i = blockIdx.x * (uint64_t) blockDim.x + threadIdx.x; i < n; i += (uint64_t) gridDim.x * blockDim.x) { uint64_t v = p[i]; b += v != i * 0x9E3779B97F4A7C15ULL; s += v; }
    atomicAdd(bad, b); atomicAdd(sum, s);
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int connectTo(const char* path) {
    int s = socket(AF_UNIX, SOCK_STREAM, 0); sockaddr_un a{}; a.sun_family = AF_UNIX; std::strncpy(a.sun_path, path, sizeof a.sun_path - 1);
    for (int i = 0; i < 200; i++) { if (connect(s, (sockaddr*) &a, sizeof a) == 0) return s; usleep(100000); }
    std::fprintf(stderr, "cannot connect\n"); std::exit(3);
}
static int client(const char* path) {
    int s = connectTo(path);
    cudaIpcMemHandle_t h; uint64_t bytes;
    if (read(s, &h, sizeof h) != (ssize_t) sizeof h || read(s, &bytes, 8) != 8) { std::fprintf(stderr, "short read\n"); return 4; }
    double t0 = now();
    CK(cudaSetDevice(0));
    void* p = nullptr; CK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
    double t1 = now();
    unsigned long long *bad, *sum; CK(cudaMalloc(&bad, 8)); CK(cudaMalloc(&sum, 8)); CK(cudaMemset(bad, 0, 8)); CK(cudaMemset(sum, 0, 8));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    check<<<1056, 256>>>((const uint64_t*) p, bytes / 8, bad, sum);
    cudaEventRecord(b); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
    float ms; cudaEventElapsedTime(&ms, a, b);
    unsigned long long hb = 0; CK(cudaMemcpy(&hb, bad, 8, cudaMemcpyDeviceToHost));
    std::printf("pid %d: opened %.2f GB in %.3f s, %llu mismatches, read at %.0f GB/s\n", (int) getpid(), bytes / 1e9, t1 - t0, hb, bytes / 1e6 / ms);
    CK(cudaIpcCloseMemHandle(p));
    return hb == 0 ? 0 : 5;
}
int main(int argc, char** argv) {
    if (argc < 3) return 1;
    if (!std::strcmp(argv[1], "client")) {
        pid_t c = fork();                                // first a forked child that has not touched CUDA before the fork (the STAR prefill child)
        if (c == 0) _exit(client(argv[2]));
        int st = 0; waitpid(c, &st, 0);
        std::printf("forked client exit %d
", WIFEXITED(st) ? WEXITSTATUS(st) : -1);
        int rc = client(argv[2]);                        // then a plain process
        return rc | (WIFEXITED(st) ? WEXITSTATUS(st) : 9);
    }
    double gb = argc > 3 ? std::atof(argv[3]) : 1; uint64_t bytes = (uint64_t) (gb * (1ull << 30)) & ~7ull;
    CK(cudaSetDevice(0));
    void* p; CK(cudaMalloc(&p, bytes)); fill<<<(unsigned) ((bytes / 8 + 255) / 256), 256>>>((uint64_t*) p, bytes / 8); CK(cudaDeviceSynchronize());
    cudaIpcMemHandle_t h; CK(cudaIpcGetMemHandle(&h, p));
    unlink(argv[2]); int ls = socket(AF_UNIX, SOCK_STREAM, 0); sockaddr_un a{}; a.sun_family = AF_UNIX; std::strncpy(a.sun_path, argv[2], sizeof a.sun_path - 1);
    if (bind(ls, (sockaddr*) &a, sizeof a) || listen(ls, 8)) { std::perror("bind"); return 3; }
    std::printf("server pid %d serving %.2f GB\n", (int) getpid(), bytes / 1e9); std::fflush(stdout);
    for (int n = 0; n < 2; n++) { int c = accept(ls, nullptr, nullptr); if (write(c, &h, sizeof h) < 0 || write(c, &bytes, 8) < 0) {} close(c); }
    sleep(3);                                            // keep the allocation alive while the clients finish
    cudaFree(p); unlink(argv[2]); return 0;
}
