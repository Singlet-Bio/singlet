// SPDX-License-Identifier: MIT
// Random-load speed of a device buffer by how it was obtained (gpu/bold-shidx, docs/gpu-engine/bold/SHARED-INDEX.md).
//   private    cudaMalloc in this process
//   vmm-local  cuMemCreate + cuMemMap in this process
//   ipc        cudaMalloc in this process, cudaIpcOpenMemHandle in a forked child
//   vmm-import cuMemCreate in this process, descriptor over SCM_RIGHTS, cuMemImportFromShareableHandle + cuMemMap in the child
// Each reads 1e8 random 8-byte words of a GB-sized buffer (a dependent chain per thread) and reports G loads/s and GB/s of useful data.
// Build: nvcc -O2 -o ipc_latency_test ipc_latency_test.cu      Run: ./ipc_latency_test [GB=12]   (one GPU; the child starts before CUDA is initialised)
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)
#define CU(x) do { CUresult e_ = (x); if (e_ != CUDA_SUCCESS) { std::fprintf(stderr, "%s:%d %s: driver error %d\n", __FILE__, __LINE__, #x, (int) e_); std::exit(2); } } while (0)
typedef unsigned long long u64;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__global__ void chase(const u64* p, u64 nw, u64 iters, u64* out) {
    u64 t = blockIdx.x * (u64) blockDim.x + threadIdx.x, s = t * 0x9E3779B97F4A7C15ULL + 1, acc = 0;
    for (u64 i = 0; i < iters; i++) { s = s * 6364136223846793005ULL + 1442695040888963407ULL + acc; acc += p[(s >> 11) % nw]; }
    if (acc == 0x123456789ULL) out[0] = acc;
}
__global__ void fillk(u64* p, u64 nw) { for (u64 i = blockIdx.x * (u64) blockDim.x + threadIdx.x; i < nw; i += (u64) gridDim.x * blockDim.x) p[i] = i * 0x9E3779B97F4A7C15ULL; }
static void bench(const char* name, void* p, u64 bytes) {
    const u64 nw = bytes / 8, thr = 1056ull * 256, total = 100000000ull, iters = total / thr;
    u64* out; CK(cudaMalloc(&out, 8));
    chase<<<1056, 256>>>((const u64*) p, nw, 4, out); CK(cudaDeviceSynchronize());   // warm up (TLB and page tables of a small part)
    const double t0 = now(); chase<<<1056, 256>>>((const u64*) p, nw, iters, out); CK(cudaDeviceSynchronize()); const double dt = now() - t0;
    std::printf("%-11s %7.3f s  %8.3f Gloads/s  %8.2f GB/s (8 B per load)\n", name, dt, iters * thr / dt / 1e9, iters * thr * 8 / dt / 1e9); std::fflush(stdout);
    CK(cudaFree(out));
}
struct Drv {
    CUresult (*create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
    CUresult (*reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
    CUresult (*map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
    CUresult (*access)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
    CUresult (*exportH)(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long);
    CUresult (*importH)(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType);
    CUresult (*gran)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
    CUresult (*release)(CUmemGenericAllocationHandle);
};
template <class F> static void get(const char* n, F& f) {
    void* p = nullptr; cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPoint(n, &p, cudaEnableDefault, &q) != cudaSuccess || q != cudaDriverEntryPointSuccess || !p) { std::fprintf(stderr, "no %s\n", n); std::exit(2); }
    f = reinterpret_cast<F>(p);
}
static Drv drv() {
    Drv d; get("cuMemCreate", d.create); get("cuMemAddressReserve", d.reserve); get("cuMemMap", d.map); get("cuMemSetAccess", d.access);
    get("cuMemExportToShareableHandle", d.exportH); get("cuMemImportFromShareableHandle", d.importH); get("cuMemGetAllocationGranularity", d.gran); get("cuMemRelease", d.release);
    return d;
}
static CUmemAllocationProp prop() {
    CUmemAllocationProp p{}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = 0; return p;
}
static void* mapIt(Drv& D, CUmemGenericAllocationHandle h, u64 sz) {
    CUdeviceptr va; CUmemAccessDesc ad{}; ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id = 0; ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU(D.reserve(&va, sz, 0, 0, 0)); CU(D.map(va, sz, 0, h, 0)); CU(D.access(va, sz, &ad, 1)); return (void*) va;
}
static void sendFd(int s, int fd, const void* data, size_t n) {
    char cb[CMSG_SPACE(sizeof(int))]; std::memset(cb, 0, sizeof cb);
    iovec iov{(void*) data, n}; msghdr m{}; m.msg_iov = &iov; m.msg_iovlen = 1; m.msg_control = cb; m.msg_controllen = sizeof cb;
    cmsghdr* c = CMSG_FIRSTHDR(&m); c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int)); std::memcpy(CMSG_DATA(c), &fd, sizeof fd);
    if (sendmsg(s, &m, 0) != (ssize_t) n) { std::perror("sendmsg"); std::exit(3); }
}
static int recvFd(int s, void* data, size_t n) {
    char cb[CMSG_SPACE(sizeof(int))]; std::memset(cb, 0, sizeof cb);
    iovec iov{data, n}; msghdr m{}; m.msg_iov = &iov; m.msg_iovlen = 1; m.msg_control = cb; m.msg_controllen = sizeof cb;
    if (recvmsg(s, &m, 0) != (ssize_t) n) { std::perror("recvmsg"); std::exit(3); }
    int fd = -1; for (cmsghdr* c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c)) if (c->cmsg_type == SCM_RIGHTS) std::memcpy(&fd, CMSG_DATA(c), sizeof fd);
    return fd;
}
int main(int argc, char** argv) {
    const u64 bytes = (u64) ((argc > 1 ? std::atof(argv[1]) : 12.0) * (1ull << 30));
    int sv[2]; if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv)) return 1;
    const pid_t ch = fork();
    if (ch == 0) {   // the importer: CUDA starts here, after the fork
        close(sv[0]);
        cudaIpcMemHandle_t h; u64 sz;
        struct { cudaIpcMemHandle_t h; u64 sz; u64 vsz; } m1;
        if (read(sv[1], &m1, sizeof m1) != (ssize_t) sizeof m1) return 4;
        CK(cudaSetDevice(0)); CK(cudaFree(0));
        void* pi = nullptr; CK(cudaIpcOpenMemHandle(&pi, m1.h, cudaIpcMemLazyEnablePeerAccess)); bench("ipc", pi, m1.sz);
        u64 vsz; int fd = recvFd(sv[1], &vsz, 8);
        Drv D = drv(); CUmemGenericAllocationHandle vh; CU(D.importH(&vh, (void*) (intptr_t) fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        void* pv = mapIt(D, vh, vsz); bench("vmm-import", pv, bytes);
        char ok = 1; if (write(sv[1], &ok, 1) != 1) return 5; (void) h; (void) sz;
        return 0;
    }
    close(sv[1]);
    CK(cudaSetDevice(0)); CK(cudaFree(0));
    Drv D = drv(); CUmemAllocationProp pr = prop(); size_t gr = 0, gm = 0;
    CU(D.gran(&gr, &pr, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)); CU(D.gran(&gm, &pr, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    std::printf("buffer %.1f GB, vmm granularity recommended %zu minimum %zu\n", bytes / 1073741824.0, gr, gm);
    const u64 vsz = (bytes + gr - 1) / gr * gr;
    void* pa; CK(cudaMalloc(&pa, bytes)); fillk<<<1056, 256>>>((u64*) pa, bytes / 8);
    CUmemGenericAllocationHandle vh; CU(D.create(&vh, vsz, &pr, 0)); void* pv = mapIt(D, vh, vsz); fillk<<<1056, 256>>>((u64*) pv, bytes / 8);
    void* pb; CK(cudaMalloc(&pb, bytes)); fillk<<<1056, 256>>>((u64*) pb, bytes / 8); CK(cudaDeviceSynchronize());
    bench("private", pa, bytes); bench("vmm-local", pv, bytes);
    struct { cudaIpcMemHandle_t h; u64 sz; u64 vsz; } m1; CK(cudaIpcGetMemHandle(&m1.h, pb)); m1.sz = bytes; m1.vsz = vsz;
    if (write(sv[0], &m1, sizeof m1) != (ssize_t) sizeof m1) return 6;
    int fd = -1; CU(D.exportH(&fd, vh, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0)); sendFd(sv[0], fd, &vsz, 8);
    char ok = 0; if (read(sv[0], &ok, 1) != 1) std::fprintf(stderr, "the importer failed\n");
    int st = 0; waitpid(ch, &st, 0); std::printf("importer exit %d\n", WEXITSTATUS(st));
    return 0;
}
