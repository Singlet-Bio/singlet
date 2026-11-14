// rt/holds (SINGLET_MAP2_POOL, default on; =0 restores per-call cudaMalloc / cudaFree): a scoped caching allocator for the device buffers that
// the map2 relock hold's record-chunk loop (engine_gpu.cu e2eBuildRC: buildRecords, E10's products, thrust's temporaries) allocated and freed
// once per chunk. While a Scope is alive, a freed block goes to a free list and the next allocation of about its size takes it back (best fit
// within 25 % + 1 MiB); the Scope's end (depth 0) returns every cached block to the driver. Outside a Scope pAlloc / pFree are cudaMalloc /
// cudaFree. The cache is bounded by SINGLET_MAP2_POOL_GB (default 3): a block freed past the bound is cudaFree'd, and a failing cudaMalloc
// drains the cache and retries, so the pool cannot turn an allocation that fits into one that fails.
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <new>

namespace singlet_devpool {

struct State {
    std::mutex mu;
    int depth = 0;
    std::multimap<size_t, void*> freeList;   // size -> block
    std::map<void*, size_t> sizes;           // blocks allocated while a Scope was alive (their rounded sizes)
    size_t cached = 0, peakCached = 0, capBytes = 0;
    unsigned long long nMalloc = 0, nHit = 0, nFreed = 0;
};
inline State& st() { static State* s = new State; return *s; }
inline bool enabled() {
    static const bool on = [] { const char* e = std::getenv("SINGLET_MAP2_POOL"); return !(e && e[0] == '0'); }();
    return on;
}
inline size_t capBytes() {
    static const size_t c = [] {
        const char* e = std::getenv("SINGLET_MAP2_POOL_GB");
        const double g = e ? std::atof(e) : 3.0;
        return (size_t) ((g > 0 ? g : 0) * (double) (1ull << 30));
    }();
    return c;
}
inline size_t roundUp(size_t b) {
    if (b < (1u << 20)) return (b + 4095) & ~(size_t) 4095;
    return (b + (1u << 20) - 1) & ~(size_t) ((1u << 20) - 1);
}
// every cached block back to the driver
inline void drain() {
    State& s = st();
    std::multimap<size_t, void*> fl;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        fl.swap(s.freeList);
        for (auto& kv : fl) s.sizes.erase(kv.second);
        s.cached = 0;
    }
    for (auto& kv : fl) { cudaFree(kv.second); }
    std::lock_guard<std::mutex> lk(s.mu);
    s.nFreed += fl.size();
}
inline cudaError_t pAlloc(void** p, size_t bytes) {
    State& s = st();
    bool act;
    { std::lock_guard<std::mutex> lk(s.mu); act = s.depth > 0; }
    if (!act) return cudaMalloc(p, bytes);
    const size_t want = roundUp(bytes ? bytes : 1);
    {
        std::lock_guard<std::mutex> lk(s.mu);
        auto it = s.freeList.lower_bound(want);
        if (it != s.freeList.end() && it->first <= want + want / 4 + (1u << 20)) {
            *p = it->second; s.cached -= it->first; s.freeList.erase(it); s.nHit++;
            return cudaSuccess;
        }
    }
    cudaError_t e = cudaMalloc(p, want);
    if (e != cudaSuccess) { cudaGetLastError(); drain(); e = cudaMalloc(p, want); }
    if (e != cudaSuccess) { *p = nullptr; return e; }
    std::lock_guard<std::mutex> lk(s.mu);
    s.sizes[*p] = want; s.nMalloc++;
    return cudaSuccess;
}
template <class T> inline cudaError_t pAlloc(T** p, size_t bytes) { return pAlloc((void**) p, bytes); }
inline void pFree(void* p) {
    if (!p) return;
    State& s = st();
    {
        std::lock_guard<std::mutex> lk(s.mu);
        auto it = s.sizes.find(p);
        if (it != s.sizes.end()) {
            if (s.depth > 0 && s.cached + it->second <= capBytes()) {
                s.freeList.emplace(it->second, p); s.cached += it->second; s.peakCached = std::max(s.peakCached, s.cached);
                return;
            }
            s.sizes.erase(it);
        }
    }
    cudaFree(p);
}
// thrust::cuda::par(alloc): thrust's temporaries through the pool
struct ThrustAlloc {
    typedef char value_type;
    char* allocate(std::ptrdiff_t n) { void* p = nullptr; if (pAlloc(&p, (size_t) n) != cudaSuccess) throw std::bad_alloc(); return (char*) p; }
    void deallocate(char* p, size_t) { pFree(p); }
};
// the pool is on for the Scope's life (nested scopes count); the outermost one's end drains it and logs one line
struct Scope {
    bool on;
    explicit Scope(const char* what) : on(enabled()), what_(what) {
        if (!on) return;
        State& s = st(); std::lock_guard<std::mutex> lk(s.mu);
        if (s.depth++ == 0) { s.nMalloc = s.nHit = s.nFreed = 0; s.peakCached = 0; }
    }
    ~Scope() {
        if (!on) return;
        State& s = st();
        bool last;
        { std::lock_guard<std::mutex> lk(s.mu); last = (s.depth == 1); if (last) s.depth = 0; else s.depth--; }
        if (!last) return;
        drain();
        std::lock_guard<std::mutex> lk(s.mu);
        fprintf(stderr, "[engine_gpu] device pool (%s): %llu cudaMalloc, %llu reuses, peak cached %.2f GB\n", what_, s.nMalloc, s.nHit,
                (double) s.peakCached / (double) (1ull << 30));
    }
    const char* what_;
};

}  // namespace singlet_devpool
