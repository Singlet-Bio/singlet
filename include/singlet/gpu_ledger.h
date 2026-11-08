// Transfer ledger of the GPU path (SINGLET_ENGINE_GPU_TRACE=1). Off by default: one branch per copy.
//
// Every cudaMemcpy and cudaMemcpyAsync of a .cu file that includes this header after <cuda_runtime.h> is counted by source line: calls,
// bytes, host seconds of the call, and whether the host side is page-locked. A Phase object (RAII) names a top-level GPU function; when it
// ends it prints its rows to stderr, heaviest first, with its wall time, then clears them:
//   [gpu-ledger] prefill 12.4 s wall: H2D 3.10 GB in 0.41 s (7.6 GB/s), D2H 1.92 GB in 0.30 s, 812 calls
//   [gpu-ledger] prefill engine_gpu.cu:1311 H2D 1 calls, 2.10 GB, 0.270 s host, 7.78 GB/s, pinned
// The prefill runs in the forked STAR child, which nsys follows only with --trace-fork-before-exec, so this is the profile any normal job
// can carry. Async copies count the host seconds of the enqueue only (their transfer time is in the stream; nsys shows it).
// It redevelops the ledger of perf/nvtx-host-ranges (a210148e, engine_gpu.cu only), which the staging merge of the Fable line dropped.
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace singlet { namespace gpu_ledger {

inline bool on() { static const bool v = std::getenv("SINGLET_ENGINE_GPU_TRACE") != nullptr; return v; }
inline double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Row { unsigned long long n = 0, bytes = 0; double sec = 0; int pinned = 0, kind = 0, async = 0; };
struct Book { std::mutex mu; std::map<std::pair<std::string, int>, Row> rows; };
inline Book& book() { static Book b; return b; }

inline const char* base(const char* f) { const char* s = std::strrchr(f, '/'); return s ? s + 1 : f; }
inline void note(const char* file, int line, const void* dst, const void* src, size_t n, cudaMemcpyKind k, double dt, bool async) {
    int pinned = 0;
    const void* h = k == cudaMemcpyHostToDevice ? src : (k == cudaMemcpyDeviceToHost ? dst : nullptr);
    if (h != nullptr) {
        cudaPointerAttributes a;
        if (cudaPointerGetAttributes(&a, h) == cudaSuccess) pinned = a.type == cudaMemoryTypeHost;
        cudaGetLastError();
    }
    Book& b = book();
    std::lock_guard<std::mutex> g(b.mu);
    Row& w = b.rows[{base(file), line}];
    w.n++; w.bytes += n; w.sec += dt; w.pinned |= pinned; w.kind = (int) k; w.async |= async;
}
inline cudaError_t copy(const char* file, int line, void* dst, const void* src, size_t n, cudaMemcpyKind k) {
    if (!on()) return ::cudaMemcpy(dst, src, n, k);
    const double t = now();
    const cudaError_t r = ::cudaMemcpy(dst, src, n, k);
    note(file, line, dst, src, n, k, now() - t, false);
    return r;
}
inline cudaError_t copyAsync(const char* file, int line, void* dst, const void* src, size_t n, cudaMemcpyKind k, cudaStream_t s = 0) {
    if (!on()) return ::cudaMemcpyAsync(dst, src, n, k, s);
    const double t = now();
    const cudaError_t r = ::cudaMemcpyAsync(dst, src, n, k, s);
    note(file, line, dst, src, n, k, now() - t, true);
    return r;
}
inline const char* kindName(int k) { return k == cudaMemcpyHostToDevice ? "H2D" : k == cudaMemcpyDeviceToHost ? "D2H" : k == cudaMemcpyDeviceToDevice ? "D2D" : "other"; }
inline void report(const char* phase, double wall) {
    if (!on()) return;
    std::vector<std::pair<std::pair<std::string, int>, Row>> v;
    { Book& b = book(); std::lock_guard<std::mutex> g(b.mu); v.assign(b.rows.begin(), b.rows.end()); b.rows.clear(); }
    double sec[4] = {0, 0, 0, 0}; unsigned long long by[4] = {0, 0, 0, 0}, calls = 0;
    for (auto& kv : v) { const int k = std::min(3, std::max(0, kv.second.kind)); sec[k] += kv.second.sec; by[k] += kv.second.bytes; calls += kv.second.n; }
    std::fprintf(stderr, "[gpu-ledger] %s %.3f s wall: H2D %.3f GB in %.3f s, D2H %.3f GB in %.3f s, D2D %.3f GB, %llu calls\n", phase, wall,
                 by[cudaMemcpyHostToDevice] / 1e9, sec[cudaMemcpyHostToDevice], by[cudaMemcpyDeviceToHost] / 1e9, sec[cudaMemcpyDeviceToHost],
                 by[cudaMemcpyDeviceToDevice] / 1e9, calls);
    std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.second.sec > y.second.sec; });
    for (auto& kv : v) {
        const Row& w = kv.second;
        std::fprintf(stderr, "[gpu-ledger] %s %s:%d %s %llu calls, %.4f GB, %.4f s host, %.2f GB/s, %s%s\n", phase, kv.first.first.c_str(),
                     kv.first.second, kindName(w.kind), w.n, w.bytes / 1e9, w.sec, w.sec > 0 ? w.bytes / 1e9 / w.sec : 0.0,
                     w.pinned ? "pinned" : "pageable", w.async ? ", async (enqueue time)" : "");
    }
}
// one top-level GPU function: its rows are printed and cleared when it returns
struct Phase {
    const char* name; double t0;
    explicit Phase(const char* n) : name(n), t0(on() ? now() : 0) {}
    ~Phase() { if (on()) report(name, now() - t0); }
};

}} // namespace singlet::gpu_ledger

#ifndef SINGLET_GPU_LEDGER_NO_WRAP
#define cudaMemcpy(d, s, n, k) ::singlet::gpu_ledger::copy(__FILE__, __LINE__, (d), (s), (n), (k))
#define cudaMemcpyAsync(...) ::singlet::gpu_ledger::copyAsync(__FILE__, __LINE__, __VA_ARGS__)
#endif
