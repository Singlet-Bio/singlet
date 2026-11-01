// singlet GPU engine: the non-host (viral / bacterial / fungal) read screen on the GPU.
// API: include/singlet/engine_gpu.h (nonhostClassify). Measured identity and speed: docs/gpu-engine/exploration.md §12.16 and
// bench/gpu/m13/README.md on the W11 branch (Singlet-Bio/singlet#61), where this code was developed as m13.cu.
//
// What it replaces: Phases A-C of NonHostScreener::classify_multi_batch_filtered (include/singlet/pileup/nonhost/): minimizer
// extraction (k = 21, w = 11), host-Bloom subtraction, the sorted-minimizer x sorted-index join and the per-read species hit
// maps, plus the host's load and merge of the 26 GB index. It returns, per read, the number of non-host minimizers (np) and the
// (read, map, species) hit counts in the CPU's first-insertion order; the unchanged host code then runs Phase D (rates, best
// category), the EM and every writer (NonHostScreener::classify_from_gpu).
// Index: the three .snhskidx files (16-byte {hash, species} entries sorted by hash) are copied as they are into HBM and given a
// radix directory over the top B hash bits; a probe is one directory read plus a short scan. Ideas re-implemented, no code
// copied, from the design-input session (read only): singlet-agents/.claude/worktrees/gpu-align-vision/bench/gpu-align/a3_hash.cu
// (random probes into a tens-of-GB table), a4_sort.cu (cub radix sort) and b1_kmer.cu (rolling canonical 2-bit k-mers).
// Exactness: all integer; no floating point on the GPU.
#include "singlet/engine_gpu.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
using singlet_engine_gpu::NonhostPaths;
using singlet_engine_gpu::NonhostResult;

constexpr int K = 21, W = 11;
constexpr uint64_t INV = ~0ULL;
constexpr uint64_t KMASK = (1ULL << (2 * K)) - 1;
constexpr uint32_t MIN_NONHOST_MINS = 5;   // NonHostScreener::kMinNonhostMins

struct __align__(16) Entry { uint64_t h; uint32_t sid; uint32_t pad; };

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

#define NH_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::string(#x) + ": " + cudaGetErrorString(e_); } while (0)

// ---------------------------------------------------------------- device code
__host__ __device__ inline uint64_t mm64(uint64_t k) {   // murmurhash64A finalizer = nonhost::murmurhash64A
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33; return k;
}
__device__ inline int base2(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return 255;
    }
}
// HostKmerFilter::contains: 7 probes, g_i = (h1 + i*h2) % n_bits, h2 forced odd
__device__ inline bool bloom_contains(const uint64_t* bits, uint64_t nbits, uint64_t m) {
    const uint64_t h1 = mm64(m ^ 0x9e3779b97f4a7c15ULL);
    const uint64_t h2 = mm64(m ^ 0x6c62272e07bb0142ULL) | 1ULL;
    for (int i = 0; i < 7; ++i) {
        const uint64_t pos = (h1 + (uint64_t)i * h2) % nbits;
        if (!((__ldg(bits + (pos >> 6)) >> (pos & 63ULL)) & 1ULL)) return false;
    }
    return true;
}
// extract_minimizers (min_sketch.h) followed by the host filter: returns the number of non-host minimizers; FILL writes them in
// window order. Rolling forward / reverse-complement 2-bit k-mers equal encode_kmer / rc_kmer of each window; a window with any
// non-ACGT base has hash INVALID_KMER exactly as canonical_hash.
template <bool FILL>
__device__ inline uint32_t scan_read(const char* s, uint32_t len, const uint64_t* bloom, uint64_t nbits, uint64_t* out) {
    const int n_kmers = (int)len - K + 1;
    if (n_kmers < W) return 0;
    uint64_t ring[16];
    uint64_t fwd = 0, rc = 0, prev = INV;
    int valid = 0;
    uint32_t cnt = 0;
    for (uint32_t p = 0; p < len; ++p) {
        const int b = base2(s[p]);
        if (b == 255) { valid = 0; fwd = 0; rc = 0; }
        else {
            fwd = ((fwd << 2) | (uint64_t)b) & KMASK;
            rc = (rc >> 2) | ((uint64_t)(b ^ 3) << (2 * (K - 1)));
            ++valid;
        }
        if (p + 1 < (uint32_t)K) continue;
        const int i = (int)p - K + 1;                       // k-mer i ends at base p
        ring[i & 15] = (valid >= K) ? mm64(fwd < rc ? fwd : rc) : INV;
        if (i < W - 1) continue;                            // window [i-W+1, i] complete
        uint64_t wm = INV;
        for (int j = i - W + 1; j <= i; ++j) { const uint64_t v = ring[j & 15]; if (v < wm) wm = v; }
        if (wm == INV) continue;                            // window all-N
        if (wm != prev) {
            prev = wm;
            if (!bloom_contains(bloom, nbits, wm)) { if (FILL) out[cnt] = wm; ++cnt; }
        }
    }
    return cnt;
}

__global__ void k_count(const char* seq, const uint64_t* off, uint32_t n, const uint64_t* bloom, uint64_t nbits, uint32_t* np, uint64_t* np64) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const uint32_t c = scan_read<false>(seq + off[i], (uint32_t)(off[i + 1] - off[i]), bloom, nbits, nullptr);
        np[i] = c >= MIN_NONHOST_MINS ? c : 0;
        np64[i] = np[i];
    }
}
// write each read's kept minimizers, then sort them ascending (the CPU's sorted flat list order for one read)
__global__ void k_fill(const char* seq, const uint64_t* off, uint32_t n, const uint64_t* bloom, uint64_t nbits, const uint32_t* np,
                       const uint64_t* qoff, uint64_t* flat, unsigned* err) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const uint32_t c = np[i];
        if (!c) continue;
        uint64_t* o = flat + qoff[i];
        const uint32_t c2 = scan_read<true>(seq + off[i], (uint32_t)(off[i + 1] - off[i]), bloom, nbits, o);
        if (c2 != c) { atomicAdd(err, 1u); continue; }
        for (uint32_t a = 1; a < c; ++a) {                  // insertion sort
            const uint64_t v = o[a]; uint32_t b = a;
            while (b > 0 && o[b - 1] > v) { o[b] = o[b - 1]; --b; }
            o[b] = v;
        }
    }
}
struct DevIndex { const Entry* e; const uint32_t* dir; uint32_t B; };
// run [lo, lo+len) of entries equal to h (entries are sorted by hash; all of them are in bucket h >> (64-B))
__device__ inline void probe(const DevIndex& x, uint64_t h, uint32_t& lo, uint32_t& len) {
    if (x.e == nullptr) { lo = 0; len = 0; return; }
    const uint64_t b = h >> (64 - x.B);
    uint32_t i = __ldg(x.dir + b);
    const uint32_t hi = __ldg(x.dir + b + 1);
    while (i < hi && x.e[i].h < h) ++i;
    uint32_t j = i;
    while (j < hi && x.e[j].h == h) ++j;
    lo = i; len = j - i;
}
__global__ void k_probe(const uint64_t* flat, uint64_t Q, DevIndex v, DevIndex b, DevIndex f, uint32_t* lo, uint32_t* len) {
    for (uint64_t q = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; q < Q; q += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t h = flat[q];
        uint32_t l, n;
        probe(v, h, l, n); lo[q] = l; len[q] = n;
        probe(b, h, l, n); lo[Q + q] = l; len[Q + q] = n;
        probe(f, h, l, n); lo[2 * Q + q] = l; len[2 * Q + q] = n;
    }
}
// per read: occurrences in the viral map and in the microbial map (bacterial + fungal)
__global__ void k_sum(const uint32_t* np, const uint64_t* qoff, uint64_t Q, const uint32_t* len, uint32_t n, uint64_t* tot64) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        uint64_t s = 0;
        if (np[i]) for (uint64_t q = qoff[i]; q < qoff[i] + np[i]; ++q) s += (uint64_t)len[q] + len[Q + q] + len[2 * Q + q];
        tot64[i] = s;
    }
}
// one key per (read, minimizer, entry) in the CPU's insertion sequence: viral segment first (minimizers ascending, entries in file
// order), then the microbial segment (per minimizer: bacterial entries, then fungal entries - std::merge puts a before b)
__global__ void k_write(const uint32_t* np, const uint64_t* qoff, uint64_t Q, const uint32_t* lo, const uint32_t* len, DevIndex v, DevIndex b,
                        DevIndex f, uint32_t n, const uint64_t* hoff, uint64_t* keys, uint32_t* vals) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if (!np[i] || hoff[i + 1] == hoff[i]) continue;
        uint64_t pos = hoff[i];
        const uint64_t q0 = qoff[i], q1 = q0 + np[i], rd = (uint64_t)i << 33;
        for (uint64_t q = q0; q < q1; ++q)
            for (uint32_t t = 0; t < len[q]; ++t, ++pos) { keys[pos] = rd | v.e[lo[q] + t].sid; vals[pos] = (uint32_t)pos; }
        for (uint64_t q = q0; q < q1; ++q) {
            for (uint32_t t = 0; t < len[Q + q]; ++t, ++pos) { keys[pos] = rd | (1ULL << 32) | b.e[lo[Q + q] + t].sid; vals[pos] = (uint32_t)pos; }
            for (uint32_t t = 0; t < len[2 * Q + q]; ++t, ++pos) { keys[pos] = rd | (1ULL << 32) | f.e[lo[2 * Q + q] + t].sid; vals[pos] = (uint32_t)pos; }
        }
    }
}
// after sort 1 (by key, stable) every (read, map, species) group is contiguous; first occurrence = min original position
__global__ void k_first(const uint32_t* vals, const uint32_t* starts, const uint32_t* counts, uint32_t G, uint32_t* first, uint32_t* gid) {
    for (uint32_t g = blockIdx.x * blockDim.x + threadIdx.x; g < G; g += gridDim.x * blockDim.x) {
        uint32_t m = 0xffffffffu;
        for (uint32_t t = starts[g]; t < starts[g] + counts[g]; ++t) m = min(m, vals[t]);
        first[g] = m; gid[g] = g;
    }
}
__global__ void k_gather(const uint32_t* order, const uint64_t* ukey, const uint32_t* counts, uint32_t G, uint64_t* fkey, uint32_t* fcnt) {
    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < G; k += gridDim.x * blockDim.x) { fkey[k] = ukey[order[k]]; fcnt[k] = counts[order[k]]; }
}
// directory over the top B bits: dir[b] = first entry index whose bucket >= b; dir[2^B] = n
__global__ void k_dir(const Entry* e, uint64_t n, uint32_t B, uint32_t* dir, unsigned long long* unsorted) {
    const int64_t nb = 1LL << B;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i <= n; i += (uint64_t)gridDim.x * blockDim.x) {
        const int64_t cur = i < n ? (int64_t)(e[i].h >> (64 - B)) : nb;
        const int64_t prv = i > 0 ? (int64_t)(e[i - 1].h >> (64 - B)) : -1;
        for (int64_t b = prv + 1; b <= cur; ++b) dir[b] = (uint32_t)i;
        if (i > 0 && i < n && e[i].h < e[i - 1].h) atomicAdd(unsorted, 1ULL);
    }
}

// ---------------------------------------------------------------- host side
// device allocations of one call, freed on every exit path
struct Arena {
    std::vector<void*> d, h;
    void* dev(size_t bytes) { void* p = nullptr; NH_CK(cudaMalloc(&p, std::max<size_t>(bytes, 16))); d.push_back(p); return p; }
    void* pinned(size_t bytes) { void* p = nullptr; NH_CK(cudaHostAlloc(&p, std::max<size_t>(bytes, 16), cudaHostAllocDefault)); h.push_back(p); return p; }
    void release() { for (void* p : d) cudaFree(p); for (void* p : h) cudaFreeHost(p); d.clear(); h.clear(); }
    ~Arena() { release(); }
};

void pread_full(int fd, void* buf, size_t sz, off_t off, const std::string& path) {
    size_t done = 0;
    while (done < sz) {
        ssize_t r = pread(fd, (char*)buf + done, sz - done, off + (off_t)done);
        if (r <= 0) throw "read failed: " + path;
        done += (size_t)r;
    }
}
// copy `total` bytes of a file (from `base`) into device memory with nthreads reader threads (pinned bounce buffers, one stream each)
void file_to_device(const std::string& path, off_t base, uint64_t total, void* d, int device, int nthreads) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw "cannot open " + path;
    const uint64_t chunk = 64ull << 20, nch = (total + chunk - 1) / chunk;
    std::atomic<uint64_t> next(0);
    std::atomic<bool> bad(false);
    std::string why;
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; ++t) th.emplace_back([&]() {
        void* buf = nullptr; cudaStream_t st = nullptr;
        try {
            NH_CK(cudaSetDevice(device));
            NH_CK(cudaHostAlloc(&buf, chunk, cudaHostAllocDefault));
            NH_CK(cudaStreamCreate(&st));
            for (;;) {
                const uint64_t c = next++;
                if (c >= nch || bad) break;
                const uint64_t off = c * chunk, sz = std::min(chunk, total - off);
                pread_full(fd, buf, sz, base + (off_t)off, path);
                NH_CK(cudaMemcpyAsync((char*)d + off, buf, sz, cudaMemcpyHostToDevice, st));
                NH_CK(cudaStreamSynchronize(st));
            }
        } catch (const std::string& e) { if (!bad.exchange(true)) why = e; }
        if (st) cudaStreamDestroy(st);
        if (buf) cudaFreeHost(buf);
    });
    for (auto& x : th) x.join();
    close(fd);
    if (bad) throw why;
}
struct Index { uint64_t n = 0; uint32_t B = 0; Entry* e = nullptr; uint32_t* dir = nullptr; };
// the header of one .snhskidx file: its entry count; B, the directory's bits
uint64_t index_header(const std::string& path, uint32_t& Bout) {
    char hdr[32];
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw "cannot open " + path;
    try { pread_full(fd, hdr, 32, 0, path); } catch (...) { close(fd); throw; }
    close(fd);
    uint32_t ver, k, w; uint64_t n;
    memcpy(&ver, hdr + 8, 4); memcpy(&k, hdr + 12, 4); memcpy(&w, hdr + 16, 4); memcpy(&n, hdr + 24, 8);
    if (memcmp(hdr, "SNHSKIDX", 8) != 0 || (ver != 1 && ver != 2) || k != (uint32_t)K || w != (uint32_t)W)
        throw "unsupported index " + path + " (k=" + std::to_string(k) + " w=" + std::to_string(w) + ")";
    if (n >= 0xffffffffull) throw "index too large for the 32-bit directory: " + path;
    uint32_t B = 1; while ((1ull << B) < n / 4 + 1) ++B;
    Bout = std::min<uint32_t>(B, 31);
    return n;
}
// the radix directory of an index whose entries are on the device
void build_dir(Index& ix, Arena& A, const std::string& path) {
    ix.dir = (uint32_t*)A.dev(((1ull << ix.B) + 1) * 4);
    unsigned long long* d_bad = (unsigned long long*)A.dev(8);
    NH_CK(cudaMemset(d_bad, 0, 8));
    k_dir<<<4096, 256>>>(ix.e, ix.n, ix.B, ix.dir, d_bad);
    NH_CK(cudaGetLastError()); NH_CK(cudaDeviceSynchronize());
    unsigned long long unsorted = 0;
    NH_CK(cudaMemcpy(&unsorted, d_bad, 8, cudaMemcpyDeviceToHost));
    if (unsorted) throw "index not sorted by hash: " + path;
}
void load_index(const std::string& path, Index& ix, Arena& A, int device, int nthreads) {
    uint32_t B = 0;
    const uint64_t n = index_header(path, B);
    ix.n = n;
    if (n == 0) return;
    ix.B = B;
    ix.e = (Entry*)A.dev(n * sizeof(Entry));
    file_to_device(path, 32, n * sizeof(Entry), ix.e, device, nthreads);
    build_dir(ix, A, path);
}
// the host Bloom filter's header: its bit count
uint64_t bloom_header(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw "cannot open " + path;
    char hdr[32];
    try { pread_full(fd, hdr, 32, 0, path); } catch (...) { close(fd); throw; }
    close(fd);
    uint32_t ver; uint64_t nbits = 0;
    memcpy(&ver, hdr + 8, 4); memcpy(&nbits, hdr + 24, 8);
    if (memcmp(hdr, "HSKBLOOM", 8) != 0 || ver != 1 || nbits == 0) throw "unsupported Bloom filter " + path;
    return nbits;
}
unsigned ceil_log2(uint64_t v) { unsigned b = 0; while ((1ull << b) < v) ++b; return b; }

// what the classification reads on the device: the three indexes and the Bloom filter
struct DevSet { Index vi, bi, fi; uint64_t* bloom = nullptr; uint64_t nbits = 0; };

// Phases A-C for n reads whose bases are h_seq[h_off[i] .. h_off[i+1]) in page-locked host memory (n > 0)
void classify_reads(const char* h_seq, const uint64_t* h_off, uint32_t n, uint64_t bytes, const DevSet& X, Arena& A, NonhostResult& out) {
    out.np.assign(n, 0);
    char* d_seq = (char*)A.dev(bytes);
    uint64_t* d_off = (uint64_t*)A.dev((n + 1) * 8ull);
    NH_CK(cudaMemcpy(d_seq, h_seq, bytes, cudaMemcpyHostToDevice));
    NH_CK(cudaMemcpy(d_off, h_off, (n + 1) * 8ull, cudaMemcpyHostToDevice));
    DevIndex dv{X.vi.e, X.vi.dir, X.vi.B}, db{X.bi.e, X.bi.dir, X.bi.B}, df{X.fi.e, X.fi.dir, X.fi.B};
    const uint64_t* d_bloom = X.bloom;
    const uint64_t nbits = X.nbits;
    auto scan = [&](uint64_t* in, uint64_t* o, int cnt) {
        size_t tb = 0; NH_CK(cub::DeviceScan::ExclusiveSum(nullptr, tb, in, o, cnt));
        void* tmp = A.dev(tb); NH_CK(cub::DeviceScan::ExclusiveSum(tmp, tb, in, o, cnt));
    };
    // ---- extract + host subtraction
    unsigned* d_err = (unsigned*)A.dev(16); NH_CK(cudaMemset(d_err, 0, 16));
    uint32_t* d_np = (uint32_t*)A.dev(n * 4ull);
    uint64_t* d_np64 = (uint64_t*)A.dev((n + 1) * 8ull);
    uint64_t* d_qoff = (uint64_t*)A.dev((n + 1) * 8ull);
    const int blk = 256, grd = (int)std::min<uint64_t>(((uint64_t)n + blk - 1) / blk, 1u << 20);
    k_count<<<grd, blk>>>(d_seq, d_off, n, d_bloom, nbits, d_np, d_np64);
    NH_CK(cudaGetLastError());
    NH_CK(cudaMemset(d_np64 + n, 0, 8));
    scan(d_np64, d_qoff, (int)n + 1);
    uint64_t Q = 0; NH_CK(cudaMemcpy(&Q, d_qoff + n, 8, cudaMemcpyDeviceToHost));
    uint64_t* d_flat = (uint64_t*)A.dev(Q * 8);
    k_fill<<<grd, blk>>>(d_seq, d_off, n, d_bloom, nbits, d_np, d_qoff, d_flat, d_err);
    NH_CK(cudaGetLastError());
    // ---- probe the three indexes
    uint32_t* d_lo = (uint32_t*)A.dev(Q * 12), *d_len = (uint32_t*)A.dev(Q * 12);
    if (Q) k_probe<<<(int)std::min<uint64_t>((Q + blk - 1) / blk, 1u << 22), blk>>>(d_flat, Q, dv, db, df, d_lo, d_len);
    NH_CK(cudaGetLastError());
    uint64_t* d_tot = (uint64_t*)A.dev((n + 1) * 8ull);
    uint64_t* d_hoff = (uint64_t*)A.dev((n + 1) * 8ull);
    k_sum<<<grd, blk>>>(d_np, d_qoff, Q, d_len, n, d_tot);
    NH_CK(cudaGetLastError());
    NH_CK(cudaMemset(d_tot + n, 0, 8));
    scan(d_tot, d_hoff, (int)n + 1);
    uint64_t H = 0; NH_CK(cudaMemcpy(&H, d_hoff + n, 8, cudaMemcpyDeviceToHost));
    if (H >= (1ull << 31)) throw std::string("more than 2^31 index hits in one sample");
    // ---- expand to one key per occurrence, sort, group, restore first-insertion order
    uint32_t G = 0;
    uint64_t* d_fkey = nullptr; uint32_t* d_fcnt = nullptr;
    if (H) {
        cub::DoubleBuffer<uint64_t> keys((uint64_t*)A.dev(H * 8), (uint64_t*)A.dev(H * 8));
        cub::DoubleBuffer<uint32_t> vals((uint32_t*)A.dev(H * 4), (uint32_t*)A.dev(H * 4));
        k_write<<<grd, blk>>>(d_np, d_qoff, Q, d_lo, d_len, dv, db, df, n, d_hoff, keys.Current(), vals.Current());
        NH_CK(cudaGetLastError());
        size_t tb = 0;
        NH_CK(cub::DeviceRadixSort::SortPairs(nullptr, tb, keys, vals, (int)H, 0, 64));
        NH_CK(cub::DeviceRadixSort::SortPairs(A.dev(tb), tb, keys, vals, (int)H, 0, 64));
        uint64_t* d_ukey = (uint64_t*)A.dev(H * 8);
        uint32_t* d_counts = (uint32_t*)A.dev(H * 4);
        uint32_t* d_nruns = (uint32_t*)A.dev(16);
        tb = 0;
        NH_CK(cub::DeviceRunLengthEncode::Encode(nullptr, tb, keys.Current(), d_ukey, d_counts, d_nruns, (int)H));
        NH_CK(cub::DeviceRunLengthEncode::Encode(A.dev(tb), tb, keys.Current(), d_ukey, d_counts, d_nruns, (int)H));
        NH_CK(cudaMemcpy(&G, d_nruns, 4, cudaMemcpyDeviceToHost));
        uint32_t* d_starts = (uint32_t*)A.dev((G + 1ull) * 4);
        tb = 0;
        NH_CK(cub::DeviceScan::ExclusiveSum(nullptr, tb, d_counts, d_starts, (int)G));
        NH_CK(cub::DeviceScan::ExclusiveSum(A.dev(tb), tb, d_counts, d_starts, (int)G));
        cub::DoubleBuffer<uint32_t> first((uint32_t*)A.dev(G * 4ull), (uint32_t*)A.dev(G * 4ull));
        cub::DoubleBuffer<uint32_t> gid((uint32_t*)A.dev(G * 4ull), (uint32_t*)A.dev(G * 4ull));
        k_first<<<(int)((G + blk - 1) / blk), blk>>>(vals.Current(), d_starts, d_counts, G, first.Current(), gid.Current());
        NH_CK(cudaGetLastError());
        tb = 0;
        const int bits = (int)std::max(1u, ceil_log2(H));
        NH_CK(cub::DeviceRadixSort::SortPairs(nullptr, tb, first, gid, (int)G, 0, bits));
        NH_CK(cub::DeviceRadixSort::SortPairs(A.dev(tb), tb, first, gid, (int)G, 0, bits));
        d_fkey = (uint64_t*)A.dev(G * 8ull);
        d_fcnt = (uint32_t*)A.dev(G * 4ull);
        k_gather<<<(int)((G + blk - 1) / blk), blk>>>(gid.Current(), d_ukey, d_counts, G, d_fkey, d_fcnt);
        NH_CK(cudaGetLastError());
    }
    // ---- results to the host
    NH_CK(cudaMemcpy(out.np.data(), d_np, n * 4ull, cudaMemcpyDeviceToHost));
    out.key.resize(G); out.cnt.resize(G);
    if (G) {
        NH_CK(cudaMemcpy(out.key.data(), d_fkey, G * 8ull, cudaMemcpyDeviceToHost));
        NH_CK(cudaMemcpy(out.cnt.data(), d_fcnt, G * 4ull, cudaMemcpyDeviceToHost));
    }
    unsigned herr = 0; NH_CK(cudaMemcpy(&herr, d_err, 4, cudaMemcpyDeviceToHost));
    if (herr) throw std::to_string(herr) + " reads gave a different minimizer count in the fill pass";
    out.nMinimizers = Q; out.nHits = H; out.nGroups = G;
}

// ---------------------------------------------------------------- E21: the gpu-pool's resident non-host server
// `singlet gpu-pool` (src/pipeline/gpu_pool.cpp, docs/gpu-engine/fable/E21-pool.md) forks one server per pool before any sample: it keeps
// the three indexes and the Bloom filter in page-locked host memory for the pool's life and owns a CUDA context on the pool's device.
// A sample's process (SINGLET_NONHOST_SERVER = the server's Unix socket) sends its reads while it holds its device slot (the lock site in
// the pipeline is unchanged); the server copies the index to the device at PCIe speed (no reader threads, no page cache, no context of
// the sample's own), runs the same kernels (classify_reads) and returns the same three arrays. Between requests the device copy is
// freed, unless another request is already waiting, so the device memory a slot sees is what it was. Any failure: the sample loads
// the index from the files itself (the code path above), so the outputs never depend on the server.
namespace srv {
const char kReq[8] = {'S', 'N', 'H', 'Q', '0', '0', '0', '1'};
const char kRep[8] = {'S', 'N', 'H', 'R', '0', '0', '0', '1'};

bool wr(int fd, const void* p, size_t n) {
    const char* c = (const char*)p;
    while (n) {
        const ssize_t r = send(fd, c, n, MSG_NOSIGNAL);   // a peer that went away: an error, never SIGPIPE (the sample's process)
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        c += r; n -= (size_t)r;
    }
    return true;
}
bool rd(int fd, void* p, size_t n) {
    char* c = (char*)p;
    while (n) {
        const ssize_t r = read(fd, c, n);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        c += r; n -= (size_t)r;
    }
    return true;
}
bool wrs(int fd, const std::string& s) { const uint32_t l = (uint32_t)s.size(); return wr(fd, &l, 4) && wr(fd, s.data(), l); }
bool rds(int fd, std::string& s) {
    uint32_t l = 0;
    if (!rd(fd, &l, 4) || l > (1u << 20)) return false;
    s.assign(l, '\0');
    return l == 0 || rd(fd, &s[0], l);
}
// a file's identity: the request's index files must be the server's (path, device, inode, size, modification time)
struct FileId {
    std::string path; dev_t dev = 0; ino_t ino = 0; off_t size = 0; time_t mtime = 0;
    bool get(const std::string& p) {
        path = p;
        if (p.empty()) return true;
        struct stat st{};
        if (stat(p.c_str(), &st) != 0) return false;
        dev = st.st_dev; ino = st.st_ino; size = st.st_size; mtime = st.st_mtime;
        return true;
    }
    bool same(const FileId& o) const { return path == o.path && dev == o.dev && ino == o.ino && size == o.size && mtime == o.mtime; }
};
// one index in page-locked host memory
struct HostIndex { FileId id; uint64_t n = 0; uint32_t B = 0; Entry* h = nullptr; };
void read_pinned(const std::string& path, off_t base, uint64_t total, char* dst, int threads) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw "cannot open " + path;
    const uint64_t piece = 64ull << 20, np = (total + piece - 1) / piece;
    std::atomic<uint64_t> next(0);
    std::atomic<bool> bad(false);
    std::vector<std::thread> th;
    for (int t = 0; t < std::max(1, threads); ++t) th.emplace_back([&]() {
        for (uint64_t c; !bad && (c = next++) < np;) {
            const uint64_t off = c * piece, sz = std::min(piece, total - off);
            try { pread_full(fd, dst + off, sz, base + (off_t)off, path); } catch (const std::string&) { bad = true; }
        }
    });
    for (auto& x : th) x.join();
    close(fd);
    if (bad) throw "read failed: " + path;
}
struct Resident {
    HostIndex v, b, f;
    FileId bloomId; uint64_t nbits = 0, nw = 0; uint64_t* bloom = nullptr;
    bool ok = false; double seconds = 0; uint64_t bytes = 0;
    void clear() {
        for (HostIndex* x : {&v, &b, &f}) { if (x->h) cudaFreeHost(x->h); *x = HostIndex(); }
        if (bloom) cudaFreeHost(bloom);
        bloom = nullptr; nbits = nw = 0; bloomId = FileId(); ok = false; bytes = 0;
    }
    bool matches(const NonhostPaths& p) const {
        FileId a, c, d, e;
        return ok && a.get(p.viral) && c.get(p.bacterial) && d.get(p.fungal) && e.get(p.bloom)
               && a.same(v.id) && c.same(b.id) && d.same(f.id) && e.same(bloomId);
    }
    void load(const NonhostPaths& p, int threads) {
        clear();
        const double t0 = now();
        auto one = [&](const std::string& path, HostIndex& x) {
            if (!x.id.get(path)) throw "cannot stat " + path;
            if (path.empty()) return;
            x.n = index_header(path, x.B);
            if (x.n == 0) return;
            NH_CK(cudaHostAlloc((void**)&x.h, x.n * sizeof(Entry), cudaHostAllocDefault));
            read_pinned(path, 32, x.n * sizeof(Entry), (char*)x.h, threads);
            bytes += x.n * sizeof(Entry);
        };
        one(p.viral, v); one(p.bacterial, b); one(p.fungal, f);
        if (p.bloom.empty() || !bloomId.get(p.bloom)) throw "cannot stat the Bloom filter " + p.bloom;
        nbits = bloom_header(p.bloom);
        nw = (nbits + 63) / 64;
        NH_CK(cudaHostAlloc((void**)&bloom, nw * 8, cudaHostAllocDefault));
        read_pinned(p.bloom, 32, nw * 8, (char*)bloom, std::min(threads, 4));
        bytes += nw * 8;
        seconds = now() - t0;
        ok = true;
    }
};
// the resident index's copy on the device (freed between requests unless one is waiting)
struct OnDevice {
    Arena A; DevSet X; bool on = false;
    void drop() { A.release(); X = DevSet(); on = false; }
    double upload(const Resident& R) {
        const double t0 = now();
        drop();
        auto one = [&](const HostIndex& h, Index& ix) {
            ix.n = h.n; ix.B = h.B;
            if (h.n == 0) return;
            ix.e = (Entry*)A.dev(h.n * sizeof(Entry));
            NH_CK(cudaMemcpy(ix.e, h.h, h.n * sizeof(Entry), cudaMemcpyHostToDevice));
            build_dir(ix, A, h.id.path);
        };
        one(R.v, X.vi); one(R.b, X.bi); one(R.f, X.fi);
        X.nbits = R.nbits;
        X.bloom = (uint64_t*)A.dev(R.nw * 8);
        NH_CK(cudaMemcpy(X.bloom, R.bloom, R.nw * 8, cudaMemcpyHostToDevice));
        NH_CK(cudaDeviceSynchronize());
        on = true;
        return now() - t0;
    }
};
}   // namespace srv

// the client half: true with `out` filled when the server answered; false (why) on any failure
bool nonhostRemote(const char* sock, const std::vector<std::string>& reads, const NonhostPaths& paths, int device, NonhostResult& out, std::string& why) {
    using namespace srv;
    const double t0 = now();
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) { why = std::string("socket: ") + strerror(errno); return false; }
    struct Closer { int fd; ~Closer() { close(fd); } } closer{fd};
    struct sockaddr_un a{};
    a.sun_family = AF_UNIX;
    if (strlen(sock) >= sizeof a.sun_path) { why = "socket path too long"; return false; }
    strcpy(a.sun_path, sock);
    struct timeval tv{900, 0};   // a server that stops answering: the sample loads the index itself
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    if (connect(fd, (struct sockaddr*)&a, sizeof a) != 0) { why = std::string("connect ") + sock + ": " + strerror(errno); return false; }
    const uint32_t n = (uint32_t)reads.size();
    std::vector<uint64_t> off(n + 1ull);
    off[0] = 0;
    for (uint32_t i = 0; i < n; ++i) off[i + 1] = off[i] + reads[i].size();
    const uint64_t bytes = off[n];
    const uint32_t dev = (uint32_t)device;
    bool ok = wr(fd, kReq, 8) && wr(fd, &dev, 4) && wrs(fd, paths.viral) && wrs(fd, paths.bacterial) && wrs(fd, paths.fungal) && wrs(fd, paths.bloom)
              && wr(fd, &n, 4) && wr(fd, &bytes, 8) && wr(fd, off.data(), (n + 1ull) * 8);
    {
        std::vector<char> buf;
        buf.reserve(8u << 20);
        for (uint32_t i = 0; ok && i < n; ++i) {
            buf.insert(buf.end(), reads[i].begin(), reads[i].end());
            if (buf.size() >= (8u << 20) || i + 1 == n) { ok = wr(fd, buf.data(), buf.size()); buf.clear(); }
        }
    }
    if (!ok) { why = std::string("sending the reads: ") + strerror(errno); return false; }
    const double tSent = now();
    char mg[8]; int32_t status = -1; std::string log;
    if (!rd(fd, mg, 8) || memcmp(mg, kRep, 8) != 0 || !rd(fd, &status, 4) || !rds(fd, log)) { why = "no answer from the server"; return false; }
    if (status != 0) { why = "the server declined: " + log; return false; }
    uint64_t q[4]; double s[2]; uint8_t kept = 0; int32_t spid = 0;
    if (!rd(fd, q, sizeof q) || !rd(fd, s, sizeof s) || !rd(fd, &kept, 1) || !rd(fd, &spid, 4)) { why = "short answer"; return false; }
    if (q[2] >= (1ull << 31)) { why = "bad answer"; return false; }
    out.np.resize(n); out.key.resize(q[2]); out.cnt.resize(q[2]);
    if (!rd(fd, out.np.data(), n * 4ull) || (q[2] && (!rd(fd, out.key.data(), q[2] * 8) || !rd(fd, out.cnt.data(), q[2] * 4)))) {
        why = "short answer (results)"; return false;
    }
    out.nMinimizers = q[0]; out.nHits = q[1]; out.nGroups = q[2]; out.indexBytes = q[3];
    const double t1 = now();
    out.secLoad = s[0];
    out.secClassify = (t1 - t0) - s[0];
    fprintf(stderr, "[nonhost] engine_gpu: served by gpu-pool's non-host server (pid %d, %s): index on the device in %.2f s (%s), "
            "%u reads sent in %.2f s, classified in %.2f s, %.2f s in all\n", (int)spid, log.c_str(), s[0],
            kept ? "kept from the previous request" : "from page-locked host memory", n, tSent - t0, s[1], t1 - t0);
    return true;
}

} // namespace

namespace singlet_engine_gpu {

bool nonhostClassify(const std::vector<std::string>& reads, const NonhostPaths& paths, int device, int loadThreads, NonhostResult& out) {
    out = NonhostResult();
    if (const char* sock = std::getenv("SINGLET_NONHOST_SERVER"); sock != nullptr && *sock && !reads.empty()
        && reads.size() < (1ull << 31)) {   // E21: the gpu-pool's resident index
        std::string why;
        if (nonhostRemote(sock, reads, paths, device, out, why)) return true;
        fprintf(stderr, "[nonhost] engine_gpu: gpu-pool's non-host server not used (%s); the index is loaded here\n", why.c_str());
        out = NonhostResult();
    }
    const double t0 = now();
    try {
        if (reads.size() >= (1ull << 31)) throw std::string("too many reads for this version");
        cudaInitDevice(device, cudaDeviceScheduleBlockingSync, 0);   // before the context exists: waiting threads sleep, not spin
        NH_CK(cudaSetDevice(device));
        Arena A;
        // ---- the resident index (viral, bacterial, fungal) and the host Bloom filter
        DevSet X;
        if (!paths.viral.empty()) load_index(paths.viral, X.vi, A, device, loadThreads);
        if (!paths.bacterial.empty()) load_index(paths.bacterial, X.bi, A, device, loadThreads);
        if (!paths.fungal.empty()) load_index(paths.fungal, X.fi, A, device, loadThreads);
        {
            X.nbits = bloom_header(paths.bloom);
            const uint64_t nw = (X.nbits + 63) / 64;
            X.bloom = (uint64_t*)A.dev(nw * 8);
            file_to_device(paths.bloom, 32, nw * 8, X.bloom, device, std::min(loadThreads, 4));
        }
        out.secLoad = now() - t0;
        out.indexBytes = (X.vi.n + X.bi.n + X.fi.n) * sizeof(Entry);
        const double t1 = now();
        // ---- reads to the device
        const uint32_t n = (uint32_t)reads.size();
        out.np.assign(n, 0);
        if (n == 0) { out.secClassify = now() - t1; A.release(); cudaDeviceReset(); return true; }
        uint64_t bytes = 0;
        for (const auto& r : reads) bytes += r.size();
        char* h_seq = (char*)A.pinned(bytes);
        uint64_t* h_off = (uint64_t*)A.pinned((n + 1) * 8ull);
        h_off[0] = 0;
        for (uint32_t i = 0; i < n; ++i) { memcpy(h_seq + h_off[i], reads[i].data(), reads[i].size()); h_off[i + 1] = h_off[i] + reads[i].size(); }
        classify_reads(h_seq, h_off, n, bytes, X, A, out);
        out.secClassify = now() - t1;
        A.release();
    } catch (const std::string& e) {
        out.log = "engine_gpu nonhost: " + e;
        out.np.clear(); out.key.clear(); out.cnt.clear();
        cudaDeviceReset();
        return false;
    }
    cudaDeviceReset();   // release the context and the 30 GB before the next sample's process may take the device
    return true;
}

// E21: the server loop (namespace srv above). Binds sock, writes one byte to readyFd (then closes it), creates the context, reads the `warm`
// set into page-locked memory (when its Bloom path is given) and serves until it is killed. A request for other files is declined (the
// sample then loads its own index) unless nothing is resident yet; a CUDA failure other than an allocation ends the server (exit 3).
int nonhostServe(const std::string& sock, int device, int loadThreads, const NonhostPaths& warm, int readyFd) {
    using namespace srv;
    signal(SIGPIPE, SIG_IGN);
    int ls = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    struct sockaddr_un a{};
    a.sun_family = AF_UNIX;
    if (ls < 0 || sock.size() >= sizeof a.sun_path) { fprintf(stderr, "[nhsrv] socket: %s\n", strerror(errno)); return 2; }
    strcpy(a.sun_path, sock.c_str());
    unlink(sock.c_str());
    if (bind(ls, (struct sockaddr*)&a, sizeof a) != 0 || listen(ls, 128) != 0) {
        fprintf(stderr, "[nhsrv] bind %s: %s\n", sock.c_str(), strerror(errno));
        return 2;
    }
    chmod(sock.c_str(), 0600);
    if (readyFd >= 0) { const char c = 1; if (write(readyFd, &c, 1) != 1) { } close(readyFd); }
    fprintf(stderr, "[nhsrv] pid %d, device %d, socket %s\n", (int)getpid(), device, sock.c_str());
    Resident R;
    OnDevice D;
    try {
        cudaInitDevice(device, cudaDeviceScheduleBlockingSync, 0);
        NH_CK(cudaSetDevice(device));
        NH_CK(cudaFree(nullptr));   // the context, once
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        fprintf(stderr, "[nhsrv] context on device %d: %.2f GB of %.2f GB in use\n", device, (tot - fr) / 1073741824.0, tot / 1073741824.0);
        if (!warm.bloom.empty()) {
            R.load(warm, loadThreads);
            fprintf(stderr, "[nhsrv] resident: %.0f MB in page-locked host memory in %.2f s (viral %s, bacterial %s, fungal %s, Bloom %s)\n",
                    R.bytes / 1048576.0, R.seconds, warm.viral.c_str(), warm.bacterial.c_str(), warm.fungal.c_str(), warm.bloom.c_str());
        }
    } catch (const std::string& e) {
        fprintf(stderr, "[nhsrv] start-up: %s (the first request's files are loaded instead)\n", e.c_str());
        R.clear();
    }
    uint64_t served = 0;
    for (;;) {
        int c = accept4(ls, nullptr, nullptr, SOCK_CLOEXEC);
        if (c < 0) { if (errno == EINTR) continue; fprintf(stderr, "[nhsrv] accept: %s\n", strerror(errno)); return 2; }
        struct timeval tv{300, 0};
        setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        const double t0 = now();
        Arena Wk;
        std::string err;
        bool fatal = false;
        NonhostResult res;
        double up = 0, cl = 0;
        const bool wasOn = D.on;
        char mg[8]; uint32_t dev = 0, n = 0; uint64_t bytes = 0;
        NonhostPaths p;
        if (!rd(c, mg, 8) || memcmp(mg, kReq, 8) != 0 || !rd(c, &dev, 4) || !rds(c, p.viral) || !rds(c, p.bacterial) || !rds(c, p.fungal)
            || !rds(c, p.bloom) || !rd(c, &n, 4) || !rd(c, &bytes, 8)) err = "bad request";
        else if ((int)dev != device) err = "the sample's device " + std::to_string(dev) + " is not the server's " + std::to_string(device);
        else if (n == 0) err = "no reads";
        try {
            if (err.empty() && !R.matches(p)) {
                if (R.ok) err = "other index files than the resident ones (" + p.viral + ", " + p.bacterial + ", " + p.fungal + ", " + p.bloom + ")";
                else {
                    D.drop();
                    R.load(p, loadThreads);
                    fprintf(stderr, "[nhsrv] resident (first request): %.0f MB in page-locked host memory in %.2f s\n", R.bytes / 1048576.0, R.seconds);
                }
            }
            if (err.empty()) {
                char* h_seq = (char*)Wk.pinned(bytes);
                uint64_t* h_off = (uint64_t*)Wk.pinned((n + 1) * 8ull);
                if (!rd(c, h_off, (n + 1) * 8ull) || !rd(c, h_seq, bytes) || h_off[0] != 0 || h_off[n] != bytes) err = "reads not received";
                else {
                    if (!D.on) up = D.upload(R);
                    const double t1 = now();
                    classify_reads(h_seq, h_off, n, bytes, D.X, Wk, res);
                    cl = now() - t1;
                    res.indexBytes = (D.X.vi.n + D.X.bi.n + D.X.fi.n) * sizeof(Entry);
                }
            }
        } catch (const std::string& e) {
            err = e;
            const bool oom = e.find("cudaErrorMemoryAllocation") != std::string::npos || e.find("out of memory") != std::string::npos;
            fatal = !oom && e.find("cuda") != std::string::npos;
            D.drop();
        } catch (const std::exception& e) {
            err = std::string("host: ") + e.what();
            D.drop();
        }
        Wk.release();
        if (err.empty()) {
            const uint64_t q[4] = {res.nMinimizers, res.nHits, res.nGroups, res.indexBytes};
            const double s[2] = {up, cl};
            const uint8_t k8 = wasOn ? 1 : 0;
            const int32_t spid = (int32_t)getpid(), st = 0;
            const std::string lg = "request " + std::to_string(served + 1);
            const bool sent = wr(c, kRep, 8) && wr(c, &st, 4) && wrs(c, lg) && wr(c, q, sizeof q) && wr(c, s, sizeof s) && wr(c, &k8, 1)
                              && wr(c, &spid, 4) && wr(c, res.np.data(), n * 4ull)
                              && (res.nGroups == 0 || (wr(c, res.key.data(), res.nGroups * 8) && wr(c, res.cnt.data(), res.nGroups * 4)));
            ++served;
            fprintf(stderr, "[nhsrv] request %llu: %u reads, index %s %.2f s, classified %.2f s, %.2f s in all%s\n", (unsigned long long)served, n,
                    wasOn ? "kept" : "uploaded", up, cl, now() - t0, sent ? "" : " (the answer was not delivered)");
        } else {
            const int32_t st = 1;
            if (wr(c, kRep, 8) && wr(c, &st, 4)) wrs(c, err);
            fprintf(stderr, "[nhsrv] declined a request: %s%s\n", err.c_str(), fatal ? " (CUDA failure: the server stops)" : "");
        }
        close(c);
        if (fatal) { close(ls); unlink(sock.c_str()); return 3; }
        struct pollfd q = {ls, POLLIN, 0};   // another sample already waiting: keep the index on the device for it
        if (D.on && poll(&q, 1, 0) <= 0) D.drop();
    }
}

} // namespace singlet_engine_gpu
