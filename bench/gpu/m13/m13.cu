// m13.cu - W11 milestone M13 ("host step 2"): the non-host (viral / bacterial / fungal) read screen on the GPU.
//
// What it replaces: Phases A-C of NonHostScreener::classify_multi_batch_filtered (prod/campaign c4a1be38):
//   minimizer extraction (k = 21, w = 11), host-Bloom subtraction, the sorted-minimizer x sorted-index join, and the per-read
//   species hit maps. It writes the per-read np and the species hit counts in the CPU's first-insertion order
//   (the file m13_dev.h in m13.patch loads); the unchanged host code then runs Phase D, the EM and all outputs.
// Index: the three .snhskidx files (16-byte {hash, species} entries sorted by hash) are copied as they are into HBM and given a
//   radix directory over the top B hash bits (about 4 entries per bucket); a probe is one directory read plus a short scan.
//   Ideas re-implemented, no code copied, from the design-input session (read only):
//   singlet-agents/.claude/worktrees/gpu-align-vision/bench/gpu-align/a3_hash.cu (random probes into a tens-of-GB table, 11.6 G/s),
//   a4_sort.cu (cub radix sort, 11 G pairs/s) and b1_kmer.cu (rolling canonical 2-bit k-mers).
// Exactness: all integer. No floating point on the GPU (rates, thresholds and the EM stay on the host).
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -o m13 m13.cu -lpthread
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static constexpr int K = 21, W = 11;
static constexpr uint64_t INV = ~0ULL;
static constexpr uint64_t KMASK = (1ULL << (2 * K)) - 1;
static constexpr uint32_t MIN_NONHOST_MINS = 5;   // NonHostScreener::kMinNonhostMins

struct __align__(16) Entry { uint64_t h; uint32_t sid; uint32_t pad; };

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

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
// extract_minimizers (min_sketch.h) followed by the host-filter: returns the number of non-host minimizers; FILL writes them
// in window order. Rolling forward / reverse-complement 2-bit k-mers equal encode_kmer / rc_kmer of each window; a window with
// any non-ACGT base has hash INVALID_KMER exactly as canonical_hash.
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
// per read: occurrences in the viral map (cv) and in the microbial map = bacterial + fungal (cm)
__global__ void k_sum(const uint32_t* np, const uint64_t* qoff, uint64_t Q, const uint32_t* len, uint32_t n, uint64_t* tot64) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        uint64_t s = 0;
        if (np[i]) for (uint64_t q = qoff[i]; q < qoff[i] + np[i]; ++q) s += (uint64_t)len[q] + len[Q + q] + len[2 * Q + q];
        tot64[i] = s;
    }
}
// write one key per (read, minimizer, entry) in the CPU's insertion sequence: viral segment first (minimizers ascending, entries in
// file order), then the microbial segment (per minimizer: bacterial entries, then fungal entries - std::merge puts a before b)
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
// after sort 1 (by key, stable) every (read, db, species) group is contiguous; first occurrence = min original position
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

// ---------------------------------------------------------------- host: index and bloom
struct Index { std::string name; uint64_t n = 0; uint32_t B = 0; Entry* e = nullptr; uint32_t* dir = nullptr; uint64_t unsorted = 0; };

static void pread_full(int fd, void* buf, size_t sz, off_t off) {
    size_t done = 0;
    while (done < sz) {
        ssize_t r = pread(fd, (char*)buf + done, sz - done, off + done);
        if (r <= 0) { fprintf(stderr, "pread failed (%zd) at %lld\n", r, (long long)(off + done)); exit(3); }
        done += (size_t)r;
    }
}
// copy `total` bytes of a file (from `base`) into device memory with nthreads reader threads (pinned bounce buffers, one stream each)
static void file_to_device(const std::string& path, off_t base, uint64_t total, void* d, int nthreads, bool dropcache) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror(path.c_str()); exit(3); }
    if (dropcache) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    const uint64_t chunk = 64ull << 20, nch = (total + chunk - 1) / chunk;
    std::atomic<uint64_t> next(0);
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; ++t) th.emplace_back([&]() {
        CK(cudaSetDevice(0));
        void* buf; CK(cudaHostAlloc(&buf, chunk, cudaHostAllocDefault));
        cudaStream_t st; CK(cudaStreamCreate(&st));
        for (;;) {
            const uint64_t c = next++;
            if (c >= nch) break;
            const uint64_t off = c * chunk, sz = std::min(chunk, total - off);
            pread_full(fd, buf, sz, base + (off_t)off);
            CK(cudaMemcpyAsync((char*)d + off, buf, sz, cudaMemcpyHostToDevice, st));
            CK(cudaStreamSynchronize(st));
        }
        CK(cudaStreamDestroy(st)); CK(cudaFreeHost(buf));
    });
    for (auto& x : th) x.join();
    close(fd);
}
static void load_index(const std::string& path, Index& ix, int nthreads, bool dropcache, double& t_read, double& t_dir) {
    char hdr[32];
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror(path.c_str()); exit(3); }
    pread_full(fd, hdr, 32, 0); close(fd);
    uint32_t ver, k, w; uint64_t n;
    memcpy(&ver, hdr + 8, 4); memcpy(&k, hdr + 12, 4); memcpy(&w, hdr + 16, 4); memcpy(&n, hdr + 24, 8);
    if (memcmp(hdr, "SNHSKIDX", 8) != 0 || k != (uint32_t)K || w != (uint32_t)W) { fprintf(stderr, "bad index %s (k=%u w=%u)\n", path.c_str(), k, w); exit(3); }
    if (n >= 0xffffffffull) { fprintf(stderr, "index too large for 32-bit directory\n"); exit(3); }
    ix.n = n;
    uint32_t B = 1; while ((1ull << B) < n / 4 + 1) ++B;
    ix.B = std::min<uint32_t>(B, 31);
    double t0 = now_s();
    CK(cudaMalloc(&ix.e, n * sizeof(Entry)));
    file_to_device(path, 32, n * sizeof(Entry), ix.e, nthreads, dropcache);
    t_read = now_s() - t0;
    t0 = now_s();
    CK(cudaMalloc(&ix.dir, ((1ull << ix.B) + 1) * 4));
    unsigned long long* d_bad; CK(cudaMalloc(&d_bad, 8)); CK(cudaMemset(d_bad, 0, 8));
    k_dir<<<4096, 256>>>(ix.e, n, ix.B, ix.dir, d_bad);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&ix.unsorted, d_bad, 8, cudaMemcpyDeviceToHost)); cudaFree(d_bad);
    t_dir = now_s() - t0;
}
static void free_index(Index& ix) { cudaFree(ix.e); cudaFree(ix.dir); ix.e = nullptr; ix.dir = nullptr; }

// ---------------------------------------------------------------- host: scratch buffers that grow once (warm-up) and are reused
struct Buf {
    void* p = nullptr; size_t cap = 0;
    void* get(size_t bytes) {
        if (bytes < 16) bytes = 16;
        if (bytes > cap) { if (p) cudaFree(p); CK(cudaMalloc(&p, bytes)); cap = bytes; }
        return p;
    }
};
struct HBuf {
    void* p = nullptr; size_t cap = 0;
    void* get(size_t bytes) {
        if (bytes < 16) bytes = 16;
        if (bytes > cap) { if (p) cudaFreeHost(p); CK(cudaHostAlloc(&p, bytes, cudaHostAllocDefault)); cap = bytes; }
        return p;
    }
};
static unsigned ceil_log2(uint64_t v) { unsigned b = 0; while ((1ull << b) < v) ++b; return b; }

struct Work {
    Buf seq, off, np, np64, qoff, flat, lo, len, tot, hoff, keys0, keys1, vals0, vals1, ukey, counts, starts, first0, first1, gid0, gid1,
        fkey, fcnt, tmp, err, nruns;
    HBuf h_np, h_fkey, h_fcnt;
    size_t bytes() const {
        return seq.cap + off.cap + np.cap + np64.cap + qoff.cap + flat.cap + lo.cap + len.cap + tot.cap + hoff.cap + keys0.cap + keys1.cap +
               vals0.cap + vals1.cap + ukey.cap + counts.cap + starts.cap + first0.cap + first1.cap + gid0.cap + gid1.cap + fkey.cap + fcnt.cap +
               tmp.cap + err.cap + nruns.cap;
    }
};
struct Stats { uint64_t Q = 0, H = 0, G = 0; double ms_h2d = 0, ms_extract = 0, ms_probe = 0, ms_expand = 0, ms_d2h = 0, ms_total = 0; };

static void dev_scan(Work& w, uint64_t* in, uint64_t* out, int n) {   // exclusive sum of u64
    size_t tb = 0; CK(cub::DeviceScan::ExclusiveSum(nullptr, tb, in, out, n));
    CK(cub::DeviceScan::ExclusiveSum(w.tmp.get(tb), tb, in, out, n));
}

struct Host {   // pinned input
    char* seq = nullptr; uint64_t* off = nullptr; uint32_t n = 0; uint64_t bytes = 0;
};

// One full pass: H2D of the reads, classification, D2H of the per-read results. Returns the stats; results land in w.h_*.
static Stats classify(Work& w, const Host& in, const uint64_t* bloom, uint64_t nbits, const Index& vi, const Index& bi, const Index& fi) {
    Stats s;
    cudaEvent_t ev[7]; for (auto& e : ev) CK(cudaEventCreate(&e));
    const uint32_t n = in.n;
    DevIndex dv{vi.e, vi.dir, vi.B}, db{bi.e, bi.dir, bi.B}, df{fi.e, fi.dir, fi.B};
    CK(cudaEventRecord(ev[0]));
    char* d_seq = (char*)w.seq.get(in.bytes);
    uint64_t* d_off = (uint64_t*)w.off.get((n + 1) * 8ull);
    CK(cudaMemcpyAsync(d_seq, in.seq, in.bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpyAsync(d_off, in.off, (n + 1) * 8ull, cudaMemcpyHostToDevice));
    CK(cudaEventRecord(ev[1]));
    // ---- extract + host subtraction
    unsigned* d_err = (unsigned*)w.err.get(16); CK(cudaMemsetAsync(d_err, 0, 16));
    uint32_t* d_np = (uint32_t*)w.np.get(n * 4ull);
    uint64_t* d_np64 = (uint64_t*)w.np64.get((n + 1) * 8ull);
    uint64_t* d_qoff = (uint64_t*)w.qoff.get((n + 1) * 8ull);
    const int blk = 256, grd = (int)std::min<uint64_t>(((uint64_t)n + blk - 1) / blk, 1u << 20);
    k_count<<<grd, blk>>>(d_seq, d_off, n, bloom, nbits, d_np, d_np64);
    CK(cudaMemsetAsync(d_np64 + n, 0, 8));
    dev_scan(w, d_np64, d_qoff, (int)n + 1);
    uint64_t Q = 0; CK(cudaMemcpy(&Q, d_qoff + n, 8, cudaMemcpyDeviceToHost));
    s.Q = Q;
    uint64_t* d_flat = (uint64_t*)w.flat.get(Q * 8);
    k_fill<<<grd, blk>>>(d_seq, d_off, n, bloom, nbits, d_np, d_qoff, d_flat, d_err);
    CK(cudaEventRecord(ev[2]));
    // ---- probe the three indexes
    uint32_t* d_lo = (uint32_t*)w.lo.get(Q * 12), *d_len = (uint32_t*)w.len.get(Q * 12);
    if (Q) k_probe<<<(int)std::min<uint64_t>((Q + blk - 1) / blk, 1u << 22), blk>>>(d_flat, Q, dv, db, df, d_lo, d_len);
    uint64_t* d_tot = (uint64_t*)w.tot.get((n + 1) * 8ull);
    uint64_t* d_hoff = (uint64_t*)w.hoff.get((n + 1) * 8ull);
    k_sum<<<grd, blk>>>(d_np, d_qoff, Q, d_len, n, d_tot);
    CK(cudaMemsetAsync(d_tot + n, 0, 8));
    dev_scan(w, d_tot, d_hoff, (int)n + 1);
    uint64_t H = 0; CK(cudaMemcpy(&H, d_hoff + n, 8, cudaMemcpyDeviceToHost));
    s.H = H;
    CK(cudaEventRecord(ev[3]));
    // ---- expand to one key per occurrence, sort, group, restore first-insertion order
    uint32_t G = 0;
    if (H >= (1ull << 31)) { fprintf(stderr, "H=%llu occurrences: beyond the 31-bit limit of this version\n", (unsigned long long)H); exit(4); }
    if (H) {
        cub::DoubleBuffer<uint64_t> keys((uint64_t*)w.keys0.get(H * 8), (uint64_t*)w.keys1.get(H * 8));
        cub::DoubleBuffer<uint32_t> vals((uint32_t*)w.vals0.get(H * 4), (uint32_t*)w.vals1.get(H * 4));
        k_write<<<grd, blk>>>(d_np, d_qoff, Q, d_lo, d_len, dv, db, df, n, d_hoff, keys.Current(), vals.Current());
        size_t tb = 0;
        CK(cub::DeviceRadixSort::SortPairs(nullptr, tb, keys, vals, (int)H, 0, 64));
        CK(cub::DeviceRadixSort::SortPairs(w.tmp.get(tb), tb, keys, vals, (int)H, 0, 64));
        uint64_t* d_ukey = (uint64_t*)w.ukey.get(H * 8);
        uint32_t* d_counts = (uint32_t*)w.counts.get(H * 4);
        uint32_t* d_nruns = (uint32_t*)w.nruns.get(16);
        tb = 0;
        CK(cub::DeviceRunLengthEncode::Encode(nullptr, tb, keys.Current(), d_ukey, d_counts, d_nruns, (int)H));
        CK(cub::DeviceRunLengthEncode::Encode(w.tmp.get(tb), tb, keys.Current(), d_ukey, d_counts, d_nruns, (int)H));
        CK(cudaMemcpy(&G, d_nruns, 4, cudaMemcpyDeviceToHost));
        uint32_t* d_starts = (uint32_t*)w.starts.get((G + 1ull) * 4);
        tb = 0;
        CK(cub::DeviceScan::ExclusiveSum(nullptr, tb, d_counts, d_starts, (int)G));
        CK(cub::DeviceScan::ExclusiveSum(w.tmp.get(tb), tb, d_counts, d_starts, (int)G));
        cub::DoubleBuffer<uint32_t> first((uint32_t*)w.first0.get(G * 4ull), (uint32_t*)w.first1.get(G * 4ull));
        cub::DoubleBuffer<uint32_t> gid((uint32_t*)w.gid0.get(G * 4ull), (uint32_t*)w.gid1.get(G * 4ull));
        k_first<<<(int)((G + blk - 1) / blk), blk>>>(vals.Current(), d_starts, d_counts, G, first.Current(), gid.Current());
        tb = 0;
        CK(cub::DeviceRadixSort::SortPairs(nullptr, tb, first, gid, (int)G, 0, (int)std::max(1u, ceil_log2(H))));
        CK(cub::DeviceRadixSort::SortPairs(w.tmp.get(tb), tb, first, gid, (int)G, 0, (int)std::max(1u, ceil_log2(H))));
        uint64_t* d_fkey = (uint64_t*)w.fkey.get(G * 8ull);
        uint32_t* d_fcnt = (uint32_t*)w.fcnt.get(G * 4ull);
        k_gather<<<(int)((G + blk - 1) / blk), blk>>>(gid.Current(), d_ukey, d_counts, G, d_fkey, d_fcnt);
    }
    s.G = G;
    CK(cudaEventRecord(ev[4]));
    // ---- results to the host
    uint32_t* h_np = (uint32_t*)w.h_np.get(n * 4ull);
    uint64_t* h_fkey = (uint64_t*)w.h_fkey.get(G * 8ull);
    uint32_t* h_fcnt = (uint32_t*)w.h_fcnt.get(G * 4ull);
    CK(cudaMemcpyAsync(h_np, d_np, n * 4ull, cudaMemcpyDeviceToHost));
    if (G) { CK(cudaMemcpyAsync(h_fkey, w.fkey.p, G * 8ull, cudaMemcpyDeviceToHost)); CK(cudaMemcpyAsync(h_fcnt, w.fcnt.p, G * 4ull, cudaMemcpyDeviceToHost)); }
    CK(cudaEventRecord(ev[5]));
    CK(cudaEventSynchronize(ev[5]));
    unsigned herr = 0; CK(cudaMemcpy(&herr, d_err, 4, cudaMemcpyDeviceToHost));
    if (herr) { fprintf(stderr, "ERROR: %u reads gave a different minimizer count in the fill pass\n", herr); exit(5); }
    float ms;
    CK(cudaEventElapsedTime(&ms, ev[0], ev[1])); s.ms_h2d = ms;
    CK(cudaEventElapsedTime(&ms, ev[1], ev[2])); s.ms_extract = ms;
    CK(cudaEventElapsedTime(&ms, ev[2], ev[3])); s.ms_probe = ms;
    CK(cudaEventElapsedTime(&ms, ev[3], ev[4])); s.ms_expand = ms;
    CK(cudaEventElapsedTime(&ms, ev[4], ev[5])); s.ms_d2h = ms;
    CK(cudaEventElapsedTime(&ms, ev[0], ev[5])); s.ms_total = ms;
    for (auto& e : ev) cudaEventDestroy(e);
    return s;
}

// ---------------------------------------------------------------- host: input
// the engine's reader: 4 lines per record (getline), a record with an empty sequence is skipped, at most maxreads kept
static void read_fastq(const std::string& path, uint64_t maxreads, Host& h, uint64_t& n_records_seen) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror(path.c_str()); exit(3); }
    struct stat st; fstat(fd, &st);
    const uint64_t fsz = st.st_size;
    std::vector<char> data(fsz);
    pread_full(fd, data.data(), fsz, 0); close(fd);
    CK(cudaHostAlloc((void**)&h.seq, fsz + 16, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**)&h.off, (std::min<uint64_t>(maxreads, fsz / 8 + 1) + 2) * 8, cudaHostAllocDefault));
    uint64_t pos = 0, nb = 0; uint32_t n = 0;
    auto getline = [&](const char*& b, uint64_t& len) -> bool {
        if (pos >= fsz) return false;
        const char* nl = (const char*)memchr(data.data() + pos, '\n', fsz - pos);
        const uint64_t e = nl ? (uint64_t)(nl - data.data()) : fsz;
        b = data.data() + pos; len = e - pos; pos = nl ? e + 1 : fsz;
        return true;
    };
    const char *a, *s, *p, *q; uint64_t la, ls, lp, lq;
    n_records_seen = 0;
    h.off[0] = 0;
    while (getline(a, la) && getline(s, ls) && getline(p, lp) && getline(q, lq)) {
        ++n_records_seen;
        if (ls) { memcpy(h.seq + nb, s, ls); nb += ls; h.off[++n] = nb; }
        if (n >= maxreads) break;
    }
    h.n = n; h.bytes = nb;
}

static uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ULL) {
    const unsigned char* c = (const unsigned char*)p;
    for (size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 1099511628211ULL; }
    return h;
}

int main(int argc, char** argv) {
    std::string reads, label = "sample", idxdir, bloomf, outf, jsonf, cpudump;
    uint64_t maxreads = 20000000; int reps = 3, nthreads = 12; bool warm_load = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--reads") reads = nx(); else if (a == "--label") label = nx(); else if (a == "--index-dir") idxdir = nx();
        else if (a == "--bloom") bloomf = nx(); else if (a == "--out") outf = nx(); else if (a == "--json") jsonf = nx();
        else if (a == "--cpu-dump") cpudump = nx(); else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--maxreads") maxreads = strtoull(argv[++i], 0, 10); else if (a == "--load-threads") nthreads = atoi(argv[++i]);
        else if (a == "--no-warm-load") warm_load = false;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device %s, %.1f GiB\n", prop.name, prop.totalGlobalMem / 1073741824.0);

    // ---- step 1: the resident index (viral, bacterial, fungal) and the host Bloom filter
    Index vi, bi, fi; vi.name = "viral"; bi.name = "bacterial"; fi.name = "fungal";
    double tr[3], td[3];
    double T0 = now_s();
    load_index(idxdir + "/viral.snhskidx", vi, nthreads, true, tr[0], td[0]);
    load_index(idxdir + "/bacterial.snhskidx", bi, nthreads, true, tr[1], td[1]);
    load_index(idxdir + "/fungal.snhskidx", fi, nthreads, true, tr[2], td[2]);
    // host Bloom filter
    uint64_t nbits = 0; uint64_t* d_bloom = nullptr; double t_bloom;
    {
        double t0 = now_s();
        int fd = open(bloomf.c_str(), O_RDONLY); if (fd < 0) { perror("bloom"); return 3; }
        char hdr[32]; pread_full(fd, hdr, 32, 0);
        if (memcmp(hdr, "HSKBLOOM", 8) != 0) { fprintf(stderr, "bad bloom\n"); return 3; }
        memcpy(&nbits, hdr + 24, 8); close(fd);
        const uint64_t nw = (nbits + 63) / 64;
        CK(cudaMalloc(&d_bloom, nw * 8));
        file_to_device(bloomf, 32, nw * 8, d_bloom, 4, true);
        t_bloom = now_s() - t0;
    }
    const double t_load_total = now_s() - T0;
    const uint64_t idx_bytes = (vi.n + bi.n + fi.n) * sizeof(Entry);
    const uint64_t dir_bytes = ((1ull << vi.B) + (1ull << bi.B) + (1ull << fi.B) + 3) * 4;
    printf("index load (cold: page cache dropped for the files first): %.2f s total; entries %.2f GB (%llu + %llu + %llu), directories %.2f GB, bloom %.0f MB (%llu bits)\n",
           t_load_total, idx_bytes / 1e9, (unsigned long long)vi.n, (unsigned long long)bi.n, (unsigned long long)fi.n, dir_bytes / 1e9, nbits / 8e6, (unsigned long long)nbits);
    for (auto* x : {&vi, &bi, &fi}) printf("  %-9s n=%llu B=%u (%llu buckets) unsorted_adjacent_pairs=%llu\n", x->name.c_str(), (unsigned long long)x->n, x->B, 1ull << x->B, (unsigned long long)x->unsorted);
    if (vi.unsorted || bi.unsorted || fi.unsorted) { fprintf(stderr, "index is not sorted by hash\n"); return 6; }
    double t_warm_total = -1;
    if (warm_load) {   // same files again, now from the page cache, into scratch copies (timing only)
        Index a, b, c; double r_, d_;
        double t0 = now_s();
        load_index(idxdir + "/viral.snhskidx", a, nthreads, false, r_, d_);
        load_index(idxdir + "/bacterial.snhskidx", b, nthreads, false, r_, d_);
        load_index(idxdir + "/fungal.snhskidx", c, nthreads, false, r_, d_);
        t_warm_total = now_s() - t0;
        free_index(a); free_index(b); free_index(c);
        printf("index load again (page cache warm): %.2f s\n", t_warm_total);
    }
    size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
    const double resident_gb = (tot - fr) / 1e9;

    // ---- step 2: the reads
    Host in; uint64_t seen = 0;
    double t0 = now_s();
    read_fastq(reads, maxreads, in, seen);
    const double t_parse = now_s() - t0;
    printf("reads: %u kept of %llu records (%.2f GB of sequence), parsed in %.2f s\n", in.n, (unsigned long long)seen, in.bytes / 1e9, t_parse);

    // ---- step 3: classify, 1 warm-up + reps timed
    Work w;
    Stats warm = classify(w, in, d_bloom, nbits, vi, bi, fi);
    printf("warm-up: total %.1f ms (Q=%llu H=%llu G=%llu)\n", warm.ms_total, (unsigned long long)warm.Q, (unsigned long long)warm.H, (unsigned long long)warm.G);
    std::vector<Stats> rs; uint64_t cks0 = 0;
    for (int r = 0; r < reps; ++r) {
        Stats s = classify(w, in, d_bloom, nbits, vi, bi, fi);
        uint64_t ck = fnv(w.h_np.p, in.n * 4ull); ck = fnv(w.h_fkey.p, s.G * 8, ck); ck = fnv(w.h_fcnt.p, s.G * 4, ck);
        if (r == 0) cks0 = ck; else if (ck != cks0) { fprintf(stderr, "NON-DETERMINISTIC output between reps\n"); return 7; }
        rs.push_back(s);
        printf("rep %d: total %.2f ms = h2d %.2f + extract %.2f + probe %.2f + expand/sort %.2f + d2h %.2f\n", r, s.ms_total, s.ms_h2d, s.ms_extract, s.ms_probe, s.ms_expand, s.ms_d2h);
    }
    auto med = [&](double Stats::*f) { std::vector<double> v; for (auto& s : rs) v.push_back(s.*f); std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    const Stats& S = rs[0];
    const double m_total = med(&Stats::ms_total), m_h2d = med(&Stats::ms_h2d), m_ext = med(&Stats::ms_extract), m_probe = med(&Stats::ms_probe),
                 m_exp = med(&Stats::ms_expand), m_d2h = med(&Stats::ms_d2h);
    const double m_compute = m_ext + m_probe + m_exp;
    printf("median of %d: total %.2f ms (compute only %.2f ms); %.1f M reads/s total, %.1f M reads/s compute only\n", reps, m_total, m_compute,
           in.n / (m_total * 1e3), in.n / (m_compute * 1e3));
    CK(cudaMemGetInfo(&fr, &tot));
    const double peak_gb = (tot - fr) / 1e9;

    // ---- step 4: results file and comparison with the CPU dump
    const uint32_t* np = (const uint32_t*)w.h_np.p; const uint64_t* fkey = (const uint64_t*)w.h_fkey.p; const uint32_t* fcnt = (const uint32_t*)w.h_fcnt.p;
    if (!outf.empty()) {
        FILE* f = fopen(outf.c_str(), "wb");
        const uint64_t nr = in.n, ng = S.G;
        fwrite("NHGPURS1", 1, 8, f); fwrite(&nr, 8, 1, f); fwrite(&ng, 8, 1, f);
        fwrite(np, 4, nr, f); fwrite(fkey, 8, ng, f); fwrite(fcnt, 4, ng, f); fclose(f);
    }
    uint64_t reads_np = 0, reads_hit = 0, reads_v = 0, reads_m = 0, maxocc = 0;
    for (uint32_t i = 0; i < in.n; ++i) reads_np += np[i] > 0;
    { uint64_t prev = ~0ull; for (uint64_t g = 0; g < S.G; ++g) { uint64_t r = fkey[g] >> 33; if (r != prev) { ++reads_hit; prev = r; } (fkey[g] >> 32 & 1 ? reads_m : reads_v)++; } }
    (void)maxocc;
    uint64_t cmp_reads = 0, bad_np = 0, bad_v = 0, bad_m = 0, bad_any = 0; bool compared = false;
    if (!cpudump.empty()) {
        compared = true;
        FILE* f = fopen(cpudump.c_str(), "rb"); if (!f) { perror(cpudump.c_str()); return 3; }
        std::vector<char> raw; { fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); raw.resize(sz); if (fread(raw.data(), 1, sz, f) != (size_t)sz) return 3; fclose(f); }
        const uint32_t* u = (const uint32_t*)raw.data(); size_t at = 0, nu = raw.size() / 4;
        uint64_t g = 0; std::vector<std::pair<uint32_t, uint32_t>> cv, cm, gv, gm;
        uint64_t shown = 0;
        for (uint32_t i = 0; i < in.n; ++i) {
            if (at >= nu) { fprintf(stderr, "CPU dump ended early at read %u\n", i); return 8; }
            const uint32_t cnp = u[at++]; uint32_t nv = u[at++];
            cv.clear(); cm.clear(); gv.clear(); gm.clear();
            for (uint32_t t = 0; t < nv; ++t, at += 2) cv.push_back({u[at], u[at + 1]});
            uint32_t nm = u[at++];
            for (uint32_t t = 0; t < nm; ++t, at += 2) cm.push_back({u[at], u[at + 1]});
            while (g < S.G && (fkey[g] >> 33) == i) { ((fkey[g] >> 32) & 1 ? gm : gv).push_back({(uint32_t)(fkey[g] & 0xffffffffu), fcnt[g]}); ++g; }
            std::sort(cv.begin(), cv.end()); std::sort(cm.begin(), cm.end()); std::sort(gv.begin(), gv.end()); std::sort(gm.begin(), gm.end());
            const bool b0 = cnp != np[i], b1 = cv != gv, b2 = cm != gm;
            bad_np += b0; bad_v += b1; bad_m += b2; bad_any += (b0 || b1 || b2);
            if ((b0 || b1 || b2) && shown++ < 10) fprintf(stderr, "MISMATCH read %u: np cpu=%u gpu=%u, viral cpu=%zu gpu=%zu, micro cpu=%zu gpu=%zu\n", i, cnp, np[i], cv.size(), gv.size(), cm.size(), gm.size());
            ++cmp_reads;
        }
        if (g != S.G) { fprintf(stderr, "GPU has %llu groups for reads beyond the CPU dump\n", (unsigned long long)(S.G - g)); bad_any++; }
        printf("compare with CPU dump: %llu reads, np mismatches %llu, viral-map mismatches %llu, microbial-map mismatches %llu, reads with any mismatch %llu\n",
               (unsigned long long)cmp_reads, (unsigned long long)bad_np, (unsigned long long)bad_v, (unsigned long long)bad_m, (unsigned long long)bad_any);
    }

    if (!jsonf.empty()) {
        FILE* f = fopen(jsonf.c_str(), "w");
        fprintf(f, "{\n  \"label\": \"%s\",\n  \"device\": \"%s\",\n", label.c_str(), prop.name);
        fprintf(f, "  \"index\": {\"entries_viral\": %llu, \"entries_bacterial\": %llu, \"entries_fungal\": %llu, \"entries_bytes\": %llu, \"directory_bytes\": %llu, \"bloom_bytes\": %llu, "
                   "\"resident_hbm_gb\": %.3f, \"dir_bits\": [%u, %u, %u], \"load_cold_s\": %.3f, \"load_warm_s\": %.3f, \"load_read_to_hbm_s\": [%.3f, %.3f, %.3f], "
                   "\"load_directory_build_s\": [%.4f, %.4f, %.4f], \"load_bloom_s\": %.3f, \"unsorted_pairs\": %llu},\n",
                (unsigned long long)vi.n, (unsigned long long)bi.n, (unsigned long long)fi.n, (unsigned long long)idx_bytes, (unsigned long long)dir_bytes,
                (unsigned long long)(nbits / 8), resident_gb, vi.B, bi.B, fi.B, t_load_total, t_warm_total, tr[0], tr[1], tr[2], td[0], td[1], td[2], t_bloom,
                (unsigned long long)(vi.unsorted + bi.unsorted + fi.unsorted));
        fprintf(f, "  \"reads\": {\"records_seen\": %llu, \"reads\": %u, \"sequence_bytes\": %llu, \"host_parse_s\": %.3f, \"reads_with_np\": %llu, \"reads_with_hits\": %llu, "
                   "\"minimizers_after_bloom\": %llu, \"occurrences\": %llu, \"groups\": %llu, \"viral_groups\": %llu, \"microbial_groups\": %llu},\n",
                (unsigned long long)seen, in.n, (unsigned long long)in.bytes, t_parse, (unsigned long long)reads_np, (unsigned long long)reads_hit,
                (unsigned long long)S.Q, (unsigned long long)S.H, (unsigned long long)S.G, (unsigned long long)reads_v, (unsigned long long)reads_m);
        fprintf(f, "  \"gpu_ms\": {\"reps\": %d, \"median_total\": %.3f, \"median_h2d\": %.3f, \"median_extract_bloom\": %.3f, \"median_probe\": %.3f, \"median_expand_sort_group\": %.3f, "
                   "\"median_d2h\": %.3f, \"median_compute_only\": %.3f, \"warmup_total\": %.3f, \"all_total\": [", reps, m_total, m_h2d, m_ext, m_probe, m_exp, m_d2h, m_compute, warm.ms_total);
        for (size_t i = 0; i < rs.size(); ++i) fprintf(f, "%s%.3f", i ? ", " : "", rs[i].ms_total);
        fprintf(f, "]},\n  \"reads_per_s\": {\"total_incl_transfers\": %.0f, \"compute_only\": %.0f},\n", in.n / (m_total * 1e-3), in.n / (m_compute * 1e-3));
        fprintf(f, "  \"hbm_gb\": {\"resident_index_and_bloom\": %.3f, \"peak_with_classify_buffers\": %.3f, \"classify_buffers\": %.3f},\n", resident_gb, peak_gb, w.bytes() / 1e9);
        fprintf(f, "  \"deterministic_across_reps\": true,\n");
        if (compared)
            fprintf(f, "  \"compare_cpu_dump\": {\"reads\": %llu, \"np_mismatch\": %llu, \"viral_map_mismatch\": %llu, \"microbial_map_mismatch\": %llu, \"reads_with_any_mismatch\": %llu},\n",
                    (unsigned long long)cmp_reads, (unsigned long long)bad_np, (unsigned long long)bad_v, (unsigned long long)bad_m, (unsigned long long)bad_any);
        fprintf(f, "  \"results_fnv64\": \"%016llx\"\n}\n", (unsigned long long)cks0);
        fclose(f);
    }
    return compared && bad_any ? 9 : 0;
}
