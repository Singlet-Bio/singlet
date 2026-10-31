// Copied from PR #61 (branch gpu/w11-prototypes, commit f84aafff), bench/gpu/m6/m6.cu: see bench/gpu/pileup/README.md there. One change
// for singlet: an exhausted arena throws ArenaExhausted (the engine retries in managed memory) instead of exit(1).
// Linked into singlet through ../pileup_gpu.cu (PILEUP_NO_MAIN); identity: docs/gpu-engine/exploration.md 12.13, 12.19.
// M6: read dedup, count-key sort, UMI collapse and CSR reduce on one GPU (docs/gpu-engine/exploration.md section 9, M6).
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false [-DNW=4] -o m6 m6.cu      (NW = u64 words per read key; reads longer than 32*NW abort)
// Run:   decode R2 | m6 --r2 - --tuples S.tuples --truth S.tuples.truth [--batches 4,16,64,0] [--reps 3] [--verify] [--json out.json]
//
// Stage a  read dedup: 2-bit pack (done on the host), 64-bit hash, CUB radix sort of (hash, index) on the top 40 hash bits,
//          neighbour compare of the FULL key (NW words + length), exact repair of every run of equal 40-bit keys that holds
//          more than one sequence (small comparison sort), ids = ranks in (hash, sequence, length) order. Batches are
//          deduplicated independently; the batch distinct lists are merged by the same procedure into global ranks.
// Stage b  count key: (cell | gene | UMI) 64-bit key + 51-bit payload (exon, intron, flags), CUB radix sort, reduce-by-key into
//          (count, flags OR, min exon, min intron) per exact (cell, gene, UMI). Batches pre-aggregate; a merge re-sorts the
//          concatenated unique lists and reduces again.
// Stage c  UMI collapse per (cell, gene): the CPU rule of include/singlet/pileup/umi_dedup.h (finalize_molecules, directional):
//          edge between two distinct UMIs iff Hamming == 1 (N packed as T) and max(count) >= 2*min(count) - 1; molecule =
//          connected component; flags OR, exon min, intron min over the component. Candidate pairs by the split-halves trick
//          (two UMIs at Hamming 1 share one half exactly); components by deterministic min-label hooking (atomicMin only).
// Stage d  molecules -> per class (exon, intron, ambiguous-by-gene) sort of (cell | feature), run-length encode, CSR (u16 clamp).
//
// Determinism: no float, no atomics that hand out slots; atomics are atomicMin/atomicOr on integers (order-free results).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <sys/types.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#ifndef NW
#define NW 4
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
static void die(const char* m) { fprintf(stderr, "FATAL: %s\n", m); exit(1); }

// ------------------------------------------------------------------------------------------------------------------------
// tuple record written by tuple_dump.patch (32 bytes)
struct Rec { uint32_t bc, gene, exon, intron; uint64_t umi; uint8_t flags, umi_len, hasN, direct; uint32_t pad; };
static_assert(sizeof(Rec) == 32, "Rec");
static const uint32_t NONE32 = 0xFFFFFFFFu;
static const uint64_t NONE64 = ~0ull;
static const int FB = 24;                      // feature bits in the class keys (exon / intron / gene index < 2^24 - 1)

struct Agg { uint32_t count, flags, exon, intron; };

// ------------------------------------------------------------------------------------------------------------------------
// arena (bump allocation of device memory; reset between runs so timed regions contain no cudaMalloc)
struct ArenaExhausted { size_t need, cap; };
struct Arena {
    char* base = nullptr; size_t cap = 0, off = 0, peak = 0;
    void init(size_t c) { cap = c; CK(cudaMalloc(&base, c)); }
    void init_managed(size_t c) { cap = c; CK(cudaMallocManaged(&base, c)); }   // unified memory: may exceed the device's
    template <class T> T* alloc(size_t n) {
        size_t b = (n * sizeof(T) + 255) & ~size_t(255); if (b == 0) b = 256;
        if (off + b > cap) { fprintf(stderr, "arena exhausted: need %zu have %zu\n", off + b, cap); throw ArenaExhausted{off + b, cap}; }
        T* p = (T*)(base + off); off += b; peak = std::max(peak, off); return p;
    }
    size_t mark() const { return off; }
    void reset(size_t m) { off = m; }
};
template <class F> static void cub_run(Arena& A, F f) { size_t bytes = 0; CK(f(nullptr, bytes)); void* t = A.alloc<char>(bytes); CK(f(t, bytes)); }

// phase timer: named marks, durations accumulate per name
struct Timer {
    std::vector<cudaEvent_t> ev; std::vector<std::string> nm; std::map<std::string, double> acc;
    void start() { clear(); mark("start"); }
    void mark(const char* n) { cudaEvent_t e; CK(cudaEventCreate(&e)); CK(cudaEventRecord(e)); ev.push_back(e); nm.push_back(n); }
    void finish() {
        if (ev.empty()) return; CK(cudaEventSynchronize(ev.back()));
        for (size_t i = 1; i < ev.size(); i++) { float ms; CK(cudaEventElapsedTime(&ms, ev[i - 1], ev[i])); acc[nm[i]] += ms; }
        for (auto e : ev) cudaEventDestroy(e); ev.clear(); nm.clear();
    }
    void clear() { for (auto e : ev) cudaEventDestroy(e); ev.clear(); nm.clear(); acc.clear(); }
};

// ------------------------------------------------------------------------------------------------------------------------
// SHA-256 (host)
struct Sha256 {
    uint32_t h[8]; uint64_t len = 0; uint8_t buf[64]; size_t nb = 0;
    Sha256() { static const uint32_t i[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}; memcpy(h, i, 32); }
    static uint32_t rr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* p) {
        static const uint32_t K[64] = {0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4*i] << 24 | (uint32_t)p[4*i+1] << 16 | (uint32_t)p[4*i+2] << 8 | p[4*i+3];
        for (int i = 16; i < 64; i++) { uint32_t s0 = rr(w[i-15],7) ^ rr(w[i-15],18) ^ (w[i-15] >> 3), s1 = rr(w[i-2],17) ^ rr(w[i-2],19) ^ (w[i-2] >> 10); w[i] = w[i-16] + s0 + w[i-7] + s1; }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rr(e,6)^rr(e,11)^rr(e,25), ch = (e&f)^(~e&g), t1 = hh+S1+ch+K[i]+w[i];
            uint32_t S0 = rr(a,2)^rr(a,13)^rr(a,22), mj = (a&b)^(a&c)^(b&c), t2 = S0+mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    void update(const void* d, size_t n) {
        const uint8_t* p = (const uint8_t*)d; len += n;
        while (n) { size_t k = std::min(n, 64 - nb); memcpy(buf + nb, p, k); nb += k; p += k; n -= k; if (nb == 64) { block(buf); nb = 0; } }
    }
    std::string hex() {
        uint64_t bits = len * 8; uint8_t pad = 0x80; update(&pad, 1); uint8_t z = 0; while (nb != 56) update(&z, 1);
        uint8_t l[8]; for (int i = 0; i < 8; i++) l[i] = (uint8_t)(bits >> (56 - 8 * i)); update(l, 8);
        char s[65]; for (int i = 0; i < 8; i++) snprintf(s + 8 * i, 9, "%08x", h[i]); return s;
    }
};

// ------------------------------------------------------------------------------------------------------------------------
// device helpers
__device__ __forceinline__ uint64_t fmix(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33; return k; }
__device__ __forceinline__ uint64_t hash_seq(const uint64_t* w, uint32_t len) {
    uint64_t h = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)len * 0xff51afd7ed558ccdULL);
#pragma unroll
    for (int i = 0; i < NW; i++) h = fmix(h ^ w[i]);
    return h;
}
__host__ __device__ __forceinline__ bool seq_eq(const uint64_t* a, uint32_t la, const uint64_t* b, uint32_t lb) {
    if (la != lb) return false;
    for (int i = 0; i < NW; i++) if (a[i] != b[i]) return false;
    return true;
}
__device__ __forceinline__ int seq_cmp(const uint64_t* a, uint32_t la, const uint64_t* b, uint32_t lb) {
    for (int i = 0; i < NW; i++) if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    if (la != lb) return la < lb ? -1 : 1;
    return 0;
}

// ---- stage a kernels ----
__global__ void k_hash(const uint64_t* W, const uint8_t* len, int n, uint64_t* key, uint32_t* idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    key[i] = hash_seq(W + (size_t)i * NW, len[i]); idx[i] = i;
}
// neq[i] = sorted element i differs from i-1 in full key; rs[i] = i starts a run of equal top-40 hash bits
__global__ void k_neq(const uint64_t* W, const uint8_t* len, const uint64_t* ks, const uint32_t* is, int n, uint8_t* neq, uint32_t* rs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    if (i == 0) { neq[0] = 1; rs[0] = 1; return; }
    uint32_t a = is[i - 1], b = is[i];
    neq[i] = !seq_eq(W + (size_t)a * NW, len[a], W + (size_t)b * NW, len[b]);
    rs[i] = (ks[i] >> 24) != (ks[i - 1] >> 24);
}
__global__ void k_mixed(const uint8_t* neq, const uint32_t* rs, const uint32_t* rid, int n, uint8_t* frun) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    if (!rs[i] && neq[i]) frun[rid[i]] = 1;          // all writers store the same value
}
__global__ void k_selflag(const uint32_t* rid, const uint8_t* frun, int n, uint8_t* sel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    sel[i] = frun[rid[i]];
}
struct Elem { uint32_t rid, idx; uint64_t hash; };
struct ElemCmp {
    const uint64_t* W; const uint8_t* len;
    __device__ bool operator()(const Elem& a, const Elem& b) const {
        if (a.rid != b.rid) return a.rid < b.rid;
        if (a.hash != b.hash) return a.hash < b.hash;
        int c = seq_cmp(W + (size_t)a.idx * NW, len[a.idx], W + (size_t)b.idx * NW, len[b.idx]);
        if (c) return c < 0;
        return a.idx < b.idx;
    }
};
__global__ void k_gather_elem(const uint32_t* pos, int nm, const uint32_t* rid, const uint32_t* is, const uint64_t* ks, Elem* E) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= nm) return;
    uint32_t p = pos[k]; E[k] = Elem{rid[p], is[p], ks[p]};
}
__global__ void k_scatter_elem(const uint32_t* pos, int nm, const Elem* E, uint32_t* is, uint64_t* ks) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= nm) return;
    uint32_t p = pos[k]; is[p] = E[k].idx; ks[p] = E[k].hash;
}
__global__ void k_bflag(const uint8_t* neq, int n, uint32_t* b) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) b[i] = neq[i]; }
__global__ void k_rank(const uint32_t* is, const uint32_t* rk, const uint8_t* neq, int n, uint32_t* lid, uint32_t* rep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    lid[is[i]] = rk[i] - 1; if (neq[i]) rep[rk[i] - 1] = is[i];
}
__global__ void k_gather_seq(const uint64_t* W, const uint8_t* len, const uint32_t* src, int n, uint32_t off, uint64_t* Wo, uint8_t* lo) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= n) return;
    uint32_t s = src[k] + off;
    for (int j = 0; j < NW; j++) Wo[(size_t)k * NW + j] = W[(size_t)s * NW + j];
    lo[k] = len[s];
}
__global__ void k_repg(const uint32_t* rep, int d, uint32_t off, uint32_t* out) { int k = blockIdx.x * blockDim.x + threadIdx.x; if (k < d) out[k] = rep[k] + off; }
__global__ void k_final_gid(const uint32_t* lid, const uint32_t* gid_rep, const uint32_t* boff, uint32_t B, int n, uint32_t* gid) {
    uint32_t i = blockIdx.x * (uint32_t)blockDim.x + threadIdx.x; if (i >= (uint32_t)n) return;
    gid[i] = gid_rep[boff[i / B] + lid[i]];
}

struct DedupStat { long long mixed_elems = 0; };
// distinct ids of n sequences (Wb, lenb); lid[i] = rank of i's sequence in (hash, sequence, length) order; rep[rank] = smallest index with it.
static int dedup_core(Arena& A, Timer& T, const uint64_t* Wb, const uint8_t* lenb, int n, uint32_t* lid, uint32_t* rep, DedupStat* st) {
    size_t mk = A.mark(); const int TB = 256, G = (n + TB - 1) / TB;
    uint64_t* k0 = A.alloc<uint64_t>(n); uint64_t* k1 = A.alloc<uint64_t>(n); uint32_t* i0 = A.alloc<uint32_t>(n); uint32_t* i1 = A.alloc<uint32_t>(n);
    int* dnum = A.alloc<int>(4);
    k_hash<<<G, TB>>>(Wb, lenb, n, k0, i0); T.mark("a.hash");
    cub::DoubleBuffer<uint64_t> dk(k0, k1); cub::DoubleBuffer<uint32_t> di(i0, i1);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, dk, di, n, 24, 64); });
    T.mark("a.sort");
    uint64_t* ks = dk.Current(); uint32_t* is = di.Current();
    uint8_t* neq = A.alloc<uint8_t>(n); uint32_t* rs = A.alloc<uint32_t>(n); uint32_t* rid = A.alloc<uint32_t>(n);
    uint8_t* frun = A.alloc<uint8_t>((size_t)n + 2); uint8_t* sel = A.alloc<uint8_t>(n);
    k_neq<<<G, TB>>>(Wb, lenb, ks, is, n, neq, rs);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceScan::InclusiveSum(t, b, rs, rid, n); });
    CK(cudaMemsetAsync(frun, 0, (size_t)n + 2));
    k_mixed<<<G, TB>>>(neq, rs, rid, n, frun);
    k_selflag<<<G, TB>>>(rid, frun, n, sel);
    uint32_t* pos = A.alloc<uint32_t>(n);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceSelect::Flagged(t, b, cub::CountingInputIterator<uint32_t>(0), sel, pos, dnum, n); });
    int nm = 0; CK(cudaMemcpy(&nm, dnum, 4, cudaMemcpyDeviceToHost));
    if (st) st->mixed_elems += nm;
    if (nm > 0) {      // exact repair: runs of equal 40-bit keys that hold several sequences are re-ordered by (hash, sequence, length, index)
        Elem* E = A.alloc<Elem>(nm);
        k_gather_elem<<<(nm + TB - 1) / TB, TB>>>(pos, nm, rid, is, ks, E);
        thrust::sort(thrust::device, thrust::device_ptr<Elem>(E), thrust::device_ptr<Elem>(E) + nm, ElemCmp{Wb, lenb});
        k_scatter_elem<<<(nm + TB - 1) / TB, TB>>>(pos, nm, E, is, ks);
        k_neq<<<G, TB>>>(Wb, lenb, ks, is, n, neq, rs);
    }
    T.mark("a.fix");
    uint32_t* bf = A.alloc<uint32_t>(n); uint32_t* rk = A.alloc<uint32_t>(n);
    k_bflag<<<G, TB>>>(neq, n, bf);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceScan::InclusiveSum(t, b, bf, rk, n); });
    k_rank<<<G, TB>>>(is, rk, neq, n, lid, rep);
    int D = 0; CK(cudaMemcpy(&D, rk + (n - 1), 4, cudaMemcpyDeviceToHost));
    T.mark("a.rank");
    A.reset(mk); return D;
}

struct DedupResult { long long D_local = 0, D_global = 0, mixed = 0; uint32_t* gid = nullptr; };
// batches of B reads (B = 0 or >= n: one batch); returns global ids per read (device pointer in arena)
static DedupResult dedup_batches(Arena& A, Timer& T, const uint64_t* W, const uint8_t* len, int n, long long B) {
    DedupResult R; DedupStat st;
    if (B <= 0 || B >= n) B = n;
    int nb = (int)((n + B - 1) / B);
    uint32_t* lid = A.alloc<uint32_t>(n); uint32_t* repg = A.alloc<uint32_t>(n);
    uint32_t* rep = A.alloc<uint32_t>(n);
    std::vector<uint32_t> boff(nb + 1, 0);
    long long Dtot = 0;
    for (int b = 0; b < nb; b++) {
        long long lo = (long long)b * B, hi = std::min<long long>(n, lo + B); int nn = (int)(hi - lo);
        int D = dedup_core(A, T, W + lo * NW, len + lo, nn, lid + lo, rep, &st);
        k_repg<<<(D + 255) / 256, 256>>>(rep, D, (uint32_t)lo, repg + Dtot);
        Dtot += D; boff[b + 1] = (uint32_t)Dtot;
    }
    R.D_local = Dtot; R.mixed = st.mixed_elems;
    if (nb == 1) { R.gid = lid; R.D_global = Dtot; return R; }
    // merge: deduplicate the concatenated batch representatives
    uint64_t* Wr = A.alloc<uint64_t>((size_t)Dtot * NW); uint8_t* lr = A.alloc<uint8_t>(Dtot);
    k_gather_seq<<<(int)((Dtot + 255) / 256), 256>>>(W, len, repg, (int)Dtot, 0, Wr, lr);
    uint32_t* gid_rep = A.alloc<uint32_t>(Dtot); uint32_t* rep2 = A.alloc<uint32_t>(Dtot);
    int DG = dedup_core(A, T, Wr, lr, (int)Dtot, gid_rep, rep2, &st);
    uint32_t* dboff = A.alloc<uint32_t>(nb + 1); CK(cudaMemcpy(dboff, boff.data(), 4 * (nb + 1), cudaMemcpyHostToDevice));
    uint32_t* gid = A.alloc<uint32_t>(n);
    k_final_gid<<<(n + 255) / 256, 256>>>(lid, gid_rep, dboff, (uint32_t)B, n, gid);
    T.mark("a.merge");
    R.gid = gid; R.D_global = DG; R.mixed = st.mixed_elems; return R;
}

// ---- stage b kernels ----
__global__ void k_mkkey(const Rec* r, int n, int gb, int ub, uint64_t* key, uint64_t* pay) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    Rec x = r[i];
    key[i] = ((uint64_t)x.bc << (gb + ub)) | ((uint64_t)x.gene << ub) | x.umi;
    uint64_t e = x.exon == NONE32 ? 0xFFFFFFull : x.exon, in = x.intron == NONE32 ? 0xFFFFFFull : x.intron;
    pay[i] = (e << 27) | (in << 3) | (x.flags & 7);
}
struct ToAgg { __host__ __device__ Agg operator()(const uint64_t& p) const {
    uint32_t e = (uint32_t)((p >> 27) & 0xFFFFFF), in = (uint32_t)((p >> 3) & 0xFFFFFF);
    return Agg{1u, (uint32_t)(p & 7), e == 0xFFFFFFu ? NONE32 : e, in == 0xFFFFFFu ? NONE32 : in}; } };
struct AggOp { __host__ __device__ Agg operator()(const Agg& a, const Agg& b) const { return Agg{a.count + b.count, a.flags | b.flags, a.exon < b.exon ? a.exon : b.exon, a.intron < b.intron ? a.intron : b.intron}; } };
__global__ void k_gather_agg(const Agg* a, const uint32_t* idx, int n, Agg* o) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = a[idx[i]]; }
__global__ void k_iota(uint32_t* p, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] = i; }

struct Layout { int cb, gb, ub, ncells; int ulen; bool directional; bool cr = false; };   // cr: intron_rule::kUmiGeneCR, the molecule rule over genes (stage_cr)

// stage b: returns number of unique (cell, gene, UMI) entries; ukey / agg are arena pointers
static int stage_b(Arena& A, Timer& T, const Rec* recs, int n, long long B, const Layout& L, uint64_t** ukey_out, Agg** agg_out) {
    if (B <= 0 || B >= n) B = n;
    int nb = (int)((n + B - 1) / B); const int TB = 256; int tb = L.cb + L.gb + L.ub;
    uint64_t* ucat = A.alloc<uint64_t>(n); Agg* acat = A.alloc<Agg>(n); long long off = 0;
    int* dnum = A.alloc<int>(4);
    for (int b = 0; b < nb; b++) {
        size_t mk = A.mark(); long long lo = (long long)b * B, hi = std::min<long long>(n, lo + B); int nn = (int)(hi - lo);
        uint64_t *k0 = A.alloc<uint64_t>(nn), *k1 = A.alloc<uint64_t>(nn), *p0 = A.alloc<uint64_t>(nn), *p1 = A.alloc<uint64_t>(nn);
        k_mkkey<<<(nn + TB - 1) / TB, TB>>>(recs + lo, nn, L.gb, L.ub, k0, p0); T.mark("b.key");
        cub::DoubleBuffer<uint64_t> dk(k0, k1), dp(p0, p1);
        cub_run(A, [&](void* t, size_t& bb) { return cub::DeviceRadixSort::SortPairs(t, bb, dk, dp, nn, 0, tb); });
        T.mark("b.sort");
        cub::TransformInputIterator<Agg, ToAgg, const uint64_t*> vin(dp.Current(), ToAgg());
        cub_run(A, [&](void* t, size_t& bb) { return cub::DeviceReduce::ReduceByKey(t, bb, dk.Current(), ucat + off, vin, acat + off, dnum, AggOp(), nn); });
        int nr = 0; CK(cudaMemcpy(&nr, dnum, 4, cudaMemcpyDeviceToHost)); off += nr;
        T.mark("b.rbk"); A.reset(mk);
    }
    int m = (int)off;
    if (nb == 1) { *ukey_out = ucat; *agg_out = acat; return m; }
    // merge pre-aggregated batches
    uint64_t *k0 = A.alloc<uint64_t>(m), *k1 = A.alloc<uint64_t>(m); uint32_t *i0 = A.alloc<uint32_t>(m), *i1 = A.alloc<uint32_t>(m);
    CK(cudaMemcpyAsync(k0, ucat, (size_t)m * 8, cudaMemcpyDeviceToDevice)); k_iota<<<(m + TB - 1) / TB, TB>>>(i0, m);
    cub::DoubleBuffer<uint64_t> dk(k0, k1); cub::DoubleBuffer<uint32_t> di(i0, i1);
    cub_run(A, [&](void* t, size_t& bb) { return cub::DeviceRadixSort::SortPairs(t, bb, dk, di, m, 0, tb); });
    Agg* ag = A.alloc<Agg>(m); k_gather_agg<<<(m + TB - 1) / TB, TB>>>(acat, di.Current(), m, ag);
    uint64_t* uk = A.alloc<uint64_t>(m); Agg* ao = A.alloc<Agg>(m);
    cub_run(A, [&](void* t, size_t& bb) { return cub::DeviceReduce::ReduceByKey(t, bb, dk.Current(), uk, ag, ao, dnum, AggOp(), m); });
    int nr = 0; CK(cudaMemcpy(&nr, dnum, 4, cudaMemcpyDeviceToHost));
    T.mark("b.merge");
    *ukey_out = uk; *agg_out = ao; return nr;
}

// ---- stage c kernels ----
__device__ __forceinline__ bool ham1(uint64_t a, uint64_t b, uint64_t mask) {
    uint64_t x = (a ^ b) & mask, y = (x | (x >> 1)) & 0x5555555555555555ULL; return y != 0 && (y & (y - 1)) == 0;
}
__device__ __forceinline__ bool count_rule(uint32_t ca, uint32_t cb) { uint64_t hi = max(ca, cb), lo = min(ca, cb); return hi >= 2 * lo - 1; }
__global__ void k_segflag(const uint64_t* uk, int m, int ub, uint32_t* sf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= m) return;
    sf[i] = (i == 0) || ((uk[i] >> ub) != (uk[i - 1] >> ub));
}
__global__ void k_active(const uint32_t* sf, int m, uint8_t* act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= m) return;
    act[i] = (!sf[i]) || (i + 1 < m && !sf[i + 1]);
}
__global__ void k_mkK(const uint64_t* uk, const uint32_t* seg, const uint32_t* actidx, int na, uint64_t umask, int lob, int h, uint64_t* K, uint32_t* V) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= na) return;
    uint32_t e = actidx[j]; uint64_t u = uk[e] & umask;
    uint64_t half = h == 0 ? (u & ((1ull << lob) - 1)) : (u >> lob);
    int hb = h == 0 ? lob : 0; (void)hb;
    K[j] = half; V[j] = e;   // segment id is merged in k_mkK2
}
__global__ void k_mkK2(const uint32_t* seg, int na, int hb, uint64_t* K, const uint32_t* V) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= na) return;
    K[j] |= (uint64_t)(seg[V[j]] - 1) << hb;
}
__global__ void k_edge_count(const uint64_t* uk, const Agg* ag, const uint64_t* K, const uint32_t* V, int na, uint64_t umask, uint32_t* cnt) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= na) return;
    uint32_t c = 0, a = V[j];
    for (int q = j + 1; q < na && K[q] == K[j]; q++) { uint32_t b = V[q]; if (ham1(uk[a], uk[b], umask) && count_rule(ag[a].count, ag[b].count)) c++; }
    cnt[j] = c;
}
__global__ void k_edge_write(const uint64_t* uk, const Agg* ag, const uint64_t* K, const uint32_t* V, int na, uint64_t umask, const uint32_t* off, uint32_t* ea, uint32_t* eb) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= na) return;
    uint32_t o = off[j], a = V[j];
    for (int q = j + 1; q < na && K[q] == K[j]; q++) { uint32_t b = V[q]; if (ham1(uk[a], uk[b], umask) && count_rule(ag[a].count, ag[b].count)) { ea[o] = a; eb[o] = b; o++; } }
}
__device__ __forceinline__ uint32_t root_of(const uint32_t* lab, uint32_t x) { while (true) { uint32_t p = *(volatile const uint32_t*)(lab + x); if (p == x) return x; x = p; } }
__global__ void k_hook(const uint32_t* ea, const uint32_t* eb, int ne, uint32_t* lab, int* changed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= ne) return;
    uint32_t ra = root_of(lab, ea[i]), rb = root_of(lab, eb[i]);
    if (ra != rb) { uint32_t hi = max(ra, rb), lo = min(ra, rb); uint32_t old = atomicMin(&lab[hi], lo); if (old > lo) *changed = 1; }
}
__global__ void k_compress(uint32_t* lab, int m) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < m) lab[i] = root_of(lab, i); }
__global__ void k_comp_init(uint32_t* cf, uint32_t* ce, uint32_t* ci, int m) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < m) { cf[i] = 0; ce[i] = NONE32; ci[i] = NONE32; } }
__global__ void k_comp_agg(const Agg* ag, const uint32_t* lab, int m, uint32_t* cf, uint32_t* ce, uint32_t* ci) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= m) return;
    uint32_t r = lab[i]; atomicOr(&cf[r], ag[i].flags); atomicMin(&ce[r], ag[i].exon); atomicMin(&ci[r], ag[i].intron);
}
// molecule label counts, as DirectionalUmiStore::finalize_molecules' MoleculeStats: spliced, unspliced, ambiguous, legacy_double
// (TouchesExon and Intron both set); zeroed by the caller before a run
__device__ unsigned long long g_molstat[5];   // [4]: molecules dropped by the molecule rule (kUmiGeneCR)
// molecules per cell (DirectionalUmiStore::per_cell_molecule_stats' unique_umis), when the caller sets it (else nullptr)
__device__ uint32_t* g_cell_mol = nullptr;
// roots emit one molecule: flags bit1 = exon-only read, bit2 = intron read (DirectionalUmiStore::kReadExonOnly / kReadIntron)
__global__ void k_emit(const uint64_t* uk, const uint32_t* lab, const uint32_t* cf, const uint32_t* ce, const uint32_t* ci, int m, int gb, int ub,
                       uint64_t* kex, uint64_t* kin, uint64_t* kam, const uint8_t* drop) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= m) return;
    kex[i] = kin[i] = kam[i] = NONE64;
    if (lab[i] != (uint32_t)i) return;
    uint32_t f = cf[i]; bool e = f & 1, in = f & 2;
    uint64_t cell = uk[i] >> (gb + ub), gene = (uk[i] >> ub) & ((1ull << gb) - 1);
    if (g_cell_mol) atomicAdd(&g_cell_mol[cell], 1u);                                    // per_cell_molecule_stats counts every molecule, dropped or not
    if (drop && drop[i]) { atomicAdd(&g_molstat[4], 1ull); return; }                     // kUmiGeneCR: finalize_all's `dropped`
    if ((f & 4) && in) atomicAdd(&g_molstat[3], 1ull);
    atomicAdd(&g_molstat[(e && in) ? 2 : (in ? 1 : 0)], 1ull);
    if (e && in) { kex[i] = (cell << FB) | ce[i]; kam[i] = (cell << FB) | gene; }      // ambiguous: exon matrix + per-gene ambiguous count
    else if (in) kin[i] = (cell << FB) | ci[i];                                          // unspliced
    else kex[i] = (cell << FB) | ce[i];                                                  // spliced
}
// ---- intron_rule::kUmiGeneCR (pileup_engine.h finalize_all): a UMI in more than one gene of a cell keeps the gene whose molecule has the most reads;
// a tie drops all of them. The UMI of a molecule is its component root in DirectionalUmiStore::collapse_molecules: the first member in (reads
// descending, UMI ascending) order that the sequential pairwise union leaves as root. That is not always the best-supported UMI (a later
// member joining the component through a lower-ranked root re-roots it), so the union is replayed per (cell, gene) group in the CPU's order.
__global__ void k_cr_rankkey(const uint32_t* seg, const Agg* ag, int m, uint64_t* key, uint32_t* idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= m) return;
    key[i] = ((uint64_t)seg[i] << 32) | (uint64_t)(0xFFFFFFFFu - ag[i].count); idx[i] = i; }
__global__ void k_cr_rankpos(const uint32_t* perm, int m, uint32_t* rp) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < m) rp[perm[i]] = i; }
__device__ __forceinline__ uint32_t cr_find(uint32_t* par, uint32_t x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; }
// one thread per (cell, gene) group (entries [gs, ge); the group's rank positions are the same index range): rumi[e] = UMI of the root of e's molecule
__global__ void k_cr_roots(const uint64_t* uk, const Agg* ag, const uint32_t* perm, const uint32_t* rp, const uint32_t* sf, int m, int ub, int ulen, int directional,
                           uint32_t* par, uint64_t* rumi) {
    int gs = blockIdx.x * blockDim.x + threadIdx.x; if (gs >= m || !sf[gs]) return;
    int ge = gs + 1; while (ge < m && !sf[ge]) ge++;
    const uint64_t umask = (1ull << ub) - 1;
    for (int p = gs; p < ge; p++) par[p] = p;
    if (directional && ge - gs > 1)
        for (int p = gs; p < ge; p++) {
            const uint32_t e = perm[p]; const uint64_t ke = uk[e], u = ke & umask, hi = ke & ~umask; const uint32_t ce = ag[e].count;
            for (int k = 0; k < ulen; k++) {
                const uint64_t cur = (u >> (2 * k)) & 3;
                for (uint64_t b = 0; b < 4; b++) {
                    if (b == cur) continue;
                    const uint64_t cand = hi | (u ^ ((cur ^ b) << (2 * k)));
                    int lo = gs, h = ge; while (lo < h) { int mid = (lo + h) >> 1; if (uk[mid] < cand) lo = mid + 1; else h = mid; }
                    if (lo >= ge || uk[lo] != cand) continue;
                    const uint32_t q = rp[lo];
                    if ((int)q > p && count_rule(ce, ag[lo].count)) { uint32_t ri = cr_find(par, p), rj = cr_find(par, q); if (ri != rj) par[rj] = ri; }
                }
            }
        }
    for (int p = gs; p < ge; p++) rumi[perm[p]] = uk[perm[cr_find(par, p)]] & umask;
}
__global__ void k_cr_csum(const Agg* ag, const uint32_t* lab, int m, uint32_t* csum) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < m) atomicAdd(&csum[lab[i]], ag[i].count); }
__global__ void k_cr_isroot(const uint32_t* lab, int m, uint8_t* f) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < m) f[i] = lab[i] == (uint32_t)i; }
__global__ void k_cr_molkey(const uint64_t* uk, const uint64_t* rumi, const uint32_t* ridx, int nr, int gb, int ub, uint64_t* key) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= nr) return; const uint32_t i = ridx[j];
    key[j] = ((uk[i] >> (gb + ub)) << ub) | rumi[i]; }
// one thread per (cell, root UMI) run of molecules: more than one gene -> keep the single best-supported molecule, a tie keeps none
__global__ void k_cr_judge(const uint64_t* key, const uint32_t* ridx, const uint32_t* csum, const uint64_t* uk, int nr, int gb, int ub, uint8_t* drop) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= nr || (j > 0 && key[j] == key[j - 1])) return;
    int e = j + 1; while (e < nr && key[e] == key[j]) e++;
    if (e - j < 2) return;
    uint32_t best = 0, best_gene = 0xFFFFFFFFu; bool tie = false; const uint64_t gm = (1ull << gb) - 1;
    for (int t = j; t < e; t++) {
        const uint32_t c = csum[ridx[t]];
        if (c > best) { best = c; best_gene = (uint32_t)((uk[ridx[t]] >> ub) & gm); tie = false; } else if (c == best) tie = true; }
    for (int t = j; t < e; t++) drop[ridx[t]] = tie || (uint32_t)((uk[ridx[t]] >> ub) & gm) != best_gene;
}
// drop[i] = 1 for the molecules (component roots i) the rule removes; the temporaries are released, drop stays
static uint8_t* stage_cr(Arena& A, const uint64_t* uk, const Agg* ag, const uint32_t* lab, const uint32_t* seg, const uint32_t* sf, int m, const Layout& L) {
    const int TB = 256; const int G = (m + TB - 1) / TB;
    uint8_t* drop = A.alloc<uint8_t>(m); CK(cudaMemset(drop, 0, m));
    const size_t mk_ = A.mark();
    uint64_t *k0 = A.alloc<uint64_t>(m), *k1 = A.alloc<uint64_t>(m); uint32_t *p0 = A.alloc<uint32_t>(m), *p1 = A.alloc<uint32_t>(m);
    int segbits = 1; { uint32_t sm = 0; CK(cudaMemcpy(&sm, seg + (m - 1), 4, cudaMemcpyDeviceToHost)); while ((1ull << segbits) <= (unsigned long long)sm) segbits++; }
    k_cr_rankkey<<<G, TB>>>(seg, ag, m, k0, p0);
    cub::DoubleBuffer<uint64_t> dk(k0, k1); cub::DoubleBuffer<uint32_t> dv(p0, p1);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, dk, dv, m, 0, 32 + segbits); });   // stable: equal counts stay in UMI order
    const uint32_t* perm = dv.Current();
    uint32_t* rp = A.alloc<uint32_t>(m); k_cr_rankpos<<<G, TB>>>(perm, m, rp);
    uint32_t* par = A.alloc<uint32_t>(m); uint64_t* rumi = A.alloc<uint64_t>(m);
    k_cr_roots<<<G, TB>>>(uk, ag, perm, rp, sf, m, L.ub, L.ulen, L.directional ? 1 : 0, par, rumi);
    uint32_t* csum = A.alloc<uint32_t>(m); CK(cudaMemset(csum, 0, (size_t)m * 4)); k_cr_csum<<<G, TB>>>(ag, lab, m, csum);
    uint8_t* isr = A.alloc<uint8_t>(m); k_cr_isroot<<<G, TB>>>(lab, m, isr);
    uint32_t* ridx = A.alloc<uint32_t>(m); int* dn = A.alloc<int>(1);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceSelect::Flagged(t, b, cub::CountingInputIterator<uint32_t>(0), isr, ridx, dn, m); });
    int nr = 0; CK(cudaMemcpy(&nr, dn, 4, cudaMemcpyDeviceToHost));
    if (nr >= 2) {
        uint64_t *mk0 = A.alloc<uint64_t>(nr), *mk1 = A.alloc<uint64_t>(nr); uint32_t* mi1 = A.alloc<uint32_t>(nr);
        const int GR = (nr + TB - 1) / TB;
        k_cr_molkey<<<GR, TB>>>(uk, rumi, ridx, nr, L.gb, L.ub, mk0);
        cub::DoubleBuffer<uint64_t> mk(mk0, mk1); cub::DoubleBuffer<uint32_t> mi(ridx, mi1);
        cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, mk, mi, nr, 0, L.cb + L.ub); });
        k_cr_judge<<<GR, TB>>>(mk.Current(), mi.Current(), csum, uk, nr, L.gb, L.ub, drop);
    }
    CK(cudaDeviceSynchronize());
    A.reset(mk_);
    return drop;
}
struct IsSet { __host__ __device__ bool operator()(const uint64_t& k) const { return k != ~0ull; } };

struct Mol { uint64_t *kex, *kin, *kam; int m; };
static Mol stage_c(Arena& A, Timer& T, const uint64_t* uk, const Agg* ag, int m, const Layout& L, long long* stat_edges, long long* stat_active, int* stat_iters,
                   const uint32_t** lab_out = nullptr) {   // lab_out: the component (root entry index) of each unique (cell, gene, UMI) entry
    const int TB = 256; int G = (m + TB - 1) / TB; int ub = L.ub;
    uint32_t* sf = A.alloc<uint32_t>(m); uint32_t* seg = A.alloc<uint32_t>(m); uint8_t* act = A.alloc<uint8_t>(m); int* dnum = A.alloc<int>(8);
    k_segflag<<<G, TB>>>(uk, m, ub, sf);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceScan::InclusiveSum(t, b, sf, seg, m); });
    k_active<<<G, TB>>>(sf, m, act);
    uint32_t* actidx = A.alloc<uint32_t>(m);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceSelect::Flagged(t, b, cub::CountingInputIterator<uint32_t>(0), act, actidx, dnum, m); });
    int na = 0, nseg = 0; CK(cudaMemcpy(&na, dnum, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&nseg, seg + (m - 1), 4, cudaMemcpyDeviceToHost));
    *stat_active = na; T.mark("c.seg");
    uint32_t* lab = A.alloc<uint32_t>(m); k_iota<<<G, TB>>>(lab, m);
    uint64_t umask = (L.ub >= 64) ? ~0ull : ((1ull << L.ub) - 1);
    int lob = 2 * (L.ulen / 2), hib = 2 * (L.ulen - L.ulen / 2);
    int segbits = 1; while ((1ull << segbits) <= (unsigned long long)nseg + 1) segbits++;
    long long ne_total = 0; int iters = 0;
    if (na > 0 && L.directional) {
        uint32_t* ea[2]; uint32_t* eb[2]; int ne[2];
        for (int h = 0; h < 2; h++) {
            int hb = h == 0 ? lob : hib;
            if (segbits + hb > 64) die("segment id + half does not fit 64 bits");
            uint64_t *K0 = A.alloc<uint64_t>(na), *K1 = A.alloc<uint64_t>(na); uint32_t *V0 = A.alloc<uint32_t>(na), *V1 = A.alloc<uint32_t>(na);
            int GA = (na + TB - 1) / TB;
            k_mkK<<<GA, TB>>>(uk, seg, actidx, na, umask, lob, h, K0, V0); k_mkK2<<<GA, TB>>>(seg, na, hb, K0, V0);
            cub::DoubleBuffer<uint64_t> dk(K0, K1); cub::DoubleBuffer<uint32_t> dv(V0, V1);
            cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, dk, dv, na, 0, segbits + hb); });
            uint32_t* cnt = A.alloc<uint32_t>(na); uint32_t* off = A.alloc<uint32_t>(na);
            k_edge_count<<<GA, TB>>>(uk, ag, dk.Current(), dv.Current(), na, umask, cnt);
            cub_run(A, [&](void* t, size_t& b) { return cub::DeviceScan::ExclusiveSum(t, b, cnt, off, na); });
            uint32_t lastc = 0, lasto = 0; CK(cudaMemcpy(&lastc, cnt + na - 1, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&lasto, off + na - 1, 4, cudaMemcpyDeviceToHost));
            ne[h] = (int)(lasto + lastc);
            // edges must outlive the reset of the sort buffers: allocate them from a region above the mark, then keep the arena offset
            ea[h] = A.alloc<uint32_t>(ne[h] + 1); eb[h] = A.alloc<uint32_t>(ne[h] + 1);
            if (ne[h] > 0) k_edge_write<<<GA, TB>>>(uk, ag, dk.Current(), dv.Current(), na, umask, off, ea[h], eb[h]);
            ne_total += ne[h];
        }
        T.mark("c.cand");
        *stat_edges = ne_total;
        int* changed = A.alloc<int>(1);
        for (iters = 0; iters < 200; iters++) {
            CK(cudaMemsetAsync(changed, 0, 4));
            for (int h = 0; h < 2; h++) if (ne[h] > 0) k_hook<<<(ne[h] + TB - 1) / TB, TB>>>(ea[0 + h], eb[0 + h], ne[h], lab, changed);
            int ch = 0; CK(cudaMemcpy(&ch, changed, 4, cudaMemcpyDeviceToHost));
            if (!ch) break;
            k_compress<<<G, TB>>>(lab, m);
        }
        if (iters >= 200) die("connected components did not converge");
        T.mark("c.cc");
    } else { T.mark("c.cand"); T.mark("c.cc"); }
    *stat_iters = iters;
    if (lab_out) *lab_out = lab;
    uint32_t *cf = A.alloc<uint32_t>(m), *ce = A.alloc<uint32_t>(m), *ci = A.alloc<uint32_t>(m);
    k_comp_init<<<G, TB>>>(cf, ce, ci, m); k_comp_agg<<<G, TB>>>(ag, lab, m, cf, ce, ci);
    Mol M; M.m = m; M.kex = A.alloc<uint64_t>(m); M.kin = A.alloc<uint64_t>(m); M.kam = A.alloc<uint64_t>(m);
    const uint8_t* drop = L.cr ? stage_cr(A, uk, ag, lab, seg, sf, m, L) : nullptr;
    k_emit<<<G, TB>>>(uk, lab, cf, ce, ci, m, L.gb, ub, M.kex, M.kin, M.kam, drop);
    T.mark("c.emit");
    return M;
}

// ---- stage d ----
__global__ void k_csr_ptr(const uint64_t* uq, int nr, int ncells, uint32_t* indptr) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c > ncells) return;
    int lo = 0, hi = nr;                       // first j with (uq[j] >> FB) >= c
    while (lo < hi) { int mid = (lo + hi) >> 1; if ((uq[mid] >> FB) < (uint64_t)c) lo = mid + 1; else hi = mid; }
    indptr[c] = lo;
}
__global__ void k_csr_fill(const uint64_t* uq, const uint32_t* cnt, int nr, uint32_t* ind, uint16_t* dat) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= nr) return;
    ind[j] = (uint32_t)(uq[j] & ((1ull << FB) - 1)); dat[j] = (uint16_t)min(cnt[j], 65535u);
}
struct CsrDev { uint32_t* indptr; uint32_t* indices; uint16_t* data; int nnz; };
static CsrDev stage_d_one(Arena& A, const uint64_t* keys, int m, const Layout& L) {
    const int TB = 256; int* dnum = A.alloc<int>(4);
    uint64_t* sel = A.alloc<uint64_t>(m);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceSelect::If(t, b, keys, sel, dnum, m, IsSet()); });
    int n = 0; CK(cudaMemcpy(&n, dnum, 4, cudaMemcpyDeviceToHost));
    CsrDev C; C.indptr = A.alloc<uint32_t>(L.ncells + 1); C.nnz = 0;
    if (n == 0) { CK(cudaMemset(C.indptr, 0, 4 * (L.ncells + 1))); C.indices = A.alloc<uint32_t>(1); C.data = A.alloc<uint16_t>(1); return C; }
    uint64_t* s1 = A.alloc<uint64_t>(n); cub::DoubleBuffer<uint64_t> dk(sel, s1);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRadixSort::SortKeys(t, b, dk, n, 0, L.cb + FB); });
    uint64_t* uq = A.alloc<uint64_t>(n); uint32_t* cnt = A.alloc<uint32_t>(n);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRunLengthEncode::Encode(t, b, dk.Current(), uq, cnt, dnum, n); });
    int nr = 0; CK(cudaMemcpy(&nr, dnum, 4, cudaMemcpyDeviceToHost));
    C.indices = A.alloc<uint32_t>(nr); C.data = A.alloc<uint16_t>(nr); C.nnz = nr;
    k_csr_ptr<<<(L.ncells + 1 + TB - 1) / TB, TB>>>(uq, nr, L.ncells, C.indptr);
    k_csr_fill<<<(nr + TB - 1) / TB, TB>>>(uq, cnt, nr, C.indices, C.data);
    return C;
}

// ------------------------------------------------------------------------------------------------------------------------
// host side
struct Csr { std::vector<uint32_t> indptr, indices; std::vector<uint16_t> data; };
static std::string csr_sha(const Csr& a, const Csr& b, const Csr& c) {
    Sha256 s; for (const Csr* x : {&a, &b, &c}) { s.update(x->indptr.data(), x->indptr.size() * 4); s.update(x->indices.data(), x->indices.size() * 4); s.update(x->data.data(), x->data.size() * 2); }
    return s.hex();
}
static Csr fetch_csr(const CsrDev& C, int ncells) {
    Csr o; o.indptr.resize(ncells + 1); CK(cudaMemcpy(o.indptr.data(), C.indptr, 4 * (ncells + 1), cudaMemcpyDeviceToHost));
    o.indices.resize(C.nnz); o.data.resize(C.nnz);
    if (C.nnz) { CK(cudaMemcpy(o.indices.data(), C.indices, 4 * (size_t)C.nnz, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(o.data.data(), C.data, 2 * (size_t)C.nnz, cudaMemcpyDeviceToHost)); }
    return o;
}
static Csr from_triplets(std::vector<std::array<uint32_t, 3>>& t, int ncells) {   // sorted by (col,row) already or sorted here
    std::sort(t.begin(), t.end(), [](const std::array<uint32_t, 3>& a, const std::array<uint32_t, 3>& b) { return a[1] != b[1] ? a[1] < b[1] : a[0] < b[0]; });
    Csr o; o.indptr.assign(ncells + 1, 0);
    for (auto& x : t) { o.indptr[x[1] + 1]++; o.indices.push_back(x[0]); o.data.push_back((uint16_t)std::min<uint32_t>(x[2], 65535u)); }
    for (int c = 0; c < ncells; c++) o.indptr[c + 1] += o.indptr[c];
    return o;
}
struct DiffStat { long long identical = 0, only_a = 0, only_b = 0, value_diff = 0, sum_a = 0, sum_b = 0; };
static DiffStat csr_diff(const Csr& a, const Csr& b, int ncells, int show) {
    DiffStat d; int shown = 0;
    for (int c = 0; c < ncells; c++) {
        uint32_t i = a.indptr[c], ie = a.indptr[c + 1], j = b.indptr[c], je = b.indptr[c + 1];
        while (i < ie || j < je) {
            if (j == je || (i < ie && a.indices[i] < b.indices[j])) { d.only_a++; d.sum_a += a.data[i]; if (shown++ < show) fprintf(stderr, "  only A: cell %d feat %u val %u\n", c, a.indices[i], a.data[i]); i++; }
            else if (i == ie || b.indices[j] < a.indices[i]) { d.only_b++; d.sum_b += b.data[j]; if (shown++ < show) fprintf(stderr, "  only B: cell %d feat %u val %u\n", c, b.indices[j], b.data[j]); j++; }
            else { d.sum_a += a.data[i]; d.sum_b += b.data[j]; if (a.data[i] == b.data[j]) d.identical++; else { d.value_diff++; if (shown++ < show) fprintf(stderr, "  diff: cell %d feat %u A %u B %u\n", c, a.indices[i], a.data[i], b.data[j]); } i++; j++; }
        }
    }
    return d;
}
struct Truth { uint32_t umi_len = 0, directional = 0, n_barcodes = 0; Csr cls[3]; bool ok = false; };
static Truth load_truth(const char* path) {
    Truth T; FILE* f = fopen(path, "rb"); if (!f) return T;
    char mg[8]; uint32_t h[4]; if (fread(mg, 1, 8, f) != 8 || memcmp(mg, "M6TRU1", 6)) die("bad truth magic");
    if (fread(h, 4, 4, f) != 4) die("truth header"); T.umi_len = h[0]; T.directional = h[1]; T.n_barcodes = h[2];
    for (int s = 0; s < 3; s++) {
        uint32_t nr, nc; uint64_t nnz; if (fread(&nr, 4, 1, f) != 1 || fread(&nc, 4, 1, f) != 1 || fread(&nnz, 8, 1, f) != 1) die("truth section");
        std::vector<std::array<uint32_t, 3>> t(nnz); for (auto& x : t) if (fread(x.data(), 4, 3, f) != 3) die("truth triplet");
        T.cls[s] = from_triplets(t, (int)T.n_barcodes);
    }
    fclose(f); T.ok = true; return T;
}

// literal host reference of the CPU rule (umi_dedup.h record_read / finalize_molecules + count_read direct path)
static void host_reference(const std::vector<Rec>& recs, int umi_len, bool directional, int ncells, Csr out[3]) {
    struct Me { uint64_t packed; uint32_t exon, intron, count; uint8_t flags; };
    std::unordered_map<uint64_t, std::vector<Me>> groups; groups.reserve(recs.size() / 2);
    std::vector<std::array<uint32_t, 3>> te, ti, ta;
    for (const Rec& r : recs) {
        if (r.direct) { if (r.flags & 2) ti.push_back({r.intron, r.bc, 1}); else te.push_back({r.exon, r.bc, 1}); continue; }
        auto& g = groups[((uint64_t)r.bc << 32) | r.gene]; bool f = false;
        for (auto& e : g) if (e.packed == r.umi) { e.count++; e.flags |= r.flags; e.exon = std::min(e.exon, r.exon); e.intron = std::min(e.intron, r.intron); f = true; break; }
        if (!f) g.push_back({r.umi, r.exon, r.intron, 1u, r.flags});
    }
    auto ham = [&](uint64_t a, uint64_t b) { uint64_t d = a ^ b; if (umi_len < 32) d &= (1ull << (2 * umi_len)) - 1; return __builtin_popcountll((d | (d >> 1)) & 0x5555555555555555ull); };
    auto emit = [&](uint32_t bc, uint32_t gene, uint8_t fl, uint32_t ex, uint32_t in) {
        bool e = fl & 1, i = fl & 2;
        if (e && i) { te.push_back({ex, bc, 1}); ta.push_back({gene, bc, 1}); } else if (i) ti.push_back({in, bc, 1}); else te.push_back({ex, bc, 1});
    };
    for (auto& kv : groups) {
        uint32_t bc = kv.first >> 32, gene = (uint32_t)kv.first; auto& en = kv.second;
        if (!directional || en.size() == 1) { for (auto& e : en) emit(bc, gene, e.flags, e.exon, e.intron); continue; }
        std::vector<Me> s(en.begin(), en.end());
        std::sort(s.begin(), s.end(), [](const Me& a, const Me& b) { return a.count != b.count ? a.count > b.count : a.packed < b.packed; });
        int n = (int)s.size(); std::vector<int> par(n); for (int i = 0; i < n; i++) par[i] = i;
        auto find = [&](int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };
        for (int i = 0; i < n - 1; i++) for (int j = i + 1; j < n; j++)
            if (ham(s[i].packed, s[j].packed) == 1 && s[i].count >= 2 * s[j].count - 1) { int ri = find(i), rj = find(j); if (ri != rj) par[rj] = ri; }
        std::vector<uint8_t> cf(n, 0); std::vector<uint32_t> ce(n, NONE32), ci(n, NONE32);
        for (int i = 0; i < n; i++) { int r = find(i); cf[r] |= s[i].flags; ce[r] = std::min(ce[r], s[i].exon); ci[r] = std::min(ci[r], s[i].intron); }
        for (int i = 0; i < n; i++) if (find(i) == i) emit(bc, gene, cf[i], ce[i], ci[i]);
    }
    out[0] = from_triplets(te, ncells); out[1] = from_triplets(ti, ncells); out[2] = from_triplets(ta, ncells);
}
// the triplet merge above counts duplicates by summing values: collapse them
static void sum_dups(Csr& c, int ncells) {
    Csr o; o.indptr.assign(ncells + 1, 0);
    for (int cell = 0; cell < ncells; cell++) {
        uint32_t i = c.indptr[cell], e = c.indptr[cell + 1];
        while (i < e) { uint32_t f = c.indices[i]; uint32_t s = 0; while (i < e && c.indices[i] == f) { s += c.data[i]; i++; } o.indices.push_back(f); o.data.push_back((uint16_t)std::min<uint32_t>(s, 65535u)); }
        o.indptr[cell + 1] = (uint32_t)o.indices.size();
    }
    c = o;
}

// FASTQ R2 -> packed words (reads with N are counted and set aside; lengths kept exactly)
struct Reads { std::vector<uint64_t> w; std::vector<uint8_t> len; long long total = 0, with_n = 0, empty = 0; std::map<int, long long> lenhist; size_t maxlen = 0; };
static void read_fastq(const char* path, Reads& R, long long max_reads, int m2len) {
    FILE* f = strcmp(path, "-") == 0 ? stdin : fopen(path, "rb"); if (!f) die("cannot open --r2");
    long long ln = 0;
    std::vector<char> big(1 << 24); setvbuf(f, big.data(), _IOFBF, big.size());
    char* l = nullptr; size_t cap = 0; ssize_t k;
    while ((k = getline(&l, &cap, f)) > 0) {
        if ((ln++ & 3) != 1) continue;
        while (k > 0 && (l[k - 1] == '\n' || l[k - 1] == '\r')) k--;
        R.total++; R.lenhist[(int)k]++; R.maxlen = std::max<size_t>(R.maxlen, k);
        if (R.total > max_reads) { R.total--; R.lenhist[(int)k]--; break; }
        if (k == 0) { R.empty++; continue; }
        if (k > 32 * NW) { if (m2len == 0) { fprintf(stderr, "read length %zd > %d: rebuild with larger -DNW\n", k, 32 * NW); exit(2); } }
        bool hasn = false; uint64_t wd[NW]; for (int i = 0; i < NW; i++) wd[i] = 0;
        int use = m2len ? m2len : (int)k;
        if (m2len && k < m2len) continue;                                  // M2 definition: shorter reads dropped
        for (int i = 0; i < use; i++) {
            uint64_t b; switch (l[i]) { case 'A': b = 0; break; case 'C': b = 1; break; case 'G': b = 2; break; case 'T': b = 3; break; default: hasn = true; b = 0; }
            wd[i >> 5] |= b << (2 * (i & 31));
        }
        if (hasn) { R.with_n++; continue; }
        for (int i = 0; i < NW; i++) R.w.push_back(wd[i]); R.len.push_back((uint8_t)use);
    }
    free(l); if (f != stdin) fclose(f);
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    const char *r2 = nullptr, *tup = nullptr, *truthp = nullptr, *json = nullptr, *label = "m6"; std::vector<long long> batches = {0}; int reps = 3, m2len = 0;
    bool verify = false, verify_host = false; long long max_reads = 1ll << 60, max_tuples = 1ll << 60; double arena_gb = 60;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i]; auto nx = [&]() { if (i + 1 >= argc) die("missing value"); return argv[++i]; };
        if (a == "--r2") r2 = nx(); else if (a == "--tuples") tup = nx(); else if (a == "--truth") truthp = nx(); else if (a == "--json") json = nx(); else if (a == "--label") label = nx();
        else if (a == "--reps") reps = atoi(nx()); else if (a == "--m2len") m2len = atoi(nx()); else if (a == "--verify") verify = true; else if (a == "--verify-host") verify_host = true;
        else if (a == "--max-reads") max_reads = atoll(nx()); else if (a == "--max-tuples") max_tuples = atoll(nx()); else if (a == "--arena-gb") arena_gb = atof(nx());
        else if (a == "--batches") { batches.clear(); std::string s = nx(); size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); batches.push_back((long long)(atof(s.substr(p, q - p).c_str()) * 1e6)); p = q + 1; } }
        else die(("unknown arg " + a).c_str());
    }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0)); int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    fprintf(stderr, "device %s, %d SMs, driver CUDA %d, runtime %d, NW=%d\n", prop.name, prop.multiProcessorCount, drv, rt, NW);
    Arena A; A.init((size_t)(arena_gb * 1e9));
    std::string out; char tmp[1024];
    // ---------------- stage a: reads ----------------
    std::vector<std::string> jsec;
    auto fmt = [&](const char* f, double v) { snprintf(tmp, sizeof tmp, f, v); return std::string(tmp); };
    std::string ja, jbc;
    Reads R; double th = 0;
    if (r2) {
        double t0 = now_s(); read_fastq(r2, R, max_reads, m2len); double t1 = now_s();
        int n = (int)R.len.size(); fprintf(stderr, "reads: %lld total, %d usable (%lld with N, %lld empty, max len %zu) parse %.1fs\n", R.total, n, R.with_n, R.empty, R.maxlen, t1 - t0);
        // length distribution summary
        std::vector<std::pair<int, long long>> lh(R.lenhist.begin(), R.lenhist.end());
        std::string hist = "{"; bool first = true; for (auto& p : lh) { if (p.second * 1000 < R.total && p.second < 1000) continue; snprintf(tmp, sizeof tmp, "%s\"%d\": %lld", first ? "" : ", ", p.first, p.second); hist += tmp; first = false; } hist += "}";
        uint64_t* dW = nullptr; uint8_t* dL = nullptr;
        uint64_t* hW; uint8_t* hL; CK(cudaMallocHost(&hW, R.w.size() * 8)); CK(cudaMallocHost(&hL, n)); memcpy(hW, R.w.data(), R.w.size() * 8); memcpy(hL, R.len.data(), n);
        dW = A.alloc<uint64_t>(R.w.size()); dL = A.alloc<uint8_t>(n);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
        CK(cudaMemcpy(dW, hW, R.w.size() * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dL, hL, n, cudaMemcpyHostToDevice)); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float h2d_ms; cudaEventElapsedTime(&h2d_ms, e0, e1);
        size_t base_mark = A.mark(); Timer T; std::map<std::string, std::vector<double>> phase; std::string rows;
        for (size_t bi = 0; bi < batches.size(); bi++) {
            long long B = batches[bi]; std::vector<double> tot; std::string sha0; DedupResult last;
            std::vector<std::string> shas;
            for (int rep = -1; rep < reps; rep++) {
                A.reset(base_mark); T.start(); DedupResult D = dedup_batches(A, T, dW, dL, n, B); T.mark("a.end"); T.finish();
                double ms = 0; for (auto& kv : T.acc) ms += kv.second; if (rep >= 0) { tot.push_back(ms); for (auto& kv : T.acc) phase[std::to_string(B) + ":" + kv.first].push_back(kv.second); }
                std::vector<uint32_t> g(n); CK(cudaMemcpy(g.data(), D.gid, 4 * (size_t)n, cudaMemcpyDeviceToHost));
                Sha256 s; s.update(g.data(), 4 * (size_t)n); shas.push_back(s.hex()); last = D;
                if (verify && rep == 0 && bi == 0) {      // exact check against a host hash map on the full key
                    struct H { const uint64_t* w; const uint8_t* l; size_t operator()(uint32_t i) const { uint64_t h = l[i]; for (int j = 0; j < NW; j++) h = h * 1000003ull ^ (w[(size_t)i * NW + j] * 0x9e3779b97f4a7c15ull); return h; } };
                    struct E { const uint64_t* w; const uint8_t* l; bool operator()(uint32_t a, uint32_t b) const { return seq_eq(w + (size_t)a * NW, l[a], w + (size_t)b * NW, l[b]); } };
                    std::unordered_set<uint32_t, H, E> S(1 << 20, H{hW, hL}, E{hW, hL}); for (int i = 0; i < n; i++) S.insert(i);
                    std::vector<uint32_t> first(D.D_global, NONE32); long long bad = 0;
                    for (int i = 0; i < n; i++) { uint32_t gi = g[i]; if (gi >= D.D_global) { bad++; continue; } if (first[gi] == NONE32) first[gi] = i; else if (!seq_eq(hW + (size_t)i * NW, hL[i], hW + (size_t)first[gi] * NW, hL[first[gi]])) bad++; }
                    fprintf(stderr, "verify a: host distinct %zu, gpu distinct %lld, id/sequence violations %lld -> %s\n", S.size(), D.D_global, bad, (S.size() == (size_t)D.D_global && bad == 0) ? "EXACT" : "MISMATCH");
                    snprintf(tmp, sizeof tmp, "\"verify\": {\"host_distinct\": %zu, \"gpu_distinct\": %lld, \"violations\": %lld, \"exact\": %s}, ", S.size(), D.D_global, bad, (S.size() == (size_t)D.D_global && bad == 0) ? "true" : "false"); jbc += tmp;
                }
            }
            bool same = true; for (auto& s : shas) same &= (s == shas[0]);
            double med = median(tot);
            snprintf(tmp, sizeof tmp, "{\"batch_reads\": %lld, \"distinct_global\": %lld, \"d_global\": %.5f, \"distinct_sum_local\": %lld, \"d_local\": %.5f, \"mixed_run_elems\": %lld, \"median_ms\": %.3f, \"reads_per_s\": %.4g, \"gid_sha256\": \"%s\", \"runs_identical\": %s}",
                     B, last.D_global, (double)last.D_global / n, last.D_local, (double)last.D_local / n, last.mixed, med, n / (med / 1e3), shas[0].c_str(), same ? "true" : "false");
            rows += (bi ? ",\n    " : "") + std::string(tmp);
            fprintf(stderr, "a: B=%lld d=%.4f (local %.4f) %.1f ms = %.2f G reads/s, sha %s same=%d\n", B, (double)last.D_global / n, (double)last.D_local / n, med, n / (med / 1e3) / 1e9, shas[0].c_str(), (int)same);
        }
        std::string ph; bool f1 = true; for (auto& kv : phase) { snprintf(tmp, sizeof tmp, "%s\"%s\": %.3f", f1 ? "" : ", ", kv.first.c_str(), median(kv.second)); ph += tmp; f1 = false; }
        snprintf(tmp, sizeof tmp, "\"reads_total\": %lld, \"reads_used\": %d, \"reads_with_N_excluded\": %lld, \"reads_empty\": %lld, \"max_len\": %zu, \"m2len\": %d, \"words\": %d, \"h2d_ms\": %.3f, \"h2d_GBps\": %.2f, ",
                 R.total, n, R.with_n, R.empty, R.maxlen, m2len, NW, h2d_ms, (R.w.size() * 8.0 + n) / 1e9 / (h2d_ms / 1e3));
        ja = std::string("\"stage_a\": {") + tmp + jbc + "\"length_histogram\": " + hist + ", \"phase_median_ms\": {" + ph + "}, \"runs\": [\n    " + rows + "\n  ]}";
        A.reset(0);
    }
    // ---------------- stages b-d: tuples ----------------
    std::string jt;
    if (tup) {
        FILE* f = fopen(tup, "rb"); if (!f) die("cannot open tuples"); char mg[8]; uint32_t hd[2];
        if (fread(mg, 1, 8, f) != 8 || memcmp(mg, "M6TUP1", 6) || fread(hd, 4, 2, f) != 2 || hd[0] != 32) die("bad tuple file");
        fseek(f, 0, SEEK_END); long long fsz = ftell(f); fseek(f, 16, SEEK_SET); long long n = std::min<long long>((fsz - 16) / 32, max_tuples);
        Rec* h; CK(cudaMallocHost(&h, n * 32)); if ((long long)fread(h, 32, n, f) != n) die("short tuple file"); fclose(f);
        uint32_t maxbc = 0, maxgene = 0, maxex = 0, maxin = 0; long long ndirect = 0, nN = 0; int umin = 99, umax = 0, bad_flags = 0;
        for (long long i = 0; i < n; i++) {
            const Rec& r = h[i]; maxbc = std::max(maxbc, r.bc); maxgene = std::max(maxgene, r.gene);
            if (r.exon != NONE32) maxex = std::max(maxex, r.exon); if (r.intron != NONE32) maxin = std::max(maxin, r.intron);
            ndirect += r.direct; nN += r.hasN; if (!r.direct) { umin = std::min<int>(umin, r.umi_len); umax = std::max<int>(umax, r.umi_len); }
            bool exn = r.exon == NONE32, inn = r.intron == NONE32; uint8_t want = !exn && inn ? 5 : (!exn && !inn ? 6 : 2); if (r.flags != want) bad_flags++;
        }
        Truth TR; if (truthp) TR = load_truth(truthp);
        Layout L; auto bits = [](uint64_t v) { int b = 1; while ((1ull << b) <= v) b++; return b; };
        L.ulen = umax; L.cb = bits(maxbc); L.gb = bits(maxgene); L.ub = 2 * umax; L.directional = TR.ok ? TR.directional : true; L.ncells = TR.ok ? (int)TR.n_barcodes : (int)maxbc + 1;
        fprintf(stderr, "tuples: %lld, max bc %u gene %u exon %u intron %u, umi_len %d..%d, direct %lld, UMI with N %lld, flags!=derived %d; key bits cell %d gene %d umi %d = %d\n",
                n, maxbc, maxgene, maxex, maxin, umin, umax, ndirect, nN, bad_flags, L.cb, L.gb, L.ub, L.cb + L.gb + L.ub);
        if (umin != umax) die("mixed UMI lengths"); if (L.cb + L.gb + L.ub > 64) die("key does not fit 64 bits"); if (maxex >= 0xFFFFFE || maxin >= 0xFFFFFE || maxgene >= 0xFFFFFE) die("feature index >= 2^24"); if (ndirect) die("direct (no-UMI) tuples present");
        Rec* dR = nullptr; dR = A.alloc<Rec>(n);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0); CK(cudaMemcpy(dR, h, n * 32, cudaMemcpyHostToDevice)); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float h2d_ms; cudaEventElapsedTime(&h2d_ms, e0, e1);
        size_t base = A.mark(); Timer T; std::map<std::string, std::vector<double>> phase; std::string rows; Csr cpu_ref[3]; bool have_ref = false; std::string truth_sha;
        if (TR.ok) truth_sha = csr_sha(TR.cls[0], TR.cls[1], TR.cls[2]);
        if (verify_host) { std::vector<Rec> v(h, h + n); double t0 = now_s(); host_reference(v, umax, L.directional, L.ncells, cpu_ref); for (int c = 0; c < 3; c++) sum_dups(cpu_ref[c], L.ncells); have_ref = true; fprintf(stderr, "host reference: %.1f s, sha %s\n", now_s() - t0, csr_sha(cpu_ref[0], cpu_ref[1], cpu_ref[2]).c_str()); }
        std::string eqjson;
        for (size_t bi = 0; bi < batches.size(); bi++) {
            long long B = batches[bi]; std::vector<std::string> shas; std::vector<double> totb, totc, totd; long long nedges = 0, nact = 0; int iters = 0, m_uniq = 0; long long nmol[3] = {0, 0, 0}; double d2h_ms = 0;
            for (int rep = -1; rep < reps; rep++) {
                A.reset(base); T.start(); uint64_t* uk; Agg* ag;
                int m = stage_b(A, T, dR, (int)n, B, L, &uk, &ag); T.mark("b.end");
                Mol M = stage_c(A, T, uk, ag, m, L, &nedges, &nact, &iters); T.mark("c.end");
                CsrDev C[3] = {stage_d_one(A, M.kex, m, L), stage_d_one(A, M.kin, m, L), stage_d_one(A, M.kam, m, L)}; T.mark("d.end"); T.finish();
                double tb = 0, tc = 0, td = 0; for (auto& kv : T.acc) { if (kv.first[0] == 'b') tb += kv.second; else if (kv.first[0] == 'c') tc += kv.second; else if (kv.first[0] == 'd') td += kv.second; }
                if (rep >= 0) { totb.push_back(tb); totc.push_back(tc); totd.push_back(td); for (auto& kv : T.acc) phase[std::to_string(B) + ":" + kv.first].push_back(kv.second); }
                double t0 = now_s(); Csr G[3]; for (int c = 0; c < 3; c++) G[c] = fetch_csr(C[c], L.ncells); d2h_ms = (now_s() - t0) * 1e3; m_uniq = m;
                for (int c = 0; c < 3; c++) nmol[c] = G[c].indices.size();
                std::string sha = csr_sha(G[0], G[1], G[2]); shas.push_back(sha);
                if (rep == 0) {
                    const char* cn[3] = {"exon", "intron", "ambiguous"}; std::string e = "{";
                    for (int c = 0; c < 3; c++) {
                        DiffStat d{}; bool have = false;
                        if (TR.ok) { d = csr_diff(G[c], TR.cls[c], L.ncells, 5); have = true; }
                        snprintf(tmp, sizeof tmp, "%s\"%s\": {\"gpu_nnz\": %zu, \"cpu_nnz\": %zu, \"identical\": %lld, \"only_gpu\": %lld, \"only_cpu\": %lld, \"value_diff\": %lld, \"gpu_sum\": %lld, \"cpu_sum\": %lld}", c ? ", " : "", cn[c], G[c].indices.size(), TR.ok ? TR.cls[c].indices.size() : 0, d.identical, d.only_a, d.only_b, d.value_diff, d.sum_a, d.sum_b); e += tmp;
                        if (have) fprintf(stderr, "equiv B=%lld %s: gpu nnz %zu cpu nnz %zu identical %lld only_gpu %lld only_cpu %lld value_diff %lld\n", B, cn[c], G[c].indices.size(), TR.cls[c].indices.size(), d.identical, d.only_a, d.only_b, d.value_diff);
                        if (have_ref) { DiffStat r = csr_diff(G[c], cpu_ref[c], L.ncells, 3); fprintf(stderr, "  vs host-literal reference %s: identical %lld only_gpu %lld only_ref %lld value_diff %lld\n", cn[c], r.identical, r.only_a, r.only_b, r.value_diff); }
                    }
                    e += "}"; if (bi == 0) eqjson = e;
                }
            }
            bool same = true; for (auto& s : shas) same &= (s == shas[0]);
            double mb = median(totb), mc = median(totc), md = median(totd), mt = mb + mc + md;
            snprintf(tmp, sizeof tmp, "{\"batch_tuples\": %lld, \"unique_cell_gene_umi\": %d, \"active_entries\": %lld, \"edges\": %lld, \"cc_iterations\": %d, \"csr_nnz\": [%lld, %lld, %lld], \"b_ms\": %.3f, \"c_ms\": %.3f, \"d_ms\": %.3f, \"chain_ms\": %.3f, \"b_tuples_per_s\": %.4g, \"c_tuples_per_s\": %.4g, \"d_tuples_per_s\": %.4g, \"chain_tuples_per_s\": %.4g, \"d2h_ms\": %.3f, \"csr_sha256\": \"%s\", \"runs_identical\": %s, \"equals_cpu_truth_sha\": %s}",
                     B, m_uniq, nact, nedges, iters, nmol[0], nmol[1], nmol[2], mb, mc, md, mt, n / (mb / 1e3), n / (mc / 1e3), n / (md / 1e3), n / (mt / 1e3), d2h_ms, shas[0].c_str(), same ? "true" : "false", (TR.ok && shas[0] == truth_sha) ? "true" : "false");
            rows += (bi ? ",\n    " : "") + std::string(tmp);
            fprintf(stderr, "bcd: B=%lld unique %d active %lld edges %lld iters %d | b %.1f c %.1f d %.1f ms chain %.1f ms = %.2f G tuples/s | sha %s same=%d eq_truth=%d\n", B, m_uniq, nact, nedges, iters, mb, mc, md, mt, n / (mt / 1e3) / 1e9, shas[0].c_str(), (int)same, (int)(TR.ok && shas[0] == truth_sha));
        }
        std::string ph; bool f1 = true; for (auto& kv : phase) { snprintf(tmp, sizeof tmp, "%s\"%s\": %.3f", f1 ? "" : ", ", kv.first.c_str(), median(kv.second)); ph += tmp; f1 = false; }
        snprintf(tmp, sizeof tmp, "\"tuples\": %lld, \"umi_len\": %d, \"directional\": %s, \"umi_with_N\": %lld, \"key_bits\": [%d, %d, %d], \"flags_not_derived\": %d, \"h2d_ms\": %.3f, \"h2d_GBps\": %.2f, \"truth_csr_sha256\": \"%s\", ", n, umax, L.directional ? "true" : "false", nN, L.cb, L.gb, L.ub, bad_flags, h2d_ms, n * 32.0 / 1e9 / (h2d_ms / 1e3), truth_sha.c_str());
        jt = std::string("\"stage_bcd\": {") + tmp + "\"equivalence_vs_cpu_truth\": " + (eqjson.empty() ? "null" : eqjson) + ", \"phase_median_ms\": {" + ph + "}, \"runs\": [\n    " + rows + "\n  ]}";
    }
    out = "{\n  \"label\": \"" + std::string(label) + "\",\n  \"device\": \"" + prop.name + "\", \"driver_cuda\": " + std::to_string(drv) + ", \"runtime_cuda\": " + std::to_string(rt) + ", \"arena_peak_GB\": " + fmt("%.2f", A.peak / 1e9) + ",\n  ";
    if (!ja.empty()) out += ja + ",\n  "; if (!jt.empty()) out += jt + ",\n  ";
    out += "\"reps\": " + std::to_string(reps) + "\n}\n";
    if (json) { FILE* f = fopen(json, "w"); fputs(out.c_str(), f); fclose(f); } else fputs(out.c_str(), stdout);
    return 0;
}
