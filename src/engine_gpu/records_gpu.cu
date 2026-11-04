// E6: hops 6 and 7 of the end-to-end path on the device (API and contract: records_gpu.h; logic: records.cuh).
// E8: linked into the singlet binary; used only under SINGLET_ENGINE_GPU_E2E=1 (mapDistinct's record build, and e2ePass at the end of this
// file: the GPU pileup's hand-off pass on the device). bench/fable/records/records_test.cu is E6's vehicle.
#include "records_gpu.h"
#include "singlet/engine_gpu.h"
#include "singlet/tail_trace.h"   // E16: the level-5 switches (dlpar, regpar)
#include <cuda_runtime.h>
#include <sched.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <algorithm>

namespace {

using e6::u64;

#define RG_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { err = std::string(#x) + ": " + cudaGetErrorString(e_); return false; } } while (0)

struct Cnt4 { u64 a, o, s, b; };   // alignments, CIGAR ops, sequence bytes, hand-off record bytes
struct Cnt4Add { __host__ __device__ Cnt4 operator()(const Cnt4& x, const Cnt4& y) const { return Cnt4{x.a + y.a, x.o + y.o, x.s + y.s, x.b + y.b}; } };

__device__ unsigned long long g_st[4];   // mate gap, no exon, bad best, mapped

// thread per record: what it will write (one thread per record, no grid stride: the block counts need every thread)
__global__ void k_count(e6::RecParams P, const uint8_t* blob, const u64* blobOff, const u64* seqOff, u64 n, Cnt4* cnt) {
    const u64 r = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    uint8_t st = 0; bool mapped = false;
    if (r < n) {
        const u64 L = seqOff[r + 1] - seqOff[r];
        const e6::RecCount rc = e6::recordCount(P, blob + blobOff[r], L);
        cnt[r] = Cnt4{rc.nAln, rc.nOps, e6::seqBytes(L), e6::recBytes(rc.nAln, rc.nOps, L)};
        st = rc.status; mapped = rc.nAln > 0;
    }
    const int nm = __syncthreads_count(mapped);
    if (threadIdx.x == 0 && nm) atomicAdd(&g_st[3], (unsigned long long)nm);
    if (st & e6::ST_MATEGAP) atomicAdd(&g_st[0], 1ULL);   // rare: per-thread atomics
    if (st & e6::ST_NOEXON) atomicAdd(&g_st[1], 1ULL);
    if (st & e6::ST_BADBEST) atomicAdd(&g_st[2], 1ULL);
}

__global__ void k_fill(e6::RecParams P, const uint8_t* blob, const u64* blobOff, const char* seq, const u64* seqOff, u64 n, const u64* chrStart,
                       const Cnt4* off, SigRec* sigs, AlnRec* alns, uint32_t* ops, uint8_t* seq2, uint8_t* bytes, u64* bytesOff, uint16_t* nm) {
    const u64 r = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (r >= n) return;
    const Cnt4 o0 = off[r], o1 = off[r + 1];
    const u64 L = seqOff[r + 1] - seqOff[r];
    e6::recordFill(P, blob + blobOff[r], seq + seqOff[r], L, chrStart, (uint32_t)(o1.a - o0.a), (uint32_t)o0.a, (uint32_t)o0.o, o0.s, sigs + r,
                   alns + o0.a, ops, seq2 + o0.s, bytes ? bytes + o0.b : nullptr, nm ? nm + o0.a : nullptr);
    if (bytesOff) { bytesOff[r] = o0.b; if (r + 1 == n) bytesOff[n] = o1.b; }
}

// ---- hop 7
__global__ void k_crkeys(const uint32_t* rr, const e6::ReadEntry* E, u64 n, uint32_t cbL, u64* keys) {
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i >= n) return;
    keys[i] = rr[i] != 0xFFFFFFFFu ? e6::crKey(E[i], cbL) : ~0ULL;
}

__device__ unsigned long long g_out[e6::RO_N + 3];   // outcomes, lowQual sum, qualNeqN sum, kept reads without a mask

__global__ void k_eval(const uint32_t* rr, const e6::ReadEntry* E, u64 n, const SigRec* sigs, const AlnRec* alns, uint32_t cbL, uint32_t umiL,
                       const u64* keys, const int32_t* vals, u64 nKeys, uint32_t* keep) {
    __shared__ unsigned int sc[e6::RO_N];
    __shared__ unsigned long long sq[3];
    if (threadIdx.x < e6::RO_N) sc[threadIdx.x] = 0;
    if (threadIdx.x < 3) sq[threadIdx.x] = 0;
    __syncthreads();
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i < n) {
        const e6::ReadEntry e = E[i];
        int32_t bc; u64 u3;
        const e6::ReadOutcome o = e6::readEval(rr[i], sigs, alns, e, cbL, umiL, keys, vals, nKeys, bc, u3);
        atomicAdd(&sc[o], 1u);
        keep[i] = o == e6::RO_KEEP;
        if (o == e6::RO_KEEP) {
            atomicAdd(&sq[0], (unsigned long long)e.lowQual); atomicAdd(&sq[1], (unsigned long long)e.qualNeqN);
            if (e.flags & 64u) atomicAdd(&sq[2], 1ULL);
        }
    }
    __syncthreads();
    if (threadIdx.x < e6::RO_N && sc[threadIdx.x]) atomicAdd(&g_out[threadIdx.x], (unsigned long long)sc[threadIdx.x]);
    if (threadIdx.x < 3 && sq[threadIdx.x]) atomicAdd(&g_out[e6::RO_N + threadIdx.x], sq[threadIdx.x]);
}

__global__ void k_scatter(const uint32_t* rr, const e6::ReadEntry* E, const u64* Q, u64 n, const SigRec* sigs, const AlnRec* alns, uint32_t cbL,
                          uint32_t umiL, const u64* keys, const int32_t* vals, u64 nKeys, const uint32_t* keep, const uint32_t* pos, ReadRec* out,
                          u64* lowq) {
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i >= n || !keep[i]) return;
    const e6::ReadEntry e = E[i];
    int32_t bc = -1; u64 u3 = 0;
    e6::readEval(rr[i], sigs, alns, e, cbL, umiL, keys, vals, nKeys, bc, u3);
    const uint32_t k = pos[i];
    ReadRec x; x.sig = rr[i]; x.bc = bc; x.umi3 = u3;
    out[k] = x;
    if (lowq) {
        const bool has = !(e.flags & 64u);
        lowq[2 * (u64)k] = has ? Q[2 * i] : 0; lowq[2 * (u64)k + 1] = has ? Q[2 * i + 1] : 0;
    }
}

struct Ev {
    cudaEvent_t a, b;
    Ev() { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~Ev() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start() { cudaEventRecord(a); }
    float stop() { cudaEventRecord(b); cudaEventSynchronize(b); float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms; }
};

} // namespace

namespace singlet_records_gpu {

bool buildRecords(const e6::RecParams& P, const uint8_t* dBlob, const uint64_t* dBlobOff, const char* dSeq, const uint64_t* dSeqOff, uint64_t nRec,
                  const uint64_t* dChrStart, bool wantBytes, DeviceRecords& out, std::string& err, bool wantNm) {
    release(out);
    out.nRec = nRec;
    if (nRec == 0) return true;
    const unsigned T = 256, G = (unsigned)((nRec + T - 1) / T);
    Cnt4* dCnt; RG_CK(cudaMalloc(&dCnt, (nRec + 1) * sizeof(Cnt4)));
    RG_CK(cudaMemset(dCnt + nRec, 0, sizeof(Cnt4)));
    { unsigned long long z[4] = {0, 0, 0, 0}; RG_CK(cudaMemcpyToSymbol(g_st, z, sizeof z)); }
    Ev ev;
    ev.start();
    k_count<<<G, T>>>(P, dBlob, dBlobOff, dSeqOff, nRec, dCnt);
    RG_CK(cudaGetLastError());
    out.msCount = ev.stop();
    ev.start();
    thrust::exclusive_scan(thrust::device_ptr<Cnt4>(dCnt), thrust::device_ptr<Cnt4>(dCnt + nRec + 1), thrust::device_ptr<Cnt4>(dCnt), Cnt4{0, 0, 0, 0},
                           Cnt4Add());
    out.msScan = ev.stop();
    Cnt4 tot; RG_CK(cudaMemcpy(&tot, dCnt + nRec, sizeof tot, cudaMemcpyDeviceToHost));
    { unsigned long long s[4]; RG_CK(cudaMemcpyFromSymbol(s, g_st, sizeof s)); out.nMateGap = s[0]; out.nNoExon = s[1]; out.nBadBest = s[2]; out.nMapped = s[3]; }
    if (tot.a > 0xFFFFFFFFull || tot.o > 0xFFFFFFFFull) { cudaFree(dCnt); err = "more than 2^32 alignments or CIGAR ops (AlnRec / SigRec offsets are 32-bit)"; return false; }
    out.nAln = tot.a; out.nOps = tot.o; out.nSeq = tot.s; out.nBytes = wantBytes ? tot.b : 0;
    RG_CK(cudaMalloc(&out.sigs, nRec * sizeof(SigRec))); RG_CK(cudaMalloc(&out.alns, std::max<u64>(1, tot.a) * sizeof(AlnRec)));
    RG_CK(cudaMalloc(&out.ops, std::max<u64>(1, tot.o) * 4)); RG_CK(cudaMalloc(&out.seq, std::max<u64>(1, tot.s)));
    if (wantBytes) { RG_CK(cudaMalloc(&out.bytes, std::max<u64>(1, tot.b))); RG_CK(cudaMalloc(&out.bytesOff, (nRec + 1) * 8)); }
    if (wantNm) RG_CK(cudaMalloc(&out.nm, std::max<u64>(1, tot.a) * 2));
    ev.start();
    k_fill<<<G, T>>>(P, dBlob, dBlobOff, dSeq, dSeqOff, nRec, dChrStart, dCnt, out.sigs, out.alns, out.ops, out.seq, out.bytes, out.bytesOff, out.nm);
    RG_CK(cudaGetLastError());
    out.msFill = ev.stop();
    RG_CK(cudaDeviceSynchronize());
    cudaFree(dCnt);
    return true;
}

void release(DeviceRecords& r) {
    cudaFree(r.sigs); cudaFree(r.alns); cudaFree(r.ops); cudaFree(r.seq); cudaFree(r.bytes); cudaFree(r.bytesOff); cudaFree(r.nm);
    r = DeviceRecords();
}

bool distinctBarcodes(const uint32_t* dReadRecord, const e6::ReadEntry* dE, uint64_t nReads, uint32_t cbL, std::vector<uint64_t>& keys, float& ms,
                      std::string& err) {
    keys.clear(); ms = 0;
    if (nReads == 0) return true;
    u64* dK; RG_CK(cudaMalloc(&dK, nReads * 8));
    Ev ev; ev.start();
    k_crkeys<<<(unsigned)((nReads + 255) / 256), 256>>>(dReadRecord, dE, nReads, cbL, dK);
    thrust::sort(thrust::device_ptr<u64>(dK), thrust::device_ptr<u64>(dK + nReads));
    const u64 nU = (u64)(thrust::unique(thrust::device_ptr<u64>(dK), thrust::device_ptr<u64>(dK + nReads)) - thrust::device_ptr<u64>(dK));
    ms = ev.stop();
    keys.resize(nU);
    RG_CK(cudaMemcpy(keys.data(), dK, nU * 8, cudaMemcpyDeviceToHost));
    cudaFree(dK);
    if (!keys.empty() && keys.back() == ~0ULL) keys.pop_back();   // reads without a record
    return true;
}

bool expandReads(const uint32_t* dReadRecord, const e6::ReadEntry* dE, const uint64_t* dQ, uint64_t nReads, const DeviceRecords& R, uint32_t cbL,
                 uint32_t umiL, const uint64_t* dKeys, const int32_t* dVals, uint64_t nKeys, DeviceReads& out, std::string& err) {
    release(out);
    if (nReads == 0) return true;
    if (cbL > 21 || umiL > 21) { err = "CR or UR longer than 21 bases (HandoffReadEntry packs 21)"; return false; }
    const unsigned T = 256, G = (unsigned)((nReads + T - 1) / T);
    uint32_t *dKeep, *dPos;
    RG_CK(cudaMalloc(&dKeep, nReads * 4)); RG_CK(cudaMalloc(&dPos, nReads * 4));
    { unsigned long long z[e6::RO_N + 3] = {}; RG_CK(cudaMemcpyToSymbol(g_out, z, sizeof z)); }
    Ev ev;
    ev.start();
    k_eval<<<G, T>>>(dReadRecord, dE, nReads, R.sigs, R.alns, cbL, umiL, dKeys, dVals, nKeys, dKeep);
    RG_CK(cudaGetLastError());
    out.msEval = ev.stop();
    ev.start();
    thrust::exclusive_scan(thrust::device_ptr<uint32_t>(dKeep), thrust::device_ptr<uint32_t>(dKeep + nReads), thrust::device_ptr<uint32_t>(dPos));
    out.msScan = ev.stop();
    uint32_t lastP, lastK;
    RG_CK(cudaMemcpy(&lastP, dPos + nReads - 1, 4, cudaMemcpyDeviceToHost)); RG_CK(cudaMemcpy(&lastK, dKeep + nReads - 1, 4, cudaMemcpyDeviceToHost));
    out.nOut = (u64)lastP + lastK;
    { unsigned long long s[e6::RO_N + 3]; RG_CK(cudaMemcpyFromSymbol(s, g_out, sizeof s));
      for (int k = 0; k < e6::RO_N; k++) out.outcome[k] = s[k];
      out.lowQualSum = s[e6::RO_N]; out.qualNeqNSum = s[e6::RO_N + 1]; out.maskAll = s[e6::RO_N + 2] == 0; }
    RG_CK(cudaMalloc(&out.reads, std::max<u64>(1, out.nOut) * sizeof(ReadRec)));
    if (dQ) RG_CK(cudaMalloc(&out.lowq, std::max<u64>(1, out.nOut) * 16));
    ev.start();
    k_scatter<<<G, T>>>(dReadRecord, dE, dQ, nReads, R.sigs, R.alns, cbL, umiL, dKeys, dVals, nKeys, dKeep, dPos, out.reads, out.lowq);
    RG_CK(cudaGetLastError());
    out.msScatter = ev.stop();
    RG_CK(cudaDeviceSynchronize());
    cudaFree(dKeep); cudaFree(dPos);
    return true;
}

void release(DeviceReads& r) {
    cudaFree(r.reads); cudaFree(r.lowq);
    r = DeviceReads();
}

} // namespace singlet_records_gpu

// ---- E8: the hand-off pass on the device (engine_gpu.h e2ePass)
namespace {

static_assert(sizeof(singlet_engine_gpu::E2ECrVal) == sizeof(e6::CrVal), "CrVal layout");
enum : int { PC_ALN = 0, PC_SEC, PC_RSAMP, PC_RHOOK, PC_CHRM, PC_NOBC, PC_BARCODED, PC_MULTI, PC_LOWMAPQ, PC_DONE, PC_UNWRITTEN, PC_BIGCLIP,
             PC_NOHOOK, PC_NOKEY, PC_NOTINSET, PC_LOWQ, PC_NEQN, PC_NOMASK, PC_PEND, PC_RHOST, PC_CB0, PC_N = PC_CB0 + 8 };
__device__ unsigned long long g_pc[PC_N];

struct PassDev {
    const uint32_t* rr; const e6::ReadEntry* E; const u64* Q; u64 n;
    const SigRec* sigs; const AlnRec* alns; const uint32_t* ops; const uint16_t* nm;
    e6::PassParams pp; const u64* keys; const e6::CrVal* vals; u64 nKeys;
    uint32_t* cellReads; uint32_t* cellUnique; uint32_t* cellMulti; unsigned long long* cellMism; unsigned long long* cellBases;
    uint32_t nCells, nCellsQc, nWl; uint32_t* wl;
};

// thread per read: the pass's decision, every count (block-reduced), per-cell and WL-ambient counts (atomics), and the keep flag
__global__ void k_pass_eval(PassDev d, uint32_t* keep) {
    __shared__ unsigned long long sc[PC_N];
    for (int k = threadIdx.x; k < PC_N; k += blockDim.x) sc[k] = 0;
    __syncthreads();
    const u64 i = blockIdx.x * (u64) blockDim.x + threadIdx.x;
    if (i < d.n) {
        const e6::ReadEntry e = d.E[i];
        e6::PassRes r;
        e6::readPass(i, d.rr[i], d.sigs, d.alns, e, d.pp, d.keys, d.vals, d.nKeys, r);
        keep[i] = r.kind == e6::PK_KEEP;
        if (r.kind != e6::PK_NOREC && r.kind != e6::PK_UNMAPPED) {
            atomicAdd(&sc[PC_ALN], (unsigned long long) r.nAln);
            if (r.nSec) atomicAdd(&sc[PC_SEC], (unsigned long long) r.nSec);
        }
        if (r.kind >= e6::PK_UNWRITTEN) atomicAdd(&sc[PC_DONE], 1ULL);
        if (r.kind == e6::PK_UNWRITTEN) atomicAdd(&sc[PC_UNWRITTEN], 1ULL);
        if (r.kind == e6::PK_BIGCLIP) atomicAdd(&sc[PC_BIGCLIP], 1ULL);
        if (r.kind == e6::PK_NOHOOK) atomicAdd(&sc[PC_NOHOOK], 1ULL);
        if (r.kind == e6::PK_NOKEY) atomicAdd(&sc[PC_NOKEY], 1ULL);
        if (r.rrna & 1) atomicAdd(&sc[PC_RSAMP], 1ULL);
        if (r.rrna & 2) atomicAdd(&sc[PC_RHOOK], 1ULL);
        if (r.rrna & 4) atomicAdd(&sc[PC_RHOST], 1ULL);
        if (r.chrm) atomicAdd(&sc[PC_CHRM], 1ULL);
        if (r.cbStatus >= 0) atomicAdd(&sc[PC_CB0 + r.cbStatus], 1ULL);
        if (r.kind == e6::PK_NOBARCODE) atomicAdd(&sc[PC_NOBC], 1ULL);
        if (r.kind == e6::PK_NOTINSET) {
            atomicAdd(&sc[PC_NOTINSET], 1ULL);
            if (r.wlCount && (uint32_t) r.wl < d.nWl) { atomicAdd(&d.wl[r.wl], 1u); if (d.pp.wlPend) atomicAdd(&sc[PC_PEND], 1ULL); }
        }
        if (r.kind == e6::PK_KEEP) {
            atomicAdd(&sc[PC_BARCODED], 1ULL);
            atomicAdd(&sc[PC_LOWQ], (unsigned long long) e.lowQual); atomicAdd(&sc[PC_NEQN], (unsigned long long) e.qualNeqN);
            if (d.Q == nullptr || (e.flags & 64u)) atomicAdd(&sc[PC_NOMASK], 1ULL);
            if (r.multi) atomicAdd(&sc[PC_MULTI], 1ULL);
            if (r.lowMapq) atomicAdd(&sc[PC_LOWMAPQ], 1ULL);
            const uint32_t c = (uint32_t) r.col;
            if (c < d.nCells) atomicAdd(&d.cellReads[c], 1u);
            if (c < d.nCellsQc) {
                if (r.multi) atomicAdd(&d.cellMulti[c], 1u);
                else {
                    const SigRec s = d.sigs[d.rr[i]];
                    const AlnRec a = d.alns[s.aln_off + r.pk];
                    atomicAdd(&d.cellUnique[c], 1u);
                    atomicAdd(&d.cellMism[c], (unsigned long long) d.nm[s.aln_off + r.pk]);
                    atomicAdd(&d.cellBases[c], (unsigned long long) e6::matchLen(a, d.ops));
                }
            }
        }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < PC_N; k += blockDim.x) if (sc[k]) atomicAdd(&g_pc[k], sc[k]);
}

__device__ unsigned long long g_app[2];   // append cursors: WL-ambient pending, rRNA reads for the host detector

// kept reads at their scanned slot (read order); the WL-ambient pending pairs and the host-detector reads appended (their order is unused:
// both only feed sums)
__global__ void k_pass_scatter(PassDev d, const uint32_t* keep, const uint32_t* pos, ReadRec* out, uint32_t* clip, u64* lowq,
                               uint32_t* pend, u64 nPend, u64* rhost, u64 nRhost) {
    const u64 i = blockIdx.x * (u64) blockDim.x + threadIdx.x;
    if (i >= d.n) return;
    if (d.rr[i] == 0xFFFFFFFFu) return;
    const e6::ReadEntry e = d.E[i];
    e6::PassRes r;
    e6::readPass(i, d.rr[i], d.sigs, d.alns, e, d.pp, d.keys, d.vals, d.nKeys, r);
    if (r.rrna & 4) { const u64 k = atomicAdd(&g_app[1], 1ULL); if (k < nRhost) rhost[k] = ((u64) d.rr[i] << 1) | r.rev; }
    if (r.kind == e6::PK_NOTINSET && r.wlCount && (uint32_t) r.wl < d.nWl && d.pp.wlPend) {
        const u64 k = atomicAdd(&g_app[0], 1ULL);
        if (k < nPend) { pend[2 * k] = d.rr[i]; pend[2 * k + 1] = r.pk; }
    }
    if (r.kind != e6::PK_KEEP) return;
    const uint32_t k = pos[i];
    ReadRec x; x.sig = d.rr[i]; x.bc = r.col; x.umi3 = r.u3;
    out[k] = x; clip[k] = r.clip;
    const bool has = d.Q != nullptr && !(e.flags & 64u);
    lowq[2 * (u64) k] = has ? d.Q[2 * i] : 0; lowq[2 * (u64) k + 1] = has ? d.Q[2 * i + 1] : 0;
}

} // namespace

namespace singlet_engine_gpu {

namespace {
// device -> pageable host memory, page-locked in place for the copy when the driver allows it (GBs: several times the pageable rate)
// (keep: it stays page-locked, E16's pileupRun uploads it again)
// E16 (level 5): the first touch of GB-sized host memory on several threads (its page faults in parallel) before it is page-locked;
// write: zero-fills fresh memory (a vector's reserved capacity, see the download), else reads a byte per page (the region the STAR child wrote)
void e16Touch(uint8_t* p, u64 n, bool write) {
    cpu_set_t cs; int T = 8;
    if (sched_getaffinity(0, sizeof cs, &cs) == 0) T = CPU_COUNT(&cs);
    T = std::max(1, std::min(T, 16));
    if (n == 0) return;
    const u64 ch = (n / T + 4095) & ~4095ull;
    std::vector<std::thread> th;
    for (int t = 0; t < T; t++) {
        const u64 a = std::min(n, (u64) t * ch), b = std::min(n, a + ch);
        if (a >= b) break;
        th.emplace_back([=] {
            if (write) { memset(p + a, 0, b - a); return; }
            const volatile uint8_t* q = p; uint8_t s = 0;
            for (u64 i = a; i < b; i += 4096) s += q[i];
            s += q[b - 1];
            (void) s;
        });
    }
    for (auto& x : th) x.join();
}
cudaError_t d2h(void* dst, const void* src, u64 n, bool keep = false) {
    if (n == 0) return cudaSuccess;
    const bool pin = n >= (64ull << 20) && cudaHostRegister(dst, n, cudaHostRegisterDefault) == cudaSuccess;
    if (!pin) cudaGetLastError();
    const cudaError_t e = cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost);
    if (pin && !keep) cudaHostUnregister(dst);
    return e;
}
// device buffers of one e2ePass call, freed on every return
struct PassBufs {
    std::vector<void*> p;
    template <class T> cudaError_t alloc(T*& d, u64 bytes) { void* v = nullptr; cudaError_t e = cudaMalloc(&v, bytes + 16); if (e == cudaSuccess) p.push_back(v); d = (T*) v; return e; }
    void release() { for (void* v : p) cudaFree(v); p.clear(); }
    ~PassBufs() { release(); }
};
}

bool e2ePass(const E2EPassIn& in, int device, const std::function<void(const std::vector<uint64_t>&, std::vector<E2ECrVal>&)>& crTable,
             E2EPassOut& out) {
    out = E2EPassOut();
    auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    const double t0 = now();
    const E2ERegion& R = *in.region;
    const E2EHeader& H = *R.hdr();
    const u64 n = H.nReads, nRec = H.nRec, nAln = H.nAln, nOps = H.nOps;
    bool pinned = false;
    double e16reg = 0, e16pre = -1, e16dl = -1;
    auto fail = [&](const char* what, cudaError_t e) {
        out.log += std::string(what) + ": " + cudaGetErrorString(e) + "; ";
        if (pinned) cudaHostUnregister(R.base);
        return false;
    };
#define EP_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return fail(#x, e_); } while (0)
    cudaInitDevice(device, cudaDeviceScheduleBlockingSync, 0);
    if (cudaSetDevice(device) != cudaSuccess) { out.log += "cannot select device; "; return false; }
    cudaFree(0);   // E16: the context now, timed on its own
    const double e16ctx = now() - t0;
    PassBufs B;
    uint32_t *dRR, *dOps, *dKeep, *dPos, *dCellR, *dCellU, *dCellM, *dWl, *dClip, *dPend;
    e6::ReadEntry* dE; u64 *dQ = nullptr, *dK, *dLowq, *dRhost;
    SigRec* dS; AlnRec* dA; uint16_t* dNm; e6::CrVal* dV; ReadRec* dOut;
    unsigned long long *dCellMis, *dCellB;
    {   // the region's arrays to the device: page-locked in place when the driver allows it (the pages exist: the STAR child wrote them)
        const double tu = now();
        // E16 "regpar" (keepPinned, level 5): this process's mapping of the region populated by several threads before the page-locking
        if (in.keepPinned && singlet_tail::cut("regpar")) { e16Touch(R.base, std::min<u64>(R.cap, H.used), false); e16pre = now() - tu; }
        const double tr = now();
        pinned = cudaHostRegister(R.base, std::min<u64>(R.cap, (H.used + 4095) & ~4095ull), cudaHostRegisterDefault) == cudaSuccess;
        if (!pinned) cudaGetLastError();
        e16reg = now() - tr;
        EP_CK(B.alloc(dRR, n * 4)); EP_CK(cudaMemcpy(dRR, R.at<uint8_t>(H.oReadRecord), n * 4, cudaMemcpyHostToDevice));
        EP_CK(B.alloc(dE, n * 24)); EP_CK(cudaMemcpy(dE, R.at<uint8_t>(H.oEntries), n * 24, cudaMemcpyHostToDevice));
        if (H.oQMask) { EP_CK(B.alloc(dQ, n * 16)); EP_CK(cudaMemcpy(dQ, R.at<uint8_t>(H.oQMask), n * 16, cudaMemcpyHostToDevice)); }
        EP_CK(B.alloc(dS, nRec * 16)); EP_CK(cudaMemcpy(dS, R.at<uint8_t>(H.oSigs), nRec * 16, cudaMemcpyHostToDevice));
        EP_CK(B.alloc(dA, nAln * 16)); EP_CK(cudaMemcpy(dA, R.at<uint8_t>(H.oAlns), nAln * 16, cudaMemcpyHostToDevice));
        EP_CK(B.alloc(dOps, nOps * 4)); EP_CK(cudaMemcpy(dOps, R.at<uint8_t>(H.oOps), nOps * 4, cudaMemcpyHostToDevice));
        EP_CK(B.alloc(dNm, nAln * 2)); EP_CK(cudaMemcpy(dNm, R.at<uint8_t>(H.oNm), nAln * 2, cudaMemcpyHostToDevice));
        out.msUpload = (now() - tu) * 1e3;
    }
    std::vector<uint64_t> keys; std::vector<E2ECrVal> vals;
    {   // the distinct CRs of the reads with a record (E6 distinctBarcodes), then the caller's table
        const double tk = now();
        std::string err;
        float ms = 0;
        if (!singlet_records_gpu::distinctBarcodes(dRR, dE, n, H.cbL, keys, ms, err)) { out.log += err + "; "; if (pinned) cudaHostUnregister(R.base); return false; }
        out.msKeys = (now() - tk) * 1e3;
        const double tt = now();
        vals.assign(keys.size(), E2ECrVal{-2, -1, 0});
        crTable(keys, vals);
        out.msTable = (now() - tt) * 1e3;
        out.nKeys = keys.size();
        EP_CK(B.alloc(dK, keys.size() * 8)); EP_CK(B.alloc(dV, vals.size() * sizeof(e6::CrVal)));
        if (!keys.empty()) {
            EP_CK(cudaMemcpy(dK, keys.data(), keys.size() * 8, cudaMemcpyHostToDevice));
            EP_CK(cudaMemcpy(dV, vals.data(), vals.size() * sizeof(e6::CrVal), cudaMemcpyHostToDevice));
        }
    }
    const unsigned T = 256, G = (unsigned) ((n + T - 1) / T);
    PassDev d{};
    unsigned long long pc[PC_N] = {};
    double e16unreg = 0, e16reset = 0;
    u64 nKeep = 0;
    {
        const double te = now();
        const u64 nc = in.nCells, nq = in.nCellsQc, nw = in.nWl;
        EP_CK(B.alloc(dCellR, nc * 4)); EP_CK(cudaMemset(dCellR, 0, nc * 4 + 4));
        EP_CK(B.alloc(dCellU, nq * 4)); EP_CK(cudaMemset(dCellU, 0, nq * 4 + 4));
        EP_CK(B.alloc(dCellM, nq * 4)); EP_CK(cudaMemset(dCellM, 0, nq * 4 + 4));
        EP_CK(B.alloc(dCellMis, nq * 8)); EP_CK(cudaMemset(dCellMis, 0, nq * 8 + 8));
        EP_CK(B.alloc(dCellB, nq * 8)); EP_CK(cudaMemset(dCellB, 0, nq * 8 + 8));
        EP_CK(B.alloc(dWl, nw * 4)); EP_CK(cudaMemset(dWl, 0, nw * 4 + 4));
        EP_CK(B.alloc(dKeep, n * 4)); EP_CK(B.alloc(dPos, n * 4));
        d.rr = dRR; d.E = dE; d.Q = dQ; d.n = n; d.sigs = dS; d.alns = dA; d.ops = dOps; d.nm = dNm;
        d.pp = e6::PassParams{H.cbL, H.umiL, in.chrmTid, in.minMapq, in.cbc ? 1u : 0u, in.wlOn ? 1u : 0u, in.wlPend ? 1u : 0u};
        d.keys = dK; d.vals = dV; d.nKeys = keys.size();
        d.cellReads = dCellR; d.cellUnique = dCellU; d.cellMulti = dCellM; d.cellMism = dCellMis; d.cellBases = dCellB;
        d.nCells = in.nCells; d.nCellsQc = in.nCellsQc; d.nWl = in.nWl; d.wl = dWl;
        { unsigned long long z[PC_N] = {}; EP_CK(cudaMemcpyToSymbol(g_pc, z, sizeof z)); }
        if (n) k_pass_eval<<<G, T>>>(d, dKeep);
        EP_CK(cudaGetLastError());
        EP_CK(cudaMemcpyFromSymbol(pc, g_pc, sizeof pc));
        if (n) {
            thrust::exclusive_scan(thrust::device_ptr<uint32_t>(dKeep), thrust::device_ptr<uint32_t>(dKeep + n), thrust::device_ptr<uint32_t>(dPos));
            uint32_t lastP = 0, lastK = 0;
            EP_CK(cudaMemcpy(&lastP, dPos + n - 1, 4, cudaMemcpyDeviceToHost)); EP_CK(cudaMemcpy(&lastK, dKeep + n - 1, 4, cudaMemcpyDeviceToHost));
            nKeep = (u64) lastP + lastK;
        }
        out.msEval = (now() - te) * 1e3;
    }
    {
        const double ts = now();
        const u64 nPend = pc[PC_PEND], nRhost = pc[PC_RHOST];
        EP_CK(B.alloc(dOut, nKeep * 16)); EP_CK(B.alloc(dClip, nKeep * 4)); EP_CK(B.alloc(dLowq, nKeep * 16));
        EP_CK(B.alloc(dPend, nPend * 8)); EP_CK(B.alloc(dRhost, nRhost * 8));
        { unsigned long long z[2] = {0, 0}; EP_CK(cudaMemcpyToSymbol(g_app, z, sizeof z)); }
        if (n) k_pass_scatter<<<G, T>>>(d, dKeep, dPos, dOut, dClip, dLowq, dPend, nPend, dRhost, nRhost);
        EP_CK(cudaGetLastError());
        EP_CK(cudaDeviceSynchronize());
        out.msScatter = (now() - ts) * 1e3;
        const double td = now();
        if (in.keepPinned && singlet_tail::cut("dlpar")) {   // E16 "dlpar" (level 5): the kept reads' arrays first touched by several threads
            // (their capacity, reserved and zero-filled in parallel; resize then fills pages that exist instead of faulting them in one by one)
            auto pre = [](auto& v, u64 n) { v.reserve(n); e16Touch((uint8_t*) v.data(), n * sizeof(v[0]), true); };
            pre(out.reads, nKeep * 16); pre(out.clip, nKeep); pre(out.lowq, nKeep * 16);
            e16dl = now() - td;
        }
        out.reads.resize(nKeep * 16); out.clip.resize(nKeep); out.lowq.resize(nKeep * 16);
        if (nKeep) {
            EP_CK(d2h(out.reads.data(), dOut, nKeep * 16, in.keepPinned));
            EP_CK(d2h(out.clip.data(), dClip, nKeep * 4));
            EP_CK(d2h(out.lowq.data(), dLowq, nKeep * 16, in.keepPinned));
        }
        out.cellReads.resize(in.nCells); out.cellUnique.resize(in.nCellsQc); out.cellMulti.resize(in.nCellsQc);
        out.cellMism.resize(in.nCellsQc); out.cellBases.resize(in.nCellsQc); out.wlCounts.resize(in.nWl);
        if (in.nCells) EP_CK(cudaMemcpy(out.cellReads.data(), dCellR, (u64) in.nCells * 4, cudaMemcpyDeviceToHost));
        if (in.nCellsQc) {
            EP_CK(cudaMemcpy(out.cellUnique.data(), dCellU, (u64) in.nCellsQc * 4, cudaMemcpyDeviceToHost));
            EP_CK(cudaMemcpy(out.cellMulti.data(), dCellM, (u64) in.nCellsQc * 4, cudaMemcpyDeviceToHost));
            EP_CK(cudaMemcpy(out.cellMism.data(), dCellMis, (u64) in.nCellsQc * 8, cudaMemcpyDeviceToHost));
            EP_CK(cudaMemcpy(out.cellBases.data(), dCellB, (u64) in.nCellsQc * 8, cudaMemcpyDeviceToHost));
        }
        if (in.nWl) EP_CK(cudaMemcpy(out.wlCounts.data(), dWl, (u64) in.nWl * 4, cudaMemcpyDeviceToHost));
        std::vector<uint32_t> pend(nPend * 2);
        if (nPend) EP_CK(cudaMemcpy(pend.data(), dPend, nPend * 8, cudaMemcpyDeviceToHost));
        out.wlPending.resize(nPend);
        for (u64 k = 0; k < nPend; k++) out.wlPending[k] = {pend[2 * k], pend[2 * k + 1]};
        out.rrnaHost.resize(nRhost);
        if (nRhost) EP_CK(cudaMemcpy(out.rrnaHost.data(), dRhost, nRhost * 8, cudaMemcpyDeviceToHost));
        out.msDownload = (now() - td) * 1e3;
    }
#undef EP_CK
    { const double tq = now(); if (pinned && !in.keepPinned) cudaHostUnregister(R.base); e16unreg = now() - tq; }
    out.totalAln = pc[PC_ALN]; out.secondary = pc[PC_SEC]; out.rrnaSampled = pc[PC_RSAMP]; out.rrnaHook = pc[PC_RHOOK]; out.chrm = pc[PC_CHRM];
    out.noBarcode = pc[PC_NOBC]; out.barcoded = pc[PC_BARCODED]; out.multimapper = pc[PC_MULTI]; out.lowMapq = pc[PC_LOWMAPQ];
    out.nDone = pc[PC_DONE]; out.nUnwritten = pc[PC_UNWRITTEN]; out.nBigClip = pc[PC_BIGCLIP]; out.nNoHook = pc[PC_NOHOOK]; out.nNoKey = pc[PC_NOKEY];
    out.nNotInSet = pc[PC_NOTINSET];
    out.lowQualSum = pc[PC_LOWQ]; out.qualNeqNSum = pc[PC_NEQN]; out.maskAll = pc[PC_NOMASK] == 0;
    for (int k = 0; k < 8; k++) out.cbStatus[k] = pc[PC_CB0 + k];
    out.msTotal = (now() - t0) * 1e3;
    char b[320];
    snprintf(b, sizeof b, "upload %.0f ms (%s), distinct CRs %.0f ms (%llu), CR table %.0f ms, eval %.0f ms, scatter %.0f ms, download %.0f ms, total %.0f ms",
             out.msUpload, pinned ? "pinned" : "pageable", out.msKeys, (unsigned long long) out.nKeys, out.msTable, out.msEval, out.msScatter,
             out.msDownload, out.msTotal);
    out.log += b;
    // the context goes, as pileupRun's does at its end: pileupRun then initialises the device afresh (cudaInitDevice on a live context fails)
    { const double tq = now(); B.release(); if (!in.keepPinned) cudaDeviceReset(); e16reset = now() - tq; }   // E16 keepPinned: the context stays
    snprintf(b, sizeof b, "; e16: context %.3f s, register %.3f s, unregister %.3f s, free + reset %.3f s%s", e16ctx, e16reg, e16unreg, e16reset,
             in.keepPinned ? (pinned ? " (region and kept reads stay page-locked)" : " (region NOT page-locked)") : "");
    out.log += b;
    if (e16pre >= 0 || e16dl >= 0) {
        snprintf(b, sizeof b, ", region populated in parallel %.3f s, kept reads' arrays touched in parallel %.3f s", e16pre, e16dl);
        out.log += b;
    }
    return true;
}

} // namespace singlet_engine_gpu
