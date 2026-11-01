// E6: hops 6 and 7 of the end-to-end path on the device (API and contract: records_gpu.h; logic: records.cuh).
// Not linked into the singlet binary (nothing in the default path changes); bench/fable/records/records_test.cu is the vehicle.
#include "records_gpu.h"
#include <cuda_runtime.h>
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
                       const Cnt4* off, SigRec* sigs, AlnRec* alns, uint32_t* ops, uint8_t* seq2, uint8_t* bytes, u64* bytesOff) {
    const u64 r = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (r >= n) return;
    const Cnt4 o0 = off[r], o1 = off[r + 1];
    const u64 L = seqOff[r + 1] - seqOff[r];
    e6::recordFill(P, blob + blobOff[r], seq + seqOff[r], L, chrStart, (uint32_t)(o1.a - o0.a), (uint32_t)o0.a, (uint32_t)o0.o, o0.s, sigs + r,
                   alns + o0.a, ops, seq2 + o0.s, bytes ? bytes + o0.b : nullptr);
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
                  const uint64_t* dChrStart, bool wantBytes, DeviceRecords& out, std::string& err) {
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
    ev.start();
    k_fill<<<G, T>>>(P, dBlob, dBlobOff, dSeq, dSeqOff, nRec, dChrStart, dCnt, out.sigs, out.alns, out.ops, out.seq, out.bytes, out.bytesOff);
    RG_CK(cudaGetLastError());
    out.msFill = ev.stop();
    RG_CK(cudaDeviceSynchronize());
    cudaFree(dCnt);
    return true;
}

void release(DeviceRecords& r) {
    cudaFree(r.sigs); cudaFree(r.alns); cudaFree(r.ops); cudaFree(r.seq); cudaFree(r.bytes); cudaFree(r.bytesOff);
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
