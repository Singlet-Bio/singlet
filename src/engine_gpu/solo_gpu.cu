// E9: STARsolo's Gene outputs on the GPU (or the host, same source). See solo_gpu.h and docs/gpu-engine/fable/E9-solo.md.
#include "solo_gpu.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <parallel/algorithm>
#include <stdexcept>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>
#include "singlet/gpu_ledger.h"   // last: it wraps cudaMemcpy / cudaMemcpyAsync (SINGLET_ENGINE_GPU_TRACE=1)

namespace e9 {
namespace {

// a device allocation that failed for lack of memory (soloRunKept then runs the Solo stage on the host instead)
struct DevOOM : std::runtime_error { using std::runtime_error::runtime_error; };
thread_local bool g_lastOOM = false;

// the device memory soloRun uses beyond what it found on entry: the lowest free memory seen at its allocations, per stage
struct DevUse {
    bool on = false; size_t free0 = 0, minFree = 0;
    std::vector<std::pair<std::string, size_t>> stage;
    void begin() { size_t t = 0; on = cudaMemGetInfo(&free0, &t) == cudaSuccess; minFree = free0; stage.clear(); }
    void note() { size_t f = 0, t = 0; if (on && cudaMemGetInfo(&f, &t) == cudaSuccess && f < minFree) minFree = f; }
    void lap(const char* name) {
        if (!on) return;
        note();
        stage.push_back({name, free0 > minFree ? free0 - minFree : 0});
        size_t f = 0, t = 0; minFree = cudaMemGetInfo(&f, &t) == cudaSuccess ? f : minFree;
    }
    size_t peak() const { size_t m = 0; for (const auto& s : stage) m = std::max(m, s.second); return m; }
};
DevUse g_use;
std::string devUseLine(const char* head) {
    std::string u = std::string("soloRun(device): ") + head;
    char t[64];
    for (const auto& st : g_use.stage) { snprintf(t, sizeof t, " %s %.2f", st.first.c_str(), st.second / 1073741824.0); u += t; }
    return u;
}

inline void ck(cudaError_t e, const char* what, int line) {
    if (e == cudaSuccess) return;
    std::string m = std::string("CUDA ") + what + " (solo_gpu.cu:" + std::to_string(line) + "): " + cudaGetErrorString(e);
    if (e == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) == cudaSuccess) m += " (device free " + std::to_string(f >> 20) + " MiB of " + std::to_string(t >> 20) + " MiB)";
        throw DevOOM(m);
    }
    throw std::runtime_error(m);
}
#define CK(x) ck((x), #x, __LINE__)

// a device or host array (host mode: inputs alias the caller's arrays)
template <class T> struct Buf {
    T* p = nullptr; u64 n = 0; bool dev = false, own = false;
    Buf() {}
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    ~Buf() { release(); }
    void release() {
        if (p && own) { if (dev) cudaFree(p); else free(p); }
        p = nullptr; n = 0; own = false;
    }
    void alloc(bool d, u64 cnt) {
        release(); dev = d; n = cnt; own = true;
        const u64 b = std::max<u64>(1, cnt) * sizeof(T);
        if (dev) { CK(cudaMalloc(&p, b)); g_use.note(); }
        else { p = (T*)malloc(b); if (!p) throw std::runtime_error("host allocation failed"); }
    }
    void zero() { if (!n) return; if (dev) CK(cudaMemset(p, 0, n * sizeof(T))); else memset(p, 0, n * sizeof(T)); }
    // a device array the caller owns, used in place (E11: the arrays mapDistinct kept)
    void alias(const T* q, u64 cnt) { release(); dev = true; p = const_cast<T*>(q); n = cnt; own = false; }
    // the caller's array: in place when it is already on the device (onDev), else as from()
    void in(bool d, bool onDev, const T* h, u64 cnt) { if (onDev) alias(h, cnt); else from(d, h, cnt); }
    void from(bool d, const T* h, u64 cnt) {
        if (!d) { release(); dev = false; p = const_cast<T*>(h); n = cnt; own = false; return; }
        alloc(d, cnt);
        if (cnt) CK(cudaMemcpy(p, h, cnt * sizeof(T), cudaMemcpyHostToDevice));
    }
    void to(T* h, u64 cnt) const {
        if (!cnt) return;
        if (dev) CK(cudaMemcpy(h, p, cnt * sizeof(T), cudaMemcpyDeviceToHost)); else memcpy(h, p, cnt * sizeof(T));
    }
    T get(u64 i) const { T v; if (dev) CK(cudaMemcpy(&v, p + i, sizeof(T), cudaMemcpyDeviceToHost)); else v = p[i]; return v; }
};

template <class F> __global__ void kEach(u64 n, F f) {
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) f(i);
}
// one rule per item, on the device (a grid-stride kernel) or the host (OpenMP)
template <class F> void each(bool dev, u64 n, F f) {
    if (n == 0) return;
    if (dev) {
        const unsigned b = 256;
        u64 g = (n + b - 1) / b;
        if (g > (1u << 20)) g = 1u << 20;
        kEach<<<(unsigned)g, b>>>(n, f);
        CK(cudaGetLastError());
    } else {
#pragma omp parallel for schedule(dynamic, 4096)
        for (long long i = 0; i < (long long)n; i++) f((u64)i);
    }
}

struct Tmp {
    void* p = nullptr; size_t n = 0;
    ~Tmp() { if (p) cudaFree(p); }
    void* need(size_t b) { if (b > n) { if (p) cudaFree(p); p = nullptr; CK(cudaMalloc(&p, std::max<size_t>(b, 1))); g_use.note(); n = b; } return p; }
};

// CUB takes item counts as int: a count past INT_MAX fails here instead of wrapping
inline void chkInt(u64 n, const char* what) {
    if (n >= 0x7fffffffULL) throw std::runtime_error(std::string("the device path takes item counts below 2^31 (") + what + ": " + std::to_string(n) + ")");
}

void sortKeys(bool dev, Buf<u64>& k, u64 n, int endBit, Tmp& tmp) {
    if (n < 2) return;
    if (dev) chkInt(n, "sort");
    if (!dev) { __gnu_parallel::sort(k.p, k.p + n); return; }
    Buf<u64> alt; alt.alloc(true, k.n);
    cub::DoubleBuffer<u64> db(k.p, alt.p);
    size_t b = 0;
    CK(cub::DeviceRadixSort::SortKeys(nullptr, b, db, (int)n, 0, endBit));
    CK(cub::DeviceRadixSort::SortKeys(tmp.need(b), b, db, (int)n, 0, endBit));
    if (db.Current() != k.p) std::swap(k.p, alt.p);
}
void sortPairs(bool dev, Buf<u64>& k, Buf<u32>& v, u64 n, int endBit, Tmp& tmp) {
    if (n < 2) return;
    if (dev) chkInt(n, "sort by key");
    if (!dev) {
        std::vector<std::pair<u64, u32>> kv(n);
        for (u64 i = 0; i < n; i++) kv[i] = {k.p[i], v.p[i]};
        __gnu_parallel::sort(kv.begin(), kv.end());
        for (u64 i = 0; i < n; i++) { k.p[i] = kv[i].first; v.p[i] = kv[i].second; }
        return;
    }
    Buf<u64> ka; ka.alloc(true, k.n);
    Buf<u32> va; va.alloc(true, v.n);
    cub::DoubleBuffer<u64> dk(k.p, ka.p);
    cub::DoubleBuffer<u32> dv(v.p, va.p);
    size_t b = 0;
    CK(cub::DeviceRadixSort::SortPairs(nullptr, b, dk, dv, (int)n, 0, endBit));
    CK(cub::DeviceRadixSort::SortPairs(tmp.need(b), b, dk, dv, (int)n, 0, endBit));
    if (dk.Current() != k.p) std::swap(k.p, ka.p);
    if (dv.Current() != v.p) std::swap(v.p, va.p);
}
u64 uniqueKeys(bool dev, const Buf<u64>& in, u64 n, Buf<u64>& out, Tmp& tmp) {
    out.alloc(dev, n);
    if (n == 0) return 0;
    if (dev) chkInt(n, "unique");
    if (!dev) return (u64)(std::unique_copy(in.p, in.p + n, out.p) - out.p);
    Buf<u64> cnt; cnt.alloc(true, 1);
    size_t b = 0;
    CK(cub::DeviceSelect::Unique(nullptr, b, in.p, out.p, cnt.p, (int)n));
    CK(cub::DeviceSelect::Unique(tmp.need(b), b, in.p, out.p, cnt.p, (int)n));
    return cnt.get(0);
}
// exclusive prefix sum of n values (out has n + 1 entries; out[n] = total)
u64 exclusiveScan(bool dev, const Buf<u32>& in, u64 n, Buf<u32>& out, Tmp& tmp) {
    out.alloc(dev, n + 1);
    if (dev) chkInt(n + 1, "scan");
    if (!dev) { u64 s = 0; for (u64 i = 0; i < n; i++) { out.p[i] = (u32)s; s += in.p[i]; } out.p[n] = (u32)s; return s; }
    // in has n + 1 entries with in[n] = 0 so out[n] is the total
    size_t b = 0;
    CK(cub::DeviceScan::ExclusiveSum(nullptr, b, in.p, out.p, (int)(n + 1)));
    CK(cub::DeviceScan::ExclusiveSum(tmp.need(b), b, in.p, out.p, (int)(n + 1)));
    return out.get(n);
}
void inclusiveScan(bool dev, const Buf<u32>& in, u64 n, Buf<u32>& out, Tmp& tmp) {
    out.alloc(dev, n);
    if (n == 0) return;
    if (dev) chkInt(n, "scan");
    if (!dev) { u64 s = 0; for (u64 i = 0; i < n; i++) { s += in.p[i]; out.p[i] = (u32)s; } return; }
    size_t b = 0;
    CK(cub::DeviceScan::InclusiveSum(nullptr, b, in.p, out.p, (int)n));
    CK(cub::DeviceScan::InclusiveSum(tmp.need(b), b, in.p, out.p, (int)n));
}

template <class T> E9HD inline u64 lowerBound(const T* a, u64 n, T x) {
    u64 lo = 0, hi = n;
    while (lo < hi) { const u64 m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; }
    return lo;
}

// v added to *p, the previous value returned (the same on both sides)
E9HD inline u64 fetchAdd(u64* p, u64 v) {
#ifdef __CUDA_ARCH__
    return (u64)atomicAdd((unsigned long long*)p, (unsigned long long)v);
#else
    return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
#endif
}

}  // namespace

// stage (c) of one read: Solo's record() and inputRecords() (SoloReadFeature_record.cpp, _inputRecords.cpp). cbm: cbMatch after the UMI check; cand,
// nCand: the whitelist matches of its CR; cb: the CB it is given (its single match, or the posterior's choice, -1 none). Adds into the counters and
// returns the read's (CB, gene, UMI) key, or ~0 when the read is not counted. One rule for the one-shot and the two-pass path.
struct CountSinks { u64* fs; u64* nb; u32* crc; u32* nru; u32* fr; u64* c; };
E9HD inline u64 countRead(const RecAnn& a, i32 cbm, const u32* cand, u32 nCand, i32 cb, u32 umi, const CountSinks& s) {
    u32 flag = recordFlag(a, cbm);
    if (cbm < 0) {   // record(): no CB match, counted as CBnotInPasslist, nothing written
        for (u32 b = 0; b < F_N; b++) if ((flag >> b) & 1) incAgg(s.nb, b);
        return ~0ULL;
    }
    bool feat = false;
    if (a.nTr == 0) incAgg(s.fs, S_noUnmapped);
    else if (a.ng == 0) incAgg(s.fs, S_noNoFeature);
    else if (a.ng > 1) { incAgg(s.fs, S_MultiFeature); if (a.nTr > 1) incAgg(s.fs, S_subMultiFeatureMultiGenomic); }
    else feat = true;
    if (feat) for (u32 k = 0; k < nCand; k++) addU32(&s.crc[cand[k]], 1);   // record(): cbReadCount for every match
    // inputRecords(); a read with several matches: its own posterior (E15)
    const bool accepted = cb >= 0;
    u64 key = ~0ULL;
    if (feat) {
        if (cbm == 0) incAgg(s.fs, S_yessubWLmatchExact);
        else if (!accepted) incAgg(s.fs, S_noTooManyWLmatches);
    }
    if (feat && accepted) {
        addU32(&s.nru[cb], 1);
        key = tupleKey((u64)cb, a.gene, umi);
        flag |= 1u << F_countedU;
        incAgg(s.c, 1);
    }
    flag |= 1u << F_cbMatch;
    if (cbm == 0) flag |= 1u << F_cbPerfect;
    else if (cbm == 1) flag |= 1u << F_cbMMunique;
    else if (accepted) flag |= 1u << F_cbMMmultiple;
    if (accepted) {
        for (u32 b = 0; b < F_N; b++) if ((flag >> b) & 1) addU32(&s.fr[(u64)cb * F_N + b], 1);
    } else {
        for (u32 b = 0; b < F_N; b++) if ((flag >> b) & 1) incAgg(s.nb, b);
    }
    return key;
}

bool soloRun(bool dev, const SoloParams& S0, const SoloModel& M, const SoloRecs& R, const SoloReads& Rd, SoloResult& out, bool keep, std::string& log) {
    ::singlet::gpu_ledger::Phase ledger_phase("solo");
    g_lastOOM = false;
    try {
        using clk = std::chrono::steady_clock;
        auto t0 = clk::now();
        if (dev) g_use.begin();
        auto lap = [&](const char* name) {
            if (dev) { CK(cudaDeviceSynchronize()); g_use.lap(name); }
            const auto t = clk::now();
            out.ms.push_back({name, std::chrono::duration<double, std::milli>(t - t0).count()});
            t0 = t;
        };
        SoloParams S = S0;
        const u32 W = (u32)M.wl.size();
        S.wlN = W;
        const u64 N = Rd.n, nRec = R.nRec;
        // CUB's scans, sorts and selects take their item counts as int (the scans N + 1): past 2^31 - 2 reads the one-shot path would count wrongly
        // (soloRunTwoPass takes any number of reads); the host path's u32 scans and indices end at 2^32 - 1
        if (dev && N + 1 >= 0x7fffffffULL) throw std::runtime_error("the one-shot device path takes at most 2^31 - 2 reads");
        if (!dev && N >= 0xffffffffULL) throw std::runtime_error("the host path takes at most 2^32 - 2 reads");
        if (W >= (1u << (64 - GENE_BITS - UMI_BITS - 1)) || S.nGe >= (1u << GENE_BITS) || 2 * S.umiL > UMI_BITS)
            throw std::runtime_error("whitelist, genes or UMI length do not fit the (CB, gene, UMI) key");
        if ((R.dev || Rd.dev) && !dev) throw std::runtime_error("device-resident inputs need soloRun(device)");
        for (u64 k = 0; k < Rd.nHost; k++)
            if (Rd.hostIdx[k] >= N || (k > 0 && Rd.hostIdx[k] <= Rd.hostIdx[k - 1])) throw std::runtime_error("host annotations: read index out of order or range");
        Tmp tmp;

        // ---- upload (model, records, reads)
        Buf<u64> dWl, dTrS, dTrE, dTrEmax; Buf<u32> dTrExI, dTrGene, dExSE; Buf<uint16_t> dTrExN; Buf<uint8_t> dTrStr;
        dWl.from(dev, M.wl.data(), W);
        dTrS.from(dev, M.trS.data(), M.trS.size()); dTrE.from(dev, M.trE.data(), M.trE.size()); dTrEmax.from(dev, M.trEmax.data(), M.trEmax.size());
        dTrExI.from(dev, M.trExI.data(), M.trExI.size()); dTrGene.from(dev, M.trGene.data(), M.trGene.size()); dExSE.from(dev, M.exSE.data(), M.exSE.size());
        dTrExN.from(dev, M.trExN.data(), M.trExN.size()); dTrStr.from(dev, M.trStr.data(), M.trStr.size());
        const TrModel TM{dTrS.p, dTrE.p, dTrEmax.p, dTrExI.p, dTrExN.p, dTrStr.p, dTrGene.p, dExSE.p, (u32)M.trS.size()};
        Buf<uint8_t> dBlob; Buf<u64> dBlobOff, dSeqOff; Buf<RecAnn> recAnn;
        if (R.recAnn) recAnn.in(dev, R.dev, R.recAnn, nRec);   // E11: per-record annotation already made (soloKeepRecords)
        else { dBlob.in(dev, R.dev, R.blob, R.blobBytes); dBlobOff.in(dev, R.dev, R.blobOff, nRec + 1); dSeqOff.in(dev, R.dev, R.seqOff, nRec + 1); }
        Buf<u64> dCr3, dUr3; Buf<u32> dRec; Buf<RecAnn> dHost;
        dCr3.in(dev, Rd.dev, Rd.cr3, N); dUr3.in(dev, Rd.dev, Rd.ur3, N); dRec.in(dev, Rd.dev, Rd.rec, N);
        if (Rd.hostAnn) dHost.from(dev, Rd.hostAnn, N);
        Buf<u32> dHostIdx; Buf<RecAnn> dHostS;   // sparse host annotations (host memory: uploaded)
        if (Rd.nHost) { dHostIdx.from(dev, Rd.hostIdx, Rd.nHost); dHostS.from(dev, Rd.hostAnnS, Rd.nHost); }
        Buf<u64> cnt; cnt.alloc(dev, 8); cnt.zero();   // 0: records with an exon overflow, 1: counted reads, 2: reads without a record, 3: bad host index, 4: multi-match reads the posterior gave a CB (E15)
        lap("upload");

        // ---- stage (b): per record its Gene annotation, per read its record's (or the host's)
        Buf<RecAnn> readAnn; readAnn.alloc(dev, N);
        {
            if (!R.recAnn) {
                recAnn.alloc(dev, nRec);
                const uint8_t* blob = dBlob.p; const u64* bo = dBlobOff.p; const u64* so = dSeqOff.p; RecAnn* ra = recAnn.p;
                const e6::RecParams RP = R.RP; const TrModel tm = TM; const SoloParams s = S; u64* c = cnt.p;
                each(dev, nRec, [=] __host__ __device__(u64 r) {
                    const RecAnn a = recAnnotate(RP, blob + bo[r], so[r + 1] - so[r], tm, s);
                    ra[r] = a;
                    if (a.bits & RA_OVERFLOW) addU64(c, 1);
                });
            }
            const u32* rec = dRec.p; const RecAnn* host = dHost.p; RecAnn* ann = readAnn.p; const RecAnn* rac = recAnn.p; u64* c = cnt.p;
            each(dev, N, [=] __host__ __device__(u64 i) {
                const u32 r = rec[i];
                RecAnn a{0xFFFFFFFFu, 0, 0, 0};
                if (r != 0xFFFFFFFFu) a = rac[r]; else { addU64(c + 2, 1); if (host) a = host[i]; }
                ann[i] = a;
            });
            if (Rd.nHost) {   // E11: the reads the host mapped, sparse
                const u32* hi = dHostIdx.p; const RecAnn* ha = dHostS.p;
                each(dev, Rd.nHost, [=] __host__ __device__(u64 k) {
                    const u32 i = hi[k];
                    if (rec[i] != 0xFFFFFFFFu) addU64(c + 3, 1);
                    ann[i] = ha[k];
                });
            }
        }
        recAnn.release(); dHostIdx.release(); dHostS.release();
        if (Rd.nHost) {
            const u64 nNone = cnt.get(2), nBad = cnt.get(3);
            if (nNone != Rd.nHost || nBad) throw std::runtime_error("host annotations: " + std::to_string(Rd.nHost) + " given, " + std::to_string(nNone)
                                                                    + " reads without a record, " + std::to_string(nBad) + " given for reads with a record");
        }
        lap("gene");

        // ---- stage (a): per distinct CR the whitelist matches; per read the UMI check, Barcodes.stats, the exact histogram; the posterior
        Buf<u64> crk; crk.alloc(dev, N);
        {
            const u64* cr3 = dCr3.p; u64* k = crk.p; const u64 mk = e6::mask3(S.cbL);
            each(dev, N, [=] __host__ __device__(u64 i) { k[i] = cr3[i] & mk; });
        }
        Buf<u64> dcr; u64 D = 0;
        {
            Buf<u64> crs; crs.alloc(dev, N);
            if (dev) CK(cudaMemcpy(crs.p, crk.p, N * 8, cudaMemcpyDeviceToDevice)); else memcpy(crs.p, crk.p, N * 8);
            sortKeys(dev, crs, N, (int)(3 * S.cbL), tmp);
            D = uniqueKeys(dev, crs, N, dcr, tmp);
        }
        out.nDistinctCR = D;
        Buf<i32> dcbm; dcbm.alloc(dev, D);
        Buf<u32> nC; nC.alloc(dev, D + 1); nC.zero();
        {
            const u64* cr = dcr.p; i32* m = dcbm.p; u32* nc = nC.p; const u64* wl = dWl.p; const u32 cbL = S.cbL;
            each(dev, D, [=] __host__ __device__(u64 d) { u32 n; m[d] = matchCBtoWL(cr[d], cbL, wl, W, nullptr, n); nc[d] = n; });
        }
        Buf<u32> cOff;
        const u64 nCand = exclusiveScan(dev, nC, D, cOff, tmp);
        out.nCand = nCand;
        Buf<u32> cand; cand.alloc(dev, nCand);
        {
            const u64* cr = dcr.p; const u32* co = cOff.p; u32* cd = cand.p; const u64* wl = dWl.p; const u32 cbL = S.cbL;
            each(dev, D, [=] __host__ __device__(u64 d) { u32 n; matchCBtoWL(cr[d], cbL, wl, W, cd + co[d], n); });
        }
        Buf<u32> crIdx, umi, exact; Buf<int8_t> cbmR; Buf<u64> bst;
        crIdx.alloc(dev, N); umi.alloc(dev, N); cbmR.alloc(dev, N); exact.alloc(dev, W); exact.zero(); bst.alloc(dev, B_N); bst.zero();
        {
            const u64* k = crk.p; const u64* cr = dcr.p; const u64 Dn = D; const u64* ur3 = dUr3.p; const i32* m = dcbm.p; const u32* co = cOff.p;
            const u32* cd = cand.p; u32* ci = crIdx.p; u32* um = umi.p; int8_t* cb = cbmR.p; u32* ex = exact.p; u64* bs = bst.p; const SoloParams s = S;
            each(dev, N, [=] __host__ __device__(u64 i) {
                const u64 d = lowerBound(cr, Dn, k[i]);
                ci[i] = (u32)d;
                u64 ub = 0;
                const i32 uc = checkUMI(ur3[i], s, ub);
                const i32 cbm = uc != 0 ? uc : m[d];   // getCBandUMI: a bad UMI replaces cbMatch (SoloReadBarcode_getCBandUMI.cpp:290-302)
                cb[i] = (int8_t)cbm;
                um[i] = (u32)ub;
                incAgg(bs, (u32)barcodeStat(cbm));
                if (cbm == 0) addU32(&ex[cd[co[d]]], 1);   // cbReadCountExact (addStats case 0)
            });
        }
        crk.release();
        Buf<i32> chosen; chosen.alloc(dev, D);   // per distinct CR with one match: that match (several matches: per read below)
        {
            const i32* m = dcbm.p; const u32* co = cOff.p; const u32* cd = cand.p; i32* ch = chosen.p;
            each(dev, D, [=] __host__ __device__(u64 d) { const i32 mm = m[d]; ch[d] = (mm == 0 || mm == 1) ? (i32)cd[co[d]] : -1; });
        }
        lap("cb");

        // ---- E15: the posterior per read for the reads with several matches (cbMatch > 1 after the UMI check), in read order; each candidate
        // takes the read's quality at its mismatch (SoloReads::cbQual) or, without per-read qualities, S.pq
        Buf<u64> mmIdx; Buf<i32> mmCb; u64 nMM = 0;   // read indices are 64-bit
        {
            Buf<u32> fl; fl.alloc(dev, N + 1);
            const int8_t* cbmp = cbmR.p; u32* f = fl.p; const u64 n = N;
            each(dev, N + 1, [=] __host__ __device__(u64 i) { f[i] = (i < n && cbmp[i] > 1) ? 1u : 0u; });
            Buf<u32> fo;
            nMM = exclusiveScan(dev, fl, N, fo, tmp);
            mmIdx.alloc(dev, nMM);
            const u32* fp = fl.p; const u32* op = fo.p; u64* mi = mmIdx.p;
            each(dev, N, [=] __host__ __device__(u64 i) { if (fp[i]) mi[op[i]] = i; });
        }
        out.nMulti = nMM;
        out.perReadQual = (bool)Rd.cbQual;
        std::vector<uint8_t> hQual;   // host mode: mmQ aliases it
        Buf<uint8_t> mmQ; Buf<double> dPq;
        if (Rd.cbQual && nMM > 0) {
            if (M.pqTab.size() != 256) throw std::runtime_error("per-read CB qualities without the quality table");
            std::vector<u64> hIdx(nMM);
            mmIdx.to(hIdx.data(), nMM);
            Buf<u64> mmCr; mmCr.alloc(dev, nMM);   // their CRs, for the caller's check that it read their lines
            {
                const u64* mi = mmIdx.p; const u32* ci = crIdx.p; const u64* cr = dcr.p; u64* o = mmCr.p;
                each(dev, nMM, [=] __host__ __device__(u64 k) { o[k] = cr[ci[mi[k]]]; });
            }
            std::vector<u64> hCr(nMM);
            mmCr.to(hCr.data(), nMM);
            mmCr.release();
            hQual.resize(nMM * S.cbL);
            std::string why;
            if (!Rd.cbQual(hIdx.data(), hCr.data(), nMM, hQual.data(), why)) throw std::runtime_error("CB qualities of the multi-match reads: " + why);
            mmQ.from(dev, hQual.data(), hQual.size());
            dPq.from(dev, M.pqTab.data(), 256);
        }
        mmCb.alloc(dev, nMM);
        {
            const u64* mi = mmIdx.p; const u32* ci = crIdx.p; const u64* cr = dcr.p; const u32* co = cOff.p; const u32* nc = nC.p; const u32* cd = cand.p;
            const u32* ex = exact.p; const u64* wl = dWl.p; const uint8_t* q = mmQ.p; const double* pt = dPq.p; i32* mc = mmCb.p; u64* c = cnt.p;
            const SoloParams s = S; const bool perRead = Rd.cbQual && nMM > 0;
            each(dev, nMM, [=] __host__ __device__(u64 k) {
                const u32 d = ci[mi[k]];
                const i64 r = cbPosteriorRead(cr[d], cd + co[d], nc[d], ex, wl, perRead ? q + k * s.cbL : nullptr, pt, s);
                mc[k] = (i32)r;
                if (r >= 0) addU64(c + 4, 1);
            });
        }
        mmQ.release(); dPq.release();
        out.nMultiAccepted = cnt.get(4);
        lap("multi");

        // ---- stage (c): per read Solo's record + inputRecords
        Buf<u64> keys; keys.alloc(dev, N);
        Buf<i32> rowCb; rowCb.alloc(dev, N);
        Buf<u32> flagRows; flagRows.alloc(dev, (u64)W * F_N); flagRows.zero();
        Buf<u64> fst, noCB; fst.alloc(dev, S_N); fst.zero(); noCB.alloc(dev, F_N); noCB.zero();
        Buf<u32> cbRC, nRU; cbRC.alloc(dev, W); cbRC.zero(); nRU.alloc(dev, W); nRU.zero();
        {
            const RecAnn* ann = readAnn.p; const int8_t* cbmp = cbmR.p; const u32* ci = crIdx.p; const u32* um = umi.p; const u32* co = cOff.p;
            const u32* nc = nC.p; const u32* cd = cand.p; const i32* ch = chosen.p; u64* ky = keys.p; i32* rc = rowCb.p;
            const CountSinks sk{fst.p, noCB.p, cbRC.p, nRU.p, flagRows.p, cnt.p};
            const u64* mi = mmIdx.p; const i32* mc = mmCb.p; const u64 nmm = nMM;
            each(dev, N, [=] __host__ __device__(u64 i) {
                const i32 cbm = cbmp[i];
                const u32 d = ci[i];
                const i32 cb = cbm < 0 ? -1 : cbm > 1 ? mc[lowerBound(mi, nmm, i)] : ch[d];
                ky[i] = countRead(ann[i], cbm, cd + co[d], nc[d], cb, um[i], sk);
                rc[i] = (cbm >= 0 && cb >= 0) ? cb : -1;
            });
        }
        lap("reads");
        if (keep) {
            out.readAnn.resize(N); readAnn.to(out.readAnn.data(), N);
            out.readCbm.resize(N); cbmR.to(out.readCbm.data(), N);
            Buf<i32> c0; c0.alloc(dev, N);
            {
                const int8_t* cbmp = cbmR.p; const u32* ci = crIdx.p; const u32* co = cOff.p; const u32* nc = nC.p; const u32* cd = cand.p; i32* o = c0.p;
                each(dev, N, [=] __host__ __device__(u64 i) { const u32 d = ci[i]; o[i] = (cbmp[i] >= 0 && nc[d] > 0) ? (i32)cd[co[d]] : -1; });
            }
            out.readCb0.resize(N); c0.to(out.readCb0.data(), N);
            {   // E15: the CB each read gets (whether or not it is counted): its single match, or its posterior's choice
                const int8_t* cbmp = cbmR.p; const u32* ci = crIdx.p; const i32* ch = chosen.p; const u64* mi = mmIdx.p; const i32* mc = mmCb.p;
                const u64 nmm = nMM; i32* o = c0.p;
                each(dev, N, [=] __host__ __device__(u64 i) {
                    const i32 cbm = cbmp[i];
                    o[i] = cbm < 0 ? -1 : cbm > 1 ? mc[lowerBound(mi, nmm, i)] : ch[ci[i]];
                });
            }
            out.readCb.resize(N); c0.to(out.readCb.data(), N);
        }
        readAnn.release(); cbmR.release(); crIdx.release(); umi.release(); dCr3.release(); dUr3.release(); dRec.release(); dHost.release();
        mmIdx.release(); mmCb.release();
        dBlob.release(); dBlobOff.release(); dSeqOff.release();
        t0 = clk::now();   // per-read validation downloads are not a stage

        // ---- the (CB, gene, UMI) of the counted reads: sort, unique; the second order (UMI halves swapped)
        const u64 nK = cnt.get(1);
        out.nKeys = nK;
        sortKeys(dev, keys, N, 64, tmp);   // the reads that are not counted (~0) sort last
        Buf<u64> tup;
        const u64 nT = uniqueKeys(dev, keys, nK, tup, tmp);
        out.nTuples = nT;
        keys.release();
        Buf<u64> k2; k2.alloc(dev, nT);
        Buf<u32> v2; v2.alloc(dev, nT);
        {
            const u64* t = tup.p; u64* a = k2.p; u32* b = v2.p; const u32 umiL = S.umiL, ml = S.umiMaskLow;
            each(dev, nT, [=] __host__ __device__(u64 j) { a[j] = swapHalvesKey(t[j], umiL, ml); b[j] = (u32)j; });
        }
        sortPairs(dev, k2, v2, nT, 63, tmp);
        lap("sort");

        // ---- 1MM_All per (CB, gene): Hamming-1 components (collapseUMIwith1MMlowHalf on both halves), then the counts per CB
        Buf<u32> par; par.alloc(dev, nT);
        {
            u32* pr = par.p;
            each(dev, nT, [=] __host__ __device__(u64 j) { pr[j] = (u32)j; });
            const u64* t = tup.p; const u64 ml = S.umiMaskLow; const u64 n = nT;
            each(dev, nT, [=] __host__ __device__(u64 j) {
                for (u64 k = j + 1; k < n; k++) {
                    const u64 x = t[k] ^ t[j];
                    if (x > ml) break;   // another (CB, gene) or another upper half
                    if (oneMM((u32)x)) ufUnion(pr, (u32)j, (u32)k);
                }
            });
            const u64* a = k2.p; const u32* b = v2.p;
            each(dev, nT, [=] __host__ __device__(u64 j) {
                for (u64 k = j + 1; k < n; k++) {
                    const u64 x = a[k] ^ a[j];
                    if (x > ml) break;
                    if (oneMM((u32)x)) ufUnion(pr, b[j], b[k]);
                }
            });
        }
        k2.release(); v2.release();
        Buf<u32> hd; hd.alloc(dev, nT);
        {
            const u64* t = tup.p; u32* h = hd.p;
            each(dev, nT, [=] __host__ __device__(u64 j) { h[j] = (j == 0 || (t[j] >> UMI_BITS) != (t[j - 1] >> UMI_BITS)) ? 1u : 0u; });
        }
        Buf<u32> sid;
        inclusiveScan(dev, hd, nT, sid, tmp);
        const u64 nSeg = nT ? sid.get(nT - 1) : 0;
        out.nSeg = nSeg;
        Buf<u64> segKey; segKey.alloc(dev, nSeg);
        Buf<u32> segCnt; segCnt.alloc(dev, nSeg); segCnt.zero();
        {
            const u64* t = tup.p; const u32* h = hd.p; const u32* si = sid.p; const u32* pr = par.p; u64* sk = segKey.p; u32* sc = segCnt.p;
            each(dev, nT, [=] __host__ __device__(u64 j) {
                const u32 s = si[j] - 1;
                if (h[j]) sk[s] = t[j] >> UMI_BITS;
                if (pr[j] == (u32)j) addU32(&sc[s], 1);
            });
        }
        Buf<u32> nUMI, nGene; nUMI.alloc(dev, W); nUMI.zero(); nGene.alloc(dev, W); nGene.zero();
        {
            const u64* sk = segKey.p; const u32* sc = segCnt.p; u32* nu = nUMI.p; u32* ng = nGene.p;
            each(dev, nSeg, [=] __host__ __device__(u64 s) {
                const u64 cb = sk[s] >> GENE_BITS;
                addU32(&nu[cb], sc[s]);
                addU32(&ng[cb], 1);
            });
        }
        lap("collapse");

        // ---- the CBs with a count or a CellReads row, compacted (a CellReads row always has its cbMatch count; nUMI, nGene and
        // nReadUnique > 0 imply cbReadCount > 0)
        Buf<u32> pres; pres.alloc(dev, (u64)W + 1); pres.zero();
        {
            const u32* crc = cbRC.p; const u32* fr = flagRows.p; u32* pr = pres.p;
            each(dev, W, [=] __host__ __device__(u64 w) { pr[w] = (crc[w] > 0 || fr[w * F_N + F_cbMatch] > 0) ? 1u : 0u; });
        }
        Buf<u32> pOff;
        const u64 nP = exclusiveScan(dev, pres, W, pOff, tmp);
        Buf<u32> pIdx, pVal; pIdx.alloc(dev, nP); pVal.alloc(dev, nP * CBV);
        {
            const u32* pr = pres.p; const u32* po = pOff.p; u32* pi = pIdx.p; u32* pv = pVal.p;
            const u32* crc = cbRC.p; const u32* nu = nUMI.p; const u32* ng = nGene.p; const u32* nru = nRU.p; const u32* fr = flagRows.p;
            each(dev, W, [=] __host__ __device__(u64 w) {
                if (!pr[w]) return;
                const u64 k = po[w];
                pi[k] = (u32)w;
                u32* v = pv + k * CBV;
                v[CV_cbReadCount] = crc[w]; v[CV_nUMI] = nu[w]; v[CV_nGene] = ng[w]; v[CV_nReadUnique] = nru[w];
                for (u32 b = 0; b < F_N; b++) v[CV_flags + b] = fr[w * F_N + b];
            });
        }
        lap("compact");

        // ---- download
        bst.to(out.bstats, B_N); fst.to(out.fstats, S_N); noCB.to(out.noCB, F_N);
        out.cbIdx.resize(nP); pIdx.to(out.cbIdx.data(), nP);
        out.cbVal.resize(nP * CBV); pVal.to(out.cbVal.data(), nP * CBV);
        std::vector<u64> sk(nSeg); segKey.to(sk.data(), nSeg);
        out.segCount.resize(nSeg); segCnt.to(out.segCount.data(), nSeg);
        out.segCb.resize(nSeg); out.segGene.resize(nSeg);
        for (u64 s = 0; s < nSeg; s++) { out.segCb[s] = (u32)(sk[s] >> GENE_BITS); out.segGene[s] = (u32)(sk[s] & ((1u << GENE_BITS) - 1)); }
        out.nRecOverflow = cnt.get(0);
        lap("download");
        if (keep) { out.rowCb.resize(N); rowCb.to(out.rowCb.data(), N); }   // the driver's CellReads order replay (not a stage)

        // ---- Features.stats rows that collapseUMIall adds (SoloFeature_collapseUMIall.cpp:21-25)
        u64 yesUMIs = 0, yesCB = 0, yesWL = 0;
        for (u64 k = 0; k < nP; k++) {
            const u32* v = &out.cbVal[k * CBV];
            yesUMIs += v[CV_nUMI]; yesCB += v[CV_nGene] > 0; yesWL += v[CV_nReadUnique];
        }
        out.fstats[S_yesUMIs] = yesUMIs; out.fstats[S_yesCellBarcodes] = yesCB;
        out.fstats[S_yesWLmatch] = yesWL; out.fstats[S_yessubWLmatch_UniqueFeature] = yesWL;
        char b[640];
        snprintf(b, sizeof b, "soloRun(%s): %llu reads, %llu records (%llu exon overflow), %llu distinct CRs, %llu matches, %llu counted reads, %llu distinct (CB,gene,UMI), %llu (CB,gene); %llu reads with several matches (%llu given a CB by the posterior, %s)\n",
                 dev ? "device" : "host", (unsigned long long)N, (unsigned long long)nRec, (unsigned long long)out.nRecOverflow, (unsigned long long)D,
                 (unsigned long long)nCand, (unsigned long long)nK, (unsigned long long)nT, (unsigned long long)nSeg, (unsigned long long)nMM,
                 (unsigned long long)out.nMultiAccepted, out.perReadQual ? "per-read CB qualities" : "one CB quality factor");
        log += b;
        if (dev && g_use.on) log += "\n" + devUseLine("device memory beyond the inputs, peak per stage (GiB):");
        return true;
    } catch (const DevOOM& e) {
        g_lastOOM = true;
        log += std::string("soloRun failed: ") + e.what() + "\n";
        if (dev && g_use.on) log += devUseLine("device memory beyond the inputs before the failure, peak per stage (GiB):") + "\n";
        if (dev) cudaGetLastError();
        return false;
    } catch (const std::exception& e) {
        log += std::string("soloRun failed: ") + e.what() + "\n";
        if (dev) cudaGetLastError();
        return false;
    }
}
bool soloLastFailedOutOfMemory() { return g_lastOOM; }

// ================================================================================================ two-pass Solo (bounded device memory)
// The same counts as soloRun(device), byte for byte, with the device holding only a read chunk (about 52 bytes per read of it) or a key
// bucket (about 36 bytes per key) besides the per-whitelist-entry arrays (about 90 bytes per entry) and the per-record annotations (8 bytes per
// record, or in host memory). Why the result is the same: every counter is a sum; the exact-match histogram is complete before any posterior is
// taken (the reads with several matches wait for pass 2); the CB is the top field of a key and no UMI collapse crosses a (CB, gene), so sorting,
// uniquing and collapsing the keys by CB range and concatenating the ranges in order gives the global sorted order.
// (types used by the lambdas and by CUB's kernels live outside the unnamed namespace, as do the functions that define the lambdas)
struct DefRead { u64 idx, cr; RecAnn a; u32 umi, pad; };   // a read with several matches, kept for pass 2 (cr: the CR as soloRun has it, masked to the CB)
static_assert(sizeof(DefRead) == 32, "DefRead");
struct IsKey { __host__ __device__ bool operator()(const u64& k) const { return k != ~0ULL; } };

namespace {

// the counted keys of one chunk (or batch), in page-locked host memory when it can be had
struct KeyBlock {
    u64* p = nullptr; u64 n = 0; bool pinned = false;
    KeyBlock() {}
    KeyBlock(const KeyBlock&) = delete;
    KeyBlock& operator=(const KeyBlock&) = delete;
    KeyBlock(KeyBlock&& o) noexcept : p(o.p), n(o.n), pinned(o.pinned) { o.p = nullptr; o.n = 0; }
    ~KeyBlock() { release(); }
    void release() { if (p) { if (pinned) cudaFreeHost(p); else free(p); } p = nullptr; n = 0; pinned = false; }
    void alloc(u64 cnt) {
        release();
        if (cnt == 0) return;
        if (cudaMallocHost((void**)&p, cnt * 8) == cudaSuccess) pinned = true;
        else { cudaGetLastError(); p = (u64*)malloc(cnt * 8); if (!p) throw std::runtime_error("host allocation failed (the counted keys)"); }
        n = cnt;
    }
};

// in[0..n) without the ~0 entries, in order, into out; returns how many
u64 selectCounted(const u64* in, u64 n, u64* out, Tmp& tmp) {
    if (n == 0) return 0;
    chkInt(n, "select");
    Buf<u64> cn; cn.alloc(true, 1);
    size_t b = 0;
    CK(cub::DeviceSelect::If(nullptr, b, in, out, cn.p, (int)n, IsKey{}));
    CK(cub::DeviceSelect::If(tmp.need(b), b, in, out, cn.p, (int)n, IsKey{}));
    return cn.get(0);
}
// the indices i < n with fl[i] != 0, ascending, into out; returns how many
u64 selectFlagged(const uint8_t* fl, u64 n, u32* out, Tmp& tmp) {
    if (n == 0) return 0;
    chkInt(n, "select");
    Buf<u64> cn; cn.alloc(true, 1);
    cub::CountingInputIterator<u32> it(0);
    size_t b = 0;
    CK(cub::DeviceSelect::Flagged(nullptr, b, it, fl, out, cn.p, (int)n));
    CK(cub::DeviceSelect::Flagged(tmp.need(b), b, it, fl, out, cn.p, (int)n));
    return cn.get(0);
}
bool envU64(const char* name, u64& v) {
    const char* e = getenv(name);
    if (!e || !*e) return false;
    char* end = nullptr;
    const unsigned long long x = strtoull(e, &end, 10);
    if (end == e) return false;
    v = (u64)x;
    return true;
}

}  // namespace

void soloTwoPassOnce(const SoloParams& S0, const SoloModel& M, const SoloRecs& R, const SoloReads& Rd, SoloResult& out, std::string& log, const SoloTwoPass& opt,
                double headroom) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    g_use.begin();
    auto lap = [&](const char* name) {
        CK(cudaDeviceSynchronize()); g_use.lap(name);
        const auto t = clk::now();
        out.ms.push_back({name, std::chrono::duration<double, std::milli>(t - t0).count()});
        t0 = t;
    };
    SoloParams S = S0;
    const u32 W = (u32)M.wl.size();
    S.wlN = W;
    const u64 N = Rd.n, nRec = R.nRec;
    if (!R.recAnn) throw std::runtime_error("the two-pass Solo needs the records' annotations (soloKeepRecords)");
    if (Rd.hostAnn) throw std::runtime_error("the two-pass Solo takes the host annotations sparse (hostIdx), not per read");
    if (W >= (1u << (64 - GENE_BITS - UMI_BITS - 1)) || S.nGe >= (1u << GENE_BITS) || 2 * S.umiL > UMI_BITS)
        throw std::runtime_error("whitelist, genes or UMI length do not fit the (CB, gene, UMI) key");
    if (S.cbL == 0 || S.cbL > 21) throw std::runtime_error("the CB length does not fit the 3-bit packed CR");
    for (u64 k = 0; k < Rd.nHost; k++)
        if (Rd.hostIdx[k] >= N || (k > 0 && Rd.hostIdx[k] <= Rd.hostIdx[k - 1])) throw std::runtime_error("host annotations: read index out of order or range");
    Tmp tmp;
    size_t frFree = 0, devTot = 0;
    CK(cudaMemGetInfo(&frFree, &devTot));
    const bool recOnDev = R.dev || Rd.dev || (!opt.recAnnHost && (double)nRec * sizeof(RecAnn) <= 0.25 * (double)frFree);

    // ---- upload (the records' annotations) and the per-whitelist-entry arrays, which stay for both passes
    Buf<u64> dWl;
    dWl.from(true, M.wl.data(), W);
    Buf<RecAnn> dRecAnn;
    if (recOnDev) dRecAnn.in(true, R.dev, R.recAnn, nRec);
    Buf<u64> cnt; cnt.alloc(true, 8); cnt.zero();   // 1: counted reads, 2: reads without a record, 3: bad host index, 4: multi-match reads given a CB, 5: this chunk's multi-match reads, 6: bucket fill
    Buf<u64> bst, fst, noCB; bst.alloc(true, B_N); bst.zero(); fst.alloc(true, S_N); fst.zero(); noCB.alloc(true, F_N); noCB.zero();
    Buf<u32> exact; exact.alloc(true, W); exact.zero();
    Buf<u32> flagRows; flagRows.alloc(true, (u64)W * F_N); flagRows.zero();
    Buf<u32> cbRC, nRU; cbRC.alloc(true, W); cbRC.zero(); nRU.alloc(true, W); nRU.zero();
    lap("upload");

    // ---- pass 1, per read chunk
    const u64 capInt = (u64)0x7fffff00ULL;
    const u64 perRead = 52;   // chunk uploads 20, annotation 8, CR sort 16 (before the keys exist) or keys 8 + flags 1 + compacted keys 8 (after)
    CK(cudaMemGetInfo(&frFree, &devTot));
    u64 C;
    if (opt.chunkReads) C = opt.chunkReads;
    else {
        const double avail = ((double)frFree * 0.9 - 256.0 * 1048576.0) * headroom;
        if (avail < (double)perRead * 65536.0) throw DevOOM("two-pass Solo: not enough free device memory for a chunk of 65536 reads");
        C = (u64)(avail / (double)perRead);
    }
    C = std::max<u64>(1, std::min<u64>(std::min<u64>(C, capInt), std::max<u64>(N, 1)));
    Buf<u64> cCr3, cUr3; Buf<u32> cRec; Buf<RecAnn> cAnn;
    std::vector<RecAnn> hAnn;
    if (N) {
        if (!Rd.dev) { cCr3.alloc(true, C); cUr3.alloc(true, C); cRec.alloc(true, C); }
        cAnn.alloc(true, C);
        if (!recOnDev) hAnn.resize(C);
    }
    std::vector<KeyBlock> blocks;
    std::vector<DefRead> defs;
    u64 nChunks = 0, nDistinctSum = 0, nCandSum = 0, nKeysSoFar = 0, hk = 0;
    const u64 mk = e6::mask3(S.cbL);
    for (u64 i0 = 0; i0 < N; i0 += C) {
        const u64 m = std::min<u64>(C, N - i0);
        nChunks++;
        const u64* cr3 = Rd.dev ? Rd.cr3 + i0 : cCr3.p;
        const u64* ur3 = Rd.dev ? Rd.ur3 + i0 : cUr3.p;
        const u32* rec = Rd.dev ? Rd.rec + i0 : cRec.p;
        if (!Rd.dev) {
            CK(cudaMemcpy(cCr3.p, Rd.cr3 + i0, m * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(cUr3.p, Rd.ur3 + i0, m * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(cRec.p, Rd.rec + i0, m * 4, cudaMemcpyHostToDevice));
        }
        // stage (b): per read its record's annotation (the host's for a read without a record, below)
        RecAnn* ann = cAnn.p;
        {
            u64* c = cnt.p;
            if (recOnDev) {
                const RecAnn* rac = dRecAnn.p;
                each(true, m, [=] __host__ __device__(u64 i) {
                    const u32 r = rec[i];
                    RecAnn a{0xFFFFFFFFu, 0, 0, 0};
                    if (r != 0xFFFFFFFFu) a = rac[r]; else incAgg(c, 2);
                    ann[i] = a;
                });
            } else {   // the annotations stay in host memory: gather the chunk's there
                const u32* hrec = Rd.rec + i0; const RecAnn* hra = R.recAnn; RecAnn* hout = hAnn.data();
#pragma omp parallel for schedule(static)
                for (long long k = 0; k < (long long)m; k++) {
                    const u32 r = hrec[k];
                    hout[k] = r != 0xFFFFFFFFu ? hra[r] : RecAnn{0xFFFFFFFFu, 0, 0, 0};
                }
                CK(cudaMemcpy(ann, hout, m * sizeof(RecAnn), cudaMemcpyHostToDevice));
                each(true, m, [=] __host__ __device__(u64 i) { if (rec[i] == 0xFFFFFFFFu) incAgg(c, 2); });
            }
        }
        {   // the reads the host mapped (sparse, ascending): their annotation
            u64 hk1 = hk;
            while (hk1 < Rd.nHost && (u64)Rd.hostIdx[hk1] < i0 + m) hk1++;
            const u64 nk = hk1 - hk;
            if (nk) {
                Buf<u32> dHi; dHi.from(true, Rd.hostIdx + hk, nk);
                Buf<RecAnn> dHa; dHa.from(true, Rd.hostAnnS + hk, nk);
                const u32* hi = dHi.p; const RecAnn* ha = dHa.p; u64* c = cnt.p;
                each(true, nk, [=] __host__ __device__(u64 k) {
                    const u64 li = (u64)hi[k] - i0;
                    if (rec[li] != 0xFFFFFFFFu) incAgg(c, 3);
                    ann[li] = ha[k];
                });
            }
            hk = hk1;
        }
        // stage (a): the distinct CRs of the chunk, their whitelist matches
        Buf<u64> dcr; u64 D = 0;
        {
            Buf<u64> crk; crk.alloc(true, m);
            { const u64* c3 = cr3; u64* k = crk.p; each(true, m, [=] __host__ __device__(u64 i) { k[i] = c3[i] & mk; }); }
            sortKeys(true, crk, m, (int)(3 * S.cbL), tmp);
            Buf<u64> uq;
            D = uniqueKeys(true, crk, m, uq, tmp);
            crk.release();
            dcr.alloc(true, D);
            if (D) CK(cudaMemcpy(dcr.p, uq.p, D * 8, cudaMemcpyDeviceToDevice));
        }
        Buf<i32> dcbm; dcbm.alloc(true, D);
        Buf<u32> nC; nC.alloc(true, D + 1); nC.zero();
        {
            const u64* cr = dcr.p; i32* mm = dcbm.p; u32* nc = nC.p; const u64* wl = dWl.p; const u32 cbL = S.cbL;
            each(true, D, [=] __host__ __device__(u64 d) { u32 n; mm[d] = matchCBtoWL(cr[d], cbL, wl, W, nullptr, n); nc[d] = n; });
        }
        Buf<u32> cOff;
        const u64 nCand = exclusiveScan(true, nC, D, cOff, tmp);
        Buf<u32> cand; cand.alloc(true, nCand);
        {
            const u64* cr = dcr.p; const u32* co = cOff.p; u32* cd = cand.p; const u64* wl = dWl.p; const u32 cbL = S.cbL;
            each(true, D, [=] __host__ __device__(u64 d) { u32 n; matchCBtoWL(cr[d], cbL, wl, W, cd + co[d], n); });
        }
        nDistinctSum += D; nCandSum += nCand;
        // per read: the UMI check, Barcodes.stats, the exact histogram, and stage (c) unless the read has several matches (those wait for pass 2)
        Buf<u64> keys; keys.alloc(true, m);
        Buf<uint8_t> dfl; dfl.alloc(true, m);
        CK(cudaMemset(cnt.p + 5, 0, 8));
        {
            const u64* cr = dcr.p; const u64 Dn = D; const i32* mm = dcbm.p; const u32* co = cOff.p; const u32* nc = nC.p; const u32* cd = cand.p;
            u64* ky = keys.p; uint8_t* fl = dfl.p; u64* bs = bst.p; u32* ex = exact.p; u64* c = cnt.p; const SoloParams s = S;
            const CountSinks sk{fst.p, noCB.p, cbRC.p, nRU.p, flagRows.p, cnt.p};
            each(true, m, [=] __host__ __device__(u64 i) {
                const u64 d = lowerBound(cr, Dn, cr3[i] & mk);
                u64 ub = 0;
                const i32 uc = checkUMI(ur3[i], s, ub);
                const i32 cbm = uc != 0 ? uc : mm[d];   // getCBandUMI: a bad UMI replaces cbMatch (SoloReadBarcode_getCBandUMI.cpp:290-302)
                incAgg(bs, (u32)barcodeStat(cbm));
                const u32* cdp = cd + co[d];
                if (cbm == 0) addU32(&ex[cdp[0]], 1);   // cbReadCountExact (addStats case 0)
                ky[i] = ~0ULL; fl[i] = 0;
                if (cbm > 1) { fl[i] = 1; incAgg(c, 5); return; }
                const i32 cb = (cbm == 0 || cbm == 1) ? (i32)cdp[0] : -1;
                ky[i] = countRead(ann[i], cbm, cdp, nc[d], cb, (u32)ub, sk);
            });
        }
        const u64 nDef = cnt.get(5), nKeysNow = cnt.get(1);
        const u64 nKc = nKeysNow - nKeysSoFar;
        nKeysSoFar = nKeysNow;
        if (nKc) {   // the counted keys to host memory
            Buf<u64> kc; kc.alloc(true, nKc);
            if (selectCounted(keys.p, m, kc.p, tmp) != nKc) throw std::runtime_error("two-pass Solo: the chunk's counted keys do not add up");
            blocks.emplace_back();
            blocks.back().alloc(nKc);
            kc.to(blocks.back().p, nKc);
        }
        keys.release();
        if (nDef) {   // the reads with several matches, compactly, for pass 2
            Buf<u32> idxL; idxL.alloc(true, nDef);
            if (selectFlagged(dfl.p, m, idxL.p, tmp) != nDef) throw std::runtime_error("two-pass Solo: the chunk's multi-match reads do not add up");
            Buf<DefRead> dd; dd.alloc(true, nDef);
            {
                const u32* il = idxL.p; DefRead* o = dd.p; const SoloParams s = S;
                each(true, nDef, [=] __host__ __device__(u64 j) {
                    const u64 i = il[j];
                    u64 ub = 0;
                    checkUMI(ur3[i], s, ub);
                    DefRead df; df.idx = i0 + i; df.cr = cr3[i] & mk; df.a = ann[i]; df.umi = (u32)ub; df.pad = 0;
                    o[j] = df;
                });
            }
            const u64 old = defs.size();
            defs.resize(old + nDef);
            dd.to(defs.data() + old, nDef);
        }
    }
    cCr3.release(); cUr3.release(); cRec.release(); cAnn.release(); std::vector<RecAnn>().swap(hAnn);
    out.nChunks = nChunks; out.nDistinctCR = nDistinctSum; out.nCand = nCandSum;
    if (Rd.nHost) {
        const u64 nNone = cnt.get(2), nBad = cnt.get(3);
        if (nNone != Rd.nHost || nBad) throw std::runtime_error("host annotations: " + std::to_string(Rd.nHost) + " given, " + std::to_string(nNone)
                                                                + " reads without a record, " + std::to_string(nBad) + " given for reads with a record");
    }
    lap("pass1");

    // ---- pass 2a: the posterior of the reads with several matches, from the complete histogram (E15), their stage (c)
    const u64 nMM = defs.size();
    out.nMulti = nMM;
    out.perReadQual = (bool)Rd.cbQual;
    {
        const bool perRead = Rd.cbQual && nMM > 0;
        Buf<double> dPq;
        if (perRead) {
            if (M.pqTab.size() != 256) throw std::runtime_error("per-read CB qualities without the quality table");
            dPq.from(true, M.pqTab.data(), 256);
        }
        const u64 Bm = std::max<u64>(1, std::min<u64>(nMM, (u64)1 << 24));
        Buf<DefRead> dDef; Buf<uint8_t> dQ; Buf<u64> bKeys;
        if (nMM) { dDef.alloc(true, Bm); bKeys.alloc(true, Bm); if (perRead) dQ.alloc(true, Bm * S.cbL); }
        std::vector<u64> hIdx, hCr; std::vector<uint8_t> hQual;
        for (u64 b0 = 0; b0 < nMM; b0 += Bm) {
            const u64 m = std::min<u64>(Bm, nMM - b0);
            CK(cudaMemcpy(dDef.p, defs.data() + b0, m * sizeof(DefRead), cudaMemcpyHostToDevice));
            if (perRead) {
                hIdx.resize(m); hCr.resize(m); hQual.resize(m * S.cbL);
                for (u64 k = 0; k < m; k++) { hIdx[k] = defs[b0 + k].idx; hCr[k] = defs[b0 + k].cr; }
                std::string why;
                if (!Rd.cbQual(hIdx.data(), hCr.data(), m, hQual.data(), why)) throw std::runtime_error("CB qualities of the multi-match reads: " + why);
                CK(cudaMemcpy(dQ.p, hQual.data(), hQual.size(), cudaMemcpyHostToDevice));
            }
            {
                const DefRead* dd = dDef.p; const u64* wl = dWl.p; const u32* ex = exact.p; const uint8_t* q = dQ.p; const double* pt = dPq.p; u64* ky = bKeys.p;
                u64* c = cnt.p; const SoloParams s = S; const u32 cbL = S.cbL; const CountSinks sk{fst.p, noCB.p, cbRC.p, nRU.p, flagRows.p, cnt.p};
                each(true, m, [=] __host__ __device__(u64 k) {
                    const DefRead df = dd[k];
                    u32 cd[64]; u32 n = 0;
                    const i32 cbm = matchCBtoWL(df.cr, cbL, wl, W, cd, n);   // the candidates in STAR's order: cbm = n > 1
                    const i64 r = cbPosteriorRead(df.cr, cd, n, ex, wl, perRead ? q + k * cbL : nullptr, pt, s);
                    if (r >= 0) addU64(c + 4, 1);
                    ky[k] = countRead(df.a, cbm, cd, n, (i32)r, df.umi, sk);
                });
            }
            const u64 before = nKeysSoFar;
            nKeysSoFar = cnt.get(1);
            const u64 nKc = nKeysSoFar - before;
            if (nKc) {
                Buf<u64> kc; kc.alloc(true, nKc);
                if (selectCounted(bKeys.p, m, kc.p, tmp) != nKc) throw std::runtime_error("two-pass Solo: the multi-match keys do not add up");
                blocks.emplace_back();
                blocks.back().alloc(nKc);
                kc.to(blocks.back().p, nKc);
            }
        }
    }
    std::vector<DefRead>().swap(defs);
    out.nMultiAccepted = cnt.get(4);
    lap("multi");

    // ---- pass 2b: the keys by whitelist-index range (a bucket), each sorted, uniqued and collapsed on the device, in bucket order
    u64 nK = 0, maxBlock = 0;
    for (const auto& b : blocks) { nK += b.n; maxBlock = std::max(maxBlock, b.n); }
    if (nK != cnt.get(1)) throw std::runtime_error("two-pass Solo: the stored keys (" + std::to_string(nK) + ") differ from the counted reads (" + std::to_string(cnt.get(1)) + ")");
    out.nKeys = nK;
    std::vector<u32> hNru(W);
    nRU.to(hNru.data(), W);   // the keys per CB: every counted read adds one
    u64 sumNru = 0, maxCb = 0;
    for (u32 w = 0; w < W; w++) { sumNru += hNru[w]; maxCb = std::max<u64>(maxCb, hNru[w]); }
    if (sumNru != nK) throw std::runtime_error("two-pass Solo: the keys per CB do not add up to the counted reads");
    CK(cudaMemGetInfo(&frFree, &devTot));
    const u64 ST = std::max<u64>(1, std::min<u64>(maxBlock, (u64)1 << 26));   // keys per upload while filling a bucket
    u64 capKeys;
    if (opt.bucketKeys) capKeys = opt.bucketKeys;
    else if (opt.bucketTarget) capKeys = std::max<u64>((nK + opt.bucketTarget - 1) / opt.bucketTarget, maxCb);
    else {
        const double avail = ((double)frFree * 0.9 - 256.0 * 1048576.0 - 8.0 * (double)ST) * headroom;
        if (nK > 0 && avail < 36.0 * 65536.0) throw DevOOM("two-pass Solo: not enough free device memory for a bucket of 65536 keys");
        capKeys = (u64)std::max(0.0, avail / 36.0);
    }
    capKeys = std::max<u64>(1, std::min<u64>(capKeys, capInt));
    struct Bkt { u32 w0, w1; u64 nk; };
    std::vector<Bkt> plan;
    {
        u64 acc = 0; u32 start = 0;
        for (u32 w = 0; w < W; w++) {
            if (hNru[w] > capKeys) throw std::runtime_error("two-pass Solo: CB " + std::to_string(w) + " has " + std::to_string(hNru[w]) + " counted reads, more than a bucket of " + std::to_string(capKeys) + " keys holds");
            if (acc > 0 && acc + hNru[w] > capKeys) { plan.push_back({start, w, acc}); start = w; acc = 0; }
            acc += hNru[w];
        }
        if (acc > 0) plan.push_back({start, W, acc});
    }
    Buf<u32> nUMI, nGene; nUMI.alloc(true, W); nUMI.zero(); nGene.alloc(true, W); nGene.zero();
    u64 nT = 0, nSeg = 0, maxBucket = 0;
    {
        Buf<u64> stg; if (nK) stg.alloc(true, ST);
        for (const Bkt& bk : plan) {
            maxBucket = std::max(maxBucket, bk.nk);
            Buf<u64> keys; keys.alloc(true, bk.nk);
            CK(cudaMemset(cnt.p + 6, 0, 8));
            for (const auto& blk : blocks)
                for (u64 off = 0; off < blk.n; off += ST) {
                    const u64 mm = std::min<u64>(ST, blk.n - off);
                    CK(cudaMemcpy(stg.p, blk.p + off, mm * 8, cudaMemcpyHostToDevice));
                    const u64* in = stg.p; u64* o = keys.p; u64* ap = cnt.p + 6; const u64 lo = bk.w0, hi = bk.w1, nk = bk.nk;
                    each(true, mm, [=] __host__ __device__(u64 i) {
                        const u64 key = in[i];
                        const u64 cb = key >> (GENE_BITS + UMI_BITS);
                        if (cb >= lo && cb < hi) { const u64 pos = fetchAdd(ap, 1); if (pos < nk) o[pos] = key; }
                    });
                }
            if (cnt.get(6) != bk.nk) throw std::runtime_error("two-pass Solo: a bucket holds another number of keys than its CBs' counts");
            // the (CB, gene, UMI) of the counted reads: sort, unique; the second order (UMI halves swapped)
            sortKeys(true, keys, bk.nk, 64, tmp);
            Buf<u64> tup;
            const u64 nTb = uniqueKeys(true, keys, bk.nk, tup, tmp);
            keys.release();
            nT += nTb;
            Buf<u64> k2; k2.alloc(true, nTb);
            Buf<u32> v2; v2.alloc(true, nTb);
            {
                const u64* t = tup.p; u64* a = k2.p; u32* b = v2.p; const u32 umiL = S.umiL, ml = S.umiMaskLow;
                each(true, nTb, [=] __host__ __device__(u64 j) { a[j] = swapHalvesKey(t[j], umiL, ml); b[j] = (u32)j; });
            }
            sortPairs(true, k2, v2, nTb, 63, tmp);
            // 1MM_All per (CB, gene): Hamming-1 components (collapseUMIwith1MMlowHalf on both halves), then the counts per CB
            Buf<u32> par; par.alloc(true, nTb);
            {
                u32* pr = par.p;
                each(true, nTb, [=] __host__ __device__(u64 j) { pr[j] = (u32)j; });
                const u64* t = tup.p; const u64 ml = S.umiMaskLow; const u64 n = nTb;
                each(true, nTb, [=] __host__ __device__(u64 j) {
                    for (u64 k = j + 1; k < n; k++) {
                        const u64 x = t[k] ^ t[j];
                        if (x > ml) break;   // another (CB, gene) or another upper half
                        if (oneMM((u32)x)) ufUnion(pr, (u32)j, (u32)k);
                    }
                });
                const u64* a = k2.p; const u32* b = v2.p;
                each(true, nTb, [=] __host__ __device__(u64 j) {
                    for (u64 k = j + 1; k < n; k++) {
                        const u64 x = a[k] ^ a[j];
                        if (x > ml) break;
                        if (oneMM((u32)x)) ufUnion(pr, b[j], b[k]);
                    }
                });
            }
            k2.release(); v2.release();
            Buf<u32> hd; hd.alloc(true, nTb);
            {
                const u64* t = tup.p; u32* h = hd.p;
                each(true, nTb, [=] __host__ __device__(u64 j) { h[j] = (j == 0 || (t[j] >> UMI_BITS) != (t[j - 1] >> UMI_BITS)) ? 1u : 0u; });
            }
            Buf<u32> sid;
            inclusiveScan(true, hd, nTb, sid, tmp);
            const u64 nSegb = nTb ? sid.get(nTb - 1) : 0;
            Buf<u64> segKey; segKey.alloc(true, nSegb);
            Buf<u32> segCnt; segCnt.alloc(true, nSegb); segCnt.zero();
            {
                const u64* t = tup.p; const u32* h = hd.p; const u32* si = sid.p; const u32* pr = par.p; u64* sk = segKey.p; u32* sc = segCnt.p;
                each(true, nTb, [=] __host__ __device__(u64 j) {
                    const u32 s = si[j] - 1;
                    if (h[j]) sk[s] = t[j] >> UMI_BITS;
                    if (pr[j] == (u32)j) addU32(&sc[s], 1);
                });
            }
            {
                const u64* sk = segKey.p; const u32* sc = segCnt.p; u32* nu = nUMI.p; u32* ng = nGene.p;
                each(true, nSegb, [=] __host__ __device__(u64 s) {
                    const u64 cb = sk[s] >> GENE_BITS;
                    addU32(&nu[cb], sc[s]);
                    addU32(&ng[cb], 1);
                });
            }
            // the bucket's segments, appended: ascending (CB, gene), and the buckets ascend in CB
            const u64 s0 = out.segCb.size();
            out.segCb.resize(s0 + nSegb); out.segGene.resize(s0 + nSegb); out.segCount.resize(s0 + nSegb);
            std::vector<u64> sk(nSegb); segKey.to(sk.data(), nSegb);
            segCnt.to(out.segCount.data() + s0, nSegb);
            for (u64 s = 0; s < nSegb; s++) { out.segCb[s0 + s] = (u32)(sk[s] >> GENE_BITS); out.segGene[s0 + s] = (u32)(sk[s] & ((1u << GENE_BITS) - 1)); }
            nSeg += nSegb;
        }
    }
    blocks.clear();
    out.nBuckets = plan.size(); out.nTuples = nT; out.nSeg = nSeg;
    lap("collapse");

    // ---- the CBs with a count or a CellReads row, compacted (a CellReads row always has its cbMatch count; nUMI, nGene and
    // nReadUnique > 0 imply cbReadCount > 0)
    Buf<u32> pres; pres.alloc(true, (u64)W + 1); pres.zero();
    {
        const u32* crc = cbRC.p; const u32* fr = flagRows.p; u32* pr = pres.p;
        each(true, W, [=] __host__ __device__(u64 w) { pr[w] = (crc[w] > 0 || fr[w * F_N + F_cbMatch] > 0) ? 1u : 0u; });
    }
    Buf<u32> pOff;
    const u64 nP = exclusiveScan(true, pres, W, pOff, tmp);
    Buf<u32> pIdx, pVal; pIdx.alloc(true, nP); pVal.alloc(true, nP * CBV);
    {
        const u32* pr = pres.p; const u32* po = pOff.p; u32* pi = pIdx.p; u32* pv = pVal.p;
        const u32* crc = cbRC.p; const u32* nu = nUMI.p; const u32* ng = nGene.p; const u32* nru = nRU.p; const u32* fr = flagRows.p;
        each(true, W, [=] __host__ __device__(u64 w) {
            if (!pr[w]) return;
            const u64 k = po[w];
            pi[k] = (u32)w;
            u32* v = pv + k * CBV;
            v[CV_cbReadCount] = crc[w]; v[CV_nUMI] = nu[w]; v[CV_nGene] = ng[w]; v[CV_nReadUnique] = nru[w];
            for (u32 b = 0; b < F_N; b++) v[CV_flags + b] = fr[w * F_N + b];
        });
    }
    lap("compact");

    // ---- download
    bst.to(out.bstats, B_N); fst.to(out.fstats, S_N); noCB.to(out.noCB, F_N);
    out.cbIdx.resize(nP); pIdx.to(out.cbIdx.data(), nP);
    out.cbVal.resize(nP * CBV); pVal.to(out.cbVal.data(), nP * CBV);
    out.nRecOverflow = 0;
    lap("download");

    // ---- Features.stats rows that collapseUMIall adds (SoloFeature_collapseUMIall.cpp:21-25)
    u64 yesUMIs = 0, yesCB = 0, yesWL = 0;
    for (u64 k = 0; k < nP; k++) {
        const u32* v = &out.cbVal[k * CBV];
        yesUMIs += v[CV_nUMI]; yesCB += v[CV_nGene] > 0; yesWL += v[CV_nReadUnique];
    }
    out.fstats[S_yesUMIs] = yesUMIs; out.fstats[S_yesCellBarcodes] = yesCB;
    out.fstats[S_yesWLmatch] = yesWL; out.fstats[S_yessubWLmatch_UniqueFeature] = yesWL;
    char b[768];
    snprintf(b, sizeof b, "soloRun(two-pass): %llu reads in %llu chunks of up to %llu, %llu records, %llu distinct CRs and %llu matches (sums over the chunks), %llu counted reads, %llu distinct (CB,gene,UMI) in %llu buckets of up to %llu keys, %llu (CB,gene); %llu reads with several matches (%llu given a CB by the posterior, %s)\n",
             (unsigned long long)N, (unsigned long long)nChunks, (unsigned long long)C, (unsigned long long)nRec, (unsigned long long)nDistinctSum,
             (unsigned long long)nCandSum, (unsigned long long)nK, (unsigned long long)nT, (unsigned long long)plan.size(), (unsigned long long)maxBucket,
             (unsigned long long)nSeg, (unsigned long long)nMM, (unsigned long long)out.nMultiAccepted, out.perReadQual ? "per-read CB qualities" : "one CB quality factor");
    log += b;
    if (g_use.on) log += "\n" + devUseLine("device memory beyond the inputs, peak per stage (GiB):") + "\n";
}

bool soloRunTwoPass(const SoloParams& S, const SoloModel& M, const SoloRecs& R, const SoloReads& Rd, SoloResult& out, std::string& log, const SoloTwoPass& opt) {
    ::singlet::gpu_ledger::Phase ledger_phase("solo");
    g_lastOOM = false;
    double headroom = 1.0;
    for (int attempt = 0;; attempt++) {
        std::string lg;
        try {
            out = SoloResult{};
            soloTwoPassOnce(S, M, R, Rd, out, lg, opt, headroom);
            log += lg;
            return true;
        } catch (const DevOOM& e) {
            cudaGetLastError();
            if (attempt < 3 && !opt.chunkReads && !opt.bucketKeys && !opt.bucketTarget) {   // sized from the free memory: try half the size
                headroom *= 0.5;
                log += std::string("soloRun(two-pass): ") + e.what() + "; retrying with chunks and buckets of half the size\n";
                continue;
            }
            g_lastOOM = true;
            log += std::string("soloRun(two-pass) failed: ") + e.what() + "\n";
            if (g_use.on) log += devUseLine("device memory beyond the inputs before the failure, peak per stage (GiB):") + "\n";
            return false;
        } catch (const std::exception& e) {
            cudaGetLastError();
            log += std::string("soloRun(two-pass) failed: ") + e.what() + "\n";
            return false;
        }
    }
}

// ================================================================================================ E11: the engine's keep API
namespace {
SoloKeep g_keep;
double msSince(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); }
}
SoloKeep& soloKeep() { return g_keep; }
void soloKeepRelease() {
    SoloKeep& K = g_keep;
    if (K.toHost.joinable()) K.toHost.join();
    if (K.cr3) cudaFree(K.cr3);
    if (K.ur3) cudaFree(K.ur3);
    if (K.rec) cudaFree(K.rec);
    if (K.recAnn) cudaFree(K.recAnn);
    if (K.hcr3) cudaFreeHost(K.hcr3);
    if (K.hur3) cudaFreeHost(K.hur3);
    if (K.hrec) cudaFreeHost(K.hrec);
    if (K.hrecAnn) cudaFreeHost(K.hrecAnn);
    K = SoloKeep{};
}
// E15 (E11 §5.3): the four kept arrays to page-locked host memory on a thread (STAR's loop runs meanwhile); the device copies are freed only
// when every copy succeeded, otherwise the arrays stay on the device (hostWhy says why)
void soloKeepToHost(int device) {
    SoloKeep& K = g_keep;
    if (!K.reads || !K.recs || K.onHost || K.toHost.joinable()) return;
    K.toHost = std::thread([device]() {
        SoloKeep& K = g_keep;
        const auto t0 = std::chrono::steady_clock::now();
        auto freeHost = [&]() {
            if (K.hcr3) cudaFreeHost(K.hcr3);
            if (K.hur3) cudaFreeHost(K.hur3);
            if (K.hrec) cudaFreeHost(K.hrec);
            if (K.hrecAnn) cudaFreeHost(K.hrecAnn);
            K.hcr3 = K.hur3 = nullptr; K.hrec = nullptr; K.hrecAnn = nullptr;
        };
        const u64 n = std::max<u64>(1, K.n), nR = std::max<u64>(1, K.nRec);
        cudaError_t e = cudaSetDevice(device);
        if (e == cudaSuccess) e = cudaMallocHost(&K.hcr3, n * 8);
        if (e == cudaSuccess) e = cudaMallocHost(&K.hur3, n * 8);
        if (e == cudaSuccess) e = cudaMallocHost(&K.hrec, n * 4);
        if (e == cudaSuccess) e = cudaMallocHost(&K.hrecAnn, nR * sizeof(RecAnn));
        if (e == cudaSuccess) e = cudaMemcpy(K.hcr3, K.cr3, n * 8, cudaMemcpyDeviceToHost);
        if (e == cudaSuccess) e = cudaMemcpy(K.hur3, K.ur3, n * 8, cudaMemcpyDeviceToHost);
        if (e == cudaSuccess) e = cudaMemcpy(K.hrec, K.rec, n * 4, cudaMemcpyDeviceToHost);
        if (e == cudaSuccess) e = cudaMemcpy(K.hrecAnn, K.recAnn, nR * sizeof(RecAnn), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) { K.hostWhy = cudaGetErrorString(e); cudaGetLastError(); freeHost(); return; }
        cudaFree(K.cr3); cudaFree(K.ur3); cudaFree(K.rec); cudaFree(K.recAnn);
        K.cr3 = K.ur3 = nullptr; K.rec = nullptr; K.recAnn = nullptr;
        K.onHost = true; K.msToHost = msSince(t0);
    });
}
bool soloKeepReads(const u64* dAux, u64 stride, u64 n, std::string& log) {
    try {
        const auto t0 = std::chrono::steady_clock::now();
        SoloKeep& K = g_keep;
        if (K.cr3) cudaFree(K.cr3);
        if (K.ur3) cudaFree(K.ur3);
        K.cr3 = K.ur3 = nullptr; K.reads = false;
        CK(cudaMalloc(&K.cr3, std::max<u64>(1, n) * 8));
        CK(cudaMalloc(&K.ur3, std::max<u64>(1, n) * 8));
        u64* c = K.cr3; u64* u = K.ur3; const u64 st = stride;
        each(true, n, [=] __host__ __device__(u64 i) { c[i] = dAux[i * st]; u[i] = dAux[i * st + 1]; });
        CK(cudaDeviceSynchronize());
        K.n = n; K.reads = true; K.msReads = msSince(t0);
        return true;
    } catch (const std::exception& e) {
        log += std::string("soloKeepReads failed: ") + e.what() + "\n";
        cudaGetLastError(); soloKeepRelease();
        return false;
    }
}
bool soloKeepRecords(const SoloParams& S, const SoloModel& M, const e6::RecParams& RP, const uint8_t* dBlob, const u64* dBlobOff, const u64* dSeqOff,
                     u64 nRec, const u32* dRec, u64 n, std::string& log) {
    try {
        const auto t0 = std::chrono::steady_clock::now();
        SoloKeep& K = g_keep;
        if (!K.reads || K.n != n) throw std::runtime_error("the reads were not kept");
        if (K.rec) cudaFree(K.rec);
        if (K.recAnn) cudaFree(K.recAnn);
        K.rec = nullptr; K.recAnn = nullptr; K.recs = false;
        Buf<u64> dTrS, dTrE, dTrEmax; Buf<u32> dTrExI, dTrGene, dExSE; Buf<uint16_t> dTrExN; Buf<uint8_t> dTrStr;
        dTrS.from(true, M.trS.data(), M.trS.size()); dTrE.from(true, M.trE.data(), M.trE.size()); dTrEmax.from(true, M.trEmax.data(), M.trEmax.size());
        dTrExI.from(true, M.trExI.data(), M.trExI.size()); dTrGene.from(true, M.trGene.data(), M.trGene.size()); dExSE.from(true, M.exSE.data(), M.exSE.size());
        dTrExN.from(true, M.trExN.data(), M.trExN.size()); dTrStr.from(true, M.trStr.data(), M.trStr.size());
        const TrModel tm{dTrS.p, dTrE.p, dTrEmax.p, dTrExI.p, dTrExN.p, dTrStr.p, dTrGene.p, dExSE.p, (u32)M.trS.size()};
        CK(cudaMalloc(&K.rec, std::max<u64>(1, n) * 4));
        CK(cudaMalloc(&K.recAnn, std::max<u64>(1, nRec) * sizeof(RecAnn)));
        if (n) CK(cudaMemcpy(K.rec, dRec, n * 4, cudaMemcpyDeviceToDevice));
        Buf<u64> cnt; cnt.alloc(true, 1); cnt.zero();
        RecAnn* ra = K.recAnn; u64* c = cnt.p; const SoloParams s = S; const e6::RecParams rp = RP;
        each(true, nRec, [=] __host__ __device__(u64 r) {
            const RecAnn a = recAnnotate(rp, dBlob + dBlobOff[r], dSeqOff[r + 1] - dSeqOff[r], tm, s);
            ra[r] = a;
            if (a.bits & RA_OVERFLOW) addU64(c, 1);
        });
        CK(cudaDeviceSynchronize());
        K.nOverflow = cnt.get(0); K.nRec = nRec; K.recs = true; K.msRecs = msSince(t0);
        return true;
    } catch (const std::exception& e) {
        log += std::string("soloKeepRecords failed: ") + e.what() + "\n";
        cudaGetLastError(); soloKeepRelease();
        return false;
    }
}
bool soloKeepReadsPart(const u64* dAux, u64 stride, u64 i0, u64 m, u64 n, std::string& log) {
    try {
        const auto t0 = std::chrono::steady_clock::now();
        SoloKeep& K = g_keep;
        if (i0 == 0) {
            if (K.cr3) cudaFree(K.cr3);
            if (K.ur3) cudaFree(K.ur3);
            K.cr3 = K.ur3 = nullptr; K.reads = false; K.msReads = 0;
            CK(cudaMalloc(&K.cr3, std::max<u64>(1, n) * 8));
            CK(cudaMalloc(&K.ur3, std::max<u64>(1, n) * 8));
        } else if (!K.cr3 || !K.ur3) throw std::runtime_error("the reads' arrays do not exist");
        u64* c = K.cr3 + i0; u64* u = K.ur3 + i0; const u64 st = stride;
        each(true, m, [=] __host__ __device__(u64 i) { c[i] = dAux[i * st]; u[i] = dAux[i * st + 1]; });
        CK(cudaDeviceSynchronize());
        K.msReads += msSince(t0);
        if (i0 + m == n) { K.n = n; K.reads = true; }
        return true;
    } catch (const std::exception& e) {
        log += std::string("soloKeepReadsPart failed: ") + e.what() + "\n";
        cudaGetLastError(); soloKeepRelease();
        return false;
    }
}
bool soloKeepRecordsBegin(u64 nRec, const u32* dRec, u64 n, std::string& log) {
    try {
        SoloKeep& K = g_keep;
        if (!K.reads || K.n != n) throw std::runtime_error("the reads were not kept");
        if (K.rec) cudaFree(K.rec);
        if (K.recAnn) cudaFree(K.recAnn);
        K.rec = nullptr; K.recAnn = nullptr; K.recs = false; K.nOverflow = 0; K.msRecs = 0;
        CK(cudaMalloc(&K.rec, std::max<u64>(1, n) * 4));
        CK(cudaMalloc(&K.recAnn, std::max<u64>(1, nRec) * sizeof(RecAnn)));
        if (n) CK(cudaMemcpy(K.rec, dRec, n * 4, cudaMemcpyDeviceToDevice));
        K.nRec = nRec;
        return true;
    } catch (const std::exception& e) {
        log += std::string("soloKeepRecordsBegin failed: ") + e.what() + "\n";
        cudaGetLastError(); soloKeepRelease();
        return false;
    }
}
bool soloKeepRecordsPart(const SoloParams& S, const SoloModel& M, const e6::RecParams& RP, const uint8_t* dBlob, const u64* dBlobOff, const u64* dSeqOff,
                         u64 r0, u64 m, std::string& log) {
    try {
        SoloKeep& K = g_keep;
        if (!K.recAnn || r0 + m > K.nRec) throw std::runtime_error("the records' annotation array does not hold this part");
        Buf<u64> dTrS, dTrE, dTrEmax; Buf<u32> dTrExI, dTrGene, dExSE; Buf<uint16_t> dTrExN; Buf<uint8_t> dTrStr;
        dTrS.from(true, M.trS.data(), M.trS.size()); dTrE.from(true, M.trE.data(), M.trE.size()); dTrEmax.from(true, M.trEmax.data(), M.trEmax.size());
        dTrExI.from(true, M.trExI.data(), M.trExI.size()); dTrGene.from(true, M.trGene.data(), M.trGene.size()); dExSE.from(true, M.exSE.data(), M.exSE.size());
        dTrExN.from(true, M.trExN.data(), M.trExN.size()); dTrStr.from(true, M.trStr.data(), M.trStr.size());
        const TrModel tm{dTrS.p, dTrE.p, dTrEmax.p, dTrExI.p, dTrExN.p, dTrStr.p, dTrGene.p, dExSE.p, (u32)M.trS.size()};
        Buf<u64> cnt; cnt.alloc(true, 1); cnt.zero();
        RecAnn* ra = K.recAnn + r0; u64* c = cnt.p; const SoloParams s = S; const e6::RecParams rp = RP;
        each(true, m, [=] __host__ __device__(u64 r) {
            const RecAnn a = recAnnotate(rp, dBlob + dBlobOff[r], dSeqOff[r + 1] - dSeqOff[r], tm, s);
            ra[r] = a;
            if (a.bits & RA_OVERFLOW) addU64(c, 1);
        });
        CK(cudaDeviceSynchronize());
        K.nOverflow += cnt.get(0);
        return true;
    } catch (const std::exception& e) {
        log += std::string("soloKeepRecordsPart failed: ") + e.what() + "\n";
        cudaGetLastError(); soloKeepRelease();
        return false;
    }
}
void soloKeepRecordsEnd(double ms) { g_keep.recs = true; g_keep.msRecs = ms; }
namespace {
// What soloRun(device) allocates beyond the kept inputs, as a bound (bytes): per read the largest live set of its stages (the whole-run
// bound is the stage (c) set: the annotation 8, the distinct-CR array 8 (sized to the reads), the CR index 4, the UMI 4, the match 1, the keys 8,
// the CellReads row 4; the sort of the keys and the CR sort live 4 to 8 more) plus CUB's scratch, per whitelist entry and per record the small
// arrays. SINGLET_SOLO_DEV_BYTES_PER_READ overrides the per-read bytes.
u64 soloDeviceNeed(u64 N, u64 nHost, const SoloModel& M) {
    double perRead = 39.0;
    if (const char* e = getenv("SINGLET_SOLO_DEV_BYTES_PER_READ")) { const double v = atof(e); if (v > 0) perRead = v; }
    const double W = (double)M.wl.size();
    const double model = 8.0 * (M.trS.size() + M.trE.size() + M.trEmax.size()) + 4.0 * (M.trExI.size() + M.trGene.size() + M.exSE.size())
                         + 2.0 * M.trExN.size() + M.trStr.size();
    const double perWl = 8.0 + 4.0 * F_N + 44.0;   // the whitelist, flagRows, exact / cbRC / nRU / pres / pOff / nUMI / nGene, the compacted rows
    return (u64)(perRead * (double)N + 12.0 * (double)nHost + perWl * W + model + (256.0 * 1048576.0));
}
}  // namespace

bool soloRunKept(int device, const SoloParams& S, const SoloModel& M, u64 nHost, const u32* hostIdx, const RecAnn* hostAnn, SoloResult& out,
                 std::string& log, const CbQualFn& cbQual) {
    ::singlet::gpu_ledger::Phase ledger_phase("solo");
    SoloKeep& K = g_keep;
    if (K.toHost.joinable()) K.toHost.join();   // E15: the copy to the host is done (or failed: the arrays are still on the device)
    if (device >= 0 && cudaSetDevice(device) != cudaSuccess) { cudaGetLastError(); log += "soloRunKept: cannot select the device\n"; soloKeepRelease(); return false; }
    if (!K.reads || !K.recs) { log += "soloRunKept: the inputs were not kept on the device\n"; soloKeepRelease(); return false; }
    if (K.nOverflow) { log += "soloRunKept: " + std::to_string(K.nOverflow) + " records with more exons than the port's limit\n"; soloKeepRelease(); return false; }
    char lb[400];
    const u64 kN = K.n, kR = K.nRec;
    const double inBytes = kN * 20.0 + kR * (double)sizeof(RecAnn);

    // Where Solo runs. One shot on the device when its working set fits in what is free now (the kept inputs included when they are still on the
    // host) and nothing asks otherwise. In two passes over read chunks (soloRunTwoPass: chunks sized from free device memory) when it does not
    // fit, at the one-shot path's 2^31 - 2 edge, when the one-shot path runs out of memory after all, or when SINGLET_SOLO_CHUNK (reads per
    // chunk), SINGLET_SOLO_BUCKET (keys per bucket) or SINGLET_SOLO_BUCKETS (about this many buckets) is set. On the host with the same source
    // (E9: soloRun(false), OpenMP loops and parallel sorts, the same counts) when even the two-pass path cannot get its memory, when
    // SINGLET_SOLO_HOST=1 says so, or past 2^31 - 1 reads (the depth the device paths are proven at); it runs from the host copies of the kept
    // inputs. Each path writes one stderr line naming itself and the reason.
    std::string hostWhy, twoWhy, oneWhy = "the free device memory could not be read";
    u64 envChunk = 0, envBucket = 0, envBuckets = 0;
    const bool setChunk = envU64("SINGLET_SOLO_CHUNK", envChunk) && envChunk > 0;
    const bool setBucket = envU64("SINGLET_SOLO_BUCKET", envBucket) && envBucket > 0;
    const bool setBuckets = envU64("SINGLET_SOLO_BUCKETS", envBuckets) && envBuckets > 0;
    {
        size_t fr = 0, tot = 0;
        const bool gotFree = cudaMemGetInfo(&fr, &tot) == cudaSuccess;
        if (!gotFree) cudaGetLastError();
        const double need = (double)soloDeviceNeed(kN, nHost, M) + (K.onHost ? inBytes : 0.0);
        const char* fh = getenv("SINGLET_SOLO_HOST");
        if (fh && fh[0] == '1') hostWhy = "SINGLET_SOLO_HOST=1";
        else if (kN > 0x7fffffffULL) hostWhy = std::to_string(kN) + " reads are past the device paths' 2^31 - 1";
        else if (setChunk || setBucket || setBuckets) twoWhy = "SINGLET_SOLO_CHUNK, _BUCKET or _BUCKETS is set";
        else if (kN + 1 >= 0x7fffffffULL) twoWhy = std::to_string(kN) + " reads are past the one-shot path's 2^31 - 2";
        else if (gotFree && need > 0.97 * (double)fr) {
            snprintf(lb, sizeof lb, "the one-shot device path needs about %.1f GB and %.1f GB are free (%.0f M reads)", need / 1e9, fr / 1e9, kN / 1e6);
            twoWhy = lb;
        } else if (gotFree) {
            snprintf(lb, sizeof lb, "needs about %.1f GB, %.1f GB free, %.0f M reads", need / 1e9, fr / 1e9, kN / 1e6);
            oneWhy = lb;
        }
    }

    // keep the host copies of the kept inputs for the host path: the page-locked ones, or a copy made from the device ones
    auto toHostRun = [&](const std::string& why) -> bool {
        const auto t0 = std::chrono::steady_clock::now();
        const u64 n = std::max<u64>(1, kN), nR = std::max<u64>(1, kR);
        u64 *hc = K.hcr3, *hu = K.hur3; u32* hr = K.hrec; RecAnn* ha = K.hrecAnn;
        std::vector<u64> bc, bu; std::vector<u32> br; std::vector<RecAnn> ba;
        if (!K.onHost) {   // the arrays are on the device: a host copy, then the device ones go
            try { bc.resize(n); bu.resize(n); br.resize(n); ba.resize(nR); } catch (const std::exception&) {
                log += "soloRunKept: Solo needs the host path (" + why + ") and the host cannot hold the copy of the kept inputs\n";
                soloKeepRelease(); return false;
            }
            cudaError_t e = cudaMemcpy(bc.data(), K.cr3, n * 8, cudaMemcpyDeviceToHost);
            if (e == cudaSuccess) e = cudaMemcpy(bu.data(), K.ur3, n * 8, cudaMemcpyDeviceToHost);
            if (e == cudaSuccess) e = cudaMemcpy(br.data(), K.rec, n * 4, cudaMemcpyDeviceToHost);
            if (e == cudaSuccess) e = cudaMemcpy(ba.data(), K.recAnn, nR * sizeof(RecAnn), cudaMemcpyDeviceToHost);
            if (e != cudaSuccess) {
                cudaGetLastError();
                log += std::string("soloRunKept: Solo needs the host path (") + why + ") and the kept inputs could not be copied to the host: " + cudaGetErrorString(e) + "\n";
                soloKeepRelease(); return false;
            }
            hc = bc.data(); hu = bu.data(); hr = br.data(); ha = ba.data();
        }
        // free the device copies (put back for a failed device attempt, or kept there) before the host run
        if (K.cr3) cudaFree(K.cr3);
        if (K.ur3) cudaFree(K.ur3);
        if (K.rec) cudaFree(K.rec);
        if (K.recAnn) cudaFree(K.recAnn);
        K.cr3 = K.ur3 = nullptr; K.rec = nullptr; K.recAnn = nullptr;
        out = SoloResult{};
        SoloRecs R; R.nRec = kR; R.dev = false; R.recAnn = ha;
        SoloReads Rd; Rd.n = kN; Rd.dev = false; Rd.cr3 = hc; Rd.ur3 = hu; Rd.rec = hr; Rd.nHost = nHost; Rd.hostIdx = hostIdx; Rd.hostAnnS = hostAnn;
        Rd.cbQual = cbQual;
        fprintf(stderr, "[engine_gpu] Solo's counting runs on the host (%s): same result as the device path, slower\n", why.c_str());
        fflush(stderr);
        const bool ok = soloRun(false, S, M, R, Rd, out, false, log);
        out.nRecOverflow = K.nOverflow;
        snprintf(lb, sizeof lb, "soloRunKept: Solo ran on the host (%s) in %.1f s\n", why.c_str(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        log += lb;
        soloKeepRelease();
        return ok;
    };
    // the device copies of the kept inputs, when the host ones are the originals (the one-shot attempt put them back)
    auto freeDevCopies = [&]() {
        if (!K.onHost) return;
        if (K.cr3) cudaFree(K.cr3);
        if (K.ur3) cudaFree(K.ur3);
        if (K.rec) cudaFree(K.rec);
        if (K.recAnn) cudaFree(K.recAnn);
        K.cr3 = K.ur3 = nullptr; K.rec = nullptr; K.recAnn = nullptr;
    };
    // soloRunTwoPass on the kept inputs: in place in page-locked host memory (K.onHost), else on the device
    auto twoPassRun = [&](const std::string& why) -> bool {
        const auto t0 = std::chrono::steady_clock::now();
        freeDevCopies();
        const bool onH = K.onHost;
        SoloRecs R2; R2.nRec = kR; R2.dev = !onH; R2.recAnn = onH ? K.hrecAnn : K.recAnn;
        SoloReads Rd2; Rd2.n = kN; Rd2.dev = !onH; Rd2.cr3 = onH ? K.hcr3 : K.cr3; Rd2.ur3 = onH ? K.hur3 : K.ur3; Rd2.rec = onH ? K.hrec : K.rec;
        Rd2.nHost = nHost; Rd2.hostIdx = hostIdx; Rd2.hostAnnS = hostAnn; Rd2.cbQual = cbQual;
        SoloTwoPass opt; opt.chunkReads = setChunk ? envChunk : 0; opt.bucketKeys = setBucket ? envBucket : 0; opt.bucketTarget = setBuckets ? envBuckets : 0;
        const char* rh = getenv("SINGLET_SOLO_RECANN_HOST");
        opt.recAnnHost = rh && rh[0] == '1';
        out = SoloResult{};
        const bool ok = soloRunTwoPass(S, M, R2, Rd2, out, log, opt);
        if (!ok && g_lastOOM) {   // not even chunks fit: the last resort
            snprintf(lb, sizeof lb, "the two-pass device path ran out of memory (%.0f M reads; %s)", kN / 1e6, why.c_str());
            return toHostRun(lb);
        }
        out.nRecOverflow = K.nOverflow;
        if (ok) {
            fprintf(stderr, "[engine_gpu] Solo path: two passes on the device, %llu chunks, %llu buckets (%s)\n", (unsigned long long)out.nChunks,
                    (unsigned long long)out.nBuckets, why.c_str());
            fflush(stderr);
        }
        snprintf(lb, sizeof lb, "soloRunKept: Solo ran in two passes (%s) in %.1f s\n", why.c_str(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        log += lb;
        soloKeepRelease();
        return ok;
    };
    if (!hostWhy.empty()) return toHostRun(hostWhy);
    if (!twoWhy.empty()) return twoPassRun(twoWhy);

    if (K.onHost) {   // E15: back to the device
        const auto t0 = std::chrono::steady_clock::now();
        const u64 n = std::max<u64>(1, K.n), nR = std::max<u64>(1, K.nRec);
        cudaError_t e = cudaMalloc(&K.cr3, n * 8);
        if (e == cudaSuccess) e = cudaMalloc(&K.ur3, n * 8);
        if (e == cudaSuccess) e = cudaMalloc(&K.rec, n * 4);
        if (e == cudaSuccess) e = cudaMalloc(&K.recAnn, nR * sizeof(RecAnn));
        if (e == cudaSuccess) e = cudaMemcpy(K.cr3, K.hcr3, n * 8, cudaMemcpyHostToDevice);
        if (e == cudaSuccess) e = cudaMemcpy(K.ur3, K.hur3, n * 8, cudaMemcpyHostToDevice);
        if (e == cudaSuccess) e = cudaMemcpy(K.rec, K.hrec, n * 4, cudaMemcpyHostToDevice);
        if (e == cudaSuccess) e = cudaMemcpy(K.recAnn, K.hrecAnn, nR * sizeof(RecAnn), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            cudaGetLastError();
            // K.cr3 ... that were allocated are freed by toHostRun; the host copies are still there
            log += std::string("soloRunKept: the kept inputs could not be put back on the device: ") + cudaGetErrorString(e) + "\n";
            return twoPassRun(std::string("the kept inputs did not fit back on the device: ") + cudaGetErrorString(e));
        }
        K.msToDev = msSince(t0);
        snprintf(lb, sizeof lb, "soloRunKept: the kept inputs (%.2f GB) were in page-locked host memory during STAR's loop: to the host %.1f ms (thread), back %.1f ms\n",
                 inBytes / 1e9, K.msToHost, K.msToDev);
        log += lb;
    } else if (!K.hostWhy.empty()) log += "soloRunKept: the kept inputs stayed on the device during STAR's loop (" + K.hostWhy + ")\n";
    SoloRecs R; R.nRec = K.nRec; R.dev = true; R.recAnn = K.recAnn;
    SoloReads Rd; Rd.n = K.n; Rd.dev = true; Rd.cr3 = K.cr3; Rd.ur3 = K.ur3; Rd.rec = K.rec; Rd.nHost = nHost; Rd.hostIdx = hostIdx; Rd.hostAnnS = hostAnn;
    Rd.cbQual = cbQual;
    const bool ok = soloRun(true, S, M, R, Rd, out, false, log);
    if (!ok && g_lastOOM) {   // the bound was too low for this run: the same stage in two passes
        snprintf(lb, sizeof lb, "the one-shot device path ran out of memory (estimate %.1f GB, %.0f M reads)", soloDeviceNeed(kN, nHost, M) / 1e9, kN / 1e6);
        return twoPassRun(lb);
    }
    out.nRecOverflow = K.nOverflow;
    if (ok) {
        fprintf(stderr, "[engine_gpu] Solo path: one shot on the device, 1 chunk, 1 bucket (%s)\n", oneWhy.c_str());
        fflush(stderr);
    }
    soloKeepRelease();
    return ok;
}

// ================================================================================================ writers (host)
namespace {
bool writeFile(const std::string& p, const std::string& s, std::string& log) {
    FILE* f = fopen(p.c_str(), "wb");
    if (!f) { log += "cannot write " + p + "\n"; return false; }
    const bool ok = fwrite(s.data(), 1, s.size(), f) == s.size();
    fclose(f);
    if (!ok) log += "short write " + p + "\n";
    return ok;
}
// decimal as ostream prints an unsigned integer
inline void appendU(std::string& s, u64 v) {
    char b[24]; int n = 0;
    do { b[n++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (n) s += b[--n];
}
// SequenceFuns.cpp:280 convertNuclInt64toString
inline void appendCB(std::string& s, u64 v, u32 L) {
    char b[32];
    for (u32 ii = 1; ii <= L; ii++) { b[L - ii] = "ACGT"[v & 3]; v >>= 2; }
    s.append(b, L);
}
// SoloReadBarcode.cpp:44 / SoloReadFeature.cpp:56 statsOut: setw(50) name, setw(15) value
std::string statsText(const char* const* names, const u64* v, u32 n) {
    std::string s; char b[128];
    for (u32 i = 0; i < n; i++) { snprintf(b, sizeof b, "%50s%15llu\n", names[i], (unsigned long long)v[i]); s += b; }
    return s;
}
const char* const kBarNames[B_N] = {"noNoAdapter", "noNoUMI", "noNoCB", "noNinCB", "noNinUMI", "noUMIhomopolymer", "noNoWLmatch", "noTooManyMM",
                                    "noTooManyWLmatches", "yesWLmatchExact", "yesOneWLmatchWithMM", "yesMultWLmatchWithMM"};
const char* const kFeatNames[S_N] = {"noUnmapped", "noNoFeature", "MultiFeature", "subMultiFeatureMultiGenomic", "noTooManyWLmatches",
                                     "noMMtoWLwithoutExact", "yesWLmatch", "yessubWLmatchExact", "yessubWLmatch_UniqueFeature", "yesCellBarcodes", "yesUMIs"};
const char* const kFlagNames[F_N] = {"cbMatch", "cbPerfect", "cbMMunique", "cbMMmultiple", "genomeU", "genomeM", "featureU", "featureM", "exonic",
                                     "intronic", "exonicAS", "intronicAS", "mito", "countedU", "countedM"};
}  // namespace

bool soloWrite(const std::string& pre, const SoloParams& S, const SoloModel& M, const SoloResult& r, const SoloFilterParams& FP,
               const std::vector<u32>& order, SoloWriteInfo& info, std::string& log, const SoloSummaryIn* sum) {
    const u32 W = (u32)M.wl.size();
    const u64 nP = r.cbIdx.size();
    const std::string g = pre + "Gene/";
    mkdir(pre.c_str(), 0755); mkdir(g.c_str(), 0755); mkdir((g + "raw").c_str(), 0755); mkdir((g + "filtered").c_str(), 0755);
    bool ok = true;
    // Barcodes.stats (Solo.cpp:62), Features.stats (SoloFeature_processRecords.cpp:67)
    ok &= writeFile(pre + "Barcodes.stats", statsText(kBarNames, r.bstats, B_N), log);
    ok &= writeFile(g + "Features.stats", statsText(kFeatNames, r.fstats, S_N), log);
    // the detected CBs (SoloFeature_sumThreads.cpp:82-98): cbReadCount > 0, in whitelist order; per CB its segments (both in CB order)
    std::vector<u32> indCB;   // index into cbIdx
    for (u64 k = 0; k < nP; k++) if (r.cbVal[k * CBV + CV_cbReadCount] > 0) indCB.push_back((u32)k);
    const u32 nCB = (u32)indCB.size();
    info.nCB = nCB;
    std::vector<u64> segB(nCB + 1, 0);
    {
        u64 s = 0;
        for (u32 icb = 0; icb < nCB; icb++) {
            const u32 w = r.cbIdx[indCB[icb]];
            while (s < r.segCb.size() && r.segCb[s] < w) s++;
            segB[icb] = s;
            while (s < r.segCb.size() && r.segCb[s] == w) s++;
            if (icb + 1 == nCB) segB[nCB] = s;
        }
        if (s != r.segCb.size()) log += "soloWrite: matrix entries for a CB without reads\n";
    }
    auto segE = [&](u32 icb) { u64 s = segB[icb]; const u32 w = r.cbIdx[indCB[icb]]; while (s < r.segCb.size() && r.segCb[s] == w) s++; return s; };
    // features.tsv (SoloFeature_outputResults.cpp:26-41; soloOutFormatFeaturesGeneField3 "Gene Expression")
    std::string feat;
    for (u32 i = 0; i < S.nGe; i++) { feat += M.geID[i]; feat += '\t'; feat += M.geName[i].empty() ? M.geID[i] : M.geName[i]; feat += "\tGene Expression\n"; }
    auto matrix = [&](bool filt, const std::vector<char>& keep, u64 nCols) {
        u64 nEnt = 0;
        for (u32 icb = 0; icb < nCB; icb++) if (!filt || keep[icb]) nEnt += r.cbVal[(u64)indCB[icb] * CBV + CV_nGene];
        std::string s = "%%MatrixMarket matrix coordinate integer general\n%\n";
        appendU(s, S.nGe); s += ' '; appendU(s, nCols); s += ' '; appendU(s, nEnt); s += '\n';
        s.reserve(s.size() + nEnt * 20);
        u32 col = 0;
        for (u32 icb = 0; icb < nCB; icb++) {
            if (filt) { if (!keep[icb]) continue; ++col; } else col = r.cbIdx[indCB[icb]] + 1;
            const u64 e = segE(icb);
            for (u64 k = segB[icb]; k < e; k++) { appendU(s, (u64)r.segGene[k] + 1); s += ' '; appendU(s, col); s += ' '; appendU(s, r.segCount[k]); s += '\n'; }
        }
        return s;
    };
    // raw (outputResults(false)): every whitelist barcode
    {
        std::string bc; bc.reserve((u64)W * (S.cbL + 1));
        for (u32 w = 0; w < W; w++) { appendCB(bc, M.wl[w], S.cbL); bc += '\n'; }
        ok &= writeFile(g + "raw/features.tsv", feat, log);
        ok &= writeFile(g + "raw/barcodes.tsv", bc, log);
        ok &= writeFile(g + "raw/matrix.mtx", matrix(false, {}, W), log);
    }
    // CellRanger2.2 (SoloFeature_cellFiltering.cpp:37-66)
    std::vector<u32> nUMIperCB(nCB);
    for (u32 icb = 0; icb < nCB; icb++) nUMIperCB[icb] = r.cbVal[(u64)indCB[icb] * CBV + CV_nUMI];
    std::vector<u32> sorted = nUMIperCB;
    std::sort(sorted.begin(), sorted.end(), [](const u32& a, const u32& b) { return a > b; });
    std::vector<char> keep(nCB, 0);
    u64 nCells = 0;
    if (nCB >= 1) {
        const u32 maxind = (u32)int(round(FP.nExpectedCells * (1.0 - FP.maxPercentile)));
        const u32 nUMImax = sorted[std::min(nCB - 1, maxind)];
        u32 nUMImin = (u32)int(round(nUMImax / FP.maxMinRatio));
        nUMImin = std::max(nUMImin, (u32)1);
        info.nUMImax = nUMImax; info.nUMImin = nUMImin;
        for (u32 icb = 0; icb < nCB; icb++) if (nUMIperCB[icb] >= nUMImin) { keep[icb] = 1; nCells++; }
    }
    info.nCells = nCells;
    if (nCells > 0) {   // cellFiltering returns before outputResults(true) when no cell passes
        std::string bc;
        for (u32 icb = 0; icb < nCB; icb++) if (keep[icb]) { appendCB(bc, M.wl[r.cbIdx[indCB[icb]]], S.cbL); bc += '\n'; }
        ok &= writeFile(g + "filtered/features.tsv", feat, log);
        ok &= writeFile(g + "filtered/barcodes.tsv", bc, log);
        ok &= writeFile(g + "filtered/matrix.mtx", matrix(true, keep, nCells), log);
    }
    // UMIperCellSorted.txt (SoloFeature_statsOutput.cpp:73-80)
    {
        std::string s;
        for (u32 n : sorted) { if (n == 0) break; appendU(s, n); s += '\n'; }
        ok &= writeFile(g + "UMIperCellSorted.txt", s, log);
    }
    // CellReads.stats (SoloFeature_statsOutput.cpp:84-116): rows in STAR's unordered_map order, given the keys' insertion order
    {
        std::string s = "CB";
        for (u32 b = 0; b < F_N; b++) { s += '\t'; s += kFlagNames[b]; }
        s += "\tnUMIunique\tnGenesUnique\tnUMImulti\tnGenesMulti\n";
        s += "CBnotInPasslist";
        for (u32 b = 0; b < F_N; b++) { s += '\t'; appendU(s, r.noCB[b]); }
        s += "\t0\t0\t0\t0\n";
        std::vector<u32> rows;   // indices into cbIdx
        for (u64 k = 0; k < nP; k++) if (r.cbVal[k * CBV + CV_flags + F_cbMatch] > 0) rows.push_back((u32)k);
        if (!order.empty()) {
            const u64 nRows = rows.size();
            std::unordered_map<u64, u32> m;
            m.reserve(nCB * 3 / 2);   // SoloFeature_countCBgeneUMI.cpp:43 (uint32 arithmetic)
            for (u32 k : order) m.insert({(u64)k, 0u});
            rows.clear();
            for (auto& kv : m) {
                const auto it = std::lower_bound(r.cbIdx.begin(), r.cbIdx.end(), (u32)kv.first);
                if (it == r.cbIdx.end() || *it != (u32)kv.first) { log += "CellReads.stats: a key without a row\n"; continue; }
                rows.push_back((u32)(it - r.cbIdx.begin()));
            }
            if (rows.size() != nRows) {
                char b[160]; snprintf(b, sizeof b, "CellReads.stats: %zu keys in the insertion order, %llu CBs with rows\n", rows.size(), (unsigned long long)nRows); log += b;
            }
        }
        s.reserve(s.size() + rows.size() * 80);
        for (u32 k : rows) {
            const u32* v = &r.cbVal[(u64)k * CBV];
            appendCB(s, M.wl[r.cbIdx[k]], S.cbL);
            for (u32 b = 0; b < F_N; b++) { s += '\t'; appendU(s, v[CV_flags + b]); }
            s += '\t'; appendU(s, v[CV_nUMI]); s += '\t'; appendU(s, v[CV_nGene]); s += "\t0\t0\n";
        }
        ok &= writeFile(g + "CellReads.stats", s, log);
    }
    // Summary.csv (SoloFeature_statsOutput.cpp:7-74; the cell numbers of cellFiltering.cpp:77-125). Doubles as ostream prints them (%g, 6 digits);
    // the means are uint64 divisions (SoloFilteredCells.h); Unique multimappers only, so the Unique+Multiple gene line is "NoMulti".
    if (sum != nullptr) {
        std::string s;
        auto num = [&](const char* k, u64 v) { s += k; s += ','; appendU(s, v); s += '\n'; };
        auto dbl = [&](const char* k, double v) { char b[64]; snprintf(b, sizeof b, "%g", v); s += k; s += ','; s += b; s += '\n'; };
        u64 nInvB = 0;
        for (u32 i = 0; i < 9; i++) nInvB += r.bstats[i];   // SoloReadBarcodeStats::numInvalidBarcodes: the first nine rows
        const u64 nInvF = r.fstats[S_noTooManyWLmatches] + r.fstats[S_noMMtoWLwithoutExact];
        const u64 uniqF = r.fstats[S_yessubWLmatch_UniqueFeature];
        num("Number of Reads", sum->readN);
        dbl("Reads With Valid Barcodes", 1.0 - double(nInvB + nInvF) / sum->readN);
        dbl("Sequencing Saturation", 1.0 - double(r.fstats[S_yesUMIs]) / uniqF);
        dbl("Q30 Bases in CB+UMI", double(sum->q30CbUmi) / sum->nCbUmi);
        dbl("Q30 Bases in RNA read", double(sum->q30Rna) / sum->nRna);
        dbl("Reads Mapped to Genome: Unique+Multiple", double(sum->mappedU + sum->mappedM) / sum->readN);
        dbl("Reads Mapped to Genome: Unique", double(sum->mappedU) / sum->readN);
        s += "Reads Mapped to Gene: Unique+Multiple Gene,NoMulti\n";
        dbl("Reads Mapped to Gene: Unique Gene", double(uniqF) / sum->readN);
        // filtered cells (nCells == 0: cellFiltering returns with every statistic 0)
        u64 nCellsF = 0, nReadIn = 0, nUMIin = 0, nGeneIn = 0, medRead = 0, medUMI = 0, medGene = 0, nGeneDet = 0;
        if (nCells > 0) {
            std::vector<u32> rd, gn;
            std::vector<char> det(S.nGe, 0);
            for (u32 icb = 0; icb < nCB; icb++) {
                if (!keep[icb]) continue;
                const u32* v = &r.cbVal[(u64)indCB[icb] * CBV];
                nCellsF++; nUMIin += v[CV_nUMI]; nReadIn += v[CV_nReadUnique];
                rd.push_back(v[CV_nReadUnique]);
                const u64 e = segE(icb);
                u32 ng1 = 0;   // cellFiltering.cpp:91-99: the cell's genes with a count > 0
                for (u64 k = segB[icb]; k < e; k++)
                    if (r.segCount[k] > 0) { ng1++; if (r.segGene[k] < S.nGe) det[r.segGene[k]] = 1; }
                nGeneIn += ng1; gn.push_back(ng1);
            }
            std::sort(rd.begin(), rd.end()); std::sort(gn.begin(), gn.end());
            medRead = rd[nCellsF / 2]; medGene = gn[nCellsF / 2]; medUMI = sorted[nCellsF / 2];
            for (char d : det) nGeneDet += d != 0;
        }
        num("Estimated Number of Cells", nCellsF);
        num("Unique Reads in Cells Mapped to Gene", nReadIn);
        dbl("Fraction of Unique Reads in Cells", double(nReadIn) / uniqF);
        num("Mean Reads per Cell", nCellsF ? nReadIn / nCellsF : 0);
        num("Median Reads per Cell", medRead);
        num("UMIs in Cells", nUMIin);
        num("Mean UMI per Cell", nCellsF ? nUMIin / nCellsF : 0);
        num("Median UMI per Cell", medUMI);
        num("Mean Gene per Cell", nCellsF ? nGeneIn / nCellsF : 0);
        num("Median Gene per Cell", medGene);
        num("Total Gene Detected", nGeneDet);
        ok &= writeFile(g + "Summary.csv", s, log);
    }
    return ok;
}

}  // namespace e9
