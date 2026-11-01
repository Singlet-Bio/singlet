// E9: STARsolo's Gene outputs on the GPU (or the host, same source). See solo_gpu.h and docs/gpu-engine/fable/E9-solo.md.
#include "solo_gpu.h"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <parallel/algorithm>
#include <stdexcept>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>

namespace e9 {
namespace {

inline void ck(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + what + " (solo_gpu.cu:" + std::to_string(line) + "): " + cudaGetErrorString(e));
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
        if (dev) CK(cudaMalloc(&p, b));
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
    void* need(size_t b) { if (b > n) { if (p) cudaFree(p); p = nullptr; CK(cudaMalloc(&p, std::max<size_t>(b, 1))); n = b; } return p; }
};

void sortKeys(bool dev, Buf<u64>& k, u64 n, int endBit, Tmp& tmp) {
    if (n < 2) return;
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

}  // namespace

bool soloRun(bool dev, const SoloParams& S0, const SoloModel& M, const SoloRecs& R, const SoloReads& Rd, SoloResult& out, bool keep, std::string& log) {
    try {
        using clk = std::chrono::steady_clock;
        auto t0 = clk::now();
        auto lap = [&](const char* name) {
            if (dev) CK(cudaDeviceSynchronize());
            const auto t = clk::now();
            out.ms.push_back({name, std::chrono::duration<double, std::milli>(t - t0).count()});
            t0 = t;
        };
        SoloParams S = S0;
        const u32 W = (u32)M.wl.size();
        S.wlN = W;
        const u64 N = Rd.n, nRec = R.nRec;
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
        Buf<u32> mmIdx; Buf<i32> mmCb; u64 nMM = 0;
        {
            Buf<u32> fl; fl.alloc(dev, N + 1);
            const int8_t* cbmp = cbmR.p; u32* f = fl.p; const u64 n = N;
            each(dev, N + 1, [=] __host__ __device__(u64 i) { f[i] = (i < n && cbmp[i] > 1) ? 1u : 0u; });
            Buf<u32> fo;
            nMM = exclusiveScan(dev, fl, N, fo, tmp);
            mmIdx.alloc(dev, nMM);
            const u32* fp = fl.p; const u32* op = fo.p; u32* mi = mmIdx.p;
            each(dev, N, [=] __host__ __device__(u64 i) { if (fp[i]) mi[op[i]] = (u32)i; });
        }
        out.nMulti = nMM;
        out.perReadQual = (bool)Rd.cbQual;
        std::vector<uint8_t> hQual;   // host mode: mmQ aliases it
        Buf<uint8_t> mmQ; Buf<double> dPq;
        if (Rd.cbQual && nMM > 0) {
            if (M.pqTab.size() != 256) throw std::runtime_error("per-read CB qualities without the quality table");
            std::vector<u32> hIdx(nMM);
            mmIdx.to(hIdx.data(), nMM);
            Buf<u64> mmCr; mmCr.alloc(dev, nMM);   // their CRs, for the caller's check that it read their lines
            {
                const u32* mi = mmIdx.p; const u32* ci = crIdx.p; const u64* cr = dcr.p; u64* o = mmCr.p;
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
            const u32* mi = mmIdx.p; const u32* ci = crIdx.p; const u64* cr = dcr.p; const u32* co = cOff.p; const u32* nc = nC.p; const u32* cd = cand.p;
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
            const u32* nc = nC.p; const u32* cd = cand.p; const i32* ch = chosen.p; u64* ky = keys.p; i32* rc = rowCb.p; u32* fr = flagRows.p;
            u64* fs = fst.p; u64* nb = noCB.p; u32* crc = cbRC.p; u32* nru = nRU.p; u64* c = cnt.p;
            const u32* mi = mmIdx.p; const i32* mc = mmCb.p; const u64 nmm = nMM;
            each(dev, N, [=] __host__ __device__(u64 i) {
                const RecAnn a = ann[i];
                const i32 cbm = cbmp[i];
                u32 flag = recordFlag(a, cbm);
                ky[i] = ~0ULL; rc[i] = -1;
                if (cbm < 0) {   // record(): no CB match, counted as CBnotInPasslist, nothing written
                    for (u32 b = 0; b < F_N; b++) if ((flag >> b) & 1) incAgg(nb, b);
                    return;
                }
                bool feat = false;
                if (a.nTr == 0) incAgg(fs, S_noUnmapped);
                else if (a.ng == 0) incAgg(fs, S_noNoFeature);
                else if (a.ng > 1) { incAgg(fs, S_MultiFeature); if (a.nTr > 1) incAgg(fs, S_subMultiFeatureMultiGenomic); }
                else feat = true;
                const u32 d = ci[i];
                if (feat) for (u32 k = 0; k < nc[d]; k++) addU32(&crc[cd[co[d] + k]], 1);   // record(): cbReadCount for every match
                // inputRecords(); a read with several matches: its own posterior (E15)
                const i32 cb = cbm > 1 ? mc[lowerBound(mi, nmm, (u32)i)] : ch[d];
                const bool accepted = cb >= 0;
                if (feat) {
                    if (cbm == 0) incAgg(fs, S_yessubWLmatchExact);
                    else if (!accepted) incAgg(fs, S_noTooManyWLmatches);
                }
                if (feat && accepted) {
                    addU32(&nru[cb], 1);
                    ky[i] = tupleKey((u64)cb, a.gene, um[i]);
                    flag |= 1u << F_countedU;
                    incAgg(c, 1);
                }
                flag |= 1u << F_cbMatch;
                if (cbm == 0) flag |= 1u << F_cbPerfect;
                else if (cbm == 1) flag |= 1u << F_cbMMunique;
                else if (accepted) flag |= 1u << F_cbMMmultiple;
                if (accepted) {
                    rc[i] = cb;
                    for (u32 b = 0; b < F_N; b++) if ((flag >> b) & 1) addU32(&fr[(u64)cb * F_N + b], 1);
                } else {
                    for (u32 b = 0; b < F_N; b++) if ((flag >> b) & 1) incAgg(nb, b);
                }
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
                const int8_t* cbmp = cbmR.p; const u32* ci = crIdx.p; const i32* ch = chosen.p; const u32* mi = mmIdx.p; const i32* mc = mmCb.p;
                const u64 nmm = nMM; i32* o = c0.p;
                each(dev, N, [=] __host__ __device__(u64 i) {
                    const i32 cbm = cbmp[i];
                    o[i] = cbm < 0 ? -1 : cbm > 1 ? mc[lowerBound(mi, nmm, (u32)i)] : ch[ci[i]];
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
        return true;
    } catch (const std::exception& e) {
        log += std::string("soloRun failed: ") + e.what() + "\n";
        if (dev) cudaGetLastError();
        return false;
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
bool soloRunKept(int device, const SoloParams& S, const SoloModel& M, u64 nHost, const u32* hostIdx, const RecAnn* hostAnn, SoloResult& out,
                 std::string& log, const CbQualFn& cbQual) {
    SoloKeep& K = g_keep;
    if (K.toHost.joinable()) K.toHost.join();   // E15: the copy to the host is done (or failed: the arrays are still on the device)
    if (device >= 0 && cudaSetDevice(device) != cudaSuccess) { cudaGetLastError(); log += "soloRunKept: cannot select the device\n"; soloKeepRelease(); return false; }
    if (!K.reads || !K.recs) { log += "soloRunKept: the inputs were not kept on the device\n"; soloKeepRelease(); return false; }
    if (K.nOverflow) { log += "soloRunKept: " + std::to_string(K.nOverflow) + " records with more exons than the port's limit\n"; soloKeepRelease(); return false; }
    char lb[200];
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
            log += std::string("soloRunKept: the kept inputs could not be put back on the device: ") + cudaGetErrorString(e) + "\n";
            soloKeepRelease();
            return false;
        }
        K.msToDev = msSince(t0);
        snprintf(lb, sizeof lb, "soloRunKept: the kept inputs (%.2f GB) were in page-locked host memory during STAR's loop: to the host %.1f ms (thread), back %.1f ms\n",
                 (K.n * 20.0 + K.nRec * 8.0) / 1e9, K.msToHost, K.msToDev);
        log += lb;
    } else if (!K.hostWhy.empty()) log += "soloRunKept: the kept inputs stayed on the device during STAR's loop (" + K.hostWhy + ")\n";
    SoloRecs R; R.nRec = K.nRec; R.dev = true; R.recAnn = K.recAnn;
    SoloReads Rd; Rd.n = K.n; Rd.dev = true; Rd.cr3 = K.cr3; Rd.ur3 = K.ur3; Rd.rec = K.rec; Rd.nHost = nHost; Rd.hostIdx = hostIdx; Rd.hostAnnS = hostAnn;
    Rd.cbQual = cbQual;
    const bool ok = soloRun(true, S, M, R, Rd, out, false, log);
    out.nRecOverflow = K.nOverflow;
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
