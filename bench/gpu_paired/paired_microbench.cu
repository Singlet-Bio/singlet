// Phase 1d microbenchmark and check harness: DRAFT, NOT COMPILED, NOT RUN. W11 builds it on an H100 (README steps in
// docs/research-beta/paired-end-prep.md section 7). Standalone: it loads a STAR genome directory itself, so it does not touch
// engine_gpu.cu or the singlet binary.
//
// Two modes.
//   --unit fixtures/cases.txt
//        The handcrafted cases of test_paired_model.py. Each case starts from the windows the Python model built and runs the
//        kernel stage under test (mapFinishPE: stitching of the windows with the mate rules, multMapSelect, mappedFilter) on the
//        host and on the device, and compares both with the model's result. No suffix array is needed.
//   --genome DIR --r1 R1.fastq --r2 R2.fastq [--n N] [--mode spliced|atac] [--batch B] [--repeat K] [--out kernel.tsv] [--dump win.bin]
//        A sample of real pairs through the whole kernel (seeding, windows, stitching, selection). Writes one TSV line per pair
//        (compare_star_sam.py compares it with a STAR run on the same pairs), the window dump (replay_model.py replays it through
//        the Python model), and times the kernel (pairs per second, status counts).
//
// Build: bench/gpu_paired/build.sh. The genome must have been built without a GTF (no sjdb): the kernel stage reuses stitch.cuh's
// sjdb code, but this loader does not load the sjdb arrays.
#include "paired_harness_common.h"

#include <cuda_runtime.h>

#include <chrono>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

// ---------------------------------------------------------------------------------------------------------------------
// Device copies of the genome (cudaMalloc; the human genome is ~3 GB G + ~25 GB SA + ~1.5 GB SAi: fits an 80 GB H100)
// ---------------------------------------------------------------------------------------------------------------------
template <class T> static T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    CK(cudaMalloc(&d, std::max<size_t>(1, v.size() * sizeof(T))));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
static GenomeView deviceView(const HostGenome& H) {
    GenomeView D = H.v;
    char* g = nullptr;
    CK(cudaMalloc(&g, H.Gbuf.size())); CK(cudaMemcpy(g, H.Gbuf.data(), H.Gbuf.size(), cudaMemcpyHostToDevice));
    D.G = g + 200;
    D.SA = H.SA.empty() ? nullptr : upload(H.SA);
    D.SAi = H.SAi.empty() ? nullptr : upload(H.SAi);
    D.chrStart = upload(H.chrStart); D.chrLength = upload(H.chrLength); D.chrBin = upload(H.chrBin);
    return D;
}

__global__ void k_unit(Params P, GenomeView g, WorkPE<PeLane>* w, RBuf* rb, uint8_t* blobBuf, int cap, int* status) {
    m5::Blob blob; blob.p = blobBuf; blob.n = 0; blob.cap = cap; blob.full = false;
    *status = (int)m5::pe::mapFinishPE(P, g, *w, *rb, blob);
}

static int runUnit(const std::string& file) {
    std::map<std::string, std::string> chroms; std::vector<std::string> order; u64 gbits = 18; std::vector<Case> cases;
    if (!parseCases(file, chroms, order, gbits, cases)) { fprintf(stderr, "cannot read %s\n", file.c_str()); return 1; }
    HostGenome H; H.gChrBinNbits = gbits;
    u64 step = 1ULL << gbits, pos = 0;
    for (auto& n : order) { H.chrStart.push_back(pos); H.chrLength.push_back(chroms[n].size()); pos += ((chroms[n].size() + step) / step) * step; }
    H.chrStart.push_back(pos);
    H.Gbuf.assign(pos + 400, 5);
    for (size_t i = 0; i < order.size(); i++) for (size_t j = 0; j < chroms[order[i]].size(); j++) H.Gbuf[200 + H.chrStart[i] + j] = nuc(chroms[order[i]][j]);
    H.v.nGenome = pos;
    H.fillChrBin(); H.finishView();
    GenomeView D = deviceView(H);
    auto w = std::make_unique<WorkPE<PeLane>>(); auto rb = std::make_unique<RBuf>();
    WorkPE<PeLane>* dW; RBuf* dR; uint8_t* dB; int* dS;
    CK(cudaMalloc(&dW, sizeof(*w))); CK(cudaMalloc(&dR, sizeof(*rb))); CK(cudaMalloc(&dB, 1 << 16)); CK(cudaMalloc(&dS, sizeof(int)));
    int bad = 0;
    for (auto& c : cases) {
        Params P = defaultParams(); applyKind(P, c.kind);
        for (auto& kv : c.ov) if (!setParam(P, kv.first, kv.second)) { fprintf(stderr, "unknown override %s\n", kv.first.c_str()); return 1; }
        std::string why;
        if (!finishParams(P, pos, gbits, why)) { fprintf(stderr, "params: %s\n", why.c_str()); return 1; }
        // host
        setupWork(*w, *rb, P, c);
        std::vector<uint8_t> hb(1 << 16); m5::Blob blob; blob.p = hb.data(); blob.n = 0; blob.cap = (int)hb.size(); blob.full = false;
        m5::Status sh = m5::pe::mapFinishPE(P, H.v, *w, *rb, blob);
        bool okHost = sh == m5::Status::Ok && matches(*w, c);
        std::string dh = describe(*w);
        // device
        setupWork(*w, *rb, P, c);
        CK(cudaMemcpy(dW, w.get(), sizeof(*w), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dR, rb.get(), sizeof(*rb), cudaMemcpyHostToDevice));
        k_unit<<<1, 1>>>(P, D, dW, dR, dB, 1 << 16, dS);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        int sd = -1; CK(cudaMemcpy(&sd, dS, sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(w.get(), dW, sizeof(*w), cudaMemcpyDeviceToHost));
        bool okDev = sd == (int)m5::Status::Ok && matches(*w, c);
        printf("%-24s host %s  device %s\n", c.name.c_str(), okHost ? "ok" : "DIFF", okDev ? "ok" : "DIFF");
        if (!okHost) printf("  host:   %s\n", dh.c_str());
        if (!okDev) printf("  device: %s\n", describe(*w).c_str());
        bad += !okHost + !okDev;
    }
    printf("%d of %zu runs differ from the model\n", bad, 2 * cases.size());
    return bad != 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// Real pairs
// ---------------------------------------------------------------------------------------------------------------------
struct Fq { std::vector<char> seq; std::vector<u64> off{0}; };
static void readFastq(const std::string& f, u64 n, Fq& q) {
    std::ifstream in(f);
    if (!in) { fprintf(stderr, "cannot open %s\n", f.c_str()); exit(1); }
    std::string h, s, p, ql;
    while (q.off.size() <= n && std::getline(in, h) && std::getline(in, s) && std::getline(in, p) && std::getline(in, ql)) {
        while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
        for (char ch : s) q.seq.push_back(nuc((char)std::toupper(ch)));
        q.off.push_back(q.seq.size());
    }
}

struct Rd {
    const uint8_t* p; size_t n, i = 0;
    i64 get() { u64 u = 0; int sh = 0; while (i < n) { uint8_t b = p[i++]; u |= (u64)(b & 0x7f) << sh; if (!(b & 0x80)) break; sh += 7; } return (i64)(u >> 1) ^ -(i64)(u & 1); }
};

// m5::encodeTr's layout (stitch.cuh:1196). Writes the primary transcript's columns.
static void decodeBlob(const uint8_t* b, size_t n, std::ostream& o) {
    Rd r{b, n};
    i64 nW = r.get(), nTr = r.get(), iBest = r.get();
    if (nW == 0) r.get();   // trInitRLength
    o << nW << '\t' << nTr << '\t' << iBest;
    for (i64 t = 0; t < nTr; t++) {
        i64 nEx = r.get();
        i64 sc[29]; for (int k = 0; k < 29; k++) sc[k] = r.get();   // rStart roStart rLength gStart gLength cStart Chr Str roStr nMatch nMM mappedLength extendL ... primaryFlag
        std::ostringstream ex, sj;
        for (i64 e = 0; e < nEx; e++) {
            i64 R = r.get(), G = r.get(), L = r.get(), F = r.get(); r.get();   // EX_R EX_G EX_L EX_iFrag EX_sjA
            ex << (e ? ";" : "") << R << ':' << G << ':' << L << ':' << F;
            if (e + 1 < nEx) { i64 c = r.get(); r.get(); r.get(); if (c >= 0) { r.get(); r.get(); } sj << (e ? "," : "") << c; }
        }
        if (t == iBest) o << '\t' << sc[6] << '\t' << sc[7] << '\t' << sc[24] << '\t' << sc[10] << '\t' << sc[9] << '\t' << sc[4] << '\t' << sc[25] << '\t' << nEx << '\t' << ex.str() << '\t' << sj.str();
    }
    if (iBest < 0) o << "\t.\t.\t.\t.\t.\t.\t.\t.\t.\t.";
}

static int runReal(const std::string& gdir, const std::string& r1, const std::string& r2, u64 N, const std::string& mode, u64 batch, int repeat,
                   const std::string& outF, const std::string& dumpF, int device) {
    CK(cudaSetDevice(device));
    HostGenome H; Params P = defaultParams(); std::string why;
    if (!loadStarGenome(gdir, H, P, why)) { fprintf(stderr, "genome: %s\n", why.c_str()); return 1; }
    applyKind(P, mode);
    if (!finishParams(P, H.v.nGenome, H.gChrBinNbits, why)) { fprintf(stderr, "params: %s\n", why.c_str()); return 1; }
    Fq a, b; readFastq(r1, N, a); readFastq(r2, N, b);
    u64 n = std::min(a.off.size(), b.off.size()) - 1;
    fprintf(stderr, "%llu pairs, mode %s: winBinNbits %llu winAnchorDistNbins %llu winFlankNbins %llu\n", (unsigned long long)n, mode.c_str(),
            (unsigned long long)P.winBinNbits, (unsigned long long)P.winAnchorDistNbins, (unsigned long long)P.winFlankNbins);
    GenomeView D = deviceView(H);
    const u64 blobCap = 1 << 13, dumpCap = 1 << 12;
    char *dM1, *dM2; u64 *dO1, *dO2;
    CK(cudaMalloc(&dM1, a.seq.size())); CK(cudaMemcpy(dM1, a.seq.data(), a.seq.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dM2, b.seq.size())); CK(cudaMemcpy(dM2, b.seq.data(), b.seq.size(), cudaMemcpyHostToDevice));
    dO1 = upload(a.off); dO2 = upload(b.off);
    WorkPE<PeLane>* dW; RBuf* dR; uint8_t *dB, *dSt, *dU; u64 *dBL, *dDump, *dDL;
    CK(cudaMalloc(&dW, batch * sizeof(WorkPE<PeLane>))); CK(cudaMalloc(&dR, batch * sizeof(RBuf)));
    CK(cudaMalloc(&dB, batch * blobCap)); CK(cudaMalloc(&dBL, batch * 8)); CK(cudaMalloc(&dDump, batch * dumpCap * 8)); CK(cudaMalloc(&dDL, batch * 8));
    CK(cudaMalloc(&dSt, batch)); CK(cudaMalloc(&dU, batch));
    std::ofstream out(outF), dump(dumpF, std::ios::binary);
    out << "idx\tstatus\tunmap\tnW\tnTr\tiBest\tChr\tStr\tscore\tnMM\tnMatch\tgLength\tiFrag\tnExons\texons(R:G:L:mate)\tcanonSJ\n";
    u64 cnt[4] = {0, 0, 0, 0}, nUnique = 0, nUnm = 0; double kernelMs = 0; u64 kernelPairs = 0;
    std::vector<uint8_t> hB(batch * blobCap), hSt(batch), hU(batch); std::vector<u64> hBL(batch), hDump(batch * dumpCap), hDL(batch);
    for (int rep = 0; rep < repeat; rep++) {
        for (u64 b0 = 0; b0 < n; b0 += batch) {
            u64 nb = std::min(batch, n - b0);
            // the offsets are absolute positions in the whole sequence buffer: shifting the offset pointer by b0 selects the batch
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0);
            m5::pe::k_pe_map<PeLane><<<(unsigned)((nb + 63) / 64), 64>>>(P, D, dM1, dO1 + b0, dM2, dO2 + b0, nb, dW, dR, dB, blobCap, dBL, dDump, dDump ? dumpCap : 0, dDL, dSt, dU);
            cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; cudaEventElapsedTime(&ms, e0, e1); kernelMs += ms; kernelPairs += nb;
            if (rep != 0) continue;   // results from the first pass only
            CK(cudaMemcpy(hB.data(), dB, nb * blobCap, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hBL.data(), dBL, nb * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hSt.data(), dSt, nb, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hU.data(), dU, nb, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hDump.data(), dDump, nb * dumpCap * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hDL.data(), dDL, nb * 8, cudaMemcpyDeviceToHost));
            for (u64 i = 0; i < nb; i++) {
                cnt[std::min<int>(3, hSt[i])]++;
                out << (b0 + i) << '\t' << (int)hSt[i] << '\t' << (int)hU[i] << '\t';
                if (hSt[i] == 0) { decodeBlob(hB.data() + i * blobCap, hBL[i], out); }
                out << '\n';
                if (hSt[i] == 0 && hU[i] == 255) nUnm++;
                u64 len = hDL[i]; dump.write((const char*)&len, 8); dump.write((const char*)(hDump.data() + i * dumpCap), len * 8);
            }
        }
    }
    fprintf(stderr, "status: ok %llu, overflow %llu (host lane), blob full %llu, other %llu\n", (unsigned long long)cnt[0], (unsigned long long)cnt[1],
            (unsigned long long)cnt[2], (unsigned long long)cnt[3]);
    fprintf(stderr, "kernel: %.1f ms for %llu pair-passes = %.0f pairs/s\n", kernelMs, (unsigned long long)kernelPairs, kernelPairs / (kernelMs / 1000.0));
    (void)nUnique; (void)nUnm;
    return 0;
}

int main(int argc, char** argv) {
    std::string unit, gdir, r1, r2, mode = "spliced", outF = "kernel.tsv", dumpF = "win.bin";
    u64 N = 100000, batch = 4096; int repeat = 3, device = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if (a == "--unit") unit = next(); else if (a == "--genome") gdir = next(); else if (a == "--r1") r1 = next(); else if (a == "--r2") r2 = next();
        else if (a == "--n") N = std::strtoull(next().c_str(), nullptr, 10); else if (a == "--mode") mode = next();
        else if (a == "--batch") batch = std::strtoull(next().c_str(), nullptr, 10); else if (a == "--repeat") repeat = atoi(next().c_str());
        else if (a == "--out") outF = next(); else if (a == "--dump") dumpF = next(); else if (a == "--device") device = atoi(next().c_str());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (!unit.empty()) return runUnit(unit);
    if (gdir.empty() || r1.empty() || r2.empty()) { fprintf(stderr, "usage: --unit cases.txt | --genome DIR --r1 f --r2 f [--n N --mode spliced|atac --batch B --repeat K --out f --dump f --device d]\n"); return 1; }
    return runReal(gdir, r1, r2, N, mode, batch, repeat, outF, dumpF, device);
}
