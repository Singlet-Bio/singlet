// E6 driver: the GPU record builder (src/engine_gpu/records_gpu.cu) against the engine's own hand-off of the same run.
// docs/gpu-engine/fable/E6-records.md.
//
// Inputs (all from one `singlet --engine gpu` run with SINGLET_GPU_PILEUP=on, kept in job /tmp):
//   --dump   SINGLET_ENGINE_GPU_DUMP of mapDistinct's output (sequences, blobs, readRecord, filter parameters, chrStart; engine_gpu.cu)
//   --handoff the SGHOFF02 file STAR's prefill wrote (ReadAlign::seqCacheAlignRecords per record), --reads its .reads (HandoffReadEntry per
//            read, written by STAR's mapping threads), --q its .reads.q (low-quality masks), --barcodes the run's barcode set (auto_barcodes.tsv)
// Checks:
//   hop 6: every record's hand-off bytes (GPU) vs the file, byte for byte; SigRec / AlnRec / ops / seq (GPU) vs Collector::prepend_handoff's
//          arrays from the file (the bridge's own code, include/singlet/pileup/gpu_pileup_bridge.h); the same records on the host (records.cuh on
//          OpenMP threads) vs the GPU.
//   hop 7: ReadRec + lowq (GPU) vs a serial host transcription of gpu_handoff_pass_entries (exact CR, no R11 correction: see the report).
// Timing: kernels (CUDA events), reps; a strided subset of ~2 M records; the host port on N threads.
#define SINGLET_HAVE_ENGINE_GPU 1
#include "singlet/pileup/gpu_pileup_bridge.h"
#include "records_gpu.h"
#include "singlet/pileup/cb_correct.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

using e6::u64;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <class T> static T* toDev(const T* h, u64 n) {
    T* d; CK(cudaMalloc(&d, std::max<u64>(1, n) * sizeof(T) + 16));
    if (n) CK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static std::vector<T> toHost(const T* d, u64 n) {
    std::vector<T> v(n);
    if (n) CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
static const uint8_t* mapFile(const std::string& p, size_t& n) {
    int fd = open(p.c_str(), O_RDONLY); if (fd < 0) { perror(p.c_str()); return nullptr; }
    struct stat st; if (fstat(fd, &st) != 0) { close(fd); return nullptr; }
    n = (size_t)st.st_size;
    void* m = n ? mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0) : nullptr; close(fd);
    if (m == MAP_FAILED) { perror("mmap"); return nullptr; }
    return (const uint8_t*)m;
}

// ---- the dump (engine_gpu.cu dumpOutput)
struct Dump {
    u64 nRec = 0, nReads = 0, seqBytes = 0, blobBytes = 0, nChr = 0, nClip = 0;
    e6::RecParams P{};
    std::vector<u64> chrStart, seqOff, blobOff; std::vector<char> seq; std::vector<uint8_t> blob; std::vector<uint32_t> readRecord;
    bool load(const std::string& p) {
        FILE* f = fopen(p.c_str(), "rb"); if (!f) { perror(p.c_str()); return false; }
        char mg[8]; u64 h[6];
        bool ok = fread(mg, 1, 8, f) == 8 && memcmp(mg, "SGEDUMP1", 8) == 0 && fread(h, 8, 6, f) == 6;
        if (!ok) { fclose(f); return false; }
        nRec = h[0]; nReads = h[1]; seqBytes = h[2]; blobBytes = h[3]; nChr = h[4]; nClip = h[5];
        long long sMin; double sOver, mOver, mmL, mmRL; u64 mMin, mmMax, mult, outMult;
        ok = fread(&sMin, 8, 1, f) && fread(&sOver, 8, 1, f) && fread(&mMin, 8, 1, f) && fread(&mOver, 8, 1, f) && fread(&mmMax, 8, 1, f)
             && fread(&mmL, 8, 1, f) && fread(&mmRL, 8, 1, f) && fread(&mult, 8, 1, f) && fread(&outMult, 8, 1, f);
        P.outFilterScoreMin = (int)sMin; P.outFilterScoreMinOverLread = sOver; P.outFilterMatchNmin = mMin; P.outFilterMatchNminOverLread = mOver;
        P.outFilterMismatchNmax = mmMax; P.outFilterMismatchNoverLmax = mmL; P.outFilterMismatchNoverReadLmax = mmRL; P.outFilterMultimapNmax = mult;
        P.outSAMmultNmax = outMult; P.outSAMmapqUnique = 255; P.outSAMflagAND = 65535; P.outSAMflagOR = 0;
        auto rd = [&](void* d, u64 n) { u64 got = 0; while (ok && got < n) { size_t k = fread((char*)d + got, 1, std::min<u64>(n - got, 1ULL << 30), f); if (!k) ok = false; got += k; } };
        chrStart.resize(nChr + 1); rd(chrStart.data(), 8 * (nChr + 1));
        seqOff.resize(nRec + 1); rd(seqOff.data(), 8 * (nRec + 1)); seq.resize(seqBytes); rd(seq.data(), seqBytes);
        blobOff.resize(nRec + 1); rd(blobOff.data(), 8 * (nRec + 1)); blob.resize(blobBytes + 16, 0); rd(blob.data(), blobBytes);
        readRecord.resize(nReads); rd(readRecord.data(), 4 * nReads);
        fclose(f);
        return ok;
    }
};

// ---- printing a hand-off record (for the first difference)
static std::string recStr(const uint8_t* p) {
    std::string s; char b[256];
    const int n = *p++;
    snprintf(b, sizeof b, "nAln=%d", n); s += b;
    for (int k = 0; k < n; k++) {
        int32_t tid, pos; uint16_t flag, nh, nm, nc; uint8_t mq;
        memcpy(&tid, p, 4); memcpy(&pos, p + 4, 4); memcpy(&flag, p + 8, 2); mq = p[10]; memcpy(&nh, p + 12, 2); memcpy(&nm, p + 14, 2); memcpy(&nc, p + 16, 2);
        snprintf(b, sizeof b, " | tid=%d pos=%d flag=%u mapq=%u z=%u NH=%u nM=%u cig=", tid, pos, flag, mq, p[11], nh, nm); s += b;
        for (int c = 0; c < nc; c++) { uint32_t v; memcpy(&v, p + 18 + 4 * c, 4); snprintf(b, sizeof b, "%u%c", v >> 4, "MIDNSHP=X???????"[v & 15]); s += b; }
        p += 18 + 4 * nc;
    }
    uint16_t L; memcpy(&L, p, 2);
    uint64_t h = 1469598103934665603ULL;   // the sequence as a hash only: no read sequence leaves job /tmp
    for (int i = 0; i < (L + 1) / 2; i++) { h ^= p[2 + i]; h *= 1099511628211ULL; }
    snprintf(b, sizeof b, " | L=%u seq-hash=%016llx", L, (unsigned long long)h); s += b;
    return s;
}
static std::string blobStr(const uint8_t* p, u64 n) {
    std::string s; const uint8_t* e = p + n;
    while (p < e) { s += std::to_string(e6::getV(p)); s += ' '; }
    return s;
}
static double median(std::vector<double> v) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
    std::string dumpP, hoP, readsP, qP, bcP, wlP, r11P, starLogP, json; int reps = 5; int hostThreads = 0; u64 subsetN = 2000000; bool skipRef = false;
    int mapqU = 255, fAnd = 65535, fOr = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--dump") dumpP = nx(); else if (a == "--handoff") hoP = nx(); else if (a == "--reads") readsP = nx(); else if (a == "--q") qP = nx();
        else if (a == "--barcodes") bcP = nx(); else if (a == "--whitelist") wlP = nx(); else if (a == "--r11-hist") r11P = nx(); else if (a == "--star-log") starLogP = nx(); else if (a == "--json") json = nx(); else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--host-threads") hostThreads = atoi(argv[++i]); else if (a == "--subset") subsetN = strtoull(argv[++i], nullptr, 10);
        else if (a == "--skip-ref") skipRef = true; else if (a == "--mapq-unique") mapqU = atoi(argv[++i]);
        else if (a == "--flag-and") fAnd = atoi(argv[++i]); else if (a == "--flag-or") fOr = atoi(argv[++i]);
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (hostThreads > 0) omp_set_num_threads(hostThreads);
    hostThreads = omp_get_max_threads();
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device %s, %d SMs; host threads %d\n", prop.name, prop.multiProcessorCount, hostThreads);
    CK(cudaFree(0));
    // ---- inputs
    double t0 = now();
    Dump D;
    if (!D.load(dumpP)) { fprintf(stderr, "cannot load the dump %s\n", dumpP.c_str()); return 2; }
    D.P.outSAMmapqUnique = mapqU; D.P.outSAMflagAND = (uint16_t)fAnd; D.P.outSAMflagOR = (uint16_t)fOr;
    const e6::RecParams& P = D.P;
    printf("dump: %llu records, %llu reads, %llu sequence bytes, %llu blob bytes (%.1f B per record), %llu chr, loaded in %.1f s\n", D.nRec, D.nReads,
           D.seqBytes, D.blobBytes, (double)D.blobBytes / std::max<u64>(1, D.nRec), D.nChr, now() - t0);
    printf("params: outFilterScoreMin %d OverLread %.3f MatchNmin %llu OverLread %.3f MismatchNmax %llu NoverLmax %.3f NoverReadLmax %.3f MultimapNmax %llu "
           "outSAMmultNmax %lld mapqUnique %d flagAND %u flagOR %u\n", P.outFilterScoreMin, P.outFilterScoreMinOverLread, P.outFilterMatchNmin,
           P.outFilterMatchNminOverLread, P.outFilterMismatchNmax, P.outFilterMismatchNoverLmax, P.outFilterMismatchNoverReadLmax, P.outFilterMultimapNmax,
           (long long)P.outSAMmultNmax, P.outSAMmapqUnique, P.outSAMflagAND, P.outSAMflagOR);
    size_t hoN = 0; const uint8_t* ho = mapFile(hoP, hoN);
    if (!ho || hoN < 48 || memcmp(ho, "SGHOFF02", 8)) { fprintf(stderr, "bad hand-off %s\n", hoP.c_str()); return 2; }
    u64 hReads, hRec; uint32_t lay[6];
    memcpy(&hReads, ho + 8, 8); memcpy(&hRec, ho + 16, 8); memcpy(lay, ho + 24, 24);
    const uint32_t* hRR = (const uint32_t*)(ho + 48);
    const u64* hOff = (const u64*)(ho + 48 + 4 * hReads);
    const uint8_t* hData = (const uint8_t*)(hOff + hRec + 1);
    const uint32_t cbL = lay[1], umiL = lay[3];
    printf("hand-off: %llu reads, %llu records, %llu record bytes; layout cbS %u cbL %u umiS %u umiL %u cbumiL %u simple %u\n", hReads, hRec, hOff[hRec],
           lay[0], lay[1], lay[2], lay[3], lay[4], lay[5]);
    if (hRec != D.nRec || hReads != D.nReads) { fprintf(stderr, "the dump and the hand-off are not from the same run\n"); return 3; }
    const bool rrSame = memcmp(hRR, D.readRecord.data(), 4 * hReads) == 0;
    printf("readRecord: dump vs hand-off %s\n", rrSame ? "identical" : "DIFFERENT");
    {   // the engine's own timings from STAR's log of the same run (--star-log, else <hand-off dir>/../out/star_Log.out): lines with ENGINE_GPU only
        std::string lp = starLogP;
        if (lp.empty()) { const size_t s = hoP.rfind('/'); lp = (s == std::string::npos ? std::string(".") : hoP.substr(0, s)) + "/../out/star_Log.out"; }
        if (FILE* lf = fopen(lp.c_str(), "r")) {
            char line[1024];
            while (fgets(line, sizeof line, lf)) if (strstr(line, "ENGINE_GPU") || strstr(line, "engine_gpu")) printf("STARLOG %s", line);
            fclose(lf);
        } else printf("STARLOG none (%s)\n", lp.c_str());
    }

    // ---- hop 6 on the GPU
    t0 = now();
    uint8_t* dBlob = toDev(D.blob.data(), D.blob.size()); u64* dBlobOff = toDev(D.blobOff.data(), D.nRec + 1);
    char* dSeq = toDev(D.seq.data(), D.seqBytes); u64* dSeqOff = toDev(D.seqOff.data(), D.nRec + 1); u64* dChr = toDev(D.chrStart.data(), D.nChr + 1);
    CK(cudaDeviceSynchronize());
    const double secUp = now() - t0;
    printf("upload of blobs + sequences: %.2f s (%.2f GB)\n", secUp, (D.blob.size() + D.seqBytes + 16.0 * D.nRec) / 1e9);
    singlet_records_gpu::DeviceRecords R; std::string err;
    std::vector<double> msC, msS, msF, msT, wallB, msA;
    for (int rep = 0; rep < reps; rep++) {   // arrays only (what the pileup needs)
        double w0 = now();
        if (!singlet_records_gpu::buildRecords(P, dBlob, dBlobOff, dSeq, dSeqOff, D.nRec, dChr, false, R, err)) { fprintf(stderr, "buildRecords: %s\n", err.c_str()); return 4; }
        wallB.push_back(now() - w0); msA.push_back(R.msCount + R.msScan + R.msFill);
    }
    for (int rep = 0; rep < reps; rep++) {   // arrays + hand-off bytes
        if (!singlet_records_gpu::buildRecords(P, dBlob, dBlobOff, dSeq, dSeqOff, D.nRec, dChr, true, R, err)) { fprintf(stderr, "buildRecords: %s\n", err.c_str()); return 4; }
        msC.push_back(R.msCount); msS.push_back(R.msScan); msF.push_back(R.msFill); msT.push_back(R.msCount + R.msScan + R.msFill);
    }
    printf("GPU records: %llu records, %llu mapped, %llu alignments, %llu CIGAR ops, %llu seq bytes, %llu hand-off bytes; mate-gap %llu, no-exon %llu, "
           "trBest=trInit %llu\n", R.nRec, R.nMapped, R.nAln, R.nOps, R.nSeq, R.nBytes, R.nMateGap, R.nNoExon, R.nBadBest);
    printf("GPU time (median of %d): arrays only %.2f ms (call wall %.2f ms); with hand-off bytes: count %.2f + scan %.2f + fill %.2f = %.2f ms"
           " -> %.1f M records/s\n", reps, median(msA), 1e3 * median(wallB), median(msC), median(msS), median(msF), median(msT),
           D.nRec / (median(msT) / 1e3) / 1e6);
    // ---- hop 6 check 1: the hand-off bytes
    t0 = now();
    std::vector<uint8_t> gB = toHost(R.bytes, R.nBytes); std::vector<u64> gO = toHost(R.bytesOff, D.nRec + 1);
    u64 recSame = 0, recDiff = 0, firstDiff = ~0ULL, nUnm = 0;
    {
        #pragma omp parallel for reduction(+ : recSame, recDiff, nUnm) schedule(static, 65536)
        for (long long r = 0; r < (long long)D.nRec; r++) {
            const u64 a0 = gO[r], a1 = gO[r + 1], b0 = hOff[r], b1 = hOff[r + 1];
            const bool same = a1 - a0 == b1 - b0 && memcmp(gB.data() + a0, hData + b0, a1 - a0) == 0;
            if (same) recSame++; else recDiff++;
            nUnm += hData[b0] == 0;
        }
        if (recDiff) for (u64 r = 0; r < D.nRec; r++) {
            const u64 a0 = gO[r], a1 = gO[r + 1], b0 = hOff[r], b1 = hOff[r + 1];
            if (!(a1 - a0 == b1 - b0 && memcmp(gB.data() + a0, hData + b0, a1 - a0) == 0)) { firstDiff = r; break; }
        }
    }
    const bool bytesAll = R.nBytes == hOff[hRec] && memcmp(gB.data(), hData, R.nBytes) == 0 && memcmp(gO.data(), hOff, 8 * (D.nRec + 1)) == 0;
    printf("HOP6 hand-off records: %llu compared, %llu identical, %llu differ (records without alignments in the file: %llu); whole data block and offsets %s"
           " (%.1f s)\n", D.nRec, recSame, recDiff, nUnm, bytesAll ? "IDENTICAL" : "DIFFERENT", now() - t0);
    if (recDiff) {
        const u64 r = firstDiff;
        printf("first differing record %llu (L=%llu):\n  blob  %s\n  gpu   %s\n  file  %s\n", r, D.seqOff[r + 1] - D.seqOff[r],
               blobStr(D.blob.data() + D.blobOff[r], D.blobOff[r + 1] - D.blobOff[r]).c_str(), recStr(gB.data() + gO[r]).c_str(), recStr(hData + hOff[r]).c_str());
        int shown = 0;
        for (u64 q = r + 1; q < D.nRec && shown < 4; q++) {
            const u64 a0 = gO[q], a1 = gO[q + 1], b0 = hOff[q], b1 = hOff[q + 1];
            if (a1 - a0 == b1 - b0 && memcmp(gB.data() + a0, hData + b0, a1 - a0) == 0) continue;
            printf("differing record %llu:\n  blob  %s\n  gpu   %s\n  file  %s\n", q, blobStr(D.blob.data() + D.blobOff[q], D.blobOff[q + 1] - D.blobOff[q]).c_str(),
                   recStr(gB.data() + gO[q]).c_str(), recStr(hData + hOff[q]).c_str());
            shown++;
        }
    }
    std::vector<uint8_t>().swap(gB);
    // ---- hop 6 check 2: the pileup's arrays vs Collector::prepend_handoff
    std::vector<SigRec> gS = toHost(R.sigs, D.nRec); std::vector<AlnRec> gA = toHost(R.alns, R.nAln);
    std::vector<uint32_t> gOps = toHost(R.ops, R.nOps); std::vector<uint8_t> gSeq = toHost(R.seq, R.nSeq);
    u64 sigSame = 0, alnSame = 0; bool opsSame = false, seqSame = false, refOk = false; double secRef = 0;
    if (!skipRef) {
        // the bridge consumes (unlinks) its hand-off: give it a hard link of the file
        const std::string lk = hoP + ".e6ref";
        unlink(lk.c_str());
        if (link(hoP.c_str(), lk.c_str()) != 0) { perror("link"); return 5; }
        setenv("SINGLET_ENGINE_GPU_HANDOFF", lk.c_str(), 1);
        singlet::gpupileup::Collector col(10);
        singlet_engine_gpu::PileupInput pi;
        t0 = now();
        if (col.handoff() == nullptr) { fprintf(stderr, "the bridge cannot load %s\n", lk.c_str()); return 5; }
        col.finish(pi);
        secRef = now() - t0;
        unlink(lk.c_str());
        const u64 nS = pi.sigs.size() / sizeof(SigRec), nA = pi.alns.size() / sizeof(AlnRec);
        const SigRec* rS = (const SigRec*)pi.sigs.data(); const AlnRec* rA = (const AlnRec*)pi.alns.data();
        for (u64 i = 0; i < std::min<u64>(nS, D.nRec); i++) sigSame += memcmp(rS + i, gS.data() + i, sizeof(SigRec)) == 0;
        for (u64 i = 0; i < std::min<u64>(nA, R.nAln); i++) alnSame += memcmp(rA + i, gA.data() + i, sizeof(AlnRec)) == 0;
        opsSame = pi.ops.size() == R.nOps && memcmp(pi.ops.data(), gOps.data(), 4 * R.nOps) == 0;
        seqSame = pi.seq.size() == R.nSeq && memcmp(pi.seq.data(), gSeq.data(), R.nSeq) == 0;
        refOk = nS == D.nRec && nA == R.nAln && sigSame == nS && alnSame == nA && opsSame && seqSame;
        printf("HOP6 pileup arrays vs Collector::prepend_handoff (%.1f s on the host): SigRec %llu / %llu identical (ref %llu), AlnRec %llu / %llu identical"
               " (ref %llu), ops %s (%llu vs ref %zu), seq %s (%llu vs ref %zu bytes) -> %s\n", secRef, sigSame, D.nRec, nS, alnSame, R.nAln, nA,
               opsSame ? "identical" : "DIFFERENT", R.nOps, pi.ops.size(), seqSame ? "identical" : "DIFFERENT", R.nSeq, pi.seq.size(), refOk ? "IDENTICAL" : "DIFFERENT");
        if (!refOk) {
            for (u64 i = 0; i < std::min<u64>(nS, D.nRec); i++) if (memcmp(rS + i, gS.data() + i, sizeof(SigRec))) {
                printf("  first SigRec diff %llu: gpu aln_off %u n_aln %u nh %u rlen %u seq_off %llu | ref %u %u %u %u %llu\n", i, gS[i].aln_off, gS[i].n_aln,
                       gS[i].nh, gS[i].rlen, (u64)gS[i].seq_off, rS[i].aln_off, rS[i].n_aln, rS[i].nh, rS[i].rlen, (u64)rS[i].seq_off);
                break;
            }
            for (u64 i = 0; i < std::min<u64>(nA, R.nAln); i++) if (memcmp(rA + i, gA.data() + i, sizeof(AlnRec))) {
                printf("  first AlnRec diff %llu: gpu tid %d pos %d ops_off %u n_ops %u flags %u mapq %u | ref %d %d %u %u %u %u\n", i, gA[i].tid, gA[i].pos,
                       gA[i].ops_off, gA[i].n_ops, gA[i].flags, gA[i].mapq, rA[i].tid, rA[i].pos, rA[i].ops_off, rA[i].n_ops, rA[i].flags, rA[i].mapq);
                break;
            }
        }
    }
    // ---- hop 6 on the host: the same source (records.cuh) on OpenMP threads, two passes + prefix sums; and identity with the GPU
    double secHost = 0; bool hostSame = false;
    {
        std::vector<e6::RecCount> rc(D.nRec);
        std::vector<SigRec> hS(D.nRec);
        t0 = now();
        #pragma omp parallel for schedule(dynamic, 4096)
        for (long long r = 0; r < (long long)D.nRec; r++) rc[r] = e6::recordCount(P, D.blob.data() + D.blobOff[r], D.seqOff[r + 1] - D.seqOff[r]);
        std::vector<u64> oA(D.nRec + 1), oO(D.nRec + 1), oS(D.nRec + 1), oB(D.nRec + 1);
        oA[0] = oO[0] = oS[0] = oB[0] = 0;
        for (u64 r = 0; r < D.nRec; r++) {
            const u64 L = D.seqOff[r + 1] - D.seqOff[r];
            oA[r + 1] = oA[r] + rc[r].nAln; oO[r + 1] = oO[r] + rc[r].nOps; oS[r + 1] = oS[r] + e6::seqBytes(L);
            oB[r + 1] = oB[r] + e6::recBytes(rc[r].nAln, rc[r].nOps, L);
        }
        std::vector<AlnRec> hA(oA[D.nRec]); std::vector<uint32_t> hO(oO[D.nRec]); std::vector<uint8_t> hQ(oS[D.nRec]), hB(oB[D.nRec]);
        #pragma omp parallel for schedule(dynamic, 4096)
        for (long long r = 0; r < (long long)D.nRec; r++)
            e6::recordFill(P, D.blob.data() + D.blobOff[r], D.seq.data() + D.seqOff[r], D.seqOff[r + 1] - D.seqOff[r], D.chrStart.data(), rc[r].nAln,
                           (uint32_t)oA[r], (uint32_t)oO[r], oS[r], &hS[r], hA.data() + oA[r], hO.data(), hQ.data() + oS[r], hB.data() + oB[r]);
        secHost = now() - t0;
        hostSame = oB[D.nRec] == hOff[hRec] && memcmp(hB.data(), hData, hB.size()) == 0 && memcmp(hS.data(), gS.data(), 16 * D.nRec) == 0
                   && hA.size() == R.nAln && memcmp(hA.data(), gA.data(), 16 * hA.size()) == 0 && hO.size() == R.nOps && memcmp(hO.data(), gOps.data(), 4 * hO.size()) == 0
                   && hQ.size() == R.nSeq && memcmp(hQ.data(), gSeq.data(), hQ.size()) == 0;
        printf("HOST port (records.cuh, %d threads): %.2f s = %.2f us per record per thread-equivalent %.3f us wall; output %s with the GPU's and the file's\n",
               hostThreads, secHost, secHost * hostThreads / D.nRec * 1e6, secHost / D.nRec * 1e6, hostSame ? "IDENTICAL" : "DIFFERENT");
    }
    // ---- ~2 M-record subset (strided, so the sequence order of the records is sampled evenly): GPU time
    double subMs = 0; u64 subN = 0;
    if (subsetN > 0 && subsetN < D.nRec) {
        const u64 k = D.nRec / subsetN;
        std::vector<u64> so{0}, bo{0}; std::vector<char> ss; std::vector<uint8_t> bb;
        for (u64 r = 0; r < D.nRec && so.size() <= subsetN; r += k) {
            ss.insert(ss.end(), D.seq.begin() + D.seqOff[r], D.seq.begin() + D.seqOff[r + 1]); so.push_back(ss.size());
            bb.insert(bb.end(), D.blob.begin() + D.blobOff[r], D.blob.begin() + D.blobOff[r + 1]); bo.push_back(bb.size());
        }
        subN = so.size() - 1; bb.resize(bb.size() + 16, 0);
        uint8_t* b2 = toDev(bb.data(), bb.size()); u64* bo2 = toDev(bo.data(), bo.size()); char* s2 = toDev(ss.data(), ss.size()); u64* so2 = toDev(so.data(), so.size());
        singlet_records_gpu::DeviceRecords R2; std::vector<double> m;
        for (int rep = 0; rep < reps; rep++) {
            if (!singlet_records_gpu::buildRecords(P, b2, bo2, s2, so2, subN, dChr, false, R2, err)) { fprintf(stderr, "subset: %s\n", err.c_str()); return 4; }
            m.push_back(R2.msCount + R2.msScan + R2.msFill);
        }
        subMs = median(m);
        printf("subset: %llu records (every %llu-th): arrays %.2f ms -> %.1f M records/s\n", subN, k, subMs, subN / (subMs / 1e3) / 1e6);
        singlet_records_gpu::release(R2); cudaFree(b2); cudaFree(bo2); cudaFree(s2); cudaFree(so2);
    }
    // ---- hop 7, in two modes of the CR table: 0 exact CR (no correction); 1 R11 (STAR's 1MM_multi_Nbase_pseudocounts correction,
    // singlet::CbCorrector, as the stream path runs it: config_.cb_corrector set). The table is the only difference; the GPU kernels and the
    // per-read rule are the same. The R11 corrector's exact histogram (cbReadCountExact) is rebuilt here from STAR's per-read entries (CR exact
    // whitelist hit, UR without N and not a homopolymer), i.e. over the reads with a record only (singlet's pass 1 counts every read of the
    // sample): it matters for CRs with several 1MM candidates (posterior), counted below.
    struct H7 { bool ran = false, ok = false; u64 kept = 0, same = 0, nKeys = 0, keysRej = 0, keysOut = 0; double msReads = 0, msKeys = 0, secResolve = 0, secRef = 0;
                u64 st[8] = {}; u64 outcome[e6::RO_N] = {}; };
    H7 h7[2];
    std::string wlUsed, histSrc; u64 wlHits = 0, wlSample = 0, histReads = 0;
    if (!readsP.empty() && !bcP.empty()) {
        size_t eN = 0, qN = 0; const uint8_t* em = mapFile(readsP, eN); const uint8_t* qm = qP.empty() ? nullptr : mapFile(qP, qN);
        if (!em || eN != 24 * hReads) { fprintf(stderr, "bad .reads (%zu bytes for %llu reads)\n", eN, hReads); return 6; }
        if (qm && qN != 16 * hReads) { printf("no usable .reads.q (%zu bytes)\n", qN); qm = nullptr; }
        const e6::ReadEntry* E = (const e6::ReadEntry*)em;
        const u64* Q = (const u64*)qm;
        // the barcode set as PileupEngine::load_barcodes reads it, keyed as BarcodeIndex::build keys it (strings instead of 64-bit hashes)
        std::vector<std::string> bcs;
        { FILE* f = fopen(bcP.c_str(), "r"); if (!f) { perror(bcP.c_str()); return 6; }
          char line[4096];
          while (fgets(line, sizeof line, f)) { std::string s(line); if (s.empty() || s[0] == '#') continue; size_t e = s.find_first_of("\t\r\n "); if (e != std::string::npos) s.resize(e); if (!s.empty()) bcs.push_back(s); }
          fclose(f); }
        std::unordered_map<std::string, int32_t> uni; uni.reserve(bcs.size() * 2);
        for (uint32_t i = 0; i < bcs.size(); i++) { uni[bcs[i]] = (int32_t)i; const size_t d = bcs[i].rfind('-'); if (d != std::string::npos && d > 0) uni[bcs[i].substr(0, d)] = (int32_t)i; }
        auto lookup = [&](const std::string& s) -> int32_t {   // BarcodeIndex::lookup: the whole string, then the part before the last '-'
            auto it = uni.find(s); if (it != uni.end()) return it->second;
            const size_t d = s.rfind('-'); if (d != std::string::npos && d > 0) { it = uni.find(s.substr(0, d)); if (it != uni.end()) return it->second; }
            return -1;
        };
        printf("barcode set: %zu barcodes\n", bcs.size());
        auto unpackS = [](u64 v, uint32_t L) { std::string s(L, 'N'); for (uint32_t j = 0; j < L; j++) s[j] = "ACGTNNNN"[(v >> (3 * j)) & 7]; return s; };
        auto urOk = [](const std::string& ur) { bool ok = !ur.empty(), homo = true; for (char ch : ur) { if (ch == 'N') ok = false; if (ch != ur[0]) homo = false; } return ok && !homo; };
        // R11: the whitelist (--whitelist, else the file of $SINGLET_WHITELIST_DIR with cbL-long barcodes that the most sampled CRs hit exactly)
        singlet::CbCorrector cc; bool ccOk = false;
        {
            std::vector<std::string> cand;
            if (!wlP.empty()) cand.push_back(wlP);
            else if (const char* wd = getenv("SINGLET_WHITELIST_DIR")) {
                for (const char* nm : {"3M-february-2018.txt", "737K-august-2016.txt", "737K-april-2014.txt", "737K-cratac-v1.txt", "gex_737K-arc-v1.txt"}) {
                    const std::string p = std::string(wd) + "/" + nm; struct stat st; if (stat(p.c_str(), &st) == 0) cand.push_back(p);
                }
            }
            std::vector<std::string> sample;
            for (u64 i = 0; i < hReads && sample.size() < 200000; i += 97) if (E[i].flags & 1) { std::string cr = unpackS(E[i].cr3, cbL); if (cr.find('N') == std::string::npos) sample.push_back(cr); }
            wlSample = sample.size();
            for (const auto& p : cand) {
                singlet::CbCorrector c;
                if (!c.load_whitelist(p) || c.length() != (int)cbL) continue;
                u64 hit = 0; for (const auto& s : sample) hit += c.find(s.c_str()) >= 0;
                printf("whitelist %s: %zu barcodes, %llu / %llu sampled CRs exact\n", p.c_str(), c.size(), hit, (u64)sample.size());
                if (hit > wlHits) { wlHits = hit; wlUsed = p; }
            }
            if (!wlUsed.empty() && 2 * wlHits > wlSample) {
                ccOk = cc.load_whitelist(wlUsed);
                // the engine's histogram (r11_hist.cpp over the .1fq: --r11-hist, else <json dir>/../r11_exact_hist.bin), else one from the entries
                std::string hp = r11P;
                if (hp.empty() && !json.empty()) { const size_t s = json.rfind('/'); hp = (s == std::string::npos ? std::string(".") : json.substr(0, s)) + "/../r11_exact_hist.bin"; }
                if (ccOk && !hp.empty()) {
                    if (FILE* hf = fopen(hp.c_str(), "rb")) {
                        char mg[8]; u64 hh[3];
                        if (fread(mg, 1, 8, hf) == 8 && memcmp(mg, "E6R11H01", 8) == 0 && fread(hh, 8, 3, hf) == 3 && hh[0] == cc.size()) {
                            u64 k = 0, p[2];
                            for (; k < hh[2] && fread(p, 8, 2, hf) == 2; k++) { cc.add_exact((int64_t)p[0], p[1]); histReads += p[1]; }
                            histSrc = k == hh[2] && histReads == hh[1] ? hp : "";
                            if (histSrc.empty()) fprintf(stderr, "truncated %s: histogram from the entries instead\n", hp.c_str());
                        }
                        fclose(hf);
                    }
                }
                if (ccOk && histSrc.empty()) {
                    histReads = 0;
                    singlet::CbCorrector c2; ccOk = c2.load_whitelist(wlUsed);
                    cc = c2;
                    for (u64 i = 0; ccOk && i < hReads; i++) {
                        if (!(E[i].flags & 1)) continue;
                        const std::string cr = unpackS(E[i].cr3, cbL), ur = unpackS(E[i].ur3, umiL);
                        if (!urOk(ur)) continue;
                        const int64_t w = cc.find(cr.c_str());
                        if (w >= 0) { cc.add_exact(w, 1); histReads++; }
                    }
                    histSrc = "entries";
                }
                printf("R11: whitelist %s, exact histogram of %llu reads from %s\n", wlUsed.c_str(), histReads,
                       histSrc == "entries" ? "the reads with an entry (approximate: no reads of the stream)" : histSrc.c_str());
            } else printf("R11: no whitelist (mode 1 skipped)\n");
        }
        uint32_t* dRR = toDev(D.readRecord.data(), hReads); e6::ReadEntry* dE = toDev(E, hReads); u64* dQ = Q ? toDev(Q, 2 * hReads) : nullptr;
        std::vector<u64> keys; float kms = 0;
        if (!singlet_records_gpu::distinctBarcodes(dRR, dE, hReads, cbL, keys, kms, err)) { fprintf(stderr, "%s\n", err.c_str()); return 6; }
        for (int mode = 0; mode < 2; mode++) {
            if (mode == 1 && !ccOk) break;
            H7& h = h7[mode]; h.ran = true; h.msKeys = kms; h.nKeys = keys.size();
            // the CR table, resolved on the host per distinct CR: column, -1 (not in the set) or -2 (no barcode)
            t0 = now();
            std::vector<int32_t> vals(keys.size());
            #pragma omp parallel for schedule(dynamic, 4096)
            for (long long k = 0; k < (long long)keys.size(); k++) {
                const std::string cr = unpackS(keys[k], cbL);
                if (mode == 0) { vals[k] = cr.find('N') != std::string::npos ? -2 : lookup(cr); continue; }
                char buf[40] = {};
                const int st = (int)cr.size() != cc.length() ? -9 : cc.resolve(cr.c_str(), nullptr, buf);
                vals[k] = st < 0 ? -2 : lookup(std::string(buf, cc.length()));
            }
            h.secResolve = now() - t0;
            for (int32_t v : vals) { h.keysRej += v == -2; h.keysOut += v == -1; }
            u64* dK = toDev(keys.data(), keys.size()); int32_t* dV = toDev(vals.data(), vals.size());
            singlet_records_gpu::DeviceReads RD; std::vector<double> me;
            for (int rep = 0; rep < reps; rep++) {
                if (!singlet_records_gpu::expandReads(dRR, dE, dQ, hReads, R, cbL, umiL, dK, dV, keys.size(), RD, err)) { fprintf(stderr, "expandReads: %s\n", err.c_str()); return 6; }
                me.push_back(RD.msEval + RD.msScan + RD.msScatter);
            }
            h.msReads = median(me);
            for (int k = 0; k < e6::RO_N; k++) h.outcome[k] = RD.outcome[k];
            printf("GPU reads [%s]: %llu distinct CRs (%.2f ms on the GPU; host table %.3f s: %llu rejected, %llu not in the set), per-read kernels %.2f ms"
                   " (eval %.2f, scan %.2f, scatter %.2f) -> %.0f M reads/s; %llu ReadRecs\n", mode ? "R11" : "exact CR", h.nKeys, h.msKeys, h.secResolve,
                   h.keysRej, h.keysOut, h.msReads, RD.msEval, RD.msScan, RD.msScatter, hReads / (h.msReads / 1e3) / 1e6, RD.nOut);
            printf("  outcomes: stream %llu, unmapped %llu, no primary %llu, unwritten %llu, clip>255 %llu, no barcode %llu, not in set %llu, kept %llu; "
                   "kept lowQual %llu qualNeqN %llu, every kept read has a mask: %s\n", RD.outcome[0], RD.outcome[1], RD.outcome[2], RD.outcome[3], RD.outcome[4],
                   RD.outcome[5], RD.outcome[6], RD.outcome[7], RD.lowQualSum, RD.qualNeqNSum, RD.maskAll ? "yes" : "no");
            std::vector<ReadRec> gR = toHost(RD.reads, RD.nOut);
            std::vector<u64> gQ = RD.lowq ? toHost(RD.lowq, 2 * RD.nOut) : std::vector<u64>();
            // host reference: gpu_handoff_pass_entries' loop, serial, from the hand-off file's records; mode 1 corrects each read's CR with
            // CbCorrector::resolve as the pass does (no table)
            t0 = now();
            std::vector<ReadRec> rR; std::vector<u64> rQ; rR.reserve(RD.nOut + 16);
            u64 cnt[e6::RO_N] = {}, lq = 0, ne = 0;
            for (u64 i = 0; i < hReads; i++) {
                const uint32_t rec = hRR[i];
                if (rec == 0xFFFFFFFFu) { cnt[e6::RO_STREAM]++; continue; }
                const uint8_t* p = hData + hOff[rec];
                const int n = *p++;
                if (n == 0) { cnt[e6::RO_UNMAPPED]++; continue; }
                int pk = -1;
                for (int k = 0; k < n; k++) { uint16_t flag, nc; memcpy(&flag, p + 8, 2); memcpy(&nc, p + 16, 2); p += 18 + 4 * nc; if (!(flag & 0x100) && pk < 0) pk = k; }
                if (pk < 0) { cnt[e6::RO_NOPRIMARY]++; continue; }
                if (!(E[i].flags & 1)) { cnt[e6::RO_UNWRITTEN]++; continue; }
                const uint32_t clip = (E[i].flags & 0xFFFF00u) ? E[i].flags : 0u;
                if (clip & 32u) { cnt[e6::RO_BIGCLIP]++; continue; }
                std::string cr = unpackS(E[i].cr3, cbL), ur = unpackS(E[i].ur3, umiL);
                bool cbok = !cr.empty() && !ur.empty() && ur[0] != ' ';
                if (mode == 0) for (char ch : cr) if (ch == 'N') cbok = false;
                bool homo = true;
                for (char ch : ur) { if (ch == 'N') cbok = false; if (ch != ur[0]) homo = false; }
                if (homo) cbok = false;
                if (mode == 1) {
                    if (!cbok) ++h.st[1];
                    else if ((int)cr.size() != cc.length()) { ++h.st[3]; cbok = false; }
                    else {
                        char buf[40] = {};
                        const int cst = cc.resolve(cr.c_str(), nullptr, buf);
                        ++h.st[cst + 4];
                        if (cst < 0) cbok = false; else cr.assign(buf, (size_t)cc.length());
                    }
                }
                if (!cbok || cr[0] == '-') { cnt[e6::RO_NOBARCODE]++; continue; }
                const int32_t bc = lookup(cr);
                if (bc < 0) { cnt[e6::RO_NOTINSET]++; continue; }
                cnt[e6::RO_KEEP]++; lq += E[i].lowQual; ne += E[i].qualNeqN;
                uint64_t u3 = 0;
                for (int j = 0; j < (int)ur.size() && j < 21; j++) { const char c = ur[j]; const uint64_t v = c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4; u3 |= v << (3 * j); }
                ReadRec x; x.sig = rec; x.bc = bc; x.umi3 = u3; rR.push_back(x);
                if (Q) { const bool has = !(E[i].flags & 64u); rQ.push_back(has ? Q[2 * i] : 0); rQ.push_back(has ? Q[2 * i + 1] : 0); }
            }
            h.secRef = now() - t0;
            h.kept = rR.size();
            for (u64 k = 0; k < std::min<u64>(rR.size(), gR.size()); k++) h.same += memcmp(&rR[k], &gR[k], sizeof(ReadRec)) == 0;
            const bool qSame = rQ == gQ;
            bool cntSame = true; for (int k = 0; k < e6::RO_N; k++) cntSame = cntSame && cnt[k] == RD.outcome[k];
            h.ok = rR.size() == gR.size() && h.same == rR.size() && qSame && cntSame && lq == RD.lowQualSum && ne == RD.qualNeqNSum;
            printf("HOP7 [%s] ReadRec vs the host transcription of gpu_handoff_pass_entries (serial, %.1f s): %llu / %llu identical (GPU %llu), lowq %s,"
                   " outcome counts %s, quality sums %s -> %s\n", mode ? "R11" : "exact CR", h.secRef, h.same, h.kept, (u64)gR.size(), qSame ? "identical" : "DIFFERENT",
                   cntSame ? "identical" : "DIFFERENT", (lq == RD.lowQualSum && ne == RD.qualNeqNSum) ? "identical" : "DIFFERENT", h.ok ? "IDENTICAL" : "DIFFERENT");
            if (mode == 1)
                printf("  [cb-correct] of the hand-off reads (the engine's line adds the stream's primary records): exact=%llu one_candidate=%llu "
                       "posterior_accepted=%llu posterior_rejected=%llu no_match=%llu two_N=%llu bad_umi=%llu\n", h.st[4], h.st[5], h.st[6], h.st[0], h.st[3], h.st[2], h.st[1]);
            if (!h.ok) {
                printf("  ref outcomes:"); for (int k = 0; k < e6::RO_N; k++) printf(" %llu", cnt[k]); printf("\n");
                for (u64 k = 0; k < std::min<u64>(rR.size(), gR.size()); k++) if (memcmp(&rR[k], &gR[k], sizeof(ReadRec))) {
                    printf("  first ReadRec diff %llu: gpu sig %u bc %d umi3 %llx | ref %u %d %llx\n", k, gR[k].sig, gR[k].bc, (u64)gR[k].umi3, rR[k].sig, rR[k].bc, (u64)rR[k].umi3);
                    break;
                }
            }
            singlet_records_gpu::release(RD);
            cudaFree(dK); cudaFree(dV);
        }
        cudaFree(dRR); cudaFree(dE); if (dQ) cudaFree(dQ);
    }
    const bool readsOk = h7[0].ran && h7[0].ok && (!h7[1].ran || h7[1].ok);
    const bool hop6 = recDiff == 0 && bytesAll && (skipRef || refOk) && hostSame && R.nMateGap == 0;
    printf("\nE6 RESULT: hop 6 %s (records %llu/%llu identical, arrays %s, host port %s, mate-gap transcripts %llu); hop 7 %s\n", hop6 ? "PASS" : "FAIL",
           recSame, D.nRec, skipRef ? "not checked" : (refOk ? "identical" : "DIFFERENT"), hostSame ? "identical" : "DIFFERENT", R.nMateGap,
           readsP.empty() ? "not run" : (readsOk ? (h7[1].ran ? "PASS (exact CR and R11)" : "PASS (exact CR only)") : "FAIL"));
    if (!json.empty()) {
        FILE* f = fopen(json.c_str(), "w");
        if (f) {
            fprintf(f, "{\n  \"device\": \"%s\",\n  \"records\": %llu,\n  \"reads\": %llu,\n  \"records_mapped\": %llu,\n  \"alignments\": %llu,\n  \"cigar_ops\": %llu,\n"
                       "  \"seq_bytes\": %llu,\n  \"handoff_bytes\": %llu,\n  \"blob_bytes\": %llu,\n  \"records_identical\": %llu,\n  \"records_differ\": %llu,\n"
                       "  \"data_block_identical\": %s,\n  \"sigrec_identical\": %llu,\n  \"alnrec_identical\": %llu,\n  \"ops_identical\": %s,\n  \"seq_identical\": %s,\n"
                       "  \"arrays_identical\": %s,\n  \"host_port_identical\": %s,\n  \"mate_gap\": %llu,\n  \"no_exon\": %llu,\n  \"bad_best\": %llu,\n"
                       "  \"gpu_ms_arrays\": %.3f,\n  \"gpu_ms_count\": %.3f,\n  \"gpu_ms_scan\": %.3f,\n  \"gpu_ms_fill\": %.3f,\n  \"gpu_ms_with_bytes\": %.3f,\n"
                       "  \"gpu_call_wall_ms\": %.3f,\n  \"upload_s\": %.3f,\n  \"subset_records\": %llu,\n  \"subset_gpu_ms\": %.3f,\n  \"host_port_s\": %.3f,\n"
                       "  \"host_threads\": %d,\n  \"prepend_handoff_s\": %.3f,\n  \"reads_ok\": %s,\n  \"whitelist\": \"%s\",\n  \"r11_hist_reads\": %llu,\n  \"r11_hist_source\": \"%s\",\n",
                    prop.name, D.nRec, D.nReads, R.nMapped, R.nAln, R.nOps, R.nSeq, R.nBytes, D.blobBytes, recSame, recDiff, bytesAll ? "true" : "false",
                    sigSame, alnSame, opsSame ? "true" : "false", seqSame ? "true" : "false", refOk ? "true" : "false", hostSame ? "true" : "false",
                    R.nMateGap, R.nNoExon, R.nBadBest, median(msA), median(msC), median(msS), median(msF), median(msT), 1e3 * median(wallB), secUp, subN, subMs,
                    secHost, hostThreads, secRef, readsOk ? "true" : "false", wlUsed.c_str(), histReads, histSrc.c_str());
            for (int m = 0; m < 2; m++) {
                const H7& h = h7[m];
                fprintf(f, "  \"%s\": {\"ran\": %s, \"ok\": %s, \"reads_kept\": %llu, \"reads_identical\": %llu, \"cr_keys\": %llu, \"cr_keys_rejected\": %llu,"
                           " \"cr_keys_not_in_set\": %llu, \"cr_keys_gpu_ms\": %.3f, \"cr_table_host_s\": %.3f, \"reads_gpu_ms\": %.3f, \"reads_ref_host_s\": %.3f,"
                           " \"outcomes\": [%llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu], \"cb_status\": [%llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu]}%s\n",
                        m ? "hop7_r11" : "hop7_exact", h.ran ? "true" : "false", h.ok ? "true" : "false", h.kept, h.same, h.nKeys, h.keysRej, h.keysOut, h.msKeys,
                        h.secResolve, h.msReads, h.secRef, h.outcome[0], h.outcome[1], h.outcome[2], h.outcome[3], h.outcome[4], h.outcome[5], h.outcome[6],
                        h.outcome[7], h.st[0], h.st[1], h.st[2], h.st[3], h.st[4], h.st[5], h.st[6], h.st[7], m ? "" : ",");
            }
            fprintf(f, "}\n");
            fclose(f);
        }
    }
    singlet_records_gpu::release(R);
    return hop6 ? 0 : 1;
}
