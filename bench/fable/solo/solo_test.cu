// E9 driver: STARsolo's Gene outputs from the GPU library (src/engine_gpu/solo_gpu.cu) against STAR's own Solo.out of the same run.
// docs/gpu-engine/fable/E9-solo.md.
//
// Inputs (one `singlet --engine gpu` run with SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=1, bench/fable/solo/jobs/truth.sbatch):
//   --dump      SINGLET_ENGINE_GPU_DUMP: mapDistinct's output (sequences, blobs, readRecord, filter parameters, chrStart; E6)
//   --solo-dir  SINGLET_SOLO_DUMP: per mapping thread solo_<t>.bin, one 48-byte record per read in the thread's order (what STAR's Solo Gene
//               feature saw: CR, UR, gene set, nTr, cbMatch, flag; src/star/ReadAlign_outputAlignments.cpp)
//   --genome-dir STAR index (chrName.txt, geneInfo.tab, transcriptInfo.tab, exonInfo.tab), --whitelist, --truth <star_Solo.out/>
// What runs on the library: per record the Gene annotation (from the blob), per read the barcode rule, the posterior, Solo's record and
// inputRecords rules, the (CB, gene, UMI) sort and the 1MM_All collapse. What the dump supplies: CR/UR per read (in the engine: STAR's
// per-read hand-off entry), and the annotation of the reads that have no record (sequences with N: STAR maps them on the host).
// Checks: per read the library's annotation / cbMatch / first match / record flag vs STAR's; every output file byte for byte vs STAR's.
#include "solo_gpu.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace e9;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- E6's dump loader (bench/fable/records/records_test.cu)
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
        seqOff.resize(nRec + 1); rd(seqOff.data(), 8 * (nRec + 1));
        // the sequences are not needed here: skipped, never held
        if (ok && fseeko(f, (off_t)seqBytes, SEEK_CUR) != 0) ok = false;
        blobOff.resize(nRec + 1); rd(blobOff.data(), 8 * (nRec + 1)); blob.resize(blobBytes + 16, 0); rd(blob.data(), blobBytes);
        readRecord.resize(nReads); rd(readRecord.data(), 4 * nReads);
        fclose(f);
        return ok;
    }
};

// ---- the hook's record (src/star/ReadAlign_outputAlignments.cpp SoloDumpRec)
struct SoloDumpRec { u64 iRead, cr3, ur3; u32 gene, readFlag; i32 cbMatch; u32 cbInd0; uint16_t nTr; uint8_t ng, bits; u32 pad; };
static_assert(sizeof(SoloDumpRec) == 48, "SoloDumpRec");

static bool readAll(const std::string& p, std::string& s) {
    std::ifstream f(p, std::ios::binary); if (!f) return false;
    std::ostringstream o; o << f.rdbuf(); s = o.str(); return true;
}
static std::vector<std::string> splitLines(const std::string& s) {
    std::vector<std::string> v; size_t a = 0;
    while (a < s.size()) { size_t b = s.find('\n', a); if (b == std::string::npos) b = s.size(); v.push_back(s.substr(a, b - a)); a = b + 1; }
    return v;
}
// SequenceFuns.cpp:262 (string form, for the whitelist)
static i64 convertNuclStrToInt64(const std::string& S, u64& v) {
    v = 0; i64 posN = -1;
    for (u64 ii = 0; ii < S.size(); ii++) {
        const char c = S[ii];
        u64 nt = (c == 'A' || c == 'a') ? 0 : (c == 'C' || c == 'c') ? 1 : (c == 'G' || c == 'g') ? 2 : (c == 'T' || c == 't') ? 3 : 4;
        if (nt > 3) { if (posN >= 0) return -2; posN = ii; nt = 0; }
        v = v << 2; v += nt;
    }
    return posN;
}

// one output file vs STAR's: identical, or the first differing line
static bool compareFile(const std::string& mine, const std::string& truth, const std::string& name, bool sortedRows, std::string& report) {
    std::string a, b; char buf[1024];
    const bool ha = readAll(mine, a), hb = readAll(truth, b);
    if (!ha || !hb) { snprintf(buf, sizeof buf, "%-32s MISSING (%s%s)\n", name.c_str(), ha ? "" : "ours ", hb ? "" : "STAR's"); report += buf; return false; }
    if (a == b) {
        size_t nl = std::count(a.begin(), a.end(), '\n');
        snprintf(buf, sizeof buf, "%-32s IDENTICAL (%zu bytes, %zu lines)\n", name.c_str(), a.size(), nl); report += buf; return true;
    }
    auto la = splitLines(a), lb = splitLines(b);
    size_t i = 0; while (i < la.size() && i < lb.size() && la[i] == lb[i]) i++;
    snprintf(buf, sizeof buf, "%-32s DIFFERS (%zu vs %zu bytes, %zu vs %zu lines); first difference at line %zu:\n", name.c_str(), a.size(), b.size(), la.size(), lb.size(), i + 1);
    report += buf;
    report += "    ours:   " + (i < la.size() ? la[i].substr(0, 300) : std::string("<end>")) + "\n";
    report += "    STAR's: " + (i < lb.size() ? lb[i].substr(0, 300) : std::string("<end>")) + "\n";
    if (sortedRows && la.size() > 2 && lb.size() > 2) {
        std::vector<std::string> sa(la.begin() + 2, la.end()), sb(lb.begin() + 2, lb.end());
        std::sort(sa.begin(), sa.end()); std::sort(sb.begin(), sb.end());
        const bool same = sa == sb && la[0] == lb[0] && la[1] == lb[1];
        snprintf(buf, sizeof buf, "    rows as a set (header lines + sorted rows): %s\n", same ? "IDENTICAL" : "DIFFER");
        report += buf;
        if (!same) {
            size_t j = 0; while (j < sa.size() && j < sb.size() && sa[j] == sb[j]) j++;
            report += "      first sorted difference: ours " + (j < sa.size() ? sa[j].substr(0, 300) : std::string("<end>")) + "\n";
            report += "                               STAR " + (j < sb.size() ? sb[j].substr(0, 300) : std::string("<end>")) + "\n";
        }
    }
    return false;
}

int main(int argc, char** argv) {
    std::string dumpP, soloDir, gDir, wlP, truth, outDir = "/tmp/e9out/";
    std::string mode = "gpu"; char qual = 'F'; int reps = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--dump") dumpP = nx(); else if (a == "--solo-dir") soloDir = nx(); else if (a == "--genome-dir") gDir = nx();
        else if (a == "--whitelist") wlP = nx(); else if (a == "--truth") truth = nx(); else if (a == "--out") outDir = nx();
        else if (a == "--mode") mode = nx(); else if (a == "--qual") qual = nx()[0]; else if (a == "--reps") reps = atoi(argv[++i]);
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (outDir.back() != '/') outDir += '/';
    if (truth.back() != '/') truth += '/';
    double T = now();
    // ---- engine dump
    Dump D;
    if (!D.load(dumpP)) { fprintf(stderr, "cannot load %s\n", dumpP.c_str()); return 1; }
    printf("[load] engine dump: %llu records, %llu reads, %llu blob bytes, %llu chromosomes (%.1f s)\n", (unsigned long long)D.nRec,
           (unsigned long long)D.nReads, (unsigned long long)D.blobBytes, (unsigned long long)D.nChr, now() - T);
    // ---- STAR's per-read Solo records (all threads)
    T = now();
    std::vector<std::vector<u32>> threadOrder;
    std::vector<SoloDumpRec> byRead(D.nReads);
    std::vector<uint8_t> seen(D.nReads, 0);
    u64 nDump = 0, dup = 0, outOfRange = 0, base = ~0ULL;
    std::vector<std::vector<SoloDumpRec>> files;
    for (int t = 0;; t++) {
        std::string p = soloDir + "/solo_" + std::to_string(t) + ".bin", s;
        if (!readAll(p, s)) break;
        files.emplace_back(s.size() / sizeof(SoloDumpRec));
        memcpy(files.back().data(), s.data(), files.back().size() * sizeof(SoloDumpRec));
        for (auto& r : files.back()) base = std::min(base, r.iRead);
    }
    for (auto& f : files) {
        threadOrder.emplace_back();
        for (auto& r : f) {
            const u64 i = r.iRead - base;
            if (i >= D.nReads) { outOfRange++; continue; }
            if (seen[i]) dup++;
            seen[i] = 1; byRead[i] = r; threadOrder.back().push_back((u32)i); nDump++;
        }
    }
    files.clear(); files.shrink_to_fit();
    u64 missing = 0; for (u64 i = 0; i < D.nReads; i++) missing += !seen[i];
    printf("[load] Solo dump: %zu thread files, %llu reads (iRead base %llu), %llu duplicates, %llu out of range, %llu reads without a record (%.1f s)\n",
           threadOrder.size(), (unsigned long long)nDump, (unsigned long long)base, (unsigned long long)dup, (unsigned long long)outOfRange,
           (unsigned long long)missing, now() - T);
    if (nDump != D.nReads || dup || outOfRange || missing) { fprintf(stderr, "the Solo dump does not cover the reads one to one\n"); }
    // ---- genome tables
    T = now();
    SoloModel M; SoloParams S;
    {
        std::ifstream f(gDir + "/chrName.txt"); std::vector<std::string> chr; std::string s;
        while (std::getline(f, s)) if (!s.empty()) chr.push_back(s);
        for (const char* m : {"chrM", "M", "MT"}) {   // genomeChrSetMitochondrial default (Genome.cpp:201: absent -> nChr)
            const i64 idx = std::find(chr.begin(), chr.end(), std::string(m)) - chr.begin();
            bool dupm = false; for (u32 k = 0; k < S.nMito; k++) dupm |= S.mitoChr[k] == idx;
            if (!dupm) S.mitoChr[S.nMito++] = idx;
        }
        printf("[load] %zu chromosomes; mito set:", chr.size()); for (u32 k = 0; k < S.nMito; k++) printf(" %lld", (long long)S.mitoChr[k]); printf("\n");
    }
    {
        std::ifstream f(gDir + "/geneInfo.tab"); u64 n; f >> n; std::string line; std::getline(f, line);
        M.geID.resize(n); M.geName.resize(n);
        for (u64 i = 0; i < n; i++) { std::getline(f, line); std::istringstream ls(line); std::string bt; ls >> M.geID[i] >> M.geName[i] >> bt; }
        S.nGe = (u32)n;
    }
    {
        std::ifstream f(gDir + "/transcriptInfo.tab"); u64 n; f >> n;
        M.trS.resize(n); M.trE.resize(n); M.trEmax.resize(n); M.trExI.resize(n); M.trExN.resize(n); M.trStr.resize(n); M.trGene.resize(n);
        for (u64 i = 0; i < n; i++) {
            std::string id; unsigned str1; f >> id >> M.trS[i] >> M.trE[i] >> M.trEmax[i] >> str1 >> M.trExN[i] >> M.trExI[i] >> M.trGene[i];
            M.trStr[i] = (uint8_t)str1;
        }
        if (!f.good()) { fprintf(stderr, "transcriptInfo.tab: read error\n"); return 1; }
    }
    {
        std::ifstream f(gDir + "/exonInfo.tab"); u64 n; f >> n; M.exSE.resize(2 * n);
        for (u64 i = 0; i < n; i++) { u32 lc; f >> M.exSE[2 * i] >> M.exSE[2 * i + 1] >> lc; }
    }
    {
        std::ifstream f(wlP); std::string s;
        while (f >> s) { u64 v; if (convertNuclStrToInt64(s, v)) M.wl.push_back(v); }   // ParametersSolo.cpp:320 (sic: only posN == 0 is dropped)
        std::sort(M.wl.begin(), M.wl.end()); M.wl.erase(std::unique(M.wl.begin(), M.wl.end()), M.wl.end());
    }
    printf("[load] genes %u, transcripts %zu, exons %zu, whitelist %zu (%.1f s)\n", S.nGe, M.trS.size(), M.exSE.size() / 2, M.wl.size(), now() - T);
    // ---- parameters
    {
        char qin = qual;
        qin -= 33; qin = qin < 33 ? qin : 33;   // SoloReadFeature_inputRecords.cpp:104-106 (QSbase = QSmax = 33)
        S.pq = std::pow(10.0, -qin / 10.0);
        S.umiMaskLow = (u32)((((u64)1) << S.umiL) - 1);
        for (u32 jj = 0; jj < 4; jj++) { S.homo[jj] = 0; for (u32 ii = 0; ii < S.umiL; ii++) S.homo[jj] = (S.homo[jj] << 2) + jj; }
        printf("[params] cbL %u umiL %u strand %d cbMinP %.9g qual '%c' -> pq %.17g umiMaskLow %x\n", S.cbL, S.umiL, S.strand, (double)S.cbMinP, qual, S.pq, S.umiMaskLow);
    }
    // ---- per-read inputs
    const u64 N = D.nReads;
    std::vector<u64> cr3(N), ur3(N);
    std::vector<RecAnn> hostAnn(N);
    u64 nNoRec = 0;
    for (u64 i = 0; i < N; i++) {
        const SoloDumpRec& r = byRead[i];
        cr3[i] = r.cr3; ur3[i] = r.ur3;
        RecAnn a{0xFFFFFFFFu, 0, 0, 0};
        if (D.readRecord[i] == 0xFFFFFFFFu) {   // a read the host mapped: STAR's annotation
            nNoRec++;
            a.nTr = r.nTr; a.ng = (uint8_t)std::min<u32>(r.ng, 2); a.gene = r.ng == 1 ? r.gene : 0xFFFFFFFFu;
            a.bits = (uint8_t)(((r.bits & 2) ? RA_MITO : 0) | (((r.bits >> 4) & 15) == 1 ? RA_EXONIC : 0));
        }
        hostAnn[i] = a;
    }
    printf("[inputs] %llu reads without a record (annotation from STAR's host mapping)\n", (unsigned long long)nNoRec);
    SoloRecs R; R.nRec = D.nRec; R.blob = D.blob.data(); R.blobBytes = D.blob.size(); R.blobOff = D.blobOff.data(); R.seqOff = D.seqOff.data(); R.RP = D.P;
    SoloReads Rd; Rd.n = N; Rd.cr3 = cr3.data(); Rd.ur3 = ur3.data(); Rd.rec = D.readRecord.data(); Rd.hostAnn = hostAnn.data();

    std::vector<std::string> modes;
    if (mode == "both") modes = {"host", "gpu"}; else modes = {mode};
    bool allOk = true;
    for (const std::string& md : modes) {
        const bool dev = md == "gpu";
        if (dev) { cudaFree(nullptr); cudaDeviceProp pr; cudaGetDeviceProperties(&pr, 0); printf("[%s] device %s\n", md.c_str(), pr.name); }
        printf("[%s] OpenMP threads %d\n", md.c_str(), omp_get_max_threads());
        SoloResult res;
        for (int rep = 0; rep < std::max(1, reps); rep++) {
            SoloResult r1; std::string log;
            const double t0 = now();
            const bool keep = rep == 0;
            if (!soloRun(dev, S, M, R, Rd, r1, keep, log)) { fprintf(stderr, "%s", log.c_str()); return 1; }
            const double t1 = now();
            printf("[%s] rep %d: %s", md.c_str(), rep, log.c_str());
            printf("[%s] rep %d stages (ms):", md.c_str(), rep);
            double tot = 0;
            for (auto& kv : r1.ms) { printf(" %s %.1f", kv.first.c_str(), kv.second); if (kv.first != "upload") tot += kv.second; }
            printf(" | compute (no upload) %.1f, wall %.1f\n", tot, (t1 - t0) * 1e3);
            if (rep == 0) res = std::move(r1);
        }
        // ---- per read vs STAR
        {
            u64 nCmp = 0, bad[8] = {}, shown[8] = {};
            const char* what[8] = {"nTr", "gene set size", "gene", "mito", "exonic", "cbMatch", "first match", "record flag"};
            for (u64 i = 0; i < N; i++) {
                const SoloDumpRec& r = byRead[i];
                const RecAnn& a = res.readAnn[i];
                bool f[8] = {};
                if (D.readRecord[i] != 0xFFFFFFFFu) {
                    nCmp++;
                    f[0] = a.nTr != r.nTr;
                    f[1] = a.ng != std::min<u32>(r.ng, 2);
                    f[2] = r.ng == 1 && a.gene != r.gene;
                    f[3] = ((a.bits & RA_MITO) != 0) != ((r.bits & 2) != 0);
                    f[4] = ((a.bits & RA_EXONIC) != 0) != ((((r.bits >> 4) & 15)) == 1);
                }
                f[5] = res.readCbm[i] != r.cbMatch;
                f[6] = res.readCb0[i] != (i32)r.cbInd0;
                f[7] = recordFlag(a, res.readCbm[i]) != r.readFlag;
                for (int k = 0; k < 8; k++) if (f[k]) {
                    bad[k]++;
                    if (shown[k]++ < 4)
                        printf("  [read %llu rec %u] %s: ours nTr=%u ng=%u gene=%d bits=%x cbm=%d cb0=%d flag=%x | STAR nTr=%u ng=%u gene=%d bits=%x cbm=%d cb0=%d flag=%x\n",
                               (unsigned long long)i, D.readRecord[i], what[k], a.nTr, a.ng, (int)a.gene, a.bits, res.readCbm[i], res.readCb0[i],
                               recordFlag(a, res.readCbm[i]), r.nTr, r.ng, (int)r.gene, r.bits, r.cbMatch, (int)r.cbInd0, r.readFlag);
                }
            }
            printf("[%s] per read vs STAR (%llu reads with a record annotated by the library; all %llu reads for the barcode and flag):\n", md.c_str(),
                   (unsigned long long)nCmp, (unsigned long long)N);
            for (int k = 0; k < 8; k++) printf("    %-14s %llu differ\n", what[k], (unsigned long long)bad[k]);
        }
        // ---- CellReads.stats: the keys in the order STAR's map got them (thread by thread, first occurrence in each thread's record order)
        std::vector<u32> order;
        {
            std::vector<uint8_t> inThread(M.wl.size(), 0);
            for (auto& to : threadOrder) {
                std::vector<u32> local;
                for (u32 i : to) { const i32 cb = res.rowCb[i]; if (cb >= 0 && !inThread[cb]) { inThread[cb] = 1; local.push_back((u32)cb); } }
                for (u32 cb : local) inThread[cb] = 0;
                order.insert(order.end(), local.begin(), local.end());
            }
        }
        // ---- the files
        const double tw = now();
        const std::string od = outDir + md + "/star_Solo.out/";
        { std::string c = "mkdir -p " + od; if (system(c.c_str()) != 0) {} }
        SoloWriteInfo wi; std::string log;
        if (!soloWrite(od, S, M, res, SoloFilterParams{}, order, wi, log)) { fprintf(stderr, "%s", log.c_str()); return 1; }
        const double twe = now();
        // the same rows sorted by CB (the order a run-independent writer would give)
        const std::string ods = outDir + md + "-sorted/star_Solo.out/";
        { std::string c = "mkdir -p " + ods; if (system(c.c_str()) != 0) {} }
        SoloWriteInfo wi2; std::string log2;
        const double tws = now();
        soloWrite(ods, S, M, res, SoloFilterParams{}, {}, wi2, log2);
        printf("[%s] write (CellReads rows sorted): %.1f ms\n", md.c_str(), (now() - tws) * 1e3);
        printf("[%s] write: %.1f ms (filter + files); nCB %llu, CellRanger2.2 nUMImax %llu nUMImin %llu, cells %llu; %s%s", md.c_str(), (twe - tw) * 1e3,
               (unsigned long long)wi.nCB, (unsigned long long)wi.nUMImax, (unsigned long long)wi.nUMImin, (unsigned long long)wi.nCells, log.c_str(), "\n");
        std::string rep;
        const char* files[] = {"Barcodes.stats", "Gene/Features.stats", "Gene/raw/barcodes.tsv", "Gene/raw/features.tsv", "Gene/raw/matrix.mtx",
                               "Gene/filtered/barcodes.tsv", "Gene/filtered/features.tsv", "Gene/filtered/matrix.mtx", "Gene/UMIperCellSorted.txt",
                               "Gene/CellReads.stats"};
        int nId = 0, nF = 0;
        for (const char* f : files) {
            const bool crs = std::string(f) == "Gene/CellReads.stats";
            const bool same = compareFile(od + f, truth + f, f, crs, rep);
            nId += same; nF++;
        }
        printf("[%s] files vs STAR's %s: %d of %d identical\n%s", md.c_str(), truth.c_str(), nId, nF, rep.c_str());
        allOk &= nId == nF;
    }
    printf("[done] %s\n", allOk ? "ALL IDENTICAL" : "DIFFERENCES");
    return 0;
}
