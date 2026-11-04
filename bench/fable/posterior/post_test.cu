// E15 driver: the per-read CB posterior of the GPU library (src/engine_gpu/solo_gpu.cu, SoloReads::cbQual) against STAR on a run whose barcode
// read has binned NovaSeq qualities. docs/gpu-engine/fable/E15-posterior.md.
//
// Inputs (one `singlet --engine cpu` run on the uncompressed 10x FASTQs with SINGLET_SOLO_DUMP; bench/fable/posterior/jobs/binned.sbatch):
//   --solo-dir   SINGLET_SOLO_DUMP: per mapping thread solo_<t>.bin, one 48-byte record per read (what STAR's Solo Gene feature saw: CR, UR,
//                gene set, nTr, cbMatch, first match, flag; src/star/ReadAlign_outputAlignments.cpp)
//   --r1         the barcode FASTQ STAR read (uncompressed): per read the CB quality characters (and its CR, checked against the dump's)
//   --genome-dir STAR index (chrName.txt, geneInfo.tab, transcriptInfo.tab, exonInfo.tab), --whitelist, --truth <star_Solo.out/>
// Every read takes STAR's own annotation (no records: SoloReads::rec all 0xFFFFFFFF); the library runs the barcode rule, the posterior,
// Solo's record and inputRecords rules, the (CB, gene, UMI) sort and the 1MM_All collapse.
// Runs: "perread" (SoloReads::cbQual from --r1: what level 5 does with qualities that vary) and "const" (one pq for every read, from 'F':
// what E11 did; the control: it must differ from STAR on this input, or the test cannot tell the two apart).
// Checks per run: per read cbMatch / first match / record flag vs STAR's dump; per multi-match read the library's CB vs a transcription of
// STAR's text-stream posterior (matchCBtoWL's cbMatchString read back with operator>>, SoloReadFeature_inputRecords.cpp:77-111, exact counts
// of getCBandUMI + 1 pseudocount); every output file byte for byte vs STAR's star_Solo.out.
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
static i64 convertNuclStrToInt64(const std::string& S, u64& v) {   // SequenceFuns.cpp:262
    v = 0; i64 posN = -1;
    for (u64 ii = 0; ii < S.size(); ii++) {
        const char c = S[ii];
        u64 nt = (c == 'A' || c == 'a') ? 0 : (c == 'C' || c == 'c') ? 1 : (c == 'G' || c == 'g') ? 2 : (c == 'T' || c == 't') ? 3 : 4;
        if (nt > 3) { if (posN >= 0) return -2; posN = ii; nt = 0; }
        v = v << 2; v += nt;
    }
    return posN;
}
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
    }
    return false;
}
static std::string unpack3(u64 cr, u32 n) {   // the dump's 3-bit packing back to bases (4: N)
    std::string s(n, 'N');
    for (u32 j = 0; j < n; j++) { const u64 nt = (cr >> (3 * j)) & 7; s[j] = nt == 0 ? 'A' : nt == 1 ? 'C' : nt == 2 ? 'G' : nt == 3 ? 'T' : 'N'; }
    return s;
}

int main(int argc, char** argv) {
    std::string soloDir, r1P, gDir, wlP, truth, outDir = "/tmp/e15out/", mode = "gpu";
    u32 umiL = 12, cbL = 16, cbS = 1, umiS = 17;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--solo-dir") soloDir = nx(); else if (a == "--r1") r1P = nx(); else if (a == "--genome-dir") gDir = nx();
        else if (a == "--whitelist") wlP = nx(); else if (a == "--truth") truth = nx(); else if (a == "--out") outDir = nx();
        else if (a == "--mode") mode = nx(); else if (a == "--umi-len") umiL = (u32)atoi(argv[++i]); else if (a == "--cb-len") cbL = (u32)atoi(argv[++i]);
        else if (a == "--cb-start") cbS = (u32)atoi(argv[++i]); else if (a == "--umi-start") umiS = (u32)atoi(argv[++i]);
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (outDir.back() != '/') outDir += '/';
    if (truth.back() != '/') truth += '/';
    const u32 cbumiL = std::max(cbS - 1 + cbL, umiS - 1 + umiL);
    double T = now();
    // ---- STAR's per-read Solo records (all threads)
    std::vector<std::vector<SoloDumpRec>> files;
    u64 base = ~0ULL, top = 0;
    for (int t = 0;; t++) {
        std::string p = soloDir + "/solo_" + std::to_string(t) + ".bin", s;
        if (!readAll(p, s)) break;
        files.emplace_back(s.size() / sizeof(SoloDumpRec));
        memcpy(files.back().data(), s.data(), files.back().size() * sizeof(SoloDumpRec));
        for (auto& r : files.back()) { base = std::min(base, r.iRead); top = std::max(top, r.iRead); }
    }
    if (files.empty()) { fprintf(stderr, "no Solo dump in %s\n", soloDir.c_str()); return 1; }
    const u64 N = top - base + 1;
    std::vector<SoloDumpRec> byRead(N);
    std::vector<uint8_t> seen(N, 0);
    std::vector<std::vector<u32>> threadOrder;
    u64 nDump = 0, dup = 0;
    for (auto& f : files) {
        threadOrder.emplace_back();
        for (auto& r : f) { const u64 i = r.iRead - base; if (seen[i]) dup++; seen[i] = 1; byRead[i] = r; threadOrder.back().push_back((u32)i); nDump++; }
    }
    files.clear(); files.shrink_to_fit();
    u64 missing = 0; for (u64 i = 0; i < N; i++) missing += !seen[i];
    printf("[load] Solo dump: %zu thread files, %llu records for reads %llu..%llu, %llu duplicates, %llu missing (%.1f s)\n", threadOrder.size(),
           (unsigned long long)nDump, (unsigned long long)base, (unsigned long long)top, (unsigned long long)dup, (unsigned long long)missing, now() - T);
    if (dup || missing) { fprintf(stderr, "the Solo dump does not cover the reads one to one\n"); return 1; }
    // ---- the barcode FASTQ: per read its CB qualities (getCBandUMI: padded with 'H' to cbumiL) and its CR (3 bits per base, as the dump packs it)
    T = now();
    std::vector<uint8_t> qual(N * cbL);
    u64 nFq = 0, crBad = 0, qh[256] = {};
    {
        std::ifstream f(r1P);
        if (!f) { fprintf(stderr, "cannot open %s\n", r1P.c_str()); return 1; }
        std::string l0, seq, l2, q;
        while (std::getline(f, l0) && std::getline(f, seq) && std::getline(f, l2) && std::getline(f, q)) {
            if (nFq >= N) { nFq++; continue; }
            if (!seq.empty() && (unsigned char)seq.back() < 33) seq.pop_back();
            if (!q.empty() && (unsigned char)q.back() < 33) q.pop_back();
            const u64 S = seq.size() < cbumiL ? cbumiL : seq.size();
            u64 v = 0;
            for (u64 j = cbS - 1; j < std::min<u64>(cbS - 1 + cbL, S) && j - (cbS - 1) < 21; j++) {
                const char c = j < seq.size() ? seq[j] : 'N';
                v |= (u64)(c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4) << (3 * (j - (cbS - 1)));
            }
            if (v != byRead[nFq].cr3) crBad++;
            for (u32 j = 0; j < cbL; j++) { const uint8_t c = cbS - 1 + j < q.size() ? (uint8_t)q[cbS - 1 + j] : (uint8_t)'H'; qual[nFq * cbL + j] = c; qh[c]++; }
            nFq++;
        }
    }
    printf("[load] barcode FASTQ: %llu reads (dump: %llu), %llu CRs differ from the dump's (%.1f s); CB quality characters:", (unsigned long long)nFq,
           (unsigned long long)N, (unsigned long long)crBad, now() - T);
    for (int c = 0; c < 256; c++) if (qh[c]) printf(" '%c' %llu", c, (unsigned long long)qh[c]);
    printf("\n");
    if (nFq != N || crBad) { fprintf(stderr, "the barcode FASTQ and the dump disagree\n"); return 1; }
    // ---- genome tables, whitelist
    T = now();
    SoloModel M; SoloParams S;
    S.cbL = cbL; S.umiL = umiL;
    {
        std::ifstream f(gDir + "/chrName.txt"); std::vector<std::string> chr; std::string s;
        while (std::getline(f, s)) if (!s.empty()) chr.push_back(s);
        for (const char* m : {"chrM", "M", "MT"}) {   // genomeChrSetMitochondrial default (Genome.cpp:201: absent -> nChr)
            const i64 idx = std::find(chr.begin(), chr.end(), std::string(m)) - chr.begin();
            bool dupm = false; for (u32 k = 0; k < S.nMito; k++) dupm |= S.mitoChr[k] == idx;
            if (!dupm) S.mitoChr[S.nMito++] = idx;
        }
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
    S.wlN = (u32)M.wl.size();
    printf("[load] genes %u, transcripts %zu, whitelist %zu (%.1f s)\n", S.nGe, M.trS.size(), M.wl.size(), now() - T);
    // ---- parameters: the quality table as e11Config builds it (QSbase = QSmax = 33, ParametersSolo.cpp:287-288)
    const char QSbase = 33, QSmax = 33;
    M.pqTab.assign(256, 0.0);
    for (int c = 0; c < 256; c++) { char qin = (char)c; qin -= QSbase; qin = qin < QSmax ? qin : QSmax; M.pqTab[c] = std::pow(10.0, -qin / 10.0); }
    S.umiMaskLow = (u32)((((u64)1) << S.umiL) - 1);
    for (u32 jj = 0; jj < 4; jj++) { S.homo[jj] = 0; for (u32 ii = 0; ii < S.umiL; ii++) S.homo[jj] = (S.homo[jj] << 2) + jj; }
    printf("[params] cbL %u umiL %u cbS %u umiS %u cbMinP %.9g\n", S.cbL, S.umiL, cbS, umiS, (double)S.cbMinP);
    // ---- STAR's posterior, transcribed: exact counts (addStats case 0, getCBandUMI.cpp:94: the reads whose CR is in the whitelist and whose UMI
    // passes the check, i.e. STAR's cbMatch 0; a bad UMI replaces cbMatch before addStats, :290-302, MATCH_CellRanger is not defined)
    // + 1 (Solo.cpp:66-70), cbMatchString as matchCBtoWL writes it, read back with operator>>, float arithmetic
    T = now();
    auto wlIdx = [&](u64 v) -> i64 { auto it = std::lower_bound(M.wl.begin(), M.wl.end(), v); return (it != M.wl.end() && *it == v) ? (i64)(it - M.wl.begin()) : -1; };
    std::vector<u32> cbTotal(M.wl.size(), 1);
    for (u64 i = 0; i < N; i++) {
        u64 v;
        if (byRead[i].cbMatch == 0 && convertNuclStrToInt64(unpack3(byRead[i].cr3, cbL), v) == -1) { const i64 c = wlIdx(v); if (c >= 0) cbTotal[c]++; }
    }
    std::vector<i32> starCb(N, -2);   // per multi-match read (STAR's cbMatch > 1) STAR's posterior choice, -1 noTooManyWLmatches; -2 not one
    u64 nStarMulti = 0, nStarAcc = 0, nConstDiff = 0;
    {
        const double pqF = M.pqTab['F'];
        const float cbMinP = S.cbMinP;
        for (u64 i = 0; i < N; i++) {
            if (byRead[i].cbMatch <= 1) continue;
            const std::string cbSeq = unpack3(byRead[i].cr3, cbL), cbQual((const char*)&qual[i * cbL], cbL);
            u64 cbB1; const i64 posN = convertNuclStrToInt64(cbSeq, cbB1);
            std::string ms; int m = 0;
            if (posN >= 0) {
                const u32 posNshift = 2 * (cbL - 1 - (u32)posN);
                for (u32 jj = 0; jj < 4; jj++) { const i64 c = wlIdx(cbB1 ^ ((u64)jj << posNshift)); if (c >= 0) { ms += ' ' + std::to_string(c) + ' ' + cbQual[posN]; m++; } }
            } else {
                for (u32 ii = 0; ii < cbL; ii++)
                    for (u32 jj = 1; jj < 4; jj++) { const i64 c = wlIdx(cbB1 ^ ((u64)jj << (ii * 2))); if (c >= 0) { ms += ' ' + std::to_string(c) + ' ' + cbQual.at(cbL - 1 - ii); m++; } }
            }
            if (m != byRead[i].cbMatch) { fprintf(stderr, "read %llu: %d matches, STAR %d\n", (unsigned long long)i, m, byRead[i].cbMatch); return 1; }
            std::istringstream st(ms);
            float ptot = 0.0, pmax = 0.0, pin, ptotF = 0.0, pmaxF = 0.0, pinF;
            i64 cb = -1, cbF = -1;
            for (int k = 0; k < m; k++) {
                u32 cbin; char qin;
                st >> cbin >> qin;
                if (cbTotal[cbin] > 0) {
                    qin -= QSbase; qin = qin < QSmax ? qin : QSmax;
                    pin = cbTotal[cbin] * std::pow(10.0, -qin / 10.0);
                    ptot += pin; if (pin > pmax) { cb = cbin; pmax = pin; }
                    pinF = cbTotal[cbin] * pqF;
                    ptotF += pinF; if (pinF > pmaxF) { cbF = cbin; pmaxF = pinF; }
                }
            }
            starCb[i] = (ptot > 0.0 && pmax >= cbMinP * ptot) ? (i32)cb : -1;
            const i32 constCb = (ptotF > 0.0 && pmaxF >= cbMinP * ptotF) ? (i32)cbF : -1;
            nStarMulti++; nStarAcc += starCb[i] >= 0; nConstDiff += constCb != starCb[i];
        }
    }
    printf("[star] posterior (transcription): %llu multi-match reads, %llu given a CB; with one quality 'F' for every base %llu of them would get "
           "another decision (%.1f s)\n", (unsigned long long)nStarMulti, (unsigned long long)nStarAcc, (unsigned long long)nConstDiff, now() - T);

    std::vector<u64> cr3(N), ur3(N);
    std::vector<RecAnn> hostAnn(N);
    std::vector<u32> rec(N, 0xFFFFFFFFu);
    for (u64 i = 0; i < N; i++) {
        const SoloDumpRec& r = byRead[i];
        cr3[i] = r.cr3; ur3[i] = r.ur3;
        RecAnn a{0xFFFFFFFFu, 0, 0, 0};
        a.nTr = r.nTr; a.ng = (uint8_t)std::min<u32>(r.ng, 2); a.gene = r.ng == 1 ? r.gene : 0xFFFFFFFFu;
        a.bits = (uint8_t)(((r.bits & 2) ? RA_MITO : 0) | (((r.bits >> 4) & 15) == 1 ? RA_EXONIC : 0));
        hostAnn[i] = a;
    }
    SoloRecs R; R.nRec = 0;   // no records: blob offsets of one element, a padded empty blob
    std::vector<u64> off1(1, 0); std::vector<uint8_t> blob0(16, 0);
    R.blob = blob0.data(); R.blobBytes = blob0.size(); R.blobOff = off1.data(); R.seqOff = off1.data();
    u64 cbQualCalls = 0, cbQualReads = 0, cbQualCrBad = 0;
    CbQualFn fetch = [&](const u32* idx, const u64* cr, u64 n, uint8_t* out, std::string& why) {
        cbQualCalls++; cbQualReads += n;
        for (u64 k = 0; k < n; k++) {
            if (idx[k] >= N) { why = "read index out of range"; return false; }
            if (cr[k] != cr3[idx[k]]) cbQualCrBad++;
            memcpy(out + k * cbL, &qual[(u64)idx[k] * cbL], cbL);
        }
        return true;
    };

    std::vector<std::string> devs = mode == "both" ? std::vector<std::string>{"host", "gpu"} : std::vector<std::string>{mode};
    bool allOk = true;
    for (const std::string& md : devs) {
        const bool dev = md == "gpu";
        if (dev) { cudaFree(nullptr); cudaDeviceProp pr; cudaGetDeviceProperties(&pr, 0); printf("[%s] device %s\n", md.c_str(), pr.name); }
        for (const std::string run : {"perread", "const"}) {
            const std::string tag = md + "/" + run;
            SoloReads Rd; Rd.n = N; Rd.cr3 = cr3.data(); Rd.ur3 = ur3.data(); Rd.rec = rec.data(); Rd.hostAnn = hostAnn.data();
            SoloParams S1 = S;
            if (run == "perread") Rd.cbQual = fetch; else S1.pq = M.pqTab['F'];
            SoloResult res; std::string log;
            cbQualCalls = cbQualReads = cbQualCrBad = 0;
            const double t0 = now();
            if (!soloRun(dev, S1, M, R, Rd, res, true, log)) { fprintf(stderr, "%s", log.c_str()); return 1; }
            printf("[%s] %s[%s] wall %.1f ms; stages (ms):", tag.c_str(), log.c_str(), tag.c_str(), (now() - t0) * 1e3);
            for (auto& kv : res.ms) printf(" %s %.1f", kv.first.c_str(), kv.second);
            printf("\n[%s] cbQual: %llu calls, %llu reads, %llu CRs differ from the reads'\n", tag.c_str(), (unsigned long long)cbQualCalls,
                   (unsigned long long)cbQualReads, (unsigned long long)cbQualCrBad);
            // per read vs STAR
            u64 bad[4] = {}, shown = 0, nMulti = 0;
            const char* what[4] = {"cbMatch", "first match", "record flag", "posterior CB"};
            for (u64 i = 0; i < N; i++) {
                const SoloDumpRec& r = byRead[i];
                const bool f0 = res.readCbm[i] != r.cbMatch, f1 = res.readCb0[i] != (i32)r.cbInd0, f2 = recordFlag(res.readAnn[i], res.readCbm[i]) != r.readFlag;
                bool f3 = false;
                if (r.cbMatch > 1) { nMulti++; f3 = res.readCb[i] != starCb[i]; }
                bad[0] += f0; bad[1] += f1; bad[2] += f2; bad[3] += f3;
                if ((f0 || f1 || f2 || f3) && shown++ < 6)
                    printf("  [read %llu] ours cbm=%d cb0=%d cb=%d | STAR cbm=%d cb0=%d posterior=%d\n", (unsigned long long)i, res.readCbm[i], res.readCb0[i],
                           res.readCb[i], r.cbMatch, (int)r.cbInd0, starCb[i]);
            }
            printf("[%s] per read vs STAR (%llu reads, %llu with several matches; library: %llu multi-match reads, %llu given a CB):", tag.c_str(),
                   (unsigned long long)N, (unsigned long long)nMulti, (unsigned long long)res.nMulti, (unsigned long long)res.nMultiAccepted);
            for (int k = 0; k < 4; k++) printf(" %s %llu differ;", what[k], (unsigned long long)bad[k]);
            printf("\n");
            // CellReads.stats rows in the order STAR's map got them (thread by thread, first occurrence in each thread's record order)
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
            const std::string od = outDir + md + "-" + run + "/star_Solo.out/";
            { std::string c = "mkdir -p " + od + "Gene/raw " + od + "Gene/filtered"; if (system(c.c_str()) != 0) {} }
            SoloWriteInfo wi; std::string wlog;
            if (!soloWrite(od, S1, M, res, SoloFilterParams{}, order, wi, wlog)) { fprintf(stderr, "%s", wlog.c_str()); return 1; }
            std::string rep;
            const char* fl[] = {"Barcodes.stats", "Gene/Features.stats", "Gene/raw/barcodes.tsv", "Gene/raw/features.tsv", "Gene/raw/matrix.mtx",
                                "Gene/filtered/barcodes.tsv", "Gene/filtered/features.tsv", "Gene/filtered/matrix.mtx", "Gene/UMIperCellSorted.txt",
                                "Gene/CellReads.stats"};
            int nId = 0, nF = 0;
            for (const char* f : fl) { nId += compareFile(od + f, truth + f, f, std::string(f) == "Gene/CellReads.stats", rep); nF++; }
            printf("[%s] files vs STAR's %s: %d of %d identical (nCB %llu, cells %llu)\n%s", tag.c_str(), truth.c_str(), nId, nF,
                   (unsigned long long)wi.nCB, (unsigned long long)wi.nCells, rep.c_str());
            if (run == "perread") allOk &= nId == nF && bad[0] + bad[1] + bad[2] + bad[3] == 0 && cbQualCrBad == 0;
        }
    }
    printf("[done] perread: %s\n", allOk ? "ALL IDENTICAL" : "DIFFERENCES");
    return allOk ? 0 : 3;
}
