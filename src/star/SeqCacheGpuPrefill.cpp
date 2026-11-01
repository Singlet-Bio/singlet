// `singlet --engine gpu`: fill STAR's sequence cache with mapping results computed on the GPU before mapping starts.
//
// Every read whose sequence is in the cache takes its mapping state from it (seqCacheRestore), exactly as for a cache hit in the
// CPU engine; every other read is mapped by STAR. The GPU results are STAR's own results for those sequences (W11: byte-identical
// on every tested sequence, 0 mismatches against STAR on a whole sample: docs/gpu-engine/exploration.md §12.12–12.14), so the
// engine's outputs are unchanged. Set by the singlet binary through SINGLET_ENGINE_GPU=1; SINGLET_STAR_SEQCACHE_VERIFY=1 makes
// STAR map every cached read anyway and count differences (the identity test, scripts/engine_gpu/identity_test.sh).
//
// The GPU path is used only where the cache key is the raw read sequence: single-end cDNA from plain files, no read clipping.
// Otherwise (and in a build without CUDA) it logs why and STAR maps every read, as in the CPU engine.
// the port's header first: STAR's IncludeDefine.h #defines uint, PC_*, EX_* and other names the port declares
#ifdef SINGLET_HAVE_ENGINE_GPU
#include <chrono>
#include "singlet/engine_gpu.h"
#include "stitch.cuh"
#endif
#include "SeqCache.h"
#include "Parameters.h"
#include "Genome.h"
#include "InOutStreams.h"
#include "ReadAlign.h"
#include "Transcriptome.h"
#include "OutSJ.h"
#include "SequenceFuns.h"
#include "ReadAlignChunk.h"
#include "GlobalVariables.h"
#include "Stats.h"
#include "star_api.h"
#include <atomic>
#include <cctype>
#include <cmath>
#include <sstream>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// E10 (SINGLET_ENGINE_GPU_E2E >= 3, docs/gpu-engine/fable/E10-hostloop.md): what STAR's per-read loop still does for a read with a GPU record
// (ReadAlign_oneRead.cpp, ReadAlign_outputAlignments.cpp). 0: everything (E8); 3: readLoad, the barcode and the Solo record only (the
// device computed its Log.final.out counters, junctions and Unmapped.out records); 4: such reads are not given to STAR at all
int g_e10Level = 0;

namespace {

#ifdef SINGLET_HAVE_ENGINE_GPU
// E10: what the host decode keeps for the host side of E10 (the unmapped reads' FASTQ records, level 4's read source): both FASTQ files
// mapped, per read the byte offset of its name line in each (0: R2, the cDNA; 1: R1, the barcode read)
struct E10Files {
    const char *m[2] = {nullptr, nullptr};
    uint64 len[2] = {0, 0};
    std::vector<uint64> at[2];
    void release() {
        for (int k = 0; k < 2; k++) {
            if (m[k] != nullptr) munmap((void *) m[k], len[k]);
            m[k] = nullptr; len[k] = 0; std::vector<uint64>().swap(at[k]);
        }
    }
};
E10Files e10f;
int e10MinQ = 0;
std::string e10DecodeWhy;   // why the decode turned E10 off
std::vector<uint8> e10sj;   // level >= 3: the device's junctions (Junction layout) for outputSJ, merged after mapping
uint64 e10nSJ = 0;
struct E10Check {           // SINGLET_ENGINE_GPU_E10_CHECK=1 at level 2: STAR's mapping threads write their entries here, compared after mapping
    bool on = false;
    std::vector<HandoffReadEntry> ent;
    std::vector<uint64> qm;
    const HandoffReadEntry *dev = nullptr;
    const uint64 *devQ = nullptr;
} e10chk;

// STAR's numeric code (SequenceFuns.cpp nucl_to_num)
inline char nucl(unsigned char c) {
    switch (c) { case 'A': case 'a': return 0; case 'C': case 'c': return 1; case 'G': case 'g': return 2; case 'T': case 't': return 3; default: return 4; }
}

int glScore(double scale, uint64 x) { return int(std::ceil(std::log2((double) x) * scale - 0.5)); }

bool fillParams(Parameters &P, m5::Params &Q, std::string &why) {
    Q = m5::Params{};
    Q.maxNsplit = P.maxNsplit; Q.seedSplitMin = P.seedSplitMin; Q.seedSearchStartLmax = P.seedSearchStartLmax;
    Q.seedSearchStartLmaxOverLread = P.seedSearchStartLmaxOverLread; Q.seedMapMin = P.seedMapMin; Q.seedSearchLmax = P.seedSearchLmax;
    Q.seedMultimapNmax = P.seedMultimapNmax; Q.seedPerReadNmax = P.seedPerReadNmax; Q.seedPerWindowNmax = P.seedPerWindowNmax;
    Q.winBinChrNbits = P.winBinChrNbits; Q.winBinNbits = P.winBinNbits; Q.winAnchorDistNbins = P.winAnchorDistNbins;
    Q.winFlankNbins = P.winFlankNbins; Q.winBinN = P.winBinN; Q.winAnchorMultimapNmax = P.winAnchorMultimapNmax;
    Q.alignWindowsPerReadNmax = P.alignWindowsPerReadNmax; Q.alignTranscriptsPerWindowNmax = P.alignTranscriptsPerWindowNmax;
    Q.alignTranscriptsPerReadNmax = P.alignTranscriptsPerReadNmax;
    Q.alignSJoverhangMin = P.alignSJoverhangMin; Q.alignSJDBoverhangMin = P.alignSJDBoverhangMin;
    Q.alignSplicedMateMapLmin = P.alignSplicedMateMapLmin; Q.alignSplicedMateMapLminOverLmate = P.alignSplicedMateMapLminOverLmate;
    Q.alignIntronMin = P.alignIntronMin; Q.alignIntronMax = P.alignIntronMax; Q.alignMatesGapMax = P.alignMatesGapMax;
    if (P.alignSJstitchMismatchNmax.size() != 4) { why = "alignSJstitchMismatchNmax needs 4 values"; return false; }
    for (int k = 0; k < 4; k++) Q.alignSJstitchMismatchNmax[k] = P.alignSJstitchMismatchNmax[k];
    Q.alignSoftClipAtReferenceEnds = P.alignSoftClipAtReferenceEnds.yes;
    for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) Q.alignEndsTypeExt[a][b] = P.alignEndsType.ext[a][b];
    Q.alignEndsProtrudeNbasesMax = P.alignEndsProtrude.nBasesMax; Q.flushRight = P.alignInsertionFlush.flushRight;
    const std::string &im = P.outFilterIntronMotifs;
    Q.intronMotifsFilter = im == "None" ? 0 : im == "RemoveNoncanonical" ? 1 : im == "RemoveNoncanonicalUnannotated" ? 2 : -1;
    Q.removeInconsistentStrands = P.outFilterIntronStrands == "RemoveInconsistentStrands";
    Q.outSAMstrandFieldType = P.outSAMstrandField.type;
    Q.scoreGenomicLengthLog2scale = P.scoreGenomicLengthLog2scale;
    Q.outFilterMultimapScoreRange = P.outFilterMultimapScoreRange; Q.outFilterMultimapNmax = P.outFilterMultimapNmax;
    Q.multOrderRandom = P.outMultimapperOrder.random; Q.outSAMmultNmax = P.outSAMmultNmax; Q.primaryAllBest = P.outSAMprimaryFlag == "AllBestScore";
    Q.outFilterScoreMin = P.outFilterScoreMin; Q.outFilterScoreMinOverLread = P.outFilterScoreMinOverLread;
    Q.outFilterMatchNmin = P.outFilterMatchNmin; Q.outFilterMatchNminOverLread = P.outFilterMatchNminOverLread;
    Q.outFilterMismatchNmax = P.outFilterMismatchNmax; Q.outFilterMismatchNoverLmax = P.outFilterMismatchNoverLmax;
    Q.outFilterMismatchNoverReadLmax = P.outFilterMismatchNoverReadLmax;
    Q.scoreGap = P.scoreGap; Q.scoreGapNoncan = P.scoreGapNoncan; Q.scoreGapGCAG = P.scoreGapGCAG; Q.scoreGapATAC = P.scoreGapATAC;
    Q.scoreDelBase = P.scoreDelBase; Q.scoreDelOpen = P.scoreDelOpen; Q.scoreInsBase = P.scoreInsBase; Q.scoreInsOpen = P.scoreInsOpen;
    Q.scoreStitchSJshift = P.scoreStitchSJshift;
    Q.sjdbScore = P.pGe.sjdbScore; Q.gSAsparseD = P.pGe.gSAsparseD; Q.gSAindexNbases = P.pGe.gSAindexNbases;
    // what the port covers (stitch.cuh header); anything else: STAR maps every read
    if (Q.multOrderRandom || Q.outSAMmultNmax != (uint64) -1 || Q.primaryAllBest) { why = "multimapper order / primary flag options"; return false; }
    if (Q.gSAsparseD > 4) { why = "genomeSAsparseD > 4"; return false; }
    if (Q.intronMotifsFilter < 0 || Q.outFilterMultimapScoreRange < 0 || Q.scoreGenomicLengthLog2scale > 0) { why = "filter options"; return false; }
    if (P.alignWindowsPerReadNmax > 10000 || P.seedPerWindowNmax > 50 || P.alignTranscriptsPerWindowNmax > 100 || P.seedPerReadNmax > 1000
        || P.maxNsplit > 50) { why = "STAR limits above the host lane's"; return false; }
    // genomic length score breakpoints, with this process's log2 (the value STAR computes)
    double s = Q.scoreGenomicLengthLog2scale;
    Q.glVal0 = glScore(s, 1); Q.glN = 0;
    uint64 x = 1; int v = Q.glVal0; const uint64 XMAX = 1ULL << 40;
    while (x < XMAX && s != 0) {
        uint64 lo = x, hi = x + 1;
        while (hi < XMAX && glScore(s, hi) == v) { lo = hi; hi = std::min(XMAX, hi * 2); }
        if (glScore(s, hi) == v) break;
        while (hi - lo > 1) { uint64 m = lo + (hi - lo) / 2; if (glScore(s, m) == v) lo = m; else hi = m; }
        if (Q.glN >= 64) { why = "genomic length score breakpoints"; return false; }
        v = glScore(s, hi); Q.glBreak[Q.glN] = hi; Q.glVal[Q.glN] = v; Q.glN++; x = hi;
    }
    for (uint64 xx = 1; xx < (1ULL << 22); xx += 1 + xx / 997)
        if (m5::genomicLengthScore(Q, xx) != glScore(s, xx)) { why = "genomic length score check"; return false; }
    return true;
}

// STAR's clipping of mate 1 for the GPU (engine_gpu.h Clip: the reads are clipped on the device before dedup, so the records are the
// clipped sequences STAR's cache is keyed by). Supported: --clipAdapterType Hamming (CellRanger4's 5' clip is per chunk, with opal).
bool fillClip(Parameters &P, singlet_engine_gpu::Clip &c, std::string &why) {
    const ReadClipInput &i5 = P.pClip.in[0], &i3 = P.pClip.in[1];
    auto at = [](const auto &v, auto d) { return v.empty() ? decltype(v[0] + d)(d) : decltype(v[0] + d)(v[0]); };
    c.n5 = at(i5.N, 0); c.after5 = at(i5.NafterAd, 0); c.n3 = at(i3.N, 0); c.after3 = at(i3.NafterAd, 0);
    const std::string ad5 = i5.adSeq.empty() ? "-" : i5.adSeq[0], ad3 = i3.adSeq.empty() ? "-" : i3.adSeq[0];
    // ClipMate::initialize: a side with no N bases and no adapter does not clip at all (its NafterAd is ignored too)
    if (c.n5 == 0 && ad5 == "-") c.after5 = 0;
    if (c.n3 == 0 && ad3 == "-") c.after3 = 0;
    c.on = c.n5 || c.after5 || c.n3 || c.after3 || ad5 != "-" || ad3 != "-";
    if (!c.on) return true;
    if (!P.pClip.adapterType.empty() && P.pClip.adapterType[0] == "CellRanger4") { c.pre = true; return true; }   // clip lengths from the host (readSequences)
    if (P.pClip.adapterType.empty() || P.pClip.adapterType[0] != "Hamming") { why = "read clipping with an unknown --clipAdapterType"; return false; }
    if (ad5 != "-") { why = "5' adapter clipping"; return false; }
    if (ad3 == "polyA") c.ad3.assign(DEF_readSeqLengthMax, 0);   // ClipMate::initialize: an all-A adapter
    else if (ad3 != "-") { c.ad3.resize(ad3.size()); convertNucleotidesToNumbers(ad3.data(), c.ad3.data(), c.ad3.size()); }
    c.adMMp = at(i3.adMMp, 0.0);
    return true;
}

// the cDNA reads (mate 1) of every input file, as STAR reads them (sequence line of each FASTQ record)
bool readSequences(Parameters &P, singlet_engine_gpu::Input &in, std::string &why, int nThreads) {
    if (P.readFilesNames.empty()) { why = "no read files"; return false; }
    for (const std::string &fn : P.readFilesNames[0]) {
        struct stat st;
        if (stat(fn.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) { why = fn + " is not a regular file (a stream cannot be read twice)"; return false; }
        int fd = open(fn.c_str(), O_RDONLY);
        if (fd < 0) { why = "cannot open " + fn; return false; }
        uint64 n = (uint64) st.st_size;
        if (n == 0) { close(fd); continue; }
        const char *m = (const char *) mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { close(fd); why = "cannot map " + fn; return false; }
        madvise((void *) m, n, MADV_SEQUENTIAL);
        // lines are numbered globally: each thread counts the newlines of its byte range, a prefix sum gives the index of its
        // first line, and it converts the sequence lines (index % 4 == 1) whose terminating newline lies in its range
        int T = std::max(1, nThreads);
        std::vector<uint64> b(T + 1), cnt(T, 0), mxT(T, 0);
        for (int t = 0; t <= T; t++) b[t] = n * t / T;
        const bool e10 = in.e10.on;   // E10: one file per mate (the caller checked)
        {
            std::vector<std::thread> th;
            for (int t = 0; t < T; t++) th.emplace_back([&, t]() {
                uint64 c = 0, mx = 0;
                const char *prev = m + b[t];
                for (const char *q = m + b[t], *e = m + b[t + 1]; (q = (const char *) memchr(q, '\n', e - q)) != nullptr; q++) {
                    c++;
                    if (e10) { mx = std::max<uint64>(mx, (uint64) (q - prev)); prev = q + 1; }
                }
                cnt[t] = c; mxT[t] = mx;
            });
            for (auto &x : th) x.join();
        }
        uint64 lines = 0;
        std::vector<uint64> first(T);
        for (int t = 0; t < T; t++) { first[t] = lines; lines += cnt[t]; }
        bool lastUnterminated = m[n - 1] != '\n';
        if (lastUnterminated) lines++;
        if (lines % 4 != 0) { munmap((void *) m, n); close(fd); why = fn + ": not 4-line FASTQ"; return false; }
        // E10: per read 2 + nw words (CR, UR from the barcode read: e10Barcodes; the quality bits of R2), the name-line offsets
        uint32 nw = 0;
        uint64 *aux = nullptr;
        std::vector<int> e10BadT(T, 0);
        if (e10) {
            uint64 mx = 0;
            for (int t = 0; t < T; t++) mx = std::max(mx, mxT[t]);
            nw = (uint32) std::max<uint64>(1, (mx + 63) / 64);
            if (nw > 4) { in.e10.on = false; in.e10.nw = 0; e10DecodeWhy = "cDNA reads over 256 bases"; }
            else {
                in.e10.nw = nw;
                in.e10.aux.assign((lines / 4) * (2 + (uint64) nw), 0);
                aux = in.e10.aux.data();
                e10f.at[0].assign(lines / 4, 0);
            }
        }
        const bool e10q = e10 && aux != nullptr;
        std::vector<std::vector<char>> seqT(T);
        std::vector<std::vector<uint64>> lenT(T);
        std::vector<std::vector<uint16>> p5T(T), p3T(T);   // --clipAdapterType CellRanger4: the clip of each read, by ClipMate::clip
        std::vector<std::vector<uint8>> infoT(T);
        {
            std::vector<std::thread> th;
            for (int t = 0; t < T; t++) th.emplace_back([&, t]() {
                uint64 li = first[t];
                const char *start = m + b[t];   // start of the line that ends at the first newline of this range
                while (start > m && start[-1] != '\n') start--;
                const char *e = m + b[t + 1];
                // sequence lines are under half of a FASTQ's bytes (the quality line is as long): no reallocation; bases through a
                // table instead of one push_back each (that loop was ~170 core-s for 66 M reads)
                seqT[t].reserve((b[t + 1] - b[t]) / 2 + 4096);
                char lut[256];
                for (int c = 0; c < 256; c++) lut[c] = nucl((unsigned char) c);
                // CellRanger4: STAR's ClipMate::clipChunk (opal alignment of the TSO, 64 reads at a time) and readLoad's ClipMate::clip, run here
                // once per read so the cache is keyed by the clipped sequences; the lengths go to the GPU and, as clippedInfo, to readLoad
                const bool cr4 = in.clip.pre;
                std::vector<std::vector<ClipMate>> cmv;
                if (cr4) P.pClip.initializeClipMates(cmv);
                std::vector<std::pair<const char *, uint64>> asc;
                std::vector<uint64> numAt;
                std::vector<char> cbuf(DEF_readSeqLengthMax + 16);
                auto flush = [&]() {
                    const int nb = (int) asc.size();
                    if (nb == 0) return;
                    ClipCR4 *cr = cmv[0][0].cr4;
                    for (int k = 0; k < nb; k++) cr->opalFillOneSeq(k, (char *) asc[k].first, (uint32) asc[k].second);
                    cr->opalAlign((uint8 *) cmv[0][0].adSeqNum.data(), cmv[0][0].adSeqNum.size(), nb);
                    for (int k = 0; k < nb; k++) {
                        const int Lm = cr->opalRes[k].endLocationTarget + 1, S = cr->opalRes[k].score;
                        const bool L0 = S < 20 || (S == 20 && Lm > 26) || (S == 21 && Lm > 30);
                        const uint8 info = (uint8) (L0 ? 0 : Lm);
                        const uint64 L = asc[k].second;
                        uint c5 = 0, c3 = 0;
                        if (L <= DEF_readSeqLengthMax) {
                            memcpy(cbuf.data(), seqT[t].data() + numAt[k], L);
                            uint Lread = (uint) L;
                            cmv[0][0].clippedInfo = (char) info;
                            c5 = cmv[0][0].clip(Lread, cbuf.data());
                            c3 = cmv[0][1].clip(Lread, cbuf.data());
                        }
                        p5T[t].push_back((uint16) std::min<uint>(c5, 65535)); p3T[t].push_back((uint16) std::min<uint>(c3, 65535)); infoT[t].push_back(info);
                    }
                    asc.clear(); numAt.clear();
                };
                auto take = [&](const char *ls, const char *le) {
                    uint64 L = le - ls;
                    while (L > 0 && ls[L - 1] == '\r') L--;
                    const size_t at = seqT[t].size();
                    seqT[t].resize(at + L);
                    char *d = seqT[t].data() + at;
                    for (uint64 k = 0; k < L; k++) d[k] = lut[(unsigned char) ls[k]];
                    lenT[t].push_back(L);
                    if (cr4) { asc.push_back({ls, L}); numAt.push_back(at); if (asc.size() == 64) flush(); }
                };
                // E10: the quality bits of R2 (STAR's handoffReadEntry: Qual0[0][j] - 33 < minQ), every base ACGTN (the device reads code 4 as
                // 'N' and rebuilds the read for the rRNA hook from the codes), the name-line offsets
                bool okc[256];
                for (int c = 0; c < 256; c++) okc[c] = c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N';
                auto e10line = [&](uint64 lix, const char *ls, const char *le) {
                    const uint64 r = lix >> 2;
                    if ((lix & 3) == 0) { e10f.at[0][r] = (uint64) (ls - m); return; }
                    uint64 L = le - ls;
                    if (L > 0 && (unsigned char) ls[L - 1] < 33) L--;   // fastqReadOneLine: one trailing control character
                    if ((lix & 3) == 1) {
                        int bad = 0;
                        for (uint64 k = 0; k < L; k++) bad |= !okc[(unsigned char) ls[k]];
                        if (bad) e10BadT[t] |= 1;
                    } else if ((lix & 3) == 3) {
                        if (L > 64 * (uint64) nw) { e10BadT[t] |= 2; return; }
                        uint64 *w = aux + r * (2 + (uint64) nw) + 2;
                        for (uint64 j = 0; j < L; j++)
                            if ((int) (unsigned char) ls[j] - 33 < e10MinQ) w[j >> 6] |= 1ull << (j & 63);
                    }
                };
                for (const char *q = m + b[t]; (q = (const char *) memchr(q, '\n', e - q)) != nullptr; q++) {
                    if ((li & 3) == 1) take(start, q);
                    if (e10q) e10line(li, start, q);
                    li++;
                    start = q + 1;
                }
                if (t == T - 1 && lastUnterminated && (li & 3) == 1) take(start, m + n);
                if (t == T - 1 && lastUnterminated && e10q) e10line(li, start, m + n);
                if (cr4) flush();
            });
            for (auto &x : th) x.join();
        }
        {   // concatenate the threads' pieces in parallel (a serial copy of several GB took ~15 s on one core)
            std::vector<uint64> seqAt(T + 1), offAt(T + 1);
            seqAt[0] = in.seq.size(); offAt[0] = in.off.size();
            for (int t = 0; t < T; t++) { seqAt[t + 1] = seqAt[t] + seqT[t].size(); offAt[t + 1] = offAt[t] + lenT[t].size(); }
            const uint64 base = in.off.back();
            in.seq.resize(seqAt[T]);
            in.off.resize(offAt[T]);
            std::vector<std::thread> th;
            for (int t = 0; t < T; t++) th.emplace_back([&, t]() {
                if (!seqT[t].empty()) memcpy(in.seq.data() + seqAt[t], seqT[t].data(), seqT[t].size());
                uint64 o = base + (seqAt[t] - seqAt[0]);
                for (uint64 k = 0; k < lenT[t].size(); k++) { o += lenT[t][k]; in.off[offAt[t] + k] = o; }
                std::vector<char>().swap(seqT[t]); std::vector<uint64>().swap(lenT[t]);
            });
            for (auto &x : th) x.join();
        }
        if (in.clip.pre)
            for (int t = 0; t < T; t++) {
                in.pre5.insert(in.pre5.end(), p5T[t].begin(), p5T[t].end());
                in.pre3.insert(in.pre3.end(), p3T[t].begin(), p3T[t].end());
                in.cr4info.insert(in.cr4info.end(), infoT[t].begin(), infoT[t].end());
            }
        if (e10q) {
            int bad = 0;
            for (int t = 0; t < T; t++) bad |= e10BadT[t];
            if (bad) { in.e10.on = false; std::vector<uint64>().swap(in.e10.aux); e10f.release();
                       e10DecodeWhy = (bad & 1) ? "a cDNA base other than ACGTN" : "a quality line longer than 256"; }
        }
        if (e10q && in.e10.on) { e10f.m[0] = m; e10f.len[0] = n; }   // E10 keeps R2 mapped (the unmapped reads' records, level 4's source)
        else munmap((void *) m, n);
        close(fd);
    }
    return true;
}

// E10: per read the CR and UR (HandoffReadEntry's 3-bit packing of SoloReadBarcode::getCBandUMI's cbSeq, umiSeq for CB_UMI_Simple) from
// the barcode read, into the aux words readSequences sized; R1 stays mapped and its name-line offsets are kept
bool e10Barcodes(Parameters &P, singlet_engine_gpu::Input &in, std::string &why, int nThreads) {
    const std::string &fn = P.readFilesNames[1][0];
    struct stat st;
    if (stat(fn.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) { why = fn + " is not a regular file"; return false; }
    int fd = open(fn.c_str(), O_RDONLY);
    if (fd < 0) { why = "cannot open " + fn; return false; }
    const uint64 n = (uint64) st.st_size;
    const uint32 nw = in.e10.nw;
    const uint64 nReads = in.e10.aux.size() / (2 + (uint64) nw);
    if (n == 0) { close(fd); why = fn + " is empty"; return false; }
    const char *m = (const char *) mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { why = "cannot map " + fn; return false; }
    madvise((void *) m, n, MADV_SEQUENTIAL);
    const int T = std::max(1, nThreads);
    std::vector<uint64> b(T + 1), cnt(T, 0), first(T);
    for (int t = 0; t <= T; t++) b[t] = n * t / T;
    {
        std::vector<std::thread> th;
        for (int t = 0; t < T; t++) th.emplace_back([&, t]() {
            uint64 c = 0;
            for (const char *q = m + b[t], *e = m + b[t + 1]; (q = (const char *) memchr(q, '\n', e - q)) != nullptr; q++) c++;
            cnt[t] = c;
        });
        for (auto &x : th) x.join();
    }
    uint64 lines = 0;
    for (int t = 0; t < T; t++) { first[t] = lines; lines += cnt[t]; }
    const bool lastUnterminated = m[n - 1] != '\n';
    if (lastUnterminated) lines++;
    if (lines != 4 * nReads) { munmap((void *) m, n); why = "the barcode read file has " + std::to_string(lines / 4) + " reads, the cDNA file " + std::to_string(nReads); return false; }
    e10f.at[1].assign(nReads, 0);
    const uint64 bL = P.pSolo.bL, cbumiL = P.pSolo.cbumiL, cbS = P.pSolo.cbS, cbL = P.pSolo.cbL, umiS = P.pSolo.umiS, umiL = P.pSolo.umiL;
    std::vector<int> badT(T, 0);
    uint64 *aux = in.e10.aux.data();
    {
        std::vector<std::thread> th;
        for (int t = 0; t < T; t++) th.emplace_back([&, t]() {
            uint64 li = first[t];
            const char *start = m + b[t];
            while (start > m && start[-1] != '\n') start--;
            const char *e = m + b[t + 1];
            auto line = [&](const char *ls, const char *le) {
                const uint64 r = li >> 2;
                if ((li & 3) == 0) { e10f.at[1][r] = (uint64) (ls - m); return; }
                if ((li & 3) != 1) return;
                uint64 L = le - ls;
                if (L > 0 && (unsigned char) ls[L - 1] < 33) L--;
                if (bL > 0 && L != bL) { badT[t] = 1; return; }   // STAR exits on this read
                const uint64 S = (bL == 0 && L < cbumiL) ? cbumiL : L;   // getCBandUMI: padded with N to cbumiL
                if (cbS - 1 > S || umiS - 1 > S) { badT[t] = 1; return; }   // substr would throw
                auto pack = [&](uint64 s0, uint64 len) {
                    const uint64 e0 = std::min(s0 + len, S);
                    uint64 v = 0;
                    for (uint64 j = s0; j < e0 && j - s0 < 21; j++) {
                        const char c = j < L ? ls[j] : 'N';
                        v |= (uint64) (c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4) << (3 * (j - s0));
                    }
                    return v;
                };
                uint64 *a = aux + r * (2 + (uint64) nw);
                a[0] = pack(cbS - 1, cbL); a[1] = pack(umiS - 1, umiL);
            };
            for (const char *q = m + b[t]; (q = (const char *) memchr(q, '\n', e - q)) != nullptr; q++) {
                line(start, q);
                li++;
                start = q + 1;
            }
            if (t == T - 1 && lastUnterminated) line(start, m + n);
        });
        for (auto &x : th) x.join();
    }
    for (int t = 0; t < T; t++) if (badT[t]) { munmap((void *) m, n); why = "a barcode read STAR would reject (length)"; return false; }
    e10f.m[1] = m; e10f.len[1] = n;
    return true;
}

// E10: STAR's view of read r's FASTQ record in mate file k: the name line's first token (processChunks: readID, removeStringEndControl) and
// passFilterIllumina from R2's name line, the sequence and quality lines without one trailing control character (fastqReadOneLine)
struct E10Rec { const char *id; uint64 idLen; char filter; const char *seq[2]; uint64 seqLen[2]; const char *qual[2]; uint64 qualLen[2]; };
inline bool e10Record(uint64 r, E10Rec &x) {
    for (int k = 0; k < 2; k++) {
        const char *m = e10f.m[k], *end = m + e10f.len[k];
        const char *l0 = m + e10f.at[k][r];
        const char *l1 = (const char *) memchr(l0, '\n', end - l0);
        if (l1 == nullptr) return false;
        if (k == 0) {
            const char *p = l0;
            while (p < l1 && isspace((unsigned char) *p)) p++;
            const char *q = p;
            while (q < end && !isspace((unsigned char) *q)) q++;
            x.id = p; x.idLen = q - p;
            if (x.idLen > 0 && (unsigned char) p[x.idLen - 1] < 33) x.idLen--;
            x.filter = 'N';
            if (q < end && *q != '\n') {   // the second field: >> skips whitespace (newlines too) and reads a token
                while (q < end && isspace((unsigned char) *q)) q++;
                const char *f = q;
                while (q < end && !isspace((unsigned char) *q)) q++;
                if (q - f >= 4 && f[1] == ':' && f[2] == 'Y' && f[3] == ':') x.filter = 'Y';
            }
        }
        const char *s0 = l1 + 1, *s1 = (const char *) memchr(s0, '\n', end - s0);
        if (s1 == nullptr) return false;
        const char *p0 = s1 + 1, *p1 = (const char *) memchr(p0, '\n', end - p0);
        if (p1 == nullptr) return false;
        const char *q0 = p1 + 1, *q1 = (const char *) memchr(q0, '\n', end - q0);
        if (q1 == nullptr) q1 = end;
        uint64 ls = s1 - s0, lq = q1 - q0;
        if (ls > 0 && (unsigned char) s0[ls - 1] < 33) ls--;
        if (lq > 0 && (unsigned char) q0[lq - 1] < 33) lq--;
        x.seq[k] = s0; x.seqLen[k] = ls; x.qual[k] = q0; x.qualLen[k] = lq;
    }
    return true;
}
// the read name as readLoad keeps it: readID (or @iReadAll with outSAMreadIDnumber), cut at the first readNameSeparatorChar
inline std::string e10ReadName(Parameters &P, uint64 r, const E10Rec &x, bool trim) {
    std::string id = P.outSAMreadIDnumber ? "@" + std::to_string(r + 1) : std::string(x.id, x.idLen);
    if (trim)
        for (char c : P.readNameSeparatorChar) { const size_t k = id.find(c); if (k != std::string::npos) id.resize(k); }
    return id;
}

// E10 level >= 3: the Unmapped.out.mate1/2 records of the reads whose record is unmapped (ReadAlign::outReadsUnmapped, single-end + barcode
// read), formatted on all threads, appended to STAR's streams before the mapping threads start (the files' order is run-dependent anyway)
bool e10WriteUnmapped(Parameters &P, const std::vector<uint32_t> &reads, int nThreads, std::string &why) {
    const int T = std::max(1, nThreads);
    const uint64 N = reads.size();
    std::vector<std::string> out[2];
    out[0].resize(T); out[1].resize(T);
    std::vector<int> bad(T, 0);
    std::vector<std::thread> th;
    for (int t = 0; t < T; t++) th.emplace_back([&, t]() {
        const uint64 k0 = N * t / T, k1 = N * (t + 1) / T;
        out[0][t].reserve((k1 - k0) * 240); out[1][t].reserve((k1 - k0) * 90);
        E10Rec x;
        for (uint64 k = k0; k < k1; k++) {
            const uint64 r = reads[k];
            if (!e10Record(r, x)) { bad[t] = 1; return; }
            const std::string nm = e10ReadName(P, r, x, true);
            for (int im = 0; im < 2; im++) {
                std::string &o = out[im][t];
                o += nm; o += ' '; o += (char) ('0' + im); o += ':'; o += x.filter; o += ": \n";
                o.append(x.seq[im], x.seqLen[im]); o += "\n+\n"; o.append(x.qual[im], x.qualLen[im]); o += '\n';
            }
        }
    });
    for (auto &x : th) x.join();
    for (int t = 0; t < T; t++) if (bad[t]) { why = "a FASTQ record could not be read back"; return false; }
    for (int im = 0; im < 2; im++)
        for (int t = 0; t < T; t++) { P.inOut->outUnmappedReadsStream[im].write(out[im][t].data(), out[im][t].size()); std::string().swap(out[im][t]); }
    return true;
}

// E10 level 4: STAR's read source (star_api.h) with only the reads that have no GPU record, in chunk format (processChunks' FASTQ branch)
struct E10Source : StarReadSource {
    std::string buf[2];
    std::vector<uint64> end[2], iRead;   // per served read: where it ends in buf[0], buf[1]; its iReadAll
    uint64 k = 0;
    void fill(char **chunkIn, uint64_t *sz, uint64_t limit, uint64_t &iReadAll, uint64_t readMapNumber, uint32_t) override {
        while (k < iRead.size() && sz[0] < limit && sz[1] < limit) {
            if (iRead[k] > readMapNumber) { k = iRead.size(); break; }
            for (int im = 0; im < 2; im++) {
                const uint64 a = k == 0 ? 0 : end[im][k - 1], l = end[im][k] - a;
                memcpy(chunkIn[im] + sz[im], buf[im].data() + a, l);
                sz[im] += l;
            }
            iReadAll = iRead[k];
            k++;
        }
    }
};
E10Source e10src;
bool e10BuildSource(Parameters &P, const uint32 *readRecord, uint64 nReads, std::string &why) {
    E10Rec x;
    for (uint64 r = 0; r < nReads; r++) {
        if (readRecord[r] != 0xFFFFFFFFu) continue;
        if (!e10Record(r, x)) { why = "a FASTQ record could not be read back"; return false; }
        const std::string line1 = e10ReadName(P, r, x, false) + ' ' + std::to_string(r + 1) + ' ' + x.filter + ' ' + std::to_string(P.readFilesIndex) + '\n';
        for (int im = 0; im < 2; im++) {
            std::string &o = e10src.buf[im];
            o += line1; o.append(x.seq[im], x.seqLen[im]); o += "\n+\n"; o.append(x.qual[im], x.qualLen[im]); o += '\n';
            e10src.end[im].push_back(o.size());
        }
        e10src.iRead.push_back(r + 1);
    }
    return true;
}
#endif

} // namespace

void seqCacheGpuPrefill(Parameters &P, Genome &G, SeqCache *&cache, Transcriptome *tr) {
    const char *e = getenv("SINGLET_ENGINE_GPU");
    if (e == nullptr || e[0] != '1' || cache == nullptr)
        return;
#ifndef SINGLET_HAVE_ENGINE_GPU
    P.inOut->logMain << "ENGINE_GPU: requested, but this binary was built without CUDA (SINGLET_ENGINE_GPU=OFF): STAR maps every read\n" << flush;
    (void) G; (void) tr;
#else
    auto skip = [&](const std::string &why) {
        P.inOut->logMain << "ENGINE_GPU: not used (" << why << "): STAR maps every read\n" << flush;
    };
    std::string why;
    if (!singlet_engine_gpu::available(&why)) return skip(why);
    if (P.readNmates != 1) return skip("paired-end alignment");
    if (P.readFilesCommandString != "" && P.readFilesCommandString != "-") return skip("readFilesCommand");
    m5::Params Q;
    if (!fillParams(P, Q, why)) return skip(why);
    m5::GenomeView H{};
    H.G = G.G; H.SA = (const m5::u64 *) G.SA.charArray; H.SAi = (const m5::u64 *) G.SAi.charArray;
    H.nGenome = G.nGenome; H.nSA = G.nSA; H.GstrandBit = G.GstrandBit; H.GstrandMask = G.GstrandMask;
    H.saBits = G.SA.wordLength; H.saiBits = G.SAi.wordLength;
    H.SAiMarkNmaskC = G.SAiMarkNmaskC; H.SAiMarkNmask = G.SAiMarkNmask; H.SAiMarkAbsentMaskC = G.SAiMarkAbsentMaskC;
    for (uint ii = 0; ii <= P.pGe.gSAindexNbases && ii < 32; ii++) H.saiStart[ii] = G.genomeSAindexStart[ii];
    std::vector<uint64> chrStart(G.chrStart.begin(), G.chrStart.end()), chrLength(G.chrLength.begin(), G.chrLength.end());
    std::vector<uint64> chrBin(G.chrBin, G.chrBin + G.chrBinN);
    H.chrStart = chrStart.data(); H.chrLength = chrLength.data(); H.chrBin = chrBin.data();
    H.nChr = chrLength.size(); H.chrBinN = chrBin.size();
    H.sjGstart = G.sjGstart; H.sjdbOverhang = G.sjdbOverhang; H.sjdbLength = G.sjdbLength; H.sjdbN = G.sjdbN;
    H.sjdbStart = G.sjdbStart; H.sjdbEnd = G.sjdbEnd; H.sjDstart = G.sjDstart; H.sjAstart = G.sjAstart;
    H.sjdbMotif = G.sjdbMotif; H.sjdbShiftLeft = G.sjdbShiftLeft; H.sjdbShiftRight = G.sjdbShiftRight; H.sjdbStrand = G.sjdbStrand;
    if (chrStart.size() < chrLength.size() + 1) return skip("chrStart size");
    time_t t0 = time(nullptr);
    singlet_engine_gpu::Input in;
    if (!fillClip(P, in.clip, why)) return skip(why);
    // E10 (SINGLET_ENGINE_GPU_E2E >= 2): where E8 runs (e2eWant's conditions below) on one cDNA and one barcode FASTQ file, the decode also
    // keeps per read the CR/UR key and the quality bits the device builds the hand-off entries from
    const int e10Level = singlet_engine_gpu::e2eLevel();
    const char *e10ChkE = getenv("SINGLET_ENGINE_GPU_E10_CHECK");
    {
        std::string w;
        const char *gp = getenv("SINGLET_GPU_PILEUP"), *hs = getenv("SINGLET_GPU_HANDOFF_SUMMARY"), *mq = getenv("SINGLET_ENGINE_GPU_HANDOFF_MINQ");
        const char *qe = getenv("SINGLET_HANDOFF_QMASK");
        if (e10Level < 2) w = "";
        else if (!(singlet_engine_gpu::e2eRegion().fd >= 0 && getenv("SINGLET_ENGINE_GPU_HANDOFF") != nullptr && mq != nullptr && gp != nullptr && strcmp(gp, "on") == 0
                   && !(hs != nullptr && hs[0] == '0') && ReadAlign::seqCacheHandoffConfig(P) && (tr != nullptr || !P.quant.gene.yes)
                   && P.pSolo.type == P.pSolo.SoloTypes::CB_UMI_Simple && P.pSolo.cbL <= 21 && P.pSolo.umiL <= 21)) w = "E8 does not run";
        else if (P.readFilesNames.size() != 2 || P.readFilesNames[0].size() != 1 || P.readFilesNames[1].size() != 1 || P.readNends != 2) w = "not one cDNA and one barcode FASTQ file";
        else if (P.readFilesTypeN == 10 || P.pSolo.barcodeRead != 1 || P.pSolo.CBtype.type != 1) w = "barcode layout";
        else if (P.outQSconversionAdd != 0) w = "outQSconversionAdd";
        else if (P.outFilterBySJoutStage != 0) w = "outFilterType BySJout";
        else if (P.pGe.transform.outSAM || P.pGe.transform.outYes || P.pGe.transform.outQuant) w = "genome transform";
        else {
            in.e10.on = true;
            e10MinQ = atoi(mq);
            in.e10.wantMask = !(qe != nullptr && qe[0] == '0');
            in.e10.outSJ = P.outSJ.yes; in.e10.sjAll = P.outSJfilterReads == "All";
            in.e10.hook = g_handoffReadHook != nullptr;
        }
        if (e10Level >= 2 && !in.e10.on) P.inOut->logMain << "ENGINE_GPU: E10 (level " << e10Level << ") not used: " << w << "\n";
    }
    if (!readSequences(P, in, why, (int) P.runThreadN)) return skip(why);
    if (e10Level >= 2 && in.e10.on) {
        std::string w;
        const auto tb = std::chrono::steady_clock::now();
        if (!e10Barcodes(P, in, w, (int) P.runThreadN)) { in.e10.on = false; std::vector<uint64>().swap(in.e10.aux); e10f.release(); }
        P.inOut->logMain << "ENGINE_GPU: E10 decode: " << (in.e10.on ? "CR/UR keys and quality bits of " + std::to_string(in.off.size() - 1) + " reads ("
                                                        + std::to_string(in.e10.nw) + " quality words each), barcode file "
                                                        + std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count()) + " s"
                                                      : "not used: " + w) << "\n";
    } else if (e10Level >= 2 && !in.e10.on) {
        if (!e10DecodeWhy.empty()) P.inOut->logMain << "ENGINE_GPU: E10 decode: not used: " << e10DecodeWhy << "\n";
        e10f.release();
    }
    time_t tRead = time(nullptr);
    // one sample at a time per GPU (several samples in flight per node): singlet_engine_gpu::lockDevice
    time_t tWait = time(nullptr);
    int lockFd = -1;
    int device = singlet_engine_gpu::lockDevice(&lockFd);
    time_t tLock = time(nullptr);
    int hostThreads = (int) P.runThreadN;
    singlet_engine_gpu::Output out;
    // E8 (SINGLET_ENGINE_GPU_E2E=1, docs/gpu-engine/fable/E8-e2e.md): the pipeline mapped a shared region before forking STAR (e2eRegion);
    // mapDistinct builds the GPU pileup's records on the device into it, and the hand-off file and its record bytes are not made. Only
    // where the hand-off "on" path with summaries and per-read entries runs; otherwise (or when the device build fails) that path runs as before.
    singlet_engine_gpu::E2ERegion &e2r = singlet_engine_gpu::e2eRegion();
    const char *gpE = getenv("SINGLET_GPU_PILEUP"), *hsE = getenv("SINGLET_GPU_HANDOFF_SUMMARY");
    bool e2eWant = e2r.fd >= 0 && singlet_engine_gpu::e2eRequested() && getenv("SINGLET_ENGINE_GPU_HANDOFF") != nullptr
        && getenv("SINGLET_ENGINE_GPU_HANDOFF_MINQ") != nullptr && gpE != nullptr && strcmp(gpE, "on") == 0 && !(hsE != nullptr && hsE[0] == '0')
        && ReadAlign::seqCacheHandoffConfig(P) && (tr != nullptr || !P.quant.gene.yes)
        && P.pSolo.type == P.pSolo.SoloTypes::CB_UMI_Simple && P.pSolo.cbL <= 21 && P.pSolo.umiL <= 21;
    if (e2eWant) {   // the memory object sized (sparse: only written pages exist) and mapped here; SINGLET_ENGINE_GPU_E2E_GB, default 512
        uint64 gb = 512;
        if (const char *g = getenv("SINGLET_ENGINE_GPU_E2E_GB")) gb = std::max<uint64>(1, strtoull(g, nullptr, 10));
        if (!e2r.attach(gb << 30)) { e2eWant = false; P.inOut->logMain << "ENGINE_GPU: E8 not used (the shared memory object could not be mapped)" << std::endl; }
    }
    if (e2eWant) { in.e2e = &e2r; in.outSAMmapqUnique = P.outSAMmapqUnique; in.outSAMflagAND = P.outSAMflagAND; in.outSAMflagOR = P.outSAMflagOR; }
    else if (e2r.fd >= 0 && e2r.base == nullptr) P.inOut->logMain << "ENGINE_GPU: E8 not used (it needs the GPU pileup \"on\" with summaries and per-read entries, CB_UMI_Simple)" << std::endl;
    bool lockReleased = false;   // mapDistinct releases the device when its device work is done (the host assembly of the records needs no GPU)
    const auto tHeld0 = std::chrono::steady_clock::now(); double heldS = 0;
    in.release = [&]() { if (!lockReleased) { lockReleased = true; heldS = std::chrono::duration<double>(std::chrono::steady_clock::now() - tHeld0).count(); singlet_engine_gpu::unlockDevice(lockFd); } };
    bool ok = singlet_engine_gpu::mapDistinct(Q, H, G.SA.lengthByte, G.SAi.lengthByte, in, device, hostThreads, out);
    in.release();
    time_t tGpu = time(nullptr);
    P.inOut->logMain << "ENGINE_GPU: device " << device << " (" << singlet_engine_gpu::deviceId(device) << "), waited " << (tLock - tWait)
                     << " s for it, held it " << heldS << " s\n" << out.log << out.e10.log;
    std::vector<uint64>().swap(in.e10.aux);
    if (!ok) { e10f.release(); return skip("GPU mapping failed"); }
    bool e10ok = out.e10.ok;   // E10: the device wrote the entries and masks into the region and computed the per-read products
    if (in.clip.pre) {   // CellRanger4: readLoad takes clippedInfo from here, clipChunk is skipped
        static std::vector<uint8> cr4keep;
        cr4keep.swap(in.cr4info);
        g_cr4Info = cr4keep.data(); g_cr4InfoN = cr4keep.size();
    }
    // a cache large enough for every result (entries are ~(32 + L/4 + blob) bytes; 3x for the ring, the 2-way index and
    // the shards' imbalance), unless SINGLET_STAR_SEQCACHE_MB was set
    uint64 nRec = out.seqOff.size() - 1;
    if (getenv("SINGLET_STAR_SEQCACHE_MB") == nullptr) {
        uint64 need = (out.blob.size() + out.seq.size() / 4 + 40 * nRec) * 3;
        uint64 mb = std::max<uint64>(4096, (need >> 20) + 1);
        setenv("SINGLET_STAR_SEQCACHE_MB", std::to_string(mb).c_str(), 1);
        SeqCache *c2 = SeqCache::create(P, G.nGenome);
        if (c2 != nullptr) {
            c2->verify = cache->verify;
            delete cache;
            cache = c2;
        };
    };
    // insert, sharded by hash across threads (SeqCache is thread-safe per shard); keep(i) picks the records
    auto insertRecords = [&](auto keep) {
        int nt = std::max(1, (int) P.runThreadN);
        std::vector<std::thread> th;
        for (int t = 0; t < nt; t++) th.emplace_back([&, t]() {
            for (uint64 i = t; i < nRec; i += nt) {
                if (!keep(i)) continue;
                const char *s = out.seq.data() + out.seqOff[i];
                uint64 L = out.seqOff[i + 1] - out.seqOff[i];
                cache->insert(SeqCache::hashSeq(s, L), s, L, out.blob.data() + out.blobOff[i], out.blobOff[i + 1] - out.blobOff[i]);
            }
        });
        for (auto &x : th) x.join();
    };
    // GPU pileup "on" with hand-off summaries: reads of a mapped sequence replay its summary and never look it up, so only the
    // unmapped sequences go into the cache (after the hand-off, which tells which they are)
    const char *gpEnv = getenv("SINGLET_GPU_PILEUP"), *hsEnv = getenv("SINGLET_GPU_HANDOFF_SUMMARY");
    const bool deferInsert = getenv("SINGLET_ENGINE_GPU_HANDOFF") != nullptr && gpEnv != nullptr && strcmp(gpEnv, "on") == 0
        && !(hsEnv != nullptr && hsEnv[0] == '0') && ReadAlign::seqCacheHandoffConfig(P) && (tr != nullptr || !P.quant.gene.yes);
    if (!deferInsert) {
        insertRecords([](uint64) { return true; });
        cache->prefilled = true;
    }
    // singlet's GPU pileup (SINGLET_GPU_PILEUP): per read its record, per record the alignments STAR writes for it (core fields)
    if (const char *hp = getenv("SINGLET_ENGINE_GPU_HANDOFF")) {
        time_t th0 = time(nullptr);
        const int nt = std::max(1, (int) P.runThreadN);
        std::vector<std::vector<uint8>> part(nt);
        std::vector<std::vector<uint64>> partOff(nt);
        // "on": per record also the summary STAR's per-read output replays for its reads (ReadAlign::seqCacheHandoffSummary)
        const char *gm0 = getenv("SINGLET_GPU_PILEUP"), *hs = getenv("SINGLET_GPU_HANDOFF_SUMMARY");
        const bool sumOn = gm0 != nullptr && strcmp(gm0, "on") == 0 && !(hs != nullptr && hs[0] == '0')
            && ReadAlign::seqCacheHandoffConfig(P) && (tr != nullptr || !P.quant.gene.yes);
        std::vector<std::vector<uint8>> sumPart(nt);
        std::vector<std::vector<uint64>> sumPartOff(nt);
        // E8: the records are in the shared region (mapDistinct): readRecord, the per-read entries and masks are reserved next to them (else
        // the file path runs), and per record only its summary is made here
        bool e2eRec = out.e2eOk && sumOn;
        uint64 e2eR = 0, e2eE = 0, e2eQ = 0;
        const char *e2eQe = getenv("SINGLET_HANDOFF_QMASK");
        const bool e2eQOn = !(e2eQe != nullptr && e2eQe[0] == '0');
        if (e2eRec) {
            const uint64 nR = out.readRecord.size();
            e2eR = e2r.take(std::max<uint64>(1, nR) * 4);
            if (e10ok && (out.e10.oQMask != 0) == e2eQOn) { e2eE = out.e10.oEntries; e2eQ = out.e10.oQMask; }   // E10: written by the device
            else {
                e10ok = false;
                e2eE = e2r.take(std::max<uint64>(1, nR) * sizeof(HandoffReadEntry));
                e2eQ = e2eQOn ? e2r.take(std::max<uint64>(1, nR) * 16) : 0;
            }
            if (!e2eR || !e2eE || (e2eQOn && !e2eQ)) { e2eRec = false; P.inOut->logMain << "ENGINE_GPU: E8: the shared region is full; the hand-off file path runs\n"; }
            else memcpy(e2r.base + e2eR, out.readRecord.data(), nR * 4);
        }
        e10ok = e10ok && e2eRec;
        // E10 level 4: no read with a record reaches STAR's loop, so no summaries are needed
        const bool e10NoSum = e10ok && e10Level >= 4 && g_starReadSource == nullptr;
        std::vector<std::thread> th;
        for (int t = 0; t < nt; t++) th.emplace_back([&, t]() {
            Transcriptome *trT = (sumOn && tr != nullptr) ? new Transcriptome(*tr) : nullptr;
            ReadAlign *ra = new ReadAlign(P, G, trT, t);
            if (sumOn) ra->chunkOutSJ = new OutSJ(P.limitOutSJoneRead, P, G);   // scratch: one sequence's junctions at a time
            const uint64 r0 = nRec * t / nt, r1 = nRec * (t + 1) / nt;
            // ~70 bytes per record with one alignment (pbmc1k_v3: 2.3 GB for 31.8 M records): reserve, so the part grows without copying
            // GBs at each doubling (untouched capacity costs no memory)
            if (!e2eRec) { part[t].reserve((size_t) (r1 - r0) * 80); partOff[t].reserve((size_t) (r1 - r0)); }
            if (sumOn) { sumPartOff[t].reserve((size_t) (r1 - r0)); sumPart[t].reserve((size_t) (r1 - r0) * 40); }   // 1.06 GB for 31.8 M records
            for (uint64 r = r0; r < (e10NoSum ? r0 : r1); r++) {
                if (e2eRec) ra->seqCacheRestoreRecord(out.seq.data() + out.seqOff[r], out.seqOff[r + 1] - out.seqOff[r], out.blob.data() + out.blobOff[r]);
                else {
                partOff[t].push_back(part[t].size());
                ra->seqCacheAlignRecords(out.seq.data() + out.seqOff[r], out.seqOff[r + 1] - out.seqOff[r], out.blob.data() + out.blobOff[r], part[t]);
                }
                if (sumOn) {
                    sumPartOff[t].push_back(sumPart[t].size());
                    ra->seqCacheHandoffSummary(sumPart[t]);
                }
            }
        });
        for (auto &x : th) x.join();
        if (g_handoffLeanAln.load() > 0)   // SINGLET_HANDOFF_LEAN=verify
            P.inOut->logMain << "ENGINE_GPU: lean alignment fields: " << g_handoffLeanAln.load() << " alignments, " << g_handoffLeanBad.load()
                             << " differ from alignBAM's" << "\n";
        static std::vector<uint8> recAln;   // per record its alignment count (the first byte of each record)
        recAln.resize(nRec);
        if (e2eRec) for (uint64 r = 0; r < nRec; r++) recAln[r] = e2r.recAln(r);
        else
        for (int t = 0, r = 0; t < nt; t++)
            for (uint64 o : partOff[t]) recAln[r++] = part[t][o];
        FILE *f = e2eRec ? nullptr : fopen(hp, "wb");
        bool okw = e2eRec || f != nullptr;
        if (e2eRec) P.inOut->logMain << "ENGINE_GPU: E8: no hand-off file: " << nRec << " records built on the device, in the shared region; summaries in "
                                     << (time(nullptr) - th0) << " s\n";
        if (f != nullptr) {
            const char magic[8] = {'S', 'G', 'H', 'O', 'F', 'F', '0', '2'};
            uint64 nReads = out.readRecord.size();
            // STAR's barcode layout (SoloReadBarcode::getCBandUMI, CB_UMI_Simple): CR = bSeq[cbS-1, +cbL), UR = bSeq[umiS-1, +umiL), the
            // barcode read padded with N to cbumiL when shorter
            uint32 lay[6] = {(uint32) P.pSolo.cbS, (uint32) P.pSolo.cbL, (uint32) P.pSolo.umiS, (uint32) P.pSolo.umiL, (uint32) P.pSolo.cbumiL,
                             (uint32) (P.pSolo.type == P.pSolo.SoloTypes::CB_UMI_Simple ? 1 : 0)};
            okw = fwrite(magic, 1, 8, f) == 8 && fwrite(&nReads, 8, 1, f) == 1 && fwrite(&nRec, 8, 1, f) == 1 && fwrite(lay, 4, 6, f) == 6
                && fwrite(out.readRecord.data(), 4, nReads, f) == nReads;
            uint64 base = 0;
            for (int t = 0; t < nt && okw; t++) {
                for (uint64 &o : partOff[t]) o += base;
                okw = partOff[t].empty() || fwrite(partOff[t].data(), 8, partOff[t].size(), f) == partOff[t].size();
                base += part[t].size();
            }
            okw = okw && fwrite(&base, 8, 1, f) == 1;
            for (int t = 0; t < nt && okw; t++) okw = part[t].empty() || fwrite(part[t].data(), 1, part[t].size(), f) == part[t].size();
            okw = (fclose(f) == 0) && okw;
            P.inOut->logMain << "ENGINE_GPU: hand-off for the GPU pileup: " << nReads << " reads, " << nRec << " records, " << base
                             << " bytes in " << (time(nullptr) - th0) << " s -> " << hp << "\n";
        }
        if (!okw) { P.inOut->logMain << "ENGINE_GPU: hand-off not written (" << hp << ")\n"; unlink(hp); }
        // "on": STAR writes no BAM records for reads with a record (the pileup has their alignments)
        const char *gm = getenv("SINGLET_GPU_PILEUP");
        if (okw && gm != nullptr && strcmp(gm, "on") == 0) {
            static std::vector<uint32> keep;
            keep.swap(out.readRecord);
            g_handoffReadRecord = keep.data(); g_handoffNReads = keep.size(); g_handoffRecAln = recAln.data();
            if (!out.clip5.empty()) {   // the GPU's clip lengths replace STAR's CPU clip for these reads (readLoad)
                static std::vector<uint16> c5, c3;
                c5.swap(out.clip5); c3.swap(out.clip3);
                g_handoffClip5 = c5.data(); g_handoffClip3 = c3.data();
                const char *cc = getenv("SINGLET_HANDOFF_CLIPCHECK");
                g_handoffClipCheck = cc != nullptr && cc[0] == '1';
            }
            P.inOut->logMain << "ENGINE_GPU: no BAM records for the reads of the " << nRec << " hand-off records\n";
            if (sumOn) {   // the summaries, concatenated in record order (threads hold contiguous record ranges)
                static std::vector<uint8> sumAll;
                static std::vector<uint64> sumOff;
                uint64 tot = 0;
                for (int t = 0; t < nt; t++) tot += sumPart[t].size();
                sumAll.resize(std::max<uint64>(tot, 1)); sumOff.resize(nRec + 1);
                uint64 base = 0, r = 0;
                for (int t = 0; t < nt; t++) {
                    if (!sumPart[t].empty()) memcpy(sumAll.data() + base, sumPart[t].data(), sumPart[t].size());
                    for (uint64 o : sumPartOff[t]) sumOff[r++] = base + o;
                    base += sumPart[t].size();
                    std::vector<uint8>().swap(sumPart[t]);
                }
                sumOff[nRec] = base;
                g_handoffSum = sumAll.data(); g_handoffSumOff = sumOff.data();
                P.inOut->logMain << "ENGINE_GPU: hand-off summaries for STAR's per-read output: " << base << " bytes\n";
            }
            // and the per-read entries (barcode, UMI, quality counters) in a shared mapping, filled by the mapping threads
            const char *mq = getenv("SINGLET_ENGINE_GPU_HANDOFF_MINQ");
            if (e2eRec) {   // E8: in the shared region (zero-filled: unwritten entries have flags 0); the magic last: the parent's pileup takes it
                singlet_engine_gpu::E2EHeader &E2H = *e2r.hdr();
                g_handoffMinQ = atoi(mq);
                if (!e10ok) {   // E8: STAR's mapping threads write the entries
                    g_handoffReads = (HandoffReadEntry *) (e2r.base + e2eE);
                    if (e2eQOn) g_handoffQMask = (uint64 *) (e2r.base + e2eQ);
                } else {        // E10: the device wrote them; the rRNA hook's verdicts for the sampled reads are added here
                    HandoffReadEntry *E = (HandoffReadEntry *) (e2r.base + e2eE);
                    const std::vector<uint64_t> &hk = out.e10.hookReads;
                    if (g_handoffReadHook != nullptr && !hk.empty()) {
                        std::vector<std::thread> tv;
                        for (int t = 0; t < nt; t++) tv.emplace_back([&, t]() {
                            std::string sq;
                            for (uint64 k = hk.size() * t / nt; k < hk.size() * (t + 1) / nt; k++) {
                                const uint64 i = hk[k], L = in.off[i + 1] - in.off[i];
                                sq.resize(L);
                                for (uint64 j = 0; j < L; j++) sq[j] = "ACGTN"[(int) in.seq[in.off[i] + j]];
                                E[i].flags |= g_handoffReadHook(i, sq.data(), L) & 0x1Cu;
                            }
                        });
                        for (auto &x : tv) x.join();
                    }
                    if (e10Level == 2 && e10ChkE != nullptr && e10ChkE[0] == '1') {   // STAR's threads write theirs elsewhere, compared after mapping
                        e10chk.on = true;
                        e10chk.ent.assign(g_handoffNReads, HandoffReadEntry{});
                        g_handoffReads = e10chk.ent.data();
                        if (e2eQOn) { e10chk.qm.assign(2 * g_handoffNReads, 0); g_handoffQMask = e10chk.qm.data(); }
                        e10chk.dev = E; e10chk.devQ = e2eQOn ? (const uint64 *) (e2r.base + e2eQ) : nullptr;
                    }
                }
                E2H.nReads = g_handoffNReads; E2H.oReadRecord = e2eR; E2H.oEntries = e2eE; E2H.oQMask = e2eQ;
                E2H.cbS = (uint32) P.pSolo.cbS; E2H.cbL = (uint32) P.pSolo.cbL; E2H.umiS = (uint32) P.pSolo.umiS; E2H.umiL = (uint32) P.pSolo.umiL;
                E2H.cbumiL = (uint32) P.pSolo.cbumiL; E2H.simple = 1;
                __sync_synchronize();
                memcpy(E2H.magic, "SGE2E001", 8);
                P.inOut->logMain << "ENGINE_GPU: E8: per-read hand-off entries" << (e2eQOn ? " and low-quality masks" : "") << " in the shared region ("
                                 << (E2H.used >> 20) << " MB in use)" << (e10ok ? ", built on the device (E10)" : "") << (e10chk.on ? "; STAR's are compared after mapping" : "")
                                 << "\n";
            }
            if (!e2eRec && mq != nullptr && P.pSolo.type == P.pSolo.SoloTypes::CB_UMI_Simple && P.pSolo.cbL <= 21 && P.pSolo.umiL <= 21) {
                const std::string rp = std::string(hp) + ".reads";
                const size_t bytes = std::max<size_t>(1, g_handoffNReads * sizeof(HandoffReadEntry));
                int fd = open(rp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
                void *m = MAP_FAILED;
                if (fd >= 0 && ftruncate(fd, (off_t) bytes) == 0) m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (fd >= 0) close(fd);
                if (m != MAP_FAILED) {   // the file is zero-filled: unwritten entries have flags 0
                    g_handoffReads = (HandoffReadEntry *) m; g_handoffMinQ = atoi(mq);
                    P.inOut->logMain << "ENGINE_GPU: per-read hand-off entries -> " << rp << "\n";
                    const char *qe = getenv("SINGLET_HANDOFF_QMASK");   // per-base quality masks for the GPU pileup (=0: off)
                    if (!(qe != nullptr && qe[0] == '0')) {
                        const std::string qp = rp + ".q";
                        const size_t qbytes = std::max<size_t>(16, g_handoffNReads * 16);
                        int qfd = open(qp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
                        void *qm = MAP_FAILED;
                        if (qfd >= 0 && ftruncate(qfd, (off_t) qbytes) == 0) qm = mmap(nullptr, qbytes, PROT_READ | PROT_WRITE, MAP_SHARED, qfd, 0);
                        if (qfd >= 0) close(qfd);
                        if (qm != MAP_FAILED) {g_handoffQMask = (uint64 *) qm; P.inOut->logMain << "ENGINE_GPU: per-read low-quality masks -> " << qp << "\n";}
                        else unlink(qp.c_str());
                    }
                } else {
                    unlink(rp.c_str());
                }
            }
        }
    }
    // E10 level >= 3: STAR's loop takes the per-read products of the reads with a record from the device (stats, junctions, unmapped records);
    // level 4 also keeps those reads out of STAR's loop (its read source serves the others only)
    if (e10ok && e10Level >= 3 && g_handoffReadRecord != nullptr && g_handoffSum != nullptr) {
        const auto t3 = std::chrono::steady_clock::now();
        typedef singlet_engine_gpu::Output::E10 X;
        Stats S;
        S.resetN();
        const uint64 *v = out.e10.stats;
        S.readN = v[X::S_readN]; S.readBases = v[X::S_readBases]; S.mappedReadsU = v[X::S_U]; S.mappedReadsM = v[X::S_M];
        S.mappedMismatchesN = v[X::S_mm]; S.mappedInsN = v[X::S_insN]; S.mappedDelN = v[X::S_delN]; S.mappedInsL = v[X::S_insL]; S.mappedDelL = v[X::S_delL];
        S.mappedBases = v[X::S_bases]; S.mappedPortion = out.e10.mappedPortion;
        for (int k = 0; k < SJ_MOTIF_SIZE; k++) S.splicesN[k] = v[X::S_spl0 + k];
        S.splicesNsjdb = v[X::S_sjdb];
        S.unmappedOther = v[X::S_uOther]; S.unmappedShort = v[X::S_uShort]; S.unmappedMismatch = v[X::S_uMismatch]; S.unmappedMulti = v[X::S_uMulti];
        S.unmappedAll = v[X::S_uAll];
        std::string w;
        bool ok3 = true;
        if (P.outReadsUnmapped == "Fastx" && !e10WriteUnmapped(P, out.e10.unmappedReads, (int) P.runThreadN, w)) ok3 = false;
        if (ok3) {
            g_statsAll.addStats(S);
            e10sj.swap(out.e10.sj); e10nSJ = out.e10.nSJ;
            g_e10Level = 3;
            if (e10Level >= 4 && g_starReadSource == nullptr) {
                if (e10BuildSource(P, g_handoffReadRecord, g_handoffNReads, w)) { g_starReadSource = &e10src; g_e10Level = 4; }
                else P.inOut->logMain << "ENGINE_GPU: E10 level 4 not used (" << w << "): level 3\n";
            }
            P.inOut->logMain << "ENGINE_GPU: E10 level " << g_e10Level << ": " << S.readN << " reads with a record (" << S.mappedReadsU << " unique, " << S.mappedReadsM
                             << " multi, " << S.unmappedAll << " unmapped: " << out.e10.unmappedReads.size() << " Unmapped.out records written), " << e10nSJ
                             << " junctions from the device; " << (g_e10Level == 4 ? std::to_string(e10src.iRead.size()) + " reads without a record go to STAR's loop" : "STAR's loop keeps readLoad, the barcode and the Solo record of these reads")
                             << " (" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t3).count() << " s)\n";
        } else {
            P.inOut->logMain << "ENGINE_GPU: E10 level 3 not used (" << w << "); STAR's loop does the per-read work\n";
        }
    }
    if (g_e10Level < 4) { std::vector<uint32_t>().swap(out.e10.unmappedReads); }
    if (deferInsert && g_e10Level >= 3) {   // E10: no read with a record looks the cache up (unmapped records: the device counted them)
        cache->prefilled = true;
        P.inOut->logMain << "ENGINE_GPU: no sequence results in the cache (E10 level " << g_e10Level << ")\n";
    } else
    if (deferInsert) {   // summaries installed: the unmapped sequences only; otherwise (hand-off not written) every record
        uint64 nIns = 0;
        if (g_handoffSum != nullptr && g_handoffRecAln != nullptr) {
            insertRecords([&](uint64 i) { return g_handoffRecAln[i] == 0; });
            for (uint64 i = 0; i < nRec; i++) nIns += g_handoffRecAln[i] == 0;
        } else {
            insertRecords([](uint64) { return true; });
            nIns = nRec;
        }
        cache->prefilled = true;
        P.inOut->logMain << "ENGINE_GPU: " << nIns << " of " << nRec << " sequence results in the cache (the others replay hand-off summaries)\n";
    }
    P.inOut->logMain << "ENGINE_GPU: cache filled with " << nRec << " sequence results in " << (time(nullptr) - t0) << " s (reading reads "
                     << (tRead - t0) << " s, GPU " << (tGpu - tLock) << " s, cache inserts " << (time(nullptr) - tGpu) << " s; device " << device << ")\n" << flush;
#endif
}

// STAR.cpp, right after the mapping threads finished: E10's after-mapping steps (the entry check at level 2; the device's junctions into the
// first thread's junction buffer at level >= 3, collapsed, so outputSJ merges them with STAR's own)
void seqCacheGpuAfterMapping(Parameters &P, ReadAlignChunk **RAchunk) {
#ifdef SINGLET_HAVE_ENGINE_GPU
    if (e10chk.on) {
        uint64 nRec = 0, nDiffE = 0, nDiffQ = 0, nUnwritten = 0;
        std::ostringstream ex;
        for (uint64 i = 0; i < g_handoffNReads; i++) {
            if (g_handoffReadRecord[i] == 0xFFFFFFFFu) continue;
            nRec++;
            const HandoffReadEntry &a = e10chk.ent[i], &b = e10chk.dev[i];
            if (a.flags == 0) nUnwritten++;
            if (memcmp(&a, &b, sizeof(HandoffReadEntry)) != 0) {
                if (nDiffE++ < 5)
                    ex << " [read " << i << ": STAR cr3 " << a.cr3 << " ur3 " << a.ur3 << " lq " << a.lowQual << " ne " << a.qualNeqN << " fl " << a.flags
                       << " | GPU cr3 " << b.cr3 << " ur3 " << b.ur3 << " lq " << b.lowQual << " ne " << b.qualNeqN << " fl " << b.flags << "]";
            }
            if (e10chk.devQ != nullptr && memcmp(&e10chk.qm[2 * i], &e10chk.devQ[2 * i], 16) != 0) nDiffQ++;
        }
        P.inOut->logMain << "ENGINE_GPU: E10 check: " << nRec << " reads with a record, entries differ " << nDiffE << ", masks differ " << nDiffQ
                         << " (STAR wrote no entry for " << nUnwritten << ")" << ex.str() << "\n" << std::flush;
        std::vector<HandoffReadEntry>().swap(e10chk.ent); std::vector<uint64>().swap(e10chk.qm);
        g_handoffReads = nullptr; g_handoffQMask = nullptr; e10chk.on = false;
    }
    if (g_e10Level >= 3 && P.outSJ.yes && e10nSJ > 0) {
        OutSJ *sj = RAchunk[0]->chunkOutSJ;
        while (sj->N + e10nSJ + 2 > sj->Nstore) sj->dataSizeIncrease();
        memcpy(sj->data + sj->N * Junction::dataSize, e10sj.data(), e10nSJ * Junction::dataSize);
        sj->N += e10nSJ;
        sj->collapseSJ();
        P.inOut->logMain << "ENGINE_GPU: E10: " << e10nSJ << " device junctions merged into the junction output\n" << std::flush;
    }
    std::vector<uint8>().swap(e10sj);
    e10f.release();
#else
    (void) P; (void) RAchunk;
#endif
}
