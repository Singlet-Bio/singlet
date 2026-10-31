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
#include "singlet/fq/fg2_runset.h"
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
#include <atomic>
#include <cmath>
#include <fstream>
#include <thread>
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

namespace {

#ifdef SINGLET_HAVE_ENGINE_GPU
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
        std::vector<uint64> b(T + 1), cnt(T, 0);
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
        std::vector<uint64> first(T);
        for (int t = 0; t < T; t++) { first[t] = lines; lines += cnt[t]; }
        bool lastUnterminated = m[n - 1] != '\n';
        if (lastUnterminated) lines++;
        if (lines % 4 != 0) { munmap((void *) m, n); close(fd); why = fn + ": not 4-line FASTQ"; return false; }
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
                for (const char *q = m + b[t]; (q = (const char *) memchr(q, '\n', e - q)) != nullptr; q++) {
                    if ((li & 3) == 1) take(start, q);
                    li++;
                    start = q + 1;
                }
                if (t == T - 1 && lastUnterminated && (li & 3) == 1) take(start, m + n);
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
        munmap((void *) m, n);
        close(fd);
    }
    return true;
}

// The same inputs from the .1fg2's table of distinct cDNA sequences instead of the FASTQ text: the "reads" of `in` are the table's
// sequences (each clipped once: the clip depends on the sequence only), unitOfRead gives the sequence id of every read in file order, and
// the per-read outputs of mapDistinct are expanded through it. `in` is left untouched when this returns false.
bool openTable(singlet::fq::Fg2RunSet &rs, const char *path, std::string &why) {
    try {
        std::vector<std::string> dirs;
        if (const char *e = getenv("SINGLET_WHITELIST_DIR")) dirs.push_back(e);
        char exe[4096];
        const ssize_t k = readlink("/proc/self/exe", exe, sizeof exe - 1);
        if (k > 0) { exe[k] = 0; std::string d(exe); const size_t sl = d.rfind('/'); if (sl != std::string::npos) dirs.push_back(d.substr(0, sl) + "/../whitelists"); }
        rs.open({std::string(path)}, dirs, false, /*tables_only=*/true);
    } catch (const std::exception &e) { why = std::string(".1fg2 table: ") + e.what(); return false; }
    return true;
}

bool readTable(Parameters &P, singlet_engine_gpu::Input &in, std::string &why, int nThreads, singlet::fq::Fg2RunSet &rs, std::vector<uint32> &unitOfRead) {
    const auto &tab = rs.table();
    const uint64 nU = tab.n, nChunks = rs.block_count(), nReads = rs.stored_reads();
    if (nU == 0 || nU >= 0xFFFFFFFFull) { why = ".1fg2 table: no sequences or too many"; return false; }
    const int T = std::max(1, nThreads);
    std::vector<uint32> ids(nReads);
    std::atomic<bool> bad(false);
    {   // the sequence id of every read
        std::vector<std::thread> th;
        for (int t = 0; t < T; t++) th.emplace_back([&, t]() {
            std::vector<uint32> c;
            try {
                for (uint64 g = t; g < nChunks; g += T) {
                    rs.decode_seq_ids((uint32_t) g, c);
                    const uint64 r0 = rs.block_read_offset((uint32_t) g), r1 = g + 1 < nChunks ? rs.block_read_offset((uint32_t) g + 1) : nReads;
                    if (c.size() != r1 - r0) { bad = true; return; }
                    memcpy(ids.data() + r0, c.data(), c.size() * sizeof(uint32));
                }
            } catch (...) { bad = true; }
        });
        for (auto &x : th) x.join();
    }
    if (bad) { why = ".1fg2 table: cannot read the sequence-id plane"; return false; }
    for (uint64 i = 0; i < nReads; i++) if (ids[i] >= nU) { why = ".1fg2 table: sequence id out of range"; return false; }

    std::vector<std::vector<char>> seqT(T);
    std::vector<std::vector<uint64>> lenT(T);
    std::vector<std::vector<uint16>> p5T(T), p3T(T);
    std::vector<std::vector<uint8>> infoT(T);
    {
        std::vector<std::thread> th;
        for (int t = 0; t < T; t++) th.emplace_back([&, t]() {
            const uint64 u0 = nU * t / T, u1 = nU * (t + 1) / T;
            seqT[t].reserve((size_t) ((u1 - u0) * 96 + 4096));
            const bool cr4 = in.clip.pre;
            std::vector<std::vector<ClipMate>> cmv;
            if (cr4) P.pClip.initializeClipMates(cmv);
            std::vector<std::string> lets(64);
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
            for (uint64 u = u0; u < u1; u++) {
                const uint64 L = tab.slen[u], at = seqT[t].size();
                seqT[t].resize(at + L);
                char *d = seqT[t].data() + at;
                for (uint64 j = 0; j < L; j++) d[j] = (char) tab.base(u, (uint32_t) j);
                for (uint64 k = tab.n_first[u]; k < tab.n_first[u + 1]; k++) d[tab.n_pos[k]] = 4;
                lenT[t].push_back(L);
                if (cr4) {
                    std::string &l = lets[asc.size()];
                    l.resize(L);
                    for (uint64 j = 0; j < L; j++) l[j] = "ACGTN"[(int) d[j]];
                    asc.push_back({l.data(), L}); numAt.push_back(at);
                    if (asc.size() == 64) flush();
                }
            }
            if (cr4) flush();
        });
        for (auto &x : th) x.join();
    }
    {   // concatenate the threads' pieces in parallel
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
    unitOfRead.swap(ids);
    return true;
}

// the first reads of the file STAR reads, against the table: the order and the sequences must be the ones the table describes
bool checkTableHead(Parameters &P, const singlet::fq::Fg2RunSet &rs, const std::vector<uint32> &unitOfRead, std::string &why) {
    const auto &tab = rs.table();
    if (P.readFilesNames.empty() || P.readFilesNames[0].empty()) { why = "no read files"; return false; }
    std::ifstream f(P.readFilesNames[0][0]);
    std::string name, seq, plus, qual;
    for (uint64 i = 0; i < 4096 && i < unitOfRead.size(); i++) {
        if (!(std::getline(f, name) && std::getline(f, seq) && std::getline(f, plus) && std::getline(f, qual))) { why = "the read file ends before the table's reads"; return false; }
        while (!seq.empty() && seq.back() == '\r') seq.pop_back();
        const uint64 u = unitOfRead[i], L = tab.slen[u];
        if (seq.size() != L) { why = "read " + std::to_string(i) + ": length differs from the table's"; return false; }
        for (uint64 j = 0; j < L; j++) {
            const char c = "ACGT"[tab.base(u, (uint32_t) j)];
            if (c != seq[j] && !(seq[j] == 'N' || seq[j] == 'n')) { why = "read " + std::to_string(i) + ": bases differ from the table's"; return false; }
        }
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
    // SINGLET_ENGINE_GPU_FG2 (set by singlet for a .1fg2 input): the distinct sequences come from its table, with no pass over the read text
    std::vector<uint32> unitOfRead;
    bool tableMode = false;
    if (const char *fg2 = getenv("SINGLET_ENGINE_GPU_FG2"); fg2 != nullptr && fg2[0] != 0 && getenv("SINGLET_ENGINE_GPU_FG2_OFF") == nullptr) {
        std::string w2;
        singlet::fq::Fg2RunSet rs;
        if (openTable(rs, fg2, w2) && readTable(P, in, w2, (int) P.runThreadN, rs, unitOfRead) && checkTableHead(P, rs, unitOfRead, w2)) tableMode = true;
        else {
            P.inOut->logMain << "ENGINE_GPU: sequence table not used (" << w2 << "): reading the FASTQ\n" << flush;
            in = singlet_engine_gpu::Input(); unitOfRead.clear();
            if (!fillClip(P, in.clip, why)) return skip(why);
        }
    }
    if (!tableMode && !readSequences(P, in, why, (int) P.runThreadN)) return skip(why);
    time_t tRead = time(nullptr);
    // one sample at a time per GPU (several samples in flight per node): singlet_engine_gpu::lockDevice
    time_t tWait = time(nullptr);
    int lockFd = -1;
    int device = singlet_engine_gpu::lockDevice(&lockFd);
    time_t tLock = time(nullptr);
    int hostThreads = (int) P.runThreadN;
    singlet_engine_gpu::Output out;
    bool lockReleased = false;   // mapDistinct releases the device when its device work is done (the host assembly of the records needs no GPU)
    const auto tHeld0 = std::chrono::steady_clock::now(); double heldS = 0;
    in.release = [&]() { if (!lockReleased) { lockReleased = true; heldS = std::chrono::duration<double>(std::chrono::steady_clock::now() - tHeld0).count(); singlet_engine_gpu::unlockDevice(lockFd); } };
    bool ok = singlet_engine_gpu::mapDistinct(Q, H, G.SA.lengthByte, G.SAi.lengthByte, in, device, hostThreads, out);
    in.release();
    time_t tGpu = time(nullptr);
    if (ok && tableMode) {   // per table sequence -> per read, in file order
        const uint64 nR = unitOfRead.size();
        const int nt = std::max(1, (int) P.runThreadN);
        auto expand = [&](auto &v) {
            if (v.empty()) return;
            std::remove_reference_t<decltype(v)> w(nR);
            std::vector<std::thread> th;
            for (int t = 0; t < nt; t++) th.emplace_back([&, t]() { for (uint64 i = nR * t / nt; i < nR * (t + 1) / nt; i++) w[i] = v[unitOfRead[i]]; });
            for (auto &x : th) x.join();
            v.swap(w);
        };
        expand(out.readRecord); expand(out.clip5); expand(out.clip3); expand(in.cr4info);
        out.nReads = nR;
        std::vector<uint32>().swap(unitOfRead);
        P.inOut->logMain << "ENGINE_GPU: sequences from the .1fg2 table: " << nR << " reads\n" << flush;
    }
    P.inOut->logMain << "ENGINE_GPU: device " << device << " (" << singlet_engine_gpu::deviceId(device) << "), waited " << (tLock - tWait)
                     << " s for it, held it " << heldS << " s\n" << out.log;
    if (!ok) return skip("GPU mapping failed");
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
        std::vector<std::thread> th;
        for (int t = 0; t < nt; t++) th.emplace_back([&, t]() {
            Transcriptome *trT = (sumOn && tr != nullptr) ? new Transcriptome(*tr) : nullptr;
            ReadAlign *ra = new ReadAlign(P, G, trT, t);
            if (sumOn) ra->chunkOutSJ = new OutSJ(P.limitOutSJoneRead, P, G);   // scratch: one sequence's junctions at a time
            const uint64 r0 = nRec * t / nt, r1 = nRec * (t + 1) / nt;
            // ~70 bytes per record with one alignment (pbmc1k_v3: 2.3 GB for 31.8 M records): reserve, so the part grows without copying
            // GBs at each doubling (untouched capacity costs no memory)
            part[t].reserve((size_t) (r1 - r0) * 80);
            partOff[t].reserve((size_t) (r1 - r0));
            if (sumOn) { sumPartOff[t].reserve((size_t) (r1 - r0)); sumPart[t].reserve((size_t) (r1 - r0) * 40); }   // 1.06 GB for 31.8 M records
            for (uint64 r = r0; r < r1; r++) {
                partOff[t].push_back(part[t].size());
                ra->seqCacheAlignRecords(out.seq.data() + out.seqOff[r], out.seqOff[r + 1] - out.seqOff[r], out.blob.data() + out.blobOff[r], part[t]);
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
        for (int t = 0, r = 0; t < nt; t++)
            for (uint64 o : partOff[t]) recAln[r++] = part[t][o];
        FILE *f = fopen(hp, "wb");
        bool okw = f != nullptr;
        if (okw) {
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
            if (mq != nullptr && P.pSolo.type == P.pSolo.SoloTypes::CB_UMI_Simple && P.pSolo.cbL <= 21 && P.pSolo.umiL <= 21) {
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
