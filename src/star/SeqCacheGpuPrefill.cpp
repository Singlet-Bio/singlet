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
#include "singlet/engine_gpu.h"
#include "stitch.cuh"
#endif
#include "SeqCache.h"
#include "Parameters.h"
#include "Genome.h"
#include "InOutStreams.h"
#include <cmath>
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

bool noClipping(Parameters &P) {
    for (int ip = 0; ip < 2; ip++) {
        const ReadClipInput &c = P.pClip.in[ip];
        if (!c.N.empty() && c.N[0] != 0) return false;
        if (!c.NafterAd.empty() && c.NafterAd[0] != 0) return false;
        if (!c.adSeq.empty() && c.adSeq[0] != "-") return false;
    }
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
        {
            std::vector<std::thread> th;
            for (int t = 0; t < T; t++) th.emplace_back([&, t]() {
                uint64 li = first[t];
                const char *start = m + b[t];   // start of the line that ends at the first newline of this range
                while (start > m && start[-1] != '\n') start--;
                const char *e = m + b[t + 1];
                auto take = [&](const char *ls, const char *le) {
                    uint64 L = le - ls;
                    while (L > 0 && ls[L - 1] == '\r') L--;
                    for (uint64 k = 0; k < L; k++) seqT[t].push_back(nucl((unsigned char) ls[k]));
                    lenT[t].push_back(L);
                };
                for (const char *q = m + b[t]; (q = (const char *) memchr(q, '\n', e - q)) != nullptr; q++) {
                    if ((li & 3) == 1) take(start, q);
                    li++;
                    start = q + 1;
                }
                if (t == T - 1 && lastUnterminated && (li & 3) == 1) take(start, m + n);
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
        munmap((void *) m, n);
        close(fd);
    }
    return true;
}
#endif

} // namespace

void seqCacheGpuPrefill(Parameters &P, Genome &G, SeqCache *&cache) {
    const char *e = getenv("SINGLET_ENGINE_GPU");
    if (e == nullptr || e[0] != '1' || cache == nullptr)
        return;
#ifndef SINGLET_HAVE_ENGINE_GPU
    P.inOut->logMain << "ENGINE_GPU: requested, but this binary was built without CUDA (SINGLET_ENGINE_GPU=OFF): STAR maps every read\n" << flush;
    (void) G;
#else
    auto skip = [&](const std::string &why) {
        P.inOut->logMain << "ENGINE_GPU: not used (" << why << "): STAR maps every read\n" << flush;
    };
    std::string why;
    if (!singlet_engine_gpu::available(&why)) return skip(why);
    if (P.readNmates != 1) return skip("paired-end alignment");
    if (!noClipping(P)) return skip("read clipping (the cache key is the clipped sequence)");
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
    if (!readSequences(P, in, why, (int) P.runThreadN)) return skip(why);
    time_t tRead = time(nullptr);
    // one sample at a time per GPU (several samples in flight per node): singlet_engine_gpu::lockDevice
    time_t tWait = time(nullptr);
    int lockFd = -1;
    int device = singlet_engine_gpu::lockDevice(&lockFd);
    time_t tLock = time(nullptr);
    int hostThreads = (int) P.runThreadN;
    singlet_engine_gpu::Output out;
    bool ok = singlet_engine_gpu::mapDistinct(Q, H, G.SA.lengthByte, G.SAi.lengthByte, in, device, hostThreads, out);
    singlet_engine_gpu::unlockDevice(lockFd);
    time_t tGpu = time(nullptr);
    P.inOut->logMain << "ENGINE_GPU: device " << device << " (" << singlet_engine_gpu::deviceId(device) << "), waited " << (tLock - tWait)
                     << " s for it\n" << out.log;
    if (!ok) return skip("GPU mapping failed");
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
    // insert, sharded by hash across threads (SeqCache is thread-safe per shard)
    {
        int nt = std::max(1, (int) P.runThreadN);
        std::vector<std::thread> th;
        for (int t = 0; t < nt; t++) th.emplace_back([&, t]() {
            for (uint64 i = t; i < nRec; i += nt) {
                const char *s = out.seq.data() + out.seqOff[i];
                uint64 L = out.seqOff[i + 1] - out.seqOff[i];
                cache->insert(SeqCache::hashSeq(s, L), s, L, out.blob.data() + out.blobOff[i], out.blobOff[i + 1] - out.blobOff[i]);
            }
        });
        for (auto &x : th) x.join();
    }
    cache->prefilled = true;
    P.inOut->logMain << "ENGINE_GPU: cache filled with " << nRec << " sequence results in " << (time(nullptr) - t0) << " s (reading reads "
                     << (tRead - t0) << " s, GPU " << (tGpu - tLock) << " s, cache inserts " << (time(nullptr) - tGpu) << " s; device " << device << ")\n" << flush;
#endif
}
