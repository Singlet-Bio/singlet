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
#include "singlet/engine_gpu_ingest.h"   // E13: the reads from the .1fq, decoded here (SeqCacheGpuIngest.cpp)
#include "stitch.cuh"
#include "solo_gpu.h"   // E11: E9's Solo library (before STAR's headers, like stitch.cuh)
#include "singlet/tail_trace.h"   // E16: marks of the level-5 tail
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
#include "ErrorWarning.h"
#include <algorithm>
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
#include <functional>
#include <string>
#include <thread>
#include <vector>

// E10 (SINGLET_ENGINE_GPU_E2E >= 3, docs/gpu-engine/fable/E10-hostloop.md): what STAR's per-read loop still does for a read with a GPU record
// (ReadAlign_oneRead.cpp, ReadAlign_outputAlignments.cpp). 0: everything (E8); 3: readLoad, the barcode and the Solo record only (the
// device computed its Log.final.out counters, junctions and Unmapped.out records); 4: such reads are not given to STAR at all; 5: as 4, and
// Solo counts on the device (E11)
int g_e10Level = 0;
// E11 (SINGLET_ENGINE_GPU_E2E=5, docs/gpu-engine/fable/E11-solowire.md): E9's soloRun counted every read on the device after mapping and wrote
// star_Solo.out/; STAR.cpp then skips Solo::processAndOutput. At level 5 STAR's loop (the reads without a record) hands over the Gene
// annotation of each read (ReadAlign::e11Annot) instead of writing Solo's records
bool g_e11SoloDone = false;

namespace {

#ifdef SINGLET_HAVE_ENGINE_GPU
// E10: what the host decode keeps for the host side of E10 (the unmapped reads' FASTQ records, level 4's read source): both FASTQ files
// mapped, per read the byte offset of its name line in each (0: R2, the cDNA; 1: R1, the barcode read)
struct E10Files {
    const char *m[2] = {nullptr, nullptr};
    uint64 len[2] = {0, 0};
    singlet_engine_gpu::NoInitVector<uint64> at[2];   // every element written by the decode (no serial zero-fill first)
    void release() {
        for (int k = 0; k < 2; k++) {
            if (m[k] != nullptr) munmap((void *) m[k], len[k]);
            m[k] = nullptr; len[k] = 0; singlet_engine_gpu::NoInitVector<uint64>().swap(at[k]);
        }
    }
};
E10Files e10f;
// E13 (SINGLET_ENGINE_GPU_INGEST=1, singlet/engine_gpu_ingest.h): the reads decoded from the .1fq into packed arrays instead of the FASTQ files
// above; their host planes (cDNA quality, barcode read) stay here until the prefill returns, the sequences are in Input (ingSeq, ingOff)
singlet_engine_gpu::ingest::PackedReads ing;
bool ingOn = false;
const char *ingSeq = nullptr;
const uint64_t *ingOff = nullptr;
// what the prefill takes from the decode: phases A (decodeReads) and B (barcodes) of SeqCacheGpuIngest.cpp and C (cr4Clip, STAR's
// CellRanger4 clip). startEarly's thread th0 runs A from STAR's first moment, beside the CR4 probe, then C with STAR's fixed CellRanger4
// sequences when the pipeline passes that clip; seqCacheGpuEarly's thread th (STAR's parameters read, before the genome load) runs B, and C
// when th0 did not clip with the ClipMates STAR built; without startEarly the prefill runs all of it. On the heap and never destroyed (an
// exit() while a thread runs must not run ~thread); per pid (the CR4 probe's forked STARs inherit the memory, not the threads).
struct IngPrep {
    std::thread th0, th;
    pid_t pid = 0;
    int threads0 = 1;
    bool early = false, cr4 = false;       // startEarly ran (with the CellRanger4 clip after the decode)
    bool started = false;                   // ingPrepare is running or ran
    bool okA = false, ok = false, clipPre = false, clipEarly = false;
    std::string why, bcWhy, clipKey;        // clipKey: the ClipMates th0 clipped with (clipKeyOf)
    singlet_engine_gpu::ingest::PackedReads reads;   // th0's decode until ingPrepare takes it into `ing`
    std::chrono::steady_clock::time_point t0;
    double secA = 0, doneA = 0, waitA = 0, secB = 0, secC = 0, doneC = 0, clipS = 0, clipGroupS = 0;
    uint64 clipDistinct = 0;                // the distinct cDNA sequences the clip ran on
    bool fromParent = false;                // phase A ran in the pipeline's process before STAR was forked (takeParent)
    int minQ = 0;
    uint32 nw = 0;
    std::vector<uint16> pre5, pre3;
    std::vector<uint8> info;
};
IngPrep *ingPrep = nullptr;
inline bool ingPrepHere() { return ingPrep != nullptr && ingPrep->pid == getpid(); }
inline double ingSince(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); }
void ingClipFree(IngPrep &W) {
    std::vector<uint16>().swap(W.pre5); std::vector<uint16>().swap(W.pre3); std::vector<uint8>().swap(W.info);
}
void ingRelease() {
    if (ingPrepHere()) {
        if (ingPrep->th.joinable()) ingPrep->th.join();
        if (ingPrep->th0.joinable()) ingPrep->th0.join();
        ingClipFree(*ingPrep);
        ingPrep->reads.release();
    }
    ing.release(); ingOn = false; ingSeq = nullptr; ingOff = nullptr;
}
// STAR's read source is free for the prefill's own (level 4/5): none set, or the pipeline's E13 chunk source over the .1fq
inline bool readSourceFree() { return g_starReadSource == nullptr || (const void *) g_starReadSource == singlet_engine_gpu::ingest::plan().fallback; }
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
// E11 (level 5): STARsolo's Gene counting on the device with E9's library. The configuration E9 ports (e11Config), the decode's quality counts
// for Gene/Summary.csv and the posterior's quality, and what soloRunKept takes after mapping
struct E11State {
    bool want = false;            // level 5 requested, E10 on and STAR's Solo configuration is the one E9 ports
    bool ready = false;           // and the decode's checks passed and the engine kept the Solo inputs on the device: level 5 runs
    std::string why;              // why not
    e9::SoloModel M;
    e9::SoloParams S;
    e9::SoloFilterParams FP;
    // the decode: the cDNA read's quality characters over whole reads (>= readQualityScoreBase + 30, all), the barcode read's CB and UMI
    // spans (SoloReadBarcode::getCBandUMI's qualHist) and the lowest and highest quality character of the CB span (the posterior's quality,
    // SoloReadFeature_inputRecords.cpp:87-89: exact with one constant pq only when every CB base gives the same qin)
    uint64 q30Rna = 0, nRna = 0, q30Cb = 0, nCb = 0;
    int cbQmin = 256, cbQmax = -1;
    uint32 qBase = 33;
    double sAdj = 0;              // the RNA count adjustment (clipped spans, reads without a record)
    int device = -1;              // the device that holds the kept inputs
    uint64 nNoRec = 0, nClipped = 0;
} e11;
// E16 "bg" (level 5): the Unmapped.out records of the reads with an unmapped record (3.3 M on pbmc1k_v3) are formatted, and the RNA Q30
// adjustment made, on a thread beside STAR's loop (4 threads under it) instead of before the loop; seqCacheGpuAfterMapping joins it,
// appends the records to STAR's streams (the files' record order is run-dependent anyway) and Solo's summary then takes the adjusted counts
// E16 "free": the prefill's reads and engine output, freed after STAR's loop (freeing them beside the loop or the chunk construction slows
// both: the kernel serialises the unmapping with the other threads' page faults)
struct E16Held { singlet_engine_gpu::Input *in = nullptr; singlet_engine_gpu::Output *out = nullptr; } e16held;
struct E16Bg {
    std::thread th; bool on = false, okU = true, okR = true; std::string whyU, whyR;
    std::vector<uint32_t> reads; std::vector<std::string> out[2]; double sU = 0, sR = 0;
    ~E16Bg() { if (th.joinable()) th.detach(); }   // a run stopped before STAR's loop ended
} e16bg;
// the Solo configuration E9's library reproduces byte for byte (E9-solo.md section 1); anything else: STAR's Solo counts (level 3)
bool e11Config(Parameters &P, Transcriptome *tr, std::string &why) {
    const ParametersSolo &s = P.pSolo;
    if (tr == nullptr || !P.quant.gene.yes || tr->nTr == 0 || tr->nGe == 0) { why = "no transcript model"; return false; }
    if (s.type != s.SoloTypes::CB_UMI_Simple || s.CBtype.type != 1) { why = "soloType"; return false; }
    if (s.nFeatures != 1 || s.features[0] != SoloFeatureTypes::Gene) { why = "soloFeatures other than Gene"; return false; }
    if (!s.cbWLyes || s.cbWL.empty() || s.cbWL.size() >= (1ull << 22)) { why = "whitelist"; return false; }
    if (!(s.CBmatchWL.mm1 && s.CBmatchWL.mm1_multi && s.CBmatchWL.mm1_multi_pc && s.CBmatchWL.mm1_multi_Nbase && !s.CBmatchWL.oneExact && !s.CBmatchWL.EditDist_2)) {
        why = "soloCBmatchWLtype other than 1MM_multi_Nbase_pseudocounts"; return false;
    }
    if (s.umiDedup.types.size() != 1 || s.umiDedup.types[0] != UMIdedup::All) { why = "soloUMIdedup other than 1MM_All"; return false; }
    // MultiMappers::initialize records only the types other than Unique: soloMultiMappers Unique (the default) leaves types empty and N = 0
    if (s.multiMap.yes.multi || s.multiMap.yes.N != 0 || !s.multiMap.types.empty()) { why = "soloMultiMappers other than Unique"; return false; }
    if (s.umiFiltering.yes || s.umiFiltering.MultiGeneUMI || s.umiFiltering.MultiGeneUMI_All || s.umiFiltering.MultiGeneUMI_CR) { why = "soloUMIfiltering"; return false; }
    if (s.cellFilter.type.empty() || s.cellFilter.type[0] != "CellRanger2.2") { why = "soloCellFilter other than CellRanger2.2"; return false; }
    if (!s.readStats.yes) { why = "no soloCellReadStats"; return false; }
    if (s.samAttrYes || s.readInfoYes[SoloFeatureTypes::Gene]) { why = "CB/UB BAM attributes"; return false; }
    if (s.outFileNames.size() < 4 || s.outFileNames[1] != "features.tsv" || s.outFileNames[2] != "barcodes.tsv" || s.outFileNames[3] != "matrix.mtx"
        || s.outFormat.featuresGeneField3 != "Gene Expression" || (!s.clusterCBfile.empty() && s.clusterCBfile != "-")) { why = "Solo output names"; return false; }
    if (s.cbL > 21 || s.umiL > 12 || tr->nGe >= (1u << 17)) { why = "CB/UMI length or gene count over E9's key widths"; return false; }
    if (P.pGe.chrSet.mito.size() > 4) { why = "more than 4 mitochondrial contigs"; return false; }
    if (P.readQualityScoreBase > 255) { why = "readQualityScoreBase"; return false; }
    e9::SoloModel &M = e11.M;
    e9::SoloParams &S = e11.S;
    M.wl.assign(s.cbWL.begin(), s.cbWL.end());
    const uint64 n = tr->nTr;
    M.trS.assign(tr->trS, tr->trS + n); M.trE.assign(tr->trE, tr->trE + n); M.trEmax.assign(tr->trEmax, tr->trEmax + n);
    M.trExI.assign(tr->trExI, tr->trExI + n); M.trGene.assign(tr->trGene, tr->trGene + n);
    M.trExN.assign(tr->trExN, tr->trExN + n); M.trStr.assign(tr->trStr, tr->trStr + n);
    M.exSE.assign(tr->exSE, tr->exSE + 2 * (uint64) tr->nEx);
    M.geID = tr->geID; M.geName = tr->geName;
    S.cbL = (uint32) s.cbL; S.umiL = (uint32) s.umiL; S.strand = s.strand; S.wlN = (uint32) M.wl.size(); S.nGe = tr->nGe;
    S.cbMinP = s.cbMinP; S.umiMaskLow = s.umiMaskLow;
    for (uint32 jj = 0; jj < 4; jj++) { S.homo[jj] = 0; for (uint32 ii = 0; ii < S.umiL; ii++) S.homo[jj] = (S.homo[jj] << 2) + jj; }   // SoloReadBarcode.cpp:15-19
    S.nMito = 0;
    for (uint64 c : P.pGe.chrSet.mito) S.mitoChr[S.nMito++] = (int64_t) c;
    S.pq = 0;   // after the decode (e11Posterior)
    e11.FP.nExpectedCells = s.cellFilter.knee.nExpectedCells; e11.FP.maxPercentile = s.cellFilter.knee.maxPercentile; e11.FP.maxMinRatio = s.cellFilter.knee.maxMinRatio;
    e11.qBase = (uint32) P.readQualityScoreBase;
    return true;
}
// the posterior's pq from the decode's CB qualities: STAR takes the quality character of the read's CB base at the mismatch (qin = min(q - QSbase,
// QSmax)); one value for every read when every CB base has q >= QSbase + QSmax, or one constant character
bool e11Posterior(Parameters &P, std::string &why) {
    const ParametersSolo &s = P.pSolo;
    if (e11.cbQmax < 0) { why = "no CB qualities"; return false; }
    if (!(e11.cbQmin >= (int) s.QSbase + (int) s.QSmax || e11.cbQmin == e11.cbQmax)) {
        why = "CB base qualities vary below QSbase + QSmax (" + std::to_string(e11.cbQmin) + ".." + std::to_string(e11.cbQmax) + "): the posterior is per read";
        return false;
    }
    char qin = (char) e11.cbQmin;
    qin -= s.QSbase;
    qin = qin < s.QSmax ? qin : s.QSmax;
    e11.S.pq = std::pow(10.0, -qin / 10.0);
    return true;
}

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
        uint64_t *aux = nullptr;
        std::vector<int> e10BadT(T, 0);
        // E11: per thread the cDNA read's quality characters >= readQualityScoreBase + 30 and all (Gene/Summary.csv's "Q30 Bases in RNA read")
        const bool e11q = e11.want;
        const unsigned e11q30 = e11.qBase + 30;
        std::vector<uint64> e11q30T(T, 0), e11nT(T, 0);
        if (e10) {
            uint64 mx = 0;
            for (int t = 0; t < T; t++) mx = std::max(mx, mxT[t]);
            nw = (uint32) std::max<uint64>(1, (mx + 63) / 64);
            if (nw > 4) { in.e10.on = false; in.e10.nw = 0; e10DecodeWhy = "cDNA reads over 256 bases"; }
            else {
                in.e10.nw = nw;
                // not zero-filled here (2 GB on one core): the parsing threads write every word (the quality words zeroed per read first)
                in.e10.aux.resize((lines / 4) * (2 + (uint64) nw));
                aux = in.e10.aux.data();
                e10f.at[0].resize(lines / 4);
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
                // E11: thread-local counts, stored once per thread (per-read updates of e11q30T[t] would share cache lines between threads)
                uint64 l30 = 0, lnq = 0;
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
                        uint64_t *w = aux + r * (2 + (uint64) nw) + 2;
                        for (uint32 k = 0; k < nw; k++) w[k] = 0;
                        for (uint64 j = 0; j < L; j++)
                            if ((int) (unsigned char) ls[j] - 33 < e10MinQ) w[j >> 6] |= 1ull << (j & 63);
                        if (e11q) {
                            uint64 c = 0;
                            for (uint64 j = 0; j < L; j++) c += (unsigned char) ls[j] >= e11q30;
                            l30 += c; lnq += L;
                        }
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
                e11q30T[t] = l30; e11nT[t] = lnq;
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
            if (e11q) for (int t = 0; t < T; t++) { e11.q30Rna += e11q30T[t]; e11.nRna += e11nT[t]; }
            if (bad) { in.e10.on = false; decltype(in.e10.aux)().swap(in.e10.aux); e10f.release();
                       e10DecodeWhy = (bad & 1) ? "a cDNA base other than ACGTN" : "a quality line longer than 256"; }
        }
        if (e10q && in.e10.on) { e10f.m[0] = m; e10f.len[0] = n; }   // E10 keeps R2 mapped (the unmapped reads' records, level 4's source)
        else munmap((void *) m, n);
        close(fd);
    }
    return true;
}

// the ClipMate fields the CellRanger4 clip depends on (ClipMate::initialize, ::clip; the opal alignment takes adSeqNum, made from adSeq)
std::string clipKeyOf(const ClipMate &m5, const ClipMate &m3) {
    std::ostringstream k;
    for (const ClipMate *m : {&m5, &m3}) k << m->type << ':' << m->N << ':' << m->NafterAd << ':' << m->adSeq << ':' << m->adMMp << ';';
    return k.str();
}

// E13 phase C: STAR's CellRanger4 clip of every packed read with the mate-1 ClipMates m5 (type 10) and m3 (type 11), as readSequences runs
// it (ClipMate::clipChunk's opal alignment of the TSO, 64 reads at a time, and readLoad's ClipMate::clip), on T threads over contiguous
// read ranges, each with its own copy of the ClipMates and its own ClipCR4 (opal's buffers; STAR has one per chunk).
// A read's clip is a function of its sequence alone (its length and bases: the opal alignment, ClipMate::clip and polyTail3p read nothing
// else, and opal's result for one sequence does not depend on the others in its batch of 64), so it runs once per distinct sequence: the
// reads are grouped by an exact-match hash table (64-bit hash of length and bases, then the bytes compared), the read that took the group's
// slot is clipped and the others copy its result. Half of a 10x sample's cDNA reads repeat another one. SINGLET_ENGINE_GPU_INGEST_CLIPDEDUP=0:
// every read is clipped.
void cr4Clip(const singlet_engine_gpu::ingest::PackedReads &R, const ClipMate &m5, const ClipMate &m3, int T, IngPrep &W) {
    const uint64 N = R.n;
    W.pre5.assign(N, 0); W.pre3.assign(N, 0); W.info.assign(N, 0);
    const uint64 Wl = (uint64) singlet_engine_gpu::ingest::plan().maxR2 + 1;
    const char *sq = R.seq.data();
    const uint64_t *of = R.off.data();
    auto par = [T](const std::function<void(int)> &f) {
        std::vector<std::thread> th;
        for (int t = 0; t < T; t++) th.emplace_back(f, t);
        for (auto &x : th) x.join();
    };
    // the groups: rep[r] = the read whose clip read r takes (r itself for the clipped ones)
    const char *dd = getenv("SINGLET_ENGINE_GPU_INGEST_CLIPDEDUP");
    const bool dedup = !(dd && dd[0] == '0') && N > 0 && N < 0xFFFFFFFFull;
    singlet_engine_gpu::NoInitVector<uint32_t> rep;
    const auto tg = std::chrono::steady_clock::now();
    W.clipDistinct = N;
    if (dedup) {
        singlet_engine_gpu::NoInitVector<uint64_t> h(N);
        uint64 cap = 1;
        while (cap < 2 * N) cap <<= 1;
        const uint64 mask = cap - 1;
        singlet_engine_gpu::NoInitVector<uint32_t> tab(cap);
        rep.resize(N);
        par([&](int t) {
            for (uint64 i = cap * t / T, e = cap * (t + 1) / T; i < e; i++) tab[i] = 0;
            for (uint64 r = N * t / T, e = N * (t + 1) / T; r < e; r++) {
                const uint64 a = of[r], L = of[r + 1] - a;
                uint64 x = 0x9E3779B97F4A7C15ull ^ (L * 0xC2B2AE3D27D4EB4Full), w;
                uint64 j = 0;
                for (; j + 8 <= L; j += 8) { memcpy(&w, sq + a + j, 8); x = (x ^ w) * 0xFF51AFD7ED558CCDull; x ^= x >> 32; }
                if (j < L) { w = 0; memcpy(&w, sq + a + j, L - j); x = (x ^ w) * 0xFF51AFD7ED558CCDull; x ^= x >> 32; }
                x ^= x >> 29; x *= 0xC4CEB9FE1A85EC53ull; x ^= x >> 32;
                h[r] = x;
            }
        });
        std::vector<uint64> nd(T, 0);
        par([&](int t) {
            uint64 n = 0;
            for (uint64 r = N * t / T, e = N * (t + 1) / T; r < e; r++) {
                const uint64 a = of[r], L = of[r + 1] - a;
                for (uint64 sl = h[r] & mask;; sl = (sl + 1) & mask) {
                    uint32_t v = __atomic_load_n(&tab[sl], __ATOMIC_ACQUIRE);
                    if (v == 0) {
                        if (__atomic_compare_exchange_n(&tab[sl], &v, (uint32_t) (r + 1), false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) { rep[r] = (uint32_t) r; n++; break; }
                    }
                    const uint64 q = v - 1;   // the read in that slot (v: the value the exchange found)
                    if (h[q] == h[r] && of[q + 1] - of[q] == L && memcmp(sq + of[q], sq + a, L) == 0) { rep[r] = (uint32_t) q; break; }
                }
            }
            nd[t] = n;
        });
        W.clipDistinct = 0;
        for (uint64 x : nd) W.clipDistinct += x;
    }
    W.clipGroupS = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();
    std::vector<double> tt(T, 0.0);
    par([&](int t) {
        const auto t0 = std::chrono::steady_clock::now();
        ClipMate c5 = m5, c3 = m3;
        ClipCR4 *cr = new ClipCR4;
        c5.cr4 = cr; c3.cr4 = cr;   // type 11 calls cr4->polyTail3p, which reads none of its members (STAR leaves that pointer unset)
        std::vector<char> asc(64 * Wl), cbuf(DEF_readSeqLengthMax + 16);
        static const char kNt[8] = {'A', 'C', 'G', 'T', 'N', 'N', 'N', 'N'};
        uint64 bat[64];
        const uint64 rE = N * (t + 1) / T;
        uint64 r = N * t / T;
        while (r < rE) {
            int nb = 0;
            for (; r < rE && nb < 64; r++) if (!dedup || rep[r] == r) bat[nb++] = r;
            if (nb == 0) break;
            for (int k = 0; k < nb; k++) {
                const uint64 a = of[bat[k]], L = of[bat[k] + 1] - a;
                char *d = asc.data() + k * Wl;
                for (uint64 j = 0; j < L && j < Wl; j++) d[j] = kNt[(int) sq[a + j] & 7];
                cr->opalFillOneSeq(k, d, (uint32) std::min<uint64>(L, Wl));
            }
            cr->opalAlign((uint8 *) c5.adSeqNum.data(), c5.adSeqNum.size(), nb);
            for (int k = 0; k < nb; k++) {
                const int Lm = cr->opalRes[k].endLocationTarget + 1, S = cr->opalRes[k].score;
                const bool L0 = S < 20 || (S == 20 && Lm > 26) || (S == 21 && Lm > 30);
                const uint8 info = (uint8) (L0 ? 0 : Lm);
                const uint64 rr = bat[k], a = of[rr], L = of[rr + 1] - a;
                uint c5n = 0, c3n = 0;
                if (L <= DEF_readSeqLengthMax) {
                    memcpy(cbuf.data(), sq + a, L);
                    uint Lread = (uint) L;
                    c5.clippedInfo = (char) info;
                    c5n = c5.clip(Lread, cbuf.data());
                    c3n = c3.clip(Lread, cbuf.data());
                }
                W.pre5[rr] = (uint16) std::min<uint>(c5n, 65535); W.pre3[rr] = (uint16) std::min<uint>(c3n, 65535); W.info[rr] = info;
            }
        }
        delete cr;
        tt[t] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    });
    if (dedup) {
        par([&](int t) {
            for (uint64 r = N * t / T, e = N * (t + 1) / T; r < e; r++) {
                const uint64 q = rep[r];
                if (q != r) { W.pre5[r] = W.pre5[q]; W.pre3[r] = W.pre3[q]; W.info[r] = W.info[q]; }
            }
        });
    }
    W.clipS = 0;
    for (double x : tt) W.clipS += x;
}

// E13 phases A (taken from startEarly's thread, or run here), B and C (taken from startEarly's thread when it clipped with the ClipMates
// STAR built, or run here), from STAR's parameters (the barcode layout and whitelist, the clip)
void ingPrepare(Parameters &P, IngPrep &W) {
    namespace ig = singlet_engine_gpu::ingest;
    const ig::Plan &pl = ig::plan();
    const ParametersSolo &so = P.pSolo;
    const int T = std::max(1, (int) P.runThreadN);
    if (W.early) {
        const auto w0 = std::chrono::steady_clock::now();
        if (W.th0.joinable()) W.th0.join();
        W.waitA = ingSince(w0);
        if (!W.okA) { ingClipFree(W); W.reads.release(); return; }
        ing.swap(W.reads); W.reads.release();
    } else if (!ig::decodeReads(T, ing, W.why)) return;
    W.secA = ing.sec;
    if (so.type != so.SoloTypes::CB_UMI_Simple || so.cbL > 21 || so.umiL > 21 || so.cbS < 1 || so.umiS < 1) { W.why = "barcode layout"; ing.release(); return; }
    const auto tb = std::chrono::steady_clock::now();
    ig::BarcodeConfig c;
    c.threads = T;
    const char *mq = getenv("SINGLET_ENGINE_GPU_HANDOFF_MINQ");   // E10's threshold (the prefill reads the same)
    W.nw = std::max<uint32>(1, (pl.maxR2 + 63) / 64);
    if (mq == nullptr || W.nw > 4) W.nw = 0;
    W.minQ = mq != nullptr ? atoi(mq) : 0;
    c.nw = W.nw; c.minQ = W.minQ;
    c.cbS0 = (uint32) so.cbS - 1; c.cbL = (uint32) so.cbL; c.umiS0 = (uint32) so.umiS - 1; c.umiL = (uint32) so.umiL;
    c.cbumiL = (uint32) so.cbumiL; c.bL = (uint32) so.bL;
    c.e11 = true; c.q30 = (unsigned) P.readQualityScoreBase + 30;
    if (so.cbWLyes && !so.cbWL.empty()) { c.wl = (const uint64_t *) so.cbWL.data(); c.nWL = so.cbWL.size(); }
    if (!ig::barcodes(c, ing, W.bcWhy)) { W.why = W.bcWhy; ing.release(); return; }
    W.secB = std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count();
    singlet_engine_gpu::Clip clip;
    std::string cw;
    W.clipPre = fillClip(P, clip, cw) && clip.pre;
    if (W.clipPre) {
        std::vector<std::vector<ClipMate>> cmv;
        P.pClip.initializeClipMates(cmv);
        if (!(W.clipEarly && W.pre5.size() == ing.n && clipKeyOf(cmv[0][0], cmv[0][1]) == W.clipKey)) {   // not th0's: clip here
            W.clipEarly = false;
            const auto tc = std::chrono::steady_clock::now();
            cr4Clip(ing, cmv[0][0], cmv[0][1], T, W);
            W.secC = ingSince(tc);
        }
    } else { W.clipEarly = false; W.secC = 0; ingClipFree(W); }   // the CR4 probe dropped the profile: th0's clip is not STAR's
    W.ok = true;
}

// E13: readSequences and e10Barcodes from the .1fq (the pipeline's head-scan plan, engine_gpu_ingest.h): the engine's input (seq, off,
// E10's aux, the CellRanger4 clip), the host planes kept in `ing`, E11's counts and R11's histogram, from seqCacheGpuEarly's thread (or
// computed here when it did not start)
bool ingestReads(Parameters &P, singlet_engine_gpu::Input &in, std::string &why, double &waited) {
    const auto w0 = std::chrono::steady_clock::now();
    if (!ingPrepHere()) {
        ingPrep = new IngPrep;
        ingPrep->pid = getpid();
    }
    IngPrep &W = *ingPrep;
    if (!W.started) { W.started = true; ingPrepare(P, W); }
    else if (W.th.joinable()) W.th.join();
    waited = ingSince(w0);
    if (!W.ok) { why = W.why; return false; }
    if (W.clipPre != in.clip.pre) { why = "the clip configuration changed"; return false; }
    if (in.clip.pre) { in.pre5.swap(W.pre5); in.pre3.swap(W.pre3); in.cr4info.swap(W.info); }
    in.seq.swap(ing.seq); in.off.swap(ing.off);
    ingSeq = in.seq.data(); ingOff = in.off.data(); ingOn = true;
    if (in.e10.on) {
        if (ing.nw == 0) {
            in.e10.on = false; in.e10.nw = 0;
            e10DecodeWhy = !W.bcWhy.empty() ? W.bcWhy : W.nw == 0 ? "cDNA reads over 256 bases" : "cDNA reads longer than the quality words hold";
        } else if (W.minQ != e10MinQ) {
            in.e10.on = false; in.e10.nw = 0; e10DecodeWhy = "the low-quality threshold differs from the decode's";
        } else { in.e10.nw = ing.nw; in.e10.aux.swap(ing.aux); }
    }
    singlet_engine_gpu::NoInitVector<uint64_t>().swap(ing.aux);
    if (e11.want && in.e10.on) {
        e11.q30Rna = ing.q30Rna; e11.nRna = ing.nRna; e11.q30Cb = ing.q30Cb; e11.nCb = ing.nCb; e11.cbQmin = ing.cbQmin; e11.cbQmax = ing.cbQmax;
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
    e10f.at[1].resize(nReads);   // every element written below (each read has a name line)
    const uint64 bL = P.pSolo.bL, cbumiL = P.pSolo.cbumiL, cbS = P.pSolo.cbS, cbL = P.pSolo.cbL, umiS = P.pSolo.umiS, umiL = P.pSolo.umiL;
    std::vector<int> badT(T, 0);
    uint64_t *aux = in.e10.aux.data();
    // E11: per thread getCBandUMI's qualHist of the CB and UMI spans (the quality line padded with 'H' to cbumiL, like the sequence with N),
    // bases >= readQualityScoreBase + 30 and all, and the lowest and highest character of the CB span (the posterior's quality)
    const bool e11q = e11.want;
    const unsigned e11q30 = e11.qBase + 30;
    std::vector<uint64> q30T(T, 0), nqT(T, 0);
    std::vector<int> qminT(T, 256), qmaxT(T, -1);
    {
        std::vector<std::thread> th;
        for (int t = 0; t < T; t++) th.emplace_back([&, t]() {
            uint64 li = first[t];
            const char *start = m + b[t];
            while (start > m && start[-1] != '\n') start--;
            const char *e = m + b[t + 1];
            // E11: thread-local quality counts, stored once per thread (per-character updates of qminT[t]/qmaxT[t] and per-read ones of
            // q30T[t]/nqT[t] share cache lines between the threads: 38.6 s instead of 1 s for this pass in gate 796377)
            uint64 lq30 = 0, lnq = 0;
            int lqmin = 256, lqmax = -1;
            auto line = [&](const char *ls, const char *le) {
                const uint64 r = li >> 2;
                if ((li & 3) == 0) { e10f.at[1][r] = (uint64) (ls - m); return; }
                if ((li & 3) == 3 && e11q) {
                    uint64 L = le - ls;
                    if (L > 0 && (unsigned char) ls[L - 1] < 33) L--;
                    const uint64 S = (bL == 0 && L < cbumiL) ? cbumiL : L;
                    auto qc = [&](uint64 j) { return j < L ? (unsigned) (unsigned char) ls[j] : (unsigned) 'H'; };
                    uint64 c = 0, nq = 0;
                    for (uint64 j = cbS - 1; j < std::min(cbS - 1 + cbL, S); j++) {
                        const unsigned q = qc(j);
                        c += q >= e11q30; nq++;
                        lqmin = std::min(lqmin, (int) q); lqmax = std::max(lqmax, (int) q);
                    }
                    for (uint64 j = umiS - 1; j < std::min(umiS - 1 + umiL, S); j++) { c += qc(j) >= e11q30; nq++; }
                    lq30 += c; lnq += nq;
                    return;
                }
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
                uint64_t *a = aux + r * (2 + (uint64) nw);
                a[0] = pack(cbS - 1, cbL); a[1] = pack(umiS - 1, umiL);
            };
            for (const char *q = m + b[t]; (q = (const char *) memchr(q, '\n', e - q)) != nullptr; q++) {
                line(start, q);
                li++;
                start = q + 1;
            }
            if (t == T - 1 && lastUnterminated) line(start, m + n);
            q30T[t] = lq30; nqT[t] = lnq; qminT[t] = lqmin; qmaxT[t] = lqmax;
        });
        for (auto &x : th) x.join();
    }
    for (int t = 0; t < T; t++) if (badT[t]) { munmap((void *) m, n); why = "a barcode read STAR would reject (length)"; return false; }
    if (e11q)
        for (int t = 0; t < T; t++) {
            e11.q30Cb += q30T[t]; e11.nCb += nqT[t];
            e11.cbQmin = std::min(e11.cbQmin, qminT[t]); e11.cbQmax = std::max(e11.cbQmax, qmaxT[t]);
        }
    e10f.m[1] = m; e10f.len[1] = n;
    return true;
}

// E10: STAR's view of read r's FASTQ record in mate file k: the name line's first token (processChunks: readID, removeStringEndControl) and
// passFilterIllumina from R2's name line, the sequence and quality lines without one trailing control character (fastqReadOneLine)
struct E10Rec { const char *id; uint64 idLen; char filter; const char *seq[2]; uint64 seqLen[2]; const char *qual[2]; uint64 qualLen[2]; std::string own; };
// E13: the same record from the packed reads: the FASTQ path's text of read r ("@r<n>", no second name field; R1's quality 'F')
inline bool ingRecord(uint64 r, E10Rec &x) {
    if (r >= ing.n) return false;
    static const char kNt[8] = {'A', 'C', 'G', 'T', 'N', 'N', 'N', 'N'};
    const uint32 b = ing.block(r);
    std::string &o = x.own;
    o.clear();
    o += "@r";
    o += std::to_string(ing.name(b, r));
    const uint64 idLen = o.size(), s0 = ingOff[r], L = ingOff[r + 1] - s0, L1 = ing.r1L;
    const size_t aS0 = o.size();
    for (uint64 j = 0; j < L; j++) o += kNt[(int) ingSeq[s0 + j] & 7];
    const size_t aQ0 = o.size();
    for (uint64 j = 0; j < L; j++) o += (char) (ing.qualPhred[ing.qcode(b, s0 + j)] + 33);
    const size_t aS1 = o.size();
    for (uint32 j = 0; j < L1; j++) o += kNt[ing.r1code(r, j)];
    const size_t aQ1 = o.size();
    o.append(L1, 'F');
    x.id = o.data(); x.idLen = idLen; x.filter = 'N';
    x.seq[0] = o.data() + aS0; x.seqLen[0] = L; x.qual[0] = o.data() + aQ0; x.qualLen[0] = L;
    x.seq[1] = o.data() + aS1; x.seqLen[1] = L1; x.qual[1] = o.data() + aQ1; x.qualLen[1] = L1;
    return true;
}
inline bool e10Record(uint64 r, E10Rec &x) {
    if (ingOn) return ingRecord(r, x);
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
// (E16: e10FormatUnmapped makes the records, e10PutUnmapped appends them to STAR's streams; at level 5 the formatting runs beside STAR's loop
// and the records are appended after it)
bool e10FormatUnmapped(Parameters &P, const std::vector<uint32_t> &reads, int nThreads, std::vector<std::string> (&out)[2], std::string &why) {
    const int T = std::max(1, nThreads);
    const uint64 N = reads.size();
    out[0].assign(T, std::string()); out[1].assign(T, std::string());
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
    return true;
}
void e10PutUnmapped(Parameters &P, std::vector<std::string> (&out)[2]) {
    for (int im = 0; im < 2; im++)
        for (size_t t = 0; t < out[im].size(); t++) { P.inOut->outUnmappedReadsStream[im].write(out[im][t].data(), out[im][t].size()); std::string().swap(out[im][t]); }
}
bool e10WriteUnmapped(Parameters &P, const std::vector<uint32_t> &reads, int nThreads, std::string &why) {
    std::vector<std::string> out[2];
    if (!e10FormatUnmapped(P, reads, nThreads, out, why)) return false;
    e10PutUnmapped(P, out);
    return true;
}

// E10 level 4: STAR's read source (star_api.h) with only the reads that have no GPU record, in chunk format (processChunks' FASTQ branch)
struct E10Source : StarReadSource {
    std::string buf[2];
    std::vector<uint64> end[2], iRead;   // per served read: where it ends in buf[0], buf[1]; its iReadAll
    uint64 k = 0;
    // E16: reads per chunk at most (level 5: ~4 chunks per mapping thread; STAR's 15 MB chunk held ~55 k of the 157 k reads of pbmc1k_v3, so
    // 3 of 18 threads mapped them all)
    uint64 maxReads = ~0ull;
    void fill(char **chunkIn, uint64_t *sz, uint64_t limit, uint64_t &iReadAll, uint64_t readMapNumber, uint32_t) override {
        const uint64 k0 = k;
        while (k < iRead.size() && sz[0] < limit && sz[1] < limit && k - k0 < maxReads) {
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

// E11: Gene/Summary.csv's "Q30 Bases in RNA read" is STAR's qualHist (ReadAlign_oneRead.cpp:34-38): per read the cDNA quality characters in
// [clip5, L - clip3). The decode counted whole reads; here a read with a record keeps only that span (its clip: the GPU's, which readLoad
// would have taken) and a read without one keeps nothing (STAR's loop counts it, with its own clip)
bool e11RnaAdjust(Parameters &P, std::string &why, int nThreads = 0) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint64 N = g_handoffNReads;
    const int T = std::max(1, nThreads > 0 ? nThreads : (int) P.runThreadN);
    const unsigned q30 = e11.qBase + 30;
    if (ingOn ? ing.n != N : (e10f.m[0] == nullptr || e10f.at[0].size() != N)) { why = "the cDNA file is not mapped"; return false; }
    std::vector<uint64> dq(T, 0), dn(T, 0), nNo(T, 0), nCl(T, 0);
    std::vector<int> bad(T, 0);
    std::vector<std::thread> th;
    for (int t = 0; t < T; t++) th.emplace_back([&, t]() {
        const char *m = e10f.m[0], *end = m + e10f.len[0];
        for (uint64 r = N * t / T; r < N * (t + 1) / T; r++) {
            const bool rec = g_handoffReadRecord[r] != 0xFFFFFFFFu;
            uint64 c5 = 0, c3 = 0;
            if (rec) {
                if (g_handoffClip5 == nullptr) continue;
                c5 = g_handoffClip5[r]; c3 = g_handoffClip3[r];
                if (c5 + c3 == 0) continue;
                nCl[t]++;
            } else nNo[t]++;
            if (ingOn) {   // E13: the quality from the packed reads
                const uint64 s0q = ingOff[r], L = ingOff[r + 1] - s0q;
                const uint32 b = ing.block(r);
                const uint64 s0 = rec ? std::min<uint64>(c5, L) : 0, s1 = rec ? (L >= c3 ? L - c3 : 0) : 0;
                uint64 h = 0, hs = 0, ns = 0;
                for (uint64 j = 0; j < L; j++) {
                    const uint64 x = (unsigned) ing.qualPhred[ing.qcode(b, s0q + j)] + 33 >= q30;
                    h += x;
                    if (j >= s0 && j < s1) { hs += x; ns++; }
                }
                dq[t] += h - hs; dn[t] += L - ns;
                continue;
            }
            const char *p = m + e10f.at[0][r];   // the name line; the quality line is the fourth
            for (int k = 0; k < 3 && p != nullptr; k++) { p = (const char *) memchr(p, '\n', end - p); if (p != nullptr) p++; }
            if (p == nullptr || p > end) { bad[t] = 1; return; }
            const char *q1 = (const char *) memchr(p, '\n', end - p);
            if (q1 == nullptr) q1 = end;
            uint64 L = q1 - p;
            if (L > 0 && (unsigned char) p[L - 1] < 33) L--;
            const uint64 s0 = rec ? std::min<uint64>(c5, L) : 0, s1 = rec ? (L >= c3 ? L - c3 : 0) : 0;
            uint64 h = 0, hs = 0, ns = 0;
            for (uint64 j = 0; j < L; j++) {
                const uint64 x = (unsigned char) p[j] >= q30;
                h += x;
                if (j >= s0 && j < s1) { hs += x; ns++; }
            }
            dq[t] += h - hs; dn[t] += L - ns;
        }
    });
    for (auto &x : th) x.join();
    for (int t = 0; t < T; t++) if (bad[t]) { why = "a cDNA quality line could not be read back"; return false; }
    for (int t = 0; t < T; t++) { e11.q30Rna -= dq[t]; e11.nRna -= dn[t]; e11.nNoRec += nNo[t]; e11.nClipped += nCl[t]; }
    e11.sAdj = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}
#endif

} // namespace

void seqCacheGpuPrefill(Parameters &P, Genome &G, SeqCache *&cache, Transcriptome *tr) {
    const char *e = getenv("SINGLET_ENGINE_GPU");
    if (e == nullptr || e[0] != '1' || cache == nullptr) {
#ifdef SINGLET_HAVE_ENGINE_GPU
        ingRelease();   // E13: a decode started for a prefill that does not run
#endif
        return;
    }
#ifndef SINGLET_HAVE_ENGINE_GPU
    P.inOut->logMain << "ENGINE_GPU: requested, but this binary was built without CUDA (SINGLET_ENGINE_GPU=OFF): STAR maps every read\n" << flush;
    (void) G; (void) tr;
#else
    auto skip = [&](const std::string &why) {
        ingRelease();
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
            if (P.pSolo.cbWLyes && !P.pSolo.cbWL.empty()) {   // R11's exact CB histogram over STAR's whitelist
                in.e10.wl = (const uint64_t *) P.pSolo.cbWL.data(); in.e10.nWL = P.pSolo.cbWL.size();
                in.e10.cbL = (uint32) P.pSolo.cbL; in.e10.umiL = (uint32) P.pSolo.umiL;
            }
        }
        if (e10Level >= 2 && !in.e10.on) P.inOut->logMain << "ENGINE_GPU: E10 (level " << e10Level << ") not used: " << w << "\n";
        // E11 (level 5): the Solo configuration E9 ports; otherwise level 3, where STAR's Solo still counts every read
        if (e10Level >= 5) {
            std::string w5 = in.e10.on ? "" : "E10 not used";
            if (in.e10.on && e11Config(P, tr, w5)) e11.want = true;
            else { e11.why = w5; P.inOut->logMain << "ENGINE_GPU: E11 (level 5) not used: " << w5 << "\n"; }
        }
    }
    // E13: with the pipeline's head-scan plan the reads come from the .1fq (no FASTQ files: STAR's read files are /dev/null)
    const bool ingest = singlet_engine_gpu::ingest::plan().on;
    if (ingest) {
        double waited = 0;
        if (!ingestReads(P, in, why, waited)) return skip("E13 decode: " + why);
        const IngPrep &W = *ingPrep;
        P.inOut->logMain << "ENGINE_GPU: E13: " << ing.n << " reads decoded from the .1fq into the packed buffers in " << W.secA << " s (" << P.runThreadN
                         << " threads; " << in.seq.size() << " cDNA bases, " << ing.qual.size() << " quality bytes, " << ing.r1.size() * 8 << " barcode-read bytes; "
                         << (!W.early ? std::string("decoded here (SINGLET_ENGINE_GPU_INGEST_EARLY=0)")
                             : (W.fromParent ? std::string("by the pipeline before it forked STAR") : "started with STAR's process, done " + std::to_string(W.doneA)
                                + " s later") + ", taken " + std::to_string(W.waitA) + " s after STAR read its parameters")
                         << "); barcodes " << W.secB << " s (R11 " << ing.r11Total << " reads), CellRanger4 clip " << W.secC << " s ("
                         << (!W.clipPre ? std::string("none")
                             : std::to_string(W.clipS) + " thread-s on " + std::to_string(W.clipDistinct) + " distinct sequences, grouped in "
                               + std::to_string(W.clipGroupS) + " s, " + (W.clipEarly ? "right after the decode with STAR's fixed CellRanger4 sequences (STAR's "
                               "ClipMates are those), done " + std::to_string(W.doneC) + " s after STAR's process started" : std::string("after the barcodes")))
                         << (W.cr4 && !W.clipEarly ? "; the clip after the decode was not STAR's" : "") << "); "
                         << (W.early ? "the prefill waited " : "all in the prefill: ") << waited << " s\n";
    } else
    if (!readSequences(P, in, why, (int) P.runThreadN)) return skip(why);
    if (e10Level >= 2 && in.e10.on) {
        std::string w;
        const auto tb = std::chrono::steady_clock::now();
        if (ingest) {}   // E13: the decode wrote the CR/UR keys
        else if (!e10Barcodes(P, in, w, (int) P.runThreadN)) { in.e10.on = false; decltype(in.e10.aux)().swap(in.e10.aux); e10f.release(); }
        P.inOut->logMain << "ENGINE_GPU: E10 decode: " << (in.e10.on ? "CR/UR keys and quality bits of " + std::to_string(in.off.size() - 1) + " reads ("
                                                        + std::to_string(in.e10.nw) + " quality words each), barcode file "
                                                        + std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count()) + " s"
                                                      : "not used: " + w) << "\n";
    } else if (e10Level >= 2 && !in.e10.on) {
        if (!e10DecodeWhy.empty()) P.inOut->logMain << "ENGINE_GPU: E10 decode: not used: " << e10DecodeWhy << "\n";
        e10f.release();
    }
    if (e11.want) {   // E11: the engine keeps Solo's inputs on the device (the reads' CR/UR, the records' Gene annotation, the read -> record map)
        std::string w5;
        if (!in.e10.on) w5 = "E10 not used";
        else if (e11Posterior(P, w5)) { in.e10.soloModel = &e11.M; in.e10.soloParams = &e11.S; }
        if (in.e10.soloModel == nullptr) { e11.want = false; e11.why = w5; P.inOut->logMain << "ENGINE_GPU: E11 (level 5) not used: " << w5 << "\n"; }
        else P.inOut->logMain << "ENGINE_GPU: E11 decode: CB+UMI qualities " << e11.q30Cb << " of " << e11.nCb << " >= Q30, RNA (whole reads) " << e11.q30Rna << " of "
                              << e11.nRna << ", CB quality characters " << e11.cbQmin << ".." << e11.cbQmax << " (posterior pq " << e11.S.pq << ")\n";
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
    // E13: R11's exact CB histogram from the decode to the pipeline (engine_gpu_ingest.h publishR11), before the device runs: the stream's CB
    // correction needs it whatever happens next; the device's, where E10 computes one, is compared after mapDistinct
    if (ingOn && e2r.base != nullptr && !ing.r11.empty())
        P.inOut->logMain << "ENGINE_GPU: E13: R11 exact CB histogram (" << ing.r11Total << " reads) "
                         << (singlet_engine_gpu::ingest::publishR11(e2r, ing.r11, ing.r11Total) ? "in the shared region for the pipeline" : "NOT written (region full)") << "\n";
    else if (e2r.fd >= 0 && e2r.base == nullptr) P.inOut->logMain << "ENGINE_GPU: E8 not used (it needs the GPU pileup \"on\" with summaries and per-read entries, CB_UMI_Simple)" << std::endl;
    bool lockReleased = false;   // mapDistinct releases the device when its device work is done (the host assembly of the records needs no GPU)
    const auto tHeld0 = std::chrono::steady_clock::now(); double heldS = 0;
    in.release = [&]() { if (!lockReleased) { lockReleased = true; heldS = std::chrono::duration<double>(std::chrono::steady_clock::now() - tHeld0).count(); singlet_engine_gpu::unlockDevice(lockFd); } };
    bool ok = singlet_engine_gpu::mapDistinct(Q, H, G.SA.lengthByte, G.SAi.lengthByte, in, device, hostThreads, out);
    in.release();
    singlet_tail::mark("prefill: mapDistinct returned (device held %.3f s)", heldS);
    time_t tGpu = time(nullptr);
    // E12: wall time of the steps after the engine (the "cache inserts" of the line below)
    auto tq = std::chrono::steady_clock::now();
    std::string tqLog;
    auto tmark = [&](const char *k) {
        const auto t = std::chrono::steady_clock::now();
        char b[64]; snprintf(b, sizeof b, "%s %.2f, ", k, std::chrono::duration<double>(t - tq).count());
        tqLog += b; tq = t;
    };
    P.inOut->logMain << "ENGINE_GPU: device " << device << " (" << singlet_engine_gpu::deviceId(device) << "), waited " << (tLock - tWait)
                     << " s for it, held it " << heldS << " s\n" << out.log << out.e10.log;
    decltype(in.e10.aux)().swap(in.e10.aux);
    if (!ok) { e10f.release(); return skip("GPU mapping failed"); }
    bool e10ok = out.e10.ok;   // E10: the device wrote the entries and masks into the region and computed the per-read products
    // E11: level 5 when the engine kept Solo's inputs (then no summaries: no read with a record reaches STAR's loop); otherwise level 3
    e11.ready = e11.want && e10ok && out.e10.soloKept;
    if (e11.want && !e11.ready) {
        e11.why = !e10ok ? "E10 products not made" : "the engine did not keep Solo's inputs";
        P.inOut->logMain << "ENGINE_GPU: E11 (level 5) not used: " << e11.why << "\n";
        e9::soloKeepRelease();
    }
    e11.device = device;
    if (const char *rp = getenv("SINGLET_ENGINE_GPU_E10_R11"); rp != nullptr && !out.e10.r11.empty()) {   // the histogram in r11_hist.cpp's format
        FILE *f = fopen(rp, "wb");
        uint64 nnz = 0;
        for (uint32_t c : out.e10.r11) nnz += c != 0;
        const uint64 h[3] = {(uint64) out.e10.r11.size(), (uint64) out.e10.r11Total, nnz};
        bool okf = f != nullptr && fwrite("E6R11H01", 1, 8, f) == 8 && fwrite(h, 8, 3, f) == 3;
        for (uint64 w = 0; okf && w < out.e10.r11.size(); w++)
            if (out.e10.r11[w]) { const uint64 pr[2] = {w, out.e10.r11[w]}; okf = fwrite(pr, 8, 2, f) == 2; }
        if (f != nullptr) okf = (fclose(f) == 0) && okf;
        P.inOut->logMain << "ENGINE_GPU: E10: R11 exact CB histogram from the device: " << out.e10.r11Total << " reads, " << nnz << " barcodes -> " << rp
                         << (okf ? "" : " (write FAILED)") << "\n";
    }
    if (ingOn && !out.e10.r11.empty()) {   // E13: the decode's R11 against the device's (both from the same CR/UR keys)
        uint64 nd = 0;
        const bool same = out.e10.r11.size() == ing.r11.size();
        for (uint64 w = 0; same && w < ing.r11.size(); w++) nd += ing.r11[w] != out.e10.r11[w];
        P.inOut->logMain << "ENGINE_GPU: E13: R11 histogram of the decode vs the device's: " << (same && nd == 0 ? "identical" : "DIFFERENT") << " ("
                         << ing.r11Total << " vs " << out.e10.r11Total << " reads" << (same ? ", " + std::to_string(nd) + " entries differ" : ", sizes differ") << ")\n";
    }
    if (in.clip.pre) {   // CellRanger4: readLoad takes clippedInfo from here, clipChunk is skipped
        static std::vector<uint8> cr4keep;
        cr4keep.swap(in.cr4info);
        g_cr4Info = cr4keep.data(); g_cr4InfoN = cr4keep.size();
    }
    // a cache large enough for every result (entries are ~(32 + L/4 + blob) bytes; 3x for the ring, the 2-way index and
    // the shards' imbalance), unless SINGLET_STAR_SEQCACHE_MB was set
    uint64 nRec = out.seqOff.size() - 1;
    // E12: made when results are inserted (before: here, at every level; at E10 level >= 3 nothing is inserted, and the new cache's
    // zeroed index cost ~1 s of the prefill)
    bool cacheSized = getenv("SINGLET_STAR_SEQCACHE_MB") != nullptr;
    auto sizeCache = [&]() {
        if (cacheSized) return;
        cacheSized = true;
        uint64 need = (out.blob.size() + out.seq.size() / 4 + 40 * nRec) * 3;
        uint64 mb = std::max<uint64>(4096, (need >> 20) + 1);
        setenv("SINGLET_STAR_SEQCACHE_MB", std::to_string(mb).c_str(), 1);
        SeqCache *c2 = SeqCache::create(P, G.nGenome);
        if (c2 != nullptr) {
            c2->verify = cache->verify;
            delete cache;
            cache = c2;
        };
        singlet_tail::mark("prefill: sequence cache re-created (%llu MB)", (unsigned long long) mb);
    };
    // insert, sharded by hash across threads (SeqCache is thread-safe per shard); keep(i) picks the records
    auto insertRecords = [&](auto keep) {
        sizeCache();
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
    tmark("start");
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
        const bool e10NoSum = e10ok && (e10Level == 4 || (e10Level >= 5 && e11.ready)) && readSourceFree();
        std::vector<std::thread> th;
        // E16 (level 5): without summaries the threads would only construct (and leak) a ReadAlign, a Transcriptome copy and an OutSJ each
        const bool e16NoThreads = e10NoSum && e2eRec && singlet_tail::cut("sumthreads");
        for (int t = 0; t < (e16NoThreads ? 0 : nt); t++) th.emplace_back([&, t]() {
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
        singlet_tail::mark("prefill: summary threads joined (%d threads, no summaries: %d)", nt, (int) e10NoSum);
        tmark("hand-off records");
        if (g_handoffLeanAln.load() > 0)   // SINGLET_HANDOFF_LEAN=verify
            P.inOut->logMain << "ENGINE_GPU: lean alignment fields: " << g_handoffLeanAln.load() << " alignments, " << g_handoffLeanBad.load()
                             << " differ from alignBAM's" << "\n";
        static std::vector<uint8> recAln;   // per record its alignment count (the first byte of each record)
        recAln.resize(nRec);
        if (e2eRec) for (uint64 r = 0; r < nRec; r++) recAln[r] = e2r.recAln(r);
        else
        for (int t = 0, r = 0; t < nt; t++)
            for (uint64 o : partOff[t]) recAln[r++] = part[t][o];
        singlet_tail::mark("prefill: alignment counts of %llu records", (unsigned long long) nRec);
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
                singlet_tail::mark("prefill: region complete (magic written, %llu MB in use)", (unsigned long long) (E2H.used >> 20));
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
    tmark("hand-off install");
    // E10 level >= 3: STAR's loop takes the per-read products of the reads with a record from the device (stats, junctions, unmapped records);
    // level 4 also keeps those reads out of STAR's loop (its read source serves the others only)
    if (e10ok && e10Level >= 3 && g_handoffReadRecord != nullptr && g_handoffSum != nullptr) {
        const auto t3 = std::chrono::steady_clock::now();
        typedef singlet_engine_gpu::Output::E10 X;
        Stats S;
        S.resetN();
        const uint64_t *v = out.e10.stats;
        S.readN = v[X::S_readN]; S.readBases = v[X::S_readBases]; S.mappedReadsU = v[X::S_U]; S.mappedReadsM = v[X::S_M];
        S.mappedMismatchesN = v[X::S_mm]; S.mappedInsN = v[X::S_insN]; S.mappedDelN = v[X::S_delN]; S.mappedInsL = v[X::S_insL]; S.mappedDelL = v[X::S_delL];
        S.mappedBases = v[X::S_bases]; S.mappedPortion = out.e10.mappedPortion;
        for (int k = 0; k < SJ_MOTIF_SIZE; k++) S.splicesN[k] = v[X::S_spl0 + k];
        S.splicesNsjdb = v[X::S_sjdb];
        S.unmappedOther = v[X::S_uOther]; S.unmappedShort = v[X::S_uShort]; S.unmappedMismatch = v[X::S_uMismatch]; S.unmappedMulti = v[X::S_uMulti];
        S.unmappedAll = v[X::S_uAll];
        std::string w;
        bool ok3 = true;
        const size_t nUnmappedRec = out.e10.unmappedReads.size();
        // E16 bg: level 5 either runs or stops the run (no level-3 fallback once the source is built), so the records can wait for the thread
        const bool e16bgOn = e10Level >= 5 && e11.ready && P.outReadsUnmapped == "Fastx" && singlet_tail::cut("bg");
        if (e16bgOn) e16bg.reads.swap(out.e10.unmappedReads);
        else
        if (P.outReadsUnmapped == "Fastx" && !e10WriteUnmapped(P, out.e10.unmappedReads, (int) P.runThreadN, w)) ok3 = false;
        singlet_tail::mark("prefill: %zu Unmapped.out records %s", nUnmappedRec, e16bgOn ? "left to the background thread" : "written");
        if (ok3) {
            g_statsAll.addStats(S);
            e10sj.swap(out.e10.sj); e10nSJ = out.e10.nSJ;
            g_e10Level = 3;
            if (e10Level >= 4 && readSourceFree() && (e10Level == 4 || e11.ready)) {
                if (e10BuildSource(P, g_handoffReadRecord, g_handoffNReads, w)) { g_starReadSource = &e10src; g_e10Level = e10Level >= 5 ? 5 : 4;
                    if (g_e10Level == 5 && singlet_tail::cut("chunk")) {   // E16: the reads spread over every mapping thread (~4 chunks each)
                        const uint64 nT = std::max<uint64>(1, P.runThreadN);
                        e10src.maxReads = std::max<uint64>(256, (e10src.iRead.size() + 4 * nT - 1) / (4 * nT));
                    }
                    singlet_tail::mark("prefill: read source of the reads without a record built (%zu reads, at most %llu per chunk)", e10src.iRead.size(),
                                       (unsigned long long) e10src.maxReads); }
                else if (e10Level >= 5) exitWithError("EXITING because of FATAL ERROR (singlet --engine gpu, E11): level 5's read source could not be built (" + w
                                                      + ") and the reads with a record have no summaries\n", std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
                else P.inOut->logMain << "ENGINE_GPU: E10 level 4 not used (" << w << "): level 3\n";
            }
            if (g_e10Level == 5 && e16bgOn) {   // E16 bg: Unmapped.out's records and the RNA Q30 adjustment beside STAR's loop
                e16bg.on = true;
                e16bg.th = std::thread([&P]() {
                    const auto a = std::chrono::steady_clock::now();
                    e16bg.okU = e10FormatUnmapped(P, e16bg.reads, 4, e16bg.out, e16bg.whyU);
                    const auto b = std::chrono::steady_clock::now();
                    e16bg.okR = e11RnaAdjust(P, e16bg.whyR, 4);
                    e16bg.sU = std::chrono::duration<double>(b - a).count();
                    e16bg.sR = std::chrono::duration<double>(std::chrono::steady_clock::now() - b).count();
                    singlet_tail::mark("prefill (bg): %zu Unmapped.out records formatted in %.3f s, RNA Q30 adjustment %.3f s", e16bg.reads.size(), e16bg.sU, e16bg.sR);
                });
            } else
            if (g_e10Level == 5) {   // E11: Gene/Summary.csv's RNA Q30 counts of the reads with a record over STAR's span [clip5, L - clip3)
                std::string w5;
                if (!e11RnaAdjust(P, w5)) exitWithError("EXITING because of FATAL ERROR (singlet --engine gpu, E11): " + w5 + "\n", std::cerr, P.inOut->logMain,
                                                        EXIT_CODE_INPUT_FILES, P);
                singlet_tail::mark("prefill: RNA Q30 adjustment done");
            }
            if (e16bgOn && !e16bg.on) {   // E16 bg, but level 5 did not start the thread: the records are written here, as before
                std::string wu;
                if (!e10WriteUnmapped(P, e16bg.reads, (int) P.runThreadN, wu)) exitWithError("EXITING because of FATAL ERROR (singlet --engine gpu, E16): " + wu + "\n",
                                                                                         std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
                std::vector<uint32_t>().swap(e16bg.reads);
            }
            P.inOut->logMain << "ENGINE_GPU: E10 level " << g_e10Level << ": " << S.readN << " reads with a record (" << S.mappedReadsU << " unique, " << S.mappedReadsM
                             << " multi, " << S.unmappedAll << " unmapped: " << nUnmappedRec << " Unmapped.out records" << (e16bg.on ? ", written after STAR's loop" : " written") << "), " << e10nSJ
                             << " junctions from the device; " << (g_e10Level >= 4 ? std::to_string(e10src.iRead.size()) + " reads without a record go to STAR's loop" : "STAR's loop keeps readLoad, the barcode and the Solo record of these reads")
                             << " (" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t3).count() << " s)\n";
        } else {
            P.inOut->logMain << "ENGINE_GPU: E10 level 3 not used (" << w << "); STAR's loop does the per-read work\n";
        }
    }
    if (e11.ready && g_e10Level != 5) {   // E11: level 5 not reached (E10's region or products): STAR's Solo counts every read
        P.inOut->logMain << "ENGINE_GPU: E11 (level 5) not used: E10 runs at level " << g_e10Level << "\n";
        e11.ready = false;
        e9::soloKeepRelease();
    }
    if (g_e10Level < 4) { std::vector<uint32_t>().swap(out.e10.unmappedReads); }
    tmark("E10 products");
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
    tmark("cache inserts");
    P.inOut->logMain << "ENGINE_GPU: E12 after the engine (s): " << tqLog << "\n";
    ingRelease();   // E13: the packed reads' host planes (their text is in Unmapped.out and level 4/5's read source by now)
    P.inOut->logMain << "ENGINE_GPU: cache filled with " << nRec << " sequence results in " << (time(nullptr) - t0) << " s (reading reads "
                     << (tRead - t0) << " s, GPU " << (tGpu - tLock) << " s, cache inserts " << (time(nullptr) - tGpu) << " s; device " << device << ")\n" << flush;
    if (g_e10Level == 5 && singlet_tail::cut("free")) {   // E16 "free": the reads (decode) and the engine's output are kept (1.1 s to free at T=18)
        e16held.in = new singlet_engine_gpu::Input(std::move(in));      // until STAR's loop is done; seqCacheGpuAfterMapping frees them on a thread
        e16held.out = new singlet_engine_gpu::Output(std::move(out));   // (the buffers move with them; nothing after this reads them)
    }
    singlet_tail::mark("prefill: end");
#endif
}

#ifdef SINGLET_HAVE_ENGINE_GPU
// E13 (engine_gpu_ingest.h): the pipeline's STAR child, before its CR4 probe. Phase A on a thread of its own (unless the pipeline decoded the
// reads before the fork: takeParent), then with cr4 the CellRanger4 clip with the ClipMates ParametersClip builds for --clipAdapterType CellRanger4 and no other clip option (initialize: the 10x TSO at 5'
// unless --clip5pAdapterSeq names one, "A" (polyA) at 3'; N and NafterAd 0; the default adMMp 0.1); ingPrepare keeps it only when STAR's own
// ClipMates give the same clipKeyOf.
void singlet_engine_gpu::ingest::startEarly(int threads, bool cr4) {
    if (ingPrepHere() || !plan().on) return;
    ingPrep = new IngPrep;
    IngPrep *W = ingPrep;
    W->pid = getpid(); W->early = true; W->cr4 = cr4; W->threads0 = std::max(1, threads);
    W->t0 = std::chrono::steady_clock::now();
    W->fromParent = takeParent(W->reads);   // the pipeline decoded the reads before it forked this process
    W->th0 = std::thread([W]() {
        W->okA = W->fromParent || decodeReads(W->threads0, W->reads, W->why);
        W->doneA = ingSince(W->t0);
        if (!W->okA || !W->cr4) return;
        ClipMate m5, m3;
        m5.type = 10; m5.initialize(0, "AAGCAGTGGTATCAACGCAGAGTACATGGG", 0, 0.1);
        m3.type = 11; m3.initialize(0, "A", 0, 0.1);
        const auto tc = std::chrono::steady_clock::now();
        cr4Clip(W->reads, m5, m3, W->threads0, *W);
        W->secC = ingSince(tc); W->doneC = ingSince(W->t0);
        W->clipKey = clipKeyOf(m5, m3); W->clipEarly = true;
        delete m5.cr4;
    });
}
#endif

// STAR.cpp, after its parameters and before the genome load (E13, SINGLET_ENGINE_GPU_INGEST=1): the decode's phase B (barcodes), and C (the
// CellRanger4 clip) unless startEarly's thread did it with STAR's ClipMates, on a thread of their own once that thread is done, so all of it
// runs beside the genome load; the prefill joins it. Nothing when the decode did not start in this process (the CR4 probe's forked STARs,
// SINGLET_ENGINE_GPU_INGEST_EARLY=0).
void seqCacheGpuEarly(Parameters &P) {
#ifdef SINGLET_HAVE_ENGINE_GPU
    const char *e = getenv("SINGLET_ENGINE_GPU");
    if (e == nullptr || e[0] != '1' || !singlet_engine_gpu::ingest::plan().on || !ingPrepHere() || ingPrep->started) return;
    IngPrep *W = ingPrep;
    W->started = true;
    Parameters *pp = &P;
    W->th = std::thread([W, pp]() { ingPrepare(*pp, *W); });
#else
    (void) P;
#endif
}

// STAR.cpp, right after the mapping threads finished: E10's after-mapping steps (the entry check at level 2; the device's junctions into the
// first thread's junction buffer at level >= 3, collapsed, so outputSJ merges them with STAR's own)
void seqCacheGpuAfterMapping(Parameters &P, ReadAlignChunk **RAchunk) {
#ifdef SINGLET_HAVE_ENGINE_GPU
    singlet_tail::mark("after mapping: start");
    // E11's Solo device part (the gather of STAR's loop annotations, soloRunKept under the device lock) as a step: E16 "solofirst" (level 5) runs it
    // first, before the other after-mapping steps, so that it holds the device before the pipeline's pileup asks for it (the BAM stream has just
    // closed); otherwise it runs where E11 ran it
    struct SoloDev { bool done = false; uint64 nHost = 0, starQ30 = 0, starN = 0; double sGather = 0;
                     std::chrono::steady_clock::time_point t0, tw, tl, tr; e9::SoloResult res; std::string log; } sd;
    auto soloDev = [&]() {
        const auto t0 = std::chrono::steady_clock::now(); sd.t0 = t0;
        // the Gene annotation of the reads STAR's loop mapped (ReadAlign::e11Annot: read index, packed RecAnn), in read order
        uint64 &nHost = sd.nHost, &starQ30 = sd.starQ30, &starN = sd.starN;
        for (int i = 0; i < (int) P.runThreadN; i++) {
            nHost += RAchunk[i]->RA->e11Ann.size() / 2;
            for (uint32 ix = 0; ix < 256; ix++) {   // SoloFeature_statsOutput.cpp:25-35 over STAR's loop (the reads without a record)
                starN += RAchunk[i]->RA->qualHist[0][ix];
                if (ix >= e11.qBase + 30) starQ30 += RAchunk[i]->RA->qualHist[0][ix];
            }
        }
        std::vector<std::pair<uint64, uint64>> ha;
        ha.reserve(nHost);
        for (int i = 0; i < (int) P.runThreadN; i++) {
            std::vector<uint64> &v = RAchunk[i]->RA->e11Ann;
            for (size_t k = 0; k + 1 < v.size(); k += 2) ha.emplace_back(v[k], v[k + 1]);
            std::vector<uint64>().swap(v);
        }
        std::sort(ha.begin(), ha.end());
        std::vector<uint32_t> hostIdx(nHost);
        std::vector<e9::RecAnn> hostAnn(nHost);
        for (uint64 k = 0; k < nHost; k++) {
            hostIdx[k] = (uint32_t) ha[k].first;
            const uint64 a = ha[k].second;
            hostAnn[k] = e9::RecAnn{(uint32_t) a, (uint16_t) (a >> 32), (uint8_t) (a >> 48), (uint8_t) (a >> 56)};
        }
        std::vector<std::pair<uint64, uint64>>().swap(ha);
        sd.sGather = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        int lockFd = -1;
        sd.tw = std::chrono::steady_clock::now();
        singlet_engine_gpu::lockDeviceSlot(e11.device, &lockFd);
        sd.tl = std::chrono::steady_clock::now();
        e9::SoloResult &res = sd.res;
        std::string &log = sd.log;
        const bool ok = nHost < (1ull << 32) && g_handoffNReads < (1ull << 32)
            && e9::soloRunKept(e11.device, e11.S, e11.M, nHost, hostIdx.data(), hostAnn.data(), res, log);
        singlet_engine_gpu::unlockDevice(lockFd);
        sd.tr = std::chrono::steady_clock::now();
        if (!ok) exitWithError("EXITING because of FATAL ERROR (singlet --engine gpu, E11): Solo's counting on the device failed: " + log, std::cerr,
                               P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
        sd.done = true;
        singlet_tail::mark("after mapping: Solo on the device done (gather %.3f s, device wait %.3f s, soloRunKept %.3f s)", sd.sGather,
                           std::chrono::duration<double>(sd.tl - sd.tw).count(), std::chrono::duration<double>(sd.tr - sd.tl).count());
    };
    if (g_e10Level == 5 && singlet_tail::cut("solofirst")) soloDev();
    if (e16bg.on) {   // E16 bg: the thread's records into STAR's streams (the mapping threads are done with them), its RNA counts for Solo's summary
        if (e16bg.th.joinable()) e16bg.th.join();
        e16bg.on = false;
        if (!e16bg.okU) exitWithError("EXITING because of FATAL ERROR (singlet --engine gpu, E16): Unmapped.out records not formatted: " + e16bg.whyU + "\n",
                                      std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
        if (!e16bg.okR) exitWithError("EXITING because of FATAL ERROR (singlet --engine gpu, E11): " + e16bg.whyR + "\n", std::cerr, P.inOut->logMain,
                                      EXIT_CODE_INPUT_FILES, P);
        e10PutUnmapped(P, e16bg.out);
        std::vector<uint32_t>().swap(e16bg.reads);
        P.inOut->logMain << "ENGINE_GPU: E16: Unmapped.out records formatted beside STAR's loop in " << e16bg.sU << " s and appended after it; RNA Q30 adjustment "
                         << e16bg.sR << " s\n" << std::flush;
        singlet_tail::mark("after mapping: background thread joined, Unmapped.out records appended");
    }
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
        singlet_tail::mark("after mapping: device junctions merged");
    }
    std::vector<uint8>().swap(e10sj);
    e10f.release();
    if (g_e10Level == 5) {   // E11: Solo's counting on the device (E9's soloRun on the kept inputs) and star_Solo.out/ from it
        if (!sd.done) soloDev();
        const uint64 nHost = sd.nHost, starQ30 = sd.starQ30, starN = sd.starN; const double sGather = sd.sGather;
        const auto tw = sd.tw, tl = sd.tl, tr = sd.tr;
        e9::SoloResult &res = sd.res; std::string &log = sd.log;
        // star_Solo.out/: STAR creates Gene/, raw/ and filtered/ in its Solo step (SoloFeature_processRecords.cpp, _outputResults.cpp, cellFiltering)
        const std::string pre = P.outFileNamePrefix + P.pSolo.outFileNames[0];
        for (const std::string &d : {pre, pre + "Gene/", pre + "Gene/raw/", pre + "Gene/filtered/"}) mkdir(d.c_str(), P.runDirPerm);
        e9::SoloSummaryIn sum;
        sum.readN = g_statsAll.readN; sum.mappedU = g_statsAll.mappedReadsU; sum.mappedM = g_statsAll.mappedReadsM;
        sum.q30CbUmi = e11.q30Cb; sum.nCbUmi = e11.nCb; sum.q30Rna = e11.q30Rna + starQ30; sum.nRna = e11.nRna + starN; sum.qualBase = e11.qBase;
        e9::SoloWriteInfo info;
        const auto tws = std::chrono::steady_clock::now();
        const bool okw = e9::soloWrite(pre, e11.S, e11.M, res, e11.FP, {}, info, log, &sum);
        const auto te = std::chrono::steady_clock::now();
        singlet_tail::mark("after mapping: Solo gather %.3f s, device wait %.3f s, soloRunKept %.3f s, soloWrite %.3f s", sGather,
                           std::chrono::duration<double>(tl - tw).count(), std::chrono::duration<double>(tr - tl).count(),
                           std::chrono::duration<double>(te - tws).count());
        if (!okw) exitWithError("EXITING because of FATAL ERROR (singlet --engine gpu, E11): star_Solo.out/ not written: " + log, std::cerr, P.inOut->logMain,
                                EXIT_CODE_INPUT_FILES, P);
        g_e11SoloDone = true;
        auto sec = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
        std::ostringstream ms;
        for (auto &kv : res.ms) ms << " " << kv.first << " " << kv.second;
        P.inOut->logMain << "ENGINE_GPU: E11: Solo on the device: " << g_handoffNReads << " reads (" << nHost << " annotated by STAR's loop; RNA Q30 adjustment "
                         << e11.nNoRec << " reads without a record, " << e11.nClipped << " clipped, " << e11.sAdj << " s), " << res.nDistinctCR << " distinct CRs, "
                         << res.nKeys << " counted reads, " << res.nTuples << " distinct (CB,gene,UMI), " << res.nSeg << " (CB,gene); gather " << sGather
                         << " s, waited " << sec(tw, tl) << " s for the device, soloRunKept " << sec(tl, tr) << " s (ms:" << ms.str() << "), write " << sec(tws, te)
                         << " s (" << info.nCB << " CBs, " << info.nCells << " cells, nUMImax " << info.nUMImax << ", nUMImin " << info.nUMImin
                         << "); STAR's Solo step skipped\n" << log << std::flush;
    }
    if (e16held.in != nullptr || e16held.out != nullptr) {   // E16 "free": the prefill's reads and engine output, on a thread (STAR's loop is done)
        singlet_engine_gpu::Input *fIn = e16held.in; singlet_engine_gpu::Output *fOut = e16held.out;
        e16held = E16Held();
        std::thread([fIn, fOut]() {
            const double a = singlet_tail::now();
            delete fIn; delete fOut;
            singlet_tail::mark("after mapping (bg): the prefill's reads and engine output freed in %.3f s", singlet_tail::now() - a);
        }).detach();
    }
#else
    (void) P; (void) RAchunk;
#endif
}
