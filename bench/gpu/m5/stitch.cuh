// M5: STAR's read mapping after the seed search, ported for the GPU (and the host, from the same source).
// docs/gpu-engine/exploration.md §12.13.
//
// One read per thread, from the read sequence to STAR's final transcripts:
//   qualitySplit -> the seed loop of ReadAlign::mapOneRead -> maxMappableLength2strands (all iDist passes) -> storeAligns
//   -> stitchPieces (createExtendWindowsWithAlign, flanks, assignAlignToWindow with FASTASSIGN) -> stitchWindowAligns
//   (stitchAlignToTranscript, extendAlign, blocksOverlap, the score-range recording) -> multMapSelect -> mappedFilter
// and writes the result in the encoding of STAR's sequence cache (ReadAlign_seqCache.cpp), so that it can be compared
// byte for byte with what STAR computed for the same sequence (src/gpu/stardump/m5_dump.h).
//
// Each function is a line-by-line port of the STAR 2.7.11b source vendored in prod/campaign (c4a1be38); the STAR file is
// named above each one. Differences are representation only:
//  - winBin (two arrays of nGenome/2^winBinNbits bins per thread in STAR) is a small open-addressing map with the same
//    read/write semantics (absent = uintWinBinMax);
//  - the recursion of stitchWindowAligns runs on an explicit stack of transcripts (one per depth, as the by-value
//    arguments of the recursion), with the same visiting order;
//  - the transcript slots of a window (trArrayPointer) are a local pointer array with the same permutations; only the
//    transcripts that can still pass multMapSelect's score range are kept across windows (their order is kept);
//  - every array has a capacity (template parameter). A read that would exceed one where STAR's own limit is larger
//    stops with Status::Overflow and goes to the heavy lane (the same code with STAR's limits).
//  - scoreGenomicLength's log2 uses breakpoints computed on the host with the C library's log2 (exact).
// Single-end only (readNmates = 1), no chimeric detection, no variation, no genome transformation, no BySJout stage 2:
// the driver checks these parameters.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// E5 (compact) variants, compile-time, each exact (docs/gpu-engine/fable/E5-compact.md):
//  M5_REFX   the exclude branch of stitchWindowAligns references its parent's transcript instead of copying it
//  M5_FIN1   finalization works on a spare slot (one copy instead of trA -> fin -> slot); recording swaps it into the
//            slot permutation
//  M5_SLOT0  a window's slot 0 gets only trInit's maxScore (the one field read while no transcript is recorded)
//  M5_S3     = M5_REFX + M5_FIN1 + M5_SLOT0
//  M5_KEEPIX the kept transcripts live in the slot pool and keep[] is an index list: keeping a window's transcript,
//            dropping one and multMapSelect's compaction move indices instead of transcripts (needs M5_FIN1)
//  M5_TR32   transcript fields that hold read offsets, lengths and counts stored in 32 bits (genome coordinates stay 64)
//  M5_TR32CHECK  count every 32-bit store that would not round-trip (g_tr32bad; must stay 0)
//  M5_BLOBW  the output encoder gathers bytes in a register and stores 8 at a time (aligned u64 stores) instead of one
//            byte store per byte; the same bytes, the same length, the same overflow rule
//  M5_ENC1   encodeTr's loops are not unrolled (one inlined putV per loop instead of one per field: less code in k_p3)
// E5 (docs/gpu-engine/fable/E5-compact.md): the exact variants S3 (exclude branch by reference, one finalize copy, no slot-0
// init), TR32 (32-bit transcript fields) and ENC1 (one encoder call site) are on by default: +18% M5, identical results.
// -DM5_BASELINE restores the pre-E5 code paths for comparison.
#ifndef M5_BASELINE
#ifndef M5_S3
#define M5_S3
#endif
#ifndef M5_TR32
#define M5_TR32
#endif
#ifndef M5_ENC1
#define M5_ENC1
#endif
#endif
#ifdef M5_S3
#ifndef M5_REFX
#define M5_REFX
#endif
#ifndef M5_FIN1
#define M5_FIN1
#endif
#ifndef M5_SLOT0
#define M5_SLOT0
#endif
#endif
#ifdef M5_TR32CHECK
#ifndef M5_TR32
#define M5_TR32
#endif
#endif
#if defined(M5_KEEPIX) && !defined(M5_FIN1)
#error "M5_KEEPIX needs M5_FIN1"
#endif

namespace m5 {

typedef unsigned long long u64;
typedef long long i64;
typedef u64 uint;     // STAR's `uint` is 64-bit (IncludeDefine.h)

constexpr int scoreMatch = 1;
constexpr u64 MARK_FRAG_SPACER_BASE = 11;
constexpr int MAX_SJ_REPEAT_SEARCH = 255;
constexpr int DEF_readSeqLengthMax = 650;
constexpr u64 uintWinBinMax = 65535;
constexpr u64 MARKER_ALL_PIECES_EXCEED_seedMultimapNmax = 999901;
constexpr u64 MARKER_NO_GOOD_WINDOW = 999903;
constexpr u64 MARKER_NO_GOOD_PIECES = 999904;
constexpr u64 MARKER_TOO_MANY_ANCHORS_PER_WINDOW = 999905;
constexpr u64 MARKER_READ_TOO_SHORT = 999910;
enum { PC_rStart, PC_Length, PC_Str, PC_Dir, PC_Nrep, PC_SAstart, PC_SAend, PC_iFrag, PC_SIZE };
enum { WC_Str, WC_Chr, WC_gStart, WC_gEnd, WC_SIZE };
enum { WA_Length, WA_rStart, WA_gStart, WA_Nrep, WA_Anchor, WA_iFrag, WA_sjA, WA_SIZE };
enum { EX_R, EX_G, EX_L, EX_iFrag, EX_sjA, EX_SIZE };

enum class Status : uint8_t { Ok = 0, Overflow = 1, BlobFull = 2, Unsupported = 3 };
// which capacity overflowed (for the heavy-lane statistics)
enum OvWhy : uint8_t { OV_NONE, OV_LREAD, OV_SPLIT, OV_PC, OV_WIN, OV_WA, OV_WINBIN, OV_EXONS, OV_WTR, OV_KEEP, OV_BLOB, OV_STEPS };

// ---------------------------------------------------------------------------------------------------------------------
// Parameters (src/gpu/stardump/m5_dump.h writes them from the running STAR) and genome view
// ---------------------------------------------------------------------------------------------------------------------
struct Params {
    u64 maxNsplit, seedSplitMin, seedSearchStartLmax, seedMapMin, seedSearchLmax, seedMultimapNmax, seedPerReadNmax, seedPerWindowNmax;
    double seedSearchStartLmaxOverLread;
    u64 winBinChrNbits, winBinNbits, winAnchorDistNbins, winFlankNbins, winBinN, winAnchorMultimapNmax;
    u64 alignWindowsPerReadNmax, alignTranscriptsPerWindowNmax, alignTranscriptsPerReadNmax;
    u64 alignSJoverhangMin, alignSJDBoverhangMin, alignSplicedMateMapLmin;
    double alignSplicedMateMapLminOverLmate;
    u64 alignIntronMin, alignIntronMax, alignMatesGapMax;
    int alignSJstitchMismatchNmax[4];
    bool alignSoftClipAtReferenceEnds, alignEndsTypeExt[2][2], flushRight;
    int alignEndsProtrudeNbasesMax;
    int intronMotifsFilter;      // 0 None, 1 RemoveNoncanonical, 2 RemoveNoncanonicalUnannotated
    bool removeInconsistentStrands;
    u64 outSAMstrandFieldType;
    double scoreGenomicLengthLog2scale;
    int outFilterMultimapScoreRange; u64 outFilterMultimapNmax;
    bool multOrderRandom; u64 outSAMmultNmax; bool primaryAllBest;
    int outFilterScoreMin; double outFilterScoreMinOverLread;
    u64 outFilterMatchNmin; double outFilterMatchNminOverLread;
    u64 outFilterMismatchNmax; double outFilterMismatchNoverLmax, outFilterMismatchNoverReadLmax;
    int scoreGap, scoreGapNoncan, scoreGapGCAG, scoreGapATAC, scoreDelBase, scoreDelOpen, scoreInsBase, scoreInsOpen, scoreStitchSJshift;
    int sjdbScore; u64 gSAsparseD, gSAindexNbases;
    // genomic length score: value for gLength >= glBreak[k] is glVal[k] (k ascending); below glBreak[0] it is glVal0
    int glN; int glVal0; u64 glBreak[64]; int glVal[64];
};

struct GenomeView {
    const char* G;            // G[-200 .. nGenome+200): STAR's G = G1 + 200, padding = 5
    const u64* SA;            // STAR's packed SA (GstrandBit+1 bits per entry)
    const u64* SAi;           // STAR's packed SAindex (GstrandBit+3 bits per entry)
    u64 nGenome, nSA, GstrandBit, GstrandMask, saBits, saiBits, SAiMarkNmaskC, SAiMarkNmask, SAiMarkAbsentMaskC;
    u64 saiStart[32];
    const u64* chrStart; const u64* chrLength; const u64* chrBin; u64 nChr, chrBinN;
    u64 sjGstart, sjdbOverhang, sjdbLength, sjdbN;
    const u64* sjdbStart; const u64* sjdbEnd; const u64* sjDstart; const u64* sjAstart;
    const uint8_t* sjdbMotif; const uint8_t* sjdbShiftLeft; const uint8_t* sjdbShiftRight; const uint8_t* sjdbStrand;
#ifdef M5_G2
    // E7 M5_G2 (device only; null on the host): the genome text 2 bits per base (word w holds positions 32w - G2OFF ..,
    // bytes > 3 stored as 0) and a bit per 64-position block that holds a byte > 3 (N, padding, spacers)
    const u64* G2 = nullptr; const uint32_t* NB = nullptr;
#endif
};

// STAR's PackedArray::operator[]: an unaligned little-endian load at bit offset ii*wordLength, masked
HD inline u64 packed_get(const u64* a, u64 ii, u64 bits) {
    u64 b = ii * bits, w = b >> 6, sh = b & 63;
    u64 v = a[w] >> sh;
    if (sh) v |= a[w + 1] << (64 - sh);
    return v & ((1ULL << bits) - 1);
}

HD inline int ctz32(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __ffs((int)x) - 1;
#else
    return __builtin_ctz(x);
#endif
}
template <class T> HD inline T mn(T a, T b) { return a < b ? a : b; }
template <class T> HD inline T mx(T a, T b) { return a > b ? a : b; }
template <class T> HD inline void swp(T& a, T& b) { T t = a; a = b; b = t; }

// ---------------------------------------------------------------------------------------------------------------------
// M5_TR32: a transcript field that holds a STAR `uint` (64-bit) read offset, length or count, stored in 32 bits. Written by
// truncation, read back sign-extended, so every value v with -2^31 <= (i64)v < 2^31 round-trips exactly, including the
// 64-bit wrap-arounds STAR's unsigned arithmetic produces for small negative intermediates (sjA = (uint)-1, x += (uint)-1).
// Arithmetic on a field happens in 64 bits on the read-back value, as in STAR. Values: read positions and lengths
// (<= Lread <= 650), base counts (<= Lread), exon/junction counts (<= MAX_N_EXONS), gap lengths (< a window's span,
// < 2^24), sjdb indices, chromosome indices. M5_TR32CHECK counts every store that would not round-trip (g_tr32bad).
// ---------------------------------------------------------------------------------------------------------------------
#ifdef M5_TR32
#ifdef M5_TR32CHECK
#ifdef __CUDACC__
__device__ unsigned long long g_tr32bad;
#endif
inline unsigned long long h_tr32bad = 0;
HD inline void tr32Bad() {
#ifdef __CUDA_ARCH__
    atomicAdd(&g_tr32bad, 1ULL);
#else
    __atomic_fetch_add(&h_tr32bad, 1ULL, __ATOMIC_RELAXED);
#endif
}
#define M5_TR32_STORE(v, x) do { (v) = (int32_t)(i64)(x); if ((uint)(i64)(v) != (x)) tr32Bad(); } while (0)
#else
#define M5_TR32_STORE(v, x) ((v) = (int32_t)(i64)(x))
#endif
struct U32 {
    int32_t v;
    HD operator uint() const { return (uint)(i64)v; }
    HD U32& operator=(uint x) { M5_TR32_STORE(v, x); return *this; }
    HD U32& operator+=(uint x) { return *this = (uint)*this + x; }
    HD U32& operator-=(uint x) { return *this = (uint)*this - x; }
    HD U32& operator++() { return *this += 1; }
    HD uint operator++(int) { uint o = *this; *this += 1; return o; }
};
typedef U32 sfield;
// an exon row: EX_G 64-bit, EX_R, EX_L, EX_iFrag, EX_sjA 32-bit; row[k] as in STAR's uint exons[][EX_SIZE]
struct ExRow {
    uint g; U32 r, l, f, s;
    HD uint get(int k) const { return k == EX_G ? g : k == EX_R ? (uint)r : k == EX_L ? (uint)l : k == EX_iFrag ? (uint)f : (uint)s; }
    HD void set(int k, uint x) { if (k == EX_G) g = x; else if (k == EX_R) r = x; else if (k == EX_L) l = x; else if (k == EX_iFrag) f = x; else s = x; }
    struct Ref {
        ExRow& e; int k;
        HD operator uint() const { return e.get(k); }
        HD Ref& operator=(uint x) { e.set(k, x); return *this; }
        HD Ref& operator=(const Ref& o) { e.set(k, o.e.get(o.k)); return *this; }
        HD Ref& operator+=(uint x) { e.set(k, e.get(k) + x); return *this; }
        HD Ref& operator-=(uint x) { e.set(k, e.get(k) - x); return *this; }
    };
    HD Ref operator[](int k) { return Ref{*this, k}; }
    HD uint operator[](int k) const { return get(k); }
};
#else
typedef uint sfield;
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Transcript (Transcript.h): the fields seed stitching and multMapSelect write; exons capped at NEX (STAR: MAX_N_EXONS 20)
// ---------------------------------------------------------------------------------------------------------------------
template <int NEX> struct Tr {
#ifdef M5_TR32
    ExRow exons[NEX];
#else
    uint exons[NEX][EX_SIZE];
#endif
    sfield shiftSJ[NEX][2];
    int canonSJ[NEX];
    uint8_t sjAnnot[NEX], sjStr[NEX];
    sfield intronMotifs[3];
    uint8_t sjMotifStrand; bool sjYes;
    sfield nExons;
    int iFrag;
    sfield rStart, roStart, rLength;
    uint gStart, gLength, cStart;
    sfield Chr, Str, roStr;
    bool primaryFlag;
    sfield nMatch, nMM, mappedLength, extendL;
    int maxScore;
    sfield nGap, lGap, nDel, nIns, lDel, lIns;
    sfield nUnique, nAnchor;
    HD void reset() {
        extendL = 0; primaryFlag = false;
        rStart = 0; roStart = 0; rLength = 0; gStart = 0; gLength = 0;
        maxScore = 0; nMatch = 0; nMM = 0;
        nGap = 0; lGap = 0; lDel = 0; lIns = 0; nDel = 0; nIns = 0;
        nUnique = nAnchor = 0;
    }
    // the copy STAR makes (Transcript operator=), restricted to the live exon rows: rows >= nExons are never read before
    // stitching writes them (stitchAlignToTranscript writes a row's 5 fields and its junction fields when it adds the exon)
    HD void copyFrom(const Tr& o) {
        nExons = o.nExons; iFrag = o.iFrag;
        rStart = o.rStart; roStart = o.roStart; rLength = o.rLength; gStart = o.gStart; gLength = o.gLength; cStart = o.cStart;
        Chr = o.Chr; Str = o.Str; roStr = o.roStr; primaryFlag = o.primaryFlag;
        nMatch = o.nMatch; nMM = o.nMM; mappedLength = o.mappedLength; extendL = o.extendL; maxScore = o.maxScore;
        nGap = o.nGap; lGap = o.lGap; nDel = o.nDel; nIns = o.nIns; lDel = o.lDel; lIns = o.lIns; nUnique = o.nUnique; nAnchor = o.nAnchor;
        intronMotifs[0] = o.intronMotifs[0]; intronMotifs[1] = o.intronMotifs[1]; intronMotifs[2] = o.intronMotifs[2];
        sjMotifStrand = o.sjMotifStrand; sjYes = o.sjYes;
        uint ne = o.nExons < (uint)NEX ? o.nExons : (uint)NEX;
        for (uint i = 0; i < ne; i++) {
#ifdef M5_TR32
            exons[i] = o.exons[i];
#else
            for (int k = 0; k < EX_SIZE; k++) exons[i][k] = o.exons[i][k];
#endif
            shiftSJ[i][0] = o.shiftSJ[i][0]; shiftSJ[i][1] = o.shiftSJ[i][1];
            canonSJ[i] = o.canonSJ[i]; sjAnnot[i] = o.sjAnnot[i]; sjStr[i] = o.sjStr[i];
        }
    }
    // bytes copyFrom moves (statistics, M5_COPYSTATS): the scalar fields, then per live exon row
    HD int copyBytes() const {
        constexpr int hdr = sizeof(nExons) + sizeof(iFrag) + 3 * sizeof(rStart) + 3 * sizeof(gStart) + 3 * sizeof(Chr) + sizeof(primaryFlag)
                          + 4 * sizeof(nMatch) + sizeof(maxScore) + 8 * sizeof(nGap) + sizeof(intronMotifs) + sizeof(sjMotifStrand) + sizeof(sjYes);
        constexpr int row = sizeof(exons[0]) + sizeof(shiftSJ[0]) + sizeof(canonSJ[0]) + sizeof(sjAnnot[0]) + sizeof(sjStr[0]);
        int ne = nExons < (uint)NEX ? (int)nExons : NEX;
        return hdr + ne * row;
    }
    HD void add(const Tr* t) {
        maxScore += t->maxScore; nMatch += t->nMatch; nMM += t->nMM;
        nGap += t->nGap; lGap += t->lGap; lDel += t->lDel; nDel += t->nDel; lIns += t->lIns; nIns += t->nIns;
        nUnique += t->nUnique;
    }
};
// the extension results only (extendAlign writes extendL, maxScore, nMatch, nMM; add() reads the counters, all 0 after reset)
struct Ext { uint extendL; int maxScore; uint nMatch, nMM; };

// ---------------------------------------------------------------------------------------------------------------------
// Capacities. Main lane: small, per thread; heavy lane: STAR's limits.
// ---------------------------------------------------------------------------------------------------------------------
template <int LREAD_, int NSPLIT_, int NP_, int NW_, int NWA_, int NWB_, int NEX_, int NWTR_, int NKEEP_, int NBLOB_>
struct Caps {
    static constexpr int LREAD = LREAD_, NSPLIT = NSPLIT_, NP = NP_, NW = NW_, NWA = NWA_, NWB = NWB_, NEX = NEX_,
                         NWTR = NWTR_, NKEEP = NKEEP_, NBLOB = NBLOB_;
    static_assert(NBLOB_ % 8 == 0, "output slots are whole words (M5_BLOBW)");
};

// read buffers: STAR's Read1[0] (bases 0..3, N = 4), Read1[1] (complement), Read1[2] (reverse complement)
#ifdef M5_G2
// E7 M5_G2: plus Read1[0] packed 2 bits per base, forward (p2[0], base j at bits 2(j%32) of word j/32) and reversed
// (p2[1]: base j of the reversed read = Read1[0][L-1-j]); words past the read are 0
template <int L> struct ReadBuf { char r[3][L + 1]; unsigned long long p2[2][L / 32 + 4]; };
#else
template <int L> struct ReadBuf { char r[3][L + 1]; };
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Per-read state (ReadAlign members that mapping touches)
// ---------------------------------------------------------------------------------------------------------------------
template <class C> struct Work {
    typedef Tr<C::NEX> T;
    // read
    uint Lread, outFilterMismatchNmaxTotal;
    uint Nsplit, splitR[3][C::NSPLIT];
    // seeds (storeAligns)
    uint nP, nA, nUM[2];
    uint PC[C::NP][PC_SIZE];
    uint storedLmin, uniqLmax, uniqLmaxInd, multLmax, multLmaxN, multNmin, multNminL, multNmax, multNmaxL;
    // windows
    uint nW, nWall;
    uint WC[C::NW][WC_SIZE];
    uint nWA[C::NW], WALrec[C::NW];
    i64 WlastAnchor[C::NW];   // STAR: uint, set to -1, compared as uint (WlastAnchor[iW]<nWA[iW] / <iA)
    uint WA[C::NW][C::NWA][WA_SIZE];
    // winBin: key = (str << 40) | bin, value = window index; absent = uintWinBinMax
    u64 wbKey[C::NWB]; uint16_t wbVal[C::NWB]; int wbN; u64 wbEpoch;
    u64 wbBits[2][16];         // filter: bit (bin & 1023) of strand str is set if some bin with these low bits was written
    u64 nScan;                 // SA entries scanned by window creation and assignment (statistics)   // key = epoch << 41 | str << 40 | bin; other epochs = empty
    // transcripts
    int maxScoreMate[1];
    uint mapMarker;
    int trBestScore; uint trBestGLength; int trBestKeep;    // trBest: index into keep[] (-1 = trInit)
    uint trInitRLength;
    // windows' transcripts that can still be selected (in window order, then within-window order), and their counts
#ifdef M5_KEEPIX
    int keepP[C::NKEEP]; int nKeep; int keepWin[C::NKEEP];   // keep[k] = slot[keepP[k]]
    uint8_t own[C::NWTR + 2 + C::NKEEP];                     // slot pool entry held by keep[]
#else
    T keep[C::NKEEP]; int nKeep; int keepWin[C::NKEEP];
#endif
    uint nWtr;                // number of windows with transcripts (STAR's nW after stitching)
    uint trNtotal;
    // stitching scratch
    T stack[C::NWA + 1];
#if defined(M5_KEEPIX)
    T slot[C::NWTR + 2 + C::NKEEP]; int slotP[C::NWTR + 1]; uint nWinTr;
    int slotSpare;            // the physical slot outside the permutation slotP: finalization's scratch
#elif defined(M5_FIN1)
    T slot[C::NWTR + 2]; int slotP[C::NWTR + 1]; uint nWinTr;
    int slotSpare;            // the physical slot outside the permutation slotP: finalization's scratch
#else
    T slot[C::NWTR + 1]; int slotP[C::NWTR + 1]; uint nWinTr;
    T fin;                    // finalize scratch (trA copy at iA>=nA)
#endif
    // multMapSelect
    uint nTr; int maxScore; int unmapType;
    OvWhy ov;
    bool stitched; u64 nEntries;
    u64 nSteps, stepBudget;    // stitching recursion steps of this read; budget (0 = none): over it, the read goes to the next lane
    long long clk[8];          // cycles: seeds, windows + assignment, stitching, selection + encoding; inside stitching:
                               // [4] stitchAlignToTranscript, [5] finalization, [6] transcript copies, [7] recording (device only)
#ifdef M5_CENSUS
    // E2 census (docs/gpu-engine/fable/E2-census.md): statistics only, never read by the mapping itself
    u64 cnStitch, cnFinal, cnRec;   // stitchAlignToTranscript calls, finalizations (leaves), transcripts recorded
    u64 cnHits;                     // SA entries that hit a window (assignEntry calls that can do something)
    uint cnMaxEx;                   // largest nExons of a transcript passed to stitchAlignToTranscript (cap NEX needs > this)
    uint cnMaxWA;                   // peak aligns in one window during assignment (cap NWA)
    uint cnKeepMax, cnWinTrMax;     // peak transcripts kept across windows (NKEEP), per window (NWTR)
    uint cnWinStitched;             // windows that went through stitchWindowAligns
    bool cnFinish;                  // mapFinish ran for this read (the per-window arrays below are this read's)
    long long cn[4];                // cycles: [0] phase 1 (k_p1, whole call), [1] assignment scan (k_p2, warp), [2] assignment, [3] phase 3 total
    uint32_t cnWinSteps[C::NW], cnWinCyc[C::NW];   // per window: recursion steps, cycles of stitchWindowAligns
    uint16_t cnWinTr[C::NW];                   // per window: transcripts recorded (nWinTr after the window)
#endif
#ifdef M5_COPYSTATS
    // transcript copies by kind: [0] include (stack), [1] exclude (stack), [2] finalize scratch, [3] record into a slot,
    // [4] slot init at a window's start, [5] window -> keep, [6] keep compaction, [7] multMapSelect compaction
    unsigned long long cpN[8], cpB[8]; long long cpClk[8];
    unsigned long long stN[4];   // [0] recursion steps (frames visited), [1] windows stitched
#endif
};
#ifdef M5_CENSUS
#define M5CN(...) do { __VA_ARGS__; } while (0)
#else
#define M5CN(...) do { } while (0)
#endif
// The per-read state that crosses the staged kernels (phase 1 -> assignment scan -> phase 3): all of Work except the
// phase-3 scratch (WA, filled by the assignment in phase 3; the transcript arrays), which is written there before it is
// read. Copied in 8-byte words (every member boundary below is 8-byte aligned). Used to run phase 1 or phase 3 on a
// per-thread copy of Work in local memory (m5 --p1local / --p3local), where the hardware interleaves the threads of a
// warp word by word, i.e. a field-major (SoA) layout without changing the source.
template <class C> HD inline void copyPersistent(Work<C>& d, const Work<C>& s) {
    typedef Work<C> W;
    const size_t skipO[] = {offsetof(W, WA),
#ifndef M5_KEEPIX
                            offsetof(W, keep),
#endif
                            offsetof(W, stack), offsetof(W, slot),
#ifndef M5_FIN1
                            offsetof(W, fin),
#endif
                            sizeof(W)};
    const size_t skipN[] = {sizeof(s.WA),
#ifndef M5_KEEPIX
                            sizeof(s.keep),
#endif
                            sizeof(s.stack), sizeof(s.slot),
#ifndef M5_FIN1
                            sizeof(s.fin),
#endif
                            0};
    const u64* sp = reinterpret_cast<const u64*>(&s);
    u64* dp = reinterpret_cast<u64*>(&d);
    size_t o = 0;
    for (int r = 0; r < (int)(sizeof(skipO) / sizeof(skipO[0])); r++) {
        for (size_t i = o / 8; i < skipO[r] / 8; i++) dp[i] = sp[i];
        o = skipO[r] + skipN[r];
    }
}
template <class C> HD inline size_t persistentBytes() {
    typedef Work<C> W;
    return sizeof(W) - sizeof(W::WA) - sizeof(W::stack) - sizeof(W::slot)
#ifndef M5_KEEPIX
           - sizeof(W::keep)
#endif
#ifndef M5_FIN1
           - sizeof(W::fin)
#endif
        ;
}
#ifdef M5_KEEPIX
#define M5KEEP(w, k) ((w).slot[(w).keepP[k]])
#else
#define M5KEEP(w, k) ((w).keep[k])
#endif
// a transcript copy (kind as above); with M5_COPYSTATS counted, its bytes and cycles summed
#if defined(M5_COPYSTATS) && defined(__CUDA_ARCH__)
#define M5CPY(w, kind, dst, src) do { long long t0_ = clock64(); (dst).copyFrom(src); (w).cpClk[kind] += clock64() - t0_; \
                                      (w).cpN[kind]++; (w).cpB[kind] += (src).copyBytes(); } while (0)
#define M5STAT(w, k) ((w).stN[k]++)
#elif defined(M5_COPYSTATS)
#define M5CPY(w, kind, dst, src) do { (dst).copyFrom(src); (w).cpN[kind]++; (w).cpB[kind] += (src).copyBytes(); } while (0)
#define M5STAT(w, k) ((w).stN[k]++)
#else
#define M5CPY(w, kind, dst, src) (dst).copyFrom(src)
#define M5STAT(w, k) do { } while (0)
#endif
#ifdef __CUDA_ARCH__
#define M5CLK(w, k, t) do { long long t1_ = clock64(); (w).clk[k] += t1_ - (t); (t) = t1_; } while (0)
#define M5T(t) long long t = clock64()
#define M5ADD(w, k, t) do { (w).clk[k] += clock64() - (t); } while (0)
#else
#define M5CLK(w, k, t) do { } while (0)
#define M5T(t) do { } while (0)
#define M5ADD(w, k, t) do { } while (0)
#endif

// ---------------------------------------------------------------------------------------------------------------------
// winBin map
// ---------------------------------------------------------------------------------------------------------------------
template <class C> HD inline u64 wbGet(const Work<C>& w, uint str, uint bin) {
    { unsigned b = (unsigned)bin & 1023; if (!((w.wbBits[str & 1][b >> 6] >> (b & 63)) & 1)) return uintWinBinMax; }
    u64 key = (w.wbEpoch << 41) | (str << 40) | bin;
    unsigned h = (unsigned)((((str << 40) | bin) * 0x9E3779B97F4A7C15ULL) >> 40) & (C::NWB - 1);
    for (int k = 0; k < C::NWB; k++) {
        unsigned i = (h + k) & (C::NWB - 1);
        if ((w.wbKey[i] >> 41) != w.wbEpoch) return uintWinBinMax;
        if (w.wbKey[i] == key) return w.wbVal[i];
    }
    return uintWinBinMax;
}
template <class C> HD inline bool wbSet(Work<C>& w, uint str, uint bin, uint val) {
    { unsigned b = (unsigned)bin & 1023; w.wbBits[str & 1][b >> 6] |= 1ULL << (b & 63); }
    u64 key = (w.wbEpoch << 41) | (str << 40) | bin;
    unsigned h = (unsigned)((((str << 40) | bin) * 0x9E3779B97F4A7C15ULL) >> 40) & (C::NWB - 1);
    for (int k = 0; k < C::NWB; k++) {
        unsigned i = (h + k) & (C::NWB - 1);
        if (w.wbKey[i] == key) { w.wbVal[i] = (uint16_t)val; return true; }
        if ((w.wbKey[i] >> 41) != w.wbEpoch) {
            if (w.wbN * 4 >= C::NWB * 3) { w.ov = OV_WINBIN; return false; }
            w.wbKey[i] = key; w.wbVal[i] = (uint16_t)val; w.wbN++; return true;
        }
    }
    w.ov = OV_WINBIN; return false;
}

// ---------------------------------------------------------------------------------------------------------------------
// SequenceFuns.cpp qualitySplit
// ---------------------------------------------------------------------------------------------------------------------
template <class C> HD inline bool qualitySplit(const char* r, uint L, uint maxNsplit, uint minLsplit, Work<C>& w) {
    uint iR = 0, iS = 0, iR1, LgoodMin = 0, iFrag = 0;
    while ((iR < L) & (iS < maxNsplit)) {
        while (iR < L && r[iR] > 3) {
            if ((u64)r[iR] == MARK_FRAG_SPACER_BASE) iFrag++;
            iR++;
        }
        if (iR == L) break;
        iR1 = iR;
        while (iR < L && r[iR] <= 3) iR++;
        if ((iR - iR1) > LgoodMin) LgoodMin = iR - iR1;
        if ((iR - iR1) < minLsplit) continue;
        if (iS >= (uint)C::NSPLIT) { w.ov = OV_SPLIT; return false; }
        w.splitR[0][iS] = iR1; w.splitR[1][iS] = iR - iR1; w.splitR[2][iS] = iFrag;
        iS++;
    }
    if (iS == 0) w.splitR[1][0] = LgoodMin;
    w.Nsplit = iS;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// SuffixArrayFuns.cpp: compareSeqToGenome (byte genome), findMultRange, maxMappableLength
// ---------------------------------------------------------------------------------------------------------------------
HD inline u64 medianUint2(u64 a, u64 b) { return a / 2 + b / 2 + (a % 2 + b % 2) / 2; }

#ifdef M5_G2
// E7 M5_G2: compareSeqToGenome on the 2-bit text, 32 bases per step (device). Read1 arrays that reach the seed search are
// {Read1[0], Read1[1], Read1[2], packed forward, packed reversed, (char*)Lread} (M5_READ1); a null [3] means the byte path.
// Exactness: the first position where the read base and the genome byte differ is found per 32-base window; a window
// holding a genome byte > 3 (N, padding, spacer; NB block bits) is compared byte by byte as before, so every mismatch
// against a byte > 3 is decided by the byte code, and in the 2-bit windows every genome byte is 0..3 (the read bases in a
// piece are 0..3: qualitySplit), where (sc > gc) is the byte comparison of the same values.
constexpr long long G2OFF = 1024;   // position p is word (p + G2OFF) / 32 of G2; the arrays cover -G2OFF .. nGenome + G2OFF
#define M5_READ1(name, rb, Lr) const char* name[6] = {(rb).r[0], (rb).r[1], (rb).r[2], (const char*)(rb).p2[0], (const char*)(rb).p2[1], \
                                                     (const char*)(uintptr_t)(Lr)}
#define M5_READ1_BYTES(name, rb) const char* name[6] = {(rb).r[0], (rb).r[1], (rb).r[2], nullptr, nullptr, nullptr}
#ifdef __CUDA_ARCH__
__device__ inline u64 g2Fwd(const u64* G2, i64 p) {        // genome bases p .. p+31, base p+j in bits 2j
    const u64 gp = (u64)(p + G2OFF), w = gp >> 5; const unsigned sh = (unsigned)(gp & 31) * 2;
    u64 v = G2[w] >> sh; if (sh) v |= G2[w + 1] << (64 - sh);
    return v;
}
__device__ inline bool g2HasN(const uint32_t* NB, i64 p) { // some genome byte in p .. p+31 is > 3
    const u64 gp = (u64)(p + G2OFF), b0 = gp >> 6, b1 = (gp + 31) >> 6;
    return ((NB[b0 >> 5] >> (b0 & 31)) | (NB[b1 >> 5] >> (b1 & 31))) & 1;
}
__device__ inline u64 rdFwd(const u64* R, i64 o) {          // packed read units o .. o+31
    const u64 w = (u64)o >> 5; const unsigned sh = (unsigned)(o & 31) * 2;
    u64 v = R[w] >> sh; if (sh) v |= R[w + 1] << (64 - sh);
    return v;
}
__device__ inline u64 rev2(u64 x) {                        // reverse the order of the 32 2-bit units
    x = __brevll(x);
    return ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
}
__device__ inline u64 compareSeqToGenome2(const GenomeView& g_, const char* const* s2, u64 S, u64 N, u64 L, u64 SAstr, bool dirR, bool dirG,
                                          bool& compRes) {
    // comparison index i (0 ..): read Read1[c][S + L + i] (dirR) or Read1[c][S - L - i], c = dirR != dirG (complement);
    // genome G[gq + i] (dirG) or G[gq - i]
    const u64* RA = (const u64*)s2[dirR ? 3 : 4];
    const i64 Lr = (i64)(uintptr_t)s2[5];
    const bool comp = dirR != dirG;
    const i64 r0 = dirR ? (i64)(S + L) : Lr - 1 - ((i64)S - (i64)L);   // Read1[0][S-L-i] = reversed[Lr-1-S+L+i]
    const i64 gq = dirG ? (i64)(SAstr + L) : (i64)(g_.nGenome - 1 - SAstr - L);
    const u64 len = N - L;
    for (u64 i = 0; i < len; i += 32) {
        const i64 gp0 = dirG ? gq + (i64)i : gq - (i64)i - 31;   // lowest genome position of this window
        if (g2HasN(g_.NB, gp0)) {
            const i64 ds = dirR ? 1 : -1, dg = dirG ? 1 : -1;
            const char* s = s2[comp ? 1 : 0] + S + (dirR ? (i64)L : -(i64)L) + (i64)i * ds;
            const char* g = g_.G + gq + (i64)i * dg;
            const u64 e = mn<u64>(len, i + 32);
            for (u64 ii = i; ii < e; ii++, s += ds, g += dg) {
                const char sc = *s, gc = *g;
                if (sc != gc) { compRes = dirG ? sc > gc : !(sc > gc || gc > 3); return ii + L; }
            }
            continue;
        }
        u64 sw = rdFwd(RA, r0 + (i64)i); if (comp) sw = ~sw;
        u64 gw = g2Fwd(g_.G2, gp0); if (!dirG) gw = rev2(gw);
        u64 d = sw ^ gw; d = (d | (d >> 1)) & 0x5555555555555555ULL;
        if (d) {
            const unsigned j = (unsigned)(__ffsll((long long)d) - 1) >> 1;
            if (i + j >= len) return N;
            const unsigned sc = (unsigned)(sw >> (2 * j)) & 3, gc = (unsigned)(gw >> (2 * j)) & 3;
            compRes = dirG ? sc > gc : sc < gc;
            return L + i + j;
        }
    }
    return N;
}
#endif
#else
#define M5_READ1(name, rb, Lr) const char* name[3] = {(rb).r[0], (rb).r[1], (rb).r[2]}
#define M5_READ1_BYTES(name, rb) const char* name[3] = {(rb).r[0], (rb).r[1], (rb).r[2]}
#endif
HD inline u64 compareSeqToGenome(const GenomeView& g_, const char* const* s2, u64 S, u64 N, u64 L, u64 iSA, bool dirR, bool& compRes) {
    u64 SAstr = packed_get(g_.SA, iSA, g_.saBits);
    bool dirG = (SAstr >> g_.GstrandBit) == 0;
    SAstr &= g_.GstrandMask;
#if defined(M5_G2) && defined(__CUDA_ARCH__)
    if (g_.G2 && s2[3]) return compareSeqToGenome2(g_, s2, S, N, L, SAstr, dirR, dirG, compRes);
#endif
#ifdef M5_CMPU
    // E7: the four direction cases as one loop with strides (the same bytes compared in the same order, the same result):
    // the read runs forward (dirR) or backward through Read1[0] (dirR == dirG) or its complement Read1[1]; the genome runs
    // forward from SAstr + L (dirG) or backward from nGenome - 1 - SAstr - L. compRes as in the case it replaces.
    {
        const i64 ds = dirR ? 1 : -1, dg = dirG ? 1 : -1;
        const char* s = s2[dirR == dirG ? 0 : 1] + S + (dirR ? (i64)L : -(i64)L);
        const char* g = g_.G + (dirG ? SAstr + L : g_.nGenome - 1 - SAstr - L);
        for (u64 ii = L; ii < N; ii++, s += ds, g += dg) {
            const char sc = *s, gc = *g;
            if (sc != gc) { compRes = dirG ? sc > gc : !(sc > gc || gc > 3); return ii; }
        }
        return N;
    }
#endif
    const char* g = g_.G;
    i64 ii;
    if (dirR && dirG) {
        const char* s = s2[0] + S + L; g += SAstr + L;
        for (ii = 0; (u64)ii < N - L; ii++)
            if (s[ii] != g[ii]) { compRes = s[ii] > g[ii]; return ii + L; }
        return N;
    } else if (dirR && !dirG) {
        const char* s = s2[1] + S + L; g += g_.nGenome - 1 - SAstr - L;
        for (ii = 0; (u64)ii < N - L; ii++)
            if (s[ii] != g[-ii]) { compRes = !(s[ii] > g[-ii] || g[-ii] > 3); return ii + L; }
        return N;
    } else if (!dirR && dirG) {
        const char* s = s2[1] + S - L; g += SAstr + L;
        for (ii = 0; (u64)ii < N - L; ii++)
            if (s[-ii] != g[ii]) { compRes = s[-ii] > g[ii]; return ii + L; }
        return N;
    } else {
        const char* s = s2[0] + S - L; g += g_.nGenome - 1 - SAstr - L;
        for (ii = 0; (u64)ii < N - L; ii++)
            if (s[-ii] != g[-ii]) { compRes = !(s[-ii] > g[-ii] || g[-ii] > 3); return ii + L; }
        return N;
    }
}

HD inline u64 findMultRange(const GenomeView& g, u64 i3, u64 L3, u64 i1, u64 L1, u64 i1a, u64 L1a, u64 i1b, u64 L1b,
                            const char* const* s, bool dirR, u64 S) {
    bool compRes;
    if (L1 < L3) { L1b = L1; i1b = i1; i1a = i3; }
    else if (L1a < L1) { L1b = L1a; i1b = i1a; i1a = i1; }
    while ((i1b + 1 < i1a) | (i1b > i1a + 1)) {
        u64 i1c = medianUint2(i1a, i1b);
        u64 L1c = compareSeqToGenome(g, s, S, L3, L1b, i1c, dirR, compRes);
        if (L1c == L3) i1a = i1c;
        else { i1b = i1c; L1b = L1c; }
    }
    return i1a;
}

HD inline u64 maxMappableLength(const GenomeView& g, const char* const* s, u64 S, u64 N, u64 i1, u64 i2, bool dirR, u64& L, u64* ind) {
    bool compRes;
    u64 L1, L2, i3, L3, L1a, L1b, L2a, L2b, i1a, i1b, i2a, i2b;
    L1 = compareSeqToGenome(g, s, S, N, L, i1, dirR, compRes);
    L2 = compareSeqToGenome(g, s, S, N, L, i2, dirR, compRes);
    L = L1 < L2 ? L1 : L2;
    L1a = L1; L1b = L1; i1a = i1; i1b = i1;
    L2a = L2; L2b = L2; i2a = i2; i2b = i2;
    i3 = i1; L3 = L1;
    while (i1 + 1 < i2) {
        i3 = medianUint2(i1, i2);
        L3 = compareSeqToGenome(g, s, S, N, L, i3, dirR, compRes);
        if (L3 == N) break;
        if (compRes) {
            if (L3 > L1) { L1b = L1a; L1a = L1; i1b = i1a; i1a = i1; }
            i1 = i3; L1 = L3;
        } else {
            if (L3 > L2) { L2b = L2a; L2a = L2; i2b = i2a; i2a = i2; }
            i2 = i3; L2 = L3;
        }
        L = L1 < L2 ? L1 : L2;
    }
    if (L3 < N) {
        if (L1 > L2) { i3 = i1; L3 = L1; } else { i3 = i2; L3 = L2; }
    }
    i1 = findMultRange(g, i3, L3, i1, L1, i1a, L1a, i1b, L1b, s, dirR, S);
    i2 = findMultRange(g, i3, L3, i2, L2, i2a, L2a, i2b, L2b, s, dirR, S);
    L = L3;
    ind[0] = i1; ind[1] = i2;
    return i2 - i1 + 1;
}

// ---------------------------------------------------------------------------------------------------------------------
// ReadAlign_storeAligns.cpp (OPTIM_STOREaligns_SIMPLE)
// ---------------------------------------------------------------------------------------------------------------------
template <class C> HD inline bool storeAligns(const Params& P, Work<C>& w, uint iDir, uint Shift, uint Nrep, uint L, const uint ind[2], uint iFrag) {
    if (Nrep > P.seedMultimapNmax) {
        if (Nrep < w.multNmin || w.multNmin == 0) { w.multNmin = Nrep; w.multNminL = L; }
        return true;
    }
    w.nUM[Nrep == 1 ? 0 : 1] += Nrep;
    w.nA += Nrep;
    uint rStart = iDir == 0 ? Shift : Shift + 1 - L;
    i64 iP;
    for (iP = (i64)w.nP - 1; iP >= 0; iP--) {
        if (w.PC[iP][0] <= rStart) {
            if ((w.PC[iP][PC_rStart] == rStart) && w.PC[iP][PC_Length] < L) continue;
            if ((w.PC[iP][PC_rStart] == rStart) && w.PC[iP][PC_Length] == L) return true;
            break;
        }
    }
    iP = iP + 1;
    if (w.nP + 1 > (uint)C::NP) { w.ov = OV_PC; return false; }   // STAR exits with an error past seedPerReadNmax
    for (i64 ii = (i64)w.nP - 1; ii >= iP; ii--)
        for (int jj = 0; jj < PC_SIZE; jj++) w.PC[ii + 1][jj] = w.PC[ii][jj];
    w.nP++;
    w.PC[iP][PC_rStart] = rStart; w.PC[iP][PC_Length] = L; w.PC[iP][PC_Dir] = iDir; w.PC[iP][PC_Nrep] = Nrep;
    w.PC[iP][PC_SAstart] = ind[0]; w.PC[iP][PC_SAend] = ind[1]; w.PC[iP][PC_iFrag] = iFrag;
    if (L < w.storedLmin) L = w.storedLmin;
    if (Nrep == 1) {
        if (L > w.uniqLmax) { w.uniqLmax = L; w.uniqLmaxInd = w.nP - 1; }
    } else {
        if (Nrep < w.multNmin || w.multNmin == 0) { w.multNmin = Nrep; w.multNminL = L; }
        if (L > w.multLmax) { w.multLmax = L; w.multLmaxN = Nrep; }
        if (Nrep > w.multNmax) { w.multNmax = Nrep; w.multNmaxL = L; }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// ReadAlign_maxMappableLength2strands.cpp
// ---------------------------------------------------------------------------------------------------------------------
template <class C> HD inline uint maxMappableLength2strands(const Params& P, const GenomeView& g, Work<C>& w, const char* const* Read1,
                                                         uint pieceStartIn, uint pieceLengthIn, uint iDir, uint& maxLbest, uint iFrag) {
    uint Nrep = 0, indStartEnd[2] = {0, 0}, maxL = 0;
    constexpr int SD = 4;   // gSAsparseD <= 4 (checked by the driver)
    uint NrepAll[SD], indAll[SD][2], maxLall[SD];
    maxLbest = 0;
    bool dirR = iDir == 0;
    uint nD = mn<uint>(pieceLengthIn, P.gSAsparseD);
    for (uint iDist = 0; iDist < nD; iDist++) {
        uint pieceStart;
        uint pieceLength = pieceLengthIn - iDist;
        uint Lmax = mn<uint>(P.gSAindexNbases, pieceLength);
        uint ind1 = 0;
        if (dirR) {
            pieceStart = pieceStartIn + iDist;
            for (uint ii = 0; ii < Lmax; ii++) { ind1 <<= 2LLU; ind1 += ((uint)Read1[0][pieceStart + ii]); }
        } else {
            pieceStart = pieceStartIn - iDist;
            for (uint ii = 0; ii < Lmax; ii++) { ind1 <<= 2LLU; ind1 += (3 - ((uint)Read1[0][pieceStart - ii])); }
        }
        uint Lind = Lmax, iSA1 = 0, iSA2;
        while (Lind > 0) {
            iSA1 = packed_get(g.SAi, g.saiStart[Lind - 1] + ind1, g.saiBits);
            if ((iSA1 & g.SAiMarkAbsentMaskC) == 0) break;
            --Lind; ind1 = ind1 >> 2;
        }
        bool iSA2good = true;
        if (g.saiStart[Lind - 1] + ind1 + 1 < g.saiStart[Lind]) {
            iSA2 = packed_get(g.SAi, g.saiStart[Lind - 1] + ind1 + 1, g.saiBits);
            if ((iSA2 & g.SAiMarkAbsentMaskC) == 0) iSA2 = (iSA2 & g.SAiMarkNmask) - 1;
            else { iSA2 = g.nSA - 1; iSA2good = false; }
        } else { iSA2 = g.nSA - 1; iSA2good = false; }
        bool iSA1noN = (iSA1 & g.SAiMarkNmaskC) == 0;
        if (Lind < P.gSAindexNbases && iSA1noN && iSA2good) {
            indStartEnd[0] = iSA1; indStartEnd[1] = iSA2; Nrep = indStartEnd[1] - indStartEnd[0] + 1; maxL = Lind;
        } else if (iSA1 == iSA2 && iSA1noN && iSA2good) {
            indStartEnd[0] = indStartEnd[1] = iSA1; Nrep = 1;
            bool comparRes;
            maxL = compareSeqToGenome(g, Read1, pieceStart, pieceLength, Lind, iSA1, dirR, comparRes);
        } else {
            maxL = (iSA2good && iSA1noN) ? Lind : 0;
            Nrep = maxMappableLength(g, Read1, pieceStart, pieceLength, iSA1 & g.SAiMarkNmask, iSA2, dirR, maxL, indStartEnd);
        }
        if (maxL + iDist > maxLbest) maxLbest = maxL + iDist;
        NrepAll[iDist] = Nrep; indAll[iDist][0] = indStartEnd[0]; indAll[iDist][1] = indStartEnd[1]; maxLall[iDist] = maxL;
    }
    for (uint iDist = 0; iDist < nD; iDist++) {
        if ((maxLall[iDist] + iDist) == maxLbest)
            if (!storeAligns(P, w, iDir, (dirR ? pieceStartIn + iDist : pieceStartIn - iDist), NrepAll[iDist], maxLall[iDist], indAll[iDist], iFrag)) return 0;
    }
    return Nrep;
}

// ---------------------------------------------------------------------------------------------------------------------
// sjAlignSplit.cpp, binarySearch2.cpp
// ---------------------------------------------------------------------------------------------------------------------
HD inline bool sjAlignSplit(uint a1, uint aLength, const GenomeView& g, uint& a1D, uint& aLengthD, uint& a1A, uint& aLengthA, uint& isj) {
    uint sj1 = (a1 - g.sjGstart) % g.sjdbLength;
    if (sj1 < g.sjdbOverhang && sj1 + aLength > g.sjdbOverhang) {
        isj = (a1 - g.sjGstart) / g.sjdbLength;
        aLengthD = g.sjdbOverhang - sj1;
        aLengthA = aLength - aLengthD;
        a1D = g.sjDstart[isj] + sj1;
        a1A = g.sjAstart[isj];
        return true;
    }
    return false;
}

HD inline int binarySearch2(uint x, uint y, const uint* X, const uint* Y, int N) {
    if (N == 0 || x > X[N - 1] || x < X[0]) return -1;
    int i1 = 0, i2 = N - 1, i3 = N / 2;
    while (i2 > i1 + 1) {
        i3 = (i1 + i2) / 2;
        if (X[i3] > x) i2 = i3; else i1 = i3;
    }
    if (x == X[i1]) i3 = i1;
    else if (x == X[i2]) i3 = i2;
    else return -1;
    for (int jj = i3; jj >= 0; jj--) {
        if (x != X[jj]) break;
        else if (y == Y[jj]) return jj;
    }
    for (int jj = i3; jj < N; jj++) {
        if (x != X[jj]) return -1;
        else if (y == Y[jj]) return jj;
    }
    return -2;
}

// ---------------------------------------------------------------------------------------------------------------------
// ReadAlign_createExtendWindowsWithAlign.cpp
// ---------------------------------------------------------------------------------------------------------------------
constexpr int EXIT_TOO_MANY_WINDOWS = 101;
template <class C> HD inline int createExtendWindowsWithAlign(const Params& P, const GenomeView& g, Work<C>& w, uint a1, uint aStr) {
    uint aBin = (a1 >> P.winBinNbits);
    uint iBinLeft = aBin, iBinRight = aBin;
    uint iBin = (uint)-1, iWin = (uint)-1, iWinRight = (uint)-1;
    if (wbGet(w, aStr, aBin) == uintWinBinMax) {
        bool flagMergeLeft = false;
        if (aBin > 0) {
            for (iBin = aBin - 1; iBin >= (aBin > P.winAnchorDistNbins ? aBin - P.winAnchorDistNbins : 0); --iBin) {
                if (wbGet(w, aStr, iBin) < uintWinBinMax) { flagMergeLeft = true; break; }
                if (iBin == 0) break;
            }
            flagMergeLeft = flagMergeLeft && (g.chrBin[iBin >> P.winBinChrNbits] == g.chrBin[aBin >> P.winBinChrNbits]);
            if (flagMergeLeft) {
                iWin = wbGet(w, aStr, iBin);
                iBinLeft = w.WC[iWin][WC_gStart];
                for (uint ii = iBin + 1; ii <= aBin; ii++) if (!wbSet(w, aStr, ii, iWin)) return -1;
            }
        }
        bool flagMergeRight = false;
        if (aBin + 1 < P.winBinN) {
            for (iBin = aBin + 1; iBin < mn<uint>(aBin + P.winAnchorDistNbins + 1, P.winBinN); ++iBin) {
                if (wbGet(w, aStr, iBin) < uintWinBinMax) { flagMergeRight = true; break; }
            }
            flagMergeRight = flagMergeRight && (g.chrBin[iBin >> P.winBinChrNbits] == g.chrBin[aBin >> P.winBinChrNbits]);
            if (flagMergeRight) {
                while (wbGet(w, aStr, iBin) == wbGet(w, aStr, iBin + 1)) ++iBin;
                iBinRight = iBin;
                iWinRight = wbGet(w, aStr, iBin);
                if (!flagMergeLeft) iWin = wbGet(w, aStr, iBin);
                for (uint ii = aBin; ii <= iBin; ii++) if (!wbSet(w, aStr, ii, iWin)) return -1;
            }
        }
        if (!flagMergeLeft && !flagMergeRight) {
            if (w.nW >= (uint)C::NW) { w.ov = OV_WIN; return -1; }
            iWin = w.nW;
            if (!wbSet(w, aStr, aBin, iWin)) return -1;
            w.WC[iWin][WC_Chr] = g.chrBin[aBin >> P.winBinChrNbits];
            w.WC[iWin][WC_Str] = aStr;
            w.WC[iWin][WC_gEnd] = w.WC[iWin][WC_gStart] = aBin;
            ++w.nW;
            if (w.nW >= P.alignWindowsPerReadNmax) {
                w.nW = P.alignWindowsPerReadNmax - 1;
                return EXIT_TOO_MANY_WINDOWS;
            }
        } else {
            w.WC[iWin][WC_gStart] = iBinLeft;
            w.WC[iWin][WC_gEnd] = iBinRight;
            if (flagMergeLeft && flagMergeRight) {
                w.WC[iWinRight][WC_gStart] = 1;
                w.WC[iWinRight][WC_gEnd] = 0;
            }
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// ReadAlign_assignAlignToWindow.cpp
// ---------------------------------------------------------------------------------------------------------------------
template <class C> HD inline bool assignAlignToWindow(const Params& P, Work<C>& w, uint a1, uint aLength, uint aStr, uint aNrep,
                                                   uint aFrag, uint aRstart, bool aAnchor, uint sjA) {
    uint iW = wbGet(w, aStr, a1 >> P.winBinNbits);
    if (iW == uintWinBinMax || (!aAnchor && aLength < w.WALrec[iW])) return true;
    {
        uint iA;
        for (iA = 0; iA < w.nWA[iW]; iA++) {
            const uint* A = w.WA[iW][iA];
            if (aFrag == A[WA_iFrag] && A[WA_sjA] == sjA && a1 + A[WA_rStart] == A[WA_gStart] + aRstart
                && ((aRstart >= A[WA_rStart] && aRstart < A[WA_rStart] + A[WA_Length])
                    || (aRstart + aLength >= A[WA_rStart] && aRstart + aLength < A[WA_rStart] + A[WA_Length]))) break;
        }
        if (iA < w.nWA[iW]) {
            if (aLength > w.WA[iW][iA][WA_Length]) {
                uint iA0;
                for (iA0 = 0; iA0 < w.nWA[iW]; iA0++)
                    if (iA0 != iA && aRstart < w.WA[iW][iA0][WA_rStart]) break;
                if (iA0 > iA) --iA0;
                if (iA0 < iA) {
                    for (uint iA1 = iA; iA1 > iA0; iA1--) for (int ii = 0; ii < WA_SIZE; ii++) w.WA[iW][iA1][ii] = w.WA[iW][iA1 - 1][ii];
                } else if (iA0 > iA) {
                    for (uint iA1 = iA; iA1 < iA0; iA1++) for (int ii = 0; ii < WA_SIZE; ii++) w.WA[iW][iA1][ii] = w.WA[iW][iA1 + 1][ii];
                }
                uint* A = w.WA[iW][iA0];
                A[WA_rStart] = aRstart; A[WA_Length] = aLength; A[WA_gStart] = a1; A[WA_Nrep] = aNrep;
                A[WA_Anchor] = (uint)(int)aAnchor; A[WA_iFrag] = aFrag; A[WA_sjA] = sjA;
            }
            return true;
        }
    }
    if (w.nWA[iW] == P.seedPerWindowNmax) {
        w.WALrec[iW] = w.Lread + 1;
        for (uint iA = 0; iA < w.nWA[iW]; iA++)
            if (w.WA[iW][iA][WA_Anchor] != 1) w.WALrec[iW] = mn(w.WALrec[iW], w.WA[iW][iA][WA_Length]);
        if (w.WALrec[iW] == w.Lread + 1) {
            w.mapMarker = MARKER_TOO_MANY_ANCHORS_PER_WINDOW;
            w.nW = 0;
            return true;
        }
        if (!aAnchor && aLength < w.WALrec[iW]) return true;
        uint iA1 = 0;
        for (uint iA = 0; iA < w.nWA[iW]; iA++) {
            if (w.WA[iW][iA][WA_Anchor] == 1 || w.WA[iW][iA][WA_Length] > w.WALrec[iW]) {
                for (int ii = 0; ii < WA_SIZE; ii++) w.WA[iW][iA1][ii] = w.WA[iW][iA][ii];
                iA1++;
            }
        }
        w.nWA[iW] = iA1;
    }
    if (aAnchor || aLength > w.WALrec[iW]) {
        if (w.nWA[iW] >= (uint)C::NWA) { w.ov = OV_WA; return false; }   // (STAR: BUG exit at seedPerWindowNmax, unreachable)
        uint iA;
        for (iA = 0; iA < w.nWA[iW]; iA++) if (aRstart < w.WA[iW][iA][WA_rStart]) break;
        for (uint iA1 = w.nWA[iW]; iA1 > iA; iA1--) for (int ii = 0; ii < WA_SIZE; ii++) w.WA[iW][iA1][ii] = w.WA[iW][iA1 - 1][ii];
        uint* A = w.WA[iW][iA];
        A[WA_rStart] = aRstart; A[WA_Length] = aLength; A[WA_gStart] = a1; A[WA_Nrep] = aNrep;
        A[WA_Anchor] = (uint)(int)aAnchor; A[WA_iFrag] = aFrag; A[WA_sjA] = sjA;
        w.nWA[iW]++;
        M5CN(if (w.nWA[iW] > w.cnMaxWA) w.cnMaxWA = (uint)w.nWA[iW]);
        if (aAnchor && (uint)w.WlastAnchor[iW] < iA) w.WlastAnchor[iW] = (i64)iA;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// extendAlign.cpp
// ---------------------------------------------------------------------------------------------------------------------
HD inline bool extendAlign(const char* R, const char* G, uint rStart, uint gStart, int dR, int dG, uint L, uint Lprev, uint nMMprev,
                           uint nMMmax, double pMMmax, bool extendToEnd, Ext* trA) {
    int iS, iG;
    int Score = 0, nMatch = 0, nMM = 0;
    trA->maxScore = 0;
    R = R + rStart;
    G = G + gStart;
    if (extendToEnd) {
        int iExt;
        for (iExt = 0; iExt < (int)L; iExt++) {
            iS = dR * iExt; iG = dG * iExt;
            if ((gStart + iG) == (uint)(-1) || G[iG] == 5) {
                trA->extendL = 0; trA->maxScore = -999999999; trA->nMatch = 0; trA->nMM = nMMmax + 1;
                return true;
            }
            if ((u64)R[iS] == MARK_FRAG_SPACER_BASE) break;
            if (R[iS] > 3 || G[iG] > 3) continue;
            if (G[iG] == R[iS]) { nMatch++; Score += scoreMatch; }
            else { nMM++; Score -= scoreMatch; }
        }
        if (iExt > 0) {
            trA->extendL = iExt; trA->maxScore = Score; trA->nMatch = nMatch; trA->nMM = nMM;
            return true;
        }
        return false;
    }
    for (int i = 0; i < (int)L; i++) {
        iS = dR * i; iG = dG * i;
        if ((gStart + iG) == (uint)(-1) || G[iG] == 5 || (u64)R[iS] == MARK_FRAG_SPACER_BASE) break;
        if (R[iS] > 3 || G[iG] > 3) continue;
        if (G[iG] == R[iS]) {
            nMatch++;
            Score += scoreMatch;
            if (Score > trA->maxScore) {
                if (nMM + nMMprev <= mn(pMMmax * double(Lprev + i + 1), double(nMMmax))) {
                    trA->extendL = i + 1; trA->maxScore = Score; trA->nMatch = nMatch; trA->nMM = nMM;
                }
            }
        } else {
            if (nMM + nMMprev >= mn(pMMmax * double(Lprev + L), double(nMMmax))) break;
            nMM++;
            Score -= scoreMatch;
        }
    }
    return trA->extendL > 0;
}
// Transcript::add of an extension result (reset() Transcript: only maxScore, nMatch, nMM are non-zero)
template <class T> HD inline void addExt(T& t, const Ext& e) { t.maxScore += e.maxScore; t.nMatch += e.nMatch; t.nMM += e.nMM; }

// ---------------------------------------------------------------------------------------------------------------------
// stitchAlignToTranscript.cpp
// ---------------------------------------------------------------------------------------------------------------------
template <class T, int NEX> HD inline int stitchAlignToTranscript(uint rAend, uint gAend, uint rBstart, uint gBstart, uint L, uint iFragB, uint sjAB,
                                                               const Params& P, const char* R, const GenomeView& gv, T* trA,
                                                               const uint outFilterMismatchNmaxTotal, bool& ovEx) {
    if (trA->nExons >= 20) return -1000010;               // MAX_N_EXONS
    if (trA->nExons >= (uint)NEX) { ovEx = true; return -1000010; }
    const char* G = gv.G;
    int Score = 0;
    if (sjAB != ((uint)-1) && trA->exons[trA->nExons - 1][EX_sjA] == sjAB
        && trA->exons[trA->nExons - 1][EX_iFrag] == iFragB && rBstart == rAend + 1 && gAend + 1 < gBstart) {
        if (gv.sjdbMotif[sjAB] == 0 && (L <= gv.sjdbShiftRight[sjAB] || trA->exons[trA->nExons - 1][EX_L] <= gv.sjdbShiftLeft[sjAB])) {
            return -1000006;
        }
        trA->exons[trA->nExons][EX_L] = L;
        trA->exons[trA->nExons][EX_R] = rBstart;
        trA->exons[trA->nExons][EX_G] = gBstart;
        trA->canonSJ[trA->nExons - 1] = gv.sjdbMotif[sjAB];
        trA->shiftSJ[trA->nExons - 1][0] = gv.sjdbShiftLeft[sjAB];
        trA->shiftSJ[trA->nExons - 1][1] = gv.sjdbShiftRight[sjAB];
        trA->sjAnnot[trA->nExons - 1] = 1;
        trA->sjStr[trA->nExons - 1] = gv.sjdbStrand[sjAB];
        trA->nExons++;
        trA->nMatch += L;
        for (uint ii = rBstart; ii < rBstart + L; ii++) Score += scoreMatch;
        Score += P.sjdbScore;
    } else {
        trA->sjAnnot[trA->nExons - 1] = 0;
        trA->sjStr[trA->nExons - 1] = 0;
        if (trA->exons[trA->nExons - 1][EX_iFrag] == iFragB) {
            uint gBend = gBstart + L - 1;
            uint rBend = rBstart + L - 1;
            if (rBend <= rAend) return -1000001;
            if (gBend <= gAend && trA->exons[trA->nExons - 1][EX_iFrag] == iFragB) return -1000002;
            if (rBstart <= rAend) {
                gBstart += rAend - rBstart + 1;
                rBstart = rAend + 1;
                L = rBend - rBstart + 1;
            }
            for (uint ii = rBstart; ii <= rBend; ii++) Score += scoreMatch;
            int gGap = gBstart - gAend - 1;
            int rGap = rBstart - rAend - 1;
            uint nMatch = L;
            uint nMM = 0;
            uint Del = 0, Ins = 0;
            uint nIns = 0, nDel = 0;
            int jR = 0;
            int jCan = 999;
            uint gBstart1 = gBstart - rGap - 1;
            if (gGap == 0 && rGap == 0) {
            } else if (gGap > 0 && rGap > 0 && rGap == gGap) {
                for (int ii = 1; ii <= rGap; ii++) {
                    if (G[gAend + ii] < 4 && R[rAend + ii] < 4) {
                        if (R[rAend + ii] == G[gAend + ii]) { Score += scoreMatch; nMatch++; }
                        else { Score -= scoreMatch; nMM++; }
                    }
                }
            } else if (gGap > rGap) {
                nDel = 1;
                Del = gGap - rGap;
                if (Del > P.alignIntronMax && P.alignIntronMax > 0) return -1000003;
                int Score1 = 0;
                int jR1 = 1;
                do {
                    jR1--;
                    if (R[rAend + jR1] != G[gBstart1 + jR1] && G[gBstart1 + jR1] < 4 && R[rAend + jR1] == G[gAend + jR1]) Score1 -= scoreMatch;
                } while (Score1 + P.scoreStitchSJshift >= 0 && int(trA->exons[trA->nExons - 1][EX_L]) + jR1 > 1);
                int maxScore2 = -999999;
                Score1 = 0;
                int jPen = 0;
                do {
                    if (R[rAend + jR1] == G[gAend + jR1] && R[rAend + jR1] != G[gBstart1 + jR1]) Score1 += scoreMatch;
                    if (R[rAend + jR1] != G[gAend + jR1] && R[rAend + jR1] == G[gBstart1 + jR1]) Score1 -= scoreMatch;
                    int jCan1 = -1;
                    int jPen1 = 0;
                    int Score2 = Score1;
                    if (Del >= P.alignIntronMin) {
                        if (G[gAend + jR1 + 1] == 2 && G[gAend + jR1 + 2] == 3 && G[gBstart1 + jR1 - 1] == 0 && G[gBstart1 + jR1] == 2) { jCan1 = 1; }
                        else if (G[gAend + jR1 + 1] == 1 && G[gAend + jR1 + 2] == 3 && G[gBstart1 + jR1 - 1] == 0 && G[gBstart1 + jR1] == 1) { jCan1 = 2; }
                        else if (G[gAend + jR1 + 1] == 2 && G[gAend + jR1 + 2] == 1 && G[gBstart1 + jR1 - 1] == 0 && G[gBstart1 + jR1] == 2) { jCan1 = 3; jPen1 = P.scoreGapGCAG; }
                        else if (G[gAend + jR1 + 1] == 1 && G[gAend + jR1 + 2] == 3 && G[gBstart1 + jR1 - 1] == 2 && G[gBstart1 + jR1] == 1) { jCan1 = 4; jPen1 = P.scoreGapGCAG; }
                        else if (G[gAend + jR1 + 1] == 0 && G[gAend + jR1 + 2] == 3 && G[gBstart1 + jR1 - 1] == 0 && G[gBstart1 + jR1] == 1) { jCan1 = 5; jPen1 = P.scoreGapATAC; }
                        else if (G[gAend + jR1 + 1] == 2 && G[gAend + jR1 + 2] == 3 && G[gBstart1 + jR1 - 1] == 0 && G[gBstart1 + jR1] == 3) { jCan1 = 6; jPen1 = P.scoreGapATAC; }
                        else { jCan1 = 0; jPen1 = P.scoreGapNoncan; }
                        Score2 += jPen1;
                    }
                    if (maxScore2 < Score2) { maxScore2 = Score2; jR = jR1; jCan = jCan1; jPen = jPen1; }
                    jR1++;
                } while (jR1 < int(rBend) - int(rAend));
                uint jjL = 0, jjR = 0;
                while (gAend + jR >= jjL && G[gAend - jjL + jR] == G[gBstart1 - jjL + jR] && G[gAend - jjL + jR] < 4 && jjL <= MAX_SJ_REPEAT_SEARCH) jjL++;
                while (gAend + jjR + jR + 1 < gv.nGenome && G[gAend + jjR + jR + 1] == G[gBstart1 + jjR + jR + 1] && G[gAend + jjR + jR + 1] < 4 && jjR <= MAX_SJ_REPEAT_SEARCH) jjR++;
                if (jCan <= 0) {
                    jR -= jjL;
                    if (int(trA->exons[trA->nExons - 1][EX_L]) + jR < 1) return -1000005;
                    jjR += jjL;
                    jjL = 0;
                }
                for (int ii = mn(1, jR + 1); ii <= mx(rGap, jR); ii++) {
                    uint g1 = (ii <= jR) ? (gAend + ii) : (gBstart1 + ii);
                    if (G[g1] < 4 && R[rAend + ii] < 4) {
                        if (R[rAend + ii] == G[g1]) {
                            if (ii >= 1 && ii <= rGap) { Score += scoreMatch; nMatch++; }
                        } else {
                            Score -= scoreMatch;
                            nMM++;
                            if (ii < 1 || ii > rGap) { Score -= scoreMatch; nMatch--; }
                        }
                    }
                }
                if (gv.sjdbN > 0) {
                    uint jS = gAend + jR + 1, jE = gBstart1 + jR;
                    int sjdbInd = binarySearch2(jS, jE, gv.sjdbStart, gv.sjdbEnd, (int)gv.sjdbN);
                    if (sjdbInd < 0) {
                        if (Del >= P.alignIntronMin) {
                            Score += P.scoreGap + jPen;
                        } else {
                            Score += Del * P.scoreDelBase + P.scoreDelOpen;
                            jCan = -1;
                            trA->sjAnnot[trA->nExons - 1] = 0;
                        }
                    } else {
                        jCan = gv.sjdbMotif[sjdbInd];
                        if (gv.sjdbMotif[sjdbInd] == 0) {
                            if (L <= gv.sjdbShiftLeft[sjdbInd] || trA->exons[trA->nExons - 1][EX_L] <= gv.sjdbShiftLeft[sjdbInd]) return -1000006;
                            jR += (int)gv.sjdbShiftLeft[sjdbInd];
                            if (rAend + jR >= rBend) return -1000006;
                            jjL = gv.sjdbShiftLeft[sjdbInd];
                            jjR = gv.sjdbShiftRight[sjdbInd];
                        }
                        trA->sjAnnot[trA->nExons - 1] = 1;
                        trA->sjStr[trA->nExons - 1] = gv.sjdbStrand[sjdbInd];
                        Score += P.sjdbScore;
                    }
                } else {
                    if (Del >= P.alignIntronMin) {
                        Score += P.scoreGap + jPen;
                    } else {
                        Score += Del * P.scoreDelBase + P.scoreDelOpen;
                        jCan = -1;
                        trA->sjAnnot[trA->nExons - 1] = 0;
                    }
                }
                trA->shiftSJ[trA->nExons - 1][0] = jjL;
                trA->shiftSJ[trA->nExons - 1][1] = jjR;
                trA->canonSJ[trA->nExons - 1] = jCan;
                if (trA->sjAnnot[trA->nExons - 1] == 0) {
                    if (jCan > 0) trA->sjStr[trA->nExons - 1] = 2 - jCan % 2;
                    else trA->sjStr[trA->nExons - 1] = 0;
                }
            } else if (rGap > gGap) {
                Ins = rGap - gGap;
                nIns = 1;
                if (gGap == 0) {
                    jR = 0;
                } else if (gGap < 0) {
                    jR = 0;
                    for (int ii = 0; ii < -gGap; ii++) Score -= scoreMatch;
                } else {
                    int Score1 = 0; int maxScore1 = 0;
                    for (int jR1 = 1; jR1 <= gGap; jR1++) {
                        if (G[gAend + jR1] < 4) {
                            Score1 += (R[rAend + jR1] == G[gAend + jR1]) ? scoreMatch : -scoreMatch;
                            Score1 += (R[rAend + Ins + jR1] == G[gAend + jR1]) ? -scoreMatch : +scoreMatch;
                        }
                        if (Score1 > maxScore1 || (Score1 == maxScore1 && P.flushRight)) { maxScore1 = Score1; jR = jR1; }
                    }
                    for (int ii = 1; ii <= gGap; ii++) {
                        uint r1 = rAend + ii + (ii <= jR ? 0 : Ins);
                        if (G[gAend + ii] < 4 && R[r1] < 4) {
                            if (R[r1] == G[gAend + ii]) { Score += scoreMatch; nMatch++; }
                            else { Score -= scoreMatch; nMM++; }
                        }
                    }
                }
                if (P.flushRight) {
                    for (; jR < (int)rBend - (int)rAend - (int)Ins; jR++)
                        if (R[rAend + jR + 1] != G[gAend + jR + 1] || G[gAend + jR + 1] == 4) break;
                    if (jR == (int)rBend - (int)rAend - (int)Ins) return -1000009;
                }
                Score += Ins * P.scoreInsBase + P.scoreInsOpen;
                jCan = -2;
            }
            if ((trA->nMM + nMM) <= outFilterMismatchNmaxTotal
                && (jCan < 0 || (jCan < 7 && nMM <= (uint)P.alignSJstitchMismatchNmax[(jCan + 1) / 2]))) {
                trA->nMM += nMM;
                trA->nMatch += nMatch;
                if (Del >= P.alignIntronMin) { trA->nGap += nDel; trA->lGap += Del; }
                else { trA->nDel += nDel; trA->lDel += Del; }
                if (Del == 0 && Ins == 0) {
                    trA->exons[trA->nExons - 1][EX_L] += rBend - rAend;
                } else if (Del > 0) {
                    trA->exons[trA->nExons - 1][EX_L] += jR;
                    trA->exons[trA->nExons][EX_L] = rBend - rAend - jR;
                    trA->exons[trA->nExons][EX_R] = rAend + jR + 1;
                    trA->exons[trA->nExons][EX_G] = gBstart1 + jR + 1;
                    trA->nExons++;
                } else if (Ins > 0) {
                    trA->nIns += nIns;
                    trA->lIns += Ins;
                    trA->exons[trA->nExons - 1][EX_L] += jR;
                    trA->exons[trA->nExons][EX_L] = rBend - rAend - jR - Ins;
                    trA->exons[trA->nExons][EX_R] = rAend + jR + Ins + 1;
                    trA->exons[trA->nExons][EX_G] = gAend + 1 + jR;
                    trA->canonSJ[trA->nExons - 1] = -2;
                    trA->sjAnnot[trA->nExons - 1] = 0;
                    trA->nExons++;
                }
            } else {
                return -1000007;
            }
        } else if (gBstart + trA->exons[0][EX_R] + P.alignEndsProtrudeNbasesMax >= trA->exons[0][EX_G] || trA->exons[0][EX_G] < trA->exons[0][EX_R]) {
            if (P.alignMatesGapMax > 0 && gBstart > trA->exons[trA->nExons - 1][EX_G] + trA->exons[trA->nExons - 1][EX_L] + P.alignMatesGapMax)
                return -1000004;
            for (uint ii = rBstart; ii < rBstart + L; ii++) Score += scoreMatch;
            Ext trExtend;
            trExtend.extendL = 0; trExtend.maxScore = 0; trExtend.nMatch = 0; trExtend.nMM = 0;
            if (extendAlign(R, G, rAend + 1, gAend + 1, 1, 1, DEF_readSeqLengthMax, trA->nMatch, trA->nMM, outFilterMismatchNmaxTotal,
                            P.outFilterMismatchNoverLmax, P.alignEndsTypeExt[trA->exons[trA->nExons - 1][EX_iFrag]][1], &trExtend)) {
                addExt(*trA, trExtend);
                Score += trExtend.maxScore;
                trA->exons[trA->nExons - 1][EX_L] += trExtend.extendL;
            }
            trA->exons[trA->nExons][EX_R] = rBstart;
            trA->exons[trA->nExons][EX_G] = gBstart;
            trA->exons[trA->nExons][EX_L] = L;
            trA->nMatch += L;
            trExtend.extendL = 0; trExtend.maxScore = 0; trExtend.nMatch = 0; trExtend.nMM = 0;
            uint extlen = P.alignEndsTypeExt[iFragB][1] ? DEF_readSeqLengthMax : gBstart - trA->exons[0][EX_G] + trA->exons[0][EX_R];
            if (extendAlign(R, G, rBstart - 1, gBstart - 1, -1, -1, extlen, trA->nMatch, trA->nMM, outFilterMismatchNmaxTotal,
                            P.outFilterMismatchNoverLmax, P.alignEndsTypeExt[iFragB][1], &trExtend)) {
                addExt(*trA, trExtend);
                Score += trExtend.maxScore;
                trA->exons[trA->nExons][EX_R] -= trExtend.extendL;
                trA->exons[trA->nExons][EX_G] -= trExtend.extendL;
                trA->exons[trA->nExons][EX_L] += trExtend.extendL;
            }
            trA->canonSJ[trA->nExons - 1] = -3;
            trA->sjAnnot[trA->nExons - 1] = 0;
            trA->nExons++;
        } else {
            return -1000008;
        }
    }
    trA->exons[trA->nExons - 1][EX_iFrag] = iFragB;
    trA->exons[trA->nExons - 1][EX_sjA] = sjAB;
    return Score;
}

// ---------------------------------------------------------------------------------------------------------------------
// blocksOverlap.cpp
// ---------------------------------------------------------------------------------------------------------------------
template <class T> HD inline uint blocksOverlap(const T& t1, const T& t2) {
    uint i1 = 0, i2 = 0, nOverlap = 0;
    while (i1 < t1.nExons && i2 < t2.nExons) {
        uint rs1 = t1.exons[i1][EX_R], rs2 = t2.exons[i2][EX_R];
        uint re1 = t1.exons[i1][EX_R] + t1.exons[i1][EX_L], re2 = t2.exons[i2][EX_R] + t2.exons[i2][EX_L];
        uint gs1 = t1.exons[i1][EX_G], gs2 = t2.exons[i2][EX_G];
        if (rs1 >= re2) i2++;
        else if (rs2 >= re1) i1++;
        else if (gs1 - rs1 != gs2 - rs2) {
            if (re1 >= re2) i2++;
            if (re2 >= re1) i1++;
        } else {
            nOverlap += mn(re1, re2) - mx(rs1, rs2);
            if (re1 >= re2) i2++;
            if (re2 >= re1) i1++;
        }
    }
    return nOverlap;
}

// genomic length score: int(ceil(log2((double)x) * scoreGenomicLengthLog2scale - 0.5)), from host-made breakpoints
HD inline int genomicLengthScore(const Params& P, uint x) {
    int v = P.glVal0;
    for (int k = 0; k < P.glN; k++) { if (x >= P.glBreak[k]) v = P.glVal[k]; else break; }
    return v;
}

// ---------------------------------------------------------------------------------------------------------------------
// stitchWindowAligns.cpp, finalization (the iA >= nA branch). trA is taken by value in STAR: `t` is a copy here.
// Records into the window's slots (wTr = slotP over slot[]).
// ---------------------------------------------------------------------------------------------------------------------
template <class C> HD inline bool finalizeTranscript(const Params& P, const GenomeView& gv, Work<C>& w, int Score, uint tR2, uint tG2,
                                                  typename Work<C>::T& trA, const char* R, const uint (*WA)[WA_SIZE]) {
    const uint Lread = w.Lread;
    Ext trAstep1;
    int vOrder[2];
    if (trA.roStr == 0) { vOrder[0] = 0; vOrder[1] = 1; } else { vOrder[0] = 1; vOrder[1] = 0; }   // EXTEND_ORDER 1
    for (int iOrd = 0; iOrd < 2; iOrd++) {
        switch (vOrder[iOrd]) {
        case 0:
            if (trA.rStart > 0) {
                trAstep1.extendL = 0; trAstep1.maxScore = 0; trAstep1.nMatch = 0; trAstep1.nMM = 0;
                uint imate = trA.exons[0][EX_iFrag];
                if (extendAlign(R, gv.G, trA.rStart - 1, trA.gStart - 1, -1, -1, trA.rStart, tR2 - trA.rStart + 1,
                                trA.nMM, w.outFilterMismatchNmaxTotal, P.outFilterMismatchNoverLmax,
                                P.alignEndsTypeExt[imate][(int)(trA.Str != imate)], &trAstep1)) {
                    addExt(trA, trAstep1);
                    Score += trAstep1.maxScore;
                    trA.exons[0][EX_R] = trA.rStart = trA.rStart - trAstep1.extendL;
                    trA.exons[0][EX_G] = trA.gStart = trA.gStart - trAstep1.extendL;
                    trA.exons[0][EX_L] += trAstep1.extendL;
                }
            }
            break;
        case 1:
            if (tR2 < Lread) {
                trAstep1.extendL = 0; trAstep1.maxScore = 0; trAstep1.nMatch = 0; trAstep1.nMM = 0;
                uint imate = trA.exons[trA.nExons - 1][EX_iFrag];
                if (extendAlign(R, gv.G, tR2 + 1, tG2 + 1, +1, +1, Lread - tR2 - 1, tR2 - trA.rStart + 1,
                                trA.nMM, w.outFilterMismatchNmaxTotal, P.outFilterMismatchNoverLmax,
                                P.alignEndsTypeExt[imate][(int)(imate == trA.Str)], &trAstep1)) {
                    addExt(trA, trAstep1);
                    Score += trAstep1.maxScore;
                    tR2 += trAstep1.extendL;
                    tG2 += trAstep1.extendL;
                    trA.exons[trA.nExons - 1][EX_L] += trAstep1.extendL;
                }
            }
        }
    }
    if (!P.alignSoftClipAtReferenceEnds &&
        ((trA.exons[trA.nExons - 1][EX_G] + Lread - trA.exons[trA.nExons - 1][EX_R]) > (gv.chrStart[trA.Chr] + gv.chrLength[trA.Chr]) ||
         trA.exons[0][EX_G] < (gv.chrStart[trA.Chr] + trA.exons[0][EX_R]))) {
        return true;
    }
    trA.rLength = 0;
    for (uint isj = 0; isj < trA.nExons; isj++) trA.rLength += trA.exons[isj][EX_L];
    trA.gLength = tG2 + 1 - trA.gStart;
    for (uint isj = 0; isj < trA.nExons - 1; isj++) {
        if (trA.canonSJ[isj] >= 0) {
            if (trA.sjAnnot[isj] == 1) {
                if ((trA.exons[isj][EX_L] < P.alignSJDBoverhangMin && (isj == 0 || trA.canonSJ[isj - 1] == -3 || (trA.sjAnnot[isj - 1] == 0 && trA.canonSJ[isj - 1] >= 0)))
                    || (trA.exons[isj + 1][EX_L] < P.alignSJDBoverhangMin && (isj == trA.nExons - 2 || trA.canonSJ[isj + 1] == -3 || (trA.sjAnnot[isj + 1] == 0 && trA.canonSJ[isj + 1] >= 0)))) return true;
            } else {
                if (trA.exons[isj][EX_L] < P.alignSJoverhangMin + trA.shiftSJ[isj][0]
                    || trA.exons[isj + 1][EX_L] < P.alignSJoverhangMin + trA.shiftSJ[isj][1]) return true;
            }
        }
    }
    if (trA.nExons > 1 && trA.sjAnnot[trA.nExons - 2] == 1 && trA.exons[trA.nExons - 1][EX_L] < P.alignSJDBoverhangMin) return true;
    uint sjN = 0;
    trA.intronMotifs[0] = 0; trA.intronMotifs[1] = 0; trA.intronMotifs[2] = 0;
    trA.sjYes = false;
    for (uint iex = 0; iex < trA.nExons - 1; iex++) {
        if (trA.canonSJ[iex] >= 0) { sjN++; trA.intronMotifs[trA.sjStr[iex]]++; trA.sjYes = true; }
    }
    if (trA.intronMotifs[1] > 0 && trA.intronMotifs[2] == 0) trA.sjMotifStrand = 1;
    else if (trA.intronMotifs[1] == 0 && trA.intronMotifs[2] > 0) trA.sjMotifStrand = 2;
    else trA.sjMotifStrand = 0;
    if (trA.intronMotifs[1] > 0 && trA.intronMotifs[2] > 0 && P.removeInconsistentStrands) return true;
    if (sjN > 0 && trA.sjMotifStrand == 0 && P.outSAMstrandFieldType == 1) return true;
    if (P.intronMotifsFilter == 1) {
        for (uint iex = 0; iex < trA.nExons - 1; iex++) if (trA.canonSJ[iex] == 0) return true;
    } else if (P.intronMotifsFilter == 2) {
        for (uint iex = 0; iex < trA.nExons - 1; iex++) if (trA.canonSJ[iex] == 0 && trA.sjAnnot[iex] == 0) return true;
    }
    {
        uint nsj = 0, exl = 0;
        for (uint iex = 0; iex < trA.nExons; iex++) {
            exl += trA.exons[iex][EX_L];
            if (iex == trA.nExons - 1 || trA.canonSJ[iex] == -3) {
                if (nsj > 0 && (exl < P.alignSplicedMateMapLmin || exl < (uint)(P.alignSplicedMateMapLminOverLmate * Lread))) return true;   // readLength[iFrag] = Lread (single end)
                exl = 0; nsj = 0;
            } else if (trA.canonSJ[iex] >= 0) {
                nsj++;
            }
        }
    }
    if (trA.exons[0][EX_iFrag] != trA.exons[trA.nExons - 1][EX_iFrag]) return true;   // mates: not supported (single end)
    if (P.scoreGenomicLengthLog2scale != 0) {
        Score += genomicLengthScore(P, trA.exons[trA.nExons - 1][EX_G] + trA.exons[trA.nExons - 1][EX_L] - trA.exons[0][EX_G]);
        Score = mx(0, Score);
    }
    trA.roStart = (trA.roStr == 0) ? trA.rStart : Lread - trA.rStart - trA.rLength;
    trA.maxScore = Score;
    if (trA.exons[0][EX_iFrag] == trA.exons[trA.nExons - 1][EX_iFrag]) {
        trA.iFrag = (int)trA.exons[0][EX_iFrag];
        w.maxScoreMate[trA.iFrag] = mx(w.maxScoreMate[trA.iFrag], Score);
    } else {
        trA.iFrag = -1;
    }
    trA.maxScore = Score;   // + variationAdjust = 0 (no variation)
    typedef typename Work<C>::T T;
    M5T(trec);
    struct RecClk { Work<C>& w; long long t; HD ~RecClk() { M5ADD(w, 7, t); } };
#ifdef __CUDA_ARCH__
    RecClk rc_{w, trec};
#endif
    if (Score + P.outFilterMultimapScoreRange >= w.slot[w.slotP[0]].maxScore
        || (trA.iFrag >= 0 && Score + P.outFilterMultimapScoreRange >= w.maxScoreMate[trA.iFrag])) {
        uint iTr = 0;
        M5CN(w.cnRec++);
        trA.mappedLength = 0;
        for (uint iex = 0; iex < trA.nExons; iex++) trA.mappedLength += trA.exons[iex][EX_L];
        while (iTr < w.nWinTr) {
            const T& old = w.slot[w.slotP[iTr]];
            uint nOverlap = blocksOverlap(trA, old);
            uint uNew = trA.mappedLength - nOverlap;
            uint uOld = old.mappedLength - nOverlap;
            if (uNew == 0 && Score < old.maxScore) {
                break;
            } else if (uOld == 0) {
                int pTr = w.slotP[iTr];
                for (uint ii = iTr + 1; ii < w.nWinTr; ii++) w.slotP[ii - 1] = w.slotP[ii];
                w.nWinTr--;
                w.slotP[w.nWinTr] = pTr;
            } else if (uOld > 0 && (uNew > 0 || Score >= old.maxScore)) {
                iTr++;
            }
            // (STAR loops forever if none applies; that needs uOld>0, uNew==0, Score>=old: excluded by the first branch? no:
            //  uNew==0 && Score>=old && uOld>0 -> third branch. uNew>0 or uOld==0 cover the rest.)
        }
        if (iTr == w.nWinTr) {
            for (iTr = 0; iTr < w.nWinTr; iTr++) {
                const T& o = w.slot[w.slotP[iTr]];
                if (Score > o.maxScore || (Score == o.maxScore && trA.gLength < o.gLength)) break;
            }
            int pTr = w.slotP[w.nWinTr];
            for (int ii = (int)w.nWinTr; ii > int(iTr); ii--) w.slotP[ii] = w.slotP[ii - 1];
#ifdef M5_FIN1
            // trA is slot[slotSpare] (stitchWindowAligns): position iTr takes it; pTr, whose content STAR overwrites here
            // and which leaves the live positions, becomes the next spare
            w.slotP[iTr] = w.slotSpare; w.slotSpare = pTr;
#else
            w.slotP[iTr] = pTr;
            M5CPY(w, 3, w.slot[w.slotP[iTr]], trA);
#endif
            if (w.nWinTr < P.alignTranscriptsPerWindowNmax) {
                if (w.nWinTr + 1 > (uint)C::NWTR) { w.ov = OV_WTR; return false; }
                w.nWinTr++;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// stitchWindowAligns.cpp, the recursion over aligns (include, then exclude), on an explicit stack.
// Frame d holds STAR's arguments for the call with iA = d: trA = stack[d], Score, tR2, tG2.
// M5_REFX: frame d's trA is stack[src] with src <= d: an include child owns a fresh copy (src = d), an exclude child shares
// its parent's (src = parent's src). stack[k] is written only by the include step of frame k-1, so while frame d is
// active (its descendants write stack[d+1..] only) stack[src] <= stack[d] stays unchanged: every frame reads exactly the
// transcript the copy would have held.
// ---------------------------------------------------------------------------------------------------------------------
template <class C> HD inline bool stitchWindowAligns(const Params& P, const GenomeView& gv, Work<C>& w, uint nA, const uint (*WA)[WA_SIZE], const char* R) {
    typedef typename Work<C>::T T;
#ifdef M5_REFX
    struct Frame { int Score; uint tR2, tG2; uint8_t phase; uint8_t src; };
#define M5_SETSRC(N, v) ((N).src = (uint8_t)(v))
#else
    struct Frame { int Score; uint tR2, tG2; uint8_t phase; };
#define M5_SETSRC(N, v) do { } while (0)
#endif
    Frame fr[C::NWA + 1];
    int d = 0;
    fr[0].Score = 0; fr[0].tR2 = 0; fr[0].tG2 = 0; fr[0].phase = 0; M5_SETSRC(fr[0], 0);
    while (d >= 0) {
#ifdef M5_CENSUS
        ++w.nSteps;   // counted with or without a budget
        if (w.stepBudget && w.nSteps > w.stepBudget) { w.ov = OV_STEPS; return false; }
#else
        if (w.stepBudget && ++w.nSteps > w.stepBudget) { w.ov = OV_STEPS; return false; }
#endif
        M5STAT(w, 0);
        Frame& F = fr[d];
        uint iA = (uint)d;
#ifdef M5_REFX
        T& trA = w.stack[F.src];
#else
        T& trA = w.stack[d];
#endif
        if (F.phase == 0) {
            if (iA >= nA && F.tR2 == 0) { d--; continue; }
            if (iA >= nA) {
                M5CN(w.cnFinal++);
                M5T(tf);
#ifdef M5_FIN1
                T& cand = w.slot[w.slotSpare];
                M5CPY(w, 2, cand, trA);
                bool okf = finalizeTranscript(P, gv, w, F.Score, F.tR2, F.tG2, cand, R, WA);
#else
                M5CPY(w, 2, w.fin, trA);
                bool okf = finalizeTranscript(P, gv, w, F.Score, F.tR2, F.tG2, w.fin, R, WA);
#endif
                M5ADD(w, 5, tf);
                if (!okf) return false;
                d--; continue;
            }
            // include this align
            int dScore = 0;
            T& trAi = w.stack[d + 1];
            { M5T(tcp); M5CPY(w, 0, trAi, trA); M5ADD(w, 6, tcp); }
            if (trA.nExons > 0) {
                bool ovEx = false;
                M5CN(w.cnStitch++; if (trA.nExons > w.cnMaxEx) w.cnMaxEx = (uint)trA.nExons);
                M5T(tsa);
                dScore = stitchAlignToTranscript<T, C::NEX>(F.tR2, F.tG2, WA[iA][WA_rStart], WA[iA][WA_gStart], WA[iA][WA_Length], WA[iA][WA_iFrag],
                                                            WA[iA][WA_sjA], P, R, gv, &trAi, w.outFilterMismatchNmaxTotal, ovEx);
                M5ADD(w, 4, tsa);
                if (ovEx) { w.ov = OV_EXONS; return false; }
            } else {
                trAi.exons[0][EX_R] = trAi.rStart = WA[iA][WA_rStart];
                trAi.exons[0][EX_G] = trAi.gStart = WA[iA][WA_gStart];
                trAi.exons[0][EX_L] = WA[iA][WA_Length];
                trAi.exons[0][EX_iFrag] = WA[iA][WA_iFrag];
                trAi.exons[0][EX_sjA] = WA[iA][WA_sjA];
                trAi.nExons = 1;
                for (uint ii = 0; ii < WA[iA][WA_Length]; ii++) dScore += scoreMatch;
                trAi.nMatch = WA[iA][WA_Length];
            }
            F.phase = 1;
            if (dScore > -1000000) {
                if (WA[iA][WA_Nrep] == 1) trAi.nUnique++;
                if (WA[iA][WA_Anchor] > 0) trAi.nAnchor++;
                Frame& N = fr[d + 1];
                N.Score = F.Score + dScore;
                N.tR2 = WA[iA][WA_rStart] + WA[iA][WA_Length] - 1;
                N.tG2 = WA[iA][WA_gStart] + WA[iA][WA_Length] - 1;
                N.phase = 0; M5_SETSRC(N, d + 1);
                d++;
                continue;
            }
        }
        if (F.phase == 1) {
            F.phase = 2;
            if (WA[iA][WA_Anchor] != 2 || trA.nAnchor > 0) {
#ifdef M5_REFX
                Frame& N = fr[d + 1];
                N.Score = F.Score; N.tR2 = F.tR2; N.tG2 = F.tG2; N.phase = 0; M5_SETSRC(N, F.src);
#else
                T& next = w.stack[d + 1];
                { M5T(tcp); M5CPY(w, 1, next, trA); M5ADD(w, 6, tcp); }
                Frame& N = fr[d + 1];
                N.Score = F.Score; N.tR2 = F.tR2; N.tG2 = F.tG2; N.phase = 0;
#endif
                d++;
                continue;
            }
        }
        d--;
    }
    return true;
#undef M5_SETSRC
}

// ---------------------------------------------------------------------------------------------------------------------
// Encoding of the sequence cache (ReadAlign_seqCache.cpp putV, forEachScalar, forEachRow)
// ---------------------------------------------------------------------------------------------------------------------
struct Blob {
    uint8_t* p; int n, cap; bool full;
    HD void putV(i64 v) {
        u64 u = ((u64)v << 1) ^ (u64)(v >> 63);
        while (u >= 0x80) { put((uint8_t)(u | 0x80)); u >>= 7; }
        put((uint8_t)u);
    }
#ifdef M5_BLOBW
    // bytes n-nb .. n-1 wait in acc (byte j at bits 8j); every 8th byte stores the word at p + n - 8 (8-aligned: p is,
    // and n - nb is a multiple of 8). flush() stores a partial word: bytes past n are 0 and stay below cap (a multiple of 8).
    u64 acc = 0; int nb = 0;
    HD void put(uint8_t b) {
        if (n < cap) {
            acc |= (u64)b << (8 * nb); n++;
            if (++nb == 8) { st8(p + n - 8, acc); acc = 0; nb = 0; }
        } else {
            full = true;
        }
    }
    HD void flush() { if (nb) { st8(p + n - nb, acc); acc = 0; nb = 0; } }
    static HD void st8(uint8_t* q, u64 x) {
#ifdef __CUDA_ARCH__
        *reinterpret_cast<u64*>(q) = x;
#else
        memcpy(q, &x, 8);
#endif
    }
#else
    HD void put(uint8_t b) { if (n < cap) p[n++] = b; else full = true; }
    HD void flush() {}
#endif
};
template <class T> HD inline void encodeTr(Blob& b, const T& t) {
    b.putV((i64)t.nExons);
    i64 v[] = {(i64)t.rStart, (i64)t.roStart, (i64)t.rLength, (i64)t.gStart, (i64)t.gLength, (i64)t.cStart,
               (i64)t.Chr, (i64)t.Str, (i64)t.roStr,
               (i64)t.nMatch, (i64)t.nMM, (i64)t.mappedLength, (i64)t.extendL,
               (i64)t.nGap, (i64)t.lGap, (i64)t.nDel, (i64)t.nIns, (i64)t.lDel, (i64)t.lIns, (i64)t.nUnique, (i64)t.nAnchor,
               (i64)t.intronMotifs[0], (i64)t.intronMotifs[1], (i64)t.intronMotifs[2],
               (i64)t.maxScore, (i64)t.iFrag, (i64)t.sjMotifStrand, (i64)(t.sjYes ? 1 : 0), (i64)(t.primaryFlag ? 1 : 0)};
#ifdef M5_ENC1
    // the same values in the same order, through one putV call site: each exon row's values gathered first
#pragma unroll 1
    for (int k = 0; k < (int)(sizeof v / sizeof v[0]); k++) b.putV(v[k]);
#pragma unroll 1
    for (uint iex = 0; iex < t.nExons; iex++) {
        i64 e[EX_SIZE + 5];
        int ne = 0;
        for (int k = 0; k < EX_SIZE; k++) e[ne++] = (i64)t.exons[iex][k];
        if (iex + 1 < t.nExons) {
            e[ne++] = t.canonSJ[iex]; e[ne++] = t.sjAnnot[iex]; e[ne++] = t.sjStr[iex];
            if (t.canonSJ[iex] >= 0) { e[ne++] = (i64)t.shiftSJ[iex][0]; e[ne++] = (i64)t.shiftSJ[iex][1]; }
        }
#pragma unroll 1
        for (int k = 0; k < ne; k++) b.putV(e[k]);
    }
#else
    for (int k = 0; k < (int)(sizeof v / sizeof v[0]); k++) b.putV(v[k]);
    for (uint iex = 0; iex < t.nExons; iex++) {
        for (int k = 0; k < EX_SIZE; k++) b.putV((i64)t.exons[iex][k]);
        if (iex + 1 < t.nExons) {
            b.putV(t.canonSJ[iex]); b.putV(t.sjAnnot[iex]); b.putV(t.sjStr[iex]);
            if (t.canonSJ[iex] >= 0) { b.putV((i64)t.shiftSJ[iex][0]); b.putV((i64)t.shiftSJ[iex][1]); }
        }
    }
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// One read: ReadAlign::oneRead's mapping part (mapOneRead, multMapSelect, mappedFilter) for a single-end read.
// R0: Read1[0] (bases 0..3, 4 = N), Lread. Writes the blob; returns the status; sets unmapType and mapMarker.
// ---------------------------------------------------------------------------------------------------------------------
// phase 1: read set-up, seeds, windows from anchors and their flanks. Sets w.stitched and w.nEntries (SA entries that the
// assignment loop visits). The assignment (assignAll, or entryHit + assignHits) and mapFinish follow.
template <class C> HD inline Status mapPhase1(const Params& P, const GenomeView& gv, Work<C>& w, ReadBuf<C::LREAD>& rb,
                                           const char* R0, uint Lread) {
    w.ov = OV_NONE;
    long long tc = 0;
#ifdef __CUDA_ARCH__
    tc = clock64();
#endif
    w.stitched = false; w.nEntries = 0;
#ifdef M5_CENSUS
    // census: no statistic of the previous read in this Work survives an early overflow (each is set again before the
    // mapping reads it, so these resets do not change results)
    w.nSteps = 0; w.Nsplit = 0; w.nP = 0; w.nW = 0; w.nWall = 0; w.nKeep = 0; w.trNtotal = 0; w.nTr = 0; w.nWinTr = 0; w.wbN = 0;
    w.mapMarker = 0; w.unmapType = -2;
    w.cnStitch = w.cnFinal = w.cnRec = w.cnHits = 0; w.cnMaxEx = w.cnMaxWA = w.cnKeepMax = w.cnWinTrMax = w.cnWinStitched = 0;
    w.cn[0] = w.cn[1] = w.cn[2] = w.cn[3] = 0; w.cnFinish = false;
#endif
    if (Lread > (uint)C::LREAD || Lread == 0) { w.ov = OV_LREAD; return Status::Overflow; }
    // oneRead: Read1[1] = complement, Read1[2] = reverse complement
    for (uint ii = 0; ii < Lread; ii++) { char c = R0[ii]; rb.r[0][ii] = c; rb.r[1][ii] = c < 4 ? 3 - c : c; }
    for (uint ii = 0; ii < Lread; ii++) rb.r[2][Lread - ii - 1] = rb.r[1][ii];
    M5_READ1_BYTES(Read1, rb);
    w.Lread = Lread;
    w.outFilterMismatchNmaxTotal = mn<uint>(P.outFilterMismatchNmax, (uint)(P.outFilterMismatchNoverReadLmax * (Lread + 0)));
    // mapOneRead
    if (Lread > 0) { if (!qualitySplit(Read1[0], Lread, P.maxNsplit, P.seedSplitMin, w)) return Status::Overflow; }
    else w.Nsplit = 0;
    // resetN
    w.mapMarker = 0; w.nA = 0; w.nP = 0; w.nW = 0; w.nTr = 0; w.nUM[0] = w.nUM[1] = 0;
    w.storedLmin = 0; w.uniqLmax = 0; w.uniqLmaxInd = 0; w.multLmax = 0; w.multLmaxN = 0; w.multNminL = 0; w.multNmin = 0; w.multNmax = 0; w.multNmaxL = 0;
    w.maxScoreMate[0] = 0;
    w.trInitRLength = 0;
    w.trBestKeep = -1; w.trBestScore = 0; w.trBestGLength = 0;     // trBest = trInit (maxScore 0, gLength 0)
    w.nKeep = 0; w.nWtr = 0; w.trNtotal = 0;
    uint seedSearchStartLmax = mn<uint>(P.seedSearchStartLmax, (uint)(P.seedSearchStartLmaxOverLread * (Lread - 1)));
    for (uint ip = 0; ip < w.Nsplit; ip++) {
        uint Nstart = P.seedSearchStartLmax > 0 && seedSearchStartLmax < w.splitR[1][ip] ? w.splitR[1][ip] / seedSearchStartLmax + 1 : 1;
        uint Lstart = w.splitR[1][ip] / Nstart;
        bool flagDirMap = true;
        for (uint iDir = 0; iDir < 2; iDir++) {
            uint Lmapped, L;
            for (uint istart = 0; istart < Nstart; istart++) {
                if (flagDirMap || istart > 0) {
                    Lmapped = 0;
                    while (istart * Lstart + Lmapped + P.seedMapMin < w.splitR[1][ip]) {
                        uint Shift = iDir == 0 ? (w.splitR[0][ip] + istart * Lstart + Lmapped)
                                               : (w.splitR[0][ip] + w.splitR[1][ip] - istart * Lstart - 1 - Lmapped);
                        uint seedLength = w.splitR[1][ip] - Lmapped - istart * Lstart;
                        maxMappableLength2strands(P, gv, w, Read1, Shift, seedLength, iDir, L, w.splitR[2][ip]);
                        if (w.ov) return Status::Overflow;
                        if (iDir == 0 && istart == 0 && Lmapped == 0 && Shift + L == w.splitR[1][ip]) flagDirMap = false;
                        Lmapped += L;
                    }
                }
                if (P.seedSearchLmax > 0) {
                    uint Shift = iDir == 0 ? (w.splitR[0][ip] + istart * Lstart) : (w.splitR[0][ip] + w.splitR[1][ip] - istart * Lstart - 1);
                    uint seedLength = mn<uint>(P.seedSearchLmax, iDir == 0 ? (w.splitR[0][ip] + w.splitR[1][ip] - Shift) : (Shift + 1));
                    maxMappableLength2strands(P, gv, w, Read1, Shift, seedLength, iDir, L, w.splitR[2][ip]);
                    if (w.ov) return Status::Overflow;
                }
            }
        }
    }
    M5CLK(w, 0, tc);
    bool stitched = false;
    if (Lread < P.outFilterMatchNmin) {
        w.mapMarker = MARKER_READ_TOO_SHORT; w.trInitRLength = 0; w.nW = 0;
    } else if (w.Nsplit == 0) {
        w.mapMarker = MARKER_NO_GOOD_PIECES; w.trInitRLength = w.splitR[1][0]; w.nW = 0;
    } else if (w.Nsplit > 0 && w.nA == 0) {
        w.mapMarker = MARKER_ALL_PIECES_EXCEED_seedMultimapNmax; w.trInitRLength = w.multNminL; w.nW = 0;
    } else if (w.Nsplit > 0 && w.nA > 0) {
        stitched = true;
    }
    if (stitched) {
        // ---- stitchPieces: windows from anchors
        // new epoch instead of clearing the map (keys of other epochs read as empty); epoch 0 is never used
        if (w.wbEpoch == 0 || w.wbEpoch >= (1ULL << 22) - 1) { for (int i = 0; i < C::NWB; i++) w.wbKey[i] = 0; w.wbEpoch = 1; }
        else w.wbEpoch++;
        w.wbN = 0;
        for (int i = 0; i < 16; i++) { w.wbBits[0][i] = 0; w.wbBits[1][i] = 0; }
        w.nW = 0;
        for (uint iP = 0; iP < w.nP; iP++) {
            if (w.PC[iP][PC_Nrep] <= P.winAnchorMultimapNmax) {
                uint aDir = w.PC[iP][PC_Dir], aLength = w.PC[iP][PC_Length];
                w.nScan += w.PC[iP][PC_SAend] - w.PC[iP][PC_SAstart] + 1;
                for (uint iSA = w.PC[iP][PC_SAstart]; iSA <= w.PC[iP][PC_SAend]; iSA++) {
                    uint a1 = packed_get(gv.SA, iSA, gv.saBits);
                    uint aStr = a1 >> gv.GstrandBit;
                    a1 &= gv.GstrandMask;
                    if (aDir == 1 && aStr == 0) aStr = 1;
                    else if (aDir == 0 && aStr == 1) a1 = gv.nGenome - (aLength + a1);
                    else if (aDir == 1 && aStr == 1) { aStr = 0; a1 = gv.nGenome - (aLength + a1); }
                    if (a1 >= gv.sjGstart) {
                        uint a1D, aLengthD, a1A, aLengthA, sj1;
                        if (sjAlignSplit(a1, aLength, gv, a1D, aLengthD, a1A, aLengthA, sj1)) {
                            int st = createExtendWindowsWithAlign(P, gv, w, a1D, aStr);
                            if (st < 0) return Status::Overflow;
                            if (st == EXIT_TOO_MANY_WINDOWS) break;
                            st = createExtendWindowsWithAlign(P, gv, w, a1A, aStr);
                            if (st < 0) return Status::Overflow;
                            if (st == EXIT_TOO_MANY_WINDOWS) break;
                        }
                    } else {
                        int st = createExtendWindowsWithAlign(P, gv, w, a1, aStr);
                        if (st < 0) return Status::Overflow;
                        if (st == EXIT_TOO_MANY_WINDOWS) break;
                    }
                }
            }
        }
        // flanks
        for (uint iWin = 0; iWin < w.nW; iWin++) {
            if (w.WC[iWin][WC_gStart] <= w.WC[iWin][WC_gEnd]) {
                uint wb = w.WC[iWin][WC_gStart];
                for (uint ii = 0; ii < P.winFlankNbins && wb > 0 && gv.chrBin[(wb - 1) >> P.winBinChrNbits] == w.WC[iWin][WC_Chr]; ii++) {
                    wb--;
                    if (!wbSet(w, w.WC[iWin][WC_Str], wb, iWin)) return Status::Overflow;
                }
                w.WC[iWin][WC_gStart] = wb;
                wb = w.WC[iWin][WC_gEnd];
                for (uint ii = 0; ii < P.winFlankNbins && wb + 1 < P.winBinN && gv.chrBin[(wb + 1) >> P.winBinChrNbits] == w.WC[iWin][WC_Chr]; ii++) {
                    wb++;
                    if (!wbSet(w, w.WC[iWin][WC_Str], wb, iWin)) return Status::Overflow;
                }
                w.WC[iWin][WC_gEnd] = wb;
            }
            w.nWA[iWin] = 0; w.WALrec[iWin] = 0; w.WlastAnchor[iWin] = -1;
        }
        w.nWall = w.nW;
        w.nEntries = 0;
        for (uint iP = 0; iP < w.nP; iP++) w.nEntries += w.PC[iP][PC_SAend] - w.PC[iP][PC_SAstart] + 1;
        M5CLK(w, 1, tc);
    }
    w.stitched = stitched;
    return Status::Ok;
}

// ---------------------------------------------------------------------------------------------------------------------
// E7 (seeds): phase 1 with flat loops, for convergent warps (docs/gpu-engine/fable/E7-seeds.md). The same statements as
// mapPhase1 in the same order per read, regrouped into three functions so that a kernel can run them as one warp-uniform
// loop per stage:
//   p1Setup   read set-up, qualitySplit, resetN, the seed iterator at the first statement of the nest
//   seedNext  advances the iterator through STAR's seed-loop nest (pieces x iDir x istart x the while loop over Lmapped,
//             then the seedSearchLmax call) to the next maxMappableLength2strands call; false at the end of the nest
//   seedAfter what the nest does with the call's result L (flagDirMap, Lmapped += L)
//   p1Windows the markers after the seed loop, then the windows from the anchors' SA entries as ONE loop over the entries
//             (createExtendWindowsWithAlign at one call site), the flanks, nWall, nEntries
// A kernel calls maxMappableLength2strands (storeAligns inside) once per seedNext; each read's calls are STAR's, in STAR's
// order, so storeAligns sees the same sequence and PC is the same.
// ---------------------------------------------------------------------------------------------------------------------
enum : uint8_t { SI_PIECE, SI_START, SI_WHILE, SI_LMAX, SI_NEXT, SI_DONE };
struct SeedIt {
    uint ssl;                                                    // seedSearchStartLmax of this read
    uint32_t ip, iDir, istart, Lmapped, Nstart, Lstart;
    uint32_t pS, pL, pF;                                         // splitR[0..2][ip] of the current piece
    uint32_t Shift, seedLength;                                  // the pending call
    uint8_t ph; bool flagDirMap;
};
template <class C> HD inline Status p1Setup(const Params& P, Work<C>& w, ReadBuf<C::LREAD>& rb, const char* R0, uint Lread, SeedIt& it) {
    w.ov = OV_NONE;
    w.stitched = false; w.nEntries = 0;
    if (Lread > (uint)C::LREAD || Lread == 0) { w.ov = OV_LREAD; return Status::Overflow; }
    for (uint ii = 0; ii < Lread; ii++) { char c = R0[ii]; rb.r[0][ii] = c; rb.r[1][ii] = c < 4 ? 3 - c : c; }
    for (uint ii = 0; ii < Lread; ii++) rb.r[2][Lread - ii - 1] = rb.r[1][ii];
#ifdef M5_G2
    for (int q = 0; q < C::LREAD / 32 + 4; q++) {   // Read1[0] packed forward and reversed (units past the read: 0)
        u64 f = 0, r = 0;
        for (int j = 0; j < 32; j++) {
            const uint a = (uint)q * 32 + j;
            if (a < Lread) { f |= (u64)(R0[a] & 3) << (2 * j); r |= (u64)(R0[Lread - 1 - a] & 3) << (2 * j); }
        }
        rb.p2[0][q] = f; rb.p2[1][q] = r;
    }
#endif
    w.Lread = Lread;
    w.outFilterMismatchNmaxTotal = mn<uint>(P.outFilterMismatchNmax, (uint)(P.outFilterMismatchNoverReadLmax * (Lread + 0)));
    if (!qualitySplit(rb.r[0], Lread, P.maxNsplit, P.seedSplitMin, w)) return Status::Overflow;   // Lread > 0 here
    w.mapMarker = 0; w.nA = 0; w.nP = 0; w.nW = 0; w.nTr = 0; w.nUM[0] = w.nUM[1] = 0;
    w.storedLmin = 0; w.uniqLmax = 0; w.uniqLmaxInd = 0; w.multLmax = 0; w.multLmaxN = 0; w.multNminL = 0; w.multNmin = 0; w.multNmax = 0; w.multNmaxL = 0;
    w.maxScoreMate[0] = 0;
    w.trInitRLength = 0;
    w.trBestKeep = -1; w.trBestScore = 0; w.trBestGLength = 0;
    w.nKeep = 0; w.nWtr = 0; w.trNtotal = 0;
    it.ssl = mn<uint>(P.seedSearchStartLmax, (uint)(P.seedSearchStartLmaxOverLread * (Lread - 1)));
    it.ip = 0; it.ph = SI_PIECE;
    return Status::Ok;
}
template <class C> HD inline bool seedNext(const Params& P, const Work<C>& w, SeedIt& it) {
    for (;;) {
        switch (it.ph) {
        case SI_PIECE:   // for (ip ...) { Nstart, Lstart; flagDirMap = true; for (iDir = 0 ...) for (istart = 0 ...)
            if (it.ip >= w.Nsplit) { it.ph = SI_DONE; return false; }
            {
                it.pS = (uint32_t)w.splitR[0][it.ip]; it.pL = (uint32_t)w.splitR[1][it.ip]; it.pF = (uint32_t)w.splitR[2][it.ip];
                const uint len = it.pL;
                const uint Nstart = P.seedSearchStartLmax > 0 && it.ssl < len ? len / it.ssl + 1 : 1;
                it.Nstart = (uint32_t)Nstart; it.Lstart = (uint32_t)(len / Nstart);
            }
            it.flagDirMap = true; it.iDir = 0; it.istart = 0; it.ph = SI_START;
            break;
        case SI_START:   // if (flagDirMap || istart > 0) { Lmapped = 0; while ... }
            if (it.flagDirMap || it.istart > 0) { it.Lmapped = 0; it.ph = SI_WHILE; }
            else it.ph = SI_LMAX;
            break;
        case SI_WHILE: {   // while (istart * Lstart + Lmapped + seedMapMin < splitR[1][ip]) { call; Lmapped += L }
            const uint off = (uint)it.istart * it.Lstart + it.Lmapped;
            if (off + P.seedMapMin < (uint)it.pL) {
                it.Shift = (uint32_t)(it.iDir == 0 ? ((uint)it.pS + off) : ((uint)it.pS + it.pL - 1 - off));
                it.seedLength = (uint32_t)((uint)it.pL - off);
                return true;
            }
            it.ph = SI_LMAX;
            break;
        }
        case SI_LMAX:      // if (seedSearchLmax > 0) { call }  (its L is not used)
            it.ph = SI_NEXT;
            if (P.seedSearchLmax > 0) {
                const uint sh = it.iDir == 0 ? ((uint)it.pS + (uint)it.istart * it.Lstart) : ((uint)it.pS + it.pL - (uint)it.istart * it.Lstart - 1);
                it.Shift = (uint32_t)sh;
                it.seedLength = (uint32_t)mn<uint>(P.seedSearchLmax, it.iDir == 0 ? ((uint)it.pS + it.pL - sh) : (sh + 1));
                return true;
            }
            break;
        case SI_NEXT:
            if (++it.istart < it.Nstart) it.ph = SI_START;
            else if (++it.iDir < 2) { it.istart = 0; it.ph = SI_START; }
            else { it.ip++; it.ph = SI_PIECE; }
            break;
        default:
            return false;
        }
    }
}
HD inline void seedAfter(SeedIt& it, uint L) {
    if (it.ph != SI_WHILE) return;   // the seedSearchLmax call
    if (it.iDir == 0 && it.istart == 0 && it.Lmapped == 0 && (uint)it.Shift + L == (uint)it.pL) it.flagDirMap = false;
    it.Lmapped = (uint32_t)((uint)it.Lmapped + L);
}
// the pending call of the iterator (one call site per kernel)
template <class C> HD inline void seedCall(const Params& P, const GenomeView& gv, Work<C>& w, const char* const* Read1, const SeedIt& it, uint& L) {
    maxMappableLength2strands(P, gv, w, Read1, (uint)it.Shift, (uint)it.seedLength, (uint)it.iDir, L, (uint)it.pF);
}
// the windows from one SA entry of an anchor (the body of stitchPieces' iSA loop); returns createExtendWindowsWithAlign's
// last status (< 0 overflow, EXIT_TOO_MANY_WINDOWS: STAR leaves this piece's iSA loop)
template <class C> HD inline int p1Entry(const Params& P, const GenomeView& gv, Work<C>& w, uint iSA, uint aDir, uint aLength) {
    uint a1 = packed_get(gv.SA, iSA, gv.saBits);
    uint aStr = a1 >> gv.GstrandBit;
    a1 &= gv.GstrandMask;
    if (aDir == 1 && aStr == 0) aStr = 1;
    else if (aDir == 0 && aStr == 1) a1 = gv.nGenome - (aLength + a1);
    else if (aDir == 1 && aStr == 1) { aStr = 0; a1 = gv.nGenome - (aLength + a1); }
    uint aX = a1, aY = 0; int na = 1;   // the anchor, or the donor and acceptor parts of an sjdb-insert alignment
    if (a1 >= gv.sjGstart) {
        uint a1D, aLengthD, a1A, aLengthA, sj1;
        if (sjAlignSplit(a1, aLength, gv, a1D, aLengthD, a1A, aLengthA, sj1)) { aX = a1D; aY = a1A; na = 2; }
        else na = 0;
    }
    int st = 0;
#pragma unroll 1
    for (int q = 0; q < na; q++) {
        st = createExtendWindowsWithAlign(P, gv, w, q == 0 ? aX : aY, aStr);
        if (st != 0) break;   // < 0 overflow; EXIT_TOO_MANY_WINDOWS: STAR's break skips the acceptor
    }
    return st;
}
template <class C> HD inline Status p1Windows(const Params& P, const GenomeView& gv, Work<C>& w) {
#ifdef __CUDA_ARCH__
    long long tc = clock64();
#endif
    const uint Lread = w.Lread;
    bool stitched = false;
    if (Lread < P.outFilterMatchNmin) {
        w.mapMarker = MARKER_READ_TOO_SHORT; w.trInitRLength = 0; w.nW = 0;
    } else if (w.Nsplit == 0) {
        w.mapMarker = MARKER_NO_GOOD_PIECES; w.trInitRLength = w.splitR[1][0]; w.nW = 0;
    } else if (w.Nsplit > 0 && w.nA == 0) {
        w.mapMarker = MARKER_ALL_PIECES_EXCEED_seedMultimapNmax; w.trInitRLength = w.multNminL; w.nW = 0;
    } else if (w.Nsplit > 0 && w.nA > 0) {
        stitched = true;
    }
    if (stitched) {
        if (w.wbEpoch == 0 || w.wbEpoch >= (1ULL << 22) - 1) { for (int i = 0; i < C::NWB; i++) w.wbKey[i] = 0; w.wbEpoch = 1; }
        else w.wbEpoch++;
        w.wbN = 0;
        for (int i = 0; i < 16; i++) { w.wbBits[0][i] = 0; w.wbBits[1][i] = 0; }
        w.nW = 0;
        // one loop over every SA entry of every anchor (pieces in order, entries in order): one entry per iteration
        const uint nP = w.nP;
        uint iP = 0, iSA = 0, iSAend = 0, aDir = 0, aLength = 0;
        bool inP = false;
        for (;;) {
            if (!inP) {
                while (iP < nP && w.PC[iP][PC_Nrep] > P.winAnchorMultimapNmax) iP++;
                if (iP >= nP) break;
                iSA = w.PC[iP][PC_SAstart]; iSAend = w.PC[iP][PC_SAend];
                aDir = w.PC[iP][PC_Dir]; aLength = w.PC[iP][PC_Length];
                w.nScan += iSAend - iSA + 1;
                inP = true;
            }
            const int st = p1Entry(P, gv, w, iSA, aDir, aLength);
            if (st < 0) return Status::Overflow;
            if (st == EXIT_TOO_MANY_WINDOWS || iSA == iSAend) { inP = false; iP++; }
            else iSA++;
        }
        // flanks
        for (uint iWin = 0; iWin < w.nW; iWin++) {
            if (w.WC[iWin][WC_gStart] <= w.WC[iWin][WC_gEnd]) {
                uint wb = w.WC[iWin][WC_gStart];
                for (uint ii = 0; ii < P.winFlankNbins && wb > 0 && gv.chrBin[(wb - 1) >> P.winBinChrNbits] == w.WC[iWin][WC_Chr]; ii++) {
                    wb--;
                    if (!wbSet(w, w.WC[iWin][WC_Str], wb, iWin)) return Status::Overflow;
                }
                w.WC[iWin][WC_gStart] = wb;
                wb = w.WC[iWin][WC_gEnd];
                for (uint ii = 0; ii < P.winFlankNbins && wb + 1 < P.winBinN && gv.chrBin[(wb + 1) >> P.winBinChrNbits] == w.WC[iWin][WC_Chr]; ii++) {
                    wb++;
                    if (!wbSet(w, w.WC[iWin][WC_Str], wb, iWin)) return Status::Overflow;
                }
                w.WC[iWin][WC_gEnd] = wb;
            }
            w.nWA[iWin] = 0; w.WALrec[iWin] = 0; w.WlastAnchor[iWin] = -1;
        }
        w.nWall = w.nW;
        w.nEntries = 0;
        for (uint iP2 = 0; iP2 < w.nP; iP2++) w.nEntries += w.PC[iP2][PC_SAend] - w.PC[iP2][PC_SAstart] + 1;
        M5CLK(w, 1, tc);
    }
    w.stitched = stitched;
    return Status::Ok;
}
// phase 1 as one call per read with the flat loops (the E7 counterpart of mapPhase1; same results)
template <class C> HD inline Status mapPhase1Flat(const Params& P, const GenomeView& gv, Work<C>& w, ReadBuf<C::LREAD>& rb,
                                               const char* R0, uint Lread) {
#ifdef __CUDA_ARCH__
    long long tc = clock64();
#endif
    SeedIt it;
    Status s = p1Setup(P, w, rb, R0, Lread, it);
    if (s != Status::Ok) return s;
    M5_READ1(Read1, rb, Lread);
    while (seedNext(P, w, it)) {
        uint L;
        seedCall(P, gv, w, Read1, it, L);
        if (w.ov) return Status::Overflow;
        seedAfter(it, L);
    }
    M5CLK(w, 0, tc);
    return p1Windows(P, gv, w);
}

// the assignment of one SA entry of piece iP (stitchPieces, FASTASSIGN path); false = overflow
template <class C> HD inline bool assignEntry(const Params& P, const GenomeView& gv, Work<C>& w, uint iP, uint iSA) {
    uint aNrep = w.PC[iP][PC_Nrep], aFrag = w.PC[iP][PC_iFrag], aLength = w.PC[iP][PC_Length], aDir = w.PC[iP][PC_Dir];
    bool aAnchor = (aNrep <= P.winAnchorMultimapNmax);
    const uint rS0 = w.PC[iP][PC_rStart], rS1 = w.Lread - (aLength + rS0);
    uint a1 = packed_get(gv.SA, iSA, gv.saBits);
    const uint s = a1 >> gv.GstrandBit;
    a1 &= gv.GstrandMask;
    if (s == 1) a1 = gv.nGenome - (aLength + a1);
    const uint aStr = (aDir == 0) ? s : 1 - s;                 // strOf[s]
    const uint aRstart = ((aDir == 0) == (s == 0)) ? rS0 : rS1;  // rsOf[s]
    if (a1 >= gv.sjGstart) {
        uint a1D, aLengthD, a1A, aLengthA, isj1;
        if (sjAlignSplit(a1, aLength, gv, a1D, aLengthD, a1A, aLengthA, isj1)) {
            if (!assignAlignToWindow(P, w, a1D, aLengthD, aStr, aNrep, aFrag, aRstart, aAnchor, isj1)) return false;
            if (!assignAlignToWindow(P, w, a1A, aLengthA, aStr, aNrep, aFrag, aRstart + aLengthD, aAnchor, isj1)) return false;
        }
    } else if (wbGet(w, aStr, a1 >> P.winBinNbits) != uintWinBinMax) {
        if (!assignAlignToWindow(P, w, a1, aLength, aStr, aNrep, aFrag, aRstart, aAnchor, (uint)-1)) return false;
    }
    return true;
}
// does assignEntry do anything for this entry? Its calls return at once when no window holds the bin, and the
// assignment never changes winBin, so this test made before the assignment gives the answer STAR gets during it.
template <class C> HD inline bool entryHit(const Params& P, const GenomeView& gv, const Work<C>& w, uint iP, uint iSA) {
    uint aLength = w.PC[iP][PC_Length], aDir = w.PC[iP][PC_Dir];
    uint a1 = packed_get(gv.SA, iSA, gv.saBits);
    const uint s = a1 >> gv.GstrandBit;
    a1 &= gv.GstrandMask;
    if (s == 1) a1 = gv.nGenome - (aLength + a1);
    const uint aStr = (aDir == 0) ? s : 1 - s;
    if (a1 >= gv.sjGstart) {
        uint a1D, aLengthD, a1A, aLengthA, isj1;
        if (!sjAlignSplit(a1, aLength, gv, a1D, aLengthD, a1A, aLengthA, isj1)) return false;
        return wbGet(w, aStr, a1D >> P.winBinNbits) != uintWinBinMax || wbGet(w, aStr, a1A >> P.winBinNbits) != uintWinBinMax;
    }
    return wbGet(w, aStr, a1 >> P.winBinNbits) != uintWinBinMax;
}
#define M5_TOO_MANY_ANCHORS(w) ((w).nW == 0 && (w).mapMarker == MARKER_TOO_MANY_ANCHORS_PER_WINDOW)
// every entry in order (host lane, wide GPU lane)
template <class C> HD inline bool assignAll(const Params& P, const GenomeView& gv, Work<C>& w) {
    for (uint iP = 0; iP < w.nP; iP++) {
        const uint iSAend = w.PC[iP][PC_SAend];
        w.nScan += iSAend - w.PC[iP][PC_SAstart] + 1;
        for (uint iSA = w.PC[iP][PC_SAstart]; iSA <= iSAend; iSA++) {
            M5CN(if (entryHit(P, gv, w, iP, iSA)) w.cnHits++);
            if (!assignEntry(P, gv, w, iP, iSA)) return false;
            if (M5_TOO_MANY_ANCHORS(w)) return true;   // nothing observable follows: STAR goes on with nW = 0
        }
    }
    return true;
}
// only the entries marked in bits (bit base + k = the read's k-th entry, pieces in order), in the same order
template <class C> HD inline bool assignHits(const Params& P, const GenomeView& gv, Work<C>& w, const uint32_t* bits, u64 base) {
    u64 off = base;
    for (uint iP = 0; iP < w.nP; iP++) {
        const u64 cnt = w.PC[iP][PC_SAend] - w.PC[iP][PC_SAstart] + 1;
        u64 b = off, e = off + cnt;
        while (b < e) {
            uint32_t word = bits[b >> 5] >> (b & 31);
            u64 left = mn<u64>(32 - (b & 31), e - b);
            if (left < 32) word &= (1u << left) - 1;
            while (word) {
                int t = ctz32(word);
                word &= word - 1;
                uint iSA = w.PC[iP][PC_SAstart] + (b + t - off);
                M5CN(w.cnHits++);
                if (!assignEntry(P, gv, w, iP, iSA)) return false;
                if (M5_TOO_MANY_ANCHORS(w)) return true;
            }
            b += left;
        }
        off = e;
    }
    return true;
}

// the rest of the read: stitching of every window, multMapSelect, mappedFilter, encoding
template <class C> HD inline Status mapFinish(const Params& P, const GenomeView& gv, Work<C>& w, ReadBuf<C::LREAD>& rb, Blob& blob) {
    typedef typename Work<C>::T T;
    const char* Read1[3] = {rb.r[0], rb.r[1], rb.r[2]};
    const uint Lread = w.Lread;
    const bool stitched = w.stitched;
    w.nSteps = 0;
    long long tc = 0;
#ifdef __CUDA_ARCH__
    tc = clock64();
#endif
    if (stitched) {
        // ---- stitch each window, keep its transcripts; trBest
        uint iW1 = 0;
#ifdef M5_CENSUS
        w.cnFinish = true;
        for (uint iW = 0; iW < w.nWall; iW++) { w.cnWinSteps[iW] = 0; w.cnWinCyc[iW] = 0; w.cnWinTr[iW] = 0; }
#endif
#ifdef M5_KEEPIX
        for (int p = 0; p < C::NWTR + 2 + C::NKEEP; p++) w.own[p] = 0;
#endif
        for (uint iW = 0; iW < w.nW; iW++) {
            if (w.nWA[iW] == 0) continue;
            if ((uint)w.WlastAnchor[iW] < w.nWA[iW]) w.WA[iW][w.WlastAnchor[iW]][WA_Anchor] = 2;
            T& trA = w.stack[0];
            trA.reset();
            trA.Chr = 0; trA.Str = 0; trA.roStr = 0; trA.cStart = 0; trA.gLength = 0; trA.nExons = 0;   // trInit
            trA.rLength = 0;
            trA.Chr = w.WC[iW][WC_Chr];
            trA.Str = w.WC[iW][WC_Str];
            trA.roStr = trA.Str;            // revertStrand = false
            trA.maxScore = 0;
            if (w.trNtotal + P.alignTranscriptsPerWindowNmax >= P.alignTranscriptsPerReadNmax) break;   // STAR: warning, stop
#ifdef M5_KEEPIX
            // the window's slot positions and spare: the first pool entries that keep[] does not hold
            for (int p = 0, j = 0; j < C::NWTR + 2; p++) if (!w.own[p]) { if (j <= C::NWTR) w.slotP[j] = p; else w.slotSpare = p; j++; }
#else
            for (int k = 0; k <= C::NWTR; k++) w.slotP[k] = k;
#endif
#if defined(M5_FIN1) && !defined(M5_KEEPIX)
            w.slotSpare = C::NWTR + 1;
#endif
#ifdef M5_KEEPIX
#define M5_SLOTAT0 w.slot[w.slotP[0]]
#else
#define M5_SLOTAT0 w.slot[0]
#endif
#ifdef M5_SLOT0
            // slot[slotP[0]] is read while nWinTr == 0 only for its maxScore (the recording test); every other read is of a
            // position < nWinTr, which a recording has written
            M5_SLOTAT0.maxScore = trA.maxScore;
#else
            M5CPY(w, 4, M5_SLOTAT0, trA);
#endif
#undef M5_SLOTAT0
            w.nWinTr = 0;
            M5STAT(w, 1);
#ifdef M5_CENSUS
            const u64 cnS0 = w.nSteps;
            long long cnT0 = 0;
#ifdef __CUDA_ARCH__
            cnT0 = clock64();
#endif
            const bool cnOk = stitchWindowAligns(P, gv, w, w.nWA[iW], (const uint (*)[WA_SIZE])w.WA[iW], Read1[trA.roStr == 0 ? 0 : 2]);
            w.cnWinSteps[iW] = (uint32_t)mn<u64>(w.nSteps - cnS0, 0xFFFFFFFFull);
#ifdef __CUDA_ARCH__
            w.cnWinCyc[iW] = (uint32_t)mn<long long>(clock64() - cnT0, 0xFFFFFFFFll);
#endif
            w.cnWinTr[iW] = (uint16_t)w.nWinTr; w.cnWinStitched++;
            if (w.nWinTr > w.cnWinTrMax) w.cnWinTrMax = w.nWinTr;
            if (!cnOk) return Status::Overflow;
#else
            if (!stitchWindowAligns(P, gv, w, w.nWA[iW], (const uint (*)[WA_SIZE])w.WA[iW], Read1[trA.roStr == 0 ? 0 : 2])) return Status::Overflow;
#endif
            if (w.nWinTr == 0) continue;
            const T& t0 = w.slot[w.slotP[0]];
            bool better = t0.maxScore > w.trBestScore || (t0.maxScore == w.trBestScore && t0.gLength < w.trBestGLength);
            // keep this window's transcripts that can still pass the score range (final maxScore >= running best)
            int runBest = mx(w.trBestScore, t0.maxScore);
            int nKeep0 = w.nKeep;
            for (uint k = 0; k < w.nWinTr; k++) {
                const T& t = w.slot[w.slotP[k]];
                if (t.maxScore + P.outFilterMultimapScoreRange >= runBest || (better && k == 0)) {
                    if (w.nKeep >= C::NKEEP) { w.ov = OV_KEEP; return Status::Overflow; }
#ifdef M5_KEEPIX
                    // the slot itself becomes keep[nKeep] (it leaves the pool until dropped; the next window's positions
                    // take other entries). Chr/Str/roStr: every transcript of a window has trInit's, so this is t0's.
                    w.keepP[w.nKeep] = w.slotP[k]; w.own[w.slotP[k]] = 1;
#else
                    M5CPY(w, 5, w.keep[w.nKeep], t);
#endif
                    M5KEEP(w, w.nKeep).Chr = t0.Chr; M5KEEP(w, w.nKeep).Str = t0.Str; M5KEEP(w, w.nKeep).roStr = t0.roStr;   // multMapSelect copies [0]'s
                    w.keepWin[w.nKeep] = (int)iW1;
                    w.nKeep++;
                }
            }
            M5CN(if ((uint)w.nKeep > w.cnKeepMax) w.cnKeepMax = (uint)w.nKeep);
            if (better) { w.trBestScore = t0.maxScore; w.trBestGLength = t0.gLength; w.trBestKeep = nKeep0; }
            // drop earlier kept transcripts that fell out of range (never trBest: it has the running best score)
            if (better) {
                int j = 0;
                for (int k = 0; k < w.nKeep; k++) {
                    if (M5KEEP(w, k).maxScore + P.outFilterMultimapScoreRange >= w.trBestScore) {
                        if (k == w.trBestKeep) w.trBestKeep = j;
#ifdef M5_KEEPIX
                        if (j != k) { w.keepP[j] = w.keepP[k]; w.keepWin[j] = w.keepWin[k]; }
#else
                        if (j != k) { M5CPY(w, 6, w.keep[j], w.keep[k]); w.keepWin[j] = w.keepWin[k]; }
#endif
                        j++;
                    }
#ifdef M5_KEEPIX
                    else w.own[w.keepP[k]] = 0;   // dropped: back to the pool
#endif
                }
                w.nKeep = j;
            }
            w.trNtotal += w.nWinTr;
            iW1++;
        }
        w.nW = iW1;
        if (w.trBestScore == 0) {   // trBest->maxScore == 0
            w.mapMarker = MARKER_NO_GOOD_WINDOW;
            w.nW = 0;
        }
    }
    if (stitched) M5CLK(w, 2, tc);
    // ---- multMapSelect
    w.nTr = 0;
    int iBest = -1;
    if (w.nW > 0) {
        w.maxScore = w.trBestScore;
        for (int k = 0; k < w.nKeep; k++) {
            if (M5KEEP(w, k).maxScore + P.outFilterMultimapScoreRange >= w.maxScore) {
#ifdef M5_KEEPIX
                if (k != (int)w.nTr) { w.keepP[w.nTr] = w.keepP[k]; }
#else
                if (k != (int)w.nTr) { M5CPY(w, 7, w.keep[w.nTr], w.keep[k]); }
#endif
                if (k == w.trBestKeep) iBest = (int)w.nTr;
                w.nTr++;
            }
        }
        if (!(w.nTr > P.outFilterMultimapNmax || w.nTr == 0)) {
            for (uint iTr = 0; iTr < w.nTr; iTr++) {
                T& t = M5KEEP(w, iTr);
                t.roStart = t.roStr == 0 ? t.rStart : Lread - t.rStart - t.rLength;
                t.cStart = t.gStart - gv.chrStart[t.Chr];
            }
            if (w.nTr == 1) {
                M5KEEP(w, 0).primaryFlag = true;
            } else {
                // outMultimapperOrder Old_2.4, outSAMmultNmax -1, outSAMprimaryFlag OneBestScore (checked by the driver)
                M5KEEP(w, iBest).primaryFlag = true;
            }
        }
    }
    // ---- mappedFilter
    w.unmapType = -1;
    if (w.nW == 0) {
        w.unmapType = 0;
    } else {
        const T& b = M5KEEP(w, iBest);
        if ((b.maxScore < P.outFilterScoreMin) || (b.maxScore < (int)(P.outFilterScoreMinOverLread * (Lread - 1)))
            || (b.nMatch < P.outFilterMatchNmin) || (b.nMatch < (uint)(P.outFilterMatchNminOverLread * (Lread - 1)))) {
            w.unmapType = 1;
        } else if ((b.nMM > w.outFilterMismatchNmaxTotal) || (double(b.nMM) / double(b.rLength) > P.outFilterMismatchNoverLmax)) {
            w.unmapType = 2;
        } else if (w.nTr > P.outFilterMultimapNmax) {
            w.unmapType = 3;
        }
    }
    // ---- encode (m5_dump.h record)
    blob.putV((i64)w.nW); blob.putV((i64)w.nTr); blob.putV((i64)(w.nW > 0 ? iBest : -1));
    if (w.nW == 0) blob.putV((i64)w.trInitRLength);
    for (uint i = 0; i < w.nTr; i++) encodeTr(blob, M5KEEP(w, i));
    blob.flush();
    M5CLK(w, 3, tc);
    if (blob.full) { w.ov = OV_BLOB; return Status::BlobFull; }
    return Status::Ok;
}


// one read, all phases in one call (host lane, wide GPU lane)
template <class C> HD inline Status mapRead(const Params& P, const GenomeView& gv, Work<C>& w, ReadBuf<C::LREAD>& rb,
                                         const char* R0, uint Lread, Blob& blob) {
    Status st = mapPhase1(P, gv, w, rb, R0, Lread);
    if (st != Status::Ok) return st;
    if (w.stitched && !assignAll(P, gv, w)) return Status::Overflow;
    return mapFinish(P, gv, w, rb, blob);
}

} // namespace m5
