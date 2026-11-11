// E9: STARsolo (CB_UMI_Simple, Gene, 1MM_multi_Nbase_pseudocounts, 1MM_All, Unique, CellRanger2.2, soloCellReadStats Standard) as rules that
// run per distinct sequence (record), per distinct CR, per read and per (CB, gene, UMI) on the device or the host from one source.
// docs/gpu-engine/fable/E9-solo.md. Every function names the STAR code it ports (src/star, STAR 2.7.11b); integer widths and float rules are
// STAR's (cbMinP and the posterior are float: MATCH_CellRanger is not defined in src/star/CMakeLists.txt).
#pragma once
#include <cstdint>
#include "records.cuh"

#ifdef __CUDACC__
#define E9HD __host__ __device__
#else
#define E9HD
#endif

namespace e9 {

typedef uint64_t u64;
typedef uint32_t u32;
typedef int64_t i64;
typedef int32_t i32;

// ---- STAR's counters (names and order = the files' rows)
// SoloReadFlagClass (SoloCommon.h): CellReads.stats columns
enum : u32 { F_cbMatch, F_cbPerfect, F_cbMMunique, F_cbMMmultiple, F_genomeU, F_genomeM, F_featureU, F_featureM, F_exonic, F_intronic,
             F_exonicAS, F_intronicAS, F_mito, F_countedU, F_countedM, F_N };
// SoloReadBarcodeStats.h: Barcodes.stats rows
enum : u32 { B_noNoAdapter, B_noNoUMI, B_noNoCB, B_noNinCB, B_noNinUMI, B_noUMIhomopolymer, B_noNoWLmatch, B_noTooManyMM, B_noTooManyWLmatches,
             B_yesWLmatchExact, B_yesOneWLmatchWithMM, B_yesMultWLmatchWithMM, B_N };
// SoloReadFeatureStats.h: Features.stats rows
enum : u32 { S_noUnmapped, S_noNoFeature, S_MultiFeature, S_subMultiFeatureMultiGenomic, S_noTooManyWLmatches, S_noMMtoWLwithoutExact,
             S_yesWLmatch, S_yessubWLmatchExact, S_yessubWLmatch_UniqueFeature, S_yesCellBarcodes, S_yesUMIs, S_N };

// ---- atomics that are the same on both sides (integer only: the results do not depend on the order)
#ifdef __CUDA_ARCH__
__device__ inline void addU32(u32* p, u32 v) { atomicAdd((unsigned*)p, (unsigned)v); }
__device__ inline void addU64(u64* p, u64 v) { atomicAdd((unsigned long long*)p, (unsigned long long)v); }
__device__ inline u32 casU32(u32* p, u32 cmp, u32 v) { return atomicCAS((unsigned*)p, cmp, v); }
__device__ inline u32 ctz32(u32 x) { return (u32)(__ffs((int)x) - 1); }
__device__ inline u32 ctz64(u64 x) { return (u32)(__ffsll((long long)x) - 1); }
// base[idx] += 1, aggregated over the warp's threads with the same idx (counters that most reads hit)
__device__ inline void incAgg(u64* base, u32 idx) {
    const unsigned mask = __activemask();
    const unsigned peers = __match_any_sync(mask, idx);
    if ((threadIdx.x & 31) == (unsigned)(__ffs((int)peers) - 1)) atomicAdd((unsigned long long*)(base + idx), (unsigned long long)__popc(peers));
}
#else
inline void incAgg(u64* base, u32 idx) { __atomic_fetch_add(base + idx, (u64)1, __ATOMIC_RELAXED); }
inline void addU32(u32* p, u32 v) { __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
inline void addU64(u64* p, u64 v) { __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
inline u32 casU32(u32* p, u32 cmp, u32 v) { __atomic_compare_exchange_n(p, &cmp, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED); return cmp; }
inline u32 ctz32(u32 x) { return (u32)__builtin_ctz(x); }
inline u32 ctz64(u64 x) { return (u32)__builtin_ctzll(x); }
#endif

// ---- parameters (ParametersSolo, parametersDefault, the pipeline's STAR command line)
struct SoloParams {
    u32 cbL = 16, umiL = 12;   // soloCBlen, soloUMIlen
    i32 strand = 0;            // soloStrand Forward (ParametersSolo.cpp:194)
    u32 wlN = 0;               // cbWLsize (sorted, deduplicated whitelist)
    u32 nGe = 0;               // Trans.nGe = featuresNumber
    float cbMinP = 0.975f;     // ParametersSolo.h:139 (float), ParametersSolo.cpp:289
    double pq = 0;             // std::pow(10.0, -qin/10.0) for the CB quality (host glibc; SoloReadFeature_inputRecords.cpp:107) when every
                               // CB base of the run gives the same qin; E15: with per-read qualities (SoloReads::cbQual) SoloModel::pqTab instead
    u32 umiMaskLow = 0;        // ParametersSolo.cpp:291
    u32 homo[4] = {0, 0, 0, 0};   // SoloReadBarcode.cpp:15-19
    i64 mitoChr[4] = {-1, -1, -1, -1};  // P.pGe.chrSet.mito (Genome.cpp:201: an absent name inserts nChr)
    u32 nMito = 0;
};

// ---- the transcript model (Transcriptome.cpp:38-80; transcriptInfo.tab, exonInfo.tab)
struct TrModel {
    const u64* trS; const u64* trE; const u64* trEmax; const u32* trExI; const uint16_t* trExN; const uint8_t* trStr; const u32* trGene;
    const u32* exSE; u32 nTr;
};

// serviceFuns.cpp:239 binarySearch1a<uint>: the last X element <= x (-1 if x < X[0])
E9HD inline i32 binarySearch1a(u64 x, const u64* X, i32 N) {
    if (x > X[N - 1]) return N - 1;
    else if (x < X[0]) return -1;
    i32 i1 = 0, i2 = N - 1, i3 = N / 2;
    while (i2 > i1 + 1) { i3 = (i1 + i2) / 2; if (X[i3] > x) i2 = i3; else i1 = i3; }
    while (i1 < N - 1 && x == X[i1 + 1]) ++i1;
    return i1;
}
// serviceFuns.cpp:212 binarySearch_leLeft<uint32>
E9HD inline bool binarySearch_leLeft(u32 x, const u32* X, u32 N, u32& i1) {
    if (x > X[N - 1] || x < X[0]) return false;
    i1 = 0;
    u32 i2 = N - 1, i3 = N / 2;
    while (i2 > i1 + 1) { i3 = (i1 + i2) / 2; if (X[i3] > x) i2 = i3; else i1 = i3; }
    while (i1 > 0 && x == X[i1 - 1]) --i1;
    return true;
}

// AlignVsTranscript.h
enum { AVT_Intron = 0, AVT_ExonIntron = 1, AVT_ExonIntronSpan = 2, AVT_Concordant = 3 };
// Transcriptome_classifyAlign.cpp:8 alignToTranscript: the status only (distTrEnds is not read by the Gene feature)
E9HD inline int alignToTranscript(const u64* G, const u64* Lx, const int* canon, u32 nEx, u64 trS1, uint16_t exN1, const u32* exSE1) {
    bool alignIntronic = false, alignExonic = false, alignSpansExonIntr = false, alignSJconcordant = true;
    for (u32 iab = 0, ex1 = 0, bS = 0, bE = 0, eE = 0, enS = 0; iab < nEx; iab++) {
        const u64 bEprev = bE;
        if (G[iab] < trS1) return -1;
        bS = (u32)(G[iab] - trS1);
        bE = (u32)(bS + Lx[iab] - 1);
        if (iab == 0 || canon[iab - 1] == -3) {
            if (!binarySearch_leLeft(bS, exSE1, 2 * (u32)exN1, ex1)) return -1;
            ex1 = ex1 / 2;
        } else if (canon[iab - 1] >= 0) {
            if (bEprev == eE && bS == enS) {
                ++ex1;
            } else {
                alignSJconcordant = false;
                if (!binarySearch_leLeft(bS, exSE1, 2 * (u32)exN1, ex1)) return -1;
                ex1 = ex1 / 2;
            }
        }
        eE = exSE1[2 * ex1 + 1];
        enS = ex1 + 1 < exN1 ? exSE1[2 * (ex1 + 1)] : 0;
        if (bS <= eE) {
            if (bE > eE) alignSpansExonIntr = true;
            alignExonic = true;
        } else {
            if (bE >= enS) alignSpansExonIntr = true;
            alignIntronic = true;
        }
    }
    if (!alignSJconcordant) return -1;
    if (alignSpansExonIntr) return AVT_ExonIntronSpan;
    else if (!alignIntronic) return AVT_Concordant;
    else return alignExonic ? AVT_ExonIntron : AVT_Intron;
}

// ---- stage (b), per record: what Solo's Gene feature sees for every read of the record's sequence
// ReadAlign_seqCache.cpp:325 seqCacheHandoffSummary (mito flag, classifyAlign over trMult[0..nTr)); an unmapped sequence (mappedFilter,
// multimapper cap: e6::recordCount = E6's port) gives Solo nTr = 0 and an empty gene set (ReadAlign_outputAlignments.cpp:145, :78 reset).
struct RecAnn { u32 gene; uint16_t nTr; uint8_t ng; uint8_t bits; };   // ng: |fSet| capped at 2; bits: 1 mito, 2 exonic, 0x40 exon overflow
enum : uint8_t { RA_MITO = 1, RA_EXONIC = 2, RA_OVERFLOW = 0x40 };
static_assert(sizeof(RecAnn) == 8, "RecAnn");
constexpr u32 MAXEX = 32;   // STAR's MAX_N_EXONS is 20 (IncludeDefine.h:131)

E9HD inline RecAnn recAnnotate(const e6::RecParams& RP, const uint8_t* blob, u64 L, const TrModel& M, const SoloParams& S) {
    RecAnn a{0xFFFFFFFFu, 0, 0, 0};
    const e6::RecCount rc = e6::recordCount(RP, blob, L);
    if (rc.unmapType >= 0) return a;
    const uint8_t* p = blob;
    const u64 nW = (u64)e6::getV(p), nTr = (u64)e6::getV(p);
    e6::getV(p);
    if (nW == 0) e6::getV(p);
    a.nTr = (uint16_t)(nTr < 65535 ? nTr : 65535);
    u64 G[MAXEX], Lx[MAXEX]; int canon[MAXEX];
    u32 g0 = 0xFFFFFFFFu; bool multi = false;
    for (u64 i = 0; i < nTr; i++) {
        e6::TrHead t; e6::parseHead(p, t);
        for (u32 k = 0; k < S.nMito; k++) if ((i64)t.Chr == S.mitoChr[k]) a.bits |= RA_MITO;
        const u64 nEx = t.nExons;
        for (u64 iex = 0; iex < nEx; iex++) {
            e6::skipV(p);                                   // EX_R
            const u64 g = (u64)e6::getV(p), l = (u64)e6::getV(p);
            e6::skipV(p); e6::skipV(p);                     // EX_iFrag, EX_sjA
            int c = 0;
            if (iex + 1 < nEx) { c = (int)e6::getV(p); e6::skipV(p); e6::skipV(p); if (c >= 0) { e6::skipV(p); e6::skipV(p); } }
            if (iex < MAXEX) { G[iex] = g; Lx[iex] = l; canon[iex] = c; }
        }
        if (nEx > MAXEX) { a.bits |= RA_OVERFLOW; continue; }
        if (nEx == 0) continue;
        // Transcriptome_classifyAlign.cpp:196-243
        u32 tr1 = (u32)binarySearch1a(G[0], M.trS, (i32)M.nTr);
        if (tr1 == (u32)-1) continue;
        const u64 aGend = G[nEx - 1] + Lx[nEx - 1] - 1;
        ++tr1;
        do {
            --tr1;
            if (aGend > M.trE[tr1] || (S.strand >= 0 && (M.trStr[tr1] == 1 ? t.Str : 1 - t.Str) != (u64)(u32)S.strand)) continue;
            const int st = alignToTranscript(G, Lx, canon, (u32)nEx, M.trS[tr1], M.trExN[tr1], M.exSE + 2 * (u64)M.trExI[tr1]);
            if (st == AVT_Concordant) {
                const u32 g = M.trGene[tr1];
                if (g0 == 0xFFFFFFFFu) g0 = g; else if (g != g0) multi = true;
            }
        } while (M.trEmax[tr1] >= aGend && tr1 > 0);
    }
    if (g0 != 0xFFFFFFFFu) {
        a.bits |= RA_EXONIC;   // classifyAlign:245 ovType = exonic when fSet is not empty
        a.ng = multi ? 2 : 1;
        a.gene = multi ? 0xFFFFFFFFu : g0;
    }
    return a;
}

// ---- stage (a), per distinct CR: SoloReadBarcode_getCBandUMI.cpp:7 matchCBtoWL with the whitelist as a sorted array (ParametersSolo.cpp:333:
// cbWL is sorted and deduplicated; hashLookupCB returns the index in it, so a binary search returns the same index)
E9HD inline i64 wlFind(const u64* wl, u32 n, u64 x) {
    u32 lo = 0, hi = n;
    while (lo < hi) { const u32 m = (lo + hi) >> 1; if (wl[m] < x) lo = m + 1; else hi = m; }
    return (lo < n && wl[lo] == x) ? (i64)lo : -1;
}
// SequenceFuns.cpp:262 convertNuclStrToInt64 over a 3-bit packed string (A0 C1 G2 T3, other 4: handoffReadEntry's packing): -1 no N, -2 >1 N
E9HD inline i64 convertNucl3(u64 s3, u32 n, u64& v) {
    v = 0; i64 posN = -1;
    for (u32 j = 0; j < n; j++) {
        u64 nt = (s3 >> (3 * j)) & 7;
        if (nt > 3) { if (posN >= 0) return -2; posN = j; nt = 0; }
        v = v << 2; v += nt;
    }
    return posN;
}
constexpr u32 MAXCAND = 48;   // 3 * cbL for cbL = 16
// returns cbMatch (-1 no match, -2 more than one N, 0 exact, >= 1 number of matches) and the matches in STAR's order (out may be null)
E9HD inline i32 matchCBtoWL(u64 cr3, u32 cbL, const u64* wl, u32 wlN, u32* out, u32& n) {
    n = 0;
    u64 cbB1;
    const i64 posN = convertNucl3(cr3, cbL, cbB1);
    if (posN == -2) return -2;
    if (posN == -1) { const i64 i = wlFind(wl, wlN, cbB1); if (i >= 0) { if (out) out[0] = (u32)i; n = 1; return 0; } }
    i32 m = 0;
    if (posN >= 0) {
        const u32 sh = 2 * (cbL - 1 - (u32)posN);
        for (u32 jj = 0; jj < 4; jj++) { const i64 i = wlFind(wl, wlN, cbB1 ^ ((u64)jj << sh)); if (i >= 0) { if (out) out[n] = (u32)i; n++; ++m; } }
    } else {
        for (u32 ii = 0; ii < cbL; ii++)
            for (u32 jj = 1; jj < 4; jj++) { const i64 i = wlFind(wl, wlN, cbB1 ^ ((u64)jj << (ii * 2))); if (i >= 0) { if (out) out[n] = (u32)i; n++; ++m; } }
    }
    return m == 0 ? -1 : m;   // mm1_multi: more than one match is kept (no -3)
}
// SoloReadBarcode_getCBandUMI.cpp:127 convertCheckUMI: 0 good (umiB set), -23 N in the UMI, -24 homopolymer
E9HD inline i32 checkUMI(u64 ur3, const SoloParams& S, u64& umiB) {
    if (convertNucl3(ur3, S.umiL, umiB) != -1) return -23;
    for (int k = 0; k < 4; k++) if (umiB == (u64)S.homo[k]) return -24;
    return 0;
}
// SoloReadBarcode_getCBandUMI.cpp:85 addStats
E9HD inline int barcodeStat(i32 cbMatch) {
    switch (cbMatch) {
        case 0: return B_yesWLmatchExact;
        case 1: return B_yesOneWLmatchWithMM;
        case -1: return B_noNoWLmatch;
        case -2: return B_noNinCB;
        case -3: return B_noTooManyWLmatches;
        case -11: return B_noNoCB;
        case -12: return B_noTooManyMM;
        case -23: return B_noNinUMI;
        case -24: return B_noUMIhomopolymer;
        default: return B_yesMultWLmatchWithMM;
    }
}
// SoloReadFeature_inputRecords.cpp:91-121: the posterior of one read whose CR has n > 1 whitelist matches (float; exact counts + 1 pseudocount,
// Solo.cpp:66-70); returns the chosen whitelist index or -1 (noTooManyWLmatches). E15: per read, with candidate k's quality character as
// matchCBtoWL writes it into cbMatchString (SoloReadBarcode_getCBandUMI.cpp:58, :71): the read's CB quality at the N (cbQual1[posN]) for a CR
// with one N, else at the mismatch (cbQual1[cbL - 1 - ii] for the 2-bit symbol ii in which the candidate differs from the CR). qual: the read's
// cbL CB quality characters; pqTab[c]: std::pow(10.0, -qin/10.0) for the character c, qin = min((char)c - QSbase, QSmax) in STAR's char
// arithmetic (:104-107, the host's pow). qual == nullptr: every candidate takes S.pq (the run's CB bases all give one qin).
E9HD inline i64 cbPosteriorRead(u64 cr3, const u32* cand, u32 n, const u32* exactCnt, const u64* wl, const uint8_t* qual, const double* pqTab,
                                const SoloParams& S) {
    u64 cbB1 = 0;
    const i64 posN = qual != nullptr ? convertNucl3(cr3, S.cbL, cbB1) : -1;
    float ptot = 0.0f, pmax = 0.0f, pin;
    i64 cb = -1;
    for (u32 ii = 0; ii < n; ii++) {
        const u32 cbin = cand[ii];
        const u32 c = exactCnt[cbin] + 1u;
        if (c > 0) {
            double pq = S.pq;
            if (qual != nullptr) {
                const u32 pos = posN >= 0 ? (u32)posN : S.cbL - 1 - ctz64(wl[cbin] ^ cbB1) / 2;
                pq = pqTab[qual[pos]];
            }
            pin = (float)((double)c * pq);
            ptot += pin;
            if (pin > pmax) { cb = cbin; pmax = pin; }
        }
    }
    if (ptot > 0.0f && pmax >= S.cbMinP * ptot) return cb;
    return -1;
}

// ---- stage (c): keys of the counted reads, (CB, gene, UMI); CB = whitelist index (22 bits), gene (17 bits), UMI (24 bits)
constexpr u32 GENE_BITS = 17, UMI_BITS = 24;
E9HD inline u64 tupleKey(u64 cb, u64 gene, u64 umi) { return (cb << (GENE_BITS + UMI_BITS)) | (gene << UMI_BITS) | umi; }

// SoloParams::umiSwapHalves (ParametersSolo.cpp:500) on the UMI part of a key
E9HD inline u64 swapHalvesKey(u64 key, u32 umiL, u32 maskLow) {
    u32 umi = (u32)(key & ((1u << UMI_BITS) - 1));
    const u32 high = umi >> umiL;
    umi &= maskLow; umi <<= umiL; umi |= high;
    return (key & ~(u64)((1u << UMI_BITS) - 1)) | umi;
}
// SoloFeature_collapseUMI_Graph.cpp:89-94: one 2-bit symbol differs
E9HD inline bool oneMM(u32 uuXor) { return !((uuXor >> (ctz32(uuXor) / 2) * 2) > 3); }

// union-find over the distinct (CB, gene, UMI); the number of roots per (CB, gene) is the 1MM_All count (collapseUMIwith1MMlowHalf colours
// plus graphNumberOfConnectedComponents = the connected components of the Hamming-1 graph of the gene's distinct UMIs)
E9HD inline u32 ufFind(u32* par, u32 x) {
    while (true) {
        const u32 p = par[x];
        if (p == x) return x;
        const u32 gp = par[p];
        if (gp != p) par[x] = gp;   // path halving (a benign race: an entry only ever points to an ancestor)
        x = p;
    }
}
E9HD inline void ufUnion(u32* par, u32 a, u32 b) {
    while (true) {
        a = ufFind(par, a); b = ufFind(par, b);
        if (a == b) return;
        if (a < b) { const u32 t = a; a = b; b = t; }   // link the larger root under the smaller
        if (casU32(&par[a], a, b) == a) return;
    }
}

}  // namespace e9
