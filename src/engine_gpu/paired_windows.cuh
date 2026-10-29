// Phase 1d (paired-end on the GPU): DRAFT, NOT COMPILED, NOT RUN. Written on a laptop without nvcc; W11 builds and checks it.
// docs/research-beta/paired-end-prep.md has the rules table (STAR file:line for every rule here) and the build and run steps.
//
// What STAR does for a pair is almost what stitch.cuh already does for a read. STAR joins the mates into ONE read
// (ReadAlign_oneRead.cpp:40-58: mate 1, a spacer base MARK_FRAG_SPACER_BASE, the reverse complement of mate 2), seeds that read
// (qualitySplit splits at the spacer and counts the mate in iFrag), puts the seeds of both mates into the same windows (a window
// is a run of 2^winBinNbits bins, so "the mates' windows are joined" means: the two mates' seeds fall into bins within
// winAnchorDistNbins, which Genome_genomeLoad.cpp:426-446 derives from alignMatesGapMax), and stitches them with the same
// recursion. stitch.cuh ports all of that, including the mate branch of stitchAlignToTranscript (-1000004, -1000008, canonSJ -3).
// Only four places assume a single mate. This header leaves stitch.cuh untouched and re-declares them in namespace m5::pe:
//
//   1. finalizeTranscript, stitchWindowAligns, mapFinish   copied from stitch.cuh with the changes marked "PE:" below
//        - stitchWindowAligns.cpp:159  the spliced-mate length check uses readLength[mate of the exon], not Lread
//        - stitchWindowAligns.cpp:179-219  the check for mates (negative insert size, protrusion, junction consistency), which
//          stitch.cuh:1048 replaces by `return true` (drop the transcript)
//        - stitchWindowAligns.cpp:232-234  maxScoreMate has two entries (stitch.cuh:206 declares one)
//   2. mapPhase1PE        mapPhase1 plus outFilterMismatchNmaxTotal from the two mate lengths (oneRead.cpp:83; stitch.cuh:1234 uses Lread)
//   3. combineMates       the combined read (oneRead.cpp:52-58)
//   4. Caps               the lane capacities for a pair (LREAD = 2 * mate + 1)
//
// The unmodified code is reused as it is: seeding (maxMappableLength2strands, storeAligns), windows (createExtendWindowsWithAlign,
// assignAlignToWindow), stitchAlignToTranscript, extendAlign, blocksOverlap, encodeTr. When 1d is approved the clean change is
// to template stitch.cuh on a `Mates` policy (readLength[], maxScoreMate[2], mateOverlapOk) and delete the copies here; the copies
// exist only so this draft needs no edit to stitch.cuh. Every copied line outside a "PE:" marker is identical to stitch.cuh except that the M5T/M5ADD timing lines are left out.
//
// Clipping is per mate (clipMate below, k_pe_map applies it before the mates are combined). What is NOT covered (the driver must route
// these to the host lane, as for single-end): sjdb-inserted seeds are covered by the shared code; peOverlap merging
// (--peOverlapNbasesMin > 0, ReadAlign_peOverlapMergeMap.cpp), --clipAdapterType CellRanger4, chimeric detection, variation, WASP and
// BySJout stage 2 are not (the driver checks them, as fillParams does).
#pragma once
#include "stitch.cuh"

namespace m5 {
namespace pe {   // m5::pe: lookup reaches m5::uint (STAR's 64-bit uint) before ::uint, as inside stitch.cuh

// Lane for pairs: LREAD = 2 * 150 + 1 rounded up. NSPLIT 8: both mates can split at Ns. NEX 20 (STAR's MAX_N_EXONS): a pair has
// up to two mates' worth of exons. Provisional sizes; a pair that overflows goes to the host lane (Status::Overflow).
typedef Caps<304, 8, 192, 128, 50, 4096, 20, 32, 64, 8192> PeLane;
// Heavy lane: STAR's limits (DEF_readSeqLengthMax is 650, so a pair of 2 x 324 is the longest STAR accepts, oneRead.cpp:43)
typedef Caps<650, 50, 1000, 10000, 50, 1 << 20, 20, 100, 4096, 1 << 20> PeHeavy;

// Work<C> with the two per-mate fields stitch.cuh does not have
template <class C> struct WorkPE : Work<C> {
    uint readLength[2];     // STAR's readLength[0], readLength[1] (after clipping; the driver does no clipping)
    int maxScoreMateP[2];   // STAR's maxScoreMate[0..1] (ReadAlign.h); Work<C>::maxScoreMate has one entry
};

// ReadAlign_oneRead.cpp:40-58. out must hold L0 + 1 + L1 bytes. m1, m2: numeric bases (A0 C1 G2 T3, N4), as STAR's Read1.
// Returns Lread = L0 + L1 + 1.
HD inline uint combineMates(const char* m1, uint L0, const char* m2, uint L1, char* out) {
    for (uint i = 0; i < L0; i++) out[i] = m1[i];
    out[L0] = (char)MARK_FRAG_SPACER_BASE;
    for (uint i = 0; i < L1; i++) {            // complement, then reverse (complementSeqNumbers leaves values > 3 unchanged)
        char c = m2[L1 - 1 - i];
        out[L0 + 1 + i] = c < 4 ? (char)(3 - c) : c;
    }
    return L0 + L1 + 1;
}

// ---------------------------------------------------------------------------------------------------------------------
// Clipping of one mate: STAR's readLoad.cpp:75-76 (5' clip, then 3' clip) with ClipMate_clip.cpp:5-78, Hamming adapter type
// (--clip5pNbases, --clip5pAfterAdapterNbases, --clip3pNbases, --clip3pAdapterSeq, --clip3pAdapterMMp, --clip3pAfterAdapterNbases),
// one MateClip per mate: ParametersClip_initialize.cpp:84-107 gives each mate its own value. STAR removes the clipped bases
// before the mates are combined, so the aligner and the combined read see only [start, start + len). c5 and c3 are STAR's
// clipMates[mate][0].clippedN and [1].clippedN, which the SAM CIGAR needs (outputTranscriptSAM.cpp:136-145, 184-186).
// The same function as k_clip in engine_gpu.cu:77-103 (mate 1 only) except for the case marked "STAR quirk" here, which k_clip does not have.
// Reference model: bench/gpu_paired/paired_model.py clip_mate.
// ---------------------------------------------------------------------------------------------------------------------
struct MateClip { uint32_t n5, after5, n3, after3, nad; double adMMp; const char* ad; };   // ad: numeric adapter (nad bases); polyA = nad 650 zeros
struct ClipSpan { u64 start, len, c5, c3; };
HD inline ClipSpan clipMate(const MateClip& c, const char* seq, u64 L0) {
    u64 L = L0, s = 0, c5 = 0, c3 = 0;
    if (c.n5 > 0) {                                  // 5' has type 0 only when N > 0 (no 5' adapter: ParametersClip_initialize.cpp:34-39)
        const u64 Lold = L;
        if (L > c.n5) { L -= c.n5; s += c.n5; c5 += c.n5; } else { L = 0; c5 = Lold; }                                   // :14-25
        if (c.after5 > 0) { if (L > c.after5) { L -= c.after5; s += c.after5; c5 += c.after5; } else { L = 0; c5 = Lold; } }   // :64-75
    }
    if (c.n3 > 0 || c.nad > 0) {                     // STAR quirk (ClipMate_initialize.cpp:22-23, ClipMate_clip.cpp:9-10): with N == 0 and no
        const u64 Lold = L;                          // adapter the clip has type -1 and does nothing, so AfterAdapterNbases alone is ignored
        if (c.n3 > 0) { if (L > c.n3) { L -= c.n3; c3 += c.n3; } else { L = 0; c3 = Lold; } }
        if (c.nad > 0) {                             // localSearch (SequenceFuns.cpp:304-326) on the 5'-clipped, N-clipped sequence
            const char* x = seq + s;
            u64 nMatchBest = 0, nMMbest = 0, ixBest = L;
            for (u64 ix = 0; ix < L; ix++) {
                u64 nMatch = 0, nMM = 0, m = (u64)c.nad < L - ix ? (u64)c.nad : L - ix;
                for (u64 iy = 0; iy < m; iy++) {
                    if (x[ix + iy] > 3) continue;
                    if (x[ix + iy] == c.ad[iy]) nMatch++; else nMM++;
                }
                if ((nMatch > nMatchBest || (nMatch == nMatchBest && nMM < nMMbest)) && nMatch > 0 && double(nMM) / double(nMatch) <= c.adMMp) {
                    ixBest = ix; nMatchBest = nMatch; nMMbest = nMM;
                }
            }
            c3 += L - ixBest; L = ixBest;
        }
        if (c.after3 > 0) { if (L > c.after3) { L -= c.after3; c3 += c.after3; } else { L = 0; c3 = Lold; } }          // :64-75
    }
    ClipSpan r; r.start = s; r.len = L; r.c5 = c5; r.c3 = c3;
    return r;
}

// ---------------------------------------------------------------------------------------------------------------------
// stitchWindowAligns.cpp:179-219: "check for correct overlap between mates", for a transcript whose first and last exons are
// in different mates. true = keep the transcript. protrude = alignEndsProtrude.nBasesMax.
// Reference model: bench/gpu_paired/paired_model.py mate_overlap_ok (same line numbers in its comments).
// ---------------------------------------------------------------------------------------------------------------------
template <class T> HD inline bool mateOverlapOk(const T& t, uint protrude, uint Lread) {
    const uint n = t.nExons;
    if (t.exons[n - 1][EX_G] + t.exons[n - 1][EX_L] <= t.exons[0][EX_G]) return false;           // :180 negative insert size
    uint iexM2 = n;
    for (uint iex = 0; iex < n - 1; iex++) {                                                       // :182 first exon of mate 2
        if (t.canonSJ[iex] == -3) { iexM2 = iex + 1; break; }
    }
    if (iexM2 >= n) return false;   // unreachable (the mate branch of stitchAlignToTranscript always writes -3); STAR would read past nExons
    if (t.exons[iexM2 - 1][EX_G] + t.exons[iexM2 - 1][EX_L] > t.exons[iexM2][EX_G]) {              // :189 mates overlap
        if (t.exons[0][EX_G] > t.exons[iexM2][EX_G] + t.exons[0][EX_R] + protrude) return false;   // :191 left start > right start + allowance
        if (t.exons[iexM2 - 1][EX_G] + t.exons[iexM2 - 1][EX_L] >
            t.exons[n - 1][EX_G] + Lread - t.exons[n - 1][EX_R] + protrude) return false;           // :193 left end > right end + allowance
        uint iex1 = 1, iex2 = iexM2 + 1;                                                           // :197 last exons of the junctions
        for (; iex1 < iexM2; iex1++)                                                               // :198 first junction that overlaps mate 2
            if (t.exons[iex1][EX_G] >= t.exons[iex2 - 1][EX_G] + t.exons[iex2 - 1][EX_L]) break;
        while (iex1 < iexM2 && iex2 < n) {                                                         // :201
            if (t.canonSJ[iex1 - 1] < 0) { iex1++; continue; }
            if (t.canonSJ[iex2 - 1] < 0) { iex2++; continue; }
            if (t.exons[iex1][EX_G] != t.exons[iex2][EX_G] ||
                (t.exons[iex1 - 1][EX_G] + t.exons[iex1 - 1][EX_L]) != (t.exons[iex2 - 1][EX_G] + t.exons[iex2 - 1][EX_L]))
                return false;                                                                      // :211 inconsistent junctions
            iex1++; iex2++;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// stitchWindowAligns.cpp finalization. Copy of m5::finalizeTranscript with the "PE:" changes.
// ---------------------------------------------------------------------------------------------------------------------
template <class C> HD inline bool finalizeTranscriptPE(const Params& P, const GenomeView& gv, WorkPE<C>& w, int Score, uint tR2, uint tG2,
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
                // PE: stitchWindowAligns.cpp:159, readLength of the mate this exon belongs to (stitch.cuh:1041 uses Lread)
                if (nsj > 0 && (exl < P.alignSplicedMateMapLmin
                                || exl < (uint)(P.alignSplicedMateMapLminOverLmate * w.readLength[trA.exons[iex][EX_iFrag]]))) return true;
                exl = 0; nsj = 0;
            } else if (trA.canonSJ[iex] >= 0) {
                nsj++;
            }
        }
    }
    // PE: stitchWindowAligns.cpp:179-219 (stitch.cuh:1048 drops every transcript that has two mates)
    if (trA.exons[0][EX_iFrag] != trA.exons[trA.nExons - 1][EX_iFrag]) {
        if (!mateOverlapOk(trA, (uint)P.alignEndsProtrudeNbasesMax, Lread)) return true;
    }
    if (P.scoreGenomicLengthLog2scale != 0) {
        Score += genomicLengthScore(P, trA.exons[trA.nExons - 1][EX_G] + trA.exons[trA.nExons - 1][EX_L] - trA.exons[0][EX_G]);
        Score = mx(0, Score);
    }
    trA.roStart = (trA.roStr == 0) ? trA.rStart : Lread - trA.rStart - trA.rLength;
    trA.maxScore = Score;
    if (trA.exons[0][EX_iFrag] == trA.exons[trA.nExons - 1][EX_iFrag]) {
        trA.iFrag = (int)trA.exons[0][EX_iFrag];
        w.maxScoreMateP[trA.iFrag] = mx(w.maxScoreMateP[trA.iFrag], Score);   // PE: two entries
    } else {
        trA.iFrag = -1;
    }
    trA.maxScore = Score;   // + variationAdjust = 0 (no variation)
    typedef typename Work<C>::T T;
    if (Score + P.outFilterMultimapScoreRange >= w.slot[w.slotP[0]].maxScore
        || (trA.iFrag >= 0 && Score + P.outFilterMultimapScoreRange >= w.maxScoreMateP[trA.iFrag])) {   // PE: two entries
        uint iTr = 0;
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
        }
        if (iTr == w.nWinTr) {
            for (iTr = 0; iTr < w.nWinTr; iTr++) {
                const T& o = w.slot[w.slotP[iTr]];
                if (Score > o.maxScore || (Score == o.maxScore && trA.gLength < o.gLength)) break;
            }
            int pTr = w.slotP[w.nWinTr];
            for (int ii = (int)w.nWinTr; ii > int(iTr); ii--) w.slotP[ii] = w.slotP[ii - 1];
            w.slotP[iTr] = pTr;
            w.slot[w.slotP[iTr]].copyFrom(trA);
            if (w.nWinTr < P.alignTranscriptsPerWindowNmax) {
                if (w.nWinTr + 1 > (uint)C::NWTR) { w.ov = OV_WTR; return false; }
                w.nWinTr++;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// stitchWindowAligns.cpp recursion on an explicit stack. Copy of m5::stitchWindowAligns, calling finalizeTranscriptPE.
// ---------------------------------------------------------------------------------------------------------------------
template <class C> HD inline bool stitchWindowAlignsPE(const Params& P, const GenomeView& gv, WorkPE<C>& w, uint nA, const uint (*WA)[WA_SIZE], const char* R) {
    typedef typename Work<C>::T T;
    struct Frame { int Score; uint tR2, tG2; uint8_t phase; };
    Frame fr[C::NWA + 1];
    int d = 0;
    fr[0].Score = 0; fr[0].tR2 = 0; fr[0].tG2 = 0; fr[0].phase = 0;
    while (d >= 0) {
        if (w.stepBudget && ++w.nSteps > w.stepBudget) { w.ov = OV_STEPS; return false; }
        Frame& F = fr[d];
        uint iA = (uint)d;
        T& trA = w.stack[d];
        if (F.phase == 0) {
            if (iA >= nA && F.tR2 == 0) { d--; continue; }
            if (iA >= nA) {
                w.fin.copyFrom(trA);
                if (!finalizeTranscriptPE(P, gv, w, F.Score, F.tR2, F.tG2, w.fin, R, WA)) return false;
                d--; continue;
            }
            int dScore = 0;
            T& trAi = w.stack[d + 1];
            trAi.copyFrom(trA);
            if (trA.nExons > 0) {
                bool ovEx = false;
                dScore = stitchAlignToTranscript<T, C::NEX>(F.tR2, F.tG2, WA[iA][WA_rStart], WA[iA][WA_gStart], WA[iA][WA_Length], WA[iA][WA_iFrag],
                                                            WA[iA][WA_sjA], P, R, gv, &trAi, w.outFilterMismatchNmaxTotal, ovEx);
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
                N.phase = 0;
                d++;
                continue;
            }
        }
        if (F.phase == 1) {
            F.phase = 2;
            if (WA[iA][WA_Anchor] != 2 || trA.nAnchor > 0) {
                T& next = w.stack[d + 1];
                next.copyFrom(trA);
                Frame& N = fr[d + 1];
                N.Score = F.Score; N.tR2 = F.tR2; N.tG2 = F.tG2; N.phase = 0;
                d++;
                continue;
            }
        }
        d--;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// Everything after the windows are filled: stitching of every window, multMapSelect, mappedFilter, encoding.
// Copy of m5::mapFinish; the only change is the call to stitchWindowAlignsPE. (mappedFilter's Lread - 1 is STAR's too:
// ReadAlign_mappedFilter.cpp:8-9 uses the combined length, so the thresholds scale with both mates.)
// ---------------------------------------------------------------------------------------------------------------------
template <class C> HD inline Status mapFinishPE(const Params& P, const GenomeView& gv, WorkPE<C>& w, ReadBuf<C::LREAD>& rb, Blob& blob) {
    typedef typename Work<C>::T T;
    const char* Read1[3] = {rb.r[0], rb.r[1], rb.r[2]};
    const uint Lread = w.Lread;
    const bool stitched = w.stitched;
    w.nSteps = 0;
    if (stitched) {
        uint iW1 = 0;
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
            if (w.trNtotal + P.alignTranscriptsPerWindowNmax >= P.alignTranscriptsPerReadNmax) break;
            for (int k = 0; k <= C::NWTR; k++) w.slotP[k] = k;
            w.slot[0].copyFrom(trA);
            w.nWinTr = 0;
            if (!stitchWindowAlignsPE(P, gv, w, w.nWA[iW], (const uint (*)[WA_SIZE])w.WA[iW], Read1[trA.roStr == 0 ? 0 : 2])) return Status::Overflow;
            if (w.nWinTr == 0) continue;
            const T& t0 = w.slot[w.slotP[0]];
            bool better = t0.maxScore > w.trBestScore || (t0.maxScore == w.trBestScore && t0.gLength < w.trBestGLength);
            int runBest = mx(w.trBestScore, t0.maxScore);
            int nKeep0 = w.nKeep;
            for (uint k = 0; k < w.nWinTr; k++) {
                const T& t = w.slot[w.slotP[k]];
                if (t.maxScore + P.outFilterMultimapScoreRange >= runBest || (better && k == 0)) {
                    if (w.nKeep >= C::NKEEP) { w.ov = OV_KEEP; return Status::Overflow; }
                    w.keep[w.nKeep].copyFrom(t);
                    w.keep[w.nKeep].Chr = t0.Chr; w.keep[w.nKeep].Str = t0.Str; w.keep[w.nKeep].roStr = t0.roStr;
                    w.keepWin[w.nKeep] = (int)iW1;
                    w.nKeep++;
                }
            }
            if (better) { w.trBestScore = t0.maxScore; w.trBestGLength = t0.gLength; w.trBestKeep = nKeep0; }
            if (better) {
                int j = 0;
                for (int k = 0; k < w.nKeep; k++) {
                    if (w.keep[k].maxScore + P.outFilterMultimapScoreRange >= w.trBestScore) {
                        if (k == w.trBestKeep) w.trBestKeep = j;
                        if (j != k) { w.keep[j].copyFrom(w.keep[k]); w.keepWin[j] = w.keepWin[k]; }
                        j++;
                    }
                }
                w.nKeep = j;
            }
            w.trNtotal += w.nWinTr;
            iW1++;
        }
        w.nW = iW1;
        if (w.trBestScore == 0) {
            w.mapMarker = MARKER_NO_GOOD_WINDOW;
            w.nW = 0;
        }
    }
    // ---- multMapSelect (ReadAlign_multMapSelect.cpp)
    w.nTr = 0;
    int iBest = -1;
    if (w.nW > 0) {
        w.maxScore = w.trBestScore;
        for (int k = 0; k < w.nKeep; k++) {
            if (w.keep[k].maxScore + P.outFilterMultimapScoreRange >= w.maxScore) {
                if (k != (int)w.nTr) { w.keep[w.nTr].copyFrom(w.keep[k]); }
                if (k == w.trBestKeep) iBest = (int)w.nTr;
                w.nTr++;
            }
        }
        if (!(w.nTr > P.outFilterMultimapNmax || w.nTr == 0)) {
            for (uint iTr = 0; iTr < w.nTr; iTr++) {
                T& t = w.keep[iTr];
                t.roStart = t.roStr == 0 ? t.rStart : Lread - t.rStart - t.rLength;
                t.cStart = t.gStart - gv.chrStart[t.Chr];
            }
            if (w.nTr == 1) w.keep[0].primaryFlag = true;
            else w.keep[iBest].primaryFlag = true;     // OneBestScore, Old_2.4 order, outSAMmultNmax -1 (checked by the driver)
        }
    }
    // ---- mappedFilter (ReadAlign_mappedFilter.cpp)
    w.unmapType = -1;
    if (w.nW == 0) {
        w.unmapType = 0;
    } else {
        const T& b = w.keep[iBest];
        if ((b.maxScore < P.outFilterScoreMin) || (b.maxScore < (int)(P.outFilterScoreMinOverLread * (Lread - 1)))
            || (b.nMatch < P.outFilterMatchNmin) || (b.nMatch < (uint)(P.outFilterMatchNminOverLread * (Lread - 1)))) {
            w.unmapType = 1;
        } else if ((b.nMM > w.outFilterMismatchNmaxTotal) || (double(b.nMM) / double(b.rLength) > P.outFilterMismatchNoverLmax)) {
            w.unmapType = 2;
        } else if (w.nTr > P.outFilterMultimapNmax) {
            w.unmapType = 3;
        }
    }
    blob.putV((i64)w.nW); blob.putV((i64)w.nTr); blob.putV((i64)(w.nW > 0 ? iBest : -1));
    if (w.nW == 0) blob.putV((i64)w.trInitRLength);
    for (uint i = 0; i < w.nTr; i++) encodeTr(blob, w.keep[i]);
    if (blob.full) { w.ov = OV_BLOB; return Status::BlobFull; }
    return Status::Ok;
}

// ---------------------------------------------------------------------------------------------------------------------
// One pair: m5::mapPhase1 on the combined read, then the paired parameters, assignment, and mapFinishPE.
// m5::mapPhase1 writes Lread-based outFilterMismatchNmaxTotal and a single maxScoreMate; nothing in phase 1 or in the assignment
// reads them, so they are set after the call (oneRead.cpp:83 and ReadAlign.cpp:125-127 resetN: maxScoreMate = 0 for each mate).
// ---------------------------------------------------------------------------------------------------------------------
template <class C> HD inline Status mapPhase1PE(const Params& P, const GenomeView& gv, WorkPE<C>& w, ReadBuf<C::LREAD>& rb,
                                              const char* R0, uint L0, uint L1) {
    Status st = m5::mapPhase1(P, gv, w, rb, R0, L0 + L1 + 1);
    w.readLength[0] = L0; w.readLength[1] = L1;
    w.outFilterMismatchNmaxTotal = mn<uint>(P.outFilterMismatchNmax, (uint)(P.outFilterMismatchNoverReadLmax * (L0 + L1)));
    w.maxScoreMateP[0] = 0; w.maxScoreMateP[1] = 0;
    return st;
}

// The dump the harness reads back: after the assignment, before stitching. Per read: nW, then per window
// Str, Chr, nWA, WlastAnchor, and WA_SIZE words per seed. Returns the words written (0 if cap too small).
template <class C> HD inline u64 dumpWindows(const WorkPE<C>& w, u64* out, u64 cap) {
    u64 n = 1;
    if (cap < 1) return 0;
    out[0] = w.stitched ? w.nW : 0;
    if (!w.stitched) return n;
    for (uint iW = 0; iW < w.nW; iW++) {
        u64 need = 4 + (u64)w.nWA[iW] * WA_SIZE;
        if (n + need > cap) return 0;
        out[n++] = w.WC[iW][WC_Str]; out[n++] = w.WC[iW][WC_Chr]; out[n++] = w.nWA[iW]; out[n++] = (u64)w.WlastAnchor[iW];
        for (uint iA = 0; iA < w.nWA[iW]; iA++) for (int k = 0; k < WA_SIZE; k++) out[n++] = w.WA[iW][iA][k];
    }
    return n;
}

#ifdef __CUDACC__
// One pair per thread: raw mates in two concatenated numeric buffers (STAR's code, N = 4), offsets in off1/off2; the clipping of
// each mate (clip1, clip2) is applied here, as readLoad.cpp does before oneRead combines the mates.
// blobs: n * blobCap bytes, blobLen[i] = bytes used. dump: n * dumpCap words (dumpLen[i] used). unmapType[i]: STAR's, 255 when
// status != Ok. clipOut: 6 words per pair: readLengthOriginal of mate 1 and 2, clipped5 and clipped3 of mate 1, of mate 2.
// A mate clipped to nothing is Status::Unsupported (host lane): STAR would combine a pair with an empty mate.
// The lane's Work<C> and ReadBuf<C::LREAD> arrays are allocated by the caller (as k_p3's in engine_gpu.cu is).
template <class C>
__global__ void k_pe_map(Params P, GenomeView g, MateClip clip1, MateClip clip2, const char* m1, const u64* off1, const char* m2,
                         const u64* off2, u64 n, WorkPE<C>* work, ReadBuf<C::LREAD>* rbuf, uint8_t* blobs, u64 blobCap, u64* blobLen,
                         u64* dump, u64 dumpCap, u64* dumpLen, uint8_t* status, uint8_t* unmapType, u64* clipOut) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    WorkPE<C>& w = work[i];
    ReadBuf<C::LREAD>& rb = rbuf[i];
    const u64 Lo1 = off1[i + 1] - off1[i], Lo2 = off2[i + 1] - off2[i];
    ClipSpan a = clipMate(clip1, m1 + off1[i], Lo1), b = clipMate(clip2, m2 + off2[i], Lo2);
    clipOut[6 * i + 0] = Lo1; clipOut[6 * i + 1] = Lo2; clipOut[6 * i + 2] = a.c5; clipOut[6 * i + 3] = a.c3;
    clipOut[6 * i + 4] = b.c5; clipOut[6 * i + 5] = b.c3;
    const uint L0 = (uint)a.len, L1 = (uint)b.len;
    char comb[C::LREAD];
    Blob blob; blob.p = blobs + i * blobCap; blob.n = 0; blob.cap = (int)blobCap; blob.full = false;
    dumpLen[i] = 0; blobLen[i] = 0; unmapType[i] = 255;
    if (L0 == 0 || L1 == 0) { status[i] = (uint8_t)Status::Unsupported; return; }
    if (L0 + L1 + 1 > (uint)C::LREAD) { status[i] = (uint8_t)Status::Overflow; return; }
    combineMates(m1 + off1[i] + a.start, L0, m2 + off2[i] + b.start, L1, comb);
    w.wbEpoch = 0;       // first use of this Work: the winBin map starts clean (as engine_gpu.cu does per thread)
    w.nScan = 0; w.stepBudget = 0;
    Status st = mapPhase1PE(P, g, w, rb, comb, L0, L1);
    if (st == Status::Ok && w.stitched && !assignAll(P, g, w)) st = Status::Overflow;
    if (st == Status::Ok) {
        dumpLen[i] = dumpWindows(w, dump + i * dumpCap, dumpCap);
        st = mapFinishPE(P, g, w, rb, blob);
    }
    status[i] = (uint8_t)st;
    if (st == Status::Ok) { blobLen[i] = (u64)blob.n; unmapType[i] = (uint8_t)w.unmapType; }
}
#endif

} // namespace pe
} // namespace m5
