// SPDX-License-Identifier: MIT
// include/singlet/pileup/tx_compat.h
//
// intron_rule::kTxCompat: STARsolo Gene's transcript concordance (src/star/Transcriptome_classifyAlign.cpp, alignToTranscript), shared by
// the CPU pileup (pileup_engine.h) and the GPU pileup (k_annotate), so both decide the same way.
//
// An alignment is concordant with a transcript when the transcript contains it (transcript start <= alignment start, alignment end <=
// transcript end), every aligned segment lies inside one exon, and every splice junction (CIGAR N) joins the end of one exon to the start
// of the next exon of that transcript exactly. Deletions do not split a segment (STAR checks the block after a deletion against the same
// exon). STARsolo Gene counts a read only when its concordant transcripts name one gene.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define SINGLET_TX_HD __host__ __device__ __forceinline__
#else
#define SINGLET_TX_HD inline
#endif

namespace singlet_tx {

constexpr int kMaxSegments = 16;    // more segments (junctions) than this: no decision, the read stays as the other rules leave it
constexpr int kMaxCandidates = 8;   // more candidate genes than this: no decision

// Segments of an alignment from its CIGAR: op codes as BAM (0 M, 1 I, 2 D, 3 N, 4 S, 5 H, 6 P, 7 =, 8 X), lengths, start position (0-based).
// Writes half-open [ss, se) per segment; returns the count, or -1 when there are more than kMaxSegments or no aligned base.
template <class OpAt>
SINGLET_TX_HD int segments(int n_ops, int32_t pos, OpAt op_at, int32_t* ss, int32_t* se) {
    int n = 0; bool open = false; int32_t rpos = pos;
    for (int i = 0; i < n_ops; i++) {
        int op; int32_t len; op_at(i, op, len);
        if (op == 0 || op == 7 || op == 8) {
            if (!open) { if (n == kMaxSegments) return -1; ss[n] = rpos; se[n] = rpos + len; n++; open = true; }
            else se[n - 1] = rpos + len;
            rpos += len;
        } else if (op == 2) {
            rpos += len; if (open) se[n - 1] = rpos;   // a deletion stays inside the segment
        } else if (op == 3) {
            rpos += len; open = false;                 // a junction ends the segment
        }
    }
    return n > 0 ? n : -1;
}

// one transcript: exons [es[k], ee[k]) sorted by start
SINGLET_TX_HD bool concordant(const int32_t* es, const int32_t* ee, uint32_t n, const int32_t* ss, const int32_t* se, int nseg) {
    if (n == 0 || nseg <= 0) return false;
    if (ss[0] < es[0] || se[nseg - 1] > ee[n - 1]) return false;
    uint32_t lo = 0, hi = n;   // the last exon starting at or before the first segment
    while (hi - lo > 1) { const uint32_t mid = lo + (hi - lo) / 2; if (es[mid] <= ss[0]) lo = mid; else hi = mid; }
    uint32_t j = lo;
    for (int k = 0; k < nseg; k++, j++) {
        if (j >= n || ss[k] < es[j] || se[k] > ee[j]) return false;
        if (k + 1 < nseg && (se[k] != ee[j] || j + 1 >= n || ss[k + 1] != es[j + 1])) return false;
    }
    return true;
}

// The candidate genes with a concordant transcript: gene g's transcripts are gtoff[g] .. gtoff[g + 1], transcript t's exons xoff[t] ..
// xoff[t + 1] of es / ee. Returns how many candidates have one (stops counting at 2) and the first such gene in `winner`.
SINGLET_TX_HD int pick(const uint32_t* cand, int ncand, const uint32_t* gtoff, const uint32_t* xoff, const int32_t* es, const int32_t* ee,
                       uint32_t ngenes, const int32_t* ss, const int32_t* se, int nseg, uint32_t& winner) {
    int found = 0;
    for (int c = 0; c < ncand; c++) {
        const uint32_t g = cand[c];
        if (g >= ngenes) continue;
        for (uint32_t t = gtoff[g]; t < gtoff[g + 1]; t++) {
            const uint32_t x0 = xoff[t], x1 = xoff[t + 1];
            if (concordant(es + x0, ee + x0, x1 - x0, ss, se, nseg)) {
                if (found == 0) winner = g;
                if (++found >= 2) return found;
                break;
            }
        }
    }
    return found;
}

}  // namespace singlet_tx
