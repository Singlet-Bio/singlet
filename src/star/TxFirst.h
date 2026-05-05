#ifndef TX_FIRST_H
#define TX_FIRST_H

/*
 * TxFirst: Transcriptome-first filtered SA fast path.
 *
 * A small SA_tx (~12 MB) containing only suffix array entries for the top ~500
 * most abundant genes is probed BEFORE the full 12 GB SA.  When the read maps
 * within SA_tx, the expensive full-SA binary search is skipped entirely.
 *
 * SA_tx and the genome G share the same coordinate space — SA_tx entries are a
 * strict subset of the original SA.  All downstream code (stitching, Solo,
 * transcriptome projection) works unchanged.
 *
 * SA indices stored in PC_SAstart/PC_SAend are tagged with TXFIRST_SA_FLAG
 * (bit 63) when they refer to SA_tx entries.  stitchPieces dispatches to the
 * correct array based on this flag.
 */

#include "IncludeDefine.h"
#include "PackedArray.h"

// High-bit flag to mark SA_tx indices stored in PC_SAstart/PC_SAend.
// SA indices are < 2^32, so bit 63 is always free.
#define TXFIRST_SA_FLAG (1ULL << 63)

struct TxFirstData {
    PackedArray SA_tx;       // filtered packed suffix array (same wordLength as SA)
    PackedArray SAi_tx;      // SA index for filtered SA (same bit packing as SAi)
    uint nSA_tx;             // number of entries in SA_tx
    uint nSAi_tx;            // number of entries in SAi_tx
    uint gSAindexNbases_tx;  // k-mer length for SAi_tx (typically 9)
    uint *genomeSAindexStart_tx; // L-mer index starts in SAi_tx
    bool loaded;             // true if SA_tx/SAi_tx were loaded successfully

    TxFirstData() : nSA_tx(0), nSAi_tx(0), gSAindexNbases_tx(0),
                    genomeSAindexStart_tx(nullptr), loaded(false) {}
};

// ============================================================================
// Byte-level genome comparison using SA_tx.
// Identical logic to the non-GENOME_PACKED compareSeqToGenome, but reads the
// SA entry from a caller-supplied PackedArray instead of mapGen.SA.
// ============================================================================
static inline uint compareSeqToGenome_tx(
    const char *G, uint nGenome, unsigned char GstrandBit, uint GstrandMask,
    PackedArray &sa_tx, char** s2, uint S, uint N, uint L,
    uint iSA, bool dirR, bool& compRes)
{
    uint SAstr = sa_tx[iSA];
    bool dirG = (SAstr >> GstrandBit) == 0;
    SAstr &= GstrandMask;

    int64 ii;

    if (dirR && dirG) {
        char* s = s2[0] + S + L;
        const char* g = G + SAstr + L;
        for (ii = 0; (uint)ii < N - L; ii++) {
            if (s[ii] != g[ii]) {
                compRes = (s[ii] > g[ii]);
                return ii + L;
            }
        }
        return N;
    } else if (dirR && !dirG) {
        char* s = s2[1] + S + L;
        const char* g = G + nGenome - 1 - SAstr - L;
        for (ii = 0; (uint)ii < N - L; ii++) {
            if (s[ii] != g[-ii]) {
                if (s[ii] > g[-ii] || g[-ii] > 3) {
                    compRes = false;
                } else {
                    compRes = true;
                }
                return ii + L;
            }
        }
        return N;
    } else if (!dirR && dirG) {
        char* s = s2[1] + S - L;
        const char* g = G + SAstr + L;
        for (ii = 0; (uint)ii < N - L; ii++) {
            if (s[-ii] != g[ii]) {
                compRes = (s[-ii] > g[ii]);
                return ii + L;
            }
        }
        return N;
    } else { // !dirR && !dirG
        char* s = s2[0] + S - L;
        const char* g = G + nGenome - 1 - SAstr - L;
        for (ii = 0; (uint)ii < N - L; ii++) {
            if (s[-ii] != g[-ii]) {
                if (s[-ii] > g[-ii] || g[-ii] > 3) {
                    compRes = false;
                } else {
                    compRes = true;
                }
                return ii + L;
            }
        }
        return N;
    }
}

// ============================================================================
// Simple binary search in SA_tx.  No prefetch tricks needed — SA_tx fits
// entirely in L3 cache.  Returns Nrep (number of alignments at best length).
// ============================================================================
static inline uint maxMappableLength_tx(
    const char *G, uint nGenome, unsigned char GstrandBit, uint GstrandMask,
    PackedArray &sa_tx, uint nSA_tx, char** s, uint S, uint N,
    uint i1, uint i2, bool dirR, uint& L, uint* indStartEnd)
{
    bool compRes;

    uint L1 = compareSeqToGenome_tx(G, nGenome, GstrandBit, GstrandMask,
                                     sa_tx, s, S, N, L, i1, dirR, compRes);
    uint L2 = compareSeqToGenome_tx(G, nGenome, GstrandBit, GstrandMask,
                                     sa_tx, s, S, N, L, i2, dirR, compRes);
    L = min(L1, L2);

    uint i3, L3;
    uint L1a = L1, i1a = i1;
    uint L2a = L2, i2a = i2;
    (void)L1a; (void)i1a; (void)L2a; (void)i2a; // suppress unused warnings
    i3 = i1; L3 = L1;

    while (i1 + 1 < i2) {
        i3 = (i1 + i2) / 2;
        L3 = compareSeqToGenome_tx(G, nGenome, GstrandBit, GstrandMask,
                                    sa_tx, s, S, N, L, i3, dirR, compRes);
        if (L3 == N) break;

        if (compRes) {
            i1 = i3; L1 = L3;
        } else {
            i2 = i3; L2 = L3;
        }
        L = min(L1, L2);
    }

    if (L3 < N) {
        if (L1 > L2) { i3 = i1; L3 = L1; }
        else          { i3 = i2; L3 = L2; }
    }

    // findMultRange equivalent: scan outward from i3 to find all entries at length L3.
    // SA_tx ranges are tiny, so linear scan is fine.
    uint lo = i3, hi = i3;
    while (lo > 0) {
        uint Ltest = compareSeqToGenome_tx(G, nGenome, GstrandBit, GstrandMask,
                                            sa_tx, s, S, L3, 0, lo - 1, dirR, compRes);
        if (Ltest < L3) break;
        lo--;
    }
    while (hi + 1 < nSA_tx) {
        uint Ltest = compareSeqToGenome_tx(G, nGenome, GstrandBit, GstrandMask,
                                            sa_tx, s, S, L3, 0, hi + 1, dirR, compRes);
        if (Ltest < L3) break;
        hi++;
    }

    L = L3;
    indStartEnd[0] = lo;
    indStartEnd[1] = hi;
    return hi - lo + 1;
}

// ============================================================================
// Inline helper to read an SA entry, dispatching to SA_tx or SA based on flag.
// ============================================================================
static inline uint readSAentry(PackedArray &SA, PackedArray &SA_tx, uint iSA) {
    if (iSA & TXFIRST_SA_FLAG) {
        return SA_tx[iSA & ~TXFIRST_SA_FLAG];
    }
    return SA[iSA];
}

#endif // TX_FIRST_H
