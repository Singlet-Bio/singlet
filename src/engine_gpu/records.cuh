// E6: the alignment records of the GPU pileup, built from the GPU mapping results (STAR's sequence-cache blobs), on the device or
// the host from one source. docs/gpu-engine/fable/E6-records.md.
//
// Hop 6 (per distinct sequence): the blob that stitch.cuh's mapFinish/encodeTr writes (= ReadAlign::seqCacheEncode) is decoded the way
// ReadAlign::seqCacheRestore decodes it, mappedFilter decides whether the sequence is mapped, and every output transcript becomes the BAM
// core fields that ReadAlign::seqCacheLeanAln computes for a single-end read (= alignBAM's, verified on 79 M alignments: host-cost-model.md
// F2), with the same expressions and the same integer widths. The result is written in two forms:
//   - the pileup's native arrays (src/engine_gpu/gpu_pileup/pileup/pileup_rec.h: SigRec, AlnRec, CIGAR ops len << 4 | op, nt16 sequence),
//     as Collector::prepend_handoff (include/singlet/pileup/gpu_pileup_bridge.h) builds them from the hand-off, so the pileup parses nothing;
//   - optionally the hand-off record bytes of ReadAlign::seqCacheAlignRecords (SGHOFF02 data), for the byte-for-byte check.
// Hop 7 (per read): the pileup's ReadRec (record, barcode column, packed UMI) under the rule of PileupEngine::gpu_handoff_pass_entries,
// given per read its record (mapDistinct's readRecord) and STAR's per-read entry (CR, UR; HandoffReadEntry) and a table that maps each
// distinct CR to its barcode column (the host resolves the distinct CRs: R11 whitelist correction, barcode universe).
//
// What the records assume, as seqCacheAlignRecords does for the hand-off: single-end, the record's sequence is the read as mapped (clipped:
// clipMates' clippedN = 0, so trimL = 0), readLength[0] = readLengthOriginal[0] = L, readLength[1] = 0, readFilter 'N'. A transcript with a mate
// gap (canonSJ -3, paired reads only) is where seqCacheLeanAln returns false and the host calls alignBAM: it is counted (status bit 1) and
// never occurs for single-end mapping.
#pragma once
#include <cstdint>
#include "gpu_pileup/pileup/pileup_rec.h"

#ifdef __CUDACC__
#define E6HD __host__ __device__
#else
#define E6HD
#endif

namespace e6 {

typedef uint64_t u64;   // = the API's uint64_t
typedef int64_t i64;

// STAR's Parameters that the records depend on (Parameters.h types: intScore = int, uint = 64-bit)
struct RecParams {
    int outFilterScoreMin;              // mappedFilter
    double outFilterScoreMinOverLread;
    u64 outFilterMatchNmin;
    double outFilterMatchNminOverLread;
    u64 outFilterMismatchNmax;          // outFilterMismatchNmaxTotal, as oneRead / seqCacheAlignRecords set it
    double outFilterMismatchNoverLmax;
    double outFilterMismatchNoverReadLmax;
    u64 outFilterMultimapNmax;
    u64 outSAMmultNmax;                 // alignments per record: min(outSAMmultNmax, nTr)
    int outSAMmapqUnique;               // alignBAM's MAPQ for nTr == 1 (default 255)
    uint16_t outSAMflagAND, outSAMflagOR;   // defaults 65535, 0
};

// per record: what the count pass found (status bits: 1 mate gap: seqCacheLeanAln would fall back to alignBAM; 2 a transcript without
// exons; 4 iBest outside [0, nTr) with nW > 0 (trBest = trInit, as seqCacheRestore does))
enum : uint8_t { ST_MATEGAP = 1, ST_NOEXON = 2, ST_BADBEST = 4 };

// ReadAlign_seqCache.cpp getV: zigzag LEB128
E6HD inline i64 getV(const uint8_t*& p) {
    u64 u = 0; int sh = 0;
    while (*p & 0x80) { u |= (u64)(*p++ & 0x7f) << sh; sh += 7; }
    u |= (u64)(*p++) << sh;
    return (i64)((u >> 1) ^ (~(u & 1) + 1));
}
E6HD inline void skipV(const uint8_t*& p) { while (*p++ & 0x80) {} }

// the transcript fields the records read (forEachScalar order: nExons, then 29 scalars; Get casts each to the field's type)
struct TrHead { u64 nExons, rLength, Chr, Str, nMatch, nMM; int maxScore; bool primaryFlag; };
E6HD inline void parseHead(const uint8_t*& p, TrHead& t) {
    t.nExons = (u64)getV(p);
    skipV(p); skipV(p);                       // rStart, roStart
    t.rLength = (u64)getV(p);
    skipV(p); skipV(p); skipV(p);             // gStart, gLength, cStart
    t.Chr = (u64)getV(p); t.Str = (u64)getV(p);
    skipV(p);                                 // roStr
    t.nMatch = (u64)getV(p); t.nMM = (u64)getV(p);
    for (int k = 11; k < 24; k++) skipV(p);   // mappedLength, extendL, nGap, lGap, nDel, nIns, lDel, lIns, nUnique, nAnchor, intronMotifs[3]
    t.maxScore = (int)getV(p);
    skipV(p); skipV(p); skipV(p);             // iFrag, sjMotifStrand, sjYes
    t.primaryFlag = getV(p) != 0;
}
// skips the exon rows of a transcript (forEachRow: 5 fields; then, except after the last exon, canonSJ sjAnnot sjStr [shiftSJ x2 if canonSJ >= 0])
E6HD inline void skipRows(const uint8_t*& p, u64 nExons) {
    for (u64 iex = 0; iex < nExons; iex++) {
        for (int k = 0; k < 5; k++) skipV(p);
        if (iex + 1 < nExons) {
            const int c = (int)getV(p); skipV(p); skipV(p);
            if (c >= 0) { skipV(p); skipV(p); }
        }
    }
}

// seqCacheLeanAln's CIGAR from the exon rows (p at the first row; p ends after the last), one op at a time through sink(u32): the same
// expressions in STAR's uint (64-bit) and the store into the uint32 array. Also returns exons[0][EX_G], exons[0][EX_iFrag] and status bits.
template <class Sink> E6HD inline uint8_t cigarOps(const uint8_t*& p, u64 nExons, u64 L, u64& g0, u64& mate, Sink& sink) {
    const u64 OPSH = 4, M = 0, I = 1, D = 2, N = 3, S = 4;
    uint8_t st = 0;
    if (nExons == 0) { g0 = 0; mate = 0; return ST_NOEXON; }   // seqCacheLeanAln reads exons[nExons-1]: never happens for a mapped transcript
    const u64 readLength0 = L, trimL = 0;
    u64 r0 = 0, gPrev = 0, rPrev = 0, lPrev = 0; int canonPrev = 0; uint8_t annotPrev = 0;
    for (u64 ii = 0; ii < nExons; ii++) {
        const u64 R = (u64)getV(p), G = (u64)getV(p), Lx = (u64)getV(p), iFrag = (u64)getV(p);
        skipV(p);   // EX_sjA
        if (ii == 0) {
            g0 = G; mate = iFrag; r0 = R;
            const u64 trimL1 = trimL + R - (R < readLength0 ? 0 : readLength0 + 1);
            if (trimL1 > 0) sink((uint32_t)(trimL1 << OPSH | S));
        } else {
            const u64 gapG = G - (gPrev + lPrev);
            const u64 gapR = R - rPrev - lPrev;
            if (gapR > 0) sink((uint32_t)(gapR << OPSH | I));
            if (canonPrev >= 0 || annotPrev == 1) sink((uint32_t)(gapG << OPSH | N));
            else if (gapG > 0) sink((uint32_t)(gapG << OPSH | D));
        }
        if (Lx > 0) sink((uint32_t)(Lx << OPSH | M));
        if (ii + 1 < nExons) {
            canonPrev = (int)getV(p); annotPrev = (uint8_t)getV(p); skipV(p);
            if (canonPrev >= 0) { skipV(p); skipV(p); }
            if (canonPrev == -3) st |= ST_MATEGAP;
        }
        gPrev = G; rPrev = R; lPrev = Lx;
    }
    // readLengthOriginal = {L, 0}; Mate selects it only when the first exon lies in mate 2 (never for single-end)
    const u64 rlo = mate == 0 ? L : 0;
    const u64 trimR1 = (r0 < readLength0 ? L : readLength0 + 1 + rlo) - rPrev - lPrev - trimL;
    if (trimR1 > 0) sink((uint32_t)(trimR1 << OPSH | S));
    return st;
}
struct CountSink { uint32_t n = 0; E6HD void operator()(uint32_t) { n++; } };

// alignBAM's flag and MAPQ for a single-end alignment (readFilter 'N')
E6HD inline uint16_t samFlag(const RecParams& P, const TrHead& t, u64 mate) {
    uint16_t f = 0;
    if (!t.primaryFlag) f |= 0x100;
    if (mate == 0) f |= (uint16_t)(t.Str * 0x10); else f |= (uint16_t)((1 - t.Str) * 0x10);
    return (uint16_t)((f & P.outSAMflagAND) | P.outSAMflagOR);
}
E6HD inline uint8_t mapq(const RecParams& P, u64 nTr) {
    int q = P.outSAMmapqUnique;
    if (nTr >= 5) q = 0; else if (nTr >= 3) q = 1; else if (nTr == 2) q = 3;
    return (uint8_t)(q & 0xff);
}

// ---- hop 6, count pass: the record's alignment count, CIGAR ops, sequence bytes and hand-off record bytes
struct RecCount { uint32_t nAln, nOps; uint8_t status; int unmapType; };
E6HD inline RecCount recordCount(const RecParams& P, const uint8_t* blob, u64 L) {
    RecCount rc{0, 0, 0, -1};
    const uint8_t* p = blob;
    const u64 nW = (u64)getV(p), nTr = (u64)getV(p);
    const i64 iBest = getV(p);
    u64 trInitRLength = 0;
    if (nW == 0) trInitRLength = (u64)getV(p);
    const u64 nOut = nTr < P.outSAMmultNmax ? nTr : P.outSAMmultNmax;
    // trBest: trMult[iBest], or trInit (reset: maxScore, nMatch, nMM 0)
    int bScore = 0; u64 bMatch = 0, bMM = 0, bRLength = trInitRLength;
    if (nW > 0 && !(iBest >= 0 && (u64)iBest < nTr)) rc.status |= ST_BADBEST;
    uint32_t ops = 0;
    for (u64 i = 0; i < nTr; i++) {
        TrHead t; parseHead(p, t);
        if ((i64)i == iBest) { bScore = t.maxScore; bMatch = t.nMatch; bMM = t.nMM; bRLength = t.rLength; }
        if (i < nOut) { CountSink c; u64 g0, mate; rc.status |= cigarOps(p, t.nExons, L, g0, mate, c); ops += c.n; }
        else skipRows(p, t.nExons);
    }
    // mappedFilter (Lread = L; outFilterMismatchNmaxTotal as seqCacheAlignRecords sets it)
    const u64 Lread = L;
    const u64 mmCap = (u64)(P.outFilterMismatchNoverReadLmax * (double)(L + 0));
    const u64 mmTotal = P.outFilterMismatchNmax < mmCap ? P.outFilterMismatchNmax : mmCap;
    if (nW == 0) rc.unmapType = 0;
    else if ((bScore < P.outFilterScoreMin) || (bScore < (int)(P.outFilterScoreMinOverLread * (double)(Lread - 1)))
             || (bMatch < P.outFilterMatchNmin) || (bMatch < (u64)(P.outFilterMatchNminOverLread * (double)(Lread - 1)))) rc.unmapType = 1;
    else if ((bMM > mmTotal) || (double(bMM) / double(bRLength) > P.outFilterMismatchNoverLmax)) rc.unmapType = 2;
    else if (nTr > P.outFilterMultimapNmax) rc.unmapType = 3;
    if (rc.unmapType < 0) { rc.nAln = (uint32_t)nOut; rc.nOps = ops; }
    else rc.status &= (uint8_t)~ST_MATEGAP;   // no alignment is written, so no fallback either
    return rc;
}
E6HD inline u64 seqBytes(u64 L) { return (L + 1) / 2; }
E6HD inline u64 recBytes(uint32_t nAln, uint32_t nOps, u64 L) { return 1 + 18 * (u64)nAln + 4 * (u64)nOps + 2 + seqBytes(L); }

E6HD inline void put16(uint8_t* d, uint16_t v) { d[0] = (uint8_t)v; d[1] = (uint8_t)(v >> 8); }
E6HD inline void put32(uint8_t* d, uint32_t v) { d[0] = (uint8_t)v; d[1] = (uint8_t)(v >> 8); d[2] = (uint8_t)(v >> 16); d[3] = (uint8_t)(v >> 24); }

// writes each op to the pileup's ops array and (if rec) after the alignment's 18 header bytes of the hand-off record
struct FillSink {
    uint32_t* ops; uint8_t* rec; uint32_t n;
    E6HD void operator()(uint32_t v) { ops[n] = v; if (rec) put32(rec + 4 * (u64)n, v); n++; }
};

// ---- hop 6, fill pass: nAln alignments (from the count pass) -> AlnRec[nAln], ops, SigRec, nt16 sequence; rec (optional) = the record bytes
// seqCacheAlignRecords appends for this sequence
E6HD inline void recordFill(const RecParams& P, const uint8_t* blob, const char* seq, u64 L, const u64* chrStart, uint32_t nAln,
                            uint32_t alnOff, uint32_t opsOff, u64 seqOff, SigRec* sig, AlnRec* alns, uint32_t* ops, uint8_t* seq2, uint8_t* rec) {
    const uint8_t* p = blob;
    const u64 nW = (u64)getV(p), nTr = (u64)getV(p);
    getV(p);
    if (nW == 0) getV(p);
    SigRec s; s.aln_off = alnOff; s.n_aln = (uint8_t)nAln; s.nh = 1; s.rlen = (uint16_t)L; s.seq_off = seqOff;
    uint8_t* d = rec;
    if (d) *d++ = (uint8_t)nAln;
    uint32_t o = opsOff;
    for (u64 i = 0; i < (u64)nAln; i++) {
        TrHead t; parseHead(p, t);
        FillSink fs{ops + o, d ? d + 18 : nullptr, 0};
        u64 g0, mate;
        cigarOps(p, t.nExons, L, g0, mate, fs);
        const int32_t refID = (int32_t)(uint32_t)t.Chr;
        const int32_t pos = (int32_t)(uint32_t)(g0 - chrStart[t.Chr]);
        const uint16_t flag = samFlag(P, t, mate);
        const uint8_t q = mapq(P, nTr);
        const uint32_t nh = (uint32_t)nTr, nm = (uint32_t)t.nMM;
        const uint16_t nh16 = (uint16_t)(nh < 65535u ? nh : 65535u), nm16 = (uint16_t)(nm < 65535u ? nm : 65535u), nc16 = (uint16_t)fs.n;
        if (i == 0) { const uint32_t a = nh16 < 1 ? 1 : nh16; s.nh = (uint8_t)(a < 255 ? a : 255); }
        AlnRec a; a.tid = refID; a.pos = pos; a.ops_off = o; a.n_ops = nc16; a.mapq = q;
        a.flags = (uint8_t)(((flag & 0x10) ? 1 : 0) | ((flag & 0x100) ? 0 : 2));
        alns[i] = a;
        if (d) {
            put32(d, (uint32_t)refID); put32(d + 4, (uint32_t)pos); put16(d + 8, flag); d[10] = q; d[11] = 0;
            put16(d + 12, nh16); put16(d + 14, nm16); put16(d + 16, nc16);
            d += 18 + 4 * (u64)fs.n;
        }
        o += fs.n;
    }
    *sig = s;
    // the sequence: BAM nt16 codes, two per byte, low nibble first, read orientation
    const uint8_t NT16[5] = {1, 2, 4, 8, 15};
    if (d) { put16(d, (uint16_t)L); d += 2; }
    for (u64 ii = 0; ii < L; ii += 2) {
        const uint8_t c0 = (uint8_t)seq[ii], lo = NT16[c0 < 4 ? c0 : 4];
        uint8_t hi = 0;
        if (ii + 1 < L) { const uint8_t c1 = (uint8_t)seq[ii + 1]; hi = NT16[c1 < 4 ? c1 : 4]; }
        const uint8_t b = (uint8_t)(lo | (hi << 4));
        seq2[ii / 2] = b;
        if (d) *d++ = b;
    }
}

// ---- hop 7: one read
// STAR's per-read hand-off entry (SeqCache.h HandoffReadEntry = gpu_pileup_bridge.h ReadEntry): CR and UR 3 bits per base (A0 C1 G2 T3,
// other 4; base 0 lowest), base-quality counters, flags (bit 0: written; bits 8-31 clip lengths and the 3' tail when clipped; 32: clip > 255)
struct ReadEntry { u64 cr3, ur3; uint16_t lowQual, qualNeqN; uint32_t flags; };
static_assert(sizeof(ReadEntry) == 24, "HandoffReadEntry layout");

// the outcome of gpu_handoff_pass_entries for a read (in the order its checks run)
enum ReadOutcome : uint8_t { RO_STREAM = 0, RO_UNMAPPED, RO_NOPRIMARY, RO_UNWRITTEN, RO_BIGCLIP, RO_NOBARCODE, RO_NOTINSET, RO_KEEP, RO_N };
E6HD inline u64 mask3(uint32_t n) { return n >= 21 ? (n >= 22 ? ~0ULL : ((1ULL << 63) - 1)) : ((1ULL << (3 * n)) - 1); }
// the CR table key: CR as the entry packs it, cbL bases
E6HD inline u64 crKey(const ReadEntry& e, uint32_t cbL) { return e.cr3 & mask3(cbL); }
// the CR table: sorted keys, per key its column (>= 0), -1 (not in the barcode set) or -2 (no barcode: CR rejected)
E6HD inline int32_t crLookup(const u64* keys, const int32_t* vals, u64 n, u64 k) {
    u64 lo = 0, hi = n;
    while (lo < hi) { const u64 m = (lo + hi) >> 1; if (keys[m] < k) lo = m + 1; else hi = m; }
    return (lo < n && keys[lo] == k) ? vals[lo] : -1;
}
E6HD inline ReadOutcome readEval(uint32_t rec, const SigRec* sigs, const AlnRec* alns, const ReadEntry& e, uint32_t cbL, uint32_t umiL,
                                 const u64* keys, const int32_t* vals, u64 nKeys, int32_t& bc, u64& u3) {
    if (rec == 0xFFFFFFFFu) return RO_STREAM;
    const SigRec s = sigs[rec];
    if (s.n_aln == 0) return RO_UNMAPPED;
    bool prim = false;
    for (uint32_t k = 0; k < s.n_aln; k++) if (alns[s.aln_off + k].flags & 2) { prim = true; break; }
    if (!prim) return RO_NOPRIMARY;
    if (!(e.flags & 1)) return RO_UNWRITTEN;
    const uint32_t clip = (e.flags & 0xFFFF00u) ? e.flags : 0u;
    if (clip & 32u) return RO_BIGCLIP;
    // UR: unpacked (codes 4..7 -> N) and checked as the pass checks the string: non-empty, no N, not a homopolymer
    bool ok = cbL > 0 && umiL > 0;
    bool homo = true; uint32_t c0 = 0; u64 u = 0;
    for (uint32_t j = 0; j < umiL; j++) {
        uint32_t c = (uint32_t)((e.ur3 >> (3 * j)) & 7); if (c > 4) c = 4;
        if (j == 0) c0 = c;
        if (c == 4) ok = false;
        if (c != c0) homo = false;
        if (j < 21) u |= (u64)c << (3 * j);
    }
    if (homo) ok = false;
    const int32_t col = ok ? crLookup(keys, vals, nKeys, crKey(e, cbL)) : -2;
    if (!ok || col == -2) return RO_NOBARCODE;
    if (col < 0) return RO_NOTINSET;
    bc = col; u3 = u;
    return RO_KEEP;
}

} // namespace e6
