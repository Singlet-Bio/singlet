// singlet GPU engine: barcode layer (align-and-count-design.md section 4.1), phases 1c and 1f.
//
// *** NOT COMPILED. Draft for the W11 session: written on a laptop without nvcc, never built, never run. ***
// Matches src/engine_gpu/barcode_layer_ref.py (28 passing tests in tests/python/test_barcode_layer_ref.py); every
// rule below names the STAR source line the reference mirrors. Rules, quirks and the gate plan:
// docs/research-beta/barcode-layer-prep.md. Not wired into engine_gpu.cu or pileup.cu.
//
// Build notes for whoever compiles it:
//  - compile with -fmad=false (or keep the __fmul_rn/__fadd_rn calls): the posterior uses float arithmetic that must
//    round as STAR's x86 code does (float pin, float ptot, float pmax, float cbMinP);
//  - nothing here uses atomics that leak into results: the exact-match histogram is an integer sum, everything else is
//    per read; cell calling and sorting stay in the existing reducers.
//
// Flow, phase 1c, per sample (reads already on the device from the hash/dedup stage or from the .1fq decode):
//   k_bl_extract_simple / k_bl_extract_complex   one thread per read   -> BlRead (cb value, N position, umi, status)
//   k_bl_exact_hist                                                    -> hist[wl]   (reads with cbMatch 0 and a good UMI)
//   k_bl_add_pseudocount                                               -> hist[wl] += 1   (1MM_multi_*_pseudocounts types)
//   k_bl_resolve_simple                                                -> BlRecord.cell_key (1MM / N / posterior)
// Complex Exact and the no-whitelist path need no histogram and no second pass.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace bl {

typedef uint8_t u8; typedef int32_t i32; typedef uint32_t u32; typedef int64_t i64; typedef uint64_t u64;

// STAR's cbMatch codes (SoloReadBarcode_getCBandUMI.cpp)
enum : i32 { OK_EXACT = 0, OK_ONE = 1, NO_WL = -1, N_IN_CB = -2, TOO_MANY = -3, NO_CB = -11, MM_SEGS = -12,
             NO_ADAPTER = -21, NO_UMI = -22, N_IN_UMI = -23, UMI_HOMO = -24, POST_REJECT = -4 /* posterior, not a STAR code */ };

// ---- layouts (host fills, passed by value to kernels) --------------------------------------------------------------
struct SimpleDev {            // CB_UMI_Simple and CB_samTagOut; starts are 0-based here (STAR's soloCBstart - 1)
    u32 cb_start, cb_len, umi_start, umi_len;
    u32 mm1, mm1_multi, mm1_multi_nbase, one_exact;   // MatchFlags
    u32 sam_tag_out;                                  // no UMI check (getCBandUMI :305-322)
    u32 has_wl;                                       // 0: soloCBwhitelist None, any N-free CB is accepted (:18-27)
    const u64* wl; u64 wl_n;                          // sorted unique 2-bit barcodes
};

struct SegDev {               // one CB_UMI_Complex segment, or the UMI (SoloBarcode)
    i32 anchor_type[2];       // 0 read start, 1 read end, 2 adapter start, 3 adapter end
    i32 anchor_dist[2];
    u32 min_len, n_len;       // lengths 1..n_len-1 are valid table entries (wl.size() in STAR)
    const u64* wl;            // concatenated per-length sorted lists
    const u32* wl_off;        // [len] start of that length's list in wl, wl_off[len+1]-wl_off[len] entries
    const u32* wl_add;        // STAR wlAdd[len]: entries in all shorter lengths
    u64 wl_factor;            // product of earlier segments' totalSize
};

struct ComplexDev {
    SegDev seg[4]; u32 n_seg; SegDev umi;
    const char* adapter; u32 adapter_len; u32 adapter_mm_max;    // adapter_len 0: soloAdapterSequence "-"
    u32 mm1;                                                     // 0: Exact (singlet's setting); 1MM is not drafted
};

// ---- per-read intermediate and the normalized record ---------------------------------------------------------------
struct BlRead {
    u64 cb_val; i32 pos_n;     // 2-bit CB value (N as A) and N position (-1 none, -2 several)
    u64 umi_key;               // raw UMI, 3 bits per base (A C G T N other): the R9 molecule key
    i32 status;                // cbMatch so far (umi_check, N, ...), 0 when the CB still has to be matched
    i32 cell;                  // exact whitelist index when found, else -1
    u32 umi_len;
};

struct BlRecord {              // the normalized read record of section 4.1 (mates and clips live with the sequence stage)
    u32 read_id;
    i64 cell_key;              // -1: dropped. WL protocols: whitelist index (Complex: mixed radix). No WL: discovered rank
    u64 umi_key; u32 umi_len;
    i32 status;                // STAR cbMatch code, or POST_REJECT
    i32 feature_id;            // feature_read only (tag / guide id, -1 none, -2 ambiguous)
    u8 feature_read;
    // clip spans: the existing ClipDev / k_clip in engine_gpu.cu (cDNA mate: clip5pNbases N 0, then the protocol's
    // 3' adapter, e.g. the 10x 5' TSO). The barcode mate is never clipped.
};

// ---- device helpers ------------------------------------------------------------------------------------------------
__device__ __forceinline__ int nt_code(char c) {      // SequenceFuns.cpp:206 convertNt01234; > 3 is N
    switch (c) { case 'A': case 'a': return 0; case 'C': case 'c': return 1; case 'G': case 'g': return 2;
                 case 'T': case 't': return 3; default: return 4; }
}
__device__ __forceinline__ int nt3(char c) {          // 3-bit umi_key code
    int k = nt_code(c); if (k < 4) return k; return (c == 'N' || c == 'n') ? 4 : 5;
}

// convertNuclStrToInt64 (SequenceFuns.cpp:260-276): the value, with the position of a single N, -2 for several
__device__ __forceinline__ u64 conv(const char* s, u32 n, i32& pos_n) {
    u64 v = 0; pos_n = -1;
    for (u32 i = 0; i < n; i++) {
        u64 nt = (u64)nt_code(s[i]);
        if (nt > 3) { if (pos_n >= 0) { pos_n = -2; return v; } pos_n = (i32)i; nt = 0; }
        v = (v << 2) + nt;
    }
    return v;
}

__device__ __forceinline__ i64 bsearch_u64(const u64* a, u64 n, u64 v) {   // index in the sorted unique list, or -1
    u64 lo = 0, hi = n;
    while (lo < hi) { u64 mid = (lo + hi) >> 1; if (a[mid] < v) lo = mid + 1; else hi = mid; }
    return (lo < n && a[lo] == v) ? (i64)lo : -1;
}

// convertCheckUMI (:127-138). homo_len is the length SoloReadBarcode saw at construction: umiL for Simple/samTagOut, 0
// for Complex (so only an all-A UMI is a "homopolymer" there; see the reference's check_umi).
__device__ __forceinline__ i32 check_umi(const char* s, u32 n, u32 homo_len, u64& umi_b) {
    i32 pn; umi_b = conv(s, n, pn);
    if (pn != -1) return N_IN_UMI;
    for (u32 j = 0; j < 4; j++) {
        u64 h = 0; for (u32 k = 0; k < homo_len; k++) h = (h << 2) + j;
        if (umi_b == h) return UMI_HOMO;
    }
    return 0;
}

__device__ __forceinline__ u64 pack3(const char* s, u32 n) {
    u64 v = 0; for (u32 i = 0; i < n; i++) v = (v << 3) | (u64)nt3(s[i]);    // n <= 21; longer UMIs go to the host lane
    return v;
}

// ---- CB_UMI_Simple / CB_samTagOut: extraction ---------------------------------------------------------------------
// seq/qual: the barcode mate of every read, offsets off[i]..off[i+1]. A read shorter than CB+UMI behaves as if padded
// with N (quality 'H') when --soloBarcodeReadLength is 0 (:231-234), which is how singlet runs STAR.
__global__ void k_bl_extract_simple(const char* seq, const u64* off, u64 n, SimpleDev L, BlRead* out) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= n) return;
    const char* s = seq + off[i]; u32 len = (u32)(off[i + 1] - off[i]);
    char cb[32], um[24];
    for (u32 k = 0; k < L.cb_len; k++)  { u32 p = L.cb_start + k;  cb[k] = p < len ? s[p] : 'N'; }
    for (u32 k = 0; k < L.umi_len; k++) { u32 p = L.umi_start + k; um[k] = p < len ? s[p] : 'N'; }
    BlRead r; r.umi_len = L.umi_len; r.umi_key = pack3(um, L.umi_len); r.status = 0; r.cell = -1;
    r.cb_val = conv(cb, L.cb_len, r.pos_n);
    if (!L.sam_tag_out) {
        u64 ub; i32 uc = check_umi(um, L.umi_len, L.umi_len, ub);
        if (uc != 0) r.status = uc;                            // :291-303: the CB is not matched, the read is dropped
    }
    if (r.status == 0) {
        if (!L.has_wl) { r.status = (r.pos_n != -1) ? N_IN_CB : OK_EXACT; }     // :18-27; the cell is the discovered rank
        else if (r.pos_n == -2) r.status = N_IN_CB;
        else if (r.pos_n == -1) {
            i64 w = bsearch_u64(L.wl, L.wl_n, r.cb_val);
            if (w >= 0) { r.cell = (i32)w; r.status = OK_EXACT; }
        }
    }
    out[i] = r;
}

// cbReadCountExact (:92-96): reads whose cbMatch ended at 0 with a whitelist. Integer adds: order-free.
__global__ void k_bl_exact_hist(const BlRead* rd, u64 n, bool has_wl, u32* hist) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= n) return;
    if (has_wl && rd[i].status == OK_EXACT && rd[i].cell >= 0) atomicAdd(&hist[rd[i].cell], 1u);
}
__global__ void k_bl_add_pseudocount(u32* hist, u64 wl_n) {          // Solo.cpp:66-71
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i < wl_n) hist[i] += 1u;
}

// ---- CB_UMI_Simple: 1MM / N / posterior (matchCBtoWL :42-84 and inputRecords :35-107) -------------------------------
// qtab[q] = pow(10.0, -q / 10.0) as double for q = 0..33, built on the host with std::pow so it is STAR's value.
// Candidate order is STAR's (last base first, substitutions 1..3; the N case jj 0..3) because ties in `pin > pmax`
// go to the first candidate. A read with up to 3*cb_len candidates is resolved by one thread; the loops are short.
__global__ void k_bl_resolve_simple(const char* seq, const char* qual, const u64* off, u64 n, SimpleDev L,
                                    const u32* counts, const double* qtab, BlRead* rd, BlRecord* rec) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= n) return;
    BlRead r = rd[i];
    BlRecord o; o.read_id = (u32)i; o.cell_key = -1; o.umi_key = r.umi_key; o.umi_len = r.umi_len;
    o.status = r.status; o.feature_id = -1; o.feature_read = 0;
    if (r.status < 0 || !L.has_wl) { rec[i] = o; return; }              // dropped before matching (or no-WL: ranked elsewhere)
    if (r.cell >= 0) { o.cell_key = r.cell; rec[i] = o; return; }       // exact
    if (!L.mm1) { o.status = NO_WL; rec[i] = o; return; }               // Exact type: no 1MM (:42)
    const char* q = qual + off[i]; u32 len = (u32)(off[i + 1] - off[i]);
    i32 cand[96]; char cq[96]; u32 nc = 0;
    if (r.pos_n >= 0) {                                                 // one N: the four bases at the N (:46-63)
        u32 shift = 2 * (L.cb_len - 1 - (u32)r.pos_n); bool matched = false;
        char qn = (L.cb_start + (u32)r.pos_n) < len ? q[L.cb_start + r.pos_n] : 'H';
        for (u32 jj = 0; jj < 4; jj++) {
            i64 w = bsearch_u64(L.wl, L.wl_n, r.cb_val ^ ((u64)jj << shift));
            if (w >= 0) {
                if (!L.mm1_multi_nbase && matched) { o.status = TOO_MANY; rec[i] = o; return; }
                matched = true; cand[nc] = (i32)w; cq[nc] = qn; nc++;
            }
        }
    } else {                                                            // 1MM: every single substitution (:64-75)
        for (u32 ii = 0; ii < L.cb_len; ii++) {
            u32 qp = L.cb_start + (L.cb_len - 1 - ii);
            char qc = qp < len ? q[qp] : 'H';
            for (u32 jj = 1; jj < 4; jj++) {
                i64 w = bsearch_u64(L.wl, L.wl_n, r.cb_val ^ ((u64)jj << (ii * 2)));
                if (w >= 0) { cand[nc] = (i32)w; cq[nc] = qc; nc++; }
            }
        }
    }
    if (nc == 0) { o.status = NO_WL; rec[i] = o; return; }
    if (nc == 1) {
        if (L.one_exact && counts[cand[0]] == 0) { o.status = POST_REJECT; rec[i] = o; return; }   // :38
        o.cell_key = cand[0]; o.status = OK_ONE; rec[i] = o; return;
    }
    if (!L.mm1_multi) { o.status = TOO_MANY; rec[i] = o; return; }      // :80
    float ptot = 0.0f, pmax = 0.0f; i32 best = -1;                      // float, as STAR's non-MATCH_CellRanger build
    for (u32 k = 0; k < nc; k++) {
        u32 c = counts[cand[k]];
        if (c > 0) {
            int qin = (int)(signed char)cq[k] - 33; if (qin > 33) qin = 33;          // QSbase = QSmax = 33
            float pin = (float)((double)c * qtab[qin < 0 ? 0 : qin]);
            ptot = __fadd_rn(ptot, pin);
            if (pin > pmax) { best = cand[k]; pmax = pin; }
        }
    }
    if (ptot > 0.0f && pmax >= __fmul_rn(0.975f, ptot)) { o.cell_key = best; o.status = 2; }   // :90 cbMinP float
    else o.status = POST_REJECT;
    rec[i] = o;
}

// ---- CB_UMI_Complex, Exact (BD, SPLiT/Parse, inDrop, Microwell, SureCell, ddSEQ with segment lists) -----------------
__device__ __forceinline__ bool seg_extract(const SegDev& sg, const char* s, i32 len, i32 ad_start, i32 ad_len,
                                            i32& b, i32& e) {          // SoloBarcode_extractBarcode.cpp:4-33
    i32 pos[2];
    for (int k = 0; k < 2; k++) {
        switch (sg.anchor_type[k]) { case 0: pos[k] = 0; break; case 1: pos[k] = len - 1; break;
                                     case 2: pos[k] = ad_start; break; default: pos[k] = ad_start + ad_len - 1; }
        pos[k] += sg.anchor_dist[k];
    }
    if (pos[0] < 0 || pos[1] > len || pos[0] > pos[1]) return false;
    b = pos[0]; e = pos[1] + 1 > len ? len : pos[1] + 1;               // substr clips at the end: pos[1] == len is one short
    return true;
}

__global__ void k_bl_extract_complex(const char* seq, const u64* off, u64 n, ComplexDev C, BlRecord* rec) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= n) return;
    const char* s = seq + off[i]; i32 len = (i32)(off[i + 1] - off[i]);
    BlRecord o; o.read_id = (u32)i; o.cell_key = -1; o.umi_key = 0; o.umi_len = 0; o.status = NO_WL;
    o.feature_id = -1; o.feature_read = 0;
    i32 ad_start = 0;
    if (C.adapter_len) {                                               // localAlignHammingDist (SequenceFuns.cpp:352-371)
        i32 best = (i32)C.adapter_len;
        if (len < (i32)C.adapter_len) best = len + 1;
        else for (i32 p = 0; p + (i32)C.adapter_len <= len; p++) {
            i32 d = 0; for (u32 j = 0; j < C.adapter_len; j++) if (C.adapter[j] != 'N' && s[p + j] != C.adapter[j]) d++;
            if (d < best) { best = d; ad_start = p; }                  // strict <: the first best position wins
        }
        if (best > (i32)C.adapter_mm_max) { o.status = NO_ADAPTER; rec[i] = o; return; }
    }
    i32 ub, ue;
    if (!seg_extract(C.umi, s, len, ad_start, (i32)C.adapter_len, ub, ue)) { o.status = NO_UMI; rec[i] = o; return; }
    u32 ul = (u32)(ue - ub); o.umi_len = ul; o.umi_key = pack3(s + ub, ul);
    u64 umi_b; i32 uc = check_umi(s + ub, ul, 0, umi_b);               // homopolymer length 0: only poly-A is rejected
    bool good = (uc == 0); i32 status = uc;
    u64 cell = 0;
    for (u32 g = 0; g < C.n_seg; g++) {                                // :357-413
        const SegDev& sg = C.seg[g]; i32 b, e;
        bool ok = seg_extract(sg, s, len, ad_start, (i32)C.adapter_len, b, e);
        u32 L = ok ? (u32)(e - b) : 0;
        if (!ok || L < sg.min_len || L >= sg.n_len || sg.wl_off[L + 1] == sg.wl_off[L]) { if (good) { status = NO_CB; good = false; } }
        if (!good) continue;
        i32 pn; u64 v = conv(s + b, L, pn);
        if (pn != -1) { status = (pn == -2) ? N_IN_CB : NO_WL; good = false; continue; }   // Exact: one N is -1 (matchCBtoWL :42)
        i64 w = bsearch_u64(sg.wl + sg.wl_off[L], sg.wl_off[L + 1] - sg.wl_off[L], v);
        if (w < 0) { status = NO_WL; good = false; continue; }         // Exact type; 1MM and -12 are not drafted
        cell += sg.wl_factor * ((u64)w + sg.wl_add[L]);
    }
    o.status = good ? 0 : status;
    // 1MM Complex (C.mm1, with the -12 rule) is not drafted: singlet runs Exact for every complex protocol.
    if (good) o.cell_key = (i64)cell;
    rec[i] = o;
}

// ---- no-whitelist protocols: discovered barcode list (singlet.cpp Phase 0.9) ----------------------------------------
// Not a kernel: sort the 2-bit barcodes (thrust::sort), run-length them (reduce_by_key), keep count >= min_reads
// (5 on the unsorted stream, 100 on the sorted path), order by count descending then barcode ascending (stream rule), and
// cell_key = position in that list. A read's cell_key is then bsearch over the list's value-sorted copy. With the N-free test
// above that is the whole path; Visium is the same with the CSV spot rows gathered by barcode afterwards (k_bl_spot_gather).
__global__ void k_bl_spot_gather(const u64* cell_bc, u64 n_cells, const u64* spot_bc, u64 n_spots, i32* spot_row) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= n_cells) return;
    spot_row[i] = (i32)bsearch_u64(spot_bc, n_spots, cell_bc[i]);      // -1: zeros in the coordinate table
}

// ---- feature barcodes (ADT, hashtag): AdtMatcher ----------------------------------------------------------------------
// Host builds, with the same algorithm as AdtMatcher::build_index, two sorted tables over 2-bit window values:
//   ex_key/ex_id   exact tag -> id (a duplicated sequence keeps the last id)
//   h1_key/h1_id   Hamming-1 neighbour not equal to any tag -> id, or -2 when two tags share it
// Tags with N, or longer than 31 bases, are not representable: those libraries use the host lane.
struct TagDev { u32 tag_len; const u64* ex_key; const i32* ex_id; u64 ex_n; const u64* h1_key; const i32* h1_id; u64 h1_n; };

__global__ void k_bl_adt_match(const char* r2, const u64* off, u64 n, TagDev T, int max_offset, i32* tag_out) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= n) return;
    const char* s = r2 + off[i]; i32 len = (i32)(off[i + 1] - off[i]); i32 res = -1;
    if (T.tag_len > 0 && len >= (i32)T.tag_len) {
        i32 limit = len - (i32)T.tag_len < max_offset ? len - (i32)T.tag_len : max_offset;
        for (i32 o = 0; o <= limit; o++) {                             // adt_matcher.h:191-232
            i32 pn; u64 v = conv(s + o, T.tag_len, pn);
            if (pn != -1) continue;                                    // a window with N matches no tag
            i64 w = bsearch_u64(T.ex_key, T.ex_n, v);
            if (w >= 0) { res = T.ex_id[w]; break; }
            w = bsearch_u64(T.h1_key, T.h1_n, v);
            if (w >= 0) { res = T.h1_id[w]; break; }                   // -2 (ambiguous) returns at once
        }
    }
    tag_out[i] = res;
}
// The cell of an ADT read is its barcode matched EXACTLY against the called set (k_bl_extract_simple with mm1 = 0), the
// UMI is R1[bc_len:]. Counting is per (cell, tag) distinct UMI: exact-string dedup on CPU today; the R9 directional rule
// would change the numbers (see the doc, open question 3).

// ---- CRISPR guides: GuideRef::match ------------------------------------------------------------------------------------
// Substring of the upper-cased query sequence. CPU walks an unordered_map, so with several matching guides its winner
// is hash-order dependent; this kernel returns the lowest id and sets `ambiguous`, and the gate must only compare reads
// with a single matching guide. Guides are padded to <= 64 bases; reverse-strand reads arrive reverse-complemented in the
// BAM, so the CPU never matches them (the GPU must see the same sequence orientation as the BAM record).
__global__ void k_bl_guide_match(const char* qseq, const u64* off, u64 n, const char* guides, const u32* g_len,
                                 u32 n_guides, u32 g_stride, i32* gid, u8* ambiguous) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; if (i >= n) return;
    const char* s = qseq + off[i]; i32 len = (i32)(off[i + 1] - off[i]); i32 first = -1; u32 hits = 0;
    for (u32 g = 0; g < n_guides; g++) {
        i32 gl = (i32)g_len[g]; const char* gs = guides + (u64)g * g_stride;
        for (i32 p = 0; p + gl <= len; p++) {
            bool eq = true;
            for (i32 j = 0; j < gl && eq; j++) { char c = s[p + j]; if (c >= 'a' && c <= 'z') c -= 32; eq = (c == gs[j]); }
            if (eq) { if (first < 0) first = (i32)g; hits++; break; }
        }
    }
    gid[i] = first; ambiguous[i] = hits > 1;
}

}  // namespace bl
