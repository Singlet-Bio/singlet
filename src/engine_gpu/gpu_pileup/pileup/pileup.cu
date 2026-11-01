// Copied unchanged from PR #61 (branch gpu/w11-prototypes, commit f84aafff), bench/gpu/pileup/pileup.cu: see bench/gpu/pileup/README.md there.
// Linked into singlet through ../pileup_gpu.cu (PILEUP_NO_MAIN); identity: docs/gpu-engine/exploration.md 12.13, 12.19.
// W11 GPU annotation + pileup stage: alignments (STAR transcripts) -> every pileup output of the CPU engine (prod/campaign c4a1be38).
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -o pileup pileup.cu          (includes ../m6/m6.cu for dedup/collapse/CSR code)
// Run:   pileup --rec RECDIR --dump DUMPDIR --label NAME [--json out.json] [--reps 3] [--arena-gb 80]
//   RECDIR  sigs/alns/ops/seq/reads.bin + meta.txt from extract.cpp (pileup_rec.h)
//   DUMPDIR anno.bin (interval trees, gene strands, SNP table), params.txt, barcodes.txt, chroms.txt, truth.bin: from the patched CPU
//           engine (pileup_dump.patch). anno.bin is the engine's own annotation index; truth.bin is what the engine's callers receive.
//
// Stages (see README.md for the specification this ports):
//   1 annotate   per alignment (not per read): sorted-probe join of every aligned block against the engine's exon and intron interval
//                trees (same pre-order result list as the CPU tree), strand rule, exon/intron gene and feature indices, classify_read
//   2 siglists   per signature: multi-mapper resolution (resolve_mm), junction, SNP-overlap, chrM-base and chrM-indel lists (count + fill)
//   3 sjids      sort of the distinct junction instances -> canonical junction ids
//   4 expand     per read: gather of the signature results -> tuples (scan + fill)
//   b c d        M6: (cell, gene, UMI) sort + reduce, directional UMI collapse, CSR  (m6.cu, unchanged)
//   5 sj         (cell, junction, UMI) sort, reads per exact UMI, R9 directional collapse per (cell, junction), molecules per (cell, junction)
//   6 snp        (cell, SNP, UMI) sort, class counts per exact UMI, R9 collapse per (cell, SNP), R1 vote per molecule, DP / AD per (cell, SNP)
//   7 mt         (cell, position, base, strand) sort + run-length encode; chrM indel tuples
// Determinism: integer sorts / scans / reduce-by-key only, no float arithmetic, no atomics on values (-fmad=false anyway).
#define main m6_unused_main
#include "../m6/m6.cu"
#undef main
#include "pileup_rec.h"
#include <fstream>
#include <sstream>
#include <functional>
#include <thread>
#include <tuple>
#include <cmath>
#include <cstdio>

static const uint32_t MT_LEN = 16569;
struct T3 { uint32_t cell, row, val; };
struct Par { int stranded = 1, reverse_strand = 0, min_mapq = 20, min_baseq = 10, count_introns = 1, count_sj = 1, count_mt = 0, umi_dedup = 1, directional = 1, umi_len = 12, chrm_tid = -1;
             uint32_t n_snps = 0, n_barcodes = 0, n_tid = 0, n_exons = 0, n_introns = 0, n_genes = 0; int has_gene_model = 1; };
struct IV { int32_t s, e; uint32_t idx; };
struct SnpRow { int32_t pos; uint32_t idx; char ref, alt; uint16_t pad; };

// ------------------------------------------------------------------------------------------------------------------------
// host loading
template <class T> static std::vector<T> read_vec(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET); std::vector<T> v(sz / sizeof(T));
    if (sz && fread(v.data(), 1, v.size() * sizeof(T), f) != v.size() * sizeof(T)) die("short read"); fclose(f); return v; }
static std::map<std::string, double> read_kv(const std::string& p) { std::map<std::string, double> m; std::ifstream f(p); std::string k; double v; while (f >> k >> v) m[k] = v; return m; }
static Par load_par(const std::string& p) {
    auto m = read_kv(p); Par r;
    r.stranded = (int)m["stranded"]; r.reverse_strand = (int)m["reverse_strand"]; r.min_mapq = (int)m["min_mapq"]; r.min_baseq = (int)m["min_baseq"]; r.count_introns = (int)m["count_introns"];
    r.count_sj = (int)m["count_sj"]; r.count_mt = (int)m["count_mt"]; r.umi_dedup = (int)m["umi_dedup"]; r.directional = (int)m["umi_dedup_directional"]; r.umi_len = (int)m["umi_len"];
    r.chrm_tid = (int)m["chrm_tid"]; r.n_snps = (uint32_t)m["n_snps"]; r.n_barcodes = (uint32_t)m["n_barcodes"]; r.n_tid = (uint32_t)m["n_tid"]; r.has_gene_model = (int)m["has_gene_model"];
    r.n_exons = (uint32_t)m["n_exons"]; r.n_introns = (uint32_t)m["n_introns"]; r.n_genes = (uint32_t)m["n_genes"]; return r; }
struct Anno { std::vector<std::vector<IV>> ex, in; std::vector<uint32_t> e2g, i2g; std::vector<char> gstr; std::vector<std::vector<SnpRow>> snp; };
static Anno load_anno(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) die("cannot open anno.bin"); char mg[8]; if (fread(mg, 1, 8, f) != 8 || memcmp(mg, "PUANN1", 6)) die("bad anno.bin");
    Anno a; uint32_t nt; if (fread(&nt, 4, 1, f) != 1) die("anno"); a.ex.resize(nt); a.in.resize(nt);
    for (uint32_t t = 0; t < nt; t++) for (int k = 0; k < 2; k++) { uint64_t n; if (fread(&n, 8, 1, f) != 1) die("anno n"); auto& v = k ? a.in[t] : a.ex[t]; v.resize(n); if (n && fread(v.data(), sizeof(IV), n, f) != n) die("anno iv"); }
    auto rv = [&](auto& v) { uint64_t n; if (fread(&n, 8, 1, f) != 1) die("anno vec"); v.resize(n); if (n && fread(v.data(), sizeof(v[0]), n, f) != n) die("anno vec data"); };
    rv(a.e2g); rv(a.i2g); rv(a.gstr);
    uint32_t ns; if (fread(&ns, 4, 1, f) != 1) die("anno snp"); a.snp.resize(ns);
    for (uint32_t t = 0; t < ns; t++) { uint64_t n; if (fread(&n, 8, 1, f) != 1) die("snp n"); a.snp[t].resize(n); if (n && fread(a.snp[t].data(), sizeof(SnpRow), n, f) != n) die("snp rows"); }
    fclose(f); return a; }
static std::vector<T3> read_acc(FILE* f, uint32_t* nr_out = nullptr, uint32_t* nc_out = nullptr) {
    uint32_t nr, nc; uint64_t nnz; if (fread(&nr, 4, 1, f) != 1 || fread(&nc, 4, 1, f) != 1 || fread(&nnz, 8, 1, f) != 1) die("truth acc header");
    std::vector<uint32_t> buf(nnz * 3); if (nnz && fread(buf.data(), 4, buf.size(), f) != buf.size()) die("truth acc data");
    std::vector<T3> v(nnz); for (uint64_t k = 0; k < nnz; k++) v[k] = T3{buf[3 * k + 1], buf[3 * k], buf[3 * k + 2]};   // (cell = col, row, val), column-major, rows ascending
    if (nr_out) *nr_out = nr; if (nc_out) *nc_out = nc; return v; }
struct CpuTruth { std::vector<T3> exon, intron, amb, sj, snp_ad, snp_dp, mt, mt_rev; std::vector<std::string> sj_names; std::vector<std::array<uint32_t, 4>> indels; uint64_t n_ambig_reads = 0; };
static CpuTruth load_cpu_truth(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) die("cannot open truth.bin"); char mg[8]; if (fread(mg, 1, 8, f) != 8 || memcmp(mg, "PUTRU1", 6)) die("bad truth.bin");
    CpuTruth t; t.exon = read_acc(f); t.intron = read_acc(f); t.amb = read_acc(f); t.sj = read_acc(f);
    uint32_t nn; if (fread(&nn, 4, 1, f) != 1) die("sj names"); t.sj_names.resize(nn);
    for (auto& s : t.sj_names) { uint32_t l; if (fread(&l, 4, 1, f) != 1) die("sj name len"); s.resize(l); if (l && fread(&s[0], 1, l, f) != l) die("sj name"); }
    t.snp_ad = read_acc(f); t.snp_dp = read_acc(f); t.mt = read_acc(f); t.mt_rev = read_acc(f);
    uint64_t ni; if (fread(&ni, 8, 1, f) != 1) die("indels"); t.indels.resize(ni); if (ni && fread(t.indels.data(), 16, ni, f) != ni) die("indel data");
    if (fread(&t.n_ambig_reads, 8, 1, f) != 1) t.n_ambig_reads = 0; fclose(f); return t; }
static std::vector<std::string> read_lines(const std::string& p) { std::vector<std::string> v; std::ifstream f(p); std::string l; while (std::getline(f, l)) v.push_back(l); return v; }

// ------------------------------------------------------------------------------------------------------------------------
// device structures
struct DTree { const int32_t *s, *e, *me; const uint32_t *idx, *off, *n; int ntid; };
struct DSnp { const int32_t* pos; const uint32_t* idx; const char* alt; const char* ref; const uint32_t *off, *n; int ntid; };
struct DPar { int stranded, reverse, count_introns, count_sj, count_mt, min_mapq, chrm_tid, snp_skip_n; uint32_t rules; int use_q; };   // rules: intron_rule::k* bits (pileup_engine.h); 1, 4, 128, 256 are implemented
struct AA { uint32_t gene, exon, intron, aln_gene, flags, multi, stat, exg, hasn, pg_a, pg_b, pad; };   // hasn: CIGAR N; pg_a / pg_b: a multigene primary's first two exon genes (the mm_buffer's first_gene / second_gene)   // stat: see k_annotate; exg: determine_exon_gene     // per alignment: classify_read result (primary) or exon hit (secondary)
struct GRes { uint32_t gene, exon, intron, flags; };                    // per signature: what count_read receives (flags == 0: nothing is counted)
struct SJL { int32_t tid, donor, acc; };
struct IND { uint32_t pos, lenins; };
static const uint32_t MAXU = 0xFFFFFFFFu;

template <class T> static T* to_dev(Arena& A, const std::vector<T>& v) { T* d = A.alloc<T>(std::max<size_t>(1, v.size())); if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
static int32_t build_aug(const std::vector<IV>& iv, std::vector<int32_t>& me, size_t lo, size_t hi) {
    if (lo >= hi) return 0; size_t mid = lo + (hi - lo) / 2; int32_t l = build_aug(iv, me, lo, mid), r = build_aug(iv, me, mid + 1, hi);
    me[mid] = std::max({iv[mid].e, l, r}); return me[mid]; }
static DTree upload_tree(Arena& A, const std::vector<std::vector<IV>>& trees) {
    std::vector<int32_t> S, E, ME; std::vector<uint32_t> I, off(trees.size()), n(trees.size());
    for (size_t t = 0; t < trees.size(); t++) {
        off[t] = (uint32_t)S.size(); n[t] = (uint32_t)trees[t].size(); std::vector<int32_t> me(trees[t].size()); build_aug(trees[t], me, 0, trees[t].size());
        for (size_t k = 0; k < trees[t].size(); k++) { S.push_back(trees[t][k].s); E.push_back(trees[t][k].e); I.push_back(trees[t][k].idx); ME.push_back(me[k]); } }
    DTree d; d.s = to_dev(A, S); d.e = to_dev(A, E); d.me = to_dev(A, ME); d.idx = to_dev(A, I); d.off = to_dev(A, off); d.n = to_dev(A, n); d.ntid = (int)trees.size(); return d; }

// the engine's IntervalTree::query, pre-order (node, left, right) with the same pruning, so results arrive in the same order
template <class F> __device__ __forceinline__ void tree_query(const DTree& T, int tid, int32_t qs, int32_t qe, F& f) {
    if (tid < 0 || tid >= T.ntid) return; uint32_t n = T.n[tid]; if (!n) return; uint32_t base = T.off[tid];
    const int32_t* S = T.s + base; const int32_t* E = T.e + base; const int32_t* ME = T.me + base; const uint32_t* I = T.idx + base;
    uint32_t slo[48], shi[48]; int sp = 0; slo[0] = 0; shi[0] = n; sp = 1;
    while (sp) {
        sp--; uint32_t lo = slo[sp], hi = shi[sp]; if (lo >= hi) continue; uint32_t mid = lo + (hi - lo) / 2;
        if (S[mid] < qe && E[mid] > qs) f(I[mid]);
        if (mid + 1 < hi && qe > S[mid]) { slo[sp] = mid + 1; shi[sp] = hi; sp++; }
        if (lo < mid && ME[lo + (mid - lo) / 2] > qs) { slo[sp] = lo; shi[sp] = mid; sp++; }
    }
}
struct StrandPar { int stranded, reverse; const char* gstr; uint32_t ng; bool read_rev; };
__device__ __forceinline__ bool wrong_strand(const StrandPar& S, uint32_t g) {
    if (!S.stranded) return false; char gs = g < S.ng ? S.gstr[g] : '.'; if (gs == '.') return false;
    bool er = (gs == '-'); if (S.reverse) er = !er; return S.read_rev != er; }
struct ExonCb {
    const StrandPar* S; const uint32_t* e2g; uint32_t ne; bool any, same; uint32_t first_idx, first_gene;
    bool sense = false, anti = false; uint32_t blk_raw = 0, blk_ok = 0, second_gene = MAXU;   // second_gene: the first accepted gene other than first_gene   // engine's strand_vote (unfiltered overlaps) and block counts
    __device__ void operator()(uint32_t idx) {
        uint32_t g = idx < ne ? e2g[idx] : MAXU;
        { char gs = g < S->ng ? S->gstr[g] : '.'; if (gs != '.') { if (S->read_rev == (gs == '-')) sense = true; else anti = true; } }
        blk_raw++;
        if (wrong_strand(*S, g)) return;
        blk_ok++;
        if (!any) { any = true; first_idx = idx; first_gene = g; same = true; } else if (g != first_gene) { same = false; if (second_gene == MAXU && g != MAXU) second_gene = g; } } };
struct IntronCb {
    const StrandPar* S; const uint32_t* i2g; uint32_t ni; uint32_t exon_gene; bool any, same; uint32_t first_idx, first_gene, match_idx;
    __device__ void operator()(uint32_t idx) {
        uint32_t g = idx < ni ? i2g[idx] : MAXU; if (wrong_strand(*S, g)) return;
        if (!any) { any = true; first_idx = idx; first_gene = g; same = true; } else if (g != first_gene) same = false;
        if (exon_gene != MAXU && g == exon_gene && match_idx == MAXU) match_idx = idx;
        if (exon_gene != MAXU && g != exon_gene) other = true; }   // classify_read's other_gene_intron
    bool other = false; };

__global__ void k_annotate(const AlnRec* A, int nA, const uint32_t* ops, DTree EX, DTree IN, const uint32_t* e2g, uint32_t ne, const uint32_t* i2g, uint32_t ni, const char* gstr, uint32_t ng, DPar P, AA* out) {
    int a = blockIdx.x * blockDim.x + threadIdx.x; if (a >= nA) return;
    AlnRec x = A[a]; bool prim = (x.flags & 2) != 0; StrandPar SP{P.stranded, P.reverse, gstr, ng, (x.flags & 1) != 0}; bool had_reject = false, hasn = false;
    ExonCb ec{&SP, e2g, ne, false, true, MAXU, MAXU};
    const uint32_t* o = ops + x.ops_off; int32_t rpos = x.pos;
    for (int i = 0; i < x.n_ops; i++) {
        uint32_t c = o[i]; int op = c & 15; int32_t len = (int32_t)(c >> 4);
        if (op == 0 || op == 7 || op == 8) {
            ec.blk_raw = 0; ec.blk_ok = 0;
            tree_query(EX, x.tid, rpos, rpos + len, ec); rpos += len;
            if (ec.blk_raw > 0 && ec.blk_ok == 0) had_reject = true;   // a block whose exons were all on the wrong strand
        }
        else if (op == 2 || op == 3) { rpos += len; if (op == 3) hasn = true; }
    }
    AA r{MAXU, MAXU, MAXU, MAXU, 0, 0, 0, MAXU, hasn ? 1u : 0u, MAXU, MAXU, 0};
    r.exg = (ec.any && ec.same) ? ec.first_gene : MAXU;   // the engine's determine_exon_gene (single gene; multi-gene -> none)
    if (!prim) { if (ec.any) { if (ec.same) { r.gene = ec.first_gene; r.exon = ec.first_idx; } else r.multi = 1; } out[a] = r; return; }
    bool unique_e = ec.any && ec.same && ec.first_gene != MAXU; uint32_t exon_gene = unique_e ? ec.first_gene : MAXU; bool pm = ec.any && !unique_e;
    IntronCb ic{&SP, i2g, ni, exon_gene, false, true, MAXU, MAXU, MAXU};
    if (P.count_introns) {
        rpos = x.pos;
        for (int i = 0; i < x.n_ops; i++) {
            uint32_t c = o[i]; int op = c & 15; int32_t len = (int32_t)(c >> 4);
            if (op == 0 || op == 7 || op == 8) { tree_query(IN, x.tid, rpos, rpos + len, ic); rpos += len; }
            else if (op == 2 || op == 3) rpos += len;
        }
    }
    uint32_t intron_gene = (ic.any && ic.same && ic.first_gene != MAXU) ? ic.first_gene : MAXU;
    if (pm) { }
    else if (exon_gene != MAXU) { r.gene = exon_gene; r.exon = ec.first_idx; r.intron = ic.match_idx; r.flags = ic.match_idx == MAXU ? 5 : 6; }
    else if (intron_gene != MAXU) { r.gene = intron_gene; r.intron = ic.first_idx; r.flags = 2; }
    r.multi = pm; r.aln_gene = exon_gene != MAXU ? exon_gene : intron_gene;
    if (pm) { r.pg_a = ec.first_gene; r.pg_b = ec.second_gene; }
    // stat (primary alignments; the engine's serial loop counters): 1 strand vote sense, 2 strand vote antisense, 4 wrong_strand
    // (no accepted exon overlap although a block had only wrong-strand exons), 8 cross_gene_intron (classify_read), 16 multigene
    r.stat = (ec.sense ? 1u : 0u) | (ec.anti ? 2u : 0u) | ((!ec.any && had_reject) ? 4u : 0u)
           | ((!pm && exon_gene != MAXU && ic.other) ? 8u : 0u) | (pm ? 16u : 0u);
    out[a] = r;
}

// one signature -> what the engine's count_read receives (NH == 1 directly, NH > 1 through the mm_buffer state machine; R7: the primary's gene, else the smallest secondary exon)
// intron rules (P.rules, PR #80): refine_read's read-level drops (kUnique: NH > 1, kNoSplicedIntron: CIGAR N) clear an intronic primary's gene (also in
// aa, which the host reads back as the classify_read gene), kMtPrefer: a multimapper that ties a chrM gene with a nuclear gene counts the chrM gene
// (note_mt_alignment / mt_numt_tie; the chrM alignment is the last one in the signature's order, as the stream's last writer)
__global__ void k_resolve(const SigRec* S, int nS, const AlnRec* A, AA* aa, const uint8_t* gmt, uint32_t ngmt, DPar P, GRes* out) {
    int s = blockIdx.x * blockDim.x + threadIdx.x; if (s >= nS) return;
    SigRec sg = S[s]; GRes r{MAXU, MAXU, MAXU, 0};
    auto is_mt = [&](uint32_t g) { return g < ngmt && gmt[g] != 0; };
    if (sg.n_aln > 0) {
        if (sg.nh == 1) {
            for (int k = 0; k < sg.n_aln; k++) if (A[sg.aln_off + k].flags & 2) {
                AA x = aa[sg.aln_off + k];
                if (x.gene != MAXU && (x.flags & 2) && (P.rules & 4) && x.hasn) { aa[sg.aln_off + k].gene = MAXU; x.gene = MAXU; }
                if (x.gene != MAXU) { r.gene = x.gene; r.exon = x.exon; r.intron = x.intron; r.flags = x.flags ? x.flags : 5; } break; }
        } else {
            bool amb = false, pa = false; uint32_t fg = MAXU, sec = MAXU, gi = MAXU, ex = MAXU, in = MAXU, fl = 0;   // pa: the primary assigned the gene
            uint32_t mg = MAXU, mex = MAXU, min_ = MAXU, mfl = 0;                                                    // the chrM gene's alignment (kMtPrefer)
            for (int k = 0; k < sg.n_aln; k++) {
                AA x = aa[sg.aln_off + k];
                if (A[sg.aln_off + k].flags & 2) {
                    if (x.gene != MAXU && (x.flags & 2) && (((P.rules & 1)) || ((P.rules & 4) && x.hasn))) { aa[sg.aln_off + k].gene = MAXU; x.gene = MAXU; }   // NH > 1 here
                    if (x.gene != MAXU) {
                        gi = x.gene; ex = x.exon; in = x.intron; fl = x.flags; pa = true;
                        if ((P.rules & 256) && is_mt(x.gene)) { mg = x.gene; mex = x.exon; min_ = x.intron; mfl = x.flags; }
                    }
                    if (x.multi) {
                        amb = true;
                        if (sec == MAXU && x.pg_a != MAXU) { if (fg == MAXU) fg = x.pg_a; if (x.pg_b != MAXU) sec = x.pg_b; }
                    }
                    if (x.aln_gene != MAXU) { if (fg == MAXU) fg = x.aln_gene; else if (x.aln_gene != fg) { amb = true; if (sec == MAXU) sec = x.aln_gene; } }
                } else {
                    if (x.multi) amb = true;
                    else if (x.gene != MAXU) {
                        if (fg == MAXU) fg = x.gene; else if (x.gene != fg) { amb = true; if (sec == MAXU) sec = x.gene; }
                        if (!pa && (gi == MAXU || x.exon < ex)) { gi = x.gene; ex = x.exon; fl = 5; }   // R7: else the smallest secondary exon
                        if ((P.rules & 256) && is_mt(x.gene)) { mg = x.gene; mex = x.exon; min_ = MAXU; mfl = 5; }
                    }
                }
            }
            if (amb && mg != MAXU && fg != MAXU && sec != MAXU && is_mt(fg) != is_mt(sec)) { r.gene = mg; r.exon = mex; r.intron = min_; r.flags = (mfl ? mfl : 5) | 256u; }   // 256: counted for the chrM gene (mt_numt_rescued)
            else if (!amb && fg != MAXU && gi != MAXU) { r.gene = gi; r.exon = ex; r.intron = in; r.flags = fl ? fl : 5; }
        }
    }
    out[s] = r;
}

__device__ __forceinline__ uint8_t comp16(uint8_t c) { return ((c & 1) << 3) | ((c & 2) << 1) | ((c & 4) >> 1) | ((c & 8) >> 3); }
// BAM nt16 code of the base at position qp of the alignment (aligned orientation); the signature stores the read in its original orientation
__device__ __forceinline__ uint8_t aligned_base(const uint8_t* seq, uint64_t off, int rlen, int qp, bool rev) {
    int i = rev ? rlen - 1 - qp : qp; uint8_t c = (seq[off + (i >> 1)] >> ((i & 1) * 4)) & 15; return rev ? comp16(c) : c; }
__device__ __forceinline__ char nt16_char(uint8_t c) { const char* t = "=ACMGRSVTWYHKDBN"; return t[c]; }
__device__ __forceinline__ int base_to_idx(uint8_t c) { return c == 1 ? 0 : c == 2 ? 1 : c == 4 ? 2 : c == 8 ? 3 : -1; }

// per signature lists from the primary alignment: junctions (CIGAR N), SNP overlaps, chrM bases, chrM indels. Pass 1 counts, pass 2 fills.
template <bool FILL>
__global__ void k_lists(const SigRec* S, int nS, const AlnRec* A, const uint32_t* ops, const uint8_t* seq, DSnp SN, DPar P,
                        uint32_t* c_sj, uint32_t* c_snp, uint32_t* c_mt, uint32_t* c_ind, SJL* o_sj, uint32_t* o_snp, uint32_t* o_mt, IND* o_ind) {
    int s = blockIdx.x * blockDim.x + threadIdx.x; if (s >= nS) return;
    SigRec sg = S[s]; int p = -1; for (int k = 0; k < sg.n_aln; k++) if (A[sg.aln_off + k].flags & 2) { p = k; break; }
    uint32_t nsj = 0, nsnp = 0, nmt = 0, nind = 0;
    uint32_t wsj = FILL ? c_sj[s] : 0, wsnp = FILL ? c_snp[s] : 0, wmt = FILL ? c_mt[s] : 0, wind = FILL ? c_ind[s] : 0;
    if (p >= 0) {
        AlnRec a = A[sg.aln_off + p]; bool high = a.mapq >= P.min_mapq, rev = (a.flags & 1) != 0;
        bool sj_ok = P.count_sj && (high || sg.nh > 1);
        bool snp_ok = high && a.tid >= 0 && a.tid < SN.ntid && SN.n[a.tid] > 0;
        bool mt_ok = P.count_mt && high && a.tid == P.chrm_tid;
        const uint32_t* o = ops + a.ops_off; int32_t rpos = a.pos; int32_t qpos = 0;
        for (int i = 0; i < a.n_ops; i++) {
            uint32_t c = o[i]; int op = c & 15; int32_t len = (int32_t)(c >> 4);
            if (op == 0 || op == 7 || op == 8) {
                if (snp_ok) {
                    const int32_t* sp = SN.pos + SN.off[a.tid]; uint32_t n = SN.n[a.tid]; uint32_t lo = 0, hi = n;
                    while (lo < hi) { uint32_t mid = (lo + hi) >> 1; if (sp[mid] < rpos) lo = mid + 1; else hi = mid; }
                    for (uint32_t k = lo; k < n && sp[k] < rpos + len; k++) {
                        int qp = qpos + (sp[k] - rpos); uint8_t code = aligned_base(seq, sg.seq_off, sg.rlen, qp, rev);
                        if (P.snp_skip_n && code == 15) continue;     // the engine skips a base with quality < min_baseq; in some samples that is exactly the N bases (extractor checks every base, meta snp_skip_n)
                        // base class as SnpConsensus::base_class (order-free-rules R1): ALT if base == alt, REF if base == ref, else OTHER
                        const char bc = nt16_char(code);
                        const uint32_t cls = bc == SN.alt[SN.off[a.tid] + k] ? 1u : (bc == SN.ref[SN.off[a.tid] + k] ? 0u : 2u);
                        if (FILL) o_snp[wsnp + nsnp] = (SN.idx[SN.off[a.tid] + k] << 2) | cls | (P.use_q ? ((uint32_t)(rev ? sg.rlen - 1 - qp : qp) << 25) : 0u);   // use_q: bits 25-31 = the base's index in the record (original orientation)
                        nsnp++; }
                }
                if (mt_ok) for (int j = 0; j < len; j++) {
                    uint32_t rp = (uint32_t)(rpos + j); if (rp >= MT_LEN) continue;
                    int bi = base_to_idx(aligned_base(seq, sg.seq_off, sg.rlen, qpos + j, rev)); if (bi < 0) continue;
                    if (FILL) o_mt[wmt + nmt] = rp * 4 + (uint32_t)bi + (P.use_q ? ((uint32_t)(rev ? sg.rlen - 1 - (qpos + j) : qpos + j) << 17) : 0u); nmt++; }   // use_q: bits 17-23 = the base's index in the record
                rpos += len; qpos += len;
            } else if (op == 1) { if (mt_ok && (uint32_t)rpos < MT_LEN) { if (FILL) o_ind[wind + nind] = IND{(uint32_t)rpos, ((uint32_t)min(len, 0xFFFF) << 1) | 1u}; nind++; } qpos += len; }
            else if (op == 4) qpos += len;
            else if (op == 2) { if (mt_ok && (uint32_t)rpos < MT_LEN) { if (FILL) o_ind[wind + nind] = IND{(uint32_t)rpos, ((uint32_t)min(len, 0xFFFF) << 1)}; nind++; } rpos += len; }
            else if (op == 3) { if (sj_ok) { if (FILL) o_sj[wsj + nsj] = SJL{a.tid, rpos, rpos + len}; nsj++; } rpos += len; }
        }
    }
    if (!FILL) { c_sj[s] = nsj; c_snp[s] = nsnp; c_mt[s] = nmt; c_ind[s] = nind; }
}

// ---- read level ----
__device__ __forceinline__ uint32_t umi_b5(uint64_t u3, int len) { uint32_t v = 0, m = 1; for (int i = 0; i < len; i++) { v += (uint32_t)((u3 >> (3 * i)) & 7) * m; m *= 5; } return v; }
__device__ __forceinline__ uint64_t umi_b2(uint64_t u3, int len) { uint64_t v = 0; for (int i = 0; i < len; i++) { uint64_t c = (u3 >> (3 * i)) & 7; v |= (c > 3 ? 3ull : c) << (2 * i); } return v; }
// per-base quality (dp.use_q): a list entry of a signature (SNP: bits 25-31, chrM base: bits 17-23) belongs to a read only if the base's quality passed
__device__ __forceinline__ bool q_pass(const uint64_t* Q, int r, uint32_t pos) { return ((Q[2 * (size_t)r + (pos >> 6)] >> (pos & 63)) & 1ull) == 0ull; }
__global__ void k_cnt_q(const ReadRec* R, int n, const uint32_t* list_off, const uint32_t* list_cnt, const uint32_t* lists, const uint64_t* Q, int qsh, uint32_t* cnt) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= n) return; ReadRec x = R[r]; if (x.bc < 0) { cnt[r] = 0; return; }
    uint32_t c = list_cnt[x.sig], b = list_off[x.sig], m = 0; for (uint32_t k = 0; k < c; k++) m += q_pass(Q, r, (lists[b + k] >> qsh) & 127u); cnt[r] = m; }
__global__ void k_gatherq(const uint64_t* Q, const uint32_t* ridx, int n, uint64_t* out) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { out[2 * (size_t)i] = Q[2 * (size_t)ridx[i]]; out[2 * (size_t)i + 1] = Q[2 * (size_t)ridx[i] + 1]; } }
__global__ void k_cnt(const ReadRec* R, int n, const uint32_t* sigcnt, uint32_t* cnt) { int r = blockIdx.x * blockDim.x + threadIdx.x; if (r < n) cnt[r] = R[r].bc >= 0 ? sigcnt[R[r].sig] : 0u; }
__global__ void k_gflag(const ReadRec* R, int n, const GRes* G, uint32_t* f) { int r = blockIdx.x * blockDim.x + threadIdx.x; if (r < n) f[r] = (R[r].bc >= 0 && G[R[r].sig].flags != 0) ? 1u : 0u; }
__global__ void k_gfill(const ReadRec* R, int n, const GRes* G, const uint32_t* f, const uint32_t* off, int ulen, Rec* out) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= n || !f[r]) return;
    ReadRec x = R[r]; GRes g = G[x.sig]; Rec o; o.bc = (uint32_t)x.bc; o.gene = g.gene; o.exon = g.exon; o.intron = g.intron; o.umi = umi_b2(x.umi3, ulen);
    o.flags = (uint8_t)g.flags; o.umi_len = (uint8_t)ulen; o.hasN = 0; o.direct = 0; o.pad = 0; out[off[r]] = o; }
__global__ void k_fill_hilo(const ReadRec* R, int n, const uint32_t* off, const uint32_t* list_off, const uint32_t* list_cnt, const uint32_t* lists, int is_snp, int hb, int ulen,
                            uint64_t* hi, uint32_t* lo, uint8_t* alt, const SigRec* S = nullptr, const GRes* G = nullptr, uint32_t* tg = nullptr, const uint64_t* Q = nullptr) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= n) return; ReadRec x = R[r]; if (x.bc < 0) return;
    uint32_t c = list_cnt[x.sig], b = list_off[x.sig], o = off[r]; uint32_t u = umi_b5(x.umi3, ulen);
    // R9 per gene: the evidence gene is the unique read's gene (cur_read_gene_ is set for NH 1 only)
    const uint32_t gene = (tg && S[x.sig].nh == 1) ? G[x.sig].gene : MAXU;
    uint32_t w = 0; for (uint32_t k = 0; k < c; k++) { uint32_t e = lists[b + k]; if (Q && !q_pass(Q, r, (e >> 25) & 127u)) continue; hi[o + w] = ((uint64_t)x.bc << hb) | (is_snp ? ((e >> 2) & (Q ? 0x7FFFFFu : 0xFFFFFFFFu)) : e); lo[o + w] = u; if (is_snp) alt[o + w] = (uint8_t)(e & 3); if (tg) tg[o + w] = gene; w++; }
}
// the gene reducer's molecule store, kept for the junction and SNP reducers (R9 per gene): the sorted unique (cell, gene, UMI) keys and the
// component (root entry index) of each after the directional collapse
struct GeneStore { const uint64_t* uk; const uint32_t* lab; int m, gb, ub; };
__global__ void k_gather32(const uint32_t* src, const uint32_t* idx, int n, uint32_t* out) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = src[idx[i]]; }
// per unique (cell, feature, UMI) entry: the component of the gene store entry (cell, gene, UMI as 2 bits per base, N as T), 0xFFFFFFFF if the
// reads had no gene or the store does not hold the UMI (r9_molecules: gene_root false)
__global__ void k_gene_rid(const uint64_t* uh, const uint32_t* ul, const uint32_t* gmin, int m, int hb, int ulen, GeneStore gs, uint32_t* rid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= m) return;
    rid[i] = 0xFFFFFFFFu; const uint32_t g = gmin[i]; if (g == MAXU) return;
    uint32_t u5 = ul[i]; uint64_t u2 = 0; for (int k = 0; k < ulen; k++) { u2 |= (uint64_t)min(u5 % 5, 3u) << (2 * k); u5 /= 5; }
    const uint64_t key = ((uh[i] >> hb) << (gs.gb + gs.ub)) | ((uint64_t)g << gs.ub) | u2;
    int lo = 0, hi = gs.m; while (lo < hi) { int mid = (lo + hi) >> 1; if (gs.uk[mid] < key) lo = mid + 1; else hi = mid; }
    if (lo < gs.m && gs.uk[lo] == key) rid[i] = gs.lab[lo];
}
__global__ void k_mflag(const uint32_t* lab, int m, uint32_t* f) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < m) f[i] = lab[i] != (uint32_t)i; }
__global__ void k_mgather(const uint64_t* uk, const uint32_t* lab, const uint32_t* idx, int n, uint64_t* a, uint64_t* b) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j < n) { a[j] = uk[idx[j]]; b[j] = uk[lab[idx[j]]]; } }
__global__ void k_fill_mt(const ReadRec* R, const AlnRec* A, const SigRec* S, int n, const uint32_t* off, const uint32_t* list_off, const uint32_t* list_cnt, const uint32_t* lists, uint64_t* key, const uint64_t* Q = nullptr) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= n) return; ReadRec x = R[r]; if (x.bc < 0) return;
    SigRec sg = S[x.sig]; uint32_t rev = 0; for (int k = 0; k < sg.n_aln; k++) if (A[sg.aln_off + k].flags & 2) { rev = A[sg.aln_off + k].flags & 1; break; }
    uint32_t c = list_cnt[x.sig], b = list_off[x.sig], o = off[r];
    uint32_t w = 0; for (uint32_t k = 0; k < c; k++) { const uint32_t e = lists[b + k]; if (Q && !q_pass(Q, r, (e >> 17) & 127u)) continue; key[o + w++] = ((uint64_t)x.bc << 18) | ((uint64_t)(Q ? (e & 0x1FFFFu) : e) << 1) | rev; }
}
__global__ void k_fill_ind(const ReadRec* R, int n, const uint32_t* off, const uint32_t* list_off, const uint32_t* list_cnt, const IND* lists, uint32_t* o_bc, IND* o_ind,
                           const uint32_t* ridx, uint32_t* o_ri) {   // o_ri: the read's index in the full input (indels keep read order across partitions)
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= n) return; ReadRec x = R[r]; if (x.bc < 0) return;
    uint32_t c = list_cnt[x.sig], b = list_off[x.sig], o = off[r]; for (uint32_t k = 0; k < c; k++) { o_bc[o + k] = (uint32_t)x.bc; o_ind[o + k] = lists[b + k]; o_ri[o + k] = ridx ? ridx[r] : (uint32_t)r; }
}
// junction table: (donor key, acceptor) ordered; ids by rank
__global__ void k_sjkey(const SJL* l, int n, uint64_t* k1, uint32_t* k2, uint32_t* idx) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; k1[i] = ((uint64_t)(uint32_t)l[i].tid << 32) | (uint32_t)l[i].donor; k2[i] = (uint32_t)l[i].acc; idx[i] = i; }
__global__ void k_gather64(const uint64_t* src, const uint32_t* idx, int n, uint64_t* out) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = src[idx[i]]; }
__global__ void k_newflag2(const uint64_t* k1, const uint32_t* k2, const uint32_t* ord, int n, uint32_t* f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    f[i] = (i == 0) || k1[ord[i]] != k1[ord[i - 1]] || k2[ord[i]] != k2[ord[i - 1]]; }
__global__ void k_jid(const uint32_t* ord, const uint32_t* rk, int n, uint32_t* jid) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) jid[ord[i]] = rk[i] - 1; }
__global__ void k_sjtab(const uint64_t* k1, const uint32_t* k2, const uint32_t* ord, const uint32_t* f, const uint32_t* rk, int n, uint64_t* t1, uint32_t* t2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n || !f[i]) return; t1[rk[i] - 1] = k1[ord[i]]; t2[rk[i] - 1] = k2[ord[i]]; }
__global__ void k_runflag(const uint64_t* hs, const uint32_t* lo, const uint32_t* ord, int n, uint32_t* first) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    first[i] = (i == 0) || hs[i] != hs[i - 1] || lo[ord[i]] != lo[ord[i - 1]]; }
__global__ void k_snpval(const uint32_t* first, const uint32_t* ord, const uint8_t* alt, int n, uint64_t* v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) v[i] = first[i] ? (1ull | ((uint64_t)alt[ord[i]] << 32)) : 0ull; }
// R1 consensus (order-free-rules.md; SnpConsensus in umi_dedup.h): per molecule (cell, SNP, R9-merged UMIs) the reads' REF / ALT / OTHER counts,
// each saturating at 255 per exact UMI; a strict maximum decides (REF: DP; ALT: DP + AD; OTHER: nothing); a tie at the top is ambiguous (nothing).
__global__ void k_snpcls(const uint32_t* ord, const uint8_t* cls, int n, uint64_t* v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) v[i] = 1ull << (16 * cls[ord[i]]); }
// R9 (order-free-rules.md; UmiFeatureCollapse / SnpConsensus::finalize in umi_dedup.h): junction and SNP molecules are collapsed per
// (cell, feature) by the gene rule. Entries are the exact (cell, feature, UMI) runs: u[i] the UMI (base 5, N distinct), w[i] its reads.
// Two UMIs are joined iff they differ at exactly one base and max(w) >= 2 min(w) - 1; a molecule is a connected component. One thread
// per (cell, feature) group runs the union-find sequentially, so the result does not depend on scheduling; no output depends on the root.
__device__ __forceinline__ bool umi5_one_mismatch(uint32_t a, uint32_t b, int len) {
    int d = 0; for (int k = 0; k < len && d < 2; k++) { d += (a % 5) != (b % 5); a /= 5; b /= 5; } return d == 1; }
__global__ void k_unique_hilo(const uint32_t* first, const uint32_t* seg, const uint64_t* hs, const uint32_t* lo, const uint32_t* ord, int n, uint64_t* uh, uint32_t* ul) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n && first[i]) { uh[seg[i] - 1] = hs[i]; ul[seg[i] - 1] = lo[ord[i]]; } }
__global__ void k_gstart(const uint64_t* uh, int m, uint32_t* f) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < m) f[i] = (i == 0 || uh[i] != uh[i - 1]) ? 1u : 0u; }
// SNP class vector (16 bits per class) -> per-UMI counts clamped at 255 (SnpConsensus::add saturates), and the UMI's reads
__global__ void k_snpw(const uint64_t* mv, int m, uint32_t* w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= m) return; uint32_t s = 0; for (int k = 0; k < 3; k++) s += min((uint32_t)((mv[i] >> (16 * k)) & 0xFFFF), 255u); w[i] = s; }
// snp == 0: root[i] = 1 for component roots (molecules per group = their sum). snp == 1: the root's value is the R1 vote over the merged
// molecule (REF: DP; ALT: DP + AD), members 0; stat: molecules, ambiguous, other.
// rid (R9 per gene, order-free-rules.md): rid[i] != 0xFFFFFFFF marks an entry whose reads were assigned a gene and whose UMI that gene's molecule
// store holds; rid[i] is the store's component (the index of its root entry). Such entries are one molecule per rid (the gene reducer's collapse
// over every read of the gene in the cell); the other entries are collapsed among themselves (r9_molecules in umi_dedup.h). rid == nullptr: all.
__global__ void k_r9(const uint32_t* gs, int ng, int m, const uint32_t* u, const uint32_t* w, const uint64_t* mv, int ulen, int directional,
                     uint32_t* root, uint32_t* isroot, uint64_t* val, uint32_t* msum, unsigned long long* stat, const uint32_t* rid = nullptr) {
    int g = blockIdx.x * blockDim.x + threadIdx.x; if (g >= ng) return;
    const uint32_t s = gs[g], e = g + 1 < ng ? gs[g + 1] : (uint32_t)m;
    for (uint32_t i = s; i < e; i++) root[i] = i;
    auto find = [&](uint32_t x) { while (root[x] != x) { root[x] = root[root[x]]; x = root[x]; } return x; };
    auto free_ = [&](uint32_t i) { return rid == nullptr || rid[i] == 0xFFFFFFFFu; };
    if (directional && e - s > 1)
        for (uint32_t i = s; i + 1 < e; i++) if (free_(i)) for (uint32_t j = i + 1; j < e; j++) if (free_(j)) {
            const uint32_t hi = max(w[i], w[j]), lo = min(w[i], w[j]);
            if (hi + 1 >= 2 * lo && umi5_one_mismatch(u[i], u[j], ulen)) { uint32_t ri = find(i), rj = find(j); if (ri != rj) root[max(ri, rj)] = min(ri, rj); }
        }
    for (uint32_t i = s; i < e; i++) {
        if (free_(i)) { root[i] = find(i); continue; }
        uint32_t f = i; for (uint32_t j = s; j < i; j++) if (rid[j] == rid[i]) { f = j; break; }   // first entry of its gene component
        root[i] = f;
    }
    if (!val) { for (uint32_t i = s; i < e; i++) isroot[i] = root[i] == i; return; }
    for (uint32_t i = s; i < e; i++) { msum[3 * i] = msum[3 * i + 1] = msum[3 * i + 2] = 0; }
    for (uint32_t i = s; i < e; i++) for (int k = 0; k < 3; k++) msum[3 * root[i] + k] += min((uint32_t)((mv[i] >> (16 * k)) & 0xFFFF), 255u);
    for (uint32_t i = s; i < e; i++) {
        val[i] = 0; if (root[i] != i) continue;
        const uint32_t* c = msum + 3 * i; int best = 0; for (int k = 1; k < 3; k++) if (c[k] > c[best]) best = k;
        bool tie = false; for (int k = 0; k < 3; k++) if (k != best && c[k] == c[best]) tie = true;
        if (tie) { atomicAdd(&stat[1], 1ull); continue; }
        if (best == 2) { atomicAdd(&stat[2], 1ull); continue; }
        atomicAdd(&stat[0], 1ull); val[i] = 1ull | (best == 1 ? (1ull << 32) : 0ull);
    }
}
// E23 (SINGLET_GPU_PILEUP_PROF=1, profiling only): sizes of the groups a per-group serial kernel walks (one thread per group), from the group
// starts on the device; appended to the run's log (g_parts_log). Downloads the starts: not for production runs.
static bool g_prof = false;
static std::vector<std::string> g_prof_log;
static void prof_groups(const char* name, const uint32_t* d_starts, int ng, int m) {
    if (!g_prof || ng <= 0) return;
    std::vector<uint32_t> s(ng); CK(cudaMemcpy(s.data(), d_starts, (size_t)ng * 4, cudaMemcpyDeviceToHost));
    std::vector<uint32_t> z(ng); double sq = 0; long long n1k = 0, n10k = 0;
    for (int g = 0; g < ng; g++) { z[g] = (g + 1 < ng ? s[g + 1] : (uint32_t)m) - s[g]; sq += (double)z[g] * z[g]; n1k += z[g] >= 1000; n10k += z[g] >= 10000; }
    std::vector<uint32_t> t = z; const size_t k = std::min<size_t>(5, t.size()); std::partial_sort(t.begin(), t.begin() + k, t.end(), std::greater<uint32_t>());
    char b[320]; snprintf(b, sizeof b, "prof %s: %d groups, %d entries, max %u (top %u %u %u %u), sum n^2 %.3g, groups >= 1000: %lld, >= 10000: %lld",
                          name, ng, m, t[0], k > 1 ? t[1] : 0, k > 2 ? t[2] : 0, k > 3 ? t[3] : 0, k > 4 ? t[4] : 0, sq, n1k, n10k);
    g_prof_log.push_back(b);
}
// group starts of the m unique entries (sorted by uh): returns the number of groups, starts in *gs
static int group_starts(Arena& A, const uint64_t* uh, int m, uint32_t** gs) {
    const int TB = 256; uint32_t* f = A.alloc<uint32_t>(m); k_gstart<<<(m + TB - 1) / TB, TB>>>(uh, m, f);
    *gs = A.alloc<uint32_t>(m); int* dn = A.alloc<int>(1);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceSelect::Flagged(t, b, cub::CountingInputIterator<uint32_t>(0), f, *gs, dn, m); });
    int ng = 0; CK(cudaMemcpy(&ng, dn, 4, cudaMemcpyDeviceToHost)); return ng; }
static bool g_snp_first = false;   // --snp-first: the sorted path's first-read rule (prod/campaign), for the original gate
static unsigned long long g_snp_stat[3] = {0, 0, 0};
__global__ void k_mtres(const ReadRec* R, int n, const GRes* G, unsigned long long* c) { int r = blockIdx.x * blockDim.x + threadIdx.x; if (r < n && R[r].bc >= 0 && (G[R[r].sig].flags & 256u)) atomicAdd(&c[3], 1ull); }
// count_read's counters over the gene tuples: [0] exon_hits, [1] intron_hits (kReadIntron), [2] reads recorded with a UMI (umi_duplicate)
__global__ void k_hits(const Rec* r, int n, unsigned long long* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    atomicAdd(&c[(r[i].flags & 2) ? 1 : 0], 1ull); if (!r[i].direct) atomicAdd(&c[2], 1ull); }
// reads recorded in the molecule store per cell (per_cell_molecule_stats' total_reads): gene tuples with a UMI
__global__ void k_cellreads(const Rec* r, int n, uint32_t* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n && !r[i].direct) atomicAdd(&out[r[i].bc], 1u); }
// E16 (singlet's level 5, set by the engine wrapper): the per-alignment results leave the device as the three arrays the host keeps (gene, stat,
// exon gene: 9 bytes per alignment) instead of the 48-byte AA records split on the host
static bool g_e16_aasplit = false;
static bool g_e16_fetchpar = false;   // E16 "fetchpar" (singlet, level 5): a partition's outputs downloaded and converted on one thread each
__global__ void k_e16_aa_split(const AA* __restrict__ aa, int n, uint32_t* __restrict__ gene, uint8_t* __restrict__ stat, uint32_t* __restrict__ exg) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { gene[i] = aa[i].gene; stat[i] = (uint8_t) aa[i].stat; exg[i] = aa[i].exg; }
}
static bool g_cell_stats = false;   // set by the engine wrapper: per-cell reads and molecules for saturation / cell QC
struct SumU64 { __host__ __device__ uint64_t operator()(const uint64_t& a, const uint64_t& b) const { return a + b; } };
struct SumU32 { __host__ __device__ uint32_t operator()(const uint32_t& a, const uint32_t& b) const { return a + b; } };

static int bits_of(uint64_t v) { int b = 1; while (b < 64 && (1ull << b) <= v) b++; return b; }
static int total_of(Arena&, const uint32_t* cnt, const uint32_t* off, int n) {      // total from the last offset + count
    if (n == 0) return 0; uint32_t a = 0, b = 0; CK(cudaMemcpy(&a, cnt + n - 1, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&b, off + n - 1, 4, cudaMemcpyDeviceToHost)); return (int)(a + b); }
static uint32_t* scan_u32(Arena& A, const uint32_t* cnt, int n) { uint32_t* off = A.alloc<uint32_t>(std::max(n, 1)); if (n) cub_run(A, [&](void* t, size_t& b) { return cub::DeviceScan::ExclusiveSum(t, b, cnt, off, n); }); return off; }
// order of n elements sorted by (hi, lo), ties in input order (two stable LSD passes); returns ord, and the sorted hi array in *hs
static uint32_t* sort_hilo(Arena& A, const uint64_t* hi, const uint32_t* lo, int n, int hibits, int lobits, uint64_t** hs) {
    const int TB = 256, G = (n + TB - 1) / TB;
    uint32_t *l0 = A.alloc<uint32_t>(n), *l1 = A.alloc<uint32_t>(n), *i0 = A.alloc<uint32_t>(n), *i1 = A.alloc<uint32_t>(n);
    CK(cudaMemcpyAsync(l0, lo, (size_t)n * 4, cudaMemcpyDeviceToDevice)); k_iota<<<G, TB>>>(i0, n);
    cub::DoubleBuffer<uint32_t> dl(l0, l1), di(i0, i1);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, dl, di, n, 0, lobits); });
    uint64_t *h0 = A.alloc<uint64_t>(n), *h1 = A.alloc<uint64_t>(n); k_gather64<<<G, TB>>>(hi, di.Current(), n, h0);
    uint32_t* j1 = di.Alternate(); cub::DoubleBuffer<uint64_t> dh(h0, h1); cub::DoubleBuffer<uint32_t> dj(di.Current(), j1);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, dh, dj, n, 0, hibits); });
    *hs = dh.Current(); return dj.Current();
}

// ------------------------------------------------------------------------------------------------------------------------
struct Inputs {
    int nS = 0, nA = 0, nR = 0, ncells = 0, ulen = 12, wantMerges = 0; Par par; DPar dp;
    SigRec* S; AlnRec* Al; uint32_t* ops; uint8_t* seq; ReadRec* R; const uint64_t* Q = nullptr;   // Q: 16-byte low-quality mask per read (dp.use_q)
    DTree EX, IN; uint32_t *e2g, *i2g; char* gstr; DSnp SN; uint32_t ne2g = 0, ni2g = 0, ngstr = 0;
    uint8_t* gmt = nullptr; uint32_t ngmt = 0;   // per gene: lies on chrM (PileupInput::geneIsMt), for kMtPrefer
};
struct Outputs {
    struct Merge { uint32_t cell, gene; uint64_t umi, root; };
    std::vector<Merge> merges;   // gene store UMIs merged into another (Inputs::wantMerges)
    std::vector<T3> exon, intron, amb, sj, snp_ad, snp_dp, mt, mt_rev; std::vector<std::array<uint32_t, 4>> indels; std::vector<std::array<uint32_t, 3>> sjtab;   // sjtab: tid, donor, acc by id
    long long n_gene_tuples = 0, n_sjl = 0, n_sj_tuples = 0, n_snp_tuples = 0, n_mt_tuples = 0, n_ind_tuples = 0, n_junctions = 0;
    std::vector<uint32_t> cell_reads, cell_mol;   // per cell, when g_cell_stats
    unsigned long long hits[4] = {0, 0, 0, 0};      // exon_hits, intron_hits, UMI reads recorded, chrM/NUMT multimappers rescued (when g_cell_stats)
    std::vector<uint32_t> aa_gene, aa_exg; std::vector<uint8_t> aa_stat;   // per alignment (when g_cell_stats): classify_read gene, k_annotate stat
    unsigned long long sj_raw_sum = 0;   // distinct (cell, junction, UMI) before the 65535 clamp (the engine's stats.sj_hits)
};
static std::vector<T3> csr_to_t3(const Csr& c, int ncells) { std::vector<T3> v; v.reserve(c.indices.size()); for (int cell = 0; cell < ncells; cell++) for (uint32_t k = c.indptr[cell]; k < c.indptr[cell + 1]; k++) v.push_back(T3{(uint32_t)cell, c.indices[k], c.data[k]}); return v; }

// ---- cell partitions and up-front sizing (align-and-count design 4.4)
constexpr int kMaxParts = 64;
static const size_t kArenaReserve = size_t(1) << 30;    // headroom for CUB temporaries the per-read bound does not see
static int g_force_parts = 0;                            // SINGLET_GPU_PILEUP_PARTS: a fixed K (the partition-invariance gate)
static int g_parts_used = 1;
static std::vector<std::string> g_parts_log;
// bytes per read and per tuple of each reducer's stream in a partition: an upper bound from the allocations of steps 4-7 (each read:
// its record, flags, offsets and counts; each tuple: its emitted keys, the sort's double buffers and the collapse's per-tuple arrays)
constexpr uint64_t kBytesRead = 64, kBytesGene = 180, kBytesSj = 110, kBytesSnp = 150, kBytesMt = 34, kBytesInd = 24;   // job 792062: measured max 56 / 153 / 95 / 131 / 28 B, + ~15%
__host__ __device__ __forceinline__ uint32_t cell_part(int32_t bc, uint32_t K) {
    uint64_t x = (uint64_t)(uint32_t)bc + 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL; x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; x ^= x >> 31;
    return (uint32_t)(x % K);
}
__global__ void k_readbytes(const ReadRec* R, int n, const GRes* G, const uint32_t* c_sj, const uint32_t* c_snp, const uint32_t* c_mt,
                            const uint32_t* c_ind, uint64_t* w) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= n) return;
    const ReadRec x = R[r]; if (x.bc < 0) { w[r] = 0; return; }
    w[r] = kBytesRead + (G[x.sig].flags != 0 ? kBytesGene : 0) + c_sj[x.sig] * kBytesSj + c_snp[x.sig] * kBytesSnp + c_mt[x.sig] * kBytesMt + c_ind[x.sig] * kBytesInd;
}
__global__ void k_binsum(const ReadRec* R, int n, const uint64_t* w, uint32_t K, unsigned long long* bins) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= n || R[r].bc < 0) return;
    atomicAdd(&bins[cell_part(R[r].bc, K)], (unsigned long long)w[r]); }
__global__ void k_partflag(const ReadRec* R, int n, uint32_t K, uint32_t p, uint32_t* f) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r < n) f[r] = (R[r].bc >= 0 && cell_part(R[r].bc, K) == p) ? 1u : 0u; }

// R9 per gene: for the m unique (cell, feature, UMI) entries (run structure `seg` over the n tuples sorted by `ord`), the gene store component of
// each, from the smallest gene over the entry's reads (UmiFeatureCollapse::add); nullptr when there is no store
static uint32_t* gene_components(Arena& A, const GeneStore* gst, const uint32_t* tg, const uint32_t* ord, const uint32_t* seg, int n, const uint64_t* uh,
                                 const uint32_t* ul, int m, int hb, int ulen, int* dn) {
    if (!gst || !tg || m == 0) return nullptr;
    const int TB = 256;
    uint32_t *gsrt = A.alloc<uint32_t>(n), *segu = A.alloc<uint32_t>(m), *gmin = A.alloc<uint32_t>(m), *rid = A.alloc<uint32_t>(m);
    k_gather32<<<(n + TB - 1) / TB, TB>>>(tg, ord, n, gsrt);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceReduce::ReduceByKey(t, b, seg, segu, gsrt, gmin, dn + 1, cub::Min(), n); });
    k_gene_rid<<<(m + TB - 1) / TB, TB>>>(uh, ul, gmin, m, hb, ulen, *gst, rid);
    return rid;
}
// ---- the reducers: each takes its partition's tuples from the read-level expansion and returns its per-(cell, feature) result on the
// device. Order-free: radix sorts, integer reductions, and the R9 collapse one thread per group.
// gene: the M6 chain (stage b: (cell, gene, UMI) sort and reduce; c: directional UMI collapse; d: per class CSR)
static void red_gene(Arena& A, Timer& T, const Inputs& in, Rec* recs, int ng, CsrDev C[3], GeneStore* gst = nullptr) {
    Layout L; L.ulen = in.ulen; L.cb = bits_of(std::max<uint64_t>(in.ncells, 1)); L.gb = bits_of(std::max<uint64_t>(in.par.n_genes, 1)); L.ub = 2 * in.ulen; L.directional = in.par.directional != 0; L.ncells = in.ncells; L.cr = (in.dp.rules & 128) != 0 && in.par.umi_dedup;
    long long ne = 0, na = 0; int it = 0;
    C[0] = C[1] = C[2] = CsrDev{nullptr, nullptr, nullptr, 0};
    if (gst) *gst = GeneStore{nullptr, nullptr, 0, L.gb, L.ub};
    if (ng > 0) {
        uint64_t* uk; Agg* ag; int m = stage_b(A, T, recs, ng, 0, L, &uk, &ag); T.mark("b.end");
        const uint32_t* lab = nullptr;
        Mol M = stage_c(A, T, uk, ag, m, L, &ne, &na, &it, &lab); T.mark("c.end");
        if (gst) *gst = GeneStore{uk, lab, m, L.gb, L.ub};
        C[0] = stage_d_one(A, M.kex, m, L); C[1] = stage_d_one(A, M.kin, m, L); C[2] = stage_d_one(A, M.kam, m, L); T.mark("d.end");
    }
}
// junction: (cell, junction, UMI) sort, reads per exact UMI, R9 collapse per (cell, junction); returns the (cell | junction) rows
static int red_junction(Arena& A, Timer& T, const Inputs& in, const ReadRec* R, int nR, const uint32_t* r_sj, const uint32_t* o_sj, const uint32_t* c_sj,
                        const uint32_t* l_jid, int n_sjt, int jb, int cb, uint64_t** sj_u, uint32_t** sj_n, const GRes* G, const GeneStore* gst) {
    const int TB = 256, GR = (nR + TB - 1) / TB, ub5 = (int)ceil(in.ulen * 2.3219280948873623) + 1;
    int n_sju = 0;
    if (n_sjt > 0) {
        uint64_t* sj_hi = A.alloc<uint64_t>(n_sjt); uint32_t* sj_lo = A.alloc<uint32_t>(n_sjt);
        uint32_t* tg = gst ? A.alloc<uint32_t>(n_sjt) : nullptr;
        k_fill_hilo<<<GR, TB>>>(R, nR, r_sj, o_sj, c_sj, l_jid, 0, jb, in.ulen, sj_hi, sj_lo, nullptr, in.S, G, tg);
        T.mark("5.sj.fill");
        uint64_t* hs; uint32_t* ord = sort_hilo(A, sj_hi, sj_lo, n_sjt, cb + jb, ub5, &hs); const int G = (n_sjt + TB - 1) / TB;
        T.mark("5.sj.sort");
        uint32_t* first = A.alloc<uint32_t>(n_sjt); k_runflag<<<G, TB>>>(hs, sj_lo, ord, n_sjt, first);
        uint32_t* seg = A.alloc<uint32_t>(n_sjt);
        cub_run(A, [&](void* t, size_t& b) { return cub::DeviceScan::InclusiveSum(t, b, first, seg, n_sjt); });
        int m = 0; CK(cudaMemcpy(&m, seg + n_sjt - 1, 4, cudaMemcpyDeviceToHost));
        uint64_t* uh = A.alloc<uint64_t>(m); uint32_t *ul = A.alloc<uint32_t>(m), *segu = A.alloc<uint32_t>(m), *w = A.alloc<uint32_t>(m); int* dn = A.alloc<int>(4);
        k_unique_hilo<<<G, TB>>>(first, seg, hs, sj_lo, ord, n_sjt, uh, ul);
        cub_run(A, [&](void* t, size_t& b) { return cub::DeviceReduce::ReduceByKey(t, b, seg, segu, cub::ConstantInputIterator<uint32_t>(1), w, dn, SumU32(), n_sjt); });
        uint32_t* gs; int ngr = group_starts(A, uh, m, &gs);
        uint32_t *root = A.alloc<uint32_t>(m), *isroot = A.alloc<uint32_t>(m);
        uint32_t* rid = gene_components(A, gst, tg, ord, seg, n_sjt, uh, ul, m, jb, in.ulen, dn);
        T.mark("5.sj.runs");
        k_r9<<<(ngr + TB - 1) / TB, TB>>>(gs, ngr, m, ul, w, nullptr, in.ulen, in.par.directional, root, isroot, nullptr, nullptr, nullptr, rid);
        T.mark("5.sj.r9"); if (g_prof) { prof_groups("sj r9 (cell, junction)", gs, ngr, m); T.mark("prof"); }
        *sj_u = A.alloc<uint64_t>(m); *sj_n = A.alloc<uint32_t>(m);
        cub_run(A, [&](void* t, size_t& b) { return cub::DeviceReduce::ReduceByKey(t, b, uh, *sj_u, isroot, *sj_n, dn, SumU32(), m); });
        CK(cudaMemcpy(&n_sju, dn, 4, cudaMemcpyDeviceToHost));
    }
    T.mark("5.sj");
    return n_sju;
}
// SNP: (cell, SNP, UMI) sort, class counts per exact UMI, R9 collapse per (cell, SNP), R1 vote per molecule; rows (cell | SNP) -> DP | AD << 32
static int red_snp(Arena& A, Timer& T, const Inputs& in, const ReadRec* R, int nR, const uint32_t* r_snp, const uint32_t* o_snp, const uint32_t* c_snp,
                   const uint32_t* l_snp, int n_snpt, int sb, int cb, uint64_t** sn_u, uint64_t** sn_v, const GRes* G, const GeneStore* gst, const uint64_t* Qp = nullptr) {
    const int TB = 256, GR = (nR + TB - 1) / TB, ub5 = (int)ceil(in.ulen * 2.3219280948873623) + 1;
    int n_snu = 0;
    if (n_snpt > 0) {
        uint64_t* sn_hi = A.alloc<uint64_t>(n_snpt); uint32_t* sn_lo = A.alloc<uint32_t>(n_snpt); uint8_t* sn_alt = A.alloc<uint8_t>(n_snpt);
        uint32_t* tg = (gst && !g_snp_first) ? A.alloc<uint32_t>(n_snpt) : nullptr;
        k_fill_hilo<<<GR, TB>>>(R, nR, r_snp, o_snp, c_snp, l_snp, 1, sb, in.ulen, sn_hi, sn_lo, sn_alt, in.S, G, tg, Qp);
        T.mark("6.snp.fill");
        uint64_t* hs; uint32_t* ord = sort_hilo(A, sn_hi, sn_lo, n_snpt, cb + sb, ub5, &hs); const int G = (n_snpt + TB - 1) / TB;
        T.mark("6.snp.sort");
        uint32_t* first = A.alloc<uint32_t>(n_snpt); k_runflag<<<G, TB>>>(hs, sn_lo, ord, n_snpt, first);
        *sn_u = A.alloc<uint64_t>(n_snpt); *sn_v = A.alloc<uint64_t>(n_snpt); int* dn = A.alloc<int>(4);
        uint64_t* v = A.alloc<uint64_t>(n_snpt);
        if (g_snp_first) {
            k_snpval<<<G, TB>>>(first, ord, sn_alt, n_snpt, v);
            cub_run(A, [&](void* t, size_t& b) { return cub::DeviceReduce::ReduceByKey(t, b, hs, *sn_u, v, *sn_v, dn, SumU64(), n_snpt); });
        } else {
            k_snpcls<<<G, TB>>>(ord, sn_alt, n_snpt, v);
            uint32_t* seg = A.alloc<uint32_t>(n_snpt);
            cub_run(A, [&](void* t, size_t& b) { return cub::DeviceScan::InclusiveSum(t, b, first, seg, n_snpt); });
            int m = 0; CK(cudaMemcpy(&m, seg + n_snpt - 1, 4, cudaMemcpyDeviceToHost));
            uint64_t* mk = A.alloc<uint64_t>(m); uint64_t* mv = A.alloc<uint64_t>(m); uint32_t *segu = A.alloc<uint32_t>(m), *mu = A.alloc<uint32_t>(m);
            k_unique_hilo<<<G, TB>>>(first, seg, hs, sn_lo, ord, n_snpt, mk, mu);
            cub_run(A, [&](void* t, size_t& b) { return cub::DeviceReduce::ReduceByKey(t, b, seg, segu, v, mv, dn, SumU64(), n_snpt); });
            uint64_t* val = A.alloc<uint64_t>(m); unsigned long long* st = A.alloc<unsigned long long>(4); CK(cudaMemset(st, 0, 32));
            // R9: per (cell, SNP) the UMIs are collapsed (weight: the UMI's clamped class counts), then R1 votes over each merged molecule
            uint32_t* w = A.alloc<uint32_t>(m); k_snpw<<<(m + TB - 1) / TB, TB>>>(mv, m, w);
            uint32_t* gs; int ngr = group_starts(A, mk, m, &gs);
            uint32_t *root = A.alloc<uint32_t>(m), *msum = A.alloc<uint32_t>(3 * (size_t)m);
            uint32_t* rid = gene_components(A, gst, tg, ord, seg, n_snpt, mk, mu, m, sb, in.ulen, dn);
            T.mark("6.snp.runs");
            k_r9<<<(ngr + TB - 1) / TB, TB>>>(gs, ngr, m, mu, w, mv, in.ulen, in.par.directional, root, nullptr, val, msum, st, rid);
            T.mark("6.snp.r9"); if (g_prof) { prof_groups("snp r9 (cell, SNP)", gs, ngr, m); T.mark("prof"); }
            unsigned long long s[3]; CK(cudaMemcpy(s, st, 24, cudaMemcpyDeviceToHost));
            for (int k = 0; k < 3; k++) g_snp_stat[k] += s[k];
            cub_run(A, [&](void* t, size_t& b) { return cub::DeviceReduce::ReduceByKey(t, b, mk, *sn_u, val, *sn_v, dn, SumU64(), m); });
        }
        CK(cudaMemcpy(&n_snu, dn, 4, cudaMemcpyDeviceToHost));
    }
    T.mark("6.snp");
    return n_snu;
}
// chrM base: (cell, position, base, strand) sort and run-length encode
static int red_mt(Arena& A, Timer& T, const Inputs& in, const ReadRec* R, int nR, const uint32_t* r_mt, const uint32_t* o_mt, const uint32_t* c_mt,
                  const uint32_t* l_mt, int n_mtt, int cb, uint64_t** mt_u, uint32_t** mt_n, const uint64_t* Qp = nullptr) {
    const int TB = 256, GR = (nR + TB - 1) / TB;
    int n_mtu = 0;
    if (n_mtt > 0) {
        uint64_t* mt_key = A.alloc<uint64_t>(n_mtt); k_fill_mt<<<GR, TB>>>(R, in.Al, in.S, nR, r_mt, o_mt, c_mt, l_mt, mt_key, Qp);
        T.mark("7.mt.fill");
        uint64_t* k1 = A.alloc<uint64_t>(n_mtt); cub::DoubleBuffer<uint64_t> dk(mt_key, k1);
        cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRadixSort::SortKeys(t, b, dk, n_mtt, 0, cb + 18); });
        T.mark("7.mt.sort");
        *mt_u = A.alloc<uint64_t>(n_mtt); *mt_n = A.alloc<uint32_t>(n_mtt); int* dn = A.alloc<int>(4);
        cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRunLengthEncode::Encode(t, b, dk.Current(), *mt_u, *mt_n, dn, n_mtt); });
        CK(cudaMemcpy(&n_mtu, dn, 4, cudaMemcpyDeviceToHost));
        T.mark("7.mt.rle");
        if (g_prof) { char b[160]; snprintf(b, sizeof b, "prof mt: %d tuples, %d distinct (cell, position, base, strand), key bits %d", n_mtt, n_mtu, cb + 18); g_prof_log.push_back(b); }
    }
    return n_mtu;
}

// E23 (SINGLET_GPU_PILEUP_PROF=1): the gene store's (cell, gene) groups (stage_c's segments: k_cr_roots walks each on one thread) and the
// directional collapse's candidate buckets (segment, half UMI: k_edge_count's runs), from the sorted unique keys downloaded to the host
static void prof_gene_groups(const Inputs& in, const GeneStore& g) {
    if (!g_prof || g.m <= 0) return;
    std::vector<uint64_t> uk(g.m); CK(cudaMemcpy(uk.data(), g.uk, (size_t)g.m * 8, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> gmt(in.ngmt); if (in.ngmt) CK(cudaMemcpy(gmt.data(), in.gmt, in.ngmt, cudaMemcpyDeviceToHost));
    const uint64_t gm = (1ull << g.gb) - 1;
    struct Grp { uint32_t n; uint64_t key; }; std::vector<Grp> gr; long long mt_entries = 0, mt_groups = 0; double sq = 0, sqmt = 0;
    for (int i = 0; i < g.m;) { int j = i; while (j < g.m && (uk[j] >> g.ub) == (uk[i] >> g.ub)) j++; gr.push_back(Grp{(uint32_t)(j - i), uk[i] >> g.ub});
        const uint32_t gene = (uint32_t)((uk[i] >> g.ub) & gm); const bool mt = gene < gmt.size() && gmt[gene]; sq += (double)(j - i) * (j - i);
        if (mt) { mt_entries += j - i; mt_groups++; sqmt += (double)(j - i) * (j - i); } i = j; }
    std::vector<Grp> t = gr; const size_t k = std::min<size_t>(8, t.size());
    std::partial_sort(t.begin(), t.begin() + k, t.end(), [](const Grp& a, const Grp& b) { return a.n > b.n; });
    std::string top; for (size_t q = 0; q < k; q++) { const uint32_t gene = (uint32_t)(t[q].key & gm); char b[96];
        snprintf(b, sizeof b, "%s%u (cell %llu gene %u%s)", q ? ", " : "", t[q].n, (unsigned long long)(t[q].key >> g.gb), gene, gene < gmt.size() && gmt[gene] ? " chrM" : ""); top += b; }
    // the directional collapse's candidate buckets: (segment, low half) and (segment, high half) of the UMI; k_edge_count does b(b-1)/2 per bucket
    const int lob = 2 * (in.ulen / 2); double bsq[2] = {0, 0}; uint32_t bmx[2] = {0, 0};
    for (int h = 0; h < 2; h++) {
        std::vector<uint64_t> kk(g.m); size_t seg = 0; for (int i = 0; i < g.m; i++) { if (i && (uk[i] >> g.ub) != (uk[i - 1] >> g.ub)) seg++;
            const uint64_t u = uk[i] & ((1ull << g.ub) - 1); kk[i] = (seg << 32) | (h == 0 ? (u & ((1ull << lob) - 1)) : (u >> lob)); }
        std::sort(kk.begin(), kk.end());
        for (size_t i = 0; i < kk.size();) { size_t j = i; while (j < kk.size() && kk[j] == kk[i]) j++; bsq[h] += (double)(j - i) * (j - i - 1) / 2; bmx[h] = std::max(bmx[h], (uint32_t)(j - i)); i = j; } }
    char b[700]; snprintf(b, sizeof b, "prof gene (cell, gene) groups: %zu groups, %d entries (UMIs), sum n^2 %.3g; chrM genes %lld groups, %lld entries, sum n^2 %.3g; "
                          "largest %s; edge buckets (low/high half): max %u/%u, pairs %.3g/%.3g",
                          gr.size(), g.m, sq, mt_groups, mt_entries, sqmt, top.c_str(), bmx[0], bmx[1], bsq[0], bsq[1]);
    g_prof_log.push_back(b);
}

static void run_pipeline(Arena& A, Timer& T, const Inputs& in, Outputs* out) {
    const int TB = 256; const int nS = in.nS, nA = in.nA, nR = in.nR;
    { const char* pf = getenv("SINGLET_GPU_PILEUP_PROF"); g_prof = pf && *pf && *pf != '0'; g_prof_log.clear(); }
    // 1 annotate
    AA* aa = A.alloc<AA>(std::max(nA, 1));
    k_annotate<<<(nA + TB - 1) / TB, TB>>>(in.Al, nA, in.ops, in.EX, in.IN, in.e2g, in.ne2g, in.i2g, in.ni2g, in.gstr, in.ngstr, in.dp, aa); T.mark("1.annotate");
    // 2 signature level
    GRes* gr = A.alloc<GRes>(nS); k_resolve<<<(nS + TB - 1) / TB, TB>>>(in.S, nS, in.Al, aa, in.gmt, in.ngmt, in.dp, gr);
    uint32_t *c_sj = A.alloc<uint32_t>(nS), *c_snp = A.alloc<uint32_t>(nS), *c_mt = A.alloc<uint32_t>(nS), *c_ind = A.alloc<uint32_t>(nS);
    k_lists<false><<<(nS + TB - 1) / TB, TB>>>(in.S, nS, in.Al, in.ops, in.seq, in.SN, in.dp, c_sj, c_snp, c_mt, c_ind, nullptr, nullptr, nullptr, nullptr);
    uint32_t *o_sj = scan_u32(A, c_sj, nS), *o_snp = scan_u32(A, c_snp, nS), *o_mt = scan_u32(A, c_mt, nS), *o_ind = scan_u32(A, c_ind, nS);
    int n_sjl = total_of(A, c_sj, o_sj, nS), n_snpl = total_of(A, c_snp, o_snp, nS), n_mtl = total_of(A, c_mt, o_mt, nS), n_indl = total_of(A, c_ind, o_ind, nS);
    SJL* l_sj = A.alloc<SJL>(std::max(n_sjl, 1)); uint32_t* l_snp = A.alloc<uint32_t>(std::max(n_snpl, 1)); uint32_t* l_mt = A.alloc<uint32_t>(std::max(n_mtl, 1)); IND* l_ind = A.alloc<IND>(std::max(n_indl, 1));
    k_lists<true><<<(nS + TB - 1) / TB, TB>>>(in.S, nS, in.Al, in.ops, in.seq, in.SN, in.dp, o_sj, o_snp, o_mt, o_ind, l_sj, l_snp, l_mt, l_ind); T.mark("2.siglists");
    // 3 junction ids
    uint32_t* l_jid = A.alloc<uint32_t>(std::max(n_sjl, 1)); uint64_t *t1 = nullptr; uint32_t* t2 = nullptr; int nj = 0;
    if (n_sjl > 0) {
        const int n = n_sjl, G = (n + TB - 1) / TB; uint64_t* k1 = A.alloc<uint64_t>(n); uint32_t *k2 = A.alloc<uint32_t>(n), *idx = A.alloc<uint32_t>(n);
        k_sjkey<<<G, TB>>>(l_sj, n, k1, k2, idx);
        uint32_t *a0 = A.alloc<uint32_t>(n), *a1 = A.alloc<uint32_t>(n), *i1 = A.alloc<uint32_t>(n); CK(cudaMemcpyAsync(a0, k2, (size_t)n * 4, cudaMemcpyDeviceToDevice));
        cub::DoubleBuffer<uint32_t> dk(a0, a1), di(idx, i1); cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, dk, di, n, 0, 32); });
        uint64_t *g0 = A.alloc<uint64_t>(n), *g1 = A.alloc<uint64_t>(n); k_gather64<<<G, TB>>>(k1, di.Current(), n, g0);
        cub::DoubleBuffer<uint64_t> dg(g0, g1); cub::DoubleBuffer<uint32_t> dj(di.Current(), di.Alternate());
        cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, dg, dj, n, 0, 64); });
        uint32_t *nf = A.alloc<uint32_t>(n), *rk = A.alloc<uint32_t>(n); k_newflag2<<<G, TB>>>(k1, k2, dj.Current(), n, nf);
        cub_run(A, [&](void* t, size_t& b) { return cub::DeviceScan::InclusiveSum(t, b, nf, rk, n); });
        CK(cudaMemcpy(&nj, rk + n - 1, 4, cudaMemcpyDeviceToHost));
        k_jid<<<G, TB>>>(dj.Current(), rk, n, l_jid);
        t1 = A.alloc<uint64_t>(nj); t2 = A.alloc<uint32_t>(nj); k_sjtab<<<G, TB>>>(k1, k2, dj.Current(), nf, rk, n, t1, t2);
    }
    T.mark("3.sjids");
    // per cell: reads and molecules recorded (saturation / cell QC), and count_read's counters; every partition adds to them
    uint32_t* cell_reads = nullptr; uint32_t* cell_mol = nullptr; unsigned long long* hits = nullptr;
    if (g_cell_stats) {
        cell_reads = A.alloc<uint32_t>(std::max(in.ncells, 1)); cell_mol = A.alloc<uint32_t>(std::max(in.ncells, 1));
        CK(cudaMemset(cell_reads, 0, std::max(in.ncells, 1) * 4)); CK(cudaMemset(cell_mol, 0, std::max(in.ncells, 1) * 4));
        hits = A.alloc<unsigned long long>(4); CK(cudaMemset(hits, 0, 32));
        CK(cudaMemcpyToSymbol(g_cell_mol, &cell_mol, sizeof(cell_mol)));
    }
    g_snp_stat[0] = g_snp_stat[1] = g_snp_stat[2] = 0;
    // ---- cell partitions (align-and-count design 4.4): every reducer's key starts with the cell, so the read-level stages run once
    // per partition of the cells (hash(cell) mod K) on that partition's reads; the signature-level tables above are shared. Sized up
    // front: the smallest K whose largest partition fits the arena (part_bytes: an upper bound per read from its tuple counts).
    const int GR0 = (nR + TB - 1) / TB;
    const size_t base = A.mark();
    int K = 1;
    std::vector<unsigned long long> est;
    {
        uint64_t* w = A.alloc<uint64_t>(std::max(nR, 1));
        k_readbytes<<<GR0, TB>>>(in.R, nR, gr, c_sj, c_snp, c_mt, c_ind, w);
        unsigned long long* bins = A.alloc<unsigned long long>(kMaxParts);
        const size_t budget = A.cap > base + kArenaReserve ? A.cap - base - kArenaReserve : 0;
        const int forced = g_force_parts;
        for (K = forced > 0 ? forced : 1; K <= kMaxParts; K++) {
            CK(cudaMemset(bins, 0, K * 8));
            if (nR > 0) k_binsum<<<GR0, TB>>>(in.R, nR, w, (uint32_t)K, bins);
            est.assign(K, 0); CK(cudaMemcpy(est.data(), bins, K * 8, cudaMemcpyDeviceToHost));
            const unsigned long long mx = *std::max_element(est.begin(), est.end());
            // K > 1 also holds the partition scratch for all reads (flags, compacted reads, their index: 24 B per read)
            const size_t scratch = K > 1 ? (size_t)nR * (4 + sizeof(ReadRec) + 4 + (in.dp.use_q ? 16 : 0)) : 0;
            if (forced > 0 || mx + scratch <= budget) break;
        }
        if (K > kMaxParts) K = kMaxParts;   // nothing fits: the largest K, and the unified-memory retry catches the rest
        A.reset(base);
    }
    g_parts_used = K; g_parts_log.clear();
    uint32_t* pf = nullptr; ReadRec* Rp = nullptr; uint32_t* ridx = nullptr; int* dnp = nullptr; uint64_t* Qg = nullptr;
    const size_t base2 = [&] {   // K > 1: partition scratch that outlives each partition (flags, compacted reads, their global index)
        if (K > 1) { pf = A.alloc<uint32_t>(std::max(nR, 1)); Rp = A.alloc<ReadRec>(std::max(nR, 1)); ridx = A.alloc<uint32_t>(std::max(nR, 1)); dnp = A.alloc<int>(1); if (in.dp.use_q) Qg = A.alloc<uint64_t>(2 * (size_t)std::max(nR, 1)); }
        return A.mark(); }();
    std::vector<std::pair<uint32_t, std::array<uint32_t, 4>>> indels;   // (global read index, indel): read order, as for K = 1
    const int jb = bits_of(std::max(nj, 1)), sb = bits_of(std::max<uint64_t>(in.par.n_snps, 1)), cb = bits_of(std::max<uint64_t>(in.ncells, 1));
    std::vector<T3>* outv[8] = {nullptr};   // the matrices, and where each partition's rows start in them (the merge below)
    std::vector<size_t> bnd[8];
    if (out) { std::vector<T3>* o8[8] = {&out->exon, &out->intron, &out->amb, &out->sj, &out->snp_ad, &out->snp_dp, &out->mt, &out->mt_rev}; for (int v = 0; v < 8; v++) outv[v] = o8[v]; }
    long long ng_all = 0, n_sjt_all = 0, n_snpt_all = 0, n_mtt_all = 0, n_indt_all = 0;
    size_t peak_all = A.peak;   // each partition measures its own peak; the run reports the largest
    for (int part = 0; part < K; part++) {
        A.reset(base2); A.peak = base2;
        int nRp = 0; const uint64_t* Qp = nullptr;   // Qp: the partition's per-read low-quality masks (dp.use_q)
        if (K == 1) { Rp = in.R; nRp = nR; Qp = in.dp.use_q ? in.Q : nullptr; }   // one partition: the reads as they are (the kernels skip reads without a barcode)
        else if (nR > 0) {
            k_partflag<<<GR0, TB>>>(in.R, nR, (uint32_t)K, (uint32_t)part, pf);
            cub_run(A, [&](void* t, size_t& b) { return cub::DeviceSelect::Flagged(t, b, in.R, pf, Rp, dnp, nR); });
            cub_run(A, [&](void* t, size_t& b) { return cub::DeviceSelect::Flagged(t, b, cub::CountingInputIterator<uint32_t>(0), pf, ridx, dnp, nR); });
            CK(cudaMemcpy(&nRp, dnp, 4, cudaMemcpyDeviceToHost));
            if (in.dp.use_q && nRp > 0) { k_gatherq<<<(nRp + TB - 1) / TB, TB>>>(in.Q, ridx, nRp, Qg); Qp = Qg; }
        }
        T.mark("part.select");
        const int GR = (nRp + TB - 1) / TB;
        const size_t s0 = A.mark();
        // 4 expand to read level: gene tuples + counts / offsets of the other streams
        uint32_t* gf = A.alloc<uint32_t>(std::max(nRp, 1)); k_gflag<<<GR, TB>>>(Rp, nRp, gr, gf);
        uint32_t* goff = scan_u32(A, gf, nRp); int ng = total_of(A, gf, goff, nRp);
        Rec* recs = A.alloc<Rec>(std::max(ng, 1)); k_gfill<<<GR, TB>>>(Rp, nRp, gr, gf, goff, in.ulen, recs);
        if (g_cell_stats && ng > 0) { k_cellreads<<<(ng + TB - 1) / TB, TB>>>(recs, ng, cell_reads); k_hits<<<(ng + TB - 1) / TB, TB>>>(recs, ng, hits); }
        if (g_cell_stats && (in.dp.rules & 256)) k_mtres<<<GR, TB>>>(Rp, nRp, gr, hits);
        uint32_t* rc = A.alloc<uint32_t>(std::max(nRp, 1));
        k_cnt<<<GR, TB>>>(Rp, nRp, c_sj, rc); uint32_t* r_sj = scan_u32(A, rc, nRp); int n_sjt = total_of(A, rc, r_sj, nRp);
        if (Qp) k_cnt_q<<<GR, TB>>>(Rp, nRp, o_snp, c_snp, l_snp, Qp, 25, rc); else k_cnt<<<GR, TB>>>(Rp, nRp, c_snp, rc); uint32_t* r_snp = scan_u32(A, rc, nRp); int n_snpt = total_of(A, rc, r_snp, nRp);
        if (Qp) k_cnt_q<<<GR, TB>>>(Rp, nRp, o_mt, c_mt, l_mt, Qp, 17, rc); else k_cnt<<<GR, TB>>>(Rp, nRp, c_mt, rc); uint32_t* r_mt = scan_u32(A, rc, nRp); int n_mtt = total_of(A, rc, r_mt, nRp);
        k_cnt<<<GR, TB>>>(Rp, nRp, c_ind, rc); uint32_t* r_ind = scan_u32(A, rc, nRp); int n_indt = total_of(A, rc, r_ind, nRp);
        const size_t s4 = A.mark();
        T.mark("4.expand");
        // the reducers: gene (M6 chain), junction, SNP, chrM base, chrM indel; each emits, sorts, collapses and reduces its own stream
        CsrDev C[3]; GeneStore gst{nullptr, nullptr, 0, 0, 0}; red_gene(A, T, in, recs, ng, C, &gst);
        const GeneStore* gsp = (in.par.umi_dedup && gst.m > 0) ? &gst : nullptr;   // R9 per gene: the junction and SNP molecules take their roots from it
        if (g_prof) { T.mark("gene"); prof_gene_groups(in, gst); T.mark("prof"); }
        if (out && in.wantMerges && gst.m > 0) {   // the host V(D)J reducer needs the same roots: the entries the collapse merged into another
            const int Gm = (gst.m + TB - 1) / TB; uint32_t* mf = A.alloc<uint32_t>(gst.m); k_mflag<<<Gm, TB>>>(gst.lab, gst.m, mf);
            uint32_t* midx = A.alloc<uint32_t>(gst.m); int* mn = A.alloc<int>(1);
            cub_run(A, [&](void* t, size_t& b) { return cub::DeviceSelect::Flagged(t, b, cub::CountingInputIterator<uint32_t>(0), mf, midx, mn, gst.m); });
            int nm = 0; CK(cudaMemcpy(&nm, mn, 4, cudaMemcpyDeviceToHost));
            if (nm > 0) {
                uint64_t *ma = A.alloc<uint64_t>(nm), *mb = A.alloc<uint64_t>(nm); k_mgather<<<(nm + TB - 1) / TB, TB>>>(gst.uk, gst.lab, midx, nm, ma, mb);
                std::vector<uint64_t> ha(nm), hb2(nm); CK(cudaMemcpy(ha.data(), ma, (size_t)nm * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hb2.data(), mb, (size_t)nm * 8, cudaMemcpyDeviceToHost));
                const uint64_t um = (1ull << gst.ub) - 1, gm = (1ull << gst.gb) - 1;
                for (int i = 0; i < nm; i++) out->merges.push_back({(uint32_t)(ha[i] >> (gst.gb + gst.ub)), (uint32_t)((ha[i] >> gst.ub) & gm), ha[i] & um, hb2[i] & um});
            }
        }
        const size_t s5 = A.mark();
        uint64_t* sj_u = nullptr; uint32_t* sj_n = nullptr; int n_sju = red_junction(A, T, in, Rp, nRp, r_sj, o_sj, c_sj, l_jid, n_sjt, jb, cb, &sj_u, &sj_n, gr, gsp);
        const size_t s6 = A.mark();
        uint64_t* sn_u = nullptr; uint64_t* sn_v = nullptr; int n_snu = red_snp(A, T, in, Rp, nRp, r_snp, o_snp, c_snp, l_snp, n_snpt, sb, cb, &sn_u, &sn_v, gr, gsp, Qp);
        const size_t s7 = A.mark();
        uint64_t* mt_u = nullptr; uint32_t* mt_n = nullptr; int n_mtu = red_mt(A, T, in, Rp, nRp, r_mt, o_mt, c_mt, l_mt, n_mtt, cb, &mt_u, &mt_n, Qp);
        uint32_t* in_bc = A.alloc<uint32_t>(std::max(n_indt, 1)); IND* in_ind = A.alloc<IND>(std::max(n_indt, 1)); uint32_t* in_ri = A.alloc<uint32_t>(std::max(n_indt, 1));
        k_fill_ind<<<GR, TB>>>(Rp, nRp, r_ind, o_ind, c_ind, l_ind, in_bc, in_ind, ridx, in_ri);
        T.mark("7.mt");
        {   // what the sizing assumed for this partition against what it used (bytes above the shared tables)
            char b[320];
            snprintf(b, sizeof b, "part %d/%d: %d reads, tuples gene %d sj %d snp %d mt %d ind %d; est %.2f GB, used %.2f GB (expand %.2f gene %.2f sj %.2f snp %.2f mt+ind %.2f)",
                     part, K, nRp, ng, n_sjt, n_snpt, n_mtt, n_indt, (K <= (int)est.size() ? est[part] : 0) / 1e9, (A.peak - base) / 1e9,
                     (s4 - s0) / 1e9, (s5 - s4) / 1e9, (s6 - s5) / 1e9, (s7 - s6) / 1e9, (A.mark() - s7) / 1e9);
            g_parts_log.push_back(b);
        }
        peak_all = std::max(peak_all, A.peak);
        ng_all += ng; n_sjt_all += n_sjt; n_snpt_all += n_snpt; n_mtt_all += n_mtt; n_indt_all += n_indt;
        if (!out) continue;
        // ---------------- fetch this partition's cells (host; its span is marked "fetch" and left out of the GPU time) ----------------
        for (int v = 0; v < 8; v++) bnd[v].push_back(outv[v]->size());
        // (E16: the steps as lambdas, called in this order, or with g_e16_fetchpar on threads: each fills its own outputs in the same order)
        auto add = [&](std::vector<T3>& v, const CsrDev& c) { std::vector<T3> t = csr_to_t3(fetch_csr(c, in.ncells), in.ncells);
                                                              if (g_e16_fetchpar && v.empty()) v.swap(t); else v.insert(v.end(), t.begin(), t.end()); };
        auto fSj = [&]() {
        if (n_sju > 0) { std::vector<uint64_t> k(n_sju); std::vector<uint32_t> c(n_sju); CK(cudaMemcpy(k.data(), sj_u, (size_t)n_sju * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(c.data(), sj_n, (size_t)n_sju * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < n_sju; i++) out->sj_raw_sum += c[i];
            for (int i = 0; i < n_sju; i++) out->sj.push_back(T3{(uint32_t)(k[i] >> jb), (uint32_t)(k[i] & ((1ull << jb) - 1)), std::min(c[i], 65535u)}); }
        };
        auto fSn = [&]() {
        if (n_snu > 0) { std::vector<uint64_t> k(n_snu), v(n_snu); CK(cudaMemcpy(k.data(), sn_u, (size_t)n_snu * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(v.data(), sn_v, (size_t)n_snu * 8, cudaMemcpyDeviceToHost));
            for (int i = 0; i < n_snu; i++) { T3 a{(uint32_t)(k[i] >> sb), (uint32_t)(k[i] & ((1ull << sb) - 1)), 0}; uint32_t dp = (uint32_t)(v[i] & 0xFFFFFFFFu), ad = (uint32_t)(v[i] >> 32);
                if (dp == 0) continue; T3 d = a; d.val = std::min(dp, 65535u); out->snp_dp.push_back(d); if (ad > 0) { a.val = std::min(ad, 65535u); out->snp_ad.push_back(a); } } }
        };
        auto fMt = [&]() {
        if (n_mtu > 0) { std::vector<uint64_t> k(n_mtu); std::vector<uint32_t> c(n_mtu); CK(cudaMemcpy(k.data(), mt_u, (size_t)n_mtu * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(c.data(), mt_n, (size_t)n_mtu * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < n_mtu;) {            // key = cell << 18 | feat << 1 | rev: entries (cell, feat, 0) and (cell, feat, 1) are adjacent
                uint64_t bk = k[i] >> 1; uint64_t tot = 0, rv = 0; int j = i;
                while (j < n_mtu && (k[j] >> 1) == bk) { tot += c[j]; if (k[j] & 1) rv += c[j]; j++; }
                uint32_t cell = (uint32_t)(bk >> 17), feat = (uint32_t)(bk & ((1u << 17) - 1));
                out->mt.push_back(T3{cell, feat, (uint32_t)std::min<uint64_t>(tot, 65535)}); if (rv > 0) out->mt_rev.push_back(T3{cell, feat, (uint32_t)std::min<uint64_t>(rv, 65535)}); i = j; } }
        };
        auto fInd = [&]() {
        if (n_indt > 0) { std::vector<uint32_t> b(n_indt), ri(n_indt); std::vector<IND> d(n_indt); CK(cudaMemcpy(b.data(), in_bc, (size_t)n_indt * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(d.data(), in_ind, (size_t)n_indt * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ri.data(), in_ri, (size_t)n_indt * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < n_indt; i++) indels.push_back({ri[i], {b[i], d[i].pos, d[i].lenins >> 1, d[i].lenins & 1}}); }
        };
        if (g_e16_fetchpar) {
            int dev = 0; cudaGetDevice(&dev);
            std::vector<std::thread> fth;
            auto run = [&](std::function<void()> f) { fth.emplace_back([dev, f]() { cudaSetDevice(dev); f(); }); };
            if (ng > 0) { run([&]() { add(out->exon, C[0]); }); run([&]() { add(out->intron, C[1]); }); run([&]() { add(out->amb, C[2]); }); }
            run(fSj); run(fSn); run(fMt); run(fInd);
            for (auto& t : fth) t.join();
        } else {
            if (ng > 0) { add(out->exon, C[0]); add(out->intron, C[1]); add(out->amb, C[2]); }
            fSj(); fSn(); fMt(); fInd();
        }
        T.mark("fetch");
    }
    A.reset(base2); A.peak = peak_all;
    T.finish();
    {   // E23: the GPU time per stage (ms, every partition summed; "fetch" is the host)
        std::string st = "stages (ms):"; char b[64];
        for (const auto& kv : T.acc) { snprintf(b, sizeof b, " %s %.1f", kv.first.c_str(), kv.second); st += b; }
        g_parts_log.push_back(st);
        for (auto& l : g_prof_log) g_parts_log.push_back(l);
    }
    if (g_cell_stats) { uint32_t* z = nullptr; CK(cudaMemcpyToSymbol(g_cell_mol, &z, sizeof(z))); }
    if (!out) return;
    // ---- canonical merge: cells are disjoint across partitions; each matrix is sorted by (cell, row), indels by read (the K = 1 order)
    // each partition's rows are sorted already: merge them (K = 1: nothing to do)
    auto byCell = [](const T3& a, const T3& b) { return a.cell != b.cell ? a.cell < b.cell : a.row < b.row; };
    for (int v = 0; v < 8; v++)
        for (size_t p = 2; p <= bnd[v].size(); p++) {
            const size_t mid = bnd[v][p - 1], end = p < bnd[v].size() ? bnd[v][p] : outv[v]->size();
            std::inplace_merge(outv[v]->begin(), outv[v]->begin() + mid, outv[v]->begin() + end, byCell);
        }
    if (K > 1) std::stable_sort(indels.begin(), indels.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (auto& e : indels) out->indels.push_back(e.second);
    const double e16m0 = now_s();   // E16: the host side of the final downloads (per-alignment results, junction table)
    if (g_cell_stats) {
        out->cell_reads.resize(std::max(in.ncells, 1)); out->cell_mol.resize(std::max(in.ncells, 1));
        CK(cudaMemcpy(out->cell_reads.data(), cell_reads, out->cell_reads.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(out->cell_mol.data(), cell_mol, out->cell_mol.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(out->hits, hits, 32, cudaMemcpyDeviceToHost));
        out->aa_gene.resize(nA); out->aa_stat.resize(nA); out->aa_exg.resize(nA);
        uint8_t* dsplit = nullptr;   // E16: outside the arena (its 3 GB margin), so the arena's layout is unchanged; without it, the host split
        if (g_e16_aasplit && nA > 0 && cudaMalloc(&dsplit, (size_t) nA * 9) != cudaSuccess) { cudaGetLastError(); dsplit = nullptr; }
        if (dsplit != nullptr) {
            uint32_t* dg = (uint32_t*) dsplit; uint32_t* de = dg + nA; uint8_t* ds = (uint8_t*) (de + nA);
            k_e16_aa_split<<<(nA + 255) / 256, 256>>>(aa, nA, dg, ds, de);
            CK(cudaGetLastError());
            CK(cudaMemcpy(out->aa_gene.data(), dg, (size_t) nA * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(out->aa_exg.data(), de, (size_t) nA * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(out->aa_stat.data(), ds, (size_t) nA, cudaMemcpyDeviceToHost));
            CK(cudaFree(dsplit));
        } else {
        std::vector<AA> h(nA);
        if (nA) CK(cudaMemcpy(h.data(), aa, (size_t)nA * sizeof(AA), cudaMemcpyDeviceToHost));
        for (int k = 0; k < nA; k++) { out->aa_gene[k] = h[k].gene; out->aa_stat[k] = (uint8_t)h[k].stat; out->aa_exg[k] = h[k].exg; }
        }
    }
    { char b[96]; snprintf(b, sizeof b, "e16: merge + per-alignment D2H %.3f s", now_s() - e16m0); g_parts_log.push_back(b); }
    out->n_gene_tuples = ng_all; out->n_sjl = n_sjl; out->n_sj_tuples = n_sjt_all; out->n_snp_tuples = n_snpt_all; out->n_mt_tuples = n_mtt_all; out->n_ind_tuples = n_indt_all; out->n_junctions = nj;
    if (nj > 0) { std::vector<uint64_t> h1(nj); std::vector<uint32_t> h2(nj); CK(cudaMemcpy(h1.data(), t1, (size_t)nj * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h2.data(), t2, (size_t)nj * 4, cudaMemcpyDeviceToHost));
        for (int i = 0; i < nj; i++) out->sjtab.push_back({(uint32_t)(h1[i] >> 32), (uint32_t)(h1[i] & 0xFFFFFFFFu), h2[i]}); }

}

// ------------------------------------------------------------------------------------------------------------------------
struct Cmp { long long identical = 0, only_gpu = 0, only_cpu = 0, value_diff = 0, gpu_nnz = 0, cpu_nnz = 0; unsigned long long gpu_sum = 0, cpu_sum = 0; };
static Cmp cmp_t3(const std::vector<T3>& g, const std::vector<T3>& c) {      // both sorted by (cell, row)
    Cmp r; r.gpu_nnz = g.size(); r.cpu_nnz = c.size(); size_t i = 0, j = 0;
    for (auto& x : g) r.gpu_sum += x.val; for (auto& x : c) r.cpu_sum += x.val;
    while (i < g.size() || j < c.size()) {
        if (i < g.size() && j < c.size() && g[i].cell == c[j].cell && g[i].row == c[j].row) { if (g[i].val == c[j].val) r.identical++; else r.value_diff++; i++; j++; }
        else if (j >= c.size() || (i < g.size() && (g[i].cell < c[j].cell || (g[i].cell == c[j].cell && g[i].row < c[j].row)))) { r.only_gpu++; i++; }
        else { r.only_cpu++; j++; } }
    return r; }
static bool is_identical(const Cmp& c) { return c.only_gpu == 0 && c.only_cpu == 0 && c.value_diff == 0 && c.gpu_nnz == c.cpu_nnz; }
static std::string cmp_json(const char* name, const Cmp& c) {
    char b[600]; snprintf(b, sizeof b, "\"%s\": {\"identical_entries\": %lld, \"only_gpu\": %lld, \"only_cpu\": %lld, \"value_diff\": %lld, \"gpu_nnz\": %lld, \"cpu_nnz\": %lld, \"gpu_sum\": %llu, \"cpu_sum\": %llu, \"identical\": %s}",
        name, c.identical, c.only_gpu, c.only_cpu, c.value_diff, c.gpu_nnz, c.cpu_nnz, c.gpu_sum, c.cpu_sum, is_identical(c) ? "true" : "false"); return b; }
static std::string sha_t3(const std::vector<T3>& v) { Sha256 s; s.update(v.data(), v.size() * sizeof(T3)); return s.hex(); }
static std::string sha_out(const Outputs& o) {
    Sha256 s; for (const std::vector<T3>* v : {&o.exon, &o.intron, &o.amb, &o.sj, &o.snp_ad, &o.snp_dp, &o.mt, &o.mt_rev}) s.update(v->data(), v->size() * sizeof(T3));
    s.update(o.sjtab.data(), o.sjtab.size() * 12); return s.hex(); }
struct SjKey { uint32_t cell, tid, donor, acc, val; bool operator<(const SjKey& o) const { return std::tie(cell, tid, donor, acc) < std::tie(o.cell, o.tid, o.donor, o.acc); } };

#ifndef PILEUP_NO_MAIN   // defined when linked into singlet (src/engine_gpu/pileup_gpu.cu)
int main(int argc, char** argv) {
    std::string rec, dump, label = "pileup", json; int reps = 3; double arena_gb = 80;
    for (int i = 1; i < argc; i++) { std::string a = argv[i]; auto nx = [&]() { if (i + 1 >= argc) die("missing value"); return std::string(argv[++i]); };
        if (a == "--rec") rec = nx(); else if (a == "--dump") dump = nx(); else if (a == "--label") label = nx(); else if (a == "--json") json = nx(); else if (a == "--reps") reps = atoi(nx().c_str());
        else if (a == "--arena-gb") arena_gb = atof(nx().c_str());
        else if (a == "--snp-first") g_snp_first = true;   // prod/campaign's first-read rule instead of the R1 consensus
        else die(("unknown arg " + a).c_str()); }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0)); fprintf(stderr, "device %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    Par par = load_par(dump + "/params.txt"); Anno an = load_anno(dump + "/anno.bin"); CpuTruth tr = load_cpu_truth(dump + "/truth.bin"); auto chroms = read_lines(dump + "/chroms.txt");
    { uint32_t mx = 0; for (auto& v : an.snp) for (auto& sr : v) mx = std::max(mx, sr.idx + 1); par.n_snps = std::max(par.n_snps, mx); }
    auto meta = read_kv(rec + "/meta.txt");
    std::vector<SigRec> hS = read_vec<SigRec>(rec + "/sigs.bin"); std::vector<AlnRec> hA = read_vec<AlnRec>(rec + "/alns.bin"); std::vector<uint32_t> hO = read_vec<uint32_t>(rec + "/ops.bin");
    std::vector<uint8_t> hQ = read_vec<uint8_t>(rec + "/seq.bin"); std::vector<ReadRec> hR = read_vec<ReadRec>(rec + "/reads.bin");
    int ulen = (int)meta["umi_len"]; if (ulen != par.umi_len) fprintf(stderr, "WARNING: UMI length in the reads (%d) differs from the engine's config (%d); using %d\n", ulen, par.umi_len, ulen);
    if (ulen > 13) die("UMI longer than 13 nt: base-5 key does not fit 32 bits");
    fprintf(stderr, "params: stranded %d reverse %d min_mapq %d min_baseq %d count_introns %d count_sj %d count_mt %d directional %d chrm_tid %d snps %u barcodes %u genes %u exons %u introns %u\n",
            par.stranded, par.reverse_strand, par.min_mapq, par.min_baseq, par.count_introns, par.count_sj, par.count_mt, par.directional, par.chrm_tid, par.n_snps, par.n_barcodes, par.n_genes, par.n_exons, par.n_introns);
    fprintf(stderr, "records: %zu reads, %zu signatures, %zu alignments, %zu ops, %zu seq bytes; E0 violations %.0f, incomplete multi-mapper groups %.0f, reads with a base below min_baseq %.0f\n",
            hR.size(), hS.size(), hA.size(), hO.size(), hQ.size(), meta["n_e0_violations"], meta["n_incomplete"], meta["n_reads_lowqual"]);
    fprintf(stderr, "quality check: bases where (quality < min_baseq) != (base is N): %.0f\n", meta["n_qual_neq_n"]);
    Arena A; A.init((size_t)(arena_gb * 1e9));
    Inputs in; in.nS = (int)hS.size(); in.nA = (int)hA.size(); in.nR = (int)hR.size(); in.ncells = (int)par.n_barcodes; in.ulen = ulen; in.par = par;
    if (meta["snp_skip_n"] < 0) die("base qualities are neither constant nor tied to N: the SNP pileup quality filter cannot be reproduced from sequences"); in.dp = DPar{par.stranded, par.reverse_strand, par.count_introns, par.count_sj, par.count_mt, par.min_mapq, par.chrm_tid, (int)meta["snp_skip_n"], 0u};
    // pinned staging + timed H2D of the per-sample records
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    in.EX = upload_tree(A, an.ex); in.IN = upload_tree(A, an.in); in.e2g = to_dev(A, an.e2g); in.i2g = to_dev(A, an.i2g); in.gstr = to_dev(A, an.gstr);
    in.ne2g = (uint32_t)an.e2g.size(); in.ni2g = (uint32_t)an.i2g.size(); in.ngstr = (uint32_t)an.gstr.size();
    { std::vector<int32_t> P; std::vector<uint32_t> I, off(an.snp.size()), n(an.snp.size()); std::vector<char> alt, ref;
      for (size_t t = 0; t < an.snp.size(); t++) { off[t] = (uint32_t)P.size(); n[t] = (uint32_t)an.snp[t].size(); for (auto& s : an.snp[t]) { P.push_back(s.pos); I.push_back(s.idx); alt.push_back(s.alt); ref.push_back(s.ref); } }
      in.SN.pos = to_dev(A, P); in.SN.idx = to_dev(A, I); in.SN.alt = to_dev(A, alt); in.SN.ref = to_dev(A, ref); in.SN.off = to_dev(A, off); in.SN.n = to_dev(A, n); in.SN.ntid = (int)an.snp.size(); }
    size_t anno_bytes = A.mark();
    cudaEventRecord(e0);
    in.S = to_dev(A, hS); in.Al = to_dev(A, hA); in.ops = to_dev(A, hO); in.seq = to_dev(A, hQ); in.R = to_dev(A, hR);
    cudaEventRecord(e1); cudaEventSynchronize(e1); float h2d_ms = 0; cudaEventElapsedTime(&h2d_ms, e0, e1);
    double in_bytes = hS.size() * 16.0 + hA.size() * 16.0 + hO.size() * 4.0 + hQ.size() + hR.size() * 16.0;
    size_t base = A.mark(); fprintf(stderr, "resident inputs %.2f GB (annotation %.2f GB), H2D of records %.0f ms (%.1f GB/s, pageable)\n", base / 1e9, anno_bytes / 1e9, h2d_ms, in_bytes / 1e9 / (h2d_ms / 1e3));
    // ---------------- runs ----------------
    Timer T; std::map<std::string, std::vector<double>> phase; std::vector<double> totals; Outputs out0, outN; std::vector<std::string> shas; double d2h_s = 0;
    for (int rep = -1; rep < reps; rep++) {
        A.reset(base); T.start(); Outputs* o = (rep == -1 || rep == reps - 1) ? (rep == -1 ? &out0 : &outN) : nullptr;
        // timing run first (no fetch inside the timed region): run twice per rep when fetching so the fetch is outside the events
        Outputs scratch; run_pipeline(A, T, in, nullptr);
        double ms = 0; for (auto& kv : T.acc) ms += kv.second; if (rep >= 0) { totals.push_back(ms); for (auto& kv : T.acc) phase[kv.first].push_back(kv.second); }
        if (rep == -1 || rep == reps - 1) { A.reset(base); T.start(); double t0 = now_s(); run_pipeline(A, T, in, o); d2h_s = now_s() - t0; shas.push_back(sha_out(*o)); }
    }
    bool deterministic = shas.size() == 2 && shas[0] == shas[1];
    // ---------------- equivalence ----------------
    const Outputs& g = outN;
    Cmp c_exon = cmp_t3(g.exon, tr.exon), c_intr = cmp_t3(g.intron, tr.intron), c_amb = cmp_t3(g.amb, tr.amb), c_ad = cmp_t3(g.snp_ad, tr.snp_ad), c_dp = cmp_t3(g.snp_dp, tr.snp_dp), c_mt = cmp_t3(g.mt, tr.mt), c_mtr = cmp_t3(g.mt_rev, tr.mt_rev);
    // junctions: canonical (chromosome, donor, acceptor) on both sides
    std::map<std::string, uint32_t> tidof; for (size_t i = 0; i < chroms.size(); i++) tidof[chroms[i]] = (uint32_t)i;
    std::vector<std::array<uint32_t, 3>> cpu_sjkey(tr.sj_names.size()); long long bad_names = 0;
    for (size_t i = 0; i < tr.sj_names.size(); i++) { const std::string& s = tr.sj_names[i]; size_t c = s.rfind(':'), d = s.find('-', c); if (c == std::string::npos || d == std::string::npos || !tidof.count(s.substr(0, c))) { bad_names++; cpu_sjkey[i] = {0, 0, 0}; continue; }
        cpu_sjkey[i] = {tidof[s.substr(0, c)], (uint32_t)atoll(s.substr(c + 1, d - c - 1).c_str()), (uint32_t)atoll(s.c_str() + d + 1)}; }
    std::vector<SjKey> sg, sc; for (auto& x : g.sj) { auto& k = g.sjtab[x.row]; sg.push_back(SjKey{x.cell, k[0], k[1], k[2], x.val}); }
    for (auto& x : tr.sj) { auto& k = cpu_sjkey[x.row]; sc.push_back(SjKey{x.cell, k[0], k[1], k[2], x.val}); }
    std::sort(sg.begin(), sg.end()); std::sort(sc.begin(), sc.end());
    Cmp c_sj; c_sj.gpu_nnz = sg.size(); c_sj.cpu_nnz = sc.size(); { size_t i = 0, j = 0; for (auto& x : sg) c_sj.gpu_sum += x.val; for (auto& x : sc) c_sj.cpu_sum += x.val;
        while (i < sg.size() || j < sc.size()) {
            if (i < sg.size() && j < sc.size() && !(sg[i] < sc[j]) && !(sc[j] < sg[i])) { if (sg[i].val == sc[j].val) c_sj.identical++; else c_sj.value_diff++; i++; j++; }
            else if (j >= sc.size() || (i < sg.size() && sg[i] < sc[j])) { c_sj.only_gpu++; i++; } else { c_sj.only_cpu++; j++; } } }
    std::vector<std::array<uint32_t, 4>> ig = g.indels, ic = tr.indels; std::sort(ig.begin(), ig.end()); std::sort(ic.begin(), ic.end()); bool ind_same = ig == ic;
    auto show = [&](const char* n, const Cmp& c) { fprintf(stderr, "equiv %-10s gpu nnz %lld cpu nnz %lld identical %lld only_gpu %lld only_cpu %lld value_diff %lld sum gpu %llu cpu %llu -> %s\n", n, c.gpu_nnz, c.cpu_nnz, c.identical, c.only_gpu, c.only_cpu, c.value_diff, c.gpu_sum, c.cpu_sum, is_identical(c) ? "IDENTICAL" : "DIFFERENT"); };
    if (!g_snp_first) printf("SNP consensus (R1): %llu molecules counted, %llu ambiguous (ties), %llu other-majority\n", g_snp_stat[0], g_snp_stat[1], g_snp_stat[2]);
    show("exon", c_exon); show("intron", c_intr); show("ambiguous", c_amb); show("junctions", c_sj); show("snp_ad", c_ad); show("snp_dp", c_dp); show("mt", c_mt); show("mt_rev", c_mtr);
    fprintf(stderr, "equiv mt_indels  gpu %zu cpu %zu -> %s;  junction names unparsed %lld; deterministic across runs %s\n", ig.size(), ic.size(), ind_same ? "IDENTICAL" : "DIFFERENT", bad_names, deterministic ? "yes" : "NO");
    // ---------------- report ----------------
    char tmp[1500]; std::string ph; bool f1 = true; double tot_med = 0; for (auto& kv : phase) { double m = median(kv.second); snprintf(tmp, sizeof tmp, "%s\"%s\": %.3f", f1 ? "" : ", ", kv.first.c_str(), m); ph += tmp; f1 = false; }
    tot_med = median(totals); double reads_per_s = hR.size() / (tot_med / 1e3);
    size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
    fprintf(stderr, "stages (median of %d after warm-up, ms): %s\ntotal %.1f ms for %zu reads = %.3f G reads/s; arena peak %.2f GB; device in use now %.2f GB of %.1f\n", reps, ph.c_str(), tot_med, hR.size(), reads_per_s / 1e9, A.peak / 1e9, (totb - freeb) / 1e9, totb / 1e9);
    std::string j = "{\n  \"label\": \"" + label + "\",\n  \"device\": \"" + prop.name + "\",\n";
    snprintf(tmp, sizeof tmp, "  \"qual_neq_n_bases\": %.0f,\n  \"reads\": %zu, \"signatures\": %zu, \"alignments\": %zu, \"barcodes\": %u, \"umi_len\": %d,\n  \"tuples\": {\"gene\": %lld, \"junction_instances\": %lld, \"junction_tuples\": %lld, \"junctions\": %lld, \"snp\": %lld, \"mt_bases\": %lld, \"mt_indels\": %lld},\n",
             meta["n_qual_neq_n"], hR.size(), hS.size(), hA.size(), par.n_barcodes, ulen, g.n_gene_tuples, g.n_sjl, g.n_sj_tuples, g.n_junctions, g.n_snp_tuples, g.n_mt_tuples, g.n_ind_tuples); j += tmp;
    j += "  \"equivalence\": {" + cmp_json("exon", c_exon) + ",\n    " + cmp_json("intron", c_intr) + ",\n    " + cmp_json("ambiguous", c_amb) + ",\n    " + cmp_json("junctions", c_sj) + ",\n    " + cmp_json("snp_ad", c_ad) + ",\n    " + cmp_json("snp_dp", c_dp) + ",\n    " + cmp_json("mt", c_mt) + ",\n    " + cmp_json("mt_rev", c_mtr) + ",\n";
    snprintf(tmp, sizeof tmp, "    \"mt_indels\": {\"gpu\": %zu, \"cpu\": %zu, \"identical\": %s}},\n", ig.size(), ic.size(), ind_same ? "true" : "false"); j += tmp;
    snprintf(tmp, sizeof tmp, "  \"deterministic_across_runs\": %s, \"output_sha256\": \"%s\",\n  \"phase_median_ms\": {%s},\n  \"total_median_ms\": %.3f, \"reads_per_s\": %.4g, \"reps\": %d,\n  \"h2d_records_ms\": %.1f, \"h2d_records_GB\": %.2f, \"second_run_with_fetch_s\": %.2f,\n  \"peak_device_memory_GB\": %.2f, \"resident_inputs_GB\": %.2f\n}\n",
             deterministic ? "true" : "false", shas[1].c_str(), ph.c_str(), tot_med, reads_per_s, reps, h2d_ms, in_bytes / 1e9, d2h_s, A.peak / 1e9, base / 1e9); j += tmp;
    if (!json.empty()) { FILE* f = fopen(json.c_str(), "w"); fputs(j.c_str(), f); fclose(f); } else fputs(j.c_str(), stdout);
    bool all = is_identical(c_exon) && is_identical(c_intr) && is_identical(c_amb) && is_identical(c_sj) && is_identical(c_ad) && is_identical(c_dp) && is_identical(c_mt) && is_identical(c_mtr) && ind_same;
    return all ? 0 : 3;
}
#endif // PILEUP_NO_MAIN
