// NOT COMPILED. Draft of the phase 1e reducers (ATAC / Multiome ATAC): fragment and peak. No CUDA toolchain was available when
// this was written; nothing here has been through nvcc, and nothing includes it yet. W11 owns building and gating it.
//
// Written against the phase 1b interface of pileup.cu (commit 475295a6): the reducer shape (emit -> sort -> collapse -> reduce, one call per
// cell partition), and its helpers Arena, cub_run, CK, Timer, bits_of, scan_u32, total_of, SumU32/SumU64, cell_part. Intended use: #include
// at the end of the helper section of pileup.cu (after sort_hilo), then call red_fragment / red_insertions per partition, merge the
// insertion tables once (peak_call), and call red_peak_count per partition. pileup.cu itself is not edited by this file.
//
// Spec and rules: docs/research-beta/atac-reducers-prep.md. Reference implementation and tests (numba, runnable on a laptop):
// bench/atac_reducers/atac_ref.py, tests/python/test_atac_reducers_ref.py. Every kernel below has a one-to-one counterpart there.
//
// Rule A8 (two-pass peaks): run red_insertions on all fragments -> merge -> peak call -> per-cell metrics -> host cell call, then run red_insertions again
// on the called cells' fragments only (filter F by is_cell[cell] before k_ins_count), merge, and call the final peaks; red_peak_count uses the final peaks.
//
// Determinism: integer sorts, scans, reduce-by-key and integer atomicAdd (commutative) only. The one floating-point stage is the Poisson
// tail in k_grid_scores; build with -fmad=false as pileup.cu does, and compare its output to the CPU port by gate (a).
#pragma once

// ------------------------------------------------------------------------------------------------------------------------------------
// input records: one per alignment record of the unspliced paired-end mode (both mates; the leftmost mate carries TLEN > 0).
// Produced by the read-level expansion of phase 1e (not written yet); only these fields are read.
struct FragRec { int32_t bc; int32_t tid; int32_t pos; int32_t tlen; int32_t mtid; uint16_t flag; uint8_t mapq; uint8_t pad; };

constexpr int kMinFrag = 10, kMaxFrag = 2000, kMinMapq = 30, kShiftL = 4, kShiftR = 5;   // atac_fragment.h
constexpr uint16_t kFProper = 0x2, kFUnmap = 0x4, kFMunmap = 0x8, kFSec = 0x100, kFSupp = 0x800;

__device__ __forceinline__ bool frag_ok(const FragRec& r) {
    if (r.bc < 0 || !(r.flag & kFProper) || (r.flag & (kFUnmap | kFMunmap | kFSec | kFSupp))) return false;
    if (r.mapq < kMinMapq || r.tlen < kMinFrag || r.tlen > kMaxFrag || r.mtid != r.tid) return false;
    return r.pos + kShiftL < r.pos + r.tlen - kShiftR;
}
__global__ void k_frag_flag(const FragRec* R, int n, uint32_t* f) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) f[i] = frag_ok(R[i]) ? 1u : 0u; }

// Fragment key: hi = cell | contig | start (cb + tb + sb bits, <= 64: 20 + 10 + 28 for hg38), lo = end - start (<= 1995, 11 bits). Sorting by
// (hi, lo) is sorting by (cell, contig, start, end).
__global__ void k_frag_fill(const FragRec* R, int n, const uint32_t* f, const uint32_t* off, int tb, int sb, uint64_t* hi, uint32_t* lo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n || !f[i]) return;
    const FragRec r = R[i]; const uint32_t o = off[i];
    const uint64_t start = (uint64_t)(r.pos + kShiftL);
    hi[o] = ((uint64_t)(uint32_t)r.bc << (tb + sb)) | ((uint64_t)(uint32_t)r.tid << sb) | start;
    lo[o] = (uint32_t)(r.tlen - kShiftR - kShiftL);   // end - start
}

struct FragDev { uint64_t* hi; uint32_t* lo; uint32_t* read_pairs; int n; int tb, sb; };   // unique fragments, sorted (cell, contig, start, end)

// fragment reducer: dedup by coordinates per cell, read_pairs = size of the run (the CPU's duplicate count). Global exact-key dedup by sort,
// never "same start as the previous record" (finding F1).
static FragDev red_fragment(Arena& A, Timer& T, const FragRec* R, int nR, int cb, int tb, int sb) {
    const int TB = 256, GR = (nR + TB - 1) / TB;
    FragDev out{nullptr, nullptr, nullptr, 0, tb, sb};
    if (nR == 0 || cb + tb + sb > 64) return out;   // the caller checks the size; > 64 bits needs a third sort pass
    uint32_t* f = A.alloc<uint32_t>(nR); k_frag_flag<<<GR, TB>>>(R, nR, f);
    uint32_t* off = scan_u32(A, f, nR); const int n = total_of(A, f, off, nR);
    if (n == 0) return out;
    uint64_t* hi = A.alloc<uint64_t>(n); uint32_t* lo = A.alloc<uint32_t>(n);
    k_frag_fill<<<GR, TB>>>(R, nR, f, off, tb, sb, hi, lo);
    uint64_t* hs; uint32_t* ord = sort_hilo(A, hi, lo, n, cb + tb + sb, 11, &hs); const int G = (n + TB - 1) / TB;
    uint32_t* first = A.alloc<uint32_t>(n); k_runflag<<<G, TB>>>(hs, lo, ord, n, first);
    uint32_t* seg = A.alloc<uint32_t>(n);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceScan::InclusiveSum(t, b, first, seg, n); });
    int m = 0; CK(cudaMemcpy(&m, seg + n - 1, 4, cudaMemcpyDeviceToHost));
    out.hi = A.alloc<uint64_t>(m); out.lo = A.alloc<uint32_t>(m); out.read_pairs = A.alloc<uint32_t>(m); out.n = m;
    uint32_t* segu = A.alloc<uint32_t>(m); int* dn = A.alloc<int>(4);
    k_unique_hilo<<<G, TB>>>(first, seg, hs, lo, ord, n, out.hi, out.lo);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceReduce::ReduceByKey(t, b, seg, segu, cub::ConstantInputIterator<uint32_t>(1), out.read_pairs, dn, SumU32(), n); });
    T.mark("e.frag");
    return out;
}

// ------------------------------------------------------------------------------------------------------------------------------------
// insertion table (pass 1 of the peak reducer). Two cut sites per unique non-mito fragment (start, end - 1); read_pairs does not weight them.
// Key = contig << 32 | position. One table per partition is reduced to (key, count) and moved to host memory; the cross-partition merge is a
// sort of the concatenation (insertions_merge), so it is a function of the multiset of insertions, not of K.
__global__ void k_ins_count(const FragDev F, const uint8_t* is_mito, uint32_t* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= F.n) return;
    const uint32_t tid = (uint32_t)((F.hi[i] >> F.sb) & ((1ull << F.tb) - 1)); c[i] = is_mito[tid] ? 0u : 2u; }
__global__ void k_ins_fill(const FragDev F, const uint32_t* c, const uint32_t* off, uint64_t* key) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= F.n || !c[i]) return;
    const uint64_t tid = (F.hi[i] >> F.sb) & ((1ull << F.tb) - 1), start = F.hi[i] & ((1ull << F.sb) - 1);
    key[off[i]] = (tid << 32) | start; key[off[i] + 1] = (tid << 32) | (start + F.lo[i] - 1); }

struct InsDev { uint64_t* key; uint32_t* count; int n; };
static InsDev red_insertions(Arena& A, Timer& T, const FragDev& F, const uint8_t* is_mito) {
    const int TB = 256; InsDev out{nullptr, nullptr, 0};
    if (F.n == 0) return out;
    uint32_t* c = A.alloc<uint32_t>(F.n); k_ins_count<<<(F.n + TB - 1) / TB, TB>>>(F, is_mito, c);
    uint32_t* off = scan_u32(A, c, F.n); const int n = total_of(A, c, off, F.n); if (n == 0) return out;
    uint64_t *k0 = A.alloc<uint64_t>(n), *k1 = A.alloc<uint64_t>(n); k_ins_fill<<<(F.n + TB - 1) / TB, TB>>>(F, c, off, k0);
    cub::DoubleBuffer<uint64_t> dk(k0, k1);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRadixSort::SortKeys(t, b, dk, n); });
    out.key = A.alloc<uint64_t>(n); out.count = A.alloc<uint32_t>(n); int* dn = A.alloc<int>(4);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceReduce::ReduceByKey(t, b, dk.Current(), out.key, cub::ConstantInputIterator<uint32_t>(1), out.count, dn, SumU32(), n); });
    CK(cudaMemcpy(&out.n, dn, 4, cudaMemcpyDeviceToHost));
    T.mark("e.ins");
    return out;
}
// merge of the K tables (host or device concatenation passed in): sort the pairs by key, sum the counts of equal keys. Integer, commutative.
static InsDev insertions_merge(Arena& A, const uint64_t* key, const uint32_t* cnt, int n) {
    InsDev out{A.alloc<uint64_t>(std::max(n, 1)), A.alloc<uint32_t>(std::max(n, 1)), 0};
    if (n == 0) return out;
    uint64_t *k1 = A.alloc<uint64_t>(n); uint32_t *c1 = A.alloc<uint32_t>(n);
    cub::DoubleBuffer<uint64_t> dk((uint64_t*)key, k1); cub::DoubleBuffer<uint32_t> dc((uint32_t*)cnt, c1);   // inputs are scratch once merged
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRadixSort::SortPairs(t, b, dk, dc, n); });
    int* dn = A.alloc<int>(4);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceReduce::ReduceByKey(t, b, dk.Current(), out.key, dc.Current(), out.count, dn, SumU32(), n); });
    CK(cudaMemcpy(&out.n, dn, 4, cudaMemcpyDeviceToHost));
    return out;
}

// ------------------------------------------------------------------------------------------------------------------------------------
// peak caller (pass 1 proper): atac_peak_caller.h on the merged table. One thread per grid position of the non-skipped contigs.
struct PeakPar { int ext = 200, step = 10, llocal = 10000, min_length = 200, max_gap = 30; double q_cutoff = 0.05, genome_size = 0; };
constexpr int kBinsP = 100000;

__device__ double poisson_upper_neg_log10_d(uint32_t k, double lam) {   // detail::poisson_upper_neg_log10, valid for k > lam
    if (k == 0) return 0.0; if (lam <= 0) return 300.0;
    const double log_t0 = -lam + k * log(lam) - lgamma((double)k + 1.0);
    double sum = 1.0, term = 1.0;
    for (uint32_t i = k + 1; i < k + 100000; ++i) { term *= lam / (double)i; sum += term; if (term < 1e-15 * sum) break; }
    const double r = -(log_t0 + log(sum)) / log(10.0);
    return r > 0.0 ? r : 0.0;
}
// insertions below x on contig c: lower_bound over the merged keys, then the prefix sum of counts (cum[i] = insertions before entry i)
__device__ __forceinline__ uint64_t ins_below(const uint64_t* key, const uint64_t* cum, int n, uint64_t kx) {
    int lo = 0, hi = n; while (lo < hi) { int mid = (lo + hi) >> 1; if (key[mid] < kx) lo = mid + 1; else hi = mid; }
    return cum[lo];   // cum has n + 1 entries
}
// goff[c]: first global grid index of contig c (skipped contigs have 0 positions). s[g] = -1 for zero bins (k == 0 or k <= local lambda).
__global__ void k_grid_scores(const uint64_t* key, const uint64_t* cum, int n, const uint32_t* goff, int nc, int total_g, int step, int half, int lhalf,
                              int ext, int llocal, double lambda_bg, uint32_t* k_out, double* s_out, unsigned long long* hist) {
    int g = blockIdx.x * blockDim.x + threadIdx.x; if (g >= total_g) return;
    int lo = 0, hi = nc; while (lo < hi) { int mid = (lo + hi) >> 1; if (goff[mid + 1] <= (uint32_t)g) lo = mid + 1; else hi = mid; }   // contig of g
    const uint64_t c = (uint64_t)lo; const int x = (int)(g - goff[lo]) * step + step / 2;
    const uint64_t k = ins_below(key, cum, n, (c << 32) | (uint32_t)(x + half)) - ins_below(key, cum, n, (c << 32) | (uint32_t)max(x - half, 0));
    double s = -1.0; uint32_t kk = (uint32_t)k;
    if (kk > 0) {
        const uint64_t nl = ins_below(key, cum, n, (c << 32) | (uint32_t)(x + lhalf)) - ins_below(key, cum, n, (c << 32) | (uint32_t)max(x - lhalf, 0));
        const double lam = fmax(lambda_bg, (double)nl * ext / llocal);
        if (kk > lam) s = poisson_upper_neg_log10_d(kk, lam); else kk = 0;
    }
    k_out[g] = kk; s_out[g] = s;
    const int b = s < 0 ? 0 : min(kBinsP, (int)(s * 100.0));
    atomicAdd(&hist[b], 1ull);   // integer, commutative
}
__global__ void k_cand_flag(const double* s, int n, double s_thresh, uint32_t* f) {
    int g = blockIdx.x * blockDim.x + threadIdx.x; if (g >= n) return; f[g] = (s[g] >= 1.0 && (double)(float)s[g] >= s_thresh) ? 1u : 0u; }

// Host part, in order (all small): n_positions = total_g; BH threshold from the 100001-bin histogram exactly as the CPU: raw[b] = b/100 + log10(n_ge)
// - log10(n_positions) for nonzero bins, running max from b = 0 stored as float, s_thresh = first bin whose float q >= -log10(q_cutoff), at least 1.
// Then candidates = positions with float(s) >= s_thresh (cub::DeviceSelect::Flagged keeps the global order = contig, position), and:
//   gs = g_in_contig * step, ge = min(gs + step, contig_len); qs = qscore[min(kBinsP, (int)((float)s * 100.0f))]   (float multiply, as the CPU)
//   new_region[i] = i == 0 || contig differs || gs[i] - ge[i-1] > max_gap;  seg = InclusiveSum(new_region)
//   per region (ReduceByKey on seg): start = first gs, end = last ge, maxk = Max(k), nlp = Max(float s), nlq = Max(qs)
//   summit grid = ReduceByKey Min over (k == maxk[seg] ? grid : UINT_MAX)  -> the first (leftmost) maximum, like the CPU's strict '>'
//   keep regions with end - start >= min_length; names peak_<i+1> in (contig, start) order
// The kernels for those steps are the same four patterns as above (flag, scan, gather, reduce-by-key) and are not repeated in this draft.

// ------------------------------------------------------------------------------------------------------------------------------------
// pass 2: cell x peak cut-site counts. Peaks are disjoint and sorted; pe/ps are the keys contig << 32 | end and | start.
__device__ __forceinline__ int peak_of_d(const uint64_t* ps, const uint64_t* pe, int np, uint64_t kx) {
    int lo = 0, hi = np; while (lo < hi) { int mid = (lo + hi) >> 1; if (pe[mid] <= kx) lo = mid + 1; else hi = mid; }   // first end > x
    return (lo < np && ps[lo] <= kx) ? lo : -1;
}
__global__ void k_pk_count(const FragDev F, const uint8_t* is_mito, const uint64_t* ps, const uint64_t* pe, int np, uint32_t* c, int* p1, int* p2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= F.n) return;
    const uint64_t tid = (F.hi[i] >> F.sb) & ((1ull << F.tb) - 1), start = F.hi[i] & ((1ull << F.sb) - 1);
    if (is_mito[tid]) { c[i] = 0; p1[i] = p2[i] = -1; return; }
    p1[i] = peak_of_d(ps, pe, np, (tid << 32) | start); p2[i] = peak_of_d(ps, pe, np, (tid << 32) | (start + F.lo[i] - 1));
    c[i] = (p1[i] >= 0) + (p2[i] >= 0);
}
__global__ void k_pk_fill(const FragDev F, int cb_shift, const uint32_t* c, const uint32_t* off, const int* p1, const int* p2, uint64_t* key) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= F.n || !c[i]) return;
    const uint64_t cell = F.hi[i] >> (F.tb + F.sb); uint32_t o = off[i];
    if (p1[i] >= 0) key[o++] = (cell << 32) | (uint32_t)p1[i];
    if (p2[i] >= 0) key[o] = (cell << 32) | (uint32_t)p2[i];
}
struct PeakCountDev { uint64_t* key; uint32_t* count; int n; };   // key = cell << 32 | peak, sorted: the sparse (cell, peak) matrix
static PeakCountDev red_peak_count(Arena& A, Timer& T, const FragDev& F, const uint8_t* is_mito, const uint64_t* ps, const uint64_t* pe, int np) {
    const int TB = 256; PeakCountDev out{nullptr, nullptr, 0};
    if (F.n == 0 || np == 0) return out;
    const int G = (F.n + TB - 1) / TB;
    uint32_t* c = A.alloc<uint32_t>(F.n); int *p1 = A.alloc<int>(F.n), *p2 = A.alloc<int>(F.n);
    k_pk_count<<<G, TB>>>(F, is_mito, ps, pe, np, c, p1, p2);
    uint32_t* off = scan_u32(A, c, F.n); const int n = total_of(A, c, off, F.n); if (n == 0) return out;
    uint64_t *k0 = A.alloc<uint64_t>(n), *k1 = A.alloc<uint64_t>(n); k_pk_fill<<<G, TB>>>(F, 0, c, off, p1, p2, k0);
    cub::DoubleBuffer<uint64_t> dk(k0, k1);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceRadixSort::SortKeys(t, b, dk, n); });
    out.key = A.alloc<uint64_t>(n); out.count = A.alloc<uint32_t>(n); int* dn = A.alloc<int>(4);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceReduce::ReduceByKey(t, b, dk.Current(), out.key, cub::ConstantInputIterator<uint32_t>(1), out.count, dn, SumU32(), n); });
    CK(cudaMemcpy(&out.n, dn, 4, cudaMemcpyDeviceToHost));
    T.mark("e.pkcount");
    return out;
}
// per-cell metrics (read_pairs, fragments, mito_fragments, peak_region_fragments, peak_region_cutsites) are reduce-by-cell of integers over the same
// FragDev; cells are disjoint across partitions, so the partition outputs concatenate in cell order. peak_region_fragments additionally needs
// the overlap test (first peak with end > start, then peak start < fragment end): the same binary search as peak_of_d.

// ------------------------------------------------------------------------------------------------------------------------------------
// Added with the first compile and run (bench/atac_reducers/atac_reducers_test.cu): the pieces the draft described but did not write.
struct AtPeak { int32_t tid, start, end, summit; float pileup, nlp, nlq; };   // atac_peak_caller.h AtacPeak

__global__ void k_u32_to_u64(const uint32_t* c, int n, uint64_t* o) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = c[i]; else if (i == n) o[n] = 0; }
// cum[i] = insertions before entry i of the merged table (n + 1 entries)
static uint64_t* ins_cum(Arena& A, const InsDev& I) {
    const int n = I.n; uint64_t *c64 = A.alloc<uint64_t>(n + 1), *cum = A.alloc<uint64_t>(n + 1);
    k_u32_to_u64<<<(n + 1 + 255) / 256, 256>>>(I.count, n, c64);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceScan::ExclusiveSum(t, b, c64, cum, n + 1); });
    return cum;
}
__global__ void k_gather_u32(const uint32_t* src, const uint32_t* idx, int n, uint32_t* o) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = src[idx[i]]; }
__global__ void k_gather_f64(const double* src, const uint32_t* idx, int n, double* o) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = src[idx[i]]; }

struct PeakDiag { uint32_t stride = 0; std::vector<uint32_t> g, k; std::vector<double> s; uint64_t n_positions = 0, n_cand = 0; double lambda_bg = 0, s_thresh = 0; };

// Peak caller on the merged insertion table: device scoring + histogram + candidate compaction, host BH threshold and region merge over the
// compacted candidates (the region merge is a sequential pass over the candidate list here; the flag + scan + reduce-by-key version is not written).
// Output equals call_peaks_insertions (atac_peak_caller.h). chrom_len / skip are by contig id.
static std::vector<AtPeak> peak_call_dev(Arena& A, Timer& T, const InsDev& I, const std::vector<int32_t>& chrom_len, const std::vector<uint8_t>& skip,
                                         const PeakPar& P = PeakPar(), PeakDiag* dg = nullptr) {
    std::vector<AtPeak> peaks; const int nc = (int)chrom_len.size(); if (I.n == 0) return peaks;
    const size_t mk = A.mark();
    const int half = P.ext / 2, lhalf = P.llocal / 2;
    std::vector<uint32_t> goff(nc + 1, 0); double glen = 0;
    for (int c = 0; c < nc; c++) { const uint32_t ng = skip[c] ? 0 : (uint32_t)((chrom_len[c] + P.step - 1) / P.step); goff[c + 1] = goff[c] + ng; if (!skip[c]) glen += chrom_len[c]; }
    const int total_g = (int)goff[nc];
    uint64_t* cum = ins_cum(A, I); uint64_t total_ins = 0; CK(cudaMemcpy(&total_ins, cum + I.n, 8, cudaMemcpyDeviceToHost));
    const double gsize = P.genome_size > 0 ? P.genome_size : 0.87 * glen, lambda_bg = gsize > 0 ? (double)total_ins * P.ext / gsize : 0.0;
    uint32_t* dgoff = A.alloc<uint32_t>(nc + 1); CK(cudaMemcpy(dgoff, goff.data(), (nc + 1) * 4, cudaMemcpyHostToDevice));
    uint32_t* kout = A.alloc<uint32_t>(total_g); double* sout = A.alloc<double>(total_g);
    unsigned long long* hist = A.alloc<unsigned long long>(kBinsP + 1); CK(cudaMemset(hist, 0, (kBinsP + 1) * 8));
    const int TB = 256, G = (total_g + TB - 1) / TB;
    k_grid_scores<<<G, TB>>>(I.key, cum, I.n, dgoff, nc, total_g, P.step, half, lhalf, P.ext, P.llocal, lambda_bg, kout, sout, hist);
    CK(cudaGetLastError());
    std::vector<unsigned long long> h(kBinsP + 1); CK(cudaMemcpy(h.data(), hist, (kBinsP + 1) * 8, cudaMemcpyDeviceToHost));
    // BH exactly as atac_peak_caller.h
    const uint64_t n_positions = (uint64_t)total_g; const double q_need = -std::log10(P.q_cutoff);
    std::vector<float> qscore(kBinsP + 1, 0.f);
    { std::vector<double> raw(kBinsP + 1, -1.0); uint64_t n_ge = 0;
      for (int b = kBinsP; b >= 0; --b) { n_ge += h[b]; if (h[b] == 0) continue; raw[b] = b / 100.0 + std::log10((double)n_ge) - std::log10((double)n_positions); }
      double run = 0.0; for (int b = 0; b <= kBinsP; ++b) { if (raw[b] > run) run = raw[b]; qscore[b] = (float)run; } }
    double s_thresh = 1e9; for (int b = 0; b <= kBinsP; ++b) if (qscore[b] >= q_need) { s_thresh = b / 100.0; break; }
    s_thresh = std::max(s_thresh, 1.0);
    // candidates in global (contig, position) order
    uint32_t* fl = A.alloc<uint32_t>(total_g); k_cand_flag<<<G, TB>>>(sout, total_g, s_thresh, fl);
    uint32_t* idx = A.alloc<uint32_t>(total_g); int* dn = A.alloc<int>(4);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceSelect::Flagged(t, b, cub::CountingInputIterator<uint32_t>(0), fl, idx, dn, total_g); });
    int ncand = 0; CK(cudaMemcpy(&ncand, dn, 4, cudaMemcpyDeviceToHost));
    std::vector<uint32_t> hg(ncand), hk(ncand); std::vector<double> hs(ncand);
    if (ncand) {
        uint32_t* ck = A.alloc<uint32_t>(ncand); double* cs = A.alloc<double>(ncand);
        k_gather_u32<<<(ncand + TB - 1) / TB, TB>>>(kout, idx, ncand, ck); k_gather_f64<<<(ncand + TB - 1) / TB, TB>>>(sout, idx, ncand, cs);
        CK(cudaMemcpy(hg.data(), idx, ncand * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hk.data(), ck, ncand * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hs.data(), cs, ncand * 8, cudaMemcpyDeviceToHost));
    }
    if (dg && dg->stride) {   // strided sample of the raw scores for the host cross-check
        const int ns = (total_g + dg->stride - 1) / dg->stride; std::vector<uint32_t> sg(ns); for (int i = 0; i < ns; i++) sg[i] = (uint32_t)i * dg->stride;
        uint32_t* dsg = A.alloc<uint32_t>(ns); CK(cudaMemcpy(dsg, sg.data(), ns * 4, cudaMemcpyHostToDevice));
        uint32_t* sk = A.alloc<uint32_t>(ns); double* ss = A.alloc<double>(ns);
        k_gather_u32<<<(ns + TB - 1) / TB, TB>>>(kout, dsg, ns, sk); k_gather_f64<<<(ns + TB - 1) / TB, TB>>>(sout, dsg, ns, ss);
        dg->g = sg; dg->k.resize(ns); dg->s.resize(ns); CK(cudaMemcpy(dg->k.data(), sk, ns * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(dg->s.data(), ss, ns * 8, cudaMemcpyDeviceToHost));
        dg->n_positions = n_positions; dg->n_cand = (uint64_t)ncand; dg->lambda_bg = lambda_bg; dg->s_thresh = s_thresh;
    }
    CK(cudaDeviceSynchronize());
    // region merge over the compacted candidates (host, sequential), as call_peaks_insertions pass 2
    AtPeak cur{}; bool open = false; int last_end = 0; int cur_tid = -1;
    auto flush = [&]() { if (open && cur.end - cur.start >= P.min_length) peaks.push_back(cur); };
    for (int i = 0; i < ncand; i++) {
        const float sf = (float)hs[i]; if ((double)sf < s_thresh) continue;
        int lo = 0, hi = nc; while (lo < hi) { int mid = (lo + hi) >> 1; if (goff[mid + 1] <= hg[i]) lo = mid + 1; else hi = mid; }
        const int c = lo, grid = (int)(hg[i] - goff[c]); const int gs = grid * P.step, ge = std::min(gs + P.step, (int)chrom_len[c]);
        const float qs = qscore[std::min(kBinsP, (int)(sf * 100.0f))];
        if (open && c == cur_tid && gs - last_end <= P.max_gap) {
            cur.end = ge; if ((float)hk[i] > cur.pileup) { cur.pileup = (float)hk[i]; cur.summit = gs + P.step / 2; }
            cur.nlp = std::max(cur.nlp, sf); cur.nlq = std::max(cur.nlq, qs);
        } else { flush(); cur = AtPeak{c, gs, ge, gs + P.step / 2, (float)hk[i], sf, qs}; cur_tid = c; open = true; }
        last_end = ge;
    }
    flush();
    A.reset(mk); T.mark("e.peakcall");
    return peaks;
}

// per-cell integer counters (the cell caller's inputs) by reduce-by-cell over a FragDev (sorted cell-major): read_pairs, fragments, mito_fragments,
// peak_region_fragments, peak_region_cutsites
struct M5 { uint32_t v[5]; };
struct SumM5 { __host__ __device__ M5 operator()(const M5& a, const M5& b) const { M5 r; for (int i = 0; i < 5; i++) r.v[i] = a.v[i] + b.v[i]; return r; } };
__device__ __forceinline__ int peak_first_end_gt(const uint64_t* pe, int np, uint64_t kx) { int lo = 0, hi = np; while (lo < hi) { int mid = (lo + hi) >> 1; if (pe[mid] <= kx) lo = mid + 1; else hi = mid; } return lo; }
__global__ void k_cm_emit(const FragDev F, const uint8_t* is_mito, const uint64_t* ps, const uint64_t* pe, int np, uint32_t* cell, M5* m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= F.n) return;
    const uint64_t tid = (F.hi[i] >> F.sb) & ((1ull << F.tb) - 1), start = F.hi[i] & ((1ull << F.sb) - 1), end = start + F.lo[i];
    cell[i] = (uint32_t)(F.hi[i] >> (F.tb + F.sb)); M5 r; for (int k = 0; k < 5; k++) r.v[k] = 0; r.v[0] = F.read_pairs[i];
    if (is_mito[tid]) { r.v[2] = 1; m[i] = r; return; }
    r.v[1] = 1;
    if (np) {
        const int p1 = peak_of_d(ps, pe, np, (tid << 32) | start), p2 = peak_of_d(ps, pe, np, (tid << 32) | (end - 1));
        r.v[4] = (p1 >= 0) + (p2 >= 0);
        const int q = peak_first_end_gt(pe, np, (tid << 32) | start);   // first peak with end > start; overlap iff its start < end
        r.v[3] = (p1 >= 0 || p2 >= 0 || (q < np && ps[q] < ((tid << 32) | end))) ? 1u : 0u;
    }
    m[i] = r;
}
struct CellMetricsDev { uint32_t* cell; M5* v; int n; };
static CellMetricsDev red_cell_metrics(Arena& A, Timer& T, const FragDev& F, const uint8_t* is_mito, const uint64_t* ps, const uint64_t* pe, int np) {
    CellMetricsDev out{nullptr, nullptr, 0}; if (F.n == 0) return out;
    uint32_t* cell = A.alloc<uint32_t>(F.n); M5* m = A.alloc<M5>(F.n); k_cm_emit<<<(F.n + 255) / 256, 256>>>(F, is_mito, ps, pe, np, cell, m);
    out.cell = A.alloc<uint32_t>(F.n); out.v = A.alloc<M5>(F.n); int* dn = A.alloc<int>(4);
    cub_run(A, [&](void* t, size_t& b) { return cub::DeviceReduce::ReduceByKey(t, b, cell, out.cell, m, out.v, dn, SumM5(), F.n); });
    CK(cudaMemcpy(&out.n, dn, 4, cudaMemcpyDeviceToHost)); T.mark("e.cellmetrics");
    return out;
}
