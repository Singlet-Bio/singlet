// Standalone test of the phase 1e ATAC reducers (src/engine_gpu/gpu_pileup/pileup/atac_reducers.cuh) on real fragments.
// Does not touch pileup.cu or engine_gpu.cu: it includes pileup.cu (PILEUP_NO_MAIN) for the 1b helpers and then the reducers.
//
// Build (CUDA 12.4 image, host gcc 11):
//   nvcc -std=c++17 -O3 -arch=sm_90 -fmad=false -ccbin /usr/bin/g++ -Isrc/engine_gpu -Iinclude bench/atac_reducers/atac_reducers_test.cu -o atac_reducers_test
//
// Inputs (all real):
//   --chroms FILE     name<TAB>length per line, in BAM header order (contig id = line number)
//   --whitelist FILE  one barcode per line (cell id = line number)
//   --recs FILE       binary FragRec records (24 B) extracted from the BAM                   (mode A: raw alignment records)
//   --tsv CMD         shell command printing a fragments.tsv (chrom start end barcode read_pairs); each fragment is expanded to read_pairs
//                     records (plus filtered decoy records), so the reducers must reproduce the CPU's own output including its duplicate
//                     counts                                                                  (mode B)
//   --cells-bc FILE   optional: barcodes of the called cells (first column, header line skipped); enables the A8 second pass
//   --configs LIST    comma list K:seed (K cell partitions, seed 0 = stream order, otherwise shuffled); the first is the baseline
//   --out DIR         outputs of the baseline: frags.bin, ins_all.bin, ins.bin, peaks_all.bin, peaks.bin, triplets.bin, metrics.bin
//   --text-frag       also write frags.tsv (the CPU's format)    --chroms-peaks-bed  write peaks.bed (the CPU's format)
//   --cols FILE       write atac_peak_counts.mtx for these column barcodes      --metrics-tsv  write metrics.tsv
//   --arena-gb N      device arena (default 80)
#define PILEUP_NO_MAIN
#include "gpu_pileup/pileup/pileup.cu"
#include "gpu_pileup/pileup/atac_reducers.cuh"
#include <random>
#include <sstream>
#include <string_view>
#include <chrono>
#include <unistd.h>

static double t_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::vector<std::string> t_lines(const std::string& p) {
    std::vector<std::string> v; std::ifstream f(p); if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
    std::string l; while (std::getline(f, l)) { if (!l.empty() && l.back() == '\r') l.pop_back(); v.push_back(l); } return v; }

struct HF { uint64_t a, b; uint32_t rp; };   // a = tid << 32 | start, b = end << 32 | cell: sorting by (a, b) is the CPU's (contig, start, end, barcode) order
static bool hf_less(const HF& x, const HF& y) { return x.a != y.a ? x.a < y.a : x.b < y.b; }
struct PartHost { std::vector<uint64_t> hi; std::vector<uint32_t> lo, rp; };

struct Out {
    std::vector<HF> frags; std::vector<uint64_t> ik; std::vector<uint32_t> ic;                       // fragments, merged insertion table (pass 1: all cells)
    std::vector<AtPeak> peaks_all, peaks; std::vector<uint64_t> ik2; std::vector<uint32_t> ic2;      // first-pass peaks, final peaks and the table behind them
    std::vector<uint64_t> tkey; std::vector<uint32_t> tcnt; std::vector<uint32_t> metrics;           // (cell << 32 | peak, count); 5 counters per whitelist id
    double t_frag = 0, t_ins = 0, t_peak = 0, t_cnt = 0;
    PeakDiag diag;
};

struct Ctx {
    std::vector<std::string> cname, wl; std::vector<int32_t> clen; std::vector<uint8_t> is_mito, skip;
    int cb = 20, tb = 8, sb = 28, ncell = 0; uint8_t* d_mito = nullptr; Arena A; Timer T; std::vector<uint8_t> cellmask;
};

static InsDev upload_ins(Arena& A, const std::vector<uint64_t>& k, const std::vector<uint32_t>& c) {
    InsDev I{A.alloc<uint64_t>(std::max<size_t>(k.size(), 1)), A.alloc<uint32_t>(std::max<size_t>(k.size(), 1)), 0};
    if (!k.empty()) { CK(cudaMemcpy(I.key, k.data(), k.size() * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(I.count, c.data(), c.size() * 4, cudaMemcpyHostToDevice)); }
    I.n = (int)k.size(); return I; }
// merge K partition tables on the device (insertions_merge sorts and sums), copy the merged table back
static void merge_tables(Ctx& X, const std::vector<std::vector<uint64_t>>& ks, const std::vector<std::vector<uint32_t>>& cs, std::vector<uint64_t>& ok, std::vector<uint32_t>& oc) {
    std::vector<uint64_t> k; std::vector<uint32_t> c;
    for (size_t i = 0; i < ks.size(); i++) { k.insert(k.end(), ks[i].begin(), ks[i].end()); c.insert(c.end(), cs[i].begin(), cs[i].end()); }
    const size_t m = X.A.mark(); InsDev in = upload_ins(X.A, k, c); InsDev mg = insertions_merge(X.A, in.key, in.count, in.n);
    ok.resize(mg.n); oc.resize(mg.n);
    if (mg.n) { CK(cudaMemcpy(ok.data(), mg.key, (size_t)mg.n * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(oc.data(), mg.count, (size_t)mg.n * 4, cudaMemcpyDeviceToHost)); }
    X.A.reset(m);
}
static FragDev upload_frag(Ctx& X, const PartHost& P, const std::vector<uint8_t>* keep) {
    const std::vector<uint64_t>* H = &P.hi; const std::vector<uint32_t> *L = &P.lo, *R = &P.rp; PartHost f;
    if (keep) { for (size_t i = 0; i < P.hi.size(); i++) if ((*keep)[P.hi[i] >> (X.tb + X.sb)]) { f.hi.push_back(P.hi[i]); f.lo.push_back(P.lo[i]); f.rp.push_back(P.rp[i]); } H = &f.hi; L = &f.lo; R = &f.rp; }
    const size_t n = H->size(); FragDev F{X.A.alloc<uint64_t>(std::max<size_t>(n, 1)), X.A.alloc<uint32_t>(std::max<size_t>(n, 1)), X.A.alloc<uint32_t>(std::max<size_t>(n, 1)), (int)n, X.tb, X.sb};
    if (n) { CK(cudaMemcpy(F.hi, H->data(), n * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(F.lo, L->data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(F.read_pairs, R->data(), n * 4, cudaMemcpyHostToDevice)); }
    return F;
}

static Out run_config(Ctx& X, const std::vector<FragRec>& R, int K, bool diag) {
    Out o; std::vector<PartHost> parts(K); std::vector<std::vector<uint64_t>> iks(K); std::vector<std::vector<uint32_t>> ics(K);
    double t0 = t_now();
    for (int p = 0; p < K; p++) {
        std::vector<FragRec> sub; const FragRec* src = R.data(); size_t n = R.size();
        if (K > 1) { for (const FragRec& r : R) if (r.bc >= 0 && cell_part(r.bc, (uint32_t)K) == (uint32_t)p) sub.push_back(r); src = sub.data(); n = sub.size(); }
        const size_t m = X.A.mark(); FragRec* dR = X.A.alloc<FragRec>(std::max<size_t>(n, 1)); if (n) CK(cudaMemcpy(dR, src, n * sizeof(FragRec), cudaMemcpyHostToDevice));
        FragDev F = red_fragment(X.A, X.T, dR, (int)n, X.cb, X.tb, X.sb); CK(cudaDeviceSynchronize()); const double t1 = t_now(); o.t_frag += t1 - t0;
        PartHost& P = parts[p]; P.hi.resize(F.n); P.lo.resize(F.n); P.rp.resize(F.n);
        if (F.n) { CK(cudaMemcpy(P.hi.data(), F.hi, (size_t)F.n * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(P.lo.data(), F.lo, (size_t)F.n * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(P.rp.data(), F.read_pairs, (size_t)F.n * 4, cudaMemcpyDeviceToHost)); }
        t0 = t_now();
        InsDev I = red_insertions(X.A, X.T, F, X.d_mito); CK(cudaDeviceSynchronize()); o.t_ins += t_now() - t0;
        iks[p].resize(I.n); ics[p].resize(I.n);
        if (I.n) { CK(cudaMemcpy(iks[p].data(), I.key, (size_t)I.n * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ics[p].data(), I.count, (size_t)I.n * 4, cudaMemcpyDeviceToHost)); }
        X.A.reset(m); t0 = t_now();
        const uint64_t cm = (1ull << X.cb) - 1, sm = (1ull << X.sb) - 1, tm = (1ull << X.tb) - 1; (void)cm;
        for (size_t i = 0; i < P.hi.size(); i++) {
            const uint64_t tid = (P.hi[i] >> X.sb) & tm, start = P.hi[i] & sm, cell = P.hi[i] >> (X.tb + X.sb), end = start + P.lo[i];
            o.frags.push_back(HF{(tid << 32) | start, (end << 32) | cell, P.rp[i]});
        }
    }
    std::sort(o.frags.begin(), o.frags.end(), hf_less);
    merge_tables(X, iks, ics, o.ik, o.ic);
    // pass 1 peaks: all resolved barcodes
    t0 = t_now();
    { const size_t m = X.A.mark(); InsDev I = upload_ins(X.A, o.ik, o.ic);
      if (diag) o.diag.stride = 997;
      o.peaks_all = peak_call_dev(X.A, X.T, I, X.clen, X.skip, PeakPar(), diag ? &o.diag : nullptr); X.A.reset(m); }
    o.t_peak += t_now() - t0;
    if (!X.cellmask.empty()) {   // A8: insertions of the called cells only, merged canonically, final peaks
        for (int p = 0; p < K; p++) {
            const size_t m = X.A.mark(); FragDev F = upload_frag(X, parts[p], &X.cellmask); InsDev I = red_insertions(X.A, X.T, F, X.d_mito); CK(cudaDeviceSynchronize());
            iks[p].assign(I.n, 0); ics[p].assign(I.n, 0);
            if (I.n) { CK(cudaMemcpy(iks[p].data(), I.key, (size_t)I.n * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ics[p].data(), I.count, (size_t)I.n * 4, cudaMemcpyDeviceToHost)); }
            X.A.reset(m);
        }
        merge_tables(X, iks, ics, o.ik2, o.ic2);
        t0 = t_now(); { const size_t m = X.A.mark(); InsDev I = upload_ins(X.A, o.ik2, o.ic2); o.peaks = peak_call_dev(X.A, X.T, I, X.clen, X.skip); X.A.reset(m); } o.t_peak += t_now() - t0;
    } else { o.peaks = o.peaks_all; o.ik2 = o.ik; o.ic2 = o.ic; }
    // pass 2: cell x peak cut-site counts and per-cell counters against the final peaks, per partition
    t0 = t_now();
    const int np = (int)o.peaks.size(); std::vector<uint64_t> hps(np), hpe(np);
    for (int i = 0; i < np; i++) { hps[i] = ((uint64_t)o.peaks[i].tid << 32) | (uint32_t)o.peaks[i].start; hpe[i] = ((uint64_t)o.peaks[i].tid << 32) | (uint32_t)o.peaks[i].end; }
    o.metrics.assign((size_t)X.ncell * 5, 0u); std::vector<std::pair<uint64_t, uint32_t>> trip;
    for (int p = 0; p < K; p++) {
        const size_t m = X.A.mark(); FragDev F = upload_frag(X, parts[p], nullptr);
        uint64_t* ps = X.A.alloc<uint64_t>(std::max(np, 1)); uint64_t* pe = X.A.alloc<uint64_t>(std::max(np, 1));
        if (np) { CK(cudaMemcpy(ps, hps.data(), np * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(pe, hpe.data(), np * 8, cudaMemcpyHostToDevice)); }
        PeakCountDev C = red_peak_count(X.A, X.T, F, X.d_mito, ps, pe, np);
        std::vector<uint64_t> ck(C.n); std::vector<uint32_t> cc(C.n);
        if (C.n) { CK(cudaMemcpy(ck.data(), C.key, (size_t)C.n * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(cc.data(), C.count, (size_t)C.n * 4, cudaMemcpyDeviceToHost)); }
        for (int i = 0; i < C.n; i++) trip.emplace_back(ck[i], cc[i]);
        CellMetricsDev M = red_cell_metrics(X.A, X.T, F, X.d_mito, ps, pe, np);
        std::vector<uint32_t> mc(M.n); std::vector<M5> mv(M.n);
        if (M.n) { CK(cudaMemcpy(mc.data(), M.cell, (size_t)M.n * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(mv.data(), M.v, (size_t)M.n * sizeof(M5), cudaMemcpyDeviceToHost)); }
        for (int i = 0; i < M.n; i++) for (int k = 0; k < 5; k++) o.metrics[(size_t)mc[i] * 5 + k] += mv[i].v[k];
        X.A.reset(m);
    }
    std::sort(trip.begin(), trip.end());
    for (auto& t : trip) { o.tkey.push_back(t.first); o.tcnt.push_back(t.second); }
    o.t_cnt += t_now() - t0;
    return o;
}

template <class T> static bool same(const std::vector<T>& a, const std::vector<T>& b) { return a.size() == b.size() && (a.empty() || !memcmp(a.data(), b.data(), a.size() * sizeof(T))); }
static bool same(const std::vector<HF>& a, const std::vector<HF>& b) {   // field-wise: HF has padding bytes
    if (a.size() != b.size()) return false; for (size_t i = 0; i < a.size(); i++) if (a[i].a != b[i].a || a[i].b != b[i].b || a[i].rp != b[i].rp) return false; return true; }
static bool compare(const char* tag, const Out& a, const Out& b) {
    const bool f = same(a.frags, b.frags), i1 = same(a.ik, b.ik) && same(a.ic, b.ic), p1 = same(a.peaks_all, b.peaks_all), i2 = same(a.ik2, b.ik2) && same(a.ic2, b.ic2),
               p2 = same(a.peaks, b.peaks), t = same(a.tkey, b.tkey) && same(a.tcnt, b.tcnt), m = same(a.metrics, b.metrics);
    printf("[inv] %-14s fragments %s  insertions %s  peaks_all %s  insertions_final %s  peaks_final %s  peak_counts %s  cell_metrics %s\n", tag, f ? "EQUAL" : "DIFF", i1 ? "EQUAL" : "DIFF",
           p1 ? "EQUAL" : "DIFF", i2 ? "EQUAL" : "DIFF", p2 ? "EQUAL" : "DIFF", t ? "EQUAL" : "DIFF", m ? "EQUAL" : "DIFF");
    return f && i1 && p1 && i2 && p2 && t && m;
}

// host recomputation of the raw grid scores (atac_peak_caller.h arithmetic with the host libm) at the sampled positions
static double host_poisson(uint32_t k, double lambda) {
    if (k == 0) return 0.0; if (lambda <= 0) return 300.0;
    const double log_t0 = -lambda + k * std::log(lambda) - std::lgamma(static_cast<double>(k) + 1.0); double sum = 1.0, term = 1.0;
    for (uint32_t i = k + 1; i < k + 100000; ++i) { term *= lambda / static_cast<double>(i); sum += term; if (term < 1e-15 * sum) break; }
    return std::max(0.0, -(log_t0 + std::log(sum)) / std::log(10.0));
}
static void check_scores(const Ctx& X, const Out& o) {
    const PeakDiag& dg = o.diag; const int nc = (int)X.clen.size(); std::vector<uint32_t> goff(nc + 1, 0);
    for (int c = 0; c < nc; c++) goff[c + 1] = goff[c] + (X.skip[c] ? 0 : (uint32_t)((X.clen[c] + 9) / 10));
    std::vector<uint64_t> cum(o.ik.size() + 1, 0); for (size_t i = 0; i < o.ik.size(); i++) cum[i + 1] = cum[i] + o.ic[i];
    auto below = [&](uint64_t kx) { return cum[std::lower_bound(o.ik.begin(), o.ik.end(), kx) - o.ik.begin()]; };
    size_t kbad = 0, sbad = 0, nz = 0; double maxd = 0, maxrel = 0; int maxd_k = 0;
    for (size_t i = 0; i < dg.g.size(); i++) {
        const uint32_t g = dg.g[i]; int c = int(std::upper_bound(goff.begin(), goff.end(), g) - goff.begin()) - 1; const int x = (int)(g - goff[c]) * 10 + 5;
        const uint64_t cc = (uint64_t)c << 32; uint32_t k = (uint32_t)(below(cc | (uint32_t)(x + 100)) - below(cc | (uint32_t)std::max(x - 100, 0))); double s = -1.0;
        if (k > 0) { const uint64_t nl = below(cc | (uint32_t)(x + 5000)) - below(cc | (uint32_t)std::max(x - 5000, 0)); const double lam = std::max(dg.lambda_bg, (double)nl * 200 / 10000);
                     if (k > lam) s = host_poisson(k, lam); else k = 0; }
        if (k != dg.k[i]) kbad++;
        if (s >= 0) { nz++; if (s != dg.s[i]) { sbad++; const double d = std::fabs(s - dg.s[i]); if (d > maxd) { maxd = d; maxd_k = (int)k; } maxrel = std::max(maxrel, d / std::max(s, 1e-300)); } }
        else if (dg.s[i] != -1.0) sbad++;
    }
    printf("[score] sampled %zu grid positions (stride %u of %llu): pileup k mismatches %zu; scored positions %zu, score not bit-equal to host %zu (max abs diff %.3g at k=%d, max rel %.3g); lambda_bg %.9g s_thresh %.2f candidates %llu\n",
           dg.g.size(), dg.stride, (unsigned long long)dg.n_positions, kbad, nz, sbad, maxd, maxd_k, maxrel, dg.lambda_bg, dg.s_thresh, (unsigned long long)dg.n_cand);
}

static void wr(const std::string& p, const void* d, size_t n) { FILE* f = fopen(p.c_str(), "wb"); if (!f) { fprintf(stderr, "cannot write %s\n", p.c_str()); exit(2); } if (n) fwrite(d, 1, n, f); fclose(f); }

int main(int argc, char** argv) {
    std::string chroms, wlp, recs, tsv, cellsbc, outd = ".", colsp, configs = "1:0"; bool text_frag = false, peaks_bed = false, metrics_tsv = false; double arena_gb = 80;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i]; auto nx = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--chroms") chroms = nx(); else if (a == "--whitelist") wlp = nx(); else if (a == "--recs") recs = nx(); else if (a == "--tsv") tsv = nx(); else if (a == "--cells-bc") cellsbc = nx();
        else if (a == "--configs") configs = nx(); else if (a == "--out") outd = nx(); else if (a == "--text-frag") text_frag = true; else if (a == "--peaks-bed") peaks_bed = true;
        else if (a == "--cols") colsp = nx(); else if (a == "--metrics-tsv") metrics_tsv = true; else if (a == "--arena-gb") arena_gb = std::stod(nx());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    Ctx X; for (auto& l : t_lines(chroms)) { auto t = l.find('\t'); X.cname.push_back(l.substr(0, t)); X.clen.push_back(std::stoi(l.substr(t + 1))); }
    X.wl = t_lines(wlp); X.ncell = (int)X.wl.size(); const int nc = (int)X.cname.size();
    for (auto& n : X.cname) { const bool mt = n == "chrM" || n == "MT" || n == "chrMT" || n == "M" || n == "mitochondrion_genome"; X.is_mito.push_back(mt); X.skip.push_back(mt); }
    X.cb = bits_of((uint64_t)X.ncell - 1); X.tb = bits_of((uint64_t)nc - 1); int32_t maxlen = 0; for (int32_t l : X.clen) maxlen = std::max(maxlen, l); X.sb = bits_of((uint64_t)maxlen);
    printf("[cfg] contigs %d (mito %d), whitelist %d, bits cell %d contig %d pos %d = %d\n", nc, (int)std::count(X.is_mito.begin(), X.is_mito.end(), (uint8_t)1), X.ncell, X.cb, X.tb, X.sb, X.cb + X.tb + X.sb);
    if (X.cb + X.tb + X.sb > 64) { printf("RESULT FAIL key does not fit in 64 bits\n"); return 1; }
    std::unordered_map<std::string_view, int> wl_id, ch_id; for (int i = 0; i < X.ncell; i++) wl_id.emplace(X.wl[i], i); for (int i = 0; i < nc; i++) ch_id.emplace(X.cname[i], i);

    // ---- records
    std::vector<FragRec> R; double t0 = t_now();
    if (!recs.empty()) {
        FILE* f = fopen(recs.c_str(), "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", recs.c_str()); return 2; }
        fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET); R.resize(sz / sizeof(FragRec)); if (fread(R.data(), sizeof(FragRec), R.size(), f) != R.size()) { fprintf(stderr, "short read\n"); return 2; } fclose(f);
    } else {
        FILE* p = popen(tsv.c_str(), "r"); if (!p) { fprintf(stderr, "popen failed\n"); return 2; }
        std::vector<char> buf(1 << 24); std::string carry; size_t nl = 0, bad = 0, ndecoy = 0; std::mt19937 rng(12345);
        auto line = [&](const char* s, size_t len) {
            const char* e = s + len; const char* t1 = (const char*)memchr(s, '\t', len); if (!t1) { bad++; return; }
            const char* t2 = (const char*)memchr(t1 + 1, '\t', e - t1 - 1); const char* t3 = t2 ? (const char*)memchr(t2 + 1, '\t', e - t2 - 1) : nullptr; const char* t4 = t3 ? (const char*)memchr(t3 + 1, '\t', e - t3 - 1) : nullptr;
            if (!t4) { bad++; return; }
            auto ci = ch_id.find(std::string_view(s, t1 - s)); auto bi = wl_id.find(std::string_view(t3 + 1, t4 - t3 - 1)); if (ci == ch_id.end() || bi == wl_id.end()) { bad++; return; }
            const int32_t st = (int32_t)strtol(t1 + 1, nullptr, 10), en = (int32_t)strtol(t2 + 1, nullptr, 10), rp = (int32_t)strtol(t4 + 1, nullptr, 10);
            FragRec r{bi->second, ci->second, st - 4, en - st + 9, ci->second, 0x3, 255, 0};   // paired + proper; leftmost mate (TLEN > 0), MAPQ 255
            for (int k = 0; k < rp; k++) R.push_back(r);
            if (nl++ % 5 == 0) {   // decoys the reducer must drop: real coordinates with a failing flag / mapq / tlen / mate / barcode
                FragRec d = r; switch (rng() % 8) { case 0: d.tlen = -d.tlen; break; case 1: d.flag |= 0x100; break; case 2: d.mapq = 3; break; case 3: d.flag |= 0x8; break; case 4: d.mtid = (d.tid + 1) % nc; break;
                    case 5: d.bc = -1; break; case 6: d.tlen = 2001; break; default: d.flag &= ~0x2; break; }
                R.push_back(d); ndecoy++;
            }
        };
        size_t got; while ((got = fread(buf.data(), 1, buf.size(), p)) > 0) {
            size_t s0 = 0; carry.append(buf.data(), 0);   // keep API simple: process complete lines, carry the tail
            std::string chunk; chunk.swap(carry); chunk.append(buf.data(), got); size_t pos = 0, nlp;
            while ((nlp = chunk.find('\n', pos)) != std::string::npos) { line(chunk.data() + pos, nlp - pos); pos = nlp + 1; }
            carry.assign(chunk, pos, std::string::npos); (void)s0;
        }
        pclose(p); printf("[load] tsv lines %zu, unparsed %zu, decoys %zu\n", nl, bad, ndecoy);
    }
    printf("[load] %zu records in %.1f s\n", R.size(), t_now() - t0);
    if (!cellsbc.empty()) {
        X.cellmask.assign(X.ncell, 0); size_t hit = 0, tot = 0; auto ls = t_lines(cellsbc);
        for (auto& l : ls) { if (l.rfind("barcode", 0) == 0) continue; auto t = l.find('\t'); auto it = wl_id.find(std::string_view(l.data(), t == std::string::npos ? l.size() : t)); tot++; if (it != wl_id.end()) { X.cellmask[it->second] = 1; hit++; } }
        printf("[cells] %zu called cells read, %zu in the whitelist\n", tot, hit);
    }

    X.A.init((size_t)(arena_gb * (1ull << 30))); std::vector<uint8_t> hm(1 << X.tb, 0); for (int i = 0; i < nc; i++) hm[i] = X.is_mito[i];
    CK(cudaMalloc(&X.d_mito, hm.size())); CK(cudaMemcpy(X.d_mito, hm.data(), hm.size(), cudaMemcpyHostToDevice));
    { cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); printf("[gpu] %s, %.1f GB, arena %.0f GB\n", pr.name, pr.totalGlobalMem / 1073741824.0, arena_gb); }

    // ---- configurations
    std::vector<std::pair<int, int>> cfgs; { std::stringstream ss(configs); std::string t; while (std::getline(ss, t, ',')) { auto c = t.find(':'); cfgs.emplace_back(std::stoi(t.substr(0, c)), std::stoi(t.substr(c + 1))); } }
    Out base; bool ok = true;
    for (size_t ci = 0; ci < cfgs.size(); ci++) {
        const int K = cfgs[ci].first, seed = cfgs[ci].second;
        if (seed) { std::mt19937_64 g(seed); std::shuffle(R.begin(), R.end(), g); }
        const double a = t_now(); Out o = run_config(X, R, K, ci == 0);
        uint64_t rps = 0; for (auto& f : o.frags) rps += f.rp;
        printf("[run K=%d seed=%d] %.1f s (fragments %.2f, insertions %.2f, peaks %.2f, counts+metrics %.2f): unique fragments %zu, read pairs in them %llu, insertion entries %zu, peaks_all %zu, peaks %zu, triplets %zu\n",
               K, seed, t_now() - a, o.t_frag, o.t_ins, o.t_peak, o.t_cnt, o.frags.size(), (unsigned long long)rps, o.ik.size(), o.peaks_all.size(), o.peaks.size(), o.tkey.size());
        if (ci == 0) {
            base = std::move(o); check_scores(X, base);
            // outputs of the baseline
            std::vector<uint32_t> fb; fb.reserve(base.frags.size() * 5);
            for (auto& f : base.frags) { fb.push_back((uint32_t)(f.a >> 32)); fb.push_back((uint32_t)f.a); fb.push_back((uint32_t)(f.b >> 32)); fb.push_back((uint32_t)f.b); fb.push_back(f.rp); }
            wr(outd + "/frags.bin", fb.data(), fb.size() * 4); { std::vector<uint32_t>().swap(fb); }
            auto wins = [&](const std::string& p, const std::vector<uint64_t>& k, const std::vector<uint32_t>& c) { uint64_t n = k.size(); FILE* f = fopen(p.c_str(), "wb"); fwrite(&n, 8, 1, f); fwrite(k.data(), 8, n, f); fwrite(c.data(), 4, n, f); fclose(f); };
            wins(outd + "/ins_all.bin", base.ik, base.ic); wins(outd + "/ins.bin", base.ik2, base.ic2);
            wr(outd + "/peaks_all.bin", base.peaks_all.data(), base.peaks_all.size() * sizeof(AtPeak)); wr(outd + "/peaks.bin", base.peaks.data(), base.peaks.size() * sizeof(AtPeak));
            { std::vector<uint32_t> t; for (size_t i = 0; i < base.tkey.size(); i++) { t.push_back((uint32_t)(base.tkey[i] >> 32)); t.push_back((uint32_t)base.tkey[i]); t.push_back(base.tcnt[i]); } wr(outd + "/triplets.bin", t.data(), t.size() * 4); }
            wr(outd + "/metrics.bin", base.metrics.data(), base.metrics.size() * 4);
            if (text_frag) {
                FILE* f = fopen((outd + "/frags.tsv").c_str(), "wb"); std::string b; b.reserve(1 << 22); char tmp[64];
                for (auto& h : base.frags) {
                    b += X.cname[h.a >> 32]; b += '\t'; b += std::to_string((uint32_t)h.a); b += '\t'; b += std::to_string(h.b >> 32); b += '\t'; b += X.wl[(uint32_t)h.b]; b += '\t'; b += std::to_string(h.rp); b += '\n';
                    if (b.size() > (1 << 22) - 256) { fwrite(b.data(), 1, b.size(), f); b.clear(); }
                } fwrite(b.data(), 1, b.size(), f); fclose(f); (void)tmp;
            }
            if (peaks_bed) {
                std::ofstream f(outd + "/peaks.bed");
                for (size_t i = 0; i < base.peaks.size(); i++) { const AtPeak& p = base.peaks[i];
                    f << X.cname[p.tid] << '\t' << p.start << '\t' << p.end << "\tpeak_" << (i + 1) << '\t' << std::min(1000, static_cast<int>(p.nlq * 10)) << "\t.\t" << p.pileup << '\t' << p.nlp << '\t' << p.nlq << '\t' << (p.summit - p.start) << '\n'; }
            }
            if (!colsp.empty()) {
                auto cols = t_lines(colsp); std::vector<int32_t> col_of(X.ncell, -1); for (size_t i = 0; i < cols.size(); i++) { auto it = wl_id.find(cols[i]); if (it != wl_id.end()) col_of[it->second] = (int32_t)i; }
                std::vector<std::pair<uint64_t, uint32_t>> t; for (size_t i = 0; i < base.tkey.size(); i++) { const int32_t c = col_of[base.tkey[i] >> 32]; if (c >= 0) t.emplace_back(((uint64_t)c << 32) | (uint32_t)base.tkey[i], base.tcnt[i]); }
                std::sort(t.begin(), t.end()); std::ofstream f(outd + "/atac_peak_counts.mtx"); f << "%%MatrixMarket matrix coordinate integer general\n" << base.peaks.size() << ' ' << cols.size() << ' ' << t.size() << '\n';
                for (auto& e : t) f << ((uint32_t)e.first + 1) << ' ' << ((e.first >> 32) + 1) << ' ' << e.second << '\n';
            }
            if (metrics_tsv) {
                std::vector<int> ids; for (int i = 0; i < X.ncell; i++) if (base.metrics[(size_t)i * 5] > 0) ids.push_back(i);
                std::sort(ids.begin(), ids.end(), [&](int a, int b) { return X.wl[a] < X.wl[b]; }); std::ofstream f(outd + "/metrics.tsv");
                for (int i : ids) { f << X.wl[i]; for (int k = 0; k < 5; k++) f << '\t' << base.metrics[(size_t)i * 5 + k]; f << '\n'; }
            }
        } else {
            char tag[32]; snprintf(tag, sizeof tag, "K=%d seed=%d", K, seed); if (!compare(tag, base, o)) ok = false;
        }
    }
    printf("RESULT %s\n", ok ? "PASS" : "FAIL"); return ok ? 0 : 1;
}
