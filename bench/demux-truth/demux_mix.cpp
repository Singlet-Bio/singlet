// SPDX-License-Identifier: MIT
// In-silico donor mixtures with known truth, run through the engine's run_demux().
// Ported to the October donor_demux.h (uint16 counts, held-out auto-K, doublet stage, staged restarts).
//
// Input: per-donor snp_ad + snp_dp matrices (Matrix Market, gz) from single-donor runs with --snps.
// A mixture takes the called cells of N samples (one donor each), keeps them as distinct cells and plants
// cross-donor doublets by merging two cells of different donors. Before the merge each cell keeps a random
// fraction of its reads (--keep-lo..--keep-hi) because a real doublet shares the sequencing depth of one droplet.
// AD and DP are summed after thinning. Columns are shuffled. Only truth labels and per-cell assignments are
// written to --out. The mixed AD/DP matrices go to --export (private, stays on the cluster).
//
//   demux_mix --samples samples.tsv --donors S1,S2,S3 --rate 0.05 --cap 2000 --mix-seed 1 --out DIR
//             [--keep-lo 0.6 --keep-hi 0.8] [--export PRIVDIR] [--runs "tag:K:threads:seed[:diag] ..."]
//             (K = -1 for auto; the diag suffix runs the truth-seeded ELBO comparison for that run)
//   demux_mix --samples samples.tsv --qc S1 --out DIR        per-sample QC (aggregates only)
//
// samples.tsv columns: id, private dir (holds snp_ad.mtx.gz, snp_dp.mtx.gz and the *_barcodes.tsv.gz), cell_calls.tsv
// Build: g++ -O2 -std=c++17 -pthread -I<repo>/include demux_mix.cpp -o demux_mix

#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "singlet/pileup/donor_demux.h"

struct Csc {
    int32_t nrows = 0, ncols = 0;
    std::vector<int32_t> indptr, indices;
    std::vector<uint16_t> data;
};

static std::vector<std::string> zlines(const std::string& path) {
    std::vector<std::string> out;
    FILE* f = popen(("zcat '" + path + "'").c_str(), "r");
    if (!f) return out;
    std::vector<char> buf(1 << 16);
    while (fgets(buf.data(), buf.size(), f)) {
        std::string s(buf.data());
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        out.push_back(s);
    }
    pclose(f);
    return out;
}

// Reads the columns whose barcode is in sel (barcode -> new column index); other columns are skipped.
static bool read_mtx_sel(const std::string& mtx, const std::string& bcs,
                         const std::unordered_map<std::string, int>& sel, int nsel, Csc& m) {
    auto bc = zlines(bcs);
    std::vector<int32_t> colmap(bc.size(), -1);
    for (size_t i = 0; i < bc.size(); ++i) {
        auto it = sel.find(bc[i].substr(0, bc[i].find('\t')));
        if (it != sel.end()) colmap[i] = it->second;
    }
    FILE* f = popen(("zcat '" + mtx + "'").c_str(), "r");
    if (!f) return false;
    std::vector<char> buf(1 << 16);
    long long nr = 0, nc = 0, nnz = 0;
    while (fgets(buf.data(), buf.size(), f)) {
        if (buf[0] == '%') continue;
        if (sscanf(buf.data(), "%lld %lld %lld", &nr, &nc, &nnz) != 3) { pclose(f); return false; }
        break;
    }
    if (nc != static_cast<long long>(bc.size())) { std::cerr << "barcode count mismatch " << mtx << "\n"; pclose(f); return false; }
    m.nrows = static_cast<int32_t>(nr);
    m.ncols = nsel;
    std::vector<std::pair<int32_t, int32_t>> rc;  // (col, row)
    std::vector<uint16_t> vals;
    while (fgets(buf.data(), buf.size(), f)) {
        char* e;
        long r = strtol(buf.data(), &e, 10);
        long c = strtol(e, &e, 10);
        long v = strtol(e, &e, 10);
        const int nc2 = colmap[c - 1];
        if (nc2 < 0) continue;
        rc.emplace_back(nc2, static_cast<int32_t>(r - 1));
        vals.push_back(static_cast<uint16_t>(v > 65535 ? 65535 : (v < 0 ? 0 : v)));
    }
    pclose(f);
    std::vector<size_t> ord(rc.size());
    for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return rc[a] < rc[b]; });
    m.indptr.assign(nsel + 1, 0);
    for (auto& p : rc) m.indptr[p.first + 1]++;
    for (int j = 0; j < nsel; ++j) m.indptr[j + 1] += m.indptr[j];
    m.indices.resize(rc.size());
    m.data.resize(rc.size());
    for (size_t k = 0; k < ord.size(); ++k) { m.indices[k] = rc[ord[k]].second; m.data[k] = vals[ord[k]]; }
    return true;
}

struct Sample { std::string id, dir, calls; Csc ad, dp; std::vector<std::string> bc; };

// One cell as (snp, alt, total) triples, DP drives (AD <= DP at every position).
struct Obs { int32_t snp; uint32_t ad, dp; };
static std::vector<Obs> cell_obs(const Sample& s, int c) {
    std::vector<Obs> o;
    int32_t j = s.ad.indptr[c];
    const int32_t je = s.ad.indptr[c + 1];
    for (int32_t q = s.dp.indptr[c]; q < s.dp.indptr[c + 1]; ++q) {
        const int32_t snp = s.dp.indices[q];
        while (j < je && s.ad.indices[j] < snp) ++j;
        const uint32_t a = (j < je && s.ad.indices[j] == snp) ? s.ad.data[j] : 0;
        if (s.dp.data[q] > 0) o.push_back({snp, std::min<uint32_t>(a, s.dp.data[q]), s.dp.data[q]});
    }
    return o;
}
// Keep each read with probability keep: alt and ref reads are thinned independently.
static void thin(std::vector<Obs>& o, double keep, std::mt19937_64& rng) {
    std::vector<Obs> r;
    r.reserve(o.size());
    for (auto& x : o) {
        std::binomial_distribution<int> ba(static_cast<int>(x.ad), keep), br(static_cast<int>(x.dp - x.ad), keep);
        const uint32_t a = ba(rng), f = br(rng);
        if (a + f > 0) r.push_back({x.snp, a, a + f});
    }
    o.swap(r);
}
static std::vector<Obs> sum_obs(const std::vector<Obs>& a, const std::vector<Obs>& b) {
    std::vector<Obs> o;
    size_t i = 0, j = 0;
    while (i < a.size() || j < b.size()) {
        if (j >= b.size() || (i < a.size() && a[i].snp < b[j].snp)) o.push_back(a[i++]);
        else if (i >= a.size() || b[j].snp < a[i].snp) o.push_back(b[j++]);
        else { o.push_back({a[i].snp, a[i].ad + b[j].ad, a[i].dp + b[j].dp}); ++i; ++j; }
    }
    return o;
}
static void put_obs(const std::vector<Obs>& o, Csc& ad, Csc& dp) {
    for (auto& x : o) {
        if (x.ad > 0) { ad.indices.push_back(x.snp); ad.data.push_back(static_cast<uint16_t>(std::min<uint32_t>(x.ad, 65535))); }
        dp.indices.push_back(x.snp); dp.data.push_back(static_cast<uint16_t>(std::min<uint32_t>(x.dp, 65535)));
    }
    ad.indptr.push_back(static_cast<int32_t>(ad.indices.size()));
    dp.indptr.push_back(static_cast<int32_t>(dp.indices.size()));
}

static double cpu_s() {
    rusage ru; getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_stime.tv_sec + 1e-6 * (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec);
}

static double ari(const std::vector<int>& a, const std::vector<int>& b) {
    std::map<std::pair<int, int>, double> tab;
    std::map<int, double> ra, rb;
    for (size_t i = 0; i < a.size(); ++i) { tab[{a[i], b[i]}] += 1; ra[a[i]] += 1; rb[b[i]] += 1; }
    auto c2 = [](double x) { return x * (x - 1) / 2; };
    double s = 0, sa = 0, sb = 0;
    for (auto& t : tab) s += c2(t.second);
    for (auto& t : ra) sa += c2(t.second);
    for (auto& t : rb) sb += c2(t.second);
    const double n2 = c2(static_cast<double>(a.size())), e = sa * sb / n2, mx = 0.5 * (sa + sb);
    return (mx - e) == 0 ? 1.0 : (s - e) / (mx - e);
}

static void write_mtx(const std::string& path, const Csc& m) {
    std::ofstream f(path);
    f << "%%MatrixMarket matrix coordinate integer general\n" << m.nrows << " " << m.ncols << " " << m.indices.size() << "\n";
    for (int c = 0; c < m.ncols; ++c)
        for (int32_t q = m.indptr[c]; q < m.indptr[c + 1]; ++q) f << (m.indices[q] + 1) << " " << (c + 1) << " " << m.data[q] << "\n";
}

static std::vector<std::string> called_cells(const Sample& s) {
    std::unordered_set<std::string> inad;
    for (auto& b : zlines(s.dir + "/snp_ad_barcodes.tsv.gz")) inad.insert(b.substr(0, b.find('\t')));
    std::vector<std::string> called;
    std::ifstream f(s.calls); std::string l; std::getline(f, l);
    while (std::getline(f, l)) { std::istringstream ss(l); std::string bc, is; ss >> bc >> is; if (is == "TRUE" && inad.count(bc)) called.push_back(bc); }
    std::sort(called.begin(), called.end());
    return called;
}

// Per-sample QC of a single-donor run. Aggregates only: no per-SNP values leave this function.
static int qc_sample(Sample& s, const std::string& out) {
    auto called = called_cells(s);
    std::unordered_map<std::string, int> sel;
    for (size_t i = 0; i < called.size(); ++i) sel[called[i]] = static_cast<int>(i);
    if (!read_mtx_sel(s.dir + "/snp_ad.mtx.gz", s.dir + "/snp_ad_barcodes.tsv.gz", sel, called.size(), s.ad) ||
        !read_mtx_sel(s.dir + "/snp_dp.mtx.gz", s.dir + "/snp_dp_barcodes.tsv.gz", sel, called.size(), s.dp)) return 2;
    const int nc = static_cast<int>(called.size());
    const size_t ns = s.dp.nrows;
    std::vector<uint32_t> tad(ns, 0), tdp(ns, 0), cell_alt(ns, 0), cell_ref(ns, 0);
    std::vector<int> ncov(nc, 0);
    for (int c = 0; c < nc; ++c) {
        for (auto& o : cell_obs(s, c)) {
            tad[o.snp] += o.ad; tdp[o.snp] += o.dp;
            if (o.ad > 0) cell_alt[o.snp]++;
            if (o.dp > o.ad) cell_ref[o.snp]++;
            ncov[c]++;
        }
    }
    std::vector<char> informative(ns, 0);
    size_t n_cov = 0, n_inf = 0;
    for (size_t j = 0; j < ns; ++j) {
        if (tdp[j] > 0) ++n_cov;
        if (cell_alt[j] >= 10 && cell_ref[j] >= 10) { informative[j] = 1; ++n_inf; }
    }
    std::vector<int> ninf(nc, 0);
    for (int c = 0; c < nc; ++c)
        for (int32_t q = s.dp.indptr[c]; q < s.dp.indptr[c + 1]; ++q) if (informative[s.dp.indices[q]]) ninf[c]++;
    auto med = [](std::vector<int> v) { if (v.empty()) return 0.0; std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return static_cast<double>(v[v.size() / 2]); };
    // Pseudobulk alt fraction over SNPs with >= 20 reads in total.
    const int NB = 20; std::vector<long> hist(NB, 0); long n20 = 0, lo = 0, mid = 0, hi = 0, off = 0;
    for (size_t j = 0; j < ns; ++j) {
        if (tdp[j] < 20) continue;
        const double f = static_cast<double>(tad[j]) / tdp[j];
        hist[std::min(NB - 1, static_cast<int>(f * NB))]++; ++n20;
        if (f < 0.1) ++lo; else if (f > 0.9) ++hi; else if (f >= 0.4 && f <= 0.6) ++mid; else ++off;
    }
    std::system(("mkdir -p '" + out + "'").c_str());
    std::ofstream j(out + "/qc_" + s.id + ".json");
    j << "{\"id\": \"" << s.id << "\", \"n_cells\": " << nc << ", \"covered_snps\": " << n_cov
      << ", \"informative_snps\": " << n_inf << ", \"median_covered_per_cell\": " << med(ncov)
      << ", \"median_informative_per_cell\": " << med(ninf) << ", \"snps_ge20_reads\": " << n20
      << ", \"frac_near0\": " << (n20 ? double(lo) / n20 : 0) << ", \"frac_near_half\": " << (n20 ? double(mid) / n20 : 0)
      << ", \"frac_near1\": " << (n20 ? double(hi) / n20 : 0) << ", \"frac_between\": " << (n20 ? double(off) / n20 : 0)
      << ", \"hist20\": [";
    for (int b = 0; b < NB; ++b) j << (b ? ", " : "") << hist[b];
    j << "]}\n";
    std::cerr << "[qc] " << s.id << ": " << nc << " cells, covered/cell " << med(ncov) << ", informative/cell " << med(ninf) << "\n";
    return 0;
}

int main(int argc, char** argv) {
    std::map<std::string, std::string> A;
    for (int i = 1; i + 1 < argc; i += 2) A[argv[i]] = argv[i + 1];
    const std::string out = A["--out"];
    const double rate = std::atof(A.count("--rate") ? A["--rate"].c_str() : "0.05");
    const int cap = std::atoi(A.count("--cap") ? A["--cap"].c_str() : "2000");
    const double keep_lo = std::atof(A.count("--keep-lo") ? A["--keep-lo"].c_str() : "0.6");
    const double keep_hi = std::atof(A.count("--keep-hi") ? A["--keep-hi"].c_str() : "0.8");
    const uint64_t mseed = std::strtoull(A.count("--mix-seed") ? A["--mix-seed"].c_str() : "1", nullptr, 10);
    std::system(("mkdir -p '" + out + "/runs'").c_str());

    // samples
    std::map<std::string, Sample> reg;
    { std::ifstream f(A["--samples"]); std::string l;
      while (std::getline(f, l)) { std::istringstream ss(l); Sample s; ss >> s.id >> s.dir >> s.calls; if (!s.id.empty() && s.id[0] != '#') reg[s.id] = s; } }
    if (A.count("--qc")) return qc_sample(reg.at(A["--qc"]), out);
    std::vector<std::string> donors;
    { std::stringstream ss(A["--donors"]); std::string t; while (std::getline(ss, t, ',')) donors.push_back(t); }
    std::mt19937_64 rng(mseed);

    for (auto& d : donors) {
        Sample& s = reg.at(d);
        auto called = called_cells(s);
        std::shuffle(called.begin(), called.end(), rng);
        if (static_cast<int>(called.size()) > cap) called.resize(cap);
        s.bc = called;
        std::unordered_map<std::string, int> sel;
        for (size_t i = 0; i < called.size(); ++i) sel[called[i]] = static_cast<int>(i);
        if (!read_mtx_sel(s.dir + "/snp_ad.mtx.gz", s.dir + "/snp_ad_barcodes.tsv.gz", sel, called.size(), s.ad) ||
            !read_mtx_sel(s.dir + "/snp_dp.mtx.gz", s.dir + "/snp_dp_barcodes.tsv.gz", sel, called.size(), s.dp)) return 2;
        std::cerr << "[mix] " << d << ": " << called.size() << " cells, AD nnz " << s.ad.indices.size() << ", DP nnz " << s.dp.indices.size() << "\n";
    }

    // pool, doublet pairs
    struct Src { int d1, c1, d2, c2; };  // d2 = -1 for singlets
    std::vector<std::pair<int, int>> pool;
    for (size_t d = 0; d < donors.size(); ++d) for (size_t c = 0; c < reg[donors[d]].bc.size(); ++c) pool.emplace_back((int)d, (int)c);
    std::shuffle(pool.begin(), pool.end(), rng);
    const size_t P = pool.size();
    size_t n_dbl = static_cast<size_t>(rate * P / (1.0 + rate) + 0.5);
    std::vector<char> used(P, 0);
    std::vector<Src> cols;
    for (size_t k = 0; k < n_dbl; ++k) {
        size_t a = 0; while (a < P && used[a]) ++a;
        if (a >= P) break;
        size_t b = a + 1; while (b < P && (used[b] || pool[b].first == pool[a].first)) ++b;
        if (b >= P) break;
        used[a] = used[b] = 1;
        cols.push_back({pool[a].first, pool[a].second, pool[b].first, pool[b].second});
    }
    for (size_t i = 0; i < P; ++i) if (!used[i]) cols.push_back({pool[i].first, pool[i].second, -1, -1});
    std::shuffle(cols.begin(), cols.end(), rng);

    // mixed matrices and truth
    const int T = static_cast<int>(cols.size());
    Csc ad, dp; ad.ncols = dp.ncols = T; ad.nrows = dp.nrows = reg[donors[0]].ad.nrows;
    ad.indptr.assign(1, 0); dp.indptr.assign(1, 0);
    std::uniform_real_distribution<double> keepd(keep_lo, keep_hi);
    std::ofstream tr(out + "/truth.tsv");
    tr << "cell\tclass\tdonor_a\tdonor_b\torig_a\torig_b\tkeep_a\tkeep_b\n";
    for (int i = 0; i < T; ++i) {
        const Src& s = cols[i];
        Sample& a = reg[donors[s.d1]];
        if (s.d2 < 0) {
            put_obs(cell_obs(a, s.c1), ad, dp);
            tr << "c" << i << "\tsinglet\t" << donors[s.d1] << "\t\t" << a.bc[s.c1] << "\t\t\t\n";
        } else {
            Sample& b = reg[donors[s.d2]];
            auto oa = cell_obs(a, s.c1), ob = cell_obs(b, s.c2);
            const double ka = keepd(rng), kb = keepd(rng);
            thin(oa, ka, rng); thin(ob, kb, rng);
            put_obs(sum_obs(oa, ob), ad, dp);
            tr << "c" << i << "\tdoublet\t" << donors[s.d1] << "\t" << donors[s.d2] << "\t" << a.bc[s.c1] << "\t" << b.bc[s.c2]
               << "\t" << ka << "\t" << kb << "\n";
        }
    }
    tr.close();
    std::cerr << "[mix] " << T << " cells, " << P << " pool, doublets " << (P - T) << "\n";
    { std::ofstream j(out + "/mix.json");
      j << "{\"donors\": \"" << A["--donors"] << "\", \"n_donors\": " << donors.size() << ", \"rate\": " << rate << ", \"cap\": " << cap
        << ", \"keep_lo\": " << keep_lo << ", \"keep_hi\": " << keep_hi
        << ", \"n_cells\": " << T << ", \"n_doublets\": " << (P - T) << ", \"ad_nnz\": " << ad.indices.size() << ", \"dp_nnz\": " << dp.indices.size() << "}\n"; }
    for (auto& d : donors) { reg[d].ad = Csc(); reg[d].dp = Csc(); }

    if (A.count("--export")) {   // private: mixed counts for Vireo and souporcell
        std::system(("mkdir -p '" + A["--export"] + "'").c_str());
        write_mtx(A["--export"] + "/mix_ad.mtx", ad);
        write_mtx(A["--export"] + "/mix_dp.mtx", dp);
        std::cerr << "[mix] exported to " << A["--export"] << "\n";
    }

    // truth donor index per cell (for the diagnostic): singlets -> donor, doublets -> 0.5 on both donors
    std::map<std::string, int> didx;
    for (size_t i = 0; i < donors.size(); ++i) didx[donors[i]] = static_cast<int>(i);

    // runs
    std::stringstream rs(A["--runs"]);
    std::string spec;
    std::ofstream jl(out + "/runs.jsonl", std::ios::app);
    std::ofstream dl(out + "/diag.jsonl", std::ios::app);
    while (rs >> spec) {
        std::vector<std::string> p; { std::stringstream ss(spec); std::string t; while (std::getline(ss, t, ':')) p.push_back(t); }
        const std::string tag = p[0];
        singlet::DonorDemuxConfig cfg;
        cfg.n_donors = std::atoi(p[1].c_str());
        cfg.threads = std::atoi(p[2].c_str());
        cfg.seed = std::strtoull(p[3].c_str(), nullptr, 10);
        const bool diag = p.size() > 4 && p[4] == "diag";
        const auto w0 = std::chrono::steady_clock::now();
        const double c0 = cpu_s();
        auto res = singlet::run_demux(ad.nrows, T, ad.indptr.data(), ad.indices.data(), ad.data.data(), ad.data.size(),
                                      dp.indptr.data(), dp.indices.data(), dp.data.data(), dp.data.size(), cfg);
        const double cpu = cpu_s() - c0;
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
        std::system(("mkdir -p '" + out + "/runs/" + tag + "'").c_str());
        std::ofstream af(out + "/runs/" + tag + "/assign.tsv");
        af << "cell\tdonor_id\tprob_max\tprob_doublet\tlabel\n";
        for (int i = 0; i < T; ++i) {
            const auto& a = res.assignments[i];
            af << "c" << i << "\t" << a.donor_id << "\t" << a.prob_max << "\t" << a.prob_doublet << "\t" << a.label << "\n";
        }
        jl << "{\"tool\": \"singlet\", \"tag\": \"" << tag << "\", \"k_arg\": " << cfg.n_donors << ", \"threads\": " << cfg.threads << ", \"seed\": " << cfg.seed
           << ", \"k_found\": " << res.n_donors_k << ", \"wall_s\": " << wall << ", \"cpu_s\": " << cpu << ", \"n_cells\": " << T << "}\n";
        jl.flush();
        std::cerr << "[mix] run " << tag << ": K=" << res.n_donors_k << " wall " << wall << " s cpu " << cpu << " s\n";

        if (diag && cfg.n_donors >= 2) {
            // ELBO at the optimum the engine finds by itself versus the ELBO of VB started from the truth labels.
            // Same covered-SNP data and the same VB code as run_demux() with K given and no cell filter.
            const int K = cfg.n_donors;
            auto cov = singlet::extract_covered_snps(ad.nrows, T, ad.indptr.data(), ad.indices.data(), ad.data.data(),
                                                     dp.indptr.data(), dp.indices.data(), dp.data.data());
            singlet::PcaCache pca = singlet::compute_pca_cache(cov, singlet::pca_dims_for_k(K), cfg.threads,
                                                               cfg.seed + static_cast<uint64_t>(K) * 100000ULL);
            singlet::VBResult found = (cfg.staged_fit && K > 1 && cov.n_cells >= 200)
                ? singlet::fit_vb_k_staged(cov, K, cfg, &pca, true)
                : singlet::fit_vb_k(cov, K, cfg, std::vector<int>{1, 0, 0, 2, 2, 2, 2}, cfg.max_iter_pre, cfg.max_iter, cfg.threads, &pca, cfg.seed, true);
            // truth init
            std::vector<std::string> tcls(T), da(T), db(T);
            { std::ifstream f(out + "/truth.tsv"); std::string l; std::getline(f, l); int i = 0;
              while (std::getline(f, l) && i < T) {
                  std::vector<std::string> q; std::stringstream ss(l); std::string t; while (std::getline(ss, t, '\t')) q.push_back(t);
                  tcls[i] = q[1]; da[i] = q[2]; db[i] = q.size() > 3 ? q[3] : ""; ++i; } }
            auto make_truth = [&](double soft) {
                singlet::VBResult t;
                t.elbo = -std::numeric_limits<double>::infinity(); t.n_iter = 0;
                t.beta_mu.assign(cov.n_covered_snps, std::vector<float>(K, 0.5f));
                t.beta_sum.assign(cov.n_covered_snps, std::vector<float>(K, 2.0f));
                t.id_prob.assign(T, std::vector<float>(K, 0.0f));
                for (int i = 0; i < T; ++i) {
                    const int a = didx[da[i]];
                    if (tcls[i] == "singlet") {
                        for (int k = 0; k < K; ++k) t.id_prob[i][k] = (k == a) ? 1.0f - soft : soft / (K - 1);
                    } else {
                        const int b = didx[db[i]];
                        t.id_prob[i][a] += 0.5f; t.id_prob[i][b] += 0.5f;
                    }
                }
                return t;
            };
            singlet::VBResult t1 = make_truth(0.0);
            singlet::run_vb_inplace(cov, K, cfg, t1, 1, cfg.threads);          // ELBO right after the truth start
            const double elbo_truth_start = t1.elbo;
            singlet::VBResult tc = make_truth(0.0);
            singlet::run_vb_inplace(cov, K, cfg, tc, cfg.max_iter, cfg.threads); // truth start run to convergence
            auto labels = [&](const singlet::VBResult& r) {
                std::vector<int> l(T);
                for (int i = 0; i < T; ++i) l[i] = static_cast<int>(std::max_element(r.id_prob[i].begin(), r.id_prob[i].end()) - r.id_prob[i].begin());
                return l;
            };
            std::vector<int> lf = labels(found), lt = labels(tc), td, a_f, a_t;
            for (int i = 0; i < T; ++i) if (tcls[i] == "singlet") { td.push_back(didx[da[i]]); a_f.push_back(lf[i]); a_t.push_back(lt[i]); }
            dl << "{\"tag\": \"" << tag << "\", \"K\": " << K << ", \"seed\": " << cfg.seed
               << ", \"elbo_found\": " << found.elbo << ", \"elbo_truth_start\": " << elbo_truth_start
               << ", \"elbo_truth_converged\": " << tc.elbo << ", \"n_iter_found\": " << found.n_iter
               << ", \"n_iter_truth\": " << tc.n_iter << ", \"ari_found_vs_truth\": " << ari(td, a_f)
               << ", \"ari_truth_converged_vs_truth\": " << ari(td, a_t) << ", \"n_cells\": " << T << "}\n";
            dl.flush();
            std::cerr << "[diag] " << tag << ": ELBO found " << found.elbo << ", truth start " << elbo_truth_start
                      << ", truth converged " << tc.elbo << "\n";
        }
    }
    return 0;
}
