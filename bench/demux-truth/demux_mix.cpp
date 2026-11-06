// SPDX-License-Identifier: MIT
// In-silico donor mixtures with known truth, run through the engine's run_demux().
//
// Input: per-sample donor/snp_ad + snp_dp matrices (Matrix Market, gz) from single-donor runs with --snps.
// A mixture takes the called cells of N samples (one donor each), keeps them as distinct cells (the
// relabeling of barcodes per donor is the column relabeling here), and plants cross-donor doublets by
// merging two cells of different donors (AD and DP are summed, which is what the pileup of the merged
// reads would give). Columns are shuffled. Only truth labels and per-cell assignments are written.
//
//   demux_mix --samples samples.tsv --donors S1,S2,S3 --rate 0.05 --cap 2000 --mix-seed 1 --out DIR
//             --runs "tag:K:threads:seed tag2:K:threads:seed ..."      (K = -1 for auto)
//
// samples.tsv columns: id, private dir (holds snp_ad.mtx.gz, snp_dp.mtx.gz and the *_barcodes.tsv.gz), cell_calls.tsv
// Build: g++ -O2 -std=c++17 -pthread -I<repo>/include demux_mix.cpp -o demux_mix

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
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
    std::vector<uint8_t> data;
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
    std::vector<uint8_t> vals;
    while (fgets(buf.data(), buf.size(), f)) {
        char* e;
        long r = strtol(buf.data(), &e, 10);
        long c = strtol(e, &e, 10);
        long v = strtol(e, &e, 10);
        const int nc2 = colmap[c - 1];
        if (nc2 < 0) continue;
        rc.emplace_back(nc2, static_cast<int32_t>(r - 1));
        vals.push_back(static_cast<uint8_t>(v > 255 ? 255 : (v < 0 ? 0 : v)));
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

static void emit(const Csc& a, int c, std::vector<int32_t>& ix, std::vector<uint8_t>& v) {
    for (int32_t q = a.indptr[c]; q < a.indptr[c + 1]; ++q) { ix.push_back(a.indices[q]); v.push_back(a.data[q]); }
}
static void emit_sum(const Csc& a, int ca, const Csc& b, int cb, std::vector<int32_t>& ix, std::vector<uint8_t>& v) {
    int32_t i = a.indptr[ca], ie = a.indptr[ca + 1], j = b.indptr[cb], je = b.indptr[cb + 1];
    while (i < ie || j < je) {
        if (j >= je || (i < ie && a.indices[i] < b.indices[j])) { ix.push_back(a.indices[i]); v.push_back(a.data[i]); ++i; }
        else if (i >= ie || b.indices[j] < a.indices[i]) { ix.push_back(b.indices[j]); v.push_back(b.data[j]); ++j; }
        else { ix.push_back(a.indices[i]); v.push_back(static_cast<uint8_t>(std::min(255, a.data[i] + b.data[j]))); ++i; ++j; }
    }
}

static double cpu_s() {
    rusage ru; getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_stime.tv_sec + 1e-6 * (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec);
}

int main(int argc, char** argv) {
    std::map<std::string, std::string> A;
    for (int i = 1; i + 1 < argc; i += 2) A[argv[i]] = argv[i + 1];
    const std::string out = A["--out"];
    const double rate = std::atof(A.count("--rate") ? A["--rate"].c_str() : "0.05");
    const int cap = std::atoi(A.count("--cap") ? A["--cap"].c_str() : "2000");
    const uint64_t mseed = std::strtoull(A.count("--mix-seed") ? A["--mix-seed"].c_str() : "1", nullptr, 10);
    std::system(("mkdir -p '" + out + "/runs'").c_str());

    // samples
    std::map<std::string, Sample> reg;
    { std::ifstream f(A["--samples"]); std::string l;
      while (std::getline(f, l)) { std::istringstream ss(l); Sample s; ss >> s.id >> s.dir >> s.calls; if (!s.id.empty() && s.id[0] != '#') reg[s.id] = s; } }
    std::vector<std::string> donors;
    { std::stringstream ss(A["--donors"]); std::string t; while (std::getline(ss, t, ',')) donors.push_back(t); }
    std::mt19937_64 rng(mseed);

    for (auto& d : donors) {
        Sample& s = reg.at(d);
        std::unordered_set<std::string> inad;
        for (auto& b : zlines(s.dir + "/snp_ad_barcodes.tsv.gz")) inad.insert(b.substr(0, b.find('\t')));
        std::vector<std::string> called;
        { std::ifstream f(s.calls); std::string l; std::getline(f, l);
          while (std::getline(f, l)) { std::istringstream ss(l); std::string bc, is; ss >> bc >> is; if (is == "TRUE" && inad.count(bc)) called.push_back(bc); } }
        std::sort(called.begin(), called.end());
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
    std::ofstream tr(out + "/truth.tsv");
    tr << "cell\tclass\tdonor_a\tdonor_b\torig_a\torig_b\n";
    for (int i = 0; i < T; ++i) {
        const Src& s = cols[i];
        Sample& a = reg[donors[s.d1]];
        if (s.d2 < 0) {
            emit(a.ad, s.c1, ad.indices, ad.data); emit(a.dp, s.c1, dp.indices, dp.data);
            tr << "c" << i << "\tsinglet\t" << donors[s.d1] << "\t\t" << a.bc[s.c1] << "\t\n";
        } else {
            Sample& b = reg[donors[s.d2]];
            emit_sum(a.ad, s.c1, b.ad, s.c2, ad.indices, ad.data); emit_sum(a.dp, s.c1, b.dp, s.c2, dp.indices, dp.data);
            tr << "c" << i << "\tdoublet\t" << donors[s.d1] << "\t" << donors[s.d2] << "\t" << a.bc[s.c1] << "\t" << b.bc[s.c2] << "\n";
        }
        ad.indptr.push_back(static_cast<int32_t>(ad.indices.size()));
        dp.indptr.push_back(static_cast<int32_t>(dp.indices.size()));
    }
    tr.close();
    std::cerr << "[mix] " << T << " cells, " << (P - 2 * 0) << " pool, doublets " << (P - T) << "\n";
    { std::ofstream j(out + "/mix.json");
      j << "{\"donors\": \"" << A["--donors"] << "\", \"n_donors\": " << donors.size() << ", \"rate\": " << rate << ", \"cap\": " << cap
        << ", \"n_cells\": " << T << ", \"n_doublets\": " << (P - T) << ", \"ad_nnz\": " << ad.indices.size() << ", \"dp_nnz\": " << dp.indices.size() << "}\n"; }
    for (auto& d : donors) { reg[d].ad = Csc(); reg[d].dp = Csc(); }

    // runs
    std::stringstream rs(A["--runs"]);
    std::string spec;
    std::ofstream jl(out + "/runs.jsonl", std::ios::app);
    while (rs >> spec) {
        std::vector<std::string> p; { std::stringstream ss(spec); std::string t; while (std::getline(ss, t, ':')) p.push_back(t); }
        const std::string tag = p[0];
        singlet::DonorDemuxConfig cfg;
        cfg.n_donors = std::atoi(p[1].c_str());
        cfg.threads = std::atoi(p[2].c_str());
        cfg.seed = std::strtoull(p[3].c_str(), nullptr, 10);
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
        jl << "{\"tag\": \"" << tag << "\", \"k_arg\": " << cfg.n_donors << ", \"threads\": " << cfg.threads << ", \"seed\": " << cfg.seed
           << ", \"k_found\": " << res.n_donors_k << ", \"wall_s\": " << wall << ", \"cpu_s\": " << cpu << ", \"n_cells\": " << T << "}\n";
        jl.flush();
        std::cerr << "[mix] run " << tag << ": K=" << res.n_donors_k << " wall " << wall << " s cpu " << cpu << " s\n";
    }
    return 0;
}
