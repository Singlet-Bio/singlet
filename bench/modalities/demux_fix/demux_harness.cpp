// SPDX-License-Identifier: MIT
// Donor-demux harness: load Singlet's donor/snp_ad.mtx.gz + snp_dp.mtx.gz (--output-format mtx),
// restrict to the called cells and run run_demux() for one or more configurations, writing
// donor_assignments.tsv (called cells) and summary.json per run. The matrices are parsed once.
//
//   demux_harness AD.mtx.gz DP.mtx.gz BARCODES.tsv.gz CELL_CALLS.tsv OUTDIR THREADS \
//       [TAG K [key=value ...]] [-- TAG K [key=value ...]] ...
// K = -1 for auto. Config keys: detect_doublets, doublet_threshold, doublet_rounds, doublet_crossfit,
// doublet_prior, geno_error, min_prob, max_donors, n_inits, seed, k_select_sep_z, k_select_sep_min, n_ward_inits, min_donor_cells, min_donor_frac, geno_conc_hom, geno_conc_het, k_select_patience,
// k_select_inits, k_select_max_iter, k_select_max_cells, k_select_min_snp_cells, fast_k_select.
//
// Build (Clipper job):  g++ -std=c++20 -O2 -pthread -I<repo>/include demux_harness.cpp -o demux_harness

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "singlet/pileup/donor_demux.h"

struct Csc {
    int32_t nrows = 0, ncols = 0;
    std::vector<int32_t> indptr, indices;
    std::vector<uint8_t> data;
};

static bool read_mtx_gz(const std::string& path, Csc& m) {
    const std::string cmd = "zcat '" + path + "'";
    FILE* f = popen(cmd.c_str(), "r");
    if (!f) return false;
    std::vector<char> buf(1 << 16);
    int64_t nnz = 0;
    // header
    while (fgets(buf.data(), buf.size(), f)) {
        if (buf[0] == '%') continue;
        long long r, c, n;
        if (sscanf(buf.data(), "%lld %lld %lld", &r, &c, &n) != 3) { pclose(f); return false; }
        m.nrows = static_cast<int32_t>(r); m.ncols = static_cast<int32_t>(c); nnz = n;
        break;
    }
    std::vector<int32_t> rows(nnz), cols(nnz);
    std::vector<uint8_t> vals(nnz);
    int64_t i = 0;
    while (i < nnz && fgets(buf.data(), buf.size(), f)) {
        char* p = buf.data();
        char* e;
        long r = strtol(p, &e, 10); p = e;
        long c = strtol(p, &e, 10); p = e;
        long v = strtol(p, &e, 10);
        rows[i] = static_cast<int32_t>(r - 1);
        cols[i] = static_cast<int32_t>(c - 1);
        vals[i] = static_cast<uint8_t>(v > 255 ? 255 : (v < 0 ? 0 : v));
        ++i;
    }
    pclose(f);
    if (i != nnz) { std::cerr << "short mtx " << path << ": " << i << " of " << nnz << "\n"; return false; }
    m.indptr.assign(m.ncols + 1, 0);
    for (int64_t k = 0; k < nnz; ++k) m.indptr[cols[k] + 1]++;
    for (int j = 0; j < m.ncols; ++j) m.indptr[j + 1] += m.indptr[j];
    m.indices.resize(nnz);
    m.data.resize(nnz);
    std::vector<int32_t> pos(m.indptr.begin(), m.indptr.end() - 1);
    for (int64_t k = 0; k < nnz; ++k) {
        const int32_t q = pos[cols[k]]++;
        m.indices[q] = rows[k];
        m.data[q] = vals[k];
    }
    // sort rows within each column (extract_covered_snps binary-searches them)
    std::vector<std::pair<int32_t, uint8_t>> tmp;
    for (int j = 0; j < m.ncols; ++j) {
        const int32_t a = m.indptr[j], b = m.indptr[j + 1];
        bool sorted = true;
        for (int32_t q = a + 1; q < b; ++q) if (m.indices[q] < m.indices[q - 1]) { sorted = false; break; }
        if (sorted) continue;
        tmp.clear();
        for (int32_t q = a; q < b; ++q) tmp.push_back({m.indices[q], m.data[q]});
        std::sort(tmp.begin(), tmp.end());
        for (int32_t q = a; q < b; ++q) { m.indices[q] = tmp[q - a].first; m.data[q] = tmp[q - a].second; }
    }
    return true;
}

static std::vector<std::string> read_lines_gz(const std::string& path) {
    std::vector<std::string> out;
    FILE* f = popen(("zcat -f '" + path + "'").c_str(), "r");
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

static void apply_kv(singlet::DonorDemuxConfig& cfg, const std::string& k, const std::string& v) {
    const double d = atof(v.c_str());
    const int i = atoi(v.c_str());
    if (k == "detect_doublets") cfg.detect_doublets = i != 0;
    else if (k == "doublet_threshold") cfg.doublet_threshold = static_cast<float>(d);
    else if (k == "doublet_rounds") cfg.doublet_rounds = i;
    else if (k == "doublet_temper") cfg.doublet_temper = static_cast<float>(d);
    else if (k == "doublet_crossfit") cfg.doublet_crossfit = i != 0;
    else if (k == "doublet_prior") cfg.doublet_prior = static_cast<float>(d);
    else if (k == "doublet_rate_per_1000") cfg.doublet_rate_per_1000 = static_cast<float>(d);
    else if (k == "doublet_min_snp_cells") cfg.doublet_min_snp_cells = i;
    else if (k == "geno_error") cfg.geno_error = static_cast<float>(d);
    else if (k == "min_prob") cfg.min_prob = static_cast<float>(d);
    else if (k == "max_donors") cfg.max_donors = i;
    else if (k == "n_inits") cfg.n_inits = i;
    else if (k == "max_iter") cfg.max_iter = i;
    else if (k == "seed") cfg.seed = static_cast<uint64_t>(atoll(v.c_str()));
    else if (k == "k_select_sep_z") cfg.k_select_sep_z = d;
    else if (k == "k_select_z") cfg.k_select_z = d;
    else if (k == "k_select_min_gain") cfg.k_select_min_gain = d;
    else if (k == "k_select_sep_min") cfg.k_select_sep_min = d;
    else if (k == "n_ward_inits") cfg.n_ward_inits = i;
    else if (k == "staged_fit") cfg.staged_fit = i != 0;
    else if (k == "explore_ward") cfg.explore_ward = i;
    else if (k == "explore_legacy") cfg.explore_legacy = i;
    else if (k == "explore_random") cfg.explore_random = i;
    else if (k == "explore_top") cfg.explore_top = i;
    else if (k == "explore_iters") cfg.explore_iters = i;
    else if (k == "core_min_rel") cfg.core_min_rel = static_cast<float>(d);
    else if (k == "split_merge_rounds") cfg.split_merge_rounds = i;
    else if (k == "split_merge_iters") cfg.split_merge_iters = i;
    else if (k == "split_merge_resplit_pairs") cfg.split_merge_resplit_pairs = i;
    else if (k == "min_donor_cells") cfg.min_donor_cells = i;
    else if (k == "min_donor_frac") cfg.min_donor_frac = static_cast<float>(d);
    else if (k == "geno_conc_hom") cfg.geno_conc_hom = static_cast<float>(d);
    else if (k == "geno_conc_het") cfg.geno_conc_het = static_cast<float>(d);
    else if (k == "k_select_patience") cfg.k_select_patience = i;
    else if (k == "k_select_inits") cfg.k_select_inits = i;
    else if (k == "k_select_max_iter") cfg.k_select_max_iter = i;
    else if (k == "k_select_max_cells") cfg.k_select_max_cells = i;
    else if (k == "k_select_min_snp_cells") cfg.k_select_min_snp_cells = i;
    else if (k == "fast_k_select") cfg.fast_k_select = i != 0;
    else std::cerr << "unknown key " << k << "\n";
}

int main(int argc, char** argv) {
    if (argc < 9) {
        std::cerr << "usage: demux_harness AD DP BARCODES CELL_CALLS OUTDIR THREADS TAG K [key=value ...] [-- TAG K ...]\n";
        return 1;
    }
    const std::string ad_path = argv[1], dp_path = argv[2], bc_path = argv[3], calls_path = argv[4];
    const std::string outdir = argv[5];
    const int threads = atoi(argv[6]);

    auto t0 = std::chrono::steady_clock::now();
    Csc ad, dp;
    if (!read_mtx_gz(ad_path, ad) || !read_mtx_gz(dp_path, dp)) { std::cerr << "mtx read failed\n"; return 1; }
    std::vector<std::string> barcodes = read_lines_gz(bc_path);
    std::cerr << "[harness] AD " << ad.nrows << "x" << ad.ncols << " nnz=" << ad.data.size()
              << "  DP nnz=" << dp.data.size() << "  barcodes=" << barcodes.size() << "  load "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";

    // called cells
    std::unordered_map<std::string, uint32_t> bc_idx;
    for (size_t i = 0; i < barcodes.size(); ++i) bc_idx[barcodes[i]] = static_cast<uint32_t>(i);
    std::vector<uint32_t> cell_indices;
    {
        auto lines = read_lines_gz(calls_path);
        // column chosen by env CALLS_COL (default is_cell); a cell is TRUE / 1
        const char* cc = getenv("CALLS_COL");
        const std::string col = cc ? cc : "is_cell";
        auto split = [](const std::string& l) {
            std::vector<std::string> f;
            size_t s = 0;
            for (size_t t = 0; t <= l.size(); ++t)
                if (t == l.size() || l[t] == '\t') { f.push_back(l.substr(s, t - s)); s = t + 1; }
            return f;
        };
        size_t ci = 1;
        if (!lines.empty()) {
            auto h = split(lines[0]);
            for (size_t i = 0; i < h.size(); ++i) if (h[i] == col) ci = i;
        }
        for (size_t i = 1; i < lines.size(); ++i) {
            auto f = split(lines[i]);
            if (f.size() <= ci) continue;
            const std::string& flag = f[ci];
            if (flag != "TRUE" && flag != "True" && flag != "1") continue;
            auto it = bc_idx.find(f[0]);
            if (it != bc_idx.end()) cell_indices.push_back(it->second);
        }
        std::sort(cell_indices.begin(), cell_indices.end());
    }
    std::cerr << "[harness] called cells: " << cell_indices.size() << "\n";

    // run groups
    int a = 7;
    while (a < argc) {
        if (std::string(argv[a]) == "--") { ++a; continue; }
        if (a + 1 >= argc) break;
        if (std::string(argv[a]) == "vbtruth" && a + 3 < argc) {
            // vbtruth TAG TRUTH.tsv.gz K: diagnostic. Run VB from the truth partition (demuxlet singlets)
            // on the called cells and print its ELBO, to compare with the ELBO of a fitted solution.
            const std::string tag = argv[a + 1], truth = argv[a + 2];
            const int K = atoi(argv[a + 3]);
            a += 4;
            auto cov = singlet::extract_covered_snps(ad.nrows, ad.ncols, ad.indptr.data(), ad.indices.data(),
                ad.data.data(), dp.indptr.data(), dp.indices.data(), dp.data.data());
            singlet::CoveredSNPData cu;
            cu.n_covered_snps = cov.n_covered_snps;
            cu.n_cells = static_cast<int>(cell_indices.size());
            for (uint32_t idx : cell_indices) cu.cells.push_back(cov.cells[idx]);
            std::unordered_map<std::string, int> donor_of;
            std::unordered_map<std::string, int> ids;
            for (const auto& l : read_lines_gz(truth)) {
                std::vector<std::string> f;
                size_t s = 0;
                for (size_t t = 0; t <= l.size(); ++t)
                    if (t == l.size() || l[t] == '\t') { f.push_back(l.substr(s, t - s)); s = t + 1; }
                if (f.size() < 8 || f[7] != "singlet" || f[4] != "ctrl") continue;
                if (!ids.count(f[3])) { const int n = static_cast<int>(ids.size()); ids[f[3]] = n; }
                donor_of[f[0].substr(0, f[0].find('-'))] = ids[f[3]];
            }
            singlet::DonorDemuxConfig cfg;
            cfg.threads = threads;
            singlet::VBResult r;
            r.elbo = -1e300; r.n_iter = 0;
            r.beta_mu.assign(cu.n_covered_snps, std::vector<float>(K, 0.5f));
            r.beta_sum.assign(cu.n_covered_snps, std::vector<float>(K, 2.0f));
            r.id_prob.assign(cu.n_cells, std::vector<float>(K, 1.0f / K));
            int nl = 0;
            for (int i = 0; i < cu.n_cells; ++i) {
                auto it = donor_of.find(barcodes[cell_indices[i]]);
                if (it == donor_of.end() || it->second >= K) continue;
                ++nl;
                for (int k = 0; k < K; ++k) r.id_prob[i][k] = (k == it->second) ? 0.97f : 0.03f / (K - 1);
            }
            singlet::run_vb_inplace(cu, K, cfg, r, 300, threads);
            std::cerr << "[harness] vbtruth " << tag << ": labelled=" << nl << " ELBO=" << r.elbo
                      << " iters=" << r.n_iter << "\n";
            continue;
        }
        if (std::string(argv[a]) == "rescore" && a + 3 < argc) {
            // rescore TAG PREV_donor_assignments.tsv K [key=value ...]: rerun only the doublet stage
            // (hard donor weights from PREV's best_singlet) so doublet parameters can be swept quickly.
            const std::string tag = argv[a + 1], prev = argv[a + 2];
            const int K = atoi(argv[a + 3]);
            a += 4;
            singlet::DonorDemuxConfig cfg;
            cfg.threads = threads;
            while (a < argc && std::string(argv[a]) != "--") {
                std::string kv = argv[a++];
                const size_t eq = kv.find('=');
                if (eq != std::string::npos) apply_kv(cfg, kv.substr(0, eq), kv.substr(eq + 1));
            }
            static bool have_cov = false;
            static singlet::CoveredSNPData cov_use;
            if (!have_cov) {
                auto cov = singlet::extract_covered_snps(ad.nrows, ad.ncols, ad.indptr.data(), ad.indices.data(),
                    ad.data.data(), dp.indptr.data(), dp.indices.data(), dp.data.data());
                cov_use.n_covered_snps = cov.n_covered_snps;
                cov_use.n_cells = static_cast<int>(cell_indices.size());
                for (uint32_t idx : cell_indices) cov_use.cells.push_back(std::move(cov.cells[idx]));
                have_cov = true;
            }
            auto lines = read_lines_gz(prev);
            std::vector<std::vector<float>> q(cell_indices.size(), std::vector<float>(K, 0.0f));
            std::vector<int> bs(cell_indices.size(), 0);
            std::vector<float> pmax(cell_indices.size(), 0.0f);
            for (size_t i = 1; i < lines.size() && i - 1 < q.size(); ++i) {
                std::vector<std::string> f;
                size_t s = 0;
                for (size_t t = 0; t <= lines[i].size(); ++t)
                    if (t == lines[i].size() || lines[i][t] == '\t') { f.push_back(lines[i].substr(s, t - s)); s = t + 1; }
                if (f.size() < 6) continue;
                const int k = atoi(f[4].c_str() + 5);   // best_singlet "donorK"
                bs[i - 1] = k;
                pmax[i - 1] = static_cast<float>(atof(f[2].c_str()));
                for (int j = 0; j < K; ++j) q[i - 1][j] = (j == k) ? 0.999f : 0.001f / (K - 1);
            }
            std::vector<singlet::CellScore> dbl;
            float eps = 0.0f;
            const auto tr = std::chrono::steady_clock::now();
            singlet::doublet_stage(cov_use, K, cfg, q, dbl, &eps);
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - tr).count();
            std::vector<singlet::DonorAssignment> as(cell_indices.size());
            std::vector<std::string> bcs;
            int n_dbl = 0;
            for (size_t i = 0; i < as.size(); ++i) {
                auto& x = as[i];
                x.best_singlet = bs[i];
                x.prob_doublet = dbl[i].p_doublet;
                x.doublet_a = dbl[i].pa; x.doublet_b = dbl[i].pb; x.doublet_llr = dbl[i].llr;
                x.prob_max = pmax[i];
                x.donor_id = K; x.label = "unassigned";
                if (dbl[i].p_doublet > cfg.doublet_threshold) { x.label = "doublet"; ++n_dbl; }
                else if (pmax[i] >= cfg.min_prob) { x.donor_id = bs[i]; x.label = "donor" + std::to_string(bs[i]); }
                bcs.push_back(barcodes[cell_indices[i]]);
            }
            const std::string rd = outdir + "/" + tag;
            if (system(("mkdir -p '" + rd + "'").c_str()) != 0) return 1;
            singlet::write_donor_assignments(rd + "/donor_assignments.tsv", as, bcs);
            std::cerr << "[harness] rescore " << tag << ": doublets=" << n_dbl << " eps=" << eps << " " << wall << " s\n";
            continue;
        }
        const std::string tag = argv[a];
        const int K = atoi(argv[a + 1]);
        a += 2;
        singlet::DonorDemuxConfig cfg;
        cfg.threads = threads;
        cfg.n_donors = K;
        std::string subset_path;   // subset=FILE: barcodes (no -1 suffix), one per line; restricts the called cells
        while (a < argc && std::string(argv[a]) != "--") {
            std::string kv = argv[a++];
            const size_t eq = kv.find('=');
            if (eq == std::string::npos) continue;
            if (kv.substr(0, eq) == "subset") { subset_path = kv.substr(eq + 1); continue; }
            apply_kv(cfg, kv.substr(0, eq), kv.substr(eq + 1));
        }
        const std::string rd = outdir + "/" + tag;
        std::string mk = "mkdir -p '" + rd + "'";
        if (system(mk.c_str()) != 0) return 1;

        std::vector<uint32_t> run_cells = cell_indices;
        if (!subset_path.empty()) {
            std::unordered_map<std::string, int> keep;
            for (const auto& l : read_lines_gz(subset_path)) keep[l.substr(0, l.find('-'))] = 1;
            run_cells.clear();
            for (uint32_t c : cell_indices)
                if (keep.count(barcodes[c].substr(0, barcodes[c].find('-')))) run_cells.push_back(c);
        }

        std::cerr << "[harness] === run " << tag << " K=" << K << " cells=" << run_cells.size() << " ===\n";
        const auto tr = std::chrono::steady_clock::now();
        singlet::DemuxResult res = singlet::run_demux(
            ad.nrows, ad.ncols, ad.indptr.data(), ad.indices.data(), ad.data.data(), ad.data.size(),
            dp.indptr.data(), dp.indices.data(), dp.data.data(), dp.data.size(), cfg, run_cells);
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - tr).count();

        // called cells only
        std::vector<singlet::DonorAssignment> as;
        std::vector<std::string> bcs;
        as.reserve(run_cells.size());
        for (uint32_t c : run_cells) { as.push_back(res.assignments[c]); bcs.push_back(barcodes[c]); }
        singlet::write_donor_assignments(rd + "/donor_assignments.tsv", as, bcs);

        int n_dbl = 0, n_un = 0;
        for (const auto& x : as) { if (x.label == "doublet") ++n_dbl; else if (x.label == "unassigned") ++n_un; }
        std::ofstream js(rd + "/summary.json");
        js << "{\"tag\":\"" << tag << "\",\"n_donors_arg\":" << K << ",\"n_donors_k\":" << res.n_donors_k
           << ",\"wall_s\":" << wall << ",\"threads\":" << threads << ",\"cells\":" << as.size()
           << ",\"doublets_called\":" << n_dbl << ",\"unassigned\":" << n_un
           << ",\"geno_error\":" << res.geno_error << ",\"k_selection\":[";
        for (size_t i = 0; i < res.k_selection.size(); ++i) {
            const auto& r = res.k_selection[i];
            js << (i ? "," : "") << "{\"K\":" << r.K << ",\"ll_per_cell\":" << r.ll_per_cell
               << ",\"gain\":" << r.gain << ",\"z\":" << r.z << ",\"accepted\":" << (r.accepted ? "true" : "false")
               << ",\"seconds\":" << r.seconds << "}";
        }
        js << "]}\n";
        std::cerr << "[harness] " << tag << ": K=" << res.n_donors_k << " wall=" << wall
                  << " s doublets=" << n_dbl << " unassigned=" << n_un << "\n";
    }
    return 0;
}
