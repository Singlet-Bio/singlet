// SPDX-License-Identifier: MIT
// Genotype doublet model and auto-K selection of donor_demux.h on synthetic pools.
//
// Pools: K donors with 0/1/2 genotypes drawn from per-SNP allele frequencies, cells with
// ~150 SNP observations each (alt reads ~ Binomial(depth, 0.01 / 0.5 / 0.99)), planted
// heterotypic doublets whose reads come from two donors 50/50.
//   1. K given: donors recovered (ARI), doublet AUROC high, doublet calls precise.
//   2. Auto-K: recovers K = 1 (single donor), 3, 5 and 8, and a pool with doublets only
//      from a few donor pairs does not grow an extra "doublet donor".
//   3. DonorAssignment fields are filled (prob_doublet, best_singlet, doublet pair) and
//      the TSV writer emits them.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/donor_demux.h"

using namespace singlet;

static int n_pass = 0, n_fail = 0;
#define CHECK(cond, msg) \
    do { \
        if (cond) { std::cout << "  PASS: " << (msg) << "\n"; ++n_pass; } \
        else      { std::cout << "  FAIL: " << (msg) << " [line " << __LINE__ << "]\n"; ++n_fail; } \
    } while (0)

struct Pool {
    int nsnp = 0, ncell = 0, K = 0;
    std::vector<int32_t> ad_ip, ad_ix, dp_ip, dp_ix;
    std::vector<uint16_t> ad_v, dp_v;
    std::vector<int> donor;       // true donor (first donor for doublets)
    std::vector<char> dbl;        // 1 = planted heterotypic doublet
};

// pair_limit > 0: doublets only between donors in the first `pair_limit` donors
static Pool make_pool(int K, int ncell, int nsnp, int obs, double dbl_rate, uint64_t seed,
                      int pair_limit = 0)
{
    Pool P;
    P.nsnp = nsnp; P.ncell = ncell; P.K = K;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    std::vector<std::vector<float>> theta(K, std::vector<float>(nsnp));
    for (int j = 0; j < nsnp; ++j) {
        const double p = 0.1 + 0.8 * U(rng);
        for (int k = 0; k < K; ++k) {
            const int g = (U(rng) < p) + (U(rng) < p);
            theta[k][j] = (g == 0) ? 0.01f : (g == 1) ? 0.5f : 0.99f;
        }
    }

    P.ad_ip.push_back(0);
    P.dp_ip.push_back(0);
    std::vector<int> perm(nsnp);
    std::iota(perm.begin(), perm.end(), 0);
    for (int c = 0; c < ncell; ++c) {
        int d1 = static_cast<int>(U(rng) * K) % K;
        int d2 = -1;
        bool is_dbl = false;
        if (K >= 2 && U(rng) < dbl_rate) {
            const int lim = pair_limit > 1 ? pair_limit : K;
            d1 = static_cast<int>(U(rng) * lim) % lim;
            do { d2 = static_cast<int>(U(rng) * lim) % lim; } while (d2 == d1);
            is_dbl = true;
        }
        P.donor.push_back(d1);
        P.dbl.push_back(is_dbl ? 1 : 0);
        // partial shuffle -> obs distinct SNPs
        for (int i = 0; i < obs; ++i) {
            const int r = i + static_cast<int>(U(rng) * (nsnp - i)) % (nsnp - i);
            std::swap(perm[i], perm[r]);
        }
        std::vector<int> snps(perm.begin(), perm.begin() + obs);
        std::sort(snps.begin(), snps.end());
        for (int j : snps) {
            const int depth = 1 + static_cast<int>(-std::log(1.0 - U(rng)) * 0.7);
            int alt = 0;
            for (int r = 0; r < depth; ++r) {
                const int dn = (is_dbl && U(rng) < 0.5) ? d2 : d1;
                if (U(rng) < theta[dn][j]) ++alt;
            }
            P.dp_ix.push_back(j);
            P.dp_v.push_back(static_cast<uint8_t>(std::min(depth, 255)));
            if (alt > 0) {
                P.ad_ix.push_back(j);
                P.ad_v.push_back(static_cast<uint8_t>(alt));
            }
        }
        P.ad_ip.push_back(static_cast<int32_t>(P.ad_ix.size()));
        P.dp_ip.push_back(static_cast<int32_t>(P.dp_ix.size()));
    }
    return P;
}

static DemuxResult run(const Pool& P, DonorDemuxConfig cfg) {
    return run_demux(P.nsnp, P.ncell,
        P.ad_ip.data(), P.ad_ix.data(), P.ad_v.data(), P.ad_v.size(),
        P.dp_ip.data(), P.dp_ix.data(), P.dp_v.data(), P.dp_v.size(), cfg);
}

static double ari(const std::vector<int>& a, const std::vector<int>& b) {
    std::map<std::pair<int, int>, double> nij;
    std::map<int, double> ai, bi;
    for (size_t i = 0; i < a.size(); ++i) { nij[{a[i], b[i]}] += 1; ai[a[i]] += 1; bi[b[i]] += 1; }
    auto c2 = [](double x) { return x * (x - 1.0) / 2.0; };
    double sij = 0, sa = 0, sb = 0;
    for (auto& kv : nij) sij += c2(kv.second);
    for (auto& kv : ai) sa += c2(kv.second);
    for (auto& kv : bi) sb += c2(kv.second);
    const double n = static_cast<double>(a.size());
    const double exp = sa * sb / c2(n);
    const double mx = 0.5 * (sa + sb);
    return (mx - exp) < 1e-12 ? 1.0 : (sij - exp) / (mx - exp);
}

static double auroc(const std::vector<double>& s, const std::vector<char>& y) {
    std::vector<int> o(s.size());
    std::iota(o.begin(), o.end(), 0);
    std::sort(o.begin(), o.end(), [&](int a, int b) { return s[a] < s[b]; });
    double rank_sum = 0, npos = 0, nneg = 0;
    for (size_t i = 0; i < o.size();) {
        size_t j = i;
        while (j < o.size() && s[o[j]] == s[o[i]]) ++j;
        const double r = 0.5 * (i + j - 1) + 1.0;
        for (size_t t = i; t < j; ++t) if (y[o[t]]) { rank_sum += r; npos += 1; } else nneg += 1;
        i = j;
    }
    return (rank_sum - npos * (npos + 1) / 2.0) / (npos * nneg);
}

static DonorDemuxConfig base_cfg() {
    DonorDemuxConfig cfg;
    cfg.threads = 4;
    cfg.seed = 7;
    cfg.n_inits = 4;
    cfg.doublet_temper = 1.0f;   // synthetic reads are independent binomial draws (no overdispersion to correct)
    return cfg;
}

int main() {
    std::cout << "=== K given: donors and doublets ===\n";
    {
        Pool P = make_pool(4, 1600, 2500, 150, 0.10, 11);
        DonorDemuxConfig cfg = base_cfg();
        cfg.n_donors = 4;
        DemuxResult r = run(P, cfg);
        CHECK(r.n_donors_k == 4, "K=4 respected");

        std::vector<int> truth, pred;
        std::vector<double> score;
        int called_dbl = 0, called_dbl_true = 0, n_dbl = 0;
        for (int c = 0; c < P.ncell; ++c) {
            const auto& a = r.assignments[c];
            score.push_back(a.prob_doublet);
            if (P.dbl[c]) ++n_dbl;
            if (a.label == "doublet") { ++called_dbl; if (P.dbl[c]) ++called_dbl_true; }
            if (!P.dbl[c]) { truth.push_back(P.donor[c]); pred.push_back(a.label == "doublet" ? 99 : (a.label == "unassigned" ? 98 : a.donor_id)); }
        }
        const double auc = auroc(score, P.dbl);
        const double a = ari(truth, pred);
        std::cout << "  singlet ARI=" << a << " doublet AUROC=" << auc << " called=" << called_dbl
                  << " (true " << called_dbl_true << " of " << n_dbl << ") eps=" << r.geno_error << "\n";
        CHECK(a > 0.97, "singlet ARI > 0.97 with planted doublets");
        CHECK(auc > 0.95, "doublet AUROC > 0.95");
        CHECK(called_dbl > 0 && static_cast<double>(called_dbl_true) / called_dbl > 0.85,
              "doublet calls: precision > 0.85");
        CHECK(static_cast<double>(called_dbl_true) / n_dbl > 0.6, "doublet calls: recall > 0.6");

        // fields
        int with_pair = 0, with_best = 0;
        for (const auto& as : r.assignments) {
            if (as.doublet_a >= 0 && as.doublet_b > as.doublet_a) ++with_pair;
            if (as.best_singlet >= 0) ++with_best;
        }
        CHECK(with_pair == P.ncell && with_best == P.ncell, "best_singlet / doublet pair filled for every cell");

        // writer emits pair names and best singlet
        std::vector<std::string> bc(P.ncell);
        for (int c = 0; c < P.ncell; ++c) bc[c] = "BC" + std::to_string(c);
        const std::string path = (std::filesystem::temp_directory_path() / "demux_dbl_assign.tsv").string();
        CHECK(write_donor_assignments(path, r.assignments, bc), "write_donor_assignments");
        std::ifstream f(path);
        std::string line;
        std::getline(f, line);
        bool pair_names = false, ok_cols = true;
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::string tok;
            int n = 0;
            std::string bd;
            while (std::getline(ss, tok, '\t')) { if (n == 5) bd = tok; ++n; }
            if (n != 7) ok_cols = false;
            if (bd.find(",donor") != std::string::npos) pair_names = true;
        }
        CHECK(ok_cols, "assignments TSV keeps 7 columns");
        CHECK(pair_names, "best_doublet column carries 'donorA,donorB'");
        f.close();
        std::error_code ec;
        std::filesystem::remove(path, ec);

        // doublet model off: nothing is called doublet, prob_doublet stays 0
        DonorDemuxConfig off = cfg;
        off.detect_doublets = false;
        DemuxResult r0 = run(P, off);
        int nd = 0; float mx = 0;
        for (const auto& as : r0.assignments) { if (as.label == "doublet") ++nd; mx = std::max(mx, as.prob_doublet); }
        CHECK(nd == 0 && mx == 0.0f, "detect_doublets=false: no doublet calls");

        // library default (tempered posteriors): ranking stays perfect, calls stay precise
        DonorDemuxConfig dflt = cfg;
        dflt.doublet_temper = DonorDemuxConfig{}.doublet_temper;
        DemuxResult rd = run(P, dflt);
        std::vector<double> sd_score;
        int cd = 0, cd_true = 0;
        for (int c = 0; c < P.ncell; ++c) {
            sd_score.push_back(rd.assignments[c].prob_doublet);
            if (rd.assignments[c].label == "doublet") { ++cd; if (P.dbl[c]) ++cd_true; }
        }
        CHECK(auroc(sd_score, P.dbl) > 0.95, "default temper: doublet AUROC > 0.95");
        CHECK(cd > 0 && static_cast<double>(cd_true) / cd > 0.95, "default temper: doublet precision > 0.95");
    }

    std::cout << "=== Auto-K ===\n";
    {
        // single donor: must give 1, everything donor0
        Pool P = make_pool(1, 800, 2000, 150, 0.0, 21);
        DonorDemuxConfig cfg = base_cfg();
        cfg.n_donors = -1;
        DemuxResult r = run(P, cfg);
        CHECK(r.n_donors_k == 1, "single donor pool: auto-K = 1");
        int d0 = 0;
        for (const auto& a : r.assignments) if (a.label == "donor0") ++d0;
        CHECK(d0 == P.ncell, "single donor pool: every cell donor0");
    }
    for (int Kt : {3, 5, 8, 12}) {
        Pool P = make_pool(Kt, 300 * Kt, 3500, 150, 0.08, 100 + Kt);
        DonorDemuxConfig cfg = base_cfg();
        cfg.n_donors = -1;
        DemuxResult r = run(P, cfg);
        std::cout << "  K_true=" << Kt << " -> K=" << r.n_donors_k << "\n";
        CHECK(r.n_donors_k == Kt, "auto-K recovers K=" + std::to_string(Kt));
        std::vector<int> truth, pred;
        for (int c = 0; c < P.ncell; ++c)
            if (!P.dbl[c]) {
                truth.push_back(P.donor[c]);
                const auto& a = r.assignments[c];
                pred.push_back(a.label == "doublet" ? 99 : (a.label == "unassigned" ? 98 : a.donor_id));
            }
        const double a = ari(truth, pred);
        std::cout << "    singlet ARI=" << a << "\n";
        CHECK(a > 0.95, "auto-K K=" + std::to_string(Kt) + ": singlet ARI > 0.95");
    }
    {
        // doublets concentrated on two donor pairs must not become an extra donor
        Pool P = make_pool(4, 1600, 2500, 150, 0.15, 55, /*pair_limit=*/3);
        DonorDemuxConfig cfg = base_cfg();
        cfg.n_donors = -1;
        DemuxResult r = run(P, cfg);
        CHECK(r.n_donors_k == 4, "concentrated doublets: no extra donor (K=4)");
    }

    std::cout << "=== identical results at any thread count ===\n";
    {
        Pool P = make_pool(4, 1200, 2000, 120, 0.10, 77);
        DemuxResult ref;
        for (int th : {1, 2, 3, 8}) {
            DonorDemuxConfig cfg = base_cfg();
            cfg.n_donors = -1;
            cfg.threads = th;
            DemuxResult r = run(P, cfg);
            if (th == 1) { ref = r; CHECK(r.n_donors_k >= 2, "auto-K finds donors (threads=1)"); continue; }
            bool same = r.n_donors_k == ref.n_donors_k && r.geno_error == ref.geno_error &&
                        r.assignments.size() == ref.assignments.size();
            for (size_t c = 0; same && c < r.assignments.size(); ++c)
                same = r.assignments[c].label == ref.assignments[c].label &&
                       r.assignments[c].prob_max == ref.assignments[c].prob_max &&
                       r.assignments[c].prob_doublet == ref.assignments[c].prob_doublet;
            CHECK(same, "auto-K and every assignment bit-identical at threads=" + std::to_string(th));
        }
    }

    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
