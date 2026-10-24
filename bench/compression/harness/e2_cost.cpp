// SPDX-License-Identifier: MIT
// E2 phase 1: ideal code length (bits) of a block-adaptive rank-1 Poisson presence model, double precision.
// This is the exploration tool for the model design (not the codec): it never touches a coder, it just
// evaluates  sum_cols  -log2 P(column | model fitted on all earlier blocks)  for cells and empties separately.
//   bin/e2_cost.exe [--B n] [--n0 n] [--rho x] [--bpo n] [--gt 0|1] [--mix 0|1] [--rounds n] file.1pz ...
//   Model: hazard h_fj = A_f * C_b(j) (natural units), p = 1 - exp(-h).  A_f = Atab[z_f], z_f = running presence count.
#include "common.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_map>

using namespace cs;
static const double LN2 = 0.6931471805599453;

struct Cfg {
    int bpo = 4;                 // depth buckets per octave
    int n0 = 64;                 // first block (columns)
    double rho = 2.0;            // geometric growth of block size until Bmax
    int Bmax = 1024;
    int gt = 1;                  // 1: Good-Turing mass for unseen features, 0: pseudo-count a_pc
    double a_pc = 0.05;
    int gtmax = 8;               // Good-Turing adjustment for z <= gtmax
    int mix = 1;                 // 1: Phi mixture inversion over depth buckets; 0: single mean exposure
    double ctrust = 20;          // bucket's own C weight: nh/(nh+ctrust)
    int rounds = 2;
    int ctxmode = 0;             // count context: 0 hazard-bits (half octave), 1 z-class x bucket
    double flat_frac = 0.7;      // flat prior: expected nnz = min(frac*N, cap)
    double cbeta = 2.0;          // count prior mass
    int whole = 0;               // 1: in-sample static (fit on the whole class first, no adaptation)
    int perm = 1;                // random coding order
    double emp_kappa = 0.0;      // empties: prior pseudo-columns from the cells profile (0 = flat start)
    double theta = 0.0;          // empties: redistribute low-z mass in proportion to (cells hazard)^theta
    int zlim = 2;                // ... for features with z <= zlim
    int verbose = 0;
};

static int bucket_of(uint64_t N, int bpo) { return (int)std::floor(std::log2((double)std::max<uint64_t>(N, 1)) * bpo + 1e-9); }

struct Model {
    Cfg cfg;
    uint32_t m;
    int nbk;
    std::vector<uint32_t> z, hz;
    std::vector<double> nh, Tn, Ns, C, Atab, Af;
    double Stot = 0;
    uint64_t ncols = 0;
    static constexpr int NCX = 64, NS = 17;
    std::vector<std::array<double, NS>> ctab;
    std::vector<double> prior_pp;   // optional per-feature prior presence count weight (empties from cells)
    std::vector<double> prior_w;    // optional per-feature redistribution weights
    Model(const Cfg& c, uint32_t m_, int nbk_) : cfg(c), m(m_), nbk(nbk_), z(m_, 0), hz(2, 0), nh(nbk_, 0), Tn(nbk_, 0), Ns(nbk_, 0),
          C(nbk_, 1), Af(m_, 0) {
        hz[0] = m;
        ctab.assign(NCX, std::array<double, NS>{});
    }
    static double interp(const std::vector<double>& xs, const std::vector<double>& ys, double t) {
        if (t <= xs.front()) return ys.front();
        if (t >= xs.back()) return ys.back();
        size_t k = std::upper_bound(xs.begin(), xs.end(), t) - xs.begin();
        double x0 = xs[k - 1], x1 = xs[k];
        double w = x1 > x0 ? (t - x0) / (x1 - x0) : 0.0;
        return ys[k - 1] + w * (ys[k] - ys[k - 1]);
    }
    void flat() {
        double a = 1.0 / m;
        Atab.assign(1, a);
        std::fill(Af.begin(), Af.end(), a);
        Stot = a * m;
        for (int b = 0; b < nbk; ++b) {
            double Nb = std::pow(2.0, (b + 0.5) / cfg.bpo);
            double kt = std::min(cfg.flat_frac * Nb, 0.2 * m);
            C[b] = -(double)m * std::log1p(-kt / m);
        }
    }
    void fit() {
        if (ncols == 0) { flat(); return; }
        size_t zmax = hz.size() - 1;
        while (zmax > 0 && hz[zmax] == 0) --zmax;
        std::vector<double> tgt(zmax + 1);
        for (size_t zz = 0; zz <= zmax; ++zz) tgt[zz] = (double)zz;
        if (cfg.gt) {
            double n0 = hz[0], n1 = hz.size() > 1 ? hz[1] : 0;
            tgt[0] = n0 > 0 ? std::max(n1 / n0, 1e-5) : 1e-5;
            for (size_t zz = 1; zz <= (size_t)cfg.gtmax && zz + 1 < hz.size(); ++zz)
                if (hz[zz] >= 5 && hz[zz + 1] >= 5) {
                    double g = (zz + 1.0) * hz[zz + 1] / hz[zz];
                    tgt[zz] = std::min<double>(std::max(g, 0.4 * zz), (double)zz);
                }
        } else {
            for (size_t zz = 0; zz <= zmax; ++zz) tgt[zz] += cfg.a_pc;
        }
        double nc = (double)ncols;
        for (auto& t : tgt) t = std::min(t, nc - 0.5);
        // initial C: previous value, buckets never seen -> proportional
        std::vector<double> Nbar(nbk);
        for (int b = 0; b < nbk; ++b) Nbar[b] = nh[b] > 0 ? Ns[b] / nh[b] : std::pow(2.0, (b + 0.5) / cfg.bpo);
        // grids
        const int NG = 1500;
        std::vector<double> xg(NG), Phi(NG);
        for (int k = 0; k < NG; ++k) xg[k] = -28.0 + 30.0 * k / (NG - 1);
        std::vector<double> A(zmax + 1);
        for (int r = 0; r < cfg.rounds; ++r) {
            // A step
            if (cfg.mix) {
                for (int k = 0; k < NG; ++k) {
                    double a = std::exp(xg[k]), s = 0;
                    for (int b = 0; b < nbk; ++b) if (nh[b] > 0) s += nh[b] * -std::expm1(-a * C[b]);
                    Phi[k] = s;
                }
            } else {
                double cbar = 0;
                for (int b = 0; b < nbk; ++b) cbar += nh[b] * C[b];
                cbar /= nc;
                for (int k = 0; k < NG; ++k) Phi[k] = nc * -std::expm1(-std::exp(xg[k]) * cbar);
            }
            for (size_t zz = 0; zz <= zmax; ++zz) A[zz] = std::exp(interp(Phi, xg, tgt[zz]));
            // C step: Psi(y) over log-binned z histogram
            std::vector<double> gA, gH;
            {
                std::vector<double> sa(160, 0), sh(160, 0);
                for (size_t zz = 0; zz <= zmax; ++zz) {
                    if (!hz[zz]) continue;
                    int bi = zz < 32 ? (int)zz : 32 + (int)(8 * std::log2(zz / 32.0));
                    bi = std::min(bi, 159);
                    sh[bi] += hz[zz];
                    sa[bi] += hz[zz] * std::log(A[zz]);
                }
                for (int bi = 0; bi < 160; ++bi) if (sh[bi] > 0) { gH.push_back(sh[bi]); gA.push_back(std::exp(sa[bi] / sh[bi])); }
            }
            const int NY = 700;
            std::vector<double> yg(NY), Psi(NY);
            for (int k = 0; k < NY; ++k) {
                yg[k] = -6.0 + 24.0 * k / (NY - 1);
                double e = std::exp(yg[k]), s = 0;
                for (size_t i = 0; i < gA.size(); ++i) s += gH[i] * -std::expm1(-gA[i] * e);
                Psi[k] = s;
            }
            std::vector<double> Cn(nbk, 0);
            for (int b = 0; b < nbk; ++b) if (nh[b] > 0) Cn[b] = std::exp(interp(Psi, yg, std::max(Tn[b] / nh[b], 1e-3)));
            // smooth: weighted log-linear fit of log C on log Nbar, blended
            double sw = 0, mx = 0, my = 0;
            for (int b = 0; b < nbk; ++b) if (nh[b] > 0) { double w = std::min(nh[b], 50.0); sw += w; mx += w * std::log(Nbar[b]); my += w * std::log(Cn[b]); }
            mx /= sw; my /= sw;
            double sxx = 0, sxy = 0;
            for (int b = 0; b < nbk; ++b) if (nh[b] > 0) { double w = std::min(nh[b], 50.0); double dx = std::log(Nbar[b]) - mx; sxx += w * dx * dx; sxy += w * dx * (std::log(Cn[b]) - my); }
            double slope = sxx > 1e-9 ? sxy / sxx : 1.0;
            slope = std::min(std::max(slope, 0.3), 1.5);
            for (int b = 0; b < nbk; ++b) {
                double fitv = my + slope * (std::log(Nbar[b]) - mx);
                double lam = nh[b] / (nh[b] + cfg.ctrust);
                C[b] = std::exp(nh[b] > 0 ? lam * std::log(Cn[b]) + (1 - lam) * fitv : fitv);
            }
        }
        Atab = A;
        // per-feature hazards (optionally with per-feature prior counts)
        if (!prior_pp.empty() && cfg.emp_kappa > 0) {
            // recompute A_f from continuous targets z_f + kappa * pp_f using the last Phi grid
            Stot = 0;
            for (uint32_t f = 0; f < m; ++f) {
                double t = z[f] + prior_pp[f];
                t = std::min(t, nc - 0.5);
                if (z[f] == 0 && t < tgt[0]) t = tgt[0];
                Af[f] = std::exp(interp(Phi, xg, t));
                Stot += Af[f];
            }
        } else {
            Stot = 0;
            for (uint32_t f = 0; f < m; ++f) { Af[f] = Atab[z[f]]; }
            if (!prior_w.empty() && cfg.theta > 0) {
                std::vector<double> sw(cfg.zlim + 1, 0), cz(cfg.zlim + 1, 0);
                for (uint32_t f = 0; f < m; ++f) if (z[f] <= (uint32_t)cfg.zlim) { sw[z[f]] += prior_w[f]; cz[z[f]] += 1; }
                for (uint32_t f = 0; f < m; ++f) if (z[f] <= (uint32_t)cfg.zlim) Af[f] = Atab[z[f]] * prior_w[f] / (sw[z[f]] / cz[z[f]]);
            }
            for (uint32_t f = 0; f < m; ++f) Stot += Af[f];
        }
    }
    // cost of a block (columns given in coding order), returns {presence bits, count bits}; then update.
    void run_block(const CanonicalMatrix& M, const uint32_t* cols, size_t nc, double& pb, double& cb, uint64_t& nnz_out) {
        pb = cb = 0;
        nnz_out = 0;
        for (size_t t = 0; t < nc; ++t) {
            uint32_t j = cols[t];
            uint32_t k0 = M.indptr[j], k1 = M.indptr[j + 1];
            uint64_t N = 0;
            for (uint32_t q = k0; q < k1; ++q) N += M.data[q];
            if (N == 0) continue;
            int b = bucket_of(N, cfg.bpo);
            double Cj = C[b];
            double hsum = 0;
            for (uint32_t q = k0; q < k1; ++q) {
                uint32_t f = M.indices[q];
                double h = Af[f] * Cj;
                hsum += h;
                pb += h > 1e-5 ? -std::log2(-std::expm1(-h)) : -std::log2(h * (1 - 0.5 * h));
                // count symbol
                uint32_t v = M.data[q];
                int cx = count_ctx(h, f);
                int s = std::min<uint32_t>(v - 1, 16);
                auto& row = ctab[cx];
                double tot = 0, mass = 0;
                for (int i = 0; i < NS; ++i) {
                    double g = i < 16 ? std::ldexp(1.0, -(i + 1)) : std::ldexp(1.0, -16);
                    double pv = row[i] + cfg.cbeta * g;
                    tot += pv;
                    if (i == s) mass = pv;
                }
                cb += -std::log2(mass / tot);
                if (v > 16) cb += 2 * std::floor(std::log2((double)(v - 15))) + 1;
            }
            pb += (Cj * Stot - hsum) / LN2;
            nnz_out += k1 - k0;
        }
    }
    int count_ctx(double h, uint32_t f) const {
        if (cfg.ctxmode == 0) return std::min(NCX - 1, std::max(0, (int)std::floor(2 * std::log2(h / LN2)) + 48));
        double pz = (double)z[f] / (double)std::max<uint64_t>(ncols, 1);
        return std::min(NCX - 1, std::max(0, (int)std::floor(2 * std::log2(pz + 1e-7)) + 28));
    }
    void train_counts(const CanonicalMatrix& M, const uint32_t* cols, size_t nc) {
        for (size_t t = 0; t < nc; ++t) {
            uint32_t j = cols[t];
            uint32_t k0 = M.indptr[j], k1 = M.indptr[j + 1];
            uint64_t N = 0;
            for (uint32_t q = k0; q < k1; ++q) N += M.data[q];
            if (N == 0) continue;
            double Cj = C[bucket_of(N, cfg.bpo)];
            for (uint32_t q = k0; q < k1; ++q) {
                uint32_t f = M.indices[q];
                ctab[count_ctx(Af[f] * Cj, f)][std::min<uint32_t>(M.data[q] - 1, 16)] += 1;
            }
        }
    }
    void update(const CanonicalMatrix& M, const uint32_t* cols, size_t nc, bool counts = true) {
        if (counts) train_counts(M, cols, nc);
        for (size_t t = 0; t < nc; ++t) {
            uint32_t j = cols[t];
            uint32_t k0 = M.indptr[j], k1 = M.indptr[j + 1];
            uint64_t N = 0;
            for (uint32_t q = k0; q < k1; ++q) N += M.data[q];
            if (N == 0) continue;
            int b = bucket_of(N, cfg.bpo);
            nh[b] += 1; Tn[b] += k1 - k0; Ns[b] += (double)N;
            ++ncols;
            if (hz.size() < ncols + 2) hz.resize(ncols + 2 + ncols / 2, 0);
            for (uint32_t q = k0; q < k1; ++q) {
                uint32_t f = M.indices[q];
                uint32_t zo = z[f];
                --hz[zo]; ++hz[zo + 1];
                z[f] = zo + 1;
            }
        }
    }
};

struct ClassResult { double pres = 0, cnt = 0; uint64_t nnz = 0; int blocks = 0; std::vector<double> blk; };

static ClassResult run_class(const CanonicalMatrix& M, std::vector<uint32_t> cols, const Cfg& cfg, int nbk, Model* keep = nullptr,
                             const Model* prior = nullptr) {
    ClassResult R;
    Model mod(cfg, M.m, nbk);
    if (prior && cfg.theta > 0) {
        mod.prior_w.assign(M.m, 0);
        double fl = 1e300;
        for (uint32_t f = 0; f < M.m; ++f) fl = std::min(fl, prior->Af[f]);
        for (uint32_t f = 0; f < M.m; ++f) mod.prior_w[f] = std::pow(std::max(prior->Af[f], fl), cfg.theta);
    }
    if (prior && cfg.emp_kappa > 0) {
        // per-feature prior presence counts from the cells model: kappa * (1-exp(-A_f * Cbar_E)) with Cbar_E guessed = 1.5 * median N later
        mod.prior_pp.assign(M.m, 0);
        double cb = 3.0;   // typical empties exposure (UMIs)
        for (uint32_t f = 0; f < M.m; ++f) mod.prior_pp[f] = cfg.emp_kappa * -std::expm1(-prior->Af[f] * cb);
    }
    if (cfg.perm) { std::mt19937_64 g(12345); std::shuffle(cols.begin(), cols.end(), g); }
    size_t pos = 0, n = cols.size();
    if (cfg.whole) {
        mod.update(M, cols.data(), n, false);
        mod.fit();
        for (auto& r : mod.ctab) r.fill(0);
        mod.train_counts(M, cols.data(), n);
        double pb, cb; uint64_t nz;
        mod.run_block(M, cols.data(), n, pb, cb, nz);
        R.pres = pb; R.cnt = cb; R.nnz = nz; R.blocks = 1;
        return R;
    }
    while (pos < n) {
        size_t sz = pos == 0 ? cfg.n0 : (size_t)std::max<double>(cfg.n0, std::min<double>(cfg.Bmax, (cfg.rho - 1.0) * pos));
        size_t e = std::min(n, pos + sz);
        mod.fit();
        double pb, cb; uint64_t nz;
        mod.run_block(M, cols.data() + pos, e - pos, pb, cb, nz);
        R.pres += pb; R.cnt += cb; R.nnz += nz; ++R.blocks;
        R.blk.push_back(nz ? (pb + cb) / nz : 0);
        mod.update(M, cols.data() + pos, e - pos);
        pos = e;
    }
    if (keep) { mod.fit(); *keep = std::move(mod); }
    return R;
}

// adaptive entropy of a symbol stream with context (KT-like, alphabet size A)
static double adaptive_bits(const std::vector<uint32_t>& sym, const std::vector<uint32_t>& ctx, uint32_t A, uint32_t NC, double alpha = 0.3) {
    std::vector<double> cnt((size_t)NC * A, 0), tot(NC, 0);
    double bits = 0;
    for (size_t i = 0; i < sym.size(); ++i) {
        size_t c = ctx[i];
        double p = (cnt[c * A + sym[i]] + alpha) / (tot[c] + alpha * A);
        bits -= std::log2(p);
        cnt[c * A + sym[i]] += 1; tot[c] += 1;
    }
    return bits;
}

static std::string parent_dir(const std::string& p) {
    auto s = p.find_last_of("/\\");
    return s == std::string::npos ? "." : p.substr(0, s);
}
static std::string leaf(const std::string& p) {
    auto s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}

int main(int argc, char** argv) {
    Cfg cfg;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--B") cfg.Bmax = std::stoi(nx());
        else if (a == "--n0") cfg.n0 = std::stoi(nx());
        else if (a == "--rho") cfg.rho = std::stod(nx());
        else if (a == "--bpo") cfg.bpo = std::stoi(nx());
        else if (a == "--gt") cfg.gt = std::stoi(nx());
        else if (a == "--apc") cfg.a_pc = std::stod(nx());
        else if (a == "--gtmax") cfg.gtmax = std::stoi(nx());
        else if (a == "--mix") cfg.mix = std::stoi(nx());
        else if (a == "--ctrust") cfg.ctrust = std::stod(nx());
        else if (a == "--rounds") cfg.rounds = std::stoi(nx());
        else if (a == "--ctx") cfg.ctxmode = std::stoi(nx());
        else if (a == "--flat") cfg.flat_frac = std::stod(nx());
        else if (a == "--whole") cfg.whole = std::stoi(nx());
        else if (a == "--perm") cfg.perm = std::stoi(nx());
        else if (a == "--kappa") cfg.emp_kappa = std::stod(nx());
        else if (a == "--theta") cfg.theta = std::stod(nx());
        else if (a == "--zlim") cfg.zlim = std::stoi(nx());
        else if (a == "-v") cfg.verbose = 1;
        else files.push_back(a);
    }
    for (auto& f : files) {
        CanonicalMatrix M = load_1pz(f);
        std::string dir = parent_dir(f);
        std::unordered_map<std::string, int> called;
        {
            std::ifstream in(dir + "/cell_calls.tsv");
            std::string line;
            std::getline(in, line);
            while (std::getline(in, line)) called[line.substr(0, line.find('\t'))] = 1;
        }
        std::vector<uint32_t> cells, empt;
        std::vector<uint8_t> isc(M.n, 0);
        for (uint32_t j = 0; j < M.n; ++j) {
            bool c = called.count(M.colnames[j]) > 0;
            isc[j] = c;
            (c ? cells : empt).push_back(j);
        }
        uint64_t Nmax = 1;
        for (uint32_t j = 0; j < M.n; ++j) { uint64_t s = 0; for (uint32_t q = M.indptr[j]; q < M.indptr[j + 1]; ++q) s += M.data[q]; Nmax = std::max(Nmax, s); }
        int nbk = bucket_of(Nmax, cfg.bpo) + 2;
        // header: depth symbols (exponent + 3 mantissa bits) with prev-exponent context, original order; flags with depth ctx
        std::vector<uint32_t> hs, hc, fs, fc;
        double raw = 0;
        uint32_t prev = 0;
        for (uint32_t j = 0; j < M.n; ++j) {
            uint64_t N = 0;
            for (uint32_t q = M.indptr[j]; q < M.indptr[j + 1]; ++q) N += M.data[q];
            uint64_t v = N + 1;
            int e = 63 - __builtin_clzll(v);
            uint32_t mant = e >= 3 ? (uint32_t)((v >> (e - 3)) & 7) : 0;
            hs.push_back(e * 8 + mant); hc.push_back(prev);
            raw += std::max(e - 3, 0);
            prev = e;
            fs.push_back(isc[j]); fc.push_back(e);
        }
        double hdr = adaptive_bits(hs, hc, 64 * 8 / 8 * 8, 64) + raw;
        double flg = adaptive_bits(fs, fc, 2, 64);
        Model cells_m(cfg, M.m, nbk);
        ClassResult rc = run_class(M, cells, cfg, nbk, &cells_m);
        ClassResult re = run_class(M, empt, cfg, nbk, nullptr, &cells_m);
        uint64_t nnz = rc.nnz + re.nnz;
        std::printf("%-22s %-6s B=%-5d n0=%-3d rho=%.2f bpo=%d gt=%d mix=%d | cells nnz=%-8llu pres %.3f cnt %.3f = %.3f | empt nnz=%-8llu pres %.3f cnt %.3f = %.3f | hdr %.3f flag %.4f b/nnz | ALL %.3f\n",
                    leaf(dir).c_str(), leaf(f).substr(0, 6).c_str(), cfg.Bmax, cfg.n0, cfg.rho, cfg.bpo, cfg.gt, cfg.mix,
                    (unsigned long long)rc.nnz, rc.nnz ? rc.pres / rc.nnz : 0, rc.nnz ? rc.cnt / rc.nnz : 0, rc.nnz ? (rc.pres + rc.cnt) / rc.nnz : 0,
                    (unsigned long long)re.nnz, re.nnz ? re.pres / re.nnz : 0, re.nnz ? re.cnt / re.nnz : 0, re.nnz ? (re.pres + re.cnt) / re.nnz : 0,
                    hdr / nnz, flg / nnz, (rc.pres + rc.cnt + re.pres + re.cnt + hdr + flg) / nnz);
        if (cfg.verbose) {
            std::printf("   cells blocks:");
            for (size_t i = 0; i < rc.blk.size(); ++i) std::printf(" %.2f", rc.blk[i]);
            std::printf("\n   empt blocks:");
            for (size_t i = 0; i < re.blk.size() && i < 24; ++i) std::printf(" %.2f", re.blk[i]);
            std::printf("\n");
        }
        std::fflush(stdout);
    }
    return 0;
}
