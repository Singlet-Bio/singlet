// EM sweeps for the multinomial topic model (PLSA / KL-NMF with simplex columns) -- training aid only.
// Build: bash tools/idea4_build_em.sh   -> bin/idea4_em.dll (called via ctypes from tools/idea4_*.py)
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <omp.h>
template <int K>
static double sweep_k(int64_t n, int G, const int64_t* indptr, const int32_t* idx, const float* x,
                      float* Theta, const float* Phi, float* Phi_acc, int update_theta, int nth) {
    std::vector<std::vector<float>> accs;
    if (Phi_acc) accs.assign(nth, std::vector<float>((size_t)G * K, 0.f));
    double ll_total = 0;
    #pragma omp parallel num_threads(nth) reduction(+:ll_total)
    {
        int tid = omp_get_thread_num();
        float* acc = Phi_acc ? accs[tid].data() : nullptr;
        #pragma omp for schedule(dynamic, 64)
        for (int64_t c = 0; c < n; ++c) {
            float th[K], tn[K];
            for (int j = 0; j < K; ++j) { th[j] = Theta[c * K + j]; tn[j] = 0.f; }
            double ll = 0;
            for (int64_t t = indptr[c]; t < indptr[c + 1]; ++t) {
                const float* ph = Phi + (size_t)idx[t] * K;
                float p = 0.f;
                for (int j = 0; j < K; ++j) p += th[j] * ph[j];
                if (p < 1e-30f) p = 1e-30f;
                float r = x[t] / p;
                ll += (double)x[t] * std::log((double)p);
                for (int j = 0; j < K; ++j) tn[j] += r * ph[j];
                if (acc) { float* a = acc + (size_t)idx[t] * K; for (int j = 0; j < K; ++j) a[j] += r * th[j]; }
            }
            ll_total += ll;
            if (update_theta) {
                float s = 0.f;
                for (int j = 0; j < K; ++j) { tn[j] *= th[j]; s += tn[j]; }
                if (s < 1e-30f) s = 1e-30f;
                for (int j = 0; j < K; ++j) Theta[c * K + j] = tn[j] / s;
            }
        }
    }
    if (Phi_acc) {
        // Phi_acc[g][j] = sum_c r * theta_cj ; multiply by Phi (done here) to get expected counts
        for (int t = 0; t < nth; ++t) {
            const float* a = accs[t].data();
            for (size_t i = 0; i < (size_t)G * K; ++i) Phi_acc[i] += a[i];
        }
        for (size_t i = 0; i < (size_t)G * K; ++i) Phi_acc[i] *= Phi[i];
    }
    return ll_total;
}

extern "C" {
__declspec(dllexport) double em_sweep(int64_t n, int G, int k, const int64_t* indptr, const int32_t* idx, const float* x,
                                      float* Theta, const float* Phi, float* Phi_acc, int update_theta, int nth) {
    if (Phi_acc) std::memset(Phi_acc, 0, sizeof(float) * (size_t)G * k);
    switch (k) {
        case 1: return sweep_k<1>(n, G, indptr, idx, x, Theta, Phi, Phi_acc, update_theta, nth);
        case 2: return sweep_k<2>(n, G, indptr, idx, x, Theta, Phi, Phi_acc, update_theta, nth);
        case 4: return sweep_k<4>(n, G, indptr, idx, x, Theta, Phi, Phi_acc, update_theta, nth);
        case 8: return sweep_k<8>(n, G, indptr, idx, x, Theta, Phi, Phi_acc, update_theta, nth);
        case 12: return sweep_k<12>(n, G, indptr, idx, x, Theta, Phi, Phi_acc, update_theta, nth);
        case 16: return sweep_k<16>(n, G, indptr, idx, x, Theta, Phi, Phi_acc, update_theta, nth);
        case 24: return sweep_k<24>(n, G, indptr, idx, x, Theta, Phi, Phi_acc, update_theta, nth);
        case 32: return sweep_k<32>(n, G, indptr, idx, x, Theta, Phi, Phi_acc, update_theta, nth);
        case 48: return sweep_k<48>(n, G, indptr, idx, x, Theta, Phi, Phi_acc, update_theta, nth);
        case 64: return sweep_k<64>(n, G, indptr, idx, x, Theta, Phi, Phi_acc, update_theta, nth);
    }
    return -1e300;
}

// P[t] = sum_j Theta[c][j] * Phi[g][j] at each nonzero
__declspec(dllexport) void pred_nz(int64_t n, int G, int k, const int64_t* indptr, const int32_t* idx,
                                   const float* Theta, const float* Phi, float* out) {
    for (int64_t c = 0; c < n; ++c)
        for (int64_t t = indptr[c]; t < indptr[c + 1]; ++t) {
            const float* ph = Phi + (size_t)idx[t] * k;
            float p = 0.f;
            for (int j = 0; j < k; ++j) p += Theta[c * k + j] * ph[j];
            out[t] = p;
        }
}

// per-cell log-likelihood sum_t x ln(P) at fixed Theta, Phi
__declspec(dllexport) void cell_ll(int64_t n, int G, int k, const int64_t* indptr, const int32_t* idx, const float* x,
                                   const float* Theta, const float* Phi, double* out) {
    for (int64_t c = 0; c < n; ++c) {
        double ll = 0;
        for (int64_t t = indptr[c]; t < indptr[c + 1]; ++t) {
            const float* ph = Phi + (size_t)idx[t] * k;
            float p = 0.f;
            for (int j = 0; j < k; ++j) p += Theta[c * k + j] * ph[j];
            if (p < 1e-30f) p = 1e-30f;
            ll += (double)x[t] * std::log((double)p);
        }
        out[c] = ll;
    }
}

// Calibration histogram. For every cell c and every gene g: lambda = N[c] * sum_j Theta[c][j]*Phi[g][j];
// bucket b = clamp(floor(4*(log2(lambda)+24)), 0, nb-1) (quarter octaves from 2^-24).
// out_all[b] += 1 for all genes; out_cnt[b*cmax + min(x,cmax)-1] += 1 for present genes.
// out_all2 / out_cnt2: same, keyed additionally by depth bucket d = clamp(floor(log2(N)/2)-2, 0, 7): index (d*nb+b).
__declspec(dllexport) void calib_hist(int64_t n, int G, int k, const int64_t* indptr, const int32_t* idx, const float* x,
                                      const float* Theta, const float* Phi, const double* N, int nb, int cmax,
                                      double* out_all, double* out_cnt, double* out_all2, double* out_cnt2) {
    std::vector<float> lam(G);
    std::vector<int> bkt(G);
    for (int64_t c = 0; c < n; ++c) {
        const float* th = Theta + c * k;
        double Nc = N[c];
        int d = (int)std::floor(std::log2(std::max(Nc, 1.0)) / 2.0) - 2; if (d < 0) d = 0; if (d > 7) d = 7;
        for (int g = 0; g < G; ++g) {
            const float* ph = Phi + (size_t)g * k;
            float p = 0.f;
            for (int j = 0; j < k; ++j) p += th[j] * ph[j];
            double l = Nc * (double)p;
            int b = (int)std::floor(4.0 * (std::log2(std::max(l, 1e-30)) + 24.0));
            if (b < 0) b = 0; if (b >= nb) b = nb - 1;
            bkt[g] = b;
            out_all[b] += 1.0; out_all2[(size_t)d * nb + b] += 1.0;
        }
        for (int64_t t = indptr[c]; t < indptr[c + 1]; ++t) {
            int b = bkt[idx[t]];
            int cc = (int)x[t]; if (cc > cmax) cc = cmax;
            out_cnt[(size_t)b * cmax + cc - 1] += 1.0;
            out_cnt2[((size_t)d * nb + b) * cmax + cc - 1] += 1.0;
        }
    }
}

// Per-gene x depth-bucket presence histogram (training side): pres[d*G+g] += 1 if present, ncell[d] += 1.
__declspec(dllexport) void gene_depth_hist(int64_t n, int G, const int64_t* indptr, const int32_t* idx,
                                           const double* N, double* pres, double* ncell) {
    for (int64_t c = 0; c < n; ++c) {
        int d = (int)std::floor(std::log2(std::max(N[c], 1.0)) / 2.0) - 2; if (d < 0) d = 0; if (d > 7) d = 7;
        ncell[d] += 1.0;
        for (int64_t t = indptr[c]; t < indptr[c + 1]; ++t) pres[(size_t)d * G + idx[t]] += 1.0;
    }
}

// Presence cost (bits) under an explicit per-gene x depth-bucket presence table P[d*G+g] (test side).
__declspec(dllexport) double gene_depth_cost(int64_t n, int G, const int64_t* indptr, const int32_t* idx,
                                             const double* N, const double* P) {
    double bits = 0;
    std::vector<double> logq(G);
    for (int64_t c = 0; c < n; ++c) {
        int d = (int)std::floor(std::log2(std::max(N[c], 1.0)) / 2.0) - 2; if (d < 0) d = 0; if (d > 7) d = 7;
        const double* Pd = P + (size_t)d * G;
        double s = 0;
        for (int g = 0; g < G; ++g) s += std::log2(1.0 - Pd[g]);
        for (int64_t t = indptr[c]; t < indptr[c + 1]; ++t) { int g = idx[t]; s += std::log2(Pd[g]) - std::log2(1.0 - Pd[g]); }
        bits -= s;
    }
    return bits;
}

// Pick, per cell, the codebook entry (nc x k) with the highest multinomial log-likelihood among the `ncand`
// candidates listed in cand[c*ncand + i]; writes the chosen index to out[c].
__declspec(dllexport) void best_code(int64_t n, int G, int k, const int64_t* indptr, const int32_t* idx, const float* x,
                                     const float* Phi, const float* Code, int nc, const int32_t* cand, int ncand, int32_t* out) {
    for (int64_t c = 0; c < n; ++c) {
        double best = -1e300; int bi = cand[c * ncand];
        for (int i = 0; i < ncand; ++i) {
            int ci = cand[c * ncand + i];
            const float* th = Code + (size_t)ci * k;
            double ll = 0;
            for (int64_t t = indptr[c]; t < indptr[c + 1]; ++t) {
                const float* ph = Phi + (size_t)idx[t] * k;
                float p = 0.f;
                for (int j = 0; j < k; ++j) p += th[j] * ph[j];
                if (p < 1e-30f) p = 1e-30f;
                ll += (double)x[t] * std::log((double)p);
            }
            if (ll > best) { best = ll; bi = ci; }
        }
        out[c] = bi;
    }
}
}
