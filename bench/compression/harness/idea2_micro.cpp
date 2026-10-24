#define I2_COUNT 1
// micro-benchmark of the order-statistic coder (decode cost per symbol)
#include "idea2m_core.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <windows.h>
static double cpu_ms() { FILETIME c, e, k, u; GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u); return (((unsigned long long)u.dwHighDateTime << 32) | u.dwLowDateTime) / 1e4; }
using namespace cs::i2;
int main(int argc, char** argv) {
    u32 L = argc > 1 ? atoi(argv[1]) : 50000;
    u32 N = argc > 2 ? atoi(argv[2]) : 200;
    std::mt19937_64 rng(1);
    std::vector<u64> W(L), Q(L + 1, 0);
    for (u32 i = 0; i < L; ++i) W[i] = (u64)(1e9 / (1.0 + i)) + 1000 + (rng() % 1000);
    for (u32 i = L; i > 0; --i) Q[i - 1] = Q[i] + W[i - 1];
    // generate N sorted draws iid from W
    std::vector<u32> xs;
    std::vector<double> cdf(L);
    double acc = 0; for (u32 i = 0; i < L; ++i) { acc += (double)W[i]; cdf[i] = acc; }
    int reps = 2000;
    std::vector<std::vector<u32>> sets(reps);
    for (auto& s : sets) {
        s.clear();
        for (u32 k = 0; k < N; ++k) { double u = (double)(rng() >> 11) / 9007199254740992.0 * acc; s.push_back((u32)(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin())); }
        std::sort(s.begin(), s.end()); s.erase(std::unique(s.begin(), s.end()), s.end());
    }
    for (int variant = 0; variant < 2; ++variant) {
    LqTab LQ; LQ.build(Q.data(), L);
    std::vector<u8> out;
    RcEnc enc(out);
    u64 tot = 0;
    for (auto& s : sets) {
        u32 lo = 0, n = (u32)s.size();
        for (u32 i = 0; i < n; ++i) {
            if (variant == 0) code_orderstat(enc, Q.data(), lo, L, n - i, s[i]);
            else code_orderstat_fast(enc, LQ, lo, L, n - i, s[i]);
            lo = s[i] + 1; ++tot;
        }
    }
    enc.finish();
    double bits = 0; for (int c = 0; c < 24; ++c) bits += enc.cost[c];
    RcDec dec(out.data(), out.data() + out.size());
    double c0_ = cpu_ms(); auto t0 = std::chrono::steady_clock::now();
    u64 bad = 0;
    for (auto& s : sets) {
        u32 lo = 0, n = (u32)s.size();
        for (u32 i = 0; i < n; ++i) {
            u32 x = variant == 0 ? code_orderstat(dec, Q.data(), lo, L, n - i, 0) : code_orderstat_fast(dec, LQ, lo, L, n - i, 0);
            if (x != s[i]) ++bad; lo = x + 1;
        }
    }
    double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); double ms = cpu_ms() - c0_; (void)wall;
    std::printf("  C() calls so far %llu\n", (unsigned long long)cs::i2::g_ccalls);
    std::printf("%s L=%u N=%u symbols=%llu bytes=%zu bits/sym=%.3f bad=%llu decode %.1f ms = %.0f ns/sym\n", variant ? "fast " : "exact", L, N, (unsigned long long)tot, out.size(), bits / tot, (unsigned long long)bad, ms, ms * 1e6 / tot);
    }
}
