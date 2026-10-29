// Track B size-floor experiment: adaptive binary context-mixing model of the rank-space gap-class stream (and value
// flags), to bound what better entropy coding / richer contexts can buy over the static pos x depth tables.
// Reports ideal code length (no actual bitstream; arithmetic-coder overhead is ~0.01%).
//   bin/track_b_cm.exe file.1pz ...
#include "common.h"
#include "track_b_util.h"
#include <cmath>

using namespace cs;
using namespace cs::tb;

static inline float stretchf(float p) { return std::log(p / (1.0f - p)); }
static inline float squashf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

struct Slot {
    float p = 0.5f;
    uint16_t n = 0;
};

struct BinModel {  // table of adaptive probabilities
    std::vector<Slot> t;
    explicit BinModel(size_t n) : t(n) {}
    inline float get(size_t i) const { return t[i].p; }
    inline void update(size_t i, int bit, float limit_rate) {
        Slot& s = t[i];
        float rate = 1.0f / (static_cast<float>(s.n) + 1.6f);
        if (rate < limit_rate) rate = limit_rate;
        s.p += (static_cast<float>(bit) - s.p) * rate;
        if (s.p < 0.0005f) s.p = 0.0005f;
        if (s.p > 0.9995f) s.p = 0.9995f;
        if (s.n < 1000) ++s.n;
    }
};

struct Mixer {
    int nin;
    std::vector<float> w;
    std::vector<float> x;
    size_t sel = 0;
    float pr = 0.5f;
    Mixer(int nin_, size_t nsets, float w0) : nin(nin_), w(nsets * nin_, w0), x(nin_) {}
    inline float mix(size_t s) {
        sel = s;
        float dot = 0;
        for (int i = 0; i < nin; ++i) dot += w[sel * nin + i] * x[i];
        pr = squashf(dot);
        if (pr < 0.0003f) pr = 0.0003f;
        if (pr > 0.9997f) pr = 0.9997f;
        return pr;
    }
    inline void update(int bit, float lr) {
        float err = (static_cast<float>(bit) - pr) * lr;
        for (int i = 0; i < nin; ++i) w[sel * nin + i] += err * x[i];
    }
};

int main(int argc, char** argv) {
    for (int a = 1; a < argc; ++a) {
        CanonicalMatrix M = load_1pz(argv[a]);
        RowPerm P = make_perm(M);
        RankMat R = to_rank_space(M, P);
        const double nnz = static_cast<double>(M.nnz);
        std::printf("== %s m=%u n=%u nnz=%llu\n", argv[a], M.m, M.n, (unsigned long long)M.nnz);
        // model sets: bitmask over {c1 pos*k, c2 pos*prevcls, c3 pos*nsofar*kc, c4 exact prev rank, c5 pos(4/oct)*kc}
        const int NM = 5;
        const int configs[][2] = {{0b00001, 1}, {0b00011, 2}, {0b00111, 3}, {0b01111, 4}, {0b11111, 5}, {0b10111, 6}};
        const char* cname[] = {"c1 pos*k", "c1+c2 prevcls", "c1+c2+c3 nsofar", "+c4 exact prev rank", "+c5 fine pos", "no c4"};
        for (int ci = 0; ci < 6; ++ci) {
            int mask = configs[ci][0];
            const size_t NODES = 128;
            std::vector<BinModel> mdl;
            const size_t sizes[NM] = {static_cast<size_t>(40) * 12 * NODES, static_cast<size_t>(40) * 16 * NODES, static_cast<size_t>(40) * 12 * 4 * NODES,
                                      static_cast<size_t>(4200) * 4 * NODES, static_cast<size_t>(80) * 4 * NODES};
            int nin = 0;
            for (int i = 0; i < NM; ++i)
                if (mask >> i & 1) { mdl.emplace_back(sizes[i]); ++nin; }
            Mixer mx(nin + 1, static_cast<size_t>(NODES) * 10, 0.3f);
            double bits_cls = 0, bits_extra = 0;
            for (uint32_t j = 0; j < M.n; ++j) {
                uint32_t k = R.ptr[j + 1] - R.ptr[j];
                if (!k) continue;
                uint32_t kraw = static_cast<uint32_t>(ilog2u(k + 1));
                uint32_t kb = std::min(kraw, 11u), kc = std::min(kraw / 3, 3u);
                uint32_t pp = 0, prevcls = 0, ns = 0;
                for (uint32_t t = R.ptr[j]; t < R.ptr[j + 1]; ++t, ++ns) {
                    uint32_t g = R.rk[t] - pp, ex, nb;
                    uint32_t cls = g_cls(g, ex, nb);
                    // position buckets
                    uint32_t q = pp + 2, lg = 31u - __builtin_clz(q);
                    uint32_t pos = std::min<uint32_t>(2 * lg + ((q >> (lg - 1)) & 1) - 2, 39);
                    uint32_t q4 = pp + 4, lg4 = 31u - __builtin_clz(q4);
                    uint32_t pos4 = std::min<uint32_t>(4 * lg4 + ((q4 >> (lg4 - 2)) & 3) - 8, 79);
                    uint32_t nsb = std::min<uint32_t>(ilog2u(ns + 1), 11);
                    size_t base[NM];
                    base[0] = (static_cast<size_t>(pos) * 12 + kb) * NODES;
                    base[1] = (static_cast<size_t>(pos) * 16 + std::min<uint32_t>(prevcls / 4 + (ns ? 1 : 0) * 0, 15)) * NODES;
                    base[2] = ((static_cast<size_t>(pos) * 12 + nsb) * 4 + kc) * NODES;
                    base[3] = ((pp < 4096 ? static_cast<size_t>(pp) : static_cast<size_t>(4096 + std::min<uint32_t>(pos, 39))) * 4 + kc) * NODES;
                    base[4] = (static_cast<size_t>(pos4) * 4 + kc) * NODES;
                    // binary tree over 7 bits of cls
                    uint32_t node = 1;
                    for (int bit_i = 6; bit_i >= 0; --bit_i) {
                        int bit = (cls >> bit_i) & 1;
                        int mi = 0;
                        for (int i = 0; i < NM; ++i)
                            if (mask >> i & 1) {
                                float p = mdl[mi].get(base[i] + node);
                                mx.x[mi] = stretchf(p);
                                ++mi;
                            }
                        mx.x[nin] = 0.3f;
                        float pr = mx.mix(static_cast<size_t>(node) * 10 + std::min<uint32_t>(pos / 4, 9));
                        bits_cls -= std::log2(bit ? pr : 1.0f - pr);
                        mx.update(bit, 0.02f);
                        mi = 0;
                        for (int i = 0; i < NM; ++i)
                            if (mask >> i & 1) { mdl[mi].update(base[i] + node, bit, 1.0f / 40); ++mi; }
                        node = node * 2 + static_cast<uint32_t>(bit);
                    }
                    bits_extra += nb;
                    pp = R.rk[t] + 1;
                    prevcls = cls;
                }
            }
            std::printf("  gap cls CM [%-22s] %.3f + extra %.3f = %.3f bits/nnz\n", cname[ci], bits_cls / nnz, bits_extra / nnz, (bits_cls + bits_extra) / nnz);
        }
    }
}
