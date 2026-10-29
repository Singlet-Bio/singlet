// Track D codecs. Owned by the Track D agent; see PLAN.md section 4.
// Every codec must satisfy decode(encode(M)) == M for all CanonicalMatrix fields.
//
//  cm_compress / cm_decompress : compact context-mixing byte coder (the "strong universal second pass").
//      bitwise prediction; inputs = order-0..4 context slots (22-bit probability + 10-bit adaptive-rate
//      count, hashed), plus a match model; one gated logistic mixer (weight set chosen by the partial byte),
//      two APM/SSE stages; 32-bit carry-less binary arithmetic coder with 12-bit probabilities.
//      All arithmetic is integer, so encoder and decoder agree bit for bit on any machine.
//  d_cm_bytes  : ref_zstd_csc_L3 output run through the CM  (second pass over an already compressed stream)
//  d_cm_raw    : raw little-endian CSC arrays + NUL-joined names run through the CM (no transform, no LZ)
//  d_cm_1pz    : the production .1pz bytes (VOCSC + zstd-3) run through the CM
#pragma once
#include "common.h"
#include "ref_zstd_csc.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace cs {
namespace dcm {

// ---------------------------------------------------------------- squash / stretch (lpaq style, integer)
inline int squash(int d) {
    static const int t[33] = {1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047,
                              2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079, 4085,
                              4089, 4092, 4093, 4094};
    if (d > 2047) return 4095;
    if (d < -2047) return 0;
    int w = d & 127;
    d = (d >> 7) + 16;
    return (t[d] * (128 - w) + t[d + 1] * w + 64) >> 7;
}
struct StretchTable {
    short t[4096];
    StretchTable() {
        int pi = 0;
        for (int x = -2047; x <= 2047; ++x) {
            int v = squash(x);
            for (int i = pi; i <= v; ++i) t[i] = (short)x;
            pi = v + 1;
        }
        for (int i = pi; i < 4096; ++i) t[i] = 2047;
    }
};
inline int stretch(int p) { static const StretchTable T; return T.t[p]; }

// Adaptive-probability slot: high 22 bits = P(1), low 10 bits = hit count (rate ~ 1/(n+1.5)).
struct Slot {
    static inline int p12(uint32_t s) { return (int)(s >> 20); }
    static inline void update(uint32_t& s, int y, int limit) {
        int n = (int)(s & 1023);
        int64_t p = s >> 10;
        int64_t err = ((int64_t)y << 22) - p;
        int64_t dt = 16384 / (n + n + 3);
        int64_t np = p + (((err >> 3) * dt) >> 10);   // = err / (n+1.5)
        if (np < 0) np = 0;
        if (np > (1 << 22) - 1) np = (1 << 22) - 1;
        s = (uint32_t)(np << 10) | (uint32_t)(n < limit ? n + 1 : limit);
    }
};

struct APM {
    std::vector<uint16_t> t;
    int idx = 0;
    explicit APM(size_t n) : t(n * 33) {
        for (size_t i = 0; i < n; ++i)
            for (int j = 0; j < 33; ++j) t[i * 33 + j] = (uint16_t)(squash((j - 16) * 128) * 16);
    }
    int pp(int pr, int cx) {
        int s = stretch(pr) + 2048;
        int lo = s >> 7, w = s & 127;
        idx = lo + cx * 33;
        int r = (t[idx] * (128 - w) + t[idx + 1] * w) >> 11;
        if (w >= 64) ++idx;      // update the nearer node
        return r;
    }
    void update(int y, int rate = 7) {
        int g = (y << 16) + (y << rate) - y - y;
        t[idx] += (g - t[idx]) >> rate;
    }
};

// The model. predict() returns P(bit==1) in 12 bits; update(bit) trains.
class Model {
public:
    static constexpr int NI = 7;             // 5 contexts + match + bias
    explicit Model(int hash_bits = 22)
        : hb_(hash_bits), tab_(5), apm1_(256), apm2_(65536), mix_w_((size_t)256 * 8 * NI, 0), ht_(1 << 20, 0) {
        tab_[0].assign(1 << 16, 0x80000000u);                                   // order-0 uses (c0) only; padded
        tab_[1].assign(1 << 16, 0x80000000u);                                   // order-1 direct: c1<<8|c0
        for (int i = 2; i < 5; ++i) tab_[i].assign((size_t)1 << hash_bits, 0x80000000u);
        mm_.assign(64 * 2, 0x80000000u);
        for (auto& w : mix_w_) w = 1 << 14;
        buf_.reserve(1 << 20);
        c0_ = 1;
        for (auto& x : h_) x = 0;
        select_slots();
    }

    int predict() {
        int ms = 0;
        // match model input
        int mi = 0;
        if (mlen_ > 0) {
            int pb = (buf_[mptr_] | 256) >> (7 - bitpos_);
            if ((pb >> 1) == c0_) {                       // predicted byte agrees with the bits seen so far
                int b = pb & 1;
                mctx_ = (std::min(mlen_, 31)) * 2 + b;
                mi = stretch(Slot::p12(mm_[mctx_]));
                ms = 1;
            }
        }
        if (!ms) mctx_ = -1;
        in_[0] = stretch(Slot::p12(*sl_[0]));
        in_[1] = stretch(Slot::p12(*sl_[1]));
        in_[2] = stretch(Slot::p12(*sl_[2]));
        in_[3] = stretch(Slot::p12(*sl_[3]));
        in_[4] = stretch(Slot::p12(*sl_[4]));
        in_[5] = mi;
        in_[6] = 256;
        int sel = c0_ * 8 + (ms ? (mlen_ >= 16 ? 2 : 1) : 0);
        w_ = &mix_w_[(size_t)sel * NI];
        int64_t dot = 0;
        for (int i = 0; i < NI; ++i) dot += (int64_t)in_[i] * w_[i];
        int d = (int)(dot >> 16);
        d = d > 2047 ? 2047 : (d < -2047 ? -2047 : d);
        pm_ = squash(d);
        int a1 = apm1_.pp(pm_, c0_);
        int a2 = apm2_.pp(pm_, (c0_ | (c1_ << 8)) & 0xffff);
        int p = (2 * pm_ + a1 + a2 + 2) >> 2;
        if (p < 1) p = 1;
        if (p > 4095) p = 4095;
        return p;
    }

    void update(int y) {
        // mixer
        int err = ((y << 12) - pm_) * 10;
        for (int i = 0; i < NI; ++i) w_[i] += (in_[i] * err + 0x2000) >> 14;
        apm1_.update(y);
        apm2_.update(y);
        Slot::update(*sl_[0], y, 255);
        Slot::update(*sl_[1], y, 255);
        Slot::update(*sl_[2], y, 255);
        Slot::update(*sl_[3], y, 127);
        Slot::update(*sl_[4], y, 127);
        if (mctx_ >= 0) Slot::update(mm_[mctx_], y, 255);
        // advance bit state
        c0_ = (c0_ << 1) | y;
        ++bitpos_;
        if (bitpos_ == 8) {
            uint8_t c = (uint8_t)(c0_ & 255);
            byte_update(c);
            c0_ = 1;
            bitpos_ = 0;
        }
        select_slots();
    }

private:
    void byte_update(uint8_t c) {
        buf_.push_back(c);
        c4_ = (c4_ << 8) | c;
        c1_ = c;
        // match model
        if (mlen_ > 0 && buf_[mptr_] == c) { ++mlen_; ++mptr_; } else mlen_ = 0;
        size_t n = buf_.size();
        if (n >= 5) {
            uint32_t h = (uint32_t)(((c4_ * 0x9E3779B1u) ^ ((uint32_t)buf_[n - 5] * 0x85EBCA6Bu)) >> 12) & ((1 << 20) - 1);
            if (mlen_ == 0) {
                uint32_t p = ht_[h];
                if (p > 0) {
                    size_t l = 0;
                    while (l < 32 && p > l && buf_[p - 1 - l] == buf_[n - 1 - l]) ++l;
                    if (l >= 5) { mlen_ = (int)l; mptr_ = p; }
                }
            }
            ht_[h] = (uint32_t)n;
        }
        if (mlen_ > 0 && mptr_ >= buf_.size()) mlen_ = 0;
        // context hashes for the next byte
        h_[2] = (uint32_t)((c4_ & 0xffff) * 0x9E3779B1u + 0x1234567u);
        h_[3] = (uint32_t)((c4_ & 0xffffff) * 0x85EBCA6Bu + 0x7654321u);
        h_[4] = (uint32_t)(c4_ * 0xC2B2AE35u + 0x2468ACEu);
    }
    void select_slots() {
        sl_[0] = &tab_[0][c0_];
        sl_[1] = &tab_[1][(c1_ << 8) | c0_];
        for (int i = 2; i < 5; ++i) {
            uint32_t ix = (h_[i] + (uint32_t)c0_ * 0x2545F491u) * 0x9E3779B1u;
            sl_[i] = &tab_[i][ix >> (32 - hb_)];
        }
    }

    int hb_;
    std::vector<std::vector<uint32_t>> tab_;
    APM apm1_, apm2_;
    std::vector<int> mix_w_;
    std::vector<uint32_t> ht_, mm_;
    std::vector<uint8_t> buf_;
    uint32_t* sl_[5] = {};
    uint32_t h_[5];
    int in_[NI] = {};
    int* w_ = nullptr;
    int pm_ = 2048, mctx_ = -1, mlen_ = 0;
    size_t mptr_ = 0;
    uint32_t c4_ = 0;
    int c0_ = 1, c1_ = 0, bitpos_ = 0;
};

// ---------------------------------------------------------------- arithmetic coder
inline std::vector<uint8_t> compress(const uint8_t* data, size_t n, int hash_bits = 22) {
    std::vector<uint8_t> out;
    for (int i = 0; i < 8; ++i) out.push_back((uint8_t)((uint64_t)n >> (8 * i)));
    auto model = std::make_unique<Model>(hash_bits);
    uint32_t x1 = 0, x2 = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) {
        int c = data[i];
        for (int b = 7; b >= 0; --b) {
            int y = (c >> b) & 1;
            int p = model->predict();
            uint32_t xmid = x1 + (uint32_t)(((uint64_t)(x2 - x1) * (uint32_t)p) >> 12);
            if (y) x2 = xmid; else x1 = xmid + 1;
            model->update(y);
            while (((x1 ^ x2) & 0xff000000u) == 0) {
                out.push_back((uint8_t)(x2 >> 24));
                x1 <<= 8;
                x2 = (x2 << 8) | 255;
            }
        }
    }
    for (int i = 0; i < 4; ++i) { out.push_back((uint8_t)(x1 >> 24)); x1 <<= 8; }
    return out;
}

inline std::vector<uint8_t> decompress(const uint8_t* in, size_t in_n, int hash_bits = 22) {
    uint64_t n = 0;
    for (int i = 0; i < 8; ++i) n |= (uint64_t)in[i] << (8 * i);
    std::vector<uint8_t> out;
    out.reserve(n);
    auto model = std::make_unique<Model>(hash_bits);
    size_t pos = 8;
    uint32_t x1 = 0, x2 = 0xffffffffu, x = 0;
    auto next = [&]() -> uint32_t { return pos < in_n ? in[pos++] : 0u; };
    for (int i = 0; i < 4; ++i) x = (x << 8) | next();
    for (uint64_t i = 0; i < n; ++i) {
        int c = 0;
        for (int b = 7; b >= 0; --b) {
            int p = model->predict();
            uint32_t xmid = x1 + (uint32_t)(((uint64_t)(x2 - x1) * (uint32_t)p) >> 12);
            int y = x <= xmid;
            if (y) x2 = xmid; else x1 = xmid + 1;
            model->update(y);
            c = (c << 1) | y;
            while (((x1 ^ x2) & 0xff000000u) == 0) {
                x1 <<= 8;
                x2 = (x2 << 8) | 255;
                x = (x << 8) | next();
            }
        }
        out.push_back((uint8_t)c);
    }
    return out;
}

}  // namespace dcm

// ---------------------------------------------------------------- codecs
inline std::vector<uint8_t> d_raw_serialize(const CanonicalMatrix& M) {
    using namespace detail;
    std::vector<uint8_t> o;
    put_u32(o, M.m); put_u32(o, M.n); put_u64(o, M.nnz); o.push_back(M.vt_code);
    uint32_t mx = 0;
    for (auto v : M.data) mx = std::max(mx, v);
    uint8_t w = (mx <= 65535) ? 2 : 4;
    o.push_back(w);
    for (auto v : M.indptr) put_u32(o, v);
    for (auto v : M.indices) put_u32(o, v);
    for (auto v : M.data) { put_u32(o, v); if (w == 2) { o.pop_back(); o.pop_back(); } }
    auto rn = join_names(M.rownames), cn = join_names(M.colnames);
    put_u64(o, rn.size()); o.insert(o.end(), rn.begin(), rn.end());
    put_u64(o, cn.size()); o.insert(o.end(), cn.begin(), cn.end());
    std::vector<uint8_t> kv;
    for (auto& [k, v] : M.user_kv) {
        kv.insert(kv.end(), k.begin(), k.end()); kv.push_back(0);
        kv.insert(kv.end(), v.begin(), v.end()); kv.push_back(0);
    }
    put_u32(o, (uint32_t)M.user_kv.size());
    put_u64(o, kv.size()); o.insert(o.end(), kv.begin(), kv.end());
    return o;
}

inline CanonicalMatrix d_raw_deserialize(const std::vector<uint8_t>& b) {
    using namespace detail;
    const uint8_t* p = b.data();
    CanonicalMatrix M;
    M.m = get_u32(p); M.n = get_u32(p); M.nnz = get_u64(p); M.vt_code = *p++;
    uint8_t w = *p++;
    M.indptr.resize((size_t)M.n + 1);
    for (auto& v : M.indptr) v = get_u32(p);
    M.indices.resize(M.nnz);
    for (auto& v : M.indices) v = get_u32(p);
    M.data.resize(M.nnz);
    for (auto& v : M.data) {
        if (w == 2) { uint16_t t; std::memcpy(&t, p, 2); p += 2; v = t; } else v = get_u32(p);
    }
    auto rd = [&](size_t count) {
        uint64_t n = get_u64(p);
        std::vector<uint8_t> v(p, p + n);
        p += n;
        return split_names(v, count);
    };
    M.rownames = rd(M.m);
    M.colnames = rd(M.n);
    uint32_t nkv = get_u32(p);
    uint64_t kn = get_u64(p);
    std::vector<uint8_t> kv(p, p + kn);
    auto parts = split_names(kv, 2 * (size_t)nkv);
    for (uint32_t i = 0; i < nkv; ++i) M.user_kv[parts[2 * i]] = parts[2 * i + 1];
    return M;
}

inline Codec make_d_cm_bytes() {
    Codec c;
    c.name = "d_cm_bytes";
    auto inner = make_ref_zstd_csc(3);
    c.encode = [inner](const CanonicalMatrix& M) {
        auto e = inner.encode(M);
        return dcm::compress(e.data(), e.size());
    };
    c.decode = [inner](const std::vector<uint8_t>& b) {
        auto e = dcm::decompress(b.data(), b.size());
        return inner.decode(e);
    };
    return c;
}

inline Codec make_d_cm_raw() {
    Codec c;
    c.name = "d_cm_raw";
    c.encode = [](const CanonicalMatrix& M) {
        auto e = d_raw_serialize(M);
        return dcm::compress(e.data(), e.size(), 23);
    };
    c.decode = [](const std::vector<uint8_t>& b) { return d_raw_deserialize(dcm::decompress(b.data(), b.size(), 23)); };
    return c;
}

inline Codec make_d_cm_1pz() {
    Codec c;
    c.name = "d_cm_1pz";
    auto inner = make_ref_identity_1pz();
    c.encode = [inner](const CanonicalMatrix& M) {
        auto e = rewrite_1pz(M, "results/.track_d_cm1pz.tmp.1pz");
        std::remove("results/.track_d_cm1pz.tmp.1pz");
        return dcm::compress(e.data(), e.size());
    };
    c.decode = [inner](const std::vector<uint8_t>& b) {
        auto e = dcm::decompress(b.data(), b.size());
        return inner.decode(e);
    };
    return c;
}

inline void register_track_d(std::vector<Codec>& v) {
    v.push_back(make_d_cm_bytes());
    v.push_back(make_d_cm_raw());
    v.push_back(make_d_cm_1pz());
}
}  // namespace cs
