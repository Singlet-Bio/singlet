// Track B: row-index coders for the index-coding survey (PLAN.md Track B item 1). Each coder encodes the sorted row ids
// of columns [c0,c1) of a chunk (universe [0,m)) into bytes and decodes them back given the column counts.
// Used by harness/track_b_index.cpp, which measures bits/nnz (raw, +zstd-3, +zstd-19) and decode Mnnz/s.
#pragma once
#include "track_b_util.h"
#include <functional>

namespace cs {
namespace tb {

struct IdxCoder {
    std::string name;
    // ptr: global indptr; idx: global row ids (ascending per column)
    std::function<void(const uint32_t* ptr, const uint32_t* idx, uint32_t c0, uint32_t c1, uint32_t m, std::vector<uint8_t>& out)> enc;
    std::function<void(const uint8_t* in, size_t len, const uint32_t* ptr, uint32_t c0, uint32_t c1, uint32_t m, uint32_t* idx_out)> dec;
};

// ---- bit-level helpers (LSB-first) ----------------------------------------------------------------------
struct BitR {
    const uint8_t* p;
    uint64_t pos = 0;
    explicit BitR(const uint8_t* base) : p(base) {}
    inline uint32_t get(int nb) {  // nb <= 32
        if (!nb) return 0;
        uint64_t w;
        std::memcpy(&w, p + (pos >> 3), 8);
        uint32_t v = static_cast<uint32_t>((w >> (pos & 7)) & ((1ull << nb) - 1));
        pos += static_cast<uint64_t>(nb);
        return v;
    }
};

// ---- 1. gap + LEB128 ---------------------------------------------------------------------------------------
inline IdxCoder coder_leb() {
    IdxCoder c;
    c.name = "gap_leb128";
    c.enc = [](const uint32_t* ptr, const uint32_t* idx, uint32_t c0, uint32_t c1, uint32_t, std::vector<uint8_t>& out) {
        for (uint32_t j = c0; j < c1; ++j) {
            int64_t prev = -1;
            for (uint32_t t = ptr[j]; t < ptr[j + 1]; ++t) { leb_put(out, static_cast<uint32_t>(idx[t] - prev - 1)); prev = idx[t]; }
        }
    };
    c.dec = [](const uint8_t* in, size_t, const uint32_t* ptr, uint32_t c0, uint32_t c1, uint32_t, uint32_t* o) {
        const uint8_t* p = in;
        for (uint32_t j = c0; j < c1; ++j) {
            uint32_t prev = 0;
            for (uint32_t t = ptr[j]; t < ptr[j + 1]; ++t) {
                uint32_t b = *p++, v = b & 127;
                if (b & 128) {
                    b = *p++; v |= (b & 127) << 7;
                    if (b & 128) { b = *p++; v |= (b & 127) << 14; if (b & 128) { b = *p++; v |= (b & 127) << 21; } }
                }
                prev += v;
                o[t] = prev;
                prev += 1;
            }
        }
    };
    return c;
}

// ---- 2. gap + byte-split planes (today's layout minus bit-plane/bitmap tricks) ------------------------------
inline IdxCoder coder_bytesplit() {
    IdxCoder c;
    c.name = "gap_bytesplit";
    c.enc = [](const uint32_t* ptr, const uint32_t* idx, uint32_t c0, uint32_t c1, uint32_t, std::vector<uint8_t>& out) {
        std::vector<uint32_t> g;
        uint32_t mx = 0;
        for (uint32_t j = c0; j < c1; ++j) {
            int64_t prev = -1;
            for (uint32_t t = ptr[j]; t < ptr[j + 1]; ++t) { uint32_t v = static_cast<uint32_t>(idx[t] - prev - 1); g.push_back(v); mx = std::max(mx, v); prev = idx[t]; }
        }
        uint8_t w = mx < 65536 ? 2 : 4;
        out.push_back(w);
        for (int b = 0; b < w; ++b)
            for (auto v : g) out.push_back(static_cast<uint8_t>(v >> (8 * b)));
    };
    c.dec = [](const uint8_t* in, size_t, const uint32_t* ptr, uint32_t c0, uint32_t c1, uint32_t, uint32_t* o) {
        uint8_t w = in[0];
        const uint8_t* pl = in + 1;
        size_t n = ptr[c1] - ptr[c0];
        size_t i = 0;
        for (uint32_t j = c0; j < c1; ++j) {
            uint32_t prev = 0;
            for (uint32_t t = ptr[j]; t < ptr[j + 1]; ++t, ++i) {
                uint32_t v = pl[i] | (static_cast<uint32_t>(pl[n + i]) << 8);
                if (w == 4) v |= (static_cast<uint32_t>(pl[2 * n + i]) << 16) | (static_cast<uint32_t>(pl[3 * n + i]) << 24);
                prev += v;
                o[t] = prev;
                prev += 1;
            }
        }
    };
    return c;
}

// ---- 3. Elias-Fano per column ---------------------------------------------------------------------------------
inline IdxCoder coder_ef() {
    IdxCoder c;
    c.name = "elias_fano";
    c.enc = [](const uint32_t* ptr, const uint32_t* idx, uint32_t c0, uint32_t c1, uint32_t m, std::vector<uint8_t>& out) {
        BitW bw;
        for (uint32_t j = c0; j < c1; ++j) {
            uint32_t k = ptr[j + 1] - ptr[j];
            if (!k) continue;
            int l = (m / k > 1) ? ilog2u(m / k) : 0;
            for (uint32_t t = ptr[j]; t < ptr[j + 1]; ++t) bw.put(idx[t] & ((1u << l) - 1), l);
            // upper: unary: for i-th element, (high_i - high_{i-1}) zeros then a one
            uint32_t prevh = 0;
            for (uint32_t t = ptr[j]; t < ptr[j + 1]; ++t) {
                uint32_t h = idx[t] >> l;
                uint32_t z = h - prevh;
                while (z >= 32) { bw.put(0, 32); z -= 32; }
                bw.put(1u << z, static_cast<int>(z) + 1);
                prevh = h;
            }
        }
        bw.finish();
        out.insert(out.end(), bw.buf.begin(), bw.buf.end());
        for (int i = 0; i < 8; ++i) out.push_back(0);
    };
    c.dec = [](const uint8_t* in, size_t, const uint32_t* ptr, uint32_t c0, uint32_t c1, uint32_t m, uint32_t* o) {
        BitR br(in);
        for (uint32_t j = c0; j < c1; ++j) {
            uint32_t k = ptr[j + 1] - ptr[j];
            if (!k) continue;
            int l = (m / k > 1) ? ilog2u(m / k) : 0;
            uint64_t lowpos = br.pos;
            br.pos += static_cast<uint64_t>(k) * static_cast<uint64_t>(l);
            // read unary part
            BitR lo(in);
            lo.pos = lowpos;
            uint32_t h = 0;
            for (uint32_t t = 0; t < k; ++t) {
                // advance to next one bit
                while (true) {
                    uint64_t w;
                    std::memcpy(&w, in + (br.pos >> 3), 8);
                    w >>= (br.pos & 7);
                    if (w) {
                        int z = __builtin_ctzll(w);
                        h += static_cast<uint32_t>(z);
                        br.pos += static_cast<uint64_t>(z) + 1;
                        break;
                    }
                    uint64_t adv = 64 - (br.pos & 7);
                    h += static_cast<uint32_t>(adv);
                    br.pos += adv;
                }
                o[ptr[j] + t] = (h << l) | lo.get(l);
            }
        }
    };
    return c;
}

// ---- 4. binary interpolative coding ------------------------------------------------------------------------------
inline void tb_put(BitW& bw, uint32_t v, uint32_t range) {  // truncated binary of v in [0,range)
    if (range <= 1) return;
    int b = ilog2u(range);
    uint32_t u = (1u << (b + 1)) - range;
    if ((1u << b) == range) { bw.put(v, b); return; }
    if (v < u) bw.put(v, b);
    else { bw.put((v + u) >> 1, b); bw.put((v + u) & 1, 1); }  // MSB-first prefix code: high b bits, then low bit
}
inline uint32_t tb_get(BitR& br, uint32_t range) {
    if (range <= 1) return 0;
    int b = ilog2u(range);
    uint32_t u = (1u << (b + 1)) - range;
    if ((1u << b) == range) return br.get(b);
    uint32_t v = br.get(b);
    if (v < u) return v;
    v = (v << 1) | br.get(1);
    return v - u;
}
inline void bic_enc(BitW& bw, const uint32_t* a, int l, int r, int64_t lo, int64_t hi) {
    if (r < l) return;
    int mid = (l + r) / 2;
    int64_t lo2 = lo + (mid - l), hi2 = hi - (r - mid);
    tb_put(bw, static_cast<uint32_t>(a[mid] - lo2), static_cast<uint32_t>(hi2 - lo2 + 1));
    bic_enc(bw, a, l, mid - 1, lo, static_cast<int64_t>(a[mid]) - 1);
    bic_enc(bw, a, mid + 1, r, static_cast<int64_t>(a[mid]) + 1, hi);
}
inline void bic_dec(BitR& br, uint32_t* a, int l, int r, int64_t lo, int64_t hi) {
    if (r < l) return;
    int mid = (l + r) / 2;
    int64_t lo2 = lo + (mid - l), hi2 = hi - (r - mid);
    uint32_t v = tb_get(br, static_cast<uint32_t>(hi2 - lo2 + 1));
    a[mid] = static_cast<uint32_t>(lo2 + v);
    bic_dec(br, a, l, mid - 1, lo, static_cast<int64_t>(a[mid]) - 1);
    bic_dec(br, a, mid + 1, r, static_cast<int64_t>(a[mid]) + 1, hi);
}
inline IdxCoder coder_bic() {
    IdxCoder c;
    c.name = "interpolative";
    c.enc = [](const uint32_t* ptr, const uint32_t* idx, uint32_t c0, uint32_t c1, uint32_t m, std::vector<uint8_t>& out) {
        BitW bw;
        for (uint32_t j = c0; j < c1; ++j) {
            uint32_t k = ptr[j + 1] - ptr[j];
            if (k) bic_enc(bw, idx + ptr[j], 0, static_cast<int>(k) - 1, 0, static_cast<int64_t>(m) - 1);
        }
        bw.finish();
        out.insert(out.end(), bw.buf.begin(), bw.buf.end());
        for (int i = 0; i < 8; ++i) out.push_back(0);
    };
    c.dec = [](const uint8_t* in, size_t, const uint32_t* ptr, uint32_t c0, uint32_t c1, uint32_t m, uint32_t* o) {
        BitR br(in);
        for (uint32_t j = c0; j < c1; ++j) {
            uint32_t k = ptr[j + 1] - ptr[j];
            if (k) bic_dec(br, o + ptr[j], 0, static_cast<int>(k) - 1, 0, static_cast<int64_t>(m) - 1);
        }
    };
    return c;
}

// ---- 5. Roaring-style containers per 2^16 block ---------------------------------------------------------------------
// per column: [u16 nblocks (LEB)] then per block: key u16, type u8 (0 array, 1 bitmap, 2 runs), card-1 u16, payload
inline IdxCoder coder_roaring() {
    IdxCoder c;
    c.name = "roaring";
    c.enc = [](const uint32_t* ptr, const uint32_t* idx, uint32_t c0, uint32_t c1, uint32_t, std::vector<uint8_t>& out) {
        auto p16 = [&](uint32_t v) { out.push_back(static_cast<uint8_t>(v)); out.push_back(static_cast<uint8_t>(v >> 8)); };
        for (uint32_t j = c0; j < c1; ++j) {
            const uint32_t* a = idx + ptr[j];
            uint32_t k = ptr[j + 1] - ptr[j];
            std::vector<uint8_t> body;
            uint32_t nblk = 0, i = 0;
            while (i < k) {
                uint32_t blk = a[i] >> 16, e = i, runs = 0;
                uint32_t prev = 0;
                bool first = true;
                while (e < k && (a[e] >> 16) == blk) { if (first || a[e] != prev + 1) ++runs; first = false; prev = a[e]; ++e; }
                uint32_t card = e - i;
                size_t arr = 2ull * card, bm = 8192, run = 4ull * runs;
                auto b16 = [&](uint32_t v) { body.push_back(static_cast<uint8_t>(v)); body.push_back(static_cast<uint8_t>(v >> 8)); };
                b16(blk);
                if (run < arr && run < bm) {
                    body.push_back(2); b16(card - 1); b16(runs);
                    uint32_t s = a[i] & 0xFFFF, len = 0;
                    uint32_t pv = s;
                    for (uint32_t q = i; q < e; ++q) {
                        uint32_t v = a[q] & 0xFFFF;
                        if (q > i && v != pv + 1) { b16(s); b16(len); s = v; len = 0; }
                        else if (q > i) ++len;
                        pv = v;
                    }
                    b16(s); b16(len);
                } else if (arr <= bm) {
                    body.push_back(0); b16(card - 1);
                    for (uint32_t q = i; q < e; ++q) b16(a[q] & 0xFFFF);
                } else {
                    body.push_back(1); b16(card - 1);
                    size_t base = body.size();
                    body.resize(base + 8192, 0);
                    for (uint32_t q = i; q < e; ++q) { uint32_t v = a[q] & 0xFFFF; body[base + (v >> 3)] |= static_cast<uint8_t>(1u << (v & 7)); }
                }
                ++nblk;
                i = e;
            }
            leb_put(out, nblk);
            out.insert(out.end(), body.begin(), body.end());
        }
        (void)p16;
    };
    c.dec = [](const uint8_t* in, size_t, const uint32_t* ptr, uint32_t c0, uint32_t c1, uint32_t, uint32_t* o) {
        const uint8_t* p = in;
        auto g16 = [&]() { uint32_t v = p[0] | (static_cast<uint32_t>(p[1]) << 8); p += 2; return v; };
        for (uint32_t j = c0; j < c1; ++j) {
            uint32_t nblk = static_cast<uint32_t>(leb_get(p));
            uint32_t* w = o + ptr[j];
            for (uint32_t b = 0; b < nblk; ++b) {
                uint32_t blk = g16() << 16;
                uint8_t type = *p++;
                uint32_t card = g16() + 1;
                if (type == 0) {
                    for (uint32_t q = 0; q < card; ++q) *w++ = blk | g16();
                } else if (type == 1) {
                    for (uint32_t wi = 0; wi < 1024; ++wi) {
                        uint64_t x;
                        std::memcpy(&x, p + 8 * wi, 8);
                        while (x) { *w++ = blk | (wi * 64 + static_cast<uint32_t>(__builtin_ctzll(x))); x &= x - 1; }
                    }
                    p += 8192;
                } else {
                    uint32_t runs = g16();
                    for (uint32_t r = 0; r < runs; ++r) {
                        uint32_t s = g16(), len = g16();
                        for (uint32_t q = 0; q <= len; ++q) *w++ = blk | (s + q);
                    }
                }
            }
        }
    };
    return c;
}

// ---- 6. head bitmap (first H ids as a bitmap when dense enough) + gap/LEB tail ----------------------------------------
inline IdxCoder coder_headbm(uint32_t H) {
    IdxCoder c;
    c.name = "headbm" + std::to_string(H) + "_leb";
    c.enc = [H](const uint32_t* ptr, const uint32_t* idx, uint32_t c0, uint32_t c1, uint32_t, std::vector<uint8_t>& out) {
        for (uint32_t j = c0; j < c1; ++j) {
            const uint32_t* a = idx + ptr[j];
            uint32_t k = ptr[j + 1] - ptr[j];
            uint32_t nh = 0;
            while (nh < k && a[nh] < H) ++nh;
            // bitmap if cheaper than LEB-ish estimate (approx 1.3 bytes per element)
            bool bm = nh * 1.3 > H / 8.0;
            out.push_back(bm ? 1 : 0);
            if (bm) {
                size_t base = out.size();
                out.resize(base + H / 8, 0);
                for (uint32_t t = 0; t < nh; ++t) out[base + (a[t] >> 3)] |= static_cast<uint8_t>(1u << (a[t] & 7));
                int64_t prev = static_cast<int64_t>(H) - 1;
                for (uint32_t t = nh; t < k; ++t) { leb_put(out, static_cast<uint32_t>(a[t] - prev - 1)); prev = a[t]; }
            } else {
                int64_t prev = -1;
                for (uint32_t t = 0; t < k; ++t) { leb_put(out, static_cast<uint32_t>(a[t] - prev - 1)); prev = a[t]; }
            }
        }
    };
    c.dec = [H](const uint8_t* in, size_t, const uint32_t* ptr, uint32_t c0, uint32_t c1, uint32_t, uint32_t* o) {
        const uint8_t* p = in;
        for (uint32_t j = c0; j < c1; ++j) {
            uint32_t k = ptr[j + 1] - ptr[j];
            uint32_t* w = o + ptr[j];
            uint32_t done = 0;
            uint32_t prev = 0;
            if (*p++) {
                for (uint32_t wi = 0; wi < H / 64; ++wi) {
                    uint64_t x;
                    std::memcpy(&x, p + 8 * wi, 8);
                    while (x) { *w++ = wi * 64 + static_cast<uint32_t>(__builtin_ctzll(x)); x &= x - 1; ++done; }
                }
                p += H / 8;
                prev = H;
            }
            for (; done < k; ++done) {
                uint32_t b = *p++, v = b & 127;
                if (b & 128) { b = *p++; v |= (b & 127) << 7; if (b & 128) { b = *p++; v |= (b & 127) << 14; if (b & 128) { b = *p++; v |= (b & 127) << 21; } } }
                prev += v;
                *w++ = prev;
                prev += 1;
            }
        }
    };
    return c;
}

}  // namespace tb
}  // namespace cs
