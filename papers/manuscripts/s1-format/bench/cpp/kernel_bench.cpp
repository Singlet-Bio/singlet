// SPDX-License-Identifier: MIT
// How fast can any lossless code decode a count matrix? One core, in memory, for the `kernels` stage (Fig. limits).
//   kernel_bench <matrix.bin> <member.pz2> <repeats> <warmup>
// matrix.bin: u64 rows, u64 cols, u64 nnz, then indptr (cols+1 x u64), row indices (nnz x u32), values (nnz x u32), the
// canonical genes x cells CSC of real data. Every code is encoded here, then timed decoding back into caller-owned u32
// row indices and u32 values (the output a CSC reader needs), checked against the input, and printed as one JSON line:
//   {"codec":..., "nnz":..., "bytes":..., "encode_s":..., "ok":true, "times":[...]}
// The codes are the textbook ones a format could adopt: raw arrays (memcpy), fixed-width bytes, LEB128 varints, Stream
// VByte (SIMD byte-aligned varints), vertical SIMD bit-packing in blocks of 256 (the layout of the fastest published
// integer codecs), an order-0 rANS entropy code, and zstd at levels 3 and 19 over LEB128 gap streams; the gap codes store row - previous row - 1 within
// a cell, the transform .s1 uses. Two references bound the result: `store_bound` only writes the output (no input can be
// decoded faster than the output is written), and `pz2` decodes the .s1 member itself on one thread (AVX2, canonical).
#include <singlet/pz2/pz2.h>
#include <zstd.h>

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace pz = singlet::pz2;
using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using Bytes = std::vector<u8>;

struct Csc {
    u64 rows = 0, cols = 0, nnz = 0;
    std::vector<u64> indptr;
    std::vector<u32> idx, val;
};

static Csc read_matrix(const char* p) {
    std::ifstream f(p, std::ios::binary);
    Csc m;
    f.read(reinterpret_cast<char*>(&m.rows), 8);
    f.read(reinterpret_cast<char*>(&m.cols), 8);
    f.read(reinterpret_cast<char*>(&m.nnz), 8);
    m.indptr.resize(m.cols + 1);
    m.idx.resize(m.nnz);
    m.val.resize(m.nnz);
    f.read(reinterpret_cast<char*>(m.indptr.data()), 8 * (m.cols + 1));
    f.read(reinterpret_cast<char*>(m.idx.data()), 4 * m.nnz);
    f.read(reinterpret_cast<char*>(m.val.data()), 4 * m.nnz);
    return m;
}

static Bytes read_file(const char* p) {
    std::ifstream f(p, std::ios::binary);
    return Bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ------------------------------------------------------------------ gaps
static std::vector<u32> to_gaps(const Csc& m) {
    std::vector<u32> g(m.nnz);
    for (u64 c = 0; c < m.cols; ++c) {
        u32 prev = 0xFFFFFFFFu;   // prev + 1 wraps to 0 at a cell's first entry
        for (u64 i = m.indptr[c]; i < m.indptr[c + 1]; ++i) { g[i] = m.idx[i] - prev - 1; prev = m.idx[i]; }
    }
    return g;
}

// in place: gaps -> row indices (the reconstruction every gap code pays)
static void gaps_to_rows(const Csc& m, u32* x) {
    for (u64 c = 0; c < m.cols; ++c) {
        u32 prev = 0xFFFFFFFFu;
        for (u64 i = m.indptr[c]; i < m.indptr[c + 1]; ++i) { prev += 1 + x[i]; x[i] = prev; }
    }
}

// ------------------------------------------------------------------ LEB128
static void leb_enc(const std::vector<u32>& v, Bytes& out) {
    for (u32 x : v) {
        while (x >= 0x80) { out.push_back(u8(x | 0x80)); x >>= 7; }
        out.push_back(u8(x));
    }
}

static const u8* leb_dec(const u8* p, u32* out, u64 n) {
    for (u64 i = 0; i < n; ++i) {
        u32 b = *p++, v = b & 0x7F;
        if (b & 0x80) {
            int s = 7;
            do { b = *p++; v |= (b & 0x7F) << s; s += 7; } while (b & 0x80);
        }
        out[i] = v;
    }
    return p;
}

// ------------------------------------------------------------------ Stream VByte (Lemire et al. 2018 layout)
static u8 svb_shuf[256][16];
static u8 svb_len[256];

static void svb_init() {
    for (int c = 0; c < 256; ++c) {
        int off = 0;
        for (int j = 0; j < 4; ++j) {
            const int l = ((c >> (2 * j)) & 3) + 1;
            for (int k = 0; k < 4; ++k) svb_shuf[c][4 * j + k] = k < l ? u8(off + k) : 0x80;
            off += l;
        }
        svb_len[c] = u8(off);
    }
}

// layout: n (u64), control bytes (one per 4 values), data bytes, 16 bytes of padding
static void svb_enc(const std::vector<u32>& v, Bytes& out) {
    const u64 n = v.size(), groups = (n + 3) / 4;
    const size_t base = out.size();
    out.resize(base + 8 + groups);
    std::memcpy(&out[base], &n, 8);
    for (u64 gi = 0; gi < groups; ++gi) {
        u8 ctrl = 0;
        for (int j = 0; j < 4; ++j) {
            const u32 x = 4 * gi + j < n ? v[4 * gi + j] : 0;
            const int l = x < (1u << 8) ? 1 : x < (1u << 16) ? 2 : x < (1u << 24) ? 3 : 4;
            ctrl |= u8((l - 1) << (2 * j));
            for (int k = 0; k < l; ++k) out.push_back(u8(x >> (8 * k)));
        }
        out[base + 8 + gi] = ctrl;
    }
    out.insert(out.end(), 16, 0);
}

static const u8* svb_dec(const u8* p, u32* out) {
    u64 n;
    std::memcpy(&n, p, 8);
    const u64 groups = (n + 3) / 4, full = n / 4;
    const u8* ctrl = p + 8;
    const u8* d = ctrl + groups;
    for (u64 gi = 0; gi < full; ++gi) {
        const u8 c = ctrl[gi];
        const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 4 * gi),
                         _mm_shuffle_epi8(x, _mm_loadu_si128(reinterpret_cast<const __m128i*>(svb_shuf[c]))));
        d += svb_len[c];
    }
    if (full < groups) {
        alignas(16) u32 t[4];
        const u8 c = ctrl[full];
        _mm_store_si128(reinterpret_cast<__m128i*>(t),
                        _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(d)),
                                         _mm_loadu_si128(reinterpret_cast<const __m128i*>(svb_shuf[c]))));
        for (u64 i = 4 * full; i < n; ++i) out[i] = t[i - 4 * full];
        d += svb_len[c];
    }
    return d + 16;
}

// ------------------------------------------------------------------ vertical SIMD bit-packing, blocks of 256
// Lane l holds values l, 8 + l, ..., 248 + l, packed at W bits each into W 32-bit words; word j of every lane is stored
// together, so one AVX2 load reads word j of all eight lanes.
// A block is one width byte followed by 32 W bytes; the width is the bit length of the block's largest value.
static int bitlen(u32 x) { return x ? 32 - __builtin_clz(x) : 0; }

static void bp_enc(const std::vector<u32>& v, Bytes& out) {
    const u64 n = v.size(), blocks = (n + 255) / 256;
    const size_t base = out.size();
    out.resize(base + 8);
    std::memcpy(&out[base], &n, 8);
    for (u64 b = 0; b < blocks; ++b) {
        u32 blk[256] = {0};
        u32 mx = 0;
        for (int i = 0; i < 256; ++i) {
            const u64 k = 256 * b + i;
            blk[i] = k < n ? v[k] : 0;
            mx |= blk[i];
        }
        const int W = bitlen(mx);
        out.push_back(u8(W));
        std::vector<u32> words(8 * W, 0);
        for (int l = 0; l < 8; ++l)
            for (int k = 0; k < 32; ++k) {
                const u64 x = blk[8 * k + l];
                const int bit = k * W, w = bit >> 5, s = bit & 31;
                if (!W) continue;
                words[8 * w + l] |= u32(x << s);
                if (s + W > 32) words[8 * (w + 1) + l] |= u32(x >> (32 - s));
            }
        const size_t at = out.size();
        out.resize(at + 4 * words.size());
        if (!words.empty()) std::memcpy(&out[at], words.data(), 4 * words.size());
    }
}

template <int W>
static void bp_unpack(const u8* in8, u32* out) {
    if constexpr (W == 0) {
        std::memset(out, 0, 1024);
    } else if constexpr (W == 32) {
        std::memcpy(out, in8, 1024);
    } else {
        const __m256i* in = reinterpret_cast<const __m256i*>(in8);
        const __m256i mask = _mm256_set1_epi32(int((1u << W) - 1));
#pragma GCC unroll 32
        for (int k = 0; k < 32; ++k) {
            const int bit = k * W, w = bit >> 5, s = bit & 31;
            __m256i x = _mm256_srli_epi32(_mm256_loadu_si256(in + w), s);
            if (s + W > 32) x = _mm256_or_si256(x, _mm256_slli_epi32(_mm256_loadu_si256(in + w + 1), 32 - s));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 8 * k), _mm256_and_si256(x, mask));
        }
    }
}

using Unpack = void (*)(const u8*, u32*);
template <int... Ws>
static constexpr std::array<Unpack, sizeof...(Ws)> unpack_table(std::integer_sequence<int, Ws...>) {
    return {&bp_unpack<Ws>...};
}
static const auto BP = unpack_table(std::make_integer_sequence<int, 33>{});

static const u8* bp_dec(const u8* p, u32* out) {
    u64 n;
    std::memcpy(&n, p, 8);
    p += 8;
    const u64 full = n / 256;
    for (u64 b = 0; b < full; ++b) {
        const int W = *p++;
        BP[W](p, out + 256 * b);
        p += 32 * W;
    }
    if (full * 256 < n) {
        alignas(32) u32 t[256];
        const int W = *p++;
        BP[W](p, t);
        p += 32 * W;
        std::memcpy(out + 256 * full, t, 4 * (n - 256 * full));
    }
    return p;
}

// ------------------------------------------------------------------ order-0 rANS (Duda 2013; byte renormalization as in ryg_rans)
// The plainest entropy code: one symbol per value, min(x, 255), with 255 escaping to an LEB128 remainder in a side stream;
// one static 12-bit frequency table per stream and four interleaved states. It is the same family of coder as .s1's
// without any of its context modelling, so it marks how fast table-driven entropy decoding runs on this core.
constexpr u32 R_SCALE = 12, R_TOT = 1u << R_SCALE, R_L = 1u << 23;
struct Rans0 {
    u32 freq[256], cum[257];
    u8 slot[R_TOT];
};

static void rans_table(const u32* freq16, Rans0& t) {
    t.cum[0] = 0;
    for (int s = 0; s < 256; ++s) { t.freq[s] = freq16[s]; t.cum[s + 1] = t.cum[s] + t.freq[s]; }
    for (int s = 0; s < 256; ++s)
        for (u32 k = t.cum[s]; k < t.cum[s + 1]; ++k) t.slot[k] = u8(s);
}

// layout: n, rANS bytes, escape bytes (u64 each), 256 frequencies (u16), rANS bytes, escape bytes, 16 bytes of padding
static void r0_enc(const std::vector<u32>& v, Bytes& out) {
    const u64 n = v.size();
    std::vector<u8> sym(n);
    Bytes esc;
    u64 cnt[256] = {0};
    for (u64 i = 0; i < n; ++i) {
        sym[i] = u8(std::min<u32>(v[i], 255));
        ++cnt[sym[i]];
        if (v[i] >= 255) leb_enc({v[i] - 255}, esc);
    }
    u32 f[256] = {0};
    int64_t sum = 0;
    for (int s = 0; s < 256; ++s) {
        f[s] = cnt[s] ? u32(std::max<u64>(1, cnt[s] * R_TOT / std::max<u64>(n, 1))) : 0;
        sum += f[s];
    }
    if (!n) { f[0] = R_TOT; sum = R_TOT; }
    while (sum != R_TOT) {   // give the difference to (or take it from) the most frequent symbol
        const int b = int(std::max_element(f, f + 256) - f);
        if (sum < R_TOT) { f[b] += u32(R_TOT - sum); sum = R_TOT; }
        else { const u32 d = u32(std::min<int64_t>(sum - R_TOT, f[b] - 1)); f[b] -= d; sum -= d; }
    }
    Rans0 t;
    rans_table(f, t);
    Bytes buf(2 * n + 64);
    u8* const end = buf.data() + buf.size();
    u8* p = end;
    u32 x[4] = {R_L, R_L, R_L, R_L};
    for (u64 i = n; i-- > 0;) {
        u32& xs = x[i & 3];
        const u32 s = sym[i], fs = t.freq[s];
        const u32 xmax = ((R_L >> R_SCALE) << 8) * fs;
        while (xs >= xmax) { *--p = u8(xs); xs >>= 8; }
        xs = ((xs / fs) << R_SCALE) + (xs % fs) + t.cum[s];
    }
    for (int j = 3; j >= 0; --j) {   // state 0 ends up first, where the decoder starts
        p -= 4;
        for (int k = 0; k < 4; ++k) p[k] = u8(x[j] >> (8 * k));
    }
    const u64 hdr[3] = {n, u64(end - p), esc.size()};
    const size_t at = out.size();
    out.resize(at + 24 + 512);
    std::memcpy(&out[at], hdr, 24);
    for (int s = 0; s < 256; ++s) { const uint16_t h = uint16_t(f[s]); std::memcpy(&out[at + 24 + 2 * s], &h, 2); }
    out.insert(out.end(), p, end);
    out.insert(out.end(), esc.begin(), esc.end());
    out.insert(out.end(), 16, 0);
}

static const u8* r0_dec(const u8* p, u32* out, Rans0& t) {
    u64 hdr[3];
    std::memcpy(hdr, p, 24);
    u32 f[256];
    for (int s = 0; s < 256; ++s) { uint16_t h; std::memcpy(&h, p + 24 + 2 * s, 2); f[s] = h; }
    rans_table(f, t);
    const u64 n = hdr[0];
    const u8* r = p + 24 + 512;
    const u8* e = r + hdr[1];
    u32 x[4];
    for (int j = 0; j < 4; ++j) { x[j] = u32(r[0]) | u32(r[1]) << 8 | u32(r[2]) << 16 | u32(r[3]) << 24; r += 4; }
    for (u64 i = 0; i < n; ++i) {
        u32& xs = x[i & 3];
        const u32 m = xs & (R_TOT - 1), s = t.slot[m];
        xs = t.freq[s] * (xs >> R_SCALE) + m - t.cum[s];
        while (xs < R_L) xs = (xs << 8) | *r++;
        u32 val = s;
        if (s == 255) { u32 rem; e = leb_dec(e, &rem, 1); val += rem; }
        out[i] = val;
    }
    return p + 24 + 512 + hdr[1] + hdr[2] + 16;
}

// ------------------------------------------------------------------ harness
struct Codec {
    std::string name;
    std::function<void(Bytes&)> encode;                 // fills the encoded bytes (timed once)
    std::function<void(const Bytes&, u32*, u32*)> decode;   // into row indices and values (timed)
};

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: kernel_bench matrix.bin member.pz2 repeats warmup\n"); return 2; }
    svb_init();
    const Csc m = read_matrix(argv[1]);
    const int reps = std::atoi(argv[3]), warm = std::atoi(argv[4]);
    const std::vector<u32> gaps = to_gaps(m);
    u32 vmax = 0;
    for (u32 v : m.val) vmax = std::max(vmax, v);
    const int wr = m.rows <= 256 ? 1 : m.rows <= 65536 ? 2 : 4, wv = vmax < 256 ? 1 : vmax < 65536 ? 2 : 4;
    std::vector<u32> oi(m.nnz), ov(m.nnz);
    std::vector<u8> scratch_g, scratch_v;   // zstd output, sized on first use
    static Rans0 r0t;                       // the decoder's table, rebuilt from each stream's frequencies

    auto zstd_codec = [&](int level) {
        return Codec{"zstd" + std::to_string(level) + "_leb_gaps",
            [&, level](Bytes& out) {
                Bytes g, v;
                leb_enc(gaps, g);
                leb_enc(m.val, v);
                ZSTD_CCtx* cc = ZSTD_createCCtx();
                ZSTD_CCtx_setParameter(cc, ZSTD_c_compressionLevel, level);
                ZSTD_CCtx_setParameter(cc, ZSTD_c_nbWorkers, 8);   // compression speed is not what is measured here
                u64 hdr[4] = {g.size(), v.size(), 0, 0};
                Bytes zg(ZSTD_compressBound(g.size())), zv(ZSTD_compressBound(v.size()));
                hdr[2] = ZSTD_compress2(cc, zg.data(), zg.size(), g.data(), g.size());
                hdr[3] = ZSTD_compress2(cc, zv.data(), zv.size(), v.data(), v.size());
                ZSTD_freeCCtx(cc);
                out.resize(32);
                std::memcpy(out.data(), hdr, 32);
                out.insert(out.end(), zg.begin(), zg.begin() + hdr[2]);
                out.insert(out.end(), zv.begin(), zv.begin() + hdr[3]);
            },
            [&](const Bytes& in, u32* ri, u32* vi) {
                u64 hdr[4];
                std::memcpy(hdr, in.data(), 32);
                scratch_g.resize(hdr[0]);
                scratch_v.resize(hdr[1]);
                ZSTD_decompress(scratch_g.data(), hdr[0], in.data() + 32, hdr[2]);
                ZSTD_decompress(scratch_v.data(), hdr[1], in.data() + 32 + hdr[2], hdr[3]);
                leb_dec(scratch_g.data(), ri, m.nnz);
                gaps_to_rows(m, ri);
                leb_dec(scratch_v.data(), vi, m.nnz);
            }};
    };

    std::vector<Codec> codecs = {
        {"store_bound", [&](Bytes&) {},
         [&](const Bytes&, u32* ri, u32* vi) { std::memset(ri, 1, 4 * m.nnz); std::memset(vi, 1, 4 * m.nnz); }},
        {"raw_u32", [&](Bytes& out) {
             out.resize(8 * m.nnz);
             std::memcpy(out.data(), m.idx.data(), 4 * m.nnz);
             std::memcpy(out.data() + 4 * m.nnz, m.val.data(), 4 * m.nnz);
         },
         [&](const Bytes& in, u32* ri, u32* vi) {
             std::memcpy(ri, in.data(), 4 * m.nnz);
             std::memcpy(vi, in.data() + 4 * m.nnz, 4 * m.nnz);
         }},
        {"fixed_bytes", [&](Bytes& out) {   // row ids and values at the narrowest whole-byte width that holds every value
             out.resize(u64(wr + wv) * m.nnz);
             for (u64 i = 0; i < m.nnz; ++i) std::memcpy(&out[wr * i], &m.idx[i], wr);
             u8* v = out.data() + wr * m.nnz;
             for (u64 i = 0; i < m.nnz; ++i) std::memcpy(v + wv * i, &m.val[i], wv);
         },
         [&](const Bytes& in, u32* ri, u32* vi) {
             const u8* p = in.data();
             auto widen = [&](const u8* s, u32* d, int w) {
                 if (w == 1) for (u64 i = 0; i < m.nnz; ++i) d[i] = s[i];
                 else if (w == 2) { const uint16_t* t = reinterpret_cast<const uint16_t*>(s); for (u64 i = 0; i < m.nnz; ++i) d[i] = t[i]; }
                 else std::memcpy(d, s, 4 * m.nnz);
             };
             widen(p, ri, wr);
             widen(p + wr * m.nnz, vi, wv);
         }},
        {"leb128_gaps", [&](Bytes& out) {
             u64 ng = 0;
             Bytes g;
             leb_enc(gaps, g);
             ng = g.size();
             out.resize(8);
             std::memcpy(out.data(), &ng, 8);
             out.insert(out.end(), g.begin(), g.end());
             leb_enc(m.val, out);
         },
         [&](const Bytes& in, u32* ri, u32* vi) {
             u64 ng;
             std::memcpy(&ng, in.data(), 8);
             leb_dec(in.data() + 8, ri, m.nnz);
             gaps_to_rows(m, ri);
             leb_dec(in.data() + 8 + ng, vi, m.nnz);
         }},
        {"streamvbyte_gaps", [&](Bytes& out) { svb_enc(gaps, out); svb_enc(m.val, out); },
         [&](const Bytes& in, u32* ri, u32* vi) {
             const u8* p = svb_dec(in.data(), ri);
             gaps_to_rows(m, ri);
             svb_dec(p, vi);
         }},
        {"rans0_gaps", [&](Bytes& out) { r0_enc(gaps, out); r0_enc(m.val, out); },
         [&](const Bytes& in, u32* ri, u32* vi) {
             const u8* p = r0_dec(in.data(), ri, r0t);
             gaps_to_rows(m, ri);
             r0_dec(p, vi, r0t);
         }},
        {"bitpack256_gaps", [&](Bytes& out) { bp_enc(gaps, out); bp_enc(m.val, out); out.insert(out.end(), 64, 0); },
         [&](const Bytes& in, u32* ri, u32* vi) {
             const u8* p = bp_dec(in.data(), ri);
             gaps_to_rows(m, ri);
             bp_dec(p, vi);
         }},
        zstd_codec(3),
        zstd_codec(19),
    };

    for (const Codec& c : codecs) {
        Bytes enc;
        double t0 = now();
        c.encode(enc);
        const double enc_s = now() - t0;
        std::string times;
        bool ok = true;
        for (int k = 0; k < warm + reps; ++k) {
            t0 = now();
            c.decode(enc, oi.data(), ov.data());
            const double s = now() - t0;
            if (k == 0 && c.name != "store_bound")
                ok = std::memcmp(oi.data(), m.idx.data(), 4 * m.nnz) == 0 && std::memcmp(ov.data(), m.val.data(), 4 * m.nnz) == 0;
            if (k >= warm) times += (times.empty() ? "" : ",") + std::to_string(s);
        }
        std::printf("{\"codec\":\"%s\",\"nnz\":%llu,\"bytes\":%llu,\"encode_s\":%.6f,\"ok\":%s,\"times\":[%s]}\n", c.name.c_str(),
                    (unsigned long long)m.nnz, (unsigned long long)enc.size(), enc_s, ok ? "true" : "false", times.c_str());
        std::fflush(stdout);
    }

    // the .s1 member itself, one thread: its own nnz, so its rate is per non-zero of the whole matrix
    const Bytes buf = read_file(argv[2]);
    pz::ByteSource src(buf);
    for (const char* isa : {"avx2"}) {
        pz::DecodeOptions opt;
        opt.threads = 1;
        opt.isa = std::string(isa) == "avx2" ? pz::Isa::Avx2 : pz::Isa::Scalar;
        opt.mode = pz::Mode::Canonical;
        pz::Matrix m0 = pz::Matrix::open(src);
        const u64 nnz = m0.header().nnz;
        std::vector<u64> indptr(m0.header().n + 1);
        std::vector<u32> indices(nnz), data(nnz);
        std::string times;
        for (int k = 0; k < warm + reps; ++k) {
            const double t0 = now();
            pz::Matrix mm = pz::Matrix::open(src);
            pz::Output out;
            out.indptr = indptr.data(); out.indptr_type = pz::IntType::U64; out.indptr_capacity = indptr.size();
            out.indices = indices.data(); out.indices_type = pz::IntType::U32; out.indices_capacity = indices.size();
            out.data = data.data(); out.data_type = pz::DType::U32; out.data_capacity = data.size();
            mm.decode(src, out, opt);
            const double s = now() - t0;
            if (k >= warm) times += (times.empty() ? "" : ",") + std::to_string(s);
        }
        std::printf("{\"codec\":\"pz2_%s\",\"nnz\":%llu,\"bytes\":%llu,\"encode_s\":0,\"ok\":true,\"isa_used\":\"%s\",\"times\":[%s]}\n",
                    isa, (unsigned long long)nnz, (unsigned long long)buf.size(), pz::isa_name(pz::resolve_isa(opt.isa)),
                    times.c_str());
        std::fflush(stdout);
    }
    return 0;
}
