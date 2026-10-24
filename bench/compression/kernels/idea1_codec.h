// SPDX-License-Identifier: MIT
// Idea 1 codec: stream format, encoder, reference decoder.  See idea1.h for the scheme.
#pragma once
#include "idea1.h"
#include <cstdlib>

namespace cs {
namespace i1 {

struct Hdr {
    uint32_t m, n, nb;
    uint64_t nnz, nwords;
    uint8_t vt;
    std::shared_ptr<Model> model;
    std::vector<uint16_t> bf;       // bucket frequencies (sum 2^BB), nb > 1 only
    const uint32_t* words = nullptr;
    const uint8_t* names = nullptr; // start of the zstd name streams
};

inline std::string model_path(const std::string& var, uint32_t m, const std::string& hold) {
    bool ok;
    std::string om = org_member(m, ok);
    if (!ok) throw std::runtime_error("idea1: no model for m=" + std::to_string(m));
    return ctx().dir + "/" + om + "_" + var + "_" + hold + ".bin";
}

// -------------------------------------------------------------------------------------------------
inline std::vector<uint8_t> encode_matrix(const CanonicalMatrix& M, const std::string& var, const std::string& hold) {
    using namespace detail;
    const Tabs& T = tabs();
    auto md = load_model(model_path(var, M.m, hold));
    if (md->m != M.m) throw std::runtime_error("idea1: model/matrix row mismatch");
    if (md->rnhash != names_hash(M.rownames)) throw std::runtime_error("idea1: rownames differ from model vocabulary");
    const uint32_t m = M.m, n = M.n, nb = md->nb;

    // per-column depth bucket
    std::vector<uint8_t> bk(n, 0);
    std::vector<uint32_t> hist(nb, 0);
    for (uint32_t c = 0; c < n; ++c) {
        uint64_t D = 0;
        for (uint32_t k = M.indptr[c]; k < M.indptr[c + 1]; ++k) D += M.data[k];
        bk[c] = (uint8_t)md->bucket_of(D);
        ++hist[bk[c]];
    }
    // bucket symbol frequencies: proportional to the histogram, every bucket >= 1, sum 2^BB
    std::vector<uint16_t> bf(nb, 0), bcum(nb + 1, 0);
    if (nb > 1) {
        const uint32_t tot = 1u << BB;
        uint32_t acc = 0, best = 0;
        for (uint32_t b = 0; b < nb; ++b) {
            uint32_t f = (uint32_t)std::max<uint64_t>(1, (uint64_t)hist[b] * tot / std::max<uint32_t>(n, 1));
            bf[b] = (uint16_t)f; acc += f;
            if (hist[b] > hist[best]) best = b;
        }
        int64_t fixed = (int64_t)bf[best] + (int64_t)tot - (int64_t)acc;
        if (fixed < 1 || fixed > (int64_t)tot) throw std::runtime_error("bucket freq overflow");
        bf[best] = (uint16_t)fixed;
        for (uint32_t b = 0; b < nb; ++b) bcum[b + 1] = (uint16_t)(bcum[b] + bf[b]);
    }

    Enc enc;
    std::vector<Sym> syms;
    double side_bits = 0;
    for (int64_t c = (int64_t)n - 1; c >= 0; --c) {
        syms.clear();
        const uint32_t b = bk[c];
        const uint64_t* L = md->L(b);
        if (nb > 1) {
            syms.push_back({bcum[b], bf[b], (uint8_t)BB});
            side_bits += std::log2((double)(1u << BB) / bf[b]);
        }
        uint32_t i1 = 0;
        for (uint32_t k = M.indptr[c]; k < M.indptr[c + 1]; ++k) {
            const uint32_t j = M.indices[k];
            put_gap(syms, T, *md, L, i1, j);
            put_count(syms, *md, j, b, M.data[k]);
            i1 = j + 1;
        }
        put_gap(syms, T, *md, L, i1, m);       // END
        for (size_t s = syms.size(); s-- > 0;) enc.put(syms[s]);
    }
    enc.flush();
    std::reverse(enc.out.begin(), enc.out.end());

    std::vector<uint8_t> o;
    put_u32(o, 0x315A5A49u);                  // "IZZ1"
    put_u64(o, T.sum);
    put_u32(o, (uint32_t)md->tag.size());
    o.insert(o.end(), md->tag.begin(), md->tag.end());
    put_u32(o, M.m); put_u32(o, M.n); put_u64(o, M.nnz); o.push_back(M.vt_code); put_u32(o, nb);
    if (nb > 1) for (uint32_t b = 0; b < nb; ++b) { o.push_back(bf[b] & 0xFF); o.push_back(bf[b] >> 8); }
    put_u64(o, enc.out.size());
    const size_t pre = o.size();
    o.resize(pre + enc.out.size() * 4);
    std::memcpy(o.data() + pre, enc.out.data(), enc.out.size() * 4);
    const size_t before_names = o.size();
    auto rn = join_names(M.rownames), cn = join_names(M.colnames);
    put_stream(o, rn.data(), rn.size(), 3);
    put_stream(o, cn.data(), cn.size(), 3);
    std::vector<uint8_t> kv;
    for (auto& [k, v] : M.user_kv) {
        kv.insert(kv.end(), k.begin(), k.end()); kv.push_back(0);
        kv.insert(kv.end(), v.begin(), v.end()); kv.push_back(0);
    }
    put_u32(o, (uint32_t)M.user_kv.size());
    put_stream(o, kv.data(), kv.size(), 3);
    ctx().names_bytes = o.size() - before_names;
    ctx().side_bytes = (uint64_t)(side_bits / 8);
    ctx().model_bytes = md->file_bytes;
    return o;
}

inline Hdr parse_header(const std::vector<uint8_t>& b) {
    using namespace detail;
    Hdr h;
    const uint8_t* p = b.data();
    if (get_u32(p) != 0x315A5A49u) throw std::runtime_error("idea1: bad magic");
    uint64_t ts = get_u64(p);
    if (ts != tabs().sum) throw std::runtime_error("idea1: integer table checksum mismatch (libm differs)");
    uint32_t tl = get_u32(p);
    std::string tag((const char*)p, tl); p += tl;
    h.m = get_u32(p); h.n = get_u32(p); h.nnz = get_u64(p); h.vt = *p++; h.nb = get_u32(p);
    h.model = load_model(tag);
    if (h.model->nb != h.nb) throw std::runtime_error("idea1: nb mismatch");
    if (h.nb > 1) { h.bf.resize(h.nb); for (uint32_t i = 0; i < h.nb; ++i) { h.bf[i] = (uint16_t)(p[0] | (p[1] << 8)); p += 2; } }
    h.nwords = get_u64(p);
    h.words = reinterpret_cast<const uint32_t*>(p);   // unaligned-safe on x86
    p += h.nwords * 4;
    h.names = p;
    return h;
}

inline void decode_names(const Hdr& h, CanonicalMatrix& M) {
    using namespace detail;
    const uint8_t* p = h.names;
    M.rownames = split_names(get_stream(p), M.m);
    M.colnames = split_names(get_stream(p), M.n);
    uint32_t nkv = get_u32(p);
    auto kv = get_stream(p);
    auto parts = split_names(kv, 2 * (size_t)nkv);
    for (uint32_t i = 0; i < nkv; ++i) M.user_kv[parts[2 * i]] = parts[2 * i + 1];
}

// -------------------------------------------------------------------------------------------------
// Reference (scalar, single state) payload decoder.
inline uint64_t g_corr = 0, g_esc = 0;
inline void decode_payload_ref(const Hdr& h, CanonicalMatrix& M) {
    const Tabs& T = tabs();
    const Model& md = *h.model;
    const uint32_t m = h.m, n = h.n, nb = h.nb;
    Dec d;
    d.x = ((uint64_t)h.words[0] << 32) | h.words[1];
    d.p = h.words + 2;
    // bucket LUT
    std::vector<uint8_t> blut;
    std::vector<uint16_t> bcum(nb + 1, 0);
    if (nb > 1) {
        blut.resize(1u << BB);
        for (uint32_t b = 0; b < nb; ++b) {
            bcum[b + 1] = (uint16_t)(bcum[b] + h.bf[b]);
            for (uint32_t s = bcum[b]; s < bcum[b + 1]; ++s) blut[s] = (uint8_t)b;
        }
    }
    M.indptr.assign(n + 1, 0);
    M.indices.resize(h.nnz);
    M.data.resize(h.nnz);
    uint32_t* ix = M.indices.data();
    uint32_t* dv = M.data.data();
    uint64_t pos = 0;
    const int PB = md.pos_bits;
    for (uint32_t c = 0; c < n; ++c) {
        uint32_t b = 0;
        if (nb > 1) {
            uint32_t slot = (uint32_t)(d.x & ((1u << BB) - 1));
            b = blut[slot];
            d.adv(bcum[b], h.bf[b], BB);
        }
        const uint64_t* L = md.L(b);
        uint32_t i1 = 0;
        for (;;) {
            // ---- next present feature
            uint32_t j;
            uint64_t slot = d.x & (MSC - 1);
            if (slot >= FCUT) {
                ++g_esc;
                d.adv((uint32_t)FCUT, (uint32_t)(MSC - FCUT), MB);
                j = (uint32_t)(d.x & ((1u << PB) - 1));
                d.adv(j, 1, PB);
            } else {
                const uint64_t x0 = L[i1];
                const uint64_t rem = MSC - slot;
                const uint64_t E = ((uint64_t)31 << FR) - log2fix(T, (uint32_t)rem);
                const uint64_t t = x0 + E;
                j = (uint32_t)(std::upper_bound(L + i1, L + m + 1, t) - L) - 1;
                uint64_t c0, c1;
                for (;;) {
                    c0 = MSC - G(T, L[j] - x0);
                    if (c0 > slot) { --j; ++g_corr; continue; }
                    c1 = MSC - G(T, L[j + 1] - x0);
                    if (c1 <= slot) { ++j; ++g_corr; continue; }
                    break;
                }
                d.adv((uint32_t)c0, (uint32_t)(c1 - c0), MB);
            }
            if (j >= m) break;
            // ---- count
            const uint16_t* row = md.ccum.data() + md.crow(j, b);
            uint32_t cs = (uint32_t)(d.x & ((1u << CB) - 1));
            uint32_t s = 0;
            while (cum_at(row, s + 1) <= cs) ++s;
            d.adv(cum_at(row, s), cum_at(row, s + 1) - cum_at(row, s), CB);
            uint32_t cnt = s + 1;
            if (s == 15) {
                uint32_t nbv = (uint32_t)(d.x & 31); d.adv(nbv, 1, 5);
                uint32_t low = 0;
                if (nbv > 16) {
                    uint32_t hi = (uint32_t)(d.x & ((1u << (nbv - 16)) - 1)); d.adv(hi, 1, nbv - 16);
                    uint32_t lo = (uint32_t)(d.x & 0xFFFF); d.adv(lo, 1, 16);
                    low = (hi << 16) | lo;
                } else if (nbv > 0) {
                    low = (uint32_t)(d.x & ((1u << nbv) - 1)); d.adv(low, 1, nbv);
                }
                cnt = 15 + (1u << nbv) + low;
            }
            ix[pos] = j; dv[pos] = cnt; ++pos;
            i1 = j + 1;
        }
        M.indptr[c + 1] = (uint32_t)pos;
    }
    if (pos != h.nnz) throw std::runtime_error("idea1: nnz mismatch after decode");
    if (std::getenv("IDEA1_STATS")) std::fprintf(stderr, "[idea1] corrections=%llu escapes=%llu nnz=%llu\n", (unsigned long long)g_corr, (unsigned long long)g_esc, (unsigned long long)h.nnz);
    g_corr = g_esc = 0;
}

inline CanonicalMatrix decode_matrix(const std::vector<uint8_t>& b, bool fast);

inline CanonicalMatrix decode_matrix_ref(const std::vector<uint8_t>& b) {
    Hdr h = parse_header(b);
    CanonicalMatrix M;
    M.m = h.m; M.n = h.n; M.nnz = h.nnz; M.vt_code = h.vt;
    Timer t;
    decode_payload_ref(h, M);
    double ms = t.ms();
    ctx().dec_payload_ms_min = std::min(ctx().dec_payload_ms_min, ms);
    decode_names(h, M);
    return M;
}

inline Codec make_codec(const std::string& name, const std::string& var, bool insample) {
    Codec c;
    c.name = name;
    c.encode = [var, insample](const CanonicalMatrix& M) {
        return encode_matrix(M, var, insample ? "ins" : ctx().hold);
    };
    c.decode = [](const std::vector<uint8_t>& b) { return decode_matrix_ref(b); };
    return c;
}

}  // namespace i1
}  // namespace cs
