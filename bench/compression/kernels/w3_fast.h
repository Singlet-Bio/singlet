// SPDX-License-Identifier: MIT
// w3_fast: codec glue, ISA dispatch, whole-matrix decode (class-order and canonical), threads, registration.
//   void cs::register_w3_fast(std::vector<Codec>&)   -> codecs "w3f", "w3f_scalar", "w3f_avx2", "w3f_avx512" (canonical decode)
#pragma once
#include "w3_fast_dec.h"
#include <atomic>
#include <functional>
#include <map>
#include <thread>
#if !defined(WF_SCALAR_ONLY) && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VBMI2__) && defined(__AVX512DQ__) && defined(__AVX512VL__) && defined(__BMI2__)
#define WF_HAVE_AVX512 1
#endif
#if !defined(WF_SCALAR_ONLY) && defined(WF_WITH_AVX2) && defined(__AVX2__)   // AVX2 path not written yet: opt-in only
#define WF_HAVE_AVX2 1
#endif
#ifdef WF_HAVE_AVX512
#include "w3_fast_avx512.h"
#endif
#ifdef WF_HAVE_AVX2
#include "w3_fast_avx2.h"
#endif

namespace cs {
namespace wf {

enum class Isa { Scalar = 0, Avx2 = 1, Avx512 = 2 };
inline const char* isa_name(Isa i) { return i == Isa::Scalar ? "scalar" : i == Isa::Avx2 ? "avx2" : "avx512"; }

// Runtime dispatch: the best path this CPU supports (compiled-in AND cpuid-reported).
inline Isa best_isa() {
#ifdef WF_HAVE_AVX512
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vbmi2") && __builtin_cpu_supports("bmi2"))
        return Isa::Avx512;
#endif
#ifdef WF_HAVE_AVX2
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2")) return Isa::Avx2;
#endif
    return Isa::Scalar;
}
inline bool isa_available(Isa i) {
    if (i == Isa::Scalar) return true;
#ifdef WF_HAVE_AVX2
    if (i == Isa::Avx2) return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2");
#endif
#ifdef WF_HAVE_AVX512
    if (i == Isa::Avx512) return best_isa() == Isa::Avx512;
#endif
    return false;
}

template <class OT>
inline void decode_chunk(Isa isa, const Prep& P, const ChunkInfo& ci, int32_t* idx_out, OT* val_out, Setup& su) {
#ifdef WF_HAVE_AVX512
    if (isa == Isa::Avx512) { decode_chunk_avx512<OT>(P, ci, idx_out, val_out, su); return; }
#endif
#ifdef WF_HAVE_AVX2
    if (isa == Isa::Avx2) { decode_chunk_avx2<OT>(P, ci, idx_out, val_out, su); return; }
#endif
    decode_chunk_scalar<OT>(P, ci, idx_out, val_out, su);
}

// Decode every chunk straight into the caller's final arrays (indptr[n+1] int32, idx[nnz] int32, val[nnz] OT).
// Rows are in class order (see RowOrder). nthreads > 1 splits the chunks over threads (chunks are independent).
template <class OT>
inline void decode_all(const Prep& P, Isa isa, int32_t* indptr, int32_t* idx, OT* val, int nthreads = 1) {
    const size_t nch = P.chunks.size();
    indptr[0] = 0;
    auto work = [&](size_t first, size_t step) {
        Setup su;
        for (size_t c = first; c < nch; c += step) {
            const ChunkInfo& ci = P.chunks[c];
            decode_chunk<OT>(isa, P, ci, idx + ci.nnz0, val + ci.nnz0, su);
            for (uint32_t j = 0; j < ci.nc; ++j) indptr[ci.c0 + j + 1] = static_cast<int32_t>(ci.nnz0 + su.colptr[j + 1]);
        }
    };
    if (nthreads <= 1 || nch <= 1) { work(0, 1); return; }
    // dynamic chunk dispatch: a shared counter, so uneven chunks do not stall the pool
    std::atomic<size_t> next{0};
    auto dyn = [&]() {
        Setup su;
        for (;;) {
            size_t c = next.fetch_add(1);
            if (c >= nch) break;
            const ChunkInfo& ci = P.chunks[c];
            decode_chunk<OT>(isa, P, ci, idx + ci.nnz0, val + ci.nnz0, su);
            for (uint32_t j = 0; j < ci.nc; ++j) indptr[ci.c0 + j + 1] = static_cast<int32_t>(ci.nnz0 + su.colptr[j + 1]);
        }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < nthreads; ++t) th.emplace_back(dyn);
    dyn();
    for (auto& t : th) t.join();
}

// ---- canonical order (original row ids ascending) ---------------------------------------------------------------------------
// Straightforward exact path: map class-order ids to original ids and sort each column. (Faster variants live in
// w3_fast_canon.h.)
inline void canonicalise_scalar(const uint32_t* perm, const int32_t* indptr, uint32_t n, uint32_t* idx, uint32_t* val) {
    std::vector<uint64_t> key;
    for (uint32_t j = 0; j < n; ++j) {
        uint32_t a = static_cast<uint32_t>(indptr[j]), b = static_cast<uint32_t>(indptr[j + 1]);
        key.resize(b - a);
        for (uint32_t t = a; t < b; ++t) key[t - a] = (static_cast<uint64_t>(perm[idx[t]]) << 32) | val[t];
        std::sort(key.begin(), key.end());
        for (uint32_t t = a; t < b; ++t) { idx[t] = static_cast<uint32_t>(key[t - a] >> 32); val[t] = static_cast<uint32_t>(key[t - a]); }
    }
}

struct Decoded {   // everything a caller needs
    Prep P;
    CanonicalMatrix M;     // rows in ORIGINAL order, names original (exact)
};

inline CanonicalMatrix decode_canonical(const std::vector<uint8_t>& b, Isa isa) {
    Prep P;
    parse_file(b.data(), P);
    CanonicalMatrix M;
    M.m = P.h.m; M.n = P.h.n; M.nnz = P.h.nnz; M.vt_code = P.h.vt;
    const uint8_t* p = P.names_p;
    get_names(p, M);
    p = P.rowc_p;
    auto rnnz = get_rowcounts(p, M.m);
    RowOrder O = order_from_counts(rnnz, P.h.cpo);
    M.indptr.resize(M.n + 1);
    M.indices.resize(M.nnz);
    M.data.resize(M.nnz);
    decode_all<uint32_t>(P, isa, reinterpret_cast<int32_t*>(M.indptr.data()), reinterpret_cast<int32_t*>(M.indices.data()), M.data.data(), 1);
    canonicalise_scalar(O.perm.data(), reinterpret_cast<const int32_t*>(M.indptr.data()), M.n, M.indices.data(), M.data.data());
    return M;
}

inline std::map<std::string, Params>& w3f_params() {
    static std::map<std::string, Params> m;
    return m;
}

inline Codec make_w3f(const std::string& name, Params prm, Isa isa, bool autoisa) {
    Codec c;
    c.name = name;
    c.encode = [prm](const CanonicalMatrix& M) { return encode_matrix(M, prm); };
    c.decode = [isa, autoisa](const std::vector<uint8_t>& b) { return decode_canonical(b, autoisa ? best_isa() : isa); };
    return c;
}

}  // namespace wf

inline void register_w3_fast(std::vector<Codec>& v) {
    wf::Params p;
    v.push_back(wf::make_w3f("w3f", p, wf::Isa::Scalar, true));
    v.push_back(wf::make_w3f("w3f_scalar", p, wf::Isa::Scalar, false));
#ifdef WF_HAVE_AVX2
    v.push_back(wf::make_w3f("w3f_avx2", p, wf::Isa::Avx2, false));
#endif
#ifdef WF_HAVE_AVX512
    v.push_back(wf::make_w3f("w3f_avx512", p, wf::Isa::Avx512, false));
#endif
}

}  // namespace cs
