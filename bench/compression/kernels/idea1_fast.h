// SPDX-License-Identifier: MIT
// Idea 1 / Idea 5, stage 4 codec wrapper: v2 stream encoder, CanonicalMatrix decode wrapper and the registry.
// The decoder core lives in idea1_core.h (see the description there).
#pragma once
#include "idea1_codec.h"
#include "idea1_core.h"

namespace cs {
namespace i1 {

// ------------------------------------------------------------------------------------------------
// Encoder, format v2
inline std::vector<uint8_t> encode_matrix_v2(const CanonicalMatrix& M, const std::string& var, const std::string& hold) {
    using namespace detail;
    auto md = load_model(model_path(var, M.m, hold));
    if (md->m != M.m) throw std::runtime_error("idea1: model/matrix row mismatch");
    if (md->rnhash != names_hash(M.rownames)) throw std::runtime_error("idea1: rownames differ from model vocabulary");
    int nl = DEFAULT_LANES;
    if (const char* e = std::getenv("IDEA1_LANES")) nl = std::atoi(e);
    EncStats st;
    std::vector<uint8_t> o = encode_v2_payload(*md, M.m, M.n, M.nnz, M.vt_code, M.indptr.data(), M.indices.data(), M.data.data(), nl, &st);
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
    ctx().model_bytes = md->file_bytes;
    ctx().side_bytes = (uint64_t)(st.bucket_bits / 8);
    return o;
}

inline CanonicalMatrix decode_matrix_v2(const std::vector<uint8_t>& b) {
    V2Hdr h = parse_v2(b.data(), b.size());
    auto F = fast_model(h.model);
    CanonicalMatrix M;
    M.m = h.m; M.n = h.n; M.nnz = h.nnz; M.vt_code = h.vt;
    M.indptr.assign(h.n + 1, 0);
    M.indices.resize(h.nnz + 64);
    M.data.resize(h.nnz + 64);
    Timer t;
    double c0 = thread_cycles();
    decode_payload_v2(h, *F, M.indptr.data(), M.indices.data(), M.data.data());
    g_pay_cyc_min = std::min(g_pay_cyc_min, thread_cycles() - c0);
    ctx().dec_payload_ms_min = std::min(ctx().dec_payload_ms_min, t.ms());
    M.indices.resize(h.nnz);
    M.data.resize(h.nnz);
    // names
    using namespace detail;
    const uint8_t* p = h.names;
    M.rownames = split_names(get_stream(p), M.m);
    M.colnames = split_names(get_stream(p), M.n);
    uint32_t nkv = get_u32(p);
    auto kv = get_stream(p);
    auto parts = split_names(kv, 2 * (size_t)nkv);
    for (uint32_t i = 0; i < nkv; ++i) M.user_kv[parts[2 * i]] = parts[2 * i + 1];
    return M;
}

inline Codec make_codec_v2(const std::string& name, const std::string& var, bool insample, bool selected = false) {
    Codec c;
    c.name = name;
    c.encode = [var, insample, selected](const CanonicalMatrix& M) {
        return encode_matrix_v2(M, var, selected ? ctx().sel_hold : (insample ? "ins" : ctx().hold));
    };
    c.decode = [](const std::vector<uint8_t>& b) { return decode_matrix_v2(b); };
    return c;
}

}  // namespace i1

inline void register_idea1(std::vector<Codec>& v) {
    v.push_back(i1::make_codec("idea1_s1", "s1", false));
    v.push_back(i1::make_codec("idea1_s2", "s2", false));
    v.push_back(i1::make_codec("idea1_s3", "s3", false));
    v.push_back(i1::make_codec("idea1_s3r", "s3r", false));
    v.push_back(i1::make_codec("idea1_s2_ins", "s2", true));
    v.push_back(i1::make_codec("idea1_s3_ins", "s3", true));
    v.push_back(i1::make_codec("idea1_s3r_ins", "s3r", true));
    v.push_back(i1::make_codec_v2("idea1_s4", "s3", false));
    v.push_back(i1::make_codec_v2("idea1_s4r", "s3r", false));
    v.push_back(i1::make_codec_v2("idea1_s4_ins", "s3", true));
    v.push_back(i1::make_codec_v2("idea1_s4r_ins", "s3r", true));
    v.push_back(i1::make_codec_v2("idea1_s5r", "s3r", false, true));   // per-file model chosen from the library (results/idea1_choice.tsv)
}
}  // namespace cs
