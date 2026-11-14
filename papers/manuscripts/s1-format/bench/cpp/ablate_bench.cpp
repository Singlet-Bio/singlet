// SPDX-License-Identifier: MIT
// Which parts of the .s1 count model earn their bits?  For the `ablate` stage.
//   ablate_bench <matrix.bin> <repeats> <warmup>
// matrix.bin: u64 rows, u64 cols, u64 nnz, then indptr (cols+1 x u64), row indices (nnz x u32), values (nnz x u32), the
// canonical genes x cells CSC of real data (the same file kernel_bench reads).  One JSON line per variant.
//
// kind "model": the ideal static code length of the codec-1 symbol streams under a chosen set of model components, charged
//   like the encoder does (the table precision, integer log2 of the frequency tables, the serialised zstd-19 tables, and the
//   exact raw "extra" bits).  Each variant adds or removes one component of the codec-1 model:
//     row order          source gene order, or stored class order (genes by how often they are detected)
//     cell-depth context the number of entries in the cell (the k context of the gap table)
//     position context   how far along the gene list the previous entry is (the p context of the gap table)
//     value contexts     the position and the cell-depth class of an entry above 1 (the residual table)
//   The full model's estimate is checked against the real encoder's payload (variant real_default).
// kind "real": the shipped encoder (pz2 encode_matrix) with one option changed, with its true payload bytes and the
//   one-thread decode rate (AVX2 canonical, in memory), checked against the input.  This measures the lane count, the
//   context configuration, the class resolution and the generic codec 2 (zstd over LEB128 gap streams).
// The shipped format and encoder are not changed: everything here uses the public encoder options and the detail headers.
#include <singlet/pz2/pz2.h>
#include <zstd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace pz = singlet::pz2;
namespace r1 = singlet::pz2::detail::r1;
using u32 = uint32_t;
using u64 = uint64_t;

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

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------------------------------- model variants
// Ideal static code length (bits) of a histogram hist[ctx * A + sym] with `pb`-bit tables, plus the serialised table bytes.
static double stream_bits(const std::vector<u64>& hist, size_t nctx, size_t A, int pb, double& table_bytes) {
    r1::EncTables T = r1::build_enc_tables(hist, nctx, A, pb);
    const auto& lg = r1::lg_table();
    double bits = 0;
    for (size_t x = 0; x < nctx; ++x)
        for (size_t s = 0; s < A; ++s) {
            const u64 c = hist[x * A + s];
            if (c) bits += double(c) * (double(u64(pb) << 32) - double(lg[T.freq[x * A + s]])) / 4294967296.0;
        }
    std::vector<uint8_t> tb;
    r1::serialize_tables(tb, T, 19);
    table_bytes = double(tb.size());
    return bits;
}

struct Variant {
    const char* name;
    bool class_order;     // stored class order (cpo 4) or source order
    int kcap, cap;        // gap contexts: cell-depth classes (0 = none), position classes (0 = none)
    int res;              // value contexts: 0 none, 1 position only, 2 depth only, 3 both
};

static r1::Cfg make_cfg(int kcap, int cap) {
    r1::Cfg c;
    c.pb = 10; c.pbv = 10; c.sh = 0; c.ks = 0;
    c.cap = uint8_t(cap); c.kcap = uint8_t(kcap); c.nk = uint8_t(kcap + 1);
    return c;
}

// Row ids -> stored ids (class order by detection counts, or identity).
static std::vector<u32> inverse_order(const Csc& m, bool class_order, int cpo) {
    std::vector<u32> inv(m.rows);
    if (!class_order) {
        for (u32 i = 0; i < m.rows; ++i) inv[i] = i;
        return inv;
    }
    std::vector<u32> cnt(m.rows, 0), key(m.rows);
    for (u64 t = 0; t < m.nnz; ++t) cnt[m.idx[t]]++;
    for (u32 i = 0; i < m.rows; ++i) key[i] = r1::row_key(cnt[i], cpo);
    const std::vector<u32> perm = r1::perm_from_keys(key);   // perm[stored] = canonical
    for (u32 s = 0; s < m.rows; ++s) inv[perm[s]] = s;
    return inv;
}

static void run_model(const Csc& m, const Variant& v, int threads) {
    const std::vector<u32> inv = inverse_order(m, v.class_order, 4);
    // the matrix in stored row space (ids ascending in every column)
    std::vector<u32> ptr32(m.cols + 1);
    for (u64 j = 0; j <= m.cols; ++j) ptr32[j] = u32(m.indptr[j]);
    pz::CSC in;
    in.m = u32(m.rows); in.n = u32(m.cols); in.nnz = m.nnz;
    in.indptr = ptr32.data(); in.indices = m.idx.data(); in.data = m.val.data(); in.data_type = pz::DType::U32;
    r1::ClsMat R;
    r1::to_class_space(in, m.indptr, m.val.data(), inv, R, threads);

    const r1::Cfg cfg = make_cfg(v.kcap, v.cap);
    const size_t A = r1::kAlpha, nctx = cfg.nctx();
    std::vector<u64> hg(nctx * A, 0), hr(size_t(r1::kResCtx) * r1::kNumCls, 0);
    r1::scan_columns(R, cfg, 0, R.n, nullptr, hg.data(), hr.data());

    // raw extra bits of the gap classes and of the residual classes (they are not entropy coded)
    double extra = 0;
    for (u32 j = 0; j < R.n; ++j) {
        u32 pp = 0;
        for (u64 t = R.ptr[j]; t < R.ptr[j + 1]; ++t) {
            u32 ex, nb;
            r1::g_cls(R.rk[t] - pp, ex, nb);
            extra += nb;
            pp = R.rk[t] + 1;
            if (R.val[t] != 1) {
                u32 vex, vnb;
                r1::g_cls(R.val[t] ? R.val[t] - 1 : 0, vex, vnb);
                extra += vnb;
            }
        }
    }
    // residual histogram with the chosen contexts: rc = position * 4 + depth class
    size_t rn = 1;
    if (v.res == 1) rn = r1::kResPos; else if (v.res == 2) rn = 4; else if (v.res == 3) rn = r1::kResCtx;
    std::vector<u64> hr2(rn * r1::kNumCls, 0);
    for (size_t rc = 0; rc < size_t(r1::kResCtx); ++rc) {
        size_t to = v.res == 3 ? rc : v.res == 1 ? rc / 4 : v.res == 2 ? rc % 4 : 0;
        for (size_t s = 0; s < size_t(r1::kNumCls); ++s) hr2[to * r1::kNumCls + s] += hr[rc * r1::kNumCls + s];
    }
    // column counts: class symbol with one 12-bit table
    std::vector<u64> hc(r1::kNumCls, 0);
    double cextra = 0;
    for (u32 j = 0; j < R.n; ++j) {
        u32 ex, nb;
        const u32 cl = r1::g_cls(u32(R.ptr[j + 1] - R.ptr[j]), ex, nb);
        hc[cl]++;
        cextra += nb;
    }
    double tg = 0, tr = 0, tc = 0;
    const double gap_bits = stream_bits(hg, nctx, A, cfg.pb, tg);
    const double res_bits = stream_bits(hr2, rn, r1::kNumCls, cfg.pbv, tr);
    const double cnt_bits = stream_bits(hc, 1, r1::kNumCls, 12, tc);
    const double total = gap_bits + res_bits + cnt_bits + extra + cextra + 8.0 * (tg + tr + tc);
    std::printf("{\"kind\":\"model\",\"variant\":\"%s\",\"nnz\":%llu,\"bits\":%.6f,\"gap_bits\":%.6f,\"res_bits\":%.6f,"
                "\"extra_bits\":%.6f,\"count_bits\":%.6f,\"table_bits\":%.6f}\n",
                v.name, (unsigned long long)m.nnz, total / double(m.nnz), gap_bits / double(m.nnz), res_bits / double(m.nnz),
                extra / double(m.nnz), (cnt_bits + cextra) / double(m.nnz), 8.0 * (tg + tr + tc) / double(m.nnz));
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------------------------------- real encoder
static void run_real(const Csc& m, const char* name, pz::EncodeOptions opt, int reps, int warm) {
    std::vector<u32> ptr32(m.cols + 1);
    for (u64 j = 0; j <= m.cols; ++j) ptr32[j] = u32(m.indptr[j]);
    pz::CSC in;
    in.m = u32(m.rows); in.n = u32(m.cols); in.nnz = m.nnz;
    in.indptr = ptr32.data(); in.indices = m.idx.data(); in.data = m.val.data(); in.data_type = pz::DType::U32;
    opt.row_names.kind = pz::NamesKind::None;
    opt.col_names.kind = pz::NamesKind::None;
    opt.compute_digest = false;
    pz::EncodeStats st;
    double t0 = now();
    std::vector<uint8_t> buf = pz::encode_matrix(in, opt, &st);
    const double enc_s = now() - t0;
    const u64 payload = st.tables + st.rows + st.chunks + st.chunk_index;

    pz::ByteSource src(buf);
    pz::DecodeOptions d;
    d.threads = 1;
    d.isa = pz::isa_supported(pz::Isa::Avx2) ? pz::Isa::Avx2 : pz::Isa::Scalar;
    d.mode = pz::Mode::Canonical;
    std::vector<u64> indptr(m.cols + 1);
    std::vector<u32> indices(m.nnz), data(m.nnz);
    std::string times;
    bool ok = true;
    for (int k = 0; k < warm + reps; ++k) {
        t0 = now();
        pz::Matrix mm = pz::Matrix::open(src);
        pz::Output out;
        out.indptr = indptr.data(); out.indptr_type = pz::IntType::U64; out.indptr_capacity = indptr.size();
        out.indices = indices.data(); out.indices_type = pz::IntType::U32; out.indices_capacity = indices.size();
        out.data = data.data(); out.data_type = pz::DType::U32; out.data_capacity = data.size();
        mm.decode(src, out, d);
        const double s = now() - t0;
        if (k == 0)
            ok = indptr == m.indptr && std::memcmp(indices.data(), m.idx.data(), 4 * m.nnz) == 0 &&
                 std::memcmp(data.data(), m.val.data(), 4 * m.nnz) == 0;
        if (k >= warm) times += (times.empty() ? "" : ",") + std::to_string(s);
    }
    std::printf("{\"kind\":\"real\",\"variant\":\"%s\",\"nnz\":%llu,\"bits\":%.6f,\"payload_bytes\":%llu,\"total_bytes\":%llu,"
                "\"codec\":%d,\"encode_s\":%.6f,\"ok\":%s,\"times\":[%s]}\n",
                name, (unsigned long long)m.nnz, 8.0 * double(payload) / double(m.nnz), (unsigned long long)payload,
                (unsigned long long)st.total, int(st.codec), enc_s, ok ? "true" : "false", times.c_str());
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: ablate_bench matrix.bin repeats warmup\n"); return 2; }
    const Csc m = read_matrix(argv[1]);
    const int reps = std::atoi(argv[2]), warm = std::atoi(argv[3]);
    if (m.nnz >= (u64(1) << 32)) { std::fprintf(stderr, "more than 2^32 non-zeros\n"); return 3; }
    const int threads = 8;

    // component ablation, built up from an order-0 coder and, from the full model, taken apart one component at a time
    const Variant vs[] = {
        {"m0_order0_source", false, 0, 0, 0},
        {"m1_class_order", true, 0, 0, 0},
        {"m2_cell_depth_ctx", true, 11, 0, 0},
        {"m3_position_ctx", true, 11, 39, 0},
        {"m4_value_ctx", true, 11, 39, 3},   // the full model (ctx_kind 0)
        {"loo_no_class_order", false, 11, 39, 3},
        {"loo_no_cell_depth_ctx", true, 0, 39, 3},
        {"loo_no_position_ctx", true, 11, 0, 3},
        {"loo_no_value_ctx", true, 11, 39, 0},
        {"loo_value_position_only", true, 11, 39, 1},
        {"loo_value_depth_only", true, 11, 39, 2},
    };
    for (const Variant& v : vs) run_model(m, v, threads);

    // the shipped encoder with one option changed
    pz::EncodeOptions def;
    run_real(m, "real_default", def, reps, warm);
    for (int k = 0; k < 5; ++k) {
        pz::EncodeOptions o; o.codec = pz::CodecId::Rans; o.ctx_kind = k;
        run_real(m, ("real_ctx" + std::to_string(k)).c_str(), o, reps, warm);
    }
    for (int cpo : {0, 1, 2, 8}) {
        pz::EncodeOptions o; o.codec = pz::CodecId::Rans; o.classes_per_octave = uint8_t(cpo);
        run_real(m, ("real_cpo" + std::to_string(cpo)).c_str(), o, reps, warm);
    }
    for (int ln : {16, 64}) {
        pz::EncodeOptions o; o.codec = pz::CodecId::Rans; o.lanes = uint8_t(ln);
        run_real(m, ("real_lanes" + std::to_string(ln)).c_str(), o, reps, warm);
    }
    {
        pz::EncodeOptions o; o.codec = pz::CodecId::Rans;
        run_real(m, "real_lanes32", o, reps, warm);
    }
    {
        pz::EncodeOptions o; o.codec = pz::CodecId::Generic;
        run_real(m, "real_codec2_zstd", o, reps, warm);
    }
    return 0;
}
