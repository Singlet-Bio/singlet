// Idea 3: single-matrix codecs built on idea3.h (gene -> counts -> interval split).
#pragma once
#include "idea3.h"

namespace cs { namespace i3 {

struct Params {
    bool oos = false;        // presence model comes from the out-of-sample model (not transmitted)
    bool flat = false;       // every row is its own gene (direct interval-level model)
    bool split_adaptive = true;
    bool split_prior = false;      // split starts from the oos prior (needs the model)
    u32 a0 = 2, inc = 32, spmf = 32;
    double shrink = 30.0, alpha = 0.25, prior_mass = 256.0;     // oos finalisation
    u8 flags() const { return (oos ? 1 : 0) | (flat ? 2 : 0) | (split_adaptive ? 4 : 0) | (split_prior ? 8 : 0); }
};

struct LastStats {
    size_t payload_bytes = 0, names_bytes = 0, side_bytes = 0;
    double bits[NCOMP] = {0};
    u64 nnz = 0, nnz_gene = 0, umi = 0;
};
inline LastStats& last_stats() { static LastStats s; return s; }

struct MatState {
    GeneMap gm;
    GeneTables self_gt;            // used when !oos
    GeneTables* gt = nullptr;
    SplitModel sm;
    KCoder kc;
    CountCoder cc;
    SplitScratch sc;
};

// side info of the self-described variant: gene rate class (6 bits, adaptive) and class x bucket presence codes
inline void code_side_self(Coder& c, GeneTables& gt, const ClassTables* ct, u32 G) {
    c.comp = C_SIDE;
    AFreq am; am.init(NCLS, 1, 32);
    gt.G = G; gt.cls.assign(G, 0);
    for (u32 g = 0; g < G; ++g) gt.cls[g] = (u8)am.code(c, c.enc ? ct->cls[g] : 0);
    std::vector<u8> used(NCLS, 0);
    for (u32 g = 0; g < G; ++g) used[gt.cls[g]] = 1;
    std::vector<u16> tab((size_t)NCLS * NB, 65535);
    for (int cl = 0; cl < NCLS; ++cl)
        if (used[cl])
            for (int b = 0; b < NB; ++b) tab[cl * NB + b] = (u16)c.bits(c.enc ? ct->tab[cl * NB + b] : 0, 16);
    gt.lp.assign((size_t)NB * G, 65535);
    for (int b = 0; b < NB; ++b)
        for (u32 g = 0; g < G; ++g) gt.lp[(size_t)b * G + g] = tab[gt.cls[g] * NB + b];
    gt.pref.clear();
}

inline void setup_state(MatState& st, const std::vector<std::string>& rownames, const Params& P, OosModel* oos) {
    st.gm = make_genemap(rownames, P.flat);
    if (P.oos) {
        if (!oos || !oos->built) throw std::runtime_error("idea3: oos model missing");
        st.gt = &oos->gt;
        if (oos->G != st.gm.G) throw std::runtime_error("idea3: oos gene count mismatch");
    } else st.gt = &st.self_gt;
    if (P.split_prior) st.sm.init_prior(&st.gm, oos->split_f0, P.split_adaptive ? P.inc : 0, P.split_adaptive ? P.spmf : 1);
    else st.sm.init_flat(&st.gm, P.a0, P.split_adaptive ? P.inc : 0, P.split_adaptive ? P.spmf : 1);
    st.cc.init(NCLS * NB);
}

// the shared per-cell loop.  enc: M is the input (gc = gene cells); dec: M is the output being built.
inline void code_cells(Coder& c, MatState& st, const CanonicalMatrix* Min, CanonicalMatrix* Mout, const GeneCells* gc) {
    const bool enc = c.enc;
    CanonicalMatrix& M = enc ? const_cast<CanonicalMatrix&>(*Min) : *Mout;   // the encoder only reads M
    const GeneMap& gm = st.gm;
    u32 maxn = 1;
    for (u32 g = 0; g < gm.G; ++g) maxn = std::max(maxn, gm.first[g + 1] - gm.first[g]);
    std::vector<u32> xs(maxn), genes, cnts;
    if (!enc) { M.indptr.assign(M.n + 1, 0); M.indices.clear(); M.data.clear(); M.indices.reserve(M.nnz); M.data.reserve(M.nnz); }
    for (u32 col = 0; col < M.n; ++col) {
        u32 k = enc ? gc->cptr[col + 1] - gc->cptr[col] : 0;
        c.comp = C_K;
        k = st.kc.code(c, k);
        if (k) {
            int b = bucket_of(k);
            genes.resize(k); cnts.resize(k);
            if (enc) for (u32 t = 0; t < k; ++t) { genes[t] = gc->gene[gc->cptr[col] + t]; cnts[t] = gc->cnt[gc->cptr[col] + t]; }
            c.comp = C_PRES;
            code_presence(c, *st.gt, b, k, genes.data());
            c.comp = C_CNT;
            for (u32 t = 0; t < k; ++t) {
                u32 ctx = (u32)st.gt->cls[genes[t]] * NB + (u32)b;
                cnts[t] = st.cc.code(c, ctx, cnts[t]);
            }
            c.comp = C_SPLIT;
            u32 pos = enc ? M.indptr[col] : 0;
            for (u32 t = 0; t < k; ++t) {
                u32 g = genes[t], base = gm.first[g], n = gm.first[g + 1] - base;
                if (enc) {
                    std::fill(xs.begin(), xs.begin() + n, 0u);
                    while (pos < M.indptr[col + 1] && gm.gene_of_row[M.indices[pos]] == g) { xs[M.indices[pos] - base] = M.data[pos]; ++pos; }
                }
                code_split(c, st.sm, g, cnts[t], xs, st.sc);
                if (!enc) for (u32 i = 0; i < n; ++i) if (xs[i]) { M.indices.push_back(base + i); M.data.push_back(xs[i]); }
            }
        }
        if (!enc) M.indptr[col + 1] = (u32)M.indices.size();
    }
}

inline std::vector<u8> encode_matrix(const CanonicalMatrix& M, const Params& P, OosModel* oos) {
    using namespace cs::detail;
    static const bool trace = std::getenv("IDEA3_TRACE") != nullptr;
    Timer tt; double tlast = 0;
    auto mark = [&](const char* what) { if (trace) { double t = tt.ms(); std::fprintf(stderr, "    [enc] %-10s %8.1f ms\n", what, t - tlast); tlast = t; } };
    std::vector<u8> o;
    o.push_back(P.flags());
    put_u32(o, M.m); put_u32(o, M.n); put_u64(o, M.nnz); o.push_back(M.vt_code);
    size_t h0 = o.size();
    auto rn = join_names(M.rownames), cn = join_names(M.colnames);
    put_stream(o, rn.data(), rn.size(), 9);
    put_stream(o, cn.data(), cn.size(), 9);
    std::vector<u8> kv;
    for (auto& [k, v] : M.user_kv) { kv.insert(kv.end(), k.begin(), k.end()); kv.push_back(0); kv.insert(kv.end(), v.begin(), v.end()); kv.push_back(0); }
    put_u32(o, (u32)M.user_kv.size());
    put_stream(o, kv.data(), kv.size(), 9);
    size_t names_bytes = o.size() - h0;
    mark("names");
    MatState st;
    setup_state(st, M.rownames, P, oos);
    mark("setup");
    GeneCells gc = to_gene_cells(M, st.gm);
    mark("genecells");
    Coder c; c.enc = true;
    if (!P.oos) {
        PresStats ps; ps.init(st.gm.G);
        add_pres_stats(ps, gc, M.n, st.gm.G);
        ClassTables ct = make_class_tables(ps, st.gm.G);
        mark("stats");
        code_side_self(c, st.self_gt, &ct, st.gm.G);
        mark("side");
    }
    code_cells(c, st, &M, nullptr, &gc);
    mark("cells");
    c.finish();
    put_u64(o, c.out.size());
    o.insert(o.end(), c.out.begin(), c.out.end());
    LastStats& ls = last_stats();
    ls.payload_bytes = c.out.size(); ls.names_bytes = names_bytes; ls.nnz = M.nnz; ls.nnz_gene = gc.gene.size();
    ls.umi = 0; for (auto v : M.data) ls.umi += v;
    for (int i = 0; i < NCOMP; ++i) ls.bits[i] = c.acc[i];
    ls.side_bytes = (size_t)(c.acc[C_SIDE] / 8);
    return o;
}

inline CanonicalMatrix decode_matrix(const std::vector<u8>& b, const Params& P, OosModel* oos) {
    using namespace cs::detail;
    const u8* p = b.data();
    u8 fl = *p++;
    if (fl != P.flags()) throw std::runtime_error("idea3: param/flag mismatch");
    CanonicalMatrix M;
    M.m = get_u32(p); M.n = get_u32(p); M.nnz = get_u64(p); M.vt_code = *p++;
    M.rownames = split_names(get_stream(p), M.m);
    M.colnames = split_names(get_stream(p), M.n);
    u32 nkv = get_u32(p);
    auto kv = get_stream(p);
    auto parts = split_names(kv, 2 * (size_t)nkv);
    for (u32 i = 0; i < nkv; ++i) M.user_kv[parts[2 * i]] = parts[2 * i + 1];
    u64 plen = get_u64(p);
    MatState st;
    setup_state(st, M.rownames, P, oos);
    Coder c; c.start_dec(p, p + plen);
    if (!P.oos) code_side_self(c, st.self_gt, nullptr, st.gm.G);
    code_cells(c, st, nullptr, &M, nullptr);
    return M;
}

}}  // namespace cs::i3
