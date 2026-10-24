// Idea 3, part B: intron matrix coded conditionally on the exon matrix of the same sample (same barcodes, same order).
//
// Per cell, the exon side (gene-level counts e_g, decoded first) is known.  The intron gene set is coded in two
// parts:  (1) for every gene g with exon counts and with introns ("S genes"), one binary decision
//         P(intron present | gene rate class, depth bucket of the cell's intron nnz, exon-count bucket of g in {1,2-3,4+});
//         (2) the remaining k_out = k_I - (#present in S) genes among the non-S genes by the bisection coder of idea3.h with
//         window probabilities that exclude the S genes (P(present | e=0, bucket, class) tables).
// Counts-given-presence get the exon-count bucket as an extra context.  The interval split is unchanged.
#pragma once
#include "idea3_codec.h"

namespace cs { namespace i3 {

constexpr int NBC = NB / 2;      // coarse buckets for the S-gene table

inline int ebucket(u32 e) { return e == 0 ? 0 : (e == 1 ? 1 : (e <= 3 ? 2 : 3)); }

struct CondTables {
    u32 G = 0;                         // intron genes
    GeneTables gt0;                    // cls + lp (P(present | e=0, bucket)) as codes
    std::vector<u16> pS;               // NCLS*NBC*3 : P1 (of 65536) for S genes
    size_t bytes() const { return gt0.bytes() + pS.size() * 2; }
};

// mapping intron gene -> exon gene id index (from gene id strings); -1 if absent
inline std::vector<int> map_e2i(const GeneMap& ge, const GeneMap& gi) {
    std::unordered_map<std::string, u32> pos;
    for (u32 g = 0; g < gi.G; ++g) pos[gi.ids[g]] = g;
    std::vector<int> e2i(ge.G, -1);
    for (u32 g = 0; g < ge.G; ++g) { auto it = pos.find(ge.ids[g]); if (it != pos.end()) e2i[g] = (int)it->second; }
    return e2i;
}

struct SInfo {                         // exon-derived info for one cell, restricted to intron genes
    std::vector<u32> pos;              // intron gene indices, ascending
    std::vector<u32> e;                // exon counts
    std::vector<i64> sps;              // prefix sums of L0 at those positions (bucket-dependent; built per cell)
};

// presence of k genes among the non-S genes (window probabilities exclude S)
inline void code_presence_excl(Coder& c, GeneTables& gt, int b, u32 k, u32* genes, const std::vector<u32>& sp, const std::vector<i64>& sps) {
    const i64* Lp = gt.prefix(b);
    const u32 G = gt.G, nS = (u32)sp.size();
    auto cntS = [&](u32 x) -> u32 { return (u32)(std::lower_bound(sp.begin(), sp.end(), x) - sp.begin()); };
    auto Dl = [&](u32 a, u32 x) -> i64 { return (Lp[x] - Lp[a]) - (sps[cntS(x)] - sps[cntS(a)]); };
    u32 lo = 0;
    for (u32 t = 0; t < k; ++t) {
        u32 r = k - t;
        // largest x with NS(x) >= r where NS(x) = non-S genes in [x, G)
        u32 a = lo, bnd = G;                           // search x in [lo, G): NS is non-increasing in x
        while (bnd - a > 1) {
            u32 mid = (a + bnd) >> 1;
            u32 ns = (G - mid) - (nS - cntS(mid));
            if (ns >= r) a = mid; else bnd = mid;
        }
        u32 hi = a + 1;
        u32 target = c.enc ? genes[t] : 0;
        u64 Fh = omexp(Dl(lo, hi));
        while (hi - lo > 1) {
            u32 mid = (lo + hi) >> 1;
            u64 Fm = omexp(Dl(lo, mid));
            u64 P = Fh ? (Fm << 16) / Fh : 32768;
            if (P < 1) P = 1;
            if (P > 65535) P = 65535;
            u32 bit = c.bin((u32)P, c.enc ? (target < mid) : 0);
            if (bit) { hi = mid; Fh = Fm; }
            else { lo = mid; Fh = omexp(Dl(lo, hi)); }
        }
        genes[t] = lo;
        lo = lo + 1;
    }
}

struct JointState {
    GeneMap gmE, gmI;
    std::vector<int> e2i;
    CondTables ct;
    SplitModel sm;
    KCoder kc;
    CountCoder cc;
    SplitScratch sc;
};

inline void code_side_joint(Coder& c, CondTables& ct, const ClassTables* cls_tab, const std::vector<u16>* tab0, const std::vector<u16>* tabS, u32 G) {
    c.comp = C_SIDE;
    AFreq am; am.init(NCLS, 1, 32);
    ct.G = G; ct.gt0.G = G; ct.gt0.cls.assign(G, 0);
    for (u32 g = 0; g < G; ++g) ct.gt0.cls[g] = (u8)am.code(c, c.enc ? cls_tab->cls[g] : 0);
    std::vector<u8> used(NCLS, 0);
    for (u32 g = 0; g < G; ++g) used[ct.gt0.cls[g]] = 1;
    std::vector<u16> t0((size_t)NCLS * NB, 65535), tS((size_t)NCLS * NBC * 3, 6554);
    for (int cl = 0; cl < NCLS; ++cl) {
        if (!used[cl]) continue;
        for (int b = 0; b < NB; ++b) t0[cl * NB + b] = (u16)c.bits(c.enc ? (*tab0)[cl * NB + b] : 0, 16);
        for (int i = 0; i < NBC * 3; ++i) tS[cl * NBC * 3 + i] = (u16)c.bits(c.enc ? (*tabS)[cl * NBC * 3 + i] : 0, 16);
    }
    ct.gt0.lp.assign((size_t)NB * G, 65535);
    for (int b = 0; b < NB; ++b)
        for (u32 g = 0; g < G; ++g) ct.gt0.lp[(size_t)b * G + g] = t0[ct.gt0.cls[g] * NB + b];
    ct.gt0.pref.clear();
    ct.pS = tS;
}

// shared per-cell loop for the conditional intron coder.  enc: Iin is the input; dec: Iout built.  E is the (decoded) exon matrix.
inline void code_cells_cond(Coder& c, JointState& st, const CanonicalMatrix* Iin, CanonicalMatrix* Iout, const GeneCells* gcI, const GeneCells& gcE, bool use_cond) {
    const bool enc = c.enc;
    CanonicalMatrix& M = enc ? const_cast<CanonicalMatrix&>(*Iin) : *Iout;
    const GeneMap& gm = st.gmI;
    u32 maxn = 1;
    for (u32 g = 0; g < gm.G; ++g) maxn = std::max(maxn, gm.first[g + 1] - gm.first[g]);
    std::vector<u32> xs(maxn), genes, cnts, sg, se, outg, flag;
    std::vector<i64> sps;
    std::vector<u32> ecount(gm.G, 0);
    GeneTables& gt0 = st.ct.gt0;
    const Tabs& T = tabs();
    std::vector<u8> mark(gm.G, 0);
    if (!enc) { M.indptr.assign(M.n + 1, 0); M.indices.clear(); M.data.clear(); M.indices.reserve(M.nnz); M.data.reserve(M.nnz); }
    for (u32 col = 0; col < M.n; ++col) {
        u32 k = enc ? gcI->cptr[col + 1] - gcI->cptr[col] : 0;
        c.comp = C_K;
        k = st.kc.code(c, k);
        if (k) {
            int b = bucket_of(k);
            genes.assign(k, 0); cnts.assign(k, 0);
            if (enc) for (u32 t = 0; t < k; ++t) { genes[t] = gcI->gene[gcI->cptr[col] + t]; cnts[t] = gcI->cnt[gcI->cptr[col] + t]; mark[genes[t]] = 1; }
            // S genes of this cell
            sg.clear(); se.clear();
            if (use_cond)
                for (u32 t = gcE.cptr[col]; t < gcE.cptr[col + 1]; ++t) {
                    int gi = st.e2i[gcE.gene[t]];
                    if (gi >= 0) { sg.push_back((u32)gi); se.push_back(gcE.cnt[t]); }
                }
            // (sorted ascending already if e2i is monotone; sort pairs to be safe)
            bool sorted = true;
            for (size_t i = 1; i < sg.size(); ++i) if (sg[i] < sg[i - 1]) { sorted = false; break; }
            if (!sorted) {
                std::vector<std::pair<u32, u32>> pr(sg.size());
                for (size_t i = 0; i < sg.size(); ++i) pr[i] = {sg[i], se[i]};
                std::sort(pr.begin(), pr.end());
                for (size_t i = 0; i < sg.size(); ++i) { sg[i] = pr[i].first; se[i] = pr[i].second; }
            }
            c.comp = C_PRES;
            u32 min_ = 0;
            outg.clear();
            for (size_t i = 0; i < sg.size(); ++i) {
                u32 g = sg[i];
                int eb = ebucket(se[i]);
                u32 P1 = st.ct.pS[(gt0.cls[g] * NBC + b / 2) * 3 + (eb - 1)];
                u32 bit = c.bin(P1, enc ? mark[g] : 0);
                if (bit) { ++min_; outg.push_back(g); }
                ecount[g] = se[i];
            }
            u32 kout = k - min_;
            // non-S part
            sps.assign(sg.size() + 1, 0);
            {
                const u16* l = gt0.lp.data() + (size_t)b * gt0.G;
                for (size_t i = 0; i < sg.size(); ++i) sps[i + 1] = sps[i] + T.Lc[l[sg[i]]];
            }
            if (kout) {
                std::vector<u32> go(kout);
                if (enc) { u32 q = 0; for (u32 t = 0; t < k; ++t) { u32 g = genes[t]; if (!std::binary_search(sg.begin(), sg.end(), g) || false) go[q++] = g; } }
                code_presence_excl(c, gt0, b, kout, go.data(), sg, sps);
                for (u32 g : go) outg.push_back(g);
            }
            std::sort(outg.begin(), outg.end());
            if (!enc) for (u32 t = 0; t < k; ++t) genes[t] = outg[t];
            c.comp = C_CNT;
            for (u32 t = 0; t < k; ++t) {
                u32 g = genes[t];
                u32 ctx = (u32)gt0.cls[g] * NB + (u32)b;
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
            for (u32 g : sg) ecount[g] = 0;
            if (enc) for (u32 t = 0; t < k; ++t) mark[genes[t]] = 0;
        }
        if (!enc) M.indptr[col + 1] = (u32)M.indices.size();
    }
}

struct JointOpts { bool use_cond = true; u32 a0 = 2, inc = 32, spmf = 32; };

inline void setup_joint(JointState& st, const CanonicalMatrix& E_names_src, const std::vector<std::string>& irows, const JointOpts& o) {
    st.gmE = make_genemap(E_names_src.rownames, false);
    st.gmI = make_genemap(irows, false);
    st.e2i = map_e2i(st.gmE, st.gmI);
    st.sm.init_flat(&st.gmI, o.a0, o.inc, o.spmf);
    st.cc.init(NCLS * NB);
}

inline std::vector<u8> encode_intron_cond(const CanonicalMatrix& I, const CanonicalMatrix& E, const JointOpts& o) {
    using namespace cs::detail;
    std::vector<u8> out;
    out.push_back(o.use_cond ? 1 : 0);
    put_u32(out, I.m); put_u32(out, I.n); put_u64(out, I.nnz); out.push_back(I.vt_code);
    size_t h0 = out.size();
    auto rn = join_names(I.rownames), cn = join_names(I.colnames);
    put_stream(out, rn.data(), rn.size(), 9);
    put_stream(out, cn.data(), cn.size(), 9);
    std::vector<u8> kv;
    for (auto& [k, v] : I.user_kv) { kv.insert(kv.end(), k.begin(), k.end()); kv.push_back(0); kv.insert(kv.end(), v.begin(), v.end()); kv.push_back(0); }
    put_u32(out, (u32)I.user_kv.size());
    put_stream(out, kv.data(), kv.size(), 9);
    size_t names_bytes = out.size() - h0;

    JointState st;
    setup_joint(st, E, I.rownames, o);
    const GeneMap& gmI = st.gmI;
    GeneCells gcI = to_gene_cells(I, gmI), gcE = to_gene_cells(E, st.gmE);
    // statistics for the tables
    PresStats ps; ps.init(gmI.G);
    add_pres_stats(ps, gcI, I.n, gmI.G);
    ClassTables ctab = make_class_tables(ps, gmI.G);       // classes by marginal intron presence rate
    const u32 Gi = gmI.G;
    std::vector<double> ngcls(NCLS, 0.0);
    for (u32 g = 0; g < Gi; ++g) ngcls[ctab.cls[g]] += 1;
    std::vector<double> N0((size_t)NCLS * NB, 0.0), Y0((size_t)NCLS * NB, 0.0);
    std::vector<double> NSt((size_t)NCLS * NBC * 3, 0.0), YSt((size_t)NCLS * NBC * 3, 0.0);
    std::vector<double> Nb(NB, 0.0), Ns0((size_t)NCLS * NB, 0.0);
    {
        std::vector<u8> mark(Gi, 0), emark(Gi, 0);
        for (u32 col = 0; col < I.n; ++col) {
            u32 k = gcI.cptr[col + 1] - gcI.cptr[col];
            if (!k) continue;
            int b = bucket_of(k);
            Nb[b] += 1;
            for (u32 t = gcI.cptr[col]; t < gcI.cptr[col + 1]; ++t) mark[gcI.gene[t]] = 1;
            if (o.use_cond) {
                for (u32 t = gcE.cptr[col]; t < gcE.cptr[col + 1]; ++t) {
                    int gi = st.e2i[gcE.gene[t]];
                    if (gi < 0) continue;
                    int eb = ebucket(gcE.cnt[t]);
                    int cl = ctab.cls[gi];
                    size_t ix = ((size_t)cl * NBC + b / 2) * 3 + (eb - 1);
                    NSt[ix] += 1; YSt[ix] += mark[gi];
                    Ns0[(size_t)cl * NB + b] += 1;        // pairs with e>0 in (cls,b)
                    emark[gi] = 1;
                }
            }
            for (u32 t = gcI.cptr[col]; t < gcI.cptr[col + 1]; ++t) { u32 g = gcI.gene[t]; if (!emark[g]) Y0[(size_t)ctab.cls[g] * NB + b] += 1; }
            for (u32 t = gcI.cptr[col]; t < gcI.cptr[col + 1]; ++t) mark[gcI.gene[t]] = 0;
            if (o.use_cond) for (u32 t = gcE.cptr[col]; t < gcE.cptr[col + 1]; ++t) { int gi = st.e2i[gcE.gene[t]]; if (gi >= 0) emark[gi] = 0; }
        }
    }
    std::vector<u16> tab0((size_t)NCLS * NB, 65535), tabS((size_t)NCLS * NBC * 3, 6554);
    for (int cl = 0; cl < NCLS; ++cl) {
        for (int b = 0; b < NB; ++b) {
            double n0 = ngcls[cl] * Nb[b] - Ns0[(size_t)cl * NB + b];
            tab0[cl * NB + b] = (u16)quant_code(n0 > 0 ? Y0[(size_t)cl * NB + b] / n0 : 0.0);
        }
        for (int i = 0; i < NBC * 3; ++i) {
            double n = NSt[(size_t)cl * NBC * 3 + i], y = YSt[(size_t)cl * NBC * 3 + i];
            double p = (y + 0.1) / (n + 0.2);
            long v = std::lround(p * 65536.0);
            tabS[cl * NBC * 3 + i] = (u16)(v < 1 ? 1 : (v > 65535 ? 65535 : v));
        }
    }
    Coder c; c.enc = true;
    code_side_joint(c, st.ct, &ctab, &tab0, &tabS, Gi);
    code_cells_cond(c, st, &I, nullptr, &gcI, gcE, o.use_cond);
    c.finish();
    put_u64(out, c.out.size());
    out.insert(out.end(), c.out.begin(), c.out.end());
    LastStats& ls = last_stats();
    ls.payload_bytes = c.out.size(); ls.names_bytes = names_bytes; ls.nnz = I.nnz; ls.nnz_gene = gcI.gene.size();
    ls.umi = 0; for (auto v : I.data) ls.umi += v;
    for (int i = 0; i < NCOMP; ++i) ls.bits[i] = c.acc[i];
    ls.side_bytes = (size_t)(c.acc[C_SIDE] / 8);
    return out;
}

inline CanonicalMatrix decode_intron_cond(const std::vector<u8>& b, const CanonicalMatrix& E, const JointOpts& o) {
    using namespace cs::detail;
    const u8* p = b.data();
    u8 fl = *p++;
    if ((fl != 0) != o.use_cond) throw std::runtime_error("joint: flag mismatch");
    CanonicalMatrix M;
    M.m = get_u32(p); M.n = get_u32(p); M.nnz = get_u64(p); M.vt_code = *p++;
    M.rownames = split_names(get_stream(p), M.m);
    M.colnames = split_names(get_stream(p), M.n);
    u32 nkv = get_u32(p);
    auto kv = get_stream(p);
    auto parts = split_names(kv, 2 * (size_t)nkv);
    for (u32 i = 0; i < nkv; ++i) M.user_kv[parts[2 * i]] = parts[2 * i + 1];
    u64 plen = get_u64(p);
    JointState st;
    setup_joint(st, E, M.rownames, o);
    GeneCells gcE = to_gene_cells(E, st.gmE);
    Coder c; c.start_dec(p, p + plen);
    code_side_joint(c, st.ct, nullptr, nullptr, nullptr, st.gmI.G);
    code_cells_cond(c, st, nullptr, &M, nullptr, gcE, o.use_cond);
    return M;
}

}}  // namespace cs::i3
