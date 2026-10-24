// Track C codec: learned (out-of-sample) probability model + range coder for .1pz count matrices.
// See notes/track_c.md. Decode path: integer arithmetic only.
#pragma once
#include "common.h"
#include "ref_zstd_csc.h"
#include "track_c_core.h"
#include <cmath>
#include <filesystem>
#include <memory>
#include <sstream>

namespace cs { namespace tc {

// ------------------------------------------------------------------ global configuration / reporting
inline std::string g_holdout;                 // "" = automatic (fingerprint table), "none" = in-sample model, else study id
inline std::string g_model_dir = "results";
inline int g_names_level = 19;                // zstd level for the names/kv streams (Track A's problem; 1 for quick experiments)
inline int g_vlimit = 250;                    // adaptation limit (count cap) for value/bucket models
inline int g_refresh = 16;                    // table refresh period (power of two) after the first 64 columns of a bucket
inline int g_coarse = 0;                      // encoder-only class coarsening experiment
inline std::vector<float> g_col_bits;         // encoder: ideal presence+value bits per column (diagnostics)
inline int g_kappa = 0;                       // >0 overrides kappa_e at encode time (experiments)
inline int g_blimit = 60;                     // adaptation limit for the bucket trees
inline int g_cptn = -1;                       // >=0 overrides cpt_n at encode time (experiments)
inline int g_r3n = 12;                        // rung 3: prior weight (pseudo-columns) of the trained cluster rates
inline bool g_stats = true;                   // encoder collects ideal-cost breakdown
struct Report {
    double side_bits = 0, bucket_bits = 0, pres_bits = 0, val_bits = 0;   // ideal (cross-entropy) costs from the encoder
    double pb[40] = {0}, vb[40] = {0}, nb[40] = {0};
    size_t rc_bytes = 0, names_bytes = 0, total_bytes = 0, model_bytes = 0;
    uint64_t nnz = 0; std::string tag; int mode = 0;
};
inline Report g_report;

static const int NB1 = TC_NB + 1;             // 33 depth buckets (0 = empty column)
static const int NC2 = 121;                   // file-class alphabet (0 = unused feature)
static const int NSYM_BITS = 7;
static const int NCX = 105;                   // CPT contexts (feature level)
static const int NST = 3;                     // gene-position states
static const int NLEV = 7;                    // unary value levels

// ------------------------------------------------------------------ model
struct Model {
    u32 m = 0, nc1 = 0, ncg1 = 0, ncg2 = 0, nbg = 0;
    u64 name_hash = 0; size_t file_bytes = 0; std::string tag;
    std::vector<u8> plevel, cls1, gfirst;
    std::vector<u16> p0_r1, p0_r1b, cpt, v0_r1, v0_r1b;
    bool ok = false; bool feat_indep = false;     // ver 2: feature-independent tables only (cross-species transfer)
    // rung 3 (optional file track_c_r3_<tag>.bin): head features + per-(cluster, tier, feature) presence codes
    bool has_r3 = false; u32 H3 = 0, K3 = 0, T3 = 0; std::vector<u32> head; std::vector<u8> R3; size_t r3_bytes = 0;
};

inline void load_r3(Model& mod, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return;
    char mg[4]; f.read(mg, 4);
    if (std::string(mg, 4) != "TCR3") return;
    u32 h, k, t; f.read((char*)&h, 4); f.read((char*)&k, 4); f.read((char*)&t, 4);
    if (k > 16 || t > 16) throw std::runtime_error("r3 model shape unsupported " + path);
    mod.head.resize(h); f.read((char*)mod.head.data(), (std::streamsize)h * 4);
    mod.R3.resize((size_t)k * t * h); f.read((char*)mod.R3.data(), (std::streamsize)mod.R3.size());
    if (!f) throw std::runtime_error("short r3 model " + path);
    mod.H3 = h; mod.K3 = k; mod.T3 = t; mod.has_r3 = true;
    mod.r3_bytes = (size_t)std::filesystem::file_size(path);
}
static const int R3_MIN_BUCKET = 17;          // columns with nnz >= 128 get a cluster id
// depth tier of a clustered column: T == 4 -> 3-bucket groups; otherwise one tier per bucket (capped at T-1)
inline int tier_of(int b, int T) {
    if (T == 4) return b <= 19 ? 0 : (b <= 22 ? 1 : (b <= 25 ? 2 : 3));
    int t = b - R3_MIN_BUCKET; return t < T ? t : T - 1;
}

inline std::shared_ptr<Model> load_model(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return nullptr;
    auto mod = std::make_shared<Model>();
    auto rd = [&](void* p, size_t n) { f.read((char*)p, (std::streamsize)n); if (!f) throw std::runtime_error("short model file " + path); };
    u32 magic, ver; rd(&magic, 4); rd(&ver, 4);
    if (magic != 0x314D4354u) return nullptr;
    mod->feat_indep = (ver == 2);
    u32 nb1, nc2, ncx, nst;
    rd(&mod->m, 4); rd(&mod->name_hash, 8); rd(&nb1, 4); rd(&mod->nc1, 4); rd(&nc2, 4); rd(&ncx, 4); rd(&nst, 4);
    rd(&mod->ncg1, 4); rd(&mod->ncg2, 4); rd(&mod->nbg, 4);
    if (nb1 != (u32)NB1 || nc2 != (u32)NC2 || ncx != (u32)NCX || nst != (u32)NST) throw std::runtime_error("model shape mismatch " + path);
    mod->plevel.resize(mod->m); mod->cls1.resize(mod->m); mod->gfirst.resize(mod->m);
    rd(mod->plevel.data(), mod->m); rd(mod->cls1.data(), mod->m); rd(mod->gfirst.data(), mod->m);
    mod->p0_r1.resize((size_t)NB1 * mod->nc1); rd(mod->p0_r1.data(), mod->p0_r1.size() * 2);
    mod->p0_r1b.resize((size_t)NB1 * NC2); rd(mod->p0_r1b.data(), mod->p0_r1b.size() * 2);
    mod->cpt.resize((size_t)NCX * NST * 127); rd(mod->cpt.data(), mod->cpt.size() * 2);
    mod->v0_r1.resize((size_t)mod->ncg1 * mod->nbg * NLEV); rd(mod->v0_r1.data(), mod->v0_r1.size() * 2);
    mod->v0_r1b.resize((size_t)mod->ncg2 * mod->nbg * NLEV); rd(mod->v0_r1b.data(), mod->v0_r1b.size() * 2);
    mod->file_bytes = (size_t)std::filesystem::file_size(path);
    mod->ok = true;
    return mod;
}

inline std::map<std::string, std::shared_ptr<Model>>& model_cache() { static std::map<std::string, std::shared_ptr<Model>> c; return c; }
inline std::shared_ptr<Model> get_model(const std::string& tag) {
    if (tag.empty()) return nullptr;
    auto& c = model_cache();
    auto it = c.find(tag);
    if (it != c.end()) return it->second;
    auto mod = load_model(g_model_dir + "/track_c_model_" + tag + ".bin");
    if (mod) { mod->tag = tag; load_r3(*mod, g_model_dir + "/track_c_r3_" + tag + ".bin"); }
    c[tag] = mod;
    return mod;
}

inline u64 hash_names(const std::vector<std::string>& v) {
    u64 h = 1469598103934665603ull;
    for (auto& s : v) h = fnv64(s.data(), s.size(), h);
    return h;
}
inline u64 hash_cols(const std::vector<std::string>& v) { return hash_names(v); }

// identify (organism_member) by the row-name hash recorded in the model files; holdout by sample fingerprint table.
inline std::string find_model_tag(const CanonicalMatrix& M, const std::string& hold_override = "") {
    static std::map<u64, std::string> rowmap;      // name_hash -> "human_exon"
    static std::map<u64, std::string> samples;     // colname hash -> study id
    static bool init = false;
    if (!init) {
        init = true;
        std::error_code ec;
        for (auto& e : std::filesystem::directory_iterator(g_model_dir, ec)) {
            std::string fn = e.path().filename().string();
            if (fn.rfind("track_c_model_", 0) != 0 || fn.size() < 4 || fn.substr(fn.size() - 4) != ".bin") continue;
            std::ifstream f(e.path(), std::ios::binary);
            u32 hd[3]; u64 nh;
            f.read((char*)hd, 12); f.read((char*)&nh, 8);
            if (hd[0] != 0x314D4354u) continue;
            std::string tag = fn.substr(14, fn.size() - 18);              // human_exon_none
            auto p = tag.find('_', tag.find('_') + 1);
            rowmap[nh] = tag.substr(0, p);
        }
        std::ifstream s(g_model_dir + "/track_c_samples.tsv");
        std::string line;
        while (std::getline(s, line)) {
            std::istringstream is(line); std::string h, st; if (is >> h >> st) samples[std::stoull(h)] = st;
        }
    }
    auto it = rowmap.find(hash_names(M.rownames));
    if (it == rowmap.end()) return "";
    std::string hold = hold_override.empty() ? g_holdout : hold_override;
    if (hold.empty()) {
        auto sm = samples.find(hash_cols(M.colnames));
        hold = (sm == samples.end()) ? "none" : sm->second;
    }
    std::string tag = it->second + "_" + hold;
    if (!std::filesystem::exists(g_model_dir + "/track_c_model_" + tag + ".bin")) tag = it->second + "_none";
    if (!std::filesystem::exists(g_model_dir + "/track_c_model_" + tag + ".bin")) return "";
    return tag;
}

// ------------------------------------------------------------------ codec parameters
struct Params {
    bool file_classes = true;     // classes from the file's own per-feature presence counts (side info) vs trained profile classes
    bool adapt = true;            // online adaptation of the (bucket, class) presence table
    int kappa_e = 1;              // prior strength (pseudo events) for adaptation
    int cpt_n = 8;                // confidence (pseudo-count) of the trained class-CPT; it then adapts to the file
    bool use_model = true;
    bool insample = false;        // use the model trained on everything (upper bound, labelled)
    bool latent = false;          // rung 3: head features coded per feature under a (cluster, tier) adaptive table; cluster id is side info
    std::string name;
};

// ------------------------------------------------------------------ varint helpers
inline void put_varint(std::vector<u8>& o, u64 v) { while (v >= 128) { o.push_back((u8)(v | 128)); v >>= 7; } o.push_back((u8)v); }
inline u64 get_varint(const u8*& p) { u64 v = 0; int s = 0; while (true) { u8 b = *p++; v |= (u64)(b & 127) << s; if (!(b & 128)) break; s += 7; } return v; }

inline int bucket_of(u32 nnz) {
    if (nnz == 0) return 0;
    int b = 0; for (int i = 0; i < TC_NB; ++i) b += (TC_BTH[i] <= nnz);
    return b;
}

// ------------------------------------------------------------------ the engine (shared by encoder and decoder)
template <class RC, bool ENC>
struct Engine {
    RC& rc;
    const Params& P;
    const Model* mod;
    u32 m, n;
    Cost* cost;

    // class structure
    int NC = 0;
    std::vector<u32> runlen, runstart;            // per class
    std::vector<u32> feature_of_pos, pos_of_feature;
    std::vector<u8> cls;                          // class per feature
    // presence tables
    std::vector<u64> H, ES; std::vector<u32> G;  // [NB1*NC]  hazard, E40(h*stride), stride
    std::vector<u64> Acnt, p0q, inv0; std::vector<u64> Ncols;
    // value models
    std::vector<ABit> vtab; std::vector<ABit> eg; std::vector<ABit> btree_unused; std::vector<ABit> ctree;
    int ncg = 0, nbg = 0;
    std::vector<ABit> cptA;                       // adaptive copy of the trained class CPT
    std::vector<u8> rate_cls;                     // per class: class in the (rate k/n) space used by the trained tables
    // rung 3: head features (positions 0..H3-1) coded per feature with adaptive bits per (row, feature);
    // rows 0..K*T-1 = (cluster, tier) for clustered columns, K*T + (b>>2) for the others
    bool latent = false; u32 H3 = 0; int K3 = 0, T3 = 4, nrow_h = 0;
    std::vector<u8> is_head; std::vector<ABit> hb; std::vector<ABit> ktree;

    Engine(RC& rc_, const Params& P_, const Model* mod_, u32 m_, u32 n_, Cost* c) : rc(rc_), P(P_), mod(mod_), m(m_), n(n_), cost(c) {}

    inline int cb(double* acc, u32 p1, int bit) {
        int r = rc.codebit(p1, bit);
        if constexpr (ENC) { if (acc) *acc += -std::log2(r ? p1 / 65536.0 : 1.0 - p1 / 65536.0); }
        return r;
    }

    // ---- class layout
    void build_layout(const std::vector<u8>& cl, int nc) {
        NC = nc; cls = cl;
        is_head.assign(m, 0);
        latent = P.latent && mod && mod->has_r3;
        if (P.latent && !latent) throw std::runtime_error("c_r3 needs a model with a track_c_r3_<tag>.bin section");
        if (latent) { H3 = mod->H3; K3 = (int)mod->K3; T3 = (int)mod->T3; for (u32 j = 0; j < H3; ++j) is_head[mod->head[j]] = 1; }
        runlen.assign(NC, 0); runstart.assign(NC + 1, 0);
        for (u32 f = 0; f < m; ++f) if (!is_head[f] && (P.file_classes ? cls[f] != 0 : true)) runlen[cls[f]]++;
        if (P.file_classes) runlen[0] = 0;
        u32 acc = H3;
        for (int c = 0; c < NC; ++c) { runstart[c] = acc; acc += runlen[c]; }
        runstart[NC] = acc;
        feature_of_pos.assign(acc, 0); pos_of_feature.assign(m, 0xFFFFFFFFu);
        for (u32 j = 0; j < H3; ++j) { feature_of_pos[j] = mod->head[j]; pos_of_feature[mod->head[j]] = j; }
        std::vector<u32> fill(NC, 0);
        for (u32 f = 0; f < m; ++f) {
            if (is_head[f]) continue;
            int c = cls[f]; if (P.file_classes && c == 0) continue;
            u32 pos = runstart[c] + fill[c]++;
            feature_of_pos[pos] = f; pos_of_feature[f] = pos;
        }
    }
    // rung 3 tables: must run after init_tables() (uses p0q for the un-clustered rows)
    void init_head() {
        if (!latent) return;
        nrow_h = K3 * T3 + R3_MIN_BUCKET;       // + one row per un-clustered bucket (0..16)
        hb.assign((size_t)nrow_h * H3, ABit());
        for (int r = 0; r < nrow_h; ++r)
            for (u32 j = 0; j < H3; ++j) {
                ABit& a = hb[(size_t)r * H3 + j];
                u64 p48;
                if (r < K3 * T3) { p48 = p48_from_code((u32)mod->R3[(size_t)r * H3 + j] * 128); a.n = (u8)g_r3n; }
                else { int bb = std::max(r - K3 * T3, 1); p48 = p0q[(size_t)bb * NC + cls[mod->head[j]]]; a.n = 4; }
                u32 p16 = (u32)(p48 >> 32);
                a.p = (u16)std::min<u32>(std::max<u32>(p16, 48), 65535 - 48);
            }
        ktree.assign((size_t)T3 * 16, ABit());
    }
    // the r3 prior weight is part of the stream (byte after cpt_n) so encoder and decoder agree
    inline int code_cluster(int t, int cid, double* acc) {
        ABit* tr = &ktree[(size_t)t * 16];
        int node = 1;
        for (int i = 3; i >= 0; --i) {
            int bit = cb(acc, clamp16(tr[node].P()), ENC ? ((cid >> i) & 1) : 0);
            tr[node].update(bit, 250);
            node = node * 2 + bit;
        }
        return node - 16;
    }

    // ---- presence tables
    static u64 krep(int c) {                      // representative count of absolute class c
        if (c <= 8) return (u64)c;
        u64 p = p48_from_code((u32)((c - 8) * 512));
        u64 k = (u64)((((u128)8) << 48) / p);
        return k < 1 ? 1 : k;
    }
    void init_tables() {
        rate_cls.assign(NC, 0);
        for (int c = 0; c < NC; ++c) {
            if (!P.file_classes) { rate_cls[c] = (u8)c; continue; }
            if (c == 0) continue;
            u64 k = krep(c); if (k > n) k = n;
            u64 y = (u64)((((u128)k) << 48) / n);
            u64 h = neg_ln_q48(y);
            u64 r = (h * 4 + LN2Q / 2) / LN2Q;
            rate_cls[c] = (u8)(1 + (r > 119 ? 119 : r));
        }
        H.assign((size_t)NB1 * NC, 0); ES.assign((size_t)NB1 * NC, 0); G.assign((size_t)NB1 * NC, 1);
        Acnt.assign((size_t)NB1 * NC, 0); Ncols.assign(NB1, 0);
        p0q.assign((size_t)NB1 * NC, 0); inv0.assign((size_t)NB1 * NC, 0);
        for (int b = 1; b < NB1; ++b)
            for (int c = 0; c < NC; ++c) {
                u64 p0;
                if (mod) {
                    u16 code = P.file_classes ? mod->p0_r1b[(size_t)b * NC2 + rate_cls[c]] : mod->p0_r1[(size_t)b * mod->nc1 + c];
                    p0 = p48_from_code(code);
                } else {   // model-free prior: presence ~ class rate k/n
                    u64 k = krep(c < 1 ? 1 : c); if (k > n) k = n;
                    p0 = (u64)((((u128)k) << 48) / n);
                }
                if (p0 < (1ull << 16)) p0 = 1ull << 16;
                if (p0 > (1ull << 48) - (1ull << 32)) p0 = (1ull << 48) - (1ull << 32);
                p0q[(size_t)b * NC + c] = p0;
                inv0[(size_t)b * NC + c] = (u64)(((u128)P.kappa_e << 48) / p0);
            }
        for (int b = 1; b < NB1; ++b) recompute_row(b);
    }
    void recompute_row(int b) {
        for (int c = 0; c < NC; ++c) {
            if (!runlen[c]) continue;
            size_t i = (size_t)b * NC + c;
            u64 p;
            if (P.adapt) {
                u64 M = Ncols[b] * (u64)runlen[c];
                u128 num = ((u128)(Acnt[i] + (u64)P.kappa_e)) << 48;
                u128 den = (u128)M + inv0[i];
                p = (u64)(num / den);
            } else p = p0q[i];
            if (p < (1ull << 16)) p = 1ull << 16;
            if (p > (1ull << 48) - (1ull << 32)) p = (1ull << 48) - (1ull << 32);
            u64 h = hazard_from_p48(p); if (h == 0) h = 1;
            H[i] = h;
            u64 g = LN2Q / h; if (g < 1) g = 1; if (g > (1u << 30)) g = 1u << 30;
            G[i] = (u32)g;
            ES[i] = E40(h * g);
        }
    }

    // ---- value models
    void init_values() {
        nbg = mod ? (int)mod->nbg : 9;
        ncg = mod ? (int)(P.file_classes ? mod->ncg2 : mod->ncg1) : 31;
        vtab.assign((size_t)ncg * nbg * NLEV, ABit());
        for (size_t i = 0; i < vtab.size(); ++i) {
            if (mod) { vtab[i].p = (P.file_classes ? mod->v0_r1b : mod->v0_r1)[i]; vtab[i].n = 6; }
            else { vtab[i].p = 14000; vtab[i].n = 0; }
        }
        eg.assign(40, ABit());
        if (mod && !mod->feat_indep) {
            cptA.assign((size_t)NCX * NST * 128, ABit());
            for (size_t r = 0; r < (size_t)NCX * NST; ++r)
                for (int nd = 1; nd < 128; ++nd) { cptA[r * 128 + nd].p = (u16)std::min<u32>(std::max<u32>(mod->cpt[r * 127 + nd - 1], 48), 65535 - 48); cptA[r * 128 + nd].n = (u8)std::min(P.cpt_n, 250); }
        }
        ctree.assign((size_t)NST * 128, ABit());
    }
    inline int code_value(int ctxv, u32 v, double* acc) {
        ABit* t = &vtab[(size_t)ctxv * NLEV];
        u32 out = 1;
        for (int l = 0; l < NLEV; ++l) {
            int bit = cb(acc, clamp16(t[l].P()), ENC ? (v > (u32)(l + 1)) : 0);
            t[l].update(bit, g_vlimit);
            if (!bit) return (int)out;
            out++;
        }
        // v >= 8: Exp-Golomb of e = v - 8
        u32 e1 = ENC ? (v - 8 + 1) : 0;
        int nbits = 0;
        if constexpr (ENC) { while ((e1 >> (nbits + 1)) != 0) nbits++; }
        int k = 0;
        while (true) {
            int cont = cb(acc, clamp16(eg[std::min(k, 39)].P()), ENC ? (k < nbits) : 0);
            eg[std::min(k, 39)].update(cont, 30);
            if (!cont) break;
            k++;
        }
        u32 val = 1;
        for (int i = k - 1; i >= 0; --i) {
            int bit = cb(acc, 32768, ENC ? ((e1 >> i) & 1) : 0);
            val = (val << 1) | (u32)bit;
        }
        return (int)(val - 1 + 8);
    }

    // ---- column bucket symbol: 6-bit adaptive tree with the previous column's bucket group as context
    // (columns are stored in roughly depth-sorted order: adjacent-column nnz correlation ~0.98).
    int prev_b = 16;
    std::vector<ABit> btree2;
    inline int code_bucket(int b, double* acc) {
        if (btree2.empty()) btree2.assign(16 * 64, ABit());
        ABit* t = &btree2[(size_t)(prev_b >> 2) * 64];
        int node = 1;
        for (int i = 5; i >= 0; --i) {
            int bit = cb(acc, clamp16(t[node].P()), ENC ? ((b >> i) & 1) : 0);
            t[node].update(bit, g_blimit);
            node = node * 2 + bit;
        }
        prev_b = node - 64;
        return prev_b;
    }

    // ---- one column. pos_in/val_in sorted by position (encoder). Output positions/values in position order.
    // returns number of entries. out arrays are appended (decoder).
    template <class VecU32>
    u32 code_column(int b, const u32* pos_in, const u32* val_in, u32 nin, VecU32& pos_out, VecU32& val_out, int cid_in = 0) {
        u32 qi = 0, cnt = 0;
        const u64* Hb = &H[(size_t)b * NC]; const u32* Gb = &G[(size_t)b * NC]; const u64* ESb = &ES[(size_t)b * NC];
        static thread_local std::vector<u32> runcnt;
        if (P.adapt) runcnt.assign(NC, 0);
        double* accp = (ENC && cost) ? &cost->pres : nullptr;
        double* accv = (ENC && cost) ? &cost->val : nullptr;
        int bg = b >> 2;
        if (latent) {
            int row;
            if (b >= R3_MIN_BUCKET) { int t = tier_of(b, T3); int cid = code_cluster(t, cid_in, (ENC && cost) ? &cost->side : nullptr); row = cid * T3 + t; }
            else row = K3 * T3 + b;
            ABit* hp = &hb[(size_t)row * H3];
            for (u32 j = 0; j < H3; ++j) {
                u32 f = mod->head[j];
                if (cls[f] == 0) continue;                           // never present in this file (file class 0): free
                int bit = cb(accp, clamp16(hp[j].P()), ENC ? (qi < nin && pos_in[qi] == j) : 0);
                hp[j].update(bit, g_vlimit);
                if (!bit) continue;
                int cg = rate_cls[cls[f]] >> 2; if (cg >= ncg) cg = ncg - 1;
                u32 v = code_value(cg * nbg + bg, ENC ? val_in[qi] : 0, accv);
                if constexpr (!ENC) { pos_out.push_back(j); val_out.push_back(v); }
                qi++; cnt++;
            }
        }
        for (int c = 0; c < NC; ++c) {
            u32 L = runlen[c]; if (!L) continue;
            u32 start = runstart[c];
            u64 h = Hb[c]; u32 stride = Gb[c];
            u32 s = 0;
            int cg = (P.file_classes ? rate_cls[c] : c) >> 2; if (cg >= ncg) cg = ncg - 1;
            int ctxv = cg * nbg + bg;
            while (s < L) {
                u32 R = L - s;
                int any_e = ENC ? (qi < nin && pos_in[qi] < start + L) : 0;
                u64 eFull = E40(h * (u64)R);
                int any = cb(accp, clamp16(eFull >> 24), any_e);
                if (!any) break;
                u32 target = ENC ? (pos_in[qi] - (start + s)) : 0;
                u32 lo = 0, hi = R;
                u64 den = eFull;
                while (hi - lo > 1) {
                    u32 span = hi - lo, mid; u64 num;
                    if (span > 2 * stride) { mid = lo + stride; num = ESb[c]; }
                    else { mid = lo + span / 2; num = E40(h * (u64)(mid - lo)); }
                    u32 p = clamp16((num << 16) / den);
                    int less = cb(accp, p, ENC ? (target < mid) : 0);
                    if (less) { hi = mid; den = num; }
                    else { lo = mid; if (hi - lo > 1) den = E40(h * (u64)(hi - lo)); }
                }
                u32 pos = start + s + lo;
                u32 v = code_value(ctxv, ENC ? val_in[qi] : 0, accv);
                if constexpr (!ENC) { pos_out.push_back(pos); val_out.push_back(v); }
                if (P.adapt) runcnt[c]++;
                s += lo + 1; qi++; cnt++;
            }
        }
        if (P.adapt) {
            for (int c = 0; c < NC; ++c) Acnt[(size_t)b * NC + c] += runcnt[c];
            Ncols[b]++;
            if (Ncols[b] <= 64 || (Ncols[b] & (u64)(g_refresh - 1)) == 0) recompute_row(b);
        }
        return cnt;
    }

    // ---- file-class side info (symbol per feature) with trained CPT (or adaptive tree)
    void code_classes(std::vector<u8>& c, int sh) {
        double* acc = (ENC && cost) ? &cost->side : nullptr;
        bool any_used = false;
        for (u32 f = 0; f < m; ++f) {
            int st;
            bool first = (mod && !mod->feat_indep) ? mod->gfirst[f] : (f == 0);
            if (first) { st = 0; any_used = false; } else st = any_used ? 2 : 1;
            int sym = ENC ? c[f] : 0;
            int node = 1;
            if (mod && !mod->feat_indep) {
                int pl = mod->plevel[f], cx;
                if (pl == 0) cx = 0;
                else if (pl <= 40) { int g = pl - 1 - (sh >> 1) + 18; cx = 1 + std::min(std::max(g, 0), 39); }
                else { int lv = pl - 41 - sh + 36; cx = 41 + std::min(std::max(lv, 0), 63); }
                ABit* row = &cptA[((size_t)cx * NST + st) * 128];
                for (int i = NSYM_BITS - 1; i >= 0; --i) {
                    int bit = cb(acc, clamp16(row[node].P()), ENC ? ((sym >> i) & 1) : 0);
                    if (P.cpt_n > 0) row[node].update(bit, 250);
                    node = node * 2 + bit;
                }
            } else {
                ABit* row = &ctree[(size_t)st * 128];
                for (int i = NSYM_BITS - 1; i >= 0; --i) {
                    int bit = cb(acc, clamp16(row[node].P()), ENC ? ((sym >> i) & 1) : 0);
                    row[node].update(bit, 30);
                    node = node * 2 + bit;
                }
            }
            int s = node - 128;
            if constexpr (!ENC) c[f] = (u8)s;
            if (s) any_used = true;
        }
    }
};

// ------------------------------------------------------------------ top-level encode/decode
inline std::vector<u8> tc_encode(const CanonicalMatrix& M, const Params& P0) {
    Params P = P0; if (g_kappa > 0) P.kappa_e = g_kappa; if (g_cptn >= 0) P.cpt_n = g_cptn;
    Timer tm;
    std::string tag = P.use_model ? find_model_tag(M, P.insample ? "none" : "") : "";
    std::shared_ptr<Model> mod = get_model(tag);
    if (!mod) tag.clear();
    if (!mod && !P.file_classes) throw std::runtime_error("profile-class codec needs a model");
    u32 m = M.m, n = M.n;
    std::vector<u8> out;
    out.push_back('T'); out.push_back('C'); out.push_back('z'); out.push_back('1');
    out.push_back((u8)((P.file_classes ? 1 : 0) | (P.adapt ? 2 : 0) | (mod ? 4 : 0) | (P.latent ? 8 : 0)));
    out.push_back((u8)P.kappa_e); out.push_back((u8)P.cpt_n); out.push_back((u8)g_r3n);
    put_varint(out, m); put_varint(out, n); put_varint(out, M.nnz); out.push_back(M.vt_code);
    put_varint(out, tag.size()); out.insert(out.end(), tag.begin(), tag.end());
    int sh = (int)std::lround(2.0 * std::log2(std::max(1.0, (double)M.nnz)));
    out.push_back((u8)(int8_t)sh);
    // names / kv streams (zstd-19, as in ref_zstd_csc)
    size_t before_names = out.size();
    {
        using namespace detail;
        auto rn = join_names(M.rownames), cn = join_names(M.colnames);
        put_stream(out, rn.data(), rn.size(), g_names_level);
        put_stream(out, cn.data(), cn.size(), g_names_level);
        std::vector<u8> kv;
        for (auto& [k, v] : M.user_kv) {
            kv.insert(kv.end(), k.begin(), k.end()); kv.push_back(0);
            kv.insert(kv.end(), v.begin(), v.end()); kv.push_back(0);
        }
        put_u32(out, (u32)M.user_kv.size());
        put_stream(out, kv.data(), kv.size(), g_names_level);
    }
    size_t names_bytes = out.size() - before_names;

    // classes
    std::vector<u8> cl(m);
    int nc;
    if (P.file_classes) {
        std::vector<u32> k(m, 0);
        for (u64 i = 0; i < M.nnz; ++i) k[M.indices[i]]++;
        for (u32 f = 0; f < m; ++f) {
            if (!k[f]) { cl[f] = 0; continue; }
            int c;
            if (g_coarse == 0) {
                if (k[f] <= 8) c = (int)k[f];
                else c = 8 + (int)std::floor(4.0 * std::log2((double)k[f] / 8.0) + 0.5);
            } else if (g_coarse == 1) {          // merge small counts, 0.5-octave bins above 8
                if (k[f] <= 2) c = (int)k[f]; else if (k[f] <= 4) c = 3; else if (k[f] <= 8) c = 6;
                else c = 8 + 2 * (int)std::floor(2.0 * std::log2((double)k[f] / 8.0) + 0.5);
            } else {                              // octave-ish: 1,2,4,8 then 1-octave bins
                if (k[f] == 1) c = 1; else if (k[f] <= 3) c = 2; else if (k[f] <= 7) c = 4; else if (k[f] <= 15) c = 8;
                else c = 8 + 4 * (int)std::floor(std::log2((double)k[f] / 11.3) + 0.5);
            }
            cl[f] = (u8)std::min(c, NC2 - 1);
        }
        nc = NC2;
    } else {
        cl = mod->cls1; nc = (int)mod->nc1;
    }

    Cost cost;
    RcEnc rc;
    Engine<RcEnc, true> eng(rc, P, mod.get(), m, n, g_stats ? &cost : nullptr);
    eng.init_values();
    if (P.file_classes) eng.code_classes(cl, sh);
    eng.build_layout(cl, nc);
    eng.init_tables();
    eng.init_head();
    // rung 3, encoder only (float is fine here: the cluster id is transmitted): per-(cluster, tier) log-likelihood ratio of
    // the head presence pattern under the trained rates; the column is assigned to the max-likelihood cluster.
    std::vector<float> LR, S0;
    if (eng.latent) {
        size_t KT = (size_t)eng.K3 * eng.T3;
        LR.assign(KT * eng.H3, 0.f); S0.assign(KT, 0.f);
        for (size_t r = 0; r < KT; ++r)
            for (u32 j = 0; j < eng.H3; ++j) {
                double p = std::pow(2.0, -(double)mod->R3[r * eng.H3 + j] / 16.0); p = std::min(std::max(p, 1e-5), 1 - 1e-5);
                LR[r * eng.H3 + j] = (float)(std::log2(p) - std::log2(1 - p)); S0[r] += (float)std::log2(1 - p);
            }
    }
    std::vector<std::pair<u32, u32>> tmp;
    std::vector<u32> pin, vin; std::vector<u32> dummy; std::vector<float> ll;
    g_col_bits.assign(n, 0.f);
    for (u32 j = 0; j < n; ++j) {
        u32 s = M.indptr[j], e = M.indptr[j + 1], len = e - s;
        int b = bucket_of(len);
        eng.code_bucket(b, (g_stats ? &cost.bucket : nullptr));
        if (len == 0) continue;
        tmp.resize(len);
        for (u32 i = 0; i < len; ++i) tmp[i] = {eng.pos_of_feature[M.indices[s + i]], M.data[s + i]};
        std::sort(tmp.begin(), tmp.end());
        pin.resize(len); vin.resize(len);
        for (u32 i = 0; i < len; ++i) { pin[i] = tmp[i].first; vin[i] = tmp[i].second; }
        int cid = 0;
        if (eng.latent && b >= R3_MIN_BUCKET) {
            int t = tier_of(b, eng.T3); ll.assign(eng.K3, 0.f);
            for (int k = 0; k < eng.K3; ++k) {
                size_t r = (size_t)k * eng.T3 + t; float acc = S0[r];
                for (u32 i = 0; i < len && pin[i] < eng.H3; ++i) acc += LR[r * eng.H3 + pin[i]];
                ll[k] = acc;
            }
            cid = (int)(std::max_element(ll.begin(), ll.end()) - ll.begin());
        }
        double p0_ = cost.pres, v0_ = cost.val;
        eng.code_column(b, pin.data(), vin.data(), len, dummy, dummy, cid);
        g_col_bits[j] = (float)((cost.pres - p0_) + (cost.val - v0_)); cost.pb[b] += cost.pres - p0_; cost.vb[b] += cost.val - v0_; cost.nb[b] += len;
    }
    rc.finish();
    put_varint(out, rc.out.size());
    out.insert(out.end(), rc.out.begin(), rc.out.end());
    g_report = Report();
    g_report.side_bits = cost.side; g_report.bucket_bits = cost.bucket; g_report.pres_bits = cost.pres; g_report.val_bits = cost.val;
    g_report.rc_bytes = rc.out.size(); g_report.names_bytes = names_bytes; g_report.total_bytes = out.size();
    g_report.model_bytes = mod ? mod->file_bytes + (P.latent ? mod->r3_bytes : 0) : 0; g_report.nnz = M.nnz; g_report.tag = tag;
    g_report.mode = out[4];
    for (int i = 0; i < 40; ++i) { g_report.pb[i] = cost.pb[i]; g_report.vb[i] = cost.vb[i]; g_report.nb[i] = cost.nb[i]; }
    return out;
}

inline CanonicalMatrix tc_decode(const std::vector<u8>& b, Params P) {
    const u8* p = b.data();
    if (b.size() < 8 || p[0] != 'T' || p[1] != 'C' || p[2] != 'z' || p[3] != '1') throw std::runtime_error("bad TCz1 stream");
    p += 4;
    u8 mode = *p++; P.kappa_e = *p++; P.cpt_n = *p++; g_r3n = *p++;
    P.file_classes = mode & 1; P.adapt = (mode >> 1) & 1; P.latent = (mode >> 3) & 1;
    CanonicalMatrix M;
    M.m = (u32)get_varint(p); M.n = (u32)get_varint(p); M.nnz = get_varint(p); M.vt_code = *p++;
    u64 tl = get_varint(p); std::string tag((const char*)p, tl); p += tl;
    int sh = (int8_t)*p++;
    std::shared_ptr<Model> mod;
    if (mode & 4) { mod = get_model(tag); if (!mod) throw std::runtime_error("model not found: " + tag); }
    {
        using namespace detail;
        M.rownames = split_names(get_stream(p), M.m);
        M.colnames = split_names(get_stream(p), M.n);
        u32 nkv = get_u32(p);
        auto kv = get_stream(p);
        auto parts = split_names(kv, 2 * (size_t)nkv);
        for (u32 i = 0; i < nkv; ++i) M.user_kv[parts[2 * i]] = parts[2 * i + 1];
    }
    u64 rl = get_varint(p);
    RcDec rc(p, p + rl);
    u32 m = M.m, n = M.n;
    Engine<RcDec, false> eng(rc, P, mod.get(), m, n, nullptr);
    std::vector<u8> cl(m, 0); int nc;
    eng.init_values();
    if (P.file_classes) { eng.code_classes(cl, sh); nc = NC2; }
    else { cl = mod->cls1; nc = (int)mod->nc1; }
    eng.build_layout(cl, nc);
    eng.init_tables();
    eng.init_head();
    M.indptr.assign(n + 1, 0);
    M.indices.reserve(M.nnz); M.data.reserve(M.nnz);
    std::vector<u32> pos, val; std::vector<std::pair<u32, u32>> tmp;
    for (u32 j = 0; j < n; ++j) {
        int bk = eng.code_bucket(0, nullptr);
        u32 cnt = 0;
        if (bk != 0) {
            pos.clear(); val.clear();
            cnt = eng.code_column(bk, nullptr, nullptr, 0, pos, val);
            tmp.resize(cnt);
            for (u32 i = 0; i < cnt; ++i) tmp[i] = {eng.feature_of_pos[pos[i]], val[i]};
            std::sort(tmp.begin(), tmp.end());
            for (u32 i = 0; i < cnt; ++i) { M.indices.push_back(tmp[i].first); M.data.push_back(tmp[i].second); }
        }
        M.indptr[j + 1] = M.indptr[j] + cnt;
    }
    return M;
}

inline Codec make_tc_codec(const Params& P) {
    Codec c;
    c.name = P.name;
    c.encode = [P](const CanonicalMatrix& M) { return tc_encode(M, P); };
    c.decode = [P](const std::vector<u8>& b) { return tc_decode(b, P); };
    // model_bytes: largest shipped model (human exon is the largest)
    size_t mb = 0, r3 = 0;
    std::error_code ec;
    for (auto& e : std::filesystem::directory_iterator(g_model_dir, ec)) {
        std::string fn = e.path().filename().string();
        if (fn.rfind("track_c_model_", 0) == 0 && fn.find("_none.bin") != std::string::npos) mb = std::max<size_t>(mb, (size_t)e.file_size());
        if (P.latent && fn.rfind("track_c_r3_", 0) == 0 && fn.find("_none.bin") != std::string::npos) r3 = std::max<size_t>(r3, (size_t)e.file_size());
    }
    c.model_bytes = P.use_model ? mb + r3 : 0;
    return c;
}

}}  // namespace cs::tc
