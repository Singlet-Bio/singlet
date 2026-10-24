// Track A deliverable 2: what do the names cost under each coder?
//
//   track_a_names --out results/track_a_names.tsv file.1pz ...
//
// Row names are identical for all samples of a (reference build, matrix kind), so each distinct list is measured
// once (keyed by hash). Barcode lists are measured per file.
#include "common.h"
#include "track_a.h"
#include <cmath>
#include <iostream>
#include <set>

using namespace cs;
using namespace cs::ta;

static std::string base_name(const std::string& p) { auto s = p.find_last_of("/\\"); return s == std::string::npos ? p : p.substr(s + 1); }
static std::string parent_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    if (s == std::string::npos || s == 0) return "";
    auto t = p.find_last_of("/\\", s - 1);
    return p.substr(t == std::string::npos ? 0 : t + 1, s - (t == std::string::npos ? 0 : t + 1));
}

struct Row { std::string sample, member, field, scheme; size_t bytes; size_t raw; double enc_ms, dec_ms; size_t model; };

int main(int argc, char** argv) {
    std::string out = "results/track_a_names.tsv";
    std::vector<std::string> files;
    bool cols_only = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out = argv[++i]; else if (a == "--cols-only") cols_only = true; else files.push_back(a);
    }
    std::vector<Row> rows;
    std::map<uint64_t, std::vector<Row>> row_cache;   // rownames results by hash
    std::vector<EC> coders = {ec_zstd(3), ec_zstd(19), ec_zstd(22, 27, true), ec_xz(), ec_brotli(11, 24), ec_bzip2()};
    auto& md = Models::get();

    auto measure = [&](const std::string& sample, const std::string& member, const std::string& field,
                       const std::vector<uint8_t>& raw, std::vector<Row>& dst) {
        for (auto& ec : coders) {
            Timer t;
            auto z = z_compress(ec, raw);
            double e = t.ms();
            std::vector<uint8_t> back(raw.size());
            double dmin = 1e300;
            for (int r = 0; r < 3; ++r) { Timer td; ec_decompress(ec, z.data(), z.size(), back.data(), back.size()); dmin = std::min(dmin, td.ms()); }
            if (back != raw) { std::cerr << "ROUNDTRIP FAIL " << ec.label << "\n"; return 1; }
            dst.push_back({sample, member, field, ec.label, z.size(), raw.size(), e, dmin, 0});
        }
        return 0;
    };

    for (auto& f : files) {
        auto M = load_1pz(f);
        std::string sample = parent_name(f), member = base_name(f);
        std::printf("== %s/%s  m=%u n=%u\n", sample.c_str(), member.c_str(), M.m, M.n);
        // ---- rownames (once per distinct list) ----
        uint64_t h = names_hash(M.rownames);
        if (!cols_only && !row_cache.count(h)) {
            std::vector<Row> rr;
            auto raw = join_nul(M.rownames);
            if (measure(sample, member, "rownames", raw, rr)) return 1;
            // production metadata block as written today (rownames + colnames TLV, zstd-3)
            auto prod = enc_names_inline_tlv(M);
            rr.push_back({sample, member, "meta_block_today(rows+cols)", "zstd3", prod.size(), raw.size(), 0, 0, 0});
            // vocabulary id only
            auto it = md.vocab.find(h);
            if (it != md.vocab.end())
                rr.push_back({sample, member, "rownames", "vocab_id(8B)+model", 8, raw.size(), 0, 0, size_t(it->second.z_len)});
            row_cache[h] = rr;
        }
        if (!cols_only) for (auto r : row_cache[h]) { r.sample = sample; r.member = member; rows.push_back(r); }
        // ---- colnames ----
        {
            std::vector<Row> cr;
            auto raw = join_nul(M.colnames);
            if (measure(sample, member, "colnames", raw, cr)) return 1;
            for (auto& r : cr) rows.push_back(r);
            std::vector<uint8_t> t; std::string wn;
            if (bc_pack2(M.colnames, t)) {
                rows.push_back({sample, member, "colnames", "pack2bit_file_order", t.size(), raw.size(), 0, 0, 0});
                rows.push_back({sample, member, "colnames", "pack2bit+zstd19", z_compress(ec_zstd(19), t).size(), raw.size(), 0, 0, 0});
            }
            if (bc_whitelist(M.colnames, t, wn)) {
                for (auto& w : md.wls) if (w.name == wn)
                    rows.push_back({sample, member, "colnames", "whitelist_index(" + wn + ")", t.size(), raw.size(), 0, 0, size_t(w.file_bytes)});
                size_t W = 0; for (auto& w : md.wls) if (w.name == wn) W = w.count;
                rows.push_back({sample, member, "colnames", "whitelist_entropy_bound", size_t(std::ceil(M.colnames.size() * std::log2(double(W)) / 8)), raw.size(), 0, 0, 0});
                // canonical (sorted) column order would only need the subset: log2 C(W, n)
                double n = double(M.colnames.size()), Wd = double(W);
                double lg = std::lgamma(Wd + 1) - std::lgamma(n + 1) - std::lgamma(Wd - n + 1);
                rows.push_back({sample, member, "colnames", "sorted_set_bound(order canonicalised)", size_t(std::ceil(lg / std::log(2.0) / 8)), raw.size(), 0, 0, 0});
            }
            if (bc_sorted_delta(M.colnames, t)) {
                rows.push_back({sample, member, "colnames", "sorted_rice_delta+order_perm", t.size(), raw.size(), 0, 0, 0});
                // components: the Rice stream alone (what a canonicalised column order would cost)
                ByteR q(t.data(), t.size());
                (void)q.u32(); (void)q.u8(); (void)q.u32(); uint32_t rl = q.u32();
                rows.push_back({sample, member, "colnames", "  of which sorted_rice_delta stream", size_t(rl), raw.size(), 0, 0, 0});
                size_t pml = t.size() - 13 - rl;
                rows.push_back({sample, member, "colnames", "  of which order permutation", pml, raw.size(), 0, 0, 0});
            }
        }
    }
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf) tsv << "sample\tmember\tfield\tscheme\tbytes\traw_bytes\tenc_ms\tdec_ms_min\tmodel_bytes\n";
    for (auto& r : rows)
        tsv << r.sample << '\t' << r.member << '\t' << r.field << '\t' << r.scheme << '\t' << r.bytes << '\t' << r.raw << '\t' << r.enc_ms << '\t' << r.dec_ms << '\t' << r.model << '\n';
    for (auto& r : rows)
        std::printf("%-22s %-18s %-28s %-42s %10zu B  (raw %zu)  enc %.0f ms dec %.1f ms model %zu\n", r.sample.c_str(), r.member.c_str(), r.field.c_str(), r.scheme.c_str(), r.bytes, r.raw, r.enc_ms, r.dec_ms, r.model);
    return 0;
}
