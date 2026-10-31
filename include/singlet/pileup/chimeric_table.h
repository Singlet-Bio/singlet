// SPDX-License-Identifier: MIT
// chimeric_table.h - per-cell chimeric / fusion junction table from STAR's chimeric junction file.
//
// With `--chimeric`, STAR runs its chimeric detection (Junctions output, multimapping format 1) beside the normal alignment; the junction
// lines carry the read's raw cell barcode and UMI as their last two columns (src/star: the barcode is parsed before detection). This
// post-pass turns them into `chimeric_junctions.tsv.gz`:
//   - the raw barcode is matched to the sample's barcodes (exact, else the unique one-mismatch neighbour; anything else is dropped and counted);
//   - lines are collapsed to molecules per (cell, junction): reads are lines, UMIs are distinct UMI strings (exact equality);
//   - a read with several chimeric alignments (num_chim_aln > 1) contributes only its best-scoring ones and is flagged `multimap`;
//   - donor and acceptor are named from the gene annotation (gene on the junction strand preferred, smallest span) and the pair is looked up in
//     the fusion catalogue (scripts/build_fusion_catalog.py: Mitelman, ChimerDB and CTAT sources; never COSMIC).
// Host-side by design: fusion reads are rare (hundreds to thousands per sample) and STAR's chimeric detection disables the sequence cache, so
// the run is a CPU run on both engines; both engines read the same file through this code, so the table is identical between them.
#pragma once

#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace singlet {
namespace chimeric {

struct Config {
    std::string junction_path;  ///< STAR's Chimeric.out.junction (format 1, with CB/UMI as the last two columns)
    std::string gtf_path;       ///< gene annotation (gene features); empty = no gene names
    std::string catalog_path;   ///< fusion catalogue (.tsv or .tsv.gz); empty = no catalogue annotation
    std::string out_path;       ///< chimeric_junctions.tsv.gz
};

struct Stats {
    uint64_t lines = 0;          ///< junction lines read
    uint64_t kept_lines = 0;     ///< lines of best-scoring alignments with a matched barcode
    uint64_t no_barcode = 0;     ///< lines without a usable raw barcode or UMI
    uint64_t unmatched = 0;      ///< lines whose barcode is in none of the sample's barcodes
    uint64_t ambiguous = 0;      ///< lines whose barcode has several one-mismatch neighbours
    uint64_t rows = 0;           ///< rows written (cell x junction)
    uint64_t cells = 0;          ///< cells with at least one row
    uint64_t annotated = 0;      ///< rows whose gene pair is in the catalogue
    bool ok = false;
};

namespace detail {

inline std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t a = 0;
    for (;;) {
        size_t b = s.find(sep, a);
        if (b == std::string::npos) { out.push_back(s.substr(a)); break; }
        out.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return out;
}

/// Lines of a plain or gzip file.
template <class Fn>
inline bool for_each_line(const std::string& path, Fn&& fn) {
    gzFile f = gzopen(path.c_str(), "rb");  // reads plain files too
    if (!f) return false;
    std::string line;
    std::vector<char> buf(1 << 16);
    for (;;) {
        char* r = gzgets(f, buf.data(), (int)buf.size());
        if (!r) break;
        line += r;
        if (!line.empty() && line.back() == '\n') {
            line.pop_back();
            if (!line.empty() && line.back() == '\r') line.pop_back();
            fn(line);
            line.clear();
        }
    }
    if (!line.empty()) fn(line);
    gzclose(f);
    return true;
}

struct Gene {
    int64_t start = 0, end = 0;  // 1-based inclusive
    char strand = '+';
    std::string name;
};

using GeneIndex = std::unordered_map<std::string, std::vector<Gene>>;

inline std::string gtf_attr(const std::string& attrs, const char* key) {
    std::string k = std::string(key) + " \"";
    size_t p = attrs.find(k);
    if (p == std::string::npos) return "";
    p += k.size();
    size_t q = attrs.find('"', p);
    return q == std::string::npos ? "" : attrs.substr(p, q - p);
}

inline GeneIndex load_genes(const std::string& gtf) {
    GeneIndex idx;
    if (gtf.empty()) return idx;
    for_each_line(gtf, [&](const std::string& ln) {
        if (ln.empty() || ln[0] == '#') return;
        auto f = split(ln, '\t');
        if (f.size() < 9 || f[2] != "gene") return;
        Gene g;
        g.start = std::stoll(f[3]);
        g.end = std::stoll(f[4]);
        g.strand = f[6].empty() ? '+' : f[6][0];
        g.name = gtf_attr(f[8], "gene_name");
        if (g.name.empty()) g.name = gtf_attr(f[8], "gene_id");
        if (!g.name.empty()) idx[f[0]].push_back(g);
    });
    return idx;
}

/// The gene at a junction side: genes containing `pos`, those on the junction strand first, then the smallest span; "" if none.
inline std::string gene_at(const GeneIndex& idx, const std::string& chr, int64_t pos, char strand) {
    auto it = idx.find(chr);
    if (it == idx.end()) return "";
    const Gene* best = nullptr;
    for (const auto& g : it->second) {
        if (pos < g.start || pos > g.end) continue;
        if (!best) { best = &g; continue; }
        const bool gs = g.strand == strand, bs = best->strand == strand;
        if (gs != bs) { if (gs) best = &g; continue; }
        const int64_t gl = g.end - g.start, bl = best->end - best->start;
        if (gl < bl || (gl == bl && g.name < best->name)) best = &g;
    }
    return best ? best->name : "";
}

/// catalogue: "A--B" -> "source1,source2|ordered" ; loaded from build_fusion_catalog.py output (gene_a, gene_b, sources, ordered)
struct CatalogHit {
    std::string sources;
    bool ordered = true;
};
using Catalog = std::unordered_map<std::string, CatalogHit>;

inline Catalog load_catalog(const std::string& path) {
    Catalog c;
    if (path.empty()) return c;
    for_each_line(path, [&](const std::string& ln) {
        if (ln.empty() || ln[0] == '#') return;
        auto f = split(ln, '\t');
        if (f.size() < 4 || f[0] == "gene_a") return;
        c[f[0] + "--" + f[1]] = CatalogHit{f[2], f[3] == "1"};
    });
    return c;
}

}  // namespace detail

/// One row per (cell, junction). Returned for tests; write_table() writes them.
struct Row {
    std::string barcode, donor_chr, acceptor_chr, donor_strand, acceptor_strand, donor_gene, acceptor_gene, sources, orientation;
    int64_t donor_pos = 0, acceptor_pos = 0;
    int junction_type = -1;
    uint64_t n_reads = 0, n_umis = 0;
    int n_chim_aln = 1, best_score = 0;
    bool multimap = false;
};

inline std::string fusion_name(const Row& r) {
    return (r.donor_gene.empty() ? std::string(".") : r.donor_gene) + "--" + (r.acceptor_gene.empty() ? std::string(".") : r.acceptor_gene);
}

/// Build the rows. `barcodes` are the sample's barcodes (the matrix columns).
inline std::vector<Row> build_rows(const std::vector<std::string>& barcodes, const Config& cfg, Stats& st) {
    std::unordered_map<std::string, uint32_t> bc_index;
    bc_index.reserve(barcodes.size() * 2);
    for (uint32_t i = 0; i < barcodes.size(); i++) bc_index.emplace(barcodes[i], i);
    const auto genes = detail::load_genes(cfg.gtf_path);
    const auto catalog = detail::load_catalog(cfg.catalog_path);

    auto resolve = [&](const std::string& raw, bool& ambiguous) -> int64_t {
        auto it = bc_index.find(raw);
        if (it != bc_index.end()) return it->second;
        static const char kBases[4] = {'A', 'C', 'G', 'T'};
        int64_t hit = -1;
        int n = 0;
        std::string t = raw;
        for (size_t i = 0; i < t.size(); i++) {
            const char orig = t[i];
            for (char b : kBases) {
                if (b == orig) continue;
                t[i] = b;
                auto j = bc_index.find(t);
                if (j != bc_index.end()) { hit = j->second; n++; }
            }
            t[i] = orig;
        }
        if (n == 1) return hit;
        ambiguous = n > 1;
        return -1;
    };

    struct Line {
        std::string chr1, chr2, read, umi;
        int64_t pos1, pos2;
        char s1, s2;
        int jtype, n_aln, score, best;
        uint32_t cell;
    };
    std::vector<Line> lines;
    detail::for_each_line(cfg.junction_path, [&](const std::string& ln) {
        if (ln.empty() || ln[0] == '#' || ln.compare(0, 10, "chr_donorA") == 0) return;
        auto f = detail::split(ln, '\t');
        st.lines++;
        if (f.size() < 22) { st.no_barcode++; return; }
        const std::string& cb = f[f.size() - 2];
        const std::string& umi = f[f.size() - 1];
        auto acgtn = [](const std::string& s) {
            return !s.empty() && s.find_first_not_of("ACGTN") == std::string::npos;
        };
        if (!acgtn(cb) || !acgtn(umi)) { st.no_barcode++; return; }
        Line l;
        l.chr1 = f[0]; l.pos1 = std::stoll(f[1]); l.s1 = f[2].empty() ? '+' : f[2][0];
        l.chr2 = f[3]; l.pos2 = std::stoll(f[4]); l.s2 = f[5].empty() ? '+' : f[5][0];
        l.jtype = std::stoi(f[6]);
        l.read = f[9]; l.umi = umi;
        l.n_aln = std::stoi(f[14]); l.score = std::stoi(f[17]); l.best = std::stoi(f[18]);
        if (l.score < l.best) return;  // a worse alignment of a multimapping chimeric read
        bool amb = false;
        const int64_t c = resolve(cb, amb);
        if (c < 0) { (amb ? st.ambiguous : st.unmatched)++; return; }
        l.cell = (uint32_t)c;
        st.kept_lines++;
        lines.push_back(std::move(l));
    });

    // collapse: (cell, junction) -> reads, distinct UMIs
    struct Agg {
        uint64_t reads = 0;
        std::set<std::string> umis;
        int n_aln = 1, best = 0;
        int jtype = -1;
    };
    using Key = std::tuple<uint32_t, std::string, int64_t, char, std::string, int64_t, char>;
    std::map<Key, Agg> agg;
    for (const auto& l : lines) {
        Agg& a = agg[Key{l.cell, l.chr1, l.pos1, l.s1, l.chr2, l.pos2, l.s2}];
        a.reads++;
        a.umis.insert(l.umi);
        a.n_aln = std::max(a.n_aln, l.n_aln);
        a.best = std::max(a.best, l.score);
        a.jtype = l.jtype;
    }
    std::vector<Row> rows;
    std::set<uint32_t> cells;
    for (const auto& kv : agg) {
        const auto& [cell, c1, p1, s1, c2, p2, s2] = kv.first;
        Row r;
        r.barcode = barcodes[cell];
        r.donor_chr = c1; r.donor_pos = p1; r.donor_strand = std::string(1, s1);
        r.acceptor_chr = c2; r.acceptor_pos = p2; r.acceptor_strand = std::string(1, s2);
        r.junction_type = kv.second.jtype;
        r.n_reads = kv.second.reads; r.n_umis = kv.second.umis.size();
        r.n_chim_aln = kv.second.n_aln; r.multimap = kv.second.n_aln > 1; r.best_score = kv.second.best;
        r.donor_gene = detail::gene_at(genes, c1, p1, s1);
        r.acceptor_gene = detail::gene_at(genes, c2, p2, s2);
        r.sources = "."; r.orientation = ".";
        if (!r.donor_gene.empty() && !r.acceptor_gene.empty()) {
            auto h = catalog.find(r.donor_gene + "--" + r.acceptor_gene);
            if (h != catalog.end()) {
                r.sources = h->second.sources;
                r.orientation = h->second.ordered ? "ordered" : "unordered";
            } else {
                auto h2 = catalog.find(r.acceptor_gene + "--" + r.donor_gene);
                if (h2 != catalog.end()) {
                    r.sources = h2->second.sources;
                    r.orientation = h2->second.ordered ? "reversed" : "unordered";
                }
            }
            if (r.sources != ".") st.annotated++;
        }
        cells.insert(cell);
        rows.push_back(std::move(r));
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        return std::tie(a.barcode, a.donor_chr, a.donor_pos, a.donor_strand, a.acceptor_chr, a.acceptor_pos, a.acceptor_strand) <
               std::tie(b.barcode, b.donor_chr, b.donor_pos, b.donor_strand, b.acceptor_chr, b.acceptor_pos, b.acceptor_strand);
    });
    st.rows = rows.size();
    st.cells = cells.size();
    return rows;
}

inline std::string header_line() {
    return "barcode\tdonor_chr\tdonor_pos\tdonor_strand\tacceptor_chr\tacceptor_pos\tacceptor_strand\tjunction_type\tdonor_gene\tacceptor_gene\t"
           "fusion\tn_reads\tn_umis\tn_chim_aln\tmultimap\tbest_score\tcatalog_sources\tcatalog_orientation\n";
}

inline std::string row_line(const Row& r) {
    std::ostringstream o;
    o << r.barcode << '\t' << r.donor_chr << '\t' << r.donor_pos << '\t' << r.donor_strand << '\t' << r.acceptor_chr << '\t' << r.acceptor_pos << '\t'
      << r.acceptor_strand << '\t' << r.junction_type << '\t' << (r.donor_gene.empty() ? "." : r.donor_gene) << '\t'
      << (r.acceptor_gene.empty() ? "." : r.acceptor_gene) << '\t' << fusion_name(r) << '\t' << r.n_reads << '\t' << r.n_umis << '\t' << r.n_chim_aln
      << '\t' << (r.multimap ? 1 : 0) << '\t' << r.best_score << '\t' << r.sources << '\t' << r.orientation << '\n';
    return o.str();
}

/// Writes cfg.out_path (gzip, level 6, no timestamp-dependent bytes) and returns the statistics.
inline Stats write_table(const std::vector<std::string>& barcodes, const Config& cfg) {
    Stats st;
    std::vector<Row> rows = build_rows(barcodes, cfg, st);
    gzFile f = gzopen(cfg.out_path.c_str(), "wb6");
    if (!f) return st;
    const std::string h = header_line();
    gzwrite(f, h.data(), (unsigned)h.size());
    for (const auto& r : rows) {
        const std::string l = row_line(r);
        gzwrite(f, l.data(), (unsigned)l.size());
    }
    st.ok = gzclose(f) == Z_OK;
    return st;
}

}  // namespace chimeric
}  // namespace singlet
