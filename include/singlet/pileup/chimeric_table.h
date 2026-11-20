// SPDX-License-Identifier: MIT
// chimeric_table.h - per-cell chimeric / fusion junction table from STAR's chimeric junction file.
//
// With `--chimeric`, STAR runs its chimeric detection (Junctions output, multimapping format 1) beside the normal alignment; the junction
// lines carry the read's raw cell barcode and UMI as their last two columns (src/star: the barcode is parsed before detection). This
// post-pass turns them into `chimeric_junctions.tsv.gz`:
//   - the raw barcode is matched to the sample's barcodes (exact, else the unique one-mismatch neighbour; anything else is dropped and counted);
//   - lines are collapsed to molecules per (cell, junction): reads are lines, UMIs are distinct UMI strings (exact equality);
//   - a read with several chimeric alignments (num_chim_aln > 1) contributes only its best-scoring ones and is flagged `multimap`;
//   - each line is put in transcript orientation: STAR reports the read's strand, so a CT/AC junction (type 2) and, on an antisense library
//     (10x 5' R2: reverse_strand), a non-canonical one are turned around (donor and acceptor swap, strands flip). Sense and antisense reads
//     of one junction then collapse to one row, the donor is the 5' partner, and genes are named on the transcript strand;
//   - the call set keeps canonical-motif junctions only (GT/AG on the transcript; non-canonical lines stay in STAR's raw junction file);
//   - junctions matching Arriba's published blacklist (MIT; Uhrig et al. 2021, Genome Res 31:448) are dropped, and gene pairs on the
//     recurrent list (scripts/build_chimeric_recurrent.py: pairs seen in unrelated non-malignant samples) are flagged in `filter`;
//   - donor and acceptor are named from the gene annotation (gene on the transcript strand preferred, smallest span) and the pair is looked up in
//     the fusion catalogue (scripts/build_fusion_catalog.py: Mitelman, ChimerDB and CTAT sources; never COSMIC).
// Host-side by design: fusion reads are rare (hundreds to thousands per sample) and STAR's chimeric detection disables the sequence cache, so
// the run is a CPU run on both engines; both engines read the same file through this code, so the table is identical between them.
#pragma once

#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace singlet {
namespace chimeric {

/// Member table of the deduplicated second pass (SINGLET_CHIM_DEDUP): the pass maps each distinct read sequence once, under the read name
/// "<id>" (0-based index of the distinct sequence in order of first appearance); this table lists, for every id, its member reads in input
/// order (the representative, the first read of the sequence, is member 0) with the cell barcode and UMI that STAR's junction line prints for
/// that read (SoloReadBarcode::getCBandUMI for CB_UMI_Simple: substrings of the barcode mate's sequence, nothing else). Bases are packed
/// 3 bits each (A 0, C 1, G 2, T 3, N 4), first base in the highest bits.
struct MemberTable {
    uint32_t cb_len = 0, umi_len = 0;       ///< bases of the cell barcode and of the UMI (each at most 21)
    std::vector<uint64_t> start;            ///< n_ids + 1 offsets into cb / umi
    std::vector<uint64_t> cb, umi;          ///< packed barcodes, members grouped by id, input order inside an id

    uint64_t n_ids() const { return start.empty() ? 0 : start.size() - 1; }
    uint64_t count(uint64_t id) const { return start[id + 1] - start[id]; }
    static int code_of(char c) {
        switch (c) { case 'A': return 0; case 'C': return 1; case 'G': return 2; case 'T': return 3; case 'N': return 4; default: return -1; }
    }
    /// packs s (all of ACGTN, at most 21 bases); false otherwise
    static bool pack(const char* s, size_t n, uint64_t& out) {
        if (n == 0 || n > 21) return false;
        uint64_t v = 0;
        for (size_t i = 0; i < n; i++) {
            const int c = code_of(s[i]);
            if (c < 0) return false;
            v = (v << 3) | (uint64_t)c;
        }
        out = v;
        return true;
    }
    static std::string unpack(uint64_t v, uint32_t n) {
        static const char kB[8] = {'A', 'C', 'G', 'T', 'N', 'N', 'N', 'N'};
        std::string s(n, 'N');
        for (uint32_t i = 0; i < n; i++) s[n - 1 - i] = kB[(v >> (3 * i)) & 7];
        return s;
    }
};

struct Config {
    const MemberTable* members = nullptr;  ///< deduplicated pass: junction lines carry the sequence id as read name (null = every read was mapped)
    std::string expanded_dump_path;        ///< with members: write every junction data line once per member (read name ".", the member's CB and UMI)
    std::string junction_path;  ///< STAR's Chimeric.out.junction (format 1, with CB/UMI as the last two columns)
    std::string gtf_path;       ///< gene annotation (gene features); empty = no gene names
    std::string catalog_path;   ///< fusion catalogue (.tsv or .tsv.gz); empty = no catalogue annotation
    std::string out_path;       ///< chimeric_junctions.tsv.gz
    std::string blacklist_path; ///< Arriba blacklist (blacklist_hg38_GRCh38_v2.5.1.tsv.gz); empty = none
    std::string recurrent_path; ///< recurrent gene pairs (gene_a, gene_b; scripts/build_chimeric_recurrent.py); empty = none
    bool reverse_strand = false;///< library reads are antisense to the transcript (the engine's strand probe): orients non-canonical lines
    bool canonical_only = true; ///< keep GT/AG junctions only (types 1 and 2); false keeps non-canonical lines too
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
    uint64_t noncanonical = 0;   ///< best-scoring lines dropped for a non-canonical motif
    uint64_t reoriented = 0;     ///< kept lines turned to transcript orientation
    uint64_t blacklisted_junctions = 0;  ///< distinct junctions dropped by the blacklist
    uint64_t blacklisted_rows = 0;       ///< rows (cell x junction) dropped by the blacklist
    uint64_t recurrent_rows = 0;         ///< rows flagged `recurrent`
    uint64_t blacklist_entries = 0;      ///< blacklist lines read
    std::string dedup_refusal;           ///< deduplicated pass: why the member table could not be used (empty = used; the rows are then not valid)
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

/// The gene at a junction side: genes containing `pos`, those on the given (transcript) strand first, then the smallest span; null if none.
inline const Gene* gene_ptr_at(const GeneIndex& idx, const std::string& chr, int64_t pos, char strand) {
    auto it = idx.find(chr);
    if (it == idx.end()) return nullptr;
    const Gene* best = nullptr;
    for (const auto& g : it->second) {
        if (pos < g.start || pos > g.end) continue;
        if (!best) { best = &g; continue; }
        const bool gs = g.strand == strand, bs = best->strand == strand;
        if (gs != bs) { if (gs) best = &g; continue; }
        const int64_t gl = g.end - g.start, bl = best->end - best->start;
        if (gl < bl || (gl == bl && g.name < best->name)) best = &g;
    }
    return best;
}

inline std::string gene_at(const GeneIndex& idx, const std::string& chr, int64_t pos, char strand) {
    const Gene* g = gene_ptr_at(idx, chr, pos, strand);
    return g ? g->name : "";
}

inline char flip_strand(char s) { return s == '-' ? '+' : '-'; }
inline std::string bare_contig(const std::string& c) { return c.compare(0, 3, "chr") == 0 ? c.substr(3) : c; }

// ---- Arriba blacklist (Arriba 2.x format: two columns per line, "#" comments) ----
// Item 1 is a position "[+-]contig:pos", a range "[+-]contig:start-end" or a gene name; item 2 is the same or a keyword.
// Supported for split-read evidence (every chimeric read here is a split read): positions (exact breakpoint, strand if given), ranges
// (the side's gene overlaps the range by more than half, as Arriba's overlapping_fraction, or the breakpoint lies in it when the side has
// no gene), gene names, and the keywords any and split_read_any. Keywords that need evidence this table does not have (split_read_donor,
// split_read_acceptor, discordant_mates, read_through, low_support, filter_spliced, not_both_spliced) never match: those entries keep the junction.
struct BlItem {
    enum Type { POS, RANGE, GENE, ANY, NEVER } type = NEVER;
    std::string contig;  // without "chr"
    bool prefix = false; // contig given as "name*"
    char strand = 0;     // 0 = undefined
    int64_t start = 0, end = 0;
    std::string gene;
};

inline bool parse_bl_item(const std::string& t, bool second, BlItem& it) {
    if (t.empty()) return false;
    if (second) {
        if (t == "any" || t == "split_read_any") { it.type = BlItem::ANY; return true; }
        if (t == "split_read_donor" || t == "split_read_acceptor" || t == "discordant_mates" || t == "read_through" || t == "low_support" ||
            t == "filter_spliced" || t == "not_both_spliced") { it.type = BlItem::NEVER; return true; }
    }
    const size_t colon = t.rfind(':');
    if (colon == std::string::npos) { it.type = BlItem::GENE; it.gene = t; return true; }
    std::string c = t.substr(0, colon), r = t.substr(colon + 1);
    if (!c.empty() && (c[0] == '+' || c[0] == '-')) { it.strand = c[0]; c = c.substr(1); }
    if (!c.empty() && c.back() == '*') { it.prefix = true; c.pop_back(); }
    it.contig = bare_contig(c);
    const size_t dash = r.find('-');
    try {
        if (dash == std::string::npos) { it.start = it.end = std::stoll(r); it.type = BlItem::POS; }
        else { it.start = std::stoll(r.substr(0, dash)); it.end = std::stoll(r.substr(dash + 1)); it.type = it.start == it.end ? BlItem::POS : BlItem::RANGE; }
    } catch (...) { return false; }
    return true;
}

struct Side {
    std::string contig;  // without "chr"
    int64_t pos = 0;
    char strand = '+';   // transcript strand
    const Gene* gene = nullptr;
};

inline bool bl_contig_eq(const BlItem& it, const std::string& c) {
    return it.prefix ? c.compare(0, it.contig.size(), it.contig) == 0 : c == it.contig;
}

inline bool bl_match(const BlItem& it, const Side& s) {
    switch (it.type) {
        case BlItem::ANY: return true;
        case BlItem::NEVER: return false;
        case BlItem::GENE: return s.gene && s.gene->name == it.gene;
        case BlItem::POS:
            return bl_contig_eq(it, s.contig) && (!it.strand || it.strand == s.strand) && s.pos == it.start;
        case BlItem::RANGE: {
            if (!bl_contig_eq(it, s.contig) || (it.strand && it.strand != s.strand)) return false;
            if (!s.gene) return s.pos >= it.start && s.pos <= it.end;
            const int64_t gs = s.gene->start, ge = s.gene->end, len = ge - gs + 1;
            const int64_t ov = std::min(ge, it.end) - std::max(gs, it.start);
            return len > 0 && ov > 0 && (double)ov / (double)len > 0.5;
        }
    }
    return false;
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
    std::string filter = "PASS";   ///< PASS, or "recurrent" (gene pair on the recurrent list)
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
    if (cfg.members != nullptr) {
        // Deduplicated pass: each junction line is for a distinct sequence (read name = its id); it is parsed once and applied to every member
        // read with that member's CB and UMI, running the checks below in the same order as the per-read path, so every counter and row equal
        // those of a pass that mapped every read.
        const MemberTable& mt = *cfg.members;
        std::unordered_map<uint64_t, int64_t> cb_memo;   // packed CB -> cell; -1 unmatched, -2 ambiguous
        FILE* dump = cfg.expanded_dump_path.empty() ? nullptr : std::fopen(cfg.expanded_dump_path.c_str(), "wb");
        bool refused = false;
        auto refuse = [&](const char* why) { if (!refused) st.dedup_refusal = why; refused = true; };
        detail::for_each_line(cfg.junction_path, [&](const std::string& ln) {
            if (refused) return;
            if (ln.empty() || ln[0] == '#' || ln.compare(0, 10, "chr_donorA") == 0) return;
            auto f = detail::split(ln, '\t');
            uint64_t id = 0;
            bool idok = f.size() > 9 && !f[9].empty() && f[9].size() <= 18 && f[9].find_first_not_of("0123456789") == std::string::npos;
            if (idok) { id = std::stoull(f[9]); idok = id < mt.n_ids(); }
            if (!idok) { refuse("a junction line without a valid sequence id"); return; }
            const uint64_t n = mt.count(id), m0 = mt.start[id];
            if (f.size() >= 22) {   // self-check: what STAR printed for the representative is the member table's entry 0
                uint64_t c0 = 0, u0 = 0;
                const std::string& pcb = f[f.size() - 2];
                const std::string& pumi = f[f.size() - 1];
                if (pcb.size() != mt.cb_len || pumi.size() != mt.umi_len || !MemberTable::pack(pcb.data(), pcb.size(), c0) ||
                    !MemberTable::pack(pumi.data(), pumi.size(), u0) || c0 != mt.cb[m0] || u0 != mt.umi[m0]) {
                    refuse("the barcode STAR printed differs from the member table");
                    return;
                }
            }
            if (dump) {
                std::string t;
                for (uint64_t k = 0; k < n; k++) {
                    f[9] = ".";
                    if (f.size() >= 22) { f[f.size() - 2] = MemberTable::unpack(mt.cb[m0 + k], mt.cb_len); f[f.size() - 1] = MemberTable::unpack(mt.umi[m0 + k], mt.umi_len); }
                    t.clear();
                    for (size_t i = 0; i < f.size(); i++) { if (i) t += '\t'; t += f[i]; }
                    t += '\n';
                    std::fwrite(t.data(), 1, t.size(), dump);
                }
            }
            st.lines += n;
            if (f.size() < 22) { st.no_barcode += n; return; }
            // the printed CB and UMI of every member are non-empty strings of ACGTN (MemberTable::pack), so the per-read check passes
            Line l;
            l.chr1 = f[0]; l.pos1 = std::stoll(f[1]); l.s1 = f[2].empty() ? '+' : f[2][0];
            l.chr2 = f[3]; l.pos2 = std::stoll(f[4]); l.s2 = f[5].empty() ? '+' : f[5][0];
            l.jtype = std::stoi(f[6]);
            l.read = f[9];
            l.n_aln = std::stoi(f[14]); l.score = std::stoi(f[17]); l.best = std::stoi(f[18]);
            if (l.score < l.best) return;  // a worse alignment of a multimapping chimeric read
            if (cfg.canonical_only && l.jtype < 1) { st.noncanonical += n; return; }
            if (l.jtype == 2 || (l.jtype < 1 && cfg.reverse_strand)) {
                std::swap(l.chr1, l.chr2);
                std::swap(l.pos1, l.pos2);
                const char a = l.s1;
                l.s1 = detail::flip_strand(l.s2);
                l.s2 = detail::flip_strand(a);
                if (l.jtype == 2) l.jtype = 1;
                st.reoriented += n;
            }
            for (uint64_t k = 0; k < n; k++) {
                const uint64_t cbc = mt.cb[m0 + k];
                auto it = cb_memo.find(cbc);
                if (it == cb_memo.end()) {
                    bool amb = false;
                    const int64_t r = resolve(MemberTable::unpack(cbc, mt.cb_len), amb);
                    it = cb_memo.emplace(cbc, r >= 0 ? r : (amb ? -2 : -1)).first;
                }
                const int64_t c = it->second;
                if (c < 0) { (c == -2 ? st.ambiguous : st.unmatched)++; continue; }
                Line lk = l;
                lk.cell = (uint32_t)c;
                lk.umi = MemberTable::unpack(mt.umi[m0 + k], mt.umi_len);
                st.kept_lines++;
                lines.push_back(std::move(lk));
            }
        });
        if (dump) std::fclose(dump);
    } else
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
        if (cfg.canonical_only && l.jtype < 1) { st.noncanonical++; return; }
        // transcript orientation: STAR's strands are the read's; CT/AC (type 2) and, on an antisense library, non-canonical lines turn around
        if (l.jtype == 2 || (l.jtype < 1 && cfg.reverse_strand)) {
            std::swap(l.chr1, l.chr2);
            std::swap(l.pos1, l.pos2);
            const char a = l.s1;
            l.s1 = detail::flip_strand(l.s2);
            l.s2 = detail::flip_strand(a);
            if (l.jtype == 2) l.jtype = 1;   // GT/AG on the transcript
            st.reoriented++;
        }
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
    // blacklist: distinct junctions (transcript orientation) and their sides, then one pass over the file
    using JKey = std::tuple<std::string, int64_t, char, std::string, int64_t, char>;
    std::map<JKey, uint32_t> jidx;
    std::vector<std::pair<detail::Side, detail::Side>> jsides;
    for (const auto& kv : agg) {
        const auto& [cell, c1, p1, s1, c2, p2, s2] = kv.first;
        (void)cell;
        auto ins = jidx.emplace(JKey{c1, p1, s1, c2, p2, s2}, (uint32_t)jsides.size());
        if (ins.second) {
            detail::Side a{detail::bare_contig(c1), p1, s1, detail::gene_ptr_at(genes, c1, p1, s1)};
            detail::Side b{detail::bare_contig(c2), p2, s2, detail::gene_ptr_at(genes, c2, p2, s2)};
            jsides.emplace_back(a, b);
        }
    }
    std::vector<char> black(jsides.size(), 0);
    if (!cfg.blacklist_path.empty() && !jsides.empty()) {
        // candidates: exact positions, 100 kb bins over breakpoints and gene spans, gene names (Arriba's own lookup scheme)
        constexpr int64_t kBin = 100000;
        std::unordered_set<std::string> pos_keys;
        std::unordered_map<std::string, std::vector<uint32_t>> by_bin, by_gene;
        auto add_bins = [&](const std::string& c, int64_t a, int64_t b, uint32_t j) {
            for (int64_t x = a / kBin; x <= b / kBin; ++x) by_bin[c + ":" + std::to_string(x)].push_back(j);
        };
        for (uint32_t j = 0; j < jsides.size(); ++j) {
            for (const detail::Side* sd : {&jsides[j].first, &jsides[j].second}) {
                pos_keys.insert(sd->contig + ":" + std::to_string(sd->pos));
                add_bins(sd->contig, sd->pos, sd->pos, j);
                if (sd->gene) { add_bins(sd->contig, sd->gene->start, sd->gene->end, j); by_gene[sd->gene->name].push_back(j); }
            }
        }
        // fast path for the bulk of the file (millions of "contig:pos  contig:pos" pairs): a position no junction has cannot match
        auto plain_pos_absent = [&](const std::string& t) {
            if (t.empty() || t[0] == '+' || t[0] == '-' || t.find('*') != std::string::npos) return false;
            const size_t c = t.rfind(':');
            if (c == std::string::npos || t.find('-', c) != std::string::npos) return false;
            return pos_keys.count(detail::bare_contig(t.substr(0, c)) + t.substr(c)) == 0;
        };
        std::vector<uint32_t> cand;
        detail::for_each_line(cfg.blacklist_path, [&](const std::string& ln) {
            if (ln.empty() || ln[0] == '#') return;
            const size_t tab = ln.find('\t');
            if (tab == std::string::npos) return;
            const size_t end2 = ln.find('\t', tab + 1);
            const std::string t1 = ln.substr(0, tab), t2 = ln.substr(tab + 1, end2 == std::string::npos ? std::string::npos : end2 - tab - 1);
            st.blacklist_entries++;
            if (plain_pos_absent(t1) || plain_pos_absent(t2)) return;
            detail::BlItem i1, i2;
            if (!detail::parse_bl_item(t1, false, i1) || !detail::parse_bl_item(t2, true, i2)) return;
            if (i2.type == detail::BlItem::NEVER) return;
            cand.clear();
            for (const detail::BlItem* it : {&i1, &i2}) {
                if (it->type == detail::BlItem::POS || it->type == detail::BlItem::RANGE) {
                    if (it->prefix) {   // rare (viral contigs such as "NC_001422.1*"): every junction is a candidate
                        for (uint32_t j = 0; j < jsides.size(); ++j) cand.push_back(j);
                        continue;
                    }
                    for (int64_t x = it->start / kBin; x <= it->end / kBin; ++x) {
                        auto f = by_bin.find(it->contig + ":" + std::to_string(x));
                        if (f != by_bin.end()) cand.insert(cand.end(), f->second.begin(), f->second.end());
                    }
                } else if (it->type == detail::BlItem::GENE) {
                    auto f = by_gene.find(it->gene);
                    if (f != by_gene.end()) cand.insert(cand.end(), f->second.begin(), f->second.end());
                }
            }
            for (uint32_t j : cand) {
                if (black[j]) continue;
                const auto& sides = jsides[j];
                if ((detail::bl_match(i1, sides.first) && detail::bl_match(i2, sides.second)) ||
                    (detail::bl_match(i1, sides.second) && detail::bl_match(i2, sides.first)))
                    black[j] = 1;
            }
        });
        for (char b : black) st.blacklisted_junctions += b ? 1 : 0;
    }
    std::set<std::pair<std::string, std::string>> recurrent;
    if (!cfg.recurrent_path.empty()) {
        detail::for_each_line(cfg.recurrent_path, [&](const std::string& ln) {
            if (ln.empty() || ln[0] == '#') return;
            auto f = detail::split(ln, '\t');
            if (f.size() < 2 || f[0] == "gene_a") return;
            recurrent.insert(std::minmax(f[0], f[1]));
        });
    }

    std::vector<Row> rows;
    std::set<uint32_t> cells;
    for (const auto& kv : agg) {
        const auto& [cell, c1, p1, s1, c2, p2, s2] = kv.first;
        if (black[jidx.at(JKey{c1, p1, s1, c2, p2, s2})]) { st.blacklisted_rows++; continue; }
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
            if (recurrent.count(std::minmax(r.donor_gene, r.acceptor_gene))) { r.filter = "recurrent"; st.recurrent_rows++; }
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
           "fusion\tn_reads\tn_umis\tn_chim_aln\tmultimap\tbest_score\tcatalog_sources\tcatalog_orientation\tfilter\n";
}

inline std::string row_line(const Row& r) {
    std::ostringstream o;
    o << r.barcode << '\t' << r.donor_chr << '\t' << r.donor_pos << '\t' << r.donor_strand << '\t' << r.acceptor_chr << '\t' << r.acceptor_pos << '\t'
      << r.acceptor_strand << '\t' << r.junction_type << '\t' << (r.donor_gene.empty() ? "." : r.donor_gene) << '\t'
      << (r.acceptor_gene.empty() ? "." : r.acceptor_gene) << '\t' << fusion_name(r) << '\t' << r.n_reads << '\t' << r.n_umis << '\t' << r.n_chim_aln
      << '\t' << (r.multimap ? 1 : 0) << '\t' << r.best_score << '\t' << r.sources << '\t' << r.orientation << '\t' << r.filter << '\n';
    return o.str();
}

/// Writes cfg.out_path (gzip, level 6, no timestamp-dependent bytes) and returns the statistics.
inline Stats write_table(const std::vector<std::string>& barcodes, const Config& cfg) {
    Stats st;
    std::vector<Row> rows = build_rows(barcodes, cfg, st);
    if (!st.dedup_refusal.empty()) return st;   // the caller runs the pass on all reads
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
