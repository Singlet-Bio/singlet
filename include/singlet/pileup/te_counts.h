// SPDX-License-Identifier: MIT
#pragma once
// te_counts.h - transposable element (TE) molecules per cell, from the aligned reads.
//
// The counting rule follows the two open single-cell TE counters it is benchmarked against (bench/readlevel/te):
//   - SoloTE (Rodriguez-Quiroz and Valdebenito-Maturana 2022): uniquely mapped reads are counted per TE locus, multi-mapped reads
//     per subfamily (the primary alignment names the subfamily);
//   - scTE (He et al. 2021): reads that overlap an annotated gene exon belong to the gene, not to a TE, and counts collapse to
//     subfamilies (metagenes).
// Reads come from the pileup's primary alignments that pass its barcode and MAPQ gates (unique low-MAPQ reads are skipped there).
// A read is assigned to the subfamily that covers most of its aligned bases (ties: no assignment) and, when unique, to the locus of
// that subfamily that covers most of them (ties: the lower locus index). Molecules are distinct (cell, feature, UMI) triples, so the
// result depends only on the set of records, not on their order: the CPU and GPU paths add the same records in different orders.
//
// Three counting rules (TeRule), chosen with --te-rule; SINGLET_TE_ALL_RULES=1 also writes te_counts.<rule>.1pz for the other two:
//   - singlet (default): the rule above.
//   - scte: a port of scTE's exclusive mode (MIT, He et al. 2021, github.com/JiekaiLab/scTE; see NOTICE). TE loci that overlap an exon
//     of a protein-coding or lincRNA gene are removed from the index (bin/scTE_build genomeIndex, scTE/annotation.py annoGtf); a read
//     counts for every remaining TE locus it overlaps, whether or not it also overlaps a gene (scTE/base.py align); and a molecule
//     (cell, UMI) counts at its leftmost read only (scTE/base.py Bam2bed, the first-read-per-barcode-and-UMI step on the sorted BAM).
//   - solote: reimplemented from the SoloTE method (Rodriguez-Quiroz and Valdebenito-Maturana 2022; method citation only, no SoloTE
//     code is used). Reads that the gene counter would assign to exactly one gene (unique, every aligned block inside an exon of that
//     gene on the sense strand) stay with the gene; every other read counts for every TE locus it overlaps by at least one base, at
//     the locus when unique and at the subfamily when multi-mapped.
//
// Annotation: a TEtranscripts-style GTF (Tam and Hammell, GRCh38_GENCODE_rmsk_TE.gtf / GRCm39_GENCODE_rmsk_TE.gtf, from UCSC
// RepeatMasker): one line per locus, gene_id = subfamily, transcript_id = locus, family_id, class_id.
#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "interval_tree.h"

namespace singlet {

/// Counting rule of a TE record (see the header comment).
enum class TeRule : uint8_t { kSinglet = 0, kScte = 1, kSoloTE = 2 };
constexpr int kTeRules = 3;
inline const char* te_rule_name(TeRule r) {
    return r == TeRule::kScte ? "scte" : r == TeRule::kSoloTE ? "solote" : "singlet";
}
/// Parse a rule name; false for an unknown name.
inline bool te_rule_parse(const std::string& s, TeRule& r) {
    if (s == "singlet") r = TeRule::kSinglet;
    else if (s == "scte") r = TeRule::kScte;
    else if (s == "solote") r = TeRule::kSoloTE;
    else return false;
    return true;
}

struct TeRecord {
    uint32_t fam;    ///< subfamily index (row of te_counts)
    uint32_t bc;     ///< barcode column
    uint64_t umi;    ///< UMI hash (0 = no UMI: each read is its own molecule)
    uint32_t locus;  ///< locus index, UINT32_MAX for a multi-mapped read
    TeRule rule = TeRule::kSinglet;
    int32_t tid = 0;  ///< the read's contig and leftmost aligned base (the scte rule keeps a molecule's leftmost read)
    int32_t pos = 0;
};

/// SoloTE method: does the gene counter own this read? True when the read is unique and exactly one gene has an exon containing every
/// aligned block on the read's sense strand (what STARsolo's GN tag marks; SoloTE counts only reads without it). `genes(s, e, out)`
/// fills the genes with an exon that contains [s, e) on the sense strand. Reimplemented from the method; no SoloTE code.
template <class GenesFn>
inline bool te_gene_assigned(const std::vector<std::pair<int32_t, int32_t>>& blocks, bool unique, GenesFn&& genes,
                             std::vector<uint32_t>& cand, std::vector<uint32_t>& tmp) {
    if (!unique || blocks.empty()) return false;
    genes(blocks[0].first, blocks[0].second, cand);
    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    for (size_t k = 1; k < blocks.size() && !cand.empty(); ++k) {
        genes(blocks[k].first, blocks[k].second, tmp);
        std::sort(tmp.begin(), tmp.end());
        size_t w = 0;
        for (uint32_t g : cand) if (std::binary_search(tmp.begin(), tmp.end(), g)) cand[w++] = g;
        cand.resize(w);
    }
    return cand.size() == 1;
}

class TeModel {
public:
    /// Load the TE GTF (plain or gzip). Returns false when the file cannot be read or holds no locus.
    bool load_gtf(const std::string& path) {
        gzFile f = gzopen(path.c_str(), "rb");
        if (!f) return false;
        std::unordered_map<std::string, uint32_t> fam_idx;
        std::vector<char> buf(1 << 16);
        std::string line;
        auto attr = [](const std::string& s, const char* key) -> std::string {
            const std::string k = std::string(key) + " \"";
            size_t p = s.find(k);
            if (p == std::string::npos) return "";
            p += k.size();
            const size_t q = s.find('"', p);
            return q == std::string::npos ? "" : s.substr(p, q - p);
        };
        while (gzgets(f, buf.data(), static_cast<int>(buf.size()))) {
            line.assign(buf.data());
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            size_t c[8];
            size_t p = 0;
            int n = 0;
            for (; n < 8; ++n) {
                p = line.find('\t', p);
                if (p == std::string::npos) break;
                c[n] = p++;
            }
            if (n < 8) continue;
            const std::string chrom = line.substr(0, c[0]);
            const long s1 = std::strtol(line.c_str() + c[2] + 1, nullptr, 10), e1 = std::strtol(line.c_str() + c[3] + 1, nullptr, 10);
            if (s1 < 1 || e1 < s1) continue;
            const std::string attrs = line.substr(c[7] + 1);
            std::string sub = attr(attrs, "gene_id");
            if (sub.empty()) continue;
            std::string cls = attr(attrs, "class_id"), famname = attr(attrs, "family_id");
            if (!cls.empty() && cls.back() == '?') cls.pop_back();
            if (!famname.empty() && famname.back() == '?') famname.pop_back();
            auto it = fam_idx.find(sub);
            uint32_t fi;
            if (it == fam_idx.end()) {
                fi = static_cast<uint32_t>(fam_names_.size());
                fam_idx.emplace(sub, fi);
                fam_names_.push_back(sub + ":" + (famname.empty() ? "NA" : famname) + ":" + (cls.empty() ? "NA" : cls));
            } else {
                fi = it->second;
            }
            const uint32_t li = static_cast<uint32_t>(locus_fam_.size());
            locus_fam_.push_back(fi);
            locus_start_.push_back(static_cast<int32_t>(s1 - 1));
            locus_end_.push_back(static_cast<int32_t>(e1));
            locus_strand_.push_back(line[c[5] + 1]);
            // locus id: the GTF's transcript_id when it names the locus (older TEtranscripts files: AluY_dup123); the 2026 files repeat
            // the subfamily there, so the id is then built from the coordinates: subfamily|chrom:start-end:strand (1-based, inclusive)
            std::string id = attr(attrs, "transcript_id");
            if (id.empty() || id == sub)
                id = sub + "|" + chrom + ":" + std::to_string(s1) + "-" + std::to_string(e1) + ":" + line[c[5] + 1];
            locus_ids_.push_back(std::move(id));
            chrom_intervals_[chrom].push_back(Interval{static_cast<int32_t>(s1 - 1), static_cast<int32_t>(e1), li});
        }
        gzclose(f);
        return !locus_fam_.empty();
    }

    /// Build the per-contig trees against the alignment's contig ids (chr prefix added or removed when only that matches).
    void resolve_chroms(const std::unordered_map<std::string, int32_t>& chrom_to_tid) {
        int32_t max_tid = -1;
        for (const auto& kv : chrom_to_tid) max_tid = std::max(max_tid, kv.second);
        trees_.assign(static_cast<size_t>(max_tid + 1), IntervalTree());
        locus_tid_.assign(locus_fam_.size(), -1);
        for (auto& kv : chrom_intervals_) {
            int32_t tid = -1;
            auto it = chrom_to_tid.find(kv.first);
            if (it == chrom_to_tid.end() && kv.first.rfind("chr", 0) == 0) it = chrom_to_tid.find(kv.first.substr(3));
            if (it == chrom_to_tid.end() && kv.first.rfind("chr", 0) != 0) it = chrom_to_tid.find("chr" + kv.first);
            if (it != chrom_to_tid.end()) tid = it->second;
            if (tid < 0) continue;
            for (const auto& iv : kv.second) locus_tid_[iv.index] = tid;
            trees_[static_cast<size_t>(tid)].build(std::move(kv.second));
            ++n_resolved_;
        }
        chrom_intervals_.clear();
    }

    bool empty() const { return locus_fam_.empty(); }
    uint32_t n_families() const { return static_cast<uint32_t>(fam_names_.size()); }
    uint32_t n_loci() const { return static_cast<uint32_t>(locus_fam_.size()); }
    uint32_t n_resolved_contigs() const { return n_resolved_; }
    const std::vector<std::string>& family_names() const { return fam_names_; }
    const std::string& locus_id(uint32_t i) const { return locus_ids_[i]; }
    uint32_t locus_family(uint32_t i) const { return locus_fam_[i]; }
    int32_t locus_tid(uint32_t i) const { return locus_tid_[i]; }
    int32_t locus_start(uint32_t i) const { return locus_start_[i]; }
    int32_t locus_end(uint32_t i) const { return locus_end_[i]; }
    char locus_strand(uint32_t i) const { return locus_strand_[i]; }

    /// Assign one alignment. `blocks` are its aligned [start, end) reference blocks; `exonic(s, e)` says whether [s, e) overlaps an
    /// annotated gene exon (on either strand). Returns false when the read is exonic, overlaps no TE, or two subfamilies tie.
    template <class ExonFn>
    bool assign(int32_t tid, const std::vector<std::pair<int32_t, int32_t>>& blocks, bool unique, ExonFn&& exonic, uint32_t& fam,
                uint32_t& locus, std::vector<uint32_t>& scratch) const {
        if (tid < 0 || static_cast<size_t>(tid) >= trees_.size() || blocks.empty()) return false;
        const IntervalTree& tr = trees_[static_cast<size_t>(tid)];
        if (tr.size() == 0) return false;
        // per locus: aligned bases inside it (a read touches a handful of loci)
        hits_.clear();
        for (const auto& b : blocks) {
            tr.query(b.first, b.second, scratch);
            for (uint32_t li : scratch) {
                const int32_t ov = std::min(b.second, locus_end_[li]) - std::max(b.first, locus_start_[li]);
                if (ov <= 0) continue;
                bool found = false;
                for (auto& h : hits_) if (h.first == li) { h.second += ov; found = true; break; }
                if (!found) hits_.emplace_back(li, ov);
            }
        }
        if (hits_.empty()) return false;
        for (const auto& b : blocks) if (exonic(b.first, b.second)) return false;
        // per subfamily: summed bases; best subfamily must be unique
        fam_hits_.clear();
        for (const auto& h : hits_) {
            const uint32_t f = locus_fam_[h.first];
            bool found = false;
            for (auto& x : fam_hits_) if (x.first == f) { x.second += h.second; found = true; break; }
            if (!found) fam_hits_.emplace_back(f, h.second);
        }
        uint32_t bf = UINT32_MAX;
        int32_t bv = 0;
        bool tie = false;
        for (const auto& x : fam_hits_) {
            if (x.second > bv) { bf = x.first; bv = x.second; tie = false; }
            else if (x.second == bv) tie = true;
        }
        if (bf == UINT32_MAX || tie) return false;
        fam = bf;
        locus = UINT32_MAX;
        if (unique) {
            int32_t lv = 0;
            for (const auto& h : hits_)
                if (locus_fam_[h.first] == bf && (h.second > lv || (h.second == lv && h.first < locus))) { locus = h.first; lv = h.second; }
        }
        return true;
    }

    // ---- scTE port (MIT, He et al. 2021; NOTICE) -------------------------------------------------------------------------------
    /// scTE contig filter, ported from scTE bin/scTE_build (chr_list, readGtf, genomeIndex): genes count on contigs 1-49, X, Y and M
    /// (with or without "chr"). A TE is kept when its contig without "chr" is one of those, or when the name holds no "chr" at all
    /// (scTE appends the TE file's contig names to chr_list but tests them with "chr" removed, so "chrUn_..." contigs drop out).
    static bool scte_main_contig(const std::string& chrom) {
        std::string c = chrom;
        for (size_t p; (p = c.find("chr")) != std::string::npos;) c.erase(p, 3);
        if (c == "X" || c == "Y" || c == "M") return true;
        if (c.empty() || c.size() > 2) return false;
        for (char ch : c) if (ch < '0' || ch > '9') return false;
        const int v = std::atoi(c.c_str());
        return v >= 1 && v <= 49;
    }

    /// scTE exclusive index: mark the TE loci that overlap an exon of a protein-coding or lincRNA gene; the scte rule never counts them.
    /// Ported from scTE bin/scTE_build genomeIndex (mode 'exclusive', with readGtf and cleanexon) and scTE/annotation.py annoGtf:
    ///   - gene lines: feature 'exon' or 'UTR', a gene_name attribute, the substring 'protein_coding' or 'lincRNA' anywhere on the line
    ///     (readGtf's 'clean' set), on a main contig (scte_main_contig);
    ///   - cleanexon: per gene_name, the union of each exon's bases range(start, end) (1-based, end excluded), cut into runs [Es, Ee];
    ///   - a TE (BED start s0, end e) drops out when s0 < Ee and e > Es for any run (scTE's test, kept to the base).
    /// Deviation: scTE keeps one contig per gene_name (its last exon's); the runs here keep each exon's own contig. Call before
    /// resolve_chroms. `nointron` ports scTE's 'nointron' mode: each gene's run is its whole span (min start to max end over all of its
    /// exon and UTR lines, every gene type: readGtf's 'raw' set), so TEs in introns drop out too.
    /// Returns the number of loci masked, or -1 when the GTF cannot be read.
    long mask_scte_exclusive(const std::string& gene_gtf, bool nointron = false) {
        gzFile f = gzopen(gene_gtf.c_str(), "rb");
        if (!f) return -1;
        // per contig, per gene_name: the exon lines' 1-based [start, end]
        std::map<std::string, std::map<std::string, std::vector<std::pair<long, long>>>> genes;
        std::vector<char> buf(1 << 16);
        std::string line;
        while (gzgets(f, buf.data(), static_cast<int>(buf.size()))) {
            line.assign(buf.data());
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            size_t c[8];
            size_t p = 0;
            int n = 0;
            for (; n < 8; ++n) {
                p = line.find('\t', p);
                if (p == std::string::npos) break;
                c[n] = p++;
            }
            if (n < 8) continue;
            const std::string feat = line.substr(c[1] + 1, c[2] - c[1] - 1);
            if (feat != "exon" && feat != "UTR") continue;
            const std::string chrom = line.substr(0, c[0]);
            if (!scte_main_contig(chrom)) continue;
            const size_t gp = line.find("gene_name \"", c[7]);
            if (gp == std::string::npos) continue;
            if (!nointron && line.find("protein_coding") == std::string::npos && line.find("lincRNA") == std::string::npos) continue;
            const size_t g0 = gp + 11, g1 = line.find('"', g0);
            if (g1 == std::string::npos) continue;
            const long l = std::strtol(line.c_str() + c[2] + 1, nullptr, 10), r = std::strtol(line.c_str() + c[3] + 1, nullptr, 10);
            genes[chrom][line.substr(g0, g1 - g0)].push_back({l, r});
        }
        gzclose(f);
        // cleanexon runs per contig: closed [Es, Ee] stored as Interval{Es, Ee}; scTE's test s0 < Ee && e > Es is then the standard
        // half-open overlap of the TE's [s0, e) with [Es, Ee)
        std::unordered_map<std::string, std::vector<Interval>> runs;
        for (auto& ck : genes) {
            auto& out = runs[ck.first];
            for (auto& gk : ck.second) {
                const auto& v = gk.second;
                std::vector<std::pair<long, long>> sets;   // closed base sets range(start, end)
                if (nointron) {
                    long lo = v[0].first, hi = v[0].second;
                    for (const auto& x : v) { lo = std::min(lo, x.first); hi = std::max(hi, x.second); }
                    if (hi - 1 >= lo) sets.push_back({lo, hi - 1});
                } else {
                    for (const auto& x : v) if (x.second - 1 >= x.first) sets.push_back({x.first, x.second - 1});
                }
                std::sort(sets.begin(), sets.end());
                for (size_t i = 0; i < sets.size();) {
                    long a = sets[i].first, b = sets[i].second;
                    size_t j = i + 1;
                    while (j < sets.size() && sets[j].first <= b + 1) { b = std::max(b, sets[j].second); ++j; }
                    out.push_back(Interval{static_cast<int32_t>(a), static_cast<int32_t>(b), static_cast<uint32_t>(out.size())});
                    i = j;
                }
            }
        }
        scte_masked_.assign(locus_fam_.size(), 0);
        long n_masked = 0;
        std::vector<uint32_t> hits;
        for (auto& kv : chrom_intervals_) {
            const bool keep_contig = scte_main_contig(kv.first) || kv.first.find("chr") == std::string::npos;
            auto it = runs.find(kv.first);
            IntervalTree tr;
            if (it != runs.end()) tr.build(it->second);
            for (const auto& iv : kv.second) {
                bool drop = !keep_contig;
                if (!drop && it != runs.end()) {
                    tr.query(iv.start, iv.end, hits);
                    for (uint32_t h : hits) {
                        const Interval& r = it->second[h];
                        if (iv.start < r.end && iv.end > r.start) { drop = true; break; }
                    }
                }
                if (drop) { scte_masked_[iv.index] = 1; ++n_masked; }
            }
        }
        return n_masked;
    }
    bool scte_masked(uint32_t locus) const { return locus < scte_masked_.size() && scte_masked_[locus]; }

    /// Every TE locus the read overlaps by at least `min_overlap` aligned bases, as (subfamily, locus) in locus order, each locus once.
    /// `masked` skips the loci scTE's exclusive index removes. Used by the scte rule (scTE/base.py align: every overlapping feature
    /// gets the read) and the solote rule (SoloTE method: every overlapping locus, minimum overlap 1 base).
    void overlaps_all(int32_t tid, const std::vector<std::pair<int32_t, int32_t>>& blocks, int32_t min_overlap, bool masked,
                      std::vector<std::pair<uint32_t, uint32_t>>& out, std::vector<uint32_t>& scratch) const {
        out.clear();
        if (tid < 0 || static_cast<size_t>(tid) >= trees_.size() || blocks.empty()) return;
        const IntervalTree& tr = trees_[static_cast<size_t>(tid)];
        if (tr.size() == 0) return;
        hits_.clear();
        for (const auto& b : blocks) {
            tr.query(b.first, b.second, scratch);
            for (uint32_t li : scratch) {
                if (masked && scte_masked(li)) continue;
                const int32_t ov = std::min(b.second, locus_end_[li]) - std::max(b.first, locus_start_[li]);
                if (ov <= 0) continue;
                bool found = false;
                for (auto& h : hits_) if (h.first == li) { h.second += ov; found = true; break; }
                if (!found) hits_.emplace_back(li, ov);
            }
        }
        for (const auto& h : hits_) if (h.second >= min_overlap) out.emplace_back(locus_fam_[h.first], h.first);
        std::sort(out.begin(), out.end(), [](const std::pair<uint32_t, uint32_t>& a, const std::pair<uint32_t, uint32_t>& b) {
            return a.second < b.second;
        });
    }

    /// One read's TE records under `rule`, appended to `out`. `exonic(s, e)`: [s, e) overlaps a gene exon on either strand (singlet
    /// rule); `genes(s, e, v)`: the genes with an exon containing [s, e) on the read's sense strand (solote rule).
    template <class ExonFn, class GenesFn>
    void records(TeRule rule, int32_t tid, const std::vector<std::pair<int32_t, int32_t>>& blocks, bool unique, uint32_t bc, uint64_t umi,
                 ExonFn&& exonic, GenesFn&& genes, std::vector<uint32_t>& scratch, std::vector<uint32_t>& scratch2,
                 std::vector<TeRecord>& out) const {
        if (blocks.empty()) return;
        const int32_t pos = blocks.front().first;
        if (rule == TeRule::kSinglet) {
            uint32_t fam = 0, locus = UINT32_MAX;
            if (assign(tid, blocks, unique, exonic, fam, locus, scratch)) out.push_back(TeRecord{fam, bc, umi, locus, rule, tid, pos});
            return;
        }
        if (rule == TeRule::kSoloTE && te_gene_assigned(blocks, unique, genes, scratch, scratch2)) return;
        overlaps_all(tid, blocks, 1, rule == TeRule::kScte, all_, scratch);
        for (const auto& fl : all_) out.push_back(TeRecord{fl.first, bc, umi, unique ? fl.second : UINT32_MAX, rule, tid, pos});
    }

private:
    std::vector<uint8_t> scte_masked_;    // scTE exclusive index: 1 = locus removed (mask_scte_exclusive)
    std::vector<std::string> fam_names_;  // subfamily:family:class
    std::vector<uint32_t> locus_fam_;
    std::vector<int32_t> locus_start_, locus_end_, locus_tid_;
    std::vector<char> locus_strand_;
    std::vector<std::string> locus_ids_;
    std::unordered_map<std::string, std::vector<Interval>> chrom_intervals_;
    std::vector<IntervalTree> trees_;
    uint32_t n_resolved_ = 0;
    // scratch of assign(): a TeModel is shared read-only by the pileup's threads, so these live per thread
    static thread_local inline std::vector<std::pair<uint32_t, int32_t>> hits_;
    static thread_local inline std::vector<std::pair<uint32_t, int32_t>> fam_hits_;
    static thread_local inline std::vector<std::pair<uint32_t, uint32_t>> all_;
};

/// te_counts (subfamilies x barcodes) and te_locus_counts (loci with a unique molecule x barcodes), CSC, uint16 (clamped).
struct TeMatrix {
    uint32_t nrows = 0, ncols = 0;
    std::vector<int32_t> indptr, indices;
    std::vector<uint16_t> data;
    std::vector<std::string> names;
};

struct TeTables {
    TeMatrix fam, locus;
    std::string loci_tsv;  ///< te_loci.tsv: one row per te_locus_counts row
    uint64_t n_records = 0, n_unique_records = 0, fam_molecules = 0, locus_molecules = 0;
};

namespace te_detail {
/// Distinct (row, col, umi) triples -> CSC (rows x n_cols). `trip` is sorted in place.
struct Trip { uint32_t row, col; uint64_t umi; uint32_t seq; };
inline TeMatrix to_csc(std::vector<Trip>& trip, uint32_t n_rows, uint32_t n_cols, uint64_t& molecules) {
    std::sort(trip.begin(), trip.end(), [](const Trip& a, const Trip& b) {
        if (a.col != b.col) return a.col < b.col;
        if (a.row != b.row) return a.row < b.row;
        if (a.umi != b.umi) return a.umi < b.umi;
        return a.seq < b.seq;
    });
    TeMatrix m;
    m.nrows = n_rows;
    m.ncols = n_cols;
    m.indptr.assign(static_cast<size_t>(n_cols) + 1, 0);
    molecules = 0;
    for (size_t i = 0; i < trip.size();) {
        size_t j = i;
        uint32_t n = 0;
        while (j < trip.size() && trip[j].col == trip[i].col && trip[j].row == trip[i].row) {
            // a record without a UMI is its own molecule; records with one UMI collapse
            if (trip[j].umi == 0 || j == i || trip[j].umi != trip[j - 1].umi) ++n;
            ++j;
        }
        m.indices.push_back(static_cast<int32_t>(trip[i].row));
        m.data.push_back(static_cast<uint16_t>(std::min<uint32_t>(n, 65535)));
        molecules += n;
        m.indptr[static_cast<size_t>(trip[i].col) + 1]++;
        i = j;
    }
    for (uint32_t c = 0; c < n_cols; ++c) m.indptr[c + 1] += m.indptr[c];
    return m;
}
}  // namespace te_detail

/// Build both matrices and the locus table from the records (any order).
/// scTE molecule rule, ported from scTE/base.py Bam2bed: on the coordinate-sorted BAM, scTE keeps the first read of each (cell barcode,
/// UMI) pair (its awk '!x[$4$5]++' step), so a molecule counts once, at its leftmost read. Order-free form: keep the records whose
/// (contig, leftmost base) is the smallest of their (cell, UMI); records without a UMI stay. Deviation: scTE takes the first of all
/// reads, genic ones included; only TE records are seen here.
inline std::vector<TeRecord> scte_leftmost_per_molecule(std::vector<TeRecord> recs) {
    auto key = [](const TeRecord& r) { return (static_cast<uint64_t>(static_cast<uint32_t>(r.tid)) << 32) | static_cast<uint32_t>(r.pos); };
    std::unordered_map<uint32_t, std::unordered_map<uint64_t, uint64_t>> best;   // cell -> UMI -> smallest key
    for (const auto& r : recs) {
        if (r.umi == 0) continue;
        auto& m = best[r.bc];
        auto it = m.find(r.umi);
        if (it == m.end()) m.emplace(r.umi, key(r));
        else it->second = std::min(it->second, key(r));
    }
    size_t w = 0;
    for (size_t i = 0; i < recs.size(); ++i)
        if (recs[i].umi == 0 || best[recs[i].bc][recs[i].umi] == key(recs[i])) recs[w++] = recs[i];
    recs.resize(w);
    return recs;
}

/// Records of `rule` only (the others are ignored), so one record vector can carry every rule (SINGLET_TE_ALL_RULES).
inline TeTables build_te_tables(const std::vector<TeRecord>& all_recs, const TeModel& model, uint32_t n_barcodes,
                                const std::vector<std::string>& tid_to_chrom, TeRule rule = TeRule::kSinglet) {
    TeTables t;
    std::vector<TeRecord> recs;
    recs.reserve(all_recs.size());
    for (const auto& r : all_recs) if (r.rule == rule) recs.push_back(r);
    if (rule == TeRule::kScte) recs = scte_leftmost_per_molecule(std::move(recs));
    t.n_records = recs.size();
    std::vector<te_detail::Trip> ft;
    ft.reserve(recs.size());
    std::vector<uint32_t> used;
    for (const auto& r : recs) {
        ft.push_back({r.fam, r.bc, r.umi, 0});
        if (r.locus != UINT32_MAX) used.push_back(r.locus);
    }
    t.n_unique_records = used.size();
    t.fam = te_detail::to_csc(ft, model.n_families(), n_barcodes, t.fam_molecules);
    t.fam.names = model.family_names();
    // loci with a unique record, in locus-index order (the GTF's order) -> rows
    std::sort(used.begin(), used.end());
    used.erase(std::unique(used.begin(), used.end()), used.end());
    std::unordered_map<uint32_t, uint32_t> row_of;
    row_of.reserve(used.size() * 2);
    for (uint32_t i = 0; i < used.size(); ++i) row_of.emplace(used[i], i);
    std::vector<te_detail::Trip> lt;
    lt.reserve(t.n_unique_records);
    for (const auto& r : recs)
        if (r.locus != UINT32_MAX) lt.push_back({row_of[r.locus], r.bc, r.umi, 0});
    t.locus = te_detail::to_csc(lt, static_cast<uint32_t>(used.size()), n_barcodes, t.locus_molecules);
    // per-row molecules and cells for the table
    std::vector<uint64_t> row_mol(used.size(), 0), row_cells(used.size(), 0);
    for (uint32_t c = 0; c < n_barcodes; ++c)
        for (int32_t k = t.locus.indptr[c]; k < t.locus.indptr[c + 1]; ++k) {
            row_mol[static_cast<size_t>(t.locus.indices[k])] += t.locus.data[k];
            row_cells[static_cast<size_t>(t.locus.indices[k])]++;
        }
    t.loci_tsv = "locus\tsubfamily\tchrom\tstart\tend\tstrand\tn_molecules\tn_cells\n";
    t.locus.names.reserve(used.size());
    for (uint32_t i = 0; i < used.size(); ++i) {
        const uint32_t li = used[i];
        const int32_t tid = model.locus_tid(li);
        const std::string chrom = (tid >= 0 && static_cast<size_t>(tid) < tid_to_chrom.size()) ? tid_to_chrom[tid] : "?";
        const std::string& fam = model.family_names()[model.locus_family(li)];
        t.locus.names.push_back(model.locus_id(li));
        t.loci_tsv += model.locus_id(li) + '\t' + fam.substr(0, fam.find(':')) + '\t' + chrom + '\t' +
                      std::to_string(model.locus_start(li) + 1) + '\t' + std::to_string(model.locus_end(li)) + '\t' +
                      model.locus_strand(li) + '\t' + std::to_string(row_mol[i]) + '\t' + std::to_string(row_cells[i]) + '\n';
    }
    return t;
}

}  // namespace singlet
