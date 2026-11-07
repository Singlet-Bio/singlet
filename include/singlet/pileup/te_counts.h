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
// Annotation: a TEtranscripts-style GTF (Tam and Hammell, GRCh38_GENCODE_rmsk_TE.gtf / GRCm39_GENCODE_rmsk_TE.gtf, from UCSC
// RepeatMasker): one line per locus, gene_id = subfamily, transcript_id = locus, family_id, class_id.
#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "interval_tree.h"

namespace singlet {

struct TeRecord {
    uint32_t fam;    ///< subfamily index (row of te_counts)
    uint32_t bc;     ///< barcode column
    uint64_t umi;    ///< UMI hash (0 = no UMI: each read is its own molecule)
    uint32_t locus;  ///< locus index, UINT32_MAX for a multi-mapped read
};

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
            std::string id = attr(attrs, "transcript_id");
            locus_ids_.push_back(id.empty() ? sub + "_" + std::to_string(li) : std::move(id));
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

private:
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
inline TeTables build_te_tables(const std::vector<TeRecord>& recs, const TeModel& model, uint32_t n_barcodes,
                                const std::vector<std::string>& tid_to_chrom) {
    TeTables t;
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
