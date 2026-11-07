// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: atac_metrics.h
// Per-barcode ATAC QC, peak x barcode and gene-activity matrices from deduplicated fragments.
//
// Definitions (each named after the tool whose convention it follows, so the numbers can be compared):
//   fragments              unique fragments outside the mitochondrial chromosome (Cell Ranger "passed_filters" /
//                          "atac_fragments")
//   read_pairs             read pairs behind all fragments incl. duplicates and chrM; duplicates = read_pairs - unique
//   mito_fragments         unique fragments on chrM / MT
//   tss_fragments          fragments overlapping TSS +- 1000 bp (Cell Ranger ATAC "TSS_fragments")
//   tss_enrichment         ArchR: (insertions within TSS +- 50 bp / 101) / (insertions in the two 100 bp flanks at
//                          +-1901..2000 bp / 200), flank floor 0.2 insertions (ArchR minNorm)
//   nucleosome_signal      Signac: fragments 147-294 bp / fragments < 147 bp
//   peak_region_fragments  fragments overlapping a peak (Cell Ranger "peak_region_fragments"); frip = / fragments
//   peak_region_cutsites   cut sites inside peaks (Cell Ranger "peak_region_cutsites")
// Peak matrix: cut sites per peak and barcode (Cell Ranger ATAC peak-barcode matrix).
// Gene activity: fragments overlapping the gene body + 2 kb upstream, protein-coding genes (Signac GeneActivity).

#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "atac_fragment.h"
#include "atac_peak_caller.h"

namespace singlet {

struct AtacBarcodeMetrics {
    uint64_t read_pairs = 0;
    uint32_t fragments = 0;
    uint32_t mito_fragments = 0;
    uint32_t tss_fragments = 0;
    uint32_t tss_insertions = 0;
    uint32_t flank_insertions = 0;
    uint32_t nfr_fragments = 0;   // < 147 bp
    uint32_t mono_fragments = 0;  // 147 - 294 bp
    uint32_t peak_region_fragments = 0;
    uint32_t peak_region_cutsites = 0;
    uint32_t gene_activity_fragments = 0;

    float tss_enrichment() const {
        const double flank = std::max(static_cast<double>(flank_insertions), 0.2) / 200.0;
        return static_cast<float>((tss_insertions / 101.0) / flank);
    }
    float nucleosome_signal() const {
        return nfr_fragments ? static_cast<float>(mono_fragments) / nfr_fragments : 0.f;
    }
    float frip() const { return fragments ? static_cast<float>(peak_region_fragments) / fragments : 0.f; }
};

struct AtacGeneRegion {
    int32_t chrom_idx;
    int32_t start, end;  // gene body + 2 kb upstream, 0-based half-open
    uint32_t gene;       // row in gene_names
};

/// TSS positions and protein-coding gene regions from a GTF (Ensembl / GENCODE / 10x).
struct AtacAnnotation {
    std::vector<std::vector<int32_t>> tss;  // per chrom, sorted unique
    std::vector<AtacGeneRegion> genes;      // sorted by (chrom, start)
    std::vector<std::string> gene_names;    // "gene_id_gene_name"

    bool load_gtf(const std::string& path, const std::unordered_map<std::string, int32_t>& chrom_map,
                  size_t n_chrom, int32_t upstream = 2000) {
        gzFile f = gzopen(path.c_str(), "rb");
        if (!f) return false;
        tss.assign(n_chrom, {});
        std::vector<char> buf(1 << 16);
        auto attr = [](const std::string& a, const char* key) -> std::string {
            const std::string k = std::string(key) + " \"";
            auto p = a.find(k);
            if (p == std::string::npos) return {};
            p += k.size();
            auto q = a.find('"', p);
            return q == std::string::npos ? std::string() : a.substr(p, q - p);
        };
        while (gzgets(f, buf.data(), static_cast<int>(buf.size()))) {
            if (buf[0] == '#') continue;
            std::string line(buf.data());
            std::string col[9];
            size_t p = 0;
            for (int i = 0; i < 9; ++i) {
                size_t q = (i < 8) ? line.find('\t', p) : line.size();
                if (q == std::string::npos) break;
                col[i] = line.substr(p, q - p);
                p = q + 1;
            }
            const bool is_tx = col[2] == "transcript", is_gene = col[2] == "gene";
            if (!is_tx && !is_gene) continue;
            auto it = chrom_map.find(col[0]);
            if (it == chrom_map.end()) continue;
            const int32_t s = std::atoi(col[3].c_str()) - 1, e = std::atoi(col[4].c_str());
            const bool plus = col[6] != "-";
            if (is_tx) {
                tss[static_cast<size_t>(it->second)].push_back(plus ? s : e - 1);
            } else {
                std::string type = attr(col[8], "gene_type");
                if (type.empty()) type = attr(col[8], "gene_biotype");
                if (type != "protein_coding") continue;
                std::string id = attr(col[8], "gene_id"), name = attr(col[8], "gene_name");
                if (name.empty()) name = id;
                AtacGeneRegion g{it->second, plus ? std::max(0, s - upstream) : s, plus ? e : e + upstream,
                                 static_cast<uint32_t>(gene_names.size())};
                gene_names.push_back(id + "_" + name);
                genes.push_back(g);
            }
        }
        gzclose(f);
        for (auto& v : tss) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
        std::sort(genes.begin(), genes.end(), [](const AtacGeneRegion& a, const AtacGeneRegion& b) {
            return a.chrom_idx != b.chrom_idx ? a.chrom_idx < b.chrom_idx : a.start < b.start;
        });
        return true;
    }
};

/// Sparse matrix in triplet form, converted to CSC for write_mtx.
struct AtacTriplets {
    std::vector<uint64_t> key;  // (col << 32) | row
    void add(uint32_t row, uint32_t col) { key.push_back((static_cast<uint64_t>(col) << 32) | row); }
    void to_csc(uint32_t n_cols, std::vector<int32_t>& indptr, std::vector<int32_t>& indices,
                std::vector<uint32_t>& data) {
        std::sort(key.begin(), key.end());
        indptr.assign(n_cols + 1, 0);
        indices.clear();
        data.clear();
        for (size_t i = 0; i < key.size();) {
            size_t j = i;
            while (j < key.size() && key[j] == key[i]) ++j;
            const uint32_t col = static_cast<uint32_t>(key[i] >> 32);
            indices.push_back(static_cast<int32_t>(key[i] & 0xffffffffu));
            data.push_back(static_cast<uint32_t>(j - i));
            ++indptr[col + 1];
            i = j;
        }
        for (uint32_t c = 0; c < n_cols; ++c) indptr[c + 1] += indptr[c];
        key.clear();
        key.shrink_to_fit();
    }
};

namespace detail {

// Merge overlapping [s, e) intervals (sorted by start) into a disjoint union.
inline std::vector<std::pair<int32_t, int32_t>> merge_intervals(std::vector<std::pair<int32_t, int32_t>> v) {
    std::sort(v.begin(), v.end());
    std::vector<std::pair<int32_t, int32_t>> out;
    for (auto& iv : v) {
        if (!out.empty() && iv.first <= out.back().second) out.back().second = std::max(out.back().second, iv.second);
        else out.push_back(iv);
    }
    return out;
}

inline bool overlaps_any(const std::vector<std::pair<int32_t, int32_t>>& u, int32_t s, int32_t e) {
    // first interval with end > s
    auto it = std::upper_bound(u.begin(), u.end(), s, [](int32_t x, const std::pair<int32_t, int32_t>& iv) {
        return x < iv.second;
    });
    return it != u.end() && it->first < e;
}

}  // namespace detail

/// Compute per-barcode metrics (indexed by fragment barcode_idx, n_barcodes entries) and, for barcodes with
/// col_of[bc] >= 0, the peak x column and gene x column matrices.
/// fragments: sorted by (chrom, start). peaks: sorted, non-overlapping per chrom (as call_peaks_insertions returns).
inline std::vector<AtacBarcodeMetrics> compute_atac_metrics(const std::vector<ATACFragment>& fragments,
                                                            size_t n_barcodes, const std::vector<bool>& is_mito,
                                                            const AtacAnnotation& ann,
                                                            const std::vector<AtacPeak>& peaks,
                                                            const std::vector<int32_t>* col_of = nullptr,
                                                            AtacTriplets* peak_mat = nullptr,
                                                            AtacTriplets* gene_mat = nullptr) {
    std::vector<AtacBarcodeMetrics> m(n_barcodes);
    const size_t nc = is_mito.size();
    // Per-chrom: TSS +- 1000 union, peak index ranges, gene regions with prefix max end.
    std::vector<std::vector<std::pair<int32_t, int32_t>>> tss_win(nc), peak_iv(nc);
    std::vector<std::vector<uint32_t>> peak_id(nc);
    for (size_t c = 0; c < nc && c < ann.tss.size(); ++c) {
        std::vector<std::pair<int32_t, int32_t>> w;
        w.reserve(ann.tss[c].size());
        for (int32_t t : ann.tss[c]) w.emplace_back(std::max(0, t - 1000), t + 1001);
        tss_win[c] = detail::merge_intervals(std::move(w));
    }
    for (uint32_t i = 0; i < peaks.size(); ++i) {
        const size_t c = static_cast<size_t>(peaks[i].chrom_idx);
        if (c >= nc) continue;
        peak_iv[c].emplace_back(peaks[i].start, peaks[i].end);
        peak_id[c].push_back(i);
    }
    std::vector<std::vector<AtacGeneRegion>> genes(nc);
    std::vector<std::vector<int32_t>> gmax(nc);
    for (const auto& g : ann.genes)
        if (static_cast<size_t>(g.chrom_idx) < nc) genes[static_cast<size_t>(g.chrom_idx)].push_back(g);
    for (size_t c = 0; c < nc; ++c) {
        int32_t mx = 0;
        for (const auto& g : genes[c]) { mx = std::max(mx, g.end); gmax[c].push_back(mx); }
    }

    auto peak_of = [&](size_t c, int32_t x) -> int64_t {
        const auto& u = peak_iv[c];
        auto it = std::upper_bound(u.begin(), u.end(), x, [](int32_t v, const std::pair<int32_t, int32_t>& iv) {
            return v < iv.second;
        });
        if (it == u.end() || it->first > x) return -1;
        return peak_id[c][static_cast<size_t>(it - u.begin())];
    };
    auto tss_count = [&](size_t c, int32_t x, AtacBarcodeMetrics& bm) {
        if (c >= ann.tss.size()) return;
        const auto& t = ann.tss[c];
        auto it = std::lower_bound(t.begin(), t.end(), x - 2000);
        for (; it != t.end() && *it <= x + 2000; ++it) {
            const int32_t d = std::abs(x - *it);
            if (d <= 50) ++bm.tss_insertions;
            else if (d >= 1901) ++bm.flank_insertions;
        }
    };

    for (const auto& f : fragments) {
        if (f.barcode_idx >= n_barcodes) continue;
        auto& bm = m[f.barcode_idx];
        const size_t c = static_cast<size_t>(f.chrom_idx);
        bm.read_pairs += f.read_pairs;
        if (c < nc && is_mito[c]) { ++bm.mito_fragments; continue; }
        ++bm.fragments;
        const int32_t len = f.end - f.start;
        if (len < 147) ++bm.nfr_fragments;
        else if (len <= 294) ++bm.mono_fragments;
        if (c >= nc) continue;
        if (detail::overlaps_any(tss_win[c], f.start, f.end)) ++bm.tss_fragments;
        tss_count(c, f.start, bm);
        tss_count(c, f.end - 1, bm);
        const int32_t col = col_of ? (*col_of)[f.barcode_idx] : -1;
        // peaks
        const int64_t p1 = peak_of(c, f.start), p2 = peak_of(c, f.end - 1);
        bm.peak_region_cutsites += (p1 >= 0) + (p2 >= 0);
        if (p1 >= 0 || p2 >= 0 || detail::overlaps_any(peak_iv[c], f.start, f.end)) ++bm.peak_region_fragments;
        if (col >= 0 && peak_mat) {
            if (p1 >= 0) peak_mat->add(static_cast<uint32_t>(p1), static_cast<uint32_t>(col));
            if (p2 >= 0) peak_mat->add(static_cast<uint32_t>(p2), static_cast<uint32_t>(col));
        }
        // gene activity: regions with start < f.end and end > f.start
        const auto& gv = genes[c];
        if (gv.empty()) continue;
        auto hi = std::lower_bound(gv.begin(), gv.end(), f.end,
                                   [](const AtacGeneRegion& g, int32_t v) { return g.start < v; });
        bool any = false;
        for (size_t k = static_cast<size_t>(hi - gv.begin()); k-- > 0;) {
            if (gmax[c][k] <= f.start) break;
            if (gv[k].end > f.start) {
                any = true;
                if (col >= 0 && gene_mat) gene_mat->add(gv[k].gene, static_cast<uint32_t>(col));
            }
        }
        if (any) ++bm.gene_activity_fragments;
    }
    return m;
}

inline bool write_atac_metrics_tsv(const std::string& path, const std::vector<AtacBarcodeMetrics>& m,
                                   const std::vector<std::string>& barcodes, const std::vector<int32_t>& order,
                                   const std::vector<uint8_t>& is_cell) {
    std::ofstream out(path);
    if (!out) return false;
    out << "barcode\tis_cell\tread_pairs\tfragments\tduplicate_read_pairs\tmito_fragments\ttss_fragments"
           "\ttss_enrichment\tnucleosome_signal\tpeak_region_fragments\tpeak_region_cutsites\tfrip\n";
    for (int32_t i : order) {
        const auto& x = m[static_cast<size_t>(i)];
        const uint64_t unique = static_cast<uint64_t>(x.fragments) + x.mito_fragments;
        out << barcodes[static_cast<size_t>(i)] << '\t' << static_cast<int>(is_cell[static_cast<size_t>(i)]) << '\t'
            << x.read_pairs << '\t' << x.fragments << '\t' << (x.read_pairs - unique) << '\t' << x.mito_fragments << '\t'
            << x.tss_fragments << '\t' << x.tss_enrichment() << '\t' << x.nucleosome_signal() << '\t'
            << x.peak_region_fragments << '\t' << x.peak_region_cutsites << '\t' << x.frip() << '\n';
    }
    return static_cast<bool>(out);
}

}  // namespace singlet
