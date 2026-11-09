#pragma once
// D-101: RNA editing at known sites (REDIportal A-to-I).
//
// The editing panel joins the SNP pileup as a block of sites after the SNP and AIM blocks (pileup_engine.h, load_edit_panel), so
// the CPU and GPU kernels count it with the same per-molecule consensus: per cell and site, `alt` = UMIs whose consensus base is
// the edited base (G on the transcript, so G for a + strand site and C for a - strand site on the genome) and `dp` = reference
// plus edited UMIs. Export cuts that block off the SNP matrices before any SNP consumer reads them (split_tail), so donor demux,
// genotypes, ASE and ancestry see exactly the rows they saw without the panel.
//
// Outputs (export.h):
//   private/editing/editing_alt.*, editing_dp.*   per-cell edited / total UMIs, the panel sites covered by >= 1 UMI in any barcode
//   private/editing/editing_sites.tsv.gz          per site over the called cells: ref and edited UMIs, level, cells covered / edited
//   editing_cells.tsv                             per called cell: edited and total UMIs over all sites, sites covered, editing index
//   editing_summary.json                          panel, coverage and the sample editing index (edited / total UMIs, called cells)
// Per-site counts are allele counts at fixed positions, so they stay owner-only with the SNP matrices (D-101: allele counts are
// owner-only); the per-cell index pools thousands of sites and is public.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <zlib.h>

#include "sparse_accumulator.h"

namespace singlet::editing {

using Csc = SparseAccumulator<uint16_t>::CSCMatrix;

/// Move rows [start, nrows) of `m` into the returned matrix (rows renumbered from 0) and leave rows [0, start) in `m`.
/// Row order inside each column is kept, so `m` is byte for byte the matrix built without the tail rows.
inline Csc split_tail(Csc& m, uint32_t start) {
    Csc t;
    t.ncols = m.ncols;
    t.nrows = m.nrows > start ? m.nrows - start : 0u;
    t.indptr.assign(static_cast<size_t>(m.ncols) + 1, 0);
    if (m.nrows <= start || m.indptr.empty()) return t;
    std::vector<int32_t> head_ptr(static_cast<size_t>(m.ncols) + 1, 0);
    size_t w = 0;
    for (uint32_t c = 0; c < m.ncols; ++c) {
        const int32_t a = m.indptr[c], b = m.indptr[c + 1];
        for (int32_t k = a; k < b; ++k) {
            const int32_t r = m.indices[k];
            if (static_cast<uint32_t>(r) >= start) {
                t.indices.push_back(r - static_cast<int32_t>(start));
                t.data.push_back(m.data[k]);
            } else {
                m.indices[w] = r;
                m.data[w] = m.data[k];
                ++w;
            }
        }
        head_ptr[c + 1] = static_cast<int32_t>(w);
        t.indptr[c + 1] = static_cast<int32_t>(t.indices.size());
    }
    m.indices.resize(w);
    m.data.resize(w);
    m.indices.shrink_to_fit();
    m.data.shrink_to_fit();
    m.indptr.swap(head_ptr);
    m.nrows = start;
    return t;
}

/// Keep only the rows listed in `keep` (ascending), renumbered 0..keep.size()-1.
inline Csc select_rows(const Csc& m, const std::vector<uint32_t>& keep) {
    std::vector<int32_t> remap(m.nrows, -1);
    for (size_t i = 0; i < keep.size(); ++i) remap[keep[i]] = static_cast<int32_t>(i);
    Csc o;
    o.nrows = static_cast<uint32_t>(keep.size());
    o.ncols = m.ncols;
    o.indptr.assign(static_cast<size_t>(m.ncols) + 1, 0);
    for (uint32_t c = 0; c < m.ncols; ++c) {
        for (int32_t k = m.indptr[c]; k < m.indptr[c + 1]; ++k) {
            const int32_t r = remap[m.indices[k]];
            if (r < 0) continue;
            o.indices.push_back(r);
            o.data.push_back(m.data[k]);
        }
        o.indptr[c + 1] = static_cast<int32_t>(o.indices.size());
    }
    return o;
}

struct SiteStats {
    uint64_t alt = 0, dp = 0;       // called cells
    uint32_t cells_cov = 0, cells_edited = 0;
    uint64_t dp_all = 0;            // every barcode
};

struct EditingResult {
    std::vector<uint32_t> covered;          // panel rows with dp >= 1 in any barcode
    std::vector<SiteStats> site;            // per panel row
    std::vector<uint64_t> cell_alt, cell_dp; // per called cell (order of cell_indices)
    std::vector<uint32_t> cell_sites;
    uint64_t alt_total = 0, dp_total = 0;   // called cells
    uint32_t n_sites_10 = 0;                // sites with >= 10 UMIs over the called cells
};

/// alt and dp share their sparsity pattern only where alt > 0; dp drives coverage.
inline EditingResult summarize(const Csc& alt, const Csc& dp, const std::vector<uint32_t>& cells) {
    EditingResult r;
    r.site.resize(dp.nrows);
    std::vector<uint8_t> is_cell(dp.ncols, 0);
    for (uint32_t c : cells) if (c < dp.ncols) is_cell[c] = 1;
    for (uint32_t c = 0; c < dp.ncols; ++c)
        for (int32_t k = dp.indptr[c]; k < dp.indptr[c + 1]; ++k) {
            auto& s = r.site[dp.indices[k]];
            s.dp_all += dp.data[k];
            if (is_cell[c]) { s.dp += dp.data[k]; if (dp.data[k] > 0) ++s.cells_cov; }
        }
    r.cell_alt.assign(cells.size(), 0); r.cell_dp.assign(cells.size(), 0); r.cell_sites.assign(cells.size(), 0);
    for (size_t i = 0; i < cells.size(); ++i) {
        const uint32_t c = cells[i];
        if (c >= dp.ncols) continue;
        for (int32_t k = dp.indptr[c]; k < dp.indptr[c + 1]; ++k) { r.cell_dp[i] += dp.data[k]; if (dp.data[k] > 0) ++r.cell_sites[i]; }
        if (c < alt.ncols)
            for (int32_t k = alt.indptr[c]; k < alt.indptr[c + 1]; ++k) {
                r.cell_alt[i] += alt.data[k];
                auto& s = r.site[alt.indices[k]];
                s.alt += alt.data[k];
                if (alt.data[k] > 0) ++s.cells_edited;
            }
    }
    for (uint32_t i = 0; i < dp.nrows; ++i) {
        if (r.site[i].dp_all > 0) r.covered.push_back(i);
        if (r.site[i].dp >= 10) ++r.n_sites_10;
        r.alt_total += r.site[i].alt; r.dp_total += r.site[i].dp;
    }
    return r;
}

/// private/editing/editing_sites.tsv.gz: covered sites, called-cell totals. `cov_names[j]` names panel row r.covered[j]
/// (chrom:pos:R>A, the row names of the private matrices).
inline bool write_sites(const std::string& path, const EditingResult& r, const std::vector<std::string>& cov_names) {
    if (cov_names.size() != r.covered.size()) return false;
    gzFile g = gzopen(path.c_str(), "wb6");
    if (!g) return false;
    gzputs(g, "site\tstrand\tref_umis\tedited_umis\tediting_level\tcells_covered\tcells_edited\tumis_all_barcodes\n");
    char buf[256];
    for (size_t j = 0; j < r.covered.size(); ++j) {
        const auto& s = r.site[r.covered[j]];
        const std::string& nm = cov_names[j];
        const char strand = nm.size() >= 3 && nm[nm.size() - 3] == 'T' ? '-' : '+';
        const uint64_t ref = s.dp - s.alt;
        int n;
        if (s.dp > 0) n = std::snprintf(buf, sizeof buf, "\t%c\t%llu\t%llu\t%.6g\t%u\t%u\t%llu\n", strand, (unsigned long long)ref,
                                        (unsigned long long)s.alt, double(s.alt) / double(s.dp), s.cells_cov, s.cells_edited, (unsigned long long)s.dp_all);
        else n = std::snprintf(buf, sizeof buf, "\t%c\t0\t0\tNA\t0\t0\t%llu\n", strand, (unsigned long long)s.dp_all);
        gzwrite(g, nm.data(), static_cast<unsigned>(nm.size()));
        gzwrite(g, buf, static_cast<unsigned>(n));
    }
    return gzclose(g) == Z_OK;
}

/// editing_cells.tsv: one row per called cell.
inline bool write_cells(const std::string& path, const EditingResult& r, const std::vector<uint32_t>& cells,
                        const std::vector<std::string>& barcodes) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    std::fputs("barcode\tedited_umis\ttotal_umis\tsites_covered\tediting_index\n", f);
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i] >= barcodes.size()) continue;
        std::fprintf(f, "%s\t%llu\t%llu\t%u\t", barcodes[cells[i]].c_str(), (unsigned long long)r.cell_alt[i],
                     (unsigned long long)r.cell_dp[i], r.cell_sites[i]);
        if (r.cell_dp[i] > 0) std::fprintf(f, "%.6g\n", double(r.cell_alt[i]) / double(r.cell_dp[i]));
        else std::fputs("NA\n", f);
    }
    return std::fclose(f) == 0;
}

inline bool write_summary(const std::string& path, const std::string& panel_path, uint32_t n_panel, const EditingResult& r,
                          size_t n_cells) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    std::string base = panel_path.substr(panel_path.find_last_of('/') == std::string::npos ? 0 : panel_path.find_last_of('/') + 1);
    std::string esc;
    for (char ch : base) { if (ch == '"' || ch == '\\') esc += '\\'; esc += ch; }
    std::fprintf(f, "{\n  \"panel\": \"%s\",\n  \"panel_sites\": %u,\n  \"sites_covered\": %zu,\n  \"sites_10_umis_called_cells\": %u,\n"
                    "  \"called_cells\": %zu,\n  \"edited_umis_called_cells\": %llu,\n  \"total_umis_called_cells\": %llu,\n  \"editing_index\": ",
                 esc.c_str(), n_panel, r.covered.size(), r.n_sites_10, n_cells, (unsigned long long)r.alt_total, (unsigned long long)r.dp_total);
    if (r.dp_total > 0) std::fprintf(f, "%.6g", double(r.alt_total) / double(r.dp_total)); else std::fputs("null", f);
    std::fputs(",\n  \"definition\": \"editing_index = edited / (reference + edited) UMIs over all panel sites and called cells; "
               "UMI-level consensus per molecule, MAPQ >= min_mapq, base quality >= min_baseq\"\n}\n", f);
    return std::fclose(f) == 0;
}

}  // namespace singlet::editing
