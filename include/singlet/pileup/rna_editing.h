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
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <zlib.h>

#include "pz_reader.h"
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
    // rows gathered into 1 MB chunks for gzwrite: the same bytes into the same deflate stream (no flush in between), fewer calls
    std::string chunk;
    chunk.reserve((1u << 20) + 512);
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
        chunk.append(nm);
        chunk.append(buf, static_cast<size_t>(n));
        if (chunk.size() >= (1u << 20)) { gzwrite(g, chunk.data(), static_cast<unsigned>(chunk.size())); chunk.clear(); }
    }
    if (!chunk.empty()) gzwrite(g, chunk.data(), static_cast<unsigned>(chunk.size()));
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

// ---------------------------------------------------------------------------------------------------------------------------
// Files written by export.h, read back: the editing tables over a final set of called cells, and the consistency check of the
// four files (the end-of-run self-check). A Visium run replaces its called set with the tissue spots after export, so the tables
// computed at export time (over the EmptyDrops-style call) are recomputed over the final set.
// ---------------------------------------------------------------------------------------------------------------------------

inline Csc csc_from_read(const pz::ReadResult& r) {
    Csc c;
    c.nrows = r.m; c.ncols = r.n;
    c.indptr.assign(r.indptr.begin(), r.indptr.end());
    c.indices.assign(r.indices.begin(), r.indices.end());
    c.data.resize(r.data.size());
    for (size_t k = 0; k < r.data.size(); ++k) c.data[k] = static_cast<uint16_t>(r.data[k]);
    return c;
}

inline std::vector<std::string> read_editing_cells_tsv_barcodes(const std::string& path) {
    std::vector<std::string> out;
    std::ifstream f(path);
    std::string ln;
    if (!std::getline(f, ln)) return out;
    while (std::getline(f, ln)) { if (ln.empty()) continue; out.push_back(ln.substr(0, ln.find('\t'))); }
    return out;
}

/// Recompute editing_sites.tsv.gz, editing_cells.tsv and editing_summary.json from the written matrices over `called` (barcodes).
/// Returns true when there is nothing to do (no editing outputs) or the three files were rewritten; false with `err` on failure.
inline bool resummarize(const std::string& out_prefix, const std::string& panel_path, uint32_t n_panel,
                        const std::unordered_set<std::string>& called, std::string* err = nullptr) {
    namespace fs = std::filesystem;
    const std::string dir = out_prefix + "/private/editing/";
    if (!fs::exists(dir + "editing_alt.1pz") || !fs::exists(dir + "editing_dp.1pz")) return true;
    try {
        const auto lim = pz::ReadLimits::trusted();
        const auto a = pz::read_1pz(dir + "editing_alt.1pz", lim);
        const auto d = pz::read_1pz(dir + "editing_dp.1pz", lim);
        if (a.m != d.m || a.n != d.n || a.rownames != d.rownames || a.colnames != d.colnames)
            throw std::runtime_error("editing_alt and editing_dp differ in shape or names");
        const Csc alt = csc_from_read(a), dp = csc_from_read(d);
        std::vector<uint32_t> cells;
        for (uint32_t j = 0; j < a.n; ++j) if (called.count(a.colnames[j])) cells.push_back(j);
        const auto er = summarize(alt, dp, cells);
        std::vector<std::string> cov_names(er.covered.size());
        for (size_t j = 0; j < er.covered.size(); ++j) cov_names[j] = a.rownames[er.covered[j]];
        auto tmp = [](const std::string& p) { return p + ".tmp"; };
        if (!write_sites(tmp(dir + "editing_sites.tsv.gz"), er, cov_names)) throw std::runtime_error("cannot write editing_sites.tsv.gz");
        if (!write_cells(tmp(out_prefix + "/editing_cells.tsv"), er, cells, a.colnames)) throw std::runtime_error("cannot write editing_cells.tsv");
        if (!write_summary(tmp(out_prefix + "/editing_summary.json"), panel_path, n_panel, er, cells.size())) throw std::runtime_error("cannot write editing_summary.json");
        fs::rename(tmp(dir + "editing_sites.tsv.gz"), dir + "editing_sites.tsv.gz");
        fs::rename(tmp(out_prefix + "/editing_cells.tsv"), out_prefix + "/editing_cells.tsv");
        fs::rename(tmp(out_prefix + "/editing_summary.json"), out_prefix + "/editing_summary.json");
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

/// The self-check of an editing output set: both matrices decode (trusted limits), have the same shape and names, the sites table
/// lists the matrix rows in order, and every table count equals the recomputation from the matrices over the barcodes of
/// editing_cells.tsv. Returns the problems found (empty when consistent); an absent editing output is no problem.
inline std::vector<std::string> check_outputs(const std::string& out_prefix) {
    namespace fs = std::filesystem;
    std::vector<std::string> bad;
    const std::string dir = out_prefix + "/private/editing/";
    if (!fs::exists(dir + "editing_alt.1pz") && !fs::exists(dir + "editing_dp.1pz")) return bad;
    try {
        const auto lim = pz::ReadLimits::trusted();
        const auto a = pz::read_1pz(dir + "editing_alt.1pz", lim);
        const auto d = pz::read_1pz(dir + "editing_dp.1pz", lim);
        if (a.m != d.m || a.n != d.n) { bad.push_back("alt and dp shapes differ"); return bad; }
        if (a.rownames != d.rownames) bad.push_back("alt and dp row names differ");
        if (a.colnames != d.colnames) bad.push_back("alt and dp column names differ");
        const auto cell_bc = read_editing_cells_tsv_barcodes(out_prefix + "/editing_cells.tsv");
        std::unordered_set<std::string> called(cell_bc.begin(), cell_bc.end());
        std::vector<uint8_t> is_cell(a.n, 0);
        size_t n_called_cols = 0;
        for (uint32_t j = 0; j < a.n; ++j) if (called.count(a.colnames[j])) { is_cell[j] = 1; ++n_called_cols; }
        if (n_called_cols != cell_bc.size()) bad.push_back("editing_cells.tsv names barcodes that are not matrix columns");
        {   // the tables must describe the run's final called set (cell_calls.tsv is_cell): a Visium run replaces it after export
            std::ifstream cf(out_prefix + "/cell_calls.tsv");
            std::string hl;
            if (cf && std::getline(cf, hl)) {
                std::vector<std::string> h; { std::stringstream hs(hl); std::string t; while (std::getline(hs, t, '\t')) h.push_back(t); }
                size_t bi = h.size(), ci = h.size();
                for (size_t k = 0; k < h.size(); ++k) { if (h[k] == "barcode") bi = k; if (h[k] == "is_cell") ci = k; }
                if (bi < h.size() && ci < h.size()) {
                    size_t n_final = 0, n_missing = 0;
                    std::string ln2;
                    while (std::getline(cf, ln2)) {
                        std::vector<std::string> v; std::stringstream ls(ln2); std::string t;
                        while (std::getline(ls, t, '\t')) v.push_back(t);
                        if (v.size() <= ci || !(v[ci] == "TRUE" || v[ci] == "True" || v[ci] == "true" || v[ci] == "1")) continue;
                        ++n_final;
                        if (!called.count(v[bi])) ++n_missing;
                    }
                    if (n_missing || n_final != cell_bc.size())
                        bad.push_back("editing_cells.tsv has " + std::to_string(cell_bc.size()) + " cells, cell_calls.tsv " + std::to_string(n_final) +
                                      " (" + std::to_string(n_missing) + " called cells are outside the editing tables)");
                }
            }
        }
        std::vector<uint64_t> alt(a.m, 0), dpc(a.m, 0), dpa(a.m, 0);
        std::vector<uint32_t> cov(a.m, 0), ced(a.m, 0);
        for (uint32_t j = 0; j < d.n; ++j)
            for (uint64_t k = d.indptr[j]; k < d.indptr[j + 1]; ++k) {
                dpa[d.indices[k]] += d.data[k];
                if (is_cell[j]) { dpc[d.indices[k]] += d.data[k]; if (d.data[k] > 0) ++cov[d.indices[k]]; }
            }
        for (uint32_t j = 0; j < a.n; ++j) if (is_cell[j])
            for (uint64_t k = a.indptr[j]; k < a.indptr[j + 1]; ++k) { alt[a.indices[k]] += a.data[k]; if (a.data[k] > 0) ++ced[a.indices[k]]; }
        gzFile g = gzopen((dir + "editing_sites.tsv.gz").c_str(), "rb");
        if (!g) { bad.push_back("editing_sites.tsv.gz missing"); return bad; }
        std::string text; std::vector<char> buf(1 << 16); int n;
        while ((n = gzread(g, buf.data(), static_cast<unsigned>(buf.size()))) > 0) text.append(buf.data(), size_t(n));
        gzclose(g);
        std::istringstream is(text);
        std::string ln;
        std::getline(is, ln);
        uint32_t row = 0, mism = 0;
        while (std::getline(is, ln)) {
            if (row >= a.m) { bad.push_back("sites table has more rows than the matrices"); break; }
            std::vector<std::string> f; std::stringstream ls(ln); std::string t;
            while (std::getline(ls, t, '\t')) f.push_back(t);
            if (f.size() < 8 || f[0] != a.rownames[row]) { bad.push_back("sites table row " + std::to_string(row) + " is not matrix row " + a.rownames[row]); break; }
            const uint64_t ed = std::stoull(f[3]), ref = std::stoull(f[2]);
            if (ed != alt[row] || ed + ref != dpc[row] || std::stoull(f[7]) != dpa[row] || std::stoul(f[5]) != cov[row] || std::stoul(f[6]) != ced[row]) {
                if (mism++ < 3) bad.push_back("site " + f[0] + ": table edited " + f[3] + " ref " + f[2] + " all " + f[7] + " but the matrices give edited " +
                    std::to_string(alt[row]) + " depth " + std::to_string(dpc[row]) + " all " + std::to_string(dpa[row]));
            }
            ++row;
        }
        if (row != a.m && bad.empty()) bad.push_back("sites table has " + std::to_string(row) + " rows, the matrices " + std::to_string(a.m));
        if (mism > 3) bad.push_back(std::to_string(mism) + " sites disagree in all");
    } catch (const std::exception& e) {
        bad.push_back(std::string("decode: ") + e.what());
    }
    return bad;
}

}  // namespace singlet::editing
