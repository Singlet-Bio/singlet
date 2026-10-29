// SPDX-License-Identifier: MIT
#pragma once
// sj_features: per-junction annotation of sj_counts, so the (deliberately unfiltered) junction list can be filtered
// like STARsolo's.
//
// sj_counts keeps every junction seen in a cell (PBMC 1k v3: 177,593 junctions, 134k with one molecule; STARsolo SJ:
// 89,541). STARsolo's SJ feature counts only junctions in STAR's SJ.out.tab, i.e. those that pass STAR's outSJfilter*
// rules (canonical motifs need fewer reads and shorter overhangs than non-canonical ones; novel junctions need unique
// reads). This writes one row per sj_counts feature (same order) with STAR's motif / annotation / read support where
// SJ.out.tab has the junction, and the molecule / cell support Singlet measured:
//
//   junction        sj_counts feature name (chrom:donor-acceptor; donor = STAR intron start - 1)
//   chrom, intron_start, intron_end   STAR 1-based intron coordinates
//   strand          0 undefined, 1 +, 2 - (STAR)        motif  0 non-canonical, 1 GT/AG, 2 CT/AC, 3 GC/AG, 4 CT/GC,
//   annotated       0/1 (STAR)                          5 AT/AC, 6 GT/AT (STAR)
//   unique_reads, multi_reads, max_overhang             STAR SJ.out.tab columns 7-9
//   star_sj_pass    1 when the junction is in SJ.out.tab (passes STAR's filters: the STARsolo-comparable set)
//   n_molecules     molecules over all barcodes (sj_counts row sum)
//   n_cells         called cells with >= 1 molecule;  n_cell_molecules  molecules in called cells
// Fields STAR does not report for a junction are "NA".

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <zlib.h>

namespace singlet {

struct StarSJ {
    int strand = 0, motif = 0, annotated = 0;
    uint32_t unique_reads = 0, multi_reads = 0, max_overhang = 0;
};

/// Load STAR SJ.out.tab keyed "chrom:start-end" (STAR 1-based intron coordinates).
inline std::unordered_map<std::string, StarSJ> load_star_sj_tab(const std::string& path) {
    std::unordered_map<std::string, StarSJ> m;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        std::string chrom;
        uint64_t a = 0, b = 0;
        StarSJ s;
        if (!(is >> chrom >> a >> b >> s.strand >> s.motif >> s.annotated >> s.unique_reads >> s.multi_reads >>
              s.max_overhang))
            continue;
        m[chrom + ":" + std::to_string(a) + "-" + std::to_string(b)] = s;
    }
    return m;
}

/// Write sj_features.tsv.gz. `sj` is the junctions x barcodes CSC (column = barcode); `is_cell` marks called cells
/// (empty = every barcode counts as a cell). Returns the number of rows with star_sj_pass = 1.
template <typename CSC>
inline uint64_t write_sj_features(const std::string& path, const std::vector<std::string>& names, const CSC& sj,
                                  const std::vector<uint8_t>& is_cell, const std::string& star_sj_tab) {
    const auto star = load_star_sj_tab(star_sj_tab);
    const size_t nr = names.size();
    std::vector<uint64_t> mol(nr, 0), cell_mol(nr, 0);
    std::vector<uint32_t> ncell(nr, 0);
    for (uint32_t j = 0; j < sj.ncols; ++j) {
        const bool c = is_cell.empty() || (j < is_cell.size() && is_cell[j]);
        for (int32_t k = sj.indptr[j]; k < sj.indptr[j + 1]; ++k) {
            const size_t r = static_cast<size_t>(sj.indices[k]);
            if (r >= nr) continue;
            mol[r] += sj.data[k];
            if (c) { cell_mol[r] += sj.data[k]; ++ncell[r]; }
        }
    }
    gzFile gz = gzopen(path.c_str(), "wb6");
    if (!gz) return 0;
    std::string buf =
        "junction\tchrom\tintron_start\tintron_end\tstrand\tmotif\tannotated\tunique_reads\tmulti_reads"
        "\tmax_overhang\tstar_sj_pass\tn_molecules\tn_cells\tn_cell_molecules\n";
    uint64_t n_pass = 0;
    for (size_t r = 0; r < nr; ++r) {
        const std::string& n = names[r];
        const size_t colon = n.rfind(':'), dash = n.rfind('-');
        std::string chrom = n, s_start = "NA", s_end = "NA";
        const StarSJ* st = nullptr;
        if (colon != std::string::npos && dash != std::string::npos && dash > colon) {
            chrom = n.substr(0, colon);
            const uint64_t donor = std::strtoull(n.c_str() + colon + 1, nullptr, 10);
            const uint64_t acc = std::strtoull(n.c_str() + dash + 1, nullptr, 10);
            s_start = std::to_string(donor + 1);
            s_end = std::to_string(acc);
            auto it = star.find(chrom + ":" + s_start + "-" + s_end);
            if (it != star.end()) st = &it->second;
        }
        buf += n + '\t' + chrom + '\t' + s_start + '\t' + s_end + '\t';
        if (st) {
            ++n_pass;
            buf += std::to_string(st->strand) + '\t' + std::to_string(st->motif) + '\t' + std::to_string(st->annotated) +
                   '\t' + std::to_string(st->unique_reads) + '\t' + std::to_string(st->multi_reads) + '\t' +
                   std::to_string(st->max_overhang) + "\t1\t";
        } else {
            buf += "NA\tNA\tNA\tNA\tNA\tNA\t0\t";
        }
        buf += std::to_string(mol[r]) + '\t' + std::to_string(ncell[r]) + '\t' + std::to_string(cell_mol[r]) + '\n';
        if (buf.size() > (1u << 20)) { gzwrite(gz, buf.data(), static_cast<unsigned>(buf.size())); buf.clear(); }
    }
    if (!buf.empty()) gzwrite(gz, buf.data(), static_cast<unsigned>(buf.size()));
    gzclose(gz);
    return n_pass;
}

}  // namespace singlet
