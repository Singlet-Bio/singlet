// SPDX-License-Identifier: MIT
#pragma once
// Per-barcode read fates from STARsolo `--soloCellReadStats Standard` (Solo.out/Gene/CellReads.stats).
//
// STAR 2.7.11b columns (SoloCommon.h statNames): CB cbMatch cbPerfect cbMMunique cbMMmultiple genomeU genomeM featureU
// featureM exonic intronic exonicAS intronicAS mito countedU countedM, then nUMIunique nGenesUnique nUMImulti
// nGenesMulti. The first data row "CBnotInPasslist" counts reads whose barcode matched no passlist entry (sample level).
// Derived per barcode:
//   unmapped   = cbMatch - genomeU - genomeM       (reads with a valid barcode that STAR did not align)
//   antisense  = exonicAS + intronicAS             (relative to --soloStrand, i.e. the library's sense strand)
//   intergenic = max(0, genomeU - exonic - intronic - exonicAS - intronicAS - mito)   (approximate: STAR's flags can
//                overlap for reads spanning exon and intron)
// BAMs are deleted after each atlas sample, so these counts are kept in cell_qc_metrics.tsv; the raw file is removed
// after it is parsed (one row per barcode with any read: tens of MB).

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet {

struct StarCellReads {
    uint64_t cb_match = 0, genome_u = 0, genome_m = 0, exonic = 0, intronic = 0, exonic_as = 0, intronic_as = 0,
             mito = 0;
    bool present = false;
    uint64_t unmapped() const {
        const uint64_t g = genome_u + genome_m;
        return cb_match > g ? cb_match - g : 0;
    }
    uint64_t antisense() const { return exonic_as + intronic_as; }
    uint64_t intergenic() const {
        const uint64_t used = exonic + intronic + exonic_as + intronic_as + mito;
        return genome_u > used ? genome_u - used : 0;
    }
};

struct StarCellReadStats {
    bool found = false;
    std::vector<StarCellReads> per_barcode;  ///< indexed like the engine's barcode list
    StarCellReads no_passlist;               ///< reads whose CB matched no passlist barcode
    uint64_t n_rows = 0;
};

inline StarCellReadStats read_star_cell_read_stats(const std::string& path, const std::vector<std::string>& barcodes) {
    StarCellReadStats r;
    std::ifstream f(path);
    if (!f) return r;
    std::string line;
    if (!std::getline(f, line)) return r;
    // header -> column positions
    std::vector<std::string> hdr;
    { std::istringstream is(line); std::string t; while (std::getline(is, t, '\t')) hdr.push_back(t); }
    auto col = [&](const char* n) -> int {
        for (size_t i = 0; i < hdr.size(); ++i) if (hdr[i] == n) return static_cast<int>(i);
        return -1;
    };
    const int c_cb = col("cbMatch"), c_gu = col("genomeU"), c_gm = col("genomeM"), c_ex = col("exonic"),
              c_in = col("intronic"), c_exa = col("exonicAS"), c_ina = col("intronicAS"), c_mt = col("mito");
    if (c_cb < 0 || c_gu < 0 || c_gm < 0) return r;
    std::unordered_map<std::string, uint32_t> idx;
    idx.reserve(barcodes.size() * 2);
    for (size_t i = 0; i < barcodes.size(); ++i) idx.emplace(barcodes[i], static_cast<uint32_t>(i));
    r.per_barcode.assign(barcodes.size(), StarCellReads{});
    r.found = true;
    std::vector<uint64_t> v;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        const std::string cb = line.substr(0, tab);
        v.assign(hdr.size(), 0);
        size_t p = tab + 1, k = 1;
        while (p <= line.size() && k < hdr.size()) {
            size_t q = line.find('\t', p);
            if (q == std::string::npos) q = line.size();
            v[k++] = std::strtoull(line.c_str() + p, nullptr, 10);
            p = q + 1;
        }
        StarCellReads s;
        s.present = true;
        s.cb_match = v[c_cb]; s.genome_u = v[c_gu]; s.genome_m = v[c_gm];
        if (c_ex >= 0) s.exonic = v[c_ex];
        if (c_in >= 0) s.intronic = v[c_in];
        if (c_exa >= 0) s.exonic_as = v[c_exa];
        if (c_ina >= 0) s.intronic_as = v[c_ina];
        if (c_mt >= 0) s.mito = v[c_mt];
        ++r.n_rows;
        if (cb == "CBnotInPasslist") { r.no_passlist = s; continue; }
        auto it = idx.find(cb);
        if (it == idx.end()) {
            // STAR writes CBs as passlist strings; the engine may carry a "-1" suffix
            it = idx.find(cb + "-1");
            if (it == idx.end()) continue;
        }
        r.per_barcode[it->second] = s;
    }
    return r;
}

}  // namespace singlet
