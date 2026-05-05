#pragma once
// singlet-pileup: barcode_rank.h
// G-BARCODE-RANK — Barcode rank data for knee / inflection plot generation.
//
// Produces a sorted rank table (rank, barcode, umi_count, is_cell) suitable
// for external knee-plot tools (e.g. barcodeRanks in DropletUtils, Cell Ranger
// barcode_rank.csv). Ranks are 1-based, sorted descending by UMI count.
//
// TSV format (tab-separated, with header):
//   rank  barcode  umi_count  is_cell
//   1     ACGTACGT...  12345  True
//   2     TTGGCCAA...  11980  True
//   ...
//
// API:
//   BarcodeRankEntry   — one rank-table row
//   BarcodeRankConfig  — output path
//   build_barcode_rank(barcodes, totals, called_cells) -> vector<BarcodeRankEntry>
//   write_barcode_rank(ranks, cfg) -> bool

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

namespace singlet {

// ---------------------------------------------------------------------------
// Config + record types
// ---------------------------------------------------------------------------

struct BarcodeRankConfig {
    std::string filepath;  ///< Output TSV path
};

struct BarcodeRankEntry {
    uint32_t    rank;      ///< 1-based rank (1 = highest UMI count)
    std::string barcode;
    uint64_t    umi_count;
    bool        is_cell;   ///< Called as a cell?
};

// ---------------------------------------------------------------------------
// build_barcode_rank
// ---------------------------------------------------------------------------

/// Build a barcode rank table sorted descending by UMI count.
///
/// @param barcodes     All barcode sequences (length N)
/// @param totals       Per-barcode total UMI count (length N)
/// @param called_cells Set of barcode indices that were called as cells
///
/// @return Rank-sorted vector; rank 1 = highest UMI count.
inline std::vector<BarcodeRankEntry> build_barcode_rank(
    const std::vector<std::string>&     barcodes,
    const std::vector<uint64_t>&        totals,
    const std::unordered_set<uint32_t>& called_cells)
{
    const uint32_t n = static_cast<uint32_t>(barcodes.size());
    std::vector<BarcodeRankEntry> out;
    out.reserve(n);

    if (n == 0) return out;

    // Build sorted order: largest UMI first, then stable by barcode string
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(),
        [&](uint32_t a, uint32_t b) {
            return totals[a] != totals[b] ? totals[a] > totals[b]
                                           : barcodes[a] < barcodes[b];
        });

    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t idx = order[r];
        BarcodeRankEntry e;
        e.rank      = r + 1u;
        e.barcode   = barcodes[idx];
        e.umi_count = totals[idx];
        e.is_cell   = (called_cells.count(idx) > 0);
        out.push_back(std::move(e));
    }
    return out;
}

// ---------------------------------------------------------------------------
// write_barcode_rank
// ---------------------------------------------------------------------------

/// Write a barcode rank table to a tab-separated file.
///
/// Header: rank\tbarcode\tumi_count\tis_cell
/// is_cell is written as "True" / "False".
///
/// @return true on success, false on IO error.
inline bool write_barcode_rank(const std::vector<BarcodeRankEntry>& ranks,
                                const BarcodeRankConfig&              cfg)
{
    std::ofstream out(cfg.filepath);
    if (!out) {
        std::cerr << "[barcode_rank] ERROR: cannot open " << cfg.filepath << "\n";
        return false;
    }

    out << "rank\tbarcode\tumi_count\tis_cell\n";
    for (const auto& e : ranks) {
        out << e.rank << '\t'
            << e.barcode << '\t'
            << e.umi_count << '\t'
            << (e.is_cell ? "True" : "False") << '\n';
    }
    return out.good();
}

} // namespace singlet
