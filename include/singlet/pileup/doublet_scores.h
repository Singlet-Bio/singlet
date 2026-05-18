// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: doublet_scores.h
// G-DOUBLET-EXPORT — Per-cell doublet score export (standalone TSV writer).
//
// Provides a simple, self-contained TSV writer for doublet scores computed by
// doublet_detect.h. Output is compatible with downstream filtering tools
// (Seurat AddMetaData, scran colData, etc.).
//
// TSV format (tab-separated, with header):
//   barcode  score  class
//   AAACCTGA...  0.83  singlet
//   AACCGGTT...  1.97  doublet
//
// API:
//   DoubletScoreEntry  — per-cell record (barcode + score + class)
//   DoubletScoreConfig — output path
//   write_doublet_scores(scores, cfg) -> bool

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace singlet {

// ---------------------------------------------------------------------------
// Config + record types
// ---------------------------------------------------------------------------

struct DoubletScoreConfig {
    std::string filepath;  ///< Output TSV path
};

struct DoubletScoreEntry {
    std::string barcode;
    double      score;       ///< Doublet score [0, inf). Values ≥ threshold flagged.
    bool        is_doublet;  ///< True if classified as doublet
    std::string class_label; ///< "singlet" or "doublet"
};

// ---------------------------------------------------------------------------
// make_doublet_score_entry — convenience builder
// ---------------------------------------------------------------------------

/// Build a DoubletScoreEntry from primitive parts.
inline DoubletScoreEntry make_doublet_score_entry(const std::string& barcode,
                                                   double             score,
                                                   bool               is_doublet)
{
    DoubletScoreEntry e;
    e.barcode     = barcode;
    e.score       = score;
    e.is_doublet  = is_doublet;
    e.class_label = is_doublet ? "doublet" : "singlet";
    return e;
}

// ---------------------------------------------------------------------------
// write_doublet_scores
// ---------------------------------------------------------------------------

/// Write per-cell doublet scores to a tab-separated file.
///
/// Header: barcode\tscore\tclass
/// Rows:   one per entry, class = "singlet" or "doublet"
///
/// @return true on success, false on IO error.
inline bool write_doublet_scores(const std::vector<DoubletScoreEntry>& scores,
                                  const DoubletScoreConfig&              cfg)
{
    std::ofstream out(cfg.filepath);
    if (!out) {
        std::cerr << "[doublet_scores] ERROR: cannot open " << cfg.filepath << "\n";
        return false;
    }

    out << "barcode\tscore\tclass\n";
    for (const auto& e : scores) {
        out << e.barcode << '\t' << e.score << '\t' << e.class_label << '\n';
    }
    return out.good();
}

} // namespace singlet
