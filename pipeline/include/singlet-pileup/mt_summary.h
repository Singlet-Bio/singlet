#pragma once
// singlet-pileup: mt_summary.h
// Per-cell mitochondrial summary TSV writer for mt_summary.tsv.
//
// Writes all required columns from DROPLET_OUTPUT_SCHEMA.md §5.4:
//   barcode  donor_id  mt_reads  mt_coverage_mean  n_loci_with_event
//   n_substitutions  n_indels  n_frameshifts  n_premature_stops
//   pct_heteroplasmic
//
// n_indels, n_frameshifts, n_premature_stops are always 0 because the
// current pileup mt CSC does not track indel or CDS-annotation events.
// This is consistent with the mt_event_caller.h deviation note.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mt_event_caller.h"

namespace singlet {
namespace mt {

// ============================================================================
// Write mt_summary.tsv from MtEventResult summaries.
//
// @param path      Full output path (e.g. "<out_prefix>/mt/mt_summary.tsv")
// @param result    MtEventResult from classify_mt_events()
// @param barcodes  Barcode strings in original matrix column order
// @returns         true on success
// ============================================================================
inline bool write_mt_summary(
    const std::string& path,
    const MtEventResult& result,
    const std::vector<std::string>& barcodes)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[mt_summary] WARN: cannot write " << path << "\n";
        return false;
    }

    f << "barcode\tdonor_id\tmt_reads\tmt_coverage_mean"
         "\tn_loci_with_event\tn_substitutions\tn_indels"
         "\tn_frameshifts\tn_premature_stops\tpct_heteroplasmic\n";

    for (uint32_t cell = 0; cell < result.n_cells; ++cell) {
        const auto& s  = result.summaries[cell];
        const std::string& bc =
            (cell < barcodes.size()) ? barcodes[cell] : "";

        f << bc                    << "\t"
          << s.donor_label         << "\t"
          << s.mt_reads            << "\t"
          << s.mt_coverage_mean    << "\t"
          << s.n_loci_with_event   << "\t"
          << s.n_substitutions     << "\t"
          << s.n_indels            << "\t"
          << s.n_frameshifts       << "\t"
          << s.n_premature_stops   << "\t"
          << s.pct_heteroplasmic   << "\n";
    }
    return true;
}

}  // namespace mt
}  // namespace singlet
