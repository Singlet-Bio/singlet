// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: mt_event_matrix.h
// Sparse uint8 per-cell × locus event matrix writer for mt_events.1pz.
//
// Thin wrapper around pz::write_1pz<uint8_t> that accepts MtEventResult from
// mt_event_caller.h and emits the .1pz with required embedded metadata.
//
// This header exists as a standalone unit so that the event accumulation step
// and the write step can be independently tested and replaced.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "mt_event_caller.h"
#include "pz_writer.h"

namespace singlet {
namespace mt {

// ============================================================================
// Write mt_events.1pz from a fully classified MtEventResult.
//
// @param path      Full output path (e.g. "<out_prefix>/mt/mt_events.1pz")
// @param result    MtEventResult from classify_mt_events()
// @param barcodes  Column names (cell barcodes, in original matrix order)
// @returns         true on success
// ============================================================================
inline bool write_mt_event_matrix(
    const std::string& path,
    const MtEventResult& result,
    const std::vector<std::string>& barcodes)
{
    // Safe indptr even for empty (zero-loci) matrix
    const std::vector<int32_t>* p_indptr  = &result.indptr;
    std::vector<int32_t> empty_indptr;
    if (result.n_loci == 0) {
        empty_indptr.assign(result.n_cells + 1, 0);
        p_indptr = &empty_indptr;
    }

    return pz::write_1pz<uint8_t>(
        path,
        result.n_loci,
        result.n_cells,
        *p_indptr,
        result.indices,
        result.data,
        result.locus_names,
        barcodes,
        /*zstd_level=*/3,
        /*chunk_cols=*/1024,
        /*threads=*/4,
        result.pz_meta);
}

}  // namespace mt
}  // namespace singlet
