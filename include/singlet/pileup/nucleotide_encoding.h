// SPDX-License-Identifier: MIT
// pileup/nucleotide_encoding.h — Shared nucleotide encoding lookup tables
//
// Consolidates base<->index conversion tables that were previously rolled
// separately in mt_heteroplasmy.h, rna_variant_caller.h and (for the htslib
// 4-bit code path) ad hoc seq_nt16_str usage in pileup_engine.h.
//
// Encodings:
//   - 0-3 index            : A=0, C=1, G=2, T=3   (the project-wide convention)
//   - htslib bam_seqi code : 1=A, 2=C, 4=G, 8=T, 15=N (4-bit nibble)
//   - ASCII char           : 'A','C','G','T','N'
//
// Note: fq/types.h keeps its own nuc::ascii_to_num table because fq/ must not
// depend on pileup/ (it sits below pileup/ in the include graph). The table
// here is the canonical reference for pileup-tier code.

#pragma once

#include <cstdint>

namespace singlet::pileup {
namespace nt {

// ── 0-3 index <-> ASCII ────────────────────────────────────────────────────

// 0-3 (and 4=N) index -> ASCII char.
inline constexpr char IDX_TO_BASE[5] = {'A', 'C', 'G', 'T', 'N'};

// ── htslib 4-bit bam_seqi code -> 0-3 index ────────────────────────────────

// htslib bam_seqi() returns 1=A, 2=C, 4=G, 8=T, 15=N. Map to 0-3 index;
// -1 for everything that is not a pure A/C/G/T base.
inline constexpr int8_t BASE_TO_IDX[16] = {
    -1,  // 0
     0,  // 1  = A
     1,  // 2  = C
    -1,  // 3
     2,  // 4  = G
    -1, -1, -1,
     3,  // 8  = T
    -1, -1, -1, -1, -1, -1,
    -1   // 15 = N
};

// ── ASCII char -> 0-3 index (4 = N / non-ACGT) ─────────────────────────────

// ASCII A/a->0, C/c->1, G/g->2, T/t->3, anything else -> 4 (N).
inline uint8_t ascii_to_idx(char c) {
    switch (c | 32) {
        case 'a': return 0;
        case 'c': return 1;
        case 'g': return 2;
        case 't': return 3;
        default:  return 4;
    }
}

}  // namespace nt
}  // namespace singlet::pileup
