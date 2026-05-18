// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>

// InDrop barcode layout detection
// Handles v1/v2/v3 variants with different barcode positions.
//
// Key bug fix (VAL1-INDROP-VARIANT):
//   SRR5945694 has InDrop v1 where UMI starts at position 41 but R1 is only 35bp.
//   The encoder stored wrong cblen/umilen.  detect_indrop_version() checks whether
//   the claimed positions exceed read length and falls back to version detection.

enum class InDropVersion { V1,
                           V2,
                           V3,
                           UNKNOWN };

// InDrop version detection heuristic:
//   v1: 22bp minimum R1 (8+8+6 CB1+CB2+UMI, no linker), cblen ~16, umilen ~6
//   v2: R1 very short (~8 bp) — only CB1; R2 carries CB2+UMI
//   v3: R1 ≥ 26bp with linker between CB1 and CB2 (like 10x layout on R2)
//
// The key detection case: when claimed CB+UMI end position > r1_length, the
// stored layout is wrong.  Fall back based purely on r1_length.
inline InDropVersion detect_indrop_version(int r1_length,
                                           int claimed_cblen,
                                           int claimed_umilen) {
    // If claimed positions fit within R1, use cblen as the primary signal
    int claimed_end = claimed_cblen + claimed_umilen;
    if (claimed_end <= r1_length) {
        if (claimed_cblen <= 8 && r1_length <= 12)
            return InDropVersion::V2;  // CB1-only short read
        if (claimed_cblen == 16 || claimed_cblen == 14)
            return InDropVersion::V1;  // 8+8 or 7+7 CB1+CB2
        if (claimed_cblen >= 24)
            return InDropVersion::V3;
    }
    // Claimed layout exceeds read length — fall back to r1_length heuristic
    if (r1_length <= 12)
        return InDropVersion::V2;
    if (r1_length <= 35)
        return InDropVersion::V1;
    if (r1_length >= 26)
        return InDropVersion::V3;
    return InDropVersion::UNKNOWN;
}

struct InDropLayout {
    int cb1_start, cb1_len;
    int cb2_start, cb2_len;
    int umi_start, umi_len;
    bool split_reads;  // true if CB1 in R1 and CB2 in R2
};

// Canonical layout for each InDrop version.
inline InDropLayout get_indrop_layout(InDropVersion version) {
    switch (version) {
        case InDropVersion::V1:
            // BC1(8) + BC2(8) + UMI(6) on R1 — all in one read
            return {0, 8, 8, 8, 16, 6, false};
        case InDropVersion::V2:
            // BC1(8) on R1, BC2(8)+UMI(6) on R2 — split across reads
            return {0, 8, 0, 8, 8, 6, true};
        case InDropVersion::V3:
            // BC1(8) + linker(22) + BC2(8) + UMI(6) on R2 — like 10x
            return {0, 8, 30, 8, 38, 6, false};
        default:
            return {0, 0, 0, 0, 0, 0, false};
    }
}
