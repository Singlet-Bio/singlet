// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

// Seq-Well and Seq-Well S^3 protocol detection helpers.
//
// Seq-Well uses 12bp CB + 8bp UMI on R1, identical to Drop-seq.
// Bug fix (VAL1-SEQWELL-DETECT): SRR32182924 got CBlen=150 because the entire
// R1 was treated as barcode.  validate_seqwell_cblen() rejects implausible lengths.

// Detect Seq-Well from protocol tag (case-insensitive, handles hyphen variants).
inline bool is_seqwell(std::string_view tag) {
    // Normalize to lowercase for comparison
    std::string low(tag);
    std::transform(low.begin(), low.end(), low.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Replace underscores and hyphens with nothing for loose matching
    std::string norm;
    for (char c : low)
        if (c != '-' && c != '_') norm += c;

    return norm == "seqwell" || norm == "seqwells3" || norm == "seqwells^3" || norm == "seqwellseq"
           // Also catch bare "seq well" variants that include "seqwell" substring
           || (norm.rfind("seqwell", 0) == 0);
}

// Seq-Well barcode layout: identical to Drop-seq
struct SeqWellLayout {
    static constexpr int CB_LEN = 12;
    static constexpr int UMI_LEN = 8;
    static constexpr int TOTAL = 20;  // CB + UMI on R1
};

// Validate that claimed CBlen is plausible for Seq-Well.
// Valid: exactly 12 (or nearby, to tolerate minor probe sets).
// Invalid: 0 or ≥ full-read lengths like 150.
inline bool validate_seqwell_cblen(int claimed_cblen) {
    return claimed_cblen >= 10 && claimed_cblen <= 16;
}
