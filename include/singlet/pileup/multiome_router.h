// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "../fq/types.h"

namespace singlet {

enum class MultiomeModality { GEX,
                              ATAC,
                              UNKNOWN };

// Detect modality from a protocol_family string.
// Uses the AssayType enum mapping where available, then falls back to
// substring detection ("atac" → ATAC, otherwise → GEX).
inline MultiomeModality detect_modality(const std::string& protocol_family) {
    // Exact known multiome tags
    if (protocol_family == "10x-arc-gex" ||
        protocol_family == "10x-multiome-gex") {
        return MultiomeModality::GEX;
    }
    if (protocol_family == "10x-arc-atac" ||
        protocol_family == "10x-multiome-atac") {
        return MultiomeModality::ATAC;
    }

    // Substring fallback uses to_lower helper below
    auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return s;
    };
    std::string lc = to_lower(protocol_family);

    // Substring fallback: any protocol containing "atac" → ATAC
    if (lc.find("atac") != std::string::npos) {
        return MultiomeModality::ATAC;
    }

    // Everything else defaults to GEX (RNA-seq protocols)
    if (!protocol_family.empty()) {
        return MultiomeModality::GEX;
    }

    return MultiomeModality::UNKNOWN;
}

// Returns true when the protocol is part of a 10x Multiome experiment
// (either GEX or ATAC component).
inline bool is_multiome_protocol(const std::string& protocol_family) {
    auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return s;
    };
    std::string lc = to_lower(protocol_family);

    // Explicit multiome tags
    if (lc == "10x-arc-gex" || lc == "10x-arc-atac" ||
        lc == "10x-multiome-gex" || lc == "10x-multiome-atac" ||
        lc == "10xmultiome" || lc == "10xarcmultiome" ||
        lc == "multiome") {
        return true;
    }
    // "arc" in the tag strongly implies multiome
    if (lc.find("arc") != std::string::npos) {
        return true;
    }
    return false;
}

// Holder for classified .1fq inputs, split by modality.
struct MultiomeInputs {
    std::vector<std::string> gex_inputs;   // GEX .1fq files
    std::vector<std::string> atac_inputs;  // ATAC .1fq files
    bool is_multiome = false;              // true when both modalities present
};

// Read the protocol_family from a .1fq file header without decoding reads.
// Returns empty string on failure.
inline std::string peek_protocol_family(const std::string& path) {
    // The .1fq Header stores assay_type (byte 57 of the fixed header).
    // We reconstruct the protocol_family string from AssayType.
    // Open the file and read just enough for the fixed header.
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "";

    // .1fq header layout: 8-byte magic + version(4) + flags(4) + n_reads(4) +
    // stream_lengths[8](16) + extra(20) + assay_type(1) = byte 57 (0-indexed)
    // Total fixed header = at minimum 64 bytes.
    uint8_t buf[64] = {};
    size_t n = std::fread(buf, 1, sizeof(buf), f);
    std::fclose(f);
    if (n < 58) return "";

    auto at = static_cast<singlet::fq::AssayType>(buf[57]);
    switch (at) {
        case singlet::fq::AssayType::SC_MULTIOME_GEX:
            return "10x-arc-gex";
        case singlet::fq::AssayType::SC_MULTIOME_ATAC:
            return "10x-arc-atac";
        case singlet::fq::AssayType::SC_ATAC:
            return "10x-atac";
        case singlet::fq::AssayType::SC_RNA_3PRIME:
            return "10x-3p-v3";
        case singlet::fq::AssayType::SC_RNA_5PRIME:
            return "10x-5p-v2";
        case singlet::fq::AssayType::SPATIAL_RNA:
            return "10x-visium";
        case singlet::fq::AssayType::BULK_RNA:
            return "bulk-rna";
        default:
            return "";
    }
}

// Classify a list of .1fq input files by modality.
// Each file's protocol_family is peeked from its header.
inline MultiomeInputs classify_inputs(const std::vector<std::string>& input_files) {
    MultiomeInputs result;
    for (const auto& path : input_files) {
        std::string pf = peek_protocol_family(path);
        MultiomeModality mod = detect_modality(pf);
        if (mod == MultiomeModality::ATAC) {
            result.atac_inputs.push_back(path);
        } else {
            // GEX or UNKNOWN — treat as GEX (most common default)
            result.gex_inputs.push_back(path);
        }
    }
    result.is_multiome = !result.gex_inputs.empty() && !result.atac_inputs.empty();
    return result;
}

// Route configuration for a multiome run.
struct MultiomeConfig {
    bool run_gex = false;
    bool run_atac = false;
    std::string gex_output_prefix;   // e.g., "out/gex/"
    std::string atac_output_prefix;  // e.g., "out/atac/"
};

// Plan a multiome (or single-modality) run given the classified inputs and
// the user-supplied output prefix.  Sub-directories gex/ and atac/ are
// appended only when both modalities are present; otherwise the prefix is
// used as-is to preserve backward compatibility.
inline MultiomeConfig plan_multiome_run(const MultiomeInputs& inputs,
                                        const std::string& output_prefix) {
    MultiomeConfig cfg;
    cfg.run_gex = !inputs.gex_inputs.empty();
    cfg.run_atac = !inputs.atac_inputs.empty();

    // Ensure prefix ends with '/' for clean sub-path construction
    std::string base = output_prefix;
    if (!base.empty() && base.back() != '/') base += '/';

    if (inputs.is_multiome) {
        cfg.gex_output_prefix = base + "gex/";
        cfg.atac_output_prefix = base + "atac/";
    } else {
        cfg.gex_output_prefix = base;
        cfg.atac_output_prefix = base;
    }
    return cfg;
}

}  // namespace singlet
