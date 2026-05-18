// SPDX-License-Identifier: MIT
#pragma once
// G-HIC: Hi-C / Micro-C contact pair extraction and QC
// 4DN Nucleome Network .pairs format compliant
// No htslib dependency — works on pre-aligned ContactPair structs

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet {

struct ContactPair {
    std::string chrom1;
    uint32_t pos1 = 0;
    bool strand1 = true;  // true = +
    std::string chrom2;
    uint32_t pos2 = 0;
    bool strand2 = true;
    uint32_t mapq1 = 0;
    uint32_t mapq2 = 0;
    std::string barcode;  // for scHi-C, empty for bulk
};

enum class ContactType {
    CIS_SHORT,      // same chrom, <20kb
    CIS_LONG,       // same chrom, >=20kb
    TRANS,          // different chrom
    SELF_LIGATION,  // same fragment (chrom equal, distance < 1kb, strands facing outward)
    DANGLING_END,   // one end of restriction fragment (same frag, strands facing inward)
    INVALID         // unmapped, low quality, etc.
};

struct HicQC {
    uint64_t total_pairs = 0;
    uint64_t valid_pairs = 0;
    uint64_t cis_short = 0;  // <20kb
    uint64_t cis_long = 0;   // >=20kb
    uint64_t trans = 0;
    uint64_t self_ligation = 0;
    uint64_t dangling_end = 0;
    uint64_t low_mapq = 0;
    uint64_t duplicate = 0;
    double valid_fraction = 0.0;
    double cis_trans_ratio = 0.0;
    double long_range_fraction = 0.0;  // cis_long / (cis_long + cis_short)
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace detail {

// Distance between two positions on the same chromosome
inline uint64_t pair_distance(uint32_t pos1, uint32_t pos2) {
    return (pos1 > pos2) ? (uint64_t)(pos1 - pos2) : (uint64_t)(pos2 - pos1);
}

// Self-ligation: same chrom, very close (<1 kb), strands pointing away from
// each other (+ on left, - on right when pos1 < pos2, or vice versa — classic
// "outward-facing" ligation artefact).
inline bool is_self_ligation(const ContactPair& p) {
    if (p.chrom1 != p.chrom2) return false;
    uint64_t dist = pair_distance(p.pos1, p.pos2);
    if (dist >= 1000) return false;
    // Outward-facing: if pos1 < pos2, strand1==false(-) and strand2==true(+)
    //                 if pos1 > pos2, strand1==true(+) and strand2==false(-)
    //                 if pos1 == pos2, treat as self-ligation too
    if (p.pos1 < p.pos2) return (!p.strand1 && p.strand2);
    if (p.pos1 > p.pos2) return (p.strand1 && !p.strand2);
    return true;
}

// Dangling end: same fragment, strands face inward (typical R1-R2 on same RE
// fragment without a ligation event).
// Inward-facing: if pos1 < pos2, strand1==true(+) and strand2==false(-)
inline bool is_dangling_end(const ContactPair& p) {
    if (p.chrom1 != p.chrom2) return false;
    uint64_t dist = pair_distance(p.pos1, p.pos2);
    if (dist >= 1000) return false;
    if (p.pos1 < p.pos2) return (p.strand1 && !p.strand2);
    if (p.pos1 > p.pos2) return (!p.strand1 && p.strand2);
    return false;
}

inline char strand_char(bool strand) { return strand ? '+' : '-'; }

}  // namespace detail

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Classify a single contact pair.
inline ContactType classify_contact(const ContactPair& pair,
                                    uint32_t min_mapq = 10,
                                    uint32_t cis_threshold = 20000) {
    // Quality filter first
    if (pair.mapq1 < min_mapq || pair.mapq2 < min_mapq) {
        return ContactType::INVALID;
    }
    // Artefact classification (before distance-based cis/trans)
    if (pair.chrom1 == pair.chrom2) {
        if (detail::is_self_ligation(pair)) return ContactType::SELF_LIGATION;
        if (detail::is_dangling_end(pair)) return ContactType::DANGLING_END;
        uint64_t dist = detail::pair_distance(pair.pos1, pair.pos2);
        return (dist < cis_threshold) ? ContactType::CIS_SHORT : ContactType::CIS_LONG;
    }
    return ContactType::TRANS;
}

/// Compute Hi-C QC stats from a vector of ContactPairs.
inline HicQC compute_hic_qc(const std::vector<ContactPair>& pairs,
                            uint32_t min_mapq = 10,
                            uint32_t cis_threshold = 20000) {
    HicQC qc;
    qc.total_pairs = pairs.size();

    for (const auto& p : pairs) {
        ContactType ct = classify_contact(p, min_mapq, cis_threshold);
        switch (ct) {
            case ContactType::CIS_SHORT:
                ++qc.cis_short;
                ++qc.valid_pairs;
                break;
            case ContactType::CIS_LONG:
                ++qc.cis_long;
                ++qc.valid_pairs;
                break;
            case ContactType::TRANS:
                ++qc.trans;
                ++qc.valid_pairs;
                break;
            case ContactType::SELF_LIGATION:
                ++qc.self_ligation;
                break;
            case ContactType::DANGLING_END:
                ++qc.dangling_end;
                break;
            case ContactType::INVALID:
                ++qc.low_mapq;
                break;
        }
    }

    if (qc.total_pairs > 0) {
        qc.valid_fraction = static_cast<double>(qc.valid_pairs) /
                            static_cast<double>(qc.total_pairs);
    }

    uint64_t cis_total = qc.cis_short + qc.cis_long;
    if (qc.trans > 0 || cis_total > 0) {
        qc.cis_trans_ratio = (qc.trans > 0)
                                 ? static_cast<double>(cis_total) / static_cast<double>(qc.trans)
                                 : std::numeric_limits<double>::infinity();
    }

    if (cis_total > 0) {
        qc.long_range_fraction = static_cast<double>(qc.cis_long) /
                                 static_cast<double>(cis_total);
    }

    return qc;
}

/// Write contacts in 4DN .pairs format.
/// Header lines start with '#'.
/// Columns: readID chr1 pos1 chr2 pos2 strand1 strand2
inline bool write_pairs(const std::string& path,
                        const std::vector<ContactPair>& pairs,
                        const std::vector<std::string>& chrom_names,
                        const std::vector<uint32_t>& chrom_sizes) {
    std::ofstream out(path);
    if (!out.is_open()) return false;

    // 4DN .pairs header
    out << "## pairs format v1.0\n";
    out << "#shape: upper triangle\n";
    for (size_t i = 0; i < chrom_names.size() && i < chrom_sizes.size(); ++i) {
        out << "#chromsize: " << chrom_names[i] << " " << chrom_sizes[i] << "\n";
    }
    out << "#columns: readID chr1 pos1 chr2 pos2 strand1 strand2\n";

    for (size_t i = 0; i < pairs.size(); ++i) {
        const auto& p = pairs[i];
        out << "read" << (i + 1) << "\t"
            << p.chrom1 << "\t" << p.pos1 << "\t"
            << p.chrom2 << "\t" << p.pos2 << "\t"
            << detail::strand_char(p.strand1) << "\t"
            << detail::strand_char(p.strand2) << "\n";
    }
    return out.good();
}

/// Write Hi-C QC metrics to a JSON file.
inline bool write_hic_qc(const std::string& path, const HicQC& qc) {
    std::ofstream out(path);
    if (!out.is_open()) return false;

    auto fmt_double = [](double v) -> std::string {
        if (std::isinf(v)) return "\"inf\"";
        if (std::isnan(v)) return "\"nan\"";
        std::ostringstream ss;
        ss.precision(6);
        ss << std::fixed << v;
        return ss.str();
    };

    out << "{\n"
        << "  \"total_pairs\": " << qc.total_pairs << ",\n"
        << "  \"valid_pairs\": " << qc.valid_pairs << ",\n"
        << "  \"cis_short\": " << qc.cis_short << ",\n"
        << "  \"cis_long\": " << qc.cis_long << ",\n"
        << "  \"trans\": " << qc.trans << ",\n"
        << "  \"self_ligation\": " << qc.self_ligation << ",\n"
        << "  \"dangling_end\": " << qc.dangling_end << ",\n"
        << "  \"low_mapq\": " << qc.low_mapq << ",\n"
        << "  \"duplicate\": " << qc.duplicate << ",\n"
        << "  \"valid_fraction\": " << fmt_double(qc.valid_fraction) << ",\n"
        << "  \"cis_trans_ratio\": " << fmt_double(qc.cis_trans_ratio) << ",\n"
        << "  \"long_range_fraction\": " << fmt_double(qc.long_range_fraction) << "\n"
        << "}\n";

    return out.good();
}

}  // namespace singlet
