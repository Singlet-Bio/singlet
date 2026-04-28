#pragma once
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet {

struct SLAMSeqParams {
    uint32_t min_tc_conversions = 2;     // minimum T→C conversions to call "new"
    float max_tc_rate_threshold = 0.5f;  // SNP filter: ignore positions with >50% T→C across all reads
    uint32_t min_base_quality = 27;      // minimum base quality at conversion site
};

struct ConversionCount {
    uint32_t tc_count = 0;           // T→C conversions in this read
    uint32_t t_positions = 0;        // total T positions covered (denominator for conversion rate)
    uint32_t other_conversions = 0;  // non-T→C mismatches (background estimation)
};

// Count T→C conversions in a read by comparing to reference.
// read_seq: the aligned read sequence (same length as ref_seq)
// ref_seq: the reference sequence at this position
// base_quals: base quality scores (Phred+33 ASCII, same length as read_seq)
// Strand: on + strand T→C; on - strand A→G (complement of T→C).
// This function handles + strand. For - strand, caller should reverse-complement
// read/ref and call again, or use count_tc_conversions_stranded below.
inline ConversionCount count_tc_conversions(const std::string& read_seq,
                                            const std::string& ref_seq,
                                            const std::string& base_quals,
                                            const SLAMSeqParams& params) {
    if (read_seq.size() != ref_seq.size()) {
        throw std::invalid_argument("read_seq and ref_seq must have the same length");
    }
    if (!base_quals.empty() && base_quals.size() != read_seq.size()) {
        throw std::invalid_argument("base_quals must match read_seq length or be empty");
    }

    ConversionCount result{};
    const size_t n = read_seq.size();
    const uint32_t min_qual_char = static_cast<uint32_t>(params.min_base_quality) + 33u;

    for (size_t i = 0; i < n; ++i) {
        const char r = read_seq[i];
        const char t = ref_seq[i];

        // Quality filter
        if (!base_quals.empty()) {
            if (static_cast<uint32_t>(static_cast<unsigned char>(base_quals[i])) < min_qual_char)
                continue;
        }

        // Count T positions in reference
        if (t == 'T' || t == 't') {
            ++result.t_positions;
            if (r == 'C' || r == 'c') {
                ++result.tc_count;
            } else if ((r != 'T' && r != 't') && r != 'N' && r != 'n') {
                ++result.other_conversions;
            }
        } else if (t != 'N' && t != 'n' && r != 'N' && r != 'n') {
            // Non-T reference mismatch → background
            if (r != t) {
                ++result.other_conversions;
            }
        }
    }
    return result;
}

// Count T→C on both strands (+ strand: T→C; - strand: A→G in read vs A in ref).
// is_reverse: whether the read mapped to the minus strand.
inline ConversionCount count_tc_conversions_stranded(const std::string& read_seq,
                                                     const std::string& ref_seq,
                                                     const std::string& base_quals,
                                                     const SLAMSeqParams& params,
                                                     bool is_reverse) {
    if (!is_reverse) {
        // + strand: T→C
        return count_tc_conversions(read_seq, ref_seq, base_quals, params);
    }
    // - strand: complement T→C is A→G
    // For a read on the minus strand, the read sequence is already reverse-complemented
    // by the aligner. T in the reference corresponds to A on the minus strand template.
    // A read base of G where the reference (+ strand) is A = T→C on the original molecule.
    if (read_seq.size() != ref_seq.size())
        throw std::invalid_argument("read_seq and ref_seq must have the same length");
    if (!base_quals.empty() && base_quals.size() != read_seq.size())
        throw std::invalid_argument("base_quals must match read_seq length or be empty");

    ConversionCount result{};
    const size_t n = read_seq.size();
    const uint32_t min_qual_char = static_cast<uint32_t>(params.min_base_quality) + 33u;

    for (size_t i = 0; i < n; ++i) {
        const char r = read_seq[i];
        const char t = ref_seq[i];

        if (!base_quals.empty()) {
            if (static_cast<uint32_t>(static_cast<unsigned char>(base_quals[i])) < min_qual_char)
                continue;
        }

        // On minus strand read: A in ref (+ strand) is T on the minus strand template
        if (t == 'A' || t == 'a') {
            ++result.t_positions;
            if (r == 'G' || r == 'g') {
                ++result.tc_count;  // A→G in minus-strand read ≡ T→C on original molecule
            } else if ((r != 'A' && r != 'a') && r != 'N' && r != 'n') {
                ++result.other_conversions;
            }
        } else if (t != 'N' && t != 'n' && r != 'N' && r != 'n') {
            if (r != t) {
                ++result.other_conversions;
            }
        }
    }
    return result;
}

// Classify a read as "new" (labeled s4U-incorporating) RNA
inline bool is_new_rna(const ConversionCount& conv, const SLAMSeqParams& params) {
    return conv.tc_count >= params.min_tc_conversions;
}

// Per-cell SLAM-seq statistics
struct CellSLAMStats {
    uint32_t total_reads = 0;
    uint32_t new_reads = 0;
    uint32_t old_reads = 0;
    float new_fraction = 0.0f;
    float mean_tc_rate = 0.0f;     // mean T→C conversion rate across all reads
    float background_rate = 0.0f;  // mean non-T→C mismatch rate (QC)
};

// Compute per-cell SLAM-seq stats from labeled read classifications.
// per_cell_conversions[cell_idx] = list of ConversionCount for each read in that cell
inline std::vector<CellSLAMStats> compute_slam_stats(
    const std::vector<std::vector<ConversionCount>>& per_cell_conversions,
    const SLAMSeqParams& params) {
    std::vector<CellSLAMStats> stats(per_cell_conversions.size());
    for (size_t ci = 0; ci < per_cell_conversions.size(); ++ci) {
        auto& s = stats[ci];
        const auto& reads = per_cell_conversions[ci];
        s.total_reads = static_cast<uint32_t>(reads.size());

        double sum_tc_rate = 0.0;
        double sum_bg_rate = 0.0;
        uint32_t rate_count = 0;

        for (const auto& conv : reads) {
            if (is_new_rna(conv, params)) {
                ++s.new_reads;
            } else {
                ++s.old_reads;
            }
            if (conv.t_positions > 0) {
                sum_tc_rate += static_cast<double>(conv.tc_count) / conv.t_positions;
                sum_bg_rate += static_cast<double>(conv.other_conversions) / conv.t_positions;
                ++rate_count;
            }
        }

        s.new_fraction = s.total_reads > 0
                             ? static_cast<float>(s.new_reads) / s.total_reads
                             : 0.0f;
        s.mean_tc_rate = rate_count > 0
                             ? static_cast<float>(sum_tc_rate / rate_count)
                             : 0.0f;
        s.background_rate = rate_count > 0
                                ? static_cast<float>(sum_bg_rate / rate_count)
                                : 0.0f;
    }
    return stats;
}

// Write slam_stats.tsv with per-cell statistics
inline bool write_slam_stats_tsv(const std::string& path,
                                 const std::vector<CellSLAMStats>& stats,
                                 const std::vector<std::string>& barcodes) {
    if (stats.size() != barcodes.size()) return false;
    std::ofstream ofs(path);
    if (!ofs) return false;
    ofs << "barcode\ttotal_reads\tnew_reads\told_reads\tnew_fraction\tmean_tc_rate\tbackground_rate\n";
    for (size_t i = 0; i < stats.size(); ++i) {
        const auto& s = stats[i];
        ofs << barcodes[i] << '\t'
            << s.total_reads << '\t'
            << s.new_reads << '\t'
            << s.old_reads << '\t'
            << s.new_fraction << '\t'
            << s.mean_tc_rate << '\t'
            << s.background_rate << '\n';
    }
    return ofs.good();
}

// Result structure for per-gene new/total count matrices
struct SLAMCountResult {
    // new RNA gene counts (CSR: genes × cells)
    std::vector<int32_t> new_indptr;
    std::vector<int32_t> new_indices;
    std::vector<uint32_t> new_data;
    // total RNA gene counts (same structure)
    std::vector<int32_t> total_indptr;
    std::vector<int32_t> total_indices;
    std::vector<uint32_t> total_data;
    uint32_t nrows = 0;  // n_genes
    uint32_t ncols = 0;  // n_cells
};

// Build per-gene SLAM count matrices from per-read labels.
// gene_cell_reads[gene_idx][cell_idx] = {new_umis, total_umis}
// Input: gene × cell vector of (new_count, total_count) pairs
inline SLAMCountResult build_slam_count_matrix(
    const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& gene_cell_counts,
    uint32_t n_genes,
    uint32_t n_cells) {
    SLAMCountResult result;
    result.nrows = n_genes;
    result.ncols = n_cells;

    result.new_indptr.resize(n_genes + 1, 0);
    result.total_indptr.resize(n_genes + 1, 0);

    for (uint32_t g = 0; g < n_genes && g < static_cast<uint32_t>(gene_cell_counts.size()); ++g) {
        for (uint32_t c = 0; c < n_cells && c < static_cast<uint32_t>(gene_cell_counts[g].size()); ++c) {
            auto [nw, tot] = gene_cell_counts[g][c];
            if (nw > 0) {
                result.new_indices.push_back(static_cast<int32_t>(c));
                result.new_data.push_back(nw);
                ++result.new_indptr[g + 1];
            }
            if (tot > 0) {
                result.total_indices.push_back(static_cast<int32_t>(c));
                result.total_data.push_back(tot);
                ++result.total_indptr[g + 1];
            }
        }
        result.new_indptr[g + 1] += result.new_indptr[g];
        result.total_indptr[g + 1] += result.total_indptr[g];
    }
    return result;
}

}  // namespace singlet
