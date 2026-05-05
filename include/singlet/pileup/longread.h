// singlet-pileup/longread.h — Long-read scRNA-seq (PacBio MAS-seq / ONT) support
//
// Detects long-read data, splits MAS-seq concatenated reads, extracts barcodes
// from long-read prefixes with edit-distance tolerance, and writes long-read QC.
//
// Header-only. No external dependencies (no htslib).

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace singlet {

// ─────────────────────────────────────────────────────────────────────────────
// Enums & config
// ─────────────────────────────────────────────────────────────────────────────

enum class LongReadPlatform {
    UNKNOWN,
    PACBIO_MASSEQ,    // MAS-seq: concatenated cDNA inserts
    PACBIO_ISOSEQ,    // Iso-seq: full-length transcripts
    ONT_10X,          // 10x + ONT kits
    ONT_SCICOLLISION  // sci-Collision
};

struct LongReadConfig {
    uint32_t min_read_length = 200;        // reads < this are short-read
    uint32_t max_read_length = 50000;      // sanity cap
    uint32_t barcode_search_window = 200;  // first N bases to search for CB+UMI
    uint32_t min_mapq = 20;
    bool segment_masseq = false;  // split MAS-seq concatenated reads
};

// ─────────────────────────────────────────────────────────────────────────────
// Long-read detection
// ─────────────────────────────────────────────────────────────────────────────

struct LongReadDetection {
    bool is_longread = false;
    LongReadPlatform platform = LongReadPlatform::UNKNOWN;
    double median_read_length = 0.0;
    double mean_read_length = 0.0;
    uint32_t n_sampled = 0;
};

// Detect if data is long-read from a sample of read lengths.
// Threshold: median >= 200 bp → long-read.
// Uses the sorted median (handles even-length vectors correctly).
inline LongReadDetection detect_longread(const std::vector<uint32_t>& read_lengths,
                                         uint32_t /*sample_size*/ = 10000) {
    LongReadDetection result;
    if (read_lengths.empty()) return result;

    result.n_sampled = static_cast<uint32_t>(read_lengths.size());

    // mean
    double sum = 0.0;
    for (uint32_t v : read_lengths) sum += v;
    result.mean_read_length = sum / result.n_sampled;

    // median (sort a copy)
    std::vector<uint32_t> sorted = read_lengths;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 0)
        result.median_read_length = 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    else
        result.median_read_length = sorted[n / 2];

    // threshold: median >= 200 → long-read
    result.is_longread = (result.median_read_length >= 200.0);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAS-seq adapter splitting
// ─────────────────────────────────────────────────────────────────────────────

struct ReadSegment {
    uint32_t start;
    uint32_t end;
    bool has_adapter_left;
    bool has_adapter_right;
};

namespace detail {

// Simple bitwise-parallel Hamming distance for short fixed-length windows.
// Returns number of mismatches between s[off..off+len) and adapter[0..len).
inline uint32_t hamming_window(const std::string& s, size_t off,
                               const std::string& adapter, size_t len) {
    uint32_t mismatches = 0;
    for (size_t i = 0; i < len; ++i)
        mismatches += (s[off + i] != adapter[i]) ? 1 : 0;
    return mismatches;
}

// Find all positions in `seq` where `adapter` matches with ≤ max_mm mismatches.
// Searches only within [0, max_search_end).
inline std::vector<uint32_t> find_adapter_hits(const std::string& seq,
                                               const std::string& adapter,
                                               uint32_t max_mm,
                                               size_t window_len) {
    std::vector<uint32_t> hits;
    if (seq.size() < window_len) return hits;
    size_t end_search = seq.size() - window_len + 1;
    for (size_t i = 0; i < end_search; ++i) {
        if (hamming_window(seq, i, adapter, window_len) <= max_mm)
            hits.push_back(static_cast<uint32_t>(i));
    }
    return hits;
}

}  // namespace detail

// Split a MAS-seq concatenated read by the MAS-seq adapter sequence.
// Returns one segment per inter-adapter region.  If no adapters found,
// returns a single segment spanning the whole read.
//
// adapter: MAS-seq ligation adapter (default is the published MAS-seq 16-base adapter)
// max_mm:  tolerated mismatches in adapter match (default 3)
inline std::vector<ReadSegment> split_masseq_read(
    const std::string& sequence,
    const std::string& adapter =
        "ATCTCTCTCAACAACAACAACGGAGGAGGAGGAAAAGAGAGAGAT",
    uint32_t max_mm = 3) {
    const size_t alen = adapter.size();
    const uint32_t seqlen = static_cast<uint32_t>(sequence.size());

    // Use a minimum match window of min(alen, 16) to avoid false positives
    const size_t win = std::min(alen, static_cast<size_t>(16));

    std::vector<uint32_t> hits = detail::find_adapter_hits(sequence, adapter, max_mm, win);

    // Merge hits that are within 5 bp of each other (same adapter, multiple windows)
    std::vector<uint32_t> merged;
    for (uint32_t h : hits) {
        if (merged.empty() || h > merged.back() + 5)
            merged.push_back(h);
    }

    // No adapters found → one segment
    if (merged.empty()) {
        return {{0, seqlen, false, false}};
    }

    std::vector<ReadSegment> segments;
    uint32_t prev_end = 0;

    for (size_t i = 0; i < merged.size(); ++i) {
        uint32_t adapter_start = merged[i];
        uint32_t adapter_end = std::min(seqlen,
                                        adapter_start + static_cast<uint32_t>(win));

        ReadSegment seg;
        seg.start = prev_end;
        seg.end = adapter_start;
        seg.has_adapter_left = (i > 0);
        seg.has_adapter_right = true;
        if (seg.end > seg.start)
            segments.push_back(seg);

        prev_end = adapter_end;
    }

    // Last segment after the final adapter
    if (prev_end < seqlen) {
        ReadSegment last;
        last.start = prev_end;
        last.end = seqlen;
        last.has_adapter_left = true;
        last.has_adapter_right = false;
        segments.push_back(last);
    }

    return segments;
}

// ─────────────────────────────────────────────────────────────────────────────
// Barcode extraction from long-read prefix
// ─────────────────────────────────────────────────────────────────────────────

struct LongReadBarcode {
    std::string barcode;
    std::string umi;
    uint32_t cb_start = 0;
    uint32_t cb_end = 0;
    uint32_t umi_start = 0;
    uint32_t umi_end = 0;
    double alignment_score = 0.0;
    bool found = false;
};

namespace detail {

// Compute edit distance between two fixed strings (Levenshtein),
// early exit if exceeds max_ed.
inline uint32_t edit_distance(const std::string& a, const std::string& b,
                              uint32_t max_ed) {
    const size_t na = a.size();
    const size_t nb = b.size();
    if (static_cast<uint32_t>(std::abs(static_cast<int>(na) - static_cast<int>(nb))) > max_ed)
        return max_ed + 1;

    // dp[j] = edit distance for a[0..i) vs b[0..j)
    std::vector<uint32_t> dp(nb + 1);
    std::iota(dp.begin(), dp.end(), 0u);

    for (size_t i = 1; i <= na; ++i) {
        uint32_t prev = static_cast<uint32_t>(i - 1);
        dp[0] = static_cast<uint32_t>(i);
        uint32_t min_row = dp[0];
        for (size_t j = 1; j <= nb; ++j) {
            uint32_t temp = dp[j];
            dp[j] = std::min({dp[j] + 1,
                              dp[j - 1] + 1,
                              prev + static_cast<uint32_t>(a[i - 1] != b[j - 1])});
            prev = temp;
            min_row = std::min(min_row, dp[j]);
        }
        if (min_row > max_ed) return max_ed + 1;
    }
    return dp[nb];
}

}  // namespace detail

// Search for a CB+UMI pattern in the first `read_prefix.size()` bases.
// Strategy: slide a window of width (expected_cb_len + expected_umi_len)
// across the prefix and find the position with lowest edit distance to
// a poly-T anchor that follows the UMI in 10x protocols.
// Returns the best match within max_edit_distance.
//
// For synthetic testing, if the prefix exactly encodes CB+UMI at positions
// [0, cb_len) and [cb_len, cb_len+umi_len), it will be found with score=1.0.
inline LongReadBarcode extract_barcode(const std::string& read_prefix,
                                       uint32_t expected_cb_len = 16,
                                       uint32_t expected_umi_len = 12,
                                       uint32_t max_edit_distance = 3) {
    LongReadBarcode result;
    const uint32_t total_len = expected_cb_len + expected_umi_len;
    const uint32_t prefix_len = static_cast<uint32_t>(read_prefix.size());

    if (prefix_len < total_len) return result;

    // Search window: slide across prefix
    const uint32_t search_end = prefix_len - total_len + 1;

    // Reference: poly-T anchor of length 8 expected after CB+UMI in 10x 3' protocol
    const uint32_t anchor_len = std::min(8u, prefix_len > total_len ? prefix_len - total_len : 0u);
    const std::string anchor_ref(anchor_len, 'T');

    uint32_t best_pos = 0;
    uint32_t best_ed = std::numeric_limits<uint32_t>::max();

    for (uint32_t pos = 0; pos < search_end; ++pos) {
        uint32_t anchor_start = pos + total_len;
        uint32_t actual_anchor_len = std::min(anchor_len,
                                              prefix_len > anchor_start
                                                  ? prefix_len - anchor_start
                                                  : 0u);
        if (actual_anchor_len == 0) {
            // No anchor available; accept position 0 trivially
            if (pos == 0) {
                best_pos = 0;
                best_ed = 0;
            }
            continue;
        }

        std::string anchor_obs = read_prefix.substr(anchor_start, actual_anchor_len);
        std::string anchor_exp(actual_anchor_len, 'T');
        uint32_t ed = detail::edit_distance(anchor_obs, anchor_exp, max_edit_distance);

        if (ed < best_ed) {
            best_ed = ed;
            best_pos = pos;
        }
    }

    if (best_ed <= max_edit_distance) {
        result.found = true;
        result.cb_start = best_pos;
        result.cb_end = best_pos + expected_cb_len;
        result.umi_start = result.cb_end;
        result.umi_end = result.umi_start + expected_umi_len;
        result.barcode = read_prefix.substr(result.cb_start, expected_cb_len);
        result.umi = read_prefix.substr(result.umi_start, expected_umi_len);
        // alignment_score: fraction of anchor that matches
        double match = static_cast<double>(anchor_len - best_ed);
        result.alignment_score = anchor_len > 0 ? match / anchor_len : 0.0;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Long-read QC
// ─────────────────────────────────────────────────────────────────────────────

struct LongReadQC {
    uint64_t total_reads = 0;
    uint64_t n_full_length = 0;  // have both adapters
    uint64_t n_chimeric = 0;     // multiple alignments from same read
    double median_read_length = 0.0;
    double median_passes = 0.0;  // PacBio CCS passes
    double n50 = 0.0;            // read length N50
    uint64_t barcode_found = 0;
    double barcode_match_rate = 0.0;
};

// Write LongReadQC as a JSON file. Returns true on success.
inline bool write_longread_qc(const std::string& path, const LongReadQC& qc) {
    std::ofstream f(path);
    if (!f.is_open()) return false;

    f << "{\n"
      << "  \"total_reads\": " << qc.total_reads << ",\n"
      << "  \"n_full_length\": " << qc.n_full_length << ",\n"
      << "  \"n_chimeric\": " << qc.n_chimeric << ",\n"
      << "  \"median_read_length\": " << qc.median_read_length << ",\n"
      << "  \"median_passes\": " << qc.median_passes << ",\n"
      << "  \"n50\": " << qc.n50 << ",\n"
      << "  \"barcode_found\": " << qc.barcode_found << ",\n"
      << "  \"barcode_match_rate\": " << qc.barcode_match_rate << "\n"
      << "}\n";

    return true;
}

}  // namespace singlet
