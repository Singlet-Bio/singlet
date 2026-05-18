// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: wgs_assembly.h — G-WGS
// WGS/de novo assembly QC, coverage analysis, and subprocess management.
// singlet handles data routing and quality assessment; actual assembly
// is delegated to external tools (hifiasm, flye, SPAdes, MEGAHIT).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace singlet {

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

enum class AssemblerBackend { HIFIASM,
                              FLYE,
                              SPADES,
                              MEGAHIT,
                              NONE };

enum class WgsMode { SHORT_READ,
                     LONG_READ_HIFI,
                     LONG_READ_ONT,
                     HYBRID };

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct AssemblyStats {
    uint64_t total_contigs = 0;
    uint64_t total_bases = 0;
    uint64_t n50 = 0;
    uint64_t n90 = 0;
    uint64_t largest_contig = 0;
    uint64_t smallest_contig = 0;
    double gc_content = 0.0;          // fraction [0,1]
    double estimated_coverage = 0.0;  // reads × avg_read_len / assembly_bases
    uint64_t reads_used = 0;
    uint64_t reads_total = 0;
};

struct ContigInfo {
    std::string name;
    uint64_t length = 0;
    double gc = 0.0;             // fraction [0,1]
    double mean_coverage = 0.0;  // from BAM alignment, 0 if unavailable
};

// ---------------------------------------------------------------------------
// 1. WGS mode detection
// ---------------------------------------------------------------------------

/// Detect WGS sequencing mode from read characteristics.
/// @param median_read_length  Median read length in bases
/// @param has_barcode         True if cell barcodes were detected
/// @param has_umi             True if UMIs were detected
/// @param instrument          Instrument string (e.g. "illumina", "pacbio", "ont")
inline WgsMode detect_wgs_mode(int median_read_length,
                               bool has_barcode,
                               bool has_umi,
                               const std::string& instrument) {
    // If barcodes or UMIs are present, this is single-cell — not classic WGS
    // Treat as HYBRID (mixed library)
    if (has_barcode || has_umi) {
        return WgsMode::HYBRID;
    }

    const std::string inst_lower = [&]() {
        std::string s = instrument;
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }();

    const bool is_ont = inst_lower.find("ont") != std::string::npos ||
                        inst_lower.find("nanopore") != std::string::npos ||
                        inst_lower.find("oxford") != std::string::npos;

    const bool is_pacbio = inst_lower.find("pacbio") != std::string::npos ||
                           inst_lower.find("hifi") != std::string::npos ||
                           inst_lower.find("sequel") != std::string::npos ||
                           inst_lower.find("revio") != std::string::npos;

    if (median_read_length > 1000) {
        if (is_ont) return WgsMode::LONG_READ_ONT;
        // PacBio HiFi or assumed HiFi for generic >1000bp
        return WgsMode::LONG_READ_HIFI;
    }

    // Short reads (<= 1000 bp) — check for mixed signals
    if (is_ont || is_pacbio) {
        return WgsMode::HYBRID;
    }

    return WgsMode::SHORT_READ;
}

// ---------------------------------------------------------------------------
// 2. Assembler selection
// ---------------------------------------------------------------------------

/// Select the default assembler for the given WGS mode.
inline AssemblerBackend select_assembler(WgsMode mode) {
    switch (mode) {
        case WgsMode::SHORT_READ:
            return AssemblerBackend::SPADES;
        case WgsMode::LONG_READ_HIFI:
            return AssemblerBackend::HIFIASM;
        case WgsMode::LONG_READ_ONT:
            return AssemblerBackend::FLYE;
        case WgsMode::HYBRID:
            return AssemblerBackend::SPADES;
    }
    return AssemblerBackend::NONE;
}

// ---------------------------------------------------------------------------
// 3. Command construction
// ---------------------------------------------------------------------------

/// Build the full assembler command line.
/// @param backend      Assembler to invoke
/// @param input_fastq  Path to input FASTQ (may be comma-separated for paired)
/// @param output_dir   Output directory
/// @param threads      Number of threads
/// @param memory_gb    Memory limit in GB (used where supported)
/// @return Shell command string ready for std::system() or subprocess
inline std::string build_assembler_command(AssemblerBackend backend,
                                           const std::string& input_fastq,
                                           const std::string& output_dir,
                                           int threads,
                                           uint64_t memory_gb) {
    std::ostringstream cmd;
    switch (backend) {
        case AssemblerBackend::HIFIASM:
            cmd << "hifiasm -o " << output_dir << "/asm"
                << " -t " << threads
                << " " << input_fastq;
            break;

        case AssemblerBackend::FLYE:
            cmd << "flye --nano-raw " << input_fastq
                << " --out-dir " << output_dir
                << " --threads " << threads;
            break;

        case AssemblerBackend::SPADES:
            cmd << "spades.py -s " << input_fastq
                << " -o " << output_dir
                << " -t " << threads
                << " -m " << memory_gb;
            break;

        case AssemblerBackend::MEGAHIT:
            cmd << "megahit -r " << input_fastq
                << " -o " << output_dir
                << " -t " << threads;
            break;

        case AssemblerBackend::NONE:
        default:
            break;
    }
    return cmd.str();
}

// ---------------------------------------------------------------------------
// 4. FASTA parsing
// ---------------------------------------------------------------------------

namespace detail {
inline double compute_gc(const std::string& seq) {
    if (seq.empty()) return 0.0;
    uint64_t gc = 0;
    for (char c : seq) {
        const char u = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (u == 'G' || u == 'C') ++gc;
    }
    return static_cast<double>(gc) / static_cast<double>(seq.size());
}
}  // namespace detail

/// Parse a multi-FASTA string and return per-contig info.
inline std::vector<ContigInfo> parse_assembly_fasta(std::string_view fasta_content) {
    std::vector<ContigInfo> contigs;
    ContigInfo current;
    std::string seq_buf;
    bool in_seq = false;

    auto flush = [&]() {
        if (in_seq && !current.name.empty()) {
            current.length = static_cast<uint64_t>(seq_buf.size());
            current.gc = detail::compute_gc(seq_buf);
            contigs.push_back(current);
        }
        current = ContigInfo{};
        seq_buf.clear();
    };

    std::string_view remaining = fasta_content;
    while (!remaining.empty()) {
        const auto newline = remaining.find('\n');
        const std::string_view line = (newline == std::string_view::npos)
                                          ? remaining
                                          : remaining.substr(0, newline);

        if (!line.empty() && line[0] == '>') {
            flush();
            in_seq = true;
            // Name is everything after '>' up to first whitespace
            const auto sp = line.find(' ');
            current.name = std::string(line.substr(1, sp == std::string_view::npos
                                                          ? std::string_view::npos
                                                          : sp - 1));
        } else if (in_seq) {
            seq_buf.append(line.data(), line.size());
        }

        remaining = (newline == std::string_view::npos)
                        ? std::string_view{}
                        : remaining.substr(newline + 1);
    }
    flush();
    return contigs;
}

// ---------------------------------------------------------------------------
// 5. Assembly statistics
// ---------------------------------------------------------------------------

/// Compute assembly-level statistics from a list of ContigInfo entries.
inline AssemblyStats compute_assembly_stats(const std::vector<ContigInfo>& contigs,
                                            uint64_t total_reads,
                                            int avg_read_length) {
    AssemblyStats stats;
    stats.reads_total = total_reads;
    stats.reads_used = total_reads;  // default: all reads used
    stats.total_contigs = static_cast<uint64_t>(contigs.size());

    if (contigs.empty()) return stats;

    // Collect lengths and compute totals
    std::vector<uint64_t> lengths;
    lengths.reserve(contigs.size());
    uint64_t total_gc_bases = 0;
    uint64_t total_bases = 0;

    for (const auto& c : contigs) {
        lengths.push_back(c.length);
        total_bases += c.length;
        // Weighted GC
        total_gc_bases += static_cast<uint64_t>(std::round(c.gc * static_cast<double>(c.length)));
    }

    stats.total_bases = total_bases;
    stats.gc_content = (total_bases > 0)
                           ? static_cast<double>(total_gc_bases) / static_cast<double>(total_bases)
                           : 0.0;

    // Sort descending for N50/N90
    std::sort(lengths.begin(), lengths.end(), std::greater<uint64_t>());
    stats.largest_contig = lengths.front();
    stats.smallest_contig = lengths.back();

    // N50: length of contig at which cumulative length ≥ 50% of total
    const uint64_t half = total_bases / 2;
    const uint64_t ninety = (total_bases * 9) / 10;
    uint64_t cumsum = 0;
    bool found_n50 = false;
    bool found_n90 = false;
    for (uint64_t l : lengths) {
        cumsum += l;
        if (!found_n50 && cumsum >= half) {
            stats.n50 = l;
            found_n50 = true;
        }
        if (!found_n90 && cumsum >= ninety) {
            stats.n90 = l;
            found_n90 = true;
        }
        if (found_n50 && found_n90) break;
    }

    // Estimated coverage
    if (total_bases > 0) {
        stats.estimated_coverage =
            static_cast<double>(total_reads) * static_cast<double>(avg_read_length) /
            static_cast<double>(total_bases);
    }

    return stats;
}

// ---------------------------------------------------------------------------
// 6. JSON output
// ---------------------------------------------------------------------------

/// Write assembly stats as JSON to outpath.
inline void write_assembly_stats(const AssemblyStats& stats, std::string_view outpath) {
    std::string path_str{outpath};
    std::ofstream f(path_str);
    f << "{\n"
      << "  \"total_contigs\": " << stats.total_contigs << ",\n"
      << "  \"total_bases\": " << stats.total_bases << ",\n"
      << "  \"n50\": " << stats.n50 << ",\n"
      << "  \"n90\": " << stats.n90 << ",\n"
      << "  \"largest_contig\": " << stats.largest_contig << ",\n"
      << "  \"smallest_contig\": " << stats.smallest_contig << ",\n"
      << "  \"gc_content\": " << stats.gc_content << ",\n"
      << "  \"estimated_coverage\": " << stats.estimated_coverage << ",\n"
      << "  \"reads_used\": " << stats.reads_used << ",\n"
      << "  \"reads_total\": " << stats.reads_total << "\n"
      << "}\n";
}

// ---------------------------------------------------------------------------
// 7. TSV report
// ---------------------------------------------------------------------------

/// Write per-contig assembly report as TSV to outpath.
inline void write_assembly_report(const AssemblyStats& stats,
                                  const std::vector<ContigInfo>& contigs,
                                  std::string_view outpath) {
    (void)stats;  // stats may be used by caller; suppress unused warning
    std::string path_str{outpath};
    std::ofstream f(path_str);
    f << "name\tlength\tgc\tmean_coverage\n";
    for (const auto& c : contigs) {
        f << c.name << "\t" << c.length << "\t"
          << c.gc << "\t" << c.mean_coverage << "\n";
    }
}

// ---------------------------------------------------------------------------
// 8. Genome size estimation from k-mer histogram
// ---------------------------------------------------------------------------

/// Estimate genome size from a k-mer frequency histogram using the
/// GenomeScope heuristic: genome_size ≈ Σ(freq × count) / peak_freq.
/// @param kmer_histogram  kmer_histogram[freq] = number of k-mers with that frequency.
///                        Index 0 is ignored (freq=0 is not meaningful).
/// @return Estimated haploid genome size in bases, or 0 if histogram is empty.
inline uint64_t estimate_genome_size_from_kmers(const std::vector<uint64_t>& kmer_histogram) {
    if (kmer_histogram.size() < 2) return 0;

    // Find the peak frequency (argmax), skipping index 0 and low-freq error k-mers
    // Convention: skip index 1 (error k-mers) to find the true peak.
    const size_t start = std::min<size_t>(2, kmer_histogram.size() - 1);
    size_t peak_freq = start;
    for (size_t i = start; i < kmer_histogram.size(); ++i) {
        if (kmer_histogram[i] > kmer_histogram[peak_freq]) {
            peak_freq = i;
        }
    }

    if (peak_freq == 0 || kmer_histogram[peak_freq] == 0) return 0;

    // Total k-mer count (sum of freq × count), excluding index 0
    uint64_t total = 0;
    for (size_t i = 1; i < kmer_histogram.size(); ++i) {
        total += static_cast<uint64_t>(i) * kmer_histogram[i];
    }

    return total / static_cast<uint64_t>(peak_freq);
}

}  // namespace singlet
