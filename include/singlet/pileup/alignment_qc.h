// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: alignment_qc.h
// SS2 + B2: Alignment QC metrics for Smart-seq2 and Bulk RNA-seq.
//
// Shared metrics (no UMIs; position-based deduplication):
//   - Mapping stats: total/mapped/unique/multi/duplicate reads
//   - Mapping rate, duplication rate
//   - Mitochondrial fraction
//   - Gene body coverage profile (100 bins, 5'→3' per strand)
//   - 5'/3' bias ratio (mean of first 10 bins / mean of last 10 bins)
//   - Genes detected (≥ threshold reads)
//   - Median gene coverage
//   - Library complexity via Lander-Waterman estimate
//
// Usage (compute from pre-collected stats — no BAM dependency):
//
//   AlignmentQCComputer qcc;
//   AlignmentQC qc = qcc.compute_from_stats(total, mapped, unique, multi, dup,
//                                            mito_frac, gene_counts, bin_profile);
//   qcc.write_report("qc_report.tsv", qc);
//
// For gene body coverage, call:
//   auto profile = qcc.compute_gene_body_coverage(per_gene_bin_counts, n_bins);
//   float ratio  = qcc.five_prime_three_prime_ratio(profile, /*head_bins=*/10);

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace singlet {

// ── QC record ────────────────────────────────────────────────────────────────

struct AlignmentQC {
    uint64_t total_reads = 0;
    uint64_t mapped_reads = 0;
    uint64_t unique_reads = 0;
    uint64_t multi_reads = 0;
    uint64_t duplicate_reads = 0;
    float mapping_rate = 0.f;                  ///< mapped_reads / total_reads
    float duplication_rate = 0.f;              ///< duplicate_reads / mapped_reads
    float mito_fraction = 0.f;                 ///< reads on MT chrom / mapped
    float five_prime_three_prime_ratio = 0.f;  ///< mean(head bins) / mean(tail bins)
    uint32_t genes_detected = 0;               ///< genes with ≥ min_reads
    float median_gene_coverage = 0.f;          ///< median reads across detected genes
    float library_complexity = 0.f;            ///< Lander-Waterman estimate of library size
};

// ── Core computation class ────────────────────────────────────────────────────

class AlignmentQCComputer {
   public:
    // ── Gene body coverage ──────────────────────────────────────────────────

    /// Aggregate per-gene coverage bin arrays into a single normalised profile.
    ///
    /// @param per_gene_bins  For each gene: n_bins coverage counts (5'→3' order).
    ///                       Genes with zero total coverage are skipped.
    /// @param n_bins         Number of bins (default 100).
    /// @return               Normalised mean coverage per bin (0–1 scale).
    std::vector<double> compute_gene_body_coverage(
        const std::vector<std::vector<double>>& per_gene_bins,
        int n_bins = 100) const {
        assert(n_bins > 0);
        std::vector<double> profile(static_cast<size_t>(n_bins), 0.0);
        int n_genes = 0;

        for (const auto& bins : per_gene_bins) {
            if (static_cast<int>(bins.size()) != n_bins) continue;
            double total = std::accumulate(bins.begin(), bins.end(), 0.0);
            if (total == 0.0) continue;
            for (int b = 0; b < n_bins; ++b)
                profile[static_cast<size_t>(b)] += bins[static_cast<size_t>(b)] / total;
            ++n_genes;
        }

        if (n_genes > 1) {
            for (auto& v : profile) v /= static_cast<double>(n_genes);
        }
        return profile;
    }

    /// Compute 5'/3' bias ratio from a coverage profile.
    ///
    /// @param profile    Gene body coverage profile (n_bins values, 5'→3').
    /// @param head_bins  Number of bins to average at each end (default 10).
    /// @return           mean(profile[0..head_bins-1]) / mean(profile[n-head_bins..n-1]).
    ///                   Returns 1.0 if denominator is 0.
    float five_prime_three_prime_ratio(const std::vector<double>& profile,
                                       int head_bins = 10) const {
        if (profile.empty() || head_bins <= 0) return 1.f;
        const int n = static_cast<int>(profile.size());
        head_bins = std::min(head_bins, n / 2);

        double five_sum = 0.0, three_sum = 0.0;
        for (int i = 0; i < head_bins; ++i) {
            five_sum += profile[static_cast<size_t>(i)];
            three_sum += profile[static_cast<size_t>(n - 1 - i)];
        }
        if (three_sum == 0.0) return 1.f;
        return static_cast<float>(five_sum / three_sum);
    }

    // ── Library complexity (Lander-Waterman) ────────────────────────────────

    /// Estimate library complexity from observed read counts.
    ///
    /// Lander-Waterman model: C/T = 1 - exp(-N/T)
    ///   C = distinct reads (unique_reads = mapped - duplicate)
    ///   N = total mapped reads
    ///   T = library size (solved numerically)
    ///
    /// @param total_mapped   Total mapped reads (N)
    /// @param distinct_reads Non-duplicate mapped reads (C)
    /// @return               Estimated library complexity T (number of distinct molecules).
    ///                       Returns distinct_reads if N == C (no duplicates seen).
    static float lander_waterman_complexity(uint64_t total_mapped,
                                            uint64_t distinct_reads) {
        if (total_mapped == 0 || distinct_reads == 0) return 0.f;
        if (distinct_reads >= total_mapped) return static_cast<float>(distinct_reads);

        // f(T) = T*(1 - exp(-N/T)) - C = 0
        // Solve with Newton-Raphson: T_{n+1} = T_n - f(T_n)/f'(T_n)
        // f'(T) = 1 - exp(-N/T) - (N/T)*exp(-N/T)
        const double N = static_cast<double>(total_mapped);
        const double C = static_cast<double>(distinct_reads);

        // Initial estimate: T ~ C / (1 - C/N) at low coverage regime
        double T = C / (1.0 - C / N);
        if (!std::isfinite(T) || T <= 0.0) T = C * 2.0;

        for (int iter = 0; iter < 50; ++iter) {
            double eNT = std::exp(-N / T);
            double f = T * (1.0 - eNT) - C;
            double fp = (1.0 - eNT) - (N / T) * eNT;
            if (std::fabs(fp) < 1e-12) break;
            double dT = f / fp;
            T -= dT;
            if (T < C) T = C;  // T must be ≥ C
            if (std::fabs(dT) < 0.5) break;
        }
        return static_cast<float>(T);
    }

    // ── Aggregate compute ────────────────────────────────────────────────────

    /// Build AlignmentQC from pre-collected statistics.
    ///
    /// @param total            Total reads (all)
    /// @param mapped           Mapped reads
    /// @param unique           Uniquely mapped reads (NH==1)
    /// @param multi            Multi-mapped reads (NH>1)
    /// @param duplicate        Duplicate reads (position-based)
    /// @param mito_frac        Mitochondrial read fraction (0–1)
    /// @param gene_counts      Per-gene read counts (for detected / median coverage)
    /// @param profile          Gene body coverage profile (n_bins) — may be empty
    /// @param min_reads        Minimum reads to call a gene detected (default 5)
    /// @param profile_head     Bins used at each end for 5'/3' ratio (default 10)
    AlignmentQC compute_from_stats(
        uint64_t total,
        uint64_t mapped,
        uint64_t unique,
        uint64_t multi,
        uint64_t duplicate,
        float mito_frac,
        const std::vector<uint32_t>& gene_counts,
        const std::vector<double>& profile,
        uint32_t min_reads = 5,
        int profile_head = 10) const {
        AlignmentQC qc;
        qc.total_reads = total;
        qc.mapped_reads = mapped;
        qc.unique_reads = unique;
        qc.multi_reads = multi;
        qc.duplicate_reads = duplicate;
        qc.mito_fraction = mito_frac;

        qc.mapping_rate = (total > 0) ? static_cast<float>(mapped) / total : 0.f;
        qc.duplication_rate = (mapped > 0) ? static_cast<float>(duplicate) / mapped : 0.f;

        // Gene detection + median coverage
        std::vector<uint32_t> detected;
        for (uint32_t c : gene_counts)
            if (c >= min_reads) detected.push_back(c);
        qc.genes_detected = static_cast<uint32_t>(detected.size());

        if (!detected.empty()) {
            std::sort(detected.begin(), detected.end());
            size_t mid = detected.size() / 2;
            qc.median_gene_coverage = (detected.size() % 2 == 0)
                                          ? static_cast<float>(detected[mid - 1] + detected[mid]) / 2.f
                                          : static_cast<float>(detected[mid]);
        }

        // Gene body coverage 5'/3' ratio
        if (!profile.empty())
            qc.five_prime_three_prime_ratio = five_prime_three_prime_ratio(profile, profile_head);
        else
            qc.five_prime_three_prime_ratio = 1.f;

        // Library complexity
        const uint64_t distinct = (mapped > duplicate) ? (mapped - duplicate) : mapped;
        qc.library_complexity = lander_waterman_complexity(mapped, distinct);

        return qc;
    }

    // ── Report writing ───────────────────────────────────────────────────────

    /// Write human-readable TSV report (key\tvalue lines).
    void write_report(const std::string& path, const AlignmentQC& qc) const {
        std::ofstream f(path);
        if (!f.is_open()) {
            std::cerr << "[alignment_qc] Cannot open: " << path << "\n";
            return;
        }
        f << std::fixed << std::setprecision(4);
        f << "total_reads\t" << qc.total_reads << "\n";
        f << "mapped_reads\t" << qc.mapped_reads << "\n";
        f << "unique_reads\t" << qc.unique_reads << "\n";
        f << "multi_reads\t" << qc.multi_reads << "\n";
        f << "duplicate_reads\t" << qc.duplicate_reads << "\n";
        f << "mapping_rate\t" << qc.mapping_rate << "\n";
        f << "duplication_rate\t" << qc.duplication_rate << "\n";
        f << "mito_fraction\t" << qc.mito_fraction << "\n";
        f << "five_prime_three_prime_ratio\t" << qc.five_prime_three_prime_ratio << "\n";
        f << "genes_detected\t" << qc.genes_detected << "\n";
        f << "median_gene_coverage\t" << qc.median_gene_coverage << "\n";
        f << "library_complexity\t" << qc.library_complexity << "\n";
    }

    /// Write gene body coverage profile as a single-line TSV (bin0\tbin1\t...).
    void write_coverage_profile(const std::string& path,
                                const std::vector<double>& profile) const {
        std::ofstream f(path);
        if (!f.is_open()) {
            std::cerr << "[alignment_qc] Cannot open: " << path << "\n";
            return;
        }
        f << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < profile.size(); ++i) {
            if (i) f << "\t";
            f << profile[i];
        }
        f << "\n";
    }
};

}  // namespace singlet
