// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet {

struct ATACPeak {
    std::string chrom;
    uint32_t start;
    uint32_t end;
    float score;      // -log10(p-value) of most significant bin
    uint32_t signal;  // total counts in peak
    float fold_enrichment;
};

struct PeakCallParams {
    uint32_t bin_size = 500;
    uint32_t background_window = 10000;  // in bp
    float pvalue_threshold = 1e-5f;
    uint32_t min_peak_width = 200;  // in bp
    uint32_t merge_distance = 500;  // in bp
};

// Poisson upper-tail p-value: P(X >= observed | lambda)
// Uses log-gamma for numerical stability
inline float poisson_pvalue(uint32_t observed, float lambda) {
    if (lambda <= 0.0f) {
        return (observed == 0) ? 1.0f : 0.0f;
    }
    if (observed == 0) return 1.0f;

    // Sum log P(X = i) for i = 0 .. observed-1, then complement
    // log P(X = k) = k*log(lambda) - lambda - lgamma(k+1)
    // Use Kahan summation in log-prob space for stability
    double log_lambda = std::log((double)lambda);
    double neg_lambda = -(double)lambda;

    // For large lambda (>1000) use normal approximation
    if (lambda > 1000.0f) {
        double z = ((double)observed - lambda) / std::sqrt(lambda);
        if (z <= 0.0) return 1.0f;
        // P(Z >= z) approximation via erfc
        double p = 0.5 * std::erfc(z / std::sqrt(2.0));
        if (p < 1e-300) p = 1e-300;
        return (float)p;
    }

    // Compute P(X < observed) = sum_{k=0}^{observed-1} P(X=k)
    // Use log-sum-exp trick: accumulate probabilities directly (lambda <= 1000)
    double cum = 0.0;
    double log_pk = neg_lambda;  // log P(X=0) = -lambda
    for (uint32_t k = 0; k < observed; ++k) {
        if (k > 0) log_pk += log_lambda - std::log((double)k);
        double pk = std::exp(log_pk);
        cum += pk;
        if (cum >= 1.0) return 0.0f;  // p-value essentially 0
    }
    double pval = 1.0 - cum;
    if (pval < 0.0) pval = 0.0;
    if (pval > 1.0) pval = 1.0;
    return (float)pval;
}

// Call peaks from per-bin aggregate counts
std::vector<ATACPeak> call_peaks(
    const std::vector<uint32_t>& bin_counts,
    const std::vector<std::string>& bin_chroms,
    const std::vector<uint32_t>& bin_starts,
    const PeakCallParams& params = PeakCallParams()) {
    const size_t N = bin_counts.size();
    if (N == 0) return {};

    // Half-window in bins
    const uint32_t hw = (params.background_window / 2) / params.bin_size;
    const float pval_thresh = params.pvalue_threshold;

    // Process chromosome by chromosome to avoid mixing backgrounds
    // Build chrom → list of bin indices
    std::unordered_map<std::string, std::vector<size_t>> chrom_bins;
    for (size_t i = 0; i < N; ++i) chrom_bins[bin_chroms[i]].push_back(i);

    std::vector<ATACPeak> peaks;

    for (auto& [chrom, indices] : chrom_bins) {
        // Sort indices by start position
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return bin_starts[a] < bin_starts[b];
        });

        const size_t M = indices.size();
        std::vector<float> pvals(M);
        std::vector<float> scores(M);
        std::vector<float> lambdas(M);

        // Precompute prefix sums for window background
        std::vector<double> prefix(M + 1, 0.0);
        for (size_t j = 0; j < M; ++j) prefix[j + 1] = prefix[j] + bin_counts[indices[j]];

        for (size_t j = 0; j < M; ++j) {
            size_t lo = (j >= hw) ? j - hw : 0;
            size_t hi = std::min(M, j + hw + 1);

            // Sum in window excluding center bin
            double win_sum = (prefix[hi] - prefix[lo]) - bin_counts[indices[j]];
            size_t win_bins = (hi - lo - 1);
            double lambda = (win_bins > 0) ? win_sum / win_bins : 1.0;
            if (lambda < 1.0) lambda = 1.0;  // floor to avoid significance from noise

            lambdas[j] = (float)lambda;
            float pv = poisson_pvalue(bin_counts[indices[j]], (float)lambda);
            pvals[j] = pv;
            scores[j] = (pv > 0.0f) ? -std::log10(pv) : 300.0f;
        }

        // Merge significant bins into peaks
        const uint32_t merge_bins = (params.merge_distance + params.bin_size - 1) / params.bin_size;
        (void)merge_bins;

        // Identify significant bins
        std::vector<bool> sig(M, false);
        for (size_t j = 0; j < M; ++j) sig[j] = (pvals[j] < pval_thresh);

        // Extend significant regions
        size_t j = 0;
        while (j < M) {
            if (!sig[j]) {
                ++j;
                continue;
            }

            // Start of a significant run
            size_t start_j = j;
            while (j < M && sig[j]) ++j;
            size_t end_j = j;  // exclusive

            // Build ATACPeak
            ATACPeak pk;
            pk.chrom = chrom;
            pk.start = bin_starts[indices[start_j]];
            pk.end = bin_starts[indices[end_j - 1]] + params.bin_size;

            uint32_t total_signal = 0;
            float best_score = 0.0f;
            float best_lambda = 1.0f;
            uint32_t best_count = 0;
            for (size_t k = start_j; k < end_j; ++k) {
                total_signal += bin_counts[indices[k]];
                if (scores[k] > best_score) {
                    best_score = scores[k];
                    best_lambda = lambdas[k];
                    best_count = bin_counts[indices[k]];
                }
            }
            pk.score = best_score;
            pk.signal = total_signal;
            pk.fold_enrichment = (best_lambda > 0.0f) ? (float)best_count / best_lambda : 0.0f;

            if ((pk.end - pk.start) >= params.min_peak_width) {
                peaks.push_back(pk);
            }
        }
    }

    // Sort peaks by chrom then start
    std::sort(peaks.begin(), peaks.end(), [](const ATACPeak& a, const ATACPeak& b) {
        if (a.chrom != b.chrom) return a.chrom < b.chrom;
        return a.start < b.start;
    });

    // Merge peaks within merge_distance on same chrom
    std::vector<ATACPeak> merged;
    for (auto& pk : peaks) {
        if (!merged.empty() && merged.back().chrom == pk.chrom &&
            pk.start <= merged.back().end + params.merge_distance) {
            auto& prev = merged.back();
            prev.end = std::max(prev.end, pk.end);
            prev.signal += pk.signal;
            if (pk.score > prev.score) {
                prev.score = pk.score;
                prev.fold_enrichment = pk.fold_enrichment;
            }
        } else {
            merged.push_back(pk);
        }
    }

    return merged;
}

// Write peaks in BED6 format
inline bool write_peaks_bed(const std::string& path, const std::vector<ATACPeak>& peaks) {
    std::ofstream out(path);
    if (!out) return false;
    for (size_t i = 0; i < peaks.size(); ++i) {
        const auto& pk = peaks[i];
        out << pk.chrom << '\t'
            << pk.start << '\t'
            << pk.end << '\t'
            << "peak_" << (i + 1) << '\t'
            << pk.score << '\t'
            << ".\n";
    }
    return true;
}

}  // namespace singlet
