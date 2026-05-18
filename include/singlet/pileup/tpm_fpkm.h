// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: tpm_fpkm.h
// TPM (Transcripts Per Million) and FPKM (Fragments Per Kilobase Million)
// normalization from raw gene counts + effective gene length from GTF exon union.
//
// TPM_g  = (count_g / length_g) / sum_i(count_i / length_i) × 1e6
// FPKM_g = (count_g × 1e9) / (length_g × total_mapped_fragments)
//
// Effective length = sum of merged (non-overlapping) exon bp per gene.
// Matches RSEM / StringTie / ENCODE conventions.

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "gene_model.h"

namespace singlet {

// Per-gene effective length record
struct GeneLength {
    std::string gene_id;
    std::string gene_name;
    uint32_t effective_length;  // merged exon bp (clamped to ≥1)
};

// Compute effective gene lengths from a GeneModel.
// Uses the flat exon_starts / exon_ends arrays (already merged during GTF build).
// For BED pseudo-gene models every "gene" has exactly one exon, so length is exact.
inline std::vector<GeneLength> compute_gene_lengths(const GeneModel& gm) {
    uint32_t ng = gm.n_genes();
    std::vector<GeneLength> lengths(ng);

    // Accumulate exon bp per gene using the exon_to_gene mapping
    std::vector<uint32_t> bp(ng, 0);
    uint32_t ne = gm.n_exons();
    for (uint32_t e = 0; e < ne; ++e) {
        uint32_t g = gm.exon_to_gene(e);
        if (g < ng) {
            int32_t len = gm.exon_end(e) - gm.exon_start(e);
            if (len > 0) bp[g] += static_cast<uint32_t>(len);
        }
    }

    const auto& gids = gm.gene_ids();
    const auto& gnames = gm.gene_names();
    for (uint32_t g = 0; g < ng; ++g) {
        lengths[g] = {
            gids.empty() ? std::string() : gids[g],
            gnames.empty() ? std::string() : gnames[g],
            bp[g] > 0 ? bp[g] : 1u  // clamp: never divide by zero
        };
    }
    return lengths;
}

// Compute TPM from raw gene counts + effective lengths.
// Returns vector of size counts.size(); all zeros if sum_rpk == 0.
inline std::vector<double> compute_tpm(const std::vector<int32_t>& counts,
                                       const std::vector<GeneLength>& lengths) {
    size_t n = counts.size();
    std::vector<double> rpk(n, 0.0);
    double sum_rpk = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (counts[i] > 0) {
            rpk[i] = static_cast<double>(counts[i]) /
                     static_cast<double>(lengths[i].effective_length);
            sum_rpk += rpk[i];
        }
    }
    std::vector<double> tpm(n, 0.0);
    if (sum_rpk > 0.0) {
        const double scale = 1.0e6 / sum_rpk;
        for (size_t i = 0; i < n; ++i)
            tpm[i] = rpk[i] * scale;
    }
    return tpm;
}

// Compute FPKM from raw gene counts + effective lengths + total mapped fragments.
// Returns vector of size counts.size(); all zeros if total_mapped == 0.
inline std::vector<double> compute_fpkm(const std::vector<int32_t>& counts,
                                        const std::vector<GeneLength>& lengths,
                                        uint64_t total_mapped) {
    size_t n = counts.size();
    std::vector<double> fpkm(n, 0.0);
    if (total_mapped > 0) {
        const double denom_base = static_cast<double>(total_mapped);
        for (size_t i = 0; i < n; ++i) {
            if (counts[i] > 0) {
                fpkm[i] = (static_cast<double>(counts[i]) * 1.0e9) /
                          (static_cast<double>(lengths[i].effective_length) * denom_base);
            }
        }
    }
    return fpkm;
}

// Write gene_expression.tsv with columns:
//   gene_id \t gene_name \t effective_length \t count \t TPM \t FPKM
// Returns false on I/O error.
inline bool write_gene_expression_tsv(const std::string& path,
                                      const std::vector<GeneLength>& lengths,
                                      const std::vector<int32_t>& counts,
                                      const std::vector<double>& tpm,
                                      const std::vector<double>& fpkm) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "[tpm_fpkm] Cannot write: " << path << "\n";
        return false;
    }
    out << "gene_id\tgene_name\teffective_length\tcount\tTPM\tFPKM\n";
    size_t n = lengths.size();
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < n; ++i) {
        out << lengths[i].gene_id << '\t'
            << lengths[i].gene_name << '\t'
            << lengths[i].effective_length << '\t'
            << counts[i] << '\t'
            << tpm[i] << '\t'
            << fpkm[i] << '\n';
    }
    return out.good();
}

}  // namespace singlet
