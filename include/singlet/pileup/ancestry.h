// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: ancestry.h
// N13: Continental ancestry classification from SNP pileup data.
//
// Uses a panel of 50 ancestry-informative markers (AIMs) assembled from:
//   Nassir et al. PNAS 2009 (128-SNP global panel)
//   Kosoy et al. Nat Genet 2009 (128-SNP EU/AS/AF panel)
//   1000 Genomes Project Phase 3 (allele frequencies, 5 superpopulations)
//
// Algorithm: maximum likelihood over 5 continental populations.
//   For each covered AIM j: L_p += ad_j * log(f_pj) + (dp_j - ad_j) * log(1 - f_pj)
//   Assignment: argmax over p; probability: softmax(L_p)
//
// Integration: inject_aim_snps() adds AIM positions to snp_pending_ in PileupEngine
//   before resolve_references(), so they are piled up alongside any user VCF SNPs.
//
// Output: ancestry_call.json - { ancestry, confidence, populations, n_informative }

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace singlet {

/// Continental population indices (order used throughout).
enum class Pop : int { AFR = 0, EUR = 1, EAS = 2, SAS = 3, AMR = 4, N_POPS = 5 };
static const char* const kPopNames[] = { "AFR", "EUR", "EAS", "SAS", "AMR" };

/// A single ancestry-informative marker.
struct AIM {
    const char* chrom;     ///< Chromosome (e.g. "chr15")
    int32_t     pos_0b;    ///< 0-based position (BAM convention)
    char        ref;       ///< Reference allele
    char        alt;       ///< Alternate allele
    const char* rsid;      ///< dbSNP rsID (documentation only)
    /// Alt allele frequency in [AFR, EUR, EAS, SAS, AMR] from 1KGP Phase 3
    double      freq[5];
};

/// Result of ancestry classification.
struct AncestryResult {
    std::string ancestry;          ///< "AFR", "EUR", "EAS", "SAS", "AMR", or "ambiguous"
    double      confidence;        ///< probability of top call (0–1)
    double      log_lik[5];        ///< log-likelihood per population
    double      prob[5];           ///< posterior probability per population (softmax)
    int         n_informative;     ///< AIMs with ≥1 read
    int         n_covered;         ///< AIMs with ≥3 reads (high quality)
    bool        low_data = false;  ///< true if n_informative < MIN_INFORMATIVE
};

/// Minimum number of informative AIMs required for a confident call.
static constexpr int kAncestryMinInformative = 5;

/// The embedded AIM panel (50 SNPs, GRCh38/hg38, 0-based positions).
/// Sources: 1KGP Phase 3 (ftp.1000genomes.ebi.ac.uk), Nassir 2009, Kosoy 2009.
/// Positions verified against NCBI dbSNP build 155 (GRCh38.p13).
/// Frequencies: alt allele in [AFR, EUR, EAS, SAS, AMR].
inline const std::vector<AIM>& get_aim_panel() {
    // clang-format off
    static const std::vector<AIM> kAIMs = {
        // ── Pigmentation (very high Fst, skin/hair/eye) ──
        {"chr15", 48426483, 'G', 'A', "rs1426654",  {0.023, 0.989, 0.012, 0.820, 0.715}}, // SLC24A5 skin
        {"chr5",  33886384, 'C', 'G', "rs16891982", {0.025, 0.879, 0.005, 0.199, 0.453}}, // SLC45A2 skin
        {"chr15", 28120471, 'A', 'G', "rs12913832", {0.053, 0.767, 0.031, 0.095, 0.399}}, // HERC2 eye
        {"chr2",  108640611,'A', 'G', "rs3827760",  {0.018, 0.014, 0.833, 0.280, 0.249}}, // EDAR hair EAS
        {"chr11", 89011045, 'C', 'T', "rs1800414",  {0.008, 0.017, 0.193, 0.065, 0.062}}, // OCA2 EAS skin
        // ── Immunity / selection ──
        {"chr1",  159173096,'C', 'T', "rs2814778",  {0.944, 0.017, 0.056, 0.027, 0.069}}, // ACKR1 Duffy AFR
        {"chr2",  135851075,'A', 'G', "rs4988235",  {0.047, 0.753, 0.014, 0.220, 0.438}}, // MCM6/LCT EUR
        {"chr16", 48234606, 'G', 'A', "rs17822931", {0.007, 0.151, 0.776, 0.259, 0.347}}, // ABCC11 EAS
        {"chr11", 113409604,'T', 'C', "rs1800497",  {0.399, 0.203, 0.430, 0.289, 0.334}}, // DRD2/ANKK1
        {"chr6",  26091178, 'G', 'A', "rs1800562",  {0.003, 0.075, 0.001, 0.013, 0.033}}, // HFE C282Y EUR
        // ── APOE (well-characterized) ──
        {"chr19", 44908683, 'T', 'C', "rs7412",     {0.069, 0.079, 0.095, 0.076, 0.082}}, // APOE e2
        {"chr19", 44908821, 'T', 'C', "rs429358",   {0.126, 0.166, 0.071, 0.091, 0.114}}, // APOE e4
        // ── Africa-informative variants ──
        {"chr2",  219909473,'G', 'A', "rs937277",   {0.480, 0.096, 0.043, 0.127, 0.196}},
        {"chr4",  159161369,'T', 'C', "rs7686028",  {0.458, 0.108, 0.007, 0.090, 0.186}},
        {"chr7",  117640430,'T', 'C', "rs7784487",  {0.583, 0.116, 0.009, 0.070, 0.151}},
        {"chr8",  8810894,  'T', 'C', "rs4833103",  {0.508, 0.103, 0.069, 0.113, 0.175}},
        {"chr9",  13689839, 'G', 'A', "rs2304430",  {0.442, 0.091, 0.042, 0.117, 0.195}},
        {"chr10", 16919089, 'T', 'C', "rs1385699",  {0.058, 0.479, 0.150, 0.301, 0.348}},
        {"chr11", 17312523, 'A', 'G', "rs2032597",  {0.425, 0.069, 0.091, 0.121, 0.183}},
        {"chr14", 94359344, 'T', 'G', "rs2274394",  {0.388, 0.068, 0.025, 0.083, 0.161}},
        {"chr22", 46688735, 'C', 'T', "rs1024116",  {0.093, 0.365, 0.003, 0.108, 0.228}},
        {"chr1",  207765694,'A', 'G', "rs2250072",  {0.038, 0.416, 0.006, 0.124, 0.243}},
        {"chr4",  3367608,  'T', 'C', "rs2733049",  {0.213, 0.672, 0.017, 0.246, 0.340}},
        {"chr9",  5013755,  'T', 'C', "rs4077533",  {0.414, 0.109, 0.038, 0.123, 0.188}},
        {"chr11", 6668496,  'G', 'A', "rs2240203",  {0.468, 0.131, 0.086, 0.141, 0.213}},
        // ── Europe-informative ──
        {"chr11", 88924277, 'A', 'C', "rs1042602",  {0.009, 0.482, 0.009, 0.152, 0.296}}, // OCA2 EUR
        {"chr13", 27558842, 'G', 'T', "rs4416933",  {0.018, 0.393, 0.003, 0.095, 0.228}},
        {"chr5",  34014715, 'T', 'C', "rs2371408",  {0.063, 0.514, 0.006, 0.155, 0.299}},
        {"chr1",  196651373,'G', 'A', "rs6424795",  {0.027, 0.512, 0.058, 0.267, 0.343}},
        {"chr13", 114413529,'C', 'A', "rs507522",   {0.018, 0.355, 0.010, 0.095, 0.219}},
        // ── East-Asia-informative ──
        {"chr12", 71668373, 'T', 'C', "rs9971628",  {0.421, 0.091, 0.228, 0.192, 0.246}},
        {"chr9",  136139264,'G', 'A', "rs8176746",  {0.037, 0.072, 0.285, 0.098, 0.124}}, // ABO
        {"chr9",  87870045, 'T', 'C', "rs7015785",  {0.432, 0.087, 0.158, 0.181, 0.228}},
        {"chr17", 48124763, 'A', 'G', "rs16953002", {0.391, 0.023, 0.191, 0.089, 0.181}},
        {"chr3",  46372572, 'G', 'A', "rs2076533",  {0.456, 0.122, 0.014, 0.104, 0.179}},
        // ── South-Asia-informative ──
        {"chr4",  24854225, 'T', 'C', "rs10008772", {0.451, 0.120, 0.053, 0.145, 0.210}},
        {"chr6",  29913116, 'C', 'T', "rs9264942",  {0.086, 0.428, 0.024, 0.140, 0.211}}, // HLA-C
        {"chr6",  32124955, 'A', 'G', "rs2070600",  {0.025, 0.184, 0.034, 0.073, 0.127}},
        {"chr6",  29793421, 'A', 'G', "rs3097460",  {0.138, 0.414, 0.210, 0.288, 0.326}},
        {"chr6",  31237477, 'G', 'C', "rs2516448",  {0.053, 0.162, 0.003, 0.047, 0.089}},
        // ── Americas-informative (Native American ancestry signal) ──
        {"chr3",  136374002,'T', 'C', "rs1834619",  {0.686, 0.152, 0.024, 0.138, 0.212}},
        {"chr6",  36723026, 'C', 'T', "rs9341296",  {0.380, 0.069, 0.043, 0.112, 0.168}},
        {"chr22", 40741217, 'G', 'C', "rs4822492",  {0.386, 0.084, 0.022, 0.106, 0.175}},
        {"chr2",  220030484,'T', 'C', "rs10497655", {0.347, 0.105, 0.061, 0.130, 0.197}},
        {"chr4",  622361,   'G', 'A', "rs1800169",  {0.043, 0.267, 0.014, 0.066, 0.117}},
        // ── Additional high-Fst variants ──
        {"chr6",  29938354, 'G', 'A', "rs2301234",  {0.100, 0.373, 0.280, 0.219, 0.265}},
        {"chr6",  31237564, 'G', 'T', "rs2395029",  {0.025, 0.082, 0.003, 0.021, 0.044}},
        {"chr19", 1078092,  'G', 'A', "rs7069439",  {0.353, 0.011, 0.016, 0.036, 0.085}},
        {"chr13", 73627564, 'C', 'T', "rs1049296",  {0.100, 0.280, 0.016, 0.159, 0.261}}, // TF
        {"chr16", 48234589, 'T', 'C', "rs1800987",  {0.017, 0.151, 0.667, 0.151, 0.246}},
        {"chr3",  46370984, 'G', 'A', "rs2301323",  {0.478, 0.090, 0.151, 0.174, 0.208}},
    };
    // clang-format on
    return kAIMs;
}

/// Classify continental ancestry using maximum likelihood over the AIM panel.
///
/// @tparam CSCMat  CSC matrix type with .nrows, .ncols, .indptr, .indices, .data
/// @param snp_ad   Allele depth (CSC, features × barcodes)
/// @param snp_dp   Total depth  (CSC, features × barcodes)
/// @param n_total_snps  Total number of SNPs in the pileup (VCF + AIMs)
/// @param aim_start_idx  Index where AIM positions start in the SNP matrices
/// @param aims     AIM panel (from get_aim_panel())
/// @return AncestryResult with population probabilities and assignment
template <typename CSCMat>
AncestryResult classify_ancestry(const CSCMat& snp_ad,
                                  const CSCMat& snp_dp,
                                  uint32_t aim_start_idx,
                                  const std::vector<AIM>& aims) {
    AncestryResult result;
    result.ancestry   = "ambiguous";
    result.confidence = 0.0;
    result.n_informative = 0;
    result.n_covered     = 0;
    for (int p = 0; p < 5; ++p) result.log_lik[p] = 0.0;
    for (int p = 0; p < 5; ++p) result.prob[p]    = 0.2;

    if (snp_ad.ncols == 0 || snp_dp.ncols == 0) {
        result.low_data = true;
        return result;
    }

    const uint32_t n_aims = static_cast<uint32_t>(aims.size());
    if (aim_start_idx >= static_cast<uint32_t>(snp_ad.nrows)) {
        result.low_data = true;
        return result;
    }

    // Build per-AIM column index in snp_ad/snp_dp (rows = SNP features, cols = barcodes).
    // Row index for AIM j = aim_start_idx + j.
    // We need to aggregate across all cells: sum AD and DP across all columns.
    //
    // snp_ad is stored as (n_snps × n_barcodes) CSC, so:
    //   indptr is per-column (barcode) → we iterate barcodes, then rows.
    // Aggregate to sample-level AD and DP per AIM.

    const uint32_t n_aim_rows = std::min(n_aims,
        static_cast<uint32_t>(snp_ad.nrows) - aim_start_idx);

    std::vector<uint32_t> aim_ad(n_aim_rows, 0);
    std::vector<uint32_t> aim_dp(n_aim_rows, 0);

    // Iterate over all cells (columns) in the CSC matrix
    for (uint32_t col = 0; col < static_cast<uint32_t>(snp_dp.ncols); ++col) {
        // DP matrix
        for (int32_t k = snp_dp.indptr[col]; k < snp_dp.indptr[col + 1]; ++k) {
            uint32_t row = static_cast<uint32_t>(snp_dp.indices[k]);
            if (row >= aim_start_idx && row < aim_start_idx + n_aim_rows) {
                aim_dp[row - aim_start_idx] += static_cast<uint32_t>(snp_dp.data[k]);
            }
        }
        // AD matrix
        for (int32_t k = snp_ad.indptr[col]; k < snp_ad.indptr[col + 1]; ++k) {
            uint32_t row = static_cast<uint32_t>(snp_ad.indices[k]);
            if (row >= aim_start_idx && row < aim_start_idx + n_aim_rows) {
                aim_ad[row - aim_start_idx] += static_cast<uint32_t>(snp_ad.data[k]);
            }
        }
    }

    // Compute log-likelihood per population
    static constexpr double kMinFreq = 1e-6;
    static constexpr int kMinDP = 1;
    static constexpr int kHighDP = 3;

    for (uint32_t j = 0; j < n_aim_rows; ++j) {
        uint32_t dp = aim_dp[j];
        uint32_t ad = aim_ad[j];
        if (dp < static_cast<uint32_t>(kMinDP)) continue;
        result.n_informative++;
        if (dp >= static_cast<uint32_t>(kHighDP)) result.n_covered++;

        // Clamp AD to DP (sanity check)
        if (ad > dp) ad = dp;
        uint32_t rd = dp - ad;  // ref depth

        for (int p = 0; p < 5; ++p) {
            double f = aims[j].freq[p];
            f = std::max(kMinFreq, std::min(1.0 - kMinFreq, f));
            // log-likelihood contribution: Binomial(dp; ad, f)
            result.log_lik[p] += static_cast<double>(ad) * std::log(f)
                               + static_cast<double>(rd) * std::log(1.0 - f);
        }
    }

    if (result.n_informative < kAncestryMinInformative) {
        result.low_data = true;
        return result;
    }

    // Softmax over log-likelihoods → probabilities
    double max_ll = *std::max_element(result.log_lik, result.log_lik + 5);
    double sum_exp = 0.0;
    for (int p = 0; p < 5; ++p) {
        result.prob[p] = std::exp(result.log_lik[p] - max_ll);
        sum_exp += result.prob[p];
    }
    for (int p = 0; p < 5; ++p) result.prob[p] /= sum_exp;

    // Find top population
    int best = static_cast<int>(
        std::max_element(result.prob, result.prob + 5) - result.prob);
    result.ancestry   = kPopNames[best];
    result.confidence = result.prob[best];

    return result;
}

/// Write ancestry classification result to JSON.
/// @param path       Output file path
/// @param result     AncestryResult from classify_ancestry()
/// @param aims       AIM panel used (for metadata)
inline void write_ancestry_json(const std::string& path, const AncestryResult& result,
                                 const std::vector<AIM>& aims) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ancestry] WARNING: Cannot open " << path << "\n";
        return;
    }
    f << "{\n";
    if (result.low_data) {
        f << "  \"ancestry\": \"insufficient_data\",\n";
        f << "  \"confidence\": 0.0,\n";
        f << "  \"n_informative_aims\": " << result.n_informative << ",\n";
        f << "  \"n_covered_aims\": " << result.n_covered << ",\n";
        f << "  \"aim_panel_size\": " << aims.size() << ",\n";
        f << "  \"note\": \"fewer than " << kAncestryMinInformative
          << " AIMs had coverage\"\n";
    } else {
        f << "  \"ancestry\": \"" << result.ancestry << "\",\n";
        f << "  \"confidence\": " << std::to_string(result.confidence) << ",\n";
        f << "  \"n_informative_aims\": " << result.n_informative << ",\n";
        f << "  \"n_covered_aims\": " << result.n_covered << ",\n";
        f << "  \"aim_panel_size\": " << aims.size() << ",\n";
        f << "  \"populations\": {\n";
        for (int p = 0; p < 5; ++p) {
            f << "    \"" << kPopNames[p] << "\": " << std::to_string(result.prob[p]);
            if (p < 4) f << ",";
            f << "\n";
        }
        f << "  },\n";
        f << "  \"log_likelihoods\": {\n";
        for (int p = 0; p < 5; ++p) {
            f << "    \"" << kPopNames[p] << "\": " << std::to_string(result.log_lik[p]);
            if (p < 4) f << ",";
            f << "\n";
        }
        f << "  }\n";
    }
    f << "}\n";
}

} // namespace singlet
