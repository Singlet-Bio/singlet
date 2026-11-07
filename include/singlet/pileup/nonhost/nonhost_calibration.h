// SPDX-License-Identifier: MIT
// nonhost_calibration.h — calibrated infection probability of a non-host candidate species.
//
// p = sigmoid(b0 + b1 * log10((reads + 0.5) / screened_reads) + b2 * mean_hit_rate)
//
// reads          EM-assigned reads of the candidate species (n_reads_assigned of nonhost_em.h)
// screened_reads host-unmapped reads that went through the screen
// mean_hit_rate  EM-weighted mean hit rate of the species (SpeciesAbundance::mean_hit_rate)
//
// The map is monotone in both features and FROZEN: the coefficients were fitted once on the development libraries listed in
// kInfectionCalibrationFitSet (logistic regression, ridge 1.0 on the slopes; bench/nonhost_calib/fit_score.py) and are not refitted
// at run time. Positives are library-taxon pairs of known infection (SARS-CoV-2, EBV), negatives are pairs of a taxon known to be absent
// from the library. No library of the held-out pool entered the fit. The probability is per candidate species; for a taxon with several
// candidate species (segments, strains) the taxon's probability is the maximum over them.
#pragma once

#include <algorithm>
#include <cmath>

namespace singlet {
namespace nonhost {

struct InfectionCalibration {
    double b0, b1, b2;
};

// FITTED COEFFICIENTS: filled in by bench/nonhost_calib/fit_score.py (see the fit set below)
inline constexpr InfectionCalibration kInfectionCalibration{0.0, 0.0, 0.0};
inline constexpr const char* kInfectionCalibrationFitSet = "unfitted";

inline double infection_probability(double reads, double screened_reads, double mean_hit_rate,
                                    const InfectionCalibration& c = kInfectionCalibration) {
    const double x1 = std::log10((std::max(reads, 0.0) + 0.5) / std::max(screened_reads, 1.0));
    const double z = c.b0 + c.b1 * x1 + c.b2 * std::max(0.0, std::min(1.0, mean_hit_rate));
    return 1.0 / (1.0 + std::exp(-z));
}

}  // namespace nonhost
}  // namespace singlet
