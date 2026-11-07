// SPDX-License-Identifier: MIT
// test_nonhost_calibration.cpp — the frozen infection-probability map of nonhost_calibration.h.
//
// 1. bounds       : p lies in (0, 1) for any input, including 0 reads and a hit rate outside [0, 1]
// 2. monotone     : p grows with the reads at a fixed screen and with the mean hit rate at fixed reads
// 3. development  : the evidence of the development libraries lands on the right side (infected libraries high, a noise species and
//                   the SARS-CoV-2 reads of the mock library low)
// 4. screen size  : the same reads out of a larger screen give a lower probability
#include "singlet/pileup/nonhost/nonhost_calibration.h"

#include <cstdlib>
#include <iostream>

using singlet::nonhost::infection_probability;

static int g_fail = 0;
#define CHECK(c)                                                                                  \
    do {                                                                                          \
        if (!(c)) { std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; ++g_fail; }            \
    } while (0)

int main() {
    // 1. bounds
    for (double reads : {0.0, 1.0, 10.0, 1e3, 1e6, 1e9})
        for (double scr : {1.0, 1e5, 1e7})
            for (double hr : {-1.0, 0.0, 0.3, 1.0, 5.0}) {
                const double p = infection_probability(reads, scr, hr);
                CHECK(p > 0.0 && p < 1.0);
            }
    // 2. monotone
    double prev = 0.0;
    for (double reads : {0.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6}) {
        const double p = infection_probability(reads, 2e6, 0.5);
        CHECK(p >= prev);
        prev = p;
    }
    prev = 0.0;
    for (double hr : {0.0, 0.1, 0.3, 0.6, 0.9, 1.0}) {
        const double p = infection_probability(5e3, 2e6, hr);
        CHECK(p >= prev);
        prev = p;
    }
    // 3. development libraries (reads, screened reads, mean hit rate)
    CHECK(infection_probability(553541, 1780120, 0.943) > 0.95);      // ravindra2021_dpi2, SARS-CoV-2
    CHECK(infection_probability(56286, 732000, 0.944) > 0.7);         // ravindra2021_dpi1, SARS-CoV-2
    CHECK(infection_probability(3912, 1164244, 0.915) < 0.15);        // ravindra2021_mock, SARS-CoV-2 reads below 0.5% of the screen
    CHECK(infection_probability(8294, 3.0e6, 0.097) < 0.01);          // Salmonella-like noise species in nuclei2k_v2
    CHECK(infection_probability(7442, 2.9e6, 0.174) < 0.05);          // pbmc5p_v1
    // 4. screen size
    CHECK(infection_probability(5e3, 1e6, 0.5) > infection_probability(5e3, 1e7, 0.5));
    if (g_fail == 0) std::cout << "all passed\n";
    return g_fail == 0 ? 0 : 1;
}
