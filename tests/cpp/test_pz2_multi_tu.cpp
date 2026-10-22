// SPDX-License-Identifier: MIT
// pz2 is header-only: two translation units including pz2.h must link into one program (no ODR / duplicate-symbol problems).
// This file holds main(); test_pz2_multi_tu_b.cpp holds an encode/decode round trip.
#include <singlet/pz2/pz2.h>
#include <cstdio>
int helper_b();
int main() {
    using namespace singlet::pz2;
    std::printf("a: %s\n", isa_report().c_str());
    return helper_b();
}
