// SPDX-License-Identifier: MIT
// test_stall_watchdog.cpp — the stall watchdog timeout comes from SINGLET_STALL_TIMEOUT for every re-arm.
//
// Regression: the checkpoints re-armed with a literal alarm(1800), so a long ATAC stage (10x Multiome 10k, > 30 min after STAR)
// died with exit 5 at SINGLET_STALL_TIMEOUT=10800. Here: the timeout parser, and (POSIX) that rearm_stall_watchdog() arms the
// alarm with the configured value, not 1800.

#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>

#include "singlet/pileup/stall_watchdog.h"

int main() {
    using singlet::parse_stall_timeout;
    assert(parse_stall_timeout(nullptr) == 1800);
    assert(parse_stall_timeout("") == 1800);
    assert(parse_stall_timeout("abc") == 1800);
    assert(parse_stall_timeout("0") == 1800);
    assert(parse_stall_timeout("-5") == 1800);
    assert(parse_stall_timeout("10800") == 10800);
    assert(parse_stall_timeout("60") == 60);
    assert(parse_stall_timeout(nullptr, 42) == 42);

#if defined(__unix__) || defined(__APPLE__)
    // the helper reads the environment once; set it before the first call
    setenv("SINGLET_STALL_TIMEOUT", "10800", 1);
    assert(singlet::stall_timeout_secs() == 10800);
    signal(SIGALRM, SIG_IGN);
    alarm(7);
    singlet::rearm_stall_watchdog();          // returns the seconds left on the previous alarm
    const unsigned left = alarm(0);           // and the re-armed countdown is the configured one
    assert(left > 10000 && left <= 10800);
    assert(left != 1800);
#endif
    std::printf("test_stall_watchdog: PASS\n");
    return 0;
}
