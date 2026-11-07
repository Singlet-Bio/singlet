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
#include <cstring>
#include <string>
#include <vector>

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

    // the exit line names the phase and the timeout (written from a signal handler: no allocation)
    {
        char buf[320];
        const size_t n = singlet::format_stall_message(buf, sizeof buf, "atac: cells and matrices", 10800);
        assert(n == std::strlen(buf));
        assert(std::string(buf) == "[singlet] STALL WATCHDOG: no progress for 10800s in phase 'atac: cells and matrices' "
                                   "(SINGLET_STALL_TIMEOUT=10800); exiting with code 5\n");
        char tiny[16];
        assert(singlet::format_stall_message(tiny, sizeof tiny, "p", 5) == 15);   // truncated, still terminated
        assert(tiny[15] == 0);
    }

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
    // a child that is making progress re-arms the countdown; the phase is remembered for the exit line
    {
        signal(SIGALRM, SIG_IGN);
        const pid_t pid = fork();
        if (pid == 0) { sleep(2); _exit(0); }
        alarm(3);
        int st = 0;
        const std::string watched = "/tmp";                     // /tmp changes often enough; the point is the code path
        assert(singlet::wait_child_with_progress(pid, &st, {watched}, "wait test") == pid);
        assert(WIFEXITED(st) && WEXITSTATUS(st) == 0);
        const unsigned left2 = alarm(0);
        assert(left2 > 10000);                                    // re-armed to 10800 while the child ran
        assert(std::string(singlet::stall_phase_ref()) == "wait test");
    }
#endif
    std::printf("test_stall_watchdog: PASS\n");
    return 0;
}
