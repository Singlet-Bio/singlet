// SPDX-License-Identifier: MIT
// stall_watchdog.h — the SIGALRM stall watchdog's timeout and re-arm, shared by singlet.cpp and the ATAC stage.
//
// singlet arms alarm(SINGLET_STALL_TIMEOUT) at start (default 1800 s) and re-arms it at each checkpoint; the handler _exit()s with
// code 5. The checkpoints used to re-arm with a literal 1800, so the environment variable only covered the first phase: a 10x
// Multiome 10k ATAC library (30+ minutes between the end of STAR and the end of the ATAC stage) died with exit 5 at
// SINGLET_STALL_TIMEOUT=10800 and no message. Every re-arm now goes through rearm_stall_watchdog().
#pragma once

#include <cstdlib>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace singlet {

/// The timeout in seconds from the text of SINGLET_STALL_TIMEOUT (null, empty, non-numeric or <= 0 -> `dflt`).
inline unsigned parse_stall_timeout(const char* s, unsigned dflt = 1800) {
    if (!s || !*s) return dflt;
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || v <= 0) return dflt;
    return static_cast<unsigned>(v);
}

/// SINGLET_STALL_TIMEOUT, read once.
inline unsigned stall_timeout_secs() {
    static const unsigned v = parse_stall_timeout(std::getenv("SINGLET_STALL_TIMEOUT"));
    return v;
}

/// Restart the watchdog's countdown: call after every phase that made progress.
inline void rearm_stall_watchdog() {
#if defined(__unix__) || defined(__APPLE__)
    ::alarm(stall_timeout_secs());
#endif
}

}  // namespace singlet
