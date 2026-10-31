// SPDX-License-Identifier: MIT
// stall_watchdog.h — the SIGALRM stall watchdog's timeout, re-arm, exit message and progress-aware child wait.
//
// singlet arms alarm(SINGLET_STALL_TIMEOUT) at start (default 1800 s) and re-arms it whenever a phase makes progress; if the
// countdown ever reaches zero the handler writes one line (phase, timeout) and _exit()s with code 5. The checkpoints used to
// re-arm with a literal 1800, so the environment variable only covered the first phase: a 10x Multiome 10k ATAC library (30+ min
// between the end of STAR and the end of the ATAC stage) died with exit 5 and no message at SINGLET_STALL_TIMEOUT=10800. Every
// re-arm now goes through rearm_stall_watchdog(phase), the pileup loop re-arms every 5 M reads, the ATAC stage after each phase,
// and the wait for a STAR child re-arms while STAR's own files (Log.progress.out, the BAM sort temp files, the BAM) change.
#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <filesystem>
#include <system_error>
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

/// The phase named at the last re-arm (a string literal; read by the signal handler).
inline const char*& stall_phase_ref() {
    static const char* p = "startup";
    return p;
}

/// "[singlet] STALL WATCHDOG: no progress for Ns in phase 'P' (SINGLET_STALL_TIMEOUT=N); exiting with code 5\n" into buf (no
/// allocation, no stdio: usable from a signal handler). Returns the length written (truncated to cap-1).
inline size_t format_stall_message(char* buf, size_t cap, const char* phase, unsigned secs) {
    size_t n = 0;
    auto put = [&](const char* s) { while (*s && n + 1 < cap) buf[n++] = *s++; };
    auto num = [&](unsigned v) {
        char t[16];
        int k = 0;
        do { t[k++] = static_cast<char>('0' + v % 10); v /= 10; } while (v && k < 15);
        while (k && n + 1 < cap) buf[n++] = t[--k];
    };
    put("[singlet] STALL WATCHDOG: no progress for ");
    num(secs);
    put("s in phase '");
    put(phase ? phase : "?");
    put("' (SINGLET_STALL_TIMEOUT=");
    num(secs);
    put("); exiting with code 5\n");
    buf[n] = 0;
    return n;
}

#if defined(__unix__) || defined(__APPLE__)
/// SIGALRM handler: one line on stderr, then exit code 5 (stalled).
inline void stall_alarm_handler(int) {
    char buf[320];
    const size_t n = format_stall_message(buf, sizeof buf, stall_phase_ref(), stall_timeout_secs());
    ssize_t w = ::write(2, buf, n);
    (void)w;
    _exit(5);
}
#endif

/// Restart the watchdog's countdown after a phase that made progress; `phase` (a string literal) names the phase that runs next.
inline void rearm_stall_watchdog(const char* phase = nullptr) {
    if (phase) stall_phase_ref() = phase;
#if defined(__unix__) || defined(__APPLE__)
    ::alarm(stall_timeout_secs());
#endif
}

#if defined(__unix__) || defined(__APPLE__)
/// Newest modification time (seconds) under `p` (a file, or a directory searched recursively); 0 if absent.
inline long newest_mtime(const std::string& p) {
    namespace fs = std::filesystem;
    std::error_code ec;
    long best = 0;
    auto one = [&](const fs::path& f) {
        struct stat st;
        if (::stat(f.c_str(), &st) == 0 && st.st_mtime > best) best = static_cast<long>(st.st_mtime);
    };
    if (!fs::exists(p, ec)) return 0;
    one(p);
    if (fs::is_directory(p, ec))
        for (fs::recursive_directory_iterator it(p, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end;
             it.increment(ec))
            one(it->path());
    return best;
}

/// waitpid(pid) that re-arms the watchdog while any watched path shows activity (STAR's progress log, BAM sort temp files, the
/// BAM). A child that makes no visible progress for the timeout still trips the watchdog. Returns waitpid's result.
inline pid_t wait_child_with_progress(pid_t pid, int* status, const std::vector<std::string>& watch, const char* phase) {
    long seen = 0;
    unsigned tick = 0;
    for (;;) {
        const pid_t r = ::waitpid(pid, status, WNOHANG);
        if (r != 0) return r;  // exited (r == pid) or error (-1)
        if (tick++ % 10 == 0) {  // every ~10 s
            long newest = 0;
            for (const auto& w : watch) {
                const long t = newest_mtime(w);
                if (t > newest) newest = t;
            }
            if (newest > seen) {
                seen = newest;
                rearm_stall_watchdog(phase);
            }
        }
        ::usleep(1000000);
    }
}
#endif

}  // namespace singlet
