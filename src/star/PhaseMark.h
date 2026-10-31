#pragma once
// SINGLET_STAR_PHASE_LOG=1: one line per phase mark in Log.out with the wall seconds since the first mark and the process's CPU seconds
// (all threads), so the host core-s of STAR's load, the GPU prefill (read, lock wait, device, inserts, hand-off) and the mapping pass can be read off.
#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <iomanip>
#include "Parameters.h"
#include "InOutStreams.h"

inline void starPhaseMark(Parameters &P, const char *name) {
    static const bool on = getenv("SINGLET_STAR_PHASE_LOG") != nullptr;
    if (!on) return;
    static const auto t0 = std::chrono::steady_clock::now();
    timespec ts; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    P.inOut->logMain << "PHASE " << name << " wall=" << std::fixed << std::setprecision(1)
                     << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                     << " cpu=" << (ts.tv_sec + ts.tv_nsec * 1e-9) << "\n" << std::flush;
}
