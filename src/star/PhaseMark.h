#pragma once
// SINGLET_STAR_PHASE_LOG=1: one line per phase mark in Log.out with the wall seconds since the first mark and the process's CPU seconds
// (all threads), so the host core-s of STAR's load, the GPU prefill (read, lock wait, device, inserts, hand-off) and the mapping pass can be read off.
#include <stdio.h>
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
    // resident memory of this process by kind (GB): anonymous, file-backed, shared memory; and its high-water mark
    double rss = 0, anon = 0, file = 0, shm = 0, hwm = 0;
    if (FILE *f = fopen("/proc/self/status", "r")) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            double kb;
            if (sscanf(line, "VmRSS: %lf", &kb) == 1) rss = kb / 1048576.0;
            else if (sscanf(line, "RssAnon: %lf", &kb) == 1) anon = kb / 1048576.0;
            else if (sscanf(line, "RssFile: %lf", &kb) == 1) file = kb / 1048576.0;
            else if (sscanf(line, "RssShmem: %lf", &kb) == 1) shm = kb / 1048576.0;
            else if (sscanf(line, "VmHWM: %lf", &kb) == 1) hwm = kb / 1048576.0;
        }
        fclose(f);
    }
    P.inOut->logMain << "PHASE " << name << " wall=" << std::fixed << std::setprecision(1)
                     << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                     << " cpu=" << (ts.tv_sec + ts.tv_nsec * 1e-9) << " rss=" << rss << " anon=" << anon << " file=" << file << " shmem=" << shm
                     << " hwm=" << hwm << "\n" << std::flush;
}
