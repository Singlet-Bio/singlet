// SPDX-License-Identifier: MIT
// Tiny sampling profiler for the E2 benchmark (Windows): a helper thread suspends the main thread every ~0.5 ms,
// records RIP, resumes it. Output: histogram of instruction addresses (resolve with addr2line on a -g build).
#pragma once
#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <map>
#include <thread>
#include <vector>

namespace e2prof {
inline std::atomic<bool> run{false}, active{false};
inline std::map<unsigned long long, unsigned long> hist;
inline HANDLE target = nullptr;
inline std::thread th;
inline void start() {
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &target, 0, FALSE, DUPLICATE_SAME_ACCESS);
    timeBeginPeriod(1);
    run = true;
    th = std::thread([] {
        while (run) {
            Sleep(1);
            if (!active) continue;
            if (SuspendThread(target) == (DWORD)-1) continue;
            CONTEXT ctx; ctx.ContextFlags = CONTEXT_CONTROL;
            if (GetThreadContext(target, &ctx)) hist[ctx.Rip >> 2]++;
            ResumeThread(target);
        }
    });
}
inline void stop_and_print(const char* path, unsigned long long image_base) {
    run = false;
    if (th.joinable()) th.join();
    FILE* f = std::fopen(path, "w");
    if (!f) return;
    std::vector<std::pair<unsigned long, unsigned long long>> v;
    unsigned long tot = 0;
    for (auto& kv : hist) { v.push_back({kv.second, kv.first << 2}); tot += kv.second; }
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
    image_base = (unsigned long long)GetModuleHandleA(nullptr);
    std::fprintf(f, "# samples %lu runtime_image_base 0x%llx\n", tot, image_base);
    for (auto& e : v) std::fprintf(f, "%lu 0x%llx\n", e.first, e.second - image_base + 0x140000000ull);
    std::fclose(f);
}
}  // namespace e2prof
#endif
