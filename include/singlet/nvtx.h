// SPDX-License-Identifier: MIT
// nvtx.h — NVTX ranges for the host side of the GPU path (docs/research-beta/gpu-profile-deep.md). Off by default and free when off:
// nothing is compiled without the NVTX headers of the CUDA toolkit (SINGLET_HAVE_NVTX), and with them a range costs one predictable branch
// until SINGLET_NVTX is set in the environment. Under `nsys profile -t nvtx` the ranges name every host phase, thread-pool task, lock wait / hold
// and file pass next to the CUDA API calls, kernels and copies.
//
//   SINGLET_NVTX("name");                 a range for the rest of the scope
//   singlet::NvtxChain ch; ch.next("a");  consecutive phases in one thread: next() closes the previous range and opens the new one
//   singlet::nvtx_mark("name");           an instant mark
#pragma once

#include <cstdlib>

#if defined(SINGLET_HAVE_NVTX) && defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define SINGLET_NVTX_ENABLED 1
#endif
#endif

namespace singlet {

#ifdef SINGLET_NVTX_ENABLED
inline bool nvtx_on() {
    static const bool on = std::getenv("SINGLET_NVTX") != nullptr;
    return on;
}
struct NvtxRange {
    bool active;
    explicit NvtxRange(const char* name) : active(nvtx_on()) { if (active) nvtxRangePushA(name); }
    ~NvtxRange() { if (active) nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};
struct NvtxChain {
    bool open = false;
    void next(const char* name) {
        if (!nvtx_on()) return;
        if (open) nvtxRangePop();
        nvtxRangePushA(name);
        open = true;
    }
    void end() {
        if (open) { nvtxRangePop(); open = false; }
    }
    ~NvtxChain() { end(); }
};
inline void nvtx_mark(const char* name) { if (nvtx_on()) nvtxMarkA(name); }
// ranges that start and end in different scopes or threads (a device lock held across calls): the id comes back from start, 0 when off
inline unsigned long long nvtx_start(const char* name) { return nvtx_on() ? (unsigned long long) nvtxRangeStartA(name) : 0ull; }
inline void nvtx_end(unsigned long long id) { if (id) nvtxRangeEnd((nvtxRangeId_t) id); }
#else
struct NvtxRange { explicit NvtxRange(const char*) {} };
struct NvtxChain { void next(const char*) {} void end() {} };
inline void nvtx_mark(const char*) {}
inline unsigned long long nvtx_start(const char*) { return 0ull; }
inline void nvtx_end(unsigned long long) {}
#endif

}  // namespace singlet

#define SINGLET_NVTX_CAT2(a, b) a##b
#define SINGLET_NVTX_CAT(a, b) SINGLET_NVTX_CAT2(a, b)
#define SINGLET_NVTX(name) ::singlet::NvtxRange SINGLET_NVTX_CAT(singlet_nvtx_range_, __LINE__)(name)
