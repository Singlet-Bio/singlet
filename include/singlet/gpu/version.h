// SPDX-License-Identifier: MIT
// singlet/gpu/version.h — version constants for the public umbrella.
//
// Pre-1.0: hardcoded in source. After the first MINOR cut, these are written
// by CMake at configure time from the project version + git describe. The
// public umbrella header `singlet::gpu.hpp` only exposes the four functions
// declared here; never mutate them by hand without bumping
// `state/release-policy.md` accordingly.

#pragma once

namespace singlet::gpu {

constexpr int version_major() noexcept { return 0; }
constexpr int version_minor() noexcept { return 1; }
constexpr int version_patch() noexcept { return 0; }

// Set by CMake at configure time once git is initialized; literal "pre-1.0"
// before that point.
constexpr const char* commit_sha() noexcept {
#ifdef SINGLET_GPU_COMMIT_SHA
    return SINGLET_GPU_COMMIT_SHA;
#else
    return "pre-1.0";
#endif
}

}  // namespace singlet::gpu
