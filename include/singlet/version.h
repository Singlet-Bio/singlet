// SPDX-License-Identifier: MIT
#pragma once
// singlet: version.h
// The one place the C++ pipeline's version and commit are read. They are what the binary prints for `--version` and in its
// banner, and what it writes into provenance.json, summary.json, pileup_stats.json and the metadata JSON that is embedded in
// every .1pz.
//
// Both come from the build and nowhere else: cmake/BuildInfo.cmake defines SINGLET_VERSION_STRING from
// project(singlet VERSION ...) in the root CMakeLists.txt and SINGLET_GIT_COMMIT from -DSINGLET_GIT_COMMIT or `git rev-parse
// HEAD`, and fails the configure step when either is missing or the version disagrees with the tag being built. There is no
// default here: a hand-written version string is how a 1.3.7 binary once reported 1.0.0.
//
// git_commit() lets the environment variable SINGLET_GIT_SHA override the compiled commit for a run (a non-empty value only).

#include <cstdlib>
#include <string>

#ifndef SINGLET_VERSION_STRING
#error "SINGLET_VERSION_STRING is not defined: build through CMake (cmake/BuildInfo.cmake defines it from the project version)"
#endif
#ifndef SINGLET_GIT_COMMIT
#define SINGLET_GIT_COMMIT "unknown"   // test-only builds without a commit; the pipeline binary cannot be configured this way
#endif

namespace singlet {

namespace version_detail {
constexpr bool nonempty(const char* s) { return s[0] != '\0'; }
}  // namespace version_detail

/// Project version, without a leading "v".
inline constexpr const char* kVersion = SINGLET_VERSION_STRING;
/// Git commit the binary was built from (40 hex digits for a pipeline build).
inline constexpr const char* kBuildCommit = SINGLET_GIT_COMMIT;

static_assert(version_detail::nonempty(kVersion), "the build stamp has an empty version");
static_assert(version_detail::nonempty(kBuildCommit), "the build stamp has an empty commit");

/// The commit to record for this run: SINGLET_GIT_SHA when set and non-empty, else the compiled-in commit.
inline std::string git_commit() {
    if (const char* e = std::getenv("SINGLET_GIT_SHA"))
        if (*e) return e;
    return kBuildCommit;
}

/// The line `singlet --version` prints: "singlet <version> commit <commit>".
inline std::string version_line() {
    return std::string("singlet ") + kVersion + " commit " + git_commit();
}

}  // namespace singlet
