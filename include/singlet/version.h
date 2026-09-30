// SPDX-License-Identifier: MIT
#pragma once
// singlet: version.h
// The one place the C++ pipeline's version lives. It is what the binary
// prints in its banner and usage text, and what it writes into
// provenance.json, summary.json, pileup_stats.json and the metadata JSON
// that is embedded in every .1pz.
//
// Keep in step with project(singlet VERSION ...) in CMakeLists.txt and
// `version` in pyproject.toml. A build may override it by defining
// SINGLET_VERSION_STRING as a string literal
// (shell: -DSINGLET_VERSION_STRING='"2.0.1"').

#ifndef SINGLET_VERSION_STRING
#define SINGLET_VERSION_STRING "2.0.0"
#endif

namespace singlet {

/// Project version, without a leading "v".
inline constexpr const char* kVersion = SINGLET_VERSION_STRING;

}  // namespace singlet
