// SPDX-License-Identifier: MIT
#pragma once
// End-of-run self-check of a finished output directory (singlet --self-check; the campaign runner passes it).
// Report only: it never changes an output. It decodes the editing matrices and recomputes the editing tables from them
// (rna_editing.h check_outputs), and writes self_check.json next to the other run outputs. The run exits with kExitSelfCheck
// when a check fails, so a bad output cannot be mistaken for a good one.

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include "rna_editing.h"

namespace singlet {

inline constexpr int kExitSelfCheck = 12;   // named exit code: self_check_failed (0 success, 4 zero_cells, 10 pileup_shortfall)

inline std::string self_check_json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
    return o;
}

/// Runs the checks on `out_prefix`, writes `out_prefix`/self_check.json and returns the number of problems found (0 = clean).
inline int run_self_check(const std::string& out_prefix) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool has_editing = std::filesystem::exists(out_prefix + "/private/editing/editing_alt.1pz") ||
                             std::filesystem::exists(out_prefix + "/private/editing/editing_dp.1pz");
    const auto problems = editing::check_outputs(out_prefix);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::ofstream f(out_prefix + "/self_check.json");
    f << "{\n  \"schema\": \"singlet-self-check/1\",\n  \"ok\": " << (problems.empty() ? "true" : "false")
      << ",\n  \"checks\": {\n    \"editing\": {\"present\": " << (has_editing ? "true" : "false") << ", \"ok\": "
      << (problems.empty() ? "true" : "false") << ", \"seconds\": " << secs << ", \"problems\": [";
    for (size_t i = 0; i < problems.size(); ++i) f << (i ? ", " : "") << "\"" << self_check_json_escape(problems[i]) << "\"";
    f << "]}\n  }\n}\n";
    return static_cast<int>(problems.size());
}

}  // namespace singlet
