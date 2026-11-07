// SPDX-License-Identifier: MIT
// include/singlet/pileup/run_restart.h
//
// State of one singlet run that must not reach a restarted run or a public output listing.
//
// The stream path restarts itself on the sorted path (--sorted-bam) when the library is far from the protocol orientation
// (restart_on_sorted_path in src/pipeline/singlet.cpp). The first pass sets variables in its own environment for the stream
// only, such as SINGLET_ENGINE_GPU_HANDOFF. The restarted run must start from the environment the user gave, or STAR treats
// the GPU hand-off reads as covered and leaves them out of the sorted BAM (gate 800361: 143,514 pileup reads instead of
// 39.5 M, exit 0). EnvSnapshot keeps the environment of main()'s entry, and restart_env() builds the restarted run's
// environment from it.
//
// The GPU hand-off writes <out>/.gpu_handoff.bin and side files next to it (.reads, .reads.q). They are engine temporaries:
// removed at the end of every run and never listed as outputs.
//
// pileup_shortfall() is the run's last check of the pileup's input: a pileup that saw far fewer mapped records than STAR
// mapped reads uniquely means reads were lost between STAR and the pileup.

#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

extern char** environ;

namespace singlet {

/// The process environment at one point in time, kept as strings ("NAME=value").
class EnvSnapshot {
public:
    void take() {
        vars_.clear();
        for (char** e = environ; e && *e; ++e) vars_.emplace_back(*e);
        taken_ = true;
    }
    bool taken() const { return taken_; }
    const std::vector<std::string>& vars() const { return vars_; }

private:
    std::vector<std::string> vars_;
    bool taken_ = false;
};

/// The process-wide snapshot that main() takes before it reads any option.
inline EnvSnapshot& entry_env() {
    static EnvSnapshot s;
    return s;
}

/// The environment of a restarted run: the snapshot with each of `set` ("NAME=value") replacing a variable of that name.
inline std::vector<std::string> restart_env(const EnvSnapshot& snap, const std::vector<std::string>& set) {
    std::vector<std::string> out;
    out.reserve(snap.vars().size() + set.size());
    auto name_of = [](const std::string& kv) { return kv.substr(0, kv.find('=')); };
    for (const auto& kv : snap.vars()) {
        bool replaced = false;
        for (const auto& s : set) replaced = replaced || name_of(s) == name_of(kv);
        if (!replaced) out.push_back(kv);
    }
    for (const auto& s : set) out.push_back(s);
    return out;
}

/// True for the GPU hand-off file and its side files: ".gpu_handoff.bin", ".gpu_handoff.bin.reads", ".gpu_handoff.bin.reads.q", ...
inline bool is_handoff_file(const std::string& name) {
    return name.rfind(".gpu_handoff", 0) == 0;
}

/// Removes the hand-off files from an output directory (top level, where the pipeline writes them). Returns how many it removed.
inline size_t remove_handoff_files(const std::string& out_dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<fs::path> victims;
    for (const auto& e : fs::directory_iterator(out_dir, ec))
        if (is_handoff_file(e.path().filename().string())) victims.push_back(e.path());
    size_t n = 0;
    for (const auto& p : victims) n += fs::remove_all(p, ec) > 0 ? 1 : 0;
    return n;
}

/// The pileup must see at least this share of STAR's uniquely mapped reads as mapped BAM records (or hand-off alignments).
/// Every uniquely mapped read gives at least one record, so a complete run is at or above 1.0. The loss of gate 800361 was 0.4%.
constexpr double kPileupShortfallFrac = 0.5;

/// True when the pileup saw fewer than kPileupShortfallFrac of STAR's uniquely mapped reads. Unknown counts (0) never trip it.
inline bool pileup_shortfall(uint64_t star_unique_reads, uint64_t pileup_mapped_records) {
    if (star_unique_reads == 0) return false;
    return static_cast<double>(pileup_mapped_records) < kPileupShortfallFrac * static_cast<double>(star_unique_reads);
}

}  // namespace singlet
