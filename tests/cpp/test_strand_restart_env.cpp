// SPDX-License-Identifier: MIT
// tests/cpp/test_strand_restart_env.cpp
//
// The stream path restarts itself on the sorted path when the library is far from the protocol orientation. Gate 800361 found
// that the restarted run kept SINGLET_ENGINE_GPU_HANDOFF from the first pass: STAR then left the hand-off reads out of the sorted
// BAM, the pileup saw 143,514 records for 39.5 M, and the run exited 0. This checks the pieces of the fix (run_restart.h):
//   1. the restarted run's environment is the environment of main()'s entry, checked through a real execve of this binary;
//   2. the hand-off file and its side files are removed and never listed as outputs;
//   3. the pileup shortfall check trips on gate 800361's counts and stays quiet on a complete run.
// tests/cpp/test_strand_restart_e2e.sh forces the restart through the singlet binary itself.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#include "singlet/pileup/run_provenance.h"
#include "singlet/pileup/run_restart.h"

namespace fs = std::filesystem;

// assert() is compiled out in Release (NDEBUG); this check always runs
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "test_strand_restart_env: FAIL line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)

// The child of check 1: runs with the restarted environment and reports what it sees through its exit status.
static int child_main() {
    if (std::getenv("SINGLET_ENGINE_GPU_HANDOFF")) return 11;        // the first pass's hand-off must not reach the restart
    if (std::getenv("SINGLET_ENGINE_GPU_HANDOFF_MINQ")) return 12;
    if (std::getenv("SINGLET_DECODE_R1")) return 13;                 // nor its temp FASTQ paths
    const char* r = std::getenv("SINGLET_STREAM_STRAND_RESTART");
    if (!r || std::string(r) != "1") return 14;
    const char* u = std::getenv("TEST_RESTART_USER_VAR");            // the user's own variables stay
    if (!u || std::string(u) != "kept") return 15;
    const char* g = std::getenv("SINGLET_GPU_PILEUP");               // including one the first pass unset ("ignored")
    if (!g || std::string(g) != "on") return 16;
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--restart-child") return child_main();

    // ---- 1. environment of the restart ----
    setenv("TEST_RESTART_USER_VAR", "kept", 1);
    setenv("SINGLET_GPU_PILEUP", "on", 1);
    unsetenv("SINGLET_ENGINE_GPU_HANDOFF");
    singlet::entry_env().take();   // main()'s entry
    // what the stream pass does after that
    setenv("SINGLET_ENGINE_GPU_HANDOFF", "/out/.gpu_handoff.bin", 1);
    setenv("SINGLET_ENGINE_GPU_HANDOFF_MINQ", "20", 1);
    setenv("SINGLET_DECODE_R1", "/dev/shm/singlet_1fq_1/R1.fastq", 1);
    unsetenv("SINGLET_GPU_PILEUP");
    auto env = singlet::restart_env(singlet::entry_env(), {"SINGLET_STREAM_STRAND_RESTART=1"});
    int n_restart = 0;
    for (const auto& kv : env) {
        CHECK(kv.rfind("SINGLET_ENGINE_GPU_HANDOFF", 0) != 0);
        n_restart += kv.rfind("SINGLET_STREAM_STRAND_RESTART=", 0) == 0;
    }
    CHECK(n_restart == 1);
    // a variable of the snapshot is replaced, not doubled
    setenv("SINGLET_STREAM_STRAND_RESTART", "0", 1);
    singlet::EnvSnapshot s2;
    s2.take();
    auto env2 = singlet::restart_env(s2, {"SINGLET_STREAM_STRAND_RESTART=1"});
    n_restart = 0;
    for (const auto& kv : env2) n_restart += kv.rfind("SINGLET_STREAM_STRAND_RESTART=", 0) == 0;
    CHECK(n_restart == 1 && env2.back() == "SINGLET_STREAM_STRAND_RESTART=1");
    unsetenv("SINGLET_STREAM_STRAND_RESTART");
    // the real exec, as restart_on_sorted_path does it
    {
        std::vector<char*> nenv;
        for (auto& kv : env) nenv.push_back(kv.data());
        nenv.push_back(nullptr);
        char flag[] = "--restart-child";
        char* nargv[] = {argv[0], flag, nullptr};
        pid_t pid = fork();
        CHECK(pid >= 0);
        if (pid == 0) {
            execve("/proc/self/exe", nargv, nenv.data());
            _exit(99);
        }
        int ws = 0;
        waitpid(pid, &ws, 0);
        if (!WIFEXITED(ws) || WEXITSTATUS(ws) != 0) {
            std::fprintf(stderr, "test_strand_restart_env: restarted child saw the wrong environment (exit %d)\n",
                         WIFEXITED(ws) ? WEXITSTATUS(ws) : -1);
            return 1;
        }
    }

    // ---- 2. hand-off files ----
    CHECK(singlet::is_handoff_file(".gpu_handoff.bin"));
    CHECK(singlet::is_handoff_file(".gpu_handoff.bin.reads"));
    CHECK(singlet::is_handoff_file(".gpu_handoff.bin.reads.q"));
    CHECK(!singlet::is_handoff_file("gpu_handoff.tsv"));
    CHECK(!singlet::is_handoff_file("star_Log.final.out"));
    fs::path dir = fs::temp_directory_path() / ("singlet_restart_env_" + std::to_string(getpid()));
    fs::create_directories(dir);
    for (const char* f : {".gpu_handoff.bin", ".gpu_handoff.bin.reads", ".gpu_handoff.bin.reads.q", "summary.json"})
        std::ofstream(dir / f) << "x";
    const std::string inv_before = singlet::outputs_inventory_json(dir.string());
    CHECK(inv_before.find(".gpu_handoff") == std::string::npos);   // never listed, even while present
    CHECK(inv_before.find("summary.json") != std::string::npos);
    CHECK(singlet::remove_handoff_files(dir.string()) == 3);
    CHECK(!fs::exists(dir / ".gpu_handoff.bin.reads.q"));
    CHECK(fs::exists(dir / "summary.json"));
    CHECK(singlet::remove_handoff_files(dir.string()) == 0);
    fs::remove_all(dir);

    // ---- 3. pileup shortfall ----
    CHECK(singlet::pileup_shortfall(21000000, 143514));        // gate 800361's order of magnitude: tripped
    CHECK(!singlet::pileup_shortfall(21000000, 39540743));     // a complete run (records include multi-mappers): quiet
    CHECK(!singlet::pileup_shortfall(21000000, 10500000));     // exactly half: quiet
    CHECK(singlet::pileup_shortfall(21000000, 10499999));
    CHECK(!singlet::pileup_shortfall(0, 0));                   // STAR's count unknown: never trips

    std::printf("test_strand_restart_env: PASS\n");
    return 0;
}
