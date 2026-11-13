// SPDX-License-Identifier: MIT
// The build stamp: version from the project version, commit compiled in, environment override, and the places that record them.
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>

#include "singlet/version.h"
#include "singlet/pileup/run_provenance.h"
#include "singlet/pileup/summary_json.h"

static void set_env(const char* v) { if (v) setenv("SINGLET_GIT_SHA", v, 1); else unsetenv("SINGLET_GIT_SHA"); }

int main() {
    // version: x.y.z, equal to the CMake project version (passed in by tests/cpp/CMakeLists.txt), never the old hard-coded default
    const std::string v = singlet::kVersion;
    assert(std::regex_match(v, std::regex("[0-9]+\.[0-9]+\.[0-9]+")));
    assert(v == SINGLET_TEST_PROJECT_VERSION && "version.h must follow project(singlet VERSION ...)");
    if (std::string(SINGLET_TEST_PROJECT_VERSION) != "1.0.0") assert(v != "1.0.0");

    // commit: compiled in and non-empty; with the test build's -DSINGLET_GIT_COMMIT (or git) it is a hex id
    const std::string c = singlet::kBuildCommit;
    assert(!c.empty());
    if (c != "unknown") assert(std::regex_match(c, std::regex("[0-9a-f]{7,40}")));

    // environment variable overrides only when non-empty
    set_env(nullptr);           assert(singlet::git_commit() == c);
    set_env("");                assert(singlet::git_commit() == c);
    set_env("deadbeefcafe");    assert(singlet::git_commit() == "deadbeefcafe");
    set_env(nullptr);

    // version line used by `singlet --version` and checked by the campaign runner against BUILD.txt
    assert(singlet::version_line() == "singlet " + v + " commit " + c);

    // recorded in pileup_stats.json's provenance block without the environment variable set
    singlet::RunProvenance rp;
    assert(rp.engine_commit == c);
    set_env("0123456789ab");
    singlet::RunProvenance rp2;
    assert(rp2.engine_commit == "0123456789ab");
    set_env(nullptr);

    std::cout << "test_version_stamp: ok (" << v << " " << c << ")\n";
    return 0;
}
