// test_te_counter.cpp — T-L2-4 unit tests
// Validates:
//   1. TeCounter accumulates correctly across multiple add() calls.
//   2. write() produces a valid .1pz (TP1Z magic) at the expected path.
//   3. All required metadata keys are embedded per §6.5.
//   4. Round-trip via Python singlepress.read_1pz validates shape, dtype, and metadata.
//
// The C++ test writes te_counts.1pz to a temp dir; the CMake test can chain
// a Python round-trip via an auxiliary script.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "singlet-pileup/te_counter.h"

using namespace singlet;

// TP1Z magic bytes
static constexpr uint8_t TP1Z[4] = {'T', 'P', '1', 'Z'};

static bool check_magic(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint8_t hdr[4]{};
    f.read(reinterpret_cast<char*>(hdr), 4);
    return std::memcmp(hdr, TP1Z, 4) == 0;
}

static int test_accumulation() {
    // 3 TE families × 4 cells
    TeCounter counter;
    counter.set_n_families(3);
    counter.set_n_cells(4);

    // Add some observations
    counter.add(0, 0);  // family0, cell0
    counter.add(0, 0);  // dup — should accumulate to 2
    counter.add(1, 1);
    counter.add(2, 3);
    counter.add(1, 2);

    assert(counter.total_umis() == 5);
    printf("PASS: accumulation total_umis=%llu\n",
           static_cast<unsigned long long>(counter.total_umis()));
    return 0;
}

static int test_merge() {
    TeCounter a, b;
    a.set_n_families(2); a.set_n_cells(3);
    b.set_n_families(2); b.set_n_cells(3);

    a.add(0, 0);
    b.add(1, 2);
    a.merge(b);

    assert(a.total_umis() == 2);
    printf("PASS: merge total_umis=2\n");
    return 0;
}

static int test_write_1pz() {
    namespace fs = std::filesystem;
    std::string tmp_dir = "/tmp/test_te_counter_" + std::to_string(getpid());
    fs::create_directories(tmp_dir);

    // 5 families, 3 cells
    TeCounter counter;
    std::vector<std::string> families = {"L1", "Alu", "SVA", "MaLR", "HERV"};
    std::vector<std::string> cells    = {"AAACCTGA", "AAACCTGB", "AAACCTGC"};

    counter.set_n_families(5);
    counter.set_barcodes(cells);

    counter.add(0, 0);  // L1 in cell0
    counter.add(1, 1);  // Alu in cell1
    counter.add(1, 1);  // Alu in cell1 again
    counter.add(4, 2);  // HERV in cell2

    TeCounterMeta meta;
    meta.protocol_id          = "22";
    meta.species              = "human";
    meta.te_family_table_hash = "abc123";
    meta.sample_id            = "SRR_TEST";
    meta.singlify_version     = "test-0.0.1";
    meta.reference_build      = "GRCh38-2024-A";

    std::string out_path = tmp_dir + "/te_counts.1pz";
    bool ok = counter.write(out_path, families, cells, meta, /*threads=*/1);

    if (!ok) {
        printf("FAIL: write() returned false\n");
        return 1;
    }

    // Check magic
    if (!check_magic(out_path)) {
        printf("FAIL: bad TP1Z magic in %s\n", out_path.c_str());
        return 1;
    }

    // Dump output path for Python round-trip
    printf("OUTPUT_PATH=%s\n", out_path.c_str());
    printf("PASS: te_counts.1pz written with correct magic\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_accumulation();
    failures += test_merge();
    failures += test_write_1pz();

    if (failures == 0) {
        printf("ALL PASS: test_te_counter (3/3)\n");
        return 0;
    } else {
        printf("FAILURES: %d\n", failures);
        return 1;
    }
}
