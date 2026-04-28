// test/test_splice_psi.cpp
// Unit tests for G-PSI: per-cell splice junction PSI (splice_psi.h).
// No BAM files required — all tests use synthetic in-memory data.

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "singlet-pileup/splice_psi.h"

using namespace singlet;

static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond, name)                                                        \
    do {                                                                         \
        if (cond) {                                                              \
            std::cout << "  PASS: " << (name) << "\n";                           \
            ++n_pass;                                                            \
        } else {                                                                 \
            std::cout << "  FAIL: " << (name) << " [line " << __LINE__ << "]\n"; \
            ++n_fail;                                                            \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, eps, name) \
    CHECK(std::fabs(static_cast<float>(a) - static_cast<float>(b)) < static_cast<float>(eps), name)

// Minimal CSC mock that satisfies compute_psi's requirements.
struct MockCSC {
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<uint16_t> data;
    uint32_t nrows;
    uint32_t ncols;
};

// ── Test 1: parse_junction_name ──────────────────────────────────────────────
static void test_parse_junction_name() {
    std::cout << "Test 1: parse_junction_name\n";
    std::string chrom;
    uint32_t start = 0, end = 0;

    CHECK(parse_junction_name("chr1:100-200", chrom, start, end), "parse valid");
    CHECK(chrom == "chr1", "chrom == chr1");
    CHECK(start == 100, "start == 100");
    CHECK(end == 200, "end == 200");

    CHECK(parse_junction_name("chrUn_gl000220:5000-6000", chrom, start, end), "parse complex chrom");
    CHECK(chrom == "chrUn_gl000220", "complex chrom name");
    CHECK(start == 5000 && end == 6000, "complex chrom start/end");

    CHECK(!parse_junction_name("no_colon_here", chrom, start, end), "reject missing colon");
    CHECK(!parse_junction_name("chr1:100", chrom, start, end), "reject missing dash");
    CHECK(!parse_junction_name("", chrom, start, end), "reject empty string");
}

// ── Test 2: build_splice_events — shared donor ───────────────────────────────
static void test_build_splice_events_donor() {
    std::cout << "Test 2: build_splice_events — shared donor\n";

    // J0: chr1:100-200, J1: chr1:100-300 share donor (start=100)
    // J2: chr1:400-500 unique
    std::vector<std::string> sj_names = {"chr1:100-200", "chr1:100-300", "chr1:400-500"};
    auto events = build_splice_events(sj_names);

    CHECK(events.size() == 1, "one event created");
    if (!events.empty()) {
        CHECK(events[0].event_type == "alt_donor", "type is alt_donor");
        CHECK(events[0].event_id == "chr1:100-D", "event_id correct");
        CHECK(events[0].junction_indices.size() == 2, "two junctions");
    }
}

// ── Test 3: build_splice_events — shared acceptor ───────────────────────────
static void test_build_splice_events_acceptor() {
    std::cout << "Test 3: build_splice_events — shared acceptor\n";

    // J0: chr1:100-200, J1: chr1:150-200 share acceptor (end=200)
    std::vector<std::string> sj_names = {"chr1:100-200", "chr1:150-200"};
    auto events = build_splice_events(sj_names);

    CHECK(events.size() == 1, "one acceptor event");
    if (!events.empty()) {
        CHECK(events[0].event_type == "alt_acceptor", "type is alt_acceptor");
        CHECK(events[0].event_id == "chr1:200-A", "acceptor event_id correct");
        CHECK(events[0].junction_indices.size() == 2, "two junctions");
    }
}

// ── Test 4: compute_psi — basic PSI values ───────────────────────────────────
static void test_compute_psi_basic() {
    std::cout << "Test 4: compute_psi — basic PSI values\n";

    // J0: chr1:100-200, J1: chr1:100-300  (shared donor at 100)
    // Cell 0: J0=3, J1=7  → PSI[J0]=0.3, PSI[J1]=0.7
    // Cell 1: J0=10        → PSI[J0]=1.0  (J1 absent)
    std::vector<std::string> sj_names = {"chr1:100-200", "chr1:100-300"};

    MockCSC sj_csc;
    sj_csc.nrows = 2;
    sj_csc.ncols = 2;
    sj_csc.indptr = {0, 2, 3};
    sj_csc.indices = {0, 1, 0};
    sj_csc.data = {3, 7, 10};

    auto result = compute_psi(sj_csc, sj_names);

    CHECK(result.nrows == 2, "nrows == 2");
    CHECK(result.ncols == 2, "ncols == 2");
    CHECK(result.events.size() == 1, "one event");
    CHECK(result.indptr.size() == 3, "indptr size");
    CHECK(result.indptr[1] - result.indptr[0] == 2, "cell 0 has 2 entries");
    CHECK(result.indptr[2] - result.indptr[1] == 1, "cell 1 has 1 entry");

    // Locate PSI values for cell 0.
    float psi_j0_c0 = -1.0f, psi_j1_c0 = -1.0f;
    for (int32_t k = result.indptr[0]; k < result.indptr[1]; ++k) {
        if (result.indices[k] == 0) psi_j0_c0 = result.data[k];
        if (result.indices[k] == 1) psi_j1_c0 = result.data[k];
    }
    CHECK_NEAR(psi_j0_c0, 0.3f, 1e-5f, "PSI[J0,c0] == 0.3");
    CHECK_NEAR(psi_j1_c0, 0.7f, 1e-5f, "PSI[J1,c0] == 0.7");

    // Cell 1: only J0.
    float psi_j0_c1 = -1.0f;
    for (int32_t k = result.indptr[1]; k < result.indptr[2]; ++k) {
        if (result.indices[k] == 0) psi_j0_c1 = result.data[k];
    }
    CHECK_NEAR(psi_j0_c1, 1.0f, 1e-5f, "PSI[J0,c1] == 1.0");
}

// ── Test 5: junction not in any event → no PSI entry ────────────────────────
static void test_psi_no_event() {
    std::cout << "Test 5: lone junction → no PSI\n";

    std::vector<std::string> sj_names = {"chr1:100-200"};

    MockCSC sj_csc;
    sj_csc.nrows = 1;
    sj_csc.ncols = 1;
    sj_csc.indptr = {0, 1};
    sj_csc.indices = {0};
    sj_csc.data = {5};

    auto result = compute_psi(sj_csc, sj_names);

    CHECK(result.events.empty(), "no events for single junction");
    CHECK(result.data.empty(), "no PSI entries emitted");
}

// ── Test 6: PSI values in [0, 1] and sum to 1.0 within event ────────────────
static void test_psi_range_and_sum() {
    std::cout << "Test 6: PSI values in [0,1] and sum to 1.0\n";

    // 3 junctions sharing a donor → one event, one cell.
    std::vector<std::string> sj_names = {"chr1:100-200", "chr1:100-300", "chr1:100-400"};

    MockCSC sj_csc;
    sj_csc.nrows = 3;
    sj_csc.ncols = 1;
    sj_csc.indptr = {0, 3};
    sj_csc.indices = {0, 1, 2};
    sj_csc.data = {1000, 500, 1};  // sum = 1501

    auto result = compute_psi(sj_csc, sj_names);

    CHECK(result.data.size() == 3, "three PSI entries");

    bool all_in_range = true;
    float total = 0.0f;
    for (float v : result.data) {
        if (v < 0.0f || v > 1.0f) all_in_range = false;
        total += v;
    }
    CHECK(all_in_range, "all PSI in [0,1]");
    CHECK_NEAR(total, 1.0f, 1e-4f, "PSI sums to 1.0 within event");
}

// ── Test 7: events need ≥2 junctions ────────────────────────────────────────
static void test_min_junctions_per_event() {
    std::cout << "Test 7: events require >= 2 junctions\n";

    // Only one junction per group → no events.
    std::vector<std::string> sj_names = {"chr1:100-200", "chr2:300-400"};
    auto events = build_splice_events(sj_names);
    CHECK(events.empty(), "no events when all junctions are unique");
}

// ── Test 8: write_splice_events produces valid TSV ───────────────────────────
static void test_write_splice_events() {
    std::cout << "Test 8: write_splice_events\n";

    std::vector<SpliceEvent> events;
    SpliceEvent ev;
    ev.event_id = "chr1:100-D";
    ev.event_type = "alt_donor";
    ev.junction_indices = {0, 1};
    events.push_back(ev);

    std::vector<std::string> sj_names = {"chr1:100-200", "chr1:100-300"};

    const std::string path = "/tmp/test_splice_events_gpsi.tsv";
    bool ok = write_splice_events(path, events, sj_names);
    CHECK(ok, "write_splice_events succeeds");

    std::ifstream f(path);
    CHECK(f.good(), "output file readable");
    std::string header, line1;
    std::getline(f, header);
    std::getline(f, line1);
    CHECK(header.find("event_id") != std::string::npos, "header has event_id");
    CHECK(header.find("event_type") != std::string::npos, "header has event_type");
    CHECK(line1.find("chr1:100-D") != std::string::npos, "event_id in line");
    CHECK(line1.find("alt_donor") != std::string::npos, "event_type in line");
    CHECK(line1.find("chr1:100-200") != std::string::npos, "junction name in line");
    CHECK(line1.find("chr1:100-300") != std::string::npos, "second junction name in line");
}

// ── Test 9: empty input → no crash ──────────────────────────────────────────
static void test_empty_input() {
    std::cout << "Test 9: empty sj_csc → no crash\n";

    MockCSC empty;
    empty.nrows = 0;
    empty.ncols = 0;
    empty.indptr = {0};

    auto result = compute_psi(empty, {});
    CHECK(result.data.empty(), "no PSI data for empty input");
    CHECK(result.events.empty(), "no events for empty input");
}

// ── main ─────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== G-PSI: splice_psi unit tests ===\n\n";

    test_parse_junction_name();
    test_build_splice_events_donor();
    test_build_splice_events_acceptor();
    test_compute_psi_basic();
    test_psi_no_event();
    test_psi_range_and_sum();
    test_min_junctions_per_event();
    test_write_splice_events();
    test_empty_input();

    std::cout << "\n=== " << n_pass << " passed, " << n_fail << " failed ===\n";
    return n_fail > 0 ? 1 : 0;
}
