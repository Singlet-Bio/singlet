// test/test_transcript_compat.cpp
// Unit tests for G-TXLEVEL: transcript-level equivalence class counting.
//
// Tests:
//   1. EC from single exon (shared by two transcripts) → EC {tx0, tx1}
//   2. EC from tx-specific exon                        → EC {tx0}
//   3. TCC matrix counts and indptr correctness
//   4. EC registry deduplication (same EC observed twice)
//   5. Exon with no transcript annotation is skipped
//   6. build_from_data + add_from_exon_csc convenience path
//   7. write_tcc_matrix / write_ec_table / write_transcript_table (smoke test)

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "singlet-pileup/transcript_compat.h"

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

// ── Test fixture ──────────────────────────────────────────────────────────────
// Gene A: tx0 (exons 0+1), tx1 (exons 0+2)   → exon 0: {tx0,tx1}; exon 1: {tx0}; exon 2: {tx1}
// Gene B: tx2 (exon 3)                         → exon 3: {tx2}
// exon_assignments_[exon_idx] = sorted tx indices
static TranscriptModel make_model() {
    std::vector<std::pair<std::string, std::string>> txs = {
        {"tx1", "GENE_A"},  // tx_idx 0
        {"tx2", "GENE_A"},  // tx_idx 1
        {"tx3", "GENE_B"},  // tx_idx 2
    };
    std::vector<std::vector<uint32_t>> ea = {
        {0, 1},  // exon 0: tx0, tx1 (shared exon in Gene A)
        {0},     // exon 1: tx0 only (tx1-specific in Gene A)
        {1},     // exon 2: tx1 only (tx2-specific in Gene A)
        {2},     // exon 3: tx2 only (Gene B)
    };
    TranscriptModel tm;
    tm.build_from_data(txs, ea);
    return tm;
}

// ── Test 1: shared exon → EC {tx0, tx1} ──────────────────────────────────────
static void test_shared_exon_ec() {
    std::cout << "Test 1: shared exon => EC {tx0, tx1}\n";
    auto tm = make_model();

    const auto& ec0 = tm.exon_to_transcripts(0);
    CHECK(ec0.size() == 2, "exon 0 has 2 transcripts");
    CHECK(!ec0.empty() && ec0[0] == 0, "exon 0 tx[0] == 0");
    CHECK(ec0.size() == 2 && ec0[1] == 1, "exon 0 tx[1] == 1");
}

// ── Test 2: tx-specific exon ──────────────────────────────────────────────────
static void test_specific_exon_ec() {
    std::cout << "Test 2: tx-specific exon => EC {tx0}\n";
    auto tm = make_model();

    const auto& ec1 = tm.exon_to_transcripts(1);
    CHECK(ec1.size() == 1, "exon 1 has 1 transcript");
    CHECK(!ec1.empty() && ec1[0] == 0, "exon 1 tx == 0");

    const auto& ec3 = tm.exon_to_transcripts(3);
    CHECK(ec3.size() == 1, "exon 3 has 1 transcript (Gene B)");
    CHECK(!ec3.empty() && ec3[0] == 2, "exon 3 tx == 2 (tx3)");
}

// ── Test 3: TCC matrix counts ─────────────────────────────────────────────────
// Cell 0: 3 reads in exon 0 → EC {tx0,tx1}=3
//          2 reads in exon 1 → EC {tx0}=2
//          1 read  in exon 3 → EC {tx2}=1
// Cell 1: 5 reads in exon 3 → EC {tx2}=5
static void test_tcc_matrix_counts() {
    std::cout << "Test 3: TCC matrix counts\n";
    auto tm = make_model();
    TCCBuilder builder;

    // Cell 0
    builder.add_exon_count(0, 0, 3, tm);  // exon 0 → EC {tx0,tx1}
    builder.add_exon_count(0, 1, 2, tm);  // exon 1 → EC {tx0}
    builder.add_exon_count(0, 3, 1, tm);  // exon 3 → EC {tx2}
    // Cell 1
    builder.add_exon_count(1, 3, 5, tm);  // exon 3 → EC {tx2}

    auto tcc = builder.build(2, tm);

    CHECK(tcc.n_cells == 2, "n_cells == 2");
    CHECK(tcc.n_ec >= 3, "n_ec >= 3");  // at least 3 distinct ECs

    // indptr[0]=0, indptr[1]=3 (cell 0 has 3 ECs), indptr[2]=4
    CHECK(tcc.indptr[0] == 0, "indptr[0] == 0");
    CHECK(tcc.indptr[1] == 3, "indptr[1] == 3  (cell 0 has 3 entries)");
    CHECK(tcc.indptr[2] == 4, "indptr[2] == 4  (cell 1 has 1 entry)");
    CHECK(tcc.data.size() == 4, "total nnz == 4");

    // Cell 0 counts must sum to 6
    uint32_t cell0_sum = 0;
    for (int k = tcc.indptr[0]; k < tcc.indptr[1]; ++k) cell0_sum += tcc.data[k];
    CHECK(cell0_sum == 6, "cell 0 count sum == 6");

    // Cell 1 count == 5
    CHECK(tcc.data[3] == 5, "cell 1 count == 5");
}

// ── Test 4: EC registry deduplication ────────────────────────────────────────
// Two adds with the same exon_idx should map to the same EC; counts accumulate.
static void test_ec_dedup() {
    std::cout << "Test 4: EC deduplication\n";
    auto tm = make_model();
    TCCBuilder builder;

    builder.add_exon_count(0, 0, 2, tm);  // EC {tx0,tx1}, count 2
    builder.add_exon_count(0, 0, 3, tm);  // same EC, count 3 → total 5

    auto tcc = builder.build(1, tm);
    CHECK(tcc.n_ec == 1, "exactly 1 EC created");
    CHECK(tcc.data.size() == 1, "one non-zero entry");
    CHECK(!tcc.data.empty() && tcc.data[0] == 5, "accumulated count == 5");
}

// ── Test 5: Exon with no transcript annotation skipped ────────────────────────
static void test_no_annotation_skipped() {
    std::cout << "Test 5: exon with no tx annotation is skipped\n";
    auto tm = make_model();
    TCCBuilder builder;

    // exon_idx 99 doesn't exist in the model
    builder.add_exon_count(0, 99, 10, tm);  // should be silently ignored

    auto tcc = builder.build(1, tm);
    CHECK(tcc.n_ec == 0, "no ECs created");
    CHECK(tcc.data.empty(), "no data");
    CHECK(tm.n_transcripts() == 3, "model still has 3 transcripts");
    (void)tcc;
}

// ── Test 6: add_from_exon_csc convenience path ───────────────────────────────
static void test_add_from_exon_csc() {
    std::cout << "Test 6: add_from_exon_csc convenience path\n";
    auto tm = make_model();
    TCCBuilder builder;

    // CSC: 4 exons × 2 cells
    // Cell 0: exon 0 = 4, exon 3 = 1
    // Cell 1: exon 1 = 2, exon 2 = 3
    std::vector<int32_t> indptr = {0, 2, 4};
    std::vector<int32_t> indices = {0, 3, 1, 2};
    std::vector<uint32_t> data = {4, 1, 2, 3};

    builder.add_from_exon_csc(indptr, indices, data, 2, tm);
    auto tcc = builder.build(2, tm);

    CHECK(tcc.indptr[1] == 2, "cell 0 has 2 ECs");
    CHECK(tcc.indptr[2] == 4, "cell 1 has 2 ECs");

    uint32_t sum0 = 0, sum1 = 0;
    for (int k = tcc.indptr[0]; k < tcc.indptr[1]; ++k) sum0 += tcc.data[k];
    for (int k = tcc.indptr[1]; k < tcc.indptr[2]; ++k) sum1 += tcc.data[k];
    CHECK(sum0 == 5, "cell 0 total == 5");
    CHECK(sum1 == 5, "cell 1 total == 5");
}

// ── Test 7: output writers smoke test ─────────────────────────────────────────
static void test_writers() {
    std::cout << "Test 7: output writers smoke test\n";
    auto tm = make_model();
    TCCBuilder builder;
    builder.add_exon_count(0, 0, 2, tm);
    builder.add_exon_count(0, 1, 1, tm);
    builder.add_exon_count(1, 3, 3, tm);
    auto tcc = builder.build(2, tm);

    const std::string mtx_path = "/tmp/test_tcc_matrix.mtx";
    const std::string ec_path = "/tmp/test_tcc_ec.tsv";
    const std::string tx_path = "/tmp/test_tcc_tx.tsv";

    CHECK(write_tcc_matrix(mtx_path, tcc), "write_tcc_matrix ok");
    CHECK(write_ec_table(ec_path, tcc), "write_ec_table ok");
    CHECK(write_transcript_table(tx_path, tcc), "write_transcript_table ok");

    // Verify mtx header
    std::ifstream mtx(mtx_path);
    std::string first_line;
    std::getline(mtx, first_line);
    CHECK(first_line.find("MatrixMarket") != std::string::npos, "mtx has MatrixMarket header");

    // Verify EC table has correct number of lines
    std::ifstream ec(ec_path);
    int ec_lines = 0;
    std::string tmp;
    while (std::getline(ec, tmp)) ++ec_lines;
    CHECK(ec_lines == static_cast<int>(tcc.n_ec), "ec_table line count == n_ec");

    // Verify transcript table has correct number of lines
    std::ifstream txf(tx_path);
    int tx_lines = 0;
    while (std::getline(txf, tmp)) ++tx_lines;
    CHECK(tx_lines == 3, "tx_table has 3 lines");

    // Cleanup
    std::remove(mtx_path.c_str());
    std::remove(ec_path.c_str());
    std::remove(tx_path.c_str());
}

// ── EC definition content verification ───────────────────────────────────────
static void test_ec_definitions() {
    std::cout << "Test 8: EC definitions have correct transcript sets\n";
    auto tm = make_model();
    TCCBuilder builder;
    builder.add_exon_count(0, 0, 1, tm);  // EC for {tx0, tx1}
    builder.add_exon_count(0, 1, 1, tm);  // EC for {tx0}
    builder.add_exon_count(0, 3, 1, tm);  // EC for {tx2}
    auto tcc = builder.build(1, tm);

    CHECK(tcc.equiv_classes.size() == 3, "3 distinct ECs");

    // Find the EC for the shared exon ({tx0, tx1} = {0, 1})
    bool found_shared = false;
    for (const auto& ec : tcc.equiv_classes) {
        if (ec.transcript_ids.size() == 2 &&
            ec.transcript_ids[0] == 0 && ec.transcript_ids[1] == 1) {
            found_shared = true;
        }
    }
    CHECK(found_shared, "EC {tx0,tx1} found in equiv_classes");

    // Find the EC for Gene B ({tx2} = {2})
    bool found_geneb = false;
    for (const auto& ec : tcc.equiv_classes) {
        if (ec.transcript_ids.size() == 1 && ec.transcript_ids[0] == 2) {
            found_geneb = true;
        }
    }
    CHECK(found_geneb, "EC {tx2} found in equiv_classes");
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== G-TXLEVEL: transcript_compat tests ===\n";
    test_shared_exon_ec();
    test_specific_exon_ec();
    test_tcc_matrix_counts();
    test_ec_dedup();
    test_no_annotation_skipped();
    test_add_from_exon_csc();
    test_writers();
    test_ec_definitions();
    std::cout << "\n"
              << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail ? 1 : 0;
}
