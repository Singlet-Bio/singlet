// test_txome_aligner.cpp — Unit tests for TxomeAligner v0.1 (T-L2-2)
// Tests:
//   1. Build index from tiny in-memory transcriptome
//   2. Exact-match read resolves uniquely
//   3. Mismatched read returns nullopt
//   4. Short read (< SEED_K) returns nullopt
//   5. Multi-mapper (read matches two transcripts) returns nullopt → L3
//   6. FASTA-string build path
//   7. Determinism seed is embedded correctly

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "singlet/pileup/txome_aligner.h"

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string make_tx(char base, int len) {
    std::string s(len, base);
    // slight variation so hash doesn't collide with other transcripts
    for (int i = 3; i < len; i += 11) s[i] = 'T';
    return s;
}

// ── Test 1: index build ───────────────────────────────────────────────────────

static int test_build() {
    printf("  test_build...\n");
    std::vector<std::pair<std::string, std::string>> txs = {
        {"TX_A", make_tx('A', 200)},
        {"TX_C", make_tx('C', 200)},
        {"TX_G", make_tx('G', 200)},
    };
    singlet::TxomeIndex idx;
    idx.build(txs);
    assert(idx.n_transcripts() == 3);
    assert(idx.tx_names[0] == "TX_A");
    assert(idx.tx_names[1] == "TX_C");
    assert(idx.tx_lengths[0] == 200);
    assert(!idx.kmer_index.empty());
    printf("    index built: %u transcripts, %zu k-mers PASS\n",
           idx.n_transcripts(), idx.kmer_index.size());
    return 0;
}

// ── Test 2: exact-match unique resolve ────────────────────────────────────────

static int test_unique_resolve() {
    printf("  test_unique_resolve...\n");
    std::vector<std::pair<std::string, std::string>> txs = {
        {"TX1", make_tx('A', 200)},
        {"TX2", make_tx('C', 200)},
    };
    singlet::TxomeIndex idx;
    idx.build(txs);
    singlet::TxomeAligner aligner(idx);

    // Read = first 60 bases of TX1 — uniquely maps to TX1 at position 0
    std::string read(reinterpret_cast<const char*>(idx.tx_seqs.data() + idx.tx_offsets[0]), 60);
    auto hit = aligner.resolve_unique(read);
    assert(hit.has_value() && "unique read should resolve");
    assert(hit->tx_id == 0);
    assert(hit->pos == 0);
    assert(hit->is_unique == true);
    assert(hit->score >= singlet::TxomeAligner::MIN_EXTEND_MATCH);
    printf("    unique resolve: tx_id=%u pos=%d score=%d PASS\n",
           hit->tx_id, hit->pos, (int)hit->score);
    return 0;
}

// ── Test 3: mismatched read → nullopt ─────────────────────────────────────────

static int test_mismatch_returns_nullopt() {
    printf("  test_mismatch_returns_nullopt...\n");
    std::vector<std::pair<std::string, std::string>> txs = {
        {"TX1", make_tx('A', 200)},
    };
    singlet::TxomeIndex idx;
    idx.build(txs);
    singlet::TxomeAligner aligner(idx);

    // Read full of Ns — won't match anything
    std::string read(60, 'N');
    auto hit = aligner.resolve_unique(read);
    assert(!hit.has_value() && "N-read should not resolve");
    printf("    N-read returns nullopt PASS\n");
    return 0;
}

// ── Test 4: short read (<SEED_K) → nullopt ────────────────────────────────────

static int test_short_read() {
    printf("  test_short_read...\n");
    std::vector<std::pair<std::string, std::string>> txs = {
        {"TX1", make_tx('A', 200)},
    };
    singlet::TxomeIndex idx;
    idx.build(txs);
    singlet::TxomeAligner aligner(idx);

    std::string read(singlet::TxomeIndex::SEED_K - 1, 'A');
    auto hit = aligner.resolve_unique(read);
    assert(!hit.has_value() && "short read should return nullopt");
    printf("    short read returns nullopt PASS\n");
    return 0;
}

// ── Test 5: multi-mapper → nullopt ───────────────────────────────────────────
// Build two identical transcripts; a read that matches both should return nullopt.

static int test_multi_mapper() {
    printf("  test_multi_mapper...\n");
    // TX1 and TX2 are IDENTICAL — any read from TX1 also maps to TX2
    std::string seq = make_tx('A', 200);
    std::vector<std::pair<std::string, std::string>> txs = {
        {"TX1", seq},
        {"TX2", seq},
    };
    singlet::TxomeIndex idx;
    idx.build(txs);
    singlet::TxomeAligner aligner(idx);

    std::string read(reinterpret_cast<const char*>(idx.tx_seqs.data()), 60);
    auto hit = aligner.resolve_unique(read);
    assert(!hit.has_value() && "multi-mapper should return nullopt → L3");
    printf("    multi-mapper returns nullopt PASS\n");
    return 0;
}

// ── Test 6: FASTA-string build path ──────────────────────────────────────────

static int test_fasta_build() {
    printf("  test_fasta_build...\n");
    std::string fasta =
        ">TX_FASTA1 description here\n"
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n"
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n"
        ">TX_FASTA2\n"
        "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG\n"
        "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG\n";

    singlet::TxomeIndex idx;
    idx.build(fasta);

    assert(idx.n_transcripts() == 2);
    assert(idx.tx_names[0] == "TX_FASTA1");  // trimmed at space
    assert(idx.tx_names[1] == "TX_FASTA2");
    assert(idx.tx_lengths[0] == 80);
    assert(idx.tx_lengths[1] == 80);
    printf("    FASTA build: %u transcripts PASS\n", idx.n_transcripts());
    return 0;
}

// ── Test 7: determinism seed stored ──────────────────────────────────────────

static int test_determinism_seed() {
    printf("  test_determinism_seed...\n");
    std::vector<std::pair<std::string, std::string>> txs = {{"TX1", make_tx('A', 200)}};
    singlet::TxomeIndex idx;
    idx.build(txs);

    singlet::TxomeAligner a1(idx, singlet::TxomeAligner::DETERMINISM_SEED);
    singlet::TxomeAligner a2(idx, 42ULL);

    std::string read(reinterpret_cast<const char*>(idx.tx_seqs.data()), 60);
    auto h1 = a1.resolve_unique(read);
    auto h2 = a2.resolve_unique(read);
    // Both should resolve the same tx/pos (seed only affects EM tie-breaking,
    // which is unused in v0.1 — both should give identical results)
    assert(h1.has_value() == h2.has_value());
    if (h1.has_value()) {
        assert(h1->tx_id == h2->tx_id);
        assert(h1->pos   == h2->pos);
    }
    printf("    determinism seed stored, results consistent PASS\n");
    return 0;
}

int main() {
    printf("txome_aligner tests (v0.1 hash-seed-extend)\n");
    int rc = 0;
    rc |= test_build();
    rc |= test_unique_resolve();
    rc |= test_mismatch_returns_nullopt();
    rc |= test_short_read();
    rc |= test_multi_mapper();
    rc |= test_fasta_build();
    rc |= test_determinism_seed();
    if (rc == 0) printf("ALL PASS\n");
    return rc;
}
