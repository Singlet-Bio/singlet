// test_txome_gene_index.cpp — Unit tests for txome_gene_index.h (L1 cascade)
// Tests: 2-bit encoding, FlatKmerHash, KmerBloomFilter, resolve_gene, save/load round-trip.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "singlet/pileup/txome_gene_index.h"

using namespace singlet;

static int g_pass = 0, g_fail = 0;

static void CHECK(bool cond, const char* name) {
    if (cond) { ++g_pass; }
    else { ++g_fail; std::fprintf(stderr, "FAIL: %s\n", name); }
}

// ── Test 1: base_to_2bit encoding ────────────────────────────────────────────

static void test_base_to_2bit() {
    CHECK(base_to_2bit('A') == 0, "A=0");
    CHECK(base_to_2bit('C') == 1, "C=1");
    CHECK(base_to_2bit('G') == 2, "G=2");
    CHECK(base_to_2bit('T') == 3, "T=3");
    CHECK(base_to_2bit('a') == 0, "a=0");
    CHECK(base_to_2bit('c') == 1, "c=1");
    CHECK(base_to_2bit('g') == 2, "g=2");
    CHECK(base_to_2bit('t') == 3, "t=3");
    CHECK(base_to_2bit('N') == 0xFF, "N=0xFF");
    CHECK(base_to_2bit('X') == 0xFF, "X=0xFF");
    CHECK(base_to_2bit('\0') == 0xFF, "null=0xFF");
}

// ── Test 2: encode_kmer_2bit ─────────────────────────────────────────────────

static void test_encode_kmer_2bit() {
    // "AA" with k=2 → 0b0000 = 0
    CHECK(encode_kmer_2bit("AA", 2) == 0, "AA=0");
    // "AC" with k=2 → 0b0001 = 1
    CHECK(encode_kmer_2bit("AC", 2) == 1, "AC=1");
    // "CA" with k=2 → 0b0100 = 4
    CHECK(encode_kmer_2bit("CA", 2) == 4, "CA=4");
    // "TT" with k=2 → 0b1111 = 15
    CHECK(encode_kmer_2bit("TT", 2) == 15, "TT=15");
    // "ACGT" with k=4 → 0b00011011 = 27
    CHECK(encode_kmer_2bit("ACGT", 4) == 0b00011011, "ACGT=27");
    // N-containing → UINT64_MAX
    CHECK(encode_kmer_2bit("AN", 2) == UINT64_MAX, "AN=invalid");
    CHECK(encode_kmer_2bit("NA", 2) == UINT64_MAX, "NA=invalid");
    // Single base k=1
    CHECK(encode_kmer_2bit("G", 1) == 2, "G_k1=2");
}

// ── Test 3: FlatKmerHash basic operations ────────────────────────────────────

static void test_flat_kmer_hash() {
    FlatKmerHash hash;

    // Build with 5 entries
    std::vector<KmerGeneEntry> entries = {
        {100, 0}, {200, 1}, {300, 2}, {400, 0}, {500, 3}
    };
    hash.build(entries);

    CHECK(hash.size() == 5, "hash_size=5");
    CHECK(hash.capacity() >= 10, "hash_cap>=10");  // ~50% load

    // All entries should be findable
    CHECK(hash.find(100) == 0, "find(100)=0");
    CHECK(hash.find(200) == 1, "find(200)=1");
    CHECK(hash.find(300) == 2, "find(300)=2");
    CHECK(hash.find(400) == 0, "find(400)=0");
    CHECK(hash.find(500) == 3, "find(500)=3");

    // Non-existent keys return UINT32_MAX
    CHECK(hash.find(101) == UINT32_MAX, "find(101)=missing");
    CHECK(hash.find(0) == UINT32_MAX, "find(0)=missing");
    CHECK(hash.find(999) == UINT32_MAX, "find(999)=missing");
}

// ── Test 4: FlatKmerHash with many entries ───────────────────────────────────

static void test_flat_kmer_hash_large() {
    FlatKmerHash hash;

    // Build with 1000 entries
    std::vector<KmerGeneEntry> entries;
    for (uint32_t i = 0; i < 1000; ++i) {
        entries.push_back({static_cast<uint64_t>(i * 17 + 3), i % 50});
    }
    hash.build(entries);

    CHECK(hash.size() == 1000, "large_hash_size=1000");

    // Verify all entries
    bool all_found = true;
    for (auto& e : entries) {
        if (hash.find(e.kmer_2bit) != e.gene_id) {
            all_found = false;
            break;
        }
    }
    CHECK(all_found, "large_hash_all_found");

    // Non-existent
    CHECK(hash.find(UINT64_MAX - 1) == UINT32_MAX, "large_hash_missing");
}

// ── Test 5: KmerBloomFilter basic operations ─────────────────────────────────

static void test_bloom_filter() {
    KmerBloomFilter bloom;

    std::vector<KmerGeneEntry> entries;
    for (uint32_t i = 0; i < 100; ++i) {
        entries.push_back({static_cast<uint64_t>(i * 1000007), i});
    }
    bloom.build(entries);

    CHECK(bloom.memory_bytes() > 0, "bloom_mem>0");

    // All inserted k-mers must be found (no false negatives)
    bool all_found = true;
    for (auto& e : entries) {
        if (!bloom.may_contain(e.kmer_2bit)) {
            all_found = false;
            break;
        }
    }
    CHECK(all_found, "bloom_no_false_negatives");

    // False positive rate should be low (test 1000 random non-members)
    int false_pos = 0;
    for (uint64_t i = 0; i < 1000; ++i) {
        uint64_t key = (i + 1) * 999999937ULL; // prime, different from inserted
        if (bloom.may_contain(key)) ++false_pos;
    }
    // With 100 entries in 2B bits, FPR should be essentially 0
    CHECK(false_pos < 50, "bloom_low_fpr");  // Allow up to 5% FPR
}

// ── Test 6: TxomeGeneIndex save/load round-trip ──────────────────────────────

static void test_save_load_roundtrip() {
    TxomeGeneIndex idx;
    idx.k_ = 10;
    idx.gene_names_ = {"GAPDH", "TP53", "BRCA1", "MYC", "EGFR"};

    // Build some fake k-mer entries
    for (uint32_t g = 0; g < 5; ++g) {
        for (uint64_t k = 0; k < 20; ++k) {
            idx.entries_.push_back({g * 1000 + k * 7, g});
        }
    }
    std::sort(idx.entries_.begin(), idx.entries_.end());

    // Build internal hash structures
    idx.kmer_hash_.build(idx.entries_);
    idx.bloom_.build(idx.entries_);

    // Save
    std::string path = "/tmp/test_txgi_roundtrip.txgi";
    idx.save(path);

    // Load into new index
    TxomeGeneIndex idx2;
    bool ok = idx2.load(path);
    CHECK(ok, "load_success");
    CHECK(idx2.k_ == 10, "k_roundtrip");
    CHECK(idx2.n_genes() == 5, "n_genes_roundtrip");
    CHECK(idx2.n_kmers() == 100, "n_kmers_roundtrip");

    // Gene names
    CHECK(idx2.gene_name(0) == "GAPDH", "gene0=GAPDH");
    CHECK(idx2.gene_name(4) == "EGFR", "gene4=EGFR");

    // Entries match
    bool entries_match = (idx2.entries_.size() == idx.entries_.size());
    if (entries_match) {
        for (size_t i = 0; i < idx.entries_.size(); ++i) {
            if (idx2.entries_[i].kmer_2bit != idx.entries_[i].kmer_2bit ||
                idx2.entries_[i].gene_id != idx.entries_[i].gene_id) {
                entries_match = false;
                break;
            }
        }
    }
    CHECK(entries_match, "entries_roundtrip");

    // Hash table works after load
    CHECK(idx2.kmer_hash_.find(idx.entries_[0].kmer_2bit) == idx.entries_[0].gene_id,
          "hash_works_after_load");

    std::remove(path.c_str());
}

// ── Test 7: load nonexistent file ────────────────────────────────────────────

static void test_load_nonexistent() {
    TxomeGeneIndex idx;
    bool ok = idx.load("/tmp/test_txgi_DOES_NOT_EXIST.txgi");
    CHECK(!ok, "load_nonexistent=false");
}

// ── Test 8: load corrupt file ────────────────────────────────────────────────

static void test_load_corrupt() {
    std::string path = "/tmp/test_txgi_corrupt.txgi";
    {
        std::ofstream f(path, std::ios::binary);
        char bad[24] = {};
        f.write(bad, 24); // wrong magic
    }
    TxomeGeneIndex idx;
    bool ok = idx.load(path);
    CHECK(!ok, "load_corrupt=false");
    std::remove(path.c_str());
}

// ── Test 9: resolve_gene with small index ────────────────────────────────────

static void test_resolve_gene_small() {
    TxomeGeneIndex idx;
    idx.k_ = 5;
    idx.gene_names_ = {"GeneA", "GeneB"};

    // Build entries for specific 5-mers
    // "ACGTC" → encode_kmer_2bit("ACGTC", 5) = 0b00 01 10 11 01 = 0x6D = 109
    uint64_t k1 = encode_kmer_2bit("ACGTC", 5);
    // "TTTTT" → 0b11 11 11 11 11 = 0x3FF = 1023
    uint64_t k2 = encode_kmer_2bit("TTTTT", 5);

    idx.entries_ = {{k1, 0}, {k2, 1}};
    std::sort(idx.entries_.begin(), idx.entries_.end());
    idx.kmer_hash_.build(idx.entries_);
    idx.bloom_.build(idx.entries_);

    // Read that contains "ACGTC" → should resolve to GeneA (id=0)
    const char* read1 = "GGACGTCGG"; // ACGTC at positions 2-6
    uint32_t g1 = idx.resolve_gene(read1, strlen(read1));
    CHECK(g1 == 0, "resolve_contains_kmer=GeneA");

    // Read that contains "TTTTT" → should resolve to GeneB (id=1)
    const char* read2 = "AATTTTTCC";
    uint32_t g2 = idx.resolve_gene(read2, strlen(read2));
    CHECK(g2 == 1, "resolve_contains_kmer=GeneB");

    // Read that contains neither → UINT32_MAX
    const char* read3 = "AAAACCCCC";
    uint32_t g3 = idx.resolve_gene(read3, strlen(read3));
    CHECK(g3 == UINT32_MAX, "resolve_no_match");

    // Read shorter than k → UINT32_MAX
    const char* read4 = "ACGT";
    uint32_t g4 = idx.resolve_gene(read4, strlen(read4));
    CHECK(g4 == UINT32_MAX, "resolve_too_short");

    // N in read should break k-mer chain
    const char* read5 = "ACNGTCAAA";
    uint32_t g5 = idx.resolve_gene(read5, strlen(read5));
    // "ACNGT" has N, so valid k-mer chain resets; "GTCAA" may or may not match
    // The key thing: it doesn't crash
    (void)g5;
    CHECK(true, "resolve_N_no_crash");
}

// ── Test 10: resolve_gene_single ─────────────────────────────────────────────

static void test_resolve_gene_single() {
    TxomeGeneIndex idx;
    idx.k_ = 4;
    idx.gene_names_ = {"X"};

    // "ACGT" → 27
    uint64_t k = encode_kmer_2bit("ACGT", 4);
    idx.entries_ = {{k, 0}};
    idx.kmer_hash_.build(idx.entries_);
    idx.bloom_.build(idx.entries_);

    // resolve_gene_single only checks the FIRST k-mer
    CHECK(idx.resolve_gene_single("ACGT", 4) == 0, "single_exact_match");
    CHECK(idx.resolve_gene_single("ACGTAAA", 7) == 0, "single_prefix_match");
    CHECK(idx.resolve_gene_single("TTTT", 4) == UINT32_MAX, "single_no_match");
    CHECK(idx.resolve_gene_single("ACG", 3) == UINT32_MAX, "single_too_short");
    CHECK(idx.resolve_gene_single("NACG", 4) == UINT32_MAX, "single_N_start");
}

// ── Test 11: KmerGeneEntry sorting ───────────────────────────────────────────

static void test_entry_sorting() {
    std::vector<KmerGeneEntry> entries = {
        {500, 2}, {100, 0}, {300, 1}, {200, 3}, {400, 0}
    };
    std::sort(entries.begin(), entries.end());
    CHECK(entries[0].kmer_2bit == 100, "sort[0]=100");
    CHECK(entries[4].kmer_2bit == 500, "sort[4]=500");
    // Gene IDs preserved
    CHECK(entries[0].gene_id == 0, "sort[0].gene=0");
    CHECK(entries[4].gene_id == 2, "sort[4].gene=2");
}

// ── Test 12: Constants ───────────────────────────────────────────────────────

static void test_constants() {
    CHECK(TxomeGeneIndex::MAGIC == 0x49475854, "magic=TXGI");
    CHECK(TxomeGeneIndex::VERSION == 1, "version=1");
    CHECK(TxomeGeneIndex::DEFAULT_K == 22, "default_k=22");
    CHECK(FlatKmerHash::EMPTY == UINT64_MAX, "empty_sentinel");
}

// ── Test 13: accessor methods ────────────────────────────────────────────────

static void test_accessors() {
    TxomeGeneIndex idx;
    idx.k_ = 15;
    idx.gene_names_ = {"A", "B", "C"};
    idx.entries_.resize(42);

    CHECK(idx.seed_k() == 15, "seed_k=15");
    CHECK(idx.n_genes() == 3, "n_genes=3");
    CHECK(idx.n_kmers() == 42, "n_kmers=42");
    CHECK(idx.gene_name(1) == "B", "gene_name(1)=B");
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    test_base_to_2bit();
    test_encode_kmer_2bit();
    test_flat_kmer_hash();
    test_flat_kmer_hash_large();
    test_bloom_filter();
    test_save_load_roundtrip();
    test_load_nonexistent();
    test_load_corrupt();
    test_resolve_gene_small();
    test_resolve_gene_single();
    test_entry_sorting();
    test_constants();
    test_accessors();

    std::printf("txome_gene_index: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
