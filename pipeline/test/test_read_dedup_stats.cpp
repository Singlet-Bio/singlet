// test/test_read_dedup_stats.cpp
// Unit tests for ReadDedupStats (B4 feature).
// Tests exercise static helpers directly — no external BAM files required.
// A synthetic in-memory BAM is constructed via htslib for process_bam() tests.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "singlet-pileup/read_dedup_stats.h"

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

// ── Test 1: estimate_library_size — zero inputs return 0 ────────────────────
static void test_library_size_zero() {
    std::cout << "Test 1: library size — zero inputs\n";
    CHECK(ReadDedupStats::estimate_library_size(0, 0) == 0.0, "total=0 distinct=0 → 0");
    CHECK(ReadDedupStats::estimate_library_size(100, 0) == 0.0, "distinct=0 → 0");
    CHECK(ReadDedupStats::estimate_library_size(0, 50) == 0.0, "total=0 → 0");
}

// ── Test 2: no duplicates — library size == distinct reads ──────────────────
static void test_library_size_no_dups() {
    std::cout << "Test 2: library size — no duplicates\n";
    double sz = ReadDedupStats::estimate_library_size(1000, 1000);
    // distinct == total → return distinct directly
    CHECK(sz == 1000.0, "no-dup library size = total");
}

// ── Test 3: 50% duplication rate — library size > distinct ──────────────────
static void test_library_size_half_dup() {
    std::cout << "Test 3: library size — 50% duplication\n";
    // total=1000 mapped, 500 distinct: T > 500
    double sz = ReadDedupStats::estimate_library_size(1000, 500);
    CHECK(sz > 500.0, "library size > distinct");
    CHECK(sz < 1e8, "library size finite");
    CHECK(std::isfinite(sz), "library size is finite");
}

// ── Test 4: 90% duplication rate — library size close to distinct (deep sequencing) ──────────────────────────────────────────────────────────────
static void test_library_size_high_dup() {
    std::cout << "Test 4: library size — 90% duplication (deeply sequenced)\n";
    // total=10000, distinct=1000: N/C=10 → already at ~10x depth.
    // Lander-Waterman: T*(1-exp(-10000/T))=1000 → T ≈ 1000 (fully saturated).
    double sz = ReadDedupStats::estimate_library_size(10000, 1000);
    CHECK(std::isfinite(sz), "library size is finite");
    CHECK(sz >= 1000.0, "library size >= distinct reads");
    CHECK(sz < 2000.0, "library size < 2x distinct (saturated regime)");
    // At 25% dup rate with 1000 reads, library is meaningfully larger
    double sz25 = ReadDedupStats::estimate_library_size(1000, 750);
    CHECK(sz25 > 750.0, "25%-dup: library size > distinct");
    CHECK(std::isfinite(sz25), "25%-dup: library size finite");
}

// ── Test 5: parse valid Illumina QNAME ──────────────────────────────────────
static void test_parse_illumina_valid() {
    std::cout << "Test 5: parse Illumina QNAME — valid\n";
    // Format: instrument:run:flowcell:lane:tile:x:y
    const char* name = "HWI-D00723:394:HHNGHBCXY:1:1101:2847:2146";
    auto c = ReadDedupStats::parse_illumina_qname(name);
    CHECK(c.valid, "valid = true");
    CHECK(c.lane == 1, "lane = 1");
    CHECK(c.tile == 1101, "tile = 1101");
    CHECK(c.x == 2847, "x = 2847");
    CHECK(c.y == 2146, "y = 2146");
}

// ── Test 6: parse non-Illumina / malformed QNAME ────────────────────────────
static void test_parse_illumina_invalid() {
    std::cout << "Test 6: parse Illumina QNAME — invalid\n";
    auto c1 = ReadDedupStats::parse_illumina_qname("read1");
    CHECK(!c1.valid, "short name → invalid");

    auto c2 = ReadDedupStats::parse_illumina_qname("");
    CHECK(!c2.valid, "empty name → invalid");

    auto c3 = ReadDedupStats::parse_illumina_qname(nullptr);
    CHECK(!c3.valid, "null name → invalid");
}

// ── Test 7: pixel_distance calculations ─────────────────────────────────────
static void test_pixel_distance() {
    std::cout << "Test 7: pixel_distance\n";
    CHECK(ReadDedupStats::pixel_distance(0, 0, 0, 0) == 0.0, "same point → 0");
    // 3-4-5 right triangle
    double d1 = ReadDedupStats::pixel_distance(0, 0, 3, 4);
    CHECK(std::fabs(d1 - 5.0) < 1e-9, "3-4-5 triangle → 5");
    // distance is symmetric
    double d2 = ReadDedupStats::pixel_distance(100, 200, 103, 204);
    double d3 = ReadDedupStats::pixel_distance(103, 204, 100, 200);
    CHECK(std::fabs(d2 - d3) < 1e-9, "distance is symmetric");
}

// ── Test 8: optical duplicate detection via distance threshold ───────────────
static void test_optical_dup_detection() {
    std::cout << "Test 8: optical duplicate detection\n";
    ReadDedupStats rds;
    ReadDedupStats::Config cfg;
    cfg.optical_distance = 2500;
    rds.set_config(cfg);

    // Two reads on the same tile, close together → optical duplicate
    ReadDedupStats::IlluminaCoords a, b;
    a.valid = b.valid = true;
    a.lane = b.lane = 1;
    a.tile = b.tile = 1101;
    a.x = 1000;
    a.y = 1000;
    b.x = 1100;
    b.y = 1100;  // distance ≈ 141 < 2500
    double d_close = ReadDedupStats::pixel_distance(a.x, a.y, b.x, b.y);
    CHECK(d_close < 2500.0, "close reads: distance < threshold");

    // Same tile but far apart → not optical duplicate
    ReadDedupStats::IlluminaCoords c;
    c.valid = true;
    c.lane = 1;
    c.tile = 1101;
    c.x = 5000;
    c.y = 5000;  // distance ≈ 5657 > 2500
    double d_far = ReadDedupStats::pixel_distance(a.x, a.y, c.x, c.y);
    CHECK(d_far > 2500.0, "far reads: distance > threshold");

    // Different tile → not optical regardless of distance
    ReadDedupStats::IlluminaCoords e;
    e.valid = true;
    e.lane = 1;
    e.tile = 2202;  // different tile
    e.x = 1010;
    e.y = 1010;
    CHECK(a.tile != e.tile, "different tile → cannot be optical dup");
}

// ── Test 9: DedupKey equality and hash consistency ──────────────────────────
static void test_dedup_key_equality() {
    std::cout << "Test 9: DedupKey equality\n";
    ReadDedupStats::DedupKey k1, k2, k3;
    k1.chrom = k2.chrom = 0;
    k3.chrom = 1;
    k1.pos = k2.pos = 100;
    k3.pos = 100;
    k1.mate_pos = k2.mate_pos = 200;
    k3.mate_pos = 200;
    k1.strand = k2.strand = 0;
    k3.strand = 0;
    k1.cigar_sig = k2.cigar_sig = 42;
    k3.cigar_sig = 42;

    CHECK(k1 == k2, "identical keys are equal");
    CHECK(!(k1 == k3), "different chrom → not equal");

    ReadDedupStats::DedupKeyHash hasher;
    CHECK(hasher(k1) == hasher(k2), "equal keys have equal hashes");
}

// ── Test 10: make_dedup_key with synthetic bam1_t record ────────────────────
static void test_make_dedup_key_synthetic() {
    std::cout << "Test 10: make_dedup_key with synthetic bam1_t\n";

    // Build a minimal bam1_t via htslib allocation
    bam1_t* b1 = bam_init1();
    bam1_t* b2 = bam_init1();

    // Set core fields manually (these fields are safe to set directly)
    b1->core.tid = 0;
    b2->core.tid = 0;
    b1->core.pos = 500;
    b2->core.pos = 500;
    b1->core.mpos = 800;
    b2->core.mpos = 800;
    b1->core.flag = 0;
    b2->core.flag = 0;  // both forward
    b1->core.n_cigar = 0;
    b2->core.n_cigar = 0;  // no CIGAR ops

    auto key1 = ReadDedupStats::make_dedup_key(b1);
    auto key2 = ReadDedupStats::make_dedup_key(b2);
    CHECK(key1 == key2, "identical records → same dedup key");

    // Reverse strand read → different key
    b2->core.flag |= BAM_FREVERSE;
    auto key3 = ReadDedupStats::make_dedup_key(b2);
    CHECK(!(key1 == key3), "reverse strand → different key");

    // Null pointer returns default key (no crash)
    auto key_null = ReadDedupStats::make_dedup_key(nullptr);
    CHECK(key_null.chrom == -1, "nullptr → default key chrom -1");

    bam_destroy1(b1);
    bam_destroy1(b2);
}

// ── Test 11: process_bam on missing file returns empty result ────────────────
static void test_process_bam_missing_file() {
    std::cout << "Test 11: process_bam — missing file\n";
    ReadDedupStats rds;
    auto res = rds.process_bam("/nonexistent/path/no.bam");
    CHECK(res.total_reads == 0, "missing file → total_reads = 0");
    CHECK(res.duplicate_reads == 0, "missing file → duplicate_reads = 0");
    CHECK(res.duplication_rate == 0.0, "missing file → duplication_rate = 0");
}

// ── Test 12: process_bam on synthetic BAM (all duplicates) ──────────────────
static void test_process_bam_synthetic() {
    std::cout << "Test 12: process_bam — synthetic BAM (paired duplicates)\n";

    // Write a tiny BAM using htslib directly to a tmp file
    const char* tmpbam = "/dev/shm/test_dedup_synth.bam";

    // Build a minimal header
    sam_hdr_t* hdr = sam_hdr_init();
    sam_hdr_add_line(hdr, "HD", "VN", "1.6", "SO", "coordinate", nullptr);
    sam_hdr_add_line(hdr, "SQ", "SN", "chr1", "LN", "248956422", nullptr);

    htsFile* wfp = hts_open(tmpbam, "wb");
    assert(wfp);
    sam_hdr_write(wfp, hdr);

    // Write 3 reads all at the same position (2 duplicates of read 0)
    // QNAME format: instrument:run:fc:lane:tile:x:y
    const char* qnames[3] = {
        "SIM:1:FCABC:1:1101:1000:1000",  // best (mapq=30)
        "SIM:1:FCABC:1:1101:1010:1010",  // optical dup (mapq=20, dist≈14)
        "SIM:1:FCABC:1:1102:1000:1000",  // PCR dup (different tile, mapq=20)
    };
    int mapqs[3] = {30, 20, 20};

    for (int i = 0; i < 3; ++i) {
        bam1_t* b = bam_init1();
        b->core.tid = 0;
        b->core.pos = 1000;
        b->core.mpos = 1100;
        b->core.flag = BAM_FPAIRED | BAM_FPROPER_PAIR;
        b->core.qual = static_cast<uint8_t>(mapqs[i]);
        b->core.n_cigar = 0;
        b->core.l_qname = static_cast<uint8_t>(std::strlen(qnames[i]) + 1);

        // Allocate data block: l_qname bytes for name
        int data_len = b->core.l_qname;
        b->data = static_cast<uint8_t*>(std::malloc(data_len));
        b->l_data = data_len;
        b->m_data = data_len;
        std::memcpy(b->data, qnames[i], b->core.l_qname);

        sam_write1(wfp, hdr, b);
        bam_destroy1(b);
    }

    hts_close(wfp);
    sam_hdr_destroy(hdr);

    // Now run dedup; sort the tmp BAM coordinate order first (it already is)
    // Add index so htslib can read it
    ReadDedupStats rds;
    ReadDedupStats::Config cfg;
    cfg.optical_distance = 2500;
    rds.set_config(cfg);

    auto res = rds.process_bam(tmpbam);

    CHECK(res.total_reads == 3, "3 total reads");
    CHECK(res.mapped_reads == 3, "3 mapped reads (no unmapped flag)");
    CHECK(res.duplicate_reads == 2, "2 duplicates marked");
    CHECK(res.optical_duplicates == 1, "1 optical dup (same tile, close)");
    CHECK(res.pcr_duplicates == 1, "1 PCR dup (different tile)");
    CHECK(res.duplication_rate > 0.0 && res.duplication_rate < 1.0,
          "duplication_rate in (0,1)");
    CHECK(res.estimated_library_size > 0.0, "library size estimated");

    std::remove(tmpbam);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== test_read_dedup_stats ===\n";
    test_library_size_zero();
    test_library_size_no_dups();
    test_library_size_half_dup();
    test_library_size_high_dup();
    test_parse_illumina_valid();
    test_parse_illumina_invalid();
    test_pixel_distance();
    test_optical_dup_detection();
    test_dedup_key_equality();
    test_make_dedup_key_synthetic();
    test_process_bam_missing_file();
    test_process_bam_synthetic();

    std::cout << "\n=== Results: " << n_pass << " passed, " << n_fail << " failed ===\n";
    return (n_fail == 0) ? 0 : 1;
}
