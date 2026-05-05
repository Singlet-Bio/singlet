// test_wgs_assembly.cpp — Unit tests for singlet/pileup/wgs_assembly.h
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "singlet/pileup/wgs_assembly.h"

using namespace singlet;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (cond) {                                                     \
            ++g_pass;                                                   \
        } else {                                                        \
            ++g_fail;                                                   \
            std::cerr << "FAIL: " << #cond                              \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        }                                                               \
    } while (0)

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static bool approx_eq(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) < tol;
}

// ── 1. WGS mode detection ─────────────────────────────────────────────────────

static void test_wgs_mode_detection() {
    // Short reads, no barcode/UMI → SHORT_READ
    CHECK(detect_wgs_mode(150, false, false, "illumina") == WgsMode::SHORT_READ);
    CHECK(detect_wgs_mode(250, false, false, "") == WgsMode::SHORT_READ);

    // Long reads, PacBio → LONG_READ_HIFI
    CHECK(detect_wgs_mode(15000, false, false, "pacbio") == WgsMode::LONG_READ_HIFI);
    CHECK(detect_wgs_mode(10000, false, false, "hifi") == WgsMode::LONG_READ_HIFI);
    CHECK(detect_wgs_mode(20000, false, false, "revio") == WgsMode::LONG_READ_HIFI);

    // Long reads, ONT → LONG_READ_ONT
    CHECK(detect_wgs_mode(10000, false, false, "ont") == WgsMode::LONG_READ_ONT);
    CHECK(detect_wgs_mode(8000, false, false, "nanopore") == WgsMode::LONG_READ_ONT);
    CHECK(detect_wgs_mode(12000, false, false, "oxford") == WgsMode::LONG_READ_ONT);

    // Long reads, generic (no instrument tag) → LONG_READ_HIFI (default for >1000bp)
    CHECK(detect_wgs_mode(5000, false, false, "unknown") == WgsMode::LONG_READ_HIFI);

    // Has barcode → HYBRID
    CHECK(detect_wgs_mode(150, true, false, "illumina") == WgsMode::HYBRID);

    // Has UMI → HYBRID
    CHECK(detect_wgs_mode(150, false, true, "illumina") == WgsMode::HYBRID);

    // Short reads on ONT/PacBio instrument → HYBRID
    CHECK(detect_wgs_mode(200, false, false, "nanopore") == WgsMode::HYBRID);
}

// ── 2. Assembler selection ────────────────────────────────────────────────────

static void test_assembler_selection() {
    CHECK(select_assembler(WgsMode::SHORT_READ) == AssemblerBackend::SPADES);
    CHECK(select_assembler(WgsMode::LONG_READ_HIFI) == AssemblerBackend::HIFIASM);
    CHECK(select_assembler(WgsMode::LONG_READ_ONT) == AssemblerBackend::FLYE);
    CHECK(select_assembler(WgsMode::HYBRID) == AssemblerBackend::SPADES);
}

// ── 3. Command construction ───────────────────────────────────────────────────

static void test_command_construction() {
    const std::string fq = "reads.fq";
    const std::string out = "/out/dir";

    // hifiasm
    {
        std::string cmd = build_assembler_command(AssemblerBackend::HIFIASM, fq, out, 20, 64);
        CHECK(cmd.find("hifiasm") != std::string::npos);
        CHECK(cmd.find("-t 20") != std::string::npos);
        CHECK(cmd.find(out) != std::string::npos);
        CHECK(cmd.find(fq) != std::string::npos);
    }

    // flye
    {
        std::string cmd = build_assembler_command(AssemblerBackend::FLYE, fq, out, 16, 32);
        CHECK(cmd.find("flye") != std::string::npos);
        CHECK(cmd.find("--threads 16") != std::string::npos);
        CHECK(cmd.find("--out-dir") != std::string::npos);
    }

    // spades
    {
        std::string cmd = build_assembler_command(AssemblerBackend::SPADES, fq, out, 8, 128);
        CHECK(cmd.find("spades.py") != std::string::npos);
        CHECK(cmd.find("-m 128") != std::string::npos);
    }

    // megahit
    {
        std::string cmd = build_assembler_command(AssemblerBackend::MEGAHIT, fq, out, 4, 16);
        CHECK(cmd.find("megahit") != std::string::npos);
        CHECK(cmd.find("-t 4") != std::string::npos);
    }

    // NONE → empty string
    {
        std::string cmd = build_assembler_command(AssemblerBackend::NONE, fq, out, 1, 1);
        CHECK(cmd.empty());
    }
}

// ── 4. FASTA parsing ──────────────────────────────────────────────────────────

static void test_fasta_parsing() {
    // Multi-contig FASTA
    const std::string multi_fasta =
        ">ctg1 len=10\n"
        "ACGTGCATGC\n"
        ">ctg2 len=6\n"
        "GCGCGC\n"
        ">ctg3\n"
        "AAAA\n";
    auto contigs = parse_assembly_fasta(multi_fasta);

    CHECK(contigs.size() == 3);
    CHECK(contigs[0].name == "ctg1");
    CHECK(contigs[0].length == 10);
    // ACGTGCATGC: G,C,G,C,A,T,G,C → G,C + G,C + G,C = 6 GC → gc = 0.6
    CHECK(approx_eq(contigs[0].gc, 0.6));

    CHECK(contigs[1].name == "ctg2");
    CHECK(contigs[1].length == 6);
    // GCGCGC: all GC → gc = 1.0
    CHECK(approx_eq(contigs[1].gc, 1.0));

    CHECK(contigs[2].name == "ctg3");
    CHECK(contigs[2].length == 4);
    // AAAA: no GC → gc = 0.0
    CHECK(approx_eq(contigs[2].gc, 0.0));

    // Single contig
    auto single = parse_assembly_fasta(">only\nACGT\n");
    CHECK(single.size() == 1);
    CHECK(single[0].length == 4);

    // Empty input
    auto empty = parse_assembly_fasta("");
    CHECK(empty.empty());
}

// ── 5. N50 / N90 calculation ──────────────────────────────────────────────────

static void test_n50_n90() {
    // Hand-verified example:
    // Contigs: 100, 200, 300, 400, 500  → total = 1500
    // Sorted desc: 500, 400, 300, 200, 100
    // N50: cumsum hits 750 at 500+400=900 → N50 = 400 (first contig where cumsum ≥ 750)
    // N90: 90% of 1500 = 1350 → cumsum: 500→400→300→200 = 1400 ≥ 1350 → N90 = 200

    std::vector<ContigInfo> contigs;
    for (uint64_t len : {100ULL, 200ULL, 300ULL, 400ULL, 500ULL}) {
        ContigInfo c;
        c.name = "c";
        c.length = len;
        c.gc = 0.5;
        contigs.push_back(c);
    }
    auto stats = compute_assembly_stats(contigs, 1000, 150);

    CHECK(stats.total_contigs == 5);
    CHECK(stats.total_bases == 1500);
    CHECK(stats.n50 == 400);
    CHECK(stats.n90 == 200);
    CHECK(stats.largest_contig == 500);
    CHECK(stats.smallest_contig == 100);
}

// ── 6. GC content ────────────────────────────────────────────────────────────

static void test_gc_content() {
    // Uniform 50% GC across all contigs
    std::vector<ContigInfo> contigs;
    for (int i = 0; i < 4; ++i) {
        ContigInfo c;
        c.name = "c";
        c.length = 100;
        c.gc = 0.5;
        contigs.push_back(c);
    }
    auto stats = compute_assembly_stats(contigs, 0, 0);
    CHECK(approx_eq(stats.gc_content, 0.5, 0.01));

    // FASTA-derived GC
    auto parsed = parse_assembly_fasta(">x\nGGGGAAAA\n");  // 4 GC / 8 = 0.5
    CHECK(approx_eq(parsed[0].gc, 0.5));
}

// ── 7. Coverage estimation ────────────────────────────────────────────────────

static void test_coverage_estimation() {
    std::vector<ContigInfo> contigs;
    {
        ContigInfo c;
        c.name = "chr1";
        c.length = 1000000;  // 1 Mbp assembly
        c.gc = 0.5;
        contigs.push_back(c);
    }
    // 10000 reads × 150 bp / 1000000 bp = 1.5×
    auto stats = compute_assembly_stats(contigs, 10000, 150);
    CHECK(approx_eq(stats.estimated_coverage, 1.5, 0.01));

    // Zero contigs → no crash, coverage = 0
    std::vector<ContigInfo> empty;
    auto stats2 = compute_assembly_stats(empty, 100, 150);
    CHECK(stats2.estimated_coverage == 0.0);
}

// ── 8. Genome size from k-mer histogram ──────────────────────────────────────

static void test_kmer_genome_size() {
    // Histogram: idx = frequency, value = count of k-mers at that frequency
    // Peak at freq=30 with 100000 k-mers → genome_size ≈ total/30
    // total = 1*5000 + 2*3000 + ... we'll build a simple one-peak histogram
    std::vector<uint64_t> hist(61, 0);
    hist[1] = 50000;  // error k-mers (to be skipped as peak)
    for (size_t f = 2; f <= 60; ++f) {
        // Gaussian-like peak at 30
        double x = static_cast<double>(f) - 30.0;
        hist[f] = static_cast<uint64_t>(100000.0 * std::exp(-0.03 * x * x));
    }
    uint64_t est = estimate_genome_size_from_kmers(hist);
    // Rough sanity: should be on the order of total_kmers/peak_freq
    CHECK(est > 0);

    // Single-peak at freq=2: simple case
    std::vector<uint64_t> simple(5, 0);
    simple[2] = 1000;  // 1000 k-mers at freq=2 → total = 2000, peak=2 → est = 1000
    uint64_t est2 = estimate_genome_size_from_kmers(simple);
    CHECK(est2 == 1000ULL);

    // Empty histogram → 0
    CHECK(estimate_genome_size_from_kmers({}) == 0ULL);

    // Single-element histogram → 0
    CHECK(estimate_genome_size_from_kmers({0}) == 0ULL);
}

// ── 9. JSON output ────────────────────────────────────────────────────────────

static void test_stats_json_output() {
    namespace fs = std::filesystem;
    const std::string tmp = "/tmp/test_wgs_stats.json";

    AssemblyStats stats;
    stats.total_contigs = 42;
    stats.total_bases = 3000000;
    stats.n50 = 75000;
    stats.n90 = 25000;
    stats.largest_contig = 200000;
    stats.smallest_contig = 1000;
    stats.gc_content = 0.41;
    stats.estimated_coverage = 30.0;
    stats.reads_used = 1000000;
    stats.reads_total = 1000000;

    write_assembly_stats(stats, tmp);
    CHECK(fs::exists(tmp));

    std::string json = slurp(tmp);
    CHECK(json.find("\"total_contigs\": 42") != std::string::npos);
    CHECK(json.find("\"n50\": 75000") != std::string::npos);
    CHECK(json.find("\"n90\": 25000") != std::string::npos);

    fs::remove(tmp);
}

// ── 10. TSV report output ─────────────────────────────────────────────────────

static void test_report_tsv_output() {
    namespace fs = std::filesystem;
    const std::string tmp = "/tmp/test_wgs_report.tsv";

    AssemblyStats stats{};
    std::vector<ContigInfo> contigs;
    {
        ContigInfo c;
        c.name = "ctg1";
        c.length = 500;
        c.gc = 0.52;
        contigs.push_back(c);
    }
    {
        ContigInfo c;
        c.name = "ctg2";
        c.length = 200;
        c.gc = 0.44;
        contigs.push_back(c);
    }

    write_assembly_report(stats, contigs, tmp);
    CHECK(fs::exists(tmp));

    std::string tsv = slurp(tmp);
    CHECK(tsv.find("ctg1") != std::string::npos);
    CHECK(tsv.find("ctg2") != std::string::npos);
    CHECK(tsv.find("name\t") != std::string::npos);

    fs::remove(tmp);
}

// ── 11. Edge cases ────────────────────────────────────────────────────────────

static void test_edge_cases() {
    // Single contig: N50 = N90 = that contig's length
    {
        ContigInfo c;
        c.name = "only";
        c.length = 1000;
        c.gc = 0.5;
        auto stats = compute_assembly_stats({c}, 0, 0);
        CHECK(stats.total_contigs == 1);
        CHECK(stats.n50 == 1000);
        CHECK(stats.n90 == 1000);
        CHECK(stats.largest_contig == 1000);
        CHECK(stats.smallest_contig == 1000);
    }

    // Zero contigs
    {
        auto stats = compute_assembly_stats({}, 500, 150);
        CHECK(stats.total_contigs == 0);
        CHECK(stats.total_bases == 0);
        CHECK(stats.n50 == 0);
    }

    // Very large contig (>4 GB, exercises uint64_t)
    {
        ContigInfo c;
        c.name = "mega";
        c.length = 5000000000ULL;
        c.gc = 0.42;
        auto stats = compute_assembly_stats({c}, 1000000ULL, 10000);
        CHECK(stats.total_bases == 5000000000ULL);
        CHECK(stats.largest_contig == 5000000000ULL);
        CHECK(stats.n50 == 5000000000ULL);
    }
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    test_wgs_mode_detection();
    test_assembler_selection();
    test_command_construction();
    test_fasta_parsing();
    test_n50_n90();
    test_gc_content();
    test_coverage_estimation();
    test_kmer_genome_size();
    test_stats_json_output();
    test_report_tsv_output();
    test_edge_cases();

    std::cout << "wgs_assembly: " << g_pass << " passed, " << g_fail << " failed\n";
    return (g_fail == 0) ? 0 : 1;
}
