// test_methyl.cpp — Unit tests for singlet-pileup/methyl.h
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "singlet-pileup/methyl.h"

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

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ── 1. Context classification ─────────────────────────────────────────────────

static void test_context_classification() {
    // CpG: C followed by G
    CHECK(classify_methyl_context("ACGT", 1) == MethylContext::CPG);

    // CHG: C followed by non-G then G  (e.g. CAG)
    CHECK(classify_methyl_context("ACAG", 1) == MethylContext::CHG);
    CHECK(classify_methyl_context("ACTG", 1) == MethylContext::CHG);

    // CHH: C followed by non-G, non-G  (e.g. CAA, CTT, CAT)
    CHECK(classify_methyl_context("ACAA", 1) == MethylContext::CHH);
    CHECK(classify_methyl_context("ACTT", 1) == MethylContext::CHH);
    CHECK(classify_methyl_context("ACAT", 1) == MethylContext::CHH);

    // Edge: C at end of string — no +1 or +2 → CHH
    CHECK(classify_methyl_context("AC", 1) == MethylContext::CHH);

    // Edge: C with only one following base that is not G → CHH
    CHECK(classify_methyl_context("ACT", 1) == MethylContext::CHH);

    // CpG case-insensitive
    CHECK(classify_methyl_context("acgt", 1) == MethylContext::CPG);

    std::cout << "context_classification: OK\n";
}

// ── 2. CIGAR walking ──────────────────────────────────────────────────────────

static void test_cigar_walking() {
    // Simple match: 10M
    {
        std::vector<std::pair<char, int>> ops;
        walk_cigar("10M", [&](char op, int len) { ops.push_back({op, len}); });
        CHECK(ops.size() == 1);
        CHECK(ops[0].first == 'M');
        CHECK(ops[0].second == 10);
    }

    // Insertion + match: 3M2I5M
    {
        std::vector<std::pair<char, int>> ops;
        walk_cigar("3M2I5M", [&](char op, int len) { ops.push_back({op, len}); });
        CHECK(ops.size() == 3);
        CHECK(ops[0] == std::make_pair('M', 3));
        CHECK(ops[1] == std::make_pair('I', 2));
        CHECK(ops[2] == std::make_pair('M', 5));
    }

    // Soft clip + match + deletion
    {
        std::vector<std::pair<char, int>> ops;
        walk_cigar("5S10M3D7M", [&](char op, int len) { ops.push_back({op, len}); });
        CHECK(ops.size() == 4);
        CHECK(ops[0] == std::make_pair('S', 5));
        CHECK(ops[1] == std::make_pair('M', 10));
        CHECK(ops[2] == std::make_pair('D', 3));
        CHECK(ops[3] == std::make_pair('M', 7));
    }

    // Splice (N) and hard clip
    {
        std::vector<std::pair<char, int>> ops;
        walk_cigar("2H8M100N8M2H", [&](char op, int len) { ops.push_back({op, len}); });
        CHECK(ops.size() == 5);
        CHECK(ops[2] == std::make_pair('N', 100));
    }

    // Empty and star
    {
        std::vector<std::pair<char, int>> ops;
        walk_cigar("", [&](char op, int len) { ops.push_back({op, len}); });
        CHECK(ops.empty());
        ops.clear();
        walk_cigar("*", [&](char op, int len) { ops.push_back({op, len}); });
        CHECK(ops.empty());
    }

    std::cout << "cigar_walking: OK\n";
}

// ── 3. Single-site methylation calling ───────────────────────────────────────

static void test_methylation_calling() {
    // Forward strand: reference "CG", pos 0
    // Read "CG" → methylated CpG
    {
        auto r = call_methylation_site("CG", "CG", 0, 0, false);
        CHECK(r.has_value());
        CHECK(r->first == MethylContext::CPG);
        CHECK(r->second == true);
    }

    // Forward strand: C→T conversion = unmethylated CpG
    {
        auto r = call_methylation_site("TG", "CG", 0, 0, false);
        CHECK(r.has_value());
        CHECK(r->first == MethylContext::CPG);
        CHECK(r->second == false);
    }

    // Forward strand: CHH context, unmethylated
    {
        auto r = call_methylation_site("TAAC", "CAAC", 0, 0, false);
        CHECK(r.has_value());
        CHECK(r->first == MethylContext::CHH);
        CHECK(r->second == false);
    }

    // Reverse strand: reference G at pos 1, read has A at pos 1 → unmethylated (G→A)
    {
        auto r = call_methylation_site("GA", "GG", 1, 1, true);
        CHECK(r.has_value());
        CHECK(r->second == false);
    }

    // Reverse strand: reference G at pos 1, read "GG" → methylated
    {
        auto r = call_methylation_site("GG", "CG", 1, 1, true);
        CHECK(r.has_value());
        CHECK(r->second == true);
    }

    // Not a C/G reference base → nullopt
    {
        auto r = call_methylation_site("AT", "AT", 0, 0, false);
        CHECK(!r.has_value());
    }

    // Out of bounds read_pos → nullopt
    {
        auto r = call_methylation_site("C", "CG", 5, 0, false);
        CHECK(!r.has_value());
    }

    std::cout << "methylation_calling: OK\n";
}

// ── 4. Bisulfite detection ────────────────────────────────────────────────────

static void test_bisulfite_detection() {
    // Heavily converted: many T, few C → bisulfite
    std::string bs_read = std::string(200, 'T') + std::string(10, 'C') + std::string(90, 'A');
    CHECK(detect_bisulfite("", bs_read) == true);

    // Normal read: balanced C and T → not bisulfite
    std::string normal_read = std::string(50, 'A') + std::string(50, 'T') +
                              std::string(50, 'C') + std::string(50, 'G');
    CHECK(detect_bisulfite("", normal_read) == false);

    // Empty read → false
    CHECK(detect_bisulfite("", "") == false);

    std::cout << "bisulfite_detection: OK\n";
}

// ── 5. Mode detection ─────────────────────────────────────────────────────────

static void test_mode_detection() {
    // RRBS: has MspI sites
    CHECK(detect_methyl_mode(0.7, 0.02, true) == MethylMode::RRBS);

    // WGBS: normal methylation
    CHECK(detect_methyl_mode(0.65, 0.02, false) == MethylMode::WGBS);

    // SCBS: very low methylation values
    CHECK(detect_methyl_mode(0.05, 0.02, false) == MethylMode::SCBS);

    // NOMe-seq: high CpG and high CHH
    CHECK(detect_methyl_mode(0.8, 0.5, false) == MethylMode::NOME_SEQ);

    std::cout << "mode_detection: OK\n";
}

// ── 6. Conversion rate estimation ─────────────────────────────────────────────

static void test_conversion_rate() {
    // Perfect conversion: 0 unconverted CHH sites
    MethylStats s{};
    s.conversion_rate_ct = 0;
    s.total_ct_contexts = 1000;
    CHECK(std::abs(estimate_conversion_rate(s) - 1.0) < 1e-9);

    // 5% unconverted → 95% efficiency
    s.conversion_rate_ct = 50;
    CHECK(std::abs(estimate_conversion_rate(s) - 0.95) < 1e-9);

    // No data → assume perfect
    MethylStats empty{};
    CHECK(std::abs(estimate_conversion_rate(empty) - 1.0) < 1e-9);

    // 100% unconverted → 0% efficiency
    s.conversion_rate_ct = 1000;
    CHECK(std::abs(estimate_conversion_rate(s) - 0.0) < 1e-9);

    std::cout << "conversion_rate: OK\n";
}

// ── 7. BED output writing ─────────────────────────────────────────────────────

static void test_bed_writing() {
    std::vector<MethylSite> sites;
    {
        MethylSite s;
        s.chrom = "chr1";
        s.pos = 100;
        s.strand_plus = true;
        s.context = MethylContext::CPG;
        s.meth_reads = 8;
        s.total_reads = 10;
        sites.push_back(s);
    }
    {
        MethylSite s;
        s.chrom = "chr1";
        s.pos = 200;
        s.strand_plus = false;
        s.context = MethylContext::CHH;
        s.meth_reads = 1;
        s.total_reads = 20;
        sites.push_back(s);
    }

    const std::string path = "/tmp/test_methyl_output.bed";
    write_methylation_bed(sites, path);

    std::string content = slurp(path);
    CHECK(!content.empty());
    CHECK(content.find("chr1") != std::string::npos);
    CHECK(content.find("CpG") != std::string::npos);
    CHECK(content.find("CHH") != std::string::npos);
    // First site: 80% methylation, 8 meth, 2 unmeth
    CHECK(content.find("80") != std::string::npos);

    std::cout << "bed_writing: OK\n";
}

// ── 8. Stats merge ────────────────────────────────────────────────────────────

static void test_stats_merge() {
    // Two callers with one site each on different positions
    MethylationCaller a, b;
    a.set_chrom("chr1");
    b.set_chrom("chr1");

    // "chr1", pos 10: CpG, forward
    {
        MethylSite s;
        s.chrom = "chr1";
        s.pos = 10;
        s.strand_plus = true;
        s.context = MethylContext::CPG;
        s.meth_reads = 3;
        s.total_reads = 5;
        a.sites_.push_back(s);
        a.stats_.conversion_rate_ct = 10;
        a.stats_.total_ct_contexts = 100;
    }
    {
        MethylSite s;
        s.chrom = "chr1";
        s.pos = 10;
        s.strand_plus = true;
        s.context = MethylContext::CPG;
        s.meth_reads = 5;
        s.total_reads = 7;
        b.sites_.push_back(s);
        b.stats_.conversion_rate_ct = 5;
        b.stats_.total_ct_contexts = 50;
    }

    a.merge(b);
    CHECK(a.sites_.size() == 1);  // same site merged
    CHECK(a.sites_[0].meth_reads == 8);
    CHECK(a.sites_[0].total_reads == 12);
    CHECK(a.stats_.conversion_rate_ct == 15);
    CHECK(a.stats_.total_ct_contexts == 150);

    // Different positions → two sites
    MethylationCaller c, d;
    {
        MethylSite s;
        s.chrom = "chr1";
        s.pos = 20;
        s.strand_plus = true;
        s.context = MethylContext::CPG;
        s.meth_reads = 2;
        s.total_reads = 4;
        c.sites_.push_back(s);
    }
    {
        MethylSite s;
        s.chrom = "chr1";
        s.pos = 30;
        s.strand_plus = true;
        s.context = MethylContext::CPG;
        s.meth_reads = 1;
        s.total_reads = 2;
        d.sites_.push_back(s);
    }
    c.merge(d);
    CHECK(c.sites_.size() == 2);

    std::cout << "stats_merge: OK\n";
}

// ── 9. MethylationCaller::process_alignment ───────────────────────────────────

static void test_process_alignment() {
    // Simple 10M alignment: "CCGGAATTCC" read over "CCGGAATTCC" ref
    // Positions 0-9, ref_start = 1000.
    // Forward strand:
    //   pos 0: C in ref, C in read → methylated CPG? ref[0]='C', ref[1]='C' → CHH? ref[0+1]='C' not G → CHH
    //   Actually ref="CCGGAATTCC": pos 0, ref[1]='C' → CHH; pos1='C', ref[2]='G' → CpG methylated
    {
        MethylationCaller caller;
        caller.set_chrom("chr1");
        std::string read_seq = "CCGGAATTCC";
        std::string ref_win = "CCGGAATTCC";
        caller.process_alignment(read_seq, ref_win, 1000, "10M", false);

        // pos 1001 (ref[1]='C', ref[2]='G') → CpG; read[1]='C' → methylated
        bool found_cpg = false;
        for (const auto& s : caller.get_sites()) {
            if (s.pos == 1001 && s.context == MethylContext::CPG && s.meth_reads == 1) {
                found_cpg = true;
            }
        }
        CHECK(found_cpg);
    }

    // Soft clip: 4S6M — read offset by 4; ref starts at ref_start
    {
        MethylationCaller caller;
        caller.set_chrom("chr2");
        std::string read_seq = "AAAACCGGTT";  // first 4 are soft-clipped
        std::string ref_win = "CCGGTT";       // ref only covers the 6M bases
        caller.process_alignment(read_seq, ref_win, 500, "4S6M", false);
        // read_pos 4 corresponds to ref_local 0 (pos 500)
        // ref[0]='C', ref[1]='C' → CHH; ref[1]='C', ref[2]='G' → CpG
        bool found_site = false;
        for (const auto& s : caller.get_sites()) {
            if (s.chrom == "chr2") found_site = true;
        }
        CHECK(found_site);
    }

    // No cytosines: all A and T
    {
        MethylationCaller caller;
        caller.set_chrom("chr3");
        caller.process_alignment("AAATATAT", "AAATATAT", 0, "8M", false);
        CHECK(caller.get_sites().empty());
    }

    std::cout << "process_alignment: OK\n";
}

// ── 10. get_stats ─────────────────────────────────────────────────────────────

static void test_get_stats() {
    MethylationCaller caller;
    caller.set_chrom("chr1");
    // Manually add sites
    {
        MethylSite s;
        s.chrom = "chr1";
        s.pos = 100;
        s.strand_plus = true;
        s.context = MethylContext::CPG;
        s.meth_reads = 4;
        s.total_reads = 5;
        caller.sites_.push_back(s);
    }
    {
        MethylSite s;
        s.chrom = "chr1";
        s.pos = 200;
        s.strand_plus = true;
        s.context = MethylContext::CHG;
        s.meth_reads = 1;
        s.total_reads = 10;
        caller.sites_.push_back(s);
    }
    {
        MethylSite s;
        s.chrom = "chr1";
        s.pos = 300;
        s.strand_plus = true;
        s.context = MethylContext::CHH;
        s.meth_reads = 0;
        s.total_reads = 8;
        caller.sites_.push_back(s);
    }
    caller.stats_.conversion_rate_ct = 8;
    caller.stats_.total_ct_contexts = 18;

    MethylStats st = caller.get_stats();
    CHECK(st.total_sites == 3);
    CHECK(st.cpg_sites == 1);
    CHECK(st.chg_sites == 1);
    CHECK(st.chh_sites == 1);
    CHECK(std::abs(st.global_cpg_methylation - 0.8) < 1e-9);
    CHECK(std::abs(st.global_chg_methylation - 0.1) < 1e-9);
    CHECK(std::abs(st.global_chh_methylation - 0.0) < 1e-9);

    std::cout << "get_stats: OK\n";
}

// ── 11. write_methyl_stats JSON ───────────────────────────────────────────────

static void test_write_methyl_stats() {
    MethylStats s{};
    s.total_sites = 500;
    s.cpg_sites = 300;
    s.chg_sites = 100;
    s.chh_sites = 100;
    s.global_cpg_methylation = 0.75;
    s.global_chg_methylation = 0.05;
    s.global_chh_methylation = 0.02;
    s.conversion_rate_ct = 20;
    s.total_ct_contexts = 1000;

    const std::string path = "/tmp/test_methyl_stats.json";
    write_methyl_stats(s, path);
    std::string content = slurp(path);
    CHECK(!content.empty());
    CHECK(content.find("total_sites") != std::string::npos);
    CHECK(content.find("500") != std::string::npos);
    CHECK(content.find("cpg_sites") != std::string::npos);

    std::cout << "write_methyl_stats: OK\n";
}

// ── 12. Edge cases ────────────────────────────────────────────────────────────

static void test_edge_cases() {
    // classify_methyl_context: negative pos → CHH
    CHECK(classify_methyl_context("ACGT", -1) == MethylContext::CHH);

    // call_methylation_site: ref_pos out of range → nullopt
    auto r = call_methylation_site("C", "CG", 0, 10, false);
    CHECK(!r.has_value());

    // detect_bisulfite: only G and N → false (no C or T)
    CHECK(detect_bisulfite("", std::string(100, 'G') + std::string(50, 'N')) == false);

    // walk_cigar: all ops including = and X
    {
        std::vector<std::pair<char, int>> ops;
        walk_cigar("3=2X1P", [&](char op, int len) { ops.push_back({op, len}); });
        CHECK(ops.size() == 3);
        CHECK(ops[0].first == '=');
        CHECK(ops[1].first == 'X');
        CHECK(ops[2].first == 'P');
    }

    // MethylationCaller with empty sites → get_stats returns 0 sites
    {
        MethylationCaller empty_caller;
        MethylStats st = empty_caller.get_stats();
        CHECK(st.total_sites == 0);
    }

    std::cout << "edge_cases: OK\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_context_classification();
    test_cigar_walking();
    test_methylation_calling();
    test_bisulfite_detection();
    test_mode_detection();
    test_conversion_rate();
    test_bed_writing();
    test_stats_merge();
    test_process_alignment();
    test_get_stats();
    test_write_methyl_stats();
    test_edge_cases();

    std::cout << "\nResult: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
