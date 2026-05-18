// SPDX-License-Identifier: MIT
// test_benchmark_suite.cpp — Unit tests for singlet/pileup/benchmark_suite.h
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "singlet/pileup/benchmark_suite.h"

using namespace singlet::pileup::benchmark;
namespace fs = std::filesystem;

#define SINGLET_TEST_HARNESS_TERSE
#include "test_harness.h"  // CHECK(cond) + g_pass / g_fail

// ── helpers ───────────────────────────────────────────────────────────────────

static BenchmarkResult make_result(const std::string& tool, Modality mod,
                                   const std::string& dataset,
                                   double wall, double concordance = -1.0) {
    BenchmarkResult r;
    r.tool_name = tool;
    r.tool_version = "1.0";
    r.modality = mod;
    r.dataset_id = dataset;
    r.total_reads = 40'000'000;
    r.threads = 16;
    r.wall_seconds = wall;
    r.concordance = concordance;
    r.hostname = "c001";
    r.date = "2026-04-13";
    return r;
}

// ── 1. Modality name round-trip ───────────────────────────────────────────────

static void test_modality_names() {
    CHECK(modality_name(Modality::SCRNA) == "scRNA");
    CHECK(modality_name(Modality::ATAC) == "ATAC");
    CHECK(modality_name(Modality::CITE_SEQ) == "CITE-seq");
    CHECK(modality_name(Modality::SMART_SEQ2) == "Smart-seq2");
    CHECK(modality_name(Modality::CHIP_SEQ) == "ChIP-seq");
}

static void test_modality_parse_round_trip() {
    for (auto m : {Modality::SCRNA, Modality::SNRNA, Modality::ATAC, Modality::CITE_SEQ,
                   Modality::VISIUM, Modality::SMART_SEQ2, Modality::BULK_RNA, Modality::VDJ,
                   Modality::CRISPR, Modality::SPATIAL, Modality::HIC, Modality::METHYL,
                   Modality::LONG_READ, Modality::CHIP_SEQ}) {
        CHECK(parse_modality(modality_name(m)) == m);
    }
}

static void test_parse_modality_unknown() {
    bool threw = false;
    try {
        parse_modality("not_a_modality");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

// ── 2. Registry add + filter ──────────────────────────────────────────────────

static void test_add_and_filter_by_modality() {
    BenchmarkRegistry reg;
    reg.add(make_result("singlet", Modality::SCRNA, "SRR1", 10.0));
    reg.add(make_result("starsolo", Modality::SCRNA, "SRR1", 30.0));
    reg.add(make_result("singlet", Modality::ATAC, "SRR2", 20.0));

    auto rna_results = reg.by_modality(Modality::SCRNA);
    CHECK(rna_results.size() == 2);

    auto atac_results = reg.by_modality(Modality::ATAC);
    CHECK(atac_results.size() == 1);
    CHECK(atac_results[0].dataset_id == "SRR2");
}

static void test_filter_by_tool() {
    BenchmarkRegistry reg;
    reg.add(make_result("singlet", Modality::SCRNA, "SRR1", 10.0));
    reg.add(make_result("starsolo", Modality::SCRNA, "SRR1", 30.0));
    reg.add(make_result("singlet", Modality::ATAC, "SRR2", 20.0));

    auto singlet_results = reg.by_tool("singlet");
    CHECK(singlet_results.size() == 2);

    auto star_results = reg.by_tool("starsolo");
    CHECK(star_results.size() == 1);
}

static void test_filter_by_dataset() {
    BenchmarkRegistry reg;
    reg.add(make_result("singlet", Modality::SCRNA, "SRR1", 10.0));
    reg.add(make_result("starsolo", Modality::SCRNA, "SRR1", 30.0));
    reg.add(make_result("singlet", Modality::ATAC, "SRR2", 20.0));

    auto srr1 = reg.by_dataset("SRR1");
    CHECK(srr1.size() == 2);

    auto srr2 = reg.by_dataset("SRR2");
    CHECK(srr2.size() == 1);
}

// ── 3. Speedup calculation ────────────────────────────────────────────────────

static void test_speedup_calculation() {
    BenchmarkRegistry reg;
    reg.add(make_result("singlet", Modality::SCRNA, "SRR1", 10.0));
    reg.add(make_result("cellranger", Modality::SCRNA, "SRR1", 30.0));

    double sp = reg.speedup("cellranger", "SRR1");
    // 30 / 10 = 3.0
    CHECK(sp > 2.99 && sp < 3.01);
}

static void test_speedup_missing_data() {
    BenchmarkRegistry reg;
    reg.add(make_result("singlet", Modality::SCRNA, "SRR1", 10.0));

    // Competitor not present
    double sp = reg.speedup("cellranger", "SRR1");
    CHECK(sp < 0.0);

    // Dataset not present
    sp = reg.speedup("cellranger", "SRR_MISSING");
    CHECK(sp < 0.0);
}

// ── 4. Pareto frontier ────────────────────────────────────────────────────────

static void test_pareto_frontier_true() {
    BenchmarkRegistry reg;
    // singlet: faster + higher concordance → on Pareto frontier
    reg.add(make_result("singlet", Modality::SCRNA, "SRR1", 10.0, 0.9998));
    reg.add(make_result("starsolo", Modality::SCRNA, "SRR1", 30.0, 0.9990));
    CHECK(reg.at_pareto_frontier("starsolo", "SRR1") == true);
}

static void test_pareto_frontier_false_lower_accuracy() {
    BenchmarkRegistry reg;
    // singlet: faster but LOWER concordance → NOT on Pareto frontier
    reg.add(make_result("singlet", Modality::SCRNA, "SRR1", 10.0, 0.9950));
    reg.add(make_result("starsolo", Modality::SCRNA, "SRR1", 30.0, 0.9990));
    CHECK(reg.at_pareto_frontier("starsolo", "SRR1") == false);
}

static void test_pareto_frontier_false_slower() {
    BenchmarkRegistry reg;
    // singlet: slower → NOT on Pareto frontier
    reg.add(make_result("singlet", Modality::SCRNA, "SRR1", 40.0, 0.9998));
    reg.add(make_result("starsolo", Modality::SCRNA, "SRR1", 30.0, 0.9990));
    CHECK(reg.at_pareto_frontier("starsolo", "SRR1") == false);
}

static void test_pareto_frontier_no_concordance() {
    BenchmarkRegistry reg;
    // No concordance data: only speed matters
    reg.add(make_result("singlet", Modality::SCRNA, "SRR1", 10.0));
    reg.add(make_result("starsolo", Modality::SCRNA, "SRR1", 30.0));
    CHECK(reg.at_pareto_frontier("starsolo", "SRR1") == true);
}

// ── 5. TSV round-trip ─────────────────────────────────────────────────────────

static void test_tsv_round_trip() {
    BenchmarkRegistry reg;
    reg.add(make_result("singlet", Modality::SCRNA, "SRR32855204", 82.3, 0.9998));
    reg.add(make_result("starsolo", Modality::SCRNA, "SRR32855204", 200.0, 0.9995));
    reg.add(make_result("singlet", Modality::ATAC, "SRR_ATAC", 55.0, -1.0));

    std::string tmp = "/tmp/bench_test_" + std::to_string(std::rand()) + ".tsv";
    write_benchmark_tsv(reg, tmp);
    BenchmarkRegistry reg2 = read_benchmark_tsv(tmp);
    fs::remove(tmp);

    CHECK(reg2.results.size() == 3);
    CHECK(reg2.results[0].tool_name == "singlet");
    CHECK(reg2.results[0].modality == Modality::SCRNA);
    CHECK(reg2.results[0].concordance > 0.9997 && reg2.results[0].concordance < 0.9999);
    CHECK(reg2.results[1].wall_seconds > 199.9 && reg2.results[1].wall_seconds < 200.1);
    CHECK(reg2.results[2].modality == Modality::ATAC);
}

// ── 6. Markdown formatting ────────────────────────────────────────────────────

static void test_markdown_formatting() {
    BenchmarkRegistry reg;
    reg.add(make_result("singlet", Modality::SCRNA, "SRR1", 10.0, 0.9998));
    reg.add(make_result("starsolo", Modality::SCRNA, "SRR1", 30.0, 0.9990));

    std::string md = format_benchmark_markdown(reg);
    CHECK(md.find("| Tool |") != std::string::npos);
    CHECK(md.find("singlet") != std::string::npos);
    CHECK(md.find("starsolo") != std::string::npos);
    CHECK(md.find("scRNA") != std::string::npos);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_modality_names();
    test_modality_parse_round_trip();
    test_parse_modality_unknown();
    test_add_and_filter_by_modality();
    test_filter_by_tool();
    test_filter_by_dataset();
    test_speedup_calculation();
    test_speedup_missing_data();
    test_pareto_frontier_true();
    test_pareto_frontier_false_lower_accuracy();
    test_pareto_frontier_false_slower();
    test_pareto_frontier_no_concordance();
    test_tsv_round_trip();
    test_markdown_formatting();

    std::cout << "benchmark_suite: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
