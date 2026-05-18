// SPDX-License-Identifier: MIT
// G-WEBSUMMARY: unit tests for web_summary.h
// Verifies HTML structure, required sections, SVG elements, and well-formedness.
// Run via ctest or standalone:
//   g++ -std=c++17 -O2 -I../include -o /tmp/test_web_summary test/test_web_summary.cpp && /tmp/test_web_summary

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/web_summary.h"

using namespace singlet;

// ── Helpers ──────────────────────────────────────────────────────────────────

static int s_passed = 0;
static int s_failed = 0;

static void check(const char* name, bool cond, const char* msg = "") {
    if (cond) {
        ++s_passed;
        std::cerr << "PASS: " << name << "\n";
    } else {
        ++s_failed;
        std::cerr << "FAIL: " << name;
        if (msg[0]) std::cerr << " — " << msg;
        std::cerr << "\n";
    }
}

static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// Build a representative WebSummaryData with synthetic but realistic values
static WebSummaryData make_test_data() {
    WebSummaryData d;
    d.sample_name       = "TestSample_GSM123456";
    d.pipeline_version  = "0.9.0-test";
    d.date              = "2026-04-13";
    d.organism          = "Homo sapiens";
    d.protocol          = "10xv3";

    d.estimated_cells          = 4500;
    d.mean_reads_per_cell      = 28000.0;
    d.median_genes_per_cell    = 2100.5;
    d.total_genes_detected     = 19432;
    d.median_umis_per_cell     = 6500.0;
    d.sequencing_saturation    = 0.713;
    d.mapping_rate_genome      = 0.924;
    d.mapping_rate_transcriptome = 0.783;
    d.mapping_rate_exonic      = 0.612;
    d.mapping_rate_intronic    = 0.171;
    d.mapping_rate_intergenic  = 0.052;

    // Barcode rank data: exponential-ish decay (5000 barcodes, sorted desc)
    d.barcode_umi_counts.resize(5000);
    for (size_t i = 0; i < d.barcode_umi_counts.size(); ++i) {
        double rank  = static_cast<double>(i + 1);
        double total = static_cast<double>(d.barcode_umi_counts.size());
        d.barcode_umi_counts[i] = static_cast<uint64_t>(
            20000.0 * std::pow(1.0 - rank / (total * 1.2), 2.5) + 1);
    }
    // Ensure sorted descending
    std::sort(d.barcode_umi_counts.begin(), d.barcode_umi_counts.end(),
              [](uint64_t a, uint64_t b) { return a > b; });
    d.cell_calling_threshold = d.barcode_umi_counts[4499];  // UMI at rank 4500

    // Genes per cell: roughly normal around 2100
    d.genes_per_cell.resize(4500);
    for (size_t i = 0; i < d.genes_per_cell.size(); ++i) {
        // Deterministic pseudo-normal using index
        int g = 2100 + static_cast<int>((i % 800) - 400);
        if (g < 200) g = 200;
        d.genes_per_cell[i] = static_cast<uint32_t>(g);
    }

    return d;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

int main() {
    // ── 1. generate_web_summary returns non-empty string ─────────────────────
    WebSummaryData d = make_test_data();
    std::string html = generate_web_summary(d);
    check("returns_non_empty", !html.empty());

    // ── 2. Well-formed HTML: has <html>, <head>, <body> ──────────────────────
    check("has_html_open_tag",  contains(html, "<html"));
    check("has_head_tag",       contains(html, "<head>") || contains(html, "<head\n"));
    check("has_body_tag",       contains(html, "<body>") || contains(html, "<body\n"));
    check("has_html_close_tag", contains(html, "</html>"));
    check("has_body_close_tag", contains(html, "</body>"));

    // ── 3. DOCTYPE ────────────────────────────────────────────────────────────
    check("has_doctype", contains(html, "<!DOCTYPE html>") || contains(html, "<!doctype html>"));

    // ── 4. Header section: sample name, organism, protocol, version ──────────
    check("has_sample_name",  contains(html, "TestSample_GSM123456"));
    check("has_organism",     contains(html, "Homo sapiens"));
    check("has_protocol",     contains(html, "10xv3"));
    check("has_version",      contains(html, "0.9.0-test"));
    check("has_date",         contains(html, "2026-04-13"));

    // ── 5. Key metrics section: all 8 metrics present ────────────────────────
    check("has_estimated_cells",     contains(html, "4,500"));
    check("has_total_genes",         contains(html, "19,432"));
    check("has_map_genome_pct",      contains(html, "92.4%"));
    check("has_map_txome_pct",       contains(html, "78.3%"));
    check("has_saturation_pct",      contains(html, "71.3%"));

    // ── 6. Mapping summary table rows ─────────────────────────────────────────
    check("has_exonic_pct",      contains(html, "61.2%"));
    check("has_intronic_pct",    contains(html, "17.1%"));
    check("has_intergenic_pct",  contains(html, "5.2%"));

    // ── 7. SVG elements present ───────────────────────────────────────────────
    check("has_svg_open",       contains(html, "<svg"));
    check("has_svg_close",      contains(html, "</svg>"));
    check("has_polyline_or_rect", contains(html, "<polyline") || contains(html, "<rect"));

    // ── 8. Knee plot section ──────────────────────────────────────────────────
    check("has_barcode_rank_label", contains(html, "Barcode Rank") || contains(html, "Barcode rank"));
    check("has_umi_axis_label",     contains(html, "UMI count") || contains(html, "UMI Count"));

    // ── 9. Gene histogram section ─────────────────────────────────────────────
    check("has_gene_distribution_label",
          contains(html, "Gene Distribution") || contains(html, "Genes per cell") ||
          contains(html, "Genes per Cell"));

    // ── 10. No external URLs (CSS, JS, fonts from CDN) ────────────────────────
    check("no_external_css",  !contains(html, "rel=\"stylesheet\"")
                           && !contains(html, "href=\"http"));
    check("no_script_src",    !contains(html, "<script src"));
    check("no_cdn_links",     !contains(html, "cdn.") && !contains(html, "googleapis.com"));

    // ── 11. Warnings present when added ──────────────────────────────────────
    {
        WebSummaryData dw = d;
        dw.warnings.push_back("Low mapping rate detected: 35.2%");
        dw.warnings.push_back("Low complexity: median genes per cell < 500");
        std::string html_w = generate_web_summary(dw);
        check("warnings_text_present",
              contains(html_w, "Low mapping rate detected"));
        check("multi_warning_present",
              contains(html_w, "Low complexity"));
    }

    // ── 12. No warnings → positive status message ─────────────────────────────
    {
        WebSummaryData dn = d;
        dn.warnings.clear();
        std::string html_n = generate_web_summary(dn);
        check("no_warnings_status", contains(html_n, "No warnings") ||
                                    contains(html_n, "no warnings") ||
                                    contains(html_n, "within expected"));
    }

    // ── 13. Empty barcode data → placeholder, not crash ───────────────────────
    {
        WebSummaryData de = d;
        de.barcode_umi_counts.clear();
        de.genes_per_cell.clear();
        std::string html_e = generate_web_summary(de);
        check("empty_data_no_crash",   !html_e.empty());
        check("empty_data_is_html",    contains(html_e, "</html>"));
        check("empty_data_has_nodata", contains(html_e, "No data") ||
                                       contains(html_e, "no data"));
    }

    // ── 14. write_web_summary writes a file ───────────────────────────────────
    {
        const std::string tmp_path = "/tmp/singlet_test_web_summary.html";
        write_web_summary(tmp_path, d);
        std::ifstream f(tmp_path);
        bool opened = f.is_open();
        check("write_creates_file", opened);
        if (opened) {
            std::ostringstream ss;
            ss << f.rdbuf();
            std::string file_content = ss.str();
            check("write_file_has_html", contains(file_content, "</html>"));
            check("write_file_nonempty", file_content.size() > 2000);
            // Cleanup
            std::remove(tmp_path.c_str());
        }
    }

    // ── 15. HTML-escape of special characters in sample name ─────────────────
    {
        WebSummaryData dx = d;
        dx.sample_name = "Sample<>&\"'Test";
        std::string html_x = generate_web_summary(dx);
        // The raw unescaped characters must not appear in a context that would
        // break HTML parsing (they should be escaped as entities)
        check("html_escape_lt",  !contains(html_x, "Sample<>&\"'Test"));
        check("html_escape_amp", contains(html_x, "&amp;") || contains(html_x, "&lt;"));
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    std::cerr << "\n=== " << s_passed << " passed, " << s_failed << " failed ===\n";
    return (s_failed == 0) ? 0 : 1;
}
