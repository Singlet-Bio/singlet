// SPDX-License-Identifier: MIT
// tests/cpp/test_chimeric_table.cpp
//
// chimeric_junctions.tsv.gz from a synthetic STAR chimeric junction file (format 1, CB/UMI as the last two columns): barcode matching
// (exact, unique one mismatch, ambiguous, unmatched), molecule collapse (reads vs distinct UMIs), the best-scoring alignment of a multimapping
// read, gene naming on the junction strand, catalogue lookup (ordered, unordered), deterministic output.

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "singlet/pileup/chimeric_table.h"
#include "test_harness.h"

using namespace singlet::chimeric;

namespace {
void put(const std::string& path, const std::string& text) { std::ofstream(path) << text; }

// columns: donor chr, pos, strand, acceptor chr, pos, strand, type, repeatL, repeatR, read, startA, cigarA, startB, cigarB, nAln, maxScore, nonChim,
// score, best, PEmerged, cb, umi
std::string jl(const std::string& c1, int p1, const std::string& s1, const std::string& c2, int p2, const std::string& s2, const std::string& read,
               int n_aln, int score, int best, const std::string& cb, const std::string& umi) {
    return c1 + "\t" + std::to_string(p1) + "\t" + s1 + "\t" + c2 + "\t" + std::to_string(p2) + "\t" + s2 + "\t1\t0\t0\t" + read +
           "\t100\t30M\t500\t40M\t" + std::to_string(n_aln) + "\t90\t60\t" + std::to_string(score) + "\t" + std::to_string(best) + "\t0\t" + cb + "\t" + umi + "\n";
}
}  // namespace

int main() {
    const std::string dir = "/tmp/test_chimeric_table_" + std::to_string(getpid());
    std::system(("mkdir -p " + dir).c_str());
    const std::string cbA = "AAACCCAAGAAACACT", cbB = "TTTGGGAAGAAACACT", cbC = "CCCCCCCCCCCCCCCC";
    // genes: GENEA chr1 1000-2000 +, GENEB chr2 5000-6000 +, GENEC chr2 5500-5600 - (small, opposite strand), GENED chr3 100-900 +
    put(dir + "/g.gtf",
        "chr1\ttest\tgene\t1000\t2000\t.\t+\t.\tgene_id \"IDA\"; gene_name \"GENEA\";\n"
        "chr2\ttest\tgene\t5000\t6000\t.\t+\t.\tgene_id \"IDB\"; gene_name \"GENEB\";\n"
        "chr2\ttest\tgene\t5500\t5600\t.\t-\t.\tgene_id \"IDC\"; gene_name \"GENEC\";\n"
        "chr3\ttest\tgene\t100\t900\t.\t+\t.\tgene_id \"IDD\"; gene_name \"GENED\";\n");
    put(dir + "/cat.tsv",
        "#built by test\ngene_a\tgene_b\tsources\tordered\nGENEA\tGENEB\tMitelman,ChimerKB\t1\nGENED\tGENEA\tMitelman\t0\nGENEB\tGENED\tCTAT_X\t1\n");
    std::string j = "chr_donorA\tbrkpt_donorA\tstrand_donorA\tchr_acceptorB\tbrkpt_acceptorB\tstrand_acceptorB\tjunction_type\trepeat_left_lenA\t"
                    "repeat_right_lenB\tread_name\tstart_alnA\tcigar_alnA\tstart_alnB\tcigar_alnB\tnum_chim_aln\tmax_poss_aln_score\tnon_chim_aln_score\t"
                    "this_chim_aln_score\tbestall_chim_aln_score\tPEmerged_bool\n";
    j += jl("chr1", 1500, "+", "chr2", 5200, "+", "r1", 1, 70, 70, cbA, "AAAA");                  // GENEA--GENEB, cell A, umi AAAA
    j += jl("chr1", 1500, "+", "chr2", 5200, "+", "r2", 1, 70, 70, cbA, "AAAA");                  // same molecule (same UMI)
    j += jl("chr1", 1500, "+", "chr2", 5200, "+", "r3", 1, 68, 68, cbA, "CCCC");                  // second molecule
    j += jl("chr1", 1500, "+", "chr2", 5200, "+", "r4", 1, 70, 70, "AAACCCAAGAAACACA", "GGGG");  // two one-mismatch neighbours: ambiguous
    j += jl("chr3", 500, "+", "chr1", 1500, "+", "r5", 1, 66, 66, cbB, "TTTT");                    // GENED--GENEA, cell B
    j += jl("chr2", 5550, "-", "chr3", 500, "+", "r6", 2, 60, 64, cbB, "ACAC");                   // worse alignment of a multimapping read: dropped
    j += jl("chr2", 5550, "-", "chr3", 500, "+", "r6", 2, 64, 64, cbB, "ACAC");                   // best: donor GENEC (minus strand, small)
    j += jl("chr2", 5300, "+", "chr3", 400, "+", "r7", 1, 62, 62, cbC, "GGAA");                    // barcode in none of the cells
    j += jl("chr2", 5300, "+", "chr3", 400, "+", "r8", 1, 62, 62, "AAACCCAAGAAACACN", "GGAA");     // N matches two neighbours: ambiguous
    j += "chr2\t1\t+\tchr3\t2\t+\t1\t0\t0\tr9\n";                                                    // short line: no barcode
    j += "# 2.7.11b   STAR --chimSegmentMin 15\n# Nreads 12 NreadsUnique 10 NreadsMulti 2\n";
    put(dir + "/j.junction", j);

    // sample barcodes: cell A, cell B, and a neighbour of cell A that makes the one-mismatch reads ambiguous
    std::vector<std::string> bcs = {cbA, cbB, "AAACCCAAGAAACACG"};
    Config cfg;
    cfg.junction_path = dir + "/j.junction";
    cfg.gtf_path = dir + "/g.gtf";
    cfg.catalog_path = dir + "/cat.tsv";
    cfg.out_path = dir + "/out.tsv.gz";
    Stats st;
    auto rows = build_rows(bcs, cfg, st);

    CHECK(st.lines == 10, "all junction lines counted, comments and header skipped");
    CHECK(st.no_barcode == 1, "short line has no barcode");
    CHECK(st.unmatched == 1, "barcode in none of the sample's barcodes is dropped");
    CHECK(st.ambiguous == 2, "two barcodes with two one-mismatch neighbours are ambiguous and dropped");
    CHECK(rows.size() == 3, "three (cell, junction) rows");
    if (rows.size() == 3) {
        const Row& r0 = rows[0];  // barcodes sort: AAACCC... (cell A) first
        CHECK(r0.barcode == cbA && r0.donor_gene == "GENEA" && r0.acceptor_gene == "GENEB", "cell A: GENEA--GENEB");
        CHECK(r0.n_reads == 3 && r0.n_umis == 2, "molecule collapse: 3 reads, 2 distinct UMIs");
        CHECK(r0.sources == "Mitelman,ChimerKB" && r0.orientation == "ordered", "ordered catalogue hit");
        CHECK(r0.donor_pos == 1500 && r0.acceptor_pos == 5200 && r0.donor_chr == "chr1" && r0.acceptor_chr == "chr2", "junction coordinates");
        const Row* d = nullptr;
        const Row* c = nullptr;
        for (const auto& r : rows) {
            if (r.donor_gene == "GENED") d = &r;
            if (r.donor_gene == "GENEC") c = &r;
        }
        CHECK(d && d->acceptor_gene == "GENEA" && d->sources == "Mitelman" && d->orientation == "unordered", "unordered catalogue entry");
        CHECK(c && c->barcode == cbB && c->n_reads == 1 && c->multimap && c->n_chim_aln == 2, "multimapping read: only the best alignment, flagged");
        CHECK(c && c->acceptor_gene == "GENED" && c->sources == ".", "no catalogue entry gives '.'");
    }
    CHECK(st.annotated == 2, "two rows are in the catalogue");

    Stats st2 = write_table(bcs, cfg);
    CHECK(st2.ok && st2.rows == 3 && st2.cells == 2, "table written");
    std::vector<std::string> lines;
    detail::for_each_line(cfg.out_path, [&](const std::string& l) { lines.push_back(l); });
    CHECK(lines.size() == 4 && lines[0].rfind("barcode\tdonor_chr", 0) == 0, "header plus three rows");
    cfg.out_path = dir + "/out2.tsv.gz";
    write_table(bcs, cfg);
    std::ifstream a(dir + "/out.tsv.gz", std::ios::binary), b(dir + "/out2.tsv.gz", std::ios::binary);
    std::string sa((std::istreambuf_iterator<char>(a)), std::istreambuf_iterator<char>());
    std::string sb((std::istreambuf_iterator<char>(b)), std::istreambuf_iterator<char>());
    CHECK(sa == sb && !sa.empty(), "output bytes are deterministic");

    Config none = cfg;
    none.gtf_path.clear();
    none.catalog_path.clear();
    Stats st3;
    auto rows3 = build_rows(bcs, none, st3);
    CHECK(rows3.size() == 3 && rows3[0].donor_gene.empty() && rows3[0].sources == ".", "without annotation files the genes and sources are empty");

    std::system(("rm -rf " + dir).c_str());
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
