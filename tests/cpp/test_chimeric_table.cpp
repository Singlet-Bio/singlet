// SPDX-License-Identifier: MIT
// tests/cpp/test_chimeric_table.cpp
//
// chimeric_junctions.tsv.gz from a synthetic STAR chimeric junction file (format 1, CB/UMI as the last two columns): barcode matching
// (exact, unique one mismatch, ambiguous, unmatched), molecule collapse (reads vs distinct UMIs), the best-scoring alignment of a multimapping
// read, gene naming on the junction strand, catalogue lookup (ordered, unordered), deterministic output; transcript orientation (CT/AC lines and
// non-canonical lines of an antisense library turn around and merge with their sense twins), the canonical-motif call set, Arriba blacklist
// entries (position pairs, gene + any, ranges, unsupported keywords) and the recurrent-pair flag.

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
std::string jt(const std::string& c1, int p1, const std::string& s1, const std::string& c2, int p2, const std::string& s2, int type,
               const std::string& read, const std::string& cb, const std::string& umi) {
    return c1 + "\t" + std::to_string(p1) + "\t" + s1 + "\t" + c2 + "\t" + std::to_string(p2) + "\t" + s2 + "\t" + std::to_string(type) + "\t0\t0\t" + read +
           "\t100\t30M\t500\t40M\t1\t90\t60\t70\t70\t0\t" + cb + "\t" + umi + "\n";
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

    // ---- transcript orientation, canonical call set, blacklist, recurrent flag ----
    // genes: FWD chr4 1000-3000 +, ASL chr4 1500-2500 - (antisense lncRNA inside FWD), PART chr5 8000-9000 -, OTHER chr6 100-900 +,
    //        BLK chr7 100-900 +, RNG chr8 1000-1200 +
    put(dir + "/g2.gtf",
        "chr4\ttest\tgene\t1000\t3000\t.\t+\t.\tgene_id \"IDF\"; gene_name \"FWD\";\n"
        "chr4\ttest\tgene\t1500\t2500\t.\t-\t.\tgene_id \"IDS\"; gene_name \"ASL\";\n"
        "chr5\ttest\tgene\t8000\t9000\t.\t-\t.\tgene_id \"IDP\"; gene_name \"PART\";\n"
        "chr6\ttest\tgene\t100\t900\t.\t+\t.\tgene_id \"IDO\"; gene_name \"OTHER\";\n"
        "chr7\ttest\tgene\t100\t900\t.\t+\t.\tgene_id \"IDK\"; gene_name \"BLK\";\n"
        "chr8\ttest\tgene\t1000\t1200\t.\t+\t.\tgene_id \"IDR\"; gene_name \"RNG\";\n");
    std::string k;
    // FWD(+) chr4:2000 -> PART(-) chr5:8500: one sense read (type 1) and one antisense read of the same molecule's junction (type 2, turned)
    k += jt("chr4", 2000, "+", "chr5", 8500, "-", 1, "s1", cbA, "AAAA");
    k += jt("chr5", 8500, "+", "chr4", 2000, "-", 2, "s2", cbA, "CCCC");
    // a non-canonical junction (type 0) of an antisense read
    k += jt("chr6", 500, "-", "chr4", 2200, "-", 0, "s3", cbA, "GGGG");
    // blacklisted: exact position pair, gene + any, range
    k += jt("chr6", 300, "+", "chr4", 1200, "+", 1, "s4", cbB, "AAAA");
    k += jt("chr7", 500, "+", "chr4", 1100, "+", 1, "s5", cbB, "CCCC");
    k += jt("chr8", 1100, "+", "chr6", 700, "+", 1, "s6", cbB, "GGGG");
    // an entry whose keyword this table cannot evaluate keeps the junction
    k += jt("chr6", 400, "+", "chr4", 1300, "+", 1, "s7", cbB, "TTTT");
    put(dir + "/k.junction", k);
    put(dir + "/bl.tsv",
        "# test blacklist\n6:300\t4:1200\nBLK\tany\nchr8:900-1300\tany\n6:400\tnot_both_spliced\n");
    put(dir + "/rec.tsv", "gene_a\tgene_b\nPART\tFWD\n");
    Config oc;
    oc.junction_path = dir + "/k.junction";
    oc.gtf_path = dir + "/g2.gtf";
    oc.out_path = dir + "/o.tsv.gz";
    oc.reverse_strand = true;
    Stats so;
    auto ro = build_rows(bcs, oc, so);
    CHECK(so.noncanonical == 1, "non-canonical line left out of the call set");
    CHECK(so.reoriented == 1, "the CT/AC line is turned to transcript orientation");
    const Row* fp = nullptr;
    for (const auto& r : ro) if (r.donor_gene == "FWD" && r.acceptor_gene == "PART") fp = &r;
    CHECK(fp && fp->n_reads == 2 && fp->n_umis == 2 && fp->donor_chr == "chr4" && fp->donor_pos == 2000 && fp->donor_strand == "+" &&
          fp->acceptor_pos == 8500 && fp->acceptor_strand == "-" && fp->junction_type == 1,
          "sense and antisense reads of one junction collapse into one transcript-oriented row");
    CHECK(fp && fp->filter == "PASS", "no recurrent list: PASS");
    CHECK(ro.size() == 5, "five rows without a blacklist");

    oc.canonical_only = false;
    Stats sn;
    auto rn = build_rows(bcs, oc, sn);
    const Row* nc = nullptr;
    for (const auto& r : rn) if (r.junction_type == 0) nc = &r;
    CHECK(nc && nc->donor_chr == "chr4" && nc->donor_pos == 2200 && nc->donor_strand == "+" && nc->donor_gene == "FWD" && nc->acceptor_gene == "OTHER",
          "antisense library: a non-canonical line turns around and is named on the transcript strand (FWD, not the antisense ASL)");
    oc.reverse_strand = false;
    Stats sn2;
    auto rn2 = build_rows(bcs, oc, sn2);
    nc = nullptr;
    for (const auto& r : rn2) if (r.junction_type == 0) nc = &r;
    CHECK(nc && nc->donor_chr == "chr6" && nc->acceptor_gene == "ASL", "sense library: the non-canonical line keeps STAR's orientation");
    oc.canonical_only = true;
    oc.reverse_strand = true;

    oc.blacklist_path = dir + "/bl.tsv";
    oc.recurrent_path = dir + "/rec.tsv";
    Stats sb;
    auto rb = build_rows(bcs, oc, sb);
    CHECK(sb.blacklist_entries == 4, "four blacklist entries read");
    CHECK(sb.blacklisted_junctions == 3 && sb.blacklisted_rows == 3, "position pair, gene + any and range entries drop their junctions");
    bool kept7 = false;
    for (const auto& r : rb) if (r.donor_chr == "chr6" && r.donor_pos == 400) kept7 = true;
    CHECK(kept7, "an entry with an unsupported keyword keeps the junction");
    fp = nullptr;
    for (const auto& r : rb) if (r.donor_gene == "FWD") fp = &r;
    CHECK(fp && fp->filter == "recurrent" && sb.recurrent_rows == 1, "gene pair on the recurrent list is flagged (either order)");
    CHECK(rb.size() == 2, "two rows remain");
    Stats sw = write_table(bcs, oc);
    std::vector<std::string> ol;
    detail::for_each_line(oc.out_path, [&](const std::string& l) { ol.push_back(l); });
    CHECK(sw.ok && ol.size() == 3 && ol[0].size() > 7 && ol[0].substr(ol[0].size() - 7) == "\tfilter", "header ends with the filter column");

    std::system(("rm -rf " + dir).c_str());
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
