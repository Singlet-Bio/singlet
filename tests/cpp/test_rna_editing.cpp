// SPDX-License-Identifier: MIT
// test_rna_editing: D-101 RNA editing at known sites (rna_editing.h, pileup_engine.h load_edit_panel).
//  1. split_tail / select_rows on a random matrix: the head equals the matrix built without the tail rows, byte for byte.
//  2. Engine end to end on a synthetic BAM: a run with an editing panel, after split_tail, gives SNP matrices identical to a run
//     without it, and the editing rows hold the expected edited / total counts (A>G on +, T>C on -, other bases and low MAPQ out).
//  3. summarize and the writers on the engine's counts.
#include "singlet/pileup/pileup_engine.h"
#include "singlet/pileup/rna_editing.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
int n_pass = 0, n_fail = 0;
void CHECK(bool ok, const std::string& what) {
    (ok ? n_pass : n_fail)++;
    std::cout << (ok ? "  PASS " : "  FAIL ") << what << "\n";
}
using Acc = singlet::SparseAccumulator<uint16_t>;
using CSC = Acc::CSCMatrix;

bool same(const CSC& a, const CSC& b) {
    return a.nrows == b.nrows && a.ncols == b.ncols && a.indptr == b.indptr && a.indices == b.indices && a.data == b.data;
}

void test_split_random() {
    std::cout << "=== split_tail / select_rows ===\n";
    std::mt19937 rng(7);
    const uint32_t nr = 50, nc = 30, start = 37;
    Acc full, head, tail;
    std::vector<std::string> bcs(nc, "x");
    full.set_barcodes(bcs); head.set_barcodes(bcs); tail.set_barcodes(bcs);
    full.set_n_features(nr); head.set_n_features(start); tail.set_n_features(nr - start);
    for (int i = 0; i < 600; ++i) {
        const uint32_t r = rng() % nr, c = rng() % nc; const uint16_t v = 1 + rng() % 5;
        full.increment(r, c, v);
        if (r < start) head.increment(r, c, v); else tail.increment(r - start, c, v);
    }
    CSC m = full.to_csc();
    CSC t = singlet::editing::split_tail(m, start);
    CHECK(same(m, head.to_csc()), "head rows equal the matrix built without the tail");
    CHECK(same(t, tail.to_csc()), "tail rows equal the tail matrix, renumbered from 0");
    CSC m2 = full.to_csc();
    CSC none = singlet::editing::split_tail(m2, nr);
    CHECK(same(m2, full.to_csc()) && none.nrows == 0 && none.indices.empty(), "start = nrows leaves the matrix unchanged");
    std::vector<uint32_t> keep{1, 4, 9};
    CSC s = singlet::editing::select_rows(t, keep);
    bool ok = s.nrows == 3;
    for (uint32_t c = 0; c < s.ncols && ok; ++c)
        for (int32_t k = s.indptr[c]; k < s.indptr[c + 1]; ++k) {
            const uint32_t orig = keep[s.indices[k]];
            bool found = false;
            for (int32_t j = t.indptr[c]; j < t.indptr[c + 1]; ++j) if (uint32_t(t.indices[j]) == orig && t.data[j] == s.data[k]) found = true;
            ok = ok && found;
        }
    size_t expect = 0;
    for (int32_t k = 0; k < (int32_t)t.indices.size(); ++k) for (auto r : keep) if (uint32_t(t.indices[k]) == r) ++expect;
    CHECK(ok && s.indices.size() == expect, "select_rows keeps exactly the listed rows");
}

// chr1 reference: positions 101-110 (1-based) read AAAAGGAAAA in the + reads below.
bool make_bam(const std::string& bam, const std::string& dir) {
    std::ofstream f(dir + "/t.sam");
    f << "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr1\tLN:1000\n";
    // barcode 1: two reads with A at 103 (ref), one with G (edited); G>T SNP at 106: G, T, T
    f << "r1\t0\tchr1\t101\t30\t10M\t*\t0\t0\tAAAAGGAAAA\t*\tCB:Z:AAACCTGAGAAACGAG-1\n";
    f << "r2\t0\tchr1\t101\t30\t10M\t*\t0\t0\tAAAAGTAAAA\t*\tCB:Z:AAACCTGAGAAACGAG-1\n";
    f << "r3\t0\tchr1\t101\t30\t10M\t*\t0\t0\tAAGAGTAAAA\t*\tCB:Z:AAACCTGAGAAACGAG-1\n";
    // barcode 2: G at 103 (edited), C at 103 (neither: out of dp), low MAPQ G (filtered)
    f << "r4\t0\tchr1\t101\t30\t10M\t*\t0\t0\tAAGAGGAAAA\t*\tCB:Z:AAACCTGCAGGTTCAC-1\n";
    f << "r5\t0\tchr1\t101\t30\t10M\t*\t0\t0\tAACAGGAAAA\t*\tCB:Z:AAACCTGCAGGTTCAC-1\n";
    f << "r6\t0\tchr1\t101\t5\t10M\t*\t0\t0\tAAGAGGAAAA\t*\tCB:Z:AAACCTGCAGGTTCAC-1\n";
    // a - strand editing site at chr1:301 (genome T, edited C): barcode 2 reads T, C, C
    f << "r7\t16\tchr1\t301\t30\t5M\t*\t0\t0\tTGGGG\t*\tCB:Z:AAACCTGCAGGTTCAC-1\n";
    f << "r8\t16\tchr1\t301\t30\t5M\t*\t0\t0\tCGGGG\t*\tCB:Z:AAACCTGCAGGTTCAC-1\n";
    f << "r9\t16\tchr1\t301\t30\t5M\t*\t0\t0\tCGGGG\t*\tCB:Z:AAACCTGCAGGTTCAC-1\n";
    f.close();
    return std::system(("samtools view -bS " + dir + "/t.sam -o " + bam + " 2>/dev/null").c_str()) == 0;
}

struct Run { CSC ad, dp; uint32_t edit_start; std::vector<std::string> snp_names, edit_names; };

Run run_engine(const std::string& dir, const std::string& bam, bool with_edit) {
    singlet::PileupConfig cfg;
    cfg.barcode_path = dir + "/bc.tsv";
    cfg.snp_path = dir + "/snps.tsv";
    if (with_edit) cfg.edit_path = dir + "/edit.tsv";
    cfg.threads = 1; cfg.min_mapq = 20; cfg.min_baseq = 10;
    singlet::PileupEngine e(cfg);
    Run r{};
    if (!e.load_references()) { std::cout << "  load_references failed\n"; return r; }
    e.run(bam);
    r.ad = e.snp_ad().to_csc(); r.dp = e.snp_dp().to_csc();
    r.edit_start = e.edit_snp_start(); r.snp_names = e.snp_names();
    for (uint32_t i = 0; i < e.n_edit_sites(); ++i) r.edit_names.push_back(e.edit_name(i));
    return r;
}

int64_t at(const CSC& m, int32_t row, uint32_t col) {
    for (int32_t k = m.indptr[col]; k < m.indptr[col + 1]; ++k) if (m.indices[k] == row) return m.data[k];
    return 0;
}

void test_engine() {
    std::cout << "=== engine: editing rows after SNP + AIM rows ===\n";
    const std::string dir = "/tmp/test_rna_editing_" + std::to_string(getpid());
    std::system(("mkdir -p " + dir).c_str());
    const std::string bam = dir + "/t.bam";
    if (!make_bam(bam, dir)) { CHECK(false, "samtools available to build the test BAM"); return; }
    { std::ofstream f(dir + "/bc.tsv"); f << "AAACCTGAGAAACGAG-1\nAAACCTGCAGGTTCAC-1\n"; }
    { std::ofstream f(dir + "/snps.tsv"); f << "chr1\t106\tG\tT\nchr1\t200\tA\tC\n"; }
    // panel: + site at 103, - site at 301, an uncovered site, and rows the loader must skip (wrong change, indel, header)
    { std::ofstream f(dir + "/edit.tsv");
      f << "#chrom\tpos\tref\talt\n" << "chr1\t103\tA\tG\n" << "chr1\t301\tT\tC\n" << "chr1\t900\tA\tG\n"
        << "chr1\t104\tC\tT\n" << "chr1\t105\tAG\tA\n"; }

    Run a = run_engine(dir, bam, false), b = run_engine(dir, bam, true);
    CHECK(a.edit_start == UINT32_MAX, "no panel: edit_snp_start unset");
    CHECK(b.edit_start == a.ad.nrows, "panel: editing rows start right after the SNP + AIM rows");
    CHECK(b.edit_names.size() == 3 && b.edit_names[0] == "chr1:103:A>G" && b.edit_names[1] == "chr1:301:T>C",
          "three editing sites loaded (A>G and T>C only), named chrom:pos:R>A");
    CHECK(b.snp_names == a.snp_names, "snp_names unchanged by the panel");
    CSC ea = singlet::editing::split_tail(b.ad, b.edit_start), ed = singlet::editing::split_tail(b.dp, b.edit_start);
    CHECK(same(b.ad, a.ad) && same(b.dp, a.dp), "SNP matrices after split_tail are byte for byte the run without the panel");
    CHECK(ea.nrows == 3 && ed.nrows == 3, "editing matrices have one row per panel site");
    CHECK(at(ed, 0, 0) == 3 && at(ea, 0, 0) == 1, "bc1 + site: 2 ref + 1 edited");
    CHECK(at(ed, 0, 1) == 1 && at(ea, 0, 1) == 1, "bc2 + site: 1 edited, C base and low-MAPQ read out");
    CHECK(at(ed, 1, 1) == 3 && at(ea, 1, 1) == 2, "bc2 - site: 1 T ref + 2 C edited");
    CHECK(at(ed, 2, 0) == 0 && at(ed, 2, 1) == 0, "uncovered site empty");

    std::cout << "=== summarize and writers ===\n";
    std::vector<uint32_t> cells{0, 1};
    auto er = singlet::editing::summarize(ea, ed, cells);
    CHECK(er.covered == std::vector<uint32_t>({0, 1}), "covered sites: the two with reads");
    CHECK(er.alt_total == 4 && er.dp_total == 7, "called-cell totals 4 edited of 7");
    CHECK(er.cell_alt[0] == 1 && er.cell_dp[0] == 3 && er.cell_alt[1] == 3 && er.cell_dp[1] == 4 && er.cell_sites[1] == 2, "per-cell sums");
    CHECK(er.site[0].cells_cov == 2 && er.site[0].cells_edited == 2 && er.site[1].cells_cov == 1, "per-site cell counts");
    std::vector<uint32_t> only1{1};
    auto er1 = singlet::editing::summarize(ea, ed, only1);
    CHECK(er1.site[0].dp == 1 && er1.site[0].dp_all == 4 && er1.alt_total == 3, "non-called barcodes count only in umis_all_barcodes");
    std::vector<std::string> bcs{"AAACCTGAGAAACGAG-1", "AAACCTGCAGGTTCAC-1"};
    CHECK(singlet::editing::write_sites(dir + "/s.tsv.gz", er, std::vector<std::string>(b.edit_names.begin(), b.edit_names.begin() + 2)), "write_sites");
    CHECK(singlet::editing::write_cells(dir + "/c.tsv", er, cells, bcs), "write_cells");
    CHECK(singlet::editing::write_summary(dir + "/e.json", dir + "/edit.tsv", 3, er, 2), "write_summary");
    std::system(("gzip -dc " + dir + "/s.tsv.gz > " + dir + "/s.tsv").c_str());
    std::ifstream sf(dir + "/s.tsv"); std::stringstream ss; ss << sf.rdbuf();
    CHECK(ss.str().find("chr1:103:A>G\t+\t2\t2\t0.5\t2\t2\t4\n") != std::string::npos &&
          ss.str().find("chr1:301:T>C\t-\t1\t2\t0.666667\t1\t1\t3\n") != std::string::npos, "site rows: strand, ref, edited, level");
    std::ifstream cf(dir + "/c.tsv"); std::stringstream cs; cs << cf.rdbuf();
    CHECK(cs.str().find("AAACCTGCAGGTTCAC-1\t3\t4\t2\t0.75\n") != std::string::npos, "cell row: edited, total, sites, index");
    std::ifstream jf(dir + "/e.json"); std::stringstream js; js << jf.rdbuf();
    CHECK(js.str().find("\"editing_index\": 0.571429") != std::string::npos && js.str().find("\"panel\": \"edit.tsv\"") != std::string::npos,
          "summary: index 4/7 and the panel's file name");
    std::system(("rm -rf " + dir).c_str());
}
}  // namespace

int main() {
    test_split_random();
    test_engine();
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail ? 1 : 0;
}
