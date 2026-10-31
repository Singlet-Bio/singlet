// SPDX-License-Identifier: MIT
// tests/cpp/test_sj_unique_only.cpp
//
// Junction molecules come from uniquely mapped alignments only (NH == 1), as STARsolo's SJ feature does. The primary alignment of a
// multi-mapped read used to add junctions that long-read truth does not support (PBMC-E pbmc1: precision 0.08, against 0.74 for the
// junctions the two counters share) for 0.8 points of recall.
//
// A synthetic BAM (SAM text + samtools), one barcode:
//   read_unique   NH:1  spliced across junction A (5M10N5M at 0-based 105)
//   read_no_nh    (no NH tag)  spliced across junction A, another UMI: a record without NH counts as unique
//   read_multi    NH:2  spliced across junction B (5M12N5M at 0-based 105), with its secondary alignment on another locus
// Default (sj_unique_only): 2 junction molecules, junction A only. With sj_unique_only = false (--sj-all-alignments): 3, A and B.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "singlet/pileup/pileup_engine.h"

static int failures = 0;
static void check(const char* name, uint64_t actual, uint64_t expected) {
    if (actual != expected) { std::cerr << "FAIL: " << name << " = " << actual << ", expected " << expected << "\n"; ++failures; }
    else std::cerr << "  OK: " << name << " = " << actual << "\n";
}

static singlet::PileupStats run_with(const std::string& dir, const std::string& bam, bool unique_only, size_t* n_names) {
    singlet::PileupConfig cfg;
    cfg.barcode_path = dir + "/bc.txt";
    cfg.exon_gtf_path = dir + "/mini.gtf";
    cfg.threads = 1;
    cfg.min_mapq = 20;
    cfg.count_introns = true;
    cfg.count_sj = true;
    cfg.umi_dedup = true;
    cfg.stranded = true;
    cfg.sj_unique_only = unique_only;
    singlet::PileupEngine eng(cfg);
    if (!eng.load_references()) { std::cerr << "FAIL: load_references\n"; ++failures; return {}; }
    auto st = eng.run(bam);
    if (n_names) *n_names = eng.sj_names().size();
    return st;
}

int main() {
    const std::string dir = "/tmp/test_sj_unique_" + std::to_string(getpid());
    if (std::system(("mkdir -p " + dir).c_str()) != 0) return 1;
    {
        std::ofstream f(dir + "/mini.gtf");
        f << "chr1\t.\tgene\t101\t160\t.\t+\t.\tgene_id \"ENSG1\"; gene_name \"GENE1\";\n";
        f << "chr1\t.\texon\t101\t110\t.\t+\t.\tgene_id \"ENSG1\"; gene_name \"GENE1\";\n";
        f << "chr1\t.\texon\t121\t130\t.\t+\t.\tgene_id \"ENSG1\"; gene_name \"GENE1\";\n";
        f << "chr1\t.\texon\t123\t160\t.\t+\t.\tgene_id \"ENSG1\"; gene_name \"GENE1\";\n";
        std::ofstream b(dir + "/bc.txt");
        b << "BARCODE1\n";
        std::ofstream s(dir + "/t.sam");
        s << "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr1\tLN:2000\n";
        s << "read_unique\t0\tchr1\t106\t255\t5M10N5M\t*\t0\t0\tAAAAAAAAAA\t*\tNH:i:1\tCB:Z:BARCODE1\tUB:Z:UMI0000001\n";
        s << "read_no_nh\t0\tchr1\t106\t255\t5M10N5M\t*\t0\t0\tAAAAAAAAAA\t*\tCB:Z:BARCODE1\tUB:Z:UMI0000002\n";
        s << "read_multi\t0\tchr1\t106\t3\t5M12N5M\t*\t0\t0\tAAAAAAAAAA\t*\tNH:i:2\tHI:i:1\tCB:Z:BARCODE1\tUB:Z:UMI0000003\n";
        s << "read_multi\t256\tchr1\t1006\t3\t10M\t*\t0\t0\tAAAAAAAAAA\t*\tNH:i:2\tHI:i:2\tCB:Z:BARCODE1\tUB:Z:UMI0000003\n";
    }
    if (std::system(("samtools view -bS " + dir + "/t.sam -o " + dir + "/t.bam 2>/dev/null").c_str()) != 0) {
        std::cerr << "SKIP: samtools not available\n";
        return 0;
    }

    size_t names = 0;
    auto st = run_with(dir, dir + "/t.bam", true, &names);
    check("default: junction molecules", st.sj_hits, 2);
    check("default: distinct junctions (the multi-mapped read's junction is not counted)", names, 1);

    names = 0;
    st = run_with(dir, dir + "/t.bam", false, &names);
    check("--sj-all-alignments: junction molecules", st.sj_hits, 3);
    check("--sj-all-alignments: distinct junctions", names, 2);

    if (std::system(("rm -rf " + dir).c_str()) != 0) { /* best effort */ }
    if (failures == 0) { std::cerr << "test_sj_unique_only: PASS\n"; return 0; }
    std::cerr << "test_sj_unique_only: " << failures << " FAILED\n";
    return 1;
}
