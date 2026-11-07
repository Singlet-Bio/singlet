// SPDX-License-Identifier: MIT
// tests/cpp/test_gene_full_set.cpp
//
// intron_rule::kGeneFullSet: a read counts only when the same-strand gene bodies (exons or introns) under ALL of its alignments
// form exactly one gene, as STARsolo GeneFull decides (Transcriptome::geneFullAlignOverlap, SoloReadFeature::record).
// Pins the serial run() and run_parallel() to the same result for each worker split.
//   gene A chr1 101-1000 (exons 101-200, 801-1000), gene B nested in A's intron (exon 401-500), gene C chr2 (exons 101-200,
//   801-900), gene D chr3 (exon 101-300). All on +.
//   r1 NH 1 in B's exon (inside A's intron)        exon priority: B          GeneFull: none (A and B)
//   r2 NH 1 in A's first exon                      A                          A
//   r3 NH 2, primary D exon, secondary C intron    D                          none (C and D)
//   r4 NH 2, primary D exon, secondary intergenic  D                          D
//   r5 NH 2, primary intergenic, secondary D exon  D (smallest secondary)     D

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

#include "singlet/pileup/pileup_engine.h"
#include "test_harness.h"

namespace {

const char* kBC = "AAACCTGAGAAACGAG-1";

std::string sam_line(const std::string& q, int flag, const std::string& chr, int pos, int mapq,
                     const std::string& cigar, int read_len, const std::string& umi, int nh = 1) {
    std::string seq(static_cast<size_t>(read_len), 'A');
    return q + "\t" + std::to_string(flag) + "\t" + chr + "\t" + std::to_string(pos) + "\t" +
           std::to_string(mapq) + "\t" + cigar + "\t*\t0\t0\t" + seq + "\t*\tCB:Z:" + kBC +
           "\tUB:Z:" + umi + "\tNH:i:" + std::to_string(nh) + "\n";
}

bool write_inputs(const std::string& dir) {
    std::ofstream g(dir + "/mini.gtf");
    auto gl = [&](const char* chr, const char* feat, int s, int e, const char* id) {
        g << chr << "\t.\t" << feat << "\t" << s << "\t" << e << "\t.\t+\t.\tgene_id \"" << id
          << "\"; gene_name \"" << id << "\"; transcript_id \"" << id << "-T\";\n";
    };
    gl("chr1", "gene", 101, 1000, "A"); gl("chr1", "exon", 101, 200, "A"); gl("chr1", "exon", 801, 1000, "A");
    gl("chr1", "gene", 401, 500, "B"); gl("chr1", "exon", 401, 500, "B");
    gl("chr2", "gene", 101, 900, "C"); gl("chr2", "exon", 101, 200, "C"); gl("chr2", "exon", 801, 900, "C");
    gl("chr3", "gene", 101, 300, "D"); gl("chr3", "exon", 101, 300, "D");
    g.close();
    std::ofstream b(dir + "/barcodes.tsv");
    b << kBC << "\n";
    b.close();
    std::ofstream f(dir + "/t.sam");
    f << "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr1\tLN:2000\n@SQ\tSN:chr2\tLN:2000\n@SQ\tSN:chr3\tLN:2000\n";
    f << sam_line("r1", 0, "chr1", 420, 255, "20M", 20, "AAAAAAAAAA");
    f << sam_line("r2", 0, "chr1", 120, 255, "20M", 20, "CCCCCCCCCC");
    f << sam_line("r3", 0, "chr3", 150, 3, "20M", 20, "GGGGGGGGGG", 2);
    f << sam_line("r3", 256, "chr2", 400, 3, "20M", 20, "GGGGGGGGGG", 2);
    f << sam_line("r4", 0, "chr3", 200, 3, "20M", 20, "TTTTTTTTTT", 2);
    f << sam_line("r4", 256, "chr2", 1500, 3, "20M", 20, "TTTTTTTTTT", 2);
    f << sam_line("r5", 0, "chr1", 1500, 3, "20M", 20, "ACACACACAC", 2);
    f << sam_line("r5", 256, "chr3", 250, 3, "20M", 20, "ACACACACAC", 2);
    f.close();
    std::string cmd = "samtools sort -o " + dir + "/t.bam " + dir + "/t.sam 2>/dev/null && samtools index " + dir + "/t.bam";
    return std::system(cmd.c_str()) == 0;
}

std::map<std::string, int> exon_by_gene(const singlet::PileupEngine& e) {
    std::map<std::string, int> out;
    const auto& gm = e.gene_model();
    auto ex = e.exons().to_csc();
    for (size_t k = 0; k < ex.data.size(); ++k)
        out[gm.gene_ids()[gm.exon_to_gene(static_cast<uint32_t>(ex.indices[k]))]] += ex.data[k];
    return out;
}

void run_case(const std::string& dir, bool parallel, uint32_t rules, int workers = 3) {
    std::string tag = std::string(parallel ? "run_parallel(" + std::to_string(workers) + ")" : "run") + " / rules " + std::to_string(rules);
    std::cout << "\n=== " << tag << " ===\n";
    singlet::PileupConfig cfg;
    cfg.barcode_path = dir + "/barcodes.tsv";
    cfg.exon_gtf_path = dir + "/mini.gtf";
    cfg.threads = 2;
    cfg.umi_len = 10;
    cfg.count_sj = false;
    cfg.aim_ancestry = false;
    cfg.count_vdj = false;
    cfg.auto_strand = false;
    cfg.intron_rules = rules;
    singlet::PileupEngine e(cfg);
    if (!e.load_references()) { CHECK(false, tag + ": load_references"); return; }
    if (parallel) e.run_parallel(dir + "/t.bam", workers); else e.run(dir + "/t.bam");
    auto g = exon_by_gene(e);
    const bool gf = rules & singlet::intron_rule::kGeneFullSet;
    const bool nep = gf || (rules & singlet::intron_rule::kNoExonPriority);
    CHECK(g["A"] == 1, tag + ": A's exonic read");
    CHECK(g["B"] == (nep ? 0 : 1), tag + ": B's exon inside A's intron counts only with exon priority");
    CHECK(g["C"] == 0, tag + ": a secondary in C's intron never assigns C");
    CHECK(g["D"] == (gf ? 2 : 3), tag + ": D loses the multimapper whose secondary lies in C's intron under kGeneFullSet");
}

}  // namespace

int main() {
    std::string dir = "/tmp/test_gene_full_set_" + std::to_string(getpid());
    std::system(("mkdir -p " + dir).c_str());
    if (!write_inputs(dir)) {
        std::cout << "SKIP: samtools not available\n";
    } else {
        const uint32_t gfs = singlet::intron_rule::kGeneFullSet;
        for (uint32_t r : {0u, (uint32_t) singlet::intron_rule::kNoExonPriority, gfs, 397u, 397u | gfs}) {
            run_case(dir, false, r);
            for (int w : {1, 2, 3}) run_case(dir, true, r, w);
        }
    }
    std::system(("rm -rf " + dir).c_str());
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
