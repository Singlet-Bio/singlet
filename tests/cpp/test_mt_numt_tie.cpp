// SPDX-License-Identifier: MIT
// tests/cpp/test_mt_numt_tie.cpp
//
// intron_rule::kMtPrefer: a multimapper tied between a chrM gene and a nuclear copy of it (NUMT) is counted
// for the chrM gene. The rule used to exist only in run_parallel() (cross-worker merge); the serial run() that
// the unsorted STAR stream goes through dropped such reads as gene-ambiguous (neuron1k mt-Atp6: 15 vs 326,548).
// Both paths now share one helper; this test pins them to the same result, whichever alignment is primary.

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
    auto gl = [&](const char* chr, const char* feat, int s, int e, const char* id, const char* nm) {
        g << chr << "\t.\t" << feat << "\t" << s << "\t" << e << "\t.\t+\t.\tgene_id \"" << id
          << "\"; gene_name \"" << nm << "\";\n";
    };
    gl("chr1", "gene", 101, 400, "NUMT1", "Atp6-ps");
    gl("chr1", "exon", 101, 400, "NUMT1", "Atp6-ps");
    gl("chr2", "gene", 101, 400, "NUMT2", "Co3-ps");
    gl("chr2", "exon", 101, 400, "NUMT2", "Co3-ps");
    gl("chrM", "gene", 101, 400, "MT1", "mt-Atp6");
    gl("chrM", "exon", 101, 400, "MT1", "mt-Atp6");
    gl("chrM", "gene", 501, 800, "MT2", "mt-Co3");
    gl("chrM", "exon", 501, 800, "MT2", "mt-Co3");
    g.close();

    std::ofstream b(dir + "/barcodes.tsv");
    b << kBC << "\n";
    b.close();

    std::ofstream f(dir + "/t.sam");
    f << "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr1\tLN:1000\n@SQ\tSN:chr2\tLN:1000\n@SQ\tSN:chrM\tLN:1000\n";
    // t1: NUMT primary (chr1), chrM secondary: the STAR pattern for mt-Atp6
    f << sam_line("t1", 0, "chr1", 150, 3, "20M", 20, "AAAAAAAAAA", 2);
    f << sam_line("t1", 256, "chrM", 150, 3, "20M", 20, "AAAAAAAAAA", 2);
    // t2: chrM primary, NUMT secondary (mt-Co3)
    f << sam_line("t2", 0, "chrM", 550, 3, "20M", 20, "CCCCCCCCCC", 2);
    f << sam_line("t2", 256, "chr2", 150, 3, "20M", 20, "CCCCCCCCCC", 2);
    // t3: two nuclear genes, no chrM: still gene-ambiguous and not counted
    f << sam_line("t3", 0, "chr1", 300, 3, "20M", 20, "GGGGGGGGGG", 2);
    f << sam_line("t3", 256, "chr2", 300, 3, "20M", 20, "GGGGGGGGGG", 2);
    // t4: a unique chrM read
    f << sam_line("t4", 0, "chrM", 600, 255, "20M", 20, "TTTTTTTTTT");
    f.close();

    std::string cmd = "samtools sort -o " + dir + "/t.bam " + dir + "/t.sam 2>/dev/null && samtools index " +
                      dir + "/t.bam";
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

void run_case(const std::string& dir, bool parallel, bool mt_prefer, int workers = 3) {
    std::string tag = std::string(parallel ? "run_parallel(" + std::to_string(workers) + ")" : "run") + (mt_prefer ? " / kMtPrefer" : " / rules 0");
    std::cout << "\n=== " << tag << " ===\n";
    singlet::PileupConfig cfg;
    cfg.barcode_path = dir + "/barcodes.tsv";
    cfg.exon_gtf_path = dir + "/mini.gtf";
    cfg.threads = 2;
    cfg.umi_len = 10;
    cfg.count_sj = false;
    cfg.aim_ancestry = false;
    cfg.count_vdj = false;
    cfg.intron_rules = mt_prefer ? singlet::intron_rule::kMtPrefer : 0;
    singlet::PileupEngine e(cfg);
    if (!e.load_references()) { CHECK(false, tag + ": load_references"); return; }
    auto st = parallel ? e.run_parallel(dir + "/t.bam", workers) : e.run(dir + "/t.bam");
    auto g = exon_by_gene(e);
    CHECK(g["MT1"] == (mt_prefer ? 1 : 0), tag + ": NUMT-primary tie counted for mt-Atp6 only with kMtPrefer");
    CHECK(g["MT2"] == (mt_prefer ? 2 : 1), tag + ": chrM-primary tie + unique read for mt-Co3");
    CHECK(g["NUMT1"] == 0 && g["NUMT2"] == 0, tag + ": nuclear copies get nothing");
    CHECK(st.mt_numt_rescued == static_cast<uint64_t>(mt_prefer ? 2 : 0), tag + ": rescued reads");
}

}  // namespace

int main() {
    std::string dir = "/tmp/test_mt_numt_tie_" + std::to_string(getpid());
    std::system(("mkdir -p " + dir).c_str());
    if (!write_inputs(dir)) {
        std::cout << "SKIP: samtools not available\n";
    } else {
        for (bool mt : {true, false}) run_case(dir, false, mt);
        // the parallel pileup must not depend on how the chromosomes are spread over the workers (1: one worker holds all; 2-4: mixtures)
        for (int w : {1, 2, 3, 4})
            for (bool mt : {true, false}) run_case(dir, true, mt, w);
    }
    std::system(("rm -rf " + dir).c_str());
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
