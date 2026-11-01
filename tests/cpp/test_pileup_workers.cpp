// SPDX-License-Identifier: MIT
// tests/cpp/test_pileup_workers.cpp
//
// The parallel pileup (run_parallel, sorted BAM) must give the same counts however the chromosomes are spread over the workers: the thread count is not
// an input of the result. Random multimappers (NUMT-like chrM + nuclear ties, nuclear pairs, three-way hits, unique reads) over many chromosomes, run with
// 1..13 workers; every statistic and every per-gene exon count must be equal to the 1-worker run.

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "singlet/pileup/pileup_engine.h"
#include "test_harness.h"

namespace {

constexpr int kChroms = 14;          // chr1..chr14, plus chrM
constexpr int kGenesPerChrom = 6;

std::string chr_name(int c) { return c == kChroms ? "chrM" : "chr" + std::to_string(c + 1); }

struct Aln { int chrom; int pos; };

bool write_inputs(const std::string& dir, uint32_t seed, int n_reads, int max_nh) {
    std::mt19937 rng(seed);
    std::ofstream g(dir + "/mini.gtf");
    for (int c = 0; c <= kChroms; ++c) {
        int ng = (c == kChroms) ? 4 : kGenesPerChrom;
        for (int k = 0; k < ng; ++k) {
            std::string id = chr_name(c) + "_g" + std::to_string(k);
            int s = 1000 + k * 5000, e = s + 599;
            g << chr_name(c) << "\t.\tgene\t" << s << "\t" << e + 800 << "\t.\t+\t.\tgene_id \"" << id << "\"; gene_name \"" << id << "\";\n";
            g << chr_name(c) << "\t.\texon\t" << s << "\t" << e << "\t.\t+\t.\tgene_id \"" << id << "\"; gene_name \"" << id << "\";\n";
            g << chr_name(c) << "\t.\texon\t" << e + 601 << "\t" << e + 800 << "\t.\t+\t.\tgene_id \"" << id << "\"; gene_name \"" << id << "\";\n";
        }
    }
    g.close();
    const char* bcs[4] = {"AAACCTGAGAAACGAG-1", "AAACCTGAGAAACTGT-1", "AAACCTGAGAAAGTGG-1", "AAACCTGAGAAATACG-1"};
    std::ofstream b(dir + "/barcodes.tsv");
    for (auto* x : bcs) b << x << "\n";
    b.close();

    std::ofstream f(dir + "/t.sam");
    f << "@HD\tVN:1.6\tSO:unsorted\n";
    for (int c = 0; c <= kChroms; ++c) f << "@SQ\tSN:" << chr_name(c) << "\tLN:100000\n";
    auto pick_aln = [&](bool in_gene_only) {
        Aln a;
        a.chrom = static_cast<int>(rng() % (kChroms + 1));
        int ng = (a.chrom == kChroms) ? 4 : kGenesPerChrom;
        int k = static_cast<int>(rng() % ng);
        bool genic = in_gene_only || (rng() % 10) < 8;
        a.pos = genic ? 1000 + k * 5000 + static_cast<int>(rng() % 500) : 1000 + k * 5000 + 3000 + static_cast<int>(rng() % 500);   // inside the gene or between genes
        return a;
    };
    const char* B = "ACGT";
    for (int i = 0; i < n_reads; ++i) {
        std::string q = "r" + std::to_string(i);
        std::string umi; for (int u = 0; u < 10; ++u) umi += B[rng() % 4];
        if (rng() % 3 == 0) umi = std::string(10, 'A') + std::string(0, 'C');   // a few duplicated UMIs
        const char* bc = bcs[rng() % 4];
        int kind = static_cast<int>(rng() % 10);
        int nh = kind < 3 ? 1 : (kind < 8 ? 2 : (kind < 9 ? 3 : 4));
        if (nh > max_nh) nh = 2;
        std::vector<Aln> alns;
        for (int a = 0; a < nh; ++a) alns.push_back(pick_aln(false));
        if (nh >= 2 && rng() % 2) alns[rng() % nh].chrom = kChroms;   // a chrM alignment: the NUMT-like tie
        if (nh >= 2 && alns[0].chrom == kChroms && rng() % 3) alns[0].chrom = static_cast<int>(rng() % kChroms);   // chrM not always primary
        for (int a = 0; a < nh; ++a) {
            int flag = a == 0 ? 0 : 256;
            f << q << "\t" << flag << "\t" << chr_name(alns[a].chrom) << "\t" << alns[a].pos << "\t" << (nh == 1 ? 255 : 3) << "\t20M\t*\t0\t0\t" << std::string(20, 'A')
              << "\t*\tCB:Z:" << bc << "\tUB:Z:" << umi << "\tNH:i:" << nh << "\n";
        }
    }
    f.close();
    std::string cmd = "samtools sort -o " + dir + "/t.bam " + dir + "/t.sam 2>/dev/null && samtools index " + dir + "/t.bam";
    return std::system(cmd.c_str()) == 0;
}

struct Result { std::map<std::string, uint64_t> st; std::map<std::string, int> genes; };

Result run_once(const std::string& dir, int workers) {
    singlet::PileupConfig cfg;
    cfg.barcode_path = dir + "/barcodes.tsv";
    cfg.exon_gtf_path = dir + "/mini.gtf";
    cfg.threads = 2;
    cfg.umi_len = 10;
    cfg.count_sj = false;
    cfg.aim_ancestry = false;
    cfg.count_vdj = false;
    cfg.intron_rules = singlet::intron_rule::kMtPrefer;
    singlet::PileupEngine e(cfg);
    Result r;
    if (!e.load_references()) { CHECK(false, "load_references"); return r; }
    auto st = workers > 0 ? e.run_parallel(dir + "/t.bam", workers) : e.run(dir + "/t.bam");
    r.st = {{"exon_hits", st.exon_hits}, {"intron_hits", st.intron_hits}, {"mt_numt_rescued", st.mt_numt_rescued}, {"multigene_reads", st.multigene_reads},
            {"umi_unique", st.umi_unique}, {"umi_duplicate", st.umi_duplicate}, {"molecules_spliced", st.molecules_spliced},
            {"molecules_unspliced", st.molecules_unspliced}, {"molecules_ambiguous", st.molecules_ambiguous}};
    const auto& gm = e.gene_model();
    auto ex = e.exons().to_csc();
    for (size_t k = 0; k < ex.data.size(); ++k)
        r.genes[gm.gene_ids()[gm.exon_to_gene(static_cast<uint32_t>(ex.indices[k]))] + "@" + std::to_string(ex.indices[k])] += ex.data[k];
    return r;
}

}  // namespace

int main() {
    std::string dir = "/tmp/test_pileup_workers_" + std::to_string(getpid());
    std::system(("mkdir -p " + dir).c_str());
    for (int max_nh : {2, 4})
    for (uint32_t seed : {1u, 2u, 3u}) {
        const std::string tag = "NH<=" + std::to_string(max_nh) + " seed " + std::to_string(seed);
        if (!write_inputs(dir, seed, 6000, max_nh)) { std::cout << "SKIP: samtools not available\n"; break; }
        Result ref = run_once(dir, 1);
        CHECK(ref.st["exon_hits"] > 0 && ref.st["mt_numt_rescued"] > 0, tag + ": the sample has counted reads and NUMT rescues");
        for (int w : {2, 3, 4, 5, 7, 8, 13}) {
            Result r = run_once(dir, w);
            for (auto& [k, v] : ref.st)
                CHECK(r.st[k] == v, tag + ", " + std::to_string(w) + " workers: " + k + " " + std::to_string(r.st[k]) + " vs 1 worker " + std::to_string(v));
            CHECK(r.genes == ref.genes, tag + ", " + std::to_string(w) + " workers: per-gene exon counts");
        }
    }
    std::system(("rm -rf " + dir).c_str());
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
