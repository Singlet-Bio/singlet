// SPDX-License-Identifier: MIT
// tests/cpp/test_gpu_pileup_te.cpp
//
// TE molecules (te_counts.h) from the GPU route's host pass (SINGLET_GPU_PILEUP=on, PileupEngine::gpu_host_pass) against the CPU pileup
// on a synthetic BAM. Compared: the TE records as export.h turns them into outputs, that is the subfamily matrix (te_counts), the locus
// matrix (te_locus_counts), the locus table (te_loci.tsv.gz) and the four totals of build_te_tables.
//
// A TE molecule is keyed by (row, cell, hash of the UMI text). The host pass holds each read's UMI packed and makes the text on demand,
// so a site that hashes the text before it is made gives every read of a chunk one key, and the reads of one cell and one TE collapse
// into one molecule. The fixture makes that visible: several UMIs per cell and TE, one UMI shared by several reads, one UMI in two loci
// of a subfamily and in two cells, multimappers (subfamily only), a spliced read, an intronic TE and an exonic read over a TE (which
// stays with the gene).
//
// SINGLET_GPU_PILEUP is read once per process (gpupileup::mode()), so the GPU side runs in a child process (this binary with --child).
// Without a GPU build or a usable CUDA device the test prints SKIP and returns 77 (ctest SKIP_RETURN_CODE).

#include <unistd.h>
#include <sys/wait.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/pileup_engine.h"
#include "test_harness.h"
#ifdef SINGLET_HAVE_ENGINE_GPU
#include "singlet/engine_gpu.h"
#endif

namespace {

const char* kBC1 = "AAACCTGAGAAACGAG-1";
const char* kBC2 = "AAACCTGAGAAAGTGG-1";

std::string sam_line(const std::string& q, int flag, const char* bc, const std::string& chr, int pos, int mapq, const std::string& cigar,
                     int read_len, const std::string& umi, int nh = 1) {
    const std::string seq(static_cast<size_t>(read_len), 'C'), qual(static_cast<size_t>(read_len), 'I');
    return q + "\t" + std::to_string(flag) + "\t" + chr + "\t" + std::to_string(pos) + "\t" + std::to_string(mapq) + "\t" + cigar +
           "\t*\t0\t0\t" + seq + "\t" + qual + "\tCB:Z:" + bc + "\tUB:Z:" + umi + "\tNH:i:" + std::to_string(nh) + "\n";
}

// Genes (1-based): chr1 '+' G1 101-400, exon1 101-200, exon2 301-400, intron 201-300; chr2 '+' G3 501-600.
// TEs: chr2 AluY_dup1 101-160, AluY_dup2 201-260, L1HS_dup1 301-380; chr1 AluSx_dup1 141-180 (inside G1's exon1), L1HS_dup2 221-290
// (inside G1's intron).
bool write_inputs(const std::string& dir) {
    std::ofstream g(dir + "/mini.gtf");
    auto gl = [&](const char* chr, const char* feat, int s, int e, const char* id, const char* nm) {
        g << chr << "\t.\t" << feat << "\t" << s << "\t" << e << "\t.\t+\t.\tgene_id \"" << id << "\"; gene_name \"" << nm << "\";\n";
    };
    gl("chr1", "gene", 101, 400, "G1", "GENE1"); gl("chr1", "exon", 101, 200, "G1", "GENE1"); gl("chr1", "exon", 301, 400, "G1", "GENE1");
    gl("chr2", "gene", 501, 600, "G3", "GENE3"); gl("chr2", "exon", 501, 600, "G3", "GENE3");
    g.close();
    std::ofstream t(dir + "/te.gtf");
    auto tl = [&](const char* chr, int s, int e, const char* sub, const char* locus, const char* fam, const char* cls) {
        t << chr << "\thg38_rmsk\texon\t" << s << "\t" << e << "\t1\t+\t.\tgene_id \"" << sub << "\"; transcript_id \"" << locus
          << "\"; family_id \"" << fam << "\"; class_id \"" << cls << "\";\n";
    };
    tl("chr2", 101, 160, "AluY", "AluY_dup1", "Alu", "SINE"); tl("chr2", 201, 260, "AluY", "AluY_dup2", "Alu", "SINE");
    tl("chr2", 301, 380, "L1HS", "L1HS_dup1", "L1", "LINE");
    tl("chr1", 141, 180, "AluSx", "AluSx_dup1", "Alu", "SINE"); tl("chr1", 221, 290, "L1HS", "L1HS_dup2", "L1", "LINE");
    t.close();
    std::ofstream b(dir + "/barcodes.tsv"); b << kBC1 << "\n" << kBC2 << "\n"; b.close();

    const char* U[] = {"", "ACGTACGTAC", "CAGTCAGTCA", "GTCAGTCAGT", "TGACTGACTG", "ACACACACAC", "AGAGAGAGAG", "ATATATATAT", "CGCGCGCGCG",
                       "CTCTCTCTCT", "GTGTGTGTGT", "AACCAACCAA", "AAGGAAGGAA", "AATTAATTAA", "CCGGCCGGCC", "CCTTCCTTCC", "GGTTGGTTGG"};
    std::ofstream f(dir + "/t.sam");
    f << "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr1\tLN:1000\n@SQ\tSN:chr2\tLN:1000\n";
    // cell 1, AluY_dup1: four UMIs, one read each (4 molecules), and three reads of a fifth UMI (1 molecule)
    for (int i = 1; i <= 4; ++i) f << sam_line("a" + std::to_string(i), 0, kBC1, "chr2", 110, 255, "20M", 20, U[i]);
    for (int i = 5; i <= 7; ++i) f << sam_line("a" + std::to_string(i), 0, kBC1, "chr2", 120, 255, "20M", 20, U[5]);
    // cell 1, AluY_dup2: U1 again (its own molecule at the locus, the same molecule as a1 at the subfamily) and a new UMI
    f << sam_line("b1", 0, kBC1, "chr2", 210, 255, "20M", 20, U[1]);
    f << sam_line("b2", 0, kBC1, "chr2", 210, 255, "20M", 20, U[6]);
    // cell 1, L1HS_dup1: two UMIs, and a multimapper whose primary lies in it (subfamily row only; the secondary is intergenic)
    f << sam_line("l1", 0, kBC1, "chr2", 310, 255, "20M", 20, U[7]);
    f << sam_line("l2", 0, kBC1, "chr2", 310, 255, "20M", 20, U[8]);
    f << sam_line("mm", 0, kBC1, "chr2", 330, 3, "20M", 20, U[9], 2);
    f << sam_line("mm", 256, kBC1, "chr1", 600, 3, "20M", 20, U[9], 2);
    // cell 1: a read in G1's exon1 over AluSx_dup1 stays with the gene; two reads in G1's intron over L1HS_dup2 are TE molecules
    f << sam_line("ex", 0, kBC1, "chr1", 145, 255, "20M", 20, U[10]);
    f << sam_line("in1", 0, kBC1, "chr1", 230, 255, "20M", 20, U[11]);
    f << sam_line("in2", 0, kBC1, "chr1", 230, 255, "20M", 20, U[12]);
    // cell 2, AluY_dup1: U1 and U2 again (another cell: their own molecules) and a third UMI; AluY_dup2: one read
    f << sam_line("c1", 0, kBC2, "chr2", 130, 255, "20M", 20, U[1]);
    f << sam_line("c2", 0, kBC2, "chr2", 130, 255, "20M", 20, U[2]);
    f << sam_line("c3", 0, kBC2, "chr2", 130, 255, "20M", 20, U[13]);
    f << sam_line("c4", 0, kBC2, "chr2", 215, 255, "20M", 20, U[14]);
    // cell 2: a multimapper in L1HS_dup1, and a spliced read whose first block lies in AluY_dup1 (the second block is intergenic)
    f << sam_line("mm2", 0, kBC2, "chr2", 340, 3, "20M", 20, U[15], 2);
    f << sam_line("mm2", 256, kBC2, "chr1", 620, 3, "20M", 20, U[15], 2);
    f << sam_line("sp", 0, kBC2, "chr2", 105, 255, "10M50N10M", 20, U[16]);
    f.close();
    const std::string cmd = "samtools sort -o " + dir + "/t.bam " + dir + "/t.sam 2>/dev/null && samtools index " + dir + "/t.bam";
    return std::system(cmd.c_str()) == 0;
}

// the run's TE outputs as text, one "key value" line per member
std::string run_dump(const std::string& dir) {
    singlet::PileupConfig cfg;
    cfg.barcode_path = dir + "/barcodes.tsv"; cfg.exon_gtf_path = dir + "/mini.gtf"; cfg.te_gtf_path = dir + "/te.gtf";
    cfg.threads = 1; cfg.umi_len = 10; cfg.umi_dedup_directional = true;
    cfg.count_sj = false; cfg.aim_ancestry = false; cfg.count_vdj = false;
    singlet::PileupEngine e(cfg);
    if (!e.load_references()) return "error load_references failed\n";
    e.run(dir + "/t.bam");
    std::ostringstream o;
    o << "has_te " << (e.has_te() ? 1 : 0) << "\n";
    const auto t = singlet::build_te_tables(e.te_records(), e.te_model(), static_cast<uint32_t>(e.barcodes().size()), e.tid_to_chrom());
    auto mat = [&](const char* name, const singlet::TeMatrix& m) {
        o << name << "_shape " << m.nrows << ' ' << m.ncols << "\n";
        o << name << "_indptr"; for (auto v : m.indptr) o << ' ' << v;
        o << "\n" << name << "_indices"; for (auto v : m.indices) o << ' ' << v;
        o << "\n" << name << "_data"; for (auto v : m.data) o << ' ' << static_cast<unsigned>(v);
        o << "\n" << name << "_names"; for (const auto& s : m.names) o << ' ' << s;
        o << "\n";
    };
    mat("te_counts", t.fam); mat("te_locus_counts", t.locus);
    std::string tsv = t.loci_tsv;
    for (char& c : tsv) { if (c == '\n') c = '|'; else if (c == '\t') c = ','; }
    o << "te_loci " << tsv << "\n";
    o << "n_records " << t.n_records << "\nn_unique_records " << t.n_unique_records << "\nfam_molecules " << t.fam_molecules
      << "\nlocus_molecules " << t.locus_molecules << "\n";
    return o.str();
}

std::map<std::string, std::string> parse(const std::string& s) {
    std::map<std::string, std::string> m; std::istringstream in(s); std::string line;
    while (std::getline(in, line)) {
        const size_t sp = line.find(' ');
        m[line.substr(0, sp)] = sp == std::string::npos ? "" : line.substr(sp + 1);
    }
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "--child") {   // the GPU side: SINGLET_GPU_PILEUP=on is in this process's environment
        std::ofstream(argv[3]) << run_dump(argv[2]);
        return 0;
    }
#ifndef SINGLET_HAVE_ENGINE_GPU
    std::cout << "SKIP: CPU-only build (no GPU pileup)\n";
    return 77;
#else
    {
        std::string why;
        if (singlet_engine_gpu::deviceCount() <= 0 || !singlet_engine_gpu::available(&why)) {
            std::cout << "SKIP: no usable CUDA device" << (why.empty() ? "" : " (" + why + ")") << "\n";
            return 77;
        }
    }
    unsetenv("SINGLET_GPU_PILEUP");
    char tmpl[] = "/tmp/test_gpu_pileup_te_XXXXXX";
    const char* dp = mkdtemp(tmpl);
    if (!dp) { std::cerr << "mkdtemp failed\n"; return 1; }
    const std::string dir = dp;
    if (!write_inputs(dir)) { std::cout << "SKIP: samtools not found (the fixture is a BAM)\n"; return 77; }
    char self[4096]; const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n <= 0) { std::cerr << "cannot find the test binary\n"; return 1; }
    self[n] = 0;

    const std::string cpu = run_dump(dir);
    const std::string out = dir + "/gpu.txt", log = dir + "/gpu.log";
    const std::string cmd = "SINGLET_GPU_PILEUP=on '" + std::string(self) + "' --child '" + dir + "' '" + out + "' 2> '" + log + "'";
    const int rc = std::system(cmd.c_str());
    CHECK(rc == 0, "GPU child exited 0");
    std::ifstream gi(out); std::stringstream gs; gs << gi.rdbuf(); const std::string gpu = gs.str();
    std::ifstream li(log); std::stringstream ls; ls << li.rdbuf(); const std::string lg = ls.str();
    CHECK(lg.find("[gpu_pileup] on:") != std::string::npos, "the GPU pileup ran ([gpu_pileup] on:)");
    CHECK(lg.find("[gpu_pileup] failed") == std::string::npos && lg.find("not run") == std::string::npos, "no GPU failure / decline");
    const auto C = parse(cpu), G = parse(gpu);
    auto num = [](const std::map<std::string, std::string>& m, const char* k) {
        return m.count(k) ? std::strtoull(m.at(k).c_str(), nullptr, 10) : ~0ull;
    };

    // the fixture exercises what it should (CPU side)
    CHECK(num(C, "has_te") == 1, "CPU: the TE model is loaded");
    CHECK(num(C, "n_records") == 20, "CPU: 20 TE records (the exonic read has none)");
    CHECK(num(C, "n_unique_records") == 18, "CPU: 18 records with a locus (the two multimappers have none)");
    CHECK(num(C, "fam_molecules") == 17, "CPU: 17 subfamily molecules");
    CHECK(num(C, "locus_molecules") == 16, "CPU: 16 locus molecules");
    CHECK(C.count("te_locus_counts_names") && C.at("te_locus_counts_names") == "AluY_dup1 AluY_dup2 L1HS_dup1 L1HS_dup2",
          "CPU: four loci have a molecule (AluSx_dup1, under the exon, has none)");

    // the GPU route's host pass gives the same outputs, member by member
    for (const char* k : {"has_te", "te_counts_shape", "te_counts_indptr", "te_counts_indices", "te_counts_data", "te_counts_names",
                          "te_locus_counts_shape", "te_locus_counts_indptr", "te_locus_counts_indices", "te_locus_counts_data",
                          "te_locus_counts_names", "te_loci", "n_records", "n_unique_records", "fam_molecules", "locus_molecules"}) {
        const bool same = C.count(k) && G.count(k) && C.at(k) == G.at(k);
        CHECK(same, std::string(k) + " GPU == CPU");
        if (!same) std::cout << "    CPU: " << (C.count(k) ? C.at(k) : "<none>") << "\n    GPU: " << (G.count(k) ? G.at(k) : "<none>") << "\n";
    }
    std::system(("rm -rf '" + dir + "'").c_str());
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
#endif
}
