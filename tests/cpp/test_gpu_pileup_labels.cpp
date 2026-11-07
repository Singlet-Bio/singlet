// SPDX-License-Identifier: MIT
// tests/cpp/test_gpu_pileup_labels.cpp
//
// The GPU pileup (SINGLET_GPU_PILEUP=on, src/engine_gpu/gpu_pileup) against the CPU pileup on a synthetic BAM, under the default intron
// rules 397 (kUnique | kNoSplicedIntron | kFlank | kUmiGeneCR | kMtPrefer) and under 389 (without kFlank). Compared: the exon / intron /
// ambiguous accumulators (to_csc: indptr, indices, data), the molecule stats (spliced, unspliced, ambiguous, legacy_double, dropped) and the
// per-barcode legacy_exon_extra. The fixture is test_molecule_labels.cpp's (G1 exon1 / intron / exon2, nested G2, G3; M1-M9) plus reads
// for the alevin-fry USA labels (the CIGAR-N bit), kFlank (5 and 6 intronic bases), the kUnique / kNoSplicedIntron order and the legacy
// exonic extra.
//
// SINGLET_GPU_PILEUP is read once per process (gpupileup::mode()), so the GPU side runs in a child process (this binary with --child).
// Without a GPU build or a usable CUDA device the test prints SKIP and returns 77 (ctest SKIP_RETURN_CODE).

#include <unistd.h>
#include <sys/wait.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

const char* kBC = "AAACCTGAGAAACGAG-1";

std::string sam_line(const std::string& q, int flag, const std::string& chr, int pos, int mapq, const std::string& cigar,
                     int read_len, const std::string& umi, int nh = 1) {
    const std::string seq(static_cast<size_t>(read_len), 'C'), qual(static_cast<size_t>(read_len), 'I');
    return q + "\t" + std::to_string(flag) + "\t" + chr + "\t" + std::to_string(pos) + "\t" + std::to_string(mapq) + "\t" + cigar +
           "\t*\t0\t0\t" + seq + "\t" + qual + "\tCB:Z:" + kBC + "\tUB:Z:" + umi + "\tNH:i:" + std::to_string(nh) + "\n";
}

// chr1 '+': G1 gene 101-400, exon1 101-200, exon2 301-400, intron 201-300 (1-based); G2 221-260 nested in the intron. chr2 '+': G3 501-600.
bool write_inputs(const std::string& dir) {
    std::ofstream g(dir + "/mini.gtf");
    auto gl = [&](const char* chr, const char* feat, int s, int e, const char* id, const char* nm) {
        g << chr << "\t.\t" << feat << "\t" << s << "\t" << e << "\t.\t+\t.\tgene_id \"" << id << "\"; gene_name \"" << nm << "\";\n";
    };
    gl("chr1", "gene", 101, 400, "G1", "GENE1"); gl("chr1", "exon", 101, 200, "G1", "GENE1"); gl("chr1", "exon", 301, 400, "G1", "GENE1");
    gl("chr1", "gene", 221, 260, "G2", "GENE2"); gl("chr1", "exon", 221, 260, "G2", "GENE2");
    gl("chr2", "gene", 501, 600, "G3", "GENE3"); gl("chr2", "exon", 501, 600, "G3", "GENE3");
    g.close();
    std::ofstream b(dir + "/barcodes.tsv"); b << kBC << "\n"; b.close();

    std::ofstream f(dir + "/t.sam");
    f << "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr1\tLN:1000\n@SQ\tSN:chr2\tLN:1000\n";
    // M1-M9 of test_molecule_labels.cpp
    f << sam_line("m1", 0, "chr1", 120, 255, "20M", 20, "AAAAAAAAAA");                 // exon-only, no junction: A
    f << sam_line("m2", 0, "chr1", 191, 255, "20M", 20, "CCCCCCCCCC");                 // exon1/intron boundary, 10 intronic bases: U
    f << sam_line("m3a", 0, "chr1", 320, 255, "20M", 20, "GGGGGGGGGG");                // exon read + intron read: U, legacy extra
    f << sam_line("m3b", 0, "chr1", 270, 255, "20M", 20, "GGGGGGGGGG");
    f << sam_line("m4", 0, "chr1", 205, 255, "10M", 10, "TTTTTTTTTT");                 // intronic: U
    f << sam_line("m5", 0, "chr1", 196, 255, "5M100N5M", 10, "ACACACACAC");            // junction read: S
    f << sam_line("m6", 0, "chr1", 230, 255, "10M", 10, "AGAGAGAGAG");                 // G2 exon: A
    for (int i = 0; i < 5; ++i) f << sam_line("m7_" + std::to_string(i), 0, "chr1", 140, 255, "20M", 20, "ACGTACGTAC");
    f << sam_line("m7x", 0, "chr1", 280, 255, "10M", 10, "ACGTACGTAA");                // merged into ACGT..C (directional): U, legacy extra
    f << sam_line("m8a", 0, "chr1", 150, 255, "20M", 20, "CTCTCTCTCT");                // exonic + intronic multimapper (kUnique drops it): A
    f << sam_line("m8b", 0, "chr1", 210, 3, "10M", 10, "CTCTCTCTCT", 2);
    f << sam_line("m8b", 256, "chr2", 100, 3, "10M", 10, "CTCTCTCTCT", 2);
    f << sam_line("m9", 0, "chr1", 160, 3, "10M", 10, "GAGAGAGAGA", 2);                // gene-ambiguous multimapper: not counted
    f << sam_line("m9", 256, "chr2", 550, 3, "10M", 10, "GAGAGAGAGA", 2);
    // kFlank: exon1 then exactly 5 intronic bases ([185, 205) 0-based) -> exonic with kFlank (A), intronic without (U)
    f << sam_line("f5", 0, "chr1", 186, 255, "20M", 20, "TGTGTGTGTG");
    //         6 intronic bases ([186, 206)) -> intronic under both (U)
    f << sam_line("f6", 0, "chr1", 187, 255, "20M", 20, "TATATATATA");
    // kNoSplicedIntron before kFlank: a junction read with 2 intronic bases is dropped (not made exonic), so its N does not reach the
    // molecule: exon-only read + dropped read = A
    f << sam_line("jda", 0, "chr1", 150, 255, "10M", 10, "GTGTGTGTGT");
    f << sam_line("jdb", 0, "chr1", 193, 255, "10M100N5M", 15, "GTGTGTGTGT");
    // CIGAR-N bit: exon-only read + junction read = S
    f << sam_line("sja", 0, "chr1", 330, 255, "20M", 20, "CGCGCGCGCG");
    f << sam_line("sjb", 0, "chr1", 191, 255, "10M100N10M", 20, "CGCGCGCGCG");
    // junction read + intronic read = U; the junction read is exon-only, so the molecule counts in legacy_exon_extra
    f << sam_line("sua", 0, "chr1", 196, 255, "5M100N5M", 10, "GCGCGCGCGC");
    f << sam_line("sub", 0, "chr1", 275, 255, "10M", 10, "GCGCGCGCGC");
    // kUnique before kFlank: a multimapper whose primary is the 5-base flank read is dropped (secondary intergenic): nothing counted
    f << sam_line("muf", 0, "chr1", 186, 3, "20M", 20, "ATATATATAT", 2);
    f << sam_line("muf", 256, "chr2", 100, 3, "20M", 20, "ATATATATAT", 2);
    // multimapper with an exonic spliced primary (secondary intergenic): counted through the primary, with its N: S
    f << sam_line("mus", 0, "chr1", 196, 3, "5M100N5M", 10, "TCTCTCTCTC", 2);
    f << sam_line("mus", 256, "chr2", 100, 3, "10M", 10, "TCTCTCTCTC", 2);
    // multimapper counted through its secondary (R7; primary spliced and intergenic): no N bit, A
    f << sam_line("mur", 0, "chr2", 100, 3, "5M100N5M", 10, "TTGGTTGGTT", 2);
    f << sam_line("mur", 256, "chr1", 160, 3, "10M", 10, "TTGGTTGGTT", 2);
    f.close();
    const std::string cmd = "samtools sort -o " + dir + "/t.bam " + dir + "/t.sam 2>/dev/null && samtools index " + dir + "/t.bam";
    return std::system(cmd.c_str()) == 0;
}

// the run's outputs as text: accumulators (incl. mixed_exon / mixed_intron), molecule stats, legacy_exon_extra, hits
std::string run_dump(const std::string& dir, uint32_t rules) {
    singlet::PileupConfig cfg;
    cfg.barcode_path = dir + "/barcodes.tsv"; cfg.exon_gtf_path = dir + "/mini.gtf";
    cfg.threads = 1; cfg.umi_len = 10; cfg.umi_dedup_directional = true;
    cfg.count_sj = false; cfg.aim_ancestry = false; cfg.count_vdj = false;
    cfg.intron_rules = rules;
    singlet::PileupEngine e(cfg);
    if (!e.load_references()) return "load_references failed";
    const auto st = e.run(dir + "/t.bam");
    std::ostringstream o;
    auto csc = [&](const char* name, const auto& m) {
        o << name << " indptr"; for (auto v : m.indptr) o << ' ' << v;
        o << "\n" << name << " indices"; for (auto v : m.indices) o << ' ' << v;
        o << "\n" << name << " data"; for (auto v : m.data) o << ' ' << v;
        o << "\n";
    };
    csc("exon", e.exons().to_csc()); csc("intron", e.introns().to_csc()); csc("ambiguous", e.ambiguous().to_csc());
    csc("mixed_exon", e.mixed_exon().to_csc()); csc("mixed_intron", e.mixed_intron().to_csc());   // export.h's pre-USA split
    o << "molecules spliced " << st.molecules_spliced << " unspliced " << st.molecules_unspliced << " ambiguous " << st.molecules_ambiguous
      << " legacy_double " << st.molecules_legacy_double << " dropped " << st.molecules_dropped_multigene_umi << "\n";
    o << "legacy_exon_extra"; for (auto v : e.legacy_exon_extra()) o << ' ' << v; o << "\n";
    o << "hits exon " << st.exon_hits << " intron " << st.intron_hits << "\n";
    return o.str();
}

// "exon indptr 0 3 ..." -> key "exon indptr"; "molecules spliced 3 ..." -> key "molecules"
std::map<std::string, std::string> parse(const std::string& s) {
    std::map<std::string, std::string> m; std::istringstream in(s); std::string line;
    while (std::getline(in, line)) {
        size_t sp = line.find(' ');
        if (sp != std::string::npos) {
            const size_t sp2 = line.find(' ', sp + 1);
            const std::string w2 = line.substr(sp + 1, sp2 == std::string::npos ? std::string::npos : sp2 - sp - 1);
            if (w2 == "indptr" || w2 == "indices" || w2 == "data") sp = sp2;
        }
        m[line.substr(0, sp)] = sp == std::string::npos ? "" : line.substr(sp + 1);
    }
    return m;
}
uint64_t field(const std::string& line, const std::string& key) {
    const size_t p = line.find(key + " "); return p == std::string::npos ? ~0ull : std::strtoull(line.c_str() + p + key.size() + 1, nullptr, 10);
}
uint64_t sum_list(const std::string& s) { std::istringstream in(s); uint64_t t = 0, v; while (in >> v) t += v; return t; }

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "--child") {   // the GPU side: SINGLET_GPU_PILEUP=on is in this process's environment
        const std::string d = run_dump(argv[2], static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 0)));
        std::ofstream(argv[4]) << d;
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
    char tmpl[] = "/tmp/test_gpu_pileup_labels_XXXXXX";
    const char* dp = mkdtemp(tmpl);
    if (!dp) { std::cerr << "mkdtemp failed\n"; return 1; }
    const std::string dir = dp;
    if (!write_inputs(dir)) { std::cout << "SKIP: samtools not found (the fixture is a BAM)\n"; return 77; }
    char self[4096]; const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n <= 0) { std::cerr << "cannot find the test binary\n"; return 1; }
    self[n] = 0;

    std::map<uint32_t, std::map<std::string, std::string>> cpu_by_rules;
    for (uint32_t rules : {397u, 389u}) {
        const std::string tag = "rules " + std::to_string(rules);
        std::cout << "\n=== " << tag << " ===\n";
        const std::string cpu = run_dump(dir, rules);
        const std::string out = dir + "/gpu_" + std::to_string(rules) + ".txt", log = dir + "/gpu_" + std::to_string(rules) + ".log";
        const std::string cmd = "SINGLET_GPU_PILEUP=on '" + std::string(self) + "' --child '" + dir + "' " + std::to_string(rules) + " '" + out + "' 2> '" + log + "'";
        const int rc = std::system(cmd.c_str());
        CHECK(rc == 0, tag + ": GPU child exited 0");
        std::ifstream gi(out); std::stringstream gs; gs << gi.rdbuf(); const std::string gpu = gs.str();
        std::ifstream li(log); std::stringstream ls; ls << li.rdbuf(); const std::string lg = ls.str();
        CHECK(lg.find("[gpu_pileup] on:") != std::string::npos, tag + ": the GPU pileup ran ([gpu_pileup] on:)");
        CHECK(lg.find("[gpu_pileup] failed") == std::string::npos && lg.find("not run") == std::string::npos, tag + ": no GPU failure / decline");
        const auto C = parse(cpu), G = parse(gpu);
        cpu_by_rules[rules] = C;
        for (const char* k : {"exon indptr", "exon indices", "exon data", "intron indptr", "intron indices", "intron data",
                              "ambiguous indptr", "ambiguous indices", "ambiguous data",
                              "mixed_exon indptr", "mixed_exon indices", "mixed_exon data",
                              "mixed_intron indptr", "mixed_intron indices", "mixed_intron data",
                              "molecules", "legacy_exon_extra", "hits"}) {
            const bool same = C.count(k) && G.count(k) && C.at(k) == G.at(k);
            CHECK(same, tag + ": " + k + " GPU == CPU");
            if (!same) std::cout << "    CPU: " << (C.count(k) ? C.at(k) : "<none>") << "\n    GPU: " << (G.count(k) ? G.at(k) : "<none>") << "\n";
        }
        // the fixture exercises what it should (CPU side; the GPU equals it above)
        const std::string& m = C.at("molecules");
        CHECK(field(m, "spliced") == 3, tag + ": 3 spliced molecules (m5, the junction + exon molecule, the spliced multimapper)");
        CHECK(sum_list(C.at("legacy_exon_extra")) == 3, tag + ": legacy_exon_extra = 3 (m3, merged m7, junction + intron)");
        CHECK(sum_list(C.at("mixed_exon data")) == 3 && sum_list(C.at("mixed_intron data")) == 3,
              tag + ": mixed_exon / mixed_intron hold the same 3 molecules (the pre-USA split's exon + 3 / intron - 3)");
    }
    {   // kFlank moves exactly one molecule (f5) from the intron to the exon matrix; the 6-base read and the dropped multimapper do not move
        const auto& a = cpu_by_rules[397]; const auto& b = cpu_by_rules[389];
        CHECK(sum_list(b.at("intron data")) == sum_list(a.at("intron data")) + 1, "kFlank: intron total 389 = 397 + 1");
        CHECK(sum_list(a.at("exon data")) == sum_list(b.at("exon data")) + 1, "kFlank: exon total 397 = 389 + 1");
        CHECK(field(a.at("molecules"), "ambiguous") == field(b.at("molecules"), "ambiguous") + 1, "kFlank: one more ambiguous molecule with 397");
    }
    std::system(("rm -rf '" + dir + "'").c_str());
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
#endif
}
