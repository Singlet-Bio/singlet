// SPDX-License-Identifier: MIT
// tests/cpp/test_multimap_record.cpp
//
// The multi-mapping record (include/singlet/pileup/multimap_record.h): which reads the engine collects as two-gene reads, and the
// classes x barcodes matrix built from them. Pins the serial run() and run_parallel() (1, 2 and 3 workers) to the same record.
//   genes (all +): A chr1 exon 101-300, B chr1 exon 201-400 (overlaps A), C chr2 exon 101-200, D chr3 exon 101-200, E chr3 exon 151-250,
//   F chr3 exon 171-270 (D, E and F overlap at 171-200), G chr4 exon 101-200, H chr5 exons 101-200 and 801-900, I chr5 exon 401-500
//   (inside H's intron: STARsolo GeneFull's gene-body set {H, I}).
//   u1, u2  NH 1 on A and B's shared exon, UMI U1          two-gene unique primary: one (A, B) molecule from two reads
//   u3      NH 1 where D, E and F overlap                  three genes: not in the record
//   m1      NH 2, C + G, UMI U2                            (C, G)
//   m2      NH 2, C + A, UMI U2                            with m1: the genes in every read of U2 are {C}: one C molecule
//   m3      NH 3, C + G + A                                three genes: not in the record
//   m4      NH 2, C + a secondary on A and B               a multi-gene secondary: not in the record
//   m5      NH 2, primary on A and B + secondary G         three genes: not in the record
//   m6      NH 2, G + C, UMI U6                            (C, G)
//   m7      NH 2, C + C (two loci of one gene)             gene-unique: counted for C, not in the record
//   u4      NH 1 on I's exon inside H's intron              gene-body set {H, I} (intron_rule::kGeneFullSet): one (H, I) molecule
//   m8      NH 2, the same place + an intergenic secondary  (H, I)
//   J chr6 exons 101-200 and 1601-1700 (a host gene, intron 201-1600), K chr7 exon 101-200 (a retrocopy's parent):
//   m9      NH 2, K exon + J intron, UMI GAGA..            7053 (kExonPrecedence): counted for K, never in the record; 2957: (J, K)
//   m10     NH 2, J intron + K exon (primary intronic)     the same, judged on the exonic secondary
// The record never holds a molecule the gene counts hold, so include_multimappers=True counts each molecule once.

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "singlet/pileup/multimap_record.h"
#include "singlet/pileup/pileup_engine.h"
#include "test_harness.h"

namespace {

const char* kBC = "AAACCTGAGAAACGAG-1";

std::string sam_line(const std::string& q, int flag, const std::string& chr, int pos, int mapq, int read_len, const std::string& umi,
                     int nh = 1) {
    std::string seq(static_cast<size_t>(read_len), 'A');
    return q + "\t" + std::to_string(flag) + "\t" + chr + "\t" + std::to_string(pos) + "\t" + std::to_string(mapq) + "\t" +
           std::to_string(read_len) + "M\t*\t0\t0\t" + seq + "\t*\tCB:Z:" + kBC + "\tUB:Z:" + umi + "\tNH:i:" + std::to_string(nh) + "\n";
}

bool write_inputs(const std::string& dir) {
    std::ofstream g(dir + "/mini.gtf");
    auto gl = [&](const char* chr, int s, int e, const char* id) {
        for (const char* feat : {"gene", "exon"})
            g << chr << "\t.\t" << feat << "\t" << s << "\t" << e << "\t.\t+\t.\tgene_id \"" << id << "\"; gene_name \"" << id
              << "\"; transcript_id \"" << id << "-T\";\n";
    };
    gl("chr1", 101, 300, "A"); gl("chr1", 201, 400, "B"); gl("chr2", 101, 200, "C");
    gl("chr3", 101, 200, "D"); gl("chr3", 151, 250, "E"); gl("chr3", 171, 270, "F"); gl("chr4", 101, 200, "G");
    const char* ha = "gene_id \"H\"; gene_name \"H\"; transcript_id \"H-T\";\n";
    g << "chr5\t.\tgene\t101\t900\t.\t+\t.\t" << ha << "chr5\t.\texon\t101\t200\t.\t+\t.\t" << ha << "chr5\t.\texon\t801\t900\t.\t+\t.\t" << ha;
    // J chr6: exons 101-200 and 1601-1700, intron 201-1600 (the host); K chr7 exon 101-200 (the parent of a retrocopy in J's intron)
    const char* ja = "gene_id \"J\"; gene_name \"J\"; transcript_id \"J-T\";\n";
    g << "chr6\t.\tgene\t101\t1700\t.\t+\t.\t" << ja << "chr6\t.\texon\t101\t200\t.\t+\t.\t" << ja << "chr6\t.\texon\t1601\t1700\t.\t+\t.\t" << ja;
    gl("chr7", 101, 200, "K");
    gl("chr5", 401, 500, "I");
    g.close();
    std::ofstream b(dir + "/barcodes.tsv");
    b << kBC << "\n";
    b.close();
    std::ofstream f(dir + "/t.sam");
    f << "@HD\tVN:1.6\tSO:unsorted\n";
    for (const char* c : {"chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7"}) f << "@SQ\tSN:" << c << "\tLN:2000\n";
    f << sam_line("u1", 0, "chr1", 210, 255, 20, "AAAAAAAAAA");
    f << sam_line("u2", 0, "chr1", 220, 255, 20, "AAAAAAAAAA");
    f << sam_line("u3", 0, "chr3", 175, 255, 20, "CCCCCCCCCC");
    f << sam_line("m1", 0, "chr2", 120, 3, 20, "GGGGGGGGGG", 2) << sam_line("m1", 256, "chr4", 120, 3, 20, "GGGGGGGGGG", 2);
    f << sam_line("m2", 0, "chr2", 130, 3, 20, "GGGGGGGGGG", 2) << sam_line("m2", 256, "chr1", 120, 3, 20, "GGGGGGGGGG", 2);
    f << sam_line("m3", 0, "chr2", 140, 3, 20, "TTTTTTTTTT", 3) << sam_line("m3", 256, "chr4", 140, 3, 20, "TTTTTTTTTT", 3)
      << sam_line("m3", 256, "chr1", 120, 3, 20, "TTTTTTTTTT", 3);
    f << sam_line("m4", 0, "chr2", 150, 3, 20, "ACACACACAC", 2) << sam_line("m4", 256, "chr1", 210, 3, 20, "ACACACACAC", 2);
    f << sam_line("m5", 0, "chr1", 230, 3, 20, "AGAGAGAGAG", 2) << sam_line("m5", 256, "chr4", 150, 3, 20, "AGAGAGAGAG", 2);
    f << sam_line("m6", 0, "chr4", 160, 3, 20, "CACACACACA", 2) << sam_line("m6", 256, "chr2", 160, 3, 20, "CACACACACA", 2);
    f << sam_line("m7", 0, "chr2", 110, 3, 20, "GTGTGTGTGT", 2) << sam_line("m7", 256, "chr2", 170, 3, 20, "GTGTGTGTGT", 2);
    f << sam_line("u4", 0, "chr5", 420, 255, 20, "TGTGTGTGTG");
    f << sam_line("m8", 0, "chr5", 430, 3, 20, "CTCTCTCTCT", 2) << sam_line("m8", 256, "chr4", 1500, 3, 20, "CTCTCTCTCT", 2);
    f << sam_line("m9", 0, "chr7", 120, 3, 20, "GAGAGAGAGA", 2) << sam_line("m9", 256, "chr6", 800, 3, 20, "GAGAGAGAGA", 2);
    f << sam_line("m10", 0, "chr6", 900, 3, 20, "CGCGCGCGCG", 2) << sam_line("m10", 256, "chr7", 140, 3, 20, "CGCGCGCGCG", 2);
    f.close();
    std::string cmd = "samtools sort -o " + dir + "/t.bam " + dir + "/t.sam 2>/dev/null && samtools index " + dir + "/t.bam";
    return std::system(cmd.c_str()) == 0;
}

typedef std::tuple<int32_t, std::string, std::string, std::string> Key;   // cell, gene a, gene b, UMI

std::vector<Key> reads_of(const singlet::PileupEngine& e) {
    const auto& ids = e.gene_model().gene_ids();
    std::vector<Key> v;
    for (const auto& a : e.ambig_reads())
        v.emplace_back(a.bc_idx, ids[std::min(a.gene_a, a.gene_b)], ids[std::max(a.gene_a, a.gene_b)], std::string(a.umi, a.umi_len));
    std::sort(v.begin(), v.end());
    return v;
}

std::map<std::string, uint32_t> record_of(const singlet::PileupEngine& e) {
    const auto r = singlet::multimap::build(e.ambig_reads(), static_cast<uint32_t>(e.barcodes().size()), e.gene_ids(), e.gene_names());
    std::map<std::string, uint32_t> m;
    for (uint32_t c = 0; c < r.ncols; ++c)
        for (int32_t k = r.indptr[c]; k < r.indptr[c + 1]; ++k) m[r.names[static_cast<size_t>(r.indices[k])]] += r.data[k];
    return m;
}

std::vector<Key> g_ref;
std::map<std::string, uint32_t> g_rec;

void run_case(const std::string& dir, bool parallel, int workers, uint32_t rules) {
    const bool on = rules & singlet::intron_rule::kExonPrecedence;
    std::string tag = (parallel ? "run_parallel(" + std::to_string(workers) + ")" : std::string("run")) + " rules " + std::to_string(rules);
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
    const auto v = reads_of(e);
    const auto rec = record_of(e);
    uint32_t k_count = 0;   // K's exon molecules
    {
        const auto& gm = e.gene_model();
        const auto ex = e.exons().to_csc();
        for (size_t i = 0; i < ex.data.size(); ++i)
            if (gm.gene_ids()[gm.exon_to_gene(static_cast<uint32_t>(ex.indices[i]))] == "K") k_count += ex.data[i];
    }
    CHECK(k_count == (on ? 2u : 0u), tag + ": K counts m9 and m10 only under kExonPrecedence");
    if (!parallel) {
        std::vector<Key> want = {Key(0, "A", "B", "AAAAAAAAAA"), Key(0, "A", "B", "AAAAAAAAAA"), Key(0, "A", "C", "GGGGGGGGGG"),
                                 Key(0, "C", "G", "CACACACACA"), Key(0, "C", "G", "GGGGGGGGGG"), Key(0, "H", "I", "CTCTCTCTCT"),
                                 Key(0, "H", "I", "TGTGTGTGTG")};
        if (!on) { want.emplace_back(0, "J", "K", "CGCGCGCGCG"); want.emplace_back(0, "J", "K", "GAGAGAGAGA"); std::sort(want.begin(), want.end()); }
        CHECK(v == want, tag + ": the two-gene reads (u1, u2, m1, m2, m6, m8, u4; m9 and m10 only without kExonPrecedence)");
        CHECK(rec.size() == (on ? 4u : 5u), tag + ": gene sets");
        CHECK(on ? rec.count("J_J|K_K") == 0 : (rec.count("J_J|K_K") && rec.at("J_J|K_K") == 2),
              tag + ": m9 and m10 are in the record only when the counts leave them out");
        CHECK(rec.count("H_H|I_I") && rec.at("H_H|I_I") == 2, tag + ": u4 and m8 are two (H, I) molecules (gene-body sets)");
        CHECK(rec.count("A_A|B_B") && rec.at("A_A|B_B") == 1, tag + ": u1 and u2 are one (A, B) molecule");
        CHECK(rec.count("C_C|G_G") && rec.at("C_C|G_G") == 1, tag + ": m6 is one (C, G) molecule");
        CHECK(rec.count("C_C") && rec.at("C_C") == 1, tag + ": U2's reads share only C");
        g_ref = v; g_rec = rec;
    } else {   // after run() with the same rules
        CHECK(v == g_ref, tag + ": the same two-gene reads as run()");
        CHECK(rec == g_rec, tag + ": the same record as run()");
    }
}

void unit_build() {
    std::cout << "\n=== build ===\n";
    using singlet::em_rescue::AmbigRead;
    auto mk = [](int32_t bc, const char* umi, uint32_t a, uint32_t b) {
        AmbigRead r; r.bc_idx = bc; r.umi_len = 4; std::memcpy(r.umi, umi, 4); r.gene_a = a; r.gene_b = b; return r;
    };
    // cell 0: UMI ACGT (1,2) twice, UMI TTTT (2,1) -> two (1,2) molecules; UMI GGGG (0,1) + (1,2) -> gene 1; UMI CCCC (0,1) + (2,3) -> none
    // cell 2: UMI ACGT (1,2) -> one; reads with a bad cell or gene are ignored
    std::vector<AmbigRead> in = {mk(0, "ACGT", 1, 2), mk(0, "ACGT", 2, 1), mk(0, "TTTT", 2, 1), mk(0, "GGGG", 0, 1), mk(0, "GGGG", 1, 2),
                                 mk(0, "CCCC", 0, 1), mk(0, "CCCC", 2, 3), mk(2, "ACGT", 1, 2), mk(5, "ACGT", 1, 2), mk(1, "ACGT", 1, 9)};
    const std::vector<std::string> ids = {"g0", "g1", "g2", "g3"}, nm = {"n0", "n1", "n2", "n3"};
    const auto r = singlet::multimap::build(in, 3, ids, nm);
    CHECK(r.nrows == 2 && r.names[0] == "g1_n1|g2_n2" && r.names[1] == "g1_n1", "build: classes in gene order, a single gene after its pairs");
    CHECK(r.ncols == 3 && r.indptr == std::vector<int32_t>({0, 2, 2, 3}), "build: column pointers");
    CHECK(r.indices == std::vector<int32_t>({0, 1, 0}) && r.data == std::vector<uint32_t>({2, 1, 1}), "build: molecules per class and cell");
    CHECK(r.molecules == 4 && r.dropped_umis == 1 && r.reads == 8, "build: molecule, dropped-UMI and read counts");
    std::vector<AmbigRead> rev(in.rbegin(), in.rend());
    const auto r2 = singlet::multimap::build(rev, 3, ids, nm);
    CHECK(r2.indptr == r.indptr && r2.indices == r.indices && r2.data == r.data && r2.names == r.names, "build: input order is irrelevant");
}

}  // namespace

int main() {
    unit_build();
    std::string dir = "/tmp/test_multimap_record_" + std::to_string(getpid());
    std::system(("mkdir -p " + dir).c_str());
    if (!write_inputs(dir)) {
        std::cout << "SKIP: samtools not available\n";
    } else {
        for (uint32_t rules : {7053u, 2957u}) {   // the default (kExonPrecedence on) and the rules before it
            run_case(dir, false, 1, rules);
            for (int w : {1, 2, 3}) run_case(dir, true, w, rules);
        }
    }
    std::system(("rm -rf " + dir).c_str());
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
