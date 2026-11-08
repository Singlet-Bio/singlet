// SPDX-License-Identifier: MIT
// tests/cpp/test_exon_precedence.cpp
//
// intron_rule::kExonPrecedence: a multimapper with an exonic alignment is judged on its exonic alignments only, as STARsolo Gene, Space Ranger
// and Cell Ranger count it. Retrocopies of highly expressed genes sit in the introns of host genes, so without the rule a read with one exonic
// alignment in gene A and others in an intron of gene B became gene-ambiguous (docs/research-beta/spatial-exon-multimapper.md).
//   A chr1 101-900 (exons 101-200, 801-900); B chr2 and C chr3 101-1000 (exons 101-200, 901-1000: introns 201-900); D chr4 (exon 101-300);
//   NUMT chr5 (exon 101-400); MT chrM (exon 101-400). All on +. Rules 2957 (default before the rule, kGeneFullSet and kMtPrefer on) and 7053.
//   r1 NH 2  primary A exon, secondary B intron              2957: ambiguous (A, B)   7053: A
//   r2 NH 2  primary B intron, secondary A exon              2957: ambiguous (A, B)   7053: A (the exonic secondary, R7)
//   r3 NH 2  primary A exon, secondary D exon                ambiguous (A, D) under both: the EM pair is the two exonic genes
//   r4 NH 2  primary B intron, secondary C intron            ambiguous (B, C) under both: no exonic alignment, judged as before
//   r5 NH 3  primary B intron, secondaries A exon 2, A exon 1  2957: ambiguous (A, B)  7053: A at its first exon (smallest secondary exon)
//   r6 NH 3  primary MT, secondaries NUMT exon, B intron      MT under both (kMtPrefer tie on {MT, NUMT})
//   r7 NH 2  primary MT, secondary C intron                   MT under both; a NUMT tie (mt_numt_rescued) only under 2957
// The result must not depend on the record order (serial run() over unsorted BAMs in several orders) or the worker split (run_parallel).
//
// Joint (barnyard) reference, contigs <genome>___<contig>: the rule decides only when all alignments lie in one genome.
//   H GRCh38___chr1 101-1000 (exons 101-200, 901-1000); R GRCm39___1 (exon 101-300, an 18S-like pseudogene); M GRCm39___2 as H.
//   b1 NH 2  primary H intron, secondary R exon     ambiguous (H, R) under both: the alignments span two genomes
//   b2 NH 2  primary M intron, secondary R exon     2957: ambiguous (M, R)   7053: R (one genome)
//   b3 NH 1  H exon                                 H under both
//
// intron_rule::kChrMPrecedence (8192), same single-genome reference; chrM 1001-2000 lies outside every gene, like the 12S / 16S rRNA that
// Cell Ranger's 2024-A references leave out. Rules 6541 and 7053, each without and with 8192.
//   m1 NH 2  primary chrM 1500, secondary NUMT exon            NUMT; with 8192 nothing
//   m2 NH 2  primary NUMT exon, secondary chrM 1500            NUMT; with 8192 nothing
//   m3 NH 2  primary MT exon, secondary B intron               MT under all four
//   m4 NH 2  primary chrM 1500, secondary A exon               A; with 8192 nothing
//   m5 NH 3  primary chrM 1500, secondaries A exon, D exon     G-EM pair (A, D); with 8192 neither counted nor an EM pair
//   m6 NH 3  primary NUMT exon, secondaries MT exon, chrM 1600 MT under all four (kMtPrefer)

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/pileup_engine.h"
#include "test_harness.h"

namespace {

const char* kBC = "AAACCTGAGAAACGAG-1";

std::string sam_line(const std::string& q, int flag, const std::string& chr, int pos, const std::string& umi, int nh) {
    const std::string seq(20, 'A');
    return q + "\t" + std::to_string(flag) + "\t" + chr + "\t" + std::to_string(pos) + "\t3\t20M\t*\t0\t0\t" + seq + "\t*\tCB:Z:" + kBC +
           "\tUB:Z:" + umi + "\tNH:i:" + std::to_string(nh) + "\n";
}

const char* kHeader = "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr1\tLN:2000\n@SQ\tSN:chr2\tLN:2000\n@SQ\tSN:chr3\tLN:2000\n"
                      "@SQ\tSN:chr4\tLN:2000\n@SQ\tSN:chr5\tLN:2000\n@SQ\tSN:chrM\tLN:2000\n";

// one entry per read: its records, primary first
std::vector<std::vector<std::string>> reads() {
    return {
        {sam_line("r1", 0, "chr1", 150, "AAAAAAAAAA", 2), sam_line("r1", 256, "chr2", 500, "AAAAAAAAAA", 2)},
        {sam_line("r2", 0, "chr2", 520, "CCCCCCCCCC", 2), sam_line("r2", 256, "chr1", 160, "CCCCCCCCCC", 2)},
        {sam_line("r3", 0, "chr1", 170, "GGGGGGGGGG", 2), sam_line("r3", 256, "chr4", 150, "GGGGGGGGGG", 2)},
        {sam_line("r4", 0, "chr2", 540, "TTTTTTTTTT", 2), sam_line("r4", 256, "chr3", 500, "TTTTTTTTTT", 2)},
        {sam_line("r5", 0, "chr2", 560, "ACACACACAC", 3), sam_line("r5", 256, "chr1", 850, "ACACACACAC", 3),
         sam_line("r5", 256, "chr1", 120, "ACACACACAC", 3)},
        {sam_line("r6", 0, "chrM", 150, "AGAGAGAGAG", 3), sam_line("r6", 256, "chr5", 150, "AGAGAGAGAG", 3),
         sam_line("r6", 256, "chr2", 600, "AGAGAGAGAG", 3)},
        {sam_line("r7", 0, "chrM", 250, "CTCTCTCTCT", 2), sam_line("r7", 256, "chr3", 600, "CTCTCTCTCT", 2)},
    };
}

bool write_reference(const std::string& dir) {
    std::ofstream g(dir + "/mini.gtf");
    auto gl = [&](const char* chr, const char* feat, int s, int e, const char* id) {
        g << chr << "\t.\t" << feat << "\t" << s << "\t" << e << "\t.\t+\t.\tgene_id \"" << id << "\"; gene_name \"" << id
          << "\"; transcript_id \"" << id << "-T\";\n";
    };
    gl("chr1", "gene", 101, 900, "A"); gl("chr1", "exon", 101, 200, "A"); gl("chr1", "exon", 801, 900, "A");
    gl("chr2", "gene", 101, 1000, "B"); gl("chr2", "exon", 101, 200, "B"); gl("chr2", "exon", 901, 1000, "B");
    gl("chr3", "gene", 101, 1000, "C"); gl("chr3", "exon", 101, 200, "C"); gl("chr3", "exon", 901, 1000, "C");
    gl("chr4", "gene", 101, 300, "D"); gl("chr4", "exon", 101, 300, "D");
    gl("chr5", "gene", 101, 400, "NUMT"); gl("chr5", "exon", 101, 400, "NUMT");
    gl("chrM", "gene", 101, 400, "MT"); gl("chrM", "exon", 101, 400, "MT");
    g.close();
    std::ofstream b(dir + "/barcodes.tsv");
    b << kBC << "\n";
    return b.good();
}

// an unsorted BAM with the records in this order
bool write_bam(const std::string& dir, const std::string& name, const std::vector<std::string>& recs) {
    std::ofstream f(dir + "/" + name + ".sam");
    f << kHeader;
    for (const auto& r : recs) f << r;
    f.close();
    const std::string cmd = "samtools view -b -o " + dir + "/" + name + ".bam " + dir + "/" + name + ".sam 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
}

struct Result {
    std::map<std::string, int> by_gene;   // exon matrix total per gene
    std::map<uint32_t, int> by_exon;      // exon matrix total per exon
    std::string exon;                     // the exon matrix (CSC) as text
    std::vector<std::string> em;          // G-EM candidate pairs "UMI gene_a gene_b", sorted
    uint64_t rescued = 0;                 // mt_numt_rescued
    std::string text() const {
        std::ostringstream o;
        o << exon << " | rescued " << rescued << " | EM";
        for (const auto& s : em) o << " [" << s << "]";
        return o.str();
    }
};

Result run_case(const std::string& dir, const std::string& bam, uint32_t rules, int workers) {
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
    Result r;
    singlet::PileupEngine e(cfg);
    if (!e.load_references()) { CHECK(false, "load_references"); return r; }
    const auto st = workers > 0 ? e.run_parallel(bam, workers) : e.run(bam);
    r.rescued = st.mt_numt_rescued;
    const auto& gm = e.gene_model();
    auto ex = e.exons().to_csc();
    std::ostringstream o;
    o << "indptr";
    for (auto v : ex.indptr) o << ' ' << v;
    o << " indices";
    for (auto v : ex.indices) o << ' ' << v;
    o << " data";
    for (auto v : ex.data) o << ' ' << v;
    r.exon = o.str();
    for (size_t k = 0; k < ex.data.size(); ++k) {
        r.by_gene[gm.gene_ids()[gm.exon_to_gene(static_cast<uint32_t>(ex.indices[k]))]] += ex.data[k];
        r.by_exon[static_cast<uint32_t>(ex.indices[k])] += ex.data[k];
    }
    for (const auto& a : e.ambig_reads()) {
        std::string ga = gm.gene_ids()[a.gene_a], gb = gm.gene_ids()[a.gene_b];
        if (gb < ga) std::swap(ga, gb);
        r.em.push_back(std::string(a.umi, a.umi_len) + " " + ga + " " + gb);
    }
    std::sort(r.em.begin(), r.em.end());
    return r;
}

// A's exons: the one with the smallest index (R7's pick among secondaries) and the one at 101-200 (r1's primary, r2's secondary)
void exons_of_a(const std::string& dir, uint32_t& smallest, uint32_t& at101) {
    smallest = at101 = UINT32_MAX;
    singlet::PileupConfig cfg;
    cfg.barcode_path = dir + "/barcodes.tsv";
    cfg.exon_gtf_path = dir + "/mini.gtf";
    cfg.aim_ancestry = false;
    cfg.count_vdj = false;
    singlet::PileupEngine e(cfg);
    if (!e.load_references()) return;
    const auto& gm = e.gene_model();
    for (uint32_t x = 0; x < gm.n_exons(); ++x) {
        if (gm.gene_ids()[gm.exon_to_gene(x)] != "A") continue;
        if (smallest == UINT32_MAX) smallest = x;
        if (gm.exon_start(x) < 200) at101 = x;
    }
}

void check_expected(const Result& r, uint32_t rules, const std::string& tag) {
    const bool on = rules & singlet::intron_rule::kExonPrecedence;
    auto g = r.by_gene;
    CHECK(g["A"] == (on ? 3 : 0), tag + ": A gets r1, r2, r5 only with kExonPrecedence");
    CHECK(g["B"] == 0 && g["C"] == 0 && g["D"] == 0 && g["NUMT"] == 0, tag + ": B, C, D, NUMT get nothing");
    CHECK(g["MT"] == 2, tag + ": MT gets r6 and r7");
    CHECK(r.rescued == static_cast<uint64_t>(on ? 1 : 2), tag + ": the NUMT tie is decided on the gene set the read is judged on");
    std::vector<std::string> em_on = {"GGGGGGGGGG A D", "TTTTTTTTTT B C"};
    std::vector<std::string> em_off = {"AAAAAAAAAA A B", "ACACACACAC A B", "CCCCCCCCCC A B", "GGGGGGGGGG A D", "TTTTTTTTTT B C"};
    CHECK(r.em == (on ? em_on : em_off), tag + ": G-EM pairs");
}

const char* kJointHeader = "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:GRCh38___chr1\tLN:2000\n@SQ\tSN:GRCm39___1\tLN:2000\n"
                           "@SQ\tSN:GRCm39___2\tLN:2000\n";

// a unique read (NH 1) needs a MAPQ above min_mapq, which the multimapper records (MAPQ 3, as STAR writes them) do not
std::string unique_line(const std::string& q, const std::string& chr, int pos, const std::string& umi) {
    std::string s = sam_line(q, 0, chr, pos, umi, 1);
    const std::string from = "\t3\t20M\t", to = "\t255\t20M\t";
    s.replace(s.find(from), from.size(), to);
    return s;
}

std::vector<std::string> joint_records() {
    return {sam_line("b1", 0, "GRCh38___chr1", 500, "AAAAAAAAAA", 2), sam_line("b1", 256, "GRCm39___1", 150, "AAAAAAAAAA", 2),
            sam_line("b2", 0, "GRCm39___2", 500, "CCCCCCCCCC", 2), sam_line("b2", 256, "GRCm39___1", 160, "CCCCCCCCCC", 2),
            unique_line("b3", "GRCh38___chr1", 150, "GGGGGGGGGG")};
}

bool write_joint(const std::string& dir, const std::string& name, const std::vector<std::string>& recs) {
    std::ofstream g(dir + "/mini.gtf");
    auto gl = [&](const char* chr, const char* feat, int s, int e, const char* id) {
        g << chr << "\t.\t" << feat << "\t" << s << "\t" << e << "\t.\t+\t.\tgene_id \"" << id << "\"; gene_name \"" << id
          << "\"; transcript_id \"" << id << "-T\";\n";
    };
    gl("GRCh38___chr1", "gene", 101, 1000, "H"); gl("GRCh38___chr1", "exon", 101, 200, "H"); gl("GRCh38___chr1", "exon", 901, 1000, "H");
    gl("GRCm39___1", "gene", 101, 300, "R"); gl("GRCm39___1", "exon", 101, 300, "R");
    gl("GRCm39___2", "gene", 101, 1000, "M"); gl("GRCm39___2", "exon", 101, 200, "M"); gl("GRCm39___2", "exon", 901, 1000, "M");
    g.close();
    std::ofstream b(dir + "/barcodes.tsv");
    b << kBC << "\n";
    b.close();
    std::ofstream f(dir + "/" + name + ".sam");
    f << kJointHeader;
    for (const auto& r : recs) f << r;
    f.close();
    return std::system(("samtools view -b -o " + dir + "/" + name + ".bam " + dir + "/" + name + ".sam 2>/dev/null").c_str()) == 0;
}

void check_joint(const std::string& dir) {
    std::system(("mkdir -p " + dir).c_str());
    const auto recs = joint_records();
    if (!write_joint(dir, "j", recs) ||
        std::system(("samtools sort -o " + dir + "/js.bam " + dir + "/j.bam 2>/dev/null && samtools index " + dir + "/js.bam").c_str()) != 0) {
        CHECK(false, "joint reference BAM");
        return;
    }
    for (uint32_t rules : {2957u, 7053u}) {
        const std::string rt = "joint rules " + std::to_string(rules);
        const bool on = rules & singlet::intron_rule::kExonPrecedence;
        const Result ref = run_case(dir, dir + "/js.bam", rules, 0);
        auto g = ref.by_gene;
        CHECK(g["H"] == 1, rt + ": H gets b3");
        CHECK(g["R"] == (on ? 1 : 0), rt + ": R gets b2 only with kExonPrecedence, never the cross-genome b1");
        CHECK(g["M"] == 0, rt + ": M gets nothing");
        const std::vector<std::string> em_on = {"AAAAAAAAAA H R"}, em_off = {"AAAAAAAAAA H R", "CCCCCCCCCC M R"};
        CHECK(ref.em == (on ? em_on : em_off), rt + ": G-EM pairs (b1 stays ambiguous)");
        if (ref.em != (on ? em_on : em_off)) std::cout << "    " << ref.text() << "\n";
        const std::vector<std::string> rev(recs.rbegin(), recs.rend());
        if (write_joint(dir, "jr", rev)) CHECK(run_case(dir, dir + "/jr.bam", rules, 0).text() == ref.text(), rt + ": run(reversed) == run(sorted)");
        for (int w : {1, 2, 3}) CHECK(run_case(dir, dir + "/js.bam", rules, w).text() == ref.text(), rt + ": run_parallel(" + std::to_string(w) + ")");
    }
    std::system(("rm -rf " + dir).c_str());
}

std::vector<std::string> chrm_records() {
    return {sam_line("m1", 0, "chrM", 1500, "AAAAAAAAAA", 2), sam_line("m1", 256, "chr5", 150, "AAAAAAAAAA", 2),
            sam_line("m2", 0, "chr5", 200, "CCCCCCCCCC", 2), sam_line("m2", 256, "chrM", 1500, "CCCCCCCCCC", 2),
            sam_line("m3", 0, "chrM", 150, "GGGGGGGGGG", 2), sam_line("m3", 256, "chr2", 500, "GGGGGGGGGG", 2),
            sam_line("m4", 0, "chrM", 1500, "TTTTTTTTTT", 2), sam_line("m4", 256, "chr1", 150, "TTTTTTTTTT", 2),
            sam_line("m5", 0, "chrM", 1500, "ACACACACAC", 3), sam_line("m5", 256, "chr1", 160, "ACACACACAC", 3),
            sam_line("m5", 256, "chr4", 150, "ACACACACAC", 3),
            sam_line("m6", 0, "chr5", 250, "AGAGAGAGAG", 3), sam_line("m6", 256, "chrM", 250, "AGAGAGAGAG", 3),
            sam_line("m6", 256, "chrM", 1600, "AGAGAGAGAG", 3)};
}

void check_chrm(const std::string& dir) {
    const auto recs = chrm_records();
    if (!write_bam(dir, "m", recs) ||
        std::system(("samtools sort -o " + dir + "/ms.bam " + dir + "/m.bam 2>/dev/null && samtools index " + dir + "/ms.bam").c_str()) != 0) {
        CHECK(false, "chrM BAM");
        return;
    }
    for (uint32_t base : {6541u, 7053u}) {
        for (bool on : {false, true}) {
            const uint32_t rules = base | (on ? singlet::intron_rule::kChrMPrecedence : 0u);
            const std::string rt = "chrM rules " + std::to_string(rules);
            const Result ref = run_case(dir, dir + "/ms.bam", rules, 0);
            auto g = ref.by_gene;
            const bool ok = g["NUMT"] == (on ? 0 : 2) && g["A"] == (on ? 0 : 1) && g["MT"] == 2 && g["B"] == 0 && g["D"] == 0;
            CHECK(g["NUMT"] == (on ? 0 : 2), rt + ": NUMT gets m1 and m2 only without kChrMPrecedence");
            CHECK(g["A"] == (on ? 0 : 1), rt + ": A gets m4 only without kChrMPrecedence");
            CHECK(g["MT"] == 2, rt + ": MT gets m3 and m6");
            CHECK(g["B"] == 0 && g["D"] == 0, rt + ": B and D get nothing");
            const std::vector<std::string> em_off = {"ACACACACAC A D"}, em_on = {};
            CHECK(ref.em == (on ? em_on : em_off), rt + ": G-EM pairs");
            if (!ok || ref.em != (on ? em_on : em_off)) std::cout << "    " << ref.text() << "\n";
            const std::vector<std::string> rev(recs.rbegin(), recs.rend());
            if (write_bam(dir, "mr", rev)) CHECK(run_case(dir, dir + "/mr.bam", rules, 0).text() == ref.text(), rt + ": run(reversed) == run(sorted)");
            for (int w : {1, 2, 3}) CHECK(run_case(dir, dir + "/ms.bam", rules, w).text() == ref.text(), rt + ": run_parallel(" + std::to_string(w) + ")");
        }
    }
}

// intron_rule::kTxCompat (16384), STARsolo Gene's transcript concordance. chr1, all on +: P (one transcript, exon 101-300) under the readthrough
// R (one transcript, exons 21-80, 101-250, 501-600). Unique reads (MAPQ 255):
//   t1 241-260 (20M)                overlaps exons of P and R; inside P's exon, past R's exon end      6541: dropped  with 16384: P
//   t2 151-170 (20M)                inside the exons of both                                            dropped under both
//   t3 241-250, 501-510 (10M250N10M) R's junction 250 -> 501; P does not contain it                     6541: dropped  with 16384: R
//   t4 241-250, 491-500 (10M240N10M) a junction no transcript has                                       dropped under both
std::string tx_line(const std::string& q, int pos, const std::string& cigar, const std::string& umi) {
    return q + "\t0\tchr1\t" + std::to_string(pos) + "\t255\t" + cigar + "\t*\t0\t0\t" + std::string(20, 'A') + "\t*\tCB:Z:" + kBC + "\tUB:Z:" + umi +
           "\tNH:i:1\n";
}

void check_tx(const std::string& dir) {
    std::system(("mkdir -p " + dir).c_str());
    {
        std::ofstream g(dir + "/mini.gtf");
        auto gl = [&](const char* feat, int s, int e, const char* gid, const char* tid) {
            g << "chr1\t.\t" << feat << "\t" << s << "\t" << e << "\t.\t+\t.\tgene_id \"" << gid << "\"; gene_name \"" << gid
              << "\"; transcript_id \"" << tid << "\";\n";
        };
        gl("gene", 101, 300, "P", "P1"); gl("exon", 101, 300, "P", "P1");
        gl("gene", 21, 600, "R", "R1"); gl("exon", 21, 80, "R", "R1"); gl("exon", 101, 250, "R", "R1"); gl("exon", 501, 600, "R", "R1");
        std::ofstream b(dir + "/barcodes.tsv");
        b << kBC << "\n";
        std::ofstream f(dir + "/t.sam");
        f << kHeader << tx_line("t1", 241, "20M", "AAAAAAAAAA") << tx_line("t2", 151, "20M", "CCCCCCCCCC")
          << tx_line("t3", 241, "10M250N10M", "GGGGGGGGGG") << tx_line("t4", 241, "10M240N10M", "TTTTTTTTTT");
    }
    if (std::system(("samtools sort -o " + dir + "/ts.bam " + dir + "/t.sam 2>/dev/null && samtools index " + dir + "/ts.bam").c_str()) != 0) {
        CHECK(false, "kTxCompat BAM");
        return;
    }
    for (bool on : {false, true}) {
        const uint32_t rules = 14733u | (on ? singlet::intron_rule::kTxCompat : 0u);
        const std::string rt = "kTxCompat rules " + std::to_string(rules);
        const Result ref = run_case(dir, dir + "/ts.bam", rules, 0);
        auto g = ref.by_gene;
        CHECK(g["P"] == (on ? 1 : 0), rt + ": P gets t1 only with kTxCompat");
        CHECK(g["R"] == (on ? 1 : 0), rt + ": R gets t3 only with kTxCompat");
        if (g["P"] != (on ? 1 : 0) || g["R"] != (on ? 1 : 0)) std::cout << "    " << ref.text() << "\n";
        for (int w : {1, 2, 3}) CHECK(run_case(dir, dir + "/ts.bam", rules, w).text() == ref.text(), rt + ": run_parallel(" + std::to_string(w) + ")");
    }
    std::system(("rm -rf " + dir).c_str());
}

}  // namespace

int main() {
    const std::string dir = "/tmp/test_exon_precedence_" + std::to_string(getpid());
    std::system(("mkdir -p " + dir).c_str());
    const auto rs = reads();
    std::vector<std::string> base;
    for (const auto& r : rs) base.insert(base.end(), r.begin(), r.end());
    bool ok = write_reference(dir) && write_bam(dir, "t", base);
    if (ok) ok = std::system(("samtools sort -o " + dir + "/sorted.bam " + dir + "/t.bam 2>/dev/null && samtools index " + dir + "/sorted.bam").c_str()) == 0;
    if (!ok) {
        std::cout << "SKIP: samtools not available\n";
        std::system(("rm -rf " + dir).c_str());
        return 0;
    }
    // record orders: as written, reversed, each read's secondaries first, and seeded shuffles of all records
    std::vector<std::pair<std::string, std::vector<std::string>>> orders;
    orders.push_back({"as_written", base});
    orders.push_back({"reversed", std::vector<std::string>(base.rbegin(), base.rend())});
    {
        std::vector<std::string> v;
        for (const auto& r : rs) v.insert(v.end(), r.rbegin(), r.rend());
        orders.push_back({"secondaries_first", v});
    }
    for (unsigned seed : {1u, 7u, 42u, 2026u}) {
        std::vector<std::string> v = base;
        std::mt19937 rng(seed);
        std::shuffle(v.begin(), v.end(), rng);
        orders.push_back({"shuffle_" + std::to_string(seed), v});
    }

    for (uint32_t rules : {2957u, 7053u}) {
        const std::string rt = "rules " + std::to_string(rules);
        std::cout << "\n=== " << rt << " ===\n";
        const Result ref = run_case(dir, dir + "/sorted.bam", rules, 0);
        check_expected(ref, rules, rt + " run(sorted)");
        for (const auto& [name, recs] : orders) {
            if (!write_bam(dir, name, recs)) { CHECK(false, "write " + name); continue; }
            const Result r = run_case(dir, dir + "/" + name + ".bam", rules, 0);
            const bool same = r.text() == ref.text();
            CHECK(same, rt + " run(" + name + ") == run(sorted)");
            if (!same) std::cout << "    sorted: " << ref.text() << "\n    " << name << ": " << r.text() << "\n";
        }
        for (int w : {1, 2, 3, 4, 6}) {
            const Result r = run_case(dir, dir + "/sorted.bam", rules, w);
            const bool same = r.text() == ref.text();
            CHECK(same, rt + " run_parallel(" + std::to_string(w) + ") == run(sorted)");
            if (!same) std::cout << "    run:      " << ref.text() << "\n    parallel: " << r.text() << "\n";
        }
    }
    {   // r5 counts at A's exon with the smallest index (R7 over the exonic alignments: the smallest secondary exon); r1 and r2 at 101-200
        uint32_t smallest = UINT32_MAX, at101 = UINT32_MAX;
        exons_of_a(dir, smallest, at101);
        CHECK(smallest != UINT32_MAX && at101 != UINT32_MAX, "A's exons found");
        auto r = run_case(dir, dir + "/sorted.bam", 7053u, 0);
        CHECK(r.by_exon[smallest] == (smallest == at101 ? 3 : 1), "7053: r5 at A's smallest exon index");
        CHECK(r.by_exon[at101] == (smallest == at101 ? 3 : 2), "7053: r1 and r2 at A's exon 101-200");
    }
    {   // without kGeneFullSet an intron-only secondary never counted (r1 counts A under 397), but an intronic primary did: r2 and r5 need the rule
        const Result off = run_case(dir, dir + "/sorted.bam", 397u, 0);
        const Result on = run_case(dir, dir + "/sorted.bam", 397u | singlet::intron_rule::kExonPrecedence, 0);
        auto a_off = off.by_gene, a_on = on.by_gene;
        CHECK(a_off["A"] == 1, "397: A gets r1 only");
        CHECK(a_on["A"] == 3, "4493: A gets r1, r2 and r5");
    }
    std::cout << "\n=== chrM precedence ===\n";
    check_chrm(dir);
    std::system(("rm -rf " + dir).c_str());
    std::cout << "\n=== transcript compatibility ===\n";
    check_tx(dir + "_tx");
    std::cout << "\n=== joint reference ===\n";
    check_joint(dir + "_joint");
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
