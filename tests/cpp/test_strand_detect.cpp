// SPDX-License-Identifier: MIT
// tests/cpp/test_strand_detect.cpp
//
// Strand auto-detection on synthetic BAMs (samtools). Two bugs left 5' 10x
// libraries counted antisense:
//   1. run_parallel()'s pre-probe read the first reads of the busiest
//      chromosome, often chrM, whose genes overlap on both strands, so it
//      rarely flipped;
//   2. it set strand_flipped_ whether or not it flipped, and that flag gated
//      the >50% wrong-strand retry, so the retry never fired.
//
// Layout: chr1 and chr2 (2 Mb) carry 40 single-exon genes each, alternating
// strand. chrM carries two overlapping genes on opposite strands (every chrM
// read is strand-uninformative) and is the busiest chromosome.
//
// Part 0 checks the encode-time 10x 3'/5' discrimination (five_prime_evidence).

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "singlet/fq/protocol.h"
#include "singlet/pileup/pileup_engine.h"
#include "test_harness.h"

namespace {

// ── Part 0: 10x 3' vs 5' from read content ────────────────────────────────
struct Spot {
    std::vector<uint8_t> r1_seq, r2_seq;
    uint16_t r1_len = 0, r2_len = 0;
};

std::vector<uint8_t> enc(const std::string& s) {
    std::vector<uint8_t> v;
    for (char c : s) v.push_back(c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4);
    return v;
}

std::string random_seq(std::mt19937& rng, int n) {
    std::string s;
    for (int i = 0; i < n; ++i) s += "ACGT"[rng() & 3];
    return s;
}

// r1_extra: bases sequenced past BC+UMI (26). R2 features at the rates measured on
// 10x data (per read): 3' has the 3'-kit TSO (4%), poly(A) 3% and poly(T) 0.2%;
// 5' has no 3'-kit TSO, poly(A) 1.5% and poly(T) 3%.
std::vector<Spot> spots(bool five, int r1_extra, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::vector<Spot> out;
    for (int i = 0; i < 4000; ++i) {
        Spot sp;
        std::string r1 = random_seq(rng, 26);
        if (r1_extra > 0)
            r1 += (five ? std::string("TTTCTTATATGGG") : std::string(13, 'T')) + random_seq(rng, r1_extra - 13);
        std::string ins;
        double x = u01(rng);
        double p_smart = five ? 0.0 : 0.04, p_a = five ? 0.015 : 0.03, p_t = five ? 0.03 : 0.002;
        if (x < p_smart) ins = "AAGCAGTGGTATCAACGCAGAGTACATGGG";
        else if (x < p_smart + p_a) ins = std::string(20, 'A');
        else if (x < p_smart + p_a + p_t) ins = std::string(20, 'T');
        std::string r2 = random_seq(rng, 30) + ins;
        r2 += random_seq(rng, 90 - static_cast<int>(r2.size()));
        sp.r1_seq = enc(r1); sp.r2_seq = enc(r2);
        sp.r1_len = static_cast<uint16_t>(r1.size()); sp.r2_len = static_cast<uint16_t>(r2.size());
        out.push_back(std::move(sp));
    }
    return out;
}

void test_five_prime_evidence() {
    std::cout << "\n=== Part 0: 10x 3'/5' read evidence ===\n";
    using singlet::fq::five_prime_evidence;
    CHECK(five_prime_evidence(spots(true, 0, 1), 26).is_5prime(), "26 bp R1, 5': no 3' TSO, polyT > polyA -> 5'");
    CHECK(!five_prime_evidence(spots(false, 0, 2), 26).is_5prime(), "26 bp R1, 3': 3' TSO and polyA in R2 -> 3'");
    CHECK(five_prime_evidence(spots(true, 30, 3), 26).is_5prime(), "long R1, 5': TSO after BC+UMI -> 5'");
    CHECK(!five_prime_evidence(spots(false, 30, 4), 26).is_5prime(), "long R1, 3': poly(dT) after BC+UMI -> 3'");
    auto few = spots(true, 0, 5);
    few.resize(300);
    CHECK(!five_prime_evidence(few, 26).is_5prime(), "too few spots -> stays 3'");
}

// ── Synthetic BAMs ────────────────────────────────────────────────────────
const char* kBC = "AAACCTGAGAAACGAG-1";
constexpr int kGenesPerChrom = 40;
constexpr int kGeneSpacing = 50000;
constexpr int kGeneLen = 2000;

std::string umi_of(uint64_t i) {
    std::string u(10, 'A');
    for (int k = 0; k < 10; ++k) { u[k] = "ACGT"[i & 3]; i >>= 2; }
    return u;
}

void write_gtf(const std::string& dir) {
    std::ofstream g(dir + "/mini.gtf");
    auto gl = [&](const std::string& chr, const char* feat, int s, int e, char strand,
                  const std::string& id) {
        g << chr << "\t.\t" << feat << "\t" << s << "\t" << e << "\t.\t" << strand
          << "\t.\tgene_id \"" << id << "\"; gene_name \"" << id << "\";\n";
    };
    for (const char* chr : {"chr1", "chr2"}) {
        for (int i = 0; i < kGenesPerChrom; ++i) {
            int s = 10000 + i * kGeneSpacing;
            char st = (i % 2) ? '-' : '+';
            std::string id = std::string(chr) + "_G" + std::to_string(i);
            gl(chr, "gene", s, s + kGeneLen - 1, st, id);
            gl(chr, "exon", s, s + kGeneLen - 1, st, id);
        }
    }
    gl("chrM", "gene", 1, 8000, '+', "MT_A");
    gl("chrM", "exon", 1, 8000, '+', "MT_A");
    gl("chrM", "gene", 1, 8000, '-', "MT_B");
    gl("chrM", "exon", 1, 8000, '-', "MT_B");
    std::ofstream b(dir + "/barcodes.tsv");
    b << kBC << "\n";
}

// antisense_frac: share of nuclear reads antisense to their gene (10x 3' is 0,
// a 5' library read this way is ~1). sort=false writes an unsorted, unindexed
// BAM, which run() must probe in-line.
bool write_bam(const std::string& dir, const std::string& name, int n_nuclear,
               double antisense_frac, int n_chrm, bool sort, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_int_distribution<int> gene(0, kGenesPerChrom - 1), off(0, kGeneLen - 21);
    std::ofstream f(dir + "/" + name + ".sam");
    f << "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr1\tLN:2000000\n@SQ\tSN:chr2\tLN:2000000\n"
         "@SQ\tSN:chrM\tLN:16569\n";
    const std::string seq(20, 'A');
    uint64_t r = 0;
    auto line = [&](const std::string& chr, int pos, bool rev) {
        f << "r" << r << "\t" << (rev ? 16 : 0) << "\t" << chr << "\t" << pos << "\t255\t20M\t*\t0\t0\t"
          << seq << "\t*\tCB:Z:" << kBC << "\tUB:Z:" << umi_of(r) << "\tNH:i:1\n";
        ++r;
    };
    // interleave nuclear and chrM reads so an unsorted stream mixes them
    int nn = 0, nm = 0;
    while (nn < n_nuclear || nm < n_chrm) {
        bool take_m = nm < n_chrm && (nn >= n_nuclear || u01(rng) < double(n_chrm) / (n_chrm + n_nuclear));
        if (take_m) {
            line("chrM", 1 + static_cast<int>(u01(rng) * 7900), u01(rng) < 0.5);
            ++nm;
        } else {
            const char* chr = u01(rng) < 0.5 ? "chr1" : "chr2";
            int g = gene(rng);
            bool minus = g % 2;
            bool anti = u01(rng) < antisense_frac;
            line(chr, 10000 + g * kGeneSpacing + off(rng), minus != anti);
            ++nn;
        }
    }
    f.close();
    std::string sam = dir + "/" + name + ".sam", bam = dir + "/" + name + ".bam";
    std::string cmd = sort ? "samtools sort -o " + bam + " " + sam + " 2>/dev/null && samtools index " + bam
                           : "samtools view -b -o " + bam + " " + sam + " 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
}

singlet::PileupConfig base_config(const std::string& dir) {
    singlet::PileupConfig cfg;
    cfg.barcode_path = dir + "/barcodes.tsv";
    cfg.exon_gtf_path = dir + "/mini.gtf";
    cfg.threads = 2;
    cfg.umi_len = 10;
    cfg.count_sj = false;
    cfg.aim_ancestry = false;
    cfg.count_vdj = false;
    return cfg;
}

struct Result {
    singlet::PileupStats st;
    bool probed = false, flipped = false, reverse = false, retried = false;
    double probe_af = 0;
};

enum class Path { Parallel, Run };

// One pass plus the post-pileup safety net, as src/pipeline/singlet.cpp does it.
Result run_pass(const std::string& dir, const std::string& bam, Path path, singlet::PileupConfig cfg) {
    Result res;
    auto engine = std::make_unique<singlet::PileupEngine>(cfg);
    if (!engine->load_references()) { CHECK(false, "load_references"); return res; }
    auto run = [&](singlet::PileupEngine& e) {
        return path == Path::Parallel ? e.run_parallel(dir + "/" + bam, 2) : e.run(dir + "/" + bam);
    };
    res.st = run(*engine);
    res.probed = engine->strand_was_probed();
    res.flipped = engine->strand_was_flipped();
    res.probe_af = engine->strand_probe().antisense_frac();
    res.retried = singlet::rerun_if_wrong_strand(engine, cfg, res.st, run);
    if (res.retried) CHECK(cfg.reverse_strand == engine->reverse_strand(), "retry updates the base config");
    res.reverse = engine->reverse_strand();
    return res;
}

void test_5prime(const std::string& dir, Path path, const char* tag) {
    std::cout << "\n=== 5' library (antisense), chrM busiest: " << tag << " ===\n";
    auto r = run_pass(dir, "five.bam", path, base_config(dir));
    CHECK(r.probed, std::string(tag) + ": probe decided");
    CHECK(r.probe_af > 0.9, std::string(tag) + ": probe sees nuclear antisense reads (chrM skipped)");
    CHECK(r.flipped, std::string(tag) + ": probe flipped to reverse");
    CHECK(r.reverse, std::string(tag) + ": counted reverse-strand");
    CHECK(!r.retried, std::string(tag) + ": no safety-net retry needed");
    CHECK(singlet::counted_antisense_frac(r.st, r.reverse) < 0.1, std::string(tag) + ": <10% wrong-strand after the flip");
    CHECK(r.st.exon_hits >= 3500, std::string(tag) + ": nuclear reads counted");
}

void test_3prime(const std::string& dir, Path path, const char* tag) {
    std::cout << "\n=== 3' library (sense): " << tag << " ===\n";
    auto r = run_pass(dir, "three.bam", path, base_config(dir));
    CHECK(r.probed && r.probe_af < 0.1, std::string(tag) + ": probe decided, sense");
    CHECK(!r.flipped && !r.reverse && !r.retried, std::string(tag) + ": stays forward");
    CHECK(singlet::counted_antisense_frac(r.st, r.reverse) < 0.1, std::string(tag) + ": <10% wrong-strand");
}

void test_protocol_says_reverse(const std::string& dir) {
    std::cout << "\n=== 3' data labelled 5' (reverse_strand preset) ===\n";
    auto cfg = base_config(dir);
    cfg.reverse_strand = true;
    auto r = run_pass(dir, "three.bam", Path::Parallel, cfg);
    CHECK(r.flipped && !r.reverse, "probe clears a wrong protocol-derived reverse_strand");
    CHECK(singlet::counted_antisense_frac(r.st, r.reverse) < 0.1, "<10% wrong-strand");
}

void test_user_forced(const std::string& dir) {
    std::cout << "\n=== user-forced orientation (auto_strand off) ===\n";
    auto cfg = base_config(dir);
    cfg.auto_strand = false;  // as --reverse-strand does; here forced forward on 5' data
    auto r = run_pass(dir, "five.bam", Path::Parallel, cfg);
    CHECK(!r.probed && !r.flipped && !r.retried && !r.reverse, "no probe, no retry: orientation kept");
    CHECK(singlet::counted_antisense_frac(r.st, r.reverse) > 0.9, "5' data counted forward: >90% antisense");
    CHECK(singlet::wrong_strand_frac(r.st) < 0.5, "... which chrM reads hide from wrong_strand / exon_hits");
}

void test_safety_net(const std::string& dir, Path path, const char* tag) {
    // 57% antisense: the probe sits in its no-call band and keeps forward; the
    // finished pass is >50% wrong-strand, so the safety net re-runs reversed.
    std::cout << "\n=== safety net (probe undecided): " << tag << " ===\n";
    auto r = run_pass(dir, "noisy.bam", path, base_config(dir));
    CHECK(r.probed && !r.flipped, std::string(tag) + ": probe ran but did not flip");
    CHECK(r.retried, std::string(tag) + ": safety net fired (probe ran, flip not applied)");
    CHECK(r.reverse, std::string(tag) + ": retry counted reverse-strand");
    CHECK(singlet::counted_antisense_frac(r.st, r.reverse) < 0.5, std::string(tag) + ": retry has <50% wrong-strand");
}

void test_inline(const std::string& dir) {
    std::cout << "\n=== streaming run(): unsorted, unindexed 5' BAM ===\n";
    auto r = run_pass(dir, "five_stream.bam", Path::Run, base_config(dir));
    CHECK(r.probed && r.flipped && r.reverse, "in-line probe flipped to reverse");
    CHECK(r.probe_af > 0.9, "in-line probe ignores chrM reads");
    CHECK(singlet::counted_antisense_frac(r.st, r.reverse) < 0.5, "most reads counted after the flip");
    CHECK(!r.retried, "no retry needed");
}

}  // namespace

int main() {
    test_five_prime_evidence();

    std::string dir = "/tmp/test_strand_detect_" + std::to_string(getpid());
    std::system(("mkdir -p " + dir).c_str());
    write_gtf(dir);
    bool ok = write_bam(dir, "five", 4000, 1.0, 20000, true, 1) &&
              write_bam(dir, "three", 4000, 0.0, 20000, true, 2) &&
              write_bam(dir, "noisy", 4000, 0.57, 20000, true, 3) &&
              write_bam(dir, "five_stream", 30000, 1.0, 20000, false, 4);
    if (!ok) {
        std::cout << "SKIP: samtools not available\n";
    } else {
        test_5prime(dir, Path::Parallel, "run_parallel");
        test_5prime(dir, Path::Run, "run (indexed BAM)");
        test_3prime(dir, Path::Parallel, "run_parallel");
        test_3prime(dir, Path::Run, "run (indexed BAM)");
        test_protocol_says_reverse(dir);
        test_user_forced(dir);
        test_safety_net(dir, Path::Parallel, "run_parallel");
        test_safety_net(dir, Path::Run, "run (indexed BAM)");
        test_inline(dir);
    }
    std::system(("rm -rf " + dir).c_str());

    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
