// SPDX-License-Identifier: MIT
// test_hla_typing: D-101 HLA typing helpers (pipeline/hla_typing.h).
//  1. KmerSet: canonical 19-mers, both strands, N breaks a k-mer.
//  2. The stride-4 prefilter keeps every read with >= 97% identity to an allele (T1K preset hla) for read lengths 44..300:
//     random alleles, reads with floor(0.03 L) substitutions at the worst spread (evenly spaced), on both strands.
//  3. read_donors: the D-52 rule (cell_status cell, a single donor label, prob_max >= 0.9), barcode suffix stripped.
//  4. genotype_json: T1K's genotype table to JSON.
//  5. the deferred route (write_pending / run_pending) against a stand-in for run-t1k: the same files as the direct route, pending/ is owner-only and
//     deleted once the summary lands, and every failure keeps pending/ and leaves no summary.
#include "singlet/pipeline/hla_typing.h"
#include "singlet/pileup/run_provenance.h"

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
std::vector<uint8_t> codes(const std::string& s) {
    std::vector<uint8_t> v; for (char c : s) { int b = singlet::hla::KmerSet::code(c); v.push_back(b < 0 ? 4 : uint8_t(b)); } return v;
}
std::string revcomp(const std::string& s) {
    std::string r(s.rbegin(), s.rend());
    for (char& c : r) c = c == 'A' ? 'T' : c == 'C' ? 'G' : c == 'G' ? 'C' : c == 'T' ? 'A' : 'N';
    return r;
}

void test_kmers() {
    std::cout << "=== KmerSet ===\n";
    singlet::hla::KmerSet ks;
    const std::string a = "ACGTTGCAAGGCTTACCGATCGGATCCATGCAAATTTGGGCCCAAT";   // 46 bases
    ks.add_sequence(a); ks.build();
    CHECK(ks.size() > 0 && ks.size() <= a.size() - 18, "at most one canonical 19-mer per position");
    auto r = codes(a.substr(5, 30)), rc = codes(revcomp(a.substr(5, 30)));
    CHECK(ks.hits(r.data(), int(r.size())) && ks.hits(rc.data(), int(rc.size())), "a 30-base piece hits on either strand");
    std::string other(40, 'A'); for (size_t i = 0; i < other.size(); ++i) other[i] = "ACGT"[(i * 7 + 3) % 4];
    auto o = codes(other);
    CHECK(!ks.hits(o.data(), int(o.size())), "an unrelated read misses");
    std::string withn = a.substr(0, 24); withn[9] = 'N';
    auto wn = codes(withn);
    CHECK(!ks.hits(wn.data(), int(wn.size())), "N inside every 19-mer of a 24-base read: no hit");
    auto sh = codes(a.substr(0, 18));
    CHECK(!ks.hits(sh.data(), int(sh.size())), "a read shorter than 19 never hits");
}

void test_prefilter_bound() {
    std::cout << "=== stride-4 prefilter keeps every >= 97% identity read (L 44..300) ===\n";
    std::mt19937 rng(11);
    std::string allele(4000, 'A');
    for (auto& c : allele) c = "ACGT"[rng() % 4];
    singlet::hla::KmerSet ks; ks.add_sequence(allele); ks.build();
    long tried = 0, missed = 0;
    for (int L = 44; L <= 300; ++L) {
        const int m = static_cast<int>(0.03 * L + 1e-9);
        for (int rep = 0; rep < 12; ++rep) {
            const int start = static_cast<int>(rng() % (allele.size() - L));
            std::string read = allele.substr(start, L);
            // spread the substitutions to minimise the longest exact run (worst case), with a random phase
            const int phase = static_cast<int>(rng() % std::max(1, L / (m + 1)));
            for (int k = 1; k <= m; ++k) {
                int pos = std::min(L - 1, (k * L) / (m + 1) - phase / (m + 1));
                char c = read[pos]; char n = c; while (n == c) n = "ACGT"[rng() % 4]; read[pos] = n;
            }
            if (rep % 2) read = revcomp(read);
            auto v = codes(read);
            ++tried; if (!ks.hits(v.data(), L)) ++missed;
        }
    }
    CHECK(missed == 0, "missed " + std::to_string(missed) + " of " + std::to_string(tried) + " reads");
}

void test_donors_and_json() {
    std::cout << "=== read_donors / genotype_json ===\n";
    const std::string dir = "/tmp/test_hla_" + std::to_string(getpid());
    std::system(("mkdir -p " + dir).c_str());
    {
        std::ofstream f(dir + "/da.tsv");
        f << "cell\tdonor_id\tprob_max\tprob_doublet\tbest_singlet\tbest_doublet\tcell_status\n"
          << "AAAC-1\tdonor0\t0.99\t0\tdonor0\tdonor0\tcell\n"
          << "AAAG-1\tdonor1\t0.95\t0\tdonor1\tdonor1\tcell\n"
          << "AAAT-1\tdonor1\t0.80\t0\tdonor1\tdonor1\tcell\n"          // low probability
          << "AACA-1\tdoublet\t0.99\t0.9\tdonor0\tdonor0,donor1\tcell\n" // doublet
          << "AACC-1\tdonor0\t0.99\t0\tdonor0\tdonor0\tambient\n"       // not a called cell
          << "AACG-1\tunassigned\t0.4\t0\tdonor1\tdonor1\tcell\n";
    }
    int nd = 0;
    auto m = singlet::hla::read_donors(dir + "/da.tsv", nd);
    CHECK(nd == 2 && m.size() == 2 && m["AAAC"] == 0 && m["AAAG"] == 1, "two cells pass the donor rule, suffix stripped");
    {
        std::ofstream f(dir + "/g.tsv");
        f << "HLA-A\t2\tA*02:01:01\t120.5\t60\tA*24:02:01\t80\t45\tA*02:01:02\n"
          << "HLA-DQB1\t1\tDQB1*06:02:01\t30\t23\t.\t0\t-1\t\n";
    }
    const std::string j = singlet::hla::genotype_json(dir + "/g.tsv");
    CHECK(j.find("\"HLA-A\": {\"alleles\": [\"A*02:01:01\", \"A*24:02:01\"], \"abundance\": [120.5, 80], \"quality\": [60, 45]}") != std::string::npos,
          "two-allele gene");
    CHECK(j.find("\"HLA-DQB1\": {\"alleles\": [\"DQB1*06:02:01\", \".\"], \"abundance\": [30, 0], \"quality\": [23, -1]}") != std::string::npos,
          "one-allele gene keeps T1K's '.' and quality -1");
    std::system(("rm -rf " + dir).c_str());
}

// ---- the deferred route, with a stand-in for T1K (perl, like the real run-t1k): it reports the number of reads it was given
const char* kFakeT1k = R"PL(#!/usr/bin/perl
my %a; my $no = 0;
while (@ARGV) { my $x = shift @ARGV; if ($x eq '--noExtraction') { $no = 1 } else { $a{$x} = shift @ARGV } }
my ($in, $o, $d) = ($a{'-u'}, $a{'-o'}, $a{'--od'});
exit 3 if $ENV{FAKE_T1K_FAIL} eq 'all' or ($ENV{FAKE_T1K_FAIL} eq 'donor' and $o =~ /^donor/);
open(I, $in) or exit 4; my @l = <I>; close I;
my $n = int(@l / 4);
open(G, ">$d/${o}_genotype.tsv"); print G "HLA-A\t1\tA*02:01:01\t$n\t60\t.\t0\t-1\t\n"; close G;
for my $s ('allele.tsv', 'allele.vcf') { open(F, ">$d/${o}_$s"); print F "$o $n\n"; close F; }
unless ($no) { open(C, ">$d/${o}_candidate.fq"); print C @l; close C; }
exit 0;
)PL";

std::string read_all(const std::string& p) { std::ifstream f(p, std::ios::binary); std::stringstream b; b << f.rdbuf(); return b.str(); }
std::string no_timing(const std::string& s) { std::stringstream in(s), o; std::string l; while (std::getline(in, l)) if (l.find("seconds") == std::string::npos) o << l << "\n"; return o.str(); }
bool exists(const std::string& p) { return singlet::hla::file_exists(p); }

/// A Session as sample_stage leaves it after the prefilter, `n` candidate reads over the two donors' barcodes.
void make_stage(singlet::hla::Session& s, const std::string& t1k, const std::string& run, int n, bool defer) {
    std::system(("mkdir -p " + run + "/tmpstage").c_str());
    { std::ofstream f(run + "/donor_assignments.tsv");
      f << "cell\tdonor_id\tprob_max\tprob_doublet\tbest_singlet\tbest_doublet\tcell_status\nAAAC-1\tdonor0\t0.99\t0\tdonor0\tdonor0\tcell\nAAAG-1\tdonor1\t0.95\t0\tdonor1\tdonor1\tcell\n"; }
    s.cfg.t1k_dir = t1k; s.cfg.out_prefix = run; s.cfg.donor_assignments = run + "/donor_assignments.tsv"; s.cfg.tmp_dir = run + "/tmpstage"; s.cfg.threads = 2; s.cfg.defer = defer;
    s.dir = run + "/private/hla"; s.tmp = s.cfg.tmp_dir; s.kept_fq = s.tmp + "/prefilter.fq"; s.made = true; s.stopped = false;
    s.res.reads_scanned = 1000; s.res.reads_kept = n; s.res.n_kmers = 77; s.res.n_alleles = 1;
    std::ofstream f(s.kept_fq);
    for (int i = 0; i < n; ++i) f << "@" << i << "_" << (i % 2 ? "AAAG" : "AAAC") << "\nACGTACGT\n+\nIIIIIIII\n";
}

/// `n` reads through the deferred route's first half: <run>/private/hla/pending/.
void make_pending(const std::string& t1k, const std::string& run, int n) {
    singlet::hla::Session s; make_stage(s, t1k, run, n, true);
    if (n > 0) singlet::hla::gzip_file(s.kept_fq, s.tmp + "/" + singlet::hla::kPendingReads, "wb1");
    singlet::hla::finish_stage(s);
}

void test_deferred_route() {
    std::cout << "=== deferred route ===\n";
    if (std::system("perl -e 1 2>/dev/null") != 0) { std::cout << "  no perl: skipped\n"; return; }
    using namespace singlet::hla;
    const std::string d = "/tmp/test_hla_defer_" + std::to_string(getpid()), t1k = d + "/t1k";
    std::system(("mkdir -p " + t1k + "/bin " + t1k + "/hlaidx").c_str());
    { std::ofstream f(t1k + "/bin/run-t1k"); f << kFakeT1k; }
    { std::ofstream f(t1k + "/hlaidx/hlaidx_rna_seq.fa"); f << ">A*02:01:01\nACGTACGTACGTACGTACGTACGT\n"; }
    { std::ofstream f(t1k + "/manifest.json"); f << "{\"tool\": \"T1K\"}\n"; }
    const std::string pendRel = "/private/hla/pending", gz = std::string("/") + kPendingReads;

    const std::string a = d + "/direct", b = d + "/defer";
    { Session s; make_stage(s, t1k, a, 10, false); type_sample(s); finish_stage(s); }
    make_pending(t1k, b, 10);
    CHECK(exists(b + pendRel + "/manifest.json") && exists(b + pendRel + gz), "--hla-defer leaves the reads and the manifest");
    CHECK(!exists(b + "/private/hla/hla_summary.json"), "no summary before the CPU step");
    struct stat st{}; stat((b + pendRel + gz).c_str(), &st);
    CHECK((st.st_mode & 077) == 0, "the candidate reads are owner-only");
    stat((b + "/private").c_str(), &st);
    CHECK((st.st_mode & 077) == 0, "private/ is owner-only");
    CHECK(singlet::is_private_output("private/hla/pending/candidate_reads.fq.gz") && singlet::is_private_output("private/hla/pending/manifest.json"), "pending/ is a private output");
    const std::string man = read_all(b + pendRel + "/manifest.json");
    std::string v;
    CHECK(manifest_field(man, "donor_assignments", v) && v == "donor_assignments.tsv", "the donor file is relative to the run directory");
    CHECK(manifest_field(man, "cdna_is_r1", v) && v == "false", "the manifest records the cDNA orientation");

    { // the run directory may move before the CPU job
      const std::string moved = d + "/moved";
      std::system(("mv " + b + " " + moved).c_str());
      const Result r = run_pending(moved + pendRel + "/", "", 2, d + "/tmp1");
      CHECK(r.landed && r.ran && r.n_donors_typed == 2, "the CPU step lands, donors typed");
      CHECK(!exists(moved + pendRel) && !exists(d + "/tmp1"), "pending/ and the work directory are deleted");
      bool same = true;
      for (const char* f : {"hla_genotype.tsv", "hla_allele.tsv", "hla_allele.vcf", "hla_candidate_reads.fq.gz", "donors/donor0_genotype.tsv", "donors/donor1_genotype.tsv"})
          if (!exists(a + "/private/hla/" + f) || read_all(a + "/private/hla/" + f) != read_all(moved + "/private/hla/" + f)) { same = false; std::cout << "  differs: " << f << "\n"; }
      CHECK(same, "every output is byte-identical to the direct route");
      CHECK(no_timing(read_all(a + "/private/hla/hla_summary.json")) == no_timing(read_all(moved + "/private/hla/hla_summary.json")), "the summary is identical apart from its timing lines");
    }
    auto retained = [&](const std::string& r, const Result& res, const char* what) {
        CHECK(!res.landed && exists(r + pendRel + "/manifest.json") && !exists(r + "/private/hla/hla_summary.json"), what);
    };
    { const std::string r = d + "/fail"; make_pending(t1k, r, 10);
      setenv("FAKE_T1K_FAIL", "all", 1);
      retained(r, run_pending(r + pendRel, "", 2, d + "/tmp2"), "T1K fails: pending/ stays, no summary");
      setenv("FAKE_T1K_FAIL", "donor", 1);
      retained(r, run_pending(r + pendRel, "", 2, d + "/tmp3"), "a donor fails: pending/ stays, no summary");
      unsetenv("FAKE_T1K_FAIL");
      const Result again = run_pending(r + pendRel, "", 2, d + "/tmp4");
      CHECK(again.landed && !exists(r + pendRel), "the job can run again after a failure");
    }
    { const std::string r = d + "/short"; make_pending(t1k, r, 10);
      std::string m = read_all(r + pendRel + "/manifest.json");
      m.replace(m.find("\"reads_kept\": 10"), 16, "\"reads_kept\": 11");
      { std::ofstream f(r + pendRel + "/manifest.json"); f << m; }
      retained(r, run_pending(r + pendRel, "", 2, d + "/tmp5"), "reads short of the manifest: refused");
    }
    { const std::string r = d + "/nomanifest"; make_pending(t1k, r, 10);
      std::remove((r + pendRel + "/manifest.json").c_str());
      const Result x = run_pending(r + pendRel, "", 2, d + "/tmp6");
      CHECK(!x.landed && exists(r + pendRel + gz), "no manifest: refused");
    }
    CHECK(!run_pending(d + "/direct/private/hla", "", 2, d + "/tmp7").landed, "a directory that is not pending/: refused");
    { const std::string r = d + "/none"; make_pending(t1k, r, 0);
      const Result x = run_pending(r + pendRel, "", 2, d + "/tmp8");
      CHECK(x.landed && !x.ran && read_all(r + "/private/hla/hla_summary.json").find("no read shares a k-mer") != std::string::npos, "no candidate read: lands, not typed, as the direct route");
    }
    std::system(("rm -rf " + d).c_str());
}
}  // namespace

int main() {
    test_kmers();
    test_prefilter_bound();
    test_donors_and_json();
    test_deferred_route();
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail ? 1 : 0;
}
