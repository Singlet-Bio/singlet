// SPDX-License-Identifier: MIT
// test_hla_typing: D-101 HLA typing helpers (pipeline/hla_typing.h).
//  1. KmerSet: canonical 19-mers, both strands, N breaks a k-mer.
//  2. The stride-4 prefilter keeps every read with >= 97% identity to an allele (T1K preset hla) for read lengths 44..300:
//     random alleles, reads with floor(0.03 L) substitutions at the worst spread (evenly spaced), on both strands.
//  3. read_donors: the D-52 rule (cell_status cell, a single donor label, prob_max >= 0.9), barcode suffix stripped.
//  4. genotype_json: T1K's genotype table to JSON.
#include "singlet/pipeline/hla_typing.h"

#include <fstream>
#include <iostream>
#include <random>
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
}  // namespace

int main() {
    test_kmers();
    test_prefilter_bound();
    test_donors_and_json();
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail ? 1 : 0;
}
