// SPDX-License-Identifier: MIT
// D-152 diagnostic: feed real inDrop reads (two FASTQ files) through the .1fq Writer with the W1 anchor and the R2 adapter trim, read the
// blocks back and report what R2 looks like (length, N fraction) with and without the adapter trim. usage: indrop_roundtrip R1.fq R2.fq N WL_DIR
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "singlet/fq/lib1fq.h"

using namespace singlet::fq;

static bool rec(std::ifstream& f, std::string& seq, std::string& q) {
    std::string a, b;
    if (!std::getline(f, a) || !std::getline(f, seq) || !std::getline(f, b) || !std::getline(f, q)) return false;
    return true;
}
static std::vector<uint8_t> num(const std::string& s) {
    std::vector<uint8_t> v;
    for (char c : s) v.push_back(c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    const long N = std::atol(argv[3]);
    for (int mode = 0; mode < 2; ++mode) {   // 0: adapter trim off, 1: on
        std::ifstream f1(argv[1]), f2(argv[2]);
        auto anchor = std::make_shared<IndropAnchor>();
        if (!anchor->load({argv[4]})) { std::cerr << "no lists\n"; return 3; }
        WriterConfig w;
        w.qual_mode = QualMode::BINNED4;
        w.n_streams = 2;
        w.r1_length = IndropAnchor::kCanonLen;
        w.r2_length = 0;
        w.polya_trim = true; w.polya_min_len = 10; w.polya_max_mm = 1;
        w.r1_indrop = anchor;
        if (mode == 1) w.r2_adapters.seqs = {AdapterSet::from_string("AGATCGGAAGAGC"), AdapterSet::from_string("AAGGCGTCACAAGCAATCACTC")};
        char path[64];
        std::snprintf(path, sizeof path, "/tmp/indrop_rt_%d.1fq", mode);
        Writer wr;
        wr.open(path, w);
        std::string s1, q1, s2, q2;
        long n = 0;
        while (n < N && rec(f1, s1, q1) && rec(f2, s2, q2)) {
            auto r1 = num(s1), r2 = num(s2);
            std::vector<uint8_t> q(r2.size(), 30), qq(r1.size(), 30);
            wr.add_read(r2.data(), static_cast<uint16_t>(r2.size()), q.data(), r1.data(), static_cast<uint16_t>(r1.size()), qq.data());
            ++n;
        }
        wr.finish("");
        Reader rd;
        rd.open(path);
        DecodedBlock b;
        uint64_t reads = 0, bases = 0, nb = 0, short20 = 0, printed = 0;
        while (rd.read_block(b)) {
            for (uint32_t i = 0; i < b.n_reads; ++i) {
                const uint8_t* s = b.r2_data.data() + b.r2_offsets[i];
                const uint16_t L = b.r2_lengths[i];
                ++reads; bases += L; short20 += (L <= 20);
                for (uint16_t j = 0; j < L; ++j) nb += (s[j] > 3);
                if (printed < 4 && reads % 997 == 0) {
                    std::string t; for (uint16_t j = 0; j < L; ++j) t.push_back("ACGTN"[s[j] > 4 ? 4 : s[j]]);
                    std::cout << "  R2[" << reads << "] len " << L << " " << t << "\n"; ++printed;
                }
            }
        }
        std::cout << "mode " << mode << " (adapter trim " << (mode ? "on" : "off") << "): reads " << reads << " mean R2 length " << (double)bases / std::max<uint64_t>(1, reads)
                  << " N fraction " << (double)nb / std::max<uint64_t>(1, bases) << " reads <=20 nt " << (double)short20 / std::max<uint64_t>(1, reads) << "\n";
        anchor->report(std::cout);
    }
    return 0;
}
