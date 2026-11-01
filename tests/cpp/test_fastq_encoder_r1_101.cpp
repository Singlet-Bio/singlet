// SPDX-License-Identifier: MIT
// tests/cpp/test_fastq_encoder_r1_101.cpp
//
// A 10x v3 library whose barcode read R1 was sequenced to 101 nt (GSE230239; the protocol reads 28): the FASTQ
// encoder used to keep all 101 bases against a 28-nt protocol and lost every read downstream. It now cuts R1 to
// the protocol's length, as the SRA encoder does. Checked with the protocol forced and with a normal 28-nt R1.

#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "singlet/fq/fastq_encoder.h"
#include "singlet/fq/reader.h"
#include "test_harness.h"

using namespace singlet::fq;

namespace {

std::string rnd(std::mt19937_64& rng, size_t n) {
    static const char b[] = "ACGT";
    std::string s(n, 'A');
    for (auto& c : s) c = b[rng() % 4];
    return s;
}

void write_pair(const std::string& p1, const std::string& p2, size_t r1_len, size_t n, std::vector<std::string>& r1_seqs) {
    std::mt19937_64 rng(5);
    std::ofstream f1(p1), f2(p2);
    for (size_t i = 0; i < n; ++i) {
        std::string a = rnd(rng, r1_len), b = rnd(rng, 91);
        r1_seqs.push_back(a);
        f1 << "@r" << i << "\n" << a << "\n+\n" << std::string(a.size(), 'I') << "\n";
        f2 << "@r" << i << "\n" << b << "\n+\n" << std::string(b.size(), 'I') << "\n";
    }
}

void run_case(size_t r1_len) {
    const std::string tag = "r1len" + std::to_string(r1_len);
    const std::string p1 = "tmp_" + tag + "_1.fastq", p2 = "tmp_" + tag + "_2.fastq", out = "tmp_" + tag + ".1fq";
    std::vector<std::string> r1;
    const size_t n = 3000;
    write_pair(p1, p2, r1_len, n, r1);

    EncoderConfig cfg;
    cfg.output_path = out;
    cfg.protocol_tag = "10x-3p-v3";
    cfg.no_dedup = true;
    cfg.no_trim = true;
    cfg.qual_mode = QualMode::NONE;
    FastqEncoder enc;
    auto st = enc.encode(p1, p2, cfg);
    CHECK(st.total_reads == n, "R1=" + std::to_string(r1_len) + ": every read is encoded");

    Reader rd;
    rd.open(out.c_str());
    CHECK(rd.header().stream_lengths[0] == 28, "R1=" + std::to_string(r1_len) + ": archive R1 length is the protocol's 28");
    DecodedBlock blk;
    size_t seen = 0;
    bool prefix_ok = true, len_ok = true;
    while (rd.read_block(blk)) {
        for (uint32_t i = 0; i < blk.n_reads; ++i, ++seen) {
            len_ok = len_ok && blk.r1_len(i) == 28;
            for (int j = 0; j < 28 && prefix_ok && seen < r1.size(); ++j)
                prefix_ok = "ACGT"[blk.r1_seq(i)[j] & 3] == r1[seen][j];
        }
    }
    CHECK(seen == n, "R1=" + std::to_string(r1_len) + ": every read is readable back");
    CHECK(len_ok && prefix_ok, "R1=" + std::to_string(r1_len) + ": stored R1 is the first 28 bases of the input");
    rd.close();
    for (const auto& f : {p1, p2, out}) std::remove(f.c_str());
}

}  // namespace

int main() {
    run_case(101);   // over-sequenced barcode read
    run_case(28);    // the normal case is unchanged
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
