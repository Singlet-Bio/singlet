// SPDX-License-Identifier: MIT
// bench/protocol_geometry/detect_fastq.cpp
//
// Runs singlet::fq::detect_protocol (the decode gate's protocol inference) on the first N spots of FASTQ pairs and prints what it decides:
//
//   detect_fastq WHITELIST_DIR LIST.tsv [N=5000]      LIST.tsv: id <tab> barcode-read.fastq <tab> cDNA-read.fastq
//   -> id <tab> tag <tab> confidence <tab> wl_match_rate <tab> r1_length <tab> barcode-read length <tab> would the encoder trim R1 (MEDIUM+ and r1_length < observed)
//
// Used by spot_length_census.sbatch to measure how many samples with a long barcode read are classified as 10x and trimmed.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/fq/protocol.h"

struct Spot {
    std::vector<uint8_t> r1_seq, r2_seq;
    uint16_t r1_len = 0, r2_len = 0;
};

static std::vector<Spot> read_pair(const std::string& f1, const std::string& f2, size_t n) {
    std::ifstream a(f1), b(f2);
    std::vector<Spot> spots;
    std::string l, s1, s2;
    auto conv = [](const std::string& s, std::vector<uint8_t>& v) {
        for (char c : s) v.push_back(singlet::fq::nuc::ascii_to_num(c));
    };
    while (spots.size() < n && std::getline(a, l)) {
        if (!std::getline(a, s1)) break;
        std::getline(a, l); std::getline(a, l);
        std::getline(b, l);
        if (!std::getline(b, s2)) break;
        std::getline(b, l); std::getline(b, l);
        Spot s;
        conv(s1, s.r1_seq); conv(s2, s.r2_seq);
        s.r1_len = static_cast<uint16_t>(s1.size());
        s.r2_len = static_cast<uint16_t>(s2.size());
        spots.push_back(std::move(s));
    }
    return spots;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: detect_fastq WHITELIST_DIR LIST.tsv [N]\n"); return 2; }
    const std::vector<std::string> dirs{argv[1]};
    const size_t n = argc > 3 ? static_cast<size_t>(std::atol(argv[3])) : 5000;
    std::ifstream list(argv[2]);
    std::string line;
    while (std::getline(list, line)) {
        std::istringstream ss(line);
        std::string id, f1, f2;
        if (!(ss >> id >> f1 >> f2)) continue;
        auto spots = read_pair(f1, f2, n);
        if (spots.empty()) { std::printf("%s\tNOREADS\t-\t-\t-\t-\t-\n", id.c_str()); continue; }
        auto r = singlet::fq::detect_protocol(spots, spots[0].r1_len, spots[0].r2_len, dirs);
        const bool trim = r.confidence >= singlet::fq::Confidence::MEDIUM && r.r1_length > 0 && r.r1_length < spots[0].r1_len;
        std::printf("%s\t%s\t%d\t%.3f\t%u\t%u\t%d\n", id.c_str(), r.tag.c_str(), static_cast<int>(r.confidence), r.wl_match_rate, r.r1_length, spots[0].r1_len, trim ? 1 : 0);
        std::fflush(stdout);
    }
    return 0;
}
