// SPDX-License-Identifier: MIT
// bench/species_detect/build_species_bloom.cpp
//
// Builds the 21-mer Bloom filter singlet's species detection loads (include/singlet/pileup/species_detect.h: <dir>/human_21mer.bloom and mouse_21mer.bloom,
// $SINGLET_SPECIES_FILTERS). The filter holds the canonical 21-mers of every exon of a reference (the union of the exon intervals of its genes.gtf, cut out of
// genome.fa), so cDNA reads of that species hit it and reads of another species hit it only through conserved sequence.
//
//   build_species_bloom --genome genome.fa --gtf genes.gtf --out human_21mer.bloom [--bits-per-kmer 14]
//
// Bloom format: singlet::pileup::BloomFilter (magic BLOM, version 1, 3 hash functions). At 14 bits per k-mer the false-positive rate is ~0.5%.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "singlet/pileup/bloom_filter.h"
#include "singlet/pileup/kmer_util.h"

static const int K = 21;

static std::string strip_chr(const std::string& s) { return s.rfind("chr", 0) == 0 ? s.substr(3) : s; }

int main(int argc, char** argv) {
    std::string genome, gtf, out;
    double bits_per_kmer = 14.0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--genome" && i + 1 < argc) genome = argv[++i];
        else if (a == "--gtf" && i + 1 < argc) gtf = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--bits-per-kmer" && i + 1 < argc) bits_per_kmer = std::atof(argv[++i]);
    }
    if (genome.empty() || gtf.empty() || out.empty()) {
        std::fprintf(stderr, "usage: build_species_bloom --genome genome.fa --gtf genes.gtf --out X.bloom [--bits-per-kmer 14]\n");
        return 2;
    }
    // 1. genome: name (first word of the header) -> sequence
    std::unordered_map<std::string, std::string> chrom;
    {
        std::ifstream fa(genome);
        std::string line, name;
        while (std::getline(fa, line)) {
            if (!line.empty() && line[0] == '>') {
                name = line.substr(1, line.find_first_of(" \t") - 1);
                chrom[name].reserve(1 << 20);
            } else if (!name.empty()) {
                for (char& c : line) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                chrom[name] += line;
            }
        }
    }
    std::unordered_map<std::string, std::string*> by_alt;   // chr-prefix-insensitive lookup
    for (auto& kv : chrom) by_alt[strip_chr(kv.first)] = &kv.second;
    // 2. exon intervals per chromosome, merged
    std::map<std::string, std::vector<std::pair<uint64_t, uint64_t>>> iv;
    {
        std::ifstream g(gtf);
        std::string line;
        while (std::getline(g, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t t1 = line.find('\t'), t2 = line.find('\t', t1 + 1), t3 = line.find('\t', t2 + 1), t4 = line.find('\t', t3 + 1), t5 = line.find('\t', t4 + 1);
            if (t5 == std::string::npos) continue;
            if (line.compare(t2 + 1, t3 - t2 - 1, "exon") != 0) continue;
            uint64_t s = std::strtoull(line.c_str() + t3 + 1, nullptr, 10), e = std::strtoull(line.c_str() + t4 + 1, nullptr, 10);
            if (s >= 1 && e >= s) iv[strip_chr(line.substr(0, t1))].emplace_back(s - 1, e);
        }
    }
    uint64_t total = 0;
    for (auto& kv : iv) {
        auto& v = kv.second;
        std::sort(v.begin(), v.end());
        std::vector<std::pair<uint64_t, uint64_t>> m;
        for (auto& p : v) {
            if (!m.empty() && p.first <= m.back().second) m.back().second = std::max(m.back().second, p.second);
            else m.push_back(p);
        }
        v.swap(m);
        for (auto& p : v) total += p.second - p.first;
    }
    std::fprintf(stderr, "[bloom] %zu chromosomes, %zu with exons, %llu exonic bp\n", chrom.size(), iv.size(), static_cast<unsigned long long>(total));
    if (total == 0) { std::fprintf(stderr, "[bloom] no exons found: chromosome names of the GTF and the FASTA do not match?\n"); return 3; }
    singlet::pileup::BloomFilter bf(static_cast<uint64_t>(static_cast<double>(total) * bits_per_kmer));
    uint64_t inserted = 0;
    const uint64_t mask = (1ULL << (2 * K)) - 1;
    for (auto& kv : iv) {
        auto it = by_alt.find(kv.first);
        if (it == by_alt.end()) continue;
        const std::string& seq = *it->second;
        for (auto& p : kv.second) {
            uint64_t kmer = 0;
            int valid = 0;
            for (uint64_t i = p.first; i < p.second && i < seq.size(); ++i) {
                int b;
                switch (seq[i]) { case 'A': b = 0; break; case 'C': b = 1; break; case 'G': b = 2; break; case 'T': b = 3; break; default: b = -1; }
                if (b < 0) { valid = 0; kmer = 0; continue; }
                kmer = ((kmer << 2) | static_cast<uint64_t>(b)) & mask;
                if (++valid >= K) { bf.insert(singlet::pileup::kmer::canonical_2bit(kmer, K)); ++inserted; }
            }
        }
    }
    bf.save(out);
    std::fprintf(stderr, "[bloom] %llu k-mers inserted, %llu bits (%.0f MB) -> %s\n", static_cast<unsigned long long>(inserted),
                 static_cast<unsigned long long>(bf.n_bits()), static_cast<double>(bf.n_bits()) / 8e6, out.c_str());
    return 0;
}
