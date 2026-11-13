// SPDX-License-Identifier: MIT
// bench/species_detect/species_panel.cpp
//
// Scores heads of reads against every species filter of the pack (<dir>/<tag>_21mer.bloom, the format species_detect.h loads) and prints
// the species call of validation/references/SPECIES_BARS.md (rule v0). It is the measuring harness for the filter pack, not a second identification
// tool: the call logic mirrors species_detect.h (hit rate per filter, per-read species-specific split for mixtures) generalized to many filters.
//
//   species_panel --filters DIR --groups groups.tsv [--max-reads 20000] [--min-len 50] < runs.tsv
//   runs.tsv: run_id <TAB> file1.fastq.gz[,file2.fastq.gz]      groups.tsv: tag <TAB> close_group (tags without a row form their own group)
//
// Output (one line per run): run, reads, kmers, flags, call, members, top1, rate1, top2, rate2, minor_fraction, reads_A, reads_B, seconds, all rates (tag=rate;...)

#include <zlib.h>
#include <dirent.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/bloom_filter.h"
#include "singlet/pileup/kmer_util.h"

static const int K = 21;
// rule v0 parameters (SPECIES_BARS.md)
static double P_MIN_RATE = 0.02, P_CALL_RATIO = 1.25, P_PAIR_RATIO = 1.5, P_MIX_FRAC = 0.15;
static int P_MIX_READS = 50, P_MIX_EXCESS = 10;

struct Filt {
    std::string tag, group;
    singlet::pileup::BloomFilter bf{64};
};

static inline int code(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}

// first max_reads records of a gzip FASTQ (truncated tails are tolerated)
static std::vector<std::string> read_head(const std::string& path, size_t max_reads) {
    std::vector<std::string> out;
    gzFile f = gzopen(path.c_str(), "rb");
    if (!f) return out;
    std::vector<char> buf(1 << 16);
    size_t line = 0;
    while (out.size() < max_reads && gzgets(f, buf.data(), static_cast<int>(buf.size()))) {
        if (line % 4 == 1) {
            size_t n = std::strlen(buf.data());
            while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) --n;
            out.emplace_back(buf.data(), n);
        }
        ++line;
    }
    gzclose(f);
    return out;
}

int main(int argc, char** argv) {
    std::string fdir, gfile;
    size_t max_reads = 20000;
    int min_len = 50;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--filters") fdir = nxt();
        else if (a == "--groups") gfile = nxt();
        else if (a == "--max-reads") max_reads = std::stoul(nxt());
        else if (a == "--min-len") min_len = std::stoi(nxt());
        else if (a == "--min-rate") P_MIN_RATE = std::stod(nxt());
        else if (a == "--call-ratio") P_CALL_RATIO = std::stod(nxt());
        else if (a == "--pair-ratio") P_PAIR_RATIO = std::stod(nxt());
        else if (a == "--mix-frac") P_MIX_FRAC = std::stod(nxt());
    }
    if (fdir.empty()) {
        std::fprintf(stderr, "usage: species_panel --filters DIR [--groups groups.tsv] < runs.tsv\n");
        return 2;
    }
    std::map<std::string, std::string> grp;
    if (!gfile.empty()) {
        std::ifstream g(gfile);
        std::string l;
        while (std::getline(g, l)) {
            auto t = l.find('\t');
            if (t != std::string::npos) grp[l.substr(0, t)] = l.substr(t + 1);
        }
    }
    std::vector<std::string> names;
    if (DIR* d = opendir(fdir.c_str())) {
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            const std::string suf = "_21mer.bloom";
            if (n.size() > suf.size() && n.compare(n.size() - suf.size(), suf.size(), suf) == 0) names.push_back(n.substr(0, n.size() - suf.size()));
        }
        closedir(d);
    }
    std::sort(names.begin(), names.end());
    std::vector<Filt> F(names.size());
    auto t_load = std::chrono::steady_clock::now();
    for (size_t i = 0; i < names.size(); ++i) {
        F[i].tag = names[i];
        auto it = grp.find(names[i]);
        F[i].group = it == grp.end() ? names[i] : it->second;
        if (!F[i].bf.load(fdir + "/" + names[i] + "_21mer.bloom")) {
            std::fprintf(stderr, "cannot load filter %s\n", names[i].c_str());
            return 3;
        }
    }
    std::fprintf(stderr, "[panel] %zu filters loaded in %.1f s\n", F.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count());
    const size_t S = F.size();
    const uint64_t mask = (1ULL << (2 * K)) - 1, POLYC = 0x15555555555ULL;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        auto tab = line.find('\t');
        std::string run = line.substr(0, tab), files = tab == std::string::npos ? "" : line.substr(tab + 1);
        auto t0 = std::chrono::steady_clock::now();
        // reads of the head, per file, then the qualifying ones interleaved across files
        std::vector<std::vector<std::string>> per;
        {
            std::stringstream ss(files);
            std::string p;
            while (std::getline(ss, p, ',')) if (!p.empty()) per.push_back(read_head(p, 100000));
        }
        std::string flags;
        std::vector<const std::string*> use;
        for (int ml : {min_len, 30}) {
            std::vector<std::vector<const std::string*>> q(per.size());
            size_t total = 0;
            for (size_t f = 0; f < per.size(); ++f)
                for (auto& r : per[f]) if (static_cast<int>(r.size()) >= ml) { q[f].push_back(&r); ++total; }
            if (total >= 2000 || ml == 30) {
                if (ml != min_len) flags += "short_reads;";
                for (size_t i = 0; use.size() < max_reads; ++i) {
                    bool any = false;
                    for (size_t f = 0; f < per.size() && use.size() < max_reads; ++f)
                        if (i < q[f].size()) { use.push_back(q[f][i]); any = true; }
                    if (!any) break;
                }
                break;
            }
        }
        std::vector<std::vector<uint16_t>> hits(use.size(), std::vector<uint16_t>(S, 0));
        std::vector<uint64_t> tot(S, 0);
        uint64_t nk = 0;
        for (size_t ri = 0; ri < use.size(); ++ri) {
            const std::string& s = *use[ri];
            uint64_t km = 0;
            int valid = 0;
            for (char c : s) {
                int b = code(c);
                if (b < 0) { valid = 0; km = 0; continue; }
                km = ((km << 2) | static_cast<uint64_t>(b)) & mask;
                if (++valid >= K) {
                    uint64_t cn = singlet::pileup::kmer::canonical_2bit(km, K);
                    if (cn == 0 || cn == POLYC) continue;
                    ++nk;
                    for (size_t si = 0; si < S; ++si)
                        if (F[si].bf.query(cn)) { ++hits[ri][si]; ++tot[si]; }
                }
            }
        }
        std::vector<double> rate(S, 0.0);
        for (size_t si = 0; si < S; ++si) rate[si] = nk ? static_cast<double>(tot[si]) / static_cast<double>(nk) : 0.0;
        std::vector<size_t> ord(S);
        for (size_t i = 0; i < S; ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return rate[a] > rate[b]; });
        std::string call = "NONE", members;
        double minor = 0;
        int nA = 0, nB = 0;
        size_t a = S ? ord[0] : 0, b = S > 1 ? ord[1] : 0;
        if (S >= 2 && nk > 0) {
            // B for the mixture test: the best filter outside A's close group
            size_t bm = a;
            bool have = false;
            for (size_t k = 1; k < S; ++k)
                if (F[ord[k]].group != F[a].group) { bm = ord[k]; have = true; break; }
            bool mixture = false;
            if (have && rate[a] >= P_MIN_RATE) {
                for (auto& h : hits) {
                    int d = static_cast<int>(h[a]) - static_cast<int>(h[bm]);
                    if (d >= P_MIX_EXCESS && 2 * h[bm] <= h[a]) ++nA;
                    else if (-d >= P_MIX_EXCESS && 2 * h[a] <= h[bm]) ++nB;
                }
                int mn = std::min(nA, nB), cl = nA + nB;
                minor = cl ? static_cast<double>(mn) / cl : 0;
                mixture = mn >= P_MIX_READS && minor >= P_MIX_FRAC;
            }
            if (mixture) {
                call = "MIXTURE";
                members = F[a].tag + "+" + F[bm].tag;
            } else if (rate[a] < P_MIN_RATE) {
                call = "NONE";
            } else {
                bool same = F[a].group == F[b].group;
                double ratio = rate[b] > 0 ? rate[a] / rate[b] : 1e9;
                if (same && ratio < P_PAIR_RATIO) {
                    call = "CLOSE_PAIR";
                    for (size_t k = 0; k < S; ++k)
                        if (F[ord[k]].group == F[a].group && rate[ord[k]] * P_PAIR_RATIO > rate[a]) members += (members.empty() ? "" : "|") + F[ord[k]].tag;
                } else if (ratio >= P_CALL_RATIO) {
                    call = "CALL";
                    members = F[a].tag;
                } else {
                    call = "AMBIGUOUS";
                }
            }
        } else if (S == 1 && nk > 0 && rate[0] >= P_MIN_RATE) {
            call = "CALL";
            members = F[0].tag;
        }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%s\t%zu\t%llu\t%s\t%s\t%s\t%s\t%.4f\t%s\t%.4f\t%.4f\t%d\t%d\t%.2f\t", run.c_str(), use.size(), static_cast<unsigned long long>(nk), flags.c_str(), call.c_str(), members.c_str(),
                    S ? F[a].tag.c_str() : "", S ? rate[a] : 0.0, S > 1 ? F[b].tag.c_str() : "", S > 1 ? rate[b] : 0.0, minor, nA, nB, sec);
        for (size_t k = 0; k < S; ++k) std::printf("%s%s=%.4f", k ? ";" : "", F[ord[k]].tag.c_str(), rate[ord[k]]);
        std::printf("\n");
        std::fflush(stdout);
    }
    return 0;
}
