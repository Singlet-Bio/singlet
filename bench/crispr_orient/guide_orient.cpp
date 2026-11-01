// Guide matching on real guide-library reads, before/after the orientation fix.
//   guide_orient R1.fastq.gz R2.fastq.gz guides.csv N reverse_fraction_percent
// R1 = 16 nt barcode + 12 nt UMI, R2 = protospacer read. Each R2 is presented to the matcher as a BAM record would
// hold it: a deterministic reverse_fraction_percent of reads is stored reverse-complemented (aligned to the reverse
// strand). Build against the old crispr_guide.h (match only) or the new one (-DNEWMATCH: match_bam with the flag).
#include "singlet/pileup/crispr_guide.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static bool next_rec(FILE* f, std::string& seq) {
    char buf[4096];
    for (int i = 0; i < 4; ++i) {
        if (!fgets(buf, sizeof buf, f)) return false;
        if (i == 1) { seq = buf; while (!seq.empty() && (seq.back() == '\n' || seq.back() == '\r')) seq.pop_back(); }
    }
    return true;
}

static void rc(std::string& s) {
    std::reverse(s.begin(), s.end());
    for (auto& c : s) c = c == 'A' ? 'T' : c == 'C' ? 'G' : c == 'G' ? 'C' : c == 'T' ? 'A' : 'N';
}

int main(int argc, char** argv) {
    if (argc < 6) return 2;
    singlet::GuideRef ref;
    if (!ref.load_csv(argv[3])) { std::cerr << "no guides\n"; return 1; }
    long n = atol(argv[4]); int revpct = atoi(argv[5]);
    FILE* f1 = popen((std::string("zcat ") + argv[1]).c_str(), "r");
    FILE* f2 = popen((std::string("zcat ") + argv[2]).c_str(), "r");
    std::string r1, r2;
    long reads = 0, rev_reads = 0, matched = 0, rev_matched = 0, amb = 0, bad = 0;
    std::unordered_set<std::string> seen;
    std::unordered_map<std::string, int> per_cell;
    uint64_t h = 88172645463325252ull;
    while (reads < n && next_rec(f1, r1) && next_rec(f2, r2)) {
        ++reads;
        h ^= h << 13; h ^= h >> 7; h ^= h << 17;
        bool rev = static_cast<int>(h % 100) < revpct;
        std::string s = r2;
        if (rev) { rc(s); ++rev_reads; }
        int gid;
#ifdef NEWMATCH
        // s is "as stored"; match_bam restores sequencing orientation when told the read is reverse
        gid = ref.match_bam(s.data(), (int)s.size(), rev);
#else
        gid = ref.match(s.data(), (int)s.size());
#endif
        if (gid == -2) { ++amb; continue; }
        if (gid < 0) continue;
        ++matched; if (rev) ++rev_matched;
        if (r1.size() < 28 || r1.find('N') != std::string::npos) { ++bad; continue; }
        std::string key = r1.substr(0, 16) + "|" + std::to_string(gid) + "|" + r1.substr(16, 12);
        if (seen.insert(key).second) per_cell[r1.substr(0, 16)]++;
    }
    std::vector<int> v;
    for (auto& kv : per_cell) v.push_back(kv.second);
    std::sort(v.begin(), v.end());
    long umis = 0; for (int x : v) umis += x;
    long big = 0; for (int x : v) if (x >= 3) ++big;
    printf("reads %ld | stored-reverse %ld | matched %ld (%.4f) | reverse matched %ld (%.4f of reverse) | ambiguous %ld | "
           "guide UMIs %ld | cells with >=1 UMI %zu | cells with >=3 UMIs %ld | median UMIs/cell(>=3) %d\n",
           reads, rev_reads, matched, (double)matched / reads, rev_matched, rev_reads ? (double)rev_matched / rev_reads : 0.0,
           amb, umis, v.size(), big, [&] { std::vector<int> w; for (int x : v) if (x >= 3) w.push_back(x); return w.empty() ? 0 : w[w.size() / 2]; }());
    return 0;
}
