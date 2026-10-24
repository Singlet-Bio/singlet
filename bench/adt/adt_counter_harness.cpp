// SPDX-License-Identifier: MIT
// Benchmark harness for include/singlet/pileup/adt_matcher.h + adt_counter.h, wired as src/pipeline/singlet.cpp's
// T4 CITE-seq pass wires them: the read's cell barcode must equal a GEX barcode exactly (no whitelist correction, no
// 3' v3 feature-barcode translation), the tag is matched at R2 offsets 0..max_offset (exact, then Hamming 1), and UMIs
// are deduplicated exactly (no UMI correction).
//
//   adt_counter_harness R1.fastq.gz R2.fastq.gz feature_ref.csv barcodes.txt cb_len umi_len max_offset out_prefix
//
// Writes <out_prefix>.mtx (tags x barcodes, MatrixMarket) and <out_prefix>.json (read statistics).
// Build: g++ -O2 -std=c++17 -I include bench/adt/adt_counter_harness.cpp -lz -o adt_counter_harness
#include <zlib.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "singlet/pileup/adt_counter.h"
#include "singlet/pileup/adt_matcher.h"

namespace {

struct GzLines {
    gzFile f;
    char buf[1 << 16];
    explicit GzLines(const char* p) : f(gzopen(p, "rb")) { if (f) gzbuffer(f, 1 << 20); }
    ~GzLines() { if (f) gzclose(f); }
    bool line(std::string& s) {
        s.clear();
        while (true) {
            char* r = gzgets(f, buf, sizeof(buf));
            if (!r) return !s.empty();
            s.append(r);
            if (!s.empty() && s.back() == '\n') { s.pop_back(); if (!s.empty() && s.back() == '\r') s.pop_back(); return true; }
        }
    }
    bool record(std::string& seq) {
        std::string h, p, q;
        if (!line(h)) return false;
        line(seq); line(p); line(q);
        return true;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "usage: adt_counter_harness R1 R2 feature_ref.csv barcodes.txt cb_len umi_len max_offset out\n";
        return 2;
    }
    const int cb_len = std::stoi(argv[5]), umi_len = std::stoi(argv[6]), max_off = std::stoi(argv[7]);
    const std::string out = argv[8];
    auto t0 = std::chrono::steady_clock::now();

    singlet::AdtMatcher matcher;
    if (!matcher.load_reference(argv[3])) { std::cerr << "cannot load reference\n"; return 1; }
    matcher.build_index();

    std::vector<std::string> barcodes;
    {
        std::ifstream f(argv[4]);
        std::string s;
        while (std::getline(f, s)) {
            if (!s.empty() && s.back() == '\r') s.pop_back();
            auto d = s.find('-');
            if (d != std::string::npos) s = s.substr(0, d);
            if (!s.empty()) barcodes.push_back(s);
        }
    }
    std::unordered_map<std::string, uint32_t> bc_to_idx;
    bc_to_idx.reserve(barcodes.size() * 2);
    for (size_t i = 0; i < barcodes.size(); ++i) bc_to_idx[barcodes[i]] = static_cast<uint32_t>(i);

    singlet::AdtCounter counter;
    counter.set_num_tags(matcher.num_tags());
    counter.set_num_cells(barcodes.size());

    GzLines r1(argv[1]), r2(argv[2]);
    if (!r1.f || !r2.f) { std::cerr << "cannot open reads\n"; return 1; }
    std::string s1, s2;
    size_t n = 0, n_bc = 0;
    while (r1.record(s1) && r2.record(s2)) {
        ++n;
        if (static_cast<int>(s1.size()) < cb_len + umi_len) continue;
        auto it = bc_to_idx.find(s1.substr(0, cb_len));
        if (it == bc_to_idx.end()) continue;
        ++n_bc;
        int tag = matcher.match(s2.c_str(), static_cast<int>(s2.size()), max_off);
        if (tag >= 0) counter.add(it->second, tag, s1.substr(cb_len, umi_len));
    }
    counter.finalize();
    auto csc = counter.counts().to_csc();

    std::ofstream m(out + ".mtx");
    m << "%%MatrixMarket matrix coordinate integer general\n";
    m << matcher.num_tags() << " " << barcodes.size() << " " << csc.data.size() << "\n";
    for (size_t j = 0; j + 1 < csc.indptr.size(); ++j)
        for (auto k = csc.indptr[j]; k < csc.indptr[j + 1]; ++k)
            m << (csc.indices[k] + 1) << " " << (j + 1) << " " << csc.data[k] << "\n";
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::ofstream js(out + ".json");
    js << "{\"reads\": " << n << ", \"reads_barcode_exact\": " << n_bc << ", \"matched\": " << matcher.matched()
       << ", \"ambiguous\": " << matcher.ambiguous() << ", \"unmatched\": " << matcher.unmatched()
       << ", \"umis\": " << counter.dedup_umis() << ", \"seconds\": " << secs << "}\n";
    std::cerr << "done: " << n << " reads, " << counter.dedup_umis() << " UMIs in " << secs << " s\n";
    return 0;
}
