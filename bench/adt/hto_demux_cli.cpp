// SPDX-License-Identifier: MIT
// hto_demux_cli: run singlet::HtoDemux (include/singlet/pileup/hto_demux.h) on a hashtag count table.
//   hto_demux_cli COUNTS.csv OUT.tsv [--reverse] [--confident-min X] [--doublet-nmads X] [--doublet-min X] [--veto X]
// COUNTS.csv: tags x cells with a header row of barcodes (first column = tag names), integer counts, the layout of the Howitt
// et al. Zenodo tables. OUT.tsv: barcode, call (a tag name, Doublet or Negative). --reverse feeds the cells in reverse order
// (the output must not change).
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/hto_demux.h"

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else if (c != '"' && c != '\r') cur += c;
    }
    out.push_back(cur);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: hto_demux_cli COUNTS.csv OUT.tsv [--reverse] [--confident-min X] ...\n"; return 2; }
    bool reverse = false;
    singlet::HtoDemuxParams params;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--reverse") reverse = true;
        else if (a == "--confident-min" && i + 1 < argc) params.confident_min = std::stod(argv[++i]);
        else if (a == "--doublet-nmads" && i + 1 < argc) params.doublet_nmads = std::stod(argv[++i]);
        else if (a == "--doublet-min" && i + 1 < argc) params.doublet_min = std::stod(argv[++i]);
        else if (a == "--veto" && i + 1 < argc) params.mixture_positive = std::stod(argv[++i]);
        else { std::cerr << "unknown argument " << a << "\n"; return 2; }
    }
    std::ifstream in(argv[1]);
    std::string line;
    std::getline(in, line);
    auto hdr = split(line);
    std::vector<std::string> barcodes(hdr.begin() + 1, hdr.end());
    std::vector<std::string> tags;
    std::vector<std::vector<uint32_t>> rows;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto f = split(line);
        tags.push_back(f[0]);
        std::vector<uint32_t> r;
        for (size_t i = 1; i < f.size(); ++i) r.push_back(static_cast<uint32_t>(std::stod(f[i])));
        rows.push_back(std::move(r));
    }
    const size_t n = barcodes.size(), k = tags.size();
    std::vector<size_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = reverse ? n - 1 - i : i;  // fed position -> original cell
    singlet::SparseAccumulator<uint32_t> acc;
    acc.set_n_features(static_cast<uint32_t>(k));
    acc.set_barcodes(barcodes);
    for (size_t t = 0; t < k; ++t)
        for (size_t c = 0; c < n; ++c)
            if (rows[t][perm[c]]) acc.increment(static_cast<uint32_t>(t), static_cast<uint32_t>(c), rows[t][perm[c]]);
    singlet::HtoDemux demux(params);
    auto res = demux.demux(acc, tags, n);
    std::vector<std::string> call(n);
    for (size_t c = 0; c < n; ++c) {
        const auto& r = res[c];
        call[perm[c]] = r.classification == "SINGLET" ? r.assignment : (r.classification == "DOUBLET" ? "Doublet" : "Negative");
    }
    std::ofstream out(argv[2]);
    out << "barcode\tcall\n";
    for (size_t c = 0; c < n; ++c) out << barcodes[c] << '\t' << call[c] << '\n';
    return 0;
}
