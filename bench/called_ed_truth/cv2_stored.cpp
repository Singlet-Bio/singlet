// SPDX-License-Identifier: MIT
// bench/called_ed_truth/cv2_stored.cpp
//
// The pipeline's called_v2 (include/singlet/pileup/called_v2.h) on the counts a Singlet run already stored, without
// re-aligning: reads RUN/{exon,intron}_counts.mtx.gz (+ _features / _barcodes .tsv.gz) and the `called` column of
// RUN/cell_calls.tsv, aggregates exon + intron per gene as the pipeline does (calledv2::gene_counts), runs calledv2::call
// and writes one row per barcode: barcode, called, is_cell (2.2), called_v2 (2.1), called_lowrna, umi_total, pvalue, fdr. The parameters go to stdout as JSON.
//
//   g++ -std=c++17 -O2 -pthread -Iinclude bench/called_ed_truth/cv2_stored.cpp -o cv2_stored
//   cv2_stored RUN_DIR OUT.tsv [threads]
//
// Used by bench/called_ed_truth/stored.sbatch to check the C++ 2.1 call against the truth-check script's recommended calls
// (docs/called-ed-truth.md). Needs `gzip` on PATH.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "singlet/pileup/called_v2.h"

namespace {

struct Csc {
    uint32_t nrows = 0, ncols = 0;
    std::vector<int64_t> indptr;
    std::vector<int32_t> indices;
    std::vector<int64_t> data;
};

FILE* open_gz(const std::string& path) {
    std::string cmd = "gzip -dc '" + path + "'";
    FILE* f = popen(cmd.c_str(), "r");
    if (!f) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(2);
    }
    return f;
}

bool read_line(FILE* f, std::string& line) {
    line.clear();
    int c;
    while ((c = std::fgetc(f)) != EOF) {
        if (c == '\n') return true;
        line.push_back(static_cast<char>(c));
    }
    return !line.empty();
}

std::vector<std::string> first_column(const std::string& path) {
    FILE* f = open_gz(path);
    std::vector<std::string> out;
    std::string line;
    while (read_line(f, line)) {
        if (line.empty()) continue;
        out.push_back(line.substr(0, line.find('\t')));
    }
    pclose(f);
    return out;
}

Csc read_mtx(const std::string& path) {
    FILE* f = open_gz(path);
    std::string line;
    do {
        if (!read_line(f, line)) {
            std::fprintf(stderr, "%s: empty\n", path.c_str());
            std::exit(2);
        }
    } while (!line.empty() && line[0] == '%');
    long long m, n, nnz;
    if (std::sscanf(line.c_str(), "%lld %lld %lld", &m, &n, &nnz) != 3) {
        std::fprintf(stderr, "%s: bad header\n", path.c_str());
        std::exit(2);
    }
    std::vector<int32_t> ri;
    std::vector<int32_t> ci;
    std::vector<int64_t> v;
    ri.reserve(nnz);
    ci.reserve(nnz);
    v.reserve(nnz);
    long long i, j, x;
    while (std::fscanf(f, "%lld %lld %lld", &i, &j, &x) == 3) {
        ri.push_back(static_cast<int32_t>(i - 1));
        ci.push_back(static_cast<int32_t>(j - 1));
        v.push_back(x);
    }
    pclose(f);
    Csc M;
    M.nrows = static_cast<uint32_t>(m);
    M.ncols = static_cast<uint32_t>(n);
    M.indptr.assign(n + 1, 0);
    for (auto c : ci) ++M.indptr[c + 1];
    for (long long c = 0; c < n; ++c) M.indptr[c + 1] += M.indptr[c];
    M.indices.resize(ri.size());
    M.data.resize(ri.size());
    std::vector<int64_t> fill(M.indptr.begin(), M.indptr.end() - 1);
    for (size_t k = 0; k < ri.size(); ++k) {
        int64_t p = fill[ci[k]]++;
        M.indices[p] = ri[k];
        M.data[p] = v[k];
    }
    for (long long c = 0; c < n; ++c) {  // rows ascending within each column
        std::vector<std::pair<int32_t, int64_t>> col;
        for (int64_t p = M.indptr[c]; p < M.indptr[c + 1]; ++p) col.emplace_back(M.indices[p], M.data[p]);
        std::sort(col.begin(), col.end());
        for (size_t k = 0; k < col.size(); ++k) {
            M.indices[M.indptr[c] + k] = col[k].first;
            M.data[M.indptr[c] + k] = col[k].second;
        }
    }
    return M;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: cv2_stored RUN_DIR OUT.tsv [threads]\n");
        return 2;
    }
    const std::string run = argv[1];
    const int threads = argc > 3 ? std::atoi(argv[3]) : 1;
    Csc ex = read_mtx(run + "/exon_counts.mtx.gz");
    Csc in = read_mtx(run + "/intron_counts.mtx.gz");
    auto exf = first_column(run + "/exon_counts_features.tsv.gz");
    auto inf = first_column(run + "/intron_counts_features.tsv.gz");
    auto bcs = first_column(run + "/exon_counts_barcodes.tsv.gz");
    auto ibcs = first_column(run + "/intron_counts_barcodes.tsv.gz");
    if (bcs != ibcs || bcs.size() != ex.ncols) {
        std::fprintf(stderr, "exon / intron barcodes differ\n");
        return 2;
    }
    // called flags from cell_calls.tsv (column "called"), matched by barcode
    std::vector<uint8_t> called(bcs.size(), 0);
    {
        FILE* f = std::fopen((run + "/cell_calls.tsv").c_str(), "r");
        if (!f) {
            std::fprintf(stderr, "no cell_calls.tsv\n");
            return 2;
        }
        std::string line;
        read_line(f, line);
        std::vector<std::string> head;
        {
            size_t s = 0;
            for (size_t e; (e = line.find('\t', s)) != std::string::npos; s = e + 1) head.push_back(line.substr(s, e - s));
            head.push_back(line.substr(s));
        }
        int cb = -1, cc = -1;
        for (size_t k = 0; k < head.size(); ++k) {
            if (head[k] == "barcode") cb = static_cast<int>(k);
            if (head[k] == "called_knee" || (head[k] == "called" && cc < 0)) cc = static_cast<int>(k);  // called_knee since the is_cell change; older files: called
        }
        if (cb < 0 || cc < 0) {
            std::fprintf(stderr, "cell_calls.tsv lacks barcode / called_knee (called)\n");
            return 2;
        }
        std::map<std::string, size_t> pos;
        for (size_t k = 0; k < bcs.size(); ++k) pos[bcs[k]] = k;
        while (read_line(f, line)) {
            std::vector<std::string> fl;
            size_t s = 0;
            for (size_t e; (e = line.find('\t', s)) != std::string::npos; s = e + 1) fl.push_back(line.substr(s, e - s));
            fl.push_back(line.substr(s));
            auto it = pos.find(fl[cb]);
            if (it != pos.end()) called[it->second] = (fl[cc] == "1" || fl[cc] == "TRUE" || fl[cc] == "True") ? 1 : 0;
        }
        std::fclose(f);
    }
    auto t0 = std::chrono::steady_clock::now();
    singlet::calledv2::GeneCounts X = singlet::calledv2::gene_counts(ex, exf, &in, &inf);
    singlet::calledv2::Result R = singlet::calledv2::call(X, called, "", singlet::calledv2::Params(), threads);
    R.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    FILE* o = std::fopen(argv[2], "w");
    std::fputs("barcode\tcalled\tis_cell\tcalled_v2\tcalled_lowrna\tumi_total\tpvalue\tfdr\n", o);
    for (size_t k = 0; k < bcs.size(); ++k) {
        std::fprintf(o, "%s\t%d\t%d\t%d\t%d\t%lld\t", bcs[k].c_str(), called[k] ? 1 : 0, R.is_cell[k] ? 1 : 0, R.called_v2[k] ? 1 : 0,
                     (R.called_v2[k] && !R.is_cell[k]) ? 1 : 0, static_cast<long long>(R.totals[k]));
        if (std::isnan(R.pvalue[k]))
            std::fputs("NA\tNA\n", o);
        else
            std::fprintf(o, "%.17g\t%.9g\n", R.pvalue[k], static_cast<double>(R.fdr[k]));
    }
    std::fclose(o);
    std::cout << singlet::calledv2::params_json(R, "") << std::endl;
    return 0;
}
