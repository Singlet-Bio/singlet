// Runs singlet::detect_doublets() (include/singlet/pileup/doublet_detect.h) on a CSC matrix read from raw binary files.
// Build: g++ -O2 -std=c++17 -pthread -I<repo>/include/singlet/pileup dbl_run.cpp -o dbl_run
// Run:   dbl_run PREFIX OUT.tsv [key=value ...]
//   PREFIX.hdr (uint32 nrows,ncols,nnz), .indptr (int32), .indices (int32), .data (uint16)
//   optional PREFIX.f2g (uint32 per row: feature -> gene, UINT32_MAX = none; used unless nomap=1)
//   keys: second=PREFIX2 (second source over the same columns, e.g. introns, with its own .f2g), rate (expected multiplet
//         rate; default auto), threads, hvg, pcs, seed, nomap, simratio, hvgmode, vpctl, knn, clf, mask
// OUT.tsv: col, score, is_doublet.  A one-line JSON summary goes to stderr.
#include "doublet_detect.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool rd(const std::string& p, void* dst, size_t bytes) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    size_t n = std::fread(dst, 1, bytes, f);
    std::fclose(f);
    return n == bytes;
}

using CSC = singlet::SparseAccumulator<uint16_t>::CSCMatrix;

static int load(const std::string& pre, CSC& m, std::vector<uint32_t>& f2g, uint32_t& n_genes, bool want_map) {
    uint32_t hdr[3];
    if (!rd(pre + ".hdr", hdr, sizeof hdr)) return 3;
    m.nrows = hdr[0]; m.ncols = hdr[1];
    m.indptr.resize(m.ncols + 1); m.indices.resize(hdr[2]); m.data.resize(hdr[2]);
    if (!rd(pre + ".indptr", m.indptr.data(), m.indptr.size() * 4)) return 4;
    if (!rd(pre + ".indices", m.indices.data(), m.indices.size() * 4)) return 5;
    if (!rd(pre + ".data", m.data.data(), m.data.size() * 2)) return 6;
    f2g.clear();
    if (want_map) {
        f2g.resize(m.nrows);
        if (rd(pre + ".f2g", f2g.data(), f2g.size() * 4)) {
            for (uint32_t g : f2g) if (g != UINT32_MAX) n_genes = std::max(n_genes, g + 1);
        } else f2g.clear();
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    std::string pre = argv[1], second;
    singlet::DoubletParams prm;
    bool nomap = false;
    int tile = 1;   // timing only: repeat the columns this many times
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        size_t e = a.find('=');
        if (e == std::string::npos) continue;
        std::string k = a.substr(0, e); double v = std::atof(a.c_str() + e + 1);
        if (k == "second") second = a.substr(e + 1);
        else if (k == "rate") prm.expected_rate = v;
        else if (k == "threads") prm.threads = (int)v;
        else if (k == "hvg") prm.n_hvg = (uint32_t)v;
        else if (k == "pcs") prm.n_pcs = (uint32_t)v;
        else if (k == "seed") prm.seed = (uint64_t)v;
        else if (k == "simratio") prm.sim_ratio = v;
        else if (k == "nomap") nomap = v != 0;
        else if (k == "hvgmode") prm.hvg_mode = (int)v;
        else if (k == "vpctl") prm.vscore_pctl = v;
        else if (k == "knn") prm.knn_scale = v;
        else if (k == "clf") prm.classifier = v != 0;
        else if (k == "mask") prm.clf_mask = (uint32_t)v;
        else if (k == "tile") tile = (int)v;
        else if (k == "ridge") prm.clf_ridge = v;
        else if (k == "sridge") prm.clf_size_ridge = v;
        else if (k == "rounds") prm.clf_rounds = (int)v;
        else if (k == "excl") prm.clf_exclude = v;
    }
    CSC m, m2;
    std::vector<uint32_t> f2g, f2g2;
    uint32_t n_genes = 0;
    if (int rc = load(pre, m, f2g, n_genes, !nomap)) return rc;
    if (!second.empty()) { if (int rc = load(second, m2, f2g2, n_genes, !nomap)) return 10 + rc; }
    if (tile > 1) {
        CSC t = m; t.ncols = m.ncols * tile;
        t.indptr.assign(1, 0); t.indices.clear(); t.data.clear();
        for (int r = 0; r < tile; ++r)
            for (uint32_t j = 0; j < m.ncols; ++j) {
                for (int32_t k = m.indptr[j]; k < m.indptr[j + 1]; ++k) { t.indices.push_back(m.indices[k]); t.data.push_back(m.data[k]); }
                t.indptr.push_back((int32_t)t.indices.size());
            }
        m = std::move(t);
        if (!second.empty()) {
            CSC t2 = m2; t2.ncols = m2.ncols * tile;
            t2.indptr.assign(1, 0); t2.indices.clear(); t2.data.clear();
            for (int r = 0; r < tile; ++r)
                for (uint32_t j = 0; j < m2.ncols; ++j) {
                    for (int32_t k = m2.indptr[j]; k < m2.indptr[j + 1]; ++k) { t2.indices.push_back(m2.indices[k]); t2.data.push_back(m2.data[k]); }
                    t2.indptr.push_back((int32_t)t2.indices.size());
                }
            m2 = std::move(t2);
        }
    }
    std::vector<uint32_t> cells(m.ncols);
    std::iota(cells.begin(), cells.end(), 0u);
    singlet::DoubletSummary sm;
    auto res = singlet::detect_doublets(m, cells, prm, f2g.empty() ? nullptr : &f2g, n_genes, &sm,
                                        second.empty() ? nullptr : &m2, f2g2.empty() ? nullptr : &f2g2);
    FILE* o = std::fopen(argv[2], "w");
    std::fprintf(o, "col\tscore\tis_doublet\n");
    for (size_t i = 0; i < res.size(); ++i) std::fprintf(o, "%zu\t%.6f\t%d\n", i, res[i].score, res[i].is_doublet ? 1 : 0);
    std::fclose(o);
    std::fprintf(stderr, "{\"n_cells\":%u,\"n_called\":%u,\"expected_rate\":%.5f,\"null_score\":%.4f,\"detectable_fraction\":%.4f,"
                 "\"n_hvg\":%u,\"n_pcs\":%u,\"k_adj\":%u,\"n_sim\":%u,\"seconds\":%.2f,\"fallback\":%d}\n",
                 sm.n_cells, sm.n_called, sm.expected_rate, sm.null_score, sm.detectable_fraction, sm.n_hvg, sm.n_pcs, sm.k, sm.n_sim,
                 sm.seconds, sm.used_fallback ? 1 : 0);
    return 0;
}
