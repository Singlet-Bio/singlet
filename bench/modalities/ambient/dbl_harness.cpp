// Calls singlet::detect_doublets() (include/singlet/pileup/doublet_detect.h) unchanged on a CSC matrix read from raw binary
// files, so the doublet module can be scored on a matrix with known (synthetic) heterotypic doublets.
// Build: g++ -O2 -std=c++17 -I<repo>/include/singlet/pileup dbl_harness.cpp -o dbl_harness
// Run:   dbl_harness PREFIX OUT.tsv      (PREFIX.hdr = uint32 nrows,ncols,nnz; .indptr int32; .indices int32; .data uint16)
#include "doublet_detect.h"

#include <cstdio>
#include <numeric>

static bool rd(const std::string& p, void* dst, size_t bytes) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    size_t n = std::fread(dst, 1, bytes, f);
    std::fclose(f);
    return n == bytes;
}

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    std::string pre = argv[1];
    uint32_t hdr[3];
    if (!rd(pre + ".hdr", hdr, sizeof hdr)) return 3;
    singlet::SparseAccumulator<uint16_t>::CSCMatrix m;
    m.nrows = hdr[0];
    m.ncols = hdr[1];
    m.indptr.resize(m.ncols + 1);
    m.indices.resize(hdr[2]);
    m.data.resize(hdr[2]);
    if (!rd(pre + ".indptr", m.indptr.data(), m.indptr.size() * 4)) return 4;
    if (!rd(pre + ".indices", m.indices.data(), m.indices.size() * 4)) return 5;
    if (!rd(pre + ".data", m.data.data(), m.data.size() * 2)) return 6;
    std::vector<uint32_t> cells(m.ncols);
    std::iota(cells.begin(), cells.end(), 0u);
    auto res = singlet::detect_doublets(m, cells);  // default expected_doublet_rate, as export.h calls it
    FILE* o = std::fopen(argv[2], "w");
    std::fprintf(o, "col\tscore\tis_doublet\n");
    for (size_t i = 0; i < res.size(); ++i) std::fprintf(o, "%zu\t%.6f\t%d\n", i, res[i].score, res[i].is_doublet ? 1 : 0);
    std::fclose(o);
    return 0;
}
