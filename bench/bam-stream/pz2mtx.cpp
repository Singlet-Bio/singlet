// pz2mtx: decode a .1pz matrix with the engine's reader and write MatrixMarket + names.
//   pz2mtx in.1pz out_prefix   ->  out_prefix.mtx, out_prefix.rows.tsv, out_prefix.cols.tsv
// Build (job): g++ -O2 -std=c++17 -I<singlet>/include pz2mtx.cpp -lzstd -o pz2mtx
#include <cstdio>
#include <fstream>
#include <iostream>
#include "singlet/pileup/pz_reader.h"

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: pz2mtx in.1pz out_prefix\n"; return 2; }
    try {
        singlet::pz::ReadLimits lim;
        lim.max_rows = 1ull << 26; lim.max_nnz = 1ull << 31; lim.max_metadata_bytes = 2ull << 30;
        lim.max_kv_entries = 1ull << 26;
        auto r = singlet::pz::read_1pz(argv[1], lim);
        std::string p = argv[2];
        FILE* f = std::fopen((p + ".mtx").c_str(), "w");
        if (!f) { std::perror("open"); return 1; }
        std::fprintf(f, "%%%%MatrixMarket matrix coordinate integer general\n%u %u %llu\n",
                     r.m, r.n, static_cast<unsigned long long>(r.nnz));
        for (uint32_t j = 0; j < r.n; ++j)
            for (uint32_t k = r.indptr[j]; k < r.indptr[j + 1]; ++k)
                std::fprintf(f, "%u %u %u\n", r.indices[k] + 1, j + 1, r.data[k]);
        std::fclose(f);
        std::ofstream rows(p + ".rows.tsv"), cols(p + ".cols.tsv");
        for (auto& s : r.rownames) rows << s << "\n";
        for (auto& s : r.colnames) cols << s << "\n";
    } catch (const std::exception& e) {
        std::cerr << "pz2mtx: " << argv[1] << ": " << e.what() << "\n";
        return 1;
    }
    return 0;
}
