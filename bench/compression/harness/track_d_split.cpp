// Track D: how many chunk-payload bytes does today's writer spend on called cells versus empty droplets?
// Re-encodes the column subsets (cells only / empties only / all) with the production writer and reports
// the VOCSC chunk bytes (sum of the chunk table), at zstd level 3 (today) and 19.
//   bin/track_d_split.exe mask_dir file.1pz...        mask file: <mask_dir>/<sample>__<member>.mask (1 byte per column, 1 = called cell)
// Output TSV on stdout: sample member subset ncols nnz level chunk_bytes
#include "common.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>

using namespace cs;

static size_t chunk_bytes(const std::vector<uint8_t>& f) {
    uint32_t nch, perm_z, ptr_z;
    std::memcpy(&nch, f.data() + 28, 4);
    std::memcpy(&perm_z, f.data() + 32, 4);
    std::memcpy(&ptr_z, f.data() + 36, 4);
    size_t off = 96 + perm_z + ptr_z, sum = 0;
    for (uint32_t i = 0; i < nch; ++i) {
        uint32_t s;
        std::memcpy(&s, f.data() + off + 4 * i, 4);
        sum += s;
    }
    return sum;
}

static CanonicalMatrix subset(const CanonicalMatrix& M, const std::vector<uint8_t>& keep) {
    CanonicalMatrix S;
    S.m = M.m;
    S.vt_code = M.vt_code;
    S.indptr.push_back(0);
    for (uint32_t j = 0; j < M.n; ++j) {
        if (!keep[j]) continue;
        for (uint32_t k = M.indptr[j]; k < M.indptr[j + 1]; ++k) {
            S.indices.push_back(M.indices[k]);
            S.data.push_back(M.data[k]);
        }
        S.indptr.push_back((uint32_t)S.indices.size());
    }
    S.n = (uint32_t)S.indptr.size() - 1;
    S.nnz = S.indices.size();
    return S;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: track_d_split mask_dir file.1pz...\n"; return 2; }
    std::string maskdir = argv[1];
    std::printf("sample\tmember\tsubset\tncols\tnnz\tlevel\tchunk_bytes\n");
    for (int a = 2; a < argc; ++a) {
        std::string f = argv[a];
        std::string norm = f;
        std::replace(norm.begin(), norm.end(), '\\', '/');
        auto s = norm.find_last_of('/');
        std::string base = norm.substr(s + 1);
        auto t = norm.find_last_of('/', s - 1);
        std::string par = norm.substr(t == std::string::npos ? 0 : t + 1, s - (t == std::string::npos ? 0 : t + 1));
        std::string member = base.substr(0, base.find("_counts"));
        auto mask = read_file(maskdir + "/" + par + "__" + member + ".mask");
        CanonicalMatrix M = load_1pz(f);
        M.rownames.clear(); M.colnames.clear(); M.user_kv.clear();
        if (mask.size() != M.n) { std::cerr << "mask size mismatch for " << f << "\n"; continue; }
        std::vector<uint8_t> all(M.n, 1), cells = mask, emp(M.n);
        for (uint32_t j = 0; j < M.n; ++j) emp[j] = !mask[j];
        struct Sub { const char* name; const std::vector<uint8_t>* keep; } subs[] = {{"all", &all}, {"cells", &cells}, {"empties", &emp}};
        for (auto& sb : subs) {
            CanonicalMatrix S = subset(M, *sb.keep);
            for (int level : {3, 19}) {
                auto bytes = rewrite_1pz(S, "results/.split.tmp.1pz", level, 1024);
                std::printf("%s\t%s\t%s\t%u\t%llu\t%d\t%zu\n", par.c_str(), member.c_str(), sb.name, S.n, (unsigned long long)S.nnz, level, chunk_bytes(bytes));
                std::fflush(stdout);
            }
        }
    }
    return 0;
}
