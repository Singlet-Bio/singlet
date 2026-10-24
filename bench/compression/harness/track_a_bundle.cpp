// Track A deliverable 1 (bundle level): what would the model-side treatments save on a whole .singlet bundle?
//
//   track_a_bundle --out results/track_a_bundle_whatif.tsv member1.1pz member2.1pz ...
//
// For every matrix: bytes of the perm block / ptr block / colsums / metadata as stored, the cost of the same
// information as per-row nnz counts (decoder re-derives the permutation), the cost of the barcodes as whitelist
// indices, and hashes of the row-name list and column-name list so duplicates across members can be counted.
#include "common.h"
#include "track_a.h"
#include <iostream>

using namespace cs;
using namespace cs::ta;
using singlet::pz::PZHeader;

static std::string base_name(const std::string& p) { auto s = p.find_last_of("/\\"); return s == std::string::npos ? p : p.substr(s + 1); }

int main(int argc, char** argv) {
    std::string out = "results/track_a_bundle_whatif.tsv";
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; if (a == "--out" && i + 1 < argc) out = argv[++i]; else files.push_back(a); }
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf) tsv << "file\tbytes\tm\tn\tnnz\tperm_z\tptr_z\tcolsums_z\tmeta_z\trowcounts_best\twhitelist_idx_bytes\tpack2bit_bytes\trownames_hash\tcolnames_hash\trownames_raw\tcolnames_raw\n";
    for (auto& f : files) {
        auto file = read_file(f);
        PZHeader h; std::memcpy(&h, file.data(), sizeof(h));
        if (h.m == 0 || h.n == 0 || h.nnz == 0) continue;
        auto M = load_1pz(f);
        Prepared P;
        P.m = M.m;
        P.rnnz.assign(M.m, 0);
        for (auto r : M.indices) P.rnnz[r]++;
        P.perm = perm_from_rnnz(P.rnnz);
        PermStats ps; enc_perm(P, 1, &ps);
        std::vector<uint8_t> t; std::string wn;
        size_t wl = bc_whitelist(M.colnames, t, wn) ? t.size() : 0;
        std::vector<uint8_t> t2;
        size_t p2 = bc_pack2(M.colnames, t2) ? t2.size() : 0;
        tsv << base_name(f) << '\t' << file.size() << '\t' << M.m << '\t' << M.n << '\t' << M.nnz << '\t' << h.perm_z_sz << '\t' << h.ptr_z_sz << '\t' << h.colsums_z_sz << '\t' << h.metadata_z_sz
            << '\t' << ps.bytes << '\t' << wl << '\t' << p2 << '\t' << names_hash(M.rownames) << '\t' << names_hash(M.colnames) << '\t' << join_nul(M.rownames).size() << '\t' << join_nul(M.colnames).size() << '\n';
        tsv.flush();
        std::printf("%s: perm %u -> rowcounts %zu ; barcodes whitelist %zu\n", base_name(f).c_str(), h.perm_z_sz, ps.bytes, wl);
    }
    return 0;
}
