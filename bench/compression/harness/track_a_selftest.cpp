// Track A self test: every container option must roundtrip; ta::prepare()+zstd-3 must reproduce the production
// chunk bytes exactly (for files written by the same zstd build as the one linked here).
//
//   track_a_selftest file.1pz ...
#include "common.h"
#include "track_a.h"
#include <iostream>

using namespace cs;
using namespace cs::ta;

int main(int argc, char** argv) {
    int fails = 0, runs = 0;
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        auto M = load_1pz(f);
        auto orig = read_file(f);
        std::printf("== %s (m=%u n=%u nnz=%llu)\n", f.c_str(), M.m, M.n, (unsigned long long)M.nnz);
        // 1. production chunk bytes
        {
            Prepared P = prepare(M, 1024, nullptr, 2);
            auto blobs = entropy_chunks(P, ec_zstd(3));
            singlet::pz::PZHeader h; std::memcpy(&h, orig.data(), sizeof(h));
            size_t off = sizeof(h) + h.perm_z_sz + h.ptr_z_sz + 4ull * h.num_chunks;
            bool same = P.nc == int(h.num_chunks);
            for (auto& b : blobs) { if (off + b.size() > orig.size() || std::memcmp(orig.data() + off, b.data(), b.size()) != 0) same = false; off += b.size(); }
            // production permutation too
            std::vector<uint8_t> raw(size_t(M.m) * 4);
            ZSTD_decompress(raw.data(), raw.size(), orig.data() + sizeof(h), h.perm_z_sz);
            std::vector<uint32_t> perm(M.m);
            std::memcpy(perm.data(), raw.data(), raw.size());
            std::printf("   prepare()+zstd3 chunk bytes identical to the file's: %s ; perm_from_rnnz == stored perm: %s\n", same ? "YES" : "no (file written by another zstd build?)",
                        perm == P.perm ? "YES" : "NO");
            if (perm != P.perm) ++fails;
        }
        auto check = [&](const Plan& pl, bool names_in_matrix) {
            auto enc = encode_plan(M, pl);
            auto D = decode_plan(enc);
            CanonicalMatrix ref = M;
            if (!names_in_matrix) { ref.rownames.clear(); ref.colnames.clear(); }
            bool ok = (D == ref);
            ++runs; if (!ok) ++fails;
            std::printf("   %-44s %10zu B  %s\n", pl.name.c_str(), enc.size(), ok ? "roundtrip OK" : "ROUNDTRIP FAIL");
        };
        for (int pm : {0, 1})
            for (int bc : {-1, 0, 1, 2, 3}) {
                Plan p; p.names_mode = 1; p.perm_mode = pm; p.force_bc = bc;
                p.name = "names_model bc=" + std::to_string(bc) + " perm_mode=" + std::to_string(pm);
                check(p, true);
            }
        { Plan p; p.names_mode = 0; p.perm_mode = 0; p.name = "production names+perm"; check(p, true); }
        for (auto ec : {ec_zstd(1), ec_zstd(19), ec_zstd(22, 27, true), ec_xz(), ec_brotli(11, 24), ec_bzip2(), ec_store()}) {
            Plan p; p.names_mode = 2; p.ec = ec; p.name = "ec " + ec.label; check(p, false);
        }
        for (int cc : {64, 256, 4096, int(M.n)}) {
            Plan p; p.names_mode = 2; p.chunk_cols = cc; p.name = "chunk_cols " + std::to_string(cc); check(p, false);
        }
        { Plan p; p.names_mode = 1; p.perm_mode = 1; p.derive_ptr = true; p.name = "names model + rowcounts + derived ptr"; check(p, true); }
        { Plan p; p.names_mode = 2; p.derive_ptr = true; p.chunk_cols = 64; p.name = "derived ptr, chunk_cols 64"; check(p, false); }
        { Plan p; p.names_mode = 2; p.ec = ec_zstd(3); p.ec.dict = load_dict_for(M); p.name = p.ec.dict ? "zstd3 + trained dictionary" : "dictionary (none for this file)"; if (p.ec.dict) check(p, false); }
    }
    std::printf("\n%d runs, %d failures\n", runs, fails);
    return fails ? 1 : 0;
}
