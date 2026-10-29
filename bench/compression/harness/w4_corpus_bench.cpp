// SPDX-License-Identifier: MIT
// w4_corpus driver (from harness/bench.cpp): ref_1pz_v1_L3, a_names_perm_model, b_rans16 on arbitrary .1pz files, with the
// per-component accounting needed for the container / payload split:
//   payload_bytes = what the codec spends on the count matrix itself
//       ref   : chunks_z + ptr_z + 4*chunks of the ORIGINAL file header (Track B's "today payload")
//       a_*   : chunk payload + chunk headers + chunk table (ptr block omitted, derived)
//       b_*   : tb::last_breakdown().payload (column counts + coder tables + rANS streams); row counts are side_bytes
//   side_bytes   = permutation (ref) / per-row nnz counts (a_*, b_*)
//   names_bytes  = row + column names + kv as the codec stores them
// usage: w4_corpus_bench [--out tsv] [--codec a,b] file.1pz...      (set TRACK_A_MODELS=results/w4_corpus_models)
#include "common.h"
#include "ref_zstd_csc.h"
#include "track_a.h"
#include "track_b.h"
#include <algorithm>
#include <iostream>
#include <sstream>

using namespace cs;

static std::string base_name(const std::string& p) { auto s = p.find_last_of("/\\"); return s == std::string::npos ? p : p.substr(s + 1); }
static std::string parent_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    if (s == std::string::npos || s == 0) return "";
    auto t = p.find_last_of("/\\", s - 1);
    size_t start = (t == std::string::npos) ? 0 : t + 1;
    return p.substr(start, s - start);
}

int main(int argc, char** argv) {
    std::vector<std::string> files, want;
    std::string out = "results/w4_corpus_bench.tsv";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--codec" && i + 1 < argc) { std::stringstream ss(argv[++i]); std::string t; while (std::getline(ss, t, ',')) want.push_back(t); }
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else files.push_back(a);
    }
    if (want.empty()) want = {"ref_1pz_v1_L3", "a_names_perm_model", "b_rans16"};
    bool new_file = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (new_file)
        tsv << "sample\tmember\tcodec\torig_bytes\tenc_bytes\tnnz\tm\tn\ttotal_bits_nnz\tpayload_bytes\tpayload_bits_nnz\tside_bytes\tnames_bytes\t"
               "ptr_bytes\tcolsums_bytes\tother_bytes\trow_mode\tbc_mode\trow_names_bytes\tcol_names_bytes\tenc_ms\tdec_ms\tmatrix_ok\n";
    for (auto& f : files) {
        try {
            auto orig = read_file(f);
            // today's component sizes from the 96-byte header (little endian):
            // 0 magic u32, 4 ver u16, 6 vt u8, 7 flags u8, 8 m u32, 12 n u32, 16 nnz u64, 24 ptr_width u8, 25 level u8, 26 pad u16,
            // 28 nch u32, 32 perm_z u32, 36 ptr_z u32, 40 chunk_cols u32, 44 feature_flags u32, 48 meta_off u64, 56 meta_z u32, 60 colsums_z u32
            auto rd32 = [&](size_t o) { uint32_t v; std::memcpy(&v, &orig[o], 4); return v; };
            uint64_t nch = rd32(28), perm_z = rd32(32), ptr_z = rd32(36), meta_z = rd32(56), colsums_z = rd32(60);
            uint64_t chunks_z = orig.size() - 96 - perm_z - ptr_z - meta_z - colsums_z - 4 * nch - 16;
            CanonicalMatrix M = load_1pz(f);
            const double nnz = std::max<double>(1.0, (double)M.nnz);
            std::printf("== %s  m=%u n=%u nnz=%llu  %zu bytes\n", f.c_str(), M.m, M.n, (unsigned long long)M.nnz, orig.size());
            for (auto& cn : want) {
                std::vector<uint8_t> enc; CanonicalMatrix D;
                double enc_ms = 0, dec_ms = 0;
                size_t payload = 0, side = 0, names = 0, ptr = 0, colsums = 0, rowb = 0, colb = 0; int rowmode = -1, bcmode = -1;
                if (cn == "ref_1pz_v1_L3") {
                    auto c = make_ref_identity_1pz();
                    Timer te; enc = c.encode(M); enc_ms = te.ms();
                    Timer td; D = c.decode(enc); dec_ms = td.ms();
                    payload = chunks_z + ptr_z + 4 * nch; side = perm_z; names = meta_z; ptr = ptr_z; colsums = colsums_z;
                } else if (cn == "a_names_perm_model") {
                    ta::Plan pl; pl.name = cn; pl.names_mode = 1; pl.perm_mode = 1; pl.derive_ptr = true;
                    ta::Stats st;
                    Timer te; enc = ta::encode_plan(M, pl, &st); enc_ms = te.ms();
                    Timer td; D = ta::decode_plan(enc); dec_ms = td.ms();
                    payload = st.chunk_payload + st.chunk_hdr + st.ctable; side = st.perm; names = st.names; ptr = st.ptr;
                    rowb = st.ns.row_bytes; colb = st.ns.col_bytes; rowmode = st.ns.row_mode; bcmode = st.ns.bc_mode;
                } else if (cn == "b_rans16") {
                    tb::RansOpts o;
                    auto c = tb::make_b_rans("b_rans16", o);
                    Timer te; enc = c.encode(M); enc_ms = te.ms();
                    auto bd = tb::last_breakdown();
                    payload = bd.payload; side = bd.side; names = bd.names;
                    Timer td; D = c.decode(enc); dec_ms = td.ms();
                } else { std::fprintf(stderr, "unknown codec %s\n", cn.c_str()); continue; }
                bool ok = (D == M);
                size_t other = enc.size() > payload + side + names + ptr + colsums ? enc.size() - payload - side - names - ptr - colsums : 0;
                std::printf("  %-20s %10zu B  total %7.2f b/nnz  payload %7.2f b/nnz  side %zu names %zu  %s\n", cn.c_str(), enc.size(),
                            enc.size() * 8.0 / nnz, payload * 8.0 / nnz, side, names, ok ? "matrix=OK" : "matrix=FAIL");
                tsv << parent_name(f) << '\t' << base_name(f) << '\t' << cn << '\t' << orig.size() << '\t' << enc.size() << '\t' << M.nnz << '\t' << M.m
                    << '\t' << M.n << '\t' << enc.size() * 8.0 / nnz << '\t' << payload << '\t' << payload * 8.0 / nnz << '\t' << side << '\t' << names
                    << '\t' << ptr << '\t' << colsums << '\t' << other << '\t' << rowmode << '\t' << bcmode << '\t' << rowb << '\t' << colb << '\t'
                    << enc_ms << '\t' << dec_ms << '\t' << (ok ? 1 : 0) << '\n';
                tsv.flush();
            }
        } catch (const std::exception& e) {
            std::printf("!! %s: %s\n", f.c_str(), e.what());
            tsv << parent_name(f) << '\t' << base_name(f) << "\tERROR\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t-1\t-1\t0\t0\t0\t0\t0\n";
            tsv.flush();
        }
    }
    return 0;
}
