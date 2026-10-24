// Track A: what does each stage of the writer's prefilter earn?
//
//   track_a_prefilter [--out results/track_a_prefilter.tsv] file.1pz ...
//
// Per chunk (chunk_cols 1024) the writer builds  raw = [varint metadata][low gap bytes][high gap bytes]
// then  prefilter = [metadata][8 bit planes of the low gap bytes][high gap bytes]  then  bitmap zero filter  then zstd.
// Variants, all compressed with zstd-3 and zstd-19 (sum over chunks):
//   P  production                       : bit planes + bitmap filter
//   NB no bit planes, bitmap filter     : [metadata][low][high] -> bitmap -> z
//   NF bit planes, no bitmap filter     : prefilter -> z
//   R  raw                              : [metadata][low][high] -> z
//   NS no byte split at all             : [metadata][gaps as u16 LE] -> z
#include "common.h"
#include "track_a.h"
#include <iostream>

using namespace cs;
using namespace cs::ta;

static std::string base_name(const std::string& p) { auto s = p.find_last_of("/\\"); return s == std::string::npos ? p : p.substr(s + 1); }
static std::string parent_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    if (s == std::string::npos || s == 0) return "";
    auto t = p.find_last_of("/\\", s - 1);
    return p.substr(t == std::string::npos ? 0 : t + 1, s - (t == std::string::npos ? 0 : t + 1));
}

int main(int argc, char** argv) {
    std::string out = "results/track_a_prefilter.tsv";
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; if (a == "--out" && i + 1 < argc) out = argv[++i]; else files.push_back(a); }
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf) tsv << "sample\tmember\tnnz\tvariant\tz3_bytes\tz19_bytes\tz3_bits_per_nnz\tz19_bits_per_nnz\n";
    EC z3 = ec_zstd(3), z19 = ec_zstd(19);
    for (auto& f : files) {
        auto M = load_1pz(f);
        const int chunk_cols = 1024;
        std::vector<uint32_t> rnnz(M.m, 0);
        for (auto r : M.indices) rnnz[r]++;
        auto perm = perm_from_rnnz(rnnz);
        std::vector<uint32_t> inv(M.m);
        for (uint32_t i = 0; i < M.m; ++i) inv[perm[i]] = i;
        std::vector<uint32_t> cc(M.n);
        for (uint32_t j = 0; j < M.n; ++j) cc[j] = M.indptr[j + 1] - M.indptr[j];
        pz::CSCView mat{M.m, M.n, M.nnz, M.indptr.data(), M.indices.data(), M.data.data()};
        const int nc = (int(M.n) + chunk_cols - 1) / chunk_cols;
        std::map<std::string, std::pair<size_t, size_t>> tot;   // variant -> (z3, z19)
        std::vector<uint8_t> tmp, packed, z;
        for (int c = 0; c < nc; ++c) {
            int64_t cs = int64_t(c) * chunk_cols, ce = std::min<int64_t>(cs + chunk_cols, M.n);
            auto v = pz::vocsc_encode_chunk(mat, inv.data(), cc.data(), cs, ce);
            size_t ng = v.n_gaps, msz = v.metadata.size();
            const int gw = v.max_gap >= 65536 ? 4 : 2;   // per-chunk width is enough for this comparison
            std::vector<uint8_t> split(ng * gw);
            if (ng) { if (gw == 2) pz::byte_split_16(v.gaps.data(), ng, split.data()); else pz::byte_split_32(v.gaps.data(), ng, split.data()); }
            auto add = [&](const char* name, const std::vector<uint8_t>& bytes) {
                tot[name].first += z_compress(z3, bytes).size();
                tot[name].second += z_compress(z19, bytes).size();
            };
            // R: raw
            std::vector<uint8_t> raw(msz + ng * gw);
            std::memcpy(raw.data(), v.metadata.data(), msz);
            if (ng) std::memcpy(raw.data() + msz, split.data(), ng * gw);
            add("R raw (byte split only)", raw);
            // NB: raw -> bitmap
            packed.resize(raw.size() + (raw.size() + 7) / 8);
            size_t psz = pz::bitmap_pack(raw.data(), raw.size(), packed.data());
            packed.resize(psz);
            add("NB no bit planes, bitmap filter", packed);
            // production prefilter
            size_t bp = ng ? 8 * ((ng + 7) / 8) : 0;
            std::vector<uint8_t> pf(msz + bp + (ng ? ng * (gw - 1) : 0));
            std::memcpy(pf.data(), v.metadata.data(), msz);
            if (ng) {
                pz::bit_planes_encode(split.data(), ng, pf.data() + msz);
                if (gw > 1) std::memcpy(pf.data() + msz + bp, split.data() + ng, ng * (gw - 1));
            }
            add("NF bit planes, no bitmap filter", pf);
            packed.resize(pf.size() + (pf.size() + 7) / 8);
            psz = pz::bitmap_pack(pf.data(), pf.size(), packed.data());
            packed.resize(psz);
            add("P production (bit planes + bitmap)", packed);
            // NS: metadata + gaps as little-endian u16/u32 interleaved (no byte split)
            std::vector<uint8_t> ns(msz + ng * 4);
            std::memcpy(ns.data(), v.metadata.data(), msz);
            if (ng) std::memcpy(ns.data() + msz, v.gaps.data(), ng * 4);
            add("NS no byte split (u32 LE gaps)", ns);
        }
        std::printf("== %s/%s nnz=%llu\n", parent_name(f).c_str(), base_name(f).c_str(), (unsigned long long)M.nnz);
        for (auto& [k, v] : tot) {
            std::printf("   %-40s z3 %10zu (%6.3f b/nnz)   z19 %10zu (%6.3f b/nnz)\n", k.c_str(), v.first, v.first * 8.0 / M.nnz, v.second, v.second * 8.0 / M.nnz);
            tsv << parent_name(f) << '\t' << base_name(f) << '\t' << M.nnz << '\t' << k << '\t' << v.first << '\t' << v.second << '\t' << v.first * 8.0 / M.nnz << '\t' << v.second * 8.0 / M.nnz << '\n';
        }
        tsv.flush();
    }
    return 0;
}
