// Track D: per-chunk codec costs for the orchestrator-oracle experiment.
//
// For every 1024-column chunk of every file this encodes the chunk's VOCSC symbol streams (identical
// file-global frequency-sorted row permutation, so all methods see the same transform) with K entropy
// coders and writes one TSV row with cheap chunk features and the byte count of each method:
//   z1,z3,z9,z19   : production prefilter (bit-plane + bitmap) then zstd level L   (z3 == today)
//   rawz19         : metadata + byte-split gaps (no bit-plane prefilter), zstd 19
//   rawxz          : metadata + byte-split gaps, xz -9e
//   rawbr          : metadata + byte-split gaps, brotli-11
//   pfxz           : production prefilter, xz -9e
//   pfcm           : production prefilter, Track D context-mixing coder
// Sizes exclude the 20-byte chunk header the writer adds (identical for every method).
//   bin/track_d_oracle.exe [--no-cm] [--out results/track_d_oracle_chunks.tsv] file.1pz...
#include "common.h"
#include "track_d.h"
#include <algorithm>
#include <brotli/encode.h>
#include <cmath>
#include <iostream>
#include <lzma.h>
#include <zstd.h>

using namespace cs;
namespace pz = singlet::pz;

static size_t zstd_size(const std::vector<uint8_t>& in, int level) {
    std::vector<uint8_t> o(ZSTD_compressBound(in.size()));
    size_t z = ZSTD_compress(o.data(), o.size(), in.data(), in.size(), level);
    if (ZSTD_isError(z)) throw std::runtime_error("zstd");
    return z;
}
static size_t xz_size(const std::vector<uint8_t>& in) {
    std::vector<uint8_t> o(lzma_stream_buffer_bound(in.size()));
    size_t pos = 0;
    if (lzma_easy_buffer_encode(9 | LZMA_PRESET_EXTREME, LZMA_CHECK_NONE, nullptr, in.data(), in.size(), o.data(), &pos, o.size()) != LZMA_OK)
        throw std::runtime_error("lzma");
    return pos;
}
static size_t brotli_size(const std::vector<uint8_t>& in) {
    size_t n = BrotliEncoderMaxCompressedSize(in.size());
    std::vector<uint8_t> o(n);
    if (!BrotliEncoderCompress(11, 24, BROTLI_MODE_GENERIC, in.size(), in.data(), &n, o.data())) throw std::runtime_error("brotli");
    return n;
}

int main(int argc, char** argv) {
    std::vector<std::string> files;
    std::string out = "results/track_d_oracle_chunks.tsv";
    bool use_cm = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-cm") use_cm = false;
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else files.push_back(a);
    }
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf)
        tsv << "sample\tmember\tchunk\tnchunks\tncols\tnnz\tncols_nz\tmean_col_nnz\tq10_col_nnz\tq50_col_nnz\tq90_col_nnz\tmax_col_nnz\t"
               "frac_ones\tmean_val\tfrac_ge3\tmean_gap\tfrac_gap0\tfrac_top64\tmeta_bytes\tngaps\tsz_z1\tsz_z3\tsz_z9\tsz_z19\tsz_rawz19\tsz_rawxz\tsz_rawbr\tsz_pfxz\tsz_pfcm\n";
    for (auto& f : files) {
        CanonicalMatrix M = load_1pz(f);
        std::string norm = f;
        std::replace(norm.begin(), norm.end(), '\\', '/');
        auto s = norm.find_last_of('/');
        std::string base = norm.substr(s + 1);
        auto t = norm.find_last_of('/', s - 1);
        std::string par = norm.substr(t == std::string::npos ? 0 : t + 1, s - (t == std::string::npos ? 0 : t + 1));
        std::string member = base.substr(0, base.find("_counts"));
        std::printf("== %s %s  n=%u nnz=%llu\n", par.c_str(), member.c_str(), M.n, (unsigned long long)M.nnz);
        // file-global frequency-sort permutation (descending row nnz, ties by row id), as the writer does
        std::vector<int32_t> rnnz(M.m, 0);
        for (uint64_t k = 0; k < M.nnz; ++k) rnnz[M.indices[k]]++;
        std::vector<uint32_t> perm(M.m);
        for (uint32_t i = 0; i < M.m; ++i) perm[i] = i;
        std::stable_sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) { return rnnz[a] > rnnz[b]; });
        std::vector<uint32_t> inv(M.m);
        for (uint32_t i = 0; i < M.m; ++i) inv[perm[i]] = i;
        std::vector<uint32_t> cc(M.n);
        for (uint32_t j = 0; j < M.n; ++j) cc[j] = M.indptr[j + 1] - M.indptr[j];
        pz::CSCView mat{M.m, M.n, M.nnz, M.indptr.data(), M.indices.data(), M.data.data()};
        const int chunk_cols = 1024;
        int nc = (int)((M.n + chunk_cols - 1) / chunk_cols);
        std::vector<pz::VocscResult> res(nc);
        bool gap16 = true;
        for (int c = 0; c < nc; ++c) {
            int64_t cs_ = (int64_t)c * chunk_cols, ce = std::min<int64_t>(cs_ + chunk_cols, M.n);
            res[c] = pz::vocsc_encode_chunk(mat, inv.data(), cc.data(), cs_, ce);
            if (res[c].max_gap >= 65536) gap16 = false;
        }
        int gw = gap16 ? 2 : 4;
        size_t tot_z3 = 0;
        for (int c = 0; c < nc; ++c) {
            auto& v = res[c];
            size_t ng = v.n_gaps, msz = v.metadata.size();
            std::vector<uint8_t> raw(msz + ng * gw);
            std::memcpy(raw.data(), v.metadata.data(), msz);
            if (ng) { if (gap16) pz::byte_split_16(v.gaps.data(), ng, raw.data() + msz); else pz::byte_split_32(v.gaps.data(), ng, raw.data() + msz); }
            size_t bp = ng ? 8 * ((ng + 7) / 8) : 0;
            size_t pf_sz = msz + bp + (ng ? ng * (gw - 1) : 0);
            std::vector<uint8_t> pf(pf_sz);
            std::memcpy(pf.data(), raw.data(), msz);
            if (ng) {
                pz::bit_planes_encode(raw.data() + msz, ng, pf.data() + msz);
                if (gw > 1) std::memcpy(pf.data() + msz + bp, raw.data() + msz + ng, ng * (gw - 1));
            }
            std::vector<uint8_t> packed(pf_sz + (pf_sz + 7) / 8);
            size_t packed_sz = pz::bitmap_pack(pf.data(), pf_sz, packed.data());
            packed.resize(packed_sz);
            // features
            int64_t cs_ = (int64_t)c * chunk_cols, ce = std::min<int64_t>(cs_ + chunk_cols, M.n);
            std::vector<uint32_t> cn;
            uint64_t nnz = 0, ones = 0, ge3 = 0, vsum = 0, top64 = 0, gap0 = 0;
            double gsum = 0;
            for (int64_t j = cs_; j < ce; ++j) {
                cn.push_back(cc[j]);
                nnz += cc[j];
                for (uint32_t k = M.indptr[j]; k < M.indptr[j + 1]; ++k) {
                    uint32_t x = M.data[k];
                    ones += (x == 1); ge3 += (x >= 3); vsum += x;
                    top64 += (inv[M.indices[k]] < 64);
                }
            }
            for (size_t g = 0; g < ng; ++g) { gsum += v.gaps[g]; gap0 += (v.gaps[g] == 0); }
            std::vector<uint32_t> sc = cn;
            std::sort(sc.begin(), sc.end());
            auto q = [&](double p) { return sc.empty() ? 0u : sc[std::min<size_t>(sc.size() - 1, (size_t)(p * sc.size()))]; };
            size_t nz = 0;
            for (auto x : cn) nz += (x > 0);
            size_t s1 = zstd_size(packed, 1), s3 = zstd_size(packed, 3), s9 = zstd_size(packed, 9), s19 = zstd_size(packed, 19);
            size_t r19 = zstd_size(raw, 19), rxz = xz_size(raw), rbr = brotli_size(raw), pxz = xz_size(packed);
            size_t pcm = 0;
            if (use_cm) pcm = dcm::compress(packed.data(), packed.size(), 20).size();
            tot_z3 += s3 + 20;
            double dn = std::max<double>(1.0, (double)nnz);
            tsv << par << '\t' << member << '\t' << c << '\t' << nc << '\t' << (ce - cs_) << '\t' << nnz << '\t' << nz << '\t'
                << (double)nnz / (ce - cs_) << '\t' << q(0.1) << '\t' << q(0.5) << '\t' << q(0.9) << '\t' << (sc.empty() ? 0 : sc.back()) << '\t'
                << ones / dn << '\t' << vsum / dn << '\t' << ge3 / dn << '\t' << gsum / std::max<double>(1.0, (double)ng) << '\t'
                << gap0 / std::max<double>(1.0, (double)ng) << '\t' << top64 / dn << '\t' << msz << '\t' << ng << '\t'
                << s1 << '\t' << s3 << '\t' << s9 << '\t' << s19 << '\t' << r19 << '\t' << rxz << '\t' << rbr << '\t' << pxz << '\t' << pcm << '\n';
            tsv.flush();
        }
        std::printf("   sum of z3 chunk blobs (incl. 20 B headers) = %zu\n", tot_z3);
        std::fflush(stdout);
    }
    return 0;
}
