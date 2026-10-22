// SPDX-License-Identifier: MIT
// w4_bugs item 5c: mutation fuzz of the PRODUCTION singlet::pz::read_1pz_buffer (include/singlet/pileup/pz_reader.h).
//   w4_bugs_fuzz <file.1pz> <seconds> <seed> <log.tsv>
// Mutation classes (each iteration picks one; "fixcrc" variants then recompute the footer CRC so that the structural checks,
// not the whole-file CRC, are what has to catch the damage):
//   flip1/flip4     flip 1 / 4 random bytes                      (raw, CRC left stale)
//   flip_fix        flip 1-4 random bytes, footer CRC recomputed
//   hdr_fix         overwrite 1-8 bytes inside the 96-byte header with random / boundary values, CRC recomputed
//   trunc / trunc_fix   cut the file at a random length (fix = append a footer with a valid CRC and magic)
//   splice_fix      copy a random segment of the file over another position, CRC recomputed
//   zero_fix        zero a random range, CRC recomputed
//   meta_fix        mutate bytes inside the (compressed) metadata block, CRC recomputed
//   tlv_fix         rebuild the metadata block with a mutated decompressed TLV (valid zstd frame), CRC recomputed
//   chunk_fix       mutate the chunk-table entries, CRC recomputed
// Outcome classes: rejected (PZReadError), accepted_same (decoded == original), accepted_diff (decoded != original: wrong data
// accepted), crash (process dies: the log line written BEFORE the run identifies it), slow (> 3 s), other_exception.
#include <singlet/pileup/pz_reader.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>
using namespace singlet::pz;

static std::vector<uint8_t> rd(const std::string& p) {
    std::ifstream f(p, std::ios::binary); f.seekg(0, std::ios::end); std::vector<uint8_t> b((size_t)f.tellg()); f.seekg(0);
    f.read((char*)b.data(), (std::streamsize)b.size()); return b;
}
static void fixcrc(std::vector<uint8_t>& b) {
    if (b.size() < 112) return;
    uint32_t c = CRC32::compute(b.data(), b.size() - 16);
    std::memcpy(b.data() + b.size() - 16, &c, 4);
}
static bool same(const ReadResult& a, const ReadResult& b) {
    return a.m == b.m && a.n == b.n && a.nnz == b.nnz && a.indptr == b.indptr && a.indices == b.indices && a.data == b.data &&
           a.rownames == b.rownames && a.colnames == b.colnames && a.user_kv == b.user_kv;
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage\n"); return 2; }
    auto orig = rd(argv[1]);
    double secs = std::atof(argv[2]);
    uint64_t seed = std::strtoull(argv[3], nullptr, 10);
    std::string logp = argv[4];
    ReadResult ref = read_1pz_buffer(orig.data(), orig.size());
    PZHeader H; std::memcpy(&H, orig.data(), 96);
    std::mt19937_64 rng(seed);
    auto R = [&](uint64_t n) { return n ? rng() % n : 0; };
    // decompressed metadata TLV for tlv_fix
    ZstdDCtx dctx;
    std::vector<uint8_t> tlv = zstd_decompress(dctx.ctx, orig.data() + H.metadata_offset, H.metadata_z_sz, 1ull << 30);
    ZstdCCtx cctx;

    std::map<std::string, std::map<std::string, uint64_t>> tally;
    auto t0 = std::chrono::steady_clock::now();
    uint64_t it = 0;
    FILE* lg = std::fopen(logp.c_str(), "w");
    std::fprintf(lg, "iter\tclass\tdetail\toutcome\tms\twhat\n");
    std::vector<std::string> accepted_diff_examples;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
        ++it;
        std::vector<uint8_t> b = orig;
        static const char* cls[] = {"flip1", "flip4", "flip_fix", "hdr_fix", "trunc", "trunc_fix", "splice_fix", "zero_fix",
                                    "meta_fix", "tlv_fix", "chunk_fix"};
        std::string c = cls[R(11)];
        char detail[160] = "";
        if (c == "flip1" || c == "flip4" || c == "flip_fix") {
            int k = c == "flip1" ? 1 : c == "flip4" ? 4 : 1 + (int)R(4);
            for (int i = 0; i < k; ++i) { size_t p = R(b.size()); b[p] ^= (uint8_t)(1u << R(8)); std::snprintf(detail, 160, "pos=%zu", p); }
            if (c == "flip_fix") fixcrc(b);
        } else if (c == "hdr_fix") {
            int k = 1 + (int)R(8);
            for (int i = 0; i < k; ++i) {
                size_t p = R(96); uint8_t v;
                switch (R(4)) { case 0: v = 0; break; case 1: v = 0xFF; break; case 2: v = (uint8_t)(b[p] + 1); break; default: v = (uint8_t)R(256); }
                b[p] = v; std::snprintf(detail, 160, "hdr_off=%zu", p);
            }
            fixcrc(b);
        } else if (c == "trunc" || c == "trunc_fix") {
            size_t L = R(b.size()); b.resize(L);
            if (c == "trunc_fix" && L > 16) { PZFooter f{}; f.magic = TP1_MAGIC; f.num_chunks = H.num_chunks; b.resize(L + 16 > b.size() ? L + 16 : b.size());
                std::memcpy(b.data() + b.size() - 16, &f, 16); fixcrc(b); }
            std::snprintf(detail, 160, "len=%zu", L);
        } else if (c == "splice_fix") {
            size_t len = 1 + R(4096), src = R(b.size()), dst = R(b.size());
            for (size_t i = 0; i < len && src + i < b.size() && dst + i < b.size() - 16; ++i) b[dst + i] = orig[src + i];
            fixcrc(b); std::snprintf(detail, 160, "src=%zu dst=%zu len=%zu", src, dst, len);
        } else if (c == "zero_fix") {
            size_t len = 1 + R(512), p = R(b.size() - 16);
            for (size_t i = 0; i < len && p + i < b.size() - 16; ++i) b[p + i] = 0;
            fixcrc(b); std::snprintf(detail, 160, "pos=%zu len=%zu", p, len);
        } else if (c == "meta_fix") {
            size_t p = H.metadata_offset + R(H.metadata_z_sz); b[p] ^= (uint8_t)(1u << R(8)); fixcrc(b);
            std::snprintf(detail, 160, "pos=%zu", p);
        } else if (c == "tlv_fix") {
            auto t = tlv;
            int k = 1 + (int)R(3);
            for (int i = 0; i < k; ++i) { size_t p = R(t.size()); switch (R(3)) { case 0: t[p] = 0; break; case 1: t[p] ^= 1u << R(8); break; default: t[p] = '\n'; } }
            if (R(4) == 0) t.resize(R(t.size()));
            auto z = zstd_compress(cctx.ctx, t.data(), t.size(), 3);
            // rebuild the file: same layout, new metadata block (it is last before the footer), header fields patched
            std::vector<uint8_t> nb(orig.begin(), orig.begin() + H.metadata_offset);
            nb.insert(nb.end(), z.begin(), z.end());
            nb.resize(nb.size() + 16);
            PZHeader h2 = H; h2.metadata_z_sz = (uint32_t)z.size(); std::memcpy(nb.data(), &h2, 96);
            std::memcpy(nb.data() + nb.size() - 16, orig.data() + orig.size() - 16, 16);
            b = nb; fixcrc(b); std::snprintf(detail, 160, "tlv_edits=%d", k);
        } else {  // chunk_fix
            size_t table = 96 + H.perm_z_sz + H.ptr_z_sz;
            size_t e = R(H.num_chunks); size_t p = table + 4 * e + R(4);
            b[p] ^= (uint8_t)(1u << R(8)); fixcrc(b); std::snprintf(detail, 160, "chunk=%zu byte=%zu", e, p);
        }
        std::fprintf(lg, "%llu\t%s\t%s\tSTART\t0\t\n", (unsigned long long)it, c.c_str(), detail); std::fflush(lg);
        auto s0 = std::chrono::steady_clock::now();
        std::string outcome, what;
        try {
            ReadLimits lim; lim.max_file_bytes = 1ull << 28;
            auto r = read_1pz_buffer(b.data(), b.size(), lim);
            if (same(r, ref)) outcome = "accepted_same";
            else { outcome = "accepted_diff";
                what = "m=" + std::to_string(r.m) + " n=" + std::to_string(r.n) + " nnz=" + std::to_string(r.nnz) + " vt=" + std::to_string(r.vt_code) +
                       (r.indices != ref.indices ? " indices_differ" : "") + (r.data != ref.data ? " data_differ" : "") +
                       (r.rownames != ref.rownames ? " rownames_differ" : "") + (r.colnames != ref.colnames ? " colnames_differ" : "") +
                       (r.user_kv != ref.user_kv ? " kv_differ" : "") + (r.indptr != ref.indptr ? " indptr_differ" : ""); }
        } catch (const PZReadError& e) { outcome = "rejected"; what = e.what(); if (what.size() > 60) what.resize(60); }
        catch (const std::exception& e) { outcome = "other_exception"; what = e.what(); }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
        if (ms > 3000 && outcome != "other_exception") outcome += "+SLOW";
        tally[c][outcome]++;
        if (outcome != "rejected") { std::fprintf(lg, "%llu\t%s\t%s\t%s\t%.1f\t%s\n", (unsigned long long)it, c.c_str(), detail, outcome.c_str(), ms, what.c_str()); std::fflush(lg); }
    }
    std::fprintf(stderr, "iterations=%llu\n", (unsigned long long)it);
    for (auto& [c, m] : tally) { for (auto& [o, n] : m) std::fprintf(stderr, "%-10s %-24s %llu\n", c.c_str(), o.c_str(), (unsigned long long)n); }
    std::fclose(lg);
    return 0;
}
