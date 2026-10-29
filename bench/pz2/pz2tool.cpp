// SPDX-License-Identifier: MIT
// pz2tool: command line helper for the independent Python reference decoder (bench/pz2/ref_decode.py) and for ad-hoc work.
//   pz2tool enc  in.1pz out.pz2 [chunk_cols] [codec 0|1|2] [lanes]      encode a legacy matrix (names inline, kv, digest)
//   pz2tool synth NAME out.pz2 out.csc                                  synthetic edge-case matrix + the expected CSC
//        NAME: escapes | zeros | f32 | tiny | empty
//   pz2tool info file.pz2                                               header summary
// out.csc layout: u32 m, u32 n, u64 nnz, u8 value_type, u8 f32flag, then indptr u32[n+1], indices u32[nnz], values u32[nnz] (f32: bit patterns)
#include <singlet/pileup/pz_reader.h>
#include <singlet/pz2/pz2.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>

using namespace singlet::pz2;

static void write_file(const char* p, const void* d, size_t n) {
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char*>(d), std::streamsize(n));
}
static std::vector<uint8_t> read_file(const char* p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void write_csc(const char* path, uint32_t m, uint32_t n, const std::vector<uint32_t>& ptr, const std::vector<uint32_t>& idx, const std::vector<uint32_t>& val, uint8_t vt) {
    std::vector<uint8_t> o;
    detail::put_u32(o, m); detail::put_u32(o, n); detail::put_u64(o, idx.size());
    o.push_back(vt); o.push_back(0);
    for (uint32_t v : ptr) detail::put_u32(o, v);
    for (uint32_t v : idx) detail::put_u32(o, v);
    for (uint32_t v : val) detail::put_u32(o, v);
    write_file(path, o.data(), o.size());
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: pz2tool enc|synth|info ...\n"); return 2; }
    std::string cmd = argv[1];
    try {
        if (cmd == "enc") {
            auto r = singlet::pz::read_1pz(argv[2]);
            CSC c;
            c.m = r.m; c.n = r.n; c.nnz = r.nnz; c.indptr = r.indptr.data(); c.indices = r.indices.data(); c.data = r.data.data();
            EncodeOptions o;
            o.row_names.blob = join_names(r.rownames);
            o.col_names.blob = join_names(r.colnames);
            o.kv = KvMap(r.user_kv.begin(), r.user_kv.end());
            if (r.vt_code >= 1 && r.vt_code <= 3) o.value_type = ValueType(r.vt_code);
            o.zstd_level = 3;
            if (argc > 4) o.chunk_cols = uint32_t(std::atoi(argv[4]));
            if (argc > 5) o.codec = CodecId(std::atoi(argv[5]));
            if (argc > 6) o.lanes = uint8_t(std::atoi(argv[6]));
            auto b = encode_matrix(c, o);
            write_file(argv[3], b.data(), b.size());
            std::printf("%zu bytes\n", b.size());
            return 0;
        }
        if (cmd == "synth") {
            std::string name = argv[2];
            std::mt19937_64 g(42);
            uint32_t m = 1000, n = 50;
            std::vector<uint32_t> ptr{0}, idx, val;
            EncodeOptions o;
            o.codec = CodecId::Auto;
            o.zstd_level = 3;
            if (name == "escapes") {   // m = 2^21 - 1: gaps wider than 16 bits; values up to 2^32 - 1; zeros; empty columns
                m = kRansMaxRows; n = 30; o.codec = CodecId::Rans; o.chunk_cols = 7; o.lanes = 16;
                for (uint32_t j = 0; j < n; ++j) {
                    std::vector<uint32_t> rows;
                    uint32_t k = (j % 5 == 0) ? 0 : 3 + uint32_t(g() % 40);
                    for (uint32_t i = 0; i < k; ++i) rows.push_back(uint32_t(g() % m));
                    if (j % 3 == 0) { rows.push_back(0); rows.push_back(m - 1); }
                    std::sort(rows.begin(), rows.end());
                    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
                    for (uint32_t r : rows) {
                        uint32_t v = g() % 6 == 0 ? uint32_t(g()) : (g() % 5 == 0 ? 0u : 1u + uint32_t(g() % 4));
                        idx.push_back(r); val.push_back(v);
                    }
                    ptr.push_back(uint32_t(idx.size()));
                }
                val[0] = 0xFFFFFFFFu;
            } else if (name == "zeros") {
                m = 300; n = 40; o.codec = CodecId::Generic; o.chunk_cols = 16;
                for (uint32_t j = 0; j < n; ++j) {
                    for (uint32_t i = 0; i < m; i += 1 + uint32_t(g() % 9)) { idx.push_back(i); val.push_back(uint32_t(g() % 4)); }
                    ptr.push_back(uint32_t(idx.size()));
                }
            } else if (name == "tiny") {
                m = 3; n = 2; idx = {0, 2, 1}; val = {5, 1, 70000}; ptr = {0, 2, 3};
                o.codec = CodecId::Rans;
            } else if (name == "empty") {
                m = 5; n = 0;
            } else if (name == "f32") {
                m = 200; n = 20; o.chunk_cols = 5;
                std::vector<float> fv;
                for (uint32_t j = 0; j < n; ++j) {
                    for (uint32_t i = 0; i < m; i += 1 + uint32_t(g() % 7)) { idx.push_back(i); fv.push_back(float(g() % 1000) * 0.25f + (i % 11 == 0 ? 0.1f : 0.0f)); }
                    ptr.push_back(uint32_t(idx.size()));
                }
                for (float f : fv) { uint32_t b; std::memcpy(&b, &f, 4); val.push_back(b); }
                CSC c;
                c.m = m; c.n = n; c.nnz = idx.size(); c.indptr = ptr.data(); c.indices = idx.data(); c.data = fv.data(); c.data_type = DType::F32;
                std::vector<std::string> rn, cn;
                for (uint32_t i = 0; i < m; ++i) rn.push_back("g" + std::to_string(i));
                for (uint32_t j = 0; j < n; ++j) cn.push_back("c" + std::to_string(j));
                o.row_names.blob = join_names(rn); o.col_names.blob = join_names(cn);
                auto b = encode_matrix(c, o);
                write_file(argv[3], b.data(), b.size());
                write_csc(argv[4], m, n, ptr, idx, val, 4);
                return 0;
            }
            std::vector<std::string> rn, cn;
            for (uint32_t i = 0; i < std::min<uint32_t>(m, 5000); ++i) rn.push_back("g" + std::to_string(i));
            for (uint32_t i = std::min<uint32_t>(m, 5000); i < m; ++i) rn.push_back("g" + std::to_string(i));
            for (uint32_t j = 0; j < n; ++j) cn.push_back("c" + std::to_string(j));
            o.row_names.kind = NamesKind::Pack; o.row_names.pack_id = PackId{{1, 2, 3, 4, 5, 6, 7, 8}};   // logical names only feed the digest
            o.row_names.blob = join_names(rn);
            o.col_names.blob = join_names(cn);
            o.kv = {{"k1", "v1"}, {"alpha", ""}};
            CSC c;
            c.m = m; c.n = n; c.nnz = idx.size(); c.indptr = ptr.data(); c.indices = idx.data(); c.data = val.data(); c.data_type = DType::U32;
            auto b = encode_matrix(c, o);
            write_file(argv[3], b.data(), b.size());
            uint32_t mx = 0;
            for (uint32_t v : val) mx = std::max(mx, v);
            write_csc(argv[4], m, n, ptr, idx, val, uint8_t(mx <= 255 ? 1 : mx <= 65535 ? 2 : 3));
            return 0;
        }
        if (cmd == "info") {
            auto b = read_file(argv[2]);
            Header h = read_header(b.data(), b.size());
            std::printf("codec %d m %u n %u nnz %llu value_type %d chunks %u chunk_cols %u lanes %d pb %d/%d kind %d digest %s total %llu prefix %llu\n", int(h.codec), h.m, h.n,
                        (unsigned long long)h.nnz, int(h.value_type), h.n_chunks, h.chunk_cols, h.lanes, h.prob_bits, h.value_prob_bits, h.ctx_kind, h.digest.hex().c_str(),
                        (unsigned long long)h.total_len, (unsigned long long)h.prefix_len());
            return 0;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 2;
}
