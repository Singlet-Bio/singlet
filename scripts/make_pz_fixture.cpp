// SPDX-License-Identifier: MIT
// Writes the small .1pz the R decode tests use when no cluster fixture is available:
//   g++ -std=c++17 -Iinclude scripts/make_pz_fixture.cpp -lzstd -o make_pz_fixture
//   ./make_pz_fixture r/tests/testthat/fixtures/sample/gene_counts.1pz
// 150 genes x 90 cells, about 8% non-zero, deterministic.
#include "singlet/pileup/pz_reader.h"
#include "singlet/pileup/pz_writer.h"

#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s out.1pz\n", argv[0]); return 2; }
    const uint32_t m = 150, n = 90;
    std::mt19937 rng(20261016);
    std::vector<int32_t> indptr{0}, indices;
    std::vector<uint16_t> data;
    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t i = 0; i < m; ++i) {
            if (rng() % 100 < 8) { indices.push_back(static_cast<int32_t>(i)); data.push_back(static_cast<uint16_t>(1 + rng() % 30)); }
        }
        indptr.push_back(static_cast<int32_t>(indices.size()));
    }
    std::vector<std::string> rn, cn;
    char buf[32];
    for (uint32_t i = 0; i < m; ++i) { std::snprintf(buf, sizeof buf, "ENSG%011u", 100000 + i * 7); rn.emplace_back(buf); }
    const char* acgt = "ACGT";
    for (uint32_t j = 0; j < n; ++j) { std::string b; for (int k = 0; k < 16; ++k) b += acgt[(rng() >> 3) & 3]; cn.push_back(b + "-1"); }
    std::map<std::string, std::string> kv{{"gsm_id", "GSM1000001"}, {"gse_id", "GSE1000"}, {"organism", "Homo sapiens"}, {"protocol", "10x"}};
    bool ok = singlet::pz::write_1pz<uint16_t>(argv[1], m, n, indptr, indices, data, rn, cn, 3, 1024, 1, kv);
    return ok ? 0 : 1;
}
