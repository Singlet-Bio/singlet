// SPDX-License-Identifier: MIT
// Regression: a .1pz whose names block exceeds 256 MiB (the 13.4 M-site editing tables of the deepest Beta samples) was written
// without complaint and then refused by the reader ("zstd: frame content size exceeds the bound for this block").
// Minimal reproduction: one row, one column, one row name of 270 MB (text that compresses about 5x, like site names).
#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "singlet/pileup/pz_reader.h"

int main() {
    using namespace singlet;
    static_assert(pz::kMaxMetadataBytes > (270ull << 20) && pz::kMaxMetadataBytes > (256ull << 20), "limit must exceed the old cap");
    assert(pz::ReadLimits().max_metadata_bytes == pz::kMaxMetadataBytes);

    std::string name;
    name.reserve(280u << 20);
    char buf[40];
    for (uint64_t i = 0; name.size() < (270u << 20); ++i) {
        int k = std::snprintf(buf, sizeof buf, "chr%u:%llu:A>G;", unsigned(i % 22 + 1), (unsigned long long)(i * 37 + 1000));
        name.append(buf, size_t(k));
    }
    std::vector<int32_t> indptr{0, 1}, indices{0};
    std::vector<uint16_t> data{3};
    const std::string path = (std::filesystem::temp_directory_path() / "pz_metadata_bound.1pz").string();
    bool ok = pz::write_1pz<uint16_t>(path, 1, 1, indptr, indices, data, {name}, {"AAAC"}, 3, 1024, 1);
    assert(ok && "a names block under the shared limit must be written");

    auto r = pz::read_1pz(path);                    // default limits: this threw before the fix
    assert(r.rownames.size() == 1 && r.rownames[0] == name);
    assert(r.colnames.size() == 1 && r.colnames[0] == "AAAC");
    std::filesystem::remove(path);

    // a block over the shared limit is refused by the writer, not written as a file nobody can read
    std::string huge(size_t(pz::kMaxMetadataBytes) + 1, 'x');
    ok = pz::write_1pz<uint16_t>(path, 1, 1, indptr, indices, data, {huge}, {"AAAC"}, 1, 1024, 1);
    assert(!ok && !std::filesystem::exists(path));
    std::cout << "test_pz_metadata_bound: ok\n";
    return 0;
}
