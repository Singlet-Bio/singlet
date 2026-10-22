// SPDX-License-Identifier: MIT
// w4_bugs 5d: writer accepts inputs the reader cannot represent / reads back differently (all with PRODUCTION write_1pz / read_1pz_buffer).
#include <singlet/pileup/pz_reader.h>
#include <cmath>
#include <cstdio>
using namespace singlet::pz;
static std::vector<uint8_t> slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {}); }
template <class F> static void probe(const char* what, F f) {
    try { f(); } catch (const PZReadError& e) { std::printf("%-58s -> PZReadError: %s\n", what, e.what()); }
    catch (const std::exception& e) { std::printf("%-58s -> %s\n", what, e.what()); }
}
int main(int argc, char**) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<int32_t> ip{0, 1, 2}, ix{0, 1};
    std::vector<uint16_t> d{1, 2};
    // 1. name with an embedded NUL: writer emits it, reader splits it into two names
    probe("rowname containing NUL", [&] {
        write_1pz<uint16_t>("w4_edge1.1pz", 2, 2, ip, ix, d, {"a\0b", "c"} /* NUL lost: const char* */, {"x", "y"}, 3, 1024, 1);
        std::string s1("a\0b", 3);
        write_1pz<uint16_t>("w4_edge1.1pz", 2, 2, ip, ix, d, {s1, "c"}, {"x", "y"}, 3, 1024, 1);
        auto b = slurp("w4_edge1.1pz"); auto r = read_1pz_buffer(b.data(), b.size());
        std::printf("rowname containing NUL: read back %zu rownames (written 2): '%s' ...\n", r.rownames.size(), r.rownames[0].c_str());
    });
    // 2. newline in a name (fine for the binary format, breaks TSV consumers)
    // 3. empty name list tail: names {"a", ""} 
    probe("trailing empty rowname", [&] {
        write_1pz<uint16_t>("w4_edge2.1pz", 2, 2, ip, ix, d, {"a", ""}, {"x", "y"}, 3, 1024, 1);
        auto b = slurp("w4_edge2.1pz"); auto r = read_1pz_buffer(b.data(), b.size());
        std::printf("trailing empty rowname: %zu names, last='%s'\n", r.rownames.size(), r.rownames.back().c_str());
    });
    // 4. key/value with empty value or duplicate-looking pairs, value containing NUL
    probe("kv value containing NUL", [&] {
        std::map<std::string, std::string> kv; kv["k"] = std::string("v\0w", 3); kv["z"] = "1";
        write_1pz<uint16_t>("w4_edge3.1pz", 2, 2, ip, ix, d, {"a", "b"}, {"x", "y"}, 3, 1024, 1, kv);
        auto b = slurp("w4_edge3.1pz"); auto r = read_1pz_buffer(b.data(), b.size());
        std::printf("kv with NUL in value: %zu entries read (written 2):", r.user_kv.size()); for (auto& p : r.user_kv) std::printf(" [%s]=[%s](%zu)", p.first.c_str(), p.second.c_str(), p.second.size()); std::printf("\n");
    });
    // 5. value > 65535 into a uint16 accumulator is clamped upstream; a uint32 value 4e9 into write_1pz
    probe("uint32 value 7e7 (vt=3)", [&] {
        std::vector<uint32_t> big{70000000u, 1};
        write_1pz<uint32_t>("w4_edge4.1pz", 2, 2, ip, ix, big, {"a", "b"}, {"x", "y"}, 3, 1024, 1);
        auto b = slurp("w4_edge4.1pz"); auto r = read_1pz_buffer(b.data(), b.size());
        std::printf("value 7e7: read back %u (vt=%d)\n", r.data[0], r.vt_code);
    });
    // 6. negative int32 value -> cast to uint32 -> huge
    if (argc > 1) probe("int32 value -1", [&] {
        std::vector<int32_t> neg{-1, 1};
        write_1pz<int32_t>("w4_edge5.1pz", 2, 2, ip, ix, neg, {"a", "b"}, {"x", "y"}, 3, 1024, 1);
        auto b = slurp("w4_edge5.1pz"); auto r = read_1pz_buffer(b.data(), b.size());
        std::printf("int32 -1: read back %u (vt=%d)\n", r.data[0], r.vt_code);
    });
    // 7. float 0.9999 / 2.5 / NaN into write_1pz<float>
    probe("float 0.9999, 2.5, NaN", [&] {
        std::vector<float> fl{0.9999f, 2.5f};
        write_1pz<float>("w4_edge6.1pz", 2, 2, ip, ix, fl, {"a", "b"}, {"x", "y"}, 3, 1024, 1);
        auto b = slurp("w4_edge6.1pz"); auto r = read_1pz_buffer(b.data(), b.size());
        std::printf("float 0.9999,2.5 -> %u,%u\n", r.data[0], r.data[1]);
        std::vector<float> nn{std::nanf(""), -3.0f};
        write_1pz<float>("w4_edge7.1pz", 2, 2, ip, ix, nn, {"a", "b"}, {"x", "y"}, 3, 1024, 1);
        b = slurp("w4_edge7.1pz"); r = read_1pz_buffer(b.data(), b.size());
        std::printf("float NaN,-3 -> %u,%u (cast of NaN/negative float to uint32 is undefined behaviour)\n", r.data[0], r.data[1]);
    });
    // 8. row index out of range (>= m): writer has no check
    probe("row index >= m (writer unchecked)", [&] {
        std::vector<int32_t> bad{0, 7};
        write_1pz<uint16_t>("w4_edge8.1pz", 2, 2, ip, bad, d, {"a", "b"}, {"x", "y"}, 3, 1024, 1);
        std::printf("writer returned for index 7 with m=2 (no exception)\n");
    });
    // 9. duplicate row index in one column, same value
    probe("duplicate (row,col) entries", [&] {
        std::vector<int32_t> ip2{0, 2, 2}, dup{1, 1}; std::vector<uint16_t> dd{3, 3};
        write_1pz<uint16_t>("w4_edge9.1pz", 2, 2, ip2, dup, dd, {"a", "b"}, {"x", "y"}, 3, 1024, 1);
        auto b = slurp("w4_edge9.1pz"); auto r = read_1pz_buffer(b.data(), b.size());
        std::printf("duplicate entries read back OK?!\n");
    });
    return 0;
}
