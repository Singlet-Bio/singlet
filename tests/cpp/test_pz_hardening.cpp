// SPDX-License-Identifier: MIT
// test_pz_hardening.cpp — pz_reader.h must reject crafted input with PZReadError
// and never read or write out of bounds.
//
// A CRC32 only detects corruption, so every mutation below re-computes the
// whole-file CRC and the structural checks have to hold on their own. Run
// under -fsanitize=address,undefined (see the sanitizer CI job).

#include "singlet/pileup/pz_reader.h"
#include "singlet/pileup/pz_writer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace pz = singlet::pz;

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); ++g_fail; } } while (0)

static std::string tmp_path(const char* name) {
    const char* d = std::getenv("TMPDIR");
    if (!d) d = std::getenv("TEMP");
    if (!d) d = "/tmp";
    return std::string(d) + "/" + name;
}

static std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    std::vector<uint8_t> b(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(b.size()));
    return b;
}

// Rewrite the footer CRC so only structural validation can reject the image.
static void fix_crc(std::vector<uint8_t>& b) {
    uint32_t crc = pz::CRC32::compute(b.data(), b.size() - sizeof(pz::PZFooter));
    std::memcpy(b.data() + b.size() - sizeof(pz::PZFooter), &crc, 4);
}

// A multi-chunk image with metadata: 60 rows x 2500 cols (3 chunks of 1024).
static std::vector<uint8_t> make_valid(uint32_t m = 60, uint32_t n = 2500) {
    std::vector<int32_t> indptr{0}, indices;
    std::vector<uint16_t> data;
    std::mt19937 rng(42);
    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t i = 0; i < m; ++i) {
            if (rng() % 7 == 0) {
                indices.push_back(static_cast<int32_t>(i));
                data.push_back(static_cast<uint16_t>(1 + rng() % 40));
            }
        }
        indptr.push_back(static_cast<int32_t>(indices.size()));
    }
    std::vector<std::string> rn, cn;
    for (uint32_t i = 0; i < m; ++i) rn.push_back("G" + std::to_string(i));
    for (uint32_t j = 0; j < n; ++j) cn.push_back("C" + std::to_string(j));
    std::map<std::string, std::string> kv{{"gsm_id", "GSM1"}};
    std::string path = tmp_path("pz_hardening_valid.1pz");
    bool ok = pz::write_1pz<uint16_t>(path, m, n, indptr, indices, data, rn, cn, 3, 1024, 1, kv);
    CHECK(ok, "write_1pz");
    return slurp(path);
}

// Returns true if the image decodes, false if it was rejected with PZReadError.
// Any other exception, crash, or sanitizer report fails the run.
static bool decodes(const std::vector<uint8_t>& b) {
    try {
        pz::ReadLimits lim;
        lim.max_nnz = 1u << 24;  // keep hostile headers cheap in tests
        (void)pz::read_1pz_buffer(b.data(), b.size(), lim);
        return true;
    } catch (const pz::PZReadError&) {
        return false;
    }
}

static pz::PZHeader hdr_of(const std::vector<uint8_t>& b) {
    pz::PZHeader h; std::memcpy(&h, b.data(), sizeof h); return h;
}
static void set_hdr(std::vector<uint8_t>& b, const pz::PZHeader& h) {
    std::memcpy(b.data(), &h, sizeof h); fix_crc(b);
}

int main() {
    const auto good = make_valid();
    CHECK(decodes(good), "baseline image decodes");

    // --- version / layout gates: only the layout write_1pz emits is accepted ---
    for (uint16_t v : {0, 2, 3, 4, 5, 0xFFFF}) {
        auto b = good; auto h = hdr_of(b); h.version = v; set_hdr(b, h);
        CHECK(!decodes(b), "non-v1 version rejected");
    }
    { auto b = good; auto h = hdr_of(b); h.feature_flags = 0;    set_hdr(b, h); CHECK(!decodes(b), "feature_flags 0 rejected"); }
    { auto b = good; auto h = hdr_of(b); h.feature_flags |= 0x4; set_hdr(b, h); CHECK(!decodes(b), "unknown feature rejected"); }
    { auto b = good; auto h = hdr_of(b); h.flags |= 0x80;        set_hdr(b, h); CHECK(!decodes(b), "unknown flag rejected"); }
    { auto b = good; auto h = hdr_of(b); h.flags &= ~pz::FLAG_HAS_PERM; set_hdr(b, h); CHECK(!decodes(b), "missing perm rejected"); }
    for (uint8_t w : {0, 1, 3, 5, 8, 255}) {
        auto b = good; auto h = hdr_of(b); h.ptr_width = w; set_hdr(b, h);
        CHECK(!decodes(b), "bad ptr_width rejected");
    }
    for (uint8_t t : {0, 4, 9, 255}) {
        auto b = good; auto h = hdr_of(b); h.vt_code = t; set_hdr(b, h);
        CHECK(!decodes(b), "bad vt_code rejected");
    }

    // --- sizes that would drive huge allocations or out-of-range reads ---
    { auto b = good; auto h = hdr_of(b); h.m = 0xFFFFFFFFu;   set_hdr(b, h); CHECK(!decodes(b), "huge m rejected"); }
    { auto b = good; auto h = hdr_of(b); h.n = 0xFFFFFFFFu;   set_hdr(b, h); CHECK(!decodes(b), "huge n rejected"); }
    { auto b = good; auto h = hdr_of(b); h.nnz = ~0ull;       set_hdr(b, h); CHECK(!decodes(b), "huge nnz rejected"); }
    { auto b = good; auto h = hdr_of(b); h.nnz = 1ull << 40;  set_hdr(b, h); CHECK(!decodes(b), "nnz > 2^32 rejected"); }
    { auto b = good; auto h = hdr_of(b); h.nnz += 1;          set_hdr(b, h); CHECK(!decodes(b), "nnz mismatch rejected"); }
    { auto b = good; auto h = hdr_of(b); h.perm_z_sz = 0xFFFFFFFFu; set_hdr(b, h); CHECK(!decodes(b), "huge perm_z_sz rejected"); }
    { auto b = good; auto h = hdr_of(b); h.ptr_z_sz  = 0xFFFFFFFFu; set_hdr(b, h); CHECK(!decodes(b), "huge ptr_z_sz rejected"); }
    { auto b = good; auto h = hdr_of(b); h.perm_z_sz = 0;           set_hdr(b, h); CHECK(!decodes(b), "zero perm_z_sz rejected"); }
    { auto b = good; auto h = hdr_of(b); h.chunk_cols = 0;          set_hdr(b, h); CHECK(!decodes(b), "chunk_cols 0 rejected"); }
    { auto b = good; auto h = hdr_of(b); h.chunk_cols = 0xFFFFFFFFu; set_hdr(b, h); CHECK(!decodes(b), "huge chunk_cols rejected"); }
    { auto b = good; auto h = hdr_of(b); h.chunk_cols = 512;        set_hdr(b, h); CHECK(!decodes(b), "chunk geometry mismatch rejected"); }
    { auto b = good; auto h = hdr_of(b); h.metadata_offset = 0xFFFFFFFFFFFFFFF0ull; set_hdr(b, h); CHECK(!decodes(b), "metadata offset overflow rejected"); }
    { auto b = good; auto h = hdr_of(b); h.metadata_offset = b.size();               set_hdr(b, h); CHECK(!decodes(b), "metadata offset past end rejected"); }
    { auto b = good; auto h = hdr_of(b); h.metadata_z_sz = 0xFFFFFFFFu;               set_hdr(b, h); CHECK(!decodes(b), "metadata size past end rejected"); }
    { auto b = good; auto h = hdr_of(b); h.m += 1; set_hdr(b, h); CHECK(!decodes(b), "m inconsistent with perm block rejected"); }

    // --- file-level damage ---
    { auto b = good; b[b.size() - 1] ^= 0xFF; CHECK(!decodes(b), "bad footer magic rejected"); }
    { auto b = good; b[200] ^= 0x01;          CHECK(!decodes(b), "CRC mismatch rejected"); }
    { std::vector<uint8_t> b; CHECK(!decodes(b), "empty buffer rejected"); }

    // --- every truncation, with and without a repaired CRC ---
    for (size_t len = 0; len < good.size(); len += (len < 400 ? 1 : 7)) {
        std::vector<uint8_t> b(good.begin(), good.begin() + static_cast<std::ptrdiff_t>(len));
        (void)decodes(b);                          // must not crash
        if (len >= sizeof(pz::PZFooter)) { fix_crc(b); (void)decodes(b); }
    }

    // --- single-byte mutations anywhere in the file, CRC repaired ---
    {
        const auto small = make_valid(12, 300);
        size_t accepted = 0, rejected = 0;
        for (size_t i = 0; i + sizeof(pz::PZFooter) < small.size(); ++i) {
            for (uint8_t x : {uint8_t(0x01), uint8_t(0x80), uint8_t(0xFF)}) {
                auto b = small; b[i] ^= x; fix_crc(b);
                (decodes(b) ? accepted : rejected)++;
            }
        }
        std::printf("single-byte mutations: %zu rejected, %zu decoded cleanly\n", rejected, accepted);
        CHECK(rejected > 0, "some mutations are rejected");
    }

    // --- seeded random mutation fuzz, CRC repaired ---
    {
        std::mt19937_64 rng(0xC0FFEE);
        const auto base = make_valid(30, 1100);
        for (int it = 0; it < 4000; ++it) {
            auto b = base;
            int edits = 1 + static_cast<int>(rng() % 6);
            for (int e = 0; e < edits; ++e) {
                size_t pos = rng() % (b.size() - sizeof(pz::PZFooter));
                switch (rng() % 3) {
                    case 0: b[pos] = static_cast<uint8_t>(rng()); break;
                    case 1: b[pos] ^= static_cast<uint8_t>(1u << (rng() % 8)); break;
                    default: if (pos + 4 <= b.size() - sizeof(pz::PZFooter)) { uint32_t v = static_cast<uint32_t>(rng()); std::memcpy(&b[pos], &v, 4); }
                }
            }
            fix_crc(b);
            (void)decodes(b);
        }
    }

    if (g_fail) { std::fprintf(stderr, "%d check(s) failed\n", g_fail); return 1; }
    std::printf("pz_reader hardening tests PASSED\n");
    return 0;
}
