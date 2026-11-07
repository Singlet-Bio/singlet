// SPDX-License-Identifier: MIT
// tests/cpp/test_min_sketch_mmap.cpp
//
// MinSketchIndex::load() maps the index file read-only (shared across processes) and
// merge() is a union of sorted runs instead of a copy. Both must answer every query
// exactly like the private, copied index did.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "singlet/pileup/nonhost/min_sketch.h"
#include "test_harness.h"

#ifdef _WIN32
static int setenv(const char* k, const char* v, int) { return _putenv_s(k, v); }
static int unsetenv(const char* k) { return _putenv_s(k, ""); }
#endif

using singlet::nonhost::MinSketchIndex;

namespace {

std::string random_seq(std::mt19937_64& rng, size_t n) {
    static const char b[] = "ACGT";
    std::string s(n, 'A');
    for (auto& c : s) c = b[rng() % 4];
    return s;
}

using Hits = std::unordered_map<uint32_t, uint32_t>;

bool same(const std::vector<Hits>& a, const std::vector<Hits>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

}  // namespace

int main() {
    std::mt19937_64 rng(11);
    std::vector<std::pair<std::string, uint32_t>> va, vb;
    for (uint32_t sp = 0; sp < 30; ++sp) va.push_back({random_seq(rng, 4000), sp});
    for (uint32_t sp = 30; sp < 60; ++sp) vb.push_back({random_seq(rng, 4000), sp});
    // a few shared sequences so one hash maps to species in both runs
    for (uint32_t i = 0; i < 5; ++i) { vb.push_back({va[i].first, 100 + i}); }

    auto a = MinSketchIndex::build_from_sequences(va);
    auto b = MinSketchIndex::build_from_sequences(vb);
    a.set_species_name(3, "speciesA3");
    b.set_species_name(33, "speciesB33");
    const std::string pa = "test_min_sketch_a.idx", pb = "test_min_sketch_b.idx";
    a.save(pa);
    b.save(pb);

    // queries: reads cut from both sets plus noise
    std::vector<std::string> reads;
    for (int i = 0; i < 400; ++i) {
        const auto& src = (i % 3 == 0 ? vb : va)[rng() % (i % 3 == 0 ? vb.size() : va.size())].first;
        reads.push_back(src.substr(rng() % 3000, 150));
    }
    for (int i = 0; i < 100; ++i) reads.push_back(random_seq(rng, 150));

    setenv("SINGLET_NONHOST_NO_MMAP", "1", 1);
    auto a_priv = MinSketchIndex::load(pa);
    auto b_priv = MinSketchIndex::load(pb);
    unsetenv("SINGLET_NONHOST_NO_MMAP");
    auto a_map = MinSketchIndex::load(pa);
    auto b_map = MinSketchIndex::load(pb);

    CHECK(a_map.size() == a_priv.size() && a_map.size() == a.size(), "mapped index has the same entry count");
    CHECK(a_map.k() == a.k() && a_map.w() == a.w(), "k and w come from the header");
    CHECK(a_map.species_name(3) == "speciesA3", "species names are read from the mapped file");

    // the old merge: a sorted copy of both (built by saving the union and re-reading it privately)
    auto merged_old = MinSketchIndex::merge(a_priv, b_priv);
    merged_old.save("test_min_sketch_m.idx");
    setenv("SINGLET_NONHOST_NO_MMAP", "1", 1);
    auto merged_copy = MinSketchIndex::load("test_min_sketch_m.idx");
    unsetenv("SINGLET_NONHOST_NO_MMAP");
    auto merged_map = MinSketchIndex::merge(a_map, b_map);

    CHECK(merged_map.size() == a.size() + b.size() && merged_copy.size() == merged_map.size(),
          "union holds every entry");
    CHECK(merged_map.species_name(33) == "speciesB33" && merged_map.species_name(3) == "speciesA3",
          "union keeps both name manifests");

    bool all_same = true;
    for (const auto& r : reads)
        all_same = all_same && merged_map.query(r) == merged_copy.query(r) &&
                   a_map.query(r) == a.query(r);
    CHECK(all_same, "query() identical on mapped, private and union indexes");

    // batch path: sorted (hash, read) list
    std::vector<std::pair<uint64_t, uint32_t>> flat;
    for (uint32_t i = 0; i < reads.size(); ++i)
        for (uint64_t h : singlet::nonhost::extract_minimizers(reads[i], 21, 11)) flat.push_back({h, i});
    std::sort(flat.begin(), flat.end());
    std::vector<Hits> r_copy(reads.size()), r_map(reads.size());
    merged_copy.batch_query_presorted(flat, r_copy);
    merged_map.batch_query_presorted(flat, r_map);
    CHECK(same(r_copy, r_map), "batch_query_presorted identical on union and merged copy");
    size_t hit_reads = 0;
    for (auto& h : r_map) hit_reads += !h.empty();
    CHECK(hit_reads >= 400, "most reads hit (sanity)");

    std::remove(pa.c_str()); std::remove(pb.c_str()); std::remove("test_min_sketch_m.idx");
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
