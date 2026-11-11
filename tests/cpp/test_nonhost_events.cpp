// SPDX-License-Identifier: MIT
// test_nonhost_events.cpp — the threaded scan of the non-host screen's batch query gives the serial scan's hit maps, insertion order included.
//
// MinSketchIndex::batch_query_events returns the scan as (read, species) events per (run, slice). Replaying the events of each read in the order of
// the vectors must give maps with the same contents and the same insertion sequence (hence iteration order) as batch_query_presorted, for any thread
// count and for an index of several runs (merge). NonHostScreener::classify_multi_batch (which uses the events) gives the same hits in the same
// order at every thread count.

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "singlet/pileup/nonhost/min_sketch.h"
#include "singlet/pileup/nonhost/nonhost_screener.h"

using namespace singlet::nonhost;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static std::string rand_seq(std::mt19937& rng, size_t n) {
    static const char* B = "ACGT";
    std::string s(n, 'A');
    for (auto& c : s) c = B[rng() & 3];
    return s;
}

int main() {
    std::mt19937 rng(5);
    const int K = 21, W = 11;
    // genomes: species 0..39; species of one family share a block, so one hash hits several species
    std::vector<std::string> genome;
    const std::string shared = rand_seq(rng, 600);
    for (int i = 0; i < 40; ++i) {
        std::string g = rand_seq(rng, 3000);
        if (i % 4 == 0) g.replace(500, shared.size(), shared);
        genome.push_back(g);
    }
    MinSketchIndex viral(K, W), bact(K, W), fung(K, W);
    for (int i = 0; i < 40; ++i) {
        MinSketchIndex& idx = i < 10 ? viral : (i < 28 ? bact : fung);
        idx.add_sequence(genome[i], static_cast<uint32_t>(i), 100000);
    }
    viral.finalize(); bact.finalize(); fung.finalize();
    const MinSketchIndex micro = MinSketchIndex::merge(bact, fung);   // two runs

    // reads: pieces of genomes with a few mutations, some random
    std::vector<std::string> reads;
    for (int i = 0; i < 6000; ++i) {
        if (i % 9 == 0) { reads.push_back(rand_seq(rng, 100)); continue; }
        const std::string& g = genome[rng() % genome.size()];
        std::string r = g.substr(rng() % (g.size() - 120), 40 + rng() % 80);
        for (int m = 0; m < 3; ++m) r[rng() % r.size()] = "ACGT"[rng() & 3];
        reads.push_back(r);
    }
    std::vector<std::pair<uint64_t, uint32_t>> flat;
    for (uint32_t ri = 0; ri < reads.size(); ++ri)
        for (uint64_t h : extract_minimizers(reads[ri], K, W)) flat.push_back({h, ri});
    std::sort(flat.begin(), flat.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    const MinSketchIndex* idxs[2] = {&viral, &micro};
    for (const MinSketchIndex* idx : idxs) {
        std::vector<std::unordered_map<uint32_t, uint32_t>> ref(reads.size());
        idx->batch_query_presorted(flat, ref);
        for (int nt : {1, 2, 3, 8}) {
            std::vector<std::vector<uint64_t>> lists;
            idx->batch_query_events(flat, nt, lists);
            std::vector<std::unordered_map<uint32_t, uint32_t>> got(reads.size());
            for (const auto& l : lists)
                for (uint64_t ev : l) got[ev >> 32][static_cast<uint32_t>(ev & 0xffffffffu)]++;
            size_t same = 0;
            for (size_t ri = 0; ri < reads.size(); ++ri) {
                std::vector<std::pair<uint32_t, uint32_t>> a(ref[ri].begin(), ref[ri].end()), b(got[ri].begin(), got[ri].end());
                if (a == b) ++same;   // contents and iteration order
            }
            CHECK(same == reads.size());
        }
    }

    // the screener's batch result does not depend on the thread count (hits in the same order)
    NonHostScreener screener(viral, micro, 0.05f);
    const auto ref_batch = screener.classify_multi_batch(reads, 0.25f, 1);
    size_t hits = 0;
    for (const auto& m : ref_batch) hits += m.viral_hits.size() + m.microbial_hits.size();
    CHECK(ref_batch.size() == reads.size());
    CHECK(hits > 1000);
    for (int nt : {2, 4, 8}) {
        const auto batch = screener.classify_multi_batch(reads, 0.25f, nt);
        CHECK(batch.size() == ref_batch.size());
        size_t same = 0;
        for (size_t ri = 0; ri < batch.size() && ri < ref_batch.size(); ++ri)
            if (batch[ri].viral_hits == ref_batch[ri].viral_hits && batch[ri].microbial_hits == ref_batch[ri].microbial_hits &&
                batch[ri].top_category == ref_batch[ri].top_category) ++same;
        CHECK(same == ref_batch.size());
    }
    if (g_fail) { std::fprintf(stderr, "%d checks failed\n", g_fail); return 1; }
    std::printf("test_nonhost_events: ok\n");
    return 0;
}
