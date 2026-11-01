// SPDX-License-Identifier: MIT
// test_nonhost_confirmed_reads.cpp — the nonhost_confirmed_reads member: pack/unpack round trip and the privacy gate.
//
// Fixture: a random "viral genome" (species 7) and a random "human genome" whose minimizers fill the host filter. The human genome
// also contains a stretch copied from the viral genome (an endogenous-like homologue). Reads: pure viral, spiked pure human,
// chimeric (viral + human halves), and reads from the shared stretch. The screen classifies them with host subtraction; the
// packed file must hold the pure viral reads and nothing that carries a host k-mer.

#include "singlet/pileup/nonhost/host_kmer_filter.h"
#include "singlet/pileup/nonhost/min_sketch.h"
#include "singlet/pileup/nonhost/nonhost_confirmed_reads.h"
#include "singlet/pileup/nonhost/nonhost_screener.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

using namespace singlet::nonhost;

static int g_failed = 0;
#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (!(cond)) { ++g_failed; std::cerr << "FAIL: " << (msg) << "\n"; }      \
        else std::cerr << "PASS: " << (msg) << "\n";                              \
    } while (0)

static std::string rand_seq(std::mt19937_64& g, int n) {
    static const char B[4] = {'A', 'C', 'G', 'T'};
    std::string s(n, 'A');
    for (auto& c : s) c = B[g() & 3];
    return s;
}

int main() {
    const int K = 21, W = 11, RL = 100;
    std::mt19937_64 gv(1), gh(2), gb(3);
    const std::string viral = rand_seq(gv, 4000);
    std::string human = rand_seq(gh, 6000);
    const std::string shared = viral.substr(2000, 600);  // homologue present in both genomes
    human.replace(1000, shared.size(), shared);

    HostKmerFilter host(2000000u, K, W);
    for (uint64_t h : extract_minimizers(human, K, W)) host.insert(h);

    MinSketchIndex viral_idx(K, W);
    viral_idx.add_sequence(viral, 7u, 50000u);
    viral_idx.finalize();
    MinSketchIndex empty_idx(K, W);
    empty_idx.finalize();
    NonHostScreener screener(viral_idx, empty_idx, 0.05f);

    const std::string CB = "AAACCCAAGAAACACT";
    std::vector<TaggedRead> tagged;
    std::set<std::string> viral_pure;  // sequences that must be written
    auto add = [&](const std::string& seq, const std::string& umi, bool must_write) {
        tagged.push_back({seq, CB, umi});
        if (must_write) viral_pure.insert(seq);
    };
    const char* umis[] = {"AAAAAAAAAA", "CCCCCCCCCC", "GGGGGGGGGG", "TTTTTTTTTT", "ACACACACAC", "GTGTGTGTGT"};
    int u = 0;
    for (int off : {0, 150, 300, 450, 700, 900}) add(viral.substr(off, RL), umis[u++ % 6], true);   // pure viral, outside the homologue
    for (int off : {0, 500, 1500, 2500, 3500, 5000}) add(human.substr(off, RL), "AAAAAAAAAC", false);    // spiked human
    for (int off : {100, 400, 800}) add(viral.substr(off, 50) + human.substr(off + 2000, 50), "AAAAAAAAAG", false);  // chimeric
    for (int off : {2100, 2300}) add(viral.substr(off, RL), "AAAAAAAAAT", false);  // homologue shared with the host
    add(viral.substr(1200, 60) + "N" + viral.substr(1261, 39), "CAAAAAAAAA", false);  // N: counted, not written

    std::vector<std::string> seqs;
    for (const auto& t : tagged) seqs.push_back(t.seq);
    const auto hits = screener.classify_multi_batch_filtered(seqs, host, 0.25f, 1);
    const std::unordered_set<uint32_t> confirmed{7u};
    const std::unordered_set<std::string> valid{CB};

    ConfirmedReadsStats st;
    const std::string buf = pack_confirmed_reads(tagged, hits, confirmed, valid, &host, 16, 10, K, W, &st);
    std::vector<UnpackedRead> back;
    CHECK(unpack_confirmed_reads(buf, back), "file round-trips (well formed)");

    // privacy: no written read contains a host minimizer, none is a human, chimeric or homologue read
    bool any_host = false, all_expected = true;
    for (const auto& r : back) {
        if (host.count_host_minimizers(extract_minimizers(r.seq, K, W)).first != 0) any_host = true;
        if (!viral_pure.count(r.seq) || r.species_id != 7u || r.cb != CB) all_expected = false;
    }
    CHECK(!any_host, "no written read has a host k-mer");
    CHECK(all_expected, "every written read is a pure viral fixture read of species 7 in the called cell");
    CHECK(back.size() == viral_pure.size(), "all pure viral reads are written (" + std::to_string(back.size()) + "/" + std::to_string(viral_pure.size()) + ")");
    std::set<std::string> written;
    for (const auto& r : back) written.insert(r.seq);
    bool human_in = false;
    for (int off : {0, 500, 1500, 2500, 3500, 5000}) human_in |= written.count(human.substr(off, RL)) > 0;
    CHECK(!human_in, "spiked human reads are absent");
    CHECK(st.n_dropped_n == 0u || st.n_dropped_n == 1u, "N read is not written");
    bool umi_ok = true;
    for (const auto& r : back) umi_ok &= r.umi.size() == 10;
    CHECK(umi_ok, "barcode and UMI round trip with their lengths");

    // the host filter is required, and so is a confirmed species and a called cell
    CHECK(pack_confirmed_reads(tagged, hits, confirmed, valid, nullptr, 16, 10, K, W).empty(), "no host filter: nothing is written");
    HostKmerFilter unloaded;
    CHECK(pack_confirmed_reads(tagged, hits, confirmed, valid, &unloaded, 16, 10, K, W).empty(), "unloaded host filter: nothing is written");
    CHECK(pack_confirmed_reads(tagged, hits, {}, valid, &host, 16, 10, K, W).empty(), "no confirmed species: nothing is written");
    CHECK(pack_confirmed_reads(tagged, hits, {99u}, valid, &host, 16, 10, K, W).empty(), "other species confirmed: nothing is written");
    CHECK(pack_confirmed_reads(tagged, hits, confirmed, {"TTTTTTTTTTTTTTTT"}, &host, 16, 10, K, W).empty(), "not a called cell: nothing is written");

    // the file does not depend on the order of the reads
    auto t2 = tagged;
    auto h2 = hits;
    for (size_t i = 0; i + 1 < t2.size(); i += 2) { std::swap(t2[i], t2[i + 1]); std::swap(h2[i], h2[i + 1]); }
    CHECK(pack_confirmed_reads(t2, h2, confirmed, valid, &host, 16, 10, K, W) == buf, "byte-identical for a permuted read order");

    // a forced host-bearing read that the screen did classify (hit list says viral) is still refused
    std::vector<TaggedRead> tf{{human.substr(10, RL), CB, "ACGTACGTAC"}};
    NonHostScreener::MultiHit forced;
    forced.top_category = NonHostCategory::VIRAL;
    forced.viral_hits.emplace_back(7u, 0.9f);
    ConfirmedReadsStats sf;
    CHECK(pack_confirmed_reads(tf, {forced}, confirmed, valid, &host, 16, 10, K, W, &sf).empty() && sf.n_excluded_host == 1u,
          "a human read mislabelled as viral is refused by the host re-check");

    std::cerr << (g_failed ? "FAILED\n" : "ALL PASSED\n");
    return g_failed ? 1 : 0;
}
