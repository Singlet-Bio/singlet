// SPDX-License-Identifier: MIT
// Whitelist-free droplet protocols (Drop-seq, Seq-Well): the barcode collapse (barcode_collapse.h) and the cliff call (called_v2.h,
// whitelistfree_call) on synthetic libraries.
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "singlet/pileup/barcode_collapse.h"
#include "singlet/pileup/called_v2.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(name, cond)                                                                  \
    do {                                                                                   \
        if (cond) ++g_pass;                                                                \
        else { ++g_fail; std::cerr << "FAIL: " << name << " at line " << __LINE__ << "\n"; } \
    } while (0)

static std::string rand_bc(std::mt19937& rng) {
    std::string s;
    for (int i = 0; i < 12; ++i) s.push_back("ACGT"[rng() % 4]);
    return s;
}

int main() {
    // ── collapse ──
    {
        std::vector<std::pair<uint64_t, std::string>> sorted = {
            {100000, "ACGTACGTACGT"}, {9000, "ACGTACGTACGA"},   // child: one substitution, 9% of the parent
            {60000, "TTTTGGGGCCCC"}, {40000, "TTTTGGGGCCCA"},   // two real cells one substitution apart: ratio 1.5, both stay
            {500, "GGGGGGGGGGGA"},
            {150, "ACGTACGTACGC"},                              // another small neighbor of the first parent, in the universe
        };
        std::unordered_map<std::string, uint64_t> all = {{"ACGTACGTACTT", 40}, {"ACGTACGTAGGT", 3}};   // 40 reads joins, 3 does not (below 5)
        auto r = singlet::collapse_barcodes(sorted, 3.0, &all, 5);
        std::unordered_map<std::string, std::string> al(r.alias.begin(), r.alias.end());
        CHECK("child merged into parent", al.count("ACGTACGTACGA") && al["ACGTACGTACGA"] == "ACGTACGTACGT");
        CHECK("second small neighbor merged", al.count("ACGTACGTACGC") && al["ACGTACGTACGC"] == "ACGTACGTACGT");
        CHECK("comparable neighbors are both kept", !al.count("TTTTGGGGCCCA") && !al.count("TTTTGGGGCCCC"));
        CHECK("a below-universe neighbor with 40 reads is aliased", al.count("ACGTACGTACTT"));
        CHECK("a neighbor under the minimum is not", !al.count("ACGTACGTAGGT"));
        CHECK("kept count", r.kept.size() == 4);
        CHECK("in-universe merges counted", r.n_children_in_universe == 2);
    }
    // ── cliff call: 1,000 cells over 1,500-20,000 UMIs, 3,000 empties of 120-500, a long tail of 1-50 UMI barcodes ──
    {
        std::mt19937 rng(11);
        singlet::calledv2::Result R;
        std::vector<int64_t> tot;
        std::lognormal_distribution<double> cells(std::log(5000), 0.6);
        for (int i = 0; i < 1000; ++i) tot.push_back(std::max<int64_t>(1500, static_cast<int64_t>(cells(rng))));
        for (int i = 0; i < 3000; ++i) tot.push_back(120 + static_cast<int64_t>(380 * (1.0 - i / 3000.0)));
        for (int i = 0; i < 60000; ++i) tot.push_back(1 + static_cast<int64_t>(rng() % 50));
        R.totals = tot;
        R.called_v2.assign(tot.size(), 0);
        R.is_cell.assign(tot.size(), 0);
        singlet::calledv2::whitelistfree_call(R);
        int64_t n = 0;
        for (auto v : R.is_cell) n += v;
        CHECK("cliff library: a cliff is found", R.wf_cliff_ratio >= 3.0 && R.is_cell_method == "whitelistfree_cliff");
        CHECK("cliff library: the call is the 1,000 cells", n >= 950 && n <= 1100);
        CHECK("cliff names in the rule", singlet::calledv2::is_whitelistfree_droplet("dropseq") && singlet::calledv2::is_whitelistfree_droplet("Seq-Well") &&
                                             !singlet::calledv2::is_whitelistfree_droplet("10x-3p-v3") && !singlet::calledv2::is_whitelistfree_droplet("indrop"));
    }
    // ── no cliff: smooth power-law curve, the 2.1 rule decides ──
    {
        singlet::calledv2::Result R;
        std::vector<int64_t> tot;
        for (int i = 1; i <= 30000; ++i) tot.push_back(std::max<int64_t>(1, static_cast<int64_t>(3000.0 * std::pow(i, -0.5))));
        R.totals = tot;
        R.called_v2.assign(tot.size(), 0);
        for (int i = 0; i < 5000; ++i) R.called_v2[i] = 1;
        R.is_cell.assign(tot.size(), 0);
        singlet::calledv2::whitelistfree_call(R);
        int64_t n = 0;
        for (auto v : R.is_cell) n += v;
        CHECK("no-cliff library falls back to the 2.1 rule", R.is_cell_method == "whitelistfree_emptydrops_2.1" && n == 5000);
    }
    std::cout << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
