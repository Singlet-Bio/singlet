// D-101 storage rule of the all-site member (allsite_store.h): the called cells only; with one donor, per-cell entries only at the sites
// where the donor is 0/1 or 1/1; with two or more donors, or no demux, every site. The shape never changes and store.json records the rule.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/sparse_accumulator.h"
#include "singlet/pileup/allsite_store.h"

namespace A = singlet::allsite;
using CSC = SparseAccumulator<uint16_t>::CSCMatrix;
static int n_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cout << "FAIL " << msg << "\n"; ++n_fail; } } while (0)

// columns of (row, value) pairs -> CSC with nrows rows
static CSC make(uint32_t nrows, const std::vector<std::vector<std::pair<int32_t, uint16_t>>>& cols) {
    CSC m; m.nrows = nrows; m.ncols = static_cast<uint32_t>(cols.size()); m.indptr.push_back(0);
    for (const auto& c : cols) { for (const auto& e : c) { m.indices.push_back(e.first); m.data.push_back(e.second); } m.indptr.push_back(static_cast<int32_t>(m.indices.size())); }
    return m;
}

int main() {
    // 3 sites x 4 barcodes; barcodes 0-2 are called cells, 3 is not.
    // site 0: heterozygous (10 of 20 ALT); site 1: homozygous ALT (20 of 20); site 2: reference with one error (1 of 60)
    const CSC dp = make(3, {{{0, 10}, {1, 10}, {2, 20}}, {{0, 10}, {1, 10}, {2, 20}}, {{2, 20}}, {{0, 5}, {2, 5}}});
    const CSC ad = make(3, {{{0, 5}, {1, 10}, {2, 1}}, {{0, 5}, {1, 10}}, {}, {{0, 5}}});
    const std::vector<uint32_t> called = {0, 1, 2};

    {   // one donor: sites 0 and 1 keep per-cell entries, site 2 does not; barcode 3 is dropped
        const auto p = A::plan_store(ad, dp, called, 1);
        CHECK(p.rule == "single-donor-variant-sites", "one donor: rule " + p.rule);
        CHECK(p.cells == 3 && p.sites == 3 && p.sites_per_cell == 2, "one donor: 3 cells, 2 of 3 sites with per-cell entries");
        CHECK(p.keep_row == std::vector<uint8_t>({1, 1, 0}), "one donor: het and hom-alt kept, hom-ref dropped");
        const CSC d = A::restrict_store(dp, p), a = A::restrict_store(ad, p);
        CHECK(d.nrows == 3 && d.ncols == 4 && d.indptr.size() == 5, "the shape is unchanged");
        CHECK(d.data.size() == 4 && a.data.size() == 4, "one donor: DP 4 and AD 4 entries left (got " + std::to_string(d.data.size()) + ", " + std::to_string(a.data.size()) + ")");
        CHECK(d.indptr[3] == d.indptr[4], "the uncalled barcode has no entries");
        for (int32_t r : d.indices) CHECK(r != 2, "no per-cell entry at the hom-ref site");
    }
    {   // two donors: every site of the called cells
        const auto p = A::plan_store(ad, dp, called, 2);
        CHECK(p.rule == "multi-donor-all-sites" && p.sites_per_cell == 3, "two donors: every site");
        const CSC d = A::restrict_store(dp, p);
        CHECK(d.data.size() == 7, "two donors: DP keeps the 7 called-cell entries");
    }
    {   // no demux
        const auto p = A::plan_store(ad, dp, called, 0);
        CHECK(p.rule == "no-demux-all-sites" && p.sites_per_cell == 3, "no demux: every site");
    }
    {   // store.json
        const auto p = A::plan_store(ad, dp, called, 1);
        const std::string path = "tmp_allsite_store.json";
        CHECK(A::write_store_json(path, p, 3, 4), "store.json is written");
        std::ifstream f(path); std::stringstream ss; ss << f.rdbuf(); const std::string j = ss.str();
        CHECK(j.find("\"rule\": \"single-donor-variant-sites\"") != std::string::npos && j.find("\"n_donors\": 1") != std::string::npos &&
              j.find("\"sites_with_per_cell_entries\": 2") != std::string::npos, "store.json records the rule, the donor count and the kept sites");
        std::remove(path.c_str());
    }
    std::cout << (n_fail ? "FAILED" : "all passed") << "\n";
    return n_fail ? 1 : 0;
}
