// SPDX-License-Identifier: MIT
// Regression: a Visium run replaces its called set (the tissue spots) after export, so editing_sites / editing_cells / editing_summary,
// summarized at export time over the first call, disagreed with cell_calls.tsv (GSM8699165: 13 tissue spots outside the export-time call
// had alt entries at sites whose table row read zero edited UMIs). resummarize() recomputes the tables over the final set;
// check_outputs() is the end-of-run self-check that finds such a disagreement.
#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include <fstream>
#include <sstream>

#include "singlet/pileup/rna_editing.h"
#include "singlet/pileup/self_check.h"

using namespace singlet;
namespace fs = std::filesystem;

int main() {
    const std::string out = (fs::temp_directory_path() / "ed_consistency_test").string();
    fs::remove_all(out);
    fs::create_directories(out + "/private/editing");

    // 4 sites x 5 barcodes. alt (edited) and dp (depth) in CSC order.
    const std::vector<std::string> rows{"chr1:10:A>G", "chr1:20:A>G", "chr2:5:T>C", "chr3:7:A>G"};
    const std::vector<std::string> cols{"AAAA", "CCCC", "GGGG", "TTTT", "ACGT"};
    editing::Csc alt, dp;
    alt.nrows = dp.nrows = 4; alt.ncols = dp.ncols = 5;
    // column -> (row, alt, dp)
    struct E { uint32_t col, row, a, d; };
    const std::vector<E> es{{0, 0, 1, 4}, {0, 1, 0, 2}, {1, 1, 2, 5}, {2, 0, 0, 3}, {2, 2, 1, 1}, {3, 3, 3, 3}, {4, 0, 2, 2}, {4, 2, 0, 6}};
    alt.indptr.assign(6, 0); dp.indptr.assign(6, 0);
    for (uint32_t c = 0; c < 5; ++c) {
        for (const auto& e : es) if (e.col == c) {
            if (e.a > 0) { alt.indices.push_back(int32_t(e.row)); alt.data.push_back(uint16_t(e.a)); }
            dp.indices.push_back(int32_t(e.row)); dp.data.push_back(uint16_t(e.d));
        }
        alt.indptr[c + 1] = int32_t(alt.indices.size()); dp.indptr[c + 1] = int32_t(dp.indices.size());
    }
    auto write = [&](const std::vector<uint32_t>& cells) {
        const auto er = editing::summarize(alt, dp, cells);
        std::vector<std::string> names;
        for (uint32_t r : er.covered) names.push_back(rows[r]);
        assert(editing::write_sites(out + "/private/editing/editing_sites.tsv.gz", er, names));
        assert(editing::write_cells(out + "/editing_cells.tsv", er, cells, cols));
        assert(editing::write_summary(out + "/editing_summary.json", "panel.tsv.gz", 4, er, cells.size()));
        return er;
    };
    auto er = write({0, 1, 2});   // the export-time call
    // covered rows are all four here (dp_all > 0), so the matrices written are the full ones
    assert(er.covered.size() == 4);
    assert(pz::write_1pz<uint16_t>(out + "/private/editing/editing_alt.1pz", 4, 5, alt.indptr, alt.indices, alt.data, rows, cols));
    assert(pz::write_1pz<uint16_t>(out + "/private/editing/editing_dp.1pz", 4, 5, dp.indptr, dp.indices, dp.data, rows, cols));

    assert(editing::check_outputs(out).empty());          // consistent as written

    // the final call differs: barcodes ACGT and TTTT are in, CCCC is out. The old tables no longer describe the cell set.
    const std::unordered_set<std::string> final_call{"AAAA", "GGGG", "TTTT", "ACGT"};
    std::string err;
    // reproduction of the defect: tables over {AAAA,CCCC,GGGG} next to a cell list of the final call
    {
        // alt entries of TTTT/ACGT sit at sites whose table row (over AAAA,CCCC,GGGG) edited count does not include them
        const auto stale = editing::summarize(alt, dp, {0, 1, 2});
        assert(stale.site[3].alt == 0 && alt.indptr[4] > alt.indptr[3]);   // TTTT has an edited UMI at chr3:7, the table says 0
    }
    {   // cell_calls.tsv of the final call (tissue spots): the tables of the export-time call are reported against it
        FILE* f = std::fopen((out + "/cell_calls.tsv").c_str(), "w");
        std::fputs("barcode\tis_cell\n", f);
        for (const auto& b : cols) std::fprintf(f, "%s\t%s\n", b.c_str(), final_call.count(b) ? "TRUE" : "FALSE");
        std::fclose(f);
        const auto bad = editing::check_outputs(out);
        assert(!bad.empty() && bad[0].find("cell_calls.tsv") != std::string::npos);
    }
    assert(editing::resummarize(out, "panel.tsv.gz", 4, final_call, &err));
    assert(editing::check_outputs(out).empty());
    const auto bc = editing::read_editing_cells_tsv_barcodes(out + "/editing_cells.tsv");
    assert(bc.size() == 4 && bc[0] == "AAAA" && bc[1] == "GGGG" && bc[2] == "TTTT" && bc[3] == "ACGT");
    const auto fresh = editing::summarize(alt, dp, {0, 2, 3, 4});
    assert(fresh.site[3].alt == 3 && fresh.alt_total == 1 + 0 + 1 + 3 + 2);

    // the end-of-run self-check (singlet --self-check): clean directory -> self_check.json ok, 0 problems
    assert(singlet::run_self_check(out) == 0);
    {
        std::ifstream f(out + "/self_check.json");
        std::stringstream ss; ss << f.rdbuf();
        assert(ss.str().find("\"ok\": true") != std::string::npos && ss.str().find("\"present\": true") != std::string::npos);
    }
    assert(singlet::kExitSelfCheck == 12);

    // a table that does not match its matrices is reported
    {
        const auto wrong = editing::summarize(alt, dp, {0, 1, 2});
        std::vector<std::string> names;
        for (uint32_t r : wrong.covered) names.push_back(rows[r]);
        assert(editing::write_sites(out + "/private/editing/editing_sites.tsv.gz", wrong, names));   // table over the old call, cells over the new
        const auto bad = editing::check_outputs(out);
        assert(!bad.empty());
    }
    // an undecodable file is reported too
    fs::resize_file(out + "/private/editing/editing_dp.1pz", 40);
    assert(!editing::check_outputs(out).empty());
    assert(singlet::run_self_check(out) > 0);   // the self-check fails, and says so in self_check.json
    {
        std::ifstream f(out + "/self_check.json");
        std::stringstream ss; ss << f.rdbuf();
        assert(ss.str().find("\"ok\": false") != std::string::npos);
    }

    fs::remove_all(out);
    std::cout << "test_editing_consistency: ok\n";
    return 0;
}
