// ed_resummarize OUT_DIR   recompute editing_sites / editing_cells / editing_summary of a finished sample directory over the called set
// of cell_calls.tsv (is_cell TRUE), then run the self-check. For samples whose called set changed after export (Visium).
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include "singlet/pileup/rna_editing.h"
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string out = argv[1];
    std::ifstream f(out + "/cell_calls.tsv");
    std::string ln;
    std::getline(f, ln);
    std::vector<std::string> h;
    { std::stringstream hs(ln); std::string t; while (std::getline(hs, t, '\t')) h.push_back(t); }
    size_t bi = 0, ci = 0;
    for (size_t k = 0; k < h.size(); ++k) { if (h[k] == "barcode") bi = k; if (h[k] == "is_cell") ci = k; }
    std::unordered_set<std::string> called;
    while (std::getline(f, ln)) {
        std::vector<std::string> v; std::stringstream ls(ln); std::string t;
        while (std::getline(ls, t, '\t')) v.push_back(t);
        if (v.size() > ci && (v[ci] == "TRUE" || v[ci] == "1")) called.insert(v[bi]);
    }
    std::cout << "called " << called.size() << "\n";
    std::string err;
    if (!singlet::editing::resummarize(out, "rna_editing_GRCh38.tsv.gz", 15587267, called, &err)) { std::cout << "FAIL " << err << "\n"; return 1; }
    auto p = singlet::editing::check_outputs(out);
    for (auto& x : p) std::cout << "BAD " << x << "\n";
    if (p.empty()) std::cout << "OK consistent\n";
    return p.empty() ? 0 : 1;
}
