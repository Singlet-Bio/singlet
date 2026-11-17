// ed_check OUT_DIR...   the end-of-run editing self-check on finished sample output directories: decodes editing_alt/dp.1pz
// and recomputes editing_sites.tsv.gz from them over the barcodes of editing_cells.tsv. Prints "OK dir seconds" or "BAD dir problem..."
#include <chrono>
#include <iostream>
#include "singlet/pileup/rna_editing.h"
int main(int argc, char** argv) {
    int bad = 0;
    for (int i = 1; i < argc; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        auto p = singlet::editing::check_outputs(argv[i]);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (p.empty()) std::cout << "OK " << argv[i] << " " << s << "\n";
        else { ++bad; std::cout << "BAD " << argv[i]; for (auto& x : p) std::cout << " | " << x; std::cout << " (" << s << " s)\n"; }
    }
    return bad ? 1 : 0;
}
