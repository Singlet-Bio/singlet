// w3_total called-set stream (the only non-derivable content of cell_calls.tsv): one flag per column of the exon matrix,
// context = quarter-octave depth bucket of the column's UMI total (Idea 2's flag coder).  Verifies the round trip and writes
// the stream.   bin/w3_total_calls.exe [--out results/w3_total_calls.tsv] [--dump DIR] exon_counts.1pz ...
// cell_calls.tsv is read from the directory of each input file.
#include "common.h"
#include "w3_total.h"
#include <cstdio>

using namespace cs;

int main(int argc, char** argv) {
    std::string out = "results/w3_total_calls.tsv", dump;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
        else files.push_back(a);
    }
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf) tsv << "file\tn\tcalled\tcalls_bytes\ttsv_bytes\tbitmap_raw_bytes\tok\n";
    for (auto& f : files) {
        CanonicalMatrix M = load_1pz(f);
        auto s = f.find_last_of("/\\");
        std::string dir = s == std::string::npos ? "." : f.substr(0, s);
        auto flag = w3::load_calls(M, dir + "/cell_calls.tsv");
        size_t called = 0;
        for (auto x : flag) called += x;
        auto enc = w3::enc_calls(M, flag);
        auto dec = w3::dec_calls(M, enc);
        bool ok = dec == flag;
        size_t tsv_bytes = read_file(dir + "/cell_calls.tsv").size();
        std::printf("%s: n=%u called=%zu  calls stream %zu B (raw bitmap %zu B, cell_calls.tsv %zu B)  %s\n", f.c_str(), M.n, called, enc.size(),
                    (size_t)((M.n + 7) / 8), tsv_bytes, ok ? "OK" : "FAIL");
        tsv << f << '\t' << M.n << '\t' << called << '\t' << enc.size() << '\t' << tsv_bytes << '\t' << (M.n + 7) / 8 << '\t' << (ok ? 1 : 0) << '\n';
        if (!dump.empty()) {
            std::string p = f;
            for (auto& c : p) if (c == '/' || c == '\\' || c == ':') c = '_';
            write_file(dump + "/" + p + ".calls", enc);
        }
    }
    return 0;
}
