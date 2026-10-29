// Train the Idea 4 calibration tables (and embed the vocabulary) into a model file.
//   bin/idea4_tables.exe model_in.i4m model_out.i4m [--T 100] train1.1pz train2.1pz ...
// Runs the encoder in histogram mode in all three high-cell modes (static, q6, codebook) plus ambient and split.
#include "idea4.h"
#include <iostream>
using namespace cs; using namespace cs::i4;

int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "usage: idea4_tables in.i4m out.i4m [--T n] train.1pz...\n"; return 2; }
    std::string in = argv[1], out = argv[2];
    u64 T = 100; std::vector<std::string> files;
    for (int i = 3; i < argc; ++i) { std::string a = argv[i]; if (a == "--T" && i + 1 < argc) T = std::strtoull(argv[++i], nullptr, 10); else files.push_back(a); }
    auto M = load_model(in);
    // provisional tables so the encoder can run (Poisson-ish presence, geometric counts)
    Hist H0; tables_from_hist(H0, *M);
    Hist H;
    bool vocab_done = M->vocab_raw_len != 0;
    for (auto& f : files) {
        std::printf("[tables] %s\n", f.c_str());
        CanonicalMatrix X = load_1pz(f);
        if (!vocab_done) {
            auto rn = detail::join_names(X.rownames);
            M->vocab_hash = fnv64(rn.data(), rn.size()); M->vocab_raw_len = (u32)rn.size();
            M->vocab_z = detail::zstd_c(rn.data(), rn.size(), 19);
            vocab_done = true;
            std::printf("[tables] vocab %zu raw -> %zu zstd-19\n", rn.size(), M->vocab_z.size());
        }
        for (int mode = 0; mode < 3; ++mode) {
            if (mode > 0 && M->k == 0) break;
            Params P; P.mode = (Mode)mode; P.T = T;
            Encoder enc(*M, P); Hist Hm; enc.hist = &Hm; enc.rc = nullptr;
            Timer t;
            for (u32 c = 0; c < X.n; ++c) enc.code_column(X, c);
            // keep only the contexts this mode is responsible for
            Hist Hsel;
            int th = enc.tab_high();
            for (int t2 = 0; t2 < NTAB; ++t2) {
                bool keep = (t2 == th) || (mode == 0 && (t2 == TAB_AMB || t2 == TAB_SPLIT));
                if (!keep) continue;
                for (int c = 0; c < NCTX; ++c) { Hsel.all[(size_t)t2 * NCTX + c] = Hm.all[(size_t)t2 * NCTX + c]; Hsel.pres[(size_t)t2 * NCTX + c] = Hm.pres[(size_t)t2 * NCTX + c]; }
                for (int c = 0; c < NCTX * CMAX; ++c) Hsel.cnt[(size_t)t2 * NCTX * CMAX + c] = Hm.cnt[(size_t)t2 * NCTX * CMAX + c];
            }
            H.add(Hsel);
            std::printf("[tables]   mode %d: %.0f ms\n", mode, t.ms());
            std::fflush(stdout);
        }
    }
    tables_from_hist(H, *M);
    save_model(out, *M);
    std::printf("[tables] wrote %s\n", out.c_str());
    return 0;
}
