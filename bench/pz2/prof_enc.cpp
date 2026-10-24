// Encode-time profile: bench/pz2/bin/prof_enc.exe file.1pz   (times encode_matrix under several option sets)
#include <singlet/pileup/pz_reader.h>
#include <singlet/pz2/pz2.h>

#include <chrono>
#include <cstdio>

using namespace singlet::pz2;
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    auto r = singlet::pz::read_1pz(argv[1]);
    CSC c;
    c.m = r.m; c.n = r.n; c.nnz = r.nnz; c.indptr = r.indptr.data(); c.indices = r.indices.data(); c.data = r.data.data();
    std::printf("%s m=%u n=%u nnz=%llu\n", argv[1], r.m, r.n, (unsigned long long)r.nnz);
    auto run = [&](const char* name, EncodeOptions o) {
        o.compute_digest = false;
        double t = now_s();
        EncodeStats st;
        auto b = encode_matrix(c, o, &st);
        std::printf("  %-28s %8.3f s  %10zu bytes  codec %d\n", name, now_s() - t, b.size(), int(st.codec));
    };
    EncodeOptions o;
    o.col_names.kind = NamesKind::Cells; o.row_names.kind = NamesKind::Pack;
    o.threads = 1;
    run("auto, 1 thread", o);
    EncodeOptions k0 = o; k0.ctx_kind = 0; k0.codec = CodecId::Rans;
    run("rans kind0, 1 thread", k0);
    EncodeOptions k4 = o; k4.ctx_kind = 4; k4.codec = CodecId::Rans;
    run("rans kind4, 1 thread", k4);
    EncodeOptions z1 = k0; z1.zstd_level = 3;
    run("rans kind0 zstd3", z1);
    EncodeOptions g = o; g.codec = CodecId::Generic;
    run("generic z19", g);
    g.zstd_level = 9;
    run("generic z9", g);
    return 0;
}
