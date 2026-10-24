// Phase profiler for the lane-rANS decoder (build with -DTB_PROFILE for the rdtsc breakdown).
//   bin/track_b_prof.exe file.1pz [cfg_kind|-1] [pb]
// Times are thread cycles (QueryThreadCycleTime) per nnz, min of 3 reps; wall ms shown for reference.
#include "common.h"
#include "track_b.h"
using namespace cs;
using namespace cs::tb;

int main(int argc, char** argv) {
    CanonicalMatrix M = load_1pz(argv[1]);
    RansOpts op;
    if (argc > 2) op.cfg_kind = std::atoi(argv[2]);
    if (argc > 3) op.pb = std::atoi(argv[3]);
    auto enc = rans_encode(M, op);
    RansFile F = rans_parse(enc);
    RowPerm P;
    const uint8_t* p = F.rowc_p;
    get_rowcounts(p, P, F.m);
    {
        const uint8_t* q = F.payload_p;
        Cfg c = Cfg::get(q);
        std::printf("payload %zu B (%.3f b/nnz) tables %zu B; cfg sh=%d cap=%d ks=%d kcap=%d nk=%d pb=%d -> gap table %zu KB\n", last_breakdown().payload,
                    last_breakdown().payload * 8.0 / F.nnz, last_breakdown().tables, c.sh, c.cap, c.ks, c.kcap, c.nk, c.pb, (c.nctx() << c.pb) * 4 / 1024);
    }
    for (int simd = 1; simd >= 0; --simd) {
        SimdScope sg(simd == 1);
        double best_open = 1e300, best_chunks = 1e300, best_canon = 1e300, w_chunks = 0;
        uint64_t prof[8] = {0};
        for (int rep = 0; rep < 3; ++rep) {
            CycTimer t;
            p = F.payload_p;
            DecState D;
            decode_open(p, D, F.n, F.chunk_cols);
            double c_open = t.cycles();
            std::vector<uint32_t> rk(F.nnz), vl(F.nnz);
            Scratch sc;
            CycTimer t2;
            Timer w2;
            const uint32_t nchunks = (F.n + F.chunk_cols - 1) / F.chunk_cols;
            for (uint32_t c = 0; c < nchunks; ++c) {
                uint32_t c0 = c * F.chunk_cols;
                decode_chunk_native(D, c, rk.data() + D.ptr[c0], vl.data() + D.ptr[c0], sc);
            }
            double c_chunks = t2.cycles();
            double wall = w2.ms();
            CycTimer t3;
            Canon cn(P.perm.data(), F.m);
            for (uint32_t j = 0; j < F.n; ++j) cn.column(rk.data() + D.ptr[j], vl.data() + D.ptr[j], D.ptr[j + 1] - D.ptr[j]);
            double c_canon = t3.cycles();
            if (c_chunks < best_chunks) { best_chunks = c_chunks; w_chunks = wall; for (int i = 0; i < 8; ++i) prof[i] = sc.prof[i]; }
            best_open = std::min(best_open, c_open);
            best_canon = std::min(best_canon, c_canon);
        }
        std::printf("simd=%d cycles/nnz: open %.2f | chunks %.2f (= %.0f Mnnz/s @3GHz; wall %.1f ms) | canon %.2f\n", simd, best_open / F.nnz,
                    best_chunks / F.nnz, kNominalGHz * 1e3 / (best_chunks / F.nnz), w_chunks, best_canon / F.nnz);
#ifdef TB_PROFILE
        std::printf("     tsc/nnz: main %.1f fill1 %.1f resctx %.1f resdec %.1f scatter %.1f\n", prof[0] / (double)F.nnz, prof[1] / (double)F.nnz,
                    prof[2] / (double)F.nnz, prof[3] / (double)F.nnz, prof[4] / (double)F.nnz);
#endif
    }
}
