// w3_total row-count coder experiment: size of the per-row nnz side stream under the shipped selection (best of 7 coders)
// and under richer adaptive contexts.  bin/w3_total_cnt.exe file.1pz ...
#include "common.h"
#include "w3_total.h"
#include <cstdio>

using namespace cs;

// context = (log of previous value, relation of the one before to it)
static int ilog(uint32_t v) { return v ? 1 + i2::ilog2u(v) : 0; }
static size_t adaptive(const std::vector<uint32_t>& v, int variant) {
    std::vector<uint8_t> o;
    i2::RcEnc rc(o);
    int nctx = variant == 0 ? 18 : (variant == 1 ? 18 * 3 : (variant == 2 ? 18 * 18 : 18 * 3 * 3));
    i2::NumCoder nc(nctx);
    uint32_t p1 = 0, p2 = 0, p3 = 0;
    for (uint32_t x : v) {
        int c1 = std::min(ilog(p1), 17), c2 = std::min(ilog(p2), 17);
        int ctx = 0;
        if (variant == 0) ctx = c1;
        else if (variant == 1) ctx = c1 * 3 + (c2 > c1 ? 2 : (c2 < c1 ? 0 : 1));
        else if (variant == 2) ctx = c1 * 18 + c2;
        else {
            int c3 = std::min(ilog(p3), 17);
            ctx = (c1 * 3 + (c2 > c1 ? 2 : (c2 < c1 ? 0 : 1))) * 3 + (c3 > c1 ? 2 : (c3 < c1 ? 0 : 1));
        }
        nc.code(rc, ctx, x);
        p3 = p2; p2 = p1; p1 = x;
    }
    rc.finish();
    return o.size();
}

int main(int argc, char** argv) {
    std::printf("file\tm\tsupported\tshipped_best(tag)\tadaptive_c1\tc1xrel2\tc1xc2\tc1xrel2xrel3\tclass_shipped(tag)\tclass_c1\n");
    for (int i = 1; i < argc; ++i) {
        CanonicalMatrix M = load_1pz(argv[i]);
        auto k = w3::row_keys(M, w3::ORD_EXACT);
        auto kc = w3::row_keys(M, w3::ORD_CLASS);
        size_t sup = 0;
        for (auto x : k) sup += x != 0;
        int t = 0, tc = 0;
        auto best = w3::enc_vals(k, &t);
        auto bestc = w3::enc_vals(kc, &tc);
        std::printf("%s\t%u\t%zu\t%zu(%d)\t%zu\t%zu\t%zu\t%zu\t%zu(%d)\t%zu\n", argv[i], M.m, sup, best.size(), t, adaptive(k, 0), adaptive(k, 1),
                    adaptive(k, 2), adaptive(k, 3), bestc.size(), tc, adaptive(kc, 0));
        std::fflush(stdout);
    }
    return 0;
}
