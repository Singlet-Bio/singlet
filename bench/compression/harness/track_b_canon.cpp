// Micro-benchmark of canonicalisation variants (rank-space native arrays -> canonical ascending original row ids).
#include "common.h"
#include "track_b.h"
#include "track_b_canon.h"
#include "track_b_sort.h"
#include <x86intrin.h>
using namespace cs;
using namespace cs::tb;

template <class T> static T& deref(T& t) { return t; }
template <class T> static T& deref(std::unique_ptr<T>& t) { return *t; }
int main(int argc, char** argv) {
    CanonicalMatrix M = load_1pz(argv[1]);
    RansOpts op;
    op.cfg_kind = 3;
    auto enc = rans_encode(M, op);
    RansFile F = rans_parse(enc);
    RowPerm P;
    const uint8_t* p = F.rowc_p;
    get_rowcounts(p, P, F.m);
    DecState D;
    p = F.payload_p;
    decode_open(p, D, F.n, F.chunk_cols);
    std::vector<uint32_t> rk(F.nnz), vl(F.nnz);
    Scratch sc;
    for (uint32_t c = 0; c * F.chunk_cols < F.n; ++c) decode_chunk_native(D, c, rk.data() + D.ptr[c * F.chunk_cols], vl.data() + D.ptr[c * F.chunk_cols], sc);
    const int NB = 8;
    uint64_t lim[NB] = {1, 4, 16, 24, 64, 256, 1024, ~0ull};
    uint64_t nnz_b[NB] = {0}, cols_b[NB] = {0};
    for (uint32_t j = 0; j < F.n; ++j) {
        uint64_t k = D.ptr[j + 1] - D.ptr[j];
        for (int b = 0; b < NB; ++b) if (k <= lim[b]) { nnz_b[b] += k; cols_b[b]++; break; }
    }
    std::printf("column size classes (<=): ");
    for (int b = 0; b < NB; ++b) std::printf(" %llu:%llu cols/%.1f%% nnz", (unsigned long long)lim[b], (unsigned long long)cols_b[b], 100.0 * nnz_b[b] / F.nnz);
    std::printf("\n");
    std::vector<uint32_t> a(F.nnz), b(F.nnz);
    auto run = [&](const char* name, auto&& mk) {
        for (int rep = 0; rep < 4; ++rep) {
            a = rk; b = vl;
            auto cn_ = mk();
            auto& cn = deref(cn_);
            CycTimer ct;
            Timer t;
            for (uint32_t j = 0; j < F.n; ++j) {
                uint32_t k = D.ptr[j + 1] - D.ptr[j];
                cn.column(a.data() + D.ptr[j], b.data() + D.ptr[j], k);
            }
            double ms = t.ms(), cyc = ct.cycles();
            bool ok = (a == M.indices) && (b == M.data);
            std::printf("canon %-8s: %.2f ms wall, %.1f thread-cycles/nnz (= %.0f Mnnz/s @3GHz) %s\n", name, ms, cyc / F.nnz, kNominalGHz * 1e3 / (cyc / F.nnz), ok ? "OK" : "MISMATCH");
        }
    };
    run("V0", [&] { return Canon(P.perm.data(), F.m); });
    run("vecsort", [&] { return std::make_unique<CanonVec>(P.perm.data(), F.m); });
    run("sort", [&] { return CanonSort(P.perm.data(), F.m); });
    run("bm_u16", [&] { return CanonBm(P.perm.data(), F.m); });
    {   // cache-effect probe: identity permutation (no random access); output mismatch expected
        std::vector<uint32_t> idp(F.m);
        for (uint32_t i = 0; i < F.m; ++i) idp[i] = i;
        for (int rep = 0; rep < 4; ++rep) {
            a = rk; b = vl;
            CanonBm cn(idp.data(), F.m);
            Timer t;
            for (uint32_t j = 0; j < F.n; ++j) cn.column(a.data() + D.ptr[j], b.data() + D.ptr[j], D.ptr[j + 1] - D.ptr[j]);
            std::printf("canon bm_ident: %.2f ms (%.1f ns/nnz)  [probe]\n", t.ms(), t.ms() * 1e6 / F.nnz);
        }
    }
}
