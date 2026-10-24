// SPDX-License-Identifier: MIT
// Idea 1 stage-4 micro benchmark (light build, no singlet/zstd headers).  Encodes a raw CSC dump (bench --rawdump) with a
// model, decodes it with the fast core, verifies bit-exactness and times the payload decode.
//   idea1_micro.exe raw.bin model.bin [nl=4] [reps=5] [abl=0] [find]
#define IDEA1_LIGHT 1
#include "idea1_core.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace cs::i1;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: idea1_micro raw.bin model.bin [nl] [reps] [abl] [find]\n"); return 2; }
    auto raw = cs::read_file(argv[1]);
    const int nl = argc > 3 ? std::atoi(argv[3]) : 4;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 5;
    g_abl = argc > 5 ? (uint32_t)std::atoi(argv[5]) : 0;
    const bool do_find = argc > 6;
    const uint8_t* p = raw.data();
    uint32_t m, n; uint64_t nnz;
    std::memcpy(&m, p, 4); std::memcpy(&n, p + 4, 4); std::memcpy(&nnz, p + 8, 8);
    const uint8_t vt = p[16];
    p += 24;
    std::vector<uint32_t> ipr(n + 1), ixr(nnz), dvr(nnz);
    std::memcpy(ipr.data(), p, (n + 1) * 4); p += (n + 1) * 4;
    std::memcpy(ixr.data(), p, nnz * 4); p += nnz * 4;
    std::memcpy(dvr.data(), p, nnz * 4);
    auto md = load_model(argv[2]);
    EncStats st;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> enc = encode_v2_payload(*md, m, n, nnz, vt, ipr.data(), ixr.data(), dvr.data(), nl, &st);
    double enc_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    V2Hdr h = parse_v2(enc.data(), enc.size());
    auto F = fast_model(h.model);
    std::vector<uint32_t> ip(h.n + 1), ix(h.nnz + 64), dv(h.nnz + 64);
    std::vector<uint32_t> ixo, ipo;
    if (g_abl & 4u) {
        uint32_t save = g_abl; g_abl = 0;
        decode_payload_v2(h, *F, ip.data(), ix.data(), dv.data());
        ixo = ix; ipo = ip; g_ixo = ixo.data(); g_ipo = ipo.data(); g_abl = save;
    }
    double cal = calibrate_chain();
    double best_c = 1e300, best_ms = 1e300;
    for (int r = 0; r < reps; ++r) {
        auto t1 = std::chrono::steady_clock::now();
        double c0 = thread_cycles();
        decode_payload_v2(h, *F, ip.data(), ix.data(), dv.data());
        best_c = std::min(best_c, thread_cycles() - c0);
        best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count());
    }
    cal = std::min(cal, calibrate_chain());
    bool ok = (g_abl == 0);
    if (ok) {
        for (uint32_t c = 0; c <= n && ok; ++c) ok = ip[c] == ipr[c];
        for (uint64_t i = 0; i < nnz && ok; ++i) ok = ix[i] == ixr[i] && dv[i] == dvr[i];
    }
    std::printf("%s nl=%d: type=%u nb=%u n=%u nnz=%llu  stream %.2f bits/nnz  enc %.0f ms  decode %.2f ms (%.2f Mnnz/s)  %.1f cyc/nnz  cal %.2f (norm %.1f cyc/nnz = %.1f Mnnz/s @3GHz)  %s\n",
                argv[1], nl, h.model->type, h.nb, h.n, (unsigned long long)h.nnz, enc.size() * 8.0 / nnz, enc_ms, best_ms, h.nnz / best_ms / 1e3,
                best_c / h.nnz, cal, best_c / h.nnz / (cal / 4.0), 3000.0 / (best_c / h.nnz / (cal / 4.0)), g_abl == 0 ? (ok ? "ROUNDTRIP=OK" : "ROUNDTRIP=FAIL") : "(ablation)");
    if (do_find) {
        std::vector<uint64_t> rec;
        g_rec = &rec;
        decode_payload_v2(h, *F, ip.data(), ix.data(), dv.data());
        g_rec = nullptr;
        const size_t N = rec.size();
        double bi = 1e300, bd = 1e300;
        uint32_t sink = 0;
        for (int r = 0; r < reps; ++r) {
            double c0 = thread_cycles();
            for (size_t i = 0; i < N; ++i) {
                const Tree& T = h.model->type == 0 ? F->trees[rec[i] & 15] : F->trees[0];
                sink += T.find(rec[i] >> 4);
            }
            bi = std::min(bi, thread_cycles() - c0);
            c0 = thread_cycles();
            uint32_t prev = 0;
            for (size_t i = 0; i < N; ++i) {
                const Tree& T = h.model->type == 0 ? F->trees[rec[i] & 15] : F->trees[0];
                prev = T.find((rec[i] >> 4) + (prev > 0xFFFFFFF0u));
            }
            sink += prev;
            bd = std::min(bd, thread_cycles() - c0);
        }
        std::printf("find replay: %zu searches, independent %.1f cyc/find, dependent %.1f cyc/find (cal %.2f)  sink=%u\n", N, bi / N, bd / N, cal, sink);
    }
    return 0;
}
