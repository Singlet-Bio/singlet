// SPDX-License-Identifier: MIT
// w3_fast driver.
//   bin/w3_fast_bench.exe verify [opts] file.1pz...      encode, then decode with every available ISA in class-order and canonical
//                                                        mode and check the results (class order against an independent reference)
//   bin/w3_fast_bench.exe time   [opts] file.1pz...      decode timing (min of reps): class-order mode into int32/uint16 arrays
// opts: --lanes N --cpo N --pb N --pbv N --ctx K --chunk C --isa scalar,avx2,avx512 --reps R --out tsv --threads T --cache DIR
#include "common.h"
#include "w3_fast.h"
#include "track_b_sort.h"   // read-only: Track B AVX-512 vector sorter, reused for the canonical (exact) mode
#include <algorithm>
#include <iostream>
#include <sstream>

using namespace cs;
using namespace cs::wf;

#ifdef _WIN32
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess(void);
extern "C" __declspec(dllimport) int __stdcall SetPriorityClass(void*, unsigned long);
extern "C" __declspec(dllimport) int __stdcall SetThreadPriority(void*, int);
extern "C" __declspec(dllimport) void* __stdcall GetCurrentThread(void);
extern "C" __declspec(dllimport) int __stdcall QueryThreadCycleTime(void*, unsigned long long*);
#endif

static std::string base_name(const std::string& p) { auto s = p.find_last_of("/\\"); return s == std::string::npos ? p : p.substr(s + 1); }
static std::string parent_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    if (s == std::string::npos || s == 0) return "";
    auto t = p.find_last_of("/\\", s - 1);
    size_t start = (t == std::string::npos) ? 0 : t + 1;
    return p.substr(start, s - start);
}
static double calib_ghz() {
    uint64_t s = 0;
    Timer t;
    for (uint64_t i = 0; i < 10000000ull; ++i)
        asm volatile("add $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0" : "+r"(s));
    return 1e8 / t.ms() / 1e6;
}
static uint64_t thread_cycles() {
    unsigned long long c = 0;
#ifdef _WIN32
    QueryThreadCycleTime(GetCurrentThread(), &c);
#endif
    return c;
}

struct Opts {
    Params prm;
    std::vector<Isa> isas;
    int reps = 5, threads = 1;
    std::string out, cache;
    std::vector<std::string> files;
};

static std::vector<uint8_t> encode_cached(const CanonicalMatrix& M, const std::string& file, const Params& prm, const Opts& o, double& enc_ms) {
    std::string key;
    if (!o.cache.empty()) {
        char buf[256];
        std::snprintf(buf, sizeof buf, "%s/%s__%s__L%d_c%d_pb%d_%d_k%d_ch%d_t%u.w3f", o.cache.c_str(), parent_name(file).c_str(),
                      base_name(file).c_str(), prm.lanes, prm.cpo, prm.pb, prm.pbv, prm.ctx, prm.chunk_cols, prm.target_nnz);
        key = buf;
        std::ifstream f(key, std::ios::binary);
        if (f) { enc_ms = 0; return read_file(key); }
    }
    Timer t;
    auto b = encode_matrix(M, prm);
    enc_ms = t.ms();
    if (!key.empty()) write_file(key, b);
    return b;
}

static bool same_cls(const ClsMat& R, const std::vector<int32_t>& indptr, const std::vector<int32_t>& idx, const std::vector<uint32_t>& val) {
    if (indptr.size() != R.ptr.size()) return false;
    for (size_t i = 0; i < indptr.size(); ++i) if (static_cast<uint32_t>(indptr[i]) != R.ptr[i]) return false;
    for (size_t i = 0; i < R.nnz; ++i) if (static_cast<uint32_t>(idx[i]) != R.rk[i] || val[i] != R.val[i]) return false;
    return true;
}

static int cmd_verify(Opts& o) {
    int fails = 0;
    for (auto& f : o.files) {
        CanonicalMatrix M = load_1pz(f);
        double enc_ms;
        auto b = encode_cached(M, f, o.prm, o, enc_ms);
        std::vector<uint32_t> rnnz(M.m, 0);
        for (auto r : M.indices) rnnz[r]++;
        RowOrder O = order_from_counts(rnnz, o.prm.cpo);
        ClsMat R = to_class_space(M, O);
        Prep P;
        parse_file(b.data(), P);
        Sizes sz;
        sz.payload = b.size() - static_cast<size_t>(P.payload_p - b.data()) + 11;
        sz.side = static_cast<size_t>(P.payload_p - P.rowc_p);
        sz.names = static_cast<size_t>(P.rowc_p - P.names_p);
        sz.tables = 0;
        std::printf("== %s m=%u n=%u nnz=%llu | enc %.0f ms | bytes %zu (payload %.3f b/nnz tables %zu side %.3f names %zu) ctx=%d chunk_cols=%d nchunks=%zu\n", f.c_str(), M.m,
                    M.n, (unsigned long long)M.nnz, enc_ms, b.size(), sz.payload * 8.0 / M.nnz, sz.tables, sz.side * 8.0 / M.nnz, sz.names, P.h.cfg.kcap == 11 && P.h.cfg.cap == 39 ? 0 : -1,
                    (int)P.h.chunk_cols, P.chunks.size());
        for (Isa isa : o.isas) {
            std::vector<int32_t> indptr(M.n + 1), idx(M.nnz);
            std::vector<uint32_t> val(M.nnz);
            Timer t;
            decode_all<uint32_t>(P, isa, indptr.data(), idx.data(), val.data(), 1);
            double ms = t.ms();
            bool okc = same_cls(R, indptr, idx, val);
            // canonical
            CanonicalMatrix D = decode_canonical(b, isa);
            bool okm = (D == M);
            std::printf("  %-7s class-order %s (%.1f ms, %.1f Mnnz/s) | canonical-exact %s\n", isa_name(isa), okc ? "OK" : "FAIL", ms, M.nnz / ms / 1e3, okm ? "OK" : "FAIL");
            if (!okc || !okm) ++fails;
        }
    }
    return fails ? 1 : 0;
}

static int cmd_time(Opts& o) {
    bool newf = !std::ifstream(o.out).good();
    std::ofstream tsv;
    if (!o.out.empty()) {
        tsv.open(o.out, std::ios::app);
        if (newf) tsv << "sample\tmember\tlanes\tcpo\tpb\tctx\tchunk_cols\tisa\tthreads\tnnz\tpayload_bits_nnz\tside_bits_nnz\ttotal_bytes\tcls_ms\tcls_Mnnz_s\tcls_est3GHz\tcanon_ms\tcanon_Mnnz_s\tcanon_est3GHz\tcore_ghz\topen_ms\n";
    }
    for (auto& f : o.files) {
        CanonicalMatrix M = load_1pz(f);
        double enc_ms;
        auto b = encode_cached(M, f, o.prm, o, enc_ms);
        auto sz = last_sizes();
        if (enc_ms == 0) {   // cached: recompute sizes from the stream
            Prep Q; parse_file(b.data(), Q);
            const uint8_t* p = Q.payload_p;
            sz.payload = b.size() - (p - b.data()) + 11;
            sz.side = Q.payload_p - Q.rowc_p;
        }
        std::vector<uint32_t> rnnz(M.m, 0);
        for (auto r : M.indices) rnnz[r]++;
        RowOrder O = order_from_counts(rnnz, o.prm.cpo);
        Timer to;
        Prep P;
        parse_file(b.data(), P);
        double open_ms = to.ms();
        std::vector<int32_t> indptr(M.n + 1), idx(M.nnz);
        std::vector<uint32_t> val(M.nnz);
        for (Isa isa : o.isas) {
            double g0 = calib_ghz();
            double best = 1e300;
            for (int r = 0; r < o.reps; ++r) {
                Timer t;
                decode_all<uint32_t>(P, isa, indptr.data(), idx.data(), val.data(), o.threads);
                best = std::min(best, t.ms());
            }
            double bestc = 1e300;
            if (o.threads == 1) {
                std::unique_ptr<tb::CanonVec> cv(new tb::CanonVec(O.perm.data(), M.m));
                for (int r = 0; r < o.reps; ++r) {
                    Timer t;
                    decode_all<uint32_t>(P, isa, indptr.data(), idx.data(), val.data(), 1);
                    for (uint32_t j = 0; j < M.n; ++j)
                        cv->column(reinterpret_cast<uint32_t*>(idx.data()) + indptr[j], val.data() + indptr[j], indptr[j + 1] - indptr[j]);
                    bestc = std::min(bestc, t.ms());
                }
            }
            double g1 = calib_ghz();
            double ghz = std::max(g0, g1);
            double rate = M.nnz / best / 1e3, ratec = bestc < 1e299 ? M.nnz / bestc / 1e3 : 0;
            std::printf("%s/%s L%d cpo%d pb%d ctx%d ch%u %-6s thr%d | payload %.3f b/nnz | class-order %7.1f ms %6.1f Mnnz/s (est@3GHz %6.1f) | canonical %7.1f ms %6.1f Mnnz/s (est@3GHz %6.1f) | core %.2f GHz open %.1f ms\n",
                        parent_name(f).c_str(), base_name(f).c_str(), P.h.L, P.h.cpo, P.h.cfg.pb, 0, P.h.chunk_cols, isa_name(isa), o.threads,
                        sz.payload * 8.0 / M.nnz, best, rate, rate * 3.0 / ghz, bestc < 1e299 ? bestc : 0, ratec, ratec * 3.0 / ghz, ghz, open_ms);
            std::fflush(stdout);
            if (tsv.is_open())
                tsv << parent_name(f) << '\t' << base_name(f) << '\t' << P.h.L << '\t' << P.h.cpo << '\t' << (int)P.h.cfg.pb << '\t' << (int)P.h.cfg.kcap << '\t' << P.h.chunk_cols << '\t'
                    << isa_name(isa) << '\t' << o.threads << '\t' << M.nnz << '\t' << sz.payload * 8.0 / M.nnz << '\t' << sz.side * 8.0 / M.nnz << '\t' << b.size() << '\t' << best << '\t' << rate << '\t'
                    << rate * 3.0 / ghz << '\t' << (bestc < 1e299 ? bestc : 0) << '\t' << ratec << '\t' << ratec * 3.0 / ghz << '\t' << ghz << '\t' << open_ms << '\n';
        }
    }
    return 0;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), 0x80);
    SetThreadPriority(GetCurrentThread(), 2);
#endif
    if (argc < 2) { std::fprintf(stderr, "usage: w3_fast_bench verify|time [opts] files...\n"); return 2; }
    std::string cmd = argv[1];
    Opts o;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--lanes") o.prm.lanes = std::stoi(next());
        else if (a == "--cpo") o.prm.cpo = std::stoi(next());
        else if (a == "--pb") o.prm.pb = std::stoi(next());
        else if (a == "--pbv") o.prm.pbv = std::stoi(next());
        else if (a == "--ctx") o.prm.ctx = std::stoi(next());
        else if (a == "--chunk") o.prm.chunk_cols = std::stoi(next());
        else if (a == "--reps") o.reps = std::stoi(next());
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--out") o.out = next();
        else if (a == "--cache") o.cache = next();
        else if (a == "--isa") {
            std::stringstream ss(next());
            std::string tok;
            while (std::getline(ss, tok, ',')) o.isas.push_back(tok == "scalar" ? Isa::Scalar : tok == "avx2" ? Isa::Avx2 : Isa::Avx512);
        } else o.files.push_back(a);
    }
    if (o.isas.empty()) for (Isa i : {Isa::Scalar, Isa::Avx2, Isa::Avx512}) if (isa_available(i)) o.isas.push_back(i);
    if (cmd == "verify") return cmd_verify(o);
    if (cmd == "time") return cmd_time(o);
    std::fprintf(stderr, "unknown command\n");
    return 2;
}
