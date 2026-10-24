// Track B driver: like bench.exe but (a) only Track B codecs, (b) reports payload bits/nnz separately from names/side
// info, (c) times payload-only decode (native rank order and canonical) next to the full decode().
// Timing: wall clock (min of reps) and thread cycles (QueryThreadCycleTime, min of reps, reported as ms at the
// 3.0 GHz nominal clock and as cycles/nnz) because the machine is shared with other agents.
//   bin/track_b_bench.exe [--codec a,b] [--out results/track_b_x.tsv] [--reps 5] [--list] file.1pz ...
#include "common.h"
#include "track_b.h"
#include <algorithm>
#include <iostream>
#include <sstream>

using namespace cs;

static std::string base_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}
static std::string parent_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    if (s == std::string::npos || s == 0) return "";
    auto t = p.find_last_of("/\\", s - 1);
    size_t start = (t == std::string::npos) ? 0 : t + 1;
    return p.substr(start, s - start);
}

// Effective core clock (GHz) from a 1-cycle dependency chain: the laptop is power/thermally throttled and shared.
static double calib_ghz() {
    uint64_t s = 0;
    Timer t;
    for (uint64_t i = 0; i < 10000000ull; ++i) {
        asm volatile("add $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0" : "+r"(s));
    }
    return 1e8 / t.ms() / 1e6;
}

struct Meas { double wall_ms = 1e300, cyc = 1e300, ghz = 0; };
template <class F> static Meas measure(int reps, F&& fn) {
    Meas m;
    m.ghz = calib_ghz();
    for (int r = 0; r < reps; ++r) {
        tb::CycTimer ct;
        Timer t;
        fn();
        m.wall_ms = std::min(m.wall_ms, t.ms());
        m.cyc = std::min(m.cyc, ct.cycles());
    }
    m.ghz = std::max(m.ghz, calib_ghz());
    return m;
}

#ifdef _WIN32
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess(void);
extern "C" __declspec(dllimport) int __stdcall SetPriorityClass(void*, unsigned long);
extern "C" __declspec(dllimport) int __stdcall SetThreadPriority(void*, int);
extern "C" __declspec(dllimport) void* __stdcall GetCurrentThread(void);
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    // The laptop is shared with other agents: ask for HIGH process priority (not REALTIME) to cut time-slicing noise in timings.
    SetPriorityClass(GetCurrentProcess(), 0x80 /*HIGH_PRIORITY_CLASS*/);
    SetThreadPriority(GetCurrentThread(), 2 /*THREAD_PRIORITY_HIGHEST*/);
#endif
    std::vector<std::string> files, want;
    std::string out = "results/track_b_run.tsv";
    int reps = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--codec" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) want.push_back(tok);
        } else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--list") {
            std::vector<Codec> v;
            register_track_b(v);
            for (auto& c : v) std::cout << c.name << "\n";
            return 0;
        } else files.push_back(a);
    }
    std::vector<Codec> codecs;
    register_track_b(codecs);
    if (!want.empty()) {
        std::vector<Codec> sel;
        for (auto& w : want)
            for (auto& c : codecs) if (c.name == w) sel.push_back(c);
        codecs = sel;
    }
    bool new_file = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (new_file)
        tsv << "sample\tmember\tcodec\tnnz\tm\tn\ttotal_bytes\tpayload_bytes\tside_bytes\tnames_bytes\ttables_bytes\tcounts_bytes\t"
               "payload_bits_nnz\ttotal_bits_nnz\tenc_ms\tdec_ms_full\tfull_Mnnz_s\t"
               "payload_ms_canon\tpayload_Mnnz_s_canon\tpayload_ms_native\tpayload_Mnnz_s_native\t"
               "cyc_nnz_full\tcyc_nnz_canon\tcyc_nnz_native\tMnnz_s_native_3GHz\tMnnz_s_canon_3GHz\t"
               "core_ghz\tcorecyc_nnz_native\tcorecyc_nnz_canon\test_Mnnz_s_native_core3GHz\test_Mnnz_s_canon_core3GHz\tmatrix_ok\n";
    for (auto& f : files) {
        CanonicalMatrix M = load_1pz(f);
        const double nnz = (double)M.nnz;
        std::printf("== %s  m=%u n=%u nnz=%llu\n", f.c_str(), M.m, M.n, (unsigned long long)M.nnz);
        for (auto& c : codecs) {
            Timer te;
            auto enc = c.encode(M);
            double enc_ms = te.ms();
            auto bd = tb::last_breakdown();
            CanonicalMatrix D;
            Meas full = measure(reps, [&] { D = c.decode(enc); });
            bool ok = (D == M);
            Meas pc, pn;
            auto it = tb::payload_hooks().find(c.name);
            if (it != tb::payload_hooks().end()) {
                auto fn_n = it->second(enc, false);
                pn = measure(reps, fn_n);
                auto fn_c = it->second(enc, true);
                pc = measure(reps, fn_c);
            }
            auto rate = [&](double ms) { return ms > 0 && ms < 1e299 ? nnz / ms / 1e3 : 0.0; };
            auto cpn = [&](const Meas& m) { return m.cyc < 1e299 ? m.cyc / nnz : -1.0; };
            auto corecyc = [&](const Meas& m) { return m.wall_ms < 1e299 ? m.wall_ms * m.ghz * 1e6 / nnz : -1.0; };
            auto estr = [&](const Meas& m) { return m.wall_ms < 1e299 ? 3000.0 / corecyc(m) : 0.0; };
            auto r3 = [&](const Meas& m) { return m.cyc < 1e299 ? tb::kNominalGHz * 1e3 / (m.cyc / nnz) : 0.0; };
            std::printf("  %-18s total %9zu B (%6.2f b/nnz) payload %9zu B (%6.2f b/nnz) | enc %7.0f ms | full %6.1f ms %5.1f Mnnz/s | "
                        "payload canon %6.1f ms (%5.1f cyc/nnz = %5.0f Mnnz/s@3GHz) native %6.1f ms (%5.1f cyc/nnz = %5.0f Mnnz/s@3GHz) %s\n",
                        c.name.c_str(), enc.size(), enc.size() * 8.0 / nnz, bd.payload, bd.payload * 8.0 / nnz, enc_ms, full.wall_ms,
                        rate(full.wall_ms), pc.wall_ms, cpn(pc), r3(pc), pn.wall_ms, cpn(pn), r3(pn), ok ? "OK" : "FAIL");
            tsv << parent_name(f) << '\t' << base_name(f) << '\t' << c.name << '\t' << M.nnz << '\t' << M.m << '\t' << M.n << '\t'
                << enc.size() << '\t' << bd.payload << '\t' << bd.side << '\t' << bd.names << '\t' << bd.tables << '\t' << bd.counts << '\t'
                << bd.payload * 8.0 / nnz << '\t' << enc.size() * 8.0 / nnz << '\t' << enc_ms << '\t' << full.wall_ms << '\t' << rate(full.wall_ms) << '\t'
                << pc.wall_ms << '\t' << rate(pc.wall_ms) << '\t' << pn.wall_ms << '\t' << rate(pn.wall_ms) << '\t'
                << cpn(full) << '\t' << cpn(pc) << '\t' << cpn(pn) << '\t' << r3(pn) << '\t' << r3(pc) << '\t'
                << std::max(pn.ghz, pc.ghz) << '\t' << corecyc(pn) << '\t' << corecyc(pc) << '\t' << estr(pn) << '\t' << estr(pc) << '\t'
                << (ok ? 1 : 0) << '\n';
            tsv.flush();
            if (!ok) std::printf("  !!! roundtrip FAILED for %s\n", c.name.c_str());
        }
    }
    return 0;
}
