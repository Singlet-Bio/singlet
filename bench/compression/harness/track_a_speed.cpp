// Track A: decode speed measured in thread CPU cycles instead of wall time.
//
// The machine is shared with several other jobs and runs all cores at 100%, so wall-clock decode times are inflated by
// 2-6x and vary run to run (the ref reader decodes pbmc_v2_typical exon in 226 ms on an idle box and in 900 ms under load).
// QueryThreadCycleTime counts only the cycles this thread actually ran (kernel + user), so it is immune to being
// descheduled; it is still inflated by SMT siblings and frequency drops, so numbers are "cycles per nnz" and an
// equivalent Mnnz/s at the 3.0 GHz base clock.
//
//   track_a_speed full    --out results/track_a_speed_full.tsv  file.1pz ...   whole-codec decode (names, perm, chunks)
//   track_a_speed entropy --out results/track_a_speed_entropy.tsv file.1pz ... entropy stage alone, chunk_cols 1024
//   track_a_speed ref     --out results/track_a_speed_ref.tsv   file.1pz ...   the production reader on the original bytes
#define NOMINMAX
#include <windows.h>
#include "common.h"
#include "track_a.h"
#include <iostream>
#include <sstream>

using namespace cs;
using namespace cs::ta;

static const double kBaseGHz = 3.0;
static inline uint64_t thread_cycles() {
    ULONG64 c = 0;
    QueryThreadCycleTime(GetCurrentThread(), &c);
    return c;
}
static std::string base_name(const std::string& p) { auto s = p.find_last_of("/\\"); return s == std::string::npos ? p : p.substr(s + 1); }
static std::string parent_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    if (s == std::string::npos || s == 0) return "";
    auto t = p.find_last_of("/\\", s - 1);
    return p.substr(t == std::string::npos ? 0 : t + 1, s - (t == std::string::npos ? 0 : t + 1));
}

struct Meas { double cyc_min = 1e300, wall_min = 1e300; };
template <class F> static Meas measure(F&& f, int reps) {
    Meas m;
    for (int r = 0; r < reps; ++r) {
        uint64_t c0 = thread_cycles();
        Timer t;
        f();
        double w = t.ms();
        uint64_t c1 = thread_cycles();
        m.cyc_min = std::min(m.cyc_min, double(c1 - c0));
        m.wall_min = std::min(m.wall_min, w);
    }
    return m;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: track_a_speed full|entropy|ref [--out tsv] [--reps n] [--codecs a,b] file...\n"; return 2; }
    std::string mode = argv[1], out = "results/track_a_speed_" + mode + ".tsv";
    int reps = 5;
    std::vector<std::string> files, want;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--codecs" && i + 1 < argc) { std::stringstream ss(argv[++i]); std::string t; while (std::getline(ss, t, ',')) want.push_back(t); }
        else files.push_back(a);
    }
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf) tsv << "sample\tmember\tmode\tname\tnnz\tbytes\tdec_cycles_per_nnz\tdec_Mnnz_s_at_3GHz\tdec_wall_ms_min\tdec_cycles_total\n";
    for (auto& f : files) {
        auto M = load_1pz(f);
        std::string sample = parent_name(f), member = base_name(f);
        if (mode == "ref") {
            auto file = read_file(f);
            Meas m = measure([&] { auto r = singlet::pz::read_1pz_buffer(file.data(), file.size()); (void)r; }, reps);
            double cpn = m.cyc_min / double(M.nnz);
            std::printf("%-22s %-18s production read_1pz_buffer: %7.1f cycles/nnz = %6.2f Mnnz/s @3GHz (wall min %.0f ms)\n", sample.c_str(), member.c_str(), cpn, kBaseGHz * 1e3 / cpn, m.wall_min);
            tsv << sample << '\t' << member << "\tref\tread_1pz_buffer\t" << M.nnz << '\t' << file.size() << '\t' << cpn << '\t' << kBaseGHz * 1e3 / cpn << '\t' << m.wall_min << '\t' << m.cyc_min << '\n';
            tsv.flush();
        } else if (mode == "full") {
            std::vector<Codec> codecs;
            cs::register_track_a(codecs);
            for (auto& c : codecs) {
                if (!want.empty() && std::find(want.begin(), want.end(), c.name) == want.end()) continue;
                auto enc = c.encode(M);
                CanonicalMatrix D;
                Meas m = measure([&] { D = c.decode(enc); }, reps);
                bool ok = (D == M);
                double cpn = m.cyc_min / double(M.nnz);
                std::printf("%-22s %-18s %-24s %10zu B  %7.1f cycles/nnz = %6.2f Mnnz/s @3GHz (wall min %.0f ms) %s\n", sample.c_str(), member.c_str(), c.name.c_str(), enc.size(), cpn, kBaseGHz * 1e3 / cpn, m.wall_min, ok ? "OK" : "MISMATCH");
                tsv << sample << '\t' << member << "\tfull\t" << c.name << '\t' << M.nnz << '\t' << enc.size() << '\t' << cpn << '\t' << kBaseGHz * 1e3 / cpn << '\t' << m.wall_min << '\t' << m.cyc_min << '\n';
                tsv.flush();
            }
        } else {   // entropy
            Prepared P = prepare(M, 1024, nullptr, 1);
            std::vector<std::string> labels = want.empty() ? std::vector<std::string>{"store", "z1", "z3", "z6", "z9", "z12", "z15", "z19", "z22", "xz", "br", "bz2"} : want;
            for (auto& l : labels) {
                EC ec;
                if (l == "store") ec = ec_store();
                else if (l == "xz") ec = ec_xz();
                else if (l == "br") ec = ec_brotli(11, 24);
                else if (l == "bz2") ec = ec_bzip2();
                else if (l == "z22l") ec = ec_zstd(22, 27, true);
                else ec = ec_zstd(std::atoi(l.c_str() + 1));
                auto blobs = entropy_chunks(P, ec);
                size_t payload = 0;
                for (auto& b : blobs) payload += b.size() - 20;
                std::vector<uint8_t> scratch;
                Meas m = measure([&] {
                    for (size_t c = 0; c < blobs.size(); ++c) {
                        uint32_t zsz, psz;
                        std::memcpy(&zsz, blobs[c].data() + 8, 4); std::memcpy(&psz, blobs[c].data() + 16, 4);
                        scratch.resize(psz);
                        ec_decompress(ec, blobs[c].data() + 20, zsz, scratch.data(), psz);
                    }
                }, reps);
                double cpn = m.cyc_min / double(M.nnz);
                std::printf("%-22s %-18s entropy %-5s payload %9zu B %6.3f bits/nnz  decode %7.2f cycles/nnz = %7.1f Mnnz/s @3GHz\n", sample.c_str(), member.c_str(), l.c_str(), payload, payload * 8.0 / double(M.nnz), cpn, kBaseGHz * 1e3 / cpn);
                tsv << sample << '\t' << member << "\tentropy\t" << l << '\t' << M.nnz << '\t' << payload << '\t' << cpn << '\t' << kBaseGHz * 1e3 / cpn << '\t' << m.wall_min << '\t' << m.cyc_min << '\n';
                tsv.flush();
            }
        }
    }
    return 0;
}
