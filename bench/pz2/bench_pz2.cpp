// SPDX-License-Identifier: MIT
// pz2 benchmark driver.
//   bench_pz2 time  [--reps 5] [--threads 1,2,4] [--isas scalar,avx2,avx512] [--cache DIR] [--chunk N] files.1pz...
//        decode throughput (Mnnz/s of whole-matrix decode, min and median of reps) per ISA x mode x threads
//   bench_pz2 chunks [--cols 128,256,512,1024,4096] [--cache DIR] [--reps 5] files.1pz...
//        payload bytes and random-access cost per chunk_cols: one random column, 1000 random columns (in-memory decode time,
//        distinct chunks touched, bytes a remote client would fetch)
//   bench_pz2 widen  files.1pz...      size / decode cost sanity numbers for the 2^21 row-id widening (see notes/m_w1.md)
// Output is TSV on stdout.  The core clock is calibrated with a dependent-add chain (1 cycle per add) before and after every file.
#include <singlet/pileup/pz_reader.h>
#include <singlet/pz2/pz2.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>

using namespace singlet::pz2;
namespace fs = std::filesystem;
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static double core_ghz() {   // 1 cycle per dependent add on every x86 core
    double best = 0;
    for (int rep = 0; rep < 5; ++rep) {
        const uint64_t N = 200000000ull;
        uint64_t x = 1;
        double t = now_s();
        for (uint64_t i = 0; i < N; i += 8) {
#if defined(__GNUC__)
            __asm__ volatile("add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0" : "+r"(x));
#else
            x += 8;
#endif
        }
        t = now_s() - t;
        best = std::max(best, double(N) / t / 1e9);
        if (x == 0) std::puts("");
    }
    return best;
}

static std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, c)) if (!t.empty()) v.push_back(t);
    return v;
}
static std::string tag_of(const std::string& path) {
    fs::path p(path);
    return p.parent_path().filename().string() + "__" + p.stem().string();
}

struct Loaded {
    singlet::pz::ReadResult r;
    CSC c;
};
static std::vector<uint8_t> read_all(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void write_all(const std::string& p, const std::vector<uint8_t>& b) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
}

static EncodeOptions base_opts() {
    EncodeOptions o;
    o.row_names.kind = NamesKind::None; o.col_names.kind = NamesKind::None;   // payload only
    o.compute_digest = false;
    o.threads = 4;
    return o;
}

static std::vector<uint8_t> get_member(const std::string& path, const std::string& cache, uint32_t chunk_cols, Loaded* keep, const EncodeOptions* eo = nullptr,
                                       uint32_t target = 0) {
    std::string cp = cache + "/" + tag_of(path) + (chunk_cols ? ".c" + std::to_string(chunk_cols) : "") + (target ? ".t" + std::to_string(target) : "") + ".pz2";
    if (!eo && fs::exists(cp)) return read_all(cp);
    Loaded L;
    L.r = singlet::pz::read_1pz(path);
    L.c.m = L.r.m; L.c.n = L.r.n; L.c.nnz = L.r.nnz; L.c.indptr = L.r.indptr.data(); L.c.indices = L.r.indices.data(); L.c.data = L.r.data.data();
    EncodeOptions o = eo ? *eo : base_opts();
    o.chunk_cols = chunk_cols;
    o.chunk_nnz_target = target;
    if (L.r.vt_code >= 1 && L.r.vt_code <= 3) o.value_type = ValueType(L.r.vt_code);
    auto b = encode_matrix(L.c, o);
    if (!eo) { fs::create_directories(cache); write_all(cp, b); }
    if (keep) *keep = std::move(L);
    return b;
}

struct Stats { double mn, med; };
static Stats time_reps(int reps, const std::function<void()>& fn) {
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) { double a = now_s(); fn(); t.push_back(now_s() - a); }
    std::sort(t.begin(), t.end());
    return {t.front(), t[t.size() / 2]};
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: bench_pz2 time|chunks|widen [options] files...\n"); return 2; }
    const std::string cmd = argv[1];
    int reps = 5;
    std::string cache = "bench/compression/results/m_w1/cache";
    std::vector<int> threads = {1, 2, 4};
    std::vector<std::string> isa_names = {"scalar", "avx2", "avx512"};
    std::vector<uint32_t> cols = {128, 256, 512, 1024, 4096};
    std::vector<uint32_t> targets;
    uint32_t chunk = 0;
    EncodeOptions sopt = base_opts();   // for `sizes`: --pb --pbv --lanes --cpo --kind --target
    bool sopt_set = false;
    std::vector<std::string> files;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--reps") reps = std::atoi(argv[++i]);
        else if (a == "--cache") cache = argv[++i];
        else if (a == "--threads") { threads.clear(); for (auto& s : split(argv[++i], ',')) threads.push_back(std::atoi(s.c_str())); }
        else if (a == "--isas") isa_names = split(argv[++i], ',');
        else if (a == "--cols") { cols.clear(); for (auto& s : split(argv[++i], ',')) cols.push_back(uint32_t(std::atoi(s.c_str()))); }
        else if (a == "--targets") { targets.clear(); for (auto& s2 : split(argv[++i], ',')) targets.push_back(uint32_t(std::atoi(s2.c_str()))); }
        else if (a == "--pb") { sopt.prob_bits = uint8_t(std::atoi(argv[++i])); sopt_set = true; }
        else if (a == "--pbv") { sopt.value_prob_bits = uint8_t(std::atoi(argv[++i])); sopt_set = true; }
        else if (a == "--lanes") { sopt.lanes = uint8_t(std::atoi(argv[++i])); sopt_set = true; }
        else if (a == "--cpo") { sopt.classes_per_octave = uint8_t(std::atoi(argv[++i])); sopt_set = true; }
        else if (a == "--kind") { sopt.ctx_kind = std::atoi(argv[++i]); sopt_set = true; }
        else if (a == "--chunk") chunk = uint32_t(std::atoi(argv[++i]));
        else files.push_back(a);
    }
    std::fprintf(stderr, "%s\n", isa_report().c_str());

    if (cmd == "time") {
        std::printf("file\tm\tn\tnnz\tbytes\tbits_per_nnz\tcols_per_chunk\tisa\tmode\tthreads\tmin_ms\tmed_ms\tMnnz_s_min\tcore_GHz\tMnnz_s_at_3GHz\n");
        for (auto& f : files) {
            std::vector<uint8_t> bytes = get_member(f, cache, chunk, nullptr);
            ByteSource src(bytes);
            Matrix mx = Matrix::open(src);
            const Header& h = mx.header();
            std::vector<uint32_t> ip(h.n + 1), ix(h.nnz + 64), dv(h.nnz + 64);
            Output o;
            o.indptr = ip.data(); o.indptr_capacity = ip.size(); o.indices = ix.data(); o.indices_capacity = ix.size(); o.data = dv.data(); o.data_capacity = dv.size();
            for (auto& in : isa_names) {
                Isa isa = in == "scalar" ? Isa::Scalar : in == "avx2" ? Isa::Avx2 : Isa::Avx512;
                if (!isa_supported(isa)) continue;
                for (Mode mode : {Mode::Native, Mode::Canonical}) {
                    for (int th : threads) {
                        if (th > 1 && isa != best_isa()) continue;   // thread scaling only for the best path
                        DecodeOptions d;
                        d.mode = mode; d.isa = isa; d.threads = th;
                        mx.decode(src, o, d);   // warm up
                        double g0 = core_ghz();
                        Stats st = time_reps(reps, [&] { mx.decode(src, o, d); });
                        double g1 = core_ghz();
                        double g = std::min(g0, g1);
                        std::printf("%s\t%u\t%u\t%llu\t%zu\t%.3f\t%u\t%s\t%s\t%d\t%.2f\t%.2f\t%.1f\t%.2f\t%.1f\n", tag_of(f).c_str(), h.m, h.n, (unsigned long long)h.nnz, bytes.size(),
                                    8.0 * double(bytes.size()) / double(std::max<uint64_t>(1, h.nnz)), h.chunk_cols, isa_name(isa), mode == Mode::Native ? "native" : "canonical", th,
                                    st.mn * 1e3, st.med * 1e3, double(h.nnz) / st.mn / 1e6, g, double(h.nnz) / st.mn / 1e6 * 3.0 / g);
                        std::fflush(stdout);
                    }
                }
            }
        }
        return 0;
    }

    if (cmd == "chunks") {
        // variants: fixed widths (--cols) and nnz-target policies (--targets)
        struct Var { std::string label; uint32_t cols, target; };
        std::vector<Var> vars;
        for (uint32_t c : cols) vars.push_back({std::to_string(c), c, 0});
        for (uint32_t t : targets) vars.push_back({"t" + std::to_string(t / 1024) + "k", 0, t});
        const int Ks[4] = {1, 10, 100, 1000};
        std::printf("file\tvariant\tcols_per_chunk\tn_chunks\tbytes\tvs_1024\tprefix_bytes\tmean_chunk_bytes\tone_col_us_median\tone_col_us_p90");
        for (int K : Ks) std::printf("\tK%d_chunks\tK%d_fetch_bytes\tK%d_runs\tK%d_decode_ms", K, K, K, K);
        std::printf("\twhole_decode_ms\n");
        for (auto& f : files) {
            size_t base1024 = get_member(f, cache, 1024, nullptr).size();
            for (auto& v : vars) {
                std::vector<uint8_t> bytes = get_member(f, cache, v.cols, nullptr, nullptr, v.target);
                ByteSource src(bytes);
                Matrix mx = Matrix::open(src);
                const Header& h = mx.header();
                std::mt19937 rng(777);
                DecodeOptions d;
                d.mode = Mode::Native;
                std::vector<double> lat;
                std::vector<uint32_t> ip(2);
                for (int it = 0; it < 300 && h.n; ++it) {
                    uint32_t c = rng() % h.n;
                    ColumnPlan pl = mx.plan_columns(c, c + 1);
                    std::vector<uint32_t> ix(pl.max_nnz + 1), dv(pl.max_nnz + 1);
                    Output o;
                    o.indptr = ip.data(); o.indptr_capacity = 2; o.indices = ix.data(); o.indices_capacity = ix.size(); o.data = dv.data(); o.data_capacity = dv.size();
                    double a0 = now_s();
                    mx.decode_columns(src, c, c + 1, o, d);
                    lat.push_back(now_s() - a0);
                }
                std::sort(lat.begin(), lat.end());
                const double mean_chunk = h.n_chunks ? double(h.section[kChunks].len) / h.n_chunks : 0;
                std::printf("%s\t%s\t%u\t%u\t%zu\t%.4f\t%llu\t%.0f\t%.1f\t%.1f", tag_of(f).c_str(), v.label.c_str(), h.chunk_cols, h.n_chunks, bytes.size(),
                            double(bytes.size()) / double(base1024), (unsigned long long)h.prefix_len(), mean_chunk, lat.empty() ? 0 : lat[lat.size() / 2] * 1e6,
                            lat.empty() ? 0 : lat[lat.size() * 9 / 10] * 1e6);
                for (int K : Ks) {
                    size_t chunks_touched = 0, fetch_bytes = h.prefix_len(), runs = 0;
                    double ms = 0;
                    if (h.n) {
                        std::vector<uint32_t> all(h.n);
                        for (uint32_t i = 0; i < h.n; ++i) all[i] = i;
                        std::shuffle(all.begin(), all.end(), rng);
                        uint32_t want = std::min<uint32_t>(uint32_t(K), h.n);
                        std::vector<uint32_t> pick(all.begin(), all.begin() + want);
                        std::sort(pick.begin(), pick.end());
                        std::vector<size_t> touched;
                        for (uint32_t c : pick) { size_t ch = mx.chunk_index().chunk_of_column(c); if (touched.empty() || touched.back() != ch) touched.push_back(ch); }
                        chunks_touched = touched.size();
                        for (size_t i = 0; i < touched.size(); ++i) {
                            fetch_bytes += mx.chunk_index().rec[touched[i]].len;
                            if (i == 0 || touched[i] != touched[i - 1] + 1) ++runs;
                        }
                        double a0 = now_s();
                        for (size_t ch : touched) {
                            uint32_t c0 = mx.chunk_index().rec[ch].first_col, c1 = mx.chunk_index().rec[ch + 1].first_col;
                            std::vector<uint32_t> ipp(c1 - c0 + 1), ix(mx.chunk_index().nnz_in_chunk(ch) + 1), dv(ix.size());
                            Output o;
                            o.indptr = ipp.data(); o.indptr_capacity = ipp.size(); o.indices = ix.data(); o.indices_capacity = ix.size(); o.data = dv.data(); o.data_capacity = dv.size();
                            mx.decode_columns(src, c0, c1, o, d);
                        }
                        ms = (now_s() - a0) * 1e3;
                    }
                    std::printf("\t%zu\t%zu\t%zu\t%.2f", chunks_touched, fetch_bytes, runs, ms);
                }
                std::vector<uint32_t> ip2(h.n + 1), ix2(h.nnz + 64), dv2(h.nnz + 64);
                Output ow;
                ow.indptr = ip2.data(); ow.indptr_capacity = ip2.size(); ow.indices = ix2.data(); ow.indices_capacity = ix2.size(); ow.data = dv2.data(); ow.data_capacity = dv2.size();
                Stats sw = time_reps(reps, [&] { mx.decode(src, ow, d); });
                std::printf("\t%.2f\n", sw.mn * 1e3);
                std::fflush(stdout);
            }
        }
        return 0;
    }
    if (cmd == "sizes") {   // section sizes and bits per nnz (names excluded); --chunk N fixes the chunk width, --lanes etc. via env-free flags below
        std::printf("file\tm\tn\tnnz\tcols_per_chunk\tn_chunks\ttables\tchunk_index\tchunks\trows\ttotal\tpayload_bits_nnz\trows_bits_nnz\ttotal_bits_nnz\n");
        for (auto& f : files) {
            std::vector<uint8_t> bytes = get_member(f, cache, chunk, nullptr, sopt_set ? &sopt : nullptr);
            Header h = read_header(bytes.data(), bytes.size());
            const double nz = double(std::max<uint64_t>(1, h.nnz));
            const double pay = double(h.section[kTables].len + h.section[kChunkIndex].len + h.section[kChunks].len);
            std::printf("%s\t%u\t%u\t%llu\t%u\t%u\t%llu\t%llu\t%llu\t%llu\t%zu\t%.3f\t%.3f\t%.3f\n", tag_of(f).c_str(), h.m, h.n, (unsigned long long)h.nnz, h.chunk_cols, h.n_chunks,
                        (unsigned long long)h.section[kTables].len, (unsigned long long)h.section[kChunkIndex].len, (unsigned long long)h.section[kChunks].len,
                        (unsigned long long)h.section[kRows].len, bytes.size(), pay * 8 / nz, double(h.section[kRows].len) * 8 / nz, double(bytes.size()) * 8 / nz);
        }
        return 0;
    }
    std::fprintf(stderr, "unknown command\n");
    return 2;
}
