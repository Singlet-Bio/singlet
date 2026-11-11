// D-101 all-site pass: the device route (allsite_device.h, src/engine_gpu/allsite_gpu.cu) against the host pass (allsite.h run()) on
// random alignment records over a four-contig STAR-style genome: spliced, clipped and indel CIGARs, both strands, secondary and low-MAPQ
// alignments, chrM, barcodes outside the set, raw UMI codes over 3, UMIs one base apart (R9), a (barcode, site) group with more than 64
// UMIs (the deep R9 path), molecules past the 255-read saturation, low-quality masks, N bases, and several chunks and blocks. Sites,
// matrices, per-site sums and every counter must match. Exits 77 (ctest: skipped) without a GPU build or a CUDA device.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include "singlet/pileup/allsite.h"

namespace A = singlet::allsite;
static int n_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cout << "FAIL " << msg << "\n"; ++n_fail; } } while (0)

struct Sig { uint32_t aln_off; uint8_t n_aln; uint8_t nh; uint16_t rlen; uint64_t seq_off; };
struct Aln { int32_t tid; int32_t pos; uint32_t ops_off; uint16_t n_ops; uint8_t flags; uint8_t mapq; };
struct Rd { uint32_t sig; int32_t bc; uint64_t umi3; };

struct Recs {
    std::vector<Sig> S; std::vector<Aln> Al; std::vector<Rd> R; std::vector<uint32_t> ops; std::vector<uint8_t> seq; std::vector<uint64_t> lq;
    uint64_t n_sigs() const { return S.size(); }
    uint64_t n_reads() const { return R.size(); }
    const Sig& sig(uint64_t i) const { return S[i]; }
    const Aln& aln(uint64_t i) const { return Al[i]; }
    const Rd& read(uint64_t i) const { return R[i]; }
    const uint32_t* ops_at(uint64_t o) const { return ops.data() + o; }
    const uint8_t* seq_at(uint64_t o) const { return seq.data() + o; }
    uint64_t lowq_word(uint64_t i) const { return i < lq.size() ? lq[i] : 0; }
};

int main() {
#ifndef SINGLET_HAVE_ENGINE_GPU
    std::cout << "SKIP: CPU-only build\n";
    return 77;
#else
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("allsite_gpu_test_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    std::mt19937_64 g(20261109);
    auto U = [&](uint64_t n) { return n ? g() % n : 0; };
    const std::vector<std::string> names = {"chr1", "chrM", "chr2", "chr3"};
    const std::vector<uint64_t> lens = {200000, 16000, 150000, 50000};
    std::vector<std::vector<uint8_t>> ref(4);
    for (int t = 0; t < 4; ++t) {
        ref[t].resize(lens[t]);
        for (auto& b : ref[t]) b = static_cast<uint8_t>(U(4));
        for (int k = 0; k < 5; ++k) { const uint64_t a = U(lens[t] - 100); for (int j = 0; j < 30; ++j) ref[t][a + j] = 4; }   // N runs
    }
    {   std::ofstream fn(dir / "chrName.txt"), fs_(dir / "chrStart.txt"), fl(dir / "chrNameLength.txt");
        std::ofstream fg(dir / "Genome", std::ios::binary);
        uint64_t start = 0;
        for (int t = 0; t < 4; ++t) {
            fn << names[t] << "\n"; fl << names[t] << "\t" << lens[t] << "\n"; fs_ << start << "\n";
            const uint64_t pad = (lens[t] + 65535) / 65536 * 65536;
            for (uint64_t i = 0; i < pad; ++i) fg.put(static_cast<char>(i < lens[t] ? ref[t][i] : 5));
            start += pad;
        }
        fs_ << start << "\n"; }
    A::Genome G; std::string err;
    if (!G.open(dir.string(), names, &err)) { std::cout << "FAIL genome " << err << "\n"; return 1; }

    // hot windows ("genes") with variant positions; two donors split the barcodes
    struct Win { int t; int32_t a, len; std::vector<int32_t> var; std::vector<uint8_t> altB; };
    std::vector<Win> wins;
    for (int w = 0; w < 40; ++w) {
        Win x; x.t = w % 5 == 0 ? 1 : (w % 3 == 0 ? 3 : (w % 2 ? 0 : 2)); x.len = 1500 + static_cast<int32_t>(U(1500));
        x.a = static_cast<int32_t>(U(lens[x.t] - x.len - 600));
        for (int k = 0; k < 6; ++k) { x.var.push_back(x.a + static_cast<int32_t>(U(x.len))); x.altB.push_back(static_cast<uint8_t>(U(3))); }
        wins.push_back(x);
    }
    const uint32_t nBc = 1000;
    Recs R;
    const int qwords = 2;
    auto randUmi = [&]() { uint64_t u = 0; for (int i = 0; i < 12; ++i) u |= static_cast<uint64_t>(U(4)) << (3 * i); return u; };
    auto addRead = [&](int t, int32_t pos, int rlenTarget, bool deep, int32_t bc, uint64_t umi, int donor) {
        // CIGAR: soft clip, M blocks with an optional splice, insertion and deletion
        std::vector<uint32_t> cig; std::vector<uint8_t> s;   // s: nt16 codes in alignment orientation
        auto m = [&](int32_t rp, int n) {
            for (int j = 0; j < n; ++j) {
                const int32_t p = rp + j; uint8_t b = (p >= 0 && p < static_cast<int32_t>(lens[t])) ? ref[t][p] : 4;
                for (const auto& w : wins) if (w.t == t) for (size_t k = 0; k < w.var.size(); ++k)
                    if (w.var[k] == p && b < 4 && ((donor + static_cast<int>(k)) % 2 == 0)) b = static_cast<uint8_t>((b + 1 + w.altB[k]) & 3);
                if (U(100) == 0) b = static_cast<uint8_t>(U(4));   // sequencing error
                uint8_t code = b < 4 ? static_cast<uint8_t>(1u << b) : 15;
                if (U(400) == 0) code = 15;
                s.push_back(code);
            }
        };
        int32_t rp = pos; int left = rlenTarget;
        const int clip = U(4) == 0 ? 1 + static_cast<int>(U(5)) : 0;
        if (clip) { cig.push_back((static_cast<uint32_t>(clip) << 4) | 4); for (int j = 0; j < clip; ++j) s.push_back(static_cast<uint8_t>(1u << U(4))); left -= clip; }
        const int a1 = std::max(10, left / 2 + static_cast<int>(U(20)) - 10);
        cig.push_back((static_cast<uint32_t>(a1) << 4) | 0); m(rp, a1); rp += a1; left -= a1;
        const int kind = static_cast<int>(U(6));
        if (kind == 0) { const int n = 100 + static_cast<int>(U(400)); cig.push_back((static_cast<uint32_t>(n) << 4) | 3); rp += n; }
        else if (kind == 1) { const int n = 1 + static_cast<int>(U(3)); cig.push_back((static_cast<uint32_t>(n) << 4) | 1); for (int j = 0; j < n; ++j) s.push_back(static_cast<uint8_t>(1u << U(4))); left -= n; }
        else if (kind == 2) { const int n = 1 + static_cast<int>(U(3)); cig.push_back((static_cast<uint32_t>(n) << 4) | 2); rp += n; }
        if (left > 0) { cig.push_back((static_cast<uint32_t>(left) << 4) | (U(10) == 0 ? 7 : 0)); m(rp, left); }
        const bool rev = U(2) == 1;
        std::vector<uint8_t> rec = s;
        if (rev) { std::reverse(rec.begin(), rec.end()); for (auto& c : rec) { static const uint8_t t16[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}; c = t16[c]; } }
        Sig sg{static_cast<uint32_t>(R.Al.size()), 1, 1, static_cast<uint16_t>(rec.size()), R.seq.size()};
        for (size_t i = 0; i < rec.size(); i += 2) R.seq.push_back(static_cast<uint8_t>(rec[i] | ((i + 1 < rec.size() ? rec[i + 1] : 0) << 4)));
        const uint8_t mapq = U(12) == 0 ? 3 : 255;
        if (U(8) == 0) {   // a secondary alignment listed first
            sg.n_aln = 2;
            R.Al.push_back({t, pos + 7, static_cast<uint32_t>(R.ops.size()), static_cast<uint16_t>(cig.size()), static_cast<uint8_t>(rev ? 1 : 0), mapq});
            R.ops.insert(R.ops.end(), cig.begin(), cig.end());
        }
        R.Al.push_back({t, pos, static_cast<uint32_t>(R.ops.size()), static_cast<uint16_t>(cig.size()), static_cast<uint8_t>(2 | (rev ? 1 : 0)), mapq});
        R.ops.insert(R.ops.end(), cig.begin(), cig.end());
        R.S.push_back(sg);
        const uint32_t sid = static_cast<uint32_t>(R.S.size() - 1);
        const int dup = deep ? 0 : static_cast<int>(U(3));   // PCR duplicates share the signature
        for (int d = 0; d <= dup; ++d) {
            R.R.push_back({sid, bc, umi});
            uint64_t w0 = 0, w1 = 0;
            for (int i = 0; i < static_cast<int>(rec.size()); ++i) if (U(50) == 0) (i < 64 ? w0 : w1) |= 1ull << (i & 63);
            R.lq.push_back(w0); R.lq.push_back(w1);
        }
    };
    for (int mol = 0; mol < 40000; ++mol) {
        const Win& w = wins[U(wins.size())];
        int32_t bc = static_cast<int32_t>(U(nBc / 2 + 40)) - 20;   // a few outside [0, nBc)
        if (bc < 0 && U(2)) bc = static_cast<int32_t>(U(nBc));
        if (bc >= static_cast<int32_t>(nBc)) bc = -1;
        const int donor = bc >= 0 && bc < 250 ? 0 : 1;
        uint64_t umi = randUmi();
        if (U(30) == 0) umi |= static_cast<uint64_t>(4 + U(4)) << (3 * U(12));   // raw codes over 3 (N)
        const int32_t pos = w.a + static_cast<int32_t>(U(w.len));
        int nr = 1 + static_cast<int>(U(4)); if (U(500) == 0) nr = 400;   // past the 255 saturation
        for (int r = 0; r < nr; ++r) {
            uint64_t u = umi;
            if (U(6) == 0) u ^= static_cast<uint64_t>(1 + U(3)) << (3 * U(12));   // one base apart: R9 joins or keeps them
            addRead(w.t, pos + static_cast<int32_t>(U(3)), 70 + static_cast<int>(U(30)), nr == 400, bc, u, donor);
        }
    }
    {   // a (barcode, site) group with 150 UMIs: the deep R9 path
        const Win& w = wins[1];
        const uint64_t base = randUmi();
        for (int k = 0; k < 150; ++k) {
            uint64_t u = base ^ (static_cast<uint64_t>(1 + (k % 3)) << (3 * (k % 12)));
            if (k >= 36) u = randUmi();
            for (int r = 0; r < 1 + k % 4; ++r) addRead(w.t, w.var[0] - 20, 80, false, 5, u, 0);
        }
    }
    std::cout << "records: " << R.S.size() << " signatures, " << R.R.size() << " reads\n";

    struct Cfg { int use_qual, skip_n, directional; const char* chunk; const char* block; const char* split; };
    const Cfg cfgs[] = {{1, 0, 1, nullptr, nullptr, nullptr}, {0, 1, 1, "40000", "200000", "3000000"}, {1, 0, 0, nullptr, "150000", "8000000"}, {0, 0, 1, "1", nullptr, "f0.7"}};
    int ran = 0;
    const int pcts[] = {1, 0, 25, 1};   // the ALT-fraction floor: default, off, one that drops many sites, default
    int ci = 0;
    for (const Cfg& c : cfgs) {
        const int pct = pcts[ci++];
        if (c.chunk) setenv("SINGLET_ALLSITE_GPU_CHUNK_READS", c.chunk, 1); else unsetenv("SINGLET_ALLSITE_GPU_CHUNK_READS");
        if (c.block) setenv("SINGLET_ALLSITE_GPU_BLOCK_HITS", c.block, 1); else unsetenv("SINGLET_ALLSITE_GPU_BLOCK_HITS");
        unsetenv("SINGLET_ALLSITE_GPU_SPLIT_BYTES"); unsetenv("SINGLET_ALLSITE_GPU_SPLIT_FRAC");   // contigs in windows: a byte limit, or "f<share of the largest>"
        if (c.split) setenv(c.split[0] == 'f' ? "SINGLET_ALLSITE_GPU_SPLIT_FRAC" : "SINGLET_ALLSITE_GPU_SPLIT_BYTES", c.split[0] == 'f' ? c.split + 1 : c.split, 1);
        if (c.block) setenv("SINGLET_ALLSITE_GPU_SLICE_BYTES", "1027", 1); else unsetenv("SINGLET_ALLSITE_GPU_SLICE_BYTES");   // 4 threads x 256 + 3: slices split records and leave a remainder
        A::Params P;
        P.skip_tid = 1; P.use_qual = c.use_qual; P.qwords = qwords; P.skip_n = c.skip_n; P.directional = c.directional; P.umi_len = 12; P.threads = 4; P.min_alt_pct = pct;
        const A::Result h = A::run(R, G, 4, nBc, P);
        A::Params Pd = P;
        bool used = false; std::string why;
        Pd.device = [&](const A::dev::Input& i, A::dev::Output& o) { used = singlet_engine_gpu::allsiteDeviceRun(i, o); why = o.why; return used; };
        const A::Result d = A::run(R, G, 4, nBc, Pd);
        if (!used) {
            if (why == "no CUDA device") { std::cout << "SKIP: " << why << "\n"; return 77; }
            std::cout << "FAIL device route declined: " << why << "\n"; ++n_fail; continue;
        }
        ++ran;
        const std::string tag = "[floor " + std::to_string(pct) + " qual " + std::to_string(c.use_qual) + " skipN " + std::to_string(c.skip_n) + " dir " + std::to_string(c.directional) + "] ";
        bool sitesEq = h.sites.size() == d.sites.size();
        for (size_t i = 0; sitesEq && i < h.sites.size(); ++i) {
            const auto& a = h.sites[i]; const auto& b = d.sites[i];
            sitesEq = a.tid == b.tid && a.pos == b.pos && a.ref == b.ref && a.alt == b.alt && a.alt_molecules == b.alt_molecules && a.alt_cells == b.alt_cells;
        }
        CHECK(sitesEq, tag + "sites (" + std::to_string(h.sites.size()) + " host, " + std::to_string(d.sites.size()) + " device)");
        CHECK(h.mismatch_obs == d.mismatch_obs, tag + "mismatch observations " + std::to_string(h.mismatch_obs) + " vs " + std::to_string(d.mismatch_obs));
        CHECK(h.molecules == d.molecules && h.ambiguous == d.ambiguous && h.other == d.other,
              tag + "molecules " + std::to_string(h.molecules) + "/" + std::to_string(h.ambiguous) + "/" + std::to_string(h.other) + " vs "
                  + std::to_string(d.molecules) + "/" + std::to_string(d.ambiguous) + "/" + std::to_string(d.other));
        CHECK(h.dp_records == d.dp_records && h.ad_records == d.ad_records, tag + "DP/AD records");
        CHECK(h.reads_used == d.reads_used, tag + "reads used");
        CHECK(h.sites_below_floor == d.sites_below_floor, tag + "sites below the floor " + std::to_string(h.sites_below_floor) + " vs " + std::to_string(d.sites_below_floor));
        CHECK(h.ad.indptr == d.ad.indptr && h.ad.indices == d.ad.indices && h.ad.data == d.ad.data, tag + "AD matrix");
        CHECK(h.dp.indptr == d.dp.indptr && h.dp.indices == d.dp.indices && h.dp.data == d.dp.data, tag + "DP matrix");
        CHECK(h.site_ad == d.site_ad && h.site_dp == d.site_dp, tag + "per-site sums");
        std::cout << tag << h.sites.size() << " sites, " << h.molecules << " molecules (ambiguous " << h.ambiguous << ", other " << h.other
                  << "), DP entries " << h.dp.data.size() << ", obs " << h.mismatch_obs << "\n";
    }
    fs::remove_all(dir);
    if (ran > 0) {   // a failure after counts reached the sink: the device route declines and the host pass gives the same result
        unsetenv("SINGLET_ALLSITE_GPU_CHUNK_READS"); unsetenv("SINGLET_ALLSITE_GPU_SLICE_BYTES"); unsetenv("SINGLET_ALLSITE_GPU_SPLIT_BYTES"); unsetenv("SINGLET_ALLSITE_GPU_SPLIT_FRAC");
        setenv("SINGLET_ALLSITE_GPU_BLOCK_HITS", "150000", 1); setenv("SINGLET_ALLSITE_GPU_FAIL_BLOCK", "2", 1);
        A::Params P;
        P.skip_tid = 1; P.use_qual = 1; P.qwords = qwords; P.umi_len = 12; P.threads = 4; P.min_alt_pct = 1;
        const A::Result h = A::run(R, G, 4, nBc, P);
        A::Params Pd = P;
        bool used = true; std::string why;
        Pd.device = [&](const A::dev::Input& i, A::dev::Output& o) { used = singlet_engine_gpu::allsiteDeviceRun(i, o); why = o.why; return used; };
        const A::Result d = A::run(R, G, 4, nBc, Pd);
        CHECK(!used && why.find("injected failure") != std::string::npos, "[injected failure] the device route declined (" + why + ")");
        bool sitesEq = h.sites.size() == d.sites.size();
        for (size_t i = 0; sitesEq && i < h.sites.size(); ++i) sitesEq = h.sites[i].pos == d.sites[i].pos && h.sites[i].alt == d.sites[i].alt && h.sites[i].alt_molecules == d.sites[i].alt_molecules;
        CHECK(sitesEq && h.ad.data == d.ad.data && h.ad.indices == d.ad.indices && h.dp.data == d.dp.data && h.site_ad == d.site_ad
              && h.sites_below_floor == d.sites_below_floor && h.molecules == d.molecules, "[injected failure] the host pass after it");
        std::cout << "[injected failure] " << why << "; host pass " << d.sites.size() << " sites\n";
        unsetenv("SINGLET_ALLSITE_GPU_FAIL_BLOCK"); unsetenv("SINGLET_ALLSITE_GPU_BLOCK_HITS");
    }
    if (ran == 0 && n_fail == 0) { std::cout << "SKIP: the device route did not run\n"; return 77; }
    std::cout << (n_fail ? "FAILED " : "OK ") << n_fail << "\n";
    return n_fail ? 1 : 0;
#endif
}
