// All-site pass: the host side of the device route's gather (allsite_device_glue.h, Input::gather and Input::resplit) with the look-ahead
// prefetch walk (default) against the plain walk (SINGLET_ALLSITE_GATHER_PF=0): the arrays of every chunk must be the same, field by field
// and byte by byte. A hook stands in for the device: it gathers the whole chunk, each entry alone and the chunks of a resplit (windows
// with their own read lists), hashes what it got, and declines, so the host pass runs as usual. CPU only; no CUDA device needed.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static uint64_t mix(uint64_t h, uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); return h * 0xff51afd7ed558ccdull; }

// hashes of everything the hook gathers, in call order
static std::vector<uint64_t> run_hook(const Recs& R, const A::Genome& G, uint32_t nBc, int qwords, bool use_qual, const char* pf) {
    if (pf) setenv("SINGLET_ALLSITE_GATHER_PF", pf, 1); else unsetenv("SINGLET_ALLSITE_GATHER_PF");
    std::vector<uint64_t> out;
    A::Params P;
    P.skip_tid = 1; P.use_qual = use_qual; P.qwords = qwords; P.umi_len = 12; P.threads = 4;
    P.device = [&](const A::dev::Input& in0, A::dev::Output&) {
        const A::dev::Input& in = in0;   // resplit replaces in.contigs in the caller's object, which this is
        auto gather_all = [&](uint32_t c0, uint32_t c1) {
            uint64_t nr = 0, no = 0, ns = 0, nq = 0;
            for (uint32_t c = c0; c < c1; ++c) { nr += in.contigs[c].reads; no += in.contigs[c].ops; ns += in.contigs[c].seqb; nq += in.contigs[c].lqw; }
            std::vector<A::dev::Read> rd(nr + 1); std::vector<uint32_t> ops(no + 1, 0xdeadbeef); std::vector<uint8_t> seq(ns + 1, 0xee); std::vector<uint64_t> lq(nq + 1, 7);
            std::memset(static_cast<void*>(rd.data()), 0, rd.size() * sizeof(A::dev::Read));
            in.gather(c0, c1, rd.data(), ops.data(), seq.data(), lq.data());
            uint64_t h = mix(0, nr);
            for (uint64_t i = 0; i < nr; ++i) {
                const auto& d = rd[i];
                h = mix(h, static_cast<uint64_t>(static_cast<uint32_t>(d.pos))); h = mix(h, static_cast<uint64_t>(static_cast<uint32_t>(d.bc))); h = mix(h, d.umi3);
                h = mix(h, d.lq); h = mix(h, d.ops); h = mix(h, d.seq); h = mix(h, d.rlen); h = mix(h, d.nops); h = mix(h, d.ci); h = mix(h, d.rev); h = mix(h, d.nlq);
            }
            for (uint64_t i = 0; i <= no; ++i) h = mix(h, ops[i]);
            for (uint64_t i = 0; i <= ns; ++i) h = mix(h, seq[i]);
            for (uint64_t i = 0; i <= nq; ++i) h = mix(h, lq[i]);
            out.push_back(h);
        };
        const uint32_t nE = static_cast<uint32_t>(in.contigs.size());
        gather_all(0, nE);
        for (uint32_t c = 0; c < nE; ++c) gather_all(c, c + 1);
        if (nE > 1) gather_all(1, nE);
        // windows: every contig over a small budget is split; chunks of the entries (a window's reads are those overlapping it)
        if (in.resplit && in.resplit(uint64_t{1} << 19, uint64_t{1} << 40)) {
            const uint32_t nW = static_cast<uint32_t>(in.contigs.size());
            out.push_back(nW);
            gather_all(0, nW);
            for (uint32_t c = 0; c < nW; ++c) gather_all(c, c + 1);
            for (uint32_t c = 0; c + 2 < nW; c += 2) gather_all(c, c + 3);
        } else out.push_back(~0ull);
        return false;   // the host pass runs
    };
    (void)A::run(R, G, 3, nBc, P);
    return out;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("allsite_gather_test_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    std::mt19937_64 g(20261114);
    auto U = [&](uint64_t n) { return n ? g() % n : 0; };
    const std::vector<std::string> names = {"chr1", "chrM", "chr2"};
    const std::vector<uint64_t> lens = {300000, 16000, 200000};
    {   std::ofstream fn(dir / "chrName.txt"), fs_(dir / "chrStart.txt"), fl(dir / "chrNameLength.txt");
        std::ofstream fg(dir / "Genome", std::ios::binary);
        uint64_t start = 0;
        for (int t = 0; t < 3; ++t) {
            fn << names[t] << "\n"; fl << names[t] << "\t" << lens[t] << "\n"; fs_ << start << "\n";
            const uint64_t pad = (lens[t] + 65535) / 65536 * 65536;
            for (uint64_t i = 0; i < pad; ++i) fg.put(static_cast<char>(i < lens[t] ? U(4) : 5));
            start += pad;
        }
        fs_ << start << "\n"; }
    A::Genome G; std::string err;
    if (!G.open(dir.string(), names, &err)) { std::cout << "FAIL genome " << err << "\n"; return 1; }

    // random records: signatures with 1-2 alignments (the primary flagged), random CIGAR lengths and sequences, shared by 1-3 reads; some
    // reads without a barcode, a few low-MAPQ ones, some on the skipped contig; 2 mask words per read
    const uint32_t nBc = 500; const int qwords = 2;
    Recs R;
    for (int s = 0; s < 30000; ++s) {
        const int t = static_cast<int>(U(10) == 0 ? 1 : (U(3) == 0 ? 2 : 0));
        const int32_t pos = static_cast<int32_t>(U(lens[t] - 400));
        const uint16_t rlen = static_cast<uint16_t>(30 + U(120));
        Sig sg{static_cast<uint32_t>(R.Al.size()), 1, 1, rlen, R.seq.size()};
        for (int i = 0; i < (rlen + 1) / 2; ++i) R.seq.push_back(static_cast<uint8_t>(U(256)));
        const uint16_t nops = static_cast<uint16_t>(1 + U(5));
        const uint32_t off = static_cast<uint32_t>(R.ops.size());
        for (int k = 0; k < nops; ++k) R.ops.push_back((static_cast<uint32_t>(1 + U(60)) << 4) | static_cast<uint32_t>(k % 2 ? 3 : 0));
        const uint8_t mapq = U(15) == 0 ? 3 : 255;
        if (U(8) == 0) { sg.n_aln = 2; R.Al.push_back({t, pos + 5, off, nops, 1, mapq}); }
        R.Al.push_back({t, pos, off, nops, static_cast<uint8_t>(2 | (U(2) ? 1 : 0)), mapq});
        R.S.push_back(sg);
        const int dup = static_cast<int>(U(3));
        for (int d = 0; d <= dup; ++d) {
            const int32_t bc = U(12) == 0 ? -1 : static_cast<int32_t>(U(nBc));
            R.R.push_back({static_cast<uint32_t>(R.S.size() - 1), bc, g() & 0xffffffffffull});
            R.lq.push_back(g()); R.lq.push_back(g());
        }
    }
    std::cout << "records: " << R.S.size() << " signatures, " << R.R.size() << " reads\n";

    for (int uq = 0; uq < 2; ++uq) {
        const std::vector<uint64_t> a = run_hook(R, G, nBc, qwords, uq != 0, "0");
        const std::vector<uint64_t> b = run_hook(R, G, nBc, qwords, uq != 0, nullptr);
        const std::vector<uint64_t> c = run_hook(R, G, nBc, qwords, uq != 0, "1");
        CHECK(a.size() > 8, "the hook gathered (" + std::to_string(a.size()) + " arrays)");
        CHECK(a == b, std::string("plain walk and default walk give the same arrays (use_qual ") + std::to_string(uq) + ")");
        CHECK(a == c, std::string("plain walk and SINGLET_ALLSITE_GATHER_PF=1 give the same arrays (use_qual ") + std::to_string(uq) + ")");
        std::cout << "use_qual " << uq << ": " << a.size() << " gathers compared\n";
    }
    unsetenv("SINGLET_ALLSITE_GATHER_PF");
    fs::remove_all(dir);
    if (n_fail) { std::cout << n_fail << " failure(s)\n"; return 1; }
    std::cout << "OK\n";
    return 0;
}
