// The all-site pass's two counting kernels (allsite.h: 0 the per-block hash map, 1 sorted records, allsite_records.h) give identical
// results on random alignment records: sites, statistics, per-(barcode, site) pairs and both matrices, at several thread counts and block
// sizes, with and without UMIs, directional or not, with quality masks or N-tied qualities. The records cover spliced, clipped, inserted
// and deleted alignments on both strands, secondary alignments, low MAPQ, chrM, barcodes outside the set, UMI collisions one base apart,
// a (barcode, site) with more than 512 UMIs (the indexed directional path) and a molecule with more than 255 reads (saturated counts).

#include <array>
#include <unordered_map>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "singlet/pileup/allsite.h"
#include "singlet/pileup/allsite_replay.h"
#include "singlet/pileup/allsite_store.h"

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
    uint64_t lowq_word(uint64_t i) const { return lq[i]; }
};

static const uint8_t kCode[4] = {1, 2, 4, 8};

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("allsite_kernel_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    std::mt19937_64 g(20261109);
    // genome: chr1 3000, chrM 300, chr2 2000 (0-3 = ACGT, a few N)
    const std::vector<std::string> tids = {"chr1", "chrM", "chr2"};
    const std::vector<int> len = {3000, 300, 2000};
    std::vector<std::vector<uint8_t>> ref(3);
    for (int t = 0; t < 3; ++t) { ref[t].resize(len[t]); for (auto& b : ref[t]) b = (g() % 200 == 0) ? 4 : static_cast<uint8_t>(g() % 4); }
    { std::ofstream f(dir / "chrName.txt"); for (auto& n : tids) f << n << "\n"; }
    { std::ofstream f(dir / "chrStart.txt"); f << "0\n4096\n4608\n8192\n"; }
    { std::ofstream f(dir / "chrNameLength.txt"); for (int t = 0; t < 3; ++t) f << tids[t] << "\t" << len[t] << "\n"; }
    { std::ofstream f(dir / "Genome", std::ios::binary); const size_t pad[3] = {4096, 512, 3584};
      for (int t = 0; t < 3; ++t) { for (uint8_t b : ref[t]) f.put(static_cast<char>(b)); for (size_t i = len[t]; i < pad[t]; ++i) f.put(5); } }
    A::Genome G; std::string err;
    CHECK(G.open(dir.string(), tids, &err), "genome: " << err);

    const uint32_t NBC = 40;
    // donor alleles at a few positions per contig, so real sites exist beside the error sites
    std::vector<std::vector<int>> var(3);
    for (int t = 0; t < 3; ++t) for (int i = 0; i < 25; ++i) var[t].push_back(static_cast<int>(g() % len[t]));
    Recs rec;
    std::vector<uint64_t> umipool; for (int i = 0; i < 60; ++i) umipool.push_back(g() & ((1ull << 36) - 1));
    auto mut1 = [&](uint64_t u) { const int p = static_cast<int>(g() % 12); return (u & ~(7ull << (3 * p))) | ((g() % 4) << (3 * p)); };
    auto add_sig = [&](int t, int pos, bool rev, uint8_t mapq, bool spliced, int bc_bias) -> uint32_t {
        // CIGAR (BAM codes: 0 M, 1 I, 2 D, 3 N, 4 S, 7 =, 8 X)
        std::vector<uint32_t> o;
        if (g() % 4 == 0) o.push_back(((1 + g() % 5) << 4) | 4);
        const int nblk = 1 + static_cast<int>(g() % 3);
        for (int b = 0; b < nblk; ++b) {
            if (b > 0) { const int k = static_cast<int>(g() % 4);
                if (k == 0 && spliced) o.push_back(((20 + g() % 300) << 4) | 3); else if (k == 1) o.push_back(((1 + g() % 3) << 4) | 2); else o.push_back(((1 + g() % 3) << 4) | 1); }
            const uint32_t op = (g() % 10 == 0) ? 7 : (g() % 15 == 0 ? 8 : 0);
            o.push_back(((10 + g() % 30) << 4) | op);
        }
        if (g() % 4 == 0) o.push_back(((1 + g() % 5) << 4) | 4);
        std::vector<uint8_t> q;   // aligned orientation, nt16
        int rp = pos;
        for (uint32_t c : o) { const int op = c & 15, l = static_cast<int>(c >> 4);
            for (int j = 0; j < l; ++j) {
                if (op == 0 || op == 7 || op == 8) {
                    const int p = rp + j; uint8_t b = (p >= 0 && p < len[t]) ? ref[t][p] : 0; if (b > 3) b = static_cast<uint8_t>(g() % 4);
                    for (int v : var[t]) if (v == p && (bc_bias % 2 == 0)) b = static_cast<uint8_t>((b + 1) % 4);
                    const uint64_t e = g() % 100;
                    q.push_back(e < 3 ? kCode[g() % 4] : e < 4 ? 15 : kCode[b]);
                } else if (op == 1 || op == 4) q.push_back(kCode[g() % 4]);
            }
            if (op == 0 || op == 2 || op == 3 || op == 7 || op == 8) rp += l; }
        const int rl = static_cast<int>(q.size());
        Sig s{static_cast<uint32_t>(rec.Al.size()), 0, 1, static_cast<uint16_t>(rl), rec.seq.size()};
        std::vector<uint8_t> orig(rl);
        for (int i = 0; i < rl; ++i) orig[i] = rev ? A::detail::comp16(q[rl - 1 - i]) : q[i];
        for (int i = 0; i < rl; i += 2) rec.seq.push_back(static_cast<uint8_t>(orig[i] | ((i + 1 < rl ? orig[i + 1] : 0) << 4)));
        if (g() % 6 == 0) {   // a secondary alignment first
            rec.Al.push_back({t, pos + 7, static_cast<uint32_t>(rec.ops.size()), static_cast<uint16_t>(o.size()), static_cast<uint8_t>(rev ? 1 : 0), 3}); s.n_aln++; s.nh = 2; }
        rec.Al.push_back({t, pos, static_cast<uint32_t>(rec.ops.size()), static_cast<uint16_t>(o.size()), static_cast<uint8_t>(2 | (rev ? 1 : 0)), mapq}); s.n_aln++;
        for (uint32_t c : o) rec.ops.push_back(c);
        rec.S.push_back(s); return static_cast<uint32_t>(rec.S.size() - 1);
    };
    auto add_read = [&](uint32_t sig, int32_t bc, uint64_t umi) {
        rec.R.push_back({sig, bc, umi});
        uint64_t w0 = 0, w1 = 0; if (g() % 3 == 0) { w0 = g() & g() & g(); w1 = g() & g() & g(); }
        rec.lq.push_back(w0); rec.lq.push_back(w1);
    };
    for (int i = 0; i < 6000; ++i) {
        const int t = static_cast<int>(g() % 3); const int pos = static_cast<int>(g() % (len[t] - 60));
        const uint8_t mapq = g() % 10 == 0 ? 3 : 255;
        const int bc0 = static_cast<int>(g() % (NBC + 4)) - 2;
        const uint32_t s = add_sig(t, pos, g() % 2, mapq, true, bc0);
        const int nr = 1 + static_cast<int>(g() % 4);
        for (int k = 0; k < nr; ++k) {
            const int32_t bc = (k == 0 || g() % 2) ? bc0 : static_cast<int32_t>(g() % NBC);
            uint64_t u = umipool[g() % umipool.size()]; if (g() % 4 == 0) u = mut1(u);
            add_read(s, bc, u);
        }
    }
    {   // a deep (barcode, site): 700 UMIs of one cell over one signature, and a molecule of 300 reads
        const uint32_t s = add_sig(0, 1200, false, 255, false, 1);
        for (int k = 0; k < 700; ++k) add_read(s, 5, g() & ((1ull << 36) - 1));
        for (int k = 0; k < 300; ++k) add_read(s, 6, umipool[0]);
        const uint32_t s2 = add_sig(0, 1205, true, 255, false, 2);
        for (int k = 0; k < 300; ++k) add_read(s2, 6, umipool[0]);
    }

    struct Cfg { int umi_dedup, umi_len, directional, use_qual, skip_n; };
    std::vector<Cfg> cfgs = {{1, 12, 1, 1, 0}, {1, 12, 0, 0, 1}, {1, 10, 1, 0, 0}, {0, 12, 1, 1, 0}, {1, 0, 1, 0, 1}, {1, 12, 1, 1, 0}};
    uint32_t nbc = NBC;
    for (size_t ci = 0; ci < cfgs.size(); ++ci) {
        const Cfg& c = cfgs[ci];
        if (ci + 1 == cfgs.size()) {   // last: barcode columns spread over 3 bytes (the radix grouping by barcode takes several passes)
            for (auto& x : rec.R) if (x.bc >= 0) x.bc = x.bc * 100003 + 7;
            nbc = NBC * 100003 + 16;
        }
        std::string ref_d, ref_p; uint64_t ref_sites = 0;
        for (int kernel : {0, 1}) for (unsigned th : {1u, 2u, 3u, 8u}) for (uint64_t bh : {uint64_t(300), uint64_t(8000000)}) {
            if (kernel == 0 && (th != 1 || bh != 300)) continue;
            A::Params P; P.min_mapq = 20; P.skip_tid = 1; P.use_qual = c.use_qual; P.qwords = 2; P.skip_n = c.skip_n;
            P.umi_dedup = c.umi_dedup; P.umi_len = c.umi_len; P.directional = c.directional; P.threads = th; P.block_hits = bh; P.kernel = kernel; P.spill_dir = th == 2 ? "-" : "";
            P.defer_build = true;
            A::Result r = A::run(rec, G, 3, nbc, P);
            r.prepare(nullptr);
            // the pairs as a set (kernel 0 gives them barcode by barcode within a contig, kernel 1 site by site); rows ascend within a barcode
            std::vector<std::array<uint32_t, 4>> pv; std::unordered_map<uint32_t, uint32_t> last; bool asc = true;
            r.for_each_pair([&](uint32_t bc, uint32_t row, uint32_t ad, uint32_t dp) {
                pv.push_back({bc, row, ad, dp}); auto it = last.find(bc); if (it != last.end() && it->second >= row) asc = false; last[bc] = row; });
            std::sort(pv.begin(), pv.end());
            A::replay::Fnv fp; for (const auto& q : pv) { fp.s(q[0]); fp.s(q[1]); fp.s(q[2]); fp.s(q[3]); }
            CHECK(asc, "rows ascend within each barcode");
            {   // the donor calls and the delta store read the pairs (kernel 1: unit by unit, in parallel)
                std::vector<uint32_t> cells; for (const auto& x : rec.R) if (x.bc >= 0 && static_cast<uint32_t>(x.bc) < nbc) cells.push_back(static_cast<uint32_t>(x.bc));
                std::sort(cells.begin(), cells.end()); cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
                std::vector<uint32_t> half; for (size_t j = 0; j < cells.size(); j += 2) half.push_back(cells[j]);
                std::vector<int32_t> donor_of(nbc, -1); for (uint32_t b : cells) donor_of[b] = static_cast<int32_t>((b / 7) % 3) - 1;
                const A::DonorCalls dc = A::donor_calls(r, donor_of, 2);
                A::DeltaStore st; A::build_store(r, nbc, half, donor_of, dc, st);
                for (size_t i = 0; i < dc.gt.size(); ++i) { fp.s(static_cast<uint32_t>(dc.gt[i] + 1)); fp.s(dc.sad[i]); fp.s(dc.sdp[i]); }
                for (uint64_t o : st.off) fp.s(static_cast<uint32_t>(o));
                for (size_t i = 0; i < st.row.size(); ++i) { fp.s(st.row[i]); fp.s(st.ad[i]); fp.s(st.dp[i]); }
                fp.s(static_cast<uint32_t>(st.full_entries)); fp.s(static_cast<uint32_t>(dc.sites_variant));
                CHECK(st.row.size() > 0, "the store keeps entries");
            }
            uint64_t e = 0; const bool ok = r.emit(true, nullptr, e) && r.emit(false, nullptr, e); r.release();
            const std::string d = A::replay::digest(r), pd = std::to_string(fp.h);
            const std::string tag = "umi_dedup " + std::to_string(c.umi_dedup) + " umi_len " + std::to_string(c.umi_len) + " directional " + std::to_string(c.directional)
                                  + " qual " + std::to_string(c.use_qual) + " kernel " + std::to_string(kernel) + " threads " + std::to_string(th) + " block " + std::to_string(bh);
            CHECK(ok, tag << ": matrices built");
            if (kernel == 0) { ref_d = d; ref_p = pd; ref_sites = r.sites.size();
                std::cout << "reference " << tag << ": " << r.sites.size() << " sites, " << r.molecules << " molecules, " << r.ambiguous << " ambiguous, " << r.other << " other, "
                          << r.dp_records << " DP records\n";
                CHECK(r.sites.size() > 20 && r.molecules > 1000, tag << ": enough sites and molecules to test");
            } else {
                CHECK(d == ref_d, tag << ": digest " << d << " vs kernel 0 " << ref_d);
                CHECK(pd == ref_p, tag << ": per-(barcode, site) pairs differ from kernel 0");
                CHECK(r.sites.size() == ref_sites, tag << ": site count");
            }
        }
    }
    // kernel 1 runs only when the key fits 64 bits (else run() takes kernel 0)
    {   // collapse_site_unsorted against collapse_site on one random site (hash table, radix by barcode) with UMIs one base apart
        const A::rec::Layout y = A::rec::layout(true, 12, 1u << 22);
        std::vector<uint64_t> k;
        for (int i = 0; i < 5000; ++i) { const uint32_t bc = static_cast<uint32_t>(g() % 300) * 13001; uint64_t u = umipool[g() % 8]; if (g() % 3 == 0) u = mut1(u);
            k.push_back(A::rec::key(y, bc, A::rec::umi_field(u, 12), static_cast<uint8_t>(g() % 3))); }
        for (int dir = 0; dir < 2; ++dir) {
            std::vector<uint64_t> a = k, b = k; std::sort(a.begin(), a.end());
            A::rec::Scratch S; A::rec::Counts ca, cb; std::vector<uint64_t> ea, eb;
            A::rec::collapse_site(a.data(), a.size(), y, dir, S, ca, [&](uint32_t bc, uint32_t dp, uint32_t ad) { ea.push_back((uint64_t(bc) << 40) | (uint64_t(dp) << 20) | ad); });
            A::rec::collapse_site_unsorted(b.data(), b.size(), y, dir, S, cb, [&](uint32_t bc, uint32_t dp, uint32_t ad) { eb.push_back((uint64_t(bc) << 40) | (uint64_t(dp) << 20) | ad); });
            std::sort(ea.begin(), ea.end()); std::sort(eb.begin(), eb.end());
            CHECK(ea == eb && ca.molecules == cb.molecules && ca.ambiguous == cb.ambiguous && ca.other == cb.other && !ea.empty(), "unsorted site collapse, directional " << dir);
        }
    }
    CHECK(A::rec::layout(true, 12, NBC).ok && A::rec::layout(false, 12, NBC).ok && !A::rec::layout(true, 21, NBC).ok, "key layout");
    fs::remove_all(dir);
    std::cout << (n_fail ? "FAILED " : "OK ") << n_fail << "\n";
    return n_fail ? 1 : 0;
}
