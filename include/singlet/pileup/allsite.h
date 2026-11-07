// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: allsite.h
// Per-cell allele counts at every covered site with alternate-allele evidence (D-101), beside the fixed panel (snp_ad / snp_dp).
//
// Input: the pass's alignment records (engine_gpu.h PileupInput: signatures, alignments, CIGAR ops, read sequences, reads with barcode
// column and UMI, per-read low-quality masks), the same records the GPU pileup reads, so a CPU build and a GPU build compute this member
// with one host implementation. Reference bases come from STAR's Genome file (genome_dir/Genome, one byte per base, 0-3 = ACGT).
//
// Rules (all order-free: every count is a function of the set of reads):
//  * reads: barcode in the set, the signature's primary alignment with MAPQ >= min_mapq, nuclear contigs (chrM has its own member);
//    bases inside M/=/X blocks, skipped when the read's low-quality mask marks them (or, when qualities are tied to N, when the base is N),
//    the same base rule as the panel pileup (pileup.cu k_lists, PileupEngine::pileup_aligned_bases).
//  * discovery: an alternate base at (contig, pos) is a mismatch against the reference base (both ACGT). A site is kept when its
//    alternate base is carried by >= min_alt_molecules distinct (barcode, UMI) molecules in >= min_alt_cells distinct barcodes. When two
//    alternate bases pass at one position, the one with more molecules is kept (ties: lower base code, A < C < G < T).
//  * counting: the panel's R1 consensus (SnpConsensus) per (barcode, site, UMI) with the R9 directional collapse per (barcode, site)
//    (no gene roots: a site's molecules are collapsed among themselves). DP = REF or ALT majority molecules, AD = ALT ones.
// Output rows are sites in (contig index, position) order, named "chrom:pos1:REF>ALT" (1-based position, as the panel's names).
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>
#include <cstdio>

#include "sparse_accumulator.h"
#include "umi_dedup.h"

namespace singlet { namespace allsite {

// STAR's Genome file, mapped read-only; base(t, pos) for the engine's contig index t (BAM header order, which is STAR's chrName order)
class Genome {
public:
    ~Genome() { if (map_ && map_ != MAP_FAILED) munmap(map_, size_); if (fd_ >= 0) close(fd_); }
    // tid_names: the BAM header's contig names (PileupEngine tid_to_chrom_); false if genome_dir lacks a file or a contig
    bool open(const std::string& genome_dir, const std::vector<std::string>& tid_names, std::string* err = nullptr) {
        std::vector<std::string> names; std::vector<uint64_t> starts, lens;
        { std::ifstream f(genome_dir + "/chrName.txt"); std::string l;
          while (std::getline(f, l)) { if (!l.empty() && l.back() == '\r') l.pop_back(); if (!l.empty()) names.push_back(l); } }
        { std::ifstream f(genome_dir + "/chrStart.txt"); uint64_t v; while (f >> v) starts.push_back(v); }
        { std::ifstream f(genome_dir + "/chrNameLength.txt"); std::string n; uint64_t v; while (f >> n >> v) lens.push_back(v); }
        if (names.empty() || starts.size() < names.size() || lens.size() != names.size()) { if (err) *err = "chrName/chrStart/chrNameLength"; return false; }
        std::unordered_map<std::string, size_t> at; for (size_t i = 0; i < names.size(); ++i) at[names[i]] = i;
        off_.assign(tid_names.size(), UINT64_MAX); len_.assign(tid_names.size(), 0);
        for (size_t t = 0; t < tid_names.size(); ++t) {
            auto it = at.find(tid_names[t]);
            if (it == at.end()) continue;   // a contig without reference bases has no sites
            off_[t] = starts[it->second]; len_[t] = lens[it->second];
        }
        fd_ = ::open((genome_dir + "/Genome").c_str(), O_RDONLY);
        if (fd_ < 0) { if (err) *err = genome_dir + "/Genome"; return false; }
        struct stat st{}; if (fstat(fd_, &st) != 0 || st.st_size <= 0) { if (err) *err = "Genome size"; return false; }
        size_ = static_cast<size_t>(st.st_size);
        map_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (map_ == MAP_FAILED) { map_ = nullptr; if (err) *err = "mmap Genome"; return false; }
        return true;
    }
    // 0-3 = A C G T, 4 = N or outside the contig
    uint8_t base(int32_t t, int32_t pos) const {
        if (t < 0 || static_cast<size_t>(t) >= off_.size() || off_[t] == UINT64_MAX || pos < 0 || static_cast<uint64_t>(pos) >= len_[t]) return 4;
        const uint64_t o = off_[t] + static_cast<uint64_t>(pos);
        if (o >= size_) return 4;
        const uint8_t v = static_cast<const uint8_t*>(map_)[o];
        return v < 4 ? v : 4;
    }
private:
    int fd_ = -1; void* map_ = nullptr; size_t size_ = 0;
    std::vector<uint64_t> off_, len_;
};

struct Params {
    int min_mapq = 20;
    int min_alt_molecules = 2;
    int min_alt_cells = 2;
    int32_t skip_tid = -1;        // chrM
    int use_qual = 0, qwords = 2, skip_n = 0;
    int umi_dedup = 1, directional = 1, umi_len = 12;
    unsigned threads = 8;
};

struct Site { int32_t tid, pos; char ref, alt; uint32_t alt_molecules, alt_cells; };

struct Result {
    std::vector<Site> sites;                                         // row order
    SparseAccumulator<uint16_t>::CSCMatrix ad, dp;                   // rows = sites, columns = barcodes
    uint64_t reads_used = 0, mismatch_obs = 0, molecules = 0, ambiguous = 0, other = 0;
};

namespace detail {
constexpr char kNt16[] = "=ACMGRSVTWYHKDBN";
inline uint8_t comp16(uint8_t c) { static const uint8_t t[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}; return t[c]; }
inline int base_idx(uint8_t c) { return c == 1 ? 0 : c == 2 ? 1 : c == 4 ? 2 : c == 8 ? 3 : -1; }
// the read base aligned at query position qp (alignment orientation), nt16 code; seq is the record in original orientation
inline uint8_t aligned_base(const uint8_t* seq, int rlen, int qp, bool rev) {
    const int i = rev ? rlen - 1 - qp : qp;
    const uint8_t c = (seq[i >> 1] >> ((i & 1) * 4)) & 15;
    return rev ? comp16(c) : c;
}
template <class Rec> inline bool low_q(const Rec& R, int qwords, uint64_t r, uint32_t idx) {
    return (R.lowq_word(static_cast<uint64_t>(qwords) * r + (idx >> 6)) >> (idx & 63)) & 1ull;
}
struct Tri { uint32_t cell, row; uint32_t val; };
struct TriAcc {   // SnpConsensus::finalize's accumulator: increment(feature, barcode, delta)
    std::vector<Tri> v;
    void increment(uint32_t f, uint32_t bc, uint32_t d) { v.push_back({bc, f, d}); }
};
struct Stats { uint64_t snp_molecules = 0, snp_molecules_ambiguous = 0, snp_molecules_other = 0; };
}  // namespace detail

// Records: n_sigs(), n_reads(), sig(i), aln(i), read(i) (SigRec / AlnRec / ReadRec of pileup_rec.h, or structs with the same fields),
// ops_at(off) and seq_at(off) (pointers to a record's CIGAR ops and packed sequence) and lowq_word(i) (64-bit mask word i). The engine
// adapts PileupInput (PileupCat over its split arrays); the unit test feeds hand-made records.
template <class Rec>
inline Result run(const Rec& R, const Genome& G, uint32_t n_tid, uint32_t n_barcodes, const Params& P) {
    using namespace detail;
    Result res;
    const uint64_t nS = R.n_sigs(), nR = R.n_reads();
    // 1. per signature: its primary alignment's contig when the alignment qualifies, else -1
    std::vector<int32_t> sig_tid(nS, -1);
    std::vector<uint8_t> sig_p(nS, 0);
    for (uint64_t s = 0; s < nS; ++s) {
        const auto& sg = R.sig(s);
        for (int k = 0; k < sg.n_aln; ++k) {
            const auto& a = R.aln(sg.aln_off + k);
            if (!(a.flags & 2)) continue;
            if (a.mapq >= P.min_mapq && a.tid >= 0 && static_cast<uint32_t>(a.tid) < n_tid && a.tid != P.skip_tid) { sig_tid[s] = a.tid; sig_p[s] = static_cast<uint8_t>(k); }
            break;
        }
    }
    // 2. reads per contig, ordered by signature then read index (deterministic)
    std::vector<uint64_t> cnt(n_tid + 1, 0);
    auto use = [&](const auto& x) { return x.bc >= 0 && static_cast<uint32_t>(x.bc) < n_barcodes && sig_tid[x.sig] >= 0; };
    for (uint64_t r = 0; r < nR; ++r) { const auto& x = R.read(r); if (use(x)) cnt[sig_tid[x.sig] + 1]++; }
    for (uint32_t t = 0; t < n_tid; ++t) cnt[t + 1] += cnt[t];
    std::vector<uint64_t> by_tid(cnt[n_tid]);
    { std::vector<uint64_t> w(cnt.begin(), cnt.end() - 1);
      for (uint64_t r = 0; r < nR; ++r) { const auto& x = R.read(r); if (use(x)) by_tid[w[sig_tid[x.sig]]++] = r; } }
    res.reads_used = by_tid.size();
    std::vector<uint32_t> order(n_tid); for (uint32_t t = 0; t < n_tid; ++t) order[t] = t;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { const uint64_t na = cnt[a + 1] - cnt[a], nb = cnt[b + 1] - cnt[b]; return na != nb ? na > nb : a < b; });

    struct PerTid { std::vector<Site> sites; std::vector<Tri> ad, dp; uint64_t obs = 0; Stats st; };
    std::vector<PerTid> out(n_tid);
    std::atomic<uint32_t> next{0};
    auto pass_q = [&](uint64_t r, uint32_t ri, uint8_t code) { return !(P.skip_n && code == 15) && !(P.use_qual && low_q(R, P.qwords, r, ri)); };   // as k_lists + q_pass
    auto walk = [&](uint64_t r, auto&& on_base) {   // on_base(pos, nt16 code, record index) for every aligned base of read r (quality: pass_q)
        const auto& x = R.read(r); const auto& sg = R.sig(x.sig); const auto& a = R.aln(sg.aln_off + sig_p[x.sig]);
        const bool rev = (a.flags & 1) != 0; const uint32_t* o = R.ops_at(a.ops_off); const uint8_t* seq = R.seq_at(sg.seq_off);
        int32_t rpos = a.pos; int qpos = 0;
        for (int i = 0; i < a.n_ops; ++i) {
            const uint32_t c = o[i]; const int op = c & 15; const int32_t len = static_cast<int32_t>(c >> 4);
            if (op == 0 || op == 7 || op == 8) {
                for (int32_t j = 0; j < len; ++j) {
                    const int qp = qpos + j; const uint8_t code = aligned_base(seq, sg.rlen, qp, rev);
                    const uint32_t ri = static_cast<uint32_t>(rev ? sg.rlen - 1 - qp : qp);
                    on_base(rpos + j, code, ri);
                }
                rpos += len; qpos += len;
            } else if (op == 1 || op == 4) qpos += len;
            else if (op == 2 || op == 3) rpos += len;
        }
    };
    auto umi_chars = [&](uint64_t u3, char* buf) { for (int i = 0; i < P.umi_len && i < 21; ++i) buf[i] = "ACGTNNNN"[(u3 >> (3 * i)) & 7]; };

    auto work = [&]() {
        for (;;) {
            const uint32_t oi = next.fetch_add(1); if (oi >= n_tid) return;
            const uint32_t t = order[oi]; const uint64_t b0 = cnt[t], b1 = cnt[t + 1]; if (b0 == b1) continue;
            PerTid& pt = out[t];
            // 3a. discovery: (pos, alt) observations with their molecule
            struct Obs { uint64_t key; int32_t bc; uint64_t umi; };
            std::vector<Obs> obs;
            for (uint64_t k = b0; k < b1; ++k) {
                const uint64_t r = by_tid[k]; const auto& x = R.read(r);
                walk(r, [&](int32_t pos, uint8_t code, uint32_t ri) {
                    const int bi = base_idx(code); if (bi < 0) return;
                    const uint8_t ref = G.base(static_cast<int32_t>(t), pos); if (ref > 3 || bi == ref || !pass_q(r, ri, code)) return;
                    obs.push_back({(static_cast<uint64_t>(static_cast<uint32_t>(pos)) << 2) | static_cast<uint64_t>(bi), x.bc, P.umi_dedup ? x.umi : r});
                });
            }
            pt.obs = obs.size();
            std::sort(obs.begin(), obs.end(), [](const Obs& a, const Obs& b) {
                if (a.key != b.key) return a.key < b.key; if (a.bc != b.bc) return a.bc < b.bc; return a.umi < b.umi; });
            for (size_t s = 0; s < obs.size();) {   // one position at a time; the best passing alternate base
                const uint64_t pos = obs[s].key >> 2; size_t e = s;
                uint32_t best_m = 0, best_c = 0; int best_b = -1;
                while (e < obs.size() && (obs[e].key >> 2) == pos) {
                    size_t f = e; uint32_t mol = 0, cells = 0;
                    while (f < obs.size() && obs[f].key == obs[e].key) {
                        if (f == e || obs[f].bc != obs[f - 1].bc) { ++cells; ++mol; }
                        else if (obs[f].umi != obs[f - 1].umi) ++mol;
                        ++f;
                    }
                    if (static_cast<int>(mol) >= P.min_alt_molecules && static_cast<int>(cells) >= P.min_alt_cells && mol > best_m) {
                        best_m = mol; best_c = cells; best_b = static_cast<int>(obs[e].key & 3);
                    }
                    e = f;
                }
                if (best_b >= 0) pt.sites.push_back({static_cast<int32_t>(t), static_cast<int32_t>(pos), "ACGT"[G.base(static_cast<int32_t>(t), static_cast<int32_t>(pos))],
                                                      "ACGT"[best_b], best_m, best_c});
                s = e;
            }
            std::vector<Obs>().swap(obs);
            if (pt.sites.empty()) continue;
            // 3b. counting: R1 consensus per (barcode, site, UMI), R9 per (barcode, site)
            std::vector<int32_t> spos(pt.sites.size()); for (size_t i = 0; i < spos.size(); ++i) spos[i] = pt.sites[i].pos;
            SnpConsensus cons; char ub[24];
            for (uint64_t k = b0; k < b1; ++k) {
                const uint64_t r = by_tid[k]; const auto& x = R.read(r);
                umi_chars(x.umi, ub);
                // the read's sites: its aligned positions increase, so one cursor over the sorted site positions
                const auto& sg = R.sig(x.sig);
                size_t cur = static_cast<size_t>(std::lower_bound(spos.begin(), spos.end(), R.aln(sg.aln_off + sig_p[x.sig]).pos) - spos.begin());
                if (cur == spos.size()) continue;
                walk(r, [&](int32_t pos, uint8_t code, uint32_t ri) {
                    while (cur < spos.size() && spos[cur] < pos) ++cur;
                    if (cur == spos.size() || spos[cur] != pos || !pass_q(r, ri, code)) return;
                    const size_t si = cur;
                    cons.add(static_cast<uint32_t>(x.bc), static_cast<uint32_t>(si), P.umi_dedup ? ub : nullptr, P.umi_len,
                             SnpConsensus::base_class(kNt16[code], pt.sites[si].ref, pt.sites[si].alt));
                });
            }
            TriAcc ad, dp;
            cons.finalize(ad, dp, pt.st, P.directional != 0);
            cons.clear();
            pt.ad.swap(ad.v); pt.dp.swap(dp.v);
        }
    };
    const unsigned nth = std::max(1u, std::min<unsigned>(P.threads, n_tid));
    std::vector<std::thread> th; for (unsigned i = 1; i < nth; ++i) th.emplace_back(work);
    work(); for (auto& x : th) x.join();

    // 4. rows in (contig, position) order; CSC by barcode
    std::vector<uint32_t> row0(n_tid + 1, 0);
    for (uint32_t t = 0; t < n_tid; ++t) row0[t + 1] = row0[t] + static_cast<uint32_t>(out[t].sites.size());
    res.sites.reserve(row0[n_tid]);
    for (uint32_t t = 0; t < n_tid; ++t) { for (const auto& s : out[t].sites) res.sites.push_back(s);
        res.mismatch_obs += out[t].obs; res.molecules += out[t].st.snp_molecules; res.ambiguous += out[t].st.snp_molecules_ambiguous; res.other += out[t].st.snp_molecules_other; }
    auto build = [&](bool is_ad, SparseAccumulator<uint16_t>::CSCMatrix& m) {
        std::vector<Tri> all;
        size_t n = 0; for (uint32_t t = 0; t < n_tid; ++t) n += (is_ad ? out[t].ad : out[t].dp).size();
        all.reserve(n);
        for (uint32_t t = 0; t < n_tid; ++t) { auto& v = is_ad ? out[t].ad : out[t].dp; for (const auto& x : v) all.push_back({x.cell, x.row + row0[t], x.val}); std::vector<Tri>().swap(v); }
        std::sort(all.begin(), all.end(), [](const Tri& a, const Tri& b) { return a.cell != b.cell ? a.cell < b.cell : a.row < b.row; });
        m.nrows = row0[n_tid]; m.ncols = n_barcodes; m.indptr.assign(n_barcodes + 1, 0); m.indices.clear(); m.data.clear();
        for (size_t i = 0; i < all.size();) {   // duplicates (one per molecule) summed, saturated at 65535
            size_t j = i; uint64_t v = 0; while (j < all.size() && all[j].cell == all[i].cell && all[j].row == all[i].row) v += all[j++].val;
            m.indices.push_back(static_cast<int32_t>(all[i].row)); m.data.push_back(static_cast<uint16_t>(std::min<uint64_t>(v, 65535)));
            if (all[i].cell < n_barcodes) m.indptr[all[i].cell + 1]++;
            i = j;
        }
        for (uint32_t c = 0; c < n_barcodes; ++c) m.indptr[c + 1] += m.indptr[c];
    };
    build(true, res.ad); build(false, res.dp);
    return res;
}

inline std::string site_name(const Site& s, const std::vector<std::string>& tid_names) {
    return tid_names[static_cast<size_t>(s.tid)] + ":" + std::to_string(s.pos + 1) + ":" + s.ref + ">" + s.alt;
}

// sites.tsv.gz: one line per row of the matrices (chrom, 1-based pos, ref, alt, ALT molecules and barcodes at discovery, summed DP and AD)
inline bool write_sites_tsv(const std::string& path, const Result& r, const std::vector<std::string>& tid_names) {
    std::vector<uint64_t> dp(r.sites.size(), 0), ad(r.sites.size(), 0);
    for (size_t k = 0; k < r.dp.data.size(); ++k) dp[static_cast<size_t>(r.dp.indices[k])] += r.dp.data[k];
    for (size_t k = 0; k < r.ad.data.size(); ++k) ad[static_cast<size_t>(r.ad.indices[k])] += r.ad.data[k];
    const std::string tmp = path + ".tmp";
    gzFile f = gzopen(tmp.c_str(), "wb6");
    if (!f) return false;
    std::string line = "chrom\tpos\tref\talt\talt_molecules\talt_barcodes\tdp\tad\n";
    bool ok = gzwrite(f, line.data(), static_cast<unsigned>(line.size())) > 0;
    for (size_t i = 0; ok && i < r.sites.size(); ++i) {
        const Site& s = r.sites[i];
        line = tid_names[static_cast<size_t>(s.tid)]; line += '\t'; line += std::to_string(s.pos + 1); line += '\t'; line += s.ref; line += '\t'; line += s.alt;
        line += '\t'; line += std::to_string(s.alt_molecules); line += '\t'; line += std::to_string(s.alt_cells);
        line += '\t'; line += std::to_string(dp[i]); line += '\t'; line += std::to_string(ad[i]); line += '\n';
        ok = gzwrite(f, line.data(), static_cast<unsigned>(line.size())) > 0;
    }
    if (gzclose(f) != Z_OK) ok = false;
    if (ok) ok = std::rename(tmp.c_str(), path.c_str()) == 0;
    if (!ok) std::remove(tmp.c_str());
    return ok;
}

}}  // namespace singlet::allsite
