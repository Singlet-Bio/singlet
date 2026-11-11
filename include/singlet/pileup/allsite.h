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
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
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
#include "allsite_records.h"
#include "allsite_device.h"

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
    // contig t's bases as stored (base() maps values over 3 to N), *len of them; nullptr when the contig has none (the device route)
    const uint8_t* contig(int32_t t, uint64_t* len) const {
        *len = 0;
        if (t < 0 || static_cast<size_t>(t) >= off_.size() || off_[t] == UINT64_MAX || off_[t] >= size_) return nullptr;
        *len = std::min<uint64_t>(len_[t], size_ - off_[t]);
        return static_cast<const uint8_t*>(map_) + off_[t];
    }
    // the raw bytes of [pos, pos + len) on contig t when all of them lie inside the contig (else null: use base())
    const uint8_t* span(int32_t t, int32_t pos, int32_t len) const {
        if (t < 0 || static_cast<size_t>(t) >= off_.size() || off_[t] == UINT64_MAX || pos < 0 || len < 0 || static_cast<uint64_t>(pos) + static_cast<uint64_t>(len) > len_[t]) return nullptr;
        const uint64_t o = off_[t] + static_cast<uint64_t>(pos);
        if (o + static_cast<uint64_t>(len) > size_) return nullptr;
        return static_cast<const uint8_t*>(map_) + o;
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
    uint64_t block_hits = 8000000;   // counting works on blocks of sites with about this many read-site hits, so each thread's
                                     // consensus table stays bounded (a whole contig of a 400M-read library needs hundreds of GB)
    bool defer_build = false;        // keep the counts as triplets; Result::finalize builds the matrices (for the called cells only)
    std::function<void()> progress;  // called (from worker threads) after each contig's discovery and each counting block; the
                                     // engine re-arms its stall watchdog here, since an LCL's pass runs well past an hour
    std::function<bool(const dev::Input&, dev::Output&)> device;   // the device route (allsite_device.h); false: the host pass runs
    int min_alt_pct = 1;             // the pooled ALT-fraction floor (rec::floor_keep): a counted site is kept only when 100 * ALT >= pct * depth
                                     // over every barcode; 0 keeps every discovered site
    std::string spill_dir;           // kernel 1 keeps its merged counts in a file here, unlinked at once, so memory does not grow with them:
                                     // "" SINGLET_ALLSITE_SPILL_DIR, else TMPDIR, else /tmp; "-" compact bytes in memory
    int kernel = -1;                 // counting kernel: 1 sorted records (allsite_records.h), 0 the per-block hash map; -1: sorted records
                                     // unless SINGLET_ALLSITE_KERNEL=old (A/B). Both give identical counts; keys that do not fit 64 bits use 0.
};

struct Site { int32_t tid, pos; char ref, alt; uint32_t alt_molecules, alt_cells; };

struct Result {
    std::vector<Site> sites;                                         // row order
    SparseAccumulator<uint16_t>::CSCMatrix ad, dp;                   // rows = sites, columns = barcodes
    uint64_t reads_used = 0, mismatch_obs = 0, molecules = 0, ambiguous = 0, other = 0;
    uint64_t dp_records = 0, ad_records = 0;                         // per-(barcode, site, molecule-batch) counts before they are summed (kept sites)
    uint64_t sites_below_floor = 0;                                  // discovered sites dropped by the ALT-fraction floor (Params::min_alt_pct)
    std::vector<uint64_t> site_dp, site_ad;                          // per site, summed over every barcode (filled by finalize)
    // With Params::defer_build the matrices stay empty until the caller builds them in two steps, once the called cells (and with them the
    // donor count) are known. prepare(keep_cols) drops the entries of other barcodes, merges the per-molecule records of each (barcode, site)
    // and fills kept_ad / kept_dp, the per-site sums over the kept barcodes. emit(is_ad, keep_rows, entries) then builds one matrix from the
    // merged entries, optionally only for the rows marked in keep_rows. The int32 column pointers bound the matrix: entries counts the
    // merged entries (what the matrix would hold, not the records they came from), and emit returns false, leaving the matrix empty,
    // when that is max_entries or more. emit does not consume the merged entries, so a smaller row set can follow a failed attempt;
    // release() frees them.
    std::vector<uint64_t> kept_dp, kept_ad;                          // per site, summed over the kept barcodes (filled by prepare)
    uint64_t max_entries = static_cast<uint64_t>(INT32_MAX);
    std::function<void(Result&, const std::vector<uint8_t>*)> prepare_fn;
    std::function<bool(Result&, bool, const std::vector<uint8_t>*, uint64_t&)> emit_fn;
    std::function<void()> release_fn;
    void prepare(const std::vector<uint8_t>* keep_cols) { if (prepare_fn) prepare_fn(*this, keep_cols); }
    bool emit(bool is_ad, const std::vector<uint8_t>* keep_rows, uint64_t& entries) { entries = 0; return emit_fn ? emit_fn(*this, is_ad, keep_rows, entries) : true; }
    void release() const { if (release_fn) release_fn(); }
    // After prepare: every merged (barcode, site) entry as f(barcode, row, ad, dp) with ad <= dp, contig by contig (rows ascend across
    // contigs) and by barcode then row within one. The delta store (allsite_store.h) reads the member through this.
    std::function<void(const std::function<void(uint32_t, uint32_t, uint32_t, uint32_t)>&)> pairs_fn;
    void for_each_pair(const std::function<void(uint32_t, uint32_t, uint32_t, uint32_t)>& f) const { if (pairs_fn) pairs_fn(f); }
    // Kernel 1 also gives the same entries unit by unit: unit_fn(i, v) fills v with unit i's entries. Units hold disjoint row ranges that
    // ascend with i, so readers may take units in parallel (threads workers) and still see rows ascend within a barcode in unit order.
    struct Pair { uint32_t cell, row, ad, dp; };
    size_t n_units = 0; unsigned threads = 1;
    std::function<void(size_t, std::vector<Pair>&)> unit_fn;
    // Both matrices over every site for the barcodes marked in keep (null: all); false when either does not fit.
    bool finalize(const std::vector<uint8_t>* keep) {
        if (!prepare_fn) return true;
        prepare(keep);
        uint64_t e = 0;
        const bool ok = emit(true, nullptr, e) && emit(false, nullptr, e);
        release();
        if (!ok) { ad = {}; dp = {}; }
        return ok;
    }
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

namespace detail {
// SINGLET_ALLSITE_TIMING=1: thread-seconds per stage of the pass (stderr only; counts and seconds)
struct Timers {
    std::atomic<uint64_t> walk{0}, sort{0}, scan{0}, member{0}, count{0}, fin{0}, wall{0}, hits{0}, blocks{0}, records{0}, distinct{0}, contigs{0};
    std::atomic<uint64_t> max_wall{0}, max_reads{0}, max_tid{0}, max_sites{0};
};
inline uint64_t now_ns() { return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()); }
}  // namespace detail

namespace detail {
struct PerTid { std::vector<Site> sites; std::vector<Tri> ad, dp; uint64_t obs = 0; Stats st; uint64_t ad_rec = 0, dp_rec = 0; };
// Rows, per-site sums and the deferred matrix builders from the per-contig counts (both kernels). ad / dp hold (barcode, site in contig,
// count) triplets in any order; prepare sums them per (barcode, site), so one triplet per molecule and one per (barcode, site) agree.
inline void finish(Result& res, std::shared_ptr<std::vector<PerTid>> out_sp, uint32_t n_tid, uint32_t n_barcodes, const Params& P) {
    auto& out = *out_sp;
    // the ALT-fraction floor (rec::floor_keep) on each contig's complete per-site sums: dropped sites leave the site list and their triplets
    // go, the rest are renumbered (molecule statistics stay those of every discovered site; the record counts are those of the kept sites)
    if (P.min_alt_pct > 0) {
        const unsigned nth = std::max(1u, std::min<unsigned>(P.threads, n_tid));
        std::atomic<uint32_t> next{0}; std::atomic<uint64_t> dropped{0};
        auto work = [&]() {
            for (uint32_t t; (t = next.fetch_add(1)) < n_tid;) {
                PerTid& pt = out[t]; const size_t ns = pt.sites.size(); if (!ns) continue;
                std::vector<uint64_t> sad(ns, 0), sdp(ns, 0);
                for (const Tri& x : pt.ad) sad[x.row] += x.val;
                for (const Tri& x : pt.dp) sdp[x.row] += x.val;
                std::vector<uint32_t> nrow(ns, UINT32_MAX); uint32_t k = 0;
                for (size_t i = 0; i < ns; ++i) if (rec::floor_keep(sad[i], sdp[i], P.min_alt_pct)) { nrow[i] = k; pt.sites[k++] = pt.sites[i]; }
                if (k == ns) continue;
                dropped += ns - k; pt.sites.resize(k);
                for (int w = 0; w < 2; ++w) {
                    auto& v = w == 0 ? pt.ad : pt.dp; size_t m = 0;
                    for (const Tri& x : v) if (nrow[x.row] != UINT32_MAX) { Tri y = x; y.row = nrow[x.row]; v[m++] = y; }
                    v.resize(m);
                }
                pt.ad_rec = pt.ad.size(); pt.dp_rec = pt.dp.size();
            }
        };
        std::vector<std::thread> th; for (unsigned i = 1; i < nth; ++i) th.emplace_back(work);
        work(); for (auto& x : th) x.join();
        res.sites_below_floor = dropped.load();
    }
    // 4. rows in (contig, position) order; CSC by barcode
    std::vector<uint32_t> row0(n_tid + 1, 0);
    for (uint32_t t = 0; t < n_tid; ++t) row0[t + 1] = row0[t] + static_cast<uint32_t>(out[t].sites.size());
    res.sites.reserve(row0[n_tid]);
    for (uint32_t t = 0; t < n_tid; ++t) { for (const auto& s : out[t].sites) res.sites.push_back(s);
        res.mismatch_obs += out[t].obs; res.molecules += out[t].st.snp_molecules; res.ambiguous += out[t].st.snp_molecules_ambiguous; res.other += out[t].st.snp_molecules_other; }
    for (uint32_t t = 0; t < n_tid; ++t) { res.ad_records += out[t].ad_rec; res.dp_records += out[t].dp_rec; }
    auto row0_sp = std::make_shared<std::vector<uint32_t>>(std::move(row0));
    const unsigned build_threads = std::max(1u, std::min<unsigned>(P.threads, n_tid));
    auto cmp = [](const Tri& a, const Tri& b) { return a.cell != b.cell ? a.cell < b.cell : a.row < b.row; };
    res.prepare_fn = [out_sp, row0_sp, n_tid, n_barcodes, build_threads, cmp](Result& r, const std::vector<uint8_t>* keep) {
        auto& out = *out_sp; const auto& row0 = *row0_sp;
        const size_t nrow = row0[n_tid];
        r.site_ad.assign(nrow, 0); r.site_dp.assign(nrow, 0); r.kept_ad.assign(nrow, 0); r.kept_dp.assign(nrow, 0);
        std::vector<uint32_t> order(n_tid); for (uint32_t t = 0; t < n_tid; ++t) order[t] = t;
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return out[a].ad.size() + out[a].dp.size() > out[b].ad.size() + out[b].dp.size(); });
        std::atomic<uint32_t> next{0};
        // Contigs own disjoint row ranges, so each worker writes its own part of the per-site sums.
        auto work = [&]() {
            for (;;) {
                const uint32_t oi = next.fetch_add(1); if (oi >= n_tid) return;
                const uint32_t t = order[oi];
                for (int w = 0; w < 2; ++w) {
                    auto& v = w == 0 ? out[t].ad : out[t].dp;
                    auto& site = w == 0 ? r.site_ad : r.site_dp; auto& kept = w == 0 ? r.kept_ad : r.kept_dp;
                    size_t m = 0;
                    for (size_t i = 0; i < v.size(); ++i) {
                        Tri x = v[i]; x.row += row0[t];
                        site[x.row] += x.val;
                        if (x.cell >= n_barcodes || (keep && !(x.cell < keep->size() && (*keep)[x.cell]))) continue;
                        kept[x.row] += x.val; v[m++] = x;
                    }
                    v.resize(m);
                    std::sort(v.begin(), v.end(), cmp);
                    size_t o = 0;
                    for (size_t i = 0; i < m;) {   // one record per molecule batch: summed per (barcode, site), saturated at 65535
                        size_t j = i; uint64_t sum = 0; while (j < m && v[j].cell == v[i].cell && v[j].row == v[i].row) sum += v[j++].val;
                        Tri y = v[i]; y.val = static_cast<uint32_t>(std::min<uint64_t>(sum, 65535)); v[o++] = y; i = j;
                    }
                    v.resize(o); v.shrink_to_fit();
                }
            }
        };
        std::vector<std::thread> th; for (unsigned i = 1; i < build_threads; ++i) th.emplace_back(work);
        work(); for (auto& x : th) x.join();
    };
    res.emit_fn = [out_sp, row0_sp, n_tid, n_barcodes](Result& r, bool is_ad, const std::vector<uint8_t>* keep_rows, uint64_t& entries) -> bool {
        auto& out = *out_sp; const auto& row0 = *row0_sp;
        auto& m = is_ad ? r.ad : r.dp;
        m = {}; m.nrows = row0[n_tid]; m.ncols = n_barcodes;
        const auto row_ok = [&](uint32_t row) { return !keep_rows || (row < keep_rows->size() && (*keep_rows)[row]); };
        std::vector<uint64_t> cnt(static_cast<size_t>(n_barcodes) + 1, 0);
        entries = 0;
        for (uint32_t t = 0; t < n_tid; ++t)
            for (const auto& x : (is_ad ? out[t].ad : out[t].dp)) if (row_ok(x.row)) { ++cnt[x.cell + 1]; ++entries; }
        if (entries >= r.max_entries) {
            std::cerr << "[allsite] " << (is_ad ? "AD" : "DP") << ": " << entries << " (barcode, site) entries do not fit the int32 column pointers\n";
            return false;
        }
        m.indptr.assign(static_cast<size_t>(n_barcodes) + 1, 0);
        for (uint32_t c = 0; c < n_barcodes; ++c) { cnt[c + 1] += cnt[c]; m.indptr[c + 1] = static_cast<int32_t>(cnt[c + 1]); }
        m.indices.resize(entries); m.data.resize(entries);
        std::vector<int32_t> pos(m.indptr.begin(), m.indptr.end() - 1);
        for (uint32_t t = 0; t < n_tid; ++t)   // contigs hold ascending row ranges and each is sorted, so rows ascend within a column
            for (const auto& x : (is_ad ? out[t].ad : out[t].dp)) if (row_ok(x.row)) { const int32_t k = pos[x.cell]++; m.indices[k] = static_cast<int32_t>(x.row); m.data[k] = static_cast<uint16_t>(x.val); }
        return true;
    };
    res.pairs_fn = [out_sp, n_tid](const std::function<void(uint32_t, uint32_t, uint32_t, uint32_t)>& f) {
        const auto& out = *out_sp;
        for (uint32_t t = 0; t < n_tid; ++t) {
            const auto& ad = out[t].ad; const auto& dp = out[t].dp; size_t ia = 0;
            for (const auto& d : dp) {   // every AD key has a DP key (a molecule that shows the ALT is a molecule that covers the site)
                while (ia < ad.size() && (ad[ia].cell < d.cell || (ad[ia].cell == d.cell && ad[ia].row < d.row))) ++ia;
                const uint32_t a = (ia < ad.size() && ad[ia].cell == d.cell && ad[ia].row == d.row) ? ad[ia].val : 0;
                f(d.cell, d.row, std::min(a, d.val), d.val);
            }
        }
    };
    res.release_fn = [out_sp, n_tid]() { auto& out = *out_sp; for (uint32_t t = 0; t < n_tid; ++t) { std::vector<Tri>().swap(out[t].ad); std::vector<Tri>().swap(out[t].dp); } };
    if (!P.defer_build) res.finalize(nullptr);
}

// Kernel 1's merged (barcode, site) counts, unit by unit, off the heap: a file in Params::spill_dir, unlinked as soon as it is made so it
// leaves with the process, or compact strings in memory when no file can be made. A unit's bytes are, per site in row order, the number of
// barcodes and then per barcode (ascending) the gap from the previous one, DP and AD, all varints. Readers decode unit by unit, so memory
// stays bounded by one unit per reader whatever the record count.
struct Spill {
    struct U { uint64_t off = 0, len = 0; uint32_t row0 = 0, ns = 0; bool mem = false; std::string b; };
    std::vector<U> u;
    int fd = -1; uint64_t end = 0, bytes = 0; std::mutex mu; std::string where = "memory";
    void open(const std::string& want, size_t n_units) {
        u.assign(n_units, U());
        std::string dir = want;
        if (dir.empty()) { const char* e = std::getenv("SINGLET_ALLSITE_SPILL_DIR"); if (!e || !*e) e = std::getenv("TMPDIR"); dir = (e && *e) ? e : "/tmp"; }
        if (dir == "-") return;
        std::string path = dir + "/singlet-allsite-XXXXXX";
        std::vector<char> p(path.begin(), path.end()); p.push_back(0);
        fd = ::mkstemp(p.data());
        if (fd >= 0) { ::unlink(p.data()); where = dir; }
    }
    void put(size_t i, uint32_t row0, uint32_t ns, std::string& b) {
        U& x = u[i]; x.row0 = row0; x.ns = ns; x.len = b.size();
        if (fd >= 0) {
            uint64_t off; { std::lock_guard<std::mutex> g(mu); off = end; end += b.size(); bytes += b.size(); }
            size_t done = 0;
            while (done < b.size()) { const ssize_t w = ::pwrite(fd, b.data() + done, b.size() - done, static_cast<off_t>(off + done)); if (w <= 0) break; done += static_cast<size_t>(w); }
            if (done == b.size()) { x.off = off; return; }
        }
        x.mem = true; x.b.swap(b); std::lock_guard<std::mutex> g(mu); bytes += x.len;
    }
    const uint8_t* get(size_t i, std::string& buf) const {
        const U& x = u[i];
        if (x.mem) return reinterpret_cast<const uint8_t*>(x.b.data());
        buf.resize(x.len); size_t done = 0;
        while (done < x.len) { const ssize_t r = ::pread(fd, &buf[done], x.len - done, static_cast<off_t>(x.off + done)); if (r <= 0) break; done += static_cast<size_t>(r); }
        if (done != x.len) { std::cerr << "[allsite] FATAL: spill read failed\n"; std::abort(); }
        return reinterpret_cast<const uint8_t*>(buf.data());
    }
    // f(barcode, row, ad, dp) for every entry of unit i, rows ascending
    template <class F> void each(size_t i, std::string& buf, F&& f) const {
        const U& x = u[i]; if (!x.len) return;
        const uint8_t* p = get(i, buf); const uint8_t* e = p + x.len;
        auto vi = [&]() { uint32_t v = 0; int sh = 0; while (p < e) { const uint8_t c = *p++; v |= static_cast<uint32_t>(c & 127) << sh; if (!(c & 128)) break; sh += 7; } return v; };
        for (uint32_t s = 0; s < x.ns; ++s) {
            const uint32_t n = vi(); uint32_t bc = 0;
            for (uint32_t j = 0; j < n; ++j) { bc += vi(); const uint32_t dp = vi(); const uint32_t ad = vi(); f(bc, x.row0 + s, ad, dp); }
        }
    }
    void clear() { std::lock_guard<std::mutex> g(mu); std::vector<U>().swap(u); if (fd >= 0) { ::close(fd); fd = -1; } }
    ~Spill() { if (fd >= 0) ::close(fd); }
    static void put_varint(std::string& o, uint32_t v) { while (v >= 128) { o.push_back(static_cast<char>((v & 127) | 128)); v >>= 7; } o.push_back(static_cast<char>(v)); }
};

// finish() for kernel 1: the same Result interface read from the spill. The counts are one value per (barcode, site), so prepare only
// sums the per-site totals and notes the kept barcodes; entries are saturated at 65535 as prepare's merge does. pairs_fn yields the
// entries site by site (rows ascend across the member, so they ascend within each barcode), which every reader accepts.
inline void finish_spill(Result& res, std::shared_ptr<std::vector<PerTid>> out_sp, std::shared_ptr<Spill> sp, uint32_t n_tid, uint32_t n_barcodes, const Params& P) {
    auto& out = *out_sp;
    for (uint32_t t = 0; t < n_tid; ++t) { for (const auto& s : out[t].sites) res.sites.push_back(s);
        res.mismatch_obs += out[t].obs; res.molecules += out[t].st.snp_molecules; res.ambiguous += out[t].st.snp_molecules_ambiguous; res.other += out[t].st.snp_molecules_other;
        res.ad_records += out[t].ad_rec; res.dp_records += out[t].dp_rec; }
    out_sp.reset();
    struct Keep { bool prepared = false, all = true; std::vector<uint8_t> v; };
    auto kp = std::make_shared<Keep>();
    const unsigned nth = std::max(1u, P.threads);
    auto kept = [n_barcodes](const Keep& k, uint32_t bc) { return !k.prepared || (bc < n_barcodes && (k.all || (bc < k.v.size() && k.v[bc]))); };
    res.prepare_fn = [sp, kp, nth, n_barcodes](Result& r, const std::vector<uint8_t>* keep) {
        const size_t nrow = r.sites.size();
        r.site_ad.assign(nrow, 0); r.site_dp.assign(nrow, 0); r.kept_ad.assign(nrow, 0); r.kept_dp.assign(nrow, 0);
        kp->prepared = true; kp->all = keep == nullptr; if (keep) kp->v = *keep; else kp->v.clear();
        const Keep& k = *kp;
        std::atomic<size_t> next{0};
        auto work = [&]() {   // units own disjoint rows
            std::string buf;
            for (size_t i; (i = next.fetch_add(1)) < sp->u.size();)
                sp->each(i, buf, [&](uint32_t bc, uint32_t row, uint32_t ad, uint32_t dp) {
                    r.site_ad[row] += ad; r.site_dp[row] += dp;
                    if (bc < n_barcodes && (k.all || (bc < k.v.size() && k.v[bc]))) { r.kept_ad[row] += ad; r.kept_dp[row] += dp; }
                });
        };
        std::vector<std::thread> th; for (unsigned i = 1; i < nth; ++i) th.emplace_back(work);
        work(); for (auto& x : th) x.join();
    };
    res.pairs_fn = [sp, kp, kept](const std::function<void(uint32_t, uint32_t, uint32_t, uint32_t)>& f) {
        std::string buf; const Keep& k = *kp;
        for (size_t i = 0; i < sp->u.size(); ++i)
            sp->each(i, buf, [&](uint32_t bc, uint32_t row, uint32_t ad, uint32_t dp) { if (kept(k, bc)) f(bc, row, std::min<uint32_t>(ad, 65535), std::min<uint32_t>(dp, 65535)); });
    };
    res.emit_fn = [sp, kp, kept, n_barcodes](Result& r, bool is_ad, const std::vector<uint8_t>* keep_rows, uint64_t& entries) -> bool {
        auto& m = is_ad ? r.ad : r.dp;
        m = {}; m.nrows = r.sites.size(); m.ncols = n_barcodes;
        const Keep& k = *kp; std::string buf;
        auto use = [&](uint32_t bc, uint32_t row, uint32_t ad) {
            return bc < n_barcodes && kept(k, bc) && (!is_ad || ad > 0) && (!keep_rows || (row < keep_rows->size() && (*keep_rows)[row])); };
        std::vector<uint64_t> cnt(static_cast<size_t>(n_barcodes) + 1, 0);
        entries = 0;
        for (size_t i = 0; i < sp->u.size(); ++i)
            sp->each(i, buf, [&](uint32_t bc, uint32_t row, uint32_t ad, uint32_t) { if (use(bc, row, ad)) { ++cnt[bc + 1]; ++entries; } });
        if (entries >= r.max_entries) {
            std::cerr << "[allsite] " << (is_ad ? "AD" : "DP") << ": " << entries << " (barcode, site) entries do not fit the int32 column pointers\n";
            return false;
        }
        m.indptr.assign(static_cast<size_t>(n_barcodes) + 1, 0);
        for (uint32_t c = 0; c < n_barcodes; ++c) { cnt[c + 1] += cnt[c]; m.indptr[c + 1] = static_cast<int32_t>(cnt[c + 1]); }
        m.indices.resize(entries); m.data.resize(entries);
        std::vector<int32_t> pos(m.indptr.begin(), m.indptr.end() - 1);
        for (size_t i = 0; i < sp->u.size(); ++i)   // rows ascend through the units, so they ascend within a column
            sp->each(i, buf, [&](uint32_t bc, uint32_t row, uint32_t ad, uint32_t dp) {
                if (!use(bc, row, ad)) return;
                const int32_t q = pos[bc]++; m.indices[q] = static_cast<int32_t>(row); m.data[q] = static_cast<uint16_t>(std::min<uint32_t>(is_ad ? ad : dp, 65535)); });
        return true;
    };
    res.n_units = sp->u.size(); res.threads = nth;
    res.unit_fn = [sp, kp, kept](size_t i, std::vector<Result::Pair>& v) {
        v.clear(); if (i >= sp->u.size()) return;
        std::string buf; const Keep& k = *kp;
        sp->each(i, buf, [&](uint32_t bc, uint32_t row, uint32_t ad, uint32_t dp) { if (kept(k, bc)) v.push_back({bc, row, std::min<uint32_t>(ad, 65535), std::min<uint32_t>(dp, 65535)}); });
    };
    res.release_fn = [sp]() { sp->clear(); };
    if (!P.defer_build) res.finalize(nullptr);
}
}  // namespace detail

// Records: n_sigs(), n_reads(), sig(i), aln(i), read(i) (SigRec / AlnRec / ReadRec of pileup_rec.h, or structs with the same fields),
// ops_at(off) and seq_at(off) (pointers to a record's CIGAR ops and packed sequence) and lowq_word(i) (64-bit mask word i). The engine
// adapts PileupInput (PileupCat over its split arrays); the unit test feeds hand-made records.
template <class Rec>
inline Result run_hashmap(const Rec& R, const Genome& G, uint32_t n_tid, uint32_t n_barcodes, const Params& P) {   // kernel 0
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

    const bool TM = std::getenv("SINGLET_ALLSITE_TIMING") != nullptr;
    Timers tmr; const uint64_t T_start = now_ns();
    auto out_sp = std::make_shared<std::vector<PerTid>>(n_tid);
    auto& out = *out_sp;
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
            const uint64_t t0 = TM ? now_ns() : 0; uint64_t tw = 0, ts = 0, tc = 0, tf = 0, nhit = 0;
            // 3a. discovery: (pos, alt) observations with their molecule
            struct Obs { uint64_t key; int32_t bc; uint64_t umi; };
            std::vector<Obs> obs;
            for (uint64_t k = b0; k < b1; ++k) {
                const uint64_t r = by_tid[k]; const auto& x = R.read(r);
                walk(r, [&](int32_t pos, uint8_t code, uint32_t ri) {
                    const int bi = base_idx(code); if (bi < 0) return;
                    const uint8_t ref = G.base(static_cast<int32_t>(t), pos); if (ref > 3 || bi == ref || !pass_q(r, ri, code)) return;
                    obs.push_back({(static_cast<uint64_t>(static_cast<uint32_t>(pos)) << 2) | static_cast<uint64_t>(bi), x.bc, P.umi_dedup ? x.umi3 : r});
                });
            }
            pt.obs = obs.size();
            const uint64_t t1 = TM ? now_ns() : 0;
            std::sort(obs.begin(), obs.end(), [](const Obs& a, const Obs& b) {
                if (a.key != b.key) return a.key < b.key; if (a.bc != b.bc) return a.bc < b.bc; return a.umi < b.umi; });
            const uint64_t t2 = TM ? now_ns() : 0;
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
            const uint64_t t3 = TM ? now_ns() : 0;
            if (TM) { tmr.walk += t1 - t0; tmr.sort += t2 - t1; tmr.scan += t3 - t2; }
            if (pt.sites.empty()) { if (TM) { tmr.contigs++; const uint64_t w = now_ns() - t0; tmr.wall += w;
                if (w > tmr.max_wall) { tmr.max_wall = w; tmr.max_reads = b1 - b0; tmr.max_tid = t; tmr.max_sites = 0; } } continue; }
            // 3b. counting: R1 consensus per (barcode, site, UMI), R9 per (barcode, site), in blocks of sites. Every molecule of a
            // (barcode, site) falls in that site's block and the consensus sorts before it collapses, so blocks change no count.
            const size_t nsite = pt.sites.size();
            std::vector<int32_t> spos(nsite); for (size_t i = 0; i < nsite; ++i) spos[i] = pt.sites[i].pos;
            // each read's site range [rf, rl) from its alignment span; the span bounds the hits per site (spliced reads overcount)
            std::vector<uint32_t> rf(b1 - b0, 0), rl(b1 - b0, 0);
            std::vector<int64_t> load(nsite + 1, 0);
            for (uint64_t k = b0; k < b1; ++k) {
                const auto& x = R.read(by_tid[k]); const auto& sg = R.sig(x.sig); const auto& a = R.aln(sg.aln_off + sig_p[x.sig]);
                const uint32_t* o = R.ops_at(a.ops_off); int64_t end = a.pos;
                for (int i = 0; i < a.n_ops; ++i) { const int op = o[i] & 15; if (op == 0 || op == 2 || op == 3 || op == 7 || op == 8) end += o[i] >> 4; }
                const size_t f = static_cast<size_t>(std::lower_bound(spos.begin(), spos.end(), a.pos) - spos.begin());
                const size_t l = static_cast<size_t>(std::lower_bound(spos.begin(), spos.end(), end) - spos.begin());
                if (f >= l) continue;
                rf[k - b0] = static_cast<uint32_t>(f); rl[k - b0] = static_cast<uint32_t>(l); load[f]++; load[l]--;
            }
            std::vector<size_t> cuts{0};
            { int64_t run = 0, acc = 0;
              for (size_t i = 0; i < nsite; ++i) { run += load[i]; acc += run; if (acc >= static_cast<int64_t>(P.block_hits) && i + 1 < nsite) { cuts.push_back(i + 1); acc = 0; } } }
            cuts.push_back(nsite);
            std::vector<int64_t>().swap(load);
            // the reads of each block, in read order: a read joins every block its site range overlaps. Scanning all reads of the
            // contig per block cost blocks x reads and ran an LCL's all-site pass past the watchdog.
            const size_t nb = cuts.size() - 1;
            std::vector<uint64_t> boff(nb + 1, 0);
            auto block_of = [&](uint32_t si) { return static_cast<size_t>(std::upper_bound(cuts.begin(), cuts.end(), static_cast<size_t>(si)) - cuts.begin()) - 1; };
            for (uint64_t k = b0; k < b1; ++k) { if (rf[k - b0] >= rl[k - b0]) continue;
                const size_t fb = block_of(rf[k - b0]), lb = block_of(rl[k - b0] - 1); for (size_t b = fb; b <= lb; ++b) boff[b + 1]++; }
            for (size_t b = 0; b < nb; ++b) boff[b + 1] += boff[b];
            std::vector<uint32_t> bread(boff[nb]);
            { std::vector<uint64_t> w(boff.begin(), boff.end() - 1);
              for (uint64_t k = b0; k < b1; ++k) { if (rf[k - b0] >= rl[k - b0]) continue;
                  const size_t fb = block_of(rf[k - b0]), lb = block_of(rl[k - b0] - 1); for (size_t b = fb; b <= lb; ++b) bread[w[b]++] = static_cast<uint32_t>(k - b0); } }
            const uint64_t t4 = TM ? now_ns() : 0; if (TM) tmr.member += t4 - t3;
            if (P.progress) P.progress();
            SnpConsensus cons; char ub[24]; TriAcc ad, dp;
            for (size_t bi = 0; bi + 1 < cuts.size(); ++bi) {
                const size_t s0 = cuts[bi], s1 = cuts[bi + 1];
                const uint64_t u0 = TM ? now_ns() : 0;
                for (uint64_t j = boff[bi]; j < boff[bi + 1]; ++j) {
                    const uint64_t k = b0 + bread[j];
                    const uint64_t r = by_tid[k]; const auto& x = R.read(r);
                    umi_chars(x.umi3, ub);
                    // the read's sites: its aligned positions increase, so one cursor over the sorted site positions
                    size_t cur = std::max<size_t>(rf[k - b0], s0);
                    walk(r, [&](int32_t pos, uint8_t code, uint32_t ri) {
                        while (cur < s1 && spos[cur] < pos) ++cur;
                        if (cur >= s1 || spos[cur] != pos || !pass_q(r, ri, code)) return;
                        const size_t si = cur;
                        ++nhit;
                        cons.add(static_cast<uint32_t>(x.bc), static_cast<uint32_t>(si), P.umi_dedup ? ub : nullptr, P.umi_len,
                                 SnpConsensus::base_class(kNt16[code], pt.sites[si].ref, pt.sites[si].alt));
                    });
                }
                const uint64_t u1 = TM ? now_ns() : 0;
                const size_t rec0 = ad.v.size() + dp.v.size();
                cons.finalize(ad, dp, pt.st, P.directional != 0);
                if (TM) { const uint64_t u2 = now_ns(); tc += u1 - u0; tf += u2 - u1; tmr.blocks++; tmr.records += ad.v.size() + dp.v.size() - rec0; }
                cons.clear();
                if (P.progress) P.progress();
            }
            pt.ad.swap(ad.v); pt.dp.swap(dp.v);
            if (TM) { tmr.count += tc; tmr.fin += tf; tmr.hits += nhit; tmr.contigs++; const uint64_t w = now_ns() - t0; tmr.wall += w;
                if (w > tmr.max_wall) { tmr.max_wall = w; tmr.max_reads = b1 - b0; tmr.max_tid = t; tmr.max_sites = nsite; } }
        }
    };
    const unsigned nth = std::max(1u, std::min<unsigned>(P.threads, n_tid));
    std::vector<std::thread> th; for (unsigned i = 1; i < nth; ++i) th.emplace_back(work);
    work(); for (auto& x : th) x.join();
    if (TM) {
        const double sec = 1e-9, wall = (now_ns() - T_start) * sec;
        std::cerr << "[allsite-timing] threads " << nth << " contigs " << tmr.contigs << " reads " << res.reads_used << " wall " << wall << " s; thread-seconds: discovery walk "
                  << tmr.walk * sec << ", discovery sort " << tmr.sort * sec << ", discovery scan " << tmr.scan * sec << ", block membership " << tmr.member * sec
                  << ", counting walk+inserts " << tmr.count * sec << ", finalize " << tmr.fin * sec << "; busy " << tmr.wall * sec << " of " << wall * nth
                  << "; largest contig (index " << tmr.max_tid << ", " << tmr.max_reads << " reads, " << tmr.max_sites << " sites) " << tmr.max_wall * sec
                  << " s; hits " << tmr.hits << ", blocks " << tmr.blocks << ", records (ad+dp) " << tmr.records << "\n";
    }

    for (uint32_t t = 0; t < n_tid; ++t) { out[t].ad_rec = out[t].ad.size(); out[t].dp_rec = out[t].dp.size(); }
    detail::finish(res, out_sp, n_tid, n_barcodes, P);
    return res;
}

namespace detail {
template <class F> inline void pfor(unsigned nth, size_t n, F&& f) {   // f(item, worker) over [0, n), items taken in order
    std::atomic<size_t> next{0};
    auto w = [&](unsigned id) { for (;;) { const size_t i = next.fetch_add(1); if (i >= n) return; f(i, id); } };
    const unsigned k = static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(nth, n)));
    std::vector<std::thread> th; for (unsigned i = 1; i < k; ++i) th.emplace_back(w, i);
    w(0); for (auto& x : th) x.join();
}
// "[allsite] counting: B of N blocks ..." at most once per 30 s, from whichever worker gets there first
struct ProgressLine {
    std::atomic<uint64_t> last{0}; uint64_t t0 = now_ns();
    void tick(uint64_t done, uint64_t total, uint64_t hits) {
        const uint64_t t = now_ns(), l = last.load();
        if (t - std::max(l, t0) < 30000000000ull) return;
        uint64_t e = l; if (!last.compare_exchange_strong(e, t)) return;
        std::string s = "[allsite] counting: " + std::to_string(done) + " of " + std::to_string(total) + " blocks, " + std::to_string(hits / 1000000)
                      + " M hits, " + std::to_string((t - t0) / 1000000000ull) + " s\n";
        std::cerr << s << std::flush;
    }
};
}  // namespace detail

// Kernel 1: the same counts from sorted records (allsite_records.h). Discovery runs over chunks of each contig's reads; counting runs over
// (contig, block) units, so threads stay busy past the largest contig. Every unit's output depends only on its reads, and units are put
// back in (contig, block) order, so the result does not depend on the thread count.
}}  // namespace singlet::allsite
#include "allsite_device_glue.h"   // Params::device: the device route's hook
namespace singlet { namespace allsite {

template <class Rec>
inline Result run_sorted(const Rec& R, const Genome& G, uint32_t n_tid, uint32_t n_barcodes, const Params& P, const rec::Layout& Y) {
    using namespace detail;
    Result res;
    const bool TM = std::getenv("SINGLET_ALLSITE_TIMING") != nullptr;
    const uint64_t T_start = now_ns();
    const uint64_t nS = R.n_sigs(), nR = R.n_reads();
    const unsigned nth = std::max(1u, P.threads);
    // 1-2. as kernel 0: each signature's qualifying primary alignment; reads per contig ordered by read index
    std::vector<int32_t> sig_tid(nS, -1);
    std::vector<uint8_t> sig_p(nS, 0);
    pfor(nth, (nS + (1u << 20) - 1) >> 20, [&](size_t c, unsigned) {
        const uint64_t e = std::min<uint64_t>(nS, (c + 1) << 20);
        for (uint64_t s = c << 20; s < e; ++s) {
            const auto& sg = R.sig(s);
            for (int k = 0; k < sg.n_aln; ++k) {
                const auto& a = R.aln(sg.aln_off + k);
                if (!(a.flags & 2)) continue;
                if (a.mapq >= P.min_mapq && a.tid >= 0 && static_cast<uint32_t>(a.tid) < n_tid && a.tid != P.skip_tid) { sig_tid[s] = a.tid; sig_p[s] = static_cast<uint8_t>(k); }
                break;
            }
        }
    });
    auto use = [&](const auto& x) { return x.bc >= 0 && static_cast<uint32_t>(x.bc) < n_barcodes && sig_tid[x.sig] >= 0; };
    // reads per contig in read order, from equal slices of the reads: slice-local counts, then each slice scatters its reads from its own
    // offsets, which gives the serial order on any thread count
    std::vector<uint64_t> cnt(n_tid + 1, 0);
    std::vector<uint64_t> by_tid;
    {
        const unsigned sth = static_cast<unsigned>(std::max<uint64_t>(1, std::min<uint64_t>(nth, nR / 65536)));
        std::vector<std::vector<uint64_t>> sc(sth, std::vector<uint64_t>(n_tid, 0));
        auto slice = [&](unsigned k, bool write) {
            const uint64_t a = nR * k / sth, b = nR * (k + 1) / sth;
            std::vector<uint64_t>& w = sc[k];
            for (uint64_t r = a; r < b; ++r) {
                if (r + 16 < b) __builtin_prefetch(&sig_tid[R.read(r + 16).sig]);
                const auto& x = R.read(r); if (!use(x)) continue;
                const int32_t t = sig_tid[x.sig];
                if (write) by_tid[w[t]++] = r; else w[t]++;
            }
        };
        pfor(sth, sth, [&](size_t k, unsigned) { slice(static_cast<unsigned>(k), false); });
        for (uint32_t t = 0; t < n_tid; ++t) { uint64_t sum = 0; for (unsigned k = 0; k < sth; ++k) sum += sc[k][t]; cnt[t + 1] = cnt[t] + sum; }
        for (uint32_t t = 0; t < n_tid; ++t) { uint64_t o = cnt[t]; for (unsigned k = 0; k < sth; ++k) { const uint64_t n = sc[k][t]; sc[k][t] = o; o += n; } }
        by_tid.resize(cnt[n_tid]);
        pfor(sth, sth, [&](size_t k, unsigned) { slice(static_cast<unsigned>(k), true); });
    }
    res.reads_used = by_tid.size();
    std::vector<uint32_t> order(n_tid); for (uint32_t t = 0; t < n_tid; ++t) order[t] = t;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { const uint64_t na = cnt[a + 1] - cnt[a], nb = cnt[b + 1] - cnt[b]; return na != nb ? na > nb : a < b; });
    const uint64_t T_prep = now_ns();

    auto pass_q = [&](uint64_t r, uint32_t ri, uint8_t code) { return !(P.skip_n && code == 15) && !(P.use_qual && low_q(R, P.qwords, r, ri)); };
    // staged prefetch of the records of the reads ahead of position j (read, then signature, then alignment and sequence, then CIGAR):
    // the walks are bound by these misses when reads arrive in stream order
    auto pf = [&](auto id, uint64_t j, uint64_t end) {
        if (j + 16 < end) __builtin_prefetch(&R.read(id(j + 16)));
        if (j + 8 < end) { const auto& x = R.read(id(j + 8)); __builtin_prefetch(&R.sig(x.sig)); __builtin_prefetch(&sig_p[x.sig]); }
        if (j + 4 < end) { const auto& x = R.read(id(j + 4)); const auto& sg = R.sig(x.sig); __builtin_prefetch(&R.aln(sg.aln_off + sig_p[x.sig])); __builtin_prefetch(R.seq_at(sg.seq_off)); }
        if (j + 2 < end) { const auto& x = R.read(id(j + 2)); const auto& sg = R.sig(x.sig); __builtin_prefetch(R.ops_at(R.aln(sg.aln_off + sig_p[x.sig]).ops_off)); }
    };
    auto by_id = [&](uint64_t k) { return by_tid[k]; };
    auto out_sp = std::make_shared<std::vector<PerTid>>(n_tid);
    auto& out = *out_sp;
    if (P.device && P.umi_dedup && P.umi_len >= 1 && P.umi_len <= 20) {   // the device route: its counts go straight into the spill
        auto spl = std::make_shared<Spill>(); spl->open(P.spill_dir, 0);
        uint64_t below = 0;
        if (device_pass(R, G, P, n_tid, n_barcodes, sig_p, cnt, by_tid, order, out, *spl, below, T_start)) {
            res.sites_below_floor = below;
            finish_spill(res, out_sp, spl, n_tid, n_barcodes, P);
            return res;
        }
    }
    std::atomic<uint64_t> t_walk{0}, t_sort{0}, t_member{0}, t_cwalk{0}, t_csort{0}, t_coll{0}, n_hits{0};
    std::atomic<uint64_t> g_1{0}, g_8{0}, g_64{0}, g_512{0}, g_big{0}, g_m2{0};

    // 3a. discovery observations over chunks of reads (chunk order = contig then read order)
    struct Obs { uint64_t key; int32_t bc; uint64_t umi; };
    constexpr uint64_t CH = 1u << 18;
    struct Chunk { uint32_t t; uint64_t k0, k1; };
    std::vector<Chunk> chunks; std::vector<size_t> ch0(n_tid + 1, 0);
    for (uint32_t t = 0; t < n_tid; ++t) { ch0[t] = chunks.size(); for (uint64_t k = cnt[t]; k < cnt[t + 1]; k += CH) chunks.push_back({t, k, std::min(cnt[t + 1], k + CH)}); }
    ch0[n_tid] = chunks.size();
    std::vector<std::vector<Obs>> cobs(chunks.size());
    std::vector<uint32_t> RF(by_tid.size(), 0), RL(by_tid.size(), 0);   // each read's alignment span here, its site range after 3b
    pfor(nth, chunks.size(), [&](size_t ci, unsigned) {
        const uint64_t u0 = TM ? now_ns() : 0;
        const Chunk& C = chunks[ci]; const int32_t t = static_cast<int32_t>(C.t); auto& obs = cobs[ci];
        static const int8_t bidx[16] = {-1, 0, 1, -1, 2, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1};   // base_idx
        for (uint64_t k = C.k0; k < C.k1; ++k) {
            pf(by_id, k, C.k1);
            const uint64_t r = by_tid[k]; const auto& x = R.read(r);
            const auto& sg = R.sig(x.sig); const auto& a = R.aln(sg.aln_off + sig_p[x.sig]);
            const bool rev = (a.flags & 1) != 0; const uint32_t* o = R.ops_at(a.ops_off); const uint8_t* seq = R.seq_at(sg.seq_off);
            const uint64_t um = P.umi_dedup ? x.umi3 : r;
            int32_t rpos = a.pos; int qpos = 0;
            for (int i = 0; i < a.n_ops; ++i) {   // walk()'s bases, with the reference bytes of each block read in one span
                const uint32_t c = o[i]; const int op = c & 15; const int32_t len = static_cast<int32_t>(c >> 4);
                if (op == 0 || op == 7 || op == 8) {
                    const uint8_t* gp = G.span(t, rpos, len);
                    for (int32_t j = 0; j < len; ++j) {
                        const int qp = qpos + j; const uint8_t code = aligned_base(seq, sg.rlen, qp, rev);
                        const int bi = bidx[code]; if (bi < 0) continue;
                        uint8_t ref = gp ? gp[j] : G.base(t, rpos + j); if (ref > 3) continue;
                        if (bi == ref || !pass_q(r, static_cast<uint32_t>(rev ? sg.rlen - 1 - qp : qp), code)) continue;
                        obs.push_back({(static_cast<uint64_t>(static_cast<uint32_t>(rpos + j)) << 2) | static_cast<uint64_t>(bi), x.bc, um});
                    }
                    rpos += len; qpos += len;
                } else if (op == 1 || op == 4) qpos += len;
                else if (op == 2 || op == 3) rpos += len;
            }
            RF[k] = static_cast<uint32_t>(a.pos); RL[k] = static_cast<uint32_t>(rpos);
        }
        if (TM) t_walk += now_ns() - u0;
        if (P.progress) P.progress();
    });
    const uint64_t T_a1 = now_ns();

    // 3b. per contig: sites (kernel 0's rule), each read's site range, blocks of about block_hits hits, each block's reads in read order
    struct Plan { std::vector<int32_t> spos; std::vector<uint8_t> refc, altc; std::vector<size_t> cuts; std::vector<uint64_t> boff; std::vector<uint32_t> bread; };
    std::vector<Plan> plan(n_tid);
    pfor(nth, n_tid, [&](size_t oi, unsigned) {
        const uint32_t t = order[oi]; const uint64_t b0 = cnt[t], b1 = cnt[t + 1]; if (b0 == b1) return;
        const uint64_t u0 = TM ? now_ns() : 0;
        PerTid& pt = out[t];
        std::vector<Obs> obs;
        { size_t n = 0; for (size_t c = ch0[t]; c < ch0[t + 1]; ++c) n += cobs[c].size(); obs.reserve(n);
          for (size_t c = ch0[t]; c < ch0[t + 1]; ++c) { obs.insert(obs.end(), cobs[c].begin(), cobs[c].end()); std::vector<Obs>().swap(cobs[c]); } }
        pt.obs = obs.size();
        std::sort(obs.begin(), obs.end(), [](const Obs& a, const Obs& b) {
            if (a.key != b.key) return a.key < b.key; if (a.bc != b.bc) return a.bc < b.bc; return a.umi < b.umi; });
        for (size_t s = 0; s < obs.size();) {
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
        const uint64_t u1 = TM ? now_ns() : 0;
        if (TM) t_sort += u1 - u0;
        if (pt.sites.empty()) return;
        Plan& C = plan[t];
        const size_t nsite = pt.sites.size();
        auto nt16 = [](char c) -> uint8_t { return c == 'A' ? 1 : c == 'C' ? 2 : c == 'G' ? 4 : 8; };
        C.spos.resize(nsite); C.refc.resize(nsite); C.altc.resize(nsite);
        for (size_t i = 0; i < nsite; ++i) { C.spos[i] = pt.sites[i].pos; C.refc[i] = nt16(pt.sites[i].ref); C.altc[i] = nt16(pt.sites[i].alt); }
        if (TM) t_sort += now_ns() - u1;
        if (P.progress) P.progress();
    });
    std::vector<std::vector<Obs>>().swap(cobs);
    // each read's site range [rf, rl) from its alignment span (the span bounds the hits per site; spliced reads overcount), over chunks
    pfor(nth, chunks.size(), [&](size_t ci, unsigned) {
        const uint64_t u0 = TM ? now_ns() : 0;
        const Chunk& Ch = chunks[ci]; const auto& spos = plan[Ch.t].spos; if (spos.empty()) return;
        for (uint64_t k = Ch.k0; k < Ch.k1; ++k) {
            const int32_t p0 = static_cast<int32_t>(RF[k]), end = static_cast<int32_t>(RL[k]);   // the span 3a kept
            const size_t f = static_cast<size_t>(std::lower_bound(spos.begin(), spos.end(), p0) - spos.begin());
            const size_t l = static_cast<size_t>(std::lower_bound(spos.begin(), spos.end(), end) - spos.begin());
            if (f < l) { RF[k] = static_cast<uint32_t>(f); RL[k] = static_cast<uint32_t>(l); } else { RF[k] = 0; RL[k] = 0; }
        }
        if (TM) t_member += now_ns() - u0;
    });
    // per contig: blocks of about block_hits hits, each block's reads in read order
    pfor(nth, n_tid, [&](size_t oi, unsigned) {
        const uint32_t t = order[oi]; const uint64_t b0 = cnt[t], b1 = cnt[t + 1]; Plan& C = plan[t]; if (C.spos.empty()) return;
        const uint64_t u0 = TM ? now_ns() : 0;
        const size_t nsite = C.spos.size();
        const uint32_t* rf = RF.data() + b0; const uint32_t* rl = RL.data() + b0;
        std::vector<int64_t> load(nsite + 1, 0);
        for (uint64_t k = 0; k < b1 - b0; ++k) if (rf[k] < rl[k]) { load[rf[k]]++; load[rl[k]]--; }
        C.cuts.assign(1, 0);
        { int64_t run = 0, acc = 0;
          for (size_t i = 0; i < nsite; ++i) { run += load[i]; acc += run; if (acc >= static_cast<int64_t>(P.block_hits) && i + 1 < nsite) { C.cuts.push_back(i + 1); acc = 0; } } }
        C.cuts.push_back(nsite);
        std::vector<int64_t>().swap(load);
        const size_t nb = C.cuts.size() - 1;
        C.boff.assign(nb + 1, 0);
        auto block_of = [&](uint32_t si) { return static_cast<size_t>(std::upper_bound(C.cuts.begin(), C.cuts.end(), static_cast<size_t>(si)) - C.cuts.begin()) - 1; };
        for (uint64_t k = 0; k < b1 - b0; ++k) { if (rf[k] >= rl[k]) continue;
            const size_t fb = block_of(rf[k]), lb = block_of(rl[k] - 1); for (size_t b = fb; b <= lb; ++b) C.boff[b + 1]++; }
        for (size_t b = 0; b < nb; ++b) C.boff[b + 1] += C.boff[b];
        C.bread.resize(C.boff[nb]);
        { std::vector<uint64_t> w(C.boff.begin(), C.boff.end() - 1);
          for (uint64_t k = 0; k < b1 - b0; ++k) { if (rf[k] >= rl[k]) continue;
              const size_t fb = block_of(rf[k]), lb = block_of(rl[k] - 1); for (size_t b = fb; b <= lb; ++b) C.bread[w[b]++] = static_cast<uint32_t>(k); } }
        if (TM) t_member += now_ns() - u0;
        if (P.progress) P.progress();
    });
    std::vector<uint32_t>().swap(RF); std::vector<uint32_t>().swap(RL);
    const uint64_t T_a2 = now_ns();

    // 3c. counting units, largest first (reads in the block; ties: contig, block)
    struct Unit { uint32_t t, b; uint64_t w; };
    std::vector<Unit> units;
    std::vector<size_t> u0(n_tid + 1, 0);
    for (uint32_t t = 0; t < n_tid; ++t) { u0[t] = units.size();
        for (uint32_t b = 0; b + 1 < plan[t].cuts.size(); ++b) units.push_back({t, b, plan[t].boff[b + 1] - plan[t].boff[b]}); }
    u0[n_tid] = units.size();
    std::vector<uint32_t> uorder(units.size()); for (size_t i = 0; i < units.size(); ++i) uorder[i] = static_cast<uint32_t>(i);
    std::sort(uorder.begin(), uorder.end(), [&](uint32_t a, uint32_t b) { return units[a].w != units[b].w ? units[a].w > units[b].w : a < b; });
    struct UnitOut { Stats st; uint64_t adr = 0, dpr = 0; std::vector<uint8_t> keep; };
    std::vector<UnitOut> uout(units.size());
    auto spl = std::make_shared<Spill>(); spl->open(P.spill_dir, units.size());
    struct Hit { uint64_t key; uint32_t site; };
    struct WS { std::vector<Hit> hits; std::vector<uint64_t> keys; std::vector<uint64_t> off; rec::Scratch sc; std::string enc, site; };
    std::vector<WS> ws(nth);
    ProgressLine pl; std::atomic<uint64_t> done{0};
    pfor(nth, units.size(), [&](size_t ui, unsigned id) {
        const uint32_t un = uorder[ui]; const Unit& U = units[un]; const uint32_t t = U.t; const Plan& C = plan[t]; UnitOut& o = uout[un];
        WS& W = ws[id];
        const uint64_t v0 = TM ? now_ns() : 0;
        const size_t s0 = C.cuts[U.b], s1 = C.cuts[U.b + 1], ns = s1 - s0; const uint64_t b0 = cnt[t];
        const int32_t* sp = C.spos.data();
        W.hits.clear();
        auto bid = [&](uint64_t jj) { return by_tid[b0 + C.bread[jj]]; };
        for (uint64_t j = C.boff[U.b]; j < C.boff[U.b + 1]; ++j) {
            pf(bid, j, C.boff[U.b + 1]);
            const uint64_t r = by_tid[b0 + C.bread[j]]; const auto& x = R.read(r);
            const auto& sg = R.sig(x.sig); const auto& a = R.aln(sg.aln_off + sig_p[x.sig]);
            const uint64_t uf = Y.umi ? rec::umi_field(x.umi3, Y.L) : 0; const uint32_t bc = static_cast<uint32_t>(x.bc);
            const bool rev = (a.flags & 1) != 0; const uint32_t* ops = R.ops_at(a.ops_off); const uint8_t* seq = R.seq_at(sg.seq_off);
            size_t cur = static_cast<size_t>(std::lower_bound(sp + s0, sp + s1, a.pos) - sp);
            int32_t rpos = a.pos; int qpos = 0;
            for (int i = 0; i < a.n_ops && cur < s1; ++i) {   // walk()'s bases, visiting only the site positions of each block
                const uint32_t c = ops[i]; const int op = c & 15; const int32_t len = static_cast<int32_t>(c >> 4);
                if (op == 0 || op == 7 || op == 8) {
                    const int32_t e = rpos + len;
                    while (cur < s1 && sp[cur] < rpos) ++cur;
                    for (; cur < s1 && sp[cur] < e; ++cur) {
                        const int qp = qpos + (sp[cur] - rpos); const uint8_t code = aligned_base(seq, sg.rlen, qp, rev);
                        if (!pass_q(r, static_cast<uint32_t>(rev ? sg.rlen - 1 - qp : qp), code)) continue;
                        const uint8_t cls = code == C.altc[cur] ? 1 : code == C.refc[cur] ? 0 : 2;
                        W.hits.push_back({rec::key(Y, bc, uf, cls), static_cast<uint32_t>(cur - s0)});
                    }
                    rpos = e; qpos += len;
                } else if (op == 1 || op == 4) qpos += len;
                else if (op == 2 || op == 3) rpos += len;
            }
        }
        const uint64_t v1 = TM ? now_ns() : 0;
        W.off.assign(ns + 1, 0);
        for (const auto& h : W.hits) W.off[h.site + 1]++;
        for (size_t i = 0; i < ns; ++i) W.off[i + 1] += W.off[i];
        W.keys.resize(W.hits.size());
        { std::vector<uint64_t>& w = W.off; for (const auto& h : W.hits) W.keys[w[h.site]++] = h.key;
          for (size_t i = ns; i > 0; --i) w[i] = w[i - 1]; w[0] = 0; }
        const uint64_t v2 = TM ? now_ns() : 0;
        rec::Counts cn;
        W.enc.clear(); o.keep.assign(ns, 0); uint32_t nkeep = 0;
        for (size_t i = 0; i < ns; ++i) {   // barcodes come out ascending; a block holds every hit of its sites, so the sums are complete
            uint32_t nbc = 0, prev = 0; uint64_t sdp = 0, sad = 0; W.site.clear();
            rec::collapse_site_unsorted(W.keys.data() + W.off[i], W.off[i + 1] - W.off[i], Y, P.directional != 0, W.sc, cn, [&](uint32_t bc, uint32_t dp, uint32_t ad) {
                Spill::put_varint(W.site, bc - prev); Spill::put_varint(W.site, dp); Spill::put_varint(W.site, ad); prev = bc; ++nbc;
                sdp += dp; sad += ad;
            });
            if (!rec::floor_keep(sad, sdp, P.min_alt_pct)) continue;   // below the ALT-fraction floor: never reaches the spill
            o.keep[i] = 1; ++nkeep; o.dpr += sdp; o.adr += sad;
            Spill::put_varint(W.enc, nbc); W.enc += W.site;
        }
        spl->put(un, 0, nkeep, W.enc);   // the unit's first row is set once every unit's kept sites are known
        o.st.snp_molecules = cn.molecules; o.st.snp_molecules_ambiguous = cn.ambiguous; o.st.snp_molecules_other = cn.other;
        if (TM) { g_1 += cn.g1; g_8 += cn.g8; g_64 += cn.g64; g_512 += cn.g512; g_big += cn.gbig; g_m2 += cn.m2; }
        const uint64_t nh = W.hits.size(); n_hits += nh;
        if (W.hits.capacity() > 4 * (P.block_hits + 1024)) { std::vector<Hit>().swap(W.hits); std::vector<uint64_t>().swap(W.keys); }
        if (TM) { const uint64_t v3 = now_ns(); t_cwalk += v1 - v0; t_csort += v2 - v1; t_coll += v3 - v2; }
        if (P.progress) P.progress();
        pl.tick(++done, units.size(), n_hits.load());
    });
    std::vector<WS>().swap(ws);
    uint32_t kept_rows = 0;
    for (uint32_t t = 0; t < n_tid; ++t) {   // units back in (contig, block) order
        PerTid& pt = out[t]; size_t k = 0;   // the kept sites, renumbered; each unit's first row follows the kept sites before it
        for (size_t u = u0[t]; u < u0[t + 1]; ++u) {
            UnitOut& o = uout[u];
            pt.st.snp_molecules += o.st.snp_molecules; pt.st.snp_molecules_ambiguous += o.st.snp_molecules_ambiguous; pt.st.snp_molecules_other += o.st.snp_molecules_other;
            pt.ad_rec += o.adr; pt.dp_rec += o.dpr;
            spl->u[u].row0 = kept_rows + static_cast<uint32_t>(k);
            const size_t s0 = plan[t].cuts[units[u].b];
            for (size_t i = 0; i < o.keep.size(); ++i) if (o.keep[i]) pt.sites[k++] = pt.sites[s0 + i];
            std::vector<uint8_t>().swap(o.keep);
        }
        res.sites_below_floor += pt.sites.size() - k;
        pt.sites.resize(k); kept_rows += static_cast<uint32_t>(k);
    }
    if (TM) std::cerr << "[allsite-timing] merged counts: " << spl->bytes << " bytes in " << spl->where << "\n";
    if (TM) {
        const double sec = 1e-9; const uint64_t T_b = now_ns();
        std::cerr << "[allsite-timing] kernel sorted, threads " << nth << ", reads " << res.reads_used << ", units " << units.size() << ", hits " << n_hits.load()
                  << "; wall s: prepare " << (T_prep - T_start) * sec << ", discovery walk " << (T_a1 - T_prep) * sec << ", sites+blocks " << (T_a2 - T_a1) * sec
                  << ", counting+assembly " << (T_b - T_a2) * sec << "; thread-s: discovery walk " << t_walk * sec << ", discovery sort+scan " << t_sort * sec
                  << ", block membership " << t_member * sec << ", counting walk " << t_cwalk * sec << ", counting sort " << t_csort * sec << ", collapse " << t_coll * sec
                  << "; groups (molecules per barcode and site) 1: " << g_1.load() << ", 2-8: " << g_8.load() << ", 9-64: " << g_64.load() << ", 65-512: " << g_512.load()
                  << ", >512: " << g_big.load() << ", sum m^2 (m >= 2) " << g_m2.load() << "\n";
    }
    finish_spill(res, out_sp, spl, n_tid, n_barcodes, P);
    return res;
}

// The pass: kernel 1 (sorted records) unless Params::kernel or SINGLET_ALLSITE_KERNEL=old asks for kernel 0, or the keys do not fit
template <class Rec>
inline Result run(const Rec& R, const Genome& G, uint32_t n_tid, uint32_t n_barcodes, const Params& P) {
    int k = P.kernel;
    if (k < 0) { const char* e = std::getenv("SINGLET_ALLSITE_KERNEL"); k = (e && std::string(e) == "old") ? 0 : 1; }
    const rec::Layout Y = rec::layout(P.umi_dedup != 0, P.umi_len, n_barcodes);
    if (k == 1 && Y.ok) return run_sorted(R, G, n_tid, n_barcodes, P, Y);
    return run_hashmap(R, G, n_tid, n_barcodes, P);
}

inline std::string site_name(const Site& s, const std::vector<std::string>& tid_names) {
    return tid_names[static_cast<size_t>(s.tid)] + ":" + std::to_string(s.pos + 1) + ":" + s.ref + ">" + s.alt;
}

// sites.tsv.gz: one line per row of the matrices (chrom, 1-based pos, ref, alt, ALT molecules and barcodes at discovery, summed DP and AD
// over every barcode, and with per_cell given, a last column: 1 when the stored matrices keep the row's per-cell entries; allsite_store.h)
inline bool write_sites_tsv(const std::string& path, const Result& r, const std::vector<std::string>& tid_names,
                            const std::vector<uint8_t>* per_cell = nullptr) {
    std::vector<uint64_t> dp(r.sites.size(), 0), ad(r.sites.size(), 0);
    if (r.site_dp.size() == r.sites.size() && r.site_ad.size() == r.sites.size()) { dp = r.site_dp; ad = r.site_ad; }
    else for (size_t k = 0; k < r.dp.data.size(); ++k) dp[static_cast<size_t>(r.dp.indices[k])] += r.dp.data[k];
    if (r.site_ad.size() != r.sites.size()) for (size_t k = 0; k < r.ad.data.size(); ++k) ad[static_cast<size_t>(r.ad.indices[k])] += r.ad.data[k];
    const std::string tmp = path + ".tmp";
    gzFile f = gzopen(tmp.c_str(), "wb6");
    if (!f) return false;
    std::string line = per_cell ? "chrom\tpos\tref\talt\talt_molecules\talt_barcodes\tdp\tad\tper_cell\n"
                                : "chrom\tpos\tref\talt\talt_molecules\talt_barcodes\tdp\tad\n";
    bool ok = gzwrite(f, line.data(), static_cast<unsigned>(line.size())) > 0;
    for (size_t i = 0; ok && i < r.sites.size(); ++i) {
        const Site& s = r.sites[i];
        line = tid_names[static_cast<size_t>(s.tid)]; line += '\t'; line += std::to_string(s.pos + 1); line += '\t'; line += s.ref; line += '\t'; line += s.alt;
        line += '\t'; line += std::to_string(s.alt_molecules); line += '\t'; line += std::to_string(s.alt_cells);
        line += '\t'; line += std::to_string(dp[i]); line += '\t'; line += std::to_string(ad[i]);
        if (per_cell) { line += '\t'; line += (i < per_cell->size() && (*per_cell)[i]) ? '1' : '0'; }
        line += '\n';
        ok = gzwrite(f, line.data(), static_cast<unsigned>(line.size())) > 0;
    }
    if (gzclose(f) != Z_OK) ok = false;
    if (ok) ok = std::rename(tmp.c_str(), path.c_str()) == 0;
    if (!ok) std::remove(tmp.c_str());
    return ok;
}

}}  // namespace singlet::allsite
