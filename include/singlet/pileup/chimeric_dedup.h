// SPDX-License-Identifier: MIT
// chimeric_dedup.h - stage A of the deduplicated --chimeric second pass (SINGLET_CHIM_DEDUP).
//
// The second pass maps every read the main pass left unmapped (and the soft-clipped ones). Many of them have the same cDNA sequence, and a
// chimeric alignment is a function of that sequence only (STAR's alignment, clip and chimeric detection use the read bases and the genome;
// the junction line's other columns are the read name and, for CB_UMI_Simple, the CB and UMI that SoloReadBarcode::getCBandUMI cuts out of
// the barcode mate's sequence). This stage reads the pass's input (cDNA mate, barcode mate) in file order, finds the distinct cDNA
// sequences by comparing the full sequences, writes one FASTQ pair of representatives (the first read of each sequence, named "@<id>"),
// and records for every read its sequence id, cell barcode and UMI (MemberTable, members grouped by id in input order).
// Two threads: the cDNA mate (hash table, representatives) and the barcode mate (barcode codes, representatives' barcode records).
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "singlet/pileup/chimeric_table.h"

namespace singlet {
namespace chimeric {

struct DedupParams {
    uint32_t cb_start = 1, cb_len = 16, umi_start = 17, umi_len = 10;   ///< STAR's --soloCBstart, --soloCBlen, --soloUMIstart, --soloUMIlen
    uint32_t barcode_len = 0;   ///< --soloBarcodeReadLength after STAR's rule (1 -> cb_len + umi_len): > 0 = every barcode read has exactly this length
};

struct DedupResult {
    bool ok = false;
    std::string reason;             ///< when !ok
    uint64_t n_reads = 0;           ///< reads in all inputs
    uint64_t n_first = 0, n_second = 0;   ///< reads of the first and of the second input pair
    uint64_t n_distinct = 0;
    uint64_t rss_peak_kb = 0;       ///< largest VmRSS of the process seen inside the stage (sampled)
    MemberTable table;
};

namespace dedup_detail {

inline uint64_t rss_kb() {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (std::fgets(line, sizeof line, f)) {
        if (std::strncmp(line, "VmRSS:", 6) == 0) { kb = std::strtoull(line + 6, nullptr, 10); break; }
    }
    std::fclose(f);
    return kb;
}

template <class T>
class ChunkedVec {
    static constexpr size_t kShift = 22, kN = size_t(1) << kShift;
    std::vector<std::unique_ptr<T[]>> blocks_;
    size_t n_ = 0;
public:
    void push_back(T v) {
        if ((n_ >> kShift) == blocks_.size()) blocks_.emplace_back(new T[kN]);
        blocks_[n_ >> kShift][n_ & (kN - 1)] = v;
        ++n_;
    }
    T operator[](size_t i) const { return blocks_[i >> kShift][i & (kN - 1)]; }
    size_t size() const { return n_; }
    void clear() { blocks_.clear(); blocks_.shrink_to_fit(); n_ = 0; }
};

/// Reads 4-line FASTQ records from a plain file.
class Reader {
    FILE* f_ = nullptr;
    std::vector<char> buf_;
    size_t b_ = 0, e_ = 0;
    bool eof_ = false;
public:
    bool bad = false;   ///< a partial record at the end of the file
    ~Reader() { close(); }
    bool open(const std::string& p) {
        close();
        f_ = std::fopen(p.c_str(), "rb");
        buf_.resize(size_t(8) << 20);
        b_ = e_ = 0; eof_ = false; bad = false;
        return f_ != nullptr;
    }
    void close() { if (f_) { std::fclose(f_); f_ = nullptr; } }
    /// the next record's four lines (valid until the next call); false at the end of the file
    bool next(std::string_view L[4]) {
        for (;;) {
            size_t p = b_, st[4] = {0, 0, 0, 0}, en[4] = {0, 0, 0, 0};
            int got = 0;
            while (got < 4) {
                const char* nl = static_cast<const char*>(std::memchr(buf_.data() + p, '\n', e_ - p));
                if (!nl) break;
                st[got] = p;
                en[got] = static_cast<size_t>(nl - buf_.data());
                p = en[got] + 1;
                ++got;
            }
            if (got == 4 || (eof_ && got == 3 && p < e_)) {
                if (got == 3) { st[3] = p; en[3] = e_; p = e_; }
                for (int i = 0; i < 4; i++) L[i] = std::string_view(buf_.data() + st[i], en[i] - st[i]);
                b_ = p;
                return true;
            }
            if (eof_) {
                if (b_ != e_) bad = true;
                return false;
            }
            if (b_ > 0) {
                std::memmove(buf_.data(), buf_.data() + b_, e_ - b_);
                e_ -= b_;
                b_ = 0;
            } else if (e_ == buf_.size()) {
                buf_.resize(buf_.size() * 2);
            }
            const size_t r = f_ ? std::fread(buf_.data() + e_, 1, buf_.size() - e_, f_) : 0;
            if (r == 0) eof_ = true;
            e_ += r;
        }
    }
    /// true when no further byte is left (call after the last record)
    bool exhausted() {
        if (b_ < e_) return false;
        if (!eof_) {
            b_ = 0;
            e_ = f_ ? std::fread(buf_.data(), 1, buf_.size(), f_) : 0;
            if (e_ == 0) eof_ = true;
        }
        return eof_ && b_ == e_;
    }
};

inline uint8_t base_code(unsigned char c) {
    switch (c) { case 'A': return 0; case 'C': return 1; case 'G': return 2; case 'T': return 3; case 'N': return 4; default: return 255; }
}

/// 3 bits per base, tight; returns the byte count, or -1 for a base outside ACGTN
inline int pack_seq(const char* s, size_t n, uint8_t* out) {
    uint32_t acc = 0;
    int nb = 0;
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        const uint8_t c = base_code(static_cast<unsigned char>(s[i]));
        if (c > 4) return -1;
        acc = (acc << 3) | c;
        nb += 3;
        while (nb >= 8) { out[o++] = static_cast<uint8_t>((acc >> (nb - 8)) & 0xFF); nb -= 8; }
    }
    if (nb > 0) out[o++] = static_cast<uint8_t>((acc << (8 - nb)) & 0xFF);
    return static_cast<int>(o);
}

inline uint64_t hash_bytes(const uint8_t* p, size_t n, size_t len) {
    uint64_t h = 0x9E3779B97F4A7C15ull ^ (len * 0xC2B2AE3D27D4EB4Full);
    while (n >= 8) {
        uint64_t w;
        std::memcpy(&w, p, 8);
        h = (h ^ w) * 0xFF51AFD7ED558CCDull;
        h ^= h >> 32;
        p += 8; n -= 8;
    }
    if (n) {
        uint64_t w = 0;
        std::memcpy(&w, p, n);
        h = (h ^ w) * 0xC4CEB9FE1A85EC53ull;
        h ^= h >> 29;
    }
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33;
    return h;
}

/// distinct sequences: packed bytes in an arena, open-addressing table of (32-bit hash tag, id + 1)
class SeqSet {
    static constexpr size_t kBlock = size_t(64) << 20;
    std::vector<std::unique_ptr<uint8_t[]>> blocks_;
    size_t used_ = kBlock;
    std::vector<uint64_t> handle_;   // block << 32 | offset
    std::vector<uint16_t> len_;
    std::vector<uint64_t> slots_;
    int bits_ = 24;
    const uint8_t* at(uint64_t h) const { return blocks_[h >> 32].get() + (h & 0xFFFFFFFFull); }
    size_t index(uint32_t tag) const { return static_cast<size_t>(tag) >> (32 - bits_); }
    void grow() {
        std::vector<uint64_t> ns(size_t(1) << (bits_ + 1), 0);
        ++bits_;
        const size_t mask = ns.size() - 1;
        for (uint64_t s : slots_) {
            if (!s) continue;
            size_t i = index(static_cast<uint32_t>(s >> 32));
            while (ns[i]) i = (i + 1) & mask;
            ns[i] = s;
        }
        slots_.swap(ns);
    }
public:
    SeqSet() : slots_(size_t(1) << 24, 0) {}
    uint64_t size() const { return handle_.size(); }
    /// id of the sequence (packed bytes p[0..n), length len in bases); isnew when it was not seen before
    uint32_t find_or_insert(const uint8_t* p, size_t n, size_t len, bool& isnew) {
        const uint64_t h = hash_bytes(p, n, len);
        const uint32_t tag = static_cast<uint32_t>(h >> 32);
        const size_t mask = slots_.size() - 1;
        size_t i = index(tag);
        for (; slots_[i]; i = (i + 1) & mask) {
            if (static_cast<uint32_t>(slots_[i] >> 32) != tag) continue;
            const uint32_t id = static_cast<uint32_t>(slots_[i]) - 1;
            if (len_[id] == len && std::memcmp(at(handle_[id]), p, n) == 0) { isnew = false; return id; }
        }
        isnew = true;
        if (used_ + n > kBlock) { blocks_.emplace_back(new uint8_t[kBlock]); used_ = 0; }
        const uint32_t id = static_cast<uint32_t>(handle_.size());
        std::memcpy(blocks_.back().get() + used_, p, n);
        handle_.push_back((static_cast<uint64_t>(blocks_.size() - 1) << 32) | used_);
        len_.push_back(static_cast<uint16_t>(len));
        used_ += n;
        slots_[i] = (static_cast<uint64_t>(tag) << 32) | (static_cast<uint64_t>(id) + 1);
        if ((handle_.size() + 1) * 10 > slots_.size() * 7) grow();
        return id;
    }
};

struct Batch {
    int file = 0;
    bool last = false;                 ///< the last batch of its file
    std::vector<uint8_t> rep;          ///< per read: 1 = first read of its sequence
};

/// STAR's getCBandUMI for CB_UMI_Simple, as codes: false (with a reason) when the result is not exactly reproducible from a packed code
inline bool barcode_codes(std::string_view s, const DedupParams& P, uint64_t& cb, uint64_t& umi, const char*& why) {
    const size_t size = s.size(), cbumi = static_cast<size_t>(P.cb_len) + P.umi_len;
    if (P.barcode_len > 0 && size != P.barcode_len) { why = "a barcode read whose length differs from --soloBarcodeReadLength"; return false; }
    const size_t eff = (P.barcode_len == 0 && size < cbumi) ? cbumi : size;   // STAR appends N to a short barcode read when the length is free
    char tmp[24];
    for (int w = 0; w < 2; w++) {
        const size_t pos = (w == 0 ? P.cb_start : P.umi_start) - 1, n = w == 0 ? P.cb_len : P.umi_len;
        if (pos + n > eff) { why = "a barcode read too short for the barcode or UMI position"; return false; }
        for (size_t i = 0; i < n; i++) tmp[i] = pos + i < size ? s[pos + i] : 'N';
        if (!MemberTable::pack(tmp, n, w == 0 ? cb : umi)) { why = "a barcode or UMI with a base outside ACGTN"; return false; }
    }
    return true;
}

}  // namespace dedup_detail

/// inputs: (cDNA file, barcode file) pairs in the order STAR will read them. out1/out2: the representatives' FASTQ pair.
inline DedupResult build_dedup(const std::vector<std::pair<std::string, std::string>>& inputs, const std::string& out1, const std::string& out2,
                               const DedupParams& P) {
    using namespace dedup_detail;
    DedupResult R;
    if (inputs.empty()) { R.reason = "no input"; return R; }
    if (P.cb_start < 1 || P.umi_start < 1 || P.cb_len < 1 || P.cb_len > 21 || P.umi_len < 1 || P.umi_len > 21) {
        R.reason = "barcode geometry outside what the member table stores";
        return R;
    }
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Batch> q;
    bool t1_done = false;
    std::atomic<bool> failed{false};
    std::string fail_reason;
    auto fail = [&](const std::string& why) {
        {
            std::lock_guard<std::mutex> lk(mu);
            if (!failed.load()) fail_reason = why;
            failed = true;
        }
        cv.notify_all();
    };
    ChunkedVec<uint32_t> ids;
    ChunkedVec<uint64_t> cbs, umis;
    uint64_t n_per_file[2] = {0, 0};
    uint64_t n_distinct = 0;
    uint64_t rss_peak = rss_kb();
    constexpr size_t kBatch = size_t(1) << 16, kQueueMax = 48, kFlush = size_t(4) << 20;

    // thread 1: cDNA mate
    auto t1 = [&]() {
        FILE* fo = std::fopen(out1.c_str(), "wb");
        if (!fo) { fail("cannot write " + out1); return; }
        SeqSet set;
        std::string ob;
        uint8_t pk[2048];
        for (size_t fi = 0; fi < inputs.size() && !failed; fi++) {
            Reader rd;
            if (!rd.open(inputs[fi].first)) { fail("cannot open " + inputs[fi].first); break; }
            std::string_view L[4];
            Batch b;
            b.file = static_cast<int>(fi);
            auto push = [&](bool last) {
                b.last = last;
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return q.size() < kQueueMax || failed.load(); });
                if (failed) return;
                q.push_back(std::move(b));
                lk.unlock();
                cv.notify_all();
                b = Batch();
                b.file = static_cast<int>(fi);
            };
            while (!failed && rd.next(L)) {
                if (L[0].empty() || L[0][0] != '@' || L[2].empty() || L[2][0] != '+' || L[1].size() != L[3].size() || L[1].empty() ||
                    L[1].size() > 650) { fail("a malformed FASTQ record in " + inputs[fi].first); break; }
                const int nb = pack_seq(L[1].data(), L[1].size(), pk);
                if (nb < 0) { fail("a cDNA base outside ACGTN"); break; }
                bool isnew = false;
                const uint32_t id = set.find_or_insert(pk, static_cast<size_t>(nb), L[1].size(), isnew);
                ids.push_back(id);
                b.rep.push_back(isnew ? 1 : 0);
                n_per_file[fi < 2 ? fi : 1]++;
                if (isnew) {
                    ob += '@'; ob += std::to_string(id); ob += '\n';
                    ob.append(L[1].data(), L[1].size()); ob += "\n+\n";
                    ob.append(L[3].data(), L[3].size()); ob += '\n';
                    if (ob.size() >= kFlush) {
                        if (std::fwrite(ob.data(), 1, ob.size(), fo) != ob.size()) { fail("write error on " + out1); break; }
                        ob.clear();
                    }
                }
                if (b.rep.size() >= kBatch) push(false);
            }
            if (failed) break;
            if (rd.bad) { fail("a partial FASTQ record at the end of " + inputs[fi].first); break; }
            push(true);
        }
        if (!failed && !ob.empty() && std::fwrite(ob.data(), 1, ob.size(), fo) != ob.size()) fail("write error on " + out1);
        if (std::fclose(fo) != 0) fail("write error on " + out1);
        n_distinct = set.size();
        {
            std::lock_guard<std::mutex> lk(mu);
            rss_peak = std::max(rss_peak, rss_kb());
            t1_done = true;
        }
        cv.notify_all();
    };

    // thread 2: barcode mate
    auto t2 = [&]() {
        FILE* fo = std::fopen(out2.c_str(), "wb");
        if (!fo) { fail("cannot write " + out2); return; }
        std::string ob;
        uint64_t next_id = 0;
        Reader rd;
        int open_file = -1;
        for (;;) {
            Batch b;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return !q.empty() || t1_done || failed.load(); });
                if (failed) break;
                if (q.empty()) break;   // t1_done
                b = std::move(q.front());
                q.pop_front();
            }
            cv.notify_all();
            if (b.file != open_file) {
                if (!rd.open(inputs[b.file].second)) { fail("cannot open " + inputs[b.file].second); break; }
                open_file = b.file;
            }
            std::string_view L[4];
            bool bad = false;
            for (uint8_t rep : b.rep) {
                if (!rd.next(L)) { fail("the barcode file " + inputs[b.file].second + " has fewer records than the cDNA file"); bad = true; break; }
                if (L[0].empty() || L[0][0] != '@' || L[2].empty() || L[2][0] != '+' || L[1].size() != L[3].size() || L[1].empty()) {
                    fail("a malformed FASTQ record in " + inputs[b.file].second); bad = true; break;
                }
                uint64_t c = 0, u = 0;
                const char* why = nullptr;
                if (!barcode_codes(L[1], P, c, u, why)) { fail(why); bad = true; break; }
                cbs.push_back(c);
                umis.push_back(u);
                if (rep) {
                    ob += '@'; ob += std::to_string(next_id++); ob += '\n';
                    ob.append(L[1].data(), L[1].size()); ob += "\n+\n";
                    ob.append(L[3].data(), L[3].size()); ob += '\n';
                    if (ob.size() >= kFlush) {
                        if (std::fwrite(ob.data(), 1, ob.size(), fo) != ob.size()) { fail("write error on " + out2); bad = true; break; }
                        ob.clear();
                    }
                }
            }
            if (bad) break;
            if (b.last && (!rd.exhausted() || rd.bad)) { fail("the barcode file " + inputs[b.file].second + " has more records than the cDNA file"); break; }
        }
        if (!failed && !ob.empty() && std::fwrite(ob.data(), 1, ob.size(), fo) != ob.size()) fail("write error on " + out2);
        if (std::fclose(fo) != 0) fail("write error on " + out2);
    };

    std::thread th2(t2);
    t1();
    th2.join();
    if (failed) { R.reason = fail_reason; return R; }
    const uint64_t N = ids.size();
    if (cbs.size() != N || umis.size() != N) { R.reason = "the barcode and cDNA files have different record counts"; return R; }
    if (n_distinct >= 0xFFFFFFFFull) { R.reason = "too many distinct sequences"; return R; }

    // group the members by id, input order inside an id (the representative is the first)
    MemberTable& T = R.table;
    T.cb_len = P.cb_len;
    T.umi_len = P.umi_len;
    T.start.assign(n_distinct + 1, 0);
    for (uint64_t i = 0; i < N; i++) T.start[ids[i] + 1]++;
    for (uint64_t i = 1; i <= n_distinct; i++) T.start[i] += T.start[i - 1];
    T.cb.resize(N);
    T.umi.resize(N);
    for (uint64_t i = 0; i < N; i++) {
        const uint64_t at = T.start[ids[i]]++;
        T.cb[at] = cbs[i];
        T.umi[at] = umis[i];
    }
    for (uint64_t i = n_distinct; i > 0; i--) T.start[i] = T.start[i - 1];
    T.start[0] = 0;
    rss_peak = std::max(rss_peak, rss_kb());
    R.n_reads = N;
    R.n_first = n_per_file[0];
    R.n_second = n_per_file[1];
    R.n_distinct = n_distinct;
    R.rss_peak_kb = rss_peak;
    R.ok = true;
    return R;
}

}  // namespace chimeric
}  // namespace singlet
