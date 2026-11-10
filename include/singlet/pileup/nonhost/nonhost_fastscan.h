// SPDX-License-Identifier: MIT
// nonhost_fastscan.h — the passes of the non-host screen over STAR's unmapped-read files, multi-threaded and reading each file once.
//
// The screen touches star_Unmapped.out.mate1 three times and mate2 once (nonhost_sample.h's sample_reads, NonHostCellMatrix::read_mate_pairs,
// write_classified), each a single-threaded getline loop. On a deep sample the files hold tens of GB, so the three passes cost minutes. Here
//   * fast_sample() makes one parallel pass over mate1. It returns the same SampledReads as sample_reads() (same sequences, same order, same
//     final threshold) and an index of the selected records (file offset and length, record number), kept for the later passes;
//   * fast_write_classified() copies the selected records that pass `keep(i)` from the mapped file, in the same order and bytes as
//     write_classified();
//   * fast_tag_reads() reads the barcode and UMI of the selected records from mate2 in parallel, giving the TaggedRead vector of
//     read_mate_pairs().
// Each returns false when the file does not have the regular 4-line layout the split relies on (a record whose name line does not start with
// '@' or whose third line does not start with '+', a chunk that does not end on a record boundary); the caller then runs the serial code,
// so the result never depends on which route ran.
//
// The kept set of sample_reads() is {records with a sequence and name hash below the final threshold}; the threshold halves while more than
// `cap` records are kept. The threshold only moves when the records so far exceed cap, so the final level is the first j with at most cap
// records of hash below T_j = UINT64_MAX >> j in the whole file: a count that does not depend on the order of the records, which lets chunks
// be scanned in parallel. A record's level is the largest j with hash < T_j, which is clz(hash + 1).
#pragma once

#include "nonhost_cell_matrix.h"   // TaggedRead
#include "nonhost_sample.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace singlet {
namespace nonhost {
namespace fastscan {

// A read-only mapping of a whole file.
class MappedFile {
public:
    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile() { close_file(); }
    bool open_file(const std::string& path) {
        close_file();
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return false;
        struct stat st{};
        if (::fstat(fd_, &st) != 0) { close_file(); return false; }
        n_ = static_cast<size_t>(st.st_size);
        if (n_ > 0) {
            void* m = ::mmap(nullptr, n_, PROT_READ, MAP_SHARED, fd_, 0);
            if (m == MAP_FAILED) { close_file(); return false; }
            p_ = static_cast<const char*>(m);
        }
        return true;
    }
    void close_file() {
        if (p_) ::munmap(const_cast<char*>(p_), n_);
        if (fd_ >= 0) ::close(fd_);
        p_ = nullptr; n_ = 0; fd_ = -1;
    }
    const char* data() const { return p_; }
    size_t size() const { return n_; }
private:
    const char* p_ = nullptr;
    size_t n_ = 0;
    int fd_ = -1;
};

// fn(i) for i in [0, n) on up to `threads` threads, indices handed out in order
template <class Fn>
inline void parallel_for(size_t n, int threads, Fn&& fn) {
    const size_t nt = std::min<size_t>(std::max(1, threads), n);
    if (nt <= 1) { for (size_t i = 0; i < n; ++i) fn(i); return; }
    std::atomic<size_t> next{0};
    std::vector<std::thread> ts;
    ts.reserve(nt);
    for (size_t t = 0; t < nt; ++t)
        ts.emplace_back([&]() { for (size_t i; (i = next.fetch_add(1)) < n;) fn(i); });
    for (auto& th : ts) th.join();
}

namespace detail {

inline const char* line_end(const char* q, const char* end) {
    const void* e = std::memchr(q, '\n', static_cast<size_t>(end - q));
    return e ? static_cast<const char*>(e) : end;
}

// start of the first record at or after byte `off` (a line starting with '@' whose third line starts with '+'; no line of the layout
// can pass this test from another phase because a base line never starts with '@' or '+'); n when there is none
inline size_t align_record_start(const char* p, size_t n, size_t off) {
    if (off == 0) return 0;
    if (off >= n) return n;
    const char* end = p + n;
    const char* q = p + off;
    if (q[-1] != '\n') {
        const char* e = line_end(q, end);
        if (e >= end) return n;
        q = e + 1;
    }
    for (int k = 0; k < 4; ++k) {
        if (q >= end) return n;
        const char* e0 = line_end(q, end);
        if (e0 >= end) return n;
        const char* l1 = e0 + 1;
        if (l1 >= end) return n;
        const char* e1 = line_end(l1, end);
        if (e1 >= end) return n;
        const char* l2 = e1 + 1;
        if (l2 >= end) return n;
        if (*q == '@' && *l2 == '+') return static_cast<size_t>(q - p);
        q = l1;
    }
    return n;
}

// chunk c of the file starts at the first record at or after byte c * chunk
struct Chunks {
    std::vector<size_t> b;   // boundaries, b.size() = chunks + 1
    size_t count() const { return b.size() - 1; }
};

inline Chunks make_chunks(const char* p, size_t n, int threads) {
    size_t chunk = n / (static_cast<size_t>(std::max(1, threads)) * 8);
    chunk = std::min<size_t>(std::max<size_t>(chunk, size_t(1) << 20), size_t(32) << 20);
    const size_t nc = std::max<size_t>(1, (n + chunk - 1) / chunk);
    Chunks c;
    c.b.assign(nc + 1, n);
    c.b[0] = 0;
    for (size_t i = 1; i < nc; ++i) c.b[i] = align_record_start(p, n, i * chunk);
    for (size_t i = 1; i < nc; ++i) c.b[i] = std::max(c.b[i], c.b[i - 1]);   // a chunk without a record start is empty
    c.b[nc] = n;
    return c;
}

// One record of [q, end): the four lines [q0,e0) [q1,e1) [q2,e2) [q3,e3), next record start in `next`.
// 1 = a record, 0 = fewer than four lines left (the serial reader stops there), -1 = not the layout
struct Rec { const char *q0, *e0, *q1, *e1, *q3, *e3, *next; };
inline int next_record(const char* q, const char* end, Rec& r) {
    if (q >= end) return 0;
    r.q0 = q;
    r.e0 = line_end(q, end);
    r.q1 = r.e0 + 1;
    if (r.q1 >= end) return 0;
    r.e1 = line_end(r.q1, end);
    const char* q2 = r.e1 + 1;
    if (q2 >= end) return 0;
    const char* e2 = line_end(q2, end);
    r.q3 = e2 + 1;
    if (r.q3 >= end) return 0;
    r.e3 = line_end(r.q3, end);
    r.next = r.e3 + 1;
    if (*r.q0 != '@' || *q2 != '+') return -1;
    return 1;
}

}  // namespace detail

// The index of the selected records of mate1, in file order
struct FastIndex {
    std::shared_ptr<MappedFile> file;
    std::vector<uint64_t> off;   // start of the record
    std::vector<uint32_t> len;   // bytes of the four lines without the final newline
    std::vector<uint64_t> rec;   // record number among all records of the file
    size_t size() const { return off.size(); }
};

// One parallel pass over mate1: the same SampledReads as sample_reads(fq, cap). False: not the regular layout (nothing is returned).
inline bool fast_sample(const std::string& path, size_t cap, int threads, SampledReads& smp, FastIndex& fi) {
    auto file = std::make_shared<MappedFile>();
    if (!file->open_file(path)) return false;
    const char* p = file->data();
    const size_t n = file->size();
    smp = SampledReads{};
    fi = FastIndex{};
    fi.file = file;
    if (n == 0) return true;
    const detail::Chunks ch = detail::make_chunks(p, n, threads);
    const size_t nc = ch.count();

    struct Kept { uint64_t off; uint32_t name_len, seq_len, rec_len, local; int8_t level; };
    struct ChunkOut { std::vector<Kept> kept; uint64_t n_rec = 0; bool ok = true; };
    std::vector<ChunkOut> outs(nc);
    std::atomic<uint64_t> gl[64];
    for (auto& g : gl) g.store(0);
    std::atomic<bool> bad{false};

    parallel_for(nc, threads, [&](size_t c) {
        if (bad.load(std::memory_order_relaxed)) return;
        ChunkOut& o = outs[c];
        const char* q = p + ch.b[c];
        const char* end = p + ch.b[c + 1];
        // the level every record must reach to be kept for now: the first j whose count so far is at most cap (a lower bound of the
        // final level, since counts only grow)
        int jcur = 0;
        {
            uint64_t s = 0;
            for (int j = 63; j >= 0; --j) { s += gl[j].load(std::memory_order_relaxed); if (s > cap) { jcur = j + 1; break; } }
        }
        uint64_t loc[64] = {0};
        detail::Rec r;
        uint32_t local = 0;
        for (;;) {
            const int st = detail::next_record(q, end, r);
            if (st < 0) { o.ok = false; break; }
            if (st == 0) break;
            q = r.next;
            if (r.e1 != r.q1) {   // a sequence
                // read_name_hash: FNV-1a of the name line after the '@' up to the first space or tab, then the 64-bit mix
                const char* ne = r.q0;
                while (ne < r.e0 && *ne != ' ' && *ne != '\t') ++ne;
                uint64_t h = 14695981039346656037ULL;
                for (const char* a = r.q0 + 1; a < ne; ++a) { h ^= static_cast<unsigned char>(*a); h *= 1099511628211ULL; }
                h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
                if (h != UINT64_MAX) {
                    const int lvl = __builtin_clzll(h + 1);
                    ++loc[lvl];
                    if (lvl >= jcur)
                        o.kept.push_back({static_cast<uint64_t>(r.q0 - p), static_cast<uint32_t>(r.e0 - r.q0), static_cast<uint32_t>(r.e1 - r.q1),
                                          static_cast<uint32_t>(r.e3 - r.q0), local, static_cast<int8_t>(lvl)});
                }
            }
            ++local;
        }
        o.n_rec = local;
        if (o.ok && q != end && c + 1 < nc) o.ok = false;   // a chunk must end on the next chunk's first record
        if (!o.ok) { bad.store(true); return; }
        for (int j = 0; j < 64; ++j) if (loc[j]) gl[j].fetch_add(loc[j], std::memory_order_relaxed);
    });
    if (bad.load()) return false;

    // final level: the first j with at most cap records at level >= j (j = 64: none)
    int jf = 64;
    {
        uint64_t s = 0;
        jf = 0;
        for (int j = 63; j >= 0; --j) { s += gl[j].load(); if (s > cap) { jf = j + 1; break; } }
    }
    smp.threshold = jf >= 64 ? 0 : (UINT64_MAX >> jf);
    smp.subsampled = jf > 0;

    std::vector<size_t> sel_pos(nc + 1, 0);       // selected records before chunk c
    std::vector<uint64_t> rec_base(nc + 1, 0);    // records before chunk c
    for (size_t c = 0; c < nc; ++c) {
        size_t k = 0;
        for (const auto& kp : outs[c].kept) if (kp.level >= jf) ++k;
        sel_pos[c + 1] = sel_pos[c] + k;
        rec_base[c + 1] = rec_base[c] + outs[c].n_rec;
    }
    const size_t ns = sel_pos[nc];
    smp.reads.resize(ns);
    fi.off.resize(ns);
    fi.len.resize(ns);
    fi.rec.resize(ns);
    parallel_for(nc, threads, [&](size_t c) {
        size_t pos = sel_pos[c];
        for (const auto& kp : outs[c].kept) {
            if (kp.level < jf) continue;
            smp.reads[pos].assign(p + kp.off + kp.name_len + 1, kp.seq_len);
            fi.off[pos] = kp.off;
            fi.len[pos] = kp.rec_len;
            fi.rec[pos] = rec_base[c] + kp.local;
            ++pos;
        }
        std::vector<Kept>().swap(outs[c].kept);
    });
    return true;
}

// The selected records i with keep(i), as write_classified() writes them (each record's four lines and a newline). False: nothing written.
template <class Keep>
inline bool fast_write_classified(const FastIndex& fi, Keep keep, int threads, const std::string& out_path, size_t* written = nullptr) {
    const char* p = fi.file ? fi.file->data() : nullptr;
    const size_t ns = fi.size();
    constexpr size_t kSeg = 1 << 16;
    const size_t nseg = (ns + kSeg - 1) / kSeg;
    std::vector<std::string> bufs(nseg);
    std::vector<size_t> cnt(nseg, 0);
    parallel_for(nseg, threads, [&](size_t s) {
        std::string& b = bufs[s];
        const size_t i1 = std::min(ns, (s + 1) * kSeg);
        for (size_t i = s * kSeg; i < i1; ++i) {
            if (!keep(i)) continue;
            b.append(p + fi.off[i], fi.len[i]);
            b.push_back('\n');
            ++cnt[s];
        }
    });
    const int fd = ::open(out_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;
    std::vector<uint64_t> at(nseg + 1, 0);
    for (size_t s = 0; s < nseg; ++s) at[s + 1] = at[s] + bufs[s].size();
    std::atomic<bool> ok{true};
    parallel_for(nseg, std::min(threads, 8), [&](size_t s) {
        const std::string& b = bufs[s];
        size_t done = 0;
        while (done < b.size()) {
            const ssize_t w = ::pwrite(fd, b.data() + done, b.size() - done, static_cast<off_t>(at[s] + done));
            if (w <= 0) { ok.store(false); return; }
            done += static_cast<size_t>(w);
        }
    });
    ::close(fd);
    if (written) { *written = 0; for (size_t c : cnt) *written += c; }
    return ok.load();
}

// The TaggedRead vector of NonHostCellMatrix::read_mate_pairs for the selected records of mate1: seq from `reads`, cb and umi from record
// fi.rec[i] of mate2 (empty when mate2 has no such record or the read is too short). False: not the regular layout.
inline bool fast_tag_reads(const std::string& mate2_path, const FastIndex& fi, const std::vector<std::string>& reads, int cb_len, int umi_len,
                           int threads, std::vector<TaggedRead>& out) {
    const size_t ns = fi.size();
    out.assign(ns, TaggedRead{});
    parallel_for((ns + 65535) / 65536, threads, [&](size_t s) {
        for (size_t i = s * 65536; i < std::min(ns, (s + 1) * 65536); ++i) out[i].seq = reads[i];
    });
    MappedFile f2;
    if (!f2.open_file(mate2_path)) return false;
    const char* p = f2.data();
    const size_t n = f2.size();
    if (n == 0 || ns == 0) return true;
    const detail::Chunks ch = detail::make_chunks(p, n, threads);
    const size_t nc = ch.count();
    // records per chunk
    std::vector<uint64_t> nrec(nc, 0);
    std::atomic<bool> bad{false};
    parallel_for(nc, threads, [&](size_t c) {
        if (bad.load(std::memory_order_relaxed)) return;
        const char* q = p + ch.b[c];
        const char* end = p + ch.b[c + 1];
        detail::Rec r;
        uint64_t k = 0;
        for (;;) {
            const int st = detail::next_record(q, end, r);
            if (st < 0) { bad.store(true); return; }
            if (st == 0) break;
            q = r.next;
            ++k;
        }
        if (q != end && c + 1 < nc) { bad.store(true); return; }
        nrec[c] = k;
    });
    if (bad.load()) return false;
    std::vector<uint64_t> base(nc + 1, 0);
    for (size_t c = 0; c < nc; ++c) base[c + 1] = base[c] + nrec[c];
    parallel_for(nc, threads, [&](size_t c) {
        size_t s = std::lower_bound(fi.rec.begin(), fi.rec.end(), base[c]) - fi.rec.begin();
        if (s >= ns || fi.rec[s] >= base[c + 1]) return;
        const char* q = p + ch.b[c];
        const char* end = p + ch.b[c + 1];
        detail::Rec r;
        for (uint64_t g = base[c]; g < base[c + 1] && s < ns; ++g) {
            if (detail::next_record(q, end, r) != 1) break;
            q = r.next;
            if (fi.rec[s] != g) continue;
            const size_t len = static_cast<size_t>(r.e1 - r.q1);
            TaggedRead& tr = out[s];
            if (cb_len > 0 && static_cast<int>(len) >= cb_len) tr.cb.assign(r.q1, static_cast<size_t>(cb_len));
            if (umi_len > 0 && static_cast<int>(len) >= cb_len + umi_len) tr.umi.assign(r.q1 + cb_len, static_cast<size_t>(umi_len));
            ++s;
            if (s < ns && fi.rec[s] >= base[c + 1]) break;
        }
    });
    return true;
}

}  // namespace fastscan
}  // namespace nonhost
}  // namespace singlet
