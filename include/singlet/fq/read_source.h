// SPDX-License-Identifier: MIT
// read_source.h — one decoder for every consumer of a sample's .1fq reads
//
// The pipeline used to decode a .1fq in three places in src/pipeline/singlet.cpp (file mode, FIFO pass 1, FIFO
// pass 2), each with its own copy of the block read, the per-read selection and the FASTQ formatting. This header
// is that logic once:
//
//   RunSet           one or more .1fq files of one sample (runs), read as one stream: blocks are numbered across the
//                    files and read offsets continue, so several runs are pooled without decoding and re-encoding.
//   RunSet::decode   pread + decompress + decode_payload of one block (thread-safe).
//   format_block     the per-read selection and FASTQ text of one decoded block (R1 = barcode read, R2 = cDNA):
//                    reads with an empty mate are dropped, duplicates expanded, the inline 5' clip and the 1,024-base
//                    guard applied when asked, the ATAC barcode prefix and the CB_UMI_Complex concat rewrite.
//                    Names are "@[BC:<bc>_]r<offset>"; R2's quality line is the stored quality (Phred + 33) or 'F'.
//   OrderedBlocks    a pool of worker threads producing one value per block, consumed in block order with a bounded
//                    look-ahead.
//   StarChunkSource  StarReadSource (singlet/star/star_api.h) over the above: decodes and formats in worker threads
//                    inside STAR's forked process and copies the records into STAR's chunk buffers with the read
//                    numbering STAR adds itself. No FASTQ text file, no FIFO, no second process.
//
// The GPU engine's prefill can read the same stream: RunSet + OrderedBlocks<FormattedBlock> (or <DecodedBlock>)
// give every read in the order STAR numbers them (iReadAll - 1 = position in the stream of formatted records).

#pragma once

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "compress.h"
#include "reader.h"
#include "types.h"

namespace singlet::fq {

// ── RunSet: the runs of one sample as one stream of blocks ──

class RunSet {
   public:
    void open(const std::vector<std::string>& paths) {
        if (paths.empty()) throw std::runtime_error("RunSet: no .1fq file");
        paths_ = paths;
        readers_.clear();
        first_block_.clear();
        read_offsets_.clear();
        uint32_t nb = 0;
        uint64_t cum = 0;
        for (size_t f = 0; f < paths.size(); ++f) {
            auto r = std::make_unique<Reader>();
            r->open(paths[f].c_str());
            r->set_verify_crc(false);
            if (f > 0) check_compatible(*readers_[0], *r, paths[f]);
            first_block_.push_back(nb);
            for (uint32_t b = 0; b < r->block_count(); ++b) {
                BlockHeader bh;
                if (::pread(r->file_descriptor(), &bh, sizeof(bh), static_cast<off_t>(r->block_offset(b))) !=
                    static_cast<ssize_t>(sizeof(bh)))
                    throw std::runtime_error("RunSet: cannot read block header in " + paths[f]);
                read_offsets_.push_back(cum);
                cum += bh.n_reads;
            }
            nb += r->block_count();
            readers_.push_back(std::move(r));
        }
        n_blocks_ = nb;
        n_stored_ = cum;
    }

    size_t n_files() const { return readers_.size(); }
    const std::string& path(size_t f) const { return paths_[f]; }
    const Reader& reader(size_t f = 0) const { return *readers_[f]; }
    const Header& header() const { return readers_[0]->header(); }
    uint32_t block_count() const { return n_blocks_; }
    uint64_t stored_reads() const { return n_stored_; }   // records stored (before duplicate expansion)
    // Offset of block g's first stored read in the whole stream: the "r<offset>" read names count stored reads.
    uint64_t block_read_offset(uint32_t g) const { return read_offsets_[g]; }

    bool has_bc_dict() const { return readers_[0]->has_bc_dict(); }
    uint32_t bc_dict_size() const { return readers_[0]->bc_dict_size(); }
    uint16_t bc_length() const { return readers_[0]->bc_length(); }
    const std::vector<uint8_t>& bc_dict_flat() const { return readers_[0]->bc_dict_flat(); }

    // Decode block g. Thread-safe (pread on each file, const reader state).
    void decode(uint32_t g, DecodedBlock& blk) const {
        size_t f = static_cast<size_t>(std::upper_bound(first_block_.begin(), first_block_.end(), g) -
                                       first_block_.begin()) - 1;
        const Reader& r = *readers_[f];
        const uint32_t bi = g - first_block_[f];
        const int fd = r.file_descriptor();
        const uint64_t off = r.block_offset(bi);
        const uint32_t comp_size = r.block_comp_size(bi);
        BlockHeader bh;
        if (::pread(fd, &bh, sizeof(bh), static_cast<off_t>(off)) != static_cast<ssize_t>(sizeof(bh)))
            throw std::runtime_error("RunSet: short read of a block header");
        std::vector<uint8_t> comp(comp_size);
        if (::pread(fd, comp.data(), comp_size, static_cast<off_t>(off + sizeof(bh))) !=
            static_cast<ssize_t>(comp_size))
            throw std::runtime_error("RunSet: short read of a block");
        const auto codec = static_cast<Codec>(r.header().codec);
        size_t est_raw = static_cast<size_t>(bh.n_reads) * 200;
        std::vector<uint8_t> raw(est_raw);
        size_t raw_size;
        try {
            raw_size = compress::decompress_block(codec, comp.data(), comp_size, raw.data(), raw.size());
        } catch (...) {
            raw.resize(est_raw * 4);
            raw_size = compress::decompress_block(codec, comp.data(), comp_size, raw.data(), raw.size());
        }
        blk.clear();
        r.decode_payload(raw.data(), raw_size, bh, blk);
    }

   private:
    // Pooled runs must decode to the same read layout and the same barcode dictionary (dictionary indices are summed
    // across runs by pass 1). Quality may differ: a run without stored quality gets 'F'.
    static void check_compatible(const Reader& a, const Reader& b, const std::string& path) {
        const Header& x = a.header();
        const Header& y = b.header();
        std::string why;
        if (x.protocol_id != y.protocol_id) why = "protocol";
        else if (x.assay_type != y.assay_type) why = "assay type";
        else if (x.stream_lengths[0] != y.stream_lengths[0]) why = "R1 length";
        else if ((x.flags & Flags::BC_DICT) != (y.flags & Flags::BC_DICT)) why = "barcode dictionary flag";
        else if (x.reserved[0] != y.reserved[0]) why = "extra streams";
        else if (a.bc_length() != b.bc_length() || a.bc_dict_flat() != b.bc_dict_flat()) why = "barcode dictionary";
        if (!why.empty())
            throw std::runtime_error("RunSet: " + path + " differs from " + "the first run in its " + why +
                                     "; these runs cannot be pooled as one stream");
    }

    std::vector<std::string> paths_;
    std::vector<std::unique_ptr<Reader>> readers_;
    std::vector<uint32_t> first_block_;
    std::vector<uint64_t> read_offsets_;
    uint32_t n_blocks_ = 0;
    uint64_t n_stored_ = 0;
};

// ── FASTQ formatting of one decoded block ──

struct ConcatSeg {
    int start = 0;
    int len = 0;
};

struct FormatOptions {
    int clip5p = 0;                 // bases clipped from R2's 5' end inline (0 = none; STAR may clip instead)
    uint16_t max_r2 = 0;            // reads with a longer R2 are dropped (0 = no guard); their offsets are still used
    bool atac = false;              // "@BC:<barcode>_r<n>" names (I2 stream > dictionary > 16 N)
    bool has_i2 = false;
    std::vector<uint8_t> atac_dict;  // flat dictionary for the ATAC name prefix (when there is no I2 stream)
    uint16_t atac_bc_len = 0;
    uint32_t bc_dict_n = 0;
    std::vector<ConcatSeg> concat;  // CB_UMI_Complex -> CB_samTagOut: R1 rewritten as [seg0]..[segN][UMI]
    int concat_umi_start = 0;
    int concat_umi_len = 0;
};

// Formatted records of one block. Record k of R1 is r1[r1_off[k] .. r1_off[k+1]), likewise R2; both mates of a
// read have the same name, name_len[k] bytes from the record start ("@...r<n>", without the newline).
struct FormattedBlock {
    std::vector<char> r1, r2;
    std::vector<uint32_t> r1_off, r2_off;   // n_records + 1 entries
    std::vector<uint16_t> name_len;
    uint32_t n_stored = 0;                  // stored reads in the block (before selection and expansion)
    std::vector<uint32_t> bc_indices;       // the decoded block's dictionary indices (pass-1 style counting)
    size_t n_records() const { return name_len.size(); }
};

namespace detail {
static constexpr char kNum2Ascii[5] = {'A', 'C', 'G', 'T', 'N'};

inline char* put_u64(char* p, uint64_t v) {
    char tmp[24];
    int n = 0;
    do {
        tmp[n++] = static_cast<char>('0' + v % 10);
        v /= 10;
    } while (v);
    while (n) *p++ = tmp[--n];
    return p;
}
}  // namespace detail

// Format every selected read of blk (whose first stored read is number read_off in the stream).
inline void format_block(const DecodedBlock& blk, uint64_t read_off, const FormatOptions& o, FormattedBlock& out) {
    using detail::kNum2Ascii;
    out.r1.clear();
    out.r2.clear();
    out.r1_off.clear();
    out.r2_off.clear();
    out.name_len.clear();
    out.n_stored = blk.n_reads;
    const bool has_dups = !blk.dup_counts.empty();
    const bool has_qual = !blk.r2_qual.empty();
    const int clip = o.clip5p > 0 ? o.clip5p : 0;
    uint16_t r1_write_fixed = 0;
    if (!o.concat.empty()) {
        for (const auto& s : o.concat) r1_write_fixed += static_cast<uint16_t>(s.len);
        r1_write_fixed += static_cast<uint16_t>(o.concat_umi_len);
    }

    // Exact size, from the same selection as the write loop below.
    size_t r1_bytes = 0, r2_bytes = 0, n_rec = 0;
    for (uint32_t i = 0; i < blk.n_reads; ++i) {
        const uint16_t r1len = blk.r1_len(i), r2len = blk.r2_len(i);
        if (r1len == 0 || r2len == 0 || (o.max_r2 && r2len > o.max_r2)) continue;
        const uint32_t rpt = has_dups ? blk.dup_counts[i] : 1;
        const uint16_t r2w = (clip > 0 && clip < r2len) ? static_cast<uint16_t>(r2len - clip) : r2len;
        const uint16_t r1w = o.concat.empty() ? r1len : r1_write_fixed;
        r1_bytes += static_cast<size_t>(rpt) * (2u * r1w + 64);
        r2_bytes += static_cast<size_t>(rpt) * (2u * r2w + 64);
        n_rec += rpt;
    }
    out.r1.resize(r1_bytes + 64);
    out.r2.resize(r2_bytes + 64);
    out.r1_off.reserve(n_rec + 1);
    out.r2_off.reserve(n_rec + 1);
    out.name_len.reserve(n_rec);
    char* r1p = out.r1.data();
    char* r2p = out.r2.data();

    uint64_t roff = read_off;
    for (uint32_t i = 0; i < blk.n_reads; ++i) {
        const uint32_t repeat = has_dups ? blk.dup_counts[i] : 1;
        const uint16_t r1len = blk.r1_len(i), r2len = blk.r2_len(i);
        // An empty mate cannot be loaded by STAR ("short read sequence line"); the FIFO writer skipped such reads
        // without advancing the read number (block offsets count them, so the next block's numbers are unchanged).
        if (r1len == 0 || r2len == 0) continue;
        // R2 longer than max_r2 is a corrupted record (real Illumina R2 <= 600 bp): dropped, its numbers used.
        if (o.max_r2 && r2len > o.max_r2) {
            roff += repeat;
            continue;
        }
        const uint8_t* r1s = blk.r1_seq(i);
        const uint8_t* r2s = blk.r2_seq(i);
        const uint8_t* q = has_qual ? blk.r2_quality(i) : nullptr;
        const uint16_t skip = (clip > 0 && static_cast<uint16_t>(clip) < r2len) ? static_cast<uint16_t>(clip) : 0;

        char prefix[40];
        int np = 0;
        if (o.atac) {
            bool set = false;
            if (o.has_i2 && !blk.i2_data.empty() && i < blk.i2_lengths.size() && blk.i2_lengths[i] > 0) {
                const uint8_t* bcp = blk.i2_seq(i);
                const uint16_t blen = std::min<uint16_t>(blk.i2_len(i), 32);
                prefix[np++] = 'B'; prefix[np++] = 'C'; prefix[np++] = ':';
                for (uint16_t j = 0; j < blen; ++j) prefix[np++] = kNum2Ascii[bcp[j]];
                prefix[np++] = '_';
                set = true;
            }
            if (!set) {
                const uint32_t idx = i < blk.bc_indices.size() ? blk.bc_indices[i] : UINT32_MAX;
                if (!o.atac_dict.empty() && o.atac_bc_len > 0 && idx < o.bc_dict_n) {
                    const uint8_t* bcp = o.atac_dict.data() + static_cast<size_t>(idx) * o.atac_bc_len;
                    prefix[np++] = 'B'; prefix[np++] = 'C'; prefix[np++] = ':';
                    for (uint16_t j = 0; j < std::min<uint16_t>(o.atac_bc_len, 32); ++j)
                        prefix[np++] = kNum2Ascii[bcp[j]];
                    prefix[np++] = '_';
                    set = true;
                }
            }
            if (!set) {
                std::memcpy(prefix, "BC:NNNNNNNNNNNNNNNN_", 20);
                np = 20;
            }
        }

        for (uint32_t rep = 0; rep < repeat; ++rep) {
            // name
            char* name = r1p;
            *r1p++ = '@';
            std::memcpy(r1p, prefix, np);
            r1p += np;
            *r1p++ = 'r';
            r1p = detail::put_u64(r1p, roff);
            const uint16_t nl = static_cast<uint16_t>(r1p - name);
            out.r1_off.push_back(static_cast<uint32_t>(name - out.r1.data()));
            out.name_len.push_back(nl);
            *r1p++ = '\n';
            // R1 (barcode read)
            if (!o.concat.empty()) {
                int w = 0;
                for (const auto& seg : o.concat)
                    for (int j = 0; j < seg.len; ++j) {
                        const int src = seg.start + j;
                        r1p[w++] = (src < static_cast<int>(r1len)) ? kNum2Ascii[r1s[src]] : 'N';
                    }
                for (int j = 0; j < o.concat_umi_len; ++j) {
                    const int src = o.concat_umi_start + j;
                    r1p[w++] = (src < static_cast<int>(r1len)) ? kNum2Ascii[r1s[src]] : 'N';
                }
                r1p += w;
                *r1p++ = '\n'; *r1p++ = '+'; *r1p++ = '\n';
                std::memset(r1p, 'F', w);
                r1p += w;
            } else {
                for (uint16_t j = 0; j < r1len; ++j) r1p[j] = kNum2Ascii[r1s[j]];
                r1p += r1len;
                *r1p++ = '\n'; *r1p++ = '+'; *r1p++ = '\n';
                std::memset(r1p, 'F', r1len);
                r1p += r1len;
            }
            *r1p++ = '\n';

            // R2 (cDNA), same name
            out.r2_off.push_back(static_cast<uint32_t>(r2p - out.r2.data()));
            std::memcpy(r2p, name, nl);
            r2p += nl;
            *r2p++ = '\n';
            const uint16_t len = r2len - skip;
            const uint8_t* s = r2s + skip;
            for (uint16_t j = 0; j < len; ++j) r2p[j] = kNum2Ascii[s[j]];
            r2p += len;
            *r2p++ = '\n'; *r2p++ = '+'; *r2p++ = '\n';
            if (q) {
                const uint8_t* qq = q + skip;
                for (uint16_t j = 0; j < len; ++j) r2p[j] = static_cast<char>(qq[j] + 33);
            } else {
                std::memset(r2p, 'F', len);
            }
            r2p += len;
            *r2p++ = '\n';
            ++roff;
        }
    }
    out.r1_off.push_back(static_cast<uint32_t>(r1p - out.r1.data()));
    out.r2_off.push_back(static_cast<uint32_t>(r2p - out.r2.data()));
    out.r1.resize(static_cast<size_t>(r1p - out.r1.data()));
    out.r2.resize(static_cast<size_t>(r2p - out.r2.data()));
}

// ── OrderedBlocks: parallel production, in-order consumption ──

template <class T>
class OrderedBlocks {
   public:
    using Work = std::function<void(uint32_t block, T& out)>;

    // Starts `threads` workers over blocks [0, n_blocks). At most `depth` blocks are produced ahead of the consumer.
    OrderedBlocks(uint32_t n_blocks, int threads, uint32_t depth, Work work)
        : n_(n_blocks), depth_(std::max<uint32_t>(depth, 1)), work_(std::move(work)) {
        const int nt = std::max(1, std::min<int>(threads, static_cast<int>(std::max<uint32_t>(n_blocks, 1))));
        for (int t = 0; t < nt; ++t) pool_.emplace_back([this] { run(); });
    }
    ~OrderedBlocks() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& th : pool_) th.join();
    }
    OrderedBlocks(const OrderedBlocks&) = delete;
    OrderedBlocks& operator=(const OrderedBlocks&) = delete;

    // The next block in order; false after the last one. Rethrows a worker's exception.
    bool next(T& out, uint32_t* block = nullptr) {
        std::unique_lock<std::mutex> lk(mu_);
        if (consumed_ >= n_) return false;
        cv_.wait(lk, [&] { return ready_.count(consumed_) || error_; });
        if (error_) std::rethrow_exception(error_);
        auto it = ready_.find(consumed_);
        out = std::move(it->second);
        ready_.erase(it);
        if (block) *block = consumed_;
        ++consumed_;
        lk.unlock();
        cv_.notify_all();
        return true;
    }

   private:
    void run() {
        for (;;) {
            uint32_t b;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || error_ || next_ >= n_ || next_ < consumed_ + depth_; });
                if (stop_ || error_ || next_ >= n_) return;
                b = next_++;
            }
            T val;
            try {
                work_(b, val);
            } catch (...) {
                std::lock_guard<std::mutex> lk(mu_);
                if (!error_) error_ = std::current_exception();
                cv_.notify_all();
                return;
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                ready_.emplace(b, std::move(val));
            }
            cv_.notify_all();
        }
    }

    const uint32_t n_;
    const uint32_t depth_;
    Work work_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::map<uint32_t, T> ready_;
    uint32_t next_ = 0, consumed_ = 0;
    bool stop_ = false;
    std::exception_ptr error_;
    std::vector<std::thread> pool_;
};

// ── StarChunkSource: the reads, decoded inside STAR's process, into its chunk buffers ──
//
// Mate order follows --readFilesIn: mate 0 is the cDNA read (R2) and mate 1 the barcode read (R1) for scRNA
// (cdna_first = true); ATAC passes R1 then R2. The bytes per record are those STAR's FASTQ branch builds from a file
// of the same records (star_api.h), so mapping, chunk boundaries and outputs are the same as with the FIFO.

template <class StarSourceBase>
class StarChunkSourceT : public StarSourceBase {
   public:
    StarChunkSourceT(std::vector<std::string> paths, FormatOptions opts, int threads, bool cdna_first)
        : paths_(std::move(paths)), opts_(std::move(opts)), threads_(threads), cdna_first_(cdna_first) {}

    // Called with STAR's mutexInRead held (one caller at a time).
    void fill(char** chunkIn, uint64_t* sizeBytes, uint64_t sizeLimit, uint64_t& iReadAll, uint64_t readMapNumber,
              uint32_t readFilesIndex) override {
        if (!started_) start();
        char suffix[48];
        while (sizeBytes[0] < sizeLimit && sizeBytes[1] < sizeLimit) {
            if (iReadAll == readMapNumber) break;
            if (k_ >= cur_.n_records()) {
                if (!pipe_ || !pipe_->next(cur_)) {
                    done_ = true;
                    break;
                }
                k_ = 0;
                continue;
            }
            ++iReadAll;
            char* p = suffix;
            *p++ = ' ';
            p = detail::put_u64(p, iReadAll);
            *p++ = ' '; *p++ = 'N'; *p++ = ' ';
            p = detail::put_u64(p, readFilesIndex);
            const size_t sl = static_cast<size_t>(p - suffix);
            const uint16_t nl = cur_.name_len[k_];
            for (int m = 0; m < 2; ++m) {
                const bool cdna = (m == 0) == cdna_first_;
                const std::vector<char>& buf = cdna ? cur_.r2 : cur_.r1;
                const std::vector<uint32_t>& off = cdna ? cur_.r2_off : cur_.r1_off;
                const char* rec = buf.data() + off[k_];
                const size_t rest = off[k_ + 1] - off[k_] - nl;   // "\n<seq>\n+\n<qual>\n"
                char* dst = chunkIn[m] + sizeBytes[m];
                std::memcpy(dst, rec, nl);
                std::memcpy(dst + nl, suffix, sl);
                std::memcpy(dst + nl + sl, rec + nl, rest);
                sizeBytes[m] += nl + sl + rest;
            }
            ++k_;
            ++reads_;
        }
    }

    uint64_t reads_delivered() const { return reads_; }

   private:
    void start() {
        started_ = true;
        runs_.open(paths_);
        const uint32_t workers = static_cast<uint32_t>(std::max(1, threads_));
        pipe_ = std::make_unique<OrderedBlocks<FormattedBlock>>(
            runs_.block_count(), threads_, 2 * workers + 2, [this](uint32_t g, FormattedBlock& fb) {
                DecodedBlock blk;
                runs_.decode(g, blk);
                format_block(blk, runs_.block_read_offset(g), opts_, fb);
            });
    }

    std::vector<std::string> paths_;
    FormatOptions opts_;
    int threads_;
    bool cdna_first_;
    RunSet runs_;
    std::unique_ptr<OrderedBlocks<FormattedBlock>> pipe_;
    FormattedBlock cur_;
    size_t k_ = 0;
    bool started_ = false, done_ = false;
    uint64_t reads_ = 0;
};

}  // namespace singlet::fq
