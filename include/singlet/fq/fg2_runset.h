// SPDX-License-Identifier: MIT
// fg2_runset.h — Fg2RunSet: the staged .1fg2 (fg2_format.h) read as a stream of DecodedBlocks, with RunSet's interface (read_source.h)
//
// A drop-in source for the consumers that read a sample through RunSet::decode(g, DecodedBlock&): block g is chunk g of the .1fg2 (32,768 reads), and the
// DecodedBlock it yields is, read for read, the one the .1fq RunSet yields for the same reads (gated by test_fg2_runset on the fixtures):
//   r1_data     barcode as SEQUENCED (an exact whitelist hit: the whitelist entry; any other read: its PL_RAWCB entry with the N bases as 4) followed by the UMI
//               (N from the exceptions); r1 length = bc_len + umi_len (protocols with the barcode at 0 and the UMI right behind it; others are refused at open)
//   r2_data     the read's cDNA from the distinct-sequence table (N from the table's N lists), r2_lengths = the read's length
//   r2_qual     the 2-bit quality bins mapped to the decoded Phred values of the header (ILLUMINA4: 2, 11, 25, 37); empty when the file has no q plane
//   bc_indices  empty and has_bc_dict() is false: no dictionary (consumers that count dictionary indices take the sequenced barcode instead)
// plus accessors for the table-fed prefill: the distinct-sequence table in memory (table()), the per-chunk planes (decode_planes: corrected barcode index,
// status, UMI word, sequence id, raw-barcode entries), the whitelist and the run table.
//
// Reads are in the order of the file (source order; barcode-sorted if the .1fg2 was built from a barcode-sorted .1fq, SRA order if from `singlet download --stage-out`).
// decode() and decode_planes() are const and thread-safe (pread, no shared mutable state). Header-only; needs libzstd and SSE4.2 (CRC32C).
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <zstd.h>

#include "fg2_format.h"
#include "reader.h"
#include "types.h"

namespace singlet::fq {

class Fg2RunSet {
   public:
    // The distinct cDNA sequences of the sample, in memory: row id = global sequence id (what PL_SEQ holds).
    struct SeqTable {
        uint64_t n = 0;
        uint32_t slot = 0;                    // bytes per sequence in seq2 (8-byte multiple)
        std::vector<uint8_t> seq2;            // n * slot, 2 bit/base, first base in the high bits, N coded as A
        std::vector<uint8_t> slen;            // n, length in bases
        std::vector<uint64_t> n_first;        // n + 1: CSR offsets into n_pos
        std::vector<uint8_t> n_pos;           // positions of the N bases, per sequence ascending
        uint8_t base(uint64_t id, uint32_t j) const { return (seq2[id * slot + (j >> 2)] >> (6 - 2 * (j & 3))) & 3; }
        bool has_n(uint64_t id) const { return n_first[id + 1] != n_first[id]; }
    };

    // Planes of one chunk, inflated (the lists are the file's, not expanded).
    struct ChunkPlanes {
        uint32_t n = 0;
        uint64_t first_read = 0;
        std::vector<uint32_t> cb;       // corrected barcode: index in the sorted, de-duplicated whitelist, or fg2::NO_CB
        std::vector<uint8_t> cbst;      // fg2::CbStatus low 3 bits + fg2::CBST_BAD_UMI flag
        std::vector<uint32_t> umi;      // 2 bit/base, first base high, N coded as A
        std::vector<uint32_t> seq;      // sequence id
        std::vector<uint8_t> len;       // cDNA length
        std::vector<uint8_t> q;         // n * seq_slot bytes of 2-bit bins, empty without q plane
        std::vector<uint32_t> exc;      // fg2::exc_pack entries (UMI N)
        std::vector<uint32_t> rawcb;    // pairs {word, meta}: word = raw barcode 2 bit/base (N as A), meta = read index | (N-base bitmask << 15)
    };

    // paths: exactly one .1fg2 (a sample is pooled into one file). whitelist_dirs: where the file named in the header lives.
    void open(const std::vector<std::string>& paths, const std::vector<std::string>& whitelist_dirs, bool verify_crc = true) {
        if (paths.size() != 1) throw std::runtime_error("Fg2RunSet: exactly one .1fg2 file per sample");
        path_ = paths[0];
        verify_crc_ = verify_crc;
        fd_ = ::open(path_.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("Fg2RunSet: cannot open " + path_);
        pread_full(&h_, sizeof h_, 0);
        if (h_.magic != fg2::FILE_MAGIC || h_.version != fg2::VERSION) throw std::runtime_error("Fg2RunSet: not a .1fg2 v2 file: " + path_);
        if (h_.header_crc32c != fg2::crc32c(&h_, offsetof(fg2::FileHeader, header_crc32c))) throw std::runtime_error("Fg2RunSet: header CRC mismatch");
        if (h_.flags & fg2::F_INCOMPLETE) throw std::runtime_error("Fg2RunSet: incomplete file");
        if (h_.cb_start != 0 || h_.umi_start != h_.cb_len || h_.r1_len != h_.cb_len + h_.umi_len)
            throw std::runtime_error("Fg2RunSet: barcode/UMI layout not supported (barcode at 0 with the UMI right behind it)");
        has_q_ = (h_.flags & fg2::F_QPLANE) != 0;
        off_t sz = ::lseek(fd_, 0, SEEK_END);
        fg2::Footer ft;
        pread_full(&ft, sizeof ft, static_cast<uint64_t>(sz) - sizeof ft);
        if (ft.magic != fg2::FOOT_MAGIC) throw std::runtime_error("Fg2RunSet: bad footer (file incomplete?)");
        std::vector<fg2::IndexEntry> idx(ft.n_chunks + ft.n_blocks);
        pread_full(idx.data(), idx.size() * sizeof(fg2::IndexEntry), ft.index_offset);
        if (ft.index_crc32c != fg2::crc32c(idx.data(), idx.size() * sizeof(fg2::IndexEntry))) throw std::runtime_error("Fg2RunSet: index CRC mismatch");
        chunks_.assign(idx.begin(), idx.begin() + ft.n_chunks);
        blocks_.assign(idx.begin() + ft.n_chunks, idx.end());
        first_read_.assign(chunks_.size() + 1, 0);
        for (size_t c = 0; c < chunks_.size(); ++c) first_read_[c + 1] = first_read_[c] + chunks_[c].n;
        runs_.resize(h_.n_runs);
        if (h_.n_runs) pread_full(runs_.data(), runs_.size() * sizeof(fg2::RunEntry), h_.runs_offset);
        load_whitelist(whitelist_dirs);
        dict_flat_.resize(wl_.size() * h_.cb_len);
        for (size_t k = 0; k < wl_.size(); ++k)
            for (uint32_t j = 0; j < h_.cb_len; ++j) dict_flat_[k * h_.cb_len + j] = static_cast<uint8_t>((wl_[k] >> (2 * (h_.cb_len - 1 - j))) & 3);
        load_table();
        hdr_.init();
        hdr_.protocol_id = h_.protocol_id;
        hdr_.qual_mode = static_cast<uint8_t>(has_q_ ? QualMode::ILLUMINA4 : QualMode::NONE);
        hdr_.n_unique = h_.n_reads;
        hdr_.n_original = h_.n_reads_in;
        hdr_.block_count = static_cast<uint32_t>(chunks_.size());
        hdr_.block_size = fg2::CHUNK_READS;
        hdr_.stream_lengths[0] = h_.r1_len;
        hdr_.stream_lengths[1] = 0;   // variable
    }

    Fg2RunSet() = default;
    Fg2RunSet(const Fg2RunSet&) = delete;
    Fg2RunSet& operator=(const Fg2RunSet&) = delete;
    ~Fg2RunSet() { if (fd_ >= 0) ::close(fd_); }

    // ── RunSet's interface ──
    size_t n_files() const { return 1; }
    const std::string& path(size_t = 0) const { return path_; }
    const Header& header() const { return hdr_; }
    uint32_t block_count() const { return static_cast<uint32_t>(chunks_.size()); }
    uint64_t stored_reads() const { return h_.n_reads; }
    uint64_t block_read_offset(uint32_t g) const { return first_read_[g]; }
    // Dictionary mode, as a .1fq with a barcode dictionary: the dictionary is the whitelist (numeric bases, bc_length bytes per entry, PL_CB order);
    // bc_indices[i] = the whitelist index for an exact hit and bc_dict_size() for every other read, so pass-1 style counting is the .1fq path's.
    bool has_bc_dict() const { return true; }
    uint32_t bc_dict_size() const { return static_cast<uint32_t>(wl_.size()); }
    uint16_t bc_length() const { return h_.cb_len; }
    const std::vector<uint8_t>& bc_dict_flat() const { return dict_flat_; }

    // ── .1fg2 specifics ──
    const fg2::FileHeader& fg2_header() const { return h_; }
    // The source .1fq's metadata JSON, verbatim, with the encoder's own key ("s9_stage") appended (what Reader::read_metadata() gave at staging time).
    std::string metadata() const {
        std::vector<uint8_t> comp(h_.meta_comp_size);
        pread_full(comp.data(), comp.size(), h_.meta_offset);
        std::string out(h_.meta_raw_size, '\0');
        size_t r = ZSTD_decompress(&out[0], out.size(), comp.data(), comp.size());
        if (ZSTD_isError(r) || r != out.size()) throw std::runtime_error("Fg2RunSet: metadata block did not inflate");
        return out;
    }
    const SeqTable& table() const { return tab_; }
    const std::vector<uint64_t>& whitelist() const { return wl_; }          // 2-bit codes, sorted, unique: PL_CB indexes this
    const std::vector<fg2::RunEntry>& runs() const { return runs_; }       // accession, first_read, n_reads, raw_bytes per run (runs_used provenance)
    bool has_quality() const { return has_q_; }
    uint8_t phred_of_bin(uint32_t b) const { return h_.q_edges[b & 3]; }

    void decode_planes(uint32_t g, ChunkPlanes& cp) const {
        std::vector<uint8_t> stored(chunks_[g].total_bytes);
        pread_full(stored.data(), stored.size(), chunks_[g].offset);
        fg2::ChunkHeader ch;
        std::memcpy(&ch, stored.data(), sizeof ch);
        if (ch.magic != fg2::CHUNK_MAGIC) throw std::runtime_error("Fg2RunSet: bad chunk header");
        cp.n = ch.n_reads;
        cp.first_read = ch.first_read;
        std::vector<uint8_t> t;
        auto plane = [&](int p, auto& dst) {
            inflate(stored.data(), ch.sec[p], (ch.flags >> p) & 1, t);
            dst.resize(t.size() / sizeof(typename std::remove_reference<decltype(dst)>::type::value_type));
            if (!t.empty()) std::memcpy(dst.data(), t.data(), t.size());
        };
        plane(fg2::PL_CB, cp.cb); plane(fg2::PL_CBST, cp.cbst); plane(fg2::PL_UMI, cp.umi); plane(fg2::PL_SEQ, cp.seq);
        plane(fg2::PL_LEN, cp.len); plane(fg2::PL_Q, cp.q); plane(fg2::PL_EXC, cp.exc); plane(fg2::PL_RAWCB, cp.rawcb);
    }

    // Decode chunk g into a DecodedBlock. Thread-safe.
    void decode(uint32_t g, DecodedBlock& blk) const {
        ChunkPlanes cp;
        decode_planes(g, cp);
        blk.clear();
        const uint32_t n = cp.n, bcl = h_.cb_len, umil = h_.umi_len, r1l = bcl + umil, slot = h_.seq_slot;
        blk.n_reads = n;
        blk.r1_data.assign(static_cast<size_t>(n) * r1l, 0);
        blk.r1_offsets.resize(n); blk.r1_lengths.assign(n, static_cast<uint16_t>(r1l));
        blk.r2_offsets.resize(n); blk.r2_lengths.resize(n);
        blk.bc_indices.resize(n);
        size_t r2_total = 0;
        for (uint32_t i = 0; i < n; ++i) r2_total += cp.len[i];
        blk.r2_data.resize(r2_total);
        if (has_q_) blk.r2_qual.resize(r2_total);
        // UMI-N exceptions by read (sorted)
        size_t ei = 0, ri = 0;
        size_t r2o = 0;
        for (uint32_t i = 0; i < n; ++i) {
            uint8_t* r1 = blk.r1_data.data() + static_cast<size_t>(i) * r1l;
            blk.r1_offsets[i] = i * r1l;
            // barcode as sequenced
            if ((cp.cbst[i] & 7) == fg2::CBST_EXACT) {
                if (cp.cb[i] >= wl_.size()) throw std::runtime_error("Fg2RunSet: barcode index outside the whitelist");
                blk.bc_indices[i] = cp.cb[i];
                const uint64_t w = wl_[cp.cb[i]];
                for (uint32_t j = 0; j < bcl; ++j) r1[j] = static_cast<uint8_t>((w >> (2 * (bcl - 1 - j))) & 3);
            } else {
                blk.bc_indices[i] = static_cast<uint32_t>(wl_.size());   // not an exact whitelist hit: the dictionary-miss sentinel
                while (ri + 1 < cp.rawcb.size() && (cp.rawcb[ri + 1] & 0x7FFF) < i) ri += 2;
                if (ri + 1 >= cp.rawcb.size() || (cp.rawcb[ri + 1] & 0x7FFF) != i) throw std::runtime_error("Fg2RunSet: missing raw-barcode entry");
                const uint32_t word = cp.rawcb[ri], nmask = cp.rawcb[ri + 1] >> 15;
                for (uint32_t j = 0; j < bcl; ++j) r1[j] = ((nmask >> j) & 1) ? 4 : static_cast<uint8_t>((word >> (2 * (bcl - 1 - j))) & 3);
            }
            // UMI
            const uint32_t u = cp.umi[i];
            for (uint32_t j = 0; j < umil; ++j) r1[bcl + j] = static_cast<uint8_t>((u >> (2 * (umil - 1 - j))) & 3);
            while (ei < cp.exc.size() && fg2::exc_idx(cp.exc[ei]) < i) ++ei;
            for (size_t e = ei; e < cp.exc.size() && fg2::exc_idx(cp.exc[e]) == i; ++e)
                if (fg2::exc_kind(cp.exc[e]) == fg2::EXC_UMI_N) r1[bcl + fg2::exc_pos(cp.exc[e])] = 4;
            // cDNA
            const uint32_t len = cp.len[i];
            const uint64_t sid = cp.seq[i];
            if (sid >= tab_.n || tab_.slen[sid] != len) throw std::runtime_error("Fg2RunSet: sequence id / length mismatch");
            blk.r2_offsets[i] = static_cast<uint32_t>(r2o);
            blk.r2_lengths[i] = static_cast<uint16_t>(len);
            uint8_t* r2 = blk.r2_data.data() + r2o;
            const uint8_t* row = &tab_.seq2[sid * tab_.slot];
            for (uint32_t j = 0; j < len; ++j) r2[j] = (row[j >> 2] >> (6 - 2 * (j & 3))) & 3;
            for (uint64_t k = tab_.n_first[sid]; k < tab_.n_first[sid + 1]; ++k) r2[tab_.n_pos[k]] = 4;
            if (has_q_) {
                uint8_t* q = blk.r2_qual.data() + r2o;
                const uint8_t* qs = &cp.q[static_cast<size_t>(i) * slot];
                for (uint32_t j = 0; j < len; ++j) q[j] = h_.q_edges[(qs[j >> 2] >> (6 - 2 * (j & 3))) & 3];
            }
            r2o += len;
        }
    }

   private:
    void pread_full(void* buf, size_t n, uint64_t off) const {
        uint8_t* p = static_cast<uint8_t*>(buf);
        while (n) {
            ssize_t r = ::pread(fd_, p, n, static_cast<off_t>(off));
            if (r <= 0) throw std::runtime_error("Fg2RunSet: short read of " + path_);
            p += r; n -= static_cast<size_t>(r); off += static_cast<uint64_t>(r);
        }
    }

    void inflate(const uint8_t* base, const fg2::SecDesc& d, bool raw, std::vector<uint8_t>& out) const {
        if (verify_crc_ && d.crc32c != fg2::crc32c(base + d.offset, d.comp_size)) throw std::runtime_error("Fg2RunSet: plane CRC mismatch");
        out.resize(d.raw_size);
        if (raw) { std::memcpy(out.data(), base + d.offset, d.raw_size); return; }
        if (d.raw_size == 0) return;
        size_t r = ZSTD_decompress(out.data(), out.size(), base + d.offset, d.comp_size);
        if (ZSTD_isError(r) || r != d.raw_size) throw std::runtime_error("Fg2RunSet: zstd plane did not inflate");
    }

    void load_whitelist(const std::vector<std::string>& dirs) {
        std::string found;
        for (const auto& d : dirs) { std::string p = d + "/" + h_.wl_name; if (std::ifstream(p).good()) { found = p; break; } }
        if (found.empty()) throw std::runtime_error(std::string("Fg2RunSet: whitelist ") + h_.wl_name + " not found");
        std::ifstream f(found);
        std::string line;
        fg2::Sha256 sha;
        { std::ifstream raw(found, std::ios::binary); char b[1 << 16]; while (raw.read(b, sizeof b) || raw.gcount()) sha.update(reinterpret_cast<const uint8_t*>(b), static_cast<size_t>(raw.gcount())); }
        uint8_t dig[32]; sha.finish(dig);
        if (std::memcmp(dig, h_.wl_sha256, 32) != 0) throw std::runtime_error("Fg2RunSet: whitelist " + found + " differs from the one the file was built with (sha256)");
        // CbCorrector::load_whitelist's order: N at position 0 skipped, N elsewhere read as A, sorted, unique
        while (std::getline(f, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
            if (line.empty()) continue;
            uint64_t v = 0; bool skip = false;
            for (size_t i = 0; i < line.size(); ++i) {
                uint64_t nt;
                switch (line[i]) { case 'A': case 'a': nt = 0; break; case 'C': case 'c': nt = 1; break; case 'G': case 'g': nt = 2; break; case 'T': case 't': nt = 3; break; default: nt = 4; }
                if (nt > 3) { if (i == 0) { skip = true; break; } nt = 0; }
                v = (v << 2) + nt;
            }
            if (!skip) wl_.push_back(v);
        }
        std::sort(wl_.begin(), wl_.end());
        wl_.erase(std::unique(wl_.begin(), wl_.end()), wl_.end());
        if (wl_.size() != h_.wl_entries) throw std::runtime_error("Fg2RunSet: whitelist entry count differs from the file's");
    }

    void load_table() {
        tab_.n = h_.n_unique; tab_.slot = h_.seq_slot;
        tab_.seq2.resize(tab_.n * tab_.slot); tab_.slen.resize(tab_.n);
        tab_.n_first.assign(tab_.n + 1, 0);
        std::vector<std::pair<uint32_t, uint8_t>> nlist;   // (global id, position)
        for (const auto& b : blocks_) {
            std::vector<uint8_t> stored(b.total_bytes), a, l, e;
            pread_full(stored.data(), stored.size(), b.offset);
            fg2::BlockHeader bh; std::memcpy(&bh, stored.data(), sizeof bh);
            if (bh.magic != fg2::SEQB_MAGIC) throw std::runtime_error("Fg2RunSet: bad sequence block");
            inflate(stored.data(), bh.sec[fg2::SP_SEQ2], bh.flags & 1, a);
            inflate(stored.data(), bh.sec[fg2::SP_SLEN], (bh.flags >> 1) & 1, l);
            inflate(stored.data(), bh.sec[fg2::SP_SEXC], (bh.flags >> 2) & 1, e);
            std::memcpy(&tab_.seq2[bh.first_seq * tab_.slot], a.data(), a.size());
            std::memcpy(&tab_.slen[bh.first_seq], l.data(), l.size());
            for (size_t k = 0; k < e.size() / 4; ++k) {
                uint32_t v; std::memcpy(&v, &e[4 * k], 4);
                nlist.push_back({static_cast<uint32_t>(bh.first_seq) + (v & 0x7FFF), static_cast<uint8_t>(v >> 15)});
            }
        }
        for (const auto& np : nlist) tab_.n_first[np.first + 1]++;
        for (uint64_t i = 0; i < tab_.n; ++i) tab_.n_first[i + 1] += tab_.n_first[i];
        tab_.n_pos.resize(nlist.size());
        std::vector<uint64_t> fill(tab_.n_first.begin(), tab_.n_first.end() - 1);
        for (const auto& np : nlist) tab_.n_pos[fill[np.first]++] = np.second;   // nlist is in (id, position) order, so positions stay ascending
    }

    std::string path_;
    int fd_ = -1;
    bool verify_crc_ = true, has_q_ = false;
    fg2::FileHeader h_{};
    Header hdr_{};
    std::vector<fg2::IndexEntry> chunks_, blocks_;
    std::vector<uint64_t> first_read_;
    std::vector<fg2::RunEntry> runs_;
    std::vector<uint64_t> wl_;
    SeqTable tab_;
    std::vector<uint8_t> dict_flat_;
};

}  // namespace singlet::fq
