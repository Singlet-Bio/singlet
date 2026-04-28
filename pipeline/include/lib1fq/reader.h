// lib1fq/reader.h — Block-by-block .1fq file reader
//
// Reads .1fq files and decodes blocks into byte-numeric sequences
// ready for STAR's Read1[] buffers.
//
// Usage:
//   lib1fq::Reader r;
//   r.open("input.1fq");
//   auto hdr = r.header();
//   lib1fq::DecodedBlock blk;
//   while (r.read_block(blk)) {
//       for (uint32_t i = 0; i < blk.n_reads; i++) {
//           // blk.r2_seq(i), blk.r2_len(i) — ready for STAR
//       }
//   }

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "compress.h"
#include "packing.h"
#include "protocol.h"
#include "types.h"

namespace lib1fq {

// Decoded block: provides access to read data in byte-numeric format
struct DecodedBlock {
    uint32_t n_reads = 0;

    // R1 data (barcode + UMI) — reconstructed from dict index if BC_DICT
    std::vector<uint8_t> r1_data;      // All R1 sequences concatenated
    std::vector<uint32_t> r1_offsets;  // Per-read start offset into r1_data
    std::vector<uint16_t> r1_lengths;

    // R2 data (cDNA / genomic)
    std::vector<uint8_t> r2_data;
    std::vector<uint32_t> r2_offsets;
    std::vector<uint16_t> r2_lengths;

    // R2 quality (decoded to phred)
    std::vector<uint8_t> r2_qual;  // Same layout as r2_data

    // BC dict indices (if BC_DICT mode; uint32_t sentinel = dict_size for unknown)
    std::vector<uint32_t> bc_indices;

    // Trim lengths (if FLAG_TRIMMED)
    std::vector<uint16_t> trim_lengths;

    // Duplicate counts (if FLAG_DEDUPED)
    std::vector<uint32_t> dup_counts;

    // I2 barcode stream (scATAC 3-read mode; present when ExtraStreams::HAS_I2)
    std::vector<uint8_t> i2_data;  // All I2 sequences concatenated
    std::vector<uint32_t> i2_offsets;
    std::vector<uint16_t> i2_lengths;

    // Accessors for read i
    const uint8_t* r1_seq(uint32_t i) const { return r1_data.data() + r1_offsets[i]; }
    uint16_t r1_len(uint32_t i) const { return r1_lengths[i]; }
    const uint8_t* r2_seq(uint32_t i) const { return r2_data.data() + r2_offsets[i]; }
    uint16_t r2_len(uint32_t i) const { return r2_lengths[i]; }
    const uint8_t* i2_seq(uint32_t i) const { return i2_data.data() + i2_offsets[i]; }
    uint16_t i2_len(uint32_t i) const { return i2_lengths[i]; }

    const uint8_t* r2_quality(uint32_t i) const {
        if (r2_qual.empty()) return nullptr;
        uint32_t off = r2_offsets[i];
        uint16_t len = r2_lengths[i];
        // Defensive: if variable-length R2 decode had wrong stride the quality
        // buffer may be shorter than expected — return nullptr to fall back to
        // uniform 'F' quality rather than reading out-of-bounds memory.
        if (static_cast<size_t>(off) + len > r2_qual.size()) return nullptr;
        return r2_qual.data() + off;
    }

    void clear() {
        n_reads = 0;
        r1_data.clear();
        r1_offsets.clear();
        r1_lengths.clear();
        r2_data.clear();
        r2_offsets.clear();
        r2_lengths.clear();
        r2_qual.clear();
        bc_indices.clear();
        trim_lengths.clear();
        dup_counts.clear();
        i2_data.clear();
        i2_offsets.clear();
        i2_lengths.clear();
    }

    void reserve(uint32_t n, uint16_t avg_r1_len, uint16_t avg_r2_len) {
        r1_data.reserve(n * avg_r1_len);
        r1_offsets.reserve(n);
        r1_lengths.reserve(n);
        r2_data.reserve(n * avg_r2_len);
        r2_offsets.reserve(n);
        r2_lengths.reserve(n);
    }
};

class Reader {
   public:
    Reader() = default;
    ~Reader() { close(); }

    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    void open(const char* path) {
        fp_ = std::fopen(path, "rb");
        if (!fp_) throw std::runtime_error("Cannot open: " + std::string(path));

        // Read header
        if (std::fread(&hdr_, sizeof(Header), 1, fp_) != 1) {
            throw std::runtime_error("Failed to read header");
        }
        if (!hdr_.valid_magic()) {
            throw std::runtime_error("Invalid .1fq magic");
        }
        if (hdr_.version > FORMAT_VERSION) {
            throw std::runtime_error("Unsupported .1fq version: " +
                                     std::to_string(hdr_.version));
        }

        codec_ = static_cast<Codec>(hdr_.codec);
        seq_enc_ = static_cast<SeqEncoding>(hdr_.seq_encoding);
        qual_mode_ = static_cast<QualMode>(hdr_.qual_mode);
        has_bc_dict_ = (hdr_.flags & Flags::BC_DICT) != 0;
        has_trimming_ = (hdr_.flags & Flags::TRIMMED) != 0;
        has_dedup_ = (hdr_.flags & Flags::DEDUPED) != 0;
        has_i2_stream_ = (hdr_.reserved[0] & ExtraStreams::HAS_I2) != 0;
        i2_decode_len_ = has_i2_stream_ ? hdr_.stream_lengths[2] : 0;

        // When TRIMMED flag is set, R2 is stored variable-length even if
        // stream_lengths[1] != 0 (backward compat with older encoders).
        r2_decode_len_ = has_trimming_ ? 0 : hdr_.stream_lengths[1];

        // Read barcode dictionary if present
        if (has_bc_dict_ && hdr_.bc_dict_size > 0) {
            read_bc_dict();
        }

        // Look up the BC offset within R1 for BC_DICT reconstruction.
        // The BC_DICT format stores BC+UMI separately; reconstruction must place BC
        // at its original position in R1 (bc_offset from the protocol spec).
        // Without this, protocols with BC NOT at position 0 (e.g. sci-RNA-seq3
        // with BC at offset 24) produce BC at the wrong position and STAR returns 0 cells.
        bc_start_ = 0;  // default: BC at start (10x-like)
        for (const auto& spec : known_protocols()) {
            if (spec.protocol_id == hdr_.protocol_id) {
                bc_start_ = spec.bc_offset;
                break;
            }
        }

        // Read block index from footer
        read_block_index();
        current_block_ = 0;
    }

    void close() {
        if (fp_) {
            std::fclose(fp_);
            fp_ = nullptr;
        }
    }

    const Header& header() const { return hdr_; }
    uint32_t block_count() const { return hdr_.block_count; }
    uint64_t total_reads() const { return hdr_.n_unique; }
    bool has_bc_dict() const { return has_bc_dict_; }
    uint32_t bc_dict_size() const { return bc_dict_n_; }

    // Access the flat BC dictionary (bc_dict_n * bc_length bytes)
    const std::vector<uint8_t>& bc_dict_flat() const { return bc_dict_flat_; }

    // Reconstruct vector-of-vectors BC dict for legacy callers (dedup, tests).
    std::vector<std::vector<uint8_t>> bc_dict() const {
        std::vector<std::vector<uint8_t>> out(bc_dict_n_);
        for (uint32_t i = 0; i < bc_dict_n_; ++i)
            out[i].assign(bc_dict_flat_.data() + i * bc_length_,
                          bc_dict_flat_.data() + (i + 1) * bc_length_);
        return out;
    }
    uint16_t bc_length() const { return bc_length_; }
    uint16_t umi_length() const {
        // UMI length = R1 length - BC length (if known).
        // Guard against uint16_t underflow: if stream_lengths[0] < bc_length_
        // (e.g. R1=10bp file encoded with BC_DICT bc_length=16), return 0
        // rather than wrapping to 65520+, which would cause decode_seq_column
        // to request gigabytes of data per read and segfault.
        if (hdr_.stream_lengths[0] > 0 && bc_length_ > 0 &&
            hdr_.stream_lengths[0] >= bc_length_)
            return hdr_.stream_lengths[0] - bc_length_;
        return 0;
    }

    // Read and decode the next block. Returns false when all blocks consumed.
    bool read_block(DecodedBlock& blk) {
        blk.clear();

        if (current_block_ >= hdr_.block_count) return false;

        // Seek to block position
        uint64_t offset = block_offsets_[current_block_];
        std::fseek(fp_, static_cast<long>(offset), SEEK_SET);

        // Read block header
        BlockHeader bh;
        if (std::fread(&bh, sizeof(BlockHeader), 1, fp_) != 1) {
            return false;
        }

        // Read compressed payload
        comp_buf_.resize(bh.compressed_size);
        if (std::fread(comp_buf_.data(), 1, bh.compressed_size, fp_) !=
            bh.compressed_size) {
            return false;
        }

        // Decompress
        size_t est_raw = static_cast<size_t>(bh.n_reads) * 200;
        raw_buf_.resize(est_raw);
        size_t raw_size;
        try {
            raw_size = compress::decompress_block(
                codec_, comp_buf_.data(), bh.compressed_size,
                raw_buf_.data(), raw_buf_.size());
        } catch (...) {
            raw_buf_.resize(est_raw * 4);
            raw_size = compress::decompress_block(
                codec_, comp_buf_.data(), bh.compressed_size,
                raw_buf_.data(), raw_buf_.size());
        }

        // Validate block CRC32 if present and verification enabled
        if (verify_crc_ && bh.raw_crc32 != 0) {
            uint32_t actual = compute_crc32(raw_buf_.data(), raw_size);
            if (actual != bh.raw_crc32) {
                throw std::runtime_error(
                    "Block CRC32 mismatch (expected " +
                    std::to_string(bh.raw_crc32) + ", got " +
                    std::to_string(actual) + ")");
            }
        }

        const uint8_t* ptr = raw_buf_.data();
        const uint8_t* end = ptr + raw_size;

        blk.n_reads = bh.n_reads;
        blk.reserve(bh.n_reads, hdr_.stream_lengths[0] ? hdr_.stream_lengths[0] : 28,
                    hdr_.stream_lengths[1] ? hdr_.stream_lengths[1] : 91);

        if (has_bc_dict_) {
            // Decode BC dict indices + UMI, reconstruct full R1
            uint16_t umi_len = umi_length();
            blk.bc_indices.resize(bh.n_reads);

            // Phase 1: decode varint BC indices
            for (uint32_t i = 0; i < bh.n_reads; ++i) {
                uint64_t idx;
                uint32_t consumed = pack::decode_varint(ptr, idx);
                ptr += consumed;
                blk.bc_indices[i] = static_cast<uint32_t>(idx);
                // If sentinel (unknown BC), raw BC bytes follow
                if (idx == bc_dict_n_) {
                    // Skip the raw BC bytes for now (handled below)
                    ptr += bc_length_;
                }
            }

            // Phase 2: decode UMI column
            std::vector<uint8_t> umi_data;
            std::vector<uint32_t> umi_offsets;
            std::vector<uint16_t> umi_lengths;
            ptr = decode_seq_column(ptr, end, bh.n_reads, umi_len,
                                    umi_data, umi_offsets, umi_lengths);

            // Phase 3: reconstruct full R1 = non-BC prefix + BC + non-BC suffix
            // placed at their original positions within the R1 layout.
            // Bug fix: the old code wrote BC at offset 0 and UMI right after,
            // using a running r1_dst pointer that advanced by (bc_len + umi_len)
            // per read — but r1_offsets used stride r1_len, so reads i>0 had
            // misaligned data.  The correct approach: compute r1_base = data +
            // i*r1_len and memcpy BC/UMI into their original R1 positions (derived
            // from bc_start_, the protocol's bc_offset looked up in known_protocols).
            // The "UMI column" in BC_DICT stores the non-BC bytes of R1 concatenated
            // in order: [R1[0..bc_start-1]][R1[bc_start+bc_len..r1_len-1]].
            uint16_t r1_len = hdr_.stream_lengths[0];
            uint32_t n = bh.n_reads;
            blk.r1_data.resize(static_cast<size_t>(n) * r1_len);  // zero-initialized
            blk.r1_offsets.resize(n);
            blk.r1_lengths.assign(n, r1_len);

            for (uint32_t i = 0; i < n; ++i) {
                blk.r1_offsets[i] = i * r1_len;
                uint8_t* r1_base = blk.r1_data.data() + static_cast<size_t>(i) * r1_len;
                const uint16_t ul = umi_lengths[i];

                // Non-BC prefix: umi_data[0..bc_start_-1] → r1[0..bc_start_-1]
                if (bc_start_ > 0 && ul > 0) {
                    const uint16_t pfx = std::min(bc_start_, ul);
                    std::memcpy(r1_base, umi_data.data() + umi_offsets[i], pfx);
                }

                // BC: bc_dict[idx] → r1[bc_start_..bc_start_+bc_length_-1]
                uint32_t idx = blk.bc_indices[i];
                if (idx < bc_dict_n_) {
                    std::memcpy(r1_base + bc_start_,
                                bc_dict_flat_.data() + static_cast<size_t>(idx) * bc_length_,
                                bc_length_);
                } else {
                    std::memset(r1_base + bc_start_, 4, bc_length_);
                }

                // Non-BC suffix: umi_data[bc_start_..ul-1] → r1[bc_start_+bc_len..r1_len-1]
                const uint16_t sfx_src = bc_start_;
                const uint16_t sfx_dst = bc_start_ + bc_length_;
                if (sfx_dst < r1_len && sfx_src < ul) {
                    const uint16_t sfx_len = std::min(
                        static_cast<uint16_t>(ul - sfx_src),
                        static_cast<uint16_t>(r1_len - sfx_dst));
                    std::memcpy(r1_base + sfx_dst,
                                umi_data.data() + umi_offsets[i] + sfx_src, sfx_len);
                }
            }
        } else {
            // Legacy: full R1 column
            ptr = decode_seq_column(ptr, end, bh.n_reads, hdr_.stream_lengths[0],
                                    blk.r1_data, blk.r1_offsets, blk.r1_lengths);
        }

        // Decode R2 sequences
        ptr = decode_seq_column(ptr, end, bh.n_reads, r2_decode_len_,
                                blk.r2_data, blk.r2_offsets, blk.r2_lengths);
        // If decode_seq_column returned end, the block is corrupt past some read.
        // Skip remaining column decoders to avoid UB reads past the buffer.
        // All corrupt reads have r2_length=0 and are skipped by the FIFO writer.
        if (ptr < end) {
            if (qual_mode_ == QualMode::BINNED4) {
                ptr = decode_qual_column_binned4(ptr, end, blk.r2_lengths, blk.r2_qual);
            } else if (qual_mode_ == QualMode::BINNED2) {
                ptr = decode_qual_column_binned2(ptr, end, blk.r2_lengths, blk.r2_qual);
            } else if (qual_mode_ == QualMode::FULL) {
                for (uint32_t i = 0; i < bh.n_reads; ++i) {
                    uint16_t len = blk.r2_lengths[i];
                    blk.r2_qual.insert(blk.r2_qual.end(), ptr, ptr + len);
                    ptr += len;
                }
            }

            // Decode trim lengths
            if (has_trimming_) {
                blk.trim_lengths.resize(bh.n_reads);
                for (uint32_t i = 0; i < bh.n_reads; ++i) {
                    uint64_t tl;
                    uint32_t consumed = pack::decode_varint(ptr, tl);
                    ptr += consumed;
                    blk.trim_lengths[i] = static_cast<uint16_t>(tl);
                }
            }

            // Decode duplicate counts
            if (has_dedup_) {
                blk.dup_counts.resize(bh.n_reads);
                for (uint32_t i = 0; i < bh.n_reads; ++i) {
                    uint64_t dc;
                    uint32_t consumed = pack::decode_varint(ptr, dc);
                    ptr += consumed;
                    blk.dup_counts[i] = static_cast<uint32_t>(dc);
                }
            }

            // Decode I2 barcode stream (scATAC 3-read mode)
            if (has_i2_stream_ && ptr < end) {
                ptr = decode_seq_column(ptr, end, bh.n_reads, i2_decode_len_,
                                        blk.i2_data, blk.i2_offsets, blk.i2_lengths);
            }
        }

        current_block_++;
        return true;
    }

    // Seek to a specific block (for random access)
    void seek_block(uint32_t block_idx) {
        if (block_idx > hdr_.block_count) {
            throw std::runtime_error("Block index out of range");
        }
        current_block_ = block_idx;
    }

    // Disable CRC verification for faster decode
    void set_verify_crc(bool v) { verify_crc_ = v; }

    // Block index accessors for parallel decode
    uint64_t block_offset(uint32_t i) const { return block_offsets_[i]; }
    uint32_t block_comp_size(uint32_t i) const { return block_comp_sizes_[i]; }
    int file_descriptor() const { return fileno(fp_); }

    // Decode a raw (already decompressed) payload into a DecodedBlock.
    // Thread-safe: uses only const reader state (dict, header).
    void decode_payload(const uint8_t* raw, size_t raw_size,
                        const BlockHeader& bh, DecodedBlock& blk) const {
        const uint8_t* ptr = raw;
        const uint8_t* end = ptr + raw_size;

        blk.clear();
        blk.n_reads = bh.n_reads;
        blk.reserve(bh.n_reads, hdr_.stream_lengths[0] ? hdr_.stream_lengths[0] : 28,
                    hdr_.stream_lengths[1] ? hdr_.stream_lengths[1] : 91);

        if (has_bc_dict_) {
            uint16_t umi_len_val = umi_length();
            blk.bc_indices.resize(bh.n_reads);
            for (uint32_t i = 0; i < bh.n_reads; ++i) {
                uint64_t idx;
                uint32_t consumed = pack::decode_varint(ptr, idx);
                ptr += consumed;
                blk.bc_indices[i] = static_cast<uint32_t>(idx);
                if (idx == bc_dict_n_) ptr += bc_length_;
            }
            std::vector<uint8_t> umi_data;
            std::vector<uint32_t> umi_offsets;
            std::vector<uint16_t> umi_lengths;
            ptr = decode_seq_column(ptr, end, bh.n_reads, umi_len_val,
                                    umi_data, umi_offsets, umi_lengths);
            uint16_t r1_len = hdr_.stream_lengths[0];
            uint32_t n = bh.n_reads;
            blk.r1_data.resize(static_cast<size_t>(n) * r1_len);  // zero-initialized
            blk.r1_offsets.resize(n);
            blk.r1_lengths.assign(n, r1_len);
            for (uint32_t i = 0; i < n; ++i) {
                blk.r1_offsets[i] = i * r1_len;
                uint8_t* r1_base = blk.r1_data.data() + static_cast<size_t>(i) * r1_len;
                const uint16_t ul = umi_lengths[i];
                if (bc_start_ > 0 && ul > 0) {
                    const uint16_t pfx = std::min(bc_start_, ul);
                    std::memcpy(r1_base, umi_data.data() + umi_offsets[i], pfx);
                }
                uint32_t idx = blk.bc_indices[i];
                if (idx < bc_dict_n_) {
                    std::memcpy(r1_base + bc_start_,
                                bc_dict_flat_.data() + static_cast<size_t>(idx) * bc_length_,
                                bc_length_);
                } else {
                    std::memset(r1_base + bc_start_, 4, bc_length_);
                }
                const uint16_t sfx_src = bc_start_;
                const uint16_t sfx_dst = bc_start_ + bc_length_;
                if (sfx_dst < r1_len && sfx_src < ul) {
                    const uint16_t sfx_len = std::min(
                        static_cast<uint16_t>(ul - sfx_src),
                        static_cast<uint16_t>(r1_len - sfx_dst));
                    std::memcpy(r1_base + sfx_dst,
                                umi_data.data() + umi_offsets[i] + sfx_src, sfx_len);
                }
            }
        } else {
            ptr = decode_seq_column(ptr, end, bh.n_reads, hdr_.stream_lengths[0],
                                    blk.r1_data, blk.r1_offsets, blk.r1_lengths);
        }
        ptr = decode_seq_column(ptr, end, bh.n_reads, r2_decode_len_,
                                blk.r2_data, blk.r2_offsets, blk.r2_lengths);
        // Guard: if decode_seq_column returned end, the block is corrupt.
        // Skip remaining column decoders to avoid UB reads past the buffer.
        if (ptr < end) {
            if (qual_mode_ == QualMode::BINNED4) {
                ptr = decode_qual_column_binned4(ptr, end, blk.r2_lengths, blk.r2_qual);
            } else if (qual_mode_ == QualMode::BINNED2) {
                ptr = decode_qual_column_binned2(ptr, end, blk.r2_lengths, blk.r2_qual);
            } else if (qual_mode_ == QualMode::FULL) {
                for (uint32_t i = 0; i < bh.n_reads; ++i) {
                    uint16_t len = blk.r2_lengths[i];
                    blk.r2_qual.insert(blk.r2_qual.end(), ptr, ptr + len);
                    ptr += len;
                }
            }
            if (has_trimming_) {
                blk.trim_lengths.resize(bh.n_reads);
                for (uint32_t i = 0; i < bh.n_reads; ++i) {
                    uint64_t tl;
                    uint32_t consumed = pack::decode_varint(ptr, tl);
                    ptr += consumed;
                    blk.trim_lengths[i] = static_cast<uint16_t>(tl);
                }
            }
            if (has_dedup_) {
                blk.dup_counts.resize(bh.n_reads);
                for (uint32_t i = 0; i < bh.n_reads; ++i) {
                    uint64_t dc;
                    uint32_t consumed = pack::decode_varint(ptr, dc);
                    ptr += consumed;
                    blk.dup_counts[i] = static_cast<uint32_t>(dc);
                }
            }
            // Decode I2 barcode stream (scATAC 3-read mode)
            if (has_i2_stream_ && ptr < end) {
                ptr = decode_seq_column(ptr, end, bh.n_reads, i2_decode_len_,
                                        blk.i2_data, blk.i2_offsets, blk.i2_lengths);
            }
        }
    }

    // Read metadata JSON (if present)
    std::string read_metadata() {
        if (hdr_.meta_size == 0) return "";

        // Metadata is stored after the last data block, before the index
        // We need to find where it starts: after all data blocks
        uint64_t meta_offset = 0;
        if (!block_offsets_.empty()) {
            // Last block offset + block header + compressed size
            uint64_t last_off = block_offsets_.back();
            std::fseek(fp_, static_cast<long>(last_off), SEEK_SET);
            BlockHeader bh;
            std::fread(&bh, sizeof(BlockHeader), 1, fp_);
            meta_offset = last_off + sizeof(BlockHeader) + bh.compressed_size;
        } else {
            meta_offset = sizeof(Header);
        }

        std::fseek(fp_, static_cast<long>(meta_offset), SEEK_SET);
        std::vector<uint8_t> comp(hdr_.meta_size);
        if (std::fread(comp.data(), 1, hdr_.meta_size, fp_) != hdr_.meta_size) {
            throw std::runtime_error("Failed to read metadata block");
        }

        // Decompress
        std::vector<uint8_t> raw(hdr_.meta_size * 10);  // generous estimate
        size_t raw_size;
        try {
            raw_size = compress::decompress_block(
                codec_, comp.data(), hdr_.meta_size,
                raw.data(), raw.size());
        } catch (...) {
            raw.resize(hdr_.meta_size * 50);
            raw_size = compress::decompress_block(
                codec_, comp.data(), hdr_.meta_size,
                raw.data(), raw.size());
        }

        return std::string(reinterpret_cast<const char*>(raw.data()), raw_size);
    }

   private:
    void read_bc_dict() {
        // Dictionary is stored right after the header:
        // [uint32_t compressed_size][compressed_data]
        uint32_t comp_size;
        if (std::fread(&comp_size, sizeof(uint32_t), 1, fp_) != 1) {
            throw std::runtime_error("Failed to read BC dict size");
        }
        std::vector<uint8_t> comp(comp_size);
        if (std::fread(comp.data(), 1, comp_size, fp_) != comp_size) {
            throw std::runtime_error("Failed to read BC dict data");
        }

        // Decompress
        std::vector<uint8_t> raw(comp_size * 20);
        size_t raw_size;
        try {
            raw_size = compress::decompress_block(
                codec_, comp.data(), comp_size, raw.data(), raw.size());
        } catch (...) {
            raw.resize(comp_size * 100);
            raw_size = compress::decompress_block(
                codec_, comp.data(), comp_size, raw.data(), raw.size());
        }

        // Parse: [n_dicts][stream][segment][varint n_entries][bc_len][seq_data]
        const uint8_t* p = raw.data();
        uint8_t n_dicts = *p++;
        (void)n_dicts;  // We only handle 1 dictionary for now
        p++;            // stream_index
        p++;            // segment_index
        uint64_t n_entries;
        uint32_t consumed = pack::decode_varint(p, n_entries);
        p += consumed;
        bc_length_ = *p++;

        bc_dict_n_ = static_cast<uint32_t>(n_entries);
        bc_dict_flat_.resize(static_cast<size_t>(n_entries) * bc_length_);
        std::memcpy(bc_dict_flat_.data(), p, static_cast<size_t>(n_entries) * bc_length_);
        p += static_cast<size_t>(n_entries) * bc_length_;
    }

    void read_block_index() {
        // Read footer from end of file
        std::fseek(fp_, -static_cast<long>(sizeof(Footer)), SEEK_END);
        Footer ftr;
        if (std::fread(&ftr, sizeof(Footer), 1, fp_) != 1) {
            throw std::runtime_error(
                ".1fq file appears truncated or corrupted: failed to read footer");
        }
        if (!ftr.valid_magic()) {
            // Provide actionable diagnostics: report header version and magic bytes found
            char found[5] = {};
            std::memcpy(found, ftr.magic, 4);
            char hex[32];
            std::snprintf(hex, sizeof(hex), "0x%02x%02x%02x%02x",
                          (uint8_t)ftr.magic[0], (uint8_t)ftr.magic[1],
                          (uint8_t)ftr.magic[2], (uint8_t)ftr.magic[3]);
            throw std::runtime_error(
                ".1fq footer magic invalid (found " + std::string(hex) +
                ", expected 0x31465100). File may be truncated, corrupted, "
                "or written by an incompatible encoder. "
                "Header format version: " +
                std::to_string(hdr_.version) +
                ", supported: " + std::to_string(FORMAT_VERSION));
        }

        if (ftr.block_count != hdr_.block_count) {
            throw std::runtime_error("Footer block count mismatch");
        }

        uint32_t n = ftr.block_count;
        block_offsets_.resize(n);
        block_comp_sizes_.resize(n);

        // Compute index offset from end-of-file to handle files >4GB.
        // ftr.index_offset is uint32_t and silently truncates when the index
        // lands beyond the 4 GiB mark (e.g. an 8.56 GB .1fq with 1010 blocks).
        // The on-disk layout is deterministic:
        //   [data blocks] [metadata block] [n×uint64 offsets] [n×uint32 comp_sizes] [footer]
        // so we can compute the exact index start position without relying on the
        // potentially-truncated ftr.index_offset field.
        {
            std::fseek(fp_, 0, SEEK_END);
            int64_t file_size = static_cast<int64_t>(std::ftell(fp_));
            int64_t index_bytes = static_cast<int64_t>(n) *
                                  static_cast<int64_t>(sizeof(uint64_t) + sizeof(uint32_t));
            int64_t index_offset = file_size -
                                   static_cast<int64_t>(sizeof(Footer)) - index_bytes;
            if (index_offset < static_cast<int64_t>(sizeof(Header))) {
                throw std::runtime_error(
                    "Block index offset is before file header — .1fq file is corrupted");
            }
            std::fseek(fp_, index_offset, SEEK_SET);
        }

        // Read offsets
        if (n > 0) {
            if (std::fread(block_offsets_.data(), sizeof(uint64_t), n, fp_) != n) {
                throw std::runtime_error("Failed to read block offsets");
            }
            if (std::fread(block_comp_sizes_.data(), sizeof(uint32_t), n, fp_) != n) {
                throw std::runtime_error("Failed to read block sizes");
            }
        }
    }

    // Decode a sequence column (2-bit packed or byte-numeric)
    const uint8_t* decode_seq_column(const uint8_t* ptr, const uint8_t* end,
                                     uint32_t n_reads, uint16_t fixed_length,
                                     std::vector<uint8_t>& data,
                                     std::vector<uint32_t>& offsets,
                                     std::vector<uint16_t>& lengths) const {
        // Fast path: fixed-length + 2-bit packed (most common for UMI and R2)
        if (fixed_length > 0 && seq_enc_ == SeqEncoding::PACKED_2BIT) {
            const uint32_t fl = fixed_length;
            const uint32_t total_bases = n_reads * fl;
            data.resize(total_bases);
            offsets.resize(n_reads);
            lengths.assign(n_reads, fixed_length);

            uint8_t* dst = data.data();
            for (uint32_t i = 0; i < n_reads; ++i) {
                offsets[i] = i * fl;
                uint8_t has_n = *ptr++;
                uint32_t pb = pack::packed_bytes(fl);
                pack::unpack_2bit(ptr, fl, dst);
                ptr += pb;
                if (has_n) {
                    uint32_t nb = pack::n_bitmap_bytes(fl);
                    pack::apply_n_bitmap(ptr, fl, dst);
                    ptr += nb;
                }
                dst += fl;
            }
            return ptr;
        }

        // General path: variable-length or byte-numeric
        for (uint32_t i = 0; i < n_reads; ++i) {
            uint16_t len = fixed_length;

            if (seq_enc_ == SeqEncoding::PACKED_2BIT) {
                if (fixed_length == 0) {
                    uint64_t vlen;
                    uint32_t consumed = pack::decode_varint(ptr, vlen);
                    ptr += consumed;
                    len = static_cast<uint16_t>(vlen);
                }

                uint8_t has_n = *ptr++;
                uint32_t pb = pack::packed_bytes(len);

                // GUARD: corrupted varint produces an anomalously large encoded length
                // (observed: r2_lengths[i]=1,437,408 for a sci-RNA-seq3 read where the
                // correct value is ~100 bp). The erroneous 'len' value cannot be used to
                // advance ptr safely — packed_bytes(1437408)=359352 would go ~340 KB
                // past the block buffer end, causing UB reads from heap memory for all
                // subsequent reads in the block (cascade corruption).
                //
                // Correct fix: when len > 1024 and fixed_length == 0, we cannot
                // recover the true byte count for this read. Fill all remaining reads in
                // the block with length=0 and return 'end' to signal block corruption to
                // the caller. The caller guards quality/trim/dedup column decodes on
                // ptr < end, so those columns are safely skipped for corrupt blocks.
                // The FIFO writer skips all r2len==0 reads, so no garbage data reaches
                // STAR.
                if (len > 1024 && fixed_length == 0) {
                    // Signal block corruption: fill remaining reads with 0-length
                    for (uint32_t j = i; j < n_reads; ++j) {
                        offsets.push_back(static_cast<uint32_t>(data.size()));
                        lengths.push_back(0);
                    }
                    return end;  // caller: ptr==end → skip quality/trim/dedup columns
                }

                uint32_t off = static_cast<uint32_t>(data.size());
                offsets.push_back(off);
                lengths.push_back(len);
                data.resize(off + len);

                pack::unpack_2bit(ptr, len, data.data() + off);
                ptr += pb;

                if (has_n) {
                    uint32_t nb = pack::n_bitmap_bytes(len);
                    pack::apply_n_bitmap(ptr, len, data.data() + off);
                    ptr += nb;
                }
            } else {
                if (fixed_length == 0) {
                    uint64_t vlen;
                    uint32_t consumed = pack::decode_varint(ptr, vlen);
                    ptr += consumed;
                    len = static_cast<uint16_t>(vlen);
                }

                uint32_t off = static_cast<uint32_t>(data.size());
                offsets.push_back(off);
                lengths.push_back(len);
                data.insert(data.end(), ptr, ptr + len);
                ptr += len;
            }
        }
        return ptr;
    }

    // Decode binned-4 quality column
    const uint8_t* decode_qual_column_binned4(const uint8_t* ptr,
                                              const uint8_t* end,
                                              const std::vector<uint16_t>& lengths,
                                              std::vector<uint8_t>& qual) const {
        // Pre-allocate total quality bytes
        size_t total = 0;
        for (uint16_t len : lengths) total += len;
        qual.resize(total);
        uint8_t* dst = qual.data();

        for (uint16_t len : lengths) {
            uint32_t pb = pack::packed_bytes(len);
            pack::unpack_qual_binned4(ptr, len, dst);
            ptr += pb;
            dst += len;
        }
        return ptr;
    }

    // Decode binned-2 quality column (1 bit/base, pass/fail at Q20)
    const uint8_t* decode_qual_column_binned2(const uint8_t* ptr,
                                              const uint8_t* end,
                                              const std::vector<uint16_t>& lengths,
                                              std::vector<uint8_t>& qual) const {
        for (uint16_t len : lengths) {
            uint32_t pb = pack::packed_bytes_binned2(len);
            uint32_t off = static_cast<uint32_t>(qual.size());
            qual.resize(off + len);
            pack::unpack_qual_binned2(ptr, len, qual.data() + off);
            ptr += pb;
        }
        return ptr;
    }

    FILE* fp_ = nullptr;
    Header hdr_;
    Codec codec_;
    SeqEncoding seq_enc_;
    QualMode qual_mode_;
    bool has_bc_dict_ = false;
    bool has_trimming_ = false;
    bool has_dedup_ = false;
    bool has_i2_stream_ = false;
    bool verify_crc_ = true;
    uint16_t r2_decode_len_ = 0;  // 0 = variable-length R2 (trimming active)
    uint16_t i2_decode_len_ = 0;  // 0 = I2 stream absent
    uint16_t bc_start_ = 0;       // BC offset within R1 for BC_DICT reconstruction

    // Barcode dictionary — flat contiguous array (n_entries × bc_length bytes)
    std::vector<uint8_t> bc_dict_flat_;
    uint32_t bc_dict_n_ = 0;
    uint16_t bc_length_ = 0;

    uint32_t current_block_ = 0;
    std::vector<uint64_t> block_offsets_;
    std::vector<uint32_t> block_comp_sizes_;

    // Reusable buffers
    std::vector<uint8_t> comp_buf_;
    std::vector<uint8_t> raw_buf_;
};

}  // namespace lib1fq
