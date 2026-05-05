// lib1fq/writer.h — Streaming .1fq file writer
//
// Accumulates reads into blocks, compresses, and writes to a file.
// Call add_read() in a loop, then finish() to finalize.
//
// Usage:
//   lib1fq::Writer w;
//   w.open("output.1fq", config);
//   while (have_reads) {
//       w.add_read(r2_seq, r2_len, r2_qual, r1_seq, r1_len, r1_qual);
//   }
//   w.finish(metadata_json);

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

#include "compress.h"
#include "packing.h"
#include "types.h"

namespace lib1fq {

// Flat open-addressing hash map specialized for uint64→uint32.
// ~10× faster to build and query than std::unordered_map for 3.7M entries
// because it does a single allocation instead of per-node malloc.
struct FlatBCMap {
    static constexpr uint64_t EMPTY = UINT64_MAX;

    void build(const std::vector<std::vector<uint8_t>>& dict,
               uint16_t bc_len,
               uint64_t (*hash_fn)(const uint8_t*, uint16_t)) {
        // Capacity = next power of 2 >= 2×dict.size() (load factor ~0.5)
        uint32_t cap = 4;
        while (cap < dict.size() * 2) cap <<= 1;
        mask_ = cap - 1;
        keys_.assign(cap, EMPTY);
        vals_.resize(cap);
        for (uint32_t i = 0; i < static_cast<uint32_t>(dict.size()); ++i) {
            uint64_t h = hash_fn(dict[i].data(), bc_len);
            uint32_t idx = static_cast<uint32_t>(h) & mask_;
            while (keys_[idx] != EMPTY)
                idx = (idx + 1) & mask_;
            keys_[idx] = h;
            vals_[idx] = i;
        }
    }

    // Returns dict index or UINT32_MAX if not found.
    uint32_t find(uint64_t key) const {
        uint32_t idx = static_cast<uint32_t>(key) & mask_;
        while (keys_[idx] != EMPTY) {
            if (keys_[idx] == key) return vals_[idx];
            idx = (idx + 1) & mask_;
        }
        return UINT32_MAX;
    }

    void clear() {
        keys_.clear();
        vals_.clear();
        mask_ = 0;
    }
    bool empty() const { return keys_.empty(); }

   private:
    std::vector<uint64_t> keys_;
    std::vector<uint32_t> vals_;
    uint32_t mask_ = 0;
};

struct WriterConfig {
    Codec codec = Codec::ZSTD;
    int codec_level = 4;
    QualMode qual_mode = QualMode::BINNED4;
    SeqEncoding seq_enc = SeqEncoding::PACKED_2BIT;
    uint32_t block_size = 100000;
    uint8_t n_streams = 2;    // R1 + R2
    uint16_t r1_length = 0;   // Fixed R1 length (0=variable)
    uint16_t r2_length = 0;   // Fixed R2 length (0=variable)
    uint16_t r2_maxlen = 0;   // Max R2 length after trim (0=unlimited)
    uint8_t protocol_id = 0;  // 0 = UNKNOWN
    Confidence confidence = Confidence::NONE;
    AssayType assay_type = AssayType::UNKNOWN;

    // Barcode dictionary (Phase 2)
    // If non-empty, R1 is split into BC (dict-indexed) + UMI (raw bytes).
    // Dictionary entries: frequency-sorted byte-numeric barcode sequences.
    std::vector<std::vector<uint8_t>> bc_dict;  // bc_dict[i] = barcode i
    uint16_t bc_offset = 0;                     // BC start position in R1
    uint16_t bc_length = 0;                     // BC length (must match bc_dict entry size)
    uint16_t umi_offset = 0;                    // UMI start position in R1
    uint16_t umi_length = 0;                    // UMI length

    // PolyA trimming (Phase 2)
    bool polya_trim = false;     // Enable polyA 3' trimming on R2
    uint8_t polya_min_len = 10;  // Min consecutive A's to trigger trim
    uint8_t polya_max_mm = 1;    // Max mismatches in polyA tail

    // Dedup mode (Phase 3) — when true, add_read accepts dup_count
    bool deduped = false;

    // Sort reads by barcode index within each block before compression.
    // Improves zstd compression ratio by grouping same-BC reads together.
    bool sort_by_bc = false;

    // ZSTD multi-threaded compression (0 = single-threaded, >0 = worker threads)
    int compress_threads = 0;

    // scATAC 3-read mode: store R1 genomic + R3 genomic + I2 barcode as separate streams.
    // When true: call add_read_atac() instead of add_read().
    bool has_i2_stream = false;
    uint16_t i2_length = 0;  // Fixed I2 (barcode) length (0=variable)
};

class Writer {
   public:
    Writer() = default;
    ~Writer() {
        if (async_flush_.valid()) async_flush_.wait();
        if (fp_) std::fclose(fp_);
        if (cctx_) ZSTD_freeCCtx(cctx_);
    }

    // Non-copyable
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    // Open output file and write header placeholder.
    void open(const char* path, const WriterConfig& cfg) {
        cfg_ = cfg;
        fp_ = std::fopen(path, "wb");
        if (!fp_) throw std::runtime_error("Cannot open for writing: " + std::string(path));

        // Set up barcode dictionary mode and optional I2 stream
        use_bc_dict_ = !cfg.bc_dict.empty() && cfg.bc_length > 0;
        has_i2_stream_ = cfg.has_i2_stream && cfg.i2_length > 0;
        if (use_bc_dict_) {
            // Build flat hash map: barcode hash → dict index
            bc_lookup_.build(cfg.bc_dict, cfg.bc_length, hash_bc);
        }

        // Initialize header with known values; block_count updated at finish
        hdr_.init();
        hdr_.n_streams = cfg.n_streams;
        hdr_.protocol_id = cfg.protocol_id;
        hdr_.confidence = static_cast<uint8_t>(cfg.confidence);
        hdr_.assay_type = static_cast<uint8_t>(cfg.assay_type);
        hdr_.codec = static_cast<uint8_t>(cfg.codec);
        hdr_.codec_level = cfg.codec_level;
        hdr_.seq_encoding = static_cast<uint8_t>(cfg.seq_enc);
        hdr_.qual_mode = static_cast<uint8_t>(cfg.qual_mode);
        hdr_.block_size = cfg.block_size;
        hdr_.stream_lengths[0] = cfg.r1_length;
        // When polyA trimming is active, R2 is stored variable-length;
        // signal this by setting stream_lengths[1] = 0.
        hdr_.stream_lengths[1] = cfg.polya_trim ? 0 : cfg.r2_length;
        // Default stream roles: R1=0, R2=1 (index streams would be I1=2, I2=3)
        hdr_.stream_roles[0] = static_cast<uint8_t>(StreamRole::R1);
        hdr_.stream_roles[1] = static_cast<uint8_t>(StreamRole::R2);
        hdr_.stream_roles[2] = static_cast<uint8_t>(StreamRole::UNKNOWN);
        hdr_.stream_roles[3] = static_cast<uint8_t>(StreamRole::UNKNOWN);
        if (has_i2_stream_) {
            hdr_.n_streams = 3;
            hdr_.stream_lengths[2] = cfg.i2_length;
            hdr_.stream_roles[2] = static_cast<uint8_t>(StreamRole::I2);
            hdr_.reserved[0] |= ExtraStreams::HAS_I2;
        }
        if (use_bc_dict_)
            hdr_.flags |= Flags::BC_DICT;
        if (cfg.polya_trim)
            hdr_.flags |= Flags::TRIMMED;
        if (cfg.deduped)
            hdr_.flags |= Flags::DEDUPED;

        // Write header placeholder (will be rewritten at finish)
        if (std::fwrite(&hdr_, sizeof(Header), 1, fp_) != 1) {
            throw std::runtime_error("Failed to write header");
        }
        file_offset_ = sizeof(Header);

        // Write barcode dictionary (compressed)
        if (use_bc_dict_) {
            write_bc_dict(cfg.bc_dict, cfg.bc_length);
        }

        // Reserve block buffers
        if (use_bc_dict_) {
            block_bc_indices_.reserve(cfg.block_size);
            block_umi_seq_.reserve(cfg.block_size * cfg.umi_length);
        }
        block_r1_seq_.reserve(cfg.block_size * 32);   // R1 is short (~28bp)
        block_r2_seq_.reserve(cfg.block_size * 100);  // R2 is ~91bp
        block_r1_lengths_.reserve(cfg.block_size);
        block_r2_lengths_.reserve(cfg.block_size);
        if (has_i2_stream_) {
            block_i2_seq_.reserve(cfg.block_size * cfg.i2_length);
            block_i2_lengths_.reserve(cfg.block_size);
        }

        if (cfg.qual_mode != QualMode::NONE) {
            block_r2_qual_.reserve(cfg.block_size * 100);
        }
        if (cfg.polya_trim) {
            block_trim_lengths_.reserve(cfg.block_size);
        }
        if (cfg.deduped) {
            block_dup_counts_.reserve(cfg.block_size);
        }

        // Create persistent ZSTD compression context for block reuse
        if (cfg.codec == Codec::ZSTD) {
            cctx_ = ZSTD_createCCtx();
            if (!cctx_) throw std::runtime_error("Failed to create ZSTD_CCtx");
            if (cfg.compress_threads > 0) {
                ZSTD_CCtx_setParameter(cctx_, ZSTD_c_nbWorkers,
                                       cfg.compress_threads);
            }
        }
    }

    // Add a 3-read ATAC record: R1 genomic + R3 genomic + I2 barcode.
    // Must only be called when has_i2_stream_=true (cfg.has_i2_stream && cfg.i2_length > 0).
    // r1_seq = R1 genomic (50bp), r3_seq = R3 genomic (49bp), i2_seq = R2 barcode (16bp).
    void add_read_atac(const uint8_t* r1_seq, uint16_t r1_len, const uint8_t* r1_qual,
                       const uint8_t* r3_seq, uint16_t r3_len, const uint8_t* r3_qual,
                       const uint8_t* i2_seq, uint16_t i2_len,
                       uint32_t dup_count = 1) {
        // R1 genomic → stored in block_r1_seq_ (legacy mode: stored in full)
        block_r1_seq_.insert(block_r1_seq_.end(), r1_seq, r1_seq + r1_len);
        block_r1_lengths_.push_back(r1_len);
        // R3 genomic → stored in block_r2_seq_ (R2 stream)
        block_r2_seq_.insert(block_r2_seq_.end(), r3_seq, r3_seq + r3_len);
        block_r2_lengths_.push_back(r3_len);
        if (cfg_.qual_mode != QualMode::NONE && r3_qual) {
            block_r2_qual_.insert(block_r2_qual_.end(), r3_qual, r3_qual + r3_len);
        }
        // I2 barcode → stored in block_i2_seq_
        block_i2_seq_.insert(block_i2_seq_.end(), i2_seq, i2_seq + i2_len);
        block_i2_lengths_.push_back(i2_len);

        if (cfg_.deduped) block_dup_counts_.push_back(dup_count);
        n_reads_in_block_++;
        total_reads_++;
        if (n_reads_in_block_ >= cfg_.block_size) flush_block();
    }

    // Add a single read (paired-end: R1 barcode/UMI + R2 cDNA).
    // Sequences must be byte-numeric (0/1/2/3/4).
    // Quality must be raw phred values.
    // dup_count: duplicate count (1 for non-deduped, >1 for collapsed duplicates)
    void add_read(const uint8_t* r2_seq, uint16_t r2_len,
                  const uint8_t* r2_qual,
                  const uint8_t* r1_seq, uint16_t r1_len,
                  const uint8_t* r1_qual,
                  uint32_t dup_count = 1) {
        // PolyA trimming on R2 3' end
        uint16_t trim_len = 0;
        uint16_t effective_r2_len = r2_len;
        if (cfg_.polya_trim && r2_len > 20) {
            trim_len = detect_polya_tail(r2_seq, r2_len,
                                         cfg_.polya_min_len, cfg_.polya_max_mm);
            effective_r2_len = r2_len - trim_len;
            // Keep at least 20bp so STAR can process the read
            if (effective_r2_len < 20) {
                trim_len = r2_len - 20;
                effective_r2_len = 20;
            }
        }

        // R2 max-length truncation (applied after polyA trim)
        if (cfg_.r2_maxlen > 0 && effective_r2_len > cfg_.r2_maxlen) {
            effective_r2_len = cfg_.r2_maxlen;
        }

        if (use_bc_dict_ && r1_len >= cfg_.bc_offset + cfg_.bc_length) {
            // Split R1 into BC (dict-indexed) + UMI (raw)
            const uint8_t* bc = r1_seq + cfg_.bc_offset;
            uint64_t h = hash_bc(bc, cfg_.bc_length);
            uint32_t bc_idx = bc_lookup_.find(h);
            if (bc_idx != UINT32_MAX) {
                block_bc_indices_.push_back(bc_idx);
            } else {
                // Unknown barcode — use sentinel index (dict size)
                block_bc_indices_.push_back(
                    static_cast<uint32_t>(cfg_.bc_dict.size()));
                // Still need to store the raw BC for unknown barcodes
                block_unknown_bc_.insert(block_unknown_bc_.end(),
                                         bc, bc + cfg_.bc_length);
            }
            // Store UMI
            const uint8_t* umi = r1_seq + cfg_.umi_offset;
            uint16_t umi_len = cfg_.umi_length;
            if (cfg_.umi_offset + umi_len > r1_len)
                umi_len = r1_len - cfg_.umi_offset;
            block_umi_seq_.insert(block_umi_seq_.end(), umi, umi + umi_len);
        } else {
            // No dict: store full R1 as before
            block_r1_seq_.insert(block_r1_seq_.end(), r1_seq, r1_seq + r1_len);
            block_r1_lengths_.push_back(r1_len);
        }

        // Append R2 (possibly trimmed)
        block_r2_seq_.insert(block_r2_seq_.end(), r2_seq, r2_seq + effective_r2_len);
        block_r2_lengths_.push_back(effective_r2_len);

        // Append quality for R2 if needed (also trimmed)
        if (cfg_.qual_mode != QualMode::NONE && r2_qual) {
            block_r2_qual_.insert(block_r2_qual_.end(),
                                  r2_qual, r2_qual + effective_r2_len);
        }

        if (cfg_.polya_trim) {
            block_trim_lengths_.push_back(trim_len);
        }

        if (cfg_.deduped) {
            block_dup_counts_.push_back(dup_count);
        }

        n_reads_in_block_++;
        total_reads_++;

        if (n_reads_in_block_ >= cfg_.block_size) {
            flush_block();
        }
    }

    // Finalize the file: flush remaining reads, write index + footer.
    // metadata_json: optional JSON string stored as compressed metadata block.
    void finish(const std::string& metadata_json = "") {
        if (!fp_) return;

        // Flush any remaining reads
        if (n_reads_in_block_ > 0) {
            flush_block();
        }

        // Wait for last async compress+write to complete
        if (async_flush_.valid()) async_flush_.wait();

        // Write metadata block (compressed JSON)
        uint32_t meta_comp_size = 0;
        if (!metadata_json.empty()) {
            size_t bound = compress::compress_bound(cfg_.codec, metadata_json.size());
            std::vector<uint8_t> comp_buf(bound);
            size_t comp_size = compress::compress_block(
                cfg_.codec, cfg_.codec_level,
                metadata_json.data(), metadata_json.size(),
                comp_buf.data(), bound);
            meta_comp_size = static_cast<uint32_t>(comp_size);
            // Write metadata compressed size + data right after last block
            if (std::fwrite(comp_buf.data(), 1, comp_size, fp_) != comp_size) {
                throw std::runtime_error("Failed to write metadata block");
            }
        }

        // Write block index
        // Use uint64_t to avoid uint32_t overflow for files >4 GiB.
        // ftr.index_offset (uint32_t) will truncate for truly large files, but
        // the reader computes the index position from EOF so it does not rely on
        // the stored ftr.index_offset value for correctness.
        uint64_t index_start = static_cast<uint64_t>(std::ftell(fp_));
        // Offsets
        if (!block_offsets_.empty()) {
            std::fwrite(block_offsets_.data(), sizeof(uint64_t),
                        block_offsets_.size(), fp_);
        }
        // Compressed sizes
        if (!block_comp_sizes_.empty()) {
            std::fwrite(block_comp_sizes_.data(), sizeof(uint32_t),
                        block_comp_sizes_.size(), fp_);
        }

        // Write footer
        Footer ftr;
        ftr.init(static_cast<uint32_t>(block_offsets_.size()), index_start);
        ftr.file_crc32 = file_crc32_;
        if (std::fwrite(&ftr, sizeof(Footer), 1, fp_) != 1) {
            throw std::runtime_error("Failed to write footer");
        }

        // Rewrite header with final counts
        hdr_.block_count = static_cast<uint32_t>(block_offsets_.size());
        hdr_.n_unique = total_reads_;
        hdr_.n_original = total_reads_;
        hdr_.meta_size = meta_comp_size;
        std::fseek(fp_, 0, SEEK_SET);
        if (std::fwrite(&hdr_, sizeof(Header), 1, fp_) != 1) {
            throw std::runtime_error("Failed to rewrite header");
        }

        // Compute file CRC32 over everything before the footer
        uint32_t fcrc = 0;
        long file_size = static_cast<long>(sizeof(Header)) +
                         static_cast<long>(file_offset_ - sizeof(Header));
        // file_offset_ tracks cumulative write position up to end of blocks;
        // metadata + index follow. Re-read from start to footer.
        std::fseek(fp_, 0, SEEK_END);
        long total_file_size = std::ftell(fp_);
        long crc_region = total_file_size - static_cast<long>(sizeof(Footer));
        std::fseek(fp_, 0, SEEK_SET);
        {
            std::vector<uint8_t> buf(65536);
            long remaining = crc_region;
            while (remaining > 0) {
                size_t chunk = static_cast<size_t>(
                    std::min(remaining, static_cast<long>(buf.size())));
                size_t got = std::fread(buf.data(), 1, chunk, fp_);
                if (got == 0) break;
                fcrc = update_crc32(fcrc, buf.data(), got);
                remaining -= static_cast<long>(got);
            }
        }

        // Rewrite footer with computed CRC
        ftr.file_crc32 = fcrc;
        std::fseek(fp_, crc_region, SEEK_SET);
        if (std::fwrite(&ftr, sizeof(Footer), 1, fp_) != 1) {
            throw std::runtime_error("Failed to rewrite footer with CRC");
        }

        std::fclose(fp_);
        fp_ = nullptr;
    }

    uint64_t total_reads_written() const { return total_reads_; }
    uint32_t blocks_written() const {
        return static_cast<uint32_t>(block_offsets_.size());
    }

   private:
    // Sort all per-read vectors by barcode index for better compression.
    // Only effective when using bc_dict (use_bc_dict_ == true).
    void sort_block_by_bc() {
        using Clock = std::chrono::high_resolution_clock;
        auto t_sort_start = Clock::now();
        uint32_t n = n_reads_in_block_;
        if (n <= 1 || !use_bc_dict_ || !cfg_.sort_by_bc) return;

        // Precompute cumulative R2 offsets for variable-length R2 comparison
        std::vector<uint32_t> r2_offsets(n + 1, 0);
        for (uint32_t i = 0; i < n; ++i)
            r2_offsets[i + 1] = r2_offsets[i] + block_r2_lengths_[i];
        const bool has_r2 = !block_r2_seq_.empty();

        // Precompute 64-bit R2 sort key from first 32 bases (2 bits each).
        // This replaces expensive ~75-byte memcmp with a single uint64 compare
        // for ~100% of comparisons (4^32 >> block_size, so ties are negligible).
        std::vector<uint64_t> r2_keys(n, 0);
        if (has_r2) {
            for (uint32_t i = 0; i < n; ++i) {
                uint64_t key = 0;
                const uint8_t* r2 = block_r2_seq_.data() + r2_offsets[i];
                uint16_t len = std::min<uint16_t>(block_r2_lengths_[i], 32);
                for (uint16_t j = 0; j < len; ++j)
                    key = (key << 2) | (r2[j] & 3);
                key <<= 2 * (32 - len);  // left-align short reads
                r2_keys[i] = key;
            }
        }

        // Two-level sort: counting sort on BC index (O(n)), then std::sort
        // within each BC group for R2 key (O(g·log g) where g is group size).
        // This is O(n) for the primary key instead of O(n log n) with std::sort.

        // Phase 1: Counting sort on BC index
        uint32_t sentinel = static_cast<uint32_t>(cfg_.bc_dict.size());
        uint32_t max_idx = sentinel;  // sentinel is the largest BC index
        // Reuse members across blocks to avoid reallocation
        sort_counts_.assign(max_idx + 1, 0);
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t idx = std::min(block_bc_indices_[i], sentinel);
            sort_counts_[idx]++;
        }
        // Prefix sum → group start positions
        sort_group_starts_.resize(max_idx + 2);
        sort_group_starts_[0] = 0;
        for (uint32_t i = 0; i <= max_idx; ++i)
            sort_group_starts_[i + 1] = sort_group_starts_[i] + sort_counts_[i];

        // Scatter into BC-sorted order
        std::vector<uint32_t> perm(n);
        // Copy group_starts for scatter (will be incremented)
        sort_offsets_.assign(sort_group_starts_.begin(), sort_group_starts_.end());
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t idx = std::min(block_bc_indices_[i], sentinel);
            perm[sort_offsets_[idx]++] = i;
        }

        // Phase 2: Within each BC group with >1 read, sort by R2 key
        auto t_after_counting = Clock::now();
        const bool has_umi = !block_umi_seq_.empty() && cfg_.umi_length > 0;
        const uint16_t ul = cfg_.umi_length;
        for (uint32_t bc = 0; bc <= max_idx; ++bc) {
            uint32_t start = sort_group_starts_[bc];
            uint32_t cnt = sort_counts_[bc];
            if (cnt <= 1) continue;
            std::sort(perm.begin() + start, perm.begin() + start + cnt,
                      [&](uint32_t a, uint32_t b) {
                          // Secondary: precomputed R2 key
                          if (r2_keys[a] != r2_keys[b])
                              return r2_keys[a] < r2_keys[b];
                          // Tertiary: full R2 memcmp for rare 32-base prefix tie
                          if (has_r2) {
                              const uint8_t* ra = block_r2_seq_.data() + r2_offsets[a];
                              const uint8_t* rb = block_r2_seq_.data() + r2_offsets[b];
                              uint16_t la = block_r2_lengths_[a];
                              uint16_t lb = block_r2_lengths_[b];
                              uint16_t skip = std::min<uint16_t>(32, std::min(la, lb));
                              uint16_t cmp_len = std::min(la, lb) - skip;
                              if (cmp_len > 0) {
                                  int cmp = std::memcmp(ra + skip, rb + skip, cmp_len);
                                  if (cmp != 0) return cmp < 0;
                              }
                              if (la != lb) return la < lb;
                          }
                          // Quaternary: UMI
                          if (has_umi) {
                              const uint8_t* ua = block_umi_seq_.data() + a * ul;
                              const uint8_t* ub = block_umi_seq_.data() + b * ul;
                              for (uint16_t k = 0; k < ul; ++k) {
                                  if (ua[k] != ub[k]) return ua[k] < ub[k];
                              }
                          }
                          return false;
                      });
        }

        // Check if already sorted (skip work)
        auto t_after_intra = Clock::now();
        bool sorted = true;
        for (uint32_t i = 0; i < n; ++i) {
            if (perm[i] != i) {
                sorted = false;
                break;
            }
        }
        if (sorted) return;

        // Helper: permute a per-read vector in-place via temp copy
        auto permute_vec = [&](auto& vec) {
            if (vec.empty()) return;
            auto tmp = vec;
            for (uint32_t i = 0; i < n; ++i) vec[i] = tmp[perm[i]];
        };

        // Build orig_unknown_bc_idx BEFORE permuting bc_indices
        // (needed for unknown BC permutation later)
        uint32_t sentinel_bc = static_cast<uint32_t>(cfg_.bc_dict.size());
        std::vector<uint32_t> orig_unk_idx;
        bool has_unknown_bc = !block_unknown_bc_.empty() && cfg_.bc_length > 0;
        if (has_unknown_bc) {
            orig_unk_idx.resize(n, UINT32_MAX);
            uint32_t unk_count = 0;
            for (uint32_t k = 0; k < n; ++k) {
                if (block_bc_indices_[k] == sentinel_bc) {
                    orig_unk_idx[k] = unk_count++;
                }
            }
        }

        // 1. Permute per-read scalar vectors
        permute_vec(block_bc_indices_);
        permute_vec(block_r2_lengths_);
        if (!block_trim_lengths_.empty()) permute_vec(block_trim_lengths_);
        if (!block_dup_counts_.empty()) permute_vec(block_dup_counts_);

        // 2. Permute variable-length R2 seq + qual using pre-computed r2_offsets
        // r2_offsets[k] = cumulative R2 offset for original read k (computed at function start)
        {
            size_t total_r2 = block_r2_seq_.size();
            sort_tmp_buf_.resize(total_r2);
            std::memcpy(sort_tmp_buf_.data(), block_r2_seq_.data(), total_r2);

            uint8_t* dst = block_r2_seq_.data();
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t src = perm[i];
                uint32_t off = r2_offsets[src];
                uint32_t len = block_r2_lengths_[i];  // = old_len[perm[i]]
                std::memcpy(dst, sort_tmp_buf_.data() + off, len);
                dst += len;
            }

            if (!block_r2_qual_.empty()) {
                size_t total_q = block_r2_qual_.size();
                sort_tmp_buf_.resize(total_q);
                std::memcpy(sort_tmp_buf_.data(), block_r2_qual_.data(), total_q);

                dst = block_r2_qual_.data();
                for (uint32_t i = 0; i < n; ++i) {
                    uint32_t src = perm[i];
                    uint32_t off = r2_offsets[src];
                    uint32_t len = block_r2_lengths_[i];
                    std::memcpy(dst, sort_tmp_buf_.data() + off, len);
                    dst += len;
                }
            }
        }

        // 3. Permute fixed-length UMI seq (in-place via reusable temp buffer)
        if (!block_umi_seq_.empty() && cfg_.umi_length > 0) {
            uint16_t ul = cfg_.umi_length;
            size_t total_umi = block_umi_seq_.size();
            sort_tmp_buf_.resize(total_umi);
            std::memcpy(sort_tmp_buf_.data(), block_umi_seq_.data(), total_umi);

            uint8_t* dst = block_umi_seq_.data();
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t src = perm[i];
                std::memcpy(dst + i * ul, sort_tmp_buf_.data() + src * ul, ul);
            }
        }

        // 4. Permute unknown BC data (uses pre-computed orig_unk_idx)
        if (has_unknown_bc) {
            uint16_t bl = cfg_.bc_length;
            size_t total_unk = block_unknown_bc_.size();
            sort_tmp_buf_.resize(total_unk);
            std::memcpy(sort_tmp_buf_.data(), block_unknown_bc_.data(), total_unk);

            block_unknown_bc_.clear();
            for (uint32_t i = 0; i < n; ++i) {
                if (block_bc_indices_[i] == sentinel_bc) {
                    uint32_t src = perm[i];
                    uint32_t ui = orig_unk_idx[src];
                    block_unknown_bc_.insert(block_unknown_bc_.end(),
                                             sort_tmp_buf_.data() + ui * bl,
                                             sort_tmp_buf_.data() + ui * bl + bl);
                }
            }
        }
        sort_count_s_ += std::chrono::duration<double>(t_after_counting - t_sort_start).count();
        sort_intra_s_ += std::chrono::duration<double>(t_after_intra - t_after_counting).count();
        sort_permute_s_ += std::chrono::duration<double>(Clock::now() - t_after_intra).count();
    }

    void flush_block() {
        if (n_reads_in_block_ == 0) return;
        auto fb_start = std::chrono::high_resolution_clock::now();

        // Wait for previous async compress+write to complete before reusing
        // back_raw_buf_, comp_buf_, cctx_, fp_, block_offsets_, block_comp_sizes_
        if (async_flush_.valid()) async_flush_.wait();

        auto after_wait = std::chrono::high_resolution_clock::now();

        sort_block_by_bc();

        auto after_sort = std::chrono::high_resolution_clock::now();
        raw_buf_.clear();
        uint32_t n = n_reads_in_block_;

        if (use_bc_dict_) {
            // -- BC dict indices (varints) --
            uint8_t var_buf[10];
            uint32_t unk_offset = 0;
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t idx = block_bc_indices_[i];
                uint32_t vn = pack::encode_varint(idx, var_buf);
                raw_buf_.insert(raw_buf_.end(), var_buf, var_buf + vn);
                if (idx == static_cast<uint32_t>(cfg_.bc_dict.size())) {
                    raw_buf_.insert(raw_buf_.end(),
                                    block_unknown_bc_.data() + unk_offset,
                                    block_unknown_bc_.data() + unk_offset + cfg_.bc_length);
                    unk_offset += cfg_.bc_length;
                }
            }
            // -- UMI column --
            if (cfg_.seq_enc == SeqEncoding::PACKED_2BIT) {
                pack_fixed_column_2bit(block_umi_seq_, cfg_.umi_length, n, raw_buf_);
            } else {
                raw_buf_.insert(raw_buf_.end(),
                                block_umi_seq_.begin(), block_umi_seq_.end());
            }
        } else {
            // -- Full R1 sequences (legacy) --
            if (cfg_.seq_enc == SeqEncoding::PACKED_2BIT) {
                pack_column_2bit(block_r1_seq_, block_r1_lengths_,
                                 cfg_.r1_length == 0, raw_buf_);
            } else {
                raw_buf_.insert(raw_buf_.end(),
                                block_r1_seq_.begin(), block_r1_seq_.end());
            }
        }

        // -- R2 sequences --
        bool r2_variable = cfg_.r2_length == 0 || cfg_.polya_trim;
        if (cfg_.seq_enc == SeqEncoding::PACKED_2BIT) {
            pack_column_2bit(block_r2_seq_, block_r2_lengths_,
                             r2_variable, raw_buf_);
        } else {
            if (r2_variable) {
                // Need length prefix for variable sequences
                uint8_t var_buf[10];
                uint32_t offset = 0;
                for (uint16_t len : block_r2_lengths_) {
                    uint32_t vn = pack::encode_varint(len, var_buf);
                    raw_buf_.insert(raw_buf_.end(), var_buf, var_buf + vn);
                    raw_buf_.insert(raw_buf_.end(),
                                    block_r2_seq_.data() + offset,
                                    block_r2_seq_.data() + offset + len);
                    offset += len;
                }
            } else {
                raw_buf_.insert(raw_buf_.end(),
                                block_r2_seq_.begin(), block_r2_seq_.end());
            }
        }

        // -- R2 quality --
        if (cfg_.qual_mode == QualMode::BINNED4 && !block_r2_qual_.empty()) {
            pack_qual_column(block_r2_qual_, block_r2_lengths_, raw_buf_);
        } else if (cfg_.qual_mode == QualMode::BINNED2 && !block_r2_qual_.empty()) {
            pack_qual_column_binned2(block_r2_qual_, block_r2_lengths_, raw_buf_);
        } else if (cfg_.qual_mode == QualMode::FULL && !block_r2_qual_.empty()) {
            raw_buf_.insert(raw_buf_.end(),
                            block_r2_qual_.begin(), block_r2_qual_.end());
        }

        // -- Trim lengths (if trimming active) --
        if (cfg_.polya_trim && !block_trim_lengths_.empty()) {
            uint8_t var_buf[10];
            for (uint16_t tl : block_trim_lengths_) {
                uint32_t vn = pack::encode_varint(tl, var_buf);
                raw_buf_.insert(raw_buf_.end(), var_buf, var_buf + vn);
            }
        }

        // -- Dup counts (if deduped) --
        if (cfg_.deduped && !block_dup_counts_.empty()) {
            uint8_t var_buf[10];
            for (uint32_t dc : block_dup_counts_) {
                uint32_t vn = pack::encode_varint(dc, var_buf);
                raw_buf_.insert(raw_buf_.end(), var_buf, var_buf + vn);
            }
        }

        // -- I2 barcode stream (scATAC 3-read mode) --
        if (has_i2_stream_ && !block_i2_seq_.empty()) {
            if (cfg_.seq_enc == SeqEncoding::PACKED_2BIT) {
                pack_column_2bit(block_i2_seq_, block_i2_lengths_,
                                 cfg_.i2_length == 0, raw_buf_);
            } else {
                raw_buf_.insert(raw_buf_.end(),
                                block_i2_seq_.begin(), block_i2_seq_.end());
            }
        }

        auto after_pack = std::chrono::high_resolution_clock::now();

        // ── Double-buffered async compress+CRC+write ──
        // Swap raw_buf_ into back buffer so async thread can compress it
        // while main thread continues reading VDB into the front buffer.
        std::swap(raw_buf_, back_raw_buf_);

        async_flush_ = std::async(std::launch::async, [this, n]() {
            auto t0 = std::chrono::high_resolution_clock::now();
            size_t bound = compress::compress_bound(cfg_.codec, back_raw_buf_.size());
            comp_buf_.resize(bound);
            size_t comp_size = compress::compress_block_ctx(
                cfg_.codec, cfg_.codec_level, cctx_,
                back_raw_buf_.data(), back_raw_buf_.size(),
                comp_buf_.data(), bound, cfg_.compress_threads);

            auto t1 = std::chrono::high_resolution_clock::now();
            uint64_t offset = file_offset_;
            block_offsets_.push_back(offset);
            block_comp_sizes_.push_back(static_cast<uint32_t>(comp_size));

            BlockHeader bh;
            bh.n_reads = n;
            bh.compressed_size = static_cast<uint32_t>(comp_size);
            bh.raw_crc32 = compute_crc32(back_raw_buf_.data(), back_raw_buf_.size());

            std::fwrite(&bh, sizeof(BlockHeader), 1, fp_);
            std::fwrite(comp_buf_.data(), 1, comp_size, fp_);
            file_offset_ += sizeof(BlockHeader) + comp_size;

            auto t2 = std::chrono::high_resolution_clock::now();
            flush_compress_s_ += std::chrono::duration<double>(t1 - t0).count();
            flush_write_s_ += std::chrono::duration<double>(t2 - t1).count();
        });

        // Reset block accumulators — main thread continues reading immediately
        block_r1_seq_.clear();
        block_r2_seq_.clear();
        block_r1_lengths_.clear();
        block_r2_lengths_.clear();
        block_r2_qual_.clear();
        block_bc_indices_.clear();
        block_umi_seq_.clear();
        block_unknown_bc_.clear();
        block_trim_lengths_.clear();
        block_dup_counts_.clear();
        block_i2_seq_.clear();
        block_i2_lengths_.clear();
        n_reads_in_block_ = 0;

        auto fb_end = std::chrono::high_resolution_clock::now();
        flush_time_s_ += std::chrono::duration<double>(fb_end - fb_start).count();
        flush_wait_s_ += std::chrono::duration<double>(after_wait - fb_start).count();
        flush_sort_s_ += std::chrono::duration<double>(after_sort - after_wait).count();
        flush_pack_s_ += std::chrono::duration<double>(after_pack - after_sort).count();
    }

    // Pack a column of sequences into 2-bit format.
    void pack_column_2bit(const std::vector<uint8_t>& seq_data,
                          const std::vector<uint16_t>& lengths,
                          bool variable_len,
                          std::vector<uint8_t>& out) {
        uint32_t off = 0;
        uint8_t var_buf[10];
        for (uint16_t len : lengths) {
            if (variable_len) {
                uint32_t vn = pack::encode_varint(len, var_buf);
                out.insert(out.end(), var_buf, var_buf + vn);
            }
            uint32_t pb = pack::packed_bytes(len);
            uint32_t nb = pack::n_bitmap_bytes(len);
            size_t old_size = out.size();
            out.resize(old_size + 1 + pb + nb);
            uint8_t* packed = out.data() + old_size + 1;
            uint8_t* n_bitmap = packed + pb;
            std::memset(n_bitmap, 0, nb);
            bool has_n = pack::pack_2bit(seq_data.data() + off, len,
                                         packed, n_bitmap);
            out[old_size] = has_n ? 1 : 0;
            if (!has_n) out.resize(old_size + 1 + pb);
            off += len;
        }
    }

    // Pack a fixed-length column (e.g., UMI where all reads have same length)
    void pack_fixed_column_2bit(const std::vector<uint8_t>& seq_data,
                                uint16_t fixed_len, uint32_t n_reads,
                                std::vector<uint8_t>& out) {
        uint32_t off = 0;
        for (uint32_t i = 0; i < n_reads; ++i) {
            uint32_t pb = pack::packed_bytes(fixed_len);
            uint32_t nb = pack::n_bitmap_bytes(fixed_len);
            size_t old_size = out.size();
            out.resize(old_size + 1 + pb + nb);
            uint8_t* packed = out.data() + old_size + 1;
            uint8_t* n_bitmap = packed + pb;
            std::memset(n_bitmap, 0, nb);
            bool has_n = pack::pack_2bit(seq_data.data() + off, fixed_len,
                                         packed, n_bitmap);
            out[old_size] = has_n ? 1 : 0;
            if (!has_n) out.resize(old_size + 1 + pb);
            off += fixed_len;
        }
    }

    // Pack quality column using binned-4 encoding
    void pack_qual_column(const std::vector<uint8_t>& qual_data,
                          const std::vector<uint16_t>& lengths,
                          std::vector<uint8_t>& out) {
        uint32_t off = 0;
        for (uint16_t len : lengths) {
            uint32_t pb = pack::packed_bytes(len);
            size_t old_size = out.size();
            out.resize(old_size + pb);
            pack::pack_qual_binned4(qual_data.data() + off, len,
                                    out.data() + old_size);
            off += len;
        }
    }

    // Pack quality column using binned-2 encoding (1 bit/base, pass/fail at Q20)
    void pack_qual_column_binned2(const std::vector<uint8_t>& qual_data,
                                  const std::vector<uint16_t>& lengths,
                                  std::vector<uint8_t>& out) {
        uint32_t off = 0;
        for (uint16_t len : lengths) {
            uint32_t pb = pack::packed_bytes_binned2(len);
            size_t old_size = out.size();
            out.resize(old_size + pb);
            pack::pack_qual_binned2(qual_data.data() + off, len,
                                    out.data() + old_size);
            off += len;
        }
    }

    // Encode lengths as varints
    void encode_lengths(const std::vector<uint16_t>& lengths,
                        std::vector<uint8_t>& out) {
        uint8_t var_buf[10];
        for (uint16_t len : lengths) {
            uint32_t n = pack::encode_varint(len, var_buf);
            out.insert(out.end(), var_buf, var_buf + n);
        }
    }

    // Write barcode dictionary (compressed) after header
    void write_bc_dict(const std::vector<std::vector<uint8_t>>& dict,
                       uint16_t bc_len) {
        // Format: [n_dicts=1][stream=0][segment=0][n_entries][bc_len][seq_data]
        std::vector<uint8_t> raw;
        raw.push_back(1);  // n_dictionaries
        raw.push_back(0);  // stream_index (R1)
        raw.push_back(0);  // segment_index (BC)
        uint32_t n_entries = static_cast<uint32_t>(dict.size());
        uint8_t var_buf[10];
        uint32_t vn = pack::encode_varint(n_entries, var_buf);
        raw.insert(raw.end(), var_buf, var_buf + vn);
        raw.push_back(static_cast<uint8_t>(bc_len));
        // Concatenated barcode sequences
        for (const auto& bc : dict) {
            raw.insert(raw.end(), bc.begin(), bc.end());
        }

        // Compress dictionary
        size_t bound = compress::compress_bound(cfg_.codec, raw.size());
        std::vector<uint8_t> comp(bound);
        size_t comp_size = compress::compress_block(
            cfg_.codec, cfg_.codec_level,
            raw.data(), raw.size(), comp.data(), bound);

        // Write compressed size + data
        uint32_t cs32 = static_cast<uint32_t>(comp_size);
        std::fwrite(&cs32, sizeof(uint32_t), 1, fp_);
        std::fwrite(comp.data(), 1, comp_size, fp_);
        file_offset_ += sizeof(uint32_t) + comp_size;
        hdr_.bc_dict_size = cs32;
    }

    // Hash a barcode for dictionary lookup (FNV-1a)
    static uint64_t hash_bc(const uint8_t* seq, uint16_t len) {
        uint64_t h = 14695981039346656037ULL;
        for (uint16_t i = 0; i < len; ++i) {
            h ^= seq[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    // Detect polyA tail at 3' end of a sequence.
    // Returns number of bases to trim (0 if no polyA found).
    static uint16_t detect_polya_tail(const uint8_t* seq, uint16_t len,
                                      uint8_t min_len, uint8_t max_mm) {
        // Scan backwards from 3' end
        uint16_t a_run = 0;
        uint8_t mm = 0;
        uint16_t trim_pos = len;

        for (int16_t i = static_cast<int16_t>(len) - 1; i >= 0; --i) {
            if (seq[i] == 0) {  // A = 0
                a_run++;
            } else {
                mm++;
                if (mm > max_mm) break;
                a_run++;
            }
            trim_pos = static_cast<uint16_t>(i);
        }

        if (a_run >= min_len) {
            return len - trim_pos;
        }
        return 0;
    }

    WriterConfig cfg_;
    FILE* fp_ = nullptr;
    ZSTD_CCtx* cctx_ = nullptr;
    Header hdr_;
    bool use_bc_dict_ = false;
    bool has_i2_stream_ = false;
    FlatBCMap bc_lookup_;

    // Block accumulation
    std::vector<uint8_t> block_r1_seq_;
    std::vector<uint8_t> block_r2_seq_;
    std::vector<uint8_t> block_r2_qual_;
    std::vector<uint16_t> block_r1_lengths_;
    std::vector<uint16_t> block_r2_lengths_;
    std::vector<uint32_t> block_bc_indices_;
    std::vector<uint8_t> block_umi_seq_;
    std::vector<uint8_t> block_unknown_bc_;
    std::vector<uint16_t> block_trim_lengths_;
    std::vector<uint32_t> block_dup_counts_;
    // I2 stream (scATAC barcode, stored as 3rd stream when has_i2_stream_=true)
    std::vector<uint8_t> block_i2_seq_;
    std::vector<uint16_t> block_i2_lengths_;
    uint32_t n_reads_in_block_ = 0;

    // File state
    uint64_t file_offset_ = 0;
    uint64_t total_reads_ = 0;
    uint32_t file_crc32_ = 0;
    std::vector<uint64_t> block_offsets_;
    std::vector<uint32_t> block_comp_sizes_;

   public:
    // Accumulated wall-clock time spent in flush_block() (sort+compress+write)
    double flush_time_s() const { return flush_time_s_; }
    double flush_wait_s() const { return flush_wait_s_; }
    double flush_sort_s() const { return flush_sort_s_; }
    double flush_pack_s() const { return flush_pack_s_; }
    double flush_compress_s() const { return flush_compress_s_; }
    double flush_write_s() const { return flush_write_s_; }
    double sort_count_s() const { return sort_count_s_; }
    double sort_intra_s() const { return sort_intra_s_; }
    double sort_permute_s() const { return sort_permute_s_; }

   private:
    double flush_time_s_ = 0.0;
    double flush_wait_s_ = 0.0;
    double flush_sort_s_ = 0.0;
    double flush_pack_s_ = 0.0;
    double flush_compress_s_ = 0.0;
    double flush_write_s_ = 0.0;
    // Sort sub-phases (debug profiling)
    double sort_count_s_ = 0.0;    // counting + prefix sum + scatter
    double sort_intra_s_ = 0.0;    // intra-group std::sort
    double sort_permute_s_ = 0.0;  // data permutation

    // Temporary buffers (reused across blocks)
    std::vector<uint8_t> raw_buf_;
    std::vector<uint8_t> back_raw_buf_;  // Double-buffer for async compress
    std::future<void> async_flush_;      // Pending async compress+write
    std::vector<uint8_t> comp_buf_;
    std::vector<uint32_t> sort_counts_;        // Counting sort workspace
    std::vector<uint32_t> sort_group_starts_;  // Group start positions
    std::vector<uint32_t> sort_offsets_;       // Scatter offsets (temp)
    std::vector<uint8_t> sort_tmp_buf_;        // Reusable temp buffer for permutation
};

}  // namespace lib1fq
