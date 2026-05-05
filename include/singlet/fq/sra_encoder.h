// lib1fq/sra_encoder.h — Streaming SRA → .1fq encoder (direct VDB path)
//
// Zero-copy VDB cursor → byte-numeric conversion → .1fq blocks.
// No FASTQ text formatting, no fmemopen, no re-parsing.
//
// Data flow:
//   VCursorCellDataDirect → ASCII DNA ptr → ascii_to_num() inline → Writer::add_read()
//   VCursorCellDataDirect → raw phred ptr → direct pass-through → Writer::add_read()
//
// Protocol auto-detection uses first 100K spots (buffered in SpotData).
// Remaining spots stream directly: VDB read → convert → compress → disk.

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lib1fq.h"
#include "sra_reader.h"

namespace lib1fq {

// ── Profiling timer ──

struct ProfileTimers {
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;
    double vdb_read_s = 0.0;    // VCursorCellDataDirect calls
    double convert_s = 0.0;     // ASCII→numeric + quality handling
    double writer_s = 0.0;      // add_read (packing, buffering, compress, flush)
    double protocol_s = 0.0;    // Auto-detection phase
    double finalize_s = 0.0;    // finish() — metadata, index, CRC
    double total_s = 0.0;       // wall clock
    uint64_t vdb_bytes = 0;     // Total bytes read from VDB
    uint64_t output_bytes = 0;  // .1fq file size
    // Writer sub-phases
    double w_wait_s = 0.0;  // Time waiting for async compress+write
    double w_sort_s = 0.0;
    double w_pack_s = 0.0;
    double w_compress_s = 0.0;  // Async: actual compress CPU time
    double w_write_s = 0.0;     // Async: actual CRC+write CPU time

    void print(uint64_t total_reads, FILE* out = stderr) const {
        double M = 1e6;
        double GB = 1e9;
        double reads_M = total_reads / M;
        double vdb_GB = vdb_bytes / GB;
        double out_GB = output_bytes / GB;
        double other_s = total_s - vdb_read_s - convert_s - writer_s - protocol_s - finalize_s;
        if (other_s < 0) other_s = 0;

        std::fprintf(out, "\n[1fq-profile] ── Encode Performance ──\n");
        std::fprintf(out, "[1fq-profile] Total: %.3fs, %.2fM reads, %.2f GB VDB → %.2f GB .1fq (%.1f×)\n",
                     total_s, reads_M, vdb_GB, out_GB,
                     vdb_GB > 0 ? vdb_GB / out_GB : 0.0);
        std::fprintf(out, "[1fq-profile] Throughput: %.2fM reads/s, %.1f MB/s VDB, %.1f MB/s .1fq\n",
                     total_s > 0 ? reads_M / total_s : 0,
                     total_s > 0 ? vdb_GB * 1000.0 / total_s : 0,
                     total_s > 0 ? out_GB * 1000.0 / total_s : 0);
        std::fprintf(out, "[1fq-profile] Phase breakdown:\n");
        std::fprintf(out, "[1fq-profile]   VDB read:    %6.3fs (%4.1f%%)\n",
                     vdb_read_s, 100.0 * vdb_read_s / total_s);
        std::fprintf(out, "[1fq-profile]   Convert:     %6.3fs (%4.1f%%)\n",
                     convert_s, 100.0 * convert_s / total_s);
        std::fprintf(out, "[1fq-profile]   Writer:      %6.3fs (%4.1f%%) [sort+pack sync, compress+write async]\n",
                     writer_s, 100.0 * writer_s / total_s);
        if (w_compress_s > 0) {
            std::fprintf(out, "[1fq-profile]     wait:      %6.3fs (%4.1f%%) [stalled on prev async]\n",
                         w_wait_s, 100.0 * w_wait_s / total_s);
            std::fprintf(out, "[1fq-profile]     sort:      %6.3fs (%4.1f%%)\n",
                         w_sort_s, 100.0 * w_sort_s / total_s);
            std::fprintf(out, "[1fq-profile]     pack:      %6.3fs (%4.1f%%)\n",
                         w_pack_s, 100.0 * w_pack_s / total_s);
            double async_total = w_compress_s + w_write_s;
            double hidden = async_total - w_wait_s;
            if (hidden < 0) hidden = 0;
            std::fprintf(out, "[1fq-profile]     compress:  %6.3fs (async, %.1fs hidden)\n",
                         w_compress_s, w_compress_s > w_wait_s ? w_compress_s : w_compress_s);
            std::fprintf(out, "[1fq-profile]     write:     %6.3fs (async)\n", w_write_s);
            std::fprintf(out, "[1fq-profile]     overlap:   %6.3fs hidden behind VDB read\n", hidden);
        }
        std::fprintf(out, "[1fq-profile]   Protocol:    %6.3fs (%4.1f%%)\n",
                     protocol_s, 100.0 * protocol_s / total_s);
        std::fprintf(out, "[1fq-profile]   Finalize:    %6.3fs (%4.1f%%) [metadata+index+CRC]\n",
                     finalize_s, 100.0 * finalize_s / total_s);
        std::fprintf(out, "[1fq-profile]   Other:       %6.3fs (%4.1f%%)\n",
                     other_s, 100.0 * other_s / total_s);
    }
};

// ── SRA → .1fq streaming encoder ──

class SraEncoder {
   public:
    struct Stats {
        uint64_t total_reads = 0;
        uint32_t blocks_written = 0;
        std::string protocol_tag;
        Confidence confidence = Confidence::NONE;
        AssayType assay_type = AssayType::UNKNOWN;
        double wl_match_rate = 0.0;
        int exit_code = 0;  // 0 = success, 2 = data_incomplete (barcode-stripped)
        ProfileTimers profile;
    };

   private:
    // Buffered spot for probe phase — stores converted numeric data
    struct SpotData {
        std::vector<uint8_t> r1_seq, r2_seq;
        std::vector<uint8_t> r1_qual, r2_qual;
        uint16_t r1_len = 0, r2_len = 0;
        // ATAC 3-read: barcode from 3rd (shortest) segment; empty for 2-segment reads.
        std::vector<uint8_t> bc_seq;
        uint16_t bc_len = 0;
    };

    // Compact buffered read for parallel VDB reading.
    // Fixed-length layout avoids per-read allocation.
    struct BufferedRead {
        uint16_t r1_len;
        uint16_t r2_len;
        // r1_seq, r2_seq, r1_qual, r2_qual stored contiguously in a flat buffer
    };

    // Flat buffer holding N reads' data contiguously.
    // Layout per read: [r1_seq(r1l)][r2_seq(r2l)][r1_qual(r1l)][r2_qual(r2l)]
    struct ReadBatch {
        std::vector<BufferedRead> reads;
        std::vector<uint8_t> data;  // flat contiguous storage
        bool has_qual = false;
        uint16_t fixed_r1l = 0, fixed_r2l = 0;  // 0 = variable length

        void reserve(uint32_t n_reads, uint16_t r1l, uint16_t r2l, bool qual) {
            reads.reserve(n_reads);
            uint32_t bytes_per_read = r1l + r2l;
            if (qual) bytes_per_read *= 2;
            data.reserve(n_reads * bytes_per_read);
            has_qual = qual;
            fixed_r1l = r1l;
            fixed_r2l = r2l;
        }

        void clear() {
            reads.clear();
            data.clear();
        }

        // Add a read with pre-converted numeric sequence data
        void add(const uint8_t* r1_seq, uint16_t r1l,
                 const uint8_t* r2_seq, uint16_t r2l,
                 const uint8_t* r1_qual, const uint8_t* r2_qual) {
            reads.push_back({r1l, r2l});
            data.insert(data.end(), r1_seq, r1_seq + r1l);
            data.insert(data.end(), r2_seq, r2_seq + r2l);
            if (has_qual) {
                data.insert(data.end(), r1_qual, r1_qual + r1l);
                if (r2_qual) data.insert(data.end(), r2_qual, r2_qual + r2l);
            }
        }

        // Get pointers for read i
        void get(uint32_t i, const uint8_t*& r1s, const uint8_t*& r2s,
                 const uint8_t*& r1q, const uint8_t*& r2q) const {
            uint16_t r1l = reads[i].r1_len;
            uint16_t r2l = reads[i].r2_len;
            uint32_t stride = r1l + r2l;
            if (has_qual) stride *= 2;
            uint32_t off = i * stride;  // only works for fixed-length
            r1s = data.data() + off;
            r2s = r1s + r1l;
            if (has_qual) {
                r1q = r2s + r2l;
                r2q = r1q + r1l;
            } else {
                r1q = nullptr;
                r2q = nullptr;
            }
        }
    };

    // Thread function: read a chunk of spots from VDB into a ReadBatch
    static void vdb_reader_thread(const std::string& sra_path,
                                  int64_t start_row, int64_t end_row,
                                  uint16_t r1l, uint16_t r2l,
                                  int32_t r1_off, int32_t r2_off,
                                  uint32_t min_bases, bool need_qual,
                                  ReadBatch& batch) {
        SraReader rdr;
        rdr.open(sra_path.c_str());
        rdr.seek(start_row);
        rdr.set_row_limit(end_row);

        uint32_t n_spots = static_cast<uint32_t>(end_row - start_row);
        batch.reserve(n_spots, r1l, r2l, need_qual);

        SraReader::RawSpot raw;
        std::vector<uint8_t> r1_buf(r1l), r2_buf(r2l);

        while (!rdr.done_range()) {
            if (!rdr.read_next_spot_fast(raw)) continue;
            if (raw.total_bases < min_bases) continue;

            // Convert ASCII → numeric
            for (uint16_t i = 0; i < r1l; ++i)
                r1_buf[i] = nuc::ascii_to_num(raw.read[r1_off + i]);
            for (uint16_t i = 0; i < r2l; ++i)
                r2_buf[i] = nuc::ascii_to_num(raw.read[r2_off + i]);

            const uint8_t* r1q = need_qual ? raw.qual + r1_off : nullptr;
            const uint8_t* r2q = need_qual ? raw.qual + r2_off : nullptr;

            batch.add(r1_buf.data(), r1l, r2_buf.data(), r2l, r1q, r2q);
        }
    }

    // Scratch buffers for streaming phase (avoid per-read allocation)
    std::vector<uint8_t> r1_buf_, r2_buf_, bc_buf_;  // bc_buf_ for ATAC 3-read barcode
    std::vector<uint8_t> r2_qual_buf_;  // qual pad buffer for fixed-length enforcement

    // Convert ASCII DNA to byte-numeric into dest buffer
    static void ascii_to_numeric(const char* src, uint16_t len,
                                 std::vector<uint8_t>& dest) {
        dest.resize(len);
        for (uint16_t i = 0; i < len; ++i)
            dest[i] = nuc::ascii_to_num(src[i]);
    }

    // Detect adapter contamination in R2 by looking for positions where one
    // base dominates (>threshold frequency), indicating a fixed adapter sequence.
    // Only valid for 5'-capture protocols (TSO readthrough); false-positive rate
    // is high for 3' protocols where polyA in short inserts mirrors the pattern.
    // Returns the trim position (r2_maxlen), or 0 if no adapter detected.
    // threshold: fraction of reads at which a base must dominate (default 0.75).
    static uint16_t detect_r2_adapter(const std::vector<SpotData>& spots,
                                       double threshold = 0.75) {
        if (spots.size() < 100) return 0;

        uint16_t max_r2 = 0;
        for (const auto& s : spots)
            if (s.r2_len > max_r2) max_r2 = s.r2_len;
        if (max_r2 < 40) return 0;

        const int MIN_RUN = 5;
        const int MIN_POS = 30;

        int first_fixed = -1;
        int run = 0;

        for (int p = MIN_POS; p < static_cast<int>(max_r2); ++p) {
            int counts[5] = {};
            int total = 0;
            for (const auto& s : spots) {
                if (p >= static_cast<int>(s.r2_len)) continue;
                uint8_t b = s.r2_seq[p];
                if (b < 5) counts[b]++;
                total++;
            }
            if (total < 50) continue;

            int max_cnt = *std::max_element(counts, counts + 4);
            if (static_cast<double>(max_cnt) / total > threshold) {
                if (++run >= MIN_RUN && first_fixed < 0)
                    first_fixed = p - MIN_RUN + 1;
            } else {
                run = 0;
            }
        }

        return (first_fixed >= MIN_POS)
                   ? static_cast<uint16_t>(first_fixed)
                   : 0;
    }

    // Detect constant 5' prefix in R2 (cDNA) reads:
    // positions 0..N-1 where ≥threshold fraction of reads share the same base.
    // Returns the clip5p_length (0 if no substantial constant prefix found).
    // min_len: minimum prefix length to report (default: 20bp).
    // threshold: fraction of reads at which a base must dominate (default: 0.80).
    // Rejects pure polyA (base=A) or polyT (base=T) prefixes to avoid false
    // positives on 3' inserts with short polyA tails.
    static uint16_t detect_r2_constant_prefix(
        const std::vector<SpotData>& probe_spots,
        uint16_t min_len   = 20,
        double   threshold = 0.80)
    {
        size_t n = std::min(probe_spots.size(), static_cast<size_t>(1000));
        if (n < 10) return 0;

        uint16_t max_r2 = 0;
        for (size_t i = 0; i < n; ++i)
            if (probe_spots[i].r2_len > max_r2) max_r2 = probe_spots[i].r2_len;
        if (max_r2 < static_cast<uint16_t>(min_len)) return 0;

        // Scan from position 0 up to min(100, max_r2) looking for a constant prefix
        uint16_t scan_end = std::min<uint16_t>(max_r2, 100);

        int prefix_len    = 0;
        int polyx_count   = 0;  // positions whose dominant base is A (0) or T (3)
        for (uint16_t p = 0; p < scan_end; ++p) {
            int counts[5] = {};  // 0=A,1=C,2=G,3=T,4=N
            int total     = 0;
            for (size_t i = 0; i < n; ++i) {
                const auto& sd = probe_spots[i];
                if (p >= sd.r2_len) continue;
                uint8_t b = sd.r2_seq[p];
                if (b < 5) counts[b]++;
                total++;
            }
            if (total < 5) break;
            int max_cnt = *std::max_element(counts, counts + 5);
            if (static_cast<double>(max_cnt) / total >= threshold) {
                prefix_len = static_cast<int>(p) + 1;
                // Track dominant base for polyA/polyT check
                int dom = static_cast<int>(
                    std::max_element(counts, counts + 5) - counts);
                if (dom == 0 || dom == 3) ++polyx_count;  // A or T
            } else {
                break;  // prefix ends here
            }
        }

        if (prefix_len < static_cast<int>(min_len)) return 0;

        // Reject if the prefix is predominantly polyA or polyT
        // (>80% of prefix positions dominated by A or T)
        if (prefix_len > 0 &&
            static_cast<double>(polyx_count) / prefix_len > 0.80)
            return 0;

        return static_cast<uint16_t>(prefix_len);
    }

   public:
    Stats encode(const std::string& sra_path, const EncoderConfig& cfg) {
        using Clock = std::chrono::high_resolution_clock;
        auto wall_start = Clock::now();

        Stats stats;
        ProfileTimers& prof = stats.profile;

        // ── Open VDB ──
        auto t0 = Clock::now();
        SraReader reader;
        reader.open(sra_path.c_str());
        uint64_t total_spots = reader.total_spots();
        prof.vdb_read_s += std::chrono::duration<double>(Clock::now() - t0).count();

        // ── Phase 1: Probe first 10K spots (buffered for protocol detection) ──
        uint32_t probe_count = std::min<uint64_t>(10000, total_spots);
        std::vector<SpotData> probe_spots;
        probe_spots.reserve(probe_count);

        SraReader::RawSpot raw;
        // Track segment offsets for fast VDB path
        int32_t probe_r1_offset = -1, probe_r2_offset = -1;
        bool probe_offsets_consistent = true;
        // Track whether any probe spot had ≥3 segments (orientation is ambiguous)
        bool had_multi_segment_spots = false;
        // Whether probe-phase orientation swap was applied for ≥3 segment spots
        bool three_seg_swapped = false;
        // ATAC 3-read: track whether ALL probe spots had exactly 3 segments + bc length
        bool all_3seg_atac = true;           // stays true only while all spots are 3-seg
        uint16_t probe_bc_len = 0;           // barcode length from 3rd segment (0 = not seen yet)
        int32_t probe_bc_offset = -1;        // barcode start offset in concatenated spot

        for (uint32_t i = 0; i < probe_count; ++i) {
            t0 = Clock::now();
            if (!reader.read_next_spot(raw)) break;
            prof.vdb_read_s += std::chrono::duration<double>(Clock::now() - t0).count();

            t0 = Clock::now();
            SpotData sd;
            if (raw.n_segments >= 3) {
                // Multiple segments (e.g. index + R1 + R2).
                // Pick the two longest segments as biological reads,
                // preserving their original order.
                had_multi_segment_spots = true;
                uint32_t bio[2] = {0, 0};
                int n_bio = 0;
                // Find two longest segments
                uint32_t sorted_idx[16];
                uint32_t ns = std::min<uint32_t>(raw.n_segments, 16);
                for (uint32_t s = 0; s < ns; ++s) sorted_idx[s] = s;
                std::sort(sorted_idx, sorted_idx + ns,
                          [&](uint32_t a, uint32_t b) {
                              return raw.seg_len[a] > raw.seg_len[b];
                          });
                bio[0] = std::min(sorted_idx[0], sorted_idx[1]);
                bio[1] = std::max(sorted_idx[0], sorted_idx[1]);
                sd.r1_len = static_cast<uint16_t>(raw.seg_len[bio[0]]);
                sd.r2_len = static_cast<uint16_t>(raw.seg_len[bio[1]]);
                // Track offsets for fast VDB path
                {
                    int32_t r1o = raw.seg_start[bio[0]];
                    int32_t r2o = raw.seg_start[bio[1]];
                    if (probe_r1_offset < 0) {
                        probe_r1_offset = r1o;
                        probe_r2_offset = r2o;
                    } else if (r1o != probe_r1_offset || r2o != probe_r2_offset) {
                        probe_offsets_consistent = false;
                    }
                }
                ascii_to_numeric(raw.read + raw.seg_start[bio[0]], sd.r1_len, sd.r1_seq);
                ascii_to_numeric(raw.read + raw.seg_start[bio[1]], sd.r2_len, sd.r2_seq);
                sd.r1_qual.assign(raw.qual + raw.seg_start[bio[0]],
                                  raw.qual + raw.seg_start[bio[0]] + sd.r1_len);
                sd.r2_qual.assign(raw.qual + raw.seg_start[bio[1]],
                                  raw.qual + raw.seg_start[bio[1]] + sd.r2_len);
                prof.vdb_bytes += (sd.r1_len + sd.r2_len) * 2;

                // ATAC 3-read: capture barcode from the 3rd (shortest) segment.
                // For 10x ATAC: sorted_idx[2] is the 16bp barcode segment.
                // Only save if exactly 3 segments and shortests is barcode-sized (8–32bp).
                if (raw.n_segments == 3 && ns >= 3) {
                    uint32_t bc_seg = sorted_idx[2];  // shortest = barcode for ATAC
                    uint16_t bc_l = static_cast<uint16_t>(raw.seg_len[bc_seg]);
                    if (bc_l >= 8 && bc_l <= 32) {  // valid barcode length range
                        ascii_to_numeric(raw.read + raw.seg_start[bc_seg], bc_l, sd.bc_seq);
                        sd.bc_len = bc_l;
                        // Track barcode offset for fast VDB path (ATAC 3-seg)
                        int32_t bc_off = raw.seg_start[bc_seg];
                        if (probe_bc_len == 0) {
                            probe_bc_len = bc_l;
                            probe_bc_offset = bc_off;
                        } else if (bc_l != probe_bc_len || bc_off != probe_bc_offset) {
                            probe_bc_len = 0;  // inconsistent → disable fast bc capture
                        }
                    } else {
                        all_3seg_atac = false;  // 3rd segment not barcode-sized
                    }
                } else {
                    all_3seg_atac = false;  // >3 segments — not standard ATAC
                }
            } else if (raw.n_segments >= 2 && raw.seg_len[0] > 0 && raw.seg_len[1] > 0) {
                sd.r1_len = static_cast<uint16_t>(raw.seg_len[0]);
                sd.r2_len = static_cast<uint16_t>(raw.seg_len[1]);
                // Track offsets for fast VDB path
                {
                    int32_t r1o = raw.seg_start[0];
                    int32_t r2o = raw.seg_start[1];
                    if (probe_r1_offset < 0) {
                        probe_r1_offset = r1o;
                        probe_r2_offset = r2o;
                    } else if (r1o != probe_r1_offset || r2o != probe_r2_offset) {
                        probe_offsets_consistent = false;
                    }
                }
                ascii_to_numeric(raw.read + raw.seg_start[0], sd.r1_len, sd.r1_seq);
                ascii_to_numeric(raw.read + raw.seg_start[1], sd.r2_len, sd.r2_seq);
                sd.r1_qual.assign(raw.qual + raw.seg_start[0],
                                  raw.qual + raw.seg_start[0] + sd.r1_len);
                sd.r2_qual.assign(raw.qual + raw.seg_start[1],
                                  raw.qual + raw.seg_start[1] + sd.r2_len);
                prof.vdb_bytes += (sd.r1_len + sd.r2_len) * 2;
                all_3seg_atac = false;  // 2-segment spot → not 3-read ATAC
            } else if (raw.n_segments >= 2 && raw.seg_len[0] == 0 && raw.seg_len[1] > 0) {
                // Empty first segment (e.g. celseq2): all data in second segment
                // Treat as single concat read in r1
                sd.r1_len = static_cast<uint16_t>(raw.seg_len[1]);
                ascii_to_numeric(raw.read + raw.seg_start[1], sd.r1_len, sd.r1_seq);
                sd.r1_qual.assign(raw.qual + raw.seg_start[1],
                                  raw.qual + raw.seg_start[1] + sd.r1_len);
                prof.vdb_bytes += sd.r1_len * 2;
                all_3seg_atac = false;
            } else if (raw.n_segments >= 1 && raw.seg_len[0] > 0) {
                sd.r1_len = static_cast<uint16_t>(raw.seg_len[0]);
                ascii_to_numeric(raw.read + raw.seg_start[0], sd.r1_len, sd.r1_seq);
                sd.r1_qual.assign(raw.qual + raw.seg_start[0],
                                  raw.qual + raw.seg_start[0] + sd.r1_len);
                prof.vdb_bytes += sd.r1_len * 2;
            } else {
                prof.convert_s += std::chrono::duration<double>(Clock::now() - t0).count();
                continue;
            }
            prof.convert_s += std::chrono::duration<double>(Clock::now() - t0).count();
            probe_spots.push_back(std::move(sd));
        }

        // ── Phase 1b: Validate probe data ──
        auto validation = validate_probe_data(probe_spots, total_spots);
        if (!validation.valid) {
            std::cerr << "[1fq-encode] ABORT: " << validation.reason << "\n";
            stats.protocol_tag = "INVALID";
            stats.confidence = Confidence::NONE;
            prof.protocol_s += std::chrono::duration<double>(Clock::now() - t0).count();
            prof.total_s = std::chrono::duration<double>(
                               Clock::now() - wall_start)
                               .count();
            return stats;
        }
        if (validation.suspect) {
            std::cerr << "[1fq-encode] WARNING: " << validation.reason << "\n";
        }

        // ── Phase 2: Auto-detect protocol ──
        t0 = Clock::now();

        uint16_t r1_len = 0, r2_len = 0;
        uint8_t protocol_id = 0;
        Confidence confidence = Confidence::NONE;
        std::string protocol_tag = "UNKNOWN";
        ProtocolCandidate detected_proto =
            {"UNKNOWN", 0, 0, 0, 0, 0, 0, 0.0, 0.0, Confidence::NONE};

        if (!probe_spots.empty()) {
            r1_len = probe_spots[0].r1_len;
            r2_len = probe_spots[0].r2_len;
            bool r1_const = true, r2_const = true;
            uint16_t r1_max = 0;
            for (const auto& s : probe_spots) {
                if (s.r1_len != r1_len) r1_const = false;
                if (s.r2_len != r2_len) r2_const = false;
                if (s.r1_len > r1_max) r1_max = s.r1_len;
            }
            if (!r1_const) {
                // Variable-length reads: use max for candidate filtering
                // but signal variable with special handling
                r1_len = r1_max;
            }
            if (!r2_const) r2_len = 0;
        }

        if (!cfg.protocol_tag.empty()) {
            const CandidateSpec* spec = find_protocol_spec(cfg.protocol_tag);
            if (spec) {
                protocol_tag = spec->tag;
                protocol_id = spec->protocol_id;
                detected_proto.tag = spec->tag;
                detected_proto.protocol_id = spec->protocol_id;
                // Only apply R1 trim for CB_UMI_Simple protocols.
                // CB_UMI_Complex protocols (bd-rhapsody, sci-rna-seq3, indrop,
                // ddseq, splitseq, microwell-seq, surecell) store composite
                // barcodes via soloCBposition offsets; trimming R1 to r1_len
                // corrupts those offsets and causes duplicate --soloCBposition
                // flags in the STAR command.
                static const std::unordered_set<std::string> complex_tags = {
                    "bd-rhapsody", "sci-rna-seq3", "indrop", "ddseq",
                    "splitseq", "microwell-seq", "surecell"
                };
                if (complex_tags.find(spec->tag) == complex_tags.end()) {
                    detected_proto.r1_length = spec->r1_len;  // R1 trim for CB_UMI_Simple only
                }
                detected_proto.bc_offset = spec->bc_offset;
                detected_proto.bc_length = spec->bc_len;
                detected_proto.umi_offset = spec->umi_offset;
                detected_proto.umi_length = spec->umi_len;
                detected_proto.confidence = Confidence::MANUAL;

                // ── Read orientation check for MANUAL protocol ──
                // Some SRAs store cDNA as spot.1 (R1) and barcode as spot.2 (R2),
                // opposite the .1fq convention (R1=barcode, R2=cDNA).
                // Use protocol-detection scoring on both orientations to decide
                // which orientation puts the barcode in R1.  Simple length
                // heuristics fail when both reads are the same length (e.g. 2×148bp
                // sci-RNA-seq3), so we score the library structure directly.
                if (!probe_spots.empty() && probe_spots[0].r2_len > 0) {
                    auto orig_score = detect_protocol(probe_spots, r1_len, r2_len,
                                                      cfg.whitelist_dirs);
                    // Try swapped orientation
                    for (auto& sd : probe_spots) {
                        std::swap(sd.r1_seq, sd.r2_seq);
                        std::swap(sd.r1_qual, sd.r2_qual);
                        std::swap(sd.r1_len, sd.r2_len);
                    }
                    std::swap(r1_len, r2_len);
                    auto swap_score = detect_protocol(probe_spots, r1_len, r2_len,
                                                      cfg.whitelist_dirs);
                    if (swap_score.score > orig_score.score) {
                        // Swapped orientation scores better — keep swap
                        std::swap(probe_r1_offset, probe_r2_offset);
                        if (had_multi_segment_spots) three_seg_swapped = true;
                        std::cerr << "[1fq-encode] R1\u2194R2 orientation swap for "
                                  << spec->tag << " (score "
                                  << orig_score.score << " \u2192 "
                                  << swap_score.score << ")\n";
                    } else {
                        // Original orientation is better — swap back
                        for (auto& sd : probe_spots) {
                            std::swap(sd.r1_seq, sd.r2_seq);
                            std::swap(sd.r1_qual, sd.r2_qual);
                            std::swap(sd.r1_len, sd.r2_len);
                        }
                        std::swap(r1_len, r2_len);
                    }
                }
            } else {
                // Unknown tag — store as-is
                protocol_tag = cfg.protocol_tag;
            }
            confidence = Confidence::MANUAL;
        } else {
            detected_proto = detect_protocol(probe_spots, r1_len, r2_len,
                                             cfg.whitelist_dirs);

            // ── Pass 1.5: Hard geometry swap ─────────────────────────────────────────
            // AUTOFIX-E2E-A2-READ-SWAP: when R1 is long cDNA (>50bp) and R2 is short
            // enough to be a barcode+UMI (≤34bp), and R1 > 2×R2, force the swap
            // unconditionally regardless of whitelist evidence.  This geometry is
            // unambiguous for all droplet sc-RNA-seq protocols: barcode+UMI reads are
            // never >34bp.  Without this rule, a deposit with an absent or low-hit
            // whitelist leaves the swap unfired and sends cDNA to STAR as the barcode
            // read (→ 0 cells).  Example: SRR34789664 (mouse 10x-3p-v3, 5M reads).
            bool geometry_swap_fired = false;
            if (should_hard_geometry_swap(r1_len, r2_len) &&
                detected_proto.confidence < Confidence::HIGH) {
                std::cerr << "[1fq-encode] Hard geometry swap: R1=" << r1_len
                          << "bp(cDNA)\u2194R2=" << r2_len
                          << "bp(barcode) — forced by asymmetric read lengths"
                          << " (AUTOFIX-E2E-A2-READ-SWAP)\n";
                for (auto& sd : probe_spots) {
                    std::swap(sd.r1_seq, sd.r2_seq);
                    std::swap(sd.r1_qual, sd.r2_qual);
                    std::swap(sd.r1_len, sd.r2_len);
                }
                std::swap(r1_len, r2_len);
                std::swap(probe_r1_offset, probe_r2_offset);
                geometry_swap_fired = true;
                // Re-detect with corrected orientation; keep result even if confidence
                // is still low — the physical swap is committed.
                auto geo_proto = detect_protocol(probe_spots, r1_len, r2_len,
                                                 cfg.whitelist_dirs);
                geo_proto.reads_swapped = false;  // already physically swapped above
                detected_proto = geo_proto;
            }

            // ── Pass 1.75: Metadata orientation probe ────────────────────────────────
            // When --metadata-json specifies a whitelist-based protocol, probe both
            // R1[0:bc_len] and R2[0:bc_len] against that whitelist.  If R2 has a
            // better hit rate (≥10%), the deposit is inverted and we force a swap.
            // This catches cases where the geometry rule did not fire (e.g. both reads
            // are similar length) but the whitelist clearly points to R2 as barcode.
            if (!geometry_swap_fired && !cfg.metadata_protocol.empty() &&
                detected_proto.confidence < Confidence::HIGH &&
                !probe_spots.empty() && probe_spots[0].r2_len > 0) {
                const CandidateSpec* meta_spec =
                    find_protocol_spec(cfg.metadata_protocol);
                if (meta_spec && !meta_spec->whitelist_file.empty() &&
                    meta_spec->bc_len > 0) {
                    std::unordered_set<uint64_t> meta_wl;
                    for (const auto& dir : cfg.whitelist_dirs) {
                        std::string wlp = dir + "/" + meta_spec->whitelist_file;
                        meta_wl = load_whitelist_hashes(wlp, meta_spec->bc_len);
                        if (!meta_wl.empty()) break;
                    }
                    if (!meta_wl.empty()) {
                        uint32_t n_samp = std::min<uint32_t>(
                            2000u, static_cast<uint32_t>(probe_spots.size()));
                        uint32_t r1_hits = 0, r2_hits = 0;
                        for (uint32_t i = 0; i < n_samp; ++i) {
                            const auto& sd = probe_spots[i];
                            if (sd.r1_len >= meta_spec->bc_len)
                                if (meta_wl.count(hash_barcode(
                                        sd.r1_seq.data(), meta_spec->bc_len)))
                                    ++r1_hits;
                            if (sd.r2_len >= meta_spec->bc_len)
                                if (meta_wl.count(hash_barcode(
                                        sd.r2_seq.data(), meta_spec->bc_len)))
                                    ++r2_hits;
                        }
                        double r1_rate = r1_hits / static_cast<double>(n_samp);
                        double r2_rate = r2_hits / static_cast<double>(n_samp);
                        if (r2_rate >= 0.10 && r2_rate > r1_rate) {
                            std::cerr << "[1fq-encode] Metadata orientation probe: R2 WL="
                                      << static_cast<int>(r2_rate * 100) << "% > R1 WL="
                                      << static_cast<int>(r1_rate * 100) << "% for "
                                      << cfg.metadata_protocol
                                      << " — forcing R1\u2194R2 swap\n";
                            for (auto& sd : probe_spots) {
                                std::swap(sd.r1_seq, sd.r2_seq);
                                std::swap(sd.r1_qual, sd.r2_qual);
                                std::swap(sd.r1_len, sd.r2_len);
                            }
                            std::swap(r1_len, r2_len);
                            std::swap(probe_r1_offset, probe_r2_offset);
                            geometry_swap_fired = true;  // prevent Pass 2 from undoing
                            auto mp = detect_protocol(probe_spots, r1_len, r2_len,
                                                      cfg.whitelist_dirs);
                            mp.reads_swapped = false;
                            detected_proto = mp;
                        }
                    }
                }
            }

            // For ≥3-segment spots, orientation is ambiguous (cDNA may be in seg 0 or 1).
            // Always evaluate both orientations; for 2-segment spots only swap on low confidence.
            // Skip if the geometry/metadata passes already committed a swap.
            if (!geometry_swap_fired &&
                (detected_proto.confidence < Confidence::MEDIUM || had_multi_segment_spots) &&
                !probe_spots.empty() && probe_spots[0].r2_len > 0) {
                // Swap R1 <-> R2 in probe spots and retry
                for (auto& sd : probe_spots) {
                    std::swap(sd.r1_seq, sd.r2_seq);
                    std::swap(sd.r1_qual, sd.r2_qual);
                    std::swap(sd.r1_len, sd.r2_len);
                }
                std::swap(r1_len, r2_len);
                auto alt = detect_protocol(probe_spots, r1_len, r2_len,
                                           cfg.whitelist_dirs);
                if (alt.score > detected_proto.score) {
                    detected_proto = alt;
                    std::cerr << "[1fq-encode] Segment swap improved detection"
                              << " (R1\u2194R2)\n";
                    // Also swap fast-VDB offsets to match the new R1/R2 assignment
                    std::swap(probe_r1_offset, probe_r2_offset);
                    if (had_multi_segment_spots) three_seg_swapped = true;
                } else {
                    // Swap back
                    for (auto& sd : probe_spots) {
                        std::swap(sd.r1_seq, sd.r2_seq);
                        std::swap(sd.r1_qual, sd.r2_qual);
                        std::swap(sd.r1_len, sd.r2_len);
                    }
                    std::swap(r1_len, r2_len);
                }
            }

            // ── Pass 3: R2-prefix barcode scan ───────────────────────────────────────
            // Handles VDB deposits where READ1 is a short primer/technical read and
            // READ2 = [primer][barcode][UMI][cDNA] (barcode is at a non-zero offset in R2).
            // Standard detection and the simple R1↔R2 swap both fail because neither
            // R1[0:bc_len] nor R2[0:bc_len] contain the barcode at offset 0.
            // Strategy: scan R2 at multiple offsets to find where WL-based protocols
            // (10x-3p-v3/v2 etc.) match the barcode whitelist.  Offsets tried are
            // multiples of 4 up to R2_len - bc_len, capped at 64bp scan range.
            // Example: SRR34789664 (mouse 10x-3p-v3) has
            //   VDB R1 = 28bp primer (not a barcode, not in 3M WL)
            //   VDB R2 = [primer_28bp][barcode_16bp][UMI_12bp][cDNA_34bp] = 90bp
            //   → barcode found at R2[28:44], offset=28.
            if (detected_proto.confidence < Confidence::MEDIUM &&
                !probe_spots.empty() &&
                probe_spots[0].r2_len > 20) {

                //  Build scan list from all whitelist-based protocols in the known
                //  registry, deduped by filename.  Previously only the two dominant
                //  10x WLs (3M-february-2018 and 737K-august-2016) were hardcoded,
                //  which missed BD Rhapsody, ATAC, arc-v1, and any future additions.
                //  AUTOFIX-E2E-A2-READ-SWAP: expand to all bundled whitelists.
                std::vector<std::pair<std::string, uint16_t>> scan_wls;
                {
                    std::unordered_map<std::string, uint16_t> seen_wls;
                    for (const auto& spec : known_protocols()) {
                        if (!spec.whitelist_file.empty() &&
                            !seen_wls.count(spec.whitelist_file)) {
                            seen_wls[spec.whitelist_file] = spec.bc_len;
                            scan_wls.push_back({spec.whitelist_file, spec.bc_len});
                        }
                    }
                }

                double best_r2_rate = 0.0;
                uint16_t best_r2_offset = 0;
                uint16_t best_r2_bc_len = 0;
                std::string best_r2_tag;

                uint32_t n_scan = std::min<uint32_t>(
                    2000u, static_cast<uint32_t>(probe_spots.size()));
                uint16_t max_r2 = probe_spots[0].r2_len;

                for (const auto& [wl_file, bc_len] : scan_wls) {
                    if (max_r2 < bc_len) continue;

                    // Load WL hashes (try all whitelist dirs)
                    std::unordered_set<uint64_t> wl_hashes;
                    for (const auto& dir : cfg.whitelist_dirs) {
                        std::string path = dir + "/" + wl_file;
                        wl_hashes = load_whitelist_hashes(path, bc_len);
                        if (!wl_hashes.empty()) break;
                    }
                    if (wl_hashes.empty()) continue;

                    // Scan R2 at every 4bp from 0 to (max_r2 - bc_len), up to 64bp
                    uint16_t scan_end = std::min<uint16_t>(max_r2 - bc_len, 64);
                    for (uint16_t off = 0; off <= scan_end; off += 4) {
                        uint32_t matches = 0;
                        for (uint32_t i = 0; i < n_scan; ++i) {
                            const auto& sd = probe_spots[i];
                            if (sd.r2_len < off + bc_len) continue;
                            const uint8_t* bc = sd.r2_seq.data() + off;
                            uint64_t h = hash_barcode(bc, bc_len);
                            if (wl_hashes.count(h)) ++matches;
                        }
                        double rate = static_cast<double>(matches) / n_scan;
                        if (rate > best_r2_rate) {
                            best_r2_rate = rate;
                            best_r2_offset = off;
                            best_r2_bc_len = bc_len;
                            // Derive tag from the first protocol that uses this whitelist
                            best_r2_tag = wl_file;  // placeholder; overwritten below
                            for (const auto& spec : known_protocols()) {
                                if (spec.whitelist_file == wl_file &&
                                    spec.bc_len == bc_len) {
                                    best_r2_tag = spec.tag;
                                    break;
                                }
                            }
                        }
                    }
                }

                // Metadata-guided acceptance: when catalog says a specific protocol
                // and its whitelist matches R2[0:bc_len] at ≥10% rate, accept the
                // swap even if best_r2_rate is below the normal 0.5 threshold.
                // Lower threshold catches deposits with barcode-read errors or
                // trimming that reduces the apparent WL hit rate.
                // AUTOFIX-E2E-A2-READ-SWAP: threshold lowered for metadata hints.
                bool meta_r2_accept = false;
                if (!cfg.metadata_protocol.empty() && best_r2_offset == 0 &&
                    best_r2_rate >= 0.10) {
                    const CandidateSpec* meta_spec =
                        find_protocol_spec(cfg.metadata_protocol);
                    if (meta_spec && meta_spec->bc_len == best_r2_bc_len) {
                        meta_r2_accept = true;
                        std::cerr << "[1fq-encode] Metadata-guided R2 accept: protocol="
                                  << cfg.metadata_protocol
                                  << " WL=" << static_cast<int>(best_r2_rate * 100)
                                  << "% at offset 0\n";
                    }
                }

                if (best_r2_rate >= 0.3 || meta_r2_accept) {
                    // Found barcode in R2 at best_r2_offset
                    // Reconstruct probe spots: new_R1 = R2[best_r2_offset : best_r2_offset+28]
                    // (bc+umi region), new_R2 discarded (or set to remainder).
                    // For the new R1 to hold the standard 28bp (bc16+umi12), the new R1
                    // starts at best_r2_offset in R2.
                    uint16_t new_r1_len = r1_len;  // keep same R1 length declaration
                    std::vector<SpotData> r2scan_spots = probe_spots;
                    for (auto& sd : r2scan_spots) {
                        if (sd.r2_len >= best_r2_offset + new_r1_len) {
                            sd.r1_seq.assign(sd.r2_seq.begin() + best_r2_offset,
                                             sd.r2_seq.begin() + best_r2_offset + new_r1_len);
                            sd.r1_qual.assign(sd.r2_qual.begin() + best_r2_offset,
                                              sd.r2_qual.begin() + best_r2_offset + new_r1_len);
                            sd.r1_len = new_r1_len;
                            // New R2: rest of original R2 after the bc+umi region
                            uint16_t r2_start = best_r2_offset + new_r1_len;
                            if (sd.r2_len > r2_start) {
                                sd.r2_seq.erase(sd.r2_seq.begin(),
                                                sd.r2_seq.begin() + r2_start);
                                sd.r2_qual.erase(sd.r2_qual.begin(),
                                                 sd.r2_qual.begin() + r2_start);
                                sd.r2_len = static_cast<uint16_t>(sd.r2_len - r2_start);
                            } else {
                                sd.r2_seq.clear(); sd.r2_qual.clear(); sd.r2_len = 0;
                            }
                        }
                    }

                    auto r2scan = detect_protocol(r2scan_spots, new_r1_len, 0,
                                                  cfg.whitelist_dirs);
                    std::cerr << "[1fq-encode] R2-scan barcode found: R2[" << best_r2_offset
                              << ":+" << new_r1_len << "] = " << best_r2_tag
                              << " (WL rate=" << std::fixed << std::setprecision(2)
                              << best_r2_rate * 100 << "%, score=" << r2scan.score << ")\n";

                    if (r2scan.score > detected_proto.score || best_r2_rate >= 0.5 ||
                        meta_r2_accept) {
                        probe_spots = std::move(r2scan_spots);
                        detected_proto = r2scan;
                        r1_len = new_r1_len;
                        r2_len = 0;  // variable
                        // Adjust VDB fast-path offsets
                        int32_t old_r2_off = probe_r2_offset;
                        probe_r1_offset = (old_r2_off >= 0)
                                          ? old_r2_off + static_cast<int32_t>(best_r2_offset)
                                          : -1;
                        probe_r2_offset = (old_r2_off >= 0)
                                          ? old_r2_off + static_cast<int32_t>(best_r2_offset + new_r1_len)
                                          : -1;
                    }
                }
            }
            // ─────────────────────────────────────────────────────────────────────────

            // ── Pass 4: Late-probe resample ──────────────────────────────────────────
            // SRAs that lead with technical/primer reads (e.g. SRR34789664 mouse
            // 10x-3p-v3 which has ~50K primer reads before any valid barcode) defeat
            // all earlier detection passes: no barcode is accessible in the first
            // 10K VDB spots.  If still sub-MEDIUM confidence, open a second reader at
            // max(50K, total_spots/4) and run detect_protocol on 5000 spots from there.
            // Those later spots are reliably cell reads.  The main encode thread still
            // starts at probe_count (10K) so no reads are lost; this reader is
            // detection-only.
            if (detected_proto.confidence < Confidence::MEDIUM && total_spots > 60000) {
                int64_t late_start = static_cast<int64_t>(
                    std::max<uint64_t>(50000ULL, total_spots / 4));
                if (late_start < static_cast<int64_t>(total_spots) - 1000) {
                    SraReader late_rdr;
                    late_rdr.open(sra_path.c_str());
                    late_rdr.seek(reader.first_row() + late_start);
                    late_rdr.set_row_limit(reader.first_row() + late_start + 5000);

                    std::vector<SpotData> late_spots;
                    late_spots.reserve(5000);
                    int32_t late_r1_off = -1, late_r2_off = -1;
                    SraReader::RawSpot lraw;
                    while (!late_rdr.done_range()) {
                        if (!late_rdr.read_next_spot(lraw)) continue;
                        SpotData lsd;
                        if (lraw.n_segments >= 2 &&
                            lraw.seg_len[0] > 0 && lraw.seg_len[1] > 0) {
                            lsd.r1_len = static_cast<uint16_t>(lraw.seg_len[0]);
                            lsd.r2_len = static_cast<uint16_t>(lraw.seg_len[1]);
                            ascii_to_numeric(lraw.read + lraw.seg_start[0],
                                             lsd.r1_len, lsd.r1_seq);
                            ascii_to_numeric(lraw.read + lraw.seg_start[1],
                                             lsd.r2_len, lsd.r2_seq);
                            lsd.r1_qual.assign(
                                lraw.qual + lraw.seg_start[0],
                                lraw.qual + lraw.seg_start[0] + lsd.r1_len);
                            lsd.r2_qual.assign(
                                lraw.qual + lraw.seg_start[1],
                                lraw.qual + lraw.seg_start[1] + lsd.r2_len);
                            if (late_r1_off < 0) {
                                late_r1_off = lraw.seg_start[0];
                                late_r2_off = lraw.seg_start[1];
                            }
                        } else if (lraw.n_segments >= 1 && lraw.seg_len[0] > 0) {
                            lsd.r1_len = static_cast<uint16_t>(lraw.seg_len[0]);
                            ascii_to_numeric(lraw.read + lraw.seg_start[0],
                                             lsd.r1_len, lsd.r1_seq);
                            lsd.r1_qual.assign(
                                lraw.qual + lraw.seg_start[0],
                                lraw.qual + lraw.seg_start[0] + lsd.r1_len);
                            if (late_r1_off < 0) late_r1_off = lraw.seg_start[0];
                        } else { continue; }
                        late_spots.push_back(std::move(lsd));
                    }

                    if (!late_spots.empty()) {
                        uint16_t late_r1 = late_spots[0].r1_len;
                        uint16_t late_r2 = late_spots[0].r2_len;
                        auto late_result = detect_protocol(
                            late_spots, late_r1, late_r2, cfg.whitelist_dirs);
                        if (late_result.score > detected_proto.score) {
                            // ── Guard 1: score margin ≥ 0.15 required ────────────────
                            // Small score margins (< 0.15) at the late-probe are noise,
                            // not real signal. E.g. 0.682 vs 0.599 is only 0.083 margin
                            // and would switch a valid 10x sample to celseq2 by
                            // coincidence (SRR5995989 / GSM3314349 failure mode).
                            double late_margin = late_result.score - detected_proto.score;
                            if (late_margin < 0.15) {
                                std::cerr << "[1fq-encode] Late-probe: rejecting switch to "
                                          << late_result.tag << " (score margin "
                                          << late_margin << " < 0.15 required)\n";
                            }
                            // ── Guard 2: proposed protocol R1 length must match ───────
                            // The new protocol's expected R1 length must be within ±6bp
                            // of the observed R1 length from the late-probe reads.
                            // Catches e.g. celseq2 (r1_len=12) matching a 28bp 10xv3
                            // R1 read by kmer coincidence.
                            else if (late_result.r1_length > 0 &&
                                     std::abs(static_cast<int>(late_result.r1_length) -
                                              static_cast<int>(late_r1)) > 6) {
                                std::cerr << "[1fq-encode] Late-probe: rejecting switch to "
                                          << late_result.tag << " (expected R1="
                                          << late_result.r1_length << "bp, observed R1="
                                          << late_r1 << "bp)\n";
                            }
                            // ── Guard 3: original detection confidence floor ───────────
                            // If the original protocol was detected with MEDIUM or
                            // higher confidence, the late-probe guard at the outer if
                            // already prevents reaching here. This check additionally
                            // blocks overrides when original confidence is LOW but the
                            // original protocol tag is a known whitelist protocol —
                            // those are unlikely to be wrong and should not be
                            // overridden by a no-whitelist protocol result.
                            else if (detected_proto.confidence >= Confidence::LOW &&
                                     detected_proto.tag != "UNKNOWN" &&
                                     late_result.confidence < Confidence::MEDIUM &&
                                     detected_proto.wl_match_rate > 0.0) {
                                std::cerr << "[1fq-encode] Late-probe: rejecting switch to "
                                          << late_result.tag << " (original "
                                          << detected_proto.tag
                                          << " has WL support; new result conf < MEDIUM)\n";
                            }
                            // ── All guards passed: accept protocol switch ─────────────
                            else {
                                std::cerr << "[1fq-encode] Late-probe resample (spot "
                                          << late_start << "+" << late_spots.size()
                                          << "): " << late_result.tag
                                          << " conf=" << static_cast<int>(late_result.confidence)
                                          << " score=" << late_result.score
                                          << " (was " << detected_proto.tag
                                          << " score=" << detected_proto.score << ")\n";
                                detected_proto = late_result;
                                r1_len = late_r1;
                                r2_len = late_r2;
                                if (late_r1_off >= 0) {
                                    probe_r1_offset = late_r1_off;
                                    probe_r2_offset = late_r2_off;
                                }
                            }
                        }
                    }
                }
            }
            // ─────────────────────────────────────────────────────────────────────────

            // ── Agnostic fallback ──
            // If known-protocol detection returned NONE, try statistical
            // structure inference as a last resort.
            if (detected_proto.confidence < Confidence::LOW) {
                auto agnostic = detect_structure_agnostic(
                    probe_spots, r1_len, r2_len);
                if (agnostic.confidence >= Confidence::LOW) {
                    detected_proto = agnostic.to_candidate();
                    // Encode inferred geometry in the tag
                    detected_proto.tag = "agnostic-bc" +
                                         std::to_string(agnostic.bc_length) + "+umi" +
                                         std::to_string(agnostic.umi_length);
                    std::cerr << "[1fq-encode] Agnostic fallback: "
                              << detected_proto.tag
                              << " (score=" << agnostic.score << ")\n";
                    if (!agnostic.description.empty())
                        std::cerr << "[1fq-encode]   " << agnostic.description
                                  << "\n";
                }
            }

            // ── AUTOFIX-VDB-READ-SWAP-PROTOCOL ──────────────────────────────────
            // After all detection passes (auto-swap + agnostic fallback): if
            // detect_protocol used inverted R2 barcode scoring (reads_swapped=true)
            // and the auto-swap block did not correct orientation (it only fires on
            // score improvement, but equal-confidence inverted vs direct gives equal
            // scores), physically swap R1↔R2 here.
            // Handles both constant-R2 (2-seg) and variable-R2 (mixed multi-seg)
            // VDB deposits where R1=cDNA and R2=CB+UMI.
            if (detected_proto.reads_swapped) {
                std::cerr << "[1fq-encode] VDB read-swap: R1=" << r1_len
                          << "bp(cDNA)\u2194R2=" << r2_len
                          << "bp(barcode) — fixing orientation\n";
                for (auto& sd : probe_spots) {
                    std::swap(sd.r1_seq, sd.r2_seq);
                    std::swap(sd.r1_qual, sd.r2_qual);
                    std::swap(sd.r1_len, sd.r2_len);
                }
                std::swap(r1_len, r2_len);
                // When r1_len is 0 after swap (VDB had variable/zero r2_len),
                // compute majority r1_len from probe spots directly.
                if (r1_len == 0 && !probe_spots.empty()) {
                    uint32_t n_r = std::min<uint32_t>(
                        2000u, static_cast<uint32_t>(probe_spots.size()));
                    std::unordered_map<uint16_t, uint32_t> r1h;
                    for (uint32_t i = 0; i < n_r; ++i)
                        if (probe_spots[i].r1_len > 0) r1h[probe_spots[i].r1_len]++;
                    uint16_t maj1 = 0; uint32_t mc1 = 0;
                    for (const auto& kv : r1h)
                        if (kv.second > mc1) { mc1 = kv.second; maj1 = kv.first; }
                    if (maj1 > 0 && mc1 >= n_r / 2) r1_len = maj1;
                }
                std::swap(probe_r1_offset, probe_r2_offset);
                three_seg_swapped = !three_seg_swapped;
                detected_proto.reads_swapped = false;  // consumed
            }

            if (detected_proto.confidence >= Confidence::LOW) {
                protocol_tag = detected_proto.tag;
                protocol_id = detected_proto.protocol_id;
                confidence = detected_proto.confidence;
                stats.wl_match_rate = detected_proto.wl_match_rate;
            }
        }

        // ── AUTOFIX-PROTOCOL-CONFIDENCE-OVERRIDE ──────────────────────────────
        // When a catalog-trusted protocol is available via EncoderConfig::metadata_protocol
        // and auto-detection produced a low-confidence (≤ LOW = 1) result that disagrees,
        // prefer the metadata protocol.  This prevents mis-classifications like
        // "seqwell → 10x-visium (c=1)" from reaching STAR with wrong CBlen/UMIlen.
        //
        // Rules:
        //  - Only applies when --protocol was NOT set (cfg.protocol_tag.empty()).
        //  - Applies at ANY confidence level (NONE/LOW/MEDIUM/HIGH): the processing
        //    catalog is curated by domain experts and always wins over VDB auto-detection.
        //  - If metadata_protocol is not in the known_protocols registry, emit a
        //    warning and keep the detected values (never silently corrupt the header).
        if (!cfg.protocol_tag.empty()) {
            // Explicit --protocol flag already committed; do not second-guess it.
        } else if (!cfg.metadata_protocol.empty()) {
            const CandidateSpec* meta_spec = find_protocol_spec(cfg.metadata_protocol);
            if (meta_spec) {
                if (meta_spec->tag != protocol_tag) {
                    std::cerr << "[protocol_detect] VDB detection=" << protocol_tag
                              << " (c=" << static_cast<int>(confidence)
                              << ") overridden by catalog metadata.protocol="
                              << cfg.metadata_protocol << "\n";
                    protocol_tag = meta_spec->tag;
                    protocol_id  = meta_spec->protocol_id;
                    confidence   = Confidence::MEDIUM;  // catalog-trusted, verified against registry
                    // Rebuild detected_proto from the spec so downstream code
                    // (re-segment, R1-trim checks) uses the correct geometry.
                    detected_proto.tag         = meta_spec->tag;
                    detected_proto.protocol_id = meta_spec->protocol_id;
                    detected_proto.bc_offset   = meta_spec->bc_offset;
                    detected_proto.bc_length   = meta_spec->bc_len;
                    detected_proto.umi_offset  = meta_spec->umi_offset;
                    detected_proto.umi_length  = meta_spec->umi_len;
                    detected_proto.r1_length   = meta_spec->r1_len;
                    detected_proto.confidence  = Confidence::MEDIUM;
                }
                // else: same protocol — no message needed.
            } else {
                std::cerr << "[protocol_detect] WARNING: metadata.protocol='"
                          << cfg.metadata_protocol
                          << "' is not in the known_protocols registry; "
                          << "keeping auto-detected=" << protocol_tag << "\n";
            }
        }
        // ─────────────────────────────────────────────────────────────────────

        // ── Re-segment probe spots if protocol R1 < observed R1 ──
        // This handles concatenated reads and over-sequenced R1.
        uint16_t split_r1 = detected_proto.r1_length;
        // Offset within concatenated read where barcode starts (0 = beginning, default).
        // Set by WL validation when barcode is found at a non-zero offset within R2.
        int32_t  resegment_r1_off = 0;
        // Set by WL validation when a 2-segment VDB deposit has R1=cDNA, R2=barcode.
        // Tells the streaming phase to swap which segment becomes R1 vs R2.
        bool     wl_orientation_swap = false;
        // Set by WL validation when barcode was found at a positive offset within R2,
        // indicating layout [cDNA][BC][UMI] — R2 (cDNA) physically precedes R1 (barcode)
        // in the raw concatenated read.  Tells streaming to extract R2 from the FRONT of
        // the concatenated bytes (offset 0 .. resegment_r1_off) instead of the tail.
        bool     resegment_r2_before_r1 = false;
        // BUGFIX (AUTOFIX-VDB-R2-VARIABLE-EMPTY): when VDB delivers all reads as
        // single-segment concatenated (seg[1]=0 → r2_len=0 from probe), resegment
        // must also fire when the protocol was provided via metadata_protocol
        // (confidence=LOW after the override block above).  Without this fix the
        // resegment guard (confidence >= MEDIUM) silently suppresses splitting and
        // every read ends up with r2_len=0 → "R2 is empty for all reads" fatal.
        // Metadata-provided protocols are catalog-trusted; splitting is safe.
        bool metadata_protocol_known = (!cfg.metadata_protocol.empty() &&
                                        split_r1 > 0);
        bool resegment = ((confidence >= Confidence::MEDIUM || metadata_protocol_known) &&
                          split_r1 > 0 && split_r1 < r1_len && r2_len == 0);
        if (resegment) {
            std::cerr << "[1fq-encode] Re-segmenting: splitting at R1="
                      << split_r1 << " (was " << r1_len << ")\n";
            for (auto& sd : probe_spots) {
                if (sd.r1_len > split_r1 && sd.r2_len == 0) {
                    sd.r2_len = sd.r1_len - split_r1;
                    sd.r2_seq.assign(sd.r1_seq.begin() + split_r1,
                                     sd.r1_seq.end());
                    sd.r2_qual.assign(sd.r1_qual.begin() + split_r1,
                                      sd.r1_qual.end());
                    sd.r1_seq.resize(split_r1);
                    sd.r1_qual.resize(split_r1);
                    sd.r1_len = split_r1;
                }
            }
            r1_len = split_r1;
            // r2_len stays 0 (variable) since remainder varies per spot
        }

        // Also handle over-sequenced R1 with existing R2 (e.g. [150, 150])
        if ((confidence >= Confidence::MEDIUM || metadata_protocol_known) &&
            split_r1 > 0 && split_r1 < r1_len && r2_len > 0) {
            std::cerr << "[1fq-encode] Trimming over-sequenced R1: "
                      << r1_len << " → " << split_r1 << "\n";
            for (auto& sd : probe_spots) {
                if (sd.r1_len > split_r1) {
                    sd.r1_seq.resize(split_r1);
                    sd.r1_qual.resize(split_r1);
                    sd.r1_len = split_r1;
                }
            }
            r1_len = split_r1;
        }

        // ── Phase 2a-wl: Barcode whitelist validation ─────────────────────────────────
        // After re-segmentation or over-sequenced R1 trim, validate that probe-spot
        // R1[bc_offset:bc_offset+bc_len] matches the protocol whitelist.
        // If match_rate < 5%, the reads are likely in wrong order; try R2 as barcode.
        // Only fires for whitelisted protocols (10x v1/v2/v3, etc.).
        if (!probe_spots.empty() && detected_proto.bc_length > 0) {
            const CandidateSpec* wl_spec = find_protocol_spec(protocol_tag);
            if (wl_spec && !wl_spec->whitelist_file.empty()) {
                std::unordered_set<uint64_t> wl_hashes;
                for (const auto& dir : cfg.whitelist_dirs) {
                    std::string wl_path = dir + "/" + wl_spec->whitelist_file;
                    wl_hashes = load_whitelist_hashes(wl_path, detected_proto.bc_length);
                    if (!wl_hashes.empty()) break;
                }
                if (!wl_hashes.empty()) {
                    const uint16_t bc_off = detected_proto.bc_offset;
                    const uint16_t bc_len = detected_proto.bc_length;
                    uint32_t n_check = std::min<uint32_t>(
                        10000u, static_cast<uint32_t>(probe_spots.size()));

                    // Count R1 barcode matches
                    uint32_t r1_matches = 0;
                    for (uint32_t i = 0; i < n_check; ++i) {
                        const auto& sd = probe_spots[i];
                        if (sd.r1_len < bc_off + bc_len) continue;
                        uint64_t h = hash_barcode(sd.r1_seq.data() + bc_off, bc_len);
                        if (wl_hashes.count(h)) ++r1_matches;
                    }
                    double r1_rate = static_cast<double>(r1_matches) / n_check;
                    std::cerr << "[1fq-encode] Barcode whitelist validation: "
                              << r1_matches << "/" << n_check
                              << " match (" << static_cast<int>(r1_rate * 100) << "%%)";

                    if (r1_rate >= 0.05) {
                        std::cerr << " — OK\n";
                    } else {
                        std::cerr << " — LOW, testing R1\u2194R2 swap\n";

                        // Try R2 as barcode at standard offset, then scan at 4bp steps.
                        // The scan handles concatenated single-segment VDB deposits where
                        // the barcode is NOT at offset 0 of the post-re-segmented R2
                        // (e.g., SRR5398238 = Kang 2018, 10x-3p-v2, barcode stored AFTER
                        // the cDNA in the VDB concatenated read).
                        uint32_t r2_matches = 0;
                        uint32_t r2_viable  = 0;
                        uint16_t best_r2_off = bc_off;  // best offset found in R2
                        double   r2_rate     = 0.0;

                        // First: standard check at protocol bc_offset (usually 0).
                        for (uint32_t i = 0; i < n_check; ++i) {
                            const auto& sd = probe_spots[i];
                            if (sd.r2_len < bc_off + bc_len) continue;
                            ++r2_viable;
                            uint64_t h = hash_barcode(
                                sd.r2_seq.data() + bc_off, bc_len);
                            if (wl_hashes.count(h)) ++r2_matches;
                        }
                        r2_rate = r2_viable > 0
                            ? static_cast<double>(r2_matches) / n_check
                            : 0.0;
                        std::cerr << "[1fq-encode] R2 barcode check (off=0): "
                                  << r2_matches << "/" << n_check
                                  << " match (" << static_cast<int>(r2_rate * 100) << "%%)\n";

                        // If standard check also fails (<5%), scan R2 at 4bp steps.
                        // Useful when barcode is stored AFTER cDNA in a concatenated read.
                        if (r2_rate < 0.05 && !probe_spots.empty() &&
                            probe_spots[0].r2_len > bc_len) {
                            uint16_t max_r2 = probe_spots[0].r2_len;
                            uint16_t scan_end = (max_r2 > bc_len)
                                ? static_cast<uint16_t>(max_r2 - bc_len)
                                : 0;
                            for (uint16_t off = 4; off <= scan_end; off += 4) {
                                uint32_t m = 0, v = 0;
                                for (uint32_t i = 0; i < n_check; ++i) {
                                    const auto& sd = probe_spots[i];
                                    if (sd.r2_len < off + bc_len) continue;
                                    ++v;
                                    uint64_t h = hash_barcode(
                                        sd.r2_seq.data() + off, bc_len);
                                    if (wl_hashes.count(h)) ++m;
                                }
                                double rate = v > 0
                                    ? static_cast<double>(m) / n_check
                                    : 0.0;
                                if (rate > r2_rate) {
                                    r2_rate     = rate;
                                    r2_matches  = m;
                                    r2_viable   = v;
                                    best_r2_off = off;
                                }
                            }
                            if (best_r2_off != bc_off) {
                                std::cerr << "[1fq-encode] R2 barcode scan: best match at"
                                          << " off=" << best_r2_off
                                          << " rate=" << static_cast<int>(r2_rate * 100)
                                          << "%%\n";
                            }
                        }

                        if (r2_viable > 0 && r2_rate >= 0.05 && r2_rate > r1_rate) {
                            std::cerr << "[1fq-encode] Read swap detected: "
                                         "using VDB Read 2 as barcode read\n";

                            if (resegment) {
                                // Concatenated single-segment case.
                                // After initial resegment: R1=concat[0:split_r1], R2=concat[split_r1:]
                                // R2[best_r2_off:best_r2_off+bc_len] has valid barcodes.
                                // True layout: [cDNA_N][BC_split_r1]
                                //   cDNA occupies concat[0 : split_r1+best_r2_off]
                                //   barcode occupies concat[split_r1+best_r2_off : +split_r1]
                                // New streaming split:
                                //   R1 (barcode) = concat[split_r1+best_r2_off : +split_r1]
                                //   R2 (cDNA)    = concat[0 : split_r1+best_r2_off]
                                resegment_r1_off = static_cast<int32_t>(split_r1) +
                                                   static_cast<int32_t>(best_r2_off);
                                resegment_r2_before_r1 = true;
                                std::cerr << "[1fq-encode] R2-before-R1 layout: "
                                             "cDNA=concat[0:" << resegment_r1_off
                                          << "] BC=concat[" << resegment_r1_off
                                          << ":" << resegment_r1_off + split_r1 << "]\n";
                                // Re-extract probe spots: new R1 = R2[best_r2_off:+split_r1]
                                //                         new R2 = R2[0:best_r2_off] (partial cDNA)
                                for (auto& sd : probe_spots) {
                                    uint16_t new_r1_start =
                                        static_cast<uint16_t>(best_r2_off);
                                    if (sd.r2_len >= new_r1_start + split_r1) {
                                        std::vector<uint8_t> new_r1_seq(
                                            sd.r2_seq.begin() + new_r1_start,
                                            sd.r2_seq.begin() + new_r1_start + split_r1);
                                        std::vector<uint8_t> new_r1_qual(
                                            sd.r2_qual.begin() + new_r1_start,
                                            sd.r2_qual.begin() + new_r1_start + split_r1);
                                        // R2 = cDNA portion: R2[0:best_r2_off]
                                        // (full cDNA = initial_R1 + R2[0:best_r2_off] but
                                        //  probe only needs a cDNA fragment for phase 2b/2e)
                                        uint16_t new_r2_len = new_r1_start;  // best_r2_off
                                        sd.r2_seq.resize(new_r2_len);
                                        sd.r2_qual.resize(new_r2_len);
                                        sd.r2_len = new_r2_len;
                                        sd.r1_seq  = std::move(new_r1_seq);
                                        sd.r1_qual = std::move(new_r1_qual);
                                        sd.r1_len  = split_r1;
                                    }
                                }
                                // r1_len stays split_r1; r2_len stays 0 (variable)
                            } else {
                                // Two-segment case: R1=cDNA, R2=barcode.
                                // Swap probe spots and signal streaming phase to swap.
                                wl_orientation_swap = true;
                                for (auto& sd : probe_spots) {
                                    std::swap(sd.r1_seq, sd.r2_seq);
                                    std::swap(sd.r1_qual, sd.r2_qual);
                                    std::swap(sd.r1_len, sd.r2_len);
                                }
                                std::swap(r1_len, r2_len);
                                std::swap(probe_r1_offset, probe_r2_offset);
                                // Force variable R2: streaming gives full cDNA length
                                if (r2_len > r1_len) r2_len = 0;
                            }
                        } else {
                            // Both orientations failed. If the BEST match rate across both
                            // orientations is ≤5%, abort: this is either a barcode-stripped
                            // SRA deposit or a CITE-seq ADT file mislabeled as GEX (random
                            // noise against a 3M whitelist hits 3-7%, which would waste
                            // 1-2 hours producing 0 cells). AUTOFIX-ENCODE-ABORT-LOW-WL-MATCH.
                            // Threshold raised from 0.1% → 5% to catch ADT-as-GEX mislabels.
                            constexpr double kBarcodeStrippedThreshold = 0.05; // 5%
                            if (std::max(r1_rate, r2_rate) <= kBarcodeStrippedThreshold) {
                                std::cerr << "[1fq-encode] ERROR: low barcode whitelist "
                                             "match in both orientations (R1: "
                                          << r1_matches << "/" << n_check
                                          << " = " << static_cast<int>(r1_rate * 100) << "%%"
                                          << ", R2: " << r2_matches << "/" << n_check
                                          << " = " << static_cast<int>(r2_rate * 100) << "%%"
                                          << "; best=" << static_cast<int>(
                                              std::max(r1_rate, r2_rate) * 100) << "%%"
                                          << " <= " << static_cast<int>(
                                              kBarcodeStrippedThreshold * 100) << "%%"
                                          << "). Data is likely barcode-stripped or ADT "
                                             "mislabeled as GEX (data_incomplete). "
                                             "Aborting download.\n";
                                stats.exit_code = 2;
                                return stats;
                            }
                            std::cerr << "[1fq-encode] WARNING: barcode WL validation "
                                         "low on both R1 ("
                                      << static_cast<int>(r1_rate * 100)
                                      << "%%) and R2 ("
                                      << static_cast<int>(r2_rate * 100)
                                      << "%%) — proceeding with original orientation\n";
                        }
                    }
                }
            }
        }
        // ─────────────────────────────────────────────────────────────────────────────

        stats.protocol_tag = protocol_tag;
        stats.confidence = confidence;

        // Safety: force-swap if R2 ≤ 12bp — no single-cell protocol has a biological
        // read this short. Catches cases where VDB assigns cDNA to segment 0 (R1) and
        // barcode to segment 1 (R2), giving high-confidence wrong detection.
        if (r2_len > 0 && r2_len <= 12) {
            std::cerr << "[1fq-encode] Safety swap: R2=" << r2_len
                      << "bp \u226412bp — forcing R1\u2194R2\n";
            for (auto& sd : probe_spots) {
                std::swap(sd.r1_seq, sd.r2_seq);
                std::swap(sd.r1_qual, sd.r2_qual);
                std::swap(sd.r1_len, sd.r2_len);
            }
            std::swap(r1_len, r2_len);
            std::swap(probe_r1_offset, probe_r2_offset);
            three_seg_swapped = !three_seg_swapped;
        }

        // ── Phase 2b: Classify assay type ──
        AssayType assay = classify_assay(protocol_tag, confidence,
                                         probe_spots, r1_len, r2_len);
        stats.assay_type = assay;
        std::cerr << "[1fq-encode] Assay: " << assay_type_name(assay) << "\n";

        // ── Phase 2b+: ATAC 3-read barcode enablement ──────────────────────────────
        // When the SRA deposit has 3 segments (R1 genomic + barcode + R3 genomic),
        // enable I2 stream in the writer so barcodes are captured and embedded in
        // .1fq QNAMEs during singlify process.  Without this, barcode is discarded
        // and fragment extraction silently assigns all reads to the N-pad barcode
        // (BC:NNNNNNNNNNNNNNNN), yielding 0 valid fragments.
        //
        // Conditions: assay = SC_ATAC or SC_MULTIOME_ATAC, AND all probe
        // spots had exactly 3 segments, AND barcode length was consistent.
        bool atac_3seg_i2 = false;
        if ((assay == AssayType::SC_ATAC || assay == AssayType::SC_MULTIOME_ATAC) &&
            all_3seg_atac && probe_bc_len > 0) {
            atac_3seg_i2 = true;
            std::cerr << "[1fq-encode] ATAC 3-read: enabling I2 barcode stream ("
                      << probe_bc_len << "bp) for fragment extraction\n";
        }

        prof.protocol_s += std::chrono::duration<double>(Clock::now() - t0).count();

        // ── Phase 2e: Detect constant 5' prefix in R2 cDNA reads ──────────────────
        // Runs after all orientation/protocol detection passes so probe_spots
        // reflect the final R1=barcode / R2=cDNA assignment.
        // Detects 20+ bp constant adapter/primer contamination at R2 start.
        // polyA/polyT guard prevents false positives on 3' poly-A tails.
        // Result is stored in .1fq metadata so process step can apply
        // --clip5pNbases N 0 automatically.
        uint16_t clip5p_length = 0;
        {
            const bool has_r2 = !probe_spots.empty() && probe_spots[0].r2_len > 0;
            const bool is_sequencing = (assay != AssayType::NOT_SEQUENCE &&
                                        assay != AssayType::UNKNOWN);
            // Skip for ATAC and DNA assays: their R2 is a genomic fragment, not
            // a cDNA with potentially adapter-prefixed content.
            const bool is_rna_or_unknown =
                (assay == AssayType::SC_RNA_3PRIME ||
                 assay == AssayType::SC_RNA_5PRIME ||
                 assay == AssayType::SC_RNA_PLATE  ||
                 assay == AssayType::BULK_RNA       ||
                 assay == AssayType::UNKNOWN);
            (void)is_sequencing;
            if (has_r2 && is_rna_or_unknown) {
                clip5p_length = detect_r2_constant_prefix(probe_spots);
                if (clip5p_length > 0) {
                    std::cerr << "[1fq-encode] R2 constant prefix detected: "
                              << clip5p_length
                              << " bp (threshold: 80%) — will embed clip5p_length="
                              << clip5p_length << " in .1fq metadata\n";
                }
            }
        }

        std::cerr << "[1fq-encode] Protocol: " << protocol_tag
                  << " (confidence: " << static_cast<int>(confidence) << ")\n";
        std::cerr << "[1fq-encode] R1 length: " << r1_len
                  << ", R2 length: "
                  << (r2_len ? std::to_string(r2_len) : "variable") << "\n";
        std::cerr << "[1fq-encode] Total spots: " << total_spots << "\n";

        // ── Phase 2c: Auto-detect adapter contamination in R2 ──
        // TSO/adapter readthrough trimming is ONLY valid for 5'-capture protocols.
        // On 3' protocols (10x-3p, Drop-seq, etc.) R2 is the cDNA read; any
        // fixed-base run at position 30+ is a polyA tail from short inserts, NOT
        // a TSO adapter.  Trimming R2 to 30bp on a 3' protocol destroys barcode
        // assignment ("Discovered 0 barcodes") because STAR gets too few cDNA bases.
        // For UNKNOWN protocol use a stricter threshold (0.85) to reduce false positives.
        // Minimum surviving length: never trim R2 below 40bp (guards against partial
        // polyA false positives even on 5' protocols).
        uint16_t auto_r2_maxlen = 0;
        if (cfg.r2_maxlen == 0 && !probe_spots.empty()) {
            const bool is_5prime = (assay == AssayType::SC_RNA_5PRIME);
            const bool is_unknown = (assay == AssayType::UNKNOWN &&
                                     confidence < Confidence::MEDIUM);
            if (is_5prime || is_unknown) {
                // Stricter threshold for ambiguous protocol to reduce false positives.
                double adapt_thresh = is_unknown ? 0.85 : 0.75;
                uint16_t candidate = detect_r2_adapter(probe_spots, adapt_thresh);
                // Floor: trim position must leave at least 40bp of cDNA for STAR.
                if (candidate >= 40) {
                    auto_r2_maxlen = candidate;
                    std::cerr << "[1fq-encode] WARNING: Adapter/TSO readthrough detected"
                                 " in R2 at position " << auto_r2_maxlen
                              << " [protocol=" << protocol_tag
                              << ", assay=" << assay_type_name(assay)
                              << "]. Auto-setting r2_maxlen=" << auto_r2_maxlen << "\n";
                } else if (candidate > 0) {
                    std::cerr << "[1fq-encode] INFO: R2 adapter candidate pos="
                              << candidate << " discarded (< 40bp floor)"
                              << " [protocol=" << protocol_tag << "]\n";
                }
            }
            // For known non-5' protocols: skip adapter detection entirely.
            // (false-positive rate from polyA tails is too high to risk)
        }

        // ── Phase 2d: Detect split-row paired format ──
        // Some SRAs (ddSEQ, inDrop, certain sci-RNA-seq3) store barcode reads
        // in the first N rows and cDNA reads in the second N rows as separate
        // VDB entries.  Row structure:
        //   Rows 1..N:   n_segments=2, seg_len=[R1, 0]  (barcode half)
        //   Rows N+1..2N: n_segments=2, seg_len=[0, R2]  (cDNA half)
        // Detect this by sampling rows near total_spots/2 when r2_len==0.
        bool split_row_format = false;
        int64_t split_offset = 0;    // first cDNA row (relative to first_row)
        uint16_t split_r2_len = 0;   // sample R2 length from cDNA half
        if (r2_len == 0 && !probe_spots.empty() && total_spots >= 200) {
            int64_t mid = static_cast<int64_t>(total_spots) / 2;
            SraReader sampler;
            sampler.open(sra_path.c_str());
            sampler.seek(reader.first_row() + mid);
            sampler.set_row_limit(reader.first_row() + mid + 10);

            int cdna_rows = 0;
            SraReader::RawSpot sraw;
            while (!sampler.done_range()) {
                if (!sampler.read_next_spot(sraw)) continue;
                if (sraw.n_segments >= 2 &&
                    sraw.seg_len[0] == 0 && sraw.seg_len[1] > 0) {
                    ++cdna_rows;
                    if (split_r2_len == 0)
                        split_r2_len = static_cast<uint16_t>(sraw.seg_len[1]);
                }
            }

            if (cdna_rows >= 5) {
                split_row_format = true;
                split_offset = mid;
                std::cerr << "[1fq-encode] Split-row paired format detected: "
                          << mid << " barcode rows + "
                          << (total_spots - mid)
                          << " cDNA rows (R2~" << split_r2_len << "bp)\n";

                // Back-fill probe spots with real R2 data from the cDNA half.
                // Probe barcode row i corresponds to cDNA row i + mid.
                SraReader r2_filler;
                r2_filler.open(sra_path.c_str());
                r2_filler.seek(reader.first_row() + mid);
                r2_filler.set_row_limit(reader.first_row() + mid +
                                        static_cast<int64_t>(probe_spots.size()));

                SraReader::RawSpot r2raw;
                uint32_t pi = 0;
                while (!r2_filler.done_range() &&
                       pi < static_cast<uint32_t>(probe_spots.size())) {
                    if (!r2_filler.read_next_spot(r2raw)) { ++pi; continue; }
                    if (r2raw.n_segments >= 2 && r2raw.seg_len[1] > 0) {
                        auto& sd = probe_spots[pi];
                        sd.r2_len = static_cast<uint16_t>(r2raw.seg_len[1]);
                        ascii_to_numeric(r2raw.read + r2raw.seg_start[1],
                                         sd.r2_len, sd.r2_seq);
                        sd.r2_qual.assign(
                            r2raw.qual + r2raw.seg_start[1],
                            r2raw.qual + r2raw.seg_start[1] + sd.r2_len);
                    }
                    ++pi;
                }
            }
        }

        // ── Phase 2e: Detect variable R2 → all-empty pattern ──
        // After all structural recovery (resegment, split-row backfill), verify
        // that probe spots have usable R2 data.  If median R2 == 0 and no
        // structural fix applies, the SRA R2 has no sequence data — abort early
        // rather than encoding millions of reads with empty R2, which causes
        // singlify process to crash with "R2 empty for all reads" after wasting
        // download bandwidth and STAR alignment time.
        uint16_t median_probe_r2 = 0;  // used in streaming phase to filter stray zero-R2 reads
        {
            uint32_t n_probe = static_cast<uint32_t>(probe_spots.size());
            if (n_probe > 0) {
                uint32_t n_r2_zero = 0;
                for (const auto& sd : probe_spots)
                    if (sd.r2_len == 0) ++n_r2_zero;
                double r2_zero_frac = static_cast<double>(n_r2_zero) / n_probe;

                // Compute median R2 length over probe spots
                std::vector<uint16_t> r2_lens_probe;
                r2_lens_probe.reserve(n_probe);
                for (const auto& sd : probe_spots)
                    r2_lens_probe.push_back(sd.r2_len);
                std::sort(r2_lens_probe.begin(), r2_lens_probe.end());
                median_probe_r2 = r2_lens_probe[r2_lens_probe.size() / 2];

                std::cerr << "[singlify] R2 variable-length check: median="
                          << median_probe_r2 << "bp, zero="
                          << static_cast<int>(r2_zero_frac * 100) << "% ("
                          << n_r2_zero << "/" << n_probe << " probe reads)\n";

                if (median_probe_r2 == 0 && !split_row_format && !resegment) {
                    // R2 is absent for the majority of probe reads and no
                    // structural fix (split-row, resegment) applies.
                    // Most likely: barcode-stripped SRA deposit or quality-only
                    // R2 column (SRA strips sequence leaving only quality scores).
                    // Abort before wasting download + alignment time.
                    std::cerr << "[singlify] ERROR: R2 (biological read) has no sequence"
                                 " data in SRA.\n"
                              << "[singlify]   median R2 = 0bp over " << n_probe
                              << " probe reads (" << static_cast<int>(r2_zero_frac * 100)
                              << "% zero-length).\n"
                              << "[singlify]   Barcode-stripped or quality-only R2 deposit."
                                 " Classifying as data_incomplete.\n";
                    stats.protocol_tag = "data_incomplete";
                    stats.confidence   = Confidence::NONE;
                    prof.total_s = std::chrono::duration<double>(
                                       Clock::now() - wall_start).count();
                    return stats;
                }
                if (r2_zero_frac > 0.10 && median_probe_r2 > 0) {
                    std::cerr << "[singlify] WARNING: R2 variable-length detected"
                                 " (median: " << median_probe_r2 << "bp, zero-length: "
                              << static_cast<int>(r2_zero_frac * 100)
                              << "%). Zero-length R2 reads will be filtered.\n";
                }
            }
        }

        // ── Phase 3: Initialize writer ──
        WriterConfig wcfg;
        wcfg.codec = cfg.codec;
        wcfg.codec_level = cfg.codec_level;
        wcfg.qual_mode = cfg.qual_mode;
        wcfg.seq_enc = cfg.seq_enc;
        wcfg.block_size = cfg.block_size;
        wcfg.n_streams = 2;
        wcfg.r1_length = r1_len;
        wcfg.r2_length = r2_len;
        wcfg.protocol_id = protocol_id;
        wcfg.confidence = confidence;
        wcfg.assay_type = assay;

        // ATAC 3-read: enable I2 barcode stream so barcodes survive into the .1fq
        // and are injected into QNAMEs during singlify process fragment extraction.
        if (atac_3seg_i2) {
            wcfg.has_i2_stream = true;
            wcfg.i2_length = probe_bc_len;
            // ATAC does not use BC dict or UMI — clear any protocol defaults
            wcfg.bc_length = 0;
            wcfg.umi_length = 0;
            wcfg.bc_offset = 0;
            wcfg.umi_offset = 0;
            wcfg.sort_by_bc = false;
        }

        apply_protocol_to_writer(wcfg, detected_proto, cfg.whitelist_dirs);

        // Clamp UMI length to what R1 can actually provide.
        // If R1 < CB+UMI (e.g. 10x-3p-v3 expects 28bp but SRR delivers 26bp),
        // add_read() would store fewer UMI bytes than cfg_.umi_length declares,
        // creating a stride mismatch in sort_block_by_bc → OOB → SIGSEGV.
        if (wcfg.umi_length > 0 && wcfg.umi_offset + wcfg.umi_length > r1_len) {
            uint16_t avail = (r1_len > wcfg.umi_offset)
                                 ? static_cast<uint16_t>(r1_len - wcfg.umi_offset)
                                 : uint16_t{0};
            std::cerr << "[1fq-encode] WARNING: R1=" << r1_len
                      << "bp < CB(" << wcfg.bc_length
                      << ")+UMI(" << wcfg.umi_length << ")="
                      << (wcfg.umi_offset + wcfg.umi_length)
                      << "bp; truncating UMI " << wcfg.umi_length
                      << " \u2192 " << avail << "bp\n";
            wcfg.umi_length = avail;
        }

        // Disable BC_DICT mode when R1 is shorter than the barcode field.
        // If all reads have r1_len < bc_offset + bc_length, add_read() puts
        // every read into the non-dict path, leaving block_bc_indices_ empty.
        // flush_block() then reads from the reserved-but-empty vector (UB),
        // writes garbage as BC indices, and the decoder later computes
        // umi_length = stream_lengths[0] - bc_length → uint16_t underflow
        // (e.g. 10 - 16 = 65530), causing decode_seq_column to try to read
        // 65530 bytes per read from a small block buffer → SIGSEGV.
        if (wcfg.bc_length > 0 && wcfg.bc_offset + wcfg.bc_length > r1_len) {
            std::cerr << "[1fq-encode] WARNING: R1=" << r1_len
                      << "bp < BC_offset(" << wcfg.bc_offset
                      << ")+BC_len(" << wcfg.bc_length
                      << ")=" << (wcfg.bc_offset + wcfg.bc_length)
                      << "bp; disabling BC_DICT (R1 too short for barcode)\n";
            wcfg.bc_dict.clear();
            wcfg.bc_length = 0;
            wcfg.bc_offset = 0;
            wcfg.umi_offset = 0;
            wcfg.umi_length = 0;
        }

        if (cfg.r2_maxlen > 0)
            wcfg.r2_maxlen = cfg.r2_maxlen;
        else if (auto_r2_maxlen > 0)
            wcfg.r2_maxlen = auto_r2_maxlen;
        if (cfg.no_trim) wcfg.polya_trim = false;
        if (cfg.no_dedup) wcfg.deduped = false;
        if (cfg.sort_by_bc) wcfg.sort_by_bc = true;
        if (cfg.vdb_threads > 1) wcfg.compress_threads = cfg.vdb_threads;

        Writer writer;
        writer.open(cfg.output_path.c_str(), wcfg);

        // ── Phase 4: Write probe spots (already buffered) ──
        t0 = Clock::now();
        if (atac_3seg_i2) {
            // ATAC 3-read: write with barcode from I2 stream
            static const uint8_t kNpad[32] = {4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                                               4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4};
            for (const auto& sd : probe_spots) {
                const uint8_t* bc = (sd.bc_len > 0 && !sd.bc_seq.empty())
                                    ? sd.bc_seq.data() : kNpad;
                uint16_t bc_l = (sd.bc_len > 0) ? sd.bc_len : probe_bc_len;
                writer.add_read_atac(
                    sd.r1_seq.data(), sd.r1_len,
                    sd.r1_qual.empty() ? nullptr : sd.r1_qual.data(),
                    sd.r2_seq.data(), sd.r2_len,
                    sd.r2_qual.empty() ? nullptr : sd.r2_qual.data(),
                    bc, bc_l);
            }
        } else {
            for (const auto& sd : probe_spots) {
                writer.add_read(
                    sd.r2_seq.data(), sd.r2_len, sd.r2_qual.data(),
                    sd.r1_seq.data(), sd.r1_len, sd.r1_qual.data());
            }
        }
        prof.writer_s += std::chrono::duration<double>(Clock::now() - t0).count();
        probe_spots.clear();
        probe_spots.shrink_to_fit();

        // ── Phase 5: Stream remaining spots (direct VDB → numeric → writer) ──
        uint64_t count = probe_count;
        bool need_qual = (cfg.qual_mode != QualMode::NONE);

        // Track whether we used segment-swap during probe (for 3-seg case)
        // swapped_segs is true if we swapped R1↔R2 during probe
        // resegment is true if we split concatenated reads at split_r1

        // No per-read timing — eliminated ~5s chrono overhead (240M clock_gettime).
        // Writer::flush_block() self-times (80 calls, negligible overhead).
        // Phase breakdown computed from: stream wall-clock minus writer flush time.
        auto stream_start = Clock::now();

        // Fast VDB path: for known 2-segment protocols with fixed lengths,
        // skip reading READ_LEN/READ_START (saves 2 of 4 VDB calls per spot).
        // Conditions: fixed R1+R2 from probe, no resegmentation, consistent offsets.
        // Disabled for ATAC 3-segment: fast path only reads 2 segments (R1+R2) and
        // cannot extract the 3rd barcode segment needed for the I2 stream.
        bool use_fast_vdb = (confidence >= Confidence::MEDIUM &&
                             r1_len > 0 && r2_len > 0 && !resegment &&
                             (split_r1 == 0 || split_r1 >= r1_len) &&
                             probe_offsets_consistent &&
                             probe_r1_offset >= 0 && probe_r2_offset >= 0 &&
                             !atac_3seg_i2);  // ATAC 3-read must use generic path
        uint16_t fast_r1l = r1_len;
        uint16_t fast_r2l = r2_len;
        int32_t fast_r1_off = probe_r1_offset;
        int32_t fast_r2_off = probe_r2_offset;
        uint32_t fast_min_bases = static_cast<uint32_t>(
            std::max(fast_r1_off + fast_r1l, fast_r2_off + fast_r2l));

        if (use_fast_vdb) {
            if (cfg.verbose)
                std::cerr << "[1fq-encode] Fast VDB path: 2 calls/spot"
                          << " (R1@" << fast_r1_off << "+" << fast_r1l
                          << " R2@" << fast_r2_off << "+" << fast_r2l << ")\n";

            if (cfg.vdb_threads > 1) {
                // ── Parallel VDB reading: N threads, each with own cursor ──
                int n_threads = cfg.vdb_threads;
                int64_t first = reader.first_row() + static_cast<int64_t>(probe_count);
                int64_t last = reader.first_row() + static_cast<int64_t>(total_spots);
                int64_t batch_rows = static_cast<int64_t>(cfg.block_size);

                // Persistent per-thread readers (avoid repeated open/close)
                std::vector<std::unique_ptr<SraReader>> thr_readers(n_threads);
                for (int t = 0; t < n_threads; ++t) {
                    thr_readers[t] = std::make_unique<SraReader>();
                    thr_readers[t]->open(sra_path.c_str());
                }

                if (cfg.verbose)
                    std::cerr << "[1fq-encode] Parallel VDB: "
                              << n_threads << " threads, batch="
                              << batch_rows << " rows\n";

                // Lambda that reads a row range into a ReadBatch
                auto read_chunk = [&](int tid, int64_t start,
                                      int64_t end, ReadBatch& batch) {
                    auto& rdr = *thr_readers[tid];
                    rdr.seek(start);
                    rdr.set_row_limit(end);
                    uint32_t n = static_cast<uint32_t>(end - start);
                    batch.reserve(n, fast_r1l, fast_r2l, need_qual);

                    SraReader::RawSpot raw_local;
                    std::vector<uint8_t> r1(fast_r1l), r2(fast_r2l);
                    while (!rdr.done_range()) {
                        if (!rdr.read_next_spot_fast(raw_local)) continue;
                        if (raw_local.total_bases < fast_min_bases) continue;
                        for (uint16_t j = 0; j < fast_r1l; ++j)
                            r1[j] = nuc::ascii_to_num(
                                raw_local.read[fast_r1_off + j]);
                        for (uint16_t j = 0; j < fast_r2l; ++j)
                            r2[j] = nuc::ascii_to_num(
                                raw_local.read[fast_r2_off + j]);
                        const uint8_t* q1 = need_qual
                                                ? raw_local.qual + fast_r1_off
                                                : nullptr;
                        const uint8_t* q2 = need_qual
                                                ? raw_local.qual + fast_r2_off
                                                : nullptr;
                        batch.add(r1.data(), fast_r1l,
                                  r2.data(), fast_r2l, q1, q2);
                    }
                };

                // Helper: launch VDB threads for a row range
                auto launch_batch = [&](int64_t bstart, int64_t bend,
                                        std::vector<ReadBatch>& batches,
                                        std::vector<std::thread>& threads) {
                    int64_t bsize = bend - bstart;
                    for (auto& b : batches) b.clear();
                    threads.clear();
                    int64_t per_thr = bsize / n_threads;
                    int64_t extra = bsize % n_threads;
                    int64_t ts = bstart;
                    for (int t = 0; t < n_threads; ++t) {
                        int64_t tc = per_thr + (t < extra ? 1 : 0);
                        threads.emplace_back(read_chunk, t, ts,
                                             ts + tc,
                                             std::ref(batches[t]));
                        ts += tc;
                    }
                };

                // Double-buffered: read batch[i+1] while feeding batch[i]
                std::vector<ReadBatch> buf_a(n_threads), buf_b(n_threads);
                std::vector<std::thread> thr_a, thr_b;
                auto* cur_buf = &buf_a;
                auto* nxt_buf = &buf_b;
                auto* cur_thr = &thr_a;
                auto* nxt_thr = &thr_b;

                // Kick off first batch
                int64_t bstart = first;
                int64_t bend = std::min(bstart + batch_rows, last);
                launch_batch(bstart, bend, *cur_buf, *cur_thr);
                bstart = bend;

                while (true) {
                    // Wait for current batch
                    for (auto& th : *cur_thr) th.join();
                    cur_thr->clear();

                    // Start next batch immediately (overlaps with feed+flush)
                    bool has_next = (bstart < last);
                    if (has_next) {
                        bend = std::min(bstart + batch_rows, last);
                        launch_batch(bstart, bend, *nxt_buf, *nxt_thr);
                        bstart = bend;
                    }

                    // Feed current batch to writer
                    for (int t = 0; t < n_threads; ++t) {
                        const auto& b = (*cur_buf)[t];
                        for (uint32_t i = 0; i < b.reads.size(); ++i) {
                            const uint8_t *r1s, *r2s, *r1q, *r2q;
                            b.get(i, r1s, r2s, r1q, r2q);
                            prof.vdb_bytes += (fast_r1l + fast_r2l) * 2;
                            writer.add_read(r2s, fast_r2l, r2q,
                                            r1s, fast_r1l, r1q);
                            ++count;
                            if (cfg.max_reads > 0 && count >= cfg.max_reads) break;
                        }
                        if (cfg.max_reads > 0 && count >= cfg.max_reads) break;
                    }
                    if (cfg.progress_cb)
                        cfg.progress_cb(count, total_spots);

                    if (!has_next) break;
                    if (cfg.max_reads > 0 && count >= cfg.max_reads) {
                        // Join any pending next-batch threads before exiting
                        for (auto& th : *nxt_thr) th.join();
                        nxt_thr->clear();
                        break;
                    }

                    // Swap buffers
                    std::swap(cur_buf, nxt_buf);
                    std::swap(cur_thr, nxt_thr);
                }
            } else {
                // Serial fast VDB path
                while (!reader.done()) {
                    if (cfg.max_reads > 0 && count >= cfg.max_reads) break;
                    if (!reader.read_next_spot_fast(raw)) continue;
                    if (raw.total_bases < fast_min_bases) continue;

                    ascii_to_numeric(raw.read + fast_r1_off, fast_r1l,
                                     r1_buf_);
                    ascii_to_numeric(raw.read + fast_r2_off, fast_r2l,
                                     r2_buf_);

                    const uint8_t* r1q = nullptr;
                    const uint8_t* r2q = nullptr;
                    if (need_qual) {
                        r1q = raw.qual + fast_r1_off;
                        r2q = raw.qual + fast_r2_off;
                    }
                    prof.vdb_bytes += (fast_r1l + fast_r2l) * 2;

                    writer.add_read(
                        r2_buf_.data(), fast_r2l, r2q,
                        r1_buf_.data(), fast_r1l, r1q);

                    ++count;
                    if (cfg.progress_cb &&
                        (count % cfg.progress_interval) == 0) {
                        cfg.progress_cb(count, total_spots);
                    }
                }
            }
        } else {
            // Generic path: read full segment info for each spot

            // ── Split-row path: interleave barcode half with cDNA half ──
            // For ddSEQ/inDrop/similar SRAs where barcodes and cDNA are in
            // separate halves of the VDB table.
            if (split_row_format) {
                // Limit the main reader to the barcode half (rows 0..split_offset)
                reader.set_row_limit(reader.first_row() +
                                     static_cast<int64_t>(split_offset));

                // Second reader for the cDNA half, starting after probe rows
                SraReader r2_stream;
                r2_stream.open(sra_path.c_str());
                r2_stream.seek(reader.first_row() + split_offset +
                               static_cast<int64_t>(probe_count));
                r2_stream.set_row_limit(reader.first_row() +
                                        static_cast<int64_t>(total_spots));

                SraReader::RawSpot r2_raw;
                while (!reader.done_range()) {
                    if (cfg.max_reads > 0 && count >= cfg.max_reads) break;
                    if (!reader.read_next_spot(raw)) continue;
                    if (r2_stream.done_range() ||
                        !r2_stream.read_next_spot(r2_raw)) continue;

                    // R1: barcode from row seg[0]; apply split_r1 trim if
                    // resegment was active (e.g. inDrop 80bp→46bp).
                    if (raw.n_segments < 1 || raw.seg_len[0] == 0) continue;
                    uint16_t r1l = static_cast<uint16_t>(raw.seg_len[0]);
                    if (resegment && r1l > split_r1) r1l = split_r1;
                    int32_t r1_off = raw.seg_start[0];

                    // R2: cDNA from paired cDNA-row seg[1]
                    uint16_t r2l = 0;
                    int32_t r2_off = 0;
                    if (r2_raw.n_segments >= 2 && r2_raw.seg_len[1] > 0) {
                        r2l = static_cast<uint16_t>(r2_raw.seg_len[1]);
                        r2_off = r2_raw.seg_start[1];
                    }

                    ascii_to_numeric(raw.read + r1_off, r1l, r1_buf_);
                    if (r2l > 0)
                        ascii_to_numeric(r2_raw.read + r2_off, r2l, r2_buf_);
                    else
                        r2_buf_.clear();

                    const uint8_t* r1q =
                        need_qual ? raw.qual + r1_off : nullptr;
                    const uint8_t* r2q =
                        (need_qual && r2l > 0) ? r2_raw.qual + r2_off : nullptr;
                    prof.vdb_bytes += (r1l + r2l) * 2;

                    // Enforce fixed R2 length contract if writer uses fixed stride
                    if (wcfg.r2_length > 0 && r2l > 0 &&
                        r2l != wcfg.r2_length) {
                        if (r2l < wcfg.r2_length) {
                            r2_buf_.resize(wcfg.r2_length);
                            std::fill(r2_buf_.begin() + r2l, r2_buf_.end(),
                                      static_cast<uint8_t>(4));
                            if (need_qual && r2q) {
                                r2_qual_buf_.assign(r2q, r2q + r2l);
                                r2_qual_buf_.resize(wcfg.r2_length,
                                                    static_cast<uint8_t>(0));
                                r2q = r2_qual_buf_.data();
                            }
                        }
                        r2l = wcfg.r2_length;
                    }

                    writer.add_read(r2_buf_.data(), r2l, r2q,
                                    r1_buf_.data(), r1l, r1q);
                    ++count;
                    if (cfg.progress_cb &&
                        (count % cfg.progress_interval) == 0)
                        cfg.progress_cb(count,
                                        static_cast<uint64_t>(split_offset));
                }
            } else {
            // ── Regular generic path ──
            while (!reader.done()) {
                if (cfg.max_reads > 0 && count >= cfg.max_reads) break;
                if (!reader.read_next_spot(raw)) continue;

                uint16_t r1l = 0, r2l = 0;
                int32_t r1_off = 0, r2_off = 0;

                if (raw.n_segments >= 3) {
                    // 3+ segments: pick two longest, using probe-determined orientation.
                    // three_seg_swapped=true means cDNA is in the higher-index segment (R1←high).
                    uint32_t sorted_idx[16];
                    uint32_t ns = std::min<uint32_t>(raw.n_segments, 16);
                    for (uint32_t s = 0; s < ns; ++s) sorted_idx[s] = s;
                    std::sort(sorted_idx, sorted_idx + ns,
                              [&](uint32_t a, uint32_t b) {
                                  return raw.seg_len[a] > raw.seg_len[b];
                              });
                    uint32_t b_low  = std::min(sorted_idx[0], sorted_idx[1]);
                    uint32_t b_high = std::max(sorted_idx[0], sorted_idx[1]);
                    uint32_t b0 = three_seg_swapped ? b_high : b_low;
                    uint32_t b1 = three_seg_swapped ? b_low  : b_high;
                    r1l = static_cast<uint16_t>(raw.seg_len[b0]);
                    r2l = static_cast<uint16_t>(raw.seg_len[b1]);
                    r1_off = raw.seg_start[b0];
                    r2_off = raw.seg_start[b1];
                } else if (raw.n_segments >= 2 && raw.seg_len[0] > 0 &&
                           raw.seg_len[1] > 0) {
                    // Normal paired-end (swap if WL validation found R1=cDNA, R2=barcode)
                    uint32_t s0 = wl_orientation_swap ? 1u : 0u;
                    uint32_t s1 = wl_orientation_swap ? 0u : 1u;
                    r1l = static_cast<uint16_t>(raw.seg_len[s0]);
                    r2l = static_cast<uint16_t>(raw.seg_len[s1]);
                    r1_off = raw.seg_start[s0];
                    r2_off = raw.seg_start[s1];
                } else if (raw.n_segments >= 2 && raw.seg_len[0] == 0 &&
                           raw.seg_len[1] > 0) {
                    // Empty first segment (e.g. celseq2): all data in seg[1]
                    if (resegment &&
                        static_cast<int32_t>(raw.seg_len[1]) >= resegment_r1_off + split_r1) {
                        r1l   = split_r1;
                        r1_off = raw.seg_start[1] + resegment_r1_off;
                        if (resegment_r2_before_r1) {
                            // Layout [cDNA][BC][UMI]: R2 is the cDNA prefix before the barcode
                            r2_off = raw.seg_start[1];
                            r2l   = static_cast<uint16_t>(resegment_r1_off);
                        } else {
                            r2_off = raw.seg_start[1] + resegment_r1_off + split_r1;
                            r2l   = static_cast<uint16_t>(
                                raw.seg_len[1] - resegment_r1_off - split_r1);
                        }
                    } else {
                        r1l = static_cast<uint16_t>(raw.seg_len[1]);
                        r1_off = raw.seg_start[1];
                    }
                } else if (resegment &&
                           static_cast<int32_t>(raw.total_bases) >= resegment_r1_off + split_r1) {
                    // Concatenated: split at detected R1 length (resegment_r1_off=0 normally;
                    // set to split_r1+best_r2_off when WL found barcode at non-zero offset)
                    r1l   = split_r1;
                    r1_off = resegment_r1_off;
                    if (resegment_r2_before_r1) {
                        // Layout [cDNA][BC][UMI]: R2 = cDNA prefix before the barcode
                        r2_off = 0;
                        r2l   = static_cast<uint16_t>(resegment_r1_off);
                    } else {
                        r2_off = resegment_r1_off + split_r1;
                        r2l   = static_cast<uint16_t>(
                            raw.total_bases - resegment_r1_off - split_r1);
                    }
                } else if (raw.n_segments >= 2 && raw.seg_len[0] > 0 &&
                           raw.seg_len[1] == 0 && resegment &&
                           static_cast<int32_t>(raw.seg_len[0]) >= resegment_r1_off + split_r1) {
                    // n_segments=2 but seg[1].len=0: treat as concatenated
                    r1l   = split_r1;
                    r1_off = raw.seg_start[0] + resegment_r1_off;
                    if (resegment_r2_before_r1) {
                        // Layout [cDNA][BC][UMI]: R2 = cDNA prefix before the barcode
                        r2_off = raw.seg_start[0];
                        r2l   = static_cast<uint16_t>(resegment_r1_off);
                    } else {
                        r2_off = raw.seg_start[0] + resegment_r1_off + split_r1;
                        r2l   = static_cast<uint16_t>(
                            raw.seg_len[0] - resegment_r1_off - split_r1);
                    }
                } else if (raw.n_segments >= 1 && raw.seg_len[0] > 0) {
                    // Single-segment (no R2): only useful if short enough for R1-only
                    r1l = static_cast<uint16_t>(raw.seg_len[0]);
                    r1_off = raw.seg_start[0];
                } else {
                    continue;
                }

                if (r1l == 0) continue;

                // Filter zero-R2 reads when median probe R2 was non-zero
                // (variable-length R2 with stray empty gaps in VDB delivery).
                // Prevents sparsely-empty R2 reads from entering the .1fq,
                // which would reduce cell-calling quality downstream.
                if (r2l == 0 && wcfg.r2_length == 0 && median_probe_r2 > 0) {
                    continue;
                }

                // Apply over-sequenced R1 trim (e.g. 150bp → 28bp)
                if (split_r1 > 0 && r1l > split_r1 && r2l > 0) {
                    r1l = split_r1;
                }

                // Convert ASCII DNA → numeric
                ascii_to_numeric(raw.read + r1_off, r1l, r1_buf_);
                if (r2l > 0) {
                    ascii_to_numeric(raw.read + r2_off, r2l, r2_buf_);
                } else {
                    r2_buf_.clear();
                }

                const uint8_t* r1q = nullptr;
                const uint8_t* r2q = nullptr;
                if (need_qual) {
                    r1q = raw.qual + r1_off;
                    r2q = (r2l > 0) ? raw.qual + r2_off : nullptr;
                }
                prof.vdb_bytes += (r1l + r2l) * 2;

                // Enforce fixed R2 length contract (mirrors FastqEncoder pad/truncate).
                // wcfg.r2_length > 0 means the writer was opened in fixed-stride mode;
                // every call to add_read() must supply exactly that length or the
                // block encoder produces misaligned quality/sequence columns.
                if (wcfg.r2_length > 0 && r2l > 0 && r2l != wcfg.r2_length) {
                    if (r2l < wcfg.r2_length) {
                        // Pad seq with N (numeric 4) and qual with phred 0 ('!')
                        r2_buf_.resize(wcfg.r2_length);
                        std::fill(r2_buf_.begin() + r2l, r2_buf_.end(),
                                  static_cast<uint8_t>(4));
                        if (need_qual && r2q) {
                            r2_qual_buf_.assign(r2q, r2q + r2l);
                            r2_qual_buf_.resize(wcfg.r2_length,
                                                static_cast<uint8_t>(0));
                            r2q = r2_qual_buf_.data();
                        }
                    }
                    // Truncate (or confirm pad): clamp length to declared fixed size
                    r2l = wcfg.r2_length;
                }

                // Feed to writer
                if (atac_3seg_i2 && raw.n_segments == 3) {
                    // ATAC 3-read: extract barcode from smallest segment and
                    // store R1 genomic, R3 genomic, and barcode via add_read_atac.
                    // The 3 segments are in sorted_idx order: sorted_idx[0] and [1]
                    // are the 2 genomic reads (already in r1_buf_/r2_buf_), and
                    // sorted_idx[2] is the barcode (smallest segment).
                    // Re-compute sorted_idx here to find the barcode's position.
                    uint32_t ns3 = std::min<uint32_t>(raw.n_segments, 16);
                    uint32_t si3[16];
                    for (uint32_t s = 0; s < ns3; ++s) si3[s] = s;
                    std::sort(si3, si3 + ns3,
                              [&](uint32_t a, uint32_t b) {
                                  return raw.seg_len[a] > raw.seg_len[b];
                              });
                    uint32_t bc_seg = si3[2];  // shortest = barcode
                    uint16_t bc_l = static_cast<uint16_t>(raw.seg_len[bc_seg]);
                    int32_t  bc_off = raw.seg_start[bc_seg];
                    if (bc_l > 0 && bc_l <= 32) {
                        bc_buf_.resize(bc_l);
                        for (uint16_t j = 0; j < bc_l; ++j)
                            bc_buf_[j] = nuc::ascii_to_num(raw.read[bc_off + j]);
                        writer.add_read_atac(
                            r1_buf_.data(), r1l, r1q,
                            r2_buf_.data(), r2l, r2q,
                            bc_buf_.data(), bc_l);
                    } else {
                        // Degenerate: use N-pad barcode
                        bc_buf_.assign(wcfg.i2_length, 4);
                        writer.add_read_atac(
                            r1_buf_.data(), r1l, r1q,
                            r2_buf_.data(), r2l, r2q,
                            bc_buf_.data(), wcfg.i2_length);
                    }
                } else {
                    writer.add_read(
                        r2_buf_.data(), r2l, r2q,
                        r1_buf_.data(), r1l, r1q);
                }

                ++count;
                if (cfg.progress_cb && (count % cfg.progress_interval) == 0) {
                    cfg.progress_cb(count, total_spots);
                }
            }
            }  // end else (regular generic path, not split-row)
        }  // end fast_vdb else

        // Compute phase breakdown from streaming wall-clock and writer flush timer
        {
            double stream_s = std::chrono::duration<double>(
                                  Clock::now() - stream_start)
                                  .count();
            double flush_s = writer.flush_time_s();
            // Writer time = flush (sort+compress+write) + add_read (append to buffers)
            // VDB read + convert + add_read = stream_s - flush_s
            // We use probe ratios to apportion VDB vs convert
            double probe_total = prof.vdb_read_s + prof.convert_s;
            double vdb_frac = probe_total > 0
                                  ? prof.vdb_read_s / probe_total
                                  : 0.5;
            double non_flush_s = stream_s - flush_s;
            prof.vdb_read_s = non_flush_s * vdb_frac;
            prof.convert_s = non_flush_s * (1.0 - vdb_frac);
            prof.writer_s = flush_s + prof.writer_s;  // probe write + stream flush
            prof.w_wait_s = writer.flush_wait_s();
            prof.w_sort_s = writer.flush_sort_s();
            prof.w_pack_s = writer.flush_pack_s();
            prof.w_compress_s = writer.flush_compress_s();
            prof.w_write_s = writer.flush_write_s();
            // Sort sub-phase detail
            if (cfg.verbose) {
                std::fprintf(stderr, "[1fq-profile]   Sort detail: count=%.3fs intra=%.3fs permute=%.3fs\n",
                             writer.sort_count_s(), writer.sort_intra_s(), writer.sort_permute_s());
            }
        }

        // ── Phase 6: Finalize ──
        t0 = Clock::now();
        std::string meta = build_metadata_json(
            sra_path, protocol_tag, protocol_id, confidence, assay,
            stats.wl_match_rate, r1_len, r2_len, count, clip5p_length, cfg);
        writer.finish(meta);
        prof.finalize_s += std::chrono::duration<double>(Clock::now() - t0).count();

        // Get output file size
        {
            FILE* fp = std::fopen(cfg.output_path.c_str(), "rb");
            if (fp) {
                std::fseek(fp, 0, SEEK_END);
                prof.output_bytes = static_cast<uint64_t>(std::ftell(fp));
                std::fclose(fp);
            }
        }

        prof.total_s = std::chrono::duration<double>(
                           Clock::now() - wall_start)
                           .count();

        stats.total_reads = count;
        stats.blocks_written = writer.blocks_written();
        return stats;
    }

   private:
    // JSON string escaper — handles quotes, backslashes, and control characters
    static std::string json_str(const std::string& s) {
        std::string out;
        out.reserve(s.size() + 4);
        out += '"';
        for (unsigned char c : s) {
            if (c == '"') {
                out += "\\\"";
            } else if (c == '\\') {
                out += "\\\\";
            } else if (c == '\n') {
                out += "\\n";
            } else if (c == '\r') {
                out += "\\r";
            } else if (c == '\t') {
                out += "\\t";
            } else if (c < 0x20) { /* skip other control chars */
            } else {
                out += static_cast<char>(c);
            }
        }
        out += '"';
        return out;
    }

    std::string build_metadata_json(
        const std::string& sra_path,
        const std::string& protocol_tag,
        uint8_t protocol_id,
        Confidence confidence,
        AssayType assay,
        double wl_match_rate,
        uint16_t r1_len,
        uint16_t r2_len,
        uint64_t total_reads,
        uint16_t clip5p_length,
        const EncoderConfig& cfg) {
        // ── Compute processing route ──
        int cb_start = 1, cb_len = 0, umi_start = 1, umi_len = 0;

        // Resolve protocol_tag to canonical form via protocol_id (handles aliases
        // like "sci-RNA" → "sci-rna-seq3", "10x-v3" → "10x-3p-v3")
        std::string canonical_tag = protocol_tag;
        if (protocol_id > 0) {
            for (const auto& spec : known_protocols()) {
                if (spec.protocol_id == protocol_id) {
                    canonical_tag = spec.tag;
                    break;
                }
            }
        }

        bool no_star = (assay == AssayType::SC_ATAC ||
                        assay == AssayType::SC_MULTIOME_ATAC ||
                        assay == AssayType::SC_METHYLOME ||
                        assay == AssayType::WGS ||
                        assay == AssayType::WES ||
                        assay == AssayType::NOT_SEQUENCE ||
                        canonical_tag == "10x-atac");

        bool is_complex = (canonical_tag == "microwell-seq" ||
                           canonical_tag == "surecell" ||
                           canonical_tag == "bd-rhapsody" ||
                           canonical_tag == "splitseq" ||
                           canonical_tag == "indrop" ||
                           canonical_tag == "ddseq" ||
                           canonical_tag == "sci-rna-seq3");

        bool is_smartseq = (assay == AssayType::SC_RNA_PLATE ||
                            canonical_tag == "strtseq");

        if (!no_star && !is_complex && !is_smartseq) {
            // Prefer protocol table values when protocol is known
            bool found_spec = false;
            if (protocol_id > 0) {
                for (const auto& spec : known_protocols()) {
                    if (spec.protocol_id == protocol_id) {
                        cb_len = spec.bc_len;
                        umi_start = spec.umi_offset + 1;  // 1-based for STAR
                        umi_len = spec.umi_len;
                        found_spec = true;
                        break;
                    }
                }
            }
            // Fall back to R1 length heuristics only when protocol unknown
            if (!found_spec) {
                // ── Agnostic fallback: parse BC/UMI lengths from the tag name ──
                // The agnostic encoder tags the protocol as "agnostic-bcN+umiM",
                // encoding the detected BC length (N) and UMI length (M) directly
                // in the tag.  Parse these instead of running the r1_len/2 heuristic,
                // which produces nonsensical values (e.g. CBlen=19, UMIlen=19 for a
                // 38bp R1) that are then rejected by singlify's sanity check.
                if (canonical_tag.size() > 8 &&
                    canonical_tag.compare(0, 8, "agnostic") == 0) {
                    auto bc_pos  = canonical_tag.find("-bc");
                    auto umi_pos = canonical_tag.find("+umi");
                    if (bc_pos != std::string::npos && umi_pos != std::string::npos) {
                        int bc_val  = std::atoi(canonical_tag.c_str() + bc_pos  + 3);
                        int umi_val = std::atoi(canonical_tag.c_str() + umi_pos + 4);
                        if (bc_val > 0 && bc_val <= 32 && umi_val > 0 && umi_val <= 16) {
                            cb_len    = bc_val;
                            umi_start = cb_len + 1;  // UMI immediately follows BC (offset=0)
                            umi_len   = umi_val;
                        }
                    }
                }
                if (cb_len > 0) {
                    // Agnostic detection provided valid values; skip heuristics.
                } else if (r1_len == 28) {
                    cb_len = 16;
                    umi_start = 17;
                    umi_len = 12;
                } else if (r1_len == 26) {
                    cb_len = 16;
                    umi_start = 17;
                    umi_len = 10;
                } else if (r1_len == 24) {
                    cb_len = 14;
                    umi_start = 15;
                    umi_len = 10;
                } else if (r1_len == 20) {
                    cb_len = 12;
                    umi_start = 13;
                    umi_len = 8;
                } else if (r1_len == 23) {
                    cb_len = 15;
                    umi_start = 16;
                    umi_len = 8;
                } else if (r1_len > 0) {
                    cb_len = r1_len / 2;
                    umi_start = cb_len + 1;
                    umi_len = r1_len - cb_len;
                }
            }
        }

        // ── Determine aligner routing ──
        const char* aligner = nullptr;
        const char* route = nullptr;
        const char* notes = nullptr;
        if (no_star) {
            if (assay == AssayType::SC_ATAC || assay == AssayType::SC_MULTIOME_ATAC) {
                aligner = "bwa-mem2";
                route = "atac";
                notes = "Fragment file generation: cellranger-atac or SnapATAC2";
            } else if (assay == AssayType::SC_METHYLOME) {
                aligner = "bismark";
                route = "bisulfite";
                notes = "Requires bisulfite-aware aligner (bismark or biscuit)";
            } else if (assay == AssayType::WGS || assay == AssayType::WES) {
                aligner = "bwa-mem2";
                route = "dna";
            } else {
                aligner = nullptr;
                route = "unsupported";
            }
        } else if (is_smartseq) {
            aligner = "starsolo";
            route = "smartseq";
        } else if (is_complex) {
            aligner = "starsolo";
            route = "cb_umi_complex";
        } else if (cb_len > 0) {
            aligner = "starsolo";
            route = "cb_umi_simple";
        } else {
            aligner = "star";
            route = "bulk_rna";
        }

        // ── EFO / ontology ──
        auto efo = assay_efo_term(assay);
        std::string proto_uri = protocol_uri(protocol_tag);

        // ── Assemble JSON ──
        std::string j;
        j.reserve(4096);
        j += "{\n";

        // ── Source & accession ──
        j += "  \"source\": " + json_str(sra_path) + ",\n";

        // ── Protocol block ──
        j += "  \"protocol\": {\n";
        j += "    \"tag\": " + json_str(protocol_tag) + ",\n";
        j += "    \"id\": " + std::to_string(protocol_id) + ",\n";
        j += "    \"confidence\": " + std::to_string(static_cast<int>(confidence)) + ",\n";
        j += "    \"whitelist_match_rate\": " + std::to_string(wl_match_rate) + ",\n";
        if (!proto_uri.empty())
            j += "    \"uri\": " + json_str(proto_uri) + "\n";
        else
            j += "    \"uri\": null\n";
        j += "  },\n";

        // ── Assay block (with EFO ontology term) ──
        j += "  \"assay\": {\n";
        j += "    \"type\": " + json_str(assay_type_name(assay)) + ",\n";
        if (efo.id && efo.id[0] != '\0') {
            j += "    \"efo_id\": " + json_str(efo.id) + ",\n";
            j += "    \"efo_label\": " + json_str(efo.label) + "\n";
        } else {
            j += "    \"efo_id\": null,\n";
            j += "    \"efo_label\": null\n";
        }
        j += "  },\n";

        // ── Streams array ──
        // Describes each read stream by name and fixed length
        j += "  \"streams\": [\n";
        if (r1_len > 0 || r2_len > 0) {
            j += "    {\"index\": 0, \"role\": \"R1\", \"fixed_length\": " +
                 std::to_string(r1_len) + "}";
            j += ",\n    {\"index\": 1, \"role\": \"R2\", \"fixed_length\": " +
                 std::to_string(r2_len) + "}";
            j += "\n";
        }
        j += "  ],\n";

        // ── Read counts ──
        j += "  \"r1_length\": " + std::to_string(r1_len) + ",\n";
        j += "  \"r2_length\": " + std::to_string(r2_len) + ",\n";
        j += "  \"total_reads\": " + std::to_string(total_reads) + ",\n";
        // clip5p_length: constant 5' prefix detected in R2 during encoding;
        // process step reads this and applies --clip5pNbases N 0 to STAR.
        if (clip5p_length > 0)
            j += "  \"clip5p_length\": " + std::to_string(clip5p_length) + ",\n";

        // ── Processing hints (aligner routing) ──
        j += "  \"processing_hints\": {\n";
        j += "    \"aligner\": " + (aligner ? json_str(aligner) : std::string("null")) + ",\n";
        j += "    \"route\": " + (route ? json_str(route) : std::string("null")) + ",\n";
        if (notes)
            j += "    \"notes\": " + json_str(notes) + ",\n";
        // Aligner-specific parameter block
        if (no_star) {
            j += "    \"aligner_params\": null\n";
        } else if (is_smartseq) {
            j += "    \"aligner_params\": {\"soloType\": \"SmartSeq\"}\n";
        } else if (is_complex) {
            j += "    \"aligner_params\": {\n";
            j += "      \"soloType\": \"CB_UMI_Complex\",\n";
            if (canonical_tag == "microwell-seq") {
                j += "      \"soloCBposition\": [\"0_0_0_5\", \"0_21_0_26\", \"0_42_0_47\"],\n";
                j += "      \"soloUMIposition\": \"0_48_0_53\"\n";
            } else if (canonical_tag == "surecell") {
                j += "      \"soloCBposition\": [\"0_0_0_5\", \"0_21_0_26\", \"0_42_0_47\"],\n";
                j += "      \"soloUMIposition\": \"0_51_0_58\"\n";
            } else if (canonical_tag == "bd-rhapsody") {
                j += "      \"soloCBposition\": [\"0_0_0_8\", \"0_21_0_29\", \"0_43_0_51\"],\n";
                j += "      \"soloUMIposition\": \"0_52_0_59\"\n";
            } else if (canonical_tag == "splitseq") {
                // R2 layout (stored as R1 after encode swap): [UMI_10bp][Rd3_8bp][linker15][Rd2_8bp][linker15][Rd1_8bp]
                // All positions are absolute from read start (anchorType=0):
                //   Rd3: 0_10_0_17  Rd2: 0_33_0_40  Rd1: 0_56_0_63  UMI: 0_0_0_9
                j += "      \"soloCBposition\": [\"0_10_0_17\", \"0_33_0_40\", \"0_56_0_63\"],\n";
                j += "      \"soloUMIposition\": \"0_0_0_9\"\n";
            } else if (canonical_tag == "indrop") {
                j += "      \"soloCBposition\": [\"0_0_0_9\", \"0_32_0_39\"],\n";
                j += "      \"soloUMIposition\": \"0_40_0_45\"\n";
            } else if (canonical_tag == "ddseq") {
                j += "      \"soloCBposition\": [\"0_0_0_5\", \"0_9_0_14\", \"0_18_0_23\"],\n";
                j += "      \"soloUMIposition\": \"0_27_0_34\"\n";
            } else if (canonical_tag == "sci-rna-seq3") {
                j += "      \"soloCBposition\": [\"0_24_0_33\"],\n";
                j += "      \"soloUMIposition\": \"0_16_0_23\"\n";
            } else {
                j += "      \"soloCBstart\": 1,\n";
                j += "      \"soloCBlen\": " + std::to_string(r1_len > 0 ? r1_len / 2 : 12) + "\n";
            }
            j += "    }\n";
        } else if (cb_len > 0) {
            j += "    \"aligner_params\": {\n";
            j += "      \"soloType\": \"CB_UMI_Simple\",\n";
            j += "      \"soloCBstart\": " + std::to_string(cb_start) + ",\n";
            j += "      \"soloCBlen\": " + std::to_string(cb_len) + ",\n";
            j += "      \"soloUMIstart\": " + std::to_string(umi_start) + ",\n";
            j += "      \"soloUMIlen\": " + std::to_string(umi_len) + "\n";
            j += "    }\n";
        } else {
            j += "    \"aligner_params\": null\n";
        }
        j += "  },\n";

        // ── SRA / GEO metadata ──
        const auto& m = cfg.sra_meta;
        j += "  \"sra_metadata\": {\n";
        // Accession hierarchy
        j += "    \"srr\": " + json_str(m.srr) + ",\n";
        j += "    \"srx\": " + json_str(m.srx) + ",\n";
        j += "    \"srp\": " + json_str(m.srp) + ",\n";
        j += "    \"gse\": " + json_str(m.gse) + ",\n";
        j += "    \"gsm\": " + json_str(m.gsm) + ",\n";
        j += "    \"biosample\": " + json_str(m.biosample) + ",\n";
        // Sample biology
        j += "    \"organism\": " + json_str(m.organism) + ",\n";
        j += "    \"taxon_id\": " + json_str(m.taxon_id) + ",\n";
        j += "    \"tissue\": " + json_str(m.tissue) + ",\n";
        j += "    \"cell_type\": " + json_str(m.cell_type) + ",\n";
        j += "    \"disease\": " + json_str(m.disease) + ",\n";
        j += "    \"developmental_stage\": " + json_str(m.developmental_stage) + ",\n";
        j += "    \"sex\": " + json_str(m.sex) + ",\n";
        // Library
        j += "    \"library_name\": " + json_str(m.library_name) + ",\n";
        j += "    \"library_strategy\": " + json_str(m.library_strategy) + ",\n";
        j += "    \"library_source\": " + json_str(m.library_source) + ",\n";
        j += "    \"library_selection\": " + json_str(m.library_selection) + ",\n";
        j += "    \"library_layout\": " + json_str(m.library_layout) + ",\n";
        // Platform
        j += "    \"platform\": " + json_str(m.platform) + ",\n";
        j += "    \"instrument\": " + json_str(m.instrument) + ",\n";
        // Submission
        j += "    \"center_name\": " + json_str(m.center_name) + ",\n";
        j += "    \"submission_id\": " + json_str(m.submission_id) + ",\n";
        j += "    \"submission_date\": " + json_str(m.submission_date) + ",\n";
        j += "    \"last_updated\": " + json_str(m.last_updated) + ",\n";
        // Descriptions
        j += "    \"title\": " + json_str(m.title) + ",\n";
        j += "    \"experiment_title\": " + json_str(m.experiment_title) + ",\n";
        j += "    \"study_title\": " + json_str(m.study_title) + ",\n";
        j += "    \"study_abstract\": " + json_str(m.study_abstract) + ",\n";
        j += "    \"sample_description\": " + json_str(m.sample_description) + ",\n";
        // Arbitrary sample attributes
        j += "    \"sample_attributes\": [";
        for (size_t i = 0; i < m.sample_attributes.size(); ++i) {
            if (i > 0) j += ", ";
            j += "{\"tag\": " + json_str(m.sample_attributes[i].first) +
                 ", \"value\": " + json_str(m.sample_attributes[i].second) + "}";
        }
        j += "]\n";
        j += "  },\n";

        // ── Encoder provenance ──
        j += "  \"encoder\": {\n";
        j += "    \"version\": \"lib1fq-0.3.0\",\n";
        j += "    \"codec\": " + std::to_string(static_cast<int>(cfg.codec)) + ",\n";
        j += "    \"codec_level\": " + std::to_string(cfg.codec_level) + ",\n";
        j += "    \"qual_mode\": " + std::to_string(static_cast<int>(cfg.qual_mode)) + ",\n";
        j += "    \"seq_encoding\": " + std::to_string(static_cast<int>(cfg.seq_enc)) + ",\n";
        j += "    \"encode_date\": \"2026-04-09\"\n";
        j += "  }\n";
        j += "}";
        return j;
    }
};

}  // namespace lib1fq
