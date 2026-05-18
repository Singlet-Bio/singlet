// SPDX-License-Identifier: MIT
// lib1fq/fastq_encoder.h — FASTQ → .1fq encoder
//
// Reads paired-end FASTQ files (optionally gzipped) and writes .1fq.
// Uses the same EncoderConfig and Writer as SraEncoder.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lib1fq.h"

namespace singlet::fq {

class FastqEncoder {
   public:
    struct Stats {
        uint64_t total_reads = 0;
        uint64_t blocks_written = 0;
        std::string protocol_tag;
        Confidence confidence = Confidence::NONE;
        double encode_seconds = 0.0;
    };

    Stats encode(const std::string& r1_path, const std::string& r2_path,
                 const EncoderConfig& cfg) {
        Stats stats;
        auto t0 = std::chrono::high_resolution_clock::now();

        FILE* r1_fp = open_fastq(r1_path);
        FILE* r2_fp = open_fastq(r2_path);
        if (!r1_fp) throw std::runtime_error("Cannot open R1: " + r1_path);
        if (!r2_fp) throw std::runtime_error("Cannot open R2: " + r2_path);

        // Buffers (used only during probe phase; hot loop uses FastqReader)
        char name_buf[65536], plus_buf[65536];
        char r1_seq_buf[65536], r1_qual_buf[65536];
        char r2_seq_buf[65536], r2_qual_buf[65536];

        // Probe phase: read a few records to determine fixed lengths
        uint16_t r1_len = 0, first_r2_len = 0;
        bool r2_fixed = true, first_read = true;

        // When protocol is forced, we only need length probing (not full detection)
        const bool need_detection = cfg.protocol_tag.empty();

        struct ProbeRead {
            std::vector<uint8_t> r1_seq, r2_seq, r1_qual, r2_qual;
        };
        std::vector<ProbeRead> probe_reads;
        probe_reads.reserve(1000);

        for (int p = 0; p < 1000; ++p) {
            if (!read_fastq_record(r1_fp, name_buf, r1_seq_buf, r1_qual_buf, plus_buf))
                break;
            if (!read_fastq_record(r2_fp, name_buf, r2_seq_buf, r2_qual_buf, plus_buf))
                break;

            uint16_t r1l = std::strlen(r1_seq_buf);
            uint16_t r2l = std::strlen(r2_seq_buf);

            if (first_read) {
                r1_len = r1l;
                first_r2_len = r2l;
                first_read = false;
            } else {
                if (r2l != first_r2_len) r2_fixed = false;
            }

            ProbeRead pr;
            pr.r1_seq.resize(r1l);
            pr.r1_qual.resize(r1l);
            for (uint16_t j = 0; j < r1l; ++j) {
                pr.r1_seq[j] = nuc::ascii_to_num(r1_seq_buf[j]);
                pr.r1_qual[j] = static_cast<uint8_t>(r1_qual_buf[j]) - 33;
            }
            pr.r2_seq.resize(r2l);
            pr.r2_qual.resize(r2l);
            for (uint16_t j = 0; j < r2l; ++j) {
                pr.r2_seq[j] = nuc::ascii_to_num(r2_seq_buf[j]);
                pr.r2_qual[j] = static_cast<uint8_t>(r2_qual_buf[j]) - 33;
            }
            probe_reads.push_back(std::move(pr));
        }

        if (probe_reads.empty()) {
            close_fastq(r1_fp, r1_path);
            close_fastq(r2_fp, r2_path);
            throw std::runtime_error("No reads in input FASTQ files");
        }

        ProtocolCandidate detected;
        if (need_detection) {
            // Full protocol auto-detection from probe reads
            struct ProbeSpot {
                const std::vector<uint8_t>& r1_seq;
                const std::vector<uint8_t>& r2_seq;
                uint16_t r1_len, r2_len;
            };
            std::vector<ProbeSpot> probe_spots;
            probe_spots.reserve(probe_reads.size());
            for (const auto& pr : probe_reads) {
                probe_spots.push_back({pr.r1_seq, pr.r2_seq,
                                       static_cast<uint16_t>(pr.r1_seq.size()),
                                       static_cast<uint16_t>(pr.r2_seq.size())});
            }
            detected = detect_protocol(
                probe_spots, r1_len,
                static_cast<uint16_t>(probe_spots.empty() ? 0 : probe_spots[0].r2_len),
                cfg.whitelist_dirs);

            // ── AUTOFIX-VDB-READ-SWAP-PROTOCOL (FASTQ path) ──────────────────
            // When FASTQ files have R1=cDNA (long) and R2=CB+UMI (short),
            // detect_protocol signals this via reads_swapped=true.
            // Swap probe_reads R1↔R2, update length bookkeeping, and swap
            // file handles so the streaming write loop reads from the correct
            // file as barcode vs cDNA.
            if (detected.reads_swapped) {
                std::cerr << "[1fq-encode] FASTQ read-swap: R1=" << r1_len
                          << "bp(cDNA)\u2194R2=" << first_r2_len
                          << "bp(barcode) — fixing orientation\n";
                for (auto& pr : probe_reads) {
                    std::swap(pr.r1_seq, pr.r2_seq);
                    std::swap(pr.r1_qual, pr.r2_qual);
                }
                // Recompute lengths from swapped probe_reads
                std::swap(r1_len, first_r2_len);
                // r2_fixed: check whether the new R2 (old R1) is fixed-length
                r2_fixed = true;
                for (const auto& pr : probe_reads)
                    if (static_cast<uint16_t>(pr.r2_seq.size()) != first_r2_len)
                        r2_fixed = false;
                // Swap file handles so the streaming loop reads barcode as R1
                std::swap(r1_fp, r2_fp);
                detected.reads_swapped = false;  // consumed
            }
        } else {
            // Skip detection — look up the forced protocol directly
            const CandidateSpec* spec = find_protocol_spec(cfg.protocol_tag);
            if (spec) {
                detected.tag = spec->tag;
                detected.protocol_id = spec->protocol_id;
                detected.bc_offset = spec->bc_offset;
                detected.bc_length = spec->bc_len;
                detected.umi_offset = spec->umi_offset;
                detected.umi_length = spec->umi_len;
                detected.confidence = Confidence::FORCE;
            } else {
                // Unknown tag — store as-is with id=0
                detected.tag = cfg.protocol_tag;
                detected.confidence = Confidence::FORCE;
            }
        }

        // Auto-detect adapter contamination in R2 (mirrors SraEncoder logic).
        // GATED on 5'-capture protocols only: on 3' protocols R2 is the cDNA read
        // and any fixed-base run at position 30+ is a polyA tail from short inserts,
        // not a TSO adapter.  Trimming R2 to 30bp on a 3' protocol destroys barcode
        // assignment ("Discovered 0 barcodes").  Use stricter threshold for UNKNOWN.
        // Minimum surviving position: 40bp floor.
        AssayType auto_assay = protocol_tag_to_assay_type(detected.tag);
        uint16_t auto_r2_maxlen = 0;
        if (cfg.r2_maxlen == 0 && probe_reads.size() >= 100) {
            const bool is_5prime = (auto_assay == AssayType::SC_RNA_5PRIME);
            const bool is_unknown = (auto_assay == AssayType::UNKNOWN &&
                                     detected.confidence < Confidence::MEDIUM);
            if (is_5prime || is_unknown) {
                double adapt_thresh = is_unknown ? 0.85 : 0.75;
                uint16_t max_r2 = 0;
                for (const auto& pr : probe_reads)
                    if (pr.r2_seq.size() > max_r2)
                        max_r2 = static_cast<uint16_t>(pr.r2_seq.size());
                if (max_r2 >= 40) {
                    const int MIN_RUN = 5, MIN_POS = 30;
                    int first_fixed = -1, run = 0;
                    for (int p = MIN_POS; p < max_r2; ++p) {
                        int counts[5] = {};
                        int total = 0;
                        for (const auto& pr : probe_reads) {
                            if (p >= static_cast<int>(pr.r2_seq.size())) continue;
                            uint8_t b = pr.r2_seq[p];
                            if (b < 5) counts[b]++;
                            total++;
                        }
                        if (total < 50) continue;
                        int mc = *std::max_element(counts, counts + 4);
                        if (static_cast<double>(mc) / total > adapt_thresh) {
                            if (++run >= MIN_RUN && first_fixed < 0)
                                first_fixed = p - MIN_RUN + 1;
                        } else {
                            run = 0;
                        }
                    }
                    // Floor: must leave at least 40bp for STAR alignment.
                    if (first_fixed >= 40) {
                        auto_r2_maxlen = static_cast<uint16_t>(first_fixed);
                        std::cerr << "[1fq-encode] WARNING: Adapter/TSO readthrough"
                                     " detected in R2 at position " << auto_r2_maxlen
                                  << " [protocol=" << detected.tag
                                  << ", assay=" << assay_type_name(auto_assay)
                                  << "]. Auto-setting r2_maxlen=" << auto_r2_maxlen << "\n";
                    } else if (first_fixed >= MIN_POS) {
                        std::cerr << "[1fq-encode] INFO: R2 adapter candidate pos="
                                  << first_fixed << " discarded (< 40bp floor)"
                                  << " [protocol=" << detected.tag << "]\n";
                    }
                }
            }
            // For known non-5' protocols: skip adapter detection entirely.
        }

        // Configure and open writer
        WriterConfig wcfg;
        wcfg.codec = cfg.codec;
        wcfg.codec_level = cfg.codec_level;
        wcfg.qual_mode = cfg.qual_mode;
        wcfg.seq_enc = cfg.seq_enc;
        wcfg.block_size = cfg.block_size;
        wcfg.n_streams = 2;
        wcfg.r1_length = r1_len;
        wcfg.r2_length = r2_fixed ? first_r2_len : 0;
        wcfg.protocol_id = detected.protocol_id;
        wcfg.confidence = detected.confidence;
        wcfg.assay_type = protocol_tag_to_assay_type(detected.tag);

        // Wire BC dict + polyA from auto-detection (Phase 2)
        apply_protocol_to_writer(wcfg, detected, cfg.whitelist_dirs);

        // Clamp UMI length to what R1 can actually provide (mirrors sra_encoder fix).
        if (wcfg.umi_length > 0 && wcfg.umi_offset + wcfg.umi_length > r1_len) {
            uint16_t avail = (r1_len > wcfg.umi_offset)
                                 ? static_cast<uint16_t>(r1_len - wcfg.umi_offset)
                                 : uint16_t{0};
            std::cerr << "[1fq-encode] WARNING: R1=" << r1_len
                      << "bp < CB+UMI=" << (wcfg.umi_offset + wcfg.umi_length)
                      << "bp; UMI truncated " << wcfg.umi_length
                      << " \u2192 " << avail << "bp\n";
            wcfg.umi_length = avail;
        }

        // Honor user overrides
        if (cfg.r2_maxlen > 0)
            wcfg.r2_maxlen = cfg.r2_maxlen;
        else if (auto_r2_maxlen > 0)
            wcfg.r2_maxlen = auto_r2_maxlen;
        if (cfg.no_trim) wcfg.polya_trim = false;
        if (cfg.no_dedup) wcfg.deduped = false;
        if (cfg.sort_by_bc) wcfg.sort_by_bc = true;

        stats.protocol_tag = detected.tag;
        stats.confidence = detected.confidence;

        Writer writer;
        writer.open(cfg.output_path.c_str(), wcfg);

        // Feed probe reads
        for (const auto& pr : probe_reads) {
            const uint8_t* r2q = (cfg.qual_mode != QualMode::NONE)
                                     ? pr.r2_qual.data()
                                     : nullptr;
            const uint8_t* r1q = (cfg.qual_mode != QualMode::NONE)
                                     ? pr.r1_qual.data()
                                     : nullptr;
            writer.add_read(pr.r2_seq.data(),
                            static_cast<uint16_t>(pr.r2_seq.size()), r2q,
                            pr.r1_seq.data(),
                            static_cast<uint16_t>(pr.r1_seq.size()), r1q);
            stats.total_reads++;
        }
        probe_reads.clear();

        // Stream remaining reads
        std::vector<uint8_t> r1_num, r2_num, r1_qnum, r2_qnum;
        uint16_t r1l, r2l, r1ql, r2ql;
        while (read_fastq_record(r1_fp, name_buf, r1_seq_buf, r1_qual_buf, plus_buf, r1l, r1ql) &&
               read_fastq_record(r2_fp, name_buf, r2_seq_buf, r2_qual_buf, plus_buf, r2l, r2ql)) {
            r1_num.resize(r1l);
            r1_qnum.resize(r1l);
            r2_num.resize(r2l);
            r2_qnum.resize(r2l);

            for (uint16_t j = 0; j < r1l; ++j) {
                r1_num[j] = nuc::ascii_to_num(r1_seq_buf[j]);
                r1_qnum[j] = static_cast<uint8_t>(r1_qual_buf[j]) - 33;
            }
            for (uint16_t j = 0; j < r2l; ++j) {
                r2_num[j] = nuc::ascii_to_num(r2_seq_buf[j]);
                r2_qnum[j] = static_cast<uint8_t>(r2_qual_buf[j]) - 33;
            }

            // Enforce fixed R2 length contract: the header was committed with
            // stream_lengths[1] = first_r2_len when r2_fixed=true. Any read
            // with a different length must be clamped here, otherwise the
            // block encoder writes variable-length data without length prefixes
            // and the reader decodes with the wrong fixed stride — corrupting
            // the entire remainder of the block.
            if (r2_fixed && r2l != first_r2_len) {
                if (r2l < first_r2_len) {
                    // Pad shorter read: extend with N (2-bit value 4, qual 0)
                    r2_num.resize(first_r2_len);
                    r2_qnum.resize(first_r2_len);
                    std::fill(r2_num.begin() + r2l, r2_num.end(),
                              static_cast<uint8_t>(4));
                    std::fill(r2_qnum.begin() + r2l, r2_qnum.end(),
                              static_cast<uint8_t>(0));
                }
                r2l = first_r2_len;  // truncate if longer, already padded if shorter
            }

            const uint8_t* r2q = (cfg.qual_mode != QualMode::NONE)
                                     ? r2_qnum.data()
                                     : nullptr;
            const uint8_t* r1q = (cfg.qual_mode != QualMode::NONE)
                                     ? r1_qnum.data()
                                     : nullptr;
            writer.add_read(r2_num.data(), r2l, r2q,
                            r1_num.data(), r1l, r1q);
            stats.total_reads++;

            if (cfg.progress_cb && (stats.total_reads % cfg.progress_interval) == 0)
                cfg.progress_cb(stats.total_reads, 0);
        }

        writer.finish();
        stats.blocks_written = stats.total_reads / cfg.block_size +
                               (stats.total_reads % cfg.block_size ? 1 : 0);

        close_fastq(r1_fp, r1_path);
        close_fastq(r2_fp, r2_path);

        auto t1 = std::chrono::high_resolution_clock::now();
        stats.encode_seconds = std::chrono::duration<double>(t1 - t0).count();
        return stats;
    }

   private:
    static bool is_gzipped(const std::string& path) {
        return path.size() >= 3 && path.compare(path.size() - 3, 3, ".gz") == 0;
    }

    static FILE* open_fastq(const std::string& path) {
        if (is_gzipped(path)) {
            std::string cmd = "zcat '" + path + "'";
            return popen(cmd.c_str(), "r");
        }
        return std::fopen(path.c_str(), "r");
    }

    static void close_fastq(FILE* fp, const std::string& path) {
        if (is_gzipped(path))
            pclose(fp);
        else
            std::fclose(fp);
    }

    // Read a FASTQ record, returning stripped lengths of seq and qual.
    // Returns false on EOF.
    static bool read_fastq_record(FILE* fp, char* name, char* seq,
                                  char* qual, char* plus,
                                  uint16_t& seq_len, uint16_t& qual_len) {
        if (!std::fgets(name, 65536, fp)) return false;
        if (!std::fgets(seq, 65536, fp)) return false;
        if (!std::fgets(plus, 65536, fp)) return false;
        if (!std::fgets(qual, 65536, fp)) return false;
        // Strip trailing newlines and return lengths in one pass
        auto strip_len = [](char* s) -> uint16_t {
            size_t len = std::strlen(s);
            while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r'))
                s[--len] = '\0';
            return static_cast<uint16_t>(len);
        };
        seq_len = strip_len(seq);
        qual_len = strip_len(qual);
        return true;
    }
    // Legacy overload for probe phase
    static bool read_fastq_record(FILE* fp, char* name, char* seq,
                                  char* qual, char* plus) {
        uint16_t sl, ql;
        return read_fastq_record(fp, name, seq, qual, plus, sl, ql);
    }

   public:
    // Encode 3-read scATAC FASTQ (R1 genomic + R2 barcode + R3 genomic) → .1fq
    // Layout: stream[0]=R1 genomic (50bp), stream[1]=R3 genomic (49bp),
    //         stream[2]=I2 barcode (16bp, role=I2).
    Stats encode_atac(const std::string& r1_path,     // genomic read 1 (50bp)
                      const std::string& r2_bc_path,  // barcode read (16bp)
                      const std::string& r3_path,     // genomic read 2 (49bp)
                      const EncoderConfig& cfg) {
        Stats stats;
        auto t0 = std::chrono::high_resolution_clock::now();

        FILE* r1_fp = open_fastq(r1_path);
        FILE* r2_fp = open_fastq(r2_bc_path);
        FILE* r3_fp = open_fastq(r3_path);
        if (!r1_fp) throw std::runtime_error("Cannot open R1: " + r1_path);
        if (!r2_fp) throw std::runtime_error("Cannot open R2/barcode: " + r2_bc_path);
        if (!r3_fp) throw std::runtime_error("Cannot open R3: " + r3_path);

        char name_buf[65536], plus_buf[65536];
        char r1_seq_buf[65536], r1_qual_buf[65536];
        char r2_seq_buf[65536], r2_qual_buf[65536];
        char r3_seq_buf[65536], r3_qual_buf[65536];

        // Probe first 8 reads to determine fixed lengths
        uint16_t r1_len = 0, r2_len = 0, r3_len = 0;
        bool first_read = true;
        struct ProbeRead3 {
            std::vector<uint8_t> r1_seq, r2_seq, r3_seq;
            std::vector<uint8_t> r1_qual, r2_qual, r3_qual;
        };
        std::vector<ProbeRead3> probe_reads;
        probe_reads.reserve(8);

        for (int p = 0; p < 8; ++p) {
            if (!read_fastq_record(r1_fp, name_buf, r1_seq_buf, r1_qual_buf, plus_buf))
                break;
            if (!read_fastq_record(r2_fp, name_buf, r2_seq_buf, r2_qual_buf, plus_buf))
                break;
            if (!read_fastq_record(r3_fp, name_buf, r3_seq_buf, r3_qual_buf, plus_buf))
                break;
            uint16_t r1l = static_cast<uint16_t>(std::strlen(r1_seq_buf));
            uint16_t r2l = static_cast<uint16_t>(std::strlen(r2_seq_buf));
            uint16_t r3l = static_cast<uint16_t>(std::strlen(r3_seq_buf));
            if (first_read) {
                r1_len = r1l;
                r2_len = r2l;
                r3_len = r3l;
                first_read = false;
            }
            ProbeRead3 pr;
            pr.r1_seq.resize(r1l);
            pr.r1_qual.resize(r1l);
            pr.r2_seq.resize(r2l);
            pr.r2_qual.resize(r2l);
            pr.r3_seq.resize(r3l);
            pr.r3_qual.resize(r3l);
            for (uint16_t j = 0; j < r1l; ++j) {
                pr.r1_seq[j] = nuc::ascii_to_num(r1_seq_buf[j]);
                pr.r1_qual[j] = static_cast<uint8_t>(r1_qual_buf[j]) - 33;
            }
            for (uint16_t j = 0; j < r2l; ++j) {
                pr.r2_seq[j] = nuc::ascii_to_num(r2_seq_buf[j]);
                pr.r2_qual[j] = static_cast<uint8_t>(r2_qual_buf[j]) - 33;
            }
            for (uint16_t j = 0; j < r3l; ++j) {
                pr.r3_seq[j] = nuc::ascii_to_num(r3_seq_buf[j]);
                pr.r3_qual[j] = static_cast<uint8_t>(r3_qual_buf[j]) - 33;
            }
            probe_reads.push_back(std::move(pr));
        }

        if (probe_reads.empty()) {
            close_fastq(r1_fp, r1_path);
            close_fastq(r2_fp, r2_bc_path);
            close_fastq(r3_fp, r3_path);
            throw std::runtime_error("No reads in ATAC FASTQ files");
        }

        // Protocol: force 10x-atac (or use cfg.protocol_tag if set)
        std::string proto = cfg.protocol_tag.empty() ? "10x-atac" : cfg.protocol_tag;

        singlet::fq::Codec codec = singlet::fq::Codec::ZSTD;
        if (cfg.codec == singlet::fq::Codec::LZ4)
            codec = singlet::fq::Codec::LZ4;
        else if (cfg.codec == singlet::fq::Codec::NONE)
            codec = singlet::fq::Codec::NONE;

        singlet::fq::QualMode qual_mode = singlet::fq::QualMode::BINNED4;
        // (quality is not typically needed for the barcode stream; R1/R3 use default)

        std::string output_path = cfg.output_path;

        WriterConfig wcfg;
        wcfg.codec = codec;
        wcfg.codec_level = cfg.codec_level;
        wcfg.qual_mode = qual_mode;
        wcfg.block_size = cfg.block_size;
        wcfg.r1_length = r1_len;  // R1 genomic (fixed 50bp)
        wcfg.r2_length = r3_len;  // R3 genomic (fixed 49bp)
        wcfg.has_i2_stream = true;
        wcfg.i2_length = r2_len;  // I2 barcode (fixed 16bp)
        wcfg.sort_by_bc = false;  // no BC dict for ATAC 3-read
        wcfg.polya_trim = false;  // no polyA in ATAC

        // Look up ATAC protocol
        const CandidateSpec* spec = find_protocol_spec(proto);
        if (spec) {
            wcfg.protocol_id = spec->protocol_id;
            wcfg.confidence = Confidence::FORCE;
        } else {
            wcfg.protocol_id = 23;  // 10x-atac default id
            wcfg.confidence = Confidence::FORCE;
        }
        wcfg.assay_type = AssayType::SC_ATAC;

        std::string out = output_path.empty() ? r1_path + ".1fq" : output_path;
        Writer writer;
        writer.open(out.c_str(), wcfg);

        // Feed probe reads
        for (const auto& pr : probe_reads) {
            const uint8_t* r1q = (qual_mode != QualMode::NONE) ? pr.r1_qual.data() : nullptr;
            const uint8_t* r3q = (qual_mode != QualMode::NONE) ? pr.r3_qual.data() : nullptr;
            writer.add_read_atac(
                pr.r1_seq.data(), static_cast<uint16_t>(pr.r1_seq.size()), r1q,
                pr.r3_seq.data(), static_cast<uint16_t>(pr.r3_seq.size()), r3q,
                pr.r2_seq.data(), static_cast<uint16_t>(pr.r2_seq.size()));
            stats.total_reads++;
        }
        probe_reads.clear();

        // Stream remaining reads
        std::vector<uint8_t> r1_num, r2_num, r3_num;
        std::vector<uint8_t> r1_qnum, r2_qnum, r3_qnum;
        uint16_t r1l, r1ql, r2l, r2ql, r3l, r3ql;
        while (
            read_fastq_record(r1_fp, name_buf, r1_seq_buf, r1_qual_buf, plus_buf, r1l, r1ql) &&
            read_fastq_record(r2_fp, name_buf, r2_seq_buf, r2_qual_buf, plus_buf, r2l, r2ql) &&
            read_fastq_record(r3_fp, name_buf, r3_seq_buf, r3_qual_buf, plus_buf, r3l, r3ql)) {
            r1_num.resize(r1l);
            r1_qnum.resize(r1l);
            r2_num.resize(r2l);
            r3_num.resize(r3l);
            r3_qnum.resize(r3l);
            for (uint16_t j = 0; j < r1l; ++j) {
                r1_num[j] = nuc::ascii_to_num(r1_seq_buf[j]);
                r1_qnum[j] = static_cast<uint8_t>(r1_qual_buf[j]) - 33;
            }
            for (uint16_t j = 0; j < r2l; ++j)
                r2_num[j] = nuc::ascii_to_num(r2_seq_buf[j]);
            for (uint16_t j = 0; j < r3l; ++j) {
                r3_num[j] = nuc::ascii_to_num(r3_seq_buf[j]);
                r3_qnum[j] = static_cast<uint8_t>(r3_qual_buf[j]) - 33;
            }
            const uint8_t* r1q = (qual_mode != QualMode::NONE) ? r1_qnum.data() : nullptr;
            const uint8_t* r3q = (qual_mode != QualMode::NONE) ? r3_qnum.data() : nullptr;
            writer.add_read_atac(r1_num.data(), r1l, r1q,
                                 r3_num.data(), r3l, r3q,
                                 r2_num.data(), r2l);
            stats.total_reads++;
            if (cfg.progress_cb && (stats.total_reads % cfg.progress_interval) == 0)
                cfg.progress_cb(stats.total_reads, 0);
        }

        writer.finish();
        stats.blocks_written = stats.total_reads / cfg.block_size +
                               (stats.total_reads % cfg.block_size ? 1 : 0);
        stats.protocol_tag = proto;
        stats.confidence = Confidence::FORCE;

        close_fastq(r1_fp, r1_path);
        close_fastq(r2_fp, r2_bc_path);
        close_fastq(r3_fp, r3_path);

        auto t1 = std::chrono::high_resolution_clock::now();
        stats.encode_seconds = std::chrono::duration<double>(t1 - t0).count();
        return stats;
    }
};  // class FastqEncoder

}  // namespace singlet::fq
