// lib1fq/dedup.h — PCR duplicate collapsing (Phase 3)
//
// Offline pass: reads a .1fq file, sorts by identity key (BC+UMI+R2_prefix),
// collapses exact duplicates with counts, writes a new deduped .1fq.

#pragma once

#include "lib1fq.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace lib1fq {

// ── Identity key for dedup — FNV-1a hash of BC+UMI+R2_prefix ──

struct DedupRead {
    uint64_t    identity_hash;
    uint32_t    block_idx;      // source block
    uint32_t    read_idx;       // index within block
    uint32_t    dup_count;      // accumulated count
};

class Deduplicator {
public:
    struct Stats {
        uint64_t total_input = 0;
        uint64_t total_output = 0;   // unique reads
        double   dedup_rate = 0.0;   // 1 - output/input
    };

    // Deduplicate a .1fq file, writing a new deduped .1fq.
    // r2_prefix_len: number of R2 bases to include in identity key (0 = all)
    Stats dedup(const char* input_path, const char* output_path,
                uint16_t r2_prefix_len = 50) {
        Stats stats;

        // Open input
        Reader reader;
        reader.open(input_path);
        const Header& hdr = reader.header();

        // Set up writer with dedup flag
        WriterConfig wcfg;
        wcfg.codec       = static_cast<Codec>(hdr.codec);
        wcfg.codec_level = hdr.codec_level;
        wcfg.qual_mode   = static_cast<QualMode>(hdr.qual_mode);
        wcfg.seq_enc     = static_cast<SeqEncoding>(hdr.seq_encoding);
        wcfg.block_size  = hdr.block_size;
        wcfg.n_streams   = hdr.n_streams;
        wcfg.r1_length   = hdr.stream_lengths[0];
        wcfg.r2_length   = 0;  // variable after trimming/dedup
        wcfg.protocol_id = hdr.protocol_id;
        wcfg.confidence  = static_cast<Confidence>(hdr.confidence);
        wcfg.deduped     = true;

        // Preserve trimming flag (we keep trim lengths from source)
        if (hdr.flags & Flags::TRIMMED) {
            wcfg.polya_trim = false;  // Don't re-trim; source is already trimmed
        }

        // If input has BC dict, copy it for the writer
        if (reader.has_bc_dict()) {
            wcfg.bc_dict = reader.bc_dict();
            wcfg.bc_length = reader.bc_length();
            wcfg.umi_length = reader.umi_length();
            wcfg.bc_offset = 0;
            wcfg.umi_offset = wcfg.bc_length;
        }

        const uint8_t bc_len  = reader.has_bc_dict() ?
                                 static_cast<uint8_t>(reader.bc_length())  : 0;
        const uint8_t umi_len = reader.has_bc_dict() ?
                                 static_cast<uint8_t>(reader.umi_length()) : 0;

        Writer writer;
        writer.open(output_path, wcfg);

        // Process block by block: exact dedup then H1 UMI collapse within BC groups.
        DecodedBlock blk;
        while (reader.read_block(blk)) {
            dedup_block(blk, writer, r2_prefix_len, stats, bc_len, umi_len);
        }

        writer.finish();

        stats.dedup_rate = (stats.total_input > 0) ?
            1.0 - static_cast<double>(stats.total_output) / stats.total_input :
            0.0;

        std::cerr << "[1fq-dedup] Input: " << stats.total_input
                  << " reads, Output: " << stats.total_output
                  << " unique (" << static_cast<int>(stats.dedup_rate * 100)
                  << "% dedup)\n";

        return stats;
    }

private:
    static constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    static constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

    static uint64_t fnv1a(const uint8_t* data, size_t len, uint64_t h = FNV_OFFSET) {
        for (size_t i = 0; i < len; ++i) {
            h ^= data[i];
            h *= FNV_PRIME;
        }
        return h;
    }

    // Compute identity hash for a read: BC + UMI + R2_prefix
    static uint64_t identity_hash(const DecodedBlock& blk, uint32_t i,
                                   uint16_t r2_prefix_len) {
        uint64_t h = FNV_OFFSET;

        // Hash R1 (contains BC+UMI)
        h = fnv1a(blk.r1_seq(i), blk.r1_len(i), h);

        // Hash R2 prefix
        uint16_t r2_len = blk.r2_len(i);
        uint16_t hash_len = (r2_prefix_len > 0 && r2_prefix_len < r2_len)
                           ? r2_prefix_len : r2_len;
        h = fnv1a(blk.r2_seq(i), hash_len, h);

        return h;
    }

    // Check if two reads are truly identical (not just hash collision)
    static bool reads_equal(const DecodedBlock& blk, uint32_t a, uint32_t b,
                             uint16_t r2_prefix_len) {
        if (blk.r1_len(a) != blk.r1_len(b)) return false;
        if (std::memcmp(blk.r1_seq(a), blk.r1_seq(b), blk.r1_len(a)) != 0)
            return false;

        uint16_t r2a = blk.r2_len(a);
        uint16_t r2b = blk.r2_len(b);
        uint16_t cmp_len_a = (r2_prefix_len > 0 && r2_prefix_len < r2a) ? r2_prefix_len : r2a;
        uint16_t cmp_len_b = (r2_prefix_len > 0 && r2_prefix_len < r2b) ? r2_prefix_len : r2b;
        if (cmp_len_a != cmp_len_b) return false;
        return std::memcmp(blk.r2_seq(a), blk.r2_seq(b), cmp_len_a) == 0;
    }

    // Unique read after exact dedup — representative index + accumulated count.
    struct UniqueRead {
        uint32_t read_idx;  // index in DecodedBlock
        uint32_t count;     // accumulated dup count
        bool     absorbed;  // merged into another read by H1 UMI collapse
    };

    // Directional H1 UMI collapse within BC groups.
    // Reads must already be exact-deduped. Directional rule: absorb a neighbour
    // whose UMI differs by exactly 1 base when own count >= 2 * neighbour count.
    static void h1_umi_collapse(const DecodedBlock& blk,
                                std::vector<UniqueRead>& reads,
                                uint8_t bc_len, uint8_t umi_len) {
        if (umi_len == 0 || umi_len > 32 || reads.size() <= 1) return;

        using UmiKey = uint64_t;

        // Pack UMI at byte offset bc_len in R1 (2-bit nucleotide, MSB-first).
        auto pack_umi = [&](size_t si) -> UmiKey {
            const uint8_t* r1 = blk.r1_seq(reads[si].read_idx);
            UmiKey key = 0;
            for (uint8_t j = 0; j < umi_len; ++j)
                key = (key << 2) | (r1[bc_len + j] & 0x3);
            return key;
        };

        // FNV-1a hash of BC bytes — used to group reads by barcode.
        auto bc_hash = [&](size_t si) -> uint64_t {
            const uint8_t* r1 = blk.r1_seq(reads[si].read_idx);
            uint64_t h = FNV_OFFSET;
            for (uint8_t j = 0; j < bc_len; ++j) { h ^= r1[j]; h *= FNV_PRIME; }
            return h;
        };

        // Sort by count descending so high-count reads are processed first.
        std::sort(reads.begin(), reads.end(),
                  [](const UniqueRead& a, const UniqueRead& b) {
                      return a.count > b.count;
                  });

        // Group sorted indices by BC hash.
        std::unordered_map<uint64_t, std::vector<size_t>> bc_groups;
        bc_groups.reserve(reads.size());
        for (size_t i = 0; i < reads.size(); ++i)
            bc_groups[bc_hash(i)].push_back(i);

        // Per-BC directional collapse.
        for (auto& [bk, group] : bc_groups) {
            if (group.size() <= 1) continue;

            // Build UMI → sorted-list index map (first insert = highest count).
            std::unordered_map<UmiKey, size_t> umi_idx;
            umi_idx.reserve(group.size());
            for (size_t gi : group)
                umi_idx.emplace(pack_umi(gi), gi);

            // Sweep from highest to lowest count.
            for (size_t gi : group) {
                if (reads[gi].absorbed) continue;
                UmiKey my_key = pack_umi(gi);

                for (uint8_t pos = 0; pos < umi_len; ++pos) {
                    uint8_t shift = 2 * (umi_len - 1 - pos);
                    uint8_t cur   = static_cast<uint8_t>((my_key >> shift) & 0x3);
                    for (uint8_t alt = 0; alt < 4; ++alt) {
                        if (alt == cur) continue;
                        UmiKey nb_key = (my_key & ~(UmiKey(0x3) << shift))
                                      | (UmiKey(alt) << shift);
                        auto it = umi_idx.find(nb_key);
                        if (it == umi_idx.end()) continue;
                        size_t ngi = it->second;
                        if (reads[ngi].absorbed) continue;
                        // Directional rule: absorb neighbour only if we have >= 2x count.
                        if (reads[gi].count >= 2 * reads[ngi].count) {
                            reads[gi].count += reads[ngi].count;
                            reads[ngi].absorbed = true;
                            umi_idx.erase(it);
                        }
                    }
                }
            }
        }
    }

    void dedup_block(const DecodedBlock& blk, Writer& writer,
                     uint16_t r2_prefix_len, Stats& stats,
                     uint8_t bc_len = 0, uint8_t umi_len = 0) {
        const uint32_t n = blk.n_reads;
        if (n == 0) return;
        stats.total_input += n;

        // ── Phase A: Exact dedup → collect unique reads ──
        std::vector<UniqueRead> unique_reads;
        unique_reads.reserve(n);

        std::unordered_map<uint64_t, std::vector<uint32_t>> hash_groups;
        hash_groups.reserve(n);
        for (uint32_t i = 0; i < n; ++i)
            hash_groups[identity_hash(blk, i, r2_prefix_len)].push_back(i);

        for (auto& [h, indices] : hash_groups) {
            if (indices.size() == 1) {
                uint32_t dc = blk.dup_counts.empty() ? 1 : blk.dup_counts[indices[0]];
                unique_reads.push_back({indices[0], dc, false});
                continue;
            }
            std::vector<bool> used(indices.size(), false);
            for (size_t a = 0; a < indices.size(); ++a) {
                if (used[a]) continue;
                uint32_t rep   = indices[a];
                uint32_t count = blk.dup_counts.empty() ? 1 : blk.dup_counts[rep];
                for (size_t b = a + 1; b < indices.size(); ++b) {
                    if (used[b]) continue;
                    if (reads_equal(blk, rep, indices[b], r2_prefix_len)) {
                        uint32_t dc_b = blk.dup_counts.empty() ? 1
                                                                : blk.dup_counts[indices[b]];
                        count += dc_b;
                        used[b] = true;
                    }
                }
                unique_reads.push_back({rep, count, false});
            }
        }

        // ── Phase B: H1 UMI collapse within BC groups ──
        if (bc_len > 0 && umi_len > 0)
            h1_umi_collapse(blk, unique_reads, bc_len, umi_len);

        // ── Phase C: Emit non-absorbed reads ──
        for (const auto& ur : unique_reads) {
            if (!ur.absorbed) {
                emit_read(blk, ur.read_idx, ur.count, writer);
                stats.total_output++;
            }
        }
    }

    static void emit_read(const DecodedBlock& blk, uint32_t i,
                           uint32_t dup_count, Writer& writer) {
        const uint8_t* r2_qual = blk.r2_quality(i);
        writer.add_read(
            blk.r2_seq(i), blk.r2_len(i), r2_qual,
            blk.r1_seq(i), blk.r1_len(i), nullptr,
            dup_count);
    }
};

} // namespace lib1fq
