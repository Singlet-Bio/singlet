// SPDX-License-Identifier: MIT
// lib1fq/benchmark.h — Codec benchmarking for .1fq (Phase 6)
//
// Tests compression codecs at various levels against synthetic and real
// sequencing data. Reports compressed size, encode throughput, decode
// throughput in a table format.

#pragma once

#include "compress.h"
#include "packing.h"
#include "types.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace singlet::fq {
namespace benchmark {

struct BenchResult {
    std::string codec_name;
    int         level;
    uint32_t    block_size;      // reads per block
    size_t      raw_bytes;
    size_t      compressed_bytes;
    double      ratio;           // raw / compressed
    double      encode_mb_s;     // MB/s
    double      decode_mb_s;     // MB/s
};

// Generate synthetic scRNA-seq-like data:
// - R1: 28 bases (16 BC + 12 UMI), moderate diversity
// - R2: 91 bases (cDNA), some duplicates to simulate PCR
// Returns byte-numeric sequences
struct SyntheticBlock {
    std::vector<uint8_t> r2_packed;     // 2-bit packed R2 column
    std::vector<uint8_t> r1_raw;        // byte-numeric R1 (for BC+UMI varint column)
    std::vector<uint8_t> qual_binned;   // 2-bit binned quality
    size_t raw_size;                     // total uncompressed bytes
};

inline SyntheticBlock generate_block(uint32_t n_reads,
                                     uint16_t r2_len = 91,
                                     uint16_t r1_len = 28,
                                     double dup_rate = 0.5,
                                     uint32_t seed = 42) {
    std::mt19937 rng(seed);
    SyntheticBlock blk;

    // Generate unique R2 sequences, then duplicate some
    uint32_t n_unique = static_cast<uint32_t>(n_reads * (1.0 - dup_rate));
    if (n_unique < 1) n_unique = 1;

    std::vector<std::vector<uint8_t>> unique_r2(n_unique);
    for (uint32_t u = 0; u < n_unique; ++u) {
        unique_r2[u].resize(r2_len);
        for (uint16_t j = 0; j < r2_len; ++j)
            unique_r2[u][j] = rng() % 4;
    }

    // Build packed R2 column (all reads)
    uint32_t packed_r2_bytes = (r2_len + 3) / 4;
    blk.r2_packed.resize(n_reads * packed_r2_bytes);

    for (uint32_t i = 0; i < n_reads; ++i) {
        uint32_t uid = rng() % n_unique;
        pack::pack_2bit(unique_r2[uid].data(), r2_len,
                        blk.r2_packed.data() + i * packed_r2_bytes, nullptr);
    }

    // Build R1 column (byte-numeric, each unique)
    blk.r1_raw.resize(n_reads * r1_len);
    for (uint32_t i = 0; i < n_reads; ++i) {
        for (uint16_t j = 0; j < r1_len; ++j)
            blk.r1_raw[i * r1_len + j] = rng() % 4;
    }

    // Build quality column (2-bit binned: 0=low 1=med 2=high 3=max)
    // Realistic: mostly high quality with occasional dips
    uint32_t packed_q_bytes = (r2_len + 3) / 4;
    blk.qual_binned.resize(n_reads * packed_q_bytes);
    std::vector<uint8_t> qual_raw(r2_len);
    for (uint32_t i = 0; i < n_reads; ++i) {
        for (uint16_t j = 0; j < r2_len; ++j) {
            // 85% high, 10% max, 4% med, 1% low
            uint32_t r = rng() % 100;
            qual_raw[j] = (r < 1) ? 0 : (r < 5) ? 1 : (r < 90) ? 2 : 3;
        }
        pack::pack_2bit(qual_raw.data(), r2_len,
                        blk.qual_binned.data() + i * packed_q_bytes, nullptr);
    }

    blk.raw_size = blk.r2_packed.size() + blk.r1_raw.size() +
                   blk.qual_binned.size();
    return blk;
}

// Benchmark a single codec/level/block against given data
inline BenchResult bench_codec(const std::string& name,
                               Codec codec, int level,
                               const uint8_t* data, size_t data_size,
                               uint32_t block_reads,
                               int iterations = 5) {
    using Clock = std::chrono::high_resolution_clock;

    // Allocate compress/decompress buffers
    size_t bound = compress::compress_bound(codec, data_size);
    std::vector<uint8_t> comp_buf(bound);
    std::vector<uint8_t> decomp_buf(data_size);

    // Warmup
    size_t comp_size = compress::compress_block(
        codec, level, data, data_size,
        comp_buf.data(), comp_buf.size());

    // Benchmark encode
    auto t0 = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        comp_size = compress::compress_block(
            codec, level, data, data_size,
            comp_buf.data(), comp_buf.size());
    }
    double encode_sec = std::chrono::duration<double>(Clock::now() - t0).count();

    // Benchmark decode
    auto t1 = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        compress::decompress_block(
            codec, comp_buf.data(), comp_size,
            decomp_buf.data(), decomp_buf.size());
    }
    double decode_sec = std::chrono::duration<double>(Clock::now() - t1).count();

    double total_mb = static_cast<double>(data_size) * iterations / (1024.0 * 1024.0);

    BenchResult r;
    r.codec_name = name;
    r.level = level;
    r.block_size = block_reads;
    r.raw_bytes = data_size;
    r.compressed_bytes = comp_size;
    r.ratio = static_cast<double>(data_size) / comp_size;
    r.encode_mb_s = total_mb / encode_sec;
    r.decode_mb_s = total_mb / decode_sec;
    return r;
}

// Print results table
inline void print_results(const std::vector<BenchResult>& results) {
    std::fprintf(stderr,
        "%-14s %5s %8s %10s %10s %7s %10s %10s\n",
        "Codec", "Level", "Block", "Raw(KB)", "Comp(KB)", "Ratio",
        "Enc(MB/s)", "Dec(MB/s)");
    std::fprintf(stderr,
        "%-14s %5s %8s %10s %10s %7s %10s %10s\n",
        "--------------", "-----", "--------", "----------",
        "----------", "-------", "----------", "----------");

    for (const auto& r : results) {
        std::fprintf(stderr,
            "%-14s %5d %8u %10.1f %10.1f %7.2f %10.1f %10.1f\n",
            r.codec_name.c_str(), r.level, r.block_size,
            r.raw_bytes / 1024.0, r.compressed_bytes / 1024.0,
            r.ratio, r.encode_mb_s, r.decode_mb_s);
    }
}

// Run the full benchmark matrix
inline std::vector<BenchResult> run_benchmark(
        uint32_t block_reads = 100000,
        uint16_t r2_len = 91,
        double dup_rate = 0.5,
        int iterations = 5) {
    std::fprintf(stderr, "[benchmark] Generating %u reads (R2=%u, dup=%.0f%%)\n",
                 block_reads, r2_len, dup_rate * 100);

    auto blk = generate_block(block_reads, r2_len, 28, dup_rate);

    std::fprintf(stderr, "[benchmark] Raw data: %.1f MB\n",
                 blk.raw_size / (1024.0 * 1024.0));
    std::fprintf(stderr, "[benchmark] Running %d iterations per test...\n\n",
                 iterations);

    // Combine all columns into one buffer (as writer would)
    std::vector<uint8_t> combined;
    combined.insert(combined.end(), blk.r2_packed.begin(), blk.r2_packed.end());
    combined.insert(combined.end(), blk.r1_raw.begin(), blk.r1_raw.end());
    combined.insert(combined.end(), blk.qual_binned.begin(), blk.qual_binned.end());

    std::vector<BenchResult> results;

    // Column-by-column testing for the main data types
    struct Column {
        const char* name;
        const uint8_t* data;
        size_t size;
    };
    Column columns[] = {
        {"R2(2bit)",  blk.r2_packed.data(),   blk.r2_packed.size()},
        {"R1(byte)",  blk.r1_raw.data(),      blk.r1_raw.size()},
        {"Qual(2bit)",blk.qual_binned.data(),  blk.qual_binned.size()},
        {"Combined",  combined.data(),         combined.size()},
    };

    struct CodecSpec {
        const char* name;
        Codec codec;
        int level;
    };
    std::vector<CodecSpec> codecs = {
        {"zstd",   Codec::ZSTD, 1},
        {"zstd",   Codec::ZSTD, 3},
        {"zstd",   Codec::ZSTD, 7},
        {"zstd",   Codec::ZSTD, 19},
#ifdef HAS_LZ4
        {"lz4",    Codec::LZ4,  0},
        {"lz4hc",  Codec::LZ4HC, 9},
#endif
        {"none",   Codec::NONE, 0},
    };

    for (const auto& col : columns) {
        std::fprintf(stderr, "\n── %s column (%zu KB) ──\n",
                     col.name, col.size / 1024);

        std::vector<BenchResult> col_results;
        for (const auto& c : codecs) {
            std::string label = std::string(col.name) + "/" + c.name;
            auto r = bench_codec(label, c.codec, c.level,
                                 col.data, col.size,
                                 block_reads, iterations);
            col_results.push_back(r);
        }
        print_results(col_results);
        results.insert(results.end(), col_results.begin(), col_results.end());
    }

    // Block size sweep (combined data, zstd-3)
    std::fprintf(stderr, "\n── Block size sweep (zstd-3) ──\n");
    uint32_t block_sizes[] = {10000, 50000, 100000, 200000, 500000};
    std::vector<BenchResult> bsweep;
    for (uint32_t bs : block_sizes) {
        uint32_t actual_reads = std::min(bs, block_reads);
        auto small_blk = generate_block(actual_reads, r2_len, 28, dup_rate, 123);

        std::vector<uint8_t> small_combined;
        small_combined.insert(small_combined.end(),
                              small_blk.r2_packed.begin(), small_blk.r2_packed.end());
        small_combined.insert(small_combined.end(),
                              small_blk.r1_raw.begin(), small_blk.r1_raw.end());
        small_combined.insert(small_combined.end(),
                              small_blk.qual_binned.begin(), small_blk.qual_binned.end());

        auto r = bench_codec("blk-sweep", Codec::ZSTD, 3,
                             small_combined.data(), small_combined.size(),
                             actual_reads, iterations);
        bsweep.push_back(r);
    }
    print_results(bsweep);
    results.insert(results.end(), bsweep.begin(), bsweep.end());

    return results;
}

} // namespace benchmark
} // namespace singlet::fq
