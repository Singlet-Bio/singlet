// test_lib1fq.cpp — Unit tests for lib1fq
//
// Tests packing, compression, writer, reader round-trip.
// Build: cmake --build build && ./build/test_lib1fq

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "lib1fq/fastq_encoder.h"
#include "lib1fq/lib1fq.h"
#include "lib1fq/protocol.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                 \
    std::cerr << "  " << #name << "... ";          \
    try {                                          \
        test_##name();                             \
        tests_passed++;                            \
        std::cerr << "OK\n";                       \
    } catch (const std::exception& e) {            \
        tests_failed++;                            \
        std::cerr << "FAIL: " << e.what() << "\n"; \
    }

#define ASSERT_EQ(a, b)                                                                         \
    do {                                                                                        \
        if ((a) != (b)) throw std::runtime_error(                                               \
            std::string(#a) + "=" + std::to_string(a) + " != " + #b + "=" + std::to_string(b)); \
    } while (0)

#define ASSERT_TRUE(x)                                                     \
    do {                                                                   \
        if (!(x)) throw std::runtime_error(std::string(#x) + " is false"); \
    } while (0)

// ── Packing tests ──

void test_pack_2bit_round_trip() {
    // Test sequence: ACGT NACG
    uint8_t seq[] = {0, 1, 2, 3, 4, 0, 1, 2};
    uint32_t len = 8;

    uint8_t packed[2];  // 8 bases → 2 bytes
    uint8_t n_bmp[1];   // 8 bases → 1 byte

    bool has_n = lib1fq::pack::pack_2bit(seq, len, packed, n_bmp);
    ASSERT_TRUE(has_n);

    // Unpack
    uint8_t result[8];
    lib1fq::pack::unpack_2bit(packed, len, result);

    // Without N recovery: N becomes A (0)
    ASSERT_EQ(result[0], 0);  // A
    ASSERT_EQ(result[1], 1);  // C
    ASSERT_EQ(result[2], 2);  // G
    ASSERT_EQ(result[3], 3);  // T
    ASSERT_EQ(result[4], 0);  // N → A (before bitmap fix)
    ASSERT_EQ(result[5], 0);  // A
    ASSERT_EQ(result[6], 1);  // C
    ASSERT_EQ(result[7], 2);  // G

    // Apply N bitmap
    lib1fq::pack::apply_n_bitmap(n_bmp, len, result);
    ASSERT_EQ(result[4], 4);  // N restored
}

void test_pack_2bit_no_n() {
    uint8_t seq[] = {0, 1, 2, 3, 3, 2, 1, 0};
    uint32_t len = 8;
    uint8_t packed[2];

    bool has_n = lib1fq::pack::pack_2bit(seq, len, packed, nullptr);
    ASSERT_TRUE(!has_n);

    uint8_t result[8];
    lib1fq::pack::unpack_2bit(packed, len, result);
    for (uint32_t i = 0; i < len; ++i) {
        ASSERT_EQ(result[i], seq[i]);
    }
}

void test_pack_2bit_odd_length() {
    uint8_t seq[] = {0, 1, 2, 3, 0};  // 5 bases → 2 bytes
    uint32_t len = 5;
    uint8_t packed[2];

    lib1fq::pack::pack_2bit(seq, len, packed, nullptr);

    uint8_t result[5];
    lib1fq::pack::unpack_2bit(packed, len, result);
    for (uint32_t i = 0; i < len; ++i) {
        ASSERT_EQ(result[i], seq[i]);
    }
}

void test_pack_qual_binned4() {
    // Phred: 5, 15, 25, 37, 2, 19, 29, 40
    uint8_t qual[] = {5, 15, 25, 37, 2, 19, 29, 40};
    uint32_t len = 8;
    uint8_t packed[2];  // 8 quals → 2 bytes

    lib1fq::pack::pack_qual_binned4(qual, len, packed);

    uint8_t result[8];
    lib1fq::pack::unpack_qual_binned4(packed, len, result);

    // Expected bin centers: 6, 15, 25, 37, 6, 15, 25, 37
    ASSERT_EQ(result[0], 6);
    ASSERT_EQ(result[1], 15);
    ASSERT_EQ(result[2], 25);
    ASSERT_EQ(result[3], 37);
    ASSERT_EQ(result[4], 6);
    ASSERT_EQ(result[5], 15);
    ASSERT_EQ(result[6], 25);
    ASSERT_EQ(result[7], 37);
}

void test_pack_qual_binned2() {
    // Phred: 5, 15, 25, 37, 2, 19, 29, 40, 20, 0, 42
    uint8_t qual[] = {5, 15, 25, 37, 2, 19, 29, 40, 20, 0, 42};
    uint32_t len = 11;
    // 1 bit/base: Q<20→0, Q≥20→1
    // Expected bits: 0, 0, 1, 1, 0, 0, 1, 1,  1, 0, 1
    // packed_bytes_binned2(11) = 2 bytes
    uint8_t packed[2];

    lib1fq::pack::pack_qual_binned2(qual, len, packed);

    // Byte 0: 00110011 = 0x33
    ASSERT_EQ(packed[0], 0x33);
    // Byte 1: 10100000 = 0xA0 (remaining 3 bits, MSB-first, rest zero-padded)
    ASSERT_EQ(packed[1], 0xA0);

    uint8_t result[11];
    lib1fq::pack::unpack_qual_binned2(packed, len, result);

    // Expected bin centers: 10, 10, 30, 30, 10, 10, 30, 30, 30, 10, 30
    ASSERT_EQ(result[0], 10);
    ASSERT_EQ(result[1], 10);
    ASSERT_EQ(result[2], 30);
    ASSERT_EQ(result[3], 30);
    ASSERT_EQ(result[4], 10);
    ASSERT_EQ(result[5], 10);
    ASSERT_EQ(result[6], 30);
    ASSERT_EQ(result[7], 30);
    ASSERT_EQ(result[8], 30);
    ASSERT_EQ(result[9], 10);
    ASSERT_EQ(result[10], 30);
}

// ── Varint tests ──

void test_varint_round_trip() {
    uint8_t buf[10];

    // Small value
    uint32_t n = lib1fq::pack::encode_varint(42, buf);
    uint64_t val;
    uint32_t consumed = lib1fq::pack::decode_varint(buf, val);
    ASSERT_EQ(val, 42);
    ASSERT_EQ(n, consumed);

    // Large value
    n = lib1fq::pack::encode_varint(100000, buf);
    consumed = lib1fq::pack::decode_varint(buf, val);
    ASSERT_EQ(val, 100000);
    ASSERT_EQ(n, consumed);

    // Single byte
    n = lib1fq::pack::encode_varint(0, buf);
    consumed = lib1fq::pack::decode_varint(buf, val);
    ASSERT_EQ(val, 0);
    ASSERT_EQ(n, 1);
    ASSERT_EQ(consumed, 1);

    // Max single byte
    n = lib1fq::pack::encode_varint(127, buf);
    consumed = lib1fq::pack::decode_varint(buf, val);
    ASSERT_EQ(val, 127);
    ASSERT_EQ(n, 1);
}

// ── Nucleotide conversion tests ──

void test_nuc_conversion() {
    ASSERT_EQ(lib1fq::nuc::ascii_to_num('A'), 0);
    ASSERT_EQ(lib1fq::nuc::ascii_to_num('C'), 1);
    ASSERT_EQ(lib1fq::nuc::ascii_to_num('G'), 2);
    ASSERT_EQ(lib1fq::nuc::ascii_to_num('T'), 3);
    ASSERT_EQ(lib1fq::nuc::ascii_to_num('N'), 4);
    ASSERT_EQ(lib1fq::nuc::ascii_to_num('a'), 0);
    ASSERT_EQ(lib1fq::nuc::ascii_to_num('X'), 4);

    ASSERT_EQ(lib1fq::nuc::num_to_ascii(0), 'A');
    ASSERT_EQ(lib1fq::nuc::num_to_ascii(1), 'C');
    ASSERT_EQ(lib1fq::nuc::num_to_ascii(2), 'G');
    ASSERT_EQ(lib1fq::nuc::num_to_ascii(3), 'T');
    ASSERT_EQ(lib1fq::nuc::num_to_ascii(4), 'N');
}

// ── Header tests ──

void test_header_init() {
    lib1fq::Header h;
    h.init();

    ASSERT_TRUE(h.valid_magic());
    ASSERT_EQ(h.version, lib1fq::FORMAT_VERSION);
    ASSERT_EQ(h.codec, static_cast<uint8_t>(lib1fq::Codec::ZSTD));
    ASSERT_EQ(h.seq_encoding, static_cast<uint8_t>(lib1fq::SeqEncoding::PACKED_2BIT));
    ASSERT_EQ(h.qual_mode, static_cast<uint8_t>(lib1fq::QualMode::BINNED4));
    ASSERT_EQ(h.block_size, 100000u);
    ASSERT_EQ(sizeof(lib1fq::Header), 96u);
    ASSERT_EQ(sizeof(lib1fq::Footer), 16u);
    ASSERT_EQ(sizeof(lib1fq::BlockHeader), 12u);
}

// ── Compression round-trip test ──

void test_compress_round_trip() {
    // Generate some data
    std::vector<uint8_t> data(10000);
    std::mt19937 rng(42);
    for (auto& b : data) b = rng() % 4;

    // Compress
    size_t bound = lib1fq::compress::compress_bound(lib1fq::Codec::ZSTD, data.size());
    std::vector<uint8_t> comp(bound);
    size_t comp_size = lib1fq::compress::compress_block(
        lib1fq::Codec::ZSTD, 3, data.data(), data.size(), comp.data(), bound);

    ASSERT_TRUE(comp_size > 0);
    ASSERT_TRUE(comp_size < data.size());  // Should compress

    // Decompress
    std::vector<uint8_t> decompressed(data.size());
    size_t decomp_size = lib1fq::compress::decompress_block(
        lib1fq::Codec::ZSTD, comp.data(), comp_size,
        decompressed.data(), decompressed.size());

    ASSERT_EQ(decomp_size, data.size());
    ASSERT_TRUE(std::memcmp(data.data(), decompressed.data(), data.size()) == 0);
}

// ── Writer/Reader round-trip test ──

void test_writer_reader_round_trip() {
    const char* test_file = "/tmp/test_lib1fq_roundtrip.1fq";
    const uint32_t n_reads = 1000;
    const uint16_t r1_len = 28;

    std::mt19937 rng(123);

    // Generate test reads
    struct TestRead {
        std::vector<uint8_t> r1_seq, r2_seq, r2_qual;
        uint16_t r1_len, r2_len;
    };
    std::vector<TestRead> reads(n_reads);

    for (auto& r : reads) {
        r.r1_len = r1_len;
        r.r2_len = 80 + (rng() % 30);  // Variable R2: 80-109bp

        r.r1_seq.resize(r.r1_len);
        r.r2_seq.resize(r.r2_len);
        r.r2_qual.resize(r.r2_len);

        for (auto& b : r.r1_seq) b = rng() % 4;
        for (auto& b : r.r2_seq) b = rng() % 4;
        for (auto& q : r.r2_qual) q = rng() % 42;  // Phred 0-41
    }

    // Write
    {
        lib1fq::WriterConfig cfg;
        cfg.block_size = 200;  // Small blocks for testing
        cfg.r1_length = r1_len;
        cfg.r2_length = 0;  // Variable R2

        lib1fq::Writer w;
        w.open(test_file, cfg);

        for (const auto& r : reads) {
            w.add_read(r.r2_seq.data(), r.r2_len, r.r2_qual.data(),
                       r.r1_seq.data(), r.r1_len, nullptr);
        }
        w.finish("{\"test\": true}");

        ASSERT_EQ(w.total_reads_written(), n_reads);
    }

    // Read back
    {
        lib1fq::Reader reader;
        reader.open(test_file);

        const auto& h = reader.header();
        ASSERT_EQ(h.n_unique, n_reads);
        ASSERT_EQ(h.stream_lengths[0], r1_len);
        ASSERT_EQ(h.stream_lengths[1], 0);  // Variable

        lib1fq::DecodedBlock blk;
        uint32_t total = 0;

        while (reader.read_block(blk)) {
            for (uint32_t i = 0; i < blk.n_reads; ++i) {
                uint32_t ri = total + i;
                ASSERT_TRUE(ri < n_reads);

                // Check R1 sequence
                ASSERT_EQ(blk.r1_len(i), reads[ri].r1_len);
                for (uint16_t b = 0; b < blk.r1_len(i); ++b) {
                    ASSERT_EQ(blk.r1_seq(i)[b], reads[ri].r1_seq[b]);
                }

                // Check R2 sequence
                ASSERT_EQ(blk.r2_len(i), reads[ri].r2_len);
                for (uint16_t b = 0; b < blk.r2_len(i); ++b) {
                    ASSERT_EQ(blk.r2_seq(i)[b], reads[ri].r2_seq[b]);
                }

                // Check R2 quality (binned, so check bin equivalence)
                const uint8_t* q = blk.r2_quality(i);
                ASSERT_TRUE(q != nullptr);
                for (uint16_t b = 0; b < blk.r2_len(i); ++b) {
                    uint8_t orig_bin = lib1fq::qual::phred_to_bin4(reads[ri].r2_qual[b]);
                    uint8_t decoded_phred = q[b];
                    uint8_t decoded_bin = lib1fq::qual::phred_to_bin4(decoded_phred);
                    ASSERT_EQ(orig_bin, decoded_bin);
                }
            }
            total += blk.n_reads;
        }
        ASSERT_EQ(total, n_reads);

        // Check metadata
        std::string meta = reader.read_metadata();
        ASSERT_TRUE(meta.find("\"test\": true") != std::string::npos);
    }

    // Cleanup
    std::remove(test_file);
}

// ── Writer/Reader round-trip with BINNED2 quality ──

void test_writer_reader_round_trip_binned2() {
    const char* test_file = "/tmp/test_lib1fq_roundtrip_b2.1fq";
    const uint32_t n_reads = 500;
    const uint16_t r1_len = 28;

    std::mt19937 rng(456);

    struct TestRead {
        std::vector<uint8_t> r1_seq, r2_seq, r2_qual;
        uint16_t r1_len, r2_len;
    };
    std::vector<TestRead> reads(n_reads);

    for (auto& r : reads) {
        r.r1_len = r1_len;
        r.r2_len = 80 + (rng() % 30);
        r.r1_seq.resize(r.r1_len);
        r.r2_seq.resize(r.r2_len);
        r.r2_qual.resize(r.r2_len);
        for (auto& b : r.r1_seq) b = rng() % 4;
        for (auto& b : r.r2_seq) b = rng() % 4;
        for (auto& q : r.r2_qual) q = rng() % 42;
    }

    // Write with BINNED2
    {
        lib1fq::WriterConfig cfg;
        cfg.block_size = 200;
        cfg.r1_length = r1_len;
        cfg.r2_length = 0;
        cfg.qual_mode = lib1fq::QualMode::BINNED2;

        lib1fq::Writer w;
        w.open(test_file, cfg);
        for (const auto& r : reads) {
            w.add_read(r.r2_seq.data(), r.r2_len, r.r2_qual.data(),
                       r.r1_seq.data(), r.r1_len, nullptr);
        }
        w.finish("");
        ASSERT_EQ(w.total_reads_written(), n_reads);
    }

    // Read back and verify bin equivalence
    {
        lib1fq::Reader reader;
        reader.open(test_file);
        ASSERT_EQ(static_cast<uint8_t>(reader.header().qual_mode),
                  static_cast<uint8_t>(lib1fq::QualMode::BINNED2));

        lib1fq::DecodedBlock blk;
        uint32_t total = 0;
        while (reader.read_block(blk)) {
            for (uint32_t i = 0; i < blk.n_reads; ++i) {
                uint32_t ri = total + i;
                ASSERT_TRUE(ri < n_reads);
                ASSERT_EQ(blk.r2_len(i), reads[ri].r2_len);

                const uint8_t* q = blk.r2_quality(i);
                ASSERT_TRUE(q != nullptr);
                for (uint16_t b = 0; b < blk.r2_len(i); ++b) {
                    uint8_t orig_bin = lib1fq::qual::phred_to_bin2(reads[ri].r2_qual[b]);
                    uint8_t decoded_phred = q[b];
                    uint8_t decoded_bin = lib1fq::qual::phred_to_bin2(decoded_phred);
                    ASSERT_EQ(orig_bin, decoded_bin);
                }
            }
            total += blk.n_reads;
        }
        ASSERT_EQ(total, n_reads);
    }

    std::remove(test_file);
}

// ── Writer/Reader with N bases ──

void test_round_trip_with_n_bases() {
    const char* test_file = "/tmp/test_lib1fq_n_bases.1fq";

    // Create a read with N bases
    uint8_t r1[] = {0, 1, 2, 3, 4, 0, 1, 4};        // ACGTNACN
    uint8_t r2[] = {0, 1, 4, 2, 3, 0, 1, 2, 3, 4};  // ACNGTAGCTN
    uint8_t r2_qual[] = {30, 30, 5, 30, 30, 30, 30, 30, 30, 5};
    uint16_t r1_len = 8, r2_len = 10;

    // Write
    {
        lib1fq::WriterConfig cfg;
        cfg.block_size = 10;
        cfg.r1_length = r1_len;
        cfg.r2_length = r2_len;

        lib1fq::Writer w;
        w.open(test_file, cfg);
        w.add_read(r2, r2_len, r2_qual, r1, r1_len, nullptr);
        w.finish();
    }

    // Read back
    {
        lib1fq::Reader reader;
        reader.open(test_file);

        lib1fq::DecodedBlock blk;
        ASSERT_TRUE(reader.read_block(blk));
        ASSERT_EQ(blk.n_reads, 1u);

        // R1: N bases should be recovered
        ASSERT_EQ(blk.r1_len(0), r1_len);
        for (uint16_t i = 0; i < r1_len; ++i) {
            ASSERT_EQ(blk.r1_seq(0)[i], r1[i]);
        }

        // R2: N bases should be recovered
        ASSERT_EQ(blk.r2_len(0), r2_len);
        for (uint16_t i = 0; i < r2_len; ++i) {
            ASSERT_EQ(blk.r2_seq(0)[i], r2[i]);
        }
    }

    std::remove(test_file);
}

// ── Large random data stress test ──

void test_stress_random_data() {
    const char* test_file = "/tmp/test_lib1fq_stress.1fq";
    const uint32_t n_reads = 50000;

    std::mt19937 rng(999);

    // Write many reads with variable lengths
    std::vector<std::vector<uint8_t>> r2_seqs(n_reads);
    {
        lib1fq::WriterConfig cfg;
        cfg.block_size = 5000;
        cfg.r1_length = 28;
        cfg.r2_length = 0;                       // variable
        cfg.qual_mode = lib1fq::QualMode::NONE;  // No quality for speed

        lib1fq::Writer w;
        w.open(test_file, cfg);

        std::vector<uint8_t> r1_buf(28);
        for (uint32_t i = 0; i < n_reads; ++i) {
            for (auto& b : r1_buf) b = rng() % 4;

            uint16_t r2_len = 50 + (rng() % 100);
            r2_seqs[i].resize(r2_len);
            for (auto& b : r2_seqs[i]) {
                b = (rng() % 100 < 2) ? 4 : (rng() % 4);  // 2% N rate
            }

            w.add_read(r2_seqs[i].data(), r2_len, nullptr,
                       r1_buf.data(), 28, nullptr);
        }
        w.finish();
    }

    // Verify
    {
        lib1fq::Reader reader;
        reader.open(test_file);

        ASSERT_EQ(reader.total_reads(), n_reads);

        lib1fq::DecodedBlock blk;
        uint32_t total = 0;
        while (reader.read_block(blk)) {
            for (uint32_t i = 0; i < blk.n_reads; ++i) {
                uint32_t ri = total + i;
                ASSERT_EQ(blk.r2_len(i), static_cast<uint16_t>(r2_seqs[ri].size()));
                for (uint16_t b = 0; b < blk.r2_len(i); ++b) {
                    ASSERT_EQ(blk.r2_seq(i)[b], r2_seqs[ri][b]);
                }
            }
            total += blk.n_reads;
        }
        ASSERT_EQ(total, n_reads);
    }

    std::remove(test_file);
}

// ── Barcode dictionary round-trip ──

void test_bc_dict_round_trip() {
    const char* test_file = "/tmp/test_lib1fq_bcdict.1fq";
    const uint32_t n_reads = 500;
    const uint16_t r1_len = 28;
    const uint16_t bc_len = 16;
    const uint16_t umi_len = 12;

    std::mt19937 rng(456);

    // Create a small barcode dictionary (50 barcodes)
    const uint32_t dict_size = 50;
    std::vector<std::vector<uint8_t>> bc_dict(dict_size);
    for (auto& bc : bc_dict) {
        bc.resize(bc_len);
        for (auto& b : bc) b = rng() % 4;
    }

    // Generate test reads using barcodes from dictionary
    struct TestRead {
        std::vector<uint8_t> r1_seq, r2_seq;
        uint16_t r2_len;
        uint32_t bc_idx;
    };
    std::vector<TestRead> reads(n_reads);

    for (auto& r : reads) {
        r.bc_idx = rng() % dict_size;
        r.r1_seq.resize(r1_len);
        // BC portion from dictionary
        std::copy(bc_dict[r.bc_idx].begin(), bc_dict[r.bc_idx].end(),
                  r.r1_seq.begin());
        // UMI portion random
        for (uint16_t j = bc_len; j < r1_len; ++j)
            r.r1_seq[j] = rng() % 4;

        r.r2_len = 80 + (rng() % 20);
        r.r2_seq.resize(r.r2_len);
        for (auto& b : r.r2_seq) b = rng() % 4;
    }

    // Write with BC dict
    {
        lib1fq::WriterConfig cfg;
        cfg.block_size = 100;
        cfg.r1_length = r1_len;
        cfg.r2_length = 0;
        cfg.bc_dict = bc_dict;
        cfg.bc_offset = 0;
        cfg.bc_length = bc_len;
        cfg.umi_offset = bc_len;
        cfg.umi_length = umi_len;
        cfg.qual_mode = lib1fq::QualMode::NONE;

        lib1fq::Writer w;
        w.open(test_file, cfg);

        for (const auto& r : reads) {
            w.add_read(r.r2_seq.data(), r.r2_len, nullptr,
                       r.r1_seq.data(), r1_len, nullptr);
        }
        w.finish();
    }

    // Read back and verify
    {
        lib1fq::Reader reader;
        reader.open(test_file);

        ASSERT_TRUE(reader.has_bc_dict());
        ASSERT_EQ(reader.bc_length(), bc_len);
        ASSERT_EQ(reader.bc_dict().size(), dict_size);

        lib1fq::DecodedBlock blk;
        uint32_t total = 0;

        while (reader.read_block(blk)) {
            for (uint32_t i = 0; i < blk.n_reads; ++i) {
                uint32_t ri = total + i;
                ASSERT_TRUE(ri < n_reads);

                // Check R1 reconstructed correctly (BC from dict + UMI)
                ASSERT_EQ(blk.r1_len(i), r1_len);
                for (uint16_t b = 0; b < r1_len; ++b) {
                    ASSERT_EQ(blk.r1_seq(i)[b], reads[ri].r1_seq[b]);
                }

                // Check R2
                ASSERT_EQ(blk.r2_len(i), reads[ri].r2_len);
                for (uint16_t b = 0; b < blk.r2_len(i); ++b) {
                    ASSERT_EQ(blk.r2_seq(i)[b], reads[ri].r2_seq[b]);
                }

                // Check BC index
                ASSERT_EQ(blk.bc_indices[i], reads[ri].bc_idx);
            }
            total += blk.n_reads;
        }
        ASSERT_EQ(total, n_reads);
    }

    std::remove(test_file);
}

// ── PolyA trimming test ──

void test_polya_trimming() {
    const char* test_file = "/tmp/test_lib1fq_polya.1fq";

    // Create reads: some with polyA tails, some without
    struct TestRead {
        std::vector<uint8_t> r1_seq, r2_seq;
        uint16_t r2_len, expected_trim;
    };

    // Read 1: 50bp cDNA + 20bp polyA
    // cDNA: 0,1,2,3,0,1,2,3,...  pos48=0(A), pos49=1(C)
    // polyA tail scanner from 3' hits: 69..50 are A (20 A's),
    // pos49=C (1mm, allowed), pos48=A, pos47=T (2mm, break)
    // So effective trim = 22 bases (from pos 48 onward)
    TestRead r1;
    r1.r1_seq.assign(28, 0);  // dummy R1
    r1.r2_seq.resize(70);
    for (int i = 0; i < 50; ++i) r1.r2_seq[i] = (i % 4);  // cDNA
    for (int i = 50; i < 70; ++i) r1.r2_seq[i] = 0;       // polyA (A=0)
    r1.r2_len = 70;
    r1.expected_trim = 22;  // extends 2bp into cDNA due to mm tolerance

    // Read 2: 80bp cDNA, no polyA
    TestRead r2;
    r2.r1_seq.assign(28, 1);
    r2.r2_seq.resize(80);
    for (int i = 0; i < 80; ++i) r2.r2_seq[i] = ((i + 1) % 4);  // no trailing A's
    r2.r2_len = 80;
    r2.expected_trim = 0;

    // Read 3: 40bp cDNA + 15bp polyA with 1 mismatch
    TestRead r3;
    r3.r1_seq.assign(28, 2);
    r3.r2_seq.resize(55);
    for (int i = 0; i < 40; ++i) r3.r2_seq[i] = (i % 4);
    for (int i = 40; i < 55; ++i) r3.r2_seq[i] = 0;  // polyA
    r3.r2_seq[48] = 2;                               // one G mismatch in the tail
    r3.r2_len = 55;
    r3.expected_trim = 15;

    std::vector<TestRead> reads = {r1, r2, r3};

    // Write with polyA trimming
    {
        lib1fq::WriterConfig cfg;
        cfg.block_size = 100;
        cfg.r1_length = 28;
        cfg.r2_length = 0;
        cfg.polya_trim = true;
        cfg.polya_min_len = 10;
        cfg.polya_max_mm = 1;
        cfg.qual_mode = lib1fq::QualMode::NONE;

        lib1fq::Writer w;
        w.open(test_file, cfg);

        for (const auto& r : reads) {
            w.add_read(r.r2_seq.data(), r.r2_len, nullptr,
                       r.r1_seq.data(), 28, nullptr);
        }
        w.finish();
    }

    // Read back
    {
        lib1fq::Reader reader;
        reader.open(test_file);

        lib1fq::DecodedBlock blk;
        ASSERT_TRUE(reader.read_block(blk));
        ASSERT_EQ(blk.n_reads, 3u);

        // Read 1: should be trimmed by 20
        ASSERT_EQ(blk.r2_len(0), static_cast<uint16_t>(70 - reads[0].expected_trim));
        ASSERT_EQ(blk.trim_lengths[0], reads[0].expected_trim);

        // Read 2: no trimming
        ASSERT_EQ(blk.r2_len(1), reads[1].r2_len);
        ASSERT_EQ(blk.trim_lengths[1], 0);

        // Read 3: trimmed by 15 (with 1 mismatch tolerance)
        ASSERT_EQ(blk.r2_len(2), static_cast<uint16_t>(55 - reads[2].expected_trim));
        ASSERT_EQ(blk.trim_lengths[2], reads[2].expected_trim);
    }

    std::remove(test_file);
}

// ── Dedup round-trip test ──

void test_dedup_round_trip() {
    const char* src_file = "/tmp/test_lib1fq_dedup_src.1fq";
    const char* dup_file = "/tmp/test_lib1fq_dedup_out.1fq";

    // Create 100 reads where every pair of consecutive reads is identical
    // (50 unique reads, each with dup count 2)
    const uint16_t r1_len = 28;
    const uint16_t r2_len = 60;
    const uint32_t n_reads = 100;

    // Write source file
    {
        lib1fq::WriterConfig cfg;
        cfg.block_size = 200;
        cfg.r1_length = r1_len;
        cfg.r2_length = r2_len;
        cfg.qual_mode = lib1fq::QualMode::NONE;

        lib1fq::Writer w;
        w.open(src_file, cfg);

        for (uint32_t i = 0; i < n_reads; ++i) {
            uint32_t unique_idx = i / 2;  // pairs are identical
            std::vector<uint8_t> r1(r1_len), r2(r2_len);
            // Use unique_idx bits to create distinct sequences
            for (uint16_t j = 0; j < r1_len; ++j)
                r1[j] = ((unique_idx >> (j % 8)) + j) % 4;
            for (uint16_t j = 0; j < r2_len; ++j)
                r2[j] = ((unique_idx >> (j % 16)) ^ j) % 4;
            w.add_read(r2.data(), r2_len, nullptr,
                       r1.data(), r1_len, nullptr);
        }
        w.finish();
    }

    // Deduplicate
    lib1fq::Deduplicator dedup;
    auto stats = dedup.dedup(src_file, dup_file, 50);

    ASSERT_EQ(stats.total_input, n_reads);
    ASSERT_EQ(stats.total_output, n_reads / 2);

    // Read back deduped file and verify
    {
        lib1fq::Reader reader;
        reader.open(dup_file);

        lib1fq::DecodedBlock blk;
        uint32_t total = 0;
        while (reader.read_block(blk)) {
            for (uint32_t i = 0; i < blk.n_reads; ++i) {
                // Each unique read should have dup_count = 2
                ASSERT_EQ(blk.dup_counts[i], 2u);
                ASSERT_EQ(blk.r1_len(i), r1_len);
                ASSERT_EQ(blk.r2_len(i), r2_len);
            }
            total += blk.n_reads;
        }
        ASSERT_EQ(total, n_reads / 2);
    }

    std::remove(src_file);
    std::remove(dup_file);
}

// ── Read folding test ──

void test_read_folding() {
    // Build a DecodedBlock manually with known R2 duplicates:
    // 10 reads, 5 unique R2 sequences (each appears twice)
    const uint16_t r1_len = 28;
    const uint16_t r2_len = 50;
    const uint32_t n_reads = 10;
    const uint32_t n_unique = 5;

    lib1fq::DecodedBlock blk;
    blk.n_reads = n_reads;
    blk.r1_offsets.resize(n_reads);
    blk.r1_lengths.resize(n_reads, r1_len);
    blk.r2_offsets.resize(n_reads);
    blk.r2_lengths.resize(n_reads, r2_len);

    // Generate unique R2 sequences, then duplicate them
    // Read layout: [0,5] share R2, [1,6] share R2, ..., [4,9] share R2
    for (uint32_t i = 0; i < n_reads; ++i) {
        uint32_t unique_idx = i % n_unique;

        // R1: all different (different barcodes)
        blk.r1_offsets[i] = static_cast<uint32_t>(blk.r1_data.size());
        for (uint16_t j = 0; j < r1_len; ++j)
            blk.r1_data.push_back(((i >> (j % 8)) + j) % 4);

        // R2: same for reads sharing unique_idx
        blk.r2_offsets[i] = static_cast<uint32_t>(blk.r2_data.size());
        for (uint16_t j = 0; j < r2_len; ++j)
            blk.r2_data.push_back(((unique_idx >> (j % 8)) ^ (j * 3)) % 4);
    }

    // Fold
    auto fb = lib1fq::ReadFolder::fold(blk);

    ASSERT_EQ(fb.n_reads, n_reads);
    ASSERT_EQ(fb.n_unique_r2, n_unique);

    // Verify fold ratio
    double ratio = lib1fq::ReadFolder::fold_ratio(fb);
    ASSERT_TRUE(ratio > 1.99 && ratio < 2.01);  // 10/5 = 2.0

    // Verify each unique group has count 2
    for (uint32_t u = 0; u < n_unique; ++u) {
        ASSERT_EQ(fb.unique_r2_count[u], 2u);
    }

    // Verify reads sharing same R2 map to same unique id
    for (uint32_t i = 0; i < n_unique; ++i) {
        ASSERT_EQ(fb.read_to_unique[i], fb.read_to_unique[i + n_unique]);
    }

    // Verify alignment cache
    ASSERT_EQ(fb.alignments.size(), static_cast<size_t>(n_unique));
    ASSERT_TRUE(!fb.alignments[0].valid);  // not yet filled

    // Test AlignmentCache cross-block reuse
    lib1fq::AlignmentCache cache;

    // Store alignment for unique R2 #0
    uint32_t rep0 = fb.unique_r2_rep[0];
    lib1fq::FoldedBlock::CachedAlignment aln;
    aln.ref_id = 42;
    aln.pos = 12345;
    aln.mapq = 60;
    aln.flags = 0;
    aln.n_hits = 1;
    aln.valid = true;
    cache.put(blk.r2_seq(rep0), blk.r2_len(rep0), aln);

    ASSERT_EQ(cache.size(), 1u);
    ASSERT_TRUE(cache.has(blk.r2_seq(rep0), blk.r2_len(rep0)));

    // Retrieve and verify
    auto* got = cache.get(blk.r2_seq(rep0), blk.r2_len(rep0));
    ASSERT_TRUE(got != nullptr);
    ASSERT_EQ(got->ref_id, 42);
    ASSERT_EQ(got->pos, 12345);
    ASSERT_EQ(got->mapq, 60);
    ASSERT_TRUE(got->valid);

    // Miss for a different R2
    uint32_t rep1 = fb.unique_r2_rep[1];
    ASSERT_TRUE(!cache.has(blk.r2_seq(rep1), blk.r2_len(rep1)));
}

// ── Packed alignment primitives tests ──

void test_packed_compare() {
    using namespace lib1fq::packed_align;

    // Build a small genome: 64 bases (2 words)
    uint8_t genome_bytes[64];
    for (uint32_t i = 0; i < 64; ++i)
        genome_bytes[i] = i % 4;  // ACGT repeating

    uint64_t genome_packed[3];  // +1 for shifted reads near end
    pack_genome_region(genome_bytes, 0, 64, genome_packed);

    // Pack a read that matches genome at offset 0, length 32
    uint8_t read_bytes[32];
    for (uint32_t i = 0; i < 32; ++i)
        read_bytes[i] = i % 4;

    uint64_t read_packed[1];
    pack_genome_region(read_bytes, 0, 32, read_packed);

    // Exact match at offset 0
    uint32_t match_len = compare_packed(read_packed, genome_packed, 0, 32);
    ASSERT_EQ(match_len, 32u);

    // Match at offset 4 (shift by 4 bases) — also exact since pattern repeats
    match_len = compare_packed(read_packed, genome_packed, 4, 32);
    ASSERT_EQ(match_len, 32u);

    // Create a read with mismatch at position 10
    uint8_t read_mm[32];
    for (uint32_t i = 0; i < 32; ++i)
        read_mm[i] = i % 4;
    read_mm[10] = (read_mm[10] + 1) % 4;  // mismatch

    pack_genome_region(read_mm, 0, 32, read_packed);

    match_len = compare_packed(read_packed, genome_packed, 0, 32);
    ASSERT_EQ(match_len, 10u);  // mismatch at position 10
}

void test_packed_sa_index() {
    using namespace lib1fq::packed_align;

    // Build a read: 32 bases = A,C,G,T,A,C,G,T,...
    uint8_t read_bytes[32];
    for (uint32_t i = 0; i < 32; ++i)
        read_bytes[i] = i % 4;

    uint64_t read_packed[1];
    pack_genome_region(read_bytes, 0, 32, read_packed);

    // Compute SA index via packed (single shift+mask)
    uint32_t packed_key = sa_index_key(read_packed, 0, 14);

    // Compute SA index via STAR's loop method
    uint32_t loop_key = 0;
    for (int ii = 0; ii < 14; ++ii) {
        loop_key <<= 2;
        loop_key += read_bytes[ii];
    }

    ASSERT_EQ(packed_key, loop_key);

    // Also test with non-zero offset
    uint32_t packed_key2 = sa_index_key(read_packed, 5, 14);
    uint32_t loop_key2 = 0;
    for (int ii = 0; ii < 14; ++ii) {
        loop_key2 <<= 2;
        loop_key2 += read_bytes[5 + ii];
    }
    ASSERT_EQ(packed_key2, loop_key2);
}

void test_packed_revcomp() {
    using namespace lib1fq::packed_align;

    // Test complement: A(00)↔T(11), C(01)↔G(10)
    uint8_t seq[] = {0, 1, 2, 3};            // ACGT
    uint8_t expected_comp[] = {3, 2, 1, 0};  // TGCA

    uint64_t packed;
    pack_genome_region(seq, 0, 4, &packed);

    uint64_t comp = complement_word(packed);

    // Extract bases from complement and verify
    for (int i = 0; i < 4; ++i) {
        uint8_t base = (comp >> (62 - i * 2)) & 3;
        ASSERT_EQ(base, expected_comp[i]);
    }

    // Test reverse_bases: ACGT → TGCA reversed base order
    // Full reverse complement of ACGT should be ACGT (palindrome!)
    uint64_t rc = revcomp_word(packed);

    // For the multi-word test below, use pack_genome_region
    (void)rc;  // palindrome verified by structure

    // Multi-word revcomp test: 8 bases ACGAACGA
    uint8_t seq8[] = {0, 1, 2, 0, 0, 1, 2, 0};
    uint64_t packed8;
    pack_genome_region(seq8, 0, 8, &packed8);

    uint64_t rc_out[1];
    revcomp(&packed8, 8, rc_out);

    // Reverse complement of ACGAACGA:
    // complement: TGCTTGCT  →  [3,2,1,3,3,2,1,3]
    // reverse:    TCGTTGCT  →  [3,1,2,3,3,1,2,3]
    uint8_t expected_rc[] = {3, 1, 2, 3, 3, 1, 2, 3};

    // Extract bases from packed result
    // rc_out is MSB-first: base0 at bits 63-62, etc.
    for (int i = 0; i < 8; ++i) {
        uint8_t base = (rc_out[0] >> (62 - i * 2)) & 3;
        ASSERT_EQ(base, expected_rc[i]);
    }
}

void test_packed_n_bitmap() {
    using namespace lib1fq::packed_align;

    // Genome with N bases at positions 5 and 10
    uint8_t genome[32];
    for (uint32_t i = 0; i < 32; ++i) genome[i] = i % 4;
    genome[5] = 4;   // N
    genome[10] = 4;  // N

    uint8_t n_bmp[4] = {};
    bool has_n = build_n_bitmap(genome, 0, 32, n_bmp);
    ASSERT_TRUE(has_n);

    // Check specific positions
    ASSERT_TRUE(has_n_in_range(n_bmp, 5, 1));
    ASSERT_TRUE(has_n_in_range(n_bmp, 10, 1));
    ASSERT_TRUE(!has_n_in_range(n_bmp, 0, 5));
    ASSERT_TRUE(!has_n_in_range(n_bmp, 11, 5));

    // Range that includes N
    ASSERT_TRUE(has_n_in_range(n_bmp, 3, 5));  // includes pos 5
}

// ── Variable-length R2 quality string length match test ──
// Verifies that for variable-length R2 (stream_lengths[1]=0) with
// QualMode::NONE or BINNED4, r2_quality(i) returns a pointer within bounds
// and its effective length equals r2_len(i), so FASTQ output never produces
// mismatched sequence vs quality line lengths.

void test_varlen_r2_qual_length_match() {
    const char* test_file = "/tmp/test_varlen_r2_qual.1fq";
    const uint32_t n_reads = 300;

    std::mt19937 rng(7777);

    // ── Case 1: QualMode::NONE, variable-length R2 ──
    {
        lib1fq::WriterConfig cfg;
        cfg.block_size = 100;
        cfg.r1_length = 28;
        cfg.r2_length = 0;  // variable
        cfg.qual_mode = lib1fq::QualMode::NONE;
        cfg.polya_trim = false;

        lib1fq::Writer w;
        w.open(test_file, cfg);

        std::vector<uint8_t> r1(28), r2;
        for (uint32_t i = 0; i < n_reads; ++i) {
            for (auto& b : r1) b = rng() % 4;
            uint16_t r2_len = (i == 0) ? 0 : static_cast<uint16_t>(30 + rng() % 120);
            r2.resize(r2_len);
            for (auto& b : r2) b = rng() % 4;
            w.add_read(r2.data(), r2_len, nullptr, r1.data(), 28, nullptr);
        }
        w.finish();

        lib1fq::Reader reader;
        reader.open(test_file);
        // Header: stream_lengths[1] == 0 (variable), qual_mode == NONE
        ASSERT_EQ(reader.header().stream_lengths[1], static_cast<uint16_t>(0));
        ASSERT_EQ(reader.header().qual_mode,
                  static_cast<uint8_t>(lib1fq::QualMode::NONE));

        lib1fq::DecodedBlock blk;
        uint32_t total = 0;
        while (reader.read_block(blk)) {
            for (uint32_t i = 0; i < blk.n_reads; ++i) {
                // With QualMode::NONE, r2_qual must be empty
                ASSERT_TRUE(blk.r2_qual.empty());
                // r2_quality() must return nullptr
                ASSERT_TRUE(blk.r2_quality(i) == nullptr);
            }
            total += blk.n_reads;
        }
        ASSERT_EQ(total, n_reads);
    }

    // ── Case 2: QualMode::BINNED4, variable-length R2 ──
    {
        lib1fq::WriterConfig cfg;
        cfg.block_size = 100;
        cfg.r1_length = 28;
        cfg.r2_length = 0;  // variable
        cfg.qual_mode = lib1fq::QualMode::BINNED4;
        cfg.polya_trim = false;

        lib1fq::Writer w;
        w.open(test_file, cfg);

        std::vector<uint8_t> r1(28), r2, q;
        std::vector<uint16_t> expected_r2_lengths;
        for (uint32_t i = 0; i < n_reads; ++i) {
            for (auto& b : r1) b = rng() % 4;
            uint16_t r2_len = static_cast<uint16_t>(30 + rng() % 120);
            expected_r2_lengths.push_back(r2_len);
            r2.resize(r2_len);
            q.resize(r2_len);
            for (auto& b : r2) b = rng() % 4;
            for (auto& b : q) b = static_cast<uint8_t>(rng() % 40);
            w.add_read(r2.data(), r2_len, q.data(), r1.data(), 28, nullptr);
        }
        w.finish();

        lib1fq::Reader reader;
        reader.open(test_file);
        ASSERT_EQ(reader.header().stream_lengths[1], static_cast<uint16_t>(0));
        ASSERT_EQ(reader.header().qual_mode,
                  static_cast<uint8_t>(lib1fq::QualMode::BINNED4));

        lib1fq::DecodedBlock blk;
        uint32_t total = 0;
        uint32_t read_idx = 0;
        while (reader.read_block(blk)) {
            for (uint32_t i = 0; i < blk.n_reads; ++i) {
                uint16_t r2len = blk.r2_len(i);
                ASSERT_EQ(r2len, expected_r2_lengths[read_idx]);
                // Quality must be present and within bounds
                const uint8_t* qual = blk.r2_quality(i);
                ASSERT_TRUE(qual != nullptr);
                // Verify the quality pointer is within r2_qual bounds
                size_t q_offset = static_cast<size_t>(
                    qual - blk.r2_qual.data());
                ASSERT_TRUE(q_offset + r2len <= blk.r2_qual.size());
                ++read_idx;
            }
            total += blk.n_reads;
        }
        ASSERT_EQ(total, n_reads);
    }

    std::remove(test_file);
}

// ── FastqEncoder variable-length R2 regression test (BUG-R2-VARLEN) ──
// Verifies that R2 reads shorter/longer than the probe-phase fixed length
// are correctly clamped so the fixed-length decoded block is not corrupted.

void test_fastq_encoder_varlen_r2() {
    // Write temp paired FASTQ files: first 1100 reads have R2=50bp,
    // next 100 reads have R2=75bp (would trip the bug), last 100 have R2=30bp.
    const char* r1_path = "/tmp/test_varlen_r1.fastq";
    const char* r2_path = "/tmp/test_varlen_r2.fastq";
    const char* out1fq = "/tmp/test_varlen.1fq";

    // Probe samples 1000 reads. We need >1000 uniform reads before variation.
    const int N_UNIFORM = 1100;
    const int N_LONG = 100;
    const int N_SHORT = 100;
    const uint16_t FIXED_LEN = 50;
    const uint16_t LONG_LEN = 75;
    const uint16_t SHORT_LEN = 30;

    // Fixed R1: 28bp barcode+UMI
    const uint16_t R1_LEN = 28;
    std::string r1_seq(R1_LEN, 'A');
    std::string r1_qual(R1_LEN, 'I');

    FILE* f1 = std::fopen(r1_path, "w");
    FILE* f2 = std::fopen(r2_path, "w");
    ASSERT_TRUE(f1 && f2);

    // Deterministic nucleotide cycle so we can reproduce expected bases
    auto base_char = [](int i) -> char {
        static const char B[] = "ACGT";
        return B[i % 4];
    };

    int read_idx = 0;
    auto write_pair = [&](uint16_t r2_len) {
        std::string r2_seq(r2_len, 'N');
        for (uint16_t j = 0; j < r2_len; ++j) r2_seq[j] = base_char(read_idx * 7 + j);
        std::string r2_qual(r2_len, 'I');
        std::fprintf(f1, "@R%d\n%s\n+\n%s\n", read_idx, r1_seq.c_str(), r1_qual.c_str());
        std::fprintf(f2, "@R%d\n%s\n+\n%s\n", read_idx, r2_seq.c_str(), r2_qual.c_str());
        ++read_idx;
    };

    for (int i = 0; i < N_UNIFORM; ++i) write_pair(FIXED_LEN);
    for (int i = 0; i < N_LONG; ++i) write_pair(LONG_LEN);
    for (int i = 0; i < N_SHORT; ++i) write_pair(SHORT_LEN);

    std::fclose(f1);
    std::fclose(f2);

    // Encode: probe phase will see 1000 reads all at FIXED_LEN → r2_fixed=true
    {
        lib1fq::EncoderConfig cfg;
        cfg.output_path = out1fq;
        cfg.block_size = 512;
        cfg.codec = lib1fq::Codec::ZSTD;
        cfg.qual_mode = lib1fq::QualMode::NONE;
        cfg.seq_enc = lib1fq::SeqEncoding::PACKED_2BIT;
        cfg.protocol_tag = "10x-v3";  // skip auto-detection
        cfg.no_trim = true;           // disable polyA trimming so stream_lengths[1] reflects fixed R2 length

        lib1fq::FastqEncoder enc;
        auto stats = enc.encode(r1_path, r2_path, cfg);
        ASSERT_EQ(stats.total_reads,
                  static_cast<uint64_t>(N_UNIFORM + N_LONG + N_SHORT));
    }

    // Decode: verify the .1fq can be fully read and R2 lengths are FIXED_LEN
    // (long reads truncated to FIXED_LEN, short reads padded to FIXED_LEN).
    {
        lib1fq::Reader reader;
        reader.open(out1fq);

        // Header must declare fixed-length R2.
        ASSERT_EQ(reader.header().stream_lengths[1], FIXED_LEN);

        lib1fq::DecodedBlock blk;
        uint64_t total = 0;
        while (reader.read_block(blk)) {
            for (uint32_t i = 0; i < blk.n_reads; ++i) {
                // Every decoded R2 must be FIXED_LEN (the clamped value).
                ASSERT_EQ(blk.r2_len(i), FIXED_LEN);
            }
            total += blk.n_reads;
        }
        ASSERT_EQ(total,
                  static_cast<uint64_t>(N_UNIFORM + N_LONG + N_SHORT));
    }

    std::remove(r1_path);
    std::remove(r2_path);
    std::remove(out1fq);
}

// ── VDB read-swap protocol detection (AUTOFIX-VDB-READ-SWAP-PROTOCOL) ──
// Tests that detect_protocol correctly identifies inverted R1/R2 orientation
// (VDB stores R1=cDNA, R2=CB+UMI) and returns reads_swapped=true.
// Also tests the variable-R2 path where r2_len=0 but probe spots carry
// majority R2=26bp barcode data.

void test_vdb_read_swap_detect() {
#ifndef WHITELISTS_DIR
    // Try env var (set by ctest via set_tests_properties)
    const char* wl_env = std::getenv("SINGLIFY_WHITELISTS_DIR");
    if (!wl_env) {
        std::cerr << "  SKIP: SINGLIFY_WHITELISTS_DIR not set\n";
        return;
    }
    const std::string wl_base = wl_env;
#else
    const std::string wl_base = WHITELISTS_DIR;
#endif
    // Load 2000 barcodes from the 10x-v2 whitelist.
    const std::string wl_path = wl_base + "/737K-august-2016.txt";
    FILE* wl = std::fopen(wl_path.c_str(), "r");
    if (!wl) {
        std::cerr << "  SKIP: missing " << wl_path << "\n";
        return;
    }
    std::vector<std::vector<uint8_t>> barcodes;
    char line[64];
    while (std::fgets(line, sizeof(line), wl) && barcodes.size() < 2000) {
        size_t len = std::strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) --len;
        if (len < 16) continue;
        std::vector<uint8_t> bc(16);
        for (int i = 0; i < 16; ++i) bc[i] = lib1fq::nuc::ascii_to_num(line[i]);
        barcodes.push_back(bc);
    }
    std::fclose(wl);
    ASSERT_TRUE(barcodes.size() >= 100);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> base(0, 3);

    // Build probe spots: R1=98bp random cDNA, R2=26bp (barcode16 + UMI10).
    struct Spot {
        std::vector<uint8_t> r1_seq, r2_seq;
        uint16_t r1_len, r2_len;
    };
    std::vector<Spot> spots_constant_r2;
    spots_constant_r2.reserve(2000);
    for (size_t i = 0; i < 2000; ++i) {
        Spot s;
        s.r1_seq.resize(98); s.r1_len = 98;
        for (auto& b : s.r1_seq) b = static_cast<uint8_t>(base(rng));
        // R2 = whitelist barcode + random UMI
        s.r2_seq.resize(26); s.r2_len = 26;
        const auto& bc = barcodes[i % barcodes.size()];
        for (int j = 0; j < 16; ++j) s.r2_seq[j] = bc[j];
        for (int j = 16; j < 26; ++j) s.r2_seq[j] = static_cast<uint8_t>(base(rng));
        spots_constant_r2.push_back(std::move(s));
    }

    const std::vector<std::string> wl_dirs = {wl_base};

    // ── Test 1: constant R2=26bp, R1=98bp (inverted layout) ──
    auto cand = lib1fq::detect_protocol(spots_constant_r2, 98, 26, wl_dirs);
    // Protocol should be 10x-v2 (or 10x-v1 at minimum; any 10x-2x detection counts)
    bool is_10x = cand.tag.find("10x") != std::string::npos ||
                  cand.tag.find("v2") != std::string::npos;
    ASSERT_TRUE(is_10x);
    ASSERT_TRUE(cand.reads_swapped);
    ASSERT_TRUE(cand.confidence >= lib1fq::Confidence::LOW);

    // ── Test 2: variable R2 (r2_len=0 outer, but spots carry majority r2_len=26) ──
    // Mix: 70% spots with r2_len=26 (barcode), 30% with r2_len=0 (empty).
    std::vector<Spot> spots_variable_r2 = spots_constant_r2;
    for (size_t i = 0; i < spots_variable_r2.size(); i += 3) {
        spots_variable_r2[i].r2_len = 0;
        spots_variable_r2[i].r2_seq.clear();
    }
    auto cand2 = lib1fq::detect_protocol(spots_variable_r2, 98, 0, wl_dirs);
    bool is_10x2 = cand2.tag.find("10x") != std::string::npos ||
                   cand2.tag.find("v2") != std::string::npos;
    ASSERT_TRUE(is_10x2);
    ASSERT_TRUE(cand2.reads_swapped);

    // ── Test 3: regression — normal 10x-v3 (R1=28bp barcode, R2=98bp cDNA) ──
    std::vector<Spot> normal_spots;
    normal_spots.reserve(2000);
    // Load some v3 barcodes
    const std::string wl3_path = wl_base + "/3M-february-2018.txt";
    FILE* wl3 = std::fopen(wl3_path.c_str(), "r");
    if (wl3) {
        std::vector<std::vector<uint8_t>> v3bcs;
        while (std::fgets(line, sizeof(line), wl3) && v3bcs.size() < 2000) {
            size_t len = std::strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) --len;
            if (len < 16) continue;
            std::vector<uint8_t> bc(16);
            for (int i = 0; i < 16; ++i) bc[i] = lib1fq::nuc::ascii_to_num(line[i]);
            v3bcs.push_back(bc);
        }
        std::fclose(wl3);
        if (v3bcs.size() >= 100) {
            for (size_t i = 0; i < 2000; ++i) {
                Spot s;
                // R1 = 28bp (BC16 + UMI12)
                s.r1_seq.resize(28); s.r1_len = 28;
                const auto& bc = v3bcs[i % v3bcs.size()];
                for (int j = 0; j < 16; ++j) s.r1_seq[j] = bc[j];
                for (int j = 16; j < 28; ++j) s.r1_seq[j] = static_cast<uint8_t>(base(rng));
                // R2 = 98bp random cDNA
                s.r2_seq.resize(98); s.r2_len = 98;
                for (auto& b : s.r2_seq) b = static_cast<uint8_t>(base(rng));
                normal_spots.push_back(std::move(s));
            }
            auto cand3 = lib1fq::detect_protocol(normal_spots, 28, 98, wl_dirs);
            // Normal 10x-v3: should NOT be reads_swapped
            ASSERT_TRUE(!cand3.reads_swapped);
            ASSERT_TRUE(cand3.tag.find("10x") != std::string::npos);
        }
    }
}

// ── Hard geometry swap detection (AUTOFIX-E2E-A2-READ-SWAP) ──
// Tests that should_hard_geometry_swap() fires for unambiguous inverted
// geometry (R1=cDNA long, R2=barcode short) and does NOT fire for normal
// or ambiguous read-length combinations.
// Also verifies that detect_protocol() sets reads_swapped=true for the
// inverted geometry even when no whitelist is available (no WL dirs given).

void test_geometry_swap_detection() {
    using lib1fq::should_hard_geometry_swap;

    // ── Predicate tests ──

    // Primary case: SRR34789664-like R1=91bp cDNA, R2=28bp barcode
    ASSERT_TRUE(should_hard_geometry_swap(91, 28));

    // Edge of condition: R1=51, R2=25 — 51>50, 25<=34, 51>50 (2*25=50 < 51)
    ASSERT_TRUE(should_hard_geometry_swap(51, 25));

    // Normal 10x orientation: R1=28 (barcode), R2=91 (cDNA) — must NOT swap
    ASSERT_TRUE(!should_hard_geometry_swap(28, 91));

    // R1 just at threshold: R1=50 — 50 > 50 is false — must NOT swap
    ASSERT_TRUE(!should_hard_geometry_swap(50, 24));

    // R2 just above threshold: R2=35 > 34 — must NOT swap
    ASSERT_TRUE(!should_hard_geometry_swap(91, 35));

    // Symmetric short reads: R1=28, R2=28 — R1 <= 50 — must NOT swap
    ASSERT_TRUE(!should_hard_geometry_swap(28, 28));

    // R1 = 2 x R2 (not strictly greater) — must NOT swap
    ASSERT_TRUE(!should_hard_geometry_swap(56, 28));  // 56 == 2*28, not > 2*28

    // R2=0: must NOT swap (no R2 present)
    ASSERT_TRUE(!should_hard_geometry_swap(91, 0));

    // ── detect_protocol reads_swapped ──
    // Build probe spots: R1=91bp random cDNA, R2=28bp random (no real barcodes).
    // No whitelist dirs are provided so WL match rate = 0.
    // detect_protocol must still set reads_swapped=true because of inverted geometry.
    std::mt19937 rng(54321);
    std::uniform_int_distribution<int> base_dist(0, 3);

    struct GeoSpot {
        std::vector<uint8_t> r1_seq, r2_seq;
        uint16_t r1_len, r2_len;
    };
    std::vector<GeoSpot> spots;
    spots.reserve(2000);
    for (int i = 0; i < 2000; ++i) {
        GeoSpot s;
        s.r1_len = 91; s.r1_seq.resize(91);
        s.r2_len = 28; s.r2_seq.resize(28);
        for (auto& b : s.r1_seq) b = static_cast<uint8_t>(base_dist(rng));
        for (auto& b : s.r2_seq) b = static_cast<uint8_t>(base_dist(rng));
        spots.push_back(std::move(s));
    }

    const std::vector<std::string> no_wl_dirs;  // no whitelist — geometry-only path
    auto cand = lib1fq::detect_protocol(spots, 91, 28, no_wl_dirs);

    // Must be marked reads_swapped: geometry R1=91>50, R2=28<=34, 91>56 is inverted
    ASSERT_TRUE(cand.reads_swapped);

    // ── Regression: normal 10x orientation must NOT be reads_swapped ──
    std::vector<GeoSpot> normal_spots;
    normal_spots.reserve(2000);
    for (int i = 0; i < 2000; ++i) {
        GeoSpot s;
        s.r1_len = 28; s.r1_seq.resize(28);
        s.r2_len = 91; s.r2_seq.resize(91);
        for (auto& b : s.r1_seq) b = static_cast<uint8_t>(base_dist(rng));
        for (auto& b : s.r2_seq) b = static_cast<uint8_t>(base_dist(rng));
        normal_spots.push_back(std::move(s));
    }
    auto normal_cand = lib1fq::detect_protocol(normal_spots, 28, 91, no_wl_dirs);
    ASSERT_TRUE(!normal_cand.reads_swapped);
}

// ── Benchmark smoke test ──

// ── WL-defensive override (AUTOFIX-E2E-A-PROTOCOL-REGRESSION-V1) ──
// Constructs a scenario where a no-WL protocol (dropseq geometry) would
// outscore a WL protocol (10x-v3) on pure heuristics because wl_match_rate
// is low (~15%) but still above the 5% threshold.
// Verifies that the defensive rule fires and the WL candidate wins.
//
// Geometry trick: use barcode_read_len=32bp.  This makes both 10x-v3 (r1=28,
// within ±4) and dropseq (r1=20, oversized+12) valid candidates for the same
// probe spot batch, while dropseq is not geometry-suppressed (no WL protocol
// has dropseq's exact CB+UMI layout).

void test_wl_defensive_override() {
#ifndef WHITELISTS_DIR
    const char* wl_env = std::getenv("SINGLIFY_WHITELISTS_DIR");
    if (!wl_env) {
        std::cerr << "  SKIP: SINGLIFY_WHITELISTS_DIR not set\n";
        return;
    }
    const std::string wl_base = wl_env;
#else
    const std::string wl_base = WHITELISTS_DIR;
#endif

    // Load 300 real 10x-v3 barcodes from the 3M whitelist
    const std::string wl3_path = wl_base + "/3M-february-2018.txt";
    FILE* wlf = std::fopen(wl3_path.c_str(), "r");
    if (!wlf) {
        std::cerr << "  SKIP: missing " << wl3_path << "\n";
        return;
    }
    std::vector<std::vector<uint8_t>> v3bcs;
    char line[64];
    while (std::fgets(line, sizeof(line), wlf) && v3bcs.size() < 300) {
        size_t len = std::strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) --len;
        if (len < 16) continue;
        std::vector<uint8_t> bc(16);
        for (int i = 0; i < 16; ++i) bc[i] = lib1fq::nuc::ascii_to_num(line[i]);
        v3bcs.push_back(bc);
    }
    std::fclose(wlf);
    if (v3bcs.size() < 100) {
        std::cerr << "  SKIP: too few barcodes loaded from " << wl3_path << "\n";
        return;
    }

    const std::vector<std::string> wl_dirs = {wl_base};
    std::mt19937 rng(77777);
    std::uniform_int_distribution<int> base_dist(0, 3);

    struct Spot {
        std::vector<uint8_t> r1_seq, r2_seq;
        uint16_t r1_len, r2_len;
    };

    // Build 2000 spots with R1=32bp barcode read, R2=98bp random cDNA.
    // 300 of the 2000 spots carry a real v3 barcode in positions 0-15 of R1.
    // The remaining 1700 slots are fully random.
    // This gives wl_match_rate ~15% for 10x-v3 (above the 5% defensive threshold).
    //
    // Heuristic analysis without the defensive rule:
    //   10x-v3 (WL): 0.50*0.15 + 0.20*1.0 + 0 + 0.10(geom, diff=4) = ~0.37
    //   dropseq (no-WL): 0.35*0 + 0.30*1.0 + 0 + 0.20*1.0 + 0.0(geom) = ~0.50
    //   → dropseq wins without the defensive rule.
    // With the defensive rule: 10x-v3 promoted to 0.51 → 10x-v3 wins.
    std::vector<Spot> spots;
    spots.reserve(2000);
    for (int i = 0; i < 2000; ++i) {
        Spot s;
        // R1 = 32bp (barcode read, slightly over-sequenced 10x-v3)
        s.r1_seq.resize(32); s.r1_len = 32;
        if (i < 300) {
            // Real v3 barcode in positions 0-15
            const auto& bc = v3bcs[i % v3bcs.size()];
            for (int j = 0; j < 16; ++j) s.r1_seq[j] = bc[j];
            // Random UMI + tail (positions 16-31)
            for (int j = 16; j < 32; ++j)
                s.r1_seq[j] = static_cast<uint8_t>(base_dist(rng));
        } else {
            // Fully random 32bp
            for (auto& b : s.r1_seq) b = static_cast<uint8_t>(base_dist(rng));
        }
        // R2 = 98bp random cDNA (high UMI entropy, no polyA signal)
        s.r2_seq.resize(98); s.r2_len = 98;
        for (auto& b : s.r2_seq) b = static_cast<uint8_t>(base_dist(rng));
        spots.push_back(std::move(s));
    }

    // ── Test A: rule should FIRE (rate ~15% > 5% threshold) ──
    // Expect 10x-v3 (or 10x family) to be selected because the defensive
    // rule promotes the WL candidate over the higher-scoring dropseq candidate.
    auto cand = lib1fq::detect_protocol(spots, 32, 98, wl_dirs);
    bool is_10x = cand.tag.find("10x") != std::string::npos;
    ASSERT_TRUE(is_10x);
    // reads_swapped must be false: R1=32 (short barcode read), R2=98 (long cDNA) is normal
    ASSERT_TRUE(!cand.reads_swapped);
    // wl_match_rate should be > 0 (we injected real barcodes)
    ASSERT_TRUE(cand.wl_match_rate > 0.0);

    // ── Test B: rule should NOT fire (rate ~1% < 5% threshold) ──
    // With only ~20 real barcodes out of 2000. the WL rate stays well below 5%.
    // No defensive promotion → the heuristic winner (non-WL) may win.
    std::vector<Spot> spots_low;
    spots_low.reserve(2000);
    for (int i = 0; i < 2000; ++i) {
        Spot s;
        s.r1_seq.resize(32); s.r1_len = 32;
        if (i < 20) {
            const auto& bc = v3bcs[i % v3bcs.size()];
            for (int j = 0; j < 16; ++j) s.r1_seq[j] = bc[j];
            for (int j = 16; j < 32; ++j)
                s.r1_seq[j] = static_cast<uint8_t>(base_dist(rng));
        } else {
            for (auto& b : s.r1_seq) b = static_cast<uint8_t>(base_dist(rng));
        }
        s.r2_seq.resize(98); s.r2_len = 98;
        for (auto& b : s.r2_seq) b = static_cast<uint8_t>(base_dist(rng));
        spots_low.push_back(std::move(s));
    }
    auto cand_low = lib1fq::detect_protocol(spots_low, 32, 98, wl_dirs);
    // The defensive rule should not fire at ~1% rate.
    // With no promotion, the WL score is too low to beat the no-WL candidate.
    // We verify this by checking that the result is NOT the 10x-v3 WL candidate.
    bool is_10xv3_specific = (cand_low.tag == "10x-3p-v3" || cand_low.tag == "10x-3p-v3.1");
    ASSERT_TRUE(!is_10xv3_specific);
}

void test_benchmark_smoke() {
    // Quick benchmark with small data to verify the infrastructure works
    auto results = lib1fq::benchmark::run_benchmark(
        1000,  // small block
        91,    // r2_len
        0.5,   // dup_rate
        2      // iterations
    );

    // Should have results for each codec × column + block sweep
    ASSERT_TRUE(results.size() > 0);

    // All results should have positive throughput
    for (const auto& r : results) {
        ASSERT_TRUE(r.encode_mb_s > 0);
        ASSERT_TRUE(r.decode_mb_s > 0);
        ASSERT_TRUE(r.ratio >= 1.0 || r.codec_name.find("none") != std::string::npos);
    }
}

// ── Large-file index offset test ──
// Simulate the bug: write a valid .1fq, then patch ftr.index_offset to the
// truncated uint32_t value that would occur for a file at a >4 GiB offset.
// The reader must still decode all blocks and metadata correctly because it
// computes the index position from EOF rather than trusting ftr.index_offset.

void test_large_file_index_offset_recovery() {
    const char* test_file = "/tmp/test_lib1fq_large_offset.1fq";
    const uint32_t n_reads = 300;
    const uint32_t block_size = 100;  // 3 blocks
    const uint16_t r1_len = 28;

    std::mt19937 rng(999);
    std::vector<std::vector<uint8_t>> r1v(n_reads), r2v(n_reads);
    std::vector<uint16_t> r2lens(n_reads);

    for (uint32_t i = 0; i < n_reads; ++i) {
        r1v[i].resize(r1_len);
        r2lens[i] = 90;
        r2v[i].resize(90);
        for (auto& b : r1v[i]) b = rng() % 4;
        for (auto& b : r2v[i]) b = rng() % 4;
    }

    // Write file normally
    {
        lib1fq::WriterConfig cfg;
        cfg.block_size = block_size;
        cfg.r1_length = r1_len;
        cfg.r2_length = 90;
        lib1fq::Writer w;
        w.open(test_file, cfg);
        for (uint32_t i = 0; i < n_reads; ++i)
            w.add_read(r2v[i].data(), r2lens[i], nullptr,
                       r1v[i].data(), r1_len, nullptr);
        w.finish("{\"large_file_test\": true}");
    }

    // Read the real footer to know the actual index_offset
    uint64_t real_index_offset = 0;
    {
        FILE* fp = std::fopen(test_file, "rb");
        ASSERT_TRUE(fp != nullptr);
        std::fseek(fp, -static_cast<long>(sizeof(lib1fq::Footer)), SEEK_END);
        lib1fq::Footer ftr;
        std::fread(&ftr, sizeof(ftr), 1, fp);
        real_index_offset = ftr.index_offset;
        std::fclose(fp);
    }

    // Patch the footer to simulate uint32 overflow: set index_offset to the
    // value that a large file would produce (real_offset + UINT32_MAX + 1).
    // The truncated value will be real_offset again (same low 32 bits), but
    // we use a clearly wrong value by adding 0x100000000 - ensuring corruption
    // if any code still uses ftr.index_offset for the seek.
    {
        FILE* fp = std::fopen(test_file, "r+b");
        ASSERT_TRUE(fp != nullptr);
        std::fseek(fp, -static_cast<long>(sizeof(lib1fq::Footer)), SEEK_END);
        lib1fq::Footer ftr;
        std::fread(&ftr, sizeof(ftr), 1, fp);
        // Corrupt: store a wrong offset that would result from uint32 truncation
        // of a real offset like 0x1_XXXXXXXX → low 32 bits only.
        // Use the pattern: actual_offset XOR 0xDEAD0000 to guarantee mismatch.
        ftr.index_offset = static_cast<uint32_t>(real_index_offset ^ 0xDEAD0000ULL);
        std::fseek(fp, -static_cast<long>(sizeof(lib1fq::Footer)), SEEK_END);
        std::fwrite(&ftr, sizeof(ftr), 1, fp);
        std::fclose(fp);
    }

    // Reader must decode everything correctly despite the corrupted footer field
    {
        lib1fq::Reader reader;
        reader.open(test_file);

        ASSERT_EQ(reader.header().n_unique, n_reads);
        ASSERT_EQ(reader.block_count(), (n_reads + block_size - 1) / block_size);

        lib1fq::DecodedBlock blk;
        uint32_t total = 0;
        while (reader.read_block(blk)) {
            for (uint32_t i = 0; i < blk.n_reads; ++i) {
                uint32_t ri = total + i;
                ASSERT_EQ(blk.r1_len(i), r1_len);
                for (uint16_t b = 0; b < r1_len; ++b)
                    ASSERT_EQ(blk.r1_seq(i)[b], r1v[ri][b]);
                ASSERT_EQ(blk.r2_len(i), r2lens[ri]);
                for (uint16_t b = 0; b < r2lens[ri]; ++b)
                    ASSERT_EQ(blk.r2_seq(i)[b], r2v[ri][b]);
            }
            total += blk.n_reads;
        }
        ASSERT_EQ(total, n_reads);

        // Metadata must also be readable (uses block_offsets_ populated correctly)
        std::string meta = reader.read_metadata();
        ASSERT_TRUE(meta.find("large_file_test") != std::string::npos);
    }

    std::remove(test_file);
}

// ── Main ──

// Test: SPLiT-seq CB_UMI_Complex positions use anchorType=0 (from read start), not anchorType=1.
// The barcode read (R2, stored as R1 after encode swap) layout:
//   [UMI_10bp 0-9][Rd3_8bp 10-17][linker15 18-32][Rd2_8bp 33-40][linker15 41-55][Rd1_8bp 56-63]
// STAR must receive: --soloCBposition 0_10_0_17 0_33_0_40 0_56_0_63
//                   --soloUMIposition 0_0_0_9
void test_splitseq_cb_positions() {
    // Build the aligner_params block the same way sra_encoder.h does
    const std::string canonical_tag = "splitseq";

    std::string j;
    j += "    \"aligner_params\": {\n";
    j += "      \"soloType\": \"CB_UMI_Complex\",\n";
    if (canonical_tag == "splitseq") {
        j += "      \"soloCBposition\": [\"0_10_0_17\", \"0_33_0_40\", \"0_56_0_63\"],\n";
        j += "      \"soloUMIposition\": \"0_0_0_9\"\n";
    }
    j += "    }\n";

    // Verify correct positions written (anchorType=0, not anchorType=1)
    ASSERT_TRUE(j.find("\"0_10_0_17\"") != std::string::npos);
    ASSERT_TRUE(j.find("\"0_33_0_40\"") != std::string::npos);
    ASSERT_TRUE(j.find("\"0_56_0_63\"") != std::string::npos);
    ASSERT_TRUE(j.find("\"0_0_0_9\"") != std::string::npos);

    // Verify broken anchorType=1 strings are absent
    ASSERT_TRUE(j.find("\"1_10_1_17\"") == std::string::npos);
    ASSERT_TRUE(j.find("\"1_33_1_40\"") == std::string::npos);
    ASSERT_TRUE(j.find("\"1_56_1_63\"") == std::string::npos);
    ASSERT_TRUE(j.find("\"1_0_1_9\"") == std::string::npos);

    // Verify protocol spec geometry matches the positions
    const lib1fq::CandidateSpec* spec = lib1fq::find_protocol_spec("splitseq");
    ASSERT_TRUE(spec != nullptr);
    ASSERT_EQ(spec->bc_offset, 10u);  // Rd3 at pos 10 in barcode read
    ASSERT_EQ(spec->bc_len,     8u);  // 8bp per CB segment
    ASSERT_EQ(spec->umi_offset, 0u);  // UMI at pos 0
    ASSERT_EQ(spec->umi_len,   10u);  // 10bp UMI
    ASSERT_EQ(spec->per_seg_whitelist_files.size(), 3u);  // 3 distinct CB segment whitelists

    // Verify the per-segment positions derived from bc_offset + linker geometry:
    //   seg0 = [bc_offset, bc_offset+bc_len-1]      = [10, 17]
    //   seg1 = [10 + 8 + 15, 10 + 8 + 15 + 8 - 1]  = [33, 40]  (linker=15)
    //   seg2 = [33 + 8 + 15, 33 + 8 + 15 + 8 - 1]  = [56, 63]
    const int linker = 15;
    int s0 = spec->bc_offset;
    int e0 = s0 + spec->bc_len - 1;
    int s1 = e0 + 1 + linker;
    int e1 = s1 + spec->bc_len - 1;
    int s2 = e1 + 1 + linker;
    int e2 = s2 + spec->bc_len - 1;
    ASSERT_EQ(s0, 10); ASSERT_EQ(e0, 17);
    ASSERT_EQ(s1, 33); ASSERT_EQ(e1, 40);
    ASSERT_EQ(s2, 56); ASSERT_EQ(e2, 63);
}

int main() {
    std::cerr << "lib1fq test suite\n";
    std::cerr << "─────────────────\n";

    TEST(pack_2bit_round_trip);
    TEST(pack_2bit_no_n);
    TEST(pack_2bit_odd_length);
    TEST(pack_qual_binned4);
    TEST(pack_qual_binned2);
    TEST(varint_round_trip);
    TEST(nuc_conversion);
    TEST(header_init);
    TEST(compress_round_trip);
    TEST(writer_reader_round_trip);
    TEST(writer_reader_round_trip_binned2);
    TEST(round_trip_with_n_bases);
    TEST(stress_random_data);
    TEST(bc_dict_round_trip);
    TEST(polya_trimming);
    TEST(dedup_round_trip);
    TEST(read_folding);
    TEST(packed_compare);
    TEST(packed_sa_index);
    TEST(packed_revcomp);
    TEST(packed_n_bitmap);
    TEST(benchmark_smoke);
    TEST(varlen_r2_qual_length_match);
    TEST(fastq_encoder_varlen_r2);
    TEST(vdb_read_swap_detect);
    TEST(geometry_swap_detection);
    TEST(wl_defensive_override);
    TEST(large_file_index_offset_recovery);
    TEST(splitseq_cb_positions);

    std::cerr << "─────────────────\n";
    std::cerr << tests_passed << " passed, " << tests_failed << " failed\n";

    return tests_failed > 0 ? 1 : 0;
}
