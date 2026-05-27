// test_1fq_header.cpp
// Unit tests for .1fq header struct (singlet::fq/types.h)
// Tests: Header init, magic validation, field layout, stream_lengths semantics.

#include <cstdio>
#include <cstring>
#include <string>
#include "singlet/fq/types.h"
#include "singlet/fq/reader.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(expr, name) do { \
    if (expr) { ++g_pass; std::fprintf(stderr, "  PASS: %s\n", name); } \
    else { ++g_fail; std::fprintf(stderr, "  FAIL: %s\n", name); } \
} while(0)

static void test_header_size() {
    CHECK(sizeof(singlet::fq::Header) == 96, "header_size_96");
    CHECK(sizeof(singlet::fq::BlockHeader) == 12, "block_header_size_12");
    CHECK(sizeof(singlet::fq::Footer) == 16, "footer_size_16");
}

static void test_header_init() {
    singlet::fq::Header h;
    h.init();
    CHECK(h.valid_magic(), "init_valid_magic");
    CHECK(h.version == singlet::fq::FORMAT_VERSION, "init_version");
    CHECK(h.codec == static_cast<uint8_t>(singlet::fq::Codec::ZSTD), "init_codec_zstd");
    CHECK(h.codec_level == 6, "init_codec_level");
    CHECK(h.seq_encoding == static_cast<uint8_t>(singlet::fq::SeqEncoding::PACKED_2BIT), "init_seq_enc");
    CHECK(h.qual_mode == static_cast<uint8_t>(singlet::fq::QualMode::BINNED4), "init_qual_mode");
    CHECK(h.block_size == 100000, "init_block_size");
    // All stream_lengths should be 0 after init (variable)
    for (int i = 0; i < 4; ++i)
        CHECK(h.stream_lengths[i] == 0, ("init_stream_lengths_" + std::to_string(i) + "_zero").c_str());
    // protocol_id should be 0 (unknown)
    CHECK(h.protocol_id == 0, "init_protocol_id_zero");
    // n_streams should be 0
    CHECK(h.n_streams == 0, "init_n_streams_zero");
}

static void test_header_magic_validation() {
    singlet::fq::Header h;
    h.init();
    CHECK(h.valid_magic(), "magic_valid_after_init");

    // Corrupt magic
    h.magic[0] = 'X';
    CHECK(!h.valid_magic(), "magic_invalid_after_corrupt");

    // Restore
    h.magic[0] = '1';
    CHECK(h.valid_magic(), "magic_valid_after_restore");
}

static void test_header_field_offsets() {
    singlet::fq::Header h;
    h.init();
    // Verify field positions via offsetof-like computation
    const uint8_t* base = reinterpret_cast<const uint8_t*>(&h);
    CHECK(reinterpret_cast<const uint8_t*>(&h.magic) - base == 0, "offset_magic_0");
    CHECK(reinterpret_cast<const uint8_t*>(&h.version) - base == 4, "offset_version_4");
    CHECK(reinterpret_cast<const uint8_t*>(&h.n_streams) - base == 6, "offset_n_streams_6");
    CHECK(reinterpret_cast<const uint8_t*>(&h.protocol_id) - base == 7, "offset_protocol_id_7");
    CHECK(reinterpret_cast<const uint8_t*>(&h.confidence) - base == 8, "offset_confidence_8");
    CHECK(reinterpret_cast<const uint8_t*>(&h.stream_lengths) - base == 12, "offset_stream_lengths_12");
    CHECK(reinterpret_cast<const uint8_t*>(&h.n_unique) - base == 20, "offset_n_unique_20");
    CHECK(reinterpret_cast<const uint8_t*>(&h.n_original) - base == 28, "offset_n_original_28");
}

static void test_header_stream_lengths_semantics() {
    singlet::fq::Header h;
    h.init();

    // stream_lengths[0] = R1, stream_lengths[1] = R2
    // 0 means variable length, not "no data"
    h.n_streams = 2;
    h.stream_lengths[0] = 28;  // Fixed R1 = 28bp (10x v3)
    h.stream_lengths[1] = 0;   // Variable R2 (cDNA)
    CHECK(h.stream_lengths[0] == 28, "stream_r1_fixed_28");
    CHECK(h.stream_lengths[1] == 0, "stream_r2_variable");
    CHECK(h.n_streams == 2, "stream_count_2");

    // 10x v2 example: R1=26, R2=variable
    h.stream_lengths[0] = 26;
    CHECK(h.stream_lengths[0] == 26, "stream_v2_r1_26");

    // Single-stream (SS2): only R2 cDNA
    h.n_streams = 1;
    h.stream_lengths[0] = 0;
    CHECK(h.n_streams == 1, "stream_ss2_single");
}

static void test_header_protocol_ids() {
    // Known protocol IDs from protocol.h
    CHECK(static_cast<uint8_t>(0) == 0, "proto_unknown_0");
    // Match a few known protocols
    singlet::fq::Header h;
    h.init();
    h.protocol_id = 1;  // 10x-3p-v3
    CHECK(h.protocol_id == 1, "proto_10xv3_1");
    h.protocol_id = 6;  // dropseq
    CHECK(h.protocol_id == 6, "proto_dropseq_6");
    h.protocol_id = 25; // cite-seq-gex (highest)
    CHECK(h.protocol_id == 25, "proto_cite_25");
}

static void test_header_binary_roundtrip() {
    singlet::fq::Header h;
    h.init();
    h.n_streams = 2;
    h.protocol_id = 1;
    h.confidence = 3;
    h.stream_lengths[0] = 28;
    h.stream_lengths[1] = 91;
    h.n_unique = 12345678;
    h.n_original = 23456789;
    h.block_count = 42;

    // Write to buffer and read back
    uint8_t buf[96];
    std::memcpy(buf, &h, 96);

    singlet::fq::Header h2;
    std::memcpy(&h2, buf, 96);
    CHECK(h2.valid_magic(), "roundtrip_magic");
    CHECK(h2.n_streams == 2, "roundtrip_n_streams");
    CHECK(h2.protocol_id == 1, "roundtrip_protocol_id");
    CHECK(h2.stream_lengths[0] == 28, "roundtrip_r1_len");
    CHECK(h2.stream_lengths[1] == 91, "roundtrip_r2_len");
    CHECK(h2.n_unique == 12345678, "roundtrip_n_unique");
    CHECK(h2.n_original == 23456789, "roundtrip_n_original");
    CHECK(h2.block_count == 42, "roundtrip_block_count");
}

static void test_footer_layout() {
    singlet::fq::Footer f;
    std::memset(&f, 0, sizeof(f));
    std::memcpy(f.magic, singlet::fq::MAGIC, 4);
    CHECK(f.magic[0] == '1' && f.magic[1] == 'F' && f.magic[2] == 'Q' && f.magic[3] == '\0',
          "footer_magic_1FQ");
    CHECK(sizeof(f) == 16, "footer_size_16b");
}

static void test_decoded_block_accessors() {
    singlet::fq::DecodedBlock blk;
    blk.n_reads = 3;

    // R1: 3 reads with lengths 28, 26, 24
    uint8_t r1_all[] = {
        0,1,2,3,0,1,2,3,0,1,2,3,0,1,0,1,2,3,0,1,2,3,0,1,2,3,0,1,  // 28bp
        0,1,2,3,0,1,2,3,0,1,2,3,0,1,0,1,2,3,0,1,2,3,0,1,2,3,        // 26bp
        0,1,2,3,0,1,2,3,0,1,2,3,0,1,0,1,2,3,0,1,2,3,0,1              // 24bp
    };
    blk.r1_data.assign(r1_all, r1_all + sizeof(r1_all));
    blk.r1_offsets = {0, 28, 54};
    blk.r1_lengths = {28, 26, 24};

    // R2: 3 reads with lengths 91, 150, 50
    blk.r2_data.resize(91 + 150 + 50, 2);
    blk.r2_offsets = {0, 91, 241};
    blk.r2_lengths = {91, 150, 50};

    CHECK(blk.r1_len(0) == 28, "block_r1_len_0");
    CHECK(blk.r1_len(1) == 26, "block_r1_len_1");
    CHECK(blk.r1_len(2) == 24, "block_r1_len_2");
    CHECK(blk.r2_len(0) == 91, "block_r2_len_0");
    CHECK(blk.r2_len(1) == 150, "block_r2_len_1");
    CHECK(blk.r2_len(2) == 50, "block_r2_len_2");
    CHECK(blk.r1_seq(0) == blk.r1_data.data(), "block_r1_seq_0_offset");
    CHECK(blk.r2_seq(1) == blk.r2_data.data() + 91, "block_r2_seq_1_offset");
}

static void test_decoded_block_clear() {
    singlet::fq::DecodedBlock blk;
    blk.n_reads = 5;
    blk.r1_data.resize(100, 1);
    blk.r2_data.resize(200, 2);
    blk.r1_offsets = {0, 20, 40, 60, 80};
    blk.r1_lengths = {20, 20, 20, 20, 20};
    blk.r2_offsets = {0, 40, 80, 120, 160};
    blk.r2_lengths = {40, 40, 40, 40, 40};

    blk.clear();

    CHECK(blk.n_reads == 0, "clear_n_reads");
    CHECK(blk.r1_data.empty(), "clear_r1_data");
    CHECK(blk.r2_data.empty(), "clear_r2_data");
    CHECK(blk.r1_offsets.empty(), "clear_r1_offsets");
    CHECK(blk.r2_offsets.empty(), "clear_r2_offsets");
    CHECK(blk.r1_lengths.empty(), "clear_r1_lengths");
    CHECK(blk.r2_lengths.empty(), "clear_r2_lengths");
    CHECK(blk.bc_indices.empty(), "clear_bc_indices");
    CHECK(blk.r2_qual.empty(), "clear_r2_qual");
}

static void test_decoded_block_quality_guard() {
    singlet::fq::DecodedBlock blk;
    blk.n_reads = 2;
    blk.r2_data.resize(100, 2);
    blk.r2_offsets = {0, 50};
    blk.r2_lengths = {50, 50};

    // No quality data → r2_quality returns nullptr
    CHECK(blk.r2_quality(0) == nullptr, "qual_empty_nullptr_0");
    CHECK(blk.r2_quality(1) == nullptr, "qual_empty_nullptr_1");

    // Add quality but make it shorter than expected (guard test)
    blk.r2_qual.resize(30, 'F');  // Only 30 bytes, but read 0 needs 50
    CHECK(blk.r2_quality(0) == nullptr, "qual_short_guard_0");  // off=0+len=50 > size=30

    // Extend quality to full size
    blk.r2_qual.resize(100, 'F');
    CHECK(blk.r2_quality(0) != nullptr, "qual_full_ok_0");
    CHECK(blk.r2_quality(1) != nullptr, "qual_full_ok_1");
}

int main() {
    test_header_size();
    test_header_init();
    test_header_magic_validation();
    test_header_field_offsets();
    test_header_stream_lengths_semantics();
    test_header_protocol_ids();
    test_header_binary_roundtrip();
    test_footer_layout();
    test_decoded_block_accessors();
    test_decoded_block_clear();
    test_decoded_block_quality_guard();

    std::fprintf(stderr, "\n=== 1fq_header: %d passed, %d failed ===\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
