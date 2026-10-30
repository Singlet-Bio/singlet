// .1fg: GPU-ready chunked read container (prototype, W11 ingest study). See docs/research-beta/ingest-format.md.
// All integers little-endian. Section payloads are standard zstd frames (nvCOMP batched-zstd compatible: content size in the
// frame header, no dictionary, no zstd checksum) or raw bytes (codec 0).
#pragma once
#include <cstdint>
#include <cstring>
#include <nmmintrin.h>

namespace fg {

static constexpr uint32_t FILE_MAGIC  = 0x00474631u;   // "1FG\0"
static constexpr uint32_t CHUNK_MAGIC = 0x314b4843u;   // "CHK1"
static constexpr uint32_t FOOT_MAGIC  = 0x31475446u;   // "FTG1"
static constexpr uint32_t CHUNK_ALIGN = 4096;          // chunk start alignment (O_DIRECT / GDS friendly)
static constexpr uint32_t SEC_ALIGN   = 16;            // section start alignment inside a chunk (nvCOMP needs 8)
static constexpr uint32_t CHUNK_PAIRS = 262144;        // max pairs per chunk
enum Codec : uint8_t { CODEC_RAW = 0, CODEC_ZSTD = 1 };
enum Flags : uint8_t { F_CRC = 1, F_INCOMPLETE = 2 };
enum Section { SEC_CBUMI = 0, SEC_R2 = 1, SEC_EXC = 2, NSEC = 3 };
// exception entry (u32): bits 0-19 pair index within chunk, 20-21 kind, 22-29 position. Kind 3 = barcode not in the whitelist (the CB word holds the raw barcode, N coded as A; the .1fq decode prints N for the whole barcode)
// kind 2 is reserved (an earlier draft listed the N positions of a raw unknown barcode: 17 entries per read when the whole barcode is N, 12.8 B/pair in GSM6470208)
enum ExcKind { EXC_R2N = 0, EXC_UMIN = 1, EXC_RESERVED = 2, EXC_BC_UNKNOWN = 3 };
static inline uint32_t exc_pack(uint32_t idx, uint32_t kind, uint32_t pos) { return idx | (kind << 20) | (pos << 22); }

#pragma pack(push, 1)
struct FileHeader {            // 256 bytes at offset 0
    uint32_t magic;            //   0
    uint16_t version;          //   4  (1)
    uint16_t header_size;      //   6  (256)
    uint8_t  codec;            //   8
    uint8_t  codec_level;      //   9
    uint8_t  flags;            //  10
    uint8_t  protocol_id;      //  11  copied from the source .1fq
    uint32_t chunk_pairs;      //  12  max pairs per chunk
    uint64_t n_pairs;          //  16  pairs stored
    uint64_t n_pairs_in;       //  24  pairs in the source
    uint64_t n_dropped;        //  32  source reads dropped (R1 or R2 length 0, or R2 > 1024, as `singlet decode`)
    uint32_t n_chunks;         //  40
    uint16_t r2_slot;          //  44  bytes per R2 slot (multiple of 8)
    uint16_t r2_max_len;       //  46
    uint8_t  bc_len, umi_len, bc_start, r1_len;   // 48..51
    uint32_t max_exc_bytes;    //  52  largest raw exception section of any chunk
    uint64_t meta_offset;      //  56
    uint32_t meta_comp_size;   //  64  zstd-compressed JSON (provenance + the source .1fq metadata verbatim)
    uint32_t meta_raw_size;    //  68
    uint64_t chunks_offset;    //  72  first chunk
    uint64_t index_offset;     //  80
    uint64_t source_bytes;     //  88  size of the source file
    uint8_t  reserved[256 - 96];
};
struct SecDesc { uint32_t comp_size, raw_size, crc32c, offset; };   // offset from the chunk start
struct ChunkHeader {           // 64 bytes
    uint32_t magic, n_pairs, n_exc, flags;
    SecDesc  sec[NSEC];
};
struct IndexEntry { uint64_t offset; uint32_t total_bytes, n_pairs; };
struct Footer { uint32_t magic, n_chunks; uint64_t index_offset; uint32_t index_crc32c, pad; };   // 24 bytes
#pragma pack(pop)
static_assert(sizeof(FileHeader) == 256, "hdr");
static_assert(sizeof(ChunkHeader) == 64, "chunk hdr");

// CRC32C (Castagnoli) with the SSE4.2 instruction, one dependent 8 B step per iteration (3-cycle latency, ~2.7 B/cycle).
static inline uint32_t crc32c(const void* data, size_t n, uint32_t crc = 0) {
    const uint8_t* p = (const uint8_t*)data; uint64_t c = ~crc;
    while (n >= 8) { uint64_t v; std::memcpy(&v, p, 8); c = _mm_crc32_u64(c, v); p += 8; n -= 8; }
    uint32_t c32 = (uint32_t)c;
    while (n--) c32 = _mm_crc32_u8(c32, *p++);
    return ~c32;
}
static inline uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

}  // namespace fg
