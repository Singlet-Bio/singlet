// .1fg v2: staged read container for the fetch tier -> GPU/CPU drain workers. DRAFT for review by S8 (consumer), 2026-10-31.
// Spec of the planes: docs/research-beta/staged-format-1fg2.md (gpu/host-offload, PR #107). This file is the byte layout.
// Extends v1 (ingest_fmt.h): same chunk/section idea, same zstd-frame planes (libzstd, nvCOMP-compatible: content size in the frame,
// no dictionary, no zstd checksum) and CRC32C of the STORED bytes. All integers little-endian. All offsets are absolute file offsets.
//
// File:   FileHeader (4096) | read chunks (each 4096-aligned) | sequence blocks (each 4096-aligned) | run table | index | Footer
// Reads are in source order (run by run), NO barcode sort. A read carries ids, not text; text lives in the sequence table.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <nmmintrin.h>

namespace fg2 {

static constexpr uint32_t FILE_MAGIC  = 0x00474631u;   // "1FG\0" (same as v1; `version` tells them apart)
static constexpr uint16_t VERSION     = 2;
static constexpr uint32_t CHUNK_MAGIC = 0x32484b43u;   // "CKH2"
static constexpr uint32_t SEQB_MAGIC  = 0x31514553u;   // "SEQ1"
static constexpr uint32_t FOOT_MAGIC  = 0x32475446u;   // "FTG2"
static constexpr uint32_t ALIGN       = 4096;          // chunk / block start alignment (O_DIRECT, GDS)
static constexpr uint32_t SEC_ALIGN   = 16;            // plane start alignment inside a chunk or block (nvCOMP needs 8)
static constexpr uint32_t CHUNK_READS = 32768;         // reads per chunk (the last one may be shorter)
static constexpr uint32_t BLOCK_SEQS  = 32768;         // sequences per table block
static constexpr uint32_t NO_CB       = 0xFFFFFFFFu;   // `cb` of a read without a whitelist barcode (cbst says why)
static constexpr uint32_t MAX_SEQ_LEN = 255;           // 8-bit length (v1 had 127); 2 x 150 runs are common. Longer cDNA is refused by the encoder

enum Codec : uint8_t { CODEC_RAW = 0, CODEC_ZSTD = 1 };
enum Flags : uint32_t {
    F_CRC        = 1u << 0,   // every plane has a CRC32C of its stored bytes
    F_QPLANE     = 1u << 1,   // reads carry the `q` plane (binned quality, 2 bit/base); qual_mode 0 sources leave it out
    F_INCOMPLETE = 1u << 2,   // writer has not finished (footer missing)
    F_CB_FIXED   = 1u << 3,   // `cb`/`cbst` are the R11 correction of this file's own exact histogram (always set by the fetch tier)
};

// Status of the barcode, 1 byte per read: low 3 bits = CbCorrector::resolve() status + 4 (S8, 2026-10-31), bit 3 = the UMI is bad.
enum CbStatus : uint8_t {
    CBST_REJECTED = 0,       // resolve() -4: several candidates, posterior < 0.975        cb = NO_CB
    CBST_TWO_N = 2,          // resolve() -2: two or more N in the barcode                 cb = NO_CB
    CBST_NO_MATCH = 3,       // resolve() -1: no candidate                                 cb = NO_CB
    CBST_EXACT = 4,          // resolve()  0: exact whitelist hit
    CBST_ONE_CAND = 5,       // resolve()  1: one single-substitution candidate
    CBST_POSTERIOR = 6,      // resolve()  2: several candidates, posterior >= 0.975
    CBST_BAD_UMI = 8,        // flag: the UMI has an N or is a homopolymer (counted, not used; cb is still set when the barcode corrects)
};

// Read planes inside a read chunk
enum ReadPlane { PL_CB = 0, PL_CBST = 1, PL_UMI = 2, PL_SEQ = 3, PL_LEN = 4, PL_Q = 5, PL_EXC = 6, PL_RAWCB = 7, NPLANE = 8 };
//   PL_CB   u32[n]            whitelist line number (0-based) of the corrected barcode, or NO_CB
//   PL_CBST u8[n]             CbStatus
//   PL_UMI  u32[n]            UMI, 2 bit/base, first base in the high bits, N coded as A (see PL_EXC)
//   PL_SEQ  u32[n]            global sequence id = block * BLOCK_SEQS + index in block
//   PL_LEN  u8[n]             cDNA length of THIS read in bases (<= the sequence's slen: reads may share a sequence of a longer length? NO:
//                             the key includes the length, so len == slen[seq]; the plane is kept so the readers need not touch the table)
//   PL_Q    u8[n * qslot]     only if F_QPLANE. 2 bit/base binned quality, first base in the high bits, zero padded, qslot = seq_slot bytes;
//                             bin of base i = 0..3; FileHeader::q_edges[bin] is the decoded Phred value the engine uses (compare with --min-baseq: only bin 0 is low)
//   PL_EXC  u32[n_exc]        sorted by read index: bits 0-14 index in chunk, 15-16 kind, 17-23 position. Kinds: UMI_N (the UMI word holds A at pos)
//   PL_RAWCB u32[2 * n_rawcb]  one 8-byte entry for EVERY read whose barcode was not an exact whitelist hit (cbst low 3 bits != CBST_EXACT), sorted by read:
//                             word  = the raw barcode as read, 2 bit/base, first base in the high bits, N coded as A (<= 16 nt)
//                             meta  = bits 0-14 read index in chunk, bits 15-19 N position + 1 (0 = no N, 1..16 = that base, 31 = two or more N)
//                             Needed so STAR's CellReads.stats (exact / corrected / no match) and the cell QC stay identical: cb/cbst give the
//                             corrected barcode and the class, this gives what was actually sequenced (also for NO_CB reads, whose raw barcode
//                             the old .1fq path kept in its dictionary-miss list).
enum ExcKind { EXC_UMI_N = 0, EXC_RESERVED1 = 1, EXC_RESERVED2 = 2, EXC_RESERVED3 = 3 };
static inline uint32_t exc_pack(uint32_t idx, uint32_t kind, uint32_t pos) { return idx | (kind << 15) | (pos << 17); }
static inline uint32_t exc_idx(uint32_t e) { return e & 0x7FFF; }
static inline uint32_t exc_kind(uint32_t e) { return (e >> 15) & 3; }
static inline uint32_t exc_pos(uint32_t e) { return (e >> 17) & 0x7F; }

// Sequence-table planes inside a block
enum SeqPlane { SP_SEQ2 = 0, SP_SLEN = 1, SP_SEXC = 2, NSEQPLANE = 3 };
//   SP_SEQ2 u8[n * seq_slot]  2 bit/base, first base in the high bits, zero padded; N coded as A (see SP_SEXC)
//   SP_SLEN u8[n]             length in bases (1..255)
//   SP_SEXC u32[n_sexc]       sorted: bits 0-14 index in block, 15-22 position of an N (one entry per N base)
// Key = the exact stored cDNA exactly as the .1fq holds it: BEFORE any 5' prefix (r2_clip5p), CR4 or 3' clip (the engine applies them on the packed codes), N and length included.
// Order = first occurrence in the read stream (deterministic). One table per file.

#pragma pack(push, 1)
struct SecDesc { uint32_t comp_size, raw_size, crc32c, offset; };   // offset from the start of the chunk / block

struct FileHeader {                 // 4096 bytes at offset 0
    uint32_t magic;                 //   0  FILE_MAGIC
    uint16_t version;               //   4  2
    uint16_t header_size;           //   6  4096
    uint8_t  codec;                 //   8  Codec
    uint8_t  codec_level;           //   9  zstd level (3 default)
    uint16_t pad0;                  //  10
    uint32_t flags;                 //  12  Flags
    uint32_t chunk_reads;           //  16  CHUNK_READS
    uint32_t block_seqs;            //  20  BLOCK_SEQS
    uint64_t n_reads;               //  24  reads stored
    uint64_t n_reads_in;            //  32  reads in the decoded runs
    uint64_t n_dropped;             //  40  dropped (R1 or R2 length 0, cDNA > 255 refused)
    uint64_t n_unique;              //  48  rows of the sequence table
    uint32_t n_chunks;              //  56
    uint32_t n_blocks;              //  60
    uint16_t seq_slot;              //  64  bytes per sequence in SP_SEQ2 (and per read in PL_Q): 8-byte multiple >= ceil(max_len / 4)
    uint16_t r2_max_len;            //  66
    uint16_t r2_clip5p;             //  68  always 0 (the engine's own probe decides the 5' clip; nothing is clipped before keying)
    uint8_t  protocol_id;           //  70  as the source .1fq
    uint8_t  cb_len, cb_start, umi_len, umi_start, r1_len;   // 71..75  geometry (0-based starts)
    uint8_t  q_edges[4];            //  76  DECODED Phred value of bins 0..3 (ILLUMINA4: 2, 11, 25, 37; source bins <10, 10-19, 20-29, >=30); zero if no q plane
    uint8_t  q_bins;                //  80  4 or 0
    uint8_t  pad1[3];               //  81
    uint32_t wl_entries;            //  84  lines in the whitelist (valid `cb` < wl_entries)
    uint8_t  wl_sha256[32];         //  88  of the whitelist file used for the correction
    char     wl_name[64];           // 120  file name, e.g. 3M-february-2018.txt
    uint64_t cb_hist_reads;         // 184  reads in the exact histogram (exact whitelist hit, good UMI)
    uint64_t status_counts[8];      // 192  reads per CbStatus (CellReads.stats cbMatch/cbPerfect/cbMMunique/cbMMmultiple...)
    uint64_t meta_offset;           // 256  zstd JSON: provenance (accessions, bytes, md5, singlet version), source .1fq metadata verbatim
    uint32_t meta_comp_size, meta_raw_size;   // 264
    uint64_t chunks_offset;         // 272
    uint64_t blocks_offset;         // 280
    uint64_t runs_offset;           // 288  run table (RunEntry[n_runs])
    uint32_t n_runs, pad2;          // 296
    uint64_t index_offset;          // 304  chunk index, then block index
    uint64_t source_bytes;          // 312  total size of the source (.sra / .1fq) files
    uint32_t header_crc32c;         // 320  of bytes 0..319
    uint8_t  reserved[4096 - 324];
};

struct ChunkHeader {                // 160 bytes at the start of every read chunk
    uint32_t magic;                 //   0  CHUNK_MAGIC
    uint32_t n_reads;               //   4
    uint32_t n_exc;                 //   8
    uint32_t flags;                 //  12  bit s set: plane s stored raw (not compressible)
    uint64_t first_read;            //  16  global index of the chunk's first read
    uint32_t n_rawcb, pad1;         //  24  entries in PL_RAWCB
    SecDesc  sec[NPLANE];           //  32  8 * 16 = 128 -> 160 total
};

struct BlockHeader {                // 72 bytes at the start of every sequence block (planes start at 80)
    uint32_t magic;                 //   0  SEQB_MAGIC
    uint32_t n_seqs;                //   4
    uint32_t n_sexc;                //   8
    uint32_t flags;                 //  12
    uint64_t first_seq;             //  16
    SecDesc  sec[NSEQPLANE];        //  24  3 * 16 = 48 -> 72 total
};

struct RunEntry { char accession[24]; uint64_t first_read, n_reads; uint64_t raw_bytes; uint32_t flags; uint32_t pad; };   // 56 bytes; runs_used provenance
struct IndexEntry { uint64_t offset; uint32_t total_bytes, n; };   // chunk: n = reads; block: n = sequences
struct Footer { uint32_t magic, n_chunks, n_blocks; uint32_t index_crc32c; uint64_t index_offset; };
#pragma pack(pop)

static_assert(sizeof(SecDesc) == 16, "SecDesc");
static_assert(sizeof(ChunkHeader) == 160, "ChunkHeader");
static_assert(sizeof(BlockHeader) == 72, "BlockHeader");
static_assert(offsetof(FileHeader, header_crc32c) == 320, "FileHeader layout: crc after the last field");
static_assert(offsetof(FileHeader, wl_sha256) == 88 && offsetof(FileHeader, status_counts) == 192 && offsetof(FileHeader, meta_offset) == 256, "FileHeader offsets");
static_assert(sizeof(FileHeader) == 4096, "FileHeader");
static_assert(sizeof(RunEntry) == 56, "RunEntry");
static_assert(sizeof(IndexEntry) == 16, "IndexEntry");

// CRC32C (Castagnoli) with the SSE4.2 instruction (as v1).
static inline uint32_t crc32c(const void* data, size_t n, uint32_t crc = 0) {
    const uint8_t* p = (const uint8_t*)data; uint64_t c = ~crc;
    while (n >= 8) { uint64_t v; std::memcpy(&v, p, 8); c = _mm_crc32_u64(c, v); p += 8; n -= 8; }
    uint32_t c32 = (uint32_t)c;
    while (n--) c32 = _mm_crc32_u8(c32, *p++);
    return ~c32;
}
static inline uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

}  // namespace fg2
