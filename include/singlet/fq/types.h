// lib1fq/types.h — Core types and constants for the .1fq format
//
// Single-header definitions shared by encoder, decoder, and tools.
// No dependencies beyond C++17 standard library.

#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace lib1fq {

// ── Magic & version ──

static constexpr uint8_t MAGIC[4] = {'1', 'F', 'Q', '\0'};
static constexpr uint16_t FORMAT_VERSION = 4;

// ── Codec identifiers ──

enum class Codec : uint8_t {
    ZSTD = 0,
    LZ4 = 1,
    LZ4HC = 2,
    RANS = 3,
    NONE = 255,
};

// ── Protocol confidence ──

enum class Confidence : uint8_t {
    NONE = 0,
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3,
    MANUAL = 4,
    FORCE = 5,
};

// ── Sequence encoding ──

enum class SeqEncoding : uint8_t {
    PACKED_2BIT = 0,   // 4 bases/byte, A=00 C=01 G=10 T=11 + N-bitmap
    BYTE_NUMERIC = 1,  // 1 base/byte, A=0 C=1 G=2 T=3 N=4
};

// ── Quality modes ──

enum class QualMode : uint8_t {
    NONE = 0,     // No quality stored
    BINNED4 = 1,  // 2 bits/base, 4 bins
    FULL = 2,     // 6 bits/base, full phred
    BINNED2 = 3,  // 1 bit/base, pass/fail at Q20
};

// ── Compression tier ──

enum class Tier : uint8_t {
    RAW = 0,             // Tier 1: full sequences
    REF_COMPRESSED = 1,  // Tier 2: delta vs reference
};

// ── Header flags ──

namespace Flags {
static constexpr uint8_t DEDUPED = 0x01;
static constexpr uint8_t SORTED = 0x02;
static constexpr uint8_t TRIMMED = 0x04;
static constexpr uint8_t BC_DICT = 0x08;
static constexpr uint8_t BC_FILTERED = 0x10;
static constexpr uint8_t DELTA = 0x20;
static constexpr uint8_t REF_COMPRESS = 0x40;
static constexpr uint8_t INCOMPLETE = 0x80;
}  // namespace Flags

// ── Extra stream flags (stored in Header::reserved[0]) ──
// Used to signal optional additional read streams beyond the primary R1+R2.
namespace ExtraStreams {
static constexpr uint8_t HAS_I2 = 0x01;  // I2 barcode stream present (scATAC 3-read)
}

// ── Assay / modality type ──

enum class AssayType : uint8_t {
    UNKNOWN = 0,           // Unclassified
    SC_RNA_3PRIME = 1,     // scRNA-seq, 3' capture (10x, Drop-seq, etc.)
    SC_RNA_5PRIME = 2,     // scRNA-seq, 5' capture (10x 5', STRT-seq)
    SC_RNA_FULL = 3,       // Full-length scRNA-seq (Smart-seq2, FLASH-seq)
    SC_ATAC = 4,           // scATAC-seq (chromatin accessibility)
    SC_MULTIOME_GEX = 5,   // Multiome — GEX component
    SC_MULTIOME_ATAC = 6,  // Multiome — ATAC component
    SPATIAL_RNA = 7,       // Spatial transcriptomics (Visium, Slide-seq)
    CITE_SEQ_GEX = 8,      // CITE-seq — GEX component
    CITE_SEQ_ADT = 9,      // CITE-seq — ADT/HTO component
    SC_METHYLOME = 10,     // scBS-seq / scNMT (bisulfite-treated)
    BULK_RNA = 11,         // Bulk RNA-seq (no cell barcodes)
    WGS = 12,              // Whole genome sequencing
    WES = 13,              // Whole exome sequencing
    AMPLICON = 14,         // Targeted amplicon (e.g. CRISPR screens)
    SC_RNA_PLATE = 15,     // Plate-based scRNA (Smart-seq2 per-well)
    NOT_SEQUENCE = 254,    // Non-sequence data (corrupt, empty)
    MULTI = 255,           // Multi-assay container (reserved)
};

// ── Stream roles ──
// Identifies the biological role of each read stream in a library.

enum class StreamRole : uint8_t {
    R1 = 0,         // Read 1 (typically barcode+UMI or cDNA)
    R2 = 1,         // Read 2 (typically cDNA insert)
    I1 = 2,         // Index read 1 (sample demux)
    I2 = 3,         // Index read 2 (cell barcode in scATAC/Multiome)
    R3 = 4,         // Read 3
    R4 = 5,         // Read 4
    FEATURE = 6,    // Feature barcode (CITE-seq ADT/HTO)
    UNKNOWN = 255,  // Unassigned
};

// ── Segment types ──

enum class SegmentType : uint8_t {
    BC = 0,       // Barcode
    UMI = 1,      // Unique molecular identifier
    CDNA = 2,     // cDNA / mRNA insert
    GENOMIC = 3,  // Genomic DNA insert (ATAC, etc.)
    LINKER = 4,   // Linker / spacer (SPLiT-seq, etc.)
    INDEX = 5,    // Sample index
    FEATURE = 6,  // Feature barcode (CITE-seq, etc.)
    RAW = 7,      // Unclassified / unknown
};

// ── Stream descriptor ──
// Describes one read stream (R1, R2, I1, I2, etc.)

struct StreamDesc {
    uint8_t index;          // 0-based stream index
    uint16_t fixed_length;  // 0 = variable length

    static constexpr uint8_t MAX_SEGMENTS = 8;
    uint8_t n_segments;
    struct Segment {
        SegmentType type;
        uint16_t offset;
        uint16_t length;  // 0 = remainder
    } segments[MAX_SEGMENTS];
};

// ── File header (96 bytes, packed) ──

#pragma pack(push, 1)
struct Header {
    uint8_t magic[4];            //  0: "1FQ\0"
    uint16_t version;            //  4: format version
    uint8_t n_streams;           //  6: read streams present (1–4)
    uint8_t protocol_id;         //  7: protocol enum (0=UNKNOWN)
    uint8_t confidence;          //  8: Confidence enum
    uint8_t flags;               //  9: bitfield (Flags::*)
    uint8_t codec;               // 10: Codec enum
    uint8_t codec_level;         // 11: codec level
    uint16_t stream_lengths[4];  // 12: per-stream fixed length (0=variable)
    uint64_t n_unique;           // 20: unique reads/spots stored
    uint64_t n_original;         // 28: original total spots
    uint32_t block_count;        // 36: number of data blocks
    uint32_t block_size;         // 40: target reads per block
    uint32_t meta_size;          // 44: metadata block compressed size
    uint32_t bc_dict_size;       // 48: barcode dict compressed size
    uint16_t registry_version;   // 52: protocol registry version
    uint8_t seq_encoding;        // 54: SeqEncoding enum
    uint8_t qual_mode;           // 55: QualMode enum
    uint8_t tier;                // 56: Tier enum
    uint8_t assay_type;          // 57: AssayType enum
    uint8_t stream_roles[4];     // 58–61: StreamRole per stream (up to 4)
    uint8_t reserved[34];        // 62–95: zero-filled

    void init() {
        std::memset(this, 0, sizeof(*this));
        std::memcpy(magic, MAGIC, 4);
        version = FORMAT_VERSION;
        codec = static_cast<uint8_t>(Codec::ZSTD);
        codec_level = 6;
        seq_encoding = static_cast<uint8_t>(SeqEncoding::PACKED_2BIT);
        qual_mode = static_cast<uint8_t>(QualMode::BINNED4);
        tier = static_cast<uint8_t>(Tier::RAW);
        block_size = 100000;
    }

    bool valid_magic() const {
        return std::memcmp(magic, MAGIC, 4) == 0;
    }
};
static_assert(sizeof(Header) == 96, "Header must be exactly 96 bytes");

// ── Block header (12 bytes, per data block) ──

struct BlockHeader {
    uint32_t n_reads;          // Reads in this block
    uint32_t compressed_size;  // Compressed payload size in bytes
    uint32_t raw_crc32;        // CRC32 of uncompressed payload
};
static_assert(sizeof(BlockHeader) == 12, "BlockHeader must be 12 bytes");

// ── Footer (16 bytes) ──

struct Footer {
    uint32_t file_crc32;    // CRC32 of entire file (excluding footer)
    uint32_t block_count;   // Redundant with header, for validation
    uint32_t index_offset;  // Byte offset of block index from file start
    uint8_t magic[4];       // "1FQ\0"

    void init(uint32_t n_blocks, uint64_t idx_off) {
        file_crc32 = 0;
        block_count = n_blocks;
        // index_offset is uint32_t; truncates for files >4 GiB.
        // Readers compute the index position from EOF and do not rely on
        // this field for correctness (it is stored only for diagnostic use).
        index_offset = static_cast<uint32_t>(idx_off);
        std::memcpy(magic, MAGIC, 4);
    }

    bool valid_magic() const {
        return std::memcmp(magic, MAGIC, 4) == 0;
    }
};
static_assert(sizeof(Footer) == 16, "Footer must be exactly 16 bytes");

#pragma pack(pop)

// ── Binned-4 quality mapping ──

namespace qual {

// Phred → 2-bit bin
inline uint8_t phred_to_bin4(uint8_t phred) {
    if (phred < 10) return 0;
    if (phred < 20) return 1;
    if (phred < 30) return 2;
    return 3;
}

// 2-bit bin → approximate Phred (bin center)
inline uint8_t bin4_to_phred(uint8_t bin) {
    static constexpr uint8_t centers[4] = {6, 15, 25, 37};
    return centers[bin & 0x03];
}

// Phred → 1-bit bin (pass/fail at Q20)
inline uint8_t phred_to_bin2(uint8_t phred) {
    return phred >= 20 ? 1 : 0;
}

// 1-bit bin → approximate Phred
inline uint8_t bin2_to_phred(uint8_t bin) {
    static constexpr uint8_t centers[2] = {10, 30};
    return centers[bin & 0x01];
}

}  // namespace qual

// ── Nucleotide encoding ──

namespace nuc {

// ASCII → byte-numeric (matches STAR's nucl_to_num)
// Uses a 256-byte lookup table instead of switch for branchless conversion.
inline uint8_t ascii_to_num(char c) {
    // A/a→0, C/c→1, G/g→2, T/t→3, everything else→4 (N)
    static constexpr uint8_t lut[256] = {
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0x00-0x0F
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0x10-0x1F
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0x20-0x2F
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0x30-0x3F
        4,
        0,
        4,
        1,
        4,
        4,
        4,
        2,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0x40-0x4F: A=0, C=1, G=2
        4,
        4,
        4,
        4,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0x50-0x5F: T=3
        4,
        0,
        4,
        1,
        4,
        4,
        4,
        2,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0x60-0x6F: a=0, c=1, g=2
        4,
        4,
        4,
        4,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0x70-0x7F: t=3
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0x80-0x8F
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0x90-0x9F
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0xA0-0xAF
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0xB0-0xBF
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0xC0-0xCF
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0xD0-0xDF
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0xE0-0xEF
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  // 0xF0-0xFF
    };
    return lut[static_cast<uint8_t>(c)];
}

// Byte-numeric → ASCII
inline char num_to_ascii(uint8_t n) {
    static constexpr char tbl[5] = {'A', 'C', 'G', 'T', 'N'};
    return (n < 5) ? tbl[n] : 'N';
}

// Byte-numeric → 2-bit (N becomes A=0, caller must track N positions)
inline uint8_t num_to_2bit(uint8_t n) {
    return n & 0x03;
}

// 2-bit → byte-numeric (without N recovery)
inline uint8_t twoBit_to_num(uint8_t b) {
    return b & 0x03;
}

}  // namespace nuc

// ── Encoder configuration (shared by SraEncoder and FastqEncoder) ──

// ── CRC32 utility (self-contained, no zlib dependency) ──

namespace detail {
inline const uint32_t* crc32_table() {
    static uint32_t table[256] = {};
    static bool initialized = false;
    if (!initialized) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t crc = i;
            for (int j = 0; j < 8; ++j)
                crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
            table[i] = crc;
        }
        initialized = true;
    }
    return table;
}
}  // namespace detail

inline uint32_t compute_crc32(const void* data, size_t len) {
    const auto* p = static_cast<const uint8_t*>(data);
    const uint32_t* tbl = detail::crc32_table();
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; ++i)
        crc = tbl[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFF;
}

inline uint32_t update_crc32(uint32_t prev, const void* data, size_t len) {
    const auto* p = static_cast<const uint8_t*>(data);
    const uint32_t* tbl = detail::crc32_table();
    uint32_t crc = prev ^ 0xFFFFFFFF;
    for (size_t i = 0; i < len; ++i)
        crc = tbl[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFF;
}

// ── SRA / GEO sample metadata ──
// Carries all descriptive metadata from SRA/GEO XML for embedding in
// the .1fq metadata block. All fields are optional — empty = unknown.

struct SraMetadata {
    // ── Accession hierarchy ──
    std::string srr;        // Run accession (SRR / ERR / DRR)
    std::string srx;        // Experiment accession
    std::string srp;        // Study/project accession
    std::string gse;        // GEO series accession
    std::string gsm;        // GEO sample accession
    std::string biosample;  // BioSample accession (SAMN...)

    // ── Sample biology ──
    std::string organism;             // e.g. "Homo sapiens"
    std::string taxon_id;             // NCBI taxonomy ID (e.g. "9606")
    std::string tissue;               // Tissue / organ
    std::string cell_type;            // Cell type or cell line
    std::string disease;              // Disease state ("normal" if healthy)
    std::string developmental_stage;  // e.g. "adult", "E14.5"
    std::string sex;                  // "male", "female", "mixed", "unknown"

    // ── Library ──
    std::string library_name;
    std::string library_strategy;   // RNA-Seq, ATAC-seq, ChIP-Seq ...
    std::string library_source;     // TRANSCRIPTOMIC, GENOMIC ...
    std::string library_selection;  // cDNA, ChIP, RANDOM ...
    std::string library_layout;     // PAIRED, SINGLE

    // ── Platform / instrument ──
    std::string platform;    // ILLUMINA, OXFORD_NANOPORE ...
    std::string instrument;  // NextSeq 500, NovaSeq 6000 ...

    // ── Submission provenance ──
    std::string center_name;
    std::string submission_id;
    std::string submission_date;  // ISO 8601 date string
    std::string last_updated;     // ISO 8601 date string

    // ── Descriptive fields ──
    std::string title;  // Run title
    std::string experiment_title;
    std::string study_title;
    std::string study_abstract;
    std::string sample_description;

    // ── Arbitrary sample attributes (from SRA SAMPLE_ATTRIBUTE) ──
    std::vector<std::pair<std::string, std::string>> sample_attributes;
};

struct EncoderConfig {
    std::string output_path;
    Codec codec = Codec::ZSTD;
    int codec_level = 4;
    QualMode qual_mode = QualMode::BINNED4;
    SeqEncoding seq_enc = SeqEncoding::PACKED_2BIT;
    uint32_t block_size = 100000;
    uint16_t r2_maxlen = 0;  // Max R2 length in bases (0=unlimited)

    // Protocol override (empty = auto-detect)
    std::string protocol_tag;

    // Catalog-provided protocol from metadata JSON.
    // When non-empty and auto-detection returns confidence <= LOW (1),
    // the detected protocol is replaced by this value (see AUTOFIX-PROTOCOL-CONFIDENCE-OVERRIDE).
    std::string metadata_protocol;

    // Whitelist paths for auto-detection (searched in order)
    std::vector<std::string> whitelist_dirs;

    // Feature disable flags (override protocol-level defaults)
    bool no_dedup = false;    // --no-dedup: skip PCR duplicate collapsing
    bool no_trim = false;     // --no-trim: skip polyA/adapter trimming
    bool sort_by_bc = false;  // --sort-by-bc: sort reads by barcode within blocks

    // Parallel VDB reading threads (1=serial default)
    int vdb_threads = 1;

    // Maximum number of reads to encode (0=unlimited)
    uint64_t max_reads = 0;

    // Verbose diagnostics
    bool verbose = false;

    // SRA / GEO metadata to embed in the .1fq metadata block
    SraMetadata sra_meta;

    // Progress callback: called every N reads
    std::function<void(uint64_t reads, uint64_t total)> progress_cb;
    uint64_t progress_interval = 1000000;
};

}  // namespace lib1fq
