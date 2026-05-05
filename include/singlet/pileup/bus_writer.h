#pragma once
// singlet-pileup: bus_writer.h
// BUS (Barcode, UMI, Set) binary format export — kallisto|bustools compatible.
// Header-only, no dependencies beyond libc.
//
// BUS v1 FILE LAYOUT:
//   [0..3]   magic    : "BUS\0"  (4 bytes, NUL-terminated string)
//   [4..7]   version  : uint32_t = 1
//   [8..11]  cb_len   : uint32_t (barcode length in bp)
//   [12..15] umi_len  : uint32_t (UMI length in bp)
//   [16..19] hlen     : uint32_t (length of text header string in bytes, excl. NUL)
//   [20..20+hlen-1]  header text (no NUL terminator in file)
//   [records...] each 32 bytes:
//     uint64_t barcode  (2-bit encoded, left-aligned, MSB first)
//     uint64_t umi      (2-bit encoded, left-aligned, MSB first)
//     int32_t  ec       (equivalence class ID / gene index; -1 = unmapped)
//     uint32_t count    (UMI count; 1 for raw per-UMI records)
//     uint32_t flags    (reserved; always 0)
//     uint32_t _pad     (padding to 32 bytes)
//
// 2-BIT ENCODING:  A=0b00, C=0b01, G=0b10, T=0b11
//   Stored left-aligned: first nucleotide in the most-significant bit pair.
//   e.g. "ACGT" (4-mer) → 0b00_01_10_11_00...00 = 0x1B000000_00000000ULL
//
// For bustools compatibility, records SHOULD be sorted by barcode before
// writing (the caller is responsible for sorting BusRecord vectors).
//
// References:
//   https://bustools.github.io/BUS_format
//   https://github.com/BUStools/BUS_notebooks_python/blob/master/busformat.ipynb

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

namespace singlet {

// ── Record ────────────────────────────────────────────────────────────────

/// One BUS record (32 bytes on disk).
struct BusRecord {
    uint64_t barcode = 0;  ///< 2-bit-encoded barcode (left-aligned, MSB first)
    uint64_t umi     = 0;  ///< 2-bit-encoded UMI (left-aligned, MSB first)
    int32_t  ec      = -1; ///< Equivalence class / gene index; -1 = unmapped
    uint32_t count   = 1;  ///< UMI count (1 for raw per-molecule records)
    uint32_t flags   = 0;  ///< Reserved; always 0
    uint32_t _pad    = 0;  ///< Padding to reach 32 bytes
};

static_assert(sizeof(BusRecord) == 32, "BusRecord must be exactly 32 bytes");

// ── Config ────────────────────────────────────────────────────────────────

struct BusWriteConfig {
    std::string filepath;
    uint32_t    cb_len      = 16;  ///< Barcode length (bp); e.g. 16 for 10x Chromium
    uint32_t    umi_len     = 12;  ///< UMI length (bp); e.g. 12 for 10x v3
    std::string header_text = "";  ///< Optional freeform text header
};

// ── Encoder / decoder helpers ─────────────────────────────────────────────

/// Encode a nucleotide string to a left-aligned 2-bit uint64_t.
/// Unrecognised characters (N, etc.) are encoded as A (0).
/// Maximum supported length is 32 nt (64-bit word).
inline uint64_t encode_bus_seq(const std::string& seq) {
    uint64_t result = 0;
    const size_t n  = seq.size() < 32 ? seq.size() : 32;
    for (size_t i = 0; i < n; ++i) {
        uint64_t bits = 0;
        switch (seq[i]) {
            case 'A': case 'a': bits = 0; break;
            case 'C': case 'c': bits = 1; break;
            case 'G': case 'g': bits = 2; break;
            case 'T': case 't': bits = 3; break;
            default:            bits = 0; break;  // N → A
        }
        result |= (bits << (62 - 2 * i));
    }
    return result;
}

/// Decode a 2-bit left-aligned uint64_t back to a nucleotide string of `len` bp.
inline std::string decode_bus_seq(uint64_t encoded, uint32_t len) {
    if (len > 32) len = 32;
    static const char BASES[4] = {'A', 'C', 'G', 'T'};
    std::string result(len, 'A');
    for (uint32_t i = 0; i < len; ++i) {
        uint32_t bits = static_cast<uint32_t>((encoded >> (62 - 2 * i)) & 0x3ULL);
        result[i] = BASES[bits];
    }
    return result;
}

/// Convenience alias matching the public API contract.
inline uint64_t encode_bus_barcode(const std::string& seq) { return encode_bus_seq(seq); }
inline std::string decode_bus_barcode(uint64_t encoded, uint32_t len) {
    return decode_bus_seq(encoded, len);
}

// ── BUS file header ───────────────────────────────────────────────────────

/// Parsed BUS file header (for read-back / validation).
struct BusHeader {
    char     magic[4] = {};    ///< Should be "BUS"
    uint32_t version  = 0;
    uint32_t cb_len   = 0;
    uint32_t umi_len  = 0;
    std::string header_text;   ///< Optional text header

    bool is_valid() const {
        return magic[0]=='B' && magic[1]=='U' && magic[2]=='S' && magic[3]=='\0'
            && version == 1;
    }
};

/// Read BUS file header (does NOT read records). Returns false on error.
inline bool read_bus_header(const std::string& filepath, BusHeader& hdr) {
    FILE* fp = std::fopen(filepath.c_str(), "rb");
    if (!fp) return false;

    // magic
    if (std::fread(hdr.magic, 1, 4, fp) != 4) { std::fclose(fp); return false; }
    if (std::fread(&hdr.version,  4, 1, fp) != 1) { std::fclose(fp); return false; }
    if (std::fread(&hdr.cb_len,   4, 1, fp) != 1) { std::fclose(fp); return false; }
    if (std::fread(&hdr.umi_len,  4, 1, fp) != 1) { std::fclose(fp); return false; }

    uint32_t hlen = 0;
    if (std::fread(&hlen, 4, 1, fp) != 1) { std::fclose(fp); return false; }

    if (hlen > 0) {
        hdr.header_text.resize(hlen);
        if (std::fread(&hdr.header_text[0], 1, hlen, fp) != hlen) {
            std::fclose(fp); return false;
        }
    }

    std::fclose(fp);
    return true;
}

/// Read all BUS records from a file (after the header). Returns empty on error.
inline std::vector<BusRecord> read_bus_records(const std::string& filepath) {
    FILE* fp = std::fopen(filepath.c_str(), "rb");
    if (!fp) return {};

    // skip header
    char m[4]; uint32_t v, cb, umi, hlen;
    if (std::fread(m,    1, 4, fp) != 4) { std::fclose(fp); return {}; }
    if (std::fread(&v,   4, 1, fp) != 1) { std::fclose(fp); return {}; }
    if (std::fread(&cb,  4, 1, fp) != 1) { std::fclose(fp); return {}; }
    if (std::fread(&umi, 4, 1, fp) != 1) { std::fclose(fp); return {}; }
    if (std::fread(&hlen,4, 1, fp) != 1) { std::fclose(fp); return {}; }
    if (hlen > 0) std::fseek(fp, hlen, SEEK_CUR);

    std::vector<BusRecord> recs;
    BusRecord rec;
    while (std::fread(&rec, sizeof(BusRecord), 1, fp) == 1) {
        recs.push_back(rec);
    }
    std::fclose(fp);
    return recs;
}

// ── Writer class ──────────────────────────────────────────────────────────

class BusWriter {
public:
    BusWriter() = default;
    ~BusWriter() { close(); }

    /// Open the BUS file and write the binary header.
    /// Returns false and logs to stderr on error.
    bool open(const BusWriteConfig& cfg) {
        cfg_     = cfg;
        written_ = 0;
        fp_ = std::fopen(cfg.filepath.c_str(), "wb");
        if (!fp_) {
            std::cerr << "[bus_writer] cannot open '" << cfg.filepath << "'\n";
            return false;
        }
        return write_header_();
    }

    /// Write one record. Returns false on I/O error.
    bool write_record(const BusRecord& rec) {
        if (!fp_) return false;
        if (std::fwrite(&rec, sizeof(BusRecord), 1, fp_) != 1) return false;
        ++written_;
        return true;
    }

    /// Write a vector of records. Returns false if any write fails.
    bool write_records(const std::vector<BusRecord>& records) {
        for (const auto& r : records) {
            if (!write_record(r)) return false;
        }
        return true;
    }

    void close() {
        if (fp_) { std::fflush(fp_); std::fclose(fp_); fp_ = nullptr; }
    }

    uint64_t records_written() const { return written_; }

private:
    FILE*           fp_      = nullptr;
    uint64_t        written_ = 0;
    BusWriteConfig  cfg_;

    bool write_header_() {
        // magic: "BUS\0"
        const char magic[4] = {'B','U','S','\0'};
        if (std::fwrite(magic, 1, 4, fp_) != 4)           return false;
        // version: 1
        uint32_t version = 1;
        if (std::fwrite(&version,       4, 1, fp_) != 1)  return false;
        // cb_len
        if (std::fwrite(&cfg_.cb_len,   4, 1, fp_) != 1)  return false;
        // umi_len
        if (std::fwrite(&cfg_.umi_len,  4, 1, fp_) != 1)  return false;
        // hlen + text
        uint32_t hlen = static_cast<uint32_t>(cfg_.header_text.size());
        if (std::fwrite(&hlen,          4, 1, fp_) != 1)  return false;
        if (hlen > 0) {
            if (std::fwrite(cfg_.header_text.data(), 1, hlen, fp_) != hlen)
                return false;
        }
        return true;
    }
};

} // namespace singlet
