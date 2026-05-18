// SPDX-License-Identifier: MIT
// sra_reader.h — Native SRA file reader using NCBI VDB C API
//
// Zero-copy reads from .sra files via VCursorCellDataDirect.
// No external tools (fastq-dump, fasterq-dump) required at runtime.
// Replaces exec-based demuxer with direct VDB library calls.
//
// Dependencies: libncbi-vdb (NCBI public domain)
// Columns used: READ, QUALITY, READ_LEN, READ_START, NAME

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <stdexcept>

// VDB C API headers (public domain, NCBI)
#include <kfc/defs.h>    // rc_t = uint32_t
#include <vdb/manager.h>
#include <vdb/database.h>
#include <vdb/table.h>
#include <vdb/cursor.h>

class SraReader {
public:
    SraReader() = default;
    ~SraReader() { close(); }

    // Non-copyable
    SraReader(const SraReader&) = delete;
    SraReader& operator=(const SraReader&) = delete;

    // Open an SRA file/accession and prepare cursor on SEQUENCE table
    void open(const char* path) {
        rc_t rc;

        rc = VDBManagerMakeRead(&mgr_, nullptr);
        if (rc) throw std::runtime_error("VDBManagerMakeRead failed: rc=" + std::to_string(rc));

        // Try opening as database first (standard SRA layout)
        rc = VDBManagerOpenDBRead(mgr_, &db_, nullptr, "%s", path);
        if (rc == 0) {
            // Database with SEQUENCE table
            rc = VDatabaseOpenTableRead(db_, &tbl_, "SEQUENCE");
            if (rc) throw std::runtime_error("VDatabaseOpenTableRead(SEQUENCE) failed: rc=" + std::to_string(rc));
        } else {
            // Might be a flat table (some older SRA files)
            db_ = nullptr;
            rc = VDBManagerOpenTableRead(mgr_, &tbl_, nullptr, "%s", path);
            if (rc) throw std::runtime_error("Cannot open SRA as database or table: " + std::string(path));
        }

        rc = VTableCreateCursorRead(tbl_, &cursor_);
        if (rc) throw std::runtime_error("VTableCreateCursorRead failed: rc=" + std::to_string(rc));

        // Add columns
        rc = VCursorAddColumn(cursor_, &col_read_, "(INSDC:dna:text)READ");
        if (rc) throw std::runtime_error("VCursorAddColumn(READ) failed: rc=" + std::to_string(rc));

        rc = VCursorAddColumn(cursor_, &col_qual_, "(INSDC:quality:phred)QUALITY");
        if (rc) throw std::runtime_error("VCursorAddColumn(QUALITY) failed: rc=" + std::to_string(rc));

        rc = VCursorAddColumn(cursor_, &col_read_len_, "(INSDC:coord:len)READ_LEN");
        if (rc) throw std::runtime_error("VCursorAddColumn(READ_LEN) failed: rc=" + std::to_string(rc));

        rc = VCursorAddColumn(cursor_, &col_read_start_, "(INSDC:coord:zero)READ_START");
        if (rc) throw std::runtime_error("VCursorAddColumn(READ_START) failed: rc=" + std::to_string(rc));

        rc = VCursorAddColumn(cursor_, &col_name_, "(ascii)NAME");
        if (rc) throw std::runtime_error("VCursorAddColumn(NAME) failed: rc=" + std::to_string(rc));

        rc = VCursorOpen(cursor_);
        if (rc) throw std::runtime_error("VCursorOpen failed: rc=" + std::to_string(rc));

        // Get row range
        rc = VCursorIdRange(cursor_, col_read_, &first_row_, &row_count_);
        if (rc) throw std::runtime_error("VCursorIdRange failed: rc=" + std::to_string(rc));

        current_row_ = first_row_;
    }

    // Close all VDB handles
    void close() {
        if (cursor_) { VCursorRelease(cursor_); cursor_ = nullptr; }
        if (tbl_)    { VTableRelease(tbl_);     tbl_ = nullptr; }
        if (db_)     { VDatabaseRelease(db_);   db_ = nullptr; }
        if (mgr_)    { VDBManagerRelease(mgr_); mgr_ = nullptr; }
    }

    int64_t total_spots() const { return static_cast<int64_t>(row_count_); }
    int64_t spots_read() const { return current_row_ - first_row_; }
    int64_t first_row() const { return first_row_; }
    bool done() const { return current_row_ >= first_row_ + static_cast<int64_t>(row_count_); }

    // Seek to a specific row (for parallel chunk reading)
    void seek(int64_t row) { current_row_ = row; }

    // Set a row limit (stop reading at this row, exclusive)
    void set_row_limit(int64_t end_row) { row_limit_ = end_row; }

    // Check if done accounting for row limit
    bool done_range() const {
        if (row_limit_ > 0) return current_row_ >= row_limit_;
        return done();
    }

    // Zero-copy spot data — pointers into VDB cache, valid until next read
    struct RawSpot {
        const char*     read;       // ASCII DNA (ACGTN), full concatenated spot
        const uint8_t*  qual;       // Raw phred scores (NOT +33 ASCII)
        const uint32_t* seg_len;    // Length of each segment (n_segments entries)
        const int32_t*  seg_start;  // Start offset of each segment
        uint32_t        n_segments; // Number of read segments (typically 2)
        uint32_t        total_bases;
    };

    // Read next spot, returning zero-copy pointers into VDB cache.
    // Pointers are valid until the next call to read_next_spot().
    // Returns false when all spots consumed.
    bool read_next_spot(RawSpot& spot) {
        if (done()) return false;

        int64_t row = current_row_++;
        uint32_t elem_bits, boff, row_len;
        rc_t rc;

        rc = VCursorCellDataDirect(cursor_, row, col_read_,
                                   &elem_bits, (const void**)&spot.read,
                                   &boff, &row_len);
        if (rc) return false;
        spot.total_bases = row_len;

        rc = VCursorCellDataDirect(cursor_, row, col_qual_,
                                   &elem_bits, (const void**)&spot.qual,
                                   &boff, &row_len);
        if (rc) return false;

        rc = VCursorCellDataDirect(cursor_, row, col_read_len_,
                                   &elem_bits, (const void**)&spot.seg_len,
                                   &boff, &spot.n_segments);
        if (rc) return false;

        rc = VCursorCellDataDirect(cursor_, row, col_read_start_,
                                   &elem_bits, (const void**)&spot.seg_start,
                                   &boff, &row_len);
        if (rc) return false;

        return true;
    }

    // Fast read: only READ + QUALITY columns (2 VDB calls instead of 4).
    // Use when segment layout is already known from probe phase.
    // Caller must split read/qual at known offsets.
    bool read_next_spot_fast(RawSpot& spot) {
        if (done()) return false;

        int64_t row = current_row_++;
        uint32_t elem_bits, boff, row_len;
        rc_t rc;

        rc = VCursorCellDataDirect(cursor_, row, col_read_,
                                   &elem_bits, (const void**)&spot.read,
                                   &boff, &row_len);
        if (rc) return false;
        spot.total_bases = row_len;

        rc = VCursorCellDataDirect(cursor_, row, col_qual_,
                                   &elem_bits, (const void**)&spot.qual,
                                   &boff, &row_len);
        if (rc) return false;

        // Segment info not read — caller must use known fixed offsets
        spot.n_segments = 0;
        spot.seg_len = nullptr;
        spot.seg_start = nullptr;

        return true;
    }

    // Write next spot as FASTQ text to R1 and R2 file handles.
    // Returns false when all spots are consumed.
    // For paired-end data, writes 4 FASTQ lines to each of r1_fp and r2_fp.
    // For single-end data, writes to r1_fp only.
    bool write_next_spot(FILE* r1_fp, FILE* r2_fp) {
        if (done()) return false;

        int64_t row = current_row_++;
        uint32_t elem_bits, boff, row_len;
        rc_t rc;

        // --- READ (zero-copy pointer to DNA text) ---
        const char* read_data;
        rc = VCursorCellDataDirect(cursor_, row, col_read_,
                                   &elem_bits, (const void**)&read_data,
                                   &boff, &row_len);
        if (rc) return false;
        uint32_t total_bases = row_len;

        // --- QUALITY (zero-copy pointer to phred scores) ---
        const uint8_t* qual_data;
        rc = VCursorCellDataDirect(cursor_, row, col_qual_,
                                   &elem_bits, (const void**)&qual_data,
                                   &boff, &row_len);
        if (rc) return false;

        // --- READ_LEN (number of bases per read segment) ---
        const uint32_t* read_len;
        uint32_t n_reads;
        rc = VCursorCellDataDirect(cursor_, row, col_read_len_,
                                   &elem_bits, (const void**)&read_len,
                                   &boff, &n_reads);
        if (rc) return false;

        // --- READ_START (start offset per read segment) ---
        const int32_t* read_start;
        uint32_t n_starts;
        rc = VCursorCellDataDirect(cursor_, row, col_read_start_,
                                   &elem_bits, (const void**)&read_start,
                                   &boff, &n_starts);
        if (rc) return false;

        // --- NAME ---
        const char* name_data;
        uint32_t name_len;
        rc = VCursorCellDataDirect(cursor_, row, col_name_,
                                   &elem_bits, (const void**)&name_data,
                                   &boff, &name_len);
        if (rc) return false;

        // Write FASTQ records
        if (n_reads >= 2 && read_len[0] > 0 && read_len[1] > 0) {
            // Paired-end spot: R1 = first segment, R2 = second segment
            write_fastq_record(r1_fp, name_data, name_len, "/1",
                              read_data + read_start[0], read_len[0],
                              qual_data + read_start[0], read_len[0]);
            write_fastq_record(r2_fp, name_data, name_len, "/2",
                              read_data + read_start[1], read_len[1],
                              qual_data + read_start[1], read_len[1]);
        } else if (n_reads == 1 || (n_reads >= 2 && read_len[1] == 0)) {
            // Single-end: write to R1 only
            uint32_t len = read_len[0];
            uint32_t off = (n_starts > 0) ? read_start[0] : 0;
            write_fastq_record(r1_fp, name_data, name_len, "",
                              read_data + off, len,
                              qual_data + off, len);
        }
        // Skip spots with 0 reads or 0-length segments

        return true;
    }

private:
    // Write a single FASTQ record (4 lines: @name, seq, +, qual)
    static void write_fastq_record(FILE* fp,
                                   const char* name, uint32_t name_len,
                                   const char* suffix,
                                   const char* seq, uint32_t seq_len,
                                   const uint8_t* qual, uint32_t qual_len) {
        // @NAME/1 or @NAME/2
        fputc('@', fp);
        fwrite(name, 1, name_len, fp);
        fputs(suffix, fp);
        fputc('\n', fp);

        // Sequence
        fwrite(seq, 1, seq_len, fp);
        fputc('\n', fp);

        // Separator
        fputs("+\n", fp);

        // Quality: convert phred int to ASCII (phred + 33)
        for (uint32_t i = 0; i < qual_len; ++i) {
            fputc(qual[i] + 33, fp);
        }
        fputc('\n', fp);
    }

    // VDB handles
    const VDBManager* mgr_    = nullptr;
    const VDatabase*  db_     = nullptr;
    const VTable*     tbl_    = nullptr;
    const VCursor*    cursor_ = nullptr;

    // Column indices
    uint32_t col_read_       = 0;
    uint32_t col_qual_       = 0;
    uint32_t col_read_len_   = 0;
    uint32_t col_read_start_ = 0;
    uint32_t col_name_       = 0;

    // Row iteration
    int64_t  first_row_   = 0;
    uint64_t row_count_   = 0;
    int64_t  current_row_ = 0;
    int64_t  row_limit_   = 0;   // 0 = no limit (use row_count_)
};
