// SPDX-License-Identifier: MIT
// fq/identify_bam.h — BGZF and BAM parsing for the identify BAM head route
//
// A Cell Ranger BAM of 10x reads holds the cell barcode and UMI only in tags (CR/CY raw barcode and quality, UR/UY raw UMI and quality,
// CB/UB the corrected pair). `singlet identify` reads a few megabytes of such a file by HTTP range, so this parser works on memory buffers:
//   - bgzf_block         one BGZF block, inflated and checked against its CRC
//   - ingest_head        the header and the records of the first buffer (the file's first bytes; the last block may be cut)
//   - ingest_sample      a buffer from the middle or the end of the file: resynchronize on a BGZF block, then on a record, then parse
// Nothing here fetches or writes files; the network side lives in identify_cli.h. No dependency except zlib and the standard library.
//
// Coordinate-sorted files (every Cell Ranger BAM): the head covers the first bases of the first chromosome and is not a random sample of
// cells. Evidence is therefore pooled over the head and spread samples, and each statistic is labeled with the samples it came from.

#pragma once

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace singlet::fq::identify::bam {

constexpr size_t kNpos = static_cast<size_t>(-1);

enum class Block { Ok, Truncated, Bad };

inline uint32_t rd32(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
inline int32_t rdi32(const uint8_t* p) { return static_cast<int32_t>(rd32(p)); }
inline uint16_t rd16(const uint8_t* p) { return uint16_t(p[0] | p[1] << 8); }

/// Inflate the BGZF block that starts at p. csize receives the block's size in the file (valid when Ok). Truncated: the buffer ends inside it.
inline Block bgzf_block(const uint8_t* p, size_t avail, size_t& csize, std::vector<uint8_t>& out) {
    if (avail >= 4 && (p[0] != 0x1f || p[1] != 0x8b || p[2] != 8 || (p[3] & 4) == 0)) return Block::Bad;
    if (avail < 12) return Block::Truncated;
    const size_t xlen = rd16(p + 10);
    if (12 + xlen > avail) return Block::Truncated;
    size_t bsize = 0;
    for (size_t o = 12; o + 4 <= 12 + xlen;) {
        const size_t slen = rd16(p + o + 2);
        if (p[o] == 'B' && p[o + 1] == 'C' && slen == 2 && o + 6 <= 12 + xlen) bsize = size_t(rd16(p + o + 4)) + 1;
        o += 4 + slen;
    }
    if (bsize < 12 + xlen + 8) return Block::Bad;
    if (bsize > avail) return Block::Truncated;
    const uint32_t isize = rd32(p + bsize - 4), crc = rd32(p + bsize - 8);
    if (isize > 65536) return Block::Bad;
    out.resize(isize);
    csize = bsize;
    if (isize == 0) return Block::Ok;
    z_stream zs;
    std::memset(&zs, 0, sizeof(zs));
    if (inflateInit2(&zs, -15) != Z_OK) return Block::Bad;
    zs.next_in = const_cast<Bytef*>(p + 12 + xlen);
    zs.avail_in = static_cast<uInt>(bsize - 12 - xlen - 8);
    zs.next_out = out.data();
    zs.avail_out = isize;
    const int rc = inflate(&zs, Z_FINISH);
    const bool ok = rc == Z_STREAM_END && zs.total_out == isize;
    inflateEnd(&zs);
    if (!ok || crc32(0, out.data(), isize) != crc) return Block::Bad;
    return Block::Ok;
}

/// First BGZF block at or after `from` that inflates, passes its CRC and is followed by another block (or the buffer's end).
inline size_t find_block(const uint8_t* d, size_t n, size_t from) {
    std::vector<uint8_t> tmp;
    for (size_t i = from; i + 18 <= n; ++i) {
        if (d[i] != 0x1f || d[i + 1] != 0x8b || d[i + 2] != 8 || d[i + 3] != 4) continue;
        size_t cs = 0;
        if (bgzf_block(d + i, n - i, cs, tmp) != Block::Ok) continue;
        const size_t nx = i + cs;
        if (nx == n || nx + 4 > n) return i;
        if (d[nx] == 0x1f && d[nx + 1] == 0x8b && d[nx + 2] == 8 && d[nx + 3] == 4) return i;
    }
    return kNpos;
}

/// Inflate consecutive blocks from `start`; stops at the first truncated or bad block. Returns the blocks decoded.
inline size_t inflate_run(const uint8_t* d, size_t n, size_t start, std::vector<uint8_t>& out, size_t* end = nullptr) {
    size_t i = start, blocks = 0;
    std::vector<uint8_t> blk;
    while (i < n) {
        size_t cs = 0;
        if (bgzf_block(d + i, n - i, cs, blk) != Block::Ok) break;
        out.insert(out.end(), blk.begin(), blk.end());
        i += cs;
        ++blocks;
    }
    if (end) *end = i;
    return blocks;
}

/// The 28-byte BGZF end-of-file marker.
inline bool has_eof_marker(const uint8_t* d, size_t n) {
    static const uint8_t m[28] = {0x1f, 0x8b, 0x08, 0x04, 0, 0, 0, 0, 0, 0xff, 0x06, 0, 0x42, 0x43, 0x02, 0, 0x1b, 0, 0x03, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    return n >= 28 && std::memcmp(d + n - 28, m, 28) == 0;
}

// ── records ──

constexpr uint16_t tkey(char a, char b) { return uint16_t(uint8_t(a) | uint8_t(b) << 8); }
inline std::string tname(uint16_t k) { return std::string{char(k & 255), char(k >> 8)}; }

struct Rec {
    int32_t ref = -1, pos = -1;
    uint16_t flag = 0;
    uint8_t mapq = 0;
    std::string seq;                                  // as stored: reference orientation (reverse complemented when flag 16)
    std::string cr, cy, ur, uy, cb, ub, cq, uq, tx, an, re;
    bool gx = false;                                  // gene tag present
    std::vector<uint16_t> tags;                       // every tag name in the record
};

enum class Parse { Ok, Incomplete, Bad };

/// Walk the tag area; fills the selected tags of r when given. False when a tag is malformed or the area does not end exactly at len.
inline bool walk_tags(const uint8_t* t, size_t len, Rec* r) {
    size_t i = 0;
    while (i < len) {
        if (i + 3 > len) return false;
        const uint16_t key = tkey(char(t[i]), char(t[i + 1]));
        const char ty = char(t[i + 2]);
        i += 3;
        size_t sz = 0;
        bool str = false;
        switch (ty) {
            case 'A': case 'c': case 'C': sz = 1; break;
            case 's': case 'S': sz = 2; break;
            case 'i': case 'I': case 'f': sz = 4; break;
            case 'Z': case 'H': str = true; break;
            case 'B': {
                if (i + 5 > len) return false;
                const char st = char(t[i]);
                size_t es = (st == 'c' || st == 'C') ? 1 : (st == 's' || st == 'S') ? 2 : (st == 'i' || st == 'I' || st == 'f') ? 4 : 0;
                if (!es) return false;
                const uint32_t cnt = rd32(t + i + 1);
                if (cnt > len) return false;
                sz = 5 + size_t(cnt) * es;
                break;
            }
            default: return false;
        }
        std::string val;
        if (str) {
            size_t j = i;
            while (j < len && t[j] != 0) ++j;
            if (j >= len) return false;
            if (r) val.assign(reinterpret_cast<const char*>(t + i), std::min<size_t>(j - i, 240));
            i = j + 1;
        } else {
            if (i + sz > len) return false;
            i += sz;
        }
        if (!r) continue;
        r->tags.push_back(key);
        if (ty == 'Z') {
            if (key == tkey('C', 'R')) r->cr = val;
            else if (key == tkey('C', 'Y')) r->cy = val;
            else if (key == tkey('U', 'R')) r->ur = val;
            else if (key == tkey('U', 'Y')) r->uy = val;
            else if (key == tkey('C', 'B')) r->cb = val;
            else if (key == tkey('U', 'B')) r->ub = val;
            else if (key == tkey('C', 'Q')) r->cq = val;
            else if (key == tkey('U', 'Q')) r->uq = val;
            else if (key == tkey('T', 'X')) r->tx = val;
            else if (key == tkey('A', 'N')) r->an = val;
        } else if (ty == 'A' && key == tkey('R', 'E')) r->re = std::string(1, char(t[i - 1]));
        if (key == tkey('G', 'X')) r->gx = true;
    }
    return i == len;
}

/// One record at p. total receives 4 + block_size. With r null the record is only validated (every field, the name and the tag area).
inline Parse parse_rec(const uint8_t* p, size_t avail, int32_t n_ref, Rec* r, size_t& total) {
    if (avail < 4) return Parse::Incomplete;
    const uint32_t bs = rd32(p);
    if (bs < 33 || bs > (1u << 24)) return Parse::Bad;
    total = 4 + size_t(bs);
    if (avail < 4 + 32) return Parse::Incomplete;
    const uint8_t* d = p + 4;
    const int32_t ref = rdi32(d), pos = rdi32(d + 4);
    const uint8_t lname = d[8], mapq = d[9];
    const uint16_t ncig = rd16(d + 12), flag = rd16(d + 14);
    const int32_t lseq = rdi32(d + 16), nref = rdi32(d + 20);
    if (ref < -1 || ref >= n_ref || nref < -1 || nref >= n_ref) return Parse::Bad;
    if (pos < -1 || lname < 2 || lseq < 0 || lseq > (1 << 20)) return Parse::Bad;
    const size_t fixed = 32 + size_t(lname) + 4 * size_t(ncig) + (size_t(lseq) + 1) / 2 + size_t(lseq);
    if (fixed > bs) return Parse::Bad;
    if (total > avail) return Parse::Incomplete;   // plausible, the rest lies beyond the buffer
    const uint8_t* nm = d + 32;
    if (nm[lname - 1] != 0) return Parse::Bad;
    for (size_t k = 0; k + 1 < lname; ++k)
        if (nm[k] < 33 || nm[k] > 126) return Parse::Bad;
    const uint8_t* sq = nm + lname + 4 * size_t(ncig);
    const uint8_t* ta = sq + (size_t(lseq) + 1) / 2 + size_t(lseq);
    if (r) {
        *r = Rec();
        r->ref = ref; r->pos = pos; r->flag = flag; r->mapq = mapq;
        static const char kNt[] = "=ACMGRSVTWYHKDBN";
        r->seq.resize(size_t(lseq));
        for (int32_t i = 0; i < lseq; ++i) r->seq[size_t(i)] = kNt[(i & 1) ? (sq[i / 2] & 15) : (sq[i / 2] >> 4)];
    }
    if (!walk_tags(ta, total - (size_t(ta - p)), r)) return Parse::Bad;
    return Parse::Ok;
}

inline std::string revcomp(const std::string& s) {
    std::string o(s.size(), 'N');
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[s.size() - 1 - i];
        o[i] = c == 'A' ? 'T' : c == 'C' ? 'G' : c == 'G' ? 'C' : c == 'T' ? 'A' : 'N';
    }
    return o;
}

// ── header ──

struct Header {
    bool ok = false;
    std::string text;
    std::vector<std::string> refs;
    size_t end = 0;   // offset of the first record in the inflated stream
};

inline bool parse_header(const std::vector<uint8_t>& d, Header& h) {
    if (d.size() < 12 || std::memcmp(d.data(), "BAM\1", 4) != 0) return false;
    const uint32_t lt = rd32(d.data() + 4);
    size_t i = 8;
    if (lt > (1u << 26) || i + lt + 4 > d.size()) return false;
    h.text.assign(reinterpret_cast<const char*>(d.data() + i), lt);
    while (!h.text.empty() && h.text.back() == 0) h.text.pop_back();
    i += lt;
    const uint32_t nr = rd32(d.data() + i);
    i += 4;
    if (nr > (1u << 22)) return false;
    for (uint32_t k = 0; k < nr; ++k) {
        if (i + 4 > d.size()) return false;
        const uint32_t ln = rd32(d.data() + i);
        if (ln > 4096 || i + 4 + ln + 4 > d.size()) return false;
        std::string nm(reinterpret_cast<const char*>(d.data() + i + 4), ln);
        while (!nm.empty() && nm.back() == 0) nm.pop_back();
        h.refs.push_back(nm);
        i += 4 + ln + 4;
    }
    h.end = i;
    h.ok = true;
    return true;
}

// ── statistics ──

struct Counts {
    uint64_t records = 0, mapped = 0, unmapped = 0, no_ref = 0, reverse = 0, secondary = 0, supplementary = 0;
    uint64_t cr = 0, cy = 0, ur = 0, uy = 0, cb = 0, ub = 0, cq = 0, uq = 0, gx = 0, tx = 0, an = 0, re = 0;
    uint64_t sense = 0, anti = 0;                        // from TX (read strand against the transcript) and AN (antisense gene alignment)
    uint64_t ordered_pairs = 0, out_of_order = 0;        // consecutive primary records of one reference with pos not decreasing
    std::map<uint16_t, uint64_t> tags;
    std::map<size_t, uint64_t> cr_len, ur_len, cb_len, ub_len;
    int32_t first_ref = -2, first_pos = -1, last_ref = -2, last_pos = -1;
    int32_t min_ref = INT32_MAX, max_ref = -1;

    void add(const Rec& r) {
        ++records;
        if (r.flag & 4) ++unmapped; else ++mapped;
        if (r.ref < 0) ++no_ref;
        if (r.flag & 16) ++reverse;
        for (uint16_t t : r.tags) ++tags[t];
        if (!r.cr.empty()) { ++cr; ++cr_len[r.cr.size()]; }
        if (!r.cy.empty()) ++cy;
        if (!r.ur.empty()) { ++ur; ++ur_len[r.ur.size()]; }
        if (!r.uy.empty()) ++uy;
        if (!r.cb.empty()) { ++cb; ++cb_len[r.cb.size()]; }
        if (!r.ub.empty()) { ++ub; ++ub_len[r.ub.size()]; }
        if (!r.cq.empty()) ++cq;
        if (!r.uq.empty()) ++uq;
        if (r.gx) ++gx;
        if (!r.re.empty()) ++re;
        // TX:Z:transcript,<strand><pos>,CIGAR ; the strand is the read's against the transcript (+ sense). AN lists antisense transcripts.
        if (!r.tx.empty()) {
            ++tx;
            const size_t c = r.tx.find(',');
            if (c != std::string::npos && c + 1 < r.tx.size()) { if (r.tx[c + 1] == '+') ++sense; else if (r.tx[c + 1] == '-') ++anti; }
        } else if (!r.an.empty()) {
            ++anti;
        }
        if (!r.an.empty()) ++an;
        if (r.ref >= 0) { min_ref = std::min(min_ref, r.ref); max_ref = std::max(max_ref, r.ref); }
        if (first_ref == -2) { first_ref = r.ref; first_pos = r.pos; }
        else if (r.ref == last_ref) { if (r.pos >= last_pos) ++ordered_pairs; else ++out_of_order; }
        else if (r.ref > last_ref || r.ref < 0) ++ordered_pairs;   // moving on to the next reference, or to the unmapped tail
        else ++out_of_order;
        last_ref = r.ref; last_pos = r.pos;
    }
};

/// One place in the file that was read.
struct Sample {
    std::string where;          // head | interior | tail
    uint64_t offset = 0, bytes = 0, blocks = 0;
    bool synced = false;        // a record start was found (always for the head)
    bool eof_marker = false;    // tail only
    Counts c;
};

struct Evidence {
    bool present = false;
    std::string url, error;
    int64_t file_size = -1;
    uint64_t bytes = 0;         // compressed bytes fetched, over all requests
    double seconds = 0;
    int requests = 0;
    // header
    bool header_ok = false;
    std::string sort_order;                    // coordinate | queryname | unsorted | unknown
    std::vector<std::string> refs_head;        // the first reference names
    uint32_t n_ref = 0;
    std::vector<std::string> program_lines;    // @PG lines, trimmed
    std::vector<std::string> comment_lines;    // @CO lines, trimmed
    std::string cellranger_version, chemistry, reference;
    // samples, pooled counts
    std::vector<Sample> samples;
    Counts pooled;
    // derived
    std::string barcode_tags;                  // "CR+UR" | "CB+UB" | "CR" | "CB" | "none"
    uint64_t barcode_records = 0;
    // whitelist rates of the barcode tag against each list, filled by the caller (needs the lists): file, family, exact, one mismatch
    struct ListRate { std::string file, family; double exact = 0, one_mm = 0; };
    std::vector<ListRate> list_rates;
    std::string list_rate_tag;                 // which tag the rates were computed on (CR or CB)
    double sense = -1;                         // sense share over reads with TX or AN, -1 without them
    uint64_t sense_reads = 0;
    std::string strand_basis;
    bool coordinate_sorted = false;
    std::string unmapped_status;               // present | not_seen | unknown
    std::string unmapped_note;
    std::vector<std::string> limits;           // what coordinate sorting does to each statistic
    bool usable_barcodes = false;              // tags present on at least half of the primary records
};

inline std::string lstrip(std::string s) { while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ')) s.pop_back(); return s; }

/// Reads @HD, @PG and @CO lines for the sort order, program, chemistry and reference.
inline void read_header_text(const Header& h, Evidence& ev) {
    ev.header_ok = true;
    ev.n_ref = static_cast<uint32_t>(h.refs.size());
    for (size_t i = 0; i < h.refs.size() && i < 3; ++i) ev.refs_head.push_back(h.refs[i]);
    ev.sort_order = "unknown";
    size_t i = 0;
    while (i < h.text.size()) {
        size_t j = h.text.find('\n', i);
        if (j == std::string::npos) j = h.text.size();
        const std::string line = lstrip(h.text.substr(i, j - i));
        i = j + 1;
        if (line.rfind("@HD", 0) == 0) {
            const size_t so = line.find("SO:");
            if (so != std::string::npos) ev.sort_order = line.substr(so + 3, line.find('\t', so) == std::string::npos ? std::string::npos : line.find('\t', so) - so - 3);
        } else if (line.rfind("@PG", 0) == 0) {
            if (ev.program_lines.size() < 8) ev.program_lines.push_back(line.substr(0, 400));
            // ID:cellranger-7.1.0 or PN:cellranger VN:7.1.0
            const size_t id = line.find("ID:");
            if (id != std::string::npos) {
                const std::string v = line.substr(id + 3, line.find('\t', id) == std::string::npos ? std::string::npos : line.find('\t', id) - id - 3);
                if (v.find("cellranger") != std::string::npos || v.find("spaceranger") != std::string::npos) ev.cellranger_version = v;
            }
            if (ev.cellranger_version.empty() && line.find("PN:cellranger") != std::string::npos) {
                const size_t vn = line.find("VN:");
                if (vn != std::string::npos) ev.cellranger_version = "cellranger-" + line.substr(vn + 3, line.find('\t', vn) == std::string::npos ? std::string::npos : line.find('\t', vn) - vn - 3);
            }
        } else if (line.rfind("@CO", 0) == 0) {
            if (ev.comment_lines.size() < 8) ev.comment_lines.push_back(line.substr(0, 400));
        }
        const size_t ch = line.find("--chemistry");
        if (ch != std::string::npos && ev.chemistry.empty()) {
            size_t b = ch + 11;
            while (b < line.size() && (line[b] == '=' || line[b] == ' ')) ++b;
            size_t e = b;
            while (e < line.size() && line[e] != ' ' && line[e] != '\t') ++e;
            ev.chemistry = line.substr(b, e - b);
        }
        for (const char* key : {"SC3Pv", "SC5P", "SC3Pv1", "SPATIAL", "threeprime", "fiveprime"})
            if (ev.chemistry.empty() && line.find(key) != std::string::npos) {
                const size_t b = line.find(key);
                size_t e = b;
                while (e < line.size() && line[e] != ' ' && line[e] != '\t' && line[e] != ',' && line[e] != '"') ++e;
                ev.chemistry = line.substr(b, e - b);
            }
        const size_t gd = line.find("--genomeDir");
        if (gd != std::string::npos && ev.reference.empty()) {
            size_t b = gd + 11;
            while (b < line.size() && line[b] == ' ') ++b;
            size_t e = b;
            while (e < line.size() && line[e] != ' ' && line[e] != '\t') ++e;
            ev.reference = line.substr(b, e - b);
        }
        const size_t tr = line.find("--transcriptome");
        if (tr != std::string::npos && ev.reference.empty()) {
            size_t b = tr + 15;
            while (b < line.size() && (line[b] == '=' || line[b] == ' ')) ++b;
            size_t e = b;
            while (e < line.size() && line[e] != ' ' && line[e] != '\t') ++e;
            ev.reference = line.substr(b, e - b);
        }
    }
}

/// The header and the records of the buffer that holds the file's first bytes. A truncated last block or record is expected.
inline bool ingest_head(Evidence& ev, const uint8_t* data, size_t n, std::vector<Rec>& recs) {
    Sample s;
    s.where = "head";
    s.bytes = n;
    s.synced = true;
    s.eof_marker = has_eof_marker(data, n);   // a small file lies wholly in the head
    std::vector<uint8_t> d;
    s.blocks = inflate_run(data, n, 0, d);
    Header h;
    if (!s.blocks || !parse_header(d, h)) {
        ev.error = s.blocks ? "the first block does not hold a BAM header" : "not a BGZF file (no valid first block)";
        return false;
    }
    read_header_text(h, ev);
    size_t i = h.end, total = 0;
    Rec r;
    const int32_t nr = static_cast<int32_t>(h.refs.size());
    for (;;) {
        const Parse p = parse_rec(d.data() + i, d.size() - i, nr, &r, total);
        if (p != Parse::Ok) break;
        i += total;
        if (r.flag & 0x900) continue;   // secondary and supplementary alignments repeat a read
        s.c.add(r);
        recs.push_back(r);
    }
    ev.n_ref = static_cast<uint32_t>(h.refs.size());
    ev.samples.push_back(std::move(s));
    return true;
}

/// A buffer from inside the file: find a BGZF block, then a run of three valid records, then parse to the end.
inline void ingest_sample(Evidence& ev, const std::string& where, uint64_t offset, const uint8_t* data, size_t n, std::vector<Rec>& recs) {
    Sample s;
    s.where = where;
    s.offset = offset;
    s.bytes = n;
    s.eof_marker = has_eof_marker(data, n);
    const size_t b = find_block(data, n, 0);
    if (b != kNpos) {
        std::vector<uint8_t> d;
        s.blocks = inflate_run(data, n, b, d);
        const int32_t nr = static_cast<int32_t>(ev.n_ref);
        size_t start = kNpos, total = 0;
        for (size_t i = 0; i + 40 < d.size() && i < 70000 && start == kNpos; ++i) {
            size_t j = i;
            int good = 0;
            while (good < 3) {
                const Parse p = parse_rec(d.data() + j, d.size() - j, nr, nullptr, total);
                if (p == Parse::Ok) { ++good; j += total; continue; }
                if (p == Parse::Incomplete && good >= 1) good = 3;   // ran into the end of the buffer after a valid record
                break;
            }
            if (good >= 3) start = i;
        }
        if (start != kNpos) {
            s.synced = true;
            size_t i = start;
            Rec r;
            for (;;) {
                if (parse_rec(d.data() + i, d.size() - i, nr, &r, total) != Parse::Ok) break;
                i += total;
                if (r.flag & 0x900) continue;
                s.c.add(r);
                recs.push_back(r);
            }
        }
    }
    ev.samples.push_back(std::move(s));
}

inline std::string top_len(const std::map<size_t, uint64_t>& m) {
    size_t best = 0;
    uint64_t bc = 0;
    for (const auto& kv : m) if (kv.second > bc) { bc = kv.second; best = kv.first; }
    return bc ? std::to_string(best) : std::string();
}
inline size_t mode_len(const std::map<size_t, uint64_t>& m) {
    size_t best = 0;
    uint64_t bc = 0;
    for (const auto& kv : m) if (kv.second > bc) { bc = kv.second; best = kv.first; }
    return best;
}

/// Pool the samples and derive the tag set, the sort and unmapped calls and the statement of limits.
inline void finalize(Evidence& ev) {
    ev.present = true;
    Counts& p = ev.pooled;
    p = Counts();
    for (const auto& s : ev.samples) {
        const Counts& c = s.c;
        p.records += c.records; p.mapped += c.mapped; p.unmapped += c.unmapped; p.no_ref += c.no_ref; p.reverse += c.reverse;
        p.cr += c.cr; p.cy += c.cy; p.ur += c.ur; p.uy += c.uy; p.cb += c.cb; p.ub += c.ub; p.cq += c.cq; p.uq += c.uq;
        p.gx += c.gx; p.tx += c.tx; p.an += c.an; p.re += c.re; p.sense += c.sense; p.anti += c.anti;
        p.ordered_pairs += c.ordered_pairs; p.out_of_order += c.out_of_order;
        for (const auto& kv : c.tags) p.tags[kv.first] += kv.second;
        for (const auto& kv : c.cr_len) p.cr_len[kv.first] += kv.second;
        for (const auto& kv : c.ur_len) p.ur_len[kv.first] += kv.second;
        for (const auto& kv : c.cb_len) p.cb_len[kv.first] += kv.second;
        for (const auto& kv : c.ub_len) p.ub_len[kv.first] += kv.second;
    }
    const double N = p.records ? double(p.records) : 1;
    const bool cr = p.cr >= 0.5 * N, ur = p.ur >= 0.5 * N, cb = p.cb >= 0.5 * N, ub = p.ub >= 0.5 * N;
    if (cr && ur) ev.barcode_tags = "CR+UR";
    else if (cb && ub) ev.barcode_tags = "CB+UB";
    else if (cr) ev.barcode_tags = "CR";
    else if (cb) ev.barcode_tags = "CB";
    else ev.barcode_tags = "none";
    ev.usable_barcodes = ev.barcode_tags != "none";
    ev.barcode_records = (cr || ur) ? std::max(p.cr, p.ur) : std::max(p.cb, p.ub);
    if (p.sense + p.anti >= 200) {
        ev.sense = double(p.sense) / double(p.sense + p.anti);
        ev.sense_reads = p.sense + p.anti;
        ev.strand_basis = "TX and AN tags (read strand against the transcript, Cell Ranger annotation)";
    } else {
        ev.strand_basis = "no TX/AN tags: the strand against genes needs an annotation, so 3'/5' falls back to the motif and mapping evidence";
    }
    const Counts& h = ev.samples.empty() ? p : ev.samples[0].c;
    ev.coordinate_sorted = ev.sort_order == "coordinate" || (ev.sort_order == "unknown" && h.records > 1000 && h.out_of_order * 100 <= h.ordered_pairs);
    // unmapped reads: a sorted BAM puts them last, so only the tail sample can show them
    uint64_t un = 0, nrec = 0;
    bool tail_synced = false;
    for (const auto& s : ev.samples) { un += s.c.unmapped; nrec += s.c.records; if (s.synced && (s.where == "tail" || s.eof_marker)) tail_synced = true; }
    if (un > 0) {
        ev.unmapped_status = "present";
        ev.unmapped_note = std::to_string(un) + " of " + std::to_string(nrec) + " primary records in the samples carry flag 4";
    } else if (ev.coordinate_sorted && !tail_synced) {
        ev.unmapped_status = "unknown";
        ev.unmapped_note = "coordinate-sorted file: unmapped reads sit after the last chromosome and the tail sample could not be read";
    } else {
        ev.unmapped_status = "not_seen";
        ev.unmapped_note = "no record with flag 4 in " + std::to_string(nrec) + " primary records" + (ev.coordinate_sorted ? " including the tail of the file" : "");
    }
    ev.limits.clear();
    if (ev.coordinate_sorted) {
        ev.limits.push_back("The head of a coordinate-sorted BAM covers the first bases of the first chromosome, so it is not a random sample of cells or genes.");
        ev.limits.push_back("Used: tag presence and lengths, the barcode list rates of raw CR (a barcode's position in the genome does not depend on the read's), and the sense share from TX/AN tags, each pooled over the head and the spread samples.");
        ev.limits.push_back("Not used: reads per cell, number of cells, duplicate rate, MAPQ and multi-mapper shares, chrM and exonic shares: all depend on the genomic region read.");
    } else {
        ev.limits.push_back("The file is not coordinate sorted; the head is the first records in file order and is treated as a sample of the run.");
    }
}

}  // namespace singlet::fq::identify::bam
