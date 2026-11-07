#pragma once
// Concatenate several .1fq files of one sample (its sequencing runs) into one, by copying their compressed blocks.
//
// No read is decoded or re-encoded: the output holds the blocks of input 1, then input 2, ... in order, under one
// header (read and block counts summed), with input 1's metadata and a rebuilt block index. Decoding the output gives
// exactly the reads of the inputs, in input order, as the old pooling did (decode every run to FASTQ, concatenate,
// `singlet encode --no-dedup --no-trim --quality none`), at the cost of a sequential copy of the compressed bytes
// instead of minutes of decode + encode and tens of GB of RAM. Unlike that pooling, per-read side columns (poly(A)
// trim lengths, duplicate counts) and the header's protocol survive.
//
// The inputs must agree on everything that decides how a block is decoded: format version, streams, flags, codec,
// sequence encoding, quality mode, stream lengths and roles, extra streams, and the barcode dictionary (absent, or
// byte-identical). A difference in protocol / confidence or metadata is reported and input 1's is kept.
#include "singlet/fq/types.h"
#include "singlet/fq/reader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet::fq {

struct ConcatReport {
    uint64_t reads = 0;
    uint32_t blocks = 0;
    std::vector<std::string> warnings;
};

namespace detail {
struct InFile {
    std::FILE* fp = nullptr;
    Header hdr{};
    std::vector<uint8_t> dict;          // [u32 comp size][comp bytes], empty if none
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> comp_sizes;
    uint64_t blocks_end = 0;            // first byte after the last block (= metadata start)
    ~InFile() { if (fp) std::fclose(fp); }
};

inline void read_exact(std::FILE* fp, void* p, size_t n, const std::string& what) {
    if (n && std::fread(p, 1, n, fp) != n) throw std::runtime_error("concat: short read in " + what);
}

inline void open_in(const std::string& path, InFile& f) {
    f.fp = std::fopen(path.c_str(), "rb");
    if (!f.fp) throw std::runtime_error("concat: cannot open " + path);
    read_exact(f.fp, &f.hdr, sizeof(Header), path);
    if (!f.hdr.valid_magic()) throw std::runtime_error("concat: not a .1fq file: " + path);
    if ((f.hdr.flags & Flags::BC_DICT) && f.hdr.bc_dict_size > 0) {
        uint32_t cs = 0;
        read_exact(f.fp, &cs, 4, path);
        f.dict.resize(4 + static_cast<size_t>(cs));
        std::memcpy(f.dict.data(), &cs, 4);
        read_exact(f.fp, f.dict.data() + 4, cs, path);
    }
    // Block index: [n x u64 offsets][n x u32 sizes][footer], located from EOF (see Reader::read_block_index)
    const uint32_t n = f.hdr.block_count;
    if (std::fseek(f.fp, 0, SEEK_END) != 0) throw std::runtime_error("concat: seek " + path);
    const int64_t size = static_cast<int64_t>(std::ftell(f.fp));
    const int64_t idx = size - static_cast<int64_t>(sizeof(Footer)) - static_cast<int64_t>(n) * 12;
    if (idx < static_cast<int64_t>(sizeof(Header))) throw std::runtime_error("concat: corrupt index in " + path);
    std::fseek(f.fp, static_cast<long>(idx), SEEK_SET);
    f.offsets.resize(n);
    f.comp_sizes.resize(n);
    read_exact(f.fp, f.offsets.data(), n * sizeof(uint64_t), path);
    read_exact(f.fp, f.comp_sizes.data(), n * sizeof(uint32_t), path);
    Footer ftr{};
    read_exact(f.fp, &ftr, sizeof(Footer), path);
    if (!ftr.valid_magic() || ftr.block_count != n) throw std::runtime_error("concat: bad footer in " + path);
    if (n == 0) {
        f.blocks_end = sizeof(Header) + f.dict.size();
    } else {
        BlockHeader bh{};
        std::fseek(f.fp, static_cast<long>(f.offsets.back()), SEEK_SET);
        read_exact(f.fp, &bh, sizeof(bh), path);
        f.blocks_end = f.offsets.back() + sizeof(BlockHeader) + bh.compressed_size;
    }
}

inline void copy_range(std::FILE* in, uint64_t off, uint64_t n, std::FILE* out, std::vector<char>& buf,
                       uint32_t& crc) {
    std::fseek(in, static_cast<long>(off), SEEK_SET);
    while (n > 0) {
        const size_t k = static_cast<size_t>(std::min<uint64_t>(n, buf.size()));
        if (std::fread(buf.data(), 1, k, in) != k) throw std::runtime_error("concat: short read while copying");
        if (std::fwrite(buf.data(), 1, k, out) != k) throw std::runtime_error("concat: write failed (disk full?)");
        crc = update_crc32(crc, reinterpret_cast<const uint8_t*>(buf.data()), k);
        n -= k;
    }
}
}  // namespace detail

inline ConcatReport concat_1fq(const std::vector<std::string>& inputs, const std::string& out_path) {
    using namespace detail;
    if (inputs.empty()) throw std::runtime_error("concat: no inputs");
    std::vector<InFile> in(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) open_in(inputs[i], in[i]);

    ConcatReport rep;
    const Header& h0 = in[0].hdr;
    for (size_t i = 1; i < in.size(); ++i) {
        const Header& h = in[i].hdr;
        auto bad = [&](const char* what) {
            throw std::runtime_error(std::string("concat: ") + inputs[i] + " differs from " + inputs[0] + " in " +
                                     what + "; these runs cannot be read as one sample stream");
        };
        if (h.version != h0.version) bad("format version");
        if (h.n_streams != h0.n_streams) bad("number of streams");
        if (h.flags != h0.flags) bad("flags (trimming / dedup / barcode dictionary)");
        if (h.codec != h0.codec) bad("codec");
        if (h.seq_encoding != h0.seq_encoding) bad("sequence encoding");
        if (h.qual_mode != h0.qual_mode) bad("quality mode");
        if (std::memcmp(h.stream_lengths, h0.stream_lengths, sizeof(h.stream_lengths)) != 0) bad("read lengths");
        if (std::memcmp(h.stream_roles, h0.stream_roles, sizeof(h.stream_roles)) != 0) bad("stream roles");
        if (h.reserved[0] != h0.reserved[0]) bad("extra streams");
        if (in[i].dict != in[0].dict) bad("barcode dictionary");
        if (h.protocol_id != h0.protocol_id || h.confidence != h0.confidence)
            rep.warnings.push_back(inputs[i] + ": protocol id " + std::to_string(h.protocol_id) + " (confidence " +
                                   std::to_string(h.confidence) + ") differs from the first run's " +
                                   std::to_string(h0.protocol_id) + " (" + std::to_string(h0.confidence) +
                                   "); the first run's is used");
        if (h.assay_type != h0.assay_type)
            rep.warnings.push_back(inputs[i] + ": assay type differs from the first run's; the first run's is used");
    }

    std::FILE* out = std::fopen(out_path.c_str(), "wb");
    if (!out) throw std::runtime_error("concat: cannot create " + out_path);
    std::vector<char> buf(8u << 20);
    uint32_t crc = 0;
    try {
        Header oh = h0;
        uint64_t n_unique = 0, n_original = 0;
        uint32_t n_blocks = 0;
        for (auto& f : in) {
            n_unique += f.hdr.n_unique;
            n_original += f.hdr.n_original;
            n_blocks += f.hdr.block_count;
        }
        oh.n_unique = n_unique;
        oh.n_original = n_original;
        oh.block_count = n_blocks;
        // metadata of input 1 (meta_size unchanged)
        if (std::fwrite(&oh, sizeof(Header), 1, out) != 1) throw std::runtime_error("concat: write failed");
        crc = update_crc32(crc, reinterpret_cast<const uint8_t*>(&oh), sizeof(Header));
        if (!in[0].dict.empty()) {
            if (std::fwrite(in[0].dict.data(), 1, in[0].dict.size(), out) != in[0].dict.size())
                throw std::runtime_error("concat: write failed");
            crc = update_crc32(crc, in[0].dict.data(), in[0].dict.size());
        }
        std::vector<uint64_t> offsets;
        std::vector<uint32_t> sizes;
        offsets.reserve(n_blocks);
        sizes.reserve(n_blocks);
        uint64_t pos = sizeof(Header) + in[0].dict.size();
        for (auto& f : in) {
            if (f.hdr.block_count == 0) continue;
            const uint64_t start = f.offsets.front();
            for (uint32_t b = 0; b < f.hdr.block_count; ++b) {
                offsets.push_back(f.offsets[b] - start + pos);
                sizes.push_back(f.comp_sizes[b]);
            }
            copy_range(f.fp, start, f.blocks_end - start, out, buf, crc);
            pos += f.blocks_end - start;
        }
        if (h0.meta_size > 0) {
            copy_range(in[0].fp, in[0].blocks_end, h0.meta_size, out, buf, crc);
            pos += h0.meta_size;
            if (in.size() > 1) {
                std::string m0;
                {
                    Reader r;
                    r.open(inputs[0].c_str());
                    m0 = r.read_metadata();
                }
                for (size_t i = 1; i < in.size(); ++i) {
                    Reader r;
                    r.open(inputs[i].c_str());
                    if (r.read_metadata() != m0)
                        rep.warnings.push_back(inputs[i] + ": metadata differs from the first run's; the first "
                                               "run's is kept");
                }
            }
        }
        const uint64_t index_start = pos;
        if (!offsets.empty()) {
            std::fwrite(offsets.data(), sizeof(uint64_t), offsets.size(), out);
            std::fwrite(sizes.data(), sizeof(uint32_t), sizes.size(), out);
            crc = update_crc32(crc, reinterpret_cast<const uint8_t*>(offsets.data()), offsets.size() * 8);
            crc = update_crc32(crc, reinterpret_cast<const uint8_t*>(sizes.data()), sizes.size() * 4);
        }
        Footer ftr;
        ftr.init(n_blocks, index_start);
        ftr.file_crc32 = crc;
        if (std::fwrite(&ftr, sizeof(Footer), 1, out) != 1) throw std::runtime_error("concat: write failed");
        if (std::fclose(out) != 0) {
            out = nullptr;
            throw std::runtime_error("concat: close failed (disk full?)");
        }
        out = nullptr;
        rep.reads = n_unique;
        rep.blocks = n_blocks;
    } catch (...) {
        if (out) std::fclose(out);
        std::remove(out_path.c_str());
        throw;
    }
    return rep;
}

}  // namespace singlet::fq
