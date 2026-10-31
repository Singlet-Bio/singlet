// singlet-stage --verify: re-read the written .1fg2 and compare every read with the source .1fq (container, planes, CRC32C, sequence table,
// N lists, binned qualities, barcode result). Included by singlet_stage.cpp.
#pragma once
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <zstd.h>
#include <singlet/fq/reader.h>
#include <singlet/pileup/cb_correct.h>
#include "fmt2.h"

namespace stage_verify {
using namespace fg2;
namespace sq = singlet::fq;

static bool inflate_plane(const uint8_t* base, const SecDesc& d, bool raw, std::vector<uint8_t>& out) {
    if (d.crc32c != fg2::crc32c(base + d.offset, d.comp_size)) return false;
    out.resize(d.raw_size);
    if (raw) { if (d.comp_size != d.raw_size) return false; std::memcpy(out.data(), base + d.offset, d.raw_size); return true; }
    if (d.raw_size == 0) return true;
    size_t r = ZSTD_decompress(out.data(), out.size(), base + d.offset, d.comp_size);
    return !ZSTD_isError(r) && r == d.raw_size;
}

static int verify_file(const std::string& path, const std::vector<std::string>& inputs, const singlet::CbCorrector& corr, int bcl, int bc0,
                       int umil, int umi0, bool has_q) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    size_t sz = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint8_t> file(sz); f.read((char*)file.data(), (std::streamsize)sz);
    FileHeader H; std::memcpy(&H, file.data(), sizeof H);
    if (H.magic != FILE_MAGIC || H.version != VERSION || H.header_crc32c != fg2::crc32c(&H, offsetof(FileHeader, header_crc32c))) {
        std::fprintf(stderr, "verify: bad header\n"); return 7;
    }
    Footer F; std::memcpy(&F, file.data() + sz - sizeof F, sizeof F);
    if (F.magic != FOOT_MAGIC) { std::fprintf(stderr, "verify: bad footer\n"); return 7; }
    std::vector<IndexEntry> idx(F.n_chunks + F.n_blocks);
    std::memcpy(idx.data(), file.data() + F.index_offset, idx.size() * sizeof(IndexEntry));
    if (F.index_crc32c != fg2::crc32c(idx.data(), idx.size() * sizeof(IndexEntry))) { std::fprintf(stderr, "verify: bad index crc\n"); return 7; }
    const uint32_t slot = H.seq_slot;
    std::vector<uint8_t> seq2, slen, hasn;
    std::vector<std::array<uint8_t, 32>> nms;
    for (uint32_t b = 0; b < F.n_blocks; b++) {
        const uint8_t* base = file.data() + idx[F.n_chunks + b].offset;
        BlockHeader bh; std::memcpy(&bh, base, sizeof bh);
        std::vector<uint8_t> a, l, e;
        if (bh.magic != SEQB_MAGIC || !inflate_plane(base, bh.sec[SP_SEQ2], bh.flags & 1, a) || !inflate_plane(base, bh.sec[SP_SLEN], (bh.flags >> 1) & 1, l) ||
            !inflate_plane(base, bh.sec[SP_SEXC], (bh.flags >> 2) & 1, e)) {
            std::fprintf(stderr, "verify: bad sequence block %u\n", b); return 7;
        }
        seq2.insert(seq2.end(), a.begin(), a.end()); slen.insert(slen.end(), l.begin(), l.end());
        size_t at = hasn.size(); hasn.resize(at + bh.n_seqs, 0); nms.resize(at + bh.n_seqs, std::array<uint8_t, 32>{});
        for (size_t k = 0; k < e.size() / 4; k++) {
            uint32_t v; std::memcpy(&v, &e[4 * k], 4);
            uint32_t i = v & 0x7FFF, pos = v >> 15;
            hasn[at + i] = 1; nms[at + i][pos >> 3] |= (uint8_t)(1u << (pos & 7));
        }
    }
    uint64_t bad = 0, nread = 0;
    uint32_t ci = 0, in_chunk = 0;
    std::vector<uint8_t> pcb, pst, pumi, pseq, plen, pq, pexc;
    std::vector<uint32_t> excv; ChunkHeader ch{};
    auto load_chunk = [&](uint32_t c) -> bool {
        const uint8_t* base = file.data() + idx[c].offset;
        std::memcpy(&ch, base, sizeof ch);
        if (ch.magic != CHUNK_MAGIC) return false;
        bool ok = inflate_plane(base, ch.sec[PL_CB], ch.flags & 1, pcb) && inflate_plane(base, ch.sec[PL_CBST], (ch.flags >> 1) & 1, pst) &&
                  inflate_plane(base, ch.sec[PL_UMI], (ch.flags >> 2) & 1, pumi) && inflate_plane(base, ch.sec[PL_SEQ], (ch.flags >> 3) & 1, pseq) &&
                  inflate_plane(base, ch.sec[PL_LEN], (ch.flags >> 4) & 1, plen) && inflate_plane(base, ch.sec[PL_Q], (ch.flags >> 5) & 1, pq) &&
                  inflate_plane(base, ch.sec[PL_EXC], (ch.flags >> 6) & 1, pexc);
        excv.resize(pexc.size() / 4);
        if (!excv.empty()) std::memcpy(excv.data(), pexc.data(), pexc.size());
        return ok;
    };
    if (F.n_chunks && !load_chunk(0)) { std::fprintf(stderr, "verify: bad chunk 0\n"); return 7; }
    std::unordered_map<uint64_t, std::pair<int64_t, uint8_t>> cbres;
    for (const auto& in : inputs) {
        sq::Reader rd; rd.open(in.c_str()); sq::DecodedBlock blk;
        while (rd.read_block(blk)) {
            for (uint32_t i = 0; i < blk.n_reads; i++) {
                const uint16_t r1l = blk.r1_len(i), r2l = blk.r2_len(i);
                if (r1l < bc0 + bcl || r1l < umi0 + umil || r2l == 0) continue;
                while (in_chunk >= ch.n_reads) {
                    if (++ci >= F.n_chunks || !load_chunk(ci)) { std::fprintf(stderr, "verify: more source reads than stored\n"); return 7; }
                    in_chunk = 0;
                }
                const uint32_t k = in_chunk++; nread++;
                const uint8_t* r1 = blk.r1_seq(i); uint64_t cbkey = 0;
                for (int j = 0; j < bcl; j++) cbkey = (cbkey << 3) | std::min<uint8_t>(r1[bc0 + j], 4);
                uint32_t umi = 0; uint16_t umin = 0; bool homo = true;
                for (int j = 0; j < umil; j++) {
                    uint8_t c = r1[umi0 + j];
                    if (c > 3) { umin |= (uint16_t)(1u << j); c = 0; }
                    umi = (umi << 2) | c;
                    if (r1[umi0 + j] != r1[umi0]) homo = false;
                }
                auto it = cbres.find(cbkey);
                if (it == cbres.end()) {
                    char s[32], ob[32];
                    for (int j = 0; j < bcl; j++) s[j] = "ACGTN"[(cbkey >> (3 * (bcl - 1 - j))) & 7];
                    s[bcl] = 0;
                    int st = corr.resolve(s, nullptr, ob);
                    it = cbres.emplace(cbkey, std::make_pair(st >= 0 ? corr.find(std::string(ob, bcl).c_str()) : -1, (uint8_t)(st + 4))).first;
                }
                uint32_t cb, u, sid;
                std::memcpy(&cb, &pcb[4 * k], 4); std::memcpy(&u, &pumi[4 * k], 4); std::memcpy(&sid, &pseq[4 * k], 4);
                bool ok = cb == (it->second.first >= 0 ? (uint32_t)it->second.first : NO_CB) &&
                          pst[k] == (uint8_t)(it->second.second | ((umin != 0 || homo) ? CBST_BAD_UMI : 0)) &&
                          u == umi && plen[k] == r2l && sid < slen.size() && slen[sid] == r2l;
                if (ok) {
                    const uint8_t* r2 = blk.r2_seq(i); const uint8_t* s2 = &seq2[(size_t)sid * slot];
                    for (uint16_t j = 0; j < r2l && ok; j++) {
                        uint8_t c = r2[j], got = (s2[j >> 2] >> (6 - 2 * (j & 3))) & 3;
                        bool isn = c > 3, tn = hasn[sid] && ((nms[sid][j >> 3] >> (j & 7)) & 1);
                        ok = isn == tn && (isn ? got == 0 : got == c);
                    }
                    if (ok && has_q) {
                        const uint8_t* qp = blk.r2_quality(i);
                        for (uint16_t j = 0; j < r2l && ok; j++) {
                            uint8_t p = qp ? qp[j] : 0, b = p >= 30 ? 3 : p >= 20 ? 2 : p >= 10 ? 1 : 0;
                            ok = ((pq[(size_t)k * slot + (j >> 2)] >> (6 - 2 * (j & 3))) & 3) == b;
                        }
                    }
                    for (int j = 0; j < umil && ok; j++) {
                        bool inexc = false;
                        for (uint32_t e : excv) if (exc_idx(e) == k && exc_pos(e) == (uint32_t)j) { inexc = true; break; }
                        ok = inexc == (bool)((umin >> j) & 1);
                    }
                }
                if (!ok) { bad++; if (bad <= 3) std::fprintf(stderr, "verify: mismatch at stored read %llu\n", (unsigned long long)(nread - 1)); }
            }
        }
    }
    std::fprintf(stderr, "verify: %llu reads compared, %llu mismatches\n", (unsigned long long)nread, (unsigned long long)bad);
    return (bad || nread != H.n_reads) ? 7 : 0;
}
}  // namespace stage_verify
