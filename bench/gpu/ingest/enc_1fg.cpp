// enc_1fg: CPU-node transform  .1fq -> .1fg  (GPU-ready chunked container, no base qualities).
//   enc_1fg --in F.1fq --out F.1fg [--codec zstd|raw] [--level 1] [--threads 4] [--crc 1] [--max-blocks N] [--dontneed 1]
//           [--selftest K]   (compare the fast parser against the library decoder on the first K blocks and exit)
// Per source .1fq block: pread, zstd-decompress, parse straight into the GPU layout (CB+UMI word, R2 slot, exception list),
// cut into chunks of <= 262144 pairs, compress each column section (zstd frame) and write chunks in order.
// Output: one JSON line on stdout (timings, bytes, CPU seconds from getrusage).
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <memory>
#include <random>
#include <numeric>
#include <fcntl.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <zstd.h>
#include <singlet/fq/reader.h>
#include <singlet/fq/packing.h>
#include "ingest_fmt.h"

using namespace singlet::fq;
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double thr_cpu() { timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static bool pread_full(int fd, void* buf, size_t n, uint64_t off) {
    uint8_t* p = (uint8_t*)buf;
    while (n) { ssize_t r = pread(fd, p, n, off); if (r <= 0) return false; p += r; n -= r; off += r; }
    return true;
}
static inline uint64_t read_varint(const uint8_t*& p) {
    uint64_t v = 0; int sh = 0;
    for (;;) { uint8_t b = *p++; v |= (uint64_t)(b & 0x7f) << sh; if (!(b & 0x80)) break; sh += 7; if (sh >= 64) break; }
    return v;
}
static uint32_t pack_bases_u32(const uint8_t* b, uint32_t n) { uint32_t v = 0; for (uint32_t i = 0; i < n; i++) v = (v << 2) | (b[i] & 3); return v; }

struct Plan {
    Header hdr; uint32_t bc_len = 0, umi_len = 0, bc_start = 0, r1_len = 0, r2_fixed = 0, r2_slot = 24, r2_max = 0, dict_n = 0;
    std::vector<uint32_t> cbtab;
    struct Blk { uint64_t off; uint32_t csz, n; };
    std::vector<Blk> blocks;
    bool fast_ok = false; std::string why;
};

// Parsed block in the output layout (before chunking)
struct Parsed {
    uint32_t n = 0;
    std::vector<uint64_t> cu;       // CB | UMI<<32 | len<<56 | has_exc<<63
    std::vector<uint8_t> r2;        // n * slot
    std::vector<uint32_t> exc;      // (idx within block, kind, pos), sorted by idx
    uint64_t dropped = 0, unk_bc = 0, nN = 0;
};

static void sort_exc(std::vector<uint32_t>& ex) {
    std::stable_sort(ex.begin(), ex.end(), [](uint32_t a, uint32_t b) { return (a & 0xFFFFF) < (b & 0xFFFFF); });
}

// Fast path: parse the decompressed payload directly. Returns false if the block needs the generic path (zero-length or
// oversized R2 anywhere, inconsistent payload).
static bool parse_fast(const Plan& P, const uint8_t* raw, size_t rawsz, uint32_t n, Parsed& o) {
    const uint8_t* p = raw; const uint8_t* end = raw + rawsz;
    const uint32_t S = P.r2_slot, bcl = P.bc_len, uml = P.umi_len;
    o.n = n; o.cu.assign(n, 0); o.r2.assign((size_t)n * S, 0); o.exc.clear(); o.unk_bc = 0; o.nN = 0;
    std::vector<uint32_t>& ex = o.exc;
    // 1) barcode dictionary indices (sentinel = dict_n, followed by the raw barcode in byte-numeric form)
    for (uint32_t i = 0; i < n; i++) {
        uint64_t idx = read_varint(p);
        if (idx < P.dict_n) o.cu[i] = P.cbtab[idx];
        else {
            o.cu[i] = pack_bases_u32(p, bcl);
            ex.push_back(fg::exc_pack(i, fg::EXC_BC_UNKNOWN, 0));
            p += bcl; o.unk_bc++;
        }
        if (p > end) return false;
    }
    // 2) UMI column
    const uint32_t upb = (uml + 3) / 4, unb = (uml + 7) / 8, ush = (8 * upb - 2 * uml) & 31;
    for (uint32_t i = 0; i < n; i++) {
        uint8_t has_n = *p++;
        uint32_t v = 0; for (uint32_t k = 0; k < upb; k++) v = (v << 8) | p[k];
        v >>= ush; p += upb;
        o.cu[i] |= (uint64_t)v << 32;
        if (has_n) {
            uint8_t tmp[16] = {0}; pack::apply_n_bitmap(p, uml, tmp);
            for (uint32_t k = 0; k < uml; k++) if (tmp[k] == 4) { ex.push_back(fg::exc_pack(i, fg::EXC_UMIN, k)); o.nN++; }
            p += unb;
        }
    }
    // 3) R2 column
    const bool var = P.r2_fixed == 0;
    std::vector<uint8_t> tmp;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t len = P.r2_fixed;
        if (var) len = (uint32_t)read_varint(p);
        if (len == 0 || len > 1024 || len > S * 4 || len > 127) return false;
        uint8_t has_n = *p++;
        uint32_t pb = (len + 3) / 4;
        uint8_t* dst = o.r2.data() + (size_t)i * S;
        std::memcpy(dst, p, pb);
        if (len & 3) dst[pb - 1] &= (uint8_t)(0xFFu << (8 - 2 * (len & 3)));       // zero the padding bits of the last byte
        p += pb;
        if (has_n) {
            tmp.assign(len, 0); pack::apply_n_bitmap(p, len, tmp.data());
            for (uint32_t k = 0; k < len; k++) if (tmp[k] == 4) { ex.push_back(fg::exc_pack(i, fg::EXC_R2N, k)); o.nN++; }
            p += (len + 7) / 8;
        }
        o.cu[i] |= (uint64_t)len << 56;
        if (p > end) return false;
    }
    sort_exc(ex);
    for (uint32_t e : ex) o.cu[e & 0xFFFFF] |= 1ull << 63;
    return p <= end;
}

// Generic path through the library decoder (byte-numeric, N = 4). Drops reads like `singlet decode` does. Unknown barcodes
// come back as N (the raw barcode is not recoverable through the library).
static void parse_generic(const Plan& P, const DecodedBlock& b, Parsed& o) {
    const uint32_t S = P.r2_slot, bcl = P.bc_len, uml = P.umi_len;
    o.n = 0; o.cu.clear(); o.r2.clear(); o.exc.clear(); o.dropped = 0; o.unk_bc = 0; o.nN = 0;
    for (uint32_t i = 0; i < b.n_reads; i++) {
        uint32_t len = b.r2_len(i);
        if (b.r1_len(i) == 0 || len == 0 || len > 1024) { o.dropped++; continue; }
        if (len > S * 4 || len > 127) { fprintf(stderr, "R2 length %u exceeds the slot (%u nt)\n", len, S * 4); exit(3); }
        const uint8_t* r1 = b.r1_seq(i); const uint8_t* s = b.r2_seq(i);
        uint32_t idx = o.n++;
        uint64_t cb = 0, um = 0; bool unknown = false;
        for (uint32_t k = 0; k < bcl; k++) { uint8_t c = r1[P.bc_start + k]; cb = (cb << 2) | (c & 3); if (c > 3) unknown = true; }
        if (unknown) { o.exc.push_back(fg::exc_pack(idx, fg::EXC_BC_UNKNOWN, 0)); o.unk_bc++; cb = 0; }
        uint32_t u = 0;
        auto um_push = [&](uint8_t c) { um = (um << 2) | (c & 3); if (c > 3) { o.exc.push_back(fg::exc_pack(idx, fg::EXC_UMIN, u)); o.nN++; } u++; };
        for (uint32_t k = 0; k < P.bc_start && u < uml; k++) um_push(r1[k]);
        for (uint32_t k = P.bc_start + bcl; k < P.r1_len && u < uml; k++) um_push(r1[k]);
        o.r2.resize((size_t)o.n * S, 0);
        uint8_t* dst = o.r2.data() + (size_t)idx * S;
        for (uint32_t k = 0; k < len; k++) {
            uint8_t c = s[k];
            dst[k >> 2] |= (uint8_t)((c & 3) << (6 - 2 * (k & 3)));
            if (c > 3) { o.exc.push_back(fg::exc_pack(idx, fg::EXC_R2N, k)); o.nN++; }
        }
        o.cu.push_back(cb | (um << 32) | ((uint64_t)len << 56));
    }
    sort_exc(o.exc);
    for (uint32_t e : o.exc) o.cu[e & 0xFFFFF] |= 1ull << 63;
}

struct Cfg { std::string in, out; int codec = fg::CODEC_ZSTD, level = 1, threads = 4, crc = 1, dontneed = 1, selftest = 0; uint32_t max_blocks = 0, chunk_pairs = fg::CHUNK_PAIRS, shuffle = 0; bool raw_sec[3] = {false, false, false}; };

struct OutChunk { std::vector<uint8_t> bytes; uint32_t n_pairs = 0; uint32_t exc_raw = 0; };
struct BlockOut { std::vector<OutChunk> chunks; uint64_t dropped = 0, unk = 0, nN = 0, n_in = 0; bool generic = false; };

int main(int argc, char** argv) {
    Cfg C;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--in") C.in = nx(); else if (k == "--out") C.out = nx();
        else if (k == "--codec") C.codec = nx() == "raw" ? fg::CODEC_RAW : fg::CODEC_ZSTD;
        else if (k == "--level") C.level = atoi(nx().c_str()); else if (k == "--threads") C.threads = atoi(nx().c_str());
        else if (k == "--crc") C.crc = atoi(nx().c_str()); else if (k == "--max-blocks") C.max_blocks = atoi(nx().c_str());
        else if (k == "--shuffle") C.shuffle = atoi(nx().c_str());   // test only: random order of the pairs inside each source block (what an unsorted SRA stream would give)
        else if (k == "--chunk-pairs") C.chunk_pairs = atoi(nx().c_str());
        else if (k == "--raw-sections") { std::string v = nx(); if (v.find("cbumi") != std::string::npos) C.raw_sec[0] = true; if (v.find("r2") != std::string::npos) C.raw_sec[1] = true; if (v.find("exc") != std::string::npos) C.raw_sec[2] = true; }
        else if (k == "--dontneed") C.dontneed = atoi(nx().c_str()); else if (k == "--selftest") C.selftest = atoi(nx().c_str());
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    Reader rd; rd.open(C.in.c_str());
    int fd = open(C.in.c_str(), O_RDONLY);
    if (C.dontneed) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    struct stat st; fstat(fd, &st);
    Plan P; P.hdr = rd.header(); const Header& h = P.hdr;
    P.r1_len = h.stream_lengths[0]; P.r2_fixed = (h.flags & Flags::TRIMMED) ? 0 : h.stream_lengths[1];
    P.bc_len = rd.bc_length(); P.umi_len = rd.umi_length(); P.dict_n = rd.bc_dict_size(); P.bc_start = 0;
    for (const auto& s : known_protocols()) if (s.protocol_id == h.protocol_id) { P.bc_start = s.bc_offset; break; }
    std::string why;
    if (!rd.has_bc_dict()) why += "no BC dict; ";
    if (h.seq_encoding != (uint8_t)SeqEncoding::PACKED_2BIT) why += "seq not 2-bit; ";
    if (h.codec != (uint8_t)Codec::ZSTD) why += "codec not zstd; ";
    if (P.bc_len == 0 || P.bc_len > 16) why += "bc_len unsupported; ";
    if (P.umi_len > 12) why += "umi_len > 12; ";
    if (P.r1_len != P.bc_len + P.umi_len) why += "R1 has bases outside barcode+UMI; ";
    if (h.flags & Flags::DEDUPED) why += "deduped file (dup counts) unsupported; ";
    if (h.tier != 0) why += "tier-2 file; ";
    if (!why.empty()) { fprintf(stderr, "unsupported source: %s\n", why.c_str()); return 4; }
    P.fast_ok = true;
    { const auto& d = rd.bc_dict_flat(); P.cbtab.resize(P.dict_n);
      for (uint32_t i = 0; i < P.dict_n; i++) P.cbtab[i] = pack_bases_u32(d.data() + (size_t)i * P.bc_len, P.bc_len); }
    uint32_t nb = h.block_count; if (C.max_blocks && C.max_blocks < nb) nb = C.max_blocks;
    P.blocks.resize(nb);
    for (uint32_t b = 0; b < nb; b++) {
        BlockHeader bh; pread_full(fd, &bh, sizeof bh, rd.block_offset(b));
        P.blocks[b] = {rd.block_offset(b) + sizeof(BlockHeader), bh.compressed_size, bh.n_reads};
    }
    // R2 slot: fixed length from the header, else the maximum over the first 8 blocks (library decode)
    uint32_t maxlen = P.r2_fixed;
    if (maxlen == 0) {
        std::vector<uint8_t> comp, raw;
        for (uint32_t b = 0; b < std::min<uint32_t>(8, nb); b++) {
            comp.resize(P.blocks[b].csz); pread_full(fd, comp.data(), comp.size(), P.blocks[b].off);
            BlockHeader bh{P.blocks[b].n, P.blocks[b].csz, 0}; raw.resize((size_t)bh.n_reads * 400);
            DecodedBlock blk; size_t rs = compress::decompress_block(Codec::ZSTD, comp.data(), comp.size(), raw.data(), raw.size());
            rd.decode_payload(raw.data(), rs, bh, blk);
            for (auto l : blk.r2_lengths) if (l <= 1024) maxlen = std::max<uint32_t>(maxlen, l);
        }
    }
    P.r2_max = maxlen; P.r2_slot = std::max<uint32_t>(24, ((maxlen + 3) / 4 + 7) / 8 * 8);
    if (maxlen > 127) { fprintf(stderr, "R2 length %u > 127 not supported by the 7-bit length field\n", maxlen); return 3; }

    if (C.selftest) {   // fast parser vs generic parser (library decode), bit for bit
        uint32_t bad = 0, tested = 0; std::vector<uint8_t> comp, raw;
        for (uint32_t b = 0; b < std::min<uint32_t>(C.selftest, nb); b++) {
            comp.resize(P.blocks[b].csz + 64); pread_full(fd, comp.data(), P.blocks[b].csz, P.blocks[b].off);
            raw.assign((size_t)P.blocks[b].n * 400 + 64, 0);
            size_t rs = compress::decompress_block(Codec::ZSTD, comp.data(), P.blocks[b].csz, raw.data(), raw.size() - 64);
            BlockHeader bh{P.blocks[b].n, P.blocks[b].csz, 0}; DecodedBlock blk; rd.decode_payload(raw.data(), rs, bh, blk);
            Parsed a, g; bool ok = parse_fast(P, raw.data(), rs, P.blocks[b].n, a); parse_generic(P, blk, g);
            // the generic path cannot see the raw unknown barcode: compare the CB field only on known barcodes
            // (and drops the N positions inside a raw unknown barcode: the fast path keeps them as kind-2 entries)
            std::vector<uint32_t> aexc; for (uint32_t e : a.exc) aexc.push_back(e);
            bool same = ok && a.n == g.n && aexc == g.exc && a.r2 == g.r2;
            if (same) {
                std::vector<char> unk(a.n, 0);
                for (uint32_t e : a.exc) if (((e >> 20) & 3) == fg::EXC_BC_UNKNOWN) unk[e & 0xFFFFF] = 1;
                for (uint32_t i = 0; i < a.n; i++) {
                    uint64_t x = a.cu[i], y = g.cu[i];
                    if (unk[i]) { x &= ~0xFFFFFFFFull; y &= ~0xFFFFFFFFull; }
                    if (x != y) { same = false; break; }
                }
            }
            tested++; if (!same) { bad++; fprintf(stderr, "selftest block %u MISMATCH (ok=%d n %u/%u exc %zu/%zu)\n", b, (int)ok, a.n, g.n, a.exc.size(), g.exc.size()); }
        }
        printf("{\"selftest_blocks\":%u,\"mismatch_blocks\":%u}\n", tested, bad);
        return bad ? 1 : 0;
    }

    // ---------------------------------------------------------------- pipeline
    std::atomic<uint32_t> next{0}; uint32_t wnext = 0;   // wnext: guarded by mu
    std::mutex mu; std::condition_variable cv; std::map<uint32_t, BlockOut> done;
    std::atomic<double> t_decomp{0}, t_parse{0}, t_comp{0}, t_crc{0}; std::atomic<uint64_t> g_sec_c[3] = {{0}, {0}, {0}}, g_sec_r[3] = {{0}, {0}, {0}};
    const uint32_t R2B = P.r2_slot;
    auto worker = [&]() {
        ZSTD_DCtx* dctx = ZSTD_createDCtx(); ZSTD_CCtx* cctx = ZSTD_createCCtx();
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, C.level);
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, 0);
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_contentSizeFlag, 1);
        std::vector<uint8_t> comp; std::unique_ptr<uint8_t[]> rawp; size_t rawcap = 0; Parsed pb;
        double td = 0, tp = 0, tc = 0, tk = 0; uint64_t sec_c[3] = {0, 0, 0}, sec_r[3] = {0, 0, 0};
        for (;;) {
            uint32_t b = next.fetch_add(1); if (b >= nb) break;
            // bound the reorder buffer: do not run more than 4 x threads blocks ahead of the writer
            { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return b < wnext + (uint32_t)C.threads * 3; }); }
            double t0 = thr_cpu();
            comp.resize(P.blocks[b].csz + 64);
            if (!pread_full(fd, comp.data(), P.blocks[b].csz, P.blocks[b].off)) { fprintf(stderr, "read fail\n"); exit(1); }
            size_t need = (size_t)P.blocks[b].n * 400 + 64;      // uninitialised: only the touched part is paged in
            if (need > rawcap) { rawp.reset(new uint8_t[need]); rawcap = need; }
            size_t rs = ZSTD_decompressDCtx(dctx, rawp.get(), rawcap - 64, comp.data(), P.blocks[b].csz);
            if (ZSTD_isError(rs)) { fprintf(stderr, "zstd: %s\n", ZSTD_getErrorName(rs)); exit(1); }
            std::memset(rawp.get() + rs, 0, 64);
            double t1 = thr_cpu();
            BlockOut bo; bo.n_in = P.blocks[b].n;
            bool ok = parse_fast(P, rawp.get(), rs, P.blocks[b].n, pb);
            if (!ok) {
                bo.generic = true;
                BlockHeader bh{P.blocks[b].n, P.blocks[b].csz, 0}; DecodedBlock blk; rd.decode_payload(rawp.get(), rs, bh, blk);
                parse_generic(P, blk, pb);
            }
            bo.dropped = pb.dropped; bo.unk = pb.unk_bc; bo.nN = pb.nN;
            if (C.shuffle) {
                std::vector<uint32_t> perm(pb.n), inv(pb.n); std::iota(perm.begin(), perm.end(), 0u);
                std::mt19937 rng(C.shuffle * 7919u + b); std::shuffle(perm.begin(), perm.end(), rng);
                for (uint32_t i = 0; i < pb.n; i++) inv[perm[i]] = i;
                Parsed q; q.n = pb.n; q.cu.resize(pb.n); q.r2.resize((size_t)pb.n * R2B);
                for (uint32_t i = 0; i < pb.n; i++) { q.cu[i] = pb.cu[perm[i]]; std::memcpy(q.r2.data() + (size_t)i * R2B, pb.r2.data() + (size_t)perm[i] * R2B, R2B); }
                for (uint32_t e : pb.exc) q.exc.push_back((e & ~0xFFFFFu) | inv[e & 0xFFFFF]);
                sort_exc(q.exc); pb = std::move(q);
            }
            double t2 = thr_cpu();
            uint32_t pos = 0, ei = 0;
            while (pos < pb.n) {
                uint32_t cn = std::min<uint32_t>(C.chunk_pairs, pb.n - pos);
                std::vector<uint32_t> ex;
                while (ei < pb.exc.size() && (pb.exc[ei] & 0xFFFFF) < pos + cn) { uint32_t e = pb.exc[ei++]; ex.push_back((e & ~0xFFFFFu) | ((e & 0xFFFFF) - pos)); }
                const uint8_t* src[3] = {(const uint8_t*)(pb.cu.data() + pos), pb.r2.data() + (size_t)pos * R2B, (const uint8_t*)ex.data()};
                size_t rawsz[3] = {(size_t)cn * 8, (size_t)cn * R2B, ex.size() * 4};
                std::vector<uint8_t> secbuf[3]; size_t csz[3]; bool is_raw[3];
                double c0 = thr_cpu();
                for (int s = 0; s < 3; s++) {
                    is_raw[s] = C.codec == fg::CODEC_RAW || C.raw_sec[s] || rawsz[s] == 0;
                    if (!is_raw[s]) {
                        secbuf[s].resize(ZSTD_compressBound(rawsz[s]));
                        csz[s] = ZSTD_compress2(cctx, secbuf[s].data(), secbuf[s].size(), src[s], rawsz[s]);
                        if (ZSTD_isError(csz[s])) { fprintf(stderr, "zstd c: %s\n", ZSTD_getErrorName(csz[s])); exit(1); }
                        if (csz[s] >= rawsz[s]) is_raw[s] = true;       // incompressible: store raw
                    }
                    if (is_raw[s]) csz[s] = rawsz[s];
                    sec_c[s] += csz[s]; sec_r[s] += rawsz[s];
                }
                double c1 = thr_cpu();
                fg::ChunkHeader ch; std::memset(&ch, 0, sizeof ch);
                ch.magic = fg::CHUNK_MAGIC; ch.n_pairs = cn; ch.n_exc = (uint32_t)ex.size(); ch.flags = 0;
                uint64_t off = sizeof(fg::ChunkHeader);
                for (int s = 0; s < 3; s++) {
                    off = fg::align_up(off, fg::SEC_ALIGN);
                    ch.sec[s] = {(uint32_t)csz[s], (uint32_t)rawsz[s], 0, (uint32_t)off};
                    if (is_raw[s] && rawsz[s]) ch.flags |= 1u << s;       // bit s: section s stored raw
                    off += csz[s];
                }
                OutChunk oc; oc.n_pairs = cn; oc.exc_raw = (uint32_t)rawsz[2];
                oc.bytes.assign(fg::align_up(off, fg::CHUNK_ALIGN), 0);
                for (int s = 0; s < 3; s++) {
                    const uint8_t* pay = is_raw[s] ? src[s] : secbuf[s].data();
                    if (csz[s]) std::memcpy(oc.bytes.data() + ch.sec[s].offset, pay, csz[s]);
                    if (C.crc) ch.sec[s].crc32c = fg::crc32c(pay, csz[s]);
                }
                double c2 = thr_cpu();
                std::memcpy(oc.bytes.data(), &ch, sizeof ch);
                tc += c1 - c0; tk += c2 - c1;
                bo.chunks.push_back(std::move(oc));
                pos += cn;
            }
            td += t1 - t0; tp += t2 - t1;
            { std::lock_guard<std::mutex> lk(mu); done.emplace(b, std::move(bo)); }
            cv.notify_all();
        }
        ZSTD_freeDCtx(dctx); ZSTD_freeCCtx(cctx);
        auto add = [](std::atomic<double>& a, double v) { double o = a.load(); while (!a.compare_exchange_weak(o, o + v)) {} };
        add(t_decomp, td); add(t_parse, tp); add(t_comp, tc); add(t_crc, tk);
        for (int q = 0; q < 3; q++) { g_sec_c[q] += sec_c[q]; g_sec_r[q] += sec_r[q]; }
    };

    double w0 = now_s(); rusage r0; getrusage(RUSAGE_SELF, &r0);
    int ofd = open(C.out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ofd < 0) { perror("open out"); return 1; }
    // layout: header (256) | meta | gap to 16 KiB | chunks (4096 aligned) | index | footer. Header and meta are written last.
    const uint64_t META_RESERVE = 16384;
    std::vector<fg::IndexEntry> index; uint64_t woff = META_RESERVE;
    uint64_t tot_pairs = 0, tot_in = 0, tot_drop = 0, tot_unk = 0, tot_nN = 0, gen_blocks = 0; uint32_t max_exc = 0;
    double wwait = 0, write_s = 0;
    std::vector<std::thread> th; for (int i = 0; i < C.threads; i++) th.emplace_back(worker);
    for (uint32_t b = 0; b < nb; b++) {
        BlockOut bo;
        { double a = now_s(); std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return done.count(b) > 0; }); bo = std::move(done[b]); done.erase(b); wnext = b + 1; wwait += now_s() - a; }
        cv.notify_all();
        tot_in += bo.n_in; tot_drop += bo.dropped; tot_unk += bo.unk; tot_nN += bo.nN; gen_blocks += bo.generic;
        for (auto& oc : bo.chunks) {
            double a = now_s();
            size_t left = oc.bytes.size(); const uint8_t* pp = oc.bytes.data(); uint64_t o = woff;
            while (left) { ssize_t r = pwrite(ofd, pp, left, o); if (r <= 0) { perror("pwrite"); return 1; } pp += r; o += r; left -= r; }
            write_s += now_s() - a;
            index.push_back({woff, (uint32_t)oc.bytes.size(), oc.n_pairs});
            woff += oc.bytes.size(); tot_pairs += oc.n_pairs; max_exc = std::max(max_exc, oc.exc_raw);
        }
    }
    for (auto& t : th) t.join();
    uint64_t index_off = woff;
    { size_t sz = index.size() * sizeof(fg::IndexEntry);
      pwrite(ofd, index.data(), sz, index_off);
      fg::Footer ft{fg::FOOT_MAGIC, (uint32_t)index.size(), index_off, fg::crc32c(index.data(), sz), 0};
      pwrite(ofd, &ft, sizeof ft, index_off + sz); }
    std::string src_meta; try { src_meta = rd.read_metadata(); } catch (...) {}
    auto esc = [](const std::string& s) { std::string o; for (char c : s) { if (c == '"' || c == '\\') o += '\\'; if ((unsigned char)c < 32) continue; o += c; } return o; };
    std::string meta = "{\"format\":\"1fg\",\"version\":1,\"source\":\"" + esc(C.in) + "\",\"source_format\":\"1fq\",\"source_bytes\":" + std::to_string(st.st_size) +
        ",\"qualities\":\"none\",\"cb_encoding\":\"2bit_u32_first_base_high\",\"umi_encoding\":\"2bit_24bit_prefix_then_suffix\",\"r2_encoding\":\"2bit_first_base_high_padded\",\"source_1fq_metadata\":\"" + esc(src_meta) + "\"}";
    std::vector<uint8_t> mc(ZSTD_compressBound(meta.size())); size_t mcs = ZSTD_compress(mc.data(), mc.size(), meta.data(), meta.size(), 3);
    if (mcs > META_RESERVE - 256) { fprintf(stderr, "meta too large\n"); return 1; }
    pwrite(ofd, mc.data(), mcs, 256);
    fg::FileHeader fh; std::memset(&fh, 0, sizeof fh);
    fh.magic = fg::FILE_MAGIC; fh.version = 1; fh.header_size = 256; fh.codec = (uint8_t)C.codec; fh.codec_level = (uint8_t)C.level;
    fh.flags = C.crc ? fg::F_CRC : 0; fh.protocol_id = h.protocol_id; fh.chunk_pairs = C.chunk_pairs; fh.n_pairs = tot_pairs; fh.n_pairs_in = tot_in;
    fh.n_dropped = tot_drop; fh.n_chunks = (uint32_t)index.size(); fh.r2_slot = (uint16_t)R2B; fh.r2_max_len = (uint16_t)P.r2_max;
    fh.bc_len = (uint8_t)P.bc_len; fh.umi_len = (uint8_t)P.umi_len; fh.bc_start = (uint8_t)P.bc_start; fh.r1_len = (uint8_t)P.r1_len; fh.max_exc_bytes = max_exc;
    fh.meta_offset = 256; fh.meta_comp_size = (uint32_t)mcs; fh.meta_raw_size = (uint32_t)meta.size(); fh.chunks_offset = META_RESERVE;
    fh.index_offset = index_off; fh.source_bytes = st.st_size;
    pwrite(ofd, &fh, sizeof fh, 0);
    fsync(ofd); close(ofd);
    double w1 = now_s(); rusage r1; getrusage(RUSAGE_SELF, &r1);
    auto tv = [](const timeval& t) { return t.tv_sec + t.tv_usec * 1e-6; };
    double cpu = tv(r1.ru_utime) + tv(r1.ru_stime) - tv(r0.ru_utime) - tv(r0.ru_stime);
    double wall = w1 - w0, Mp = tot_in / 1e6;
    struct stat so; stat(C.out.c_str(), &so);
    // in_bytes: the source blocks actually processed (whole file unless --max-blocks)
    uint64_t in_bytes = st.st_size;
    if (nb < h.block_count) { in_bytes = 0; for (auto& b : P.blocks) in_bytes += b.csz + sizeof(BlockHeader); }
    printf("{\"tool\":\"enc_1fg\",\"in\":\"%s\",\"out\":\"%s\",\"codec\":\"%s\",\"raw_sections\":\"%d%d%d\",\"level\":%d,\"threads\":%d,\"crc\":%d,\"pairs_in\":%llu,\"pairs_out\":%llu,"
           "\"dropped\":%llu,\"unknown_bc\":%llu,\"n_exc_N\":%llu,\"generic_blocks\":%llu,\"chunks\":%zu,\"r2_slot\":%u,\"r2_max_len\":%u,"
           "\"in_bytes\":%llu,\"out_bytes\":%lld,\"out_over_in\":%.4f,\"out_bytes_per_pair\":%.3f,\"in_bytes_per_pair\":%.3f,\"wall_s\":%.3f,\"cpu_s_total\":%.3f,"
           "\"cpu_s_per_Mpairs\":%.4f,\"Mpairs_per_s\":%.2f,\"in_MBps\":%.1f,\"out_MBps\":%.1f,"
           "\"cpu_split_per_Mpairs\":{\"read_zstd_decomp\":%.4f,\"parse\":%.4f,\"zstd_compress\":%.4f,\"copy_crc\":%.4f},\"writer_wait_s\":%.2f,\"write_s\":%.2f,\"chunk_pairs\":%u,\"section_bytes\":{\"cbumi_comp\":%llu,\"cbumi_raw\":%llu,\"r2_comp\":%llu,\"r2_raw\":%llu,\"exc_comp\":%llu,\"exc_raw\":%llu}}\n",
           C.in.c_str(), C.out.c_str(), C.codec ? "zstd" : "raw", (int)C.raw_sec[0], (int)C.raw_sec[1], (int)C.raw_sec[2], C.level, C.threads, C.crc, (unsigned long long)tot_in, (unsigned long long)tot_pairs,
           (unsigned long long)tot_drop, (unsigned long long)tot_unk, (unsigned long long)tot_nN, (unsigned long long)gen_blocks, index.size(), R2B, P.r2_max,
           (unsigned long long)in_bytes, (long long)so.st_size, (double)so.st_size / in_bytes, (double)so.st_size / tot_pairs, (double)in_bytes / tot_in, wall, cpu,
           cpu / Mp, Mp / wall, in_bytes / 1e6 / wall, so.st_size / 1e6 / wall,
           t_decomp.load() / Mp, t_parse.load() / Mp, t_comp.load() / Mp, t_crc.load() / Mp, wwait, write_s, C.chunk_pairs,
           (unsigned long long)g_sec_c[0].load(), (unsigned long long)g_sec_r[0].load(), (unsigned long long)g_sec_c[1].load(), (unsigned long long)g_sec_r[1].load(), (unsigned long long)g_sec_c[2].load(), (unsigned long long)g_sec_r[2].load());
    return 0;
}
