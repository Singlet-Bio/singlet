// M8: host -> device ingest pipeline for the GPU engine, with dummy kernels (docs/gpu-engine/exploration.md, section 9, M8).
//
// Build (CUDA devel image; zstd from the conda env pbench):
//   nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo -I<repo>/include -I$PB/include -Xcompiler -pthread -o m8_ingest m8_ingest.cu -L$PB/lib -lzstd
// Modes:
//   m8_ingest --file F [--threads N] [--spin NS] [--ring 4] [--mb 256] [--dev 0] [--max-pairs P] [--prefetch D]
//             [--dontneed 1] [--verify] [--crc] [--label S]    one pipeline run (one JSON line on stdout)
//   m8_ingest --inspect F [--selftest K]                                           header, block table, decoder self-test
//   m8_ingest --copy SRC DST [--streams T]                                         timed copy (shared FS -> NVMe)
//   m8_ingest --bw [--dev 0] [--reps 30]                                           pinned H2D / D2H bandwidth, 256 MB
//   m8_ingest --resid F                                                            page-cache residency of F (mincore)
//
// Pipeline: N decode threads claim .1fq blocks in order, pread the compressed block (optionally with a WILLNEED prefetch D
// blocks ahead), zstd-decompress it and write the GPU payload layout directly into a pinned ring buffer (4 x 256 MB).
// Batch layout (struct of arrays over the batch's n pairs):
//   R2   : n x R2B bytes, 2 bits per base, first base in the top bits, zero padded (R2B = 24 for <= 96 nt; widened to the next
//          multiple of 8 B for longer reads, from the header's R2 length or the first blocks if the length is variable)
//   CBUMI: n x 8 bytes, little-endian u64: bits 0-31 CB (2 bit/base, bc_length <= 16), bits 32-55 UMI (2 bit/base, <= 12 nt),
//          bit 56-62 R2 length (<= 127), bit 63 any-N flag (R2 or UMI; the base is stored as A, like the .1fq packing)
//   QUAL : n x QB bytes (QB = R2B), binned-4 quality, 2 bits per base, as stored in the file
// i.e. 56 B per pair for 91-96 nt reads (the design note counts 53 B unpadded: 23 + 7 + 23).
// A submitter thread copies each full batch with cudaMemcpyAsync on a copy stream into one of 3 device buffers; events gate a
// compute stream that runs a dummy kernel which reads every byte (XOR fold, checked against the host in --verify) and spins
// until n * spin_ns nanoseconds have passed since the kernel started. A releaser thread frees the pinned slot when the copy
// event completes.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <numeric>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime.h>

#include <singlet/fq/reader.h>
#include <zstd.h>

using namespace singlet::fq;
using Clock = std::chrono::steady_clock;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static double now_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
static double epoch_s() { timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static double thread_cpu_s() { timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

// ---------------------------------------------------------------------------------------------------------------- args
struct Args {
    std::string file, label, copy_src, copy_dst, inspect, resid;
    int threads = 3, ring = 4, dev = 0, streams = 1, selftest = 0, reps = 30;
    double spin_ns = 0;
    size_t mb = 256;
    uint64_t max_pairs = 0;
    int prefetch = -1;      // blocks ahead; -1 = 2 x threads; 0 = off
    int dontneed = 1;
    bool verify = false, crc = false, bw = false, qsynth = true;
};
static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto nx = [&]() -> const char* { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", k.c_str()); exit(2); } return argv[++i]; };
        if (k == "--file") a.file = nx();
        else if (k == "--threads") a.threads = atoi(nx());
        else if (k == "--spin") a.spin_ns = atof(nx());
        else if (k == "--ring") a.ring = atoi(nx());
        else if (k == "--mb") a.mb = atol(nx());
        else if (k == "--dev") a.dev = atoi(nx());
        else if (k == "--max-pairs") a.max_pairs = strtoull(nx(), nullptr, 10);
        else if (k == "--prefetch") a.prefetch = atoi(nx());
        else if (k == "--dontneed") a.dontneed = atoi(nx());
        else if (k == "--verify") a.verify = true;
        else if (k == "--crc") a.crc = true;
        else if (k == "--qual") { std::string v = nx(); a.qsynth = (v == "synth"); }   // synth (default) | none
        else if (k == "--label") a.label = nx();
        else if (k == "--inspect") a.inspect = nx();
        else if (k == "--selftest") a.selftest = atoi(nx());
        else if (k == "--copy") { a.copy_src = nx(); a.copy_dst = nx(); }
        else if (k == "--streams") a.streams = atoi(nx());
        else if (k == "--bw") a.bw = true;
        else if (k == "--reps") a.reps = atoi(nx());
        else if (k == "--resid") a.resid = nx();
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); exit(2); }
    }
    return a;
}

// ------------------------------------------------------------------------------------------------------- page cache util
static double resident_frac(const char* path) {
    int fd = open(path, O_RDONLY); if (fd < 0) return -1;
    struct stat st; fstat(fd, &st);
    if (st.st_size == 0) { close(fd); return 0; }
    void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { close(fd); return -1; }
    size_t pg = sysconf(_SC_PAGESIZE), np = (st.st_size + pg - 1) / pg;
    std::vector<unsigned char> v(np);
    double r = -1;
    if (mincore(m, st.st_size, v.data()) == 0) { size_t c = 0; for (auto x : v) c += x & 1; r = double(c) / np; }
    munmap(m, st.st_size); close(fd);
    return r;
}
static void drop_cache(const char* path) {
    int fd = open(path, O_RDONLY); if (fd < 0) return;
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
}
static bool pread_full(int fd, void* buf, size_t n, uint64_t off) {
    uint8_t* p = (uint8_t*)buf;
    while (n) { ssize_t r = pread(fd, p, n, off); if (r <= 0) return false; p += r; n -= r; off += r; }
    return true;
}

// ---------------------------------------------------------------------------------------------------------- file layout
struct Layout {
    uint32_t R2B = 24, QB = 24;       // slot bytes per pair for R2 and quality
    uint32_t bytes_per_pair() const { return R2B + 8 + QB; }
};

struct BlockPlan {
    uint64_t off; uint32_t csz, nreads, batch; uint64_t first_in_batch;   // pair offset of this block inside its batch
};
struct BatchPlan { uint32_t first_block, nblocks; uint64_t pairs; };

struct FilePlan {
    Header hdr;
    std::vector<BlockPlan> blocks;
    std::vector<BatchPlan> batches;
    uint64_t pairs = 0;
    Layout lay;
    uint32_t bc_len = 0, umi_len = 0, bc_start = 0, r1_len = 0, r2_fixed = 0;
    uint32_t dict_n = 0;
    std::vector<uint32_t> cbtab;       // dict index -> 2-bit CB
    bool fast_ok = false;
    bool qual_file = false;            // the file stores binned-4 quality (the atlas files do not)
    bool qsynth = true;
    std::string why_not_fast;
    uint32_t max_r2 = 0;
};

static uint32_t pack_bases_u32(const uint8_t* b, uint32_t n) { uint32_t v = 0; for (uint32_t i = 0; i < n; i++) v = (v << 2) | (b[i] & 3); return v; }

// Build the plan: header, dictionary, block table (n_reads per block from the 12 B block headers), batches.
static void build_plan(const char* path, int fd, Reader& rd, size_t batch_bytes, uint64_t max_pairs, bool qsynth, FilePlan& P) {
    P.hdr = rd.header();
    const Header& h = P.hdr;
    P.r1_len = h.stream_lengths[0];
    P.r2_fixed = (h.flags & Flags::TRIMMED) ? 0 : h.stream_lengths[1];
    P.bc_len = rd.bc_length();
    P.umi_len = rd.umi_length();
    P.dict_n = rd.bc_dict_size();
    P.bc_start = 0;
    for (const auto& s : known_protocols()) if (s.protocol_id == h.protocol_id) { P.bc_start = s.bc_offset; break; }
    std::string why;
    if (!rd.has_bc_dict()) why += "no BC dict; ";
    if (h.seq_encoding != (uint8_t)SeqEncoding::PACKED_2BIT) why += "seq not 2-bit; ";
    if (h.qual_mode != (uint8_t)QualMode::BINNED4 && h.qual_mode != (uint8_t)QualMode::NONE) why += "qual mode unsupported; ";
    P.qual_file = h.qual_mode == (uint8_t)QualMode::BINNED4; P.qsynth = qsynth;
    if (P.bc_len == 0 || P.bc_len > 16) why += "bc_len unsupported; ";
    if (P.umi_len > 12) why += "umi_len > 12; ";
    if (h.codec != (uint8_t)Codec::ZSTD) why += "codec not zstd; ";
    P.why_not_fast = why; P.fast_ok = why.empty();
    if (P.fast_ok) {
        const auto& d = rd.bc_dict_flat();
        P.cbtab.resize(P.dict_n);
        for (uint32_t i = 0; i < P.dict_n; i++) P.cbtab[i] = pack_bases_u32(d.data() + (size_t)i * P.bc_len, P.bc_len);
    }
    uint32_t nb = h.block_count;
    P.blocks.resize(nb);
    for (uint32_t b = 0; b < nb; b++) {
        BlockHeader bh;
        if (!pread_full(fd, &bh, sizeof bh, rd.block_offset(b))) { fprintf(stderr, "block header read failed %u\n", b); exit(1); }
        P.blocks[b].off = rd.block_offset(b) + sizeof(BlockHeader);
        P.blocks[b].csz = bh.compressed_size;
        P.blocks[b].nreads = bh.n_reads;
    }
    // R2 length: fixed from header; if variable (trimmed), the maximum over the first 8 blocks (decoded with the Reader)
    uint32_t maxlen = P.r2_fixed;
    if (maxlen == 0) {
        std::vector<uint8_t> comp, raw;
        for (uint32_t b = 0; b < std::min<uint32_t>(8, nb); b++) {
            DecodedBlock blk;
            comp.resize(P.blocks[b].csz); pread_full(fd, comp.data(), comp.size(), P.blocks[b].off);
            BlockHeader bh{P.blocks[b].nreads, P.blocks[b].csz, 0};
            raw.resize((size_t)bh.n_reads * 400);
            size_t rs = compress::decompress_block(Codec::ZSTD, comp.data(), comp.size(), raw.data(), raw.size());
            rd.decode_payload(raw.data(), rs, bh, blk);
            for (auto l : blk.r2_lengths) maxlen = std::max<uint32_t>(maxlen, l);
        }
    }
    P.max_r2 = maxlen;
    uint32_t pb = (maxlen + 3) / 4;
    P.lay.R2B = std::max<uint32_t>(24, (pb + 7) / 8 * 8);
    P.lay.QB = (P.qual_file || P.qsynth) ? P.lay.R2B : 0;
    if (maxlen > 127) { P.fast_ok = false; P.why_not_fast += "r2 length > 127 (CBUMI length field); "; }
    // batches
    size_t cap = batch_bytes / P.lay.bytes_per_pair();
    uint64_t total = 0;
    BatchPlan cur{0, 0, 0};
    for (uint32_t b = 0; b < nb; b++) {
        if (max_pairs && total >= max_pairs) { nb = b; P.blocks.resize(nb); break; }
        uint32_t n = P.blocks[b].nreads;
        if (cur.nblocks && cur.pairs + n > cap) { P.batches.push_back(cur); cur = {b, 0, 0}; }
        P.blocks[b].batch = (uint32_t)P.batches.size();
        P.blocks[b].first_in_batch = cur.pairs;
        cur.nblocks++; cur.pairs += n; total += n;
    }
    if (cur.nblocks) P.batches.push_back(cur);
    P.pairs = total;
}

// ---------------------------------------------------------------------------------------------------- block decoding
struct BlockStats { uint64_t nN = 0, trunc = 0, unk_bc = 0; };

struct Cols { uint8_t *r2, *cu, *q; };     // column bases of the batch buffer

static inline uint64_t read_varint(const uint8_t*& p) {
    uint64_t v = 0; int sh = 0;
    for (;;) { uint8_t b = *p++; v |= (uint64_t)(b & 0x7f) << sh; if (!(b & 0x80)) break; sh += 7; if (sh >= 64) break; }
    return v;
}

// Copy pb (<= 24) packed bytes into a zero-padded 24 B slot without a variable-length memcpy: three unconditional 8 B loads
// (the raw buffer has >= 64 B of slack) masked to the valid bytes.
static inline void load_slot24(const uint8_t* src, uint32_t pb, uint64_t w[3]) {
    std::memcpy(w, src, 24);
    for (int j = 0; j < 3; j++) {
        uint32_t valid = pb > (uint32_t)(8 * j) ? std::min<uint32_t>(pb - 8 * j, 8u) : 0u;
        w[j] &= valid >= 8 ? ~0ull : ((1ull << (8 * valid)) - 1);
    }
}

// Direct decode of a decompressed payload into the GPU layout. out points at the block's first pair in each column.
// raw must be followed by >= 64 readable bytes. Returns false if the payload is inconsistent.
// cbv holds per read: CB (bits 0-31) and the N flag (bit 63, only for an unknown barcode that contains N).
static bool pack_block_fast(const FilePlan& P, const uint8_t* raw, size_t rawsz, uint32_t n, Cols out, BlockStats& st,
                            std::vector<uint64_t>& cbv, std::vector<uint8_t>& lenv) {
    const uint8_t* p = raw; const uint8_t* end = raw + rawsz;
    const uint32_t R2B = P.lay.R2B, QB = P.lay.QB, bcl = P.bc_len, uml = P.umi_len;
    cbv.resize(n); lenv.resize(n);
    // 1) barcode dictionary indices (varint); after the sentinel index the raw (byte-numeric) barcode follows
    for (uint32_t i = 0; i < n; i++) {
        uint64_t idx = read_varint(p);
        if (idx < P.dict_n) cbv[i] = P.cbtab[idx];
        else {
            uint64_t nn = 0; for (uint32_t k = 0; k < bcl; k++) nn |= p[k] > 3;
            cbv[i] = (uint64_t)pack_bases_u32(p, bcl) | (nn << 63); p += bcl; st.unk_bc++;
        }
    }
    // 2) UMI column (fixed length, has_n byte + packed + optional N bitmap)
    uint64_t* cu = reinterpret_cast<uint64_t*>(out.cu);
    const uint32_t upb = (uml + 3) / 4, unb = (uml + 7) / 8, ush = (8 * upb - 2 * uml) & 31;
    for (uint32_t i = 0; i < n; i++) {
        uint8_t has_n = *p++;
        uint32_t v = 0;
        for (uint32_t k = 0; k < upb; k++) v = (v << 8) | p[k];
        v >>= ush;
        p += upb;
        uint64_t flags = cbv[i] >> 63;
        if (has_n) { p += unb; flags = 1; st.nN++; }
        cu[i] = (cbv[i] & 0xFFFFFFFFull) | ((uint64_t)v << 32) | (flags << 63);
    }
    // 3) R2 column (and, in the synthetic layout, the Q slot filled from the R2 bits: a stand-in for parse+copy of quality)
    const bool var = P.r2_fixed == 0;
    const bool synthQ = QB && !P.qual_file;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t len = P.r2_fixed;
        if (var) len = (uint32_t)read_varint(p);
        uint8_t has_n = *p++;
        uint32_t pb = (len + 3) / 4, olen = len;
        uint8_t* dst = out.r2 + (size_t)i * R2B;
        if (R2B == 24 && pb <= 24) {
            uint64_t w[3]; load_slot24(p, pb, w);
            std::memcpy(dst, w, 24);
            if (synthQ) std::memcpy(out.q + (size_t)i * QB, w, 24);
        } else {
            uint32_t cp = pb < R2B ? pb : R2B;
            std::memcpy(dst, p, cp);
            if (cp < R2B) std::memset(dst + cp, 0, R2B - cp);
            if (synthQ) std::memcpy(out.q + (size_t)i * QB, dst, QB);
            if (pb > R2B) { st.trunc++; len = R2B * 4; }
        }
        p += pb;
        if (has_n) { p += (olen + 7) / 8; cu[i] |= (1ull << 63); st.nN++; }
        lenv[i] = (uint8_t)len;
        cu[i] |= (uint64_t)(len & 127) << 56;
        if (p > end) return false;
    }
    // 4) binned-4 quality column, when the file has one (the atlas files do not: qual_mode 0)
    if (QB && P.qual_file) {
        for (uint32_t i = 0; i < n; i++) {
            uint32_t len = lenv[i], pb = (len + 3) / 4;
            uint8_t* dst = out.q + (size_t)i * QB;
            uint32_t cp = pb < QB ? pb : QB;
            std::memcpy(dst, p, cp);
            if (cp < QB) std::memset(dst + cp, 0, QB - cp);
            p += pb;
        }
    }
    return p <= end;
}

// Reference/fallback: decode with the library (byte-numeric) and re-pack into the same layout.
static void pack_block_from_decoded(const FilePlan& P, const DecodedBlock& b, Cols out, BlockStats& st) {
    const uint32_t R2B = P.lay.R2B, QB = P.lay.QB, bcl = P.bc_len, uml = P.umi_len;
    uint64_t* cu = reinterpret_cast<uint64_t*>(out.cu);
    for (uint32_t i = 0; i < b.n_reads; i++) {
        const uint8_t* r1 = b.r1_seq(i);
        uint8_t bc[16], um[16];
        for (uint32_t k = 0; k < bcl; k++) bc[k] = r1[P.bc_start + k];
        uint32_t u = 0;
        for (uint32_t k = 0; k < P.bc_start && u < uml; k++) um[u++] = r1[k];
        for (uint32_t k = P.bc_start + bcl; k < P.r1_len && u < uml; k++) um[u++] = r1[k];
        bool anyN = false;
        for (uint32_t k = 0; k < uml; k++) anyN |= um[k] > 3;
        uint32_t len = b.r2_len(i);
        const uint8_t* s = b.r2_seq(i);
        uint8_t* dst = out.r2 + (size_t)i * R2B; std::memset(dst, 0, R2B);
        uint32_t used = std::min(len, R2B * 4);
        for (uint32_t k = 0; k < used; k++) { anyN |= s[k] > 3; dst[k / 4] |= (s[k] & 3) << (6 - 2 * (k % 4)); }
        if (len > R2B * 4) st.trunc++;
        uint8_t* qd = out.q + (size_t)i * QB; if (QB) std::memset(qd, 0, QB);
        const uint8_t* q = QB && P.qual_file ? b.r2_quality(i) : nullptr;
        if (QB && !P.qual_file) std::memcpy(qd, dst, QB);
        if (q) for (uint32_t k = 0; k < used; k++) { uint8_t ph = q[k]; uint8_t bin = ph < 10 ? 0 : ph < 20 ? 1 : ph < 30 ? 2 : 3; qd[k / 4] |= bin << (6 - 2 * (k % 4)); }
        cu[i] = (uint64_t)pack_bases_u32(bc, bcl) | ((uint64_t)pack_bases_u32(um, uml) << 32) | ((uint64_t)(used & 127) << 56) | ((uint64_t)anyN << 63);
        if (anyN) st.nN++;
    }
}

// ------------------------------------------------------------------------------------------------------------ inspect
static int do_inspect(const Args& a) {
    Reader rd; rd.open(a.inspect.c_str());
    int fd = open(a.inspect.c_str(), O_RDONLY);
    FilePlan P; build_plan(a.inspect.c_str(), fd, rd, a.mb << 20, 0, a.qsynth, P);
    const Header& h = P.hdr;
    uint32_t mn = ~0u, mx = 0; uint64_t tot = 0, csum = 0;
    for (auto& b : P.blocks) { mn = std::min(mn, b.nreads); mx = std::max(mx, b.nreads); tot += b.nreads; csum += b.csz; }
    struct stat st; fstat(fd, &st);
    printf("{\"file\":\"%s\",\"size\":%lld,\"version\":%u,\"protocol_id\":%u,\"flags\":%u,\"codec\":%u,\"codec_level\":%u,\"n_streams\":%u,"
           "\"stream_lengths\":[%u,%u,%u,%u],\"n_unique\":%llu,\"n_original\":%llu,\"block_count\":%u,\"block_size\":%u,\"seq_encoding\":%u,"
           "\"qual_mode\":%u,\"tier\":%u,\"bc_dict_n\":%u,\"bc_len\":%u,\"umi_len\":%u,\"bc_start\":%u,\"r2_fixed\":%u,\"max_r2\":%u,"
           "\"block_reads_min\":%u,\"block_reads_max\":%u,\"sum_reads\":%llu,\"comp_bytes\":%llu,\"bytes_per_pair_file\":%.2f,"
           "\"R2B\":%u,\"payload_bytes_per_pair\":%u,\"batches_at_%zuMB\":%zu,\"fast_ok\":%s,\"why_not_fast\":\"%s\"}\n",
           a.inspect.c_str(), (long long)st.st_size, h.version, h.protocol_id, h.flags, h.codec, h.codec_level, h.n_streams,
           h.stream_lengths[0], h.stream_lengths[1], h.stream_lengths[2], h.stream_lengths[3], (unsigned long long)h.n_unique,
           (unsigned long long)h.n_original, h.block_count, h.block_size, h.seq_encoding, h.qual_mode, h.tier, P.dict_n, P.bc_len,
           P.umi_len, P.bc_start, P.r2_fixed, P.max_r2, mn, mx, (unsigned long long)tot, (unsigned long long)csum,
           double(st.st_size) / double(tot), P.lay.R2B, P.lay.bytes_per_pair(), a.mb, P.batches.size(), P.fast_ok ? "true" : "false",
           P.why_not_fast.c_str());
    if (a.selftest > 0 && P.fast_ok) {
        // compare the direct decoder with Reader::decode_payload + re-pack on evenly spaced blocks
        uint32_t nb = P.blocks.size(), K = std::min<uint32_t>(a.selftest, nb);
        ZSTD_DCtx* dctx = ZSTD_createDCtx();
        std::vector<uint8_t> comp, raw; std::vector<uint64_t> cbv; std::vector<uint8_t> lenv;
        uint64_t bad = 0, checked = 0, nN = 0;
        for (uint32_t k = 0; k < K; k++) {
            uint32_t b = (uint64_t)k * nb / K;
            const BlockPlan& bp = P.blocks[b];
            comp.resize(bp.csz); pread_full(fd, comp.data(), bp.csz, bp.off);
            unsigned long long fs = ZSTD_getFrameContentSize(comp.data(), comp.size());
            size_t cap = (fs == ZSTD_CONTENTSIZE_UNKNOWN || fs == ZSTD_CONTENTSIZE_ERROR) ? (size_t)bp.nreads * 400 : fs;
            raw.assign(cap + 64, 0);
            size_t rs = ZSTD_decompressDCtx(dctx, raw.data(), cap, comp.data(), comp.size());
            if (ZSTD_isError(rs)) { printf("zstd error %s\n", ZSTD_getErrorName(rs)); return 1; }
            size_t n = bp.nreads;
            std::vector<uint8_t> A(n * P.lay.R2B), Aq(n * P.lay.QB), B(n * P.lay.R2B), Bq(n * P.lay.QB);
            std::vector<uint64_t> Ac(n), Bc(n);
            BlockStats s1, s2;
            Cols o1{A.data(), (uint8_t*)Ac.data(), Aq.data()}, o2{B.data(), (uint8_t*)Bc.data(), Bq.data()};
            bool ok = pack_block_fast(P, raw.data(), rs, n, o1, s1, cbv, lenv);
            DecodedBlock blk; BlockHeader bh{bp.nreads, bp.csz, 0};
            rd.decode_payload(raw.data(), rs, bh, blk);
            pack_block_from_decoded(P, blk, o2, s2);
            for (size_t i = 0; i < n; i++) if (blk.bc_indices[i] >= P.dict_n) { Ac[i] &= ~(0xFFFFFFFFull | (1ull << 63)); Bc[i] &= ~(0xFFFFFFFFull | (1ull << 63)); }
            bool eq = ok && A == B && Aq == Bq && Ac == Bc;
            if (!eq) { bad++; size_t i = 0; while (i < n && std::memcmp(&Ac[i], &Bc[i], 8) == 0 && !std::memcmp(&A[i * P.lay.R2B], &B[i * P.lay.R2B], P.lay.R2B) && !std::memcmp(&Aq[i * P.lay.QB], &Bq[i * P.lay.QB], P.lay.QB)) i++;
                       printf("MISMATCH block %u first bad read %zu ok=%d cu %llx vs %llx\n", b, i, (int)ok, i < n ? (unsigned long long)Ac[i] : 0ull, i < n ? (unsigned long long)Bc[i] : 0ull); }
            checked += n; nN += s1.nN;
        }
        printf("{\"selftest_blocks\":%u,\"selftest_reads\":%llu,\"mismatching_blocks\":%llu,\"reads_with_N_flag\":%llu}\n", K,
               (unsigned long long)checked, (unsigned long long)bad, (unsigned long long)nN);
        return bad ? 1 : 0;
    }
    return 0;
}

// -------------------------------------------------------------------------------------------------------------- copy
static int do_copy(const Args& a) {
    int in = open(a.copy_src.c_str(), O_RDONLY); if (in < 0) { perror("open src"); return 1; }
    struct stat st; fstat(in, &st);
    double res_before = resident_frac(a.copy_src.c_str());
    posix_fadvise(in, 0, 0, POSIX_FADV_DONTNEED);
    double res_after = resident_frac(a.copy_src.c_str());
    int out = open(a.copy_dst.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644); if (out < 0) { perror("open dst"); return 1; }
    if (ftruncate(out, st.st_size) != 0) perror("ftruncate");
    int T = std::max(1, a.streams);
    uint64_t chunk = 8ull << 20, per = (st.st_size + T - 1) / T;
    double t0 = now_s(); rusage r0, r1; getrusage(RUSAGE_SELF, &r0);
    std::vector<std::thread> th;
    for (int t = 0; t < T; t++) th.emplace_back([&, t] {
        std::vector<uint8_t> buf(chunk);
        uint64_t lo = t * per, hi = std::min<uint64_t>(st.st_size, lo + per);
        posix_fadvise(in, lo, hi - lo, POSIX_FADV_SEQUENTIAL);
        for (uint64_t o = lo; o < hi; o += chunk) {
            size_t n = std::min<uint64_t>(chunk, hi - o);
            if (!pread_full(in, buf.data(), n, o)) { fprintf(stderr, "read fail\n"); exit(1); }
            ssize_t w = pwrite(out, buf.data(), n, o); if (w != (ssize_t)n) { fprintf(stderr, "write fail\n"); exit(1); }
        }
    });
    for (auto& x : th) x.join();
    double t1 = now_s();
    fdatasync(out);
    double t2 = now_s(); getrusage(RUSAGE_SELF, &r1);
    posix_fadvise(out, 0, 0, POSIX_FADV_DONTNEED);      // make later reads of the NVMe copy cold
    double cpu = (r1.ru_utime.tv_sec - r0.ru_utime.tv_sec) + (r1.ru_utime.tv_usec - r0.ru_utime.tv_usec) * 1e-6 +
                 (r1.ru_stime.tv_sec - r0.ru_stime.tv_sec) + (r1.ru_stime.tv_usec - r0.ru_stime.tv_usec) * 1e-6;
    printf("{\"copy_src\":\"%s\",\"bytes\":%lld,\"streams\":%d,\"resident_before\":%.4f,\"resident_after_dontneed\":%.4f,"
           "\"read_write_s\":%.3f,\"with_fsync_s\":%.3f,\"MBps_read_write\":%.1f,\"MBps_with_fsync\":%.1f,\"cpu_s\":%.2f,\"epoch\":%.1f}\n",
           a.copy_src.c_str(), (long long)st.st_size, T, res_before, res_after, t1 - t0, t2 - t0, st.st_size / 1e6 / (t1 - t0),
           st.st_size / 1e6 / (t2 - t0), cpu, epoch_s());
    close(in); close(out);
    return 0;
}

// ----------------------------------------------------------------------------------------------------------- bandwidth
static int do_bw(const Args& a) {
    CK(cudaSetDevice(a.dev));
    size_t B = 256ull << 20; void *h, *d;
    CK(cudaHostAlloc(&h, B, cudaHostAllocDefault)); CK(cudaMalloc(&d, B)); memset(h, 1, B);
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto run = [&](bool h2d) {
        std::vector<double> v;
        for (int r = 0; r < a.reps + 3; r++) {
            CK(cudaEventRecord(e0, s));
            CK(cudaMemcpyAsync(h2d ? d : h, h2d ? h : d, B, h2d ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, s));
            CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            if (r >= 3) v.push_back(B / 1e9 / (ms * 1e-3));
        }
        std::sort(v.begin(), v.end()); return v;
    };
    double t0 = now_s(); epoch_s();
    auto a1 = run(true), a2 = run(false);
    printf("{\"bw_dev\":%d,\"bytes\":%zu,\"h2d_GBps_median\":%.2f,\"h2d_min\":%.2f,\"h2d_max\":%.2f,\"d2h_GBps_median\":%.2f,\"d2h_min\":%.2f,\"d2h_max\":%.2f,"
           "\"duration_s\":%.2f,\"epoch\":%.1f}\n", a.dev, B, a1[a1.size() / 2], a1.front(), a1.back(), a2[a2.size() / 2], a2.front(), a2.back(),
           now_s() - t0, epoch_s());
    return 0;
}

// ----------------------------------------------------------------------------------------------------- dummy kernel
__device__ __forceinline__ uint64_t gtimer() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__global__ void dummy_kernel(const uint4* __restrict__ buf, size_t nvec, unsigned long long* out, unsigned long long spin_total_ns) {
    uint64_t t0 = gtimer();
    uint64_t acc = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) {
        uint4 v = __ldg(buf + i);
        acc ^= ((uint64_t)v.y << 32 | v.x) ^ ((uint64_t)v.w << 32 | v.z);
    }
    while (gtimer() - t0 < spin_total_ns) __nanosleep(200);
    // fold the block's XOR into the output (xor is order independent)
    atomicXor(out, (unsigned long long)acc);
}

// ----------------------------------------------------------------------------------------------------------- pipeline
struct Shared {
    const FilePlan* P; const Args* A; int fd;
    uint8_t* host[16]; int ring;
    std::atomic<uint32_t> next_block{0};
    std::mutex mu; std::condition_variable cv;
    std::vector<uint32_t> done;                 // blocks finished per batch
    uint32_t released = 0;                      // batches whose pinned slot is free again
    uint32_t enqueued = 0;                      // batches whose copy has been enqueued
    std::vector<double> ready_ts, decode_sum;   // per batch (decode_sum: thread seconds spent in blocks of the batch)
    std::vector<double> decode_cpu_sum;
};
struct TStats { double io = 0, zstd = 0, pack = 0, wait = 0, cpu = 0, crc = 0; uint64_t raw = 0, comp = 0, reads = 0, blocks = 0; BlockStats bs; };

static Cols cols_for(const FilePlan& P, uint8_t* base, const BatchPlan& b) {
    Cols c; c.r2 = base; c.cu = base + (size_t)b.pairs * P.lay.R2B; c.q = c.cu + (size_t)b.pairs * 8; return c;
}
static size_t batch_used_bytes(const FilePlan& P, const BatchPlan& b) {
    size_t u = (size_t)b.pairs * P.lay.bytes_per_pair(); return (u + 15) / 16 * 16;
}

static void decode_thread(Shared* S, TStats* ts, int tid) {
    const FilePlan& P = *S->P; const Args& A = *S->A;
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    std::vector<uint8_t> comp, raw; std::vector<uint64_t> cbv; std::vector<uint8_t> lenv;
    double c0 = thread_cpu_s();
    uint32_t nb = P.blocks.size();
    int pf = A.prefetch < 0 ? 2 * A.threads : A.prefetch;
    for (;;) {
        uint32_t b = S->next_block.fetch_add(1);
        if (b >= nb) break;
        if (pf > 0 && b + pf < nb) { const BlockPlan& q = P.blocks[b + pf]; posix_fadvise(S->fd, q.off, q.csz, POSIX_FADV_WILLNEED); }
        const BlockPlan& bp = P.blocks[b];
        // wait for the pinned slot (batch B needs batch B-ring released)
        double tw0 = now_s();
        {
            std::unique_lock<std::mutex> lk(S->mu);
            S->cv.wait(lk, [&] { return S->released + S->ring > bp.batch; });
        }
        double tw1 = now_s(); ts->wait += tw1 - tw0;
        const BatchPlan& bt = P.batches[bp.batch];
        Cols cb = cols_for(P, S->host[bp.batch % S->ring], bt);
        Cols out{cb.r2 + (size_t)bp.first_in_batch * P.lay.R2B, cb.cu + (size_t)bp.first_in_batch * 8, cb.q + (size_t)bp.first_in_batch * P.lay.QB};
        // read
        comp.resize(bp.csz + 64);
        if (!pread_full(S->fd, comp.data(), bp.csz, bp.off)) { fprintf(stderr, "pread failed block %u\n", b); exit(1); }
        double t1 = now_s();
        // decompress
        unsigned long long fs = ZSTD_getFrameContentSize(comp.data(), bp.csz);
        size_t cap = (fs == ZSTD_CONTENTSIZE_UNKNOWN || fs == ZSTD_CONTENTSIZE_ERROR) ? (size_t)bp.nreads * 400 : (size_t)fs;
        if (raw.size() < cap + 64) raw.resize(cap + 64);
        size_t rs = ZSTD_decompressDCtx(dctx, raw.data(), cap, comp.data(), bp.csz);
        if (ZSTD_isError(rs)) { fprintf(stderr, "zstd: %s\n", ZSTD_getErrorName(rs)); exit(1); }
        std::memset(raw.data() + rs, 0, 64);
        double t2 = now_s();
        if (A.crc) { BlockHeader h2; (void)h2; volatile uint32_t c = compute_crc32(raw.data(), rs); (void)c; }
        double t2b = now_s(); ts->crc += t2b - t2;
        // pack
        if (!pack_block_fast(P, raw.data(), rs, bp.nreads, out, ts->bs, cbv, lenv)) { fprintf(stderr, "payload inconsistent block %u\n", b); exit(1); }
        double t3 = now_s();
        ts->io += t1 - tw1; ts->zstd += t2 - t1; ts->pack += t3 - t2b; ts->raw += rs; ts->comp += bp.csz; ts->reads += bp.nreads; ts->blocks++;
        {
            std::lock_guard<std::mutex> lk(S->mu);
            S->decode_sum[bp.batch] += t3 - tw1;
            if (++S->done[bp.batch] == bt.nblocks) S->ready_ts[bp.batch] = t3;
        }
        S->cv.notify_all();
    }
    ts->cpu = thread_cpu_s() - c0;
    ZSTD_freeDCtx(dctx);
}

static double cpu_s_self() { rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6 + r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6; }
static void cpu_split_self(double& u, double& s) { rusage r; getrusage(RUSAGE_SELF, &r); u = r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6; s = r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6; }

static int do_run(const Args& A) {
    fprintf(stderr, "zstd runtime %s (header %s)\n", ZSTD_versionString(), ZSTD_VERSION_STRING);
    CK(cudaSetDevice(A.dev));
    CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, A.dev));
    Reader rd; rd.open(A.file.c_str());
    int fd = open(A.file.c_str(), O_RDONLY); if (fd < 0) { perror("open"); return 1; }
    double res_before = resident_frac(A.file.c_str()), res_after = res_before;
    if (A.dontneed) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); res_after = resident_frac(A.file.c_str()); }
    double tp0 = now_s();
    FilePlan P; build_plan(A.file.c_str(), fd, rd, A.mb << 20, A.max_pairs, A.qsynth, P);
    if (!P.fast_ok) { fprintf(stderr, "direct decoder unsupported for this file: %s\n", P.why_not_fast.c_str()); return 1; }
    if (A.dontneed) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);     // the plan read touched 12 B per block only; drop again
    double plan_s = now_s() - tp0;
    uint32_t NB = P.batches.size();
    size_t batch_bytes = A.mb << 20;
    double ts0 = now_s();
    Shared S; S.P = &P; S.A = &A; S.fd = fd; S.ring = A.ring;
    for (int i = 0; i < A.ring; i++) CK(cudaHostAlloc(&S.host[i], batch_bytes + 64, cudaHostAllocDefault));
    const int ND = 3;
    uint8_t* dbuf[ND]; for (int i = 0; i < ND; i++) CK(cudaMalloc(&dbuf[i], batch_bytes + 64));
    unsigned long long* dres; CK(cudaMalloc(&dres, NB * 8)); CK(cudaMemset(dres, 0, NB * 8));
    cudaStream_t sc, sk; CK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&sk, cudaStreamNonBlocking));
    std::vector<cudaEvent_t> eh0(NB), eh1(NB), ek0(NB), ek1(NB);
    for (uint32_t i = 0; i < NB; i++) { CK(cudaEventCreateWithFlags(&eh0[i], cudaEventBlockingSync)); CK(cudaEventCreateWithFlags(&eh1[i], cudaEventBlockingSync));
                                        CK(cudaEventCreateWithFlags(&ek0[i], cudaEventBlockingSync)); CK(cudaEventCreateWithFlags(&ek1[i], cudaEventBlockingSync)); }
    int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, dummy_kernel, 256, 0));
    int grid = prop.multiProcessorCount * occ;
    // warm up the kernel and the copy path (not timed): one tiny launch
    dummy_kernel<<<grid, 256, 0, sk>>>((const uint4*)dbuf[0], 1024, dres, 0); CK(cudaStreamSynchronize(sk));
    CK(cudaMemset(dres, 0, NB * 8));
    S.done.assign(NB, 0); S.ready_ts.assign(NB, 0); S.decode_sum.assign(NB, 0);
    double setup_s = now_s() - ts0;
    std::vector<uint64_t> host_xor(NB, 0);
    std::vector<double> submit_wait(NB, 0), submit_ts(NB, 0);

    // --- run ---
    double u0, s0; cpu_split_self(u0, s0);
    double t_start = now_s(), epoch_start = epoch_s();
    std::vector<TStats> TS(A.threads);
    std::vector<std::thread> th;
    for (int t = 0; t < A.threads; t++) th.emplace_back(decode_thread, &S, &TS[t], t);
    std::thread releaser([&] {
        for (uint32_t B = 0; B < NB; B++) {
            { std::unique_lock<std::mutex> lk(S.mu); S.cv.wait(lk, [&] { return S.enqueued > B; }); }
            CK(cudaEventSynchronize(eh1[B]));
            { std::lock_guard<std::mutex> lk(S.mu); S.released = B + 1; }
            S.cv.notify_all();
        }
    });
    double kernel_enq_end = 0;
    for (uint32_t B = 0; B < NB; B++) {
        double w0 = now_s();
        { std::unique_lock<std::mutex> lk(S.mu); S.cv.wait(lk, [&] { return S.done[B] == P.batches[B].nblocks; }); }
        double w1 = now_s(); submit_wait[B] = w1 - w0; submit_ts[B] = w1;
        const BatchPlan& bt = P.batches[B];
        size_t used = batch_used_bytes(P, bt);
        uint8_t* hb = S.host[B % A.ring]; uint8_t* db = dbuf[B % ND];
        if (A.verify) { uint64_t x = 0; const uint64_t* w = (const uint64_t*)hb; for (size_t i = 0; i < used / 8; i++) x ^= w[i]; host_xor[B] = x; }
        if (B >= (uint32_t)ND) CK(cudaStreamWaitEvent(sc, ek1[B - ND], 0));      // device buffer free once that kernel finished
        CK(cudaEventRecord(eh0[B], sc));
        CK(cudaMemcpyAsync(db, hb, used, cudaMemcpyHostToDevice, sc));
        CK(cudaEventRecord(eh1[B], sc));
        { std::lock_guard<std::mutex> lk(S.mu); S.enqueued = B + 1; }
        S.cv.notify_all();
        CK(cudaStreamWaitEvent(sk, eh1[B], 0));
        CK(cudaEventRecord(ek0[B], sk));
        dummy_kernel<<<grid, 256, 0, sk>>>((const uint4*)db, used / 16, dres + B, (unsigned long long)(A.spin_ns * (double)bt.pairs));
        CK(cudaEventRecord(ek1[B], sk));
    }
    kernel_enq_end = now_s();
    for (auto& x : th) x.join();
    double t_decode_done = now_s();
    releaser.join();
    CK(cudaEventSynchronize(ek1[NB - 1])); CK(cudaStreamSynchronize(sk)); CK(cudaStreamSynchronize(sc));
    double t_end = now_s();
    double u1, s1; cpu_split_self(u1, s1);

    // --- metrics ---
    std::vector<double> h2d_ms(NB), k_ms(NB), gap_ms(NB, 0), k_end_s(NB), k_start_s(NB);
    for (uint32_t B = 0; B < NB; B++) {
        float f; CK(cudaEventElapsedTime(&f, eh0[B], eh1[B])); h2d_ms[B] = f;
        CK(cudaEventElapsedTime(&f, ek0[B], ek1[B])); k_ms[B] = f;
        if (B) { CK(cudaEventElapsedTime(&f, ek1[B - 1], ek0[B])); gap_ms[B] = f; }
        CK(cudaEventElapsedTime(&f, ek0[0], ek1[B])); k_end_s[B] = f * 1e-3;
        CK(cudaEventElapsedTime(&f, ek0[0], ek0[B])); k_start_s[B] = f * 1e-3;
    }
    std::vector<unsigned long long> dxor(NB); CK(cudaMemcpy(dxor.data(), dres, NB * 8, cudaMemcpyDeviceToHost));
    uint64_t verify_bad = 0; if (A.verify) for (uint32_t B = 0; B < NB; B++) verify_bad += (dxor[B] != host_xor[B]);

    TStats T; for (auto& x : TS) { T.io += x.io; T.zstd += x.zstd; T.pack += x.pack; T.wait += x.wait; T.cpu += x.cpu; T.crc += x.crc; T.raw += x.raw; T.comp += x.comp; T.reads += x.reads; T.blocks += x.blocks;
                                   T.bs.nN += x.bs.nN; T.bs.trunc += x.bs.trunc; T.bs.unk_bc += x.bs.unk_bc; }
    double wall = t_end - t_start;
    uint64_t pairs = P.pairs;
    // steady state: from the end of kernel 2 to the end of the last kernel (kernel clock) -> pairs of batches 3..NB-1
    int skip = NB > 6 ? 3 : (NB > 2 ? 1 : 0);
    double steady_pairs = 0; for (uint32_t B = skip; B < NB; B++) steady_pairs += P.batches[B].pairs;
    double steady_s = skip ? (k_end_s[NB - 1] - k_end_s[skip - 1]) : k_end_s[NB - 1];
    // idle gaps over the steady window
    double gap_sum = 0, kern_sum = 0; for (uint32_t B = skip ? skip : 1; B < NB; B++) { gap_sum += gap_ms[B]; kern_sum += k_ms[B]; }
    double win = skip ? (k_end_s[NB - 1] - k_start_s[skip]) : (k_end_s[NB - 1] - k_start_s[0]);
    double idle_pct = 100.0 * gap_sum / (gap_sum + kern_sum);
    double submit_wait_sum = 0; for (uint32_t B = skip; B < NB; B++) submit_wait_sum += submit_wait[B];
    double h2d_sum = std::accumulate(h2d_ms.begin(), h2d_ms.end(), 0.0), hbytes = 0; for (auto& b : P.batches) hbytes += batch_used_bytes(P, b);
    double cpu_total = (u1 - u0) + (s1 - s0);
    double dec_cpu = T.cpu;
    (void)win; (void)kernel_enq_end;

    printf("{\"bench\":\"m8\",\"label\":\"%s\",\"file\":\"%s\",\"threads\":%d,\"spin_ns\":%.1f,\"dev\":%d,\"gpu\":\"%s\",\"ring\":%d,\"batch_MB\":%zu,"
           "\"qual\":\"%s\",\"max_pairs\":%llu,\"prefetch\":%d,\"dontneed\":%d,\"crc\":%d,\"verify\":%d,\"verify_mismatch_batches\":%llu,"
           "\"epoch_start\":%.2f,\"epoch_end\":%.2f,\"pairs\":%llu,\"batches\":%u,\"R2B\":%u,\"payload_B_per_pair\":%u,\"max_r2_len\":%u,"
           "\"wall_s\":%.3f,\"plan_s\":%.3f,\"setup_s\":%.3f,\"decode_done_s\":%.3f,"
           "\"pairs_per_s_M_total\":%.2f,\"pairs_per_s_M_steady\":%.2f,\"steady_s\":%.3f,"
           "\"gpu_idle_pct_steady\":%.2f,\"gpu_gap_ms_sum\":%.1f,\"gpu_kernel_ms_sum\":%.1f,\"submitter_wait_s_steady\":%.3f,"
           "\"h2d_ms_mean\":%.3f,\"h2d_GBps\":%.2f,\"kernel_ms_mean\":%.3f,"
           "\"host_cpu_s\":%.3f,\"host_user_s\":%.3f,\"host_sys_s\":%.3f,\"host_cores_avg\":%.3f,\"host_core_s_per_Mpair\":%.4f,"
           "\"decode_thread_cpu_s\":%.3f,\"decode_core_s_per_Mpair\":%.4f,"
           "\"thread_s\":{\"io_pread\":%.3f,\"zstd\":%.3f,\"pack\":%.3f,\"crc\":%.3f,\"ring_wait\":%.3f},"
           "\"per_thread_Mpairs_per_s_active\":%.2f,\"per_thread_ns_per_pair\":{\"zstd\":%.2f,\"pack\":%.2f,\"io\":%.2f},"
           "\"comp_bytes\":%llu,\"raw_bytes\":%llu,\"file_B_per_pair\":%.2f,\"file_read_MBps\":%.1f,"
           "\"cache_resident_before\":%.4f,\"cache_resident_after_dontneed\":%.4f,"
           "\"reads_with_N\":%llu,\"truncated\":%llu,\"unknown_bc\":%llu,\"per_batch\":{",
           A.label.c_str(), A.file.c_str(), A.threads, A.spin_ns, A.dev, prop.name, A.ring, A.mb, P.qual_file ? "file" : (P.qsynth ? "synth" : "none"), (unsigned long long)A.max_pairs, A.prefetch, A.dontneed,
           (int)A.crc, (int)A.verify, (unsigned long long)verify_bad, epoch_start, epoch_s(), (unsigned long long)pairs, NB, P.lay.R2B, P.lay.bytes_per_pair(), P.max_r2,
           wall, plan_s, setup_s, t_decode_done - t_start,
           pairs / 1e6 / wall, steady_pairs / 1e6 / steady_s, steady_s, idle_pct, gap_sum, kern_sum, submit_wait_sum,
           h2d_sum / NB, hbytes / 1e9 / (h2d_sum * 1e-3), std::accumulate(k_ms.begin(), k_ms.end(), 0.0) / NB,
           cpu_total, u1 - u0, s1 - s0, cpu_total / wall, cpu_total / (pairs / 1e6), dec_cpu, dec_cpu / (pairs / 1e6),
           T.io, T.zstd, T.pack, T.crc, T.wait,
           pairs / 1e6 / std::max(1e-9, T.io + T.zstd + T.pack + T.crc), T.zstd * 1e9 / pairs, T.pack * 1e9 / pairs, T.io * 1e9 / pairs,
           (unsigned long long)T.comp, (unsigned long long)T.raw, double(T.comp) / pairs, T.comp / 1e6 / wall,
           res_before, res_after, (unsigned long long)T.bs.nN, (unsigned long long)T.bs.trunc, (unsigned long long)T.bs.unk_bc);
    auto arr = [&](const char* name, const std::vector<double>& v, bool last = false) {
        printf("\"%s\":[", name); for (size_t i = 0; i < v.size(); i++) printf("%s%.3f", i ? "," : "", v[i]); printf("]%s", last ? "" : ",");
    };
    std::vector<double> rdy(NB), dsum(NB);
    for (uint32_t B = 0; B < NB; B++) { rdy[B] = S.ready_ts[B] - t_start; dsum[B] = S.decode_sum[B] * 1e3 / P.batches[B].nblocks; }
    arr("ready_s", rdy); arr("decode_ms_per_block_mean", dsum); arr("h2d_ms", h2d_ms); arr("kernel_ms", k_ms); arr("gap_ms", gap_ms); arr("submit_wait_s", submit_wait, true);
    printf("}}\n");
    fflush(stdout);
    for (int i = 0; i < A.ring; i++) cudaFreeHost(S.host[i]);
    for (int i = 0; i < ND; i++) cudaFree(dbuf[i]);
    return A.verify && verify_bad ? 3 : 0;
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (!a.inspect.empty()) return do_inspect(a);
    if (!a.copy_src.empty()) return do_copy(a);
    if (a.bw) return do_bw(a);
    if (!a.resid.empty()) { printf("{\"file\":\"%s\",\"resident_frac\":%.4f}\n", a.resid.c_str(), resident_frac(a.resid.c_str())); return 0; }
    if (a.file.empty()) { fprintf(stderr, "see header of m8_ingest.cu for usage\n"); return 2; }
    return do_run(a);
}
