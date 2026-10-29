// dec_1fg: CPU-engine compatibility check for .1fg. Decodes a .1fg on the host into the byte-numeric R1/R2 reads the CPU engine
// consumes (what lib1fq's Reader::decode_payload produces from a .1fq: A,C,G,T = 0..3, N = 4; R1 = barcode placed at bc_start inside
// UMI prefix/suffix; unknown barcode = N), and (with --fq) decodes the source .1fq with the library; both sides hash every read
// (order-independent sum of a position-keyed 64-bit mix), so equality of the two hashes means the CPU engine would see identical bytes.
//   dec_1fg --fg F.1fg [--fq F.1fq] [--threads N] [--nohash]
// One JSON line per side with thread-CPU seconds per M pairs. Build: g++ -O3 -std=c++17 -I<singlet include> -I<zstd> dec_1fg.cpp -lzstd -pthread
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <zstd.h>
#include <singlet/fq/reader.h>
#include "ingest_fmt.h"

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double thr_cpu() { timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static bool pread_full(int fd, void* buf, size_t n, uint64_t off) {
    uint8_t* p = (uint8_t*)buf;
    while (n) { ssize_t r = pread(fd, p, n, off); if (r <= 0) return false; p += r; n -= r; off += r; }
    return true;
}
static inline uint64_t mix(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33; return x; }
// hash one read pair: index, R1 bytes (r1_len padded to 8), R2 length and bytes (padded to 8)
static inline uint64_t hash_pair(uint64_t gi, const uint8_t* r1, uint32_t l1, const uint8_t* r2, uint32_t l2) {
    uint64_t h = mix(gi + 0x9E3779B97F4A7C15ull);
    uint8_t tmp[8];
    for (uint32_t i = 0; i < l1; i += 8) { uint64_t w = 0; uint32_t n = l1 - i < 8 ? l1 - i : 8; std::memset(tmp, 0, 8); std::memcpy(tmp, r1 + i, n); std::memcpy(&w, tmp, 8); h = (h ^ w) * 0x100000001b3ull; h ^= h >> 29; }
    h ^= l2 * 0x9E3779B97F4A7C15ull;
    for (uint32_t i = 0; i < l2; i += 8) { uint64_t w = 0; uint32_t n = l2 - i < 8 ? l2 - i : 8; std::memset(tmp, 0, 8); std::memcpy(tmp, r2 + i, n); std::memcpy(&w, tmp, 8); h = (h ^ w) * 0x100000001b3ull; h ^= h >> 29; }
    return mix(h);
}

int main(int argc, char** argv) {
    std::string fgp, fqp; int threads = 4; bool nohash = false, meta = false;
    for (int i = 1; i < argc; i++) { std::string k = argv[i]; if (k == "--meta") meta = true;
        if (k == "--fg") fgp = argv[++i]; else if (k == "--fq") fqp = argv[++i]; else if (k == "--threads") threads = atoi(argv[++i]); else if (k == "--nohash") nohash = true; }
    if (meta) {   // print the header fields and the embedded metadata JSON, then exit
        int fd = open(fgp.c_str(), O_RDONLY); fg::FileHeader H; pread_full(fd, &H, sizeof H, 0);
        std::vector<uint8_t> c(H.meta_comp_size), r(H.meta_raw_size + 1); pread_full(fd, c.data(), c.size(), H.meta_offset);
        size_t n = ZSTD_decompress(r.data(), r.size(), c.data(), c.size()); r[ZSTD_isError(n) ? 0 : n] = 0;
        printf("{\"header\":{\"version\":%u,\"codec\":%u,\"level\":%u,\"n_pairs\":%llu,\"n_pairs_in\":%llu,\"n_dropped\":%llu,\"n_chunks\":%u,\"r2_slot\":%u,\"bc_len\":%u,\"umi_len\":%u,\"bc_start\":%u,\"r1_len\":%u,\"meta_raw_size\":%u},\"meta_decompressed_ok\":%d,\"meta\":%s}\n",
               H.version, H.codec, H.codec_level, (unsigned long long)H.n_pairs, (unsigned long long)H.n_pairs_in, (unsigned long long)H.n_dropped, H.n_chunks, H.r2_slot, H.bc_len, H.umi_len, H.bc_start, H.r1_len, H.meta_raw_size, (int)!ZSTD_isError(n), (const char*)r.data());
        return 0;
    }
    // ------------------------------------------------------------------------------------------------------ .1fg side
    uint64_t hash_fg = 0, pairs_fg = 0; uint32_t bc_len = 0, umi_len = 0, bc_start = 0, r1_len = 0;
    {
        int fd = open(fgp.c_str(), O_RDONLY); posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        fg::FileHeader H; pread_full(fd, &H, sizeof H, 0);
        off_t sz = lseek(fd, 0, SEEK_END); fg::Footer F; pread_full(fd, &F, sizeof F, sz - sizeof F);
        std::vector<fg::IndexEntry> idx(F.n_chunks); pread_full(fd, idx.data(), idx.size() * sizeof(fg::IndexEntry), F.index_offset);
        std::vector<uint64_t> first(idx.size() + 1, 0); for (size_t c = 0; c < idx.size(); c++) first[c + 1] = first[c] + idx[c].n_pairs;
        bc_len = H.bc_len; umi_len = H.umi_len; bc_start = H.bc_start; r1_len = H.r1_len;
        std::atomic<uint32_t> next{0}; std::atomic<uint64_t> hsum{0}; std::atomic<double> cpu{0};
        double w0 = now_s();
        auto worker = [&]() {
            double c0 = thr_cpu(); ZSTD_DCtx* dctx = ZSTD_createDCtx(); std::vector<uint8_t> buf, sec[3]; std::vector<uint8_t> r1b, r2b; uint64_t hs = 0;
            for (;;) {
                uint32_t c = next.fetch_add(1); if (c >= idx.size()) break;
                buf.resize(idx[c].total_bytes); pread_full(fd, buf.data(), buf.size(), idx[c].offset);
                const fg::ChunkHeader* ch = (const fg::ChunkHeader*)buf.data(); uint32_t n = ch->n_pairs;
                const uint8_t* ptr[3];
                for (int s = 0; s < 3; s++) {
                    const uint8_t* src = buf.data() + ch->sec[s].offset;
                    if (ch->flags & (1u << s) || ch->sec[s].raw_size == 0) ptr[s] = src;
                    else { sec[s].resize(ch->sec[s].raw_size); size_t r = ZSTD_decompressDCtx(dctx, sec[s].data(), sec[s].size(), src, ch->sec[s].comp_size); if (ZSTD_isError(r) || r != sec[s].size()) { fprintf(stderr, "zstd error chunk %u\n", c); exit(1); } ptr[s] = sec[s].data(); }
                }
                const uint64_t* cu = (const uint64_t*)ptr[0]; const uint8_t* r2 = ptr[1]; const uint32_t* ex = (const uint32_t*)ptr[2]; uint32_t ne = ch->n_exc, ei = 0;
                r1b.assign(r1_len, 0); r2b.assign(H.r2_slot * 4, 0);
                for (uint32_t i = 0; i < n; i++) {
                    uint64_t w = cu[i]; uint32_t cb = (uint32_t)w, um = (uint32_t)(w >> 32) & 0xFFFFFF, len = (w >> 56) & 127;
                    // R1 = UMI[0:bc_start] CB UMI[bc_start:]
                    uint32_t u = 0, o = 0;
                    for (; o < bc_start; o++, u++) r1b[o] = (um >> (2 * (umi_len - 1 - u))) & 3;
                    for (uint32_t k = 0; k < bc_len; k++) r1b[o + k] = (cb >> (2 * (bc_len - 1 - k))) & 3;
                    o += bc_len;
                    for (; u < umi_len; o++, u++) r1b[o] = (um >> (2 * (umi_len - 1 - u))) & 3;
                    const uint8_t* rp = r2 + (size_t)i * H.r2_slot;
                    for (uint32_t k = 0; k < len; k++) r2b[k] = (rp[k >> 2] >> (6 - 2 * (k & 3))) & 3;
                    while (ei < ne && (ex[ei] & 0xFFFFF) == i) {
                        uint32_t kind = (ex[ei] >> 20) & 3, pos = ex[ei] >> 22;
                        if (kind == fg::EXC_R2N) r2b[pos] = 4;
                        else if (kind == fg::EXC_UMIN) r1b[pos < bc_start ? pos : pos + bc_len] = 4;
                        else if (kind == fg::EXC_RESERVED) {}
                        else for (uint32_t k = 0; k < bc_len; k++) r1b[bc_start + k] = 4;
                        ei++;
                    }
                    if (!nohash) hs += hash_pair(first[c] + i, r1b.data(), r1_len, r2b.data(), len);
                }
            }
            ZSTD_freeDCtx(dctx);
            hsum += hs; double e = thr_cpu() - c0; double o = cpu.load(); while (!cpu.compare_exchange_weak(o, o + e)) {}
        };
        std::vector<std::thread> th; for (int t = 0; t < threads; t++) th.emplace_back(worker); for (auto& t : th) t.join();
        double w1 = now_s(); hash_fg = hsum.load(); pairs_fg = first.back();
        printf("{\"side\":\"1fg_host_decode\",\"file\":\"%s\",\"threads\":%d,\"pairs\":%llu,\"hash\":\"%016llx\",\"wall_s\":%.3f,\"cpu_s\":%.3f,\"cpu_s_per_Mpairs\":%.4f,\"Mpairs_per_s\":%.2f,\"hashing\":%d}\n",
               fgp.c_str(), threads, (unsigned long long)pairs_fg, (unsigned long long)hash_fg, w1 - w0, cpu.load(), cpu.load() / (pairs_fg / 1e6), pairs_fg / 1e6 / (w1 - w0), (int)!nohash);
        fflush(stdout);
    }
    // ------------------------------------------------------------------------------------------------------ .1fq side
    if (!fqp.empty()) {
        using namespace singlet::fq;
        Reader rd; rd.open(fqp.c_str()); int fd = open(fqp.c_str(), O_RDONLY); posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        uint32_t nb = rd.block_count(); std::vector<uint64_t> first(nb + 1, 0); std::vector<BlockHeader> bhs(nb);
        for (uint32_t b = 0; b < nb; b++) { pread_full(fd, &bhs[b], sizeof(BlockHeader), rd.block_offset(b)); first[b + 1] = first[b] + bhs[b].n_reads; }
        std::atomic<uint32_t> next{0}; std::atomic<uint64_t> hsum{0}, dropped{0}; std::atomic<double> cpu{0};
        double w0 = now_s();
        auto worker = [&]() {
            double c0 = thr_cpu(); std::vector<uint8_t> comp, raw; uint64_t hs = 0, dr = 0;
            for (;;) {
                uint32_t b = next.fetch_add(1); if (b >= nb) break;
                comp.resize(bhs[b].compressed_size); pread_full(fd, comp.data(), comp.size(), rd.block_offset(b) + sizeof(BlockHeader));
                raw.resize((size_t)bhs[b].n_reads * 400);
                size_t rs = compress::decompress_block(Codec::ZSTD, comp.data(), comp.size(), raw.data(), raw.size());
                DecodedBlock blk; rd.decode_payload(raw.data(), rs, bhs[b], blk);
                for (uint32_t i = 0; i < blk.n_reads; i++) {
                    uint32_t l1 = blk.r1_len(i), l2 = blk.r2_len(i);
                    if (l1 == 0 || l2 == 0 || l2 > 1024) { dr++; continue; }
                    if (!nohash) hs += hash_pair(first[b] + i, blk.r1_seq(i), l1, blk.r2_seq(i), l2);
                }
            }
            hsum += hs; dropped += dr; double e = thr_cpu() - c0; double o = cpu.load(); while (!cpu.compare_exchange_weak(o, o + e)) {}
        };
        std::vector<std::thread> th; for (int t = 0; t < threads; t++) th.emplace_back(worker); for (auto& t : th) t.join();
        double w1 = now_s(); uint64_t pairs = first.back();
        printf("{\"side\":\"1fq_library_decode\",\"file\":\"%s\",\"threads\":%d,\"pairs\":%llu,\"dropped\":%llu,\"hash\":\"%016llx\",\"wall_s\":%.3f,\"cpu_s\":%.3f,\"cpu_s_per_Mpairs\":%.4f,\"Mpairs_per_s\":%.2f,\"hashing\":%d,\"hash_equal_to_1fg\":%d}\n",
               fqp.c_str(), threads, (unsigned long long)pairs, (unsigned long long)dropped.load(), (unsigned long long)hsum.load(), w1 - w0, cpu.load(), cpu.load() / (pairs / 1e6), pairs / 1e6 / (w1 - w0), (int)!nohash,
               (int)(!nohash && hsum.load() == hash_fg && pairs == pairs_fg && dropped.load() == 0));
    }
    return 0;
}
