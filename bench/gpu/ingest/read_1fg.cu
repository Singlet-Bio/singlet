// read_1fg: GPU-node reader for .1fg (prototype). Reads chunks from the file, copies the compressed bytes to the GPU, decodes
// there (nvCOMP batched zstd, or a plain copy for codec raw) into the M8 batch layout, and consumes the batch with a checksum
// kernel. Modes:
//   e2e (default)  reader threads pread batches into a pinned ring; one GPU thread does H2D + decode + consume. One JSON line.
//   --no-decode    same, but only the compressed bytes are copied (file read + H2D): the host cost of feeding the GPU
//   --gpu-decode-bench K   resident decode: K batches stay on the device, decoded --reps times; pure GPU decode rate
//   --verify R1FIFO R2FIFO   after decoding each batch copy it back and compare every pair with the `.1fq` decode text
//                            (FASTQ records produced by `singlet decode FILE --r1 R1FIFO --r2 R2FIFO`)
// Decoded batch layout (struct of arrays over the batch's pairs, as in M8): CBUMI n x 8 B (CB bits 0-31, UMI bits 32-55,
// R2 length bits 56-62, has-exception bit 63); R2 n x R2SLOT B (2 bit per base, first base in the top bits); exception words
// per chunk at a fixed stride.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -I$ND/include -Xcompiler -pthread -o read_1fg read_1fg.cu -L$ND/lib64 -lnvcomp -Xlinker -rpath,$ND/lib64
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <deque>
#include <tuple>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime.h>
#include <nvcomp/zstd.h>
#include "ingest_fmt.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define NV(x) do { nvcompStatus_t s_ = (x); if (s_ != nvcompSuccess) { fprintf(stderr, "%s:%d nvcomp status %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double thr_cpu() { timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static bool pread_full(int fd, void* buf, size_t n, uint64_t off) {
    uint8_t* p = (uint8_t*)buf;
    while (n) { ssize_t r = pread(fd, p, n, off); if (r <= 0) return false; p += r; n -= r; off += r; }
    return true;
}

struct Args {
    std::string file, label, v1, v2;
    int threads = 2, bchunks = 8, ring = 4, dev = 0, crc = 0, dontneed = 1, reps = 20, gpubench = 0;
    bool nodecode = false, verify = false; uint32_t max_chunks = 0;
};

// consume kernel: read every decoded byte, fold into a checksum
__global__ void fold_kernel(const uint64_t* a, size_t na, const uint64_t* b, size_t nb, unsigned long long* out) {
    uint64_t x = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < na; i += (size_t)gridDim.x * blockDim.x) x ^= a[i] * 0x9E3779B97F4A7C15ull + i;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nb; i += (size_t)gridDim.x * blockDim.x) x ^= b[i] * 0xC2B2AE3D27D4EB4Full + i;
    for (int o = 16; o; o >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
    if ((threadIdx.x & 31) == 0) atomicXor(out, (unsigned long long)x);
}

struct BatchInfo { uint32_t c0, nch; uint64_t off, bytes; uint64_t pairs; };

struct Slot {            // pinned ring slot
    uint8_t* buf = nullptr;
    int64_t turn = 0;    // the batch index this slot is next to be filled with (batch k uses slot k % ring when turn == k)
    int64_t ready = -1;  // batch index currently loaded
};

int main(int argc, char** argv) {
    Args A;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--file") A.file = nx(); else if (k == "--threads") A.threads = atoi(nx().c_str());
        else if (k == "--batch-chunks") A.bchunks = atoi(nx().c_str()); else if (k == "--ring") A.ring = atoi(nx().c_str());
        else if (k == "--dev") A.dev = atoi(nx().c_str()); else if (k == "--crc") A.crc = atoi(nx().c_str());
        else if (k == "--dontneed") A.dontneed = atoi(nx().c_str()); else if (k == "--reps") A.reps = atoi(nx().c_str());
        else if (k == "--max-chunks") A.max_chunks = atoi(nx().c_str()); else if (k == "--label") A.label = nx();
        else if (k == "--no-decode") A.nodecode = true; else if (k == "--gpu-decode-bench") A.gpubench = atoi(nx().c_str());
        else if (k == "--verify") { A.verify = true; A.v1 = nx(); A.v2 = nx(); }
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    int fd = open(A.file.c_str(), O_RDONLY); if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    fg::FileHeader H; if (!pread_full(fd, &H, sizeof H, 0) || H.magic != fg::FILE_MAGIC) { fprintf(stderr, "not a .1fg file\n"); return 1; }
    if (A.dontneed) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    fg::Footer F; pread_full(fd, &F, sizeof F, st.st_size - sizeof F);
    std::vector<fg::IndexEntry> idx(F.n_chunks);
    pread_full(fd, idx.data(), idx.size() * sizeof(fg::IndexEntry), F.index_offset);
    if (fg::crc32c(idx.data(), idx.size() * sizeof(fg::IndexEntry)) != F.index_crc32c) { fprintf(stderr, "index crc mismatch\n"); return 1; }
    uint32_t nchunks = F.n_chunks; if (A.max_chunks && A.max_chunks < nchunks) nchunks = A.max_chunks;
    const bool zstd = true;   // temp storage and the nvCOMP path are always set up; sections stored raw are copied instead
    const uint32_t R2B = H.r2_slot; const uint32_t CP = H.chunk_pairs;
    const uint32_t EXC_STRIDE = (uint32_t)fg::align_up(std::max<uint32_t>(H.max_exc_bytes, 8), 256);
    // batches of B consecutive chunks (one contiguous file range each)
    std::vector<BatchInfo> batches; uint64_t total_pairs = 0, total_bytes = 0, max_bytes = 0;
    for (uint32_t c = 0; c < nchunks; c += A.bchunks) {
        BatchInfo b; b.c0 = c; b.nch = std::min<uint32_t>(A.bchunks, nchunks - c); b.off = idx[c].offset;
        b.bytes = idx[c + b.nch - 1].offset + idx[c + b.nch - 1].total_bytes - b.off; b.pairs = 0;
        for (uint32_t j = 0; j < b.nch; j++) b.pairs += idx[c + j].n_pairs;
        batches.push_back(b); total_pairs += b.pairs; total_bytes += b.bytes; max_bytes = std::max(max_bytes, b.bytes);
    }
    const int NB = (int)batches.size(); const int B = A.bchunks;
    uint64_t max_pairs_b = 0; for (auto& b : batches) max_pairs_b = std::max(max_pairs_b, b.pairs);

    CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)); CK(cudaSetDevice(A.dev));
    cudaStream_t s_copy, s_comp; CK(cudaStreamCreateWithFlags(&s_copy, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&s_comp, cudaStreamNonBlocking));
    // pinned ring
    std::vector<Slot> slots(A.ring);
    for (int i = 0; i < A.ring; i++) { CK(cudaHostAlloc((void**)&slots[i].buf, max_bytes + 4096, cudaHostAllocDefault)); slots[i].turn = i; }
    // device double buffers
    struct Dev {
        uint8_t* comp; uint64_t* cbumi; uint8_t* r2; uint8_t* exc;
        const void** cptr; size_t *csz, *ubuf, *uact; void** optr; nvcompStatus_t* stat;
        const void** h_cptr; size_t *h_csz, *h_ubuf, *h_uact; void** h_optr; nvcompStatus_t* h_stat;   // pinned mirrors
        std::vector<std::tuple<void*, const void*, size_t>> d2d; cudaEvent_t h2d, done; bool used = false; int64_t k = -1; int nitems = 0; uint64_t pairs = 0;
        unsigned long long* sum;
    };
    const int ND = 2; std::vector<Dev> dv(ND);
    const int ITEMS = 3 * B;
    size_t temp_bytes = 0; void* d_temp = nullptr;
    nvcompBatchedZstdDecompressOpts_t opts = nvcompBatchedZstdDecompressDefaultOpts;
    for (auto& d : dv) {
        CK(cudaMalloc(&d.comp, max_bytes + 4096));
        CK(cudaMalloc(&d.cbumi, (size_t)B * CP * 8)); CK(cudaMalloc(&d.r2, (size_t)B * CP * R2B)); CK(cudaMalloc(&d.exc, (size_t)B * EXC_STRIDE));
        CK(cudaMalloc(&d.cptr, ITEMS * 8)); CK(cudaMalloc(&d.csz, ITEMS * 8)); CK(cudaMalloc(&d.ubuf, ITEMS * 8)); CK(cudaMalloc(&d.uact, ITEMS * 8));
        CK(cudaMalloc(&d.optr, ITEMS * 8)); CK(cudaMalloc(&d.stat, ITEMS * sizeof(nvcompStatus_t)));
        CK(cudaHostAlloc((void**)&d.h_cptr, ITEMS * 8, 0)); CK(cudaHostAlloc((void**)&d.h_csz, ITEMS * 8, 0)); CK(cudaHostAlloc((void**)&d.h_ubuf, ITEMS * 8, 0));
        CK(cudaHostAlloc((void**)&d.h_uact, ITEMS * 8, 0)); CK(cudaHostAlloc((void**)&d.h_optr, ITEMS * 8, 0)); CK(cudaHostAlloc((void**)&d.h_stat, ITEMS * sizeof(nvcompStatus_t), 0));
        CK(cudaMalloc(&d.sum, 8)); CK(cudaMemset(d.sum, 0, 8));
        CK(cudaEventCreateWithFlags(&d.h2d, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&d.done, cudaEventDisableTiming | cudaEventBlockingSync));
    }
    if (zstd) {
        NV(nvcompBatchedZstdDecompressGetTempSize(ITEMS, (size_t)CP * R2B, opts, &temp_bytes, (size_t)B * CP * (8 + R2B) + (size_t)B * EXC_STRIDE, s_comp));
        CK(cudaMalloc(&d_temp, temp_bytes + 256));
    }

    // chunk headers of a batch resident in a pinned slot (host view)
    auto chunk_hdr = [&](const Slot& s, const BatchInfo& b, uint32_t j) { return (const fg::ChunkHeader*)(s.buf + (idx[b.c0 + j].offset - b.off)); };

    // Enqueue H2D + decode + consume for batch k (slot s -> device slot d). Returns after enqueue (asynchronous).
    auto enqueue = [&](Dev& d, const Slot& s, const BatchInfo& b, int64_t k, bool decode, bool bench_only_decode) {
        CK(cudaMemcpyAsync(d.comp, s.buf, b.bytes, cudaMemcpyHostToDevice, s_copy)); CK(cudaEventRecord(d.h2d, s_copy));
        d.k = k; d.pairs = b.pairs; d.nitems = 0; d.d2d.clear();
        if (!decode) return;
        int n = 0; uint64_t first = 0;
        for (uint32_t j = 0; j < b.nch; j++) {
            const fg::ChunkHeader* ch = chunk_hdr(s, b, j);
            uint64_t base = idx[b.c0 + j].offset - b.off;
            uint8_t* dst[3] = {(uint8_t*)(d.cbumi + first), d.r2 + first * R2B, d.exc + (size_t)j * EXC_STRIDE};
            for (int sec = 0; sec < 3; sec++) {
                if (ch->sec[sec].raw_size == 0) continue;
                const uint8_t* src = d.comp + base + ch->sec[sec].offset;
                if (ch->flags & (1u << sec)) d.d2d.emplace_back(dst[sec], src, ch->sec[sec].raw_size);   // stored raw: device-to-device copy
                else { d.h_cptr[n] = src; d.h_csz[n] = ch->sec[sec].comp_size; d.h_ubuf[n] = ch->sec[sec].raw_size; d.h_optr[n] = dst[sec]; n++; }
            }
            first += ch->n_pairs;
        }
        d.nitems = n;
        CK(cudaStreamWaitEvent(s_comp, d.h2d, 0));
        for (auto& [dd, sr, sz] : d.d2d) CK(cudaMemcpyAsync(dd, sr, sz, cudaMemcpyDeviceToDevice, s_comp));
        if (n) {
            CK(cudaMemcpyAsync(d.cptr, d.h_cptr, n * 8, cudaMemcpyHostToDevice, s_comp)); CK(cudaMemcpyAsync(d.csz, d.h_csz, n * 8, cudaMemcpyHostToDevice, s_comp));
            CK(cudaMemcpyAsync(d.ubuf, d.h_ubuf, n * 8, cudaMemcpyHostToDevice, s_comp)); CK(cudaMemcpyAsync(d.optr, d.h_optr, n * 8, cudaMemcpyHostToDevice, s_comp));
            NV(nvcompBatchedZstdDecompressAsync(d.cptr, d.csz, d.ubuf, d.uact, n, d_temp, temp_bytes, d.optr, opts, d.stat, s_comp));
            CK(cudaMemcpyAsync(d.h_stat, d.stat, n * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost, s_comp));
            CK(cudaMemcpyAsync(d.h_uact, d.uact, n * 8, cudaMemcpyDeviceToHost, s_comp));
        }
        fold_kernel<<<1024, 256, 0, s_comp>>>(d.cbumi, b.pairs, (const uint64_t*)d.r2, b.pairs * R2B / 8, d.sum);
        CK(cudaGetLastError());
        CK(cudaEventRecord(d.done, s_comp));
    };

    auto check_status = [&](Dev& d, const BatchInfo& b, const Slot* s) -> bool {
        bool ok = true;
        for (int i = 0; i < d.nitems; i++) if (d.h_stat[i] != nvcompSuccess || d.h_uact[i] != d.h_ubuf[i]) { ok = false; fprintf(stderr, "batch %lld item %d status %d actual %zu expect %zu\n", (long long)d.k, i, (int)d.h_stat[i], d.h_uact[i], d.h_ubuf[i]); break; }
        return ok;
    };

    // ------------------------------------------------------------------------------------------ GPU-decode benchmark
    if (A.gpubench) {
        int K = std::min(A.gpubench, NB); if (K > ND) K = ND;   // only ND device slots: use the first ND batches
        // load K batches into pinned slots and device, then decode repeatedly
        std::vector<Slot> keep(K);
        for (int k = 0; k < K; k++) {
            keep[k].buf = slots[k].buf; pread_full(fd, keep[k].buf, batches[k].bytes, batches[k].off);
            enqueue(dv[k], keep[k], batches[k], k, true, false);
        }
        CK(cudaStreamSynchronize(s_comp));
        bool ok = true; for (int k = 0; k < K; k++) ok &= check_status(dv[k], batches[k], &keep[k]);
        // build decode-only item lists once (done inside enqueue); re-run only the decode calls
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        uint64_t pairs_once = 0, comp_once = 0, raw_once = 0; for (int k = 0; k < K; k++) { pairs_once += batches[k].pairs; comp_once += batches[k].bytes; }
        for (int k = 0; k < K; k++) { for (int i = 0; i < dv[k].nitems; i++) raw_once += dv[k].h_ubuf[i]; for (auto& t : dv[k].d2d) raw_once += std::get<2>(t); }
        std::vector<double> ms;
        for (int rep = 0; rep < A.reps + 2; rep++) {
            CK(cudaEventRecord(e0, s_comp));
            for (int k = 0; k < K; k++) {
                for (auto& [dd, sr, sz] : dv[k].d2d) CK(cudaMemcpyAsync(dd, sr, sz, cudaMemcpyDeviceToDevice, s_comp));
                if (dv[k].nitems) NV(nvcompBatchedZstdDecompressAsync(dv[k].cptr, dv[k].csz, dv[k].ubuf, dv[k].uact, dv[k].nitems, d_temp, temp_bytes, dv[k].optr, opts, dv[k].stat, s_comp));
            }
            CK(cudaEventRecord(e1, s_comp)); CK(cudaEventSynchronize(e1));
            float m; CK(cudaEventElapsedTime(&m, e0, e1)); if (rep >= 2) ms.push_back(m);
        }
        std::sort(ms.begin(), ms.end()); double med = ms[ms.size() / 2], mn = ms.front();
        printf("{\"tool\":\"read_1fg\",\"mode\":\"gpu_decode_bench\",\"label\":\"%s\",\"file\":\"%s\",\"codec\":\"%s\",\"batches\":%d,\"batch_chunks\":%d,\"pairs_per_iter\":%llu,"
               "\"comp_bytes_per_iter\":%llu,\"raw_bytes_per_iter\":%llu,\"reps\":%d,\"ms_median\":%.3f,\"ms_min\":%.3f,\"decoded_GBps_median\":%.2f,\"decoded_GBps_best\":%.2f,"
               "\"comp_GBps_median\":%.2f,\"Mpairs_per_s_median\":%.1f,\"Mpairs_per_s_best\":%.1f,\"status_ok\":%d,\"nvcomp_temp_bytes\":%zu}\n",
               A.label.c_str(), A.file.c_str(), H.codec == fg::CODEC_ZSTD ? "zstd" : "raw", K, B, (unsigned long long)pairs_once, (unsigned long long)comp_once, (unsigned long long)raw_once, A.reps, med, mn,
               raw_once / (med * 1e-3) / 1e9, raw_once / (mn * 1e-3) / 1e9, comp_once / (med * 1e-3) / 1e9, pairs_once / (med * 1e-3) / 1e6, pairs_once / (mn * 1e-3) / 1e6, (int)ok, temp_bytes);
        return ok ? 0 : 1;
    }

    // ------------------------------------------------------------------------------------------------ pipeline
    std::mutex mu; std::condition_variable cv;
    std::atomic<int64_t> next{0}; std::atomic<uint64_t> crc_bad{0};
    std::atomic<double> cpu_read{0};
    auto reader = [&]() {
        double c0 = thr_cpu();
        for (;;) {
            int64_t k = next.fetch_add(1); if (k >= NB) break;
            Slot& s = slots[k % A.ring];
            { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return s.turn == k; }); }
            const BatchInfo& b = batches[k];
            if (!pread_full(fd, s.buf, b.bytes, b.off)) { fprintf(stderr, "read failed batch %lld\n", (long long)k); exit(1); }
            if (A.crc) {
                for (uint32_t j = 0; j < b.nch; j++) {
                    const fg::ChunkHeader* ch = (const fg::ChunkHeader*)(s.buf + (idx[b.c0 + j].offset - b.off));
                    for (int sec = 0; sec < 3; sec++)
                        if (fg::crc32c(((const uint8_t*)ch) + ch->sec[sec].offset, ch->sec[sec].comp_size) != ch->sec[sec].crc32c) crc_bad++;
                }
            }
            { std::lock_guard<std::mutex> lk(mu); s.ready = k; }
            cv.notify_all();
        }
        double e = thr_cpu(); double o = cpu_read.load(); while (!cpu_read.compare_exchange_weak(o, o + (e - c0))) {}
    };

    // verification state
    FILE *f1 = nullptr, *f2 = nullptr; uint64_t v_pairs = 0, v_bad = 0, v_exc_pairs = 0; std::string v_first_bad;
    if (A.verify) { f1 = fopen(A.v1.c_str(), "r"); f2 = fopen(A.v2.c_str(), "r"); if (!f1 || !f2) { perror("open fifo"); return 1; } static char b1[1 << 22], b2[1 << 22]; setvbuf(f1, b1, _IOFBF, sizeof b1); setvbuf(f2, b2, _IOFBF, sizeof b2); }
    std::vector<uint64_t> h_cu; std::vector<uint8_t> h_r2, h_exc;
    char* line = nullptr; size_t lcap = 0;
    auto getl = [&](FILE* f) -> ssize_t { return getline(&line, &lcap, f); };
    auto verify_batch = [&](Dev& d, const Slot& s, const BatchInfo& b) {
        h_cu.resize(b.pairs); h_r2.resize(b.pairs * R2B); h_exc.resize((size_t)b.nch * EXC_STRIDE);
        CK(cudaMemcpy(h_cu.data(), d.cbumi, b.pairs * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h_r2.data(), d.r2, b.pairs * R2B, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_exc.data(), d.exc, h_exc.size(), cudaMemcpyDeviceToHost));
        static const char ACGT[4] = {'A', 'C', 'G', 'T'};
        uint64_t first = 0; std::string r1s, r2s, ln;
        for (uint32_t j = 0; j < b.nch; j++) {
            const fg::ChunkHeader* ch = chunk_hdr(s, b, j);
            const uint32_t* ex = (const uint32_t*)(h_exc.data() + (size_t)j * EXC_STRIDE); uint32_t ne = ch->n_exc, ei = 0;
            for (uint32_t i = 0; i < ch->n_pairs; i++) {
                uint64_t w = h_cu[first + i]; uint32_t cb = (uint32_t)w, len = (w >> 56) & 127; uint32_t um = (uint32_t)(w >> 32) & 0xFFFFFF;
                std::string cbs(H.bc_len, 'A'), ums(H.umi_len, 'A'); r2s.assign(len, 'A');
                for (uint32_t k = 0; k < H.bc_len; k++) cbs[k] = ACGT[(cb >> (2 * (H.bc_len - 1 - k))) & 3];
                for (uint32_t k = 0; k < H.umi_len; k++) ums[k] = ACGT[(um >> (2 * (H.umi_len - 1 - k))) & 3];
                const uint8_t* r2 = h_r2.data() + (first + i) * R2B;
                for (uint32_t k = 0; k < len; k++) r2s[k] = ACGT[(r2[k >> 2] >> (6 - 2 * (k & 3))) & 3];
                bool hasx = (w >> 63) != 0; if (hasx) v_exc_pairs++;
                uint32_t nex = 0;
                while (ei < ne && (ex[ei] & 0xFFFFF) == i) {
                    uint32_t kind = (ex[ei] >> 20) & 3, pos = ex[ei] >> 22; nex++;
                    if (kind == fg::EXC_R2N) r2s[pos] = 'N'; else if (kind == fg::EXC_UMIN) ums[pos] = 'N';
                    else if (kind == fg::EXC_RESERVED) {} else cbs.assign(H.bc_len, 'N');
                    ei++;
                }
                if (hasx != (nex > 0)) v_bad++;
                r1s.assign(ums, 0, H.bc_start); r1s += cbs; r1s.append(ums, H.bc_start, std::string::npos);
                // compare with the FASTQ records of the .1fq decode: @r<N>, seq, +, quality
                char nm[48]; snprintf(nm, sizeof nm, "@r%llu\n", (unsigned long long)v_pairs);
                bool bad = false; std::string why;
                for (int fi = 0; fi < 2 && !bad; fi++) {
                    FILE* f = fi ? f2 : f1; const std::string& seq = fi ? r2s : r1s;
                    if (getl(f) < 0) { bad = true; why = "eof"; break; }
                    if (strcmp(line, nm) != 0) { bad = true; why = std::string("name ") + line; break; }
                    if (getl(f) < 0) { bad = true; why = "eof"; break; }
                    if (std::string(line, strlen(line) - 1) != seq) { bad = true; why = std::string(fi ? "R2 " : "R1 ") + "got " + line + " want " + seq; break; }
                    getl(f); getl(f);
                    if (strlen(line) != seq.size() + 1) { bad = true; why = "qual length"; break; }
                }
                if (bad) { v_bad++; if (v_first_bad.empty()) v_first_bad = "pair " + std::to_string(v_pairs) + ": " + why; }
                v_pairs++;
            }
            first += ch->n_pairs;
        }
    };

    // GPU thread = main thread
    std::vector<std::thread> th; double w0 = now_s(); rusage r0; getrusage(RUSAGE_SELF, &r0); double cpu_main0 = thr_cpu();
    for (int i = 0; i < A.threads; i++) th.emplace_back(reader);
    std::deque<std::pair<int64_t, int>> inflight;       // (batch, dev slot) whose pinned slot is not yet released
    auto release_done = [&](bool block) {
        while (!inflight.empty()) {
            auto [k, di] = inflight.front();
            cudaError_t q = cudaEventQuery(dv[di].h2d);
            if (q == cudaErrorNotReady) { if (!block) break; CK(cudaEventSynchronize(dv[di].h2d)); }
            else CK(q);
            { std::lock_guard<std::mutex> lk(mu); slots[k % A.ring].turn = k + A.ring; }
            cv.notify_all(); inflight.pop_front();
        }
    };
    bool all_ok = true; double gpu_first = 0; uint64_t ck_pairs = 0;
    for (int64_t k = 0; k < NB; k++) {
        Slot& s = slots[k % A.ring];
        for (;;) {
            { std::unique_lock<std::mutex> lk(mu); if (cv.wait_for(lk, std::chrono::milliseconds(1), [&] { return s.ready == k; })) break; }
            release_done(false);
        }
        Dev& d = dv[k % ND];
        if (d.used) { CK(cudaEventSynchronize(d.done)); if (!A.nodecode) all_ok &= check_status(d, batches[d.k], nullptr); }
        if (A.verify && d.used) { /* verification of the previous use happens right after its own batch below */ }
        d.used = true;
        enqueue(d, s, batches[k], k, !A.nodecode, false);
        inflight.emplace_back(k, (int)(k % ND));
        if (A.verify) {
            CK(cudaEventSynchronize(d.done)); all_ok &= check_status(d, batches[k], &s);
            // the pinned slot is still needed for the chunk headers: verify before releasing
            verify_batch(d, s, batches[k]);
        }
        ck_pairs += batches[k].pairs;
        release_done(false);
        if (k == 0) gpu_first = now_s();
    }
    for (int di = 0; di < ND; di++) if (dv[di].used) { CK(cudaEventSynchronize(dv[di].done)); if (!A.nodecode) all_ok &= check_status(dv[di], batches[dv[di].k], nullptr); }
    CK(cudaStreamSynchronize(s_copy)); CK(cudaStreamSynchronize(s_comp));
    release_done(true);
    double w1 = now_s(); for (auto& t : th) t.join();
    rusage r1; getrusage(RUSAGE_SELF, &r1); double cpu_main = thr_cpu() - cpu_main0;
    auto tv = [](const timeval& t) { return t.tv_sec + t.tv_usec * 1e-6; };
    double cpu = tv(r1.ru_utime) + tv(r1.ru_stime) - tv(r0.ru_utime) - tv(r0.ru_stime);
    double wall = w1 - w0, Mp = total_pairs / 1e6;
    unsigned long long hsum[2] = {0, 0}; CK(cudaMemcpy(&hsum[0], dv[0].sum, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&hsum[1], dv[1].sum, 8, cudaMemcpyDeviceToHost));
    printf("{\"tool\":\"read_1fg\",\"mode\":\"%s\",\"label\":\"%s\",\"file\":\"%s\",\"codec\":\"%s\",\"threads\":%d,\"batch_chunks\":%d,\"ring\":%d,\"crc\":%d,\"pairs\":%llu,\"file_bytes_read\":%llu,"
           "\"bytes_per_pair\":%.3f,\"wall_s\":%.3f,\"Mpairs_per_s\":%.2f,\"file_MBps\":%.1f,\"cpu_s_total\":%.3f,\"cpu_s_per_Mpairs\":%.4f,\"cpu_reader_threads_s_per_Mpairs\":%.4f,"
           "\"cpu_gpu_thread_s_per_Mpairs\":%.4f,\"crc_mismatch\":%llu,\"status_ok\":%d,\"checksum\":\"%016llx%016llx\"",
           A.verify ? "verify" : (A.nodecode ? "read_h2d_only" : "e2e"), A.label.c_str(), A.file.c_str(), H.codec == fg::CODEC_ZSTD ? "zstd" : "raw", A.threads, B, A.ring, A.crc,
           (unsigned long long)total_pairs, (unsigned long long)total_bytes, (double)total_bytes / total_pairs, wall, Mp / wall, total_bytes / 1e6 / wall, cpu, cpu / Mp,
           cpu_read.load() / Mp, cpu_main / Mp, (unsigned long long)crc_bad.load(), (int)all_ok, hsum[0], hsum[1]);
    if (A.verify) printf(",\"verify_pairs\":%llu,\"verify_mismatch\":%llu,\"verify_pairs_with_exceptions\":%llu,\"verify_first_mismatch\":\"%s\"", (unsigned long long)v_pairs, (unsigned long long)v_bad, (unsigned long long)v_exc_pairs, v_first_bad.c_str());
    printf("}\n");
    return (all_ok && crc_bad == 0 && v_bad == 0) ? 0 : 1;
}
