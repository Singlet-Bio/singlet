// E13 (docs/gpu-engine/fable/E13-ingest.md; SINGLET_ENGINE_GPU_INGEST=1): STAR's prefill takes the sample's reads straight from the .1fq
// into the packed reads (singlet/engine_gpu_ingest.h) instead of reading back the FASTQ text the pipeline used to write.
//   Phase A (decodeReads): one pass over the blocks, on the job's threads, each block into its own ranges of the final arrays (the
//     pipeline's head scan counted every block's reads and bases): the cDNA codes and offsets (the engine's input), the cDNA quality as
//     2-bit codes, the barcode read as 3-bit codes, and the bases per quality code. It needs nothing from STAR: the pipeline runs it
//     right after its head scan, beside its reference load (startParent), and STAR's process inherits the result through fork().
//   Phase B (barcodes): what needs STAR's barcode layout and whitelist: E10's per-read words (CR/UR keys, low-quality bits), E11's quality
//     counts, R11's exact CB histogram.
// The bytes are those the FASTQ path gave STAR (read_source.h format_block with the file-mode options: no 5' clip, no length guard; reads
// with an empty mate dropped; R1's quality 'F'; R2's the stored quality + 33), read by the same RunSet decoder.
// No STAR headers here (the .1fq headers and STAR's IncludeDefine.h do not mix); STAR's own per-read work (the CellRanger4 clip) is in
// SeqCacheGpuPrefill.cpp.
#ifdef SINGLET_HAVE_ENGINE_GPU
#include "singlet/engine_gpu_ingest.h"
#include "singlet/engine_gpu_ingest_fused.h"
#include "singlet/fq/read_source.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace singlet_engine_gpu {
namespace ingest {

namespace {
// The fused pass (decodeScan): the head scan's options and per-block results, and the fixed cDNA length that bounds the arrays
struct Fused {
    const singlet::fq::RunSet::HeadScanOptions* so = nullptr;
    std::vector<singlet::fq::RunSet::BlockHead>* heads = nullptr;
    uint32_t r2Fixed = 0;
};

bool decodeImpl(int threads, PackedReads& out, std::string& why, const Fused* fz) {
    const auto t0 = std::chrono::steady_clock::now();
    const Plan& pl = plan();
    if (!(fz || pl.on) || !pl.runs) { why = "no head-scan plan"; return false; }
    const singlet::fq::RunSet& runs = *pl.runs;
    const uint32_t nB = pl.nBlocks;
    if (fz ? (runs.block_count() != nB || fz->heads->size() != (size_t) nB)
           : (runs.block_count() != nB || pl.first.size() != (size_t) nB + 1 || pl.baseAt.size() != (size_t) nB + 1 || pl.nameAt.size() < nB)) {
        why = "the head scan's plan does not match the .1fq"; return false;
    }
    if (pl.r1L == 0 || pl.r1L > 63) { why = "barcode read length " + std::to_string(pl.r1L); return false; }
    out.release();
    // Fused: every block's offsets are found as the blocks decode (block b is written once the counts of blocks 0 .. b are known), into
    // arrays sized for every stored read at the fixed cDNA length and cut to the reads and bases given to STAR at the end (the untouched
    // tail of a reservation takes no memory). Otherwise the head scan's plan gives them.
    uint64_t N = 0, B = 0, Q = 0;
    if (fz) {
        out.first.assign((size_t) nB + 1, 0); out.baseAt.assign((size_t) nB + 1, 0); out.qualAt.assign((size_t) nB + 1, 0);
        out.nameAt.assign((size_t) nB + 1, 0);
        // the bound on a block's cDNA bases: the header's read length (trimming only shortens reads), else four per byte of its decompressed
        // payload (2-bit codes), from the ZSTD frame header
        try {
            for (uint32_t b = 0; b < nB; b++) {
                out.nameAt[b] = runs.block_read_offset(b);
                const uint64_t nr = runs.block_reads(b);
                N += nr;
                if (fz->r2Fixed) B += nr * fz->r2Fixed;
                else {
                    const uint64_t raw = runs.block_raw_size(b);
                    if (raw == 0) { why = "block " + std::to_string(b) + " records no decompressed size"; return false; }
                    B += 4 * raw;
                }
            }
        } catch (const std::exception& e) { why = e.what(); return false; }
        out.nameAt[nB] = N;
        Q = B / 4 + nB;
    } else {
        N = pl.reads(); B = pl.bases();
        out.first = pl.first; out.baseAt = pl.baseAt; out.nameAt = pl.nameAt;
        out.qualAt.assign((size_t) nB + 1, 0);
        for (uint32_t b = 0; b < nB; b++) out.qualAt[b + 1] = out.qualAt[b] + (pl.baseAt[b + 1] - pl.baseAt[b] + 3) / 4;
        Q = out.qualAt[nB];
    }
    out.n = N;
    out.r1L = pl.r1L; out.r1w = (pl.r1L + 20) / 21;
    try {   // the fused pass's reservations are bounds: a refusal falls back to the two passes
        out.seq = std::make_shared<PackedSeq>();
        out.seq->w.resize(std::max<uint64_t>(1, Q));   // 2 bits per base, the quality plane's blocks
        out.seq->runs.resize(nB);
        out.off.resize(N + 1);
        out.qual.resize(std::max<uint64_t>(1, Q));
        out.r1.resize(std::max<uint64_t>(1, N * out.r1w));
    } catch (const std::bad_alloc&) { out.release(); why = "the packed reads' arrays could not be reserved"; return false; }
    out.off[0] = 0;
    const std::vector<uint64_t>& F = fz ? out.first : pl.first;
    const std::vector<uint64_t>& BA = fz ? out.baseAt : pl.baseAt;
    // fused: the counts per block as they arrive, and the frontier below which every block's offsets are known
    std::vector<uint64_t> cntN(fz ? nB : 0), cntB(fz ? nB : 0);
    std::vector<uint8_t> ready(fz ? nB : 0, 0);
    uint32_t front = 0;
    std::mutex fmu;
    std::condition_variable fcv;
    std::memcpy(out.qualPhred, pl.qualPhred, 4);
    int codeOf[256];
    for (int k = 0; k < 256; k++) codeOf[k] = -1;
    for (int k = (int) std::max<uint32_t>(1, pl.qualCodes) - 1; k >= 0; k--) codeOf[pl.qualPhred[k]] = k;   // the first code of a Phred wins
    // The quality column as stored (SINGLET_INGEST_QUAL_COPY=0: unpacked to Phred and packed again, as before): a byte of the .1fq's 2-bit
    // codes (first base in the top bits) maps through one table to the byte of ours (first base in the low bits, codeOf of its Phred), and
    // the bases per code come from a histogram of those bytes. A source table with a code outside codeOf decodes that block as before.
    const char* qcE = std::getenv("SINGLET_INGEST_QUAL_COPY");
    const bool qualCopy = !(qcE && qcE[0] == '0');
    const int T = std::max(1, std::min<int>(threads, (int) std::max<uint32_t>(1, nB)));
    std::atomic<uint32_t> next{0};
    std::atomic<bool> stop{false};
    std::mutex mu;
    std::string err;
    struct alignas(64) Acc { uint64_t hist[4] = {0, 0, 0, 0}; };
    std::vector<Acc> acc(T);
    auto fail = [&](const std::string& w) {
        std::lock_guard<std::mutex> lk(mu);
        if (err.empty()) err = w;
        stop = true;
    };
    auto worker = [&](int tid) {
        uint64_t hist[4] = {0, 0, 0, 0};
        singlet::fq::DecodedBlock blk;
        blk.keep_packed_qual = qualCopy;
        const uint8_t* mapFor = nullptr;   // the source table qmap was built for
        bool mapOk = false;
        uint8_t qmap[256];
        uint64_t bytesOf[256];
        for (uint32_t b; !stop && (b = next.fetch_add(1)) < nB;) {
            try {
                if (fz) runs.decode_scan(b, blk, *fz->so, (*fz->heads)[b]);
                else runs.decode(b, blk);
                if (!blk.r2_qual_packed.empty() && blk.r2_qual_packed_phred != mapFor) {
                    mapFor = blk.r2_qual_packed_phred;
                    int cOf[4];
                    mapOk = true;
                    for (int c = 0; c < 4; c++) { cOf[c] = codeOf[mapFor[c]]; mapOk = mapOk && cOf[c] >= 0; }
                    if (mapOk)
                        for (int v = 0; v < 256; v++)
                            qmap[v] = (uint8_t) (cOf[(v >> 6) & 3] | cOf[(v >> 4) & 3] << 2 | cOf[(v >> 2) & 3] << 4 | cOf[v & 3] << 6);
                }
                if (!blk.r2_qual_packed.empty() && !mapOk) {   // a code outside the quality mode: decoded as before, which finds it if used
                    blk.keep_packed_qual = false;
                    runs.decode(b, blk);
                    blk.keep_packed_qual = true;
                }
            } catch (const std::exception& ex) {
                fail(std::string("block ") + std::to_string(b) + ": " + ex.what());
                return;
            }
            if (fz) {   // publish this block's counts, advance the frontier, and wait until it passes this block
                const auto& h = (*fz->heads)[b];
                bool over = false;
                {
                    std::unique_lock<std::mutex> lk(fmu);
                    cntN[b] = h.n_out; cntB[b] = h.bases_out; ready[b] = 1;
                    while (front < nB && ready[front]) {
                        out.first[front + 1] = out.first[front] + cntN[front];
                        out.baseAt[front + 1] = out.baseAt[front] + cntB[front];
                        out.qualAt[front + 1] = out.qualAt[front] + (cntB[front] + 3) / 4;
                        if (out.first[front + 1] > N || out.baseAt[front + 1] > B || out.qualAt[front + 1] > Q) { over = true; break; }
                        front++;
                    }
                    fcv.notify_all();
                    while (!over && front <= b && !stop) fcv.wait_for(lk, std::chrono::milliseconds(20));
                }
                if (over) { fail("block counts beyond the reads and bases the block headers allow"); fcv.notify_all(); return; }
                if (stop) return;
            }
            const uint64_t rEnd = F[b + 1], posEnd = BA[b + 1];
            uint64_t r = F[b], pos = BA[b], k = 0;
            uint8_t* qd = out.qual.data() + out.qualAt[b];
            PackedSeqWriter sw(out.seq->w.data() + out.qualAt[b], BA[b], &out.seq->runs[b]);
            uint8_t qacc = 0;
            const bool packedQ = !blk.r2_qual_packed.empty();
            const bool hasQ = !blk.r2_qual.empty();
            const uint8_t* qs = packedQ ? blk.r2_qual_packed.data() : nullptr;   // the next read's packed quality
            const uint8_t* qsEnd = qs + blk.r2_qual_packed.size();
            if (packedQ) std::memset(bytesOf, 0, sizeof bytesOf);
            for (uint32_t i = 0; i < blk.n_reads; i++) {
                const uint16_t L1 = blk.r1_len(i), L2 = blk.r2_len(i);
                const uint8_t* q0 = qs;
                if (packedQ) qs += singlet::fq::pack::packed_bytes(L2);   // every read's quality is stored, those not given to STAR too
                if (L1 == 0 || L2 == 0) continue;   // not given to STAR (format_block)
                if (r >= rEnd || pos + L2 > posEnd) { fail("block " + std::to_string(b) + " has more reads or bases than the head scan counted"); return; }
                if (L1 != pl.r1L) { fail("barcode read length " + std::to_string(L1) + " in block " + std::to_string(b)); return; }
                // cDNA: codes and offsets
                sw.put((const uint8_t*) blk.r2_seq(i), L2);
                if (sw.bad) { fail("block " + std::to_string(b) + ": a cDNA base code other than A, C, G, T or N"); return; }
                out.off[r + 1] = pos + L2;
                if (packedQ) {
                    if (qs > qsEnd) { fail("block " + std::to_string(b) + ": the quality column is shorter than its reads"); return; }
                    // four bases per source byte; m codes of the current output byte are pending in qacc
                    const unsigned m = (unsigned) (k & 3), sh = 2 * m;
                    const uint32_t full = L2 >> 2, rem = L2 & 3;
                    for (uint32_t j = 0; j < full; j++) {
                        const uint8_t v = qmap[q0[j]];
                        bytesOf[v]++;
                        const uint32_t cmb = (uint32_t) qacc | (uint32_t) v << sh;
                        qd[k >> 2] = (uint8_t) cmb;
                        qacc = (uint8_t) (cmb >> 8);
                        k += 4;
                    }
                    if (rem) {
                        const uint8_t v = (uint8_t) (qmap[q0[full]] & ((1u << (2 * rem)) - 1));
                        for (uint32_t j = 0; j < rem; j++) hist[(v >> (2 * j)) & 3]++;
                        const uint32_t cmb = (uint32_t) qacc | (uint32_t) v << sh;
                        if (m + rem >= 4) { qd[k >> 2] = (uint8_t) cmb; qacc = (uint8_t) (cmb >> 8); }
                        else qacc = (uint8_t) cmb;
                        k += rem;
                    }
                } else {
                // cDNA quality: 2-bit codes (a read without stored quality is 'F', Phred 37, as format_block writes it)
                const uint8_t* q = hasQ ? blk.r2_quality(i) : nullptr;
                for (uint32_t j = 0; j < L2; j++) {
                    const uint8_t ph = q ? q[j] : (uint8_t) ('F' - 33);
                    const int c = codeOf[ph];
                    if (c < 0) { fail("a quality value (" + std::to_string(ph) + ") outside the .1fq's quality mode"); return; }
                    hist[c]++;
                    qacc |= (uint8_t) (c << (2 * (k & 3)));
                    if ((k & 3) == 3) { qd[k >> 2] = qacc; qacc = 0; }
                    k++;
                }
                }
                // barcode read: 3-bit codes, 21 per word
                const uint8_t* s1 = blk.r1_seq(i);
                uint64_t* rw = out.r1.data() + r * out.r1w;
                for (uint32_t x = 0, j = 0; x < out.r1w; x++) {   // a word at a time
                    uint64_t v = 0;
                    for (uint32_t e = std::min<uint32_t>(L1, j + 21), s = 0; j < e; j++, s += 3) v |= (uint64_t) std::min<uint8_t>(s1[j], 4) << s;
                    rw[x] = v;
                }
                pos += L2;
                r++;
            }
            if ((k & 3) != 0) qd[k >> 2] = qacc;
            sw.finish();
            if (packedQ)
                for (int v = 0; v < 256; v++)
                    if (bytesOf[v]) for (int j = 0; j < 4; j++) hist[(v >> (2 * j)) & 3] += bytesOf[v];
            if (r != rEnd || pos != posEnd) { fail("block " + std::to_string(b) + " has fewer reads or bases than the head scan counted"); return; }
        }
        for (int c = 0; c < 4; c++) acc[tid].hist[c] = hist[c];
    };
    {
        std::vector<std::thread> th;
        for (int t = 1; t < T; t++) th.emplace_back(worker, t);
        worker(0);
        for (auto& x : th) x.join();
    }
    if (!err.empty()) { why = err; out.release(); return false; }
    if (fz) {   // the reads and bases given to STAR
        N = out.first[nB];
        out.n = N;
        out.off.resize(N + 1);
        out.qual.resize(std::max<uint64_t>(1, out.qualAt[nB]));
        out.r1.resize(std::max<uint64_t>(1, N * out.r1w));
        out.seq->w.resize(std::max<uint64_t>(1, out.qualAt[nB]));
    }
    out.seq->baseAt = out.baseAt; out.seq->byteAt = out.qualAt; out.seq->total = out.baseAt[nB];
    for (const Acc& a : acc)
        for (int c = 0; c < 4; c++) out.qualHist[c] += a.hist[c];
    out.sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}
}  // namespace

bool decodeReads(int threads, PackedReads& out, std::string& why) { return decodeImpl(threads, out, why, nullptr); }

bool barcodes(const BarcodeConfig& cfg, PackedReads& out, std::string& why) {
    const uint64_t N = out.n;
    if (out.off.size() != N + 1 || out.r1w == 0) { why = "no decoded reads"; return false; }
    if (cfg.cbL == 0 || cfg.cbL > 21 || cfg.umiL == 0 || cfg.umiL > 21) { why = "CB/UMI over 21 bases"; return false; }
    const uint64_t L1 = out.r1L;
    const uint64_t cbS0 = cfg.cbS0, cbL = cfg.cbL, umiS0 = cfg.umiS0, umiL = cfg.umiL, cbumiL = cfg.cbumiL, bL = cfg.bL;
    // SoloReadBarcode::getCBandUMI for CB_UMI_Simple: a barcode read shorter than cbumiL is padded with N when the length is not fixed
    // (bL 0); STAR rejects a read whose length is not bL. Every barcode read here has the same length, so the layout holds for all or none.
    const uint64_t S = (bL == 0 && L1 < cbumiL) ? cbumiL : L1;
    const bool layoutBad = (bL > 0 && L1 != bL) || cbS0 > S || umiS0 > S;
    const uint32_t nw = cfg.nw;
    const uint64_t stride = 2 + (uint64_t) nw;
    out.nw = 0;
    NoInitVector<uint64_t>().swap(out.aux);
    if (nw > 0 && !layoutBad) { out.nw = nw; out.aux.resize(N * stride); }
    // the low-quality bits per quality code
    bool lowQ[4];
    for (int c = 0; c < 4; c++) lowQ[c] = (int) out.qualPhred[c] < cfg.minQ;
    // and per quality byte (four codes, the first base in the low bits): the four bits (SINGLET_INGEST_QUAL_COPY=0: base by base, as before)
    const char* qcE = std::getenv("SINGLET_INGEST_QUAL_COPY");
    const bool lowTable = !(qcE && qcE[0] == '0');
    uint8_t lowOf[256];
    for (int v = 0; v < 256; v++) {
        lowOf[v] = 0;
        for (int j = 0; j < 4; j++) if (lowQ[(v >> (2 * j)) & 3]) lowOf[v] |= (uint8_t) (1u << j);
    }
    // two quality bytes (eight codes) at a time: the eight bits; no low code at all (a 'F'-only file): no quality words to fill
    const bool anyLow = lowQ[0] || lowQ[1] || lowQ[2] || lowQ[3];
    std::vector<uint8_t> lowOf16(lowTable ? 65536 : 0);
    for (size_t v = 0; v < lowOf16.size(); v++) lowOf16[v] = (uint8_t) (lowOf[v & 255] | lowOf[v >> 8] << 4);
    // the CB and UMI spans straight from the barcode read's words (same 3-bit codes, 21 per word) when they lie inside the read
    const bool cbFast = cbS0 + cbL <= L1, umiFast = umiS0 + umiL <= L1;
    auto span = [&](const uint64_t* rw, uint64_t s0, uint64_t len) {
        const uint64_t w0 = s0 / 21, o0 = s0 % 21, n0 = 21 - o0;
        uint64_t v = rw[w0] >> (3 * o0);
        if (len > n0) v |= rw[w0 + 1] << (3 * n0);
        return v & ((1ull << (3 * len)) - 1);
    };
    uint64_t umiN = 0, umiOne = 0;   // per UMI base: the N bit (code 4) and the low bit, for the R11 test below
    for (uint64_t j = 0; j < umiL; j++) { umiN |= 4ull << (3 * j); umiOne |= 1ull << (3 * j); }
    const bool r11On = cfg.wl != nullptr && cfg.nWL > 0 && !layoutBad;
    std::vector<uint64_t> dense(r11On ? cfg.nWL : 0, 0);
    const int T = std::max(1, std::min<int>(cfg.threads, (int) std::max<uint64_t>(1, N / 4096)));
    struct alignas(64) Acc { uint64_t r11 = 0; bool tooLong = false; };
    std::vector<Acc> acc(T);
    auto worker = [&](int t) {
        const uint64_t r0 = N * t / T, rE = N * (t + 1) / T;
        uint32_t b = r0 < N ? out.block(r0) : 0;
        uint64_t lastCr = ~0ull, run = 0, nr11 = 0;
        int64_t lastW = -1;
        auto flush = [&]() { if (run && lastW >= 0) __atomic_fetch_add(&dense[(size_t) lastW], run, __ATOMIC_RELAXED); run = 0; };
        auto pack = [&](uint64_t r, uint64_t s0, uint64_t len) {
            const uint64_t e0 = std::min(s0 + len, S);
            uint64_t v = 0;
            for (uint64_t j = s0; j < e0 && j - s0 < 21; j++) v |= (uint64_t) (j < L1 ? out.r1code(r, (uint32_t) j) : 4) << (3 * (j - s0));
            return v;
        };
        for (uint64_t r = r0; r < rE; r++) {
            if (layoutBad) break;
            while (r >= out.first[b + 1]) b++;
            const uint64_t* rw = out.r1.data() + r * out.r1w;
            const uint64_t cr = cbFast ? span(rw, cbS0, cbL) : pack(r, cbS0, cbL), ur = umiFast ? span(rw, umiS0, umiL) : pack(r, umiS0, umiL);
            if (out.nw) {
                uint64_t* w = out.aux.data() + r * stride;
                w[0] = cr; w[1] = ur;
                uint64_t* q = w + 2;
                for (uint32_t x = 0; x < nw; x++) q[x] = 0;
                const uint64_t s0 = out.off[r], L = out.off[r + 1] - s0;
                if (L > 64 * (uint64_t) nw) { acc[t].tooLong = true; continue; }
                if (!anyLow) {
                } else if (lowTable) {
                    // the read's bases k0 .. k0 + L of its block: single bases up to a byte boundary, then a byte (four codes) at a time
                    const uint64_t k0 = s0 - out.baseAt[b];
                    const uint8_t* qb = out.qual.data() + out.qualAt[b];
                    uint64_t j = 0;
                    for (; j < L && ((k0 + j) & 3) != 0; j++)
                        if (lowQ[out.qcode(b, s0 + j)]) q[j >> 6] |= 1ull << (j & 63);
                    for (; j + 32 <= L; j += 32) {   // eight bytes, 32 codes: a 32-bit mask
                        const uint8_t* p = qb + ((k0 + j) >> 2);
                        const uint64_t mk = (uint64_t) lowOf16[p[0] | p[1] << 8] | (uint64_t) lowOf16[p[2] | p[3] << 8] << 8
                                          | (uint64_t) lowOf16[p[4] | p[5] << 8] << 16 | (uint64_t) lowOf16[p[6] | p[7] << 8] << 24;
                        if (!mk) continue;
                        const unsigned o = (unsigned) (j & 63);
                        q[j >> 6] |= mk << o;
                        if (o > 32) q[(j >> 6) + 1] |= mk >> (64 - o);
                    }
                    for (; j + 4 <= L; j += 4) {
                        const uint64_t mk = lowOf[qb[(k0 + j) >> 2]];
                        if (!mk) continue;
                        const unsigned o = (unsigned) (j & 63);
                        q[j >> 6] |= mk << o;
                        if (o > 60) q[(j >> 6) + 1] |= mk >> (64 - o);
                    }
                    for (; j < L; j++)
                        if (lowQ[out.qcode(b, s0 + j)]) q[j >> 6] |= 1ull << (j & 63);
                } else {
                    for (uint64_t j = 0; j < L; j++)
                        if (lowQ[out.qcode(b, s0 + j)]) q[j >> 6] |= 1ull << (j & 63);
                }
            }
            // R11 (hostloop_gpu.cu k_r11 on the same keys): a UMI without N that is not a homopolymer, an N-free CB exact in the whitelist.
            // The .1fq is barcode-sorted, so consecutive reads mostly share their CB: one whitelist lookup per run of equal keys.
            if (r11On) {
                // no N (bit 2 of every code clear) and not one code repeated umiL times
                const bool ok = (ur & umiN) == 0, homo = ur == (ur & 7) * umiOne;
                if (ok && !homo) {
                    if (cr != lastCr) {
                        flush();
                        lastCr = cr; lastW = -1;
                        uint64_t key = 0;
                        bool nfree = true;
                        for (uint64_t j = 0; j < cbL; j++) {
                            const uint64_t c = (cr >> (3 * j)) & 7;
                            if (c > 3) { nfree = false; break; }
                            key = key << 2 | c;
                        }
                        if (nfree) {
                            const uint64_t* it = std::lower_bound(cfg.wl, cfg.wl + cfg.nWL, key);
                            if (it != cfg.wl + cfg.nWL && *it == key) lastW = it - cfg.wl;
                        }
                    }
                    if (lastW >= 0) { run++; nr11++; }
                }
            }
        }
        flush();
        acc[t].r11 = nr11;
    };
    {
        std::vector<std::thread> th;
        for (int t = 1; t < T; t++) th.emplace_back(worker, t);
        worker(0);
        for (auto& x : th) x.join();
    }
    out.r11Total = 0;
    bool tooLong = false;
    for (const Acc& a : acc) { out.r11Total += a.r11; tooLong = tooLong || a.tooLong; }
    if (tooLong) { out.nw = 0; NoInitVector<uint64_t>().swap(out.aux); }
    out.r11.swap(dense);
    // E11: the RNA Q30 count from the quality codes; the barcode read's quality is 'F' (and 'H' where the layout pads it with N)
    out.q30Rna = out.nRna = out.q30Cb = out.nCb = 0; out.cbQmin = 256; out.cbQmax = -1;
    if (cfg.e11) {
        for (int c = 0; c < 4; c++) { out.nRna += out.qualHist[c]; if ((unsigned) out.qualPhred[c] + 33 >= cfg.q30) out.q30Rna += out.qualHist[c]; }
        if (!layoutBad) {
            auto qc = [&](uint64_t j) { return j < L1 ? (unsigned) 'F' : (unsigned) 'H'; };
            uint64_t q30 = 0, n = 0;
            for (uint64_t j = cbS0; j < std::min(cbS0 + cbL, S); j++) {
                const unsigned qq = qc(j);
                q30 += qq >= cfg.q30; n++;
                out.cbQmin = std::min(out.cbQmin, (int) qq); out.cbQmax = std::max(out.cbQmax, (int) qq);
            }
            for (uint64_t j = umiS0; j < std::min(umiS0 + umiL, S); j++) { q30 += qc(j) >= cfg.q30; n++; }
            out.q30Cb = q30 * N; out.nCb = n * N;
            if (N == 0) { out.cbQmin = 256; out.cbQmax = -1; }
        }
    }
    if (layoutBad) why = "a barcode read STAR would reject (length)";
    return true;
}

// ---- phase A in the pipeline's process (engine_gpu_ingest.h startParent). On the heap and never destroyed (an exit() while the thread runs
// must not run ~thread); per pid: joined and released by the process that started it, taken by the STAR process forked after the join.
namespace {
struct ParentA {
    std::thread th;
    pid_t pid = 0;
    int threads = 1;
    PackedReads out;
    bool ok = false, joined = false;
    std::string why;
    std::chrono::steady_clock::time_point t0;
    double done = 0;
};
ParentA* parentA_ = nullptr;
}  // namespace

void startParent(int threads) {
    if (parentA_ != nullptr || !plan().on) return;
    ParentA* a = parentA_ = new ParentA;
    a->pid = getpid();
    a->threads = std::max(1, threads);
    a->t0 = std::chrono::steady_clock::now();
    a->th = std::thread([a]() {
        a->ok = decodeReads(a->threads, a->out, a->why);
        a->done = std::chrono::duration<double>(std::chrono::steady_clock::now() - a->t0).count();
    });
}
bool decodeScan(int threads, const singlet::fq::RunSet::HeadScanOptions& so, std::vector<singlet::fq::RunSet::BlockHead>& heads, uint32_t r2Fixed,
                std::string& why) {
    if (parentA_ != nullptr) { why = "the packed reads are decoded already"; return false; }
    ParentA* a = new ParentA;
    a->pid = getpid();
    a->threads = std::max(1, threads);
    a->t0 = std::chrono::steady_clock::now();
    Fused fz;
    fz.so = &so; fz.heads = &heads; fz.r2Fixed = r2Fixed;
    a->ok = decodeImpl(a->threads, a->out, a->why, &fz);
    if (!a->ok) { why = a->why; a->out.release(); delete a; return false; }
    a->done = 0;
    parentA_ = a;   // startParent finds it decoded; joinParent and takeParent as for the background decode
    return true;
}
void dropFused() {
    if (parentA_ == nullptr || parentA_->pid != getpid() || parentA_->th.joinable()) return;
    parentA_->out.release();
    parentA_ = nullptr;
}
bool joinParent(double& waited, std::string& log) {
    waited = 0;
    if (parentA_ == nullptr || parentA_->pid != getpid()) return false;
    const auto w0 = std::chrono::steady_clock::now();
    if (parentA_->th.joinable()) parentA_->th.join();
    waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    parentA_->joined = true;
    if (parentA_->ok)
        log = std::to_string(parentA_->out.n) + " reads decoded into the packed buffers in " + std::to_string(parentA_->out.sec) + " s ("
            + std::to_string(parentA_->threads) + " threads, done " + std::to_string(parentA_->done) + " s after the head scan); STAR's process takes them";
    else log = "the decode into the packed buffers failed (" + parentA_->why + "): STAR's process decodes the .1fq";
    return parentA_->ok;
}
bool takeParent(PackedReads& out) {
    if (parentA_ == nullptr || !parentA_->joined || !parentA_->ok || parentA_->out.n == 0) return false;
    out.swap(parentA_->out);
    parentA_->out.release();
    return true;
}
void releaseParent() {
    if (parentA_ == nullptr || parentA_->pid != getpid()) return;
    if (parentA_->th.joinable()) parentA_->th.join();
    parentA_->out.release();
}

}  // namespace ingest
}  // namespace singlet_engine_gpu
#endif
