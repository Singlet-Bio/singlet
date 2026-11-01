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
#include "singlet/fq/read_source.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace singlet_engine_gpu {
namespace ingest {

bool decodeReads(int threads, PackedReads& out, std::string& why) {
    const auto t0 = std::chrono::steady_clock::now();
    const Plan& pl = plan();
    if (!pl.on || !pl.runs) { why = "no head-scan plan"; return false; }
    const singlet::fq::RunSet& runs = *pl.runs;
    const uint32_t nB = pl.nBlocks;
    if (runs.block_count() != nB || pl.first.size() != (size_t) nB + 1 || pl.baseAt.size() != (size_t) nB + 1 || pl.nameAt.size() < nB) {
        why = "the head scan's plan does not match the .1fq"; return false;
    }
    if (pl.r1L == 0 || pl.r1L > 63) { why = "barcode read length " + std::to_string(pl.r1L); return false; }
    const uint64_t N = pl.reads(), B = pl.bases();
    out.release();
    out.n = N;
    out.first = pl.first; out.baseAt = pl.baseAt; out.nameAt = pl.nameAt;
    out.qualAt.assign((size_t) nB + 1, 0);
    for (uint32_t b = 0; b < nB; b++) out.qualAt[b + 1] = out.qualAt[b] + (pl.baseAt[b + 1] - pl.baseAt[b] + 3) / 4;
    out.seq.resize(B);
    out.off.resize(N + 1);
    out.off[0] = 0;
    out.qual.resize(std::max<uint64_t>(1, out.qualAt[nB]));
    out.r1L = pl.r1L; out.r1w = (pl.r1L + 20) / 21;
    out.r1.resize(std::max<uint64_t>(1, N * out.r1w));
    std::memcpy(out.qualPhred, pl.qualPhred, 4);
    int codeOf[256];
    for (int k = 0; k < 256; k++) codeOf[k] = -1;
    for (int k = (int) std::max<uint32_t>(1, pl.qualCodes) - 1; k >= 0; k--) codeOf[pl.qualPhred[k]] = k;   // the first code of a Phred wins
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
        for (uint32_t b; !stop && (b = next.fetch_add(1)) < nB;) {
            try {
                runs.decode(b, blk);
            } catch (const std::exception& ex) {
                fail(std::string("block ") + std::to_string(b) + ": " + ex.what());
                return;
            }
            const uint64_t rEnd = pl.first[b + 1], posEnd = pl.baseAt[b + 1];
            uint64_t r = pl.first[b], pos = pl.baseAt[b], k = 0;
            uint8_t* qd = out.qual.data() + out.qualAt[b];
            uint8_t qacc = 0;
            const bool hasQ = !blk.r2_qual.empty();
            for (uint32_t i = 0; i < blk.n_reads; i++) {
                const uint16_t L1 = blk.r1_len(i), L2 = blk.r2_len(i);
                if (L1 == 0 || L2 == 0) continue;   // not given to STAR (format_block)
                if (r >= rEnd || pos + L2 > posEnd) { fail("block " + std::to_string(b) + " has more reads or bases than the head scan counted"); return; }
                if (L1 != pl.r1L) { fail("barcode read length " + std::to_string(L1) + " in block " + std::to_string(b)); return; }
                // cDNA: codes and offsets
                std::memcpy(out.seq.data() + pos, blk.r2_seq(i), L2);
                out.off[r + 1] = pos + L2;
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
                // barcode read: 3-bit codes, 21 per word
                const uint8_t* s1 = blk.r1_seq(i);
                uint64_t* rw = out.r1.data() + r * out.r1w;
                for (uint32_t x = 0; x < out.r1w; x++) rw[x] = 0;
                for (uint32_t j = 0; j < L1; j++) rw[j / 21] |= (uint64_t) std::min<uint8_t>(s1[j], 4) << (3 * (j % 21));
                pos += L2;
                r++;
            }
            if ((k & 3) != 0) qd[k >> 2] = qacc;
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
    for (const Acc& a : acc)
        for (int c = 0; c < 4; c++) out.qualHist[c] += a.hist[c];
    out.sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

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
            const uint64_t cr = pack(r, cbS0, cbL), ur = pack(r, umiS0, umiL);
            if (out.nw) {
                uint64_t* w = out.aux.data() + r * stride;
                w[0] = cr; w[1] = ur;
                uint64_t* q = w + 2;
                for (uint32_t x = 0; x < nw; x++) q[x] = 0;
                const uint64_t s0 = out.off[r], L = out.off[r + 1] - s0;
                if (L > 64 * (uint64_t) nw) { acc[t].tooLong = true; continue; }
                for (uint64_t j = 0; j < L; j++)
                    if (lowQ[out.qcode(b, s0 + j)]) q[j >> 6] |= 1ull << (j & 63);
            }
            // R11 (hostloop_gpu.cu k_r11 on the same keys): a UMI without N that is not a homopolymer, an N-free CB exact in the whitelist.
            // The .1fq is barcode-sorted, so consecutive reads mostly share their CB: one whitelist lookup per run of equal keys.
            if (r11On) {
                bool ok = true, homo = true;
                const uint64_t u0 = ur & 7;
                for (uint64_t j = 0; j < umiL; j++) {
                    const uint64_t c = (ur >> (3 * j)) & 7;
                    if (c > 3) { ok = false; break; }
                    homo = homo && c == u0;
                }
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
