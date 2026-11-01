// E13 (docs/gpu-engine/fable/E13-ingest.md; SINGLET_ENGINE_GPU_INGEST=1): STAR's prefill decodes the sample's .1fq straight into the packed
// reads (singlet/engine_gpu_ingest.h) instead of reading back the FASTQ text the pipeline used to write. One pass over the blocks, on the
// job's threads, each block into its own ranges of the final arrays (the pipeline's head scan counted every block's reads and bases):
//   the cDNA sequence codes and offsets (the engine's input), E10's per-read words (CR/UR keys and low-quality bits), the cDNA quality
//   as 2-bit codes and the barcode read as 3-bit codes (the text of the reads E10's host side writes or hands to STAR's loop), E11's
//   quality counts and R11's exact CB histogram.
// The bytes are those the FASTQ path gave STAR (read_source.h format_block with the file-mode options: no 5' clip, no length guard; reads
// with an empty mate dropped; R1's quality 'F'; R2's the stored quality + 33), read by the same RunSet decoder.
// No STAR headers here (the .1fq headers and STAR's IncludeDefine.h do not mix); STAR's own per-read work (the CellRanger4 clip) runs in
// the prefill's callback on the decoding threads.
#ifdef SINGLET_HAVE_ENGINE_GPU
#include "singlet/fq/read_source.h"
#include "singlet/engine_gpu_ingest.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace singlet_engine_gpu {
namespace ingest {

bool decode(const DecodeConfig& cfg, PackedReads& out, std::string& why) {
    const auto t0 = std::chrono::steady_clock::now();
    const Plan& pl = plan();
    if (!pl.on || !pl.runs) { why = "no head-scan plan"; return false; }
    const singlet::fq::RunSet& runs = *pl.runs;
    const uint32_t nB = pl.nBlocks;
    if (runs.block_count() != nB || pl.first.size() != (size_t) nB + 1 || pl.baseAt.size() != (size_t) nB + 1 || pl.nameAt.size() < nB) {
        why = "the head scan's plan does not match the .1fq"; return false;
    }
    if (pl.r1L == 0 || pl.r1L > 63) { why = "barcode read length " + std::to_string(pl.r1L); return false; }
    if (cfg.cbL == 0 || cfg.cbL > 21 || cfg.umiL == 0 || cfg.umiL > 21) { why = "CB/UMI over 21 bases"; return false; }
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
    out.nw = cfg.e10 ? cfg.nw : 0;
    const uint64_t stride = 2 + (uint64_t) out.nw;
    if (cfg.e10) out.aux.resize(N * stride);
    std::memcpy(out.qualPhred, pl.qualPhred, 4);
    int codeOf[256];
    for (int k = 0; k < 256; k++) codeOf[k] = -1;
    for (int k = (int) std::max<uint32_t>(1, pl.qualCodes) - 1; k >= 0; k--) codeOf[pl.qualPhred[k]] = k;   // the first code of a Phred wins
    const bool r11On = cfg.wl != nullptr && cfg.nWL > 0;
    std::vector<uint64_t> dense(r11On ? cfg.nWL : 0, 0);
    const int T = std::max(1, std::min<int>(cfg.threads, (int) std::max<uint32_t>(1, nB)));
    std::atomic<uint32_t> next{0};
    std::atomic<bool> stop{false};
    std::mutex mu;
    std::string err;
    struct Acc { uint64_t q30Rna = 0, nRna = 0, q30Cb = 0, nCb = 0, r11 = 0; int qmin = 256, qmax = -1; char pad[64]; };
    std::vector<Acc> acc(T);
    const int e10MinQ = cfg.minQ;
    const unsigned q30 = cfg.q30;
    const uint64_t cbS0 = cfg.cbS0, cbL = cfg.cbL, umiS0 = cfg.umiS0, umiL = cfg.umiL, cbumiL = cfg.cbumiL, bL = cfg.bL;
    std::atomic<bool> e10bad{false};
    auto fail = [&](const std::string& w) {
        std::lock_guard<std::mutex> lk(mu);
        if (err.empty()) err = w;
        stop = true;
    };
    auto worker = [&](int tid) {
        Acc a;
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
            // R11: the .1fq is barcode-sorted, so consecutive reads mostly share their CB: one whitelist lookup per run of equal keys
            uint64_t lastCr = ~0ull, run = 0;
            int64_t lastW = -1;
            auto flush = [&]() { if (run && lastW >= 0) __atomic_fetch_add(&dense[(size_t) lastW], run, __ATOMIC_RELAXED); run = 0; };
            for (uint32_t i = 0; i < blk.n_reads; i++) {
                const uint16_t L1 = blk.r1_len(i), L2 = blk.r2_len(i);
                if (L1 == 0 || L2 == 0) continue;   // not given to STAR (format_block)
                if (r >= rEnd || pos + L2 > posEnd) { fail("block " + std::to_string(b) + " has more reads or bases than the head scan counted"); return; }
                if (L1 != pl.r1L) { fail("barcode read length " + std::to_string(L1) + " in block " + std::to_string(b)); return; }
                // cDNA: codes and offsets
                const uint8_t* s2 = blk.r2_seq(i);
                std::memcpy(out.seq.data() + pos, s2, L2);
                out.off[r + 1] = pos + L2;
                // cDNA quality: 2-bit codes; E10's low-quality bits; E11's Q30 count
                uint64_t* w = cfg.e10 ? out.aux.data() + r * stride + 2 : nullptr;
                if (w) for (uint32_t x = 0; x < out.nw; x++) w[x] = 0;
                if (w && L2 > 64 * (uint64_t) out.nw) { fail("a cDNA read longer than the quality words hold"); return; }
                const uint8_t* q = hasQ ? blk.r2_quality(i) : nullptr;
                uint64_t c30 = 0;
                for (uint32_t j = 0; j < L2; j++) {
                    const uint8_t ph = q ? q[j] : (uint8_t) ('F' - 33);
                    const int c = codeOf[ph];
                    if (c < 0) { fail("a quality value (" + std::to_string(ph) + ") outside the .1fq's quality mode"); return; }
                    qacc |= (uint8_t) (c << (2 * (k & 3)));
                    if ((k & 3) == 3) { qd[k >> 2] = qacc; qacc = 0; }
                    k++;
                    if (w && (int) ph < e10MinQ) w[j >> 6] |= 1ull << (j & 63);
                    c30 += (unsigned) ph + 33 >= q30;
                }
                a.q30Rna += c30; a.nRna += L2;
                // barcode read: 3-bit codes; CR/UR keys (SoloReadBarcode::getCBandUMI for CB_UMI_Simple, HandoffReadEntry's packing)
                const uint8_t* s1 = blk.r1_seq(i);
                uint64_t* rw = out.r1.data() + r * out.r1w;
                for (uint32_t x = 0; x < out.r1w; x++) rw[x] = 0;
                for (uint32_t j = 0; j < L1; j++) rw[j / 21] |= (uint64_t) std::min<uint8_t>(s1[j], 4) << (3 * (j % 21));
                const uint64_t S = (bL == 0 && L1 < cbumiL) ? cbumiL : L1;   // padded with N to cbumiL
                const bool layoutBad = (bL > 0 && L1 != bL) || cbS0 > S || umiS0 > S;
                auto pack = [&](uint64_t s0, uint64_t len) {
                    const uint64_t e0 = std::min(s0 + len, S);
                    uint64_t v = 0;
                    for (uint64_t j = s0; j < e0 && j - s0 < 21; j++) v |= (uint64_t) (j < L1 ? std::min<uint8_t>(s1[j], 4) : 4) << (3 * (j - s0));
                    return v;
                };
                const uint64_t cr = layoutBad ? 0 : pack(cbS0, cbL), ur = layoutBad ? 0 : pack(umiS0, umiL);
                if (w) {
                    if (layoutBad) e10bad = true;
                    w[-2] = cr; w[-1] = ur;
                }
                // E11: the CB and UMI spans of the barcode read's quality ('F', padded with 'H' like the sequence with N)
                if (cfg.e11 && !layoutBad) {
                    auto qc = [&](uint64_t j) { return j < L1 ? (unsigned) 'F' : (unsigned) 'H'; };
                    for (uint64_t j = cbS0; j < std::min(cbS0 + cbL, S); j++) {
                        const unsigned qq = qc(j);
                        a.q30Cb += qq >= q30; a.nCb++;
                        a.qmin = std::min(a.qmin, (int) qq); a.qmax = std::max(a.qmax, (int) qq);
                    }
                    for (uint64_t j = umiS0; j < std::min(umiS0 + umiL, S); j++) { a.q30Cb += qc(j) >= q30; a.nCb++; }
                }
                // R11 (hostloop_gpu.cu k_r11 on the same keys): a UMI without N that is not a homopolymer, an N-free CB exact in the whitelist
                if (r11On && !layoutBad) {
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
                        if (lastW >= 0) { run++; a.r11++; }
                    }
                }
                pos += L2;
                r++;
            }
            flush();
            if ((k & 3) != 0) qd[k >> 2] = qacc;
            if (r != rEnd || pos != posEnd) { fail("block " + std::to_string(b) + " has fewer reads or bases than the head scan counted"); return; }
            if (cfg.perBlock) cfg.perBlock(tid, pl.first[b], rEnd);
        }
        acc[tid] = a;
    };
    {
        std::vector<std::thread> th;
        for (int t = 1; t < T; t++) th.emplace_back(worker, t);
        worker(0);
        for (auto& x : th) x.join();
    }
    if (!err.empty()) { why = err; out.release(); return false; }
    for (const Acc& a : acc) {
        out.q30Rna += a.q30Rna; out.nRna += a.nRna; out.q30Cb += a.q30Cb; out.nCb += a.nCb; out.r11Total += a.r11;
        out.cbQmin = std::min(out.cbQmin, a.qmin); out.cbQmax = std::max(out.cbQmax, a.qmax);
    }
    if (e10bad) { out.nw = 0; NoInitVector<uint64_t>().swap(out.aux); out.q30Cb = out.nCb = 0; out.cbQmin = 256; out.cbQmax = -1; }
    out.r11.swap(dense);
    out.sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace ingest
}  // namespace singlet_engine_gpu
#endif
