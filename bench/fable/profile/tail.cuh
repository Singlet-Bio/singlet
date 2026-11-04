// E1 (docs/gpu-engine/fable/E1-profile.md): batch-tail instrumentation of the staged lanes in bench/gpu/m5/m5.cu.
// Compiled in only with -DM5_TAIL (binary m5t); the default build of m5.cu does not include this file.
// Every thread of k_p1 and k_p3 records its start and end (%globaltimer, ns), its clock64() cycles and its SM. After each
// batch of the last rep the host turns them into: the span, the thread/warp/block-slot utilisation against the occupancy
// limit (slots = resident blocks per SM x threads per block x SMs), the drain (from the last block start to the end),
// finish-time percentiles and a 20-bin active-thread curve, plus the per-batch g_clk deltas.
#pragma once
#include <vector>
#include <algorithm>
struct TailRec { unsigned long long t0, t1, c0, c1, nEnt, stitch; uint32_t sm, nW, steps, pad; };   // nEnt..steps: k_p3 only
__device__ TailRec* g_tail[2];   // [0] k_p1, [1] k_p3: one record per thread of the current batch (nullptr: off)
__device__ __forceinline__ unsigned long long tail_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t tail_sm() { uint32_t s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
// start and end go straight to memory (nothing stays live across the kernel body, so registers and occupancy stay as in m5)
#define TAIL_BEGIN(which, i) do { TailRec* a_ = g_tail[which]; if (a_) { a_[i].t0 = tail_gt(); a_[i].c0 = clock64(); } } while (0)
// k_p3: SA entries of the read, windows, stitching steps, stitching + selection cycles (w.clk[2] + w.clk[3])
#define TAIL_INFO(i, ne, nw, st, sc) do { TailRec* a_ = g_tail[1]; \
    if (a_) { a_[i].nEnt = (ne); a_[i].nW = (nw); a_[i].steps = (st); a_[i].stitch = (sc); } } while (0)
#define TAIL_END(which, i) do { TailRec* a_ = g_tail[which]; \
    if (a_) { const long long c_ = clock64(); a_[i].t1 = tail_gt(); a_[i].c1 = (unsigned long long)c_; a_[i].sm = tail_sm(); } } while (0)

struct TailHost {
    TailRec* d[2] = {nullptr, nullptr};
    std::vector<TailRec> h;
    const char* lane = "";
    int sms = 0;
    unsigned long long clkPrev[8] = {0};
    cudaEvent_t ev[2];         // per batch: before k_p2, after k_p3 (m5.cu records q[0..3] around the phases)
    float prevEnd = 0;         // ms from the rep's start event to the end of the previous batch
    double evSum[6] = {0, 0, 0, 0, 0, 0};
    // lane totals per kernel: span, thread-, warp-, block-slot areas (ns x threads), slot x span, drain
    double tSpan[2] = {0, 0}, tThr[2] = {0, 0}, tWarp[2] = {0, 0}, tBlk[2] = {0, 0}, tSlot[2] = {0, 0}, tDrain[2] = {0, 0}, tCyc[2] = {0, 0};
    void init(unsigned long long NB, const char* name, int nsm) {
        lane = name; sms = nsm; cudaEventCreate(&ev[0]); cudaEventCreate(&ev[1]);
        for (int k = 0; k < 2; k++) cudaMalloc(&d[k], NB * sizeof(TailRec));
        h.resize(NB);
    }
    void arm(bool on) {   // records written only while armed (the last rep)
        TailRec* p[2] = {on ? d[0] : nullptr, on ? d[1] : nullptr};
        cudaMemcpyToSymbol(g_tail, p, sizeof p);
        if (on) cudaMemcpyFromSymbol(clkPrev, g_clk, sizeof clkPrev);
        prevEnd = 0;
    }
    void release() { arm(false); for (int k = 0; k < 2; k++) if (d[k]) cudaFree(d[k]); d[0] = d[1] = nullptr; }
    // one kernel of one batch: nb threads used (grid ceil(nb/T) x T), bps resident blocks per SM (occupancy API)
    void batch(int which, unsigned long long nb, int T, int bps, int ib) {
        if (!nb) return;
        cudaMemcpy(h.data(), d[which], nb * sizeof(TailRec), cudaMemcpyDeviceToHost);
        unsigned long long tmin = ~0ULL, tmax = 0, lastBlockStart = 0;
        double thr = 0, cyc = 0, longest = 0, warpA = 0, blkA = 0;
        for (unsigned long long i = 0; i < nb; i++) {
            tmin = std::min(tmin, h[i].t0); tmax = std::max(tmax, h[i].t1);
            double dd = (double)(h[i].t1 - h[i].t0); thr += dd; cyc += (double)(h[i].c1 - h[i].c0); longest = std::max(longest, dd);
        }
        for (unsigned long long w0 = 0; w0 < nb; w0 += 32) {
            unsigned long long a = ~0ULL, b = 0;
            for (unsigned long long i = w0; i < std::min(nb, w0 + 32); i++) { a = std::min(a, h[i].t0); b = std::max(b, h[i].t1); }
            warpA += 32.0 * (double)(b - a);
        }
        for (unsigned long long b0 = 0; b0 < nb; b0 += T) {
            unsigned long long a = ~0ULL, b = 0;
            for (unsigned long long i = b0; i < std::min(nb, b0 + T); i++) { a = std::min(a, h[i].t0); b = std::max(b, h[i].t1); }
            blkA += (double)T * (double)(b - a); lastBlockStart = std::max(lastBlockStart, a);
        }
        const double span = (double)(tmax - tmin), slots = (double)bps * T * sms, area = slots * span;
        std::vector<double> fin(nb);
        for (unsigned long long i = 0; i < nb; i++) fin[i] = (double)(h[i].t1 - tmin);
        std::sort(fin.begin(), fin.end());
        auto pct = [&](double q) { unsigned long long k = (unsigned long long)std::min<double>((double)nb - 1, q * (double)nb); return fin[k] / span; };
        const int NBIN = 20; double curve[NBIN] = {0};
        for (unsigned long long i = 0; i < nb; i++) {
            double a = (double)(h[i].t0 - tmin), b = (double)(h[i].t1 - tmin);
            int ba = std::min(NBIN - 1, (int)(a / span * NBIN)), bb = std::min(NBIN - 1, (int)(b / span * NBIN));
            for (int k = ba; k <= bb; k++) {
                double lo = std::max(a, span * k / NBIN), hi = std::min(b, span * (k + 1) / NBIN);
                if (hi > lo) curve[k] += hi - lo;
            }
        }
        const double drain = (double)(tmax - lastBlockStart);
        fprintf(stderr, "TAIL lane=%s batch=%d kernel=k_p%d n=%llu T=%d bps=%d slots=%.0f span_ms=%.3f busy_thr=%.4f busy_warp=%.4f busy_blk=%.4f "
                        "drain_ms=%.3f drain_share=%.4f fin_p50=%.4f fin_p90=%.4f fin_p99=%.4f fin_p999=%.4f longest_share=%.4f clk_ghz=%.3f curve=",
                lane, ib, which ? 3 : 1, nb, T, bps, slots, span / 1e6, thr / area, warpA / area, blkA / area, drain / 1e6, drain / span,
                pct(0.5), pct(0.9), pct(0.99), pct(0.999), longest / span, cyc / thr);
        for (int k = 0; k < NBIN; k++) fprintf(stderr, "%s%.3f", k ? "," : "", curve[k] / (slots * span / NBIN));
        fprintf(stderr, "\n");
        tSpan[which] += span; tThr[which] += thr; tWarp[which] += warpA; tBlk[which] += blkA; tSlot[which] += area; tDrain[which] += drain; tCyc[which] += cyc;
        if (which == 1) {   // the longest reads: what they spend their time on (assignment = total - stitching - selection)
            std::vector<unsigned long long> ord(nb); for (unsigned long long i = 0; i < nb; i++) ord[i] = i;
            std::sort(ord.begin(), ord.end(), [&](unsigned long long a, unsigned long long b) { return h[a].t1 - h[a].t0 > h[b].t1 - h[b].t0; });
            const double ghz = thr > 0 ? cyc / thr : 1.78;
            double cAll = 0, sAll = 0, cTop = 0, sTop = 0; const unsigned long long nTop = std::max<unsigned long long>(1, nb / 100);
            for (unsigned long long r = 0; r < nb; r++) { const TailRec& x = h[ord[r]]; double c = (double)(x.c1 - x.c0), sc = (double)x.stitch;
                cAll += c; sAll += sc; if (r < nTop) { cTop += c; sTop += sc; } }
            fprintf(stderr, "TAILWHO lane=%s batch=%d stitch_select_share_all=%.4f stitch_select_share_top1pct=%.4f top1pct_thread_time_share=%.4f\n",
                    lane, ib, sAll / cAll, sTop / cTop, cTop / cAll);
            for (int r = 0; r < 5 && r < (int)nb; r++) { const TailRec& x = h[ord[r]];
                fprintf(stderr, "TAILTOP lane=%s batch=%d rank=%d ms=%.3f start_ms=%.3f nEnt=%llu nW=%u steps=%u stitch_select_ms=%.3f rest_ms=%.3f\n",
                        lane, ib, r, (x.t1 - x.t0) / 1e6, (x.t0 - tmin) / 1e6, x.nEnt, x.nW, x.steps, x.stitch / ghz / 1e6,
                        ((double)(x.c1 - x.c0) - (double)x.stitch) / ghz / 1e6); }
        }
        if (which == 1) {   // g_clk deltas of this batch (k_p3 adds each read's phase-1 and phase-3 counters at its end)
            unsigned long long c[8]; cudaMemcpyFromSymbol(c, g_clk, sizeof c);
            fprintf(stderr, "TAILCLK lane=%s batch=%d n=%llu seeds=%llu windows=%llu stitching=%llu select=%llu sat=%llu fin=%llu copies=%llu rec=%llu\n",
                    lane, ib, nb, c[0] - clkPrev[0], c[1] - clkPrev[1], c[2] - clkPrev[2], c[3] - clkPrev[3], c[4] - clkPrev[4], c[5] - clkPrev[5],
                    c[6] - clkPrev[6], c[7] - clkPrev[7]);
            for (int k = 0; k < 8; k++) clkPrev[k] = c[k];
        }
    }
    // event times of one batch (ms): gap since the previous batch's end, k_p1, scan + 2 sync copies + memset, k_p2, k_p3, copy_status
    void events(cudaEvent_t e0, const cudaEvent_t* q, int ib, unsigned long long nb) {
        float t0, t1, ta, t2, tb, t3;
        cudaEventElapsedTime(&t0, e0, q[0]); cudaEventElapsedTime(&t1, e0, q[1]); cudaEventElapsedTime(&ta, e0, ev[0]);
        cudaEventElapsedTime(&t2, e0, q[2]); cudaEventElapsedTime(&tb, e0, ev[1]); cudaEventElapsedTime(&t3, e0, q[3]);
        double v[6] = {t0 - prevEnd, t1 - t0, ta - t1, t2 - ta, tb - t2, t3 - tb};
        fprintf(stderr, "TAILEV lane=%s batch=%d n=%llu gap_ms=%.4f p1_ms=%.4f scan_sync_ms=%.4f p2_ms=%.4f p3_ms=%.4f cs_ms=%.4f\n",
                lane, ib, nb, v[0], v[1], v[2], v[3], v[4], v[5]);
        for (int k = 0; k < 6; k++) evSum[k] += v[k];
        prevEnd = t3;
    }
    void summary() {
        fprintf(stderr, "TAILEVSUM lane=%s gap_ms=%.3f p1_ms=%.3f scan_sync_ms=%.3f p2_ms=%.3f p3_ms=%.3f cs_ms=%.3f\n",
                lane, evSum[0], evSum[1], evSum[2], evSum[3], evSum[4], evSum[5]);
        for (int k = 0; k < 2; k++) if (tSlot[k] > 0)
            fprintf(stderr, "TAILSUM lane=%s kernel=k_p%d span_ms=%.3f busy_thr=%.4f busy_warp=%.4f busy_blk=%.4f drain_share=%.4f clk_ghz=%.3f\n",
                    lane, k ? 3 : 1, tSpan[k] / 1e6, tThr[k] / tSlot[k], tWarp[k] / tSlot[k], tBlk[k] / tSlot[k], tDrain[k] / tSpan[k], tCyc[k] / tThr[k]);
    }
};
