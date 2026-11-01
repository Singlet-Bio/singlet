// E1 (docs/gpu-engine/fable/E1-profile.md): batch-tail instrumentation of the staged lanes in bench/gpu/m5/m5.cu.
// Compiled in only with -DM5_TAIL (binary m5t); the default build of m5.cu does not include this file.
// Every thread of k_p1 and k_p3 records its start and end (%globaltimer, ns), its clock64() cycles and its SM. After each
// batch of the last rep the host turns them into: the span, the thread/warp/block-slot utilisation against the occupancy
// limit (slots = resident blocks per SM x threads per block x SMs), the drain (from the last block start to the end),
// finish-time percentiles and a 20-bin active-thread curve, plus the per-batch g_clk deltas.
#pragma once
#include <vector>
#include <algorithm>
struct TailRec { unsigned long long t0, t1, cyc; uint32_t sm, pad; };
__device__ TailRec* g_tail[2];   // [0] k_p1, [1] k_p3: one record per thread of the current batch (nullptr: off)
__device__ __forceinline__ unsigned long long tail_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t tail_sm() { uint32_t s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
#define TAIL_BEGIN const unsigned long long tg0_ = tail_gt(); const long long tc0_ = clock64();
#define TAIL_END(which, i) do { const long long tc1_ = clock64(); TailRec* a_ = g_tail[which]; \
    if (a_) a_[i] = TailRec{tg0_, tail_gt(), (unsigned long long)(tc1_ - tc0_), tail_sm(), 0}; } while (0)

struct TailHost {
    TailRec* d[2] = {nullptr, nullptr};
    std::vector<TailRec> h;
    const char* lane = "";
    int sms = 0;
    unsigned long long clkPrev[8] = {0};
    // lane totals per kernel: span, thread-, warp-, block-slot areas (ns x threads), slot x span, drain
    double tSpan[2] = {0, 0}, tThr[2] = {0, 0}, tWarp[2] = {0, 0}, tBlk[2] = {0, 0}, tSlot[2] = {0, 0}, tDrain[2] = {0, 0}, tCyc[2] = {0, 0};
    void init(unsigned long long NB, const char* name, int nsm) {
        lane = name; sms = nsm;
        for (int k = 0; k < 2; k++) cudaMalloc(&d[k], NB * sizeof(TailRec));
        h.resize(NB);
    }
    void arm(bool on) {   // records written only while armed (the last rep)
        TailRec* p[2] = {on ? d[0] : nullptr, on ? d[1] : nullptr};
        cudaMemcpyToSymbol(g_tail, p, sizeof p);
        if (on) cudaMemcpyFromSymbol(clkPrev, g_clk, sizeof clkPrev);
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
            double dd = (double)(h[i].t1 - h[i].t0); thr += dd; cyc += (double)h[i].cyc; longest = std::max(longest, dd);
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
        if (which == 1) {   // g_clk deltas of this batch (k_p3 adds each read's phase-1 and phase-3 counters at its end)
            unsigned long long c[8]; cudaMemcpyFromSymbol(c, g_clk, sizeof c);
            fprintf(stderr, "TAILCLK lane=%s batch=%d n=%llu seeds=%llu windows=%llu stitching=%llu select=%llu sat=%llu fin=%llu copies=%llu rec=%llu\n",
                    lane, ib, nb, c[0] - clkPrev[0], c[1] - clkPrev[1], c[2] - clkPrev[2], c[3] - clkPrev[3], c[4] - clkPrev[4], c[5] - clkPrev[5],
                    c[6] - clkPrev[6], c[7] - clkPrev[7]);
            for (int k = 0; k < 8; k++) clkPrev[k] = c[k];
        }
    }
    void summary() {
        for (int k = 0; k < 2; k++) if (tSlot[k] > 0)
            fprintf(stderr, "TAILSUM lane=%s kernel=k_p%d span_ms=%.3f busy_thr=%.4f busy_warp=%.4f busy_blk=%.4f drain_share=%.4f clk_ghz=%.3f\n",
                    lane, k ? 3 : 1, tSpan[k] / 1e6, tThr[k] / tSlot[k], tWarp[k] / tSlot[k], tBlk[k] / tSlot[k], tDrain[k] / tSpan[k], tCyc[k] / tThr[k]);
    }
};
