// E2 census (docs/gpu-engine/fable/E2-census.md): per-read and per-window records of the mapping state and cost.
// Built only with -DM5_CENSUS; statistics only, the mapping never reads them. Layout read by bench/fable/census/census_agg.py
// (keep the two in step: the file header carries the record sizes).
#pragma once
#include "stitch.cuh"

namespace m5 {

#pragma pack(push, 1)
struct CenRead {                    // one per (distinct read, lane it ran in)
    uint32_t read, count;           // index of the distinct read in the truth; multiplicity (reads with this sequence)
    uint16_t Lread; uint8_t lane, status;           // lane: 0 A, 1 M, 2 B, 3 host; Status
    uint8_t ov, final_, ident; int8_t unmapType;    // OvWhy; this lane gave the result; result identical to STAR; -2 = not finished
    uint8_t stitched, nSplit; uint16_t blobLen;
    uint32_t mapMarker;
    uint32_t nP, maxNrep, nRepGt1, nRepGtWin;       // stored seeds; their largest Nrep; seeds with Nrep > 1, > winAnchorMultimapNmax
    uint64_t sumNrep;
    uint32_t nWall, nW;             // windows created (at overflow: so far); windows with transcripts (STAR's nW after stitching)
    uint32_t nWAsum, nWAmax, nWAne, nWApeak;        // aligns per window after assignment: sum, max, windows with >= 1; peak during it
    uint64_t nEntries, nHits;       // SA entries of all seeds (the scan); entries that hit a window (assignments)
    uint64_t nSteps;                // recursion steps (include/exclude frames) over all windows
    uint32_t maxWinSteps, nWinStitched, winTrMax, keepMax;   // steps of the costliest window; windows stitched; peaks of NWTR, NKEEP
    uint32_t trNtotal, nTr, maxEx, wbN;             // transcripts recorded over windows; final; largest nExons stitched onto; winBin entries
    uint64_t nStitch, nFinal, nRec;                 // stitchAlignToTranscript calls, finalizations, recordings
    uint32_t maxWinCycW, pad2;      // window index of the costliest window (by steps)
    int64_t clk[8];                 // Work::clk: seeds, windows, stitching, select+encode | stitchAlignToTranscript, finalization, copies, recording
    int64_t cn[4];                  // cycles: k_p1 call, k_p2 scan (warp), assignment, k_p3 call; host lane: cn[3] = ns of the whole read
};
struct CenWin {                     // one per stitched window of a read, in the lane that gave the read's result
    uint32_t read; uint16_t iW; uint8_t lane, nWA;
    uint32_t steps, cyc;            // recursion steps, cycles (GPU) of stitchWindowAligns for this window
    uint16_t nTr, pad;              // transcripts recorded in the window
};
#pragma pack(pop)
static_assert(sizeof(CenRead) == 256, "CenRead layout");
static_assert(sizeof(CenWin) == 20, "CenWin layout");

template <class C> HD inline void cenFill(const Params& P, const Work<C>& w, uint8_t lane, uint8_t status, uint16_t blobLen, CenRead& r) {
    r.lane = lane; r.status = status; r.ov = (uint8_t)w.ov; r.final_ = 0; r.ident = 0;
    r.unmapType = (int8_t)(status == (uint8_t)Status::Ok ? w.unmapType : -2);
    r.stitched = w.stitched; r.nSplit = (uint8_t)mn<u64>(w.Nsplit, 255); r.blobLen = blobLen;
    r.mapMarker = (uint32_t)w.mapMarker;
    const uint nP = mn<uint>(w.nP, (uint)C::NP);
    r.nP = (uint32_t)w.nP; r.maxNrep = 0; r.nRepGt1 = 0; r.nRepGtWin = 0; r.sumNrep = 0;
    for (uint iP = 0; iP < nP; iP++) {
        uint nr = w.PC[iP][PC_Nrep];
        r.sumNrep += nr; if (nr > r.maxNrep) r.maxNrep = (uint32_t)nr;
        if (nr > 1) r.nRepGt1++;
        if (nr > P.winAnchorMultimapNmax) r.nRepGtWin++;
    }
    const uint nWall = w.stitched ? w.nWall : mn<uint>(w.nW, (uint)C::NW);
    r.nWall = (uint32_t)nWall; r.nW = (uint32_t)w.nW;
    r.nWAsum = r.nWAmax = r.nWAne = 0; r.nWApeak = 0;
    r.maxWinSteps = 0; r.maxWinCycW = 0; r.nWinStitched = 0;
#ifdef M5_CENSUS
    r.nWApeak = (uint32_t)w.cnMaxWA;
    if (w.stitched) {
        for (uint iW = 0; iW < nWall; iW++) {
            uint a = w.nWA[iW];
            r.nWAsum += (uint32_t)a; if (a > r.nWAmax) r.nWAmax = (uint32_t)a; if (a) r.nWAne++;
        }
    }
    if (w.cnFinish)
        for (uint iW = 0; iW < nWall; iW++)
            if (w.cnWinSteps[iW] > r.maxWinSteps) { r.maxWinSteps = w.cnWinSteps[iW]; r.maxWinCycW = (uint32_t)iW; }
    r.nWinStitched = (uint32_t)w.cnWinStitched;
    r.nHits = w.cnHits; r.nStitch = w.cnStitch; r.nFinal = w.cnFinal; r.nRec = w.cnRec;
    r.winTrMax = (uint32_t)w.cnWinTrMax; r.keepMax = (uint32_t)w.cnKeepMax; r.maxEx = (uint32_t)w.cnMaxEx;
    for (int k = 0; k < 4; k++) r.cn[k] = w.cn[k];
#endif
    r.nEntries = w.nEntries; r.nSteps = w.nSteps;
    r.trNtotal = (uint32_t)w.trNtotal; r.nTr = (uint32_t)w.nTr; r.wbN = (uint32_t)w.wbN; r.pad2 = 0;
    for (int k = 0; k < 8; k++) r.clk[k] = w.clk[k];
}

}  // namespace m5
