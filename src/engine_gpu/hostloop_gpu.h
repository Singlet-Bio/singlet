// E10 (docs/gpu-engine/fable/E10-hostloop.md): what STAR's per-read loop still produces for the reads with a GPU record, on the device.
//   entries():  per read the GPU pileup's hand-off entry (STAR's ReadAlign::handoffReadEntry: CR/UR, quality counters, 128-bit low-quality mask,
//               clip flags, the rRNA hook's sample bit) from the original R2 (STAR's numeric code), the clip of k_clip and what the host decode
//               packed per read (CR/UR key, quality bits);
//   products(): per record its read multiplicity, then the Log.final.out counters (Stats: oneRead, mappedFilter, outputAlignments /
//               outputHandoffSummary, transcriptStats), the junctions of SJ.out.tab collapsed as STAR's outputSJ collapses them
//               (outputTranscriptSJ per transcript, the in-read duplicate rule, counts summed, overhangs maxed), the reads whose record is
//               unmapped (outReadsUnmapped) and the reads in the hook's sample; the entries of reads without a record are zeroed.
// Called by mapDistinct (engine_gpu.cu) under SINGLET_ENGINE_GPU_E2E >= 2 only. Plain C++ header; device pointers as plain pointers.
#pragma once
#include <cstdint>
#include <string>
#include "records.cuh"
#include "singlet/engine_gpu.h"

namespace singlet_hostloop_gpu {

// dSeq/dOff: the original reads (before clipping); dCs/dCl: per read the clipped span (absolute start, length), or null (no clipping);
// dAux: per read 2 + nw words (cr3, ur3, quality bits); dEnt: 24 bytes per read; dMask: 16 bytes per read or null
bool entries(const char* dSeq, const uint64_t* dOff, const uint64_t* dCs, const uint64_t* dCl, uint64_t n, const uint64_t* dAux, uint32_t nw,
             bool hook, e6::ReadEntry* dEnt, uint64_t* dMask, float& ms, std::string& err);

// R11's exact CB histogram over every read (bench/fable/records/r11_hist.cpp's rule on the CR/UR keys of dAux): hist[nWL] (zeroed here) per
// whitelist entry (dWL: STAR's cbWL, sorted)
bool r11(const uint64_t* dAux, uint32_t nw, uint64_t n, const uint64_t* dWL, uint64_t nWL, uint32_t cbL, uint32_t umiL, uint32_t* dHist, float& ms,
         std::string& err);

// dBlob/dBOff/dSOff: the records (gathered, device); dRR: per read its record or 0xFFFFFFFF; fills E10 (stats, sj, unmappedReads, hookReads,
// recMapped); zeroes the entries/masks of reads without a record. Device buffers it allocates are freed before it returns.
// E17: genomeEnd (chrStart[nChr]) sets the junction sort key's split (start << gb | gap: gb = 32 below 2^32 bases, else 64 - its bits); with
// dEnt null (the entries and masks already copied to the host) neither the zeroing nor the hook list is made here (the caller does both)
bool products(const e6::RecParams& P, const uint8_t* dBlob, const uint64_t* dBOff, const uint64_t* dSOff, uint64_t nRec, const uint32_t* dRR, uint64_t n,
              e6::ReadEntry* dEnt, uint64_t* dMask, bool outSJ, bool sjAll, singlet_engine_gpu::Output::E10& out, std::string& err,
              uint64_t genomeEnd = 0);

} // namespace singlet_hostloop_gpu
