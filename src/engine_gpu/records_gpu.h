// E6: the GPU pileup's input built on the device from the GPU mapping results (hops 6 and 7 of the end-to-end path;
// docs/gpu-engine/fable/E6-records.md). Kernels: records_gpu.cu; per-record and per-read logic: records.cuh (host and device).
//
// Hop 6, buildRecords: per distinct sequence (mapDistinct's record r: its sequence and sequence-cache blob, device-resident) the SigRec, its
// AlnRecs, CIGAR ops and nt16 sequence, exactly as Collector::prepend_handoff makes them from the hand-off record that
// ReadAlign::seqCacheAlignRecords writes for r (and, with wantBytes, those record bytes and offsets: the SGHOFF02 data block).
// Hop 7, expandReads: per input read its ReadRec (record, barcode column, packed UMI) and low-quality mask, in read order, for the reads
// PileupEngine::gpu_handoff_pass_entries hands to the pileup, from mapDistinct's readRecord, STAR's per-read entries (CR, UR) and a CR table
// (distinctBarcodes lists the distinct CRs; the host resolves each one: whitelist correction R11, barcode set) .
// Plain C++ header: device pointers are passed as plain pointers; every call is synchronous on the default stream.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "records.cuh"

namespace singlet_records_gpu {

struct DeviceRecords {   // device arrays (owned; release()), counts and timings
    SigRec* sigs = nullptr; AlnRec* alns = nullptr; uint32_t* ops = nullptr; uint8_t* seq = nullptr;
    uint8_t* bytes = nullptr; uint64_t* bytesOff = nullptr;   // wantBytes: the hand-off records (seqCacheAlignRecords) and their offsets [nRec + 1]
    uint16_t* nm = nullptr;                                   // wantNm (E8): per alignment its nM (min(nMM, 65535)), parallel to alns
    uint64_t nRec = 0, nAln = 0, nOps = 0, nSeq = 0, nBytes = 0;
    uint64_t nMateGap = 0, nNoExon = 0, nBadBest = 0;   // records with a transcript seqCacheLeanAln hands to alignBAM / without exons / trBest = trInit
    uint64_t nMapped = 0;                               // records with >= 1 alignment
    float msCount = 0, msScan = 0, msFill = 0;          // kernel times (CUDA events)
};
// dBlob/dBlobOff: the records' blobs (record r at dBlob + dBlobOff[r]); dSeq/dSeqOff: their sequences (STAR's numeric code, 1 byte per
// base); dChrStart: STAR's chrStart. Allocates the outputs.
bool buildRecords(const e6::RecParams& P, const uint8_t* dBlob, const uint64_t* dBlobOff, const char* dSeq, const uint64_t* dSeqOff, uint64_t nRec,
                  const uint64_t* dChrStart, bool wantBytes, DeviceRecords& out, std::string& err, bool wantNm = false);
void release(DeviceRecords& r);

struct DeviceReads {
    ReadRec* reads = nullptr; uint64_t* lowq = nullptr;   // lowq: 2 words per read (PileupInput::lowq), only when a mask array was given
    uint64_t nOut = 0;
    uint64_t outcome[e6::RO_N] = {};                      // reads per ReadOutcome
    uint64_t lowQualSum = 0, qualNeqNSum = 0;             // the kept reads' quality counters (Collector::add_quality_counts64)
    bool maskAll = true;                                  // every kept read had a mask (entry flag 64 clear)
    float msEval = 0, msScan = 0, msScatter = 0;
};
// the distinct CR keys (crKey) of the reads that have a record, sorted (thrust sort + unique on the device)
bool distinctBarcodes(const uint32_t* dReadRecord, const e6::ReadEntry* dE, uint64_t nReads, uint32_t cbL, std::vector<uint64_t>& keys,
                      float& ms, std::string& err);
// dKeys/dVals: the CR table (sorted keys; column >= 0, -1 not in the barcode set, -2 rejected); dQ: STAR's per-read low-quality masks
// (16 bytes per read) or null
bool expandReads(const uint32_t* dReadRecord, const e6::ReadEntry* dE, const uint64_t* dQ, uint64_t nReads, const DeviceRecords& R, uint32_t cbL,
                 uint32_t umiL, const uint64_t* dKeys, const int32_t* dVals, uint64_t nKeys, DeviceReads& out, std::string& err);
void release(DeviceReads& r);

} // namespace singlet_records_gpu
