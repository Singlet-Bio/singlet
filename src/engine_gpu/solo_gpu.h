// E9: STARsolo's Gene outputs from the device records and the per-read entries (docs/gpu-engine/fable/E9-solo.md). Not wired into the engine.
//   soloRun:   stage (b) per record the Gene annotation (classifyAlign on the record's blob), per read its record's annotation (or the host's
//              for a read the host mapped); stage (a) per distinct CR the whitelist matches, per read the UMI check, Barcodes.stats and the
//              exact-match histogram, per distinct multi-match CR the posterior; stage (c) per read Solo's record/inputRecords rule: flags,
//              Features.stats, CellReads.stats rows, the (CB, gene, UMI) keys; sort + unique + 1MM_All (Hamming-1 components) per (CB, gene).
//              device = true runs it on the GPU (CUB sorts), false on the host (OpenMP loops over the same functions, parallel std::sort).
//   soloWrite: CellRanger2.2 filtering and STAR's files, byte for byte (Barcodes.stats, Gene/Features.stats, Gene/raw, Gene/filtered,
//              Gene/UMIperCellSorted.txt, Gene/CellReads.stats).
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "solo.cuh"

namespace e9 {

struct SoloModel {                       // host arrays (Transcriptome.cpp, ParametersSolo.cpp)
    std::vector<u64> wl;                 // sorted, deduplicated whitelist (2 bits per base, first base highest)
    std::vector<u64> trS, trE, trEmax;
    std::vector<u32> trExI, trGene, exSE;
    std::vector<uint16_t> trExN;
    std::vector<uint8_t> trStr;
    std::vector<std::string> geID, geName;   // geneInfo.tab
};
struct SoloRecs {                        // the distinct sequences' records (mapDistinct's blobs)
    u64 nRec = 0;
    const uint8_t* blob = nullptr; u64 blobBytes = 0;   // blobBytes includes >= 16 bytes of padding
    const u64* blobOff = nullptr;        // nRec + 1
    const u64* seqOff = nullptr;         // nRec + 1 (record sequence length = seqOff[r+1] - seqOff[r])
    e6::RecParams RP{};
};
struct SoloReads {                       // per read, in read order
    u64 n = 0;
    const u64* cr3 = nullptr;            // CR, 3 bits per base (HandoffReadEntry packing)
    const u64* ur3 = nullptr;            // UR
    const u32* rec = nullptr;            // record index, 0xFFFFFFFF: no record (the host mapped the read)
    const RecAnn* hostAnn = nullptr;     // the host's annotation for reads without a record (indexed by read)
};
struct SoloResult {
    u64 bstats[B_N] = {}, fstats[S_N] = {}, noCB[F_N] = {};
    // per CB that has a count or a CellReads row (compacted on the device), ascending whitelist index: cbIdx[k], and at cbVal[k * CBV + .]
    // cbReadCount, nUMI, nGene, nReadUnique, then the F_N CellReads.stats counts of the CB's row
    std::vector<u32> cbIdx, cbVal;
    std::vector<u32> segCb, segGene, segCount;                 // the raw matrix, in (CB, gene) order
    std::vector<i32> rowCb;                                    // per read (keepPerRead): the CB whose CellReads row counts it, -1 none
    std::vector<RecAnn> readAnn;                               // per read (keepPerRead)
    std::vector<int8_t> readCbm;                               // per read cbMatch after the UMI check (keepPerRead)
    std::vector<i32> readCb0;                                  // per read first whitelist match (-1 none) (keepPerRead)
    u64 nDistinctCR = 0, nCand = 0, nKeys = 0, nTuples = 0, nSeg = 0, nRecOverflow = 0;
    std::vector<std::pair<std::string, double>> ms;            // stage timings (ms)
};
constexpr u32 CBV = 4 + F_N;
enum : u32 { CV_cbReadCount, CV_nUMI, CV_nGene, CV_nReadUnique, CV_flags };
bool soloRun(bool device, const SoloParams& S, const SoloModel& M, const SoloRecs& R, const SoloReads& Rd, SoloResult& out, bool keepPerRead,
             std::string& log);

// record-time flag of a read (SoloReadFeature_record.cpp:26-60, :95-118): what STAR writes to its Solo file (or counts as CBnotInPasslist)
E9HD inline u32 recordFlag(const RecAnn& a, i32 cbm) {
    u32 f = 0;
    if (a.nTr == 1) f |= 1u << F_genomeU; else if (a.nTr > 1) f |= 1u << F_genomeM;
    if (a.bits & RA_MITO) f |= 1u << F_mito;
    if (a.bits & RA_EXONIC) f |= 1u << F_exonic;
    if (cbm < 0) {
        if (a.ng == 1) f |= 1u << F_featureU; else if (a.ng > 1) f |= 1u << F_featureM;
        f |= 1u << F_cbMatch;
    } else if (a.nTr > 0) {
        if (a.ng > 1) f |= 1u << F_featureM; else if (a.ng == 1) f |= 1u << F_featureU;
    }
    return f;
}

struct SoloFilterParams { double nExpectedCells = 3000, maxPercentile = 0.99, maxMinRatio = 10; };   // CellRanger2.2 3000 0.99 10
struct SoloWriteInfo { u64 nCB = 0, nCells = 0, nUMImin = 0, nUMImax = 0; };
// outPrefix = the star_Solo.out/ directory (with the trailing '/'). rowInsertOrder: the CellReads.stats keys in the order STAR's
// unordered_map received them (empty: rows sorted by whitelist index).
bool soloWrite(const std::string& outPrefix, const SoloParams& S, const SoloModel& M, const SoloResult& r, const SoloFilterParams& FP,
               const std::vector<u32>& rowInsertOrder, SoloWriteInfo& info, std::string& log);

}  // namespace e9
