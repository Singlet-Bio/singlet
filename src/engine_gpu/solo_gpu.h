// E9: STARsolo's Gene outputs from the device records and the per-read entries (docs/gpu-engine/fable/E9-solo.md). Wired into the engine by
// E11 (docs/gpu-engine/fable/E11-solowire.md, SINGLET_ENGINE_GPU_E2E=5): the keep API at the end holds on the device what soloRun needs after
// STAR's loop, soloRunKept runs it there, soloWrite writes STAR's files (and Gene/Summary.csv when given SoloSummaryIn).
//   soloRun:   stage (b) per record the Gene annotation (classifyAlign on the record's blob), per read its record's annotation (or the host's
//              for a read the host mapped); stage (a) per distinct CR the whitelist matches, per read the UMI check, Barcodes.stats and the
//              exact-match histogram, per distinct multi-match CR the posterior; stage (c) per read Solo's record/inputRecords rule: flags,
//              Features.stats, CellReads.stats rows, the (CB, gene, UMI) keys; sort + unique + 1MM_All (Hamming-1 components) per (CB, gene).
//              device = true runs it on the GPU (CUB sorts), false on the host (OpenMP loops over the same functions, parallel std::sort).
//   soloWrite: CellRanger2.2 filtering and STAR's files, byte for byte (Barcodes.stats, Gene/Features.stats, Gene/raw, Gene/filtered,
//              Gene/UMIperCellSorted.txt, Gene/CellReads.stats).
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>
#include "solo.cuh"

namespace e9 {

// E15: the CB quality characters of the reads whose CR matches several whitelist entries with one mismatch (after the UMI check), which the
// posterior takes per read. Called once, after the CB stage, with those reads (idx ascending; cr3[k]: read idx[k]'s CR as soloRun has it; host
// memory): writes cbL characters per read to out[k * cbL ..] (what STAR's getCBandUMI has in cbQual: the barcode read's quality line, 'H' past
// its end) and returns true, or false with why (the caller checks that the line it read has the read's CR)
using CbQualFn = std::function<bool(const u32* idx, const u64* cr3, u64 n, uint8_t* out, std::string& why)>;

struct SoloModel {                       // host arrays (Transcriptome.cpp, ParametersSolo.cpp)
    std::vector<u64> wl;                 // sorted, deduplicated whitelist (2 bits per base, first base highest)
    // E15: per quality character c (0..255) the posterior's factor std::pow(10.0, -qin/10.0), qin = min((char)c - QSbase, QSmax) in STAR's char
    // arithmetic (SoloReadFeature_inputRecords.cpp:104-107; ParametersSolo.cpp:287-288), computed with the host's pow; needed with SoloReads::cbQual
    std::vector<double> pqTab;
    std::vector<u64> trS, trE, trEmax;
    std::vector<u32> trExI, trGene, exSE;
    std::vector<uint16_t> trExN;
    std::vector<uint8_t> trStr;
    std::vector<std::string> geID, geName;   // geneInfo.tab
};
struct SoloRecs {                        // the distinct sequences' records (mapDistinct's blobs)
    u64 nRec = 0;
    bool dev = false;                    // the pointers below are device pointers, used in place (soloRun(device) only)
    const RecAnn* recAnn = nullptr;      // per record its Gene annotation, already computed (soloKeepRecords): the blobs are not read
    const uint8_t* blob = nullptr; u64 blobBytes = 0;   // blobBytes includes >= 16 bytes of padding
    const u64* blobOff = nullptr;        // nRec + 1
    const u64* seqOff = nullptr;         // nRec + 1 (record sequence length = seqOff[r+1] - seqOff[r])
    e6::RecParams RP{};
};
struct SoloReads {                       // per read, in read order
    u64 n = 0;
    bool dev = false;                    // cr3, ur3, rec are device pointers, used in place (soloRun(device) only)
    const u64* cr3 = nullptr;            // CR, 3 bits per base (HandoffReadEntry packing)
    const u64* ur3 = nullptr;            // UR
    const u32* rec = nullptr;            // record index, 0xFFFFFFFF: no record (the host mapped the read)
    const RecAnn* hostAnn = nullptr;     // the host's annotation for reads without a record (indexed by read)
    u64 nHost = 0;                       // or sparse, in host memory: hostAnnS[k] is the annotation of read hostIdx[k] (a read without a record)
    const u32* hostIdx = nullptr;
    const RecAnn* hostAnnS = nullptr;
    CbQualFn cbQual;   // E15 (CbQualFn); not set: every candidate takes SoloParams::pq (the run's CB bases all give one qin)
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
    u64 nMulti = 0, nMultiAccepted = 0;                        // E15: reads with several matches; their posterior chose a CB
    bool perReadQual = false;                                  // E15: their posterior took the reads' own CB qualities (SoloReads::cbQual)
    std::vector<i32> readCb;                                   // E15, per read (keepPerRead): the CB the read gets (single match or posterior), -1 none
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
// Gene/Summary.csv (SoloFeature_statsOutput.cpp:7-74) needs, besides the counts, STAR's mapping counters (g_statsAll) and the Q30 base counts of
// the CB+UMI (SoloReadBarcode::qualHist) and of the RNA read inside its clip (ReadAlign::qualHist); bases with quality >= base + 30 and all bases
struct SoloSummaryIn { u64 readN = 0, mappedU = 0, mappedM = 0, q30CbUmi = 0, nCbUmi = 0, q30Rna = 0, nRna = 0; u32 qualBase = 33; };
// outPrefix = the star_Solo.out/ directory (with the trailing '/'). rowInsertOrder: the CellReads.stats keys in the order STAR's
// unordered_map received them (empty: rows sorted by whitelist index). sum: also Gene/Summary.csv.
bool soloWrite(const std::string& outPrefix, const SoloParams& S, const SoloModel& M, const SoloResult& r, const SoloFilterParams& FP,
               const std::vector<u32>& rowInsertOrder, SoloWriteInfo& info, std::string& log, const SoloSummaryIn* sum = nullptr);

// ---- E11: the engine's use. mapDistinct keeps on the device, while it holds it, what soloRun needs after STAR's loop (the records leave the
// device before the loop); soloRunKept runs soloRun(device) on them once STAR has mapped the reads without a record.
struct SoloKeep {
    bool reads = false, recs = false;    // soloKeepReads / soloKeepRecords done
    u64 n = 0, nRec = 0, nOverflow = 0;  // reads, records, records with an exon overflow (must be 0)
    u64 *cr3 = nullptr, *ur3 = nullptr;  // device, per read
    u32* rec = nullptr;                  // device, per read: its record or 0xFFFFFFFF
    RecAnn* recAnn = nullptr;            // device, per record
    double msReads = 0, msRecs = 0;      // the two steps (ms, device included)
    // E15 (E11 §5.3): soloKeepToHost moves the four arrays to page-locked host memory on a thread while STAR's loop runs (the device's memory is
    // free for another sample); soloRunKept puts them back on the device. Until the thread is joined, only it touches the fields below and the
    // device pointers above.
    std::thread toHost;
    bool onHost = false;                 // the arrays are in h* (the device copies freed)
    u64 *hcr3 = nullptr, *hur3 = nullptr; u32* hrec = nullptr; RecAnn* hrecAnn = nullptr;
    double msToHost = 0, msToDev = 0;    // the copies (ms; to the host: page-locked allocation included)
    std::string hostWhy;                 // why the arrays stayed on the device
};
SoloKeep& soloKeep();
// per read CR and UR from E10's aux words (device; `stride` words per read, cr3 and ur3 first) into contiguous device arrays
bool soloKeepReads(const u64* dAux, u64 stride, u64 n, std::string& log);
// per record its Gene annotation (stage (b) on the blobs, while they are on the device) and a copy of the read -> record map (device)
bool soloKeepRecords(const SoloParams& S, const SoloModel& M, const e6::RecParams& RP, const uint8_t* dBlob, const u64* dBlobOff, const u64* dSeqOff,
                     u64 nRec, const u32* dRec, u64 n, std::string& log);
void soloKeepRelease();   // joins soloKeepToHost's thread first
// E15: start the thread that moves the kept arrays to page-locked host memory (on `device`); when a step fails they stay on the device
void soloKeepToHost(int device);
// soloRun(device) on the kept arrays (on `device`, selected for the calling thread; back from the host first after soloKeepToHost) plus the
// annotation of the reads the host mapped (sparse, host memory, ascending read index); cbQual: SoloReads::cbQual (E15); releases the kept arrays
bool soloRunKept(int device, const SoloParams& S, const SoloModel& M, u64 nHost, const u32* hostIdx, const RecAnn* hostAnn, SoloResult& out,
                 std::string& log, const CbQualFn& cbQual = {});

}  // namespace e9
