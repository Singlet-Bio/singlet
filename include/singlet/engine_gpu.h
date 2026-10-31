// singlet GPU engine (optional CUDA build): STAR's read mapping for the distinct cDNA sequences of a sample, on the GPU.
//
// The result of each distinct sequence is STAR's own mapping state in the encoding of STAR's sequence cache
// (src/star/ReadAlign_seqCache.cpp), byte-identical to what STAR computes for that sequence (W11, docs/gpu-engine/).
// STAR's cache is filled with these results before mapping (src/star/SeqCacheGpuPrefill.cpp), so every later stage of the
// engine runs unchanged and the outputs are those of the CPU engine. Plain C++ header: no CUDA types.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace m5 { struct Params; struct GenomeView; }

namespace singlet_engine_gpu {

// STAR's read clipping for mate 1 (ClipMate::clip, --clipAdapterType Hamming), applied to every read on the GPU before dedup, so the
// distinct sequences and the records are the clipped sequences STAR's cache is keyed by. 5': N bases, then NafterAd (STAR has no 5'
// adapter search for Hamming). 3': N bases, then the adapter (localSearch with adMMp; "polyA" is an all-A adapter), then NafterAd.
struct Clip {
    bool on = false;
    uint32_t n5 = 0, after5 = 0, n3 = 0, after3 = 0;
    std::vector<char> ad3;           // 3' adapter, numeric; empty: none
    double adMMp = 0;
    bool pre = false;                // the clip lengths come from the host (Input::pre5 / pre3): --clipAdapterType CellRanger4
};

struct Input {
    std::vector<char> seq;           // concatenated sequences, STAR's numeric code (A0 C1 G2 T3, other 4)
    std::vector<uint64_t> off{0};    // seq[off[i] .. off[i+1]) is sequence i
    Clip clip;
    std::vector<uint16_t> pre5, pre3;   // with clip.pre: per read the bases STAR's CPU clip removes from its 5' and 3' ends
    std::vector<uint8_t> cr4info;       // not read by the engine: per read STAR's CellRanger4 5' clip (what clipChunk would leave in the FASTQ)
};

struct Output {
    // one record per distinct sequence without N: sequence (numeric), mapping result (sequence-cache encoding)
    std::vector<char> seq;
    std::vector<uint64_t> seqOff{0};
    std::vector<uint8_t> blob;
    std::vector<uint64_t> blobOff{0};
    std::vector<uint16_t> clip5, clip3;   // per input read when clipping is on: bases clipped from its 5' and 3' ends
    std::vector<uint32_t> readRecord;   // per input read: index of its record above, or 0xFFFFFFFF (sequence with N, or not mapped)
    uint64_t nReads = 0, nDistinct = 0, nWithN = 0, nHostLane = 0, nFailed = 0;
    double secUpload = 0, secDedup = 0, secMapGpu = 0, secHostLane = 0;
    std::string log;
};

// true if a CUDA device of compute capability >= 8.0 is usable
bool available(std::string* why = nullptr);
int deviceCount();
std::string deviceId(int device);   // PCI bus id, e.g. 0000:4e:00.0

// map the distinct sequences of `in` on device `device`; the host lane (reads past the GPU lanes' capacities) uses
// `hostThreads` threads. `host` holds host pointers to STAR's genome and suffix arrays (G with its 200-byte padding on each side).
// saBytes / saiBytes: sizes of the packed SA and SAindex arrays behind host.SA / host.SAi.
bool mapDistinct(const m5::Params& P, const m5::GenomeView& host, uint64_t saBytes, uint64_t saiBytes, const Input& in, int device,
                 int hostThreads, Output& out);

// One sample at a time per GPU (several samples in flight per node): an exclusive lock per device, named by PCI bus id so jobs that
// see different GPUs under the same index do not wait for each other. SINGLET_ENGINE_GPU_DEVICE pins the device; otherwise the
// first free one is taken, or the process waits for device (pid mod devices). Returns the device; *lockFd is closed by unlockDevice.
inline int lockDevice(int* lockFd) {
    auto lockPath = [](int d) { return "/tmp/singlet_engine_gpu_" + deviceId(d) + ".lock"; };
    const char* dv = std::getenv("SINGLET_ENGINE_GPU_DEVICE");
    const int nDev = deviceCount() > 0 ? deviceCount() : 1;
    int device = -1;
    *lockFd = -1;
    if (dv != nullptr) {
        device = std::atoi(dv);
        *lockFd = open(lockPath(device).c_str(), O_CREAT | O_RDWR, 0666);
        if (*lockFd >= 0) flock(*lockFd, LOCK_EX);
        return device;
    }
    for (int d = 0; d < nDev; d++) {
        int fd = open(lockPath(d).c_str(), O_CREAT | O_RDWR, 0666);
        if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0) { *lockFd = fd; return d; }
        if (fd >= 0) close(fd);
    }
    device = (int) (getpid() % nDev);
    *lockFd = open(lockPath(device).c_str(), O_CREAT | O_RDWR, 0666);
    if (*lockFd >= 0) flock(*lockFd, LOCK_EX);
    return device;
}
inline void unlockDevice(int lockFd) { if (lockFd >= 0) { flock(lockFd, LOCK_UN); close(lockFd); } }

// ---- non-host screen (nonhost_gpu.cu): Phases A-C of NonHostScreener::classify_multi_batch_filtered on the GPU
struct NonhostPaths { std::string viral, bacterial, fungal, bloom; };   // empty index path = index absent (as on the CPU)
struct NonhostResult {
    std::vector<uint32_t> np;    // per read: non-host minimizers after the host filter (0 when fewer than 5)
    std::vector<uint64_t> key;   // per (read, map, species) group, in the CPU's first-insertion order: read << 33 | map << 32 | species
    std::vector<uint32_t> cnt;   // hits of the group
    uint64_t nMinimizers = 0, nHits = 0, nGroups = 0, indexBytes = 0;
    double secLoad = 0, secClassify = 0;
    std::string log;
};
// loads the three indexes and the Bloom filter into HBM (loadThreads reader threads), classifies `reads` and frees the device.
// false on any failure (the caller then runs the CPU screen).
bool nonhostClassify(const std::vector<std::string>& reads, const NonhostPaths& paths, int device, int loadThreads, NonhostResult& out);

// ---- pileup (pileup_gpu.cu; kernels in src/engine_gpu/gpu_pileup/, from PR #61 bench/gpu/pileup): the exon / intron / ambiguous
// matrices, junctions, SNP AD/DP (order-free-rules R1 consensus), chrM allele counts and indels, from per-signature alignment records.
// Record layout: src/engine_gpu/gpu_pileup/pileup/pileup_rec.h (AlnRec / SigRec / ReadRec, 16 bytes each).
struct PileupIv { int32_t s, e; uint32_t idx; };                        // = singlet::Interval
struct PileupSnpRow { int32_t pos; uint32_t idx; char ref, alt; uint16_t pad; };
struct PileupTriplet { uint32_t cell, row, val; };                      // sorted by (cell, row)
// R9 per gene: a UMI of the (cell, gene) molecule store that the directional collapse merged into another (2-bit packed UMIs, N as T)
struct PileupGeneMerge { uint32_t cell, gene; uint64_t umi, root; };
struct PileupInput {
    // annotation: the engine's own interval trees as sorted arrays (the GPU replays the CPU's pre-order traversal), gene maps, SNP table
    std::vector<std::vector<PileupIv>> exonTrees, intronTrees;
    std::vector<uint32_t> exonToGene, intronToGene;
    std::vector<char> geneStrand;
    std::vector<std::vector<PileupSnpRow>> snps;
    int stranded = 1, reverseStrand = 0, minMapq = 20, minBaseq = 10, countIntrons = 1, countSj = 1, countMt = 0, umiDedup = 1, directional = 1,
        umiLen = 12, chrmTid = -1, hasGeneModel = 1, snpSkipN = 0;
    // per-base quality: useQual = 1 when the reads' qualities are neither constant nor tied to N (snpSkipN would be -1); lowq then holds 16 bytes per read
    // (same order as `reads`): bit i = base i of the record sequence, in the record's (original read) orientation, has quality < minBaseq
    int useQual = 0;
    std::vector<uint8_t> lowq;
    uint32_t intronRules = 0;               // intron_rule::k* bits of the engine (pileup_engine.h); the GPU implements kUnique, kNoSplicedIntron, kUmiGeneCR, kMtPrefer
    std::vector<uint8_t> geneIsMt;          // per gene: on chrM (kMtPrefer)
    int wantGeneMerges = 0;   // return PileupOutput::geneMerges (the host collapses V(D)J molecules per (cell, gene), R9)
    uint32_t nSnps = 0, nBarcodes = 0, nTid = 0, nExons = 0, nIntrons = 0, nGenes = 0;
    // records, raw pileup_rec.h structs
    std::vector<uint8_t> sigs, alns, reads;   // SigRec, AlnRec, ReadRec arrays
    std::vector<uint32_t> ops;
    std::vector<uint8_t> seq;
};
struct PileupOutput {
    std::vector<PileupTriplet> exon, intron, amb, sj, snpAd, snpDp, mt, mtRev;
    std::vector<std::array<uint32_t, 3>> sjtab;      // junction id -> (tid, donor, acceptor), ids in (tid, donor, acceptor) order
    std::vector<std::array<uint32_t, 4>> indels;     // (cell, pos, len, is_ins)
    std::vector<PileupGeneMerge> geneMerges;         // only when wantGeneMerges; sorted by (cell, gene, umi)
    uint64_t snpMolecules = 0, snpAmbiguous = 0, snpOther = 0;
    // the engine's PileupStats counters these outputs replace: molecule labels (DirectionalUmiStore::MoleculeStats), SNP overlaps
    // passing the quality gate (snp_hits), distinct (cell, junction, UMI) (sj_hits), chrM bases counted (mt_pileup_bases)
    uint64_t molSpliced = 0, molUnspliced = 0, molAmbiguous = 0, molLegacyDouble = 0, molDropped = 0, mtRescued = 0, snpHits = 0, sjHits = 0, mtBases = 0;
    // per cell (barcode column): reads recorded in the molecule store and molecules after directional collapse
    // (DirectionalUmiStore::per_cell_molecule_stats: the inputs of saturation_metrics.tsv and cell_qc_metrics.tsv)
    std::vector<uint32_t> cellReads, cellMolecules;
    // count_read's counters over the gene tuples, and per alignment the classify_read gene and the serial loop's per-alignment
    // flags (1 strand vote sense, 2 antisense, 4 wrong_strand, 8 cross_gene_intron, 16 multigene)
    uint64_t hitsExon = 0, hitsIntron = 0, umiRecorded = 0;
    std::vector<uint32_t> alnGene, alnExonGene;   // alnExonGene: the engine's determine_exon_gene per alignment
    std::vector<uint8_t> alnStat;
    double msGpu = 0, msTotal = 0, peakGB = 0;
    std::string log;
};
// Runs the pileup on `device`. NOTE: the kernels' CUDA error checks still exit the process (research code); callers isolate it.
bool pileupRun(const PileupInput& in, int device, PileupOutput& out);

} // namespace singlet_engine_gpu
