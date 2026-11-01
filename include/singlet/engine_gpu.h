// singlet GPU engine (optional CUDA build): STAR's read mapping for the distinct cDNA sequences of a sample, on the GPU.
//
// The result of each distinct sequence is STAR's own mapping state in the encoding of STAR's sequence cache
// (src/star/ReadAlign_seqCache.cpp), byte-identical to what STAR computes for that sequence (W11, docs/gpu-engine/).
// STAR's cache is filled with these results before mapping (src/star/SeqCacheGpuPrefill.cpp), so every later stage of the
// engine runs unchanged and the outputs are those of the CPU engine. Plain C++ header: no CUDA types.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <string>
#include <vector>
#include <array>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace m5 { struct Params; struct GenomeView; }
namespace e9 { struct SoloModel; struct SoloParams; }   // E11: src/engine_gpu/solo_gpu.h

namespace singlet_engine_gpu {

// ---- E8 (SINGLET_ENGINE_GPU_E2E=1; docs/gpu-engine/fable/E8-e2e.md): the GPU pileup's input built on the device, no hand-off file.
// The pipeline forks STAR, and STAR's prefill (mapDistinct) is where the mapping results are on the device; the pileup runs in the parent.
// The parent creates an anonymous shared memory object (memfd) before the fork (e2eRegion().create(); nothing mapped, so the parent's
// VmPeak, which summary.json reports as memory_tier, does not grow by a reservation); STAR's prefill sizes it (sparse) and maps it (attach),
// mapDistinct builds the pileup's records on the device from its device-resident results and copies them into it, STAR's mapping threads
// write the per-read entries into it, and the parent maps the bytes in use (mapUsed) and takes the records and the per-read pass from it.
// Layout: this header at offset 0, then arrays at the byte offsets below (64-byte aligned). magic is written last by the prefill.
struct E2EHeader {
    char magic[8];                     // "SGE2E001" once the records are complete
    uint64_t cap, used;                // region bytes, bytes in use
    uint64_t nReads, nRec, nAln, nOps, nSeq;
    uint32_t cbS, cbL, umiS, umiL, cbumiL, simple;    // STAR's barcode layout (CB_UMI_Simple), as in the SGHOFF02 hand-off
    uint64_t oReadRecord, oSigs, oAlns, oOps, oNm, oSeq, oEntries, oQMask;   // byte offsets (oQMask 0: no masks)
    uint64_t nMateGap, nNoExon, nBadBest, nMapped;    // buildRecords' status counts
    double msGather, msBuild, msD2H;                  // device steps in mapDistinct
    // E13 (SINGLET_ENGINE_GPU_INGEST=1, engine_gpu_ingest.h): R11's exact CB histogram for the pipeline's stream CB correction, from STAR's
    // prefill: nR11 pairs (u64 whitelist index, u64 reads) at oR11 over an r11NWL-entry whitelist, r11Total reads; r11Magic "SGR11H01"
    // once written (independent of magic: written before the device runs)
    char r11Magic[8];
    uint64_t oR11, nR11, r11NWL, r11Total;
};
struct E2ERegion {
    uint8_t* base = nullptr; uint64_t cap = 0; int fd = -1;   // fd: the memfd, created by the parent before the fork (inherited by STAR)
    E2EHeader* hdr() const { return (E2EHeader*) base; }
    bool ready() const { return base != nullptr && std::memcmp(hdr()->magic, "SGE2E001", 8) == 0; }
    template <class T> T* at(uint64_t off) const { return off ? (T*) (base + off) : nullptr; }
    // reserves bytes (64-byte aligned) after what is in use; 0 if the region is full
    uint64_t take(uint64_t bytes) {
        E2EHeader* h = hdr(); const uint64_t o = (h->used + 63) & ~63ull;
        if (o + bytes > cap) return 0;
        h->used = o + bytes; return o;
    }
    // record r's alignment count (SigRec::n_aln, gpu_pileup/pileup/pileup_rec.h: 16-byte records, n_aln at byte 4)
    uint8_t recAln(uint64_t r) const { return base[hdr()->oSigs + 16 * r + 4]; }
    // parent, before the fork: the memfd (raw syscall: the toolchain's sysroot may predate glibc's memfd_create)
    bool create() {
#ifndef __NR_memfd_create
#if defined(__x86_64__)
#define __NR_memfd_create 319
#elif defined(__aarch64__)
#define __NR_memfd_create 279
#endif
#endif
#ifdef __NR_memfd_create
        if (fd < 0) fd = (int) syscall(__NR_memfd_create, "singlet-e2e", 1u /* MFD_CLOEXEC */);
#endif
        return fd >= 0;
    }
    // STAR's prefill: sized (sparse: tmpfs pages exist once written) and mapped; the header initialised
    bool attach(uint64_t bytes) {
        if (fd < 0 || base != nullptr || ftruncate(fd, (off_t) bytes) != 0) return false;
        void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) return false;
        base = (uint8_t*) m; cap = bytes;
        std::memset(base, 0, sizeof(E2EHeader)); hdr()->cap = cap; hdr()->used = sizeof(E2EHeader);
        return true;
    }
    // parent, once the prefill wrote the magic: maps the bytes in use; false (nothing mapped) before that or without E8
    bool mapUsed() {
        if (base != nullptr) return ready();
        E2EHeader h;
        if (fd < 0 || pread(fd, &h, sizeof h, 0) != (ssize_t) sizeof h || std::memcmp(h.magic, "SGE2E001", 8) != 0) return false;
        const uint64_t len = (h.used + 4095) & ~4095ull;
        void* m = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) return false;
        base = (uint8_t*) m; cap = len;
        return true;
    }
    void release() { if (base != nullptr) munmap(base, cap); base = nullptr; cap = 0; if (fd >= 0) close(fd); fd = -1; }
};
// one per process: created by the pipeline before it forks STAR (the child inherits the fd)
inline E2ERegion& e2eRegion() { static E2ERegion r; return r; }
// SINGLET_ENGINE_GPU_E2E: 1 = E8; E10 (docs/gpu-engine/fable/E10-hostloop.md) adds, cumulatively:
//   2 = the per-read entries (CR/UR, quality counters and masks, clip flags) built on the device from R1 and R2 instead of in STAR's mapping
//       threads, and the per-read products of STAR's loop for the reads with a record computed on the device: Log.final.out's counters,
//       SJ.out.tab's junctions, the list of unmapped reads (STAR's loop still does them too; SINGLET_ENGINE_GPU_E10_CHECK=1 compares the entries)
//   3 = STAR's loop takes them from the device: for a read with a record it keeps only readLoad, the barcode and the Solo record
//   4 = reads with a record are not given to STAR at all (only the reads with N and the host lane's failures are); STAR Solo misses them
//   5 = E11 (docs/gpu-engine/fable/E11-solowire.md): level 4 with STARsolo on the device (E9's library): mapDistinct keeps the reads' CR/UR,
//       the read -> record map and the records' Gene annotation on the device; after STAR's loop (which maps the reads without a record and
//       records their annotation instead of Solo's record) soloRun counts every read on the device and writes star_Solo.out; STAR's Solo
//       post-map step does not run. Where E9's port does not cover the Solo options, level 5 runs as level 3 (STAR's Solo sees every read).
//   0 (or any other value) = none of these: the GPU maps the distinct sequences and STAR's loop does the rest (the path before E8)
// The variable is read from the environment by every process (the pipeline, STAR's child, tail_trace.h). Unset, it is 0 here; the pipeline
// sets it before anything reads it (applyDefaults below), so a run of `--engine gpu` with SINGLET_GPU_PILEUP=on is at level 5 by default.
inline int e2eLevel() { const char* e = std::getenv("SINGLET_ENGINE_GPU_E2E"); return (e != nullptr && e[0] >= '1' && e[0] <= '5') ? e[0] - '0' : 0; }
inline bool e2eRequested() { return e2eLevel() >= 1; }

// E19 (docs/gpu-engine/fable/E19-release.md, GPU-PATH.md): the defaults of the GPU path. The pipeline calls this once, after its options and
// before it reads any of these variables, when --engine gpu is set. With the GPU pileup on (SINGLET_GPU_PILEUP=on):
//   - SINGLET_ENGINE_GPU_E2E unset -> 5 (E8 + E10 + E11: records, per-read products and STARsolo on the device; E16's tail cuts are on at 5);
//     a set value is kept (0: the path before E8; 1-4 as before);
//   - at level 5, SINGLET_ENGINE_GPU_INGEST unset -> 1 (E13: the .1fq decoded straight into the prefill's buffers); 0 keeps the FASTQ files.
//     Below level 5 the ingest stays opt-in, as before;
//   - SINGLET_E16_TRACE unset -> 0 (E16's "[e16]" timing marks off; =1 turns them on for the step tables).
// Without the GPU pileup nothing is set (levels 1-5 need it). The values go into the environment so that STAR's child, which the pipeline
// forks later, and every getenv() reader see the same ones. Every part falls back on its own and says so on stderr (GPU-PATH.md §3).
// Returns one line for the run's log.
constexpr int kDefaultE2ELevel = 5;
inline std::string applyDefaults() {
    const char* gp = std::getenv("SINGLET_GPU_PILEUP");
    if (!(gp != nullptr && std::strcmp(gp, "on") == 0))
        return "SINGLET_GPU_PILEUP is not \"on\": GPU mapping only (levels 1-5 and the ingest need the GPU pileup)";
    std::string l;
    if (std::getenv("SINGLET_ENGINE_GPU_E2E") == nullptr) {
        setenv("SINGLET_ENGINE_GPU_E2E", std::to_string(kDefaultE2ELevel).c_str(), 1);
        l += "level " + std::to_string(kDefaultE2ELevel) + " (default)";
    } else l += "level " + std::to_string(e2eLevel()) + " (SINGLET_ENGINE_GPU_E2E=" + std::getenv("SINGLET_ENGINE_GPU_E2E") + ")";
    const char* ig = std::getenv("SINGLET_ENGINE_GPU_INGEST");
    if (ig == nullptr && e2eLevel() == 5) { setenv("SINGLET_ENGINE_GPU_INGEST", "1", 1); l += ", ingest (default)"; }
    else if (ig != nullptr && std::strcmp(ig, "1") == 0) l += e2eLevel() >= 1 ? ", ingest (SINGLET_ENGINE_GPU_INGEST=1)" : ", no ingest (it needs a level)";
    else l += ig != nullptr ? ", no ingest (SINGLET_ENGINE_GPU_INGEST=" + std::string(ig) + ")" : ", no ingest (opt-in below level 5)";
    if (e2eLevel() == 5) { const char* c = std::getenv("SINGLET_E16"); l += (c != nullptr && c[0] == '0') ? ", E16's tail cuts off (SINGLET_E16=0)" : ", E16's tail"; }
    if (std::getenv("SINGLET_E16_TRACE") == nullptr) setenv("SINGLET_E16_TRACE", "0", 1);
    return l + "; each falls back on its own with a line on stderr";
}

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

// an allocator that default-initializes (resize does not zero-fill): the decode's large per-read arrays are first touched by the parsing
// threads that write every element, instead of a serial fill first (E11 step 4)
template <class T> struct NoInitAlloc : std::allocator<T> {
    template <class U> struct rebind { using other = NoInitAlloc<U>; };
    NoInitAlloc() = default;
    template <class U> NoInitAlloc(const NoInitAlloc<U>&) noexcept {}
    template <class U> void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) { ::new ((void*) p) U; }
    template <class U, class... A> void construct(U* p, A&&... a) { ::new ((void*) p) U(std::forward<A>(a)...); }
};
template <class T> using NoInitVector = std::vector<T, NoInitAlloc<T>>;

struct Input {
    NoInitVector<char> seq;          // concatenated sequences, STAR's numeric code (A0 C1 G2 T3, other 4); not zero-filled on resize (E13)
    NoInitVector<uint64_t> off{0};   // seq[off[i] .. off[i+1]) is sequence i
    Clip clip;
    std::vector<uint16_t> pre5, pre3;   // with clip.pre: per read the bases STAR's CPU clip removes from its 5' and 3' ends
    std::function<void()> release;      // called by mapDistinct once its last device work is done (the caller's device lock); the host assembly of the records runs after it
    std::vector<uint8_t> cr4info;       // not read by the engine: per read STAR's CellRanger4 5' clip (what clipChunk would leave in the FASTQ)
    // E8: with e2e set, mapDistinct also builds the pileup's records on the device and copies them into e2e (E2EHeader); the host lane
    // then runs before the device is released. STAR's alignBAM parameters the records need besides Params' filters:
    E2ERegion* e2e = nullptr;
    int outSAMmapqUnique = 255; uint16_t outSAMflagAND = 65535, outSAMflagOR = 0;
    // E10 (level >= 2, with e2e): per read what the host decode packs (the target ingest format: CB+UMI key and quality bits), from which the
    // device builds the per-read entries; and STAR's options the per-read products depend on
    struct E10 {
        bool on = false;
        uint32_t nw = 0;                 // quality words per read (bit j of the read's words: base j of R2 has quality < minQ)
        NoInitVector<uint64_t> aux;      // per read, stride 2 + nw: cr3, ur3 (HandoffReadEntry's packing), quality words (every word written by the decode)
        bool wantMask = true;            // the 128-bit low-quality masks (SINGLET_HANDOFF_QMASK != 0)
        bool outSJ = true, sjAll = true; // P.outSJ.yes; outSJfilterReads == "All" (else unique reads only)
        bool hook = false;               // g_handoffReadHook is set: sampled clipped reads get entry bit 2 and the host's detector bits 3-4
        // R11's exact CB histogram (bench/fable/records/r11_hist.cpp): STAR's whitelist (ParametersSolo::cbWL: sorted 2-bit keys), CB and UMI lengths
        const uint64_t* wl = nullptr; uint64_t nWL = 0; uint32_t cbL = 0, umiL = 0;
        // E11 (level 5): with these set, mapDistinct keeps on the device what soloRun needs after STAR's loop (e9::soloKeep)
        const e9::SoloModel* soloModel = nullptr; const e9::SoloParams* soloParams = nullptr;
    } e10;
};

struct Output {
    // one record per distinct sequence without N: sequence (numeric), mapping result (sequence-cache encoding). E12: allocated once at their
    // final size without a zero-fill (every element is written by mapDistinct's parallel copies)
    NoInitVector<char> seq;
    NoInitVector<uint64_t> seqOff{0};
    NoInitVector<uint8_t> blob;
    NoInitVector<uint64_t> blobOff{0};
    std::vector<uint16_t> clip5, clip3;   // per input read when clipping is on: bases clipped from its 5' and 3' ends
    std::vector<uint32_t> readRecord;   // per input read: index of its record above, or 0xFFFFFFFF (sequence with N, or not mapped)
    uint64_t nReads = 0, nDistinct = 0, nWithN = 0, nHostLane = 0, nFailed = 0;
    double secUpload = 0, secDedup = 0, secMapGpu = 0, secHostLane = 0;
    bool e2eOk = false;                 // E8: the records are in Input::e2e (arrays and nRec..nSeq; readRecord and magic are the caller's)
    // E10: what the device computed for the reads with a record (valid with e10.ok)
    struct E10 {
        bool ok = false;
        uint64_t oEntries = 0, oQMask = 0;          // region offsets of the per-read entries and masks (written by the device; zero for reads without a record)
        std::vector<uint64_t> hookReads;            // reads (with a record) in the hook's sample: entry bit 2 set; the caller ORs in the detector's bits 3-4
        // Stats (STAR's Log.final.out counters) of the reads with a record: readN, readBases, mappedReadsU, mappedReadsM, mappedMismatchesN, mappedInsN,
        // mappedDelN, mappedInsL, mappedDelL, mappedBases, splicesN[7], splicesNsjdb, unmappedOther, unmappedShort, unmappedMismatch, unmappedMulti, unmappedAll
        enum { S_readN, S_readBases, S_U, S_M, S_mm, S_insN, S_delN, S_insL, S_delL, S_bases, S_spl0, S_sjdb = S_spl0 + 7, S_uOther, S_uShort, S_uMismatch,
               S_uMulti, S_uAll, S_N };
        uint64_t stats[S_N] = {};
        double mappedPortion = 0;
        uint64_t nOddRec = 0;                       // records that are mapped with no transcript (never seen): their reads must stay on STAR's path
        std::vector<uint8_t> sj;                    // the junctions of the reads with a record, collapsed (STAR's Junction layout, 27 bytes), in compareSJ order
        uint64_t nSJ = 0, nSJentries = 0;
        int sjGapBits = 32;                         // E17: the junction sort key's gap bits (start << sjGapBits | gap; 31 on genomes above 2^32 bases)
        std::vector<uint32_t> unmappedReads;        // reads whose record is unmapped (STAR writes them to Unmapped.out.mate1/2), read order
        std::vector<uint8_t> recMapped;             // per record: 1 if its reads replay a summary (mapped), 0 if unmapped
        double msEntries = 0, msUpAux = 0, msProducts = 0, msD2H = 0;
        // R11: per whitelist entry the reads with that exact CB and a UMI without N that is not a homopolymer (empty: not computed)
        std::vector<uint32_t> r11;
        uint64_t r11Total = 0;
        double msR11 = 0;
        // E11: the Solo inputs are kept on the device (e9::soloKeep: CR/UR per read, read -> record, the records' Gene annotation)
        bool soloKept = false;
        std::string log;
    } e10;
    // E17: the sample did not fit the device memory budget (memBudget): mapDistinct failed before the allocation that would exceed it, and the
    // caller stops the run with budgetWhy (SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu: STAR maps every read instead)
    bool budgetFail = false;
    std::string budgetWhy;
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

// E17 (docs/gpu-engine/fable/E17-budget.md): one sample's device memory budget in bytes on a device of `total` bytes. SINGLET_ENGINE_GPU_MEM_GB
// (GB of 2^30 bytes; 0: no budget, E12's behaviour: lane B and the pileup's arena take the free memory less 3 GB), else (total - 2 GB) /
// SINGLET_ENGINE_GPU_SLOTS. It bounds what the sample's process holds on the device in each phase: the prefill (context, index, reads, lanes,
// records, E10/E11 products and keep: mapDistinct sizes its chunk and lane batches from it) and the pileup (its arena). The non-host screen's
// index (24.9 GB, fixed) is outside it.
inline uint64_t memBudget(uint64_t total) {
    if (const char* g = std::getenv("SINGLET_ENGINE_GPU_MEM_GB")) { const double v = std::atof(g); return v > 0 ? (uint64_t) (v * (double) (1ull << 30)) : 0; }
    const char* sl = std::getenv("SINGLET_ENGINE_GPU_SLOTS");
    const uint64_t slots = sl != nullptr ? (uint64_t) std::min(8, std::max(1, std::atoi(sl))) : 1, margin = 2ull << 30;
    return (total > margin ? total - margin : total) / slots;
}
// Samples per GPU: SINGLET_ENGINE_GPU_SLOTS (default 1 = one sample at a time) exclusive locks per device, named by PCI bus id so jobs that see different
// GPUs under the same index do not wait for each other (slot k > 0 has ".k" appended to the lock file name). Each sample keeps its device memory within
// memBudget (above: (device memory - 2 GB) / slots unless SINGLET_ENGINE_GPU_MEM_GB sets it). SINGLET_ENGINE_GPU_DEVICE pins the device; otherwise the first
// free slot of the first device is taken, or the process waits for a slot of device
// (pid mod devices). Returns the device; *lockFd is closed by unlockDevice.
inline int lockDevice(int* lockFd) {
    auto lockPath = [](int d, int k) { return "/tmp/singlet_engine_gpu_" + deviceId(d) + (k > 0 ? "." + std::to_string(k) : std::string()) + ".lock"; };
    const char* dv = std::getenv("SINGLET_ENGINE_GPU_DEVICE");
    const char* sl = std::getenv("SINGLET_ENGINE_GPU_SLOTS");
    const int slots = sl != nullptr ? std::min(8, std::max(1, std::atoi(sl))) : 1;
    const int nDev = deviceCount() > 0 ? deviceCount() : 1;
    int device = -1;
    *lockFd = -1;
    if (dv != nullptr) {
        device = std::atoi(dv);
        for (int k = 0; k < slots; k++) {
            int fd = open(lockPath(device, k).c_str(), O_CREAT | O_RDWR, 0666);
            if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0) { *lockFd = fd; return device; }
            if (fd >= 0) close(fd);
        }
        *lockFd = open(lockPath(device, (int) (getpid() % slots)).c_str(), O_CREAT | O_RDWR, 0666);
        if (*lockFd >= 0) flock(*lockFd, LOCK_EX);
        return device;
    }
    for (int k = 0; k < slots; k++)
        for (int d = 0; d < nDev; d++) {
            int fd = open(lockPath(d, k).c_str(), O_CREAT | O_RDWR, 0666);
            if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0) { *lockFd = fd; return d; }
            if (fd >= 0) close(fd);
        }
    device = (int) (getpid() % nDev);
    *lockFd = open(lockPath(device, (int) ((getpid() / nDev) % slots)).c_str(), O_CREAT | O_RDWR, 0666);
    if (*lockFd >= 0) flock(*lockFd, LOCK_EX);
    return device;
}
inline void unlockDevice(int lockFd) { if (lockFd >= 0) { flock(lockFd, LOCK_UN); close(lockFd); } }
// a slot of one given device (E11: the sample's second, short use of the device that holds its kept Solo inputs, after STAR's loop): the first
// free slot, or a wait for slot (pid mod slots). *lockFd is closed by unlockDevice.
inline void lockDeviceSlot(int device, int* lockFd) {
    const char* sl = std::getenv("SINGLET_ENGINE_GPU_SLOTS");
    const int slots = sl != nullptr ? std::min(8, std::max(1, std::atoi(sl))) : 1;
    auto lockPath = [&](int k) { return "/tmp/singlet_engine_gpu_" + deviceId(device) + (k > 0 ? "." + std::to_string(k) : std::string()) + ".lock"; };
    *lockFd = -1;
    for (int k = 0; k < slots; k++) {
        int fd = open(lockPath(k).c_str(), O_CREAT | O_RDWR, 0666);
        if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0) { *lockFd = fd; return; }
        if (fd >= 0) close(fd);
    }
    *lockFd = open(lockPath((int) (getpid() % slots)).c_str(), O_CREAT | O_RDWR, 0666);
    if (*lockFd >= 0) flock(*lockFd, LOCK_EX);
}

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
    // E16 (docs/gpu-engine/fable/E16-tail.md; level 5): each array's leading part read in place (pre*: bytes) instead of copied to the front of
    // the vector above, which then holds only the rest: the device-built records in the shared region (sigs, alns, ops, seq) and the hand-off
    // pass's kept reads and masks (reads, lowq). Element i of the whole array: PileupCat. `keep` owns what the pre parts point into.
    struct Pre { const uint8_t* p = nullptr; uint64_t n = 0; };
    Pre preSigs, preAlns, preOps, preSeq, preReads, preLowq;
    std::shared_ptr<void> keep;
};
// E16: an array of PileupInput as [pre part | vector part]; at(i) points into the part that holds element i (a record's alignments, CIGAR ops
// and sequence are in the part its signature is in)
template <class T> struct PileupCat {
    const T* a = nullptr; uint64_t na = 0; const T* b = nullptr; uint64_t nb = 0;
    const T& operator[](uint64_t i) const { return i < na ? a[i] : b[i - na]; }
    const T* at(uint64_t i) const { return i < na ? a + i : b + (i - na); }
    uint64_t size() const { return na + nb; }
};
template <class T> PileupCat<T> pileupCat(const PileupInput::Pre& pre, const void* v, uint64_t vBytes) {
    return PileupCat<T>{(const T*) pre.p, pre.n / sizeof(T), (const T*) v, vBytes / sizeof(T)};
}
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
    // E16 "resetbg" (level 5): pileupRun left the device's context (and the inputs page-locked in it) in place; the caller keeps the device
    // lock and calls pileupRelease(device) (then unlocks) before the input's memory goes
    bool resetDeferred = false;
};
// Runs the pileup on `device`. NOTE: the kernels' CUDA error checks still exit the process (research code); callers isolate it.
bool pileupRun(const PileupInput& in, int device, PileupOutput& out);
// E16: the device reset pileupRun deferred (PileupOutput::resetDeferred), from any thread
void pileupRelease(int device);

// ---- E8: the hand-off pass of the GPU pileup on the device (records_gpu.cu): PileupEngine::gpu_handoff_pass_entries' per-read rule and every
// count it keeps, for the reads of E2E records, from the region's arrays. The CR table (R11 correction, barcode set, WL-ambient index) is the
// caller's: crTable gets the distinct CR keys (3 bits per base as HandoffReadEntry packs them, sorted) and fills per key {col, wl, status}.
struct E2ECrVal { int32_t col, wl; uint32_t status; };   // = e6::CrVal
struct E2EPassIn {
    const E2ERegion* region = nullptr;
    int32_t chrmTid = -1; uint32_t minMapq = 0;          // (uint8_t) config.min_mapq
    bool cbc = false, wlOn = false, wlPend = false;      // R11 corrector present; WL-ambient counts kept; WL-ambient gene hits wanted
    uint32_t nCells = 0, nCellsQc = 0, nWl = 0;           // per_cell_reads_, per_cell_qc_, wl_umi_counts_ sizes
    // E16: the region and the kept reads' arrays (reads, lowq) stay page-locked in a context that stays (no device reset) for pileupRun's
    // uploads, which then run at the pinned rate; pileupRun's own device reset ends both
    bool keepPinned = false;
};
struct E2EPassOut {
    std::vector<uint8_t> reads;      // ReadRec per kept read, read order (= the pass's add_handoff_read order)
    std::vector<uint32_t> clip;      // per kept read: its entry flags when clipped, else 0 (Collector::clips)
    std::vector<uint8_t> lowq;       // 16 bytes per kept read (zero where the read has no mask)
    bool maskAll = true;             // every kept read had a mask
    uint64_t lowQualSum = 0, qualNeqNSum = 0;
    // PileupStats: total_reads, mapped_reads (+= alignments), secondary_reads, rrna_reads_sampled, rrna_reads (clipped reads, by the hook),
    // chrm_reads, no_barcode, barcoded_reads, multimapper_reads, low_mapq
    uint64_t totalAln = 0, secondary = 0, rrnaSampled = 0, rrnaHook = 0, chrm = 0, noBarcode = 0, barcoded = 0, multimapper = 0, lowMapq = 0;
    uint64_t nDone = 0, nUnwritten = 0, nBigClip = 0, nNoHook = 0, nNoKey = 0, nNotInSet = 0;
    uint64_t cbStatus[8] = {};
    std::vector<uint32_t> cellReads, cellUnique, cellMulti; std::vector<uint64_t> cellMism, cellBases;   // per barcode column
    std::vector<uint32_t> wlCounts;                       // per WL-ambient index
    std::vector<std::pair<uint32_t, uint32_t>> wlPending; // (record, primary's alignment index), read order
    std::vector<uint64_t> rrnaHost;                       // sampled unclipped reads: record << 1 | reverse (the host runs the detector)
    uint64_t nKeys = 0;
    double msKeys = 0, msTable = 0, msEval = 0, msScatter = 0, msUpload = 0, msDownload = 0, msTotal = 0;
    std::string log;
};
bool e2ePass(const E2EPassIn& in, int device,
             const std::function<void(const std::vector<uint64_t>& keys, std::vector<E2ECrVal>& vals)>& crTable, E2EPassOut& out);

} // namespace singlet_engine_gpu
