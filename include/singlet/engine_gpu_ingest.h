// SPDX-License-Identifier: MIT
#pragma once
// E13 (docs/gpu-engine/fable/E13-ingest.md; SINGLET_ENGINE_GPU_INGEST=1): on the GPU path the .1fq goes from disk into the prefill's packed
// buffers directly. The pipeline no longer decodes it to FASTQ temp files: it scans the blocks' heads (barcode counts for barcode discovery,
// the 5' prefix probe, the per-block read and base counts), forks STAR with no read files, and STAR's prefill decodes the blocks on the job's
// threads straight into the arrays below (src/star/SeqCacheGpuIngest.cpp). STAR's loop takes the reads the device did not cover from a read
// source built from the same arrays (level 4/5), or from a chunk source over the .1fq (read_source.h) where the prefill does not run.
// R11's exact CB histogram goes back to the pipeline through the E8 region's header (E2EHeader::r11Magic; the handshake is below).
//
// Plain C++ header (no CUDA, no STAR, no .1fq headers): included by the pipeline (src/pipeline/singlet.cpp) and STAR's prefill.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include "singlet/engine_gpu.h"

namespace singlet::fq { class RunSet; }

namespace singlet_engine_gpu {
namespace ingest {

// ---- The prefill's packed-read input: the reads of one sample, in STAR's read order (read r is STAR's iReadAll r + 1), as a struct of arrays.
//
// The .1fq decode (SeqCacheGpuIngest.cpp) fills it from RunSet::decode, block by block, each block into its own ranges (the pipeline's head
// scan gave every block's read and base counts, so every array is sized once and written once, in parallel). What a .1fg2 reader (fg2_format.h,
// the staged format: cDNA plane, corrected-CB index, UMI plane, quality plane per chunk) would fill instead, field by field:
//   seq, off      its cDNA plane unpacked to STAR's numeric code, and the read lengths (no text; N is code 4)
//   aux           E10's per-read words: CR and UR 3-bit packed from its CB and UMI planes (CR = the raw barcode the corrected index came
//                 from), then the low-quality bits of the cDNA read from its quality plane (Phred < the pileup's minimum base quality)
//   qual          its quality plane as is when it holds 2-bit codes (qualPhred maps a code to Phred)
//   r1            the barcode read (CB + UMI, plus anything else the protocol keeps in it) 3-bit packed; its quality is 'F' (a .1fq's)
//   first, nameAt the chunk table: chunk b holds reads [first[b], first[b + 1]), named "@r<nameAt[b] + (r - first[b])>"
//   q30*, cbQ*    E11's quality counts (Gene/Summary.csv's Q30 rows, the posterior's CB quality); a .1fg2 can store them per chunk
//   r11           R11's exact CB histogram: with a corrected-CB index, a count per whitelist entry over the reads whose CR is exact
// The engine takes seq, off and aux (swapped into Input, the layout it already reads); the host side of E10 (Unmapped.out records, level
// 4/5's read source, E11's RNA Q30 adjustment) reads the text of a read back from seq, qual and r1.
struct PackedReads {
    uint64_t n = 0;                          // reads
    NoInitVector<char> seq;                  // cDNA (mate 1 for STAR), STAR's numeric code A0 C1 G2 T3 N4; read r is [off[r], off[r + 1])
    NoInitVector<uint64_t> off;              // n + 1
    uint32_t nw = 0;                         // E10: quality words per read (0: no aux)
    NoInitVector<uint64_t> aux;              // E10: per read cr3, ur3, nw words (Input::E10::aux's layout)
    // cDNA quality: 2-bit codes, 4 bases per byte (base k of block b at bits 2 * (k & 3) of byte qualAt[b] + k / 4; blocks start on a byte)
    NoInitVector<uint8_t> qual;
    uint8_t qualPhred[4] = {37, 37, 37, 37}; // Phred of each code (the .1fq's quality mode: BINNED4 6/15/25/37, ILLUMINA4, BINNED2, none 'F')
    // barcode read (mate 2): fixed length r1L, 3-bit codes, 21 bases per word, r1w words per read
    uint32_t r1L = 0, r1w = 0;
    NoInitVector<uint64_t> r1;
    // per block (nBlocks + 1 entries): first read, first cDNA base, first quality byte; and the name of the block's first read
    std::vector<uint64_t> first, baseAt, qualAt, nameAt;
    // E11's counts over the same pass: cDNA quality characters >= qBase + 30 and all; the barcode read's CB and UMI spans; CB characters
    uint64_t q30Rna = 0, nRna = 0, q30Cb = 0, nCb = 0;
    int cbQmin = 256, cbQmax = -1;
    // R11: per whitelist entry (STAR's cbWL order) the reads whose CB is that entry exactly and whose UMI has no N and is not a homopolymer
    std::vector<uint64_t> r11;
    uint64_t r11Total = 0;
    double sec = 0;                          // the decode's wall time

    uint32_t block(uint64_t r) const { return (uint32_t) (std::upper_bound(first.begin(), first.end(), r) - first.begin()) - 1; }
    uint8_t qcode(uint32_t b, uint64_t pos) const {   // pos: the base's index in seq
        const uint64_t k = pos - baseAt[b];
        return (uint8_t) ((qual[qualAt[b] + (k >> 2)] >> (2 * (k & 3))) & 3);
    }
    uint8_t r1code(uint64_t r, uint32_t j) const { return (uint8_t) ((r1[r * r1w + j / 21] >> (3 * (j % 21))) & 7); }
    uint64_t name(uint32_t b, uint64_t r) const { return nameAt[b] + (r - first[b]); }
    void release() { PackedReads e; swap(e); }
    void swap(PackedReads& o) {
        std::swap(n, o.n); seq.swap(o.seq); off.swap(o.off); std::swap(nw, o.nw); aux.swap(o.aux); qual.swap(o.qual);
        for (int k = 0; k < 4; k++) std::swap(qualPhred[k], o.qualPhred[k]);
        std::swap(r1L, o.r1L); std::swap(r1w, o.r1w); r1.swap(o.r1); first.swap(o.first); baseAt.swap(o.baseAt); qualAt.swap(o.qualAt);
        nameAt.swap(o.nameAt); std::swap(q30Rna, o.q30Rna); std::swap(nRna, o.nRna); std::swap(q30Cb, o.q30Cb); std::swap(nCb, o.nCb);
        std::swap(cbQmin, o.cbQmin); std::swap(cbQmax, o.cbQmax); r11.swap(o.r11); std::swap(r11Total, o.r11Total); std::swap(sec, o.sec);
    }
};

// ---- What the pipeline's head scan hands to STAR's process (set before the fork; the forked process inherits it, no exec)
struct Plan {
    bool on = false;
    std::vector<std::string> paths;                      // the sample's .1fq runs
    std::shared_ptr<const singlet::fq::RunSet> runs;     // opened by the pipeline (its descriptors are inherited)
    uint32_t nBlocks = 0;
    std::vector<uint64_t> first, baseAt, nameAt;         // per block: first read given to STAR, first cDNA base, first stored read (nBlocks + 1)
    uint32_t maxR2 = 0, r1L = 0;                         // longest cDNA read, the barcode read's (fixed) length
    uint32_t dictN = 0;                                  // barcode dictionary entries
    uint8_t qualPhred[4] = {37, 37, 37, 37};             // the quality mode's Phred per 2-bit code
    uint32_t qualCodes = 1;                              // codes in use (1: no stored quality, every base 'F')
    const void* fallback = nullptr;                      // the read source the pipeline gave STAR (every read, a chunk source over the .1fq)
    int r11Fd = -1;                                      // the pipeline: its own descriptor of the E8 region (R11's histogram outlives the region's release)
    uint64_t reads() const { return first.empty() ? 0 : first.back(); }
    uint64_t bases() const { return baseAt.empty() ? 0 : baseAt.back(); }
};
inline Plan& plan() { static Plan p; return p; }

// ---- The decode (SeqCacheGpuIngest.cpp): every block of plan().runs into `out`, on `threads` threads
struct DecodeConfig {
    int threads = 1;
    bool e10 = false;                         // build aux (E10): cr3, ur3, nw quality words per read
    uint32_t nw = 0;
    int minQ = 0;                             // E10: bit j of a read's quality words is set when base j has Phred < minQ
    uint32_t cbS0 = 0, cbL = 0, umiS0 = 0, umiL = 0, cbumiL = 0, bL = 0;   // STAR's barcode layout (CB_UMI_Simple), 0-based starts
    bool e11 = false;                         // E11's quality counts
    unsigned q30 = 63;                        // readQualityScoreBase + 30, as a quality character
    const uint64_t* wl = nullptr;             // R11: STAR's whitelist (ParametersSolo::cbWL, sorted 2-bit keys, first base highest)
    uint64_t nWL = 0;
    // after a block's reads are in seq and off: STAR's per-read work on them (the CellRanger4 clip), on the decoding thread `tid`; reads
    // [r0, r1), the first one starting at seq[s0] (off[r0] is written by the thread that decodes the previous block: not read here)
    std::function<void(int tid, uint64_t r0, uint64_t r1, uint64_t s0)> perBlock;
};
bool decode(const DecodeConfig& cfg, PackedReads& out, std::string& why);

// ---- R11's handshake (E2EHeader::r11Magic). STAR's prefill writes the histogram into the E8 region right after its decode (publishR11);
// the pipeline's CB corrector (cb_correct.h, set_pending) reads it the first time a correction needs the exact counts (fetchR11). That is
// after STAR's first BAM record reached the pipeline, so after the prefill returned; when the region holds no histogram then (no region,
// or the prefill did not run or stopped before its decode), the pipeline computes it from the .1fq itself (the head scan's R11 mode).
inline bool publishR11(E2ERegion& R, const std::vector<uint64_t>& dense, uint64_t total) {
    if (R.base == nullptr) return false;
    uint64_t nnz = 0;
    for (uint64_t c : dense) nnz += c != 0;
    const uint64_t o = R.take(std::max<uint64_t>(1, nnz) * 16);
    if (o == 0) return false;
    uint64_t* p = (uint64_t*) (R.base + o);
    for (uint64_t w = 0; w < dense.size(); w++) if (dense[w]) { *p++ = w; *p++ = dense[w]; }
    E2EHeader& h = *R.hdr();
    h.oR11 = o; h.nR11 = nnz; h.r11NWL = dense.size(); h.r11Total = total;
    __sync_synchronize();
    std::memcpy(h.r11Magic, "SGR11H01", 8);
    return true;
}
// the pipeline: false when the region holds no histogram (nothing read)
inline bool fetchR11(int fd, std::vector<std::pair<uint64_t, uint64_t>>& pairs, uint64_t& nWL, uint64_t& total) {
    E2EHeader h;
    if (fd < 0 || pread(fd, &h, sizeof h, 0) != (ssize_t) sizeof h || std::memcmp(h.r11Magic, "SGR11H01", 8) != 0) return false;
    pairs.resize(h.nR11);
    const size_t bytes = (size_t) h.nR11 * 16;
    size_t got = 0;
    while (got < bytes) {
        const ssize_t k = pread(fd, (char*) pairs.data() + got, bytes - got, (off_t) (h.oR11 + got));
        if (k <= 0) return false;
        got += (size_t) k;
    }
    nWL = h.r11NWL; total = h.r11Total;
    return true;
}

}  // namespace ingest
}  // namespace singlet_engine_gpu
