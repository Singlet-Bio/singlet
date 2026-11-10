// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: allsite_device.h
// The device route of the all-site pass (allsite.h, D-101): plain host types shared by allsite.h's hook (Params::device), the CUDA side
// (src/engine_gpu/allsite_gpu.cu, allsiteDeviceRun) and the unit test. No CUDA types here, so a CPU-only build includes it unchanged.
//
// The host hands the device the pass's reads contig by contig (gather), and the device returns what run()'s contig workers return: the
// sites of each contig (discovery) and, per (barcode, site), the DP and AD molecule counts of the R1 vote after the R9 collapse
// (counting), with the pass's molecule statistics. The rules are allsite.h's, computed as sorts:
//  * discovery: mismatch observations (contig, pos, alt, barcode, UMI) sorted; distinct molecules and barcodes per (pos, alt).
//  * counting: read-site hits ((site, barcode), (UMI, class)) sorted; the class counts of each (barcode, site, UMI), saturated at 255 as
//    SnpConsensus::add does; R9's components per (barcode, site): two UMIs at one base apart are joined when the larger read count is at
//    least twice the smaller minus one, which is umi3_directional_roots' rule for its sorted order; the merged class counts vote as
//    SnpConsensus::finalize does.
// Every step is a function of the set of reads, so neither the thread count nor the device changes a count.
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace singlet { namespace allsite { namespace dev {

// One read, as gathered by the host: its primary alignment, barcode column and raw UMI (ReadRec::umi3), and where its CIGAR ops, packed
// sequence (original orientation, nt16 nibbles) and low-quality mask words sit in the chunk's arrays.
struct Read {
    int32_t pos; int32_t bc; uint64_t umi3;
    uint64_t lq;          // first mask word (nlq words follow)
    uint32_t ops, seq;    // first op, first sequence byte
    uint16_t rlen, nops;
    uint16_t ci;          // the contig's index within the chunk
    uint8_t rev, nlq;
};
static_assert(sizeof(Read) == 40, "dev::Read layout");

struct Contig {           // one contig with reads, in the order the host gathers them
    int32_t tid = -1;
    uint64_t reads = 0, ops = 0, seqb = 0, lqw = 0;   // sizes of its gathered arrays
    const uint8_t* ref = nullptr; uint64_t ref_len = 0;   // reference bases (STAR Genome bytes: 0-3 ACGT, else N)
};

struct ContigOut {
    std::vector<int32_t> pos; std::vector<uint8_t> ref, alt; std::vector<uint32_t> mol, cells;   // sites, by position
};

struct Input {
    std::vector<Contig> contigs;
    // fill the reads of contigs [c0, c1) back to back: reads, ops, seq, lq sized by the contigs' totals; Read::ops/seq/lq are offsets
    // from the start of those arrays and Read::ci = contig index - c0
    std::function<void(uint32_t c0, uint32_t c1, Read* reads, uint32_t* ops, uint8_t* seq, uint64_t* lq)> gather;
    std::function<void()> progress;
    // the counts of sites [s0, s0 + ns) of contig c (indices within the contig), in the spill's unit layout (allsite.h Spill): per site
    // the number of barcodes, then per barcode (ascending) the gap from the previous one, DP and AD, all varints. Called once per block
    // and contig, in block order; the device route holds no more than one block of counts.
    std::function<void(uint32_t c, uint32_t s0, uint32_t ns, std::string& bytes)> sink;
    int min_alt_molecules = 2, min_alt_cells = 2, use_qual = 0, skip_n = 0, directional = 1, umi_len = 12;
    uint32_t n_barcodes = 0;
    unsigned threads = 1;   // host threads for the pinned copies
};

struct Output {
    std::vector<ContigOut> contigs;   // parallel to Input::contigs
    uint64_t obs = 0, molecules = 0, ambiguous = 0, other = 0, ad_molecules = 0, hits = 0, entries = 0, groups = 0;
    double sec = 0, sec_gather = 0, sec_upload = 0, sec_disc = 0, sec_count = 0;
    double sec_hits = 0, sec_sort = 0, sec_entries = 0, sec_r9 = 0, sec_out = 0;   // counting stages (SINGLET_ALLSITE_GPU_TIMING=1)
    uint64_t arena_bytes = 0, chunks = 0, blocks = 0, spill_bytes = 0;
    std::string why;                  // why the device route declined or failed (the host pass then runs)
};

}}}  // namespace singlet::allsite::dev

namespace singlet_engine_gpu {
// The device route (allsite_gpu.cu, GPU builds). Returns false, with Output::why, when it cannot run (no device, too little free memory,
// SINGLET_ALLSITE_GPU=0, a layout it does not handle); the caller then runs the host pass, which gives the same result.
bool allsiteDeviceRun(const singlet::allsite::dev::Input& in, singlet::allsite::dev::Output& out);
}
