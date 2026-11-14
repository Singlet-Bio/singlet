// SPDX-License-Identifier: MIT
// tests/cpp/test_gpu_pileup_stream.cpp
//
// The GPU pileup's streamed layout (SINGLET_PILEUP_STREAM, default on: reads in host memory and uploaded per cell partition, transient tables in the
// arena's top region) against the earlier layout (SINGLET_PILEUP_STREAM=0) on a synthetic input, every output field. The streamed run is repeated with
// a forced K (SINGLET_GPU_PILEUP_PARTS) and with a tiny per-partition budget (SINGLET_PILEUP_STREAM_BUDGET_KB) so that the sizing picks many
// partitions on this small input; the log must show more than one partition for those runs.
// The fixture: one contig, four genes (two exons and an intron each), six SNP rows, exonic / spliced / intronic / intergenic signatures, multimappers,
// low-MAPQ alignments, reverse alignments; reads over 1000 cells with repeated UMIs, some without a barcode.
// Prints SKIP and returns 77 (ctest: skipped) in a CPU-only build or without a usable CUDA device.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "test_harness.h"
#ifdef SINGLET_HAVE_ENGINE_GPU
#include "singlet/engine_gpu.h"
#endif

#ifdef SINGLET_HAVE_ENGINE_GPU
namespace {
using namespace singlet_engine_gpu;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; uint64_t x = s; x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; return x; }
    uint32_t below(uint32_t n) { return (uint32_t)(next() % n); }
};

// pileup_rec.h's records
struct SigRecT { uint32_t aln_off; uint8_t n_aln; uint8_t nh; uint16_t rlen; uint64_t seq_off; };
struct AlnRecT { int32_t tid; int32_t pos; uint32_t ops_off; uint16_t n_ops; uint8_t flags; uint8_t mapq; };
struct ReadRecT { uint32_t sig; int32_t bc; uint64_t umi3; };
static_assert(sizeof(SigRecT) == 16 && sizeof(AlnRecT) == 16 && sizeof(ReadRecT) == 16, "record sizes");

constexpr int kGenes = 4, kRlen = 90, kCells = 1000, kSigs = 80, kReads = 24000;

template <class T> void putBytes(std::vector<uint8_t>& v, const std::vector<T>& a) {
    v.resize(a.size() * sizeof(T));
    if (!a.empty()) memcpy(v.data(), a.data(), v.size());
}

PileupInput makeInput() {
    PileupInput pi;
    // annotation: gene g's exons [1000 g + 100, +200) and [+300, +400), intron [+200, +300)
    pi.exonTrees.resize(1); pi.intronTrees.resize(1);
    for (int g = 0; g < kGenes; g++) {
        pi.exonTrees[0].push_back(PileupIv{1000 * g + 100, 1000 * g + 200, (uint32_t)(2 * g)});
        pi.exonTrees[0].push_back(PileupIv{1000 * g + 300, 1000 * g + 400, (uint32_t)(2 * g + 1)});
        pi.intronTrees[0].push_back(PileupIv{1000 * g + 200, 1000 * g + 300, (uint32_t) g});
        pi.exonToGene.push_back(g); pi.exonToGene.push_back(g);
        pi.intronToGene.push_back(g);
        pi.geneStrand.push_back(g == 2 ? '-' : '+');
    }
    pi.snps.resize(1);
    { uint32_t idx = 0; for (int g = 0; g < 3; g++) for (int off : {120, 150}) pi.snps[0].push_back(PileupSnpRow{1000 * g + off, idx++, 'A', 'C', 0}); }
    pi.stranded = 0; pi.reverseStrand = 0; pi.minMapq = 20; pi.minBaseq = 10; pi.countIntrons = 1; pi.countSj = 1; pi.countMt = 0; pi.umiDedup = 1;
    pi.directional = 1; pi.umiLen = 12; pi.chrmTid = -1; pi.hasGeneModel = 1; pi.snpSkipN = 0; pi.useQual = 0; pi.intronRules = 0;
    pi.nSnps = 6; pi.nBarcodes = kCells; pi.nTid = 1; pi.nExons = 2 * kGenes; pi.nIntrons = kGenes; pi.nGenes = kGenes;
    pi.wantReads = 1; pi.wantGeneMerges = 1; pi.wantUmiKeys = 1;
    // signatures and their alignments
    Rng r(12345);
    std::vector<SigRecT> sigs; std::vector<AlnRecT> alns; std::vector<uint32_t> ops; std::vector<uint8_t> seq;
    auto addAln = [&](int pos, bool spliced, bool primary, int mapq) {
        AlnRecT a{}; a.tid = 0; a.pos = pos; a.ops_off = (uint32_t) ops.size(); a.flags = (uint8_t)((primary ? 2 : 0) | (r.below(2) ? 1 : 0)); a.mapq = (uint8_t) mapq;
        if (spliced) { ops.push_back(50u << 4 | 0u); ops.push_back(100u << 4 | 3u); ops.push_back(40u << 4 | 0u); a.n_ops = 3; }
        else { ops.push_back((uint32_t) kRlen << 4 | 0u); a.n_ops = 1; }
        alns.push_back(a);
    };
    for (int s = 0; s < kSigs; s++) {
        const int g = (int) r.below(kGenes), kind = (int)(r.below(10));
        SigRecT sg{}; sg.aln_off = (uint32_t) alns.size(); sg.rlen = kRlen; sg.seq_off = seq.size();
        const int mapq = r.below(12) == 0 ? 3 : 255;
        if (kind < 4) { addAln(1000 * g + 105, false, true, mapq); sg.n_aln = 1; sg.nh = 1; }              // exonic (covers the SNP rows of genes 0-2)
        else if (kind < 6) { addAln(1000 * g + 150, true, true, mapq); sg.n_aln = 1; sg.nh = 1; }          // spliced
        else if (kind < 7) { addAln(1000 * g + 205, false, true, mapq); sg.n_aln = 1; sg.nh = 1; }         // intronic
        else if (kind < 8) { addAln(1000 * g + 500, false, true, mapq); sg.n_aln = 1; sg.nh = 1; }         // intergenic
        else { const int g2 = (g + 1) % kGenes; addAln(1000 * g + 105, false, true, mapq); addAln(1000 * g2 + 105, false, false, mapq); sg.n_aln = 2; sg.nh = 2; }   // multimapper
        for (int i = 0; i < kRlen / 2; i++) { static const uint8_t nt[4] = {1, 2, 4, 8}; seq.push_back((uint8_t)(nt[r.below(4)] | (nt[r.below(4)] << 4))); }
        sigs.push_back(sg);
    }
    std::vector<ReadRecT> reads;
    std::vector<uint32_t> pool; for (int i = 0; i < 40; i++) pool.push_back((uint32_t) r.next() & 0xFFFFFFu);
    for (int i = 0; i < kReads; i++) {
        ReadRecT x{}; x.sig = r.below(kSigs); x.bc = r.below(33) == 0 ? -1 : (int32_t) r.below(kCells);
        const uint32_t u = pool[r.below((uint32_t) pool.size())]; uint64_t u3 = 0;
        for (int k = 0; k < 12; k++) u3 |= (uint64_t)((u >> (2 * k)) & 3) << (3 * k);
        x.umi3 = u3; reads.push_back(x);
    }
    putBytes(pi.sigs, sigs); putBytes(pi.alns, alns); putBytes(pi.reads, reads);
    pi.ops = ops; pi.seq = seq;
    return pi;
}

template <class T> bool sameVec(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}
bool sameAmbig(const std::vector<PileupAmbig>& a, const std::vector<PileupAmbig>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) if (a[i].bc != b[i].bc || a[i].geneA != b[i].geneA || a[i].geneB != b[i].geneB || a[i].umi3 != b[i].umi3) return false;
    return true;
}
// every output field of the two runs (not the timings, peak or log)
bool sameOut(const PileupOutput& a, const PileupOutput& b, const char* tag) {
    bool ok = true;
    auto chk = [&](bool c, const char* what) { if (!c) { std::cout << "  MISMATCH (" << tag << "): " << what << "\n"; ok = false; } };
    chk(sameVec(a.exon, b.exon), "exon"); chk(sameVec(a.intron, b.intron), "intron"); chk(sameVec(a.amb, b.amb), "amb"); chk(sameVec(a.sj, b.sj), "sj");
    chk(sameVec(a.snpAd, b.snpAd), "snpAd"); chk(sameVec(a.snpDp, b.snpDp), "snpDp"); chk(sameVec(a.mt, b.mt), "mt"); chk(sameVec(a.mtRev, b.mtRev), "mtRev");
    chk(sameAmbig(a.ambig, b.ambig), "ambig"); chk(sameVec(a.exonReads, b.exonReads), "exonReads"); chk(sameVec(a.intronReads, b.intronReads), "intronReads");
    chk(a.histCell == b.histCell, "histCell"); chk(a.hist == b.hist, "hist"); chk(a.sjtab == b.sjtab, "sjtab"); chk(a.indels == b.indels, "indels");
    chk(sameVec(a.geneMerges, b.geneMerges), "geneMerges"); chk(a.umiKeys == b.umiKeys, "umiKeys");
    chk(a.snpMolecules == b.snpMolecules && a.snpAmbiguous == b.snpAmbiguous && a.snpOther == b.snpOther, "snp counters");
    chk(a.editHits == b.editHits && a.editMolecules == b.editMolecules && a.editAmbiguous == b.editAmbiguous && a.editOther == b.editOther, "edit counters");
    chk(a.molSpliced == b.molSpliced && a.molUnspliced == b.molUnspliced && a.molAmbiguous == b.molAmbiguous && a.molLegacyDouble == b.molLegacyDouble &&
        a.molDropped == b.molDropped && a.mtRescued == b.mtRescued && a.snpHits == b.snpHits && a.sjHits == b.sjHits && a.mtBases == b.mtBases, "molecule counters");
    chk(a.cellReads == b.cellReads, "cellReads"); chk(a.cellMolecules == b.cellMolecules, "cellMolecules"); chk(a.cellMixed == b.cellMixed, "cellMixed");
    chk(sameVec(a.mixedExon, b.mixedExon), "mixedExon"); chk(sameVec(a.mixedIntron, b.mixedIntron), "mixedIntron");
    chk(a.hitsExon == b.hitsExon && a.hitsIntron == b.hitsIntron && a.umiRecorded == b.umiRecorded, "count_read counters");
    chk(a.alnGene == b.alnGene, "alnGene"); chk(a.alnExonGene == b.alnExonGene, "alnExonGene"); chk(a.alnStat == b.alnStat, "alnStat");
    return ok;
}
int partsOf(const std::string& log) {   // "cell partitions: N"
    const size_t p = log.find("cell partitions: ");
    return p == std::string::npos ? -1 : atoi(log.c_str() + p + 17);
}
bool runOnce(const PileupInput& pi, PileupOutput& po) {
    const bool ok = pileupRun(pi, 0, po);
    if (po.resetDeferred) pileupRelease(0);
    return ok;
}

int run() {
    const PileupInput pi = makeInput();
    setenv("SINGLET_PILEUP_STREAM", "0", 1);
    unsetenv("SINGLET_GPU_PILEUP_PARTS"); unsetenv("SINGLET_PILEUP_STREAM_BUDGET_KB");
    PileupOutput ref;
    const bool okRef = runOnce(pi, ref);
    CHECK(okRef, "unstreamed run (SINGLET_PILEUP_STREAM=0)");
    if (!okRef) return 1;
    CHECK(ref.log.find("pileup stream off") != std::string::npos, "the log says the stream is off");
    CHECK(!ref.exon.empty() && !ref.sj.empty() && !ref.snpDp.empty(), "the fixture produces gene, junction and SNP rows");

    struct Case { const char* name; const char* stream; const char* parts; const char* budgetKb; int minParts; };
    const Case cases[] = {
        {"streamed, default budget", nullptr, nullptr, nullptr, 1},
        {"streamed, K forced to 1", nullptr, "1", nullptr, 1},
        {"streamed, K forced to 7", nullptr, "7", nullptr, 7},
        {"streamed, K forced to 16", nullptr, "16", nullptr, 16},
        {"streamed, tiny budget (400 KB)", nullptr, nullptr, "400", 3},
        {"streamed, tiny budget (150 KB)", nullptr, nullptr, "150", 3},
    };
    int rc = 0;
    for (const Case& c : cases) {
        if (c.stream) setenv("SINGLET_PILEUP_STREAM", c.stream, 1); else unsetenv("SINGLET_PILEUP_STREAM");
        if (c.parts) setenv("SINGLET_GPU_PILEUP_PARTS", c.parts, 1); else unsetenv("SINGLET_GPU_PILEUP_PARTS");
        if (c.budgetKb) setenv("SINGLET_PILEUP_STREAM_BUDGET_KB", c.budgetKb, 1); else unsetenv("SINGLET_PILEUP_STREAM_BUDGET_KB");
        PileupOutput po;
        const bool ok = runOnce(pi, po);
        CHECK(ok, c.name);
        if (!ok) { rc = 1; continue; }
        const int k = partsOf(po.log);
        std::cout << "  " << c.name << ": " << k << " cell partitions, peak " << po.peakGB << " GB (unstreamed " << ref.peakGB << " GB)\n";
        CHECK(po.log.find("pileup stream on") != std::string::npos, "the log says the stream is on");
        CHECK(k >= c.minParts, "the streamed path ran the expected number of partitions");
        CHECK(sameOut(ref, po, c.name), "outputs identical to the unstreamed run");
    }
    return rc == 0 && n_fail == 0 ? 0 : 1;
}
}  // namespace
#endif

int main() {
#ifndef SINGLET_HAVE_ENGINE_GPU
    std::cout << "SKIP: CPU-only build (no GPU engine)\n";
    return 77;
#else
    std::string why;
    if (singlet_engine_gpu::deviceCount() <= 0 || !singlet_engine_gpu::available(&why)) {
        std::cout << "SKIP: no usable CUDA device" << (why.empty() ? "" : " (" + why + ")") << "\n";
        return 77;
    }
    const int rc = run();
    std::cout << (rc == 0 ? "ALL PASSED" : "FAILED") << " (" << n_pass << " checks passed, " << n_fail << " failed)\n";
    return rc;
#endif
}
