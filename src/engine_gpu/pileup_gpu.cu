// singlet GPU engine: the pileup stage (API: include/singlet/engine_gpu.h, pileupRun).
// The kernels are PR #61's bench/gpu/pileup (copied unchanged into gpu_pileup/, with M6's dedup / UMI collapse / CSR code); this file
// only replaces that program's file loading with the engine's in-memory inputs. Identity with the CPU engine: exploration.md 12.13
// (sorted path), 12.19 (order-free stream path, R1 consensus: every output identical on pbmc1k_v3, job 790376).
#define PILEUP_NO_MAIN
#include "gpu_pileup/pileup/pileup.cu"
#include "singlet/engine_gpu.h"
#include "singlet/tail_trace.h"   // E16: the level-5 switches (digest)

static_assert(sizeof(IV) == sizeof(singlet_engine_gpu::PileupIv), "IV layout");
static_assert(sizeof(SnpRow) == sizeof(singlet_engine_gpu::PileupSnpRow), "SnpRow layout");
static_assert(sizeof(T3) == sizeof(singlet_engine_gpu::PileupTriplet), "T3 layout");

namespace singlet_engine_gpu {

static void copyT3(const std::vector<T3>& a, std::vector<PileupTriplet>& b) {
    b.resize(a.size());
    if (!a.empty()) memcpy(b.data(), a.data(), a.size() * sizeof(T3));
}

bool pileupRun(const PileupInput& pi, int device, PileupOutput& po) {
    po = PileupOutput();
    const double t0 = now_s();
    // E16: wall time of each host-side step (the GPU time is po.msGpu); appended to po.log
    std::string e16; double e16t = t0;
    auto e16step = [&](const char* nm) { const double t = now_s(); char b[64]; snprintf(b, sizeof b, "%s %.3f, ", nm, t - e16t); e16 += b; e16t = t; };
    cudaInitDevice(device, cudaDeviceScheduleBlockingSync, 0);
    cudaGetLastError();   // E16: the hand-off pass may have kept its context (E2EPassIn::keepPinned): then the initialisation fails, harmlessly
    if (cudaSetDevice(device) != cudaSuccess) { po.log = "cannot select device"; return false; }
    cudaFree(0);   // the context now, so that its creation is timed on its own
    e16step("context");
    if (pi.umiLen > 13) { po.log = "UMI longer than 13 nt"; return false; }
    // ---- the research program's Par / Anno from the engine's structures
    Par par;
    par.stranded = pi.stranded; par.reverse_strand = pi.reverseStrand; par.min_mapq = pi.minMapq; par.min_baseq = pi.minBaseq;
    par.count_introns = pi.countIntrons; par.count_sj = pi.countSj; par.count_mt = pi.countMt; par.umi_dedup = pi.umiDedup;
    par.directional = pi.directional; par.umi_len = pi.umiLen; par.chrm_tid = pi.chrmTid; par.n_snps = pi.nSnps;
    par.n_barcodes = pi.nBarcodes; par.n_tid = pi.nTid; par.n_exons = pi.nExons; par.n_introns = pi.nIntrons; par.n_genes = pi.nGenes;
    par.has_gene_model = pi.hasGeneModel;
    Anno an;
    an.ex.resize(pi.exonTrees.size()); an.in.resize(pi.intronTrees.size()); an.snp.resize(pi.snps.size());
    for (size_t t = 0; t < pi.exonTrees.size(); t++) { an.ex[t].resize(pi.exonTrees[t].size()); if (!an.ex[t].empty()) memcpy(an.ex[t].data(), pi.exonTrees[t].data(), an.ex[t].size() * sizeof(IV)); }
    for (size_t t = 0; t < pi.intronTrees.size(); t++) { an.in[t].resize(pi.intronTrees[t].size()); if (!an.in[t].empty()) memcpy(an.in[t].data(), pi.intronTrees[t].data(), an.in[t].size() * sizeof(IV)); }
    for (size_t t = 0; t < pi.snps.size(); t++) { an.snp[t].resize(pi.snps[t].size()); if (!an.snp[t].empty()) memcpy(an.snp[t].data(), pi.snps[t].data(), an.snp[t].size() * sizeof(SnpRow)); }
    an.e2g = pi.exonToGene; an.i2g = pi.intronToGene; an.gstr = pi.geneStrand;
    { uint32_t mx = 0; for (auto& v : an.snp) for (auto& sr : v) mx = std::max(mx, sr.idx + 1); par.n_snps = std::max(par.n_snps, mx); }
    // E16: an array's leading part may be read in place (PileupInput::Pre), the vector holding the rest
    const size_t nS = (pi.preSigs.n + pi.sigs.size()) / sizeof(SigRec), nA = (pi.preAlns.n + pi.alns.size()) / sizeof(AlnRec),
                 nR = (pi.preReads.n + pi.reads.size()) / sizeof(ReadRec);
    e16step("annotation copied");
    // ---- device memory: one arena, sized from the free memory (the research program reserved 80 GB; pbmc1k_v3 peaks at ~25 GB
    // without the annotation bug)
    size_t freeB = 0, totB = 0; cudaMemGetInfo(&freeB, &totB);
    // a sample whose working set exceeds the device (GSM7656149: 223 M barcoded reads needed ~100 GB of 94) runs again in unified
    // memory twice the device's size: the part beyond the device pages to the host, the results are the same
    for (int attempt = 0; attempt < 2; attempt++) {
    Arena A;
    try {
    if (attempt == 0) {
        size_t arena = freeB > (size_t(3) << 30) ? freeB - (size_t(3) << 30) : freeB / 2;
        // SINGLET_GPU_PILEUP_ARENA_GB: cap the arena (samples sharing the GPU; the partitions K adapt to it)
        if (const char* ag = getenv("SINGLET_GPU_PILEUP_ARENA_GB")) arena = std::min<size_t>(arena, (size_t) (atof(ag) * (1ull << 30)));
        A.init(arena);
    }
    else A.init_managed(2 * totB);
    e16step("arena");
    Inputs in; in.nS = (int)nS; in.nA = (int)nA; in.nR = (int)nR; in.ncells = (int)par.n_barcodes; in.ulen = pi.umiLen; in.par = par; in.wantMerges = pi.wantGeneMerges;
    in.dp = DPar{par.stranded, par.reverse_strand, par.count_introns, par.count_sj, par.count_mt, par.min_mapq, par.chrm_tid, pi.snpSkipN, pi.intronRules, pi.useQual};
    in.EX = upload_tree(A, an.ex); in.IN = upload_tree(A, an.in); in.e2g = to_dev(A, an.e2g); in.i2g = to_dev(A, an.i2g); in.gstr = to_dev(A, an.gstr);
    in.gmt = to_dev(A, pi.geneIsMt); in.ngmt = (uint32_t)pi.geneIsMt.size();
    in.ne2g = (uint32_t)an.e2g.size(); in.ni2g = (uint32_t)an.i2g.size(); in.ngstr = (uint32_t)an.gstr.size();
    {
        std::vector<int32_t> P; std::vector<uint32_t> I, off(an.snp.size()), n(an.snp.size()); std::vector<char> alt, ref;
        for (size_t t = 0; t < an.snp.size(); t++) {
            off[t] = (uint32_t)P.size(); n[t] = (uint32_t)an.snp[t].size();
            for (auto& s : an.snp[t]) { P.push_back(s.pos); I.push_back(s.idx); alt.push_back(s.alt); ref.push_back(s.ref); }
        }
        in.SN.pos = to_dev(A, P); in.SN.idx = to_dev(A, I); in.SN.alt = to_dev(A, alt); in.SN.ref = to_dev(A, ref); in.SN.off = to_dev(A, off);
        in.SN.n = to_dev(A, n); in.SN.ntid = (int)an.snp.size();
    }
    e16step("annotation H2D");
    auto devBytes = [&](const std::vector<uint8_t>& v) { uint8_t* d = A.alloc<uint8_t>(std::max<size_t>(1, v.size())); if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size(), cudaMemcpyHostToDevice)); return d; };
    // E16: [pre part | vector part] into one device array (the same bytes, the same arena allocation as the vector alone would take)
    auto devCat = [&](const PileupInput::Pre& pre, const void* v, size_t vb, size_t elem) {
        uint8_t* d = A.alloc<uint8_t>(std::max<size_t>(1, pre.n + vb));
        if (pre.n) CK(cudaMemcpy(d, pre.p, pre.n, cudaMemcpyHostToDevice));
        if (vb) CK(cudaMemcpy(d + pre.n, v, vb, cudaMemcpyHostToDevice));
        (void) elem; return d;
    };
    const bool cat = pi.preSigs.n || pi.preAlns.n || pi.preOps.n || pi.preSeq.n || pi.preReads.n || pi.preLowq.n;
    if (!cat) {
    in.S = (SigRec*)devBytes(pi.sigs); in.Al = (AlnRec*)devBytes(pi.alns); in.R = (ReadRec*)devBytes(pi.reads);
    in.ops = to_dev(A, pi.ops); in.seq = devBytes(pi.seq);
    in.Q = pi.useQual ? (const uint64_t*) devBytes(pi.lowq) : nullptr;   // 16 bytes per read (PileupInput::lowq)
    } else {
    in.S = (SigRec*) devCat(pi.preSigs, pi.sigs.data(), pi.sigs.size(), sizeof(SigRec));
    in.Al = (AlnRec*) devCat(pi.preAlns, pi.alns.data(), pi.alns.size(), sizeof(AlnRec));
    in.R = (ReadRec*) devCat(pi.preReads, pi.reads.data(), pi.reads.size(), sizeof(ReadRec));
    in.ops = (uint32_t*) devCat(pi.preOps, pi.ops.data(), pi.ops.size() * 4, 4);
    in.seq = devCat(pi.preSeq, pi.seq.data(), pi.seq.size(), 1);
    in.Q = pi.useQual ? (const uint64_t*) devCat(pi.preLowq, pi.lowq.data(), pi.lowq.size(), 16) : nullptr;
    }
    e16step("records H2D");
    // ---- one run with outputs
    { unsigned long long z[5] = {0, 0, 0, 0, 0}; CK(cudaMemcpyToSymbol(g_molstat, z, sizeof z)); }
    g_cell_stats = true;
    { const char* pk = getenv("SINGLET_GPU_PILEUP_PARTS"); g_force_parts = pk ? std::max(0, std::min(atoi(pk), kMaxParts)) : 0; }
    g_e16_aasplit = singlet_tail::cut("aasplit");   // E16 (level 5): the per-alignment results split on the device before their download
    Timer T; T.start();
    Outputs o;
    run_pipeline(A, T, in, &o);
    e16step("run_pipeline");
    { unsigned long long ms[5]; CK(cudaMemcpyFromSymbol(ms, g_molstat, sizeof ms));
      po.molDropped = ms[4]; po.molSpliced = ms[0]; po.molUnspliced = ms[1]; po.molAmbiguous = ms[2]; po.molLegacyDouble = ms[3]; }
    // E16 "digest" (level 5): skipped; it hashes every output tuple (4.3 GB of chrM tuples on pbmc1k_v3: 2.4 s) for a log line only
    if (singlet_tail::cut("digest")) po.log += "digest skipped (E16); ";
    else {   // digest of everything the run returns (the partition-invariance gate compares it across K)
        Sha256 h; h.update(sha_out(o).data(), 64);
        { auto ind = o.indels; std::sort(ind.begin(), ind.end()); h.update(ind.data(), ind.size() * 16); }   // as a set: their order follows the stream's read order h.update(o.cell_reads.data(), o.cell_reads.size() * 4); h.update(o.cell_mol.data(), o.cell_mol.size() * 4);
        h.update(o.hits, sizeof o.hits); h.update(&o.sj_raw_sum, 8); h.update(g_snp_stat, sizeof g_snp_stat);
        unsigned long long ms[5]; CK(cudaMemcpyFromSymbol(ms, g_molstat, sizeof ms)); h.update(ms, sizeof ms);
        long long n[4] = {o.n_gene_tuples, o.n_sj_tuples, o.n_snp_tuples, o.n_mt_tuples}; h.update(n, sizeof n);
        po.log += "digest " + h.hex() + "; ";
    }
    e16step("digest");
    po.snpHits = (uint64_t) o.n_snp_tuples; po.sjHits = o.sj_raw_sum; po.mtBases = (uint64_t) o.n_mt_tuples;
    po.cellReads.swap(o.cell_reads); po.cellMolecules.swap(o.cell_mol);
    po.hitsExon = o.hits[0]; po.hitsIntron = o.hits[1]; po.umiRecorded = o.hits[2]; po.mtRescued = o.hits[3];
    po.alnGene.swap(o.aa_gene); po.alnStat.swap(o.aa_stat); po.alnExonGene.swap(o.aa_exg);
    for (auto& kv : T.acc) if (kv.first != "fetch") po.msGpu += kv.second;   // "fetch": the host copying a partition's results out
    { char b[160]; snprintf(b, sizeof b, "partition select %.1f ms (in the GPU time), host fetch between partitions %.1f ms; ", T.acc.count("part.select") ? T.acc["part.select"] : 0.0, T.acc.count("fetch") ? T.acc["fetch"] : 0.0); po.log += b; }
    copyT3(o.exon, po.exon); copyT3(o.intron, po.intron); copyT3(o.amb, po.amb); copyT3(o.sj, po.sj);
    copyT3(o.snp_ad, po.snpAd); copyT3(o.snp_dp, po.snpDp); copyT3(o.mt, po.mt); copyT3(o.mt_rev, po.mtRev);
    po.sjtab = o.sjtab; po.indels = o.indels;
    {   // R9 per gene: the merged gene-store UMIs, (cell, gene, umi) order
        std::sort(o.merges.begin(), o.merges.end(), [](const auto& x, const auto& y) { return std::tie(x.cell, x.gene, x.umi) < std::tie(y.cell, y.gene, y.umi); });
        po.geneMerges.resize(o.merges.size());
        for (size_t i = 0; i < o.merges.size(); i++) po.geneMerges[i] = PileupGeneMerge{o.merges[i].cell, o.merges[i].gene, o.merges[i].umi, o.merges[i].root};
    }
    po.snpMolecules = g_snp_stat[0]; po.snpAmbiguous = g_snp_stat[1]; po.snpOther = g_snp_stat[2];
    po.peakGB = A.peak / 1e9;
    po.log += "cell partitions: " + std::to_string(g_parts_used) + (g_force_parts > 0 ? " (forced)" : "") + "; ";
    for (auto& l : g_parts_log) po.log += l + "; ";
    e16step("outputs copied");
    cudaFree(A.base);
    e16step("arena freed");
    cudaDeviceReset();
    e16step("device reset");
    po.msTotal = (now_s() - t0) * 1e3;
    po.log += "e16 steps (s): " + e16 + "; ";
    if (attempt > 0) po.log += "unified memory (the device arena was too small); ";
    return true;
    } catch (const ArenaExhausted& e) {
        cudaDeviceSynchronize();
        cudaFree(A.base);
        if (attempt > 0) {
            po.log += "GPU memory: " + std::to_string(e.need >> 20) + " MB needed, " + std::to_string(e.cap >> 20) + " MB unified arena";
            cudaDeviceReset();
            return false;
        }
    }
    }
    return false;
}

} // namespace singlet_engine_gpu
