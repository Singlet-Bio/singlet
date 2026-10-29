// singlet GPU engine: the pileup stage (API: include/singlet/engine_gpu.h, pileupRun).
// The kernels are PR #61's bench/gpu/pileup (copied unchanged into gpu_pileup/, with M6's dedup / UMI collapse / CSR code); this file
// only replaces that program's file loading with the engine's in-memory inputs. Identity with the CPU engine: exploration.md 12.13
// (sorted path), 12.19 (order-free stream path, R1 consensus: every output identical on pbmc1k_v3, job 790376).
#define PILEUP_NO_MAIN
#include "gpu_pileup/pileup/pileup.cu"
#include "singlet/engine_gpu.h"

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
    cudaInitDevice(device, cudaDeviceScheduleBlockingSync, 0);
    if (cudaSetDevice(device) != cudaSuccess) { po.log = "cannot select device"; return false; }
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
    const size_t nS = pi.sigs.size() / sizeof(SigRec), nA = pi.alns.size() / sizeof(AlnRec), nR = pi.reads.size() / sizeof(ReadRec);
    // ---- device memory: one arena, sized from the free memory (the research program reserved 80 GB; pbmc1k_v3 peaks at ~25 GB
    // without the annotation bug)
    size_t freeB = 0, totB = 0; cudaMemGetInfo(&freeB, &totB);
    // a sample whose working set exceeds the device (GSM7656149: 223 M barcoded reads needed ~100 GB of 94) runs again in unified
    // memory twice the device's size: the part beyond the device pages to the host, the results are the same
    for (int attempt = 0; attempt < 2; attempt++) {
    Arena A;
    try {
    if (attempt == 0) A.init(freeB > (size_t(3) << 30) ? freeB - (size_t(3) << 30) : freeB / 2);
    else A.init_managed(2 * totB);
    Inputs in; in.nS = (int)nS; in.nA = (int)nA; in.nR = (int)nR; in.ncells = (int)par.n_barcodes; in.ulen = pi.umiLen; in.par = par;
    in.dp = DPar{par.stranded, par.reverse_strand, par.count_introns, par.count_sj, par.count_mt, par.min_mapq, par.chrm_tid, pi.snpSkipN};
    in.EX = upload_tree(A, an.ex); in.IN = upload_tree(A, an.in); in.e2g = to_dev(A, an.e2g); in.i2g = to_dev(A, an.i2g); in.gstr = to_dev(A, an.gstr);
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
    auto devBytes = [&](const std::vector<uint8_t>& v) { uint8_t* d = A.alloc<uint8_t>(std::max<size_t>(1, v.size())); if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size(), cudaMemcpyHostToDevice)); return d; };
    in.S = (SigRec*)devBytes(pi.sigs); in.Al = (AlnRec*)devBytes(pi.alns); in.R = (ReadRec*)devBytes(pi.reads);
    in.ops = to_dev(A, pi.ops); in.seq = devBytes(pi.seq);
    // ---- one run with outputs
    { unsigned long long z[4] = {0, 0, 0, 0}; CK(cudaMemcpyToSymbol(g_molstat, z, sizeof z)); }
    g_cell_stats = true;
    Timer T; T.start();
    Outputs o;
    run_pipeline(A, T, in, &o);
    { unsigned long long ms[4]; CK(cudaMemcpyFromSymbol(ms, g_molstat, sizeof ms));
      po.molSpliced = ms[0]; po.molUnspliced = ms[1]; po.molAmbiguous = ms[2]; po.molLegacyDouble = ms[3]; }
    po.snpHits = (uint64_t) o.n_snp_tuples; po.sjHits = o.sj_raw_sum; po.mtBases = (uint64_t) o.n_mt_tuples;
    po.cellReads.swap(o.cell_reads); po.cellMolecules.swap(o.cell_mol);
    po.hitsExon = o.hits[0]; po.hitsIntron = o.hits[1]; po.umiRecorded = o.hits[2];
    po.alnGene.swap(o.aa_gene); po.alnStat.swap(o.aa_stat); po.alnExonGene.swap(o.aa_exg);
    for (auto& kv : T.acc) po.msGpu += kv.second;
    copyT3(o.exon, po.exon); copyT3(o.intron, po.intron); copyT3(o.amb, po.amb); copyT3(o.sj, po.sj);
    copyT3(o.snp_ad, po.snpAd); copyT3(o.snp_dp, po.snpDp); copyT3(o.mt, po.mt); copyT3(o.mt_rev, po.mtRev);
    po.sjtab = o.sjtab; po.indels = o.indels;
    po.snpMolecules = g_snp_stat[0]; po.snpAmbiguous = g_snp_stat[1]; po.snpOther = g_snp_stat[2];
    po.peakGB = A.peak / 1e9;
    cudaFree(A.base);
    cudaDeviceReset();
    po.msTotal = (now_s() - t0) * 1e3;
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
