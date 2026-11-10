// singlet GPU engine: the pileup stage (API: include/singlet/engine_gpu.h, pileupRun).
// The kernels are PR #61's bench/gpu/pileup (copied unchanged into gpu_pileup/, with M6's dedup / UMI collapse / CSR code); this file
// only replaces that program's file loading with the engine's in-memory inputs. Identity with the CPU engine: exploration.md 12.13
// (sorted path), 12.19 (order-free stream path, R1 consensus: every output identical on pbmc1k_v3, job 790376).
#define PILEUP_NO_MAIN
#include "gpu_pileup/pileup/pileup.cu"
#include "singlet/engine_gpu.h"
#include "singlet/qmask_words.h"   // P3-ENG-14
#include <cstring>
#include <glob.h>
#include <signal.h>
#include <sys/file.h>
#include "singlet/tail_trace.h"   // E16: the level-5 switches (digest)
#include "singlet/gpu_ledger.h"   // last: it wraps cudaMemcpy / cudaMemcpyAsync (SINGLET_ENGINE_GPU_TRACE=1)

static_assert(sizeof(IV) == sizeof(singlet_engine_gpu::PileupIv), "IV layout");
static_assert(sizeof(SnpRow) == sizeof(singlet_engine_gpu::PileupSnpRow), "SnpRow layout");
static_assert(sizeof(T3) == sizeof(singlet_engine_gpu::PileupTriplet), "T3 layout");

namespace singlet_engine_gpu {

static void copyT3(const std::vector<T3>& a, std::vector<PileupTriplet>& b) {
    b.resize(a.size());
    if (!a.empty()) memcpy(b.data(), a.data(), a.size() * sizeof(T3));
}


// A deep sample's pileup needs more than its slot's share of the device (E17 budget) and than the free memory while the shared genome index
// (SINGLET_GPU_SHARED_INDEX holder) still sits on it. Second attempt: ask the holders of this device to leave (they exit when no sample maps the
// index), take the device's other slots so that no other sample's prefill races for the memory, and size the arena from what is free.
struct PileupWiden {
    std::vector<int> fds;          // the other slots' locks, held until release()
    int slots = 1, held = 1;       // slots of the device; slots this pileup counts as its own (its first plus the ones in fds)
    unsigned long long evicted = 0, freedB = 0; int holders = 0; double seconds = 0;
    void release() { for (int fd : fds) { flock(fd, LOCK_UN); close(fd); } fds.clear(); }
    ~PileupWiden() { release(); }
};
static size_t pileupWiden(int device, size_t totB, size_t prevArena, PileupWiden& W) {
    const double t0 = now_s();
    size_t f0 = 0, tt = 0; cudaMemGetInfo(&f0, &tt);
    // (1) the holders of this device leave
    glob_t g{};
    const std::string pat = "/tmp/singlet_gpu_holder." + deviceId(device) + ".*.info";
    if (glob(pat.c_str(), 0, nullptr, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; i++) {
            FILE* f = fopen(g.gl_pathv[i], "r"); if (!f) continue;
            long pid = 0; unsigned long long b = 0;
            if (fscanf(f, "%ld %llu", &pid, &b) == 2 && pid > 0 && kill((pid_t) pid, 0) == 0 && kill((pid_t) pid, SIGUSR1) == 0) { W.evicted += b; W.holders++; }
            fclose(f);
        }
    }
    globfree(&g);
    // (2) the other slots (flock on a new descriptor conflicts with this process's own slot, so that one is never taken: S - 1 are wanted)
    W.slots = gpuSlots();
    const char* we = getenv("SINGLET_GPU_PILEUP_SLOT_WAIT_S");
    const double waitS = we ? atof(we) : 20.0;
    std::vector<char> mine(W.slots, 0);
    for (;;) {
        for (int k = 0; k < W.slots && (int) W.fds.size() < W.slots - 1; k++) {
            if (mine[k]) continue;
            const int fd = open(slotLockPath(device, k).c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
            if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0) { W.fds.push_back(fd); mine[k] = 1; }
            else if (fd >= 0) close(fd);
        }
        if ((int) W.fds.size() >= W.slots - 1 || now_s() - t0 >= waitS) break;
        usleep(50000);
    }
    W.held = 1 + (int) W.fds.size();
    // (3) the holders' memory comes back when their processes end
    size_t f1 = f0;
    for (int i = 0; i < 100; i++) {
        cudaMemGetInfo(&f1, &tt);
        if (W.holders == 0 || f1 >= f0 + (size_t) (0.9 * (double) W.evicted)) break;
        usleep(100000);
    }
    W.freedB = f1 > f0 ? f1 - f0 : 0;
    size_t arena = f1 > (size_t(3) << 30) ? f1 - (size_t(3) << 30) : f1 / 2;
    if (const uint64_t bud = singlet_engine_gpu::memBudget(tt)) {   // the budget of the slots this pileup now holds
        const uint64_t mine_b = bud * (uint64_t) W.held;
        arena = std::min<size_t>(arena, mine_b > (uint64_t(3) << 30) ? mine_b - (uint64_t(3) << 30) : mine_b / 2);
    }
    if (const char* ag = getenv("SINGLET_GPU_PILEUP_ARENA_GB")) arena = std::min<size_t>(arena, (size_t) (atof(ag) * (1ull << 30)));
    W.seconds = now_s() - t0;
    return arena > prevArena + (size_t(256) << 20) ? arena : 0;   // 0: nothing gained
}

bool pileupRun(const PileupInput& pi, int device, PileupOutput& po) {
    ::singlet::gpu_ledger::Phase ledger_phase("pileup");
    po = PileupOutput();
    const double t0 = now_s();
    // E16: wall time of each host-side step (the GPU time is po.msGpu); appended to po.log
    std::string e16; double e16t = t0;
    auto e16step = [&](const char* nm) { const double t = now_s(); char b[64]; snprintf(b, sizeof b, "%s %.3f, ", nm, t - e16t); e16 += b; e16t = t; };
    cudaInitDevice(device, cudaDeviceScheduleBlockingSync, singlet_engine_gpu::gpuInitFlags());
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
    an.ex.resize(pi.exonTrees.size()); an.in.resize(pi.intronTrees.size());   // an.snp stays empty: the SNP rows are read from pi.snps (below)
    for (size_t t = 0; t < pi.exonTrees.size(); t++) { an.ex[t].resize(pi.exonTrees[t].size()); if (!an.ex[t].empty()) memcpy(an.ex[t].data(), pi.exonTrees[t].data(), an.ex[t].size() * sizeof(IV)); }
    for (size_t t = 0; t < pi.intronTrees.size(); t++) { an.in[t].resize(pi.intronTrees[t].size()); if (!an.in[t].empty()) memcpy(an.in[t].data(), pi.intronTrees[t].data(), an.in[t].size() * sizeof(IV)); }
    an.e2g = pi.exonToGene; an.i2g = pi.intronToGene; an.gstr = pi.geneStrand;
    { uint32_t mx = 0; for (auto& v : pi.snps) for (auto& sr : v) mx = std::max(mx, sr.idx + 1); par.n_snps = std::max(par.n_snps, mx); }
    // E16: an array's leading part may be read in place (PileupInput::Pre), the vector holding the rest
    const size_t nS = (pi.preSigs.n + pi.sigs.size()) / sizeof(SigRec), nA = (pi.preAlns.n + pi.alns.size()) / sizeof(AlnRec),
                 nR = (pi.preReads.n + pi.reads.size()) / sizeof(ReadRec);
    e16step("annotation copied");
    // ---- device memory: one arena, sized from the free memory (the research program reserved 80 GB; pbmc1k_v3 peaks at ~25 GB
    // without the annotation bug)
    size_t freeB = 0, totB = 0; cudaMemGetInfo(&freeB, &totB);
    // a sample whose working set exceeds the device (GSM7656149: 223 M barcoded reads needed ~100 GB of 94) runs again in unified
    // memory twice the device's size: the part beyond the device pages to the host, the results are the same
    size_t arenaB = 0;   // E17: the device arena's size (logged)
    PileupWiden W; std::string notes;
    for (int attempt = 0; attempt < 3; attempt++) {
    Arena A;
    try {
    if (attempt == 0) {
        size_t arena = freeB > (size_t(3) << 30) ? freeB - (size_t(3) << 30) : freeB / 2;
        // E17 (docs/gpu-engine/fable/E17-budget.md): within the sample's device memory budget (singlet_engine_gpu::memBudget; less 3 GB for
        // this process's context and STAR's, which may still hold the E11 keep), not the free memory another sample's prefill races for
        if (const uint64_t bud = singlet_engine_gpu::memBudget(totB)) arena = std::min<size_t>(arena, bud > (uint64_t(3) << 30) ? bud - (uint64_t(3) << 30) : bud / 2);
        // SINGLET_GPU_PILEUP_ARENA_GB: cap the arena (samples sharing the GPU; the partitions K adapt to it)
        if (const char* ag = getenv("SINGLET_GPU_PILEUP_ARENA_GB")) arena = std::min<size_t>(arena, (size_t) (atof(ag) * (1ull << 30)));
        A.init(arena); arenaB = arena;
    }
    else if (attempt == 1) {
        const size_t arena = pileupWiden(device, totB, arenaB, W);
        char nb[256]; snprintf(nb, sizeof nb, "wider arena: %d holder(s) asked to leave (%.1f GB freed), %d of %d slots held, %.2f s: ", W.holders, W.freedB / 1e9, W.held, W.slots, W.seconds);
        notes += nb;
        if (arena == 0) { notes += "no gain; "; W.release(); continue; }
        size_t a = arena;
        while (!A.try_init(a)) { a -= a / 16; if (a < arenaB) break; }   // the free memory moved: a little less
        if (A.base == nullptr) { notes += "allocation failed; "; W.release(); continue; }
        arenaB = A.cap; snprintf(nb, sizeof nb, "arena %.1f GB; ", A.cap / 1e9); notes += nb;
    }
    else A.init_managed(2 * totB);
    e16step("arena");
    Inputs in; in.nS = (int)nS; in.nA = (int)nA; in.nR = (int)nR; in.ncells = (int)par.n_barcodes; in.ulen = pi.umiLen; in.par = par; in.wantMerges = pi.wantGeneMerges; in.wantUmiKeys = pi.wantUmiKeys;
    in.dp = DPar{par.stranded, par.reverse_strand, par.count_introns, par.count_sj, par.count_mt, par.min_mapq, par.chrm_tid, pi.snpSkipN, pi.intronRules, pi.useQual, (int) singlet_qmask::words()};
    in.EX = upload_tree(A, an.ex); in.IN = upload_tree(A, an.in); in.e2g = to_dev(A, an.e2g); in.i2g = to_dev(A, an.i2g); in.gstr = to_dev(A, an.gstr);
    in.gmt = to_dev(A, pi.geneIsMt); in.ngmt = (uint32_t)pi.geneIsMt.size();
    if (!pi.tidSpecies.empty()) { in.tsp = to_dev(A, pi.tidSpecies); in.ntsp = (uint32_t)pi.tidSpecies.size(); }
    if ((pi.intronRules & (16384 | 32768)) && !pi.txGeneOff.empty()) {   // kTxCompat / kStarGene: transcript models
        in.TX.gtoff = to_dev(A, pi.txGeneOff); in.TX.xoff = to_dev(A, pi.txExonOff); in.TX.es = to_dev(A, pi.txExonStart); in.TX.ee = to_dev(A, pi.txExonEnd);
        in.TX.ng = (uint32_t)pi.txGeneOff.size() - 1;
    }
    if ((pi.intronRules & 8) && !pi.geneStart.empty()) {   // kFlank: the gene bodies and their merged exons
        in.GM.gs = to_dev(A, pi.geneStart); in.GM.ge = to_dev(A, pi.geneEnd); in.GM.es = to_dev(A, pi.exonStart); in.GM.ee = to_dev(A, pi.exonEnd);
        in.GM.eoff = to_dev(A, pi.geneExonOff); in.GM.ex = to_dev(A, pi.geneExons); in.GM.ng = (uint32_t)pi.geneStart.size();
    }
    in.ne2g = (uint32_t)an.e2g.size(); in.ni2g = (uint32_t)an.i2g.size(); in.ngstr = (uint32_t)an.gstr.size();
    {
        // the rows per contig as four flat arrays (E24: sized first and filled in place; with the editing panel they hold ~23 M rows)
        const size_t nt = pi.snps.size(); std::vector<uint32_t> off(nt), n(nt); size_t tot = 0;
        for (size_t t = 0; t < nt; t++) { off[t] = (uint32_t)tot; n[t] = (uint32_t)pi.snps[t].size(); tot += n[t]; }
        std::vector<int32_t> P(tot); std::vector<uint32_t> I(tot); std::vector<char> alt(tot), ref(tot);
        for (size_t t = 0; t < nt; t++) {
            size_t j = off[t];
            for (const auto& s : pi.snps[t]) { P[j] = s.pos; I[j] = s.idx; alt[j] = s.alt; ref[j] = s.ref; j++; }
        }
        in.SN.pos = to_dev(A, P); in.SN.idx = to_dev(A, I); in.SN.alt = to_dev(A, alt); in.SN.ref = to_dev(A, ref); in.SN.off = to_dev(A, off);
        in.SN.n = to_dev(A, n); in.SN.ntid = (int)nt;
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
    in.Q = pi.useQual ? (const uint64_t*) devBytes(pi.lowq) : nullptr;   // singlet_qmask::words() words per read (PileupInput::lowq)
    } else {
    in.S = (SigRec*) devCat(pi.preSigs, pi.sigs.data(), pi.sigs.size(), sizeof(SigRec));
    in.Al = (AlnRec*) devCat(pi.preAlns, pi.alns.data(), pi.alns.size(), sizeof(AlnRec));
    in.R = (ReadRec*) devCat(pi.preReads, pi.reads.data(), pi.reads.size(), sizeof(ReadRec));
    in.ops = (uint32_t*) devCat(pi.preOps, pi.ops.data(), pi.ops.size() * 4, 4);
    in.seq = devCat(pi.preSeq, pi.seq.data(), pi.seq.size(), 1);
    in.Q = pi.useQual ? (const uint64_t*) devCat(pi.preLowq, pi.lowq.data(), pi.lowq.size(), 8 * singlet_qmask::words()) : nullptr;
    }
    e16step("records H2D");
    // ---- one run with outputs
    { unsigned long long z[5] = {0, 0, 0, 0, 0}; CK(cudaMemcpyToSymbol(g_molstat, z, sizeof z)); }
    g_cell_stats = true;
    g_want_reads = pi.wantReads != 0;
    g_edit_start = pi.editStart;   // D-101: red_snp counts the editing rows apart (g_edit_stat)
    g_want_mixed = true;   // the engine's mixed_exon / mixed_intron (export.h's pre-USA exon / intron split)
    { const char* pk = getenv("SINGLET_GPU_PILEUP_PARTS"); g_force_parts = pk ? std::max(0, std::min(atoi(pk), kMaxParts)) : 0; }
    g_e16_aasplit = singlet_tail::cut("aasplit");   // E16 (level 5): the per-alignment results split on the device before their download
    g_e16_fetchpar = singlet_tail::cut("fetchpar");
    g_e24_post = singlet_tail::cut("lockpost");   // E24: the fetch's host half and the merge after the unlock (post_run, below)
    Timer T; T.start();
    Outputs o;
    run_pipeline(A, T, in, &o);
    e16step("run_pipeline");
    unsigned long long ms[5]; CK(cudaMemcpyFromSymbol(ms, g_molstat, sizeof ms));
    po.molDropped = ms[4]; po.molSpliced = ms[0]; po.molUnspliced = ms[1]; po.molAmbiguous = ms[2]; po.molLegacyDouble = ms[3];
    po.peakGB = A.peak / 1e9;
    // the last device work is done (run_pipeline downloaded every output): free the arena and let the next sample have the GPU while the host
    // post-processing (digest, result copies) runs. E16 "resetbg" (level 5) leaves the reset and the unlock to the caller's thread instead.
    cudaFree(A.base);
    W.release();
    e16step("arena freed");
    if (singlet_tail::cut("resetbg")) po.resetDeferred = true;   // the caller's thread resets it (pileupRelease) and then unlocks the device
    else { cudaDeviceReset(); if (pi.release) pi.release(); }
    e16step(po.resetDeferred ? "device reset deferred" : "device reset, lock released");
    // E24 "lockpost": the device's work is done. With the deferred reset in E20's order (unlock, then reset) the device is unlocked here,
    // before the host half of the fetch and the merge (post_run), instead of when pileupRun returns
    if (g_e24_post) {
        const char* ul = getenv("SINGLET_GPU_PILEUP_UNLOCK");
        const bool early = po.resetDeferred && pi.release && !(ul != nullptr && (strcmp(ul, "reset") == 0 || strcmp(ul, "late") == 0));
        if (early) { pi.release(); po.unlockedEarly = true; }
        post_run(&o);
        e16step(early ? "host half after the unlock (E24)" : "host half (E24)");
    }
    // E16 "digest" (level 5): skipped; it hashes every output tuple (4.3 GB of chrM tuples on pbmc1k_v3: 2.4 s) for a log line only
    if (singlet_tail::cut("digest")) po.log += "digest skipped (E16); ";
    else {   // digest of everything the run returns (the partition-invariance gate compares it across K)
        Sha256 h; h.update(sha_out(o).data(), 64);
        { auto ind = o.indels; std::sort(ind.begin(), ind.end()); h.update(ind.data(), ind.size() * 16); }   // as a set: their order follows the stream's read order
        h.update(o.cell_reads.data(), o.cell_reads.size() * 4); h.update(o.cell_mol.data(), o.cell_mol.size() * 4);
        h.update(o.mixed_exon.data(), o.mixed_exon.size() * sizeof(T3)); h.update(o.mixed_intron.data(), o.mixed_intron.size() * sizeof(T3));
        h.update(o.cell_mixed.data(), o.cell_mixed.size() * 4); h.update(o.hits, sizeof o.hits); h.update(&o.sj_raw_sum, 8); h.update(g_snp_stat, sizeof g_snp_stat);
        h.update(ms, sizeof ms);
        long long n[4] = {o.n_gene_tuples, o.n_sj_tuples, o.n_snp_tuples, o.n_mt_tuples}; h.update(n, sizeof n);
        po.log += "digest " + h.hex() + "; ";
    }
    e16step("digest");
    po.snpHits = (uint64_t) o.n_snp_tuples; po.sjHits = o.sj_raw_sum; po.mtBases = (uint64_t) o.n_mt_tuples;
    po.cellReads.swap(o.cell_reads); po.cellMolecules.swap(o.cell_mol); po.cellMixed.swap(o.cell_mixed);
    po.hitsExon = o.hits[0]; po.hitsIntron = o.hits[1]; po.umiRecorded = o.hits[2]; po.mtRescued = o.hits[3];
    po.alnGene.swap(o.aa_gene); po.alnStat.swap(o.aa_stat); po.alnExonGene.swap(o.aa_exg);
    for (auto& kv : T.acc) if (kv.first != "fetch") po.msGpu += kv.second;   // "fetch": the host copying a partition's results out
    { char b[160]; snprintf(b, sizeof b, "partition select %.1f ms (in the GPU time), host fetch between partitions %.1f ms; ", T.acc.count("part.select") ? T.acc["part.select"] : 0.0, T.acc.count("fetch") ? T.acc["fetch"] : 0.0); po.log += b; }
    // multimap record: every barcoded read of a signature k_resolve marked (the CPU engine's AmbigRead per read), in read order. Host work
    // after the device's release, on 8 threads (chunks of the read array, joined in chunk order), beside the tuple copies below.
    std::thread mmth([&]() {
        const auto R = pileupCat<ReadRec>(pi.preReads, pi.reads.data(), pi.reads.size());
        const uint64_t n = R.size(), nsig = o.amb_sig.size();
        constexpr int K = 8;
        std::vector<std::vector<PileupAmbig>> part(K);
        std::vector<std::thread> th;
        for (int k = 0; k < K; k++)
            th.emplace_back([&, k]() {
                for (uint64_t r = n * k / K, e = n * (k + 1) / K; r < e; r++) {
                    const ReadRec& x = R[r];
                    if (x.bc < 0 || x.sig >= nsig) continue;
                    const uint2 g = o.amb_sig[x.sig];
                    if (g.x != MAXU) part[k].push_back(PileupAmbig{x.bc, g.x, g.y, x.umi3});
                }
            });
        for (auto& t : th) t.join();
        size_t tot = 0;
        for (const auto& v : part) tot += v.size();
        po.ambig.reserve(tot);
        for (const auto& v : part) po.ambig.insert(po.ambig.end(), v.begin(), v.end());
    });
    // E16 "copypar" (level 5): the eight tuple arrays copied on threads, beside the rest of this and the device's release (joined at the end)
    std::vector<std::thread> e16cp;
    if (singlet_tail::cut("copypar")) {
        std::vector<std::pair<const std::vector<T3>*, std::vector<PileupTriplet>*>> cp = {{&o.exon, &po.exon}, {&o.intron, &po.intron}, {&o.amb, &po.amb},
            {&o.sj, &po.sj}, {&o.snp_ad, &po.snpAd}, {&o.snp_dp, &po.snpDp}, {&o.mt, &po.mt}, {&o.mt_rev, &po.mtRev},
            {&o.mixed_exon, &po.mixedExon}, {&o.mixed_intron, &po.mixedIntron}};   // campaign's pre-USA exon / intron split
        if (pi.wantReads) {   // campaign's reads per molecule
            cp.push_back({&o.exon_reads, &po.exonReads}); cp.push_back({&o.intron_reads, &po.intronReads});
            po.histCell.swap(o.hist_cell); po.hist.swap(o.hist);
        }
        for (const auto& c : cp) e16cp.emplace_back([c]() { copyT3(*c.first, *c.second); });
    } else {
    copyT3(o.exon, po.exon); copyT3(o.intron, po.intron); copyT3(o.amb, po.amb); copyT3(o.sj, po.sj);
    copyT3(o.mixed_exon, po.mixedExon); copyT3(o.mixed_intron, po.mixedIntron);
    if (pi.wantReads) { copyT3(o.exon_reads, po.exonReads); copyT3(o.intron_reads, po.intronReads); po.histCell.swap(o.hist_cell); po.hist.swap(o.hist); }
    copyT3(o.snp_ad, po.snpAd); copyT3(o.snp_dp, po.snpDp); copyT3(o.mt, po.mt); copyT3(o.mt_rev, po.mtRev);
    }
    po.sjtab = o.sjtab; po.indels = o.indels; po.umiKeys.swap(o.umi_keys);
    {   // R9 per gene: the merged gene-store UMIs, (cell, gene, umi) order
        std::sort(o.merges.begin(), o.merges.end(), [](const auto& x, const auto& y) { return std::tie(x.cell, x.gene, x.umi) < std::tie(y.cell, y.gene, y.umi); });
        po.geneMerges.resize(o.merges.size());
        for (size_t i = 0; i < o.merges.size(); i++) po.geneMerges[i] = PileupGeneMerge{o.merges[i].cell, o.merges[i].gene, o.merges[i].umi, o.merges[i].root};
    }
    po.snpMolecules = g_snp_stat[0]; po.snpAmbiguous = g_snp_stat[1]; po.snpOther = g_snp_stat[2];
    po.editHits = g_edit_stat[0]; po.editMolecules = g_edit_stat[1]; po.editAmbiguous = g_edit_stat[2]; po.editOther = g_edit_stat[3];
    if (attempt < 2) { char ab[160]; snprintf(ab, sizeof ab, "arena %.1f GB (E17 budget %.1f GB, free at start %.1f GB), peak %.1f GB; ", arenaB / 1073741824.0,
                                              singlet_engine_gpu::memBudget(totB) / 1073741824.0, freeB / 1073741824.0, A.peak / 1073741824.0); po.log += ab; }
    po.log += "cell partitions: " + std::to_string(g_parts_used) + (g_force_parts > 0 ? " (forced)" : "") + "; ";
    for (auto& l : g_parts_log) po.log += l + "; ";
    e16step("outputs copied");
    if (!e16cp.empty()) { for (auto& t : e16cp) t.join(); e16step("tuple copies joined"); }
    mmth.join();
    std::vector<uint2>().swap(o.amb_sig);
    e16step("multimap reads joined");
    po.msTotal = (now_s() - t0) * 1e3;
    po.log += "e16 steps (s): " + e16 + "; ";
    if (attempt == 1) po.log += notes + "device arena, no unified memory; ";
    if (attempt > 1) po.log += notes + "unified memory (the device arena was too small); ";
    return true;
    } catch (const ArenaExhausted& e) {
        cudaDeviceSynchronize();
        cudaFree(A.base);
        { char nb[160]; snprintf(nb, sizeof nb, "attempt %d: %zu MB needed, %zu MB arena; ", attempt, e.need >> 20, e.cap >> 20); notes += nb; }
        if (attempt > 1) {
            po.log += "GPU memory: " + std::to_string(e.need >> 20) + " MB needed, " + std::to_string(e.cap >> 20) + " MB unified arena";
            cudaDeviceReset();
            return false;
        }
    }
    }
    return false;
}

void pileupRelease(int device) {
    cudaSetDevice(device);
    cudaDeviceReset();
}

} // namespace singlet_engine_gpu
