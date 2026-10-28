// BAM -> GPU pileup input records (stand-in for M5's output until M5 exists).
//   extract BAM DUMPDIR OUTDIR [--threads N]
// DUMPDIR holds barcodes.txt and params.txt written by the patched engine (pileup_dump.patch). OUTDIR gets sigs.bin, alns.bin, ops.bin, seq.bin,
// reads.bin and meta.txt (record layouts in pileup_rec.h). Run it inside a job's /tmp: the files contain read sequences.
//
// CB -> column uses the engine's rule (BarcodeIndex: full string, then the string up to the last '-'; later barcodes overwrite).
// A multi-mapper (NH > 1) is collected by qname until NH alignments are seen (the engine's mm_buffer); a group that never completes is
// still resolved, like the engine's cross-worker merge does. Reads with a missing or unknown CB are dropped (the engine returns before
// counting anything for them). A read without UB/UR aborts (the engine would count it without UMI; none occurs in the campaign data).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <htslib/sam.h>
#include "pileup_rec.h"

struct H128 { uint64_t a, b; bool operator==(const H128& o) const { return a == o.a && b == o.b; } };
struct H128Hash { size_t operator()(const H128& h) const { return h.a ^ (h.b * 0x9e3779b97f4a7c15ULL); } };
static inline uint64_t mix(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33; return k; }
struct Hasher {
    uint64_t a = 0x243f6a8885a308d3ULL, b = 0x13198a2e03707344ULL;
    void add(uint64_t v) { a = mix(a ^ v) + 0x9e3779b97f4a7c15ULL; b = mix(b + v * 0xff51afd7ed558ccdULL) ^ (a >> 7); }
    H128 get() const { return H128{mix(a), mix(b ^ a)}; }
};

static const uint8_t COMP[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};   // nt16 complement (bit reversal)

struct AlnTmp { int32_t tid, pos; uint8_t flags, mapq; std::vector<uint32_t> ops; };
struct Group { std::vector<AlnTmp> alns; int seen = 0, nh = 0; int primary = -1; int32_t bc = -1; uint64_t umi3 = 0; std::vector<uint8_t> seq; };

static std::vector<SigRec> sigs; static std::vector<AlnRec> alns; static std::vector<uint32_t> opsv; static std::vector<uint8_t> seqv;
static std::vector<ReadRec> reads;
static std::unordered_map<H128, uint32_t, H128Hash> sigmap, seqmap_first_sig;
static std::unordered_map<uint64_t, uint32_t> seqonly;      // sequence hash -> first signature (E0 check)
static uint64_t n_e0_violations = 0, n_incomplete = 0, n_nopri = 0;

static void finalize(Group& g) {
    if (g.primary < 0) { n_nopri++; return; }
    const uint32_t rlen = (uint32_t)g.seq.size();
    Hasher hs; for (size_t i = 0; i < rlen; i += 8) { uint64_t v = 0; for (size_t j = 0; j < 8 && i + j < rlen; j++) v |= (uint64_t)g.seq[i + j] << (8 * j); hs.add(v); }
    hs.add(rlen); H128 hseq = hs.get();
    hs.add(g.nh); hs.add(g.alns.size());
    for (auto& a : g.alns) { hs.add(((uint64_t)(uint32_t)a.tid << 32) | (uint32_t)a.pos); hs.add(((uint64_t)a.flags << 8) | a.mapq); hs.add(a.ops.size()); for (uint32_t o : a.ops) hs.add(o); }
    H128 h = hs.get();
    auto it = sigmap.find(h); uint32_t sid;
    if (it == sigmap.end()) {
        sid = (uint32_t)sigs.size(); sigmap.emplace(h, sid);
        SigRec s; s.aln_off = (uint32_t)alns.size(); s.n_aln = (uint8_t)g.alns.size(); s.nh = (uint8_t)std::min(g.nh, 255); s.rlen = (uint16_t)rlen; s.seq_off = seqv.size();
        for (auto& a : g.alns) { AlnRec r; r.tid = a.tid; r.pos = a.pos; r.ops_off = (uint32_t)opsv.size(); r.n_ops = (uint16_t)a.ops.size(); r.flags = a.flags; r.mapq = a.mapq; opsv.insert(opsv.end(), a.ops.begin(), a.ops.end()); alns.push_back(r); }
        for (uint32_t i = 0; i < rlen; i += 2) seqv.push_back((uint8_t)(g.seq[i] | ((i + 1 < rlen ? g.seq[i + 1] : 0) << 4)));
        sigs.push_back(s);
        auto jt = seqonly.find(hseq.a ^ (hseq.b * 31)); if (jt == seqonly.end()) seqonly.emplace(hseq.a ^ (hseq.b * 31), sid); else n_e0_violations++;
    } else sid = it->second;
    reads.push_back(ReadRec{sid, g.bc, g.umi3});
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: extract BAM DUMPDIR OUTDIR [--threads N]\n"); return 1; }
    std::string bam = argv[1], dd = argv[2], od = argv[3]; int thr = 8;
    for (int i = 4; i < argc; i++) if (!strcmp(argv[i], "--threads") && i + 1 < argc) thr = atoi(argv[++i]);
    std::vector<std::string> bcs; { std::ifstream f(dd + "/barcodes.txt"); std::string l; while (std::getline(f, l)) if (!l.empty()) bcs.push_back(l); }
    int min_baseq = 10; { std::ifstream f(dd + "/params.txt"); std::string k; long v; while (f >> k >> v) if (k == "min_baseq") min_baseq = (int)v; }
    std::unordered_map<std::string, int32_t> bcmap; bcmap.reserve(bcs.size() * 3);
    for (uint32_t i = 0; i < bcs.size(); i++) { bcmap[bcs[i]] = i; size_t d = bcs[i].rfind('-'); if (d != std::string::npos && d > 0) bcmap[bcs[i].substr(0, d)] = i; }
    auto lookup = [&](const char* s) -> int32_t {
        std::string str(s); auto it = bcmap.find(str); if (it != bcmap.end()) return it->second;
        size_t d = str.rfind('-'); if (d != std::string::npos && d > 0) { it = bcmap.find(str.substr(0, d)); if (it != bcmap.end()) return it->second; }
        return -1; };
    htsFile* fp = hts_open(bam.c_str(), "r"); if (!fp) { fprintf(stderr, "cannot open %s\n", bam.c_str()); return 1; }
    hts_set_threads(fp, thr); sam_hdr_t* hdr = sam_hdr_read(fp); bam1_t* b = bam_init1();
    uint64_t n_rec = 0, n_unmap = 0, n_supp = 0, n_nocb = 0, n_badcb = 0, n_noumi = 0, n_secondary = 0, n_primary = 0, n_nh1 = 0, n_mm_reads = 0;
    uint64_t n_noqual = 0, n_lowqual_reads = 0, n_umi_len_var = 0, n_umi_bad = 0, n_qual_neq_n = 0, n_lowqual_bases = 0, n_n_bases = 0; int umi_len = -1;
    std::unordered_map<std::string, Group> pend; pend.reserve(1 << 22);
    std::vector<uint32_t> tmpops;
    while (sam_read1(fp, hdr, b) >= 0) {
        n_rec++;
        if (b->core.flag & BAM_FUNMAP) { n_unmap++; continue; }
        if (b->core.flag & BAM_FSUPPLEMENTARY) { n_supp++; continue; }
        bool sec = (b->core.flag & BAM_FSECONDARY) != 0;
        uint8_t* cbt = bam_aux_get(b, "CB"); if (!cbt) { n_nocb++; continue; }
        int32_t bc = lookup(bam_aux2Z(cbt)); if (bc < 0) { n_badcb++; continue; }
        int nh = 1; if (uint8_t* t = bam_aux_get(b, "NH")) { nh = (int)bam_aux2i(t); if (nh < 1) nh = 1; }
        AlnTmp a; a.tid = b->core.tid; a.pos = (int32_t)b->core.pos; a.flags = (uint8_t)(((b->core.flag & BAM_FREVERSE) ? 1 : 0) | (sec ? 0 : 2)); a.mapq = b->core.qual;
        const uint32_t* cg = bam_get_cigar(b); a.ops.assign(cg, cg + b->core.n_cigar);
        sec ? n_secondary++ : n_primary++;
        Group* gp; Group local;
        std::string qn;
        if (nh == 1 && !sec) { gp = &local; gp->nh = 1; gp->seen = 0; }
        else { qn = bam_get_qname(b); gp = &pend[qn]; if (gp->nh == 0) gp->nh = std::min(nh, 255); }
        Group& g = *gp; g.alns.push_back(std::move(a)); g.seen++;
        if (!sec) {
            g.primary = (int)g.alns.size() - 1; g.bc = bc;
            uint8_t* ut = bam_aux_get(b, "UB"); if (!ut) ut = bam_aux_get(b, "UR");
            if (!ut) { n_noumi++; } else {
                const char* us = bam_aux2Z(ut); int ul = (int)strlen(us); if (umi_len < 0) umi_len = ul; if (ul != umi_len) n_umi_len_var++;
                uint64_t u3 = 0; for (int i = 0; i < ul && i < 21; i++) { char c = us[i]; uint64_t v; if (c == 'A') v = 0; else if (c == 'C') v = 1; else if (c == 'G') v = 2; else if (c == 'T') v = 3; else if (c == 'N') v = 4; else { v = 4; n_umi_bad++; } u3 |= v << (3 * i); }
                g.umi3 = u3; }
            int L = b->core.l_qseq; const uint8_t* sq = bam_get_seq(b); g.seq.resize(L);
            bool rev = (b->core.flag & BAM_FREVERSE) != 0;
            for (int i = 0; i < L; i++) g.seq[i] = rev ? COMP[bam_seqi(sq, L - 1 - i)] : (uint8_t)bam_seqi(sq, i);
            const uint8_t* ql = bam_get_qual(b); if (L > 0 && ql[0] == 0xff) n_noqual++; else { bool low = false; for (int i = 0; i < L; i++) { bool lq = ql[i] < min_baseq, isn = bam_seqi(sq, i) == 15; if (lq) { low = true; n_lowqual_bases++; } if (isn) n_n_bases++; if (lq != isn) n_qual_neq_n++; } if (low) n_lowqual_reads++; }
        }
        if (nh == 1 && !sec) { n_nh1++; finalize(g); }
        else if (g.seen >= g.nh) { n_mm_reads++; finalize(g); pend.erase(qn); }
    }
    for (auto& kv : pend) { n_incomplete++; n_mm_reads++; finalize(kv.second); }
    fprintf(stderr, "extract: %llu records (unmapped %llu, supplementary %llu), primary %llu secondary %llu; dropped no CB %llu, CB not in set %llu; NH==1 reads %llu, multi-mapper reads %llu (incomplete groups %llu, no primary %llu)\n",
            (unsigned long long)n_rec, (unsigned long long)n_unmap, (unsigned long long)n_supp, (unsigned long long)n_primary, (unsigned long long)n_secondary, (unsigned long long)n_nocb, (unsigned long long)n_badcb,
            (unsigned long long)n_nh1, (unsigned long long)n_mm_reads, (unsigned long long)n_incomplete, (unsigned long long)n_nopri);
    fprintf(stderr, "extract: reads %zu, signatures %zu (%.4f), alignments %zu, E0 violations (same sequence, different transcript set) %llu; no UMI %llu, variable UMI length %llu, non-ACGTN UMI chars %llu; "
            "primary records with no quality %llu, reads with a base below min_baseq=%d: %llu (%llu bases; %llu N bases; bases where (quality < min_baseq) != (base is N): %llu)\n",
            reads.size(), sigs.size(), (double)sigs.size() / std::max<size_t>(1, reads.size()), alns.size(), (unsigned long long)n_e0_violations, (unsigned long long)n_noumi, (unsigned long long)n_umi_len_var,
            (unsigned long long)n_umi_bad, (unsigned long long)n_noqual, min_baseq, (unsigned long long)n_lowqual_reads, (unsigned long long)n_lowqual_bases, (unsigned long long)n_n_bases, (unsigned long long)n_qual_neq_n);
    const int snp_skip_n = n_lowqual_bases == 0 ? 0 : (n_qual_neq_n == 0 ? 1 : -1);   // SNP pileup quality rule: no low-quality base / low quality == N / mixed (cannot be reproduced from sequences)
    fprintf(stderr, "extract: SNP-pileup base-quality rule: %s\n", snp_skip_n == 0 ? "no base below min_baseq (nothing skipped)" : snp_skip_n == 1 ? "skip N bases (quality < min_baseq exactly on N)" : "MIXED (not reproducible)");
    if (n_noumi) { fprintf(stderr, "FATAL: reads without UMI present\n"); return 2; }
    auto wr = [&](const char* name, const void* p, size_t bytes) { std::string path = od + "/" + name; FILE* f = fopen(path.c_str(), "wb"); if (!f) { perror(path.c_str()); exit(1); } fwrite(p, 1, bytes, f); fclose(f); };
    wr("sigs.bin", sigs.data(), sigs.size() * sizeof(SigRec)); wr("alns.bin", alns.data(), alns.size() * sizeof(AlnRec)); wr("ops.bin", opsv.data(), opsv.size() * 4);
    wr("seq.bin", seqv.data(), seqv.size()); wr("reads.bin", reads.data(), reads.size() * sizeof(ReadRec));
    { std::string path = od + "/meta.txt"; FILE* f = fopen(path.c_str(), "w");
      fprintf(f, "n_reads %zu\nn_sigs %zu\nn_alns %zu\nn_ops %zu\nseq_bytes %zu\numi_len %d\nn_records %llu\nn_e0_violations %llu\nn_incomplete %llu\nn_primary_noqual %llu\nn_reads_lowqual %llu\nn_qual_neq_n %llu\nsnp_skip_n %d\nn_nocb %llu\nn_badcb %llu\nn_barcodes %zu\n",
              reads.size(), sigs.size(), alns.size(), opsv.size(), seqv.size(), umi_len, (unsigned long long)n_rec, (unsigned long long)n_e0_violations, (unsigned long long)n_incomplete,
              (unsigned long long)n_noqual, (unsigned long long)n_lowqual_reads, (unsigned long long)n_qual_neq_n, snp_skip_n, (unsigned long long)n_nocb, (unsigned long long)n_badcb, bcs.size());
      fclose(f); }
    return 0;
}
