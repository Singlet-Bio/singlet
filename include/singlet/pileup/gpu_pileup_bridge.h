// W11 GPU pileup inside singlet (stream path, --engine gpu builds): collects compact alignment records from the record stream the
// serial pileup reads, runs the GPU pileup (include/singlet/engine_gpu.h pileupRun) on them, and compares every covered output with the
// CPU engine's accumulators in the same process. SINGLET_GPU_PILEUP=compare turns it on; unset, nothing is collected or run.
//
// The collector is PR #61's bench/gpu/pileup/extract.cpp fed record by record: a read's alignments are grouped by qname until NH of them
// are seen (STAR emits them together), and reads with the same sequence and alignment set share one signature. Output identity under
// docs/research-beta/order-free-rules.md: exploration.md 12.19 (job 790376).
#pragma once
#ifdef SINGLET_HAVE_ENGINE_GPU
#include "singlet/engine_gpu.h"
#include "gpu_pileup/pileup/pileup_rec.h"   // src/engine_gpu is on the include path of GPU builds
#include <htslib/sam.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace singlet { namespace gpupileup {

// SINGLET_GPU_PILEUP: "compare" = CPU pileup as usual, GPU alongside, every covered output compared in-process;
// "on" = the covered steps are skipped on the CPU and their accumulators and counters come from the GPU.
inline int mode() {
    static const int m = [] { const char* p = std::getenv("SINGLET_GPU_PILEUP");
                              return !p ? 0 : std::strcmp(p, "compare") == 0 ? 1 : std::strcmp(p, "on") == 0 ? 2 : 0; }();
    return m;
}
inline bool enabled() { return mode() != 0; }

class Collector {
public:
    explicit Collector(int min_baseq) : min_baseq_(min_baseq) { pend_.reserve(1 << 20); }

    // one record of the stream; bc = the engine's barcode column for it (-1: no CB or not in the barcode set)
    void add(const bam1_t* b, int32_t bc) {
        if (b->core.flag & (BAM_FUNMAP | BAM_FSUPPLEMENTARY)) return;
        if (bc < 0) return;
        const bool sec = (b->core.flag & BAM_FSECONDARY) != 0;
        int nh = 1;
        if (uint8_t* t = bam_aux_get(b, "NH")) { nh = (int) bam_aux2i(t); if (nh < 1) nh = 1; }
        Aln a;
        a.tid = b->core.tid; a.pos = (int32_t) b->core.pos; a.mapq = b->core.qual;
        a.flags = (uint8_t) (((b->core.flag & BAM_FREVERSE) ? 1 : 0) | (sec ? 0 : 2));
        const uint32_t* cg = bam_get_cigar(b);
        a.ops.assign(cg, cg + b->core.n_cigar);
        Group local; Group* gp; uint64_t qn = 0;
        if (nh == 1 && !sec) { gp = &local; gp->nh = 1; }
        else { qn = qname_hash(bam_get_qname(b)); gp = &pend_[qn]; if (gp->nh == 0) gp->nh = std::min(nh, 255); }
        Group& g = *gp;
        g.alns.push_back(std::move(a)); g.seen++;
        if (!sec) {
            g.primary = (int) g.alns.size() - 1; g.bc = bc;
            uint8_t* ut = bam_aux_get(b, "UB"); if (!ut) ut = bam_aux_get(b, "UR");
            if (ut) {
                const char* us = bam_aux2Z(ut); const int ul = (int) std::strlen(us);
                if (umi_len_ < 0) umi_len_ = ul;
                uint64_t u3 = 0;
                for (int i = 0; i < ul && i < 21; i++) {
                    const char c = us[i];
                    const uint64_t v = c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4;
                    u3 |= v << (3 * i);
                }
                g.umi3 = u3;
            } else no_umi_++;
            const int L = b->core.l_qseq; const uint8_t* sq = bam_get_seq(b); const bool rev = (b->core.flag & BAM_FREVERSE) != 0;
            static const uint8_t COMP[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
            g.seq.resize(L);
            for (int i = 0; i < L; i++) g.seq[i] = rev ? COMP[bam_seqi(sq, L - 1 - i)] : (uint8_t) bam_seqi(sq, i);
            const uint8_t* ql = bam_get_qual(b);
            if (!(L > 0 && ql[0] == 0xff))
                for (int i = 0; i < L; i++) {
                    const bool lq = ql[i] < min_baseq_, isn = bam_seqi(sq, i) == 15;
                    if (lq) lowqual_bases_++;
                    if (lq != isn) qual_neq_n_++;
                }
        }
        if (nh == 1 && !sec) finalize(g);
        else if (g.seen >= g.nh) { finalize(g); pend_.erase(qn); }
    }

    // flushes incomplete multimapper groups (resolved anyway, as the engine does) and moves the records into `in`
    void finish(singlet_engine_gpu::PileupInput& in) {
        for (auto& kv : pend_) { incomplete_++; finalize(kv.second); }
        pend_.clear();
        in.sigs.assign((const uint8_t*) sigs_.data(), (const uint8_t*) (sigs_.data() + sigs_.size()));
        in.alns.assign((const uint8_t*) alns_.data(), (const uint8_t*) (alns_.data() + alns_.size()));
        in.reads.assign((const uint8_t*) reads_.data(), (const uint8_t*) (reads_.data() + reads_.size()));
        in.ops.swap(ops_); in.seq.swap(seq_);
        in.snpSkipN = (lowqual_bases_ == 0 ? 0 : (qual_neq_n_ == 0 ? 1 : -1));
        if (no_umi_ > 0) in.snpSkipN = -1;   // reads without a UMI: not reproducible from the packed UMI (never seen in the campaign data)
        if (umi_len_ > 0) in.umiLen = umi_len_;
    }
    size_t n_reads() const { return reads_.size(); }
    size_t n_sigs() const { return sigs_.size(); }
    uint64_t incomplete() const { return incomplete_; }
    uint64_t no_umi() const { return no_umi_; }

private:
    struct Aln { int32_t tid, pos; uint8_t flags, mapq; std::vector<uint32_t> ops; };
    struct Group { std::vector<Aln> alns; int seen = 0, nh = 0, primary = -1; int32_t bc = -1; uint64_t umi3 = 0; std::vector<uint8_t> seq; };
    // FNV-1a over the qname: groups a multimapper's alignments (64-bit, as the engine's mm_buffer)
    static uint64_t qname_hash(const char* q) { uint64_t h = 14695981039346656037ULL; for (; *q; ++q) { h ^= (uint8_t) *q; h *= 1099511628211ULL; } return h; }
    struct H128 { uint64_t a, b; bool operator==(const H128& o) const { return a == o.a && b == o.b; } };
    struct H128Hash { size_t operator()(const H128& h) const { return h.a ^ (h.b * 0x9e3779b97f4a7c15ULL); } };
    static uint64_t mix(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33; return k; }
    struct Hasher {
        uint64_t a = 0x243f6a8885a308d3ULL, b = 0x13198a2e03707344ULL;
        void add(uint64_t v) { a = mix(a ^ v) + 0x9e3779b97f4a7c15ULL; b = mix(b + v * 0xff51afd7ed558ccdULL) ^ (a >> 7); }
        H128 get() const { return H128{mix(a), mix(b ^ a)}; }
    };
    void finalize(Group& g) {
        if (g.primary < 0) return;
        const uint32_t rlen = (uint32_t) g.seq.size();
        // every read is its own signature: deduplication by (sequence, alignments) cost the serial thread more than it saved the GPU
        uint32_t sid;
        {
            sid = (uint32_t) sigs_.size();
            SigRec s; s.aln_off = (uint32_t) alns_.size(); s.n_aln = (uint8_t) g.alns.size(); s.nh = (uint8_t) std::min(g.nh, 255);
            s.rlen = (uint16_t) rlen; s.seq_off = seq_.size();
            for (auto& a : g.alns) {
                AlnRec r; r.tid = a.tid; r.pos = a.pos; r.ops_off = (uint32_t) ops_.size(); r.n_ops = (uint16_t) a.ops.size(); r.flags = a.flags; r.mapq = a.mapq;
                ops_.insert(ops_.end(), a.ops.begin(), a.ops.end()); alns_.push_back(r);
            }
            for (uint32_t i = 0; i < rlen; i += 2) seq_.push_back((uint8_t) (g.seq[i] | ((i + 1 < rlen ? g.seq[i + 1] : 0) << 4)));
            sigs_.push_back(s);
        }
        reads_.push_back(ReadRec{sid, g.bc, g.umi3});
    }

    int min_baseq_; int umi_len_ = -1;
    uint64_t lowqual_bases_ = 0, qual_neq_n_ = 0, incomplete_ = 0, no_umi_ = 0;
    std::unordered_map<uint64_t, Group> pend_;   // keyed like the engine's mm_buffer (64-bit qname hash)
    std::unordered_map<H128, uint32_t, H128Hash> sigmap_;
    std::vector<SigRec> sigs_; std::vector<AlnRec> alns_; std::vector<ReadRec> reads_; std::vector<uint32_t> ops_; std::vector<uint8_t> seq_;
};

// ---- compare: CPU accumulator (after to_csc: sums and clamps) vs GPU triplets, entry by entry
template <typename Acc>
inline std::vector<singlet_engine_gpu::PileupTriplet> cpu_triplets(const Acc& acc) {
    std::vector<singlet_engine_gpu::PileupTriplet> v;
    auto csc = acc.to_csc();
    const uint32_t nc = (uint32_t) (csc.indptr.size() ? csc.indptr.size() - 1 : 0);
    for (uint32_t c = 0; c < nc; ++c)
        for (int32_t k = csc.indptr[c]; k < csc.indptr[c + 1]; ++k)
            v.push_back({c, (uint32_t) csc.indices[k], (uint32_t) csc.data[k]});
    return v;   // columns (cells) then rows ascending
}
inline std::string cmp_line(const char* name, const std::vector<singlet_engine_gpu::PileupTriplet>& g,
                            const std::vector<singlet_engine_gpu::PileupTriplet>& c, bool& all) {
    size_t i = 0, j = 0; long long same = 0, og = 0, oc = 0, vd = 0;
    while (i < g.size() || j < c.size()) {
        if (j == c.size() || (i < g.size() && std::tie(g[i].cell, g[i].row) < std::tie(c[j].cell, c[j].row))) { og++; i++; }
        else if (i == g.size() || std::tie(c[j].cell, c[j].row) < std::tie(g[i].cell, g[i].row)) { oc++; j++; }
        else { if (g[i].val == c[j].val) same++; else vd++; i++; j++; }
    }
    const bool ok = og == 0 && oc == 0 && vd == 0;
    all = all && ok;
    char b[256];
    std::snprintf(b, sizeof b, "%s %s (gpu %zu, cpu %zu entries; only gpu %lld, only cpu %lld, value differs %lld)", name, ok ? "identical" : "DIFFERENT",
                  g.size(), c.size(), og, oc, vd);
    return b;
}

}} // namespace singlet::gpupileup
#endif // SINGLET_HAVE_ENGINE_GPU
