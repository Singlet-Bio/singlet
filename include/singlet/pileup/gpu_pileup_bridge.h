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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
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

// The GPU prefill's hand-off (src/star/SeqCacheGpuPrefill.cpp, SINGLET_ENGINE_GPU_HANDOFF): per read (in input order, the read named
// r<index>) its record or 0xFFFFFFFF, per record the alignments STAR writes for that sequence (ReadAlign::seqCacheAlignRecords).
// FASTQ lines from a file through a 4 MB buffer (getc per character was the hand-off pass's bottleneck)
struct LineReader {
    FILE* f = nullptr; std::vector<char> buf; size_t pos = 0, len = 0; bool eof = false;
    explicit LineReader(const char* path) : f(path ? std::fopen(path, "rb") : nullptr), buf(4 << 20) {}
    ~LineReader() { if (f) std::fclose(f); }
    bool ok() const { return f != nullptr; }
    // the next line without its terminator (and without a trailing \r), NUL-terminated in out
    bool get(std::string& out) {
        out.clear();
        for (;;) {
            if (pos == len) { if (eof) return !out.empty(); len = std::fread(buf.data(), 1, buf.size(), f); pos = 0; if (len == 0) { eof = true; return !out.empty(); } }
            const char* s = buf.data() + pos; const char* e = (const char*) std::memchr(s, '\n', len - pos);
            if (e) { out.append(s, e - s); pos += (e - s) + 1; if (!out.empty() && out.back() == '\r') out.pop_back(); return true; }
            out.append(s, len - pos); pos = len;
        }
    }
};
// STAR's per-read hand-off entry (HandoffReadEntry in src/star/SeqCache.h): CR and UR, 3 bits per base (A0 C1 G2 T3 other 4, base 0
// lowest), and the cDNA read's base-quality counters; flags bit 0 set when STAR wrote it
struct ReadEntry { uint64_t cr3, ur3; uint16_t low_qual, qual_neq_n; uint32_t flags; };
static_assert(sizeof(ReadEntry) == 24, "ReadEntry layout is STAR's HandoffReadEntry");
// per-base quality of a read, for the SNP and chrM base filters: bit i = base i of the record sequence (original read orientation, 5' clip removed) has quality < min_baseq
struct QM { uint64_t m[2]; };
struct Handoff {
    const uint8_t* map = nullptr; size_t size = 0;
    uint64_t n_reads = 0, n_rec = 0; const uint32_t* read_rec = nullptr; const uint64_t* rec_off = nullptr; const uint8_t* data = nullptr;
    uint32_t cbS = 0, cbL = 0, umiS = 0, umiL = 0, cbumiL = 0, simple = 0;   // STAR's barcode layout (v2)
    const QM* qmask = nullptr; size_t q_size = 0; std::string q_path;   // "on": <path>.reads.q, 16 bytes per read, written by STAR's mapping threads
    const ReadEntry* reads = nullptr; size_t reads_size = 0; std::string reads_path;   // "on": <path>.reads, written by STAR's mapping threads
    // the per-read entries, once STAR has exited ("on" only); false if there are none
    bool load_reads() {
        if (reads) return true;
        if (reads_path.empty()) return false;
        int fd = open(reads_path.c_str(), O_RDONLY); if (fd < 0) return false;
        struct stat st; if (fstat(fd, &st) != 0 || (uint64_t) st.st_size != n_reads * sizeof(ReadEntry) || n_reads == 0) { close(fd); return false; }
        void* m = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
        if (m == MAP_FAILED) return false;
        madvise(m, (size_t) st.st_size, MADV_SEQUENTIAL);
        reads = (const ReadEntry*) m; reads_size = (size_t) st.st_size;
        return true;
    }
    bool load_q() {
        if (qmask) return true;
        if (q_path.empty()) return false;
        int fd = open(q_path.c_str(), O_RDONLY); if (fd < 0) return false;
        struct stat st; if (fstat(fd, &st) != 0 || (uint64_t) st.st_size != n_reads * sizeof(QM) || n_reads == 0) { close(fd); return false; }
        void* m = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
        if (m == MAP_FAILED) return false;
        madvise(m, (size_t) st.st_size, MADV_SEQUENTIAL);
        qmask = (const QM*) m; q_size = (size_t) st.st_size;
        return true;
    }
    bool load(const char* path) {
        int fd = open(path, O_RDONLY); if (fd < 0) return false;
        struct stat st; if (fstat(fd, &st) != 0 || st.st_size < 24) { close(fd); return false; }
        size = (size_t) st.st_size;
        void* m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
        if (m == MAP_FAILED) return false;
        map = (const uint8_t*) m;
        if (std::memcmp(map, "SGHOFF02", 8) != 0) { unmap(); return false; }
        std::memcpy(&n_reads, map + 8, 8); std::memcpy(&n_rec, map + 16, 8);
        uint32_t lay[6]; std::memcpy(lay, map + 24, 24);
        cbS = lay[0]; cbL = lay[1]; umiS = lay[2]; umiL = lay[3]; cbumiL = lay[4]; simple = lay[5];
        read_rec = (const uint32_t*) (map + 48);
        rec_off = (const uint64_t*) (map + 48 + 4 * n_reads);
        data = (const uint8_t*) (rec_off + n_rec + 1);
        if ((size_t) (data - map) + rec_off[n_rec] > size) { unmap(); return false; }
        reads_path = std::string(path) + ".reads"; q_path = reads_path + ".q";
        return true;
    }
    void unmap() {
        if (map) munmap((void*) map, size); map = nullptr;
        if (reads) { munmap((void*) reads, reads_size); unlink(reads_path.c_str()); } reads = nullptr;
        if (qmask) { munmap((void*) qmask, q_size); } qmask = nullptr; if (!q_path.empty()) unlink(q_path.c_str());
    }
    ~Handoff() { unmap(); }
};

class Collector {
public:
    explicit Collector(int min_baseq) : min_baseq_(min_baseq) {
        if (const char* q = std::getenv("SINGLET_GPU_PILEUP_QUAL")) force_q_ = std::strcmp(q, "force") == 0;
        pend_.reserve(1 << 20);
        if (const char* hp = std::getenv("SINGLET_ENGINE_GPU_HANDOFF")) handoff_path_ = hp;
    }

    // one record of the stream; bc = the engine's barcode column for it (-1: no CB or not in the barcode set)
    void add(const bam1_t* b, int32_t bc) {
        if (b->core.flag & (BAM_FUNMAP | BAM_FSUPPLEMENTARY)) return;
        if (bc < 0) return;
        const bool sec = (b->core.flag & BAM_FSECONDARY) != 0;
        if (e2e() != nullptr) {   // E8: a read with a device-built record never comes through the stream ("on": STAR writes none); counted, the pass fails
            const char* qn = bam_get_qname(b); const singlet_engine_gpu::E2EHeader& H = *e2e_->hdr();
            uint64_t idx = 0; bool num = qn[0] == 'r' && qn[1] != 0;
            for (const char* q = qn + 1; num && *q; ++q) { if (*q < '0' || *q > '9') num = false; else idx = idx * 10 + (uint64_t) (*q - '0'); }
            if (num && idx < H.nReads && e2e_->at<uint32_t>(H.oReadRecord)[idx] != 0xFFFFFFFFu) { e2e_stream_++; return; }
        }
        if (!ho_tried_) { ho_tried_ = true; if (!handoff_path_.empty() && ho_.load(handoff_path_.c_str())) ho_ok_ = true; }
        if (ho_ok_) {   // reads with a GPU record: the alignments come from the hand-off; the stream gives barcode, UMI, qualities
            const char* qn = bam_get_qname(b);
            uint64_t idx = 0; bool num = qn[0] == 'r' && qn[1] != 0;
            for (const char* q = qn + 1; num && *q; ++q) { if (*q < '0' || *q > '9') num = false; else idx = idx * 10 + (uint64_t) (*q - '0'); }
            if (num && idx < ho_.n_reads && ho_.read_rec[idx] != 0xFFFFFFFFu) {
                if (sec) return;
                int nh = 1; const char* ub = nullptr; const char* ur = nullptr;
                scan_aux(b, nh, ub, ur);
                const char* us = ub ? ub : ur;
                uint64_t u3 = 0;
                if (us) {
                    const int ul = (int) std::strlen(us);
                    if (umi_len_ < 0) umi_len_ = ul;
                    for (int i = 0; i < ul && i < 21; i++) { const char c = us[i]; const uint64_t v = c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4; u3 |= v << (3 * i); }
                } else no_umi_++;
                qual_check(b);
                const uint32_t rec = ho_.read_rec[idx];
                check_primary(b, rec);
                ho_reads_.push_back(ReadRec{rec, bc, u3}); ho_clip_.push_back(0); ho_q_.push_back(bam_mask(b));
                ho_idx_.push_back(idx);
                return;
            }
        }
        int nh = 1; const char* ub = nullptr; const char* ur = nullptr;
        scan_aux(b, nh, ub, ur);   // one pass over the tags (NH, UB, UR) instead of three bam_aux_get scans
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
            const char* us = ub ? ub : ur;
            if (us) {
                const int ul = (int) std::strlen(us);
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
            qual_check(b); g.qm = bam_mask(b);
        }
        if (nh == 1 && !sec) finalize(g);
        else if (g.seen >= g.nh) { finalize(g); pend_.erase(qn); }
    }

    // flushes incomplete multimapper groups (resolved anyway, as the engine does) and moves the records into `in`
    void finish(singlet_engine_gpu::PileupInput& in) {
        for (auto& kv : pend_) { incomplete_++; finalize(kv.second); }
        pend_.clear();
        singlet_tail::mark("collector: pending groups flushed");
        if (ho_ok_) prepend_handoff();
        if (e2e_ != nullptr) prepend_e2e(in); else {   // E8: the device-built records go into `in` directly
        e2e_unmap();   // E8 requested but not used (the hand-off file path ran): the memory object is released
        in.sigs.assign((const uint8_t*) sigs_.data(), (const uint8_t*) (sigs_.data() + sigs_.size()));
        in.alns.assign((const uint8_t*) alns_.data(), (const uint8_t*) (alns_.data() + alns_.size()));
        in.reads.assign((const uint8_t*) reads_.data(), (const uint8_t*) (reads_.data() + reads_.size()));
        in.ops.swap(ops_); in.seq.swap(seq_); }
        in.snpSkipN = (lowqual_bases_ == 0 ? 0 : (qual_neq_n_ == 0 ? 1 : -1));
        if (no_umi_ > 0) in.snpSkipN = -1;   // reads without a UMI: not reproducible from the packed UMI (never seen in the campaign data)
        if (umi_len_ > 0) in.umiLen = umi_len_;
        if ((in.snpSkipN < 0 || force_q_) && no_umi_ == 0) {   // per-base quality: the SNP and chrM filters read each base's own quality
            if (q_ok_ && reads_q_.size() == reads_.size() && in.nSnps <= 0x7FFFFFu) {
                in.useQual = 1; in.snpSkipN = 0;
                in.lowq.assign((const uint8_t*) reads_q_.data(), (const uint8_t*) (reads_q_.data() + reads_q_.size()));
                singlet_tail::mark("collector: low-quality masks copied into the input (%zu reads)", reads_q_.size());
                std::cerr << "[gpu_pileup] per-base quality: low-quality masks for " << reads_q_.size() << " reads" << (force_q_ ? " (forced)" : "") << "\n";
            } else std::cerr << "[gpu_pileup] per-base quality needed but unavailable (masks " << (q_ok_ ? "missing for some reads" : "not representable: read over 128 bases or no STAR masks") << ")\n";
        }
    }
    size_t n_reads() const { return reads_.size() + ho_reads_.size(); }
    size_t n_handoff_reads() const { return ho_reads_.size(); }
    uint64_t handoff_mismatches() const { return ho_mismatch_; }
    bool handoff_used() const { return ho_ok_ || e2e_ != nullptr; }
    // E8 (SINGLET_ENGINE_GPU_E2E=1): the shared region with the device-built records, once the prefill completed it; null otherwise
    const singlet_engine_gpu::E2ERegion* e2e() {
        if (!e2e_tried_) { e2e_tried_ = true; singlet_engine_gpu::E2ERegion& r = singlet_engine_gpu::e2eRegion(); if (r.mapUsed()) e2e_ = &r; }
        return e2e_;
    }
    uint64_t e2e_stream_reads() const { return e2e_stream_; }
    // E8: the device pass's kept reads (ReadRec bytes, read order), their clip flags and low-quality masks (16 bytes each; maskAll: every
    // read had one), as add_handoff_read_u3 would have appended them one by one
    void add_e2e_reads(const std::vector<uint8_t>& reads, const std::vector<uint32_t>& clip, const std::vector<uint8_t>& lowq, bool mask_all, int umi_len) {
        const size_t n = clip.size(), r0 = ho_reads_.size();
        if (n && umi_len_ < 0) umi_len_ = umi_len;
        ho_reads_.resize(r0 + n); if (n) std::memcpy((void*) (ho_reads_.data() + r0), reads.data(), n * sizeof(ReadRec));
        ho_clip_.insert(ho_clip_.end(), clip.begin(), clip.end());
        const size_t q0 = ho_q_.size(); ho_q_.resize(q0 + n); if (n) std::memcpy((void*) (ho_q_.data() + q0), lowq.data(), n * sizeof(QM));
        if (!mask_all) q_ok_ = false;
        for (uint32_t c : clip) if (c) n_clipped_++;
    }
    // the hand-off (loaded on first use); null if there is none
    const Handoff* handoff() {
        if (!ho_tried_) { ho_tried_ = true; if (!handoff_path_.empty() && ho_.load(handoff_path_.c_str())) ho_ok_ = true; }
        return ho_ok_ ? &ho_ : nullptr;
    }
    // "on", after STAR has exited: map STAR's per-read entries (handoff()->reads); false if there are none
    bool load_handoff_reads() { return handoff() != nullptr && ho_.load_reads(); }
    // STAR's per-read low-quality masks (the hand-off's .reads.q); null if there are none
    const QM* handoff_qmask() { return handoff() != nullptr && ho_.load_q() ? ho_.qmask : nullptr; }
    // a hand-off read that STAR wrote no BAM record for ("on"): its record, barcode column and UMI (A0 C1 G2 T3 N4, 3 bits each)
    // clip: the read's entry flags when it was clipped (STAR's HandoffReadEntry: clip lengths and the 3' tail), else 0
    void add_handoff_read(uint32_t rec, int32_t bc, const char* umi, int umi_len, uint32_t clip = 0) {
        uint64_t u3 = 0;
        if (umi_len_ < 0) umi_len_ = umi_len;
        for (int i = 0; i < umi_len && i < 21; i++) { const char c = umi[i]; const uint64_t v = c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4; u3 |= v << (3 * i); }
        ho_reads_.push_back(ReadRec{rec, bc, u3}); ho_clip_.push_back(clip); q_ok_ = false;   // this path has no qualities
        if (clip) n_clipped_++;
    }
    // the same with the UMI already packed (the parallel hand-off pass packs it in its workers)
    void add_handoff_read_u3(uint32_t rec, int32_t bc, uint64_t u3, int umi_len, uint32_t clip = 0, const QM* qm = nullptr) {
        if (umi_len_ < 0) umi_len_ = umi_len;
        ho_reads_.push_back(ReadRec{rec, bc, u3}); ho_clip_.push_back(clip);
        if (qm) ho_q_.push_back(*qm); else { ho_q_.push_back(QM{{0, 0}}); q_ok_ = false; }
        if (clip) n_clipped_++;
    }
    // per read of the final input (hand-off reads first): its clip flags, 0 when not clipped
    const std::vector<uint32_t>& clips() const { return ho_clip_; }
    uint32_t read_clip(size_t r) const { return r < ho_clip_.size() ? ho_clip_[r] : 0u; }
    uint64_t n_clipped() const { return n_clipped_; }
    // the low-quality mask of a BAM record, in the read's original orientation (the signature stores the read that way); qualities missing: nothing is low
    QM bam_mask(const bam1_t* b) {
        QM q{{0, 0}}; const int L = b->core.l_qseq; const uint8_t* ql = bam_get_qual(b);
        if (L <= 0 || ql[0] == 0xff) return q;
        if (L > 128) { q_ok_ = false; return q; }
        const bool rev = (b->core.flag & BAM_FREVERSE) != 0;
        for (int i = 0; i < L; i++) if (ql[rev ? L - 1 - i : i] < min_baseq_) q.m[i >> 6] |= 1ull << (i & 63);
        return q;
    }
    // the base-quality rule of qual_check on a FASTQ record (Phred+33): BAM qualities are the FASTQ's
    void add_quality(const char* seq, const char* qual, int L) {
        for (int i = 0; i < L; i++) {
            const bool lq = (qual[i] - 33) < min_baseq_, isn = seq[i] == 'N';
            if (lq) lowqual_bases_++;
            if (lq != isn) qual_neq_n_++;
        }
    }
    // the same counters, as STAR's mapping threads counted them for a hand-off read (ReadEntry)
    void add_quality_counts(uint32_t low_qual, uint32_t qual_neq_n) { lowqual_bases_ += low_qual; qual_neq_n_ += qual_neq_n; }
    void add_quality_counts64(uint64_t low_qual, uint64_t qual_neq_n) { lowqual_bases_ += low_qual; qual_neq_n_ += qual_neq_n; }
    size_t n_sigs() const { return sigs_.size(); }
    uint64_t incomplete() const { return incomplete_; }
    uint64_t no_umi() const { return no_umi_; }

private:
    // the SNP pileup's base-quality rule (see snpSkipN): every base of every primary
    void qual_check(const bam1_t* b) {
        const int L = b->core.l_qseq; const uint8_t* sq = bam_get_seq(b); const uint8_t* ql = bam_get_qual(b);
        if (L > 0 && ql[0] == 0xff) return;
        for (int i = 0; i < L; i++) {
            const bool lq = ql[i] < min_baseq_, isn = bam_seqi(sq, i) == 15;
            if (lq) lowqual_bases_++;
            if (lq != isn) qual_neq_n_++;
        }
    }
    // the hand-off's primary alignment of record rec vs the stream's primary record: position, strand, MAPQ, CIGAR, NH (the "on" mode's
    // alignment decisions against the BAM STAR writes for the same read: the identity check that goes with each alignment-gate run)
    void check_primary(const bam1_t* b, uint32_t rec) {
        const uint8_t* p = ho_.data + ho_.rec_off[rec];
        const int n = *p++;
        for (int k = 0; k < n; k++) {
            int32_t tid, pos; uint16_t flag, nh, nm, nc; uint8_t mapq;
            std::memcpy(&tid, p, 4); std::memcpy(&pos, p + 4, 4); std::memcpy(&flag, p + 8, 2); mapq = p[10];
            std::memcpy(&nh, p + 12, 2); std::memcpy(&nm, p + 14, 2); std::memcpy(&nc, p + 16, 2);
            const uint8_t* cig = p + 18; p = cig + 4 * nc;
            if (flag & BAM_FSECONDARY) continue;
            const bool same = tid == b->core.tid && pos == (int32_t) b->core.pos && mapq == b->core.qual && nc == b->core.n_cigar
                && (flag & BAM_FREVERSE) == (b->core.flag & BAM_FREVERSE) && std::memcmp(cig, bam_get_cigar(b), 4 * nc) == 0
                && [&] { const uint8_t* t = bam_aux_get(b, "NH"); return t == nullptr || bam_aux2i(t) == (int64_t) nh; }();
            if (!same) ho_mismatch_++;
            return;
        }
        ho_mismatch_++;   // no primary in the record although the stream has one
    }
    // hand-off records become signatures 0..n_rec-1; the stream's own signatures (reads without a record) follow
    void prepend_handoff() {
        std::vector<SigRec> sg; std::vector<AlnRec> al; std::vector<uint32_t> op; std::vector<uint8_t> sq;
        sg.reserve(ho_.n_rec);
        // upper bounds from the hand-off's size (a record is >= 1 + 22 bytes per alignment, 4 per CIGAR operation, 1 per 2 bases), plus
        // what the stream collected: the vectors then never reallocate (each doubling copied GBs; untouched capacity costs no memory)
        {
            const uint64_t bytes = ho_.n_rec ? ho_.rec_off[ho_.n_rec] : 0;
            al.reserve(bytes / 22 + alns_.size() + 1); op.reserve(bytes / 4 + ops_.size() + 1); sq.reserve(bytes + seq_.size() + 1);
        }
        for (uint64_t r = 0; r < ho_.n_rec; r++) {
            const uint8_t* p = ho_.data + ho_.rec_off[r];
            const int n = *p++;
            SigRec s; s.aln_off = (uint32_t) al.size(); s.n_aln = (uint8_t) n; s.nh = 1; s.seq_off = sq.size();
            for (int k = 0; k < n; k++) {
                int32_t tid, pos; uint16_t flag, nh, nm, nc; uint8_t mapq;
                std::memcpy(&tid, p, 4); std::memcpy(&pos, p + 4, 4); std::memcpy(&flag, p + 8, 2); mapq = p[10];
                std::memcpy(&nh, p + 12, 2); std::memcpy(&nm, p + 14, 2); std::memcpy(&nc, p + 16, 2);
                if (k == 0) s.nh = (uint8_t) std::min<uint32_t>(std::max<uint32_t>(nh, 1), 255);
                AlnRec a; a.tid = tid; a.pos = pos; a.ops_off = (uint32_t) op.size(); a.n_ops = nc; a.mapq = mapq;
                a.flags = (uint8_t) (((flag & BAM_FREVERSE) ? 1 : 0) | ((flag & BAM_FSECONDARY) ? 0 : 2));
                const uint8_t* cig = p + 18;
                for (int c = 0; c < nc; c++) { uint32_t v; std::memcpy(&v, cig + 4 * c, 4); op.push_back(v); }
                p = cig + 4 * nc;
                al.push_back(a);
            }
            uint16_t L; std::memcpy(&L, p, 2); p += 2;
            s.rlen = L;
            sq.insert(sq.end(), p, p + (L + 1) / 2);
            sg.push_back(s);
        }
        const uint32_t s0 = (uint32_t) sg.size(), a0 = (uint32_t) al.size(), o0 = (uint32_t) op.size(); const uint64_t q0 = sq.size();
        for (auto& s : sigs_) { s.aln_off += a0; s.seq_off += q0; sg.push_back(s); }
        for (auto& a : alns_) { a.ops_off += o0; al.push_back(a); }
        op.insert(op.end(), ops_.begin(), ops_.end());
        sq.insert(sq.end(), seq_.begin(), seq_.end());
        for (auto& r : reads_) r.sig += s0;
        ho_reads_.insert(ho_reads_.end(), reads_.begin(), reads_.end());
        reads_.swap(ho_reads_);
        ho_q_.insert(ho_q_.end(), reads_q_.begin(), reads_q_.end()); reads_q_.swap(ho_q_);
        sigs_.swap(sg); alns_.swap(al); ops_.swap(op); seq_.swap(sq);
        ho_.unmap();
        unlink(handoff_path_.c_str());
    }
    // E8: the device-built records (shared region) become signatures 0..nRec-1, as prepend_handoff makes them from the file (E6: the same
    // arrays); the stream's own follow, shifted. Written into `in` directly; the region is released afterwards.
    void prepend_e2e(singlet_engine_gpu::PileupInput& in) {
        const singlet_engine_gpu::E2EHeader& H = *e2e_->hdr();
        const uint64_t nR = H.nRec, nA = H.nAln, nO = H.nOps, nQ = H.nSeq;
        for (auto& s : sigs_) { s.aln_off += (uint32_t) nA; s.seq_off += nQ; }
        for (auto& a : alns_) a.ops_off += (uint32_t) nO;
        for (auto& r : reads_) r.sig += (uint32_t) nR;
        auto cat = [](auto& dst, const void* a, size_t na, const void* b, size_t nb) {
            dst.resize((na + nb) / sizeof(dst[0]));
            if (na) std::memcpy((void*) dst.data(), a, na);
            if (nb) std::memcpy((uint8_t*) dst.data() + na, b, nb);
        };
        singlet_tail::mark("collector: e2e: stream offsets shifted");
        cat(in.sigs, e2e_->base + H.oSigs, nR * sizeof(SigRec), sigs_.data(), sigs_.size() * sizeof(SigRec));
        cat(in.alns, e2e_->base + H.oAlns, nA * sizeof(AlnRec), alns_.data(), alns_.size() * sizeof(AlnRec));
        singlet_tail::mark("collector: e2e: signatures and alignments copied");
        cat(in.ops, e2e_->base + H.oOps, nO * 4, ops_.data(), ops_.size() * 4);
        cat(in.seq, e2e_->base + H.oSeq, nQ, seq_.data(), seq_.size());
        singlet_tail::mark("collector: e2e: CIGAR ops and sequences copied");
        ho_reads_.insert(ho_reads_.end(), reads_.begin(), reads_.end());
        reads_.swap(ho_reads_);
        in.reads.assign((const uint8_t*) reads_.data(), (const uint8_t*) (reads_.data() + reads_.size()));
        singlet_tail::mark("collector: e2e: reads copied into the input");
        ho_q_.insert(ho_q_.end(), reads_q_.begin(), reads_q_.end()); reads_q_.swap(ho_q_);
        std::vector<SigRec>().swap(sigs_); std::vector<AlnRec>().swap(alns_); std::vector<uint32_t>().swap(ops_); std::vector<uint8_t>().swap(seq_);
        singlet_tail::mark("collector: e2e: masks appended, stream vectors freed");
        e2e_unmap();
        singlet_tail::mark("collector: e2e: region unmapped and closed");
    }
    static void e2e_unmap() {
        singlet_engine_gpu::e2eRegion().release();   // unmapped and the memfd closed: its pages go
    }
    const singlet_engine_gpu::E2ERegion* e2e_ = nullptr; bool e2e_tried_ = false; uint64_t e2e_stream_ = 0;
    std::string handoff_path_; Handoff ho_; bool ho_tried_ = false, ho_ok_ = false; uint64_t ho_mismatch_ = 0;
    std::vector<ReadRec> ho_reads_;
    std::vector<QM> reads_q_, ho_q_; bool q_ok_ = true, force_q_ = false;   // per-read low-quality masks, parallel to reads_ and ho_reads_
    std::vector<uint32_t> ho_clip_; uint64_t n_clipped_ = 0;
    std::vector<uint64_t> ho_idx_;
public:
    // check mode: (read index, barcode column, packed UMI) of every stream-derived hand-off read
    void stream_truth(std::vector<uint64_t>& idx, std::vector<ReadRec>& rr) const { idx = ho_idx_; rr = ho_reads_; }
private:
    // NH (integer types), UB and UR (Z) from one pass over the aux tags; same values as bam_aux_get + bam_aux2i / bam_aux2Z
    static void scan_aux(const bam1_t* b, int& nh, const char*& ub, const char*& ur) {
        const uint8_t* s = bam_get_aux(b); const uint8_t* end = b->data + b->l_data;
        while (s + 3 <= end) {
            const char t0 = (char) s[0], t1 = (char) s[1]; const uint8_t ty = s[2]; const uint8_t* v = s + 3;
            if (ty == 'Z' || ty == 'H') {
                if (t0 == 'U' && t1 == 'B' && ty == 'Z') ub = (const char*) v;
                else if (t0 == 'U' && t1 == 'R' && ty == 'Z') ur = (const char*) v;
                while (v < end && *v) ++v; s = v + 1; continue;
            }
            int64_t iv = 0; int w = 0;
            switch (ty) {
                case 'A': case 'c': iv = (int8_t) v[0]; w = 1; break;
                case 'C': iv = v[0]; w = 1; break;
                case 's': { int16_t x; std::memcpy(&x, v, 2); iv = x; w = 2; break; }
                case 'S': { uint16_t x; std::memcpy(&x, v, 2); iv = x; w = 2; break; }
                case 'i': { int32_t x; std::memcpy(&x, v, 4); iv = x; w = 4; break; }
                case 'I': { uint32_t x; std::memcpy(&x, v, 4); iv = x; w = 4; break; }
                case 'f': w = 4; break;
                case 'd': w = 8; break;
                case 'B': { const uint8_t sub = v[0]; uint32_t n; std::memcpy(&n, v + 1, 4);
                            const int es = (sub == 'c' || sub == 'C') ? 1 : (sub == 's' || sub == 'S') ? 2 : 4; w = 5 + (int) n * es; break; }
                default: return;   // malformed: stop, as bam_aux_get would not find later tags either
            }
            if (t0 == 'N' && t1 == 'H' && ty != 'A' && ty != 'f' && ty != 'd' && ty != 'B') { nh = (int) iv; if (nh < 1) nh = 1; }
            s = v + w;
        }
    }
    struct Aln { int32_t tid, pos; uint8_t flags, mapq; std::vector<uint32_t> ops; };
    struct Group { std::vector<Aln> alns; int seen = 0, nh = 0, primary = -1; int32_t bc = -1; uint64_t umi3 = 0; std::vector<uint8_t> seq; QM qm{{0, 0}}; };
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
        reads_.push_back(ReadRec{sid, g.bc, g.umi3}); reads_q_.push_back(g.qm);
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
