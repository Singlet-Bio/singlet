// singlet-pileup: rna_variant_caller.h
// De novo RNA variant discovery from aligned BAM (no prior VCF required).
// Designed for Smart-seq2 (SS3) and bulk RNA (B3) where per-gene coverage
// is high enough for single-sample variant calling.
//
// Uses htslib bam_plp_* per-position pileup API.
// Splice-junction (CIGAR N) reads are skipped automatically via is_refskip.
// RNA editing (A-to-I etc.) is kept — downstream decides.
//
// Output formats: simplified TSV (chrom, pos, ref, alt, coverage, alt_freq)
// and minimal VCF (FILTER=PASS for every called variant).

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// htslib — only included when BAM processing is needed
#include <htslib/hts.h>
#include <htslib/sam.h>

namespace singlet {

// ── Base helpers ──────────────────────────────────────────────────────────────

// htslib bam_seqi 4-bit encoding → [0,3] index (A=0,C=1,G=2,T=3) or -1
static constexpr int8_t SEQI_TO_IDX[16] = {
    -1,  // 0 = unknown
    0,   // 1 = A
    1,   // 2 = C
    -1,  // 3
    2,   // 4 = G
    -1, -1, -1,
    3,  // 8 = T
    -1, -1, -1, -1, -1, -1,
    -1  // 15 = N
};

static constexpr char IDX_TO_BASE[4] = {'A', 'C', 'G', 'T'};

// ── Per-position allele counts ────────────────────────────────────────────────

struct AlleleCount {
    uint32_t counts[4] = {0, 0, 0, 0};  // A, C, G, T

    void add(int base_idx) {
        if (base_idx >= 0 && base_idx < 4) counts[base_idx]++;
    }

    uint32_t total() const {
        return counts[0] + counts[1] + counts[2] + counts[3];
    }

    // Returns indices/counts for the two most-common alleles.
    // first_idx == -1 when total==0.
    void top_two(int& first_idx, uint32_t& first_cnt,
                 int& second_idx, uint32_t& second_cnt) const {
        first_idx = second_idx = -1;
        first_cnt = second_cnt = 0;
        for (int i = 0; i < 4; ++i) {
            if (counts[i] > first_cnt) {
                second_idx = first_idx;
                second_cnt = first_cnt;
                first_idx = i;
                first_cnt = counts[i];
            } else if (counts[i] > second_cnt) {
                second_idx = i;
                second_cnt = counts[i];
            }
        }
    }
};

// ── Variant record ────────────────────────────────────────────────────────────

struct RNAVariant {
    int32_t chrom_idx;  // index into the header contig list
    int32_t position;   // 0-based genomic position
    char ref_allele;    // majority allele (or FASTA-derived if ref supplied)
    char alt_allele;    // minority allele
    uint32_t ref_count;
    uint32_t alt_count;
    float vaf;        // alt / (ref + alt)
    uint8_t quality;  // placeholder — set to 60 (max MAPQ seen)
};

// ── Variant caller ────────────────────────────────────────────────────────────

class RNAVariantCaller {
   public:
    // ── Configuration ──────────────────────────────────────────────────────

    void set_min_coverage(int min_cov) { min_cov_ = min_cov; }
    void set_min_alt_count(int min_alt) { min_alt_ = min_alt; }
    void set_min_vaf(float min_vaf) { min_vaf_ = min_vaf; }
    void set_min_base_quality(int min_bq) { min_bq_ = min_bq; }

    // ── Core: call a single position directly (testable without BAM I/O) ──

    // Returns true if a variant is called; fills `out` on success.
    // ref_base_idx: ACGT index of reference allele, or -1 to infer from majority.
    bool call_position(int32_t chrom_idx, int32_t pos,
                       const AlleleCount& ac,
                       RNAVariant& out,
                       int ref_base_idx = -1) const {
        uint32_t cov = ac.total();
        if ((int)cov < min_cov_) return false;

        int fi, si;
        uint32_t fc, sc;
        ac.top_two(fi, fc, si, sc);
        if (fi < 0) return false;

        // ref = supplied index or majority allele
        int ref_idx = (ref_base_idx >= 0 && ref_base_idx < 4) ? ref_base_idx : fi;
        uint32_t ref_c = ac.counts[ref_idx];

        // alt = best allele that is not ref
        int alt_idx = -1;
        uint32_t alt_c = 0;
        for (int i = 0; i < 4; ++i) {
            if (i == ref_idx) continue;
            if (ac.counts[i] > alt_c) {
                alt_c = ac.counts[i];
                alt_idx = i;
            }
        }
        if (alt_idx < 0 || (int)alt_c < min_alt_) return false;

        float vaf = static_cast<float>(alt_c) / static_cast<float>(ref_c + alt_c);
        if (vaf < min_vaf_) return false;

        out.chrom_idx = chrom_idx;
        out.position = pos;
        out.ref_allele = IDX_TO_BASE[ref_idx];
        out.alt_allele = IDX_TO_BASE[alt_idx];
        out.ref_count = ref_c;
        out.alt_count = alt_c;
        out.vaf = vaf;
        out.quality = 60;
        return true;
    }

    // ── BAM processing ──────────────────────────────────────────────────────

    // Read sorted/unsorted BAM and call variants de novo.
    // ref_fasta: path to genome FASTA for ref-allele validation (optional).
    // When empty, majority allele is treated as reference.
    void process_bam(const std::string& bam_path,
                     const std::string& ref_fasta = "") {
        (void)ref_fasta;  // future: use fai-indexed FASTA for ref validation

        htsFile* fp = hts_open(bam_path.c_str(), "r");
        if (!fp) throw std::runtime_error("rna_variant_caller: cannot open " + bam_path);

        sam_hdr_t* hdr = sam_hdr_read(fp);
        if (!hdr) {
            hts_close(fp);
            throw std::runtime_error("rna_variant_caller: cannot read header");
        }

        // Build contig name map
        chrom_names_.clear();
        for (int i = 0; i < sam_hdr_nref(hdr); ++i)
            chrom_names_.push_back(sam_hdr_tid2name(hdr, i));

        // Pileup callback data
        struct PlpData {
            htsFile* fp;
            sam_hdr_t* hdr;
        };
        PlpData pdata{fp, hdr};

        auto read_fn = [](void* data, bam1_t* b) -> int {
            PlpData* d = static_cast<PlpData*>(data);
            return sam_read1(d->fp, d->hdr, b);
        };

        bam_plp_t plp = bam_plp_init(read_fn, &pdata);
        bam_plp_set_maxcnt(plp, 1000000);  // cap to avoid OOM on deep coverage

        int tid = -1, pos = -1, n_plp = 0;
        const bam_pileup1_t* pile;

        while ((pile = bam_plp_next(plp, &tid, &pos, &n_plp)) != nullptr) {
            AlleleCount ac;
            for (int i = 0; i < n_plp; ++i) {
                const bam_pileup1_t* p = &pile[i];
                if (p->is_del || p->is_refskip) continue;  // skip deletions + CIGAR N (splice junctions)
                int bq = bam_get_qual(p->b)[p->qpos];
                if (bq < min_bq_) continue;
                uint8_t seqi = bam_seqi(bam_get_seq(p->b), p->qpos);
                ac.add(SEQI_TO_IDX[seqi]);
            }
            ++positions_examined_;

            RNAVariant v;
            if (call_position(tid, pos, ac, v))
                variants_.push_back(v);
        }

        bam_plp_destroy(plp);
        sam_hdr_destroy(hdr);
        hts_close(fp);
    }

    // ── Output ─────────────────────────────────────────────────────────────

    // TSV: chrom  pos(1-based)  ref  alt  coverage  alt_freq
    void write_tsv(const std::string& path,
                   const std::vector<std::string>& chrom_names) const {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("rna_variant_caller: cannot write " + path);
        f << "chrom\tpos\tref\talt\tcoverage\talt_freq\n";
        for (const auto& v : variants_) {
            const std::string& chrom = (v.chrom_idx >= 0 &&
                                        (size_t)v.chrom_idx < chrom_names.size())
                                           ? chrom_names[v.chrom_idx]
                                           : "?";
            f << chrom << '\t'
              << (v.position + 1) << '\t'
              << v.ref_allele << '\t'
              << v.alt_allele << '\t'
              << (v.ref_count + v.alt_count) << '\t'
              << v.vaf << '\n';
        }
    }

    // VCF: minimal VCF 4.2 with DP and AF in INFO
    void write_vcf(const std::string& path,
                   const std::vector<std::string>& chrom_names) const {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("rna_variant_caller: cannot write " + path);
        f << "##fileformat=VCFv4.2\n";
        f << "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total depth\">\n";
        f << "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Alt allele frequency\">\n";
        for (const auto& cn : chrom_names)
            f << "##contig=<ID=" << cn << ">\n";
        f << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n";
        for (const auto& v : variants_) {
            const std::string& chrom = (v.chrom_idx >= 0 &&
                                        (size_t)v.chrom_idx < chrom_names.size())
                                           ? chrom_names[v.chrom_idx]
                                           : "?";
            f << chrom << '\t'
              << (v.position + 1) << '\t'
              << ".\t"
              << v.ref_allele << '\t'
              << v.alt_allele << '\t'
              << static_cast<int>(v.quality) << '\t'
              << "PASS\t"
              << "DP=" << (v.ref_count + v.alt_count)
              << ";AF=" << v.vaf << '\n';
        }
    }

    // ── Accessors ──────────────────────────────────────────────────────────

    const std::vector<RNAVariant>& variants() const { return variants_; }
    size_t positions_examined() const { return positions_examined_; }
    size_t variants_called() const { return variants_.size(); }

    // Contig names populated after process_bam()
    const std::vector<std::string>& chrom_names() const { return chrom_names_; }

    void reset() {
        variants_.clear();
        chrom_names_.clear();
        positions_examined_ = 0;
    }

   private:
    int min_cov_ = 10;
    int min_alt_ = 3;
    float min_vaf_ = 0.1f;
    int min_bq_ = 20;

    std::vector<RNAVariant> variants_;
    std::vector<std::string> chrom_names_;
    size_t positions_examined_ = 0;
};

}  // namespace singlet
