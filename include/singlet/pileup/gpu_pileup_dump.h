// W11 GPU pileup harness (debug only; no effect unless the variables are set).
//
//   SINGLET_PILEUP_DUMP=<dir>   after the pileup pass, write what the callers receive and the annotation the GPU replays, in the
//                               format of PR #61 bench/gpu/pileup (pileup_rec.h, pileup.cu): truth.bin (exon, intron, ambiguous,
//                               junctions with names, SNP AD/DP, chrM allele counts total/reverse, chrM indels; to_csc sums and clamps
//                               included), anno.bin (interval trees as sorted arrays, gene maps and strands, SNP table), params.txt,
//                               barcodes.txt, chroms.txt.
//   SINGLET_STREAM_TEE=<file>   keep an uncompressed BAM copy of the record stream the serial pileup reads (STAR's unsorted output),
//                               so the GPU pileup can be fed the same records (bench/gpu/pileup/extract.cpp).
#pragma once
#include <htslib/sam.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace singlet { namespace gpudump {

inline const char* dir() { static const char* p = std::getenv("SINGLET_PILEUP_DUMP"); return (p && *p) ? p : nullptr; }

struct StreamTee {
    samFile* fp = nullptr;
    explicit StreamTee(const sam_hdr_t* hdr) {
        const char* p = std::getenv("SINGLET_STREAM_TEE");
        if (!p || !*p) return;
        fp = sam_open(p, "wbu");
        if (fp && sam_hdr_write(fp, hdr) < 0) { sam_close(fp); fp = nullptr; }
        if (!fp) std::fprintf(stderr, "[gpudump] cannot write the stream copy %s\n", p);
    }
    void write(const sam_hdr_t* hdr, const bam1_t* b) { if (fp && sam_write1(fp, hdr, b) < 0) { sam_close(fp); fp = nullptr; } }
    ~StreamTee() { if (fp) sam_close(fp); }
};

struct Params { int stranded, reverse_strand, min_mapq, min_baseq, count_introns, count_sj, count_mt, umi_dedup, directional, umi_len, chrm_tid;
                uint32_t n_snps; int has_gene_model; uint32_t n_exons, n_introns, n_genes; };

template <typename T> inline void w(FILE* f, const T& v) { std::fwrite(&v, sizeof(T), 1, f); }

template <typename Acc>
inline void write_acc(FILE* f, const Acc& acc) {   // CSC triplets (row, col, value) after to_csc (sum + clamp), columns then rows ascending
    auto csc = acc.to_csc();
    uint32_t nr = acc.n_features(), nc = static_cast<uint32_t>(csc.indptr.size() ? csc.indptr.size() - 1 : 0);
    uint64_t nnz = csc.indices.size();
    w<uint32_t>(f, nr); w<uint32_t>(f, nc); w<uint64_t>(f, nnz);
    for (uint32_t c = 0; c < nc; ++c)
        for (int32_t k = csc.indptr[c]; k < csc.indptr[c + 1]; ++k) {
            w<uint32_t>(f, static_cast<uint32_t>(csc.indices[k])); w<uint32_t>(f, c); w<uint32_t>(f, static_cast<uint32_t>(csc.data[k]));
        }
}

template <typename Trees, typename Snps, typename Acc16, typename Acc8, typename Indels>
inline void write_all(const Trees& et, const Trees& it, const std::vector<uint32_t>& e2g, const std::vector<uint32_t>& i2g,
                      const std::vector<char>& gstr, const Snps& chrom_snps, uint32_t n_tid,
                      const Acc16& exon, const Acc16& intron, const Acc16& amb, const Acc16& sj, const std::vector<std::string>& sj_names,
                      const Acc8& snp_ad, const Acc8& snp_dp, const Acc16& mt, const Acc16& mt_rev, const Indels& indels, uint64_t n_ambig,
                      const std::vector<std::string>& barcodes, const std::vector<std::string>& chroms, const Params& P) {
    const std::string d(dir());
    if (FILE* f = std::fopen((d + "/anno.bin").c_str(), "wb")) {
        std::fwrite("PUANN1\0\0", 1, 8, f);
        w<uint32_t>(f, n_tid);
        for (uint32_t t = 0; t < n_tid; ++t)
            for (int k = 0; k < 2; ++k) {
                const auto* tr = (k == 0 ? (t < et.size() ? &et[t] : nullptr) : (t < it.size() ? &it[t] : nullptr));
                uint64_t n = tr ? tr->sorted_intervals().size() : 0;
                w<uint64_t>(f, n);
                if (n) std::fwrite(tr->sorted_intervals().data(), sizeof(tr->sorted_intervals()[0]), n, f);
            }
        auto wv = [&](const auto& v) { uint64_t n = v.size(); w<uint64_t>(f, n); if (n) std::fwrite(v.data(), sizeof(v[0]), n, f); };
        wv(e2g); wv(i2g); wv(gstr);
        w<uint32_t>(f, static_cast<uint32_t>(chrom_snps.size()));
        for (const auto& v : chrom_snps) {
            uint64_t n = v.size(); w<uint64_t>(f, n);
            for (const auto& s : v) { w<int32_t>(f, s.pos); w<uint32_t>(f, s.index); w<char>(f, s.ref_allele); w<char>(f, s.alt_allele); w<uint16_t>(f, 0); }
        }
        uint64_t n_iv = 0;
        for (uint32_t t = 0; t < n_tid; ++t) { if (t < et.size()) n_iv += et[t].sorted_intervals().size(); if (t < it.size()) n_iv += it[t].sorted_intervals().size(); }
        uint64_t n_snp = 0; for (const auto& v : chrom_snps) n_snp += v.size();
        std::fprintf(stderr, "[gpudump] anno.bin: %u contigs, %llu intervals, e2g %zu, i2g %zu, gene strands %zu, %zu SNP contigs, %llu SNPs, %ld bytes\n",
                     n_tid, (unsigned long long)n_iv, e2g.size(), i2g.size(), gstr.size(), chrom_snps.size(), (unsigned long long)n_snp, std::ftell(f));
        std::fclose(f);
    }
    if (FILE* f = std::fopen((d + "/truth.bin").c_str(), "wb")) {
        std::fwrite("PUTRU1\0\0", 1, 8, f);
        write_acc(f, exon); write_acc(f, intron); write_acc(f, amb);
        write_acc(f, sj);
        w<uint32_t>(f, static_cast<uint32_t>(sj_names.size()));
        for (const auto& nm : sj_names) { w<uint32_t>(f, static_cast<uint32_t>(nm.size())); std::fwrite(nm.data(), 1, nm.size(), f); }
        write_acc(f, snp_ad); write_acc(f, snp_dp);
        write_acc(f, mt); write_acc(f, mt_rev);
        w<uint64_t>(f, static_cast<uint64_t>(indels.size()));
        for (const auto& e : indels) { w<uint32_t>(f, e.bc_idx); w<uint32_t>(f, e.pos); w<uint32_t>(f, static_cast<uint32_t>(e.len)); w<uint32_t>(f, e.is_ins ? 1u : 0u); }
        w<uint64_t>(f, n_ambig);
        std::fclose(f);
    }
    if (FILE* f = std::fopen((d + "/params.txt").c_str(), "w")) {
        std::fprintf(f, "stranded %d\nreverse_strand %d\nmin_mapq %d\nmin_baseq %d\ncount_introns %d\ncount_sj %d\ncount_mt %d\numi_dedup %d\n"
                        "umi_dedup_directional %d\numi_len %d\nchrm_tid %d\nn_snps %u\nn_barcodes %zu\nn_tid %u\nhas_gene_model %d\nn_exons %u\n"
                        "n_introns %u\nn_genes %u\n",
                     P.stranded, P.reverse_strand, P.min_mapq, P.min_baseq, P.count_introns, P.count_sj, P.count_mt, P.umi_dedup, P.directional,
                     P.umi_len, P.chrm_tid, P.n_snps, barcodes.size(), n_tid, P.has_gene_model, P.n_exons, P.n_introns, P.n_genes);
        std::fclose(f);
    }
    if (FILE* f = std::fopen((d + "/barcodes.txt").c_str(), "w")) { for (const auto& b : barcodes) std::fprintf(f, "%s\n", b.c_str()); std::fclose(f); }
    if (FILE* f = std::fopen((d + "/chroms.txt").c_str(), "w")) { for (const auto& c : chroms) std::fprintf(f, "%s\n", c.c_str()); std::fclose(f); }
    std::fprintf(stderr, "[gpudump] wrote anno.bin truth.bin params.txt barcodes.txt chroms.txt -> %s\n", d.c_str());
}

}} // namespace singlet::gpudump
