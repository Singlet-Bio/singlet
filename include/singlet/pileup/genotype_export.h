#pragma once
// Genotype outputs kept for every sample (D-52, 2026-11-06). Private: everything here is written under <out>/private/genotype/ and never
// reaches a public object (docs/CELL-ARCHITECTURE.md; scripts/atlas_rerun keeps private/ in the per-sample private sidecar).
//
//   private/genotype/donors.vcf.gz      one multi-sample VCF per sample, a column per demux donor, GT:GQ:AD:DP:PL. Depths are summed over
//                                       the called cells assigned to the donor (cell_status "cell", prob_max >= 0.9); ambient back-assigned
//                                       barcodes and, at --n-donors 1, uncalled barcodes are left out. The call is R12 (call_donor_genotype:
//                                       fixed-error binomial, flat prior, GT only when the posterior reaches 0.99); GQ and PL come from the
//                                       same three likelihoods.
//   private/genotype/cellsnp/           the cellsnp-lite layout (cellSNP.base.vcf.gz, cellSNP.samples.tsv, cellSNP.tag.{AD,DP,OTH}.mtx):
//                                       UMI-level alt and ref+alt depths per called cell at the callable panel SNPs, i.e. DP summed over the
//                                       called cells >= 20 (cellsnp-lite mode 1a defaults minCOUNT 20, minMAF 0). `vireo -c <dir>` reads it.
//                                       The engine keeps no other-allele count per cell, so OTH is an empty matrix of the same shape.
//
// Everything depends only on the SNP matrices, the called cells and the donor assignments, so the CPU and GPU paths write the same bytes
// (gzip via zlib's gzopen: no timestamp in the header). scripts/genotype/backfill_genotypes.py writes the same content from an existing
// run's private/snp_{ad,dp}.1pz + cell_calls.tsv + donor_assignments.tsv; keep the two in step.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <zlib.h>

#include "donor_demux.h"

namespace singlet {
namespace genotype {

constexpr uint32_t kCellsnpMinCount = 20;   // cellsnp-lite --minCOUNT default
constexpr float kDonorMinProb = 0.9f;       // aggregate_donor_depths' default
constexpr double kGenoErr = 0.01;           // call_donor_genotype's defaults
constexpr double kGenoPost = 0.99;

struct GzOut {
    gzFile f = nullptr;
    std::string buf;
    explicit GzOut(const std::string& path) { f = gzopen(path.c_str(), "wb6"); buf.reserve(1 << 20); }
    bool ok() const { return f != nullptr; }
    void put(const std::string& s) { buf += s; if (buf.size() >= (1u << 20)) flush(); }
    void flush() { if (f && !buf.empty()) gzwrite(f, buf.data(), static_cast<unsigned>(buf.size())); buf.clear(); }
    ~GzOut() { flush(); if (f) gzclose(f); }
};

/// Genotype fields of one donor at one site: GT, GQ (phred of 1 - P(best), capped at 99) and PL (phred-scaled normalized likelihoods),
/// from the three binomial log-likelihoods of call_donor_genotype. GT is "./." unless the best posterior reaches kGenoPost, which is
/// call_donor_genotype's rule (same expressions, same result).
inline std::string donor_fields(uint32_t ad, uint32_t dp)
{
    if (dp == 0) return "./.:.:0,0:0:.";   // every FORMAT field present: some readers index the fields without the VCF's dropping rule
    if (ad > dp) ad = dp;
    const double err = kGenoErr;
    const double a = static_cast<double>(ad), r = static_cast<double>(dp - ad);
    const double l[3] = {r * std::log1p(-err) + a * std::log(err),
                         static_cast<double>(dp) * std::log(0.5),
                         a * std::log1p(-err) + r * std::log(err)};
    const double m = std::max(l[0], std::max(l[1], l[2]));
    const double e[3] = {std::exp(l[0] - m), std::exp(l[1] - m), std::exp(l[2] - m)};
    const double z = e[0] + e[1] + e[2];
    int best = 0;
    for (int g = 1; g < 3; ++g) if (e[g] > e[best]) best = g;
    static const char* kGt[3] = {"0/0", "0/1", "1/1"};
    const char* gt = (e[best] / z >= kGenoPost) ? kGt[best] : "./.";
    double rest = 0.0;
    for (int g = 0; g < 3; ++g) if (g != best) rest += e[g];
    const double q = rest / z;
    long long gq = q <= 0.0 ? 99 : static_cast<long long>(std::floor(-10.0 * std::log10(q) + 0.5));
    if (gq > 99) gq = 99;
    long long pl[3];
    for (int g = 0; g < 3; ++g) pl[g] = static_cast<long long>(std::floor(-10.0 * (l[g] - m) / std::log(10.0) + 0.5));
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s:%lld:%u,%u:%u:%lld,%lld,%lld", gt, gq, dp - ad, ad, dp, pl[0], pl[1], pl[2]);
    return buf;
}

/// Sorted, de-duplicated copy of the called cells.
inline std::vector<uint32_t> sorted_cells(const std::vector<uint32_t>& cells)
{
    std::vector<uint32_t> v(cells);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

inline std::string contig_header(const std::vector<std::string>& snp_names, const std::vector<uint32_t>& rows)
{
    std::string h, last;
    std::vector<std::string> seen;
    for (uint32_t s : rows) {
        std::string chrom; int32_t pos; char ref, alt;
        parse_snp_name(snp_names[s], chrom, pos, ref, alt);
        if (chrom == last) continue;
        last = chrom;
        if (std::find(seen.begin(), seen.end(), chrom) != seen.end()) continue;
        seen.push_back(chrom);
        h += "##contig=<ID=" + chrom + ">\n";
    }
    return h;
}

/// private/genotype/cellsnp/. CSC inputs are SNPs x barcodes (snp_ad_csc / snp_dp_csc). Returns the number of callable sites, or -1 on a
/// write failure.
template <class CSC>
inline long long write_cellsnp_dir(const std::string& dir, const CSC& ad, const CSC& dp,
                                   const std::vector<uint32_t>& called, const std::vector<std::string>& barcodes,
                                   const std::vector<std::string>& snp_names, uint32_t min_count = kCellsnpMinCount)
{
    std::filesystem::create_directories(dir);
    const auto cells = sorted_cells(called);
    const uint32_t n_snps = dp.nrows;
    std::vector<uint64_t> agg_dp(n_snps, 0), agg_ad(n_snps, 0);
    for (uint32_t c : cells) {
        if (c >= dp.ncols) continue;
        for (int64_t i = dp.indptr[c]; i < dp.indptr[c + 1]; ++i) agg_dp[dp.indices[i]] += dp.data[i];
        for (int64_t i = ad.indptr[c]; i < ad.indptr[c + 1]; ++i) agg_ad[ad.indices[i]] += ad.data[i];
    }
    std::vector<int64_t> row_of(n_snps, -1);
    std::vector<uint32_t> rows;
    for (uint32_t s = 0; s < n_snps && s < snp_names.size(); ++s)
        if (agg_dp[s] >= min_count) { row_of[s] = static_cast<int64_t>(rows.size()); rows.push_back(s); }

    // base VCF: the callable sites with their totals over the called cells
    {
        GzOut f(dir + "/cellSNP.base.vcf.gz");
        if (!f.ok()) return -1;
        f.put("##fileformat=VCFv4.2\n##source=singlet genotype_export (cellsnp-lite layout)\n"
              "##singlet_callable=DP summed over called cells >= " + std::to_string(min_count) + "\n");
        f.put(contig_header(snp_names, rows));
        f.put("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"total counts for ALT and REF\">\n"
              "##INFO=<ID=AD,Number=1,Type=Integer,Description=\"total counts for ALT\">\n"
              "##INFO=<ID=OTH,Number=1,Type=Integer,Description=\"total counts for other bases from REF and ALT\">\n"
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n");
        char buf[256];
        for (uint32_t s : rows) {
            std::string chrom; int32_t pos; char ref, alt;
            parse_snp_name(snp_names[s], chrom, pos, ref, alt);
            int n = std::snprintf(buf, sizeof(buf), "%s\t%d\t.\t%c\t%c\t.\tPASS\tAD=%llu;DP=%llu;OTH=0\n", chrom.c_str(), pos, ref, alt,
                                  static_cast<unsigned long long>(agg_ad[s]), static_cast<unsigned long long>(agg_dp[s]));
            if (n > 0) f.put(std::string(buf, static_cast<size_t>(n)));
        }
    }
    {
        FILE* f = std::fopen((dir + "/cellSNP.samples.tsv").c_str(), "wb");
        if (!f) return -1;
        for (uint32_t c : cells) std::fprintf(f, "%s\n", c < barcodes.size() ? barcodes[c].c_str() : std::to_string(c).c_str());
        std::fclose(f);
    }
    // AD / DP: entries sorted by (site, cell), 1-based, cellsnp-lite's tab-separated MatrixMarket
    auto write_mtx_tag = [&](const std::string& path, const CSC& m) -> bool {
        std::vector<std::pair<uint64_t, uint32_t>> e;
        for (uint32_t j = 0; j < cells.size(); ++j) {
            const uint32_t c = cells[j];
            if (c >= m.ncols) continue;
            for (int64_t i = m.indptr[c]; i < m.indptr[c + 1]; ++i) {
                const int64_t r = row_of[m.indices[i]];
                if (r >= 0 && m.data[i] > 0) e.emplace_back((static_cast<uint64_t>(r) << 32) | j, static_cast<uint32_t>(m.data[i]));
            }
        }
        std::sort(e.begin(), e.end());
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        std::fprintf(f, "%%%%MatrixMarket matrix coordinate integer general\n%%\n%zu\t%zu\t%zu\n", rows.size(), cells.size(), e.size());
        for (const auto& [k, v] : e)
            std::fprintf(f, "%llu\t%llu\t%u\n", static_cast<unsigned long long>((k >> 32) + 1),
                         static_cast<unsigned long long>((k & 0xffffffffull) + 1), v);
        return std::fclose(f) == 0;
    };
    if (!write_mtx_tag(dir + "/cellSNP.tag.AD.mtx", ad)) return -1;
    if (!write_mtx_tag(dir + "/cellSNP.tag.DP.mtx", dp)) return -1;
    {
        FILE* f = std::fopen((dir + "/cellSNP.tag.OTH.mtx").c_str(), "wb");
        if (!f) return -1;
        std::fprintf(f, "%%%%MatrixMarket matrix coordinate integer general\n%%\n%zu\t%zu\t0\n", rows.size(), cells.size());
        std::fclose(f);
    }
    return static_cast<long long>(rows.size());
}

/// private/genotype/donors.vcf.gz. Returns the number of sites written, or -1 on a write failure, 0 when there is no donor.
template <class CSC>
inline long long write_donors_vcf(const std::string& path, const DemuxResult& demux, const CSC& ad, const CSC& dp,
                                  const std::vector<uint32_t>& called, const std::vector<std::string>& snp_names)
{
    const int K = demux.n_donors_k;
    if (K <= 0) return 0;
    const uint32_t n_snps = dp.nrows;
    const auto cells = sorted_cells(called);
    // the cells that count: called, cell_status "cell", a single donor with prob_max >= 0.9
    std::vector<std::pair<uint32_t, int>> use;
    for (uint32_t c : cells) {
        if (c >= dp.ncols || c >= demux.assignments.size()) continue;
        const auto& a = demux.assignments[c];
        if (a.cell_status != "cell" || a.donor_id < 0 || a.donor_id >= K || a.prob_max < kDonorMinProb) continue;
        if (a.label != "donor" + std::to_string(a.donor_id)) continue;   // doublet / unassigned (the rule the backfill reads from the TSV)
        use.emplace_back(c, a.donor_id);
    }
    // pass 1: sites with depth in those cells (panel order); pass 2: per-donor sums over those sites only
    std::vector<int64_t> site_of(n_snps, -1);
    for (const auto& [c, k] : use)
        for (int64_t i = dp.indptr[c]; i < dp.indptr[c + 1]; ++i) if (dp.data[i] > 0) site_of[dp.indices[i]] = 0;
    std::vector<uint32_t> rows;
    for (uint32_t s = 0; s < n_snps; ++s) {
        if (site_of[s] < 0) continue;
        if (s >= snp_names.size()) { site_of[s] = -1; continue; }
        site_of[s] = static_cast<int64_t>(rows.size());
        rows.push_back(s);
    }
    std::vector<uint32_t> sdp(rows.size() * K, 0), sad(rows.size() * K, 0);
    for (const auto& [c, k] : use) {
        for (int64_t i = dp.indptr[c]; i < dp.indptr[c + 1]; ++i) {
            const int64_t r = site_of[dp.indices[i]];
            if (r >= 0) sdp[static_cast<size_t>(r) * K + k] += dp.data[i];
        }
        for (int64_t i = ad.indptr[c]; i < ad.indptr[c + 1]; ++i) {
            const int64_t r = site_of[ad.indices[i]];
            if (r >= 0) sad[static_cast<size_t>(r) * K + k] += ad.data[i];
        }
    }

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    GzOut f(path);
    if (!f.ok()) return -1;
    f.put("##fileformat=VCFv4.2\n##source=singlet genotype_export\n"
          "##singlet_genotype_model=binomial on donor-summed UMI counts, error 0.01, flat prior; GT when the posterior >= 0.99\n"
          "##singlet_cells=called cells (cell_status cell) with donor prob_max >= 0.9\n");
    f.put(contig_header(snp_names, rows));
    f.put("##FILTER=<ID=PASS,Description=\"All filters passed\">\n"
          "##INFO=<ID=NS,Number=1,Type=Integer,Description=\"Number of donors with coverage at the site\">\n"
          "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype; ./. when the best posterior is below 0.99\">\n"
          "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Phred-scaled 1 - posterior of the most likely genotype, capped at 99\">\n"
          "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Ref and alt UMI counts summed over the donor's cells\">\n"
          "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Ref + alt UMI count summed over the donor's cells\">\n"
          "##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"Phred-scaled genotype likelihoods, normalized to the best\">\n"
          "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT");
    for (int k = 0; k < K; ++k) f.put("\tdonor" + std::to_string(k));
    f.put("\n");
    for (size_t ri = 0; ri < rows.size(); ++ri) {
        const uint32_t s = rows[ri];
        std::string chrom; int32_t pos; char ref, alt;
        parse_snp_name(snp_names[s], chrom, pos, ref, alt);
        int ns = 0;
        for (int k = 0; k < K; ++k) if (sdp[ri * K + k] > 0) ++ns;
        std::string line = chrom + "\t" + std::to_string(pos) + "\t" + snp_names[s] + "\t" + ref + "\t" + alt + "\t.\tPASS\tNS=" +
                           std::to_string(ns) + "\tGT:GQ:AD:DP:PL";
        for (int k = 0; k < K; ++k) {
            line += '\t';
            line += donor_fields(sad[ri * K + k], sdp[ri * K + k]);
        }
        line += '\n';
        f.put(line);
    }
    return static_cast<long long>(rows.size());
}

}  // namespace genotype
}  // namespace singlet
