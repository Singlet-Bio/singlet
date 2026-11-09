// D-52 backfill with the engine's own writers: reads an existing run's private/snp_{ad,dp}.1pz (with limits raised to the 1000G panel's
// 7.35 M rows; the Python binding caps rows at 4.2 M), cell_calls.tsv and donor_assignments.tsv, and calls genotype_export.h's
// write_cellsnp_dir / write_donors_vcf, so the output is what the engine writes at the end of a run of this branch.
//
//   genotype_backfill <run out dir> <dest dir>      -> <dest>/donors.vcf.gz, <dest>/cellsnp/, <dest>/cells.vcf.gz and, when the run has
//                                                     the D-101 all-site member (private/allsite/allele_{ad,dp}.1pz), <dest>/donors_allsite.vcf.gz
// Build (Clipper job): $CXX -O2 -std=c++17 -I include scripts/genotype/genotype_backfill.cpp -lzstd -lz -pthread
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "singlet/pileup/pz_reader.h"
#include "singlet/pileup/genotype_export.h"

struct CSC {
    uint32_t nrows = 0, ncols = 0;
    std::vector<uint32_t> indptr, indices, data;
};

static CSC load(const std::string& path, std::vector<std::string>& rows, std::vector<std::string>& cols) {
    singlet::pz::ReadLimits lim;
    lim.max_rows = 1ull << 26;
    lim.max_file_bytes = 1ull << 36;
    lim.max_nnz = 1ull << 32;
    auto r = singlet::pz::read_1pz(path, lim);
    CSC c;
    c.nrows = r.m; c.ncols = r.n;
    c.indptr = std::move(r.indptr); c.indices = std::move(r.indices); c.data = std::move(r.data);
    rows = std::move(r.rownames); cols = std::move(r.colnames);
    return c;
}

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v; std::string f; std::istringstream ss(s);
    while (std::getline(ss, f, '\t')) v.push_back(f);
    return v;
}

static int col_of(const std::vector<std::string>& head, const char* name) {
    for (size_t i = 0; i < head.size(); ++i) if (head[i] == name) return static_cast<int>(i);
    std::cerr << "missing column " << name << "\n"; std::exit(2);
}

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: genotype_backfill <run out dir> <dest dir>\n"; return 2; }
    const std::string run = argv[1], dest = argv[2];
    std::vector<std::string> snps, bcs, snps2, bcs2;
    const CSC ad = load(run + "/private/snp_ad.1pz", snps, bcs);
    const CSC dp = load(run + "/private/snp_dp.1pz", snps2, bcs2);
    if (snps != snps2 || bcs != bcs2 || ad.ncols != dp.ncols) { std::cerr << "snp_ad / snp_dp disagree\n"; return 3; }
    std::unordered_map<std::string, uint32_t> idx;
    for (uint32_t i = 0; i < bcs.size(); ++i) idx[bcs[i]] = i;

    std::vector<uint32_t> called;
    {
        std::ifstream f(run + "/cell_calls.tsv"); std::string line;
        std::getline(f, line); const int ci = col_of(split(line), "is_cell");
        while (std::getline(f, line)) {
            auto p = split(line);
            if (static_cast<int>(p.size()) > ci && p[ci] == "TRUE") { auto it = idx.find(p[0]); if (it != idx.end()) called.push_back(it->second); }
        }
    }
    singlet::DemuxResult dm;
    dm.assignments.assign(bcs.size(), {-1, 0.f, 0.f, "unassigned", "ambient"});
    {
        std::ifstream f(run + "/donor_assignments.tsv"); std::string line;
        if (f && std::getline(f, line)) {
            auto h = split(line);
            const int il = col_of(h, "donor_id"), ip = col_of(h, "prob_max"), is = col_of(h, "cell_status");
            while (std::getline(f, line)) {
                auto p = split(line);
                auto it = idx.find(p[0]);
                if (it == idx.end()) continue;
                auto& a = dm.assignments[it->second];
                a.label = p[il]; a.prob_max = std::strtof(p[ip].c_str(), nullptr); a.cell_status = p[is];
                if (a.label.rfind("donor", 0) == 0 && a.label.size() > 5 && a.label.find_first_not_of("0123456789", 5) == std::string::npos) {
                    a.donor_id = std::atoi(a.label.c_str() + 5);
                    dm.n_donors_k = std::max(dm.n_donors_k, a.donor_id + 1);
                }
            }
        }
    }
    const long long nc = singlet::genotype::write_cellsnp_dir(dest + "/cellsnp", ad, dp, called, bcs, snps);
    const long long nv = singlet::genotype::write_donors_vcf(dest + "/donors.vcf.gz", dm, ad, dp, called, snps);
    // D-101: all-site donor genotypes and the per-cell VCF (panel + all-site sites)
    std::vector<std::string> as_names, as_bcs, as_names2, as_bcs2;
    CSC aad, adp;
    long long na = 0;
    const bool has_as = std::filesystem::exists(run + "/private/allsite/allele_ad.1pz");
    if (has_as) {
        aad = load(run + "/private/allsite/allele_ad.1pz", as_names, as_bcs);
        adp = load(run + "/private/allsite/allele_dp.1pz", as_names2, as_bcs2);
        if (as_names != as_names2 || as_bcs != bcs || as_bcs2 != bcs) { std::cerr << "allele_ad / allele_dp / snp_ad disagree\n"; return 3; }
        na = singlet::genotype::write_donors_vcf(dest + "/donors_allsite.vcf.gz", dm, aad, adp, called, as_names);
    }
    std::vector<std::string> contigs;   // reference order: the all-site rows are in contig order; then any panel-only contig
    auto add_contig = [&](const std::string& n) {
        const std::string c = n.substr(0, n.find(':'));
        if ((contigs.empty() || contigs.back() != c) && std::find(contigs.begin(), contigs.end(), c) == contigs.end()) contigs.push_back(c);
    };
    for (const auto& n : as_names) add_contig(n);
    for (const auto& n : snps) add_contig(n);
    std::vector<singlet::genotype::SiteSet<CSC>> sets{{&ad, &dp, &snps, "panel"}};
    if (has_as) sets.push_back({&aad, &adp, &as_names, "allsite"});
    const long long ncv = singlet::genotype::write_cells_vcf(dest + "/cells.vcf.gz", dm, sets, called, bcs, contigs);
    std::printf("{\"n_barcodes\": %zu, \"n_called_cells\": %zu, \"n_panel_snps\": %zu, \"n_callable_sites\": %lld, \"n_donors\": %d, \"n_donor_sites\": %lld, "
                "\"n_allsite_sites\": %zu, \"n_donor_allsite_sites\": %lld, \"n_cells_vcf_rows\": %lld}\n",
                bcs.size(), called.size(), snps.size(), nc, dm.n_donors_k, nv, as_names.size(), na, ncv);
    return (nc < 0 || nv < 0 || na < 0 || ncv < 0) ? 1 : 0;
}
