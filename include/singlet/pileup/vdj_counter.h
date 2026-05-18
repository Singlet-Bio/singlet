// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: vdj_counter.h — N17
// V(D)J immune receptor gene usage counting per cell barcode.
//
// Approach: gene-body intervals (no exon hierarchy) for all V/D/J/C segments.
// Any read whose alignment overlaps a VDJ gene body is counted.
// UMI deduplication: exact (same pattern as exon counting).
// No strand filtering: VDJ recombined loci produce complex read orientations.
//
// Supported biotypes (Ensembl GTF gene_biotype field):
//   IG_V_gene, IG_D_gene, IG_J_gene, IG_C_gene     (immunoglobulin)
//   TR_V_gene, TR_D_gene, TR_J_gene, TR_C_gene     (T-cell receptor)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <zlib.h>

#include "interval_tree.h"

namespace singlet {

/// Set of V(D)J biotypes to collect.
static inline bool is_vdj_biotype(const char* biotype) {
    return (strncmp(biotype, "IG_V_gene", 9) == 0 ||
            strncmp(biotype, "IG_D_gene", 9) == 0 ||
            strncmp(biotype, "IG_J_gene", 9) == 0 ||
            strncmp(biotype, "IG_C_gene", 9) == 0 ||
            strncmp(biotype, "TR_V_gene", 9) == 0 ||
            strncmp(biotype, "TR_D_gene", 9) == 0 ||
            strncmp(biotype, "TR_J_gene", 9) == 0 ||
            strncmp(biotype, "TR_C_gene", 9) == 0);
}

/// Lightweight model: gene-body intervals for all VDJ segments.
class VdjModel {
public:
    VdjModel() = default;

    /// Parse a GTF (plain or .gz) and extract VDJ gene-body intervals.
    /// Returns false on I/O error.
    bool load_gtf(const std::string& path) {
        bool is_gz = (path.size() >= 3 && path.substr(path.size() - 3) == ".gz");
        gzFile gz = nullptr;
        std::ifstream plain;

        if (is_gz) {
            gz = gzopen(path.c_str(), "r");
            if (!gz) {
                std::cerr << "[vdj_model] Cannot open: " << path << "\n";
                return false;
            }
            gzbuffer(gz, 256 * 1024);
        } else {
            plain.open(path);
            if (!plain.is_open()) {
                std::cerr << "[vdj_model] Cannot open: " << path << "\n";
                return false;
            }
        }

        auto read_line = [&](std::string& line) -> bool {
            if (gz) {
                line.clear();
                char buf[8192];
                while (gzgets(gz, buf, sizeof(buf))) {
                    line += buf;
                    if (!line.empty() && line.back() == '\n') {
                        line.pop_back();
                        if (!line.empty() && line.back() == '\r') line.pop_back();
                        return true;
                    }
                }
                return !line.empty();
            } else {
                return static_cast<bool>(std::getline(plain, line));
            }
        };

        // Temporary: accumulate gene-body intervals per chromosome
        struct RawGene {
            std::string chrom;
            int32_t start;
            int32_t end;
            std::string gene_name;
            std::string biotype;
        };
        std::vector<RawGene> raw;

        std::string line;
        while (read_line(line)) {
            if (line.empty() || line[0] == '#') continue;

            // Tab-split: chrom[0] source[1] feature[2] start[3] end[4] . strand[6] . attrs[8]
            const char* p = line.c_str();
            const char* fields[9];
            int fi = 0;
            fields[0] = p;
            while (*p && fi < 8) {
                if (*p == '\t') { fi++; fields[fi] = p + 1; }
                ++p;
            }
            if (fi < 8) continue;

            // Only "gene" records carry gene_biotype
            const char* feat_s = fields[2];
            const char* feat_e = fields[3] - 1;
            if (feat_e - feat_s != 4 || strncmp(feat_s, "gene", 4) != 0) continue;

            // Extract gene_biotype or gene_type from attrs
            const char* attrs = fields[8];
            std::string biotype = extract_attr(attrs, "gene_biotype");
            if (biotype.empty()) biotype = extract_attr(attrs, "gene_type");  // GENCODE format
            if (biotype.empty() || !is_vdj_biotype(biotype.c_str())) continue;

            std::string chrom(fields[0], fields[1] - 1);
            int32_t start = atoi(fields[3]) - 1;   // GTF 1-based → 0-based
            int32_t end   = atoi(fields[4]);        // GTF end inclusive → keep as exclusive

            std::string gene_name = extract_attr(attrs, "gene_name");
            if (gene_name.empty()) gene_name = extract_attr(attrs, "gene_id");

            raw.push_back({std::move(chrom), start, end,
                           std::move(gene_name), std::move(biotype)});
        }
        if (gz) gzclose(gz);

        if (raw.empty()) {
            std::cerr << "[vdj_model] WARNING: No VDJ genes found in: " << path
                      << " (check gene_biotype field)\n";
            return true;  // not an error — dataset may simply have no VDJ annotations
        }

        // Sort by chrom then position for deterministic output
        std::sort(raw.begin(), raw.end(), [](const RawGene& a, const RawGene& b) {
            if (a.chrom != b.chrom) return a.chrom < b.chrom;
            return a.start < b.start;
        });

        // Build gene list + temp chrom buckets
        gene_names_.reserve(raw.size());
        gene_biotypes_.reserve(raw.size());
        std::unordered_map<std::string, std::vector<Interval>> chrom_intervals;

        for (const auto& g : raw) {
            uint32_t idx = static_cast<uint32_t>(gene_names_.size());
            std::string display = g.gene_name + " (" + g.biotype + ")";
            gene_names_.push_back(std::move(display));
            gene_biotypes_.push_back(g.biotype);
            chrom_intervals[g.chrom].push_back({g.start, g.end, idx});
        }

        pending_intervals_ = std::move(chrom_intervals);
        std::cerr << "[vdj_model] Found " << gene_names_.size() << " VDJ genes\n";
        return true;
    }

    /// Resolve chromosome names against the BAM header chromosome map.
    /// Must be called after load_gtf() and once the BAM header is available.
    void resolve_chroms(const std::unordered_map<std::string, int32_t>& chrom_to_tid) {
        int32_t max_tid = -1;
        for (auto& [_, tid] : chrom_to_tid) {
            if (tid > max_tid) max_tid = tid;
        }
        if (max_tid < 0) return;
        trees_.resize(max_tid + 1);

        uint32_t n_resolved = 0;
        for (auto& [chrom, intervals] : pending_intervals_) {
            int32_t tid = resolve_one(chrom, chrom_to_tid);
            if (tid >= 0) {
                trees_[tid].build(std::move(intervals));
                ++n_resolved;
            }
        }
        pending_intervals_.clear();
        std::cerr << "[vdj_model] Resolved " << n_resolved << " VDJ chromosomes\n";
    }

    /// Query overlapping VDJ genes for read span [start, end).
    /// Appends gene indices to results (does not clear).
    void query(int32_t tid, int32_t start, int32_t end,
               std::vector<uint32_t>& results) const {
        if (tid >= 0 && static_cast<size_t>(tid) < trees_.size()) {
            trees_[tid].query(start, end, results);
        }
    }

    uint32_t n_genes() const { return static_cast<uint32_t>(gene_names_.size()); }
    bool empty() const { return gene_names_.empty(); }
    const std::vector<std::string>& gene_names() const { return gene_names_; }

private:
    std::vector<std::string> gene_names_;    ///< "IGHV1-2 (IG_V_gene)"
    std::vector<std::string> gene_biotypes_; ///< raw biotype strings
    std::vector<IntervalTree> trees_;        ///< indexed by BAM tid
    std::unordered_map<std::string, std::vector<Interval>> pending_intervals_;

    static std::string extract_attr(const char* attrs, const char* key) {
        std::string search = std::string(key) + " \"";
        const char* p = strstr(attrs, search.c_str());
        if (!p) return {};
        p += search.size();
        const char* q = strchr(p, '"');
        if (!q) return {};
        return std::string(p, q);
    }

    static int32_t resolve_one(const std::string& chrom,
                               const std::unordered_map<std::string, int32_t>& m) {
        auto it = m.find(chrom);
        if (it != m.end()) return it->second;
        if (chrom.substr(0, 3) != "chr") {
            std::string with_chr = "chr" + chrom;
            if (chrom == "MT") with_chr = "chrM";
            it = m.find(with_chr);
            if (it != m.end()) return it->second;
        } else {
            std::string without_chr = chrom.substr(3);
            if (chrom == "chrM") without_chr = "MT";
            it = m.find(without_chr);
            if (it != m.end()) return it->second;
        }
        return -1;
    }
};

} // namespace singlet
