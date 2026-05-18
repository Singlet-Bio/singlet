// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: gene_model.h
// Hierarchical gene model: Gene → Exons + Introns + Splice Junctions
// Built from GTF, provides fast interval queries and gene-level aggregation.
//
// Design:
// - Parse GTF once: extract genes + exons
// - Compute introns as gene body minus merged exon intervals
// - Gene-level counts = sum of exon counts + sum of intron counts
// - Splice junctions = CIGAR N operations (donor-acceptor pairs)
// - Each feature (exon/intron) has a global index and a gene_id

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <zlib.h>

#include "interval_tree.h"

namespace singlet {

// A gene with its exons and computed introns
struct GeneRecord {
    std::string gene_id;
    std::string gene_name;
    std::string chrom;
    int32_t start;      // 0-based gene body start
    int32_t end;        // 0-based gene body end (exclusive)
    char strand;
    uint32_t gene_idx;  // index in gene-level output

    // Exon global indices (into the flat exon feature list)
    std::vector<uint32_t> exon_indices;
    // Intron global indices
    std::vector<uint32_t> intron_indices;
};

class GeneModel {
public:
    GeneModel() = default;

    // Load genes and exons from GTF file.
    // Returns false on error.
    bool load_gtf(const std::string& path) {
        // Use gzip-aware reader
        bool is_gz = (path.size() >= 3 && path.substr(path.size() - 3) == ".gz");
        gzFile gz = nullptr;
        std::ifstream plain;

        if (is_gz) {
            gz = gzopen(path.c_str(), "r");
            if (!gz) {
                std::cerr << "[gene_model] Cannot open: " << path << "\n";
                return false;
            }
            gzbuffer(gz, 256 * 1024);
        } else {
            plain.open(path);
            if (!plain.is_open()) {
                std::cerr << "[gene_model] Cannot open: " << path << "\n";
                return false;
            }
        }

        // Temporary: collect exon intervals per gene_id
        // gene_id → {chrom, start, end, strand, gene_name}

        std::unordered_map<std::string, RawGene> gene_map;
        std::vector<RawExon> raw_exons;

        auto read_line = [&](std::string& line) -> bool {
            if (gz) {
                char buf[8192];
                line.clear();
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

        std::string line;
        while (read_line(line)) {
            if (line.empty() || line[0] == '#') continue;

            // Fast field extraction: chrom[0] source[1] feature[2] start[3] end[4] score[5] strand[6] frame[7] attrs[8]
            const char* p = line.c_str();
            const char* fields[9];
            int fi = 0;
            fields[0] = p;
            while (*p && fi < 8) {
                if (*p == '\t') {
                    fi++;
                    fields[fi] = p + 1;
                }
                p++;
            }
            if (fi < 8) continue;

            // Extract feature type (field 2)
            const char* feat_start = fields[2];
            const char* feat_end = fields[3] - 1;  // before tab

            bool is_gene = (feat_end - feat_start == 4 && strncmp(feat_start, "gene", 4) == 0);
            bool is_exon = (feat_end - feat_start == 4 && strncmp(feat_start, "exon", 4) == 0);
            if (!is_gene && !is_exon) continue;

            // Parse chrom, start, end, strand
            std::string chrom(fields[0], fields[1] - 1);
            int32_t start = atoi(fields[3]) - 1;  // GTF 1-based → 0-based
            int32_t end = atoi(fields[4]);         // GTF end is inclusive, but we keep as-is for half-open
            char strand = *fields[6];

            // Extract gene_id and gene_name from attributes
            const char* attrs = fields[8];
            std::string gene_id = extract_attr(attrs, "gene_id");
            std::string gene_name = extract_attr(attrs, "gene_name");

            if (gene_id.empty()) continue;

            if (is_gene) {
                auto& g = gene_map[gene_id];
                g.chrom = chrom;
                g.start = start;
                g.end = end;
                g.strand = strand;
                g.gene_name = gene_name;
            } else if (is_exon) {
                raw_exons.push_back({chrom, start, end, gene_id, gene_name});
            }
        }

        if (gz) gzclose(gz);

        std::cerr << "[gene_model] Parsed " << gene_map.size() << " genes, "
                  << raw_exons.size() << " exons from GTF\n";

        // Build gene records
        build_model(gene_map, raw_exons);
        return true;
    }

    // Load from BED file (exons only, no gene hierarchy)
    bool load_bed(const std::string& path) {
        gzFile gz = nullptr;
        std::ifstream plain;
        bool is_gz = (path.size() >= 3 && path.substr(path.size() - 3) == ".gz");

        if (is_gz) {
            gz = gzopen(path.c_str(), "r");
            if (!gz) return false;
        } else {
            plain.open(path);
            if (!plain.is_open()) return false;
        }

        auto read_line = [&](std::string& line) -> bool {
            if (gz) {
                char buf[4096];
                line.clear();
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

        std::string line;
        while (read_line(line)) {
            if (line.empty() || line[0] == '#') continue;
            char chrom[64] = {}, bed_name[256] = {};
            int32_t start = 0, end = 0;
            if (sscanf(line.c_str(), "%63s\t%d\t%d\t%255s", chrom, &start, &end, bed_name) < 3) continue;

            std::string name = bed_name[0] ? std::string(bed_name) :
                (std::string(chrom) + ":" + std::to_string(start) + "-" + std::to_string(end));

            uint32_t idx = static_cast<uint32_t>(exon_names_.size());
            Interval iv{start, end, idx};
            exon_names_.push_back(name);
            exon_starts_.push_back(start);
            exon_ends_.push_back(end);
            // BED format: each exon is its own pseudo-gene (no gene hierarchy)
            exon_to_gene_.push_back(idx);
            gene_strands_.push_back('.');
            chrom_exon_intervals_[std::string(chrom)].push_back(iv);
            n_exons_++;
        }

        if (gz) gzclose(gz);
        std::cerr << "[gene_model] Loaded " << n_exons_ << " exons from BED (no gene hierarchy)\n";
        return true;
    }

    // Resolve chromosome names against BAM header
    void resolve_chroms(const std::unordered_map<std::string, int32_t>& chrom_to_tid) {
        chrom_to_tid_ = &chrom_to_tid;

        // Find max tid for vector sizing
        int32_t max_tid = -1;
        for (auto& [_, tid] : chrom_to_tid) {
            if (tid > max_tid) max_tid = tid;
        }
        if (max_tid >= 0) {
            exon_trees_.resize(max_tid + 1);
            intron_trees_.resize(max_tid + 1);
        }

        // Build interval trees for exons per tid
        uint32_t n_exon_chroms = 0;
        for (auto& [chrom, intervals] : chrom_exon_intervals_) {
            int32_t tid = resolve_chrom(chrom);
            if (tid >= 0) {
                exon_trees_[tid].build(std::move(intervals));
                n_exon_chroms++;
            }
        }
        chrom_exon_intervals_.clear();

        // Build interval trees for introns per tid
        uint32_t n_intron_chroms = 0;
        for (auto& [chrom, intervals] : chrom_intron_intervals_) {
            int32_t tid = resolve_chrom(chrom);
            if (tid >= 0) {
                intron_trees_[tid].build(std::move(intervals));
                n_intron_chroms++;
            }
        }
        chrom_intron_intervals_.clear();

        std::cerr << "[gene_model] Resolved to " << n_exon_chroms
                  << " exon chromosomes, " << n_intron_chroms
                  << " intron chromosomes\n";
    }

    // Query which exons overlap [start, end)
    void query_exons(int32_t tid, int32_t start, int32_t end,
                     std::vector<uint32_t>& results) const {
        if (tid >= 0 && static_cast<size_t>(tid) < exon_trees_.size()) {
            exon_trees_[tid].query(start, end, results);
        } else {
            results.clear();
        }
    }

    // Query which introns overlap [start, end)
    void query_introns(int32_t tid, int32_t start, int32_t end,
                       std::vector<uint32_t>& results) const {
        if (tid >= 0 && static_cast<size_t>(tid) < intron_trees_.size()) {
            intron_trees_[tid].query(start, end, results);
        } else {
            results.clear();
        }
    }

    // Accessors
    uint32_t n_exons() const { return n_exons_; }
    uint32_t n_introns() const { return n_introns_; }
    uint32_t n_genes() const { return static_cast<uint32_t>(genes_.size()); }

    // True if genes were loaded from GTF (real gene hierarchy, not BED pseudo-genes)
    bool has_gene_hierarchy() const { return has_gene_hierarchy_; }

    const std::vector<std::string>& exon_names() const { return exon_names_; }
    const std::vector<std::string>& intron_names() const { return intron_names_; }
    const std::vector<std::string>& gene_names() const { return gene_names_; }
    const std::vector<std::string>& gene_ids() const { return gene_ids_; }
    const std::vector<GeneRecord>& genes() const { return genes_; }

    // Get the gene index for a given exon index (for UMI dedup grouping)
    uint32_t exon_to_gene(uint32_t exon_idx) const {
        if (exon_idx < exon_to_gene_.size()) return exon_to_gene_[exon_idx];
        return UINT32_MAX;
    }

    // Get the gene index for a given intron index
    uint32_t intron_to_gene(uint32_t intron_idx) const {
        if (intron_idx < intron_to_gene_.size()) return intron_to_gene_[intron_idx];
        return UINT32_MAX;
    }

    // Get exon boundaries for containment checking
    int32_t exon_start(uint32_t exon_idx) const { return exon_starts_[exon_idx]; }
    int32_t exon_end(uint32_t exon_idx) const { return exon_ends_[exon_idx]; }

    // Get gene strand ('+', '-', or '.')
    char gene_strand(uint32_t gene_idx) const {
        if (gene_idx < gene_strands_.size()) return gene_strands_[gene_idx];
        return '.';
    }

    // Compute gene-level counts by summing exon and intron counts per gene.
    // Returns vectors of size n_genes.
    // exon_csc and intron_csc must have same number of columns (barcodes).
    template <typename T>
    void sum_to_genes(const std::vector<int32_t>& exon_indptr,
                      const std::vector<int32_t>& exon_indices,
                      const std::vector<T>& exon_data,
                      const std::vector<int32_t>& intron_indptr,
                      const std::vector<int32_t>& intron_indices,
                      const std::vector<T>& intron_data,
                      uint32_t n_barcodes,
                      std::vector<int32_t>& gene_indptr,
                      std::vector<int32_t>& gene_indices,
                      std::vector<T>& gene_data) const {
        uint32_t ng = static_cast<uint32_t>(genes_.size());
        gene_indptr.assign(n_barcodes + 1, 0);
        gene_indices.clear();
        gene_data.clear();

        // For each barcode column
        for (uint32_t bc = 0; bc < n_barcodes; ++bc) {
            // Accumulate gene counts for this barcode
            std::vector<T> gene_counts(ng, 0);

            // Sum exon counts
            for (int32_t k = exon_indptr[bc]; k < exon_indptr[bc + 1]; ++k) {
                uint32_t exon_idx = static_cast<uint32_t>(exon_indices[k]);
                uint32_t gene_idx = exon_to_gene(exon_idx);
                if (gene_idx < ng) {
                    gene_counts[gene_idx] += exon_data[k];
                }
            }

            // Sum intron counts
            for (int32_t k = intron_indptr[bc]; k < intron_indptr[bc + 1]; ++k) {
                uint32_t intron_idx = static_cast<uint32_t>(intron_indices[k]);
                uint32_t gene_idx = intron_to_gene(intron_idx);
                if (gene_idx < ng) {
                    gene_counts[gene_idx] += intron_data[k];
                }
            }

            // Emit nonzeros
            for (uint32_t g = 0; g < ng; ++g) {
                if (gene_counts[g] > 0) {
                    gene_indices.push_back(static_cast<int32_t>(g));
                    gene_data.push_back(gene_counts[g]);
                }
            }
            gene_indptr[bc + 1] = static_cast<int32_t>(gene_data.size());
        }
    }

private:
    // Gene records
    std::vector<GeneRecord> genes_;
    std::vector<std::string> gene_names_;
    std::vector<std::string> gene_ids_;
    bool has_gene_hierarchy_ = false;  // true for GTF, false for BED

    // Exon features
    uint32_t n_exons_ = 0;
    std::vector<std::string> exon_names_;
    std::vector<uint32_t> exon_to_gene_;  // exon global idx → gene idx
    std::vector<int32_t> exon_starts_;    // exon start (0-based)
    std::vector<int32_t> exon_ends_;      // exon end (0-based, exclusive)

    // Gene strands (indexed by gene_idx)
    std::vector<char> gene_strands_;

    // Intron features
    uint32_t n_introns_ = 0;
    std::vector<std::string> intron_names_;
    std::vector<uint32_t> intron_to_gene_;  // intron global idx → gene idx

    // Temporary: intervals before chromosome resolution
    std::unordered_map<std::string, std::vector<Interval>> chrom_exon_intervals_;
    std::unordered_map<std::string, std::vector<Interval>> chrom_intron_intervals_;

    // Resolved interval trees per tid (vector-indexed for O(1) lookup)
    std::vector<IntervalTree> exon_trees_;
    std::vector<IntervalTree> intron_trees_;

    // Pointer to chrom_to_tid map (from PileupEngine)
    const std::unordered_map<std::string, int32_t>* chrom_to_tid_ = nullptr;

    int32_t resolve_chrom(const std::string& chrom) const {
        if (!chrom_to_tid_) return -1;
        auto it = chrom_to_tid_->find(chrom);
        if (it != chrom_to_tid_->end()) return it->second;
        // Try chr prefix normalization
        if (chrom.substr(0, 3) != "chr") {
            std::string with_chr = "chr" + chrom;
            if (chrom == "MT") with_chr = "chrM";
            it = chrom_to_tid_->find(with_chr);
            if (it != chrom_to_tid_->end()) return it->second;
        } else {
            std::string without_chr = chrom.substr(3);
            if (chrom == "chrM") without_chr = "MT";
            it = chrom_to_tid_->find(without_chr);
            if (it != chrom_to_tid_->end()) return it->second;
        }
        return -1;
    }

    static std::string extract_attr(const char* attrs, const char* key) {
        // Search for key "value" pattern in GTF attributes
        std::string search = std::string(key) + " \"";
        const char* p = strstr(attrs, search.c_str());
        if (!p) return {};
        p += search.size();
        const char* q = strchr(p, '"');
        if (!q) return {};
        return std::string(p, q);
    }

    struct RawGene {
        std::string chrom;
        int32_t start = 0;
        int32_t end = 0;
        char strand = '.';
        std::string gene_name;
    };

    struct RawExon {
        std::string chrom;
        int32_t start = 0;
        int32_t end = 0;
        std::string gene_id;
        std::string gene_name;
    };

    void build_model(const std::unordered_map<std::string, RawGene>& gene_map,
                     const std::vector<RawExon>& raw_exons) {
        // Group exons by gene_id
        std::unordered_map<std::string, std::vector<std::pair<int32_t, int32_t>>> exons_per_gene;
        for (const auto& ex : raw_exons) {
            exons_per_gene[ex.gene_id].push_back({ex.start, ex.end});
        }

        // Build genes in deterministic order (sorted by gene_id)
        std::vector<std::string> sorted_gene_ids;
        sorted_gene_ids.reserve(gene_map.size());
        for (const auto& [gid, _] : gene_map) {
            // Only include genes that have exons
            if (exons_per_gene.count(gid)) {
                sorted_gene_ids.push_back(gid);
            }
        }
        std::sort(sorted_gene_ids.begin(), sorted_gene_ids.end());

        uint32_t gene_idx = 0;
        for (const auto& gid : sorted_gene_ids) {
            const auto& ginfo = gene_map.at(gid);
            GeneRecord gene;
            gene.gene_id = gid;
            gene.gene_name = ginfo.gene_name;
            gene.chrom = ginfo.chrom;
            gene.start = ginfo.start;
            gene.end = ginfo.end;
            gene.strand = ginfo.strand;
            gene.gene_idx = gene_idx;

            // Sort and merge overlapping exons for this gene
            auto& exons = exons_per_gene[gid];
            std::sort(exons.begin(), exons.end());
            auto merged = merge_intervals(exons);

            // Add exon intervals
            for (const auto& [estart, eend] : merged) {
                uint32_t eidx = n_exons_++;
                std::string ename = gid + "_" + ginfo.gene_name + "_" + ginfo.chrom + ":" +
                                   std::to_string(estart) + "-" + std::to_string(eend);
                exon_names_.push_back(ename);
                exon_to_gene_.push_back(gene_idx);
                exon_starts_.push_back(estart);
                exon_ends_.push_back(eend);
                gene.exon_indices.push_back(eidx);

                Interval iv{estart, eend, eidx};
                chrom_exon_intervals_[ginfo.chrom].push_back(iv);
            }

            // Compute intron intervals (gene body minus merged exons)
            auto introns = compute_introns(ginfo.start, ginfo.end, merged);
            for (const auto& [istart, iend] : introns) {
                uint32_t iidx = n_introns_++;
                std::string iname = gid + "_" + ginfo.gene_name + "_intron_" + ginfo.chrom + ":" +
                                   std::to_string(istart) + "-" + std::to_string(iend);
                intron_names_.push_back(iname);
                intron_to_gene_.push_back(gene_idx);
                gene.intron_indices.push_back(iidx);

                Interval iv{istart, iend, iidx};
                chrom_intron_intervals_[ginfo.chrom].push_back(iv);
            }

            genes_.push_back(std::move(gene));
            gene_names_.push_back(ginfo.gene_name.empty() ? gid : ginfo.gene_name);
            gene_ids_.push_back(gid);
            gene_strands_.push_back(ginfo.strand);
            gene_idx++;
        }

        std::cerr << "[gene_model] Built model: " << genes_.size() << " genes, "
                  << n_exons_ << " exons, " << n_introns_ << " introns\n";
        has_gene_hierarchy_ = true;
    }

    static std::vector<std::pair<int32_t, int32_t>> merge_intervals(
            std::vector<std::pair<int32_t, int32_t>>& intervals) {
        std::vector<std::pair<int32_t, int32_t>> merged;
        if (intervals.empty()) return merged;
        std::sort(intervals.begin(), intervals.end());
        merged.push_back(intervals[0]);
        for (size_t i = 1; i < intervals.size(); ++i) {
            if (intervals[i].first <= merged.back().second) {
                merged.back().second = std::max(merged.back().second, intervals[i].second);
            } else {
                merged.push_back(intervals[i]);
            }
        }
        return merged;
    }

    static std::vector<std::pair<int32_t, int32_t>> compute_introns(
            int32_t gene_start, int32_t gene_end,
            const std::vector<std::pair<int32_t, int32_t>>& merged_exons) {
        std::vector<std::pair<int32_t, int32_t>> introns;
        if (merged_exons.empty()) return introns;

        // Intron before first exon (within gene body)
        if (gene_start < merged_exons[0].first) {
            introns.push_back({gene_start, merged_exons[0].first});
        }

        // Introns between exons
        for (size_t i = 1; i < merged_exons.size(); ++i) {
            int32_t istart = merged_exons[i - 1].second;
            int32_t iend = merged_exons[i].first;
            if (istart < iend) {
                introns.push_back({istart, iend});
            }
        }

        // Intron after last exon (within gene body)
        if (merged_exons.back().second < gene_end) {
            introns.push_back({merged_exons.back().second, gene_end});
        }

        return introns;
    }
};

} // namespace singlet
