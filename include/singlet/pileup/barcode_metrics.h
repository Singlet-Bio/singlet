// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: barcode_metrics.h
// G-BARCODE-METRICS — Per-barcode metrics TSV writer (Cell Ranger per_barcode_metrics.csv style).
//
// Writes a tab-separated file with per-barcode read and UMI counts, QC metrics,
// and cell-calling status. Compatible with downstream tools that consume
// Cell Ranger per_barcode_metrics.csv (just tab-separated instead of CSV).
//
// Usage:
//   std::vector<BarcodeMetric> m = build_barcode_metrics(barcodes, umis, n_genes, cells);
//   write_barcode_metrics(m, {"per_barcode_metrics.tsv"});

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

namespace singlet {

// ---------------------------------------------------------------------------
// Per-barcode metric record
// ---------------------------------------------------------------------------
struct BarcodeMetric {
    std::string barcode;

    uint64_t total_reads      = 0;  ///< Total reads assigned to this barcode
    uint64_t mapped_reads     = 0;  ///< Reads with a valid genome alignment
    uint64_t unique_reads     = 0;  ///< Uniquely mapped reads
    uint64_t exonic_reads     = 0;  ///< Reads overlapping an exon
    uint64_t intronic_reads   = 0;  ///< Reads overlapping an intron (not exonic)
    uint64_t intergenic_reads = 0;  ///< Reads in intergenic regions
    uint64_t antisense_reads  = 0;  ///< Reads mapped to the antisense strand

    uint32_t n_umi   = 0;  ///< Unique UMI count
    uint32_t n_genes = 0;  ///< Number of genes with ≥1 UMI

    double saturation    = 0.0;  ///< Sequencing saturation: 1 - (n_umi / total_reads)
    double mito_fraction = 0.0;  ///< Fraction of UMIs from mitochondrial genes

    bool is_cell = false;  ///< Barcode called as a cell (by EmptyDrops or knee)
};

// ---------------------------------------------------------------------------
// Writer configuration
// ---------------------------------------------------------------------------
struct BarcodeMetricsConfig {
    std::string filepath;             ///< Output path for the TSV file
    bool        header     = true;    ///< Write column header row
    bool        cells_only = false;   ///< If true, only write is_cell == true rows
};

// ---------------------------------------------------------------------------
// Header fields (matches the struct member order above)
// ---------------------------------------------------------------------------
static const char* kBarcodeMetricsHeader =
    "barcode\t"
    "total_reads\t"
    "mapped_reads\t"
    "unique_reads\t"
    "exonic_reads\t"
    "intronic_reads\t"
    "intergenic_reads\t"
    "antisense_reads\t"
    "n_umi\t"
    "n_genes\t"
    "saturation\t"
    "mito_fraction\t"
    "is_cell\n";

// ---------------------------------------------------------------------------
// Write per-barcode metrics to a TSV file
// ---------------------------------------------------------------------------
inline bool write_barcode_metrics(const std::vector<BarcodeMetric>& metrics,
                                  const BarcodeMetricsConfig&       cfg) {
    if (cfg.filepath.empty()) return false;

    std::FILE* fp = std::fopen(cfg.filepath.c_str(), "w");
    if (!fp) return false;

    if (cfg.header) {
        if (std::fputs(kBarcodeMetricsHeader, fp) == EOF) {
            std::fclose(fp);
            return false;
        }
    }

    for (const auto& m : metrics) {
        if (cfg.cells_only && !m.is_cell) continue;

        char buf[1024];
        int  n = std::snprintf(buf, sizeof(buf),
                               "%s\t"
                               "%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t"
                               "%u\t%u\t"
                               "%.6f\t%.6f\t"
                               "%d\n",
                               m.barcode.c_str(),
                               static_cast<unsigned long long>(m.total_reads),
                               static_cast<unsigned long long>(m.mapped_reads),
                               static_cast<unsigned long long>(m.unique_reads),
                               static_cast<unsigned long long>(m.exonic_reads),
                               static_cast<unsigned long long>(m.intronic_reads),
                               static_cast<unsigned long long>(m.intergenic_reads),
                               static_cast<unsigned long long>(m.antisense_reads),
                               m.n_umi,
                               m.n_genes,
                               m.saturation,
                               m.mito_fraction,
                               m.is_cell ? 1 : 0);
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(buf)) {
            std::fclose(fp);
            return false;
        }
        if (std::fwrite(buf, 1, static_cast<size_t>(n), fp) != static_cast<size_t>(n)) {
            std::fclose(fp);
            return false;
        }
    }

    std::fclose(fp);
    return true;
}

// ---------------------------------------------------------------------------
// Build BarcodeMetric vector from pileup outputs
//
// @param barcodes     Ordered barcode strings (same indexing as count vectors)
// @param total_umis   Total UMI count per barcode
// @param n_genes      Gene count per barcode
// @param called_cells Set of barcode indices called as cells
// ---------------------------------------------------------------------------
inline std::vector<BarcodeMetric> build_barcode_metrics(
    const std::vector<std::string>&    barcodes,
    const std::vector<uint64_t>&       total_umis,
    const std::vector<uint32_t>&       n_genes,
    const std::unordered_set<uint32_t>& called_cells) {

    std::vector<BarcodeMetric> out;
    out.reserve(barcodes.size());

    for (size_t i = 0; i < barcodes.size(); ++i) {
        BarcodeMetric m;
        m.barcode = barcodes[i];

        if (i < total_umis.size()) {
            m.n_umi         = static_cast<uint32_t>(total_umis[i]);
            // Conservatively treat every UMI as one mapped read for the stub
            m.mapped_reads  = total_umis[i];
            m.total_reads   = total_umis[i];
        }
        if (i < n_genes.size()) {
            m.n_genes = n_genes[i];
        }
        // Saturation = 1 - (unique_UMIs / total_reads); with reads=umis here
        // this gives 0; real saturation needs raw read counts from BAM.
        m.saturation = (m.total_reads > 0)
                           ? 1.0 - (static_cast<double>(m.n_umi) /
                                    static_cast<double>(m.total_reads))
                           : 0.0;

        m.is_cell = called_cells.count(static_cast<uint32_t>(i)) > 0;
        out.push_back(std::move(m));
    }
    return out;
}

}  // namespace singlet
