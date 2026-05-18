// SPDX-License-Identifier: MIT
#pragma once
// benchmark_suite.h — Lightweight benchmark registry for singlet vs SOTA tools
// Tracks performance and accuracy results across modalities; NOT a runtime runner.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet::pileup::benchmark {

enum class Modality {
    SCRNA,
    SNRNA,
    ATAC,
    CITE_SEQ,
    VISIUM,
    SMART_SEQ2,
    BULK_RNA,
    VDJ,
    CRISPR,
    SPATIAL,
    HIC,
    METHYL,
    LONG_READ,
    CHIP_SEQ
};

struct BenchmarkResult {
    std::string tool_name;  // "singlet", "cellranger", "starsolo", etc.
    std::string tool_version;
    Modality modality;
    std::string dataset_id;  // SRR accession or dataset name
    uint64_t total_reads = 0;
    int threads = 1;

    // Timing
    double wall_seconds = 0.0;
    double user_seconds = 0.0;

    // Memory
    uint64_t peak_rss_bytes = 0;

    // Accuracy (optional)
    double concordance = -1.0;  // Pearson r vs gold standard; -1 = not measured
    uint64_t cells_called = 0;
    double unique_mapping_rate = -1.0;  // -1 = not measured

    // Metadata
    std::string hostname;
    std::string date;  // ISO 8601
    std::string notes;
};

// ── Modality name helpers ─────────────────────────────────────────────────────

inline std::string modality_name(Modality m) {
    switch (m) {
        case Modality::SCRNA:
            return "scRNA";
        case Modality::SNRNA:
            return "snRNA";
        case Modality::ATAC:
            return "ATAC";
        case Modality::CITE_SEQ:
            return "CITE-seq";
        case Modality::VISIUM:
            return "Visium";
        case Modality::SMART_SEQ2:
            return "Smart-seq2";
        case Modality::BULK_RNA:
            return "bulk_RNA";
        case Modality::VDJ:
            return "VDJ";
        case Modality::CRISPR:
            return "CRISPR";
        case Modality::SPATIAL:
            return "spatial";
        case Modality::HIC:
            return "HiC";
        case Modality::METHYL:
            return "methyl";
        case Modality::LONG_READ:
            return "long_read";
        case Modality::CHIP_SEQ:
            return "ChIP-seq";
        default:
            return "unknown";
    }
}

inline Modality parse_modality(const std::string& s) {
    if (s == "scRNA") return Modality::SCRNA;
    if (s == "snRNA") return Modality::SNRNA;
    if (s == "ATAC") return Modality::ATAC;
    if (s == "CITE-seq") return Modality::CITE_SEQ;
    if (s == "Visium") return Modality::VISIUM;
    if (s == "Smart-seq2") return Modality::SMART_SEQ2;
    if (s == "bulk_RNA") return Modality::BULK_RNA;
    if (s == "VDJ") return Modality::VDJ;
    if (s == "CRISPR") return Modality::CRISPR;
    if (s == "spatial") return Modality::SPATIAL;
    if (s == "HiC") return Modality::HIC;
    if (s == "methyl") return Modality::METHYL;
    if (s == "long_read") return Modality::LONG_READ;
    if (s == "ChIP-seq") return Modality::CHIP_SEQ;
    throw std::invalid_argument("Unknown modality: " + s);
}

// ── Benchmark Registry ────────────────────────────────────────────────────────

struct BenchmarkRegistry {
    std::vector<BenchmarkResult> results;

    void add(const BenchmarkResult& r) { results.push_back(r); }

    std::vector<BenchmarkResult> by_modality(Modality m) const {
        std::vector<BenchmarkResult> out;
        for (const auto& r : results)
            if (r.modality == m) out.push_back(r);
        return out;
    }

    std::vector<BenchmarkResult> by_tool(const std::string& tool) const {
        std::vector<BenchmarkResult> out;
        for (const auto& r : results)
            if (r.tool_name == tool) out.push_back(r);
        return out;
    }

    std::vector<BenchmarkResult> by_dataset(const std::string& dataset) const {
        std::vector<BenchmarkResult> out;
        for (const auto& r : results)
            if (r.dataset_id == dataset) out.push_back(r);
        return out;
    }

    // Returns competitor_wall / singlet_wall (>1 means singlet is faster).
    // Returns -1.0 if either result is missing.
    double speedup(const std::string& competitor, const std::string& dataset) const {
        double singlet_wall = -1.0, competitor_wall = -1.0;
        for (const auto& r : results) {
            if (r.dataset_id != dataset) continue;
            if (r.tool_name == "singlet") singlet_wall = r.wall_seconds;
            if (r.tool_name == competitor) competitor_wall = r.wall_seconds;
        }
        if (singlet_wall <= 0.0 || competitor_wall <= 0.0) return -1.0;
        return competitor_wall / singlet_wall;
    }

    // Returns true iff singlet is faster AND has concordance >= competitor on this dataset.
    // Returns false if data is missing or singlet is slower.
    bool at_pareto_frontier(const std::string& competitor, const std::string& dataset) const {
        double singlet_wall = -1.0, competitor_wall = -1.0;
        double singlet_conc = -1.0, competitor_conc = -1.0;
        for (const auto& r : results) {
            if (r.dataset_id != dataset) continue;
            if (r.tool_name == "singlet") {
                singlet_wall = r.wall_seconds;
                singlet_conc = r.concordance;
            }
            if (r.tool_name == competitor) {
                competitor_wall = r.wall_seconds;
                competitor_conc = r.concordance;
            }
        }
        if (singlet_wall <= 0.0 || competitor_wall <= 0.0) return false;
        bool faster = singlet_wall < competitor_wall;
        // If concordance measurements exist, require singlet >= competitor
        bool as_accurate = true;
        if (singlet_conc >= 0.0 && competitor_conc >= 0.0)
            as_accurate = (singlet_conc >= competitor_conc);
        return faster && as_accurate;
    }
};

// ── I/O ──────────────────────────────────────────────────────────────────────

// TSV column order
static const char* const TSV_HEADER =
    "tool_name\ttool_version\tmodality\tdataset_id\ttotal_reads\tthreads"
    "\twall_seconds\tuser_seconds\tpeak_rss_bytes\tconcordance"
    "\tcells_called\tunique_mapping_rate\thostname\tdate\tnotes\n";

inline void write_benchmark_tsv(const BenchmarkRegistry& reg, const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);
    f << TSV_HEADER;
    for (const auto& r : reg.results) {
        f << r.tool_name << '\t'
          << r.tool_version << '\t'
          << modality_name(r.modality) << '\t'
          << r.dataset_id << '\t'
          << r.total_reads << '\t'
          << r.threads << '\t'
          << std::fixed << std::setprecision(3) << r.wall_seconds << '\t'
          << std::fixed << std::setprecision(3) << r.user_seconds << '\t'
          << r.peak_rss_bytes << '\t'
          << std::fixed << std::setprecision(6) << r.concordance << '\t'
          << r.cells_called << '\t'
          << std::fixed << std::setprecision(6) << r.unique_mapping_rate << '\t'
          << r.hostname << '\t'
          << r.date << '\t'
          << r.notes << '\n';
    }
}

inline BenchmarkRegistry read_benchmark_tsv(const std::string& path) {
    BenchmarkRegistry reg;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open for reading: " + path);
    std::string line;
    std::getline(f, line);  // skip header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        BenchmarkResult r;
        std::string mod_str;
        auto get = [&](auto& field) {
            std::string tok;
            std::getline(ss, tok, '\t');
            std::istringstream ts(tok);
            ts >> field;
        };
        std::getline(ss, r.tool_name, '\t');
        std::getline(ss, r.tool_version, '\t');
        std::getline(ss, mod_str, '\t');
        std::getline(ss, r.dataset_id, '\t');
        get(r.total_reads);
        get(r.threads);
        get(r.wall_seconds);
        get(r.user_seconds);
        get(r.peak_rss_bytes);
        get(r.concordance);
        get(r.cells_called);
        get(r.unique_mapping_rate);
        std::getline(ss, r.hostname, '\t');
        std::getline(ss, r.date, '\t');
        std::getline(ss, r.notes);
        r.modality = parse_modality(mod_str);
        reg.results.push_back(r);
    }
    return reg;
}

inline std::string format_benchmark_markdown(const BenchmarkRegistry& reg) {
    std::ostringstream md;
    md << "| Tool | Version | Modality | Dataset | Reads | Threads | Wall(s) | Concordance | Speedup |\n";
    md << "|------|---------|----------|---------|-------|---------|---------|-------------|--------|\n";

    // Collect unique datasets
    std::vector<std::string> datasets;
    for (const auto& r : reg.results) {
        if (std::find(datasets.begin(), datasets.end(), r.dataset_id) == datasets.end())
            datasets.push_back(r.dataset_id);
    }

    for (const auto& r : reg.results) {
        double sp = -1.0;
        if (r.tool_name != "singlet") {
            sp = reg.speedup(r.tool_name, r.dataset_id);
        }
        std::string sp_str = (sp > 0) ? (std::to_string(sp).substr(0, 4) + "x") : "—";
        std::string conc_str = (r.concordance >= 0) ? std::to_string(r.concordance).substr(0, 6) : "—";

        md << "| " << r.tool_name
           << " | " << r.tool_version
           << " | " << modality_name(r.modality)
           << " | " << r.dataset_id
           << " | " << r.total_reads
           << " | " << r.threads
           << " | " << std::fixed << std::setprecision(1) << r.wall_seconds
           << " | " << conc_str
           << " | " << sp_str
           << " |\n";
    }
    return md.str();
}

}  // namespace singlet::pileup::benchmark
