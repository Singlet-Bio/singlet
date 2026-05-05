#pragma once
// singlet-pileup: mirna.h
// G-MIRNA: miRNA quantification from small RNA-seq data.
//
// Parses miRBase GFF3 annotations and builds a k=18 exact + 1-mismatch hash
// index for rapid per-read miRNA assignment.
//
// Thread-safety: MirnaCounter::count_read() is NOT thread-safe (modifies
// counts_).  Build a per-thread counter or protect with a mutex.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace singlet {

// ── Data types ────────────────────────────────────────────────────────────────

struct MirnaEntry {
    std::string name;       ///< miRNA name  (e.g. "hsa-miR-21-5p")
    std::string accession;  ///< miRBase accession (e.g. "MIMAT0000076")
    std::string sequence;   ///< Mature miRNA DNA sequence (T-encoded)
    int mature_start;       ///< 1-based start on precursor
    int mature_end;         ///< 1-based end on precursor
};

// ── GFF3 parser ───────────────────────────────────────────────────────────────

/// Parse a miRBase GFF3 file content and return mature miRNA entries filtered
/// by species prefix (e.g. "hsa" for human, "mmu" for mouse, "" for all).
///
/// Only "miRNA" feature lines are processed; "miRNA_primary_transcript" lines
/// are skipped.  The "Name" and "Accession" attributes are extracted from column 9.
inline std::vector<MirnaEntry> parse_mirbase_gff3(std::string_view content,
                                                  std::string_view species) {
    std::vector<MirnaEntry> result;
    std::istringstream ss{std::string(content)};
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '#') continue;

        // Split into 9 tab-separated columns.
        std::vector<std::string> cols;
        cols.reserve(9);
        {
            std::string tok;
            std::istringstream ls(line);
            while (std::getline(ls, tok, '\t')) cols.push_back(tok);
        }
        if (cols.size() < 9) continue;

        // We only want mature miRNA features.
        if (cols[2] != "miRNA") continue;

        // Parse start/end (1-based GFF3 coordinates).
        int start = 0, end = 0;
        try {
            start = std::stoi(cols[3]);
            end = std::stoi(cols[4]);
        } catch (...) {
            continue;
        }

        // Extract Name and Accession from attribute column.
        std::string name, accession;
        {
            std::istringstream attrs(cols[8]);
            std::string pair;
            while (std::getline(attrs, pair, ';')) {
                auto eq = pair.find('=');
                if (eq == std::string::npos) continue;
                std::string key = pair.substr(0, eq);
                std::string val = pair.substr(eq + 1);
                if (key == "Name") name = val;
                if (key == "Accession") accession = val;
            }
        }
        if (name.empty()) continue;

        // Species filter: name must start with "<species>-".
        if (!species.empty()) {
            std::string prefix = std::string(species) + "-";
            if (name.size() < prefix.size() ||
                name.substr(0, prefix.size()) != prefix) continue;
        }

        MirnaEntry e;
        e.name = std::move(name);
        e.accession = std::move(accession);
        e.mature_start = start;
        e.mature_end = end;
        result.push_back(std::move(e));
    }
    return result;
}

// ── Small RNA detection ───────────────────────────────────────────────────────

/// Returns true if the read length distribution is characteristic of a small
/// RNA library (median read length < 35 bp).
inline bool detect_small_rna(const std::vector<int>& read_lengths, int /*n_reads*/) {
    if (read_lengths.empty()) return false;
    std::vector<int> sorted = read_lengths;
    std::sort(sorted.begin(), sorted.end());
    double median = (sorted.size() % 2 == 0)
                        ? 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2])
                        : static_cast<double>(sorted[sorted.size() / 2]);
    return median < 35.0;
}

// ── MirnaCounter ─────────────────────────────────────────────────────────────

class MirnaCounter {
   public:
    static constexpr int KMER_LEN = 18;

    /// Build exact + 1-mismatch k-mer index from mature miRNA sequences.
    void build_index(const std::vector<MirnaEntry>& mirnas) {
        mirnas_ = mirnas;
        kmer_map_.clear();

        for (size_t i = 0; i < mirnas_.size(); ++i) {
            const std::string& seq = mirnas_[i].sequence;
            if (static_cast<int>(seq.size()) < KMER_LEN) continue;

            // Exact k-mers from every position.
            for (int pos = 0; pos + KMER_LEN <= static_cast<int>(seq.size()); ++pos) {
                std::string kmer = seq.substr(pos, KMER_LEN);
                kmer_map_.emplace(kmer, static_cast<uint32_t>(i));
            }

            // 1-mismatch k-mers from the seed region (first KMER_LEN bases).
            if (static_cast<int>(seq.size()) >= KMER_LEN) {
                std::string seed = seq.substr(0, KMER_LEN);
                for (int p = 0; p < KMER_LEN; ++p) {
                    char orig = seed[p];
                    for (char alt : {'A', 'C', 'G', 'T'}) {
                        if (alt == orig) continue;
                        seed[p] = alt;
                        kmer_map_.emplace(seed, static_cast<uint32_t>(i));
                        seed[p] = orig;
                    }
                }
            }
        }
    }

    /// Match a read sequence against the index.  Returns miRNA name if found.
    std::optional<std::string> count_read(std::string_view seq) {
        if (static_cast<int>(seq.size()) < KMER_LEN) return std::nullopt;

        for (int pos = 0; pos + KMER_LEN <= static_cast<int>(seq.size()); ++pos) {
            std::string kmer(seq.substr(pos, KMER_LEN));
            auto it = kmer_map_.find(kmer);
            if (it != kmer_map_.end()) {
                const std::string& name = mirnas_[it->second].name;
                ++counts_[name];
                return name;
            }
        }
        return std::nullopt;
    }

    /// Per-miRNA read counts accumulated so far.
    const std::unordered_map<std::string, uint64_t>& get_counts() const {
        return counts_;
    }

    /// Total reads counted (sum of all miRNA counts).
    uint64_t total_counted() const {
        uint64_t tot = 0;
        for (auto& [k, v] : counts_) tot += v;
        return tot;
    }

    /// Write mirna_counts.tsv: name, accession, count, RPM.
    void write_mirna_counts(std::string_view outpath) const {
        double total = static_cast<double>(total_counted());
        std::string path(outpath);
        std::ofstream f(path);
        f << "name\taccession\tcount\tRPM\n";
        for (const auto& e : mirnas_) {
            auto it = counts_.find(e.name);
            uint64_t cnt = (it != counts_.end()) ? it->second : 0;
            double rpm = (total > 0) ? cnt / total * 1e6 : 0.0;
            f << e.name << '\t' << e.accession << '\t' << cnt << '\t'
              << static_cast<long long>(std::round(rpm)) << '\n';
        }
    }

   private:
    std::vector<MirnaEntry> mirnas_;
    std::unordered_map<std::string, uint32_t> kmer_map_;  ///< kmer → mirna index
    std::unordered_map<std::string, uint64_t> counts_;
};

/// Free-function wrapper matching the documented API.
inline void write_mirna_counts(const MirnaCounter& counter,
                               std::string_view outpath) {
    counter.write_mirna_counts(outpath);
}

}  // namespace singlet
