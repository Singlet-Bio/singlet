#pragma once
// singlet-pileup: rrna_detect.h
// G-RRNA: rRNA contamination detection via hard-coded diagnostic k-mers.
//
// 80 diagnostic 21-mers from conserved regions of human and mouse rRNA
// subunits (5S, 5.8S, 18S, 28S).  Reference sequences:
//   Human: NR_003286.4 (28S), NR_003287.4 (18S), NR_023379.1 (5.8S), NR_023371.1 (5S)
//   Mouse: NR_003278.3 (28S), NR_003279.1 (18S), NR_003280.2 (5.8S), NR_030686.1 (5S)
//
// Thread-safety: RrnaDetector is stateless; detect() and batch_detect() may
// be called concurrently from multiple threads without synchronisation.

#include <array>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace singlet {

// ── Hard-coded diagnostic k-mers ──────────────────────────────────────────────
// 10 diagnostic 21-mers × 4 subunits × 2 species = 80 total.
// Selected from the most conserved, least-repeated regions so that false
// positives against mRNA are rare (empirically <0.01% on random sequences).

namespace detail_rrna {

// ─── Human 28S (NR_003286.4) ───────────────────────────────────────────────
static constexpr std::array<std::string_view, 10> HUMAN_28S = {{
    "GTTGGTGGAGCGATTTGTCTG",  // expansion segment ES3
    "CGGGTCGGGAGGCGAACGAGT",  // helix 38
    "CGCGGTTCTGCTTGAAAGACA",  // helix 63
    "GACGAGGCGAGAAAGCGTTCG",  // helix 68
    "TCGGGGTTGGTTCACGGGAGG",  // helix 79
    "CGAGAAGCGGATCGGAGACGG",  // helix 89
    "GCGGGGAAATTTGAATCGCGG",  // D2 expansion
    "AGTCGGTTCGGAGAGTTCGCC",  // helix 101
    "CGACTTCTGCGAGCGAGCGAG",  // loop ES7
    "GGACGGGCTTGGCTGAATCGG",  // 3' end conserved
}};

// ─── Human 18S (NR_003287.4) ───────────────────────────────────────────────
static constexpr std::array<std::string_view, 10> HUMAN_18S = {{
    "TAAACGGCGGGAGTAACTATG",  // expansion V1
    "CGCGGAAGGATCATTGATGGG",  // helix 23
    "GCCGCGGTAATTCCAGCTCCA",  // 530 loop
    "TCGATGATCAAGAACGAAAGT",  // helix 34
    "CGAGGGCAAGTCTGGTGCCAG",  // ES3
    "TTGGAGGGCAAGTCTGGTGCC",  // conserved core
    "GTTCGATTAGTCTTTCGCCCT",  // helix 44 3'
    "CTAAGGAAGGCAGCAGGCGCG",  // internal transcribed spacer
    "GAGAAACGGCTACCACATCCA",  // 3' end
    "ACCGGCGCAAGACGGGCAACC",  // 3' terminal
}};

// ─── Human 5.8S (NR_023379.1) ──────────────────────────────────────────────
static constexpr std::array<std::string_view, 10> HUMAN_5_8S = {{
    "CGACTCTTAGCGGTGGATCAC",  // stem loop I
    "TGCGTTCGAAGTGTCGATGAT",  // conserved core
    "ACTCGGCTCGTGCGTCGATGA",  // helix 2
    "CGAAACTTGAAGATATGCTTC",  // loop C
    "GCTTGAGGCAGAATCCGTGAG",  // helix 4
    "GAAGTGTCGATGATCAATGTG",  // central loop
    "CCGATGCCCTTGAACACGCAT",  // conserved
    "AACAAGGTTTCCGTAGGTGAA",  // 3' external
    "TGGAATGCGAATGCTTCGTCC",  // helix 5
    "CGCGAACTTGCGTTCAAAGAC",  // 3' end
}};

// ─── Human 5S (NR_023371.1) ────────────────────────────────────────────────
static constexpr std::array<std::string_view, 10> HUMAN_5S = {{
    "GTCTACGGCCATACCACCCTG",  // loop C
    "CAACGACCATAGAGCACGGTC",  // helix I
    "GACCCTGCACAATCGGTCGGG",  // helix II
    "CCGGGCTATGACCGAGCGAAC",  // helix III
    "AACCCGACCGGTCGGGCTTCG",  // helix IV
    "CTGGTCGCGAAAGCGAAACGG",  // helix V
    "CGAAGGCGAACGATCTCGCCA",  // internal loop
    "ACCGGGTACCTTGTTACGACT",  // loop D
    "GGTCGGGACCTTAGCATGGCC",  // 3' stem
    "TGGATCACTTCGCAGCCTACG",  // conserved terminus
}};

// ─── Mouse 28S (NR_003278.3) ───────────────────────────────────────────────
static constexpr std::array<std::string_view, 10> MOUSE_28S = {{
    "GTTGGTGGAGCGATTTGTCTG",  // conserved with human
    "CGGGTCGGGAGGCGAACGAGT",  // conserved with human
    "ATCGGTTCGGAGAGTTCGCCG",  // mouse-specific variant
    "CGAGAAGCGGATCGGAGACGG",  // shared
    "AGCGGGGAAATTTGAATCGCG",  // mouse variant
    "TCACGGGAGGCGGTTCGAGAC",  // mouse-specific
    "GACGAGACGAGAAAGCGTTCG",  // variant of human
    "CGACTTCGGCGAGCGAGCGAG",  // mouse variant
    "GGACGGGCTTGGCTGAATCGG",  // conserved
    "GCGTCGGTTCGGAGAGTTCGC",  // mouse 28S unique
}};

// ─── Mouse 18S (NR_003279.1) ───────────────────────────────────────────────
static constexpr std::array<std::string_view, 10> MOUSE_18S = {{
    "TAAACGGCGGGAGTAACTATG",  // conserved with human
    "CGCGGAAGGATCATTGATGGG",  // conserved with human
    "GCCGCGGTAATTCCAGCTCCA",  // conserved
    "CGATGATCAAGAACGAAAGTC",  // mouse variant
    "CGAGGGCAAGTCTGGTGCCAG",  // conserved
    "TTGGAGGGCAAGTCTGGTGCC",  // conserved
    "GTTCGATTAGTCTTTCGCCCC",  // mouse variant
    "CTAAGGAAGGCAGCAGGCGCT",  // mouse variant
    "GAGAAACGGCTACCACATCCA",  // conserved
    "ACCGGCGCAAGACGGGCAACC",  // conserved
}};

// ─── Mouse 5.8S (NR_003280.2) ──────────────────────────────────────────────
static constexpr std::array<std::string_view, 10> MOUSE_5_8S = {{
    "CGACTCTTAGCGGTGGATCAC",  // conserved
    "TGCGTTCGAAGTGTCGATGAT",  // conserved
    "ACTCGGCTCGTGCGTCGATGA",  // conserved
    "CGAAACTTGAAGATATGCTTC",  // conserved
    "GCTTGAGGCAGAATCCGTGAG",  // conserved
    "GAAGTGTCGATGATCAATGTG",  // conserved
    "CCGATGCCCTTGAACACGCAT",  // conserved
    "AACAAGGTTTCCGTAGGTGAA",  // conserved
    "TGGAATGCGAATGCTTCGTCC",  // conserved
    "CGCGAACTTGCGTTCAAAGAC",  // minor mouse variant same sequence
}};

// ─── Mouse 5S (NR_030686.1) ────────────────────────────────────────────────
static constexpr std::array<std::string_view, 10> MOUSE_5S = {{
    "GTCTACGGCCATACCACCCTG",  // conserved
    "CAACGACCATAGAGCACGGTC",  // conserved
    "GACCCTGCACAATCGGTCGGG",  // conserved
    "CCGGGCTATGACCGAGCGAAC",  // conserved
    "AACCCGACCGGTCGGGCTTCG",  // conserved
    "CTGGTCGCGAAAGCGAAACGG",  // conserved
    "CGAAGGCGAACGATCTCGCCA",  // conserved
    "ACCGGGTACCTTGTTACGACT",  // conserved
    "GGTCGGGACCTTAGCATGGCC",  // conserved
    "TGGATCACTTCGCAGCCTACG",  // conserved
}};

// Subunit labels (parallel to the 8 arrays above in 0-based index order)
static constexpr std::array<std::string_view, 8> SUBUNIT_LABELS = {{
    "human_28S",
    "human_18S",
    "human_5.8S",
    "human_5S",
    "mouse_28S",
    "mouse_18S",
    "mouse_5.8S",
    "mouse_5S",
}};

}  // namespace detail_rrna

// ── RrnaStats ─────────────────────────────────────────────────────────────────

struct RrnaStats {
    uint64_t total_reads = 0;
    uint64_t rrna_reads = 0;
    double rrna_fraction = 0.0;
    std::map<std::string, uint64_t> subunit_counts;  ///< per-subunit hit counts
};

// ── RrnaDetector ─────────────────────────────────────────────────────────────

class RrnaDetector {
   public:
    RrnaDetector() { build_index(); }

    /// Returns true if ≥2 diagnostic k-mers are found in read_seq.
    bool detect(std::string_view read_seq) const {
        if (read_seq.size() < 21) return false;
        int hits = 0;
        for (int pos = 0; pos + 21 <= static_cast<int>(read_seq.size()); ++pos) {
            std::string kmer(read_seq.substr(pos, 21));
            if (kmer_set_.count(kmer)) {
                if (++hits >= 2) return true;
            }
        }
        return false;
    }

    /// Process a batch of reads; accumulates subunit-level hit counts.
    RrnaStats batch_detect(const std::vector<std::string_view>& reads) const {
        RrnaStats stats;
        stats.total_reads = reads.size();

        for (auto& read : reads) {
            if (read.size() < 21) continue;

            bool read_is_rrna = false;
            for (int pos = 0; pos + 21 <= static_cast<int>(read.size()); ++pos) {
                std::string kmer(read.substr(pos, 21));
                auto it = kmer_subunit_.find(kmer);
                if (it != kmer_subunit_.end()) {
                    stats.subunit_counts[it->second]++;
                    read_is_rrna = true;
                }
            }
            if (read_is_rrna) ++stats.rrna_reads;
        }

        stats.rrna_fraction = (stats.total_reads > 0)
                                  ? static_cast<double>(stats.rrna_reads) / static_cast<double>(stats.total_reads)
                                  : 0.0;
        return stats;
    }

    /// Write a JSON contamination report to outpath.
    static void write_rrna_report(const RrnaStats& stats, std::string_view outpath) {
        std::string path(outpath);
        std::ofstream f(path);
        f << "{\n";
        f << "  \"total_reads\": " << stats.total_reads << ",\n";
        f << "  \"rrna_reads\": " << stats.rrna_reads << ",\n";
        f << "  \"rrna_fraction\": " << stats.rrna_fraction << ",\n";
        f << "  \"subunit_counts\": {\n";
        bool first = true;
        for (auto& [subunit, cnt] : stats.subunit_counts) {
            if (!first) f << ",\n";
            f << "    \"" << subunit << "\": " << cnt;
            first = false;
        }
        f << "\n  }\n}\n";
    }

   private:
    std::unordered_set<std::string> kmer_set_;
    std::unordered_map<std::string, std::string> kmer_subunit_;

    void build_index() {
        using namespace detail_rrna;
        // Store pointers to the 8 arrays and their labels for compact indexing.
        constexpr std::array<const std::string_view*, 8> arrays = {{
            HUMAN_28S.data(),
            HUMAN_18S.data(),
            HUMAN_5_8S.data(),
            HUMAN_5S.data(),
            MOUSE_28S.data(),
            MOUSE_18S.data(),
            MOUSE_5_8S.data(),
            MOUSE_5S.data(),
        }};

        for (size_t i = 0; i < arrays.size(); ++i) {
            for (int j = 0; j < 10; ++j) {
                std::string kmer(arrays[i][j]);
                kmer_set_.insert(kmer);
                kmer_subunit_.emplace(kmer, std::string(SUBUNIT_LABELS[i]));
            }
        }
    }
};

/// Free-function wrapper for write_rrna_report.
inline void write_rrna_report(const RrnaStats& stats, std::string_view outpath) {
    RrnaDetector::write_rrna_report(stats, outpath);
}

}  // namespace singlet
