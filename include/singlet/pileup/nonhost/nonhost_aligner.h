// nonhost_aligner.h — Seed-chain-extend secondary alignment for EM-detected nonhost species
//
// After em_deconvolve() identifies candidate species above an abundance threshold, this
// module performs per-species secondary alignment:
//
//   1. Load species reference FASTA from
//        ${SINGLIFY_REF_BASE}/nonhost/{kingdom}_genomes/species_{id}.fna
//   2. Build a minimizer → (contig_id, position) seed index (k=21, w=11).
//   3. For each unmapped read: extract minimizers, gather seed hits per contig,
//      sort positions, find longest chain of seeds within 500 bp gap.
//   4. Read with chain ≥ 3 seeds on any contig = aligned.
//
// Output: nonhost_secondary_alignment.tsv (written by NonHostAligner::write_tsv).
//
// Wiring (NONHOST-SECONDARY-ALIGN):
//   Called from run_nonhost_em_screening() in singlify.cpp immediately after
//   em_deconvolve() returns.  SINGLIFY_REF_BASE must be set in the environment.
//   Missing FASTAs are skipped gracefully (logged at INFO level).
//
// Pipeline insertion point: inside run_nonhost_em_screening(), after em_deconvolve()
// CLI flag: none (automatic when --nonhost-db is provided and SINGLIFY_REF_BASE is set).

#pragma once

#include "nonhost_em.h"     // SpeciesAbundance, NonHostCategory
#include "min_sketch.h"     // extract_minimizers, canonical_hash, INVALID_KMER

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet {
namespace nonhost {

// ── SeedPos ──────────────────────────────────────────────────────────────────

// A single seed position in the reference, used during chain construction.
struct SeedPos {
    uint32_t contig_id;   // 0-indexed contig within the reference FASTA
    uint32_t position;    // 0-based position of the minimizer's anchor k-mer on that contig
};

// ── SecondaryAlignResult ─────────────────────────────────────────────────────

// Per-species output from NonHostAligner::align().
struct SecondaryAlignResult {
    uint32_t        species_id    = 0;
    NonHostCategory kingdom       = NonHostCategory::UNKNOWN;
    uint32_t        aligned_reads = 0;    // reads with ≥3 chained seeds on any contig
    uint32_t        total_reads   = 0;    // total reads queried against this species
    float           mapping_rate  = 0.0f; // aligned_reads / total_reads
    float           coverage      = 0.0f; // seed-covered ref positions / total_ref_length (approx)
};

// ── SpeciesRefIndex ──────────────────────────────────────────────────────────

// Minimizer → position index for a single reference FASTA.
// Reuses extract_minimizers() / canonical_hash() from min_sketch.h.
// Not thread-safe during build; read-only after build_from_fasta() returns.
class SpeciesRefIndex {
public:
    static constexpr int K = 21;   // k-mer length (must match min_sketch.h defaults)
    static constexpr int W = 11;   // window size

    // Total reference sequence length in bases (sum of all contig lengths).
    uint64_t total_ref_length() const { return total_ref_length_; }

    // Number of contigs loaded.
    size_t n_contigs() const { return contig_lengths_.size(); }

    // Build index from a FASTA file.
    // Each FASTA record becomes one contig (numbered 0, 1, 2, …).
    // Returns true on success; false if the file cannot be opened.
    bool build_from_fasta(const std::string& fasta_path) {
        std::ifstream f(fasta_path);
        if (!f) return false;

        seed_table_.clear();
        contig_lengths_.clear();
        total_ref_length_ = 0;

        std::string line, seq;
        uint32_t contig_id = 0;

        auto flush_contig = [&]() {
            if (seq.empty()) return;
            contig_lengths_.push_back(static_cast<uint32_t>(seq.size()));
            total_ref_length_ += seq.size();
            index_contig(seq, contig_id);
            ++contig_id;
            seq.clear();
        };

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (line[0] == '>') {
                flush_contig();
            } else {
                seq += line;
            }
        }
        flush_contig();
        return !contig_lengths_.empty();
    }

    // Build index directly from in-memory sequences (useful for unit tests).
    // Each string in seqs becomes one contig.
    void build_from_sequences(const std::vector<std::string>& seqs) {
        seed_table_.clear();
        contig_lengths_.clear();
        total_ref_length_ = 0;
        for (uint32_t i = 0; i < static_cast<uint32_t>(seqs.size()); ++i) {
            contig_lengths_.push_back(static_cast<uint32_t>(seqs[i].size()));
            total_ref_length_ += seqs[i].size();
            index_contig(seqs[i], i);
        }
    }

    // Lookup all (contig_id, position) pairs for a canonical minimizer hash.
    // Returns nullptr if the hash is not present in the index.
    const std::vector<SeedPos>* lookup(uint64_t kmer_hash) const {
        auto it = seed_table_.find(kmer_hash);
        if (it == seed_table_.end()) return nullptr;
        return &it->second;
    }

    const std::vector<uint32_t>& contig_lengths() const { return contig_lengths_; }

private:
    // Index all minimizers of a single contig sequence.
    void index_contig(const std::string& seq, uint32_t contig_id) {
        const int n_kmers = static_cast<int>(seq.size()) - K + 1;
        if (n_kmers < W) return;

        // Pre-compute canonical hashes for every k-mer position.
        std::vector<uint64_t> h(n_kmers);
        for (int i = 0; i < n_kmers; ++i)
            h[i] = canonical_hash(seq.data() + i, K);

        // Slide window of size W: store the position of the minimum-hash k-mer.
        // Deduplicate: only insert when the minimizer changes (same dedup as extract_minimizers).
        uint64_t prev_min  = INVALID_KMER;
        for (int i = 0; i + W <= n_kmers; ++i) {
            uint64_t win_min  = INVALID_KMER;
            int      best_pos = -1;
            for (int j = i; j < i + W; ++j) {
                if (h[j] < win_min) { win_min = h[j]; best_pos = j; }
            }
            if (win_min == INVALID_KMER) continue;           // window all-N
            if (win_min == prev_min)     continue;           // same as last window
            seed_table_[win_min].push_back({contig_id, static_cast<uint32_t>(best_pos)});
            prev_min = win_min;
        }
    }

    std::unordered_map<uint64_t, std::vector<SeedPos>> seed_table_;
    std::vector<uint32_t>                              contig_lengths_;
    uint64_t                                           total_ref_length_ = 0;
};

// ── Chain helpers ─────────────────────────────────────────────────────────────

// Return the length of the longest chain in a sorted list of reference positions,
// where consecutive seeds are no more than max_gap bases apart.
// Chain length = number of seeds in the chain, not the span in bases.
inline int longest_chain(const std::vector<uint32_t>& sorted_positions,
                          uint32_t max_gap = 500) {
    if (sorted_positions.empty()) return 0;
    int best = 1, run = 1;
    for (size_t i = 1; i < sorted_positions.size(); ++i) {
        if (sorted_positions[i] - sorted_positions[i - 1] <= max_gap) {
            ++run;
            if (run > best) best = run;
        } else {
            run = 1;
        }
    }
    return best;
}

// ── NonHostAligner ────────────────────────────────────────────────────────────

// Performs secondary alignment of unmapped reads against EM-detected species
// whose relative_abundance ≥ abundance_threshold.
//
// FASTA resolution:
//   fasta_path = ref_base + "/nonhost/" + kingdom_dir(kingdom)
//              + "/species_" + to_string(species_id) + ".fna"
// where kingdom_dir is "viral_genomes", "bacterial_genomes", or "fungal_genomes".
// Missing FASTAs are skipped with a warning (non-fatal).
class NonHostAligner {
public:
    static constexpr float    DEFAULT_ABUNDANCE_THRESHOLD = 0.001f;  // 0.1%
    static constexpr int      MIN_CHAIN_SEEDS             = 3;
    static constexpr uint32_t MAX_GAP_BP                  = 500;

    // ref_base: path to the singlify reference directory (SINGLIFY_REF_BASE).
    explicit NonHostAligner(const std::string& ref_base,
                            float abundance_threshold = DEFAULT_ABUNDANCE_THRESHOLD)
        : ref_base_(ref_base), abundance_threshold_(abundance_threshold) {}

    // Align reads against each EM-detected species above the abundance threshold.
    //
    // reads          : unmapped read sequences (same batch passed to EM)
    // em_abundances  : output of em_deconvolve(), sorted descending by relative_abundance
    //
    // Returns one SecondaryAlignResult per species whose FASTA was found on disk.
    // Species whose FASTA does not exist are silently skipped (warning logged).
    std::vector<SecondaryAlignResult> align(
            const std::vector<std::string>&      reads,
            const std::vector<SpeciesAbundance>& em_abundances) const {

        std::vector<SecondaryAlignResult> results;
        const uint32_t n_reads = static_cast<uint32_t>(reads.size());

        for (const auto& sa : em_abundances) {
            if (sa.relative_abundance < abundance_threshold_) continue;

            // Resolve FASTA path
            const std::string fna_path = fasta_path_for(sa.species_id, sa.kingdom);

            SpeciesRefIndex idx;
            if (!idx.build_from_fasta(fna_path)) {
                std::cerr << "[nonhost-aligner] SKIP: FASTA not found: " << fna_path << "\n";
                continue;
            }

            SecondaryAlignResult res;
            res.species_id  = sa.species_id;
            res.kingdom     = sa.kingdom;
            res.total_reads = n_reads;

            uint64_t seed_hits_total = 0;   // accumulator for coverage approximation

            for (const auto& read : reads) {
                auto mins = extract_minimizers(read, SpeciesRefIndex::K, SpeciesRefIndex::W);
                if (mins.empty()) continue;

                // Collect reference positions hit by this read, grouped by contig.
                std::unordered_map<uint32_t, std::vector<uint32_t>> contig_hits;
                for (uint64_t mhash : mins) {
                    const auto* positions = idx.lookup(mhash);
                    if (!positions) continue;
                    for (const SeedPos& sp : *positions) {
                        contig_hits[sp.contig_id].push_back(sp.position);
                        ++seed_hits_total;
                    }
                }

                // A read is aligned if any contig has a chain of ≥ MIN_CHAIN_SEEDS seeds
                // with consecutive seeds no more than MAX_GAP_BP apart.
                bool aligned = false;
                for (auto& [cid, pos_list] : contig_hits) {
                    std::sort(pos_list.begin(), pos_list.end());
                    if (longest_chain(pos_list, MAX_GAP_BP) >= MIN_CHAIN_SEEDS) {
                        aligned = true;
                        break;
                    }
                }
                if (aligned) ++res.aligned_reads;
            }

            if (n_reads > 0)
                res.mapping_rate = static_cast<float>(res.aligned_reads) /
                                   static_cast<float>(n_reads);

            const uint64_t ref_len = idx.total_ref_length();
            if (ref_len > 0) {
                // Approximate coverage: cap seed hits at reference length to avoid >1.
                uint64_t capped = std::min(seed_hits_total, ref_len);
                res.coverage = static_cast<float>(capped) / static_cast<float>(ref_len);
            }

            results.push_back(res);
        }
        return results;
    }

    // Write nonhost_secondary_alignment.tsv to out_prefix.
    // Call after align(); returns quickly if results is empty.
    static void write_tsv(const std::vector<SecondaryAlignResult>& results,
                          const std::string& out_prefix) {
        const std::string path = out_prefix + "/nonhost_secondary_alignment.tsv";
        std::ofstream tsv(path);
        if (!tsv.is_open()) {
            std::cerr << "[nonhost-aligner] WARNING: cannot open for write: " << path << "\n";
            return;
        }
        tsv << "kingdom\tspecies_id\taligned_reads\ttotal_reads\tmapping_rate\tcoverage\n";
        for (const auto& r : results) {
            const char* kname = "UNKNOWN";
            switch (r.kingdom) {
                case NonHostCategory::VIRAL:     kname = "VIRAL";     break;
                case NonHostCategory::BACTERIAL: kname = "BACTERIAL"; break;
                case NonHostCategory::FUNGAL:    kname = "FUNGAL";    break;
                default: break;
            }
            tsv << kname << "\t" << r.species_id << "\t"
                << r.aligned_reads << "\t" << r.total_reads << "\t"
                << std::fixed << std::setprecision(6) << r.mapping_rate << "\t"
                << r.coverage << "\n";
        }
        std::cerr << "[nonhost-aligner] Secondary alignment: " << results.size()
                  << " species → " << path << "\n";
    }

private:
    std::string ref_base_;
    float       abundance_threshold_;

    static const char* kingdom_dir(NonHostCategory k) noexcept {
        switch (k) {
            case NonHostCategory::VIRAL:     return "viral_genomes";
            case NonHostCategory::BACTERIAL: return "bacterial_genomes";
            case NonHostCategory::FUNGAL:    return "fungal_genomes";
            default:                          return "unknown_genomes";
        }
    }

    std::string fasta_path_for(uint32_t species_id, NonHostCategory k) const {
        return ref_base_ + "/nonhost/" + kingdom_dir(k)
             + "/species_" + std::to_string(species_id) + ".fna";
    }
};

} // namespace nonhost
} // namespace singlet
