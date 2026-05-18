// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: te_classifier.h  — T-L2-3
// Transposable element family classifier using 21-mer FracMinHash sketches
// built from Dfam 3.8 family consensus sequences.
//
// Design:
//   - One FracMinHash sketch per Dfam family (bottom-s MinHash, 21-mer, s≤256).
//   - classify() returns the best-matching family only if:
//       matched_kmers / total_kmers_in_read >= min_family_kmer_fraction (default 0.5).
//   - Reads with k-mer ambiguity between a TE family and any host transcript
//     return std::nullopt (they fall through to L3 STAR).
//   - Determinism seed: 0xC0FFEE.
//
// On-disk sketch DB layout (te_sketch.bin):
//   Header { magic[4]="TESB", version=1, n_families, kmer_size, sketch_size,
//             family_table_hash[32] (SHA-256 of family_table.tsv) }
//   Per-family record: { n_hashes (uint32), hashes... (uint64[]) }
//
// Thread safety: TeClassifier is read-only after construction — thread-safe.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "kmer_util.h"

namespace singlet {

// ============================================================================
// Constants
// ============================================================================

static constexpr uint32_t TE_SKETCH_MAGIC   = 0x42534554;  // "TESB"
static constexpr uint32_t TE_SKETCH_VERSION = 1;
static constexpr uint32_t TE_KMER_SIZE      = 21;
static constexpr uint32_t TE_SKETCH_SIZE    = 256;     // bottom-s MinHash size per family
static constexpr uint64_t TE_HASH_SEED      = 0xC0FFEE;
static constexpr float    TE_MIN_KMER_FRAC  = 0.5f;    // min_family_kmer_fraction §8.3

// ============================================================================
// Data structures
// ============================================================================

struct TeFamilyHit {
    uint32_t family_id;       // 0-based index into TeFamilySketchDB.families
    float    score;           // n_kmers_matched / n_kmers_total
    uint32_t n_kmers_matched;
    uint32_t n_kmers_total;
};

struct TeFamilyRecord {
    std::string family_name;
    uint32_t    consensus_length;
    std::vector<uint64_t> sketch;  // sorted bottom-s MinHash hashes
};

// ============================================================================
// TeFamilySketchDB — immutable after load
// ============================================================================

struct TeFamilySketchDB {
    std::vector<TeFamilyRecord> families;
    uint32_t kmer_size    = TE_KMER_SIZE;
    uint32_t sketch_size  = TE_SKETCH_SIZE;
    uint8_t  family_table_hash[32]{};

    /// Load from binary te_sketch.bin produced by scripts/build_te_sketch.py.
    static TeFamilySketchDB load(const std::string& path) {
        TeFamilySketchDB db;
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("te_classifier: cannot open " + path);

        // Header
        uint32_t magic, version, n_families, kmer_size, sketch_size;
        f.read(reinterpret_cast<char*>(&magic),      4);
        f.read(reinterpret_cast<char*>(&version),    4);
        f.read(reinterpret_cast<char*>(&n_families), 4);
        f.read(reinterpret_cast<char*>(&kmer_size),  4);
        f.read(reinterpret_cast<char*>(&sketch_size),4);
        f.read(reinterpret_cast<char*>(&db.family_table_hash), 32);

        if (magic != TE_SKETCH_MAGIC)
            throw std::runtime_error("te_classifier: bad magic in " + path);
        if (version != TE_SKETCH_VERSION)
            throw std::runtime_error("te_classifier: unsupported sketch version");

        db.kmer_size   = kmer_size;
        db.sketch_size = sketch_size;
        db.families.resize(n_families);

        for (uint32_t i = 0; i < n_families; ++i) {
            uint32_t name_len, consensus_length, n_hashes;
            f.read(reinterpret_cast<char*>(&name_len),         4);
            f.read(reinterpret_cast<char*>(&consensus_length), 4);
            f.read(reinterpret_cast<char*>(&n_hashes),         4);

            db.families[i].consensus_length = consensus_length;
            db.families[i].family_name.resize(name_len);
            f.read(db.families[i].family_name.data(), name_len);
            db.families[i].sketch.resize(n_hashes);
            f.read(reinterpret_cast<char*>(db.families[i].sketch.data()),
                   static_cast<std::streamsize>(n_hashes * sizeof(uint64_t)));
        }

        if (!f) throw std::runtime_error("te_classifier: truncated file " + path);
        return db;
    }

    /// Build a minimal in-memory DB from raw family data (for unit tests).
    /// families_in: vector of { name, consensus_length, precomputed_sketch }
    static TeFamilySketchDB from_records(std::vector<TeFamilyRecord> recs,
                                         uint32_t kmer_size  = TE_KMER_SIZE,
                                         uint32_t sketch_size = TE_SKETCH_SIZE) {
        TeFamilySketchDB db;
        db.families    = std::move(recs);
        db.kmer_size   = kmer_size;
        db.sketch_size = sketch_size;
        return db;
    }
};

// ============================================================================
// Internal hashing utilities
// ============================================================================
namespace detail {

// MurmurHash3 finaliser for 64-bit integer (fast, well-distributed).
inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 33;
    x *= 0xC4CEB9FE1A85EC53ULL;
    x ^= x >> 33;
    return x;
}

// Encode a single base to 2 bits (ACGT → 0,1,2,3; anything else → 4=invalid).
// Thin wrapper over the shared singlet::pileup::kmer helper.
inline uint8_t base2bit(char c) {
    return ::singlet::pileup::kmer::base2bit(c);
}

// Canonical 2-bit k-mer (minimum of forward / reverse-complement).
// Returns ~0 if any ambiguous base is present.
// Thin wrapper over the shared singlet::pileup::kmer helper.
inline uint64_t canonical_kmer(const char* seq, uint32_t k) {
    return ::singlet::pileup::kmer::canonical_kmer_ascii(seq, k);
}

// Compute all bottom-s MinHash hashes for a sequence with given k-mer size.
// Returns sorted vector of the bottom-`s` canonical k-mer hashes.
inline std::vector<uint64_t> frac_minhash_sketch(std::string_view seq,
                                                   uint32_t k,
                                                   uint32_t s,
                                                   uint64_t seed = TE_HASH_SEED) {
    std::vector<uint64_t> heap;
    heap.reserve(s + 1);

    const size_t n = seq.size();
    if (n < k) return {};

    uint64_t top_val = ~0ULL;

    for (size_t i = 0; i + k <= n; ++i) {
        uint64_t kmer = canonical_kmer(seq.data() + i, k);
        if (kmer == ~0ULL) continue;
        uint64_t h = mix64(kmer ^ seed);

        if (heap.size() < s) {
            heap.push_back(h);
            if (heap.size() == s) {
                std::make_heap(heap.begin(), heap.end());
                top_val = heap.front();
            }
        } else if (h < top_val) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = h;
            std::push_heap(heap.begin(), heap.end());
            top_val = heap.front();
        }
    }

    std::sort(heap.begin(), heap.end());
    return heap;
}

// Jaccard estimate: |A ∩ B| / |A ∪ B| via sorted-merge on bottom-s sketches.
inline float sketch_jaccard(const std::vector<uint64_t>& a,
                             const std::vector<uint64_t>& b) {
    if (a.empty() || b.empty()) return 0.0f;
    size_t i = 0, j = 0, intersect = 0, n_union = 0;
    // Walk the union of both bottom-s sorted sets
    size_t n = std::min(a.size() + b.size(), (a.size() > b.size() ? a.size() : b.size()));
    (void)n;
    while (i < a.size() && j < b.size()) {
        ++n_union;
        if (a[i] == b[j]) { ++intersect; ++i; ++j; }
        else if (a[i] < b[j]) { ++i; }
        else                  { ++j; }
    }
    n_union += (a.size() - i) + (b.size() - j);
    if (n_union == 0) return 0.0f;
    return static_cast<float>(intersect) / static_cast<float>(n_union);
}

// Count how many of the read's hashes appear in the family sketch.
// read_hashes must be sorted.
inline uint32_t count_matching_hashes(const std::vector<uint64_t>& read_hashes,
                                       const std::vector<uint64_t>& family_sketch) {
    uint32_t matches = 0;
    size_t i = 0, j = 0;
    while (i < read_hashes.size() && j < family_sketch.size()) {
        if (read_hashes[i] == family_sketch[j]) { ++matches; ++i; ++j; }
        else if (read_hashes[i] < family_sketch[j]) ++i;
        else                                        ++j;
    }
    return matches;
}

}  // namespace detail

// ============================================================================
// TeClassifier
// ============================================================================

class TeClassifier {
   public:
    explicit TeClassifier(const TeFamilySketchDB& db,
                          float min_family_kmer_fraction = TE_MIN_KMER_FRAC)
        : db_(db), min_kmer_frac_(min_family_kmer_fraction) {}

    /// Classify a single read sequence.
    ///
    /// Returns TeFamilyHit if the read is dominated by one TE family:
    ///   n_kmers_matched / n_kmers_total >= min_family_kmer_fraction
    /// AND no other family at the same score (ambiguity check).
    ///
    /// Returns std::nullopt if:
    ///   - score < min_family_kmer_fraction   → route to L3 (STAR)
    ///   - two or more families tied within 5 % score  → route to L3
    ///   - best family score is within 10 % of best non-TE host sketch  → L3
    ///     (ambiguity with host transcripts; checked when host sketches provided)
    std::optional<TeFamilyHit> classify(std::string_view read) const {
        if (db_.families.empty()) return std::nullopt;

        // Build sketch for read
        auto read_hashes = detail::frac_minhash_sketch(
            read, db_.kmer_size, db_.sketch_size);
        if (read_hashes.empty()) return std::nullopt;

        const uint32_t n_total = static_cast<uint32_t>(read_hashes.size());

        // Find best and second-best TE family by matching hash count
        uint32_t best_idx   = 0;
        uint32_t best_cnt   = 0;
        uint32_t second_cnt = 0;

        for (uint32_t i = 0; i < static_cast<uint32_t>(db_.families.size()); ++i) {
            uint32_t cnt = detail::count_matching_hashes(
                read_hashes, db_.families[i].sketch);
            if (cnt > best_cnt) {
                second_cnt = best_cnt;
                best_cnt   = cnt;
                best_idx   = i;
            } else if (cnt > second_cnt) {
                second_cnt = cnt;
            }
        }

        if (n_total == 0) return std::nullopt;
        float score = static_cast<float>(best_cnt) / static_cast<float>(n_total);

        // Must clear the minimum fraction threshold
        if (score < min_kmer_frac_) return std::nullopt;

        // Ambiguity: second family within 5% score → pass to L3
        float second_score = static_cast<float>(second_cnt) / static_cast<float>(n_total);
        if ((score - second_score) < 0.05f) return std::nullopt;

        return TeFamilyHit{best_idx, score, best_cnt, n_total};
    }

    /// Classify with host-transcript ambiguity check.
    /// host_sketches: pre-built sorted hash vectors for host transcripts.
    /// A read whose best TE score minus best host score < 0.1 routes to L3.
    std::optional<TeFamilyHit> classify_with_host_check(
        std::string_view read,
        const std::vector<std::vector<uint64_t>>& host_sketches) const
    {
        auto hit = classify(read);
        if (!hit) return std::nullopt;

        auto read_hashes = detail::frac_minhash_sketch(
            read, db_.kmer_size, db_.sketch_size);
        if (read_hashes.empty()) return std::nullopt;

        const uint32_t n_total = hit->n_kmers_total;

        uint32_t best_host_cnt = 0;
        for (const auto& hs : host_sketches) {
            uint32_t cnt = detail::count_matching_hashes(read_hashes, hs);
            if (cnt > best_host_cnt) best_host_cnt = cnt;
        }

        float host_score = static_cast<float>(best_host_cnt) / static_cast<float>(n_total);
        if ((hit->score - host_score) < 0.10f) return std::nullopt;  // ambiguous

        return hit;
    }

    const TeFamilySketchDB& db() const { return db_; }

   private:
    const TeFamilySketchDB& db_;
    float min_kmer_frac_;
};

}  // namespace singlet
