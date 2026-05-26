#pragma once
// singlet-pileup: te_kmer_index.h — T-L2-6: Dense k-mer TE family index
//
// Replaces the FracMinHash-based te_classifier.h with an exact k-mer lookup.
//
// Design:
//   - Pre-built binary index mapping family-unique 22-mers → family_id.
//   - Only k-mers unique to ONE TE family are stored (multi-family discarded).
//   - Bloom filter pre-check rejects >99.5% of non-matching k-mers.
//   - Dense stride-1 scan counts per-family hits across entire read.
//   - Classification: plurality vote with minimum hit count.
//
// Binary format (.teki):
//   [4B magic "TEKI"] [4B version=1] [4B k] [4B n_families] [4B n_kmers] [4B reserved]
//   [n_families × NUL-terminated family names]
//   [n_kmers × 12B entries: (8B kmer_2bit, 4B family_id)]  ← sorted by kmer_2bit
//
// Thread safety: TeKmerIndex is read-only after construction — thread-safe.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace singlet {

// ── Constants ────────────────────────────────────────────────────────────────

static constexpr uint32_t TEKI_MAGIC   = 0x494B4554;  // "TEKI"
static constexpr uint32_t TEKI_VERSION = 1;
static constexpr uint32_t TEKI_KMER_K  = 22;
static constexpr uint32_t TEKI_MIN_HITS = 3;   // min k-mer hits to classify

// ── Data structures ──────────────────────────────────────────────────────────

#pragma pack(push, 1)
struct TeKmerEntry {
    uint64_t kmer_2bit;   // 2-bit encoded canonical k-mer (44 bits for k=22)
    uint32_t family_id;   // index into family_names

    bool operator<(const TeKmerEntry& o) const noexcept {
        return kmer_2bit < o.kmer_2bit;
    }
};
#pragma pack(pop)
static_assert(sizeof(TeKmerEntry) == 12, "TeKmerEntry must be 12 bytes (packed)");

struct TeKmerHit {
    uint32_t family_id;
    float    score;           // n_hits / n_valid_kmers
    uint32_t n_hits;
    uint32_t n_valid_kmers;
};

// ── Bloom filter (matches txome_gene_index.h approach) ───────────────────────

class TeBloomFilter {
public:
    void build(const std::vector<TeKmerEntry>& entries) {
        constexpr uint64_t NBITS = 1ULL << 27; // 128M bits = 16MB (TE index much smaller than gene)
        mask_ = NBITS - 1;
        bits_.assign(NBITS / 64, 0);
        for (const auto& e : entries) {
            uint64_t h1 = hash1_(e.kmer_2bit);
            uint64_t h2 = hash2_(e.kmer_2bit);
            uint64_t h3 = hash3_(e.kmer_2bit);
            bits_[h1 / 64] |= (1ULL << (h1 % 64));
            bits_[h2 / 64] |= (1ULL << (h2 % 64));
            bits_[h3 / 64] |= (1ULL << (h3 % 64));
        }
    }

    bool maybe_contains(uint64_t kmer) const noexcept {
        uint64_t h1 = hash1_(kmer);
        uint64_t h2 = hash2_(kmer);
        uint64_t h3 = hash3_(kmer);
        return (bits_[h1 / 64] & (1ULL << (h1 % 64))) &&
               (bits_[h2 / 64] & (1ULL << (h2 % 64))) &&
               (bits_[h3 / 64] & (1ULL << (h3 % 64)));
    }

    bool empty() const noexcept { return bits_.empty(); }

private:
    std::vector<uint64_t> bits_;
    uint64_t mask_ = 0;

    uint64_t hash1_(uint64_t x) const noexcept {
        x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL; x ^= x >> 33;
        return x & mask_;
    }
    uint64_t hash2_(uint64_t x) const noexcept {
        x ^= x >> 29; x *= 0xC4CEB9FE1A85EC53ULL; x ^= x >> 32;
        return x & mask_;
    }
    uint64_t hash3_(uint64_t x) const noexcept {
        x = (~x) + (x << 18); x ^= (x >> 31); x *= 21;
        x ^= (x >> 11); x += (x << 6); x ^= (x >> 22);
        return x & mask_;
    }
};

// ── TeKmerIndex — immutable after load ───────────────────────────────────────

class TeKmerIndex {
public:
    TeKmerIndex() = default;

    /// Load from binary .teki file
    static TeKmerIndex load(const std::string& path) {
        TeKmerIndex idx;
        std::ifstream f(path, std::ios::binary);
        if (!f) return idx;  // not found = disabled (soft fail)

        uint32_t magic, version, k, n_families, n_kmers, reserved;
        f.read(reinterpret_cast<char*>(&magic),      4);
        f.read(reinterpret_cast<char*>(&version),    4);
        f.read(reinterpret_cast<char*>(&k),          4);
        f.read(reinterpret_cast<char*>(&n_families), 4);
        f.read(reinterpret_cast<char*>(&n_kmers),    4);
        f.read(reinterpret_cast<char*>(&reserved),   4);

        if (magic != TEKI_MAGIC || version != TEKI_VERSION) return idx;

        idx.k_ = k;
        idx.family_names_.resize(n_families);

        // Read NUL-terminated family names
        for (uint32_t i = 0; i < n_families; ++i) {
            std::string name;
            char c;
            while (f.get(c) && c != '\0') name += c;
            idx.family_names_[i] = std::move(name);
        }

        // Read k-mer entries (chunked to handle large files >256MB)
        idx.entries_.resize(n_kmers);
        {
            char* dst = reinterpret_cast<char*>(idx.entries_.data());
            size_t total_bytes = static_cast<size_t>(n_kmers) * sizeof(TeKmerEntry);
            constexpr size_t CHUNK = 64ULL * 1024 * 1024;  // 64MB per read
            size_t offset = 0;
            while (offset < total_bytes && f) {
                size_t to_read = std::min(CHUNK, total_bytes - offset);
                f.read(dst + offset, static_cast<std::streamsize>(to_read));
                offset += to_read;
            }
        }

        if (!f) {
            idx.entries_.clear();
            return idx;
        }

        // Verify sorted (binary search requires it)
        assert(std::is_sorted(idx.entries_.begin(), idx.entries_.end()));

        // Build Bloom filter
        idx.bloom_.build(idx.entries_);

        std::fprintf(stderr, "[te-kmer-index] loaded %u family-unique %u-mers "
                     "across %u families from %s\n",
                     n_kmers, k, n_families, path.c_str());
        return idx;
    }

    /// Build in-memory from raw data (for unit tests)
    static TeKmerIndex from_entries(std::vector<TeKmerEntry> entries,
                                    std::vector<std::string> family_names,
                                    uint32_t k = TEKI_KMER_K) {
        TeKmerIndex idx;
        idx.k_ = k;
        idx.family_names_ = std::move(family_names);
        std::sort(entries.begin(), entries.end());
        idx.entries_ = std::move(entries);
        idx.bloom_.build(idx.entries_);
        return idx;
    }

    bool loaded() const noexcept { return !entries_.empty(); }
    uint32_t k() const noexcept { return k_; }
    size_t n_kmers() const noexcept { return entries_.size(); }
    size_t n_families() const noexcept { return family_names_.size(); }
    const std::string& family_name(uint32_t id) const { return family_names_[id]; }

    /// Classify a read by exact k-mer voting.
    ///
    /// Returns TeKmerHit if a single TE family collects >= min_hits k-mers
    /// AND no other family is within 50% of the top count (ambiguity).
    /// Returns nullopt otherwise (route to L3 STAR).
    std::optional<TeKmerHit> classify(std::string_view read,
                                       uint32_t min_hits = TEKI_MIN_HITS) const {
        if (entries_.empty() || read.size() < k_) return std::nullopt;

        // Temporary per-family hit counts (stack-allocated for small families)
        thread_local std::vector<uint32_t> counts;
        const size_t nf = family_names_.size();
        counts.assign(nf, 0);

        uint32_t n_valid = 0;

        // Rolling 2-bit encode + lookup
        const size_t n_pos = read.size() - k_ + 1;
        for (size_t i = 0; i < n_pos; ++i) {
            uint64_t kmer = encode_canonical_(read.data() + i, k_);
            if (kmer == UINT64_MAX) continue;
            ++n_valid;

            // Bloom filter pre-check
            if (!bloom_.maybe_contains(kmer)) continue;

            // Binary search in sorted entries
            auto it = std::lower_bound(
                entries_.begin(), entries_.end(),
                TeKmerEntry{kmer, 0});
            if (it != entries_.end() && it->kmer_2bit == kmer) {
                ++counts[it->family_id];
            }
        }

        if (n_valid == 0) return std::nullopt;

        // Find best and second-best
        uint32_t best_id  = 0, best_cnt  = 0;
        uint32_t second_cnt = 0;
        for (uint32_t i = 0; i < static_cast<uint32_t>(nf); ++i) {
            if (counts[i] > best_cnt) {
                second_cnt = best_cnt;
                best_cnt   = counts[i];
                best_id    = i;
            } else if (counts[i] > second_cnt) {
                second_cnt = counts[i];
            }
        }

        // Must have minimum hits
        if (best_cnt < min_hits) return std::nullopt;

        // Ambiguity: second family must be < 50% of top
        if (second_cnt * 2 >= best_cnt) return std::nullopt;

        float score = static_cast<float>(best_cnt) / static_cast<float>(n_valid);
        return TeKmerHit{best_id, score, best_cnt, n_valid};
    }

private:
    uint32_t k_ = TEKI_KMER_K;
    std::vector<std::string>  family_names_;
    std::vector<TeKmerEntry>  entries_;     // sorted by kmer_2bit
    TeBloomFilter             bloom_;

    // 2-bit encode canonical (min of fwd/rev) — same approach as txome_gene_index.h
    static uint64_t encode_canonical_(const char* seq, uint32_t k) noexcept {
        uint64_t fwd = 0, rev = 0;
        static constexpr uint8_t rc_bit[4] = {3, 2, 1, 0};
        const uint64_t mask = (k < 32) ? ((1ULL << (2 * k)) - 1) : ~0ULL;
        for (uint32_t i = 0; i < k; ++i) {
            uint8_t b;
            switch (seq[i] | 32) {
                case 'a': b = 0; break;
                case 'c': b = 1; break;
                case 'g': b = 2; break;
                case 't': b = 3; break;
                default:  return UINT64_MAX;
            }
            fwd = ((fwd << 2) | b) & mask;
            rev = (rev >> 2) | (static_cast<uint64_t>(rc_bit[b]) << (2 * (k - 1)));
        }
        return std::min(fwd, rev);
    }
};

}  // namespace singlet
