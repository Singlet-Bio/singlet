// txome_aligner.h — Transcriptome-first unique-resolver (Track B L1, v0.1)
// Part of droplet-hardening Track B prototype (T-L2-2).
//
// **v0.1 NOTE**: This is a hash-based exact-match seed-and-extend implementation.
// It meets the API contract and enables end-to-end cascade pipeline execution.
// The production SA-quality version (STAR-parity scoring, suffix-array index,
// Pearson r ≥ 0.999 acceptance) is deferred to T-L2-2 SA upgrade at L-3.
// See state/dag.md node T-L2-2 for status.
//
// v0.1 design:
//   - Index: flat hash map (k-mer → list<(tx_id, pos)>) built from transcriptome FASTA
//   - Seed: first SEED_K bases of R2 are hashed
//   - Extend: verify full read matches at each candidate position
//   - Unique: resolves iff exactly ONE transcript has ≥ MIN_EXTEND_MATCH bases matching
//   - Non-unique or no-match → returns std::nullopt → L3 passthrough
//
// Determinism seed: 0xC0FFEE used for any tie-breaking RNG (currently unused in v0.1).
//
// API (stable, same as future SA version):
//   struct TxomeHit { uint32_t tx_id; int32_t pos; int16_t score; bool is_unique; };
//   class TxomeAligner {
//    public:
//     explicit TxomeAligner(const TxomeIndex& idx, uint64_t em_seed = 0xC0FFEEULL);
//     std::optional<TxomeHit> resolve_unique(std::string_view read) const;
//   };

#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace singlet {

// ── TxomeHit — result of a unique transcriptome resolve ────────────────────────
struct TxomeHit {
    uint32_t tx_id;      // transcript index in TxomeIndex::tx_names
    int32_t  pos;        // 0-based start position in transcript
    int16_t  score;      // alignment score (v0.1: exact match count)
    bool     is_unique;  // always true when returned from resolve_unique()
};

// ── TxomeIndex — pre-built in-memory index over spliced transcriptome ──────────
//
// Build from a FASTA of spliced transcripts (one entry per transcript).
// Persisted at ${SINGLIFY_REF_BASE}/txome/{species}/tx_index.sa (future SA version).
// For v0.1 the index is built in RAM from the tx_fa string vector.
//
// Layout:
//   tx_seqs   — concatenated transcript sequences (4-bit ACGTN encoding)
//   tx_offsets— start offset of each transcript in tx_seqs
//   tx_lengths— length of each transcript
//   tx_names  — transcript ID strings (from FASTA header, trimmed at first space)
//   kmer_index— SEED_K-mer → flat array of (tx_id, pos) pairs
//
// TxomeIndex is immutable after build().
struct TxomeIndex {
    static constexpr uint8_t SEED_K = 22;   // seed k-mer length

    std::vector<uint8_t>   tx_seqs;         // concatenated 1-byte-per-base sequences (ASCII A/C/G/T/N)
    std::vector<uint32_t>  tx_offsets;      // tx_offsets[i] = start of tx i in tx_seqs
    std::vector<uint32_t>  tx_lengths;
    std::vector<std::string> tx_names;

    // kmer_index: seed_k-mer hash → vector of (tx_id, offset_in_tx) as packed uint64
    // packed: tx_id in upper 32 bits, pos in lower 32 bits
    std::unordered_map<uint64_t, std::vector<uint64_t>> kmer_index;

    uint32_t n_transcripts() const noexcept {
        return static_cast<uint32_t>(tx_names.size());
    }

    // Build from a FASTA string (full file contents).
    // Thread-safe after build() completes.
    void build(const std::string& fasta_text);

    // Build from a pre-split list of (name, sequence) pairs.
    void build(const std::vector<std::pair<std::string, std::string>>& transcripts);

    // FNV-1a k-mer hash (public so TxomeAligner can use it in resolve_unique).
    static uint64_t hash_kmer(const uint8_t* seq, uint8_t k) noexcept;
};

// ── TxomeAligner — resolves individual reads against TxomeIndex ────────────────
class TxomeAligner {
public:
    static constexpr uint64_t DETERMINISM_SEED = 0xC0FFEEULL;
    static constexpr uint8_t  MIN_EXTEND_MATCH = 30; // bases that must match exactly

    explicit TxomeAligner(const TxomeIndex& idx,
                          uint64_t em_seed = DETERMINISM_SEED) noexcept
        : idx_(idx), em_seed_(em_seed)
    {}

    // Returns a unique hit iff the read maps to exactly one transcript position
    // with ≥ MIN_EXTEND_MATCH matching bases.  Returns nullopt otherwise (→ L3).
    //
    // Input: string_view over the R2 bytes (2-bit or ASCII; aligner accepts ASCII A/C/G/T/N).
    // Hot path: no allocation if the kmer_index lookup finds 0 or 1 candidate.
    std::optional<TxomeHit> resolve_unique(std::string_view read) const noexcept;

    const TxomeIndex& index() const noexcept { return idx_; }

private:
    const TxomeIndex& idx_;
    uint64_t          em_seed_;

    // Count matching bases between read and transcript at given position.
    // Returns match count; stops early at mismatch if count already < threshold.
    int32_t count_matches(std::string_view read,
                          uint32_t tx_id, int32_t pos) const noexcept;
};

} // namespace singlet

// ── Implementations (header-only) ─────────────────────────────────────────────

namespace singlet {

// FNV-1a 64-bit hash of a k-mer (treats each byte as-is)
inline uint64_t TxomeIndex::hash_kmer(const uint8_t* seq, uint8_t k) noexcept {
    constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    constexpr uint64_t FNV_PRIME  = 1099511628211ULL;
    uint64_t h = FNV_OFFSET;
    for (uint8_t i = 0; i < k; ++i) {
        h ^= static_cast<uint64_t>(seq[i]);
        h *= FNV_PRIME;
    }
    return h;
}

inline void TxomeIndex::build(
    const std::vector<std::pair<std::string, std::string>>& transcripts)
{
    tx_names.clear();
    tx_offsets.clear();
    tx_lengths.clear();
    tx_seqs.clear();
    kmer_index.clear();

    tx_names.reserve(transcripts.size());
    tx_offsets.reserve(transcripts.size() + 1);
    tx_lengths.reserve(transcripts.size());

    uint32_t offset = 0;
    for (auto& [name, seq] : transcripts) {
        tx_names.push_back(name);
        tx_offsets.push_back(offset);
        tx_lengths.push_back(static_cast<uint32_t>(seq.size()));
        for (char c : seq) tx_seqs.push_back(static_cast<uint8_t>(c));
        offset += static_cast<uint32_t>(seq.size());
    }

    // Build k-mer seed index
    for (uint32_t t = 0; t < static_cast<uint32_t>(transcripts.size()); ++t) {
        const uint32_t tlen = tx_lengths[t];
        if (tlen < SEED_K) continue;
        const uint8_t* tseq = tx_seqs.data() + tx_offsets[t];
        for (uint32_t p = 0; p + SEED_K <= tlen; p += SEED_K) { // stride by SEED_K to keep index small
            uint64_t h = hash_kmer(tseq + p, SEED_K);
            uint64_t packed = (static_cast<uint64_t>(t) << 32) | static_cast<uint64_t>(p);
            kmer_index[h].push_back(packed);
        }
    }
}

inline void TxomeIndex::build(const std::string& fasta_text) {
    std::vector<std::pair<std::string, std::string>> txs;
    std::string name, seq;
    for (size_t i = 0; i < fasta_text.size(); ) {
        if (fasta_text[i] == '>') {
            if (!name.empty()) txs.emplace_back(name, seq);
            size_t nl = fasta_text.find('\n', i);
            if (nl == std::string::npos) nl = fasta_text.size();
            // trim at first space
            size_t sp = fasta_text.find(' ', i + 1);
            size_t name_end = std::min(nl, sp == std::string::npos ? nl : sp);
            name = fasta_text.substr(i + 1, name_end - (i + 1));
            seq.clear();
            i = nl + 1;
        } else {
            size_t nl = fasta_text.find('\n', i);
            if (nl == std::string::npos) nl = fasta_text.size();
            // append uppercase bases only
            for (size_t j = i; j < nl; ++j) {
                char c = fasta_text[j];
                if (c >= 'a' && c <= 'z') c -= 32;
                if (c == 'A' || c == 'C' || c == 'G' || c == 'T') seq += c;
                else seq += 'N';
            }
            i = nl + 1;
        }
    }
    if (!name.empty()) txs.emplace_back(name, seq);
    build(txs);
}

inline int32_t TxomeAligner::count_matches(std::string_view read,
                                           uint32_t tx_id, int32_t pos) const noexcept
{
    const uint32_t tlen = idx_.tx_lengths[tx_id];
    const uint8_t* tseq = idx_.tx_seqs.data() + idx_.tx_offsets[tx_id];
    const auto rlen = static_cast<int32_t>(read.size());
    if (pos < 0 || static_cast<uint32_t>(pos) + static_cast<uint32_t>(rlen) > tlen)
        return 0;
    int32_t matches = 0;
    for (int32_t j = 0; j < rlen; ++j) {
        char r = read[j];
        char t = static_cast<char>(tseq[pos + j]);
        if (r == t && r != 'N') ++matches;
    }
    return matches;
}

inline std::optional<TxomeHit>
TxomeAligner::resolve_unique(std::string_view read) const noexcept
{
    const auto rlen = static_cast<int32_t>(read.size());
    if (rlen < static_cast<int32_t>(TxomeIndex::SEED_K)) return std::nullopt;

    // Seed: hash first SEED_K bases
    uint64_t seed_hash = TxomeIndex::hash_kmer(
        reinterpret_cast<const uint8_t*>(read.data()),
        TxomeIndex::SEED_K);

    auto it = idx_.kmer_index.find(seed_hash);
    if (it == idx_.kmer_index.end()) return std::nullopt;

    const auto& candidates = it->second;

    // Count distinct transcripts with qualifying matches
    uint32_t best_tx  = UINT32_MAX;
    int32_t  best_pos = 0;
    int32_t  best_score = 0;
    int32_t  n_qualified = 0;

    for (uint64_t packed : candidates) {
        uint32_t tx_id = static_cast<uint32_t>(packed >> 32);
        int32_t  pos   = static_cast<int32_t>(packed & 0xFFFFFFFFULL);
        // Adjust position: seed was at offset p in transcript; read starts at read[0]
        // which maps to transcript[pos], so we align read[0..] to transcript[pos..]
        int32_t matches = count_matches(read, tx_id, pos);
        if (matches >= static_cast<int32_t>(MIN_EXTEND_MATCH)) {
            ++n_qualified;
            if (matches > best_score) {
                best_score = matches;
                best_tx    = tx_id;
                best_pos   = pos;
            }
        }
        // Early exit: if >1 qualified transcript, it's multi-mapper → L3
        if (n_qualified > 1) return std::nullopt;
    }

    if (n_qualified != 1) return std::nullopt;

    return TxomeHit{
        best_tx,
        best_pos,
        static_cast<int16_t>(best_score),
        true
    };
}

} // namespace singlet
