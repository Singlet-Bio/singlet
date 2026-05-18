// SPDX-License-Identifier: MIT
// pileup/kmer_util.h — Shared k-mer reverse-complement / canonicalization helpers
//
// Consolidates reverse-complement and canonical-k-mer logic that was previously
// rolled separately in species_detect.h, mt_event_caller.h, te_classifier.h,
// minimizer_index.h, bloom_filter.h and fq/packed_align.h.
//
// Encodings handled:
//   - 2-bit packed k-mers in a uint64_t (A=0,C=1,G=2,T=3, MSB-first per shift loop)
//   - byte-numeric sequences (one base per byte, A=0,C=1,G=2,T=3, N>3)
//   - ASCII sequences (A/C/G/T case-insensitive, anything else ambiguous)
//
// Note: fq/packed_align.h keeps its own bit-parallel revcomp_word()/revcomp()
// for full 32-base-per-word packed multi-word sequences — that is a different
// abstraction (bit-parallel, multi-word) and is intentionally not folded here.

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

namespace singlet::pileup {
namespace kmer {

// Sentinel returned by canonical_* helpers when an ambiguous base is present.
inline constexpr uint64_t INVALID_KMER = ~0ULL;

// ── 2-bit packed k-mer helpers ─────────────────────────────────────────────

// Reverse-complement a 2-bit-encoded k-mer of length k (k <= 32).
// Complement is bitwise (A<->T, C<->G == XOR 0b11 per 2-bit symbol); the
// symbols are then reversed.
inline uint64_t revcomp_2bit(uint64_t kmer, int k) {
    const uint64_t mask = (k < 32) ? ((1ULL << (2 * k)) - 1) : ~0ULL;
    uint64_t rc_bits = (~kmer) & mask;  // complement every 2-bit symbol
    uint64_t out = 0;
    for (int i = 0; i < k; ++i) {
        out = (out << 2) | (rc_bits & 3);
        rc_bits >>= 2;
    }
    return out & mask;
}

// Canonical form of a forward 2-bit k-mer = min(forward, reverse-complement).
inline uint64_t canonical_2bit(uint64_t fwd, int k) {
    uint64_t rc = revcomp_2bit(fwd, k);
    return fwd < rc ? fwd : rc;
}

// ── ASCII helpers ──────────────────────────────────────────────────────────

// Encode a single ASCII base to 2 bits (ACGT -> 0,1,2,3; anything else -> 4).
inline uint8_t base2bit(char c) {
    switch (c | 32) {
        case 'a': return 0;
        case 'c': return 1;
        case 'g': return 2;
        case 't': return 3;
        default:  return 4;
    }
}

// Canonical 2-bit k-mer from an ASCII sequence pointer (length k).
// Returns INVALID_KMER if any ambiguous base is present.
inline uint64_t canonical_kmer_ascii(const char* seq, uint32_t k) {
    uint64_t fwd = 0, rev = 0;
    static constexpr uint8_t rc_bit[4] = {3, 2, 1, 0};
    const uint64_t mask = (k < 32) ? ((1ULL << (2 * k)) - 1) : ~0ULL;
    for (uint32_t i = 0; i < k; ++i) {
        uint8_t b = base2bit(seq[i]);
        if (b == 4) return INVALID_KMER;  // ambiguous base
        fwd = ((fwd << 2) | b) & mask;
        rev = (rev >> 2) | (static_cast<uint64_t>(rc_bit[b]) << (2 * (k - 1)));
    }
    return std::min(fwd, rev);
}

// Reverse-complement an ASCII string (ACGT case-sensitive upper; others -> N).
inline std::string revcomp_ascii(const std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i) {
        switch (s[i]) {
            case 'A': r += 'T'; break;
            case 'T': r += 'A'; break;
            case 'C': r += 'G'; break;
            case 'G': r += 'C'; break;
            default:  r += 'N'; break;
        }
    }
    return r;
}

// ── byte-numeric helpers (one base/byte, A=0,C=1,G=2,T=3, N>3) ─────────────

// Canonical 2-bit k-mer from a byte-numeric sequence pointer (length k).
// Returns INVALID_KMER if any byte is > 3 (ambiguous / N).
inline uint64_t canonical_kmer_numeric(const uint8_t* seq, int k) {
    uint64_t fwd = 0, rc = 0;
    for (int i = 0; i < k; ++i) {
        if (seq[i] > 3) return INVALID_KMER;
        fwd = (fwd << 2) | seq[i];
        rc  = (rc >> 2) | (static_cast<uint64_t>(3u - seq[i]) << (2 * (k - 1)));
    }
    return std::min(fwd, rc);
}

}  // namespace kmer
}  // namespace singlet::pileup
