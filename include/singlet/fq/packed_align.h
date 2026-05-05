// lib1fq/packed_align.h — 2-bit packed alignment primitives (Phase 5)
//
// Provides 64-bit XOR-based sequence comparison (32 bases per operation),
// packed SA index lookup, and complement/reverse-complement for 2-bit data.
// These are building blocks for a STAR integration that operates on packed
// genome and read data directly, bypassing byte-numeric representation.

#pragma once

#include "packing.h"

#include <cstdint>
#include <cstring>

namespace lib1fq {
namespace packed_align {

// ── 64-bit packed comparison (32 bases per XOR) ──

// Compare packed read to packed genome starting at a base offset.
// Returns the number of matching bases from the start (mismatch position),
// or `length` if all bases match.
//
// read_packed:  2-bit packed read, left-aligned in uint64_t words
// genome_packed: 2-bit packed genome, left-aligned in uint64_t words
// genome_start:  base offset into genome
// length:        number of bases to compare
inline uint32_t compare_packed(const uint64_t* read_packed,
                               const uint64_t* genome_packed,
                               uint64_t genome_start,
                               uint32_t length) {
    const uint32_t word_idx = static_cast<uint32_t>(genome_start / 32);
    const uint32_t bit_shift = static_cast<uint32_t>((genome_start % 32) * 2);
    const uint64_t* g_ptr = genome_packed + word_idx;

    const uint32_t n_words = (length + 31) / 32;

    for (uint32_t w = 0; w < n_words; ++w) {
        // Extract 32 genome bases aligned to the read word
        uint64_t g_aligned;
        if (bit_shift == 0) {
            g_aligned = g_ptr[w];
        } else {
            g_aligned = (g_ptr[w] << bit_shift) |
                        (g_ptr[w + 1] >> (64 - bit_shift));
        }

        // Mask out unused trailing bases in the last word
        uint32_t bases_this_word = (w == n_words - 1) ?
            (length - w * 32) : 32;
        if (bases_this_word < 32) {
            uint64_t mask = ~((1ULL << (64 - bases_this_word * 2)) - 1);
            g_aligned &= mask;
            uint64_t r_masked = read_packed[w] & mask;
            uint64_t diff = g_aligned ^ r_masked;
            if (diff != 0) {
                return w * 32 + static_cast<uint32_t>(__builtin_clzll(diff) / 2);
            }
        } else {
            uint64_t diff = g_aligned ^ read_packed[w];
            if (diff != 0) {
                return w * 32 + static_cast<uint32_t>(__builtin_clzll(diff) / 2);
            }
        }
    }
    return length;  // all bases match
}

// ── SA index extraction ──

// Extract SA index key from packed read (single shift+mask).
// Replaces the 14-iteration loop in STAR's maxMappableLength2strands:
//   ind1 = 0; for (ii=0; ii<14; ii++) { ind1<<=2; ind1+=(uint)Read[ii]; }
//
// read_packed: 2-bit packed read (MSB-first)
// piece_start: base offset in the read
// key_length:  number of bases for SA index (STAR default: 14)
inline uint32_t sa_index_key(const uint64_t* read_packed,
                             uint32_t piece_start,
                             uint32_t key_length) {
    // Which word and bit offset
    uint32_t word = piece_start / 32;
    uint32_t bit_off = (piece_start % 32) * 2;
    uint32_t key_bits = key_length * 2;

    uint64_t val;
    if (bit_off + key_bits <= 64) {
        // Key fits in one word
        val = read_packed[word] >> (64 - bit_off - key_bits);
    } else {
        // Key spans two words
        val = (read_packed[word] << (bit_off + key_bits - 64)) |
              (read_packed[word + 1] >> (128 - bit_off - key_bits));
    }
    return static_cast<uint32_t>(val & ((1ULL << key_bits) - 1));
}

// ── 2-bit complement and reverse complement ──

// Complement: A(00)↔T(11), C(01)↔G(10) — XOR with all-1s
inline uint64_t complement_word(uint64_t packed) {
    return packed ^ 0xFFFFFFFFFFFFFFFFULL;
}

// Reverse base order within a 64-bit word (32 bases).
// Swaps pairs of bits to reverse the sequence.
inline uint64_t reverse_bases(uint64_t x) {
    // Swap adjacent 2-bit pairs
    x = ((x & 0x3333333333333333ULL) << 2) |
        ((x >> 2) & 0x3333333333333333ULL);
    // Swap 4-bit nibbles
    x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4) |
        ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
    // Swap bytes
    x = __builtin_bswap64(x);
    return x;
}

// Reverse complement a single word (32 bases)
inline uint64_t revcomp_word(uint64_t packed) {
    return reverse_bases(complement_word(packed));
}

// Reverse complement a multi-word packed sequence.
// Input: packed[0..n_words-1], length bases (MSB-first).
// Output: out[0..n_words-1], reverse complemented (MSB-first).
inline void revcomp(const uint64_t* packed, uint32_t length,
                    uint64_t* out) {
    uint32_t n_words = (length + 31) / 32;
    uint32_t tail_bases = length % 32;
    uint32_t tail_shift = (tail_bases > 0) ? (32 - tail_bases) * 2 : 0;

    // Reverse-complement each word and reverse word order
    for (uint32_t i = 0; i < n_words; ++i) {
        out[i] = revcomp_word(packed[n_words - 1 - i]);
    }

    // If length is not a multiple of 32, shift everything left
    // to maintain MSB-first alignment
    if (tail_shift > 0) {
        for (uint32_t i = 0; i < n_words - 1; ++i) {
            out[i] = (out[i] << tail_shift) |
                     (out[i + 1] >> (64 - tail_shift));
        }
        out[n_words - 1] <<= tail_shift;
    }
}

// ── Pack byte-numeric genome region to 64-bit words ──

// Pack a contiguous byte-numeric (0-4) region into 2-bit words.
// N bases (4) are stored as A (0). Caller uses N-bitmap for accuracy.
inline void pack_genome_region(const uint8_t* genome_bytes,
                               uint64_t start, uint32_t length,
                               uint64_t* packed_out) {
    uint32_t n_words = (length + 31) / 32;
    const uint8_t* src = genome_bytes + start;

    for (uint32_t w = 0; w < n_words; ++w) {
        uint64_t word = 0;
        uint32_t bases = (w == n_words - 1) ?
            (length - w * 32) : 32;
        for (uint32_t b = 0; b < bases; ++b) {
            uint8_t val = src[w * 32 + b];
            if (val > 3) val = 0;  // N → A
            word = (word << 2) | val;
        }
        // Left-align if partial word
        if (bases < 32) word <<= (32 - bases) * 2;
        packed_out[w] = word;
    }
}

// ── N-bitmap utilities ──

// Build N-bitmap for a genome region. Returns true if any N found.
inline bool build_n_bitmap(const uint8_t* genome_bytes,
                           uint64_t start, uint32_t length,
                           uint8_t* n_bitmap) {
    bool found_n = false;
    uint32_t n_bytes = (length + 7) / 8;
    std::memset(n_bitmap, 0, n_bytes);

    for (uint32_t i = 0; i < length; ++i) {
        if (genome_bytes[start + i] == 4) {
            n_bitmap[i / 8] |= (1 << (7 - (i % 8)));
            found_n = true;
        }
    }
    return found_n;
}

// Check if any base in [start, start+length) has N in the bitmap
inline bool has_n_in_range(const uint8_t* n_bitmap,
                           uint32_t start, uint32_t length) {
    for (uint32_t i = start; i < start + length; ++i) {
        if (n_bitmap[i / 8] & (1 << (7 - (i % 8))))
            return true;
    }
    return false;
}

} // namespace packed_align
} // namespace lib1fq
