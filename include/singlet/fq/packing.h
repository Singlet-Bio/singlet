// lib1fq/packing.h — 2-bit sequence packing and unpacking
//
// Pack DNA sequences (byte-numeric 0/1/2/3/4) into 2-bit representation
// (A=00, C=01, G=10, T=11) with a separate N-bitmap for ambiguous bases.
//
// Layout for a sequence of length L:
//   packed_bytes = ceil(L / 4)  — 4 bases per byte, MSB-first
//   n_bitmap bytes = ceil(L / 8) — 1 bit per base (only if has_n)
//
// Performance target: >4 GB/s pack, >8 GB/s unpack (single thread)

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace lib1fq {
namespace pack {

// ── Pack byte-numeric sequence → 2-bit packed + optional N-bitmap ──
//
// src:      byte-numeric (0/1/2/3/4), length = n_bases
// dst:      output buffer, must hold at least packed_bytes(n_bases) bytes
// n_bitmap: output buffer for N positions, must hold n_bitmap_bytes(n_bases)
//           Pass nullptr if you know there are no N's.
// Returns:  true if any N bases were found

inline constexpr uint32_t packed_bytes(uint32_t n_bases) {
    return (n_bases + 3) / 4;
}

inline constexpr uint32_t n_bitmap_bytes(uint32_t n_bases) {
    return (n_bases + 7) / 8;
}

inline bool pack_2bit(const uint8_t* src, uint32_t n_bases,
                      uint8_t* dst, uint8_t* n_bitmap) {
    bool has_n = false;

    if (n_bitmap) {
        std::memset(n_bitmap, 0, n_bitmap_bytes(n_bases));
    }

    // Process 4 bases at a time
    uint32_t full_bytes = n_bases / 4;
    uint32_t i = 0;
    for (uint32_t b = 0; b < full_bytes; ++b) {
        uint8_t v0 = src[i], v1 = src[i+1], v2 = src[i+2], v3 = src[i+3];

        // Check for N (value 4)
        if ((v0 | v1 | v2 | v3) > 3) {
            has_n = true;
            if (n_bitmap) {
                if (v0 > 3) n_bitmap[i/8]     |= (1u << (7 - (i   % 8)));
                if (v1 > 3) n_bitmap[(i+1)/8]  |= (1u << (7 - ((i+1) % 8)));
                if (v2 > 3) n_bitmap[(i+2)/8]  |= (1u << (7 - ((i+2) % 8)));
                if (v3 > 3) n_bitmap[(i+3)/8]  |= (1u << (7 - ((i+3) % 8)));
            }
            v0 &= 0x03; v1 &= 0x03; v2 &= 0x03; v3 &= 0x03;
        }

        dst[b] = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3;
        i += 4;
    }

    // Handle remaining 1–3 bases
    uint32_t remain = n_bases - full_bytes * 4;
    if (remain > 0) {
        uint8_t last = 0;
        for (uint32_t r = 0; r < remain; ++r) {
            uint8_t v = src[i + r];
            if (v > 3) {
                has_n = true;
                if (n_bitmap) {
                    n_bitmap[(i+r)/8] |= (1u << (7 - ((i+r) % 8)));
                }
                v &= 0x03;
            }
            last |= (v << (6 - r * 2));
        }
        dst[full_bytes] = last;
    }

    return has_n;
}

// ── Unpack 2-bit → byte-numeric (without N recovery) ──
//
// src:      2-bit packed data, ceil(n_bases/4) bytes
// n_bases:  number of bases
// dst:      output buffer, n_bases bytes (0/1/2/3, no N's)

inline void unpack_2bit(const uint8_t* src, uint32_t n_bases, uint8_t* dst) {
    uint32_t full_bytes = n_bases / 4;
    uint32_t i = 0;
    for (uint32_t b = 0; b < full_bytes; ++b) {
        uint8_t v = src[b];
        dst[i]   = (v >> 6) & 0x03;
        dst[i+1] = (v >> 4) & 0x03;
        dst[i+2] = (v >> 2) & 0x03;
        dst[i+3] =  v       & 0x03;
        i += 4;
    }
    uint32_t remain = n_bases - full_bytes * 4;
    for (uint32_t r = 0; r < remain; ++r) {
        dst[i + r] = (src[full_bytes] >> (6 - r * 2)) & 0x03;
    }
}

// ── Restore N bases from bitmap ──

inline void apply_n_bitmap(const uint8_t* n_bitmap, uint32_t n_bases,
                           uint8_t* seq) {
    for (uint32_t i = 0; i < n_bases; ++i) {
        if (n_bitmap[i / 8] & (1u << (7 - (i % 8)))) {
            seq[i] = 4; // N
        }
    }
}

// ── Pack quality scores to binned-4 (2 bits each) ──
// phred_src: raw phred values, n_bases
// dst: packed output, ceil(n_bases/4) bytes

inline void pack_qual_binned4(const uint8_t* phred_src, uint32_t n_bases,
                              uint8_t* dst) {
    uint32_t full_bytes = n_bases / 4;
    uint32_t i = 0;
    for (uint32_t b = 0; b < full_bytes; ++b) {
        uint8_t q0 = (phred_src[i]   < 10) ? 0 : (phred_src[i]   < 20) ? 1 : (phred_src[i]   < 30) ? 2 : 3;
        uint8_t q1 = (phred_src[i+1] < 10) ? 0 : (phred_src[i+1] < 20) ? 1 : (phred_src[i+1] < 30) ? 2 : 3;
        uint8_t q2 = (phred_src[i+2] < 10) ? 0 : (phred_src[i+2] < 20) ? 1 : (phred_src[i+2] < 30) ? 2 : 3;
        uint8_t q3 = (phred_src[i+3] < 10) ? 0 : (phred_src[i+3] < 20) ? 1 : (phred_src[i+3] < 30) ? 2 : 3;
        dst[b] = (q0 << 6) | (q1 << 4) | (q2 << 2) | q3;
        i += 4;
    }
    uint32_t remain = n_bases - full_bytes * 4;
    if (remain > 0) {
        uint8_t last = 0;
        for (uint32_t r = 0; r < remain; ++r) {
            uint8_t q = (phred_src[i+r] < 10) ? 0 : (phred_src[i+r] < 20) ? 1 :
                        (phred_src[i+r] < 30) ? 2 : 3;
            last |= (q << (6 - r * 2));
        }
        dst[full_bytes] = last;
    }
}

// ── Unpack binned-4 quality → Phred values ──

inline void unpack_qual_binned4(const uint8_t* src, uint32_t n_bases,
                                uint8_t* dst) {
    static constexpr uint8_t centers[4] = {6, 15, 25, 37};
    uint32_t full_bytes = n_bases / 4;
    uint32_t i = 0;
    for (uint32_t b = 0; b < full_bytes; ++b) {
        uint8_t v = src[b];
        dst[i]   = centers[(v >> 6) & 0x03];
        dst[i+1] = centers[(v >> 4) & 0x03];
        dst[i+2] = centers[(v >> 2) & 0x03];
        dst[i+3] = centers[ v       & 0x03];
        i += 4;
    }
    uint32_t remain = n_bases - full_bytes * 4;
    for (uint32_t r = 0; r < remain; ++r) {
        dst[i + r] = centers[(src[full_bytes] >> (6 - r * 2)) & 0x03];
    }
}

// ── Pack quality scores to binned-2 (1 bit each, pass/fail at Q20) ──
// phred_src: raw phred values, n_bases
// dst: packed output, ceil(n_bases/8) bytes

inline constexpr uint32_t packed_bytes_binned2(uint32_t n_bases) {
    return (n_bases + 7) / 8;
}

inline void pack_qual_binned2(const uint8_t* phred_src, uint32_t n_bases,
                              uint8_t* dst) {
    uint32_t full_bytes = n_bases / 8;
    uint32_t i = 0;
    for (uint32_t b = 0; b < full_bytes; ++b) {
        uint8_t byte = 0;
        byte |= (phred_src[i]   >= 20 ? 1u : 0u) << 7;
        byte |= (phred_src[i+1] >= 20 ? 1u : 0u) << 6;
        byte |= (phred_src[i+2] >= 20 ? 1u : 0u) << 5;
        byte |= (phred_src[i+3] >= 20 ? 1u : 0u) << 4;
        byte |= (phred_src[i+4] >= 20 ? 1u : 0u) << 3;
        byte |= (phred_src[i+5] >= 20 ? 1u : 0u) << 2;
        byte |= (phred_src[i+6] >= 20 ? 1u : 0u) << 1;
        byte |= (phred_src[i+7] >= 20 ? 1u : 0u);
        dst[b] = byte;
        i += 8;
    }
    uint32_t remain = n_bases - full_bytes * 8;
    if (remain > 0) {
        uint8_t last = 0;
        for (uint32_t r = 0; r < remain; ++r) {
            last |= (phred_src[i+r] >= 20 ? 1u : 0u) << (7 - r);
        }
        dst[full_bytes] = last;
    }
}

// ── Unpack binned-2 quality → Phred values ──

inline void unpack_qual_binned2(const uint8_t* src, uint32_t n_bases,
                                uint8_t* dst) {
    static constexpr uint8_t centers[2] = {10, 30};
    uint32_t full_bytes = n_bases / 8;
    uint32_t i = 0;
    for (uint32_t b = 0; b < full_bytes; ++b) {
        uint8_t v = src[b];
        dst[i]   = centers[(v >> 7) & 0x01];
        dst[i+1] = centers[(v >> 6) & 0x01];
        dst[i+2] = centers[(v >> 5) & 0x01];
        dst[i+3] = centers[(v >> 4) & 0x01];
        dst[i+4] = centers[(v >> 3) & 0x01];
        dst[i+5] = centers[(v >> 2) & 0x01];
        dst[i+6] = centers[(v >> 1) & 0x01];
        dst[i+7] = centers[ v       & 0x01];
        i += 8;
    }
    uint32_t remain = n_bases - full_bytes * 8;
    for (uint32_t r = 0; r < remain; ++r) {
        dst[i + r] = centers[(src[full_bytes] >> (7 - r)) & 0x01];
    }
}

// ── Varint encoding ──

inline uint32_t encode_varint(uint64_t value, uint8_t* dst) {
    uint32_t n = 0;
    while (value >= 0x80) {
        dst[n++] = static_cast<uint8_t>(value & 0x7F) | 0x80;
        value >>= 7;
    }
    dst[n++] = static_cast<uint8_t>(value);
    return n;
}

inline uint32_t decode_varint(const uint8_t* src, uint64_t& value) {
    value = 0;
    uint32_t shift = 0;
    uint32_t n = 0;
    while (true) {
        uint8_t byte = src[n];
        value |= static_cast<uint64_t>(byte & 0x7F) << shift;
        ++n;
        if ((byte & 0x80) == 0) break;
        shift += 7;
        if (shift >= 64) break; // overflow protection
    }
    return n;
}

} // namespace pack
} // namespace lib1fq
