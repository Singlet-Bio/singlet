#ifndef H_GenomePacked
#define H_GenomePacked

#include <cstdint>
#include <cstring>

// 2-bit packed genome for cache-efficient SA binary search.
//
// Encoding: A=00, C=01, G=10, T=11 (matches STAR's 0,1,2,3 numeric encoding)
// N/spacer bases stored as A(00); tracked separately via sparse N-word flags.
// Packing: MSB-first — base[0] in bits 63:62, base[1] in bits 61:60, etc.
//
// Memory layout:
//   G_packed[w]  = 32 consecutive genome bases in uint64_t (word w covers positions w*32..w*32+31)
//   G_N_words[j] = 64-bit word where bit k=1 means G_packed[j*64+k] contains ≥1 N/spacer base
//
// Human genome (3.2B bases):  G_packed ≈ 800 MB,  G_N_words ≈ 1.6 MB

struct GenomePacked {
    uint64_t *G_packed;
    uint64_t *G_N_words;
    uint64_t n_packed_words;
    uint64_t n_N_flag_words;

    GenomePacked() : G_packed(nullptr), G_N_words(nullptr), n_packed_words(0), n_N_flag_words(0) {}

    void free() {
        delete[] G_packed;  G_packed  = nullptr;
        delete[] G_N_words; G_N_words = nullptr;
    }
};

//=============================================================================
// Packing
//=============================================================================

// Pack byte-per-base genome G[0..nGenome-1] into 2-bit representation.
inline void packGenome(const char *G, uint64_t nGenome, GenomePacked &gp) {
    gp.n_packed_words = (nGenome + 31) / 32;
    gp.n_N_flag_words = (gp.n_packed_words + 63) / 64;

    gp.G_packed  = new uint64_t[gp.n_packed_words + 1]();  // +1 sentinel for genomeWordAt straddling
    gp.G_N_words = new uint64_t[gp.n_N_flag_words]();

    for (uint64_t w = 0; w < gp.n_packed_words; w++) {
        uint64_t packed = 0;
        bool has_N = false;
        uint64_t base_start = w * 32;

        for (int b = 0; b < 32; b++) {
            uint64_t pos = base_start + b;
            if (pos < nGenome) {
                char c = G[pos];
                if (c >= 0 && c <= 3) {
                    packed |= ((uint64_t)c << (62 - 2 * b));
                } else {
                    has_N = true;  // N(4)/spacer(5) → A(00), already zero
                }
            }
        }
        gp.G_packed[w] = packed;
        if (has_N)
            gp.G_N_words[w / 64] |= (1ULL << (w % 64));
    }
}

//=============================================================================
// Single-base access
//=============================================================================

// Extract one base (0-3) from packed genome. No N detection.
static inline char genomeBasePacked(const uint64_t *G_packed, uint64_t pos) {
    return (char)((G_packed[pos >> 5] >> (62 - 2 * (pos & 31))) & 3);
}

// Check if any N exists in the 32-base word containing position `pos`.
static inline bool wordHasN(const uint64_t *G_N_words, uint64_t pos) {
    uint64_t w = pos >> 5;
    return (G_N_words[w >> 6] >> (w & 63)) & 1;
}

//=============================================================================
// Word-level access
//=============================================================================

// 32 bases starting at arbitrary genome position `pos`, packed MSB-first.
static inline uint64_t genomeWordAt(const uint64_t *G_packed, uint64_t pos) {
    uint64_t w = pos >> 5;
    unsigned shift = (unsigned)(pos & 31) * 2;
    if (shift == 0)
        return G_packed[w];
    return (G_packed[w] << shift) | (G_packed[w + 1] >> (64 - shift));
}

// Reverse the order of 32 2-bit bases within a uint64_t.
// Input MSB-first: base[0] base[1] ... base[31]
// Output MSB-first: base[31] base[30] ... base[0]
static inline uint64_t reversePackedWord(uint64_t v) {
    // Swap adjacent 2-bit pairs
    v = ((v & 0x3333333333333333ULL) << 2) | ((v >> 2) & 0x3333333333333333ULL);
    // Swap adjacent 4-bit nibbles
    v = ((v & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((v >> 4) & 0x0F0F0F0F0F0F0F0FULL);
    // Reverse bytes
    v = __builtin_bswap64(v);
    return v;
}

// 32 bases ending at genome position `pos`, returned in reverse order (pos, pos-1, ..., pos-31).
// Used for reverse-strand genome access.
static inline uint64_t genomeWordAtReverse(const uint64_t *G_packed, uint64_t pos) {
    // pos-31 through pos are the 32 bases; genomeWordAt gives them in forward order
    return reversePackedWord(genomeWordAt(G_packed, pos - 31));
}

// Check N flags for a range of genome positions [lo, hi] (inclusive).
// Returns true if any 32-base word in the range contains an N.
static inline bool rangeHasN(const uint64_t *G_N_words, uint64_t lo, uint64_t hi) {
    uint64_t w_lo = lo >> 5;
    uint64_t w_hi = hi >> 5;
    for (uint64_t w = w_lo; w <= w_hi; w++) {
        if ((G_N_words[w >> 6] >> (w & 63)) & 1)
            return true;
    }
    return false;
}

//=============================================================================
// Read packing helpers
//=============================================================================

// Pack `len` bases from read (byte-numeric 0-3, 4=N) into uint64_t words, MSB-first.
// N bases packed as A(0). Returns true if any N found.
// `out` must have room for (len+31)/32 + 1 words (+1 sentinel).
static inline bool packReadForward(const char *read, unsigned len, uint64_t *out) {
    bool has_N = false;
    unsigned n_words = (len + 31) / 32;
    for (unsigned w = 0; w < n_words; w++) {
        uint64_t packed = 0;
        unsigned base_start = w * 32;
        unsigned end = (base_start + 32 <= len) ? 32 : (len - base_start);
        for (unsigned b = 0; b < end; b++) {
            char c = read[base_start + b];
            if (__builtin_expect(c > 3, 0)) { has_N = true; continue; }
            packed |= ((uint64_t)(unsigned char)c << (62 - 2 * b));
        }
        out[w] = packed;
    }
    out[n_words] = 0;  // sentinel
    return has_N;
}

// Pack `len` bases from read going BACKWARD: read[0], read[-1], read[-2], ..., read[-(len-1)]
// Result is MSB-first (read[0] in bits 63:62).
static inline bool packReadBackward(const char *read, unsigned len, uint64_t *out) {
    bool has_N = false;
    unsigned n_words = (len + 31) / 32;
    for (unsigned w = 0; w < n_words; w++) {
        uint64_t packed = 0;
        unsigned base_start = w * 32;
        unsigned end = (base_start + 32 <= len) ? 32 : (len - base_start);
        for (unsigned b = 0; b < end; b++) {
            char c = read[-(int)(base_start + b)];
            if (__builtin_expect(c > 3, 0)) { has_N = true; continue; }
            packed |= ((uint64_t)(unsigned char)c << (62 - 2 * b));
        }
        out[w] = packed;
    }
    out[n_words] = 0;  // sentinel
    return has_N;
}

#endif
