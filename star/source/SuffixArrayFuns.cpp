#include "SuffixArrayFuns.h"
#include "PackedArray.h"
#ifdef GENOME_PACKED
#include "GenomePacked.h"
#endif

inline uint medianUint2(uint a, uint b)
{
    // returns (a+b)/2
    return a/2 + b/2 + (a%2 + b%2)/2;
};

#ifdef GENOME_PACKED

//=============================================================================
// 2-bit packed genome variant of compareSeqToGenome.
// All 4 direction branches use word-level XOR comparison (32 bases/op).
// Falls back to byte G[] only for the rare N-containing windows.
//=============================================================================

// Word-level forward comparison: read at s[0..], genome at G_packed[gpos..].
// Returns match length (0-based from the start of the comparison).
// On mismatch, sets compRes to (read_base > genome_base).
static inline uint packedForward(
    const uint64_t *G_packed, const uint64_t *G_N_words,
    const char *G_byte, // byte genome for N fallback
    const char *s,      // read pointer (forward)
    uint64 gpos,        // genome start position
    uint comp_len,      // max bases to compare
    bool &compRes)
{
    uint matched = 0;

    while (matched < comp_len) {
        uint remaining = comp_len - matched;
        uint word_bases = (remaining < 32) ? remaining : 32;

        // Check N flags for genome window
        if (__builtin_expect(rangeHasN(G_N_words, gpos + matched, gpos + matched + word_bases - 1), 0)) {
            // Rare: N in genome window — byte fallback
            const char *g = G_byte + gpos;
            for (uint b = 0; b < word_bases; b++) {
                if (s[matched + b] != g[matched + b]) {
                    compRes = (s[matched + b] > g[matched + b]);
                    return matched + b;
                }
            }
            matched += word_bases;
            continue;
        }

        // Pack read bases into a word
        uint64_t r_word = 0;
        bool read_has_N = false;
        for (uint b = 0; b < word_bases; b++) {
            char c = s[matched + b];
            if (__builtin_expect(c > 3, 0)) { read_has_N = true; break; }
            r_word |= ((uint64_t)(unsigned char)c << (62 - 2*b));
        }

        if (__builtin_expect(read_has_N, 0)) {
            const char *g = G_byte + gpos;
            for (uint b = 0; b < word_bases; b++) {
                if (s[matched + b] != g[matched + b]) {
                    compRes = (s[matched + b] > g[matched + b]);
                    return matched + b;
                }
            }
            matched += word_bases;
            continue;
        }

        // Get packed genome word
        uint64_t g_word = genomeWordAt(G_packed, gpos + matched);

        // Mask trailing bits for partial last word
        if (word_bases < 32) {
            uint64_t mask = ~((1ULL << (64 - word_bases * 2)) - 1);
            g_word &= mask;
            r_word &= mask;
        }

        uint64_t diff = g_word ^ r_word;
        if (diff != 0) {
            unsigned pos = (unsigned)(__builtin_clzll(diff) / 2);
            char gb = (char)((g_word >> (62 - 2*pos)) & 3);
            char rb = (char)((r_word >> (62 - 2*pos)) & 3);
            compRes = (rb > gb);
            return matched + pos;
        }
        matched += word_bases;
    }
    return comp_len;
}

// Word-level reverse-genome comparison: genome accessed at gpos, gpos-1, gpos-2, ...
// Read accessed forward at s[0], s[1], ...
// Used for dirG=false branches (reverse strand genome).
static inline uint packedReverse(
    const uint64_t *G_packed, const uint64_t *G_N_words,
    const char *G_byte,
    const char *s,
    uint64 gpos,        // genome start position (highest, decreasing)
    uint comp_len,
    bool &compRes,
    bool compN)         // if true, N/spacer in genome → compRes=false (dirR&&!dirG or !dirR&&!dirG behavior)
{
    uint matched = 0;

    while (matched < comp_len) {
        uint remaining = comp_len - matched;
        uint word_bases = (remaining < 32) ? remaining : 32;
        uint64 g_start = gpos - matched;  // current genome position (decreasing)
        uint64 g_end = g_start - (word_bases - 1);   // lowest genome position in window

        // Check N flags
        if (__builtin_expect(rangeHasN(G_N_words, g_end, g_start), 0)) {
            // N fallback: byte-by-byte, genome going backwards
            for (uint b = 0; b < word_bases; b++) {
                char gb = G_byte[gpos - matched - b];
                char sb = s[matched + b];
                if (sb != gb) {
                    if (compN && gb > 3) compRes = false;
                    else if (sb > gb || gb > 3) compRes = false;
                    else compRes = true;
                    return matched + b;
                }
            }
            matched += word_bases;
            continue;
        }

        // Pack read
        uint64_t r_word = 0;
        bool read_has_N = false;
        for (uint b = 0; b < word_bases; b++) {
            char c = s[matched + b];
            if (__builtin_expect(c > 3, 0)) { read_has_N = true; break; }
            r_word |= ((uint64_t)(unsigned char)c << (62 - 2*b));
        }

        if (__builtin_expect(read_has_N, 0)) {
            for (uint b = 0; b < word_bases; b++) {
                char gb = G_byte[gpos - matched - b];
                char sb = s[matched + b];
                if (sb != gb) {
                    if (compN && gb > 3) compRes = false;
                    else if (sb > gb || gb > 3) compRes = false;
                    else compRes = true;
                    return matched + b;
                }
            }
            matched += word_bases;
            continue;
        }

        // Get reversed genome word: bases at gpos-matched, gpos-matched-1, ..., gpos-matched-31
        // genomeWordAt gives forward order starting at g_end, reversePackedWord flips to descending
        uint64_t g_word;
        if (word_bases == 32) {
            g_word = reversePackedWord(genomeWordAt(G_packed, g_end));
        } else {
            // Partial word: get 32 bases ending at g_start, reverse, then mask
            // Get forward word starting at g_end
            uint64_t tmp = genomeWordAt(G_packed, g_end);
            // We only need the last word_bases of this 32-base window, reversed
            // After reverse: MSB has base at g_end+31, ..., LSB has base at g_end
            // We want bases at g_start, g_start-1, ..., g_end in MSB-first order
            // g_start = g_end + word_bases - 1
            // After reversing the full 32: position [0] = base at g_end+31
            // We need positions [32-word_bases .. 31] → shift left by (32-word_bases)*2
            tmp = reversePackedWord(tmp);
            g_word = tmp << ((32 - word_bases) * 2);
        }

        // Mask trailing bits
        if (word_bases < 32) {
            uint64_t mask = ~((1ULL << (64 - word_bases * 2)) - 1);
            g_word &= mask;
            r_word &= mask;
        }

        uint64_t diff = g_word ^ r_word;
        if (diff != 0) {
            unsigned pos = (unsigned)(__builtin_clzll(diff) / 2);
            char gb = (char)((g_word >> (62 - 2*pos)) & 3);
            char rb = (char)((r_word >> (62 - 2*pos)) & 3);
            // Reverse-genome branches: comparison direction is inverted
            // (original: s > g → compRes=false; s < g → compRes=true)
            compRes = (gb > rb);
            return matched + pos;
        }
        matched += word_bases;
    }
    return comp_len;
}

uint compareSeqToGenome(Genome &mapGen, char** s2, uint S, uint N, uint L, uint iSA, bool dirR, bool& compRes)
{
    uint SAstr=mapGen.SA[iSA];
    bool dirG = (SAstr>>mapGen.GstrandBit) == 0;
    SAstr &= mapGen.GstrandMask;

    const uint64_t *GP = mapGen.gPacked.G_packed;
    const uint64_t *GN = mapGen.gPacked.G_N_words;

    if (dirR && dirG) {
        // Forward read, forward genome — word-level XOR
        uint len = packedForward(GP, GN, mapGen.G, s2[0] + S + L, SAstr + L, N - L, compRes);
        return len + L;
    } else if (dirR && !dirG) {
        // Forward read (complement), reverse genome — word-level reversed XOR
        uint len = packedReverse(GP, GN, mapGen.G, s2[1] + S + L,
                                 mapGen.nGenome - 1 - SAstr - L, N - L, compRes, true);
        return len + L;
    } else if (!dirR && dirG) {
        // Reverse read (complement from S-L going down), forward genome
        // Read: s[-0], s[-1], s[-2], ... ; Genome: G[gpos], G[gpos+1], ...
        // Pack read backward, compare with forward genome word-level
        const char *s = s2[1] + S - L;
        uint64 gpos = SAstr + L;
        uint comp_len = N - L;
        uint matched = 0;

        while (matched < comp_len) {
            uint remaining = comp_len - matched;
            uint word_bases = (remaining < 32) ? remaining : 32;

            // Check genome N flags
            if (__builtin_expect(rangeHasN(GN, gpos + matched, gpos + matched + word_bases - 1), 0)) {
                for (uint b = 0; b < word_bases; b++) {
                    char gb = mapGen.G[gpos + matched + b];
                    char sb = s[-(int64)(matched + b)];
                    if (sb != gb) {
                        compRes = (sb > gb);
                        return matched + b + L;
                    }
                }
                matched += word_bases;
                continue;
            }

            // Pack read backward into word
            uint64_t r_word = 0;
            bool read_has_N = false;
            for (uint b = 0; b < word_bases; b++) {
                char c = s[-(int64)(matched + b)];
                if (__builtin_expect(c > 3, 0)) { read_has_N = true; break; }
                r_word |= ((uint64_t)(unsigned char)c << (62 - 2*b));
            }

            if (__builtin_expect(read_has_N, 0)) {
                for (uint b = 0; b < word_bases; b++) {
                    char gb = mapGen.G[gpos + matched + b];
                    char sb = s[-(int64)(matched + b)];
                    if (sb != gb) {
                        compRes = (sb > gb);
                        return matched + b + L;
                    }
                }
                matched += word_bases;
                continue;
            }

            uint64_t g_word = genomeWordAt(GP, gpos + matched);
            if (word_bases < 32) {
                uint64_t mask = ~((1ULL << (64 - word_bases * 2)) - 1);
                g_word &= mask;
                r_word &= mask;
            }

            uint64_t diff = g_word ^ r_word;
            if (diff != 0) {
                unsigned pos = (unsigned)(__builtin_clzll(diff) / 2);
                char gb = (char)((g_word >> (62 - 2*pos)) & 3);
                char rb = (char)((r_word >> (62 - 2*pos)) & 3);
                compRes = (rb > gb);
                return matched + pos + L;
            }
            matched += word_bases;
        }
        return N;
    } else {
        // Reverse read, reverse genome
        // Read: s2[0][S-L-0], s2[0][S-L-1], ... ; Genome: G[gpos], G[gpos-1], ...
        const char *s = s2[0] + S - L;
        uint64 gpos = mapGen.nGenome - 1 - SAstr - L;
        uint comp_len = N - L;
        uint matched = 0;

        while (matched < comp_len) {
            uint remaining = comp_len - matched;
            uint word_bases = (remaining < 32) ? remaining : 32;
            uint64 g_start = gpos - matched;
            uint64 g_end = g_start - (word_bases - 1);

            if (__builtin_expect(rangeHasN(GN, g_end, g_start), 0)) {
                for (uint b = 0; b < word_bases; b++) {
                    char gb = mapGen.G[gpos - matched - b];
                    char sb = s[-(int64)(matched + b)];
                    if (sb != gb) {
                        if (sb > gb || gb > 3)
                            compRes = false;
                        else
                            compRes = true;
                        return matched + b + L;
                    }
                }
                matched += word_bases;
                continue;
            }

            uint64_t r_word = 0;
            bool read_has_N = false;
            for (uint b = 0; b < word_bases; b++) {
                char c = s[-(int64)(matched + b)];
                if (__builtin_expect(c > 3, 0)) { read_has_N = true; break; }
                r_word |= ((uint64_t)(unsigned char)c << (62 - 2*b));
            }

            if (__builtin_expect(read_has_N, 0)) {
                for (uint b = 0; b < word_bases; b++) {
                    char gb = mapGen.G[gpos - matched - b];
                    char sb = s[-(int64)(matched + b)];
                    if (sb != gb) {
                        if (sb > gb || gb > 3)
                            compRes = false;
                        else
                            compRes = true;
                        return matched + b + L;
                    }
                }
                matched += word_bases;
                continue;
            }

            // Reversed genome word
            uint64_t g_word;
            if (word_bases == 32) {
                g_word = reversePackedWord(genomeWordAt(GP, g_end));
            } else {
                uint64_t tmp = reversePackedWord(genomeWordAt(GP, g_end));
                g_word = tmp << ((32 - word_bases) * 2);
            }

            if (word_bases < 32) {
                uint64_t mask = ~((1ULL << (64 - word_bases * 2)) - 1);
                g_word &= mask;
                r_word &= mask;
            }

            uint64_t diff = g_word ^ r_word;
            if (diff != 0) {
                unsigned pos = (unsigned)(__builtin_clzll(diff) / 2);
                char gb = (char)((g_word >> (62 - 2*pos)) & 3);
                char rb = (char)((r_word >> (62 - 2*pos)) & 3);
                // Reverse genome: comparison inverted
                compRes = (gb > rb);
                return matched + pos + L;
            }
            matched += word_bases;
        }
        return N;
    }
};

#else // !GENOME_PACKED — original byte-per-base implementation

uint compareSeqToGenome(Genome &mapGen, char** s2, uint S, uint N, uint L, uint iSA, bool dirR, bool& compRes)
{
    /* compare s to g, find the maximum identity length
     * s2[0] read sequence; s2[1] complementary sequence
     * S position to start search from in s2[0],s2[1]
     * dirR forward or reverse direction search on read sequence
     */

    register int64 ii;

    uint SAstr=mapGen.SA[iSA];
    bool dirG = (SAstr>>mapGen.GstrandBit) == 0; //forward or reverse strand of the genome
    SAstr &= mapGen.GstrandMask;

    char *g=mapGen.G;

#ifdef SA_GENOME_PREFETCH
    {
        // Prefetch the genome window we are about to compare against.
        // Issues before the comparison loop so the ~300-cycle DRAM load can begin.
        const char* gp = dirG ? (g + SAstr + L) : (g + mapGen.nGenome - 1 - SAstr - L);
        __builtin_prefetch(gp,      0, 0);
        __builtin_prefetch(gp + 64, 0, 0);  // covers reads up to ~128 bp
    }
#endif

    if (dirR && dirG) {//forward on read, forward on genome
        char* s  = s2[0] + S + L;
        g += SAstr + L;
        for (ii=0;(uint) ii < N-L; ii++)
        {
            if (s[ii]!=g[ii])
            {
                if (s[ii]>g[ii])
                {
                    compRes=true;
                    return ii+L;
                } else
                {
                    compRes=false;
                    return ii+L;
                };
            };
        };
        return N; //exact match
    } else if (dirR && !dirG) {
        char* s  = s2[1] + S + L;
        g += mapGen.nGenome-1-SAstr - L;
        for (ii=0; (uint) ii < N-L; ii++)
        {
            if (s[ii]!=g[-ii])
            {
                if (s[ii]>g[-ii] || g[-ii]>3)
                {
                    compRes=false;
                    return ii+L;
                } else
                {
                    compRes=true;
                    return ii+L;
                };
            };
        };
        return N;
    } else if (!dirR && dirG) {
        char* s  = s2[1] + S - L;
        g += SAstr + L;
        for (ii=0; (uint) ii < N-L; ii++)
        {
            if (s[-ii]!=g[ii])
            {
                if (s[-ii]>g[ii]) {
                    compRes=true;
                    return ii+L;

                } else
                {
                    compRes=false;
                    return ii+L;
                };
            };
        };
        return N;
    } else {//if (!dirR && !dirG)
        char* s  = s2[0] + S - L;
        g += mapGen.nGenome-1-SAstr - L;
        for (ii=0; (uint) ii < N-L; ii++)
        {
            if (s[-ii]!=g[-ii])
            {
                if (s[-ii]>g[-ii] || g[-ii]>3)
                {
                    compRes=false;
                    return ii+L;
                } else
                {
                    compRes=true;
                    return ii+L;
                };
            };
        };
        return N;
    };
};

#endif // GENOME_PACKED

uint findMultRange(Genome &mapGen, uint i3, uint L3, uint i1, uint L1, uint i1a, uint L1a, uint i1b, uint L1b, char** s, bool dirR, uint S)
{    // given SA index i3 and identity length L3, return the index of the farthest element with the same length, starting from i1,L1 or i1a,L1a, or i1b,L1b

    bool compRes;

    if (L1<L3) { //search between i1 and i3
        L1b=L1; i1b=i1; i1a=i3;
    }
    else {
        if (L1a<L1) {//search between i1a and i1b, else: search bewtween i1a and i1b
            L1b=L1a; i1b=i1a; i1a=i1;
        };
    };

    // Prefetch SA entry for first midpoint
    if ((i1b+1<i1a)|(i1b>i1a+1)) {
        uint i1c_next=medianUint2(i1a,i1b);
        __builtin_prefetch(mapGen.SA.charArray + (uint64_t)i1c_next * mapGen.SA.wordLength / 8, 0, 0);
    }

    while ( (i1b+1<i1a)|(i1b>i1a+1) ) { //L1a is the target length, i1a...i1b is the initial range, i1c,L1c is the value in the middle
        uint i1c=medianUint2(i1a,i1b);

#ifdef SA_SPECULATIVE_PREFETCH
        // Dual speculative prefetch: both possible next midpoints BEFORE comparison.
        // compareSeqToGenome takes ~300 cycles (SA + genome DRAM loads), giving
        // enough prefetch time for the correct child to arrive in cache.
        //   if L1c == L3 → i1a=i1c → next mid = median(i1c, i1b)
        //   if L1c <  L3 → i1b=i1c → next mid = median(i1a, i1c)
        {
            const size_t wl = mapGen.SA.wordLength;
            char* sa = mapGen.SA.charArray;
            __builtin_prefetch(sa + (uint64_t)medianUint2(i1c, i1b) * wl/8, 0, 0);
            __builtin_prefetch(sa + (uint64_t)medianUint2(i1a, i1c) * wl/8, 0, 0);
        }
#endif

        uint L1c=compareSeqToGenome(mapGen,s,S,L3,L1b,i1c,dirR,compRes);
        if (L1c==L3) {
            i1a=i1c;
        }
        else { //L1c<L3, move i1c
            i1b=i1c;L1b=L1c;
        };

#ifndef SA_SPECULATIVE_PREFETCH
        // Legacy single post-iteration prefetch (current next midpoint, ~10 cycles before use)
        if ((i1b+1<i1a)|(i1b>i1a+1)) {
            uint i1c_next=medianUint2(i1a,i1b);
            __builtin_prefetch(mapGen.SA.charArray + (uint64_t)i1c_next * mapGen.SA.wordLength / 8, 0, 0);
        }
#endif
    };
    return i1a;
};

#ifdef SA_BATCH_FINDRANGE
// Process the two independent findMultRange calls (lo and hi boundary extensions)
// simultaneously, interleaving their SA and genome prefetches.
// When lo comparison runs (~300 cycles for genome load), hi's SA prefetch is in flight,
// and vice versa — effectively halving the serial latency of the two boundary searches.
static void findMultRangePair(
    Genome &mapGen, uint i3, uint L3,
    // lo (i1) side inputs:
    uint lo_i1, uint lo_L1, uint lo_i1a, uint lo_L1a, uint lo_i1b, uint lo_L1b,
    // hi (i2) side inputs:
    uint hi_i1, uint hi_L1, uint hi_i1a, uint hi_L1a, uint hi_i1b, uint hi_L1b,
    char** s, bool dirR, uint S,
    uint& lo_out, uint& hi_out)
{
    // Replicate findMultRange's initial state setup for both sides.
    if (lo_L1 < L3) {
        lo_L1b=lo_L1; lo_i1b=lo_i1; lo_i1a=i3;
    } else if (lo_L1a < lo_L1) {
        lo_L1b=lo_L1a; lo_i1b=lo_i1a; lo_i1a=lo_i1;
    }
    if (hi_L1 < L3) {
        hi_L1b=hi_L1; hi_i1b=hi_i1; hi_i1a=i3;
    } else if (hi_L1a < hi_L1) {
        hi_L1b=hi_L1a; hi_i1b=hi_i1a; hi_i1a=hi_i1;
    }

    bool lo_active = (lo_i1b+1<lo_i1a)|(lo_i1b>lo_i1a+1);
    bool hi_active = (hi_i1b+1<hi_i1a)|(hi_i1b>hi_i1a+1);

    // Prefetch initial midpoints for both searches simultaneously.
    if (lo_active)
        __builtin_prefetch(mapGen.SA.charArray + (uint64_t)medianUint2(lo_i1a,lo_i1b) * mapGen.SA.wordLength/8, 0, 0);
    if (hi_active)
        __builtin_prefetch(mapGen.SA.charArray + (uint64_t)medianUint2(hi_i1a,hi_i1b) * mapGen.SA.wordLength/8, 0, 0);

    while (lo_active || hi_active) {
        bool compRes;
        const size_t wl = mapGen.SA.wordLength;
        char* sa = mapGen.SA.charArray;

        if (lo_active) {
            uint lo_mc = medianUint2(lo_i1a, lo_i1b);
            // Speculative dual prefetch for lo's children.
            // if L1c==L3 → i1a=mc → next mid = median(mc, i1b)
            // if L1c< L3 → i1b=mc → next mid = median(i1a, mc)
            __builtin_prefetch(sa + (uint64_t)medianUint2(lo_mc,  lo_i1b) * wl/8, 0, 0);
            __builtin_prefetch(sa + (uint64_t)medianUint2(lo_i1a, lo_mc)  * wl/8, 0, 0);
            // Also prefetch hi's current midpoint so it arrives during lo comparison.
            if (hi_active)
                __builtin_prefetch(sa + (uint64_t)medianUint2(hi_i1a,hi_i1b) * wl/8, 0, 0);

            uint lo_Lc = compareSeqToGenome(mapGen, s, S, L3, lo_L1b, lo_mc, dirR, compRes);
            if (lo_Lc == L3) { lo_i1a = lo_mc; }
            else              { lo_i1b = lo_mc; lo_L1b = lo_Lc; }
            lo_active = (lo_i1b+1<lo_i1a)|(lo_i1b>lo_i1a+1);
        }

        if (hi_active) {
            uint hi_mc = medianUint2(hi_i1a, hi_i1b);
            // Speculative dual prefetch for hi's children.
            __builtin_prefetch(sa + (uint64_t)medianUint2(hi_mc,  hi_i1b) * wl/8, 0, 0);
            __builtin_prefetch(sa + (uint64_t)medianUint2(hi_i1a, hi_mc)  * wl/8, 0, 0);

            uint hi_Lc = compareSeqToGenome(mapGen, s, S, L3, hi_L1b, hi_mc, dirR, compRes);
            if (hi_Lc == L3) { hi_i1a = hi_mc; }
            else              { hi_i1b = hi_mc; hi_L1b = hi_Lc; }
            hi_active = (hi_i1b+1<hi_i1a)|(hi_i1b>hi_i1a+1);
        }
    }

    lo_out = lo_i1a;
    hi_out = hi_i1a;
}
#endif // SA_BATCH_FINDRANGE

uint maxMappableLength(Genome &mapGen, char** s, uint S, uint N, uint i1, uint i2, bool dirR, uint& L, uint* indStartEnd)
{
    /* find minimum mappable length of sequence s to the genome g with suffix array SA; length(s)=N; [i1 i2] is initial suffix array search bounds.
     * returns number of mappings (1=unique);range indStartEnd; min mapped length = L
     * binary search in SA space
     */

    bool compRes;

    uint L1,L2,i3,L3,L1a,L1b,L2a,L2b,i1a,i1b,i2a,i2b;

    L1=compareSeqToGenome(mapGen,s,S,N,L,i1,dirR,compRes);
    L2=compareSeqToGenome(mapGen,s,S,N,L,i2,dirR,compRes);

    L= min(L1,L2);

    L1a=L1;L1b=L1;i1a=i1;i1b=i1;
    L2a=L2;L2b=L2;i2a=i2;i2b=i2;   // track boundaries of best matching suffix array ranges

    i3=i1;L3=L1; //in case i1+1>=i2 and no iteration of the loop below is ever made

    // Prefetch SA for first midpoint
    if (i1+1<i2) {
        uint i3_next=medianUint2(i1,i2);
        __builtin_prefetch(mapGen.SA.charArray + (uint64_t)i3_next * mapGen.SA.wordLength / 8, 0, 0);
    }

    while (i1+1<i2) {//main binary search loop
        i3=medianUint2(i1,i2);

#ifdef SA_SPECULATIVE_PREFETCH
        // Dual speculative prefetch: both possible next midpoints BEFORE comparison.
        // compareSeqToGenome stalls ~300 cycles on the genome DRAM load, hiding latency.
        //   compRes=true  → i1=i3 → next mid = median(i3, i2)
        //   compRes=false → i2=i3 → next mid = median(i1, i3)
        {
            const size_t wl = mapGen.SA.wordLength;
            char* sa = mapGen.SA.charArray;
            __builtin_prefetch(sa + (uint64_t)medianUint2(i3, i2) * wl/8, 0, 0);
            __builtin_prefetch(sa + (uint64_t)medianUint2(i1, i3) * wl/8, 0, 0);
        }
#endif

        L3=compareSeqToGenome(mapGen,s,S,N,L,i3,dirR,compRes);

        if (L3==N) break; //found exact match, exit the binary search

        if (compRes) { //move 1 to 3
            if (L3>L1) {
               L1b=L1a; L1a=L1; i1b=i1a; i1a=i1;
            };
            i1=i3;L1=L3;
        }
        else {
            if (L3>L2) { //move 2 to 3
               L2b=L2a; L2a=L2; i2b=i2a; i2a=i2;
            };
            i2=i3;L2=L3;
        };
        L= min(L1,L2);

#ifndef SA_SPECULATIVE_PREFETCH
        // Legacy single post-iteration prefetch
        if (i1+1<i2) {
            uint i3_next=medianUint2(i1,i2);
            __builtin_prefetch(mapGen.SA.charArray + (uint64_t)i3_next * mapGen.SA.wordLength / 8, 0, 0);
        }
#endif
    };

    if (L3<N) {//choose longest alignment length between L1 and L2
        if (L1>L2) {
            i3=i1;L3=L1;
        } else {
            i3=i2;L3=L2;
        };
    };

#ifdef SA_BATCH_FINDRANGE
    // Process both boundary extensions simultaneously, interleaving their memory accesses
    // so hi's SA/genome loads overlap with lo's comparison time (and vice versa).
    {
        uint lo_out, hi_out;
        findMultRangePair(mapGen, i3, L3,
            i1, L1, i1a, L1a, i1b, L1b,
            i2, L2, i2a, L2a, i2b, L2b,
            s, dirR, S, lo_out, hi_out);
        i1 = lo_out; i2 = hi_out;
    }
#else
    i1=findMultRange(mapGen,i3,L3,i1,L1,i1a,L1a,i1b,L1b,s,dirR,S);
    i2=findMultRange(mapGen,i3,L3,i2,L2,i2a,L2a,i2b,L2b,s,dirR,S);
#endif

    L=L3; //output
    indStartEnd[0]=i1;
    indStartEnd[1]=i2;

    return i2-i1+1;
};


int compareRefEnds (Genome &mapGen, uint64 SAstr,  uint64 gInsert, bool strG, bool strR)
{
    if ( strG)
    {// + strand g
       return strR ? (SAstr < gInsert ? 1:-1) : 1;
    } else
    {// - strand g
       return strR ? -1 : ( gInsert==-1LLU ? -1 : ( SAstr < mapGen.nGenome-gInsert ? 1:-1) );
    };
};

uint compareSeqToGenome1(Genome &mapGen, char** s2, uint S, uint N, uint L, uint iSA, bool dirR, uint64 gInsert, int & compRes)
{
    /* compare s to g, find the maximum identity length
     * s2[0] read sequence; s2[1] complementary sequence
     * S position to start search from in s2[0],s2[1]
     * dirR: strand of the s
     * different treatment of 5 (spacer) in the sequence and genome
     * 5 is allowed in the sequence
     * 5 in the genome is < than 5 in the sequence
     */

    //TODO no need for complementary sequence

    register int64 ii;

    uint SAstr=mapGen.SA[iSA];
    bool dirG = (SAstr>>mapGen.GstrandBit) == 0; //forward or reverse strand of the genome
    SAstr &= mapGen.GstrandMask;
    char *g=mapGen.G;

    if (dirG) {//forward on read, forward on genome
        char* s  = s2[0] + S + L;
        g += SAstr + L;
        for (ii=0;(uint) ii < N-L; ii++)
        {
            if (s[ii]!=g[ii])
            {
                if (s[ii]>g[ii])
                {
                    compRes=1;
                    return ii+L;
                } else
                {
                    compRes=-1;
                    return ii+L;
                };
            } else if (s[ii]==GENOME_spacingChar)
            {//this already implies the s[ii]==g[ii]
                compRes=compareRefEnds (mapGen, SAstr, gInsert, dirG, dirR);
                return ii+L;
            };
        };
//         if (s[ii]>g[ii]) {compRes=true;} else {compRes=false;};
        return N; //exact match
    }
    else
    {
        char* s  = s2[1] + S + L;
        g += mapGen.nGenome-1-SAstr - L;
        for (ii=0; (uint) ii < N-L; ii++)
        {
            if (s[ii]!=g[-ii])
            {
                char s1=s[ii],g1=g[-ii];
                if (s1<4) s1=3-s1;
                if (g1<4) g1=3-g1;
                if (s1>g1) {
                    compRes=1;
                    return ii+L;
                } else
                {
                    compRes=-1;
                    return ii+L;
                };
                break;
            } else if (s[ii]==GENOME_spacingChar)
            {//this already implies the s[ii]==g[ii]
                compRes=compareRefEnds (mapGen, SAstr, gInsert, dirG, dirR);
                return ii+L;
            };
        };
        return N;
    };
};


uint suffixArraySearch1(Genome &mapGen, char** s, uint S, uint N, uint64 gInsert, bool strR, uint i1, uint i2, uint L)
{
    /* binary search in SA space
     * s[0],s[1] - query sequence, complementary sequence
     * S - start offset
     * N - sequence length
     * g - genome sequence
     * gInsert - index where the sequence insertion happened
     * SA - suffix array
     * strR - strand of the query sequence
     * i1,i2 = starting indices in SA
     * L - starting length
     * output: first SA index > searched string, i.e. g[SA[index-1]]<s<g[SA[index]]
     */

    int compRes;

    uint L1=compareSeqToGenome1(mapGen,s,S,N,L,i1,strR,gInsert,compRes);
    if (compRes<0)
    {// the sequence is smaller than the first index of the SA, cannot proceed
        L=L1;
        return 0;
    };

    uint L2=compareSeqToGenome1(mapGen, s,S,N,L,i2,strR,gInsert,compRes);
    if (compRes>0)
    {//the sequence is bigger than the last SA index, return a huge number
        L=L2;
        return -2llu;
    };

    L=min(L1,L2);

    uint i3=i1,L3=L1; //in case i1+1>=i2 an not iteration of the loope below is ever made
    while (i1+1<i2) {//main binary search loop
        i3=medianUint2(i1,i2);
        L3=compareSeqToGenome1(mapGen,s,S,N,L,i3,strR,gInsert,compRes);//cannot do this because these sj sequences contains spacers=5
        if (L3==N) {//this should not really happen
            L=N;
            return i3;
//             cerr << "Bug L3==N"<<endl;
//             exit(-1); //found exact match of the whole read length, exit the binary search
        };

        if (compRes>0)
        { //move 1 to 3
            i1=i3;L1=L3;
        } else if (compRes<0)
        {//move 2 to 3
            i2=i3;L2=L3;
        }
        L= min(L1,L2);
    };
    return i2; //index at i2 is always bigger than the sequence
};

uint funCalcSAiFromSA(char* gSeq, PackedArray& gSA, Genome &mapGen, uint iSA, int L, int & iL4)
{
    uint SAstr=gSA[iSA];
    bool dirG = (SAstr>>mapGen.GstrandBit) == 0; //forward or reverse strand of the genome
    SAstr &= mapGen.GstrandMask;
    iL4=-1;
    register uint saind=0;
    if (dirG)
    {
        register uint128 g1=*( (uint128*) (gSeq+SAstr) );
        for (int ii=0; ii<L; ii++)
        {
            register char g2=(char) g1;
            if (g2>3)
            {
                iL4=ii;
                saind <<= 2*(L-ii);
                return saind;
            };
            saind=saind<<2;
            saind+=g2;
            g1=g1>>8;
        };
        return saind;
    } else
    {
        register uint128 g1=*( (uint128*) (gSeq+mapGen.nGenome-SAstr-16) );
        for (int ii=0; ii<L; ii++)
        {
            register char g2=(char) (g1>>(8*(15-ii)));
            if (g2>3)
            {
                iL4=ii;
                saind <<= 2*(L-ii);
                return saind;
            };
            saind=saind<<2;
            saind+=3-g2;
        };
        return saind;
    };

};

int64 funCalcSAi(char *G, uint iL)
{
    int64 ind1=0;
    for (uint iL1=0;iL1<=iL;iL1++) {
        uint g=(uint) G[iL1];
        if (g>3) {
            return -ind1;
        } else {
            ind1 <<= 2;
            ind1 += g;
        };
   };
   return ind1;
};
