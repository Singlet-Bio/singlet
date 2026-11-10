// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: allsite_records.h
// The all-site counting pass (allsite.h) as sorted records: one 64-bit key per (read, site) hit, grouped by site, sorted, and collapsed
// into molecules with the panel's rules (SnpConsensus R1 vote, R9 directional collapse per (barcode, site)). A host kernel and a device
// kernel that produce the same keys per site get the same counts from collapse_site() (sorted) or collapse_site_unsorted().
//
// Key of one hit at one site (site index kept beside the key, not in it):
//   with UMIs:    key = bc << (3L + 2) | umi << 2 | cls      L = UMI length (1..20), umi = L 3-bit codes, first base in the high bits
//                                                           (A0 C1 G2 T3, anything else 4: umi_pack3 without its leading 1 bit)
//   without UMIs: key = bc << 2 | cls                       (every hit is its own molecule)
//   cls: 0 REF, 1 ALT, 2 other (SnpConsensus::base_class)
// Within a site, sorting the keys ascending puts each barcode's hits together, each UMI's hits together inside it, and the classes last.
// collapse_site() needs exactly that order; the order of sites and of barcodes inside a site changes no count.
#include <algorithm>
#include <cstdint>
#include <vector>

#include "umi_dedup.h"

namespace singlet { namespace allsite { namespace rec {

struct Layout {
    bool ok = false;        // the key fits 64 bits (else the caller keeps the hash-map kernel)
    bool umi = false;       // keys carry a UMI
    int L = 0;              // UMI length
    int bc_shift = 2;       // key >> bc_shift = barcode
    uint64_t umi_lead = 0;  // 1 << 3L: the leading bit umi_pack3 puts before the codes
};

inline Layout layout(bool has_umi, int umi_len, uint32_t n_barcodes) {
    Layout y;
    int bc_bits = 1; while (bc_bits < 32 && (1ull << bc_bits) < static_cast<uint64_t>(n_barcodes)) ++bc_bits;
    y.umi = has_umi && umi_len > 0;
    if (y.umi) {
        if (umi_len > 20) return y;
        y.L = umi_len; y.bc_shift = 3 * umi_len + 2; y.umi_lead = 1ull << (3 * umi_len);
    }
    y.ok = y.bc_shift + bc_bits <= 64;
    return y;
}

// ReadRec::umi3 (3 bits per character, first character in the low bits) as the key's UMI field: umi_pack3 of the characters allsite.h
// gives SnpConsensus (A C G T for codes 0-3, N for 4-7), without the leading bit
inline uint64_t umi_field(uint64_t u3, int L) {
    uint64_t p = 0;
    for (int i = 0; i < L; ++i) { const uint64_t c = (u3 >> (3 * i)) & 7; p = (p << 3) | (c < 4 ? c : 4); }
    return p;
}

inline uint64_t key(const Layout& y, uint32_t bc, uint64_t umi_f, uint8_t cls) {
    return (static_cast<uint64_t>(bc) << y.bc_shift) | (y.umi ? (umi_f << 2) : 0) | cls;
}


struct Counts { uint64_t molecules = 0, ambiguous = 0, other = 0; };   // SnpConsensus's snp_molecules, _ambiguous, _other

// One exact (barcode, site, UMI) molecule: UMI with umi_pack3's leading bit, class counts saturated at 255, reads = their sum
struct Ent { uint64_t umi; uint32_t bc, reads; uint32_t n[3]; };

struct Scratch {
    std::vector<Ent> e, e2;
    std::vector<uint32_t> root, merged;
    struct Slot { uint64_t k; uint8_t c[3]; };
    std::vector<Slot> tab;
};

// The group interface: the molecules of one (barcode, site), in any order. Adds the barcode's REF-or-ALT molecules to dp (ALT ones to ad)
// and every molecule to st, as SnpConsensus::finalize with R9 (r9_molecules without a gene store): UMIs by reads descending then UMI,
// directional roots, class counts summed per root, strict-majority vote.
inline void collapse_group(Ent* p, size_t m, bool directional, Scratch& S, Counts& st, uint32_t& dp, uint32_t& ad) {
    auto vote = [&](const uint32_t* c) {
        uint8_t best = 0;
        for (uint8_t i = 1; i < 3; ++i) if (c[i] > c[best]) best = i;
        for (uint8_t i = 0; i < 3; ++i) if (i != best && c[i] == c[best]) { ++st.ambiguous; return; }
        if (best == 2) { ++st.other; return; }
        ++st.molecules; ++dp; if (best == 1) ++ad;
    };
    if (m == 1 || !directional) { for (size_t i = 0; i < m; ++i) vote(p[i].n); return; }
    std::sort(p, p + m, [](const Ent& a, const Ent& b) { return a.reads != b.reads ? a.reads > b.reads : a.umi < b.umi; });
    umi3_directional_roots(m, [&](size_t i) { return p[i].umi; }, [&](size_t i) { return p[i].reads; }, S.root);
    S.merged.assign(3 * m, 0);
    for (size_t i = 0; i < m; ++i) for (int j = 0; j < 3; ++j) S.merged[3 * S.root[i] + j] += p[i].n[j];
    for (size_t i = 0; i < m; ++i) if (S.root[i] == i) vote(&S.merged[3 * i]);
}

// One site's keys k[0, n), sorted ascending. emit(bc, dp, ad) once per barcode with at least one REF or ALT molecule (dp >= 1, ad <= dp).
template <class Emit>
inline void collapse_site(const uint64_t* k, size_t n, const Layout& y, bool directional, Scratch& S, Counts& st, Emit&& emit) {
    const int sh = y.bc_shift;
    for (size_t g = 0; g < n;) {   // one barcode
        const uint64_t bc = k[g] >> sh;
        size_t ge = g; while (ge < n && (k[ge] >> sh) == bc) ++ge;
        uint32_t dp = 0, ad = 0;
        if (!y.umi) {   // every hit is its own molecule
            for (size_t i = g; i < ge; ++i) { const uint8_t c = k[i] & 3; if (c == 2) ++st.other; else { ++st.molecules; ++dp; if (c == 1) ++ad; } }
        } else {
            S.e.clear();
            for (size_t i = g; i < ge;) {   // one UMI
                const uint64_t u = k[i] >> 2; uint32_t c[3] = {0, 0, 0};
                while (i < ge && (k[i] >> 2) == u) { ++c[k[i] & 3]; ++i; }
                Ent en; en.umi = (u & (y.umi_lead - 1)) | y.umi_lead; en.bc = static_cast<uint32_t>(bc);
                for (int j = 0; j < 3; ++j) en.n[j] = std::min<uint32_t>(c[j], 255);
                en.reads = en.n[0] + en.n[1] + en.n[2];
                S.e.push_back(en);
            }
            collapse_group(S.e.data(), S.e.size(), directional, S, st, dp, ad);
        }
        if (dp) emit(static_cast<uint32_t>(bc), dp, ad);
        g = ge;
    }
}

// The same result from one site's keys in any order (the host kernel): exact molecules through a small hash table, grouped by barcode with
// a radix pass over the barcode bits, then collapse_group per barcode. Barcodes come out ascending. Small sites and keys without UMIs are
// sorted in place and go through collapse_site.
template <class Emit>
inline void collapse_site_unsorted(uint64_t* k, size_t n, const Layout& y, bool directional, Scratch& S, Counts& st, Emit&& emit) {
    if (!y.umi || n < 48) { std::sort(k, k + n); collapse_site(k, n, y, directional, S, st, emit); return; }
    int bits = 6; while ((1ull << bits) < 2 * n) ++bits;
    const size_t cap = size_t(1) << bits, mask = cap - 1;
    S.tab.assign(cap, Scratch::Slot{0, {0, 0, 0}});
    for (size_t i = 0; i < n; ++i) {   // key >> 2 + 1: never 0, so 0 marks an empty slot
        const uint64_t mk = (k[i] >> 2) + 1; const uint8_t c = k[i] & 3;
        size_t h = static_cast<size_t>((mk * 0x9E3779B97F4A7C15ull) >> (64 - bits));
        while (S.tab[h].k != mk && S.tab[h].k != 0) h = (h + 1) & mask;
        Scratch::Slot& sl = S.tab[h]; sl.k = mk; if (sl.c[c] < 255) ++sl.c[c];
    }
    S.e.clear();
    const int ub = 3 * y.L; uint32_t bor = 0, band = ~0u;
    for (const auto& sl : S.tab) {
        if (!sl.k) continue;
        const uint64_t u = sl.k - 1;
        Ent en; en.umi = (u & (y.umi_lead - 1)) | y.umi_lead; en.bc = static_cast<uint32_t>(u >> ub);
        en.n[0] = sl.c[0]; en.n[1] = sl.c[1]; en.n[2] = sl.c[2]; en.reads = en.n[0] + en.n[1] + en.n[2];
        bor |= en.bc; band &= en.bc;
        S.e.push_back(en);
    }
    // LSD radix by barcode over the bytes that differ
    const uint32_t diff = bor ^ band; S.e2.resize(S.e.size());
    for (int sh = 0; sh < 32; sh += 8) {
        if (!((diff >> sh) & 0xff)) continue;
        uint32_t cnt[257] = {0};
        for (const auto& en : S.e) ++cnt[((en.bc >> sh) & 0xff) + 1];
        for (int i = 0; i < 256; ++i) cnt[i + 1] += cnt[i];
        for (const auto& en : S.e) S.e2[cnt[(en.bc >> sh) & 0xff]++] = en;
        S.e.swap(S.e2);
    }
    const size_t m = S.e.size();
    for (size_t g = 0; g < m;) {
        size_t ge = g + 1; while (ge < m && S.e[ge].bc == S.e[g].bc) ++ge;
        uint32_t dp = 0, ad = 0;
        collapse_group(S.e.data() + g, ge - g, directional, S, st, dp, ad);
        if (dp) emit(S.e[g].bc, dp, ad);
        g = ge;
    }
}

}}}  // namespace singlet::allsite::rec
