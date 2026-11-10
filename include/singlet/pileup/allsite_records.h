// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: allsite_records.h
// The all-site counting pass (allsite.h) as sorted records: one 64-bit key per (read, site) hit, grouped by site, sorted, and collapsed
// into molecules with the panel's rules (SnpConsensus R1 vote, R9 directional collapse per (barcode, site)). A host kernel and a device
// kernel that produce the same sorted keys per site get the same counts from collapse_site().
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

struct Scratch {
    struct Ent { uint64_t umi; uint32_t reads; uint32_t n[3]; };
    std::vector<Ent> e;
    std::vector<uint32_t> root;
    std::vector<uint32_t> merged;
};

// One site's keys k[0, n), sorted ascending. emit(bc, dp, ad) once per barcode with at least one REF or ALT molecule (dp >= 1, ad <= dp):
// the number of molecules whose strict majority is REF or ALT, and of those the ALT ones. st counts every molecule as allsite.h does.
template <class Emit>
inline void collapse_site(const uint64_t* k, size_t n, const Layout& y, bool directional, Scratch& S, Counts& st, Emit&& emit) {
    uint32_t dp = 0, ad = 0;
    auto vote = [&](const uint32_t* c) {
        uint8_t best = 0;
        for (uint8_t i = 1; i < 3; ++i) if (c[i] > c[best]) best = i;
        for (uint8_t i = 0; i < 3; ++i) if (i != best && c[i] == c[best]) { ++st.ambiguous; return; }
        if (best == 2) { ++st.other; return; }
        ++st.molecules; ++dp; if (best == 1) ++ad;
    };
    const int sh = y.bc_shift;
    for (size_t g = 0; g < n;) {   // one barcode
        const uint64_t bc = k[g] >> sh;
        size_t ge = g; while (ge < n && (k[ge] >> sh) == bc) ++ge;
        dp = 0; ad = 0;
        if (!y.umi) {
            for (size_t i = g; i < ge; ++i) { const uint8_t c = k[i] & 3; if (c == 2) ++st.other; else { ++st.molecules; ++dp; if (c == 1) ++ad; } }
        } else {
            S.e.clear();
            for (size_t i = g; i < ge;) {   // one UMI: class counts saturated at 255 (SnpConsensus's uint8 counts)
                const uint64_t u = k[i] >> 2; uint32_t c[3] = {0, 0, 0};
                while (i < ge && (k[i] >> 2) == u) { ++c[k[i] & 3]; ++i; }
                Scratch::Ent en; en.umi = (u & (y.umi_lead - 1)) | y.umi_lead;
                for (int j = 0; j < 3; ++j) en.n[j] = std::min<uint32_t>(c[j], 255);
                en.reads = en.n[0] + en.n[1] + en.n[2];
                S.e.push_back(en);
            }
            const size_t m = S.e.size();
            if (m == 1 || !directional) {
                for (const auto& en : S.e) vote(en.n);
            } else {
                // r9_molecules with no gene store: UMIs by reads descending then UMI, directional roots, class counts summed per root
                std::sort(S.e.begin(), S.e.end(), [](const Scratch::Ent& a, const Scratch::Ent& b) { return a.reads != b.reads ? a.reads > b.reads : a.umi < b.umi; });
                umi3_directional_roots(m, [&](size_t i) { return S.e[i].umi; }, [&](size_t i) { return S.e[i].reads; }, S.root);
                S.merged.assign(3 * m, 0);
                for (size_t i = 0; i < m; ++i) for (int j = 0; j < 3; ++j) S.merged[3 * S.root[i] + j] += S.e[i].n[j];
                for (size_t i = 0; i < m; ++i) if (S.root[i] == i) vote(&S.merged[3 * i]);
            }
        }
        if (dp) emit(static_cast<uint32_t>(bc), dp, ad);
        g = ge;
    }
}

}}}  // namespace singlet::allsite::rec
