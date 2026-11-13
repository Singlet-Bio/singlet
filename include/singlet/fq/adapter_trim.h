// SPDX-License-Identifier: MIT
#pragma once
// 3' adapter trimming of a cDNA read (numeric bases A0 C1 G2 T3, 4 = N).
//
// A short insert is read through into the library's other end. The read then holds a few cDNA bases and the adapter, which
// STAR cannot place (a 54-nt inDrop read with a 12-nt insert is 78% adapter). The cut is the first position where an adapter
// or a prefix of it, running off the end of the read, matches: at least 8 overlapping bases, one mismatch from 12 and two from
// 20 overlapping bases. N counts as a mismatch.
#include <cstdint>
#include <string>
#include <vector>

namespace singlet::fq {

struct AdapterSet {
    std::vector<std::vector<uint8_t>> seqs;
    int min_overlap = 8;
    bool empty() const { return seqs.empty(); }
    static std::vector<uint8_t> from_string(const std::string& s) {
        std::vector<uint8_t> v;
        for (char c : s) v.push_back(c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4);
        return v;
    }
    /// Length to keep: n when no adapter is found.
    uint16_t cut(const uint8_t* s, uint16_t n) const {
        uint16_t best = n;
        for (const auto& a : seqs) {
            const int L = static_cast<int>(a.size());
            for (int i = 0; i + min_overlap <= n && i < best; ++i) {
                const int ov = std::min(L, static_cast<int>(n) - i);
                if (ov < min_overlap) break;
                const int allow = ov >= 20 ? 2 : ov >= 12 ? 1 : 0;
                int mm = 0;
                for (int j = 0; j < ov && mm <= allow; ++j) mm += (s[i + j] != a[static_cast<size_t>(j)] || s[i + j] > 3);
                if (mm <= allow) { best = static_cast<uint16_t>(i); break; }
            }
        }
        return best;
    }
};

}  // namespace singlet::fq
