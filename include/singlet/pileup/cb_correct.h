// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: cb_correct.h
// order-free-rules R11: STAR's CB_UMI_Simple whitelist correction, 1MM_multi_Nbase_pseudocounts, for the unsorted stream,
// which has no corrected CB tag. Same whitelist, same rules, same float posterior as STAR 2.7.11b:
//   whitelist     2-bit values, sorted and de-duplicated; an entry with an N at position 0 is skipped, an N elsewhere
//                 reads as A (ParametersSolo.cpp, convertNuclStrToInt64)
//   two or more N rejected (-2)
//   no N          exact hit -> that entry (status 0); else every single substitution, last base first, codes XOR 1..3
//   one N         the four bases at the N in order A C G T (the N packs as A), every hit kept (Nbase)
//   no hit -1; one hit taken as is (pseudocount type: no oneExact rule)
//   several hits  posterior: count = exact reads + 1 (pseudocount), q = min(qual - 33, 33),
//                 pin = (float)(count * pow(10, -q / 10)), ptot and pmax float, first candidate wins ties,
//                 accept when ptot > 0 and pmax >= 0.975f * ptot (else -4)
//                 (SoloReadBarcode_getCBandUMI.cpp, SoloReadFeature_inputRecords.cpp, ParametersSolo.cpp:287-289)
// The exact counts are STAR's cbReadCountExact: one per read whose CB is an exact whitelist hit and whose UMI passes the
// UMI check, over the whole sample. singlet fills them in pass 1 of the .1fq, before STAR starts, so a read's correction
// never depends on which reads came before it.
// Barcode-read quality: singlet writes R1 with quality 'F' for every base (singlet.cpp, R1 record), so q = 33 for every
// candidate and the posterior reduces to counts; resolve() takes a quality string anyway.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace singlet {

class CbCorrector {
public:
    enum Status : int { kExact = 0, kOneCandidate = 1, kPosterior = 2,
                        kNoMatch = -1, kTwoN = -2, kRejected = -4 };

    /// Loads STAR's whitelist. False when the file is unreadable, empty, or the barcodes are not all `len` long.
    bool load_whitelist(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        while (std::getline(f, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
            if (line.empty()) continue;
            if (len_ == 0) len_ = static_cast<int>(line.size());
            if (static_cast<int>(line.size()) != len_ || len_ > 31) return false;
            uint64_t v = 0;
            if (pack(line.c_str(), len_, v) == 0) continue;   // N at position 0: skipped, as STAR does
            wl_.push_back(v);
        }
        std::sort(wl_.begin(), wl_.end());
        wl_.erase(std::unique(wl_.begin(), wl_.end()), wl_.end());
        exact_.assign(wl_.size(), 0);
        return !wl_.empty();
    }

    int length() const { return len_; }
    size_t size() const { return wl_.size(); }
    bool loaded() const { return !wl_.empty(); }

    /// Whitelist index of an N-free barcode, or -1.
    int64_t find(const char* cb) const {
        uint64_t v = 0;
        if (pack(cb, len_, v) != -1) return -1;
        return index_of(v);
    }

    /// STAR's exact histogram (pass 1): `n` reads with this exact whitelist barcode and a good UMI.
    void add_exact(int64_t wl_index, uint64_t n) {
        if (wl_index >= 0 && static_cast<size_t>(wl_index) < exact_.size()) exact_[wl_index] += n;
    }
    uint64_t exact_count(int64_t wl_index) const { return exact_[wl_index]; }
    /// The whitelist barcode at `wl_index` (len() chars, no terminator).
    void barcode(int64_t wl_index, char* out) const { unpack(wl_[wl_index], out); }

    /// Corrects `cb` (len() bases). On status >= 0 writes the whitelist barcode (len() chars, no terminator) to `out`.
    /// `qual` may be null: every base then counts as singlet's 'F'.
    int resolve(const char* cb, const char* qual, char* out) const {
        uint64_t v = 0;
        const int64_t posN = pack(cb, len_, v);
        if (posN == -2) return kTwoN;
        if (posN == -1) {
            const int64_t e = index_of(v);
            if (e >= 0) { unpack(wl_[e], out); return kExact; }
        }
        Cand cand_[3 * 32];   // at most 3 per base
        size_t nc = 0;
        if (posN >= 0) {
            const uint32_t shift = 2 * static_cast<uint32_t>(len_ - 1 - posN);
            for (uint64_t jj = 0; jj < 4; ++jj) {
                const int64_t c = index_of(v ^ (jj << shift));
                if (c >= 0) cand_[nc++] = {c, qual ? qual[posN] : 'F'};
            }
        } else {
            for (int ii = 0; ii < len_; ++ii)
                for (uint64_t jj = 1; jj < 4; ++jj) {
                    const int64_t c = index_of(v ^ (jj << (2 * ii)));
                    if (c >= 0) cand_[nc++] = {c, qual ? qual[len_ - 1 - ii] : 'F'};
                }
        }
        if (nc == 0) return kNoMatch;
        if (nc == 1) { unpack(wl_[cand_[0].idx], out); return kOneCandidate; }
        float ptot = 0.0f, pmax = 0.0f, pin;
        int64_t best = -1;
        for (size_t k = 0; k < nc; ++k) {
            const Cand& c = cand_[k];
            const uint64_t total = exact_[c.idx] + 1;   // pseudocount
            char qin = static_cast<char>(c.q - kQSbase);
            qin = qin < kQSmax ? qin : kQSmax;
            pin = static_cast<float>(static_cast<double>(total) * std::pow(10.0, -qin / 10.0));
            ptot += pin;
            if (pin > pmax) { best = c.idx; pmax = pin; }
        }
        if (ptot > 0.0f && pmax >= kCbMinP * ptot) { unpack(wl_[best], out); return kPosterior; }
        return kRejected;
    }

private:
    static constexpr char kQSbase = 33, kQSmax = 33;
    static constexpr float kCbMinP = 0.975f;
    struct Cand { int64_t idx; char q; };

    /// STAR's convertNuclStrToInt64: returns -1 (no N), the N's position, or -2 (two or more N).
    static int64_t pack(const char* s, int len, uint64_t& v) {
        v = 0;
        int64_t posN = -1;
        for (int i = 0; i < len; ++i) {
            uint64_t nt;
            switch (s[i]) { case 'A': case 'a': nt = 0; break; case 'C': case 'c': nt = 1; break;
                            case 'G': case 'g': nt = 2; break; case 'T': case 't': nt = 3; break; default: nt = 4; }
            if (nt > 3) {
                if (posN >= 0) return -2;
                posN = i;
                nt = 0;
            }
            v = (v << 2) + nt;
        }
        return posN;
    }
    void unpack(uint64_t v, char* out) const {
        static constexpr char kNt[4] = {'A', 'C', 'G', 'T'};
        for (int i = len_ - 1; i >= 0; --i) { out[i] = kNt[v & 3]; v >>= 2; }
    }
    int64_t index_of(uint64_t v) const {
        auto it = std::lower_bound(wl_.begin(), wl_.end(), v);
        return (it != wl_.end() && *it == v) ? static_cast<int64_t>(it - wl_.begin()) : -1;
    }

    int len_ = 0;
    std::vector<uint64_t> wl_;
    std::vector<uint64_t> exact_;
};

} // namespace singlet
