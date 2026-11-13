// SPDX-License-Identifier: MIT
#pragma once
// inDrop barcode read with a variable-length first barcode.
//
// The barcode read of inDrop v1/v2 (and the 44-60 nt barcode read of v3) is
//     [BC1, 8-11 nt] [W1 linker, 22 nt: GAGTGATTGCTTGTGACGCCTT] [BC2, 8 nt] [UMI, 6 nt] [poly(T)]
// BC1 comes in four lengths (96 barcodes of each) and is the only part whose position moves. A fixed 10-nt window takes
// the right bases from one read in four. The W1 linker is the anchor: STARsolo's inDrop recipe uses the same one
// (--soloAdapterSequence GAGTGATTGCTTGTGACGCCTT, --soloCBposition 0_0_2_-1 3_1_3_8, --soloUMIposition 3_9_3_14,
// STARsolo is MIT licensed). This header does the anchoring at encode time, so every later stage keeps fixed positions:
//     canonicalize() finds W1 (up to kMaxW1Mismatch mismatches, BC1 length 7-12), corrects BC1 and BC2 to the whitelists
//     (exact or one unambiguous substitution) and writes a fixed 46-nt read
//     [BC1 in 11 slots, right-padded] [BC2, 8] [UMI, 6] [rest of the read, N-filled to 46]
// Reads without a resolvable barcode get N in the barcode and UMI slots (counted, never counted as a cell barcode).
// Positions of the canonical read (soloCBposition 0_0_0_10, 0_11_0_18; soloUMIposition 0_19_0_24) go into the .1fq metadata.
#include <array>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace singlet::fq {

class IndropAnchor {
   public:
    static constexpr int kCanonLen = 46;
    static constexpr int kBc1Slots = 11;
    static constexpr int kBc2Len = 8;
    static constexpr int kUmiLen = 6;
    static constexpr int kW1Len = 22;
    static constexpr int kMaxW1Mismatch = 3;
    static constexpr int kBc1MinLen = 7, kBc1MaxLen = 12;   // W1 offsets searched (8-11 are real; 7 and 12 absorb an indel)
    static constexpr int kCbEnd0 = 10, kBc2Start = 11, kBc2End = 18, kUmiStart = 19, kUmiEnd = 24;   // canonical positions, 0-based inclusive

    enum Fail : int { kOk = 0, kShort = 1, kNoW1 = 2, kBc1 = 3, kBc2 = 4 };

    bool load(const std::vector<std::string>& dirs, const std::string& bc1_file = "indrop_bc1.txt",
              const std::string& bc2_file = "indrop_bc2.txt") {
        for (const auto& d : dirs) {
            std::vector<std::string> b1, b2;
            if (!read_list(d + "/" + bc1_file, b1) || !read_list(d + "/" + bc2_file, b2)) continue;
            if (b1.empty() || b2.empty()) continue;
            build(b1, b2);
            loaded_ = true;
            return true;
        }
        return false;
    }
    bool loaded() const { return loaded_; }

    /// `r`: numeric bases (A0 C1 G2 T3, 4 = N), `n` its length. `out`: kCanonLen bytes. Returns Fail.
    int canonicalize(const uint8_t* r, uint16_t n, uint8_t* out) const {
        for (int i = 0; i < kCanonLen; ++i) out[i] = 4;
        ++total_;
        int best_p = -1, best_mm = kMaxW1Mismatch + 1;
        for (int p = kBc1MinLen; p <= kBc1MaxLen; ++p) {
            if (p + kW1Len + kBc2Len + kUmiLen > n) break;
            int mm = 0;
            for (int j = 0; j < kW1Len && mm < best_mm; ++j) mm += (r[p + j] != kW1[j]);
            if (mm < best_mm) { best_mm = mm; best_p = p; }
        }
        if (best_p < 0) {
            if (n < kBc1MinLen + kW1Len + kBc2Len + kUmiLen) { ++f_short_; return kShort; }
            ++f_now1_;
            return kNoW1;
        }
        const int p = best_p;
        const int i1 = lookup(r, p);
        if (i1 < 0) { ++f_bc1_; return kBc1; }
        const int i2 = lookup2(r + p + kW1Len);
        if (i2 < 0) { ++f_bc2_; return kBc2; }
        const std::vector<uint8_t>& b1 = bc1_[static_cast<size_t>(i1)];
        for (int i = 0; i < kBc1Slots; ++i) out[i] = i < static_cast<int>(b1.size()) ? b1[static_cast<size_t>(i)] : pad_;
        const std::vector<uint8_t>& b2 = bc2_[static_cast<size_t>(i2)];
        for (int i = 0; i < kBc2Len; ++i) out[kBc2Start + i] = b2[static_cast<size_t>(i)];
        const uint8_t* u = r + p + kW1Len + kBc2Len;
        for (int i = 0; i < kUmiLen; ++i) out[kUmiStart + i] = u[i];
        ++ok_;
        return kOk;
    }

    void report(std::ostream& os) const {
        const uint64_t t = total_.load();
        if (t == 0) return;
        auto pc = [&](uint64_t v) { return 100.0 * static_cast<double>(v) / static_cast<double>(t); };
        os << "[1fq-encode] inDrop W1-anchored barcodes: " << t << " reads, " << ok_.load() << " resolved (" << pc(ok_.load())
           << "%); no W1 " << f_now1_.load() << " (" << pc(f_now1_.load()) << "%), BC1 not in list " << f_bc1_.load() << " ("
           << pc(f_bc1_.load()) << "%), BC2 not in list " << f_bc2_.load() << " (" << pc(f_bc2_.load()) << "%), too short "
           << f_short_.load() << "\n";
    }
    void reset_stats() const { total_ = 0; ok_ = 0; f_short_ = 0; f_now1_ = 0; f_bc1_ = 0; f_bc2_ = 0; }
    uint64_t resolved() const { return ok_.load(); }
    uint64_t total() const { return total_.load(); }

   private:
    static constexpr uint8_t kW1[22] = {2, 0, 2, 3, 2, 0, 3, 3, 2, 1, 3, 3, 2, 3, 2, 0, 1, 2, 1, 1, 3, 3};   // GAGTGATTGCTTGTGACGCCTT

    static bool read_list(const std::string& path, std::vector<std::string>& out) {
        std::ifstream f(path);
        if (!f) return false;
        std::string ln;
        while (std::getline(f, ln)) {
            while (!ln.empty() && (ln.back() == '\r' || ln.back() == ' ' || ln.back() == '\t')) ln.pop_back();
            if (!ln.empty()) out.push_back(ln);
        }
        return true;
    }
    static bool num(char c, uint8_t& v) {
        switch (c) {
            case 'A': case 'a': v = 0; return true;
            case 'C': case 'c': v = 1; return true;
            case 'G': case 'g': v = 2; return true;
            case 'T': case 't': v = 3; return true;
            default: return false;
        }
    }
    static uint32_t pack(const uint8_t* b, int len) {   // len <= 11; N reads as A (a single N then costs one mismatch)
        uint32_t v = static_cast<uint32_t>(len) << 24;
        for (int i = 0; i < len; ++i) v |= static_cast<uint32_t>(b[i] < 4 ? b[i] : 0) << (2 * i);
        return v;
    }
    static void add_variants(std::unordered_map<uint32_t, int>& m, const std::unordered_set<uint32_t>& exact,
                             const std::vector<uint8_t>& e, int idx) {
        const int L = static_cast<int>(e.size());
        std::vector<uint8_t> t(e);
        for (int i = 0; i < L; ++i)
            for (uint8_t c = 0; c < 4; ++c) {
                if (c == e[static_cast<size_t>(i)]) continue;
                t[static_cast<size_t>(i)] = c;
                const uint32_t k = pack(t.data(), L);
                if (!exact.count(k)) {   // an exact entry always wins over a substitution
                    auto it = m.find(k);
                    if (it == m.end()) m.emplace(k, idx);
                    else if (it->second != idx) it->second = -2;   // two entries one substitution away: ambiguous, not corrected
                }
                t[static_cast<size_t>(i)] = e[static_cast<size_t>(i)];
            }
    }
    void build(const std::vector<std::string>& l1, const std::vector<std::string>& l2) {
        auto conv = [](const std::vector<std::string>& l, std::vector<std::vector<uint8_t>>& out) {
            for (const auto& s : l) {
                std::vector<uint8_t> v;
                bool ok = true;
                for (char c : s) { uint8_t x; if (!num(c, x)) { ok = false; break; } v.push_back(x); }
                if (ok && v.size() >= 6 && v.size() <= 11) out.push_back(std::move(v));
            }
        };
        bc1_.clear(); bc2_.clear(); m1_.clear(); m2_.clear();
        conv(l1, bc1_);
        conv(l2, bc2_);
        // padding base for BC1: the first of A, C, G, T that keeps the padded strings of all entries distinct
        pad_ = 4;
        for (uint8_t cand = 0; cand < 4 && pad_ == 4; ++cand) {
            std::unordered_map<std::string, int> seen;
            bool clash = false;
            for (const auto& e : bc1_) {
                std::string s;
                for (int i = 0; i < kBc1Slots; ++i)
                    s.push_back(static_cast<char>('0' + (i < static_cast<int>(e.size()) ? e[static_cast<size_t>(i)] : cand)));
                if (++seen[s] > 1) { clash = true; break; }
            }
            if (!clash) pad_ = cand;
        }
        auto fill = [&](std::unordered_map<uint32_t, int>& m, const std::vector<std::vector<uint8_t>>& v) {
            std::unordered_set<uint32_t> exact;
            for (size_t i = 0; i < v.size(); ++i) {
                const uint32_t k = pack(v[i].data(), static_cast<int>(v[i].size()));
                m[k] = static_cast<int>(i);
                exact.insert(k);
            }
            for (size_t i = 0; i < v.size(); ++i) add_variants(m, exact, v[i], static_cast<int>(i));
        };
        fill(m1_, bc1_);
        fill(m2_, bc2_);
    }
    int lookup(const uint8_t* r, int len) const {   // BC1 is the `len` bases before W1, from read position 0
        auto it = m1_.find(pack(r, len));
        return it == m1_.end() ? -1 : (it->second < 0 ? -1 : it->second);
    }
    int lookup2(const uint8_t* r) const {
        auto it = m2_.find(pack(r, kBc2Len));
        return it == m2_.end() ? -1 : (it->second < 0 ? -1 : it->second);
    }

    bool loaded_ = false;
    uint8_t pad_ = 4;
    std::vector<std::vector<uint8_t>> bc1_, bc2_;
    std::unordered_map<uint32_t, int> m1_, m2_;
    mutable std::atomic<uint64_t> total_{0}, ok_{0}, f_short_{0}, f_now1_{0}, f_bc1_{0}, f_bc2_{0};
};

}  // namespace singlet::fq
