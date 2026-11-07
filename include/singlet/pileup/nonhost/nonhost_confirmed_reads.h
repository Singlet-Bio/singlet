// SPDX-License-Identifier: MIT
// nonhost_confirmed_reads.h — the cDNA reads of EM-confirmed non-host species, packed for the nonhost_confirmed_reads member.
//
// The one place a sequenced read leaves the engine: Zach's exception to "no reads kept" is confirmed non-host species reads only.
// A read is written only when ALL of these hold (pack_confirmed_reads()):
//   1. the host k-mer filter was loaded (without it nothing is written: a host read cannot be recognised);
//   2. the read is a STAR-unmapped mate1 record that the screen classified (nonhost_sample.h), i.e. no host alignment;
//   3. the read carries not one minimizer in the host filter (stricter than the screen, which only needs >= kMinNonhostMins
//      non-host minimizers; a chimeric read with any host k-mer is left out);
//   4. its best-scoring EM-confirmed species is known (same rule as the per-cell matrix: highest hit rate among confirmed species);
//   5. its barcode is a called cell and barcode, UMI and sequence are plain ACGT (a read with an N is counted, not written).
// Records are sorted by (species id, barcode, UMI, sequence) so the file does not depend on the order STAR wrote the reads in
// (order-free-rules R8).
//
// File (little endian):
//   "SNCR" u8 version=1 u8 cb_len u8 umi_len u8 pad
//   u32 n_species u32 n_reads u32 n_dropped_n u32 n_excluded_host
//   n_species x u32 species_id (ascending)
//   n_reads x { u16 species_index, u8 len, u64 cb (2 bits/base, first base in the high bits), u32 umi, ceil(len/4) bytes seq (first
//               base in the high bits of byte 0) }
// cb_len <= 32 and umi_len <= 16.
#pragma once

#include "host_kmer_filter.h"
#include "min_sketch.h"
#include "nonhost_cell_matrix.h"
#include "nonhost_screener.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace singlet {
namespace nonhost {

struct ConfirmedReadsStats {
    uint32_t n_reads = 0;          // written
    uint32_t n_dropped_n = 0;      // confirmed reads left out because barcode, UMI or sequence has a non-ACGT base
    uint32_t n_excluded_host = 0;  // confirmed reads left out because a minimizer is in the host filter
    uint32_t n_species = 0;
};

// best-scoring EM-confirmed species of one read (the per-cell matrix rule); UINT32_MAX when none
inline uint32_t confirmed_read_species(const NonHostScreener::MultiHit& mh,
                                       const std::unordered_set<uint32_t>& confirmed) {
    uint32_t best = UINT32_MAX;
    float best_rate = 0.0f;
    for (const auto& [sid, rate] : mh.viral_hits)
        if (confirmed.count(sid) && rate > best_rate) { best_rate = rate; best = sid; }
    for (const auto& [sid, rate] : mh.microbial_hits)
        if (confirmed.count(sid) && rate > best_rate) { best_rate = rate; best = sid; }
    return best;
}

namespace detail {
inline int base2(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}
inline bool pack_bases(const std::string& s, uint64_t& out) {
    out = 0;
    for (char c : s) {
        const int b = base2(c);
        if (b < 0) return false;
        out = (out << 2) | static_cast<uint64_t>(b);
    }
    return true;
}
template <class T> inline void put(std::string& o, T v) {
    for (size_t i = 0; i < sizeof(T); ++i) o.push_back(static_cast<char>((static_cast<uint64_t>(v) >> (8 * i)) & 0xff));
}
}  // namespace detail

// Builds the file content. `tagged`, `multi_hits` are parallel (one entry per classified read). Returns an empty string when
// nothing is written, so the caller does not create the member: no host filter, no confirmed species, or no read passes.
inline std::string pack_confirmed_reads(const std::vector<TaggedRead>& tagged,
                                        const std::vector<NonHostScreener::MultiHit>& multi_hits,
                                        const std::unordered_set<uint32_t>& confirmed,
                                        const std::unordered_set<std::string>& valid_barcodes,
                                        const HostKmerFilter* host_filter, int cb_len, int umi_len, int k, int w,
                                        ConfirmedReadsStats* stats_out = nullptr) {
    ConfirmedReadsStats st;
    if (stats_out) *stats_out = st;
    if (!host_filter || !host_filter->loaded() || confirmed.empty() || cb_len <= 0 || cb_len > 32 || umi_len <= 0 || umi_len > 16)
        return {};
    struct Rec { uint32_t sid; uint64_t cb; uint32_t umi; std::string seq; };
    std::vector<Rec> recs;
    const size_t n = std::min(tagged.size(), multi_hits.size());
    for (size_t i = 0; i < n; ++i) {
        const TaggedRead& tr = tagged[i];
        if (tr.cb.empty() || tr.seq.empty() || tr.seq.size() > 255) continue;
        if (!valid_barcodes.empty() && !valid_barcodes.count(tr.cb)) continue;
        const uint32_t sid = confirmed_read_species(multi_hits[i], confirmed);
        if (sid == UINT32_MAX) continue;
        // privacy gate: no minimizer of the read may be a host k-mer
        const auto mins = extract_minimizers(tr.seq, k, w);
        if (host_filter->count_host_minimizers(mins).first != 0) { ++st.n_excluded_host; continue; }
        uint64_t cb = 0, umi = 0;
        if (static_cast<int>(tr.cb.size()) != cb_len || static_cast<int>(tr.umi.size()) != umi_len ||
            !detail::pack_bases(tr.cb, cb) || !detail::pack_bases(tr.umi, umi) ||
            std::any_of(tr.seq.begin(), tr.seq.end(), [](char c) { return detail::base2(c) < 0; })) {
            ++st.n_dropped_n;
            continue;
        }
        std::string seq = tr.seq;
        for (char& c : seq) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        recs.push_back({sid, cb, static_cast<uint32_t>(umi), std::move(seq)});
    }
    std::sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b) {
        return std::tie(a.sid, a.cb, a.umi, a.seq) < std::tie(b.sid, b.cb, b.umi, b.seq);
    });
    st.n_reads = static_cast<uint32_t>(recs.size());
    std::vector<uint32_t> sids;
    for (const auto& r : recs) if (sids.empty() || sids.back() != r.sid) sids.push_back(r.sid);
    st.n_species = static_cast<uint32_t>(sids.size());
    if (stats_out) *stats_out = st;
    if (recs.empty()) return {};

    std::string o;
    o += "SNCR";
    o.push_back(1); o.push_back(static_cast<char>(cb_len)); o.push_back(static_cast<char>(umi_len)); o.push_back(0);
    detail::put<uint32_t>(o, st.n_species);
    detail::put<uint32_t>(o, st.n_reads);
    detail::put<uint32_t>(o, st.n_dropped_n);
    detail::put<uint32_t>(o, st.n_excluded_host);
    for (uint32_t s : sids) detail::put<uint32_t>(o, s);
    for (const auto& r : recs) {
        const auto idx = static_cast<uint16_t>(std::lower_bound(sids.begin(), sids.end(), r.sid) - sids.begin());
        detail::put<uint16_t>(o, idx);
        o.push_back(static_cast<char>(r.seq.size()));
        detail::put<uint64_t>(o, r.cb);
        detail::put<uint32_t>(o, r.umi);
        for (size_t p = 0; p < r.seq.size(); p += 4) {
            uint8_t byte = 0;
            for (size_t j = 0; j < 4; ++j) byte = static_cast<uint8_t>((byte << 2) | (p + j < r.seq.size() ? detail::base2(r.seq[p + j]) : 0));
            o.push_back(static_cast<char>(byte));
        }
    }
    return o;
}

struct UnpackedRead { uint32_t species_id; std::string cb, umi, seq; };

// Reads a file written by pack_confirmed_reads; returns false when it is malformed.
inline bool unpack_confirmed_reads(const std::string& buf, std::vector<UnpackedRead>& out) {
    out.clear();
    if (buf.size() < 24 || std::memcmp(buf.data(), "SNCR", 4) != 0 || buf[4] != 1) return false;
    const int cb_len = static_cast<uint8_t>(buf[5]), umi_len = static_cast<uint8_t>(buf[6]);
    auto rd = [&](size_t off, int nb) { uint64_t v = 0; for (int i = 0; i < nb; ++i) v |= static_cast<uint64_t>(static_cast<uint8_t>(buf[off + i])) << (8 * i); return v; };
    const uint32_t ns = static_cast<uint32_t>(rd(8, 4)), nr = static_cast<uint32_t>(rd(12, 4));
    size_t pos = 24;
    if (buf.size() < pos + 4ull * ns) return false;
    std::vector<uint32_t> sids(ns);
    for (uint32_t i = 0; i < ns; ++i, pos += 4) sids[i] = static_cast<uint32_t>(rd(pos, 4));
    static const char B[4] = {'A', 'C', 'G', 'T'};
    auto unbases = [&](uint64_t v, int len) { std::string s(len, 'A'); for (int i = len - 1; i >= 0; --i, v >>= 2) s[i] = B[v & 3]; return s; };
    for (uint32_t r = 0; r < nr; ++r) {
        if (buf.size() < pos + 15) return false;
        const uint16_t idx = static_cast<uint16_t>(rd(pos, 2));
        const int len = static_cast<uint8_t>(buf[pos + 2]);
        const uint64_t cb = rd(pos + 3, 8), umi = rd(pos + 11, 4);
        pos += 15;
        const size_t nbytes = (static_cast<size_t>(len) + 3) / 4;
        if (idx >= ns || buf.size() < pos + nbytes) return false;
        std::string seq;
        for (size_t p = 0; p < nbytes; ++p)
            for (int j = 0; j < 4 && p * 4 + j < static_cast<size_t>(len); ++j)
                seq.push_back(B[(static_cast<uint8_t>(buf[pos + p]) >> (6 - 2 * j)) & 3]);
        pos += nbytes;
        out.push_back({sids[idx], unbases(cb, cb_len), unbases(umi, umi_len), seq});
    }
    return pos == buf.size();
}

}  // namespace nonhost
}  // namespace singlet
