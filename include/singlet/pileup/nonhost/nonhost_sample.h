// SPDX-License-Identifier: MIT
// nonhost_sample.h — which unmapped reads the non-host screen classifies, and the per-record outputs that line up with it.
//
// The screen classifies the records of star_Unmapped.out.mate1 that have a sequence and whose read-name hash is below a threshold
// (the threshold halves while more than `cap` records are kept, so the sample depends on the set of reads, not on the order STAR
// wrote them in; order-free-rules R8). The classification vector (multi_hits) has one entry per selected record, in file order.
// Everything that pairs a record with its classification (nonhost_reads.fq, the per-cell matrix) must take the same records:
// SampledReads::selected(). Indexing the classification by the raw record number pairs a read with another read's verdict as soon
// as the sample is smaller than the file (found on nuclei2k_v2: 116 M unmapped reads, 20 M cap).
#pragma once

#include <algorithm>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace singlet {
namespace nonhost {

// FNV-1a of the read name (up to the first space or tab, without the leading '@'), finalised with a 64-bit mix
inline uint64_t read_name_hash(const std::string& name_line) {
    const size_t e = name_line.find_first_of(" \t");
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 1; i < (e == std::string::npos ? name_line.size() : e); ++i) {
        h ^= static_cast<unsigned char>(name_line[i]);
        h *= 1099511628211ULL;
    }
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    return h;
}

struct SampledReads {
    std::vector<std::string> reads;   // sequences of the selected records, in file order
    uint64_t threshold = UINT64_MAX;  // final name-hash threshold
    bool subsampled = false;
    bool selected(const std::string& name_line, const std::string& seq_line) const {
        return !seq_line.empty() && read_name_hash(name_line) < threshold;
    }
};

// One pass over a 4-line FASTQ stream. The kept set is {records with a sequence and hash < final threshold}.
inline SampledReads sample_reads(std::istream& fq, size_t cap) {
    SampledReads out;
    std::vector<std::pair<uint64_t, std::string>> kept;
    std::string name_l, seq_l, plus_l, qual_l;
    while (std::getline(fq, name_l) && std::getline(fq, seq_l) && std::getline(fq, plus_l) && std::getline(fq, qual_l)) {
        if (seq_l.empty()) continue;
        const uint64_t h = read_name_hash(name_l);
        if (h >= out.threshold) continue;
        kept.emplace_back(h, std::move(seq_l));
        while (kept.size() > cap) {
            out.threshold >>= 1;
            out.subsampled = true;
            const uint64_t thr = out.threshold;
            kept.erase(std::remove_if(kept.begin(), kept.end(), [thr](const auto& p) { return p.first >= thr; }), kept.end());
        }
    }
    out.reads.reserve(kept.size());
    for (auto& p : kept) out.reads.push_back(std::move(p.second));
    return out;
}

// Writes the selected records whose classification passes `keep(i)` (i = index among the selected records = index into the
// classification vector, which may be shorter than the selection). Returns the number written.
template <class Keep>
inline size_t write_classified(std::istream& fq, std::ostream& out, const SampledReads& s, Keep keep) {
    std::string name_l, seq_l, plus_l, qual_l;
    size_t idx = 0, written = 0;
    while (std::getline(fq, name_l) && std::getline(fq, seq_l) && std::getline(fq, plus_l) && std::getline(fq, qual_l)) {
        if (!s.selected(name_l, seq_l)) continue;
        if (keep(idx)) {
            out << name_l << '\n' << seq_l << '\n' << plus_l << '\n' << qual_l << '\n';
            ++written;
        }
        ++idx;
    }
    return written;
}

}  // namespace nonhost
}  // namespace singlet
