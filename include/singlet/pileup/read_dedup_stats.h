// SPDX-License-Identifier: MIT
// singlet-pileup: read_dedup_stats.h
// PCR + optical duplicate detection and library complexity estimation for
// coordinate-sorted BAM files (Bulk RNA / Smart-seq2 modes).
//
// Algorithm (Picard-style):
//   - Group mapped reads by (chrom, pos, strand, mate_pos, CIGAR prefix)
//   - Within each group: keep highest-MAPQ read, mark rest BAM_FDUP (0x400)
//   - Optical duplicates: same lane+tile, Euclidean distance < optical_distance
//     pixels (parsed from Illumina QNAME: instrument:run:flowcell:lane:tile:x:y)
//   - Library complexity: Lander-Waterman Newton-Raphson (same as alignment_qc.h)
//
// BAM input MUST be coordinate-sorted. Output BAM (optional): duplicate reads
// have flag 0x400 set.

#pragma once

#include <htslib/hts.h>
#include <htslib/sam.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet {

class ReadDedupStats {
   public:
    // ── Configuration ──────────────────────────────────────────────────────

    struct Config {
        int optical_distance = 2500;  ///< pixel threshold for optical dups
        bool mark_duplicates = true;  ///< set SAM 0x400 flag in output BAM
        std::string out_bam_path;     ///< empty = no output BAM written
    };

    // ── Result ────────────────────────────────────────────────────────────

    struct Result {
        uint64_t total_reads = 0;
        uint64_t mapped_reads = 0;
        uint64_t duplicate_reads = 0;         ///< total marked duplicates
        uint64_t optical_duplicates = 0;      ///< subset: optical (tile-proximity)
        uint64_t pcr_duplicates = 0;          ///< duplicate_reads - optical_duplicates
        double duplication_rate = 0.0;        ///< duplicate_reads / mapped_reads
        double estimated_library_size = 0.0;  ///< Lander-Waterman estimate
        double unique_rate = 0.0;             ///< 1 - duplication_rate
    };

    void set_config(const Config& cfg) { config_ = cfg; }

    /// Process a coordinate-sorted BAM; returns dedup statistics.
    Result process_bam(const std::string& bam_path);

    // ── Public helpers (exposed for unit testing) ─────────────────────────

    /// Parsed Illumina tile coordinates from a read name.
    struct IlluminaCoords {
        int lane = -1;
        int tile = -1;
        int x = -1;
        int y = -1;
        bool valid = false;
    };

    /// Parse Illumina read name: instrument:run:flowcell:lane:tile:x:y
    /// Returns invalid coords if the name does not match the 7-field format.
    static IlluminaCoords parse_illumina_qname(const char* name);

    /// Euclidean pixel distance on the same flowcell tile.
    static double pixel_distance(int x1, int y1, int x2, int y2);

    /// Lander-Waterman library-complexity estimator (Newton-Raphson).
    /// @param total_mapped   N — total mapped reads
    /// @param distinct_reads C — non-duplicate mapped reads
    /// @return Estimated library size T (distinct molecules).
    static double estimate_library_size(uint64_t total_mapped,
                                        uint64_t distinct_reads);

    // ── Dedup key ──────────────────────────────────────────────────────────

    /// Unique identity key for a duplicate group.
    struct DedupKey {
        int32_t chrom = -1;
        int32_t pos = -1;
        int32_t mate_pos = -1;
        uint8_t strand = 0;      ///< 0=forward, 1=reverse
        uint32_t cigar_sig = 0;  ///< FNV-1a hash of ≤8 leading CIGAR ops

        bool operator==(const DedupKey& o) const {
            return chrom == o.chrom && pos == o.pos && mate_pos == o.mate_pos && strand == o.strand && cigar_sig == o.cigar_sig;
        }
    };

    struct DedupKeyHash {
        size_t operator()(const DedupKey& k) const noexcept {
            // Boost-style hash combine
            auto combine = [](size_t& h, size_t v) {
                h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
            };
            size_t h = std::hash<int32_t>{}(k.chrom);
            combine(h, std::hash<int32_t>{}(k.pos));
            combine(h, std::hash<int32_t>{}(k.mate_pos));
            combine(h, std::hash<uint8_t>{}(k.strand));
            combine(h, std::hash<uint32_t>{}(k.cigar_sig));
            return h;
        }
    };

    /// Build a DedupKey from a mapped bam1_t record.
    static DedupKey make_dedup_key(const bam1_t* b);

   private:
    Config config_;

    // True when two reads (same dedup group) are optical duplicates.
    bool is_optical_duplicate(const IlluminaCoords& a,
                              const IlluminaCoords& b) const;
};

// ── Inline implementations ────────────────────────────────────────────────────

inline ReadDedupStats::IlluminaCoords
ReadDedupStats::parse_illumina_qname(const char* name) {
    IlluminaCoords c;
    if (!name || !*name) return c;

    // Walk colon-separated fields; capture fields 3-6 (0-based) = lane,tile,x,y
    int field = 0;
    int vals[4] = {-1, -1, -1, -1};
    const char* p = name;

    while (*p) {
        if (field >= 3 && field <= 6) {
            char* end;
            long v = std::strtol(p, &end, 10);
            if (end > p) vals[field - 3] = static_cast<int>(v);
        }
        // Advance to next colon or end
        while (*p && *p != ':') ++p;
        if (*p == ':') {
            ++p;
            ++field;
        }
    }

    if (field >= 6 && vals[0] >= 0 && vals[1] >= 0 && vals[2] >= 0 && vals[3] >= 0) {
        c.lane = vals[0];
        c.tile = vals[1];
        c.x = vals[2];
        c.y = vals[3];
        c.valid = true;
    }
    return c;
}

inline double ReadDedupStats::pixel_distance(int x1, int y1, int x2, int y2) {
    double dx = static_cast<double>(x1 - x2);
    double dy = static_cast<double>(y1 - y2);
    return std::sqrt(dx * dx + dy * dy);
}

inline double ReadDedupStats::estimate_library_size(uint64_t total_mapped,
                                                    uint64_t distinct_reads) {
    if (total_mapped == 0 || distinct_reads == 0) return 0.0;
    if (distinct_reads >= total_mapped)
        return static_cast<double>(distinct_reads);

    const double N = static_cast<double>(total_mapped);
    const double C = static_cast<double>(distinct_reads);

    // Initial estimate from low-coverage approximation T ≈ C / (1 - C/N)
    double T = C / (1.0 - C / N);
    if (!std::isfinite(T) || T <= 0.0) T = C * 2.0;

    for (int iter = 0; iter < 100; ++iter) {
        double eNT = std::exp(-N / T);
        double f = T * (1.0 - eNT) - C;           // f(T)
        double fp = (1.0 - eNT) - (N / T) * eNT;  // f'(T)
        if (std::fabs(fp) < 1e-12) break;
        double dT = f / fp;
        T -= dT;
        if (T < C) T = C;  // T must be ≥ distinct reads
        if (std::fabs(dT) < 0.5) break;
    }
    return T;
}

inline ReadDedupStats::DedupKey ReadDedupStats::make_dedup_key(const bam1_t* b) {
    DedupKey k;
    if (!b) return k;
    k.chrom = b->core.tid;
    k.pos = b->core.pos;
    k.mate_pos = (b->core.flag & BAM_FPAIRED) ? b->core.mpos : -1;
    k.strand = (b->core.flag & BAM_FREVERSE) ? 1u : 0u;

    // FNV-1a hash of the leading min(8, n_cigar) CIGAR ops
    const uint32_t* cigar = bam_get_cigar(b);
    uint32_t n = std::min<uint32_t>(b->core.n_cigar, 8u);
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < n; ++i) {
        h ^= cigar[i];
        h *= 16777619u;
    }
    k.cigar_sig = h;
    return k;
}

inline bool ReadDedupStats::is_optical_duplicate(const IlluminaCoords& a,
                                                 const IlluminaCoords& b) const {
    if (!a.valid || !b.valid) return false;
    if (a.lane != b.lane || a.tile != b.tile) return false;
    return pixel_distance(a.x, a.y, b.x, b.y) < static_cast<double>(config_.optical_distance);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
inline ReadDedupStats::Result ReadDedupStats::process_bam(const std::string& bam_path) {
    Result res;

    htsFile* fp = hts_open(bam_path.c_str(), "r");
    if (!fp) {
        std::cerr << "[read_dedup_stats] Cannot open BAM: " << bam_path << "\n";
        return res;
    }

    sam_hdr_t* hdr = sam_hdr_read(fp);
    if (!hdr) {
        hts_close(fp);
        std::cerr << "[read_dedup_stats] Cannot read header: " << bam_path << "\n";
        return res;
    }

    // Optional output BAM
    htsFile* out_fp = nullptr;
    if (!config_.out_bam_path.empty()) {
        out_fp = hts_open(config_.out_bam_path.c_str(), "wb");
        if (out_fp) (void)sam_hdr_write(out_fp, hdr);
    }

    // ── Per-position read group ──────────────────────────────────────────

    struct Entry {
        bam1_t* b;
        int mapq;
        IlluminaCoords coords;
    };

    std::vector<Entry> group;
    int32_t grp_chrom = -1;
    int32_t grp_pos = -1;

    // Process one coordinate group: sub-group by DedupKey, mark duplicates.
    auto flush_group = [&]() {
        if (group.empty()) return;

        // Sub-group by DedupKey
        std::unordered_map<DedupKey, std::vector<size_t>, DedupKeyHash> sub;
        sub.reserve(group.size());
        for (size_t i = 0; i < group.size(); ++i)
            sub[make_dedup_key(group[i].b)].push_back(i);

        for (auto& [key, indices] : sub) {
            if (indices.size() < 2) continue;

            // Keep the highest-MAPQ read; mark rest as duplicates
            size_t best = indices[0];
            for (size_t idx : indices)
                if (group[idx].mapq > group[best].mapq) best = idx;

            for (size_t idx : indices) {
                if (idx == best) continue;
                if (config_.mark_duplicates)
                    group[idx].b->core.flag |= BAM_FDUP;
                ++res.duplicate_reads;

                // Optical vs PCR
                if (is_optical_duplicate(group[idx].coords, group[best].coords))
                    ++res.optical_duplicates;
            }
        }

        // Write to output BAM (all reads in group, flags updated)
        if (out_fp) {
            for (auto& e : group)
                (void)sam_write1(out_fp, hdr, e.b);
        }

        for (auto& e : group) bam_destroy1(e.b);
        group.clear();
        grp_chrom = grp_pos = -1;
    };

    bam1_t* b = bam_init1();

    while (sam_read1(fp, hdr, b) >= 0) {
        ++res.total_reads;

        if (b->core.flag & BAM_FUNMAP) {
            if (out_fp) (void)sam_write1(out_fp, hdr, b);
            continue;
        }
        ++res.mapped_reads;

        // Flush when we move to a new reference position
        if (b->core.tid != grp_chrom || b->core.pos != grp_pos) {
            flush_group();
            grp_chrom = b->core.tid;
            grp_pos = b->core.pos;
        }

        group.push_back({bam_dup1(b),
                         static_cast<int>(b->core.qual),
                         parse_illumina_qname(bam_get_qname(b))});
    }
    flush_group();  // last group

    bam_destroy1(b);
    sam_hdr_destroy(hdr);
    hts_close(fp);
    if (out_fp) hts_close(out_fp);

    // ── Finalize ──────────────────────────────────────────────────────────
    res.pcr_duplicates = (res.duplicate_reads >= res.optical_duplicates)
                             ? (res.duplicate_reads - res.optical_duplicates)
                             : 0;

    if (res.mapped_reads > 0)
        res.duplication_rate = static_cast<double>(res.duplicate_reads) / static_cast<double>(res.mapped_reads);
    res.unique_rate = 1.0 - res.duplication_rate;

    const uint64_t distinct = (res.mapped_reads > res.duplicate_reads)
                                  ? (res.mapped_reads - res.duplicate_reads)
                                  : res.mapped_reads;
    res.estimated_library_size = estimate_library_size(res.mapped_reads, distinct);

    return res;
}
#pragma GCC diagnostic pop

}  // namespace singlet
