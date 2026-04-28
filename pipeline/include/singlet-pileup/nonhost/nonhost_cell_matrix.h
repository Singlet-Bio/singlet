// nonhost_cell_matrix.h — Per-cell non-host species count matrix (NONHOST-CB-SPECIES-MATRIX)
//
// After NonHostScreener::classify_multi_batch() assigns reads to species via EM,
// this module correlates those assignments with cell barcodes from the paired
// STAR mate2 FASTQ to produce a cell-by-species UMI count matrix.
//
// Overview:
//   STAR outputs unmapped reads in two paired files when --outReadsUnmapped Fastx:
//     star_Unmapped.out.mate1 — cDNA reads (R2); passed to EM screening
//     star_Unmapped.out.mate2 — barcode+UMI reads (R1); parsed here for CB/UMI
//
//   For each read i:
//     1. Extract best species from multi_hits[i] (max hit_rate across viral + microbial)
//     2. Parse cb = seq[0:cb_len] and umi = seq[cb_len:cb_len+umi_len] from mate2[i]
//     3. If cb is in valid_barcodes and a species is assigned:
//          increment n_reads  for (cb, species_id)
//          insert umi into UMI set for (cb, species_id) → n_umis = |set| (exact dedup)
//   n_genes is always 0 (no GFF annotation at this stage).
//
// Output: nonhost_per_cell.tsv
//   barcode | species_name | species_taxon_id | kingdom | n_reads | n_umis | n_genes
//
// Wiring (NONHOST-CB-SPECIES-MATRIX):
//   After run_nonhost_em_screening() completes and multi_hits is in scope, if
//   valid_barcodes is non-empty (scRNA-seq / CITE-seq path), call:
//
//     auto tagged = NonHostCellMatrix::read_mate_pairs(
//         out_prefix + "/star_Unmapped.out.mate1",
//         out_prefix + "/star_Unmapped.out.mate2",
//         cb_len, umi_len);
//     auto mat = NonHostCellMatrix::build(tagged, multi_hits,
//                                         valid_barcodes, species_names, kingdom_map);
//     mat.write(out_prefix);
//
//   See INTEGRATION_NOTES.md for the full diff against singlify.cpp.
//
// Thread safety: not thread-safe during build(); read-only after build() returns.

#pragma once

#include "nonhost_screener.h"   // NonHostScreener::MultiHit, NonHostCategory

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace singlet {
namespace nonhost {

// ── TaggedRead ────────────────────────────────────────────────────────────────

// One unmapped read, paired with its raw cell barcode and UMI extracted from
// the corresponding mate2 barcode+UMI read.  The cb and umi strings may be
// empty when mate2 is absent or the barcode+UMI read is too short.
struct TaggedRead {
    std::string seq;  // cDNA sequence (from mate1; used for EM alignment upstream)
    std::string cb;   // raw cell barcode extracted from mate2 read
    std::string umi;  // raw UMI extracted from mate2 read
};

// ── CellSpeciesRecord ─────────────────────────────────────────────────────────

// One row in the output nonhost_per_cell.tsv: a single (cell, species) pair.
struct CellSpeciesRecord {
    std::string     cb;               // called cell barcode
    std::string     species_name;     // display name from species map
    uint32_t        species_taxon_id; // numeric species ID from EM
    NonHostCategory kingdom;          // VIRAL / BACTERIAL / FUNGAL
    uint32_t        n_reads;          // total aligned reads from this cell to this species
    uint32_t        n_umis;           // unique UMIs (exact dedup on UMI string)
    uint32_t        n_genes;          // always 0 (no GFF annotation at nonhost stage)
};

// ── NonHostCellMatrix ─────────────────────────────────────────────────────────

class NonHostCellMatrix {
public:
    // ── build() ───────────────────────────────────────────────────────────────
    //
    // Assemble the cell×species count matrix from per-read data.
    //
    // tagged_reads   : parallel to multi_hits; index i = cDNA read i
    // multi_hits     : per-read EM soft classifications from NonHostScreener
    // valid_barcodes : called cell set from EmptyDrops (raw barcodes, 0-offset)
    // species_names  : species_id → display name (may be empty → write numeric ID)
    // kingdom_map    : species_id → NonHostCategory (fallback: from top_category)
    //
    // Returns a NonHostCellMatrix whose records() are sorted by (cb, species_name).
    // Reads whose cb is not in valid_barcodes, or whose multi_hits entry has no
    // viral or microbial hits, are silently excluded.
    static NonHostCellMatrix build(
            const std::vector<TaggedRead>&                          tagged_reads,
            const std::vector<NonHostScreener::MultiHit>&           multi_hits,
            const std::unordered_set<std::string>&                  valid_barcodes,
            const std::unordered_map<uint32_t, std::string>&        species_names,
            const std::unordered_map<uint32_t, NonHostCategory>&    kingdom_map) {

        NonHostCellMatrix mat;

        const size_t n = std::min(tagged_reads.size(), multi_hits.size());
        if (n == 0) return mat;

        // Accumulator: (cb, species_id) → {n_reads, UMI set}
        // Using std::map for deterministic iteration order (sorted by cb then species_id).
        struct Accum {
            uint32_t                     n_reads = 0;
            std::unordered_set<std::string> umis_seen;
        };
        std::map<std::pair<std::string, uint32_t>, Accum> accum;

        for (size_t i = 0; i < n; ++i) {
            const TaggedRead&                  tr = tagged_reads[i];
            const NonHostScreener::MultiHit&   mh = multi_hits[i];

            // Skip reads without a barcode
            if (tr.cb.empty()) continue;

            // Skip reads not in the called cell set
            if (!valid_barcodes.empty() && valid_barcodes.find(tr.cb) == valid_barcodes.end())
                continue;

            // Find best-scoring species assignment across viral + microbial
            uint32_t        best_sid   = UINT32_MAX;
            float           best_rate  = 0.0f;
            NonHostCategory best_king  = NonHostCategory::UNKNOWN;

            for (const auto& [sid, rate] : mh.viral_hits) {
                if (rate > best_rate) {
                    best_rate  = rate;
                    best_sid   = sid;
                    best_king  = NonHostCategory::VIRAL;
                }
            }
            for (const auto& [sid, rate] : mh.microbial_hits) {
                if (rate > best_rate) {
                    best_rate  = rate;
                    best_sid   = sid;
                    best_king  = NonHostCategory::BACTERIAL;
                }
            }

            // Skip reads with no species assignment
            if (best_sid == UINT32_MAX) continue;

            // Accumulate
            auto key = std::make_pair(tr.cb, best_sid);
            auto& acc = accum[key];
            ++acc.n_reads;
            if (!tr.umi.empty())    // skip empty UMI — not counted toward n_umis
                acc.umis_seen.insert(tr.umi);

            // kingom_map overrides mh-derived kingdom if available
            auto kit = kingdom_map.find(best_sid);
            if (kit != kingdom_map.end())
                best_king = kit->second;

            (void)best_king; // stored below from accum iteration
        }

        // Convert accumulator to sorted vector of CellSpeciesRecord
        mat.records_.reserve(accum.size());
        for (const auto& [key, acc] : accum) {
            const auto& [cb, sid] = key;

            CellSpeciesRecord rec;
            rec.cb               = cb;
            rec.species_taxon_id = sid;
            rec.n_reads          = acc.n_reads;
            rec.n_umis           = static_cast<uint32_t>(acc.umis_seen.size());
            rec.n_genes          = 0;

            // Species name
            auto nit = species_names.find(sid);
            rec.species_name = (nit != species_names.end()) ? nit->second
                                                             : std::to_string(sid);

            // Kingdom (prefer explicit map; fall back to per-read inference)
            auto kit = kingdom_map.find(sid);
            if (kit != kingdom_map.end()) {
                rec.kingdom = kit->second;
            } else {
                // Infer from first matching hit in any read
                rec.kingdom = NonHostCategory::UNKNOWN;
                for (size_t i = 0; i < n; ++i) {
                    const auto& mh = multi_hits[i];
                    for (const auto& [msid, rate] : mh.viral_hits) {
                        if (msid == sid) { rec.kingdom = NonHostCategory::VIRAL; goto kingdom_done; }
                    }
                    for (const auto& [msid, rate] : mh.microbial_hits) {
                        if (msid == sid) { rec.kingdom = NonHostCategory::BACTERIAL; goto kingdom_done; }
                    }
                }
                kingdom_done:;
            }

            mat.records_.push_back(std::move(rec));
        }

        // Already sorted by (cb, species_id) from std::map iteration.
        // Re-sort by (cb, species_name) for stable, name-based output order.
        std::sort(mat.records_.begin(), mat.records_.end(),
                  [](const CellSpeciesRecord& a, const CellSpeciesRecord& b) {
                      if (a.cb != b.cb) return a.cb < b.cb;
                      return a.species_name < b.species_name;
                  });

        return mat;
    }

    // ── read_mate_pairs() ──────────────────────────────────────────────────────
    //
    // Parse paired unmapped FASTQ files produced by STAR (--outReadsUnmapped Fastx)
    // and extract a TaggedRead per cDNA read in the same order as the EM inputs.
    //
    // mate1_path : star_Unmapped.out.mate1 — cDNA reads (R2)
    // mate2_path : star_Unmapped.out.mate2 — barcode+UMI reads (R1)
    // cb_len     : barcode length in mate2 (e.g. 16 for 10x v3)
    // umi_len    : UMI length in mate2 following the barcode (e.g. 12 for 10x v3)
    //
    // Returns one TaggedRead per cDNA read.  If mate2 is absent or shorter than
    // mate1, the remaining reads get empty cb and umi strings (will be ignored
    // by build() since they are not in valid_barcodes).
    static std::vector<TaggedRead> read_mate_pairs(
            const std::string& mate1_path,
            const std::string& mate2_path,
            int cb_len,
            int umi_len) {

        std::vector<TaggedRead> result;
        std::ifstream fq1(mate1_path);
        std::ifstream fq2(mate2_path);

        std::string n1, s1, p1, q1;  // mate1 lines
        std::string n2, s2, p2, q2;  // mate2 lines

        while (std::getline(fq1, n1) && std::getline(fq1, s1) &&
               std::getline(fq1, p1) && std::getline(fq1, q1)) {
            TaggedRead tr;
            tr.seq = s1;

            // Read corresponding mate2 record (if available)
            if (fq2.is_open() &&
                std::getline(fq2, n2) && std::getline(fq2, s2) &&
                std::getline(fq2, p2) && std::getline(fq2, q2)) {
                // Extract barcode: first cb_len bases
                if (cb_len > 0 && static_cast<int>(s2.size()) >= cb_len)
                    tr.cb = s2.substr(0, cb_len);
                // Extract UMI: next umi_len bases after barcode
                if (umi_len > 0 &&
                    static_cast<int>(s2.size()) >= cb_len + umi_len)
                    tr.umi = s2.substr(cb_len, umi_len);
                // If R1 is shorter than cb_len + umi_len (edge case), fields stay empty
            }

            result.push_back(std::move(tr));
        }

        return result;
    }

    // ── write() ───────────────────────────────────────────────────────────────
    //
    // Write nonhost_per_cell.tsv to out_prefix.
    // Silently returns if records() is empty (bulk/ATAC assays with no cell barcodes).
    void write(const std::string& out_prefix) const {
        if (records_.empty()) return;

        const std::string path = out_prefix + "/nonhost_per_cell.tsv";
        std::ofstream tsv(path);
        if (!tsv.is_open()) {
            std::cerr << "[nonhost-cell-matrix] WARNING: cannot write "
                      << path << "\n";
            return;
        }

        tsv << "barcode\tspecies_name\tspecies_taxon_id\tkingdom"
               "\tn_reads\tn_umis\tn_genes\n";

        for (const auto& r : records_) {
            const char* kname = "UNKNOWN";
            switch (r.kingdom) {
                case NonHostCategory::VIRAL:     kname = "VIRAL";     break;
                case NonHostCategory::BACTERIAL: kname = "BACTERIAL"; break;
                case NonHostCategory::FUNGAL:    kname = "FUNGAL";    break;
                default: break;
            }
            tsv << r.cb             << "\t"
                << r.species_name   << "\t"
                << r.species_taxon_id << "\t"
                << kname            << "\t"
                << r.n_reads        << "\t"
                << r.n_umis         << "\t"
                << r.n_genes        << "\n";
        }

        std::cerr << "[nonhost-cell-matrix] " << records_.size()
                  << " cell×species records → " << path << "\n";
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    const std::vector<CellSpeciesRecord>& records() const { return records_; }
    size_t n_records() const { return records_.size(); }

private:
    std::vector<CellSpeciesRecord> records_;
};

} // namespace nonhost
} // namespace singlet
