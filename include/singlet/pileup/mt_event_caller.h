// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: mt_event_caller.h
// Donor-aware mitochondrial event caller (Gate G6, Track A).
//
// Event codes (DROPLET_OUTPUT_SCHEMA.md §5.3):
//   0 = ref              (never emitted — sparse default)
//   1 = sub_A            substitution, alt allele is A
//   2 = sub_C            substitution, alt allele is C
//   3 = sub_G            substitution, alt allele is G
//   4 = sub_T            substitution, alt allele is T
//   5 = insertion        CIGAR BAM_CINS op on chrM-mapped read
//   6 = deletion         CIGAR BAM_CDEL op on chrM-mapped read
//   7 = frameshift       indel with len % 3 != 0, located inside a mt CDS region
//   8 = premature_stop   substitution that introduces a stop codon (vertebrate mt code)
//   9 = complex          multiple distinct events at the same locus in the same cell
//
// Thresholds (§5.3.1):
//   min_coverage_per_locus  = 3   total depth at position in cell
//   min_alt_count_per_locus = 1   absolute alt read count
//   heteroplasmy_threshold  = 0.02 alt VAF ≥ 2% to emit event

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "donor_demux.h"
#include "mt_heteroplasmy.h"
#include "sparse_accumulator.h"
#include "pz_writer.h"
#include "kmer_util.h"

namespace singlet {
namespace mt {

// ============================================================================
// Schema constants (§5.3.1)
// ============================================================================
static constexpr int   kMinCoveragePerLocus  = 3;
static constexpr int   kMinAltCountPerLocus  = 1;
static constexpr float kHeteroplasmyThresh   = 0.02f;

// ============================================================================
// Event codes
// ============================================================================
static constexpr uint8_t kEvRef       = 0;   // never stored
static constexpr uint8_t kEvSubA      = 1;
static constexpr uint8_t kEvSubC      = 2;
static constexpr uint8_t kEvSubG      = 3;
static constexpr uint8_t kEvSubT      = 4;
static constexpr uint8_t kEvInsertion = 5;   // CIGAR BAM_CINS on chrM read
static constexpr uint8_t kEvDeletion  = 6;   // CIGAR BAM_CDEL on chrM read
static constexpr uint8_t kEvFrameshift = 7;  // indel with len % 3 != 0 inside mt CDS
static constexpr uint8_t kEvPremStop  = 8;   // substitution creating vertebrate mt stop codon
static constexpr uint8_t kEvComplex   = 9;

// base_idx (A=0,C=1,G=2,T=3) → substitution event code
static constexpr uint8_t kBaseToEvent[4] = {
    kEvSubA, kEvSubC, kEvSubG, kEvSubT
};

// ============================================================================
// Per-cell mt event COO entry (intermediate accumulation)
// ============================================================================
struct MtEventEntry {
    uint32_t cell_idx;   // barcode column index in mt_csc
    uint32_t mt_pos;     // 0-based chrM position
    uint8_t  event_code; // event code 1-9
};

// ============================================================================
// Per-cell mt summary row
// ============================================================================
struct MtCellSummary {
    uint32_t cell_idx;
    int      donor_id;           // -1 = unassigned / doublet
    std::string donor_label;     // "donor0", ..., "doublet", "unassigned"
    uint32_t mt_reads;           // total base counts across all mt loci
    float    mt_coverage_mean;   // mean depth per covered locus (depth ≥ 1)
    uint32_t n_loci_with_event;
    uint32_t n_substitutions;
    uint32_t n_indels;           // always 0 (no indel tracking)
    uint32_t n_frameshifts;      // always 0
    uint32_t n_premature_stops;  // always 0
    float    pct_heteroplasmic;  // 100 * n_loci_with_event / n_loci_covered
};

// ============================================================================
// Full G6 result structure
// ============================================================================
struct MtEventResult {
    // Sparse matrix for mt_events.1pz (rows=loci, cols=cells)
    uint32_t n_loci  = 0;
    uint32_t n_cells = 0;
    std::vector<int32_t>  indptr;         // [n_cells + 1]
    std::vector<int32_t>  indices;        // row (locus) indices
    std::vector<uint8_t>  data;           // event codes
    std::vector<std::string> locus_names; // "chrM:N" (1-based)

    // Per-donor consensus (each MT_LEN chars)
    std::vector<std::string> donor_consensus;

    // Per-cell summaries for mt_summary.tsv
    std::vector<MtCellSummary> summaries;

    // Embedded .1pz metadata
    std::map<std::string, std::string> pz_meta;
};

// ============================================================================
// Build per-donor mt consensus sequences.
//
// For each position, the consensus base is the most common allele across
// all cells assigned to that donor.  Positions with total depth below
// kMinCoveragePerLocus remain 'N'.
// ============================================================================
inline std::vector<std::string> build_donor_consensus(
    const int32_t* indptr,
    const int32_t* indices,
    const uint16_t* data,
    uint32_t n_cells,
    const std::vector<DonorAssignment>& assignments,
    int n_donors)
{
    std::vector<std::string> consensus(
        static_cast<size_t>(std::max(n_donors, 0)),
        std::string(MT_LEN, 'N'));

    if (n_donors <= 0 || n_cells == 0) return consensus;

    // Per-donor base-count accumulator [n_donors × MT_LEN][4]
    // Heap-allocated to avoid stack overflow for n_donors > 1.
    const size_t slots = static_cast<size_t>(n_donors) * MT_LEN;
    std::vector<std::array<uint64_t, 4>> dp(slots,
        std::array<uint64_t,4>{0,0,0,0});

    for (uint32_t cell = 0; cell < n_cells; ++cell) {
        if (cell >= assignments.size()) continue;
        const int d = assignments[cell].donor_id;
        if (d < 0 || d >= n_donors) continue;  // unassigned / doublet

        for (int32_t k = indptr[cell]; k < indptr[cell + 1]; ++k) {
            uint32_t feat = static_cast<uint32_t>(indices[k]);
            if (feat >= MT_N_FEATURES) continue;
            uint32_t pos  = feat / 4;
            uint32_t base = feat % 4;
            dp[static_cast<size_t>(d) * MT_LEN + pos][base] +=
                static_cast<uint64_t>(data[k]);
        }
    }

    for (int d = 0; d < n_donors; ++d) {
        for (uint32_t pos = 0; pos < MT_LEN; ++pos) {
            auto& bc = dp[static_cast<size_t>(d) * MT_LEN + pos];
            uint64_t total = bc[0] + bc[1] + bc[2] + bc[3];
            if (total < static_cast<uint64_t>(kMinCoveragePerLocus)) continue;
            int best = 0;
            for (int b = 1; b < 4; ++b)
                if (bc[b] > bc[best]) best = b;
            consensus[d][pos] = IDX_TO_BASE[best];
        }
    }
    return consensus;
}

// ============================================================================
// Simple FNV-1a 32-bit hash (surrogate for consensus FASTA md5).
// Labelled "md5" in metadata per spec key name; readers must accept
// this shortened digest.
// ============================================================================
inline std::string fnv32_hex(const std::string& s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s) { h ^= c; h *= 16777619u; }
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%08x", h);
    return std::string(buf);
}

// ============================================================================
// Write donor{N}_mt_consensus.fa
// ============================================================================
inline void write_donor_consensus_fa(
    const std::string& mt_dir,
    int donor_idx,
    const std::string& consensus,
    const std::string& ref_build)
{
    const std::string path =
        mt_dir + "/donor" + std::to_string(donor_idx) + "_mt_consensus.fa";
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[mt_event_caller] WARN: cannot write " << path << "\n";
        return;
    }
    f << ">donor" << donor_idx << "_mt"
      << "  source=singlet-vb-demux"
      << " ref=chrM"
      << " ref_build=" << ref_build << "\n";
    for (size_t i = 0; i < consensus.size(); i += 60)
        f << consensus.substr(i, 60) << "\n";
    std::cerr << "[mt_event_caller] Wrote " << path << "\n";
}

// ============================================================================
// Write donor{N}_mt_variants.vcf
// Variants = positions where donor consensus differs from species reference
// (species_ref = majority allele pooled across all cells, approximating rCRS).
// ============================================================================
inline void write_donor_variants_vcf(
    const std::string& mt_dir,
    int donor_idx,
    const std::string& donor_consensus,
    const std::string& species_ref,
    const std::string& ref_build,
    const std::string& singlet_version)
{
    const std::string path =
        mt_dir + "/donor" + std::to_string(donor_idx) + "_mt_variants.vcf";
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[mt_event_caller] WARN: cannot write " << path << "\n";
        return;
    }
    f << "##fileformat=VCFv4.2\n"
      << "##source=singlet-vb-demux\n"
      << "##singlet_version=" << singlet_version << "\n"
      << "##donor=donor" << donor_idx << "\n"
      << "##ref_build=" << ref_build << "\n"
      << "##contig=<ID=chrM,length=" << MT_LEN << ">\n"
      << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n";

    uint32_t n_variants = 0;
    for (uint32_t pos = 0; pos < MT_LEN; ++pos) {
        char ref_base = (pos < species_ref.size()) ? species_ref[pos] : 'N';
        char alt_base = donor_consensus[pos];
        if (alt_base == 'N' || ref_base == 'N') continue;
        if (alt_base == ref_base) continue;
        f << "chrM\t" << (pos + 1) << "\t.\t"
          << ref_base << "\t" << alt_base << "\t.\t.\t.\n";
        ++n_variants;
    }
    std::cerr << "[mt_event_caller] Wrote " << path
              << " (" << n_variants << " variants)\n";
}

// ============================================================================
// mt CDS region for frameshift (code 7) and premature-stop (code 8) detection
// ============================================================================
struct MtCdsRegion {
    uint32_t start;       ///< 0-based inclusive (GTF 1-based - 1)
    uint32_t end;         ///< 0-based exclusive (GTF 1-based end)
    int      frame;       ///< reading frame phase 0/1/2 as in GTF
    bool     is_rev;      ///< true = L-strand (GTF strand "-")
    std::string gene;     ///< gene_name attribute value
};

/// Load chrM CDS regions from a GTF file.
/// Returns empty vector on failure (codes 7/8 are silently skipped).
inline std::vector<MtCdsRegion> load_mt_cds_regions(const std::string& gtf_path) {
    std::vector<MtCdsRegion> regions;
    if (gtf_path.empty()) return regions;
    std::ifstream f(gtf_path);
    if (!f.is_open()) return regions;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        // Split into tab fields: chrom(0), src(1), feature(2), start(3), end(4),
        //   score(5), strand(6), frame(7), attributes(8)
        auto tab = [&](int n) -> std::pair<size_t,size_t> {
            size_t s = 0;
            for (int t = 0; t < n; ++t) {
                s = line.find('\t', s);
                if (s == std::string::npos) return {std::string::npos, 0};
                ++s;
            }
            size_t e = line.find('\t', s);
            if (e == std::string::npos) e = line.size();
            return {s, e};
        };
        auto [c0,c1] = tab(0); if (c0 == std::string::npos) continue;
        std::string chrom = line.substr(c0, c1-c0);
        if (chrom != "chrM" && chrom != "MT") continue;

        auto [f0,f1] = tab(2); if (f0 == std::string::npos) continue;
        if (line.substr(f0, f1-f0) != "CDS") continue;

        auto [s0,s1] = tab(3); if (s0 == std::string::npos) continue;
        auto [e0,e1] = tab(4); if (e0 == std::string::npos) continue;
        uint32_t gstart = static_cast<uint32_t>(std::stoul(line.substr(s0, s1-s0))) - 1;
        uint32_t gend   = static_cast<uint32_t>(std::stoul(line.substr(e0, e1-e0)));

        auto [str0,str1] = tab(6); if (str0 == std::string::npos) continue;
        bool is_rev = (line[str0] == '-');

        auto [fr0,fr1] = tab(7); if (fr0 == std::string::npos) continue;
        int frame = (line[fr0] == '1') ? 1 : (line[fr0] == '2') ? 2 : 0;

        // Extract gene_name from attributes
        std::string gene;
        auto [a0,a1] = tab(8);
        if (a0 != std::string::npos) {
            const std::string attrs = line.substr(a0, a1-a0);
            auto gp = attrs.find("gene_name");
            if (gp == std::string::npos) gp = attrs.find("gene_id");
            if (gp != std::string::npos) {
                auto q1 = attrs.find('"', gp);
                auto q2 = (q1 != std::string::npos) ? attrs.find('"', q1+1) : std::string::npos;
                if (q1 != std::string::npos && q2 != std::string::npos)
                    gene = attrs.substr(q1+1, q2-q1-1);
            }
        }
        regions.push_back({gstart, gend, frame, is_rev, gene});
    }
    std::cerr << "[mt_event_caller] Loaded " << regions.size()
              << " chrM CDS regions from GTF\n";
    return regions;
}

/// Find the CDS region containing 0-based position pos (or nullptr).
static inline const MtCdsRegion* find_mt_cds(
    uint32_t pos, const std::vector<MtCdsRegion>& cds)
{
    for (const auto& r : cds)
        if (pos >= r.start && pos < r.end) return &r;
    return nullptr;
}

/// Return true if the indel length is frame-shifting (not divisible by 3).
static inline bool is_mt_frameshift(uint16_t len) { return (len % 3) != 0; }

/// Reverse-complement a string (ACGTN).
/// Thin wrapper over the shared singlet::pileup::kmer helper.
static inline std::string mt_revcomp(const std::string& s) {
    return ::singlet::pileup::kmer::revcomp_ascii(s);
}

/// Vertebrate mitochondrial genetic code stop codons (NCBI table 2).
/// TGA = Trp (NOT stop), AGA/AGG = Stop (not Arg).
static inline bool is_mt_stop_codon(char b0, char b1, char b2) {
    if (b0=='T' && b1=='A' && (b2=='A'||b2=='G')) return true;   // TAA, TAG
    if (b0=='A' && b1=='G' && (b2=='A'||b2=='G')) return true;   // AGA, AGG
    return false;
}

/// Return true if substituting alt_base at pos in donor_consensus creates a
/// stop codon in the given CDS (vertebrate mt code, handles +/- strand).
static inline bool mt_introduces_stop(
    uint32_t pos, char alt_base,
    const std::string& donor_consensus,
    const MtCdsRegion& cds)
{
    if (pos >= donor_consensus.size()) return false;
    // Compute 0-based position within CDS
    int cds_offset = static_cast<int>(pos) - static_cast<int>(cds.start);
    if (cds_offset < 0) return false;
    // Adjust for reading frame phase
    int fa = cds_offset + cds.frame;
    int codon_pos = fa % 3;             // position within the codon (0,1,2)
    int codon_rel = fa - codon_pos;     // codon start relative to (cds.start - cds.frame)
    int c0 = static_cast<int>(cds.start) - cds.frame + codon_rel;
    if (c0 < 0 || c0+2 >= static_cast<int>(donor_consensus.size())) return false;
    char b0 = donor_consensus[c0], b1 = donor_consensus[c0+1], b2 = donor_consensus[c0+2];
    if (codon_pos == 0) b0 = alt_base;
    else if (codon_pos == 1) b1 = alt_base;
    else b2 = alt_base;
    if (cds.is_rev) {
        // Reverse complement the codon for L-strand genes
        char tmp0=b2, tmp1=b1, tmp2=b0;
        auto rc=[](char c){return c=='A'?'T':c=='T'?'A':c=='C'?'G':c=='G'?'C':'N';};
        b0=rc(tmp0); b1=rc(tmp1); b2=rc(tmp2);
    }
    return is_mt_stop_codon(b0, b1, b2);
}

// ============================================================================
// Classify per-cell mt events against donor consensus.
//
// Returns MtEventResult with:
//   - COO → CSC conversion for mt_events.1pz
//   - Per-cell summaries for mt_summary.tsv
//   - Embedded .1pz metadata
// ============================================================================
inline MtEventResult classify_mt_events(
    const int32_t* indptr,
    const int32_t* indices,
    const uint16_t* data,
    uint32_t n_cells,
    const std::vector<DonorAssignment>& assignments,
    const std::vector<std::string>& donor_consensus,
    const std::vector<std::string>& barcodes,
    const std::string& ref_build,
    const std::vector<MtIndelEvent>& indels = {},
    const std::vector<MtCdsRegion>&  cds_regions = {})
{
    MtEventResult result;
    result.n_cells = n_cells;
    const int n_donors = static_cast<int>(donor_consensus.size());

    // Pre-sort indels by bc_idx for efficient per-cell lookup
    std::vector<MtIndelEvent> sorted_indels = indels;
    std::sort(sorted_indels.begin(), sorted_indels.end(),
              [](const MtIndelEvent& a, const MtIndelEvent& b) {
                  return a.bc_idx < b.bc_idx || (a.bc_idx == b.bc_idx && a.pos < b.pos);
              });

    // COO accumulation (cell_idx, pos, code) — sorted by cell_idx since we
    // iterate cells in order.
    std::vector<MtEventEntry> coo;
    coo.reserve(n_cells * 4);

    result.summaries.resize(n_cells);
    size_t ind_scan = 0;  // rolling start for per-cell indel lookup

    for (uint32_t cell = 0; cell < n_cells; ++cell) {
        auto& summ       = result.summaries[cell];
        summ.cell_idx    = cell;
        const int d      = (cell < assignments.size()) ? assignments[cell].donor_id : -1;
        summ.donor_id    = d;
        // Label: use the label from DonorAssignment when available
        summ.donor_label = (cell < assignments.size())
            ? assignments[cell].label
            : "unassigned";
        summ.mt_reads           = 0;
        summ.mt_coverage_mean   = 0.f;
        summ.n_loci_with_event  = 0;
        summ.n_substitutions    = 0;
        summ.n_indels           = 0;
        summ.n_frameshifts      = 0;
        summ.n_premature_stops  = 0;
        summ.pct_heteroplasmic  = 0.f;

        const std::string* cons_seq =
            (d >= 0 && d < n_donors) ? &donor_consensus[d] : nullptr;

        // Collect per-position base counts for this cell
        struct PosBase { uint32_t pos; uint32_t base; uint16_t cnt; };
        std::vector<PosBase> cell_pb;
        cell_pb.reserve(
            static_cast<size_t>(indptr[cell + 1] - indptr[cell]));
        for (int32_t k = indptr[cell]; k < indptr[cell + 1]; ++k) {
            uint32_t feat = static_cast<uint32_t>(indices[k]);
            if (feat >= MT_N_FEATURES || data[k] == 0) continue;
            cell_pb.push_back({feat / 4, feat % 4, data[k]});
            summ.mt_reads += data[k];
        }

        // per-position event map: pos → raw event code (will be merged with indels)
        std::unordered_map<uint32_t, uint8_t> pos_events;

        if (!cell_pb.empty()) {
            // Sort by position
            std::sort(cell_pb.begin(), cell_pb.end(),
                      [](const PosBase& a, const PosBase& b) {
                          return a.pos < b.pos;
                      });

            uint32_t n_covered = 0;
            size_t i = 0;
            while (i < cell_pb.size()) {
                const uint32_t cur_pos = cell_pb[i].pos;
                uint16_t counts[4] = {0, 0, 0, 0};
                while (i < cell_pb.size() && cell_pb[i].pos == cur_pos) {
                    counts[cell_pb[i].base] += cell_pb[i].cnt;
                    ++i;
                }
                const uint32_t total = counts[0]+counts[1]+counts[2]+counts[3];
                if (total == 0) continue;
                ++n_covered;

                // Skip event calling if no donor consensus or coverage too low
                if (!cons_seq || cur_pos >= cons_seq->size()) continue;
                const char cons_base = (*cons_seq)[cur_pos];
                if (cons_base == 'N') continue;
                if (static_cast<int>(total) < kMinCoveragePerLocus) continue;

                // Find consensus base index
                int cons_idx = -1;
                for (int b = 0; b < 4; ++b) {
                    if (IDX_TO_BASE[b] == cons_base) { cons_idx = b; break; }
                }
                if (cons_idx < 0) continue;

                // Count alt alleles above threshold
                int n_alts    = 0;
                int first_alt = -1;
                for (int b = 0; b < 4; ++b) {
                    if (b == cons_idx) continue;
                    if (counts[b] < static_cast<uint16_t>(kMinAltCountPerLocus)) continue;
                    const float vaf = static_cast<float>(counts[b]) /
                                      static_cast<float>(total);
                    if (vaf < kHeteroplasmyThresh) continue;
                    ++n_alts;
                    if (first_alt < 0) first_alt = b;
                }
                if (n_alts == 0) continue;

                const uint8_t code = (n_alts > 1)
                    ? kEvComplex
                    : kBaseToEvent[first_alt];
                pos_events[cur_pos] = code;  // collect; emit after indel merge
            }

            if (n_covered > 0) {
                summ.mt_coverage_mean = static_cast<float>(summ.mt_reads) /
                                        static_cast<float>(n_covered);
                // pct_heteroplasmic will be recomputed after final n_loci_with_event
            }
        }

        // ── Add indel events (codes 5/6/7) ──────────────────────────────────
        // Advance rolling scan past earlier cells
        while (ind_scan < sorted_indels.size() && sorted_indels[ind_scan].bc_idx < cell)
            ++ind_scan;
        for (size_t j = ind_scan;
             j < sorted_indels.size() && sorted_indels[j].bc_idx == cell; ++j) {
            const MtIndelEvent& ind = sorted_indels[j];
            if (ind.pos >= MT_LEN) continue;

            uint8_t ind_code;
            const MtCdsRegion* cds = cds_regions.empty()
                ? nullptr : find_mt_cds(ind.pos, cds_regions);
            if (cds && is_mt_frameshift(ind.len))
                ind_code = kEvFrameshift;
            else
                ind_code = ind.is_ins ? kEvInsertion : kEvDeletion;

            auto it = pos_events.find(ind.pos);
            if (it != pos_events.end()) {
                // Already have a sub event here → upgrade to complex
                it->second = kEvComplex;
            } else {
                pos_events[ind.pos] = ind_code;
            }
        }

        // ── Code-8: check subs in CDS for stop codon introduction ────────────
        if (!cds_regions.empty() && cons_seq) {
            for (auto& [pos, code] : pos_events) {
                if (code < kEvSubA || code > kEvSubT) continue;  // not a plain sub
                const MtCdsRegion* cds = find_mt_cds(pos, cds_regions);
                if (!cds) continue;
                char alt = IDX_TO_BASE[code - 1];  // code 1-4 → base idx 0-3
                if (mt_introduces_stop(pos, alt, *cons_seq, *cds))
                    code = kEvPremStop;
            }
        }

        // ── Emit events from pos_events → coo, update summaries ──────────────
        for (const auto& [pos, code] : pos_events) {
            coo.push_back({cell, pos, code});
            ++summ.n_loci_with_event;
            if (code >= kEvSubA && code <= kEvSubT) ++summ.n_substitutions;
            else if (code == kEvInsertion || code == kEvDeletion) ++summ.n_indels;
            else if (code == kEvFrameshift) ++summ.n_frameshifts;
            else if (code == kEvPremStop)   ++summ.n_premature_stops;
            // kEvComplex (9) increments no specific counter
        }

        // Recompute pct_heteroplasmic using final n_loci_with_event
        if (summ.mt_coverage_mean > 0.f && summ.n_loci_with_event > 0) {
            // n_covered is captured locally above; use mt_reads proxy
            // (coverage_mean = mt_reads / n_covered → n_covered = mt_reads / coverage_mean)
            float n_cov = summ.mt_coverage_mean > 0.f
                ? static_cast<float>(summ.mt_reads) / summ.mt_coverage_mean
                : 1.f;
            summ.pct_heteroplasmic =
                static_cast<float>(summ.n_loci_with_event) / n_cov * 100.f;
        }
    }  // for cell

    // ── Build sorted unique loci ──
    std::vector<uint32_t> unique_loci;
    for (const auto& e : coo) {
        unique_loci.push_back(e.mt_pos);
    }
    std::sort(unique_loci.begin(), unique_loci.end());
    unique_loci.erase(
        std::unique(unique_loci.begin(), unique_loci.end()),
        unique_loci.end());
    result.n_loci = static_cast<uint32_t>(unique_loci.size());

    // pos → locus index map
    std::vector<uint32_t> pos_to_locus(MT_LEN, UINT32_MAX);
    for (uint32_t li = 0; li < result.n_loci; ++li)
        pos_to_locus[unique_loci[li]] = li;

    // Row names: "chrM:<1-based-pos>"
    result.locus_names.resize(result.n_loci);
    for (uint32_t li = 0; li < result.n_loci; ++li)
        result.locus_names[li] = "chrM:" + std::to_string(unique_loci[li] + 1);

    // ── COO → CSC (loci × cells) ──
    // COO is already sorted by cell_idx (outer loop order above)
    result.indptr.resize(n_cells + 1, 0);
    result.indices.reserve(coo.size());
    result.data.reserve(coo.size());
    {
        size_t coo_i = 0;
        for (uint32_t cell = 0; cell < n_cells; ++cell) {
            while (coo_i < coo.size() && coo[coo_i].cell_idx == cell) {
                const uint32_t li = pos_to_locus[coo[coo_i].mt_pos];
                result.indices.push_back(static_cast<int32_t>(li));
                result.data.push_back(coo[coo_i].event_code);
                ++coo_i;
            }
            result.indptr[cell + 1] =
                static_cast<int32_t>(result.indices.size());
        }
    }

    // ── Build .1pz embedded metadata ──
    auto& m = result.pz_meta;
    m["schema_version"]           = "1.1";
    m["matrix_kind"]              = "mt_events";
    m["value_dtype"]              = "uint8";
    m["ref_build"]                = ref_build;
    m["ref_chrom"]                = "chrM";
    m["min_coverage_per_locus"]   = std::to_string(kMinCoveragePerLocus);
    m["min_alt_count_per_locus"]  = std::to_string(kMinAltCountPerLocus);
    m["heteroplasmy_threshold"]   = "0.02";
    m["event_code_map"] =
        R"({"1":"substitution_A","2":"substitution_C","3":"substitution_G",)"
        R"("4":"substitution_T","5":"insertion","6":"deletion",)"
        R"("7":"frameshift","8":"premature_stop","9":"complex"})";

    // donor_consensus: donor_id → hash of consensus string
    {
        std::string dc_json = "{";
        for (int d = 0; d < n_donors; ++d) {
            if (d > 0) dc_json += ",";
            dc_json += "\"donor" + std::to_string(d) + "\":\"md5:";
            dc_json += fnv32_hex(donor_consensus[d]);
            dc_json += "\"";
        }
        dc_json += "}";
        m["donor_consensus"] = dc_json;
    }

    result.donor_consensus = donor_consensus;
    return result;
}

/// Stats returned by write_mt_donor_outputs for embedding in summary.json.
struct MtDonorOutputStats {
    int      n_donors_with_consensus = 0;
    uint64_t n_mt_events_total       = 0;
};

// ============================================================================
// Top-level G6 orchestrator.
// Call from export.h after demux_thread.join().
// Creates mt/ subdirectory and writes all four G6 artifact families.
// Returns basic stats for embedding in summary.json.
// ============================================================================
inline MtDonorOutputStats write_mt_donor_outputs(
    const std::string& out_prefix,
    const SparseAccumulator<uint16_t>::CSCMatrix& mt_csc,
    const DemuxResult& demux_result,
    const std::vector<std::string>& barcodes,
    const std::map<std::string, std::string>& user_meta,
    const std::vector<MtIndelEvent>& mt_indels = {},
    const std::string& gtf_path = "")
{
    MtDonorOutputStats stats;
    if (mt_csc.data.empty() || mt_csc.ncols == 0) {
        std::cerr << "[mt_event_caller] No mt pileup data — G6 skipped\n";
        return stats;
    }

    const std::string mt_dir = out_prefix + "/mt";
    std::filesystem::create_directories(mt_dir);

    // Extract metadata values with safe fallbacks
    auto meta_get = [&](const std::string& k, const std::string& def) {
        auto it = user_meta.find(k);
        return it != user_meta.end() ? it->second : def;
    };
    const std::string ref_build =
        meta_get("reference_build", "GRCh38-2024-A");
    const std::string singlet_version =
        meta_get("singlet_version", "");

    const int n_donors = demux_result.n_donors_k;
    const auto& assignments = demux_result.assignments;

    // 1. Build per-donor consensus sequences
    auto donor_cons = build_donor_consensus(
        mt_csc.indptr.data(), mt_csc.indices.data(), mt_csc.data.data(),
        mt_csc.ncols, assignments, n_donors);

    // 2. Infer species reference from all cells (approximates rCRS)
    std::string species_ref = infer_ref_from_data(
        mt_csc.indptr.data(), mt_csc.indices.data(), mt_csc.data.data(),
        mt_csc.ncols);

    // 3. Write per-donor FASTA + VCF
    for (int d = 0; d < n_donors; ++d) {
        write_donor_consensus_fa(mt_dir, d, donor_cons[d], ref_build);
        write_donor_variants_vcf(mt_dir, d, donor_cons[d], species_ref,
                                 ref_build, singlet_version);
        // Count donors that have at least some consensus (not all-N)
        bool has_seq = false;
        for (char c : donor_cons[d]) { if (c != 'N') { has_seq = true; break; } }
        if (has_seq) ++stats.n_donors_with_consensus;
    }

    // 4. Load mt CDS regions for codes 7/8
    auto cds_regions = load_mt_cds_regions(gtf_path);

    // 5. Classify per-cell events (indels + subs + CDS annotation)
    MtEventResult evt = classify_mt_events(
        mt_csc.indptr.data(), mt_csc.indices.data(), mt_csc.data.data(),
        mt_csc.ncols, assignments, donor_cons, barcodes, ref_build,
        mt_indels, cds_regions);

    stats.n_mt_events_total = static_cast<uint64_t>(evt.data.size());

    // 6. Write mt_events.1pz
    {
        std::vector<int32_t>  write_indptr  = evt.indptr;
        std::vector<int32_t>  write_indices = evt.indices;
        std::vector<uint8_t>  write_data    = evt.data;
        std::vector<std::string> write_loci = evt.locus_names;

        if (evt.n_loci == 0) {
            write_indptr.assign(evt.n_cells + 1, 0);
        }

        pz::write_1pz(mt_dir + "/mt_events.1pz",
            evt.n_loci, evt.n_cells,
            write_indptr, write_indices, write_data,
            write_loci, barcodes,
            3, 1024, 4, evt.pz_meta);
    }

    // 7. Write mt_summary.tsv
    {
        const std::string tsv_path = mt_dir + "/mt_summary.tsv";
        std::ofstream tsv(tsv_path);
        if (!tsv.is_open()) {
            std::cerr << "[mt_event_caller] WARN: cannot write " << tsv_path << "\n";
        } else {
            tsv << "barcode\tdonor_id\tmt_reads\tmt_coverage_mean"
                   "\tn_loci_with_event\tn_substitutions\tn_indels"
                   "\tn_frameshifts\tn_premature_stops\tpct_heteroplasmic\n";
            for (uint32_t cell = 0; cell < evt.n_cells; ++cell) {
                const auto& s  = evt.summaries[cell];
                const std::string& bc =
                    (cell < barcodes.size()) ? barcodes[cell] : "";
                tsv << bc              << "\t"
                    << s.donor_label   << "\t"
                    << s.mt_reads      << "\t"
                    << s.mt_coverage_mean     << "\t"
                    << s.n_loci_with_event    << "\t"
                    << s.n_substitutions      << "\t"
                    << s.n_indels             << "\t"
                    << s.n_frameshifts        << "\t"
                    << s.n_premature_stops    << "\t"
                    << s.pct_heteroplasmic    << "\n";
            }
        }
    }

    // Count event codes present for log
    std::map<uint8_t,uint32_t> code_counts;
    for (uint8_t c : evt.data) ++code_counts[c];
    std::cerr << "[mt_event_caller] G6 complete:"
              << " n_donors=" << n_donors
              << " n_loci=" << evt.n_loci
              << " n_events=" << evt.data.size();
    for (const auto& [code, cnt] : code_counts)
        std::cerr << " code" << (int)code << "=" << cnt;
    std::cerr << "\n";

    return stats;
}

}  // namespace mt
}  // namespace singlet
