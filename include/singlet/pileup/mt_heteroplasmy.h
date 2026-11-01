// SPDX-License-Identifier: MIT
// singlet-pileup: mt_heteroplasmy.h
// Mitochondrial variant / heteroplasmy calling from the inline chrM base pileup.
// Replaces mgatk (Python/Snakemake) with a single-pass C++ implementation.
//
// Data model: SparseAccumulator<uint16_t> with feature = pos*4 + base_idx
//   base_idx: A=0, C=1, G=2, T=3
//   n_features = 16569 * 4 = 66276 (rCRS reference length)
// An optional second matrix of the same layout holds the REVERSE-strand subset of those counts
// (forward-strand count = total - reverse), enabling mgatk-style strand-concordance filters.
//
// Reference: alleles are always called against a real chrM reference (extract_mt_reference() from the STAR genome,
// GRCh38 chrM == rCRS, 16,569 bp), never against the pool majority. The pool-majority inference is kept only as a
// last-resort fallback (it hides every site where the pool differs from rCRS).
//
// Two kinds of call (validation: bench/modalities/mt_fix/):
//   * heteroplasmic sites (call_mt_variants): intermediate allele fractions, support from several cells, alt seen on
//     both strands, alt strand ratio matching the reference, over-dispersed across cells, not in a masked repeat.
//     These (and only these) become the PUBLIC mt_variants.tsv / mt_heteroplasmy matrix.
//   * homoplasmic haplotype variants (call_group_haplotypes): allele fraction >= 0.9 in a donor (group) pseudobulk.
//     These are genotype-like and go to the PRIVATE output directory (per-donor VCFs, private/mt_sites.tsv).

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "nucleotide_encoding.h"

namespace singlet {
namespace mt {

// rCRS reference length
static constexpr uint32_t MT_LEN = 16569;
static constexpr uint32_t MT_N_FEATURES = MT_LEN * 4;  // pos*4 + base

// Base encoding tables — canonical definitions live in nucleotide_encoding.h.
//   BASE_TO_IDX: htslib bam_seqi 4-bit code (1=A,2=C,4=G,8=T,15=N) -> 0-3 index
//   IDX_TO_BASE: 0-3 index -> ASCII char
using ::singlet::pileup::nt::BASE_TO_IDX;
using ::singlet::pileup::nt::IDX_TO_BASE;

// Inline feature index computation
inline uint32_t mt_feature(uint32_t pos, int base_idx) {
    return pos * 4 + static_cast<uint32_t>(base_idx);
}

/// ASCII base -> 0..3 (A,C,G,T), -1 otherwise.
inline int mt_base_index(char c) {
    switch (c) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default:  return -1;
    }
}

// A variable mt position detected across cells
struct MtVariant {
    uint32_t pos = 0;           // 0-based mt position
    char ref_allele = 'N';      // reference allele (chrM reference of the genome; rCRS for GRCh38)
    char alt_allele = 'N';      // alt allele of this call
    uint32_t n_cells_covered = 0;   // cells with >= min_cell_depth reads at the position
    uint32_t n_cells_het = 0;       // cells showing heteroplasmy (VAF entries in the matrix)
    float mean_vaf = 0.f;           // mean VAF across heteroplasmic cells
    // Evidence columns (appended to mt_variants.tsv)
    uint64_t pool_depth = 0;        // pooled depth at the position (called cells)
    float    pool_af = 0.f;         // pooled alt / depth
    uint32_t n_cells_alt = 0;       // cells with >= min_alt_reads_cell alt reads at depth >= min_evidence_depth
    uint32_t alt_fwd = 0, alt_rev = 0;
    float    strand_corr = std::numeric_limits<float>::quiet_NaN();    // per-cell fwd/rev alt-VAF correlation
    float    overdispersion = std::numeric_limits<float>::quiet_NaN(); // observed / binomial variance of cell VAF
    float    vmr = std::numeric_limits<float>::quiet_NaN();            // variance-to-mean ratio of cell VAF
};

/// Rejection reasons recorded per candidate site (bit mask).
enum MtFail : uint32_t {
    kFailFewCells    = 1,   ///< fewer than min_cells_alt carrier cells (heteroplasmy: cells significant against the
                            ///< site background rate; haplotype-like: cells with >= min_alt_reads_cell alt reads)
    kFailStrand      = 2,   ///< alt not on both strands / alt strand ratio differs from the reference's
    kFailStrandCorr  = 4,   ///< per-cell fwd/rev alt VAF not correlated
    kFailOverdisp    = 8,   ///< cell VAFs not over-dispersed vs binomial (uniform noise, not a clonal variant)
    kFailFewHetCells = 16,  ///< fewer than min_cells_het cells with a VAF entry
};

/// One evaluated candidate (position, alt allele) of the pooled caller.
struct MtSite {
    uint32_t pos = 0;
    char ref = 'N', alt = 'N';
    uint64_t depth = 0, alt_n = 0, alt_f = 0, alt_r = 0;
    /// position strand profile used for the strand test: reference-allele reads, or the reads of all alleles when the
    /// pool carries < 20 reference reads (positions where nearly everyone differs from the reference, e.g. 1438A>G)
    uint64_t ref_f = 0, ref_r = 0;
    uint32_t n_cells_evid = 0, n_cells_alt = 0, n_cells_hom = 0, n_cells_cov = 0, n_cells_het = 0;
    uint32_t n_cells_sig = 0;   ///< carrier cells whose alt count is significant against the site's background rate
    float mean_vaf = 0.f;
    float vmr = std::numeric_limits<float>::quiet_NaN();
    float overdispersion = std::numeric_limits<float>::quiet_NaN();
    float strand_corr = std::numeric_limits<float>::quiet_NaN();
    uint32_t fail = 0;
    uint8_t  cls = 0;   ///< 0 rejected, 1 heteroplasmic (public), 2 haplotype-like (private: most carrying cells ~homoplasmic)
};

// Per-cell heteroplasmy result at variable positions
struct MtHetResult {
    std::vector<MtVariant> variants;        // heteroplasmic sites (public)
    // Sparse matrix: variants x cells (float VAF values)
    // Stored as CSC for direct .1pz writing
    std::vector<int32_t> indptr;   // [n_cells + 1]
    std::vector<int32_t> indices;  // row indices (variant indices)
    std::vector<float> data;       // VAF values
    uint32_t n_variants = 0;       // nrows
    uint32_t n_cells = 0;          // ncols
    // Everything evaluated (all classes) — PRIVATE (haplotype-like sites are genotype-derived)
    std::vector<MtSite> sites;
    std::string ref_source;        // "reference" or "pool_majority_fallback"
    uint32_t n_ref_mismatch_pool = 0;  // positions where the pool majority base != reference (diagnostic)
};

// ---------------------------------------------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------------------------------------------

/// Known artefact-prone chrM intervals (1-based, inclusive): homopolymer / repeat / alignment-ambiguous.
///   302-316 poly-C (D-loop), 513-525 CA repeat, 3105-3109 (N at 3107 in rCRS), 16182-16194 poly-C (HVS-I),
///   plus 455-468 (poly-C), 566-573 (poly-C) and 12418-12425 (poly-A).
inline std::vector<std::pair<uint32_t, uint32_t>> default_mt_mask() {
    return {{302, 316}, {455, 468}, {513, 525}, {566, 573}, {3105, 3109}, {12418, 12425}, {16182, 16194}};
}

struct MtCallerConfig {
    // ---- per-cell
    int   min_cell_depth     = 10;    ///< per-cell depth required for a VAF entry in the heteroplasmy matrix
    int   min_alt_reads_cell = 2;     ///< alt reads a cell needs to count as carrying the allele / get a VAF entry
    int   min_evidence_depth = 3;     ///< per-cell depth for a cell to count as evidence (carrying or not)
    float min_af             = 0.01f; ///< matrix VAF window
    float max_af             = 0.95f;
    // ---- per-site
    int   min_pool_depth     = 20;    ///< pooled depth to evaluate a site
    int   min_pool_alt       = 5;     ///< pooled alt reads to be a candidate
    int   min_cells_alt      = 3;     ///< cells carrying the allele (>= min_alt_reads_cell reads)
    double sig_alpha         = 1e-6;  ///< a carrier cell is "significant" if P(X >= alt | depth, site background AF) < alpha
                                      ///< (3' scRNA reads carry ~0.1-2% systematic error; only cells whose alt count cannot be
                                      ///< error support a heteroplasmic call)
    int   min_cells_het      = 2;     ///< cells with a matrix VAF entry
    float hom_af             = 0.9f;  ///< a single cell is "homoplasmic alt" at VAF >= hom_af (haplotype-like site test)
    float max_hom_frac       = 0.5f;  ///< site is haplotype-like (not heteroplasmic) if more carrying cells than this are homoplasmic
    // ---- strand
    int   min_strand_alt     = 1;     ///< alt reads required on EACH strand when the position has both strands
    float strand_both_min    = 0.15f; ///< position "has both strands" if reference reverse fraction in [x, 1-x]
    float max_strand_delta   = 0.30f; ///< skewed positions: |alt reverse fraction - ref reverse fraction| bound
    float min_strand_corr    = 0.5f;  ///< per-cell fwd/rev alt-VAF Pearson r (mgatk uses 0.65 on deeper data)
    int   min_strand_corr_cells = 10; ///< cells covered on both strands needed to evaluate the correlation
    // ---- dispersion
    int   min_overdisp_cells = 20;    ///< evidence cells needed to evaluate over-dispersion
    float min_overdispersion = 1.0f;  ///< observed VAF variance / binomial variance (VMR-style); 1 = binomial, uniform error is not above it
    // ---- haplotype (donor / group pseudobulk) calls
    float hap_af             = 0.8f;  ///< donor pseudobulk alt fraction for a homoplasmic haplotype call. Below hom_af on purpose:
                                      ///< ambient chrM RNA from the other donors of the pool dilutes true donor-defining variants
                                      ///< to ~0.75-0.9 (sweep in bench/modalities/mt_fix/results: 0.9 -> P1.00/R0.51, 0.8 -> P0.87/R0.75
                                      ///< against AF>=0.9 truth)
    int   hap_min_depth      = 10;
    int   hap_min_alt        = 5;
    int   hap_min_cells      = 3;     ///< cells in the group with >= 1 alt read (capped at the group size)
    bool  hap_use_mask       = false; ///< homoplasmic donor variants in repeat tracts (310T>C, 16189T>C ...) are real haplotype
                                      ///< markers; the artefact mask is for heteroplasmy, so it is off for donor-level calls
    // ---- masks (1-based inclusive intervals)
    std::vector<std::pair<uint32_t, uint32_t>> mask = default_mt_mask();
    int   mask_pad           = 6;     ///< heteroplasmy only: alignment ambiguity of a repeat tract spreads a few bases either side
                                      ///< (e.g. a cluster of 6 adjacent "variants" shared by the same cells at 3099-3104, next to 3105-3109)
};

/// Boolean per-position mask from 1-based inclusive intervals.
inline std::vector<uint8_t> build_mt_mask(const std::vector<std::pair<uint32_t, uint32_t>>& iv, uint32_t pad = 0) {
    std::vector<uint8_t> m(MT_LEN, 0);
    for (const auto& r : iv) {
        const uint32_t lo = r.first > pad ? r.first - pad : 1, hi = r.second + pad;
        for (uint32_t p = std::max<uint32_t>(lo, 1); p <= hi; ++p)
            if (p - 1 < MT_LEN) m[p - 1] = 1;
    }
    return m;
}

/// Normalise a chrM reference: upper-case, exactly MT_LEN long (longer sequences are truncated — the pileup ignores
/// positions >= MT_LEN —, shorter ones padded with 'N' so those positions are never called), non-ACGT -> 'N'.
/// Returns an empty string when the input is empty.
inline std::string normalize_mt_reference(const std::string& seq) {
    if (seq.empty()) return {};
    std::string r(MT_LEN, 'N');
    const size_t n = std::min<size_t>(seq.size(), MT_LEN);
    for (size_t i = 0; i < n; ++i) {
        char c = seq[i];
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 32);
        r[i] = (mt_base_index(c) >= 0) ? c : 'N';
    }
    return r;
}

// Load rCRS reference from a file (16,569 bases, upper-case ACGT)
// Returns empty string on failure.
inline std::string load_rcrs(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::string ref;
    ref.reserve(MT_LEN);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '>') continue;  // skip FASTA header
        for (char c : line) {
            if (c >= 'A' && c <= 'Z') ref.push_back(c);
            else if (c >= 'a' && c <= 'z') ref.push_back(c - 32);
        }
    }
    return ref;
}

// Fallback only: pool-majority base per position (hides every site where the pool differs from the reference).
inline std::string infer_ref_from_data(
    const int32_t* indptr, const int32_t* indices,
    const uint16_t* data, uint32_t n_cells)
{
    std::string ref(MT_LEN, 'N');
    std::vector<std::array<uint64_t, 4>> pos_counts(MT_LEN, std::array<uint64_t, 4>{0, 0, 0, 0});
    const int64_t total_nnz = indptr[n_cells];
    for (int64_t i = 0; i < total_nnz; ++i) {
        const uint32_t feat = static_cast<uint32_t>(indices[i]);
        if (feat >= MT_N_FEATURES) continue;
        pos_counts[feat / 4][feat % 4] += data[i];
    }
    for (uint32_t pos = 0; pos < MT_LEN; ++pos) {
        int best = 0;
        for (int b = 1; b < 4; ++b)
            if (pos_counts[pos][b] > pos_counts[pos][best]) best = b;
        if (pos_counts[pos][best] > 0) ref[pos] = IDX_TO_BASE[best];
    }
    return ref;
}

// ---------------------------------------------------------------------------------------------------------------------
// Count access
// ---------------------------------------------------------------------------------------------------------------------

/// Non-owning view of a features x cells CSC allele-count matrix (feature = pos*4 + base).
struct MtCounts {
    const int32_t*  indptr  = nullptr;
    const int32_t*  indices = nullptr;
    const uint16_t* data    = nullptr;
    uint32_t        n_cells = 0;
    bool valid() const { return indptr != nullptr && n_cells > 0; }
};

/// Dense per-cell scratch of one cell's allele counts (total and reverse-strand), reused across cells.
class MtCellPileup {
public:
    struct Slot {
        uint32_t t[4] = {0, 0, 0, 0};   ///< total counts by base
        uint32_t r[4] = {0, 0, 0, 0};   ///< reverse-strand counts by base (clamped to t)
        bool seen = false;
    };
    MtCellPileup() : v_(MT_LEN) { touched_.reserve(4096); }

    void load(const MtCounts& tot, const MtCounts* rev, uint32_t cell) {
        clear();
        for (int32_t k = tot.indptr[cell]; k < tot.indptr[cell + 1]; ++k) {
            const uint32_t feat = static_cast<uint32_t>(tot.indices[k]);
            if (feat >= MT_N_FEATURES || tot.data[k] == 0) continue;
            Slot& s = v_[feat / 4];
            if (!s.seen) { s.seen = true; touched_.push_back(feat / 4); }
            s.t[feat % 4] += tot.data[k];
        }
        if (rev && rev->valid() && cell < rev->n_cells) {
            for (int32_t k = rev->indptr[cell]; k < rev->indptr[cell + 1]; ++k) {
                const uint32_t feat = static_cast<uint32_t>(rev->indices[k]);
                if (feat >= MT_N_FEATURES) continue;
                Slot& s = v_[feat / 4];
                if (!s.seen) continue;   // reverse counts are a subset of the totals
                s.r[feat % 4] += rev->data[k];
            }
            for (uint32_t p : touched_)
                for (int b = 0; b < 4; ++b)
                    if (v_[p].r[b] > v_[p].t[b]) v_[p].r[b] = v_[p].t[b];
        }
    }
    void clear() {
        for (uint32_t p : touched_) v_[p] = Slot();
        touched_.clear();
    }
    const std::vector<uint32_t>& positions() const { return touched_; }
    const Slot& at(uint32_t pos) const { return v_[pos]; }

private:
    std::vector<Slot> v_;
    std::vector<uint32_t> touched_;
};

inline bool mt_cell_selected(const std::vector<uint8_t>* mask, uint32_t c) {
    return mask == nullptr || mask->empty() || (c < mask->size() && (*mask)[c] != 0);
}

// ---------------------------------------------------------------------------------------------------------------------
// Strand concordance
// ---------------------------------------------------------------------------------------------------------------------

/// Is the strand distribution of the alt reads compatible with a real variant?
///  * positions covered on both strands (reference reverse fraction in [strand_both_min, 1-strand_both_min]) need alt
///    reads on BOTH strands (>= min_strand_alt each);
///  * strand-skewed positions (3' reads follow the transcript strand) need the alt reverse fraction within
///    max_strand_delta of the reference's, i.e. alt must not appear on the strand the reference does not use.
inline bool mt_strand_consistent(uint64_t alt_f, uint64_t alt_r, uint64_t ref_f, uint64_t ref_r,
                                 const MtCallerConfig& cfg)
{
    const uint64_t alt = alt_f + alt_r;
    if (alt == 0) return false;
    const uint64_t ref = ref_f + ref_r;
    const double alt_rev_frac = static_cast<double>(alt_r) / static_cast<double>(alt);
    if (ref < 10) {
        // Reference too thin to know the position's strand: require alt on both strands only when alt is abundant.
        return alt < 6 || (alt_f >= 1 && alt_r >= 1);
    }
    const double rho = static_cast<double>(ref_r) / static_cast<double>(ref);
    if (rho >= cfg.strand_both_min && rho <= 1.0 - cfg.strand_both_min) {
        return alt_f >= static_cast<uint64_t>(cfg.min_strand_alt) && alt_r >= static_cast<uint64_t>(cfg.min_strand_alt);
    }
    return std::fabs(alt_rev_frac - rho) <= cfg.max_strand_delta;
}

// ---------------------------------------------------------------------------------------------------------------------
// Pooled heteroplasmy caller
// ---------------------------------------------------------------------------------------------------------------------

namespace detail {
/// P(X >= k) for X ~ Binomial(n, p), summed upward from k in log space (terms fall off fast for small p).
inline double binom_tail(uint32_t n, uint32_t k, double p) {
    if (k == 0) return 1.0;
    if (k > n) return 0.0;
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return 1.0;
    const double lp = std::log(p), lq = std::log1p(-p);
    const double ln_fact_n = std::lgamma(n + 1.0);
    double tot = 0.0;
    for (uint32_t j = k; j <= n; ++j) {
        const double lt = ln_fact_n - std::lgamma(j + 1.0) - std::lgamma(n - j + 1.0) + j * lp + (n - j) * lq;
        const double t = std::exp(lt);
        tot += t;
        if (j > k && t < tot * 1e-12) break;
    }
    return std::min(1.0, tot);
}

struct CandAcc {
    // evidence cells (depth >= min_evidence_depth)
    uint32_t n_evid = 0, n_alt = 0, n_hom = 0, n_cov = 0, n_het = 0, n_sig = 0, n_alt1 = 0;
    struct Carrier { uint32_t depth, alt; float cell_err; };
    std::vector<Carrier> carriers;   // cells with >= min_alt_reads_cell alt reads
    double sum_v = 0, sum_v2 = 0, sum_inv_n = 0, sum_alt = 0, sum_dp = 0, sum_het_v = 0;
    // strand correlation of per-cell fwd / rev alt VAF (cells covered on both strands)
    uint32_t n_s = 0;
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
};
}  // namespace detail

/// Call heteroplasmic mt variants from pooled per-cell allele counts.
///
/// @param tot   total allele counts, features x cells (feature = pos*4 + base)
/// @param rev   reverse-strand subset of `tot` (same layout) or nullptr: without it strand filters are skipped
/// @param ref_sequence  chrM reference (any case; normalised to MT_LEN). Empty => pool-majority fallback.
/// @param cell_mask  per-cell 0/1 (e.g. called cells); empty/nullptr = all cells. Excluded cells keep their columns
///                   (empty) in the output matrix but contribute no evidence.
inline MtHetResult call_mt_variants(const MtCounts& tot, const MtCounts* rev, const std::string& ref_sequence,
                                    const MtCallerConfig& cfg = MtCallerConfig(),
                                    const std::vector<uint8_t>* cell_mask = nullptr)
{
    MtHetResult result;
    const uint32_t n_cells = tot.n_cells;
    result.n_cells = n_cells;
    result.indptr.assign(n_cells + 1, 0);
    if (!tot.valid()) return result;
    const bool have_rev = rev && rev->valid() && rev->n_cells == n_cells;
    const MtCounts* revp = have_rev ? rev : nullptr;

    std::string ref = normalize_mt_reference(ref_sequence);
    result.ref_source = "reference";
    if (ref.empty()) {
        ref = infer_ref_from_data(tot.indptr, tot.indices, tot.data, n_cells);
        result.ref_source = "pool_majority_fallback";
        std::cerr << "[mt] WARNING: no chrM reference — falling back to the pool majority base\n";
    }
    const std::vector<uint8_t> masked = build_mt_mask(cfg.mask, static_cast<uint32_t>(cfg.mask_pad));

    // ---- pass 1: pooled counts over selected cells
    std::vector<std::array<uint64_t, 4>> pT(MT_LEN, std::array<uint64_t, 4>{0, 0, 0, 0});
    std::vector<std::array<uint64_t, 4>> pR(MT_LEN, std::array<uint64_t, 4>{0, 0, 0, 0});
    for (uint32_t c = 0; c < n_cells; ++c) {
        if (!mt_cell_selected(cell_mask, c)) continue;
        for (int32_t k = tot.indptr[c]; k < tot.indptr[c + 1]; ++k) {
            const uint32_t feat = static_cast<uint32_t>(tot.indices[k]);
            if (feat < MT_N_FEATURES) pT[feat / 4][feat % 4] += tot.data[k];
        }
        if (revp) {
            for (int32_t k = revp->indptr[c]; k < revp->indptr[c + 1]; ++k) {
                const uint32_t feat = static_cast<uint32_t>(revp->indices[k]);
                if (feat < MT_N_FEATURES) pR[feat / 4][feat % 4] += revp->data[k];
            }
        }
    }
    for (uint32_t p = 0; p < MT_LEN; ++p)
        for (int b = 0; b < 4; ++b) if (pR[p][b] > pT[p][b]) pR[p][b] = pT[p][b];

    {   // diagnostic: where the pool majority disagrees with the reference
        for (uint32_t p = 0; p < MT_LEN; ++p) {
            const int ri = mt_base_index(ref[p]);
            if (ri < 0) continue;
            uint64_t d = pT[p][0] + pT[p][1] + pT[p][2] + pT[p][3];
            if (d < 100) continue;
            int best = 0;
            for (int b = 1; b < 4; ++b) if (pT[p][b] > pT[p][best]) best = b;
            if (best != ri) ++result.n_ref_mismatch_pool;
        }
    }

    // ---- candidates: (pos, alt)
    std::vector<int32_t> cand_of(static_cast<size_t>(MT_LEN) * 4, -1);
    std::vector<MtSite> cands;
    for (uint32_t p = 0; p < MT_LEN; ++p) {
        if (masked[p]) continue;
        const int ri = mt_base_index(ref[p]);
        if (ri < 0) continue;
        const uint64_t dp = pT[p][0] + pT[p][1] + pT[p][2] + pT[p][3];
        if (dp < static_cast<uint64_t>(cfg.min_pool_depth)) continue;
        for (int b = 0; b < 4; ++b) {
            if (b == ri || pT[p][b] < static_cast<uint64_t>(cfg.min_pool_alt)) continue;
            MtSite s;
            s.pos = p; s.ref = ref[p]; s.alt = IDX_TO_BASE[b];
            s.depth = dp; s.alt_n = pT[p][b];
            s.alt_r = pR[p][b]; s.alt_f = pT[p][b] - pR[p][b];
            if (pT[p][ri] >= 20) {
                s.ref_r = pR[p][ri]; s.ref_f = pT[p][ri] - pR[p][ri];
            } else {
                s.ref_r = pR[p][0] + pR[p][1] + pR[p][2] + pR[p][3];
                s.ref_f = dp - s.ref_r;
            }
            cand_of[static_cast<size_t>(p) * 4 + b] = static_cast<int32_t>(cands.size());
            cands.push_back(s);
        }
    }
    if (cands.empty()) return result;
    std::vector<detail::CandAcc> acc(cands.size());

    // ---- per-cell background error rate: non-reference fraction over positions where the pool shows no variant
    // (3' scRNA reads carry cell-specific error: damaged / low-quality cells show 10x the median). Shrunk towards
    // the pool-wide rate with 1000 pseudo-reads so shallow cells do not get noisy estimates.
    MtCellPileup cp;
    std::vector<float> cell_err(n_cells, 0.f);
    {
        std::vector<uint8_t> pos_ok(MT_LEN, 0);
        for (uint32_t p = 0; p < MT_LEN; ++p) {
            const int ri = mt_base_index(ref[p]);
            if (ri < 0 || masked[p]) continue;
            const uint64_t dp = pT[p][0] + pT[p][1] + pT[p][2] + pT[p][3];
            if (dp == 0) continue;
            if (static_cast<double>(dp - pT[p][ri]) / dp < 0.05) pos_ok[p] = 1;
        }
        std::vector<double> c_non(n_cells, 0.0), c_dp(n_cells, 0.0);
        double g_non = 0, g_dp = 0;
        for (uint32_t c = 0; c < n_cells; ++c) {
            if (!mt_cell_selected(cell_mask, c)) continue;
            cp.load(tot, nullptr, c);
            for (uint32_t p : cp.positions()) {
                if (!pos_ok[p]) continue;
                const auto& sl = cp.at(p);
                const int ri = mt_base_index(ref[p]);
                const double dp = sl.t[0] + sl.t[1] + sl.t[2] + sl.t[3];
                c_dp[c] += dp; c_non[c] += dp - sl.t[ri];
            }
            g_non += c_non[c]; g_dp += c_dp[c];
        }
        const double e0 = g_dp > 0 ? g_non / g_dp : 0.003;
        for (uint32_t c = 0; c < n_cells; ++c)
            cell_err[c] = static_cast<float>((c_non[c] + 1000.0 * e0) / (c_dp[c] + 1000.0));
    }

    // ---- pass 2: per-cell evidence at candidate positions
    for (uint32_t c = 0; c < n_cells; ++c) {
        if (!mt_cell_selected(cell_mask, c)) continue;
        cp.load(tot, revp, c);
        for (uint32_t p : cp.positions()) {
            const auto& s = cp.at(p);
            const int ri = mt_base_index(ref[p]);
            if (ri < 0) continue;
            const uint32_t depth = s.t[0] + s.t[1] + s.t[2] + s.t[3];
            if (depth == 0) continue;
            for (int b = 0; b < 4; ++b) {
                const int32_t ci = cand_of[static_cast<size_t>(p) * 4 + b];
                if (ci < 0) continue;
                detail::CandAcc& a = acc[ci];
                const uint32_t alt = s.t[b];
                const double vaf = static_cast<double>(alt) / depth;
                if (depth >= static_cast<uint32_t>(cfg.min_cell_depth)) {
                    ++a.n_cov;
                    if (alt >= static_cast<uint32_t>(cfg.min_alt_reads_cell) &&
                        vaf >= cfg.min_af && vaf <= cfg.max_af) {
                        ++a.n_het;
                        a.sum_het_v += vaf;
                    }
                }
                if (depth >= static_cast<uint32_t>(cfg.min_evidence_depth)) {
                    ++a.n_evid;
                    a.sum_v += vaf; a.sum_v2 += vaf * vaf;
                    a.sum_inv_n += 1.0 / depth;
                    a.sum_alt += alt; a.sum_dp += depth;
                    if (alt >= 1) ++a.n_alt1;
                    if (alt >= static_cast<uint32_t>(cfg.min_alt_reads_cell)) {
                        ++a.n_alt;
                        if (vaf >= cfg.hom_af) ++a.n_hom;
                        a.carriers.push_back({depth, alt, cell_err[c]});
                    }
                }
                if (revp) {
                    const double af_ = static_cast<double>(alt - s.r[b]);
                    const double ar_ = static_cast<double>(s.r[b]);
                    const double rf_ = static_cast<double>(s.t[ri] - s.r[ri]);
                    const double rr_ = static_cast<double>(s.r[ri]);
                    if (af_ + rf_ >= 1 && ar_ + rr_ >= 1) {
                        const double x = af_ / (af_ + rf_), y = ar_ / (ar_ + rr_);
                        ++a.n_s; a.sx += x; a.sy += y; a.sxx += x * x; a.syy += y * y; a.sxy += x * y;
                    }
                }
            }
        }
    }

    // ---- site decisions
    for (size_t i = 0; i < cands.size(); ++i) {
        MtSite& s = cands[i];
        const detail::CandAcc& a = acc[i];
        // background alt rate of the site: cells' alt presence capped at one read (a carrier cell adds at most one),
        // so strongly clonal variants do not inflate their own noise floor
        const double bg = a.sum_dp > 0 ? std::max(1e-4, static_cast<double>(a.n_alt1) / a.sum_dp) : 1e-4;
        uint32_t n_sig = 0;
        for (const auto& c : a.carriers)   // a cell must beat both the site background and its own error rate
            if (detail::binom_tail(c.depth, c.alt, std::max(bg, static_cast<double>(c.cell_err))) < cfg.sig_alpha) ++n_sig;
        s.n_cells_evid = a.n_evid; s.n_cells_alt = a.n_alt; s.n_cells_hom = a.n_hom; s.n_cells_sig = n_sig;
        s.n_cells_cov = a.n_cov;   s.n_cells_het = a.n_het;
        s.mean_vaf = a.n_het ? static_cast<float>(a.sum_het_v / a.n_het) : 0.f;
        if (a.n_evid > 1) {
            const double mean = a.sum_v / a.n_evid;
            const double var = std::max(0.0, a.sum_v2 / a.n_evid - mean * mean);
            if (mean > 0) s.vmr = static_cast<float>(var / mean);
            const double pbar = a.sum_dp > 0 ? a.sum_alt / a.sum_dp : 0.0;
            const double expv = pbar * (1.0 - pbar) * (a.sum_inv_n / a.n_evid);
            if (expv > 0) s.overdispersion = static_cast<float>(var / expv);
        }
        if (a.n_s >= static_cast<uint32_t>(cfg.min_strand_corr_cells)) {
            const double n = a.n_s;
            const double vx = a.sxx - a.sx * a.sx / n, vy = a.syy - a.sy * a.sy / n;
            const double cxy = a.sxy - a.sx * a.sy / n;
            if (vx > 1e-12 && vy > 1e-12) s.strand_corr = static_cast<float>(cxy / std::sqrt(vx * vy));
        }

        const bool hap_like = s.n_cells_alt > 0 &&
            static_cast<double>(s.n_cells_hom) / s.n_cells_alt > cfg.max_hom_frac;
        uint32_t fail = 0;
        const uint32_t carriers = hap_like ? s.n_cells_alt : s.n_cells_sig;
        if (carriers < static_cast<uint32_t>(cfg.min_cells_alt)) fail |= kFailFewCells;
        if (revp && !mt_strand_consistent(s.alt_f, s.alt_r, s.ref_f, s.ref_r, cfg)) fail |= kFailStrand;
        {   // the per-cell fwd/rev correlation only means something where the position is read on both strands
            const uint64_t rt = s.ref_f + s.ref_r;
            const double rho = rt ? static_cast<double>(s.ref_r) / rt : 0.0;
            const bool both = rt >= 10 && rho >= cfg.strand_both_min && rho <= 1.0 - cfg.strand_both_min;
            if (revp && both && !std::isnan(s.strand_corr) && s.strand_corr < cfg.min_strand_corr) fail |= kFailStrandCorr;
        }

        if (!hap_like) {
            if (static_cast<int>(s.n_cells_evid) >= cfg.min_overdisp_cells && !std::isnan(s.overdispersion) &&
                s.overdispersion < cfg.min_overdispersion)
                fail |= kFailOverdisp;
            if (s.n_cells_het < static_cast<uint32_t>(cfg.min_cells_het)) fail |= kFailFewHetCells;
        }
        s.fail = fail;
        if (fail == 0) s.cls = hap_like ? 2 : 1;
    }

    // ---- accepted heteroplasmic sites -> rows
    std::vector<int32_t> row_of_cand(cands.size(), -1);
    for (size_t i = 0; i < cands.size(); ++i) {
        const MtSite& s = cands[i];
        if (s.cls != 1) continue;
        MtVariant v;
        v.pos = s.pos; v.ref_allele = s.ref; v.alt_allele = s.alt;
        v.n_cells_covered = s.n_cells_cov; v.n_cells_het = s.n_cells_het; v.mean_vaf = s.mean_vaf;
        v.pool_depth = s.depth; v.pool_af = static_cast<float>(static_cast<double>(s.alt_n) / s.depth);
        v.n_cells_alt = s.n_cells_alt;
        v.alt_fwd = static_cast<uint32_t>(s.alt_f); v.alt_rev = static_cast<uint32_t>(s.alt_r);
        v.strand_corr = s.strand_corr; v.overdispersion = s.overdispersion; v.vmr = s.vmr;
        row_of_cand[i] = static_cast<int32_t>(result.variants.size());
        result.variants.push_back(v);
    }
    result.sites = std::move(cands);
    result.n_variants = static_cast<uint32_t>(result.variants.size());
    if (result.n_variants == 0) return result;

    // ---- pass 3: per-cell VAF entries at accepted sites (rows are in position order -> sorted within a column)
    for (uint32_t c = 0; c < n_cells; ++c) {
        if (mt_cell_selected(cell_mask, c)) {
            cp.load(tot, nullptr, c);
            std::vector<std::pair<int32_t, float>> ent;
            for (uint32_t p : cp.positions()) {
                const auto& s = cp.at(p);
                const uint32_t depth = s.t[0] + s.t[1] + s.t[2] + s.t[3];
                if (depth < static_cast<uint32_t>(cfg.min_cell_depth)) continue;
                for (int b = 0; b < 4; ++b) {
                    const int32_t ci = cand_of[static_cast<size_t>(p) * 4 + b];
                    if (ci < 0 || row_of_cand[ci] < 0) continue;
                    const uint32_t alt = s.t[b];
                    if (alt < static_cast<uint32_t>(cfg.min_alt_reads_cell)) continue;
                    const float vaf = static_cast<float>(alt) / static_cast<float>(depth);
                    if (vaf >= cfg.min_af && vaf <= cfg.max_af) ent.emplace_back(row_of_cand[ci], vaf);
                }
            }
            std::sort(ent.begin(), ent.end());
            for (const auto& e : ent) { result.indices.push_back(e.first); result.data.push_back(e.second); }
        }
        result.indptr[c + 1] = static_cast<int32_t>(result.indices.size());
    }
    return result;
}

/// Backward-compatible entry point (same signature as before): pooled caller with default filters, no strand data.
/// Prefer call_mt_variants() with a real reference and the reverse-strand matrix.
inline MtHetResult compute_heteroplasmy(
    const int32_t* indptr, const int32_t* indices,
    const uint16_t* data, uint32_t n_features, uint32_t n_cells,
    const std::string& ref_sequence = "",
    float min_af = 0.01f,
    float max_af = 0.95f,
    int min_coverage = 10,
    int min_cells_het = 2)
{
    (void)n_features;
    MtCounts tot; tot.indptr = indptr; tot.indices = indices; tot.data = data; tot.n_cells = n_cells;
    MtCallerConfig cfg;
    cfg.min_af = min_af; cfg.max_af = max_af;
    cfg.min_cell_depth = min_coverage; cfg.min_cells_het = min_cells_het;
    return call_mt_variants(tot, nullptr, ref_sequence, cfg, nullptr);
}

// ---------------------------------------------------------------------------------------------------------------------
// Group (donor) pseudobulk haplotype calls — homoplasmic variants. PRIVATE outputs only.
// ---------------------------------------------------------------------------------------------------------------------

struct MtHapCall {
    uint32_t pos = 0;
    char ref = 'N', alt = 'N';
    uint32_t dp = 0, ad_ref = 0, ad_alt = 0, alt_f = 0, alt_r = 0, n_cells_alt = 0;
    float af = 0.f;
};

/// Call per-group homoplasmic (AF >= hap_af) variants against the reference from group pseudobulks.
/// @param cell_group  per-cell group id (0..n_groups-1) or -1 to skip the cell
inline std::vector<std::vector<MtHapCall>> call_group_haplotypes(
    const MtCounts& tot, const MtCounts* rev,
    const std::vector<int>& cell_group, int n_groups,
    const std::string& ref_sequence, const MtCallerConfig& cfg = MtCallerConfig(),
    std::vector<std::string>* debug = nullptr)   // optional: one TSV line per near-miss with the reason it failed
{
    std::vector<std::vector<MtHapCall>> out(static_cast<size_t>(std::max(n_groups, 0)));
    const std::string ref = normalize_mt_reference(ref_sequence);
    if (n_groups <= 0 || !tot.valid() || ref.empty()) return out;
    const bool have_rev = rev && rev->valid() && rev->n_cells == tot.n_cells;
    const MtCounts* revp = have_rev ? rev : nullptr;
    const std::vector<uint8_t> masked = build_mt_mask(cfg.mask);

    struct G { uint32_t t[4] = {0, 0, 0, 0}; uint32_t r[4] = {0, 0, 0, 0}; uint16_t nc[4] = {0, 0, 0, 0}; };
    std::vector<G> g(static_cast<size_t>(n_groups) * MT_LEN);
    std::vector<uint32_t> group_cells(static_cast<size_t>(n_groups), 0);

    MtCellPileup cp;
    for (uint32_t c = 0; c < tot.n_cells && c < cell_group.size(); ++c) {
        const int d = cell_group[c];
        if (d < 0 || d >= n_groups) continue;
        ++group_cells[d];
        cp.load(tot, revp, c);
        for (uint32_t p : cp.positions()) {
            const auto& s = cp.at(p);
            G& e = g[static_cast<size_t>(d) * MT_LEN + p];
            for (int b = 0; b < 4; ++b) {
                e.t[b] += s.t[b]; e.r[b] += s.r[b];
                if (s.t[b] > 0 && e.nc[b] < 65535) ++e.nc[b];
            }
        }
    }

    // pooled (all groups) strand profile per position: the strand test compares a donor's alt reads with how the
    // position is read in the pool (reference-allele reads, or all reads where the pool has few reference reads)
    std::vector<std::array<uint64_t, 8>> pool(MT_LEN);
    for (uint32_t p = 0; p < MT_LEN; ++p) {
        pool[p].fill(0);
        for (int d = 0; d < n_groups; ++d) {
            const G& e = g[static_cast<size_t>(d) * MT_LEN + p];
            for (int b = 0; b < 4; ++b) { pool[p][b] += e.t[b]; pool[p][4 + b] += std::min(e.r[b], e.t[b]); }
        }
    }
    for (int d = 0; d < n_groups; ++d) {
        for (uint32_t p = 0; p < MT_LEN; ++p) {
            if (cfg.hap_use_mask && masked[p]) continue;
            const int ri = mt_base_index(ref[p]);
            if (ri < 0) continue;
            const G& e = g[static_cast<size_t>(d) * MT_LEN + p];
            const uint32_t dp = e.t[0] + e.t[1] + e.t[2] + e.t[3];
            if (dp < static_cast<uint32_t>(cfg.hap_min_depth)) continue;
            int best = -1;
            for (int b = 0; b < 4; ++b)
                if (b != ri && (best < 0 || e.t[b] > e.t[best])) best = b;
            if (best < 0) continue;
            const double af = static_cast<double>(e.t[best]) / dp;
            const uint64_t alt_r = std::min(e.r[best], e.t[best]);
            const uint64_t alt_f = e.t[best] - alt_r;
            uint64_t ref_r = pool[p][4 + ri], ref_f = pool[p][ri] - pool[p][4 + ri];
            if (pool[p][ri] < 20) {
                ref_r = pool[p][4] + pool[p][5] + pool[p][6] + pool[p][7];
                ref_f = pool[p][0] + pool[p][1] + pool[p][2] + pool[p][3] - ref_r;
            }
            const uint32_t need_cells = std::min<uint32_t>(static_cast<uint32_t>(cfg.hap_min_cells), group_cells[d]);
            const char* why = "ok";
            if (e.t[best] < static_cast<uint32_t>(cfg.hap_min_alt)) why = "few_alt_reads";
            else if (af < cfg.hap_af) why = "af";
            else if (e.nc[best] < need_cells) why = "few_cells";
            else if (revp && !mt_strand_consistent(alt_f, alt_r, ref_f, ref_r, cfg)) why = "strand";
            if (debug && af >= 0.5) {
                std::ostringstream o;
                o << d << "\t" << (p + 1) << "\t" << ref[p] << "\t" << IDX_TO_BASE[best] << "\t" << dp << "\t" << e.t[best]
                  << "\t" << alt_f << "\t" << alt_r << "\t" << ref_f << "\t" << ref_r << "\t" << e.nc[best]
                  << "\t" << af << "\t" << (masked[p] ? 1 : 0) << "\t" << why;
                debug->push_back(o.str());
            }
            if (why[0] != 'o') continue;
            MtHapCall h;
            h.pos = p; h.ref = ref[p]; h.alt = IDX_TO_BASE[best];
            h.dp = dp; h.ad_ref = e.t[ri]; h.ad_alt = e.t[best];
            h.alt_f = static_cast<uint32_t>(alt_f); h.alt_r = static_cast<uint32_t>(alt_r);
            h.n_cells_alt = e.nc[best]; h.af = static_cast<float>(af);
            out[d].push_back(h);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------------------------------------------------

// Write the heteroplasmic-site summary TSV (PUBLIC: heteroplasmic sites only).
inline void write_mt_variants(const std::string& path, const MtHetResult& het) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[mt] Cannot open: " << path << "\n";
        return;
    }
    f << "pos\tref\talt\tn_cells_covered\tn_cells_het\tmean_vaf"
         "\tpool_depth\tpool_af\tn_cells_alt\talt_fwd\talt_rev\tstrand_corr\toverdispersion\tvmr\n";
    auto num = [&](float x) { if (std::isnan(x)) f << "NA"; else f << x; };
    for (const auto& v : het.variants) {
        f << (v.pos + 1) << "\t"  // 1-based for output
          << v.ref_allele << "\t"
          << v.alt_allele << "\t"
          << v.n_cells_covered << "\t"
          << v.n_cells_het << "\t"
          << v.mean_vaf << "\t"
          << v.pool_depth << "\t" << v.pool_af << "\t" << v.n_cells_alt << "\t"
          << v.alt_fwd << "\t" << v.alt_rev << "\t";
        num(v.strand_corr); f << "\t"; num(v.overdispersion); f << "\t"; num(v.vmr); f << "\n";
    }
    std::cerr << "[mt] Wrote " << het.variants.size() << " heteroplasmic mt sites to " << path << "\n";
}

// Write EVERY evaluated candidate (all classes, with rejection reasons). PRIVATE: includes haplotype-like
// (donor-defining) sites.
inline void write_mt_sites_private(const std::string& path, const MtHetResult& het) {
    std::ofstream f(path);
    if (!f.is_open()) { std::cerr << "[mt] Cannot open: " << path << "\n"; return; }
    f << "pos\tref\talt\tclass\tfail\tdepth\talt_n\talt_fwd\talt_rev\tref_fwd\tref_rev\tn_cells_alt\tn_cells_hom\tn_cells_sig"
         "\tn_cells_evid\tn_cells_cov\tn_cells_het\tmean_vaf\tstrand_corr\toverdispersion\tvmr\n";
    auto num = [&](float x) { if (std::isnan(x)) f << "NA"; else f << x; };
    for (const auto& s : het.sites) {
        f << (s.pos + 1) << "\t" << s.ref << "\t" << s.alt << "\t"
          << (s.cls == 1 ? "het" : s.cls == 2 ? "haplotype" : "rejected") << "\t" << s.fail << "\t"
          << s.depth << "\t" << s.alt_n << "\t" << s.alt_f << "\t" << s.alt_r << "\t" << s.ref_f << "\t" << s.ref_r
          << "\t" << s.n_cells_alt << "\t" << s.n_cells_hom << "\t" << s.n_cells_sig << "\t" << s.n_cells_evid << "\t" << s.n_cells_cov
          << "\t" << s.n_cells_het << "\t" << s.mean_vaf << "\t";
        num(s.strand_corr); f << "\t"; num(s.overdispersion); f << "\t"; num(s.vmr); f << "\n";
    }
}

// Write per-cell allele count matrix (all 4 bases) as feature names
inline std::vector<std::string> mt_feature_names() {
    std::vector<std::string> names;
    names.reserve(MT_N_FEATURES);
    for (uint32_t pos = 0; pos < MT_LEN; ++pos) {
        for (int b = 0; b < 4; ++b) {
            names.push_back("chrM:" + std::to_string(pos + 1) + ":" + IDX_TO_BASE[b]);
        }
    }
    return names;
}

// Write variant feature names for the heteroplasmy matrix
inline std::vector<std::string> het_feature_names(const MtHetResult& het) {
    std::vector<std::string> names;
    names.reserve(het.variants.size());
    for (const auto& v : het.variants) {
        names.push_back("chrM:" + std::to_string(v.pos + 1) + ":" +
                         v.ref_allele + ">" + v.alt_allele);
    }
    return names;
}

/// A single mt indel event captured during CIGAR parsing.
struct MtIndelEvent {
    uint32_t bc_idx;  ///< cell barcode column index
    uint32_t pos;     ///< 0-based chrM anchor position
    uint16_t len;     ///< CIGAR op length in bases
    bool     is_ins;  ///< true=BAM_CINS, false=BAM_CDEL
};

} // namespace mt
} // namespace singlet
