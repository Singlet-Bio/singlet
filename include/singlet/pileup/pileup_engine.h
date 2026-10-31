// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: pileup_engine.h
// Core streaming BAM pileup engine for Pipeline V2.
// Reads BAM records (sorted or unsorted, file or stdin) and accumulates:
// 1. Per-exon counts (UMI-deduplicated) → gene-level derivation
// 2. Per-intron counts (UMI-deduplicated) → GeneFull = exon + intron
// 3. Splice junction counts from CIGAR N operations
// 4. SNP allele counts (AD/DP) at known positions (raw reads, no UMI dedup)
// 5. (RNA editing counts at REDIportal sites: NOT IMPLEMENTED; there is no site list, counter or writer. Not shipped in Beta.)
// 6. chrM reads extracted to separate BAM (for mgatk)
//
// Multi-mapper handling: reads with NH > 1 are counted as 1/NH fractional.
// UMI deduplication: (barcode, gene, UMI) triples deduplicated for exon/intron.
//
// All lookups are O(1) hash or O(log n) interval tree.

#include <htslib/sam.h>
#include <htslib/hts.h>
#include <htslib/bgzf.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <map>
#include <tuple>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <sched.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "stall_watchdog.h"
#include "ancestry.h"
#include "crispr_guide.h"
#include "gene_model.h"
#include "interval_tree.h"
#include "mt_heteroplasmy.h"
#include "em_rescue.h"
#include "sparse_accumulator.h"
#include "umi_dedup.h"
#include "cb_correct.h"
#include "vdj_counter.h"
#include "rrna_detect.h"
#include "polya_sites.h"
#include "gpu_pileup_dump.h"   // W11: SINGLET_PILEUP_DUMP / SINGLET_STREAM_TEE debug hooks (no effect unless set)
#include "gpu_pileup_bridge.h" // W11: SINGLET_GPU_PILEUP=compare (GPU builds only; no effect unless set)

namespace singlet {
// Timing and memory milestones of the end-of-stream finalize (stderr only, no effect on any output).
inline void fin_mark(const char* what) {
    static const auto t0 = std::chrono::steady_clock::now();
    static auto last = t0;
    const auto now = std::chrono::steady_clock::now();
    double rss = 0, anon = 0, hwm = 0;
    if (FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f)) {
            double kb;
            if (std::sscanf(line, "VmRSS: %lf", &kb) == 1) rss = kb;
            else if (std::sscanf(line, "RssAnon: %lf", &kb) == 1) anon = kb;
            else if (std::sscanf(line, "VmHWM: %lf", &kb) == 1) hwm = kb;
        }
        std::fclose(f);
    }
    std::fprintf(stderr, "[fin] %s: %.1f s (+%.1f s), rss %.1f GB, anon %.1f GB, peak %.1f GB\n", what,
                 std::chrono::duration<double>(now - t0).count(), std::chrono::duration<double>(now - last).count(),
                 rss / 1048576, anon / 1048576, hwm / 1048576);
    last = now;
}
}  // namespace singlet

namespace singlet {

// Compact SNP site representation
struct SNPSite {
    uint32_t index;     // column index in output matrix
    char ref_allele;    // reference allele
    char alt_allele;    // alternate allele
};

// Per-chromosome sorted SNP data for cache-friendly binary search pileup
struct ChrSNP {
    int32_t pos;        // 0-based position (sorted within chromosome)
    uint32_t index;     // column index in output matrix
    char ref_allele;
    char alt_allele;
};

// Configuration for the pileup engine
struct PileupConfig {
    // --stream-unsorted: STAR writes an unsorted BAM with no CB tag; the barcode is taken from CR. A read whose UR has an N or is
    // a homopolymer has no CB (STAR's UMI check). With cb_corrector (order-free-rules R11) the CR is corrected against STAR's
    // whitelist by STAR's 1MM_multi_Nbase_pseudocounts rule; without it only an N-free CR is taken, exactly. Serial run() only.
    bool cb_from_cr = false;
    const CbCorrector* cb_corrector = nullptr;   // owned by the caller, outlives the engine
    // Stream (order-free-rules R5): count the whole pass in the protocol orientation, no first-N-reads probe;
    // strand_probe() then reports the aggregate over all informative reads.
    bool strand_prior_only = false;
    // Stream: stop the pass early when the first reads show the library is far from the protocol orientation (antisense share above
    // kStrandEarlyAbortFrac over at least kStrandEarlyAbortMinReads informative reads). The caller then restarts on the sorted path,
    // whose strand probe handles any chemistry; the stats come back with strand_contradiction set. Serial run() only.
    bool strand_early_abort = false;
    std::string snp_path;           // VCF/TSV of SNP positions
    std::string exon_gtf_path;      // GTF/BED for exon intervals (used for GeneModel)
    std::string barcode_path;       // Filtered barcode list
    std::string out_chrm_bam;       // Output: chrM BAM
    int threads = 4;
    int finalize_threads = 1;       // end-of-stream finalize (molecule labelling); the result does not depend on it
    int min_mapq = 20;              // Minimum mapping quality
    int min_baseq = 10;             // Minimum base quality for the SNP pileup (there is no editing pileup)
    bool count_introns = true;      // Count intronic reads
    bool count_sj = true;           // Extract splice junctions
    bool umi_dedup = true;          // UMI deduplication for exon/intron
    bool umi_dedup_directional = false; // N6: directional 1-Hamming error correction (run() and run_parallel())
    bool reads_matrix = true;       // exon_reads / intron_reads matrices and the per-cell reads-per-UMI histogram (needs umi_dedup); never changes the counts
    int umi_len = 12;               // UMI length in bases (for N6 directional Hamming)
    bool stranded = true;           // Strand-aware counting (10x Forward)
    bool reverse_strand = false;    // Invert strand filter (for 5' protocols)
    bool auto_strand = true;        // Probe orientation on nuclear exonic reads; may set or clear reverse_strand
    bool count_mt = false;          // Inline chrM allele pileup for heteroplasmy
    bool count_polya = false;       // Record poly(A)-tailed read 3' ends (polya_sites.h); 3' protocols only
    bool aim_ancestry = true;          // N13: inject built-in AIM panel for ancestry classification
    std::string guide_ref_path;         // N18: CRISPR guide CSV (name,sequence); empty = disabled
    bool count_vdj = true;              // N17: V(D)J gene usage counting; requires exon_gtf_path
    // N22: Full-whitelist ambient profiling — gives EmptyDrops a representative empty-droplet pool
    std::string whitelist_path;         // Full barcode whitelist (e.g. 3M-february-2018.txt); empty = disabled
    uint32_t wl_ambient_ceil = 50;      // UMI ceiling: WL barcodes with ≤ this count define the ambient pool
    // Intron-assignment rules (intron_rule::k* bits). SINGLET_INTRON_RULES overrides it
    // (decimal or 0x hex) for experiments; see docs/pipeline-validation.md §4.4.
    // Default 389 = 133 (kUnique | kNoSplicedIntron | kUmiGeneCR) + 256 (kMtPrefer), chosen on real truth (sweep2: matches or beats every
    // other combination on pbmc v3/v2, nuclei, hgmm, CITE against STARsolo Velocyto and alevin-fry USA).
    uint32_t intron_rules = 389;
};

namespace intron_rule {
// Rules on reads (applied after the read's gene and exon/intron evidence are known)
constexpr uint32_t kUnique          = 1;    ///< a multi-mapped read (NH > 1) gives no intronic evidence
constexpr uint32_t kContained       = 2;    ///< an intron-touching read must lie inside its gene body
constexpr uint32_t kNoSplicedIntron = 4;    ///< a spliced read (CIGAR N) gives no intronic evidence
constexpr uint32_t kFlank           = 8;    ///< <= 5 intronic bases of an exon-touching read are exonic (velocyto MIN_FLANK)
constexpr uint32_t kMtPrefer        = 256;  ///< multimapper split between a chrM gene and a nuclear gene (NUMT): count the chrM gene
constexpr uint32_t kNoExonPriority  = 16;   ///< an exonic read of gene A inside an intron of gene B is not counted
// Rules on molecules (applied after UMI collapse, over all genes of a cell)
constexpr uint32_t kUmiGene         = 32;   ///< UMI in >1 gene of a cell: drop its unspliced/ambiguous molecules
constexpr uint32_t kUmiGeneAll      = 64;   ///< UMI in >1 gene of a cell: drop all its molecules (velocyto)
constexpr uint32_t kUmiGeneCR       = 128;  ///< UMI in >1 gene of a cell: keep the gene with most reads (ties: none)
constexpr uint32_t kMoleculeRules   = kUmiGene | kUmiGeneAll | kUmiGeneCR;
}  // namespace intron_rule

// Statistics collected during pileup
/// Per-cell alignment QC from the BAM records of one barcode (primary, barcoded reads; lost once BAMs are deleted).
struct CellAlignQC {
    uint32_t multimapped = 0;    ///< reads with NH > 1
    uint32_t unique = 0;         ///< reads with NH == 1
    uint64_t mismatches = 0;     ///< sum of STAR nM over unique reads
    uint64_t aligned_bases = 0;  ///< sum of M/=/X CIGAR lengths over unique reads (mismatch-rate denominator)
    void merge(const CellAlignQC& o) {
        multimapped += o.multimapped; unique += o.unique;
        mismatches += o.mismatches; aligned_bases += o.aligned_bases;
    }
};

inline void add_cell_align_qc(CellAlignQC& q, const bam1_t* b, int nh) {
    if (nh > 1) { ++q.multimapped; return; }
    ++q.unique;
    if (const uint8_t* nm = bam_aux_get(b, "nM")) {
        int64_t v = bam_aux2i(nm);
        if (v > 0) q.mismatches += static_cast<uint64_t>(v);
    }
    const uint32_t* cig = bam_get_cigar(b);
    for (uint32_t k = 0; k < b->core.n_cigar; ++k) {
        const int op = bam_cigar_op(cig[k]);
        if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) q.aligned_bases += bam_cigar_oplen(cig[k]);
    }
}

struct PileupStats {
    uint64_t total_reads = 0;
    uint64_t mapped_reads = 0;
    uint64_t barcoded_reads = 0;
    uint64_t snp_hits = 0;
    uint64_t exon_hits = 0;
    uint64_t intron_hits = 0;
    uint64_t sj_hits = 0;
    uint64_t chrm_reads = 0;
    uint64_t mt_pileup_bases = 0;
    uint64_t low_mapq = 0;
    uint64_t secondary_reads = 0;
    uint64_t no_barcode = 0;
    uint64_t no_umi = 0;
    uint64_t umi_unique = 0;
    uint64_t umi_duplicate = 0;
    uint64_t multimapper_reads = 0;
    uint64_t multigene_reads = 0;
    uint64_t wrong_strand = 0;
    // Nuclear primary reads on exons of one strand only, relative to the forward
    // (10x 3') orientation whatever orientation was counted (see StrandEvidence).
    uint64_t strand_sense = 0;
    uint64_t strand_antisense = 0;
    bool strand_contradiction = false;   // run() stopped early: strand_early_abort and the stream is far from the protocol orientation
    uint64_t vdj_hits = 0;
    // G-RRNA: rRNA contamination sampling
    uint64_t rrna_reads = 0;          ///< reads identified as rRNA (sampled subset)
    uint64_t rrna_reads_sampled = 0;  ///< total reads checked for rRNA
    // SNP allele consensus per (barcode, SNP, UMI) molecule (order-free-rules R1)
    uint64_t snp_molecules = 0;            ///< molecules with a REF or ALT majority (in DP)
    uint64_t snp_molecules_ambiguous = 0;  ///< tie between the two largest base classes: in neither AD nor DP
    uint64_t snp_molecules_other = 0;      ///< majority base is neither REF nor ALT: in neither AD nor DP
    // Molecule labels (each UMI-deduplicated (barcode, gene, UMI) counted once)
    uint64_t molecules_spliced = 0;
    uint64_t molecules_unspliced = 0;
    uint64_t molecules_ambiguous = 0;
    uint64_t molecules_legacy_double = 0;  ///< would have been in both exon and intron matrices before the fix
    uint64_t molecules_dropped_multigene_umi = 0;  ///< intron_rule::kUmiGene*: molecules of UMIs in >1 gene
    uint32_t intron_rules = 0;                     ///< intron_rule::k* bits the run used
    uint64_t mt_numt_rescued = 0;  ///< intron_rule::kMtPrefer: chrM/NUMT multimappers counted for the chrM gene
    uint64_t cross_gene_intron_reads = 0;  ///< exonic reads of one gene also in another gene's intron (old: counted for both)
    double wall_time_s = 0;
};

// Orientation-free strand evidence. A read is "sense" if it lies on exons of
// '+' genes when forward and '-' genes when reversed (the 10x 3' orientation),
// "antisense" if the opposite, and uninformative if it touches exons of both
// strands (or none).
struct StrandEvidence {
    uint64_t sense = 0;
    uint64_t antisense = 0;
    uint64_t informative() const { return sense + antisense; }
    double antisense_frac() const {
        return informative() ? static_cast<double>(antisense) / informative() : 0.0;
    }
};

// Strand decisions. The probe flips only on a clear majority; the post-pileup
// safety net (rerun_if_wrong_strand) re-runs whenever a finished pass counted
// more wrong-strand than right-strand exonic reads.
constexpr uint64_t kStrandProbeMinReads = 1000;     // indexed probe: minimum informative reads
constexpr uint64_t kStrandProbeTarget = 20000;      // indexed probe: informative reads wanted
constexpr uint64_t kStrandInlineProbeReads = 10000; // streaming probe: decide after this many
constexpr double kStrandProbeReverse = 0.6;         // antisense share above this -> reverse
constexpr double kStrandProbeForward = 0.4;         // antisense share below this -> forward
constexpr double kStrandRetryFrac = 0.5;            // safety net: wrong-strand share above this
constexpr uint64_t kStrandRetryMinReads = 1000;
constexpr uint64_t kStrandEarlyAbortMinReads = 200000;   // stream: informative reads before an early stop is considered
constexpr double kStrandEarlyAbortFrac = 0.9;            // stream: counted-antisense share that stops the pass (clear, order-robust)

/// Wrong-strand share of the exonic reads of a finished pass (wrong_strand over
/// wrong_strand + exon_hits). Reads touching exons of both strands, as most chrM
/// reads do, count as hits, so it understates a wrong orientation.
inline double wrong_strand_frac(const PileupStats& s) {
    uint64_t n = s.exon_hits + s.wrong_strand;
    return n ? static_cast<double>(s.wrong_strand) / n : 0.0;
}

/// Share of the nuclear strand-informative reads of a finished pass that are
/// antisense to the orientation it counted with (reverse = reverse_strand).
inline double counted_antisense_frac(const PileupStats& s, bool reverse) {
    uint64_t n = s.strand_sense + s.strand_antisense;
    return n ? static_cast<double>(reverse ? s.strand_sense : s.strand_antisense) / n : 0.0;
}

/// True when a stream pass should stop now and restart on the sorted path: enough informative reads, and a clear majority of them
/// antisense to the orientation being counted.
inline bool strand_early_abort_due(const PileupStats& s, bool reverse) {
    return s.strand_sense + s.strand_antisense >= kStrandEarlyAbortMinReads &&
           counted_antisense_frac(s, reverse) > kStrandEarlyAbortFrac;
}

/// mitochondrial contig names (genes on both strands; useless for the strand probe)
inline bool is_mito_contig(const std::string& name) {
    return name == "chrM" || name == "MT" || name == "chrMT" || name == "M";
}

// Lightweight profiling counters using RDTSC — near-zero overhead
// Enabled by defining SINGLET_PROFILE at compile time
#ifdef SINGLET_PROFILE
struct ProfileCounters {
    uint64_t t_bam_io = 0;           // sam_read1 calls
    uint64_t t_secondary = 0;        // secondary alignment gene tracking
    uint64_t t_barcode = 0;          // CB lookup + barcode filter
    uint64_t t_nh_mapq_umi = 0;      // NH + MAPQ + UMI extraction
    uint64_t t_exon_intron = 0;      // exon/intron interval tree queries
    uint64_t t_sj = 0;              // splice junction extraction
    uint64_t t_variant = 0;          // SNP/editing pileup (binary search + CIGAR walk)
    uint64_t t_mt_pileup = 0;        // chrM base pileup
    uint64_t t_mm_buffer = 0;        // multi-mapper buffer management
    uint64_t t_chrm_buf = 0;         // chrM bam_dup1 buffering
    uint64_t n_samples = 0;          // number of reads where timing was active

    static inline uint64_t rdtsc() {
        uint32_t lo, hi;
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }

    void print(double tsc_freq_ghz, double total_wall_s) const {
        double to_s = 1.0 / (tsc_freq_ghz * 1e9);
        auto pct = [&](uint64_t ticks) -> double {
            return (ticks * to_s) / total_wall_s * 100.0;
        };
        auto sec = [&](uint64_t ticks) -> double {
            return ticks * to_s;
        };
        std::cerr << "\n[PROFILE] Per-operation breakdown (sampled, extrapolated):\n"
                  << "  BAM I/O (sam_read1):  " << sec(t_bam_io) << "s (" << pct(t_bam_io) << "%)\n"
                  << "  Secondary tracking:   " << sec(t_secondary) << "s (" << pct(t_secondary) << "%)\n"
                  << "  Barcode lookup:       " << sec(t_barcode) << "s (" << pct(t_barcode) << "%)\n"
                  << "  NH/MAPQ/UMI:          " << sec(t_nh_mapq_umi) << "s (" << pct(t_nh_mapq_umi) << "%)\n"
                  << "  Exon/Intron queries:  " << sec(t_exon_intron) << "s (" << pct(t_exon_intron) << "%)\n"
                  << "  Splice junctions:     " << sec(t_sj) << "s (" << pct(t_sj) << "%)\n"
                  << "  Variant pileup:       " << sec(t_variant) << "s (" << pct(t_variant) << "%)\n"
                  << "  mt base pileup:       " << sec(t_mt_pileup) << "s (" << pct(t_mt_pileup) << "%)\n"
                  << "  Multi-mapper buffer:  " << sec(t_mm_buffer) << "s (" << pct(t_mm_buffer) << "%)\n"
                  << "  chrM buffering:       " << sec(t_chrm_buf) << "s (" << pct(t_chrm_buf) << "%)\n"
                  << "  Samples: " << n_samples << " reads\n";
        double accounted = sec(t_bam_io + t_secondary + t_barcode + t_nh_mapq_umi +
                               t_exon_intron + t_sj + t_variant + t_mt_pileup +
                               t_mm_buffer + t_chrm_buf);
        std::cerr << "  Accounted: " << accounted << "s (" << (accounted / total_wall_s * 100.0)
                  << "%), Unaccounted: " << (total_wall_s - accounted)
                  << "s (" << ((total_wall_s - accounted) / total_wall_s * 100.0) << "%)\n\n";
    }
};

// Estimate TSC frequency from a short calibration
static inline double calibrate_tsc_ghz() {
    auto c0 = ProfileCounters::rdtsc();
    auto t0 = std::chrono::steady_clock::now();
    // Busy-wait ~10ms
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.01) {}
    auto c1 = ProfileCounters::rdtsc();
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    return (c1 - c0) / elapsed / 1e9;
}

#define PROF_START(var) uint64_t _prof_##var = ProfileCounters::rdtsc()
#define PROF_END(pc, var) (pc).t_##var += ProfileCounters::rdtsc() - _prof_##var
#define PROF_BAM_IO_RESTART _bam_io_start = ProfileCounters::rdtsc()
#define PROF_BAM_IO_ACCUM prof.t_bam_io += ProfileCounters::rdtsc() - _bam_io_start
#else
#define PROF_START(var) ((void)0)
#define PROF_END(pc, var) ((void)0)
#define PROF_BAM_IO_RESTART ((void)0)
#define PROF_BAM_IO_ACCUM ((void)0)
#endif

// Inline CB tag scanner — avoids htslib bam_aux_get() function call overhead.
// Returns pointer to CB string value (past the 'Z' type byte), or nullptr if
// no CB:Z tag found.  O(1) for STARsolo BAMs where CB is the first aux tag.
static inline const char* inline_scan_cb(const bam1_t* b) {
    uint8_t* s = bam_get_aux(b);
    const uint8_t* end = b->data + b->l_data;
    while (s + 4 <= end) {
        if (s[0] == 'C' && s[1] == 'B' && s[2] == 'Z') {
            return reinterpret_cast<const char*>(s + 3);
        }
        s += 2;         // tag name
        uint8_t ty = *s++;
        if (ty == 'Z' || ty == 'H') {
            while (s < end && *s) ++s;
            ++s;        // skip null terminator
        } else if (ty == 'A' || ty == 'c' || ty == 'C') { s += 1; }
        else if (ty == 's' || ty == 'S') { s += 2; }
        else if (ty == 'i' || ty == 'I' || ty == 'f') { s += 4; }
        else if (ty == 'd') { s += 8; }
        else if (ty == 'B') {
            uint8_t sub = *s++;
            uint32_t n; memcpy(&n, s, 4); s += 4;
            s += n * ((sub == 'f' || sub == 'i' || sub == 'I') ? 4 :
                      (sub == 's' || sub == 'S') ? 2 : (sub == 'd') ? 8 : 1);
        } else break;
    }
    return nullptr;
}

// CB for --stream-unsorted (see PileupConfig::cb_from_cr). Returns a pointer to the CR value, or nullptr when STAR would have
// written no CB (N-containing barcode, or a UMI that is N-containing or a homopolymer). One pass over the aux tags.
static inline const char* inline_scan_cb_from_cr(const bam1_t* b, bool reject_n = true) {
    uint8_t* s = bam_get_aux(b);
    const uint8_t* end = b->data + b->l_data;
    const char* cr = nullptr; const char* ur = nullptr;
    while (s + 4 <= end && !(cr && ur)) {
        const char n0 = static_cast<char>(s[0]), n1 = static_cast<char>(s[1]); const uint8_t ty = s[2];
        if (ty == 'Z' || ty == 'H') {
            if (n0 == 'C' && n1 == 'R' && ty == 'Z') cr = reinterpret_cast<const char*>(s + 3);
            else if (n0 == 'U' && n1 == 'R' && ty == 'Z') ur = reinterpret_cast<const char*>(s + 3);
            s += 3; while (s < end && *s) ++s; ++s;
        } else if (ty == 'A' || ty == 'c' || ty == 'C') { s += 4; }
        else if (ty == 's' || ty == 'S') { s += 5; }
        else if (ty == 'i' || ty == 'I' || ty == 'f') { s += 7; }
        else if (ty == 'd') { s += 11; }
        else if (ty == 'B') {
            const uint8_t sub = s[3]; uint32_t n; memcpy(&n, s + 4, 4);
            s += 8 + static_cast<size_t>(n) * ((sub == 'f' || sub == 'i' || sub == 'I') ? 4 : (sub == 's' || sub == 'S') ? 2 : (sub == 'd') ? 8 : 1);
        } else break;
    }
    if (!cr || !ur || ur[0] == ' ') return nullptr;
    if (reject_n) for (const char* q = cr; *q; ++q) if (*q == 'N') return nullptr;
    bool homo = true;
    for (const char* q = ur; *q; ++q) { if (*q == 'N') return nullptr; if (*q != ur[0]) homo = false; }
    return homo ? nullptr : cr;
}

class PileupEngine {
public:
    explicit PileupEngine(const PileupConfig& config) : config_(config) {
        if (const char* r = std::getenv("SINGLET_INTRON_RULES"))
            config_.intron_rules = static_cast<uint32_t>(std::strtoul(r, nullptr, 0));
        if (const char* d = std::getenv("SINGLET_MOLECULE_DUMP")) molecule_dump_ = d;
    }

    // Reload only the barcode list from a new file, keeping all other references
    // (gene model, SNPs) intact. Used to refresh barcodes from STAR CB tags
    // after alignment, replacing raw decode-time barcodes with whitelist-corrected ones.
    bool reload_barcodes(const std::string& path) {
        config_.barcode_path = path;
        barcodes_.clear();
        bc_index_.map.clear();
        per_cell_reads_.clear();
        per_cell_qc_.clear();
        return load_barcodes();
    }

    // Load all reference data into memory
    bool load_references() {
        std::cerr << "[singlet-pileup] Loading references...\n";
        auto ref_t0 = std::chrono::high_resolution_clock::now();

        if (!load_barcodes()) return false;

        // N22: Load full whitelist for ambient profiling (I/O-bound; done before parallel SNP/GTF load)
        if (!config_.whitelist_path.empty() &&
            config_.whitelist_path != "None" && config_.whitelist_path != "none") {
            load_whitelist();
        }

        // Load SNPs and GTF in parallel (independent I/O-bound tasks)
        bool snp_ok = true, gtf_ok = true;

        bool need_snps = !config_.snp_path.empty();
        bool need_exons = !config_.exon_gtf_path.empty();

        std::thread snp_thread;
        std::thread gtf_thread;

        if (need_snps) {
            snp_thread = std::thread([&]() { snp_ok = load_snps(); });
        }

        if (need_exons) {
            gtf_thread = std::thread([&]() {
                const auto& path = config_.exon_gtf_path;
                bool is_bed = (path.find(".bed") != std::string::npos);
                gtf_ok = is_bed ? gene_model_.load_bed(path) : gene_model_.load_gtf(path);
                // N17: load VDJ intervals from same GTF (not BED)
                if (gtf_ok && !is_bed && config_.count_vdj) {
                    if (vdj_model_.load_gtf(path) && !vdj_model_.empty()) {
                        has_vdj_ = true;
                    }
                }
            });
        }

        if (snp_thread.joinable()) snp_thread.join();
        if (gtf_thread.joinable()) gtf_thread.join();

        if (!snp_ok || !gtf_ok) return false;
        if (need_exons) has_gene_model_ = true;

        // N13: inject built-in AIM panel (always, unless disabled)
        if (config_.aim_ancestry) {
            load_aim_panel();
        }

        // N18: load CRISPR guide reference if provided
        if (!config_.guide_ref_path.empty()) {
            if (!guide_ref_.load_csv(config_.guide_ref_path)) {
                std::cerr << "[singlet-pileup] WARNING: Failed to load guide ref: "
                          << config_.guide_ref_path << "\n";
            } else {
                std::cerr << "[singlet-pileup] Loaded " << guide_ref_.size()
                          << " CRISPR guides from " << config_.guide_ref_path << "\n";
            }
        }

        auto ref_t1 = std::chrono::high_resolution_clock::now();
        std::cerr << "[singlet-pileup] References loaded in "
                  << std::chrono::duration<double>(ref_t1 - ref_t0).count() << "s: "
                  << barcodes_.size() << " barcodes, "
                  << snp_pending_.size() << " SNP positions, "
                  << gene_model_.n_exons() << " exons, "
                  << gene_model_.n_introns() << " introns, "
                  << gene_model_.n_genes() << " genes\n";
        return true;
    }

    // Run the pileup on BAM from stdin (or file)
    PileupStats run(const std::string& bam_path = "-") {
        PileupStats stats;
        auto t0 = std::chrono::high_resolution_clock::now();

        // Open BAM input
        htsFile* fp = hts_open(bam_path.c_str(), "r");
        if (!fp) {
            std::cerr << "[singlet-pileup] ERROR: Cannot open BAM input: " << bam_path << "\n";
            return stats;
        }

        if (config_.threads > 1) {
            hts_set_threads(fp, config_.threads);
        }

        sam_hdr_t* hdr = sam_hdr_read(fp);
        if (!hdr) {
            std::cerr << "[singlet-pileup] ERROR: Cannot read BAM header\n";
            hts_close(fp);
            return stats;
        }

        // Map chromosome names and resolve pending references
        build_chrom_map(hdr);
        resolve_references();

        // Open chrM BAM output if requested
        // Deferred chrM BAM output: buffer reads during main loop, write after.
        // Writing chrM reads inline causes ~10s of Lustre back-pressure.
        // By deferring, the main loop runs at in-memory speed (~18-20s vs ~30s).
        htsFile* chrm_fp = nullptr;
        sam_hdr_t* chrm_hdr = nullptr;
        std::vector<bam1_t*> chrm_buffer;
        if (!config_.out_chrm_bam.empty()) {
            chrm_fp = hts_open(config_.out_chrm_bam.c_str(), "wb1");
            if (chrm_fp) {
                hts_set_threads(chrm_fp, 2);  // background compression/I/O for deferred write
                chrm_hdr = sam_hdr_dup(hdr);
                if (sam_hdr_write(chrm_fp, chrm_hdr) < 0) {
                    std::cerr << "[singlet-pileup] WARNING: Failed to write chrM BAM header\n";
                    hts_close(chrm_fp);
                    chrm_fp = nullptr;
                }
                chrm_buffer.reserve(4000000);  // ~3.8M chrM reads typical
            }
        }

        // Initialize accumulators
        uint32_t n_barcodes = static_cast<uint32_t>(barcodes_.size());
        per_cell_reads_.assign(n_barcodes, 0);  // N20
        per_cell_qc_.assign(n_barcodes, CellAlignQC{});
        snp_ad_.set_barcodes(barcodes_);
        snp_ad_.set_n_features(n_snps_loaded_);
        snp_dp_.set_barcodes(barcodes_);
        snp_dp_.set_n_features(n_snps_loaded_);

        if (has_gene_model_) {
            exon_acc_.set_barcodes(barcodes_);
            exon_acc_.set_n_features(gene_model_.n_exons());
            if (config_.count_introns) {
                intron_acc_.set_barcodes(barcodes_);
                intron_acc_.set_n_features(gene_model_.n_introns());
            }
            if (reads_matrix_on()) {
                exon_reads_acc_.set_barcodes(barcodes_);
                exon_reads_acc_.set_n_features(gene_model_.n_exons());
                if (config_.count_introns) {
                    intron_reads_acc_.set_barcodes(barcodes_);
                    intron_reads_acc_.set_n_features(gene_model_.n_introns());
                }
            }
        }

        // N22: Initialize WL ambient gene profile (sized after gene model is loaded)
        if (!wl_umi_counts_.empty() && has_gene_model_) {
            wl_ambient_gene_counts_.assign(gene_model_.n_genes(), 0ULL);
        }

        // Initialize mt allele accumulator (pos*4 + base_idx, 16569*4 = 66276 features)
        if (config_.count_mt) {
            mt_acc_.set_barcodes(barcodes_);
            mt_acc_.set_n_features(mt::MT_N_FEATURES);
            mt_acc_rev_.set_barcodes(barcodes_);
            mt_acc_rev_.set_n_features(mt::MT_N_FEATURES);
            mt_acc_.set_compact_limit(kMtCompactTriplets);
            mt_acc_rev_.set_compact_limit(kMtCompactTriplets);
        }

        // N18: Initialize CRISPR guide accumulator
        if (!guide_ref_.empty()) {
            guide_ctr_.init(guide_ref_.size(), barcodes_);
        }

        // N17: Initialize VDJ accumulator
        if (has_vdj_) {
            vdj_acc_.set_barcodes(barcodes_);
            vdj_acc_.set_n_features(vdj_model_.n_genes());
        }

        if (has_gene_model_ && config_.count_introns) {
            amb_acc_.set_barcodes(barcodes_);
            amb_acc_.set_n_features(gene_model_.n_genes());
        }

        // ── Multi-mapper gene-uniqueness buffer ──
        // STARsolo counts multi-mapped reads (NH > 1) if ALL alignments
        // map to the same gene ("gene-unique" multi-mappers). To replicate
        // this, we buffer multi-mapped reads and resolve across all alignments.
        struct MultiHitEntry {
            int32_t bc_idx = -1;
            uint32_t exon_idx = UINT32_MAX;     // assigned read's exon
            uint32_t gene_idx = UINT32_MAX;     // assigned gene (primary, else a secondary's exon gene)
            uint32_t intron_idx = UINT32_MAX;   // assigned read's intron
            uint8_t read_flags = 0;             // DirectionalUmiStore::kRead* of the assigned alignment
            uint8_t read_diag = 0;              // DirectionalUmiStore::kDiag* of the assigned alignment
            char umi[16] = {};
            uint8_t umi_len = 0;
            uint8_t nh = 0;
            uint8_t seen = 0;
            bool gene_ambiguous = false;
            bool primary_assigned = false;      // gene/exon came from the primary alignment
            uint32_t first_gene  = UINT32_MAX;   // first non-intergenic gene seen
            uint32_t second_gene = UINT32_MAX;   // conflicting gene for EM rescue
            // intron_rule::kMtPrefer: the alignment that lies in a chrM gene
            uint32_t mt_gene = UINT32_MAX, mt_exon = UINT32_MAX, mt_intron = UINT32_MAX;
            uint8_t mt_flags = 0, mt_diag = 0;
        };
        std::unordered_map<uint64_t, MultiHitEntry> mm_buffer;
        mm_buffer.reserve(500000);

        // Lambda to resolve a completed multi-hit entry
        auto resolve_mm = [&](MultiHitEntry& e, PileupStats& st) {
            if (e.bc_idx < 0) return;
            if (mt_numt_tie(e)) {   // chrM gene vs. its nuclear copy: count the chrM gene (see mt_numt_tie)
                count_read(mol_store_, exon_acc_, intron_acc_, st, static_cast<uint32_t>(e.bc_idx),
                           mt_rescue_features(e), e.umi_len > 0 ? e.umi : nullptr, e.umi_len);
                st.mt_numt_rescued++;
                return;
            }
            if (e.gene_ambiguous) {
                // G-EM: collect for post-pileup EM rescue
                if (e.first_gene != UINT32_MAX && e.second_gene != UINT32_MAX) {
                    em_rescue::AmbigRead ar;
                    ar.bc_idx  = e.bc_idx;
                    ar.umi_len = e.umi_len;
                    std::memcpy(ar.umi, e.umi, e.umi_len);
                    ar.gene_a  = e.first_gene;
                    ar.gene_b  = e.second_gene;
                    ambig_reads_.push_back(ar);
                }
                return;
            }
            // Gene-unique: first_gene is the only gene (or all intergenic)
            if (e.first_gene == UINT32_MAX || e.gene_idx == UINT32_MAX) return;
            ReadFeatures rf{e.gene_idx, e.exon_idx, e.intron_idx, e.read_flags,
                            static_cast<uint8_t>(e.read_diag | DirectionalUmiStore::kDiagMulti)};
            count_read(mol_store_, exon_acc_, intron_acc_, st, static_cast<uint32_t>(e.bc_idx),
                       rf, e.umi_len > 0 ? e.umi : nullptr, e.umi_len);
        };

        // ── Strand probe ──
        // An indexed BAM file is sampled across nuclear chromosomes before the
        // pass; a stream (STAR pipe, unsorted or unindexed BAM) is probed in-line.
        bool inline_probe = false;
        if (config_.stranded && config_.auto_strand && has_gene_model_ && !strand_probed_ && !config_.strand_prior_only)
            inline_probe = !probe_strand_file(bam_path, hdr);

        // ── Main read loop ──
        bam1_t* b = bam_init1();
        std::vector<uint32_t> exon_overlaps;
        std::vector<uint32_t> intron_overlaps;

#ifdef SINGLET_PROFILE
        ProfileCounters prof;
        double tsc_ghz = calibrate_tsc_ghz();
        std::cerr << "[PROFILE] TSC frequency: " << tsc_ghz << " GHz\n";
        uint64_t _bam_io_start = ProfileCounters::rdtsc();
#endif

        gpudump::StreamTee tee(hdr);   // W11 debug: SINGLET_STREAM_TEE=<file> keeps a copy of the record stream
#ifdef SINGLET_HAVE_ENGINE_GPU
        if (gpupileup::enabled()) gpu_col_.reset(new gpupileup::Collector(config_.min_baseq));   // W11: GPU pileup alongside, compared at the end
        gpu_covers_ = gpu_col_ && gpupileup::mode() == 2;   // W11 "on": the GPU replaces the covered steps below
#endif
        while (sam_read1(fp, hdr, b) >= 0) {
            tee.write(hdr, b);
#ifdef SINGLET_HAVE_ENGINE_GPU
            if (gpu_col_) {
                const char* gcb = config_.cb_from_cr ? stream_cb(b, false) : inline_scan_cb(b);   // R11: the collector sees the same corrected CB as the serial loop
                gpu_col_->add(b, (gcb && gcb[0] != '-') ? bc_index_.lookup(gcb) : -1);
            }
#endif
            // Prefetch aux data region into L2 cache while doing flag checks.
            __builtin_prefetch(bam_get_aux(b), 0, 2);
#ifdef SINGLET_PROFILE
            prof.t_bam_io += ProfileCounters::rdtsc() - _bam_io_start;
            prof.n_samples++;
#endif
            stats.total_reads++;

            if (stats.total_reads % 5000000 == 0) {
                singlet::rearm_stall_watchdog("pileup");   // progress: 5 M more reads
                std::cerr << "[singlet-pileup] Processed " << stats.total_reads / 1000000
                          << "M reads, " << stats.barcoded_reads << " barcoded, "
                          << stats.snp_hits << " SNP hits, "
                          << stats.exon_hits << " exon, "
                          << stats.intron_hits << " intron, "
                          << mm_buffer.size() << " mm_buf\n";
            }
            if (config_.strand_early_abort && (stats.total_reads & 0xFFFFF) == 0 &&
                strand_early_abort_due(stats, config_.reverse_strand)) {
                stats.strand_contradiction = true;   // the caller restarts on the sorted path
                break;
            }

            if (b->core.flag & BAM_FUNMAP) { PROF_BAM_IO_RESTART; continue; }
            stats.mapped_reads++;

            // G-RRNA: rRNA contamination on a 1/64 sample fixed by the read name, once per read
            // (order-free-rules R4)
            if (b->core.l_qseq >= 21 && !(b->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY))
                && (hash_qname(bam_get_qname(b)) & 63) == 0) {
                ++stats.rrna_reads_sampled;
                const uint8_t* qseq = bam_get_seq(b);
                if (qseq) {
                    std::string _rrna_seq(static_cast<size_t>(b->core.l_qseq), 'N');
                    for (int _qi = 0; _qi < b->core.l_qseq; ++_qi)
                        _rrna_seq[static_cast<size_t>(_qi)] = seq_nt16_str[bam_seqi(qseq, _qi)];
                    if (rrna_detector_.detect(_rrna_seq)) ++stats.rrna_reads;
                }
            }

            // Always skip supplementary alignments
            if (b->core.flag & BAM_FSUPPLEMENTARY) { PROF_BAM_IO_RESTART; continue; }

            bool is_secondary = (b->core.flag & BAM_FSECONDARY) != 0;

            // NH extraction is DEFERRED until after barcode check to avoid
            // ~70M bam_aux_get scans for non-barcoded reads.
            int32_t tid = b->core.tid;

            // ── Secondary multi-mapped alignment: track gene only ──
            if (is_secondary) {
                PROF_START(secondary);
                stats.secondary_reads++;
                if (has_gene_model_ && !gpu_covers_) {   // W11 "on": multimappers are resolved on the GPU
                    // Inline CB scanner — same as primary path, avoids htslib call overhead
                    const char* cb_sec_str = config_.cb_from_cr ? stream_cb(b, false) : inline_scan_cb(b);
                    if (cb_sec_str && cb_sec_str[0] != '-') {
                        int32_t bc_check = bc_index_.lookup(cb_sec_str);

                        // All alignments of the same read share the same barcode.
                        // If barcode is invalid, no entry exists in mm_buffer.
                        if (bc_check >= 0) {
                            // Defer NH extraction to here (only ~6% of secondaries)
                            int nh_sec = 1;
                            uint8_t* nh_tag = bam_aux_get(b, "NH");
                            if (nh_tag) { nh_sec = bam_aux2i(nh_tag); if (nh_sec < 1) nh_sec = 1; }

                            if (nh_sec > 1) {
                                uint64_t qh = hash_qname(bam_get_qname(b));
                                auto it = mm_buffer.find(qh);
                                if (it != mm_buffer.end()) {
                                    // Entry exists — update gene tracking
                                    auto hit = determine_exon_gene(b, tid);
                                    it->second.seen++;
                                    if (hit.multi_gene) {
                                        it->second.gene_ambiguous = true;
                                    } else if (hit.gene_idx != UINT32_MAX) {
                                        if (it->second.first_gene == UINT32_MAX) {
                                            it->second.first_gene = hit.gene_idx;
                                        } else if (hit.gene_idx != it->second.first_gene) {
                                            it->second.gene_ambiguous = true;
                                            // G-EM: record second candidate for rescue
                                            if (it->second.second_gene == UINT32_MAX)
                                                it->second.second_gene = hit.gene_idx;
                                        }
                                        // Without a primary assignment, the smallest exon over the
                                        // secondaries, not the first seen (order-free-rules R7)
                                        if (!it->second.primary_assigned &&
                                            (it->second.gene_idx == UINT32_MAX ||
                                             hit.exon_idx < it->second.exon_idx)) {
                                            it->second.gene_idx = hit.gene_idx;
                                            it->second.exon_idx = hit.exon_idx;
                                            it->second.read_flags = kExonicReadFlags;
                                        }
                                        note_mt_alignment(it->second, hit.gene_idx, hit.exon_idx, UINT32_MAX, kExonicReadFlags, 0);
                                    }
                                    if (it->second.seen >= it->second.nh && it->second.nh > 0) {
                                        resolve_mm(it->second, stats);
                                        mm_buffer.erase(it);
                                    }
                                } else {
                                    // Not in buffer yet — create entry
                                    auto hit = determine_exon_gene(b, tid);
                                    auto& entry = mm_buffer[qh];
                                    entry.seen++;
                                    if (entry.nh == 0) entry.nh = static_cast<uint8_t>(std::min(nh_sec, 255));
                                    if (hit.multi_gene) {
                                        entry.gene_ambiguous = true;
                                    } else if (hit.gene_idx != UINT32_MAX) {
                                        entry.first_gene = hit.gene_idx;
                                        entry.gene_idx = hit.gene_idx;
                                        entry.exon_idx = hit.exon_idx;
                                        entry.read_flags = kExonicReadFlags;
                                        note_mt_alignment(entry, hit.gene_idx, hit.exon_idx, UINT32_MAX, kExonicReadFlags, 0);
                                    }
                                }
                            }
                        }
                    }
                }
                PROF_END(prof, secondary);
                PROF_BAM_IO_RESTART;
                continue;
            }

            // ── Primary alignment processing ──
            // chrM extraction first (buffer for deferred write — no I/O in hot loop)
            PROF_START(chrm_buf);
            if (is_chrm(tid)) {
                if (chrm_fp) chrm_buffer.push_back(bam_dup1(b));
                stats.chrm_reads++;
            }
            PROF_END(prof, chrm_buf);

            // CB + barcode check (skip ~94% of primaries before NH extraction)
            PROF_START(barcode);
            const char* cb_str = config_.cb_from_cr ? stream_cb(b, true) : inline_scan_cb(b);
            if (cb_check_) check_cb_against_star(b, cb_str);
            if (!cb_str || cb_str[0] == '-') {
                stats.no_barcode++;
                PROF_END(prof, barcode);
                PROF_BAM_IO_RESTART;
                continue;
            }
            if (test_drop_read(b)) continue;
            int32_t bc_idx = bc_index_.lookup(cb_str);
            cur_read_gene_ = UINT32_MAX;
            if (bc_idx < 0) {
                // N22: WL ambient profiling — accumulate gene hits from WL-only barcodes
                if (!wl_umi_counts_.empty() &&
                    b->core.qual >= static_cast<uint8_t>(config_.min_mapq)) {
                    int32_t wl_idx = wl_amb_index_.lookup(cb_str);
                    if (wl_idx >= 0) {
                        wl_umi_counts_[static_cast<size_t>(wl_idx)]++;
                        if (has_gene_model_ && !wl_ambient_gene_counts_.empty()) {
                            auto hit = determine_exon_gene(b, tid);
                            if (hit.gene_idx != UINT32_MAX && !hit.multi_gene)
                                wl_ambient_gene_counts_[hit.gene_idx]++;
                        }
                    }
                }
                PROF_END(prof, barcode);
                PROF_BAM_IO_RESTART;
                continue;
            }
            stats.barcoded_reads++;
            if (static_cast<uint32_t>(bc_idx) < per_cell_reads_.size())
                per_cell_reads_[static_cast<uint32_t>(bc_idx)]++;  // N20
            PROF_END(prof, barcode);

            // Defer NH extraction to here (only ~6% of primaries reach this point)
            PROF_START(nh_mapq_umi);
            int nh = 1;
            {
                uint8_t* nh_tag = bam_aux_get(b, "NH");
                if (nh_tag) { nh = bam_aux2i(nh_tag); if (nh < 1) nh = 1; }
            }
            if (nh > 1) stats.multimapper_reads++;
            if (static_cast<uint32_t>(bc_idx) < per_cell_qc_.size())
                add_cell_align_qc(per_cell_qc_[static_cast<uint32_t>(bc_idx)], b, nh);

            bool high_mapq = (b->core.qual >= static_cast<uint8_t>(config_.min_mapq));
            if (!high_mapq && nh == 1) {
                stats.low_mapq++;
                PROF_END(prof, nh_mapq_umi);
                PROF_BAM_IO_RESTART;
                continue;  // unique read with low MAPQ — skip entirely
            }
            if (!high_mapq) stats.low_mapq++;

            // ── Extract UB (UMI) tag ──
            const char* umi_str = nullptr;
            int umi_len = 0;
            if (config_.umi_dedup) {
                uint8_t* ub_tag = bam_aux_get(b, "UB");
                if (!ub_tag) {
                    ub_tag = bam_aux_get(b, "UR");
                }
                if (ub_tag) {
                    umi_str = bam_aux2Z(ub_tag);
                    umi_len = static_cast<int>(strlen(umi_str));
                } else {
                    stats.no_umi++;
                }
            }
            PROF_END(prof, nh_mapq_umi);

            // N18: CRISPR guide counting (substring match on query sequence)
            if (!guide_ref_.empty()) {
                const uint8_t* qseq = bam_get_seq(b);
                int qlen = b->core.l_qseq;
                if (qseq && qlen > 0) {
                    // Decode BAM-encoded sequence into scratch buffer
                    guide_seq_buf_.resize(static_cast<size_t>(qlen));
                    for (int qi = 0; qi < qlen; ++qi)
                        guide_seq_buf_[qi] = seq_nt16_str[bam_seqi(qseq, qi)];
                    int gid = guide_ref_.match_bam(guide_seq_buf_.data(), qlen, (b->core.flag & BAM_FREVERSE) != 0);
                    if (gid >= 0) {
                        guide_ctr_.count(static_cast<uint32_t>(bc_idx),
                                         static_cast<uint32_t>(gid),
                                         umi_str, umi_len);
                    }
                }
            }

            // ── Exon + Intron counting ──
            if (has_gene_model_) {
                PROF_START(exon_intron);
                if (!gpu_covers_) {   // W11 "on": classification and everything that uses it comes from the GPU (gpu_pileup_apply)
                // Extract aligned blocks from CIGAR
                aligned_blocks_.clear();
                bool read_has_n = false;
                {
                    const uint32_t* cigar = bam_get_cigar(b);
                    int32_t rpos = b->core.pos;
                    for (int ci = 0; ci < b->core.n_cigar; ++ci) {
                        int op = bam_cigar_op(cigar[ci]);
                        int len = bam_cigar_oplen(cigar[ci]);
                        switch (op) {
                            case BAM_CMATCH:
                            case BAM_CEQUAL:
                            case BAM_CDIFF:
                                aligned_blocks_.push_back({rpos, rpos + len});
                                rpos += len;
                                break;
                            case BAM_CREF_SKIP:
                                read_has_n = true;
                                rpos += len;
                                break;
                            case BAM_CDEL:
                                rpos += len;
                                break;
                            case BAM_CINS:
                            case BAM_CSOFT_CLIP:
                            case BAM_CHARD_CLIP:
                            case BAM_CPAD:
                                break;
                        }
                    }
                }

                // Exon gene determination (any-overlap, strand-aware — matches STARsolo Gene feature)
                exon_overlaps.clear();
                bool read_reverse = (b->core.flag & BAM_FREVERSE) != 0;
                bool had_strand_reject = false;

                bool vote_sense = false, vote_anti = false;
                for (const auto& [bstart, bend] : aligned_blocks_) {
                    gene_model_.query_exons(tid, bstart, bend, block_overlaps_);
                    strand_vote(block_overlaps_, read_reverse, vote_sense, vote_anti);
                    size_t n_before = exon_overlaps.size();
                    for (uint32_t idx : block_overlaps_) {
                        bool wrong_strand = false;
                        if (config_.stranded) {
                            uint32_t g = gene_model_.exon_to_gene(idx);
                            char gs = gene_model_.gene_strand(g);
                            if (gs != '.') {
                                bool expect_reverse = (gs == '-');
                                if (config_.reverse_strand) expect_reverse = !expect_reverse;
                                if (read_reverse != expect_reverse) wrong_strand = true;
                            }
                        }
                        if (!wrong_strand) {
                            exon_overlaps.push_back(idx);
                        }
                    }
                    // Track when exons existed but all were filtered by strand (for auto-detect)
                    if (!block_overlaps_.empty() && exon_overlaps.size() == n_before) {
                        had_strand_reject = true;
                    }
                }

                if (exon_overlaps.empty() && had_strand_reject) {
                    stats.wrong_strand++;
                }

                if (vote_sense != vote_anti && !mito_tid_[tid] && !(b->core.flag & BAM_FSECONDARY))
                    (vote_sense ? stats.strand_sense : stats.strand_antisense)++;
                // Streaming strand probe: decide after kStrandInlineProbeReads
                // informative nuclear reads. Reads before the decision keep the
                // starting orientation (<0.1% of a typical run).
                if (inline_probe && stats.strand_sense + stats.strand_antisense >= kStrandInlineProbeReads) {
                    inline_probe = false;
                    apply_strand_probe({stats.strand_sense, stats.strand_antisense},
                                       "first nuclear exonic reads");
                }

                // Determine exon gene (any-overlap, single-gene requirement)
                uint32_t exon_gene = UINT32_MAX;
                uint32_t first_exon_idx = UINT32_MAX;
                bool primary_multigene = false;
                if (!exon_overlaps.empty()) {
                    uint32_t first_gene = gene_model_.exon_to_gene(exon_overlaps[0]);
                    bool unique_gene = (first_gene != UINT32_MAX);
                    for (size_t i = 1; i < exon_overlaps.size() && unique_gene; ++i) {
                        uint32_t g = gene_model_.exon_to_gene(exon_overlaps[i]);
                        if (g != first_gene) unique_gene = false;
                    }
                    if (unique_gene) {
                        exon_gene = first_gene;
                        first_exon_idx = exon_overlaps[0];
                    } else {
                        stats.multigene_reads++;
                        primary_multigene = true;
                    }
                }

                // Determine intron gene
                uint32_t intron_gene = UINT32_MAX;
                uint32_t first_intron_idx = UINT32_MAX;
                if (config_.count_introns) {
                    intron_overlaps.clear();
                    for (const auto& [bstart, bend] : aligned_blocks_) {
                        gene_model_.query_introns(tid, bstart, bend, block_overlaps_);
                        for (uint32_t idx : block_overlaps_) {
                            if (config_.stranded) {
                                uint32_t g = gene_model_.intron_to_gene(idx);
                                char gs = gene_model_.gene_strand(g);
                                if (gs != '.') {
                                    bool expect_reverse = (gs == '-');
                                    if (config_.reverse_strand) expect_reverse = !expect_reverse;
                                    if (read_reverse != expect_reverse) continue;
                                }
                            }
                            intron_overlaps.push_back(idx);
                        }
                    }

                    // Gene uniqueness check (no sort needed — duplicates don't affect result)
                    if (!intron_overlaps.empty()) {
                        uint32_t first_gene = gene_model_.intron_to_gene(intron_overlaps[0]);
                        bool unique_gene = (first_gene != UINT32_MAX);
                        for (size_t i = 1; i < intron_overlaps.size() && unique_gene; ++i) {
                            uint32_t g = gene_model_.intron_to_gene(intron_overlaps[i]);
                            if (g != first_gene) unique_gene = false;
                        }
                        if (unique_gene) {
                            intron_gene = first_gene;
                            first_intron_idx = intron_overlaps[0];
                        }
                    }
                }

                ReadFeatures rf = classify_read(exon_gene, first_exon_idx, primary_multigene,
                                                intron_overlaps, intron_gene, first_intron_idx,
                                                stats);
                refine_read(rf, aligned_blocks_, read_has_n, nh, tid, read_reverse, block_overlaps_);
                if (nh == 1) cur_read_gene_ = rf.gene;   // R9: evidence UMIs are corrected within (cell, gene)
                if (nh == 1 && config_.count_polya && !config_.reverse_strand && rf.gene != UINT32_MAX) {
                    uint64_t pk;
                    if (polya_site_from_read(b, 6, 0.8, pk))
                        polya_recs_.push_back({pk, rf.gene, static_cast<uint32_t>(bc_idx),
                                               polya_umi_hash(umi_str, umi_len)});
                }
                if (nh == 1) {
                    // ── Unique read: count (molecules are labelled at finalize) ──
                    count_read(mol_store_, exon_acc_, intron_acc_, stats,
                               static_cast<uint32_t>(bc_idx), rf, umi_str, umi_len);
                } else {
                    // ── Multi-mapped primary: buffer for cross-alignment resolution ──
                    uint64_t qh = hash_qname(bam_get_qname(b));
                    auto& entry = mm_buffer[qh];
                    entry.nh = static_cast<uint8_t>(std::min(nh, 255));
                    entry.seen++;
                    entry.bc_idx = bc_idx;
                    // The primary's assignment wins over one a secondary provided
                    if (rf.gene != UINT32_MAX) {
                        entry.primary_assigned = true;
                        entry.gene_idx = rf.gene;
                        entry.exon_idx = rf.exon_idx;
                        entry.intron_idx = rf.intron_idx;
                        entry.read_flags = rf.flags;
                        entry.read_diag = rf.diag;
                        note_mt_alignment(entry, rf.gene, rf.exon_idx, rf.intron_idx, rf.flags, rf.diag);
                    }
                    if (umi_str && umi_len > 0) {
                        int clen = std::min(umi_len, 15);
                        std::memcpy(entry.umi, umi_str, clen);
                        entry.umi[clen] = '\0';
                        entry.umi_len = static_cast<uint8_t>(umi_len);
                    }

                    // If primary position overlaps multiple genes, mark as ambiguous
                    if (primary_multigene) {
                        entry.gene_ambiguous = true;
                        // G-EM: extract two candidate genes from exon_overlaps for rescue
                        if (entry.second_gene == UINT32_MAX && !exon_overlaps.empty()) {
                            uint32_t ga = gene_model_.exon_to_gene(exon_overlaps[0]);
                            if (ga != UINT32_MAX) {
                                if (entry.first_gene == UINT32_MAX) entry.first_gene = ga;
                                for (size_t _i = 1; _i < exon_overlaps.size(); ++_i) {
                                    uint32_t gb = gene_model_.exon_to_gene(exon_overlaps[_i]);
                                    if (gb != UINT32_MAX && gb != ga) {
                                        entry.second_gene = gb; break;
                                    }
                                }
                            }
                        }
                    }

                    // Track gene across alignments (use exon gene preferentially)
                    uint32_t aln_gene = (exon_gene != UINT32_MAX) ? exon_gene : intron_gene;
                    if (aln_gene != UINT32_MAX) {
                        if (entry.first_gene == UINT32_MAX) {
                            entry.first_gene = aln_gene;
                        } else if (aln_gene != entry.first_gene) {
                            entry.gene_ambiguous = true;
                            // G-EM: aln_gene is the conflicting second candidate
                            if (entry.second_gene == UINT32_MAX)
                                entry.second_gene = aln_gene;
                        }
                    }

                    // Check if all alignments seen
                    if (entry.seen >= entry.nh && entry.nh > 0) {
                        resolve_mm(entry, stats);
                        mm_buffer.erase(qh);
                    }
                }

                // N17: VDJ gene usage counting
                if (has_vdj_ && !aligned_blocks_.empty()) {
                    vdj_overlaps_.clear();
                    int32_t vdj_read_start = aligned_blocks_.front().first;
                    int32_t vdj_read_end   = aligned_blocks_.back().second;
                    vdj_model_.query(tid, vdj_read_start, vdj_read_end, vdj_overlaps_);
                    if (!vdj_overlaps_.empty()) {
                        uint32_t vdj_gene = vdj_overlaps_[0];
                        if (config_.umi_dedup) {                  // R9: counted at finalize
                            vdj_umi_.add(static_cast<uint32_t>(bc_idx), vdj_gene, umi_str, umi_len, cur_read_gene_);
                        } else {
                            vdj_acc_.increment(vdj_gene, static_cast<uint32_t>(bc_idx));
                            stats.vdj_hits++;
                        }
                    }
                }

                }   // !gpu_covers_
                PROF_END(prof, exon_intron);

                // Splice junction extraction (always for primary alignments)
                if (config_.count_sj) {
                    PROF_START(sj);
                    if (!gpu_covers_) extract_splice_junctions(b, tid, static_cast<uint32_t>(bc_idx), umi_str, umi_len, stats);
                    PROF_END(prof, sj);
                }
            }

            // ── Position-level pileup (SNPs, editing) ──
            // Only use high-MAPQ reads for variant calling accuracy
            if (high_mapq) {
                PROF_START(variant);
                int32_t end_pos_v = gpu_covers_ ? 0 : bam_endpos(b);   // W11 "on": the GPU does the SNP pileup; skip the search too
                if (!gpu_covers_ && has_variant_overlap(tid, b->core.pos, end_pos_v)) {
                    pileup_aligned_bases(b, tid, static_cast<uint32_t>(bc_idx),
                                         umi_str, umi_len, stats);
                }
                PROF_END(prof, variant);
            }

            // ── Inline chrM base pileup for heteroplasmy ──
            // MAPQ gate: multimapped reads (NH>1, low MAPQ; e.g. NUMT-derived) reach this point, so require high_mapq here.
            if (config_.count_mt && high_mapq && is_chrm(tid)) {
                PROF_START(mt_pileup);
                if (!gpu_covers_) pileup_mt_bases(b, static_cast<uint32_t>(bc_idx), stats);
                PROF_END(prof, mt_pileup);
            }
            PROF_BAM_IO_RESTART;
        }
        PROF_BAM_IO_ACCUM;  // account for last failed sam_read1

        if (gpu_covers_) std::cerr << "[gpu_pileup] stream drained\n";   // S8: log timestamps place the post-stream steps
        // ── Flush remaining multi-mapper buffer entries ──
        // These are reads where not all NH alignments appeared in the BAM
        uint64_t mm_flushed = 0;
        for (auto& [qname, entry] : mm_buffer) {
            resolve_mm(entry, stats);
            mm_flushed++;
        }
        mm_buffer.clear();
        if (mm_flushed > 0) {
            std::cerr << "[singlet-pileup] Flushed " << mm_flushed
                      << " incomplete multi-mapped entries\n";
        }

#ifdef SINGLET_HAVE_ENGINE_GPU
        if (gpu_covers_ && !gpu_handoff_pass(stats)) {
            std::cerr << "[gpu_pileup] ERROR: the hand-off pass failed (see above); rerun with SINGLET_GPU_PILEUP unset\n";
            std::exit(9);
        }
        if (gpu_covers_ && !gpu_pileup_apply(stats)) {   // W11 "on": covered accumulators and counters from the GPU
            std::cerr << "[gpu_pileup] ERROR: the GPU pileup failed after the CPU skipped its steps; rerun with SINGLET_GPU_PILEUP unset\n";
            std::exit(9);
        }
#endif
        // ── Order-free finalize (docs/research-beta/order-free-rules.md) ──
        fin_mark("stream done, finalize starts");
        snp_cons_.finalize(snp_ad_, snp_dp_, stats, config_.umi_dedup_directional, gene_roots(mol_store_));  // R1 over R9
        fin_mark("SNP consensus finalize");
        snp_cons_.clear();
        fin_mark("SNP consensus freed");
        finalize_umi_features(sj_umi_dedup_, sj_acc_, stats.sj_hits, mol_store_);   // R9, before the molecule store
        fin_mark("junction UMI finalize");
#ifdef SINGLET_HAVE_ENGINE_GPU
        if (gpu_covers_) {   // the gene store lives on the GPU
            if (!vdj_umi_.empty()) {
                stats.vdj_hits += vdj_umi_.finalize([&](uint32_t bc, uint32_t f, uint32_t n) { vdj_acc_.increment(f, bc, n); },
                                                    config_.umi_dedup_directional, GpuGeneRoots{&gpu_merges_});
                vdj_umi_.clear();
            }
        } else
#endif
        finalize_umi_features(vdj_umi_, vdj_acc_, stats.vdj_hits, mol_store_);      // is finalized below
        fin_mark("VDJ UMI finalize");
        canonicalize_splice_junctions();               // R7
        fin_mark("junctions canonicalized");
        if (config_.strand_prior_only)                  // R5: report the aggregate over all informative reads
            strand_probe_ = StrandEvidence{stats.strand_sense, stats.strand_antisense};

        // ── UMI collapse + spliced/unspliced/ambiguous labelling ──
        {
            std::cerr << "[singlet-pileup] Molecule labelling ("
                      << mol_store_.n_molecule_groups() << " cell×gene groups, "
                      << (config_.umi_dedup_directional ? "directional" : "exact")
                      << " UMI collapse)...\n";
            auto ms = finalize_all({{&mol_store_, &exon_acc_, &intron_acc_, &amb_acc_, true}});
            fin_mark("molecule labelling done");
            add_molecule_stats(stats, ms);
#ifdef SINGLET_HAVE_ENGINE_GPU
            if (gpu_covers_) add_molecule_stats(stats, gpu_ms_);   // W11 "on": the GPU's molecules (the CPU store is empty)
#endif
        }
        if (config_.umi_dedup && config_.umi_dedup_directional) {
            uint64_t exon_deduped = stats.molecules_spliced + stats.molecules_ambiguous;
            uint64_t intron_deduped = stats.molecules_unspliced;
            std::cerr << "[singlet-pileup] Directional dedup: "
                      << exon_deduped << " exon + " << intron_deduped << " intron unique UMIs\n";
        }

        bam_destroy1(b);
        sam_hdr_destroy(hdr);
        hts_close(fp);

        auto t1 = std::chrono::high_resolution_clock::now();
        stats.wall_time_s = std::chrono::duration<double>(t1 - t0).count();

#ifdef SINGLET_PROFILE
        prof.print(tsc_ghz, stats.wall_time_s);
#endif

        // ── Deferred chrM BAM write (outside the processing timer) ──
        if (chrm_fp) {
            auto tc0 = std::chrono::high_resolution_clock::now();
            for (auto* r : chrm_buffer) {
                if (sam_write1(chrm_fp, chrm_hdr, r) < 0) {
                    std::cerr << "[singlet-pileup] WARNING: Failed to write chrM read\n";
                }
                bam_destroy1(r);
            }
            chrm_buffer.clear();
            if (chrm_hdr) sam_hdr_destroy(chrm_hdr);
            hts_close(chrm_fp);
            auto tc1 = std::chrono::high_resolution_clock::now();
            std::cerr << "[singlet-pileup] chrM BAM write: "
                      << stats.chrm_reads << " reads in "
                      << std::chrono::duration<double>(tc1 - tc0).count() << "s\n";
        }

        report_cb_status();   // R11
        report_cb_check();
        std::cerr << "[singlet-pileup] Done: " << stats.total_reads << " reads, "
                  << stats.mapped_reads << " mapped, "
                  << stats.barcoded_reads << " barcoded, "
                  << stats.secondary_reads << " secondary, "
                  << stats.snp_hits << " SNP hits, "
                  << stats.exon_hits << " exon hits, "
                  << stats.intron_hits << " intron hits, "
                  << stats.sj_hits << " SJ, "
                  << stats.chrm_reads << " chrM, "
                  << stats.mt_pileup_bases << " mt_bases, "
                  << stats.multigene_reads << " multigene, "
                  << stats.wrong_strand << " wrong_strand, "
                  << stats.vdj_hits << " vdj_hits, "
                  << "UMI " << stats.umi_unique << " unique / "
                  << stats.umi_duplicate << " dup in "
                  << stats.wall_time_s << "s\n";

        gpudump_write(static_cast<uint32_t>(tid_to_chrom_.size()));   // W11 debug (SINGLET_PILEUP_DUMP); hdr is already freed here
#ifdef SINGLET_HAVE_ENGINE_GPU
        gpu_pileup_compare();   // W11 (SINGLET_GPU_PILEUP=compare)
#endif
        return stats;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Region-parallel BAM processing using BAM index (.bai)
    // Each worker opens an independent BAM handle, seeks to assigned
    // chromosomes, and processes reads with fully local state.
    // Gives near-linear speedup on multi-core nodes.
    // ═══════════════════════════════════════════════════════════════════

    // Per-worker mutable state (no sharing between workers)
    struct WorkerContext {
        PileupStats stats;
        SparseAccumulator<uint16_t> snp_ad, snp_dp;
        SparseAccumulator<uint16_t> exon, intron;

        // SJ tracking: per-worker local index + names
        std::unordered_map<uint64_t, uint32_t> sj_index;
        std::vector<std::string> sj_names;
        SparseAccumulator<uint16_t> sj;

        // chrM allele counts (if count_mt enabled)
        SparseAccumulator<uint16_t> mt;
        // chrM reverse-strand allele counts (subset of `mt`; forward = mt - mt_rev)
        SparseAccumulator<uint16_t> mt_rev;

        // chrM indel events captured during CIGAR walk
        std::vector<mt::MtIndelEvent> mt_indels;

        UmiFeatureCollapse sj_umi;   // R9: junction molecules, collapsed at finalize
        uint32_t cur_gene = UINT32_MAX;  // R9: gene of the current unique read
        SnpConsensus snp_cons;  // per-molecule SNP allele consensus (order-free-rules R1)
        // Gene-model molecules (per worker, labelled and finalized after thread join)
        DirectionalUmiStore mol_store;
        SparseAccumulator<uint16_t> amb;  // ambiguous molecules per gene

        // N18: CRISPR guide counts (per-worker)
        SparseAccumulator<uint16_t> guide;
        UmiDedup guide_umi;
        std::string guide_seq_buf;  // per-worker scratch for BAM seq decode

        // N17: VDJ gene usage (per-worker)
        SparseAccumulator<uint16_t> vdj;
        UmiFeatureCollapse vdj_umi;  // R9: V(D)J molecules, collapsed at finalize
        std::vector<uint32_t> vdj_overlaps;  // scratch

        // Multi-mapper buffer (per-worker)
        struct MultiHitEntry {
            int32_t bc_idx = -1;
            uint32_t exon_idx = UINT32_MAX;
            uint32_t gene_idx = UINT32_MAX;
            uint32_t intron_idx = UINT32_MAX;
            uint8_t read_flags = 0;
            uint8_t read_diag = 0;
            char umi[16] = {};
            uint8_t umi_len = 0;
            uint8_t nh = 0;
            uint8_t seen = 0;
            bool gene_ambiguous = false;
            bool primary_assigned = false;
            uint32_t first_gene  = UINT32_MAX;
            uint32_t second_gene = UINT32_MAX;  // conflicting gene for EM rescue
            // intron_rule::kMtPrefer: the alignment that lies in a chrM gene
            uint32_t mt_gene = UINT32_MAX, mt_exon = UINT32_MAX, mt_intron = UINT32_MAX;
            uint8_t mt_flags = 0, mt_diag = 0;
        };
        std::unordered_map<uint64_t, MultiHitEntry> mm_buffer;

        // G-EM: per-worker ambiguous reads for EM rescue
        std::vector<em_rescue::AmbigRead> ambig_reads;

        // chrM buffer (per-worker)
        std::vector<bam1_t*> chrm_buffer;

        // Scratch vectors (per-worker, avoids allocation in hot loop)
        std::vector<std::pair<int32_t, int32_t>> aligned_blocks;
        std::vector<uint32_t> block_overlaps;
        std::vector<uint32_t> broad_overlap_genes;
        std::vector<uint32_t> mm_overlaps;
        std::vector<uint32_t> exon_overlaps;
        std::vector<uint32_t> intron_overlaps;
        // N20: per-cell read counts
        std::vector<uint32_t> cell_reads;
        std::vector<CellAlignQC> cell_qc;  // per-cell alignment QC (multimappers, nM mismatches)
        std::vector<PolyARecord> polya;    // poly(A)-tailed read ends (count_polya)
        // G-RRNA: per-worker rRNA detector
        singlet::RrnaDetector rrna_detector;
        // N22: per-worker WL ambient gene profile (populated when whitelist_path is set)
        std::vector<uint64_t> wl_ambient_genes;
    };

    PileupStats run_parallel(const std::string& bam_path, int n_workers) {
        auto t0 = std::chrono::high_resolution_clock::now();
        (void)t0;

        // ── Open BAM, load header, check index ──
        htsFile* master_fp = hts_open(bam_path.c_str(), "r");
        if (!master_fp) {
            std::cerr << "[parallel] ERROR: Cannot open BAM: " << bam_path << "\n";
            return {};
        }
        sam_hdr_t* hdr = sam_hdr_read(master_fp);
        if (!hdr) {
            hts_close(master_fp);
            return {};
        }
        hts_idx_t* idx = sam_index_load(master_fp, bam_path.c_str());
        if (!idx) {
            std::cerr << "[parallel] No BAM index found, falling back to single-threaded\n";
            sam_hdr_destroy(hdr);
            hts_close(master_fp);
            return run(bam_path);
        }

        // Map chromosome names and resolve pending references
        build_chrom_map(hdr);
        resolve_references();

        int n_chroms = sam_hdr_nref(hdr);
        uint32_t n_barcodes = static_cast<uint32_t>(barcodes_.size());

        // ── Estimate reads per chromosome for load balancing ──
        std::vector<uint64_t> chrom_reads(n_chroms, 0);
        for (int tid = 0; tid < n_chroms; ++tid) {
            uint64_t mapped, unmapped;
            if (hts_idx_get_stat(idx, tid, &mapped, &unmapped) == 0) {
                chrom_reads[tid] = mapped + unmapped;
            }
        }

        // Sort chromosomes by descending read count for greedy load balance
        std::vector<int> chrom_order(n_chroms);
        std::iota(chrom_order.begin(), chrom_order.end(), 0);
        std::sort(chrom_order.begin(), chrom_order.end(),
            [&](int a, int b) { return chrom_reads[a] > chrom_reads[b]; });

        // Greedy assignment: each chromosome to the least-loaded worker
        std::vector<std::vector<int>> worker_chroms(n_workers);
        std::vector<uint64_t> worker_load(n_workers, 0);
        for (int tid : chrom_order) {
            if (chrom_reads[tid] == 0) continue;
            int best = static_cast<int>(
                std::min_element(worker_load.begin(), worker_load.end()) - worker_load.begin());
            worker_chroms[best].push_back(tid);
            worker_load[best] += chrom_reads[tid];
        }

        // Log assignment
        for (int w = 0; w < n_workers; ++w) {
            if (worker_chroms[w].empty()) continue;
            std::cerr << "[parallel] Worker " << w << ": "
                      << worker_chroms[w].size() << " chroms, ~"
                      << worker_load[w] << " reads\n";
        }

        // ── Strand probe (parallel path) ──
        // Workers start concurrently and cannot probe in-line, so sample the
        // indexed BAM across nuclear chromosomes before spawning them.
        if (config_.stranded && config_.auto_strand && has_gene_model_ && !strand_probed_)
            probe_strand_indexed(master_fp, hdr, idx);

        // reads-per-UMI matrices are filled by finalize_all() on this thread, not per worker
        if (reads_matrix_on()) {
            exon_reads_acc_.set_barcodes(barcodes_);
            exon_reads_acc_.set_n_features(gene_model_.n_exons());
            if (config_.count_introns) {
                intron_reads_acc_.set_barcodes(barcodes_);
                intron_reads_acc_.set_n_features(gene_model_.n_introns());
            }
        }

        // ── Initialize per-worker contexts ──
        std::vector<WorkerContext> workers(n_workers);
        for (int w = 0; w < n_workers; ++w) {
            auto& wc = workers[w];
            wc.snp_ad.set_barcodes(barcodes_);
            wc.snp_ad.set_n_features(n_snps_loaded_);
            wc.snp_dp.set_barcodes(barcodes_);
            wc.snp_dp.set_n_features(n_snps_loaded_);
            if (has_gene_model_) {
                wc.exon.set_barcodes(barcodes_);
                wc.exon.set_n_features(gene_model_.n_exons());
                if (config_.count_introns) {
                    wc.intron.set_barcodes(barcodes_);
                    wc.intron.set_n_features(gene_model_.n_introns());
                    wc.amb.set_barcodes(barcodes_);
                    wc.amb.set_n_features(gene_model_.n_genes());
                }
            }
            if (config_.count_mt) {
                wc.mt.set_barcodes(barcodes_);
                wc.mt.set_n_features(mt::MT_N_FEATURES);
                wc.mt_rev.set_barcodes(barcodes_);
                wc.mt_rev.set_n_features(mt::MT_N_FEATURES);
                wc.mt.set_compact_limit(kMtCompactTriplets);
                wc.mt_rev.set_compact_limit(kMtCompactTriplets);
            }
            // N18: CRISPR guide worker init
            if (!guide_ref_.empty()) {
                wc.guide.set_barcodes(barcodes_);
                wc.guide.set_n_features(static_cast<uint32_t>(guide_ref_.size()));
                wc.guide_umi.reserve(guide_ref_.size() * barcodes_.size() / 10 + 1000);
            }
            // N17: VDJ worker init
            if (has_vdj_) {
                wc.vdj.set_barcodes(barcodes_);
                wc.vdj.set_n_features(vdj_model_.n_genes());
            }
            wc.mm_buffer.reserve(100000);
            wc.cell_reads.assign(n_barcodes, 0);  // N20
            wc.cell_qc.assign(n_barcodes, CellAlignQC{});
            // N22: WL ambient gene profile per worker
            if (!wl_amb_index_.empty() && has_gene_model_)
                wc.wl_ambient_genes.assign(gene_model_.n_genes(), 0ULL);
        }

        // N22: Initialize global WL ambient gene profile for run_parallel()
        if (!wl_amb_index_.empty() && has_gene_model_) {
            wl_ambient_gene_counts_.assign(gene_model_.n_genes(), 0ULL);
        }

        // ── Launch worker threads ──
        auto proc_t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;

        for (int w = 0; w < n_workers; ++w) {
            if (worker_chroms[w].empty()) continue;
            threads.emplace_back([&, w]() {
                auto& wc = workers[w];

                // Each worker opens its own BAM handle for independent I/O
                htsFile* wfp = hts_open(bam_path.c_str(), "r");
                if (!wfp) return;
                hts_set_threads(wfp, 1);  // 1 decompression thread per worker

                sam_hdr_t* whdr = sam_hdr_read(wfp);
                if (!whdr) { hts_close(wfp); return; }

                bam1_t* b = bam_init1();

                for (int tid : worker_chroms[w]) {
                    hts_itr_t* itr = sam_itr_queryi(idx, tid, 0, HTS_POS_MAX);
                    if (!itr) continue;

                    while (sam_itr_next(wfp, itr, b) >= 0) {
                        process_read_parallel(wc, b, tid, whdr);
                    }
                    hts_itr_destroy(itr);
                }

                // Flush remaining mm_buffer entries — only resolve complete ones.
                // Partial entries (seen < nh) are kept for cross-worker merge after join.
                for (auto it = wc.mm_buffer.begin(); it != wc.mm_buffer.end(); ) {
                    if (it->second.nh > 0 && it->second.seen >= it->second.nh) {
                        resolve_mm_parallel(wc, it->second);
                        it = wc.mm_buffer.erase(it);
                    } else {
                        ++it;  // defer to cross-worker merge
                    }
                }
                // wc.mm_buffer retains partial entries for cross-worker merging

                bam_destroy1(b);
                sam_hdr_destroy(whdr);
                hts_close(wfp);
            });
        }

        for (auto& t : threads) t.join();

        auto proc_t1 = std::chrono::high_resolution_clock::now();
        double proc_s = std::chrono::duration<double>(proc_t1 - proc_t0).count();

        // ── Cross-worker multi-mapper merge ──
        // Reads whose primary alignment is on chromosome A and secondary on chromosome B
        // are split across workers and cannot be resolved within a single worker.
        // Partial mm_buffer entries (seen < nh) are merged here across all workers,
        // BEFORE the molecule stores are finalized, so a resolved read joins the
        // store that already holds its (barcode, gene) and its UMI is counted once.
        uint64_t xw_exon = 0, xw_mt = 0;
        {
            std::unordered_map<uint64_t, WorkerContext::MultiHitEntry> global_mm;
            global_mm.reserve(500000);

            for (int w = 0; w < n_workers; ++w) {
                for (auto& [qh, wentry] : workers[w].mm_buffer) {
                    auto& m = global_mm[qh];
                    m.seen += wentry.seen;
                    if (m.nh == 0 && wentry.nh > 0) m.nh = wentry.nh;
                    if (m.bc_idx < 0 && wentry.bc_idx >= 0) m.bc_idx = wentry.bc_idx;
                    if (wentry.mt_gene != UINT32_MAX) {
                        m.mt_gene = wentry.mt_gene; m.mt_exon = wentry.mt_exon; m.mt_intron = wentry.mt_intron;
                        m.mt_flags = wentry.mt_flags; m.mt_diag = wentry.mt_diag;
                    }
                    if (m.umi_len == 0 && wentry.umi_len > 0) {
                        m.umi_len = wentry.umi_len;
                        std::memcpy(m.umi, wentry.umi, wentry.umi_len);
                    }
                    if (wentry.gene_ambiguous) {
                        m.gene_ambiguous = true;
                        // G-EM: propagate known candidates from worker entry
                        if (m.second_gene == UINT32_MAX && wentry.second_gene != UINT32_MAX)
                            m.second_gene = wentry.second_gene;
                    } else {
                        uint32_t eg = (wentry.gene_idx != UINT32_MAX) ? wentry.gene_idx
                                    : (wentry.first_gene != UINT32_MAX) ? wentry.first_gene
                                    : UINT32_MAX;
                        if (eg != UINT32_MAX) {
                            if (m.first_gene == UINT32_MAX) {
                                m.first_gene = eg;
                            }
                            if (eg != m.first_gene) {
                                m.gene_ambiguous = true;
                                // G-EM: save conflicting gene as second candidate
                                if (m.second_gene == UINT32_MAX) m.second_gene = eg;
                                m.gene_idx = UINT32_MAX;
                                m.exon_idx = UINT32_MAX;
                            } else if (wentry.gene_idx != UINT32_MAX && !m.primary_assigned &&
                                       (wentry.primary_assigned || m.gene_idx == UINT32_MAX ||
                                        wentry.exon_idx < m.exon_idx)) {
                                // The primary's assignment wins; otherwise the smallest
                                // secondary exon (order-free-rules R7)
                                m.primary_assigned = wentry.primary_assigned;
                                m.gene_idx = wentry.gene_idx;
                                m.exon_idx = wentry.exon_idx;
                                m.intron_idx = wentry.intron_idx;
                                m.read_flags = wentry.read_flags;
                                m.read_diag = wentry.read_diag;
                            }
                        }
                    }
                }
            }

            PileupStats xw_stats;
            for (auto& [qh, m] : global_mm) {
                // intron_rule::kMtPrefer (see resolve_mm_parallel): chrM gene vs. its nuclear copy
                if (m.gene_ambiguous && m.bc_idx >= 0 && m.mt_gene != UINT32_MAX &&
                    m.first_gene != UINT32_MAX && m.second_gene != UINT32_MAX &&
                    is_mt_gene(m.first_gene) != is_mt_gene(m.second_gene)) {
                    const uint32_t bc = static_cast<uint32_t>(m.bc_idx);
                    int target = 0;
                    for (int w = 0; w < n_workers; ++w)
                        if (workers[w].mol_store.has_group(bc, m.mt_gene)) { target = w; break; }
                    auto& tw = workers[target];
                    ReadFeatures rf{m.mt_gene, m.mt_exon, m.mt_intron, m.mt_flags,
                                    static_cast<uint8_t>(m.mt_diag | DirectionalUmiStore::kDiagMulti)};
                    count_read(tw.mol_store, tw.exon, tw.intron, xw_stats, bc, rf,
                               m.umi_len > 0 ? m.umi : nullptr, m.umi_len);
                    xw_mt++;
                    continue;
                }
                if (m.gene_ambiguous || m.bc_idx < 0) {
                    // G-EM: collect cross-worker ambiguous reads for rescue
                    if (m.gene_ambiguous && m.bc_idx >= 0 &&
                        m.first_gene != UINT32_MAX && m.second_gene != UINT32_MAX) {
                        em_rescue::AmbigRead ar;
                        ar.bc_idx  = m.bc_idx;
                        ar.umi_len = m.umi_len;
                        std::memcpy(ar.umi, m.umi, m.umi_len);
                        ar.gene_a  = m.first_gene;
                        ar.gene_b  = m.second_gene;
                        ambig_reads_.push_back(ar);
                    }
                    continue;
                }
                if (m.first_gene == UINT32_MAX || m.gene_idx == UINT32_MAX) continue;

                const uint32_t bc = static_cast<uint32_t>(m.bc_idx);
                int target = 0;  // no store has this (bc, gene) yet: any store is the only one
                for (int w = 0; w < n_workers; ++w)
                    if (workers[w].mol_store.has_group(bc, m.gene_idx)) { target = w; break; }
                auto& tw = workers[target];
                ReadFeatures rf{m.gene_idx, m.exon_idx, m.intron_idx, m.read_flags,
                                static_cast<uint8_t>(m.read_diag | DirectionalUmiStore::kDiagMulti)};
                count_read(tw.mol_store, tw.exon, tw.intron, xw_stats, bc, rf,
                           m.umi_len > 0 ? m.umi : nullptr, m.umi_len);
                xw_exon++;
            }
            workers[0].stats.exon_hits += xw_stats.exon_hits;
            workers[0].stats.mt_numt_rescued += xw_mt;
            workers[0].stats.intron_hits += xw_stats.intron_hits;
            workers[0].stats.umi_duplicate += xw_stats.umi_duplicate;

            std::cerr << "[parallel-cross-worker] " << global_mm.size()
                      << " cross-worker entries → " << xw_exon << " gene-assigned reads\n";
        }

        // ── UMI collapse + spliced/unspliced/ambiguous labelling (per worker) ──
        // Genes don't span chromosomes and cross-worker reads were routed above,
        // so per-worker finalize is equivalent to a global one.
        std::vector<FinalizeTarget> fin_targets;
        for (int w = 0; w < n_workers; ++w)
            fin_targets.push_back({&workers[w].mol_store, &workers[w].exon, &workers[w].intron, &workers[w].amb});
        DirectionalUmiStore::MoleculeStats mol_stats = finalize_all(fin_targets);
        // the stores are cleared after R9 has used them (below)

        // ── Merge worker outputs ──
        PileupStats stats;
        for (int w = 0; w < n_workers; ++w) {
            auto& ws = workers[w].stats;
            stats.total_reads += ws.total_reads;
            stats.mapped_reads += ws.mapped_reads;
            stats.barcoded_reads += ws.barcoded_reads;
            stats.snp_hits += ws.snp_hits;
            stats.exon_hits += ws.exon_hits;
            stats.intron_hits += ws.intron_hits;
            stats.sj_hits += ws.sj_hits;
            stats.chrm_reads += ws.chrm_reads;
            stats.mt_pileup_bases += ws.mt_pileup_bases;
            stats.low_mapq += ws.low_mapq;
            stats.secondary_reads += ws.secondary_reads;
            stats.no_barcode += ws.no_barcode;
            stats.no_umi += ws.no_umi;
            stats.umi_unique += ws.umi_unique;
            stats.umi_duplicate += ws.umi_duplicate;
            stats.multimapper_reads += ws.multimapper_reads;
            stats.multigene_reads += ws.multigene_reads;
            stats.wrong_strand += ws.wrong_strand;
            stats.strand_sense += ws.strand_sense;
            stats.strand_antisense += ws.strand_antisense;
            stats.vdj_hits += ws.vdj_hits;
            stats.rrna_reads += ws.rrna_reads;
            stats.rrna_reads_sampled += ws.rrna_reads_sampled;
        }
        stats.cross_gene_intron_reads = 0;
        stats.mt_numt_rescued = 0;
        for (int w = 0; w < n_workers; ++w) {
            stats.cross_gene_intron_reads += workers[w].stats.cross_gene_intron_reads;
            stats.mt_numt_rescued += workers[w].stats.mt_numt_rescued;
        }
        add_molecule_stats(stats, mol_stats);

        // G-EM: merge per-worker ambiguous reads into engine's ambig_reads_
        {
            size_t total_ambig = 0;
            for (int w = 0; w < n_workers; ++w) total_ambig += workers[w].ambig_reads.size();
            ambig_reads_.reserve(ambig_reads_.size() + total_ambig);
            for (int w = 0; w < n_workers; ++w) {
                for (auto& ar : workers[w].ambig_reads)
                    ambig_reads_.push_back(ar);
                workers[w].ambig_reads.clear();
            }
        }

        // Merge COO buffers by taking ownership (move semantics)
        // N20: merge per-cell read counts
        per_cell_reads_.assign(n_barcodes, 0);
        for (int w = 0; w < n_workers; ++w) {
            const auto& cr = workers[w].cell_reads;
            for (uint32_t i = 0; i < n_barcodes && i < cr.size(); ++i)
                per_cell_reads_[i] += cr[i];
        }
        {
            size_t np = polya_recs_.size();
            for (int w = 0; w < n_workers; ++w) np += workers[w].polya.size();
            polya_recs_.reserve(np);
            for (int w = 0; w < n_workers; ++w) {
                polya_recs_.insert(polya_recs_.end(), workers[w].polya.begin(), workers[w].polya.end());
                std::vector<PolyARecord>().swap(workers[w].polya);
            }
        }
        per_cell_qc_.assign(n_barcodes, CellAlignQC{});
        for (int w = 0; w < n_workers; ++w) {
            const auto& cq = workers[w].cell_qc;
            for (uint32_t i = 0; i < n_barcodes && i < cq.size(); ++i) per_cell_qc_[i].merge(cq[i]);
        }
        // SNP AD/DP: consensus per molecule in its worker (order-free-rules R1), then merge
        for (auto& wc : workers) {
            wc.snp_cons.finalize(wc.snp_ad, wc.snp_dp, stats, config_.umi_dedup_directional,
                                 gene_roots(wc.mol_store));  // R1 over R9 molecules
            wc.snp_cons.clear();
            // R9: a junction, V(D)J gene or gene lies on one chromosome, so all its reads are in one worker
            finalize_umi_features(wc.sj_umi, wc.sj, stats.sj_hits, wc.mol_store);
            finalize_umi_features(wc.vdj_umi, wc.vdj, stats.vdj_hits, wc.mol_store);
            wc.mol_store.clear_molecules();
        }
        merge_accumulators(snp_ad_, workers, [](WorkerContext& w) -> SparseAccumulator<uint16_t>& { return w.snp_ad; });
        merge_accumulators(snp_dp_, workers, [](WorkerContext& w) -> SparseAccumulator<uint16_t>& { return w.snp_dp; });
        if (config_.count_mt) {
            merge_accumulators(mt_acc_, workers, [](WorkerContext& w) -> SparseAccumulator<uint16_t>& { return w.mt; });
            merge_accumulators(mt_acc_rev_, workers, [](WorkerContext& w) -> SparseAccumulator<uint16_t>& { return w.mt_rev; });
            for (auto& wc : workers) {
                mt_indels_.insert(mt_indels_.end(),
                    std::make_move_iterator(wc.mt_indels.begin()),
                    std::make_move_iterator(wc.mt_indels.end()));
                wc.mt_indels.clear();
            }
        }
        if (has_gene_model_) {
            merge_accumulators(exon_acc_, workers, [](WorkerContext& w) -> SparseAccumulator<uint16_t>& { return w.exon; });
            if (config_.count_introns) {
                merge_accumulators(intron_acc_, workers, [](WorkerContext& w) -> SparseAccumulator<uint16_t>& { return w.intron; });
                merge_accumulators(amb_acc_, workers, [](WorkerContext& w) -> SparseAccumulator<uint16_t>& { return w.amb; });
            }
        }

        // Merge SJ: reassign indices to build global SJ list, then canonical order (order-free-rules R7)
        merge_splice_junctions(workers);
        canonicalize_splice_junctions();

        // N18: Merge CRISPR guide accumulators
        if (!guide_ref_.empty()) {
            guide_ctr_.init(guide_ref_.size(), barcodes_);
            merge_accumulators(guide_ctr_.accumulator(), workers,
                [](WorkerContext& w) -> SparseAccumulator<uint16_t>& { return w.guide; });
        }
        // N17: Merge VDJ accumulators
        if (has_vdj_) {
            merge_accumulators(vdj_acc_, workers,
                [](WorkerContext& w) -> SparseAccumulator<uint16_t>& { return w.vdj; });
        }

        // N22: Merge per-worker WL ambient gene profiles into the global vector
        if (!wl_ambient_gene_counts_.empty()) {
            for (int w = 0; w < n_workers; ++w) {
                const auto& wg = workers[w].wl_ambient_genes;
                for (uint32_t g = 0; g < static_cast<uint32_t>(wl_ambient_gene_counts_.size()); ++g)
                    wl_ambient_gene_counts_[g] += wg[g];
            }
        }

        // ── Deferred chrM BAM write (parallel mode) ──
        // Workers accumulated bam1_t* in wc.chrm_buffer during processing.
        // Write after all workers join to avoid Lustre back-pressure in hot loop.
        if (!config_.out_chrm_bam.empty()) {
            auto tc0 = std::chrono::high_resolution_clock::now();
            htsFile* chrm_fp = hts_open(config_.out_chrm_bam.c_str(), "wb1");
            if (chrm_fp) {
                hts_set_threads(chrm_fp, 2);
                sam_hdr_t* chrm_hdr = sam_hdr_dup(hdr);
                bool hdr_ok = (sam_hdr_write(chrm_fp, chrm_hdr) >= 0);
                if (!hdr_ok)
                    std::cerr << "[singlet-pileup] WARNING: Failed to write chrM BAM header\n";
                if (hdr_ok) {
                    for (int w = 0; w < n_workers; ++w) {
                        for (auto* r : workers[w].chrm_buffer) {
                            if (sam_write1(chrm_fp, chrm_hdr, r) < 0)
                                std::cerr << "[singlet-pileup] WARNING: Failed to write chrM read\n";
                            bam_destroy1(r);
                        }
                        workers[w].chrm_buffer.clear();
                    }
                } else {
                    for (int w = 0; w < n_workers; ++w) {
                        for (auto* r : workers[w].chrm_buffer) bam_destroy1(r);
                        workers[w].chrm_buffer.clear();
                    }
                }
                sam_hdr_destroy(chrm_hdr);
                hts_close(chrm_fp);
            } else {
                std::cerr << "[singlet-pileup] WARNING: Could not open chrM BAM for writing: "
                          << config_.out_chrm_bam << "\n";
                for (int w = 0; w < n_workers; ++w) {
                    for (auto* r : workers[w].chrm_buffer) bam_destroy1(r);
                    workers[w].chrm_buffer.clear();
                }
            }
            auto tc1 = std::chrono::high_resolution_clock::now();
            std::cerr << "[singlet-pileup] chrM BAM write (parallel): "
                      << stats.chrm_reads << " reads in "
                      << std::chrono::duration<double>(tc1 - tc0).count() << "s\n";
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        stats.wall_time_s = std::chrono::duration<double>(t1 - proc_t0).count();

        std::cerr << "[parallel] Processing: " << proc_s << "s ("
                  << n_workers << " workers)\n";
        report_cb_check();   // R11 (sorted path, SINGLET_TEST_CB_CHECK)
        std::cerr << "[singlet-pileup] Done: " << stats.total_reads << " reads, "
                  << stats.mapped_reads << " mapped, "
                  << stats.barcoded_reads << " barcoded, "
                  << stats.secondary_reads << " secondary, "
                  << stats.snp_hits << " SNP hits, "
                  << stats.exon_hits << " exon hits, "
                  << stats.intron_hits << " intron hits, "
                  << stats.sj_hits << " SJ, "
                  << stats.chrm_reads << " chrM, "
                  << stats.mt_pileup_bases << " mt_bases, "
                  << stats.multigene_reads << " multigene, "
                  << stats.wrong_strand << " wrong_strand, "
                  << stats.vdj_hits << " vdj_hits, "
                  << "UMI " << stats.umi_unique << " unique / "
                  << stats.umi_duplicate << " dup in "
                  << stats.wall_time_s << "s\n";

        hts_idx_destroy(idx);
        sam_hdr_destroy(hdr);
        hts_close(master_fp);

        return stats;
    }

    // W11 GPU pileup harness: with SINGLET_PILEUP_DUMP=<dir>, writes the accumulators the callers receive (truth.bin), the annotation
    // arrays (anno.bin), parameters, barcodes and contig names, in the format of PR #61 bench/gpu/pileup. Unset: nothing happens.
    void gpudump_write(uint32_t n_tid) {
        if (!gpudump::dir()) return;
        gpudump::write_all(gene_model_.exon_trees_view(), gene_model_.intron_trees_view(), gene_model_.exon_to_gene_view(),
                           gene_model_.intron_to_gene_view(), gene_model_.gene_strands_view(), chrom_snps_, n_tid,
                           exon_acc_, intron_acc_, amb_acc_, sj_acc_, sj_names_, snp_ad_, snp_dp_, mt_acc_, mt_acc_rev_, mt_indels_,
                           ambig_reads_.size(), barcodes_, tid_to_chrom_,
                           gpudump::Params{(int)config_.stranded, (int)config_.reverse_strand, config_.min_mapq, config_.min_baseq,
                                           (int)config_.count_introns, (int)config_.count_sj, (int)config_.count_mt, (int)config_.umi_dedup,
                                           (int)config_.umi_dedup_directional, config_.umi_len, chrm_tid_, n_snps_loaded_,
                                           (int)has_gene_model_, gene_model_.n_exons(), gene_model_.n_introns(), gene_model_.n_genes()});
    }

#ifdef SINGLET_HAVE_ENGINE_GPU
    // W11: runs the GPU pileup on the records collected from this pass and compares each covered output with this engine's accumulators
    // (exon, intron, ambiguous, junctions by contig/donor/acceptor, SNP AD/DP, chrM total/reverse, chrM indels, R1 molecule counters).
    std::unique_ptr<gpupileup::Collector> gpu_col_;
    PileupStats* gpu_stats_ = nullptr;
    // this pass's records plus this engine's annotation and parameters, as the GPU pileup takes them; false if not reproducible
    bool gpu_build_input(singlet_engine_gpu::PileupInput& in) {
        using namespace singlet_engine_gpu;
        const auto& et = gene_model_.exon_trees_view(); const auto& it = gene_model_.intron_trees_view();
        const uint32_t n_tid = static_cast<uint32_t>(tid_to_chrom_.size());
        in.exonTrees.resize(n_tid); in.intronTrees.resize(n_tid);
        for (uint32_t t = 0; t < n_tid; ++t) {
            if (t < et.size()) for (const auto& v : et[t].sorted_intervals()) in.exonTrees[t].push_back({v.start, v.end, v.index});
            if (t < it.size()) for (const auto& v : it[t].sorted_intervals()) in.intronTrees[t].push_back({v.start, v.end, v.index});
        }
        in.exonToGene = gene_model_.exon_to_gene_view(); in.intronToGene = gene_model_.intron_to_gene_view(); in.geneStrand = gene_model_.gene_strands_view();
        in.snps.resize(chrom_snps_.size());
        for (size_t t = 0; t < chrom_snps_.size(); ++t)
            for (const auto& x : chrom_snps_[t]) in.snps[t].push_back({x.pos, x.index, x.ref_allele, x.alt_allele, 0});
        in.stranded = config_.stranded; in.reverseStrand = config_.reverse_strand; in.minMapq = config_.min_mapq; in.minBaseq = config_.min_baseq;
        in.countIntrons = config_.count_introns; in.countSj = config_.count_sj; in.countMt = config_.count_mt; in.umiDedup = config_.umi_dedup;
        in.directional = config_.umi_dedup_directional; in.umiLen = config_.umi_len; in.chrmTid = chrm_tid_; in.hasGeneModel = has_gene_model_;
        in.intronRules = config_.intron_rules;
        in.geneIsMt.assign(gene_model_.n_genes(), 0);
        for (uint32_t g = 0; g < gene_model_.n_genes(); ++g) in.geneIsMt[g] = is_mt_gene(g) ? 1 : 0;
        in.wantGeneMerges = (has_vdj_ && config_.umi_dedup) ? 1 : 0;   // R9 per gene for the host's V(D)J reducer
        in.wantReads = reads_matrix_on() ? 1 : 0;   // reads per molecule (exon_reads / intron_reads / histogram)
        in.nSnps = n_snps_loaded_; in.nBarcodes = static_cast<uint32_t>(barcodes_.size()); in.nTid = n_tid;
        in.nExons = gene_model_.n_exons(); in.nIntrons = gene_model_.n_introns(); in.nGenes = gene_model_.n_genes();
        const size_t nreads = gpu_col_->n_reads(), nsigs = gpu_col_->n_sigs(); const uint64_t inc = gpu_col_->incomplete();
        if (gpu_col_->handoff_used())
            std::cerr << "[gpu_pileup] hand-off: " << gpu_col_->n_handoff_reads() << " reads take their alignments from the GPU prefill, "
                      << gpu_col_->handoff_mismatches() << " primary alignments differ from the stream's\n";
        gpu_col_->finish(in);
        gpu_clip_ = gpu_col_->clips();   // gpu_host_pass rebuilds the poly(A) site of a clipped read from these, after gpu_col_ is gone
        gpu_col_.reset();
        if (in.snpSkipN < 0) { std::cerr << "[gpu_pileup] not run: base qualities neither constant nor tied to N\n"; return false; }
        std::cerr << "[gpu_pileup] " << nreads << " reads, " << nsigs << " signatures (" << inc << " incomplete groups)\n";
        return true;
    }
    void gpu_pileup_compare() {
        if (!gpu_col_) return;
        { PileupStats dummy; if (gpu_col_->handoff()) gpu_handoff_pass(dummy, true); }
        using namespace singlet_engine_gpu;
        PileupInput in;
        if (!gpu_build_input(in)) return;
        const uint32_t n_tid = in.nTid;
        int lock_fd = -1;
        const int device = lockDevice(&lock_fd);
        PileupOutput out;
        const bool ok = pileupRun(in, device, out);
        unlockDevice(lock_fd);
        if (!ok) { std::cerr << "[gpu_pileup] failed: " << out.log << "\n"; return; }
        std::cerr << "[gpu_pileup] GPU " << out.msGpu
                  << " ms, total " << out.msTotal << " ms, peak " << out.peakGB << " GB; R1 molecules " << out.snpMolecules << ", ambiguous "
                  << out.snpAmbiguous << ", other " << out.snpOther << "\n";
        bool all = true;
        std::cerr << "[gpu_pileup] " << gpupileup::cmp_line("exon", out.exon, gpupileup::cpu_triplets(exon_acc_), all) << "\n";
        std::cerr << "[gpu_pileup] " << gpupileup::cmp_line("intron", out.intron, gpupileup::cpu_triplets(intron_acc_), all) << "\n";
        std::cerr << "[gpu_pileup] " << gpupileup::cmp_line("ambiguous", out.amb, gpupileup::cpu_triplets(amb_acc_), all) << "\n";
        std::cerr << "[gpu_pileup] " << gpupileup::cmp_line("snp_ad", out.snpAd, gpupileup::cpu_triplets(snp_ad_), all) << "\n";
        std::cerr << "[gpu_pileup] " << gpupileup::cmp_line("snp_dp", out.snpDp, gpupileup::cpu_triplets(snp_dp_), all) << "\n";
        std::cerr << "[gpu_pileup] " << gpupileup::cmp_line("mt", out.mt, gpupileup::cpu_triplets(mt_acc_), all) << "\n";
        std::cerr << "[gpu_pileup] " << gpupileup::cmp_line("mt_rev", out.mtRev, gpupileup::cpu_triplets(mt_acc_rev_), all) << "\n";
        {   // junctions by (contig, donor, acceptor): the CPU's names are "contig:donor-acceptor"
            std::map<std::string, uint32_t> tidof;
            for (uint32_t t = 0; t < n_tid; ++t) tidof[tid_to_chrom_[t]] = t;
            std::vector<std::array<uint32_t, 3>> cpukey(sj_names_.size());
            for (size_t i = 0; i < sj_names_.size(); ++i) {
                const std::string& nm = sj_names_[i]; const size_t c = nm.rfind(':'), d = nm.find('-', c);
                if (c == std::string::npos || d == std::string::npos || !tidof.count(nm.substr(0, c))) { cpukey[i] = {~0u, 0, 0}; continue; }
                cpukey[i] = {tidof[nm.substr(0, c)], (uint32_t) std::atoll(nm.substr(c + 1, d - c - 1).c_str()), (uint32_t) std::atoll(nm.c_str() + d + 1)};
            }
            using K = std::array<uint32_t, 5>;   // cell, tid, donor, acceptor, value
            std::vector<K> g, c;
            for (const auto& x : out.sj) { const auto& k = out.sjtab[x.row]; g.push_back({x.cell, k[0], k[1], k[2], x.val}); }
            for (const auto& x : gpupileup::cpu_triplets(sj_acc_)) { const auto& k = cpukey[x.row]; c.push_back({x.cell, k[0], k[1], k[2], x.val}); }
            std::sort(g.begin(), g.end()); std::sort(c.begin(), c.end());
            const bool ok_sj = g == c; all = all && ok_sj;
            std::cerr << "[gpu_pileup] junctions " << (ok_sj ? "identical" : "DIFFERENT") << " (gpu " << g.size() << ", cpu " << c.size() << " entries)\n";
        }
        {
            std::vector<std::array<uint32_t, 4>> c;
            for (const auto& e : mt_indels_) c.push_back({e.bc_idx, e.pos, (uint32_t) e.len, e.is_ins ? 1u : 0u});
            auto g = out.indels; std::sort(g.begin(), g.end()); std::sort(c.begin(), c.end());
            const bool ok_in = g == c; all = all && ok_in;
            std::cerr << "[gpu_pileup] mt_indels " << (ok_in ? "identical" : "DIFFERENT") << " (gpu " << g.size() << ", cpu " << c.size() << ")\n";
        }
        std::cerr << "[gpu_pileup] RESULT " << (all ? "IDENTICAL" : "DIFFERENT") << "\n";
    }

    // W11 "on": runs the GPU pileup on this pass's records and puts its outputs where the skipped CPU steps would have: the exon,
    // intron and ambiguous accumulators (molecules), junctions (rows in (contig, donor, acceptor) order, the R7 order), SNP AD/DP (R1),
    // chrM allele counts and indels, and the counters those steps keep. The molecule label counts are applied after the CPU finalize.
    DirectionalUmiStore::MoleculeStats gpu_ms_{};
    std::vector<uint32_t> gpu_clip_;   // per read of the GPU pileup's input: its hand-off clip flags (0: not clipped)
    template <typename V>
    static void gpu_put(SparseAccumulator<V>& acc, const std::vector<singlet_engine_gpu::PileupTriplet>& v) {
        for (const auto& t : v) acc.increment(t.row, t.cell, static_cast<V>(t.val));
    }
    // W11 "on": what the serial loop derives from each barcoded primary's classification, from the GPU's per-alignment results: the
    // strand votes (non-mito contigs), wrong_strand, multigene, cross_gene_intron counters, poly(A) records (NH 1, gene assigned,
    // forward-oriented library) and VDJ usage. Same conditions as the loop: primary, barcode in the set, past the MAPQ gate.
    bool gpu_host_pass(const singlet_engine_gpu::PileupInput& in, const singlet_engine_gpu::PileupOutput& out, PileupStats& stats) {
        const SigRec* S = reinterpret_cast<const SigRec*>(in.sigs.data());
        const AlnRec* A = reinterpret_cast<const AlnRec*>(in.alns.data());
        const ReadRec* R = reinterpret_cast<const ReadRec*>(in.reads.data());
        const size_t nR = in.reads.size() / sizeof(ReadRec);
        if (out.alnStat.size() != in.alns.size() / sizeof(AlnRec)) { std::cerr << "[gpu_pileup] per-alignment results missing\n"; return false; }
        uint64_t pd[16] = {0}; const bool pdbg = std::getenv("SINGLET_POLYA_DEBUG") != nullptr;   // diagnostics: why the poly(A) rebuild keeps or drops a read
        static const char UMI_CH[5] = {'A', 'C', 'G', 'T', 'N'};
        static const uint8_t COMP[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
        const int ulen = in.umiLen > 0 ? in.umiLen : 0;
        struct HostLocal {   // one chunk of reads: its counters and, in read order, its poly(A) records and V(D)J hits
            PileupStats st; uint64_t pd[16] = {0}; std::vector<PolyARecord> polya; std::vector<uint32_t> ov;
            struct VdjEv { uint32_t bc, vg, gene; int ulen; char umi[24]; };
            std::vector<VdjEv> vdj;
        };
        // the chunks run on the job's cores (the streaming pileup is one thread; STAR has exited) and are merged in read order, so every
        // counter and the order of the poly(A) and V(D)J records are those of the serial loop (SINGLET_HANDOFF_PASS_SERIAL=1: one chunk)
        cpu_set_t cs; int nT = 1;
        if (std::getenv("SINGLET_HANDOFF_PASS_SERIAL") == nullptr && sched_getaffinity(0, sizeof(cs), &cs) == 0) nT = std::max(1, std::min(CPU_COUNT(&cs), 32));
        if (const char* e = std::getenv("SINGLET_HANDOFF_PASS_THREADS")) nT = std::max(1, std::atoi(e));
        if (nR < (1u << 16)) nT = 1;
        std::vector<HostLocal> hls(nT);
        auto runChunk = [&](int t) {
            HostLocal& L = hls[t];
            const size_t lo = nR * t / nT, hi = nR * (t + 1) / nT;
            char umi[32];
            for (size_t r = lo; r < hi; ++r) {
            const ReadRec& rr = R[r];
            const SigRec& sg = S[rr.sig];
            int p = -1;
            for (int k = 0; k < sg.n_aln; ++k) if (A[sg.aln_off + k].flags & 2) { p = k; break; }
            if (p < 0) continue;
            const AlnRec& a = A[sg.aln_off + p];
            const int nh = sg.nh;
            if (a.mapq < static_cast<uint8_t>(config_.min_mapq) && nh == 1) continue;   // the loop skips unique low-MAPQ reads
            const uint8_t st = out.alnStat[sg.aln_off + p];
            const uint32_t gene = out.alnGene[sg.aln_off + p];
            const bool vs = st & 1, va = st & 2;
            if (vs != va && a.tid >= 0 && static_cast<size_t>(a.tid) < mito_tid_.size() && !mito_tid_[a.tid])
                (vs ? L.st.strand_sense : L.st.strand_antisense)++;
            if (st & 4) L.st.wrong_strand++;
            if (st & 16) L.st.multigene_reads++;
            if (st & 8) L.st.cross_gene_intron_reads++;
            for (int i = 0; i < ulen && i < 21; ++i) umi[i] = UMI_CH[std::min<uint64_t>((rr.umi3 >> (3 * i)) & 7, 4)];
            umi[std::min(ulen, 21)] = 0;
            const char* umi_str = ulen > 0 ? umi : nullptr;
            const int umi_len = ulen > 0 ? std::min(ulen, 21) : 0;
            const uint32_t* ops = in.ops.data() + a.ops_off;
            const bool rev = (a.flags & 1) != 0;
            // poly(A): polya_site_from_read on the BAM-orientation sequence (the collector stores the read's own orientation)
            if (nh == 1 && config_.count_polya && !config_.reverse_strand && gene != UINT32_MAX && a.n_ops >= 1) {   // BAM CIGAR of >= 2 ops (a clipped read adds soft clips)
                const int qlen = sg.rlen;
                auto base = [&](int j) -> uint8_t {   // BAM nt16 code of query position j
                    const int i = rev ? qlen - 1 - j : j;
                    const uint8_t v = (in.seq[sg.seq_off + (i >> 1)] >> ((i & 1) * 4)) & 15;
                    return rev ? COMP[v] : v;
                };
                uint64_t key = 0; bool ok = false;
                const uint32_t clip = r < gpu_clip_.size() ? gpu_clip_[r] : 0u;
                L.pd[0]++; if (clip) L.pd[1]++;
                if (clip) {
                    // a clipped read: the BAM sequence is the whole read and its CIGAR has the clipped bases as soft clips, so the 3' soft
                    // clip (in read orientation; the BAM's last op when forward, its first when reverse) is the record's own plus the tail
                    const int c5 = (int) ((clip >> 8) & 255), c3 = (int) ((clip >> 16) & 255), nAt = (int) (clip >> 24);
                    const int ql = qlen + c5 + c3;
                    const uint32_t o3 = rev ? ops[0] : ops[a.n_ops - 1], o5 = rev ? ops[a.n_ops - 1] : ops[0];
                    const int s3 = (o3 & 15) == BAM_CSOFT_CLIP ? (int) (o3 >> 4) : 0;
                    const int nb = a.n_ops + ((c3 > 0 && s3 == 0) ? 1 : 0) + ((c5 > 0 && (o5 & 15) != BAM_CSOFT_CLIP) ? 1 : 0);
                    const int cl = s3 + c3;
                    auto rbase = [&](int j) -> uint8_t { return (in.seq[sg.seq_off + (j >> 1)] >> ((j & 1) * 4)) & 15; };   // record, read orientation
                    if (nb < 2) L.pd[2]++; else if (cl < 6 || cl > ql) L.pd[3]++;
                    if (nb >= 2 && cl >= 6 && cl <= ql) {
                        int na = nAt; for (int j = qlen - s3; j < qlen; ++j) na += rbase(j) == 1;
                        const bool firstA = s3 > 0 ? rbase(qlen - s3) == 1 : (clip & 2u) != 0;
                        if (!(na >= 0.8 * cl)) L.pd[4]++; else if (!firstA) L.pd[5]++;
                        if (na >= 0.8 * cl && firstA) {
                            if (!rev) {
                                int64_t end = a.pos;
                                for (int k = 0; k < a.n_ops; ++k) { const int op = ops[k] & 15; if (op == 0 || op == 2 || op == 3 || op == 7 || op == 8) end += ops[k] >> 4; }
                                if (end > 0) { key = polya_site_key(a.tid, false, static_cast<uint32_t>(end - 1)); ok = true; }
                            } else { key = polya_site_key(a.tid, true, static_cast<uint32_t>(a.pos)); ok = true; }
                        }
                    }
                } else if (a.n_ops >= 2 && !rev) {
                    const uint32_t c = ops[a.n_ops - 1];
                    const int cl = static_cast<int>(c >> 4);
                    if (!clip && (c & 15) == BAM_CSOFT_CLIP) { L.pd[9]++; if (cl >= 6) L.pd[10]++; if (cl > qlen) L.pd[13]++; }
                    if ((c & 15) == BAM_CSOFT_CLIP && cl >= 6 && cl <= qlen) {
                        int na = 0; for (int i = qlen - cl; i < qlen; ++i) na += base(i) == 1;
                        if (!(na >= 0.8 * cl)) L.pd[11]++; else if (base(qlen - cl) != 1) L.pd[12]++;
                        if (na >= 0.8 * cl && base(qlen - cl) == 1) {
                            int64_t end = a.pos;
                            for (int k = 0; k < a.n_ops; ++k) { const int op = ops[k] & 15; if (op == 0 || op == 2 || op == 3 || op == 7 || op == 8) end += ops[k] >> 4; }
                            if (end > 0) { key = polya_site_key(a.tid, false, static_cast<uint32_t>(end - 1)); ok = true; }
                        }
                    }
                } else if (a.n_ops >= 2) {
                    const uint32_t c = ops[0];
                    const int cl = static_cast<int>(c >> 4);
                    if (!clip && (c & 15) == BAM_CSOFT_CLIP) { L.pd[9]++; L.pd[14]++; if (cl >= 6) L.pd[10]++; if (cl > qlen) L.pd[13]++; }
                    if ((c & 15) == BAM_CSOFT_CLIP && cl >= 6 && cl <= qlen) {
                        int nt = 0; for (int i = 0; i < cl; ++i) nt += base(i) == 8;
                        if (!(nt >= 0.8 * cl)) L.pd[11]++; else if (base(cl - 1) != 8) L.pd[12]++;
                        if (nt >= 0.8 * cl && base(cl - 1) == 8) { key = polya_site_key(a.tid, true, static_cast<uint32_t>(a.pos)); ok = true; }
                    }
                }
                if (ok) { L.polya.push_back({key, gene, static_cast<uint32_t>(rr.bc), polya_umi_hash(umi_str, umi_len)}); L.pd[clip ? 6 : 7]++; }
                else if (!clip) L.pd[8]++;
            }
            // VDJ usage: the first VDJ gene overlapping [first aligned block start, last aligned block end)
            if (has_vdj_) {
                int32_t rpos = a.pos, s0 = -1, e0 = -1;
                for (int k = 0; k < a.n_ops; ++k) {
                    const int op = ops[k] & 15; const int32_t len = static_cast<int32_t>(ops[k] >> 4);
                    if (op == 0 || op == 7 || op == 8) { if (s0 < 0) s0 = rpos; rpos += len; e0 = rpos; }
                    else if (op == 2 || op == 3) rpos += len;
                }
                if (s0 >= 0) {
                    L.ov.clear();
                    vdj_model_.query(a.tid, s0, e0, L.ov);
                    if (!L.ov.empty()) {   // replayed in read order after the chunks (vdj_umi_ and vdj_acc_ are not thread-safe)
                        HostLocal::VdjEv ev; ev.bc = static_cast<uint32_t>(rr.bc); ev.vg = L.ov[0]; ev.gene = nh == 1 ? gene : UINT32_MAX; ev.ulen = umi_len;
                        std::memcpy(ev.umi, umi, sizeof(ev.umi));
                        L.vdj.push_back(ev);
                    }
                }
            }
            }
        };
        if (nT == 1) runChunk(0);
        else { std::vector<std::thread> th; for (int t = 0; t < nT; ++t) th.emplace_back(runChunk, t); for (auto& x : th) x.join(); }
        for (int t = 0; t < nT; ++t) {
            HostLocal& L = hls[t];
            stats.strand_sense += L.st.strand_sense; stats.strand_antisense += L.st.strand_antisense; stats.wrong_strand += L.st.wrong_strand;
            stats.multigene_reads += L.st.multigene_reads; stats.cross_gene_intron_reads += L.st.cross_gene_intron_reads;
            for (int k = 0; k < 16; k++) pd[k] += L.pd[k];
            polya_recs_.insert(polya_recs_.end(), L.polya.begin(), L.polya.end());
            for (const auto& ev : L.vdj) {
                const char* us = ev.ulen > 0 ? ev.umi : nullptr;
                if (config_.umi_dedup) vdj_umi_.add(ev.bc, ev.vg, us, ev.ulen, ev.gene);   // R9: counted at finalize, per (cell, gene)
                else { vdj_acc_.increment(ev.vg, ev.bc); stats.vdj_hits++; }
            }
        }
        if (pdbg) std::cerr << "[polya-debug] eligible " << pd[0] << ", clipped " << pd[1] << ": nb<2 " << pd[2] << ", cl out of range " << pd[3]
                            << ", A-fraction " << pd[4] << ", not firstA " << pd[5] << ", kept clipped " << pd[6] << "; unclipped kept " << pd[7] << ", unclipped dropped " << pd[8] << ", unclipped fwd/rev with S " << pd[9] << ", S>=6 " << pd[10] << ", A-fraction fail " << pd[11] << ", first base fail " << pd[12] << ", cl>qlen " << pd[13] << ", rev " << pd[14] << "\n";
        return true;
    }
    // W11 "on": the reads STAR wrote no BAM record for (hand-off records; ReadAlign_outputAlignments.cpp): what the serial loop does
    // with their records before classification, from the hand-off alignments and the decoded R1 / R2 reads (read r<i> = record i of
    // the files): record counters, the R4 rRNA sample, chrM reads, barcode (CB from CR), per-cell reads and alignment QC, the MAPQ gate,
    // and the WL-ambient barcodes (genes after the GPU). Returns false if the decoded reads cannot be read.
    std::vector<std::pair<uint32_t, uint32_t>> gpu_wl_pending_;   // (record, WL index) of WL-ambient reads, genes from the GPU
    /// STAR's g_handoffReadHook (src/star/SeqCache.h), called in STAR's mapping threads for each clipped hand-off read: whether the
    /// read is in the R4 rRNA sample (bit 2) and the detector's verdict on the read as the BAM would store it, forward (bit 3) and
    /// reverse-complemented (bit 4). The pass picks the orientation of the read's primary alignment.
    static unsigned int handoff_read_hook(unsigned long long i, const char* seq, unsigned long long len) {
        if (len < 21 || (hash_qname(("r" + std::to_string(i)).c_str()) & 63) != 0) return 0;
        static const RrnaDetector det;
        std::string f(len, 'N'), r(len, 'N');
        for (unsigned long long j = 0; j < len; j++) f[j] = seq_nt16_str[seq_nt16_table[(unsigned char) seq[j]]];
        for (unsigned long long j = 0; j < len; j++) {
            const char c = f[len - 1 - j];
            r[j] = c == 'A' ? 'T' : c == 'C' ? 'G' : c == 'G' ? 'C' : c == 'T' ? 'A' : 'N';
        }
        return 4u | (det.detect(f) ? 8u : 0u) | (det.detect(r) ? 16u : 0u);
    }
    // gpu_handoff_pass for STAR's per-read entries, on all cores. The read range is cut into contiguous chunks; each chunk runs the serial
    // loop's body with its own counters and outputs, and the chunks are merged in read order, so the collector gets the same sequence
    // of hand-off reads and every counter the same total as the serial pass (SINGLET_HANDOFF_PASS_SERIAL=1 runs that one).
    void gpu_pileup_pending_append(const std::vector<std::pair<uint32_t, uint32_t>>& v) { gpu_wl_pending_.insert(gpu_wl_pending_.end(), v.begin(), v.end()); }
    bool gpu_handoff_pass_entries(PileupStats& stats, const gpupileup::Handoff* H, const gpupileup::ReadEntry* E) {
        struct Out { uint32_t rec; int32_t bc; uint64_t u3; uint32_t clip; gpupileup::QM qm; bool hasq; };
        struct Ctx {
            PileupStats st; uint64_t n_done = 0, n_unwritten = 0, n_bigclip = 0, n_nohook = 0, lowq = 0, neqn = 0;
            uint64_t cb_status[8] = {}; char cb_buf[40] = {};
            std::vector<uint32_t> cell_reads, wl_counts; std::vector<CellAlignQC> cell_qc;
            std::vector<std::pair<uint32_t, uint32_t>> wl_pending; std::vector<Out> out;
        };
        auto unpack = [](uint64_t v, uint32_t L, std::string& s) {
            static const char kB[8] = {'A', 'C', 'G', 'T', 'N', 'N', 'N', 'N'};
            s.resize(L); for (uint32_t j = 0; j < L; j++) s[j] = kB[(v >> (3 * j)) & 7];
        };
        char comp[256]; for (int i = 0; i < 256; i++) comp[i] = 'N';
        comp[(int)'A'] = 'T'; comp[(int)'C'] = 'G'; comp[(int)'G'] = 'C'; comp[(int)'T'] = 'A';
        const CbCorrector* const cbc = config_.cb_corrector;   // R11: as stream_cb, the CR is corrected against the whitelist
        // the streaming pileup is one thread (config_.threads); STAR has exited, so the pass may use the job's cores
        cpu_set_t cs; int nT = std::max(1, config_.threads);
        if (sched_getaffinity(0, sizeof(cs), &cs) == 0) nT = std::max(nT, std::min(CPU_COUNT(&cs), 32));
        if (const char* e = std::getenv("SINGLET_HANDOFF_PASS_THREADS")) nT = std::max(1, std::atoi(e));
        const uint64_t N = H->n_reads;
        if (N < (1u << 16)) nT = 1;
        const gpupileup::QM* const QMs = gpu_col_->handoff_qmask();   // STAR's per-read low-quality masks (null if STAR wrote none)
        std::vector<Ctx> ctxs(nT);
        auto work = [&](int t) {
            Ctx& c = ctxs[t];
            const uint64_t lo = N * t / nT, hi = N * (t + 1) / nT;
            c.cell_reads.assign(per_cell_reads_.size(), 0); c.cell_qc.assign(per_cell_qc_.size(), CellAlignQC{});
            c.wl_counts.assign(wl_umi_counts_.size(), 0);
            PileupStats& st = c.st;
            std::string cr, ur, rseq;
            for (uint64_t i = lo; i < hi; ++i) {
                if (i + 16 < hi) { const uint32_t r16 = H->read_rec[i + 16]; if (r16 != 0xFFFFFFFFu) __builtin_prefetch(H->rec_off + r16); }
                if (i + 8 < hi) { const uint32_t r8 = H->read_rec[i + 8]; if (r8 != 0xFFFFFFFFu) __builtin_prefetch(H->data + H->rec_off[r8]); }
                const uint32_t rec = H->read_rec[i];
                if (rec == 0xFFFFFFFFu) continue;   // read with N (or not mapped on the GPU): it came through the stream
                const uint8_t* p = H->data + H->rec_off[rec];
                const int n = *p++;
                if (n == 0) continue;               // unmapped: STAR writes no record (outSAMunmapped None)
                int32_t ptid = -1; uint16_t pflag = 0, pnh = 1, pnm = 0; uint8_t pmapq = 0; uint64_t pmlen = 0; int pk = -1, nsec = 0;
                for (int k = 0; k < n; k++) {
                    int32_t tid, pos; uint16_t flag, nh, nm, nc; uint8_t mapq;
                    std::memcpy(&tid, p, 4); std::memcpy(&pos, p + 4, 4); std::memcpy(&flag, p + 8, 2); mapq = p[10];
                    std::memcpy(&nh, p + 12, 2); std::memcpy(&nm, p + 14, 2); std::memcpy(&nc, p + 16, 2);
                    const uint8_t* cig = p + 18; p = cig + 4 * nc;
                    if (flag & BAM_FSECONDARY) { nsec++; continue; }
                    if (pk < 0) {
                        pk = k; ptid = tid; pflag = flag; pnh = nh; pnm = nm; pmapq = mapq;
                        for (int cc = 0; cc < nc; cc++) { uint32_t v; std::memcpy(&v, cig + 4 * cc, 4); const int op = v & 15; if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) pmlen += v >> 4; }
                    }
                }
                st.total_reads += n; st.mapped_reads += n; st.secondary_reads += nsec;
                if (pk < 0) continue;
                ++c.n_done;
                if (!(E[i].flags & 1)) { ++c.n_unwritten; continue; }
                uint16_t rl = 0; std::memcpy(&rl, p, 2); const uint8_t* rs = p + 2;
                const uint32_t clip = (E[i].flags & 0xFFFF00u) ? E[i].flags : 0u;
                if (clip & 32u) { ++c.n_bigclip; continue; }
                const int L2 = (int) rl + (int) ((clip >> 8) & 255) + (int) ((clip >> 16) & 255);
                bool rrna_sampled = false;
                if (L2 >= 21) {
                    char qb[24]; char* q = qb + 23; *q = 0;
                    uint64_t v = i; do { *--q = (char) ('0' + v % 10); v /= 10; } while (v);
                    *--q = 'r';
                    rrna_sampled = (hash_qname(q) & 63) == 0;
                }
                if (rrna_sampled) {
                    ++st.rrna_reads_sampled;
                    if (clip) {
                        if (!(clip & 4u)) { ++c.n_nohook; continue; }
                        if (clip & ((pflag & BAM_FREVERSE) ? 16u : 8u)) ++st.rrna_reads;
                    } else { rseq.resize(L2); for (int j = 0; j < L2; j++) rseq[j] = seq_nt16_str[(rs[j >> 1] >> ((j & 1) * 4)) & 15]; }
                    if (!clip) {
                        if (pflag & BAM_FREVERSE) { const std::string fw = rseq; for (int j = 0; j < L2; j++) rseq[j] = comp[(uint8_t) fw[L2 - 1 - j]]; }
                        if (rrna_detector_.detect(rseq)) ++st.rrna_reads;
                    }
                }
                if (ptid == chrm_tid_) st.chrm_reads++;
                unpack(E[i].cr3, H->cbL, cr); unpack(E[i].ur3, H->umiL, ur);
                bool cbok = !cr.empty() && !ur.empty() && ur[0] != ' ';
                if (!cbc) for (char ch : cr) if (ch == 'N') cbok = false;
                bool homo = true;
                for (char ch : ur) { if (ch == 'N') cbok = false; if (ch != ur[0]) homo = false; }
                if (homo) cbok = false;
                if (cbc) {
                    if (!cbok) ++c.cb_status[1];
                    else if (static_cast<int>(cr.size()) != cbc->length()) { ++c.cb_status[3]; cbok = false; }
                    else {
                        const int cst = cbc->resolve(cr.c_str(), nullptr, c.cb_buf);
                        ++c.cb_status[cst + 4];
                        if (cst < 0) cbok = false;
                        else cr.assign(c.cb_buf, static_cast<size_t>(cbc->length()));
                    }
                }
                if (!cbok || cr[0] == '-') { st.no_barcode++; continue; }
                const int32_t bc = bc_index_.lookup(cr.c_str());
                if (bc < 0) {
                    if (!wl_umi_counts_.empty() && pmapq >= static_cast<uint8_t>(config_.min_mapq)) {
                        const int32_t wl = wl_amb_index_.lookup(cr.c_str());
                        if (wl >= 0) {
                            c.wl_counts[static_cast<size_t>(wl)]++;
                            if (has_gene_model_ && !wl_ambient_gene_counts_.empty()) c.wl_pending.push_back({rec, static_cast<uint32_t>(pk)});
                        }
                    }
                    continue;
                }
                c.lowq += E[i].low_qual; c.neqn += E[i].qual_neq_n;
                uint64_t u3 = 0;
                for (int j = 0; j < (int) ur.size() && j < 21; j++) { const char ch = ur[j]; const uint64_t v = ch == 'A' ? 0 : ch == 'C' ? 1 : ch == 'G' ? 2 : ch == 'T' ? 3 : 4; u3 |= v << (3 * j); }
                c.out.push_back({rec, bc, u3, clip, QMs != nullptr ? QMs[i] : gpupileup::QM{{0, 0}}, QMs != nullptr && !(E[i].flags & 64u)});
                st.barcoded_reads++;
                if (static_cast<uint32_t>(bc) < c.cell_reads.size()) c.cell_reads[static_cast<uint32_t>(bc)]++;
                const int nh = std::max<int>(pnh, 1);
                if (nh > 1) st.multimapper_reads++;
                if (static_cast<uint32_t>(bc) < c.cell_qc.size()) {
                    CellAlignQC& q = c.cell_qc[static_cast<uint32_t>(bc)];
                    if (nh > 1) ++q.multimapped; else { ++q.unique; q.mismatches += pnm; q.aligned_bases += pmlen; }
                }
                const bool high = pmapq >= static_cast<uint8_t>(config_.min_mapq);
                if (!high) st.low_mapq++;
                if (!high && nh == 1) continue;
                if (config_.umi_dedup && ur.empty()) st.no_umi++;
            }
        };
        if (nT == 1) work(0);
        else { std::vector<std::thread> th; for (int t = 0; t < nT; ++t) th.emplace_back(work, t); for (auto& x : th) x.join(); }
        uint64_t n_done = 0, n_unwritten = 0, n_bigclip = 0, n_nohook = 0;
        for (int t = 0; t < nT; ++t) {   // in read order
            Ctx& c = ctxs[t]; const PileupStats& s = c.st;
            stats.total_reads += s.total_reads; stats.mapped_reads += s.mapped_reads; stats.secondary_reads += s.secondary_reads;
            stats.rrna_reads_sampled += s.rrna_reads_sampled; stats.rrna_reads += s.rrna_reads; stats.chrm_reads += s.chrm_reads;
            stats.no_barcode += s.no_barcode; stats.barcoded_reads += s.barcoded_reads; stats.multimapper_reads += s.multimapper_reads;
            stats.low_mapq += s.low_mapq; stats.no_umi += s.no_umi;
            n_done += c.n_done; n_unwritten += c.n_unwritten; n_bigclip += c.n_bigclip; n_nohook += c.n_nohook;
            for (int k = 0; k < 8; k++) cb_status_[k] += c.cb_status[k];
            for (size_t k = 0; k < c.cell_reads.size(); k++) per_cell_reads_[k] += c.cell_reads[k];
            for (size_t k = 0; k < c.cell_qc.size(); k++) per_cell_qc_[k].merge(c.cell_qc[k]);
            for (size_t k = 0; k < c.wl_counts.size(); k++) wl_umi_counts_[k] += c.wl_counts[k];
            gpu_pileup_pending_append(c.wl_pending);
            gpu_col_->add_quality_counts64(c.lowq, c.neqn);
            for (const Out& o : c.out) gpu_col_->add_handoff_read_u3(o.rec, o.bc, o.u3, static_cast<int>(H->umiL), o.clip, o.hasq ? &o.qm : nullptr);
            std::vector<Out>().swap(c.out);
        }
        if (n_unwritten) { std::cerr << "[gpu_pileup] hand-off: STAR wrote no entry for " << n_unwritten << " hand-off reads\n"; return false; }
        if (n_bigclip || n_nohook) {
            std::cerr << "[gpu_pileup] hand-off: " << n_bigclip << " reads clipped by more than 255 bases, " << n_nohook
                      << " sampled clipped reads without the rRNA verdict (g_handoffReadHook unset)\n";
            return false;
        }
        std::cerr << "[gpu_pileup] hand-off pass: " << n_done << " reads without BAM records (barcodes and qualities from STAR's per-read entries)\n";
        return true;
    }
    bool gpu_handoff_pass(PileupStats& stats, bool check = false) {
        const gpupileup::Handoff* H = gpu_col_ ? gpu_col_->handoff() : nullptr;
        if (!H) return true;   // no hand-off: every read came through the stream
        if (!H->simple) { std::cerr << "[gpu_pileup] hand-off: barcode layout is not CB_UMI_Simple\n"; return false; }
        // "on": STAR's mapping threads wrote each hand-off read's CR, UR and quality counters (ReadEntry); otherwise (and in check
        // mode) they come from the decoded R1/R2, cut as STAR cuts them
        const bool ent = !check && gpu_col_->load_handoff_reads();
        std::cerr << "[gpu_pileup] hand-off pass: entries loaded\n";
        const gpupileup::ReadEntry* E = ent ? H->reads : nullptr;
        if (ent && (std::getenv("SINGLET_HANDOFF_PASS_SERIAL") == nullptr)) {
            // "on": the pass needs only the hand-off and STAR's per-read entries, so it runs on all cores (S8 stage 8)
            if (gpu_col_->n_handoff_reads() != 0) {
                std::cerr << "[gpu_pileup] hand-off: " << gpu_col_->n_handoff_reads() << " hand-off reads also came through the stream (STAR's skip set differs)\n";
                return false;
            }
            return gpu_handoff_pass_entries(stats, H, E);
        }
        gpupileup::LineReader f1(ent ? nullptr : std::getenv("SINGLET_DECODE_R1")), f2(ent ? nullptr : std::getenv("SINGLET_DECODE_R2"));
        if (!ent && (!f1.ok() || !f2.ok())) return false;
        auto unpack = [](uint64_t v, uint32_t L, std::string& s) {
            static const char kB[8] = {'A', 'C', 'G', 'T', 'N', 'N', 'N', 'N'};
            s.resize(L); for (uint32_t j = 0; j < L; j++) s[j] = kB[(v >> (3 * j)) & 7];
        };
        char comp[256]; for (int i = 0; i < 256; i++) comp[i] = 'N';
        comp[(int)'A'] = 'T'; comp[(int)'C'] = 'G'; comp[(int)'G'] = 'C'; comp[(int)'T'] = 'A';
        std::string l1[4], l2[4];
        // check mode: the stream's own barcode column and UMI of every hand-off read (STAR still wrote their records)
        std::vector<int32_t> t_bc; std::vector<uint64_t> t_u3; uint64_t c_only_pass = 0, c_only_stream = 0, c_bc = 0, c_umi = 0, c_same = 0;
        if (check) {
            std::vector<uint64_t> idx; std::vector<ReadRec> rr; gpu_col_->stream_truth(idx, rr);
            t_bc.assign(H->n_reads, -2); t_u3.assign(H->n_reads, 0);
            for (size_t k = 0; k < idx.size(); ++k) if (idx[k] < H->n_reads) { t_bc[idx[k]] = rr[k].bc; t_u3[idx[k]] = rr[k].umi3; }
        }
        // "on": STAR wrote no records for hand-off reads, so none may have reached the collector from the stream
        if (!check && gpu_col_->n_handoff_reads() != 0) {
            std::cerr << "[gpu_pileup] hand-off: " << gpu_col_->n_handoff_reads() << " hand-off reads also came through the stream (STAR's skip set differs)\n";
            return false;
        }
        PileupStats scratch;
        PileupStats& st = check ? scratch : stats;
        uint64_t n_done = 0;
        std::string bseq, cr, ur, rseq;
        uint64_t n_unwritten = 0; uint64_t n_bigclip = 0, n_nohook = 0;
        for (uint64_t i = 0; i < H->n_reads; ++i) {
            // a read's record is a random access into the hand-off (rec_off, then the record) and into the barcode table: two DRAM misses
            // per read in a loop that is serial; prefetch them a few reads ahead (no effect on the result)
            if (i + 16 < H->n_reads) { const uint32_t r16 = H->read_rec[i + 16]; if (r16 != 0xFFFFFFFFu) __builtin_prefetch(H->rec_off + r16); }
            if (i + 8 < H->n_reads) { const uint32_t r8 = H->read_rec[i + 8]; if (r8 != 0xFFFFFFFFu) __builtin_prefetch(H->data + H->rec_off[r8]); }
            if (!ent) {
                bool ok = true;
                for (int k = 0; k < 4 && ok; ++k) ok = f1.get(l1[k]) && f2.get(l2[k]);
                if (!ok) return false;
            }
            const uint32_t rec = H->read_rec[i];
            if (rec == 0xFFFFFFFFu) continue;   // read with N (or not mapped on the GPU): it came through the stream
            const uint8_t* p = H->data + H->rec_off[rec];
            const int n = *p++;
            if (n == 0) continue;               // unmapped: STAR writes no record (outSAMunmapped None)
            // the record's alignments; the primary
            int32_t ptid = -1, ppos = 0; uint16_t pflag = 0, pnh = 1, pnm = 0; uint8_t pmapq = 0; uint64_t pmlen = 0; int pk = -1, nsec = 0;
            for (int k = 0; k < n; k++) {
                int32_t tid, pos; uint16_t flag, nh, nm, nc; uint8_t mapq;
                std::memcpy(&tid, p, 4); std::memcpy(&pos, p + 4, 4); std::memcpy(&flag, p + 8, 2); mapq = p[10];
                std::memcpy(&nh, p + 12, 2); std::memcpy(&nm, p + 14, 2); std::memcpy(&nc, p + 16, 2);
                const uint8_t* cig = p + 18; p = cig + 4 * nc;
                if (flag & BAM_FSECONDARY) { nsec++; continue; }
                if (pk < 0) {
                    pk = k; ptid = tid; ppos = pos; pflag = flag; pnh = nh; pnm = nm; pmapq = mapq;
                    for (int c = 0; c < nc; c++) { uint32_t v; std::memcpy(&v, cig + 4 * c, 4); const int op = v & 15; if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) pmlen += v >> 4; }
                }
            }
            (void) ppos;
            st.total_reads += n; st.mapped_reads += n; st.secondary_reads += nsec;
            if (pk < 0) continue;
            ++n_done;
            if (ent && !(E[i].flags & 1)) { ++n_unwritten; continue; }
            // the cDNA read: the record's sequence (the prefill's distinct sequences are the R2 lines as decoded), or R2
            uint16_t rl = 0; std::memcpy(&rl, p, 2); const uint8_t* rs = p + 2;
            // a clipped read (STAR's --clip3pAdapterSeq / clip N bases): its record is the clipped sequence; the BAM would hold the
            // whole read, so its length, rRNA verdict and 3' tail come from the entry (HandoffReadEntry flags)
            const uint32_t clip = (ent && (E[i].flags & 0xFFFF00u)) ? E[i].flags : 0u;
            if (clip & 32u) { ++n_bigclip; continue; }
            const int L2 = ent ? (int) rl + (int) ((clip >> 8) & 255) + (int) ((clip >> 16) & 255) : (int) l2[1].size();
            // R4 rRNA sample (primary; once per read), on the sequence as the BAM stores it
            // the R4 sample is 1 read in 64 by the hash of "r<index>"; built in a stack buffer, and only for reads long enough to be sampled
            // (std::string + to_string for every one of ~100 M reads was ~50 ns each)
            bool rrna_sampled = false;
            if (L2 >= 21) {
                char qb[24]; char* q = qb + 23; *q = 0;
                uint64_t v = i; do { *--q = (char) ('0' + v % 10); v /= 10; } while (v);
                *--q = 'r';
                rrna_sampled = (hash_qname(q) & 63) == 0;
            }
            if (rrna_sampled) {
                ++st.rrna_reads_sampled;
                if (clip) {
                    if (!(clip & 4u)) { ++n_nohook; continue; }
                    if (clip & ((pflag & BAM_FREVERSE) ? 16u : 8u)) ++st.rrna_reads;
                } else if (ent) { rseq.resize(L2); for (int j = 0; j < L2; j++) rseq[j] = seq_nt16_str[(rs[j >> 1] >> ((j & 1) * 4)) & 15]; }
                else rseq.assign(l2[1].data(), L2);
                if (!clip) {
                    if (pflag & BAM_FREVERSE) { const std::string fw = rseq; for (int j = 0; j < L2; j++) rseq[j] = comp[(uint8_t) fw[L2 - 1 - j]]; }
                    if (rrna_detector_.detect(rseq)) ++st.rrna_reads;
                }
            }
            if (ptid == chrm_tid_) st.chrm_reads++;
            // CB from CR (the stream's rule), CR / UR as STAR cuts them
            if (ent) { unpack(E[i].cr3, H->cbL, cr); unpack(E[i].ur3, H->umiL, ur); }
            else {
                bseq = l1[1];
                if (bseq.size() < H->cbumiL) bseq.append(H->cbumiL - bseq.size(), 'N');
                cr = bseq.substr(H->cbS - 1, H->cbL); ur = bseq.substr(H->umiS - 1, H->umiL);
            }
            bool cbok = !cr.empty() && !ur.empty() && ur[0] != ' ';
            const CbCorrector* const cbc = config_.cb_corrector;   // R11: as stream_cb, the CR is corrected against the whitelist
            if (!cbc) for (char c : cr) if (c == 'N') cbok = false;
            bool homo = true;
            for (char c : ur) { if (c == 'N') cbok = false; if (c != ur[0]) homo = false; }
            if (homo) cbok = false;
            if (cbc) {
                if (!cbok) ++cb_status_[1];
                else if (static_cast<int>(cr.size()) != cbc->length()) { ++cb_status_[3]; cbok = false; }
                else {
                    const int cst = cbc->resolve(cr.c_str(), nullptr, cb_buf_);
                    ++cb_status_[cst + 4];
                    if (cst < 0) cbok = false;
                    else cr.assign(cb_buf_, static_cast<size_t>(cbc->length()));
                }
            }
            if (!cbok || cr[0] == '-') { st.no_barcode++; continue; }
            const int32_t bc = bc_index_.lookup(cr.c_str());
            if (bc < 0) {
                if (!wl_umi_counts_.empty() && pmapq >= static_cast<uint8_t>(config_.min_mapq)) {
                    const int32_t wl = wl_amb_index_.lookup(cr.c_str());
                    if (wl >= 0) {
                        wl_umi_counts_[static_cast<size_t>(wl)]++;
                        if (has_gene_model_ && !wl_ambient_gene_counts_.empty()) gpu_wl_pending_.push_back({rec, static_cast<uint32_t>(pk)});
                    }
                }
                continue;
            }
            if (check) {   // compare with the stream's barcode and UMI for this read
                uint64_t u3 = 0;
                for (int j = 0; j < (int) ur.size() && j < 21; j++) { const char c = ur[j]; const uint64_t v = c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4; u3 |= v << (3 * j); }
                if (t_bc[i] == -2) c_only_pass++; else if (t_bc[i] != bc) c_bc++; else if (t_u3[i] != u3) c_umi++; else c_same++;
                t_bc[i] = -3;
                continue;
            }
            // every barcoded primary, as the collector does (the GPU gates MAPQ)
            if (ent) gpu_col_->add_quality_counts(E[i].low_qual, E[i].qual_neq_n);
            else gpu_col_->add_quality(l2[1].data(), l2[3].data(), std::min<int>(L2, (int) l2[3].size()));
            gpu_col_->add_handoff_read(rec, bc, ur.c_str(), (int) ur.size(), clip);
            st.barcoded_reads++;
            if (static_cast<uint32_t>(bc) < per_cell_reads_.size()) per_cell_reads_[static_cast<uint32_t>(bc)]++;
            const int nh = std::max<int>(pnh, 1);
            if (nh > 1) st.multimapper_reads++;
            if (static_cast<uint32_t>(bc) < per_cell_qc_.size()) {
                CellAlignQC& q = per_cell_qc_[static_cast<uint32_t>(bc)];
                if (nh > 1) ++q.multimapped; else { ++q.unique; q.mismatches += pnm; q.aligned_bases += pmlen; }
            }
            const bool high = pmapq >= static_cast<uint8_t>(config_.min_mapq);
            if (!high) st.low_mapq++;
            if (!high && nh == 1) continue;
            if (config_.umi_dedup && ur.empty()) st.no_umi++;
        }
        if (check) {
            for (uint64_t i = 0; i < H->n_reads; ++i) if (t_bc[i] >= -1) c_only_stream++;
            std::cerr << "[gpu_pileup] hand-off check: " << c_same << " reads with the stream's barcode and UMI, " << c_bc << " barcode differs, "
                      << c_umi << " UMI differs, " << c_only_pass << " only in the pass, " << c_only_stream << " only in the stream\n";
            return true;
        }
        if (n_unwritten) {
            std::cerr << "[gpu_pileup] hand-off: STAR wrote no entry for " << n_unwritten << " hand-off reads\n";
            return false;
        }
        if (n_bigclip || n_nohook) {
            std::cerr << "[gpu_pileup] hand-off: " << n_bigclip << " reads clipped by more than 255 bases, " << n_nohook
                      << " sampled clipped reads without the rRNA verdict (g_handoffReadHook unset)\n";
            return false;
        }
        std::cerr << "[gpu_pileup] hand-off pass: " << n_done << " reads without BAM records (barcodes and qualities from "
                  << (ent ? "STAR's per-read entries" : "R1/R2") << ")\n";
        return true;
    }
    bool gpu_pileup_apply(PileupStats& stats) {
        using namespace singlet_engine_gpu;
        const auto tA0 = std::chrono::steady_clock::now();
        auto since = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };
        PileupInput in;
        if (!gpu_build_input(in)) return false;
        const double sBuild = since(tA0);
        int lock_fd = -1;
        const auto tL0 = std::chrono::steady_clock::now();
        const int device = lockDevice(&lock_fd);
        const double sWait = since(tL0);
        PileupOutput out;
        const auto tA1 = std::chrono::steady_clock::now();
        const bool ok = pileupRun(in, device, out);
        const double sRun = since(tA1);
        unlockDevice(lock_fd);
        std::cerr << "[gpu_pileup] device lock: waited " << sWait << " s, held " << sRun << " s\n";
        if (!ok) { std::cerr << "[gpu_pileup] failed: " << out.log << "\n"; return false; }
        gpu_put(exon_acc_, out.exon); gpu_put(intron_acc_, out.intron); gpu_put(amb_acc_, out.amb);
        if (reads_matrix_on()) {   // reads per molecule: the GPU's sums and histogram (the CPU store is empty)
            gpu_put(exon_reads_acc_, out.exonReads); gpu_put(intron_reads_acc_, out.intronReads);
            for (size_t i = 0; i < out.histCell.size(); ++i)
                std::copy_n(out.hist.begin() + i * kReadsHistBins, kReadsHistBins, read_hist_[out.histCell[i]].begin());
        }
        gpu_put(snp_ad_, out.snpAd); gpu_put(snp_dp_, out.snpDp); gpu_put(mt_acc_, out.mt); gpu_put(mt_acc_rev_, out.mtRev);
        sj_names_.clear(); sj_index_.clear();
        // the GPU's table has every junction of every signature (hand-off records include reads without a cell barcode); the CPU
        // names a junction when a counted read has it, so keep the junctions with a count, in table (R7) order
        std::vector<uint32_t> sj_new(out.sjtab.size(), UINT32_MAX);
        for (const auto& t : out.sj) if (t.row < sj_new.size()) sj_new[t.row] = 0;
        for (size_t j = 0; j < out.sjtab.size(); ++j) {
            if (sj_new[j] == UINT32_MAX) continue;
            const auto& k = out.sjtab[j];
            sj_new[j] = static_cast<uint32_t>(sj_names_.size());
            sj_index_[((static_cast<uint64_t>(k[0]) << 32) | k[1]) ^ (static_cast<uint64_t>(k[2]) * 0x9e3779b97f4a7c15ULL)] =
                sj_new[j];   // keyed as extract_splice_junctions keys them
            sj_names_.push_back(tid_to_chrom_[k[0]] + ":" + std::to_string(k[1]) + "-" + std::to_string(k[2]));
        }
        for (auto& t : out.sj) t.row = sj_new[t.row];
        if (!sj_names_.empty()) {   // the CPU grows the feature count in steps of 1000 as junctions appear (extract_splice_junctions)
            const uint32_t n = static_cast<uint32_t>(sj_names_.size());
            sj_acc_.set_barcodes(barcodes_); sj_acc_.set_n_features(((n - 1) / 1000 + 1) * 1000);
        }
        gpu_put(sj_acc_, out.sj);
        for (const auto& e : out.indels) mt_indels_.push_back(mt::MtIndelEvent{e[0], e[1], static_cast<uint16_t>(e[2]), e[3] != 0});
        stats.snp_hits += out.snpHits; stats.sj_hits += out.sjHits; stats.mt_pileup_bases += out.mtBases;
        stats.snp_molecules += out.snpMolecules; stats.snp_molecules_ambiguous += out.snpAmbiguous; stats.snp_molecules_other += out.snpOther;
        gpu_ms_ = DirectionalUmiStore::MoleculeStats{};
        gpu_ms_.spliced = out.molSpliced; gpu_ms_.unspliced = out.molUnspliced; gpu_ms_.ambiguous = out.molAmbiguous;
        gpu_ms_.legacy_double = out.molLegacyDouble; gpu_ms_.dropped = out.molDropped;
        gpu_cell_reads_.swap(out.cellReads); gpu_cell_mols_.swap(out.cellMolecules);
        gpu_merges_.swap(out.geneMerges);
        stats.mt_numt_rescued += out.mtRescued; stats.exon_hits += out.hitsExon; stats.intron_hits += out.hitsIntron; stats.umi_duplicate += out.umiRecorded;
        if (!gpu_wl_pending_.empty()) {   // hand-off reads with a WL-ambient barcode: their primary's exon gene (determine_exon_gene)
            const SigRec* S = reinterpret_cast<const SigRec*>(in.sigs.data());
            for (const auto& [rec, k] : gpu_wl_pending_) {
                const uint32_t g = out.alnExonGene[S[rec].aln_off + k];
                if (g != UINT32_MAX && g < wl_ambient_gene_counts_.size()) wl_ambient_gene_counts_[g]++;
            }
            gpu_wl_pending_.clear();
        }
        const auto tA2 = std::chrono::steady_clock::now();
        if (has_gene_model_ && !gpu_host_pass(in, out, stats)) return false;
        std::cerr << "[gpu_pileup] apply: build input " << sBuild << " s, pileupRun " << sRun << " s, host pass " << since(tA2) << " s" << "\n";
        std::cerr << "[gpu_pileup] on: GPU " << out.msGpu << " ms, total " << out.msTotal << " ms, peak " << out.peakGB << " GB; molecules "
                  << gpu_ms_.total() << ", junctions " << sj_names_.size() << ", R1 molecules " << out.snpMolecules
                  << (out.log.empty() ? "" : "; ") << out.log << "\n";
        return true;
    }
#endif

#ifdef SINGLET_HAVE_ENGINE_GPU
    // R9 per gene, GPU path: the CPU gene store is empty, so the V(D)J reducer takes a UMI's root from the GPU's gene store: the UMIs its
    // directional collapse merged into another, sorted by (cell, gene, UMI); any other UMI of an assigned read is its own root
    std::vector<singlet_engine_gpu::PileupGeneMerge> gpu_merges_;
    struct GpuGeneRoots {
        const std::vector<singlet_engine_gpu::PileupGeneMerge>* merges;
        bool operator()(uint32_t bc, uint32_t gene, uint64_t umi3, uint64_t& root) const {
            int len = 0;
            const uint64_t p2 = umi3_to_2bit(umi3, &len);
            root = p2;
            const auto it = std::lower_bound(merges->begin(), merges->end(), std::make_tuple(bc, gene, p2),
                [](const singlet_engine_gpu::PileupGeneMerge& m, const std::tuple<uint32_t, uint32_t, uint64_t>& k) {
                    return std::tie(m.cell, m.gene, m.umi) < k; });
            if (it != merges->end() && it->cell == bc && it->gene == gene && it->umi == p2) root = it->root;
            return true;
        }
    };
#endif
    bool gpu_covers_ = false;   // W11: SINGLET_GPU_PILEUP=on in a GPU build (always false in a CPU-only build)
    std::vector<uint32_t> gpu_cell_reads_, gpu_cell_mols_;   // W11 "on": per-cell molecule-store reads and molecules from the GPU
    // W11 "on": the per-cell saturation inputs the molecule store would give (empty when the CPU store was used)
    bool gpu_saturation(std::vector<std::pair<uint32_t, std::pair<uint64_t, uint64_t>>>& cells) const {
        if (!gpu_covers_ || gpu_cell_reads_.empty()) return false;
        cells.clear();
        for (uint32_t c = 0; c < gpu_cell_reads_.size(); ++c)
            if (gpu_cell_reads_[c] > 0) cells.push_back({c, {gpu_cell_reads_[c], gpu_cell_mols_[c]}});
        return true;
    }
    // Get accumulators for export
    const SparseAccumulator<uint16_t>& snp_ad() const { return snp_ad_; }
    const SparseAccumulator<uint16_t>& snp_dp() const { return snp_dp_; }
    const SparseAccumulator<uint16_t>& exons() const { return exon_acc_; }
    const SparseAccumulator<uint16_t>& introns() const { return intron_acc_; }
    const SparseAccumulator<uint16_t>& splice_junctions() const { return sj_acc_; }
    const SparseAccumulator<uint16_t>& mt_alleles() const { return mt_acc_; }
    /// Reverse-strand subset of mt_alleles() (same feature layout); forward-strand = total - reverse.
    const SparseAccumulator<uint16_t>& mt_alleles_rev() const { return mt_acc_rev_; }
    const std::vector<mt::MtIndelEvent>& mt_indels() const { return mt_indels_; }

    // N18: CRISPR guide counts accessor
    const SparseAccumulator<uint16_t>& guide_counts() const { return guide_ctr_.accumulator(); }
    std::vector<std::string> guide_names() const { return guide_ref_.names(); }
    // N17: VDJ gene usage accessor
    const SparseAccumulator<uint16_t>& vdj_counts() const { return vdj_acc_; }
    const std::vector<std::string>& vdj_names() const { return vdj_model_.gene_names(); }
    bool has_vdj() const { return has_vdj_; }

    const std::vector<std::string>& barcodes() const { return barcodes_; }
    /// N20: per-barcode read counts (barcoded primary reads, pre-dedup)
    const std::vector<uint32_t>& per_cell_reads() const { return per_cell_reads_; }
    /// Per-cell alignment QC over barcoded primary reads (multimappers, nM mismatches, aligned bases).
    const std::vector<CellAlignQC>& per_cell_align_qc() const { return per_cell_qc_; }
    /// Poly(A)-tailed read ends (count_polya); export clusters them (polya_sites.h). Mutable: export sorts in place.
    std::vector<PolyARecord>& polya_records() const { return polya_recs_; }
    const std::vector<std::string>& tid_to_chrom() const { return tid_to_chrom_; }
    /// N13: first SNP index belonging to the built-in AIM panel (UINT32_MAX if not injected)
    uint32_t aim_snp_start() const { return aim_snp_start_; }
    const std::vector<std::string>& snp_names() const { return snp_names_; }
    const std::vector<std::string>& exon_names() const { return gene_model_.exon_names(); }
    const std::vector<std::string>& intron_names() const { return gene_model_.intron_names(); }
    const std::vector<std::string>& gene_names() const { return gene_model_.gene_names(); }
    const std::vector<std::string>& gene_ids() const { return gene_model_.gene_ids(); }
    const std::vector<std::string>& sj_names() const { return sj_names_; }

    const GeneModel& gene_model() const { return gene_model_; }
    const UmiDedup& umi_stats() const { return umi_dedup_; }
    bool has_gene_model() const { return has_gene_model_; }
    /// True if the auto-strand probe changed reverse_strand (not merely ran).
    bool strand_was_flipped() const { return strand_flipped_; }
    /// True if the auto-strand probe saw enough informative reads to decide.
    bool strand_was_probed() const { return strand_probed_; }
    /// Strand evidence the probe saw (zero if it did not run).
    const StrandEvidence& strand_probe() const { return strand_probe_; }
    /// Orientation the pass counted with (after any probe flip).
    bool reverse_strand() const { return config_.reverse_strand; }
    /// Per-molecule dump path (diagnostics; "" = none). Defaults to SINGLET_MOLECULE_DUMP.
    void set_molecule_dump(const std::string& path) { molecule_dump_ = path; }
    /// Carry the probe record of an earlier pass over to this (retry) engine.
    void keep_strand_probe(const PileupEngine& prev) {
        strand_probed_ = prev.strand_probed_;
        strand_probe_ = prev.strand_probe_;
    }

    /// N22: Per-WL-barcode UMI counts (run() serial path only; empty in run_parallel())
    const std::vector<uint32_t>& wl_umi_counts() const { return wl_umi_counts_; }
    /// N22: Global gene hit sums from WL-only (non-auto-discovered) barcodes
    const std::vector<uint64_t>& wl_ambient_gene_counts() const { return wl_ambient_gene_counts_; }
    /// N22: UMI ceiling used to classify WL barcodes as ambient
    uint32_t wl_ambient_ceil() const { return config_.wl_ambient_ceil; }

    // N7: Molecule store for saturation computation (run() path; empty after run_parallel())
    const DirectionalUmiStore& dir_exon_store() const { return mol_store_; }
    /// Ambiguous molecules per gene (rows = genes). They are also counted in exons().
    const SparseAccumulator<uint16_t>& ambiguous() const { return amb_acc_; }

    /// Reads per molecule, same layout as exons() / introns() (uint32: not clamped at 65535). Empty unless reads_matrix_on().
    bool reads_matrix_on() const { return config_.reads_matrix && config_.umi_dedup && has_gene_model_; }
    static constexpr int kReadsHistBins = 16;   ///< bin k-1 = molecules with k reads, k = 1..15; bin 15 = 16 or more
    const SparseAccumulator<uint32_t>& exon_reads() const { return exon_reads_acc_; }
    const SparseAccumulator<uint32_t>& intron_reads() const { return intron_reads_acc_; }
    /// Per-cell histogram of reads per emitted molecule, barcode-index order, nonzero cells only.
    const std::map<uint32_t, std::array<uint32_t, kReadsHistBins>>& reads_per_umi_hist() const { return read_hist_; }

    // G-EM: Ambiguous reads buffered for post-pileup EM rescue
    const std::vector<em_rescue::AmbigRead>& ambig_reads() const { return ambig_reads_; }

private:
    PileupConfig config_;
    // Reads per molecule (written by finalize_all on the main thread, or put from the GPU): never read by the counting path
    mutable SparseAccumulator<uint32_t> exon_reads_acc_, intron_reads_acc_;
    mutable std::map<uint32_t, std::array<uint32_t, kReadsHistBins>> read_hist_;
    void add_read_hist(uint32_t bc, uint32_t reads) const {
        auto& h = read_hist_[bc];
        h[std::min<uint32_t>(reads, kReadsHistBins) - 1]++;
    }
    std::string molecule_dump_;     // SINGLET_MOLECULE_DUMP: per-molecule TSV (gz) for diagnostics
    bool strand_probed_ = false;    // Auto-strand probe reached a decision
    bool strand_flipped_ = false;   // ... and changed reverse_strand
    StrandEvidence strand_probe_;

    // Sense / antisense vote of one aligned block's exon overlaps (see StrandEvidence).
    void strand_vote(const std::vector<uint32_t>& overlaps, bool read_reverse,
                     bool& sense, bool& anti) const {
        for (uint32_t eidx : overlaps) {
            char gs = gene_model_.gene_strand(gene_model_.exon_to_gene(eidx));
            if (gs == '.') continue;
            if (read_reverse == (gs == '-')) sense = true; else anti = true;
        }
    }

    // Record the probe and set the orientation. Flips only on a clear majority
    // (antisense share > kStrandProbeReverse or < kStrandProbeForward); in between
    // (unstranded or noisy) the configured orientation stays. Returns true if flipped.
    bool apply_strand_probe(const StrandEvidence& ev, const char* source) {
        strand_probe_ = ev;
        if (ev.informative() < kStrandProbeMinReads) return false;
        strand_probed_ = true;
        double af = ev.antisense_frac();
        bool want_reverse;
        if (af > kStrandProbeReverse) want_reverse = true;
        else if (af < kStrandProbeForward) want_reverse = false;
        else return false;
        if (want_reverse == config_.reverse_strand) return false;
        config_.reverse_strand = want_reverse;
        strand_flipped_ = true;
        std::cerr << "[singlet-pileup] Strand probe: " << static_cast<int>(af * 100)
                  << "% of " << ev.informative() << " informative reads antisense ("
                  << source << ") -> " << (want_reverse ? "reverse" : "forward")
                  << "-strand counting\n";
        return true;
    }

    // Sample an indexed BAM for strand evidence: the kStrandProbeChroms nuclear
    // chromosomes with the most reads (that have exons), kStrandProbeWindows windows
    // spread along each. The first reads of a sorted chromosome cover only a few
    // genes, and chrM carries genes on both strands. Returns true if the probe decided.
    static constexpr int kStrandProbeChroms = 4;
    static constexpr int kStrandProbeWindows = 16;
    bool probe_strand_indexed(htsFile* fp, sam_hdr_t* hdr, hts_idx_t* idx) {
        int n = sam_hdr_nref(hdr);
        std::vector<std::pair<uint64_t, int>> cand;
        std::vector<uint32_t> ov;
        for (int tid = 0; tid < n; ++tid) {
            if (is_mito_contig(sam_hdr_tid2name(hdr, tid))) continue;
            uint64_t mapped = 0, unmapped = 0;
            if (hts_idx_get_stat(idx, tid, &mapped, &unmapped) != 0 || mapped == 0) continue;
            gene_model_.query_exons(tid, 0, static_cast<int32_t>(
                std::min<hts_pos_t>(sam_hdr_tid2len(hdr, tid), INT32_MAX)), ov);
            if (ov.empty()) continue;
            cand.push_back({mapped, tid});
        }
        std::sort(cand.rbegin(), cand.rend());
        if (cand.size() > static_cast<size_t>(kStrandProbeChroms)) cand.resize(kStrandProbeChroms);
        if (cand.empty()) return false;

        const uint64_t per_window = kStrandProbeTarget / (cand.size() * kStrandProbeWindows) + 1;
        const uint64_t scan_cap = per_window * 50;  // reads per window, informative or not
        StrandEvidence ev;
        bam1_t* b = bam_init1();
        std::vector<std::pair<int32_t, int32_t>> blocks;
        for (const auto& [nreads, tid] : cand) {
            (void)nreads;
            hts_pos_t len = sam_hdr_tid2len(hdr, tid);
            for (int w = 0; w < kStrandProbeWindows; ++w) {
                hts_pos_t beg = len * w / kStrandProbeWindows;
                hts_pos_t end = len * (w + 1) / kStrandProbeWindows;
                hts_itr_t* itr = sam_itr_queryi(idx, tid, beg, end);
                if (!itr) continue;
                uint64_t got = 0, scanned = 0;
                while (got < per_window && scanned < scan_cap && sam_itr_next(fp, itr, b) >= 0) {
                    if (b->core.pos < beg) continue;  // starts in the previous window
                    ++scanned;
                    if (b->core.flag & (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY)) continue;
                    if (b->core.qual < config_.min_mapq) continue;
                    blocks.clear();
                    int32_t rpos = static_cast<int32_t>(b->core.pos);
                    const uint32_t* cig = bam_get_cigar(b);
                    for (uint32_t ci = 0; ci < b->core.n_cigar; ++ci) {
                        int op = bam_cigar_op(cig[ci]);
                        int32_t l = static_cast<int32_t>(bam_cigar_oplen(cig[ci]));
                        if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                            blocks.push_back({rpos, rpos + l});
                            rpos += l;
                        } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
                            rpos += l;
                        }
                    }
                    bool rrev = (b->core.flag & BAM_FREVERSE) != 0, vs = false, va = false;
                    for (const auto& [bs, be] : blocks) {
                        gene_model_.query_exons(tid, bs, be, ov);
                        strand_vote(ov, rrev, vs, va);
                    }
                    if (vs != va) { (vs ? ev.sense : ev.antisense)++; ++got; }
                }
                hts_itr_destroy(itr);
            }
        }
        bam_destroy1(b);
        if (ev.informative() < kStrandProbeMinReads) {
            strand_probe_ = ev;
            return false;
        }
        apply_strand_probe(ev, "nuclear chromosomes");
        return true;
    }

    // probe_strand_indexed() on a BAM file with an index; false for streams.
    bool probe_strand_file(const std::string& path, sam_hdr_t* hdr) {
        if (path.empty() || path == "-" || path.rfind("/dev/", 0) == 0) return false;
        htsFile* fp = hts_open(path.c_str(), "r");
        if (!fp) return false;
        bool decided = false;
        if (hts_idx_t* idx = sam_index_load(fp, path.c_str())) {
            decided = probe_strand_indexed(fp, hdr, idx);
            hts_idx_destroy(idx);
        }
        hts_close(fp);
        return decided;
    }
    std::vector<std::string> barcodes_;
    std::vector<uint32_t> per_cell_reads_;  // N20: barcoded primary reads per cell (pre-dedup)
    std::vector<CellAlignQC> per_cell_qc_;  // per-cell alignment QC (see CellAlignQC)
    mutable std::vector<PolyARecord> polya_recs_;  // poly(A)-tailed read ends (count_polya)
    // Fast barcode lookup: C-string → index, no std::string construction
    struct BarcodeIndex {
        std::unordered_map<uint64_t, int32_t> map;
        void build(const std::vector<std::string>& barcodes) {
            map.reserve(barcodes.size() * 2);
            for (uint32_t i = 0; i < barcodes.size(); ++i) {
                const auto& bc = barcodes[i];
                map[hash_cstr(bc.c_str(), bc.size())] = static_cast<int32_t>(i);
                auto dash = bc.rfind('-');
                if (dash != std::string::npos && dash > 0) {
                    map[hash_cstr(bc.c_str(), dash)] = static_cast<int32_t>(i);
                }
            }
        }
        int32_t lookup(const char* s) const {
            size_t len = 0;
            const char* p = s;
            const char* dash = nullptr;
            while (*p) { if (*p == '-') dash = p; ++len; ++p; }
            auto it = map.find(hash_cstr(s, len));
            if (it != map.end()) return it->second;
            if (dash && dash > s) {
                it = map.find(hash_cstr(s, static_cast<size_t>(dash - s)));
                if (it != map.end()) return it->second;
            }
            return -1;
        }
    } bc_index_;

    // N22: Full-whitelist ambient index (separate from auto-discovered barcodes in bc_index_)
    // Enables EmptyDrops to use a representative empty-droplet pool from the full barcode whitelist.
    struct WlAmbientIndex {
        std::unordered_map<uint64_t, uint32_t> map;
        void build(const std::vector<std::string>& barcodes) {
            map.reserve(barcodes.size() * 2);
            for (uint32_t i = 0; i < static_cast<uint32_t>(barcodes.size()); ++i) {
                const auto& bc = barcodes[i];
                map[hash_cstr(bc.c_str(), bc.size())] = i;
                auto dash = bc.rfind('-');
                if (dash != std::string::npos && dash > 0)
                    map[hash_cstr(bc.c_str(), dash)] = i;
            }
        }
        int32_t lookup(const char* s) const {
            size_t len = 0;
            const char* p = s;
            const char* dash = nullptr;
            while (*p) { if (*p == '-') dash = p; ++len; ++p; }
            auto it = map.find(hash_cstr(s, len));
            if (it != map.end()) return static_cast<int32_t>(it->second);
            if (dash && dash > s) {
                it = map.find(hash_cstr(s, static_cast<size_t>(dash - s)));
                if (it != map.end()) return static_cast<int32_t>(it->second);
            }
            return -1;
        }
        bool empty() const { return map.empty(); }
    } wl_amb_index_;
    std::vector<uint32_t> wl_umi_counts_;          // [i] = UMI count for WL barcode i (run() serial path)
    std::vector<uint64_t> wl_ambient_gene_counts_; // global gene hit sums from WL-only barcodes

    // Gene model (exons, introns, gene hierarchy)
    GeneModel gene_model_;
    bool has_gene_model_ = false;

    // Gene-model molecules: one store for exonic and intronic reads, so each
    // (barcode, gene, UMI) is labelled spliced/unspliced/ambiguous exactly once
    // (run() single-thread path; run_parallel() uses per-worker stores).
    DirectionalUmiStore mol_store_;
    SparseAccumulator<uint16_t> amb_acc_;  // ambiguous molecules per gene (also in exon_acc_)
    UmiDedup umi_dedup_;  // kept for umi_stats(); gene-model counting no longer uses it

    // Per-chromosome sorted SNP data (binary search, cache-friendly)
    std::vector<std::vector<ChrSNP>> chrom_snps_;      // indexed by tid
    std::vector<std::string> snp_names_;
    uint32_t n_snps_loaded_ = 0;
    uint32_t aim_snp_start_ = UINT32_MAX;   // N13: first AIM index in snp matrices

    // Splice junction tracking
    std::unordered_map<uint64_t, uint32_t> sj_index_;  // (tid<<32|donor)^acceptor → sj idx
    std::vector<std::string> sj_names_;

    // Chromosome name ↔ BAM tid
    std::unordered_map<std::string, int32_t> chrom_to_tid_;
    std::vector<std::string> tid_to_chrom_;  // Reverse map: tid → chrom name
    int32_t chrm_tid_ = -1;
    std::vector<char> mito_tid_;  // per tid: mitochondrial contig (excluded from strand evidence)

    // Accumulators
    SparseAccumulator<uint16_t> snp_ad_;
    SparseAccumulator<uint16_t> snp_dp_;
    SparseAccumulator<uint16_t> exon_acc_;
    SparseAccumulator<uint16_t> intron_acc_;
    SparseAccumulator<uint16_t> sj_acc_;
    // chrM keeps one triplet per aligned base; fold duplicates once this many are held
    // (memory then follows distinct (position x base, barcode), not chrM reads).
    static constexpr size_t kMtCompactTriplets = size_t(16) << 20;
    SparseAccumulator<uint16_t> mt_acc_;   // chrM allele counts: feat = pos*4 + base
    SparseAccumulator<uint16_t> mt_acc_rev_;  // chrM reverse-strand allele counts (same layout)
    std::vector<mt::MtIndelEvent> mt_indels_;  // chrM indels captured during CIGAR walk
    SnpConsensus snp_cons_;   // per-molecule SNP allele consensus (order-free-rules R1)
    UmiFeatureCollapse sj_umi_dedup_;  // splice-junction molecules (order-free-rules R9)
    uint32_t cur_read_gene_ = UINT32_MAX;  // R9: gene of the current unique read, for its junction/V(D)J/SNP evidence

    // G-EM: ambiguous reads buffered for post-pileup EM rescue
    std::vector<em_rescue::AmbigRead> ambig_reads_;

    // N18: CRISPR guide accumulators
    GuideRef    guide_ref_;
    GuideCounter guide_ctr_;
    std::string guide_seq_buf_;  // scratch: BAM-decoded query sequence for guide matching

    // N17: VDJ model and accumulator
    VdjModel vdj_model_;
    bool has_vdj_ = false;
    SparseAccumulator<uint16_t> vdj_acc_;
    UmiFeatureCollapse vdj_umi_;       // V(D)J molecules (order-free-rules R9)

    // G-RRNA: rRNA contamination detector (stateless, shared across serial path)
    singlet::RrnaDetector rrna_detector_;

    // Reusable scratch vectors for aligned block queries (avoid per-read allocation)
    mutable std::vector<std::pair<int32_t, int32_t>> aligned_blocks_;
    mutable std::vector<uint32_t> block_overlaps_;
    mutable std::vector<uint32_t> broad_overlap_genes_;  // any-overlap genes for ambiguity
    mutable std::vector<uint32_t> mm_overlaps_;           // scratch for determine_exon_gene
    mutable std::vector<uint32_t> vdj_overlaps_;          // N17: scratch for VDJ queries

    // Lightweight gene determination for secondary multi-mapped alignments.
    // Returns {gene_idx, exon_idx, multi_gene}. gene_idx = UINT32_MAX if intergenic/ambiguous.
    struct GeneHit {
        uint32_t gene_idx = UINT32_MAX;
        uint32_t exon_idx = UINT32_MAX;
        bool multi_gene = false;  // true if overlaps 2+ genes (vs intergenic)
    };

    // ── Hash functions for zero-allocation lookups ──
    static uint64_t hash_cstr(const char* s, size_t len) {
        uint64_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < len; ++i) {
            h ^= static_cast<uint8_t>(s[i]);
            h *= 1099511628211ULL;
        }
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        return h;
    }

    // Test hook (ws-w3b, R10b stability gate): SINGLET_TEST_DROP_READS_PCT=P skips every alignment of the reads whose
    // name hashes into P percent, as if those reads had not been sequenced. 0 / unset = off.
    static bool test_drop_read(const bam1_t* b) {
        static const uint64_t pct = [] {
            const char* e = std::getenv("SINGLET_TEST_DROP_READS_PCT");
            return e ? static_cast<uint64_t>(std::strtoull(e, nullptr, 10)) : uint64_t(0);
        }();
        if (pct == 0) return false;
        const uint64_t h = hash_qname(bam_get_qname(b)) * 0x9E3779B97F4A7C15ULL;
        return (h >> 32) % 100 < pct;
    }

    // R11: the stream's CB (PileupConfig::cb_from_cr). `count` tallies primary records by correction status.
    char cb_buf_[40] = {};
    uint64_t cb_status_[8] = {};   // index status + 4: -4 rejected, -3 bad UMI, -2 two N, -1 no match, 0 exact, 1 one, 2 posterior
    const char* stream_cb(const bam1_t* b, bool count) {
        const CbCorrector* cc = config_.cb_corrector;
        const char* cr = inline_scan_cb_from_cr(b, cc == nullptr);
        if (!cc) return cr;
        if (!cr) { if (count) ++cb_status_[1]; return nullptr; }
        if (static_cast<int>(std::strlen(cr)) != cc->length()) { if (count) ++cb_status_[3]; return nullptr; }
        const int st = cc->resolve(cr, nullptr, cb_buf_);
        if (count) ++cb_status_[st + 4];
        if (st < 0) return nullptr;
        cb_buf_[cc->length()] = '\0';
        return cb_buf_;
    }
    void report_cb_status() const {
        if (!config_.cb_corrector) return;
        std::cerr << "[cb-correct] primary records: exact=" << cb_status_[4] << " one_candidate=" << cb_status_[5]
                  << " posterior_accepted=" << cb_status_[6] << " posterior_rejected=" << cb_status_[0]
                  << " no_match=" << cb_status_[3] << " two_N=" << cb_status_[2] << " bad_umi=" << cb_status_[1]
                  << " (whitelist " << config_.cb_corrector->size() << ")\n";
    }

    // R11 check (sorted path, SINGLET_TEST_CB_CHECK=1): the corrector's CB from CR/UR against STAR's CB tag, per primary record.
    const bool cb_check_ = [] { const char* e = std::getenv("SINGLET_TEST_CB_CHECK"); return e && e[0] == '1'; }();
    std::atomic<uint64_t> cb_check_n_[4] = {};   // [0] agree (same CB, or neither has one), [2] differ, [3] STAR has a CB and ours none
    std::atomic<uint32_t> cb_check_shown_{0};
    void check_cb_against_star(const bam1_t* b, const char* star_cb) {
        const CbCorrector* cc = config_.cb_corrector;
        if (!cc || config_.cb_from_cr) return;
        const char* cr = inline_scan_cb_from_cr(b, false);   // nullptr: bad UMI or no CR
        char ours[40] = {};
        int st = -3;
        if (cr && static_cast<int>(std::strlen(cr)) == cc->length()) st = cc->resolve(cr, nullptr, ours);
        const bool star_none = !star_cb || star_cb[0] == '-';
        const bool ours_none = st < 0;
        int k;
        if (star_none && ours_none) k = 1;
        else if (star_none || ours_none) k = ours_none ? 3 : 2;
        else k = std::strncmp(star_cb, ours, cc->length()) == 0 ? 0 : 2;
        ++cb_check_n_[k == 1 ? 0 : k];
        if ((k == 2 || k == 3) && cb_check_shown_.fetch_add(1) < 20) {
            std::cerr << "[cb-check] differ: CR=" << (cr ? cr : "(none)") << " STAR CB=" << (star_none ? "-" : star_cb)
                      << " ours=" << (ours_none ? "-" : std::string(ours, cc->length())) << " status=" << st << "\n";
        }
    }
    void report_cb_check() const {
        if (!cb_check_) return;
        std::cerr << "[cb-check] primary records: agree=" << cb_check_n_[0] << " differ=" << cb_check_n_[2]
                  << " ours_none_star_cb=" << cb_check_n_[3] << "\n";
    }

    static uint64_t hash_qname(const char* s) {
        uint64_t h = 14695981039346656037ULL;
        while (*s) {
            h ^= static_cast<uint8_t>(*s++);
            h *= 1099511628211ULL;
        }
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        return h;
    }

    GeneHit determine_exon_gene(bam1_t* b, int32_t tid) {
        GeneHit result;
        if (!has_gene_model_) return result;

        // Build aligned blocks from CIGAR
        aligned_blocks_.clear();
        const uint32_t* cigar = bam_get_cigar(b);
        int32_t rpos = b->core.pos;
        for (int ci = 0; ci < b->core.n_cigar; ++ci) {
            int op = bam_cigar_op(cigar[ci]);
            int len = bam_cigar_oplen(cigar[ci]);
            switch (op) {
                case BAM_CMATCH:
                case BAM_CEQUAL:
                case BAM_CDIFF:
                    aligned_blocks_.push_back({rpos, rpos + len});
                    rpos += len;
                    break;
                case BAM_CDEL:
                case BAM_CREF_SKIP:
                    rpos += len;
                    break;
                default:
                    break;
            }
        }

        // Check exon any-overlap (strand-aware — matches STARsolo Gene feature)
        bool read_reverse = (b->core.flag & BAM_FREVERSE) != 0;
        mm_overlaps_.clear();
        for (const auto& [bstart, bend] : aligned_blocks_) {
            gene_model_.query_exons(tid, bstart, bend, block_overlaps_);
            for (uint32_t idx : block_overlaps_) {
                bool wrong_strand = false;
                if (config_.stranded) {
                    uint32_t g = gene_model_.exon_to_gene(idx);
                    char gs = gene_model_.gene_strand(g);
                    if (gs != '.') {
                        bool expect_reverse = (gs == '-');
                        if (config_.reverse_strand) expect_reverse = !expect_reverse;
                        if (read_reverse != expect_reverse) wrong_strand = true;
                    }
                }
                if (!wrong_strand) {
                    mm_overlaps_.push_back(idx);
                }
            }
        }

        if (mm_overlaps_.empty()) return result;

        // Check single-gene (any-overlap-based)
        uint32_t gene = gene_model_.exon_to_gene(mm_overlaps_[0]);
        for (size_t i = 1; i < mm_overlaps_.size(); ++i) {
            uint32_t g = gene_model_.exon_to_gene(mm_overlaps_[i]);
            if (g != gene) {
                result.multi_gene = true;
                return result;
            }
        }

        result.gene_idx = gene;
        result.exon_idx = mm_overlaps_[0];
        return result;
    }

    static uint64_t pos_key(int32_t tid, int32_t pos) {
        return (static_cast<uint64_t>(tid) << 32) | static_cast<uint64_t>(pos);
    }

    bool is_chrm(int32_t tid) const { return tid == chrm_tid_; }

    // ── RAII gzip-aware line reader ──
    class LineReader {
    public:
        explicit LineReader(const std::string& path) {
            bool is_gz = (path.size() >= 3 && path.substr(path.size() - 3) == ".gz");
            if (is_gz) {
                gz_ = gzopen(path.c_str(), "r");
                if (gz_) gzbuffer(gz_, 256 * 1024);
                ok_ = (gz_ != nullptr);
            } else {
                f_.open(path);
                ok_ = f_.is_open();
            }
        }
        ~LineReader() { if (gz_) gzclose(gz_); }
        bool good() const { return ok_; }
        bool getline(std::string& line) {
            if (gz_) {
                line.clear();
                char buf[4096];
                while (gzgets(gz_, buf, sizeof(buf))) {
                    line += buf;
                    if (!line.empty() && line.back() == '\n') {
                        line.pop_back();
                        if (!line.empty() && line.back() == '\r') line.pop_back();
                        return true;
                    }
                }
                return !line.empty();
            } else {
                return static_cast<bool>(std::getline(f_, line));
            }
        }
    private:
        gzFile gz_ = nullptr;
        std::ifstream f_;
        bool ok_ = false;
    };

    // ── Chromosome name resolution ──
    void build_chrom_map(sam_hdr_t* hdr) {
        int n = sam_hdr_nref(hdr);
        tid_to_chrom_.resize(n);
        mito_tid_.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            std::string name = sam_hdr_tid2name(hdr, i);
            chrom_to_tid_[name] = i;
            tid_to_chrom_[i] = name;
            mito_tid_[i] = is_mito_contig(name);
            if (name == "chrM" || name == "MT") {
                chrm_tid_ = i;
            }
        }
    }

    int32_t resolve_chrom(const std::string& chrom) const {
        auto it = chrom_to_tid_.find(chrom);
        if (it != chrom_to_tid_.end()) return it->second;

        // Try adding "chr" prefix
        if (chrom.substr(0, 3) != "chr") {
            std::string with_chr = "chr" + chrom;
            if (chrom == "MT") with_chr = "chrM";
            it = chrom_to_tid_.find(with_chr);
            if (it != chrom_to_tid_.end()) return it->second;
        }

        // Try removing "chr" prefix
        if (chrom.substr(0, 3) == "chr") {
            std::string without_chr = chrom.substr(3);
            if (chrom == "chrM") without_chr = "MT";
            it = chrom_to_tid_.find(without_chr);
            if (it != chrom_to_tid_.end()) return it->second;
        }

        return -1;
    }

    // ── Reference loaders ──

    // N13: Inject built-in AIM positions into snp_pending_ so they are piled up
    // alongside any user-provided VCF SNPs.  Called after load_snps().
    void load_aim_panel() {
        const auto& aims = singlet::get_aim_panel();
        aim_snp_start_ = n_snps_loaded_;
        for (const auto& aim : aims) {
            SNPSite site;
            site.index     = n_snps_loaded_++;
            site.ref_allele = aim.ref;
            site.alt_allele = aim.alt;
            snp_pending_.push_back({std::string(aim.chrom), aim.pos_0b, site});
            // name: rsid (chrom:pos+1:R>A)
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%s:%d:%c>%c",
                          aim.chrom, aim.pos_0b + 1, aim.ref, aim.alt);
            snp_names_.emplace_back(buf);
        }
        std::cerr << "[singlet-pileup] Injected " << aims.size()
                  << " AIM positions for ancestry (start_idx=" << aim_snp_start_ << ")\n";
    }

    bool load_barcodes() {
        LineReader f(config_.barcode_path);
        if (!f.good()) {
            std::cerr << "[singlet-pileup] ERROR: Cannot open barcode file: "
                      << config_.barcode_path << "\n";
            return false;
        }
        std::string line;
        while (f.getline(line)) {
            if (!line.empty() && line[0] != '#') {
                auto end = line.find_first_of("\t\r\n ");
                if (end != std::string::npos) line.resize(end);
                if (!line.empty()) {
                    barcodes_.push_back(line);
                }
            }
        }
        bc_index_.build(barcodes_);
        std::cerr << "[singlet-pileup] Loaded " << barcodes_.size() << " barcodes\n";
        return !barcodes_.empty();
    }

    // N22: Load barcode whitelist for ambient profiling.
    // Capped at kMaxWlAmbientBarcodes to bound hash-map and UMI-count allocations;
    // 50K barcodes is more than sufficient for a representative ambient RNA profile.
    // Builds wl_amb_index_ and sizes wl_umi_counts_.
    // wl_ambient_gene_counts_ is sized later (in run()/run_parallel()) once gene model is loaded.
    static constexpr size_t kMaxWlAmbientBarcodes = 50000u;

    bool load_whitelist() {
        LineReader f(config_.whitelist_path);
        if (!f.good()) {
            std::cerr << "[singlet-pileup] WARNING: Cannot open whitelist for ambient profiling: "
                      << config_.whitelist_path << "\n";
            return false;
        }
        std::vector<std::string> wl_barcodes;
        wl_barcodes.reserve(std::min<size_t>(kMaxWlAmbientBarcodes, 4000000u));
        std::string line;
        while (f.getline(line)) {
            if (!line.empty() && line[0] != '#') {
                auto end = line.find_first_of("\t\r\n ");
                if (end != std::string::npos) line.resize(end);
                if (!line.empty()) {
                    wl_barcodes.push_back(std::move(line));
                    if (wl_barcodes.size() >= kMaxWlAmbientBarcodes) break;
                }
            }
        }
        if (wl_barcodes.empty()) {
            std::cerr << "[singlet-pileup] WARNING: Empty whitelist — WL ambient profiling disabled\n";
            return false;
        }
        wl_amb_index_.build(wl_barcodes);
        wl_umi_counts_.assign(wl_barcodes.size(), 0u);
        std::cerr << "[singlet-pileup] WL ambient profiling: " << wl_barcodes.size()
                  << " barcodes sampled (cap=" << kMaxWlAmbientBarcodes << ")\n";
        return true;
    }



    bool load_snps() {
        LineReader f(config_.snp_path);
        if (!f.good()) {
            std::cerr << "[singlet-pileup] ERROR: Cannot open SNP file: "
                      << config_.snp_path << "\n";
            return false;
        }

        // Intern chromosome strings to avoid 7M+ allocations
        std::unordered_map<std::string, uint16_t> chrom_intern;
        std::vector<std::string> chrom_strings;

        std::string line;
        uint32_t idx = 0;
        while (f.getline(line)) {
            if (line.empty() || line[0] == '#') continue;

            // Fast hand-coded tab parser (replaces sscanf)
            const char* p = line.c_str();
            const char* end = p + line.size();

            // Field 0: CHROM
            const char* chrom_start = p;
            while (p < end && *p != '\t') ++p;
            if (p >= end) continue;
            size_t chrom_len = p - chrom_start;
            ++p; // skip tab

            // Field 1: POS
            int32_t pos = 0;
            while (p < end && *p >= '0' && *p <= '9') {
                pos = pos * 10 + (*p - '0');
                ++p;
            }
            if (p >= end || *p != '\t') continue;
            ++p; // skip tab

            // Check: does field 2 look like REF (single char + tab)?
            // If so, format is CHROM\tPOS\tREF\tALT
            // Otherwise, format is CHROM\tPOS\tID\tREF\tALT
            const char* f2_start = p;
            while (p < end && *p != '\t') ++p;
            if (p >= end) continue;
            size_t f2_len = p - f2_start;
            ++p; // skip tab

            char ref_allele, alt_allele;
            if (f2_len == 1 && (*f2_start == 'A' || *f2_start == 'C' || *f2_start == 'G' || *f2_start == 'T')) {
                // 4-field format: CHROM POS REF ALT
                ref_allele = *f2_start;
                const char* alt_start = p;
                while (p < end && *p != '\t' && *p != '\n') ++p;
                if (p - alt_start != 1) continue; // only SNVs
                alt_allele = *alt_start;
            } else {
                // 5-field format: CHROM POS ID REF ALT
                const char* ref_start = p;
                while (p < end && *p != '\t') ++p;
                if (p >= end || p - ref_start != 1) continue;
                ref_allele = *ref_start;
                ++p; // skip tab
                const char* alt_start = p;
                while (p < end && *p != '\t' && *p != '\n') ++p;
                if (p - alt_start != 1) continue;
                alt_allele = *alt_start;
            }

            // All formats use 1-based positions → convert to 0-based
            pos--;

            // Intern chromosome string
            std::string chrom_str(chrom_start, chrom_len);
            auto [it, inserted] = chrom_intern.try_emplace(chrom_str,
                static_cast<uint16_t>(chrom_strings.size()));
            if (inserted) chrom_strings.push_back(std::move(chrom_str));

            SNPSite site;
            site.index = idx;
            site.ref_allele = ref_allele;
            site.alt_allele = alt_allele;

            snp_pending_.push_back({chrom_strings[it->second], pos, site});

            // Build name: chr:pos:R>A (pos is 1-based in output)
            char name_buf[128];
            int n = snprintf(name_buf, sizeof(name_buf), "%.*s:%d:%c>%c",
                            static_cast<int>(chrom_len), chrom_start,
                            pos + 1, ref_allele, alt_allele);
            snp_names_.emplace_back(name_buf, n);
            idx++;
        }

        n_snps_loaded_ = idx;
        std::cerr << "[singlet-pileup] Loaded " << idx << " SNP sites\n";
        return true;
    }

    // Pending SNPs to be resolved against BAM header
    struct PendingSNP {
        std::string chrom;
        int32_t pos;
        SNPSite site;
    };
    std::vector<PendingSNP> snp_pending_;

    void resolve_pending_snps() {
        // Determine max tid for vector sizing
        int32_t max_tid = -1;
        for (auto& p : snp_pending_) {
            int32_t tid = resolve_chrom(p.chrom);
            if (tid > max_tid) max_tid = tid;
        }
        if (max_tid >= 0) {
            if (static_cast<size_t>(max_tid) >= chrom_snps_.size())
                chrom_snps_.resize(max_tid + 1);
        }

        uint32_t n_resolved = 0;
        for (auto& p : snp_pending_) {
            int32_t tid = resolve_chrom(p.chrom);
            if (tid >= 0) {
                chrom_snps_[tid].push_back({p.pos, p.site.index,
                                            p.site.ref_allele, p.site.alt_allele});
                n_resolved++;
            }
        }
        // Sort per-chromosome for binary search
        for (auto& snps : chrom_snps_) {
            std::sort(snps.begin(), snps.end(),
                [](const ChrSNP& a, const ChrSNP& b) { return a.pos < b.pos; });
        }
        snp_pending_.clear();
        snp_pending_.shrink_to_fit();
        std::cerr << "[singlet-pileup] Resolved " << n_resolved
                  << " SNPs to BAM chromosome IDs\n";
    }

    // ── Fast variant overlap check ──
    // Uses per-chromosome sorted vectors + binary search to determine
    // if any SNP falls within [start, end). O(log n) per read.
    bool has_variant_overlap(int32_t tid, int32_t start, int32_t end) const {
        if (static_cast<size_t>(tid) < chrom_snps_.size() && !chrom_snps_[tid].empty()) {
            const auto& snps = chrom_snps_[tid];
            auto lb = std::lower_bound(snps.begin(), snps.end(), start,
                [](const ChrSNP& s, int32_t p) { return s.pos < p; });
            if (lb != snps.end() && lb->pos < end) return true;
        }
        return false;
    }

    // ── Splice junction extraction from CIGAR ──
    // CIGAR N operations represent introns (splice junctions).
    // For each N op: donor = ref_pos (last exonic base + 1), acceptor = ref_pos + len
    void extract_splice_junctions(bam1_t* b, int32_t tid, uint32_t bc_idx,
                                  const char* umi_str, int umi_len,
                                  PileupStats& stats) {
        const uint32_t* cigar = bam_get_cigar(b);
        int32_t ref_pos = b->core.pos;
        int n_cigar = b->core.n_cigar;

        for (int ci = 0; ci < n_cigar; ++ci) {
            int op = bam_cigar_op(cigar[ci]);
            int len = bam_cigar_oplen(cigar[ci]);

            switch (op) {
                case BAM_CMATCH:
                case BAM_CEQUAL:
                case BAM_CDIFF:
                case BAM_CDEL:
                    ref_pos += len;
                    break;

                case BAM_CREF_SKIP: {
                    // This is a splice junction!
                    int32_t donor = ref_pos;
                    int32_t acceptor = ref_pos + len;

                    // Hash key for this SJ
                    uint64_t sj_key = (static_cast<uint64_t>(tid) << 32) |
                                      static_cast<uint64_t>(donor);
                    sj_key ^= static_cast<uint64_t>(acceptor) * 0x9e3779b97f4a7c15ULL;

                    auto it = sj_index_.find(sj_key);
                    uint32_t sj_idx;
                    if (it != sj_index_.end()) {
                        sj_idx = it->second;
                    } else {
                        sj_idx = static_cast<uint32_t>(sj_names_.size());
                        sj_index_[sj_key] = sj_idx;

                        // Construct name: chr:donor-acceptor
                        const std::string& chrom = tid_to_chrom_[tid];
                        std::string sj_name = chrom + ":" +
                                            std::to_string(donor) + "-" +
                                            std::to_string(acceptor);
                        sj_names_.push_back(sj_name);
                    }

                    // Lazy-init SJ accumulator on first SJ
                    if (sj_acc_.n_features() == 0 || sj_idx >= sj_acc_.n_features()) {
                        sj_acc_.set_barcodes(barcodes_);
                        sj_acc_.set_n_features(sj_idx + 1000);  // batch resize
                    }
                    // UMI dedup: exact match on (barcode, junction_id, UMI)
                    // R9: molecules are counted at finalize, after UMI correction
                    if (config_.umi_dedup) {
                        sj_umi_dedup_.add(bc_idx, sj_idx, umi_str, umi_len, cur_read_gene_);
                    } else {
                        sj_acc_.increment(sj_idx, bc_idx);
                        stats.sj_hits++;
                    }

                    ref_pos += len;
                    break;
                }

                case BAM_CINS:
                case BAM_CSOFT_CLIP:
                case BAM_CHARD_CLIP:
                case BAM_CPAD:
                    break;
            }
        }
    }

    // ── CIGAR-based pileup with binary search (replaces per-base hash lookups) ──

    void pileup_aligned_bases(bam1_t* b, int32_t tid, uint32_t bc_idx,
                              const char* umi_str, int umi_len,
                              PileupStats& stats) {
        const uint32_t* cigar = bam_get_cigar(b);
        int32_t ref_pos = b->core.pos;
        int32_t query_pos = 0;
        const uint8_t* seq = bam_get_seq(b);
        const uint8_t* qual = bam_get_qual(b);

        // Get per-chromosome sorted variant arrays (O(1) vector index)
        const std::vector<ChrSNP>* snps = nullptr;
        if (static_cast<size_t>(tid) < chrom_snps_.size() && !chrom_snps_[tid].empty())
            snps = &chrom_snps_[tid];

        for (int ci = 0; ci < b->core.n_cigar; ++ci) {
            int op = bam_cigar_op(cigar[ci]);
            int len = bam_cigar_oplen(cigar[ci]);

            switch (op) {
                case BAM_CMATCH:
                case BAM_CEQUAL:
                case BAM_CDIFF: {
                    int32_t block_end = ref_pos + len;

                    // Binary search for SNPs in [ref_pos, block_end)
                    if (snps) {
                        auto it = std::lower_bound(snps->begin(), snps->end(), ref_pos,
                            [](const ChrSNP& s, int32_t p) { return s.pos < p; });
                        for (; it != snps->end() && it->pos < block_end; ++it) {
                            int offset = it->pos - ref_pos;
                            if (qual[0] != 0xff && qual[query_pos + offset] < config_.min_baseq)
                                continue;
                            char base = seq_nt16_str[bam_seqi(seq, query_pos + offset)];
                            // one allele per molecule, by consensus at finalize (order-free-rules R1)
                            snp_cons_.add(bc_idx, it->index, config_.umi_dedup ? umi_str : nullptr, umi_len,
                                          SnpConsensus::base_class(base, it->ref_allele, it->alt_allele),
                                          cur_read_gene_);
                            stats.snp_hits++;
                        }
                    }

                    ref_pos += len;
                    query_pos += len;
                    break;
                }

                case BAM_CINS:
                case BAM_CSOFT_CLIP:
                    query_pos += len;
                    break;

                case BAM_CDEL:
                case BAM_CREF_SKIP:
                    ref_pos += len;
                    break;

                case BAM_CHARD_CLIP:
                case BAM_CPAD:
                    break;
            }
        }
    }

    // ── Inline chrM base pileup for heteroplasmy ──
    // Walk CIGAR, extract base at each reference position, increment mt_acc_
    void pileup_mt_bases(bam1_t* b, uint32_t bc_idx, PileupStats& stats) {
        const uint32_t* cigar = bam_get_cigar(b);
        int32_t ref_pos = b->core.pos;
        int32_t query_pos = 0;
        const uint8_t* seq = bam_get_seq(b);
        const uint8_t* qual = bam_get_qual(b);

        for (int ci = 0; ci < b->core.n_cigar; ++ci) {
            int op = bam_cigar_op(cigar[ci]);
            int len = bam_cigar_oplen(cigar[ci]);
            switch (op) {
                case BAM_CMATCH:
                case BAM_CEQUAL:
                case BAM_CDIFF:
                    for (int i = 0; i < len; ++i) {
                        uint32_t rp = static_cast<uint32_t>(ref_pos + i);
                        if (rp >= mt::MT_LEN) continue;
                        if (qual[0] != 0xff && qual[query_pos + i] < config_.min_baseq)
                            continue;
                        int base_code = bam_seqi(seq, query_pos + i);
                        int8_t bidx = mt::BASE_TO_IDX[base_code];
                        if (bidx < 0) continue;
                        mt_acc_.increment(mt::mt_feature(rp, bidx), bc_idx);
                        if (b->core.flag & BAM_FREVERSE)
                            mt_acc_rev_.increment(mt::mt_feature(rp, bidx), bc_idx);
                        stats.mt_pileup_bases++;
                    }
                    ref_pos += len;
                    query_pos += len;
                    break;
                case BAM_CINS:
                    if (static_cast<uint32_t>(ref_pos) < mt::MT_LEN)
                        mt_indels_.push_back({bc_idx, static_cast<uint32_t>(ref_pos), static_cast<uint16_t>(std::min(len, 0xFFFF)), true});
                    query_pos += len;
                    break;
                case BAM_CSOFT_CLIP:
                    query_pos += len;
                    break;
                case BAM_CDEL:
                    if (static_cast<uint32_t>(ref_pos) < mt::MT_LEN)
                        mt_indels_.push_back({bc_idx, static_cast<uint32_t>(ref_pos), static_cast<uint16_t>(std::min(len, 0xFFFF)), false});
                    ref_pos += len;
                    break;
                case BAM_CREF_SKIP:
                    ref_pos += len;
                    break;
                default:
                    break;
            }
        }
    }

    // Per-worker mt pileup (parallel mode)
    void pileup_mt_bases_parallel(WorkerContext& wc, bam1_t* b, uint32_t bc_idx) {
        const uint32_t* cigar = bam_get_cigar(b);
        int32_t ref_pos = b->core.pos;
        int32_t query_pos = 0;
        const uint8_t* seq = bam_get_seq(b);
        const uint8_t* qual = bam_get_qual(b);

        for (int ci = 0; ci < b->core.n_cigar; ++ci) {
            int op = bam_cigar_op(cigar[ci]);
            int len = bam_cigar_oplen(cigar[ci]);
            switch (op) {
                case BAM_CMATCH:
                case BAM_CEQUAL:
                case BAM_CDIFF:
                    for (int i = 0; i < len; ++i) {
                        uint32_t rp = static_cast<uint32_t>(ref_pos + i);
                        if (rp >= mt::MT_LEN) continue;
                        if (qual[0] != 0xff && qual[query_pos + i] < config_.min_baseq)
                            continue;
                        int base_code = bam_seqi(seq, query_pos + i);
                        int8_t bidx = mt::BASE_TO_IDX[base_code];
                        if (bidx < 0) continue;
                        wc.mt.increment(mt::mt_feature(rp, bidx), bc_idx);
                        if (b->core.flag & BAM_FREVERSE)
                            wc.mt_rev.increment(mt::mt_feature(rp, bidx), bc_idx);
                        wc.stats.mt_pileup_bases++;
                    }
                    ref_pos += len;
                    query_pos += len;
                    break;
                case BAM_CINS:
                    if (static_cast<uint32_t>(ref_pos) < mt::MT_LEN)
                        wc.mt_indels.push_back({bc_idx, static_cast<uint32_t>(ref_pos), static_cast<uint16_t>(std::min(len, 0xFFFF)), true});
                    query_pos += len;
                    break;
                case BAM_CSOFT_CLIP:
                    query_pos += len;
                    break;
                case BAM_CDEL:
                    if (static_cast<uint32_t>(ref_pos) < mt::MT_LEN)
                        wc.mt_indels.push_back({bc_idx, static_cast<uint32_t>(ref_pos), static_cast<uint16_t>(std::min(len, 0xFFFF)), false});
                    ref_pos += len;
                    break;
                case BAM_CREF_SKIP:
                    ref_pos += len;
                    break;
                default:
                    break;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Parallel helper methods (per-worker state, thread-safe)
    // ═══════════════════════════════════════════════════════════════════

    // Thread-safe gene determination using per-worker scratch vectors
    GeneHit determine_exon_gene_parallel(WorkerContext& wc, bam1_t* b, int32_t tid) {
        GeneHit result;
        if (!has_gene_model_) return result;

        wc.aligned_blocks.clear();
        const uint32_t* cigar = bam_get_cigar(b);
        int32_t rpos = b->core.pos;
        for (int ci = 0; ci < b->core.n_cigar; ++ci) {
            int op = bam_cigar_op(cigar[ci]);
            int len = bam_cigar_oplen(cigar[ci]);
            switch (op) {
                case BAM_CMATCH: case BAM_CEQUAL: case BAM_CDIFF:
                    wc.aligned_blocks.push_back({rpos, rpos + len});
                    rpos += len; break;
                case BAM_CDEL: case BAM_CREF_SKIP:
                    rpos += len; break;
                default: break;
            }
        }

        bool read_reverse = (b->core.flag & BAM_FREVERSE) != 0;
        wc.mm_overlaps.clear();
        for (const auto& [bstart, bend] : wc.aligned_blocks) {
            gene_model_.query_exons(tid, bstart, bend, wc.block_overlaps);
            for (uint32_t idx : wc.block_overlaps) {
                bool wrong_strand = false;
                if (config_.stranded) {
                    uint32_t g = gene_model_.exon_to_gene(idx);
                    char gs = gene_model_.gene_strand(g);
                    if (gs != '.') {
                        bool expect_reverse = (gs == '-');
                        if (config_.reverse_strand) expect_reverse = !expect_reverse;
                        if (read_reverse != expect_reverse) wrong_strand = true;
                    }
                }
                if (!wrong_strand)
                    wc.mm_overlaps.push_back(idx);
            }
        }

        if (wc.mm_overlaps.empty()) return result;

        uint32_t gene = gene_model_.exon_to_gene(wc.mm_overlaps[0]);
        for (size_t i = 1; i < wc.mm_overlaps.size(); ++i) {
            if (gene_model_.exon_to_gene(wc.mm_overlaps[i]) != gene) {
                result.multi_gene = true;
                return result;
            }
        }

        result.gene_idx = gene;
        result.exon_idx = wc.mm_overlaps[0];
        return result;
    }

    bool is_mt_gene(uint32_t g) const {
        const std::string& c = gene_model_.genes()[g].chrom;
        return c == "chrM" || c == "MT" || c == "chrMT";
    }

    // Resolve a completed multi-hit entry using per-worker state
    void resolve_mm_parallel(WorkerContext& wc, typename WorkerContext::MultiHitEntry& e) {
        if (e.bc_idx < 0) return;
        // A read aligning equally to a chrM gene and to a nuclear copy of it (NUMT, annotated as a
        // pseudogene) comes from the mitochondrial genome (hundreds of copies per cell vs. one nuclear
        // copy): count it for the chrM gene instead of dropping it as gene-ambiguous.
        if (mt_numt_tie(e)) {
            count_read(wc.mol_store, wc.exon, wc.intron, wc.stats, static_cast<uint32_t>(e.bc_idx),
                       mt_rescue_features(e), e.umi_len > 0 ? e.umi : nullptr, e.umi_len);
            wc.stats.mt_numt_rescued++;
            return;
        }
        if (e.gene_ambiguous) {
            // G-EM: collect for post-pileup EM rescue
            if (e.first_gene != UINT32_MAX && e.second_gene != UINT32_MAX) {
                em_rescue::AmbigRead ar;
                ar.bc_idx  = e.bc_idx;
                ar.umi_len = e.umi_len;
                std::memcpy(ar.umi, e.umi, e.umi_len);
                ar.gene_a  = e.first_gene;
                ar.gene_b  = e.second_gene;
                wc.ambig_reads.push_back(ar);
            }
            return;
        }
        if (e.first_gene == UINT32_MAX || e.gene_idx == UINT32_MAX) return;
        ReadFeatures rf{e.gene_idx, e.exon_idx, e.intron_idx, e.read_flags,
                        static_cast<uint8_t>(e.read_diag | DirectionalUmiStore::kDiagMulti)};
        count_read(wc.mol_store, wc.exon, wc.intron, wc.stats, static_cast<uint32_t>(e.bc_idx),
                   rf, e.umi_len > 0 ? e.umi : nullptr, e.umi_len);
    }

    // ── Read → (gene, spliced/unspliced evidence) ──────────────────────────
    // A read is assigned to the single gene whose exons it overlaps (exon
    // priority, like STARsolo GeneFull_Ex50pAS); failing that, to the single
    // gene whose introns it overlaps. Intron hits of other genes are ignored:
    // the pre-fix engine counted an exonic read of gene A that also lay in an
    // intron of gene B once for A and once for B.
    struct ReadFeatures {
        uint32_t gene = UINT32_MAX;
        uint32_t exon_idx = UINT32_MAX;
        uint32_t intron_idx = UINT32_MAX;
        uint8_t flags = 0;  // DirectionalUmiStore::kRead*
        uint8_t diag = 0;   // DirectionalUmiStore::kDiag*
    };
    static constexpr uint8_t kExonicReadFlags =
        DirectionalUmiStore::kReadExonOnly | DirectionalUmiStore::kReadTouchesExon;

    // intron_rule::kMtPrefer, shared by the serial and the parallel multimapper buffers (E = either MultiHitEntry):
    // remember the alignment that lies in a chrM gene ...
    template <class E>
    void note_mt_alignment(E& e, uint32_t gene, uint32_t exon, uint32_t intron, uint8_t flags, uint8_t diag) const {
        if (!(config_.intron_rules & intron_rule::kMtPrefer) || gene == UINT32_MAX || !is_mt_gene(gene)) return;
        e.mt_gene = gene; e.mt_exon = exon; e.mt_intron = intron; e.mt_flags = flags; e.mt_diag = diag;
    }
    // ... a read aligning equally to a chrM gene and a nuclear gene (NUMT) comes from the mitochondrial genome
    // (hundreds of copies per cell vs. one nuclear copy), so it is counted for the chrM gene instead of dropped
    // as gene-ambiguous.
    template <class E>
    bool mt_numt_tie(const E& e) const {
        return e.gene_ambiguous && e.mt_gene != UINT32_MAX && e.first_gene != UINT32_MAX &&
               e.second_gene != UINT32_MAX && is_mt_gene(e.first_gene) != is_mt_gene(e.second_gene);
    }
    template <class E>
    static ReadFeatures mt_rescue_features(const E& e) {
        return ReadFeatures{e.mt_gene, e.mt_exon, e.mt_intron, e.mt_flags,
                            static_cast<uint8_t>(e.mt_diag | DirectionalUmiStore::kDiagMulti)};
    }

    ReadFeatures classify_read(uint32_t exon_gene, uint32_t first_exon_idx, bool exon_multigene,
                               const std::vector<uint32_t>& intron_overlaps,
                               uint32_t intron_gene, uint32_t first_intron_idx,
                               PileupStats& st) const {
        ReadFeatures rf;
        if (exon_multigene) return rf;
        if (exon_gene != UINT32_MAX) {
            rf.gene = exon_gene;
            rf.exon_idx = first_exon_idx;
            bool other_gene_intron = false;
            for (uint32_t idx : intron_overlaps) {
                if (gene_model_.intron_to_gene(idx) == exon_gene) {
                    if (rf.intron_idx == UINT32_MAX) rf.intron_idx = idx;
                } else {
                    other_gene_intron = true;
                }
            }
            if (other_gene_intron) {
                st.cross_gene_intron_reads++;
                if (config_.intron_rules & intron_rule::kNoExonPriority) return ReadFeatures{};
                rf.diag |= DirectionalUmiStore::kDiagOtherGene;
            }
            rf.flags = (rf.intron_idx == UINT32_MAX)
                ? kExonicReadFlags
                : static_cast<uint8_t>(DirectionalUmiStore::kReadIntron |
                                       DirectionalUmiStore::kReadTouchesExon);
        } else if (intron_gene != UINT32_MAX) {
            rf.gene = intron_gene;
            rf.intron_idx = first_intron_idx;
            rf.flags = DirectionalUmiStore::kReadIntron;
        }
        return rf;
    }

    // Read diagnostics (kDiag*) and the read-level intron rules (intron_rule::k*).
    // Only reads with intronic evidence are changed: a rule either drops the
    // read (gene = none) or, for kFlank, makes a short exon overhang exonic.
    void refine_read(ReadFeatures& rf, const std::vector<std::pair<int32_t, int32_t>>& blocks,
                     bool has_n, int nh, int32_t tid, bool read_reverse,
                     std::vector<uint32_t>& scratch) const {
        if (rf.gene == UINT32_MAX) return;
        using D = DirectionalUmiStore;
        const uint32_t rules = config_.intron_rules;
        const bool intronic = (rf.flags & D::kReadIntron) != 0;
        const auto& g = gene_model_.genes()[rf.gene];
        if (nh > 1) rf.diag |= D::kDiagMulti;
        if (has_n) rf.diag |= D::kDiagSpliced;
        for (const auto& [bs, be] : blocks)
            if (bs < g.start || be > g.end) { rf.diag |= D::kDiagOutside; break; }
        if (intronic && (rf.flags & D::kReadTouchesExon)) {
            // intronic bases = bases inside the gene body minus bases in its (merged, sorted) exons
            int64_t in_body = 0, in_exon = 0;
            for (const auto& [bs, be] : blocks) {
                in_body += std::max(0, std::min(be, g.end) - std::max(bs, g.start));
                for (uint32_t ei : g.exon_indices) {
                    const int32_t es = gene_model_.exon_start(ei), ee = gene_model_.exon_end(ei);
                    if (es >= be) break;
                    in_exon += std::max(0, std::min(be, ee) - std::max(bs, es));
                }
            }
            if (in_body - in_exon <= 5) rf.diag |= D::kDiagFlank;
        }
        if (!molecule_dump_.empty()) {
            // opposite-strand genes (exons or introns) under the read: diagnostics only
            bool anti = false;
            for (const auto& [bs, be] : blocks) {
                for (int k = 0; k < 2 && !anti; ++k) {
                    if (k == 0) gene_model_.query_exons(tid, bs, be, scratch);
                    else gene_model_.query_introns(tid, bs, be, scratch);
                    for (uint32_t idx : scratch) {
                        uint32_t og = k == 0 ? gene_model_.exon_to_gene(idx) : gene_model_.intron_to_gene(idx);
                        char gs = gene_model_.gene_strand(og);
                        if (gs == '.') continue;
                        bool expect_reverse = (gs == '-');
                        if (config_.reverse_strand) expect_reverse = !expect_reverse;
                        if (read_reverse != expect_reverse) { anti = true; break; }
                    }
                }
                if (anti) break;
            }
            if (anti) rf.diag |= D::kDiagAntisense;
        }
        if (!intronic) return;
        if (((rules & intron_rule::kUnique) && nh > 1) ||
            ((rules & intron_rule::kContained) && (rf.diag & D::kDiagOutside)) ||
            ((rules & intron_rule::kNoSplicedIntron) && has_n)) {
            rf = ReadFeatures{};
            return;
        }
        if ((rules & intron_rule::kFlank) && (rf.diag & D::kDiagFlank)) {
            rf.flags = kExonicReadFlags;
            rf.intron_idx = UINT32_MAX;
        }
    }

    // Count one gene-assigned read. With a UMI the read joins its molecule in
    // `store` (labelled once at finalize); without one each read is its own
    // molecule: exonic reads go to the exon matrix, intron-touching reads to
    // the intron matrix.
    void count_read(DirectionalUmiStore& store, SparseAccumulator<uint16_t>& exon,
                    SparseAccumulator<uint16_t>& intron, PileupStats& st, uint32_t bc_idx,
                    const ReadFeatures& rf, const char* umi_str, int umi_len) {
        if (rf.gene == UINT32_MAX) return;
        const uint8_t flags = rf.flags ? rf.flags : kExonicReadFlags;
        const bool intronic = (flags & DirectionalUmiStore::kReadIntron) != 0;
        if (intronic && !config_.count_introns) return;
        if (intronic) st.intron_hits++; else st.exon_hits++;
        if (gpu_covers_) {   // W11 "on": the GPU collapses the molecules (gpu_pileup_apply)
            if (config_.umi_dedup && umi_str && umi_len > 0) st.umi_duplicate++;
            return;
        }
        if (config_.umi_dedup && umi_str && umi_len > 0) {
            store.record_read(bc_idx, rf.gene, umi_pack_2bit(umi_str, umi_len),
                              rf.exon_idx, rf.intron_idx, flags, rf.diag);
            st.umi_duplicate++;  // reads recorded; molecules are subtracted at finalize
        } else if (intronic) {
            intron.increment(rf.intron_idx, bc_idx);
        } else {
            exon.increment(rf.exon_idx, bc_idx);
        }
    }

    // Where each emitted molecule's reads go (reads_matrix_on(); else a no-op)
    auto reads_sink() const {
        const bool on = reads_matrix_on();
        return [this, on](bool intronic, uint32_t feat, uint32_t bc, uint32_t reads) {
            if (!on) return;
            (intronic ? intron_reads_acc_ : exon_reads_acc_).increment(feat, bc, reads);
            add_read_hist(bc, reads);
        };
    }

    // Label and emit every molecule of `store`; ambiguous molecules also go to
    // `amb` (per gene). Returns the label counts.
    DirectionalUmiStore::MoleculeStats finalize_store(const DirectionalUmiStore& store,
                                                      SparseAccumulator<uint16_t>& exon,
                                                      SparseAccumulator<uint16_t>& intron,
                                                      SparseAccumulator<uint16_t>& amb) const {
        const bool track_amb = config_.count_introns && has_gene_model_;
        return store.finalize_molecules(
            exon, intron,
            [&](uint32_t bc, uint32_t gene) { if (track_amb) amb.increment(gene, bc); },
            config_.umi_len, config_.umi_dedup_directional, reads_sink());
    }

    // Label and emit the molecules of several stores (one per worker; each store
    // emits into its own accumulators). With no molecule rule and no dump this is
    // finalize_store() per store. Otherwise the molecules of all stores are
    // collapsed first so a UMI can be checked across every gene of its cell
    // (genes on different chromosomes live in different workers' stores).
    struct FinalizeTarget {
        DirectionalUmiStore* store;
        SparseAccumulator<uint16_t>* exon;
        SparseAccumulator<uint16_t>* intron;
        SparseAccumulator<uint16_t>* amb;
        bool consume = false;   // give the store's molecule groups back while collapsing (nothing reads them afterwards)
    };
    DirectionalUmiStore::MoleculeStats finalize_all(const std::vector<FinalizeTarget>& targets) const {
        using D = DirectionalUmiStore;
        const uint32_t mrules = config_.intron_rules & intron_rule::kMoleculeRules;
        D::MoleculeStats st;
        if (!mrules && molecule_dump_.empty()) {
            for (const auto& t : targets) st += finalize_store(*t.store, *t.exon, *t.intron, *t.amb);
            return st;
        }
        const bool track_amb = config_.count_introns && has_gene_model_;
        std::vector<std::vector<D::Molecule>> mols(targets.size());
        for (size_t i = 0; i < targets.size(); ++i)
        {
            if (targets[i].consume) targets[i].store->consume_molecules(config_.umi_len, config_.umi_dedup_directional, mols[i]);
            else targets[i].store->collapse_molecules(config_.umi_len, config_.umi_dedup_directional, mols[i]);
        }
        fin_mark("molecules collapsed");

        // Each (bc, gene, root UMI) is one molecule, so a (bc, UMI) seen k times is in k genes.
        struct UmiKey {
            uint64_t umi; uint32_t bc;
            bool operator==(const UmiKey& o) const { return umi == o.umi && bc == o.bc; }
        };
        struct UmiKeyHash {
            size_t operator()(const UmiKey& k) const {
                uint64_t h = k.umi * 0x9e3779b97f4a7c15ULL ^ (static_cast<uint64_t>(k.bc) + 0x632be59bd9b4e019ULL);
                h ^= h >> 31; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 29;
                return static_cast<size_t>(h);
            }
        };
        // Multi-gene UMI table, in shards by barcode so that the table of one shard is all that is in memory
        // (the whole table cost ~10 GB on a deep sample) and the shards fill on several threads. A molecule's
        // verdict depends only on the molecules sharing its (bc, UMI), whichever order they arrive in.
        struct UmiGenes { uint32_t n_genes = 0, best_count = 0, best_gene = UINT32_MAX; bool tie = false; };
        std::vector<std::vector<uint8_t>> drop(targets.size());
        if (mrules) {
            for (size_t i = 0; i < targets.size(); ++i) drop[i].assign(mols[i].size(), 0);
            constexpr uint32_t kShards = 16;
            std::atomic<uint32_t> next_shard{0};
            auto shard_worker = [&]() {
                for (uint32_t sh; (sh = next_shard.fetch_add(1)) < kShards;) {
                    std::unordered_map<UmiKey, UmiGenes, UmiKeyHash> ug;
                    for (const auto& v : mols)
                        for (const auto& m : v) {
                            if (m.bc % kShards != sh) continue;
                            auto& u = ug[{m.umi, m.bc}];
                            u.n_genes++;
                            if (m.count > u.best_count) { u.best_count = m.count; u.best_gene = m.gene; u.tie = false; }
                            else if (m.count == u.best_count) u.tie = true;
                        }
                    for (size_t i = 0; i < mols.size(); ++i)
                        for (size_t k = 0; k < mols[i].size(); ++k) {
                            const auto& m = mols[i][k];
                            if (m.bc % kShards != sh) continue;
                            const auto& u = ug.find({m.umi, m.bc})->second;
                            bool d;
                            if (u.n_genes <= 1) d = false;
                            else if (mrules & intron_rule::kUmiGeneCR) d = u.tie || m.gene != u.best_gene;
                            else if (mrules & intron_rule::kUmiGeneAll) d = true;
                            else d = m.label() != D::MoleculeLabel::Spliced;   // kUmiGene
                            drop[i][k] = d ? 1 : 0;
                        }
                }
            };
            const int nt = std::max(1, std::min<int>(config_.finalize_threads, kShards));
            std::vector<std::thread> pool;
            for (int t = 1; t < nt; ++t) pool.emplace_back(shard_worker);
            shard_worker();
            for (auto& th : pool) th.join();
        }
        fin_mark("multi-gene UMI table built");

        gzFile dump = molecule_dump_.empty() ? nullptr : gzopen(molecule_dump_.c_str(), "wb1");
        if (dump) gzputs(dump, "barcode\tgene_id\tumi\tlabel\tflags\tdiag\treads\n");
        std::string umi_s(static_cast<size_t>(config_.umi_len > 0 ? config_.umi_len : 12), 'N');
        std::string line;
        for (size_t i = 0; i < targets.size(); ++i) {
            const auto& t = targets[i];
            for (size_t k = 0; k < mols[i].size(); ++k) {
                const auto& m = mols[i][k];
                const bool dropm = mrules && drop[i][k];
                if (dropm) st.dropped++;
                else D::emit_molecule(m, *t.exon, *t.intron,
                                      [&](uint32_t bc, uint32_t gene) { if (track_amb) t.amb->increment(gene, bc); }, st, reads_sink());
                if (!dump) continue;
                for (size_t k = 0; k < umi_s.size(); ++k) umi_s[k] = "ACGT"[(m.umi >> (2 * k)) & 3];
                const char lab = dropm ? 'D' : "SUA"[static_cast<int>(m.label())];
                line.clear();
                line += (m.bc < barcodes_.size() ? barcodes_[m.bc] : std::to_string(m.bc));
                line += '\t'; line += gene_model_.gene_ids()[m.gene];
                line += '\t'; line += umi_s;
                line += '\t'; line += lab;
                line += '\t'; line += std::to_string(m.flags);
                line += '\t'; line += std::to_string(m.diag);
                line += '\t'; line += std::to_string(m.count);
                line += '\n';
                gzwrite(dump, line.data(), static_cast<unsigned>(line.size()));
            }
        }
        if (dump) {
            gzclose(dump);
            std::cerr << "[singlet-pileup] molecule dump: " << molecule_dump_ << "\n";
        }
        if (mrules)
            std::cerr << "[singlet-pileup] intron rules 0x" << std::hex << config_.intron_rules << std::dec
                      << ": " << st.dropped << " molecules of multi-gene UMIs dropped\n";
        return st;
    }

    void add_molecule_stats(PileupStats& st, const DirectionalUmiStore::MoleculeStats& ms) const {
        st.molecules_spliced += ms.spliced;
        st.molecules_unspliced += ms.unspliced;
        st.molecules_ambiguous += ms.ambiguous;
        st.molecules_legacy_double += ms.legacy_double;
        st.molecules_dropped_multigene_umi += ms.dropped;
        st.intron_rules = config_.intron_rules;
        st.umi_unique = ms.total();
        // count_read() counted every recorded read in umi_duplicate
        st.umi_duplicate = st.umi_duplicate > ms.total() ? st.umi_duplicate - ms.total() : 0;
    }

    // Per-worker splice junction extraction
    void extract_splice_junctions_parallel(WorkerContext& wc, bam1_t* b,
                                           int32_t tid, uint32_t bc_idx,
                                           const char* umi_str, int umi_len) {
        const uint32_t* cigar = bam_get_cigar(b);
        int32_t ref_pos = b->core.pos;
        for (int ci = 0; ci < b->core.n_cigar; ++ci) {
            int op = bam_cigar_op(cigar[ci]);
            int len = bam_cigar_oplen(cigar[ci]);
            switch (op) {
                case BAM_CMATCH: case BAM_CEQUAL: case BAM_CDIFF: case BAM_CDEL:
                    ref_pos += len; break;
                case BAM_CREF_SKIP: {
                    int32_t donor = ref_pos;
                    int32_t acceptor = ref_pos + len;
                    uint64_t sj_key = (static_cast<uint64_t>(tid) << 32) |
                                      static_cast<uint64_t>(donor);
                    sj_key ^= static_cast<uint64_t>(acceptor) * 0x9e3779b97f4a7c15ULL;

                    auto it = wc.sj_index.find(sj_key);
                    uint32_t sj_idx;
                    if (it != wc.sj_index.end()) {
                        sj_idx = it->second;
                    } else {
                        sj_idx = static_cast<uint32_t>(wc.sj_names.size());
                        wc.sj_index[sj_key] = sj_idx;
                        const std::string& chrom = tid_to_chrom_[tid];
                        wc.sj_names.push_back(chrom + ":" +
                            std::to_string(donor) + "-" + std::to_string(acceptor));
                    }
                    if (wc.sj.n_features() == 0 || sj_idx >= wc.sj.n_features()) {
                        wc.sj.set_barcodes(barcodes_);
                        wc.sj.set_n_features(sj_idx + 1000);
                    }
                    // UMI dedup: exact match on (barcode, junction_id, UMI)
                    if (config_.umi_dedup) {                      // R9: counted at finalize
                        wc.sj_umi.add(bc_idx, sj_idx, umi_str, umi_len, wc.cur_gene);
                    } else {
                        wc.sj.increment(sj_idx, bc_idx);
                        wc.stats.sj_hits++;
                    }
                    ref_pos += len;
                    break;
                }
                default: break;
            }
        }
    }

    // Per-worker variant pileup (SNP + editing) with per-worker UMI dedup
    void pileup_aligned_bases_parallel(WorkerContext& wc, bam1_t* b,
                                       int32_t tid, uint32_t bc_idx,
                                       const char* umi_str, int umi_len) {
        const uint32_t* cigar = bam_get_cigar(b);
        int32_t ref_pos = b->core.pos;
        int32_t query_pos = 0;
        const uint8_t* seq = bam_get_seq(b);
        const uint8_t* qual = bam_get_qual(b);

        const std::vector<ChrSNP>* snps = nullptr;
        if (static_cast<size_t>(tid) < chrom_snps_.size() && !chrom_snps_[tid].empty())
            snps = &chrom_snps_[tid];

        for (int ci = 0; ci < b->core.n_cigar; ++ci) {
            int op = bam_cigar_op(cigar[ci]);
            int len = bam_cigar_oplen(cigar[ci]);
            switch (op) {
                case BAM_CMATCH: case BAM_CEQUAL: case BAM_CDIFF: {
                    int32_t block_end = ref_pos + len;
                    if (snps) {
                        auto it = std::lower_bound(snps->begin(), snps->end(), ref_pos,
                            [](const ChrSNP& s, int32_t p) { return s.pos < p; });
                        for (; it != snps->end() && it->pos < block_end; ++it) {
                            int offset = it->pos - ref_pos;
                            if (qual[0] != 0xff && qual[query_pos + offset] < config_.min_baseq)
                                continue;
                            char base = seq_nt16_str[bam_seqi(seq, query_pos + offset)];
                            // one allele per molecule, by consensus at finalize (order-free-rules R1)
                            wc.snp_cons.add(bc_idx, it->index, config_.umi_dedup ? umi_str : nullptr, umi_len,
                                            SnpConsensus::base_class(base, it->ref_allele, it->alt_allele),
                                            wc.cur_gene);
                            wc.stats.snp_hits++;
                        }
                    }
                    ref_pos += len;
                    query_pos += len;
                    break;
                }
                case BAM_CINS: case BAM_CSOFT_CLIP:
                    query_pos += len; break;
                case BAM_CDEL: case BAM_CREF_SKIP:
                    ref_pos += len; break;
                default: break;
            }
        }
    }

    // Process a single BAM read using per-worker state (thread-safe)
    void process_read_parallel(WorkerContext& wc, bam1_t* b,
                               int32_t /*region_tid*/, sam_hdr_t* /*hdr*/) {
        wc.stats.total_reads++;
        if (b->core.flag & BAM_FUNMAP) return;
        wc.stats.mapped_reads++;

        // G-RRNA: rRNA contamination on a 1/64 sample fixed by the read name, once per read
        // (order-free-rules R4)
        if (b->core.l_qseq >= 21 && !(b->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY))
            && (hash_qname(bam_get_qname(b)) & 63) == 0) {
            ++wc.stats.rrna_reads_sampled;
            const uint8_t* qseq = bam_get_seq(b);
            if (qseq) {
                std::string _rrna_seq(static_cast<size_t>(b->core.l_qseq), 'N');
                for (int _qi = 0; _qi < b->core.l_qseq; ++_qi)
                    _rrna_seq[static_cast<size_t>(_qi)] = seq_nt16_str[bam_seqi(qseq, _qi)];
                if (wc.rrna_detector.detect(_rrna_seq)) ++wc.stats.rrna_reads;
            }
        }

        if (b->core.flag & BAM_FSUPPLEMENTARY) return;

        int32_t tid = b->core.tid;
        bool is_secondary = (b->core.flag & BAM_FSECONDARY) != 0;

        // ── Secondary multi-mapped alignment ──
        if (is_secondary) {
            wc.stats.secondary_reads++;
            if (has_gene_model_) {
                uint8_t* cb_sec = bam_aux_get(b, "CB");
                if (cb_sec) {
                    const char* cb_sec_str = bam_aux2Z(cb_sec);
                    if (cb_sec_str[0] != '-') {
                        int32_t bc_check = bc_index_.lookup(cb_sec_str);
                        if (bc_check >= 0) {
                            int nh_sec = 1;
                            uint8_t* nh_tag = bam_aux_get(b, "NH");
                            if (nh_tag) { nh_sec = bam_aux2i(nh_tag); if (nh_sec < 1) nh_sec = 1; }
                            if (nh_sec > 1) {
                                uint64_t qh = hash_qname(bam_get_qname(b));
                                auto it = wc.mm_buffer.find(qh);
                                if (it != wc.mm_buffer.end()) {
                                    auto hit = determine_exon_gene_parallel(wc, b, tid);
                                    it->second.seen++;
                                    if (hit.multi_gene) {
                                        it->second.gene_ambiguous = true;
                                    } else if (hit.gene_idx != UINT32_MAX) {
                                        if (it->second.first_gene == UINT32_MAX)
                                            it->second.first_gene = hit.gene_idx;
                                        else if (hit.gene_idx != it->second.first_gene) {
                                            it->second.gene_ambiguous = true;
                                            // G-EM: record second candidate for rescue
                                            if (it->second.second_gene == UINT32_MAX)
                                                it->second.second_gene = hit.gene_idx;
                                        }
                                        note_mt_alignment(it->second, hit.gene_idx, hit.exon_idx, UINT32_MAX, kExonicReadFlags, 0);
                                        // Without a primary assignment, the smallest exon over the
                                        // secondaries, not the first seen (order-free-rules R7)
                                        if (!it->second.primary_assigned &&
                                            (it->second.gene_idx == UINT32_MAX ||
                                             hit.exon_idx < it->second.exon_idx)) {
                                            it->second.gene_idx = hit.gene_idx;
                                            it->second.exon_idx = hit.exon_idx;
                                            it->second.read_flags = kExonicReadFlags;
                                        }
                                    }
                                    if (it->second.seen >= it->second.nh && it->second.nh > 0) {
                                        resolve_mm_parallel(wc, it->second);
                                        wc.mm_buffer.erase(it);
                                    }
                                } else {
                                    auto hit = determine_exon_gene_parallel(wc, b, tid);
                                    auto& entry = wc.mm_buffer[qh];
                                    entry.seen++;
                                    if (entry.nh == 0) entry.nh = static_cast<uint8_t>(std::min(nh_sec, 255));
                                    if (hit.multi_gene) {
                                        entry.gene_ambiguous = true;
                                    } else if (hit.gene_idx != UINT32_MAX) {
                                        entry.first_gene = hit.gene_idx;
                                        entry.gene_idx = hit.gene_idx;
                                        entry.exon_idx = hit.exon_idx;
                                        entry.read_flags = kExonicReadFlags;
                                        note_mt_alignment(entry, hit.gene_idx, hit.exon_idx, UINT32_MAX, kExonicReadFlags, 0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return;
        }

        // ── Primary alignment ──
        if (is_chrm(tid)) {
            if (!config_.out_chrm_bam.empty()) wc.chrm_buffer.push_back(bam_dup1(b));
            wc.stats.chrm_reads++;
        }

        uint8_t* cb_tag = bam_aux_get(b, "CB");
        if (!cb_tag) {
            // N22-CR: No CB tag — read didn't match the auto-discovered WL.
            // In CB_samTagOut mode STAR still writes the raw barcode to CR.
            // Use CR for WL ambient profiling so reads from full-WL barcodes
            // (outside the auto_barcodes.tsv) contribute to the ambient profile.
            if (!wc.wl_ambient_genes.empty() && has_gene_model_ &&
                b->core.qual >= static_cast<uint8_t>(config_.min_mapq)) {
                uint8_t* cr_tag = bam_aux_get(b, "CR");
                if (cr_tag) {
                    const char* cr_str = bam_aux2Z(cr_tag);
                    int32_t wl_idx = wl_amb_index_.lookup(cr_str);
                    if (wl_idx >= 0) {
                        auto hit = determine_exon_gene_parallel(wc, b, tid);
                        if (hit.gene_idx != UINT32_MAX && !hit.multi_gene)
                            wc.wl_ambient_genes[hit.gene_idx]++;
                    }
                }
            }
            wc.stats.no_barcode++; return;
        }
        const char* cb_str = bam_aux2Z(cb_tag);
        if (cb_check_ && !(b->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY))) check_cb_against_star(b, cb_str);
        if (test_drop_read(b)) return;
        int32_t bc_idx = bc_index_.lookup(cb_str);
        wc.cur_gene = UINT32_MAX;
        if (bc_idx < 0) {
            // N22: WL ambient profiling (parallel path) — CB present but not in
            // the auto-discovered barcode set.  Try the full WL ambient index.
            if (!wc.wl_ambient_genes.empty() &&
                b->core.qual >= static_cast<uint8_t>(config_.min_mapq)) {
                int32_t wl_idx = wl_amb_index_.lookup(cb_str);
                if (wl_idx >= 0 && has_gene_model_) {
                    auto hit = determine_exon_gene_parallel(wc, b, tid);
                    if (hit.gene_idx != UINT32_MAX && !hit.multi_gene)
                        wc.wl_ambient_genes[hit.gene_idx]++;
                }
            }
            return;
        }
        wc.stats.barcoded_reads++;
        if (static_cast<uint32_t>(bc_idx) < wc.cell_reads.size())
            wc.cell_reads[static_cast<uint32_t>(bc_idx)]++;  // N20

        int nh = 1;
        { uint8_t* nh_tag = bam_aux_get(b, "NH");
          if (nh_tag) { nh = bam_aux2i(nh_tag); if (nh < 1) nh = 1; } }
        if (nh > 1) wc.stats.multimapper_reads++;
        if (static_cast<uint32_t>(bc_idx) < wc.cell_qc.size())
            add_cell_align_qc(wc.cell_qc[static_cast<uint32_t>(bc_idx)], b, nh);

        bool high_mapq = (b->core.qual >= static_cast<uint8_t>(config_.min_mapq));
        if (!high_mapq && nh == 1) { wc.stats.low_mapq++; return; }
        if (!high_mapq) wc.stats.low_mapq++;

        const char* umi_str = nullptr;
        int umi_len = 0;
        if (config_.umi_dedup) {
            uint8_t* ub_tag = bam_aux_get(b, "UB");
            if (!ub_tag) ub_tag = bam_aux_get(b, "UR");
            if (ub_tag) {
                umi_str = bam_aux2Z(ub_tag);
                umi_len = static_cast<int>(strlen(umi_str));
            } else {
                wc.stats.no_umi++;
            }
        }

        // N18: CRISPR guide counting (parallel path)
        if (!guide_ref_.empty()) {
            const uint8_t* qseq = bam_get_seq(b);
            int qlen = b->core.l_qseq;
            if (qseq && qlen > 0) {
                wc.guide_seq_buf.resize(static_cast<size_t>(qlen));
                for (int qi = 0; qi < qlen; ++qi)
                    wc.guide_seq_buf[qi] = seq_nt16_str[bam_seqi(qseq, qi)];
                int gid = guide_ref_.match_bam(wc.guide_seq_buf.data(), qlen, (b->core.flag & BAM_FREVERSE) != 0);
                if (gid >= 0) {
                    bool should_count = true;
                    if (umi_len > 0 && umi_str)
                        should_count = wc.guide_umi.insert(
                            static_cast<uint32_t>(bc_idx),
                            static_cast<uint32_t>(gid), umi_str, umi_len);
                    if (should_count)
                        wc.guide.increment(static_cast<uint32_t>(gid),
                                           static_cast<uint32_t>(bc_idx));
                }
            }
        }

        // ── Exon + Intron counting ──
        if (has_gene_model_) {
            wc.aligned_blocks.clear();
            bool read_has_n = false;
            {
                const uint32_t* cigar = bam_get_cigar(b);
                int32_t rpos = b->core.pos;
                for (int ci = 0; ci < b->core.n_cigar; ++ci) {
                    int op = bam_cigar_op(cigar[ci]);
                    int len = bam_cigar_oplen(cigar[ci]);
                    switch (op) {
                        case BAM_CMATCH: case BAM_CEQUAL: case BAM_CDIFF:
                            wc.aligned_blocks.push_back({rpos, rpos + len});
                            rpos += len; break;
                        case BAM_CREF_SKIP:
                            read_has_n = true;
                            rpos += len; break;
                        case BAM_CDEL:
                            rpos += len; break;
                        default: break;
                    }
                }
            }

            wc.exon_overlaps.clear();
            bool read_reverse = (b->core.flag & BAM_FREVERSE) != 0;
            bool had_strand_reject = false;
            bool vote_sense = false, vote_anti = false;

            for (const auto& [bstart, bend] : wc.aligned_blocks) {
                gene_model_.query_exons(tid, bstart, bend, wc.block_overlaps);
                strand_vote(wc.block_overlaps, read_reverse, vote_sense, vote_anti);
                size_t n_before = wc.exon_overlaps.size();
                for (uint32_t idx : wc.block_overlaps) {
                    bool wrong_strand = false;
                    if (config_.stranded) {
                        uint32_t g = gene_model_.exon_to_gene(idx);
                        char gs = gene_model_.gene_strand(g);
                        if (gs != '.') {
                            bool expect_reverse = (gs == '-');
                            if (config_.reverse_strand) expect_reverse = !expect_reverse;
                            if (read_reverse != expect_reverse) wrong_strand = true;
                        }
                    }
                    if (!wrong_strand)
                        wc.exon_overlaps.push_back(idx);
                }
                if (!wc.block_overlaps.empty() && wc.exon_overlaps.size() == n_before)
                    had_strand_reject = true;
            }

            if (wc.exon_overlaps.empty() && had_strand_reject) wc.stats.wrong_strand++;
            if (vote_sense != vote_anti && !mito_tid_[tid] && !(b->core.flag & BAM_FSECONDARY))
                (vote_sense ? wc.stats.strand_sense : wc.stats.strand_antisense)++;

            uint32_t exon_gene = UINT32_MAX;
            uint32_t first_exon_idx = UINT32_MAX;
            bool primary_multigene = false;
            if (!wc.exon_overlaps.empty()) {
                uint32_t first_gene = gene_model_.exon_to_gene(wc.exon_overlaps[0]);
                bool unique_gene = (first_gene != UINT32_MAX);
                for (size_t i = 1; i < wc.exon_overlaps.size() && unique_gene; ++i) {
                    if (gene_model_.exon_to_gene(wc.exon_overlaps[i]) != first_gene)
                        unique_gene = false;
                }
                if (unique_gene) {
                    exon_gene = first_gene;
                    first_exon_idx = wc.exon_overlaps[0];
                } else {
                    wc.stats.multigene_reads++;
                    primary_multigene = true;
                }
            }

            uint32_t intron_gene = UINT32_MAX;
            uint32_t first_intron_idx = UINT32_MAX;
            if (config_.count_introns) {
                wc.intron_overlaps.clear();
                for (const auto& [bstart, bend] : wc.aligned_blocks) {
                    gene_model_.query_introns(tid, bstart, bend, wc.block_overlaps);
                    for (uint32_t idx : wc.block_overlaps) {
                        if (config_.stranded) {
                            uint32_t g = gene_model_.intron_to_gene(idx);
                            char gs = gene_model_.gene_strand(g);
                            if (gs != '.') {
                                bool expect_reverse = (gs == '-');
                                if (config_.reverse_strand) expect_reverse = !expect_reverse;
                                if (read_reverse != expect_reverse) continue;
                            }
                        }
                        wc.intron_overlaps.push_back(idx);
                    }
                }
                if (!wc.intron_overlaps.empty()) {
                    uint32_t first_gene = gene_model_.intron_to_gene(wc.intron_overlaps[0]);
                    bool unique_gene = (first_gene != UINT32_MAX);
                    for (size_t i = 1; i < wc.intron_overlaps.size() && unique_gene; ++i) {
                        if (gene_model_.intron_to_gene(wc.intron_overlaps[i]) != first_gene)
                            unique_gene = false;
                    }
                    if (unique_gene) {
                        intron_gene = first_gene;
                        first_intron_idx = wc.intron_overlaps[0];
                    }
                }
            }

            ReadFeatures rf = classify_read(exon_gene, first_exon_idx, primary_multigene,
                                            wc.intron_overlaps, intron_gene, first_intron_idx,
                                            wc.stats);
            refine_read(rf, wc.aligned_blocks, read_has_n, nh, tid, read_reverse, wc.block_overlaps);
            if (nh == 1) wc.cur_gene = rf.gene;   // R9: evidence UMIs are corrected within (cell, gene)
            if (nh == 1 && config_.count_polya && !config_.reverse_strand && rf.gene != UINT32_MAX) {
                uint64_t pk;
                if (polya_site_from_read(b, 6, 0.8, pk))
                    wc.polya.push_back({pk, rf.gene, static_cast<uint32_t>(bc_idx),
                                        polya_umi_hash(umi_str, umi_len)});
            }
            if (nh == 1) {
                count_read(wc.mol_store, wc.exon, wc.intron, wc.stats,
                           static_cast<uint32_t>(bc_idx), rf, umi_str, umi_len);
            } else {
                uint64_t qh = hash_qname(bam_get_qname(b));
                auto& entry = wc.mm_buffer[qh];
                entry.nh = static_cast<uint8_t>(std::min(nh, 255));
                entry.seen++;
                entry.bc_idx = bc_idx;
                if (rf.gene != UINT32_MAX) {
                    entry.primary_assigned = true;
                    entry.gene_idx = rf.gene;
                    entry.exon_idx = rf.exon_idx;
                    entry.intron_idx = rf.intron_idx;
                    entry.read_flags = rf.flags;
                    entry.read_diag = rf.diag;
                    note_mt_alignment(entry, rf.gene, rf.exon_idx, rf.intron_idx, rf.flags, rf.diag);
                }
                if (umi_str && umi_len > 0) {
                    int clen = std::min(umi_len, 15);
                    std::memcpy(entry.umi, umi_str, clen);
                    entry.umi[clen] = '\0';
                    entry.umi_len = static_cast<uint8_t>(umi_len);
                }
                if (primary_multigene) {
                    entry.gene_ambiguous = true;
                    // G-EM: extract two candidate genes from wc.exon_overlaps for rescue
                    if (entry.second_gene == UINT32_MAX && !wc.exon_overlaps.empty()) {
                        uint32_t ga = gene_model_.exon_to_gene(wc.exon_overlaps[0]);
                        if (ga != UINT32_MAX) {
                            if (entry.first_gene == UINT32_MAX) entry.first_gene = ga;
                            for (size_t _i = 1; _i < wc.exon_overlaps.size(); ++_i) {
                                uint32_t gb = gene_model_.exon_to_gene(wc.exon_overlaps[_i]);
                                if (gb != UINT32_MAX && gb != ga) {
                                    entry.second_gene = gb; break;
                                }
                            }
                        }
                    }
                }

                uint32_t aln_gene = (exon_gene != UINT32_MAX) ? exon_gene : intron_gene;
                if (aln_gene != UINT32_MAX) {
                    if (entry.first_gene == UINT32_MAX) entry.first_gene = aln_gene;
                    else if (aln_gene != entry.first_gene) {
                        entry.gene_ambiguous = true;
                        // G-EM: aln_gene is the conflicting second candidate
                        if (entry.second_gene == UINT32_MAX)
                            entry.second_gene = aln_gene;
                    }
                }

                if (entry.seen >= entry.nh && entry.nh > 0) {
                    resolve_mm_parallel(wc, entry);
                    wc.mm_buffer.erase(qh);
                }
            }

            if (config_.count_sj) {
                extract_splice_junctions_parallel(wc, b, tid, static_cast<uint32_t>(bc_idx), umi_str, umi_len);
            }

            // N17: VDJ gene usage counting (parallel path)
            if (has_vdj_ && !wc.aligned_blocks.empty()) {
                wc.vdj_overlaps.clear();
                int32_t vdj_rs = wc.aligned_blocks.front().first;
                int32_t vdj_re = wc.aligned_blocks.back().second;
                vdj_model_.query(tid, vdj_rs, vdj_re, wc.vdj_overlaps);
                if (!wc.vdj_overlaps.empty()) {
                    uint32_t vdj_gene = wc.vdj_overlaps[0];
                    if (config_.umi_dedup) {                      // R9: counted at finalize
                        wc.vdj_umi.add(static_cast<uint32_t>(bc_idx), vdj_gene, umi_str, umi_len, wc.cur_gene);
                    } else {
                        wc.vdj.increment(vdj_gene, static_cast<uint32_t>(bc_idx));
                        wc.stats.vdj_hits++;
                    }
                }
            }
        }

        // ── Variant pileup ──
        if (high_mapq) {
            int32_t end_pos_v = bam_endpos(b);
            if (has_variant_overlap(tid, b->core.pos, end_pos_v)) {
                pileup_aligned_bases_parallel(wc, b, tid, static_cast<uint32_t>(bc_idx),
                                              umi_str, umi_len);
            }
        }

        // ── Inline chrM base pileup for heteroplasmy ──
        // MAPQ gate: multimapped reads (NH>1, low MAPQ; e.g. NUMT-derived) reach this point, so require high_mapq here.
        if (config_.count_mt && high_mapq && is_chrm(tid)) {
            pileup_mt_bases_parallel(wc, b, static_cast<uint32_t>(bc_idx));
        }
    }

    // ── Merge helpers ──

    template <typename T, typename Fn>
    void merge_accumulators(SparseAccumulator<T>& target,
                            std::vector<WorkerContext>& workers, Fn&& get) {
        target.set_barcodes(barcodes_);
        uint32_t max_feat = 0;
        for (auto& w : workers) {
            uint32_t f = get(w).n_features();
            if (f > max_feat) max_feat = f;
        }
        target.set_n_features(max_feat);
        for (auto& w : workers) {
            target.merge_from(get(w));
        }
    }

    // R9: an evidence UMI's root within its (cell, gene) group of the molecule store, as the gene counts collapse it
    struct GeneRoots {
        const DirectionalUmiStore* genes;
        bool directional;
        bool operator()(uint32_t bc, uint32_t gene, uint64_t umi3, uint64_t& root) const {
            int len = 0;
            const uint64_t p2 = umi3_to_2bit(umi3, &len);
            return genes->gene_root(bc, gene, p2, len, directional, root);
        }
    };
    GeneRoots gene_roots(const DirectionalUmiStore& genes) const {
        return GeneRoots{&genes, config_.umi_dedup_directional};
    }

    // R9: emit the molecules of one feature kind into its accumulator and add them to the hit counter
    template <typename AccT>
    void finalize_umi_features(UmiFeatureCollapse& store, AccT& acc, uint64_t& hits, const DirectionalUmiStore& genes) {
        if (store.empty()) return;
        hits += store.finalize([&](uint32_t bc, uint32_t f, uint32_t n) { acc.increment(f, bc, n); },
                               config_.umi_dedup_directional, gene_roots(genes));
        store.clear();
    }

    // Canonical junction order (order-free-rules R7): rows sorted by (contig index, donor, acceptor) instead of
    // first seen, so sj_counts / sj_features / splice_* do not depend on read order or worker count.
    void canonicalize_splice_junctions() {
        const size_t n = sj_names_.size();
        // The serial pass grows sj_acc_ in batches of 1000 rows; trim to the named junctions so the matrix
        // has exactly one row per name (a padded sj_counts.1pz fails to read: rownames != rows)
        if (n > 0) sj_acc_.set_n_features(static_cast<uint32_t>(n));
        if (n < 2) return;
        std::unordered_map<std::string, int32_t> chrom_tid;
        for (size_t t = 0; t < tid_to_chrom_.size(); ++t) chrom_tid.emplace(tid_to_chrom_[t], static_cast<int32_t>(t));
        struct Key { int32_t tid; int64_t donor; int64_t acceptor; uint32_t old; };
        std::vector<Key> keys(n);
        for (size_t i = 0; i < n; ++i) {
            const std::string& s = sj_names_[i];      // chrom:donor-acceptor (chrom may contain ':' or '-')
            const size_t c = s.rfind(':'), d = s.rfind('-');
            auto it = chrom_tid.find(s.substr(0, c));
            keys[i] = {it == chrom_tid.end() ? INT32_MAX : it->second, std::stoll(s.substr(c + 1, d - c - 1)),
                       std::stoll(s.substr(d + 1)), static_cast<uint32_t>(i)};
        }
        std::sort(keys.begin(), keys.end(), [](const Key& a, const Key& b) {
            if (a.tid != b.tid) return a.tid < b.tid;
            if (a.donor != b.donor) return a.donor < b.donor;
            return a.acceptor < b.acceptor;
        });
        std::vector<uint32_t> new_idx(n);
        std::vector<std::string> names(n);
        for (size_t r = 0; r < n; ++r) {
            new_idx[keys[r].old] = static_cast<uint32_t>(r);
            names[r] = std::move(sj_names_[keys[r].old]);
        }
        sj_names_ = std::move(names);
        for (auto& kv : sj_index_) kv.second = new_idx[kv.second];
        sj_acc_.remap_rows(new_idx);
    }

    void merge_splice_junctions(std::vector<WorkerContext>& workers) {
        // Build global SJ name → index mapping from all workers.
        // Each worker had local SJ indices; we remap into a global set.
        sj_names_.clear();
        sj_index_.clear();

        // First pass: collect all unique SJ names and build global indices
        std::unordered_map<std::string, uint32_t> name_to_global;
        for (auto& wc : workers) {
            for (auto& name : wc.sj_names) {
                auto [it, inserted] = name_to_global.try_emplace(
                    name, static_cast<uint32_t>(sj_names_.size()));
                if (inserted) {
                    sj_names_.push_back(name);
                }
            }
        }

        // Initialize global SJ accumulator
        uint32_t n_sj = static_cast<uint32_t>(sj_names_.size());
        if (n_sj == 0) return;
        sj_acc_.set_barcodes(barcodes_);
        sj_acc_.set_n_features(n_sj);

        // Second pass: re-insert each worker's SJ COO data with remapped indices
        for (auto& wc : workers) {
            if (wc.sj.nnz_raw() == 0) continue;

            // Build local → global remap for this worker
            std::vector<uint32_t> remap(wc.sj_names.size());
            for (size_t i = 0; i < wc.sj_names.size(); ++i) {
                remap[i] = name_to_global[wc.sj_names[i]];
            }

            // Extract and remap: convert to CSC then re-add as COO
            // This is simpler than exposing raw COO vectors
            auto csc = wc.sj.to_csc();
            for (uint32_t col = 0; col < csc.ncols; ++col) {
                for (int32_t j = csc.indptr[col]; j < csc.indptr[col + 1]; ++j) {
                    uint32_t local_row = static_cast<uint32_t>(csc.indices[j]);
                    uint32_t global_row = (local_row < remap.size())
                                          ? remap[local_row] : local_row;
                    for (uint16_t v = 0; v < csc.data[j]; ++v) {
                        sj_acc_.increment(global_row, col);
                    }
                }
            }
            wc.sj.clear();
        }
    }

public:
    void resolve_references() {
        resolve_pending_snps();
        if (has_gene_model_) {
            gene_model_.resolve_chroms(chrom_to_tid_);
        }
        if (has_vdj_) {
            vdj_model_.resolve_chroms(chrom_to_tid_);
        }
    }
};

/// Gene-level layers of a finished pass, one line per (barcode, gene) with any count:
/// barcode, gene_id, exon (spliced + ambiguous), intron (unspliced), ambiguous. Barcodes with
/// fewer than `min_umi` molecules in total are left out. Used by the SINGLET_INTRON_SWEEP
/// experiment (src/pipeline/singlet.cpp) to compare intron rules on one alignment.
inline bool write_gene_layers_tsv(const PileupEngine& e, const std::string& path, uint32_t min_umi = 100) {
    const auto& gm = e.gene_model();
    const uint32_t ng = gm.n_genes();
    auto ex = e.exons().to_csc();
    auto in = e.introns().to_csc();
    auto am = e.ambiguous().to_csc();
    gzFile f = gzopen(path.c_str(), "wb1");
    if (!f) return false;
    gzputs(f, "barcode\tgene_id\texon\tintron\tambiguous\n");
    std::vector<uint32_t> ge(ng), gi(ng), ga(ng);
    std::vector<uint32_t> touched;
    const auto& bcs = e.barcodes();
    std::string line;
    for (uint32_t c = 0; c < ex.ncols; ++c) {
        touched.clear();
        uint64_t tot = 0;
        auto add = [&](const auto& m, std::vector<uint32_t>& v, bool exon_rows, bool intron_rows) {
            if (c >= m.ncols) return;
            for (int32_t k = m.indptr[c]; k < m.indptr[c + 1]; ++k) {
                uint32_t r = static_cast<uint32_t>(m.indices[k]);
                uint32_t g = exon_rows ? gm.exon_to_gene(r) : intron_rows ? gm.intron_to_gene(r) : r;
                if (g >= ng) continue;
                if (!ge[g] && !gi[g] && !ga[g]) touched.push_back(g);
                v[g] += m.data[k];
            }
        };
        add(ex, ge, true, false);
        add(in, gi, false, true);
        add(am, ga, false, false);
        for (uint32_t g : touched) tot += ge[g] + gi[g];
        if (tot >= min_umi && c < bcs.size()) {
            std::sort(touched.begin(), touched.end());
            for (uint32_t g : touched) {
                line.clear();
                line += bcs[c]; line += '\t'; line += gm.gene_ids()[g];
                line += '\t'; line += std::to_string(ge[g]);
                line += '\t'; line += std::to_string(gi[g]);
                line += '\t'; line += std::to_string(ga[g]);
                line += '\n';
                gzwrite(f, line.data(), static_cast<unsigned>(line.size()));
            }
        }
        for (uint32_t g : touched) ge[g] = gi[g] = ga[g] = 0;
    }
    gzclose(f);
    return true;
}

/// Post-pileup strand safety net. If most nuclear strand-informative reads of a
/// finished pass are antisense to the orientation it counted with
/// (counted_antisense_frac > kStrandRetryFrac), the probe missed: rebuild the engine from `base` with the opposite orientation of
/// the pass, auto-strand off, and run it again with `run(PileupEngine&)`. The input
/// must be re-readable (a BAM file, not a pipe). Returns true if the pass was
/// re-run; `engine`, `stats` and base.reverse_strand then describe the retry. Off
/// when base.auto_strand is false (orientation forced by the user, or already retried). The
/// old engine is kept if the new one cannot load its references.
template <class RunFn>
inline bool rerun_if_wrong_strand(std::unique_ptr<PileupEngine>& engine, PileupConfig& base,
                                  PileupStats& stats, RunFn&& run) {
    if (!engine || !base.stranded || !base.auto_strand || base.exon_gtf_path.empty()) return false;
    if (stats.strand_sense + stats.strand_antisense < kStrandRetryMinReads) return false;
    double before = counted_antisense_frac(stats, engine->reverse_strand());
    if (before <= kStrandRetryFrac) return false;
    PileupConfig cfg = base;
    cfg.reverse_strand = !engine->reverse_strand();
    cfg.auto_strand = false;
    std::cerr << "[singlet-pileup] Strand safety net: " << static_cast<int>(before * 100)
              << "% of nuclear reads antisense to the counted strand -> re-running " << (cfg.reverse_strand ? "reverse" : "forward")
              << "-strand\n";
    auto next = std::make_unique<PileupEngine>(cfg);
    if (!next->load_references()) {
        std::cerr << "[singlet-pileup] WARNING: could not reload references for the strand retry; "
                     "keeping the first pass\n";
        return false;
    }
    next->keep_strand_probe(*engine);
    engine.reset();
    engine = std::move(next);
    stats = run(*engine);
    base.reverse_strand = cfg.reverse_strand;
    base.auto_strand = false;
    std::cerr << "[singlet-pileup] Strand retry: "
              << static_cast<int>(counted_antisense_frac(stats, cfg.reverse_strand) * 100)
              << "% of nuclear reads antisense to the counted strand\n";
    return true;
}

} // namespace singlet
