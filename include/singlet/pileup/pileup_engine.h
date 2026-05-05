#pragma once
// singlet-pileup: pileup_engine.h
// Core streaming BAM pileup engine for Pipeline V2.
// Reads BAM records (sorted or unsorted, file or stdin) and accumulates:
// 1. Per-exon counts (UMI-deduplicated) → gene-level derivation
// 2. Per-intron counts (UMI-deduplicated) → GeneFull = exon + intron
// 3. Splice junction counts from CIGAR N operations
// 4. SNP allele counts (AD/DP) at known positions (raw reads, no UMI dedup)
// 5. RNA editing (A/G) counts at REDIportal sites
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
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ancestry.h"
#include "crispr_guide.h"
#include "gene_model.h"
#include "interval_tree.h"
#include "mt_heteroplasmy.h"
#include "em_rescue.h"
#include "sparse_accumulator.h"
#include "umi_dedup.h"
#include "vdj_counter.h"
#include "rrna_detect.h"

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
    std::string snp_path;           // VCF/TSV of SNP positions
    std::string exon_gtf_path;      // GTF/BED for exon intervals (used for GeneModel)
    std::string barcode_path;       // Filtered barcode list
    std::string out_chrm_bam;       // Output: chrM BAM
    int threads = 4;
    int min_mapq = 20;              // Minimum mapping quality
    int min_baseq = 10;             // Minimum base quality for SNP/editing
    bool count_introns = true;      // Count intronic reads
    bool count_sj = true;           // Extract splice junctions
    bool umi_dedup = true;          // UMI deduplication for exon/intron
    bool umi_dedup_directional = false; // N6: directional 1-Hamming error correction (run() and run_parallel())
    int umi_len = 12;               // UMI length in bases (for N6 directional Hamming)
    bool stranded = true;           // Strand-aware counting (10x Forward)
    bool reverse_strand = false;    // Invert strand filter (for 5' protocols)
    bool count_mt = false;          // Inline chrM allele pileup for heteroplasmy
    bool aim_ancestry = true;          // N13: inject built-in AIM panel for ancestry classification
    std::string guide_ref_path;         // N18: CRISPR guide CSV (name,sequence); empty = disabled
    bool count_vdj = true;              // N17: V(D)J gene usage counting; requires exon_gtf_path
    // N22: Full-whitelist ambient profiling — gives EmptyDrops a representative empty-droplet pool
    std::string whitelist_path;         // Full barcode whitelist (e.g. 3M-february-2018.txt); empty = disabled
    uint32_t wl_ambient_ceil = 50;      // UMI ceiling: WL barcodes with ≤ this count define the ambient pool
};

// Statistics collected during pileup
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
    uint64_t vdj_hits = 0;
    // G-RRNA: rRNA contamination sampling
    uint64_t rrna_reads = 0;          ///< reads identified as rRNA (sampled subset)
    uint64_t rrna_reads_sampled = 0;  ///< total reads checked for rRNA
    double wall_time_s = 0;
};

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

class PileupEngine {
public:
    explicit PileupEngine(const PileupConfig& config) : config_(config) {}

    // Reload only the barcode list from a new file, keeping all other references
    // (gene model, SNPs) intact. Used to refresh barcodes from STAR CB tags
    // after alignment, replacing raw decode-time barcodes with whitelist-corrected ones.
    bool reload_barcodes(const std::string& path) {
        config_.barcode_path = path;
        barcodes_.clear();
        bc_index_.map.clear();
        per_cell_reads_.clear();
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
        }

        // N22: Initialize WL ambient gene profile (sized after gene model is loaded)
        if (!wl_umi_counts_.empty() && has_gene_model_) {
            wl_ambient_gene_counts_.assign(gene_model_.n_genes(), 0ULL);
        }

        // Initialize mt allele accumulator (pos*4 + base_idx, 16569*4 = 66276 features)
        if (config_.count_mt) {
            mt_acc_.set_barcodes(barcodes_);
            mt_acc_.set_n_features(mt::MT_N_FEATURES);
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

        // Reserve UMI dedup capacity
        if (config_.umi_dedup) {
            umi_dedup_.reserve(5000000);  // ~5M unique UMIs typical
        }

        // ── Multi-mapper gene-uniqueness buffer ──
        // STARsolo counts multi-mapped reads (NH > 1) if ALL alignments
        // map to the same gene ("gene-unique" multi-mappers). To replicate
        // this, we buffer multi-mapped reads and resolve across all alignments.
        struct MultiHitEntry {
            int32_t bc_idx = -1;
            uint32_t exon_idx = UINT32_MAX;     // primary alignment exon
            uint32_t gene_idx = UINT32_MAX;     // primary gene
            uint32_t intron_idx = UINT32_MAX;   // primary intron
            uint32_t intron_gene = UINT32_MAX;  // primary intron gene
            char umi[16] = {};
            uint8_t umi_len = 0;
            uint8_t nh = 0;
            uint8_t seen = 0;
            bool gene_ambiguous = false;
            uint32_t first_gene  = UINT32_MAX;   // first non-intergenic gene seen
            uint32_t second_gene = UINT32_MAX;   // conflicting gene for EM rescue
        };
        std::unordered_map<uint64_t, MultiHitEntry> mm_buffer;
        mm_buffer.reserve(500000);

        // Lambda to resolve a completed multi-hit entry
        auto resolve_mm = [&](MultiHitEntry& e, PileupStats& st) {
            if (e.bc_idx < 0) return;
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
            if (e.first_gene == UINT32_MAX) return;  // all intergenic

            // Count exon hit
            if (e.gene_idx != UINT32_MAX && e.exon_idx != UINT32_MAX) {
                if (config_.umi_dedup && config_.umi_dedup_directional && e.umi_len > 0) {
                    // N6: defer to directional store — finalize() will emit corrected count
                    uint64_t packed = umi_pack_2bit(e.umi, static_cast<int>(e.umi_len));
                    dir_exon_store_.record(static_cast<uint32_t>(e.bc_idx), e.gene_idx,
                                           e.exon_idx, packed);
                    st.exon_hits++;
                } else {
                    bool should_count = true;
                    if (config_.umi_dedup && e.umi_len > 0) {
                        should_count = umi_dedup_.insert(
                            static_cast<uint32_t>(e.bc_idx), e.gene_idx,
                            e.umi, static_cast<int>(e.umi_len));
                        if (should_count) st.umi_unique++;
                        else st.umi_duplicate++;
                    }
                    if (should_count) {
                        exon_acc_.increment(e.exon_idx, static_cast<uint32_t>(e.bc_idx));
                        st.exon_hits++;
                    }
                }
            }
            // Count intron hit
            if (config_.count_introns && e.intron_gene != UINT32_MAX &&
                e.intron_idx != UINT32_MAX) {
                if (config_.umi_dedup && config_.umi_dedup_directional && e.umi_len > 0) {
                    uint64_t packed = umi_pack_2bit(e.umi, static_cast<int>(e.umi_len));
                    dir_intron_store_.record(static_cast<uint32_t>(e.bc_idx), e.intron_gene,
                                             e.intron_idx, packed);
                    st.intron_hits++;
                } else {
                    bool should_count = true;
                    if (config_.umi_dedup && e.umi_len > 0) {
                        should_count = intron_umi_dedup_.insert(
                            static_cast<uint32_t>(e.bc_idx), e.intron_gene,
                            e.umi, static_cast<int>(e.umi_len));
                    }
                    if (should_count) {
                        intron_acc_.increment(e.intron_idx, static_cast<uint32_t>(e.bc_idx));
                        st.intron_hits++;
                    }
                }
            }
        };

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

        while (sam_read1(fp, hdr, b) >= 0) {
            // Prefetch aux data region into L2 cache while doing flag checks.
            __builtin_prefetch(bam_get_aux(b), 0, 2);
#ifdef SINGLET_PROFILE
            prof.t_bam_io += ProfileCounters::rdtsc() - _bam_io_start;
            prof.n_samples++;
#endif
            stats.total_reads++;

            if (stats.total_reads % 5000000 == 0) {
                std::cerr << "[singlet-pileup] Processed " << stats.total_reads / 1000000
                          << "M reads, " << stats.barcoded_reads << " barcoded, "
                          << stats.snp_hits << " SNP hits, "
                          << stats.exon_hits << " exon, "
                          << stats.intron_hits << " intron, "
                          << mm_buffer.size() << " mm_buf\n";
            }

            if (b->core.flag & BAM_FUNMAP) { PROF_BAM_IO_RESTART; continue; }
            stats.mapped_reads++;

            // G-RRNA: sample every 1000th mapped read for rRNA contamination
            if (b->core.l_qseq >= 21 && stats.mapped_reads % 1000 == 0) {
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
                if (has_gene_model_) {
                    // Inline CB scanner — same as primary path, avoids htslib call overhead
                    const char* cb_sec_str = inline_scan_cb(b);
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
                                        if (it->second.gene_idx == UINT32_MAX) {
                                            it->second.gene_idx = hit.gene_idx;
                                            it->second.exon_idx = hit.exon_idx;
                                        }
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
            const char* cb_str = inline_scan_cb(b);
            if (!cb_str || cb_str[0] == '-') {
                stats.no_barcode++;
                PROF_END(prof, barcode);
                PROF_BAM_IO_RESTART;
                continue;
            }
            int32_t bc_idx = bc_index_.lookup(cb_str);
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
                    int gid = guide_ref_.match(guide_seq_buf_.data(), qlen);
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
                // Extract aligned blocks from CIGAR
                aligned_blocks_.clear();
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
                            case BAM_CDEL:
                            case BAM_CREF_SKIP:
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

                for (const auto& [bstart, bend] : aligned_blocks_) {
                    gene_model_.query_exons(tid, bstart, bend, block_overlaps_);
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

                // Auto-detect wrong strand: after 10K strand-informative reads,
                // if >40% are wrong-strand, flip to reverse_strand mode.
                // The ~10K "wrong" reads are <0.1% of a typical run → negligible error.
                if (!strand_flipped_ && !config_.reverse_strand &&
                    (stats.exon_hits + stats.wrong_strand >= 10000)) {
                    strand_flipped_ = true;  // one-shot probe
                    double ws_frac = static_cast<double>(stats.wrong_strand) /
                                     (stats.exon_hits + stats.wrong_strand);
                    if (ws_frac > 0.4) {
                        config_.reverse_strand = true;
                        std::cerr << "[singlet-pileup] Auto-detected 5' protocol ("
                                  << static_cast<int>(ws_frac * 100)
                                  << "% wrong-strand in first 10K exonic reads). "
                                  << "Switching to reverse-strand counting.\n";
                    }
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

                if (nh == 1) {
                    // ── Unique read: count immediately ──
                    if (exon_gene != UINT32_MAX) {
                        if (config_.umi_dedup && config_.umi_dedup_directional && umi_str) {
                            // N6: defer to directional store
                            uint64_t packed = umi_pack_2bit(umi_str, umi_len);
                            dir_exon_store_.record(static_cast<uint32_t>(bc_idx), exon_gene,
                                                   first_exon_idx, packed);
                            stats.exon_hits++;
                        } else {
                            bool should_count = true;
                            if (config_.umi_dedup && umi_str) {
                                should_count = umi_dedup_.insert(
                                    static_cast<uint32_t>(bc_idx), exon_gene, umi_str, umi_len);
                                if (should_count) stats.umi_unique++;
                                else stats.umi_duplicate++;
                            }
                            if (should_count) {
                                exon_acc_.increment(first_exon_idx, static_cast<uint32_t>(bc_idx));
                                stats.exon_hits++;
                            }
                        }
                    }
                    if (config_.count_introns && intron_gene != UINT32_MAX) {
                        if (config_.umi_dedup && config_.umi_dedup_directional && umi_str) {
                            // N6: defer to directional store
                            uint64_t packed = umi_pack_2bit(umi_str, umi_len);
                            dir_intron_store_.record(static_cast<uint32_t>(bc_idx), intron_gene,
                                                     first_intron_idx, packed);
                            stats.intron_hits++;
                        } else {
                            bool should_count = true;
                            if (config_.umi_dedup && umi_str) {
                                should_count = intron_umi_dedup_.insert(
                                    static_cast<uint32_t>(bc_idx), intron_gene, umi_str, umi_len);
                            }
                            if (should_count) {
                                intron_acc_.increment(first_intron_idx, static_cast<uint32_t>(bc_idx));
                                stats.intron_hits++;
                            }
                        }
                    }
                } else {
                    // ── Multi-mapped primary: buffer for cross-alignment resolution ──
                    uint64_t qh = hash_qname(bam_get_qname(b));
                    auto& entry = mm_buffer[qh];
                    entry.nh = static_cast<uint8_t>(std::min(nh, 255));
                    entry.seen++;
                    entry.bc_idx = bc_idx;
                    // Only update gene/exon info if primary found a gene,
                    // or if no secondary has provided gene info yet
                    if (exon_gene != UINT32_MAX) {
                        entry.exon_idx = first_exon_idx;
                        entry.gene_idx = exon_gene;
                    }
                    if (intron_gene != UINT32_MAX && entry.intron_gene == UINT32_MAX) {
                        entry.intron_idx = first_intron_idx;
                        entry.intron_gene = intron_gene;
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
                        bool should_count = true;
                        if (config_.umi_dedup && umi_str) {
                            should_count = vdj_umi_.insert(
                                static_cast<uint32_t>(bc_idx), vdj_gene, umi_str, umi_len);
                        }
                        if (should_count) {
                            vdj_acc_.increment(vdj_gene, static_cast<uint32_t>(bc_idx));
                            stats.vdj_hits++;
                        }
                    }
                }

                PROF_END(prof, exon_intron);

                // Splice junction extraction (always for primary alignments)
                if (config_.count_sj) {
                    PROF_START(sj);
                    extract_splice_junctions(b, tid, static_cast<uint32_t>(bc_idx), umi_str, umi_len, stats);
                    PROF_END(prof, sj);
                }
            }

            // ── Position-level pileup (SNPs, editing) ──
            // Only use high-MAPQ reads for variant calling accuracy
            if (high_mapq) {
                PROF_START(variant);
                int32_t end_pos_v = bam_endpos(b);
                if (has_variant_overlap(tid, b->core.pos, end_pos_v)) {
                    pileup_aligned_bases(b, tid, static_cast<uint32_t>(bc_idx),
                                         umi_str, umi_len, stats);
                }
                PROF_END(prof, variant);
            }

            // ── Inline chrM base pileup for heteroplasmy ──
            if (config_.count_mt && is_chrm(tid)) {
                PROF_START(mt_pileup);
                pileup_mt_bases(b, static_cast<uint32_t>(bc_idx), stats);
                PROF_END(prof, mt_pileup);
            }
            PROF_BAM_IO_RESTART;
        }
        PROF_BAM_IO_ACCUM;  // account for last failed sam_read1

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

        // ── N6: Directional UMI correction (run() single-thread path only) ──
        if (config_.umi_dedup && config_.umi_dedup_directional) {
            std::cerr << "[singlet-pileup] Directional UMI correction ("
                      << dir_exon_store_.n_groups() << " cell×gene groups)...\n";
            uint64_t exon_deduped = dir_exon_store_.finalize(exon_acc_, config_.umi_len);
            uint64_t intron_deduped = 0;
            if (config_.count_introns)
                intron_deduped = dir_intron_store_.finalize(intron_acc_, config_.umi_len);
            stats.umi_unique = exon_deduped + intron_deduped;
            // exon_hits/intron_hits were counted provisionally during recording;
            // the corrected matrix counts are in exon_acc_/intron_acc_.
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
        SparseAccumulator<uint8_t> snp_ad, snp_dp;
        SparseAccumulator<uint16_t> exon, intron;

        // SJ tracking: per-worker local index + names
        std::unordered_map<uint64_t, uint32_t> sj_index;
        std::vector<std::string> sj_names;
        SparseAccumulator<uint16_t> sj;

        // chrM allele counts (if count_mt enabled)
        SparseAccumulator<uint16_t> mt;

        // chrM indel events captured during CIGAR walk
        std::vector<mt::MtIndelEvent> mt_indels;

        UmiDedup exon_umi, intron_umi, snp_umi, sj_umi;
        // N6: Directional UMI stores (parallel path — per-worker, finalized after thread join)
        DirectionalUmiStore dir_exon_store;
        DirectionalUmiStore dir_intron_store;

        // N18: CRISPR guide counts (per-worker)
        SparseAccumulator<uint16_t> guide;
        UmiDedup guide_umi;
        std::string guide_seq_buf;  // per-worker scratch for BAM seq decode

        // N17: VDJ gene usage (per-worker)
        SparseAccumulator<uint16_t> vdj;
        UmiDedup vdj_umi;
        std::vector<uint32_t> vdj_overlaps;  // scratch

        // Multi-mapper buffer (per-worker)
        struct MultiHitEntry {
            int32_t bc_idx = -1;
            uint32_t exon_idx = UINT32_MAX;
            uint32_t gene_idx = UINT32_MAX;
            uint32_t intron_idx = UINT32_MAX;
            uint32_t intron_gene = UINT32_MAX;
            char umi[16] = {};
            uint8_t umi_len = 0;
            uint8_t nh = 0;
            uint8_t seen = 0;
            bool gene_ambiguous = false;
            uint32_t first_gene  = UINT32_MAX;
            uint32_t second_gene = UINT32_MAX;  // conflicting gene for EM rescue
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
        // G-RRNA: per-worker rRNA detector
        singlet::RrnaDetector rrna_detector;
        // N22: per-worker WL ambient gene profile (populated when whitelist_path is set)
        std::vector<uint64_t> wl_ambient_genes;
    };

    PileupStats run_parallel(const std::string& bam_path, int n_workers) {
        auto t0 = std::chrono::high_resolution_clock::now();

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

        // ── Auto-strand pre-probe (parallel path) ──
        // The streaming run() auto-detects strand at 10K reads. In parallel mode
        // workers start concurrently and never run that probe, so for reverse-strand
        // protocols (e.g. 10xv3) all reads are mis-classified. Fix: scan the most
        // read-rich chromosome sequentially before spawning workers.
        if (config_.stranded && !strand_flipped_ && has_gene_model_ && n_chroms > 0) {
            int probe_tid = chrom_order[0];  // highest-read chromosome
            hts_itr_t* pitr = sam_itr_queryi(idx, probe_tid, 0, HTS_POS_MAX);
            if (pitr) {
                bam1_t* pb = bam_init1();
                uint64_t ph = 0, pw = 0;  // probe: exon_hits, wrong_strand
                std::vector<std::pair<int32_t,int32_t>> pblocks;
                std::vector<uint32_t> poverlaps;
                while (sam_itr_next(master_fp, pitr, pb) >= 0 && ph + pw < 20000) {
                    if (pb->core.flag & (BAM_FUNMAP|BAM_FSECONDARY|BAM_FSUPPLEMENTARY)) continue;
                    pblocks.clear();
                    int32_t rpos = pb->core.pos;
                    const uint32_t* cig = bam_get_cigar(pb);
                    for (int ci = 0; ci < pb->core.n_cigar; ++ci) {
                        int op = bam_cigar_op(cig[ci]), len = bam_cigar_oplen(cig[ci]);
                        if (op==BAM_CMATCH||op==BAM_CEQUAL||op==BAM_CDIFF)
                            { pblocks.push_back({rpos, rpos+len}); rpos += len; }
                        else if (op==BAM_CDEL||op==BAM_CREF_SKIP) rpos += len;
                    }
                    bool rrev = (pb->core.flag & BAM_FREVERSE) != 0;
                    bool hit = false, reject = false;
                    for (const auto& [bs, be] : pblocks) {
                        gene_model_.query_exons(probe_tid, bs, be, poverlaps);
                        for (uint32_t eidx : poverlaps) {
                            uint32_t g = gene_model_.exon_to_gene(eidx);
                            char gs = gene_model_.gene_strand(g);
                            if (gs == '.') { hit = true; continue; }
                            bool exp_rev = (gs == '-');
                            if (rrev == exp_rev) hit = true; else reject = true;
                        }
                    }
                    if (hit) ph++; else if (reject) pw++;
                }
                bam_destroy1(pb);
                hts_itr_destroy(pitr);
                if (ph + pw >= 1000) {
                    strand_flipped_ = true;
                    double wf = static_cast<double>(pw) / (ph + pw);
                    if (wf > 0.4) {
                        config_.reverse_strand = true;
                        std::cerr << "[singlet-pileup] Auto-detected 5' protocol ("
                                  << static_cast<int>(wf * 100)
                                  << "% wrong-strand in parallel pre-probe). "
                                  << "Switching to reverse-strand counting.\n";
                    }
                }
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
                }
            }
            if (config_.umi_dedup) {
                wc.exon_umi.reserve(1000000);
                wc.intron_umi.reserve(500000);
            }
            if (config_.count_mt) {
                wc.mt.set_barcodes(barcodes_);
                wc.mt.set_n_features(mt::MT_N_FEATURES);
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

        // ── N6: Directional UMI correction (parallel path) — per-worker finalize ──
        // Each worker held all reads for its chromosomes; since genes don't span chromosomes,
        // per-worker finalize is equivalent to global finalize.
        uint64_t dir_exon_deduped = 0, dir_intron_deduped = 0;
        if (config_.umi_dedup && config_.umi_dedup_directional) {
            for (int w = 0; w < n_workers; ++w) {
                dir_exon_deduped += workers[w].dir_exon_store.finalize(
                    workers[w].exon, config_.umi_len);
                if (config_.count_introns)
                    dir_intron_deduped += workers[w].dir_intron_store.finalize(
                        workers[w].intron, config_.umi_len);
            }
            std::cerr << "[singlet-pileup] Directional UMI correction (parallel): "
                      << dir_exon_deduped << " exon + " << dir_intron_deduped
                      << " intron unique UMIs\n";
        }

        auto proc_t1 = std::chrono::high_resolution_clock::now();
        double proc_s = std::chrono::duration<double>(proc_t1 - proc_t0).count();

        // ── Cross-worker multi-mapper merge ──
        // Reads whose primary alignment is on chromosome A and secondary on chromosome B
        // are split across workers and cannot be resolved within a single worker.
        // Partial mm_buffer entries (seen < nh) are merged here across all workers.
        uint64_t xw_exon = 0, xw_umi_u = 0, xw_umi_d = 0;
        {
            std::unordered_map<uint64_t, WorkerContext::MultiHitEntry> global_mm;
            global_mm.reserve(500000);

            for (int w = 0; w < n_workers; ++w) {
                for (auto& [qh, wentry] : workers[w].mm_buffer) {
                    auto& m = global_mm[qh];
                    m.seen += wentry.seen;
                    if (m.nh == 0 && wentry.nh > 0) m.nh = wentry.nh;
                    if (m.bc_idx < 0 && wentry.bc_idx >= 0) m.bc_idx = wentry.bc_idx;
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
                                if (m.gene_idx == UINT32_MAX && wentry.gene_idx != UINT32_MAX) {
                                    m.gene_idx = wentry.gene_idx;
                                    m.exon_idx = wentry.exon_idx;
                                }
                            } else if (eg != m.first_gene) {
                                m.gene_ambiguous = true;
                                // G-EM: save conflicting gene as second candidate
                                if (m.second_gene == UINT32_MAX) m.second_gene = eg;
                                m.gene_idx = UINT32_MAX;
                                m.exon_idx = UINT32_MAX;
                            }
                        }
                    }
                    if (config_.count_introns && m.intron_gene == UINT32_MAX &&
                        wentry.intron_gene != UINT32_MAX) {
                        m.intron_gene = wentry.intron_gene;
                        m.intron_idx = wentry.intron_idx;
                    }
                }
            }

            UmiDedup global_mm_umi;
            global_mm_umi.reserve(100000);

            for (auto& [qh, m] : global_mm) {
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
                if (m.first_gene == UINT32_MAX) continue;
                if (m.gene_idx == UINT32_MAX || m.exon_idx == UINT32_MAX) continue;

                bool should_count = true;
                if (config_.umi_dedup && m.umi_len > 0) {
                    should_count = global_mm_umi.insert(
                        static_cast<uint32_t>(m.bc_idx), m.gene_idx,
                        m.umi, static_cast<int>(m.umi_len));
                    if (should_count) xw_umi_u++;
                    else xw_umi_d++;
                }
                if (should_count) {
                    exon_acc_.increment(m.exon_idx, static_cast<uint32_t>(m.bc_idx));
                    xw_exon++;
                }
            }

            std::cerr << "[parallel-cross-worker] " << global_mm.size()
                      << " cross-worker entries → " << xw_exon << " exon hits, "
                      << xw_umi_u << " UMI unique\n";
        }

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
            stats.vdj_hits += ws.vdj_hits;
            stats.rrna_reads += ws.rrna_reads;
            stats.rrna_reads_sampled += ws.rrna_reads_sampled;
        }
        // Add cross-worker multi-mapper stats
        stats.exon_hits += xw_exon;
        stats.umi_unique += xw_umi_u;
        stats.umi_duplicate += xw_umi_d;
        // N6: replace umi_unique with directional dedup count (mirrors run() behaviour)
        // Include cross-worker simple-dedup unique count (xw multi-mappers use simple dedup)
        if (config_.umi_dedup && config_.umi_dedup_directional)
            stats.umi_unique = dir_exon_deduped + dir_intron_deduped + xw_umi_u;

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
        // SNP AD/DP
        merge_accumulators(snp_ad_, workers, [](WorkerContext& w) -> SparseAccumulator<uint8_t>& { return w.snp_ad; });
        merge_accumulators(snp_dp_, workers, [](WorkerContext& w) -> SparseAccumulator<uint8_t>& { return w.snp_dp; });
        if (config_.count_mt) {
            merge_accumulators(mt_acc_, workers, [](WorkerContext& w) -> SparseAccumulator<uint16_t>& { return w.mt; });
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
            }
        }

        // Merge SJ: reassign indices to build global SJ list
        merge_splice_junctions(workers);

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

    // Get accumulators for export
    const SparseAccumulator<uint8_t>& snp_ad() const { return snp_ad_; }
    const SparseAccumulator<uint8_t>& snp_dp() const { return snp_dp_; }
    const SparseAccumulator<uint16_t>& exons() const { return exon_acc_; }
    const SparseAccumulator<uint16_t>& introns() const { return intron_acc_; }
    const SparseAccumulator<uint16_t>& splice_junctions() const { return sj_acc_; }
    const SparseAccumulator<uint16_t>& mt_alleles() const { return mt_acc_; }
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
    bool strand_was_flipped() const { return strand_flipped_; }

    /// N22: Per-WL-barcode UMI counts (run() serial path only; empty in run_parallel())
    const std::vector<uint32_t>& wl_umi_counts() const { return wl_umi_counts_; }
    /// N22: Global gene hit sums from WL-only (non-auto-discovered) barcodes
    const std::vector<uint64_t>& wl_ambient_gene_counts() const { return wl_ambient_gene_counts_; }
    /// N22: UMI ceiling used to classify WL barcodes as ambient
    uint32_t wl_ambient_ceil() const { return config_.wl_ambient_ceil; }

    // N7: Access directional UMI store for saturation computation
    const DirectionalUmiStore& dir_exon_store() const { return dir_exon_store_; }

    // G-EM: Ambiguous reads buffered for post-pileup EM rescue
    const std::vector<em_rescue::AmbigRead>& ambig_reads() const { return ambig_reads_; }

private:
    PileupConfig config_;
    bool strand_flipped_ = false;   // Set if auto-strand probe flipped reverse_strand
    std::vector<std::string> barcodes_;
    std::vector<uint32_t> per_cell_reads_;  // N20: barcoded primary reads per cell (pre-dedup)
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

    // UMI deduplication (separate instances for exonic vs intronic)
    UmiDedup umi_dedup_;
    UmiDedup intron_umi_dedup_;



    // N6: Directional UMI error correction stores (run() single-thread path)
    // In run_parallel(), per-worker DirectionalUmiStore instances in WorkerContext are used instead.
    DirectionalUmiStore dir_exon_store_;
    DirectionalUmiStore dir_intron_store_;

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

    // Accumulators
    SparseAccumulator<uint8_t> snp_ad_;
    SparseAccumulator<uint8_t> snp_dp_;
    SparseAccumulator<uint16_t> exon_acc_;
    SparseAccumulator<uint16_t> intron_acc_;
    SparseAccumulator<uint16_t> sj_acc_;
    SparseAccumulator<uint16_t> mt_acc_;   // chrM allele counts: feat = pos*4 + base
    std::vector<mt::MtIndelEvent> mt_indels_;  // chrM indels captured during CIGAR walk
    UmiDedup snp_umi_dedup_;  // UMI dedup for SNP positions
    UmiDedup sj_umi_dedup_;  // UMI dedup for splice junctions

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
    UmiDedup vdj_umi_;

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
        for (int i = 0; i < n; ++i) {
            std::string name = sam_hdr_tid2name(hdr, i);
            chrom_to_tid_[name] = i;
            tid_to_chrom_[i] = name;
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
                    bool should_count = true;
                    if (config_.umi_dedup && umi_str) {
                        should_count = sj_umi_dedup_.insert(bc_idx, sj_idx, umi_str, umi_len);
                    }
                    if (should_count) {
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
                            bool umi_new = !config_.umi_dedup || !umi_str ||
                                           snp_umi_dedup_.insert(bc_idx, it->index,
                                                                umi_str, umi_len);
                            if (umi_new) {
                                snp_dp_.increment(it->index, bc_idx, 1);
                                if (base == it->alt_allele) {
                                    snp_ad_.increment(it->index, bc_idx, 1);
                                }
                            }
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

    // Resolve a completed multi-hit entry using per-worker state
    void resolve_mm_parallel(WorkerContext& wc, typename WorkerContext::MultiHitEntry& e) {
        if (e.bc_idx < 0) return;
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
        if (e.first_gene == UINT32_MAX) return;

        if (e.gene_idx != UINT32_MAX && e.exon_idx != UINT32_MAX) {
            if (config_.umi_dedup && config_.umi_dedup_directional && e.umi_len > 0) {
                // N6: defer to directional store — finalize() will emit corrected count
                wc.dir_exon_store.record(static_cast<uint32_t>(e.bc_idx), e.gene_idx,
                                         e.exon_idx, umi_pack_2bit(e.umi, static_cast<int>(e.umi_len)));
                wc.stats.exon_hits++;
            } else {
                bool should_count = true;
                if (config_.umi_dedup && e.umi_len > 0) {
                    should_count = wc.exon_umi.insert(
                        static_cast<uint32_t>(e.bc_idx), e.gene_idx,
                        e.umi, static_cast<int>(e.umi_len));
                    if (should_count) wc.stats.umi_unique++;
                    else wc.stats.umi_duplicate++;
                }
                if (should_count) {
                    wc.exon.increment(e.exon_idx, static_cast<uint32_t>(e.bc_idx));
                    wc.stats.exon_hits++;
                }
            }
        }
        if (config_.count_introns && e.intron_gene != UINT32_MAX &&
            e.intron_idx != UINT32_MAX) {
            if (config_.umi_dedup && config_.umi_dedup_directional && e.umi_len > 0) {
                // N6: defer to directional store — finalize() will emit corrected count
                wc.dir_intron_store.record(static_cast<uint32_t>(e.bc_idx), e.intron_gene,
                                           e.intron_idx, umi_pack_2bit(e.umi, static_cast<int>(e.umi_len)));
                wc.stats.intron_hits++;
            } else {
                bool should_count = true;
                if (config_.umi_dedup && e.umi_len > 0) {
                    should_count = wc.intron_umi.insert(
                        static_cast<uint32_t>(e.bc_idx), e.intron_gene,
                        e.umi, static_cast<int>(e.umi_len));
                }
                if (should_count) {
                    wc.intron.increment(e.intron_idx, static_cast<uint32_t>(e.bc_idx));
                    wc.stats.intron_hits++;
                }
            }
        }
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
                    bool should_count = true;
                    if (config_.umi_dedup && umi_str) {
                        should_count = wc.sj_umi.insert(bc_idx, sj_idx, umi_str, umi_len);
                    }
                    if (should_count) {
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
                            bool umi_new = !config_.umi_dedup || !umi_str ||
                                           wc.snp_umi.insert(bc_idx, it->index,
                                                             umi_str, umi_len);
                            if (umi_new) {
                                wc.snp_dp.increment(it->index, bc_idx, 1);
                                if (base == it->alt_allele)
                                    wc.snp_ad.increment(it->index, bc_idx, 1);
                            }
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

        // G-RRNA: sample every 1000th mapped read for rRNA contamination
        if (b->core.l_qseq >= 21 && wc.stats.mapped_reads % 1000 == 0) {
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
                                        if (it->second.gene_idx == UINT32_MAX) {
                                            it->second.gene_idx = hit.gene_idx;
                                            it->second.exon_idx = hit.exon_idx;
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
        int32_t bc_idx = bc_index_.lookup(cb_str);
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
                int gid = guide_ref_.match(wc.guide_seq_buf.data(), qlen);
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
                        case BAM_CDEL: case BAM_CREF_SKIP:
                            rpos += len; break;
                        default: break;
                    }
                }
            }

            wc.exon_overlaps.clear();
            bool read_reverse = (b->core.flag & BAM_FREVERSE) != 0;
            bool had_strand_reject = false;

            for (const auto& [bstart, bend] : wc.aligned_blocks) {
                gene_model_.query_exons(tid, bstart, bend, wc.block_overlaps);
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

            if (nh == 1) {
                if (exon_gene != UINT32_MAX) {
                    if (config_.umi_dedup && config_.umi_dedup_directional && umi_str) {
                        // N6: defer to directional store — finalize() will emit corrected count
                        wc.dir_exon_store.record(static_cast<uint32_t>(bc_idx), exon_gene,
                                                 first_exon_idx, umi_pack_2bit(umi_str, umi_len));
                        wc.stats.exon_hits++;
                    } else {
                        bool should_count = true;
                        if (config_.umi_dedup && umi_str) {
                            should_count = wc.exon_umi.insert(
                                static_cast<uint32_t>(bc_idx), exon_gene, umi_str, umi_len);
                            if (should_count) wc.stats.umi_unique++;
                            else wc.stats.umi_duplicate++;
                        }
                        if (should_count) {
                            wc.exon.increment(first_exon_idx, static_cast<uint32_t>(bc_idx));
                            wc.stats.exon_hits++;
                        }
                    }
                }
                if (config_.count_introns && intron_gene != UINT32_MAX) {
                    if (config_.umi_dedup && config_.umi_dedup_directional && umi_str) {
                        // N6: defer to directional store — finalize() will emit corrected count
                        wc.dir_intron_store.record(static_cast<uint32_t>(bc_idx), intron_gene,
                                                   first_intron_idx, umi_pack_2bit(umi_str, umi_len));
                        wc.stats.intron_hits++;
                    } else {
                        bool should_count = true;
                        if (config_.umi_dedup && umi_str) {
                            should_count = wc.intron_umi.insert(
                                static_cast<uint32_t>(bc_idx), intron_gene, umi_str, umi_len);
                        }
                        if (should_count) {
                            wc.intron.increment(first_intron_idx, static_cast<uint32_t>(bc_idx));
                            wc.stats.intron_hits++;
                        }
                    }
                }
            } else {
                uint64_t qh = hash_qname(bam_get_qname(b));
                auto& entry = wc.mm_buffer[qh];
                entry.nh = static_cast<uint8_t>(std::min(nh, 255));
                entry.seen++;
                entry.bc_idx = bc_idx;
                if (exon_gene != UINT32_MAX) {
                    entry.exon_idx = first_exon_idx;
                    entry.gene_idx = exon_gene;
                }
                if (intron_gene != UINT32_MAX && entry.intron_gene == UINT32_MAX) {
                    entry.intron_idx = first_intron_idx;
                    entry.intron_gene = intron_gene;
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
                    bool should_count = true;
                    if (config_.umi_dedup && umi_str) {
                        should_count = wc.vdj_umi.insert(
                            static_cast<uint32_t>(bc_idx), vdj_gene, umi_str, umi_len);
                    }
                    if (should_count) {
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
        if (config_.count_mt && is_chrm(tid)) {
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

} // namespace singlet
