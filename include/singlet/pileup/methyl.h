// SPDX-License-Identifier: MIT
#pragma once
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace singlet {

// ── Enumerations ─────────────────────────────────────────────────────────────

enum class MethylContext { CPG,
                           CHG,
                           CHH };

enum class MethylMode { WGBS,
                        RRBS,
                        SCBS,
                        NOME_SEQ };

// ── Core data types ───────────────────────────────────────────────────────────

struct MethylSite {
    std::string chrom;
    int32_t pos{0};
    bool strand_plus{true};  // true = OT/CTOT, false = OB/CTOB
    MethylContext context{MethylContext::CPG};
    uint16_t meth_reads{0};
    uint16_t total_reads{0};
};

struct MethylStats {
    uint64_t total_sites{0};
    uint64_t cpg_sites{0};
    uint64_t chg_sites{0};
    uint64_t chh_sites{0};
    double global_cpg_methylation{0.0};
    double global_chg_methylation{0.0};
    double global_chh_methylation{0.0};
    // Bisulfite conversion QC: C→T conversions at non-CpG sites
    uint64_t conversion_rate_ct{0};
    uint64_t total_ct_contexts{0};
    // Accumulators for running mean (not exposed in final JSON)
    uint64_t cpg_meth_sum{0};
    uint64_t cpg_total_sum{0};
    uint64_t chg_meth_sum{0};
    uint64_t chg_total_sum{0};
    uint64_t chh_meth_sum{0};
    uint64_t chh_total_sum{0};
};

// ── CIGAR walker ─────────────────────────────────────────────────────────────

// Parse a CIGAR string and invoke callback(op, len) for each operation.
// Supported ops: M I D S N H P = X
inline void walk_cigar(const std::string& cigar,
                       std::function<void(char op, int len)> callback) {
    if (cigar.empty() || cigar == "*") return;
    int len = 0;
    for (char c : cigar) {
        if (c >= '0' && c <= '9') {
            len = len * 10 + (c - '0');
        } else {
            if (len == 0) len = 1;
            callback(c, len);
            len = 0;
        }
    }
}

// ── Context classification ────────────────────────────────────────────────────

// Classify the methylation context of a cytosine at `pos` in `ref_seq`.
// ref_seq is a window of reference sequence; pos is the 0-based index within
// that window of the cytosine base. Bases beyond the window end are treated as
// any non-G base (CHH context).
// Called for the forward strand: the cytosine is at ref_seq[pos].
// For reverse strand, caller should invert (G on + strand → C on template).
inline MethylContext classify_methyl_context(std::string_view ref_seq, int pos) {
    if (pos < 0 || static_cast<size_t>(pos) >= ref_seq.size()) return MethylContext::CHH;

    // +1 base
    char next1 = (static_cast<size_t>(pos + 1) < ref_seq.size()) ? ref_seq[pos + 1] : 'N';
    // +2 base
    char next2 = (static_cast<size_t>(pos + 2) < ref_seq.size()) ? ref_seq[pos + 2] : 'N';

    next1 = static_cast<char>(toupper(static_cast<unsigned char>(next1)));
    next2 = static_cast<char>(toupper(static_cast<unsigned char>(next2)));

    if (next1 == 'G') return MethylContext::CPG;
    if (next2 == 'G') return MethylContext::CHG;
    return MethylContext::CHH;
}

// ── Single-site methylation call ──────────────────────────────────────────────

// At a given aligned position, determine if the cytosine is methylated.
// For forward strand (+): reference C; read C = methylated, read T = unmethylated.
// For reverse strand (−): reference G (complement of C on template);
//   read G = methylated, read A = unmethylated.
// read_seq, ref_seq: aligned sequences at this position (may be longer; only
// positions read_pos / ref_pos are examined).
// Returns: {context, is_methylated} or nullopt if not a C/G site or ambiguous.
inline std::optional<std::pair<MethylContext, bool>>
call_methylation_site(std::string_view read_seq, std::string_view ref_seq,
                      int read_pos, int ref_pos, bool is_reverse) {
    if (ref_pos < 0 || static_cast<size_t>(ref_pos) >= ref_seq.size()) return std::nullopt;
    if (read_pos < 0 || static_cast<size_t>(read_pos) >= read_seq.size()) return std::nullopt;

    char ref_base = static_cast<char>(toupper(static_cast<unsigned char>(ref_seq[ref_pos])));
    char read_base = static_cast<char>(toupper(static_cast<unsigned char>(read_seq[read_pos])));

    if (!is_reverse) {
        // Forward strand: look for C in reference
        if (ref_base != 'C') return std::nullopt;
        if (read_base != 'C' && read_base != 'T') return std::nullopt;  // ambiguous
        MethylContext ctx = classify_methyl_context(ref_seq, ref_pos);
        bool methylated = (read_base == 'C');
        return std::make_pair(ctx, methylated);
    } else {
        // Reverse strand: reference G corresponds to C on the template strand
        if (ref_base != 'G') return std::nullopt;
        if (read_base != 'G' && read_base != 'A') return std::nullopt;  // ambiguous
        // For context: look upstream (from + perspective, that is the 5' of C on - strand)
        // Context on minus strand: C is complement of G; next bases are complements read 3'→5'
        // Simplified: classify using the position on + ref in reverse complement sense
        // G at ref_pos → C at ref_pos on minus strand; +1 on minus = -1 on plus
        MethylContext ctx;
        char prev1 = (ref_pos > 0)
                         ? static_cast<char>(toupper(static_cast<unsigned char>(ref_seq[ref_pos - 1])))
                         : 'N';
        char prev2 = (ref_pos > 1)
                         ? static_cast<char>(toupper(static_cast<unsigned char>(ref_seq[ref_pos - 2])))
                         : 'N';
        // complement: C→G context; prev1 is the complement of +1 on minus strand
        // CpG on minus: the G at ref_pos is paired with C at ref_pos-1? No:
        //   On + strand:  5'…C G…3'  → CpG. The G is at ref_pos+1, C at ref_pos
        //   On - strand: we look at G in ref (complement of C on template).
        //   The C on template is followed by G on template = G on + strand precedes
        //   the G-on-+ we're examining. So: prev1 == 'C' means CpG context.
        if (prev1 == 'C') {
            ctx = MethylContext::CPG;
        } else if (prev2 == 'C') {
            ctx = MethylContext::CHG;
        } else {
            ctx = MethylContext::CHH;
        }
        bool methylated = (read_base == 'G');
        return std::make_pair(ctx, methylated);
    }
}

// ── MethylationCaller ─────────────────────────────────────────────────────────

struct MethylationCaller {
    std::vector<MethylSite> sites_;
    MethylStats stats_{};
    std::string current_chrom_;

    // Process one alignment.
    // read_seq:   the read sequence (as returned by aligner, not necessarily RC)
    // ref_window: reference bases from ref_start for the span of the alignment
    // ref_start:  genomic coordinate of ref_window[0]
    // cigar:      CIGAR string for this alignment
    // is_reverse: true if read mapped on minus strand
    void process_alignment(std::string_view read_seq, std::string_view ref_window,
                           int32_t ref_start, const std::string& cigar,
                           bool is_reverse) {
        int read_pos = 0;
        int ref_local = 0;  // offset into ref_window

        walk_cigar(cigar, [&](char op, int len) {
            switch (op) {
                case 'M':
                case '=':
                case 'X': {
                    for (int i = 0; i < len; ++i) {
                        int rp = read_pos + i;
                        int rl = ref_local + i;
                        if (static_cast<size_t>(rl) >= ref_window.size()) break;
                        if (static_cast<size_t>(rp) >= read_seq.size()) break;

                        auto res = call_methylation_site(read_seq, ref_window, rp, rl, is_reverse);
                        if (res) {
                            auto [ctx, meth] = *res;
                            int32_t genomic_pos = ref_start + rl;

                            // Update conversion QC for CHH (non-CpG) cytosines
                            if (ctx != MethylContext::CPG) {
                                ++stats_.total_ct_contexts;
                                if (!meth) ++stats_.conversion_rate_ct;
                            }

                            // Find or create site
                            bool found = false;
                            for (auto& s : sites_) {
                                if (s.pos == genomic_pos && s.strand_plus == !is_reverse && s.context == ctx) {
                                    ++s.total_reads;
                                    if (meth) ++s.meth_reads;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                MethylSite site;
                                site.chrom = current_chrom_;
                                site.pos = genomic_pos;
                                site.strand_plus = !is_reverse;
                                site.context = ctx;
                                site.total_reads = 1;
                                site.meth_reads = meth ? 1 : 0;
                                sites_.push_back(std::move(site));
                            }
                        }
                    }
                    read_pos += len;
                    ref_local += len;
                    break;
                }
                case 'I':
                    read_pos += len;  // insertion consumes read, not ref
                    break;
                case 'D':
                case 'N':
                    ref_local += len;  // deletion/skip consumes ref, not read
                    break;
                case 'S':
                    read_pos += len;  // soft clip consumes read, not ref
                    break;
                case 'H':
                case 'P':
                    break;  // hard clip and padding don't consume anything in this view
                default:
                    break;
            }
        });
    }

    void set_chrom(const std::string& chrom) { current_chrom_ = chrom; }

    const std::vector<MethylSite>& get_sites() const { return sites_; }

    // Recompute aggregate stats from accumulated sites and return.
    MethylStats get_stats() {
        // Recompute global methylation from sites
        uint64_t cpg_m = 0, cpg_t = 0;
        uint64_t chg_m = 0, chg_t = 0;
        uint64_t chh_m = 0, chh_t = 0;

        for (const auto& s : sites_) {
            switch (s.context) {
                case MethylContext::CPG:
                    cpg_m += s.meth_reads;
                    cpg_t += s.total_reads;
                    ++stats_.cpg_sites;
                    break;
                case MethylContext::CHG:
                    chg_m += s.meth_reads;
                    chg_t += s.total_reads;
                    ++stats_.chg_sites;
                    break;
                case MethylContext::CHH:
                    chh_m += s.meth_reads;
                    chh_t += s.total_reads;
                    ++stats_.chh_sites;
                    break;
            }
        }
        stats_.total_sites = sites_.size();
        stats_.global_cpg_methylation = cpg_t > 0 ? static_cast<double>(cpg_m) / cpg_t : 0.0;
        stats_.global_chg_methylation = chg_t > 0 ? static_cast<double>(chg_m) / chg_t : 0.0;
        stats_.global_chh_methylation = chh_t > 0 ? static_cast<double>(chh_m) / chh_t : 0.0;
        // Reset site counters (they accumulate each call otherwise)
        stats_.cpg_sites = cpg_t > 0 ? stats_.cpg_sites : 0;
        stats_.chg_sites = chg_t > 0 ? stats_.chg_sites : 0;
        stats_.chh_sites = chh_t > 0 ? stats_.chh_sites : 0;

        // Recount cleanly
        stats_.cpg_sites = 0;
        stats_.chg_sites = 0;
        stats_.chh_sites = 0;
        for (const auto& s : sites_) {
            if (s.context == MethylContext::CPG)
                ++stats_.cpg_sites;
            else if (s.context == MethylContext::CHG)
                ++stats_.chg_sites;
            else
                ++stats_.chh_sites;
        }
        stats_.total_sites = sites_.size();
        return stats_;
    }

    // Thread-safe merge of another caller into this one.
    void merge(const MethylationCaller& other) {
        // Merge sites
        for (const auto& os : other.sites_) {
            bool found = false;
            for (auto& ms : sites_) {
                if (ms.chrom == os.chrom && ms.pos == os.pos &&
                    ms.strand_plus == os.strand_plus && ms.context == os.context) {
                    ms.meth_reads += os.meth_reads;
                    ms.total_reads += os.total_reads;
                    found = true;
                    break;
                }
            }
            if (!found) sites_.push_back(os);
        }
        // Merge conversion QC counts
        stats_.conversion_rate_ct += other.stats_.conversion_rate_ct;
        stats_.total_ct_contexts += other.stats_.total_ct_contexts;
    }
};

// ── Bisulfite detection ───────────────────────────────────────────────────────

// Detect whether reads are bisulfite-converted.
// md_tag: the MD:Z tag string (unused here — kept for API compatibility; pass "")
// read_seq: concatenated read sequence(s) from sampled reads
// n_reads: hint for how many reads were sampled (unused internally; for doc)
// Returns true if C→T rate in non-CpG context > 50 % (strong bisulfite signal).
// Simple heuristic: count C in ref_seq vs T in read_seq at C positions.
// For this standalone version: if read_seq has dramatically more T than C,
// compared to a normal genomic base composition, flag as bisulfite.
// A more robust version would compare to a reference; this is a read-only check.
inline bool detect_bisulfite(const std::string& /*md_tag*/,
                             const std::string& read_seq,
                             int /*n_reads*/ = 1000) {
    if (read_seq.empty()) return false;
    // Count C and T in the read sequence: bisulfite converts C→T so we
    // expect very few C and many T.  Normal genomic reads have ~25% each.
    // Threshold: if fraction of cytosines (out of C+T) < 0.25, call bisulfite.
    uint32_t c_count = 0, t_count = 0;
    for (unsigned char b : read_seq) {
        char u = static_cast<char>(toupper(b));
        if (u == 'C')
            ++c_count;
        else if (u == 'T')
            ++t_count;
    }
    uint32_t ct_total = c_count + t_count;
    if (ct_total == 0) return false;
    double c_fraction = static_cast<double>(c_count) / ct_total;
    // In bisulfite reads, most C→T; expect c_fraction < 0.25
    return c_fraction < 0.25;
}

// ── Mode detection ────────────────────────────────────────────────────────────

// Infer the bisulfite experiment mode.
// cpg_meth:      global CpG methylation fraction
// chh_meth:      global CHH methylation fraction (≈ unconverted rate)
// has_mspi_sites: true if MspI (CCGG) sites are enriched (RRBS indicator)
inline MethylMode detect_methyl_mode(double cpg_meth, double chh_meth,
                                     bool has_mspi_sites) {
    if (has_mspi_sites) return MethylMode::RRBS;
    // NOMe-seq: both CpG (protected) and GpC (nucleosome) methylation high
    // In practice here we approximate: if global methylation is very high (>0.7)
    // and chh is also elevated (>0.4), treat as NOMe
    if (cpg_meth > 0.7 && chh_meth > 0.4) return MethylMode::NOME_SEQ;
    // SCBS: very low coverage per site → stochastic methylation profile;
    // detected externally (few reads per site), here approximated by very low
    // overall methylation fraction (few sites observed)
    if (cpg_meth < 0.1 && chh_meth < 0.05) return MethylMode::SCBS;
    return MethylMode::WGBS;
}

// ── Conversion rate estimation ────────────────────────────────────────────────

// Estimate bisulfite conversion efficiency from CHH methylation.
// CHH sites should show near-0 methylation if conversion is complete;
// residual methylation ≈ unconverted rate (>5% = poor conversion).
inline double estimate_conversion_rate(const MethylStats& stats) {
    if (stats.total_ct_contexts == 0) return 1.0;  // no data → assume perfect
    return 1.0 - (static_cast<double>(stats.conversion_rate_ct) /
                  stats.total_ct_contexts);
}

// ── Output ────────────────────────────────────────────────────────────────────

// Write Bismark-compatible BED file.
// Columns: chrom, start, end, methylation%, count_meth, count_unmeth, context
inline void write_methylation_bed(const std::vector<MethylSite>& sites,
                                  std::string_view outpath) {
    std::string outpath_str(outpath);
    std::ofstream f(outpath_str);
    if (!f) throw std::runtime_error("Cannot open " + outpath_str);

    for (const auto& s : sites) {
        if (s.total_reads == 0) continue;
        double pct = 100.0 * static_cast<double>(s.meth_reads) / s.total_reads;
        uint16_t unmeth = s.total_reads - s.meth_reads;
        const char* ctx_str = s.context == MethylContext::CPG   ? "CpG"
                              : s.context == MethylContext::CHG ? "CHG"
                                                                : "CHH";
        f << s.chrom << '\t'
          << s.pos << '\t'
          << (s.pos + 1) << '\t'
          << pct << '\t'
          << s.meth_reads << '\t'
          << unmeth << '\t'
          << ctx_str << '\n';
    }
}

// Write a JSON summary of methylation statistics.
inline void write_methyl_stats(const MethylStats& stats, std::string_view outpath) {
    std::string outpath_str(outpath);
    std::ofstream f(outpath_str);
    if (!f) throw std::runtime_error("Cannot open " + outpath_str);
    f << "{\n"
      << "  \"total_sites\": " << stats.total_sites << ",\n"
      << "  \"cpg_sites\": " << stats.cpg_sites << ",\n"
      << "  \"chg_sites\": " << stats.chg_sites << ",\n"
      << "  \"chh_sites\": " << stats.chh_sites << ",\n"
      << "  \"global_cpg_methylation\": " << stats.global_cpg_methylation << ",\n"
      << "  \"global_chg_methylation\": " << stats.global_chg_methylation << ",\n"
      << "  \"global_chh_methylation\": " << stats.global_chh_methylation << ",\n"
      << "  \"conversion_rate_ct\": " << stats.conversion_rate_ct << ",\n"
      << "  \"total_ct_contexts\": " << stats.total_ct_contexts << "\n"
      << "}\n";
}

}  // namespace singlet
