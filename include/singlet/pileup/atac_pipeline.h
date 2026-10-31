// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: atac_pipeline.h
// scATAC / Multiome ATAC post-alignment: coordinate-sorted STAR BAM (barcode in QNAME) -> fragments, peaks, per-barcode
// QC, cell calls, peak and gene-activity matrices, sample QC verdict. Used by the pipeline's ATAC branch and by the
// `singlet atac` subcommand (re-run from a kept BAM).
//
// Outputs in out_dir:
//   fragments.tsv.gz (+ .tbi)           chrom, start, end, barcode, read_pairs (bgzip, Cell Ranger layout)
//   peaks.bed                           insertion peaks (atac_peak_caller.h)
//   atac_barcode_metrics.tsv            per barcode with >= 1 fragment (atac_metrics.h definitions)
//   atac_peak_counts.mtx.gz             cut sites, peaks x columns; _features.tsv.gz (chr:start-end), _barcodes.tsv.gz
//   atac_gene_activity.mtx.gz           fragments, protein-coding genes x columns; _features / _barcodes
//   atac_cells.tsv                      called barcodes
//   atac_summary.json                   sample metrics, thresholds and pass/fail
// Columns of the matrices are the barcodes with >= min_column_fragments fragments or called, sorted by barcode.

#include <htslib/bgzf.h>
#include <htslib/sam.h>
#include <htslib/tbx.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "atac_barcodes.h"
#include "atac_cell_caller.h"
#include "atac_fragment.h"
#include "atac_metrics.h"
#include "atac_peak_caller.h"
#include "mtx_writer.h"
#include "stall_watchdog.h"

namespace singlet {

struct AtacRunOptions {
    std::string bam;
    std::string out_dir;
    std::string gtf;
    std::vector<std::string> whitelists;  // candidates (plain list or "atac<TAB>gex" pairs); best hit rate wins
    std::string paired_gex_dir;           // Multiome: GEX run output (cell_calls.tsv) for the joint cell call
    uint32_t min_column_fragments = 100;
    uint32_t min_cell_peak_fragments = 100;  // floor on fragments in peaks for a called cell
    int32_t max_fragment = 2000;
    InsertionPeakParams peaks;
};

// Sample-level QC thresholds (pass/fail). Sources: Cell Ranger ATAC/ARC web-summary alerts, ENCODE ATAC standards,
// Signac/ArchR vignettes. See docs/atac-spatial-benchmark.md.
struct AtacQcThresholds {
    double min_valid_barcode_frac = 0.75;  // MAPQ>=30 pairs whose barcode resolves (Cell Ranger warns < 0.85)
    double min_median_tss_enrichment = 4.0;  // ArchR default cell filter; ENCODE bulk TSS < 5 concerning
    double min_frip_cells = 0.20;            // fragments in peaks over called cells' fragments (10x alert < 0.20)
    double min_frac_fragments_in_cells = 0.30;  // 10x alert < 0.30
    uint32_t min_cells = 100;
    double max_median_nucleosome_signal = 4.0;  // Signac vignette cut
};

namespace atac_detail {

inline std::string json_num(double x) {
    if (!std::isfinite(x)) return "null";
    std::ostringstream o;
    o.precision(6);
    o << x;
    return o.str();
}

inline double median(std::vector<double> v) {
    if (v.empty()) return 0;
    const size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(m), v.end());
    double hi = v[m];
    if (v.size() % 2) return hi;
    double lo = *std::max_element(v.begin(), v.begin() + static_cast<long>(m));
    return 0.5 * (lo + hi);
}

inline bool is_mito_name(const std::string& n) {
    return n == "chrM" || n == "MT" || n == "chrMT" || n == "M" || n == "mitochondrion_genome";
}

}  // namespace atac_detail

/// Returns 0 on success.
inline int run_atac_from_bam(const AtacRunOptions& opt) {
    using namespace atac_detail;
    samFile* fp = sam_open(opt.bam.c_str(), "rb");
    if (!fp) { std::cerr << "[atac] cannot open " << opt.bam << "\n"; return 1; }
    bam_hdr_t* hdr = sam_hdr_read(fp);
    std::vector<std::string> chrom_names;
    std::vector<int32_t> chrom_len;
    std::vector<bool> is_mito;
    std::unordered_map<std::string, int32_t> chrom_map;
    for (int i = 0; i < hdr->n_targets; ++i) {
        chrom_names.emplace_back(hdr->target_name[i]);
        chrom_len.push_back(static_cast<int32_t>(hdr->target_len[i]));
        is_mito.push_back(is_mito_name(chrom_names.back()));
        chrom_map[chrom_names.back()] = i;
    }

    // ── Whitelist candidates and layout detection on a hash sample of the fragment records (A2) ──
    std::vector<std::unique_ptr<AtacBarcodeCorrector>> cands;
    for (const auto& w : opt.whitelists) {
        auto c = std::make_unique<AtacBarcodeCorrector>();
        if (c->load_whitelist(w)) {
            std::cerr << "[atac] whitelist candidate " << w << ": " << c->size() << " barcodes"
                      << (c->has_pairs() ? " (ATAC-GEX pairs)" : "") << "\n";
            cands.push_back(std::move(c));
        } else {
            std::cerr << "[atac] WARNING: cannot read whitelist " << w << "\n";
        }
    }
    if (cands.empty()) { std::cerr << "[atac] ERROR: no whitelist\n"; return 1; }

    bam1_t* rec = bam_init1();
    // Rule A2 (docs/research-beta/atac-reducers-prep.md): layout and whitelist are chosen on the records with
    // qname_hash % 64 == 0, all of them (not the first N of a coordinate-sorted BAM); the prior counts every record. The pass
    // keeps the sample and the raw barcode counts, then decides once at the end, so the result is a function of the set of records.
    std::vector<std::string> first;   // hash-sampled raw barcodes
    std::unordered_map<std::string, uint64_t> raw_counts;
    uint64_t pass1_records = 0;
    AtacBarcodeCorrector* cor = nullptr;
    std::vector<std::string> wl_names;
    for (const auto& w : opt.whitelists) {
        AtacBarcodeCorrector probe;
        if (probe.load_whitelist(w)) wl_names.push_back(w);
    }
    std::string chosen_wl;
    auto choose = [&]() {
        double best = -1;
        size_t bi = 0;
        for (size_t i = 0; i < cands.size(); ++i) {
            for (const auto& r : first) cands[i]->observe(r);
            auto lay = cands[i]->detect_layout();
            std::cerr << "[atac] layout on " << first.size() << " hash-sampled reads, whitelist " << i << ": offset " << lay.offset
                      << (lay.revcomp ? " revcomp" : " forward") << ", exact hit rate " << lay.hit_rate << "\n";
            if (lay.hit_rate > best) { best = lay.hit_rate; bi = i; }   // ties: the earlier whitelist
        }
        cor = cands[bi].get();
        chosen_wl = wl_names[bi];
        for (const auto& [raw, n] : raw_counts) cor->count(raw, n);   // integer adds: iteration order does not matter
        first.clear();
        raw_counts.clear();
        std::cerr << "[atac] using whitelist " << bi << "\n";
    };
    while (sam_read1(fp, hdr, rec) >= 0) {
        if (!ATACFragmentExtractor::is_fragment_record(rec)) continue;
        auto raw = ATACFragmentExtractor::raw_barcode(rec);
        if (raw.empty()) continue;
        ++pass1_records;
        ++raw_counts[std::string(raw)];
        if (AtacBarcodeCorrector::is_layout_sample(bam_get_qname(rec))) first.emplace_back(raw);
    }
    if (raw_counts.empty()) { std::cerr << "[atac] ERROR: no barcoded fragment records\n"; return 1; }
    choose();
    bam_hdr_destroy(hdr);
    sam_close(fp);

    // ── Pass 2: fragments ──
    fp = sam_open(opt.bam.c_str(), "rb");
    hdr = sam_hdr_read(fp);
    ATACFragmentExtractor ex;
    ex.set_barcode_corrector(cor);
    ex.set_max_fragment_size(opt.max_fragment);
    while (sam_read1(fp, hdr, rec) >= 0) ex.process_record(rec, hdr);
    bam_destroy1(rec);
    bam_hdr_destroy(hdr);
    sam_close(fp);
    ex.finalise();
    const auto& frags = ex.fragments();
    const auto& cs = cor->stats();
    std::cerr << "[atac] fragments: " << ex.unique_fragments() << " unique, " << ex.duplicate_fragments()
              << " duplicate read pairs; barcodes exact " << cs.exact << ", corrected " << cs.corrected
              << ", ambiguous " << cs.ambiguous << ", unmatched " << cs.unmatched + cs.invalid << "\n";

    const size_t nbc = cor->size();
    std::vector<std::string> out_bc(nbc);
    for (size_t i = 0; i < nbc; ++i) out_bc[i] = cor->output_barcode(static_cast<uint32_t>(i));

    // ── fragments.tsv.gz (bgzip + tabix) ──
    {
        const std::string path = opt.out_dir + "/fragments.tsv.gz";
        BGZF* bz = bgzf_open(path.c_str(), "w");
        if (bz) {
            std::string buf;
            buf.reserve(1 << 20);
            char line[512];
            for (const auto& f : frags) {
                int n = std::snprintf(line, sizeof(line), "%s\t%d\t%d\t%s\t%u\n",
                                      chrom_names[static_cast<size_t>(f.chrom_idx)].c_str(), f.start, f.end,
                                      out_bc[f.barcode_idx].c_str(), f.read_pairs);
                buf.append(line, static_cast<size_t>(n));
                if (buf.size() > (1u << 20) - 512) { if (bgzf_write(bz, buf.data(), buf.size()) < 0) break; buf.clear(); }
            }
            if (!buf.empty() && bgzf_write(bz, buf.data(), buf.size()) < 0) std::cerr << "[atac] ERROR: fragments write failed\n";
            bgzf_close(bz);
            if (tbx_index_build(path.c_str(), 0, &tbx_conf_bed) != 0)
                std::cerr << "[atac] WARNING: tabix index failed\n";
        }
    }

    // ── Peaks on all whitelisted fragments ──
    std::vector<std::vector<int32_t>> ins(chrom_names.size());
    for (const auto& f : frags) {
        const size_t c = static_cast<size_t>(f.chrom_idx);
        if (is_mito[c]) continue;
        ins[c].push_back(f.start);
        ins[c].push_back(f.end - 1);
    }
    auto peaks = call_peaks_insertions(ins, chrom_len, is_mito, opt.peaks);
    { std::vector<std::vector<int32_t>>().swap(ins); }
    write_insertion_peaks_bed(opt.out_dir + "/peaks.bed", peaks, chrom_names);
    singlet::rearm_stall_watchdog();   // each ATAC phase is progress (a 10k library takes > 30 min in total)
    std::cerr << "[atac] peaks: " << peaks.size() << "\n";

    // ── Annotation, metrics, cell calls ──
    AtacAnnotation ann;
    if (!opt.gtf.empty() && !ann.load_gtf(opt.gtf, chrom_map, chrom_names.size()))
        std::cerr << "[atac] WARNING: cannot read GTF " << opt.gtf << " (no TSS / gene activity)\n";
    if (ann.tss.empty()) ann.tss.assign(chrom_names.size(), {});
    auto met = compute_atac_metrics(frags, nbc, is_mito, ann, peaks);

    std::vector<uint32_t> peak_frag(nbc);
    for (size_t i = 0; i < nbc; ++i) peak_frag[i] = met[i].peak_region_fragments;
    std::vector<uint8_t> is_cell;
    MixtureCellCall call;
    bool joint = false;
    if (!opt.paired_gex_dir.empty()) {
        std::ifstream in(opt.paired_gex_dir + "/cell_calls.tsv");
        std::string line;
        std::unordered_map<std::string, uint32_t> gex;
        if (in && std::getline(in, line)) {
            std::vector<std::string> h;
            std::stringstream hs(line);
            std::string t;
            while (std::getline(hs, t, '\t')) h.push_back(t);
            int ub = -1;
            for (size_t k = 0; k < h.size(); ++k) if (h[k] == "umi_total") ub = static_cast<int>(k);
            while (ub >= 0 && std::getline(in, line)) {
                std::stringstream ls(line);
                std::vector<std::string> f;
                while (std::getline(ls, t, '\t')) f.push_back(t);
                if (static_cast<int>(f.size()) > ub) gex[f[0]] = static_cast<uint32_t>(std::strtoul(f[ub].c_str(), nullptr, 10));
            }
        }
        if (!gex.empty()) {
            std::vector<uint32_t> g(nbc, 0);
            size_t hit = 0;
            for (size_t i = 0; i < nbc; ++i) {
                auto it = gex.find(out_bc[i]);
                if (it != gex.end()) { g[i] = it->second; ++hit; }
            }
            std::cerr << "[atac] joint call with GEX: " << gex.size() << " GEX barcodes, " << hit << " paired\n";
            call = call_cells_mixture({g, peak_frag}, is_cell, 10, opt.min_cell_peak_fragments);
            joint = true;
        } else {
            std::cerr << "[atac] WARNING: no umi_total in " << opt.paired_gex_dir << "/cell_calls.tsv; ATAC-only call\n";
        }
    }
    if (!joint) call = call_cells_mixture({peak_frag}, is_cell, 10, opt.min_cell_peak_fragments);
    std::cerr << "[atac] cells: " << call.n_cells << " of " << call.n_candidates << " candidates"
              << (joint ? " (joint GEX+ATAC)" : "") << "\n";

    // ── Columns and matrices ──
    std::vector<int32_t> order;
    for (size_t i = 0; i < nbc; ++i)
        if (met[i].fragments >= opt.min_column_fragments || is_cell[i]) order.push_back(static_cast<int32_t>(i));
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return out_bc[a] < out_bc[b]; });
    std::vector<int32_t> col_of(nbc, -1);
    std::vector<std::string> col_names;
    for (size_t k = 0; k < order.size(); ++k) {
        col_of[static_cast<size_t>(order[k])] = static_cast<int32_t>(k);
        col_names.push_back(out_bc[static_cast<size_t>(order[k])]);
    }
    singlet::rearm_stall_watchdog();
    AtacTriplets pm, gm;
    compute_atac_metrics(frags, nbc, is_mito, ann, peaks, &col_of, &pm, &gm);
    const uint32_t ncol = static_cast<uint32_t>(order.size());
    {
        std::vector<int32_t> ip, ix;
        std::vector<uint32_t> d;
        pm.to_csc(ncol, ip, ix, d);
        write_mtx(opt.out_dir + "/atac_peak_counts.mtx.gz", static_cast<uint32_t>(peaks.size()), ncol, ip, ix, d);
        std::vector<std::string> pn;
        for (const auto& p : peaks)
            pn.push_back(chrom_names[static_cast<size_t>(p.chrom_idx)] + ":" + std::to_string(p.start) + "-" +
                         std::to_string(p.end));
        write_names(opt.out_dir + "/atac_peak_counts_features.tsv.gz", pn);
        write_names(opt.out_dir + "/atac_peak_counts_barcodes.tsv.gz", col_names);
        gm.to_csc(ncol, ip, ix, d);
        write_mtx(opt.out_dir + "/atac_gene_activity.mtx.gz", static_cast<uint32_t>(ann.gene_names.size()), ncol, ip,
                  ix, d);
        write_names(opt.out_dir + "/atac_gene_activity_features.tsv.gz", ann.gene_names);
        write_names(opt.out_dir + "/atac_gene_activity_barcodes.tsv.gz", col_names);
    }

    singlet::rearm_stall_watchdog();
    // ── Per-barcode metrics and cells ──
    std::vector<int32_t> any;
    for (size_t i = 0; i < nbc; ++i) if (met[i].read_pairs > 0) any.push_back(static_cast<int32_t>(i));
    std::sort(any.begin(), any.end(), [&](int32_t a, int32_t b) { return out_bc[a] < out_bc[b]; });
    write_atac_metrics_tsv(opt.out_dir + "/atac_barcode_metrics.tsv", met, out_bc, any, is_cell);
    {
        std::ofstream c(opt.out_dir + "/atac_cells.tsv");
        c << "barcode\tatac_barcode\n";
        for (int32_t i : any) if (is_cell[static_cast<size_t>(i)]) c << out_bc[i] << '\t' << cor->atac_barcode(i) << '\n';
    }

    // ── Sample summary and QC verdict ──
    AtacQcThresholds th;
    uint64_t rp_all = 0, rp_valid = 0, frag_all = 0, frag_cells = 0, peak_cells = 0;
    std::vector<double> tsse, ns, fr, nf;
    for (size_t i = 0; i < nbc; ++i) {
        rp_valid += met[i].read_pairs;
        frag_all += met[i].fragments;
        if (!is_cell[i]) continue;
        frag_cells += met[i].fragments;
        peak_cells += met[i].peak_region_fragments;
        tsse.push_back(met[i].tss_enrichment());
        ns.push_back(met[i].nucleosome_signal());
        fr.push_back(met[i].frip());
        nf.push_back(met[i].fragments);
    }
    rp_all = pass1_records;
    const double valid = rp_all ? static_cast<double>(cs.exact + cs.corrected) / static_cast<double>(cs.reads) : 0;
    const double med_tsse = median(tsse), med_ns = median(ns), frip_cells = frag_cells ? double(peak_cells) / frag_cells : 0;
    const double in_cells = frag_all ? double(frag_cells) / frag_all : 0;
    struct Rule { const char* name; double value; double thr; bool ge; };
    std::vector<Rule> rules = {
        {"valid_barcode_fraction", valid, th.min_valid_barcode_frac, true},
        {"median_tss_enrichment", med_tsse, th.min_median_tss_enrichment, true},
        {"frip_cells", frip_cells, th.min_frip_cells, true},
        {"fraction_fragments_in_cells", in_cells, th.min_frac_fragments_in_cells, true},
        {"cells", static_cast<double>(call.n_cells), static_cast<double>(th.min_cells), true},
        {"median_nucleosome_signal", med_ns, th.max_median_nucleosome_signal, false},
    };
    bool pass = true;
    std::ostringstream js;
    js << "{\n  \"schema\": \"singlet-atac-summary/1\",\n  \"whitelist\": \"" << chosen_wl << "\",\n";
    js << "  \"barcode_layout\": {\"offset\": " << cor->layout().offset << ", \"revcomp\": "
       << (cor->layout().revcomp ? "true" : "false") << ", \"exact_hit_rate\": " << json_num(cor->layout().hit_rate)
       << ", \"paired_gex\": " << (cor->has_pairs() ? "true" : "false") << "},\n";
    js << "  \"fragment_read_pairs\": " << rp_all << ",\n  \"barcode_exact\": " << cs.exact << ",\n  \"barcode_corrected\": "
       << cs.corrected << ",\n  \"barcode_unresolved\": " << (cs.ambiguous + cs.unmatched + cs.invalid) << ",\n";
    js << "  \"unique_fragments\": " << ex.unique_fragments() << ",\n  \"duplicate_read_pairs\": " << ex.duplicate_fragments()
       << ",\n  \"low_mapq_pairs\": " << ex.low_mapq_reads() << ",\n  \"peaks\": " << peaks.size() << ",\n";
    js << "  \"cells\": " << call.n_cells << ",\n  \"cell_call\": {\"method\": \"" << (joint ? "mixture-gex-atac" : "mixture-atac")
       << "\", \"candidates\": " << call.n_candidates << ", \"weight_cell\": " << json_num(call.weight_cell)
       << ", \"threshold_log10_peak_fragments\": " << (joint ? std::string("null") : json_num(call.threshold_log10))
       << "},\n";
    js << "  \"median_fragments_per_cell\": " << json_num(median(nf)) << ",\n  \"median_frip\": " << json_num(median(fr))
       << ",\n  \"columns\": " << ncol << ",\n  \"qc\": {\n";
    for (size_t k = 0; k < rules.size(); ++k) {
        const auto& r = rules[k];
        const bool ok = r.ge ? r.value >= r.thr : r.value <= r.thr;
        pass = pass && ok;
        js << "    \"" << r.name << "\": {\"value\": " << json_num(r.value) << ", \"threshold\": " << json_num(r.thr)
           << ", \"rule\": \"" << (r.ge ? ">=" : "<=") << "\", \"pass\": " << (ok ? "true" : "false") << "},\n";
    }
    js << "    \"pass\": " << (pass ? "true" : "false") << "\n  }\n}\n";
    std::ofstream(opt.out_dir + "/atac_summary.json") << js.str();
    std::cerr << "[atac] QC " << (pass ? "PASS" : "FAIL") << ": cells " << call.n_cells << ", median TSSe " << med_tsse
              << ", FRiP(cells) " << frip_cells << ", in-cells " << in_cells << "\n";
    return 0;
}

}  // namespace singlet
