// SPDX-License-Identifier: MIT
// arriba_cascade_cli.cpp - runs Arriba's filter and scoring cascade (include/singlet/pileup/arriba_cascade.h) on one sample's chimeric reads.
// Derived from Arriba 2.5.1 (arriba.cpp: main), MIT license; see NOTICE.
//
// Build: g++ -O2 -std=c++17 -Iinclude src/pipeline/arriba_cascade_cli.cpp -lz -o arriba_cascade
// Usage: arriba_cascade --junction J --labels L --gtf G --genome FA [--fastq R2.fq[.gz] ...] [--gene-counts gene_expression.tsv]
//                       [--star-log Log.final.out | --mapped-reads N] [--blacklist B] --out PREFIX
// Writes PREFIX.fusions.tsv.gz (calls in Arriba's order), PREFIX.discarded.tsv.gz, PREFIX.log.tsv (remaining after each step) and
// PREFIX.reads.tsv (reads by the filter that removed them).

#include <cstdio>
#include <fstream>
#include <iostream>

#include "singlet/pileup/arriba_cascade.h"

using namespace singlet::arriba;

static unsigned long star_mapped_reads(const std::string& path) {
    unsigned long n = 0;
    detail::for_each_line(path, [&](const std::string& ln) {
        auto bar = ln.find('|');
        if (bar == std::string::npos) return;
        std::string key = ln.substr(0, bar);
        if (key.find("Uniquely mapped reads number") != std::string::npos || key.find("Number of reads mapped to multiple loci") != std::string::npos)
            n += std::strtoul(ln.substr(bar + 1).c_str(), nullptr, 10);
    });
    return n;
}

int main(int argc, char** argv) {
    config_t cfg;
    std::string labels, out, star_log;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); } return argv[++i]; };
        if (a == "--junction") cfg.input.junction_path = next();
        else if (a == "--labels") labels = next();
        else if (a == "--fastq") cfg.input.fastq_paths.push_back(next());
        else if (a == "--gtf") cfg.gtf_path = next();
        else if (a == "--genome") cfg.assembly_path = next();
        else if (a == "--gene-counts") cfg.gene_counts_path = next();
        else if (a == "--star-log") star_log = next();
        else if (a == "--mapped-reads") cfg.mapped_reads = std::strtoul(next().c_str(), nullptr, 10);
        else if (a == "--blacklist") cfg.options.blacklist_path = next();
        else if (a == "--out") out = next();
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    if (cfg.input.junction_path.empty() || labels.empty() || cfg.gtf_path.empty() || out.empty()) {
        std::cerr << "need --junction, --labels, --gtf and --out\n";
        return 2;
    }
    detail::for_each_line(labels, [&](const std::string& ln) {
        auto t = ln.find('\t');
        std::string bc = ln.substr(0, t);
        if (!bc.empty() && bc != "barcode") cfg.input.barcodes.push_back(bc);
    });
    if (!star_log.empty()) cfg.mapped_reads = star_mapped_reads(star_log);
    result_t r;
    run(cfg, r);
    if (!r.ok) { std::cerr << "error: " << r.error << "\n"; return 1; }
    std::ofstream lg(out + ".log.tsv");
    lg << "step\tremaining\n";
    lg << "junction_lines\t" << r.load.lines << "\n";
    lg << "no_barcode\t" << r.load.no_barcode << "\n" << "barcode_unmatched\t" << r.load.unmatched << "\n" << "barcode_ambiguous\t" << r.load.ambiguous << "\n";
    lg << "malformed\t" << r.load.malformed << "\n" << "with_sequence\t" << r.load.with_sequence << "\n";
    lg << "strandedness\t" << (r.strandedness == STRANDEDNESS_YES ? "yes" : r.strandedness == STRANDEDNESS_REVERSE ? "reverse" : "no") << "\n";
    lg << "mapped_reads\t" << cfg.mapped_reads << "\n";
    lg << "gene_counts_loaded\t" << r.coverage.count_by_gene_id.size() << "\n";
    for (auto& s : r.log) lg << s.first << "\t" << s.second << "\n";
    std::map<std::string, unsigned long> by_filter;
    for (auto& kv : r.alignments) by_filter[kv.second.filter == FILTER_none ? "passed" : filter_name(kv.second.filter)]++;
    std::ofstream rd(out + ".reads.tsv");
    rd << "filter\treads\n";
    for (auto& kv : by_filter) rd << kv.first << "\t" << kv.second << "\n";
    std::map<std::string, unsigned long> fu_by_filter;
    for (auto& kv : r.fusions) fu_by_filter[kv.second.filter == FILTER_none ? std::string("passed_") + confidence_name(kv.second.confidence) : filter_name(kv.second.filter)]++;
    std::ofstream fb(out + ".fusion_filters.tsv");
    fb << "filter\tfusions\n";
    for (auto& kv : fu_by_filter) fb << kv.first << "\t" << kv.second << "\n";
    if (!write_fusions(r, out + ".fusions.tsv.gz", false) || !write_fusions(r, out + ".discarded.tsv.gz", true)) { std::cerr << "cannot write output\n"; return 1; }
    std::cerr << "ok: " << r.alignments.size() << " chimeric reads, " << r.fusions.size() << " candidate fusions\n";
    return 0;
}
