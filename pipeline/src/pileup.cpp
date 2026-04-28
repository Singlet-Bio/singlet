// singlet-pileup: main.cpp
// CLI entry point for the streaming BAM pileup engine (Pipeline V2).
//
// Usage:
//   singlet-pileup [options] <input.bam>
//   samtools view -h input.bam | singlet-pileup [options] -
//
// Options:
//   --barcodes FILE    Filtered barcode list (one per line)
//   --snps FILE        VCF/TSV of SNP positions to genotype
//   --exons FILE       GTF/BED for exon-level counting (GTF enables gene model)
//   --out-prefix DIR   Output prefix for all matrices
//   --out-chrm FILE    Output chrM BAM path
//   --threads N        Number of BAM decompression threads (default: 4)
//   --min-mapq N       Minimum mapping quality (default: 20)
//   --min-baseq N      Minimum base quality for SNP (default: 10)
//   --no-introns       Disable intron counting
//   --no-sj            Disable splice junction extraction
//   --no-umi-dedup     Disable UMI deduplication

#include <sys/stat.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "singlet-pileup/export.h"
#include "singlet-pileup/pileup_engine.h"

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <input.bam | ->\n\n"
              << "Streaming single-cell BAM pileup engine (Pipeline V2).\n"
              << "Extracts per-exon/intron counts, splice junctions, SNP genotypes,\n"
              << "and chrM pileup in a single pass.\n\n"
              << "Required:\n"
              << "  --barcodes FILE    Filtered barcode list (one per line)\n\n"
              << "Feature layers (at least one required):\n"
              << "  --snps FILE        VCF/TSV of SNP positions\n"
              << "  --exons FILE       GTF/BED for gene model\n\n"
              << "Output:\n"
              << "  --out-prefix DIR   Output directory prefix [./pileup_out]\n"
              << "  --out-chrm FILE    Output chrM BAM (optional)\n\n"
              << "Parameters:\n"
              << "  --threads N        BAM decompression threads [4]\n"
              << "  --min-mapq N       Minimum MAPQ [20]\n"
              << "  --min-baseq N      Minimum base quality [10]\n"
              << "  --no-introns       Disable intron counting\n"
              << "  --no-sj            Disable splice junction extraction\n"
              << "  --no-umi-dedup     Disable UMI deduplication\n"
              << "  --umi-dedup-directional  Enable N6 directional 1-Hamming UMI correction\n"
              << "  --workers N        Parallel workers (region-parallel BAM, needs .bai) [1]\n"
              << "  --output-format F  Output format: 1pz (default) or mtx\n"
              << "  --pipeline         Enable inline donor demux + mt heteroplasmy\n"
              << "  --n-donors N       Number of donors for demux (-1=auto) [-1]\n"
              << "  -h, --help         Show this help\n";
}

static bool mkdir_p(const std::string& path) {
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) return true;
    auto pos = path.rfind('/');
    if (pos != std::string::npos && pos > 0) {
        mkdir_p(path.substr(0, pos));
    }
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

int main(int argc, char* argv[]) {
    singlet::PileupConfig config;
    std::string bam_path = "-";
    std::string out_prefix = "./pileup_out";
    int n_workers = 1;
    std::string output_format = "1pz";
    bool pipeline_mode = false;
    int n_donors = -1;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else if (arg == "--barcodes" && i + 1 < argc) {
            config.barcode_path = argv[++i];
        } else if (arg == "--snps" && i + 1 < argc) {
            config.snp_path = argv[++i];
        } else if (arg == "--exons" && i + 1 < argc) {
            config.exon_gtf_path = argv[++i];
        } else if (arg == "--out-prefix" && i + 1 < argc) {
            out_prefix = argv[++i];
        } else if (arg == "--out-chrm" && i + 1 < argc) {
            config.out_chrm_bam = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            config.threads = std::atoi(argv[++i]);
        } else if (arg == "--min-mapq" && i + 1 < argc) {
            config.min_mapq = std::atoi(argv[++i]);
        } else if (arg == "--min-baseq" && i + 1 < argc) {
            config.min_baseq = std::atoi(argv[++i]);
        } else if (arg == "--no-introns") {
            config.count_introns = false;
        } else if (arg == "--no-sj") {
            config.count_sj = false;
        } else if (arg == "--no-umi-dedup") {
            config.umi_dedup = false;
        } else if (arg == "--umi-dedup-directional") {
            config.umi_dedup_directional = true;
        } else if (arg == "--workers" && i + 1 < argc) {
            n_workers = std::atoi(argv[++i]);
            if (n_workers < 1) n_workers = 1;
        } else if (arg == "--output-format" && i + 1 < argc) {
            output_format = argv[++i];
        } else if (arg == "--pipeline") {
            pipeline_mode = true;
        } else if (arg == "--n-donors" && i + 1 < argc) {
            n_donors = std::atoi(argv[++i]);
        } else if (arg[0] != '-' || arg == "-") {
            bam_path = arg;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    // Validate
    if (config.barcode_path.empty()) {
        std::cerr << "ERROR: --barcodes is required\n";
        usage(argv[0]);
        return 1;
    }

    bool has_layer = !config.snp_path.empty() || !config.exon_gtf_path.empty();
    if (!has_layer) {
        std::cerr << "ERROR: At least one feature layer is required "
                     "(--snps or --exons)\n";
        usage(argv[0]);
        return 1;
    }

    // Set up output paths
    mkdir_p(out_prefix);

    // Enable mt pileup in pipeline mode
    if (pipeline_mode) config.count_mt = true;

    // Run engine
    std::cerr << "singlet-pileup v0.2.0 (Pipeline V2)\n";
    std::cerr << "Input: " << bam_path << "\n";
    std::cerr << "Output: " << out_prefix << "\n";
    std::cerr << "Threads: " << config.threads << "\n";
    std::cerr << "Workers: " << n_workers << (n_workers > 1 ? " (region-parallel)" : " (sequential)") << "\n";
    std::cerr << "Output format: " << output_format << "\n";
    if (pipeline_mode) std::cerr << "Pipeline mode: ON (demux + mt)\n";
    std::cerr << "UMI dedup: " << (config.umi_dedup ? "ON" : "OFF");
    if (config.umi_dedup && config.umi_dedup_directional) std::cerr << " (directional N6)";
    std::cerr << "\n";
    std::cerr << "Introns: " << (config.count_introns ? "ON" : "OFF") << "\n";
    std::cerr << "Splice junctions: " << (config.count_sj ? "ON" : "OFF") << "\n";

    singlet::PileupEngine engine(config);

    if (!engine.load_references()) {
        std::cerr << "ERROR: Failed to load references\n";
        return 1;
    }

    auto stats = (n_workers > 1)
                     ? engine.run_parallel(bam_path, n_workers)
                     : engine.run(bam_path);

    if (stats.total_reads == 0) {
        std::cerr << "ERROR: No reads processed. Skipping export.\n";
        return 1;
    }

    // Print summary stats as JSON
    singlet::write_stats_json(out_prefix + "/pileup_stats.json",
                              stats, "0.2.0", stats.wall_time_s);
    std::cerr << "Stats written to: " << out_prefix << "/pileup_stats.json\n";

    // ── Export matrices ──
    singlet::ExportConfig export_cfg;
    export_cfg.out_prefix = out_prefix;
    export_cfg.output_format = output_format;
    export_cfg.pipeline_mode = pipeline_mode;
    export_cfg.n_donors = n_donors;
    export_cfg.threads = config.threads;

    auto export_stats = singlet::export_results(engine, config, export_cfg);

    std::cerr << "All outputs written to: " << out_prefix << "\n";
    return 0;
}
