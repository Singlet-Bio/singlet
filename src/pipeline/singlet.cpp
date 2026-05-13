// singlet: .1fq → align → pileup → .1pz
//
// Self-contained monorepo binary.  The aligner (STAR, vendored in src/star/)
// is compiled in as a CMake OBJECT library.  In processing
// mode, singlet forks a child that calls star_main_impl(), which writes
// unsorted BAM to a pipe; the parent reads the pipe through the
// singlet-pileup engine and exports .1pz matrices.
//
// .1fq is the ONLY production input format.  Raw data (SRA, FASTQ) must
// first be archived to .1fq via `singlet download` or `singlet encode`,
// then processed with `singlet process` (or the bare `singlet <file.1fq>`
// shorthand).  Legacy --reads and --sra flags still work but print
// deprecation warnings.
//
// Pipeline stages:
//   1. .1fq reader → decode blocks (byte-numeric, pre-parsed BC/UMI)
//   2. Aligner (bundled STAR, forked child, star_main_impl)
//   3. Pileup engine (parent, streaming BAM pipe)
//   4. Export → .1pz
//
// Performance notes:
//   - STAR --outBAMcompression 0 eliminates gzip encode/decode overhead
//   - SAM pipe has zero overhead vs file-based sequential (profiled: 26.7s == 26.7s)
//   - Pileup processes faster than STAR produces → alignment is the bottleneck
//   - Total wall time ≈ STAR alignment time + pileup export time (~3s)

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <algorithm>
#include <fcntl.h>
#include <fstream>
#include <zlib.h>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <filesystem>
#include "singlet/pileup/pileup_engine.h"
#include "singlet/pileup/export.h"
#include "singlet/pileup/species_detect.h"
#include "singlet/pileup/nonhost/min_sketch.h"
#include "singlet/pileup/nonhost/host_kmer_filter.h"
#include "singlet/pileup/nonhost/nonhost_screener.h"
#include "singlet/pileup/nonhost/nonhost_em.h"
#include "singlet/pileup/nonhost/nonhost_aligner.h"
#include "singlet/pileup/nonhost/nonhost_cell_matrix.h"
#include "singlet/pileup/tiny_dataset_guard.h"
#include "singlet/pileup/mega_sort_params.h"
#include "singlet/pileup/atac_fragment.h"
#include "singlet/pileup/atac_bin_counter.h"
#include "singlet/pileup/atac_peaks.h"
#include "singlet/pileup/atac_qc.h"
#include "singlet/pileup/atac_cell_caller.h"
#include "singlet/pileup/adt_matcher.h"
#include "singlet/pileup/adt_counter.h"
#include "singlet/pileup/hto_demux.h"
#include "singlet/pileup/visium_spatial.h"
#include "singlet/pileup/cascade_router.h"
#include "singlet/pileup/cascade_stats_writer.h"
#include "singlet/pileup/visium_qc.h"
#include "singlet/pileup/rna_variant_caller.h"
#include "singlet/pileup/read_dedup_stats.h"
#include "singlet/pileup/tagged_bam.h"
#include "singlet/fq/sra_reader.h"
#include "singlet/fq/lib1fq.h"
#include "singlet/fq/sra_encoder.h"
#include "singlet/fq/fastq_encoder.h"
#include "singlet/star/star_api.h"  // star_main_impl() — bundled aligner

// ── Timing helper ──
struct StopWatch {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t0;
    StopWatch() : t0(Clock::now()) {}
    double elapsed_s() const {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }
};

static bool mkdir_p(const std::string& path) {
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) return true;
    auto pos = path.rfind('/');
    if (pos != std::string::npos && pos > 0) mkdir_p(path.substr(0, pos));
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

// ════════════════════════════════════════════════════════════
// singlet download — Stream SRA → .1fq (no alignment, no temp files)
// ════════════════════════════════════════════════════════════

static int cmd_download(int argc, char* argv[]) {
    std::string sra_input;    // .sra path or SRR accession
    std::string output_path;
    std::string protocol_tag;
    std::string qual_str  = "none";   // alignment-only: quality unused by STAR CB matching
    std::string codec_str = "zstd";
    int    codec_level = 4;
    uint32_t block_size = 500000;
    uint16_t r2_maxlen = 0;
    bool   no_dedup = true;   // streaming = no dedup by default
    bool   no_trim  = false;
    bool   sort_by_bc = true;
    bool   show_profile = true;
    int    vdb_threads = 4;
    uint64_t max_reads = 0;
    std::vector<std::string> whitelist_dirs;
    std::string metadata_json_path;  // GEO metadata JSON output path
    std::string swap_reads_mode = "auto";  // B-G2-1: auto|on|off|none

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cerr
                << "singlet download — Stream SRA → .1fq archive\n\n"
                << "Usage: singlet download <SRR_accession | file.sra> [OPTIONS]\n\n"
                << "Input:\n"
                << "  Positional arg       SRR accession or local .sra file path\n\n"
                << "Output:\n"
                << "  -o, --output FILE    Output .1fq file [derived from input]\n\n"
                << "Options:\n"
                << "  --protocol TAG       Force protocol (skip auto-detect)\n"
                << "  --quality MODE       none|binned|binned2|full (default: none — STAR does not use base quality for alignment)\n"
                << "  --no-trim            Skip polyA/adapter trimming\n"
                << "  --codec NAME         zstd|lz4|none (default: zstd)\n"
                << "  --codec-level N      Codec level (default: 4)\n"
                << "  --block-size N       Reads per block (default: 500000)\n"
                << "  --r2-maxlen N        Max R2 bases to keep (0=all, default: 0)\n"
                << "  --vdb-threads N      Parallel VDB reader threads (default: 4)\n"
                << "  --max-reads N        Stop after N reads (0=all, default: 0)\n"
                << "  --whitelist-dir DIR  Search dir for whitelists (repeatable)\n"
                << "  --no-profile         Suppress profiling output\n"
                << "  --metadata-json FILE Write metadata JSON file (auto-fills singlet_version,\n"
                << "                       pipeline_date, srr_ids, protocol, read_count)\n"
                << "  --swap-reads MODE    R1\u2194R2 swap mode: auto (default)|on|off|none\n"
                << "                       auto: heuristic swap passes (geometry + whitelist)\n"
                << "                       on  : force swap regardless of detection\n"
                << "                       off/none: skip all swap passes\n";
            return 0;
        }
        else if ((arg == "-o" || arg == "--output") && i+1 < argc)
            output_path = argv[++i];
        else if (arg == "--protocol" && i+1 < argc)
            protocol_tag = argv[++i];
        else if (arg == "--quality" && i+1 < argc)
            qual_str = argv[++i];
        else if (arg == "--codec" && i+1 < argc)
            codec_str = argv[++i];
        else if (arg == "--codec-level" && i+1 < argc)
            codec_level = std::atoi(argv[++i]);
        else if (arg == "--block-size" && i+1 < argc)
            block_size = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (arg == "--r2-maxlen" && i+1 < argc)
            r2_maxlen = static_cast<uint16_t>(std::atoi(argv[++i]));
        else if (arg == "--whitelist-dir" && i+1 < argc)
            whitelist_dirs.push_back(argv[++i]);
        else if (arg == "--vdb-threads" && i+1 < argc)
            vdb_threads = std::atoi(argv[++i]);
        else if (arg == "--max-reads" && i+1 < argc)
            max_reads = static_cast<uint64_t>(std::atoll(argv[++i]));
        else if (arg == "--no-trim") no_trim = true;
        else if (arg == "--sort-by-bc") sort_by_bc = true;
        else if (arg == "--no-sort-by-bc") sort_by_bc = false;
        else if (arg == "--no-profile") show_profile = false;
        else if (arg == "--metadata-json" && i+1 < argc) metadata_json_path = argv[++i];
        else if (arg == "--swap-reads" && i+1 < argc) swap_reads_mode = argv[++i];
        else if (arg[0] != '-' && sra_input.empty())
            sra_input = arg;
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        }
    }

    if (sra_input.empty()) {
        std::cerr << "ERROR: SRA accession or .sra file required\n"
                  << "Usage: singlet download <SRRxxxxxxx | file.sra> [-o output.1fq]\n";
        return 1;
    }

    // Default output: accession.1fq
    if (output_path.empty()) {
        std::string base = sra_input;
        // Strip path and extensions
        auto slash = base.rfind('/');
        if (slash != std::string::npos) base = base.substr(slash + 1);
        auto dot = base.rfind('.');
        if (dot != std::string::npos) base = base.substr(0, dot);
        output_path = base + ".1fq";
    }

    // Parse quality mode
    lib1fq::QualMode qual_mode = lib1fq::QualMode::BINNED4;
    if (qual_str == "none") qual_mode = lib1fq::QualMode::NONE;
    else if (qual_str == "full") qual_mode = lib1fq::QualMode::FULL;
    else if (qual_str == "binned2") qual_mode = lib1fq::QualMode::BINNED2;

    // Parse codec
    lib1fq::Codec codec = lib1fq::Codec::ZSTD;
    if (codec_str == "lz4") codec = lib1fq::Codec::LZ4;
    else if (codec_str == "none") codec = lib1fq::Codec::NONE;

    // Build config
    lib1fq::EncoderConfig ecfg;
    ecfg.output_path = output_path;
    ecfg.codec       = codec;
    ecfg.codec_level = codec_level;
    ecfg.qual_mode   = qual_mode;
    ecfg.block_size  = block_size;
    ecfg.r2_maxlen   = r2_maxlen;
    ecfg.protocol_tag = (protocol_tag == "auto") ? "" : protocol_tag;
    // Auto-discover whitelist dir from binary location when user didn't pass --whitelist-dir
    if (whitelist_dirs.empty()) {
        char buf[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (len > 0) {
            buf[len] = '\0';
            std::string exe_path(buf, len);
            auto pos = exe_path.rfind('/');
            if (pos != std::string::npos) {
                std::string wl_dir = exe_path.substr(0, pos) + "/../whitelists";
                struct stat st;
                if (stat(wl_dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
                    whitelist_dirs.push_back(wl_dir);
            }
        }
    }
    ecfg.whitelist_dirs = whitelist_dirs;
    ecfg.no_dedup = no_dedup;
    ecfg.no_trim  = no_trim;
    ecfg.sort_by_bc = sort_by_bc;
    ecfg.vdb_threads = vdb_threads;
    ecfg.max_reads = max_reads;
    ecfg.progress_cb = [max_reads](uint64_t reads, uint64_t total) {
        if (max_reads > 0) total = max_reads;
        double pct = (total > 0) ? 100.0 * reads / total : 0.0;
        std::cerr << "\r[singlet-download] " << reads << " / " << total
                  << " reads (" << static_cast<int>(pct) << "%)" << std::flush;
    };
    ecfg.progress_interval = 500000;

    // Read declared read_count and catalog protocol from pre-written metadata JSON (if any).
    // The job script writes the full GEO metadata JSON before calling download;
    // we read it here so we can:
    //   (a) sanity-check the actual spot count afterward, and
    //   (b) supply the catalog-trusted protocol to the encoder so it can override
    //       a low-confidence auto-detection (AUTOFIX-PROTOCOL-CONFIDENCE-OVERRIDE).
    uint64_t declared_read_count = 0;
    std::string catalog_protocol;   // catalog-provided protocol tag (may be empty)
    if (!metadata_json_path.empty()) {
        std::ifstream pre_mf(metadata_json_path);
        if (pre_mf.is_open()) {
            std::string json_content((std::istreambuf_iterator<char>(pre_mf)),
                                      std::istreambuf_iterator<char>());
            pre_mf.close();

            // ── Extract "read_count": <integer> ──
            {
                auto pos = json_content.find("\"read_count\"");
                if (pos != std::string::npos) {
                    pos = json_content.find(':', pos);
                    if (pos != std::string::npos) {
                        ++pos;
                        while (pos < json_content.size() &&
                               (json_content[pos] == ' ' || json_content[pos] == '\t' ||
                                json_content[pos] == '\n' || json_content[pos] == '\r'))
                            ++pos;
                        uint64_t v = 0;
                        bool got_digit = false;
                        while (pos < json_content.size() &&
                               json_content[pos] >= '0' && json_content[pos] <= '9') {
                            v = v * 10 + static_cast<uint64_t>(json_content[pos] - '0');
                            ++pos;
                            got_digit = true;
                        }
                        if (got_digit) declared_read_count = v;
                    }
                }
            }

            // ── Extract "protocol" or "protocol_inferred": "<string>" ──
            // Only honoured when --protocol was NOT already set on the command line
            // (an explicit --protocol flag always wins outright via ecfg.protocol_tag).
            // Also checks "protocol_inferred" as a fallback for job scripts that use the
            // catalog field name directly (e.g. {"protocol_inferred": "dnbelab"}).
            if (protocol_tag.empty()) {
                // Try "protocol" first, then "protocol_inferred" as fallback
                for (const char* key : {"\"protocol\"", "\"protocol_inferred\""}) {
                    if (!catalog_protocol.empty()) break;
                    // Ensure we match the full key (not e.g. "protocol_id")
                    auto pos = json_content.find(key);
                    while (pos != std::string::npos) {
                        // Verify next non-whitespace char after key is ':' (not another letter)
                        size_t after_key = pos + std::string(key).size();
                        // skip any whitespace between key and ':'
                        size_t p2 = after_key;
                        while (p2 < json_content.size() &&
                               (json_content[p2] == ' ' || json_content[p2] == '\t')) ++p2;
                        if (p2 < json_content.size() && json_content[p2] == ':') {
                            // Valid key:value pair — extract string value
                            ++p2;
                            while (p2 < json_content.size() &&
                                   (json_content[p2] == ' ' || json_content[p2] == '\t' ||
                                    json_content[p2] == '\n' || json_content[p2] == '\r'))
                                ++p2;
                            if (p2 < json_content.size() && json_content[p2] == '"') {
                                ++p2;  // skip opening quote
                                while (p2 < json_content.size() && json_content[p2] != '"')
                                    catalog_protocol += json_content[p2++];
                            }
                            break;
                        }
                        // Not a valid field, skip and search again
                        pos = json_content.find(key, pos + 1);
                    }
                }
            }
        }
    }
    // Pass the catalog-trusted protocol to the encoder for low-confidence override.
    if (!catalog_protocol.empty())
        ecfg.metadata_protocol = catalog_protocol;

    // B-G2-1: pass swap_reads_mode to encoder
    ecfg.swap_reads_mode = swap_reads_mode;
    if (swap_reads_mode != "auto")
        std::cerr << "[singlet-download] swap-reads mode: " << swap_reads_mode << "\n";

    std::cerr << "[singlet-download] " << sra_input << " → " << output_path << "\n";
    if (declared_read_count > 0)
        std::cerr << "[singlet-download] Declared read count (catalog): " << declared_read_count << "\n";
    if (!catalog_protocol.empty())
        std::cerr << "[singlet-download] Catalog protocol (metadata): " << catalog_protocol << "\n";

    // Encode
    lib1fq::SraEncoder encoder;
    auto stats = encoder.encode(sra_input, ecfg);

    if (stats.exit_code != 0) return stats.exit_code;

    std::cerr << "\n[singlet-download] Done: " << stats.total_reads << " reads"
              << ", " << stats.blocks_written << " blocks\n";
    std::cerr << "[singlet-download] Protocol: " << stats.protocol_tag
              << " (confidence: " << static_cast<int>(stats.confidence) << ")\n";
    std::cerr << "[singlet-download] Output: " << output_path << "\n";

    if (show_profile) {
        stats.profile.print(stats.total_reads);
    }

    // ── Post-download sanity check ──────────────────────────────────────────
    // VDB occasionally returns a truncated/stub SRA with only a handful of
    // reads (the "1826 reads" class of failures).  Catch this here so the job
    // exits 2 immediately instead of wasting a STAR slot and producing a
    // misleading pipeline_crash failure category.
    //
    // Exit code 2 = download_fail (distinct from 1 = usage/setup error).
    {
        const uint64_t actual = stats.total_reads;

        // Unconditional absolute floor — <MIN_READS_FOR_PIPELINE reads is never
        // a real single-cell library regardless of what the catalog says.
        if (actual < singlet_pileup::TinyDatasetGuard::MIN_READS_FOR_PIPELINE) {
            std::fprintf(stderr,
                "[singlet] WARNING: Only %zu reads downloaded"
                " (minimum %zu for meaningful analysis). Aborting.\n",
                static_cast<size_t>(actual),
                static_cast<size_t>(singlet_pileup::TinyDatasetGuard::MIN_READS_FOR_PIPELINE));
            return 2;
        }

        if (declared_read_count > 0) {
            // Catalog-backed check: actual must be ≥ 10% of declared AND ≥ 100 000
            const uint64_t floor_pct  = static_cast<uint64_t>(0.10 * static_cast<double>(declared_read_count));
            const uint64_t floor_abs  = 100000ULL;
            const uint64_t floor      = std::max(floor_pct, floor_abs);
            if (actual < floor) {
                std::fprintf(stderr,
                    "[singlet-download] ERROR: VDB returned %zu spots but catalog declared "
                    "%zu reads (%.1f%%). This typically indicates VDB rate-limiting, a "
                    "corrupt/stub SRA entry, or a network truncation. "
                    "Exiting with code 2 (download_fail) to avoid wasting a STAR slot.\n",
                    static_cast<size_t>(actual),
                    static_cast<size_t>(declared_read_count),
                    actual * 100.0 / static_cast<double>(declared_read_count));
                return 2;
            }
        } else {
            // No metadata — absolute floor: <10 000 reads is never a real scRNA run
            if (actual < 10000ULL) {
                std::fprintf(stderr,
                    "[singlet-download] ERROR: VDB returned only %zu spots with no catalog "
                    "metadata to compare against. A real scRNA-seq run always has ≥10 000 reads. "
                    "Exiting with code 2 (download_fail).\n",
                    static_cast<size_t>(actual));
                return 2;
            }
        }
    }
    // ────────────────────────────────────────────────────────────────────────

    // Write metadata JSON if requested
    if (!metadata_json_path.empty()) {
        // Compute today's date as YYYY-MM-DD
        std::time_t now = std::time(nullptr);
        char date_buf[16];
        std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%d", std::localtime(&now));

        // Extract SRR accession (strip path+extension for the srr_ids field)
        std::string srr_acc = sra_input;
        {
            auto sl = srr_acc.rfind('/');
            if (sl != std::string::npos) srr_acc = srr_acc.substr(sl + 1);
            auto dt = srr_acc.rfind('.');
            if (dt != std::string::npos) srr_acc = srr_acc.substr(0, dt);
        }

        // Preserve fields from any pre-existing metadata JSON (e.g. written by
        // the job script before calling singlet download).  Crucially: organism,
        // taxon_id, gsm_id, gse_id, modality, geo_title, geo_source_name must NOT
        // be overwritten with empty values — they were populated from the catalog
        // by the job script and are critical for species auto-detection in the
        // subsequent singlet process step.
        std::string pre_organism, pre_taxon_id, pre_gsm, pre_gse;
        std::string pre_modality, pre_geo_title, pre_geo_source;
        {
            std::ifstream pre_mf2(metadata_json_path);
            if (pre_mf2.is_open()) {
                std::string pre_json((std::istreambuf_iterator<char>(pre_mf2)), {});
                // Re-use species_detect's JSON extractor (same simple logic)
                auto extract = [&](const std::string& key) -> std::string {
                    auto pos = pre_json.find("\"" + key + "\"");
                    if (pos == std::string::npos) return "";
                    pos = pre_json.find(':', pos);
                    if (pos == std::string::npos) return "";
                    pos = pre_json.find('"', pos + 1);
                    if (pos == std::string::npos) return "";
                    auto end = pre_json.find('"', pos + 1);
                    if (end == std::string::npos) return "";
                    return pre_json.substr(pos + 1, end - pos - 1);
                };
                pre_organism   = extract("organism");
                pre_taxon_id   = extract("taxon_id");
                pre_gsm        = extract("gsm_id");
                pre_gse        = extract("gse_id");
                pre_modality   = extract("modality");
                pre_geo_title  = extract("geo_title");
                pre_geo_source = extract("geo_source_name");
            }
        }

        std::ofstream mf(metadata_json_path);
        if (!mf.is_open()) {
            std::cerr << "ERROR: cannot write metadata JSON: " << metadata_json_path << "\n";
            return 1;
        }
        mf << "{\n"
           << "  \"gsm_id\": \"" << pre_gsm << "\",\n"
           << "  \"gse_id\": \"" << pre_gse << "\",\n"
           << "  \"organism\": \"" << pre_organism << "\",\n"
           << "  \"taxon_id\": " << (pre_taxon_id.empty() ? "0" : pre_taxon_id) << ",\n"
           << "  \"protocol\": \"" << stats.protocol_tag << "\",\n"
           << "  \"modality\": \"" << pre_modality << "\",\n"
           << "  \"srr_ids\": [\"" << srr_acc << "\"],\n"
           << "  \"read_count\": " << stats.total_reads << ",\n"
           << "  \"geo_title\": \"" << pre_geo_title << "\",\n"
           << "  \"geo_source_name\": \"" << pre_geo_source << "\",\n"
           << "  \"singlet_version\": \"0.3.0\",\n"
           << "  \"pipeline_date\": \"" << date_buf << "\"\n"
           << "}\n";
        mf.close();
        if (!pre_organism.empty())
            std::cerr << "[singlet-download] Metadata JSON preserved organism: "
                      << pre_organism << " (taxon " << pre_taxon_id << ")\n";
        std::cerr << "[singlet-download] Metadata JSON: " << metadata_json_path << "\n";
    }

    return 0;
}

// ════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════
// singlet encode — FASTQ pair → .1fq (archive raw data before processing)
// ════════════════════════════════════════════════════════════

static int cmd_encode(int argc, char* argv[]) {
    std::string reads_r1, reads_r2, reads_r3;  // r3 optional (ATAC 3-read)
    std::string output_path;
    std::string protocol_tag;
    std::string qual_str = "binned";
    std::string codec_str = "zstd";
    int codec_level = 4;
    uint32_t block_size = 500000;
    uint16_t r2_maxlen = 0;
    bool no_dedup = false;
    bool no_trim = false;
    bool sort_by_bc = true;
    bool verbose = false;
    std::vector<std::string> whitelist_dirs;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cerr
                << "singlet encode — Archive FASTQ pair → .1fq\n\n"
                << "Usage: singlet encode --reads R1.fq.gz R2.fq.gz -o sample.1fq\n\n"
                << "Input:\n"
                << "  --reads R1 R2        Paired FASTQ files (.gz supported)\n\n"
                << "Output:\n"
                << "  -o, --output FILE    Output .1fq file\n\n"
                << "Options:\n"
                << "  --protocol TAG       Force protocol (skip auto-detect)\n"
                << "  --quality MODE       none|binned|binned2|full (default: binned)\n"
                << "  --no-dedup           Skip PCR duplicate collapsing\n"
                << "  --no-trim            Skip polyA/adapter trimming\n"
                << "  --codec NAME         zstd|lz4|none (default: zstd)\n"
                << "  --codec-level N      Codec level (default: 4)\n"
                << "  --block-size N       Reads per block (default: 500000)\n"
                << "  --r2-maxlen N        Max R2 bases to keep (0=all)\n"
                << "  --whitelist-dir DIR  Search dir for whitelists (repeatable)\n"
                << "  --verbose, -v        Print detection diagnostics\n";
            return 0;
        }
        else if (arg == "--reads" && i+2 < argc) {
            reads_r1 = argv[++i]; reads_r2 = argv[++i];
            // Peek ahead: if 3rd arg exists and doesn't start with '-', it's R3 (ATAC)
            if (i+1 < argc && argv[i+1][0] != '-') reads_r3 = argv[++i];
        }
        else if ((arg == "-o" || arg == "--output") && i+1 < argc)
            output_path = argv[++i];
        else if (arg == "--protocol" && i+1 < argc)
            protocol_tag = argv[++i];
        else if (arg == "--quality" && i+1 < argc)
            qual_str = argv[++i];
        else if (arg == "--codec" && i+1 < argc)
            codec_str = argv[++i];
        else if (arg == "--codec-level" && i+1 < argc)
            codec_level = std::atoi(argv[++i]);
        else if (arg == "--block-size" && i+1 < argc)
            block_size = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (arg == "--r2-maxlen" && i+1 < argc)
            r2_maxlen = static_cast<uint16_t>(std::atoi(argv[++i]));
        else if (arg == "--whitelist-dir" && i+1 < argc)
            whitelist_dirs.push_back(argv[++i]);
        else if (arg == "--no-dedup") no_dedup = true;
        else if (arg == "--no-trim") no_trim = true;
        else if (arg == "--sort-by-bc") sort_by_bc = true;
        else if (arg == "--no-sort-by-bc") sort_by_bc = false;
        else if (arg == "--verbose" || arg == "-v") verbose = true;
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        }
    }

    if (reads_r1.empty()) {
        std::cerr << "ERROR: --reads R1 R2 required\n"
                  << "Usage: singlet encode --reads R1.fq.gz R2.fq.gz -o sample.1fq\n";
        return 1;
    }
    if (output_path.empty()) {
        std::string base = reads_r1;
        auto dot = base.rfind('.');
        if (dot != std::string::npos) base = base.substr(0, dot);
        auto dot2 = base.rfind('.');
        if (dot2 != std::string::npos) base = base.substr(0, dot2);
        output_path = base + ".1fq";
    }

    lib1fq::QualMode qual_mode = lib1fq::QualMode::BINNED4;
    if (qual_str == "none") qual_mode = lib1fq::QualMode::NONE;
    else if (qual_str == "full") qual_mode = lib1fq::QualMode::FULL;
    else if (qual_str == "binned2") qual_mode = lib1fq::QualMode::BINNED2;

    lib1fq::Codec codec = lib1fq::Codec::ZSTD;
    if (codec_str == "lz4") codec = lib1fq::Codec::LZ4;
    else if (codec_str == "none") codec = lib1fq::Codec::NONE;

    lib1fq::EncoderConfig ecfg;
    ecfg.output_path = output_path;
    ecfg.codec = codec;
    ecfg.codec_level = codec_level;
    ecfg.qual_mode = qual_mode;
    ecfg.block_size = block_size;
    ecfg.r2_maxlen = r2_maxlen;
    ecfg.protocol_tag = (protocol_tag == "auto") ? "" : protocol_tag;
    // Auto-discover whitelist dir from binary location when user didn't pass --whitelist-dir
    if (whitelist_dirs.empty()) {
        char buf[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (len > 0) {
            buf[len] = '\0';
            std::string exe_path(buf, len);
            auto pos = exe_path.rfind('/');
            if (pos != std::string::npos) {
                std::string wl_dir = exe_path.substr(0, pos) + "/../whitelists";
                struct stat st;
                if (stat(wl_dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
                    whitelist_dirs.push_back(wl_dir);
            }
        }
    }
    ecfg.whitelist_dirs = whitelist_dirs;
    ecfg.no_dedup = no_dedup;
    ecfg.no_trim = no_trim;
    ecfg.sort_by_bc = sort_by_bc;
    ecfg.verbose = verbose;
    ecfg.progress_cb = [](uint64_t reads, uint64_t total) {
        double pct = (total > 0) ? 100.0 * reads / total : 0.0;
        std::cerr << "\r[singlet-encode] " << reads << " / " << total
                  << " reads (" << static_cast<int>(pct) << "%)" << std::flush;
    };

    lib1fq::FastqEncoder encoder;

    if (!reads_r3.empty()) {
        // 3-read ATAC mode: R1 genomic + R2 barcode + R3 genomic
        std::cerr << "[singlet-encode] ATAC 3-read mode: "
                  << reads_r1 << " + " << reads_r2 << "(bc) + " << reads_r3
                  << " → " << output_path << "\n";
        auto stats = encoder.encode_atac(reads_r1, reads_r2, reads_r3, ecfg);
        std::cerr << "\n[singlet-encode] Done: " << stats.total_reads << " reads"
                  << ", " << stats.blocks_written << " blocks\n";
        std::cerr << "[singlet-encode] Protocol: " << stats.protocol_tag
                  << " (I2 barcode stream stored)\n";
        std::cerr << "[singlet-encode] Output: " << output_path << "\n";
    } else {
        std::cerr << "[singlet-encode] " << reads_r1 << " + " << reads_r2
                  << " → " << output_path << "\n";
        auto stats = encoder.encode(reads_r1, reads_r2, ecfg);
        std::cerr << "\n[singlet-encode] Done: " << stats.total_reads << " reads"
                  << ", " << stats.blocks_written << " blocks\n";
        std::cerr << "[singlet-encode] Protocol: " << stats.protocol_tag
                  << " (confidence: " << static_cast<int>(stats.confidence) << ")\n";
        std::cerr << "[singlet-encode] Output: " << output_path << "\n";
    }
    return 0;
}

// ════════════════════════════════════════════════════════════
// singlet decode — .1fq → FASTQ pair (standalone, no alignment)
// ════════════════════════════════════════════════════════════

static int cmd_decode(int argc, char* argv[]) {
    std::string input_path;
    std::string r1_path, r2_path;
    int threads = 4;
    bool verify = false;
    bool header_only = false;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cerr
                << "singlet decode — Decode .1fq archive to FASTQ\n\n"
                << "Usage: singlet decode <file.1fq> [OPTIONS]\n\n"
                << "Options:\n"
                << "  --r1 FILE       Output R1 FASTQ (barcode+UMI) [stdout summary]\n"
                << "  --r2 FILE       Output R2 FASTQ (cDNA)\n"
                << "  --threads N     Decode threads [4]\n"
                << "  --verify        Verify CRC32 checksums\n"
                << "  --header        Print header info only (no decode)\n"
                << "  -h, --help      Show this help\n";
            return 0;
        }
        else if (arg == "--r1" && i+1 < argc) r1_path = argv[++i];
        else if (arg == "--r2" && i+1 < argc) r2_path = argv[++i];
        else if (arg == "--threads" && i+1 < argc) threads = std::atoi(argv[++i]);
        else if (arg == "--verify") verify = true;
        else if (arg == "--header") header_only = true;
        else if (arg[0] != '-' && input_path.empty()) input_path = arg;
        else {
            std::cerr << "Unknown option: " << arg << "\n"; return 1;
        }
    }

    if (input_path.empty()) {
        std::cerr << "ERROR: no input .1fq file specified\n"; return 1;
    }

    lib1fq::Reader reader;
    reader.open(input_path.c_str());
    reader.set_verify_crc(verify);
    auto hdr = reader.header();

    // Print header info
    auto proto_name = [](uint8_t id) -> const char* {
        switch (id) {
            case 0: return "UNKNOWN"; case 1: return "10xv3"; case 2: return "10xv2";
            case 3: return "10xv1"; case 4: return "10xv3-5p"; case 5: return "10xv2-5p";
            case 6: return "dropseq"; case 7: return "celseq2"; case 8: return "marsseq2";
            case 9: return "sci-rna-seq3"; case 10: return "bd-rhapsody";
            case 11: return "splitseq"; case 12: return "indrop";
            case 13: return "10xv4"; case 14: return "10xv4-5p";
            case 15: return "dnbelab-c4"; case 16: return "seqwell";
            case 17: return "ddseq"; case 18: return "quartzseq2";
            case 19: return "microwell-seq"; case 20: return "surecell";
            case 21: return "strtseq"; case 22: return "10x-arc-gex";
            case 23: return "10x-atac"; case 24: return "10x-visium";
            default: return "other";
        }
    };

    std::cerr << "[1fq-decode] File: " << input_path << "\n"
              << "[1fq-decode] Version: " << hdr.version
              << ", Protocol: " << proto_name(hdr.protocol_id)
              << " (id=" << (int)hdr.protocol_id
              << ", confidence=" << (int)hdr.confidence << ")\n"
              << "[1fq-decode] Reads: " << hdr.n_unique
              << ", Blocks: " << hdr.block_count
              << ", Streams: " << (int)hdr.n_streams << "\n"
              << "[1fq-decode] R1 length: "
              << (hdr.stream_lengths[0] ? std::to_string(hdr.stream_lengths[0]) : "variable")
              << ", R2 length: "
              << (hdr.stream_lengths[1] ? std::to_string(hdr.stream_lengths[1]) : "variable")
              << "\n";

    if (header_only) return 0;

    // If no output specified, just print header
    if (r1_path.empty() && r2_path.empty()) {
        std::cerr << "[1fq-decode] No output files specified (--r1 / --r2). Use --header for header-only.\n";
        return 0;
    }

    // Open output files
    FILE* r1_fp = nullptr;
    FILE* r2_fp = nullptr;
    if (!r1_path.empty()) {
        r1_fp = fopen(r1_path.c_str(), "w");
        if (!r1_fp) { std::cerr << "ERROR: cannot open " << r1_path << "\n"; return 1; }
        static char r1_wbuf[1 << 20];
        setvbuf(r1_fp, r1_wbuf, _IOFBF, sizeof(r1_wbuf));
    }
    if (!r2_path.empty()) {
        r2_fp = fopen(r2_path.c_str(), "w");
        if (!r2_fp) { std::cerr << "ERROR: cannot open " << r2_path << "\n"; return 1; }
        static char r2_wbuf[1 << 20];
        setvbuf(r2_fp, r2_wbuf, _IOFBF, sizeof(r2_wbuf));
    }

    StopWatch sw;
    const uint32_t total_blocks = reader.block_count();
    const int fd = reader.file_descriptor();
    const auto codec = static_cast<lib1fq::Codec>(hdr.codec);
    const int decode_threads = std::min(threads, static_cast<int>(total_blocks));

    static constexpr char NUM_TO_ASCII[5] = {'A','C','G','T','N'};

    struct FmtBlock {
        std::vector<char> r1, r2;
        size_t r1_len = 0, r2_len = 0;
        uint32_t n_reads = 0;
    };

    auto process_block = [&](uint32_t bi, uint64_t read_off) -> FmtBlock {
        FmtBlock fb;
        uint64_t file_off = reader.block_offset(bi);
        uint32_t comp_size = reader.block_comp_size(bi);
        lib1fq::BlockHeader bh;
        ::pread(fd, &bh, sizeof(bh), static_cast<off_t>(file_off));
        std::vector<uint8_t> comp_buf(comp_size);
        ::pread(fd, comp_buf.data(), comp_size,
                static_cast<off_t>(file_off + sizeof(bh)));
        size_t est_raw = static_cast<size_t>(bh.n_reads) * 200;
        std::vector<uint8_t> raw_buf(est_raw);
        size_t raw_size;
        try {
            raw_size = lib1fq::compress::decompress_block(
                codec, comp_buf.data(), comp_size, raw_buf.data(), raw_buf.size());
        } catch (...) {
            raw_buf.resize(est_raw * 4);
            raw_size = lib1fq::compress::decompress_block(
                codec, comp_buf.data(), comp_size, raw_buf.data(), raw_buf.size());
        }
        lib1fq::DecodedBlock blk;
        reader.decode_payload(raw_buf.data(), raw_size, bh, blk);
        fb.n_reads = blk.n_reads;
        uint16_t r1_max = blk.r1_lengths.empty() ? 28 : *std::max_element(blk.r1_lengths.begin(), blk.r1_lengths.end());
        uint16_t r2_max = blk.r2_lengths.empty() ? 91 : *std::max_element(blk.r2_lengths.begin(), blk.r2_lengths.end());
        uint64_t total_reps = 0;
        const bool has_dups = !blk.dup_counts.empty();
        for (uint32_t i = 0; i < blk.n_reads; ++i)
            total_reps += has_dups ? blk.dup_counts[i] : 1;
        if (r1_fp) fb.r1.resize(total_reps * (static_cast<size_t>(r1_max) * 2 + 30));
        if (r2_fp) fb.r2.resize(total_reps * (static_cast<size_t>(r2_max) * 2 + 30));
        char* r1p = fb.r1.data();
        char* r2p = fb.r2.data();
        bool has_qual = !blk.r2_qual.empty();
        uint64_t roff = read_off;
        for (uint32_t i = 0; i < blk.n_reads; ++i) {
            const uint32_t repeat = has_dups ? blk.dup_counts[i] : 1;
            uint16_t r1len = blk.r1_len(i);
            uint16_t r2len = blk.r2_len(i);
            // Skip zero-length reads — STAR cannot handle 0-length sequences.
            if (r1len == 0 || r2len == 0) continue;
            // Skip corrupted large-R2 reads (same guard as FIFO process_block2).
            if (r2len > 1024) { roff += repeat; continue; }
            const uint8_t* r1s = blk.r1_seq(i);
            const uint8_t* r2s = blk.r2_seq(i);
            const uint8_t* q = has_qual ? blk.r2_quality(i) : nullptr;
            for (uint32_t rep = 0; rep < repeat; ++rep) {
                if (r1_fp) {
                    int n = sprintf(r1p, "@r%lu\n", (unsigned long)roff);
                    char* s1 = r1p + n;
                    for (uint16_t j = 0; j < r1len; ++j) s1[j] = NUM_TO_ASCII[r1s[j]];
                    s1 += r1len;
                    *s1++ = '\n'; *s1++ = '+'; *s1++ = '\n';
                    memset(s1, 'F', r1len);
                    s1 += r1len;
                    *s1++ = '\n';
                    r1p = s1;
                }
                if (r2_fp) {
                    int n = sprintf(r2p, "@r%lu\n", (unsigned long)roff);
                    char* s2 = r2p + n;
                    const char* seq_start = s2;
                    for (uint16_t j = 0; j < r2len; ++j) s2[j] = NUM_TO_ASCII[r2s[j]];
                    s2 += r2len;
                    // seq_len derived from actual bytes written ensures quality
                    // string length always matches sequence length per read,
                    // even for variable-length R2 with QualMode==NONE.
                    const uint16_t seq_len = static_cast<uint16_t>(s2 - seq_start);
                    *s2++ = '\n'; *s2++ = '+'; *s2++ = '\n';
                    if (q) { for (uint16_t j = 0; j < seq_len; ++j) s2[j] = static_cast<char>(q[j] + 33); }
                    else { memset(s2, 'F', seq_len); }
                    s2 += seq_len;
                    *s2++ = '\n';
                    r2p = s2;
                }
                roff++;
            }
        }
        fb.r1_len = r1_fp ? static_cast<size_t>(r1p - fb.r1.data()) : 0;
        fb.r2_len = r2_fp ? static_cast<size_t>(r2p - fb.r2.data()) : 0;
        return fb;
    };

    // Pre-compute read offsets
    std::vector<uint64_t> block_read_offsets(total_blocks);
    {
        uint64_t cum = 0;
        for (uint32_t b = 0; b < total_blocks; ++b) {
            block_read_offsets[b] = cum;
            lib1fq::BlockHeader bh;
            ::pread(fd, &bh, sizeof(bh), static_cast<off_t>(reader.block_offset(b)));
            cum += bh.n_reads;
        }
    }

    uint64_t reads_written = 0;
    uint32_t batch_size = static_cast<uint32_t>(decode_threads);
    for (uint32_t base = 0; base < total_blocks; base += batch_size) {
        uint32_t batch_end = std::min(base + batch_size, total_blocks);
        uint32_t n_batch = batch_end - base;
        std::vector<std::future<FmtBlock>> futures;
        futures.reserve(n_batch);
        for (uint32_t b = base; b < batch_end; ++b)
            futures.push_back(std::async(std::launch::async, process_block, b, block_read_offsets[b]));
        for (uint32_t i = 0; i < n_batch; ++i) {
            FmtBlock fb = futures[i].get();
            if (r1_fp) fwrite(fb.r1.data(), 1, fb.r1_len, r1_fp);
            if (r2_fp) fwrite(fb.r2.data(), 1, fb.r2_len, r2_fp);
            reads_written += fb.n_reads;
        }
    }

    if (r1_fp) fclose(r1_fp);
    if (r2_fp) fclose(r2_fp);

    std::cerr << "[1fq-decode] Done: " << reads_written
              << " reads (" << decode_threads << "T), "
              << sw.elapsed_s() << "s\n";
    if (verify) std::cerr << "[1fq-decode] CRC verification: PASS\n";
    return 0;
}

// ════════════════════════════════════════════════════════════
// singlet archive — Strip quality from .1fq for cold storage
// ════════════════════════════════════════════════════════════

static int cmd_archive(int argc, char* argv[]) {
    std::string input_path, output_path;
    int codec_level = 6;   // Higher compression for cold storage
    bool verbose = false;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cerr
                << "singlet archive — Strip quality scores for cold storage\n\n"
                << "Usage: singlet archive <input.1fq> -o <output.1fq>\n\n"
                << "  After pileup is complete, quality scores are no longer needed.\n"
                << "  This command re-encodes the .1fq without quality, reducing size ~25%.\n"
                << "  The resulting archive can still be aligned but quality will be uniform.\n\n"
                << "Options:\n"
                << "  -o, --output FILE    Output .1fq path (default: input.noq.1fq)\n"
                << "  --codec-level N      zstd compression level (default: 6)\n"
                << "  --verbose, -v        Print progress\n"
                << "  -h, --help           Show this help\n";
            return 0;
        }
        else if ((arg == "-o" || arg == "--output") && i+1 < argc)
            output_path = argv[++i];
        else if (arg == "--codec-level" && i+1 < argc)
            codec_level = std::atoi(argv[++i]);
        else if (arg == "--verbose" || arg == "-v") verbose = true;
        else if (arg[0] != '-' && input_path.empty()) input_path = arg;
        else { std::cerr << "Unknown option: " << arg << "\n"; return 1; }
    }

    if (input_path.empty()) {
        std::cerr << "ERROR: input .1fq file required\n"
                  << "Usage: singlet archive <input.1fq> -o <output.1fq>\n";
        return 1;
    }

    if (output_path.empty()) {
        // Insert .noq before .1fq
        auto pos = input_path.rfind(".1fq");
        if (pos != std::string::npos)
            output_path = input_path.substr(0, pos) + ".noq.1fq";
        else
            output_path = input_path + ".noq.1fq";
    }

    lib1fq::Reader reader;
    reader.open(input_path.c_str());
    const lib1fq::Header& hdr = reader.header();

    if (hdr.qual_mode == static_cast<uint8_t>(lib1fq::QualMode::NONE)) {
        std::cerr << "[archive] Input already has no quality stored — nothing to strip.\n";
        return 0;
    }

    // Build writer config mirroring input, but with qual_mode=NONE
    lib1fq::WriterConfig wcfg;
    wcfg.qual_mode      = lib1fq::QualMode::NONE;
    wcfg.codec          = static_cast<lib1fq::Codec>(hdr.codec);
    wcfg.codec_level    = codec_level;
    wcfg.n_streams      = hdr.n_streams;
    wcfg.r1_length      = hdr.stream_lengths[0];
    // If original had variable R2 (TRIMMED flag), keep variable (0); else preserve fixed
    wcfg.r2_length      = (hdr.flags & lib1fq::Flags::TRIMMED)
                              ? uint16_t(0) : hdr.stream_lengths[1];
    wcfg.protocol_id    = hdr.protocol_id;
    wcfg.confidence     = static_cast<lib1fq::Confidence>(hdr.confidence);
    wcfg.assay_type     = static_cast<lib1fq::AssayType>(hdr.assay_type);
    wcfg.deduped        = (hdr.flags & lib1fq::Flags::DEDUPED) != 0;
    wcfg.polya_trim     = false;  // already trimmed if applicable
    wcfg.sort_by_bc     = false;  // preserve original ordering
    wcfg.block_size     = hdr.block_size > 0 ? hdr.block_size : 100000;

    lib1fq::Writer writer;
    writer.open(output_path.c_str(), wcfg);

    uint64_t n_reads = 0;
    lib1fq::DecodedBlock blk;
    while (reader.read_block(blk)) {
        for (uint32_t i = 0; i < blk.n_reads; ++i) {
            uint32_t dc = (wcfg.deduped && !blk.dup_counts.empty())
                              ? blk.dup_counts[i] : 1;
            writer.add_read(
                blk.r2_seq(i), blk.r2_len(i), nullptr,   // R2, no quality
                blk.r1_seq(i), blk.r1_len(i), nullptr,   // R1, no quality
                dc);
        }
        n_reads += blk.n_reads;
        if (verbose)
            std::cerr << "\r[archive] " << n_reads << " / " << hdr.n_unique
                      << " reads" << std::flush;
    }

    // Pass through metadata JSON if present
    std::string meta = reader.read_metadata();
    writer.finish(meta);

    if (verbose) std::cerr << "\n";
    std::cerr << "[archive] Done: " << n_reads << " reads → " << output_path << "\n";
    return 0;
}

// ════════════════════════════════════════════════════════════
// N21: Shared-memory genome helpers
// ════════════════════════════════════════════════════════════

// Return the System V shared memory key STAR uses for genome_dir.
// STAR sets shmKey = st_ino of the genome directory (Genome.cpp:19).
static key_t star_genome_shmkey(const std::string& genome_dir) {
    struct stat st{};
    if (stat(genome_dir.c_str(), &st) != 0) return (key_t)-1;
    return static_cast<key_t>(st.st_ino);
}

// Return true if STAR's shared memory segment for genome_dir is present.
static bool star_genome_in_shm(const std::string& genome_dir) {
    key_t k = star_genome_shmkey(genome_dir);
    if (k == (key_t)-1) return false;
    return shmget(k, 0, 0) != -1;
}

// Run STAR genome load or unload in a forked child (STAR calls exit() internally).
// load_mode: "LoadAndExit" or "Remove"
// Returns child exit status (0 = success).
static int run_star_genome_op(const std::string& genome_dir,
                               const std::string& load_mode,
                               const std::string& out_prefix) {
    mkdir_p(out_prefix);

    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "ERROR: fork() failed: " << strerror(errno) << "\n";
        return 1;
    }
    if (pid == 0) {
        // CHILD — redirect STAR stdout/stderr to log file
        std::string logfile = out_prefix + "/star_genome.log";
        int fd = open(logfile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            dup2(fd, STDOUT_FILENO);
            dup2(fd, STDERR_FILENO);
            close(fd);
        }
        std::vector<const char*> args;
        args.push_back("STAR");
        args.push_back("--runMode");          args.push_back("alignReads");
        args.push_back("--genomeDir");        args.push_back(genome_dir.c_str());
        args.push_back("--genomeLoad");       args.push_back(load_mode.c_str());
        std::string pfx = out_prefix + "/star_";
        args.push_back("--outFileNamePrefix"); args.push_back(pfx.c_str());
        args.push_back("--outSAMtype");        args.push_back("None");
        // LoadAndExit/Remove exit from inside genomeLoad() before reading inputs,
        // but STAR still validates --readFilesIn at parameter parse time.
        args.push_back("--readFilesIn");       args.push_back("/dev/null");
        args.push_back("--runThreadN");        args.push_back("1");
        args.push_back(nullptr);
        int rc = star_main_impl(static_cast<int>(args.size()) - 1,
                                const_cast<char**>(args.data()));
        _exit(rc);
    }
    int wstatus = 0;
    waitpid(pid, &wstatus, 0);
    return WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : 1;
}

// ── NONHOST-EM: shared screening helper ──────────────────────────────────────
// Run EM-based nonhost screening on STAR unmapped reads.
// Reads ${out_prefix}/star_Unmapped.out.mate1, runs EM deconvolution, and writes:
//   • nonhost_em_abundance.tsv  — per-species EM abundances
//   • nonhost_summary.json      — kingdom fractions + em_species_count + top_pathogens[]
//
// Called from all post-STAR assay paths (scRNA, Bulk RNA, ATAC/Visium).
// Non-fatal: logs a warning on any error and returns without crashing.
static void run_nonhost_em_screening(const std::string& viral_db_path,
                                      const std::string& bacterial_db_path,
                                      const std::string& fungal_db_path,
                                      const std::string& nonhost_ref_dir,
                                      const std::string& out_prefix,
                                      const std::unordered_set<std::string>& valid_barcodes,
                                      int cb_len,
                                      int umi_len,
                                      int n_threads = 1) {
    using namespace singlet::nonhost;

    if (viral_db_path.empty() && bacterial_db_path.empty() && fungal_db_path.empty())
        return;

    std::string mate1_path = out_prefix + "/star_Unmapped.out.mate1";
    struct stat mate1_st{};
    if (stat(mate1_path.c_str(), &mate1_st) != 0 || mate1_st.st_size == 0) {
        std::cerr << "[nonhost] WARNING: nonhost screening requested but "
                  << mate1_path << " not found (did STAR produce unmapped reads?)\n";
        return;
    }

    std::cerr << "[nonhost] Loading screening indices...\n";
    try {
        // ── Load viral / bacterial / fungal indices ───────────────────────────
        int ref_k = 21, ref_w = 11;

        MinSketchIndex viral_idx(ref_k, ref_w);
        viral_idx.finalize();
        if (!viral_db_path.empty()) {
            std::cerr << "[nonhost] Loading viral index: " << viral_db_path << "\n";
            viral_idx = MinSketchIndex::load(viral_db_path);
            ref_k = viral_idx.k(); ref_w = viral_idx.w();
        }

        // fungal_species_ids: set of species_ids from the fungal index.
        // Used to assign NonHostCategory::FUNGAL instead of BACTERIAL.
        std::unordered_set<uint32_t> fungal_species_ids;

        MinSketchIndex microbial_idx(ref_k, ref_w);
        microbial_idx.finalize();

        if (!bacterial_db_path.empty()) {
            std::cerr << "[nonhost] Loading bacterial index: " << bacterial_db_path << "\n";
            MinSketchIndex bact_idx = MinSketchIndex::load(bacterial_db_path);
            if (microbial_idx.empty())
                microbial_idx = std::move(bact_idx);
            else
                microbial_idx = MinSketchIndex::merge(microbial_idx, bact_idx);
            ref_k = microbial_idx.k(); ref_w = microbial_idx.w();
        }

        if (!fungal_db_path.empty()) {
            std::cerr << "[nonhost] Loading fungal index: " << fungal_db_path << "\n";
            MinSketchIndex fung_idx = MinSketchIndex::load(fungal_db_path);
            // Record all species_ids from the fungal index before merging
            for (uint32_t sid = 0;
                 sid < static_cast<uint32_t>(fung_idx.n_species_names()); ++sid)
                if (!fung_idx.species_name(sid).empty())
                    fungal_species_ids.insert(sid);
            if (microbial_idx.empty())
                microbial_idx = std::move(fung_idx);
            else
                microbial_idx = MinSketchIndex::merge(microbial_idx, fung_idx);
        }

        // If no viral DB, the viral slot is an empty index (correct k/w still needed)
        if (viral_idx.empty() && !microbial_idx.empty()) {
            viral_idx = MinSketchIndex(microbial_idx.k(), microbial_idx.w());
            viral_idx.finalize();
        }

        // Screener with FUNGAL species set for correct kingdom assignment
        NonHostScreener screener(viral_idx, microbial_idx, 0.05f,
                                 std::move(fungal_species_ids));

        // Helper to look up species names across both indices
        auto species_name_lookup = [&](uint32_t sid) -> std::string {
            std::string n = viral_idx.species_name(sid);
            if (!n.empty()) return n;
            return microbial_idx.species_name(sid);
        };

        // ── NONHOST-HOST-SUBTRACT: try to load host Bloom filter ──────────────
        HostKmerFilter host_filter;
        bool using_host_filter = false;
        {
            const int eff_k = !viral_idx.empty()    ? viral_idx.k()    :
                              !microbial_idx.empty() ? microbial_idx.k(): 21;
            const std::string& first_db = !viral_db_path.empty()    ? viral_db_path    :
                                          !bacterial_db_path.empty() ? bacterial_db_path :
                                                                        fungal_db_path;
            std::string db_dir = first_db.substr(0, first_db.rfind('/') + 1);
            std::string bloom_path = db_dir + "host_" + std::to_string(eff_k) + "mer.bloom";

            const char* ref_base = getenv("SINGLET_REF_BASE");
            std::string bloom_fallback;
            if (!HostKmerFilter::exists(bloom_path) && ref_base)
                bloom_fallback = std::string(ref_base) + "/nonhost/host_"
                                 + std::to_string(eff_k) + "mer.bloom";

            const std::string& active_path =
                HostKmerFilter::exists(bloom_path)       ? bloom_path    :
                (!bloom_fallback.empty() &&
                 HostKmerFilter::exists(bloom_fallback)) ? bloom_fallback :
                                                           bloom_path;

            if (HostKmerFilter::exists(active_path)) {
                std::cerr << "[nonhost] Loading host Bloom filter: " << active_path << "\n";
                host_filter       = HostKmerFilter::load(active_path);
                using_host_filter = true;
            } else {
                std::cerr << "[nonhost] WARNING: host Bloom filter not found at "
                          << bloom_path << " — host-shared k-mer FP suppression disabled.\n"
                          << "[nonhost]   Build with: singlet build-host-filter"
                             " --genome-fasta <genome.fa> --output " << bloom_path << "\n";
            }
        }

        // Read unmapped FASTQ (4 lines per record), cap at 20M reads to prevent OOM
        // when mapping rate is near-zero and nearly all reads are unmapped.
        // EM deconvolution is statistical — 20M reads is more than sufficient for
        // species-level classification (even 1M would suffice).
        static constexpr size_t MAX_NONHOST_READS = 20'000'000;
        std::vector<std::string> reads;
        {
            std::ifstream fq(mate1_path);
            std::string name_l, seq_l, plus_l, qual_l;
            while (std::getline(fq, name_l) && std::getline(fq, seq_l) &&
                   std::getline(fq, plus_l) && std::getline(fq, qual_l)) {
                if (!seq_l.empty()) reads.push_back(seq_l);
                if (reads.size() >= MAX_NONHOST_READS) break;
            }
        }
        const size_t n_total = reads.size();
        if (n_total >= MAX_NONHOST_READS)
            std::cerr << "[nonhost] Capped at " << MAX_NONHOST_READS
                      << " reads (file larger; subsampling for EM)\n";
        std::cerr << "[nonhost] EM screening " << n_total << " unmapped reads"
                  << (using_host_filter ? " (host-subtraction ON)" : "") << "...\n";

        // Parallelise k-mer screening over --threads workers (dominant bottleneck:
        // 9.1M reads × ~820M hash lookups vs 25 GB index; each lookup is a DRAM
        // cache miss.  The batch sort-merge approach replaces O(reads × log DB)
        // random DRAM accesses with one parallel sort + one sequential linear scan,
        // giving ~100× speedup for 9.1M-read batches).
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(std::max(1, n_threads));
#endif
        // Classify with soft multi-hit threshold (EM input)
        // Use host-filtered version when Bloom filter is loaded
        std::vector<NonHostScreener::MultiHit> multi_hits;
        if (using_host_filter)
            multi_hits = screener.classify_multi_batch_filtered(reads, host_filter, 0.25f, n_threads);
        else
            multi_hits = screener.classify_multi_batch(reads, 0.25f, n_threads);

        // Build kingdom map (FUNGAL correctly labelled via screener.fungal_species())
        auto kingdom_map = build_kingdom_map(multi_hits, screener.fungal_species());
        int  n_em_iter   = 0;
        const uint32_t min_abs_reads = std::max(500u,
            static_cast<uint32_t>(0.005f * static_cast<float>(n_total)));
        auto em_result = em_deconvolve(multi_hits, kingdom_map,
                                       /*max_iter=*/500, /*tol=*/1e-6f,
                                       /*min_abundance=*/1e-4f,
                                       /*min_prefilter_hr=*/0.05f,
                                       /*min_report_hr=*/0.30f, &n_em_iter,
                                       /*min_abs_reads=*/min_abs_reads);

        std::cerr << "[nonhost] EM converged in " << n_em_iter
                  << " iterations, " << em_result.size() << " species above threshold\n";

        // ── Secondary alignment (NONHOST-SECONDARY-ALIGN) ────────────────────
        // For each EM-detected species, try seed-chain-extend alignment against
        // a per-species FASTA. Missing FASTAs are logged and skipped gracefully.
        if (!em_result.empty()) {
            const char* ref_base_env = getenv("SINGLET_REF_BASE");
            const std::string eff_ref_dir = !nonhost_ref_dir.empty() ? nonhost_ref_dir :
                                            ref_base_env             ? ref_base_env    :
                                                                        std::string{};
            if (!eff_ref_dir.empty()) {
                for (const auto& sa : em_result) {
                    const std::string kdir =
                        sa.kingdom == NonHostCategory::VIRAL   ? "viral_genomes"    :
                        sa.kingdom == NonHostCategory::FUNGAL  ? "fungal_genomes"   :
                                                                  "bacterial_genomes";
                    std::string fna = eff_ref_dir + "/nonhost/" + kdir
                                    + "/species_" + std::to_string(sa.species_id) + ".fna";
                    struct stat fna_st{};
                    if (stat(fna.c_str(), &fna_st) == 0)
                        std::cerr << "[nonhost] species_id=" << sa.species_id
                                  << " confirmed (FASTA found)\n";
                    else
                        std::cerr << "[nonhost] WARNING: species_id=" << sa.species_id
                                  << " confirmed by EM but no reference FASTA available"
                                     " for secondary alignment (" << fna << " not found)\n";
                }
                singlet::nonhost::NonHostAligner sec_aligner(eff_ref_dir);
                auto sec_results = sec_aligner.align(reads, em_result);
                if (!sec_results.empty())
                    singlet::nonhost::NonHostAligner::write_tsv(sec_results, out_prefix + "/nonhost");
            } else {
                std::cerr << "[nonhost] Skipping secondary alignment:"
                             " set SINGLET_REF_BASE or --nonhost-ref-dir\n";
            }
        }

        // ── NONHOST-CB-SPECIES-MATRIX: build per-cell species count matrix ────
        if (!valid_barcodes.empty()) {
            const std::string mate2_path = out_prefix + "/star_Unmapped.out.mate2";
            struct stat mate2_st{};
            if (stat(mate2_path.c_str(), &mate2_st) == 0 && mate2_st.st_size > 0) {
                auto tagged = singlet::nonhost::NonHostCellMatrix::read_mate_pairs(
                    mate1_path, mate2_path, cb_len, umi_len);

                std::unordered_map<uint32_t, singlet::nonhost::NonHostCategory> kmap_cell;
                std::unordered_map<uint32_t, std::string> snames;
                for (const auto& sa : em_result) {
                    kmap_cell[sa.species_id] = sa.kingdom;
                    snames[sa.species_id]    = species_name_lookup(sa.species_id);
                }

                auto cell_mat = singlet::nonhost::NonHostCellMatrix::build(
                    tagged, multi_hits, valid_barcodes, snames, kmap_cell);
                cell_mat.write(out_prefix + "/nonhost");
                std::cerr << "[nonhost] per-cell matrix: " << cell_mat.n_records()
                          << " cell\xc3\x97species records\n";
            }
        }

        // ── Write nonhost_em_abundance.tsv ───────────────────────────────────
        {
            std::filesystem::create_directories(out_prefix + "/nonhost");
            std::ofstream tsv(out_prefix + "/nonhost/nonhost_em_abundance.tsv");
            if (tsv.is_open()) {
                tsv << "kingdom\tspecies_id\tspecies_name\trelative_abundance"
                       "\treads_assigned\tmean_hit_rate\tconfidence\n";
                for (const auto& sa : em_result) {
                    const char* kname = "";
                    switch (sa.kingdom) {
                        case NonHostCategory::VIRAL:     kname = "VIRAL";     break;
                        case NonHostCategory::BACTERIAL: kname = "BACTERIAL"; break;
                        case NonHostCategory::FUNGAL:    kname = "FUNGAL";    break;
                        default:                          kname = "UNKNOWN";   break;
                    }
                    tsv << kname << "\t" << sa.species_id << "\t"
                        << species_name_lookup(sa.species_id) << "\t"
                        << std::fixed << std::setprecision(6)
                        << sa.relative_abundance << "\t"
                        << sa.n_reads_assigned << "\t"
                        << sa.mean_hit_rate << "\t"
                        << singlet::nonhost::confidence_str(sa.confidence) << "\n";
                }
            }
        }

        // ── Kingdom fraction tallies (for nonhost_summary.json) ──────────────
        size_t viral_hits = 0, bacterial_hits = 0, fungal_hits = 0, unknown_hits = 0;
        for (const auto& mh : multi_hits) {
            switch (mh.top_category) {
                case NonHostCategory::VIRAL:     ++viral_hits;     break;
                case NonHostCategory::BACTERIAL: ++bacterial_hits; break;
                case NonHostCategory::FUNGAL:    ++fungal_hits;    break;
                default:                          ++unknown_hits;   break;
            }
        }
        // Reads with no multi-hits go into unknown bucket
        unknown_hits += n_total - multi_hits.size();  // multi_hits may be shorter if some reads are too short

        const size_t n = n_total;
        double vf = n > 0 ? static_cast<double>(viral_hits)    / n : 0.0;
        double bf = n > 0 ? static_cast<double>(bacterial_hits) / n : 0.0;
        double ff = n > 0 ? static_cast<double>(fungal_hits)    / n : 0.0;

        // ── Write nonhost_summary.json ────────────────────────────────────────
        {
            std::string json_path = out_prefix + "/nonhost/nonhost_summary.json";
            std::ofstream jf(json_path);
            if (jf.is_open()) {
                // Build kingdoms_screened list for JSON
                std::string ks_list = "[";
                bool first_ks = true;
                if (!viral_db_path.empty())     { ks_list += (first_ks?"":",")+std::string("\"VIRAL\"");     first_ks=false; }
                if (!bacterial_db_path.empty()) { ks_list += (first_ks?"":",")+std::string("\"BACTERIAL\""); first_ks=false; }
                if (!fungal_db_path.empty())    { ks_list += (first_ks?"":",")+std::string("\"FUNGAL\"");    first_ks=false; }
                ks_list += "]";

                jf << "{\n"
                   << "  \"total_unmapped_reads\": " << n << ",\n"
                   << "  \"screened_reads\": "        << n << ",\n"
                   << "  \"kingdoms_screened\": "     << ks_list << ",\n"
                   << "  \"host_filter_active\": "   << (using_host_filter ? "true" : "false") << ",\n"
                   << "  \"viral_hits\": "            << viral_hits << ",\n"
                   << "  \"bacterial_hits\": "        << bacterial_hits << ",\n"
                   << "  \"fungal_hits\": "           << fungal_hits << ",\n"
                   << "  \"unknown_hits\": "          << unknown_hits << ",\n"
                   << "  \"viral_fraction\": "        << std::fixed << std::setprecision(6) << vf << ",\n"
                   << "  \"bacterial_fraction\": "    << bf << ",\n"
                   << "  \"fungal_fraction\": "       << ff << ",\n"
                   << "  \"em_species_count\": "      << em_result.size() << ",\n"
                   << "  \"em_iterations\": "         << n_em_iter << ",\n"
                   << "  \"top_pathogens\": [\n";

                size_t top_n = std::min(em_result.size(), static_cast<size_t>(10));
                for (size_t pi = 0; pi < top_n; ++pi) {
                    const auto& sa = em_result[pi];
                    const char* kname = "";
                    switch (sa.kingdom) {
                        case NonHostCategory::VIRAL:     kname = "VIRAL";     break;
                        case NonHostCategory::BACTERIAL: kname = "BACTERIAL"; break;
                        case NonHostCategory::FUNGAL:    kname = "FUNGAL";    break;
                        default:                          kname = "UNKNOWN";   break;
                    }
                    jf << "    {\"species_id\": " << sa.species_id
                       << ", \"species_name\": \"" << species_name_lookup(sa.species_id) << "\""
                       << ", \"kingdom\": \"" << kname << "\""
                       << ", \"relative_abundance\": " << sa.relative_abundance
                       << ", \"reads_assigned\": "     << sa.n_reads_assigned << "}";
                    if (pi + 1 < top_n) jf << ",";
                    jf << "\n";
                }
                jf << "  ]\n}\n";

                std::cerr << "[nonhost] em_species=" << em_result.size()
                          << " viral=" << viral_hits << " bacterial=" << bacterial_hits
                          << " → " << json_path << "\n";
            }
        }

        // If any kingdom fraction > 1%, write nonhost_reads.fq for downstream mapping
        bool any_positive = (vf > 0.01 || bf > 0.01 || ff > 0.01);
        if (any_positive) {
            std::string fq_out_path = out_prefix + "/nonhost_reads.fq";
            std::ofstream fq_out(fq_out_path);
            std::ifstream fq_in(mate1_path);
            std::string name_l, seq_l, plus_l, qual_l;
            size_t idx_r = 0;
            while (std::getline(fq_in, name_l) && std::getline(fq_in, seq_l) &&
                   std::getline(fq_in, plus_l) && std::getline(fq_in, qual_l)) {
                if (idx_r < multi_hits.size() &&
                    multi_hits[idx_r].top_category != NonHostCategory::UNKNOWN) {
                    fq_out << name_l << "\n" << seq_l << "\n"
                           << plus_l << "\n" << qual_l << "\n";
                }
                ++idx_r;
            }
            std::cerr << "[nonhost] Wrote classified reads → " << fq_out_path << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "[nonhost] WARNING: screening failed: " << e.what() << "\n";
        // Non-fatal — pipeline continues without nonhost output
    }
}

// ════════════════════════════════════════════════════════════
// singlet build-nonhost-db — Build MinSketchIndex from a FASTA
//
// Usage:
//   singlet build-nonhost-db <output.snhskidx> \
//       --kingdom VIRAL|BACTERIAL|FUNGAL \
//       --fasta <input.fna[.gz]> [--threads N]
//
// Reads FASTA (plain or gzip via zcat pipe), builds a MinSketchIndex
// (k=21, w=11), and writes it to <output.snhskidx>.
// Each FASTA contig is assigned a sequential species_id.
// Progress is logged every 100 sequences.
//
// Part of NONHOST-VIRALDB / NONHOST-MICROBIALDB DAG tasks.
// ════════════════════════════════════════════════════════════

static int cmd_build_nonhost_db(int argc, char* argv[]) {
    using namespace singlet::nonhost;

    std::string output_path;
    std::string fasta_path;
    std::string kingdom = "VIRAL";

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fasta"   && i+1 < argc) fasta_path   = argv[++i];
        else if (arg == "--kingdom"  && i+1 < argc) kingdom      = argv[++i];
        else if (arg == "--threads"  && i+1 < argc) { /* reserved — index build is single-threaded */ (void)argv[++i]; }
        else if (arg == "-h" || arg == "--help") {
            std::cerr << "Usage: singlet build-nonhost-db <output.snhskidx>"
                         " --fasta <in.fna[.gz]> --kingdom VIRAL|BACTERIAL|FUNGAL [--threads N]\n";
            return 0;
        }
        else if (!arg.empty() && arg[0] != '-') output_path = arg;
    }

    if (output_path.empty() || fasta_path.empty()) {
        std::cerr << "Usage: singlet build-nonhost-db <output.snhskidx>"
                     " --fasta <in.fna[.gz]> --kingdom VIRAL|BACTERIAL|FUNGAL [--threads N]\n";
        return 1;
    }

    // Create output directory if needed
    {
        auto pos = output_path.rfind('/');
        if (pos != std::string::npos && pos > 0)
            mkdir_p(output_path.substr(0, pos));
    }

    std::cerr << "[build-nonhost-db] Kingdom=" << kingdom
              << " fasta=" << fasta_path << " out=" << output_path << "\n";

    MinSketchIndex idx(21, 11);
    uint32_t species_id = 0;
    uint32_t seq_count  = 0;

    // FASTA streaming callback — called with an open FILE*
    auto process_fasta = [&](FILE* fp) {
        char line_buf[65536];
        std::string seq;
        while (std::fgets(line_buf, sizeof(line_buf), fp)) {
            size_t len = std::strlen(line_buf);
            while (len > 0 && (line_buf[len-1] == '\n' || line_buf[len-1] == '\r')) --len;
            if (len == 0) continue;
            if (line_buf[0] == '>') {
                if (!seq.empty()) {
                    idx.add_sequence(seq, species_id, 50000u);
                    seq.clear();
                    ++seq_count;
                    if (seq_count % 100 == 0)
                        std::cerr << "[build-nonhost-db] " << seq_count
                                  << " seqs, index size=" << idx.size() << "\n";
                }
                ++species_id;
                // Capture species name from FASTA header: ">ACC DESC" → "ACC DESC"
                if (len > 1) {
                    std::string hdr_name(line_buf + 1, len - 1);
                    idx.set_species_name(species_id, hdr_name);
                }
            } else {
                seq.append(line_buf, len);
            }
        }
        if (!seq.empty()) {
            idx.add_sequence(seq, species_id, 50000u);
            ++seq_count;
        }
    };

    bool is_gz = fasta_path.size() >= 3 &&
                 fasta_path.substr(fasta_path.size() - 3) == ".gz";

    if (is_gz) {
        // Use popen("zcat") so we don't need a zlib dependency in this code path
        std::string cmd = "zcat '" + fasta_path + "'";
        FILE* fp = ::popen(cmd.c_str(), "r");
        if (!fp) {
            std::cerr << "[build-nonhost-db] ERROR: cannot run: " << cmd << "\n";
            return 1;
        }
        process_fasta(fp);
        int rc = ::pclose(fp);
        if (rc != 0)
            std::cerr << "[build-nonhost-db] WARNING: zcat exited " << rc << "\n";
    } else {
        FILE* fp = std::fopen(fasta_path.c_str(), "r");
        if (!fp) {
            std::cerr << "[build-nonhost-db] ERROR: cannot open FASTA: " << fasta_path << "\n";
            return 1;
        }
        process_fasta(fp);
        std::fclose(fp);
    }

    std::cerr << "[build-nonhost-db] Finalizing: " << seq_count
              << " seqs, " << idx.size() << " minimizer entries...\n";
    idx.finalize();

    std::cerr << "[build-nonhost-db] Saving to " << output_path << "\n";
    idx.save(output_path);

    struct stat st{};
    stat(output_path.c_str(), &st);
    std::cerr << "[build-nonhost-db] Done! kingdom=" << kingdom
              << " entries=" << idx.size()
              << " size=" << st.st_size / 1024 << " KB\n";
    return 0;
}

// ════════════════════════════════════════════════════════════
// singlet build-host-filter — Build HostKmerFilter Bloom filter
//
// Usage:
//   singlet build-host-filter \
//       --genome-fasta <genome.fa[.gz]> \
//       --output <host_21mer.bloom> \
//       [--kmer K] [--window W] [--threads N]
//
// Reads the host genome FASTA (plain or gzip), extracts minimizers with (k, w)
// (must match the NonHostScreener .snhskidx parameters), and writes a Bloom
// filter to the output path.
//
// For GRCh38 (3.1 GB FASTA, k=21, w=11): filter ≈ 313 MB, build time < 5 min.
// ════════════════════════════════════════════════════════════

static int cmd_build_host_filter(int argc, char* argv[]) {
    using namespace singlet::nonhost;

    std::string fasta_path;
    std::string output_path;
    int k        = 21;
    int w        = 11;
    int nthreads = 1;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--genome-fasta" && i+1 < argc) fasta_path  = argv[++i];
        else if (arg == "--output"       && i+1 < argc) output_path = argv[++i];
        else if (arg == "--kmer"         && i+1 < argc) k           = std::stoi(argv[++i]);
        else if (arg == "--window"       && i+1 < argc) w           = std::stoi(argv[++i]);
        else if (arg == "--threads"      && i+1 < argc) nthreads    = std::stoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            std::cerr << "Usage: singlet build-host-filter"
                         " --genome-fasta <genome.fa[.gz]> --output <host_21mer.bloom>"
                         " [--kmer 21] [--window 11] [--threads N]\n";
            return 0;
        }
        else if (!arg.empty() && arg[0] != '-') {
            // positional: first non-flag arg is output path
            if (output_path.empty()) output_path = arg;
        }
    }

    if (fasta_path.empty() || output_path.empty()) {
        std::cerr << "Usage: singlet build-host-filter"
                     " --genome-fasta <genome.fa[.gz]> --output <host_21mer.bloom>"
                     " [--kmer 21] [--window 11] [--threads N]\n";
        return 1;
    }

    // Create output directory if needed
    {
        auto pos = output_path.rfind('/');
        if (pos != std::string::npos && pos > 0)
            mkdir_p(output_path.substr(0, pos));
    }

    std::cerr << "[build-host-filter] Building Bloom filter:"
              << " fasta=" << fasta_path
              << " k="     << k
              << " w="     << w
              << " threads=" << nthreads
              << " out="   << output_path << "\n";

    auto flt = HostKmerFilter::build_from_fasta(
        fasta_path, k, w, /*n_elements=*/0, /*log_progress=*/true, nthreads);

    std::cerr << "[build-host-filter] Saving to " << output_path << " ...\n";
    flt.save(output_path);

    struct stat st{};
    ::stat(output_path.c_str(), &st);
    std::cerr << "[build-host-filter] Done! filter=" << st.st_size / (1024*1024)
              << " MB (k=" << k << " w=" << w << ")\n";
    return 0;
}

// ════════════════════════════════════════════════════════════
// singlet genome build-minimizers — Build MinimizerSAIndex
// Data structure: flat sorted array of {hash, SA_lo, SA_hi} (16 B/entry)
// Serialized as minimizerIndex.bin alongside the STAR genome files.
// ════════════════════════════════════════════════════════════

struct MinimizerRecord {      // 16 bytes/entry
    uint64_t hash;            // canonical minimizer hash (2-bit packed k-mer)
    uint32_t SA_lo;           // lower SA bound (inclusive)
    uint32_t SA_hi;           // upper SA bound (inclusive)
};

// Binary header for minimizerIndex.bin
struct MinimizerIndexHeader { // 32 bytes
    char     magic[8];        // "MINIIDX\0"
    uint32_t k;               // k-mer length (21)
    uint32_t w;               // window size (10)
    uint32_t T;               // max SA range (100)
    uint32_t reserved;
    uint64_t nRecords;
};

// Read an entry from a PackedArray (STAR SA format)
// wordLength bits per entry, stored little-endian
static inline uint64_t packed_sa_get(const char* arr, uint64_t ii, int bits) {
    uint64_t b = (uint64_t)ii * (uint64_t)bits;
    uint64_t B = b >> 3;       // byte offset
    int      S = (int)(b & 7); // bit offset within byte
    uint64_t word;
    memcpy(&word, arr + B, sizeof(word));
    uint64_t mask = (bits < 64) ? ((1ULL << bits) - 1) : ~0ULL;
    return (word >> S) & mask;
}

// Compute canonical minimizer hash (k=21, w=10) for seq[0..k+w-2]
// Uses 2-bit encoding: A=0,C=1,G=2,T=3 (genome bytes are pre-encoded)
// Returns 0 if any N base (value > 3) encountered in the window.
static uint64_t compute_minimizer(const uint8_t* seq, int k, int w) {
    const uint64_t MASK = (1ULL << (2 * k)) - 1;   // 42-bit mask for k=21

    // Build first k-mer (forward)
    uint64_t fwd = 0;
    for (int i = 0; i < k; i++) {
        if (seq[i] > 3) return 0;  // N base
        fwd = (fwd << 2) | seq[i];
    }
    // Build first k-mer (reverse complement)
    uint64_t rc = 0;
    for (int i = k - 1; i >= 0; i--)
        rc = (rc << 2) | (3u - seq[i]);

    uint64_t best = std::min(fwd, rc);

    for (int j = 1; j < w; j++) {
        uint8_t b = seq[j + k - 1];
        if (b > 3) return 0;
        // Rolling forward: drop oldest base (MSB), shift left, add new base
        fwd = ((fwd << 2) & MASK) | b;
        // Rolling RC: drop oldest RC base (LSB = complement of oldest fwd), add new
        rc = (rc >> 2) | ((uint64_t)(3u - b) << (2 * (k - 1)));
        uint64_t canon = std::min(fwd, rc);
        if (canon < best) best = canon;
    }
    return best; // 0 means "poly-A canonical" — treat as skip (same as N)
}

// Open-addressing hash table entry (sentinel: hash == 0 → empty)
struct HEntry {
    uint64_t hash;
    uint32_t lo;
    uint32_t hi;
};

// Insert or update (hash, sa_index) into the hash table (linear probing)
static inline void ht_update(HEntry* table, uint64_t mask, uint64_t h, uint32_t i) {
    for (uint64_t slot = h & mask; ; slot = (slot + 1) & mask) {
        HEntry& e = table[slot];
        if (e.hash == 0) {
            e.hash = h; e.lo = i; e.hi = i;
            return;
        }
        if (e.hash == h) {
            if (i < e.lo) e.lo = i;
            if (i > e.hi) e.hi = i;
            return;
        }
    }
}

static int cmd_genome_build_minimizers(const std::string& genome_dir) {
    constexpr int k = 21, w = 10, T = 100;
    const int window_len = k + w - 1;  // 30 bases

    // ── 1. Parse genomeParameters.txt ──
    uint32_t GstrandBit = 32;
    uint64_t nSAbyte = 0, nGenomeByte = 0;
    {
        std::string path = genome_dir + "/genomeParameters.txt";
        std::ifstream f(path);
        if (!f) {
            std::cerr << "ERROR: cannot open " << path << "\n";
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            auto gs = line.find("GstrandBit");
            if (gs != std::string::npos) {
                sscanf(line.c_str() + gs, "GstrandBit %u", &GstrandBit);
            }
            auto fs = line.find("genomeFileSizes");
            if (fs != std::string::npos) {
                sscanf(line.c_str() + fs, "genomeFileSizes %llu %llu",
                       (unsigned long long*)&nGenomeByte,
                       (unsigned long long*)&nSAbyte);
            }
        }
    }
    if (nSAbyte == 0 || nGenomeByte == 0) {
        std::cerr << "ERROR: could not parse genomeParameters.txt "
                     "(GstrandBit/genomeFileSizes missing)\n";
        return 1;
    }
    int      sa_bits = GstrandBit + 1;  // 33 for GRCh38
    uint64_t nSA     = (nSAbyte * 8) / sa_bits;

    std::cerr << "[build-minimizers] genome=" << nGenomeByte/1073741824.0 << " GiB"
              << "  SA=" << nSAbyte/1073741824.0 << " GiB"
              << "  nSA=" << nSA
              << "  sa_bits=" << sa_bits << "\n"
              << "[build-minimizers] k=" << k << "  w=" << w << "  T=" << T << "\n";

    // ── 2. mmap Genome ──
    std::string genome_path = genome_dir + "/Genome";
    int fd_genome = open(genome_path.c_str(), O_RDONLY);
    if (fd_genome < 0) {
        std::cerr << "ERROR: cannot open " << genome_path << "\n"; return 1;
    }
    void* genome_ptr = mmap(nullptr, nGenomeByte, PROT_READ, MAP_PRIVATE, fd_genome, 0);
    if (genome_ptr == MAP_FAILED) {
        std::cerr << "ERROR: mmap Genome failed\n"; return 1;
    }
    madvise(genome_ptr, nGenomeByte, MADV_RANDOM);
    const uint8_t* genome = static_cast<const uint8_t*>(genome_ptr);

    // ── 3. mmap SA ──
    std::string sa_path = genome_dir + "/SA";
    int fd_sa = open(sa_path.c_str(), O_RDONLY);
    if (fd_sa < 0) {
        std::cerr << "ERROR: cannot open " << sa_path << "\n"; return 1;
    }
    // Map 8 extra bytes so packed_sa_get can safely read a 64-bit word at the end
    uint64_t sa_map_size = nSAbyte + 8;
    void* sa_ptr = mmap(nullptr, sa_map_size, PROT_READ,
                        MAP_PRIVATE, fd_sa, 0);
    if (sa_ptr == MAP_FAILED) {
        std::cerr << "ERROR: mmap SA failed\n"; return 1;
    }
    madvise(sa_ptr, nSAbyte, MADV_SEQUENTIAL);
    const char* sa_arr = static_cast<const char*>(sa_ptr);

    // ── 4. Allocate hash table ──
    // Need ~2× expected distinct minimizers as table size (load ~0.3)
    // GRCh38 (~300M distinct) → 2^30 = 1.07B entries = 16 GiB
    // GRCm39 (~250M distinct) → 2^29 = 536M entries = 8 GiB
    int table_bits = (nGenomeByte >= 2500000000ULL) ? 30 : 29;
    uint64_t table_size = 1ULL << table_bits;
    uint64_t table_mask = table_size - 1;
    uint64_t table_bytes = table_size * sizeof(HEntry);

    std::cerr << "[build-minimizers] Hash table: 2^" << table_bits
              << " entries = " << table_bytes / 1073741824.0 << " GiB"
              << " (demand-paged)\n";

    // mmap with MAP_ANONYMOUS + demand paging (kernel zeros on first touch)
    HEntry* table = static_cast<HEntry*>(
        mmap(nullptr, table_bytes, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (table == MAP_FAILED) {
        std::cerr << "ERROR: cannot allocate " << table_bytes / 1073741824.0
                  << " GiB hash table\n";
        return 1;
    }

    // ── 5. Scan SA, compute minimizer for each genome position ──
    std::cerr << "[build-minimizers] Scanning SA...\n";

    constexpr int PREFETCH_DIST = 32;
    const uint64_t max_g = nGenomeByte - (uint64_t)window_len;
    uint64_t n_skip = 0;

    for (uint64_t i = 0; i < nSA; i++) {
        // Prefetch future genome accesses
        if (__builtin_expect(i + PREFETCH_DIST < nSA, 1)) {
            uint64_t gp = packed_sa_get(sa_arr, i + PREFETCH_DIST, sa_bits);
            if (gp <= max_g)
                __builtin_prefetch(genome + gp, 0, 0);
        }

        uint64_t g = packed_sa_get(sa_arr, i, sa_bits);
        if (__builtin_expect(g > max_g, 0)) { n_skip++; continue; }

        uint64_t h = compute_minimizer(genome + g, k, w);
        if (__builtin_expect(h == 0, 0)) { n_skip++; continue; }

        ht_update(table, table_mask, h, static_cast<uint32_t>(i));
    }

    std::cerr << "[build-minimizers] Scan done. skipped=" << n_skip << "\n";
    munmap(sa_ptr, sa_map_size);
    close(fd_sa);
    munmap(genome_ptr, nGenomeByte);
    close(fd_genome);

    // ── 6. Collect valid entries (filter SA_hi - SA_lo > T) ──
    std::cerr << "[build-minimizers] Collecting valid records (T=" << T << ")...\n";

    std::vector<MinimizerRecord> records;
    records.reserve(300000000);

    for (uint64_t s = 0; s < table_size; s++) {
        const HEntry& e = table[s];
        if (e.hash == 0) continue;
        if ((e.hi - e.lo) > static_cast<uint32_t>(T)) continue;
        records.push_back({e.hash, e.lo, e.hi});
    }
    munmap(table, table_bytes);

    std::cerr << "[build-minimizers] Collected " << records.size()
              << " records. Sorting by hash...\n";

    // ── 7. Sort by hash for binary search at query time ──
    std::sort(records.begin(), records.end(),
              [](const MinimizerRecord& a, const MinimizerRecord& b) {
                  return a.hash < b.hash;
              });

    // ── 8. Write minimizerIndex.bin ──
    std::string out_path = genome_dir + "/minimizerIndex.bin";
    std::cerr << "[build-minimizers] Writing " << out_path << "...\n";

    FILE* out = fopen(out_path.c_str(), "wb");
    if (!out) {
        std::cerr << "ERROR: cannot write " << out_path << "\n"; return 1;
    }
    MinimizerIndexHeader hdr{};
    memcpy(hdr.magic, "MINIIDX", 8);
    hdr.k        = k;
    hdr.w        = w;
    hdr.T        = T;
    hdr.reserved = 0;
    hdr.nRecords = records.size();
    fwrite(&hdr, sizeof(hdr), 1, out);
    fwrite(records.data(), sizeof(MinimizerRecord), records.size(), out);
    fclose(out);

    uint64_t file_bytes = sizeof(hdr) + records.size() * sizeof(MinimizerRecord);
    std::cerr << "[build-minimizers] Done!\n"
              << "  records    : " << records.size() << "\n"
              << "  index size : " << file_bytes / 1073741824.0 << " GiB\n"
              << "  output     : " << out_path << "\n";
    return 0;
}

// ════════════════════════════════════════════════════════════
// singlet nonhost — Cache k-mer indices in /dev/shm for fast loading
// ════════════════════════════════════════════════════════════

static const char* NONHOST_SHM_DIR = "/dev/shm/singlet_nonhost";

static int cmd_nonhost(int argc, char* argv[]) {
    if (argc < 1) {
        std::cerr
            << "singlet nonhost — Cache nonhost screening indices in /dev/shm\n\n"
            << "Usage:\n"
            << "  singlet nonhost load [--ref-base PATH]  Copy indices to /dev/shm/singlet_nonhost/\n"
            << "  singlet nonhost unload                  Remove cached indices from /dev/shm\n"
            << "  singlet nonhost status                  Show which indices are cached\n\n"
            << "When indices are cached, subsequent 'singlet process' runs load from RAM disk\n"
            << "(~5-10s) instead of NFS (~1300s).\n";
        return 0;
    }
    std::string op = argv[0];
    if (op == "help" || op == "-h" || op == "--help") {
        std::cerr
            << "singlet nonhost — Cache nonhost screening indices in /dev/shm\n\n"
            << "Usage:\n"
            << "  singlet nonhost load [--ref-base PATH]  Copy indices to /dev/shm/singlet_nonhost/\n"
            << "  singlet nonhost unload                  Remove cached indices from /dev/shm\n"
            << "  singlet nonhost status                  Show which indices are cached\n\n"
            << "When indices are cached, subsequent 'singlet process' runs load from RAM disk\n"
            << "(~5-10s) instead of NFS (~1300s).\n";
        return 0;
    }

    auto file_size_gb = [](const std::string& p) -> double {
        struct stat st{};
        if (stat(p.c_str(), &st) != 0) return -1.0;
        return static_cast<double>(st.st_size) / 1073741824.0;
    };

    if (op == "status") {
        std::cerr << "[singlet nonhost] Cache directory: " << NONHOST_SHM_DIR << "\n";
        bool any = false;
        for (const char* name : {"viral.snhskidx", "bacterial.snhskidx", "fungal.snhskidx"}) {
            std::string p = std::string(NONHOST_SHM_DIR) + "/" + name;
            double gb = file_size_gb(p);
            if (gb >= 0) {
                std::cerr << "  CACHED  " << name << "  (" << gb << " GiB)\n";
                any = true;
            } else {
                std::cerr << "  missing " << name << "\n";
            }
        }
        if (!any) {
            std::cerr << "  (no indices cached)\n";
            return 1;
        }
        return 0;
    }

    if (op == "unload") {
        struct stat st{};
        if (stat(NONHOST_SHM_DIR, &st) != 0) {
            std::cerr << "[singlet nonhost] Cache directory not found — nothing to unload.\n";
            return 0;
        }
        std::cerr << "[singlet nonhost] Removing cache directory: " << NONHOST_SHM_DIR << "\n";
        std::string cmd = std::string("rm -rf ") + NONHOST_SHM_DIR;
        int rc = system(cmd.c_str());
        if (rc != 0) {
            std::cerr << "[singlet nonhost] ERROR: rm -rf failed (rc=" << rc << ")\n";
            return 1;
        }
        std::cerr << "[singlet nonhost] Nonhost cache removed.\n";
        return 0;
    }

    if (op == "load") {
        // Resolve ref_base: --ref-base flag or SINGLET_REF_BASE env
        std::string ref_base_val;
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if ((a == "--ref-base" || a == "-r") && i + 1 < argc)
                ref_base_val = argv[++i];
        }
        if (ref_base_val.empty()) {
            const char* env = getenv("SINGLET_REF_BASE");
            if (env) ref_base_val = env;
        }
        if (ref_base_val.empty()) {
            std::cerr << "[singlet nonhost] ERROR: --ref-base not specified and SINGLET_REF_BASE not set.\n";
            return 1;
        }
        std::string nhdir = ref_base_val + "/nonhost/";
        // Verify source directory exists
        struct stat nhst{};
        if (stat(nhdir.c_str(), &nhst) != 0) {
            std::cerr << "[singlet nonhost] ERROR: nonhost source directory not found: " << nhdir << "\n";
            return 1;
        }

        // Create destination directory
        if (mkdir(NONHOST_SHM_DIR, 0755) != 0 && errno != EEXIST) {
            std::cerr << "[singlet nonhost] ERROR: cannot create " << NONHOST_SHM_DIR
                      << ": " << strerror(errno) << "\n";
            return 1;
        }

        // Copy each index file
        int overall_rc = 0;
        for (const char* name : {"viral.snhskidx", "bacterial.snhskidx", "fungal.snhskidx"}) {
            std::string src = nhdir + name;
            std::string dst = std::string(NONHOST_SHM_DIR) + "/" + name;
            struct stat src_st{};
            if (stat(src.c_str(), &src_st) != 0) {
                std::cerr << "[singlet nonhost] Skipping " << name << " (not found in " << nhdir << ")\n";
                continue;
            }
            // Skip if already cached with the same size
            struct stat dst_st{};
            if (stat(dst.c_str(), &dst_st) == 0 && dst_st.st_size == src_st.st_size) {
                std::cerr << "[singlet nonhost] Already cached: " << name
                          << " (" << static_cast<double>(src_st.st_size)/1073741824.0 << " GiB)\n";
                continue;
            }
            double gb = static_cast<double>(src_st.st_size) / 1073741824.0;
            std::cerr << "[singlet nonhost] Copying " << name << " (" << gb << " GiB) ...\n";
            StopWatch sw;
            // Use cp via system() — this is a one-time setup operation; cp uses kernel splice/sendfile
            std::string cp_cmd = "cp " + src + " " + dst;
            int rc = system(cp_cmd.c_str());
            if (rc != 0) {
                std::cerr << "[singlet nonhost] ERROR: cp failed for " << name << " (rc=" << rc << ")\n";
                overall_rc = 1;
            } else {
                std::cerr << "[singlet nonhost] Cached " << name
                          << " in " << sw.elapsed_s() << "s (" << gb << " GiB)\n";
            }
        }
        if (overall_rc == 0)
            std::cerr << "[singlet nonhost] All available indices cached in " << NONHOST_SHM_DIR << ".\n"
                      << "Subsequent singlet process runs will load from RAM disk.\n";
        return overall_rc;
    }

    std::cerr << "ERROR: unknown nonhost subcommand: " << op << "\n"
              << "Use: singlet nonhost load|unload|status\n";
    return 1;
}

// ════════════════════════════════════════════════════════════
// singlet genome — Load/unload genome into System V shared memory
// ════════════════════════════════════════════════════════════
static int cmd_genome(int argc, char* argv[]) {
    if (argc < 1) {
        std::cerr
            << "singlet genome — Manage genome in shared memory\n\n"
            << "Usage:\n"
            << "  singlet genome load <genome-dir>          Load genome into shared RAM\n"
            << "  singlet genome unload <genome-dir>        Remove genome from shared RAM\n"
            << "  singlet genome status <genome-dir>        Check if genome is loaded\n"
            << "  singlet genome build-minimizers \n"
            << "           --genome-dir <path>               Build minimizerIndex.bin\n\n"
            << "When the genome is loaded, subsequent 'singlet process' runs skip\n"
            << "the ~40s genome loading phase automatically.\n";
        return 0;
    }
    std::string op = argv[0];
    if (op == "help" || op == "-h" || op == "--help") {
        std::cerr
            << "singlet genome — Manage genome in shared memory\n\n"
            << "Usage:\n"
            << "  singlet genome load <genome-dir>          Load genome into shared RAM\n"
            << "  singlet genome unload <genome-dir>        Remove genome from shared RAM\n"
            << "  singlet genome status <genome-dir>        Check if genome is loaded\n"
            << "  singlet genome build-minimizers \n"
            << "           --genome-dir <path>               Build minimizerIndex.bin\n\n"
            << "When the genome is loaded, subsequent 'singlet process' runs skip\n"
            << "the ~40s genome loading phase automatically.\n";
        return 0;
    }
    // build-minimizers uses different arg parsing (--genome-dir flag)
    if (op == "build-minimizers") {
        std::string gdir;
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if ((a == "--genome-dir" || a == "-g") && i + 1 < argc)
                gdir = argv[++i];
            else if (!a.empty() && a[0] != '-')
                gdir = a;  // positional fallback
        }
        if (gdir.empty()) {
            std::cerr << "Usage: singlet genome build-minimizers --genome-dir <path>\n";
            return 1;
        }
        struct stat bst{};
        if (stat(gdir.c_str(), &bst) != 0) {
            std::cerr << "ERROR: genome-dir not found: " << gdir << "\n";
            return 1;
        }
        return cmd_genome_build_minimizers(gdir);
    }
    if (argc < 2) {
        std::cerr << "ERROR: genome-dir required\n"
                  << "Usage: singlet genome " << op << " <genome-dir>\n";
        return 1;
    }
    std::string genome_dir = argv[1];
    struct stat st{};
    if (stat(genome_dir.c_str(), &st) != 0) {
        std::cerr << "ERROR: genome-dir not found: " << genome_dir << "\n";
        return 1;
    }

    if (op == "status") {
        bool loaded = star_genome_in_shm(genome_dir);
        key_t k = star_genome_shmkey(genome_dir);
        std::cerr << "[singlet genome] " << genome_dir << "\n"
                  << "  shmKey = 0x" << std::hex << k << std::dec << "\n"
                  << "  status = " << (loaded ? "LOADED (in shared memory)" : "NOT loaded") << "\n";
        return loaded ? 0 : 1;
    }

    if (op == "load") {
        if (star_genome_in_shm(genome_dir)) {
            std::cerr << "[singlet genome] Genome already in shared memory — nothing to do.\n";
            return 0;
        }
        std::cerr << "[singlet genome] Loading genome into shared memory: " << genome_dir << "\n";
        StopWatch sw;
        std::string tmp_prefix = std::string("/dev/shm/singlet_genome_load_") + std::to_string(getpid());
        int rc = run_star_genome_op(genome_dir, "LoadAndExit", tmp_prefix);
        // STAR exits with 0 from within LoadAndExit (calls exit(0) internally),
        // so rc==0 is expected. Verify the segment actually appeared.
        if (!star_genome_in_shm(genome_dir)) {
            std::cerr << "[singlet genome] ERROR: genome load failed (shmid not found after STAR exit).\n"
                      << "  STAR log: " << tmp_prefix << "/star_genome.log\n";
            return 1;
        }
        std::cerr << "[singlet genome] Genome loaded in " << sw.elapsed_s() << "s. "
                  << "Subsequent singlet process runs will reuse shared memory.\n";
        // Clean up STAR log dir
        std::string rmcmd = "rm -rf " + tmp_prefix;
        system(rmcmd.c_str());
        return 0;
    }

    if (op == "unload") {
        if (!star_genome_in_shm(genome_dir)) {
            std::cerr << "[singlet genome] Genome not in shared memory — nothing to unload.\n";
            return 0;
        }
        std::cerr << "[singlet genome] Removing genome from shared memory: " << genome_dir << "\n";
        std::string tmp_prefix = std::string("/dev/shm/singlet_genome_unload_") + std::to_string(getpid());
        int rc = run_star_genome_op(genome_dir, "Remove", tmp_prefix);
        if (star_genome_in_shm(genome_dir)) {
            std::cerr << "[singlet genome] WARNING: genome may still be in shared memory.\n";
        } else {
            std::cerr << "[singlet genome] Genome removed from shared memory.\n";
        }
        std::string rmcmd = "rm -rf " + tmp_prefix;
        system(rmcmd.c_str());
        return 0;
    }

    std::cerr << "ERROR: unknown genome subcommand: " << op << "\n"
              << "Use: singlet genome load|unload|status <genome-dir>\n";
    return 1;
}

static void usage(const char* prog) {
    std::cerr
        << "singlet v0.3.0 — .1fq → align → pileup → .1pz\n\n"
        << "Subcommands:\n"
        << "  " << prog << " <file.1fq> [options]        Process .1fq → .1pz (align + pileup)\n"
        << "  " << prog << " download <SRR> [-o]         Stream SRA → .1fq archive\n"
        << "  " << prog << " encode --reads R1 R2 [-o]   Encode FASTQ pair → .1fq archive\n"
        << "  " << prog << " decode <file.1fq> [--r1 R1 --r2 R2]  Decode .1fq → FASTQ\n"
        << "  " << prog << " genome load <genome-dir>    Load genome into shared RAM (~40s, once)\n"
        << "  " << prog << " genome unload <genome-dir>  Remove genome from shared RAM\n"
        << "  " << prog << " genome status <genome-dir>  Check if genome is in shared RAM\n"
        << "  " << prog << " nonhost load                Cache nonhost k-mer indices in /dev/shm (~60s)\n"
        << "  " << prog << " nonhost unload              Remove nonhost cache from /dev/shm\n"
        << "  " << prog << " nonhost status              Show cached nonhost index sizes\n\n"
        << "Processing options (requires .1fq input):\n"
        << "  --genome-dir DIR       Aligner genome index directory\n"
        << "  --ref-base DIR         Reference root dir for auto-detection (overrides\n"
        << "                         SINGLET_REF_BASE env var)\n"
        << "  --whitelist FILE       Barcode whitelist (for barcode correction)\n"
        << "  --barcodes FILE        Filtered barcode list\n\n"
        << "Feature layers (at least one required):\n"
        << "  --exons FILE           GTF for gene model\n"
        << "  --snps FILE            VCF of SNP positions\n"
        << "  --feature-ref FILE     CITE-seq ADT/HTO tag reference CSV (name,sequence)\n\n"
        << "Output:\n"
        << "  --out-prefix DIR       Output directory [./singlet_out]\n"
        << "  --output-format F      1pz (default), mtx, h5ad, loom, or all (1pz+h5ad)\n"
        << "  --tagged-bam [PATH]    Write Cell Ranger-compatible tagged BAM with CB/UB/GX/GN/RE\n"
        << "                         tags (default: {out_prefix}/possorted_genome_bam.bam).\n"
        << "                         Requires parallel pileup (BAM on disk).\n\n"
        << "Performance:\n"
        << "  --threads N            Alignment threads [auto: $SLURM_CPUS_PER_TASK or hardware_concurrency]\n"
        << "  --genome-shared [MODE] Shared-memory genome mode: auto (default), on, off\n"
        << "                         auto: enable on Clipper-class HPC if genome is in SHM\n"
        << "                         on: always use LoadAndKeep (genome must be pre-loaded)\n"
        << "                         off: always load genome from disk\n"
        << "  --genome-load MODE     Genome loading [NoSharedMemory]\n\n"
        << "Optional:\n"
        << "  --pipeline             Enable donor demux + mt heteroplasmy\n"
        << "  --cascade MODE         Cascade aligner: off (default), on, auto\n"
        << "  --te-classify MODE     TE/repeat classification: off (default), on\n"
        << "  --cascade-stats        Write cascade_stats.json to output directory\n"
        << "  --txome-index PATH     Transcriptome FASTA for L1 resolve (tx.fa)\n"
        << "  --te-db PATH           TE sketch DB for L2 classify (te_sketch.bin)\n"
        << "  --n-donors N           Number of donors (-1=auto) [-1]\n"
        << "  --umi-len N            UMI length [12]\n"
        << "  --star-extra 'ARGS'    Extra aligner arguments (quoted)\n"
        << "  --min-mapq N           Minimum MAPQ [20]\n"
        << "  --min-baseq N          Minimum base quality [10]\n"
        << "  --no-introns           Disable intron counting\n"
        << "  --no-sj                Disable splice junction extraction\n"
        << "  --no-umi-dedup         Disable UMI deduplication\n"
        << "  --cell-calling         Force EmptyDrops cell calling (default: auto when no --barcodes)\n"
        << "  --no-cell-calling      Disable EmptyDrops cell calling\n"
        << "  --fdr-threshold F      FDR cutoff for cell calls [0.001]\n"
        << "  --lower-umi N          Lower UMI bound for ambient pool [100]\n"
        << "  --raw-matrix           Write raw_feature_bc_matrix/ (all barcodes) alongside filtered\n"
        << "  --metadata-json FILE   JSON file with GEO/pipeline metadata to embed in .1pz output\n"
        << "  --nonhost-db PATH      Viral sketch database (.snhskidx); auto-discovered from SINGLET_REF_BASE/nonhost/\n"
        << "  --bacterial-db PATH    Bacterial sketch database; auto-discovered alongside viral DB\n"
        << "  --fungal-db PATH       Fungal sketch database; auto-discovered alongside viral DB\n"
        << "  --nonhost-ref-dir PATH Directory of per-species FASTAs for secondary alignment\n"
        << "  --no-viral-screen      Disable viral screening\n"
        << "  --no-microbial-screen  Disable bacterial+fungal screening\n"
        << "  --nonhost-only         Screen only, skip Phase 2 alignment (future)\n"
        << "  --bam-sort-mode MODE   BAM sort strategy: auto (default), memory, or disk\n"
        << "                         auto: picks mode based on estimated BAM size vs available RAM\n"
        << "                         memory: force in-memory sort (current behaviour, fastest)\n"
        << "                         disk: force disk-backed sort (prevents OOM on very large samples)\n\n"
        << "Legacy (deprecated — archive to .1fq first, then process):\n"
        << "  --reads R2 R1          FASTQ files (use 'singlet encode' instead)\n"
        << "  --sra FILE             SRA file (use 'singlet download' instead)\n"
        << "  --bam-file BAM         Read from existing BAM (debug only)\n"
        << "  --1fq FILE             Equivalent to positional .1fq argument\n"
        << "  --star-threads N       Alias for --threads\n"
        << "  --variant-calling      Enable RNA variant calling (Smart-seq2/Bulk only)\n"
        << "  --min-coverage INT     Min depth for variant call [10]\n"
        << "  --min-alt-count INT    Min alt-allele reads for variant call [3]\n"
        << "  --min-vaf FLOAT        Min variant allele frequency [0.1]\n"
        << "  --dedup-stats          Collect PCR+optical duplicate statistics (SS2/Bulk only)\n"
        << "  --optical-distance INT Pixel distance threshold for optical dups [2500]\n"
        << "  -h, --help             Show this help\n";
}

int main(int argc, char* argv[]) {
    // ── Subcommand dispatch ──
    if (argc >= 2) {
        std::string sub = argv[1];
        try {
            if (sub == "download")
                return cmd_download(argc - 2, argv + 2);
            if (sub == "encode")
                return cmd_encode(argc - 2, argv + 2);
            if (sub == "decode")
                return cmd_decode(argc - 2, argv + 2);
            if (sub == "archive")
                return cmd_archive(argc - 2, argv + 2);
            if (sub == "genome")
                return cmd_genome(argc - 2, argv + 2);
            if (sub == "nonhost")
                return cmd_nonhost(argc - 2, argv + 2);
            if (sub == "build-nonhost-db")
                return cmd_build_nonhost_db(argc - 2, argv + 2);
            if (sub == "build-host-filter")
                return cmd_build_host_filter(argc - 2, argv + 2);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << "\n";
            return 1;
        }
        if (sub == "process" && argc >= 3) {
            // `singlet process file.1fq [opts]` → shift argv so file.1fq
            // is picked up as positional arg below
            argc--; argv++;
        }
    }

    // ── Parse arguments ──
    std::string genome_dir, reads_r2, reads_r1;
    std::string ref_base;
    std::string barcode_file, whitelist_file;
    std::string out_prefix = "./singlet_out";
    std::string output_format = "1pz";
    std::string genome_load = "NoSharedMemory";
    std::string bam_file;
    std::string sra_file;
    std::string onefq_file;
    std::vector<std::string> star_extra;
    int star_threads = 0;  // 0 = auto-detect
    bool threads_specified = false;
    int pileup_threads = 1;  // bgzf not useful for uncompressed pipe from STAR
    int parallel_pileup = -1; // -1=auto, 0=off, 1=on
    int n_donors = -1;
    int umi_len = 12;
    bool umi_len_explicit = false;  // S5-UMI-LEN: true when --umi-len given by user
    bool pipeline_mode = false;
    bool no_dir_dedup_explicit = false; // true if user passed --no-umi-dedup-directional
    bool user_provided_barcodes = false; // true if --barcodes was explicitly given
    int cell_calling_flag = -1;          // -1=auto, 0=off, 1=on (--cell-calling / --no-cell-calling)
    double fdr_threshold_flag = 0.01;
    int lower_umi_flag = 100;
    int forced_cells_flag = 0;           // --forced-cells N: skip EmptyDrops, take top-N by UMI
    int expect_cells_flag = 0;           // --expect-cells N: hint for cell calling knee detection
    bool write_raw_matrix = false;       // --raw-matrix: write raw_feature_bc_matrix/ alongside filtered
    bool variant_calling = false;
    int  vc_min_coverage = 10;
    int  vc_min_alt_count = 3;
    float vc_min_vaf = 0.1f;
    bool do_dedup_stats = false;
    int  optical_distance = 2500;

    std::string feature_ref_path;    // T4: CITE-seq ADT/HTO reference CSV
    std::string tissue_positions_path; // V2: Visium tissue_positions.csv
    std::string metadata_json_path;  // GEO metadata JSON for .1pz embedding
    std::string tagged_bam_path;     // G-TAGGED-BAM: output path (empty = default)
    std::string nonhost_db_path;     // NONHOST-UNMAPPED-CAPTURE: primary (viral) screening index (legacy)
    std::string bacterial_db_path;   // NONHOST: bacterial screening index
    std::string fungal_db_path;      // NONHOST: fungal screening index
    std::string nonhost_ref_dir;     // NONHOST-SECONDARY-ALIGN: directory for per-species FASTAs
    bool no_viral_screen    = false; // --no-viral-screen: skip viral index loading
    bool no_microbial_screen = false;// --no-microbial-screen: skip bacterial+fungal index loading
    bool nonhost_only       = false; // --nonhost-only: screen but skip Phase 2 alignment (future)
    bool tagged_bam_enabled = false; // G-TAGGED-BAM: write tagged BAM after pileup
    uint32_t visium_min_umi = 100;     // V3: UMI threshold for tissue coverage metric
    std::string bam_sort_mode = "auto"; // ADAPTIVE-BAM-SORT: auto|memory|disk

    // Track B cascade (T-L2-7 CLI flags)
    std::string cascade_mode = "off";    // --cascade {off,on,auto}
    std::string te_classify_mode = "off"; // --te-classify {off,on}
    bool cascade_stats_enabled = false;   // --cascade-stats
    std::string txome_index_path;         // --txome-index PATH (tx.fa for L1)
    std::string te_db_path;              // --te-db PATH (te_sketch.bin for L2)

    // PileupConfig fields we parse directly
    singlet::PileupConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else if (arg == "--genome-dir" && i+1 < argc) genome_dir = argv[++i];
        else if (arg == "--ref-base" && i+1 < argc) ref_base = argv[++i];
        else if (arg == "--reads" && i+2 < argc) { reads_r2 = argv[++i]; reads_r1 = argv[++i]; }
        else if (arg == "--barcodes" && i+1 < argc) { barcode_file = argv[++i]; user_provided_barcodes = true; }
        else if (arg == "--whitelist" && i+1 < argc) whitelist_file = argv[++i];
        else if (arg == "--exons" && i+1 < argc) config.exon_gtf_path = argv[++i];
        else if (arg == "--snps" && i+1 < argc) config.snp_path = argv[++i];
        else if (arg == "--out-prefix" && i+1 < argc) out_prefix = argv[++i];
        else if (arg == "--star-threads" && i+1 < argc) { star_threads = std::atoi(argv[++i]); threads_specified = true; }
        else if (arg == "--pileup-threads" && i+1 < argc) pileup_threads = std::atoi(argv[++i]);
        else if (arg == "--parallel-pileup") parallel_pileup = 1;
        else if (arg == "--no-parallel-pileup") parallel_pileup = 0;
        else if (arg == "--output-format" && i+1 < argc) output_format = argv[++i];
        else if (arg == "--pipeline") pipeline_mode = true;
        else if (arg == "--cascade" && i+1 < argc) cascade_mode = argv[++i];
        else if (arg == "--te-classify" && i+1 < argc) te_classify_mode = argv[++i];
        else if (arg == "--cascade-stats") cascade_stats_enabled = true;
        else if (arg == "--txome-index" && i+1 < argc) txome_index_path = argv[++i];
        else if (arg == "--te-db" && i+1 < argc) te_db_path = argv[++i];
        else if (arg == "--n-donors" && i+1 < argc) n_donors = std::atoi(argv[++i]);
        else if (arg == "--genome-load" && i+1 < argc) genome_load = argv[++i];
        else if (arg == "--genome-shared") {
            // Optional value: auto (default), on, off
            if (i + 1 < argc) {
                std::string nxt = argv[i + 1];
                if (nxt == "on" || nxt == "off" || nxt == "auto") {
                    ++i;
                    if (nxt == "on")  genome_load = "LoadAndKeep";
                    else if (nxt == "off") genome_load = "_off_";
                    // "auto" leaves genome_load = "NoSharedMemory" → N21 decides
                } else {
                    // No value: treat as legacy bare flag = "on"
                    genome_load = "LoadAndKeep";
                }
            } else {
                genome_load = "LoadAndKeep";  // legacy bare flag = on
            }
        }
        else if (arg == "--umi-len" && i+1 < argc) { umi_len = std::atoi(argv[++i]); umi_len_explicit = true; }
        else if (arg == "--bam-file" && i+1 < argc) bam_file = argv[++i];
        else if (arg == "--min-mapq" && i+1 < argc) config.min_mapq = std::atoi(argv[++i]);
        else if (arg == "--min-baseq" && i+1 < argc) config.min_baseq = std::atoi(argv[++i]);
        else if (arg == "--no-introns") config.count_introns = false;
        else if (arg == "--no-sj") config.count_sj = false;
        else if (arg == "--umi-dedup-directional") config.umi_dedup_directional = true;
        else if (arg == "--no-umi-dedup-directional") { config.umi_dedup_directional = false; no_dir_dedup_explicit = true; }
        else if (arg == "--no-umi-dedup") config.umi_dedup = false;
        else if (arg == "--reverse-strand") config.reverse_strand = true;
        else if (arg == "--guide-ref" && i+1 < argc) config.guide_ref_path = argv[++i]; // N18: CRISPR guide CSV
        else if (arg == "--feature-ref" && i+1 < argc) feature_ref_path = argv[++i]; // T4: CITE-seq ADT/HTO reference CSV
        else if (arg == "--cell-calling") cell_calling_flag = 1;
        else if (arg == "--no-cell-calling") cell_calling_flag = 0;
        else if (arg == "--fdr-threshold" && i+1 < argc) fdr_threshold_flag = std::atof(argv[++i]);
        else if (arg == "--lower-umi" && i+1 < argc) lower_umi_flag = std::atoi(argv[++i]);
        else if (arg == "--forced-cells" && i+1 < argc) forced_cells_flag = std::atoi(argv[++i]);
        else if (arg == "--expect-cells" && i+1 < argc) expect_cells_flag = std::atoi(argv[++i]);
        else if (arg == "--raw-matrix") write_raw_matrix = true;
        else if (arg == "--variant-calling") variant_calling = true;
        else if (arg == "--min-coverage" && i+1 < argc) vc_min_coverage = std::atoi(argv[++i]);
        else if (arg == "--min-alt-count" && i+1 < argc) vc_min_alt_count = std::atoi(argv[++i]);
        else if (arg == "--min-vaf" && i+1 < argc) vc_min_vaf = std::stof(argv[++i]);
        else if (arg == "--dedup-stats") do_dedup_stats = true;
        else if (arg == "--optical-distance" && i+1 < argc) optical_distance = std::atoi(argv[++i]);
        else if (arg == "--tissue-positions" && i+1 < argc) tissue_positions_path = argv[++i]; // V2
        else if (arg == "--visium-min-umi" && i+1 < argc) visium_min_umi = static_cast<uint32_t>(std::atoi(argv[++i])); // V3
        else if (arg == "--threads" && i+1 < argc) { star_threads = std::atoi(argv[++i]); threads_specified = true; }
        else if (arg == "--sra" && i+1 < argc) sra_file = argv[++i];
        else if (arg == "--1fq" && i+1 < argc) onefq_file = argv[++i];
        else if (arg[0] != '-' && onefq_file.empty() && arg.size() > 4 &&
                 arg.substr(arg.size() - 4) == ".1fq") {
            onefq_file = arg;  // positional .1fq argument
        }
        else if (arg == "--star-extra" && i+1 < argc) {
            std::istringstream iss(argv[++i]);
            std::string token;
            while (iss >> token) star_extra.push_back(token);
        }
        else if (arg == "--metadata-json" && i+1 < argc) metadata_json_path = argv[++i];
        else if (arg == "--bam-sort-mode" && i+1 < argc) {
            bam_sort_mode = argv[++i];
            if (bam_sort_mode != "auto" && bam_sort_mode != "memory" && bam_sort_mode != "disk") {
                std::cerr << "ERROR: --bam-sort-mode must be auto, memory, or disk\n";
                return 1;
            }
        }
        else if (arg == "--tagged-bam") {
            tagged_bam_enabled = true;
            // Optional path: consume next arg if it doesn't look like a flag
            if (i + 1 < argc && argv[i + 1][0] != '-')
                tagged_bam_path = argv[++i];
        }
        else if (arg == "--nonhost-db"        && i+1 < argc) nonhost_db_path    = argv[++i]; // NONHOST-UNMAPPED-CAPTURE (legacy; treated as viral DB)
        else if (arg == "--bacterial-db"      && i+1 < argc) bacterial_db_path  = argv[++i]; // NONHOST: bacterial index
        else if (arg == "--fungal-db"         && i+1 < argc) fungal_db_path     = argv[++i]; // NONHOST: fungal index
        else if (arg == "--nonhost-ref-dir"   && i+1 < argc) nonhost_ref_dir    = argv[++i]; // NONHOST-SECONDARY-ALIGN: per-species FASTA dir
        else if (arg == "--no-viral-screen")    no_viral_screen    = true;  // skip viral index
        else if (arg == "--no-microbial-screen") no_microbial_screen = true; // skip bacterial+fungal
        else if (arg == "--nonhost-only")        nonhost_only       = true;  // screen-only mode (future)
        else { std::cerr << "Unknown option: " << arg << "\n"; usage(argv[0]); return 1; }
    }

    // ── N22: Resolve thread count ──
    if (!threads_specified) {
        const char* slurm_cpus = std::getenv("SLURM_CPUS_PER_TASK");
        if (slurm_cpus && *slurm_cpus) {
            star_threads = std::atoi(slurm_cpus);
        } else {
            star_threads = static_cast<int>(std::thread::hardware_concurrency());
            if (star_threads <= 0) star_threads = 8;  // conservative fallback
        }
    }

    // ── NONHOST-AUTO-DISCOVER: resolve viral/bacterial/fungal DB paths ──────────
    // Priority order:
    //   1. Explicit CLI flags (--nonhost-db / --bacterial-db / --fungal-db)
    //   2. Auto-discovery from same directory as the first explicit DB
    //   3. Auto-discovery from $SINGLET_REF_BASE/nonhost/ (auto-enable when indices exist)
    //
    // --nonhost-db is treated as the viral DB (legacy alias); if neither --nonhost-db
    // nor --bacterial-db/--fungal-db is given, we auto-discover from SINGLET_REF_BASE.
    {
        // Treat legacy --nonhost-db as the viral path
        if (!nonhost_db_path.empty() && !no_viral_screen) {
            // viral_db_path was not set yet — promote nonhost_db_path to it
        }
        // We will resolve effective paths into local vars and then assign them.
        // Use separate block to avoid shadowing the outer declarations.

        // Step 1: determine the viral DB path (unless disabled)
        std::string eff_viral_db;
        if (!no_viral_screen) {
            if (!nonhost_db_path.empty())
                eff_viral_db = nonhost_db_path;  // explicit --nonhost-db → viral slot
        }

        // Step 2: determine bacterial/fungal paths (unless microbial screen disabled)
        std::string eff_bact_db;
        std::string eff_fung_db;
        if (!no_microbial_screen) {
            eff_bact_db = bacterial_db_path;
            eff_fung_db = fungal_db_path;
        }

        // Step 3: auto-discover from same directory as the first explicit DB
        if (!eff_viral_db.empty() || !eff_bact_db.empty() || !eff_fung_db.empty()) {
            const std::string& anchor = !eff_viral_db.empty() ? eff_viral_db :
                                        !eff_bact_db.empty()  ? eff_bact_db  :
                                                                  eff_fung_db;
            std::string db_dir = anchor.substr(0, anchor.rfind('/') + 1);
            auto file_exists = [](const std::string& p) {
                struct stat st{};
                return stat(p.c_str(), &st) == 0;
            };
            if (eff_viral_db.empty() && !no_viral_screen) {
                std::string candidate = db_dir + "viral.snhskidx";
                if (file_exists(candidate)) eff_viral_db = candidate;
            }
            if (eff_bact_db.empty() && !no_microbial_screen) {
                std::string candidate = db_dir + "bacterial.snhskidx";
                if (file_exists(candidate)) eff_bact_db = candidate;
            }
            if (eff_fung_db.empty() && !no_microbial_screen) {
                std::string candidate = db_dir + "fungal.snhskidx";
                if (file_exists(candidate)) eff_fung_db = candidate;
            }
        }

        // Step 4: auto-discover from /dev/shm/singlet_nonhost/ (cached) or
        //         $SINGLET_REF_BASE/nonhost/ (NFS fallback).  (auto-enable)
        if (eff_viral_db.empty() && eff_bact_db.empty() && eff_fung_db.empty()) {
            auto file_exists = [](const std::string& p) {
                struct stat st{};
                return stat(p.c_str(), &st) == 0;
            };
            // /dev/shm cache takes priority — deterministic, not subject to NFS cache pressure
            const std::string shm_nhdir = std::string(NONHOST_SHM_DIR) + "/";
            bool shm_has_any = file_exists(shm_nhdir + "viral.snhskidx")
                             || file_exists(shm_nhdir + "bacterial.snhskidx")
                             || file_exists(shm_nhdir + "fungal.snhskidx");
            const char* ref_base_env = getenv("SINGLET_REF_BASE");
            std::string nfs_nhdir = ref_base_env ? std::string(ref_base_env) + "/nonhost/" : "";
            // Probe both directories; shm preferred when file exists there
            auto resolve_db = [&](const std::string& filename) -> std::string {
                std::string shm_p = shm_nhdir + filename;
                if (file_exists(shm_p)) return shm_p;  // RAM-disk cache hit
                if (!nfs_nhdir.empty()) {
                    std::string nfs_p = nfs_nhdir + filename;
                    if (file_exists(nfs_p)) return nfs_p;  // NFS fallback
                }
                return {};  // not found
            };
            if (shm_has_any || ref_base_env) {
                if (!no_viral_screen) {
                    std::string v = resolve_db("viral.snhskidx");
                    if (!v.empty()) eff_viral_db = v;
                }
                if (!no_microbial_screen) {
                    std::string b = resolve_db("bacterial.snhskidx");
                    if (!b.empty()) eff_bact_db = b;
                    std::string f = resolve_db("fungal.snhskidx");
                    if (!f.empty()) eff_fung_db = f;
                }
            }
        }

        // Commit resolved paths back to the named variables used in screening calls
        nonhost_db_path    = eff_viral_db;   // viral slot (also satisfies old !nonhost_db_path.empty() checks)
        bacterial_db_path  = eff_bact_db;
        fungal_db_path     = eff_fung_db;

        if (!nonhost_db_path.empty() || !bacterial_db_path.empty() || !fungal_db_path.empty()) {
            std::cerr << "[nonhost] Screening enabled:"
                      << (!nonhost_db_path.empty()   ? " viral="    + nonhost_db_path   : "")
                      << (!bacterial_db_path.empty()  ? " bacterial=" + bacterial_db_path : "")
                      << (!fungal_db_path.empty()     ? " fungal="   + fungal_db_path    : "")
                      << "\n";
        }
    }

    // ── Validate ──
    bool sra_mode = !sra_file.empty();
    bool onefq_mode = !onefq_file.empty();

    // Deprecation warnings for legacy input paths
    if (!reads_r2.empty()) {
        std::cerr << "WARNING: --reads is deprecated. Archive to .1fq first:\n"
                  << "  singlet encode --reads " << reads_r1 << " " << reads_r2
                  << " -o sample.1fq\n"
                  << "  singlet sample.1fq [options]\n"
                  << "Proceeding with legacy FASTQ path...\n\n";
    }
    if (sra_mode) {
        std::cerr << "WARNING: --sra is deprecated. Archive to .1fq first:\n"
                  << "  singlet download " << sra_file << " -o sample.1fq\n"
                  << "  singlet sample.1fq [options]\n"
                  << "Proceeding with legacy SRA path...\n\n";
    }
    if (!bam_file.empty()) {
        std::cerr << "NOTE: --bam-file is a debug/interop path. "
                  << "Production workflow: singlet <file.1fq> [options]\n\n";
    }

    if ((int)!reads_r2.empty() + (int)sra_mode + (int)onefq_mode + (int)!bam_file.empty() > 1) {
        std::cerr << "ERROR: --reads, --sra, --1fq, --bam-file are mutually exclusive\n";
        return 1;
    }

    // ── METADATA-JSON PROTOCOL OVERRIDE: parse early to allow overriding .1fq header protocol ──
    // If --metadata-json contains a "protocol" key with a valid known-protocol tag,
    // that tag overrides the protocol_id embedded in the .1fq header for ALL downstream
    // parameter selection: whitelist, CBlen, UMIlen, adapter3p, soloType.
    uint8_t meta_json_proto_id = 255;   // 255 = no override
    std::string meta_json_proto_tag;    // human-readable tag for log messages
    if (!metadata_json_path.empty()) {
        std::ifstream mj_early(metadata_json_path);
        if (mj_early) {
            std::string mjson((std::istreambuf_iterator<char>(mj_early)),
                               std::istreambuf_iterator<char>());
            // Simple extraction: find "protocol" : "<value>" in the JSON
            auto mj_extract_str = [&](const std::string& key) -> std::string {
                auto pos = mjson.find('"' + key + '"');
                if (pos == std::string::npos) return "";
                pos = mjson.find(':', pos);
                if (pos == std::string::npos) return "";
                pos = mjson.find('"', pos + 1);
                if (pos == std::string::npos) return "";
                auto end = mjson.find('"', pos + 1);
                if (end == std::string::npos) return "";
                return mjson.substr(pos + 1, end - pos - 1);
            };
            std::string proto_override = mj_extract_str("protocol");
            if (!proto_override.empty()) {
                const lib1fq::CandidateSpec* spec = lib1fq::find_protocol_spec(proto_override);
                if (spec) {
                    meta_json_proto_id  = spec->protocol_id;
                    meta_json_proto_tag = spec->tag;
                } else {
                    std::cerr << "[singlet] WARNING: --metadata-json protocol='" << proto_override
                              << "' not found in known_protocols() — ignoring override\n";
                }
            }
        }
    }

    // ── N4: Auto-resolve whitelist from .1fq metadata when --whitelist not given ──
    // Use a raw 96-byte header read rather than Reader::open() so that files with
    // a corrupted/unreadable BC dict or block index still resolve their whitelist.
    // Reader::open() decompresses the BC dict (can throw: wrong codec, truncated data)
    // and reads the block index (can throw: mismatched footer). A silent catch would
    // leave whitelist_file empty and produce a misleading "requires --genome-dir" error.
    bool whitelist_auto_resolved = false;
    if (onefq_mode && whitelist_file.empty()) {
        uint8_t pid = 0;
        bool header_ok = false;
        {
            FILE* fp_probe = std::fopen(onefq_file.c_str(), "rb");
            if (fp_probe) {
                lib1fq::Header raw_hdr;
                if (std::fread(&raw_hdr, sizeof(lib1fq::Header), 1, fp_probe) == 1 &&
                    raw_hdr.valid_magic()) {
                    pid = raw_hdr.protocol_id;
                    // Apply metadata-json protocol override to whitelist resolution
                    if (meta_json_proto_id != 255)
                        pid = meta_json_proto_id;
                    header_ok = true;
                }
                std::fclose(fp_probe);
            }
        }

        if (header_ok) {
            std::string wl_name, proto_tag;
            for (const auto& spec : lib1fq::known_protocols()) {
                if (spec.protocol_id == pid) {
                    wl_name = spec.whitelist_file;
                    proto_tag = spec.tag;
                    break;
                }
            }
            if (!wl_name.empty()) {
                // Resolve path: <binary-dir>/../whitelists/<file>
                bool resolved = false;
                char exe_buf[PATH_MAX];
                ssize_t exe_len = ::readlink("/proc/self/exe", exe_buf, sizeof(exe_buf)-1);
                if (exe_len > 0) {
                    exe_buf[exe_len] = '\0';
                    std::string bin_dir(exe_buf);
                    auto slp = bin_dir.rfind('/');
                    if (slp != std::string::npos) bin_dir.resize(slp);
                    std::string candidate = bin_dir + "/../whitelists/" + wl_name;
                    struct stat wl_stat;
                    if (stat(candidate.c_str(), &wl_stat) == 0) {
                        whitelist_file = candidate;
                        whitelist_auto_resolved = true;
                        resolved = true;
                        std::cerr << "[singlet] Auto-resolved whitelist: " << proto_tag
                                  << " → " << wl_name << "\n";
                    }
                }
                if (!resolved) {
                    // Whitelist file not found at install path (or readlink failed)
                    std::cerr << "[singlet] Whitelist " << wl_name
                              << " not found for " << proto_tag
                              << " — using barcode auto-discovery\n";
                    whitelist_file = "None";
                    whitelist_auto_resolved = true;
                }
            } else {
                // Whitelist-free protocol or unknown protocol_id
                whitelist_file = "None";
                whitelist_auto_resolved = true;
                std::cerr << "[singlet] Protocol "
                          << (proto_tag.empty() ? "unknown" : proto_tag)
                          << " (id=" << (int)pid << ") is whitelist-free"
                             " — using barcode auto-discovery\n";
            }
        } else {
            // Cannot read .1fq header (file not accessible or invalid magic) —
            // fall back to barcode auto-discovery rather than leaving whitelist empty.
            whitelist_file = "None";
            whitelist_auto_resolved = true;
            std::cerr << "[singlet] WARNING: cannot read .1fq header for whitelist "
                         "auto-resolution — using barcode auto-discovery\n";
        }
    }

    // ── N1: Species auto-detection — resolve genome_dir when not supplied ──
    if (onefq_mode && genome_dir.empty()) {
        // Resolve reference base: --ref-base > SINGLET_REF_BASE env > skip
        if (ref_base.empty()) {
            const char* env_ref = std::getenv("SINGLET_REF_BASE");
            if (env_ref && *env_ref) ref_base = env_ref;
        }
        if (!ref_base.empty()) {
            auto det = species_detect::detect(onefq_file, /*verbose=*/true,
                                               metadata_json_path);
            if (det.confident()) {
                std::string candidate = det.suggested_star_dir(ref_base);
                if (!candidate.empty()) {
                    struct stat gd_st;
                    if (stat(candidate.c_str(), &gd_st) == 0) {
                        genome_dir = candidate;
                        std::cerr << "[singlet] Auto-resolved genome: " << det.genome_tag
                                  << " → " << genome_dir << "\n";
                        // Auto-resolve GTF from {species_dir}/genes/genes.gtf if --exons not supplied
                        if (config.exon_gtf_path.empty()) {
                            auto sep = genome_dir.rfind('/');
                            if (sep != std::string::npos) {
                                std::string gtf_candidate = genome_dir.substr(0, sep) + "/genes/genes.gtf";
                                struct stat gtf_st;
                                if (stat(gtf_candidate.c_str(), &gtf_st) == 0) {
                                    config.exon_gtf_path = gtf_candidate;
                                    std::cerr << "[singlet] Auto-resolved GTF: " << config.exon_gtf_path << "\n";
                                }
                            }
                        }
                    } else {
                        std::cerr << "[singlet] Species detected as " << det.genome_tag
                                  << " but genome dir not found: " << candidate << "\n"
                                  << "  Set --genome-dir explicitly or check --ref-base.\n";
                    }
                }
            }
        }
    }

    // ── N21: Shared-memory genome auto-detection (B-G3-2) ──
    // --genome-shared auto (default): enable on Clipper-class HPC nodes if the
    // genome is already in shared memory.  --genome-shared on/off override.
    // "_off_" sentinel means explicit --genome-shared off → skip N21.
    if (genome_load == "_off_") {
        genome_load = "NoSharedMemory";
    } else if (genome_load == "NoSharedMemory" && !genome_dir.empty()) {
        // Auto-detect Clipper-class node: SLURM_CLUSTER_NAME contains "clipper",
        // or SLURMD_NODENAME matches c[0-9]+, or /etc/slurm/slurm.conf exists.
        auto is_clipper_node = []() -> bool {
            const char* cn = std::getenv("SLURM_CLUSTER_NAME");
            if (cn) {
                std::string s = cn;
                for (auto& c : s) c = static_cast<char>(std::tolower(c));
                if (s.find("clipper") != std::string::npos) return true;
            }
            const char* nd = std::getenv("SLURMD_NODENAME");
            if (nd && nd[0] == 'c' && std::isdigit(nd[1])) return true;
            struct stat st{};
            if (stat("/etc/slurm/slurm.conf", &st) == 0) return true;
            return false;
        };
        if (is_clipper_node() && star_genome_in_shm(genome_dir)) {
            genome_load = "LoadAndKeep";
            std::cerr << "[singlet] N21: Clipper node + genome in shared memory — using LoadAndKeep.\n";
        }
    }

    // Primary path: .1fq input
    if (onefq_mode) {
        if (genome_dir.empty() || whitelist_file.empty()) {
            std::cerr << "ERROR: .1fq processing requires --genome-dir (and optionally --whitelist)\n"
                         "  Tip: Whitelist is auto-resolved from .1fq metadata. Use --whitelist None\n"
                         "  for protocols without a universal whitelist (e.g. Drop-seq)\n"
                         "  Tip: Set SINGLET_REF_BASE or use --ref-base for genome auto-detection\n";
            usage(argv[0]);
            return 1;
        }
        // Validate whitelist file exists (unless it's "None")
        if (whitelist_file != "None" && whitelist_file != "none") {
            struct stat wl_st;
            if (stat(whitelist_file.c_str(), &wl_st) != 0) {
                std::cerr << "ERROR: whitelist file not found: " << whitelist_file << "\n";
                return 1;
            }
        }
        struct stat fq_st;
        if (stat(onefq_file.c_str(), &fq_st) != 0) {
            std::cerr << "ERROR: .1fq file not found: " << onefq_file << "\n";
            return 1;
        }
    }
    // Legacy paths
    else if (bam_file.empty() && !sra_mode) {
        if (genome_dir.empty() || reads_r2.empty() ||
            reads_r1.empty() || whitelist_file.empty()) {
            std::cerr << "ERROR: .1fq file required. Usage:\n"
                         "  singlet <file.1fq> --genome-dir DIR --whitelist WL "
                         "--barcodes BC --exons GTF\n\n"
                         "To archive raw data first:\n"
                         "  singlet download <SRR> -o sample.1fq   (from SRA)\n"
                         "  singlet encode --reads R1 R2 -o sample.1fq  (from FASTQ)\n";
            return 1;
        }
    }
    if (sra_mode) {
        if (genome_dir.empty() || whitelist_file.empty()) {
            std::cerr << "ERROR: --sra mode requires --genome-dir, --whitelist\n";
            usage(argv[0]);
            return 1;
        }
        struct stat sra_st;
        if (stat(sra_file.c_str(), &sra_st) != 0) {
            std::cerr << "ERROR: SRA file not found: " << sra_file << "\n";
            return 1;
        }
    }
    // ── Assay early detection: peek .1fq assay_type before loading references ──
    // Runs before engine_ptr creation to allow ATAC/Smart-seq2/Bulk-RNA specific paths.
    bool is_atac_mode = false;
    bool is_smart_seq2_mode = false;
    bool is_bulk_rna_mode = false;
    bool is_cite_seq_mode = false;
    bool is_visium_mode   = false;
    bool is_feature_barcode_only = false; // B-G2-3/B-G8-2: CITE_SEQ_ADT only, no GEX
    uint64_t early_n_reads = 0;           // total_reads from early header probe
    double decode_time = 0.0;             // §3.2 per-stage: .1fq decode wall time
    uint64_t decode_reads_out = 0;        // §3.2 per-stage: reads decoded from .1fq
    if (onefq_mode) {
        lib1fq::Reader early_assay_probe;
        try {
            early_assay_probe.open(onefq_file.c_str());
            auto eahdr = early_assay_probe.header();
            is_atac_mode = (eahdr.assay_type == static_cast<uint8_t>(lib1fq::AssayType::SC_ATAC) ||
                            eahdr.assay_type == static_cast<uint8_t>(lib1fq::AssayType::SC_MULTIOME_ATAC));
            is_smart_seq2_mode = (eahdr.assay_type == static_cast<uint8_t>(lib1fq::AssayType::SC_RNA_PLATE));
            is_bulk_rna_mode  = (eahdr.assay_type == static_cast<uint8_t>(lib1fq::AssayType::BULK_RNA));
            is_cite_seq_mode  = (eahdr.assay_type == static_cast<uint8_t>(lib1fq::AssayType::CITE_SEQ_GEX) ||
                                 eahdr.assay_type == static_cast<uint8_t>(lib1fq::AssayType::CITE_SEQ_ADT));
            is_visium_mode    = (eahdr.assay_type == static_cast<uint8_t>(lib1fq::AssayType::SPATIAL_RNA));
            // B-G2-3: feature-barcode-only = ADT/HTO stream, no GEX
            is_feature_barcode_only = (eahdr.assay_type == static_cast<uint8_t>(lib1fq::AssayType::CITE_SEQ_ADT));
            if (is_atac_mode)
                std::cerr << "[singlet] ATAC assay detected (assay_type=" << (int)eahdr.assay_type << ")\n";
            if (is_cite_seq_mode)
                std::cerr << "[singlet] CITE-seq assay detected (assay_type=" << (int)eahdr.assay_type << ")\n";
            if (is_feature_barcode_only)
                std::cerr << "[singlet] Feature-barcode-only (ADT/HTO) assay detected\n";
            if (is_visium_mode)
                std::cerr << "[singlet] Visium spatial assay detected (assay_type=" << (int)eahdr.assay_type << ")\n";
            if (is_smart_seq2_mode)
                std::cerr << "[singlet] Smart-seq2/plate assay detected (assay_type=" << (int)eahdr.assay_type << ")\n";
            if (is_bulk_rna_mode)
                std::cerr << "[singlet] Bulk RNA-seq assay detected (assay_type=" << (int)eahdr.assay_type << ")\n";
        } catch (...) {}
    }

    // Smart-seq2 / Bulk RNA: no barcodes/whitelist needed — one sample per run.
    // Set barcode_file to a non-empty sentinel so needs_barcode_discovery stays false.
    if (is_smart_seq2_mode || is_bulk_rna_mode) {
        whitelist_auto_resolved = true;
        barcode_file = "SMARTSEQ2_NONE";  // sentinel: bypasses barcode discovery
        whitelist_file = "None";
    }
    if (!is_smart_seq2_mode && !is_bulk_rna_mode && !whitelist_auto_resolved && barcode_file.empty() && (whitelist_file.empty() ||
        (whitelist_file != "None" && whitelist_file != "none"))) {
        std::cerr << "ERROR: --barcodes is required (or use --whitelist None for auto-discovery)\n"; return 1;
    }
    if (!is_atac_mode && !is_smart_seq2_mode && !is_bulk_rna_mode &&
        !(is_cite_seq_mode && !feature_ref_path.empty() && config.exon_gtf_path.empty()) &&
        config.exon_gtf_path.empty() && config.snp_path.empty()) {
        std::cerr << "ERROR: At least one of --exons or --snps required\n"; return 1;
    }
    // Smart-seq2 / Bulk RNA require a GTF (gene model for counting)
    if ((is_smart_seq2_mode || is_bulk_rna_mode) && config.exon_gtf_path.empty()) {
        std::cerr << "ERROR: " << (is_bulk_rna_mode ? "Bulk RNA" : "Smart-seq2") << " mode requires --exons GTF\n"; return 1;
    }

    mkdir_p(out_prefix);

    // B-G2-3 / B-G8-2: Feature-barcode-only library detected (CITE_SEQ_ADT assay, no GEX).
    // Per DROPLET_PRODUCTION_CONTRACT.md §1.2 + §5, this is out-of-scope for the GEX lane.
    // Emit summary.json with status=feature_barcode_not_gex and exit 0 cleanly.
    // Do NOT invoke STAR.
    if (is_feature_barcode_only && config.exon_gtf_path.empty()) {
        std::string fb_sample_id = onefq_file;
        { auto p = fb_sample_id.rfind('/');   if (p != std::string::npos) fb_sample_id = fb_sample_id.substr(p+1); }
        { auto p = fb_sample_id.rfind(".1fq"); if (p != std::string::npos) fb_sample_id = fb_sample_id.substr(0, p); }
        {
            std::ofstream sj(out_prefix + "/summary.json");
            if (sj) {
                sj << "{\n"
                   << "  \"schema_version\": \"1.0\",\n"
                   << "  \"sample_id\": \"" << fb_sample_id << "\",\n"
                   << "  \"status\": \"feature_barcode_not_gex\",\n"
                   << "  \"protocol_id\": 0,\n"
                   << "  \"protocol_name\": \"CITE_SEQ_ADT\",\n"
                   << "  \"species\": \"\",\n"
                   << "  \"reference_build\": \"\",\n"
                   << "  \"n_input_reads\": " << early_n_reads << ",\n"
                   << "  \"n_uniquely_mapped\": 0,\n"
                   << "  \"uniquely_mapped_pct\": 0.0,\n"
                   << "  \"n_cells_called\": 0,\n"
                   << "  \"median_umi_per_cell\": 0.0,\n"
                   << "  \"median_genes_per_cell\": 0.0,\n"
                   << "  \"memory_tier\": \"unknown\",\n"
                   << "  \"peak_rss_gb\": 0.0,\n"
                   << "  \"wall_seconds\": 0.0,\n"
                   << "  \"track\": \"A\",\n"
                   << "  \"donor\": null,\n"
                   << "  \"mt\": null,\n"
                   << "  \"nonhost\": null,\n"
                   << "  \"reason\": \"feature-barcode-only (ADT/HTO) library; no GEX reads present; not in-scope for scRNA lane\"\n"
                   << "}\n";
            }
        }
        std::cerr << "[singlet] feature_barcode_not_gex: summary.json written to " << out_prefix << "\n";
        return 0;  // B-G2-3: clean exit per contract §5
    }

    // Detected barcode length from .1fq metadata (set in Phase 0.5b, used in Phase 1)
    int detected_bc_len = 0;
    int detected_umi_len = 0;
    // N10: protocol-specific 3' adapter sequence (from known_protocols().adapter3p)
    std::string detected_adapter3p;
    // Complex barcode params from .1fq metadata (CB_UMI_Complex)
    std::string detected_soloType;
    std::vector<std::string> detected_cb_positions;
    std::string detected_umi_position;
    int detected_cb_start = 1;  // 1-based for STAR
    int detected_umi_start = -1; // -1 = derive from bc_len+1
    bool reads_inverted = false; // true if barcode is in R2 (R1=cDNA, R2=barcode)
    uint16_t onefq_r1_len = 0; // R1 length from .1fq header (for UMI bounds check)
    uint64_t onefq_n_reads = 0;  // total reads from .1fq header (for tiny-dataset guard)

    // ── Finish config ──
    config.barcode_path = barcode_file;
    // N22: Set full whitelist for ambient profiling (only when a real whitelist file exists)
    if (!whitelist_file.empty() && whitelist_file != "None" && whitelist_file != "none")
        config.whitelist_path = whitelist_file;
    config.threads = pileup_threads;
    config.umi_len = umi_len;
    if (pipeline_mode) config.count_mt = true;
    if (pipeline_mode && !config.umi_dedup_directional && !no_dir_dedup_explicit) config.umi_dedup_directional = true;

    // ── Track B cascade validation (T-L2-7) ─────────────────────────────────
    bool cascade_enabled = false;
    bool te_classify_enabled = false;
    if (cascade_mode == "on") cascade_enabled = true;
    else if (cascade_mode == "auto") cascade_enabled = true; // auto defaults to on when indexes present
    else if (cascade_mode != "off") {
        std::cerr << "ERROR: --cascade must be 'off', 'on', or 'auto' (got '" << cascade_mode << "')\n";
        return 1;
    }
    if (te_classify_mode == "on") te_classify_enabled = true;
    else if (te_classify_mode != "off") {
        std::cerr << "ERROR: --te-classify must be 'off' or 'on' (got '" << te_classify_mode << "')\n";
        return 1;
    }
    if (te_classify_enabled && !cascade_enabled) {
        std::cerr << "WARNING: --te-classify on requires --cascade on; enabling cascade.\n";
        cascade_enabled = true;
    }

    StopWatch total_sw;
    std::cerr << "singlet v0.3.0\n";
    if (cascade_enabled)
        std::cerr << "  cascade: " << cascade_mode << " | te-classify: " << te_classify_mode << "\n";

    // ── Track B: Load cascade indices when enabled (T-L2-9 observation mode) ──
    // In observation mode, reads are still written to STAR FIFOs, but we count
    // how many L1/L2 would resolve to populate cascade_stats.json accurately.
    std::unique_ptr<singlet::TxomeIndex> txome_idx;
    std::unique_ptr<singlet::TxomeAligner> txome_aligner;
    std::unique_ptr<singlet::TeFamilySketchDB> te_db;
    std::unique_ptr<singlet::TeClassifier> te_classifier;

    if (cascade_enabled && !txome_index_path.empty()) {
        StopWatch txome_sw;
        std::cerr << "[cascade] Loading transcriptome index: " << txome_index_path << "\n";
        txome_idx = std::make_unique<singlet::TxomeIndex>();
        // Read FASTA file
        std::ifstream txf(txome_index_path);
        if (!txf) {
            std::cerr << "ERROR: Cannot open --txome-index '" << txome_index_path << "'\n";
            return 1;
        }
        std::string fasta_text((std::istreambuf_iterator<char>(txf)),
                               std::istreambuf_iterator<char>());
        txf.close();
        txome_idx->build(fasta_text);
        txome_aligner = std::make_unique<singlet::TxomeAligner>(*txome_idx);
        std::cerr << "[cascade] Txome index: " << txome_idx->n_transcripts()
                  << " transcripts loaded (" << txome_sw.elapsed_s() << "s)\n";
    } else if (cascade_enabled && txome_index_path.empty()) {
        std::cerr << "[cascade] WARNING: --txome-index not provided; L1 resolve disabled\n";
    }

    if (te_classify_enabled && !te_db_path.empty()) {
        StopWatch te_sw;
        std::cerr << "[cascade] Loading TE sketch DB: " << te_db_path << "\n";
        try {
            te_db = std::make_unique<singlet::TeFamilySketchDB>(
                singlet::TeFamilySketchDB::load(te_db_path));
            te_classifier = std::make_unique<singlet::TeClassifier>(*te_db);
            std::cerr << "[cascade] TE DB: " << te_db->families.size()
                      << " families loaded (" << te_sw.elapsed_s() << "s)\n";
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to load --te-db '" << te_db_path << "': " << e.what() << "\n";
            return 1;
        }
    } else if (te_classify_enabled && te_db_path.empty()) {
        std::cerr << "[cascade] WARNING: --te-db not provided; L2 classify disabled\n";
    }

    // ── Phase 0: Load pileup references ──
    // Loads barcodes, SNPs, GTF into memory BEFORE starting STAR.
    // This overlaps reference loading with STAR's genome load when using
    // --genomeLoad LoadAndKeep (genome already in shared memory).
    std::cerr << "[singlet] Loading pileup references...\n";
    StopWatch ref_sw;

    // If barcode discovery is needed (--whitelist None or auto-resolved whitelist, no --barcodes),
    // defer engine creation until after .1fq decode + barcode extraction.
    bool needs_barcode_discovery = barcode_file.empty() &&
        (whitelist_file == "None" || whitelist_file == "none" || whitelist_auto_resolved);
    
    std::unique_ptr<singlet::PileupEngine> engine_ptr;
    if (!needs_barcode_discovery && !is_atac_mode && !is_smart_seq2_mode && !is_bulk_rna_mode && !is_cite_seq_mode) {
        engine_ptr = std::make_unique<singlet::PileupEngine>(config);
        if (!engine_ptr->load_references()) {
            std::cerr << "ERROR: Failed to load pileup references\n";
            return 1;
        }
        std::cerr << "[singlet] References loaded: " << ref_sw.elapsed_s() << "s\n";
    }

    // ── Phase 0.5: SRA streaming setup (FIFOs + native VDB reader) ──
    // If --sra is given, create two named FIFOs and fork a demuxer child
    // that reads the .sra file directly via NCBI VDB C API (zero-copy),
    // splits paired reads, and writes FASTQ text to R1/R2 FIFOs.
    // STAR reads from the FIFOs as if they were regular FASTQ files.
    // No external tools. No temporary FASTQ files ever touch disk.
    std::string fifo_dir;
    std::string r1_fifo_path, r2_fifo_path;
    pid_t demux_pid = -1;

    if (sra_mode) {
        std::cerr << "[singlet] SRA mode (native VDB): " << sra_file << "\n";

        // Create temp directory for FIFOs
        fifo_dir = out_prefix + "/.sra_fifos";
        mkdir_p(fifo_dir);
        r1_fifo_path = fifo_dir + "/R1.fifo";
        r2_fifo_path = fifo_dir + "/R2.fifo";

        // Remove stale FIFOs from previous runs
        unlink(r1_fifo_path.c_str());
        unlink(r2_fifo_path.c_str());

        if (mkfifo(r1_fifo_path.c_str(), 0600) != 0) {
            std::cerr << "ERROR: mkfifo R1 failed: " << strerror(errno) << "\n";
            return 1;
        }
        if (mkfifo(r2_fifo_path.c_str(), 0600) != 0) {
            std::cerr << "ERROR: mkfifo R2 failed: " << strerror(errno) << "\n";
            return 1;
        }

        // Readiness pipe: demuxer signals parent once FIFOs are open for writing
        int demux_ready[2];
        if (pipe(demux_ready) != 0) {
            std::cerr << "ERROR: demux ready pipe failed: " << strerror(errno) << "\n";
            return 1;
        }

        demux_pid = fork();
        if (demux_pid < 0) {
            std::cerr << "ERROR: fork() for SRA demuxer failed: " << strerror(errno) << "\n";
            return 1;
        }

        if (demux_pid == 0) {
            // ── SRA DEMUXER CHILD (native VDB reader) ──
            close(demux_ready[0]);

            // Open FIFOs with O_RDWR (never blocks — standard FIFO trick).
            int r1_fd = open(r1_fifo_path.c_str(), O_RDWR);
            int r2_fd = open(r2_fifo_path.c_str(), O_RDWR);
            if (r1_fd < 0 || r2_fd < 0) {
                std::cerr << "[sra-demux] ERROR: open FIFOs failed: "
                          << strerror(errno) << "\n";
                char fail = 0;
                (void)write(demux_ready[1], &fail, 1);
                _exit(1);
            }

            // Enlarge FIFO kernel buffers to 1 MB (Linux)
#ifdef __linux__
            fcntl(r1_fd, F_SETPIPE_SZ, 1 << 20);
            fcntl(r2_fd, F_SETPIPE_SZ, 1 << 20);
#endif

            // Signal parent: FIFOs are open, safe to fork STAR
            char ok = 1;
            (void)write(demux_ready[1], &ok, 1);
            close(demux_ready[1]);

            // Buffered FILE* wrappers over FIFO fds (1 MB buffers)
            FILE* r1_fp = fdopen(r1_fd, "w");
            FILE* r2_fp = fdopen(r2_fd, "w");
            if (!r1_fp || !r2_fp) {
                std::cerr << "[sra-demux] ERROR: fdopen FIFOs failed\n";
                _exit(1);
            }
            static char r1_wbuf[1 << 20];
            static char r2_wbuf[1 << 20];
            setvbuf(r1_fp, r1_wbuf, _IOFBF, sizeof(r1_wbuf));
            setvbuf(r2_fp, r2_wbuf, _IOFBF, sizeof(r2_wbuf));

            // Open SRA file via native VDB reader (zero-copy)
            int exit_code = 0;
            try {
                SraReader reader;
                reader.open(sra_file.c_str());
                std::cerr << "[sra-demux] Opened: " << reader.total_spots()
                          << " spots\n";

                while (reader.write_next_spot(r1_fp, r2_fp)) {
                    // Progress every 1M spots
                    if ((reader.spots_read() % 1000000) == 0) {
                        std::cerr << "[sra-demux] " << reader.spots_read()
                                  << " / " << reader.total_spots() << " spots\n";
                    }
                }

                std::cerr << "[sra-demux] Done: " << reader.spots_read()
                          << " spots written\n";
            } catch (const std::exception& e) {
                std::cerr << "[sra-demux] ERROR: " << e.what() << "\n";
                exit_code = 1;
            }

            fflush(r1_fp); fclose(r1_fp);
            fflush(r2_fp); fclose(r2_fp);
            _exit(exit_code);
        }

        // ── PARENT: wait for demuxer to confirm FIFOs are open ──
        close(demux_ready[1]);
        char ready_byte = 0;
        if (read(demux_ready[0], &ready_byte, 1) != 1 || ready_byte != 1) {
            std::cerr << "ERROR: SRA demuxer failed to initialize\n";
            return 1;
        }
        close(demux_ready[0]);

        // Point STAR's read inputs at the FIFOs
        reads_r1 = r1_fifo_path;
        reads_r2 = r2_fifo_path;

        std::cerr << "[singlet] SRA demuxer pid=" << demux_pid << "\n";
    }

    // Barcode frequency data captured during .1fq decode (avoids separate
    // R1.fastq scan for barcode discovery — saves ~10s).
    uint32_t onefq_bc_dict_n = 0;
    std::vector<uint64_t> onefq_bc_freq;
    std::vector<uint8_t> onefq_bc_dict_flat;
    uint16_t onefq_bc_length = 0;
    std::unordered_map<uint64_t, uint64_t> onefq_bc_counts;  // non-dict path (packed keys)

    int r2_clip5p = 0;  // bases to clip from R2 5' end (detected during R2 validation)

    // ── FIFO decode state ──
    // FIFO decode is the default for .1fq mode: eliminates 50-300 GB temp FASTQ files
    // by streaming decode → named FIFOs → STAR reads live. Set SINGLET_FILE_DECODE=1
    // to revert to the legacy file-based decode (writes full FASTQs to TMPDIR).
    // Pass 1: lightweight BC count + R2 prefix capture (no FASTQ files written)
    // Pass 2: full decode streamed to named FIFOs → STAR reads live from them
    const bool use_fifo_decode = onefq_mode &&
        !(getenv("SINGLET_FILE_DECODE") &&
          std::string(getenv("SINGLET_FILE_DECODE")) == "1");
    std::thread fifo_writer_thread;    // Pass 2 writer (started after Phase 0.9)
    std::string fifo_r1_path;          // original R1 stream FIFO path (before reads_inverted swap)
    std::string fifo_r2_path;          // original R2 stream FIFO path
    std::vector<std::string> fifo_first25_r2;  // first 25 R2 seqs captured in Pass 1

    // Hoisted from inner decode block so per-segment whitelist discovery
    // (Phase 0.9+) can read them after the onefq_mode block closes.
    struct ComplexExtraSeg { int r1_start; int len; }; // -1 start = skip
    std::vector<ComplexExtraSeg> complex_extra_segs;
    std::vector<std::unordered_map<uint64_t, uint64_t>> onefq_per_seg_counts;
    // Protocol id cached from .1fq header for use after onefq_mode block.
    uint8_t onefq_protocol_id = 255;

    // ── CB_UMI_Complex → CB_samTagOut pre-concatenation state ──
    // When enabled, the decode step rewrites R1 as [seg0][seg1]...[segN][UMI],
    // generating a fixed-width simple barcode for STAR CB_samTagOut.  This
    // avoids CB_UMI_Complex's 200-400 GB RAM requirement.
    struct ComplexConcatSeg { int start; int end; int len; }; // 0-based in original R1
    std::vector<ComplexConcatSeg> complex_concat_segs_all;   // all CB segments
    int  concat_umi_start_0  = -1;  // 0-based UMI start in original R1
    int  concat_umi_len_c    = 0;   // UMI length
    int  concat_bc_total     = 0;   // sum of all segment lengths → new soloCBlen
    bool use_complex_concat  = false;
    std::string combined_wl_file;   // path to Cartesian product whitelist

    // ── Phase 0.5b: .1fq streaming setup (FIFOs + lib1fq reader) ──
    // If --1fq is given, create two named FIFOs and fork a demuxer child
    // that reads the .1fq file via lib1fq::Reader, decodes blocks to
    // byte-numeric, converts to FASTQ text, and writes to R1/R2 FIFOs.
    // STAR reads from the FIFOs exactly like the SRA path.
    if (onefq_mode) {
        std::cerr << "[singlet] .1fq mode: " << onefq_file << "\n";

        // Read .1fq header + metadata to auto-configure STAR params
        {
            lib1fq::Reader probe;
            try {
                probe.open(onefq_file.c_str());
            } catch (const std::exception& e) {
                std::cerr << "[singlet] ERROR: Cannot open .1fq file '" << onefq_file
                          << "': " << e.what() << "\n";
                return 1;
            }
            auto hdr = probe.header();
            uint16_t r1_len = hdr.stream_lengths[0]; // R1 = barcode + UMI

            detected_umi_len = 0;
            int detected_r2_len = 0;  // from metadata r2_length

            // Try metadata JSON first (authoritative source from encoder)
            if (hdr.meta_size > 0) {
                try {
                    std::string meta = probe.read_metadata();
                    // Extract soloCBlen and soloUMIlen from metadata JSON
                    auto extract_int = [&](const std::string& key) -> int {
                        auto pos = meta.find("\"" + key + "\"");
                        if (pos == std::string::npos) return -1;
                        pos = meta.find(':', pos);
                        if (pos == std::string::npos) return -1;
                        ++pos;
                        while (pos < meta.size() && meta[pos] == ' ') ++pos;
                        return std::atoi(meta.c_str() + pos);
                    };
                    auto extract_str = [&](const std::string& key) -> std::string {
                        auto pos = meta.find("\"" + key + "\"");
                        if (pos == std::string::npos) return "";
                        pos = meta.find(':', pos);
                        if (pos == std::string::npos) return "";
                        pos = meta.find('"', pos + 1);
                        if (pos == std::string::npos) return "";
                        auto end = meta.find('"', pos + 1);
                        if (end == std::string::npos) return "";
                        return meta.substr(pos + 1, end - pos - 1);
                    };
                    int cb = extract_int("soloCBlen");
                    int ui = extract_int("soloUMIlen");
                    if (cb > 0 && ui > 0) {
                        detected_bc_len = cb;
                        detected_umi_len = ui;
                        std::cerr << "[singlet] .1fq metadata: soloCBlen="
                                  << cb << " soloUMIlen=" << ui << "\n";
                    }
                    int r2l = extract_int("r2_length");
                    if (r2l > 0) detected_r2_len = r2l;
                    // Extract complex barcode parameters
                    std::string solo_type = extract_str("soloType");
                    if (!solo_type.empty()) {
                        detected_soloType = solo_type;
                        std::cerr << "[singlet] .1fq metadata: soloType="
                                  << solo_type << "\n";
                    }
                    std::string umi_pos = extract_str("soloUMIposition");
                    if (!umi_pos.empty()) {
                        detected_umi_position = umi_pos;
                    }
                    // Extract soloCBposition array
                    auto cbp_pos = meta.find("\"soloCBposition\"");
                    if (cbp_pos != std::string::npos) {
                        auto arr_start = meta.find('[', cbp_pos);
                        auto arr_end = meta.find(']', arr_start);
                        if (arr_start != std::string::npos && arr_end != std::string::npos) {
                            std::string arr = meta.substr(arr_start + 1, arr_end - arr_start - 1);
                            size_t p = 0;
                            while ((p = arr.find('"', p)) != std::string::npos) {
                                auto e = arr.find('"', p + 1);
                                if (e == std::string::npos) break;
                                detected_cb_positions.push_back(arr.substr(p + 1, e - p - 1));
                                p = e + 1;
                            }
                        }
                    }
                    // Extract soloCBstart/soloUMIstart for simple types
                    int cb_start_val = extract_int("soloCBstart");
                    if (cb_start_val > 0) detected_cb_start = cb_start_val;
                    int umi_start_val = extract_int("soloUMIstart");
                    if (umi_start_val > 0) detected_umi_start = umi_start_val;

                    // Fix: protocol.tag is nested inside "protocol": {...}, extract "tag" directly
                    std::string proto_tag_meta = extract_str("tag");
                    if (!proto_tag_meta.empty())
                        std::cerr << "[singlet] .1fq metadata: protocol=" << proto_tag_meta << "\n";
                    // S5-UMI-LEN: fallback tag-string → UMI length when metadata
                    // lacked soloUMIlen and user did not explicitly set --umi-len
                    if (detected_umi_len == 0 && !umi_len_explicit && !proto_tag_meta.empty()) {
                        std::string tag_lc = proto_tag_meta;
                        for (char& c : tag_lc) c = static_cast<char>(std::tolower(c));
                        // 10x family: v2/v1→10bp, v3/v4/default→12bp
                        if (tag_lc.find("10x") != std::string::npos ||
                            tag_lc.find("chromium") != std::string::npos) {
                            if (tag_lc.find("v2") != std::string::npos ||
                                tag_lc.find("v1") != std::string::npos)
                                detected_umi_len = 10;
                            else
                                detected_umi_len = 12;
                        } else if (tag_lc.find("drop")    != std::string::npos ||
                                   tag_lc.find("ddseq")   != std::string::npos ||
                                   tag_lc.find("sci")     != std::string::npos ||
                                   tag_lc.find("smart-seq3") != std::string::npos ||
                                   tag_lc.find("seqwell") != std::string::npos ||
                                   tag_lc.find("quartz")  != std::string::npos ||
                                   tag_lc.find("mars")    != std::string::npos) {
                            detected_umi_len = 8;
                        } else if (tag_lc.find("indrop")  != std::string::npos ||
                                   tag_lc.find("celseq")  != std::string::npos ||
                                   tag_lc.find("cel-seq") != std::string::npos ||
                                   tag_lc.find("strt")    != std::string::npos ||
                                   tag_lc.find("microwell") != std::string::npos) {
                            detected_umi_len = 6;
                        } else if (tag_lc.find("share")   != std::string::npos ||
                                   tag_lc.find("dnbe")    != std::string::npos ||
                                   tag_lc.find("parse")   != std::string::npos ||
                                   tag_lc.find("split")   != std::string::npos) {
                            detected_umi_len = 12;
                        }
                        if (detected_umi_len > 0)
                            std::cerr << "[singlet] S5-UMI-LEN: tag='" << proto_tag_meta
                                      << "' → umi_len=" << detected_umi_len << "\n";
                    }
                    // Read clip5p_length: constant 5' prefix detected during .1fq encoding
                    // (1000-read, 80% threshold, ≥20bp detector). Apply as r2_clip5p so
                    // STAR gets --clip5pNbases N 0 (clip cDNA only, not barcode read).
                    int clip5p_from_meta = extract_int("clip5p_length");
                    if (clip5p_from_meta > 0) {
                        r2_clip5p = clip5p_from_meta;
                        std::cerr << "[singlet] Auto-clip5p: " << r2_clip5p
                                  << " bp from cDNA read (.1fq metadata)\n";
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[singlet] .1fq metadata read error: "
                              << e.what() << " (falling back to header)\n";
                }
            }

            // ── METADATA-JSON PROTOCOL OVERRIDE: patch hdr.protocol_id before N2/N10/CellRanger4 ──
            // Only fires when the metadata protocol DIFFERS from the .1fq header protocol.
            // When they match, the header is already correct and no action is needed.
            if (meta_json_proto_id != 255 && meta_json_proto_id != hdr.protocol_id) {
                // Determine old tag for log message
                std::string old_tag;
                for (const auto& sp : lib1fq::known_protocols())
                    if (sp.protocol_id == hdr.protocol_id) { old_tag = sp.tag; break; }
                if (old_tag.empty()) old_tag = std::to_string((int)hdr.protocol_id);
                std::cerr << "[singlet] Protocol override: .1fq header protocol_id="
                          << old_tag << " overridden by --metadata-json protocol="
                          << meta_json_proto_tag << "\n";
                hdr.protocol_id = meta_json_proto_id;
                // Clear any CB/UMI lengths decoded from the old .1fq metadata
                // so that N2 re-derives them from the corrected protocol_id.
                if (detected_bc_len > 0 || detected_umi_len > 0) {
                    detected_bc_len  = 0;
                    detected_umi_len = 0;
                    detected_soloType.clear();
                    detected_cb_positions.clear();
                    detected_umi_position.clear();
                    std::cerr << "[singlet] Protocol override: cleared stale CB/UMI params"
                                 " derived from old .1fq header protocol\n";
                }
            }

            // ── N2: Protocol_id → known_protocols() lookup ──
            // Bridge binary protocol_id → STAR CB/UMI config for files without
            // or with incomplete metadata JSON. Runs when metadata JSON didn't
            // provide CB/UMI lengths and protocol isn't already CB_UMI_Complex.
            if (detected_bc_len == 0 && detected_umi_len == 0
                && hdr.protocol_id != 255
                && detected_soloType != "CB_UMI_Complex") {
                for (const auto& spec : lib1fq::known_protocols()) {
                    if (spec.protocol_id == hdr.protocol_id
                        && spec.bc_len > 0 && spec.umi_len > 0) {
                        detected_bc_len  = spec.bc_len;
                        detected_umi_len = spec.umi_len;
                        // CandidateSpec offsets are 0-based; STAR uses 1-based
                        detected_cb_start  = static_cast<int>(spec.bc_offset)  + 1;
                        detected_umi_start = static_cast<int>(spec.umi_offset) + 1;
                        if (detected_soloType.empty())
                            detected_soloType = "CB_UMI_Simple";
                        // N10: capture protocol-specific 3' adapter
                        if (!spec.adapter3p.empty())
                            detected_adapter3p = spec.adapter3p;
                        std::cerr << "[singlet] N2: protocol_id=" << (int)hdr.protocol_id
                                  << " (" << spec.tag << ")"
                                  << " → CB=" << spec.bc_len << "bp@" << detected_cb_start
                                  << " UMI=" << spec.umi_len << "bp@" << detected_umi_start
                                  << "\n";
                        break;
                    }
                }
            }

            // N10: Always look up adapter3p by protocol_id (runs even when metadata
            // provided CB/UMI so N2 was skipped).
            if (detected_adapter3p.empty() && hdr.protocol_id != 255) {
                for (const auto& spec : lib1fq::known_protocols()) {
                    if (spec.protocol_id == hdr.protocol_id && !spec.adapter3p.empty()) {
                        detected_adapter3p = spec.adapter3p;
                        std::cerr << "[singlet] N10: adapter3p auto-selected for "
                                  << spec.tag << " ("
                                  << detected_adapter3p.substr(0, 12) << "...)\n";
                        break;
                    }
                }
            }

            // Fallback: heuristic from R1 length if metadata didn't provide values
            // But first: detect inverted reads (barcode in R2, cDNA in R1)
            uint16_t r2_len_hdr = hdr.stream_lengths[1];
            // For variable-length R2, use metadata r2_length
            int r2_for_invert = (r2_len_hdr > 0) ? r2_len_hdr : detected_r2_len;
            // ATAC guard: for ATAC assays, stream[1] can legitimately be a short barcode
            // (16bp) from a 2-segment VDB deposit (R1 genomic + barcode, no R3).
            // reads_inverted must NOT fire here — swapping would send the barcode as
            // STAR's first read and the genomic R1 as its second, producing near-zero
            // alignments and 0 fragments.  ATAC barcodes belong in QNAME (I2 stream
            // for encode_atac, or N-padded fallback); they are never a signal to swap.
            if (!is_atac_mode &&
                r1_len > 40 && r2_for_invert > 0 && r2_for_invert <= 34 &&
                r1_len > r2_for_invert * 2) {
                reads_inverted = true;
                // Override BC/UMI params from R2 (the actual barcode read)
                if (r2_for_invert == 14)      { detected_bc_len = 8;  detected_umi_len = 6;  }
                else if (r2_for_invert == 20) { detected_bc_len = 12; detected_umi_len = 8;  }
                else if (r2_for_invert == 26) { detected_bc_len = 16; detected_umi_len = 10; }
                else if (r2_for_invert == 28) { detected_bc_len = 16; detected_umi_len = 12; }
                else { detected_bc_len = r2_for_invert * 2 / 3;
                       detected_umi_len = r2_for_invert - detected_bc_len; }
                detected_cb_start = 1;
                detected_umi_start = detected_bc_len + 1;
                std::cerr << "[singlet] Inverted reads detected: R1=" << r1_len
                          << "bp (cDNA), R2=" << r2_for_invert << "bp (barcode)\n"
                          << "[singlet] Override: BC=" << detected_bc_len
                          << " UMI=" << detected_umi_len << "\n";
            } else if (detected_umi_len == 0 && r1_len > 0) {
                if (r1_len == 28)      { detected_bc_len = 16; detected_umi_len = 12; }
                else if (r1_len == 26) { detected_bc_len = 16; detected_umi_len = 10; }
                else if (r1_len == 24) { detected_bc_len = 14; detected_umi_len = 10; }
                else if (r1_len == 20) { detected_bc_len = 12; detected_umi_len = 8;  }
                else if (r1_len == 12) { detected_bc_len = 6;  detected_umi_len = 6;  }
                else if (r1_len == 15) { detected_bc_len = 7;  detected_umi_len = 8;  }
                if (detected_umi_len > 0) {
                    std::cerr << "[singlet] .1fq header heuristic: R1=" << r1_len
                              << "bp → BC=" << detected_bc_len
                              << " UMI=" << detected_umi_len << "\n";
                }
            }

            if (detected_umi_len > 0 && !umi_len_explicit) {
                umi_len = detected_umi_len;
            } else if (r1_len > 0 && detected_umi_len == 0) {
                std::cerr << "[singlet] .1fq header: R1=" << r1_len
                          << "bp (unknown geometry, using --umi-len=" << umi_len << ")\n";
            }

            // ── Guard 1: Barcode/UMI length sanity check ──
            // No real scRNA-seq protocol has CBlen > 32 or UMIlen > 16.
            // Values this large indicate bad metadata from a failed protocol detection.
            if (detected_bc_len > 32 || detected_umi_len > 16) {
                std::cerr << "[singlet] ERROR: Invalid barcode configuration"
                          << " (CBlen=" << detected_bc_len
                          << ", UMIlen=" << detected_umi_len << ").\n"
                          << "[singlet] This suggests the protocol was not correctly "
                             "detected during download.\n"
                          << "[singlet] Try re-downloading with --protocol <protocol_name> "
                             "to specify the protocol explicitly.\n";
                return 1;
            }

            std::cerr << "[singlet] .1fq: " << hdr.n_unique << " reads, "
                      << hdr.block_count << " blocks, protocol_id="
                      << (int)hdr.protocol_id
                      << ", confidence=" << (int)hdr.confidence
                      << ", codec=" << (int)hdr.codec << "\n";

            // ── B-G2-5: AUTOFIX-SPECIES-VAL-R2-SHORT ──
            // Reject LOW-confidence auto-detections when R2 is short (< 50 bp).
            // Short R2 + LOW confidence indicates adapter-trimmed R2 (e.g. ENA fallback
            // auto-trim) that caused the protocol detector to misassign (e.g. → 10x-ATAC).
            // Map to terminal autodetect_species_fail per DROPLET_PRODUCTION_CONTRACT §5.
            // Skip ATAC mode (short R2 barcodes are expected there).
            if (!is_atac_mode &&
                hdr.confidence <= static_cast<uint8_t>(lib1fq::Confidence::LOW) &&
                detected_r2_len > 0 && detected_r2_len < 50) {
                std::cerr << "[singlet] AUTOFIX-SPECIES-VAL-R2-SHORT:"
                          << " confidence=LOW, R2=" << detected_r2_len
                          << "bp < 50bp threshold — protocol_id=" << (int)hdr.protocol_id
                          << " may be misdetected due to adapter-trimmed R2.\n"
                          << "[singlet] status=autodetect_species_fail\n"
                          << "[singlet] Re-download with explicit --protocol or check ENA"
                             " adapter trimming settings.\n";
                return 3;  // autodetect_species_fail terminal exit
            }

            // Auto-detect 5' protocol → reverse strand filter
            // IDs: 4=10xv3-5p, 5=10xv2-5p, 14=10xv4-5p, 21=strtseq
            if (!config.reverse_strand) {
                uint8_t pid = hdr.protocol_id;
                if (pid == 4 || pid == 5 || pid == 14 || pid == 21) {
                    config.reverse_strand = true;
                    std::cerr << "[singlet] 5' protocol detected → using reverse strand\n";
                }
            }
            onefq_r1_len = r1_len;
            onefq_n_reads = hdr.n_unique;
            // Cache protocol_id for use after this block (e.g., per-segment whitelist discovery).
            onefq_protocol_id = hdr.protocol_id;

            // ── Absolute minimum read count — abort before STAR to avoid fast-crash waste ──
            if (onefq_n_reads < singlet_pileup::TinyDatasetGuard::MIN_READS_FOR_PIPELINE) {
                std::fprintf(stderr,
                    "[singlet] WARNING: Only %llu reads in .1fq file"
                    " (minimum %zu for meaningful analysis). Aborting.\n",
                    (unsigned long long)onefq_n_reads,
                    static_cast<size_t>(singlet_pileup::TinyDatasetGuard::MIN_READS_FOR_PIPELINE));
                return 2;
            }

            // G-TINY: warn on very small datasets before alignment
            {
                using singlet_pileup::TinyDatasetGuard;
                std::string tiny_warn = TinyDatasetGuard::warning_message(onefq_n_reads);
                if (!tiny_warn.empty())
                    std::cerr << "[singlet] WARNING: " << tiny_warn << "\n";
            }
        }

        // ── Decode .1fq to temp FASTQ files ──
        // Write uncompressed FASTQ to TMPDIR (typically /dev/shm for speed).
        // STAR reads from regular files — avoids the named-pipe deadlock
        // that occurs with dual-FIFO streaming.

        // Pre-compute barcode geometry for in-decode counting
        // (same logic as Phase 0.9 barcode discovery)
        int discover_bc_start_0 = (detected_cb_start > 0 ? detected_cb_start - 1 : 0);
        int discover_bc_len = (detected_bc_len > 0 ? detected_bc_len : 12);
        if (detected_soloType == "CB_UMI_Complex" && !detected_cb_positions.empty()) {
            auto parse_pos_pre = [](const std::string& p) -> std::pair<int,int> {
                int vals[4] = {0,0,0,0};
                size_t pos = 0; int vi = 0;
                for (size_t i = 0; i <= p.size() && vi < 4; ++i) {
                    if (i == p.size() || p[i] == '_') {
                        vals[vi++] = std::atoi(p.substr(pos, i - pos).c_str());
                        pos = i + 1;
                    }
                }
                return {vals[1], vals[3]};
            };
            auto [cb_s, cb_e] = parse_pos_pre(detected_cb_positions[0]);
            discover_bc_start_0 = cb_s;
            discover_bc_len = cb_e - cb_s + 1;
        }
        const bool count_barcodes_in_decode =
            barcode_file.empty() &&
            (whitelist_file == "None" || whitelist_file == "none" || whitelist_auto_resolved) &&
            discover_bc_len > 0;

        // ── CB_UMI_Complex: per-segment position cache for multi-segment whitelist discovery ──
        // For protocols like BD Rhapsody (3 × 9bp CBs) and SPLiT-seq (3 × 8bp CBs), discover
        // barcodes at each additional CB segment position during the decode pass so CB_UMI_Complex
        // can be used with per-segment whitelists instead of falling back to CB_samTagOut.
        // Only handles anchorType==0 (absolute from read start, format "0_start_0_end").
        // All supported CB_UMI_Complex protocols use anchorType==0; anchorType==1 is not used.
        // NOTE: complex_extra_segs and onefq_per_seg_counts are declared above (outer scope)
        // so they remain accessible after this if(onefq_mode) block closes.
        if (count_barcodes_in_decode && detected_soloType == "CB_UMI_Complex" &&
            detected_cb_positions.size() > 1) {
            for (size_t seg = 1; seg < detected_cb_positions.size(); ++seg) {
                const auto& p = detected_cb_positions[seg];
                // Parse "atype0_dist0_atype1_dist1"
                int vals[4] = {}; size_t px = 0; int vi = 0;
                for (size_t ci = 0; ci <= p.size() && vi < 4; ++ci) {
                    if (ci == p.size() || p[ci] == '_') {
                        vals[vi++] = std::atoi(p.c_str() + px);
                        px = ci + 1;
                    }
                }
                int atype0 = vals[0]; // 0=abs-from-start; 1=read-end; 2=adapter
                int dist0  = vals[1]; // start offset
                int dist1  = vals[3]; // end offset
                int seg_len = dist1 - dist0 + 1;
                if (atype0 == 0 && vals[2] == 0 && dist0 >= 0 && seg_len > 0 && seg_len <= 21)
                    complex_extra_segs.push_back({dist0, seg_len});
                else
                    complex_extra_segs.push_back({-1, 0}); // skip
                onefq_per_seg_counts.push_back({});
            }
        }

        // ── CB_UMI_Complex → CB_samTagOut concat mode: pre-parse all segment positions ──
        // When all CB segments have anchorType 0 (absolute from read start) and the UMI
        // position is known, rewrite R1 as [seg0][seg1]...[segN][UMI] during decode.
        // This replaces CB_UMI_Complex entirely, cutting STAR RAM from 200-400 GB to ~5 GB.
        if (detected_soloType == "CB_UMI_Complex" && detected_cb_positions.size() > 1
                && !detected_umi_position.empty()) {
            auto parse_pos_cc = [](const std::string& p) -> std::array<int,4> {
                std::array<int,4> v{}; size_t px = 0; int vi = 0;
                for (size_t ci = 0; ci <= p.size() && vi < 4; ++ci)
                    if (ci == p.size() || p[ci] == '_')
                        { v[vi++] = std::atoi(p.c_str() + px); px = ci + 1; }
                return v;
            };
            bool cc_ok = true;
            for (const auto& cp : detected_cb_positions) {
                auto v = parse_pos_cc(cp);
                if (v[0] != 0 || v[2] != 0 || v[1] < 0 || v[3] < v[1]) { cc_ok = false; break; }
                complex_concat_segs_all.push_back({v[1], v[3], v[3] - v[1] + 1});
            }
            if (cc_ok) {
                auto vu = parse_pos_cc(detected_umi_position);
                if (vu[0] == 0 && vu[2] == 0 && vu[1] >= 0 && vu[3] >= vu[1]) {
                    concat_umi_start_0 = vu[1];
                    concat_umi_len_c   = vu[3] - vu[1] + 1;
                } else cc_ok = false;
            }
            if (cc_ok && !complex_concat_segs_all.empty() && concat_umi_len_c > 0) {
                for (const auto& seg : complex_concat_segs_all) concat_bc_total += seg.len;
                use_complex_concat = true;
                std::cerr << "[singlet] CB_UMI_Complex\xE2\x86\x92concat: "
                          << complex_concat_segs_all.size() << " segs \xE2\x86\x92 BClen="
                          << concat_bc_total << " UMIlen=" << concat_umi_len_c << "\n";
            } else {
                complex_concat_segs_all.clear();
                concat_umi_start_0 = -1; concat_umi_len_c = 0; concat_bc_total = 0;
            }
        }

        std::string tmpdir = "/dev/shm";
        if (const char* td = std::getenv("TMPDIR"))
            tmpdir = td;
        std::string decode_dir = tmpdir + "/singlet_1fq_" + std::to_string(getpid());
        mkdir_p(decode_dir);
        std::string r1_tmp = decode_dir + "/R1.fastq";
        std::string r2_tmp = decode_dir + "/R2.fastq";

        if (!use_fifo_decode) {
            // ── File mode: decode all blocks to FASTQ files on /dev/shm ──
            StopWatch decode_sw;
            FILE* r1_fp = fopen(r1_tmp.c_str(), "w");
            FILE* r2_fp = fopen(r2_tmp.c_str(), "w");
            if (!r1_fp || !r2_fp) {
                std::cerr << "ERROR: cannot create temp FASTQ files in "
                          << decode_dir << ": " << strerror(errno) << "\n";
                return 1;
            }
            static char r1_wbuf[1 << 20];
            static char r2_wbuf[1 << 20];
            setvbuf(r1_fp, r1_wbuf, _IOFBF, sizeof(r1_wbuf));
            setvbuf(r2_fp, r2_wbuf, _IOFBF, sizeof(r2_wbuf));

            lib1fq::Reader reader;
            try {
                reader.open(onefq_file.c_str());
            } catch (const std::exception& e) {
                std::cerr << "[singlet] ERROR: Cannot read .1fq file '" << onefq_file
                          << "': " << e.what() << "\n";
                _exit(1);
            }
            reader.set_verify_crc(false);  // speed: skip CRC for pipeline decode
            auto hdr = reader.header();
            std::cerr << "[1fq-decode] Decoding " << hdr.n_unique
                      << " reads to " << decode_dir << "\n";

            // BC frequency accumulator — populated during decode to avoid
            // a separate R1.fastq scanning pass for barcode discovery.
            const uint32_t bc_dict_n = reader.bc_dict_size();
            const bool has_bc_dict = reader.has_bc_dict();

            const uint32_t total_blocks = reader.block_count();
            const int fd = reader.file_descriptor();
            const auto codec = static_cast<lib1fq::Codec>(hdr.codec);
            const int decode_threads = std::min(static_cast<int>(star_threads), static_cast<int>(total_blocks));

            static constexpr char NUM_TO_ASCII[5] = {'A','C','G','T','N'};

            // Formatted block output
            struct FmtBlock {
                std::vector<char> r1, r2;
                size_t r1_len = 0, r2_len = 0;
                uint32_t n_reads = 0;
                std::vector<uint32_t> bc_indices;  // for barcode frequency counting (BC dict path)
                std::unordered_map<uint64_t, uint64_t> bc_counts_packed; // non-dict path (packed 3-bit keys)
                // Extra segments for CB_UMI_Complex per-segment whitelist discovery (seg 1..N-1)
                std::vector<std::unordered_map<uint64_t, uint64_t>> extra_seg_counts;
            };

            // ATAC: pre-load BC dict for QNAME injection (stream[0]=R1 genomic,
            // stream[1]=R3 genomic; barcode is in per-read BC dict slot or I2 stream).
            std::vector<uint8_t> atac_bc_dict_bytes;
            uint16_t atac_bc_len = 0;
            bool has_i2_stream = is_atac_mode &&
                                 (reader.header().reserved[0] & lib1fq::ExtraStreams::HAS_I2) != 0;
            if (has_i2_stream) {
                atac_bc_len = reader.header().stream_lengths[2];
                std::cerr << "[1fq-decode] ATAC: I2 barcode stream " << atac_bc_len << "bp\n";
                // Feed atac_bc_len into detected_bc_len so Phase 0.9 barcode discovery
                // unpacks packed 3-bit keys with the correct I2 length (not the default 12bp).
                if (atac_bc_len > 0 && detected_bc_len == 0)
                    detected_bc_len = static_cast<int>(atac_bc_len);
            } else if (is_atac_mode && has_bc_dict && bc_dict_n > 0) {
                atac_bc_dict_bytes = reader.bc_dict_flat();
                atac_bc_len        = reader.bc_length();
                std::cerr << "[1fq-decode] ATAC: BC dict " << bc_dict_n
                          << " entries \xC3\x97 " << atac_bc_len << "bp\n";
            }

            // Worker: pread + decompress + decode + format one block
            auto process_block = [&](uint32_t bi, uint64_t read_off) -> FmtBlock {
                FmtBlock fb;
                uint64_t file_off = reader.block_offset(bi);
                uint32_t comp_size = reader.block_comp_size(bi);

                lib1fq::BlockHeader bh;
                ::pread(fd, &bh, sizeof(bh), static_cast<off_t>(file_off));

                std::vector<uint8_t> comp_buf(comp_size);
                ::pread(fd, comp_buf.data(), comp_size,
                        static_cast<off_t>(file_off + sizeof(bh)));

                size_t est_raw = static_cast<size_t>(bh.n_reads) * 200;
                std::vector<uint8_t> raw_buf(est_raw);
                size_t raw_size;
                try {
                    raw_size = lib1fq::compress::decompress_block(
                        codec, comp_buf.data(), comp_size,
                        raw_buf.data(), raw_buf.size());
                } catch (...) {
                    raw_buf.resize(est_raw * 4);
                    raw_size = lib1fq::compress::decompress_block(
                        codec, comp_buf.data(), comp_size,
                        raw_buf.data(), raw_buf.size());
                }

                lib1fq::DecodedBlock blk;
                reader.decode_payload(raw_buf.data(), raw_size, bh, blk);

                fb.n_reads = blk.n_reads;
                if (has_bc_dict)
                    fb.bc_indices = std::move(blk.bc_indices);
                uint16_t r1_max = blk.r1_lengths.empty() ? 28 : *std::max_element(blk.r1_lengths.begin(), blk.r1_lengths.end());
                uint16_t r2_max = blk.r2_lengths.empty() ? 91 : *std::max_element(blk.r2_lengths.begin(), blk.r2_lengths.end());
                // Account for dup expansion
                uint64_t total_reps = 0;
                const bool has_dups = !blk.dup_counts.empty();
                for (uint32_t i = 0; i < blk.n_reads; ++i)
                    total_reps += has_dups ? blk.dup_counts[i] : 1;
                fb.r1.resize(total_reps * (static_cast<size_t>(r1_max) * 2 + (is_atac_mode ? 55 : 30)));
                fb.r2.resize(total_reps * (static_cast<size_t>(r2_max) * 2 + (is_atac_mode ? 55 : 30)));
                char* r1p = fb.r1.data();
                char* r2p = fb.r2.data();
                bool has_qual = !blk.r2_qual.empty();
                uint64_t roff = read_off;

                for (uint32_t i = 0; i < blk.n_reads; ++i) {
                    const uint32_t repeat = has_dups ? blk.dup_counts[i] : 1;
                    uint16_t r1len = blk.r1_len(i);
                    uint16_t r2len = blk.r2_len(i);

                    // Skip reads with zero-length R1 or R2 — STAR's readLoad
                    // calls exitWithError when Lread < 1, producing either
                    // "short read sequence line" or "read files are not consistent"
                    // depending on thread timing.  Zero-length R2 is non-alignable.
                    // Skip BOTH R1 and R2 for this read to keep files in sync.
                    if (r1len == 0 || r2len == 0) continue;

                    const uint8_t* r1s = blk.r1_seq(i);
                    const uint8_t* r2s = blk.r2_seq(i);
                    const uint8_t* q = has_qual ? blk.r2_quality(i) : nullptr;

                    // Count barcodes per-block for fast barcode discovery
                    // Pack base values (0-4, 3 bits each) into uint64_t key
                    // to avoid string heap allocation (bc_len ≤ 21 fits in 63 bits)
                    if (has_i2_stream) {
                        // ATAC 3-read: barcodes are in the I2 stream, NOT R1 genomic reads.
                        // Count from I2 so atac_bc_map matches the QNAME-embedded barcodes.
                        if (!blk.i2_data.empty() &&
                            i < blk.i2_lengths.size() && blk.i2_lengths[i] > 0) {
                            const uint8_t* bcp = blk.i2_seq(i);
                            uint16_t blen = blk.i2_len(i);
                            if (blen > 0 && blen <= 21) {
                                uint64_t key = 0;
                                for (uint16_t j = 0; j < blen; ++j)
                                    key = (key << 3) | bcp[j];
                                fb.bc_counts_packed[key] += repeat;
                            }
                        }
                    } else if (count_barcodes_in_decode && !has_bc_dict &&
                               r1len >= discover_bc_start_0 + discover_bc_len) {
                        uint64_t key = 0;
                        const uint8_t* bcs = r1s + discover_bc_start_0;
                        for (int j = 0; j < discover_bc_len; ++j)
                            key = (key << 3) | bcs[j];
                        fb.bc_counts_packed[key] += repeat;
                        // Extra segments for CB_UMI_Complex (segs 1..N-1): same packed key
                        if (!complex_extra_segs.empty()) {
                            if (fb.extra_seg_counts.empty())
                                fb.extra_seg_counts.resize(complex_extra_segs.size());
                            for (size_t ex = 0; ex < complex_extra_segs.size(); ++ex) {
                                const auto& es = complex_extra_segs[ex];
                                if (es.r1_start < 0 || r1len < es.r1_start + es.len) continue;
                                uint64_t ekey = 0;
                                const uint8_t* ebcs = r1s + es.r1_start;
                                for (int j = 0; j < es.len; ++j) ekey = (ekey << 3) | ebcs[j];
                                fb.extra_seg_counts[ex][ekey] += repeat;
                            }
                        }
                    }

                    for (uint32_t rep = 0; rep < repeat; ++rep) {
                        // Build ATAC QNAME prefix "BC:BARCODE_" when in ATAC mode.
                        // Barcode source priority: I2 stream > BC dict > N-padding.
                        char atac_qname_prefix[32] = {}; // "BC:XXXXXXXXXXXXXXXX_" max 23 chars
                        if (is_atac_mode) {
                            bool bc_set = false;
                            // 1. I2 stream (3-read FASTQ encoded)
                            if (has_i2_stream && !blk.i2_data.empty() &&
                                i < blk.i2_lengths.size() && blk.i2_lengths[i] > 0) {
                                const uint8_t* bcp = blk.i2_seq(i);
                                uint16_t blen = blk.i2_len(i);
                                int np = 0;
                                atac_qname_prefix[np++] = 'B';
                                atac_qname_prefix[np++] = 'C';
                                atac_qname_prefix[np++] = ':';
                                for (uint16_t j = 0; j < blen; ++j)
                                    atac_qname_prefix[np++] = NUM_TO_ASCII[bcp[j]];
                                atac_qname_prefix[np++] = '_';
                                atac_qname_prefix[np]   = '\0';
                                bc_set = true;
                            }
                            // 2. BC dict slot
                            if (!bc_set) {
                                uint32_t bc_idx = (!fb.bc_indices.empty() && i < fb.bc_indices.size())
                                                  ? fb.bc_indices[i] : UINT32_MAX;
                                if (!atac_bc_dict_bytes.empty() && atac_bc_len > 0 &&
                                    bc_idx < bc_dict_n) {
                                    const uint8_t* bcp = atac_bc_dict_bytes.data() +
                                                         static_cast<size_t>(bc_idx) * atac_bc_len;
                                    int np = 0;
                                    atac_qname_prefix[np++] = 'B';
                                    atac_qname_prefix[np++] = 'C';
                                    atac_qname_prefix[np++] = ':';
                                    for (uint16_t j = 0; j < atac_bc_len; ++j)
                                        atac_qname_prefix[np++] = NUM_TO_ASCII[bcp[j]];
                                    atac_qname_prefix[np++] = '_';
                                    atac_qname_prefix[np]   = '\0';
                                    bc_set = true;
                                }
                            }
                            // 3. N-pad fallback
                            if (!bc_set) {
                                int np = sprintf(atac_qname_prefix, "BC:NNNNNNNNNNNNNNNN_");
                                (void)np;
                            }
                        }

                        // R1 record
                        int n = is_atac_mode
                            ? sprintf(r1p, "@%sr%lu\n", atac_qname_prefix, (unsigned long)roff)
                            : sprintf(r1p, "@r%lu\n", (unsigned long)roff);
                        char* s1 = r1p + n;
                        if (use_complex_concat && !complex_concat_segs_all.empty()) {
                            // Rewrite R1 as [seg0][seg1]...[segN][UMI] for CB_samTagOut concat
                            int wpos = 0;
                            for (const auto& seg : complex_concat_segs_all) {
                                for (int j = 0; j < seg.len; ++j) {
                                    int src = seg.start + j;
                                    s1[wpos++] = (src < (int)r1len) ? NUM_TO_ASCII[r1s[src]] : 'N';
                                }
                            }
                            for (int j = 0; j < concat_umi_len_c; ++j) {
                                int src = concat_umi_start_0 + j;
                                s1[wpos++] = (src < (int)r1len) ? NUM_TO_ASCII[r1s[src]] : 'N';
                            }
                            s1 += wpos;
                            *s1++ = '\n'; *s1++ = '+'; *s1++ = '\n';
                            memset(s1, 'F', wpos); s1 += wpos; *s1++ = '\n';
                        } else {
                            for (uint16_t j = 0; j < r1len; ++j)
                                s1[j] = NUM_TO_ASCII[r1s[j]];
                            s1 += r1len;
                            *s1++ = '\n'; *s1++ = '+'; *s1++ = '\n';
                            memset(s1, 'F', r1len);
                            s1 += r1len;
                            *s1++ = '\n';
                        }
                        r1p = s1;

                        // R2 record
                        n = is_atac_mode
                            ? sprintf(r2p, "@%sr%lu\n", atac_qname_prefix, (unsigned long)roff)
                            : sprintf(r2p, "@r%lu\n", (unsigned long)roff);
                        char* s2 = r2p + n;
                        const char* r2_seq_start = s2;
                        for (uint16_t j = 0; j < r2len; ++j)
                            s2[j] = NUM_TO_ASCII[r2s[j]];
                        s2 += r2len;
                        // seq_len derived from actual bytes written ensures quality
                        // string length always matches sequence length per read,
                        // even for variable-length R2 with QualMode==NONE.
                        const uint16_t r2_seq_len =
                            static_cast<uint16_t>(s2 - r2_seq_start);
                        *s2++ = '\n'; *s2++ = '+'; *s2++ = '\n';
                        if (q) {
                            for (uint16_t j = 0; j < r2_seq_len; ++j)
                                s2[j] = static_cast<char>(q[j] + 33);
                        } else {
                            memset(s2, 'F', r2_seq_len);
                        }
                        s2 += r2_seq_len;
                        *s2++ = '\n';
                        r2p = s2;

                        roff++;
                    }
                }

                fb.r1_len = static_cast<size_t>(r1p - fb.r1.data());
                fb.r2_len = static_cast<size_t>(r2p - fb.r2.data());
                return fb;
            };

            // Pre-compute cumulative read offsets for naming
            std::vector<uint64_t> block_read_offsets(total_blocks);
            {
                uint64_t cum = 0;
                for (uint32_t b = 0; b < total_blocks; ++b) {
                    block_read_offsets[b] = cum;
                    lib1fq::BlockHeader bh;
                    ::pread(fd, &bh, sizeof(bh),
                            static_cast<off_t>(reader.block_offset(b)));
                    // Include dup expansion in offset calculation
                    cum += bh.n_reads;  // conservative; matches non-dedup case
                }
            }

            // BC frequency counts — populated during decode, used for
            // barcode discovery to avoid a separate R1.fastq scanning pass.
            std::vector<uint64_t> bc_freq;
            if (has_bc_dict && bc_dict_n > 0)
                bc_freq.resize(bc_dict_n + 1, 0);

            // Merged barcode counts for non-BC-dict path (packed 3-bit keys)
            std::unordered_map<uint64_t, uint64_t> merged_bc_counts;

            uint64_t reads_written = 0;
            uint32_t batch_size = static_cast<uint32_t>(decode_threads);
            for (uint32_t base = 0; base < total_blocks; base += batch_size) {
                uint32_t batch_end = std::min(base + batch_size, total_blocks);
                uint32_t n_batch = batch_end - base;

                std::vector<std::future<FmtBlock>> futures;
                futures.reserve(n_batch);
                for (uint32_t b = base; b < batch_end; ++b)
                    futures.push_back(std::async(std::launch::async,
                                     process_block, b, block_read_offsets[b]));

                for (uint32_t i = 0; i < n_batch; ++i) {
                    FmtBlock fb = futures[i].get();
                    fwrite(fb.r1.data(), 1, fb.r1_len, r1_fp);
                    fwrite(fb.r2.data(), 1, fb.r2_len, r2_fp);
                    reads_written += fb.n_reads;
                    // Accumulate BC index counts (sequential — no contention)
                    if (!bc_freq.empty()) {
                        for (uint32_t idx : fb.bc_indices) {
                            if (idx <= bc_dict_n) bc_freq[idx]++;
                        }
                    }
                    // Merge per-block barcode counts (non-BC-dict path)
                    for (auto& [key, cnt] : fb.bc_counts_packed)
                        merged_bc_counts[key] += cnt;
                    // Merge extra-segment counts for CB_UMI_Complex
                    if (!fb.extra_seg_counts.empty()) {
                        if (onefq_per_seg_counts.empty())
                            onefq_per_seg_counts.resize(fb.extra_seg_counts.size());
                        for (size_t ex = 0; ex < fb.extra_seg_counts.size(); ++ex)
                            for (auto& [key, cnt] : fb.extra_seg_counts[ex])
                                onefq_per_seg_counts[ex][key] += cnt;
                    }
                }
            }

            // Store BC dictionary and frequency counts for barcode discovery
            if (has_bc_dict && !bc_freq.empty()) {
                onefq_bc_dict_n = bc_dict_n;
                onefq_bc_freq = std::move(bc_freq);
                onefq_bc_dict_flat = reader.bc_dict_flat();
                onefq_bc_length = reader.bc_length();
            }
            // Store merged barcode counts (non-BC-dict path)
            if (!merged_bc_counts.empty())
                onefq_bc_counts = std::move(merged_bc_counts);

            fclose(r1_fp);
            fclose(r2_fp);
            decode_time = decode_sw.elapsed_s();
            decode_reads_out = reads_written;
            std::cerr << "[1fq-decode] Done: " << reads_written
                      << " reads (" << decode_threads << "T), "
                      << decode_time << "s\n";
        }  // end !use_fifo_decode file-mode block

        if (use_fifo_decode) {
            // ── FIFO mode: Pass 1 — lightweight BC count + first-25 R2 capture ──
            // Iterates all blocks sequentially to gather BC frequency data for Phase 0.9
            // and the first 25 R2 sequences for prefix detection.
            // No FASTQ files are written.  Peak memory: ~50 MB per block (one at a time).
            StopWatch pass1_sw;
            lib1fq::Reader reader;
            try {
                reader.open(onefq_file.c_str());
            } catch (const std::exception& e) {
                std::cerr << "[1fq-pass1] ERROR: Cannot open .1fq file: " << e.what() << "\n";
                _exit(1);
            }
            reader.set_verify_crc(false);
            auto hdr_p1      = reader.header();
            const uint32_t bc_dict_n_p1  = reader.bc_dict_size();
            const bool     has_bc_dict_p1 = reader.has_bc_dict();
            const uint32_t total_blocks_p1 = reader.block_count();
            const int      fd_p1    = reader.file_descriptor();
            const auto     codec_p1 = static_cast<lib1fq::Codec>(hdr_p1.codec);
            bool has_i2_p1 = is_atac_mode &&
                             (hdr_p1.reserved[0] & lib1fq::ExtraStreams::HAS_I2) != 0;

            // FIFO mode: set detected_bc_len from I2 stream length so Phase 0.9
            // unpacks barcodes with the correct length (not the default 12bp).
            // In file-mode decode this is set inside the !use_fifo_decode block.
            if (has_i2_p1 && detected_bc_len == 0) {
                uint16_t i2_len_p1 = hdr_p1.stream_lengths[2];
                if (i2_len_p1 > 0) {
                    detected_bc_len = static_cast<int>(i2_len_p1);
                    std::cerr << "[1fq-pass1] ATAC: I2 barcode stream "
                              << i2_len_p1 << "bp (FIFO mode)\n";
                }
            }

            std::vector<uint64_t> bc_freq_p1;
            if (has_bc_dict_p1 && bc_dict_n_p1 > 0)
                bc_freq_p1.resize(bc_dict_n_p1 + 1, 0);
            std::unordered_map<uint64_t, uint64_t> merged_bc_counts_p1;

            for (uint32_t bi = 0; bi < total_blocks_p1; ++bi) {
                uint64_t file_off_p1  = reader.block_offset(bi);
                uint32_t comp_size_p1 = reader.block_comp_size(bi);
                lib1fq::BlockHeader bh_p1;
                ::pread(fd_p1, &bh_p1, sizeof(bh_p1), static_cast<off_t>(file_off_p1));
                std::vector<uint8_t> comp_buf_p1(comp_size_p1);
                ::pread(fd_p1, comp_buf_p1.data(), comp_size_p1,
                        static_cast<off_t>(file_off_p1 + sizeof(bh_p1)));
                size_t est_raw_p1 = static_cast<size_t>(bh_p1.n_reads) * 200;
                std::vector<uint8_t> raw_buf_p1(est_raw_p1);
                size_t raw_size_p1;
                try {
                    raw_size_p1 = lib1fq::compress::decompress_block(
                        codec_p1, comp_buf_p1.data(), comp_size_p1,
                        raw_buf_p1.data(), raw_buf_p1.size());
                } catch (...) {
                    raw_buf_p1.resize(est_raw_p1 * 4);
                    raw_size_p1 = lib1fq::compress::decompress_block(
                        codec_p1, comp_buf_p1.data(), comp_size_p1,
                        raw_buf_p1.data(), raw_buf_p1.size());
                }
                lib1fq::DecodedBlock blk_p1;
                reader.decode_payload(raw_buf_p1.data(), raw_size_p1, bh_p1, blk_p1);

                const bool has_dups_p1 = !blk_p1.dup_counts.empty();
                for (uint32_t i = 0; i < blk_p1.n_reads; ++i) {
                    uint32_t repeat_p1 = has_dups_p1 ? blk_p1.dup_counts[i] : 1;
                    uint16_t r1len_p1  = blk_p1.r1_len(i);
                    uint16_t r2len_p1  = blk_p1.r2_len(i);
                    // Capture first 25 R2 sequences from first block for prefix detection
                    if (bi == 0 && fifo_first25_r2.size() < 25 && r2len_p1 > 0) {
                        static constexpr char N2AP[5] = {'A','C','G','T','N'};
                        const uint8_t* r2s_p1 = blk_p1.r2_seq(i);
                        std::string seq_p1(r2len_p1, ' ');
                        for (uint16_t j = 0; j < r2len_p1; ++j)
                            seq_p1[j] = N2AP[r2s_p1[j]];
                        fifo_first25_r2.push_back(std::move(seq_p1));
                    }
                    // BC counting (same logic as file-decode pass)
                    if (has_i2_p1) {
                        if (!blk_p1.i2_data.empty() && i < blk_p1.i2_lengths.size() &&
                            blk_p1.i2_lengths[i] > 0) {
                            const uint8_t* bcp = blk_p1.i2_seq(i);
                            uint16_t blen = blk_p1.i2_len(i);
                            if (blen > 0 && blen <= 21) {
                                uint64_t key = 0;
                                for (uint16_t j = 0; j < blen; ++j) key = (key << 3) | bcp[j];
                                merged_bc_counts_p1[key] += repeat_p1;
                            }
                        }
                    } else if (count_barcodes_in_decode && !has_bc_dict_p1 &&
                               r1len_p1 >= discover_bc_start_0 + discover_bc_len) {
                        uint64_t key = 0;
                        const uint8_t* bcs = blk_p1.r1_seq(i) + discover_bc_start_0;
                        for (int j = 0; j < discover_bc_len; ++j)
                            key = (key << 3) | bcs[j];
                        merged_bc_counts_p1[key] += repeat_p1;
                        // Extra segments for CB_UMI_Complex per-segment discovery
                        if (!complex_extra_segs.empty()) {
                            if (onefq_per_seg_counts.empty())
                                onefq_per_seg_counts.resize(complex_extra_segs.size());
                            for (size_t ex = 0; ex < complex_extra_segs.size(); ++ex) {
                                const auto& es = complex_extra_segs[ex];
                                if (es.r1_start < 0 ||
                                    r1len_p1 < static_cast<uint16_t>(es.r1_start + es.len)) continue;
                                uint64_t ekey = 0;
                                const uint8_t* ebcs = blk_p1.r1_seq(i) + es.r1_start;
                                for (int j = 0; j < es.len; ++j) ekey = (ekey << 3) | ebcs[j];
                                onefq_per_seg_counts[ex][ekey] += repeat_p1;
                            }
                        }
                    }
                    if (has_bc_dict_p1 && !blk_p1.bc_indices.empty() &&
                        i < blk_p1.bc_indices.size()) {
                        uint32_t idx = blk_p1.bc_indices[i];
                        if (idx <= bc_dict_n_p1) bc_freq_p1[idx] += repeat_p1;
                    }
                }
            }

            // Store BC data for Phase 0.9 fast paths
            if (has_bc_dict_p1 && !bc_freq_p1.empty()) {
                onefq_bc_dict_n    = bc_dict_n_p1;
                onefq_bc_freq      = std::move(bc_freq_p1);
                onefq_bc_dict_flat = reader.bc_dict_flat();
                onefq_bc_length    = reader.bc_length();
            }
            if (!merged_bc_counts_p1.empty())
                onefq_bc_counts = std::move(merged_bc_counts_p1);

            std::cerr << "[1fq-pass1] Scanned " << total_blocks_p1
                      << " blocks (" << pass1_sw.elapsed_s() << "s)\n";
            fifo_r1_path = r1_tmp;
            fifo_r2_path = r2_tmp;
        }

        reads_r1 = r1_tmp;
        reads_r2 = r2_tmp;

        // Sanity check: verify R2 (biological read) and R1 (barcode read) are
        // present and long enough.  B-G2-2: abort before STAR if either mate is
        // empty; write data_incomplete summary.json per contract §5.
        // Also detect constant 5' prefixes (e.g., adapter/primer contamination)
        // that would prevent STAR alignment.
        {
            bool has_seq = false;
            bool has_r1_seq = true;  // assume present unless file mode proves otherwise
            size_t max_r2_len = 0;
            std::vector<std::string> probe_seqs;

            if (use_fifo_decode) {
                // FIFO mode: use sequences captured in Pass 1 (no file to open)
                probe_seqs = fifo_first25_r2;
                for (const auto& s : probe_seqs) {
                    has_seq = true;
                    if (s.size() > max_r2_len) max_r2_len = s.size();
                }
            } else {
                // File mode: read first 25 R2 sequences from the decoded FASTQ
                FILE* r2_check = fopen(r2_tmp.c_str(), "r");
                if (r2_check) {
                    char line[4096];
                    int lineno = 0;
                    while (fgets(line, sizeof(line), r2_check) && lineno < 100) {
                        if (lineno % 4 == 1) { // sequence line
                            size_t len = strlen(line);
                            if (len > 1) len--; // strip newline
                            if (len > 0) has_seq = true;
                            if (len > max_r2_len) max_r2_len = len;
                            if (probe_seqs.size() < 25 && len > 0)
                                probe_seqs.emplace_back(line, len);
                        }
                        ++lineno;
                    }
                    fclose(r2_check);
                }
                // B-G2-2: also verify R1 (barcode read) is non-empty
                FILE* r1_check = fopen(r1_tmp.c_str(), "r");
                if (r1_check) {
                    has_r1_seq = false;  // reset — only set if we find a non-empty seq line
                    char line[4096];
                    int lineno = 0;
                    while (fgets(line, sizeof(line), r1_check) && lineno < 8) {
                        if (lineno % 4 == 1) {
                            size_t len = strlen(line);
                            if (len > 1) len--;
                            if (len > 0) { has_r1_seq = true; break; }
                        }
                        ++lineno;
                    }
                    fclose(r1_check);
                }
            }

                // Detect constant 5' prefix: count consecutive positions
                // from position 0 where >75% of reads have the same base.
                if (probe_seqs.size() >= 10 && max_r2_len >= 30) {
                    int prefix_len = 0;
                    for (size_t p = 0; p < max_r2_len; ++p) {
                        int counts[256] = {};
                        int total = 0;
                        for (const auto& seq : probe_seqs) {
                            if (p < seq.size()) {
                                counts[static_cast<unsigned char>(seq[p])]++;
                                total++;
                            }
                        }
                        if (total < 5) break;
                        int max_cnt = *std::max_element(counts, counts + 256);
                        if (static_cast<double>(max_cnt) / total > 0.75) {
                            prefix_len = static_cast<int>(p) + 1;
                        } else {
                            break;
                        }
                    }
                    // Only clip if prefix is substantial (>10bp) — short TSO
                    // prefixes (2-3bp GGG) are normal and should not be clipped.
                    // Safety cap: skip clipping if it would leave <30bp remaining
                    // (e.g. BD-Rhapsody targeted panels with 36bp R2).
                    // Use max() to avoid overwriting a longer clip value from .1fq
                    // metadata (written by the encode-time 1000-read detector).
                    if (prefix_len > 10) {
                        int remaining = static_cast<int>(max_r2_len) - prefix_len;
                        if (remaining >= 15) {
                            if (prefix_len > r2_clip5p) {
                                r2_clip5p = prefix_len;
                                std::cerr << "[singlet] WARNING: Detected " << prefix_len
                                          << "bp constant 5' prefix in R2 reads.\n"
                                          << "[singlet] Will trim " << prefix_len
                                          << " bases from R2 5' end (file level; cDNA only).\n";
                            }
                        } else {
                            std::cerr << "[singlet] WARNING: Detected " << prefix_len
                                      << "bp constant 5' prefix in R2 (total " << max_r2_len
                                      << "bp). Clipping would leave <15bp — skipping clip.\n";
                        }
                    }
                }
                if (!has_seq || !has_r1_seq) {
                    const bool r2_empty = !has_seq;
                    const bool r1_empty = !has_r1_seq;
                    std::cerr << "[singlet] ERROR: ";
                    if (r1_empty && r2_empty)
                        std::cerr << "Both R1 and R2 (barcode + biological reads) are empty.\n";
                    else if (r2_empty)
                        std::cerr << "R2 (biological read) is empty for all reads.\n";
                    else
                        std::cerr << "R1 (barcode read) is empty for all reads.\n";
                    std::cerr << "[singlet] This may be a single-end dataset or protocol misclassification.\n"
                              << "[singlet] Cannot align without paired-end biological read data.\n";
                    if (!use_fifo_decode) {
                        unlink(r1_tmp.c_str());
                        unlink(r2_tmp.c_str());
                        rmdir(decode_dir.c_str());
                    }
                    // B-G2-2: write data_incomplete summary.json before exiting.
                    // Downstream pipeline reads this to categorise the failure correctly.
                    // Exit 0 = run terminated cleanly per contract; it is not a crash.
                    {
                        mkdir_p(out_prefix);
                        // Extract sample_id from .1fq filename (stem only)
                        std::string sample_id = onefq_file;
                        { auto p = sample_id.rfind('/');  if (p != std::string::npos) sample_id = sample_id.substr(p+1); }
                        { auto p = sample_id.rfind(".1fq"); if (p != std::string::npos) sample_id = sample_id.substr(0, p); }

                        // Compose reason string
                        std::string di_reason;
                        if (r1_empty && r2_empty)
                            di_reason = "decode produced empty R1 and R2; suspected protocol misdetection or VDB orientation issue; re-run with --protocol or --swap-reads";
                        else if (r2_empty)
                            di_reason = "decode produced empty R2; suspected protocol misdetection or VDB orientation issue; re-run with --protocol or --swap-reads";
                        else
                            di_reason = "decode produced empty R1 (barcode read); suspected protocol misdetection; re-run with --protocol";

                        // summary.json — all REQUIRED_SUMMARY_KEYS with zero/null values
                        {
                            std::ofstream sj(out_prefix + "/summary.json");
                            if (sj) {
                                sj << "{\n"
                                   << "  \"schema_version\": \"1.0\",\n"
                                   << "  \"sample_id\": \"" << sample_id << "\",\n"
                                   << "  \"status\": \"data_incomplete\",\n"
                                   << "  \"protocol_id\": 0,\n"
                                   << "  \"protocol_name\": \"\",\n"
                                   << "  \"species\": \"\",\n"
                                   << "  \"reference_build\": \"\",\n"
                                   << "  \"n_input_reads\": " << onefq_n_reads << ",\n"
                                   << "  \"n_uniquely_mapped\": 0,\n"
                                   << "  \"uniquely_mapped_pct\": 0.0,\n"
                                   << "  \"n_cells_called\": 0,\n"
                                   << "  \"median_umi_per_cell\": 0.0,\n"
                                   << "  \"median_genes_per_cell\": 0.0,\n"
                                   << "  \"memory_tier\": \"unknown\",\n"
                                   << "  \"peak_rss_gb\": 0.0,\n"
                                   << "  \"wall_seconds\": " << total_sw.elapsed_s() << ",\n"
                                   << "  \"track\": \"A\",\n"
                                   << "  \"donor\": null,\n"
                                   << "  \"mt\": null,\n"
                                   << "  \"nonhost\": null,\n"
                                   << "  \"reason\": \"" << di_reason << "\"\n"
                                   << "}\n";
                            }
                        }
                        // provenance.json — minimal, all REQUIRED_PROVENANCE_KEYS
                        {
                            std::ofstream pj(out_prefix + "/provenance.json");
                            if (pj) {
                                pj << "{\n"
                                   << "  \"schema_version\": \"1.0\",\n"
                                   << "  \"singlet_git_sha\": \"unknown\",\n"
                                   << "  \"build_flags\": \"\",\n"
                                   << "  \"command_line\": \"\",\n"
                                   << "  \"env\": {},\n"
                                   << "  \"host\": \"\",\n"
                                   << "  \"input\": \"" << onefq_file << "\",\n"
                                   << "  \"references\": {},\n"
                                   << "  \"output_schema_version\": \"1.0\"\n"
                                   << "}\n";
                            }
                        }
                        std::cerr << "[singlet] data_incomplete: summary.json written to "
                                  << out_prefix << "\n";
                    }
                    return 0;  // B-G2-2: clean exit per contract (summary.json carries status)
                }
                if (max_r2_len < 30) {
                    std::cerr << "[singlet] WARNING: R2 (biological read) max length is "
                              << max_r2_len << "bp (< 30bp minimum for alignment).\n"
                              << "[singlet] Alignment results may be poor. "
                              << "Check that protocol detection assigned R1/R2 correctly.\n";
                }
        }

        // Swap R1/R2 for inverted protocols (barcode in R2, cDNA in R1)
        // so STAR gets --readFilesIn cDNA barcode in the correct order
        if (reads_inverted) {
            std::swap(reads_r1, reads_r2);
            std::cerr << "[singlet] Swapped R1↔R2 for STAR (barcode was in R2)\n";
        }

        // File-level 5' trim of the cDNA read (reads_r2 after any R1/R2 swap).
        //
        // WHY file-level (primary path):
        // File-level trimming is unambiguous: reads_r2 is always the cDNA file
        // at this point (after any reads_inverted swap), reads_r1 is always the
        // barcode file and is never touched.  r2_clip5p is 0 for inverted protocols
        // (barcode reads are too short for prefix detection) so the swap is safe.
        //
        // STAR fallback (when file-level open fails):
        // Uses two-value "--clip5pNbases N 0" — clip cDNA (file 1), skip barcode
        // (file 2).  The single-value form clips BOTH mates and destroys barcodes
        // → 0 cells called (confirmed empirically, job 360648 / SRR34789664).
        if (r2_clip5p > 0) {
            if (use_fifo_decode) {
                // FIFO mode: r2_clip5p will be applied inline during Pass 2 FIFO streaming.
                // Do NOT reset here — the FIFO creation block (after Phase 0.9) captures the
                // value and resets r2_clip5p to 0 before STAR args are built.
                std::cerr << "[singlet] 5' clip: " << r2_clip5p
                          << "bp will be applied inline in Pass 2 FIFO encoder\n";
            } else {
            std::string reads_r2_clipped = reads_r2 + ".clip5p";
            FILE* clip_in  = fopen(reads_r2.c_str(), "r");
            FILE* clip_out = fopen(reads_r2_clipped.c_str(), "w");
            bool clip_ok = (clip_in && clip_out);
            if (clip_ok) {
                char cline[65536];
                int cln = 0;
                while (fgets(cline, sizeof(cline), clip_in)) {
                    int lmod = cline[0] == '\0' ? 0 : (cln % 4);
                    if (lmod == 1 || lmod == 3) {
                        // Sequence (1) or quality (3) line — skip first r2_clip5p bytes.
                        size_t clen = strlen(cline);
                        size_t has_nl = (clen > 0 && cline[clen-1] == '\n') ? 1 : 0;
                        size_t skip = (size_t)r2_clip5p < (clen - has_nl)
                                      ? (size_t)r2_clip5p : (clen - has_nl);
                        fputs(cline + skip, clip_out);
                    } else {
                        fputs(cline, clip_out);
                    }
                    ++cln;
                }
                fclose(clip_in);  clip_in  = nullptr;
                fclose(clip_out); clip_out = nullptr;
                // Atomically replace the original R2 file with the trimmed version.
                unlink(reads_r2.c_str());
                rename(reads_r2_clipped.c_str(), reads_r2.c_str());
                std::cerr << "[singlet] 5' clip: trimmed " << r2_clip5p
                          << "bp from cDNA reads (file level); barcode reads untouched\n";
                r2_clip5p = 0;  // signal that clipping is done — no STAR flag needed
            } else {
                if (clip_in)  { fclose(clip_in);  clip_in  = nullptr; }
                if (clip_out) { fclose(clip_out); unlink(reads_r2_clipped.c_str()); }
                std::cerr << "[singlet] WARNING: Could not open R2 temp file for file-level 5' trim; "
                          << "r2_clip5p=" << r2_clip5p << " will be passed to STAR as fallback\n";
                // r2_clip5p left non-zero → handled by the STAR fallback block below
            }
            }  // end !use_fifo_decode else branch
        }

        std::cerr << "[singlet] .1fq decoded to temp files\n";
    }

    // ── Phase 0.9: Barcode discovery for --whitelist None / auto-resolved whitelist without --barcodes ──
    std::string auto_barcode_file;  // temp file for auto-discovered barcodes
    if (barcode_file.empty() && (whitelist_file == "None" || whitelist_file == "none" || whitelist_auto_resolved)) {

        // ── Shallow-library whitelist fallback helper ────────────────────────────
        // Called when the ≥100-reads filter produces 0 barcodes (shallow libraries
        // with ~3-4M reads spread across millions of barcodes cannot meet this
        // threshold).  If the protocol whitelist was auto-resolved, load it as the
        // candidate set and let STAR's CB-matching + EmptyDrops filter further.
        // Returns the number of barcodes written to auto_barcode_file (0 on failure).
        // ─────────────────────────────────────────────────────────────────────────
        auto apply_whitelist_fallback = [&](
            const std::string& discovery_path_label
        ) -> size_t {
            bool wl_is_real = (whitelist_auto_resolved &&
                               whitelist_file != "None" && whitelist_file != "none" &&
                               !whitelist_file.empty());
            if (!wl_is_real) return 0;
            std::ifstream wl_in(whitelist_file);
            if (!wl_in) {
                std::cerr << "[auto_barcodes] WARNING: whitelist fallback failed"
                          << " (cannot read " << whitelist_file << ")\n";
                return 0;
            }
            std::vector<std::string> wl_bcs;
            wl_bcs.reserve(4000000);
            std::string wl_line;
            while (std::getline(wl_in, wl_line)) {
                if (!wl_line.empty()) wl_bcs.push_back(wl_line);
            }
            wl_in.close();
            if (wl_bcs.empty()) return 0;
            std::ofstream bc_out(auto_barcode_file);
            for (const auto& bc : wl_bcs) bc_out << bc << "\n";
            bc_out.close();
            barcode_file = auto_barcode_file;
            config.barcode_path = barcode_file;
            std::cerr << "[auto_barcodes] fallback: ≥100 yielded 0 on "
                      << discovery_path_label << ", using whitelist ("
                      << wl_bcs.size() << " barcodes from "
                      << whitelist_file << ")\n";
            return wl_bcs.size();
        };

        // Compute barcode geometry (used by all paths below)
        int bc_start_0 = (detected_cb_start > 0 ? detected_cb_start - 1 : 0);
        int bc_len = (detected_bc_len > 0 ? detected_bc_len : 12);
        if (detected_soloType == "CB_UMI_Complex" && !detected_cb_positions.empty()) {
            auto parse_pos_09 = [](const std::string& p) -> std::pair<int,int> {
                int vals[4] = {0,0,0,0};
                size_t pos = 0; int vi = 0;
                for (size_t i = 0; i <= p.size() && vi < 4; ++i) {
                    if (i == p.size() || p[i] == '_') {
                        vals[vi++] = std::atoi(p.substr(pos, i - pos).c_str());
                        pos = i + 1;
                    }
                }
                return {vals[1], vals[3]};
            };
            auto [cb_s, cb_e] = parse_pos_09(detected_cb_positions[0]);
            bc_start_0 = cb_s;
            bc_len = cb_e - cb_s + 1;
        }

        // Fast path: if we captured BC index frequencies during .1fq decode,
        // use them directly instead of scanning the decoded R1.fastq file.
        // This saves ~10s by avoiding a sequential text scan of all reads.
        if (onefq_bc_dict_n > 0 && !onefq_bc_freq.empty() && onefq_bc_length > 0) {
            static constexpr char NUM_TO_ASCII_BC[5] = {'A','C','G','T','N'};

            // Build sorted (count, barcode_string) list from BC index frequencies
            std::vector<std::pair<uint64_t, std::string>> sorted_bcs;
            for (uint32_t idx = 0; idx < onefq_bc_dict_n; ++idx) {
                if (onefq_bc_freq[idx] >= 100) {
                    // Convert numeric BC to ASCII string
                    std::string bc(onefq_bc_length, ' ');
                    for (uint16_t j = 0; j < onefq_bc_length; ++j) {
                        bc[j] = NUM_TO_ASCII_BC[onefq_bc_dict_flat[
                            static_cast<size_t>(idx) * onefq_bc_length + j]];
                    }
                    sorted_bcs.emplace_back(onefq_bc_freq[idx], std::move(bc));
                }
            }
            std::sort(sorted_bcs.begin(), sorted_bcs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

            // Write barcode file
            auto_barcode_file = out_prefix + "/auto_barcodes.tsv";
            std::ofstream bc_out(auto_barcode_file);
            for (auto& [cnt, bc] : sorted_bcs)
                bc_out << bc << "\n";
            bc_out.close();
            barcode_file = auto_barcode_file;
            config.barcode_path = barcode_file;

            std::cerr << "[singlet] Discovered " << sorted_bcs.size()
                      << " barcodes (≥100 reads) from .1fq BC dictionary ("
                      << onefq_bc_dict_n << " entries)\n";

            if (sorted_bcs.empty()) {
                // Shallow library: try whitelist fallback before aborting
                size_t n_fallback = apply_whitelist_fallback("BC-dict");
                if (n_fallback == 0) {
                    std::cerr << "ERROR: No barcodes in BC dictionary have ≥100 reads"
                              << " and whitelist fallback unavailable.\n";
                    return 1;
                }
            }

            // Load pileup engine with discovered barcodes
            if (!is_atac_mode) {
                engine_ptr = std::make_unique<singlet::PileupEngine>(config);
                if (!engine_ptr->load_references()) {
                    std::cerr << "ERROR: Failed to load pileup references after barcode discovery\n";
                    return 1;
                }
            }
        } else if (!onefq_bc_counts.empty()) {
            // Fast path 2: barcode counts accumulated during .1fq decode
            // (parallel counting with packed 3-bit keys, no BC dictionary needed)
            static constexpr char UNPACK_BASE[5] = {'A','C','G','T','N'};
            std::vector<std::pair<uint64_t, std::string>> sorted_bcs;
            for (auto& [key, cnt] : onefq_bc_counts) {
                if (cnt >= 100) {
                    // Unpack 3-bit-per-base uint64_t key to ASCII string
                    std::string bc(bc_len, ' ');
                    uint64_t k = key;
                    for (int j = bc_len - 1; j >= 0; --j) {
                        bc[j] = UNPACK_BASE[k & 0x7];
                        k >>= 3;
                    }
                    // Skip all-N or all-G barcodes
                    bool valid = false;
                    for (char c : bc) {
                        if (c != 'N' && c != 'G') { valid = true; break; }
                    }
                    if (valid) sorted_bcs.emplace_back(cnt, std::move(bc));
                }
            }
            std::sort(sorted_bcs.begin(), sorted_bcs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

            auto_barcode_file = out_prefix + "/auto_barcodes.tsv";
            std::ofstream bc_out(auto_barcode_file);
            for (auto& [cnt, bc] : sorted_bcs)
                bc_out << bc << "\n";
            bc_out.close();
            barcode_file = auto_barcode_file;
            config.barcode_path = barcode_file;

            std::cerr << "[singlet] Discovered " << sorted_bcs.size()
                      << " barcodes (≥100 reads) from decode-time counting ("
                      << onefq_bc_counts.size() << " unique)\n";

            // ── Guard 2: Barcode explosion warning ──
            if (onefq_bc_counts.size() > 200000) {
                uint64_t total_reads_g2 = 0;
                for (auto& [k, v] : onefq_bc_counts) total_reads_g2 += v;
                double mean_g2 = onefq_bc_counts.empty() ? 0.0
                    : static_cast<double>(total_reads_g2) / onefq_bc_counts.size();
                if (mean_g2 < 5.0) {
                    std::cerr << "[singlet] WARNING: Barcode explosion detected — "
                              << onefq_bc_counts.size() << " unique barcodes with mean "
                              << std::fixed << std::setprecision(1) << mean_g2
                              << " reads/barcode. Check protocol detection.\n";
                }
            }

            if (sorted_bcs.empty()) {
                // Shallow library: try whitelist fallback, then progressive threshold
                size_t n_fallback = apply_whitelist_fallback("decode-time-count");
                if (n_fallback == 0) {
                    // Progressive threshold using onefq_bc_counts (uint64 packed keys)
                    for (uint64_t thr : {50ULL, 20ULL, 10ULL}) {
                        sorted_bcs.clear();
                        for (auto& [key, cnt] : onefq_bc_counts) {
                            if (cnt >= thr) {
                                std::string bc(bc_len, ' ');
                                uint64_t k = key;
                                for (int j = bc_len - 1; j >= 0; --j) {
                                    bc[j] = UNPACK_BASE[k & 0x7]; k >>= 3;
                                }
                                bool valid = false;
                                for (char c : bc) { if (c != 'N' && c != 'G') { valid = true; break; } }
                                if (valid) sorted_bcs.emplace_back(cnt, std::move(bc));
                            }
                        }
                        if (sorted_bcs.size() >= 10) {
                            std::sort(sorted_bcs.begin(), sorted_bcs.end(),
                                      [](const auto& a, const auto& b){ return a.first > b.first; });
                            std::ofstream bc_out(auto_barcode_file);
                            for (auto& [cnt, bc] : sorted_bcs) bc_out << bc << "\n";
                            bc_out.close();
                            barcode_file = auto_barcode_file;
                            config.barcode_path = barcode_file;
                            std::cerr << "[auto_barcodes] fallback: using ≥" << thr
                                      << " threshold (" << sorted_bcs.size() << " barcodes)\n";
                            n_fallback = sorted_bcs.size();
                            break;
                        }
                    }
                }
                if (n_fallback == 0) {
                    // Top-5000 last resort
                    std::vector<std::pair<uint64_t, std::string>> all_bcs;
                    all_bcs.reserve(onefq_bc_counts.size());
                    for (auto& [key, cnt] : onefq_bc_counts) {
                        std::string bc(bc_len, ' '); uint64_t k = key;
                        for (int j = bc_len - 1; j >= 0; --j) { bc[j] = UNPACK_BASE[k & 0x7]; k >>= 3; }
                        bool valid = false;
                        for (char c : bc) { if (c != 'N' && c != 'G') { valid = true; break; } }
                        if (valid) all_bcs.emplace_back(cnt, std::move(bc));
                    }
                    std::sort(all_bcs.begin(), all_bcs.end(),
                              [](const auto& a, const auto& b){ return a.first > b.first; });
                    size_t take = std::min(all_bcs.size(), size_t(5000));
                    if (take >= 10) {
                        std::ofstream bc_out(auto_barcode_file);
                        for (size_t i = 0; i < take; ++i) bc_out << all_bcs[i].second << "\n";
                        bc_out.close();
                        barcode_file = auto_barcode_file;
                        config.barcode_path = barcode_file;
                        std::cerr << "[auto_barcodes] fallback: top-" << take
                                  << " barcodes by count (max=" << all_bcs[0].first << ")\n";
                        n_fallback = take;
                    }
                }
                if (n_fallback == 0) {
                    std::cerr << "ERROR: No barcodes discovered with ≥100 reads from decode"
                              << " and all fallbacks exhausted.\n";
                    return 1;
                }
            }

            if (!is_atac_mode) {
                engine_ptr = std::make_unique<singlet::PileupEngine>(config);
                if (!engine_ptr->load_references()) {
                    std::cerr << "ERROR: Failed to load pileup references after barcode discovery\n";
                    return 1;
                }
            }
        } else {
        // Fallback: scan decoded R1.fastq for barcodes (non-.1fq or no BC dict)
        // Extract barcodes from decoded R1 and count frequencies.
        // The barcode read is reads_r1 (unless inverted — then it was already swapped).
        // For STAR, R1 is the barcode read (reads_r1 after any swap).
        // Actually reads_r1 is the barcode read BEFORE swap; after swap for inverted,
        // reads_r1 = cDNA and reads_r2 = barcode. But we need the BARCODE read.
        std::string bc_read_file = reads_inverted ? reads_r2 : reads_r1;

        // For CB_UMI_Complex protocols, log the discovery positions.
        if (detected_soloType == "CB_UMI_Complex" && !detected_cb_positions.empty()) {
            std::cerr << "[singlet] CB_UMI_Complex: using first CB segment for discovery "
                      << "(start=" << bc_start_0 << ", len=" << bc_len << ")\n";
        }

        std::cerr << "[singlet] Auto-discovering barcodes from " << bc_read_file
                  << " (start=" << bc_start_0 << ", len=" << bc_len << ")\n";

        std::unordered_map<std::string, uint32_t> bc_counts;
        std::ifstream bc_in(bc_read_file);
        std::string line;
        uint64_t line_num = 0;
        while (std::getline(bc_in, line)) {
            line_num++;
            if (line_num % 4 != 2) continue; // sequence lines only (line 2 of each FASTQ record)
            if ((int)line.size() >= bc_start_0 + bc_len) {
                std::string bc = line.substr(bc_start_0, bc_len);
                // Skip all-N or all-G barcodes
                bool valid = false;
                for (char c : bc) {
                    if (c != 'N' && c != 'G') { valid = true; break; }
                }
                if (valid) bc_counts[bc]++;
            }
        }
        bc_in.close();

        // Filter: keep barcodes with ≥100 reads, sort by count descending
        std::vector<std::pair<uint32_t, std::string>> sorted_bcs;
        sorted_bcs.reserve(bc_counts.size());
        for (auto& [bc, cnt] : bc_counts) {
            if (cnt >= 100) sorted_bcs.push_back({cnt, bc});
        }
        std::sort(sorted_bcs.begin(), sorted_bcs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Write to temp file
        auto_barcode_file = out_prefix + "/auto_barcodes.tsv";
        std::ofstream bc_out(auto_barcode_file);
        for (auto& [cnt, bc] : sorted_bcs) {
            bc_out << bc << "\n";
        }
        bc_out.close();
        barcode_file = auto_barcode_file;
        config.barcode_path = barcode_file;

        std::cerr << "[singlet] Discovered " << sorted_bcs.size()
                  << " barcodes (≥100 reads) from " << bc_counts.size()
                  << " unique sequences\n";

        // ── Guard 2: Barcode explosion warning (fallback text-scan path) ──
        if (bc_counts.size() > 200000) {
            uint64_t total_reads_g2b = 0;
            for (auto& [bc, cnt] : bc_counts) total_reads_g2b += cnt;
            double mean_g2b = bc_counts.empty() ? 0.0
                : static_cast<double>(total_reads_g2b) / bc_counts.size();
            if (mean_g2b < 5.0) {
                std::cerr << "[singlet] WARNING: Barcode explosion detected — "
                          << bc_counts.size() << " unique barcodes with mean "
                          << std::fixed << std::setprecision(1) << mean_g2b
                          << " reads/barcode. Check protocol detection.\n";
            }
        }

        if (sorted_bcs.empty()) {
            // Retry with standard 10x-v3 barcode geometry if initial detection
            // used different parameters (common for low-confidence detections).
            bool retried = false;
            if (bc_len != 16 || bc_start_0 != 0) {
                std::cerr << "[singlet] No barcodes found with detected params"
                          << " (start=" << bc_start_0 << ", len=" << bc_len
                          << "). Retrying with 10x-v3 defaults (start=0, len=16)...\n";
                bc_counts.clear();
                std::ifstream bc_retry(bc_read_file);
                line_num = 0;
                while (std::getline(bc_retry, line)) {
                    line_num++;
                    if (line_num % 4 != 2) continue;
                    if ((int)line.size() >= 16) {
                        std::string bc = line.substr(0, 16);
                        bool valid = false;
                        for (char c : bc) {
                            if (c != 'N' && c != 'G') { valid = true; break; }
                        }
                        if (valid) bc_counts[bc]++;
                    }
                }
                bc_retry.close();
                sorted_bcs.clear();
                for (auto& [bc, cnt] : bc_counts) {
                    if (cnt >= 100) sorted_bcs.push_back({cnt, bc});
                }
                std::sort(sorted_bcs.begin(), sorted_bcs.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
                if (!sorted_bcs.empty()) {
                    retried = true;
                    detected_bc_len = 16;
                    detected_cb_start = 1;
                    detected_umi_start = 17;
                    umi_len = 12;
                    std::cerr << "[singlet] Retry succeeded: " << sorted_bcs.size()
                              << " barcodes found with 10x-v3 geometry\n";
                    // Rewrite barcode file
                    std::ofstream bc_rewrite(auto_barcode_file);
                    for (auto& [cnt, bc] : sorted_bcs) {
                        bc_rewrite << bc << "\n";
                    }
                    bc_rewrite.close();
                }
            }
            if (sorted_bcs.empty()) {
                // Shallow library: try whitelist fallback, then progressive threshold
                size_t n_fallback = apply_whitelist_fallback("text-scan");
                if (n_fallback == 0) {
                    for (uint32_t thr : {50u, 20u, 10u}) {
                        sorted_bcs.clear();
                        for (auto& [bc, cnt] : bc_counts)
                            if (cnt >= thr) sorted_bcs.push_back({cnt, bc});
                        if (sorted_bcs.size() >= 10) {
                            std::sort(sorted_bcs.begin(), sorted_bcs.end(),
                                      [](const auto& a, const auto& b){ return a.first > b.first; });
                            std::ofstream bc_out(auto_barcode_file);
                            for (auto& [cnt, bc] : sorted_bcs) bc_out << bc << "\n";
                            bc_out.close();
                            barcode_file = auto_barcode_file;
                            config.barcode_path = barcode_file;
                            std::cerr << "[auto_barcodes] fallback: using ≥" << thr
                                      << " threshold (" << sorted_bcs.size() << " barcodes)\n";
                            n_fallback = sorted_bcs.size();
                            break;
                        }
                    }
                }
                if (n_fallback == 0) {
                    // Top-5000 last resort
                    sorted_bcs.clear();
                    for (auto& [bc, cnt] : bc_counts) sorted_bcs.push_back({cnt, bc});
                    std::sort(sorted_bcs.begin(), sorted_bcs.end(),
                              [](const auto& a, const auto& b){ return a.first > b.first; });
                    if (sorted_bcs.size() > 5000) sorted_bcs.resize(5000);
                    if (sorted_bcs.size() >= 10) {
                        std::ofstream bc_out(auto_barcode_file);
                        for (auto& [cnt, bc] : sorted_bcs) bc_out << bc << "\n";
                        bc_out.close();
                        barcode_file = auto_barcode_file;
                        config.barcode_path = barcode_file;
                        std::cerr << "[auto_barcodes] fallback: top-" << sorted_bcs.size()
                                  << " barcodes by count (max=" << sorted_bcs[0].first << ")\n";
                        n_fallback = sorted_bcs.size();
                    }
                }
                if (n_fallback == 0) {
                    std::cerr << "ERROR: No barcodes discovered with ≥100 reads. "
                              << "Check protocol detection or provide --barcodes manually.\n";
                    return 1;
                }
            }
        }

        // Now create and load the pileup engine with discovered barcodes
        if (!is_atac_mode) {
            engine_ptr = std::make_unique<singlet::PileupEngine>(config);
            if (!engine_ptr->load_references()) {
                std::cerr << "ERROR: Failed to load pileup references after barcode discovery\n";
                return 1;
            }
        }
        } // end else (fallback text scanning)
        std::cerr << "[singlet] References loaded: " << ref_sw.elapsed_s() << "s\n";
    }

    // ── Per-segment whitelist discovery for CB_UMI_Complex protocols ──
    // After Phase 0.9 has discovered seg-0 barcodes (in auto_barcodes.tsv),
    // write per-segment whitelist files for segs 1..N-1 from onefq_per_seg_counts.
    // These are used with CB_UMI_Complex STAR mode instead of falling back to
    // CB_samTagOut (which uses only seg-0 and produces a wrong composite barcode).
    // Per-segment files: auto_barcodes_seg0.tsv (= auto_barcodes.tsv), auto_barcodes_seg1.tsv, ...
    std::vector<std::string> per_seg_barcode_files; // length == n_segs if all discovered, else empty
    if (detected_soloType == "CB_UMI_Complex" &&
        detected_cb_positions.size() > 1 &&
        !auto_barcode_file.empty() &&
        !onefq_per_seg_counts.empty() &&
        onefq_per_seg_counts.size() == complex_extra_segs.size() &&
        !complex_extra_segs.empty()) {
        static constexpr char UNPACK_BASE_SEG[5] = {'A','C','G','T','N'};
        constexpr uint64_t MIN_SEG_READS = 100;
        bool all_segs_ok = true;
        // seg-0: already in auto_barcode_file
        per_seg_barcode_files.push_back(auto_barcode_file);
        for (size_t ex = 0; ex < onefq_per_seg_counts.size(); ++ex) {
            if (complex_extra_segs[ex].r1_start < 0) {
                // This segment position is not in R1 (e.g. R2-relative for SPLiT-seq).
                // Cannot discover from current data; abort per-segment mode.
                all_segs_ok = false;
                break;
            }
            int seg_len = complex_extra_segs[ex].len;
            std::vector<std::pair<uint64_t,std::string>> seg_bcs;
            for (auto& [key, cnt] : onefq_per_seg_counts[ex]) {
                if (cnt >= MIN_SEG_READS && seg_len > 0 && seg_len <= 21) {
                    std::string bc(seg_len, ' ');
                    uint64_t k = key;
                    for (int j = seg_len - 1; j >= 0; --j) {
                        bc[j] = UNPACK_BASE_SEG[k & 0x7]; k >>= 3;
                    }
                    bool valid = false;
                    for (char c : bc) { if (c != 'N' && c != 'G') { valid = true; break; } }
                    if (valid) seg_bcs.emplace_back(cnt, std::move(bc));
                }
            }
            if (seg_bcs.empty()) {
                all_segs_ok = false;
                break;
            }
            std::sort(seg_bcs.begin(), seg_bcs.end(),
                      [](const auto& a, const auto& b){ return a.first > b.first; });
            std::string seg_file = out_prefix + "/auto_barcodes_seg" +
                                   std::to_string(ex + 1) + ".tsv";
            {
                std::ofstream out(seg_file);
                for (auto& [cnt, bc] : seg_bcs) out << bc << "\n";
            }
            per_seg_barcode_files.push_back(seg_file);
            std::cerr << "[singlet] CB_UMI_Complex seg" << (ex + 1)
                      << ": " << seg_bcs.size() << " barcodes discovered (≥" << MIN_SEG_READS
                      << " reads) → " << seg_file << "\n";
        }
        if (!all_segs_ok || per_seg_barcode_files.size() != detected_cb_positions.size()) {
            // Clear: not all segments discovered, fall back to CB_samTagOut
            per_seg_barcode_files.clear();
            std::cerr << "[singlet] CB_UMI_Complex: incomplete per-segment discovery;"
                         " falling back to CB_samTagOut with seg-0\n";
        } else {
            // Explosion guard: check Cartesian product of per-segment whitelist sizes.
            // If product > 2M combinations, fall back to CB_samTagOut to avoid bad_alloc.
            size_t product = 1;
            bool explodes = false;
            for (auto& f : per_seg_barcode_files) {
                size_t n = 0;
                std::ifstream inf(f);
                std::string ln;
                while (std::getline(inf, ln)) if (!ln.empty()) ++n;
                if (n == 0 || (product > 0 && product > 2000000ULL / n)) {
                    explodes = true; break;
                }
                product *= n;
            }
            if (explodes) {
                per_seg_barcode_files.clear();
                std::cerr << "[singlet] CB_UMI_Complex: per-segment Cartesian product > 2M;"
                             " falling back to CB_samTagOut with seg-0\n";
            } else {
                std::cerr << "[singlet] CB_UMI_Complex: using CB_UMI_Complex mode with "
                          << per_seg_barcode_files.size() << " per-segment whitelist files"
                          << " (product=" << product << " max combinations)\n";
            }
        }
    }

    // ── Combined concatenated whitelist for CB_UMI_Complex → CB_samTagOut ──
    // Generate Cartesian product whitelist from per-segment WL files.
    // Runs in the PARENT so the combined WL file exists before STAR is forked.
    if (use_complex_concat) {
        const size_t n_segs = detected_cb_positions.size();
        std::vector<std::string> all_seg_wl;
        if (per_seg_barcode_files.size() == n_segs) {
            all_seg_wl = per_seg_barcode_files;  // auto-discovered in Phase 0.9
        } else {
            // Try disk-based per-segment WL files
            std::string bin_dir;
            {
                char exe_buf[PATH_MAX]{};
                ssize_t exe_len = ::readlink("/proc/self/exe", exe_buf, sizeof(exe_buf)-1);
                if (exe_len > 0) {
                    exe_buf[exe_len] = '\0';
                    bin_dir = exe_buf;
                    auto sl = bin_dir.rfind('/');
                    if (sl != std::string::npos) bin_dir.resize(sl);
                }
            }
            std::string proto_tag;
            for (const auto& sp : lib1fq::known_protocols())
                if (sp.protocol_id == onefq_protocol_id) { proto_tag = sp.tag; break; }
            if (!meta_json_proto_tag.empty()) proto_tag = meta_json_proto_tag;
            if (!bin_dir.empty() && !proto_tag.empty())
                all_seg_wl = lib1fq::resolve_per_seg_whitelists(bin_dir, proto_tag, n_segs);
            // Fallback: replicate a single available WL across all segments
            if (all_seg_wl.empty()) {
                const std::string& eff = (!barcode_file.empty()
                    && barcode_file != "None" && barcode_file != "none"
                    && barcode_file != "SMARTSEQ2_NONE") ? barcode_file : whitelist_file;
                if (!eff.empty() && eff != "None" && eff != "none")
                    all_seg_wl.assign(n_segs, eff);
            }
        }
        if (all_seg_wl.size() == n_segs) {
            std::vector<std::vector<std::string>> seg_entries(n_segs);
            bool cc_wl_ok = true;
            for (size_t s = 0; s < n_segs && cc_wl_ok; ++s) {
                std::ifstream inf(all_seg_wl[s]);
                if (!inf) { cc_wl_ok = false; break; }
                std::string ln;
                while (std::getline(inf, ln)) if (!ln.empty()) seg_entries[s].push_back(ln);
                if (seg_entries[s].empty()) cc_wl_ok = false;
            }
            if (cc_wl_ok) {
                size_t product = 1;
                for (auto& se : seg_entries) product *= se.size();
                const char* td_env = std::getenv("TMPDIR");
                std::string wl_base = td_env ? td_env : "/dev/shm";
                std::string wl_path = wl_base + "/singlet_combwl_"
                                      + std::to_string(getpid()) + ".txt";
                std::ofstream wl_out(wl_path);
                if (!wl_out) {
                    std::cerr << "[singlet] WARNING: cannot write combined WL to "
                              << wl_path << " — concat mode without whitelist\n";
                } else {
                    const size_t ns = n_segs;
                    std::vector<size_t> sz(ns), idx(ns, 0);
                    for (size_t s = 0; s < ns; ++s) sz[s] = seg_entries[s].size();
                    while (true) {
                        for (size_t s = 0; s < ns; ++s) wl_out << seg_entries[s][idx[s]];
                        wl_out << '\n';
                        int carry = static_cast<int>(ns) - 1;
                        while (carry >= 0) { if (++idx[carry] < sz[carry]) break; idx[carry--] = 0; }
                        if (carry < 0) break;
                    }
                    wl_out.close();
                    combined_wl_file = wl_path;
                    std::cerr << "[singlet] Combined WL: " << product
                              << " entries \xE2\x86\x92 " << wl_path << "\n";

                    // ── Barcode pre-scan: filter Cartesian WL to observed barcodes ──
                    // SPLiT-seq/BD Rhapsody: full 884K WL × 39K genes × 4B = 138 GB STAR RAM.
                    // Scan first min(5M, total) reads to discover which concatenated barcodes
                    // actually appear in the data; replace combined_wl_file with the filtered
                    // subset.  Reduces WL from 884K → ~1K-50K → STAR RAM fits in 128G nodes.
                    // The concatenation uses the SAME segment positions as the FIFO rewriter,
                    // so the filtered WL matches exactly what STAR receives.
                    if (!onefq_file.empty() && !complex_concat_segs_all.empty()
                            && concat_bc_total > 0) {
                        static constexpr char PRESCAN_N2A[5] = {'A','C','G','T','N'};
                        static constexpr uint64_t PRESCAN_LIMIT  = 5'000'000ULL;
                        static constexpr uint32_t MIN_COUNT_HI   = 5;   // ≥1M reads scanned
                        static constexpr uint32_t MIN_COUNT_LO   = 2;   // <1M reads scanned

                        // WL guard disabled: the per-segment WL files follow round numbering
                        // (rd1→rd2→rd3) while detected_cb_positions follow physical R1 order
                        // (rd3→rd2→rd1 for SPLiT-seq).  This segment-order mismatch causes
                        // the Cartesian product to contain wrong-order concatenations that
                        // never match the prescan's physical-order extraction.  The count
                        // threshold (≥5 for ≥1M reads) is sufficient noise filtering.
                        std::unordered_set<std::string> full_wl_set;
                        const bool use_wl_guard = false;
                        if (use_wl_guard) {
                            std::ifstream wl_in_ps(wl_path);
                            std::string ln_ps;
                            full_wl_set.reserve(static_cast<size_t>(product));
                            while (std::getline(wl_in_ps, ln_ps))
                                if (!ln_ps.empty()) full_wl_set.insert(std::move(ln_ps));
                        }

                        std::unordered_map<std::string, uint32_t> prescan_counts;
                        prescan_counts.reserve(std::min(product, uint64_t(200'000)));
                        uint64_t reads_scanned = 0;
                        bool prescan_ok = false;
                        try {
                            lib1fq::Reader ps_reader;
                            ps_reader.open(onefq_file.c_str());
                            ps_reader.set_verify_crc(false);

                            // BC dict encodes only segment-0 BC + UMI in the decoded R1;
                            // all other segment positions are zero-filled (decoded as 'A').
                            // Pre-scan would extract wrong data for segments 1..N-1 →
                            // pseudo-barcodes never accumulate to min_count → 0 kept.
                            // Skip pre-scan and retain full Cartesian WL for these files.
                            // (Encoder fix in protocol.h prevents new files from hitting this.)
                            if (ps_reader.has_bc_dict() &&
                                    complex_concat_segs_all.size() > 1) {
                                std::cerr << "[singlet] Prescan: BC dict encodes segment-0 only"
                                             " (" << ps_reader.bc_length() << "bp BC + "
                                          << ps_reader.umi_length() << "bp UMI); decoded R1 "
                                             "lacks segment 1.."
                                          << (complex_concat_segs_all.size() - 1)
                                          << " data — keeping full Cartesian WL\n"
                                             "[singlet] NOTE: re-download this sample to get "
                                             "full-R1 .1fq and enable pre-scan\n";
                                prescan_ok = true;  // not an error; no reads_scanned
                            } else {
                            lib1fq::DecodedBlock blk;
                            std::string bc_buf(static_cast<size_t>(concat_bc_total), 'N');
                            while (ps_reader.read_block(blk) &&
                                   reads_scanned < PRESCAN_LIMIT) {
                                for (uint32_t ri = 0; ri < blk.n_reads &&
                                                      reads_scanned < PRESCAN_LIMIT; ++ri) {
                                    const uint8_t* r1s  = blk.r1_seq(ri);
                                    const uint16_t r1len = blk.r1_len(ri);
                                    int wpos = 0;
                                    bool valid = true;
                                    for (const auto& seg : complex_concat_segs_all) {
                                        for (int j = 0; j < seg.len; ++j) {
                                            const int src = seg.start + j;
                                            if (src >= static_cast<int>(r1len)) {
                                                valid = false; break;
                                            }
                                            const uint8_t v = r1s[src];
                                            bc_buf[wpos++] = (v < 5) ? PRESCAN_N2A[v] : 'N';
                                        }
                                        if (!valid) break;
                                    }
                                    if (!valid) continue;
                                    ++prescan_counts[bc_buf];
                                    ++reads_scanned;
                                }
                            }
                            prescan_ok = true;
                            }  // end else (no BC dict / single-segment BC dict)
                        } catch (const std::exception& ps_ex) {
                            std::cerr << "[singlet] Prescan: reader error: "
                                      << ps_ex.what()
                                      << " — keeping full Cartesian WL\n";
                        }

                        if (prescan_ok && reads_scanned > 0) {
                            const uint32_t min_cnt =
                                (reads_scanned >= 1'000'000ULL) ? MIN_COUNT_HI : MIN_COUNT_LO;
                            std::string filt_path = wl_base
                                + "/singlet_combwl_filtered_"
                                + std::to_string(getpid()) + ".txt";
                            std::ofstream filt_out(filt_path);
                            uint64_t n_kept = 0;
                            if (filt_out) {
                                for (const auto& kv : prescan_counts) {
                                    if (kv.second < min_cnt) continue;
                                    if (use_wl_guard &&
                                        full_wl_set.count(kv.first) == 0) continue;
                                    filt_out << kv.first << '\n';
                                    ++n_kept;
                                }
                                filt_out.close();
                                if (n_kept > 0) {
                                    std::remove(wl_path.c_str());  // delete full Cartesian WL
                                    combined_wl_file = filt_path;
                                    std::cerr << "[singlet] Prescan: "
                                              << reads_scanned << " reads \xE2\x86\x92 "
                                              << n_kept << "/" << product
                                              << " barcodes (min_count=" << min_cnt
                                              << ") \xE2\x86\x92 " << filt_path << "\n";
                                } else {
                                    std::remove(filt_path.c_str());
                                    std::cerr << "[singlet] Prescan: 0 barcodes met"
                                                 " threshold — keeping full Cartesian WL\n";
                                }
                            } else {
                                std::cerr << "[singlet] Prescan: cannot write "
                                          << filt_path
                                          << " — keeping full Cartesian WL\n";
                            }
                        }
                    }  // end barcode pre-scan
                }
            } else {
                std::cerr << "[singlet] CB_UMI_Complex\xE2\x86\x92concat: "
                             "per-seg WL read failed, running without whitelist\n";
            }
        } else {
            std::cerr << "[singlet] CB_UMI_Complex\xE2\x86\x92concat: "
                         "no per-seg WLs found, running without whitelist\n";
        }
    }

    // ── FIFO mode: create named pipes + spawn Pass 2 writer thread ──
    // Must run AFTER Phase 0.9 (barcode discovery consumed onefq_bc_counts from Pass 1)
    // and BEFORE the STAR child fork (STAR opens FIFOs for reading; writer uses O_RDWR trick to
    // avoid deadlock — same approach as the SRA streaming path).
    if (use_fifo_decode && onefq_mode) {
        // Ensure decode directory exists (was set up in Pass 1 preamble)
        {
            std::string tmpdir_fifo = "/dev/shm";
            if (const char* td = getenv("TMPDIR")) tmpdir_fifo = td;
            mkdir_p(tmpdir_fifo + "/singlet_1fq_" + std::to_string(getpid()));
        }
        // Remove stale files / FIFOs at these paths
        unlink(fifo_r1_path.c_str());
        unlink(fifo_r2_path.c_str());
        if (mkfifo(fifo_r1_path.c_str(), 0600) != 0) {
            std::cerr << "[singlet] ERROR: mkfifo R1 FIFO failed: " << strerror(errno) << "\n";
            return 1;
        }
        if (mkfifo(fifo_r2_path.c_str(), 0600) != 0) {
            std::cerr << "[singlet] ERROR: mkfifo R2 FIFO failed: " << strerror(errno) << "\n";
            return 1;
        }
        std::cerr << "[1fq-fifo] FIFOs created: " << fifo_r1_path << ", " << fifo_r2_path << "\n";

        // Snapshot r2_clip5p for the thread, then reset to 0 so STAR --clip5pNbases
        // fallback does NOT fire (clipping is applied inline in the writer thread).
        const int  thr_clip    = r2_clip5p;
        r2_clip5p = 0;
        // All other captures are by value for thread safety
        const std::string thr_onefq   = onefq_file;
        const int         thr_threads = star_threads;
        const bool        thr_atac    = is_atac_mode;
        const std::string thr_r1      = fifo_r1_path;
        const std::string thr_r2      = fifo_r2_path;
        // Concat captures for CB_UMI_Complex→CB_samTagOut R1 rewriting
        const bool thr_use_concat        = use_complex_concat;
        const auto thr_concat_segs       = complex_concat_segs_all;
        const int  thr_concat_umi_start  = concat_umi_start_0;
        const int  thr_concat_umi_len    = concat_umi_len_c;

        fifo_writer_thread = std::thread(
            [thr_onefq, thr_clip, thr_threads, thr_atac, thr_r1, thr_r2,
             thr_use_concat, thr_concat_segs, thr_concat_umi_start, thr_concat_umi_len]() {
            signal(SIGPIPE, SIG_IGN);  // prevent SIGPIPE if STAR exits before writer finishes

            // Open both FIFOs with O_RDWR (non-blocking — avoids deadlock with STAR opener)
            int r1_fd = open(thr_r1.c_str(), O_RDWR);
            int r2_fd = open(thr_r2.c_str(), O_RDWR);
            if (r1_fd < 0 || r2_fd < 0) {
                std::cerr << "[1fq-fifo] ERROR: open FIFOs: " << strerror(errno) << "\n";
                if (r1_fd >= 0) close(r1_fd);
                if (r2_fd >= 0) close(r2_fd);
                return;
            }
#ifdef __linux__
            fcntl(r1_fd, F_SETPIPE_SZ, 1 << 20);
            fcntl(r2_fd, F_SETPIPE_SZ, 1 << 20);
#endif
            FILE* r1_fp = fdopen(r1_fd, "w");
            FILE* r2_fp = fdopen(r2_fd, "w");
            if (!r1_fp || !r2_fp) {
                std::cerr << "[1fq-fifo] ERROR: fdopen failed\n";
                if (!r1_fp && r1_fd >= 0) close(r1_fd);
                if (!r2_fp && r2_fd >= 0) close(r2_fd);
                return;
            }
            std::vector<char> r1_wbuf(1 << 16), r2_wbuf(1 << 16);  // 64 KB: low latency + reasonable syscall cost
            setvbuf(r1_fp, r1_wbuf.data(), _IOFBF, r1_wbuf.size());
            setvbuf(r2_fp, r2_wbuf.data(), _IOFBF, r2_wbuf.size());

            lib1fq::Reader reader;
            try {
                reader.open(thr_onefq.c_str());
            } catch (const std::exception& e) {
                std::cerr << "[1fq-fifo] ERROR: Cannot open .1fq: " << e.what() << "\n";
                fclose(r1_fp); fclose(r2_fp);
                return;
            }
            reader.set_verify_crc(false);
            auto hdr2 = reader.header();
            const uint32_t total_blocks = reader.block_count();
            const int      fd2   = reader.file_descriptor();
            const auto     codec = static_cast<lib1fq::Codec>(hdr2.codec);
            const int      n_thr = std::min(thr_threads, static_cast<int>(total_blocks));
            const bool  has_bc_dict2 = reader.has_bc_dict();
            const uint32_t bc_dict_n2 = reader.bc_dict_size();

            // Guard: BC dict + multi-segment concat → decoded R1 only has segment-0 BC + UMI;
            // all other segment positions are zero → wrong barcodes for all reads.
            // Disable concat mode so STAR falls back to CB_UMI_Complex with the full WL.
            // This sample needs re-downloading with the fixed encoder (no BC dict for
            // multi-segment protocols) to produce correct concat barcodes.
            const bool fifo_use_concat2 = thr_use_concat &&
                !(has_bc_dict2 && thr_concat_segs.size() > 1);
            if (thr_use_concat && has_bc_dict2 && thr_concat_segs.size() > 1) {
                std::cerr << "[1fq-fifo] WARNING: BC dict .1fq with "
                          << thr_concat_segs.size() << " concat segments — "
                             "decoded R1 lacks segment 1.."
                          << (thr_concat_segs.size() - 1) << " data; "
                             "disabling concat rewrite; re-download to fix\n";
            }
            bool has_i2_stream2 = thr_atac &&
                                  (hdr2.reserved[0] & lib1fq::ExtraStreams::HAS_I2) != 0;
            std::vector<uint8_t> atac_bc_dict2;
            uint16_t atac_bc_len2 = 0;
            if (has_i2_stream2) {
                atac_bc_len2 = hdr2.stream_lengths[2];
            } else if (thr_atac && has_bc_dict2 && bc_dict_n2 > 0) {
                atac_bc_dict2 = reader.bc_dict_flat();
                atac_bc_len2  = reader.bc_length();
            }

            static constexpr char N2A2[5] = {'A','C','G','T','N'};

            // Pre-compute cumulative read offsets (for read naming)
            std::vector<uint64_t> block_read_offsets(total_blocks);
            {
                uint64_t cum = 0;
                for (uint32_t b = 0; b < total_blocks; ++b) {
                    block_read_offsets[b] = cum;
                    lib1fq::BlockHeader bh;
                    ::pread(fd2, &bh, sizeof(bh), static_cast<off_t>(reader.block_offset(b)));
                    cum += bh.n_reads;
                }
            }

            struct FmtBlock2 {
                std::vector<char> r1, r2;
                size_t r1_len = 0, r2_len = 0;
                uint32_t n_reads = 0;
                std::vector<uint32_t> bc_indices;
                std::vector<uint32_t> r1_rec_off;  // start byte of each rep's R1 record in r1
                std::vector<uint32_t> r2_rec_off;  // start byte of each rep's R2 record in r2
            };

            // Pass 2 process_block: same FASTQ formatting as main decode,
            // without BC counting and with inline r2_clip5p application.
            auto process_block2 = [&](uint32_t bi, uint64_t read_off) -> FmtBlock2 {
                FmtBlock2 fb;
                uint64_t file_off  = reader.block_offset(bi);
                uint32_t comp_size = reader.block_comp_size(bi);
                lib1fq::BlockHeader bh;
                ::pread(fd2, &bh, sizeof(bh), static_cast<off_t>(file_off));
                std::vector<uint8_t> comp_buf(comp_size);
                ::pread(fd2, comp_buf.data(), comp_size,
                        static_cast<off_t>(file_off + sizeof(bh)));
                size_t est_raw = static_cast<size_t>(bh.n_reads) * 200;
                std::vector<uint8_t> raw_buf(est_raw);
                size_t raw_size;
                try {
                    raw_size = lib1fq::compress::decompress_block(
                        codec, comp_buf.data(), comp_size,
                        raw_buf.data(), raw_buf.size());
                } catch (...) {
                    raw_buf.resize(est_raw * 4);
                    raw_size = lib1fq::compress::decompress_block(
                        codec, comp_buf.data(), comp_size,
                        raw_buf.data(), raw_buf.size());
                }
                lib1fq::DecodedBlock blk;
                reader.decode_payload(raw_buf.data(), raw_size, bh, blk);

                fb.n_reads = blk.n_reads;
                if (has_bc_dict2) fb.bc_indices = std::move(blk.bc_indices);
                const bool has_dups2 = !blk.dup_counts.empty();

                // FIX-FIFO-OOM: use per-read exact sum instead of max_element × total_reps.
                // If any single read has an anomalously large R2 (e.g. a corrupted .1fq
                // with r2_length ≈ 36,000 bp), the old max-based formula allocated
                // total_reps × r2_max × 2 bytes which exploded to ~576 GB for 97-block
                // sci-RNA-seq3 files. Per-read sum is strictly correct and bounded.
                const int r2_clip = (thr_clip > 0) ? thr_clip : 0;
                size_t r1_buf_bytes = 0;
                size_t r2_buf_bytes = 0;
                uint64_t total_reps = 0;
                for (uint32_t i = 0; i < blk.n_reads; ++i) {
                    uint16_t r1len = blk.r1_lengths.empty() ? 28 : blk.r1_len(i);
                    uint16_t r2len = blk.r2_lengths.empty() ? 91 : blk.r2_len(i);
                    // SIZE guard: skip reads that will also be skipped in the WRITE section.
                    // Keeps SIZE and WRITE sections consistent, preventing any OOM from
                    // anomalously large r2_lengths decoded from a corrupted .1fq block.
                    if (r1len == 0 || r2len == 0 || r2len > 1024) continue;
                    uint16_t r2len2 = (r2_clip > 0 && r2_clip < r2len)
                                      ? static_cast<uint16_t>(r2len - r2_clip) : r2len;
                    // For concat rewrite, R1 is always thr_concat_segs total len + umi
                    uint16_t r1_write_len = r1len;
                    if (fifo_use_concat2 && !thr_concat_segs.empty()) {
                        r1_write_len = 0;
                        for (const auto& seg : thr_concat_segs) r1_write_len += static_cast<uint16_t>(seg.len);
                        r1_write_len += static_cast<uint16_t>(thr_concat_umi_len);
                    }
                    const uint32_t rpt = has_dups2 ? blk.dup_counts[i] : 1;
                    total_reps += rpt;
                    // Each FASTQ record: @name\nseq\n+\nqual\n
                    // name ≈ 20 bytes for "@rNNNNNNNNN\n", overhead +20 per record
                    r1_buf_bytes += static_cast<size_t>(rpt) *
                        (static_cast<size_t>(r1_write_len) * 2 + (thr_atac ? 55 : 30) + 4);
                    r2_buf_bytes += static_cast<size_t>(rpt) *
                        (static_cast<size_t>(r2len2) * 2 + (thr_atac ? 55 : 30) + 14);
                }
                fb.r1.resize(r1_buf_bytes + 64);   // +64 for trailing sentinel
                fb.r2.resize(r2_buf_bytes + 64);
                fb.r1_rec_off.reserve(static_cast<size_t>(total_reps));
                fb.r2_rec_off.reserve(static_cast<size_t>(total_reps));
                char* r1p = fb.r1.data();
                char* r2p = fb.r2.data();
                bool has_qual = !blk.r2_qual.empty();
                uint64_t roff = read_off;

                for (uint32_t i = 0; i < blk.n_reads; ++i) {
                    const uint32_t repeat = has_dups2 ? blk.dup_counts[i] : 1;
                    uint16_t r1len = blk.r1_len(i);
                    uint16_t r2len = blk.r2_len(i);
                    if (r1len == 0 || r2len == 0) continue;
                    // Guard: skip reads with anomalously large R2 (corrupted .1fq).
                    // Real Illumina R2 is ≤600bp; anything >1024bp is data corruption.
                    // The matching SIZE section conservatively allocated for these reads,
                    // but passing them to STAR causes a FASTQ-parse fatal error.
                    if (r2len > 1024) {
                        roff += repeat;
                        continue;
                    }
                    const uint8_t* r1s = blk.r1_seq(i);
                    const uint8_t* r2s = blk.r2_seq(i);
                    const uint8_t* q   = has_qual ? blk.r2_quality(i) : nullptr;

                    // Inline 5' clip of R2 (cDNA read)
                    uint16_t r2_skip = (thr_clip > 0 &&
                                        static_cast<uint16_t>(thr_clip) < r2len)
                                       ? static_cast<uint16_t>(thr_clip) : 0;

                    for (uint32_t rep = 0; rep < repeat; ++rep) {
                        // ATAC QNAME prefix (same logic as main decode)
                        char atac_prefix[32] = {};
                        if (thr_atac) {
                            bool bc_set = false;
                            if (has_i2_stream2 && !blk.i2_data.empty() &&
                                i < blk.i2_lengths.size() && blk.i2_lengths[i] > 0) {
                                const uint8_t* bcp = blk.i2_seq(i);
                                uint16_t blen = blk.i2_len(i);
                                int np = 0;
                                atac_prefix[np++]='B'; atac_prefix[np++]='C'; atac_prefix[np++]=':';
                                for (uint16_t j = 0; j < blen; ++j)
                                    atac_prefix[np++] = N2A2[bcp[j]];
                                atac_prefix[np++]='_'; atac_prefix[np]='\0';
                                bc_set = true;
                            }
                            if (!bc_set) {
                                uint32_t bc_idx = (!fb.bc_indices.empty() &&
                                                   i < fb.bc_indices.size())
                                                  ? fb.bc_indices[i] : UINT32_MAX;
                                if (!atac_bc_dict2.empty() && atac_bc_len2 > 0 &&
                                    bc_idx < bc_dict_n2) {
                                    const uint8_t* bcp = atac_bc_dict2.data() +
                                        static_cast<size_t>(bc_idx) * atac_bc_len2;
                                    int np = 0;
                                    atac_prefix[np++]='B'; atac_prefix[np++]='C'; atac_prefix[np++]=':';
                                    for (uint16_t j = 0; j < atac_bc_len2; ++j)
                                        atac_prefix[np++] = N2A2[bcp[j]];
                                    atac_prefix[np++]='_'; atac_prefix[np]='\0';
                                    bc_set = true;
                                }
                            }
                            if (!bc_set) sprintf(atac_prefix, "BC:NNNNNNNNNNNNNNNN_");
                        }

                        // R1 record
                        fb.r1_rec_off.push_back(static_cast<uint32_t>(r1p - fb.r1.data()));
                        int n = thr_atac
                            ? sprintf(r1p, "@%sr%lu\n", atac_prefix, (unsigned long)roff)
                            : sprintf(r1p, "@r%lu\n", (unsigned long)roff);
                        char* s1 = r1p + n;
                        if (fifo_use_concat2 && !thr_concat_segs.empty()) {
                            // Rewrite R1 as [seg0]...[segN][UMI] for CB_samTagOut concat
                            int wpos = 0;
                            for (const auto& seg : thr_concat_segs) {
                                for (int j = 0; j < seg.len; ++j) {
                                    int src = seg.start + j;
                                    s1[wpos++] = (src < (int)r1len) ? N2A2[r1s[src]] : 'N';
                                }
                            }
                            for (int j = 0; j < thr_concat_umi_len; ++j) {
                                int src = thr_concat_umi_start + j;
                                s1[wpos++] = (src < (int)r1len) ? N2A2[r1s[src]] : 'N';
                            }
                            s1 += wpos;
                            *s1++ = '\n'; *s1++ = '+'; *s1++ = '\n';
                            memset(s1, 'F', wpos); s1 += wpos; *s1++ = '\n';
                        } else {
                            for (uint16_t j = 0; j < r1len; ++j) s1[j] = N2A2[r1s[j]];
                            s1 += r1len;
                            *s1++ = '\n'; *s1++ = '+'; *s1++ = '\n';
                            memset(s1, 'F', r1len); s1 += r1len; *s1++ = '\n';
                        }
                        r1p = s1;

                        // R2 record (with inline 5' clip)
                        fb.r2_rec_off.push_back(static_cast<uint32_t>(r2p - fb.r2.data()));
                        n = thr_atac
                            ? sprintf(r2p, "@%sr%lu\n", atac_prefix, (unsigned long)roff)
                            : sprintf(r2p, "@r%lu\n", (unsigned long)roff);
                        char* s2 = r2p + n;
                        const uint8_t* r2s2 = r2s + r2_skip;
                        const uint8_t* q2   = q ? q + r2_skip : nullptr;
                        uint16_t r2len2 = r2len - r2_skip;
                        for (uint16_t j = 0; j < r2len2; ++j) s2[j] = N2A2[r2s2[j]];
                        s2 += r2len2;
                        const uint16_t r2_seq_len2 = static_cast<uint16_t>(r2len2);
                        *s2++ = '\n'; *s2++ = '+'; *s2++ = '\n';
                        if (q2) {
                            for (uint16_t j = 0; j < r2_seq_len2; ++j)
                                s2[j] = static_cast<char>(q2[j] + 33);
                        } else {
                            memset(s2, 'F', r2_seq_len2);
                        }
                        s2 += r2_seq_len2; *s2++ = '\n';
                        r2p = s2;
                        roff++;
                    }
                }

                fb.r1_len = static_cast<size_t>(r1p - fb.r1.data());
                fb.r2_len = static_cast<size_t>(r2p - fb.r2.data());
                return fb;
            };  // end process_block2

            // Write loop: interleaved per-record writes.
            // ROOT CAUSE of previous deadlock: old code wrote _all_ R1 data for a block
            // in one fwrite(), then all R2.  For large blocks (>13K reads), the R1
            // fwrite() blocked when the R1 pipe filled while STAR was waiting for R2
            // data that hadn't been written yet — classic 3-way FIFO deadlock.
            //
            // FIX: write R1_i then R2_i for every read i before moving to read i+1.
            // STAR always has a matching R2 record available whenever the R1 pipe is
            // approaching capacity, so it can drain both pipes and never stalls.
            // Neither pipe accumulates more than one 64-KB buffer flush of data ahead
            // of the other, eliminating the blocking asymmetry.
            const uint32_t batch_size = static_cast<uint32_t>(n_thr);
            uint64_t reads_written = 0;
            bool write_ok = true;

            for (uint32_t base = 0; base < total_blocks && write_ok; base += batch_size) {
                uint32_t batch_end = std::min(base + batch_size, total_blocks);
                uint32_t n_batch = batch_end - base;
                std::vector<std::future<FmtBlock2>> futures;
                futures.reserve(n_batch);
                for (uint32_t b = base; b < batch_end; ++b)
                    futures.push_back(std::async(std::launch::async,
                                      process_block2, b, block_read_offsets[b]));
                for (uint32_t i = 0; i < n_batch && write_ok; ++i) {
                    FmtBlock2 fb = futures[i].get();

                    // Sentinel: ensures r1_rec_off[ri+1] is always valid.
                    fb.r1_rec_off.push_back(static_cast<uint32_t>(fb.r1_len));
                    fb.r2_rec_off.push_back(static_cast<uint32_t>(fb.r2_len));

                    const char*    r1d    = fb.r1.data();
                    const char*    r2d    = fb.r2.data();
                    const uint32_t n_recs = static_cast<uint32_t>(fb.r1_rec_off.size()) - 1;

                    // Interleaved per-record writes: R1_i immediately followed by R2_i.
                    for (uint32_t ri = 0; ri < n_recs && write_ok; ++ri) {
                        const size_t r1sz = fb.r1_rec_off[ri+1] - fb.r1_rec_off[ri];
                        const size_t r2sz = fb.r2_rec_off[ri+1] - fb.r2_rec_off[ri];
                        if (fwrite(r1d + fb.r1_rec_off[ri], 1, r1sz, r1_fp) != r1sz ||
                            fwrite(r2d + fb.r2_rec_off[ri], 1, r2sz, r2_fp) != r2sz)
                            write_ok = false;
                    }
                    // Flush both FIFOs at block boundary so STAR sees all records.
                    if (write_ok &&
                        (fflush(r1_fp) != 0 || fflush(r2_fp) != 0))
                        write_ok = false;
                    reads_written += fb.n_reads;
                }
            }

            fclose(r1_fp);
            fclose(r2_fp);
            std::cerr << "[1fq-fifo] Pass 2 done: " << reads_written << " reads\n";
        });  // end fifo_writer_thread lambda

        std::cerr << "[1fq-fifo] FIFO writer thread started\n";
    }


    // ── Phase 1: Set up BAM source ──
    // Decide whether to use parallel pileup (sorted BAM to /dev/shm + indexed).
    // Auto mode: enable when /dev/shm has >=10 GB free and we're running STAR.
    if (parallel_pileup == -1 && bam_file.empty()) {
        struct statvfs svfs;
        if (statvfs("/dev/shm", &svfs) == 0) {
            uint64_t free_bytes = (uint64_t)svfs.f_bavail * svfs.f_frsize;
            parallel_pileup = (free_bytes >= (uint64_t)10 * 1024 * 1024 * 1024) ? 1 : 0;
            std::cerr << "[singlet] /dev/shm free: " << free_bytes / (1024*1024*1024)
                      << " GB → parallel_pileup=" << parallel_pileup << "\n";
        } else {
            parallel_pileup = 0;
        }
    } else if (parallel_pileup == -1) {
        parallel_pileup = 0;  // existing BAM: caller decides
    }

    // ATAC requires sorted BAM (ATACFragmentExtractor reads coordinate-sorted records).
    if (is_atac_mode && parallel_pileup != 1) {
        parallel_pileup = 1;
        std::cerr << "[singlet] ATAC mode: forcing sorted BAM (parallel_pileup=1)\n";
    }
    // Smart-seq2 / Bulk RNA: counting reads from sorted BAM (no pileup engine needed).
    if ((is_smart_seq2_mode || is_bulk_rna_mode) && parallel_pileup != 1) {
        parallel_pileup = 1;
        std::cerr << "[singlet] " << (is_bulk_rna_mode ? "Bulk RNA" : "Smart-seq2")
                  << " mode: forcing sorted BAM (parallel_pileup=1)\n";
    }

    std::string parallel_bam_path; // set when parallel_pileup=1
    std::string parallel_bai_path;

    std::string bam_source;
    pid_t star_pid = -1;
    int pipe_fd[2] = {-1, -1};

    if (!bam_file.empty()) {
        // Mode A: existing BAM
        bam_source = bam_file;
        std::cerr << "[singlet] Reading from BAM: " << bam_file << "\n";
    } else {
        // Mode B: fork STAR — either write sorted BAM to file (parallel) or pipe (streaming)
        if (parallel_pileup != 1) {
            if (pipe(pipe_fd) != 0) {
                std::cerr << "ERROR: pipe() failed: " << strerror(errno) << "\n";
                return 1;
            }
            // Increase pipe buffer for throughput (Linux)
#ifdef __linux__
            fcntl(pipe_fd[0], F_SETPIPE_SZ, 1 << 20);
#endif
        }

        // Raise soft file-descriptor limit for STAR BAMsort temp files.
        // Default ulimit -n (1024) is too low; STAR opens >1024 temp files during BAMsort.
        {
            struct rlimit rl{};
            if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < 65536) {
                rl.rlim_cur = std::min(static_cast<rlim_t>(65536), rl.rlim_max);
                setrlimit(RLIMIT_NOFILE, &rl);  // inherited by fork'd STAR child
            }
        }
        star_pid = fork();
        if (star_pid < 0) {
            std::cerr << "ERROR: fork() failed: " << strerror(errno) << "\n";
            return 1;
        }

        if (star_pid == 0) {
            // ── CHILD: call bundled STAR directly (no exec) ──
            if (parallel_pileup != 1) {
                // Streaming path: redirect stdout to write end of pipe
                close(pipe_fd[0]);
                if (pipe_fd[1] != STDOUT_FILENO) {
                    dup2(pipe_fd[1], STDOUT_FILENO);
                    close(pipe_fd[1]);
                }
            }
            // Parallel path: no pipe redirect; STAR writes sorted BAM to file

            // ── Close inherited write-side FIFO fds (prevents EOF hang) ──
            // The FIFO writer thread in the parent opens the decode FIFOs with
            // O_RDWR BEFORE this fork.  The child inherits those fds; their
            // presence keeps the write-side open count > 0 even after the parent
            // closes them, so STAR's O_RDONLY reader thread never receives EOF
            // and hangs indefinitely on pipe_read.
            // Fix: find all inherited fds pointing to the two FIFO inodes and
            // close them in the child.  Uses fstat() (no dirent / readlink needed).
            if (use_fifo_decode && !fifo_r1_path.empty()) {
                struct stat st1{}, st2{};
                if (::stat(fifo_r1_path.c_str(), &st1) == 0 &&
                    ::stat(fifo_r2_path.c_str(), &st2) == 0) {
                    int max_fd = static_cast<int>(::sysconf(_SC_OPEN_MAX));
                    if (max_fd < 0 || max_fd > 4096) max_fd = 4096;
                    for (int fd = 3; fd < max_fd; ++fd) {
                        struct stat st{};
                        if (::fstat(fd, &st) != 0) continue;
                        if ((st.st_dev == st1.st_dev && st.st_ino == st1.st_ino) ||
                            (st.st_dev == st2.st_dev && st.st_ino == st2.st_ino))
                            ::close(fd);
                    }
                }
            }

            // Build STAR args vector — identical content to the previous
            // execv call; we now pass it to star_main_impl() directly.
            std::vector<const char*> args;
            args.push_back("STAR");  // argv[0] placeholder (STAR ignores it)
            args.push_back("--runMode");       args.push_back("alignReads");
            args.push_back("--genomeDir");     args.push_back(genome_dir.c_str());
            args.push_back("--readFilesIn");
            if (is_atac_mode) {
                // ATAC PE-DNA: R1.fastq = cDNA_left (R1 genomic, 50bp),
                //              R2.fastq = cDNA_right (R3 genomic, 49bp stored as r2 in .1fq).
                // Pass in natural R1→R3 order.  Barcode is embedded in QNAME, NOT sent to STAR.
                args.push_back(reads_r1.c_str());  // cDNA left  (R1 genomic)
                args.push_back(reads_r2.c_str());  // cDNA right (R3 genomic)
            } else {
                // scRNA / CITE-seq / etc: R2.fastq = cDNA, R1.fastq = barcode.
                // STARsolo requires --readFilesIn cDNA barcode (barcode read last).
                args.push_back(reads_r2.c_str());
                args.push_back(reads_r1.c_str());
            }

            // For gzipped FASTQ files, use zcat via --readFilesCommand.
            // For SRA FIFOs, use 'cat' via --readFilesCommand.
            // For .1fq temp files (uncompressed), STAR reads directly (no pipe overhead).
            if (sra_mode) {
                args.push_back("--readFilesCommand");
                args.push_back("cat");
            } else if (!onefq_mode) {
                // Legacy --reads path: gzipped FASTQs
                args.push_back("--readFilesCommand");
                args.push_back("zcat");
            }
            // onefq_mode: no --readFilesCommand → STAR reads uncompressed files directly

            // ── ATAC mode: PE-DNA alignment — no soloType, splice-disabled ──
            if (is_atac_mode) {
                std::string threads_s_atac = std::to_string(star_threads);
                std::string star_prefix_atac = out_prefix + "/star_";
                std::string bamsort_ram_s_atac;

                // Disable splicing for DNA-mode alignment
                args.push_back("--alignIntronMax");      args.push_back("1");
                args.push_back("--alignMatesGapMax");    args.push_back("2000");
                args.push_back("--alignSJDBoverhangMin"); args.push_back("999");

                // Sorted BAM output (required by ATACFragmentExtractor)
                args.push_back("--outSAMtype");    args.push_back("BAM"); args.push_back("SortedByCoordinate");
                args.push_back("--outBAMcompression"); args.push_back("0");

                // limitBAMsortRAM
                if (genome_load != "NoSharedMemory") {
                    uint64_t bam_sort_ram = 30ULL * 1024 * 1024 * 1024;
                    if (!genome_dir.empty()) {
                        std::string sa_path_atac = genome_dir + "/SA";
                        struct stat sa_st_atac{};
                        if (stat(sa_path_atac.c_str(), &sa_st_atac) == 0 && sa_st_atac.st_size > 0)
                            bam_sort_ram = static_cast<uint64_t>(sa_st_atac.st_size) * 2
                                           + 4ULL * 1024 * 1024 * 1024;
                    }
                    bamsort_ram_s_atac = std::to_string(bam_sort_ram);
                    args.push_back("--limitBAMsortRAM"); args.push_back(bamsort_ram_s_atac.c_str());
                }

                // Minimal SAM attributes — no CB/UR tags (barcode is in QNAME)
                args.push_back("--outSAMattributes"); args.push_back("NH"); args.push_back("AS");
                args.push_back("--outSAMunmapped");   args.push_back("None");

                args.push_back("--runThreadN");  args.push_back(threads_s_atac.c_str());
                args.push_back("--genomeLoad");  args.push_back(genome_load.c_str());
                args.push_back("--outFileNamePrefix"); args.push_back(star_prefix_atac.c_str());

                for (const auto& a : star_extra) args.push_back(a.c_str());
                args.push_back(nullptr);

                int rc = star_main_impl(
                    static_cast<int>(args.size()) - 1,
                    const_cast<char**>(args.data())
                );
                _exit(rc);
            }

            // ── Smart-seq2 mode: standard PE alignment — no soloType, no barcode ──
            if (is_smart_seq2_mode) {
                std::string threads_s_ss2 = std::to_string(star_threads);
                std::string star_prefix_ss2 = out_prefix + "/star_";
                std::string bamsort_ram_ss2;

                // Standard spliced PE alignment (all STAR defaults for RNA-seq)
                args.push_back("--outSAMtype");    args.push_back("BAM"); args.push_back("SortedByCoordinate");
                args.push_back("--outBAMcompression"); args.push_back("0");

                // limitBAMsortRAM when using shared genome memory
                if (genome_load != "NoSharedMemory") {
                    uint64_t bam_sort_ram = 30ULL * 1024 * 1024 * 1024;
                    if (!genome_dir.empty()) {
                        std::string sa_path_ss2 = genome_dir + "/SA";
                        struct stat sa_st_ss2{};
                        if (stat(sa_path_ss2.c_str(), &sa_st_ss2) == 0 && sa_st_ss2.st_size > 0)
                            bam_sort_ram = static_cast<uint64_t>(sa_st_ss2.st_size) * 2
                                           + 4ULL * 1024 * 1024 * 1024;
                    }
                    bamsort_ram_ss2 = std::to_string(bam_sort_ram);
                    args.push_back("--limitBAMsortRAM"); args.push_back(bamsort_ram_ss2.c_str());
                }

                // Minimal SAM attributes — no CB/UR (no barcode in Smart-seq2)
                args.push_back("--outSAMattributes"); args.push_back("NH"); args.push_back("AS");
                args.push_back("--outSAMunmapped");   args.push_back("None");

                args.push_back("--runThreadN");  args.push_back(threads_s_ss2.c_str());
                args.push_back("--genomeLoad");  args.push_back(genome_load.c_str());
                args.push_back("--outFileNamePrefix"); args.push_back(star_prefix_ss2.c_str());

                for (const auto& a : star_extra) args.push_back(a.c_str());
                args.push_back(nullptr);

                int rc = star_main_impl(
                    static_cast<int>(args.size()) - 1,
                    const_cast<char**>(args.data())
                );
                _exit(rc);
            }

            // ── Bulk RNA mode: standard PE or SE alignment — no soloType, no barcode ──
            if (is_bulk_rna_mode) {
                std::string threads_s_bulk = std::to_string(star_threads);
                std::string star_prefix_bulk = out_prefix + "/star_";
                std::string bamsort_ram_bulk;

                // Standard spliced PE/SE alignment
                args.push_back("--outSAMtype");    args.push_back("BAM"); args.push_back("SortedByCoordinate");
                args.push_back("--outBAMcompression"); args.push_back("0");

                // limitBAMsortRAM when using shared genome memory
                if (genome_load != "NoSharedMemory") {
                    uint64_t bam_sort_ram = 30ULL * 1024 * 1024 * 1024;
                    if (!genome_dir.empty()) {
                        std::string sa_path_bulk = genome_dir + "/SA";
                        struct stat sa_st_bulk{};
                        if (stat(sa_path_bulk.c_str(), &sa_st_bulk) == 0 && sa_st_bulk.st_size > 0)
                            bam_sort_ram = static_cast<uint64_t>(sa_st_bulk.st_size) * 2
                                           + 4ULL * 1024 * 1024 * 1024;
                    }
                    bamsort_ram_bulk = std::to_string(bam_sort_ram);
                    args.push_back("--limitBAMsortRAM"); args.push_back(bamsort_ram_bulk.c_str());
                }

                // Include XS SAM attribute so strandedness auto-detect can use splice-junction strand
                args.push_back("--outSAMattributes"); args.push_back("NH"); args.push_back("AS"); args.push_back("XS");
                args.push_back("--outSAMunmapped");   args.push_back("None");

                args.push_back("--runThreadN");  args.push_back(threads_s_bulk.c_str());
                args.push_back("--genomeLoad");  args.push_back(genome_load.c_str());
                args.push_back("--outFileNamePrefix"); args.push_back(star_prefix_bulk.c_str());

                // SE mode: if reads_r2 is empty (single-end), fix up readFilesIn to only list R1.
                // The readFilesIn was already pushed above as: args[readFilesIn_idx] = R2, R1.
                // For SE, R1 holds the cDNA read and R2 will be empty; remove the trailing entry.
                // Rewrite readFilesIn: find and patch.
                // NOTE: STAR gets --readFilesIn cDNA [barcode]. For bulk SE, cDNA = reads_r1.
                // For bulk PE, cDNA is both R1 and R2 (symmetric — use reads_r2 and reads_r1 as-is).
                bool bulk_se = reads_r2.empty() || reads_r2 == reads_r1;
                if (bulk_se) {
                    // Find the readFilesIn argument index and trim to single file.
                    // args already has reads_r2 and reads_r1 pushed; remove reads_r1 (second file).
                    // Locate the two consecutive paths after --readFilesIn.
                    for (int ai = 1; ai + 1 < (int)args.size(); ++ai) {
                        if (args[ai] && std::string(args[ai]) == "--readFilesIn") {
                            // args[ai+1]=reads_r2, args[ai+2]=reads_r1 — erase reads_r1 slot
                            if (ai + 2 < (int)args.size()) {
                                args.erase(args.begin() + ai + 2);
                            }
                            break;
                        }
                    }
                    std::cerr << "[singlet] Bulk RNA: single-end mode\n";
                } else {
                    std::cerr << "[singlet] Bulk RNA: paired-end mode\n";
                }

                for (const auto& a : star_extra) args.push_back(a.c_str());
                args.push_back(nullptr);

                int rc = star_main_impl(
                    static_cast<int>(args.size()) - 1,
                    const_cast<char**>(args.data())
                );
                _exit(rc);
            }
            bool no_whitelist = (whitelist_file == "None" || whitelist_file == "none");

            // ── Resolve binary directory for CB_UMI_Complex whitelist lookup ──
            // Used by Priority 3 (disk-based per-segment whitelists) below.
            std::string star_bin_dir;
            {
                char exe_buf[PATH_MAX]{};
                ssize_t exe_len = ::readlink("/proc/self/exe", exe_buf, sizeof(exe_buf)-1);
                if (exe_len > 0) {
                    exe_buf[exe_len] = '\0';
                    star_bin_dir = exe_buf;
                    auto sl = star_bin_dir.rfind('/');
                    if (sl != std::string::npos) star_bin_dir.resize(sl);
                }
            }
            // Protocol tag for the current .1fq (used by resolve_per_seg_whitelists).
            // onefq_protocol_id is set from hdr.protocol_id (with any metadata override)
            // before the if(onefq_mode) block closes.
            std::string complex_protocol_tag;
            for (const auto& sp : lib1fq::known_protocols()) {
                if (sp.protocol_id == onefq_protocol_id) {
                    complex_protocol_tag = sp.tag;
                    break;
                }
            }
            // If metadata-json overrode the protocol, use that tag instead.
            if (!meta_json_proto_tag.empty()) complex_protocol_tag = meta_json_proto_tag;

            // For whitelist-free CB_UMI_Complex protocols, Phase 0.9 auto-discovery may have
            // populated barcode_file with discovered barcodes. Use that file as the effective
            // STAR whitelist instead of the literal string "None" — STAR CB_UMI_Complex
            // attempts to open "None" as a file path and exits 109.
            const bool has_auto_barcodes = (no_whitelist
                && !barcode_file.empty()
                && barcode_file != "None" && barcode_file != "none"
                && barcode_file != "SMARTSEQ2_NONE");
            // effective_whitelist_for_complex: resolves to auto-discovered barcodes when
            // available, otherwise whitelist_file (which may be "None" for simple protocols).
            const std::string effective_whitelist_for_complex =
                has_auto_barcodes ? barcode_file : whitelist_file;
            bool use_complex = (detected_soloType == "CB_UMI_Complex" && !detected_cb_positions.empty());
            bool umi_len_set = false; // track if --soloUMIlen was already pushed
            bool solo_type_is_sam_tag_out = false; // track CB_samTagOut selection for soloCBmatchWLtype
            std::string cbl_s, umi_start_s, cb_start_s, umi_s; // must outlive args vector

            if (use_complex) {
                // CB_UMI_Complex: use native STAR CB_UMI_Complex mode for all multi-segment
                // protocols (BD Rhapsody, SPLiT-seq, inDrop, microwell-seq, SureCell, ddSEQ).
                // The pipeline produces SortedByCoordinate BAM, so CB_UMI_Complex is fully
                // supported. Single-segment protocols fall back to CB_samTagOut for efficiency.
                if (detected_cb_positions.size() == 1) {
                    // Single CB segment: use CB_samTagOut (faster for simple geometry)
                    auto parse_pos = [](const std::string& p) -> std::pair<int,int> {
                        // Format: "readIndex_startPos_readIndex2_endPos"
                        int vals[4] = {0,0,0,0};
                        size_t pos = 0; int vi = 0;
                        for (size_t i = 0; i <= p.size() && vi < 4; ++i) {
                            if (i == p.size() || p[i] == '_') {
                                vals[vi++] = std::atoi(p.substr(pos, i - pos).c_str());
                                pos = i + 1;
                            }
                        }
                        return {vals[1], vals[3]}; // startPos, endPos (0-based)
                    };
                    auto [cb_s, cb_e] = parse_pos(detected_cb_positions[0]);
                    auto [umi_s_pos, umi_e_pos] = parse_pos(detected_umi_position);
                    int cb_len_calc = cb_e - cb_s + 1;
                    int umi_len_calc = umi_e_pos - umi_s_pos + 1;

                    std::cerr << "[singlet] CB_UMI_Complex→CB_samTagOut (1 segment): CB@"
                              << (cb_s+1) << ":" << cb_len_calc
                              << " UMI@" << (umi_s_pos+1) << ":" << umi_len_calc << "\n";

                    args.push_back("--soloType"); args.push_back("CB_samTagOut");
                    solo_type_is_sam_tag_out = true;
                    args.push_back("--soloCBwhitelist");
                    if (no_whitelist && !has_auto_barcodes) { args.push_back("None"); }
                    else { args.push_back(effective_whitelist_for_complex.c_str()); }
                    cbl_s = std::to_string(cb_len_calc);
                    cb_start_s = std::to_string(cb_s + 1); // STAR 1-based
                    umi_start_s = std::to_string(umi_s_pos + 1);
                    umi_s = std::to_string(umi_len_calc);
                    args.push_back("--soloCBlen");    args.push_back(cbl_s.c_str());
                    args.push_back("--soloCBstart");  args.push_back(cb_start_s.c_str());
                    args.push_back("--soloUMIstart"); args.push_back(umi_start_s.c_str());
                    args.push_back("--soloUMIlen");   args.push_back(umi_s.c_str());
                    args.push_back("--soloBarcodeReadLength"); args.push_back("0");
                    umi_len_set = true;
                    use_complex = false;
                } else {
                    // Multi-segment CB (BD Rhapsody 3×9bp, inDrop 2-part, ddSEQ 3×6bp, etc.)
                    std::cerr << "[singlet] CB_UMI_Complex: "
                              << detected_cb_positions.size() << " CB segments + UMI="
                              << detected_umi_position << "\n";

                    if (use_complex_concat && concat_bc_total > 0 && concat_umi_len_c > 0) {
                        // ── CB_samTagOut concat: R1 was rewritten as [seg0]...[segN][UMI] ──
                        // Avoids CB_UMI_Complex's 200-400 GB RAM; ~5 GB for 884K WL + Exact.
                        static std::string concat_wl_arg;
                        concat_wl_arg = !combined_wl_file.empty() ? combined_wl_file
                            : (has_auto_barcodes ? barcode_file : std::string("None"));
                        std::cerr << "[singlet] CB_UMI_Complex\xE2\x86\x92CB_samTagOut concat:"
                                  << " BClen=" << concat_bc_total
                                  << " UMIlen=" << concat_umi_len_c
                                  << " WL=" << concat_wl_arg << "\n";
                        cbl_s       = std::to_string(concat_bc_total);
                        umi_start_s = std::to_string(concat_bc_total + 1); // 1-based
                        umi_s       = std::to_string(concat_umi_len_c);
                        args.push_back("--soloType");        args.push_back("CB_samTagOut");
                        solo_type_is_sam_tag_out = true;
                        args.push_back("--soloCBwhitelist"); args.push_back(concat_wl_arg.c_str());
                        args.push_back("--soloCBlen");       args.push_back(cbl_s.c_str());
                        args.push_back("--soloCBstart");     args.push_back("1");
                        args.push_back("--soloUMIstart");    args.push_back(umi_start_s.c_str());
                        args.push_back("--soloUMIlen");      args.push_back(umi_s.c_str());
                        args.push_back("--soloBarcodeReadLength"); args.push_back("0");
                        umi_len_set = true;
                    } else {

                    // ── Multi-segment CB_UMI_Complex STAR argument selection ──
                    // Priority 1: Per-segment auto-discovered barcodes (best: each segment's
                    //             actual barcodes, explosion-checked in the post-Phase-0.9 block).
                    // Priority 2: Has-auto-barcodes (seg-0 only): fall back to CB_samTagOut.
                    // Priority 3: Fixed per-segment whitelist from disk (surecell_bc.txt etc.).
                    //             Guard: never pass "None" to CB_UMI_Complex (STAR exit 109).
                    const bool use_per_seg_wl =
                        per_seg_barcode_files.size() == detected_cb_positions.size();

                    auto parse_pos_ms = [](const std::string& p) -> std::pair<int,int> {
                        int vals[4] = {0,0,0,0};
                        size_t pos_s = 0; int vi = 0;
                        for (size_t i = 0; i <= p.size() && vi < 4; ++i) {
                            if (i == p.size() || p[i] == '_') {
                                vals[vi++] = std::atoi(p.substr(pos_s, i - pos_s).c_str());
                                pos_s = i + 1;
                            }
                        }
                        return {vals[1], vals[3]};
                    };

                    if (use_per_seg_wl) {
                        // Priority 1: CB_UMI_Complex with per-segment auto-discovered barcodes.
                        // per_seg_barcode_files[0..N-1] are valid whitelist files.
                        args.push_back("--soloType"); args.push_back("CB_UMI_Complex");
                        args.push_back("--soloCBwhitelist");
                        // Strings must outlive args — keep them in a static vector
                        static std::vector<std::string> per_seg_wl_strs;
                        per_seg_wl_strs = per_seg_barcode_files;
                        for (auto& wl : per_seg_wl_strs)
                            args.push_back(wl.c_str());
                        // --soloCBposition must appear EXACTLY ONCE followed by all values.
                        static std::vector<std::string> cb_pos_strs_ps;
                        cb_pos_strs_ps = detected_cb_positions;
                        args.push_back("--soloCBposition");
                        for (const auto& cp : cb_pos_strs_ps)
                            args.push_back(cp.c_str());
                        if (!detected_umi_position.empty()) {
                            umi_start_s = detected_umi_position;
                            args.push_back("--soloUMIposition");
                            args.push_back(umi_start_s.c_str());
                        }
                        args.push_back("--soloBarcodeReadLength"); args.push_back("0");
                        if (!detected_umi_position.empty()) {
                            auto [us, ue] = parse_pos_ms(detected_umi_position);
                            umi_s = std::to_string(ue - us + 1);
                            args.push_back("--soloUMIlen"); args.push_back(umi_s.c_str());
                            umi_len_set = true;
                        }
                        std::cerr << "[singlet] CB_UMI_Complex: using CB_UMI_Complex with "
                                  << per_seg_wl_strs.size() << " per-segment whitelist files\n";
                    } else if (has_auto_barcodes) {
                        // Priority 2: Seg-0 auto-barcodes only → CB_samTagOut fallback.
                        // auto_barcodes.tsv covers seg-0 (smallest segment in discovery);
                        // CB_samTagOut writes the raw seg-0 barcode to the CB tag, and the
                        // pileup engine matches against the same set.
                        auto [cb_s0, cb_e0] = parse_pos_ms(detected_cb_positions[0]);
                        int cb_len_ms = cb_e0 - cb_s0 + 1;
                        auto [umi_s0, umi_e0] = parse_pos_ms(detected_umi_position);
                        int umi_len_ms = umi_e0 - umi_s0 + 1;

                        std::cerr << "[singlet] CB_UMI_Complex→CB_samTagOut"
                                  << " (auto-barcodes: seg0 CB@" << (cb_s0+1) << ":" << cb_len_ms
                                  << " UMI@" << (umi_s0+1) << ":" << umi_len_ms << ")\n";

                        args.push_back("--soloType"); args.push_back("CB_samTagOut");
                        solo_type_is_sam_tag_out = true;
                        args.push_back("--soloCBwhitelist");
                        args.push_back(effective_whitelist_for_complex.c_str());
                        cbl_s      = std::to_string(cb_len_ms);
                        cb_start_s = std::to_string(cb_s0 + 1);   // STAR 1-based
                        umi_start_s = std::to_string(umi_s0 + 1);
                        umi_s       = std::to_string(umi_len_ms);
                        args.push_back("--soloCBlen");    args.push_back(cbl_s.c_str());
                        args.push_back("--soloCBstart");  args.push_back(cb_start_s.c_str());
                        args.push_back("--soloUMIstart"); args.push_back(umi_start_s.c_str());
                        args.push_back("--soloUMIlen");   args.push_back(umi_s.c_str());
                        args.push_back("--soloBarcodeReadLength"); args.push_back("0");
                        umi_len_set = true;
                    } else {
                        // Priority 3: Fixed per-segment whitelist from disk.
                        // Look up per-segment whitelists from the protocol registry first.
                        // This handles BD Rhapsody (bd_cls1.txt × 3), SureCell (surecell_bc.txt × 3),
                        // microwell-seq (microwell_bc1_bare.txt × 3), and inDrop (bc1 + bc2).
                        const std::size_t n_segs3 = detected_cb_positions.size();
                        std::vector<std::string> resolved_wls;
                        if (!star_bin_dir.empty() && !complex_protocol_tag.empty()) {
                            resolved_wls = lib1fq::resolve_per_seg_whitelists(
                                star_bin_dir, complex_protocol_tag, n_segs3);
                        }
                        // If resolve_per_seg_whitelists returned nothing, try the single
                        // effective_whitelist_for_complex (valid file, replicated across segs).
                        if (resolved_wls.empty() &&
                            effective_whitelist_for_complex != "None" &&
                            effective_whitelist_for_complex != "none" &&
                            !effective_whitelist_for_complex.empty()) {
                            resolved_wls.assign(n_segs3, effective_whitelist_for_complex);
                        }
                        if (!resolved_wls.empty()) {
                            // Fixed per-segment whitelist: use CB_UMI_Complex natively.
                            // Whitelist product stays bounded (e.g. surecell 96^3 = 884K).
                            std::cerr << "[singlet] CB_UMI_Complex: using disk per-segment"
                                      << " whitelists (" << resolved_wls.size() << " segs)\n";
                            args.push_back("--soloType"); args.push_back("CB_UMI_Complex");
                            // STAR requires one --soloCBwhitelist entry per CB segment.
                            args.push_back("--soloCBwhitelist");
                            // resolved_wls strings must outlive args (stored in static vec).
                            static std::vector<std::string> disk_wl_strs;
                            disk_wl_strs = resolved_wls;
                            for (const auto& wl : disk_wl_strs)
                                args.push_back(wl.c_str());
                            // --soloCBposition must appear EXACTLY ONCE followed by all values.
                            // Repeating it per segment causes STAR exit 102.
                            static std::vector<std::string> cb_pos_strs;
                            cb_pos_strs = detected_cb_positions;
                            args.push_back("--soloCBposition");
                            for (const auto& cp : cb_pos_strs)
                                args.push_back(cp.c_str());
                            if (!detected_umi_position.empty()) {
                                umi_start_s = detected_umi_position;
                                args.push_back("--soloUMIposition");
                                args.push_back(umi_start_s.c_str());
                            }
                            args.push_back("--soloBarcodeReadLength"); args.push_back("0");
                            if (!detected_umi_position.empty()) {
                                auto [us3, ue3] = parse_pos_ms(detected_umi_position);
                                umi_s = std::to_string(ue3 - us3 + 1);
                                args.push_back("--soloUMIlen"); args.push_back(umi_s.c_str());
                                umi_len_set = true;
                            }
                        } else {
                            // Last resort: no disk whitelists and no auto-barcodes.
                            // Fall back to CB_samTagOut with seg-0 geometry (whitelist-free).
                            // STAR CB_samTagOut accepts "None"; CB_UMI_Complex does not.
                            // This allows EmptyDrops to call cells without a barcode whitelist.
                            std::cerr << "[singlet] WARNING: CB_UMI_Complex: no whitelist files"
                                         " found for '" << complex_protocol_tag
                                      << "'. Using CB_samTagOut seg-0 fallback (whitelist-free).\n"
                                      << "  Install whitelist files in whitelists/ for full"
                                         " CB_UMI_Complex mode.\n";
                            auto [cb_s_fb, cb_e_fb] = parse_pos_ms(detected_cb_positions[0]);
                            int cb_len_fb = cb_e_fb - cb_s_fb + 1;
                            int umi_s_pos_fb = 0, umi_len_fb = 8; // defaults
                            if (!detected_umi_position.empty()) {
                                auto [us_fb, ue_fb] = parse_pos_ms(detected_umi_position);
                                umi_s_pos_fb = us_fb;
                                umi_len_fb   = ue_fb - us_fb + 1;
                            }
                            args.push_back("--soloType"); args.push_back("CB_samTagOut");
                            solo_type_is_sam_tag_out = true;
                            args.push_back("--soloCBwhitelist"); args.push_back("None");
                            cbl_s      = std::to_string(cb_len_fb);
                            cb_start_s = std::to_string(cb_s_fb + 1);         // STAR 1-based
                            umi_start_s = std::to_string(umi_s_pos_fb + 1);
                            umi_s       = std::to_string(umi_len_fb);
                            args.push_back("--soloCBlen");    args.push_back(cbl_s.c_str());
                            args.push_back("--soloCBstart");  args.push_back(cb_start_s.c_str());
                            args.push_back("--soloUMIstart"); args.push_back(umi_start_s.c_str());
                            args.push_back("--soloUMIlen");   args.push_back(umi_s.c_str());
                            args.push_back("--soloBarcodeReadLength"); args.push_back("0");
                            umi_len_set = true;
                        }
                    }
                    } // end else (!use_complex_concat)
                    use_complex = false;
                }
            } else {
                // Simple-protocol barcode handling.
                //
                // CB_UMI_Simple path (preferred): when a full protocol whitelist is
                // available (e.g. 3M-february-2018.txt for 10x v3), use STAR's native
                // CB_UMI_Simple mode.  STAR matches reads against all 3.7M WL barcodes,
                // producing CB:Z tags for every barcode within 1MM of the WL.  The
                // pileup then collects reads from ~100K+ ambient barcodes into
                // wl_ambient_gene_counts_, giving EmptyDrops a calibrated null model.
                //
                // Root cause of AUTOFIX-10XV3-LARGE-EMPTYDROPS: using CB_samTagOut with
                // the 11K auto_barcodes.tsv as the STAR whitelist eliminated the ambient
                // tail (only 764 ambient barcodes, 13% gene coverage).  EmptyDrops tested
                // 99.8% of barcodes and called nearly all as cells.
                //
                // CB_samTagOut fallback: whitelist-free protocols (no WL on disk), or
                // protocols where the pre-built WL doesn't match the chemistry (e.g.
                // 10x-arc-gex mapped to wrong v3 3M list).  Uses auto_barcodes.tsv so
                // STAR still has a reference for CB correction.
                const bool use_cb_umi_simple = (detected_soloType == "CB_UMI_Simple")
                    && !no_whitelist
                    && !whitelist_file.empty()
                    && whitelist_file != "None" && whitelist_file != "none";

                if (use_cb_umi_simple) {
                    // Native CB_UMI_Simple: STAR uses the full protocol whitelist.
                    // auto_barcodes.tsv is still written as a reference artifact but is
                    // NOT used as STAR's whitelist — EmptyDrops receives the full ambient
                    // barcode distribution via wl_ambient_gene_counts_.
                    int cb = detected_bc_len > 0 ? detected_bc_len : 16;
                    cbl_s      = std::to_string(cb);
                    cb_start_s = std::to_string(detected_cb_start);
                    umi_start_s = std::to_string(detected_umi_start > 0 ? detected_umi_start : cb + 1);
                    std::cerr << "[singlet] CB_UMI_Simple: native STAR mode"
                              << " CB@" << detected_cb_start << ":" << cb
                              << " UMI@" << umi_start_s << ":" << umi_len
                              << " WL=" << whitelist_file << "\n";
                    args.push_back("--soloType");        args.push_back("CB_UMI_Simple");
                    args.push_back("--soloCBwhitelist"); args.push_back(whitelist_file.c_str());
                    args.push_back("--soloCBlen");       args.push_back(cbl_s.c_str());
                    args.push_back("--soloCBstart");     args.push_back(cb_start_s.c_str());
                    args.push_back("--soloUMIstart");    args.push_back(umi_start_s.c_str());
                    args.push_back("--soloBarcodeReadLength"); args.push_back("0");
                } else {
                    // CB_samTagOut fallback: whitelist-free or WL-missing protocols.
                    args.push_back("--soloType"); args.push_back("CB_samTagOut");
                    solo_type_is_sam_tag_out = true;
                    args.push_back("--soloCBwhitelist");
                    // When Phase 0.9 auto-discovery produced barcodes AND the whitelist was
                    // auto-resolved from the protocol ID (not user-supplied), use the
                    // discovered barcodes file as the STAR whitelist.  This handles protocols
                    // where the pre-built WL doesn't match the chemistry (e.g. 10x-arc-gex
                    // symlinked to wrong v3 3M list: STAR assigns CB tags to only ~9K / 29M
                    // reads → 0 cells).
                    const bool use_auto_as_star_wl = whitelist_auto_resolved
                        && !auto_barcode_file.empty()
                        && !barcode_file.empty()
                        && barcode_file == auto_barcode_file
                        && barcode_file != "SMARTSEQ2_NONE";
                    if (no_whitelist) {
                        args.push_back("None");
                        int cb = detected_bc_len > 0 ? detected_bc_len : 16;
                        cbl_s = std::to_string(cb);
                        cb_start_s = std::to_string(detected_cb_start);
                        umi_start_s = std::to_string(detected_umi_start > 0 ? detected_umi_start : cb + 1);
                        args.push_back("--soloCBlen");    args.push_back(cbl_s.c_str());
                        args.push_back("--soloCBstart");  args.push_back(cb_start_s.c_str());
                        args.push_back("--soloUMIstart"); args.push_back(umi_start_s.c_str());
                        args.push_back("--soloBarcodeReadLength"); args.push_back("0");
                    } else if (use_auto_as_star_wl) {
                        // Use auto-discovered barcodes as STAR whitelist: handles protocols
                        // where the pre-built WL doesn't match (e.g. 10x-arc-gex wrong file).
                        std::cerr << "[singlet] auto_barcodes → STAR whitelist"
                                  << " (CB_samTagOut fallback: WL may not match chemistry)\n";
                        args.push_back(barcode_file.c_str());
                        args.push_back("--soloBarcodeReadLength"); args.push_back("0");
                    } else {
                        args.push_back(whitelist_file.c_str());
                        args.push_back("--soloBarcodeReadLength"); args.push_back("0");
                    }
                }
            }
            if (!umi_len_set) {
                // Only add soloUMIlen if not already set by complex-protocol fallback
                umi_s = std::to_string(umi_len);
                args.push_back("--soloUMIlen");    args.push_back(umi_s.c_str());
            }

            // Guard: soloUMIstart + soloUMIlen must fit within R1 length.
            // CBs work fine (BD-Rhapsody 96 barcodes, inDrop 5855 barcodes found) but some
            // SRA deposits have shorter R1 than the protocol spec expects (e.g. BD-Rhapsody
            // R1=35bp vs spec 60bp, inDrop R1=22bp vs spec 46bp).  Rather than hard-failing,
            // clamp the UMI position so STAR proceeds; UMI dedup will be less effective but
            // the count matrix is still valid.
            {
                int umi_start_val = std::atoi(umi_start_s.c_str());
                int umi_len_val   = std::atoi(umi_s.c_str());
                if (onefq_r1_len > 0 && umi_start_val > 0 && umi_len_val > 0 &&
                    umi_start_val + umi_len_val - 1 > static_cast<int>(onefq_r1_len)) {
                    int new_start, new_len;
                    if (umi_start_val > static_cast<int>(onefq_r1_len)) {
                        // UMI start is entirely past R1: use minimal 1-base UMI at last position
                        new_start = static_cast<int>(onefq_r1_len);
                        new_len   = 1;
                    } else {
                        // UMI start fits inside R1; truncate length to available bases
                        new_start = umi_start_val;
                        new_len   = static_cast<int>(onefq_r1_len) - umi_start_val + 1;
                    }
                    std::cerr << "[singlet] WARNING: UMI position extends beyond R1 read length.\n"
                              << "  soloUMIstart=" << umi_start_val
                              << " + soloUMIlen=" << umi_len_val
                              << " = " << (umi_start_val + umi_len_val - 1)
                              << " > R1_len=" << onefq_r1_len << "\n"
                              << "  Clamping UMI to soloUMIstart=" << new_start
                              << " soloUMIlen=" << new_len
                              << " (UMI dedup less effective; alignment proceeds)\n";
                    // Update the string variables and patch the args vector so STAR
                    // receives the clamped values.  umi_start_s / umi_s are declared at
                    // the same scope as args and outlive this block.
                    umi_start_s = std::to_string(new_start);
                    umi_s       = std::to_string(new_len);
                    for (std::size_t ai = 0; ai + 1 < args.size(); ++ai) {
                        if (std::string_view(args[ai]) == "--soloUMIstart")
                            args[ai + 1] = umi_start_s.c_str();
                        if (std::string_view(args[ai]) == "--soloUMIlen")
                            args[ai + 1] = umi_s.c_str();
                    }
                }
            }

            // BAM output: sorted-to-file (parallel pileup) or unsorted-to-pipe (streaming)
            //
            // ADAPTIVE-BAM-SORT: Three-mode strategy to prevent OOM on large samples while
            // preserving speed on normal samples.
            //
            // Mode 1 — in-memory (default for small/mid samples):
            //   limitBAMsortRAM = 75% of SLURM allocation (or existing heuristic)
            //   No disk spill. Fastest path, current behaviour.
            //
            // Mode 2 — hybrid disk-assisted:
            //   Activates when estimated BAM > 60% of available RAM.
            //   limitBAMsortRAM = 40% of available RAM; outBAMsortingBinsN = 100.
            //   STAR spills overflow bins to tmpdir. ~10-20% BAM-sort overhead.
            //
            // Mode 3 — full disk sort:
            //   Activates when estimated BAM >= 2× available RAM.
            //   limitBAMsortRAM = min(32G, 25% RAM); outBAMsortingBinsN = 200.
            //   Most sorting happens on disk. ~2-3× BAM-sort overhead but prevents OOM.
            //
            // Estimation: ~200 bytes/aligned read average.
            // Override: --bam-sort-mode memory|disk|auto
            std::string bam_sort_ram_s;
            std::string bam_sort_bins_s;
            std::string bam_sort_threads_s;
            if (parallel_pileup == 1) {
                // ── ADAPTIVE-BAM-SORT: compute available RAM and estimate BAM size ──
                const char* slurm_mem_env = getenv("SLURM_MEM_PER_NODE");
                uint64_t available_ram_bytes;
                uint64_t alloc_gib;
                if (slurm_mem_env) {
                    uint64_t slurm_mb = static_cast<uint64_t>(std::atoll(slurm_mem_env));
                    available_ram_bytes = slurm_mb * 1024ULL * 1024ULL;
                    alloc_gib = slurm_mb / 1024;
                } else {
                    available_ram_bytes = 128ULL * 1024 * 1024 * 1024;  // 128 GiB fallback
                    alloc_gib = 128;
                }

                // Estimate sorted BAM size: ~200 bytes per read (covers compressed + index overhead)
                // For large samples (>200M) use 200 bytes; for smaller samples this is conservative.
                uint64_t est_bam_bytes = (onefq_n_reads > 0)
                    ? onefq_n_reads * 200ULL
                    : 0ULL;

                // Determine sort mode
                // Mode thresholds: est_bam >= 2.0× RAM → disk; >= 0.6× RAM → hybrid; else memory
                enum class BamSortMode { Memory, Hybrid, Disk };
                BamSortMode sort_mode;
                if (bam_sort_mode == "memory") {
                    sort_mode = BamSortMode::Memory;
                } else if (bam_sort_mode == "disk") {
                    sort_mode = BamSortMode::Disk;
                } else {
                    // auto: decide by ratio
                    if (est_bam_bytes >= available_ram_bytes * 2) {
                        sort_mode = BamSortMode::Disk;
                    } else if (est_bam_bytes >= (available_ram_bytes * 6 / 10)) {
                        sort_mode = BamSortMode::Hybrid;
                    } else {
                        sort_mode = BamSortMode::Memory;
                    }
                }

                // AUTOFIX-BD-RHAPSODY-OOM: Fallback memory-tier selection for known
                // multi-segment protocols.  use_complex is reset to false inside the
                // soloType argument block above (both the single-CB and multi-CB branches
                // end with use_complex=false so later logic cannot distinguish CB_samTagOut
                // from CB_UMI_Complex).  Re-set use_complex=true here — ONLY for the
                // memory-tier decision below — when the protocol's known_protocols() entry
                // has per_seg_whitelist_files (bd-rhapsody, splitseq, indrop, sci-rna-seq3).
                // This ensures compression=1 and the lower RAM cap are applied regardless
                // of whether detected_cb_positions was populated during auto-discovery.
                // Skip complex memory tier when concat mode is active:
                // CB_samTagOut with combined WL uses ~5 GB, not 200-400 GB.
                if (!use_complex && !use_complex_concat && !complex_protocol_tag.empty()) {
                    for (const auto& sp : lib1fq::known_protocols()) {
                        if (sp.tag == complex_protocol_tag &&
                            !sp.per_seg_whitelist_files.empty()) {
                            use_complex = true;
                            std::cerr << "[singlet] CB_UMI_COMPLEX memory tier: '"
                                      << complex_protocol_tag
                                      << "' has " << sp.per_seg_whitelist_files.size()
                                      << " per-seg whitelists — applying complex BAM"
                                         " compression + RAM cap\n";
                            break;
                        }
                    }
                }

                // BAM compression tier (see include/singlet/pileup/mega_sort_params.h):
                //   normal  (<=200M)           compression=0  (uncompressed; fastest)
                //   mega    (>200M, <500M)     compression=1  (4x smaller BAM, ~5% CPU)
                //   ultra   (>=500M)           compression=0  (uncompressed; the tight
                //                                              15%/64-GiB limitBAMsortRAM
                //                                              cap + ULTRA_SORT_BINS=100
                //                                              is what addresses
                //                                              AUTOFIX-MEGA-SORT-RSS-OVERAGE
                //                                              on 666M/817M samples.)
                //   complex (CB_UMI_Complex)   compression=1  (BD Rhapsody; combinatorial
                //                                              match tables + uncompressed
                //                                              BAM OOMs at 30M on 192G)
                const bool is_mega = (onefq_n_reads > singlet_pileup::mega_sort::MEGA_READ_THRESHOLD);
                const bool is_ultra = (onefq_n_reads >= singlet_pileup::mega_sort::ULTRA_READ_THRESHOLD);
                const int bam_comp_level = singlet_pileup::mega_sort::compression_level(
                    onefq_n_reads, use_complex);
                // compression_level() returns only 0 or 1. Use string literals with
                // static storage duration — NOT a local std::string whose .c_str() would
                // become a dangling pointer after this if(parallel_pileup==1) block closes
                // (before star_main_impl reads args). Bug: bam_comp_str_storage was a local
                // string declared at 16-space scope; when the block closed it was destroyed,
                // leaving args with a dangling pointer that read garbage (e.g. the STAR PID).
                static const char* const bam_comp_strs[] = {"0","1","2","3","4","5","6","7","8","9"};
                const char* bam_comp_str = bam_comp_strs[std::min(bam_comp_level, 9)];
                args.push_back("--outSAMtype");    args.push_back("BAM"); args.push_back("SortedByCoordinate");
                args.push_back("--outBAMcompression"); args.push_back(bam_comp_str);
                if (is_ultra)
                    std::cerr << "[singlet] ULTRA: --outBAMcompression=" << bam_comp_level
                              << " (n_reads=" << onefq_n_reads
                              << " >= " << singlet_pileup::mega_sort::ULTRA_READ_THRESHOLD
                              << "; memory lever is limitBAMsortRAM+bins, not compression)\n";
                else if (is_mega)
                    std::cerr << "[singlet] MEGA: --outBAMcompression=" << bam_comp_level
                              << " (n_reads=" << onefq_n_reads << ")\n";
                if (use_complex && !is_mega && !is_ultra)
                    std::cerr << "[singlet] CB_UMI_COMPLEX: --outBAMcompression=" << bam_comp_level
                              << " (combinatorial barcode matching)\n";

                // ── Compute limitBAMsortRAM based on selected mode ──
                // --limitBAMsortRAM: always cap BAM sort memory to prevent OOM on large samples.
                // Required non-zero for LoadAndKeep/LoadAndRemove; also important in NoSharedMemory
                // mode where STAR otherwise uses all available RAM for sorting.
                uint64_t bam_sort_ram = 0;

                if (sort_mode == BamSortMode::Memory) {
                    // B-G3-4: try deterministic tier-table lookup first.
                    // slurm_tier_bamsort_ram() maps known SLURM allocation sizes
                    // (64/128/192/384 GB) to pinned limitBAMsortRAM values (25/50/75/150 GB).
                    uint64_t tier_bam = singlet_pileup::mega_sort::slurm_tier_bamsort_ram(
                        available_ram_bytes);
                    if (tier_bam > 0) {
                        bam_sort_ram = tier_bam;
                        std::cerr << "[singlet] B-G3-4: SLURM tier "
                                  << alloc_gib << "GiB → limitBAMsortRAM="
                                  << (bam_sort_ram >> 30) << "GiB (deterministic)\n";
                    } else {
                    // S5-BAM-RAM: derive from SA size (3× SA + 8 GiB overhead), fallback 60 GiB.
                    bam_sort_ram = 60ULL * 1024 * 1024 * 1024;
                    if (!genome_dir.empty()) {
                        std::string sa_path = genome_dir + "/SA";
                        struct stat sa_st{};
                        if (stat(sa_path.c_str(), &sa_st) == 0 && sa_st.st_size > 0) {
                            bam_sort_ram = static_cast<uint64_t>(sa_st.st_size) * 3
                                           + 8ULL * 1024 * 1024 * 1024;
                            std::cerr << "[singlet] S5-BAM-RAM: SA="
                                      << (sa_st.st_size >> 30)
                                      << "GiB → limitBAMsortRAM="
                                      << (bam_sort_ram >> 30) << "GiB\n";
                        }
                    }
                    // MEGA: scale up for high-read-count samples where SA*3+8G is insufficient.
                    if (onefq_n_reads > 0) {
                        uint64_t read_based = (onefq_n_reads / 1000000ULL) * 300000000ULL;
                        if (read_based > bam_sort_ram) {
                            bam_sort_ram = read_based;
                            std::cerr << "[singlet] MEGA: " << onefq_n_reads
                                      << " reads → limitBAMsortRAM="
                                      << (bam_sort_ram >> 30) << "GiB\n";
                        }
                    }
                    // G-TINY: cap limitBAMsortRAM based on actual read count for small datasets
                    if (onefq_n_reads > 0) {
                        uint64_t tiny_ram = singlet_pileup::TinyDatasetGuard::recommended_bam_sort_ram(onefq_n_reads);
                        if (tiny_ram < bam_sort_ram) {
                            bam_sort_ram = tiny_ram;
                            std::cerr << "[singlet] G-TINY: " << onefq_n_reads
                                      << " reads → limitBAMsortRAM="
                                      << (bam_sort_ram >> 30) << "GiB\n";
                        }
                    }
                    }  // end else (SA-based fallback: tier table did not match)

                    // SLURM-CAP: apply ram_cap_bytes() safety ceiling AFTER both the
                    // tier-table and SA-based paths. This ensures ULTRA samples (≥500M
                    // reads) get capped to 15%/64GiB even when the tier table matched.
                    // Previously this was inside the `else` branch only — causing OOM
                    // on 647M-read samples at 384G tier (FAILURE-9).
                    {
                        uint64_t max_bam_sort = singlet_pileup::mega_sort::ram_cap_bytes(
                            onefq_n_reads, available_ram_bytes, use_complex);
                        auto tier = singlet_pileup::mega_sort::tier_for(onefq_n_reads, use_complex);
                        if (max_bam_sort > 0 && bam_sort_ram > max_bam_sort) {
                            uint64_t auto_computed = bam_sort_ram;
                            bam_sort_ram = max_bam_sort;
                            std::cerr << "[singlet] limitBAMsortRAM ["
                                      << singlet_pileup::mega_sort::tier_name(tier) << "]: "
                                      << (auto_computed >> 30) << " GiB → capped at "
                                      << (bam_sort_ram >> 30) << " GiB (node allocation: "
                                      << alloc_gib << " GiB)\n";
                        }
                    }

                    // ULTRA-BINS: ≥500M reads in memory-sort mode — use ULTRA_SORT_BINS
                    // (100) bins to reduce per-bin merge buffer allocations. Previous
                    // value of 500 triggered STAR's "expected vs actual bin size
                    // disagreement" pathology on GSM7102845 (bin #494). 100 is 2×
                    // STAR's default (50) and well inside its tested range.
                    if (is_ultra) {
                        bam_sort_bins_s = std::to_string(
                            singlet_pileup::mega_sort::ULTRA_SORT_BINS);
                        std::cerr << "[singlet] ULTRA: outBAMsortingBinsN="
                                  << singlet_pileup::mega_sort::ULTRA_SORT_BINS
                                  << " (reduces per-bin merge memory, limitBAMsortRAM="
                                  << (bam_sort_ram >> 30) << "GiB)\n";
                    }

                } else if (sort_mode == BamSortMode::Hybrid) {
                    // Mode 2: disk-assisted. 40% of available RAM for sort; 100 bins.
                    bam_sort_ram = (available_ram_bytes * 2) / 5;
                    int sort_threads = std::min(4, star_threads / 2);
                    if (sort_threads < 1) sort_threads = 1;
                    bam_sort_bins_s = "100";
                    bam_sort_threads_s = std::to_string(sort_threads);
                    std::cerr << "[singlet] BAM sort strategy: hybrid (est BAM: "
                              << (est_bam_bytes >> 30) << "GB, avail RAM: "
                              << alloc_gib << "GB, limitBAMsortRAM: "
                              << (bam_sort_ram >> 30) << "GB, bins: 100)\n";

                } else {
                    // Mode 3: full disk sort. min(32G, 25% RAM); 200 bins.
                    uint64_t quarter_ram = available_ram_bytes / 4;
                    uint64_t cap_32g = 32ULL * 1024 * 1024 * 1024;
                    bam_sort_ram = (quarter_ram < cap_32g) ? quarter_ram : cap_32g;
                    bam_sort_bins_s = "200";
                    std::cerr << "[singlet] BAM sort strategy: disk (est BAM: "
                              << (est_bam_bytes >> 30) << "GB, avail RAM: "
                              << alloc_gib << "GB, limitBAMsortRAM: "
                              << (bam_sort_ram >> 30) << "GB, bins: 200)\n";
                }

                // Ensure we always emit the mode for Mode 1 so logs are consistent
                if (sort_mode == BamSortMode::Memory) {
                    std::cerr << "[singlet] BAM sort strategy: memory (est BAM: "
                              << (est_bam_bytes >> 30) << "GB, avail RAM: "
                              << alloc_gib << "GB, limitBAMsortRAM: "
                              << (bam_sort_ram >> 30) << "GB)\n";
                }

                // AUTOFIX-SCI3-CBSAM-HANG: for CB_samTagOut protocols (sci-RNA-seq3, inDrop,
                // etc.) in Memory sort mode with >10M reads, force 200 sort bins.
                // Default 50 bins can cause STAR's binSort to take 100+ minutes on 48M-read
                // samples where each SAM record carries extra CB/UR tags (+40 bytes/read).
                // 200 bins reduces per-bin merge memory and eliminates the pathological hang.
                if (sort_mode == BamSortMode::Memory && solo_type_is_sam_tag_out
                        && onefq_n_reads > 10000000ULL && bam_sort_bins_s.empty()) {
                    bam_sort_bins_s = "200";
                    std::cerr << "[singlet] CB_samTagOut sort bins: 200 ("
                              << onefq_n_reads << " reads, AUTOFIX-SCI3-CBSAM-HANG)\n";
                }

                bam_sort_ram_s = std::to_string(bam_sort_ram);
                args.push_back("--limitBAMsortRAM"); args.push_back(bam_sort_ram_s.c_str());

                // Disk/hybrid modes: emit bin count and per-sort thread count
                if (!bam_sort_bins_s.empty()) {
                    args.push_back("--outBAMsortingBinsN"); args.push_back(bam_sort_bins_s.c_str());
                }
                if (!bam_sort_threads_s.empty()) {
                    args.push_back("--outBAMsortingThreadN"); args.push_back(bam_sort_threads_s.c_str());
                }

                // Track whether disk sort is active for tmpdir selection below
                // (used by the star_tmpdir block that follows)
                (void)sort_mode; // captured via star_tmpdir block closure
                // Store sort_mode result for tmpdir selection (re-derive there from bam_sort_bins_s)
                // bam_sort_bins_s non-empty ↔ disk or hybrid mode active

            } else {
                args.push_back("--outSAMtype");    args.push_back("BAM"); args.push_back("Unsorted");
                args.push_back("--outStd");        args.push_back("BAM_Unsorted");
                args.push_back("--outBAMcompression"); args.push_back("0");
            }

            // SAM attributes needed by pileup
            args.push_back("--outSAMattributes");
            args.push_back("NH"); args.push_back("nM");
            args.push_back("AS"); args.push_back("CR"); args.push_back("UR");
            args.push_back("CB");

            // Skip unmapped reads (pileup ignores them anyway)
            args.push_back("--outSAMunmapped"); args.push_back("None");

            // NONHOST-UNMAPPED-CAPTURE: write unmapped reads to FASTQ when any nonhost index is provided.
            // STAR writes: ${out_prefix}star_Unmapped.out.mate1 (cDNA, R2) and .mate2 (barcode, R1).
            // NonHostScreener runs on mate1 (cDNA) after STAR+pileup complete.
            if (!nonhost_db_path.empty() || !bacterial_db_path.empty() || !fungal_db_path.empty()) {
                args.push_back("--outReadsUnmapped"); args.push_back("Fastx");
            }

            // Quality / barcode matching
            // Adapter trimming: 5' capture protocols use TSO adapter trim via
            // --clip3pAdapterSeq (from known_protocols().adapter3p).
            // 3' protocols (10x v2/v3/v4, Drop-seq, etc.) use NO explicit adapter
            // clipping: STAR's alignment soft-clipping handles polyA naturally.
            // NOTE: --clipAdapterType CellRanger4 was removed because STAR's
            // CellRanger4 mode relies on quality-score heuristics that break
            // when singlet decodes with --quality none (constant Phred=30),
            // causing 0% mapping on mouse 10xv3 and other samples.
            if (!detected_adapter3p.empty()) {
                args.push_back("--clip3pAdapterSeq");
                args.push_back(detected_adapter3p.c_str());
                args.push_back("--clip3pAdapterMMp");
                args.push_back("0.1");  // ≤10% mismatch rate in adapter match
            }
            // Score filter: absolute min=10 covers short-read protocols (e.g. DNBelab 30bp).
            // STAR's default outFilterScoreMinOverLread=0.66 handles long reads.
            // Combined: max(10, 0.66*readLen) → 20 for 30bp, 66 for 100bp.
            args.push_back("--outFilterScoreMin");   args.push_back("10");
            args.push_back("--soloCBmatchWLtype");
            if (solo_type_is_sam_tag_out || use_complex || use_complex_concat) {
                args.push_back("Exact");  // CB_samTagOut / CB_UMI_Complex / concat: 1MM_multi not supported by CB_samTagOut; combinatorial 1MM is memory-prohibitive for CB_UMI_Complex
            } else {
                args.push_back("1MM_multi_Nbase_pseudocounts");  // CB_UMI_Simple: 1-Hamming correction; cell calling uses STAR Solo.out
            }

            // Threading
            std::string threads_s = std::to_string(star_threads);
            args.push_back("--runThreadN");          args.push_back(threads_s.c_str());

            // Genome loading
            args.push_back("--genomeLoad");          args.push_back(genome_load.c_str());

            // 5' clip for constant R2 prefix contamination.
            // The clip is applied at the FASTQ file level (above, after the reads_inverted
            // swap) so reads_r2 already has the trimmed cDNA reads by this point.
            // r2_clip5p is reset to 0 after a successful file-level trim, so this
            // block only fires as a fallback when the file open failed.
            //
            // ALWAYS use the two-value form "--clip5pNbases N 0":
            //   File 1 = cDNA (reads_r2, after any R1/R2 swap) → clip N bases.
            //   File 2 = barcode (reads_r1)                    → clip 0 bases.
            //   The single-value form clips BOTH mates equally, destroying all
            //   barcodes → 0 cells called. Confirmed empirically (job 360648,
            //   SRR34789664: single-value → 0 cells; two-value form required).
            std::string clip5p_str;
            if (r2_clip5p > 0) {
                clip5p_str = std::to_string(r2_clip5p);
                args.push_back("--clip5pNbases");
                args.push_back(clip5p_str.c_str());  // cDNA mate: clip N bases
                args.push_back("0");                 // barcode mate: never clip
            }

            // STAR log/temp files
            std::string star_prefix = out_prefix + "/star_";
            args.push_back("--outFileNamePrefix");   args.push_back(star_prefix.c_str());

            // STAR sort temp directory.
            // For in-memory sort (no disk spill): use /dev/shm for fast I/O.
            // For disk/hybrid sort OR large samples (>200M reads): use NFS or /tmp to
            // avoid exhausting /dev/shm, which is RAM-backed — spilling to /dev/shm
            // would negate the purpose of disk sort entirely.
            // An 800M-read sort on /dev/shm can require 100-200 GiB of RAM-backed tmpfs.
            std::string star_tmpdir;
            {
                // Disk/hybrid sort is active when bam_sort_bins_s is non-empty (parallel path).
                const bool disk_sort_active = !bam_sort_bins_s.empty();
                const bool mega_tmpdir = ((onefq_n_reads > 200000000ULL)
                                          && (getenv("SLURM_MEM_PER_NODE") != nullptr))
                                         || disk_sort_active;
                if (mega_tmpdir) {
                    // Prefer /tmp (disk-backed, not RAM) for the overflow bins.
                    // /tmp is always on local disk on Clipper nodes. Fall back to NFS out_prefix
                    // if /tmp has less than 100 GiB free (very conservative — overflow is at most
                    // a few tens of GB per sort run for typical cluster jobs).
                    const char* tmp_env = getenv("TMPDIR");
                    std::string tmp_root = (tmp_env && *tmp_env) ? std::string(tmp_env) : "/tmp";
                    struct statvfs tmp_svfs;
                    uint64_t tmp_free_bytes = 0;
                    if (statvfs(tmp_root.c_str(), &tmp_svfs) == 0)
                        tmp_free_bytes = tmp_svfs.f_bsize * tmp_svfs.f_bavail;
                    const uint64_t min_tmp_needed = 50ULL * 1024 * 1024 * 1024;  // 50 GiB threshold
                    if (tmp_free_bytes >= min_tmp_needed) {
                        star_tmpdir = tmp_root + "/singlet_star_" + std::to_string(getpid());
                        std::cerr << "[singlet] ADAPTIVE-BAM-SORT: outTmpDir=" << star_tmpdir
                                  << " (/tmp, " << (tmp_free_bytes >> 30) << "GiB free)\n";
                    } else {
                        // /tmp too small — fall back to NFS output directory (always has space)
                        star_tmpdir = out_prefix + "/star_tmp_" + std::to_string(getpid());
                        std::cerr << "[singlet] ADAPTIVE-BAM-SORT: outTmpDir=" << star_tmpdir
                                  << " (NFS fallback; /tmp only "
                                  << (tmp_free_bytes >> 30) << "GiB free)\n";
                    }
                } else {
                    star_tmpdir = "/dev/shm/singlet_star_" + std::to_string(getpid());
                }
            }
            if (parallel_pileup == 1) {
                args.push_back("--outTmpDir");
                args.push_back(star_tmpdir.c_str());
            }

            // Extra user args
            for (const auto& a : star_extra) args.push_back(a.c_str());

            args.push_back(nullptr);

            // Call bundled STAR directly — no exec, no external binary needed.
            int rc = star_main_impl(
                static_cast<int>(args.size()) - 1,
                const_cast<char**>(args.data())
            );
            _exit(rc);
        }

        // ── PARENT ──
        if (parallel_pileup == 1) {
            // Parallel path: STAR writes sorted BAM to file; no pipe in parent.
            std::string star_prefix = out_prefix + "/star_";
            parallel_bam_path = star_prefix + "Aligned.sortedByCoord.out.bam";
            parallel_bai_path = parallel_bam_path + ".bai";
            bam_source = parallel_bam_path;
            std::cerr << "[singlet] STAR pid=" << star_pid
                      << ", " << star_threads << " threads, sorted BAM → " << parallel_bam_path << "\n";
        } else {
            close(pipe_fd[1]);
            bam_source = "/dev/fd/" + std::to_string(pipe_fd[0]);
            std::cerr << "[singlet] STAR pid=" << star_pid
                      << ", " << star_threads << " threads, pipe fd=" << pipe_fd[0] << "\n";
        }
    }

    // ── Phase 2: Run pileup ──
    // Parallel path: wait for STAR, index BAM, run_parallel().
    // Streaming path: run() concurrently with STAR, then waitpid.

    singlet::PileupStats stats;
    double pileup_time = 0.0;
    double star_time = 0.0;
    int star_exit = 0;

    if (parallel_pileup == 1 && star_pid > 0) {
        // ── 2a: Wait for STAR to finish writing sorted BAM ──
        std::cerr << "[singlet] Waiting for STAR to finish sorting...\n";
        StopWatch star_sw;
        int wstatus;
        waitpid(star_pid, &wstatus, 0);
        // Join FIFO writer thread now — it finishes just before STAR exits (writer
        // closes FIFOs → STAR gets EOF → STAR finishes sorting → parent wakes here).
        if (fifo_writer_thread.joinable()) fifo_writer_thread.join();
        // Note: .1fq is user-provided input — never delete it. Orchestrator scripts
        // handle cleanup of intermediate files when appropriate.
        if (WIFEXITED(wstatus)) {
            star_exit = WEXITSTATUS(wstatus);
            if (star_exit != 0) {
                std::cerr << "[singlet] ERROR: STAR exited " << star_exit << "\n";
                return star_exit;
            }
            star_time = star_sw.elapsed_s();
            std::cerr << "[singlet] STAR completed in " << star_time << "s\n";
        } else if (WIFSIGNALED(wstatus)) {
            std::cerr << "[singlet] ERROR: STAR killed by signal " << WTERMSIG(wstatus) << "\n";
            return 1;
        }

        // ── 2b: Index sorted BAM with htslib ──
        std::cerr << "[singlet] Indexing BAM: " << parallel_bam_path << "\n";
        StopWatch idx_sw;
        int idx_ret = sam_index_build3(parallel_bam_path.c_str(),
                                       parallel_bai_path.c_str(),
                                       0,          // min_shift=0 → BAI format
                                       star_threads);
        if (idx_ret < 0) {
            std::cerr << "[singlet] ERROR: BAM index failed (" << idx_ret << ")\n";
            return 1;
        }
        std::cerr << "[singlet] BAM indexed in " << idx_sw.elapsed_s() << "s\n";

        // ── 2c-ATAC: Fragment extraction + bin counting (ATAC mode only) ──
        if (is_atac_mode) {
            std::cerr << "[singlet] ATAC: extracting fragments from sorted BAM...\n";
            StopWatch atac_sw;

            // Build barcode → index map from discovered/user-provided barcode file
            std::unordered_map<std::string, uint32_t> atac_bc_map;
            if (!barcode_file.empty() && barcode_file != "None" && barcode_file != "none") {
                std::ifstream bcf(barcode_file);
                std::string bc;
                uint32_t idx = 0;
                while (std::getline(bcf, bc)) {
                    if (!bc.empty()) atac_bc_map[bc] = idx++;
                }
                std::cerr << "[singlet] ATAC: loaded " << atac_bc_map.size() << " barcodes\n";
            }

            // Stream BAM records through ATACFragmentExtractor
            singlet::ATACFragmentExtractor extractor;
            extractor.set_barcode_map(atac_bc_map);

            samFile*    bam_fp  = sam_open(parallel_bam_path.c_str(), "rb");
            bam_hdr_t*  bam_hdr_atac = sam_hdr_read(bam_fp);
            bam1_t*     rec     = bam_init1();
            while (sam_read1(bam_fp, bam_hdr_atac, rec) >= 0)
                extractor.process_record(rec, bam_hdr_atac);
            bam_destroy1(rec);

            extractor.finalise();

            // Build chromosome sizes from BAM header (for bin counter)
            std::vector<std::pair<std::string, int32_t>> chrom_sizes;
            chrom_sizes.reserve(static_cast<size_t>(bam_hdr_atac->n_targets));
            for (int ci = 0; ci < bam_hdr_atac->n_targets; ++ci)
                chrom_sizes.emplace_back(bam_hdr_atac->target_name[ci],
                                         static_cast<int32_t>(bam_hdr_atac->target_len[ci]));

            bam_hdr_destroy(bam_hdr_atac);
            sam_close(bam_fp);

            std::cerr << "[singlet] ATAC: " << extractor.unique_fragments()
                      << " unique fragments (" << extractor.duplicate_fragments()
                      << " dupes)\n";

            // Bin counting (500 bp bins)
            singlet::ATACBinCounter counter;
            counter.set_bin_width(500);
            counter.set_chromosome_sizes(chrom_sizes);
            counter.count_fragments(extractor.fragments());

            const size_t n_cells = atac_bc_map.size();
            auto csc = counter.to_csc(n_cells);

            // Export: matrix, features, barcodes
            singlet::write_mtx(out_prefix + "/atac_matrix.mtx",
                               static_cast<uint32_t>(counter.total_bins()),
                               static_cast<uint32_t>(n_cells),
                               csc.indptr, csc.indices, csc.data);
            auto feat_labels = counter.bin_features();
            singlet::write_names(out_prefix + "/atac_features.tsv", feat_labels);
            std::vector<std::string> bc_list(n_cells);
            for (const auto& [bc, idx] : atac_bc_map) bc_list[idx] = bc;
            singlet::write_names(out_prefix + "/atac_barcodes.tsv", bc_list);

            // ── Write fragments.tsv.gz (BED-like: chrom, start, end, barcode, count) ──
            {
                const auto& frags = extractor.fragments();
                const std::string frag_path = out_prefix + "/fragments.tsv.gz";
                gzFile ftsv = gzopen(frag_path.c_str(), "wb");
                if (!ftsv) {
                    std::cerr << "[singlet] ERROR: could not open " << frag_path << " for writing\n";
                } else {
                    char buf[4096];
                    for (const auto& f : frags) {
                        int n = std::snprintf(buf, sizeof(buf), "%s\t%d\t%d\t%s\t1\n",
                                             chrom_sizes[static_cast<size_t>(f.chrom_idx)].first.c_str(),
                                             f.start, f.end,
                                             bc_list[f.barcode_idx].c_str());
                        if (n > 0) gzwrite(ftsv, buf, static_cast<unsigned>(n));
                    }
                    gzclose(ftsv);
                    std::cerr << "[singlet] ATAC: wrote " << frags.size()
                              << " fragments to fragments.tsv.gz\n";
                }
            }

            // ── A3 TSS enrichment QC ──────────────────────────────────────────────
            singlet::ATACQCComputer qc_computer;
            {
                std::unordered_map<std::string, int32_t> cmap;
                cmap.reserve(chrom_sizes.size());
                for (int32_t ci = 0; ci < static_cast<int32_t>(chrom_sizes.size()); ++ci)
                    cmap[chrom_sizes[static_cast<size_t>(ci)].first] = ci;
                qc_computer.set_chrom_map(cmap);
            }
            if (!config.exon_gtf_path.empty() &&
                config.exon_gtf_path != "None" && config.exon_gtf_path != "none")
                qc_computer.load_tss_from_gtf(config.exon_gtf_path);
            qc_computer.compute(extractor.fragments(), extractor,
                                static_cast<uint32_t>(n_cells), &counter);
            qc_computer.write_qc_tsv(out_prefix + "/atac_qc.tsv", bc_list);

            // ── A7 Cell calling ──────────────────────────────────────────────────
            {
                const auto& cell_qc = qc_computer.cell_qc();
                std::vector<uint64_t> frag_counts(n_cells);
                std::vector<double>   tss_scores(n_cells);
                std::vector<double>   frip_scores(n_cells);
                for (size_t i = 0; i < n_cells; ++i) {
                    frag_counts[i] = cell_qc[i].total_fragments;
                    tss_scores[i]  = static_cast<double>(cell_qc[i].tss_enrichment);
                    frip_scores[i] = static_cast<double>(cell_qc[i].frip_approx);
                }
                singlet::AtacCellCaller caller;
                std::vector<singlet::AtacCellCaller::CellCallResult> call_results;
                auto summary = caller.call_cells(bc_list, frag_counts, tss_scores,
                                                 frip_scores, call_results);
                std::ofstream ctsv(out_prefix + "/atac_cells.tsv");
                ctsv << "barcode\tis_cell\tunique_fragments\ttss_enrichment"
                        "\tfrip\tfilter_reason\n";
                for (const auto& r : call_results)
                    ctsv << r.barcode << '\t'
                         << static_cast<int>(r.is_cell) << '\t'
                         << r.unique_fragments << '\t'
                         << r.tss_enrichment << '\t'
                         << r.frip << '\t'
                         << r.filter_reason << '\n';
                std::cerr << "[singlet] ATAC: " << summary.cells_called
                          << "/" << summary.total_barcodes
                          << " cells called (frag_threshold=" << summary.fragment_threshold
                          << ", median_tss=" << summary.median_tss_cells << ")\n";
            }

            // ── A-PEAKS: Peak calling from aggregate bin counts ──────────────────
            {
                // Sum CSC across all cells to get per-bin aggregate counts
                std::vector<uint32_t> bin_agg(counter.total_bins(), 0u);
                for (size_t _k = 0; _k < csc.indices.size(); ++_k)
                    bin_agg[static_cast<size_t>(csc.indices[_k])] +=
                        static_cast<uint32_t>(csc.data[_k]);

                // Build per-bin chrom and start arrays from chromosome sizes
                constexpr int32_t kBinWidth = 500;
                std::vector<std::string> bin_chroms;
                std::vector<uint32_t>   bin_starts;
                bin_chroms.reserve(counter.total_bins());
                bin_starts.reserve(counter.total_bins());
                for (const auto& [chrom, clen] : chrom_sizes) {
                    for (int32_t b = 0; static_cast<int32_t>(b * kBinWidth) < clen; ++b) {
                        bin_chroms.push_back(chrom);
                        bin_starts.push_back(static_cast<uint32_t>(b * kBinWidth));
                    }
                }

                auto peaks = singlet::call_peaks(bin_agg, bin_chroms, bin_starts);
                if (singlet::write_peaks_bed(out_prefix + "/peaks.bed", peaks))
                    std::cerr << "[singlet] ATAC peaks: " << peaks.size()
                              << " peaks called, wrote peaks.bed\n";
                else
                    std::cerr << "[singlet] ATAC peaks: write_peaks_bed failed\n";
            }

            // Cleanup temp BAM + BAI
            unlink(parallel_bam_path.c_str());
            unlink(parallel_bai_path.c_str());

            // Cleanup .1fq temp FASTQ files
            if (onefq_mode) {
                std::string tmpdir2 = "/dev/shm";
                if (const char* td = std::getenv("TMPDIR")) tmpdir2 = td;
                std::string dec_dir2 = tmpdir2 + "/singlet_1fq_" + std::to_string(getpid());
                unlink((dec_dir2 + "/R1.fastq").c_str());
                unlink((dec_dir2 + "/R2.fastq").c_str());
                rmdir(dec_dir2.c_str());
            }

            std::cerr << "[singlet] ATAC: " << counter.total_bins() << " bins × "
                      << n_cells << " cells, " << atac_sw.elapsed_s() << "s\n";
            std::cerr << "[singlet] ATAC outputs: " << out_prefix << "\n";
            std::cerr << "[singlet] Total: " << total_sw.elapsed_s() << "s\n";
            // Nonhost EM screening: screen STAR unmapped reads
            if (!nonhost_db_path.empty() || !bacterial_db_path.empty() || !fungal_db_path.empty())
                run_nonhost_em_screening(nonhost_db_path, bacterial_db_path, fungal_db_path,
                                         nonhost_ref_dir, out_prefix, {}, 0, 0, star_threads);
            return 0;
        }

        // ── 2c-SS2: Smart-seq2 read counting (plate mode, one sample = one cell) ──
        if (is_smart_seq2_mode) {
            std::cerr << "[singlet] Smart-seq2: counting reads per gene from sorted BAM...\n";
            StopWatch ss2_sw;

            // Load gene model (GTF already validated above)
            singlet::GeneModel ss2_gm;
            if (!ss2_gm.load_gtf(config.exon_gtf_path)) {
                std::cerr << "ERROR: Smart-seq2: failed to load GTF: " << config.exon_gtf_path << "\n";
                return 1;
            }

            const uint32_t n_genes = static_cast<uint32_t>(ss2_gm.gene_ids().size());
            std::vector<uint32_t> gene_counts(n_genes, 0);
            uint64_t total_reads = 0, counted_reads = 0, ambiguous_reads = 0;

            // Scan sorted BAM: for each primary alignment, find overlapping exons → gene
            samFile*   bam_fp  = sam_open(parallel_bam_path.c_str(), "rb");
            bam_hdr_t* bam_hdr_ss2 = sam_hdr_read(bam_fp);
            bam1_t*    rec     = bam_init1();

            while (sam_read1(bam_fp, bam_hdr_ss2, rec) >= 0) {
                // Skip secondary / supplementary / unmapped / QC-fail
                if (rec->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY |
                                      BAM_FUNMAP | BAM_FQCFAIL)) continue;
                // Only count R2 (read2) for pairedend — or all reads if single-end
                // For full-length RNA (Smart-seq2) both mates cover cDNA: count each once
                total_reads++;

                int32_t tid   = rec->core.tid;
                int32_t start = rec->core.pos;
                int32_t end   = bam_endpos(rec);

                // NH filter: skip multi-mappers
                uint8_t* nh_tag = bam_aux_get(rec, "NH");
                if (nh_tag && bam_aux2i(nh_tag) > 1) continue;

                // Query exons overlapping this read alignment
                std::vector<uint32_t> exon_hits;
                ss2_gm.query_exons(tid, start, end, exon_hits);
                if (exon_hits.empty()) continue;

                // Collect unique genes hit
                std::vector<uint32_t> genes_hit;
                for (uint32_t ei : exon_hits) {
                    uint32_t gi = ss2_gm.exon_to_gene(ei);
                    if (genes_hit.empty() || genes_hit.back() != gi) {
                        bool dup = false;
                        for (uint32_t x : genes_hit) if (x == gi) { dup = true; break; }
                        if (!dup) genes_hit.push_back(gi);
                    }
                }

                if (genes_hit.size() == 1) {
                    // Unambiguous: exactly one gene
                    gene_counts[genes_hit[0]]++;
                    counted_reads++;
                } else if (genes_hit.size() > 1) {
                    ambiguous_reads++;
                    // featureCounts default: discard ambiguous multi-gene reads
                }
            }

            bam_destroy1(rec);
            bam_hdr_destroy(bam_hdr_ss2);
            sam_close(bam_fp);

            // Build CSC (single-column: one cell = this sample)
            // Features = genes with count > 0
            std::vector<int32_t> indptr(2, 0);
            std::vector<int32_t> indices;
            std::vector<uint32_t> data;
            for (uint32_t gi = 0; gi < n_genes; ++gi) {
                if (gene_counts[gi] > 0) {
                    indices.push_back(static_cast<int32_t>(gi));
                    data.push_back(gene_counts[gi]);
                }
            }
            indptr[1] = static_cast<int32_t>(data.size());

            // Sample name: derive from .1fq filename stem
            std::string sample_name = onefq_file;
            {
                auto slash = sample_name.rfind('/');
                if (slash != std::string::npos) sample_name = sample_name.substr(slash + 1);
                auto dot = sample_name.rfind('.');
                if (dot != std::string::npos) sample_name = sample_name.substr(0, dot);
            }

            singlet::write_mtx(out_prefix + "/exon_counts.mtx",
                               n_genes, 1,
                               indptr, indices, data);
            singlet::write_names(out_prefix + "/exon_counts_features.tsv", ss2_gm.gene_ids());
            singlet::write_names(out_prefix + "/exon_counts_barcodes.tsv", {sample_name});

            // Variant calling (Smart-seq2)
            if (variant_calling) {
                std::cerr << "[singlet] Smart-seq2: running RNA variant caller...\n";
                singlet::RNAVariantCaller vc_ss2;
                vc_ss2.set_min_coverage(vc_min_coverage);
                vc_ss2.set_min_alt_count(vc_min_alt_count);
                vc_ss2.set_min_vaf(vc_min_vaf);
                vc_ss2.process_bam(parallel_bam_path);
                vc_ss2.write_vcf(out_prefix + "/variants.vcf", vc_ss2.chrom_names());
                vc_ss2.write_tsv(out_prefix + "/variants.tsv", vc_ss2.chrom_names());
                std::cerr << "[singlet] Smart-seq2: " << vc_ss2.variants_called()
                          << " variants called at " << vc_ss2.positions_examined()
                          << " positions\n";
            }

            // Dedup stats (Smart-seq2)
            if (do_dedup_stats) {
                std::cerr << "[singlet] Smart-seq2: collecting dedup stats...\n";
                singlet::ReadDedupStats dedup_ss2;
                singlet::ReadDedupStats::Config dedup_cfg;
                dedup_cfg.optical_distance = optical_distance;
                dedup_ss2.set_config(dedup_cfg);
                auto dr = dedup_ss2.process_bam(parallel_bam_path);
                std::string dedup_tsv = out_prefix + "/dedup_stats.tsv";
                { std::ofstream out(dedup_tsv);
                  out << "total_reads\t"            << dr.total_reads            << "\n"
                      << "mapped_reads\t"           << dr.mapped_reads           << "\n"
                      << "duplicate_reads\t"        << dr.duplicate_reads        << "\n"
                      << "optical_duplicates\t"     << dr.optical_duplicates     << "\n"
                      << "pcr_duplicates\t"         << dr.pcr_duplicates         << "\n"
                      << "duplication_rate\t"       << dr.duplication_rate       << "\n"
                      << "estimated_library_size\t" << dr.estimated_library_size << "\n"
                      << "unique_rate\t"            << dr.unique_rate            << "\n"; }
                std::cerr << "[dedup] " << dr.duplicate_reads << "/" << dr.mapped_reads
                          << " duplicates (" << (dr.duplication_rate*100) << "%), library size ~"
                          << dr.estimated_library_size << "\n";
            }

            // Cleanup
            unlink(parallel_bam_path.c_str());
            unlink(parallel_bai_path.c_str());
            if (onefq_mode) {
                std::string tmpdir_ss2 = "/dev/shm";
                if (const char* td = std::getenv("TMPDIR")) tmpdir_ss2 = td;
                std::string dec_dir_ss2 = tmpdir_ss2 + "/singlet_1fq_" + std::to_string(getpid());
                unlink((dec_dir_ss2 + "/R1.fastq").c_str());
                unlink((dec_dir_ss2 + "/R2.fastq").c_str());
                rmdir(dec_dir_ss2.c_str());
            }

            std::cerr << "[singlet] Smart-seq2: " << total_reads << " total, "
                      << counted_reads << " counted, "
                      << ambiguous_reads << " ambiguous, "
                      << data.size() << " genes expressed, "
                      << ss2_sw.elapsed_s() << "s\n";
            std::cerr << "[singlet] SS2 outputs: " << out_prefix << "\n";
            std::cerr << "[singlet] Total: " << total_sw.elapsed_s() << "s\n";
            return 0;
        }

        // ── 2c-Bulk: Bulk RNA-seq gene counting from sorted BAM ──
        if (is_bulk_rna_mode) {
            std::cerr << "[singlet] Bulk RNA: counting reads per gene from sorted BAM...\n";
            StopWatch bulk_sw;

            singlet::GeneModel bulk_gm;
            if (!bulk_gm.load_gtf(config.exon_gtf_path)) {
                std::cerr << "ERROR: Bulk RNA: failed to load GTF: " << config.exon_gtf_path << "\n";
                return 1;
            }

            const uint32_t n_genes_bulk = static_cast<uint32_t>(bulk_gm.gene_ids().size());

            // ── Strandedness auto-detect: scan first 10000 unique-mapper spliced reads ──
            // XS:A:+ / XS:A:- gives the transcript-strand of the spliced alignment.
            // For forward-stranded (e.g. TruSeq stranded): R1 maps antisense → flag&16≠0 when XS=+
            // For reverse-stranded (dUTP): R1 maps sense → flag&16=0 when XS=+
            // Heuristic: count (R1 forward on + gene) vs (R1 reverse on + gene) for gene-overlapping reads.
            // strand_mode: 0 = unstranded, 1 = forward (sense), 2 = reverse (antisense = dUTP)
            int strand_mode = 0;
            {
                samFile*   sdet_fp  = sam_open(parallel_bam_path.c_str(), "rb");
                bam_hdr_t* sdet_hdr = sam_hdr_read(sdet_fp);
                bam1_t*    sdet_rec = bam_init1();
                uint64_t sense_hits = 0, antisense_hits = 0, probe_reads = 0;
                while (sam_read1(sdet_fp, sdet_hdr, sdet_rec) >= 0 && probe_reads < 50000) {
                    if (sdet_rec->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY | BAM_FUNMAP | BAM_FQCFAIL)) continue;
                    uint8_t* nh = bam_aux_get(sdet_rec, "NH");
                    if (nh && bam_aux2i(nh) > 1) continue;
                    uint8_t* xs = bam_aux_get(sdet_rec, "XS");
                    if (!xs) continue;  // non-spliced read: skip for strand probe
                    char xs_strand = bam_aux2A(xs);
                    bool read_reverse = (sdet_rec->core.flag & BAM_FREVERSE) != 0;
                    bool is_read2    = (sdet_rec->core.flag & BAM_FREAD2)   != 0;
                    // For paired-end: flip expected sense for R2 (opposite orientation of R1)
                    if (is_read2) read_reverse = !read_reverse;
                    // sense = R1 forward on + strand transcript (or R1 reverse on - strand)
                    bool is_sense = (xs_strand == '+') ? !read_reverse : read_reverse;
                    if (is_sense) sense_hits++; else antisense_hits++;
                    probe_reads++;
                }
                bam_destroy1(sdet_rec);
                bam_hdr_destroy(sdet_hdr);
                sam_close(sdet_fp);
                uint64_t total_strand = sense_hits + antisense_hits;
                if (total_strand >= 500) {
                    double sense_frac = static_cast<double>(sense_hits) / total_strand;
                    if (sense_frac > 0.65)        { strand_mode = 1; }  // forward-stranded
                    else if (sense_frac < 0.35)   { strand_mode = 2; }  // reverse-stranded (dUTP)
                    else                           { strand_mode = 0; }  // unstranded
                }
                const char* strand_names[] = {"unstranded", "forward", "reverse(dUTP)"};
                std::cerr << "[singlet] Bulk RNA: strandedness="
                          << strand_names[strand_mode]
                          << " (sense=" << sense_hits << " antisense=" << antisense_hits
                          << " probed=" << probe_reads << ")\n";
            }

            // ── Count reads per gene using detected strandedness ──
            std::vector<uint32_t> bulk_gene_counts(n_genes_bulk, 0);
            uint64_t bulk_total = 0, bulk_counted = 0, bulk_ambiguous = 0, bulk_wrong_strand = 0;

            samFile*   bam_fp_bulk  = sam_open(parallel_bam_path.c_str(), "rb");
            bam_hdr_t* bam_hdr_bulk = sam_hdr_read(bam_fp_bulk);
            bam1_t*    rec_bulk     = bam_init1();

            while (sam_read1(bam_fp_bulk, bam_hdr_bulk, rec_bulk) >= 0) {
                if (rec_bulk->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY | BAM_FUNMAP | BAM_FQCFAIL)) continue;
                bulk_total++;
                uint8_t* nh_bulk = bam_aux_get(rec_bulk, "NH");
                if (nh_bulk && bam_aux2i(nh_bulk) > 1) continue;

                int32_t tid_b   = rec_bulk->core.tid;
                int32_t start_b = rec_bulk->core.pos;
                int32_t end_b   = bam_endpos(rec_bulk);

                std::vector<uint32_t> exon_hits_b;
                bulk_gm.query_exons(tid_b, start_b, end_b, exon_hits_b);
                if (exon_hits_b.empty()) continue;

                std::vector<uint32_t> genes_hit_b;
                for (uint32_t ei : exon_hits_b) {
                    uint32_t gi = bulk_gm.exon_to_gene(ei);
                    bool dup = false;
                    for (uint32_t x : genes_hit_b) if (x == gi) { dup = true; break; }
                    if (!dup) genes_hit_b.push_back(gi);
                }

                if (genes_hit_b.size() != 1) {
                    if (genes_hit_b.size() > 1) bulk_ambiguous++;
                    continue;
                }

                // Strandedness filter (unstranded = always count)
                if (strand_mode != 0) {
                    char gene_str = bulk_gm.gene_strand(genes_hit_b[0]);
                    if (gene_str != '.') {
                        bool read_rev  = (rec_bulk->core.flag & BAM_FREVERSE) != 0;
                        bool is_r2     = (rec_bulk->core.flag & BAM_FREAD2)   != 0;
                        if (is_r2) read_rev = !read_rev;
                        // for forward-stranded: R1 maps sense → expect read_rev == (gene_str=='-')
                        bool expect_rev = (gene_str == '-');
                        if (strand_mode == 2) expect_rev = !expect_rev;  // dUTP: antisense R1
                        if (read_rev != expect_rev) { bulk_wrong_strand++; continue; }
                    }
                }

                bulk_gene_counts[genes_hit_b[0]]++;
                bulk_counted++;
            }

            bam_destroy1(rec_bulk);
            bam_hdr_destroy(bam_hdr_bulk);
            sam_close(bam_fp_bulk);

            // Build single-column CSC matrix
            std::vector<int32_t> bulk_indptr(2, 0);
            // ── Variant calling (Bulk RNA) ──
            if (variant_calling) {
                std::cerr << "[singlet] Bulk RNA: running RNA variant caller...\n";
                singlet::RNAVariantCaller vc_bulk;
                vc_bulk.set_min_coverage(vc_min_coverage);
                vc_bulk.set_min_alt_count(vc_min_alt_count);
                vc_bulk.set_min_vaf(vc_min_vaf);
                vc_bulk.process_bam(parallel_bam_path);
                vc_bulk.write_vcf(out_prefix + "/variants.vcf", vc_bulk.chrom_names());
                vc_bulk.write_tsv(out_prefix + "/variants.tsv", vc_bulk.chrom_names());
                std::cerr << "[singlet] Bulk RNA: " << vc_bulk.variants_called()
                          << " variants called at " << vc_bulk.positions_examined()
                          << " positions\n";
            }

            // ── Dedup stats (Bulk RNA) ──
            if (do_dedup_stats) {
                std::cerr << "[singlet] Bulk RNA: collecting dedup stats...\n";
                singlet::ReadDedupStats dedup_bulk;
                singlet::ReadDedupStats::Config dedup_cfg_bulk;
                dedup_cfg_bulk.optical_distance = optical_distance;
                dedup_bulk.set_config(dedup_cfg_bulk);
                auto dr = dedup_bulk.process_bam(parallel_bam_path);
                std::string dedup_tsv = out_prefix + "/dedup_stats.tsv";
                { std::ofstream out(dedup_tsv);
                  out << "total_reads\t"            << dr.total_reads            << "\n"
                      << "mapped_reads\t"           << dr.mapped_reads           << "\n"
                      << "duplicate_reads\t"        << dr.duplicate_reads        << "\n"
                      << "optical_duplicates\t"     << dr.optical_duplicates     << "\n"
                      << "pcr_duplicates\t"         << dr.pcr_duplicates         << "\n"
                      << "duplication_rate\t"       << dr.duplication_rate       << "\n"
                      << "estimated_library_size\t" << dr.estimated_library_size << "\n"
                      << "unique_rate\t"            << dr.unique_rate            << "\n"; }
                std::cerr << "[dedup] " << dr.duplicate_reads << "/" << dr.mapped_reads
                          << " duplicates (" << (dr.duplication_rate*100) << "%), library size ~"
                          << dr.estimated_library_size << "\n";
            }

            std::vector<int32_t> bulk_indices;
            std::vector<uint32_t> bulk_data;
            for (uint32_t gi = 0; gi < n_genes_bulk; ++gi) {
                if (bulk_gene_counts[gi] > 0) {
                    bulk_indices.push_back(static_cast<int32_t>(gi));
                    bulk_data.push_back(bulk_gene_counts[gi]);
                }
            }
            bulk_indptr[1] = static_cast<int32_t>(bulk_data.size());

            std::string bulk_sample = onefq_file;
            { auto s = bulk_sample.rfind('/'); if (s != std::string::npos) bulk_sample = bulk_sample.substr(s+1);
              auto d = bulk_sample.rfind('.'); if (d != std::string::npos) bulk_sample = bulk_sample.substr(0, d); }

            singlet::write_mtx(out_prefix + "/exon_counts.mtx",
                               n_genes_bulk, 1, bulk_indptr, bulk_indices, bulk_data);
            singlet::write_names(out_prefix + "/exon_counts_features.tsv", bulk_gm.gene_ids());
            singlet::write_names(out_prefix + "/exon_counts_barcodes.tsv", {bulk_sample});

            unlink(parallel_bam_path.c_str());
            unlink(parallel_bai_path.c_str());
            if (onefq_mode) {
                std::string tmpdir_bulk = "/dev/shm";
                if (const char* td = std::getenv("TMPDIR")) tmpdir_bulk = td;
                std::string dec_dir_bulk = tmpdir_bulk + "/singlet_1fq_" + std::to_string(getpid());
                unlink((dec_dir_bulk + "/R1.fastq").c_str());
                unlink((dec_dir_bulk + "/R2.fastq").c_str());
                rmdir(dec_dir_bulk.c_str());
            }

            const char* strand_label[] = {"unstranded", "forward", "reverse"};
            std::cerr << "[singlet] Bulk RNA: " << bulk_total << " total, "
                      << bulk_counted << " counted, "
                      << bulk_ambiguous << " ambiguous, "
                      << bulk_wrong_strand << " wrong-strand, "
                      << bulk_data.size() << " genes expressed, "
                      << "strand=" << strand_label[strand_mode] << ", "
                      << bulk_sw.elapsed_s() << "s\n";
            std::cerr << "[singlet] Bulk outputs: " << out_prefix << "\n";
            std::cerr << "[singlet] Total: " << total_sw.elapsed_s() << "s\n";
            // Nonhost EM screening: screen STAR unmapped reads
            if (!nonhost_db_path.empty() || !bacterial_db_path.empty() || !fungal_db_path.empty())
                run_nonhost_em_screening(nonhost_db_path, bacterial_db_path, fungal_db_path,
                                         nonhost_ref_dir, out_prefix, {}, 0, 0, star_threads);
            return 0;
        }
        // ── 2b-2: CB-tag barcode refresh ──────────────────────────────────────────
        // After STAR aligns and the BAM is indexed, re-derive the pileup barcode list
        // from STAR's whitelist-corrected CB:Z tags in the sorted BAM.  This corrects
        // the systematic mismatch between raw R1 barcodes (written to auto_barcodes.tsv
        // at decode time) and the whitelist-corrected sequences that STAR emits via the
        // 1MM-correction path:  for reads where raw_barcode ≠ whitelist_barcode (≥1MM),
        // the CB:Z tag contains the whitelist barcode while auto_barcodes.tsv has the
        // raw barcode — the barcode hash-map lookup in the pileup engine then misses,
        // yielding dramatically under-counted barcoded reads.
        if (needs_barcode_discovery && !auto_barcode_file.empty() && whitelist_auto_resolved
            && engine_ptr) {
            StopWatch cb_scan_sw;
            std::unordered_map<std::string, uint32_t> cb_freq;
            cb_freq.reserve(50000);
            samFile*   cbf    = sam_open(parallel_bam_path.c_str(), "rb");
            if (cbf) {
                bam_hdr_t* cbhdr = sam_hdr_read(cbf);
                bam1_t*    crec  = bam_init1();
                while (sam_read1(cbf, cbhdr, crec) >= 0) {
                    if (crec->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY)) continue;
                    uint8_t* cb_aux = bam_aux_get(crec, "CB");
                    if (cb_aux && *cb_aux == 'Z') {
                        const char* cb_str = bam_aux2Z(cb_aux);
                        // Skip STAR's sentinel value for no-whitelist-match reads
                        if (cb_str && cb_str[0] && cb_str[0] != '-') cb_freq[cb_str]++;
                    }
                }
                bam_destroy1(crec);
                bam_hdr_destroy(cbhdr);
                sam_close(cbf);
            }
            // Build sorted barcode list; use a low threshold so shallow samples
            // with few CB-tagged reads per cell are not dropped.
            constexpr uint32_t CB_MIN_READS = 5;
            std::vector<std::pair<uint32_t, std::string>> cb_sorted;
            cb_sorted.reserve(cb_freq.size());
            for (const auto& [bc, cnt] : cb_freq)
                if (cnt >= CB_MIN_READS) cb_sorted.emplace_back(cnt, bc);
            std::sort(cb_sorted.begin(), cb_sorted.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            if (!cb_sorted.empty()) {
                std::ofstream bc_out(auto_barcode_file);
                for (const auto& [cnt, bc] : cb_sorted) bc_out << bc << "\n";
                bc_out.close();
                config.barcode_path = auto_barcode_file;
                engine_ptr->reload_barcodes(auto_barcode_file);
                std::cerr << "[singlet] CB-tag barcode refresh: " << cb_sorted.size()
                          << " barcodes (>=" << CB_MIN_READS << " CB reads) in "
                          << cb_scan_sw.elapsed_s() << "s\n";
            } else {
                std::cerr << "[singlet] CB-tag barcode refresh: no barcodes with >="
                          << CB_MIN_READS << " CB reads; keeping decode-time barcodes\n";
            }
        }

        std::cerr << "[singlet] Pileup: parallel (" << star_threads << " workers) on indexed BAM\n";
        StopWatch pileup_sw;
        stats = engine_ptr->run_parallel(parallel_bam_path, star_threads);
        pileup_time = pileup_sw.elapsed_s();
        std::cerr << "[singlet] Pileup: " << stats.total_reads << " reads, " << pileup_time << "s";
        if (stats.total_reads > 0)
            std::cerr << " (" << static_cast<int>(stats.total_reads / pileup_time / 1e6) << "M reads/s)";
        std::cerr << "\n";

        // Auto-correct strand: if >50% of exonic reads are wrong-strand and the engine
        // didn't already auto-flip during the pass, re-run the pileup on the same BAM
        // with config.reverse_strand=true. This corrects 5' protocol misclassification
        // (e.g. 10x-5p-v2 detected as 10xv3) without requiring a STAR re-alignment.
        // The retry fires at most once (config.reverse_strand guards re-entry).
        if (!config.exon_gtf_path.empty() && stats.exon_hits > 0 && stats.wrong_strand > 0 &&
            !engine_ptr->strand_was_flipped() && !config.reverse_strand) {
            double ws_frac = static_cast<double>(stats.wrong_strand) /
                             (stats.exon_hits + stats.wrong_strand);
            if (ws_frac > 0.50) {
                std::cerr << "[singlet] Auto-correcting strand: "
                          << static_cast<int>(ws_frac * 100)
                          << "% wrong-strand reads detected → re-running pileup with reversed strand\n";
                config.reverse_strand = true;
                engine_ptr = std::make_unique<singlet::PileupEngine>(config);
                if (!engine_ptr->load_references()) {
                    std::cerr << "[singlet] WARNING: Could not reload references for strand retry; "
                              << "keeping initial pileup result\n";
                    config.reverse_strand = false; // revert so exported JSON reflects truth
                } else {
                    StopWatch retry_sw;
                    stats = engine_ptr->run_parallel(parallel_bam_path, star_threads);
                    pileup_time += retry_sw.elapsed_s();
                    std::cerr << "[singlet] Strand-retry pileup: "
                              << stats.total_reads << " reads, "
                              << retry_sw.elapsed_s() << "s\n";
                }
            }
        }

        // ── 2d: Delete temp BAM + BAI ──
        // Skip deletion if --tagged-bam is enabled; the BAM is still needed for
        // the second-pass tagged-BAM writer that runs after pileup export.
        if (!tagged_bam_enabled) {
            unlink(parallel_bam_path.c_str());
            unlink(parallel_bai_path.c_str());
        }
    } else {
        // ── Streaming path: run pileup concurrently with STAR ──
        std::string source_desc = bam_file.empty()
            ? (onefq_mode ? "STAR pipe (.1fq stream)" :
               sra_mode ? "STAR pipe (SRA stream)" : "STAR pipe")
            : "BAM file";
        std::cerr << "[singlet] Pileup: streaming from " << source_desc << "...\n";
        StopWatch pileup_sw;
        stats = engine_ptr->run(bam_source);
        pileup_time = pileup_sw.elapsed_s();
        std::cerr << "[singlet] Pileup: " << stats.total_reads << " reads, "
                  << pileup_time << "s";
        if (stats.total_reads > 0)
            std::cerr << " (" << static_cast<int>(stats.total_reads / pileup_time / 1e6) << "M reads/s)";
        std::cerr << "\n";

        // Diagnostic: warn if wrong_strand fraction is high (possible 5' protocol)
        if (stats.exon_hits > 0 && stats.wrong_strand > 0 &&
            !engine_ptr->strand_was_flipped()) {
            double ws_frac = static_cast<double>(stats.wrong_strand) /
                             (stats.exon_hits + stats.wrong_strand);
            if (ws_frac > 0.35 && !config.reverse_strand) {
                std::cerr << "[singlet] WARNING: " << static_cast<int>(ws_frac * 100)
                          << "% wrong-strand reads detected. This sample may use a 5' capture "
                          << "protocol. Re-run with --reverse-strand for correct gene counting.\n";
            }
        }

        // Wait for STAR
        if (star_pid > 0) {
            int wstatus;
            waitpid(star_pid, &wstatus, 0);
            close(pipe_fd[0]);
            if (WIFEXITED(wstatus)) {
                star_exit = WEXITSTATUS(wstatus);
                if (star_exit != 0)
                    std::cerr << "[singlet] WARNING: STAR exited " << star_exit << "\n";
                else
                    std::cerr << "[singlet] STAR completed\n";
            } else if (WIFSIGNALED(wstatus)) {
                std::cerr << "[singlet] ERROR: STAR killed by signal " << WTERMSIG(wstatus) << "\n";
                if (fifo_writer_thread.joinable()) fifo_writer_thread.join();
                return 1;
            }
        }
        // Join FIFO writer thread (if FIFO decode mode) after STAR completes
        if (fifo_writer_thread.joinable()) fifo_writer_thread.join();
    }

    // ── BUG-MAPPING-GUARD: Warn on very low mapping rate (wrong genome) ──
    // Parse STAR's Log.final.out for uniquely mapped reads %.
    // This catches wrong-genome runs that silently produce empty output.
    if (star_pid > 0 || !bam_file.empty()) {
        std::string log_path = out_prefix + "/star_Log.final.out";
        if (FILE* lf = std::fopen(log_path.c_str(), "r")) {
            char line[256];
            while (std::fgets(line, sizeof(line), lf)) {
                if (std::strstr(line, "Uniquely mapped reads %")) {
                    // Format: "   Uniquely mapped reads % |	0.00%" (tab or spaces after |)
                    const char* pipe = std::strrchr(line, '|');
                    if (pipe) {
                        double rate = std::atof(pipe + 1);
                        if (rate < 1.0) {
                            std::cerr << "[singlet] WARNING: Very low mapping rate ("
                                      << std::fixed << std::setprecision(2) << rate
                                      << "%). Possible wrong reference genome.\n";
                        }
                    }
                    break;
                }
            }
            std::fclose(lf);
        }
    }

    // ── Wait for SRA demuxer + cleanup FIFOs ──
    int demux_exit = 0;
    if (demux_pid > 0) {
        int wstatus;
        waitpid(demux_pid, &wstatus, 0);
        if (WIFEXITED(wstatus)) {
            demux_exit = WEXITSTATUS(wstatus);
            if (demux_exit != 0)
                std::cerr << "[singlet] WARNING: SRA demuxer exited " << demux_exit << "\n";
            else
                std::cerr << "[singlet] SRA demuxer completed\n";
        }
        // Cleanup FIFOs and temp dir
        unlink(r1_fifo_path.c_str());
        unlink(r2_fifo_path.c_str());
        rmdir(fifo_dir.c_str());
    }

    // Cleanup .1fq temp FASTQ files
    if (onefq_mode) {
        std::string tmpdir = "/dev/shm";
        if (const char* td = std::getenv("TMPDIR"))
            tmpdir = td;
        std::string decode_dir = tmpdir + "/singlet_1fq_" + std::to_string(getpid());
        unlink((decode_dir + "/R1.fastq").c_str());
        unlink((decode_dir + "/R2.fastq").c_str());
        rmdir(decode_dir.c_str());
    }

    if (stats.total_reads == 0) {
        std::cerr << "[singlet] ERROR: No reads processed\n";
        return 1;
    }

    // ── NONHOST-UNMAPPED-CAPTURE: Screen STAR unmapped reads (EM deconvolution) ─
    // When any nonhost index is provided, STAR wrote unmapped cDNA reads to
    // ${out_prefix}/star_Unmapped.out.mate1 (mate1 = R2 = cDNA, per readFilesIn order).
    // run_nonhost_em_screening() classifies reads with soft multi-hit threshold,
    // runs EM deconvolution, and writes nonhost_em_abundance.tsv + nonhost_summary.json.
    // Visium flows through this path (no early return above).
    if (!nonhost_db_path.empty() || !bacterial_db_path.empty() || !fungal_db_path.empty()) {
        std::unordered_set<std::string> nonhost_valid_bcs;
        if (engine_ptr) {
            const auto& bc_vec = engine_ptr->barcodes();
            nonhost_valid_bcs.insert(bc_vec.begin(), bc_vec.end());
        }
        run_nonhost_em_screening(nonhost_db_path, bacterial_db_path, fungal_db_path,
            nonhost_ref_dir, out_prefix, nonhost_valid_bcs, detected_bc_len, umi_len, star_threads);
    }

    // ── Phase 3: Export ──
    std::cerr << "[singlet] Exporting feature matrices...\n";

    singlet::ExportConfig export_cfg;
    export_cfg.out_prefix = out_prefix;
    export_cfg.output_format = output_format;
    export_cfg.pipeline_mode = pipeline_mode;
    export_cfg.n_donors = n_donors;
    export_cfg.threads = config.threads;
    export_cfg.genome_dir = genome_dir;

    // ── §3.6 fields for summary.json ──
    {
        auto get_proto_name = [](uint8_t id) -> const char* {
            switch (id) {
                case 0: return "unknown"; case 1: return "10xv3"; case 2: return "10xv2";
                case 3: return "10xv1"; case 4: return "10xv3-5p"; case 5: return "10xv2-5p";
                case 6: return "dropseq"; case 7: return "celseq2"; case 8: return "marsseq2";
                case 9: return "sci-rna-seq3"; case 10: return "bd-rhapsody";
                case 11: return "splitseq"; case 12: return "indrop";
                case 13: return "10xv4"; case 14: return "10xv4-5p";
                case 15: return "dnbelab-c4"; case 16: return "seqwell";
                case 17: return "ddseq"; case 18: return "quartzseq2";
                case 19: return "microwell-seq"; case 20: return "surecell";
                case 21: return "strtseq"; case 22: return "10x-arc-gex";
                case 23: return "10x-atac"; case 24: return "10x-visium";
                default: return "other";
            }
        };
        export_cfg.protocol_id   = static_cast<int>(onefq_protocol_id);
        export_cfg.protocol_name = get_proto_name(onefq_protocol_id);
        // reference_build + species from genome_dir path
        std::string gb = genome_dir;
        while (!gb.empty() && gb.back() == '/') gb.pop_back();
        auto p1 = gb.rfind('/');
        std::string leaf = (p1 != std::string::npos) ? gb.substr(p1+1) : gb;
        if (leaf.find("star") != std::string::npos || leaf.find("STAR") != std::string::npos) {
            gb = (p1 != std::string::npos) ? gb.substr(0, p1) : gb;
            auto p2 = gb.rfind('/');
            leaf = (p2 != std::string::npos) ? gb.substr(p2+1) : gb;
        }
        export_cfg.reference_build = leaf;
        std::string lb = leaf;
        for (char& c : lb) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if      (lb.find("grch") != std::string::npos || lb.find("hg") != std::string::npos)
            export_cfg.species = "human";
        else if (lb.find("grcm") != std::string::npos || lb.find("mm") != std::string::npos)
            export_cfg.species = "mouse";
        // Peak RSS
        {
            std::ifstream proc_f("/proc/self/status");
            std::string pln;
            while (std::getline(proc_f, pln)) {
                if (pln.rfind("VmPeak:", 0) == 0) {
                    const char* pp = pln.c_str() + 7;
                    while (*pp == ' ' || *pp == '\t') ++pp;
                    export_cfg.peak_rss_gb = std::strtod(pp, nullptr) / (1024.0 * 1024.0);
                    break;
                }
            }
        }
        if      (export_cfg.peak_rss_gb < 8.0)   export_cfg.memory_tier = "Small";
        else if (export_cfg.peak_rss_gb < 64.0)  export_cfg.memory_tier = "Medium";
        else if (export_cfg.peak_rss_gb < 256.0) export_cfg.memory_tier = "Large";
        else                                      export_cfg.memory_tier = "XLarge";
        export_cfg.nonhost_was_screened = (!nonhost_db_path.empty() ||
                                           !bacterial_db_path.empty() ||
                                           !fungal_db_path.empty());
    }

    // ── N8: Populate provenance manifest ──
    export_cfg.provenance.singlet_version      = "0.3.0";
    export_cfg.provenance.input_file            = onefq_file.empty() ? sra_file : onefq_file;
    export_cfg.provenance.input_reads           = stats.total_reads;
    export_cfg.provenance.genome_dir            = genome_dir;
    export_cfg.provenance.gtf_path              = config.exon_gtf_path;
    export_cfg.provenance.threads               = star_threads;
    export_cfg.provenance.umi_dedup             = config.umi_dedup;
    export_cfg.provenance.umi_dedup_directional = config.umi_dedup_directional;
    export_cfg.provenance.pipeline              = pipeline_mode;
    export_cfg.provenance.wall_seconds          = total_sw.elapsed_s();
    export_cfg.provenance.pileup_seconds        = pileup_time;
    export_cfg.provenance.snp_vcf_path          = config.snp_path;
    export_cfg.provenance.whitelist_name        = whitelist_file;
    export_cfg.provenance.cascade_enabled       = cascade_enabled;
    export_cfg.provenance.cascade_mode          = cascade_mode;
    export_cfg.provenance.te_classify_mode      = te_classify_mode;

    // N5: emit cell calling when --barcodes was NOT explicitly provided (auto mode)
    bool do_cell_calling = (cell_calling_flag == 1) ||
        (cell_calling_flag == -1 && !user_provided_barcodes);
    export_cfg.run_cell_calling = do_cell_calling && !config.exon_gtf_path.empty();
    export_cfg.fdr_threshold = fdr_threshold_flag;
    export_cfg.lower_umi = static_cast<uint64_t>(lower_umi_flag);
    export_cfg.forced_cells = forced_cells_flag;
    export_cfg.expect_cells = expect_cells_flag;
    export_cfg.write_raw_matrix = write_raw_matrix;
    export_cfg.pileup_stats = &stats;

    // §3.2: Populate per-stage read statistics for read_stats.tsv
    {
        using SS = singlet::PipelineStageStats;
        // Encode stage: .1fq decode → FASTQ (file-mode only; FIFO overlaps with align)
        if (decode_reads_out > 0 || decode_time > 0) {
            SS enc;
            enc.stage = "encode";
            enc.reads_in = decode_reads_out;  // .1fq contains this many read pairs
            enc.reads_out = decode_reads_out;  // all decoded successfully
            enc.wall_seconds = decode_time;
            export_cfg.stage_stats.push_back(std::move(enc));
        }
        // Align stage: STAR genome alignment
        {
            SS aln;
            aln.stage = "align";
            aln.reads_in = decode_reads_out > 0 ? decode_reads_out : stats.total_reads;
            aln.reads_out = stats.mapped_reads;
            if (stats.total_reads > 0 && stats.mapped_reads == 0)
                aln.fail_reason = "unmapped";
            aln.wall_seconds = star_time;
            export_cfg.stage_stats.push_back(std::move(aln));
        }
        // Pileup stage: BAM → per-cell count matrices
        {
            SS pil;
            pil.stage = "pileup";
            pil.reads_in = stats.total_reads;
            pil.reads_out = stats.mapped_reads;  // reads assigned to genes/cells
            pil.wall_seconds = pileup_time;
            export_cfg.stage_stats.push_back(std::move(pil));
        }
    }

    // Load --metadata-json into export_cfg.user_meta (embedded in every .1pz)
    if (!metadata_json_path.empty()) {
        std::ifstream mf(metadata_json_path);
        if (!mf.is_open()) {
            std::cerr << "ERROR: cannot open metadata JSON: " << metadata_json_path << "\n";
            return 1;
        }
        // Simple line-by-line key-value parser for flat JSON objects.
        // Handles: "key": "value" (string values only; non-string values treated as strings).
        // Does not require nlohmann/json — keeps the dependency footprint minimal.
        std::string line;
        while (std::getline(mf, line)) {
            // Find first '"' for key start
            auto k0 = line.find('"');
            if (k0 == std::string::npos) continue;
            auto k1 = line.find('"', k0 + 1);
            if (k1 == std::string::npos) continue;
            std::string key = line.substr(k0 + 1, k1 - k0 - 1);
            if (key.empty()) continue;
            // Find ':' separator
            auto colon = line.find(':', k1 + 1);
            if (colon == std::string::npos) continue;
            // Value: strip leading/trailing whitespace; strip surrounding quotes if present
            std::string raw = line.substr(colon + 1);
            // trim whitespace
            size_t vs = raw.find_first_not_of(" \t\r\n");
            if (vs == std::string::npos) continue;
            size_t ve = raw.find_last_not_of(" \t\r\n,");
            raw = raw.substr(vs, ve - vs + 1);
            std::string value;
            if (!raw.empty() && raw.front() == '"') {
                // quoted string: strip outer quotes (no escape handling needed for GEO metadata)
                size_t q1 = 1;
                size_t q2 = raw.rfind('"');
                value = (q2 > q1) ? raw.substr(q1, q2 - q1) : raw.substr(q1);
            } else {
                // number/boolean/null: store as string as-is
                value = raw;
                // strip trailing comma
                if (!value.empty() && value.back() == ',') value.pop_back();
            }
            export_cfg.user_meta[key] = value;
        }
        std::cerr << "[singlet] Loaded " << export_cfg.user_meta.size()
                  << " metadata fields from " << metadata_json_path << "\n";
    }

    auto export_stats = singlet::export_results(*engine_ptr, config, export_cfg);

    // ── T4: CITE-seq ADT/HTO processing ──────────────────────────────────────
    // After GEX export, do a second pass through the .1fq to match ADT tags.
    // For CITE_SEQ_ADT assay: R2 = antibody tag read; uses current reader (R1+R2).
    // For CITE_SEQ_GEX assay with 3-stream bundle: feature stream decoding
    // requires future lib1fq reader extension (tracked in ROADMAP.md T4-EXT).
    if (is_cite_seq_mode && !feature_ref_path.empty() && onefq_mode) {
        std::cerr << "[singlet] CITE-seq: processing ADT/HTO reads from " << feature_ref_path << "\n";
        StopWatch adt_sw;

        // ── Load tag reference ─────────────────────────────────────────────
        singlet::AdtMatcher matcher;
        if (!matcher.load_reference(feature_ref_path)) {
            std::cerr << "[singlet] ERROR: Failed to load feature reference: "
                      << feature_ref_path << "\n";
        } else {
            matcher.build_index();
            std::cerr << "[singlet] CITE-seq: loaded " << matcher.num_tags()
                      << " tags from " << feature_ref_path << "\n";

            // ── Barcode → index map (from GEX barcodes) ────────────────────
            const auto& gex_barcodes = engine_ptr->barcodes();
            std::unordered_map<uint64_t, uint32_t> bc_dict_idx;
            bc_dict_idx.reserve(gex_barcodes.size() * 2);
            size_t n_cells = gex_barcodes.size();

            // ── Second pass: decode .1fq, match tags, count ────────────────
            singlet::AdtCounter counter;
            counter.set_num_tags(matcher.num_tags());
            counter.set_num_cells(n_cells);

            // Helper: UMI string from packed bytes
            auto umi_str = [](const uint8_t* seq, uint16_t len) -> std::string {
                static constexpr char B2C[5] = {'A','C','G','T','N'};
                std::string s(len, 'N');
                for (uint16_t j = 0; j < len; ++j)
                    s[j] = B2C[seq[j] < 5 ? seq[j] : 4];
                return s;
            };
            // Helper: ASCII tag string from packed bytes
            auto tag_str = [](const uint8_t* seq, uint16_t len) -> std::string {
                static constexpr char B2C[5] = {'A','C','G','T','N'};
                std::string s(len, 'N');
                for (uint16_t j = 0; j < len; ++j)
                    s[j] = B2C[seq[j] < 5 ? seq[j] : 4];
                return s;
            };

            {
                lib1fq::Reader adt_reader;
                adt_reader.open(onefq_file.c_str());
                const auto& adt_hdr = adt_reader.header();
                uint16_t bc_len = adt_reader.bc_length();
                uint16_t umi_len_adt = static_cast<uint16_t>(adt_reader.umi_length());
                if (umi_len_adt == 0) umi_len_adt = static_cast<uint16_t>(umi_len);

                // Build BC-dict → barcode string map for lookup
                // (same dict as GEX pass; bc_indices in block map to barcodes)
                std::vector<std::string> bc_dict_strings;
                if (adt_reader.has_bc_dict()) {
                    const auto& flat = adt_reader.bc_dict_flat();
                    size_t dict_n = adt_reader.bc_dict_size();
                    bc_dict_strings.reserve(dict_n);
                    for (size_t di = 0; di < dict_n; ++di) {
                        const uint8_t* bcp = flat.data() + di * bc_len;
                        static constexpr char B2C[5] = {'A','C','G','T','N'};
                        std::string bc(bc_len, 'N');
                        for (uint16_t j = 0; j < bc_len; ++j)
                            bc[j] = B2C[bcp[j] < 5 ? bcp[j] : 4];
                        bc_dict_strings.push_back(bc);
                    }
                }

                // Build barcode → cell_index map from GEX barcodes
                std::unordered_map<std::string, uint32_t> bc_to_idx;
                bc_to_idx.reserve(n_cells * 2);
                for (size_t ci = 0; ci < n_cells; ++ci)
                    bc_to_idx[gex_barcodes[ci]] = static_cast<uint32_t>(ci);

                lib1fq::DecodedBlock adt_blk;
                while (adt_reader.read_block(adt_blk)) {
                    for (uint32_t i = 0; i < adt_blk.n_reads; ++i) {
                        // Get barcode string
                        std::string bc_seq;
                        if (!bc_dict_strings.empty() && !adt_blk.bc_indices.empty()) {
                            uint32_t bc_i = adt_blk.bc_indices[i];
                            if (bc_i < bc_dict_strings.size())
                                bc_seq = bc_dict_strings[bc_i];
                        }
                        if (bc_seq.empty()) {
                            const uint8_t* r1s = adt_blk.r1_seq(i);
                            uint16_t r1l = adt_blk.r1_len(i);
                            uint16_t blk_bc = (r1l > umi_len_adt) ? (r1l - umi_len_adt) : r1l;
                            static constexpr char B2C[5] = {'A','C','G','T','N'};
                            bc_seq.reserve(blk_bc);
                            for (uint16_t j = 0; j < blk_bc; ++j)
                                bc_seq += B2C[r1s[j] < 5 ? r1s[j] : 4];
                        }

                        auto bc_it = bc_to_idx.find(bc_seq);
                        if (bc_it == bc_to_idx.end()) continue;  // not a valid cell

                        // Get UMI
                        const uint8_t* r1s = adt_blk.r1_seq(i);
                        uint16_t r1l = adt_blk.r1_len(i);
                        uint16_t blk_bc = (r1l > umi_len_adt) ? (r1l - umi_len_adt) : r1l;
                        std::string umi = umi_str(r1s + blk_bc, umi_len_adt);

                        // Get tag sequence (R2)
                        const uint8_t* r2s = adt_blk.r2_seq(i);
                        uint16_t r2l = adt_blk.r2_len(i);
                        std::string tag_seq = tag_str(r2s, r2l);

                        int tag_id = matcher.match(tag_seq.c_str(), static_cast<int>(tag_seq.size()));
                        if (tag_id >= 0) {
                            counter.add(bc_it->second, tag_id, umi);
                        }
                    }
                }
            }

            counter.finalize();

            // ── Export ────────────────────────────────────────────────────
            auto adt_csc = counter.counts().to_csc();
            singlet::write_mtx(out_prefix + "/adt_counts.mtx",
                               static_cast<uint32_t>(matcher.num_tags()),
                               static_cast<uint32_t>(n_cells),
                               adt_csc.indptr, adt_csc.indices, adt_csc.data);
            singlet::write_names(out_prefix + "/adt_features.tsv", matcher.names());
            singlet::write_names(out_prefix + "/adt_barcodes.tsv", gex_barcodes);

            std::cerr << "[singlet] CITE-seq: matched=" << matcher.matched()
                      << " ambiguous=" << matcher.ambiguous()
                      << " unmatched=" << matcher.unmatched() << "\n";

            // ── HTO demux (if all tags look like hashtags) ─────────────────
            // Heuristic: treat as HTO if tag name prefix is "HTO" or "Hashtag"
            bool is_hto = false;
            if (matcher.num_tags() >= 2) {
                size_t hto_name_count = 0;
                for (const auto& tag : matcher.tags()) {
                    std::string ln = tag.name;
                    for (auto& c : ln) c = static_cast<char>(
                        std::tolower(static_cast<unsigned char>(c)));
                    if (ln.find("hto")     != std::string::npos ||
                        ln.find("hashtag") != std::string::npos)
                        ++hto_name_count;
                }
                is_hto = (hto_name_count == matcher.num_tags());
            }
            if (is_hto) {
                std::cerr << "[singlet] CITE-seq: HTO detected — running demux\n";
                singlet::HtoDemux demux;
                auto hto_results = demux.demux(counter.counts(),
                                               matcher.names(),
                                               n_cells);
                demux.write_tsv(out_prefix + "/hto_classification.tsv",
                                gex_barcodes, hto_results);
                std::cerr << "[singlet] CITE-seq: HTO classification written\n";
            }

            std::cerr << "[singlet] CITE-seq: ADT/HTO done in " << adt_sw.elapsed_s() << "s\n";
        }
    }

    // ── V2+V3: Visium spatial coordinate integration + QC ──────────────────
    // When the assay is SPATIAL_RNA (Visium), wire spatial coordinates from
    // tissue_positions.csv into the output and emit per-spot QC metrics.
    // The pileup engine already counted genes per barcode (spot); V2 just
    // annotates those barcodes with spatial metadata. No new counting needed.
    if (is_visium_mode && engine_ptr) {
        const auto& barcodes = engine_ptr->barcodes();
        singlet::VisiumSpatialParser spatial_parser;
        bool positions_loaded = false;

        if (!tissue_positions_path.empty()) {
            positions_loaded = spatial_parser.load_positions(tissue_positions_path);
            if (!positions_loaded)
                std::cerr << "[visium] WARNING: could not load tissue_positions.csv: "
                          << tissue_positions_path << "\n";
        }

        if (!positions_loaded) {
            // No positions file: still write spatial_coordinates.tsv with empty coords.
            std::cerr << "[visium] No tissue_positions.csv provided (use --tissue-positions);"
                         " spatial coordinates will be zero-filled.\n";
        } else {
            std::cerr << "[visium] Loaded " << spatial_parser.total_spots()
                      << " spots (" << spatial_parser.in_tissue() << " in-tissue) from "
                      << tissue_positions_path << "\n";
        }

        // V2: write spatial_coordinates.tsv
        spatial_parser.write_coordinates_tsv(
            out_prefix + "/spatial_coordinates.tsv", barcodes);
        std::cerr << "[visium] Spatial coordinates written: "
                  << out_prefix << "/spatial_coordinates.tsv\n";

        // V3: build per-spot QC and compute summary
        // Compute per-cell metrics from the engine's exon/intron accumulators.
        // This mirrors what export_results does for cell_qc_metrics.tsv, but
        // we need the vectors here to annotate spots with UMI counts.
        std::vector<uint32_t> spot_umis, spot_genes;
        std::vector<float>    spot_mt;
        if (!config.exon_gtf_path.empty()
            && engine_ptr->gene_model().has_gene_hierarchy()
            && engine_ptr->exons().nnz_raw() > 0) {
            auto ex_csc  = engine_ptr->exons().to_csc();
            auto in_csc  = config.count_introns
                ? engine_ptr->introns().to_csc()
                : singlet::SparseAccumulator<uint16_t>::CSCMatrix{};
            auto qc = singlet::compute_cell_qc(ex_csc, in_csc,
                                               engine_ptr->gene_model());
            spot_umis  = std::move(qc.total_umis);
            spot_genes = std::move(qc.total_genes);
            spot_mt    = std::move(qc.mt_pct);
        }

        auto spot_records = singlet::build_spot_qc(
            barcodes, spot_umis, spot_genes, spot_mt, spatial_parser);
        auto qc_summary = singlet::compute_visium_summary(spot_records, visium_min_umi);

        singlet::write_visium_qc_tsv(out_prefix + "/visium_qc.tsv", spot_records);
        singlet::log_visium_summary(qc_summary);
        std::cerr << "[visium] Per-spot QC written: " << out_prefix << "/visium_qc.tsv\n";
    }

    // ── G-TAGGED-BAM: second-pass tagged BAM output (post-pileup) ──────────
    // Reads the coordinate-sorted STAR BAM from disk, adds CB/UB/GX/GN/RE/xf
    // tags to every primary mapped read, and writes a Cell Ranger-compatible
    // output BAM with optional .bai index.
    // Requires parallel_pileup=1 (BAM on disk); skipped with a warning otherwise.
    if (tagged_bam_enabled && engine_ptr) {
        if (tagged_bam_path.empty())
            tagged_bam_path = out_prefix + "/possorted_genome_bam.bam";

        if (parallel_bam_path.empty()) {
            std::cerr << "[tagged_bam] WARNING: --tagged-bam requires parallel pileup mode"
                         " (BAM on disk). Skipping tagged BAM output.\n"
                         "             Ensure /dev/shm has ≥10 GB free, or pass"
                         " --parallel-pileup explicitly.\n";
        } else {
            std::cerr << "[tagged_bam] Writing: " << tagged_bam_path << "\n";
            StopWatch tbam_sw;

            samFile*   src_fp  = sam_open(parallel_bam_path.c_str(), "rb");
            sam_hdr_t* src_hdr = src_fp ? sam_hdr_read(src_fp) : nullptr;

            if (!src_fp || !src_hdr) {
                std::cerr << "[tagged_bam] ERROR: cannot open STAR BAM: "
                          << parallel_bam_path << "\n";
            } else {
                // Build called-barcode set from pileup engine's corrected barcodes
                std::unordered_set<std::string> called_set(
                    engine_ptr->barcodes().begin(),
                    engine_ptr->barcodes().end());

                singlet::TaggedBamConfig tbcfg;
                tbcfg.output_path      = tagged_bam_path;
                tbcfg.add_index        = true;
                tbcfg.threads          = std::min(star_threads, 4);
                tbcfg.called_barcodes  = &called_set;

                singlet::TaggedBamWriter writer;
                if (!writer.open(tbcfg, src_hdr)) {
                    std::cerr << "[tagged_bam] ERROR: cannot create output BAM: "
                              << tagged_bam_path << "\n";
                } else {
                    const singlet::GeneModel& gm = engine_ptr->gene_model();
                    const bool has_gm = !config.exon_gtf_path.empty()
                                        && gm.n_genes() > 0;
                    std::vector<uint32_t> hits;
                    bam1_t* b = bam_init1();
                    uint64_t n_written = 0, n_total = 0;

                    while (sam_read1(src_fp, src_hdr, b) >= 0) {
                        ++n_total;
                        // Skip secondary / unmapped reads
                        if (b->core.flag & BAM_FSECONDARY) continue;
                        if (b->core.flag & BAM_FUNMAP)      continue;

                        // Extract barcode/UMI tags written by STAR (CB_samTagOut mode)
                        std::string cb, cr, ub, ur;
                        if (const uint8_t* t = bam_aux_get(b, "CB")) cb = bam_aux2Z(t);
                        if (const uint8_t* t = bam_aux_get(b, "CR")) cr = bam_aux2Z(t);
                        if (const uint8_t* t = bam_aux_get(b, "UB")) ub = bam_aux2Z(t);
                        if (const uint8_t* t = bam_aux_get(b, "UR")) ur = bam_aux2Z(t);

                        // Determine gene + region type from genomic alignment position
                        std::string gene_id, gene_name;
                        char region_type = 'I';
                        if (has_gm) {
                            int32_t tid     = b->core.tid;
                            int32_t pos     = b->core.pos;
                            int32_t end_pos = bam_endpos(b);
                            gm.query_exons(tid, pos, end_pos, hits);
                            if (!hits.empty()) {
                                uint32_t g = gm.exon_to_gene(hits[0]);
                                if (g < gm.n_genes()) {
                                    gene_id     = gm.gene_ids()[g];
                                    gene_name   = gm.gene_names()[g];
                                    region_type = 'E';
                                }
                            } else {
                                gm.query_introns(tid, pos, end_pos, hits);
                                if (!hits.empty()) {
                                    uint32_t g = gm.intron_to_gene(hits[0]);
                                    if (g < gm.n_genes()) {
                                        gene_id     = gm.gene_ids()[g];
                                        gene_name   = gm.gene_names()[g];
                                        region_type = 'N';
                                    }
                                }
                            }
                        }

                        bool wrote = writer.write_record(
                            b, cb, cr, /*bc_qual=*/"",
                            ub, ur, /*umi_qual=*/"",
                            gene_id, gene_name, region_type, /*xf=*/1);
                        if (wrote) ++n_written;
                    }
                    bam_destroy1(b);
                    writer.close();
                    std::cerr << "[tagged_bam] wrote=" << n_written
                              << "/" << n_total << " reads in "
                              << tbam_sw.elapsed_s() << "s: "
                              << tagged_bam_path << "\n";
                }
                if (src_hdr) sam_hdr_destroy(src_hdr);
            }
            if (src_fp) sam_close(src_fp);

            // Clean up the source BAM after tagged BAM write (kept alive above)
            if (!parallel_bam_path.empty()) {
                unlink(parallel_bam_path.c_str());
                unlink(parallel_bai_path.c_str());
            }
        }
    }

    // Add wrong-strand warning for parallel path (not emitted inside the parallel branch above)
    if (parallel_pileup == 1 && stats.exon_hits > 0 && stats.wrong_strand > 0 &&
        !engine_ptr->strand_was_flipped()) {
        double ws_frac = static_cast<double>(stats.wrong_strand) /
                         (stats.exon_hits + stats.wrong_strand);
        if (ws_frac > 0.35 && !config.reverse_strand) {
            std::cerr << "[singlet] WARNING: " << static_cast<int>(ws_frac * 100)
                      << "% wrong-strand reads detected. This sample may use a 5' capture "
                      << "protocol. Re-run with --reverse-strand for correct gene counting.\n";
        }
    }

    // ── Write stats ──
    std::string mode_str = bam_file.empty()
        ? (parallel_pileup == 1
               ? (onefq_mode ? "1fq_parallel" : sra_mode ? "sra_parallel" : "star_parallel")
               : (onefq_mode ? "1fq_stream" : sra_mode ? "sra_stream" : "star_pipe"))
        : "bam_file";
    std::string extra_json =
        "  \"mode\": \"" + mode_str + "\",\n"
        "  \"star_threads\": " + std::to_string(star_threads) + ",\n"
        "  \"pileup_threads\": " + std::to_string(parallel_pileup == 1 ? star_threads : pileup_threads) + ",\n"
        "  \"n_barcodes\": " + std::to_string(engine_ptr->barcodes().size()) + ",\n"
        "  \"strand_auto_flipped\": " + (engine_ptr->strand_was_flipped() ? "true" : "false") + ",\n"
        "  \"reverse_strand\": " + (config.reverse_strand ? "true" : "false") + ",\n";
    singlet::write_stats_json(out_prefix + "/pileup_stats.json",
                              stats, "singlet-0.3.0", total_sw.elapsed_s(), extra_json);

    // ── Track B: write cascade_stats.json if enabled (T-L2-8) ───────────────
    if (cascade_enabled && cascade_stats_enabled) {
        // Cascade was enabled but reads still go through STAR in this prototype.
        // Write a stub stats file recording cascade mode for downstream schema validation.
        singlet::CascadeStats cs;
        cs.cascade_enabled = true;
        cs.deterministic = true;
        cs.em_seed = 0xC0FFEE;
        // In the full integration (Phase 2), l1/l2/l3/l4 counts are populated
        // from the CascadeRouter hot loop. For now, all reads go through L3 (STAR).
        cs.L1_txome.reads_in    = stats.total_reads;
        cs.L1_txome.resolved    = 0;
        cs.L1_txome.passthrough = stats.total_reads;
        cs.L2_te.reads_in       = stats.total_reads;
        cs.L2_te.resolved       = 0;
        cs.L2_te.passthrough    = stats.total_reads;
        cs.L3_star.reads_in     = stats.total_reads;
        cs.L3_star.mapped       = stats.mapped_reads;
        cs.L3_star.unmapped     = stats.total_reads - stats.mapped_reads;
        cs.L4_nonhost.reads_in  = stats.total_reads - stats.mapped_reads;
        cs.L4_nonhost.resolved  = 0;
        cs.L4_nonhost.unmappable = stats.total_reads - stats.mapped_reads;
        singlet::write_cascade_stats(out_prefix, cs);
        std::cerr << "[cascade] cascade_stats.json written (mode=" << cascade_mode << ")\n";
    }

    std::cerr << "[singlet] Total: " << total_sw.elapsed_s() << "s\n";
    std::cerr << "[singlet] Outputs: " << out_prefix << "\n";
    return star_exit;
}
