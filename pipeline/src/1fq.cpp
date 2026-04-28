// 1fq — CLI tool for .1fq format operations
//
// Subcommands:
//   1fq encode  — Encode SRA/FASTQ → .1fq
//   1fq inspect — Show .1fq file metadata
//   1fq decode  — Decode .1fq → FASTQ
//
// Usage:
//   1fq encode --sra SRRxxxxxxx.sra -o output.1fq [options]
//   1fq inspect input.1fq
//   1fq decode input.1fq -o R1.fastq.gz R2.fastq.gz

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include "lib1fq/lib1fq.h"
#include "lib1fq/sra_encoder.h"
#include "lib1fq/fastq_encoder.h"
#include "lib1fq/dedup.h"

// ── Timing helper ──
struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t0 = Clock::now();
    double elapsed() const {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }
};

// ────────────────────────── encode ──────────────────────────

static int cmd_encode(int argc, char* argv[]) {
    std::string sra_path, output_path;
    std::string reads_r1, reads_r2;
    std::string accession;
    std::string protocol_tag;
    std::string qual_str = "binned";
    std::string codec_str = "zstd";
    int codec_level = 4;
    uint32_t block_size = 500000;
    uint16_t r2_maxlen = 0;
    std::vector<std::string> whitelist_dirs;
    bool no_dedup = false;
    bool no_trim = false;
    bool sort_by_bc = true;
    bool verbose = false;
    int vdb_threads = 4;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--sra" || arg == "-s") && i+1 < argc)
            sra_path = argv[++i];
        else if ((arg == "--accession") && i+1 < argc)
            accession = argv[++i];
        else if (arg == "--reads" && i+2 < argc) {
            reads_r1 = argv[++i];
            reads_r2 = argv[++i];
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
            block_size = std::atoi(argv[++i]);
        else if (arg == "--r2-maxlen" && i+1 < argc)
            r2_maxlen = static_cast<uint16_t>(std::atoi(argv[++i]));
        else if (arg == "--whitelist-dir" && i+1 < argc)
            whitelist_dirs.push_back(argv[++i]);
        else if (arg == "--no-dedup")
            no_dedup = true;
        else if (arg == "--no-trim")
            no_trim = true;
        else if (arg == "--sort-by-bc")
            sort_by_bc = true;
        else if (arg == "--no-sort-by-bc")
            sort_by_bc = false;
        else if (arg == "--vdb-threads" && i+1 < argc)
            vdb_threads = std::atoi(argv[++i]);
        else if (arg == "--verbose" || arg == "-v")
            verbose = true;
        else if (arg == "-h" || arg == "--help") {
            std::cerr
                << "Usage: 1fq encode [OPTIONS]\n\n"
                << "Input (one required):\n"
                << "  --sra FILE           SRA file path\n"
                << "  --accession SRRxxx   Stream from NCBI (VDB resolves accession)\n"
                << "  --reads R1 R2        Paired FASTQ files (.gz supported)\n\n"
                << "Output:\n"
                << "  -o, --output FILE    Output .1fq file\n\n"
                << "Options:\n"
                << "  --protocol TAG       Force protocol (skip auto-detect)\n"
                << "  --quality MODE       none|binned|binned2|full (default: binned)\n"
                << "  --no-dedup           Skip PCR duplicate collapsing\n"
                << "  --no-trim            Skip polyA/adapter trimming\n"
                << "  --sort-by-bc         Sort reads by barcode within blocks (default: on)\n"
                << "  --no-sort-by-bc      Disable barcode sort\n"
                << "  --codec NAME         zstd|lz4|none (default: zstd)\n"
                << "  --codec-level N      Codec level (default: 4)\n"
                << "  --block-size N       Reads per block (default: 500000)\n"
                << "  --r2-maxlen N        Max R2 bases to keep (0=all, default: 0)\n"
                << "  --vdb-threads N      Parallel VDB reader threads (default: 4)\n"
                << "  --whitelist-dir DIR  Search dir for whitelists (repeatable)\n"
                << "  --verbose, -v        Print detection diagnostics\n";
            return 0;
        }
    }

    if (sra_path.empty() && reads_r1.empty() && accession.empty()) {
        std::cerr << "ERROR: --sra, --accession, or --reads required\n";
        return 1;
    }

    // --accession resolves to --sra (VDB handles network transparently)
    if (!accession.empty()) {
        if (!sra_path.empty()) {
            std::cerr << "ERROR: --sra and --accession are mutually exclusive\n";
            return 1;
        }
        sra_path = accession;  // VDB accepts accession strings directly
    }

    if (!sra_path.empty() && !reads_r1.empty()) {
        std::cerr << "ERROR: --sra and --reads are mutually exclusive\n";
        return 1;
    }
    if (output_path.empty()) {
        // Default: derive from first input with .1fq extension
        std::string base = sra_path.empty() ? reads_r1 : sra_path;
        auto dot = base.rfind('.');
        // Strip double extensions like .fastq.gz
        if (dot != std::string::npos) {
            base = base.substr(0, dot);
            auto dot2 = base.rfind('.');
            if (dot2 != std::string::npos) base = base.substr(0, dot2);
        }
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

    // Build encoder config
    lib1fq::EncoderConfig ecfg;
    ecfg.output_path = output_path;
    ecfg.codec = codec;
    ecfg.codec_level = codec_level;
    ecfg.qual_mode = qual_mode;
    ecfg.block_size = block_size;
    ecfg.r2_maxlen = r2_maxlen;
    ecfg.protocol_tag = (protocol_tag == "auto") ? "" : protocol_tag;
    ecfg.whitelist_dirs = whitelist_dirs;
    ecfg.no_dedup = no_dedup;
    ecfg.no_trim = no_trim;
    ecfg.sort_by_bc = sort_by_bc;
    ecfg.vdb_threads = vdb_threads;
    ecfg.verbose = verbose;
    ecfg.progress_cb = [](uint64_t reads, uint64_t total) {
        double pct = (total > 0) ? 100.0 * reads / total : 0.0;
        std::cerr << "\r[1fq-encode] " << reads << " / " << total
                  << " reads (" << static_cast<int>(pct) << "%)" << std::flush;
    };

    Timer t;
    uint64_t total_reads = 0;
    uint32_t blocks_written = 0;
    std::string protocol_result;
    int confidence_result = 0;

    if (!reads_r1.empty()) {
        // FASTQ input path
        lib1fq::FastqEncoder encoder;
        auto stats = encoder.encode(reads_r1, reads_r2, ecfg);
        total_reads = stats.total_reads;
        blocks_written = stats.blocks_written;
        protocol_result = stats.protocol_tag;
        confidence_result = static_cast<int>(stats.confidence);
    } else {
        // SRA input path
        lib1fq::SraEncoder encoder;
        auto stats = encoder.encode(sra_path, ecfg);
        if (stats.exit_code != 0) return stats.exit_code;
        total_reads = stats.total_reads;
        blocks_written = stats.blocks_written;
        protocol_result = stats.protocol_tag;
        confidence_result = static_cast<int>(stats.confidence);
        if (verbose) stats.profile.print(total_reads);
    }

    std::cerr << "\n[1fq-encode] Done: " << total_reads << " reads"
              << ", " << blocks_written << " blocks"
              << ", " << t.elapsed() << "s\n";
    std::cerr << "[1fq-encode] Protocol: " << protocol_result
              << " (confidence: " << confidence_result << ")\n";
    std::cerr << "[1fq-encode] Output: " << output_path << "\n";

    return 0;
}

// ────────────────────────── inspect ──────────────────────────

static int cmd_inspect(int argc, char* argv[]) {
    if (argc < 1) {
        std::cerr << "Usage: 1fq inspect <file.1fq>\n";
        return 1;
    }

    std::string path = argv[0];
    lib1fq::Reader reader;

    try {
        reader.open(path.c_str());
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    const auto& h = reader.header();

    // Codec name
    const char* codec_name = "unknown";
    switch (static_cast<lib1fq::Codec>(h.codec)) {
        case lib1fq::Codec::ZSTD: codec_name = "zstd"; break;
        case lib1fq::Codec::LZ4:  codec_name = "lz4"; break;
        case lib1fq::Codec::LZ4HC: codec_name = "lz4hc"; break;
        case lib1fq::Codec::NONE: codec_name = "none"; break;
        default: break;
    }

    // Quality mode name
    const char* qual_name = "unknown";
    switch (static_cast<lib1fq::QualMode>(h.qual_mode)) {
        case lib1fq::QualMode::NONE:    qual_name = "none"; break;
        case lib1fq::QualMode::BINNED4: qual_name = "binned-4"; break;
        case lib1fq::QualMode::BINNED2: qual_name = "binned-2"; break;
        case lib1fq::QualMode::FULL:    qual_name = "full"; break;
    }

    // Encoding name
    const char* enc_name = "unknown";
    switch (static_cast<lib1fq::SeqEncoding>(h.seq_encoding)) {
        case lib1fq::SeqEncoding::PACKED_2BIT:  enc_name = "2-bit packed"; break;
        case lib1fq::SeqEncoding::BYTE_NUMERIC: enc_name = "byte-numeric"; break;
    }

    // Confidence name
    const char* conf_name = "NONE";
    switch (static_cast<lib1fq::Confidence>(h.confidence)) {
        case lib1fq::Confidence::NONE:   conf_name = "NONE"; break;
        case lib1fq::Confidence::LOW:    conf_name = "LOW"; break;
        case lib1fq::Confidence::MEDIUM: conf_name = "MEDIUM"; break;
        case lib1fq::Confidence::HIGH:   conf_name = "HIGH"; break;
        case lib1fq::Confidence::MANUAL: conf_name = "MANUAL"; break;
        case lib1fq::Confidence::FORCE:  conf_name = "FORCE"; break;
    }

    std::cout << "File: " << path << "\n";
    std::cout << "Format: v" << h.version
              << ", codec: " << codec_name << "-" << static_cast<int>(h.codec_level) << "\n";
    std::cout << "Protocol ID: " << static_cast<int>(h.protocol_id)
              << " (" << conf_name << ")\n";
    std::cout << "Assay: " << lib1fq::assay_type_name(
                  static_cast<lib1fq::AssayType>(h.assay_type)) << "\n";
    std::cout << "Streams: " << static_cast<int>(h.n_streams) << "\n";
    std::cout << "  R1 length: " << (h.stream_lengths[0] ? std::to_string(h.stream_lengths[0]) : "variable") << "\n";
    std::cout << "  R2 length: " << (h.stream_lengths[1] ? std::to_string(h.stream_lengths[1]) : "variable") << "\n";
    std::cout << "Encoding: " << enc_name << "\n";
    std::cout << "Quality: " << qual_name << "\n";
    std::cout << "Tier: " << static_cast<int>(h.tier) << "\n";
    std::cout << "Reads: " << h.n_unique;
    if (h.n_original != h.n_unique)
        std::cout << " (original: " << h.n_original << ")";
    std::cout << "\n";
    std::cout << "Blocks: " << h.block_count
              << " x " << h.block_size << "\n";

    // Flags
    std::cout << "Flags:";
    if (h.flags & lib1fq::Flags::DEDUPED)       std::cout << " DEDUPED";
    if (h.flags & lib1fq::Flags::SORTED)         std::cout << " SORTED";
    if (h.flags & lib1fq::Flags::TRIMMED)        std::cout << " TRIMMED";
    if (h.flags & lib1fq::Flags::BC_DICT)        std::cout << " BC_DICT";
    if (h.flags & lib1fq::Flags::BC_FILTERED)    std::cout << " BC_FILTERED";
    if (h.flags & lib1fq::Flags::DELTA)          std::cout << " DELTA";
    if (h.flags & lib1fq::Flags::REF_COMPRESS)   std::cout << " REF_COMPRESS";
    if (h.flags & lib1fq::Flags::INCOMPLETE)      std::cout << " INCOMPLETE";
    if (h.flags == 0) std::cout << " (none)";
    std::cout << "\n";

    // Metadata
    if (h.meta_size > 0) {
        try {
            std::string meta = reader.read_metadata();
            std::cout << "\nMetadata (" << meta.size() << " bytes):\n"
                      << meta << "\n";
        } catch (const std::exception& e) {
            std::cout << "\nMetadata: (error: " << e.what() << ")\n";
        }
    }

    return 0;
}

// ────────────────────────── decode ──────────────────────────

// Fast integer-to-string for read names (avoids sprintf overhead).
// Writes digits into buf (must have room for 20 chars + NUL). Returns length.
static inline int uint_to_str(uint64_t v, char* buf) {
    if (v == 0) { buf[0] = '0'; return 1; }
    char tmp[20];
    int n = 0;
    while (v > 0) { tmp[n++] = '0' + static_cast<char>(v % 10); v /= 10; }
    for (int i = 0; i < n; ++i) buf[i] = tmp[n - 1 - i];
    return n;
}

static int cmd_decode(int argc, char* argv[]) {
    if (argc < 1) {
        std::cerr << "Usage: 1fq decode <file.1fq> [-o R1.fastq R2.fastq] [--threads N] [--no-verify]\n";
        return 1;
    }

    std::string input_path = argv[0];
    std::string r1_path, r2_path;
    bool no_verify = false;
    int n_threads = 4;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-o" || arg == "--output") && i+2 < argc) {
            r1_path = argv[++i];
            r2_path = argv[++i];
        } else if (arg == "--no-verify") {
            no_verify = true;
        } else if ((arg == "-t" || arg == "--threads") && i+1 < argc) {
            n_threads = std::max(1, std::atoi(argv[++i]));
        }
    }

    if (r1_path.empty()) {
        r1_path = input_path + ".R1.fastq";
        r2_path = input_path + ".R2.fastq";
    }

    lib1fq::Reader reader;
    reader.open(input_path.c_str());
    if (no_verify) reader.set_verify_crc(false);

    FILE* r1_fp = std::fopen(r1_path.c_str(), "w");
    FILE* r2_fp = std::fopen(r2_path.c_str(), "w");
    if (!r1_fp || !r2_fp) {
        std::cerr << "ERROR: Cannot open output files\n";
        return 1;
    }
    std::setvbuf(r1_fp, nullptr, _IOFBF, 256 * 1024);
    std::setvbuf(r2_fp, nullptr, _IOFBF, 256 * 1024);

    const uint32_t total_blocks = reader.block_count();
    const int fd = reader.file_descriptor();
    const auto codec = static_cast<lib1fq::Codec>(reader.header().codec);

    // Formatted output for a single block
    struct FormattedBlock {
        std::vector<char> r1_text;
        std::vector<char> r2_text;
        size_t r1_len = 0;
        size_t r2_len = 0;
        uint32_t n_reads = 0;
    };

    // Worker function: pread + decompress + decode + format one block
    auto process_block = [&](uint32_t block_idx, uint64_t read_offset) -> FormattedBlock {
        FormattedBlock fb;

        // Read compressed payload via pread (thread-safe, no seeking)
        uint64_t file_off = reader.block_offset(block_idx);
        uint32_t comp_size = reader.block_comp_size(block_idx);

        lib1fq::BlockHeader bh;
        ::pread(fd, &bh, sizeof(bh), static_cast<off_t>(file_off));

        std::vector<uint8_t> comp_buf(comp_size);
        ::pread(fd, comp_buf.data(), comp_size,
                static_cast<off_t>(file_off + sizeof(bh)));

        // Decompress
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

        // Decode
        lib1fq::DecodedBlock blk;
        reader.decode_payload(raw_buf.data(), raw_size, bh, blk);

        // Format FASTQ
        uint32_t n = blk.n_reads;
        fb.n_reads = n;
        uint16_t r1_est = blk.r1_lengths.empty() ? 28 : *std::max_element(blk.r1_lengths.begin(), blk.r1_lengths.end());
        uint16_t r2_est = blk.r2_lengths.empty() ? 91 : *std::max_element(blk.r2_lengths.begin(), blk.r2_lengths.end());
        fb.r1_text.resize(n * (static_cast<size_t>(r1_est) * 2 + 30));
        fb.r2_text.resize(n * (static_cast<size_t>(r2_est) * 2 + 30));

        static constexpr char SEQ_LUT[8] = {'A','C','G','T','N','N','N','N'};
        char* r1p = fb.r1_text.data();
        char* r2p = fb.r2_text.data();
        bool has_qual = !blk.r2_qual.empty();

        for (uint32_t i = 0; i < n; ++i) {
            uint16_t r1_len = blk.r1_len(i);
            uint16_t r2_len = blk.r2_len(i);

            char numbuf[20];
            int nlen = uint_to_str(read_offset + i, numbuf);

            // R1
            *r1p++ = '@';
            std::memcpy(r1p, "read_", 5); r1p += 5;
            std::memcpy(r1p, numbuf, nlen); r1p += nlen;
            *r1p++ = '/'; *r1p++ = '1'; *r1p++ = '\n';
            const uint8_t* r1s = blk.r1_seq(i);
            for (uint16_t b = 0; b < r1_len; ++b)
                r1p[b] = SEQ_LUT[r1s[b] & 0x07];
            r1p += r1_len;
            *r1p++ = '\n'; *r1p++ = '+'; *r1p++ = '\n';
            std::memset(r1p, 'I', r1_len);
            r1p += r1_len;
            *r1p++ = '\n';

            // R2
            *r2p++ = '@';
            std::memcpy(r2p, "read_", 5); r2p += 5;
            std::memcpy(r2p, numbuf, nlen); r2p += nlen;
            *r2p++ = '/'; *r2p++ = '2'; *r2p++ = '\n';
            const uint8_t* r2s = blk.r2_seq(i);
            for (uint16_t b = 0; b < r2_len; ++b)
                r2p[b] = SEQ_LUT[r2s[b] & 0x07];
            r2p += r2_len;
            *r2p++ = '\n'; *r2p++ = '+'; *r2p++ = '\n';
            if (has_qual) {
                const uint8_t* q = blk.r2_quality(i);
                for (uint16_t b = 0; b < r2_len; ++b)
                    r2p[b] = static_cast<char>(q[b] + 33);
            } else {
                std::memset(r2p, '?', r2_len);
            }
            r2p += r2_len;
            *r2p++ = '\n';
        }

        fb.r1_len = static_cast<size_t>(r1p - fb.r1_text.data());
        fb.r2_len = static_cast<size_t>(r2p - fb.r2_text.data());
        return fb;
    };

    // Parallel decode: process N blocks at a time, write in order
    uint64_t total_reads = 0;
    uint32_t batch_size = static_cast<uint32_t>(n_threads);

    // Pre-compute read offsets for each block (needed for read naming)
    std::vector<uint64_t> block_read_offsets(total_blocks);
    {
        // Read the block headers to get n_reads per block
        uint64_t cumulative = 0;
        for (uint32_t b = 0; b < total_blocks; ++b) {
            block_read_offsets[b] = cumulative;
            lib1fq::BlockHeader bh;
            ::pread(fd, &bh, sizeof(bh), static_cast<off_t>(reader.block_offset(b)));
            cumulative += bh.n_reads;
        }
    }

    for (uint32_t base = 0; base < total_blocks; base += batch_size) {
        uint32_t batch_end = std::min(base + batch_size, total_blocks);
        uint32_t n_batch = batch_end - base;

        // Launch parallel workers
        std::vector<std::future<FormattedBlock>> futures;
        futures.reserve(n_batch);
        for (uint32_t b = base; b < batch_end; ++b) {
            futures.push_back(std::async(std::launch::async, process_block, b, block_read_offsets[b]));
        }

        // Collect and write in order
        for (uint32_t i = 0; i < n_batch; ++i) {
            FormattedBlock fb = futures[i].get();
            std::fwrite(fb.r1_text.data(), 1, fb.r1_len, r1_fp);
            std::fwrite(fb.r2_text.data(), 1, fb.r2_len, r2_fp);
            total_reads += fb.n_reads;
        }

        if ((base / batch_size) % 2 == 0) {
            std::cerr << "\r[1fq-decode] " << total_reads << " reads" << std::flush;
        }
    }

    std::fclose(r1_fp);
    std::fclose(r2_fp);

    std::cerr << "\n[1fq-decode] Done: " << total_reads << " reads (" << n_threads << " threads)\n";
    std::cerr << "[1fq-decode] Output: " << r1_path << ", " << r2_path << "\n";

    return 0;
}

// ────────────────────────── benchmark ──────────────────────────

static int cmd_benchmark(int argc, char* argv[]) {
    uint32_t block_reads = 100000;
    uint16_t r2_len = 91;
    double dup_rate = 0.5;
    int iterations = 5;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--reads" && i+1 < argc)
            block_reads = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (arg == "--r2-len" && i+1 < argc)
            r2_len = static_cast<uint16_t>(std::atoi(argv[++i]));
        else if (arg == "--dup-rate" && i+1 < argc)
            dup_rate = std::atof(argv[++i]);
        else if (arg == "--iterations" && i+1 < argc)
            iterations = std::atoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            std::cerr
                << "1fq benchmark — Codec benchmarking\n\n"
                << "Options:\n"
                << "  --reads N       Reads per block (default: 100000)\n"
                << "  --r2-len N      R2 length (default: 91)\n"
                << "  --dup-rate F    Duplicate rate 0-1 (default: 0.5)\n"
                << "  --iterations N  Iterations per test (default: 5)\n";
            return 0;
        }
    }

    lib1fq::benchmark::run_benchmark(block_reads, r2_len, dup_rate, iterations);
    return 0;
}

// ────────────────────────── refcompress ──────────────────────────

static int cmd_refcompress(int argc, char* argv[]) {
    std::string input_path, output_path;
    std::string genome_dir, alignments_path;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i+1 < argc)
            input_path = argv[++i];
        else if ((arg == "-o" || arg == "--output") && i+1 < argc)
            output_path = argv[++i];
        else if (arg == "--genome-dir" && i+1 < argc)
            genome_dir = argv[++i];
        else if (arg == "--alignments" && i+1 < argc)
            alignments_path = argv[++i];
        else if (arg == "-h" || arg == "--help") {
            std::cerr
                << "1fq refcompress — Reference-compress .1fq (tier 1 → tier 2)\n\n"
                << "Usage: 1fq refcompress -i input.1fq -o output.1fq \\\n"
                << "           --genome-dir /path/to/star/genome \\\n"
                << "           --alignments /path/to/Aligned.bam\n\n"
                << "Options:\n"
                << "  -i, --input FILE       Input .1fq (tier 1)\n"
                << "  -o, --output FILE      Output .1fq (tier 2)\n"
                << "  --genome-dir DIR       STAR genome directory\n"
                << "  --alignments FILE      BAM file from singlify alignment\n";
            return 0;
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: -i and -o are required\n";
        return 1;
    }
    if (genome_dir.empty() || alignments_path.empty()) {
        std::cerr << "Error: --genome-dir and --alignments are required\n";
        return 1;
    }

    // Verify input is tier 1
    lib1fq::Reader reader;
    reader.open(input_path.c_str());
    if (reader.header().tier != static_cast<uint8_t>(lib1fq::Tier::RAW)) {
        std::cerr << "Error: input is already reference-compressed (tier 2)\n";
        return 1;
    }

    std::cerr << "[1fq-refcompress] Input: " << input_path << "\n";
    std::cerr << "[1fq-refcompress] Reads: " << reader.header().n_unique << "\n";
    std::cerr << "[1fq-refcompress] Genome: " << genome_dir << "\n";
    std::cerr << "[1fq-refcompress] Alignments: " << alignments_path << "\n";
    std::cerr << "\n[1fq-refcompress] Reference compression requires genome "
              << "loading and BAM parsing.\n"
              << "[1fq-refcompress] This operation will be available after "
              << "singlify alignment integration is complete.\n";

    reader.close();
    return 0;
}

// ────────────────────────── main ──────────────────────────

static void usage() {
    std::cerr
        << "1fq — Singlet FASTQ format tool\n\n"
        << "Usage: 1fq <command> [options]\n\n"
        << "Commands:\n"
        << "  encode      Encode SRA/FASTQ → .1fq\n"
        << "  dedup       Collapse PCR duplicates in a .1fq file\n"
        << "  inspect     Show .1fq file metadata\n"
        << "  decode      Decode .1fq → FASTQ\n"
        << "  refcompress Tier 1 → tier 2 reference compression\n"
        << "  benchmark   Codec benchmarking\n\n"
        << "Run '1fq <command> --help' for command-specific options.\n";
}

static int cmd_dedup(int argc, char* argv[]) {
    std::string input_path, output_path;
    uint16_t r2_prefix_len = 50;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-o" || arg == "--output") && i+1 < argc)
            output_path = argv[++i];
        else if (arg == "--r2-prefix" && i+1 < argc)
            r2_prefix_len = static_cast<uint16_t>(std::atoi(argv[++i]));
        else if (arg == "-h" || arg == "--help") {
            std::cerr
                << "Usage: 1fq dedup INPUT.1fq -o OUTPUT.1fq [options]\n\n"
                << "Options:\n"
                << "  -o, --output FILE    Output .1fq file\n"
                << "  --r2-prefix N        R2 prefix length for identity key (default: 50)\n";
            return 0;
        } else if (input_path.empty() && arg[0] != '-') {
            input_path = arg;
        }
    }

    if (input_path.empty()) {
        std::cerr << "ERROR: input .1fq file required\n";
        return 1;
    }
    if (output_path.empty()) {
        std::cerr << "ERROR: -o output file required\n";
        return 1;
    }

    try {
        lib1fq::Deduplicator deduper;
        deduper.dedup(input_path.c_str(), output_path.c_str(), r2_prefix_len);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(); return 1; }

    std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help") { usage(); return 0; }

    try {
        if (cmd == "encode")      return cmd_encode(argc - 2, argv + 2);
        if (cmd == "dedup")       return cmd_dedup(argc - 2, argv + 2);
        if (cmd == "inspect")     return cmd_inspect(argc - 2, argv + 2);
        if (cmd == "decode")      return cmd_decode(argc - 2, argv + 2);
        if (cmd == "refcompress") return cmd_refcompress(argc - 2, argv + 2);
        if (cmd == "benchmark")   return cmd_benchmark(argc - 2, argv + 2);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    std::cerr << "Unknown command: " << cmd << "\n";
    usage();
    return 1;
}
