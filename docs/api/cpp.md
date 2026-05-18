# C++ API Reference

All C++ APIs are header-only and live under `include/singlet/`. Add `include/` to your include path.

```cmake
find_package(Singlet REQUIRED)
target_link_libraries(myapp PRIVATE singlet::pz singlet::fq singlet::pileup)
```

---

## singlet::fq — `.1fq` format codec

Include: `#include "singlet/fq/lib1fq.h"`

Namespace: `singlet::fq`

### Core types (`types.h`)

```cpp
// Compression codec
enum class Codec : uint8_t { ZSTD=0, LZ4=1, LZ4HC=2, RANS=3, NONE=255 };

// Quality storage mode
enum class QualMode : uint8_t {
    NONE    = 0,   // no quality stored (sufficient for STAR alignment)
    BINNED4 = 1,   // 2 bits/base: 4 quality bins
    FULL    = 2,   // 6 bits/base: full Phred
    BINNED2 = 3,   // 1 bit/base: pass/fail at Q20
};

// Sequence encoding
enum class SeqEncoding : uint8_t {
    PACKED_2BIT  = 0,  // 4 bases/byte — archival; lib1fq internal
    BYTE_NUMERIC = 1,  // 1 base/byte — A=0 C=1 G=2 T=3 N=4 — direct STAR input
};

// Protocol detection confidence
enum class Confidence : uint8_t { NONE=0, LOW=1, MEDIUM=2, HIGH=3, MANUAL=4, FORCE=5 };

// File flags (OR'd together in block header)
namespace Flags {
    DEDUPED       = 0x01  // reads collapsed to unique (BC,UMI,R2) triples
    SORTED        = 0x02  // sorted by barcode
    TRIMMED       = 0x04  // polyA/adapter trimming applied
    BC_DICT       = 0x08  // barcode dictionary present  
    BC_FILTERED   = 0x10  // only whitelisted barcodes retained
    DELTA         = 0x20  // delta encoding vs reference
    REF_COMPRESS  = 0x40  // reference-based compression tier 2 applied
    INCOMPLETE    = 0x80  // writer interrupted; file may be truncated
}
```

### Writer (`writer.h`)

```cpp
struct WriterConfig {
    Codec      codec        = Codec::ZSTD;
    int        codec_level  = 3;
    QualMode   qual_mode    = QualMode::BINNED4;
    uint32_t   block_size   = 100000;  // reads per block
    uint16_t   r2_maxlen    = 0;       // truncate R2 to this length (0 = unlimited)
    uint8_t    protocol_id  = 0;       // 0 = UNKNOWN
    Confidence confidence   = Confidence::NONE;
    // Barcode dictionary (for BC_DICT flag)
    std::vector<std::vector<uint8_t>> bc_dict;
    uint16_t   bc_offset    = 0;
    uint16_t   bc_length    = 0;
    uint16_t   umi_offset   = 0;
    uint16_t   umi_length   = 0;
    bool       polya_trim   = false;
    bool       sort_by_bc   = false;
    bool       no_dedup     = false;
};

class Writer {
public:
    void open(const std::string& path, const WriterConfig& cfg);

    // Add one paired read. Sequences are byte-numeric (A=0 C=1 G=2 T=3 N=4).
    // qual may be nullptr when qual_mode == NONE.
    void add_read(const uint8_t* r2, uint16_t r2_len, const uint8_t* r2_qual,
                  const uint8_t* r1, uint16_t r1_len, const uint8_t* r1_qual);

    // Finalize file. metadata_json is written to the file footer (may be "{}").
    void finish(const std::string& metadata_json = "{}");
};
```

### Reader (`reader.h`)

```cpp
class Reader {
public:
    bool open(const std::string& path);

    // Returns false at EOF.
    // Sequences are delivered as byte-numeric (BYTE_NUMERIC encoding) regardless
    // of storage encoding — the reader unpacks 2-bit to byte-numeric automatically.
    bool next_block(ReaderBlock& block);

    const FileHeader& header() const;
};

struct ReaderBlock {
    std::vector<uint8_t> r2_seq;    // flat, byte-numeric A=0 C=1 G=2 T=3 N=4
    std::vector<uint8_t> r2_qual;   // flat, Phred-scaled (empty if NONE)
    std::vector<uint8_t> r1_seq;
    std::vector<uint8_t> r1_qual;
    std::vector<uint16_t> r2_lens;  // per-read R2 lengths
    std::vector<uint16_t> r1_lens;  // per-read R1 lengths
    uint32_t n_reads = 0;
};
```

### SRA encoder (`sra_encoder.h`)

Streams directly from an SRA file (via NCBI VDB) to `.1fq`. No intermediate FASTQ file.

```cpp
struct EncoderConfig {
    std::string output_path;
    Codec       codec        = Codec::ZSTD;
    int         codec_level  = 3;
    QualMode    qual_mode    = QualMode::BINNED4;
    uint32_t    block_size   = 500000;
    uint16_t    r2_maxlen    = 0;
    std::string protocol_tag;     // "" = auto-detect
    std::vector<std::string> whitelist_dirs;
    bool        no_dedup     = false;
    bool        no_trim      = false;
    bool        sort_by_bc   = false;
    int         vdb_threads  = 4;
    // Progress callback (optional): called every progress_interval reads
    std::function<void(uint64_t reads, uint64_t total)> progress_cb;
    uint64_t    progress_interval = 500000;
};

struct EncoderStats {
    uint64_t total_reads     = 0;
    uint64_t blocks_written  = 0;
    std::string protocol_tag;
    Confidence  confidence;
    ProfileStats profile;
};

class SraEncoder {
public:
    EncoderStats encode(const std::string& sra_path_or_accession,
                        const EncoderConfig& cfg);
};
```

### FASTQ encoder (`fastq_encoder.h`)

Encodes a FASTQ pair (R1/R2) to `.1fq`.

```cpp
class FastqEncoder {
public:
    EncoderStats encode(const std::string& r1_path,
                        const std::string& r2_path,
                        const EncoderConfig& cfg);
};
```

### Deduplication (`dedup.h`)

Removes duplicate reads (same barcode + UMI + R2) from a `.1fq` file. Uses H1 UMI collapse.

```cpp
struct DedupConfig {
    std::string input_path;
    std::string output_path;
    bool preserve_dup_count = false;  // store duplicate count in output
};

struct DedupStats {
    uint64_t input_reads    = 0;
    uint64_t unique_reads   = 0;
    uint64_t dup_rate       = 0;  // percentage * 100
};

DedupStats dedup_1fq(const DedupConfig& cfg);
```

---

## singlet::pileup — streaming BAM pileup engine

Include: `#include "singlet/pileup/pileup_engine.h"` and `#include "singlet/pileup/export.h"`

Namespace: `singlet`

### PileupConfig

```cpp
struct PileupConfig {
    std::string snp_path;        // path to VCF/TSV of SNP positions (for AD/DP)
    std::string exon_gtf_path;   // path to GTF for exon intervals
    std::string barcode_path;    // path to filtered barcode list (one per line)
    std::string out_chrm_bam;    // path for chrM BAM output (leave empty to skip)
    int    threads       = 4;
    int    min_mapq      = 20;
    int    min_baseq     = 10;
    bool   count_introns = true;   // intronic UMI counts
    bool   count_sj      = true;   // splice junction extraction (CIGAR N ops)
    bool   umi_dedup     = true;   // (BC, gene, UMI) triple deduplication
    bool   stranded      = true;   // strand-aware counting (10x Forward)
    bool   count_mt      = false;  // chrM heteroplasmy; enable with --pipeline
};
```

### PileupEngine

```cpp
class PileupEngine {
public:
    explicit PileupEngine(const PileupConfig& cfg);

    // Load references (SNPs, GTF, barcodes) into memory.
    // Call before run(). Returns false on error.
    bool load_references();

    // Process BAM. Source may be a file path or "-" for stdin.
    // Blocks until the BAM stream is exhausted.
    PileupStats run(const std::string& bam_source);

    // Access accumulated sparse matrices after run().
    // All matrices are in COO format; use export_results() to convert to CSC.
    const CooMatrix& exon_counts()   const;
    const CooMatrix& intron_counts() const;
    const CooMatrix& sj_counts()     const;
    const CooMatrix& snp_ad()        const;
    const CooMatrix& snp_dp()        const;
    const std::vector<std::string>& barcodes() const;
};

struct PileupStats {
    uint64_t total_reads    = 0;
    uint64_t mapped_reads   = 0;
    uint64_t barcoded_reads = 0;
    uint64_t snp_hits       = 0;
    uint64_t exon_hits      = 0;
    // ... additional fields
};
```

### ExportConfig and export_results

```cpp
struct ExportConfig {
    std::string out_prefix;               // output directory
    std::string output_format = "1pz";    // "1pz" or "mtx"
    bool pipeline_mode = false;           // enable demux + mt heteroplasmy
    int  n_donors      = -1;              // -1 = auto-detect from SNP data
    int  threads       = 1;
};

// Export all pileup matrices to disk.
// Handles: COO→CSC conversion (parallel), mt heteroplasmy (pipeline mode),
//          donor demultiplexing (pipeline mode), parallel matrix writes.
ExportStats export_results(const PileupEngine& engine,
                           const PileupConfig& pileup_cfg,
                           const ExportConfig& export_cfg);
```

### Typical usage

```cpp
#include "singlet/pileup/pileup_engine.h"
#include "singlet/pileup/export.h"

singlet::PileupConfig cfg;
cfg.exon_gtf_path = "genes.gtf";
cfg.snp_path      = "snps.vcf.gz";
cfg.barcode_path  = "barcodes.tsv";
cfg.threads       = 8;

singlet::PileupEngine engine(cfg);
engine.load_references();
auto stats = engine.run("alignments.bam");  // or "-" for stdin BAM pipe

singlet::ExportConfig ecfg;
ecfg.out_prefix     = "results/";
ecfg.output_format  = "1pz";
singlet::export_results(engine, cfg, ecfg);
```

---

## singlet::star — bundled STAR entry point

Include: `#include "singlet/star/star_api.h"`

```cpp
extern "C++" int star_main_impl(int argc, char* argv[]);
```

`star_main_impl()` is the renamed `main()` of the vendored STAR aligner. It accepts the same command-line arguments as the STAR binary. singlify calls it in a forked child process, which writes unsorted BAM to stdout (a pipe) that the parent reads into the pileup engine.

You do not need to call `star_main_impl()` directly — `singlify` handles the fork+pipe+pileup orchestration. This header is exposed for advanced use cases (e.g., embedding singlify as a library).
