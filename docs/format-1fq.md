# .1fq — Singlet FASTQ Format Specification (v3.1)

## Overview

`.1fq` is a column-oriented, block-compressed archival format for sequencing
reads, **purpose-built for the singlify pipeline**. It is not designed for
generic aligner compatibility — it integrates directly with singlify's internal
STAR alignment engine and singlet-pileup feature extraction.

**Design goals**, in priority order:

1. **Streaming SRA→.1fq encode** at download speed (single-threaded, never store .sra)
2. **Direct 2-bit→numeric feed** into STAR's suffix array (no ASCII round-trip)
3. **Deep compression** (8–20× over gzipped FASTQ, depending on protocol)
4. **Quality encoding** sufficient for transcriptomic variant calling (per-donor VCF)
5. **Reference-aware R2** post-alignment compression tier (additional 3–5×)
6. **Protocol safety** — wrong chemistry = missed compression, never wrong data
7. **Read folding** — deduplicate at R2-sequence level, align each unique R2 once (5–8×)
8. **2-bit alignment path** — packed genome/reads for 4× cache, 16× comparison (1.5–2×)

### Architecture in the singlify pipeline

```
DOWNLOAD (streaming):
  NCBI S3  ─────→  SRA VDB stream  ─────→  .1fq encoder  ─────→  disk (.1fq)
                   (never saved)            (single-thread)        (archival)

PROCESS (later):
  .1fq on disk  ─→  lib1fq reader  ─→  singlify core  ─→  .1pz
                     (numeric bytes      (STAR + pileup     (counts)
                      + pre-parsed BC      in-process)
                      + UMI + quality)
```

The .sra file never touches disk. The .1fq file is written once during
download and read many times for (re-)processing.

---

## 1. Streaming SRA → .1fq Encode

### 1.1 Architecture: destructive compression at download speed

The existing `SraReader` in singlify uses VDB's `VCursorCellDataDirect` for
zero-copy access to SRA columns: `READ`, `QUALITY`, `READ_LEN`, `READ_START`,
`NAME`. The .1fq encoder wraps this same VDB cursor but writes .1fq blocks
instead of FASTQ text:

```
NCBI S3 download (bandwidth-limited, ~100-200 MB/s)
  → VDB cursor (zero-copy column access, in-process)
  → Chemistry auto-detect (first 100K spots, ~10ms)
  → .1fq block encoder (100K reads/block, single-thread)
  → zstd compress + write to disk
```

**Single-threaded**: the download is the bottleneck (~2-3M spots/sec from S3).
The encoder must keep up with download speed. At ~200 bytes/spot raw and
~40 bytes/spot encoded, even a single core can encode at ~50M spots/sec
(limited by zstd), so the encoder is never the bottleneck.

**Never stores .sra**: the VDB cursor streams directly from the download
pipe. The .1fq encoder consumes spots as they arrive. If the download is
interrupted, the .1fq file is valid up to the last complete block (blocks
are self-contained).

### 1.2 Two-pass vs single-pass

| Mode | Passes | Dedup | Memory | When to use |
|---|---|---|---|---|
| **Stream** (default) | 1 | No | ~50MB | Download-time encode |
| **Dedup** | 2 (or 1 + 2GB RAM) | Yes | ~2GB | Post-download optimization |

Stream mode is used during download. Dedup mode can run later as an optional
offline pass that reads the .1fq, deduplicates, and writes a new .1fq.

### 1.3 Catching the download stream

The SRA toolkit's `prefetch` writes to disk. To avoid this:

```c
// Option A: Direct VDB download (no prefetch, no sra file)
// VDBManager resolves accession → NCBI S3 URL → HTTP GET → VDB pages → cursor
// This is what fasterq-dump does internally, but we skip the FASTQ step.
VDBManagerOpenDBRead(mgr, &db, NULL, "SRR29320040");
// VDB transparently fetches pages from S3 as cursor advances

// Option B: Pipe from prefetch (if VDB direct streaming is unreliable)
// prefetch --output-file /dev/stdout SRR29320040 | 1fq_encode --stdin-sra -o out.1fq
// Encoder reads VDB pages from stdin, never writes .sra to disk
```

Option A is cleanest — VDB already supports transparent network streaming.
The encoder just opens the accession string and VDB handles the HTTP.

---

## 2. Protocol Registry & Auto-Detection

### 2.1 Protocol registry (data-driven, extensible)

Each protocol is a TOML entry defining read layout, whitelists, and
validation thresholds. Adding a new chemistry = adding a row. No code changes.

```toml
[protocol.10x-3p-v3]
description = "10x Chromium 3' v3/v3.1"
star_solo_type = "CB_UMI_Simple"
star_solo_cb_start = 1
star_solo_cb_len = 16
star_solo_umi_start = 17
star_solo_umi_len = 12
streams = 2
r1_length = 28
r2_length_min = 50
r1_segments = [
    { type = "BC",  offset = 0, length = 16, whitelist = "3M-february-2018.txt" },
    { type = "UMI", offset = 16, length = 12 },
]
r2_segments = [
    { type = "CDNA", offset = 0, length = 0 },
]
identity_segments = ["BC", "UMI", "CDNA:prefix(50)"]
polyA_expected = true
polyA_read = "R2"
polyA_end = "3prime"
umi_entropy_min = 1.8
whitelist_match_threshold = 0.50
```

### 2.2 Complete protocol catalog

#### Droplet scRNA-seq

| Tag | R1 | R2 | BC len | UMI len | Whitelist | STAR soloType |
|---|---|---|---|---|---|---|
| `10x-3p-v1` | BC(14)+UMI(10) | cDNA | 14 | 10 | 737K | CB_UMI_Simple |
| `10x-3p-v2` | BC(16)+UMI(10) | cDNA | 16 | 10 | 737K | CB_UMI_Simple |
| `10x-3p-v3` | BC(16)+UMI(12) | cDNA | 16 | 12 | 6.8M | CB_UMI_Simple |
| `10x-3p-v4` | BC(16)+UMI(12) | cDNA | 16 | 12 | 6.8M | CB_UMI_Simple |
| `10x-5p-v2` | BC(16)+UMI(10) | cDNA | 16 | 10 | 737K | CB_UMI_Simple |
| `10x-5p-v3` | BC(16)+UMI(12) | cDNA | 16 | 12 | 6.8M | CB_UMI_Simple |
| `dropseq` | BC(12)+UMI(8) | cDNA | 12 | 8 | knee | CB_UMI_Simple |
| `indrop-v3` | BC1(8)+LNK(22)+BC2(8)+UMI(6) | cDNA | 16+lnk | 6 | 384×384 | CB_UMI_Complex |
| `celseq2` | BC(6)+UMI(6) | cDNA | 6 | 6 | 384 | CB_UMI_Simple |
| `marsseq2` | BC(7)+UMI(8) | cDNA | 7 | 8 | 384 | CB_UMI_Simple |
| `seqwell` | BC(12)+UMI(8) | cDNA | 12 | 8 | knee | CB_UMI_Simple |
| `microwell` | BC(6+6+6)+UMI(6) | cDNA | 18 | 6 | 3-part | CB_UMI_Complex |
| `strtseq` | BC(6)+UMI(5) | cDNA | 6 | 5 | 48/96 | CB_UMI_Simple |
| `quartzseq2` | BC(15)+UMI(8) | cDNA | 15 | 8 | 1536 | CB_UMI_Simple |
| `icell8` | BC(11) | cDNA | 11 | — | varies | CB_UMI_Simple |
| `ddseq` | BC(6+6+6)+UMI(8) | cDNA | 18 | 8 | 3-part | CB_UMI_Complex |
| `surecell` | BC(6+6+6)+UMI(8) | cDNA | 18 | 8 | 3-part | CB_UMI_Complex |
| `dnbelab-c4` | BC(10+10+10)+UMI(10) | cDNA | 30 | 10 | 3-part | CB_UMI_Complex |
| `scopeseq` | BC(12)+UMI(8) | cDNA | 12 | 8 | knee | CB_UMI_Simple |
| `rhapsody` | CLS1(9)+BC1(9)+CLS2(21)+BC2(9)+CLS3(13)+BC3(9)+UMI(8) | cDNA | 27+cls | 8 | 96×96×96 | CB_UMI_Complex |

#### Combinatorial indexing scRNA-seq

| Tag | R1 | R2 | Notes | STAR soloType |
|---|---|---|---|---|
| `parse-v1` | cDNA | BC1(8)+LNK(4)+BC2(8)+LNK(4)+BC3(8)+UMI(10) | R2 has BCs | CB_UMI_Complex |
| `parse-mega` | cDNA | BC1–4(8ea)+LNK+UMI(10) | 4 rounds | CB_UMI_Complex |
| `splitseq` | cDNA | BC1(8)+LNK+BC2(8)+LNK+BC3(8)+UMI(10) | Like Parse | CB_UMI_Complex |
| `scirna3` | cDNA | cDNA | BC via I1/I2 | CB_UMI_Complex |
| `scale` | cDNA | BC1+BC2+UMI | Varies by kit | CB_UMI_Complex |

#### Plate-based / full-length scRNA-seq

| Tag | R1 | R2 | Notes | STAR soloType |
|---|---|---|---|---|
| `smartseq2` | cDNA | cDNA | No BC/UMI; 1 cell = 1 run | SmartSeq |
| `smartseq3` | UMI(11)+cDNA | cDNA | UMI at R1 5' | CB_UMI_Simple |
| `flashseq` | cDNA | cDNA | Like SS2 | SmartSeq |

#### Feature barcoding

| Tag | R1 | R2 | Notes |
|---|---|---|---|
| `citeseq-a` | BC(16)+UMI(10–12) | FEATURE(15) | ADT tag |
| `citeseq-bc` | BC(16)+UMI(12) | FEATURE(15) | 10x-compatible |
| `perturbseq` | BC(16)+UMI(12) | FEATURE(20) | Guide RNA |
| `cellhash` | BC(16)+UMI(12) | FEATURE(15) | HTO |

#### Single-cell ATAC-seq

| Tag | R1 | R2 | BC source | STAR soloType |
|---|---|---|---|---|
| `10x-atac-v1` | GENOMIC | GENOMIC | I2: BC(16) | CB_UMI_Simple |
| `10x-atac-v2` | GENOMIC | GENOMIC | I2: BC(16) | CB_UMI_Simple |
| `10x-multi-atac` | BC(16)+GENOMIC | GENOMIC | R1 prefix | CB_UMI_Simple |
| `sciatac` | GENOMIC | GENOMIC | I1+I2 | CB_UMI_Complex |

#### Spatial

| Tag | R1 | R2 | BC len | UMI len | Whitelist |
|---|---|---|---|---|---|
| `visium-v1` | BC(16)+UMI(12) | cDNA | 16 | 12 | 4,992 |
| `visium-hd` | BC(16)+UMI(12) | cDNA | 16 | 12 | ~600K |
| `visium-ffpe` | BC(16)+UMI(12) | cDNA | 16 | 12 | probe |
| `slideseq2` | BC(8)+UMI(9) | cDNA | 8 | 9 | bead |
| `stereoseq` | BC(25)+UMI(10) | cDNA | 25 | 10 | CID |
| `dbitseq` | BC-A(8)+BC-B(8) | cDNA | 16 | — | channel |

#### Immune profiling

| Tag | R1 | R2 | Notes |
|---|---|---|---|
| `10x-vdj-v2` | BC(16)+UMI(10) | cDNA | TCR/BCR |
| `10x-vdj-v3` | BC(16)+UMI(12) | cDNA | TCR/BCR |

#### Epigenomic

| Tag | R1 | R2 | Notes |
|---|---|---|---|
| `cuttag` | GENOMIC | GENOMIC | Tn5, PE |
| `cutrun` | GENOMIC | GENOMIC | MNase, PE |
| `sc-cuttag` | BC+GENOMIC | GENOMIC | 10x or comb BC |
| `chipseq` | GENOMIC | GENOMIC | Enrichment |

#### Bulk

| Tag | R1 | R2 | Notes |
|---|---|---|---|
| `bulk-rna-pe` | cDNA | cDNA | PE |
| `bulk-rna-se` | cDNA | — | SE |
| `wgs-pe` | GENOMIC | GENOMIC | Whole genome |
| `wes-pe` | GENOMIC | GENOMIC | Exome |
| `wgbs-pe` | GENOMIC | GENOMIC | C→T skew |
| `cage` | cDNA(27) | — | 5' cap, SE |

### 2.3 Auto-detection algorithm

Runs on **first 100K reads** (~10 ms), cost dominated by whitelist hash lookups.

```
PHASE 1 — STRUCTURAL FINGERPRINT (1K spots):
  Count streams (1–4), measure per-stream lengths
  Detect constant-sequence regions (linker entropy < 0.5 bits/pos)
  → Coarse candidate set by length fingerprint

PHASE 2 — WHITELIST SCORING (10K reads per candidate):
  For each candidate: extract barcode bytes at defined offset
  Hash against candidate whitelist (exact + hamming-1)
  hamming1_rate > 50% → survives    < 20% → rejected

PHASE 3 — ENTROPY & COMPOSITION (10K reads):
  UMI region: per-position Shannon entropy
    >1.8 bits/pos → true UMI    <1.5 → not UMI
  Linker region: per-position entropy
    <0.5 → true linker (constant motif)
  Insert: GC%, C-depletion (bisulfite), ACGT balance

PHASE 4 — INSERT CHARACTERIZATION (10K reads):
  PolyA at 3' of insert: >30% reads → 3' scRNA
  TSO motif at 5': >20% → 5' protocol
  Tn5 mosaic end (CTGTCTCTTATACACATCT): >20% → ATAC

PHASE 5 — BARCODE CONCENTRATION (100K reads):
  Rank-frequency of detected barcodes:
    500–20K BCs hold >90% reads → droplet (clear knee)
    96–384 BCs, ~equal fraction → plate
    Uniform → no real barcodes (bulk/wrong WL)

PHASE 6 — SCORE & COMMIT:
  Multi-signal conjunction score:
    score = Σ(wi × signal_i)  for WL match, UMI entropy,
            polyA, knee shape, linker, insert motif
  ≥ 0.85 → HIGH    0.60–0.85 → MEDIUM
  0.40–0.60 → LOW    < 0.40 → UNKNOWN
```

### 2.4 Validation without alignment

Chemistry validation uses **internal consistency across independent signals**,
not alignment. The conjunction of:

1. **Whitelist match** (23-bit hash check — random 16-mers match at 0.16%;
   real 10x data matches at >85%)
2. **Barcode knee** (500–20K BCs hold >90% reads vs uniform noise)
3. **UMI entropy** (pseudo-random ~2.0 bits vs adapter <1.5 bits)
4. **Insert profile** (polyA for 3', TSO for 5', Tn5 for ATAC, C-depletion for bisulfite)
5. **Metadata cross-check** (when SRA annotations available)

...has a false-positive rate that is effectively zero for HIGH confidence.
Each signal tests a different biological/technical property.

### 2.5 Confidence & behavior

| Level | Score | Optimizations applied |
|---|---|---|
| `HIGH` (≥0.85) | >70% WL | BC dict + dedup + trim + singlify STARsolo params |
| `MEDIUM` (0.60–0.85) | 50–70% WL | BC dict + trim; skip dedup |
| `LOW` (0.40–0.60) | 20–50% WL | BC dict only |
| `UNKNOWN` (<0.40) | <20% WL | Base layer only; singlify uses generic mode |
| `MANUAL` | User override | Full optimizations + singlify uses specified params |

### 2.6 Safety guarantee

Every protocol-aware transform has a verification gate:

| Transform | Gate | Failure mode |
|---|---|---|
| Barcode dictionary | <70% WL match → fail | Raw 2-bit encoding |
| PCR dedup | No identity matches → 0 dedup | Reads stored verbatim |
| PolyA trim | No polyA found | No trimming |
| singlify STARsolo params | Metadata mismatch → fallback | Generic alignment |

**The .1fq file always preserves all original nucleotides.** Wrong protocol =
worse compression ratio, not wrong alignment results.

---

## 3. Direct Read Feed into singlify (No ASCII)

### 3.1 Current data path (wasteful)

```
SRA VDB cursor → ASCII FASTQ text → FIFO → STAR readLoad()
  → convertNucleotidesToNumbers() → char[0/1/2/3/4] → suffix array lookup
```

STAR's `readLoad()` reads ASCII FASTQ via `getline()`, then
`convertNucleotidesToNumbers()` applies a 256-byte lookup table to convert
ASCII `ACGTN` → numeric `01234`, one byte at a time. Quality is parsed but
largely unused for seed matching.

### 3.2 .1fq native feed path

```
.1fq block → zstd decompress → numeric bytes already in 0/1/2/3/4 encoding
  → copy directly into STAR's Read1[] buffer → suffix array lookup
```

**What changes in STAR:**

STAR's `readLoad()` is replaced by a `readLoad1fq()` function that:

1. Receives a pointer to the decompressed .1fq block data
2. For each read in the block:
   - Copies R2 CDNA segment (already 0/1/2/3/4 bytes) into `Read1[0][0..Lread-1]`
   - Copies barcode bytes into `Read1[1][0..bc_len-1]`
   - Copies UMI bytes into `Read1[1][bc_len..bc_len+umi_len-1]`
   - Sets quality to constant Q30 (or reads from quality column if present)
   - Sets read name to spot index (4 bytes → formatted as "spot_%u")
3. Returns to STAR's normal alignment flow (`mapOneRead()`)

**What's eliminated:**
- All FIFO/pipe infrastructure (no named FIFOs, no fork, no pipe buffers)
- ASCII encode in VDB→FASTQ text conversion
- ASCII parse in STAR's `readLoad()` (`getline`, field splitting)
- `convertNucleotidesToNumbers()` per-base lookup (already numeric)
- Read name string allocation and formatting

**Data format in .1fq blocks:**

The CDNA/GENOMIC segments can be stored in either of two formats:

| Format | Bits/base | Decode | Use case |
|---|---|---|---|
| **2-bit packed** (A=00 C=01 G=10 T=11) | 2 + N-bitmap | Unpack to byte[0/1/2/3] | Archival (smaller) |
| **Byte numeric** (A=0 C=1 G=2 T=3, N=4) | 8 | memcpy | Fast decode (zero convert) |

Default: **2-bit packed** for archival storage. The unpacking loop
(2-bit → byte) is a tight shift+mask operation, ~4 GB/s single-threaded.
Still 22× faster than STAR's alignment rate.

For maximum decode speed (at cost of 4× larger blocks), byte-numeric format
can be used. A flag in the header selects the format.

### 3.3 Pre-parsed barcode and UMI

The .1fq file already has barcodes extracted, dictionary-encoded, and
whitelist-matched. Instead of STAR redoing barcode extraction and whitelist
correction, singlify can:

1. Read barcode dictionary index from .1fq block
2. Look up the actual barcode sequence from the dictionary
3. Set the CB SAM tag directly (skip STARsolo's barcode matching)
4. Pass UMI bytes as the UR tag

This eliminates STARsolo's per-read barcode extraction and correction code.
The .1fq encoder already did this during the auto-detection phase with
verified accuracy.

### 3.4 Integration modes

```
singlify --1fq input.1fq [--protocol manual-override] \
         --genome-dir /path/to/star \
         --barcodes filtered.tsv \
         --exons genes.gtf --snps variants.vcf \
         --out-prefix ./output

Internal flow:
  1. lib1fq opens input.1fq, reads header + metadata
  2. Extract protocol, confidence, segment maps, STARsolo params
  3. If confidence < MEDIUM and no --protocol override → error + suggestion
  4. Load STAR genome + pileup references (existing flow)
  5. Block-by-block decode:
     For each block (100K reads):
       a. zstd decompress
       b. Unpack columns: BC dict indices, UMI bytes, R2 numeric seq
       c. For each read: populate STAR's Read1[]/Qual0[] buffers directly
       d. STAR aligns → BAM records stream to pileup engine
  6. Pileup engine processes BAM same as today
  7. Export .1pz
```

---

## 4. Quality Score Encoding

### 4.1 Why quality matters for singlify

singlify's pileup engine uses `min_baseq = 10` for SNP and RNA editing
pileup. This filters low-quality bases at variant positions to avoid
false-positive allele calls. The per-donor VCF construction depends on
accurate allele depth (AD/DP), which requires per-base quality.

**Dropping quality entirely** means SNP pileup uses all bases equally →
higher false-positive rate in variant calling.

However, for gene expression counting (exon/intron/SJ), quality scores
are irrelevant — STAR's alignment is sequence-based, and UMI dedup is
barcode+gene based.

### 4.2 Quality encoding strategy

Three tiers, selectable per-file:

| Tier | Bits/base | Storage | Variant calling | Expression |
|---|---|---|---|---|
| **None** | 0 | 0% overhead | Not supported | Full support |
| **Binned-4** (default) | 2 | ~6% overhead | Good (>95% concordance) | Full support |
| **Full Phred** | 6 | ~18% overhead | Exact | Full support |

**Binned-4 is the default** — it preserves enough quality information for
variant calling while adding minimal storage overhead.

### 4.3 Binned-4 quality scheme

Map Phred 0–41 to 4 bins:

| Bin | Phred range | Encoded value | Decoded Phred | Meaning |
|---|---|---|---|---|
| 0 | 0–9 | 00 | 6 | Low quality (below min_baseq=10) |
| 1 | 10–19 | 01 | 15 | Marginal quality |
| 2 | 20–29 | 10 | 25 | Good quality |
| 3 | 30–41 | 11 | 37 | High quality |

**The critical threshold is preserved**: bin 0 (Phred <10) vs bins 1–3
(Phred ≥10). This is exactly the `min_baseq=10` filter used by singlify's
SNP pileup. Binned-4 quality has >95% concordance with full-quality
variant calls because the dominant signal is "low vs not-low."

At decode time, each 2-bit quality value maps to the bin center Phred value.

### 4.4 Quality column layout in blocks

For the R2/cDNA segment (where variant calls happen):

```
[2-bit packed quality: ceil(r2_len * 2 / 8) bytes per read, n_reads total]
```

For R1 barcode+UMI: quality is not stored (barcodes use whitelist matching,
not quality-based error correction in our pipeline).

Storage overhead for 100bp R2, binned quality:
- 100 bases × 2 bits = 25 bytes/read
- vs 100 bases × 8 bits = 100 bytes/read (ASCII Phred in FASTQ)
- 4× smaller than raw quality, ~6% of total .1fq file size

---

## 5. Reference-Aware R2 Compression (Tier 2)

### 5.1 Concept

After initial .1fq creation (at download time) and first singlify alignment,
we know the genomic mapping position for each read. R2 sequences that align
to the same region share near-identical sequence with the reference. We can
delta-encode R2 against the transcript reference:

```
R2 raw:  ATCGATCGATCGATCGATCG...  (100 bytes as 2-bit = 25 bytes)
Ref pos: chr1:12345, +strand
Ref seq: ATCGATCGATCGATCGATCG...
Delta:   00000000000100000000...   (mismatches only, ~2-5%)
```

For a read with 3% mismatch rate: ~3 mismatches per 100bp.
Delta encoding: 3 position+base pairs = ~6 bytes vs 25 bytes = 4× savings on R2.

### 5.2 Format: two-tier .1fq

```
Tier 1 (.1fq):
  Created at download time. Full sequences + binned quality.
  Self-contained. Can always decode without reference.

Tier 2 (.1fq with FLAG_REF_COMPRESSED):
  Created after first singlify run. R2/cDNA column replaced with:
    [uint32_t ref_tid]              // chromosome/transcript ID
    [uint32_t ref_pos]              // 0-based position
    [uint8_t  strand]               // 0=fwd, 1=rev
    [uint8_t  n_mismatches]         // number of mismatches vs reference
    [varint   mismatch_positions[]] // offsets from read start
    [2-bit    mismatch_bases[]]     // what the read actually has
    [varint   soft_clip_len]        // 5' soft clip length
    [2-bit    soft_clip_bases[]]    // soft clip sequence
```

### 5.3 Compression savings

| R2 representation | Bytes/read (100bp) | vs raw 2-bit |
|---|---|---|
| Raw 2-bit | 25 | 1× |
| Ref-compressed (3% mm, mapped) | ~8 | 3× |
| Ref-compressed (1% mm, mapped) | ~5 | 5× |
| Unmapped reads | 25 (raw fallback) | 1× |

With ~85% mapping rate and ~3% mismatch: effective compression 2.5× on R2.
Combined with other columns: **total .1fq shrinks by ~30–40%** in tier 2.

### 5.4 Trigger and workflow

```
1. Download → .1fq tier 1 (stream encode, fast)
2. singlify processes .1fq → .1pz (alignment + counting)
3. Background job: 1fq_refcompress reads alignment output + .1fq tier 1
   → writes .1fq tier 2 (in-place upgrade or new file)
4. Future singlify re-runs read tier 2 .1fq:
   Reconstruct R2 from reference + delta at decode time
```

Tier 2 requires the genome reference at decode time. This is always
available in singlify (it loads the STAR genome index anyway).

### 5.5 Decode for singlify

The tier 2 decoder:
1. Looks up `ref_tid:ref_pos` in the loaded genome sequence
2. Copies reference bases as numeric (0/1/2/3)
3. Applies mismatches at indicated positions
4. Prepends soft-clip bases
5. Result: same numeric byte array as tier 1 decode

No speed penalty: reference lookup is a memory read from the already-loaded
genome, mismatch application is ~3 byte writes.

---

## 6. PCR Duplicate Collapsing

### 6.1 Identity segments (protocol-dependent)

| Protocol class | Identity key | Typical dedup rate |
|---|---|---|
| 10x 3'/5', Visium, VDJ | BC + UMI + cDNA(prefix 50bp) | 30–70% |
| Drop-seq, Seq-Well | BC + UMI + cDNA(prefix 50bp) | 20–50% |
| CEL-Seq2, MARS-seq | BC + UMI + cDNA(prefix 30bp) | 20–40% |
| Parse, SPLiT-seq, Rhapsody | BC-parts + UMI + cDNA(prefix 50bp) | 20–40% |
| Smart-seq2, FLASH-seq | R1(full) + R2(prefix 50bp) | 10–40% |
| ATAC | R1(prefix 50bp) + R2(prefix 50bp) | 20–60% |
| Bulk RNA/WGS PE | R1(prefix 50bp) + R2(prefix 50bp) | 5–30% |
| UNKNOWN | All streams concatenated | Very conservative |

### 6.2 singlify integration: dedup-aware counting

When `FLAG_DEDUPED` is set, each .1fq read has a duplicate count. Two modes:

**Mode A — Expand** (default): Emit each read `count` times to STAR. STAR
aligns once per unique read, pileup counts normally. This is correct but
wastes STAR cycles re-aligning identical sequences.

**Mode B — Count-forward** (optimized): Emit each unique read once to STAR.
The alignment produces one BAM record. singlify's pileup engine receives the
BAM record + the duplicate count from the .1fq block, and multiplies
`increment()` calls by the count:

```cpp
// In pileup engine, when processing a deduped .1fq read:
uint32_t dup_count = lib1fq_block.dup_count(read_idx);
exon_acc_.increment(exon_idx, bc_idx, dup_count);  // add count, not 1
```

This skips alignment of duplicate reads entirely. For a sample with 50%
duplication: **2× speedup in STAR alignment** (the dominant bottleneck).

Note: UMI deduplication in pileup is a separate step (BC+gene+UMI).
Format-level dedup collapses exact sequence duplicates (BC+UMI+R2_prefix).
Some of these may map to different genes (multi-mappers), so count-forward
mode must handle this correctly — duplicate reads must still be tracked
through the multi-mapper resolution buffer.

---

## 7. PolyA / Adapter Trimming

**Optional**, applied after protocol detection, only to `CDNA` or `GENOMIC`
segments:

| Trim type | Applied when | Method |
|---|---|---|
| PolyA (3') | 3' scRNA | ≥10 A's at 3' end (1mm) |
| PolyT (5') | 5' scRNA | ≥10 T's at 5' end (1mm) |
| TruSeq adapter | Any | AGATCGGAAGAGC match at 3' |
| Nextera adapter | Tn5-based | CTGTCTCTTATACACATCT at 3' |
| TSO artifact | SS3, 10x 5' | TSO sequence at 5' |

For **unknown** protocols: no trimming (safe default).

Trimmed bases not stored. Original length per-read in `FLAG_TRIMMED` column.
At decode: trimmed positions filled with A (polyA) or N (adapter).

---

## 8. Compression Codec Strategy

### 8.1 Codec selection

The format stores a codec ID in the header. Different .1fq files can use
different codecs. All blocks in one file use the same codec.

| Codec ID | Name | Compress | Decompress | Use case |
|---|---|---|---|---|
| 0 | zstd (level in header) | Good | ~4 GB/s | **Default** |
| 1 | lz4 | Lower | ~6 GB/s | Maximum decode speed |
| 2 | lz4hc | Better | ~6 GB/s | Better ratio, same decode |
| 3 | rANS | Varies | ~2 GB/s | Optimal for symbol streams |
| 255 | none | 1× | memcpy | Benchmarking |

### 8.2 Benchmarking plan

```
Test matrix:
  Samples: 10x v3 (100M reads), SS2 (20M), ATAC (80M), bulk (50M), Visium (30M)
  Codecs: zstd -{1,3,7,19}, lz4, lz4hc-9, rANS
  Per column: barcode dict IDs, UMI 2-bit, sorted cDNA 2-bit, quality 2-bit

  Per test: encode 100 blocks × 100K reads
  Metrics: compressed size, encode MB/s, decode MB/s
  Floor: decode > 500 MB/s (must exceed STAR's ~300 MB/s consumption)
```

### 8.3 Pre-compression transforms

| Transform | Applied to | Effect |
|---|---|---|
| **Prefix sort** | cDNA/GENOMIC columns | Long LZ matches |
| **Delta encoding** | Sorted 2-bit sequences | Smaller diffs |
| **Frequency sort** | BC dict IDs | Small varints dominate |
| **RLE** | N-bitmaps, quality | Compress sparse data |

---

## 9. File Format (binary layout)

```
┌───────────────────────────────────────────────────┐
│  Header (96 bytes, fixed)                         │
├───────────────────────────────────────────────────┤
│  Metadata Block (compressed JSON)                 │
│    Protocol, confidence, segment maps, STARsolo   │
│    params, autodetect diagnostics, accession,     │
│    all validation signals, trimming stats          │
├───────────────────────────────────────────────────┤
│  Barcode Dictionary(ies) (optional, compressed)   │
├───────────────────────────────────────────────────┤
│  Data Block 0 (compressed)                        │
│  Data Block 1 (compressed)                        │
│  ...                                              │
│  Data Block N-1 (compressed)                      │
├───────────────────────────────────────────────────┤
│  Block Index (offsets + sizes)                    │
├───────────────────────────────────────────────────┤
│  Footer (16 bytes, fixed)                         │
└───────────────────────────────────────────────────┘
```

### 9.1 Header (96 bytes)

```c
struct OneFQHeader {
    uint8_t  magic[4];          //  0: "1FQ\0"
    uint16_t version;           //  4: format version (3)
    uint8_t  n_streams;         //  6: read streams present (1–4)
    uint8_t  protocol_id;       //  7: protocol enum (0=UNKNOWN)
    uint8_t  confidence;        //  8: 0=NONE 1=LOW 2=MED 3=HIGH 4=MANUAL 5=FORCE
    uint8_t  flags;             //  9: see below
    uint8_t  codec;             // 10: compression codec enum
    uint8_t  codec_level;       // 11: codec level
    uint16_t stream_lengths[4]; // 12: per-stream length (0=variable)
    uint64_t n_unique;          // 20: unique reads/spots stored
    uint64_t n_original;        // 28: original total spots
    uint32_t block_count;       // 36: number of data blocks
    uint32_t block_size;        // 40: target reads per block
    uint32_t meta_size;         // 44: metadata block compressed size
    uint32_t bc_dict_size;      // 48: barcode dict compressed size
    uint16_t registry_version;  // 52: protocol registry version
    uint8_t  seq_encoding;      // 54: 0=2bit_packed, 1=byte_numeric
    uint8_t  qual_mode;         // 55: 0=none, 1=binned4, 2=full_phred
    uint8_t  tier;              // 56: 0=tier1(raw), 1=tier2(ref-compressed)
    uint8_t  reserved[39];      // 57–95: zero-filled
};
```

### 9.2 Flags byte

```
FLAG_DEDUPED       = 0x01  // PCR duplicate counts present
FLAG_SORTED        = 0x02  // Reads prefix-sorted within blocks
FLAG_TRIMMED       = 0x04  // PolyA/adapter trimming applied
FLAG_BC_DICT       = 0x08  // Barcode dictionary encoding active
FLAG_BC_FILTERED   = 0x10  // Non-whitelist barcodes discarded
FLAG_DELTA         = 0x20  // Delta encoding on sorted sequences
FLAG_REF_COMPRESS  = 0x40  // Tier 2: R2 is reference-compressed
FLAG_INCOMPLETE    = 0x80  // Download was interrupted; valid to last block
```

### 9.3 Metadata Block (compressed JSON)

Stores everything needed for singlify to configure STAR:

```json
{
  "accession": "SRR29320040",
  "protocol": "10x-3p-v3",
  "confidence": "HIGH",
  "confidence_score": 0.92,
  "star_params": {
    "soloType": "CB_UMI_Simple",
    "soloCBstart": 1,
    "soloCBlen": 16,
    "soloUMIstart": 17,
    "soloUMIlen": 12,
    "clipAdapterType": "CellRanger4",
    "soloBarcodeReadLength": 0
  },
  "autodetect": {
    "whitelist_match_rate": 0.87,
    "whitelist_hamming1_rate": 0.93,
    "umi_entropy_per_position": [1.98, 1.97, 1.99, 1.95, 1.96, 1.98,
                                  1.97, 1.99, 1.94, 1.96, 1.98, 1.97],
    "polya_fraction": 0.42,
    "tso_fraction": 0.01,
    "tn5_fraction": 0.00,
    "gc_content_insert": 0.47,
    "knee_n_cells": 8432,
    "candidates_tested": [
      {"tag": "10x-3p-v3", "wl_match": 0.93, "score": 0.92, "result": "SELECTED"},
      {"tag": "10x-3p-v2", "wl_match": 0.02, "score": 0.12,
       "result": "REJECTED", "reason": "whitelist match 2%"}
    ]
  },
  "sra_metadata_protocol": "10x Chromium v3",
  "sra_metadata_agrees": true,
  "reads": [
    {"index": 0, "label": "R1", "length": 28,
     "segments": [
       {"type": "BC", "offset": 0, "length": 16, "whitelist": "10x_v3"},
       {"type": "UMI", "offset": 16, "length": 12}
     ]},
    {"index": 1, "label": "R2", "length": 0,
     "segments": [{"type": "CDNA", "offset": 0, "length": 0}]}
  ],
  "trimming": {
    "applied": true,
    "polya_trimmed_fraction": 0.38,
    "adapter_trimmed_fraction": 0.05,
    "mean_bases_trimmed": 12.3
  },
  "dedup": {
    "applied": false,
    "reason": "stream_encode_mode"
  },
  "ref_compression": {
    "applied": false,
    "tier": 1
  },
  "encode_date": "2026-04-09",
  "encoder_version": "1fq-0.3.0"
}
```

### 9.4 Barcode Dictionary (optional)

```
[uint8_t n_dictionaries]
For each:
  [uint8_t  stream_index]
  [uint8_t  segment_index]
  [uint32_t n_entries]
  [uint8_t  bc_length]
  [byte_numeric sequences: n_entries × bc_length bytes]
```

Barcodes stored as byte-numeric (0/1/2/3) for direct use by singlify.
Frequency-sorted: index 0 = most common barcode.

### 9.5 Data Block Format

Each block: ~100K reads, independently decodable.

```
[uint32_t n_reads_in_block]
[uint32_t compressed_size]
[uint32_t raw_crc32]
[compressed_payload]:

  // Per read stream, per segment:
  If BC segment with dict:
    [varint dict_index × n_reads]
  Else (UMI, CDNA, GENOMIC, RAW, FEATURE, INDEX):
    If seq_encoding == 2bit_packed:
      [2-bit packed × n_reads, + N-bitmap if any N's]
    If seq_encoding == byte_numeric:
      [byte(0/1/2/3/4) × length × n_reads]
  If variable length:
    [varint length × n_reads]

  // Tier 2 ref-compressed CDNA (replaces raw CDNA above):
  If FLAG_REF_COMPRESS:
    [uint32_t ref_tid × n_reads]
    [uint32_t ref_pos × n_reads]
    [uint8_t  strand × n_reads]         // 0/1 packed as bits
    [uint8_t  n_mismatches × n_reads]
    [varint   mismatch_data[]]          // (pos, base) pairs
    [varint   softclip_len × n_reads]
    [2-bit    softclip_bases[]]
    [bitmap   unmapped_reads]           // fallback to raw for unmapped
    [raw 2-bit for unmapped reads]

  // Trailing optional sections:
  [dup_counts: varint × n_reads]        // if FLAG_DEDUPED
  [trim_lengths: varint × n_reads]      // if FLAG_TRIMMED
  [quality_r2: 2-bit × n_reads × r2_len] // if qual_mode == binned4
  [quality_r2: 6-bit × n_reads × r2_len] // if qual_mode == full_phred
```

### 9.6 Block Index

```
[uint64_t block_offset[block_count]]
[uint32_t block_comp_size[block_count]]
```

### 9.7 Footer (16 bytes)

```c
struct OneFQFooter {
    uint32_t file_crc32;
    uint32_t block_count;
    uint32_t index_offset;
    uint8_t  magic[4];     // "1FQ\0"
};
```

---

## 10. Size Estimates

### 10.1 Typical 10x v3 sample (100M reads, 100bp R2)

| Configuration | Per sample | 91K catalog | vs .sra |
|---|---|---|---|
| Gzipped FASTQ (R1+R2) | ~8.0 GB | ~728 TB | 0.6× |
| .sra (NCBI VDB) | ~5.0 GB | ~455 TB | 1× |
| .1fq tier 1 stream (no qual, no dedup) | ~1.8 GB | ~164 TB | 2.8× |
| .1fq tier 1 stream (binned4 qual) | ~2.0 GB | ~182 TB | 2.5× |
| .1fq tier 1 + dedup (binned4 qual) | ~1.1 GB | ~100 TB | 4.5× |
| .1fq tier 1 + dedup + sort (binned4) | ~0.8 GB | ~73 TB | 6.3× |
| .1fq tier 2 ref-compress (binned4, dedup) | ~0.4 GB | ~36 TB | 12.5× |

### 10.2 Bulk RNA-seq (50M PE150 reads)

| Configuration | Per sample | vs .sra |
|---|---|---|
| .sra | ~2.5 GB | 1× |
| .1fq tier 1 (binned4, sorted) | ~0.9 GB | 2.8× |
| .1fq tier 1 + dedup (binned4) | ~0.6 GB | 4.2× |

---

## 11. Decode Performance

Single-thread decode: ~67M reads/sec (100K-read blocks, zstd decompress +
2-bit unpack to byte-numeric).

STAR consumes at ~3M reads/sec → decode provides **22× headroom**.
Decode is never the bottleneck.

Memory for decode: one block buffer (~25 MB uncompressed for 100K reads).

---

## 12. Tools

### 12.1 `1fq encode`

```
1fq encode [OPTIONS] -o output.1fq <input>

Input:
  input.fastq.gz [R2.fastq.gz]   FASTQ files
  --sra FILE                      Local .sra file (VDB)
  --accession SRRxxxxxxx          Stream from NCBI (never store .sra)

Options:
  --protocol TAG        Force protocol (skip auto-detect)
  --protocol auto       Auto-detect (default)
  --quality none|binned|full   Quality mode (default: binned)
  --no-dedup            Skip PCR duplicate collapsing
  --no-trim             Skip polyA/adapter trimming
  --codec zstd|lz4|...  Compression codec (default: zstd)
  --codec-level N        Codec level (default: 3)
  --block-size N         Reads per block (default: 100000)
  --registry FILE        Custom protocol registry
  --verbose              Print detection diagnostics
```

### 12.2 `1fq refcompress` (tier 1 → tier 2)

```
1fq refcompress -i input.1fq -o output.1fq \
    --genome-dir /path/to/star/genome \
    --alignments /path/to/singlify/Aligned.bam
```

### 12.3 `1fq inspect`

```
1fq inspect input.1fq

Output:
  File: SRR29320040.1fq (tier 1)
  Format: v3, codec: zstd-3
  Protocol: 10x-3p-v3 (HIGH, score: 0.92)
  Streams: 2 (R1: 28bp, R2: variable)
  Quality: binned-4
  Reads: 72,743,000 (no dedup)
  Blocks: 728 × 100K
  Size: 1,847 MB (2.7× vs .sra)
  STARsolo: CB_UMI_Simple, CB=1:16, UMI=17:28
  Whitelist match: 93% (hamming-1)
  UMI entropy: 1.97 bits/pos
  PolyA: 42% of reads
```

### 12.4 `1fq decode` (.1fq → FASTQ, for interop)

```
1fq decode input.1fq -o R1.fastq.gz R2.fastq.gz
1fq decode input.1fq --fifo /tmp/R1.fifo /tmp/R2.fifo   # streaming
```

---

## 13. Implementation Roadmap

### Phase 1: Streaming encode + singlify native read

1. `lib1fq`: C library for encode/decode (single header, like stb)
2. `SraReader` → `Sra1fqEncoder`: VDB cursor → .1fq blocks (stream mode)
3. `1fqBlockReader`: block-by-block decode to numeric bytes
4. singlify `--1fq` mode: block reader → STAR Read1[] buffers
5. VDB accession streaming: direct NCBI download → .1fq

### Phase 2: Protocol-aware optimizations

6. Chemistry auto-detect (phase 1-6 algorithm)
7. Barcode dictionary encoding
8. Pre-parsed BC → skip STARsolo barcode matching
9. PolyA/adapter trimming

### Phase 3: Dedup + reference compression

10. PCR duplicate collapsing (offline pass)
11. Count-forward mode in singlify pileup
12. Reference-aware R2 compression (tier 2)

### Phase 4: Read folding + alignment cache

13. R2-only sequence dedup in .1fq encoder (unique_seq_id mapping)
14. Alignment result cache in singlify core
15. Count-forward with cached alignments in pileup
16. Prefix-based seed cache for sorted blocks (optional)

### Phase 5: 2-bit packed alignment (vendored STAR)

17. `compareSeqToGenome` rewrite: 64-bit XOR (32 bases/op)
18. Packed genome loading (`G_packed` + N-bitmap)
19. SA index build from packed reads (single shift+mask)
20. Precomputed reverse complement genome strand
21. Custom genome index generation for singlify

### Phase 6: Performance tuning

22. Codec benchmarking (zstd levels, lz4, etc.)
23. Byte-numeric vs 2-bit packed benchmarking
24. Block size tuning
25. Memory-mapped block reading

---

## 14. Read Folding for Alignment Speedup

### 14.1 The opportunity

STAR aligns every read independently, even when many reads are identical.
scRNA-seq libraries are dominated by PCR duplicates and low-complexity
transcripts, so the same R2 cDNA sequence appears thousands of times.
Aligning each copy is pure waste.

Three levels of folding are possible, each with diminishing returns:

| Level | Identity key | Typical fold factor | Where it helps |
|---|---|---|---|
| **Exact R2 dedup** | BC + UMI + full R2 | 1.4–3.3× (30–70% dup) | Skip full alignment |
| **R2-only dedup** | Full R2 (ignore BC/UMI) | 3–10× | More folding, decouple from BC |
| **Prefix-50bp dedup** | R2 first 50 bp | 5–15× | Seed sharing, extension diverges |

### 14.2 Level 1 — Exact BC+UMI+R2 dedup (already in §6)

Already specified. Each unique (BC, UMI, R2) triple is stored once with a
duplicate count. Count-forward mode emits each unique read once to STAR, then
multiplies pileup increments by the count.

**Savings**: For a 10x v3 sample with 100M reads and 60% duplication: 100M →
40M unique reads = **2.5× STAR speedup**. This is already in the spec.

### 14.3 Level 2 — R2-only sequence dedup (alignment cache)

Many reads from *different* barcodes or with *different* UMIs share the
same R2 cDNA sequence (same transcript fragment, independent capture events).
For alignment purposes, all we need is the R2→genome mapping; the BC/UMI
only matters during pileup counting.

**Implementation — alignment result cache**:
```
R2-only dedup (in .1fq encoding):
  1. Sort reads by R2 sequence (already in §8.3 prefix sort)
  2. Assign a unique_seq_id to each distinct R2
  3. Store mapping: read_idx → unique_seq_id in block metadata
  4. During alignment: align each unique_seq_id once
  5. During pileup: for each read, look up alignment of its unique_seq_id,
     then apply BC/UMI/dup_count to pileup counting
```

**Savings**: A typical 10x v3 sample with 40M unique (BC+UMI+R2) reads
has ~8–20M unique R2 sequences (many barcodes capture identical fragments).
40M → 12M unique R2 = **additional 2–5× over level 1**.

Combined with level 1: 100M → 12M = **~8× fewer STAR alignments**.

**Caveat**: The alignment cache must store the full SAM record (CIGAR, mapping
quality, multi-mapper flags) per unique R2. For 12M unique R2s at ~40 bytes
per cached alignment: ~480 MB memory. Acceptable.

### 14.4 Level 3 — Prefix-based seed sharing

STAR's `maxMappableLength2strands` builds a suffix array index from the
first 14 bases of each seed (the `gSAindexNbases=14` lookup). Reads sharing
the same 14bp prefix hit the same SA index entry and start binary search
from the same `[iSA1, iSA2]` range.

With reads sorted by R2 (from level 2), sequential reads share long
prefixes. A **seed cache** can exploit this:

```
Seed cache (within a sorted block of reads):
  struct SeedCacheEntry {
      uint64_t prefix_28bit;   // 14 bases packed
      uint64_t iSA1, iSA2;    // SA range for this prefix
      uint     L_matched;      // extension length from previous call
  };

  For each read:
    prefix = pack_14bp(R2[0..13]);
    if (cache.prefix == prefix) {
        // Skip SA index lookup entirely
        // Start extension from cache.iSA1, cache.iSA2
    } else {
        // Normal SA index lookup, populate cache
    }
```

**Savings**: SA index lookup is cheap (~5% of STAR time), but the shared
starting range can also accelerate the extension binary search. Net benefit:
~5–10% additional speedup over level 2. Minor but free with sorted order.

### 14.5 Quantified impact

| Scenario | Reads aligned | STAR time vs baseline |
|---|---|---|
| No dedup (current singlify) | 100M | 1.0× |
| Level 1: exact BC+UMI+R2 dedup (§6) | 40M | 0.40× |
| Level 2: R2-only alignment cache | 12M | 0.12× |
| Level 2 + prefix seed cache | 12M | 0.11× |

For a sample that takes 30 minutes in STAR today: level 2 brings it to ~4
minutes. The pileup engine becomes the bottleneck.

### 14.6 Where to implement

- **Level 1**: Already specified (§6, count-forward mode). In .1fq encoder.
- **Level 2**: New alignment cache in singlify core. The .1fq encoder stores
  the `read_idx → unique_seq_id` mapping in block metadata. singlify's
  `readLoad1fq()` emits unique R2s, caches BAM results, and replays them
  during pileup with per-read BC/UMI/count.
- **Level 3**: Optional seed cache in STAR's `maxMappableLength2strands`.
  Trivial to add if reads are sorted (check if prefix matches previous).

---

## 15. 2-Bit Native Alignment: STAR Internal Analysis

### 15.1 How STAR currently encodes data

Discovered by reading the vendored STAR source:

| Data | Storage | Size (human genome) | Source file |
|---|---|---|---|
| Genome `G` | `char*`, 1 byte/base | 3.2 GB per strand | `Genome.h:26` |
| Reads `Read1[0]` | `char[]`, 1 byte/base (0/1/2/3/4) | ~91 bytes/read | `ReadAlign.h` |
| Suffix array `SA` | `PackedArray`, ~30 bits/entry | ~25 GB | `Genome.h:28` |
| SA index `SAi` | `PackedArray`, variable bits | ~1.5 GB | `Genome.h:29` |

**Key insight**: Only the suffix array indices are bit-packed. The genome
sequence and reads are stored at full byte width — 4× waste for data that
uses only 3 of 8 bits.

### 15.2 STAR's seeding path (annotated)

```
readLoad() → convertNucleotidesToNumbers()
  ASCII 'ACGTN' → byte[0,1,2,3,4] via nucl_to_num[256] lookup       ← eliminated by .1fq

mapOneRead() → qualitySplit() → loop over fragments

maxMappableLength2strands():
  // Build SA index key from bytes — ALREADY CONVERTS TO 2-BIT
  ind1 = 0;
  for (ii = 0; ii < 14; ii++) {
      ind1 <<= 2;                                                    ← 2-bit shift
      ind1 += (uint)Read1[0][pieceStart + ii];                       ← byte→2-bit
  }
  iSA1 = SAi[genomeSAindexStart[13] + ind1];                         ← SA index lookup

maxMappableLength():
  // Binary search in SA, compare read to genome BYTE BY BYTE
  compareSeqToGenome():
    for (ii = 0; ii < N-L; ii++) {
        if (s[ii] != g[ii]) {                                        ← 1 base/cycle
            return ii + L;
        }
    }
```

The seeding phase already collapses byte-numeric to 2-bit (line 28-29 of
`ReadAlign_maxMappableLength2strands.cpp`). The extension phase in
`compareSeqToGenome` (`SuffixArrayFuns.cpp:10-100`) does per-byte comparison
against the byte-numeric genome `G`.

### 15.3 What 2-bit packed alignment would look like

**Genome storage** (4× smaller, 4× better cache):

```
// Current: 3.2 GB per strand, 1 base/byte
char *G;                           // G[i] ∈ {0,1,2,3,4}

// Proposed: 0.8 GB per strand, 4 bases/byte
uint64_t *G_packed;                // 32 bases per uint64, 2 bits each
uint64_t *G_packed_rc;             // reverse complement, precomputed
uint8_t  *G_N_bitmap;             // 1 bit/base, marks N positions (sparse)
```

**SA index lookup** (trivial change):

```
// Current (14-iteration loop):
uint ind1 = 0;
for (ii = 0; ii < 14; ii++) {
    ind1 <<= 2;
    ind1 += (uint)Read1[0][pieceStart + ii];
}

// Proposed (single mask from packed read):
uint ind1 = (read_packed[0] >> (64 - 28)) & 0x0FFFFFFF;
```

**Extension comparison** (16–32× fewer operations):

```
// Current: 1 cycle per base
for (ii = 0; ii < N-L; ii++) {
    if (s[ii] != g[ii]) return ii + L;
}

// Proposed (32 bases per 64-bit XOR):
uint64_t *g_ptr = G_packed + (genome_pos / 32);
uint shift = (genome_pos % 32) * 2;
for (word = 0; word < n_words; word++) {
    uint64_t g_aligned = (g_ptr[word] << shift) | (g_ptr[word+1] >> (64 - shift));
    uint64_t diff = g_aligned ^ read_packed[word];
    if (diff != 0) {
        return word * 32 + (__builtin_ctzll(diff) / 2) + L;
    }
}
return read_length;  // full match
```

With SSE2: 64 bases per 128-bit XOR. With AVX2: 128 bases per instruction.
A 91bp 10x R2 read compares in **1–2 SIMD instructions** instead of 91
byte comparisons.

### 15.4 The N-base problem

2-bit encoding has 4 values (A=00 C=01 G=10 T=11) — no room for N.

| Strategy | Overhead | False matches | Practical? |
|---|---|---|---|
| Separate N-bitmap, check after XOR | 1 extra bit/base (amortized <0.1%) | None | Best for genome |
| Treat N as A, re-check on match | Zero | Rare (~0.1% of bases) | Best for reads |
| 3-bit encoding (A=00 C=01 G=10 T=11 N=100) | Kills 32-base-per-word property | None | No |

In practice: Illumina reads have <0.1% N bases. The genome has N runs
at centromeres/telomeres. Strategy: packed genome with N-bitmap (sparse,
RLE-compressed), packed reads treating N as A with rare re-check.

### 15.5 Complement handling

STAR's `complementSeqNumbers()` uses 8-byte XOR with `0x0303030303030303ULL`
(flips 0↔3, 1↔2 in byte-numeric space). In 2-bit space:

```
// 2-bit complement: XOR with all-1s (bit flip: 00↔11, 01↔10)
uint64_t complement = packed_read ^ 0xFFFFFFFFFFFFFFFFULL;
// Reverse: swap pairs of bits to reverse base order
uint64_t reversed = bit_reverse_pairs(complement);
```

Precomputing both strands of the genome (`G_packed` and `G_packed_rc`)
eliminates per-read complement computation entirely.

### 15.6 Performance impact analysis

| Component | Current (byte) | 2-bit packed | Speedup |
|---|---|---|---|
| **Genome memory** | 6.4 GB (both strands) | 1.6 GB (both strands) | 4× smaller |
| **L3 cache efficacy** | ~64 bases/cache line | ~256 bases/cache line | 4× more hits |
| **SA index build** | 14-iter loop | 1 shift+mask | ~10× (minor) |
| **Extension compare** | 1 base/cycle | 32 bases/64-bit XOR | 16–32× per op |
| **Overall STAR rate** | ~3M reads/sec | est. ~4.5–6M reads/sec | 1.5–2× |

**Why only 1.5–2× overall** despite 16× faster comparison:
- SA lookup (random memory access) is the true bottleneck, not comparison
- The extension comparison is only ~30–40% of total STAR time
- Random genome access (SA-indexed) limits SIMD benefit — the load is the
  bottleneck, not the XOR
- The 4× cache improvement for genome is the *real* win — fewer L3 misses
  during SA binary search

### 15.7 Implementation difficulty

| Change | Effort | Risk |
|---|---|---|
| Genome loading (byte → packed + N-bitmap) | Medium | Low |
| Read format change (byte → packed) | Low | Low (from .1fq, already packed) |
| `compareSeqToGenome` rewrite (4 variants) | Medium | Medium (edge cases) |
| `maxMappableLength2strands` index build | Low | Low |
| `complementSeqNumbers` → precomputed | Low | Low |
| Genome generation / index compatibility | High | High (index format change) |
| **Total** | **Medium-High** | **Medium** |

**Critical risk**: Changing the genome representation potentially requires
regenerating STAR genome indices. For our vendored STAR with fixed genome
builds, this is a one-time cost. But it means the modified STAR can't use
standard genome indices.

### 15.8 Recommendation

The 2-bit optimization is valuable but should be Phase 5 (after the .1fq
format and read folding are working):

1. **Read folding (§14) delivers 5–8× speedup** with minimal STAR changes
2. **2-bit alignment delivers 1.5–2× speedup** with significant STAR changes
3. Combined: **8–16× total speedup** over current singlify STAR alignment
4. The two optimizations are orthogonal — read folding reduces the number of
   STAR invocations, 2-bit makes each invocation faster

For the .1fq format: store reads in 2-bit packed format (already specified
in §3.2). This is correct regardless of whether STAR unpacks to bytes or
operates on packed data. The format doesn't need to change.

---

## 16. Open Questions

1. **VDB direct streaming reliability**: Can VDB transparently stream from
   NCBI S3 reliably for long downloads (hours)? Need to test with interruption
   recovery. Fallback: pipe from `prefetch --output-file /dev/stdout`.

2. **Dedup before or after quality encoding**: If we dedup first (collapse
   identical BC+UMI+R2), what quality score do we keep? Options: max, mean,
   first-seen. First-seen is simplest and consistent with "representative read."

3. **Tier 2 regeneration**: If we update the genome reference (new GTF), tier 2
   files are invalidated. Store reference hash in metadata to detect this.

4. **Long-read SRA**: PacBio/ONT reads use different structure (single long
   read). Support in .1fq or separate format? Likely separate — the compression
   strategies are fundamentally different.

5. **STAR code changes**: The `readLoad1fq()` function requires modifying
   STAR source. Since we maintain a vendored copy, this is feasible.
   Minimal change: new `readFilesType` value + a ~50-line read loader.

6. **R2-only alignment cache memory**: Caching BAM results for 12M unique R2s
   at ~40 bytes each = ~480 MB. Acceptable for processing, but investigate
   whether a smaller representation (CIGAR + pos ≈ 16 bytes) suffices.

7. **2-bit genome index compatibility**: A 2-bit packed genome breaks
   compatibility with standard STAR genome indices. Evaluate whether to
   maintain dual-format loading (byte for standard, packed for .1fq mode)
   or commit to a custom genome build for singlify.

8. **Read folding vs multi-mapper resolution**: When using R2-only alignment
   cache (§14.3), a single R2 may multi-map to N locations. Each originating
   (BC, UMI) pair needs independent multi-mapper resolution in pileup. Verify
   that the count-forward approach in §6.2 handles this correctly.
