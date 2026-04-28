# Alignment Software Landscape Review

**Date**: April 2026  
**Context**: Singlify pipeline uses STAR (singlet-lite branch) for full-genome spliced alignment of scRNA-seq reads. This document surveys the alignment software landscape — full aligners, pseudoaligners, and alignment-free tools — to understand algorithmic choices and identify techniques with step-change potential.

---

## Table of Contents

1. [Full Genome Aligners](#1-full-genome-aligners)
   - [STAR](#11-star)
   - [HISAT2](#12-hisat2)
   - [BWA-MEM2](#13-bwa-mem2)
   - [minimap2](#14-minimap2)
2. [Pseudoaligners](#2-pseudoaligners)
   - [kallisto](#21-kallisto)
   - [salmon / RapMap](#22-salmon--rapmap)
   - [alevin-fry / piscem](#23-alevin-fry--piscem)
3. [Hybrid & Alignment-Free Tools](#3-hybrid--alignment-free-tools)
   - [Accel-Align](#31-accel-align)
   - [STARsolo](#32-starsolo)
4. [Comparative Analysis](#4-comparative-analysis)
5. [Key Codebase Notes](#5-key-codebase-notes)

---

## 1. Full Genome Aligners

These produce base-level CIGAR alignments against a full genome reference — required for variant calling, splice junction discovery, and the full singlet-pileup feature set.

### 1.1 STAR

**Paper**: Dobin et al., Bioinformatics 2013  
**Algorithm class**: Suffix Array (SA) based seed-and-extend  
**Index**: Uncompressed suffix array (~24GB for human; can be sparsified with `--genomeSAsparseD 3` to ~8GB at 30–50% speed cost) + SAindex prefix table (14-mer L-mer lookup with cascading resolution)  

**Algorithm**:
1. **Seed finding** (`maxMappableLength`): For each position in the read, find the Maximum Mappable Prefix (MMP) — the longest suffix-array-consistent substring starting at that position.
   - Uses the SAindex prefix table to narrow the SA range with an L-mer lookup (L=14 by default). The SAindex uses mark bits (`SAiMarkAbsentMask`, `SAiMarkNmask`) to encode absent L-mers without wasting space.
   - Binary search within the SA range, comparing read sequence to genome at `G[SA[mid]]` using word-level XOR (8 bytes at a time, 4 strand combinations).
   - Post-SAi, the binary search typically needs only 2–4 iterations (range is already narrow), but each iteration causes a random DRAM access into the 3.1GB genome.
2. **Seed stitching** (`stitchPieces`): Seeds from both strands are stitched together using a scored dynamic programming approach that handles indels, splice junctions, and chimeric alignments.
3. **Splice junction handling**: Known SJ annotations from GTF are used to seed; novel junctions discovered from GT/AG dinucleotides.
4. **Transcriptome alignment**: Mapped genomic coordinates are projected onto transcriptome coordinates for quantification.

**Key data structures** (from `Genome.h`):
- `G` — byte-encoded genome (A=0, C=1, G=2, T=3) ~3.1GB for human
- `SA` — PackedArray suffix array (33-bit entries for hg38) ~12.4GB (or 4.1GB at sparseD=3)
- `SAi` — PackedArray prefix index with mark bits, multiple resolution levels
- `genomeSAindexStart[]` — offsets for each L-mer prefix level

**Key source files** (singlet-lite branch):
- `ReadAlign_maxMappableLength2strands.cpp` (lines 24–48): SAindex prefix lookup, cascading L-mer search
- `SuffixArrayFuns.cpp` (lines 607–735): Core `maxMappableLength()` binary search; (lines 14–150): `compareSeqToGenome` with word-level XOR
- `ReadAlign_mapOneRead.cpp` (lines 35–115): Per-piece mapping, both strands
- `ReadAlign_stitchPieces.cpp`: Piece stitching with SA_LAZY_WINBIN optimization

**Performance**: ~48% faster than stock in singlet-lite branch via PGO+LTO, 14-mer prefix sort, consecutive dedup, NUMA interleave, lazy winBin, boundary prefetch. Remaining bottle-neck: **42% of wall time is `G[SA[mid]]` random DRAM access** — a data-dependent pointer chase that cannot be prefetched within sequential binary search.

**Memory**: ~32GB (full SA) or ~16GB (sparse SA at `sparseD=3`)

---

### 1.2 HISAT2

**Paper**: Kim, Langmead & Salzberg, Nature Biotechnology 2019  
**Algorithm class**: Hierarchical Graph FM-Index (HGFM)  
**Index**: One global GFM + ~55,000 local GFM indexes (each covering ~56Kbp)

**Algorithm**:
1. **Hierarchical indexing**: Global FM-index narrows to a genomic region, then local FM-index provides fine-grained alignment. This two-level approach gives FM-index accuracy with dramatically lower memory than a monolithic FM-index.
2. **Graph FM-index**: Extends BWT to encode known genomic variants (SNPs, small indels) directly in the index graph. Reads containing known variants align without penalty — crucial for population-level alignment.
3. **Splice-aware**: Inherits from HISAT/TopHat2 lineage. Supports known and novel splice junctions.
4. **Repeat handling** (v2.2.0): Dedicated repeat index for reads mapping to >5 locations, producing one repeat alignment per read with API access to all positions.

**Strengths**:
- Very low memory footprint: **6.7GB** for human genome + 12.3M SNPs
- Graph-aware alignment removes reference bias at known variant sites
- Well-suited for population genetics workflows

**Weaknesses**:
- Slower than STAR for scRNA-seq (30–80% additional CPU time vs non-graph mode)
- FM-index character-by-character traversal is inherently serial; no equivalent of STAR's word-level XOR comparison
- Local index switching adds overhead for highly multi-mapping reads
- No integrated scRNA-seq barcode/UMI handling (unlike STARsolo)

**Key insight for us**: The hierarchical local/global index architecture is conceptually related to our "cached genomic blocks" idea — pre-organizing the reference by spatial locality to improve cache performance.

---

### 1.3 BWA-MEM2

**Paper**: Vasimuddin Md et al., IEEE IPDPS 2019  
**Algorithm class**: FM-index seeding + Smith-Waterman extension (SIMD-accelerated)  
**Index**: FM-index (2bit.64 format), ~10GB for human genome on disk

**Algorithm** (same as BWA-MEM, but architecture-optimized):
1. **Seeding**: Find Super-Maximal Exact Matches (SMEMs) using FM-index — seeds that cannot be extended in either direction.
2. **Chaining**: Chain seeds along the reference.
3. **Extension**: Smith-Waterman alignment using SIMD (SSE/AVX) vectorized DP.

**Performance optimizations** (from the paper — directly relevant to our work):
- **Cache reuse**: Reorganized memory access patterns for L1/L2/L3 cache locality
- **Simplified algorithms**: Reduced branch mispredictions and instruction count
- **Contiguous allocations**: Replaced fragmented small allocations with large contiguous buffers
- **Software prefetching**: Prefetch index entries before they're needed
- **SIMD utilization**: Vectorized the three dominant kernels

**Result**: 2× speedup on seed finding, 183× on a sorting kernel, 8× on extension → **3.5× end-to-end single-thread speedup** over BWA-MEM.

**Experimental branches**:
- **LISA (Learned Indexes for Sequence Analysis)**: Replaces FM-index seeding with learned index structures. **4.5× speedup on seeding** but 120GB memory footprint. *This is the most directly relevant prior work for our "learned custom genome index" idea.*
- **ERT (Enumerated Radix Trees)**: Replaces FM-index with radix tree for seeding. 10–30% end-to-end speedup, ~60GB index.

**Key insight for us**: BWA-MEM2's systematic approach to architecture-aware optimization (cache, prefetch, SIMD, allocation) mirrors our singlet-lite work. Their LISA branch demonstrates that learned indexes can dramatically speed seeding but at massive memory cost — the key challenge is making learned indexes practical in the 16–32GB memory regime.

---

### 1.4 minimap2

**Paper**: Li, Bioinformatics 2018  
**Algorithm class**: Minimizer-based seed-chain-align  
**Index**: (k,w)-minimizer hash table

**Algorithm**:
1. **Index construction**: Extract (k,w)-minimizers from reference (default k=15, w=10). Store in hash table. Only minimizers (position-minimum within each window of w consecutive k-mers) are indexed — this is ~1/(w+1) of all k-mers.
2. **Seed collection**: For each query minimizer, look up hits in hash table. Filter seeds that are too frequent (`-f 0.0002` — skip k-mers occurring in >0.02% of the genome).
3. **Chaining**: Sort seeds by reference position. Dynamic programming chaining with gap penalties to find co-linear seed chains.
4. **Alignment extension**: Base-level Smith-Waterman with Z-drop heuristic (stop if score drops too far below peak).

**For short RNA-seq reads** (`-x splice:sr`):
- Uses k=15, intron-length-aware chaining.
- Supports annotation-guided alignment with `--junc-bed`.
- **3× faster than BWA-MEM** for short reads.

**Strengths**:
- Extremely fast and versatile (long reads, short reads, assembly, splice-aware)
- Minimizer index is compact and cache-friendly
- Frequency-based seed filtering elegantly handles repeats
- Chaining is highly efficient for finding optimal seed orderings

**Weaknesses**:
- Default k=15 is too short for uniqueness in human genome — mitigated by frequency filter but adds noise
- Not designed with scRNA-seq barcodes/UMI in mind
- No suffix array — cannot find maximum mappable lengths as precisely as STAR
- Z-drop heuristic can miss distant exon junctions (addressed by `--junc-bed`)

**Key insight for us**: The minimizer concept — indexing only a deterministic subset of k-mers — drastically reduces index size while maintaining sensitivity. Combined with frequency-based filtering these are building blocks for our "multi-species k-mer filter" idea.

---

## 2. Pseudoaligners

These determine *which* transcript(s) a read originates from without producing base-level alignments. Much faster than full alignment but lose positional/CIGAR information.

### 2.1 kallisto

**Paper**: Bray et al., Nature Biotechnology 2016  
**Algorithm class**: Colored compacted de Bruijn graph (CDBG) + pseudoalignment to equivalence classes  
**Index**: T-DBG (transcriptome de Bruijn graph) with minimal perfect hash function (MPHF, via BooPHF)

**Algorithm**:
1. **Index construction** (`BuildDeBruijnGraph`, `BuildEquivalenceClasses`):
   - Build a compacted de Bruijn graph from the transcriptome using the Bifrost library.
   - Walk each transcript through the CDBG. For each unitig encountered, record which transcripts contain it.
   - Assign equivalence classes (ECs): sets of transcripts that share a unitig. Mosaic ECs handle unitigs that span boundaries between transcript subsets — the unitig is split at positions where the transcript set changes.
   - EC cardinality capped at 250 by default to avoid combinatorial explosion from highly-shared regions.

2. **Pseudoalignment** (`match()` function in `KmerIndex.cpp`):
   - For each k-mer in the read, look up the unitig in the CDBG using MPHF.
   - Skip-jump optimization: if the current k-mer maps to the same unitig as the previous one, advance by the remaining unitig length without per-base hashing — amortizes the MPHF lookup cost.
   - Intersect ECs across all k-mers in the read to get the final EC assignment.
   - D-list filtering: reads matching a "decoy list" (e.g., genomic sequences, contaminants) are filtered out. This is conceptually similar to our "multi-species k-mer filter" idea.

3. **Quantification**: EM algorithm distributes reads among transcripts based on their EC memberships.

**Key data structures**:
- MPHF (minimal perfect hash function): Maps each k-mer to a unique integer in [0, n). No collisions, O(n) space, O(1) lookup. Uses BooPHF library.
- Equivalence class table: EC_id → {transcript_id set}
- Unitig table: unitig_id → (sequence, EC*, length, position info)

**Key source files** (`kallisto/src/KmerIndex.cpp`):
- `BuildTranscripts()`: Read FASTA, polyA clipping
- `BuildDeBruijnGraph()`: CDBG construction via Bifrost (configurable k, minimizer length g)
- `BuildEquivalenceClasses()`: Walk transcripts through CDBG, assign unitig→transcript maps, mosaic EC creation
- `PopulateMosaicECs()`: Break unitigs at transcript boundary positions, create per-segment ECs
- `match()`: Core pseudoalignment — k-mer iteration, CDBG lookup, skip-jumping, EC intersection
- `match_long()`: Long-read variant with incremental search fallback

**Performance**: **Orders of magnitude faster than alignment** for transcript quantification. ~10M reads/minute single-threaded. Index is ~2–4GB for human transcriptome.

**Limitations**:
- Transcriptome-only — cannot discover novel junctions, intergenic reads, or intronic reads
- No CIGAR strings — cannot do variant calling, splice junction refinement, or exon/intron counting
- K-mer-based — a single sequencing error in a k-mer breaks the chain; relies on redundancy across the read
- EC intersection can collapse for highly multi-mapped reads

**Key insight for us**: kallisto's skip-jumping along unitigs is brilliant — it amortizes hash lookup cost by exploiting the sequential structure of de Bruijn graphs. The D-list concept for filtering contaminant species is directly applicable to our multi-species pre-screening idea.

---

### 2.2 salmon / RapMap

**Paper**: Patro et al., Nature Methods 2017 (salmon); Srivastava et al., Bioinformatics 2016 (RapMap)  
**Algorithm class**: Quasi-mapping (RapMap) → selective alignment (salmon)  

**Quasi-mapping algorithm** (RapMap):
1. Extract k-mers from read, look up in hash table built over transcriptome.
2. For each matching position, extend greedily using suffix array to find maximal matches.
3. Chain matches into quasi-mappings — similar to minimap2's seed-chain but without base-level extension.
4. Report transcript hits and approximate positions without full CIGAR.

**Selective alignment** (salmon, later development):
- After quasi-mapping, perform base-level alignment of the read to the candidate transcripts.
- Uses a scoring model that penalizes mismatches, gaps, and soft-clips.
- Significantly improves accuracy over pure quasi-mapping, at moderate speed cost.
- Still transcriptome-only but more robust to sequencing errors.

**Quantification innovations**:
- **Online lightweight inference**: Fragment-level assignment with EM-like variational inference.
- **GC bias correction**: Models fragment-level GC content bias.
- **Sequence-specific bias**: Models non-uniform read starts from sequence context.
- **USA mode** (alevin-fry): Reports Unspliced, Spliced, and Ambiguous counts separately — important for RNA velocity.

**Key insight for us**: salmon's selective alignment is a "hybrid" approach — fast pre-screening followed by targeted base-level alignment. This two-phase strategy (cheap filter → expensive verification) is exactly the paradigm we should adopt.

---

### 2.3 alevin-fry / piscem

**Paper**: He et al., Nature Methods 2022  
**Tool chain**: piscem (Rust mapper) → alevin-fry (Rust quantifier)  
**Index**: Colored compacted de Bruijn graph via cuttlefish2 + SSHash (MPHF on k-mers organized by minimizer)

**Architecture**:
1. **piscem build**: Constructs a compacted colored de Bruijn graph from reference FASTAs using cuttlefish2. Builds an SSHash index over k-mers organized by minimizer classes. Produces equivalence class lookup table.
2. **piscem map-sc**: Maps scRNA-seq reads against the piscem index. Outputs RAD (Reduced Alignment Data) format — a compact binary format encoding (barcode, UMI, EC) triples.
   - `--check-ambig-hits`: Extra checking of k-mers too ambiguous for chaining — improves specificity.
   - Geometry specification (`--geometry`): Flexible barcode/UMI layout syntax (`1{b[16]u[12]x:}2{r:}`).
3. **alevin-fry generate-permit-list**: Cell barcode filtering (external whitelist or knee-point).
4. **alevin-fry collate**: Sort RAD records by cell barcode for streaming quantification.
5. **alevin-fry quant**: Per-cell UMI deduplication and gene quantification. Supports `cr-like` (CellRanger-like) and various other resolution strategies.

**Key innovations**:
- **SSHash**: Minimal perfect hash function organized by minimizer classes — more cache-friendly than random MPHF because k-mers sharing a minimizer are stored contiguously.
- **RAD format**: Binary intermediate format. Much smaller than BAM, but carries the information needed for scRNA quantification. Avoids the overhead of full SAM/BAM.
- **Splici index**: "spliced + intronic" reference. Indexes both the transcriptome and intronic sequences flanked by read-length extensions around exon boundaries. Enables USA-mode quantification (spliced, unspliced, ambiguous) without full genome alignment.
- **Mapped-filtered-EC**: Intermediate representation that preserves enough information for downstream quantification while discarding redundant mapping details.

**Performance**: Fastest scRNA-seq quantification pipeline currently available. piscem mapping is ~2–5× faster than salmon alevin with similar or better accuracy.

**Key insight for us**: The splici index concept is clever — it captures most of the intron read information (critical for RNA velocity / pre-mRNA) without full genome alignment. However, it cannot detect novel junctions, intergenic reads, or variants. The SSHash data structure (minimizer-organized MPHF) is relevant to our learned index ideas.

---

## 3. Hybrid & Alignment-Free Tools

### 3.1 Accel-Align

**Paper**: Yan, Chaturvedi & Appuswamy, BMC Bioinformatics 2021  
**Algorithm class**: Seed–Embed–Extend  

**Algorithm**:
1. **Seeding**: Extract 32-mers from the read, look up in a simple hash table built from the reference genome.
2. **Embedding**: Convert candidate genomic regions and the read into compact vector representations (embeddings). Compute Hamming distance between embeddings to rapidly filter out poor candidates — **much cheaper than Smith-Waterman**.
3. **Extension**: Only for candidates surviving the embedding filter, run full base-level alignment (KSW2 or WFA).

**Key innovation**: The embedding step acts as a learned/approximate filter between seeding and extension. By converting sequences to low-dimensional vectors and comparing distances, it eliminates ~90% of false-positive seed extensions without running full DP alignment.

**Modes**:
- Full alignment mode (default): Seed → Embed → Extend with CIGAR output
- Alignment-free mode (`-x`): Reports position without CIGAR (just seed + embed)

**Performance**: Competitive with BWA-MEM and Bowtie2 in accuracy. Speed improvements come primarily from avoiding unnecessary Smith-Waterman calls.

**Key insight for us**: The embed-and-filter paradigm is directly applicable to STAR's `G[SA[mid]]` bottleneck. Instead of doing full genome comparison at every binary search step, we could use a compact embedding of the SA neighborhood to quickly accept/reject candidates. This is a form of "learned approximate comparison."

---

### 3.2 STARsolo

**Paper**: Kaminow, Yunusov & Dobin, bioRxiv 2021  
**Architecture**: STAR alignment + integrated scRNA-seq processing

STARsolo extends STAR with:
- **Barcode extraction**: Cell barcode + UMI parsing from R1 reads with 1-mismatch correction against a whitelist (supports 10X v2/v3, inDrop, Smart-seq, complex barcodes).
- **UMI deduplication**: `1MM_All`, `1MM_Directional`, `1MM_CR` (CellRanger-compatible), `Exact`, `NoDedup`.
- **Multi-gene recovery**: Uniform, PropUnique, EM, Rescue strategies for reads mapping to multiple genes.
- **Cell filtering**: CellRanger2.2 knee filter and EmptyDrops_CR.
- **Velocyto** mode: Spliced/Unspliced/Ambiguous counting (similar to velocyto.py).
- **Quantification features**: Gene, GeneFull (exon+intron), SJ (splice junctions).

**Matching CellRanger**: With `--soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR --clipAdapterType CellRanger4 --outFilterScoreMin 30`, produces nearly identical counts to CellRanger 4/5.

**Performance**: ~10× faster than CellRanger (same alignment, less overhead). Uses the same genome index as standard STAR.

**Key insight for us**: STARsolo's feature set is the closest "official" equivalent to what singlify does. Our singlet-pileup goes further (SNP detection, expression-aware variant calling, richer intron/exon model), but STARsolo demonstrates that genome-aligned scRNA processing is the production standard.

---

## 4. Comparative Analysis

### Index Structures

| Tool | Index Type | Size (human) | Memory | Build Time |
|------|-----------|-------------|--------|------------|
| STAR | Suffix Array + SAindex | 24GB (full), 8GB (sparse) | 32GB / 16GB | ~30 min |
| HISAT2 | Hierarchical Graph FM-index | 6.2GB | 6.7GB | ~60 min |
| BWA-MEM2 | FM-index (2bit.64) | 10GB | 10GB | ~60 min |
| minimap2 | Minimizer hash table | ~7GB | ~8GB | ~3 min |
| kallisto | CDBG + MPHF | ~2GB (txome) | ~4GB | ~10 min |
| salmon | SA + hash table (txome) | ~1GB (txome) | ~8GB | ~10 min |
| piscem | CDBG + SSHash | ~2GB (txome) | ~4GB | ~10 min |

### Algorithmic Paradigms

| Paradigm | Representative | Seed Strategy | Extension | Output |
|----------|---------------|--------------|-----------|--------|
| SA binary search | STAR | Max Mappable Prefix via SA | Stitch + score | Full CIGAR |
| FM-index traverse | HISAT2, BWA-MEM2 | SMEM / exact match | DP extension | Full CIGAR |
| Minimizer seed-chain | minimap2 | (k,w)-minimizer lookup | DP chain + align | Full CIGAR |
| CDBG pseudoalign | kallisto | k-mer unitig walk | EC intersection | Transcript set |
| Quasi-map + sel-align | salmon | k-mer / SA hybrid | Optional DP | Transcript + pos |
| Seed-embed-extend | Accel-Align | 32-mer hash | Embed filter + DP | Full CIGAR |

### Speed Hierarchy (approximate, short scRNA-seq reads)

1. **kallisto/piscem** (pseudoalignment): ~10–30M reads/min — no CIGAR
2. **salmon selective alignment**: ~3–10M reads/min — transcriptome CIGAR
3. **minimap2** (`splice:sr`): ~2–5M reads/min — genome CIGAR
4. **STAR** (singlet-lite): ~1.5–3M reads/min (8 threads, warm) — genome CIGAR
5. **HISAT2**: ~1–2M reads/min — genome CIGAR
6. **BWA-MEM2**: ~1–3M reads/min — genome CIGAR (not splice-aware)

### What We Need vs. What Each Tool Provides

| Requirement | STAR | kallisto | salmon | minimap2 | HISAT2 |
|------------|------|---------|--------|----------|--------|
| Full genome CIGAR | ✅ | ❌ | ❌ (txome only) | ✅ | ✅ |
| Splice-aware | ✅ | N/A | N/A | ✅ (`splice:sr`) | ✅ |
| Intron reads | ✅ | ❌ | ✅ (splici) | ✅ | ✅ |
| Novel junctions | ✅ | ❌ | ❌ | ✅ | ✅ |
| Variant detection | ✅ (via CIGAR) | ❌ | ❌ | ✅ | ✅ |
| scRNA barcode/UMI | ✅ (STARsolo) | ✅ (bustools) | ✅ (alevin) | ❌ | ❌ |
| Low memory (~16GB) | ✅ (sparse) | ✅ | ✅ | ✅ | ✅ |

**Conclusion**: For the singlify pipeline's requirements (full genome CIGAR with variant calling, splice junctions, exon/intron classification, and SNP detection), **only full genome aligners work**. The question is not whether to replace full alignment, but **how to make it faster using ideas from pseudoaligners and alignment-free methods**. Critically, our optimizations must be **species-agnostic** — working for any Ensembl reference genome without human-specific tuning. See ALIGNMENT_ALGORITHMS_PROPOSAL.md proposals #11 (Adaptive Runtime Reference Prioritization) and #12 (Species-Agnostic Self-Tuning Architecture) for the design principles that ensure this.

---

## 5. Key Codebase Notes

### STAR (singlet-lite branch) — `/mnt/home/debruinz/Singlet-AI/STAR/source/`

| File | Lines | Insight |
|------|-------|---------|
| `ReadAlign_maxMappableLength2strands.cpp` | 24–48 | SAindex prefix lookup: cascading L-mer search with mark bits. The SAiMarkAbsentMask and SAiMarkNmask encode the absence or exact count of SA entries for each prefix — elegantly avoids storing empty prefixes. |
| `SuffixArrayFuns.cpp` | 607–735 | The core binary search. `compareSeqToGenome` at lines 14–150 loads 8 bytes at a time, XORs, and uses `__builtin_ctzll` to find the first mismatch. This is the hottest loop in the entire pipeline. |
| `SuffixArrayFuns.cpp` | 14–150 | `compareSeqToGenome`: 4 strand combinations (forward/reverse × read/complement). Word-level XOR comparison is the key pattern — 8 genome bases compared per instruction. |
| `ReadAlign_stitchPieces.cpp` | all | Piece stitching with SA_LAZY_WINBIN. The winBin array (~94KB) tracks covered genome windows; lazy reset avoids memset per read. |
| `ReadAlign_mapOneRead.cpp` | 35–115 | Per-piece mapping, both strands, multiple attempts. The retry logic with lengthened seeds handles edge cases. |
| `Genome.h` | all | Core data structures: `G` (byte genome), `SA` (PackedArray, 33-bit entries), `SAi` (PackedArray with mark bits), `genomeSAindexStart[]`. |
| `SoloBarcode_extractBarcode.cpp` | all | Barcode extraction and whitelist matching. Open-addressing hash in singlet-lite for fast lookup. |
| `SoloFeature_collapseUMI_Graph.cpp` | 30–45 | UMI dedup via graph coloring with directional collapse — the standard scRNA approach. |

### kallisto — `kallisto/src/KmerIndex.cpp`

| Function | Insight |
|----------|---------|
| `BuildDeBruijnGraph()` | Uses Bifrost CDBG library with configurable k and minimizer length g. The CDBG construction is the memory-intensive step. |
| `BuildEquivalenceClasses()` | Walks each transcript through the CDBG, collecting unitig→transcript mappings. Mosaic ECs are created at unitig positions where the transcript set changes. EC cardinality capped at 250. |
| `PopulateMosaicECs()` | Splits unitigs at transcript boundary positions. Creates per-segment ECs. Critical for handling multi-transcript unitig regions accurately. |
| `match()` | **The pseudoalignment hot loop**: iterate k-mers, find unitigs in CDBG via MPHF, skip-jump along unitig (advance by remaining unitig length when EC unchanged), intersect ECs. The skip-jump is the key optimization — it converts O(read_length) hashes into O(num_unitig_transitions) hashes. |
| D-list handling | Reads matching decoy sequences (genomic, contaminant) are classified as D-list hits and filtered. The D-list is stored as additional "transcripts" in the CDBG with special EC treatment. |

### singlify — `/mnt/home/debruinz/Singlet-AI/singlify/`

| File | Insight |
|------|---------|
| `singlify.cpp` | Fork/pipe orchestration. Parent loads gene model + SNP database, creates FIFO pipe, forks STAR as child with BAM piped to parent. |
| `pileup_engine.h` | Per-read BAM parse → CIGAR walk → exon/intron/SNP/SJ counting. Interval tree for feature assignment. Multi-mapper 1/NH weighting. **This is why we need full CIGAR** — every base position matters. |
| `gene_model.h` | GTF parsing, exon→intron computation. Builds the interval tree for fast positional queries. |
| `.1fq` codec | Block-structured parallel I/O. Per-column compression (ZSTD-3), BINNED4 quality, barcode sort, 500K-read blocks. Parallel decode achieves 8.1× speedup. |

---

*This document is a living record. Update it as new tools emerge or existing algorithms are better understood.*
