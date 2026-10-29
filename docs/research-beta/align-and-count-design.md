# Phase 1: one GPU align-and-count engine for every protocol (design, 2026-10-28)

Status: design for review. Design and gates come before prototype code (Zach, 2026-10-28, via the orchestrator). Nothing here is
built yet beyond the W11 path in [gpu-pileup-design.md](gpu-pileup-design.md), which covers the 10x 3' case only.

## 1. Requirement

Zach, 2026-10-28:
- **Generality is a design requirement.** The GPU align-and-count engine must serve every protocol singlet supports, not only
  10x 3' droplets:
  - droplet: 3' and 5', single- and paired-end;
  - plate-based: per-well or per-cell FASTQs, no barcode in the read, often full-length paired-end;
  - spatial: spot barcodes and coordinates;
  - ATAC and Multiome: unspliced genome alignment, fragments and peaks rather than genes;
  - feature barcoding, CRISPR and VDJ, where singlet already supports them.
- **Single GPU for now.** One H100 NVL must handle the largest samples. GPU tables are sized up front or partitioned.
- Read layout comes from the protocol table. Alignment mode is a parameter of one engine. Counting is pluggable reducers keyed by
  (cell, molecule, feature).

## 2. What exists today (W11, measured)

Today `--engine gpu` is STAR on the CPU, with three GPU additions:
1. **GPU prefill.** It maps each distinct cDNA sequence once on the GPU (seed, window, stitch) and fills STAR's sequence cache.
   It runs only for single-end reads without clipping.
2. **Hand-off.** STAR's own records go to the pileup once per distinct sequence. STAR writes no BAM for those reads, and replays a
   per-sequence summary for its own outputs (stats, `SJ.out.tab`, Solo).
3. **GPU pileup.** Molecules, matrices, junctions, SNP and chrM tallies are counted by sort-based reductions on the GPU, under the
   order-free rules (R1-R7 in bench/bam-stream's `order-free-rules.md`).

Measured on pbmc1k_v3 (10x 3' v3, 66.6 M read pairs), same node, gate (a) passing:
- 11.4-11.5 core-s per M read pairs, against 15.3 for the CPU pileup;
- break-even is 12.7.

Six more samples pass gate (a). One of them, GSM7656149 (223 M barcoded reads), needed about 100 GB of GPU memory against 94 GB on
the device. The unified-memory retry in 4783428f is a safety net, not the design (see section 4).

## 3. Which protocols fall back to STAR on the CPU today

Source: `known_protocols()` in `include/singlet/fq/protocol.h:66-190` and the guards in `src/star/SeqCacheGpuPrefill.cpp`,
`src/star/SeqCache.cpp:72-79`, `src/pipeline/singlet.cpp:5004-5039` and `src/star/ReadAlign_seqCache.cpp:196-200`.
- **Prefill:** whether the GPU maps the reads.
- **Hand-off and pileup:** whether STAR's BAM stream is replaced. Every "no" falls back to CPU STAR plus the CPU pileup.

| Protocol | Read layout | GPU prefill | Hand-off, summary, GPU pileup | What blocks it |
|---|---|---|---|---|
| 10x 3' v1-v4, CITE-seq GEX, Quartz-seq2 (whitelist installed) | CB+UMI in R1, cDNA single-end | yes | yes | — |
| **10x 5' v2/v3/v4, 10x ARC GEX** | CB+UMI in R1, cDNA single-end | **no** | no hand-off (the pileup reads the stream) | TSO trimmed by STAR's `--clip3pAdapterSeq`; the prefill refuses any clipping (`noClipping`, SeqCacheGpuPrefill.cpp:101-109, 215) |
| Drop-seq, Seq-Well, CEL-seq2, MARS-seq2, ddSEQ, Visium | CB+UMI in R1, no whitelist (Visium: spot list not bundled) | yes | no | `CB_samTagOut` and the sorted BAM path (singlet.cpp:5014) |
| BD Rhapsody, SPLiT/Parse, inDrop, Microwell-seq, SureCell | multi-segment barcodes | yes | no | `CB_UMI_Complex`, sorted path (5011) |
| sci-RNA-seq3, DNBelab C4 | CB+UMI, Simple or Complex by metadata | yes | only when Simple | same as the two rows above |
| STRT-seq | CB, no UMI, TSO | no | no | clipping; no UMI |
| **ATAC, Multiome ATAC** | barcode in QNAME, paired-end, unspliced (`--alignIntronMax 1 --alignMatesGapMax 2000`) | **no** | no | the sequence cache is off for paired-end (SeqCache.cpp:72); sorted path |
| **Smart-seq2 / plate** | cell = file, no UMI, paired-end, full-length | **no** | no | paired-end; sorted path (5010) |
| Bulk RNA, single- or paired-end | no cell | single-end only | no | no soloType; paired-end |
| Any protocol from `--sra` or gzipped `--reads` | — | no | no | `readFilesCommand` (SeqCacheGpuPrefill.cpp:216); stream needs `.1fq` (5007) |
| Any protocol with `--guide-ref` | per row | per row | no hand-off | singlet.cpp:5032 |
| ADT / feature barcodes only, VDJ | not aligned by STAR | — | — | Python modules (`singlet.vdj`, `singlet.crispr`); VDJ is a run-screen flag |

Correction: an earlier W11 note called 10x 5' v2 "aligned paired-end". It is single-end (STAR counts `readNends - 1` mates under
any soloType, ParametersSolo.cpp:130); the clipping guard is what blocks it. GSM6470208 (10x 5' v2) passed gate (a) on the stream
path without the hand-off.

## 4. Architecture

One engine, three layers. Every layer is configured from the protocol table and never by protocol-specific code paths.

### 4.1 Read layout: from the protocol table to one normalized read record

Each input read becomes one fixed record:
`(read_id, cell_key, umi_key, mates[1..2] (seq, qual, clip spans), feature_read?)`.

**Where `cell_key` comes from:**
- **In-read barcode with a whitelist** (10x and the like): 1MM_multi correction as STARsolo does it, from the existing
  `CB_UMI_Simple` / `CB_UMI_Complex` layouts.
- **In-read barcode without a whitelist** (Drop-seq family): the raw barcode, then knee-based or bead-error correction.
- **One file per cell or well** (plate): the file index.
- **Spot barcode** (spatial): the barcode and its `(x, y)` from the spot list, carried as feature metadata.
- **Combinatorial segments** (SPLiT, sci): the concatenated corrected segments.

**`umi_key`:** the UMI from the read when there is one. Otherwise it is empty, and the molecule key comes from the alignment
(section 4.3).

**Clipping is part of the record.** Adapter and TSO trimming (the 5' case) and 5' prefixes become clip spans computed on the
GPU before seeding. That removes the guard that blocks 10x 5' today. The cache and dedup key is the clipped sequence, as STAR's own
cache key is.

**Feature reads** (ADT, CRISPR guides, hashtags) skip alignment and go straight to the feature-barcode reducer.

### 4.2 Alignment: one aligner, mode as a parameter

- **Mode:** `{spliced | unspliced}` × `{single | paired}`, plus strand handling.
  - Spliced uses sjdb insertion and the intron limits.
  - Unspliced is ATAC: `alignIntronMax 1`, mate gap ≤ 2000, no sjdb.
- **Paired-end:**
  - Seeds come from both mates. Windows are joined across mates under the mate-gap limit, as STAR's paired windows are.
  - The dedup key is the ordered mate pair.
  - Full-length plate reads deduplicate poorly, so their cost is per read. That is still on the GPU; dedup is an optimisation,
    not a requirement.
- **Exactness contract:** unchanged from W11. For the order-independent outputs, the GPU result equals STAR's for the same
  parameters. Where the GPU cannot reach a decision, the read goes to a CPU STAR lane (today's "host lane", 86,511 reads on
  pbmc1k_v3).

### 4.3 Counting: pluggable reducers keyed by (cell, molecule, feature)

Every reducer is the same pipeline, sort-based and order-free (no atomics leak into results):
1. Emit tuples `(cell, molecule, feature, payload)` from each aligned read.
2. Radix sort.
3. Collapse molecules (exact or directional UMI rules), resolve per molecule, then reduce per (cell, feature).

The W11 GPU pileup already has this shape for genes, junctions, SNP alleles and chrM bases. Phase 1 makes it an interface:

| Reducer | Feature | Molecule key | Notes |
|---|---|---|---|
| gene | gene / exon / intron / ambiguous (spliced, unspliced, ambiguous) | UMI, or for no-UMI protocols the read | exists (W11) |
| junction | (contig, donor, acceptor) | UMI, directional collapse (R9) | exists; exact-UMI today, R9 change below |
| SNP allele | panel site | UMI, directional collapse (R9), R1 consensus over the merged molecule | exists; exact-UMI today, R9 change below |
| chrM base / indel | position | UMI | exists |
| fragment (ATAC) | (contig, start, end) | the fragment itself: dedup by coordinates per cell | new |
| peak (ATAC) | peak id | fragment | new; two passes: call peaks from all fragments, then count cell × peak |
| spot (spatial) | gene, as in the gene reducer | UMI | cell = spot; coordinates as metadata. Only the cell-key source is new |
| feature barcode (ADT, CRISPR, hashtag) | feature id | UMI | new on the GPU, no alignment |
| plate no-UMI | gene | none (read counts) or (start, strand) dedup by protocol option | new option of the gene reducer |

**Cross-cell tables** need a canonical merge:
- junction ids;
- the peak set;
- the saturation curve.

Each is built by a sort over all cells, so results do not depend on how the reads were partitioned.

**One UMI rule for every UMI reducer (R9, from W3b).**

W3b found that the unsorted stream collapsed junction, V(D)J and SNP molecules on the raw UR, while the sorted path had STAR's
corrected UB. On 5′ the stream overcounted junctions by 3.4% and V(D)J by 6.8%. Rule R9 (`order-free-rules.md`, perf/bam-stream
253c2c65; reference code `UmiFeatureCollapse`, `umi3_directional_roots` and `SnpConsensus::finalize` in `umi_dedup.h`) gives these
reducers the gene reducer's collapse:
- the molecule key is the raw UMI, packed 3 bits per base (A C G T N other), so N is a base of its own; UB is never read;
- per (cell, feature), two UMIs of the same length are joined when exactly one base differs and `max(reads) ≥ 2·min(reads) − 1`;
- a molecule is a connected component; reads without a UMI are one molecule each; UMIs longer than 21 bases match exactly;
- this runs at finalize, after all reads, so it is order-free;
- the CPU's union-find visits UMIs sorted by reads (descending), then packed UMI. That order picks only the root, and no output
  depends on the root;
- for SNP, a UMI's weight is the sum of its REF / ALT / OTHER read counts, each clamped at 255. The R1 consensus (strict maximum,
  a tie is ambiguous) runs over the summed counts of the merged molecule, not per raw UMI.

Every reducer with a UMI molecule key in the table above follows the same rule, keyed by (cell, feature), where the feature is
the gene, junction id, SNP site or feature barcode. Only the per-component payload merge differs:
- gene: flags OR, smallest exon / intron (2-bit packing, N as T, as `DirectionalUmiStore` has always done);
- junction and V(D)J: none (one count per component);
- SNP: sum of the REF / ALT / OTHER read counts, then the R1 decision;
- feature barcode: none.

chrM bases and indels stay outside R9 until the rules say otherwise.

**GPU pileup change for R9 and R7** (commit 493586b4 on gpu/engine-gpu-stream; it merges perf/bam-stream 253c2c65):

| Output | Before (exact UMI) | Now |
|---|---|---|
| Junctions | `pileup.cu` step 5: first of each exact (cell, junction, UMI) run | reads per exact (cell, junction, UMI) run, then `k_r9`: one thread per (cell, junction) group runs the CPU's union-find sequentially (so it is order-free and exact); molecules = component roots |
| SNP DP / AD | step 6: R1 vote per exact (cell, SNP, UMI) | class counts per exact UMI clamped at 255, weight = their sum, `k_r9` per (cell, SNP), R1 vote over each component's summed counts |
| V(D)J gene usage | CPU (`gpu_handoff_pass`), first read per exact UMI | CPU, through the stream's R9 store (`vdj_umi_.add`, collapsed in the shared finalize) |
| Multimapper exon (R7) | `k_resolve`: the first secondary's exon | the primary's assignment, else the smallest secondary exon index |

The UMIs in steps 5 and 6 were already base 5 with N distinct, which matches R9's packing for 12-base UMIs. A thread per group is
cheap because almost every group has one to a few UMIs. A pathological group of n UMIs costs n²/2 tests on one thread. If a
sample shows that, large groups move to a block per group with the same result.

The `--snp-first` gate mode (the sorted path's first-read rule) stays, for comparison with prod/campaign only.

Gate (a) against the CPU stream is rerun on the six multi-sample gate samples once the waiting Alpha GPU job has finished (build 791871, gate
array 791872).

This lands in the W11 GPU pileup before phase 1a. Phase 1 reuses it unchanged.

### 4.4 One GPU for the largest samples: size up front, partition by cell

Measured peak device memory for the GPU pileup:
- 25 GB at 60 M barcoded reads (pbmc1k_v3);
- 31 GB at 105 M (GSM7036403);
- about 100 GB at 223 M (GSM7656149, 1.05 M signatures).

**Size up front.** Before any allocation, estimate bytes from the input counts:
- reads, records and alignments;
- the tuples each reducer emits, an upper bound from read counts × the per-read emit bound;
- the CUB sort buffers (2× keys and values).

**Partition by cell.** If the estimate exceeds the device budget (free memory minus a reserve), partition by `hash(cell) mod K`
with the smallest K that fits, then run the reducers once per partition:
- Every reducer's key starts with the cell, so partitions are independent except for the cross-cell tables in section 4.3.
- Those tables are merged with a canonical sort after the partitions finish, and the per-partition outputs are remapped to the
  merged ids.
- Results are identical for every K; that is a gate (section 5).

The unified-memory retry stays only as a guard against a wrong estimate, and must log when it fires.

The aligner side is already bounded: the prefill maps distinct sequences in batches.

## 5. Gates (before any protocol class is switched on)

For each protocol class, on real public samples with real ground truth (the benchmark rule):
- **(a) Identity with the CPU engine** for every order-independent output, under the same parameters. This is the W11 gate
  (`gpu_pileup_gate.sbatch`, `out_equiv.py`).
- **(b) Order-dependent outputs** follow the order-free rules and equal the CPU streaming path. Report how each differs from
  prod/campaign.
- **(c) Partition invariance:** the same sample with forced K = 1, 2 and 7 gives byte-identical outputs.
- **(d) Capability contracts** in `bench/public/capabilities/` for the class: atac, multiome, spatial, vdj, crispr_guides and the
  GEX ones.
- **(e) Cost:** core-s per M read pairs and GPU-s per sample on one H100 against the CPU engine on the same node.
  - It must not be worse than the CPU path.
  - Break-even, from the campaign's cost model, is 12.7 core-s per M pairs for GEX.
- **(f) Largest sample per class** runs on one H100 without the unified-memory retry firing.

**Samples per class** (proposed, from what is on Clipper):
- 10x 3': pbmc1k_v3, GSM4006843, GSM7036403, GSM7656149.
- 10x 5': GSM6470208 and a 5' v3 run.
- ATAC and Multiome: atac_pbmc_5k_nextgem, Multiome 10k (`$WORK/data/atac-spatial/`).
- Visium: the two slides in the atac-spatial benchmark.
- Plate: a Smart-seq2 study from the atlas.
- Drop-seq family and complex barcodes: one study each.
- Feature barcodes: the ADT and CRISPR cohort rows.

## 6. Phases (each ends at its gate; a phase that fails its kill criterion stops)

| Phase | Scope | Kill criterion |
|---|---|---|
| 1a | Clip spans on the GPU, so 10x 5' and ARC GEX get the prefill, hand-off and summary | gate (a) fails on 5' after clipping is matched to STAR's `clip3pAdapterSeq` semantics |
| 1b | Reducer interface over the W11 kernels; cell partitioning with up-front sizing | (c) or (f) fails, or partitioning costs > 10% of GPU time |
| 1c | Stream and hand-off for whitelist-free and complex barcodes (`CB_samTagOut`, `CB_UMI_Complex`) | gate (a) on the Drop-seq and complex samples |
| 1d | Paired-end alignment on the GPU (mate windows); plate layout (cell = file); no-UMI gene reducer | ≥ 99.9% identical primary alignments on unique reads, or (a) fails |
| 1e | Unspliced mode; fragment and peak reducers (ATAC, Multiome) | (a) against the CPU fragments, and (d) atac/multiome |
| 1f | Spot cell keys (Visium); feature-barcode reducer (ADT, CRISPR, hashtags) | (a) and (d) spatial / crispr_guides |

**Order.** 1a and 1b first:
- 1a moves the largest share of the atlas off the CPU path (10x 5').
- 1b is what makes "one GPU for the largest samples" a property rather than luck.

1d and 1e carry the most aligner risk, so their microbenchmarks (paired windows, unspliced seeding) run before their phase starts.

## 7. Not in scope here

- Multi-GPU (Zach: single GPU for now).
- Replacing STAR's barcode correction rules: the engine reproduces them.
- Public outputs of reads, BAMs or genotypes: unchanged policy.
