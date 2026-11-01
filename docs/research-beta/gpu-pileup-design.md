# GPU pileup in `singlet --engine gpu`: design (W11, 2026-10-28)

Branch `gpu/engine-gpu-stream`, stacked on PR #66. It includes `perf/bam-stream` (PR #63) unchanged as the CPU reference.
Spec for anything that combines evidence across reads: `docs/research-beta/order-free-rules.md` (bam-stream session, PR #60).
Gates:
- **(a)** every order-independent output is identical to the CPU engine's;
- **(b)** the outputs the spec defines (R1 SNP consensus, R4 rRNA hash sample, R5 strand, R6 barcode universe, R7 canonical
  order) are identical to the CPU streaming path implementing the same spec;
- every difference from `prod/campaign`'s sorted output is listed by file, size and cause.

## Why the data are per distinct sequence

STAR maps a cDNA read from its sequence alone (the exactness argument of `src/star/SeqCache.h`). So every alignment property is
a function of the read's sequence: chromosome, position, strand, CIGAR, NH, nM, MAPQ, primary flag, gene/exon/intron assignment,
junctions, SNP alleles, chrM bases, poly(A) clip, VDJ gene and rRNA k-mers.

The GPU already holds one mapping result per distinct sequence: 31.9 M for the 66.6 M reads of pbmc1k_v3, and the 0.17% of reads
with N in the R2 sequence, which STAR maps itself. What a read adds is only its barcode, its UMI and its identity:
- the qname hash, for R4;
- the read number, for grouping a read's alignments.

So the pileup becomes a join:

    per distinct sequence s (GPU, once):  alignments(s) -> features(s)
    per read r (GPU, 66.6 M):            (s(r), barcode column b(r), UMI u(r), qname hash h(r))
    outputs = reductions over reads of features(s(r)) keyed by b(r), u(r)

That is the decomposition the research GPU pileup already uses (PR #61 `bench/gpu/pileup/`: "signatures" = distinct
(sequence, alignment set)), measured identical to the CPU engine on 4 samples for its 9 outputs.

## Inputs, and where they come from on the GPU path (no BAM)

| input | source |
|---|---|
| distinct sequence id per read | the GPU dedup in `mapDistinct` (`order`, `head`): already computed, today discarded |
| alignments per distinct sequence | the M5 result blob (STAR's cache encoding: transcripts with exons, strand, nM, scores, nTr, trBest), decoded to (chr, start, strand, primary, MAPQ, CIGAR) as STAR's `alignBAM` does for single-end reads |
| reads STAR maps itself (R2 with N; host-lane failures) | STAR's own per-read pass emits records for these reads only, as an unsorted stream |
| barcode column per read | `.1fq` R1 → CR; CB = CR unless CR has N or UR has N or is a homopolymer (#63 `inline_scan_cb_from_cr`); column = index in the R6 universe |
| UMI per read | `.1fq` R1 (UR) |
| qname hash per read | the read name as STAR writes it, hashed with the engine's `hash_qname` |
| gene model, SNP panel, chrM, VDJ, rRNA k-mers | the engine's own loaded references (the research pileup's `anno.bin` layout: the interval trees as sorted arrays, so the CPU's pre-order is reproduced) |

STAR still runs its per-read pass, but only for its summary outputs: `Log.final.out`, `SJ.out.tab`, `Solo.out/Barcodes.stats`
and the unmapped reads that feed the non-host screen. These are byte-identical by construction. Its BAM output is off except for
the reads that took no GPU result. Removing the rest of that pass (cache lookups, FASTQ parsing) is a later step.

## Outputs

**Covered by the research port (identical on 4 samples):**
- exon, intron and ambiguous matrices (M6 UMI collapse);
- the junction table;
- SNP AD/DP;
- chrM base counts, total and reverse;
- chrM indel events.

**To add** (per-distinct property × per-read key; simple reductions):

| output | per distinct sequence | per read |
|---|---|---|
| `per_cell_reads`, `no_barcode`, totals | primary exists, mapped | b(r) |
| CellAlignQC (multimapped, mismatch rate) | NH, nM, aligned length of the primary | b(r) |
| strand sense / antisense (R5 confirmation) | gene strand vs read strand of the primary | b(r) valid |
| poly(A) records | terminal soft clip, A/T run, end position, gene | b(r), u(r) |
| VDJ gene usage | VDJ gene from the first/last block | b(r), u(r) |
| rRNA sample (R4) | rRNA hit of the stored sequence | h(r) % 64 == 0 |
| WL ambient profile | gene of the primary | CR in whitelist but not in the universe |
| CRISPR guides | guide match | b(r), u(r) (only when a guide reference is given) |

**Spec rules on the GPU:**
- R1: per (b, SNP, u), count REF/ALT/other reads; strict majority decides; ties go to `snp_molecules_ambiguous`.
- R4: hash sample.
- R5: protocol orientation; sense/antisense over all reads; exit code on contradiction.
- R6: the universe from pass 1.
- R7: junction rows sorted by (contig, donor, acceptor); other discovery axes by natural key.

Order-free by construction: every reduction is a sort plus segmented reduce or a commutative count. No result depends on read
order, so thread count and rerun cannot change it.

## Steps, each gated

1. **Hand-off, with STAR's own code for the alignments.** In the STAR child, right after the prefill, run STAR's
   `seqCacheRestore` + `alignBAM` once per distinct sequence: 31.9 M calls instead of 66.6 M per-read calls, on STAR's threads,
   with a placeholder name and quality. Keep each alignment's core fields: flag, chr, pos, MAPQ, CIGAR, NH, nM. They are then
   exactly STAR's by construction, with no re-implementation of `alignBAM`'s flag, MAPQ and CIGAR rules.
   **Original plan for this step:** `SeqCacheGpuPrefill` keeps the per-read distinct id and the per-distinct results (device-resident while the GPU
   is held; on the host, in shared memory for the parent). Per distinct sequence, decode its alignments into the research
   pileup's `AlnRec` layout.
   - Gate: the alignments equal those `extract.cpp` builds from the CPU engine's BAM for the same reads.
2. **Covered outputs** through the research kernels, fed from step 1 instead of the BAM.
   - Gate (a): exon, intron, ambiguous and junction counts and chrM equal to the CPU engine's.
   - Gate (b): SNP under R1 and junction order under R7 equal to the CPU streaming path, once #63 implements them.
3. **The table above.**
   - Gate (a) per output, against the CPU streaming path.
4. **Switch off** STAR's BAM output for GPU-served reads and the CPU pileup on the GPU path.
   - Gates: identity_test, the node runs, the CPU-only build unchanged.
   - Remeasure core-s per M pairs, node rate, memory and cost.

## Projected effect (from measured stages, pbmc1k_v3, core-s per M read pairs)

| | today, `--engine gpu` (sorted BAM) | + #63 stream (no sort, index, CB refresh) | + steps 1–4 |
|---|---|---|---|
| total | 16.1 (measured, job 790078) | about 14 (projected) | about 11 (projected): STAR's BAM records (1.2) and the CPU pileup (1.5) move to the GPU |

Below the 12.7 core-s per M pairs cloud break-even on a 54-core GPU node. The next lever after that is STAR's remaining per-read
pass (cache lookups, FASTQ parsing: about 3).

## Status and next decision (2026-10-28, afternoon)

- **Done:** the GPU pileup is linked into `singlet` (`src/engine_gpu/pileup_gpu.cu`). `SINGLET_GPU_PILEUP=compare` checks it
  in-process: all 9 covered outputs and the R1 counters are identical to the CPU pileup of the same run (pbmc1k_v3, job 790407).
- **First switch-over, `SINGLET_GPU_PILEUP=on` (build 68e48256):**
  - The CPU skips the SNP and chrM walks, junction extraction and the UMI store with its final collapse.
  - Their accumulators, junction rows (R7 order) and counters come from the GPU: `snp_hits`, `sj_hits`, `mt_pileup_bases`,
    the R1 molecule counters, and the molecule labels (M6 now counts spliced, unspliced, ambiguous and legacy_double).
  - The CPU keeps the per-read classification, which strand, poly(A), VDJ, the WL ambient profile and the stats counters need.
  - The gate is job 790461: every output against the CPU pileup, plus CPU per run.
- **Measured cost of the serial pileup in stream mode** (pbmc1k_v3, job 790421, one thread, 286 s):

  | step | seconds | covered by the first cut? |
  |---|---|---|
  | exon/intron classification | 68 | stays |
  | BAM parse and pipe wait | 54 | stays |
  | SNP | 26 | covered |
  | chrM | 8.5 | covered |
  | junctions | 7.6 | covered |
  | barcode, NH/UMI, secondary | 20 | stays |
  | unaccounted, including molecule labelling | 101 | partly covered |

- **Full offload, the next decision.** Moving the classification to the GPU (68 s, about 1 core-s per M pairs) means extending
  `k_annotate` to emit, per alignment:
  - strand votes over the unfiltered overlaps;
  - the strand-reject and multigene flags;
  - `classify_read`'s counters;
  - VDJ genes;
  - poly(A) sites.

  A host pass over the collected reads would then add these up per read. The reads outside the barcode set, which feed the WL
  ambient profile, would also need collecting. Removing the BAM entirely (records straight from the GPU prefill) would also take
  away the 54 s parse and STAR's BAM generation, but requires the same ports. Sized at a day or more. It goes ahead after the
  first cut's gate and measured saving.

## Gate of the first cut (job 790461, pbmc1k_v3, `--engine gpu`, measured)

- **CPU:** 1,196 core-s with `SINGLET_GPU_PILEUP=on`, against 1,230 and 1,186 for two runs with the CPU pileup, so within noise.
  The serial pileup took 316 s against 278 s. The collector's per-record cost (signature hashing into a 27.6 M-entry map, vector
  allocations) in the same thread cancels the skipped work.
- **Output differences against the CPU pileup:**
  - `sj_counts.1pz` and `splice_psi.1pz`: the feature count. The CPU grows it in steps of 1000; fixed in the next build.
  - `saturation_metrics.tsv` is missing and `cell_qc_metrics.tsv` differs: both read the CPU molecule store, which "on" empties.
  - `summary.json` `memory_tier`: a peak-RSS bucket, not data.
- **Conclusion:** while the CPU loop still parses the BAM and classifies every read, a GPU pileup cannot reduce CPU. The next
  step is the full offload with no BAM: records come from the GPU prefill, and every pileup output is computed from them,
  including the per-molecule list behind saturation and cell QC. "On" stays an off-by-default experiment, and compare mode is
  the per-output identity harness.

## Curve toward the break-even (pbmc1k_v3, `--engine gpu`, campaign configuration; measured)

| build | gate (a), every data output | CPU pileup: core-s (per M pairs) | `SINGLET_GPU_PILEUP=on`: core-s (per M pairs) | wall, CPU pileup / on | job |
|---|---|---|---|---|---|
| first cut (SNP, chrM, junctions, molecules on the GPU) | 3 outputs differed | 1,230 / 1,186 | 1,196 | 430 / 438 s | 790461 |
| + molecule list for saturation and cell QC, cheaper collector | **pass** | 1,214 (18.2) | 1,172 (17.6) | 420 / 407 s | 790520 |
| + classification, strand, poly(A) and VDJ from the GPU | **pass** | 1,230 (18.5) | **1,175 (17.6)** | 456 / **376 s** | 790552 |
| + hand-off: the prefill's alignments go to the pileup; STAR writes no BAM records for those reads (60.3 M of 60.4 M barcoded) | **pass** | 1,273 (19.1) | 1,180 (17.7) | 464 / 397 s | 790789 |
| + STAR's mapping threads write each hand-off read's CR, UR and quality counters; no FASTQ re-read | **pass** | 1,228 (18.4) | **1,102 (16.5)** | 426 / **349 s** | 790820 (node g052) |
| + STAR replays a per-sequence summary (stats, junction records, Solo Gene annotation) instead of restoring each hand-off read's alignments | **pass** | 1,321 (19.8) | 1,065 (16.0) | 473 / 379 s | 790990 (node g051, slower) |

| + only unmapped sequences go into STAR's cache (mapped ones replay summaries) | **pass** | 1,314 | 972 | 475 / 377 s | 791012 (node g051) |

Same node (job 791013, g052, one run each, per M pairs = core-s / 66.6 M):

| build | core-s | per M pairs | wall |
|---|---|---|---|
| CPU pileup | 1,227 | 18.4 | 424 s |
| + per-read entries (ad24936d) | 1,128 | 16.9 | 350 s |
| + summary replay (c389cf56) | 996 | 15.0 | 355 s |
| + cache trimmed (0b560145) | **929** | **13.9** | 356 s |

The break-even of 12.7 per M pairs is 846 core-s, 83 below that build.

The prefill converted R2's sequence lines one `push_back` per base, about 170 core-s for 66.6 M reads (STAR log with timestamps,
job 791104). A table into a buffer reserved up front (cfd66207) passes gate (a) (job 791113). Same node (g051, job 791114), runs
alternated:

| build | core-s | per M pairs | wall |
|---|---|---|---|
| CPU pileup (cfd66207; it uses the same prefill) | 1,020 | 15.3 | 409 s |
| `on`, 0b560145 | 893, 913 | 13.4, 13.7 | 336, 338 s |
| `on`, cfd66207 | **765, 760** | **11.5, 11.4** | **328, 327 s** |

That is below the 12.7 break-even on pbmc1k_v3.

**Six more samples, gate (a), cfd66207** (jobs 791172-791175; GSM7656149 on 4783428f, job 791331). Every sample passes. The
differences are the known order, timing and memory-tier ones, plus `nonhost_em_abundance.tsv` on two samples (same order-dependent
non-host EM as the iteration count).

| sample | protocol | CPU pileup core-s | `on` core-s | wall CPU / on | hand-off |
|---|---|---|---|---|---|
| GSM6996272 | 10x 3' | 102 | 100 | 41 / 40 s | yes |
| GSM6996274 | 10x 3' | 136 | 125 | 54 / 50 s | yes |
| GSM4006843 | 10x 3' | 1,093 | 763 | 474 / 340 s | yes |
| GSM7036403 | 10x 3' | 1,727 | 1,200 | 696 / 540 s | yes |
| GSM6470208 | 10x 5' v2 | 4,331 | 4,114 | 575 / 461 s | no: the prefill refuses STAR's TSO clipping |
| GSM7656149 | 10x 3' | 4,355 | 3,196 | 1,854 / 1,418 s | yes; GPU arena retried in unified memory (peak 104 GB) |

**One H100 with half a node** (27 cores, 350 GB, short-gpu; Zach: single GPU for now). Samples run concurrently; bin32 (4783428f).
Core-s is the job cgroup's.

| run | samples (read pairs) | mode | makespan | core-s | per M pairs | read pairs/s | peak memory | mean GPU use | job |
|---|---|---|---|---|---|---|---|---|---|
| 3 samples, 9 threads each | pbmc1k_v3, GSM7036403, GSM4006843 (244.9 M) | `on` | 653 s | 2,724 | 11.1 | 0.375 M | 158 GB | 8% | 791375 |
| | | CPU pileup | 790 s | 3,883 | 15.9 | 0.310 M | 176 GB | 6% | 791375 |
| 6 samples, 5 threads each | the 3 above plus GSM7656149, GSM6996274, GSM6470208 (634 M) | `on` | 1,726 s | 10,526 | 16.6 | 0.367 M | 286 GB | 10% | 791376 |
| | | CPU pileup | 1,968 s | 12,741 | 20.1 | 0.322 M | 321 GB | 7% | 791377 |

- Per-sample peak RSS: 43-63 GB for the 66-112 M pair samples; GSM7656149 (263 M) 116 GB; GSM6470208 26 GB.
- The 6-sample set includes GSM6470208 (10x 5', no GPU path, 122 M pairs, about 34 core-s per M pairs on its own), which raises
  the per-M-pairs figure.
- The GPU is busy 7-10% of the time, and the cores average 4-6 of 27. Makespan is set by the largest sample (GSM7656149: 29 min
  `on`, 33 min CPU pileup). Packing more small samples per GPU, or giving the largest sample more threads, would raise throughput.
- The node script's byte-level identity check also flags files with timings (`Log.final.out`, JSON `seconds` fields such as
  `ambient_contamination.json`, `read_stats.tsv`). The gates' `out_equiv` treats those as timing-only; a 9-thread single-sample
  check (job 791493: CPU, `on`, CPU again) shows only the known differences.

GSM7656149 first failed (job 791172): its 223 M barcoded reads needed about 100 GB of GPU memory against 94 GB. The arena now throws,
and `pileupRun` retries once in unified memory (4783428f). That is a guard; the design is up-front sizing with cell partitioning
([align-and-count-design.md](align-and-count-design.md) section 4.4). Still to do before the flag can default on: the multi-sample gates (identity test,
3- and 6-sample node runs, CPU path unchanged), node rate, memory and cost, and the order-dependent differences against
prod/campaign.

The summary step first failed (job 790899: 50 files differed, 7 M alignments lost). The prefill never set STAR's per-read mismatch
limit (`outFilterMismatchNmaxTotal`), so its `mappedFilter` read an uninitialized value. That had been harmless by luck since the
hand-off began. It is set now, and STAR skips BAM only for reads whose record has alignments, so any disagreement between STAR and
the hand-off reaches the pileup's check instead of losing reads.

How the hand-off was checked: with `SINGLET_GPU_PILEUP=compare`, the pass rebuilds each hand-off read's barcode and UMI from R1 and
compares them with the stream's. All 60,338,992 reads matched (job 790769). Two bugs found on the way:
- STAR's `iReadAll` is 1-based, while the hand-off's per-read list starts at 0, so STAR skipped the wrong reads (job 790699: 48
  files differed). The pass now fails if any hand-off read also comes through the stream.
- The GPU's junction table spans every hand-off record, including reads without a cell barcode. Only junctions with a count are
  kept now, as the CPU does (job 790778: 13K extra zero-count junctions).

Skipping BAM records saves no CPU yet: the serial pileup drops from 316 to 269 s, but the prefill's record building and the FASTQ
pass offset that.

Remaining differences in every pass: the non-host EM iteration count (D4b), the record order of `star_Unmapped.out.mate*` and
`nonhost_reads.fq`, timings, and `summary.json` `memory_tier`, a peak-RSS bucket that the collector's memory pushes past 64 GB.
The CPU-pileup run varies by about ±3% across nodes.

**Projected:** the serial pileup thread is down to 251 s (BAM read, tag scans, collector, cheap per-record outputs). Removing all
of it brings the GPU path to about 13.8 core-s per M pairs. The 12.7 break-even also needs STAR's per-read pass reduced. Next
steps:
1. Compact records instead of the BAM stream, with the cheap per-record outputs moved onto them.
2. STAR's BAM-record building off the per-read pass.

## R9 / R7 and phase 1a gates (2026-10-29, measured)

R9 (directional UMI collapse for junction, V(D)J and SNP molecules) and R7 (a multimapper takes the smallest secondary exon) are in the GPU pileup:
commit 493586b4, which merges perf/bam-stream 253c2c65, so the CPU side is W3b's code. Phase 1a (read clipping on the GPU, so 10x 5' takes the
prefill and the hand-off) is commit b50f7501. Gate (a) is `--engine gpu` with the CPU pileup against the GPU pileup ("on"), 18 threads, one
H100. Every sample shows only the 7 known differences (nonhost / Unmapped read order, timing fields, memory_tier).

| sample | binary | CPU pileup core-s | on core-s | note |
|---|---|---|---|---|
| GSM6996272 | 493586b4 | 109 | 102 | |
| GSM6996274 | 493586b4 / b50f7501 | 143 / 132 | 131 / 124 | |
| GSM7036403 | 493586b4 / b50f7501 | 1,945 / 1,827 | 1,292 / 1,195 | |
| GSM7656149 | 493586b4 | 4,800 | 3,410 | peak 106 GB, unified-memory retry; R1 molecules 23.38 M -> 23.00 M under R9 |
| GSM6470208 (10x 5' v2) | 493586b4 | 4,510 | 4,361 | no prefill (clipping) |
| GSM6470208 (10x 5' v2) | **b50f7501** | **2,943** | **2,274** | prefill + hand-off: 108.8 M of 121.8 M reads skip BAM; rRNA sample identical (4,680 / 1,711,402) |
| GSM4006843 | b50f7501 | 1,359 | 857 | re-staged from SRA (SRR9900123, 66.7 M reads) into `data/w11/`; 95.3% of reads skip BAM; job 792023 |

Jobs 791872 (R9), 791995 and 792023 (1a). GSM4006843's earlier input (`data/bam-stream/`) was deleted, so it was re-staged from SRA (job 792022).

## Phase 1b and the host-cost changes: gates (2026-10-29, measured)

Binaries: 1b at 475295a6 (job 792062); 1b fixes + host-cost F1-F3 + N1/N2 at 4fe12e9f (bin38, job 792223). One H100, 18 threads.

**(c) Partition invariance** (GSM7036403, K forced to 1, 2, 7, job 792223): the digest of every GPU pileup output (matrices, junction table,
indels as a set, per-cell reads and molecules, counters) is identical, `93a9547e...`, for all three. Every output file is identical too; the
only differing files also differ between two K = 1 runs (nonhost and Unmapped read order, timing fields). GPU time 302 / 301 / 333 ms,
of which partition select (sizing + compaction) 100 / 96 / 104 ms; host fetch of the results ~3.1 s at every K. K = 1 now skips the
compaction (1fd5b98d), which removes that 0.1 s.

**(f) Largest sample:** GSM7656149, automatic K = 2 (estimate 46.1 / 46.2 GB, used 44.4 / 44.5 GB), peak 72 GB, no unified-memory retry;
gate (a) passes. With the earlier coefficients it chose K = 3 (job 792062). The K = 7 estimate on GSM7036403 (4.9 GB) was below use
(5.9 GB) until the partition scratch (24 B per read) was budgeted (1fd5b98d).

**(a) with the host-cost changes** (bin38): GSM7036403, GSM7656149, GSM6470208, GSM4006843 pass (only the known differences).
`SINGLET_HANDOFF_LEAN=verify`: 0 of 46,661,077 (GSM6470208) and 0 of 32,743,513 (pbmc1k_v3) lean alignments differ from alignBAM's, and the
verify run's outputs equal the default's. `SINGLET_HANDOFF_CLIPCHECK=1` on GSM6470208 (28.3 M clipped reads): no read where the GPU's
clip lengths differ from STAR's CPU clip.

| sample | CPU pileup core-s | GPU pileup core-s | per M reads (GPU pileup) | before host-cost (GPU pileup) |
|---|---|---|---|---|
| GSM6470208 (5′) | 2,365 | **1,302** | **10.7** | 2,274 (18.7) |
| GSM7036403 | 1,945 | 1,196 | 10.7 | 1,292 |
| GSM7656149 | 4,315 | 2,898 | 11.0 | 3,145 |
| GSM4006843 | 1,202 | 744 | 11.2 | 802 |
| pbmc1k_v3 | — | 697 | 10.5 | 769 |

Stage profile of GSM6470208 on a STAR_PROFILE build (job 792224, `bench/host_cost/stage_profile_handoff.py`, intervals inside STAR from
STAR's log times): 1,480 core-s (12.15 per M reads): decode 1.08, STAR start 0.07, GPU prefill 2.48, STAR per-read pass 5.75, hand-off
pass 0.30, GPU pileup + apply + host pass 0.35, non-host screen and writers 1.08, export 0.90 per M reads.
