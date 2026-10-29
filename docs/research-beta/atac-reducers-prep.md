# Phase 1e preparation: ATAC fragment and peak reducers

Status: preparation only, 2026-10-29. Nothing here is switched on, compiled on a GPU, or run on a cluster. Phase 1e is not approved to
switch on. Branch `claude/gpu-atac-reducers-270cf8`, no PR.

What exists:

| piece | where | state |
|---|---|---|
| CPU semantics and order-dependence audit | this file, sections 1-2 | done |
| Proposed order-free rules A1-A7 | section 3 | proposal, for Zach |
| Reference reducers (numba/numpy) and CPU oracle ports | `bench/atac_reducers/atac_ref.py`, fixture generator `sim.py` | done |
| Tests | `tests/python/test_atac_reducers_ref.py` | 14 pass (section 4) |
| CUDA drafts | `src/engine_gpu/gpu_pileup/pileup/atac_reducers.cuh` | **not compiled**, not included anywhere |
| What W11 builds and gates | section 6 | list |

`pileup.cu` and `engine_gpu.cu` are untouched. The drafts are written against the 1b reducer interface (commit 475295a6): `Arena`,
`cub_run`, `sort_hilo`, `k_runflag`, `k_unique_hilo`, `scan_u32`, `total_of`, `SumU32`, `cell_part`.

## 0. Which CPU code

**The CPU ATAC path that matters is on the unmerged PR #48 branch**, `origin/claude/atac-spatial-benchmark-2ba7df` (tip `afc8fe35`,
benchmark binary `singlet-atacsp` = 88a41ff on prod/modality-fixes 4f110f2). This worktree's `main`-derived tree still has the older
code (`include/singlet/pileup/atac_fragment.h`, `atac_peaks.h`, `atac_bin_counter.h`, `fragments_writer.h`, `multiome_router.h`, and the ATAC
block of `src/pipeline/singlet.cpp` around line 7498). The older code is superseded and is **not** the target of gate (a):

| older code on this tree | defect (all fixed or replaced on PR #48) |
|---|---|
| `atac_fragment.h:108,123` keeps READ1 only and takes `b->core.pos` as the fragment start | when READ1 is the right mate (about half of all pairs) `pos` is the right mate's start, so the fragment is shifted by up to its length |
| `atac_fragment.h:144,162` | fragments sorted by (chrom, start) only; ties come out in `std::sort`'s arbitrary order, so `fragments.tsv.gz` is not byte-reproducible |
| `singlet.cpp:7573` | duplicate count column is always `1` |
| `atac_peaks.h:86` | peaks from 500 bp bin counts of all overlapped bins; lexicographic chromosome order for peak names; normal approximation above lambda 1000 (`:46`) |
| (none) | no cell x peak matrix at all, only the 500 bp bin matrix |

Everything below describes PR #48. Line references are `origin/claude/atac-spatial-benchmark-2ba7df:include/singlet/pileup/<file>:<line>`.
The GPU gate (a) therefore needs a CPU build from that branch (or its merge into the campaign build, PR #48 open problem 3).

## 1. Semantics of the CPU ATAC path (PR #48)

### 1.1 Fragments (`atac_fragment.h`)

| step | exact rule | ref |
|---|---|---|
| record filter | proper pair, not unmapped, mate not unmapped, not secondary/supplementary | `:112-118` |
| MAPQ | `>= 30` (STAR: 255 unique; 3, 1, 0 multimappers), so multimappers are dropped | `:121` |
| one record per pair | only the leftmost mate, `TLEN > 0` (either mate may be READ1); `mtid == tid` | `:128-129` |
| insert size | `10 <= TLEN <= 2000` | `:133`, `:264-265` |
| geometry and Tn5 shift | `start = pos + 4`, `end = pos + TLEN - 5`, 0-based half-open; drop if `start >= end` | `:136-142` |
| barcode | raw `BC:<raw>_` QNAME prefix, corrected to a whitelist index; unresolved or all-N dropped | `:223-262`, `atac_barcodes.h` |
| dedup | key (contig, start, end, barcode); duplicates add to `read_pairs`; **the dedup table is cleared when (contig, start) changes** | `:150-166` |
| output order | sorted by (contig, start, end, barcode): a total order | `:194-200` |
| output | `fragments.tsv.gz`: chrom, start, end, barcode (GEX barcode for Multiome), `read_pairs`; bgzip + tabix | `atac_pipeline.h:189-205` |

### 1.2 Barcode layout, correction, Multiome translation (`atac_barcodes.h`, `atac_pipeline.h`)

| step | rule | ref |
|---|---|---|
| layout and whitelist choice | the first 200,000 fragment records **in BAM order** are observed; the (offset, strand) window and the whitelist candidate with the highest exact hit rate win, ties to the first | `atac_pipeline.h:156`, `atac_barcodes.h:77`, `:138` |
| prior | exact whitelist hits counted over all records | `atac_barcodes.h` `count()` |
| correction | exact match, else the unique Hamming-1 neighbour with posterior >= 0.975 under the count prior (+0.5 pseudocount); a single N is tried as every base | `atac_barcodes.h:208` |
| Multiome | ATAC barcode index -> GEX barcode through the ARC pair table (`output_barcode`); the ATAC matrices are labelled by the GEX barcode | `atac_barcodes.h:129` |
| column set | barcodes with `>= 100` fragments or called as cells, sorted by output barcode string | `atac_pipeline.h:276` |

### 1.3 Peaks (`atac_peak_caller.h`)

| step | rule | ref |
|---|---|---|
| input | both Tn5 cut sites (`start`, `end - 1`) of every unique **non-mito** fragment of **every** resolved barcode (cells and empty droplets), one count per unique fragment, not weighted by `read_pairs` | `atac_pipeline.h:212-218` |
| background | `lambda_bg = total_insertions * 200 / (0.87 * sum of non-skipped contig lengths)` | `:83` |
| grid | positions `x = g*10 + 5`, pileup = insertions in `[x-100, x+100)`; local rate = max(lambda_bg, insertions in `[x-5000, x+5000)` * 200/10000); zero or `k <= lambda` is a "zero bin" | `:95-110` (`:109` local rate) |
| score | `s = -log10 P(X >= k)`, exact upper tail, double (`lgamma`, forward series to 1e-15 relative) | `:50-65`, `:111` |
| FDR | histogram of `int(s*100)` over **all** grid positions of all non-skipped contigs (zero bins in bin 0); Benjamini-Hochberg score per bin, running max, stored as float; `s_thresh` = first bin with q >= -log10 0.05, at least 1 | `:112-140` |
| regions | positions with `float(s) >= s_thresh`, merged when `gs - previous_ge <= 30`, kept if length `>= 200`; summit = first (leftmost) maximal pileup (`>` strict); p and q = max over the region | `:150-170` |
| order | contig index order, then start; names `peak_<i+1>` | `:165-171` (flush order) |

### 1.4 Cell x peak counts and per-cell metrics (`atac_metrics.h`)

| item | rule | ref |
|---|---|---|
| cell x peak matrix | **cut sites** per (barcode, peak) (Cell Ranger convention): each of `start`, `end - 1` inside a peak adds 1, so a fragment with both ends in one peak counts 2; chrM skipped; peaks are disjoint | `:251-252` |
| `peak_region_fragments` | fragment with a cut site in a peak, or overlapping one | `:249` |
| `peak_region_cutsites`, `fragments`, `mito_fragments`, `read_pairs` | integer sums per barcode | `:235-250` |
| matrix assembly | triplets sorted by (column, row), runs counted | `atac_metrics.h:129-150` |
| cell call | `call_cells_mixture` on per-barcode vectors in barcode-index order (joint with GEX UMIs when `--paired-gex`) | `atac_pipeline.h:262,268` |

The 500 bp bin matrix of the old path is not part of PR #48. Gene activity, TSS enrichment and the other QC metrics are per-cell integer sums of
the same shape as `cell_metrics` in the reference; they are outside phase 1e's reducers (the TSS and gene-region joins are static
annotation lookups per fragment end) and are listed under "later" in section 6.

## 2. Order dependence in the CPU ATAC path

| # | where | dependence | effect | proposed handling |
|---|---|---|---|---|
| F1 | `atac_fragment.h:150-166` | the dedup table is cleared whenever (contig, start) changes ("starts arrive sorted") | on any order other than a coordinate-sorted BAM, duplicates separated by other starts are kept as separate fragments. In the fixture: 5,087 "fragments" for shuffled input vs 3,689 sorted. The GPU stream is unsorted, so this rule cannot be copied | **A1** |
| F2 | `atac_pipeline.h:156`, `atac_barcodes.h:138` | layout and whitelist chosen from the first 200,000 fragment records in BAM order | in a coordinate-sorted BAM those are the start of chr1: position-biased and different for a different sort. Practically stable (hit rates are far apart) but not a function of the set of reads | **A2** |
| F3 | `atac_peak_caller.h:150-170` | none after the insertions are sorted; the histogram is integer; the region merge runs over the canonical (contig, position) order | order-free given the multiset of insertions | keep; **A3** states the merge |
| F4 | `atac_metrics.h:129-150`, `atac_pipeline.h:276` | none: triplets are sorted, columns are sorted by barcode string | order-free | keep |
| F5 | `atac_cell_caller.h` (mixture EM, `call_cells_mixture`) | float EM over per-barcode vectors in barcode-index order | deterministic if the barcode axis is canonical; not audited line by line | keep on the host, on the canonical axis; list for the gate |
| F6 | `atac_barcodes.h:208` | ties between equal-count neighbours | cannot reach 0.975, so they are ambiguous, not order-dependent | keep |
| F7 | `atac_fragment.h:121-127` | multimappers | dropped by MAPQ >= 30 (unique only), decided inside one read | keep; the GPU aligner must give the same unique/multi call (gate (a)) |

Not order-dependent but worth knowing (reported, not changed): the cut-site pileup uses all barcodes (empty droplets and ambient fragments
shape the peak set; PR #48 open problem 2 is the 10k peak precision, higher recall than MACS3, F1 0.830 vs 0.837); `read_pairs` does not weight
cut sites in the peak caller but counts in `read_pairs` metrics.

## 3. Proposed order-free rules (in the style of `order-free-rules.md`)

Notation as there. These extend R1-R10 for the ATAC reducers; they need Zach's approval before they are implemented.
Principle unchanged: each rule is the most defensible biologically, not a copy of the sorted path.

### A1. Fragment dedup: a global exact key

Today: duplicates are merged only while consecutive records share (contig, start) (F1).

Rule: collect every emitted tuple (cell, contig, shifted start, shifted end), sort, and collapse equal tuples; `read_pairs` is the run length.
The set of fragments and their `read_pairs` are functions of the set of records.

Why: a PCR duplicate is the same template, wherever its record sits in the stream. The sorted-BAM rule is a memory optimisation that happens
to be correct only for one order. Identical to the CPU on a sorted BAM (tested: fragments equal the CPU port field by field), and it differs only
where the CPU is wrong. Dedup is by exact coordinates and cell, as Cell Ranger and ArchR do; no tolerance of ±1 bp and no barcode-collapse
across cells.

### A2. Barcode layout and whitelist: a hash sample

Today: first 200,000 fragment records in BAM order (F2).

Rule: the layout (offset, strand) and the whitelist candidate are chosen on the records with `hash64(qname) % 64 == 0` (the engine's `hash_qname`,
as R4), all of them, not a prefix. Ties: higher hit rate, then smaller whitelist index, then forward before reverse complement, then smaller
offset. The prior counts remain exact counts over all records.

Why: a uniform sample fixed by the read has no position bias; a prefix of a sorted BAM is the first megabases of chr1. This sits in the barcode
stage, upstream of the reducers; it is listed because gate (a) compares barcodes first.

### A3. Peak set: one canonical table, one threshold, one merge

Rule:
1. Insertions are the multiset of `(contig, position)` cut sites of unique non-mito fragments of all resolved barcodes (as today). Each cell
   partition reduces its own insertion table to sorted `(key, count)`; the merged table is the sorted union with counts summed (integer, commutative).
   The peak caller runs once, on the merged table, never per partition.
2. The only cross-position quantities are `lambda_bg` (a sum), the 100,001-bin integer score histogram (integer atomics) and the BH threshold taken
   from it. All three come from the merged table, so K cannot change them.
3. Candidates are positions in (contig, position) order; regions merge across gaps `<= 30` bp within a contig; the summit is the leftmost maximal
   pileup; p and q are maxima. Peak order is contig index then start, so peak ids are identical for every K.
4. Floats: scores are double, stored as float32 where the CPU does; `-fmad=false`. The significance decision is on the bin histogram (0.01 units), so a last-bit
   `lgamma` difference between CPU and GPU can only matter for a position within about 1e-13 of a bin edge. Gate (a) requires the peak set equal to the CPU's;
   any difference is reported with its position and score.

Why: the peak set is the one cross-cell table of this class (design 4.3); a canonical sort-and-sum makes it a function of the data.

### A4. Cell x peak: cut sites, columns in barcode order

Rule: unchanged from the CPU: each fragment end inside a peak adds one count to (cell, peak); chrM is skipped. The reducer's output is the sorted
sparse matrix over cell ordinals; the written columns (barcodes with `>= 100` fragments or called) are selected and ordered by output barcode string on the host
after the cell call.

Why: cut sites are the unit the Tn5 cuts and the peak caller uses; counting fragments (ArchR, Signac fragment-overlap) is a different, also valid, estimator but
changes the count scale. Keeping the CPU's choice keeps gate (a) meaningful. The cell axis is the whitelist index, a canonical order.

### A5. Per-cell metrics and the cell call stay integer sums plus a host mixture

Rule: `read_pairs`, `fragments`, `mito_fragments`, `peak_region_fragments`, `peak_region_cutsites` are integer reduce-by-cell. The mixture cell call runs on the host
over the merged per-cell table in whitelist-index order (and, for Multiome, joined to the GEX cell calls by output barcode string).

Why: float EM over a fixed order is deterministic (R7's last bullet); only integer work needs the GPU.

### A6. Multiome pairing is a lookup, not a join of streams

Rule: the ATAC cell ordinal is the index in the ATAC whitelist; the pair table maps it to the GEX barcode at write time. Fragments, peaks and the
matrix are computed on ATAC ordinals; the joint cell call joins on the output barcode string.

Why: pairing is a fixed table, so it commutes with every reducer.

### A7. Multimapped fragments: unique only

Rule: MAPQ >= 30 (unique under STAR). Not changed; reported so the GPU aligner's unique call is part of gate (a).

## 4. Reference reducers and test results

`bench/atac_reducers/atac_ref.py`: reducers in emit -> sort -> collapse -> reduce shape (`fragment_emit/reduce/merge`, `insertion_emit/reduce/merge`,
`peak_call`, `peak_count_emit/reduce/merge`, `cell_metrics`), plus sequential ports of the CPU code used as oracles (`cpu_fragments`, `cpu_peak_call`,
`cpu_cell_metrics_and_matrix`). The oracles take records in the order given, as the CPU does. `peak_call` is parallel-shaped: independent per-position
scores, an integer histogram, a threshold, a flag-scan-segment-reduce region merge. `cpu_peak_call` is the CPU's two-pointer scan with float32 where the C++ has float.
`run_all(rec, K, ...)` runs the whole thing in K cell partitions (`cell_part` is the splitmix hash of `pileup.cu`).

Fixtures are generated from a seed (`sim.py`), 11,524 records, 369 KB: either mate as READ1, PCR duplicates (632 duplicated fragments of 3,689), MAPQ 0/1/3 multimappers,
secondary, mate unmapped, cross-contig mates, improper pairs, 3% unresolved barcodes, 12 planted accessible regions on three contigs plus chrM.
There is no real-data test here (no HPC runs allowed); the real-data gates are in section 6.

Run:

```bash
python -m pytest tests/python/test_atac_reducers_ref.py -q
```

Result on the laptop (Python 3.11.9, numpy 2.2.4, numba 0.64.0): **14 passed**.

| test | what it shows |
|---|---|
| `test_poisson_tail_matches_long_double` | the tail equals the C++ test's long-double forward sum to 1e-6 relative |
| `test_fragments_equal_cpu_on_coordinate_sorted_input` | reducer = CPU port, all five fields, 3,689 fragments, max `read_pairs` > 1 |
| `test_fragments_invariant_under_shuffle` | three shuffles give identical output |
| `test_cpu_dedup_is_order_dependent_when_unsorted` | F1 reproduced: CPU port gives 5,087 on shuffled input vs 3,689 sorted; the reducer gives 3,689 |
| `test_fragment_geometry_is_leftmost_mate_for_either_read1` | left mate is READ1 in about half the pairs; the rule does not depend on it |
| `test_fragments_partition_invariant[K=1,2,7]` | merged partition output is byte-identical; partitions are disjoint in cell |
| `test_peaks_equal_cpu_port_and_find_planted` | 12 peaks, equal to the CPU port field by field (coordinates, summit, pileup, -log10 p, -log10 q as float32); 12/12 planted regions found |
| `test_insertion_table_merge_is_canonical` | the merged insertion table is byte-identical for K = 1, 2, 7 |
| `test_peak_counts_and_metrics_equal_cpu` | cell x peak triplets and the five per-cell counters equal the CPU port (610 nonzeros, 2,641 cut sites) |
| `test_end_to_end_byte_identical_over_K_and_order` | SHA-256 of fragments, insertion table, peaks, peak counts and metrics is identical for K = 1, 2, 7 and for the as-written order and two shuffles |
| `test_unbalanced_partitions_and_empty_partition` | K = 7 with 3 cells (empty partitions) equals K = 1 |
| `test_fixture_is_small` | fixture < 1 MB |

Limits of these tests: the oracle ports were written from the C++ by reading it, not run against it (no C++ compiler on the laptop). The arithmetic
(`lgamma`, `log`) is Python's libm and numba's, not the cluster's glibc or CUDA's. The CPU numbers on real data are what gate (a) is for.

## 5. How the peak table merges

1. Each cell partition emits two keys `contig << 32 | position` per unique non-mito fragment, sorts, and reduces to `(key, count)` (`red_insertions`).
   A partition holds only its own cells' fragments, so its table is bounded by the partition budget already sized up front.
2. The per-partition tables are moved to host memory (they are the one cross-partition state; at 10x Multiome 10k scale a few hundred million fragments
   give at most `2 N * 12 B`, a few GB, which fits host memory; if it must stay on the device, merge pairwise by key as the tables arrive).
3. After the last partition, one radix sort by key and one reduce-by-key (sum) give the merged table (`insertions_merge`). It is sorted by
   (contig, position) with counts, and is the same for every K because it is a sort-and-sum over the same multiset.
4. `lambda_bg` comes from the total count and the contig lengths; per-position scoring uses `lower_bound` on the merged keys and the prefix sum of the counts
   (the `cum` array: `cum[i]` = insertions before entry i), so one thread per grid position needs no neighbours.
5. Scores go into the 100,001-bin integer histogram by `atomicAdd` (commutative). The BH threshold is computed on the host from the histogram (100,001 entries).
6. Candidate positions are compacted in global (contig, position) order, regions are found by flag + inclusive scan, and per-region reductions are reduce-by-key
   (max pileup, leftmost argmax, max p, max q). The peak table is then sorted by construction.
7. Pass 2 re-reads the retained fragment tables of each partition and counts cut sites per (cell, peak) against the merged peak table (`red_peak_count`).
   The fragment tables stay resident per partition, or are recomputed from the reads if the partition loop frees them.

## 6. What W11 must build and gate

Build (in this order; none of it is done):

1. The unspliced paired-end read-level expansion that produces `FragRec` per alignment record, with `--alignIntronMax 1 --alignMatesGapMax 2000`, the
   PR #48 STAR options (`Extend5pOfReads12`, protruding mates kept), MAPQ and flags as STAR sets them. This is the real work of phase 1e; the reducers
   only consume it.
2. Barcode stage: layout/whitelist choice by A2, correction with the count prior, Multiome pair table.
3. `red_fragment`, `red_insertions`, `insertions_merge`, the peak kernels (the region-merge kernels are described, not written, in the draft), `red_peak_count`,
   per-cell metric reduce-by-key. Compile with `-fmad=false`.
4. The partition loop for the two passes (fragments and insertions per partition; one merge and one peak call; counts per partition).
5. Host side: the BH threshold, the mixture cell call, writers for `fragments.tsv.gz` (+ tabix), `peaks.bed`, `atac_peak_counts.mtx.gz`, `atac_barcode_metrics.tsv`.

Gates (design section 5), samples in `$WORK/data/atac-spatial/` (the PR #48 runs under `$WORK/pipeline-benchmark/runs/atacsp/` keep `atac_sorted.bam` and the CPU outputs):

| gate | sample | check |
|---|---|---|
| (a) identity with the CPU | `atac_pbmc_5k_nextgem`, `pbmc_multiome3k_atac`, `pbmc_multiome10k_atac` (and the two GEX runs for the Multiome pairing) | `fragments.tsv.gz` identical (same lines, same order), `peaks.bed` identical (position, summit, pileup, scores), `atac_peak_counts.mtx.gz` identical, `atac_barcode_metrics.tsv` identical, barcode correction counts identical. Report any difference with its cause. Build the CPU side from `singlet-atacsp` (88a41ff) or the merge of PR #48. |
| (c) partition invariance | the same three samples | forced K = 1, 2, 7 give byte-identical outputs of all of the above |
| (d) contracts | atac: 5k, Multiome 10k; multiome: 3k, 10k | `bench/public/capabilities/{atac,multiome}.py` exist only on the PR #48 branch; run them on the GPU outputs on the same commit as the other capabilities |

Also for (a): the shuffled-order replay as in the CPU gates (three record orders into the reducers, identical outputs), now meaningful for F1.

Known risks for W11, in the order they are likely to bite:
- F1 will make GPU fragment counts differ from the CPU **on an unsorted stream only if the CPU were fed one**; on the CPU's sorted BAM they agree. If the GPU count differs from
  the CPU on the sorted-BAM runs, it is the aligner (unique/multimapper, protruding mates), not the reducer.
- A device `lgamma`/`log` can differ from the host's in the last bit; the histogram bins are 0.01 wide, so peak sets should still be identical; if one peak boundary differs,
  look at its score against a bin edge.
- Cell axis size: the whitelist is 736K entries; per-cell arrays are sized by it, not by the number of cells seen.
- `cb + tb + sb <= 64` for the fragment hi key (20 + 10 + 28 for hg38); genomes with contigs > 268 Mb or > 1024 contigs need a third sort pass.
- The mixture cell call (F5) was not audited line by line; check that the host path is deterministic when it is ported.

## 7. For Zach

- Approve or amend A1-A7 (A1 and A2 change the sorted-BAM rules; A3-A7 restate the CPU's behaviour as order-free).
- PR #48 is still a draft and its CPU path is the gate-(a) baseline; the engine branch does not contain it.
- Not changed, but a question: peaks are called from every resolved barcode (empty droplets included), as on the CPU. Calling from the cell set after the cell call would be
  circular (the cell call uses peak fragments); the other option is a first pass with a fragment-count floor. Left as is.
