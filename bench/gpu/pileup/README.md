# W11 GPU annotation and pileup stage

Follow-on to M6 (`../m6/`, dedup, count sort, UMI collapse). Inputs: STAR transcripts per distinct read signature plus the per-read
(barcode, UMI, signature id). Outputs: every pileup output the CPU engine produces (`prod/campaign` c4a1be38,
`include/singlet/pileup/pileup_engine.h`, `run_parallel`), compared entry by entry with the engine's own accumulators.
Result file: `../results/pileup.json` (plus `../results/pileup_<sample>.json`). **Result: all nine outputs identical to the CPU engine on GSM4006843, GSM5411066 and GSM6470208 (and a 2 M-read PBMC subsample); 40-63 ms of GPU time for 61-89 M reads (1.1-1.5 G reads/s), 17-25 GB peak device memory; removes 3.3-7.6 host core-s per M read pairs. Details in section 5.**

Sources: `pileup.cu` (kernels, host loader, comparison, JSON; includes `../m6/m6.cu` for stages b-d), `extract.cpp` (BAM to records),
`pileup_rec.h` (record layout), `pileup_dump.patch` (CPU-engine dev patch: truth and annotation dump), `jobs/*.sbatch`.
Design input (read only, ideas re-implemented, no code copied): `singlet-agents/.claude/worktrees/gpu-align-vision/bench/gpu-pileup/pileup.cu`
and `results/RESULTS.md` (pileup as sort + merge-join + segmented reduce with an arena allocator, 60.5 ms for 60.7 M records; its CPU reference
was its own, not the Singlet engine). What was taken from it: the sort / join / segmented-reduce decomposition and the arena allocator
(M6's arena is reused). What differs: every rule below is the Singlet engine's, and the gate is the engine's accumulators.

## 1. Specification: what the CPU pileup computes (c4a1be38)

Campaign call: `singlet --pipeline ... --parallel-pileup`: STAR writes a coordinate-sorted BAM; `run_parallel` gives each worker a set of whole
chromosomes (greedy by record count) and calls `process_read_parallel` for every record. `config`: `min_mapq` 20, `min_baseq` 10,
`count_introns` true, `count_sj` true, `count_mt` true (pipeline mode), `umi_dedup` true, `umi_dedup_directional` true, `stranded` true,
`reverse_strand` = auto (probe, 5' header, strand safety net re-run; the final value is read from the dump), `umi_len` = detected.

### 1.1 Which alignments reach the counters (per BAM record, `process_read_parallel`)
| rule | where |
|---|---|
| unmapped (0x4): dropped (counted as total only); supplementary (0x800): dropped | start of function |
| secondary (0x100): only used for the multi-mapper buffer (1.3); never for junctions, SNPs, chrM | `is_secondary` branch |
| primary, no `CB` tag: dropped (`no_barcode`); `CB` not in the barcode set: dropped | `bc_index_.lookup` |
| `CB` lookup: exact string, else the string up to the last `-`; barcodes are the engine's refreshed list (CB-tag refresh, >= `CB_MIN_READS`); later duplicates overwrite | `BarcodeIndex` |
| `NH` tag (default 1, floor 1); `high_mapq` = MAPQ >= 20. Primary with NH == 1 and MAPQ < 20: dropped. With NH > 1 the primary goes on regardless of MAPQ. (STAR MAPQ: 255 for NH 1, 3 for 2, 1 for 3-4, 0 above, so `high_mapq` is NH == 1 here) | after CB check |
| UMI: tag `UB`, else `UR` (the campaign has only `UR`: STAR's UMI correction is off); raw string, length = `strlen`; no tag: counted without UMI (never occurred, the extractor aborts) | UMI block |
| base quality: `qual[0] != 0xff && qual[i] < 10` skips a base in the SNP and chrM pileups. The campaign decodes with `--quality none`: qualities are a function of the base, not of the read. In some samples N bases get quality < 10 (pbmc2m: all 2 795 N bases), in others every base is >= 10 and N bases count in DP (GSM4006843, GSM5411066, GSM6470208). The extractor checks every base and sets the rule (section 5.1) | pileup walkers |

### 1.2 Gene assignment, one primary alignment (gene model = `GeneModel`, trees = `IntervalTree`)
1. Aligned blocks = the M / = / X operations of the CIGAR (`BAM_CMATCH`, `CEQUAL`, `CDIFF`); D and N advance the reference only; I, S, H, P nothing.
   Block = `[ref_pos, ref_pos + len)`, 0-based half-open.
2. Exon query: for each block in order, `exon_trees_[tid].query(start, end)` returns every merged-exon interval with `start_i < end` and `end_i > start`
   (tid without a tree: nothing). The result list order is the tree's **pre-order** (node, left subtree, right subtree) over an implicit balanced BST
   built on the intervals sorted by (start, end) (`std::sort`, ties in implementation order), with augmented subtree max-end pruning. This order decides
   `first_exon_idx`; the GPU replays the same traversal on the dumped array.
3. Strand filter (only if `stranded`): for an interval of gene g with strand `gs` (`+`, `-`, `.`; `.` or unknown gene passes): `expect_reverse = (gs == '-')`,
   flipped if `reverse_strand`; the interval is dropped when `read_reverse != expect_reverse`. The same filter applies to introns (`continue`).
4. Over the filtered exon list `L_e` (block order, then tree order): empty -> no exon gene. `first_gene = gene(L_e[0])`; `unique = first_gene != NONE and
   every gene(L_e[i]) == first_gene`. unique -> `exon_gene = first_gene`, `first_exon_idx = L_e[0]`; not unique -> **multigene**: the read is not counted
   (`classify_read` returns nothing), even if it lies in one gene's introns.
5. Intron query (only if `count_introns`): same walk over `intron_trees_` with the strand filter; list `L_i`. `intron_gene` = unique gene of `L_i`
   (same rule) with `first_intron_idx = L_i[0]`.
6. `classify_read`: exon gene present: `gene = exon_gene`, `exon_idx = first_exon_idx`, `intron_idx` = **first index in `L_i` whose gene equals `exon_gene`**
   (intron hits of other genes are ignored; counted only in a stat); `flags = ExonOnly|TouchesExon (5)` if no such intron, else `Intron|TouchesExon (6)`.
   No exon gene and a unique `intron_gene`: `gene = intron_gene`, `intron_idx = first_intron_idx`, `flags = Intron (2)`.
7. `count_read` (NH == 1: immediately): no gene: nothing. `flags = rf.flags ? rf.flags : 5`. With a UMI the read goes to `DirectionalUmiStore::record_read`.
   Tuple = (barcode column, gene, packed UMI, exon_idx, intron_idx, flags).

### 1.3 Multi-mappers (NH > 1), the `mm_buffer` state machine (in-worker path; qname-keyed, resolved when `seen >= NH`)
Alignments of one read are processed in BAM order (coordinate order inside a chromosome). State: `gene_idx, exon_idx, intron_idx, flags, first_gene, ambiguous`.
- primary: if its `rf.gene != NONE`, **overwrite** `gene_idx / exon_idx / intron_idx / flags` with `rf`; `ambiguous |= primary_multigene`;
  `aln_gene = exon_gene != NONE ? exon_gene : intron_gene`; if `aln_gene != NONE`: `first_gene` unset -> set, else differing -> `ambiguous`.
- secondary (exon query only, `determine_exon_gene`: same walk, same strand filter, single-gene test on the whole list): multi-gene -> `ambiguous`;
  single gene g: `first_gene` unset -> g, differing -> `ambiguous`; if `gene_idx` is still NONE: `gene_idx = g, exon_idx = first overlap, flags = 5` (intron stays NONE).
  So: the primary's features win if it has a gene, otherwise the first secondary (in BAM order) that has one.
- resolution: `ambiguous` -> not counted (goes to the EM-rescue list, not part of these matrices); `first_gene == NONE or gene_idx == NONE` -> nothing;
  else `count_read(gene_idx, exon_idx, intron_idx, flags ? flags : 5)`. Groups that never complete (`seen < NH`) are resolved in the cross-worker merge by the same
  gene test; alignments of one read on chromosomes of different workers use a different tie rule for `exon_idx` (worker order, primary first); the GPU uses the BAM-order rule.
- Junctions, SNPs and chrM are only done for the primary alignment (secondaries return early).

### 1.4 UMI collapse and matrices (M6, `umi_dedup.h`, unchanged here)
`umi_pack_2bit` (A0 C1 G2, **T, N and anything else 3**); per (cell, gene) reads with equal packed UMI merge (count, flags OR, exon_idx min, intron_idx min); with
more than one UMI: edge iff Hamming == 1 and `count_i >= 2 * count_j - 1`; molecule = connected component; flags OR / exon min / intron min over the component.
Label: `ExonOnly` bit and `Intron` bit -> ambiguous (exon matrix at min exon index **and** the per-gene ambiguous count); `Intron` only -> unspliced (intron matrix);
else spliced (exon matrix). `SparseAccumulator::to_csc` sums per (feature, cell) and **clamps to 65535** (uint16) (SNP matrices: **255**, uint8).

### 1.5 Junctions (`extract_splice_junctions_parallel`, primary alignment, `count_sj`)
For every `N` operation: donor = reference position before the gap, acceptor = donor + len; name `chrom:donor-acceptor` (0-based). Condition: the read passed 1.1 (valid CB and
not the NH == 1 and MAPQ < 20 case). **Exact-UMI dedup** on (barcode, junction, raw UMI string with N distinct from T, FNV hash set); first occurrence counts; no
dependence on order. Per-worker accumulators are re-merged by name (junction ids differ between runs; compared by (chromosome, donor, acceptor)); clamp 65535 per worker, then
again 65535 at the end.

### 1.6 SNP AD / DP (`pileup_aligned_bases_parallel`)
Condition: `high_mapq` and the read overlaps a panel SNP. SNP table: per chromosome sorted by position (`std::sort`, equal positions in any order), entries (pos 0-based, column index,
ref, alt). The walker goes over M / = / X blocks; for every SNP with `pos` in the block: base = `seq_nt16_str[code]` at query position (soft clips and insertions advance the query);
quality filter; `umi_new = !umi_dedup || insert(barcode, snp column, raw UMI)` (exact string, **first read in BAM order** wins); if new: `DP += 1`, and `AD += 1` iff the base character
equals the alt character (an `N` base counts in DP only). Clamp 255 per (SNP, cell). The order dependence is real: two reads of the same (cell, SNP, UMI) can show different bases;
the GPU keeps the reads in BAM order and takes the first.

### 1.7 chrM pileup and mt variant inputs (`pileup_mt_bases_parallel`, `count_mt`)
Condition: `high_mapq` and `tid == chrM`. For every M / = / X base: `rp = ref_pos + i` (uint32); skip `rp >= 16569` (`MT_LEN`, rCRS, also for mouse); quality filter; `bam_seqi` -> `BASE_TO_IDX`
(A0 C1 G2 T3, N and others skipped); `mt_acc[rp * 4 + base] += 1` per **read** (no UMI dedup), and the same in `mt_acc_rev` if the read is reverse. Clamp 65535.
Indels: `I` at `ref_pos`, `D` at `ref_pos` (before advancing), only if `ref_pos < 16569`: event (barcode, ref_pos, `min(len, 65535)`, is_ins) per read, no dedup, kept as a list (compared as a multiset).

### 1.8 Tie-breaks and clamps, in one list
exon_idx / intron_idx per molecule: min over reads; per read: first in tree pre-order (exon, first block with a hit); first intron of the exon gene; multi-mapper: primary wins else first secondary; UMI N packed as T for genes but distinct
for SJ and SNP dedup; Hamming rule `count_i >= 2 * count_j - 1`; strand filter per interval; chrM `rp < 16569`; clamps 65535 (exon, intron, ambiguous, junction, mt, mt_rev) and 255 (AD, DP); indel length 65535.

### 1.9 Not ported (not part of the gate)
EM-rescue list of ambiguous multi-mappers (`ambig_reads_`; only its length is in the dump), poly(A) site records, VDJ gene usage, CRISPR guides, rRNA sampling, whole-whitelist ambient profile, per-cell alignment QC and read counts, chrM BAM output,
the strand probe and the strand safety net (the GPU takes `stranded` and `reverse_strand` as parameters, final values from the dump), and everything downstream of the accumulators (EM, ambient correction, export). `FASTASSIGN` (PR #41) is a STAR window-assignment
speed-up that leaves the BAM unchanged; it has no pileup rule and nothing in the engine's pileup depends on it.

## 2. GPU input record and where it comes from

`pileup_rec.h`. The unit is a **signature**: one distinct (read sequence, STAR transcript set). STAR maps a cDNA read from its sequence alone, so all reads with one
sequence share their transcripts (E0); the extractor hashes (sequence, NH, every alignment's chromosome, start, strand, primary flag, MAPQ, CIGAR) and counts how many sequences
have more than one signature: **0 in all four samples**. Per signature: NH, the read length, the read sequence (original orientation, BAM nt16 nibbles), and its alignments
(`AlnRec`: chromosome id, 0-based start, reverse flag, primary flag, MAPQ, CIGAR as BAM-encoded operations; alignments in BAM coordinate order). Per read: signature id, barcode
column (-1 when CB is missing or not in the barcode set), raw UMI (3 bits per character, N distinct), in the BAM order of the primary alignment (this order decides the first read of a
(cell, SNP, UMI) in the SNP pileup). The per-sample signature fraction (signatures / reads) is 0.38 (GSM4006843), 0.38 (GSM5411066), 0.25 (GSM6470208).

M5 (the STAR stitch port) emits STAR's final transcripts per distinct sequence: chr, strand, start, blocks (read start, genome start, length), nM, NH, HI, primary flag, score. The mapping to
`AlnRec` is: CIGAR operations from the blocks as STAR's `alignBAM` writes them (gap between blocks: read gap -> I; genome gap with a junction (`canonSJ >= 0` or annotated) -> N; other genome gap -> D;
read ends -> S), MAPQ from NH (255 / 3 / 1 / 0), reverse flag from the strand. Until M5 exists, `extract.cpp` builds the same records from the campaign's STAR BAM:
`extract BAM DUMPDIR OUTDIR` (CB -> column by the engine's `BarcodeIndex` rule; multi-mappers collected by qname until NH alignments are seen; a read with a missing or unknown CB is dropped;
no UB/UR aborts). Reads, sequences and BAMs exist only in the job's `/tmp`.

Truth and annotation come from a dev build of the CPU engine: `pileup_dump.patch` (applies to c4a1be38; `SINGLET_PILEUP_DUMP=<dir>`) writes, after the last `run_parallel`
(a strand re-run overwrites): `anno.bin` = the engine's own interval trees (the sorted arrays, so the pre-order is identical), gene strands, exon/intron to gene maps and the resolved SNP table; `truth.bin` =
the accumulators the callers receive (exon, intron, ambiguous, junctions with names, SNP AD/DP, chrM allele counts total and reverse, chrM indel list, `to_csc` sums and clamps included);
`params.txt` (stranded, final reverse_strand, min_mapq, min_baseq, chrM tid, ...), `barcodes.txt`, `chroms.txt`. With the variable unset the patch does nothing.
Engine binary: non-PGO build of c4a1be38 (job 789562), `SINGLET_KEEP_STAR_BAM=1`, the campaign command line plus `--parallel-pileup --threads 24`.

## 3. Kernels (`pileup.cu`)

1. **annotate** (one thread per alignment, not per read): walks the CIGAR; for each M/=/X block queries the exon tree (and, for primaries, the intron tree) with the engine's own pre-order traversal and pruning on the
   dumped arrays (a *sorted-probe interval join*: alignments are in coordinate order, so neighbouring threads walk the same tree paths; a plain sort + merge-join would not reproduce the CPU's result-list
   order, which fixes `first_exon_idx`, without an extra rank key). Strand filter, exon gene / multigene, intron gene, `classify_read` as in 1.2.
2. **signature level**: `resolve` = the multi-mapper state machine of 1.3 over the signature's alignments (NH == 1 direct); lists (count + exclusive scan + fill) from the primary alignment: junctions (N ops),
   SNP overlaps (per block binary search in the per-chromosome SNP array; base from the nibble sequence with strand correction), chrM base features, chrM indels.
3. **junction ids**: two stable radix passes sort the distinct junction instances (chromosome, donor, acceptor) -> ids by rank.
4. **expand**: per read, scan of the list sizes, then fill: gene tuples (the M6 `Rec`), junction (cell, id, UMI), SNP (cell, SNP, UMI, ref/alt), chrM (cell, feature, strand), indels.
5. **M6 b, c, d** on the gene tuples: unchanged (`../m6/m6.cu` is `#include`d).
6. **junction / SNP / chrM**: junction and SNP tuples go through two stable radix passes (UMI, then cell/feature; reads are in BAM order so the first of a run is the earliest read), first-of-run flag,
   reduce-by-key (junction: count of distinct UMIs; SNP: DP = unique (cell, SNP, UMI), AD = alt flag of the first read); chrM keys (cell, feature, strand) sorted and run-length encoded; clamps 65535 / 255.
No floating point, no atomics on values, no thrust sort; compiled `-arch=sm_90 -fmad=false`.

## 4. Method of the gate

For each sample one job (`jobs/pu_run.sbatch`) runs: dev engine on the real `.1fq` (CPU truth + annotation dump) -> `extract` (BAM -> records, BAM deleted) -> `pileup` on one H100 NVL.
`pileup` runs the full pipeline 1 warm-up + 3 timed times (CUDA events between stages, no host work in the timed region except the 4-byte size read-backs between stages), then twice more with the outputs fetched
(first and last run): their SHA-256 are equal (deterministic across runs). Every output is then compared with `truth.bin`: matrices entry by entry on (cell, row, value) (identical = same entries, same values, same count);
junctions on (chromosome, donor, acceptor) because the engine's junction ids depend on worker order; chrM indels as sorted multisets. Exit code 0 only if all nine compare identical.

## 5. Results (real data; H100 NVL; job ids in the table)

### 5.1 Identity with the CPU engine (c4a1be38), all outputs

| output (entries) | GSM4006843 (human 3' v2) job 789669 | GSM5411066 (mouse 3' v3) job 789670 | GSM6470208 (human 5' v2) job 789671 | pbmc2m (dev) job 789652 |
|---|---|---|---|---|
| exon matrix | **identical** (24 604 078) | **identical** (9 771 259) | **identical** (8 931 249) | **identical** (507 374) |
| intron matrix | **identical** (6 031 894) | **identical** (5 357 748) | **identical** (2 762 989) | **identical** (383 927) |
| ambiguous (per gene) | **identical** (39 356) | **identical** (128 722) | **identical** (413 682) | **identical** (747) |
| junction table | **identical** (8 586 171; 170 321 junctions) | **identical** (3 496 649; 206 952) | **identical** (7 532 947; 266 889) | **identical** (158 898) |
| SNP AD | **identical** (1 656 215) | **identical** (12 144) | **identical** (1 200 484) | **identical** (62 649) |
| SNP DP | **identical** (5 613 873) | **identical** (2 054 044) | **identical** (3 870 197) | **identical** (209 364) |
| chrM base counts (total) | **identical** (56 926 070) | **identical** (20 271 143) | **identical** (61 569 961) | **identical** (4 005 955) |
| chrM base counts (reverse strand) | **identical** (4 821 008) | **identical** (612 264) | **identical** (55 847 955) | **identical** (61 323) |
| chrM indel events (multiset) | **identical** (42 928) | **identical** (5 048) | **identical** (28 822) | **identical** (1 572) |
| deterministic across runs | yes | yes | yes | yes |

Sums of counts also agree (for example GSM4006843 exon 45 120 052, equal to the CPU truth of M6 job 789189; its 55 792 540 gene tuples equal the tuple count the M6 dump logged). 5' used `reverse_strand = 1` (header-detected; final value from the dump).
Per-sample details: `../results/pileup_<sample>.json`, logs in `../results/pileup_logs/`, merged file `../results/pileup.json`.

Two things the gate caught (both fixed in the extractor / kernels, neither by relaxing the gate): (a) the SNP pileup base-quality filter. The engine skips a base with `qual < min_baseq`.
In pbmc2m the decoded qualities make exactly the N bases low-quality (2 795 of 2 795), in GSM4006843, GSM5411066 and GSM6470208 every base has quality >= 10 and N bases are counted in DP (285 638 / 80 156 / 593 652 N bases).
The first GPU version skipped N bases everywhere and differed in DP/AD (pbmc2m: 3 DP entries; GSM4006843: 195 + 88 DP entries and 1 AD entry; GSM5411066: 5 + 2 DP entries). The extractor now checks every base of every primary record and writes the rule
(`snp_skip_n`: nothing skipped, or skip N, or abort when quality is neither constant nor tied to N, because per-read qualities are not part of the record). (b) nothing else: all other outputs matched on the first run.

### 5.2 GPU time (median of 3 after 1 warm-up, ms; transfers excluded)

| stage | GSM4006843 | GSM5411066 | GSM6470208 | pbmc2m |
|---|---|---|---|---|
| 1 annotate (per alignment) | 4.39 | 4.29 | 5.13 | 0.30 |
| 2 signature lists + multi-mapper resolve | 3.81 | 3.04 | 4.06 | 0.46 |
| 3 junction ids | 0.73 | 0.50 | 0.94 | 0.23 |
| 4 expand to read tuples | 12.35 | 9.16 | 17.93 | 0.72 |
| M6 b+c+d (gene collapse, CSR) | 21.61 | 15.88 | 16.20 | 1.28 |
| 5 junction count | 2.41 | 1.58 | 3.81 | 0.20 |
| 6 SNP AD/DP | 1.27 | 0.78 | 1.50 | 0.23 |
| 7 chrM counts | 8.70 | 4.37 | 13.55 | 0.53 |
| **total** | **55.3** | **39.6** | **63.1** | **4.0** |
| mapped reads in the records (M) / distinct signatures (M) / alignments (M) | 60.87 / 23.07 / 25.89 | 61.02 / 23.17 / 25.77 | 88.53 / 21.80 / 34.54 | 1.74 / 1.16 / 1.29 |
| **G reads/s** (records / total) | **1.10** | **1.54** | **1.40** | 0.44 |
| G read pairs/s (sample pairs 66.7 / 69.6 / 121.8 M / total) | 1.21 | 1.76 | 1.93 | |
| tuples: gene (M) / SNP (M) / chrM bases (M) / junction (M) | 55.8 / 7.4 / 179.6 / 15.8 | 47.5 / 4.0 / 91.4 / 10.1 | 62.9 / 9.5 / 277.4 / 25.8 | |
| **peak device memory (arena, GB; inputs resident included)** | **23.8** (inputs 3.1) | **16.7** (3.0) | **25.1** (3.6) | 0.9 |
| H2D of the records, pageable (ms / GB) | 762 / 3.0 | 802 / 3.0 | 891 / 3.5 | 27 / 0.1 |

The largest costs are the M6 collapse and the per-read expansion (memory traffic of 60-90 M reads x 16 B records plus up to 280 M chrM base tuples); expand plus gene collapse alone run at 1.8-2.8 G reads/s, in line with the 2.6 G pairs/s the exploration estimate gives for the count path. The whole stage is 1.1-1.5 G reads/s. H2D of the records from pageable memory (3-3.5 GB) takes 14-20 times the compute: in the GPU engine the records are produced on the device by M5, so this is
the extractor's cost, not the stage's. The arena peak is an exact count of the bump allocator (everything the stage allocates), not an nvidia-smi sample (the arena is reserved up front: 80 GB).

### 5.3 Host work removed (engine's own stage timers, campaign binary, 4 threads, two repeats each; `$W/bam-stream/runs/<sample>/t4_r{1,2}`)

| core-s per M read pairs | GSM4006843 | GSM5411066 | GSM6470208 |
|---|---|---|---|
| pileup + annotation (`[singlet] Pileup:`, wall x 4 workers) | 3.92 (65.4 s) | 2.87 (50.0 s) | 6.02 (183.2 s) |
| BAM index + CB-tag barcode refresh (wall, counted at 1 core: lower bound) | 0.44 (5.5 + 24.0 s) | 0.38 (6.4 + 19.9 s) | 1.62 (133.9 + 63.1 s) |
| **removed by this stage (sum)** | **4.36** | **3.25** | **7.63** |
| not counted: STAR coordinate BAM sort (4 threads) | 20.9 | 16.9 | 74.9 |

The pileup timer covers everything of `run_parallel`: per-read annotation and gene assignment, UMI collapse (finalize), SNP/junction/chrM walks and the accumulator merges. The 5' sample is slow on the host because one
chromosome holds 118 M of its 332 M BAM records (one worker does most of it) and 222 M of the records are secondary alignments. The BAM sort is listed because a GPU engine that never writes the BAM removes it too, but it belongs to the alignment output,
not to the pileup/annotation stages.

## 6. Job ids and budget

Build 789562 (the first build, 789541, failed in 2 s on the private-ref fetch: a branch checked out by the campaign worktree), compiles 789558, 789650, 789651, 789667 (debug-cpu, no GPU; 789647-789649 superseded), GPU runs:
pbmc2m 789645 (first run, SNP N-quality bug) and 789652 (final); GSM4006843 789653 (first run) and 789669 (final); GSM5411066 789654 and 789670; GSM6470208 789671. 789610 (debug-gpu) was cancelled while pending (QOSMaxCpuPerUserLimit).
GPU time used: **1.10 GPU-hours** (sum of the wall time of the GPU jobs, one H100 NVL each; limit 5). CPU time: **31.4 core-hours** allocated (GPU jobs 28 CPUs x 1.10 h = 30.8, builds and compiles 0.6; limit 150).
All engine runs were inside the GPU jobs (each job = engine + extractor + GPU, the reads and BAM only in its `/tmp`); nothing was written to `$W` but logs and the small result files, the dev binary and the sources.

## 7. Limits and deviations

- **Inputs are CPU-engine BAMs, not M5 output.** The record content is what M5 has to provide (CIGAR-equivalent blocks with gap types, primary flag, NH, MAPQ, and the STAR coordinate-sort order of reads: the extractor's BAM order stands for
  (chromosome, start, STAR tie order) that fixes the first read of a (cell, SNP, UMI) and the first secondary of a multi-mapper). If M5's order key differs from STAR's sort the AD matrix and a handful of multi-mapper feature indices can differ.
- **Multi-mapper tie rule.** The GPU uses the in-worker rule (BAM order). The engine uses a different rule for a read whose alignments sit on chromosomes of different workers (primary first, then worker order); the two
  differ only when the primary has no gene and two secondaries hit the same gene in different exons. No difference appeared in any of the four samples, although the cross-worker path is heavily exercised (the engine logged 4.57 M / 7.00 M / 22.78 M / 0.11 M cross-worker multi-mapper entries, of which 3.86 M / 1.13 M / 6.09 M / 0.09 M were gene-assigned, for GSM4006843 / GSM5411066 / GSM6470208 / pbmc2m), but the GPU does not reproduce the worker-order rule and is not guaranteed to match when the primary has no gene.
- **Base quality.** Reproduced under constant quality (the `--quality none` decode), verified base by base; samples with per-read qualities need a quality stream and abort in the extractor.
- **UMI.** The CPU dedup sets for junctions and SNPs are 64-bit hash sets; the GPU compares exact strings. They agree on all data (a collision would have shown as a count difference). UMI length <= 13 for the exact-string key, and M6's limits (cell + gene + UMI bits <= 64, feature index < 2^24 - 1).
- **Not ported** (1.9): EM-rescue list, poly(A) records, VDJ, guides, rRNA sampling, ambient profile, per-cell alignment QC, chrM BAM, strand probe and safety net.
- Four samples only (three required, one 2 M-read development subsample); the arena was sized at 80 GB, the peak at 25 GB.

## 8. Reproduce

`jobs/pu_build.sbatch` (dev build) -> `jobs/pu_compile.sbatch` (`nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false`; `extract.cpp` with the pbench g++ and htslib) ->
`sbatch --partition=short-gpu --time=01:30:00 --export=ALL,SET=GSM4006843 jobs/pu_run.sbatch` -> `python make_pileup_json.py` (after copying `$W/jobs/w11/pileup/results/<SET>.json` to `../results/pileup_<SET>.json`).
Sources are uploaded to `$W/jobs/w11/pileup/` (`gpu/pileup`, `gpu/m6`). Exit code of `pileup` is 0 only when every output is identical.
