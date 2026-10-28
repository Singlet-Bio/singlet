# W3b: does retiring the coordinate-sorted BAM cut atlas compute? (phases 1, 2, 4, 5 done; phase 3 partly)

Protocol: [bam-stream-protocol.md](bam-stream-protocol.md). Scripts and the raw per-run record: `bench/bam-stream/` (`results.json`).
Draft PR Singlet-Bio/singlet#60 (this report) and branch `perf/bam-stream` (one debug env var, no change in default behaviour).

## Update 2026-10-28: the streaming prototype (flag `--stream-unsorted`) exists and runs

Branch `perf/bam-stream` (draft PR Singlet-Bio/singlet#63; merged with `prod/campaign` at `c4a1be38`, so #41 and #62 are included). The
flag is opt-in; the sorted path is the default and unchanged. What it does: STAR writes an unsorted BAM to a pipe without `CB`; the serial
streaming pileup runs concurrently with STAR; the barcode comes from `CR` by the rule in phase 3 (`inline_scan_cb_from_cr`); there is
no sort, index, CB-tag scan, strand probe re-run or BAM on disk. Binary: commit `db696122a32f2e1e5cf41d167ca84d1485458ace`
(not PGO), sha256 `c54ec8ec22d8ccc0cc56b775ae73d29c5200cbc95105db74ce8a65c198a90978`.

**Saving (same binary, flag off vs on, 4 threads, exclusive node, barcode list held fixed):**

| set | sorted wall (s) | streaming wall (s) | saving | streaming RSS | cgroup peak | job `/tmp` peak |
|---|---|---|---|---|---|---|
| GSM4006843 (3', 74 M records) | 966 | 776 | **19.6%** | 57 GB | 58 GB | 3.7 GB |
| GSM7901264 (5', 112 M records) | 1,246 | 1,013 | **18.7%** | 53 GB | 54 GB | 7.0 GB |

The sorted path peaks at 54-74 GB RSS, so **peak memory does not grow**. Four more datasets were run in streaming mode only
(GSM7036403 997 s, GSM6470208 1,295 s, GSM5411066 526 s, GSM7656149 3,169 s; RSS 54-73 GB). They have no same-binary sorted run, so
their savings cannot be attributed: against the exclusive replicate of the pinned binary they are -33 to -48%, against the post-#41 PGO
binary -24 to -45%, and both comparisons include the effect of the other speed-ups in `c4a1be38` (and PGO).
The like-for-like figure is the 19-20% above, which already meets criterion (a).

**Identity: not byte-identical; the cause is not fully isolated.** Held to the same barcode list and order, 37 of 63 output files
(3') and 34 of 61 (5') are byte-identical; the rest differ by small amounts:
- Molecule counts differ by at most 0.002%: GSM4006843 `molecules_spliced` 45,064,155 vs 45,064,250, `total_umis` 45,103,837 vs
  45,103,935; GSM7901264 `molecules_spliced` 4,718,591 vs 4,724,251 (0.12%).
- Junction hits differ by +0.05% (3') and +3.4% (5'), and the junction feature table is larger in streaming mode.
- `rrna_reads`, `ase_counts.tsv`, `snp_ad/dp.1pz`, `cell_qc_metrics.tsv` and the downstream files built from them differ.
- Without the fixed list the streaming run chooses a different barcode set (13,302 vs 13,098 cells; 60.85 M vs 59.53 M barcoded reads):
  the sorted path re-derives its list from a full scan of the sorted BAM (at least 5 primary records per `CB`, in the order the
  scan first sees them, ties included). A streaming path has no such scan; this is the "barcode-order dependence" W2 already
  tracks, and needs either the pass-1 set or a pre-scan of the unsorted BAM.
- Running the sorted path with one pileup thread (to rule out the parallel pileup's cross-worker multimapper count) leaves 25-26
  differing files, so the parallel pileup is not the main cause. The remaining candidates (UMI-dedup and multimapper tie-breaks that
  depend on read order, a different strand probe sample - 10,000 reads, antisense fraction 0.022 vs 0.056, same call - and
  junction filtering) were not separated.

**Not done:** the determinism runs (4/8/16 threads, rerun) and the Kang and high-mtDNA samples. Alpha's `stmig` array was waiting on
resources in the scavenger partition, so under the Alpha-first rule I stopped submitting; I cancelled three exclusive jobs
(789728-789730) rather than hold whole nodes. They are ready to submit (`phase1.sbatch` with `SUF=_b1`, `T=4/8/16`, `BCFROM`).

## Result before the prototype (kept for the record)

Retiring the sorted BAM would save **an estimated 19-39% of atlas core-hours (medians across scenarios; 13-58% across
sample profiles)**, so the phase 2 stop rule (15%) does not fire. The feared identity risk, rebuilding STAR's barcode correction, is
**absent**: in this pipeline the `.1fq` encoding has already resolved every barcode, and STAR's `1MM_multi_Nbase_pseudocounts` never
changes one (zero one-mismatch matches in all six datasets; `CB == CR` on every read with a valid barcode, checked record by
record on 111.8 M BAM records of a 5' sample). **No streaming prototype exists yet**: the existing streaming path cannot run with the
campaign configuration, and the remaining work (below) is the two-week job. So criterion (a) is a bound, not a measurement, and (b)
and (c) are untested. The cheaper alternatives recover little (in-RAM sort about 2% of wall; external `samtools sort` at most about
7%). **Recommendation: build the prototype, scoped as in "What the prototype needs".**

## Binaries (sha256 of the file that ran)

| name | commit | sha256 | used for |
|---|---|---|---|
| `singlet-campaign` (pinned) | `02aa9b938b54e0f92bba00c44c07b883a8660aa2` | `261d3a50ed624572418d30489dfd58904323f6e0d8820bd5260dfe7fb5038e6c` | main baseline, in-RAM sort, streaming-pileup cost |
| `singlet-pc-28969b8` (PGO) | `28969b87b2ca237f9a2344792e04b59f3cabf3e9` (`prod/campaign` after #41 FASTASSIGN) | `0317a5b6414fc8c470d374a10379318029a22787965081a037b593e8a585db33` | re-baseline after #41 merged (1 replicate, 4 threads) |
| `singlet-bs-dev` | `f8def8c8e2e161227911750077f8d623bf0eeac8` (`perf/bam-stream`, no PGO) | `a70965ae6584e1db99ab467de8507c8956614138c42b479a0ed54486a50a8360` | barcode check (`SINGLET_KEEP_STAR_BAM=1`, `SINGLET_EXTRA_SAM_ATTRS=CY`), external sort |

PR #62 (`c4a1be38`, serial speed-ups) merged after these runs and was not measured.

## Datasets (real public data) and what is missing

| id | what | reads (M) |
|---|---|---|
| GSM4006843 | human 3' | 66.7 (stand-in for the 66 M PBMC reference) |
| GSM5411066 | mouse 3' v3 | 69.6 |
| GSM7036403 | human 3' v3, deep | 111.6 |
| GSM7901264 | human 5' | 100.3 (stand-in for 122 M) |
| GSM6470208 | two runs, 3' v3 (#47 concatenation) | 121.8 |
| GSM7656149 | human 3' v3, very deep | about 260 |

Not covered: the Kang genotype pool and a high-mtDNA sample (they were to supply demux, doublet, SNP and mtDNA truth in phase 5;
with no streaming output to compare, they are not needed yet), and the exact 66 M and 122 M samples. GSM6250705 was dropped: its
staged `.1fq` is corrupt (STAR: "quality string length is not equal to sequence length" at read 30.5 M).

## Phase 1: baseline of the sorted path

Setup: Clipper `short-bigmem`, 80 GB, disk sort, private genome, `--nice=10000`, arrays of `%2`. Replicates 1-2 at 4 threads ran four to a
node (co-tenant CPUs 32-44, recorded per run); replicate 3 and all 16-thread runs ran `--exclusive` (one job per node).
**Co-tenancy inflates the sort up to 5x** (GSM6470208 sort: 2,274 s shared, 844 s exclusive; GSM7036403: 383 and 859 s shared,
313 s exclusive). Deviations from the protocol: 16 threads has 2 replicates, not 3, and the re-baseline on the post-#41 binary has 1
(arrays cancelled to fit the budget). Medians, seconds:

| set | T | n | wall | STAR map | **sort** | index | CB refresh | pileup | sort+idx+refresh / wall | RSS GB |
|---|---|---|---|---|---|---|---|---|---|---|
| GSM4006843 | 4 | 3 | 1231 | 571 | 317 | 6 | 24 | 65 | 28% | 57 |
| GSM4006843 | 16 | 2 | 638 | 134 | 173 | 8 | 21 | 65 | 32% | 56 |
| GSM5411066 (mouse) | 4 | 3 | 875 | 314 | 287 | 7 | 20 | 51 | 36% | 53 |
| GSM5411066 | 16 | 2 | 557 | 82 | 183 | 11 | 19 | 40 | 38% | 54 |
| GSM7036403 | 4 | 3 | 1637 | 715 | 383 | 10 | 33 | 92 | 26% | 58 |
| GSM7036403 | 16 | 2 | 937 | 166 | 291 | 14 | 31 | 75 | 36% | 57 |
| GSM7901264 (5') | 4 | 3 | 1601 | 828 | 397 | 9 | 25 | 53 | 27% | 54 |
| GSM7901264 | 16 | 2 | 797 | 195 | 245 | 13 | 25 | 46 | 36% | 54 |
| GSM6470208 (2 runs) | 4 | 3 | 4199 | 1058 | 2274 | 118 | 67 | 197 | 59% | 61 |
| GSM6470208 | 16 | 2 | 1709 | 258 | 833 | 105 | 65 | 182 | 59% | 59 |
| GSM7656149 (260 M) | 4 | 3 | 5395 | 3038 | 950 | 25 | 99 | 200 | 20% | 74 |
| GSM7656149 | 16 | 2 | 2610 | 682 | 778 | 26 | 98 | 186 | 35% | 72 |

Observations:
- The sort does **not scale with threads** (GSM4006843, exclusive node: 170 s at 4 threads, 173 s at 16), so its share of wall grows
  with threads.
- The two-run sample (#47 path) is the worst case: its sort is 54% of wall at 4 threads.
- Peak job `/tmp` is 16-85 GB. `memory.peak` reads 81,920 MB (the cgroup cap, page cache) on most runs; RSS is 53-74 GB.
- Post-#41 binary (`28969b8`, exclusive nodes, 1 replicate): mapping is 10-27% faster, and sort+index+refresh is 20-42% of wall.

## Phase 2: upper bound

Streaming pileup cost, measured with the existing serial `--bam-file` streaming path on GSM4006843: **60.85 M reads in 108-115 s**
(1.8 s per M reads), against 65 s for the four-worker parallel pileup of the same sample. The input was the `--tagged-bam` BAM, which
holds 82% of the reads (the pipeline deletes STAR's own BAM), so the cost is scaled by reads.

Atlas weighting (`atlas_bound.py`): 56,606 single-cell samples (`samples.parquet`, `reads`, else `catalog_read_count`; median 207 M),
wall = 130 s fixed + linear in reads per measured profile; saving = sort + index + CB refresh + (parallel pileup - streaming pileup).
Scenario B: the streaming pileup costs what today's parallel pileup does. Scenario A: it stays serial (measured). Share of atlas
core-hours saved (min / median / max over the six sample profiles):

| configuration | B (parallel streaming) | A (serial streaming) |
|---|---|---|
| 4 threads, pinned binary, all replicates | 19.9 / 29.4 / 58.2% | 15.6 / 23.8 / 51.3% |
| 16 threads | 34.5 / 39.1 / 57.9% | 22.9 / 27.4 / 40.2% |
| 4 threads, post-#41 binary | 21.5 / 25.2 / 41.6% | 13.9 / 19.4 / 29.1% |
| 4 threads, in-RAM sort as the baseline | 20.2 / 21.0 / 24.5% | 12.6 / 16.7 / 17.4% |

All medians exceed 15%; the minimum of scenario A falls to 12.6-13.9% on the post-#41 and in-RAM baselines. Caveats:
- The sort share is extrapolated linearly to the atlas median depth (207 M reads, deeper than all but the 260 M profile).
- Clean (exclusive-node) sort times are lower than production's four-to-five-way packing, so these medians sit between the shared and
  exclusive timings.
- A streaming pileup piped from STAR would overlap the mapping (about 1.8 s per M reads against about 6 s per M for STAR at 4
  threads). That favours B but is unproven.

## Phase 3: barcode correction (the identity risk), and what is not built

**STAR's barcode correction is a no-op here.** `Barcodes.stats` of all six datasets: `yesOneWLmatchWithMM = 0`,
`yesMultWLmatchWithMM = 0`, `noNoWLmatch = 0`; every non-N barcode is already an exact whitelist match (the `.1fq` encoding resolves
it). Record-level check on STAR's sorted BAM (`SINGLET_KEEP_STAR_BAM=1`, `cb_tally.awk`):

| sample | BAM records | `CB == CR` | `CR` all N, `CB = -` | valid `CR`, bad UMI (N or homopolymer), `CB = -` | any other |
|---|---|---|---|---|---|
| GSM7901264 (5') | 111,842,216 | 108,281,494 | 3,544,340 | 16,382 | **0** |
| GSM4006843 (3'), earlier tally | 73,746,398 | 71,051,134 | 2,695,264 (both `-` classes together) | | not split |

So a streaming path reproduces STAR's `CB` exactly with this rule: `CB = CR`, except `CB` is absent when `CR` contains N or the read's `UR`
contains N or is a homopolymer. It needs no pass-1 counts, pseudocounts or qualities. Also, the sorted BAM written today has no `UB`
tag, so the pileup already counts UMIs from `UR`.

**The existing streaming path does not work for the campaign configuration.** `--no-parallel-pileup` makes STAR write an unsorted BAM to
a pipe, but the command still asks for the `CB` attribute with `CB_UMI_Simple`, which STAR allows only in a sorted BAM: all four trial
runs died at STAR start ("CB and/or UB attributes ... can only be output in the sorted BAM"), then "Cannot read BAM header".

### What the prototype needs (not done)

1. STAR: unsorted output without `CB`; the pileup applies the `CB` rule above (the engine reads `CB` in `inline_scan_cb`,
   `pileup_engine.h` around lines 274, 2925 and 2989, and in `singlet.cpp` around lines 7747 and 8413).
2. The barcode set: today a full BAM scan counts reads per `CB` (at least 5) to refresh `auto_barcodes.tsv` (21-100 s). A streaming
   pass has to take the pass-1 `.1fq` set, or write the unsorted BAM once and scan it twice (no sort, no index).
3. Strand: take it from the header and run screen; drop the strand probe and re-run (they need random access).
4. Order-independent accumulation in every caller, with per-position tallies for SNP, mtDNA and introns; measure peak memory.
5. `--keep-sorted-bam` as the debug path (`SINGLET_KEEP_STAR_BAM=1` already exists and is what the tests above used).
6. Identity: `perf_equiv` on all eight protocol datasets, plus the Kang pool and a high-mtDNA sample for demux, doublet, SNP and mtDNA truth.

## Phase 4: cheaper alternatives (measured)

| option | result | verdict |
|---|---|---|
| In-RAM sort (`--bam-sort-mode memory`, 400 GB, exclusive node) | sort 158, 281, 231 s vs 170, 313, 241 s on disk (GSM4006843, 7036403, 7901264): **7-10% of the sort, about 2% of wall**; cgroup peak up to 111 GB | not worth it: the sort is CPU-bound, not I/O-bound |
| External `samtools sort` of a hash-shuffled BAM on the NVMe `/tmp`, 20 GB | 96.5 s at 4 threads, 88.9 s at 16, plus 8 s index, against about 170 s for STAR's in-run sort | **at most about 6-7% of wall**, before adding the cost of STAR writing the unsorted BAM (not measured) |
| Skip the BAM for counts-only samples | not measured; Beta policy has no such class | none |

Both alternatives recover well under 15%.

## Phase 5: identity and determinism (existing paths only; there is no streaming output to test yet)

`out_equiv.py`, 56-60 output files per run (full records in `results.json`):
- **Reruns of the same binary and thread count:** identical except timings and the byte-count provenance field.
- **Disk sort vs in-RAM sort; pinned binary vs post-#41 binary:** identical except timings, sort-mode and byte-count provenance.
- **4 vs 16 threads:** 7 files differ (`sj_counts.1pz`, `sj_features.tsv.gz`, `splice_psi.1pz`, `rrna_report.json` and others): a pre-existing
  thread-count dependence of the junction and rRNA outputs, not a streaming effect.
- **Non-host EM table:** differs between reruns of the same binary (job 789001, reported by the orchestrator): pre-existing nondeterminism.

## Phase 6: decision (after the prototype)

| criterion | state |
|---|---|
| (a) measured saving at least 15% of atlas core-hours | **met on the like-for-like pair: 18.7-19.6% of wall** (3' and 5', 4 threads, exclusive nodes); core-hours follow wall at fixed threads. The atlas-weighted bound (19-39% medians) says the saving grows with depth. Production's shared-node sort is slower, which favours streaming further |
| (b) identical outputs, or every difference benchmarked and accepted by W7 | **not met yet**: 37 of 63 / 34 of 61 files byte-identical, the rest differ by 0.002% (molecules) to 3.4% (junction hits); the barcode list needs a rule; the cause of the residue is not isolated; determinism untested |
| (c) peak memory fits 16 jobs per 512 GB node | **no regression**: RSS 53-73 GB streaming vs 54-74 GB sorted; the packing plan's per-job budget (about 32 GB) is unmet by both and depends on the shared-genome work, not on this change |

**Recommendation: proceed with the streaming path as an opt-in flag, but do not make it the Beta default until (b) is closed.** The
saving is real (about 19% of wall at idle nodes, more under co-tenancy) and costs no memory. Closing (b) needs, in order:
1. A barcode-set rule that does not depend on BAM order (sort ties lexicographically in both paths; take the set from the pass-1
   `.1fq` counts or a pre-scan of the unsorted BAM). W2 has this as bug 5.
2. Isolate the residue by feeding the same unsorted-order BAM to the serial pileup with a sorted-order BAM of the same reads
   (`--bam-file`), which separates read-order effects from the strand probe and the junction filter.
3. Determinism (4/8/16 threads, rerun) and the Kang and high-mtDNA samples for demux, doublet, SNP and mtDNA truth.
4. W7's acceptance of whatever remains. If it cannot be accepted, keep the sorted path as the default for the Beta software
   freeze and fall back to the external sort at about 6-7%.

## Housekeeping

- The staged `.1fq` files under `$WORK/data/bam-stream/` (about 4.4 GB) are deleted once this report is reviewed.
- Clipper cost: roughly 120 core-hours of shared-node baseline plus about 20 node-hours exclusive, all at nice 10000 and `%2` or less.
- Jobs are named `bs-*`; scripts are in `bench/bam-stream/`.

## Update 2026-10-28 (b): determinism runs on GSM4006843

Streaming (`--stream-unsorted`, binary `singlet-bs-stream`) vs the sorted path, 4/8/16 threads and reruns, one exclusive node per run
(jobs 790002, 790010-790015; compare jobs 790026-790032). Barcode list fixed from the sorted run (runs 1-4, fixed list disables cell
calling, so summary says `align_zero_cells` and the job exits 4: expected) or auto (runs r3-r5, cells=13,302, status success).

| pair | files identical | differing |
|---|---|---|
| streaming T4 vs T8 | 48/63 | nonhost EM, pileup_stats, donor VCF + coverage, snp_ad, read_stats, rrna_report, sj_counts, sj_features, splice_events, splice_psi, star_Log.std.out |
| streaming T4 vs T16 | 48/63 | same 12 |
| streaming T4 rerun | 51/63 | the above minus donor VCF/coverage/snp_ad (9) |
| streaming auto T4 rerun | 49/63 | 12 |
| streaming auto T4 vs T16 | 50/63 | 12 |
| **sorted T4 rerun (control)** | **56/63** | nonhost EM, read_stats (1 line), 13 bytes of a pileup_stats provenance size |

Reading: the exon/intron/gene count matrices are identical across threads and reruns in streaming. Junction counts (and the
splice tables built from them), rRNA sampling, read_stats and the donor SNP tallies are not: they vary run to run, where the sorted
path reproduces them exactly (only the pre-existing nonhost EM and a one-line read_stats differ). The likely cause is STAR's unsorted
emission order (thread timing), which read-order-dependent steps (first-N sampling, junction filters, SNP tallies) inherit.

Criterion (b) is therefore not met and is not near the noise floor for junctions/SNPs; it needs those steps made order-independent
(or the stream re-ordered) before streaming can be the default. (a) and (c) are unchanged. Not run: shuffled-order test, Kang pool,
high-mtDNA sample (staging needed). Recommendation stands: opt-in flag only.

## Update 2026-10-28 (c): what actually varies, and why

Job 790050 diffed the differing files field by field (runs_d1 streaming T4 r1/r2/T16; runs_s0 sorted T4 r1/r2; sorted vs streaming).

**Run-to-run variation inside streaming is a byte-order effect, not a count effect.**

| file | what differs | cause (code) |
|---|---|---|
| `sj_counts.1pz`, `sj_features.tsv.gz`, `splice_events.tsv`, `splice_psi.1pz` | nothing in content: 170,281 junctions in both, 0 only-in-A/B, per-junction `n_molecules`/`n_cells` differ in **0 rows**; only row order | junction index assigned first-seen (`sj_names_.push_back` in `extract_splice_junctions`): STAR's unsorted emission order is thread-timing dependent, so the feature order differs; counts are set-based (`UmiDedup` = exact-match hash set) and order-independent |
| `rrna_report.json`, `pileup_stats.json:/rrna_*` | 19 vs 23 vs 22 rRNA hits of ~73.5k sampled | "every 1000th mapped read" sampling: a different 1/1000 slice each run; 1-in-1000 sampling noise, not a signal change |
| `read_stats.tsv` | one line: `wall_seconds` of the pileup stage | timing only (the TSV is not in out_equiv's timing-ignore list) |
| `pileup_stats.json:/strand_probe_antisense_frac` | 0.0218 (T4) vs 0.0189 (T16); same decision (forward) | inline probe uses the first 10,000 informative reads of the stream; the sorted path samples 64 windows over 4 chromosomes (18,835 reads, 5.6%) |
| `nonhost/nonhost_em_abundance.tsv` | 6th decimal of a probability; the **sorted rerun also differs here**, including one read count (57,473 vs 57,474) | pre-existing floating-point/threading nondeterminism in the nonhost EM (not a BAM-order effect) |
| `private/snp_ad.1pz`, `donor0.vcf`, `donor0_coverage.tsv` | identical between same-T reruns (byte-identical files), differ across T4/T8/T16; `snp_hits` identical (7,398,620) in all | donor demux is threaded (`dcfg.threads = export_cfg.threads`, VB with seed 42): the reduction order changes with the thread count. Whether `snp_ad` itself differs or only its downstream is settled by the sorted T16 control (job 790051/790052, below) |

Every pileup tally that is a count (`exon_hits 48,836,959`, `intron_hits 6,934,289`, `sj_hits 15,116,871`, `snp_hits`, `secondary_reads`, `multigene_reads`, `wrong_strand`) is identical across all five streaming runs. The exon, intron and gene matrices are byte-identical. So the serial `run()` is order-independent in what it counts; it is order-dependent only in (i) the order it names junction features, (ii) which reads the 1/1000 rRNA sampler sees, and (iii) which reads feed the strand probe.

**Sorted vs streaming (same binary, same barcode list) is systematic, not stochastic.** Both streaming T4 runs give the same numbers, so the gap to the sorted run is deterministic:

- `sj_hits` 15,109,243 (sorted) vs 15,116,871 (streaming), +0.05%; 1,418 of 170,281 junctions differ in `n_molecules`. Exon/intron/SNP hit counts are identical. Not yet isolated; see the shuffle test below.
- The large `n_cells`/`n_cell_molecules` differences in `sj_features` (51,838 rows) are an artifact of the comparison setup: `--barcodes` disables cell calling, so the streaming run's "cells" are all 119,449 barcodes. Not a streaming effect.
- Molecules 0.002-0.12%: the only order-dependent decision in gene counting is the EM-rescue candidate pair for ambiguous multimappers (`first_gene`/`second_gene` = the first two genes seen; in a sorted BAM "first" is leftmost on the genome, in the stream it is STAR's primary-then-secondaries order). Not yet proven to be the whole residue.
- Strand probe: a different sample (genome-wide windows vs first 10,000 informative reads). Same decision on both test samples. On a library whose antisense share is near the 0.4-0.6 band the two probes could decide differently, and in the stream the reads before the decision keep the starting orientation.
- Barcode set with auto selection (13,098 sorted vs 13,302 streaming, each reproducible within its mode): the sorted path refreshes the candidate list from a CB-tag scan of the BAM (>=5 primary records, ties by first-seen); the stream has no scan, so the candidate universe and hence EmptyDrops' ambient model differ (alpha 33,506 vs 36,219).

**Does sorting create artifacts of its own?** Sorting does not change any count; the sorted path's advantage is reproducibility of bytes (56/63 identical on rerun, the rest timing/FP). What sorting changes is the order in which per-read decisions see evidence: (a) a multimapper's alignments are scattered across the genome, so `mm_buffer` must hold up to 500k partial entries and the parallel path merges partial entries across workers at region boundaries; in the stream every read's alignments are adjacent and primary-first, so entries complete immediately and nothing is flushed incomplete. (b) The parallel pileup partitions by region, so per-worker UMI dedup and SJ dedup are merged afterwards; the stream dedups globally. (c) The strand probe can sample the genome. So the stream order is the cleaner order for per-read logic (multimappers, dedup); the sorted file is better only for genome-sampled probes. Neither is more correct on the evidence here; the sorted path is simply the one whose first-seen choices are reproducible because the file order is.

**How to make the result provably order-independent** (engine work in `pileup_engine.h` and export; W2 owns the file):

1. Name features canonically: sort `sj_names_` (chrom, donor, acceptor) and remap `sj_acc_` before export; build splice events from the sorted list. Removes the whole junction/splice byte difference.
2. Sample by hash, not by position: rRNA sample = reads with `hash(qname) % 1000 == 0`. Deterministic and order-free; also removes the periodicity.
3. Canonical EM-rescue candidates: keep the candidate gene set per multimapper (or the two smallest gene indices), not the first two seen.
4. Candidate barcodes from the `.1fq` pass-1 (CR counts; barcodes are whitelist-exact), ordered by count with lexicographic ties. Same list in both modes, no BAM scan.
5. Strand probe: take the orientation from the protocol table and use the probe only to confirm; or buffer the pre-decision reads' exon candidates for both orientations and commit after the decision.
6. Threaded floating point (demux VB, nonhost EM): fixed partition and ordered reduction. Pre-existing, shared by both paths.
7. Acceptance test: run `run()` on the same unsorted BAM in three shuffled orders (`samtools collate`) and require byte-identical outputs. This proves order-independence directly, without STAR's timing in the loop, and separates the remaining sorted-vs-stream residue (`sj_hits` +0.05%, molecules) into "order" vs "different input" (CB from CR, strand probe).

**Performance around streaming.** In the stream the pileup is serial and runs alongside STAR: pileup wall 628 s at T4 (STAR 9.5 min) and 282 s at T16 (STAR 4.4 min, 909 M reads/h, about 250k reads/s). The serial pileup processed 73.7 M reads in 282 s, about 261k reads/s, so at 16 threads it is already the bottleneck; above that the saving shrinks. Options, in order of value:

- Barcode-sharded streaming pileup: route each record by `hash(CB) % W` to one of W workers with their own accumulators and dedup sets. All reads of a barcode (and all alignments of a read, which share CB) land in one worker, so per-worker dedup and `mm_buffer` are exact with no merge step, the result is order-independent by construction, and memory does not duplicate (disjoint barcode sets). This is the existing `run_parallel` worker structure partitioned by barcode instead of region.
- Keep `--outBAMcompression 0` on the pipe (already so); the remaining stream cost is BAM record parsing, which the sharded design parallelises.
- The sort itself (phase 4) does not scale with threads and is inflated up to 5x by co-tenancy; streaming removes it entirely, which is the whole 19%. The in-RAM/NVMe sort alternatives measured in phase 4 recover less than that.

## Update 2026-10-28 (d): sorted T16 control, and the SNP-allele cause

Sorted path, T4 vs T16 (jobs 790051/790052): 52/63 identical. Differing: nonhost EM (FP, pre-existing), rRNA sample (19 vs 20 of ~73.5k), `read_stats` timing, and `sj_counts`/`sj_features`/`splice_psi` by byte order only (the parallel pileup also names junctions in discovery order, which changes with the worker count). **`snp_ad.1pz`, `donor0.vcf` and `donor0_coverage.tsv` are byte-identical across T in the sorted path**, so their thread dependence is streaming-specific.

Cause, from the code, correcting the demux-threads guess in (c): `export_cfg.threads = config.threads = pileup_threads = 1` in both modes, so the demux is single-threaded, and `SparseAccumulator::to_csc()` accumulates densely and sorts rows, so it is order-free. The order dependence is in the SNP pileup itself (`pileup_aligned_bases`, pileup_engine.h ~2511): for each (barcode, SNP, UMI) the **first read seen** decides the allele: `umi_new` is true once per molecule, DP is incremented, and AD is incremented only if *that* read's base equals ALT. When two reads of one molecule disagree at the SNP (sequencing or PCR error), which one arrives first decides whether the molecule counts as ALT. DP is unaffected (`snp_dp.1pz` is identical everywhere), AD is not, and the donor VCF and coverage are computed from AD. `snp_hits` is identical because it counts every overlap before dedup.

Why this shows up across thread counts but not across same-T reruns: STAR's unsorted output interleaves whole per-thread chunks, so the relative order of two distant reads of the same molecule is set mostly by the chunking (a function of T) and only rarely by timing; changing T re-chunks everything. The sorted path is immune only because STAR's sort breaks coordinate ties by read index, which fixes the "first" read: a reproducible rule, not a principled one. The same rule also makes sorted-vs-stream AD differ.

Canonical fix (adds to the list in (c)): decide a molecule's allele by majority base over its reads (ties to REF, or to the lexicographically smaller read name), i.e. buffer per-(barcode, SNP, UMI) base counts and emit at finalize, as the molecule store already does for gene labels. Order-free, and arguably more accurate than first-read-wins. Memory: one small counter per covered molecule-SNP pair.

Updated criterion (b) status: every count tally and the exon/intron/gene matrices are order-independent today; the known order-dependent rules are (1) junction feature numbering (bytes only), (2) positional 1/1000 rRNA sampling, (3) first-10,000-reads strand probe, (4) first-two-genes-seen EM-rescue candidates, (5) first-read-wins SNP allele, (6) barcode candidate list from a BAM scan vs none. All six have a canonical replacement above, and the shuffled-BAM acceptance test in (c) proves each one closed.

Engine task for W2 with the six canonical rules and the shuffled-BAM acceptance test: `w2-task-order-independent-pileup.md`.

## Update 2026-10-28 (e): unsorted streaming is the default (Zach's decision)

Zach, 2026-10-28: unsorted BAMs are the go-to; sorting is not essential for STAR. Implemented on `perf/bam-stream` (draft PR #63, commit `b3099e4`): the pipeline streams STAR's unsorted BAM into the serial pileup by default and falls back to the sorted path automatically where a mode still reads the sorted file. `--sorted-bam` forces the old path; `--stream-unsorted` is kept and now only reports why it was ignored.

| case | path | why |
|---|---|---|
| scRNA, `CB_UMI_Simple` (10x 3'/5', Drop-seq, most of the atlas) | **stream (default)** | pileup counts are order-independent (measured, update (c)) |
| existing `--bam-file` input | sorted reader as given | no STAR run |
| `--parallel-pileup` given | sorted | explicit request |
| ATAC | sorted | `ATACFragmentExtractor` reads coordinate-sorted records |
| SMART-seq2 / bulk | sorted | plate/bulk counting reads the sorted BAM |
| `CB_UMI_Complex`, `CB_samTagOut` (inDrops, SPLiT-seq, BD Rhapsody, ...) | sorted | STAR writes corrected CB only into a sorted BAM; the CR-as-CB rule is proven only for whitelist-exact `.1fq` barcodes of CB_UMI_Simple |
| `--tagged-bam` | sorted | rewrites the sorted BAM |
| `SINGLET_KEEP_STAR_BAM=1` | sorted | keeps the BAM for aligner equivalence checks |

What "sorting is not essential" does and does not cover: STAR aligns and STARsolo counts identically either way; what needs `SortedByCoordinate` in STAR is the `CB`/`UB` SAM attributes (`ParametersSolo.cpp`: "CB and/or UB attributes ... can only be output in the sorted BAM file") and the bedGraph signal output. That is why the stream derives CB from CR. The pileup's own requirement for sorting was only the parallel region split and the indexed strand probe; the serial pileup needs neither.

Not yet sort-free, follow-ups: ATAC fragments, SMART-seq2/bulk counting, and the complex barcode protocols (need a CB-correction replica or CB from the `.1fq` pass). Until the six canonical rules land (`w2-task-order-independent-pileup.md`), streaming output is byte-reproducible only at a fixed thread count, and its SNP AD, junction byte order, rRNA sample and auto barcode set differ from Alpha's sorted outputs (sizes in updates (b)-(d)). That is the accepted cost of the default today; the acceptance test in the W2 task is what makes it identical.

GPU (W11, PR #61): the GPU pileup prototype does not read a sorted BAM. It sorts its own records on the device (sort + merge-join + segmented reduce) and was written to reproduce the CPU engine's sorted-BAM tie rules ("the GPU uses the BAM-order rule", `bench/gpu/pileup/README.md`). So the STAR coordinate sort is not on the GPU path at all; the saving there is on the CPU baseline it is compared against (the sort is part of the ~85% "STAR, sort, pileup and CB refresh" in `bench/gpu/cost/README.md`), which makes the GPU cost case slightly harder, not easier. The canonical rules matter for W11 too: once the CPU engine is order-free, the GPU can drop the BAM-order emulation and match by construction.
