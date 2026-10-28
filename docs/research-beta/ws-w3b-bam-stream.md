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
