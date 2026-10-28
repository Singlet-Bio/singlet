# W3b: does retiring the coordinate-sorted BAM cut atlas compute? (phases 1, 2, 4, 5 done; phase 3 partly)

Protocol: [bam-stream-protocol.md](bam-stream-protocol.md). Scripts and the raw per-run record: `bench/bam-stream/` (`results.json`).
Draft PR Singlet-Bio/singlet#60 (this report) and branch `perf/bam-stream` (one debug env var, no change in default behaviour).

## Result in one paragraph

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

## Phase 6: decision

| criterion | state |
|---|---|
| (a) measured saving at least 15% of atlas core-hours | **bounded, not measured**: medians 19-39%; the lowest per-profile minimum is 12.6% |
| (b) identical outputs, or every difference accepted | **untested**; the largest identified risk (barcode correction) is shown to be absent |
| (c) peak memory fits 16 jobs per 512 GB node | **unknown** for streaming; the sorted path's RSS is 53-74 GB |

The three criteria cannot yet all be shown to hold, so this is **not yet a "proceed"**. The saving bound is large enough, and the main
identity risk small enough, to justify finishing the prototype (steps 1-6 above) before the Beta Software Freeze. If it misses, the best
cheap fallback is the external sort at about 6-7%. Memory is the open item for the packing plan: the sorted path already needs 53-74 GB
RSS, and any in-memory per-position tally comes on top.

## Housekeeping

- The staged `.1fq` files under `$WORK/data/bam-stream/` (about 4.4 GB) are deleted once this report is reviewed.
- Clipper cost: roughly 120 core-hours of shared-node baseline plus about 20 node-hours exclusive, all at nice 10000 and `%2` or less.
- Jobs are named `bs-*`; scripts are in `bench/bam-stream/`.
