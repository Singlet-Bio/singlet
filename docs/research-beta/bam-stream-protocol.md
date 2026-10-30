# Protocol: can streaming alignments (no sorted BAM) substantially accelerate the engine?

Written 2026-10-27 for one Sonnet 5.5 session (W3b). It answers one question with measurements on real data: does replacing
STAR's coordinate-sorted BAM with a stream of unsorted alignments into the pileup cut the atlas cost substantially, with
identical outputs? The background is in [rollout-plan.md](rollout-plan.md) §3b and [audit-2026-10-27.md](audit-2026-10-27.md)
§3.

## Rules

- **Clipper:** one bridge (never start it; if it is down, stop and report). Never run `sync.sh`.
- **Alpha first:** jobs run with `--nice=10000` and arrays of `%4` or less, stay off partitions where Alpha has pending
  jobs, and stop if an Alpha job is waiting (rollout-plan §9).
- **No Anvil work.**
- **Real data only.**
- **Compute estimate:** before each batch, state its node-hours.
- **Keep nothing:** BAMs and reads stay in job `/tmp` and are deleted. Long-term storage is limited to ~10 TB on Clipper
  for everything.
- **Files:** `bench/bam-stream/` (scripts, results JSON) and this note's results section. Engine changes go in a draft PR
  that W2 (owner of `pileup_engine.h`) reviews.
- **Base build:** the pinned campaign binary plus whatever the FASTASSIGN session merges. Record the commit and sha256 of
  every binary measured.

## Datasets (real, public)

| id | why |
|---|---|
| 3′ v3 PBMC, ~66 M pairs | the reference point (845 s idle-node profile exists) |
| 3′ deep, ~110 M pairs | where the sort was 49% of wall |
| 3′ very deep, ~260 M pairs | depth scaling |
| 5′, ~122 M pairs | where the sort was 60% of wall |
| mouse 3′ | strand and NUMT paths |
| multi-run sample | #47 concatenation path |
| Kang genotype pool | demux, doublets and SNP pileup have real truth |
| one high-mtDNA-coverage sample | mtDNA pileup |

## Phase 1: baseline (current sorted path)

For each dataset, at 4 and 16 threads, with 3 replicates:
- **Stage timers:**
  - mapping;
  - unsorted bins;
  - the sort;
  - the index;
  - the CB-tag refresh;
  - the pileup passes;
  - the strand re-run;
  - export.
- **Resources:**
  - peak memory (cgroup `memory.peak`);
  - CPU seconds;
  - bytes read and written (`/proc/<pid>/io`);
  - peak `/tmp` usage.
- **Node load:** the node type, and whether the node was shared. Co-tenant I/O on `/tmp` inflated the sort before, so
  record `sacct` and the node's load average.

## Phase 2: the upper bound, before writing any engine code

Estimate the most that streaming could save: the sort, the index, the CB-tag refresh, and the extra BAM passes, minus a
measured streaming pileup cost.
- **Streaming cost:** run the existing `--no-parallel-pileup` streaming `run()` on a BAM.
- **Atlas weighting:** weight the result by the atlas depth distribution. Use per-sample read pairs from the planner's
  ENA `sra_bytes`, bucketed by chemistry.
- **Report:** the atlas core-hours saved, as a share of the refitted total.
- **Stop rule:** if the upper bound is **under 15%** of atlas core-hours, stop here and recommend phase 4's cheapest
  alternative.

## Phase 3: a correct streaming prototype

Only if phase 2 passes. Timebox: 2 weeks from the start of phase 3.
1. **STAR output:** `--outSAMtype BAM Unsorted` (or SAM on a pipe), with the raw `CR`/`UR` tags and CellReads.stats.
2. **Barcode correction:** apply STAR's `1MM_multi_Nbase_pseudocounts` correction in the pileup, from the pass-1
   exact-match counts. This is the identity risk. Test it first, on the barcode table alone, against the sorted path's
   `CB`/`UB`.
3. **Strand:** take it from the header and the run screen; drop the strand re-run.
4. **Order-independent accumulation:**
   - every caller (counts, junctions, introns, SNP, mtDNA, demux, doublet, ambient) gives the same result in any read
     order;
   - per-position tallies use hash maps or chrM arrays;
   - measure peak memory, including whole-genome intron and SNP tallies.
5. **Debug path:** keep the sorted path behind `--keep-sorted-bam`.

## Phase 4: cheaper alternatives, measured on the same datasets

1. **In-memory sort** where node memory allows (`--bam-sort-mode memory` with a measured buffer).
2. **External sort:** STAR's unsorted output sorted by a multithreaded external sort on the node's NVMe `/tmp`.
3. **Skip the BAM** for samples whose public layers need no positions (counts only), if the Beta policy has any such
   class.

## Phase 5: identity and determinism

- **Identity:** compare every pipeline output, byte for byte, between the sorted path and each candidate, using
  `perf_equiv`. Classify each difference by size and cause.
- **Determinism:** 4, 8 and 16 threads, shuffled run order, and two reruns must give identical outputs.

## Phase 6: decision

**Proceed** to the Beta engine only if all three hold:
- (a) the measured atlas saving is **at least 15%** of core-hours;
- (b) outputs are identical, or every difference has been benchmarked and accepted by W7;
- (c) peak memory still fits the packing plan (16 jobs per 512 GB node with the shared genome).

**Otherwise**, recommend the best phase 4 option.

**Report:**
- a table of wall time, CPU time and peak memory per dataset and path;
- the atlas projection;
- the identity summary;
- the recommendation.

Put it in `docs/research-beta/ws-w3b-bam-stream.md` and `bench/bam-stream/results.json`.

## Results

(filled in by the session)
