# W3b: does retiring the coordinate-sorted BAM cut atlas compute? (interim, 2026-10-27)

Protocol: [bam-stream-protocol.md](bam-stream-protocol.md). Scripts and raw per-run JSON: `bench/bam-stream/`. Status: **phase 1 under way, phase 2
bound computed on 4-thread data, the 15% stop rule did not fire.** Phases 3-6 not started.

## Binary and environment

- Base build: `bin/singlet-campaign` on Clipper, embeds commit `02aa9b938b54e0f92bba00c44c07b883a8660aa2` (`prod/campaign` head),
  sha256 `261d3a50ed624572418d30489dfd58904323f6e0d8820bd5260dfe7fb5038e6c`. #41 (FASTASSIGN) was not merged into the production
  stack while this ran; no re-baseline.
- Runs: Clipper `short-bigmem` (b001), 4 threads (16 follow), 80 GB, `--bam-sort-mode disk`, private genome, `--nice=10000`, `%4`.
  The four concurrent jobs of one array shared a node, so each run's co-tenant CPUs (32-44) and load are recorded. Timings
  carry that noise: replicate sort times differ up to 2.2x (GSM7036403: 383 s vs 859 s).
- `memory.peak` reads 81,920 MB (the cgroup cap) on the large runs: it counts page cache, so it is not an RSS; `maxrss` is 51-58 GB.

## Datasets actually measured (real public data)

| id | what | reads (M) | note |
|---|---|---|---|
| GSM4006843 | human 3' v3, 121 bp | 66.7 | stand-in for the 66 M PBMC reference point (not the same sample) |
| GSM5411066 | mouse 3' v3 | 69.6 | GRCm39-2024-A |
| GSM7036403 | human 3' v3 deep | 111.6 (128.4 M aligned incl. secondary) | |
| GSM7901264 | human 5' | 100.3 | the catalog says 10xv3_5prime; the header probe reports 10xv2 |
| GSM6470208 | two runs, 3' v3 | 121.8 | #47 path via the concatenated `.1fq` |
| GSM7656149 | human 3' v3 very deep | ~260 | |

Not measured yet: the Kang genotype pool and a high-mtDNA sample (needed for phase 5 truth), the 122 M 5' sample (a 100 M one
stands in). GSM6250705 was dropped: its staged `.1fq` is corrupt (STAR: "quality string length is not equal to sequence length"
at read 30.5 M).

## Phase 1 (4 threads, replicates 1-2 so far; seconds)

| set | wall | STAR map | solo | **sort** | index | CB refresh | pileup | sort+idx+refresh share of wall |
|---|---|---|---|---|---|---|---|---|
| GSM4006843 | 1293 / 1231 | 571 / 575 | 52 | 379 / 317 | 5.5 | 24 | 65 | 31% / 28% |
| GSM5411066 | 875 / 917 | 314 / 337 | 46 | 287 / 303 | 6 | 20 | 49-51 | 36% / 36% |
| GSM7036403 | 1637 / 2173 | 715 / 804 | 72 / 83 | 383 / 859 | 10 | 33 | 104 / 83 | 26% / 41% |
| GSM7901264 (5') | 1601 / 2001 | 828 / 913 | 78 / 61 | 397 / 736 | 9 | 25 | 53 | 27% / 38% |
| GSM6470208 (2 runs) | 4199 / 4320 | 1058 / 1276 | 116 / 176 | 2274 / 2290 | 150 / 118 | 59 / 67 | 197 / 169 | 59% / 57% |
| GSM7656149 (260 M) | 5626 (r1) | 3038 | 225 | 1382 | 25 | 102 | 200 | 26% |

The sort is 23-54% of wall at 4 threads; the earlier "49% at 110 M" was a shared-node case, and an idle-node replicate of the
same sample shows 23%. Peak job `/tmp`: 15-85 GB (the two-run sample 85 GB).

## Phase 2: the upper bound

Streaming pileup cost (existing `--no-parallel-pileup` path, `--bam-file`, job `bs-p2`): 60.85 M reads in **108 s serial** (1.78 s
per M reads; ~1.5 s per M of total reads, since the BAM used held only the barcoded 82%), against 65 s for the parallel pileup
of the same sample. Because `--bam-file` has no way to keep STAR's own sorted BAM (the pipeline deletes it), the input was the
`--tagged-bam` output, which holds only the barcoded reads; the cost is scaled by read count.

Atlas weighting (`atlas_bound.py`): 56,606 single-cell samples (10x v2/v3/5'/suspect, `samples.parquet`), median 207 M reads, total
15.8 G reads; wall = 130 s fixed + linear in reads per dataset profile. Saving share of atlas core-hours (4 threads), over
the six measured profiles:

| scenario | min | median | max |
|---|---|---|---|
| B: streaming pileup scales like today's parallel pileup | 26.8% | 35.5% | 57.9% |
| A: streaming pileup stays serial (measured cost) | 22.7% | 29.9% | 50.8% |

**Both scenarios are well above the 15% threshold, so the phase 2 stop rule does not fire.** Caveats: the profiles come from
5-7 samples and the atlas median (207 M) is above all but one of them, so the sort share is extrapolated linearly in depth;
replicate 3 and the 16-thread runs are pending; no streaming prototype exists, so this is an upper bound that also ignores
the cost of unsorted-BAM transport and the barcode-correction replica.

## Next

Phase 3 (the `1MM_multi_Nbase_pseudocounts` replica, tested on the barcode table first) needs W2's agreement before touching
`pileup_engine.h`; the first test needs a BAM with raw CR/UR and STAR's pass-1 counts, which the campaign binary does not keep.
