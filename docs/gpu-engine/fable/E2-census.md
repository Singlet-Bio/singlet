# E2 census: per-read state and cost of the GPU mapping lanes

Wave 1, experiment E2 of the GPU aligner program (`singlet-agents/docs/GPU_ALIGNER_STUDY_2026-10-31.md`, Part II.4;
questions Q2 and Q7). Branch `gpu/fable/census` (from `gpu/fable-engine` cf92cdb9). Instrumentation only: the mapping
results are byte-identical with and without it (section 1).

**Summary.** Per-read state and cost were recorded for about 2 M distinct reads per sample (pbmc1k_v3, pbmc5p_v1 and nuclei2k_v2), in every lane each read
ran in. The mapping stays byte-identical with the counters on. The main findings:

- **Window count after phase 1 sets the cost.** Reads with more than 16 windows are 25.6% (24.1%) of distinct reads
  and take 72.6% (70.1%) of all thread cycles and 79% (76%) of recursion steps. In nuclei2k_v2 they are 37.5% of
  reads and take 80% of cycles. Ranking reads by `nWall` alone captures
  as many cycles as any fitted predictor.
- **Warps are badly mixed.** With reads in today's order, k_p3 in lane A uses at most 15% of the warp's step slots.
  Sorting each batch by `nWall` (known before k_p3) raises this to 49% (36%) for steps and 99.9% for windows. This
  is the cheapest exact change found.
- **The cascade spends 18–21% of its thread cycles in lanes a read overflows.** Lanes M/B redo phase 1. Reads that
  overflowed in phase 1 sit idle in k_p3 warps.
- **Phase 1 is 30–32% of all thread cycles.** k_p1 takes 40% of lane A's time. The seed search alone is 37.5% of
  lane A's thread cycles.
- **Unmapped "too short" reads are a large cost.** They are 7% of reads but 19.5% of cycles in pbmc1k_v3, and 23% of
  reads but 54.5% of cycles in pbmc5p_v1. In nuclei2k_v2 they are 23% of reads and 47% of cycles.

Numbers are pbmc1k_v3, with pbmc5p_v1 in parentheses where given. Section 11 compares all three samples. The full tables are in
`bench/fable/census/results/census_<sample>.md`; histograms and fits are in the matching `.json`.

## 1. Identity and overhead

| sample | build | distinct identical | reads identical | GPU ms | lane A / M / B ms | k_p1 / k_p2 / k_p3 lane A ms |
|---|---|---|---|---|---|---|
| pbmc1k_v3 | m5 (default) | 1,991,926 / 1,991,926 | 4,118,999 / 4,118,999 | 861.1 | 269.2 / 336.6 / 255.3 | 106.6 / 16.8 / 145.8 |
| pbmc1k_v3 | m5c (census) | 1,991,926 / 1,991,926 | 4,118,999 / 4,118,999 | 899.6 | 284.8 / 357.6 / 257.2 | 107.2 / 17.1 / 160.4 |
| pbmc5p_v1 | m5 (default) | 1,850,645 / 1,850,654 | 6,118,905 / 6,118,914 | 917.2 | 333.4 / 283.2 / 300.6 | 101.6 / 13.0 / 218.7 |
| pbmc5p_v1 | m5c (census) | 1,850,645 / 1,850,654 | 6,118,905 / 6,118,914 | 957.3 | 347.7 / 303.9 / 305.7 | 102.6 / 13.2 / 231.8 |

- **Identity.** Both builds give the same results as STAR. In pbmc5p_v1, 9 distinct reads differ, and they differ the
  same way in both builds. They exceed `LREAD` in every lane, including the host lane (`heavy_failures` 9,
  `lread` 9). This is a limit of the harness that predates the census, not an effect of it.
- **Timing.** The default build of this branch runs in 861.1 ms, against 860.7 ms in the reference job 795242. The
  census build costs +4.5% (+4.4%) in total: +6% on lanes A and M (k_p3) and +1–2% on lane B.

## 2. Lane flow and where the thread cycles go

| lane | reads in | finished | overflow reasons (OvWhy) | thread Gcycles |
|---|---|---|---|---|
| A | 1,991,926 (100%) | 1,467,349 | win 510,638; keep 10,574; wa 2,792; wtr 546; steps 26; winbin 1 | 26,555 |
| M | 524,577 (26.3%) | 433,837 | win 75,712; wa 13,739; keep 1,246; wtr 43 | 36,588 |
| B | 90,740 (4.6%) | 85,436 | win 2,622; steps 2,223; keep 316; wtr 143 | 13,236 |
| host | 5,304 (0.27%) | 5,304 | | |
| pbmc5p_v1 A / M / B / host | 1,850,654 / 470,063 / 114,516 / 9,747 | 1,380,591 / 355,547 / 104,769 / 9,747 | A: win 451,294, wa 10,928, keep 6,631, wtr 1,115; M: win 64,210, **wa 49,520**; B: **steps 7,846**, win 1,562 | 26,696 / 31,365 / 17,111 |

**Where the cycles go.** Of all thread cycles (k_p1 + k_p3 of every lane a read went through):

| share of all thread cycles | pbmc1k_v3 | pbmc5p_v1 |
|---|---|---|
| stitching | 48.4% | 52.0% |
| phase 1 | 31.9% | 29.6% |
| assignment | 9.5% | 10.0% |
| lanes the read overflowed | 18.3% | 20.6% |

**By lane and phase.** Shares of each lane's k_p1 + k_p3 thread cycles (pbmc5p_v1 in parentheses):

| lane | k_p1 | seeds | windows + flanks | k_p3 | assignment | stitching | select + encode |
|---|---|---|---|---|---|---|---|
| A | 46.6% (42.8%) | 37.5% (34.5%) | 6.0% (5.5%) | 53.4% (57.2%) | 7.6% (7.1%) | 25.5% (34.7%) | 0.6% (0.6%) |
| M | 25.4% (25.8%) | 11.4% (13.1%) | 10.2% (9.2%) | 74.6% (74.2%) | 10.7% (10.9%) | 41.2% (38.2%) | 0.1% (0.1%) |
| B | 20.3% (16.0%) | 5.0% (5.0%) | 13.1% (9.2%) | 79.7% (84.0%) | 9.9% (12.8%) | 57.3% (55.4%) | 0.0% (0.0%) |

**What thread cycles measure.** Thread cycles behave like a step function of the lane: the mean per read is 13.4 M in
A, 69.9 M in M and 145.5 M in B. A thread's elapsed time is set by its warp and its SM, not by its own work (Method,
caveat). So the cycle shares below say where today's lane-slot time goes. The step counts give the intrinsic work.

## 3. Per-read distributions (final-lane record = the read's full state; all distinct reads)

| quantity | p50 | p90 | p99 | p99.9 | max | p99 in A / M / B / host |
|---|---|---|---|---|---|---|
| seeds `nP` | 6 | 15 | 22 | 26 | 31 | 15 / 24 / 26 / 27 |
| windows `nWall` | 3 | 42 | 91 | 133 | 238 | 16 / 63 / 123 / 194 |
| aligns over windows `nWAsum` | 4 | 52 | 135 | 250 | 1,180 | 24 / 116 / 233 / 546 |
| largest window `nWAmax` | 1 | 4 | 8 | 50 | 50 | 6 / 7 / 50 / 49 |
| SA entries `nEntries` | 10 | 4,286 | 12,072 | 19,218 | 43,279 | 9,731 / 13,761 / 17,816 / 25,456 |
| entries hitting a window `nHits` | 7 | 62 | 171 | 350 | 8,560 | 32 / 144 / 332 / 869 |
| recursion steps `nSteps` | 24 | 285 | 864 | 2,276 | 87,282 | 162 / 765 / 1,772 / 21,877 |
| steps of the largest window | 5 | 29 | 109 | 649 | 46,403 | 65 / 89 / 649 / 10,255 |
| transcripts kept `keepMax` | 1 | 5 | 14 | 38 | 146 | 6 / 18 / 44 / 88 |
| transcripts per window `winTrMax` | 1 | 2 | 5 | 13 | 100 | 3 / 5 / 14 / 45 |
| transcripts `nTr` | 1 | 1 | 6 | 28 | 146 | 3 / 10 / 38 / 87 |
| exons `maxEx` | 0 | 1 | 2 | 3 | 5 | 2 / 2 / 3 / 3 |
| cascade Mcycles | 14.5 | 93.5 | 265 | 306 | 359 | 25.5 / 125.9 / 315.5 / – |

In pbmc5p_v1 the shape is the same. The step tail is heavier: `nSteps` has p99 1,151, p99.9 7,286 and max 84,849.
Log2 histograms per lane are in the JSON (`distributions.<q>.hist_<lane>`).

## 4. Shares by window count and by largest window

| `nWall` | reads | reads × multiplicity | cascade cycles | stitching cycles | steps | pbmc5p_v1 reads / cycles / steps |
|---|---|---|---|---|---|---|
| 1 | 40.2% | 47.6% | 13.5% | 10.0% | 2.6% | 49.2% / 16.8% / 4.7% |
| 2-4 | 15.0% | 20.3% | 5.4% | 2.7% | 3.5% | 12.1% / 4.9% / 3.9% |
| 5-16 | 19.1% | 15.6% | 8.5% | 6.8% | 14.5% | 14.3% / 8.2% / 15.3% |
| 17-64 | 21.8% | 13.6% | 49.6% | 50.5% | 55.4% | 20.9% / 49.2% / 55.4% |
| 65-128 | 3.7% | 2.0% | 22.2% | 29.0% | 22.4% | 3.4% / 20.4% / 20.0% |
| >128 | 0.13% | 0.07% | 0.8% | 1.1% | 1.5% | 0.08% / 0.5% / 0.8% |

(0 windows: 0.06% of reads.)

| largest window (aligns) | reads | cascade cycles | stitching cycles | steps |
|---|---|---|---|---|
| 1 | 54.6% | 28.1% | 23.3% | 13.5% |
| 2 | 22.9% | 29.6% | 30.8% | 25.4% |
| 3-4 | 16.3% | 25.3% | 26.9% | 28.4% |
| 5-8 | 5.3% | 12.2% | 14.1% | 21.9% |
| 9-16 | 0.67% | 3.4% | 3.7% | 5.7% |
| 17-32 | 0.13% | 0.8% | 0.8% | 2.1% |
| 33-50 | 0.15% | 0.5% | 0.5% | 3.1% |

**Joint behaviour of cost and read state** (GPU-finished reads; Pearson correlation of log x with log cascade cycles):

| x | correlation |
|---|---|
| `nWall` | 0.85 |
| `nHits` | 0.85 |
| `nWAsum` | 0.84 |
| `nSteps` | 0.81 |
| `nP` | 0.65 |
| `nEntries` | 0.58 |
| `maxNrep` | 0.57 |

**Median cascade cycles by `nWall` bucket** (a step at 16, the lane-A cap):

| `nWall` | median Mcycles |
|---|---|
| 1 | 12.3 |
| 8-15 | 14.2 |
| 16-31 | 82.0 |
| 32-63 | 83.5 |
| 64-127 | 235 |

The full joint tables (cycles vs `nWall`, `nEntries`, `nSteps`, with medians, means and p99 per bucket) are under
`joint` in the JSON.

## 5. Windows

- **Count and steps per window.** There are 26.6 M stitched windows (22.3 M in pbmc5p_v1). Steps per window: p50 5,
  p90 11, p99 41, p99.9 127, max 46,403 (pbmc5p_v1: 5 / 13 / 71 / 479 / 53,223).
- **Single-align windows dominate.** 81% of windows have one align, and they always take exactly 5 steps; together
  they hold 53% of all steps. Steps grow with the window's aligns: p50 11 / 19 / 29 / 89 / 461 / 2,653 for 2 / 3 / 4 /
  8 / 20 / 50 aligns.
- **Fixed cost per window.** A window's elapsed cycles barely depend on its steps. The mean is 0.91 M cycles with one
  align and 1.09 M with two, so a fixed per-window part (finalization, extension, scoring, recording) dominates.
- **Recursion vs a chaining DP.** The recursion takes 2.04e8 steps over all windows. A chaining DP would evaluate
  5.04e7 pairs (sum of nWA(nWA+1)/2), a ratio of 4.05 (3.59 in pbmc5p_v1). The excess sits in the few windows with
  many aligns.
- **Is a read dominated by one window?** Of the reads with at least 2 stitched windows, the largest window holds more
  than 50% of the read's steps in 11.2% (17.6%), and more than 90% in 0.25% (0.89%).
  - Over all reads, the largest windows hold 16.1% (24.2%) of all steps. This is the critical path of a fully
    window-parallel lane.
  - For lane-M reads the share is 8.6%, and for lane-B reads 9.8%.

## 6. Splits by transcripts, multiplicity and STAR outcome

| class | reads | reads × mult. | cascade cycles | finish A / M / B / host (%) | median `nWall` | pbmc5p_v1 reads / cycles |
|---|---|---|---|---|---|---|
| nTr = 1 | 91.6% | 90.1% | 82.2% | 76.9 / 19.7 / 3.3 / 0.2 | 2 | 82.1% / 60.6% |
| nTr 2-10 | 7.9% | 8.9% | 16.0% | 40.8 / 44.8 / 13.3 / 1.0 | 22 | 16.9% / 36.5% |
| nTr > 10 | 0.39% | 0.32% | 1.6% | 0 / 53.4 / 37.5 / 9.1 | 41 | 0.46% / 2.1% |
| singleton | 80.0% | 38.7% | 87.1% | 70.6 / 24.2 / 4.9 / 0.3 | 4 | 81.5% / 91.6% |
| 2-9 copies | 18.8% | 25.6% | 12.4% | 85.6 / 12.5 / 1.7 / 0.1 | 1 | 16.3% / 7.5% |
| >= 10 copies | 1.1% | 35.7% | 0.6% | 92.8 / 6.4 / 0.7 / 0.1 | 1 | 2.2% / 1.0% |
| mapped | 92.4% | 95.1% | 79.1% | 78.7 / 18.2 / 2.9 / 0.2 | 2 | 76.1% / 43.7% |
| unmapped: too short (1) | 7.2% | 3.9% | 19.5% | 12.6 / 67.1 / 19.2 / 1.1 | 37 | **23.2% / 54.5%** |
| unmapped: other (0) | 0.16% | 0.76% | 0.26% | 35.2 / 0 / 64.8 / 0 | 7 | 0.57% / 0.78% |
| unmapped: multimapping (3) | 0.28% | 0.26% | 1.2% | 0 / 51.5 / 38.0 / 10.5 | 34 | 0.19% / 1.1% |

- **Multimappers.** Reads with nTr > 1 are 8.3% (17.3%) of reads and take 17.6% (38.6%) of cycles.
- **"Too short" reads.** These are reads STAR rejects as too short after stitching. They have many windows (median 37;
  28 in pbmc5p_v1). In pbmc5p_v1 they take more than half of all GPU thread cycles.

## 7. Cost predictor and two-class router

**Fit.** The model is fitted on half of the GPU-finished reads and scored on the other half. The feature sets are:

- `phase1`: what k_p1 knows (`nP`, `nSplit`, `maxNrep`, `sumNrep`, seeds with Nrep > 1 and > 50, `nWall`, `nEntries`,
  `Lread`, `wbN`).
- `+scan`: adds k_p2's `nHits`.
- `+assign`: adds `nWAsum`, `nWAmax` and `nWAne` after the assignment.

The tree is a histogram regression tree of depth 6.

| target | features | OLS R² | log-log R² | tree R² | pbmc5p_v1 tree R² |
|---|---|---|---|---|---|
| cascade cycles | phase1 | 0.779 | 0.743 | 0.866 | 0.781 |
| cascade cycles | +scan | 0.783 | 0.760 | 0.877 | 0.818 |
| cascade cycles | +scan +assign | 0.793 | 0.761 | 0.890 | 0.874 |
| steps | phase1 | 0.832 | 0.953 | 0.894 | 0.775 (log-log 0.864) |
| steps | +scan | 0.870 | 0.962 | 0.919 | 0.857 (0.881) |
| steps | +scan +assign | 0.966 | 0.989 | 0.952 | 0.929 (0.970) |

**Routing.** The router sends the predicted top share of reads to the wide class. A wide share of 30% / 20% / 10% means
routing at the 70th / 80th / 90th percentile.

| router | 30% wide | 20% wide | 10% wide |
|---|---|---|---|
| oracle (true cycles) | 76.2% | 63.7% | 41.7% |
| log-log, phase 1 | 74.6% | 59.6% | 37.3% |
| log-log, phase 1 + scan | 74.7% | 59.9% | 37.7% |
| rank by `nWall` | 74.7% | 59.7% | 37.4% |
| rank by `nHits` | 74.9% | 59.7% | 38.3% |
| rank by `nEntries` | 58.0% | 36.8% | 19.6% |
| pbmc5p_v1: oracle / log-log / `nWall` | 76.4 / 73.4 / 73.0% | 65.7 / 61.1 / 60.7% | 44.8 / 37.7 / 37.1% |

- **Today's split.** The rule `nWall > 16` (today's lane A cap) sends 25.5% (24.1%) of reads, carrying 72.3% (69.7%)
  of cycles.
- **Two-class split at the 70th percentile.** Ranking by `nWall` sends 99.1% of today's lane-A overflow reads wide, and
  the narrow class then has p99 26 Mcycles.
- **A fitted model adds nothing.** It does no better than `nWall`, because the cycles mostly encode the lane, and the
  lane encodes `nWall`. For intrinsic work (steps), phase-1 state predicts well: log-log R² is 0.95 and Spearman 0.98.

## 8. Capacities

**Smallest cap covering a share of distinct reads**, all reads (pbmc5p_v1 99.9% in parentheses):

| cap | 99% | 99.9% | 99.99% | max | today A / M / B |
|---|---|---|---|---|---|
| NSPLIT | 1 | 2 | 3 | 4 | 4 / 4 / 8 |
| NP | 22 | 26 (26) | 28 | 31 | 32 / 32 / 128 |
| NW | 91 | 133 (126) | 173 | 238 | 16 / 64 / 128 |
| NWA | 8 | 50 (50) | 50 | 50 | 16 / 8 / 50 |
| NEX | 3 | 4 (4) | 4 | 6 | 8 / 8 / 20 |
| NWTR | 5 | 13 (21) | 34 | 100 | 8 / 8 / 32 |
| NKEEP | 14 | 38 (37) | 74 | 146 | 8 / 32 / 64 |
| NWB | 1,105 | 1,619 (1,525) | 2,103 | 2,927 | 256 / 1,024 / 4,096 |
| NBLOB | 323 | 1,310 (1,323) | 3,333 | 8,476 | 640 / 2,048 / 4,096 |
| steps | 864 | 2,276 (7,286) | 8,818 | 87,282 | 1,024 / 2,048 / 2,048 |

**Predicted overflow of each cap set**, where a read overflows if any cap is exceeded:

| cap set | overflow (distinct) | overflow (× multiplicity) | cycle share of the overflow |
|---|---|---|---|
| 99% caps | 4.87% (6.12%) | 3.28% (5.32%) | 19.2% (22.4%) |
| 99.9% caps | 0.43% (0.47%) | 0.31% (1.35%) | 2.4% (2.5%) |
| 99.99% caps | 0.053% (0.047%) | 0.067% (0.022%) | 0.3% (0.3%) |
| today's lane A | 26.3% (25.4%) | | 74.3% (73.6%) |

**By class: narrow (`nWall` <= 16) vs wide (> 16).** Each cell gives 99.9% / 99.99% / max; pbmc5p_v1 99.9% in
parentheses.

| cap | narrow | wide |
|---|---|---|
| NP | 19 (19) / 21 / 25 | 27 (27) / 29 / 31 |
| NW | 16 / 16 / 16 | 158 (147) / 195 / 238 |
| NWA | 50 (50) / 50 / 50, p99 6 (12) | 50 / 50 / 50, p99 13 (50) |
| aligns pooled over windows (`NWAsum`) | 51 (88) / 119 / 308 | 342 (462) / 552 / 1,180 |
| NEX | 4 (4) / 4 / 6 | 4 (4) / 5 / 6 |
| NWTR | 8 (15) / 29 / 100 | 18 (29) / 41 / 72 |
| NKEEP | 12 (15) / 40 / 110 | 57 (51) / 88 / 146 |
| NWB | 195 (195) / 207 / 281 | 1,913 (1,777) / 2,367 / 2,927 |
| NBLOB | 523 (640) / 2,053 / 6,328 | 2,452 (2,127) / 4,164 / 8,476 |
| steps | 550 (2,642) / 6,031 / 84,612 | 3,570 (11,898) / 11,891 / 87,282 |

**Wide-class shape** (reads with more than 16 windows):

- Windows with at least one align: p50 36, p99 117.
- Aligns over all windows: p50 44, p99 198, max 1,180.
- The largest window has 0-1 aligns in 19% of these reads, 2-4 aligns in 64%, 5-8 aligns in 15% and more than 8 in
  2.6%. The grid is `wide_shape` in the JSON.

## 9. Overflow reasons per lane and the phase that detects them

Phase 1 is `k_p1`: seeds, windows, flanks, with OvWhy lread / split / pc / win / winbin. Phase 3 is `k_p3`: the
assignment (wa), and stitching (steps / wtr / keep / exons). The wasted Gcycles are thread cycles of the overflowing
lane: k_p1 / k_p3.

| lane | overflow reads | share of lane | phase 1 | phase 3 assignment | phase 3 stitching | wasted Gcycles p1 / p3 |
|---|---|---|---|---|---|---|
| A | 524,577 | 26.3% | 510,639 (97.3%) | 2,792 | 11,146 | 3,393 / 3,495 |
| M | 90,740 | 17.3% | 75,712 (83.4%) | 13,739 | 1,289 | 1,622 / 4,634 |
| B | 5,304 | 5.9% | 2,622 | 0 | 2,682 | 151 / 654 |
| pbmc5p_v1 A | 470,063 | 25.4% | 451,305 (96.0%) | 10,928 | 7,830 | 3,450 / 3,392 |
| pbmc5p_v1 M | 114,516 | 24.4% | 64,219 (56.1%) | 49,520 | 777 | 1,982 / 5,031 |
| pbmc5p_v1 B | 9,747 | 8.5% | 1,571 | 0 | 8,176 | 207 / 1,446 |

**What the waste is.**

- Lane A's phase-3 waste (3.5 T) is almost all reads that overflowed in phase 1. They take no step in k_p3, but they
  keep a lane of their warp until the warp finishes.
- Lane M's phase-3 waste is mostly `wa`: windows with more than 8 aligns, against M's NWA of 8. That is 13.7 k reads
  in pbmc1k_v3 and 49.5 k in pbmc5p_v1.

**Can lane A's phase-3 overflows be seen from phase-1 state?** They are 13.9 k reads (18.8 k), 0.94% (1.34%) of the
reads that pass phase 1. A held-out test of a depth-8 tree gives:

| flagged share of reads | recall, phase 1 | precision | recall, + scan hits | precision |
|---|---|---|---|---|
| 1% | 52.8% (27.3%) | 49.5% | 62.0% (60.1%) | 58.1% |
| 2% | 62.6% (46.8%) | 29.3% | 73.4% (81.0%) | 34.4% |
| 5% | 79.0% (79.4%) | 14.8% | 88.2% (93.9%) | 16.5% |
| 10% | 92.6% (96.1%) | 8.7% | 98.0% (99.4%) | 9.2% |

So 97% of lane A's overflows are known after phase 1 (`win`). The remaining phase-3 overflows can be caught before
k_p3 by flagging 5-10% of reads, which would cost wide-lane capacity. Raising the narrow caps that fail is cheaper:
NKEEP 8→16 and NWA via a pooled align array.

## 10. Control-flow signatures and SIMT work efficiency

Number of distinct signatures needed to cover 50% / 80% / 95% of reads, and of cascade cycles:

| signature | distinct | reads 50 / 80 / 95% | cycles 50 / 80 / 95% | pbmc5p_v1: distinct, reads, cycles |
|---|---|---|---|---|
| exact (`nWall`, `nWAsum`, `nP`, stitched, unmapType, log2 steps) | 138,253 | 31 / 3,151 / 49,168 | 4,377 / 43,049 / 94,655 | 139,479; 15 / 3,505 / 52,883; 6,904 / 50,946 / 98,534 |
| bucketed (log2 of the counts, same fields) | 925 | 6 / 29 / 71 | 15 / 45 / 97 | 988; 5 / 33 / 82; 14 / 51 / 133 |
| phase 1 only (`nWall`, `nP`, stitched) | 4,110 | 15 / 289 / 1,053 | 367 / 1,092 / 1,904 | 3,714; 7 / 216 / 893; 263 / 847 / 1,676 |
| phase 1 bucketed (log2 `nWall`, `nP`, `nEntries`, stitched) | 354 | 7 / 44 / 106 | 22 / 54 / 107 | 291; 5 / 32 / 84; 16 / 46 / 89 |

**SIMT work efficiency.** The simulation takes warps of 32 consecutive reads of a lane batch and computes Σ work /
(32 × the warp's max work). It assumes a thread idles once its read is done. It is an upper bound on the SIMT
efficiency E1 measures, because branches inside a step diverge too; E1 measured 6% (1.94 active lanes of 32).
Reordering happens within each launch batch, after k_p1/k_p2:

| lane (reads reaching k_p3) | order | steps | windows | SA hits | pbmc5p_v1 steps / windows / hits |
|---|---|---|---|---|---|
| A (1,481,273) | as run (read order) | 14.9% | 19.5% | 20.9% | 14.3 / 18.9 / 20.4% |
| A | phase-1 bucketed signature | 47.4% | 81.1% | 64.8% | 34.7 / 82.8 / 59.0% |
| A | `nWall`, then `nEntries` | 49.1% | 99.9% | 71.4% | 35.5 / 99.8 / 63.5% |
| A | oracle: steps | 99.6% | 77.9% | 62.6% | 99.6 / 59.7 / 51.0% |
| M (448,865) | as run | 31.3% | 48.9% | 34.7% | 30.2 / 46.9 / 35.4% |
| M | phase-1 bucketed signature | 61.8% | 75.4% | 60.8% | 55.4 / 74.6 / 59.6% |
| M | `nWall`, then `nEntries` | 68.7% | 99.9% | 66.9% | 59.8 / 100.0 / 64.4% |
| M | oracle: steps | 99.9% | 69.1% | 51.8% | 99.9 / 63.7 / 53.0% |
| B (88,118) | as run | 35.6% | 61.7% | 34.8% | 39.6 / 53.4 / 34.7% |
| B | phase-1 bucketed signature | 55.9% | 70.7% | 46.3% | 50.4 / 70.8 / 41.8% |
| B | `nWall`, then `nEntries` | 56.9% | 99.8% | 45.2% | 46.8 / 99.8 / 37.1% |
| B | oracle: steps | 99.8% | 73.9% | 47.3% | 99.9 / 60.5 / 43.0% |

**What the simulation says.**

- **Read order is close to random in work.** In lane A, about 26% of the threads in k_p3 have nothing to do: they
  overflowed in phase 1. The rest mix reads with 1 and 16 windows.
- **Sorting by `nWall` makes the window loop uniform.** Window-loop efficiency reaches about 100% in every lane. Step
  efficiency rises 3.3× (2.5×) in lane A and 2.2× (2.0×) in lane M.
- **Steps and windows cannot both be perfect.** Sorting by the true step count (the oracle) makes steps uniform but
  leaves the window loop at 60-78%.
- **A key of phase-1 quantities gets most of the gain.** Given how few signatures cover the reads, this answers E1's
  question: re-convergence by sorting works. The remaining divergence is the recursion's own branching and the
  long-tailed steps of windows with many aligns.

## 11. Three samples side by side

nuclei2k_v2 (mouse nuclei, GRCm39) ran as job 795348 on debug-gpu. It had 8 CPUs, so its host-lane ms is not
comparable. Its full tables are in `census_nuclei2k_v2.md`.

| quantity | pbmc1k_v3 | pbmc5p_v1 | nuclei2k_v2 |
|---|---|---|---|
| distinct reads (all reads) | 1,991,926 (4,118,999) | 1,850,654 (6,118,914) | 1,967,994 (2,899,428) |
| identical to STAR, default / census build | 100% / 100% | 1,850,645 / 1,850,645 (9 `lread`) | 100% / 100% |
| GPU ms, default / census build | 861.1 / 899.6 | 917.2 / 957.3 | 823.3 / 857.7 |
| lane A / M / B ms (default build) | 269 / 337 / 255 | 333 / 283 / 301 | 222 / 395 / 206 |
| reads overflowing lane A | 26.3% | 25.4% | 38.7% |
| `nWall` p50 / p99 / max | 3 / 91 / 238 | 2 / 88 / 226 | 9 / 79 / 206 |
| reads with `nWall` > 16: reads / cycles / steps | 25.6 / 72.6 / 79.4% | 24.1 / 70.1 / 76.2% | 37.5 / 79.9 / 80.9% |
| thread cycles in lanes the read overflowed | 18.3% | 20.6% | 17.3% |
| phase 1 share of thread cycles | 31.9% | 29.6% | 33.7% |
| seeds share of lane A thread cycles | 37.5% | 34.5% | 35.4% |
| "too short" unmapped: reads / cycles | 7.2 / 19.5% | 23.2 / 54.5% | 23.1 / 47.1% |
| multimappers (nTr > 1): reads / cycles | 8.3 / 17.6% | 17.3 / 38.6% | 14.1 / 27.9% |
| singletons: distinct reads / cycles | 80.0 / 87.1% | 81.5 / 91.6% | 88.0 / 95.0% |
| critical path of a window-parallel lane (largest window / all steps) | 16.1% | 24.2% | 14.4% |
| router, rank by `nWall` at 30 / 20 / 10% wide (oracle) | 74.7 / 59.7 / 37.4% (76.2 / 63.7 / 41.7) | 73.0 / 60.7 / 37.1% (76.4 / 65.7 / 44.8) | 66.1 / 47.2 / 27.8% (69.9 / 52.4 / 32.5) |
| steps predicted from phase 1, log-log R² | 0.953 | 0.864 | 0.970 |
| overflow at 99.9% caps: distinct / cycle share | 0.43 / 2.4% | 0.47 / 2.5% | 0.50 / 2.1% |
| bucketed signatures for 50 / 80 / 95% of reads | 6 / 29 / 71 | 5 / 33 / 82 | 8 / 24 / 57 |
| lane A step efficiency: as run → sorted by `nWall` | 14.9 → 49.1% | 14.3 → 35.5% | 17.5 → 58.4% |
| lane M step efficiency: as run → sorted by `nWall` | 31.3 → 68.7% | 30.2 → 59.8% | 36.2 → 73.4% |
| window-loop efficiency: as run → sorted (A / M / B) | 19.5 / 48.9 / 61.7 → 99.9 / 99.9 / 99.8% | 18.9 / 46.9 / 53.4 → 99.8 / 100 / 99.8% | 21.8 / 51.2 / 60.9 → 99.9 / 99.9 / 99.7% |

The nuclei sample differs from the PBMC samples in three ways:

- **More windows.** The `nWall` median is 9, and 38.7% of reads leave lane A. As a result lane M, not lane A, is the
  most expensive lane.
- **More "too short" work.** Reads STAR rejects as too short take 47% of the cycles.
- **Steps are more predictable.** The design conclusions are the same in all three samples.

## Design consequences

**1. Route once, on the window count, right after phase 1.**

- `nWall` is a sufficient router: it captures 74.7% of cycles at a 30% wide share, against 76.2% for the oracle and
  74.6% for the best fitted model. A learned cost model is not needed.
- `nWall > 16` sends 25.6% of reads (72.6% of cycles, 79% of steps) to a wide class. 97% of today's lane-A overflows
  are this same test (`win`).
- The two classes:
  - narrow: one thread per read, `nWall` <= 16;
  - wide: one warp per read.
- Reads that later fail a narrow cap (0.7% (1.0%) of reads: `keep`, `wa`, `wtr`, steps) can keep the existing
  overflow path.

**2. Do not redo phase 1, and do not carry idle reads into k_p3.**

- The cascade spends 18-21% of its thread cycles in lanes a read overflows:
  - k_p1 work in A and M is thrown away for reads that then redo it in the next lane: 3.4 T + 1.6 T thread cycles;
  - reads that already overflowed hold k_p3 warp lanes in A: 3.5 T.
- Phase 1 should run once with the window list kept or rebuilt from carried seeds (`nP` <= 32). Each batch should be
  compacted (or sorted, point 3) before k_p3.

**3. Sort each batch by `nWall` (then `nEntries`) before k_p3.**

- This is exact (it only permutes reads) and costs one 32-bit key sort per batch. In the simulation it lifts lane A's
  k_p3 step efficiency from 15% to 49% (14% to 36% in pbmc5p_v1), and its window-loop efficiency from 20% to 99.9%.
  Lane M's step efficiency goes from 31% to 69%.
- Signatures are few: 6 / 29 / 71 bucketed shapes cover 50 / 80 / 95% of reads. So sorting groups like reads, without
  needing exact shapes.
- Expect a large part of k_p3 time to come back in every lane. k_p3 is 146 of 269 ms in lane A. Measured gains will be
  lower than the step bound, because divergence inside a step remains.

**4. Narrow-lane state (99.9% of the narrow class, both samples).**

| cap | value |
|---|---|
| NP | 24 |
| NW | 16 |
| aligns | one pool of 96 per read instead of 16 × NWA (`NWAsum` 99.9% = 51 / 88), with per-window offsets |
| NEX | 4-5 |
| NWTR | 8-16 |
| NKEEP | 16 |
| NBLOB | 640 |
| step budget | 1,024-2,048 (the pbmc5p_v1 narrow tail reaches 2,642 at 99.9%) |

The pooled align array is the main change. It replaces the per-window NWA = 16 slab. In the wide class (a pool of
about 600), it also removes lane M's `wa` overflows (NWA 8), which are 49.5 k reads in pbmc5p_v1.

**5. Wide lane: parallelise over windows, not over the recursion.**

- 81% of windows have one align and take exactly 5 steps. Per-window time is dominated by a fixed finalization cost.
- The largest window holds 8.6-9.8% of the steps of a lane-M/B read. A warp that gives each lane one window has a
  critical path close to Σ steps / 32.
- State per wide read:
  - p99: 117 windows with aligns and 198 aligns;
  - 99.99%: 552 pooled aligns and 195 windows;
  - max: 1,180 aligns and 238 windows.
  - This state belongs in shared or global memory per warp (E5 sizes the transcript part). With 16-byte aligns, the
    99.99% align pool is about 9 KB.
- Replacing the recursion by a chaining DP saves only the many-align windows (steps / pairs = 4.05). That is not the
  first lever.

**6. Exact early rejection of "too short" reads (to investigate).**

- These reads take 19.5% of cycles in pbmc1k_v3 and 54.5% in pbmc5p_v1, on 7% / 23% of reads, and have many windows.
- A provable upper bound on the attainable score from phase-1 state could skip their stitching. Two examples:
  - covered read bases plus the maximal extension;
  - STAR's `outFilterScoreMinOverLread` / `outFilterMatchNminOverLread` thresholds.
- Such a bound has to be exact for every read, or identity is lost. It is a candidate for a wave-2 experiment, not a
  finding.

**7. Phase 1 needs its own work.** It takes 30-32% of all thread cycles.

- In lane A, k_p1 is 47% of thread cycles and 107 of 269 ms (18.7 M reads/s).
- The seed search alone is 37.5% of lane A's thread cycles, which is 80% of k_p1.
- Lanes M and B spend another 11% and 5% of their thread cycles seeding the same reads again.
- Seed search is the largest single item for the narrow class. It is a separate wave-2 target:
  - warp-cooperative SA search;
  - or the seed list carried between lanes.

**8. Duplicates are not the problem.** Reads seen 10 or more times are 36% (61%) of reads but 0.6% (1.0%) of cycles.
After dedup, cost is set by singletons: 80% of distinct reads and 87% (92%) of cycles.

## Jobs, commits, files

**Commits.**

- Instrumentation: `f58be38d` (records). The binaries were built from `f07066a0`; `bench/gpu/m5` is unchanged after it.
- Aggregation: `3e76e104`.
- Branch `gpu/fable/census` from `cf92cdb9`.

**Jobs on Clipper.**

| job | what |
|---|---|
| 795233 | compile at `f58be38d` (debug-cpu) |
| 795243 | compile at `f07066a0` (debug-cpu) |
| 795244_0 | pbmc1k_v3 run (H100 NVL, 370 s) |
| 795244_1 | pbmc5p_v1 run (339 s) |
| 795332 | re-aggregation |
| 795343 | re-aggregation with the SIMT section |
| 795351 | re-aggregation with the phase table (final pbmc tables) |
| 795348 | nuclei2k_v2 run (debug-gpu, 8 CPUs, 583 s); the short-gpu task 795331_2 was cancelled unrun |

**Files.**

- In the repo, `bench/fable/census/results/`:
  - `census_<sample>.{md,json}`: all tables and histograms;
  - `census_<sample>.meta.json`;
  - `m5_<sample>_{m5,m5c}.json`: harness timing and identity.
- On Clipper, `$W/jobs/fable/census/results/raw/`: raw records `census_<sample>.{reads,wins}.bin.gz` (256 B per read
  and lane, 20 B per window; no sequences). Re-aggregate with
  `LABELS=... sbatch bench/fable/census/jobs/census_agg.sbatch`.


## Method

**What is instrumented.** `bench/gpu/m5/stitch.cuh` and `bench/gpu/m5/m5.cu` gain counters under `-DM5_CENSUS` only (the
default build compiles exactly the previous code paths; the census build adds statistics that the mapping never reads):

| counter | meaning |
|---|---|
| `nSteps` | include/exclude frames of `stitchWindowAligns`, counted with or without a step budget |
| `cnHits` | SA entries that hit a window (the `assignEntry` calls that can do something; on the host lane counted with `entryHit`) |
| `cnMaxWA` | peak aligns of one window during the assignment (what the `NWA` capacity has to hold) |
| `cnMaxEx` | largest `nExons` of a transcript passed to `stitchAlignToTranscript` (`NEX` must exceed it, up to STAR's 20) |
| `cnKeepMax`, `cnWinTrMax` | peak transcripts kept across windows (`NKEEP`), per window (`NWTR`) |
| `cnStitch`, `cnFinal`, `cnRec` | `stitchAlignToTranscript` calls, finalizations (leaves of the recursion), recordings |
| `cn[0..3]` | thread cycles: the whole `k_p1` call (seeds, windows, flanks), the `k_p2` scan (warp cycles), the assignment in `k_p3`, the whole `k_p3` call (assignment + stitching + select + encode) |
| per window | steps, cycles and transcripts recorded of every stitched window |

The existing `Work::clk[8]` counters (seeds, windows, stitching, select+encode; inside stitching
`stitchAlignToTranscript`, finalization, copies, recording) are written into the records as they are. `clk[1]` covers window
creation and flanks only: in the staged lanes the assignment runs in `k_p3` before `mapFinish` starts its clock, so it was
never counted; `cn[2]` now counts it. Every statistic of a read is reset at the start of `mapPhase1` in the census build, so
a read that overflows early does not inherit the previous read's values (all of these fields are set again before the
mapping reads them; the identity check below confirms it).

**Records** (`bench/gpu/m5/census.cuh`): one 256-byte record per (distinct read, lane it ran in) with lane, status,
OvWhy, unmapType, multiplicity, seeds (`nP`, max/sum `Nrep`, seeds with `Nrep` > 1 and > `winAnchorMultimapNmax`),
windows (created `nWall`, with transcripts `nW`), aligns per window (sum, max, non-empty windows, peak), SA entries
(`nEntries`) and hits, steps, max steps of one window and its index, peaks of every capacity, transcripts (`trNtotal`,
`nTr`), the 8 phase counters and the 4 call counters; and one 20-byte record per stitched window of the lane that gave the
read's result (window index, aligns, steps, cycles, transcripts). Files `PREFIX.reads.bin`, `PREFIX.wins.bin`,
`PREFIX.meta.json` (`m5c --census PREFIX`). No sequences are written.

**Aggregation**: `bench/fable/census/census_agg.py` (numpy only) joins the records per read. A read's *final* record is
the one of the lane that produced its result; its values are the read's full state (a lane that overflowed stopped
counting at its capacity). Cost is thread cycles (`clock64`): *cascade cycles* = the sum over every GPU lane the read went
through of `k_p1` + `k_p3` cycles (the work today's cascade spends on it, including the lanes it overflowed), *final-lane
cycles* = the same for the finishing lane only. Reads that finish on the host have no GPU cycles of the final lane; their
wasted GPU cycles in lanes A/M/B are counted.

**Caveat on cycles.** `clock64` deltas are elapsed SM cycles of one thread: they include the time its warp spends on other
branches (SIMT divergence inside the warp) and the time the SM spends on other warps. They rank reads by cost and their
sums apportion a lane's time between classes of reads, but they are not exclusive instruction counts. The deterministic
work counts (steps, entries, hits) are given next to them.

**Runs.** Job array `fable-census` on Clipper (H100 NVL, CUDA 13.3 image), the m5 harness recipe of
`bench/gpu/m5/jobs/m5_run.sbatch` in one job per sample: index copy to `/tmp`, decode, distinct sequences, every k-th of
the sorted list to 2 M (the standard dump set), `stardump` truth, then `m5` (default build) and `m5c` (census build), both
`--reps 1` and the engine's lane settings (`src/engine_gpu/engine_gpu.cu`): budgets 1,024 / 2,048 / 2,048 steps,
batches of 262,144 / 262,144 / 65,536 reads for lanes A / M / B. Job script: `bench/fable/census/jobs/census_run.sbatch`;
build: `bench/fable/census/jobs/census_compile.sbatch`.
