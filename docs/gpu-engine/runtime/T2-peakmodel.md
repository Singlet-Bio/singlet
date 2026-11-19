# T2 peak model: what an item allocates on the host, and an offline test of a limit-aware gate (peakmodel)

Task `peakmodel` of throughput track 2 (branch `t2/peakmodel`, from `gpu/throughput-2` f2e3598b). Analysis only: no file under `src/` or `include/` changed. Build in every run: `bin/t2-hostcut5`
(v1.4.0 + the hostcut cuts), default switches. Sizes are GiB; "GB" appears only inside quoted program lines (the engine and the pool print GiB with that label). "Above the idle baseline"
means cgroup anon+shmem of the job minus the cgroup's anon+shmem when no item runs (section 0). Estimates and predictions are marked as such; everything else is a measurement or a size
read from the code and checked against a logged figure.

Scripts (`bench/gpu-runtime/`): `t2-peak-profiles.py` (solo profiles above the correct baseline), `t2-peak-engine-lines.py` (the engine's size lines per class), `t2-peak-ledger.py` (the ledger),
`t2-peak-fit.py` (predictor, margin, phase rules), `t2-admit-replay2.py` (replay of the mixes), `t2-peak-gate-tests.py` (safety on the single-job recordings, the rule at the real starts of 818937 / 818940,
the 365 GiB triples), `jobs/t2-peakmodel-item.sbatch` (one item, `SINGLET_E16_TRACE=1`, copies the item's STAR Log.out). Data: `t2-peakmodel-predictor.json`, `t2-peakmodel-classes.json`.

## Summary

| question | answer |
|---|---|
| The baseline of T2-pooladmit.md | **Wrong by 20.4 GiB.** It took 22.03 GiB (the 5 s before the first start) while the non-host server's 24.7 GiB page-locked index was still coming up. With nothing running the cgroup holds **42.5 GiB** (pool genome 17.65 + server 24.7; median 42.54 of the level 1 s after items 0-16 end, range 42.39-43.33). Every "TRUE" figure of that document is 20.4 GiB too high; the end-of-item levels of "21-24" it read as "no drift" are this offset. Corrected solo peaks: 4.3 to 230.0 GiB (table 1). The pool estimate (61.8 + 0.2235 x) is above the true peak of all 18 classes by 25 to 123 GiB (not 5 to 103). Its statement that the RSS sum equals the true peak from 100 M to 1,000 M reads is the same artefact: 1.11-1.58 with the corrected baseline. The prompt's step list (25-42, 59, 75-86, 147-149, 207-210, 250) is the old figure minus 20.4: 4-21, 38, 54-66, 126-129, 187-189, 230. |
| Where the steps come from | Not from the read count alone. Four regimes, set by the device budget (31 GiB in this harness: `SINGLET_ENGINE_GPU_MEM_GB=31`), the cDNA bases and the distinct sequences D: (i) engine path in one pass (n up to 93 M here), (ii) engine path with read chunks (RC, "P3-MEM-03", from about 144 M reads at 90 bp, 93 M at 150 bp with the CellRanger4 clip on), (iii) **fallback to STAR's loop** (level 0) in the non-RC range when the records' device need exceeds the budget: d100s_3p and nuclei2k_v2 (62.4 and 59.1 GiB at 100-119 M reads, against 20 GiB for 75 M), (iv) the 150 bp classes carry twice the bases per read (d500s_3p, 126.3 GiB at 500 M reads, equal to d700s_3p at 703 M). AUX_COMPACT acts only in RC. |
| Ledger | For the 16 classes on the engine path: peak = planes + region + 0.9 x Solo keep + late store + (6.06 + 0.0238 n[M]); residual (measured - sum) -6.7 to +5.8 GiB, rmse 3.5 (table 3). Planes (24 n + bases/2 bytes) and region ((28 + 8 nw) n + 16 D + 18 nAln + 4 nOps + nSeq bytes) are exact: checked against the logged "released N GiB of host planes" and "N MB in use". The last term is the remainder and is **not itemised** (up to 37 GiB at d2600_3p). |
| Predictor | P_B(n, bases) = 6.60 + 0.0817 n[M] + 1.0588 bases[G] outside the zone 70-160 M reads; inside it max(P_B, planes + 6.06 + 0.49 n). On the 18 classes: worst under-prediction **4.7 GiB** (d1000_3p), rmse 3.2 on the 14 fitted classes, leave-one-out worst 6.1. With n and the file size only (P_A): worst 10.4, leave-one-out 27.8. Margin: 7 from the solo data; **12 recommended** (zero violations on all five recordings needs 10.1; the same class d2600_3p ran 15.6 above its solo run in mix 818763). |
| Phase rules | F0 = P + m before the level line, F1 = min(F0, 0.90 P + 7.1 + m) after it, F2 = min(F1, 0.50 P + 2.6 + m) after the waited line. Worst (rule - measured) on the 18 profiles at m = 12: +7.3, +11.1, +12.0 (no class above its rule). The current code (0.4 x estimate after the level line, min(that, 80) after the waited line) is under the measured later maximum by up to 55.1 and 6.2 GiB. |
| (a) safety | m = 12 (and 10.1): zero violations at every second of 818763, 818882, 818938, 818937, 818940. m = 0: 8 episodes in the mixes (largest 10.0, d2600_3p in 818763), 678 s in 818938 (largest 4.7). Solo peaks add up in the mixes: B + sum P is 1.7 GiB under the run's peak in 818763 (two items in flight) and 71 GiB over it in 818882 (three items). |
| (b) 818937 / 818940 | The rule refuses items 1 and 2 at their real starts in both jobs (item 1: free 109.3 / 54.1 GiB, still to allocate 32.3 / 27.0, F0 71.9: margin to the reserve -34.9 / -84.8). Counterfactual starts: +5, +129, +243 s (200 GiB) and +4, +128, +159 s (150 GiB); overlay peaks 96.5 / 96.6 GiB against the limits. No kill in the counterfactual. It serialises three items that fitted in 200 GiB (real peak 157.1). |
| (c) 365 GiB lane | PREDICTION. At the free-node level measured here (free + cgroup anon+shmem = 444-472 GiB) **the current gate admits no triple whose true peaks plus baseline exceed 365 GiB**. It does at 510 GiB and above; the smallest such triples: d1000_3p + d500s_3p + d700s_3p (510), d2600_3p + nuclei2k_v2 + 2nd nuclei2k_v2 (540-550), d100s_3p + d100s_3p + d2600_3p (650-740); d100s_3p + d2600_3p + d500s_3p appears at 530. The confirmation job was not run (section 3c). |
| (d) waits | Replays 818882 / 818763 (baseline censored at the real start 16,354 / 16,130 s; real 19,195 / 20,570): new rule at cache share 0.25: 11,688 / 9,703 s (m = 12), 6,471 / 6,326 (m = 0); at cache share 1.0: 371 / 354 s. |

## 0. The idle baseline

818938 (one worker, 18 items back to back, exclusive): the cgroup's anon+shmem in the sampler's first rows is 0, 14.5, 22.2, 24.7 GiB (818937 the same) while the pool process holds 17.65 GiB RssAnon from the start and the
non-host server's RssShmem goes 15.0 to 24.7 GiB in the first 20 s (`memsamp.tsv`, item -2). Items start at +7 s. The level 1 s after the end of items 0-16 is 42.4-43.3 (median 42.54): pool 17.65 + server 24.7. It is the same
before every item (`min during item` 42.4). In a job with two pools sharing one non-host copy (the mixes) the minimum of `mon.tsv` is 60.4 and 60.0 is used (2 x 17.65 + 24.7). After an item's server request the server's shm is 25.85 (+1.1).
`t2-peak-profiles.py` writes the profiles above the measured baseline. In T2-pooladmit.md's "true" replay column (6,119 / 7,914 s) the head's charge was 20.4 GiB too high (the in-flight items' remaining growth is a difference and is not affected).

## 1. Part 1: where the true peak comes from

### 1.1 Corrected solo peaks (818938; the three repeats are in 1.8)

```
sample           reads(M) bases(G)   peak peak@s level@s waited@s   end@s
pbmc2m                2.0     0.2    4.3      7      29      31      33
pbmc10m              10.0     0.9    6.6      7      15      19      22
pbmc1k_cite_v3       45.4     4.1   15.7     22      25      35      42
pbmc5p_v1            60.0     5.5   16.4     20      24      44      55
hgmm1k_v3            63.1     5.7   19.0     27      31      47      69
pbmc1k_v3            66.6     6.0   21.1     26      29      44      53
hgmm1k_v2            75.2     6.8   20.3     26      30      47      65
neuron1k_v3          92.9     8.4   38.0     76      62      98     110
d100s_3p            100.0    14.9   59.1    132     134     339     356  FELL BACK (level 0)
nuclei2k_v2         118.7    11.6   62.4    180     155     343     355  FELL BACK (level 0)
d300_5p             299.6    27.3   61.7     69      86     126     146
d300_3p             300.2    27.2   54.1     47      58     108     124
hgmm5k_v3           313.0    28.3   65.7     96      97     160     295
d500s_3p            500.5    74.7  126.3    192     213     379     452
d700s_3p            703.4    62.8  128.8    130     154     384     475
d1000_3p            999.8    88.5  186.6    214     263     536    1079
d1000s_3p          1004.6    91.2  189.2    208     255     537     692
d2600_3p           1303.7   115.7  230.0    331     333     520     649
```
(peak = max of anon+shmem above 42.54 while the item runs; seconds after the item's start.)

### 1.2 What the code allocates, and the regimes

The item is the pipeline process (`sg-scan`, the parent) and its forked STAR child (`star-prefill`; the two `star-probe` forks of the CR4 probe share the parent's pages and are in the pool's RSS sum twice). The child's RSS includes the pool's
resident genome (17.65 GiB, shared copy-on-write): the cgroup counts it once, in the idle baseline. Item level = child anon - 17.65 + child shmem + parent + the server's growth. The gate's own `sharedGb` subtracts the same figure.

1. **Decode (parent, before the STAR fork; `SeqCacheGpuIngest.cpp`, `engine_gpu_ingest.h` PackedReads).** One array per plane, written once: `off` 8 B/read (n + 1 u64), barcode read `r1` 8 x r1w = 16 B/read (28 bases at 21 per word), cDNA `seq` 2 bits/base with
   an exact N record (bases/4 B), cDNA quality 2 bits/base (bases/4 B). Planes = 24 n + bases/2 bytes: 82.9 GiB at d2600_3p (logged: 26.94 + 26.94 + 19.43 + 9.71 = 83.02). The STAR child inherits them by fork; they are freed by E22
   8-38 s after `mapDistinct` returns (d500s_3p 8 s, d2600_3p 38 s in the marks). Then, in the child, the `aux` array: (2 + nw) x 8 B/read with nw = ceil(longest read / 64) (32 B at 91 bp, 40 B at 150 bp; 38.85 GiB at d2600_3p), the CellRanger4 clip arrays (4 B/read) and `cr4info` (1 B/read).
2. **mapDistinct (`engine_gpu.cu`).** The device budget (`memBudget`) is 31 GiB here. The reads phase needs `readsPhase(lean) = max(RD + 28 n + AUX + ENT + QM, RD + CLIP, RD + 53 n)` with RD = bases + 8 n (1 B/base on the device), AUX = (16 + 8 nw) n, ENT + QM = (24 + 8 nw) n,
   CLIP = 29 n + bases when the CR4 clip is on (0 otherwise). **RC** (read chunks) when `context + tables + slack` (about 2.1 GiB) + readsPhase(lean) exceeds the budget: n above about 144 M reads at 90 bp with the clip, 93 M at 150 bp, 158 M / 114 M without the clip
   (estimate from the formula; the 18 classes agree: every class from 299.6 M on logs "read chunks", none below 119 M does). In RC the E10 entries and masks go to the region as they are made, AUX_COMPACT keeps only CR/UR (16 B/read) and releases the aux pages per chunk,
   the host records are released as the records phase uploads them (handover). The records' host side: Out.seq = 2 x nSeq bytes, blob, offsets 16 D, recSrc + recRead 12 D (d500s_3p: 22.6 + 11.0 + 2.5 + 1.9 GiB), readRecord 4 n, clip5/3 4 n, cr4info n.
3. **Fallback.** Without RC, `e2eBuild` checks `recNeed = used + 28 D + blob + seq + 101 D + seq/2 + 24 D + 4 n + 0.25 GiB + keep` against the budget; above it the records are not built on the device ("E8 records not built ... needs about 35.07 GB" d100s_3p, "43.93 GB" nuclei2k_v2, budget 31.00),
   E10/E11 are off (level 0) and STAR's loop does the per-read work from the hand-off file: the host then holds planes + the host records + STAR's sequence cache, created at 3 x (blob + seq/4 + 40 D) (d100s_3p: "28148 MB", exactly 3 x (4.7 + 8.6/4 + 40 D) GiB) and filled by the loop.
   The fallback needs D above about 62 M (D / n 0.62 and 0.67 for the two classes that fell back, 0.2-0.5 for the others) and no RC. It is invisible before the engine's dedup.
4. **Post-return bump (the peak of most classes).** After `mapDistinct` returns, until E22 frees the planes: planes + the E2E region (shared memory, mapped by child and parent) + E11's kept inputs in page-locked host memory (counted as shmem) + the compact store of the unmapped reads (U x (1.25 L + 38) B, built from the planes
   on a thread from the return) + what STAR keeps for its loop. In RC classes a second moment, the end of the lanes (`split.free`: planes + host records + 16 n compact CR/UR + the region's entries and masks 40 n), reaches the same size.
5. **Tail (parent).** Hand-off pass reads/clip/lowq (E2EPassOut 18 GiB at d500s_3p), pileup, all-site, export: at most 20 GiB above the level at the waited line (section 2). For neuron1k_v3 (81 % of the reads unmapped on this genome) the late store is 11.1 GiB and the peak falls 14 s after the level line.

Not found as a large host term: hash tables rounded to powers of two (the dedup partitions and the hash arrays are on the device; the host sees `Ip` / groups per read chunk), anything tiered by depth other than RC / AUX_COMPACT / the fallback.
The ledger closes without them to within 7 GiB.

### 1.3 The ledger

Engine path (16 classes), GiB. planes: formula, equal to the logged E22 figure; region: logged, equal to the formula in the file; Solo keep: 0.9 x the logged "Solo inputs kept ... GB"; late: the logged E22 compact store; a+bn: 6.06 + 0.0238 n[M] (remainder, not itemised).
`peak - t(level)` is the measured peak second relative to the level line (negative: before it).

```
class             n (M) |  planes  region .9 keep    late    a+bn   U/n |     sum measured |  resid   peak-lvl(s)
pbmc2m              2.0 |     0.1     0.2     0.0     0.0     6.1  0.06 |     6.5     4.3 |   -2.2   -22
pbmc10m            10.0 |     0.6     0.9     0.2     0.1     6.3  0.06 |     8.1     6.6 |   -1.6    -8
pbmc1k_cite_v3     45.4 |     2.9     3.6     1.0     0.4     7.1  0.05 |    15.0    15.7 |    0.6    -3
pbmc5p_v1          60.0 |     3.9     3.9     1.2     0.7     7.5  0.08 |    17.2    16.4 |   -0.8    -4
hgmm1k_v3          63.1 |     4.1     4.4     1.3     2.6     7.6  0.28 |    19.9    19.0 |   -0.9    -4
pbmc1k_v3          66.6 |     4.3     5.2     1.4     0.5     7.6  0.05 |    19.1    21.1 |    2.0    -3
hgmm1k_v2          75.2 |     4.8     4.7     1.5     3.2     7.9  0.29 |    22.1    20.3 |   -1.9    -4
neuron1k_v3        92.9 |     6.0     5.9     1.9    11.1     8.3  0.81 |    33.1    38.0 |    4.9   +14
d300_5p           299.6 |    19.4    18.0     5.9     3.2    13.2  0.07 |    59.6    61.7 |    2.0   -16
d300_3p           300.2 |    19.4    16.9     5.8     1.7    13.2  0.04 |    57.0    54.1 |   -2.9   -11
hgmm5k_v3         313.0 |    20.2    19.7     6.3    12.6    13.5  0.28 |    72.4    65.7 |   -6.7    -1
d500s_3p          500.5 |    46.0    41.7    10.2     4.7    18.0  0.04 |   120.5   126.3 |    5.8   -22
d700s_3p          703.4 |    45.0    43.2    14.0     2.4    22.8  0.03 |   127.4   128.8 |    1.3   -24
d1000_3p          999.8 |    63.6    63.1    20.1     7.0    29.9  0.05 |   183.6   186.6 |    3.0   -48
d1000s_3p        1004.6 |    64.9    65.6    20.3     4.9    30.0  0.03 |   185.7   189.2 |    3.5   -46
d2600_3p         1303.7 |    83.0    67.6    24.9    23.8    37.1  0.13 |   236.4   230.0 |   -6.4    -2
```
Residual (measured - sum): -6.7 to +5.8, rmse 3.5. The residual term a + b n (6.06 + 0.0238 n[M], 25.6 B/read): the parent item process (2.4-3.1 GiB RSS after the fork, memsamp), the server's growth (1.1), and STAR's private structures. At d2600_3p it is 37.1 GiB of which the code reading
accounts for about 30 (readRecord 4.9 + clip 4.9 + cr4info 1.2 + record offsets 3.1 kept to STAR's loop, per T2-hostmem.md's table, and a 16.4 GiB private index copy in that trace); for d300_3p (13.2) no private index copy is visible in the marks (anon at region complete 44.4 = planes 19.3 + late 1.4 + shared 17.65 + 6.1). Not resolved.
For hgmm1k / hgmm5k / neuron1k (mouse reads on the human genome) U / n is 0.28-0.81 and the late store is 2.6 / 12.6 / 11.1 GiB.

Fallback classes (level 0), `planes + 6.06 + 0.70 GiB per M distinct sequences`: d100s_3p 9.2 + 6.06 + 43.3 = 58.5 (measured 59.1, three-class repeat 58.2), nuclei2k_v2 8.0 + 6.06 + 55.9 = 70.0 (measured 62.4, residual -7.6; its own trace was not made).
The d100s_3p trace (818961): "sizes @summary": anon 69.6, shmem 4.5; planes 9.2, host records seq 8.6 + blob 4.7 + offsets 0.9, sequence cache 28,148 MB created after it; the level 0 profile climbs from 20 to 58 GiB over +0..+140 s (STAR's loop) and falls to 18 at +150 s.

End of the lanes (`split.free`, RC), from the e16 marks (child anon - 17.65 + child shmem = the child's share of the cgroup): the same sizes at the other moment:

```
class     child at split.free | planes  host recs (seq+blob+offs)  recSrc+recRead  readRecord+clip5/3+cr4info  CR/UR 16n  region entries+masks (shm) | sum   rest
d300_3p   58.0 - 17.65 + 11.6 = 51.95 | 19.4  10.0 (4.7+4.4+0.9)  0.7  2.5  4.5  11.6 | 48.7  3.3     (job 818960; measured peak 54.1 at this second)
d500s_3p  116.9 - 17.65 + 22.8 = 122.05 | 46.0  36.1 (22.6+11.0+2.5)  1.9  4.3  7.5  22.8 | 118.6  3.5   (job 818959; measured peak 126.3 at +203 s)
d2600_3p  186.5 - 17.65 + 49.0 = 217.85 | 82.9  39.6 (14.3+22.2+3.1)  2.3  11.0  19.4  49.0 | 204.2  13.7   (job 818641, hostcut5; 10 s sampler)
```
At this moment blob is the only term the pool log does not carry (70-116 B per record, 70.7 d500s_3p, 79.4 d300_3p, 115.8 d2600_3p).

### 1.4 What is known when

| term | before the item starts | after the decode / later |
|---|---|---|
| reads n | `.1fq` header, u64 at byte 20 (n_unique; n_original at 28), the pool reads the low u32 | |
| cDNA bases, longest read | **not in the header** (96 bytes: magic, version, n_streams @6, protocol_id @7, confidence @8, flags @9, codec @10-11, stream_lengths u16 x4 @12 (R1 = 28, R2 = 0 variable), n_unique @20, n_original @28, block_count @36, block_size @40, meta_size @44, bc_dict_size @48); per-block read and base counts are in the block heads (the pipeline's head scan reads them) | the `[1fq-ingest] Scanned N blocks: R reads, B cDNA bases, longest L` line, +3 s (pbmc) to +40 s (d2600_3p) after the start, before the STAR fork |
| file size | `stat` | |
| device budget, slots, threads, engine switches | command line / environment (`SINGLET_ENGINE_GPU_MEM_GB`, `_SLOTS`, hostcut switches); they set the RC and fallback thresholds | |
| CR4 clip on / off | no (probe on 200 k reads, +20 s) | probe line |
| nw (aux and masks words) | from the longest read | scan line |
| D, nAln, nOps, nSeq (region, host records, cache) | no | "N reads, D distinct" and "E8 records on the device ... nAln, ops, seq bytes" in the item's STAR Log.out only, not in the pool's log |
| U (unmapped reads, late store) | no (U / n 0.03-0.81 over the 18 classes) | "E10 level 5: ... U unmapped" in Log.out only |
| fallback (level 0) | no | "[engine_gpu] fallback: ... E8 records not built" on stderr, at +86 s (d100s_3p) |

### 1.5 The predictor

P_B(n, bases) = 6.604 + 0.0817 n[M] + 1.0588 bases[G] (fitted on the 14 classes outside the zone; GiB above the baseline). In bytes: 87.7 B per read and 1.14 B per base, against the ledger's per-read terms (24 + 16 + 44 + 9 + D- and U-driven) and 1.0-1.25 B/base.
Zone [70, 160) M reads (fallback possible: no RC and D up to n): P = max(P_B, planes + 6.06 + 0.49 n[M]), with the fallback coefficient 0.70 GiB per M distinct and D taken as 0.70 n (measured D / n 0.62, 0.67). P_A(n, S) = 6.495 + 0.1161 n[M] + 2.6736 S[G] with the same zone.
err = measured - P:

```
sample           reads(M) measured |     P_B   err_B   P_B+m | P_A   err_A
pbmc2m               2.0      4.3 |     7.0    -2.7    19.0 |   6.9    -2.6
pbmc10m             10.0      6.6 |     8.4    -1.8    20.4 |   8.4    -1.8
pbmc1k_cite_v3      45.4     15.7 |    14.7     1.0    26.7 |  15.1     0.6
pbmc5p_v1           60.0     16.4 |    17.3    -0.9    29.3 |  17.0    -0.6
hgmm1k_v3           63.1     19.0 |    17.8     1.2    29.8 |  18.0     1.0
pbmc1k_v3           66.6     21.1 |    18.4     2.7    30.4 |  19.1     2.0
hgmm1k_v2           75.2     20.3 |    47.8   -27.5    59.8 |  47.8   -27.5   (zone)
neuron1k_v3         92.9     38.0 |    57.6   -19.5    69.6 |  57.6   -19.5   (zone)
d100s_3p           100.0     59.1 |    64.2    -5.1    76.2 |  64.2    -5.1   (zone)
nuclei2k_v2        118.7     62.4 |    72.3    -9.9    84.3 |  72.3    -9.9   (zone)
d300_5p            299.6     61.7 |    59.9     1.7    71.9 |  59.8     1.9
d300_3p            300.2     54.1 |    59.9    -5.8    71.9 |  57.4    -3.3
hgmm5k_v3          313.0     65.7 |    62.1     3.6    74.1 |  63.7     1.9
d500s_3p           500.5    126.3 |   126.5    -0.2   138.5 | 115.9    10.4
d700s_3p           703.4    128.8 |   130.6    -1.8   142.6 | 135.1    -6.3
d1000_3p           999.8    186.6 |   182.0     4.7   194.0 | 189.1    -2.4
d1000s_3p         1004.6    189.2 |   185.2     4.0   197.2 | 196.8    -7.6
d2600_3p         1303.7    230.0 |   235.6    -5.7   247.6 | 223.1     6.8
worst under-prediction over all 18 (measured - P): B 4.7 (d1000_3p), A 10.4 (d500s_3p)
```
(P_B + m is printed with m = 12.) The zone's four classes are over-predicted by 5 to 27 GiB: before the start the pool cannot tell a fallback (D above 62 M) from a normal run (D 22-35 M).

**Margin.** Solo: in-sample worst under-prediction 4.7 (d1000_3p), leave-one-out 6.1, so 7. Mixes: in 818763 the d2600_3p item reached 245.6 above the baseline (solo 230.0, P_B 235.6): 10.0 above P. The smallest margin with zero violations of section 3a on all five recordings is 10.1; **12 is recommended** (2 GiB over that).
Run-to-run: the three repeats (1.8) agree with 818938 within 0.9 GiB; d2600_3p differs by 15.6 between runs (230.0 solo, 245.6 in 818763), so one class sets the margin.

**Limits.** (1) The fit is on 18 points of one build; P_B is not validated outside 2-1,304 M reads and 90-150 bp. (2) U / n: the late store is 150 B per unmapped read (229 at 150 bp). P_B has seen U / n up to 0.81 only at 93 M reads; a deep sample of another species (U / n 0.8 at 1 G reads) would add up to about 120 GiB that P_B does not contain.
(3) The 31 GiB device budget sets the RC and fallback thresholds; another budget or slot count moves them (not measured). (4) P_A with a margin of 11 (its in-sample worst 10.4) is a stand-in for the first seconds only: its leave-one-out worst is 27.8.

### 1.6 The in-flight figure (to be printed, not implemented)

With D, nAln, nOps, nSeq, U, the Solo keep and nw the ledger gives the item's peak to rmse 3.5 (not better than P_B on these 18 classes, but it contains no class average, so it does not depend on U / n or D / n). The engine writes the sizes only to the STAR Log.out. A line on stderr (the pool already reads the item's log for the level and waited lines):

`[engine_gpu] peakmodel: n=<reads> nw=<words> D=<distinct sequences> rc=<0|1>` once, right after the dedup (the "P3-MEM-03 ... distinct" moment: `step rc.dedup` is 35 s after "device locked" at d500s_3p, +71 s from the item's start, and 55 s after it at d2600_3p in 818528), with the fallback decision `recNeed` and the budget;
`[engine_gpu] peakmodel: records=<nRec> aln=<nAln> ops=<nOps> seqbytes=<nSeq> region_bytes=<R> hostrecs_bytes=<seq+blob+offsets>` when the records' counts exist (the "E8 records on the device" line, at `split.free`); `[engine_gpu] peakmodel: unmapped=<U> late_bytes=<L> solokeep_bytes=<K>` at the "E10 level 5" line (1-10 s before the level line).
After the first line the zone's fallback can be decided (D and the budget give `recNeed`); after the second the region is exact; after the third the whole ledger.

### 1.7 Which build

All of 818938 and of the three repeats ran `bin/t2-hostcut5` (v1.4.0 + hostcut rounds 1-2; patch against b21de2cf) with the switches at their defaults, which are all ON (code: `getenv(...)`, absent = on): `SINGLET_ENGINE_GPU_LEAN_DROP`, `_CLIP16`, `_READREP_FREE`, `_VERIFY_FREE`, `_AUX_COMPACT`, `_PRE_RELEASE`
(plus E22's `_EARLY_RELEASE`, which is older and on in v1.4.0). AUX_COMPACT acts only in RC (condition `RC && e10 && keepLate && Solo`): the eight classes from 299.6 M reads. PRE_RELEASE needs level 5, the ingest, the GPU pileup on and no soft-clip stream: the 16 engine-path classes. Both fallback classes use none of the engine-side cuts.

The launch build `54f57754` (`bin/multirun`, v1.4.0 + the two gpu-pool admission fixes) has **none** of them, so its peaks are higher and the P_B constants above do not hold for it. T2-hostmem.md has the before / after: per-sample mean item peaks (RSS sum, GiB; old build 816788 / new build 818882):
d2600 342.3 / 322.1; d1000 270.8 / 213.9; d700s 189.8 / 149.7; d500s 190.1 / 146.9; hgmm5k 109.6 / 103.7; d300_5p 102.3 / 85.0; d300_3p 90.5 / 74.3; nuclei2k 81.0 / 80.9; d100s 87.7 / 88.3; the smaller classes within 1.5; d2600 alone, cgroup peak (10 s sampler): 323.3 and 324.9 against 262.2. ESTIMATE, not measured:
the launch build's solo peaks above the baseline are the hostcut5 figures plus about 61 (d2600), 57 (d1000), 43 (d500s), 41 (d700s), 16-17 (d300), 6 (hgmm5k) and 0-1 GiB below 119 M reads. No solo recording with 1 Hz exists for 54f57754.

### 1.8 Repeats (jobs 818959, 818960, 818961)

Single items on g051, one H100, not exclusive, `--mem=300G`, `SINGLET_E16_TRACE=1`, STAR Log.out kept; all exit 0, `memory.events` all 0. Peak above 42.54 (818938 in brackets): d500s_3p 126.5 (126.3), d300_3p 54.2 (54.1), d100s_3p 58.2 (59.1). They gave the e16 marks of three classes and the Log.out lines.
The per-class engine figures of the 16 other classes come from `engine.c*.log` of mix 818882 (same build, same inputs): D, nAln, nOps, nSeq, U, region, Solo keep, mode / RC (`t2-peakmodel-classes.json`).

```
class            mode  rc    n(M)    D(M) nAln(M) nSeq(G)   U(M) region MiB  planes   late soloKeep
pbmc2m         normal   -     2.0    1.35    1.38    0.06   0.12       193    0.13   0.02   0.05
pbmc10m        normal   -    10.0    6.02    6.11    0.27   0.58       904    0.64   0.08   0.25
pbmc1k_cite_v3 normal   -    45.4   22.39   22.75    1.00   2.47      3704    2.93   0.35   1.09
pbmc5p_v1      normal   -    60.0   18.51   17.57    0.84   4.82      4015    3.88   0.71   1.35
hgmm1k_v3      normal   -    63.1   24.20   18.09    1.10  17.85      4479    4.06   2.60   1.46
pbmc1k_v3      normal   -    66.6   31.74   31.83    1.41   3.54      5334    4.29   0.49   1.59
hgmm1k_v2      normal   -    75.2   22.30   15.25    1.01  22.05      4813    4.84   3.21   1.68
neuron1k_v3    normal   -    92.9   34.83    4.45    1.58  75.67      6060    5.97  11.06   2.14
d100s_3p         lean   -   100.0   61.83       -       -      -         -       -      -      -   fallback, records need 35.07 GB
nuclei2k_v2      lean   -   118.7   79.81       -       -      -         -       -      -      -   fallback, records need 43.93 GB
d300_5p          lean  10   299.6   69.95   75.65    3.19  21.46     18437   19.39   3.16   6.55
d300_3p          lean  10   300.2   59.48   58.10    2.58  12.13     17305   19.36   1.66   6.48
hgmm5k_v3        lean  10   313.0   93.31   69.47    4.24  86.70     20204   20.16  12.65   7.01
d500s_3p         lean  22   500.5  167.00  160.53   12.14  21.80     42746   45.95   4.65  11.35
d700s_3p         lean  23   703.4  184.54  190.17    8.04  17.84     44284   44.98   2.42  15.54
d1000_3p         lean  31   999.8  289.89  301.91   11.99  54.15     64653   63.56   6.96  22.32
d1000s_3p        lean  32  1004.6  314.68  316.92   13.84  33.71     67185   64.93   4.86  22.61
d2600_3p         lean  41  1303.7  205.98  169.08    7.71 170.06     69199   83.02  23.76  27.72
```

## 2. Part 2: what an item can still allocate, by phase

Largest level (above the baseline) an item reaches from the three moments the pool sees; P is P_B (zone applied), m = 12; all in GiB.

```
sample             P |     F0   peak  F0-pk |     F1  M_lvl   F1-M |     F2 M_wait   F2-M
pbmc2m           7.0 |   19.0    4.3   14.7 |   19.0    2.3   16.7 |   18.1    2.3   15.8
pbmc10m          8.4 |   20.4    6.6   13.8 |   20.4    4.7   15.7 |   18.8    3.5   15.3
pbmc1k_cite_v3  14.7 |   26.7   15.7   11.0 |   26.7   10.5   16.1 |   22.0    6.8   15.2
pbmc5p_v1       17.3 |   29.3   16.4   12.9 |   29.3   12.3   17.0 |   23.3   11.3   12.0
hgmm1k_v3       17.8 |   29.8   19.0   10.8 |   29.8   16.5   13.3 |   23.5   11.1   12.5
pbmc1k_v3       18.4 |   30.4   21.1    9.3 |   30.4   15.0   15.4 |   23.8    9.2   14.6
hgmm1k_v2       47.8 |   59.8   20.3   39.5 |   59.8   16.4   43.4 |   38.5    8.4   30.1
neuron1k_v3     57.6 |   69.6   38.0   31.5 |   69.6   38.0   31.5 |   43.4    9.6   33.9
d100s_3p        64.2 |   76.2   59.1   17.1 |   76.2   56.5   19.8 |   46.8   16.0   30.8
nuclei2k_v2     72.3 |   84.3   62.4   21.9 |   84.1   62.4   21.8 |   50.8   10.4   40.4
d300_5p         59.9 |   71.9   61.7   10.3 |   71.9   44.9   27.0 |   44.6   20.3   24.3
d300_3p         59.9 |   71.9   54.1   17.8 |   71.9   46.5   25.3 |   44.6   22.3   22.2
hgmm5k_v3       62.1 |   74.1   65.7    8.4 |   74.1   63.0   11.1 |   45.7   22.6   23.1
d500s_3p       126.5 |  138.5  126.3   12.2 |  133.0   86.5   46.5 |   77.9   54.1   23.8
d700s_3p       130.6 |  142.6  128.8   13.8 |  136.6   92.4   44.2 |   79.9   59.9   20.0
d1000_3p       182.0 |  194.0  186.6    7.3 |  182.9  123.2   59.6 |  105.6   86.2   19.4
d1000s_3p      185.2 |  197.2  189.2    8.0 |  185.8  128.6   57.2 |  107.3   81.1   26.2
d2600_3p       235.6 |  247.6  230.0   17.7 |  231.1  196.3   34.8 |  132.5   52.1   80.4
smallest (rule - measured): F0 7.3, F1 11.1, F2 12.0 GiB (positive: no class is above its rule)
```
Rules: **F0 = P + m** at the start; **F1 = min(F0, 0.90 P + 7.1 + m)** from the "[engine_gpu] level" line; **F2 = min(F1, 0.50 P + 2.6 + m)** from the "[allsite] async: waited" line. The intercepts 7.1 and 2.6 are the smallest that cover the 18 measured later maxima at those slopes; at m = 12 the worst margin is +7.3 / +11.1 / +12.0,
the largest over-prediction 39.5 / 59.6 / 80.4 (hgmm1k_v2, d1000_3p, d2600_3p).

Facts behind the shape: the peak is at or just before the level line for the 300 M classes and d2600_3p (hgmm5k_v3 at -1 s, d2600_3p -2 s), so after that line the item still reaches 0.73-0.96 of its peak at 300 M (hgmm5k_v3 63.0 of 65.7), and 0.85 at d2600_3p (196.3 of 230.0; the bump and the hand-off pass). The deep RC classes reach 0.66-0.72 of their peak after the level line.
After the waited line the growth is at most 20.4 GiB (d1000_3p: 65.8 at the line, M_wait 86.2) and the later maximum is 0.15-0.65 of P (0.22 for d2600_3p). The pool's own RSS sum minus 17.65 is not usable as a per-item level: at the level line it is 228 GiB for d2600_3p against a true level of 150, and 77.5 against 46.8 for hgmm5k_v3 (the probes' shared pages and the region are counted
in every process); the rule therefore takes the cgroup's total, section 3.

The current code, for comparison (0.4 x est after the level line, min(that, 80) after the waited line; est = 61.8 + 0.2235 n[M]), is under the later maximum by 55.1 GiB after the level line (d2600_3p: 141.3 against 196.3; 27.0 nuclei2k_v2, 22.8 d100s_3p, 17.0 d500s_3p, 14.0 d1000s_3p, 10.3 hgmm5k_v3) and by 6.2 after the waited line (d1000_3p: 80.0 against 86.2; d1000s_3p 1.1), and over it by 13.8-31.2 elsewhere.

## 3. Part 3: the rule offline

Rule as prescribed: `free = min(node rule as today, job rule)`, job rule = memory.max - memory.current + 0.5 (file - shmem); `that = max(0, sum_i F_i - (cgroup anon+shmem now - idle baseline))`; admit when `free - that - F0(new) >= 40` or when nothing else runs. F0(new) = P + m (the margin is part of it; with P alone the decisions in 3b are the same).

**Limits of the replay (as in T2-pooladmit.md).** Only the head's wait is re-timed; the items in flight, MemFree, the cgroup series and the other pool are those of the real run (10 s series for the mixes, interpolated; 1 s for the single jobs). Cached - Shmem is modelled (MemTotal - MemFree - cgroup anon+shmem - K, K 17.1 and 11.3 GB; not in any sample);
file - shmem of the mix cgroup is memory.current - anon - shmem; memory.max of the mixes is taken as 700 GiB; the idle baseline of a mix (two pools, one shared server copy) is 60.0. A replayed wait that passes the real start is **censored** at it (after it the series contains the head itself), which is why the baseline here (16,354 / 16,130 s) is
below T2-pooladmit.md's uncensored 17,368 / 18,652. Nothing here says what a run under the rule would have done: no knock-on, no change of what starts when.

### 3a Safety on the recordings

At every second, S(t) = baseline + sum of F_i over the items in flight against M(t) = the largest cgroup anon+shmem from t until the next item starts (the set only shrinks); a violation is S < M. Baselines: 42.54 (818938, 818937, 818940), 60.0 (mixes).

| recording | seconds | violations m = 0 | m = 7 | m = 10.1 | m = 12 |
|---|---|---|---|---|---|
| 818938 (18 items one at a time) | 4,924 | 678 s, largest 4.7 (d1000_3p, +2,659 s) | 0 | 0 | 0 |
| 818937 (3 x d300_3p, 200 GiB) | 219 | 0 | 0 | 0 | 0 |
| 818940 (3 x d300_3p, 150 GiB) | 141 | 0 | 0 | 0 | 0 |
| 818763 (mix A1, 26 waits) | 11,408 | 832 s in 5 episodes: +2,006 s 261 s **10.0** (d2600_3p alone in flight: S 295.6, M 305.6); +7,841 s 216 s 4.6 (d1000_3p); +0 s 257 s 4.4 (d1000_3p); +3,063 s 22 s 3.2 (d1000s_3p); +1,710 s 76 s 0.3 (hgmm5k_v3) | 229 s, 3.0 (the d2600_3p episode) | 0 | 0 |
| 818882 (mix A1b, 22 waits) | 10,627 | 622 s in 3 episodes: +0 s 214 s 4.7 (d1000_3p); +7,459 s 201 s 2.4 (d1000_3p); +2,439 s 207 s 1.6 (d1000s_3p) | 0 | 0 | 0 |

Do solo peaks add up in a mix? The run's peaks: 818763 314.6 GiB at +2,265 s with d2600_3p and pbmc5p_v1 in flight: baseline + sum P = 312.9 (the sum is 1.7 below the run); 818882 308.4 at +6,474 s with d700s_3p (after its level line), d100s_3p and d500s_3p: baseline + sum P = 381.4 (71 above). The peaks of items in flight are mostly not simultaneous, but the single item can exceed its solo figure:
d2600_3p reached 245.6 above the baseline in 818763 (solo 230.0), d1000_3p 4.7 above P in both mixes (solo 186.6 against P 182.0).

### 3b The rule at the real starts of 818937 and 818940

```
(b) 818937 (--mem=200G): memory.max 200 GiB, F0(d300_3p) = 71.9 GiB (P 59.9 + m 12.0), reserve 40
   item 0 real start +5 s: nothing else runs: admitted
   item 1 real start +45 s: node 360.9 job 109.3 -> free 109.3; items still to allocate 32.3; F0 71.9; margin to the reserve -34.9 -> WAIT
   item 2 real start +85 s: node 317.1 job 64.0 -> free 64.0; items still to allocate 59.1; F0 71.9; margin to the reserve -107.0 -> WAIT
   counterfactual starts (rule applied, series rebuilt from the solo profile): +5 s, +129 s, +243 s
   counterfactual solo-overlay peak anon+shmem 96.5 GiB against memory.max 200; the recording: peak 157.1
(b) 818940 (--mem=150G): memory.max 150 GiB, F0(d300_3p) = 71.9 GiB (P 59.9 + m 12.0), reserve 40
   item 0 real start +4 s: nothing else runs: admitted
   item 1 real start +44 s: node 357.2 job 54.1 -> free 54.1; items still to allocate 27.0; F0 71.9; margin to the reserve -84.8 -> WAIT
   item 2 real start +85 s: node 328.9 job 24.4 -> free 24.4; items still to allocate 64.9; F0 71.9; margin to the reserve -152.4 -> WAIT
   counterfactual starts (rule applied, series rebuilt from the solo profile): +4 s, +128 s, +159 s
   counterfactual solo-overlay peak anon+shmem 96.6 GiB against memory.max 150; the recording: peak 147.4
```
Answer: in neither job would the rule have admitted items 1 and 2 at the times they started (the job term binds, the node term is 317-361 GiB). In the 200 GiB job it would have started them at about +129 and +243 s, in the 150 GiB job at about +128 and +159 s (counterfactual: the series rebuilt as baseline + the solo d300_3p
profile at the new offsets, MemFree and the file-cache part taken from the recording; an assumption, not a measurement). No kill is expected there (peak 96.5-96.6 against 150 / 200). The cost: the three d300_3p items fitted in 200 GiB (real peak 157.1 above zero, 114.6 above the baseline) but F0 x 3 + baseline = 258. With F0 = P alone (59.9) the margins are 10 GiB better and every decision is the same.

### 3c The 365 GiB lane with three workers (a PREDICTION from the recorded profiles)

Model: the current gate (`gateTry`): admit when `free - own - est >= 40`, free = MemFree + 0.25 (Cached - Shmem), est = 61.8 + 0.2235 n[M], own = sum over items in flight of max(0, charge - held) with charge x1.0 until the level line, x0.4 after it, min(that, 80) after the waited line, held = the pool's RSS sum minus 17.65 (from `memsamp.tsv`);
stagger 40 s after the start of an item with est >= 100; three workers; items admitted at the earliest second the inequality holds (alone = admitted). The node's free memory enters as `FE0` = free + the job's cgroup anon+shmem, so that free(t) = FE0 - (baseline + sum of the items' levels): measured 453.7 / 443.4 / 444.3 (818937 at its three starts) and 455.2 / 444.4 / 449.3 (818940) on g051,
347-380 on g052 in 818938 (after the harness's 35 s input copy had filled the cache). The true levels are the solo profiles overlaid at those start times; the limit is 365 (an OOM needs anon+shmem near it: 818940 was killed at 147.4 of 150).

| FE0 (GiB) | multisets of three classes (of 1,140) that the current gate admits together with an overlay peak above 365 | smallest by sum of solo peaks | the limit-aware rule (m = 12) admits any of them |
|---|---|---|---|
| 450-500 | **0** | | |
| 510 | 2 | d1000_3p + d500s_3p + d700s_3p (order d500s > d700s > d1000, starts 0 / 40 / 214 s; overlay 386.7 at +372 s) | 0 of 2 |
| 550 | 48 | d2600_3p + nuclei2k_v2 + nuclei2k_v2 (354.7; starts 0 / 0 / 1 s; overlay 379.4) | 0 of 48 |
| 650 | 144 | d100s_3p + d100s_3p + d2600_3p (348.2; d2600 first, then the two d100s at 40 s; overlay 370.5 at +172 s); also d500s x 3 (0 / 40 / 80 s, overlay 399.8), d1000_3p x 3 (445.6), d1000s_3p x 3 (455.3) | 0 of 144 |
| 740 | 161 | d100s_3p + d100s_3p + d2600_3p (same) | 0 of 161 |

So: on the node states measured in this task (FE0 444-472 GiB: g051 and g052 with 350-450 GB of other tenants' file cache), no triple is predicted; the threshold is FE0 about 505-510. Above it the first triples are the three deepest 150 bp / 90 bp classes, then d2600_3p with two mid-size items; with a
cache-free node (FE0 740) 161 multisets, among them three equal d500s_3p, d1000_3p or d1000s_3p items. "Smallest" is by sum of solo true peaks; by reads it is d500s_3p x 3 (1.5 G reads) at FE0 650-740.
**The confirmation job was not run.** It needs a node whose gate is not holding the items back (FE0 of about 540 or more); g051 is idle now with MemFree 351 GiB (Cached 376-395 GB at the starts of 818959-818961 on it: FE0 about 450), g052 is allocated: the gate would hold the third item and the job would show nothing.

### 3d Wait seconds of the mixes

Rule evaluated every second inside each wait window of the head item (22 and 26 logged waits), with the in-flight sets of both pools; the node term as today at cache share 0.25 or 1.0; job term with memory.max 700 GiB (it does not bind: memory.peak of 818882 is 407.3 GiB); F_i from the item's level and waited lines (`pool*.marks`, 1 s), P_B of its class.
"baseline" = today's inequality replayed the same way (censored); "cache 1.0" = today's inequality with the whole file cache counted free.

```
                       real    baseline   today, cache 1.0   NEW .25 (m=0)  NEW .25 (m=12)   NEW 1.0 (m=12)
818882 (22 waits)     19,195    16,354         1,778            6,471          11,688            371
818763 (26 waits)     20,570    16,130           620            6,326           9,703            354
```
(m = 10.1: 10,947 and 9,662 s; m = 7: 9,998 and 8,043.) Waits under 10 s in 818882: 0 baseline, 3 new .25 (m = 12), 19 of 22 new 1.0; censored at the real start: 3 of the new .25 waits. The wait that remains at cache share 0.25 is the node term (MemFree + 0.25 x cache is small: 12-219 GiB MemFree at the waits); at cache share 1.0 the new rule waits 371 / 354 s.
The deep heads wait longest under the new rule too (d2600_3p first wait 1,242 s of 1,259; d700s_3p 1,113 of 1,113). Total: the rule does not change the cache question of T2-pooladmit.md; it makes the in-flight charge true (11,688 against 16,354 at the same cache share, m = 12; 6,471 with no margin).

## Open points for the implementation

- Baseline: the idle baseline of a job is pool (17.65) + non-host server (24.7, +1.1 when used) = 42.5 GiB with one pool, 60.0 with two pools on a node sharing one server copy. It is the cgroup's anon+shmem when no item runs; the pool can measure it at its start (after the server is up, 20 s) or take the ancestors' `memory.stat` anon + shmem before the first item.
- The "that" term needs `anon` and `shmem` of the same job-level `memory.stat` that `avail_gb()` already reads (the ancestor with a finite memory.max, smallest headroom wins); page-locked host memory (E11 keep, the E2E region) is `shmem` there, not `file`.
- `avail_gb()`'s `0.5 x (file - shmem)` is a measured trade: in 818940 the job's non-shmem cache fell to 2-10 GiB before the kill, so the job cache buys tens of GiB at most.
- P needs the cDNA bases: not in the 96-byte header (fields in 1.4); the first block's counts (the head scan reads them) or the "[1fq-ingest] Scanned" line, +3 to +40 s after the start. Until then P_A (n, file size) with margin 11, or P_B with an assumed 90-150 bp; the two predictors differ by up to 10 GiB at 500 M reads.
- The zone (70-160 M reads) charges 48-72 GiB for items that peak at 20-62: the four classes in it are over-predicted by 5-27 GiB. The line of 1.6 (D after the dedup, +25-70 s) lets the pool drop the zone figure when `recNeed` is below the budget, and gives the figure itself when it is not.
- The thresholds (RC, fallback) depend on `SINGLET_ENGINE_GPU_MEM_GB` and `_SLOTS` of the lane (31 GiB, 2 slots here): the campaign's `h100x1` settings were not looked at; the constants hold for this harness.
- The margin: 12 GiB (smallest with zero violations on the five recordings: 10.1). d2600_3p varied by 15.6 GiB between two runs; classes beyond 1.3 G reads or other species (U / n) are outside the fit.
- Phase lines: the pool already sees "[engine_gpu] level" and "[allsite] async: waited"; the level line follows the peak by 1-10 s for the 300 M classes and d2600_3p (the peak is at the return of the prefill), by 11-48 s for the other RC classes (end of the lanes).
- The rule's `free` is a GiB figure of the cgroup while the pool's other quantities print "GB" of GiB; keep one unit.
- Today's per-item RSS sum (`held`) double counts the CR4 probe forks and the region: RSS sum of 818938 / true peak (corrected baseline) 1.11-1.58 from 100 M reads up (d2600_3p 325.6 / 230.0 = 1.42, d1000_3p 1.13, d300_5p 1.58) and 3-13 times at 2-10 M reads; T2-pooladmit.md's "RSS sum equals the true peak between 100 M and 1,000 M reads" is also an effect of the baseline. A rule on the cgroup's total does not need it.
- Not done: no change under src/ or include/; no job of other agents touched; no remote rm; nothing pushed.

## Jobs and data

| id | what | node | note |
|---|---|---|---|
| 818938 | solo profiles, 18 items one at a time (T2-pooladmit.md) | g052 | the data of 1.1, 2, 3a; read again with the corrected baseline |
| 818937, 818940 | 3 x d300_3p, 200 / 150 GiB (T2-pooladmit.md) | g051 | 3a, 3b |
| 818763, 818882 | mixes (T2-hostmem.md) | g052 | 3a, 3d; engine lines of 818882 for the per-class sizes |
| 818959 | d500s_3p, `SINGLET_E16_TRACE=1`, `--mem=300G` | g051 | exit 0, oom_kill 0, memory.peak 191.7 GiB (205,861,408,768 B), peak above baseline 126.5 |
| 818960 | d300_3p, same | g051 | exit 0, oom_kill 0, memory.peak 113.4 GiB (121,806,135,296 B), peak 54.2 |
| 818961 | d100s_3p, same | g051 | exit 0, oom_kill 0, memory.peak 127.9 GiB (137,350,062,080 B), peak 58.2, level 0 (fallback) |

Results: `$W/jobs/gpu-runtime/results/t2-peakmodel-818959-d500s`, `-818960-d300_3p`, `-818961-d100s` (`cg1hz.tsv`, `markers.tsv`, `e16.txt`, `engine.log`, `star_Log.out`, `pool0/`). The confirmation job of 3c was not submitted.
