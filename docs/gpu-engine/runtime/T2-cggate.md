# T2 cggate: host-memory admission on the job cgroup (`SINGLET_POOL_PEAKADMIT=1`)

Task `cggate` of throughput track 2 (branch `t2/cggate` from `gpu/throughput-2` d141971c). Data and the rule's derivation: T2-peakmodel.md; the problem: T2-pooladmit.md. Sizes are GiB (the
pool's older lines print "GB" of GiB; the lines added here print "GiB"). Everything marked ESTIMATE is not a measurement.

## 1. The rule as implemented

**Switch.** `SINGLET_POOL_PEAKADMIT` (default 0). With 0 the pool behaves as before: same admissions, same log lines, same files (the pool writes nothing new, the engine prints nothing new). With 1 and
`SINGLET_POOL_MEMADMIT` not 0, the pool looks for a job cgroup (below); without one it logs `peak admission (SINGLET_POOL_PEAKADMIT=1) is not available: ...; the current gate is used` and runs the old gate.
In mode 1 the pool exports `SINGLET_ENGINE_GPU_PEAKMODEL=1` to its items (only then).

**Constants** (one place: `include/singlet/gpu_pool_peak.h`, struct `Consts`, each overridable by the environment variable named in the last column; `fromEnv()`; the unit test uses the defaults).

| constant | value | meaning | variable |
|---|---|---|---|
| a, b, c | 6.604407719095382, 0.08168172883122637, 1.0588433857586883 | P_B = a + b n[M] + c bases[G] | `SINGLET_POOL_PEAK_A`, `_B`, `_C` |
| zone | [70, 160) M reads | fallback to STAR's loop possible, not visible before the start | `SINGLET_POOL_PEAK_ZONE_LO_M`, `_ZONE_HI_M` |
| fbC0, fbK, fbGibPerMD | 6.06, 0.49, 0.70 | zone: P = max(P_B, planes + 6.06 + 0.49 n[M]); fallback figure planes + 6.06 + 0.70 D[M] | `SINGLET_POOL_PEAK_FB_C0`, `_FB_K`, `_FB_GIB_PER_MD` |
| margin | 12 | added to every figure | `SINGLET_POOL_PEAK_MARGIN_GB` |
| level, waited | 0.9 P + 7.077160443572787; 0.5 P + 2.6432036829225236 | F1 = min(F0, level), F2 = min(F1, waited), all + margin | `SINGLET_POOL_PEAK_LEVEL_U`, `_LEVEL_V`, `_WAITED_U`, `_WAITED_V` |
| ledger | 6.06, 0.0238, 0.9 | L = planes + region + 0.9 x solo keep + late store + 6.06 + 0.0238 n[M] | `SINGLET_POOL_PEAK_LEDGER_A`, `_LEDGER_B`, `_LEDGER_KEEP` |
| n limit | 1,400 M reads | above it the fit is not used | `SINGLET_POOL_PEAK_NMAX_M` |
| fitted regime | `SINGLET_ENGINE_GPU_MEM_GB` 31 +- 1, `SINGLET_ENGINE_GPU_SLOTS` 2 +- 0 | see A | `SINGLET_POOL_PEAK_FIT_MEM_GB`, `_FIT_MEM_TOL_GB`, `_FIT_SLOTS`, `_FIT_SLOTS_TOL` |
| engine's fallback expectation | need = 4.67 + 0.492 D[M] GiB against the device budget | two-point fit to the two logged fallbacks (d100s_3p 35.07 at D 61.83 M, nuclei2k_v2 43.93 at 79.81 M) | in the engine (not overridable) |
| cache shares | node 0.25 (`SINGLET_POOL_CACHE_SHARE`, as before), job 0.5 | | `SINGLET_POOL_CG_CACHE_SHARE` |
| reserve, bounded wait | 40, 1800 s | unchanged | `SINGLET_POOL_MEM_RESERVE_GB`, `SINGLET_POOL_MEMADMIT_MAXWAIT_S` |
| baseline rule | settle 30 s, stable 10 s, range 0.5 GiB, first-start wait 120 s, pools expected 0 (unknown) | see D | `SINGLET_POOL_PEAK_SETTLE_S`, `_STABLE_S`, `_B_RANGE_GB`, `_B_WAIT_S`, `_NPOOLS` |

**A. The item's figure.** `P` = `peakP(n, bases)` of the header read count n (all runs of a sample, as the old estimate: sum over the inputs) and the cDNA bases: for each input, reads x (bases per read of the first block,
from `RunSet::scan_head(0)`, `bases_out / n_out`, the pipeline's own head scan on block 0). Zone rule inside [70, 160) M. Figure F0 = P + margin. The old estimate (61.8 + 0.2235 n, `SINGLET_POOL_ITEM_GB` overriding)
is used instead, with one pool.log line `item N: peak model not used (<why>): the old estimate, X GB, is the figure`, when: the item's n or bases cannot be read (an input without a `.1fq` header, a `.1fg2`, a first block
the head scan does not support or cannot read); n above 1,400 M; any of the six hostcut switches (`SINGLET_ENGINE_GPU_LEAN_DROP`, `CLIP16`, `READREP_FREE`, `VERIFY_FREE`, `AUX_COMPACT`, `PRE_RELEASE`) is set to anything but
unset, empty or `1` (`0` and CLIP16's `check` change what the host holds); `SINGLET_ENGINE_GPU_MEM_GB` unset or outside 31 +- 1; `SINGLET_ENGINE_GPU_SLOTS` not 2.
*Why these two.* The regimes (read chunks from about 144 M reads at 90 bp, fallback to STAR's loop in the zone) are set by the device budget, `memBudget()`: `SINGLET_ENGINE_GPU_MEM_GB` if set, else (device - 2 GiB - shared index) / slots.
The pool never touches CUDA, so with MEM_GB unset it cannot know the budget: not applicable. With MEM_GB set the budget does not depend on the slots, and I found no other code path in which the slot count changes what the host holds
(`SINGLET_ENGINE_GPU_ADMIT` acts only when MEM_GB is unset); all 18 profiles were measured at 2 slots with the host-side overlap of two items in the mapping phase, so the slot tolerance is 0 (override
`SINGLET_POOL_PEAK_FIT_SLOTS_TOL`). A budget within +-1 GiB moves the read-chunk threshold by about 3 % (ESTIMATE from the formulas of T2-peakmodel.md 1.2): inside the zone's margin.

**B. What an item will still reach, by phase.** F0 before the engine's `[engine_gpu] level` line, F1 after it, F2 after `[allsite] async: waited` (`phaseFigure`; the old-estimate items keep today's 1.0 / 0.4 / min(that, 80)).

**C. Figures from the engine** (`SINGLET_ENGINE_GPU_PEAKMODEL=1`; stderr of the item, read by the pool as it reads the level line):
1. `[engine_gpu] peakmodel: n=<reads> nw=<words> D=<distinct> rc=<0|1> bases=<cDNA bases> budget_gib=<device budget> need_est_gib=<records phase need estimated from D> fb=<0|1>` right after the dedup (both paths, at `memMark("index+distinct")`).
   `fb` = not read-chunked and need_est > budget. The doc asked for the fallback decision here; the engine takes it later (E8's records, `recNeed` needs the blob and sequence sizes that do not exist at the dedup), so `fb` is an estimate from D alone.
   For an item in the zone: `fb=0` -> P = P_B (linear, from the engine's n and bases) + margin; `fb=1` -> P = planes + 6.06 + 0.70 D[M] + margin. (Outside the zone `fb=1` raises P to the fallback formula, never lowers.)
2. `[engine_gpu] peakmodel: records=<nRec> aln=<nAln> ops=<nOps> seqbytes=<nSeq> region_bytes=<R> hostrecs_bytes=<seq+blob+offsets>` where the records' counts exist (both E8 builders); R by the formula of T2-peakmodel.md 1.3.
3. `[engine_gpu] peakmodel: unmapped=<U> late_bytes=<L> solokeep_bytes=<K>` at the "E10 level" line of the prefill; K = 20 n + 8 records (the "Solo inputs kept ... GB" figure, 1e9).
4. (added) `[engine_gpu] peakmodel: fallback=1 D=<D>` when E8's records are not built (the real decision, at +86 s for d100s_3p): the figure is raised to the fallback formula if the first line missed it.
After lines 1-3 the pool computes the ledger L (`ledgerL`, planes from the engine's n and bases, the solo keep as the logged 1e9 figure the fit used, 0.9 x it). If L + margin is above the figure of the moment, the figure is raised to it
for the phase it is in, and P is raised to L (so the later phases follow from L). Never lowered by the ledger. Every change of a figure is one line: `peak figure: item N a -> b GiB (P x, reason)`.

**D. Accounting on the job cgroup.** The job cgroup is the one `avail_gb()` finds (ancestor with a finite `memory.max` and the smallest headroom; cgroup v2 only; none, or no `anon` in `memory.stat` -> mode 1 not available).
AS = anon + shmem of its `memory.stat`. What the items in flight of the whole job can still allocate = `max(0, sum of the figures of all items in flight of all pools - (AS - B))`; each pool publishes its figure sum in its gate file
(`/dev/shm/singlet-pool-mem-<job>-<uid>/<pid>`: the old five fields, then `figSum ready lastEvt`; `rem` carries figSum so an old-gate pool in the same job reads something sensible). With B unknown nothing counts as held (the sum of the figures: too
high is safe).

*B (the idle baseline).* Too low would overstate what items hold; it was 20.4 GiB too low in T2-pooladmit.md because the pool reads "ready" at +5-7 s, but the non-host server (a child it forks first, 24.7 GiB of page-locked index) says
"listening" at once and loads for another 14.3 s (`[nhsrv] resident: ... in 14.33 s` in `nhsrv.log`, 818938). What the pool knows: its own genome is resident when it is ready; the server is loaded only when `report/nhsrv.log` has `[nhsrv] resident`
(with no warm-start files it loads on the first request, which prints the same word; no server or an ended one: nothing to wait for); the other pools' state through their gate files. The rule (`pkMeasure`): B is taken only when ALL hold:
(1) this pool's `ready` (genome resident and server loaded) and every other live pool's `ready` (a pool announces itself in the gate directory at its start, `ready` 0, from a heartbeat thread, so a pool that is still loading is seen);
(2) no item of any pool is in flight (own count and the others' published counts, under the gate flock);
(3) no item of any pool started or ended in the last 30 s (`lastEvt`): an ended item's memory is still being returned (the 1 s after an end are the only point T2-peakmodel measured, and the pages come back for seconds);
(4) the cgroup's AS has been flat over the last 10 s (at least 6 samples, 1 Hz, max - min <= 0.5 GiB).
B = the largest sample of that window. Stored once for the job in `<gate dir>/baseline` (`B nPools time`), under the flock; only ever raised by a later idle moment (the gpu-holder an item starts is then included); it is invalid, and the
items count as holding nothing, while more pools are alive than at its measurement (a pool that appeared later brought its genome and server). A baseline left by an earlier run in the same job is removed when the pool starts and no other pool is alive.
The very first start of a job waits, up to 120 s after the pool is ready, for B (the only moment it can be measured), then goes ahead with B unknown. `SINGLET_POOL_PEAK_NPOOLS` (default 0 = unknown) makes the measurement wait for that
many pools. Residual risk: a pool that has not yet created its gate file at the moment of the measurement (its file appears within the first second of its start; a sibling needs at least 15 s to get its server resident).

**E. Free memory = min(node term, job term).** Node term exactly as before: (MemFree + `SINGLET_POOL_CACHE_SHARE` (0.25) x (Cached - Shmem)). Job term: memory.max - memory.current + `SINGLET_POOL_CG_CACHE_SHARE` (0.5) x (file - shmem) of the job cgroup.
An item starts when nothing else is in flight in the job (own and other pools, as before), or when `free - (what the items in flight can still allocate) - figure(new item) - (the older waiting item of another pool, as before) >= 40`.
Skip-ahead (16 items behind the head, 900 s grace) is the old logic with `pkTry`. The bounded wait stays for the node term: after 1800 s an item that only the node term refuses is started (log: `the wait is bounded; only the node term refused`). When the
job term refuses (alone or with the node term) the pool keeps waiting, logs once `no forced start`, and the head starts when the items in flight have ended (then nothing else is in flight and the "alone" rule admits it). The wait line
`memory admission: item N waits (figure ...; free: node term X, job term Y GiB; the items in flight can still allocate ...; projected node P, job Q; refused by the <node|job|node and job> term)` prints both terms; it is printed again when the refusing term changes.
The stagger (`SINGLET_POOL_STAGGER_S`) still uses the old estimate of the item last started, as before.
With `SINGLET_POOL_MEMSAMPLE` set, `memsamp-cg.tsv` (beside `memsamp.tsv`) gets, each period: `t AS B figures can_still_allocate node_free job_free items_job B_valid`. Whenever AS - B exceeds the sum of all figures
the pool logs (at most every 30 s) `model exceeded: job anon+shmem a - baseline b = c GiB held by the items, above the sum of their figures d by e GiB ...` and the run ends with `peak admission: model exceeded in X s of the refreshes (largest excess Y GiB)`.

**F. `--chimeric` items (orchestrator's addendum, mode 1 only).** The peakmodel recordings had no `--chimeric`; after the engine such an item forks a CPU STAR that loads the genome index privately (T2-chimpass.md, jobs 819008 / 819009: the job
cgroup's anon+shmem rises by 16.6 / 16.7 / 17.1 GiB at 8 threads over the level at the pass's start for GSM6250953 / pbmc5p_v1 / d300_5p, 18.0 at 16 threads, 19.3 at 27). If the item's own argument list (the pool's `--args` plus the queue line's extra arguments, token `--chimeric`;
never the environment) contains it, every phase figure is `max(F, 0.50 P + 2.6432 + chimGb + margin)` = `max(F, 0.50 P + 22.1 + margin)` with `SINGLET_POOL_PEAK_CHIM_GB` (default 19.5, the largest measured rise 19.3 plus 0.2), also at admission (F0). `phaseFigure(P, phase, k, chimeric)`; the pool logs `--chimeric: floor 0.5 P + 22.1 + margin` on the item's line.
**The basis is three samples, one thread count (8) each for two of them (pbmc5p_v1, d300_5p; GSM6250953 was measured at 8, 16 and 27).** Mode 0 does not know the term.
The three recordings with their real n and cDNA bases (the `[1fq-ingest] Scanned` lines of the jobs: pbmc5p_v1 60.000 M reads / 5.460 G bases, GSM6250953 121.366 M / 18.205 G, d300_5p 299.628 M / 27.266 G); level = item level above the 42.5 GiB one-pool baseline during the pass (T2-chimpass.md); margin 12:

| sample | P (start) | P after the engine's line 1 (no fallback) | F2 without the term | floor / F2 with the term (P after line 1) | measured level in the pass |
|---|---|---|---|---|---|
| pbmc5p_v1 | 17.3 | 17.3 | 23.3 | 42.8 | 25.8 |
| GSM6250953 | 76.7 (zone, fallback figure) | 35.8 | 53.0 (from P 76.7) / 32.5 (from 35.8) | 72.5 (from P 76.7) / 52.0 (from 35.8) | 28.0 |
| d300_5p | 59.9 | 59.9 | 44.6 | 64.1 | 33.7 |

(the orchestrator's guessed 31.6 / 40 / 56.4 came from guessed bases.) The figure includes the 12 GiB margin: without it the floors are 30.8 / 40.0 (from 35.8) / 52.1 against 25.8 / 28.0 / 33.7 measured: above the measurement by 5.0 / 12.0 / 18.4. For pbmc5p_v1 F2 without the term (23.3) was below
the measured 25.8 (the pass, not the engine, is its peak); with it 42.8. GSM6250953 is in the zone: with the first line saying no fallback (it ran the engine path) the pool charges 52.0 after the waited line; before that line the zone's fallback figure applies (F0 88.7).

Results that predate the term: everything in sections 3-6 below ran on `bin/t2-cggate-2` (jobs 818978 ... 819016), none of whose items carried `--chimeric` (the harness's PROD=1 does not add it): the term changes none of those figures. The build with the term is `t2-cggate-4` (jobs 819024 for -3: its ctest failed only on
a 1e-6 tolerance in the new test, fixed; 819029 for -4).

## 2. Offline replay and the C++ (test 2)

Test 1 (`tests/cpp/test_gpu_pool_peak.cpp`, ctest `test_gpu_pool_peak`, passed in the build log of bin/t2-cggate-2: "100% tests passed out of 188", `ctest exit 0`): the 18 classes of `t2-peakmodel-predictor.json` (n, bases of the doc's table, one decimal), P_B and the zone
rule within 0.15 GiB (the doc's n and bases are rounded; measured worst difference of the C++ formula from the json: 0.042 GiB, computed here in Python from the same constants), F0 / F1 / F2 against the section 2 table of T2-peakmodel.md within 0.06 (its print precision; 0 of 54 off),
the fallback figures of d100s_3p / nuclei2k_v2 (58.5 / 70.0), the ledger of d500s_3p (120.5) and d2600_3p (236.4), the engine's fallback expectation on the four D values, `fitValid`'s range and regime rules.

Replay and C++, constant by constant (`t2-admit-replay2.py` takes the json; the C++ has the same numbers in `Consts`): a, b, c, zone [70, 160), fallback 6.06 / 0.49 / 0.70, level 0.9 x P + 7.077160443572787, waited 0.5 x P + 2.6432036829225236, reserve 40, bounded wait 1800 s,
stagger 40 s, node cache share 0.25, job cache share 0.5: **equal**. Differences: the margin (the json says 7, the replay's `--margin` default is 7, the C++ 12 as the doc recommends; the replay was run with `--margin 12`); the baseline (the replay takes 60.0 / 42.54 as a constant, the C++ measures B, section 1 D);
the replay re-times every wait with the bound after 1800 s whatever the term, the C++ does not force the start when the job term refuses; the replay has no first-start wait, no figure changes from the engine's lines (zone drop, ledger raise), and a fixed worker count 3.
`t2-admit-replay2.py` could not be re-run end to end here: its inputs (`pool*.marks`, `mem2.tsv`, `mon.tsv`, `items.tsv`, `samples.all` derived from the recordings of 818763 / 818882) are not on Clipper under results/ (only the raw `rt-node-*` directories) and not in the repository; it was copied to
`$W/jobs/gpu-runtime/t2-admit-replay2.py` with the json and fails at `pool0.marks`. The hand-computed cases above stand in for it.

## 3. Build and identity gate (test 3)

Build `t2-cggate-2` = job 818973 (`BASE=b21de2cf`, patch `patches/t2-cggate-2.diff` sha256 fdcd2999...): `ctest exit 0`, "100% tests passed out of 188" (`test_gpu_pool_peak` among them), sm_70 sm_75 sm_90.
(`t2-cggate` = job 818972 failed to compile: `poolpeak::` is `singlet::poolpeak::`; a namespace alias was added; nothing else was wrong; its `bin/t2-cggate/` has no binary.)
Identity gate, RTX 8000 (g005 / g008, started at once; the command of the brief, `BINDIR=bin/t2-cggate-2`):

| job | switches | result |
|---|---|---|
| 818978 | all default (off) | 8 lines `VERDICT <sample> cpu:gpu IDENTICAL (except E11's run-dependent files; out_equiv 91-92/5-7/4)` (pbmc2m, pbmc10m, pbmc1k_v3, pbmc1k_v2, pbmc1k_cite_v3, pbmc5p_v1, neuron1k_v3, nuclei2k_v2); no fallback line in any fallbacks.txt; no `peakmodel` line anywhere |
| 818980 | `SINGLET_ENGINE_GPU_PEAKMODEL=1` added to `--export` | the same 8 verdict lines; fallbacks.txt without a fallback line; `engine_lines.txt` of every sample carries the three peakmodel lines |

(The wording "except E11's run-dependent files" is the gate's standing text for all gate jobs.) On RTX 8000 the gate has no `SINGLET_ENGINE_GPU_MEM_GB`, the budget is 42.30 GiB, nuclei2k_v2 does not fall back. The engine's first line for it said `fb=1` (need_est 43.94 against 42.30); the
records were then built (the real need is below the estimate by more than 1.6 GiB here): the two-point fit errs towards the fallback, i.e. conservative (the pool would have charged the fallback figure for an item that did not fall back). The lines of the other samples:
neuron1k_v3 `n=92902231 D=34691779 rc=0 budget_gib=42.30 need_est_gib=21.74 fb=0`, pbmc1k_v3 `D=31741646 ... need_est_gib=20.29 fb=0`.

## 4. Limit jobs, mode 1, one H100 (test 4; `t2-cggate-cg.sbatch`, three `d300_3p`, 3 workers, `SINGLET_POOL_PEAKADMIT=1`, `MEMSAMPLE=2`, 1 Hz sampler; node g051; mode 0 for reference: 818937 200G all fitted, peak 157.1; 818940 150G, two of three killed)

| | 818979 (`--mem=200G`) | 818981 (`--mem=150G`) |
|---|---|---|
| oom_kill / memory.events | 0 / `low 0 high 0 max 0 oom 0 oom_kill 0 oom_group_kill 0` | the same |
| memory.peak | 131,971,911,680 B (122.9 GiB) | 131,910,066,176 B (122.9 GiB) |
| pool wall | 407 s | 409 s |
| idle baseline B | 42.43 GiB at +18 s (job anon+shmem 42.07-42.43 over 10 s, 1 pool, server resident); the first start waited for it | the same, 42.43 GiB |
| start of item 0 / 1 / 2 (s after the pool's first line) | +18 / +156 / +280 (item 1 started when item 0 ended: 138.6 s wall; item 2 when item 1 ended: 123.7 s) | +18 / +157 / +280 |
| waits | 2 (items 1, 2), 182 s waited in all; both refused by the **job term** | the same, 182 s |
| wait lines | `item 1 waits (figure 72 GiB; free: node term 352, job term 103 GiB; the items in flight can still allocate 26 GiB (figures 72, job anon+shmem 88, baseline 42.4) ...; projected node 254, job 5; refused by the job term)` | `item 1 waits (figure 72 GiB; free: node term 353, job term 55 GiB; ... can still allocate 28 GiB ...; projected node 254, job -45; refused by the job term)` |
| figure changes of an item | 71.9 (F0) -> 71.9 after the level line (ledger 56.9 GiB, not above) -> 44.6 after the waited line | the same |
| model exceeded | `peak admission: model exceeded in 0 s of the refreshes (largest excess 0.0 GiB)` | the same |

Engine lines (item 0 of 818979, the same in all items): `n=300223349 nw=2 D=59480329 rc=1 bases=27164540368 budget_gib=31.00 need_est_gib=33.93 fb=0` / `records=59480328 aln=58101322 ops=84507716 seqbytes=2577968087 region_bytes=18123335351 hostrecs_bytes=10754202009` / `unmapped=12128581 late_bytes=1787643871 solokeep_bytes=6480309604`.
The ledger for d300_3p is L = 56.9 GiB against P_B = 59.9 (measured solo peak 54.1).
Replay prediction of T2-peakmodel.md 3b (starts about +5, +129, +243 s for 200 G; +4, +128, +159 s for 150 G, counterfactual from the solo profile): measured +18, +156, +280 s (200G); +18, +157, +280 s (150G). Item 1 and item 2 start 27 and 37 s later than predicted: 13 s of that is the
first-start wait for the baseline, the rest is item wall (138.6 s for the first item against 124 s in the solo profile). For 150G the replay admitted item 2 at +159 s (overlapping item 1); the pool did not: at that moment the job term was 46 GiB free with 21 GiB still to allocate by item 1, 72 for the new one (projected -47).
The three items ran one after the other in both jobs; the mode-0 200G job had them all in flight (peak 157.1 GiB above zero) and the three fitted.

## 5. 365 GiB lane, one H100, three workers (test 5; 818982, node g051, three `d500s_3p`, `--mem=365G`, mode 1)

| | |
|---|---|
| oom_kill / memory.events | 0 / `low 0 high 0 max 0 oom 0 oom_kill 0 oom_group_kill 0` |
| memory.peak | 328,990,703,616 B (306.4 GiB) of 365 |
| pool wall | 1159 s; items 673.5 / 653.5 / 463.6 s wall, all exit 0 |
| B | 42.43 GiB (+12 s, window 42.07-42.43) |
| starts (s after item 0's start) | item 0 +0, item 1 +40 (the old stagger), item 2 +674: admitted when item 0 ended (waited 593 s) |
| the wait | `item 2 waits (figure 138 GiB; free: node term 234, job term 152 GiB; the items in flight can still allocate 115 GiB (figures 277, job anon+shmem 204, baseline 42.4) ...; projected node -20, job -102; refused by the node and job term)`, later `... node term 285, job term 208 GiB; ... 62 GiB (figures 156 ...); projected node 85, job 7; refused by the job term`; after item 0 ended `admitted after 593 s (projected node 161 GiB, job 85 GiB)` |
| figures | F0 138.5 (P 126.5 + 12) -> 132.9 after the level line (ledger 120.5, not above) -> 77.9 after the waited line |
| model exceeded | 0 s, largest excess 0.0 GiB |

Three d500s_3p solo peaks plus the baseline are 3 x 126.3 + 42.5 = 421 GiB, over 365. The pool kept two in flight (peak 306.4) and started the third when the first ended.
**Mode 0 on the same triple: not run.** The probe job 818983 (`t2-cggate-probe.sbatch`, two minutes on g051 after the mode-1 run, nothing else of mine on the node): MemFree 341-343, Cached 399, Shmem 0 GiB, so today's `free` = MemFree + 0.25 x (Cached - Shmem) = 441-442 GiB (cache share 1.0: 740-742),
against the 505-510 the doc says today's gate needs to admit all three; skipped as the brief says. (The mode-1 run itself, 20 min earlier on the same node, started from MemFree 342 / Cached 399.) No kill in mode 0 was therefore observed and none is claimed.

Gate of the build with the chimeric term (`t2-cggate-4`, job 819029: `ctest exit 0`, 188 of 188), switches default, RTX 8000: job 819033, 8 lines `VERDICT <sample> cpu:gpu IDENTICAL (except E11's run-dependent files ...)`, no fallback line.
