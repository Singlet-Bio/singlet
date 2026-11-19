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

## 2. Offline replay and the C++ (test 2)

Test 1 (`tests/cpp/test_gpu_pool_peak.cpp`, ctest `test_gpu_pool_peak`, passed in the build log of bin/t2-cggate-2: "100% tests passed out of 188", `ctest exit 0`): the 18 classes of `t2-peakmodel-predictor.json` (n, bases of the doc's table, one decimal), P_B and the zone
rule within 0.15 GiB (the doc's n and bases are rounded; measured worst difference of the C++ formula from the json: 0.042 GiB, computed here in Python from the same constants), F0 / F1 / F2 against the section 2 table of T2-peakmodel.md within 0.06 (its print precision; 0 of 54 off),
the fallback figures of d100s_3p / nuclei2k_v2 (58.5 / 70.0), the ledger of d500s_3p (120.5) and d2600_3p (236.4), the engine's fallback expectation on the four D values, `fitValid`'s range and regime rules.

Replay and C++, constant by constant (`t2-admit-replay2.py` takes the json; the C++ has the same numbers in `Consts`): a, b, c, zone [70, 160), fallback 6.06 / 0.49 / 0.70, level 0.9 x P + 7.077160443572787, waited 0.5 x P + 2.6432036829225236, reserve 40, bounded wait 1800 s,
stagger 40 s, node cache share 0.25, job cache share 0.5: **equal**. Differences: the margin (the json says 7, the replay's `--margin` default is 7, the C++ 12 as the doc recommends; the replay was run with `--margin 12`); the baseline (the replay takes 60.0 / 42.54 as a constant, the C++ measures B, section 1 D);
the replay re-times every wait with the bound after 1800 s whatever the term, the C++ does not force the start when the job term refuses; the replay has no first-start wait, no figure changes from the engine's lines (zone drop, ledger raise), and a fixed worker count 3.
`t2-admit-replay2.py` could not be re-run end to end here: its inputs (`pool*.marks`, `mem2.tsv`, `mon.tsv`, `items.tsv`, `samples.all` derived from the recordings of 818763 / 818882) are not on Clipper under results/ (only the raw `rt-node-*` directories) and not in the repository; it was copied to
`$W/jobs/gpu-runtime/t2-admit-replay2.py` with the json and fails at `pool0.marks`. The hand-computed cases above stand in for it.
