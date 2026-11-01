# E7: convergent warps in phase 1 (seed search and window creation)

Branch `gpu/fable/seeds` (base: `gpu/fable/compact` with E5's defaults on). Code: `bench/gpu/m5/stitch.cuh`,
`bench/gpu/m5/m5.cu`; jobs: `bench/gpu/m5/jobs/e7_compile.sbatch` (every variant built side by side, ptxas -v kept) and
`bench/gpu/m5/jobs/e7_run.sbatch` (truth regenerated in the job, baseline and every variant in one job, identity for each,
then Nsight Compute on lane A's phase-1 kernels).


## Summary

- **Gate not met.** Best exact variant, flatqs8b (`-DM5_SEEDFLAT -DM5_P1SPLIT -DM5_SEEDSMEM -DM5_P1MINB=8 -DM5_P1WPB`):
  lane A phase 1 **78.9 ms vs 100.1 ms = 0.79x** (gate 0.5x), lane M 0.79x, lane B 0.79x; M5 **2.82 -> 3.02 M distinct
  reads/s (+7.1%)**. Identity 1,991,926 / 1,991,926 in every row of every run (final run 796044, commit 3040da0).
- Warps do **not** converge: active threads per instruction 3.38 in the best seed kernel vs 3.77 in the baseline k_p1
  (stepping one memory operation per lane per iteration: 2.2-3.9). The 0.79x comes from persistent warps with lane
  refill (no batch tail) and occupancy (64 registers, iterator in shared memory: 8 blocks per SM, 47.8% vs 27.1%
  achieved), i.e. more memory chains in flight in a latency-bound phase.
- Did not help: the flat iterator thread per read (1.28x), the 2-bit packed compare (register blowup, no gain at equal
  occupancy), stepped calls (1.16-6.5x), windows fused into the seed kernel, prefetch of the binary search's next SA
  words, out-of-line compare, register caps with spills. Phase 1 once with lane M's capacities would save at most 37 ms
  (baseline kernels) or 14 ms (flat kernels).

## Question

Phase 1 (`k_p1`: STAR's seed search, then the windows from the anchors) is ~40% of lane A (E5: 99.5 of 244 ms) and runs
one thread per read. STAR's seed loop is a nest (pieces x 2 directions x Nstart starts x a while loop over Lmapped, plus
the seedSearchLmax call), and each `maxMappableLength2strands` call runs a data-dependent binary search over the SA with
long `compareSeqToGenome` byte loops, inlined at five call sites, each with four direction cases. Lanes of a warp sit in
different iterations of different loop levels and at different inlined copies, so most of the warp idles at the nest's
convergence barriers (E1: ~2 active threads per warp instruction in phase 1). Hypothesis: flattening the nest into an
iterator, so that every iteration of ONE warp-uniform loop makes exactly one `maxMappableLength2strands` call per active
lane at one call site, and keeping lanes busy with persistent warps that refill finished lanes, raises the active lanes
per instruction and cuts phase 1. Gate: identity 100% and lane A's phase 1 <= 0.5x baseline (target 0.3x).

## Design as built

All variants are compile-time macros; all off = the baseline (E5's defaults: compact state in lane A, `--bin` order etc.
unchanged). Each macro implies the ones it needs.

| macro | what it changes |
|---|---|
| `M5_CMPU` | `compareSeqToGenome`'s four direction cases (dirR x dirG) as one loop with strides: the read runs forward or backward through `Read1[0]` or its complement `Read1[1]`, the genome forward from `SAstr + L` or backward from `nGenome - 1 - SAstr - L`; the same bytes compared in the same order, `compRes` as in the case it replaces |
| `M5_SEEDFLAT` | phase 1 with flat loops (implies `M5_CMPU`): `p1Setup` (read set-up, `qualitySplit`, resets), `seedNext` (an explicit state machine `SeedIt` over STAR's nest: `SI_PIECE`, `SI_START`, `SI_WHILE`, `SI_LMAX`, `SI_NEXT`) that stops at each pending `maxMappableLength2strands` call with its `Shift`/`seedLength`/`iDir`/piece, `seedCall` (the single call site) and `seedAfter` (`flagDirMap`, `Lmapped += L`); `p1Windows`: the markers, then ONE loop over every SA entry of every anchor (`p1Entry`: one entry, the donor/acceptor split of sjdb inserts as a two-iteration loop around ONE `createExtendWindowsWithAlign` call site), the flanks, `nWall`, `nEntries`. `k_p1` runs `mapPhase1Flat` (thread per read) |
| `M5_P1SPLIT` | phase 1 as two kernels: `k_p1s`, persistent warps (grid = SMs x resident blocks) in which every iteration each lane holding a read makes that read's next seed call; a lane whose read is finished (no more calls, or overflow) takes the batch's next read through a warp-aggregated atomic counter (`__ballot_sync` of the lanes that need a read, one `atomicAdd` by the leader, `__shfl_sync`, rank by `__popc`), so a warp keeps 32 reads in flight until the batch runs out; then `k_p1w`, the windows, thread per read |
| `M5_P1WPERS` | the windows kernel persistent too (`k_p1wp`): `p1Windows` as a resumable iterator (`WinIt`: one anchor SA entry, or one window's flanks, per step), lanes refill as in `k_p1s` |
| `M5_G2` | `compareSeqToGenome` on a 2-bit genome text, 32 bases per step: `G2` (genome 2 bits per base, 0.8 GB, positions -1024 .. nGenome+1024) and `NB` (1 bit per 64-base block: some genome byte > 3 in it; 6 MB, L2-resident), built on the device at start-up; `Read1[0]` packed forward and reversed per read in `ReadBuf`. A window of 32 comparisons whose genome positions touch an `NB` block is compared byte by byte as before; otherwise XOR of the read and genome words (complemented / 2-bit-reversed for the strand), first differing unit by `__ffsll` |
| `M5_SEEDSTEP` | (round 2) `k_p1s` steps each read's current `maxMappableLength2strands` call **one memory operation at a time**: `MmIt` holds the call's state (the iDist pass, the SAi lookup loop, `maxMappableLength`'s binary search, both `findMultRange` searches), `mmOp` performs the pending operation (an SAi lookup or one `compareSeqToGenome`, the only call site), `mmAdvance` consumes its result and runs the control flow to the next operation; a complete call runs `storeAligns` for its passes and moves the seed iterator on. A warp iteration is one memory operation per lane |
| `M5_STEPSMEM` | (round 3) `MmIt` in shared memory (128 x 200 B per block) instead of registers |
| `M5_P1FUSE` | the windows in `k_p1s` (`p1Windows` at one site, in the iteration a read's seeds complete); no windows kernel |
| `M5_P1MINB=n` | `__launch_bounds__(128, n)` on the phase-1 kernels (register cap) |
| `M5_CMPNI` | `compareSeqToGenome` out of line: one copy instead of five inlined ones |
| `M5_SEEDSMEM` | (round 4) the seed iterator `SeedIt` of `k_p1s` in shared memory (56 B x 128 = 7 KB per block, ptxas: 7168 B smem) instead of registers, so that a 64-register cap fits without spills |
| `M5_PREF` | (round 4) binary-search prefetch: while a lane compares at the midpoint of `maxMappableLength`'s or `findMultRange`'s search, the SA words of both possible next midpoints are prefetched into L2 (`prefetch.global.L2`) |
| `M5_PREF2` | (round 4) `M5_PREF`, plus both candidate SA values loaded (`ld.global.nc`) and the genome lines of both next compares prefetched |
| `M5_PREF3` | (round 4) `M5_PREF`, plus the SAi words of passes iDist >= 1 prefetched before pass 0 of `maxMappableLength2strands` (built in 796029, not run) |
| `M5_P1WPB` | (final) the persistent windows kernel `k_p1wp` in lane B only (lanes A and M keep the thread-per-read `k_p1w`) |
| `--p1probe` (run-time) | phase 1 alone over every read with lane A's and with lane M's capacities (timing, overflow count, reads beyond lane A's window capacity): the cost of running phase 1 once with lane M's window capacity |

### Exactness

Identity is checked byte for byte against STAR for every distinct read in every run; the arguments why it holds:

- **storeAligns order.** `storeAligns` inserts into the piece list `PC` in an order-sensitive way (ties broken by
  arrival), so the sequence of `maxMappableLength2strands` calls per read must be STAR's. `seedNext` is a mechanical
  transcription of the nest: the same loop bounds (`Nstart`, `Lstart` per piece), the same conditions (`flagDirMap ||
  istart > 0`; `istart * Lstart + Lmapped + seedMapMin < pieceLength`), the same `Shift`/`seedLength` expressions, the
  seedSearchLmax call after the while loop of each `(iDir, istart)`, and `flagDirMap`/`Lmapped` updated by `seedAfter`
  only for while-loop calls. Each read's calls therefore happen in STAR's order with STAR's arguments; interleaving with
  other reads' calls (other lanes) does not matter because every read has its own `Work`.
- **iDist passes.** `maxMappableLength2strands` itself is unchanged: its `nD = min(pieceLength, gSAsparseD)` passes, the
  `maxL + iDist == maxLbest` test and the `storeAligns` calls in iDist order run inside the one call, as before.
- **compareSeqToGenome.** `M5_CMPU` compares the same byte pairs in the same order and returns the same index and
  `compRes` (for `dirG == false` the case's `!(s > g || g > 3)`). `M5_G2`: within a 32-unit window, the first index where
  the 2-bit codes differ is the first index where the bytes differ provided every genome byte in the window is 0..3
  (read bases in a piece are 0..3 by `qualitySplit`); windows whose genome positions include any byte > 3 (N, padding,
  chromosome spacers) are compared byte by byte with the `M5_CMPU` loop. For bytes 0..3 `sc > gc` is the byte comparison,
  and for `dirG == false` `!(sc > gc || gc > 3)` with `gc <= 3` and `sc != gc` is `sc < gc`. Read positions beyond the
  compared length are never used (`i + j >= len` returns `N`).
- **Stepped calls (`M5_SEEDSTEP`).** `mmAdvance` is `maxMappableLength2strands`, `maxMappableLength` and `findMultRange`
  with each `compareSeqToGenome` / SAi lookup replaced by "return, and resume here with its result" (labels at the
  resumption points, the loop variables in `MmIt`). The same operations run in the same order with the same arguments
  (`compareSeqToGenome(S = pieceStart, N, L, iSA)`: `N = pieceLength` in `maxMappableLength`, `N = L3, L = L1b` in
  `findMultRange`), the per-pass results (`Nrep`, `ind`, `maxL`) are kept per iDist and `storeAligns` runs after the
  last pass in iDist order, and an overflow in `storeAligns` ends the call as before. Lengths are held in 32 bits (they
  never exceed the piece length), SA indices and SAi values in 64 bits, and the `Lind = 0` corner keeps STAR's
  `saiStart[Lind - 1]` index arithmetic in 64 bits.
- **Windows.** `p1Windows` visits the anchors' SA entries in STAR's order (pieces in order, entries in order, pieces with
  `Nrep > winAnchorMultimapNmax` skipped), calls `createExtendWindowsWithAlign` for the donor then the acceptor part of
  an sjdb insert, leaves the piece's entry loop on `EXIT_TOO_MANY_WINDOWS` (STAR's `break`), and returns overflow on a
  negative status, exactly as `mapPhase1`. The flanks loop is unchanged; `WinIt` runs the same statements in the same
  order, one entry or one window per step.
- **Persistent kernels.** A read's state is `work[k]`/`rbuf[k]` indexed by the read's position in the batch, as in the
  thread-per-read kernels, so the window-bin map's epoch (`wbEpoch`, which carries across batches per slot) sees the
  same sequence per slot. Every read's `status[k]`/`nEnt[k]` is written exactly once.

## Method

- Data: pbmc1k_v3, the standard 2 M distinct-sequence set (MAXD = 2,000,000: every 16th of the 31,870,813 distinct cDNA
  sequences, 1,991,926 sequences carrying 4,118,999 of the 66.6 M reads), STAR 2.7.11b truth regenerated in every job.
- GPU: H100 NVL (Clipper), CUDA 13.3, `-O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo`. `--reps 3` after a warm-up,
  median per lane; phase times are the mean of the timed runs. Standard budgets and batches (`--budget-a 1024
  --budget-m 2048 --budget-b 2048 --batch-a 524288 --batch-m 393216 --batch-b 65536`), E5 defaults on.
- Every variant runs in the same job as the baseline (the baseline twice, first and last). Identity: every distinct
  read's encoded result byte for byte against STAR's.
- "phase 1" is k_p1 (or k_p1s + k_p1w/k_p1wp) summed over the lane's batches; "seeds kernel" is k_p1s alone.
- Nsight Compute: `ncu --set full --kernel-name regex:k_p1 --launch-skip S --launch-count N` on lane A's second batch
  (524,288 reads), summarised by `bench/fable/profile/ncu_summary.py` (E1).

## Round 1: flat loops, persistent seeds, 2-bit compare (compile 795824, run 795825, commit 2a404cb)

Phase 1 per lane in ms (k_p1s alone in parentheses), lane totals (median of 3) and M5 throughput. Identity 1,991,926 /
1,991,926 (100%) in every row.

| variant | A phase 1 | M phase 1 | B phase 1 | lane A | lane M | lane B | M distinct/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| m5 (baseline, first) | 99.3 | 79.5 | 37.2 | 212.1 | 268.0 | 222.9 | 2.83 |
| m5 (baseline, last) | 99.4 | 79.4 | 37.3 | 212.4 | 266.8 | 223.1 | 2.84 |
| cmpu (strided compare only) | 91.6 | 78.6 | 36.6 | 204.6 | 267.8 | 222.3 | 2.87 |
| flat (iterator, thread per read) | 127.1 | 90.5 | 51.5 | 240.3 | 278.7 | 236.0 | 2.64 |
| flatp (persistent seeds + windows kernel) | **82.2 (61.2)** | **65.3 (26.0)** | 45.3 (10.7) | **195.5** | **253.9** | 230.8 | **2.93** |
| flatpw (both persistent) | 85.4 (61.2) | 87.3 (26.0) | **32.3** (10.7) | 198.6 | 276.7 | 219.0 | 2.87 |
| flatp8 (flatp, 64-register cap) | 95.0 (74.0) | 73.2 (34.0) | 46.8 (12.2) | 208.7 | 262.0 | 232.0 | 2.83 |
| flatpg (flatp + 2-bit compare) | 130.0 (109.0) | 88.4 (49.2) | 46.0 (11.3) | 243.1 | 277.3 | 233.3 | 2.64 |
| flatpwg | 133.0 (108.8) | 110.4 (49.2) | 32.9 (11.3) | 246.0 | 299.8 | 218.2 | 2.61 |
| flatpwg8 (64-register cap) | 103.2 (79.0) | 95.9 (34.7) | 34.1 (12.3) | 216.3 | 285.0 | 218.9 | 2.77 |

Registers (ptxas, lane A; lanes M and B the same unless noted): baseline k_p1 95 (5 blocks of 128 per SM); cmpu 94;
flat 80 (6 blocks); flatp k_p1s 72 (7 blocks), k_p1w 64; flatp8 k_p1s 64 with 48 B spill stores / 56 B spill loads;
flatpg / flatpwg k_p1s **192** (2 blocks per SM), flatpwg8 k_p1s 64 with 1,544 B spill stores / 1,556 B loads. No spills
elsewhere.

Thread-cycle shares, lane A (seeds / windows / stitching / select+encode): baseline 58.7 / 10.0 / 30.9 / 0.4 %; flat
71.8 / 2.6 / 25.3 / 0.4; flatp 61.0 / 9.7 / 29.0 / 0.4; flatpw 56.9 / 15.7 / 27.0 / 0.4. (In the persistent kernels a
read's seed interval runs from its set-up to its last call, including the iterations its warp spends on other lanes.)

Nsight Compute, lane A batch 2 (524,288 reads):

| kernel | ms | regs | occupancy achieved | active threads / warp instr. | warp cycles / issued instr. | top stalls (share) | L1 hit | L2 hit | DRAM throughput | issue slots busy |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| baseline k_p1 | 26.09 | 95 | 27.0% | **3.56** | 18.0 | long_scoreboard 68.5%, wait 14.0%, no_instruction 7.2% | 80.1% | 62.2% | 20.2% | 22.9% |
| flatpwg k_p1s (192 regs) | 28.19 | 192 | 12.4% | **6.06** | 13.1 | long_scoreboard 73.0%, wait 13.6% | 77.9% | 59.2% | 15.0% | 14.9% |
| flatpwg k_p1wp | 6.37 | 64 | 48.4% | 8.17 | 66.0 | long_scoreboard 91.6% | 68.0% | 57.2% | 31.5% | 11.4% |

Reading of round 1:

- The flat iterator with one thread per read (flat) is **slower** than STAR's nest (127 vs 99 ms): the switch-based
  iterator adds control flow and its lanes still finish their reads at different times. Only with lane refill (flatp)
  does the flat loop pay off: 0.83x of the baseline phase 1 in lane A, 0.82x in lane M.
- Converged calls are not converged warps: in the persistent seed kernel each lane makes one
  `maxMappableLength2strands` call per iteration, yet only ~6 of 32 lanes are active per instruction (baseline 3.6). The
  divergence is *inside* the call: the SAi lookups, the binary search of `maxMappableLength`, the two `findMultRange`
  searches and the byte-compare loops all have data-dependent trip counts.
- The phase is latency bound (long_scoreboard 68-73% of stalls, DRAM 15-20% of peak, issue slots 15-23% busy).
- The 2-bit compare as built needs 192 registers in k_p1s (2 blocks per SM), which costs more than it saves; capped at 64
  it spills 1.5 KB. Separating the windows into their own kernel costs ~21 ms in lane A (82.2 - 61.2), more than they
  cost inside the baseline k_p1; the persistent windows kernel is worse for lanes A and M and better only for lane B.

## Rounds 2 and 3: one memory operation per lane per iteration (compiles 795852 at 7b2e031 and 795933 at f753d75, run 795952)

Round 1 said the divergence sits inside `maxMappableLength2strands`, so round 2 steps the call itself (`M5_SEEDSTEP`):
every warp iteration each lane makes exactly one SAi lookup or one `compareSeqToGenome`, at one call site, then runs its
call's control flow to the next operation. Round 3 moves the stepped call's state to shared memory (`M5_STEPSMEM`)
because in registers it needs 164-204 registers (2-3 blocks per SM). `M5_P1FUSE` folds the windows into the seed kernel.
All in one job (run 795952), identity 1,991,926 / 1,991,926 (100%) in **every** row.

| variant | k_p1s regs (blocks/SM) | A phase 1 (k_p1s) | M phase 1 (k_p1s) | B phase 1 (k_p1s) | lane A | lane M | lane B | M distinct/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m5 (baseline, first) | k_p1 95 (5) | 99.2 | 79.7 | 37.4 | 212.4 | 268.2 | 222.1 | 2.83 |
| m5 (baseline, last) | | 99.3 | 79.7 | 37.2 | 212.6 | 270.0 | 222.6 | 2.82 |
| flatp (rebuilt; 80 regs after the fused-kernel restructure) | 80 (6) | 91.0 (70.0) | 68.8 (29.6) | 44.9 (10.4) | 204.1 | 257.1 | 230.0 | 2.88 |
| flatpni (compare out of line) | 96 (5) | 95.6 (74.6) | 71.4 (32.2) | 44.9 (10.3) | 209.0 | 259.6 | 233.1 | 2.84 |
| flatpf (windows fused) | 96 (5) | 118.4 | 90.8 | 51.0 | 231.4 | 278.8 | 236.0 | 2.67 |
| flatpg (2-bit, packing loop not unrolled) | 188 (2) | 131.2 (110.1) | 88.8 (49.5) | 46.0 (11.3) | 245.1 | 280.0 | 230.8 | 2.64 |
| step | 164 (3) | 197.5 (176.5) | 114.9 (75.7) | 52.8 (18.2) | 310.8 | 303.4 | 238.4 | 2.34 |
| stepg | 166 (3) | 183.8 (162.8) | 108.6 (69.4) | 50.8 (16.1) | 296.8 | 297.4 | 236.2 | 2.40 |
| step8 (64-register cap, 1.3 KB spills) | 64 (8) | 662.0 (641.0) | 309.7 (270.4) | 68.0 (33.5) | 775.5 | 497.9 | 254.8 | 1.30 |
| stepg8 (1.3 KB spills) | 64 (8) | 460.6 (439.5) | 229.9 (190.7) | 63.8 (29.3) | 573.8 | 420.2 | 249.5 | 1.60 |
| stepsm (state in shared memory, 25.6 KB/block) | 56 (8) | 123.5 (102.5) | 82.9 (43.6) | 52.8 (18.2) | 236.6 | 272.3 | 239.7 | 2.66 |
| stepsm6 | 78 (6) | 122.8 (101.0) | 87.0 (43.2) | 49.8 (15.1) | 236.4 | 275.6 | 235.1 | 2.67 |
| stepsm8 | 64 (8) | 122.4 (101.3) | 81.7 (42.5) | 51.6 (17.1) | 235.9 | 270.5 | 237.2 | 2.68 |
| stepgsm | 64 (8) | 133.0 (112.0) | 88.1 (48.9) | 50.7 (16.1) | 246.3 | 276.9 | 236.8 | 2.62 |
| stepgsm6 | 78 (6) | 115.1 (93.4) | 84.2 (40.2) | 48.1 (13.6) | 228.2 | 274.4 | 233.4 | 2.71 |
| stepgsm8 | 64 (8) | 132.6 (111.6) | 88.1 (48.9) | 50.8 (16.2) | 246.1 | 277.2 | 236.5 | 2.62 |
| stepsmf (fused) | 82 (5) | 220.6 | 145.6 | 66.7 | 334.1 | 334.8 | 252.6 | 2.16 |
| stepgsmf (fused) | 96 (5) | 230.9 | 147.1 | 65.3 | 344.5 | 336.2 | 251.2 | 2.14 |
| stepf (fused) | 204 (2) | 368.8 | 265.3 | 86.7 | 482.4 | 454.9 | 272.1 | 1.65 |
| stepgf (fused) | 204 (2) | 362.1 | 245.3 | 82.6 | 475.6 | 435.1 | 270.5 | 1.69 |

Nsight Compute, lane A batch 2 (run 795952):

| kernel | ms | regs | occupancy achieved | active threads / warp instr. | warp cycles / issued instr. | issue slots busy | IPC | top stalls | L1 hit | L2 hit | DRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| baseline k_p1 | 25.94 | 95 | 27.0% | 3.56 | 18.0 | 24.2% | 0.96 | long_scoreboard 68.5%, wait 14.0% | 80.1% | 62.1% | 20.4% |
| stepg k_p1s | 42.34 | 166 | 18.6% | 3.48 | 11.1 | 27.0% | 1.07 | long_scoreboard 52.8%, wait 16.3%, no_instruction 14.2% | 72.3% | 80.7% | 17.3% |
| stepsm k_p1s | 26.16 | 56 | 48.4% | **2.24** | 18.6 | 42.1% | 1.67 | long_scoreboard 64.0%, wait 14.9% | 53.1% | 63.0% | 22.7% |
| stepgsm k_p1s | 28.84 | 64 | 48.8% | 3.91 | 24.1 | 32.7% | 1.30 | long_scoreboard 65.7%, wait 9.2% | 40.9% | 61.2% | 32.0% |
| stepgsmf k_p1s (fused) | 60.00 | 96 | 30.9% | 2.80 | 20.1 | 24.7% | 0.98 | no_instruction 53.0%, long_scoreboard 24.9% | 63.1% | 67.8% | 11.6% |
| k_p1w (windows, thread per read) | 5.50 | 64 | 40.8% | 3.79 | 23.6 | 28.3% | 1.11 | long_scoreboard 77.2% | 77.8% | 69.0% | 23.8% |

Reading of rounds 2 and 3:

- **Stepping the call does not converge the warp.** The memory operation now has one call site and every lane makes
  one per iteration, yet the stepped kernels run at 2.2-3.9 active threads per instruction, no better than the
  baseline's 3.6. Between two memory operations each lane runs a different part of the call's control flow (nine
  resumption points, the pass bookkeeping, `storeAligns` at the end of a call, the iterator), and the warp executes each
  distinct path once per iteration: stepsm issues far more instructions (issue slots 42% busy, IPC 1.67 vs 0.96) and
  still waits on memory 64% of the time. The convergent part (the loads) is a small fraction of the instructions.
- **Register pressure decides the rest.** The call's state in registers needs 164-204 registers (2-3 blocks per SM) and
  is 1.8-3.7x slower than the baseline phase 1; capped at 64 registers it spills 1.3 KB and is 4.6-6.5x slower; in
  shared memory (stepsm, 56 registers, 8 blocks) it is still 1.24x slower than the baseline (123 vs 99 ms).
- **The 2-bit compare is not where the time goes**: neutral to slightly negative wherever it does not cost occupancy
  (stepgsm 133 vs stepsm 123 ms; stepgsm6 115 vs stepsm6 123 ms), and with the call inline it needs 188 registers.
- **Fusing the windows into the seed kernel is worse in every combination** (flatpf 118 vs flatp 91 ms in lane A):
  the windows are a long per-read loop that the other 31 lanes wait for, and the fused kernel needs 96 registers.
- Calling `compareSeqToGenome` out of line (flatpni) costs 5% (96 registers, 656 B stack).

## Round 4: occupancy and prefetch on the call-level kernel (compiles 796016 at e1a3ecb and 796022 at 4d49bcc, run 796023 on g052)

Back to the call-level persistent kernel (flatp, now "flatq": `k_p1s` finishes reads inline again, 72 registers, 7
blocks per SM as in round 1), with: the seed iterator in shared memory (`M5_SEEDSMEM`, 7 KB per block) so that a
64-register cap (`M5_P1MINB=8`) fits 8 blocks per SM without spills ("flatqs8"); binary-search prefetch (`M5_PREF`: the
SA words of both possible next midpoints prefetched into L2 during each compare; `M5_PREF2`: also loads those two SA
values and prefetches the genome lines of both next compares); the 2-bit compare at 7-8 blocks; persistent windows.
Run 796023 ran on node g052 (rounds 1-3 on g051): the baseline's lane B phase 1 is 40.5 ms there (37.2 on g051).
Identity 1,991,926 / 1,991,926 (100%) in every row.

| variant | k_p1s regs (blocks/SM) | A phase 1 (k_p1s) | M phase 1 (k_p1s) | B phase 1 (k_p1s) | lane A | lane M | lane B | M distinct/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| m5 (baseline, first) | k_p1 95 (5) | 100.0 | 79.5 | 40.4 | 212.9 | 269.5 | 223.3 | 2.82 |
| m5 (baseline, middle, --p1probe) | | 100.1 | 79.2 | 40.5 | 213.2 | 270.0 | 223.0 | 2.82 |
| m5 (baseline, last) | | 100.0 | 79.4 | 40.6 | 213.1 | 271.5 | 224.8 | 2.81 |
| m5p (baseline + PREF) | k_p1 96 (5) | 100.3 | 79.9 | 40.5 | 214.6 | 270.5 | 225.7 | 2.80 |
| m5p2 (baseline + PREF2) | k_p1 96 (5) | 111.2 | 82.8 | 40.5 | 224.2 | 273.9 | 223.0 | 2.76 |
| flatq | 72 (7) | 82.5 (61.3) | 64.7 (26.2) | 46.1 (10.8) | 195.4 | 257.3 | 230.4 | 2.92 |
| flatqs (iterator in shared memory) | 72 (7) | 82.5 (61.4) | 64.1 (25.6) | 45.5 (10.2) | 195.3 | 254.6 | 227.7 | 2.94 |
| **flatqs8** (+ 64-register cap) | 64 (8) | **78.9 (57.8)** | **62.5 (24.0)** | 45.5 (10.1) | **192.0** | **252.9** | 228.4 | **2.96** |
| flatqp (+ PREF) | 72 (7) | 83.8 (62.7) | 65.2 (26.8) | 46.1 (10.7) | 196.8 | 256.0 | 227.5 | 2.93 |
| flatqp2 (+ PREF2) | 80 (6) | 95.5 (74.3) | 71.6 (33.1) | 46.6 (11.2) | 208.7 | 263.1 | 228.9 | 2.84 |
| flatqs8p (+ PREF) | 64 (8) | 80.5 (59.3) | 63.4 (24.8) | 45.5 (10.0) | 193.3 | 255.5 | 230.7 | 2.93 |
| flatqs8p (+ PREF, --p1probe) | 64 (8) | 80.7 (59.6) | 63.3 (24.8) | 45.5 (10.1) | 193.9 | 253.7 | 227.9 | 2.95 |
| flatqs8p2 (+ PREF2) | 64 (8) | 87.1 (65.9) | 66.0 (27.5) | 45.8 (10.6) | 200.2 | 257.5 | 228.5 | 2.90 |
| flatqg7 (2-bit, 1.1 KB spills) | 72 (7) | 94.8 (73.4) | 75.0 (31.3) | 46.4 (11.0) | 207.8 | 265.7 | 228.9 | 2.84 |
| flatqsg8 (2-bit, 1.1 KB spills) | 64 (8) | 100.2 (79.0) | 72.4 (33.9) | 46.9 (11.4) | 213.2 | 263.8 | 231.2 | 2.81 |
| flatqw (persistent windows) | 72 (7) | 85.3 (61.4) | 87.4 (26.2) | **32.7** (10.8) | 198.5 | 278.0 | 215.7 | 2.88 |
| flatqs8w (persistent windows) | 64 (8) | 81.5 (57.6) | 85.2 (24.0) | **32.0** (10.1) | 194.7 | 275.0 | **215.1** | 2.91 |

Reading of round 4:

- **Occupancy is the one knob that works**: 7 blocks per SM (72 registers) 61.3 ms, 6 blocks (80 registers, round 2's
  rebuild) 70.0 ms, 8 blocks (64 registers, iterator in shared memory, no spills) 57.8 ms for lane A's seed kernel.
  Capping registers *with* spills (flatp8 in round 1: 48 B) loses more than the extra warps gain.
- **Prefetch does not help** (PREF +1-2%, PREF2 +11-14%: the extra loads and registers cost more): the binary-search SA
  reads are not the latency that matters (the SA range of a 14-mer is small, so most binary-search SA words already sit
  in L1/L2: L1 hit 78-80%, L2 hit ~60% in every phase-1 profile). The remaining long-scoreboard stall is therefore
  likely on short dependent chains (per-byte compares, `Work` fields, the read buffers) rather than on DRAM misses;
  this was not verified at source level.
- The persistent windows kernel is the best windows kernel for lane B only (32 vs 45.5 ms) and the worst for lane M (85
  vs 62.5 ms). `M5_P1WPB` (persistent windows in lane B only) combines the two; measured in the final run.

Nsight Compute, lane A batch 2 (run 796023):

| kernel | ms | regs | occupancy achieved | active threads / warp instr. | warp cycles / issued instr. | issue slots busy | IPC | top stalls | L1 hit | L2 hit | DRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| flatq k_p1s | 16.50 | 72 | 42.1% | 3.33 | 19.6 | 34.8% | 1.37 | long_scoreboard 70.9%, wait 13.1% | 76.4% | 57.9% | 28.5% |
| flatq k_p1w | 5.78 | 64 | 40.8% | 4.24 | 24.6 | 27.2% | 1.06 | long_scoreboard 78.3%, wait 10.5% | 77.9% | 69.2% | 23.8% |
| flatqs8p k_p1s | 16.09 | 64 | 47.9% | 3.53 | 20.5 | 38.0% | 1.50 | long_scoreboard 72.3%, wait 11.8% | 65.0% | 57.1% | 36.0% |
| flatqs8p k_p1w | 5.81 | 64 | 40.9% | 4.26 | 24.5 | 27.2% | 1.07 | long_scoreboard 78.4% | 77.8% | 69.0% | 23.9% |
| m5p k_p1 (baseline + PREF) | 26.84 | 96 | 27.2% | 3.80 | 17.2 | 25.6% | 1.01 | long_scoreboard 63.7%, wait 14.2%, no_instruction 11.2% | 75.5% | 61.1% | 23.1% |

The persistent seed kernel is faster than the baseline's k_p1 (16.5 + 5.8 = 22.3 ms vs 25.9 ms for the same batch)
**without** running more lanes per instruction (3.3-3.5 vs 3.6): the gain comes from keeping every resident warp busy
until the batch is exhausted (no tail of half-empty warps, k_p1's achieved occupancy is 27% of a 31% theoretical) and
from more resident warps (42-48% achieved occupancy), i.e. more independent memory chains in flight, not from
convergence. Issue slots are 35-38% busy and IPC 1.4-1.5 against 0.96: the SM does more useful work per cycle because
more warps are resident, while each warp is as divergent as before.

## Phase 1 once with lane M's window capacity (`--p1probe`, run 796023)

`--p1probe` times phase 1 alone over all 1,991,926 reads with lane A's capacities and with lane M's (median of 3 after
a warm-up, the same kernels as the lane runs) and counts, after the run with lane M's capacities, the reads that
overflow, the reads with more than lane A's 16 windows, and the reads within lane A's capacities.

| kernels | phase 1, all reads, lane A caps | phase 1, all reads, lane M caps | cascade today: A phase 1 + M phase 1 | saved |
|---|---:|---:|---:|---:|
| baseline (m5) | 100.1 ms | 142.3 ms | 100.1 + 79.2 = 179.3 ms | **37.0 ms** (21% of phase 1, 5.2% of the 705 ms GPU total) |
| flatqs8p | 80.7 ms | 129.6 ms | 80.7 + 63.3 = 144.0 ms | **14.4 ms** (10% of phase 1, 2.1% of the GPU total) |

With lane M's capacities, 75,712 reads overflow (the reads lane B takes today: lane M's overflow), 434,926 have more
than 16 windows (they go to lane M's phase 3), and 1,481,288 stay within lane A's capacities (lane A today: 1,481,287
finish in A; the one-read difference is the read that overflows A's window-bin map, `winbin=1`). The saving does not
count the cost of running phase 3 of the 1.48 M lane-A reads on lane M's 67.9 KB `Work` layout instead of lane A's 36 KB
one (E5 showed lane A's phase 3 is sensitive to the size of its state), nor the conversion or the bookkeeping; it is an
upper bound. With the faster flat kernels the saving shrinks to 14 ms because phase 1 with lane M's capacities is
memory-bound on the larger `Work` (129.6 ms vs 80.7 ms with A's).

## Final run: flatqs8b against the baseline (compile 796040 at fc646e6, run 796044 at 3040da0 on g052)

flatqs8b = `-DM5_SEEDFLAT -DM5_P1SPLIT -DM5_SEEDSMEM -DM5_P1MINB=8 -DM5_P1WPB`: the persistent refilling seed kernel with
the seed iterator in shared memory and a 64-register cap (8 blocks per SM, no spills), the thread-per-read windows
kernel in lanes A and M and the persistent windows kernel in lane B. One job, baseline and variants alternating (m5,
flatqs8b, flatqb, flatqs8, flatqs8b, m5, flatqs8b, m5), each the median of 3 after a warm-up. Identity 1,991,926 /
1,991,926 (100%), reads weighted 4,118,999 / 4,118,999, in **all 8 rows**. The table gives the median of the three
baseline rows and of the three flatqs8b rows (the three runs of each agree within 0.35 ms in phase 1 and 2.4 ms in the
lane totals).

| variant | k_p1s regs (blocks/SM) | A phase 1 (k_p1s) | M phase 1 (k_p1s) | B phase 1 (k_p1s) | lane A | lane M | lane B | GPU total | M distinct/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m5 (baseline, median of 3) | k_p1 95 (5) | 100.05 | 79.54 | 40.46 | 213.2 | 270.1 | 223.9 | 707.2 | 2.82 |
| **flatqs8b** (median of 3) | 64 (8) | **78.93** (57.76) | **62.57** (24.00) | **32.01** (10.12) | **192.0** | **253.5** | **214.7** | **660.2** | **3.02** |
| flatqs8b / baseline | | **0.789x** | 0.787x | 0.791x | 0.901x | 0.939x | 0.959x | 0.934x | **+7.1%** |
| flatqb (no shared-memory iterator, no cap) | 72 (7) | 82.47 (61.29) | 64.63 (26.14) | 32.72 (10.78) | 195.9 | 256.2 | 217.2 | 669.3 | 2.98 |
| flatqs8 (thread-per-read windows in lane B too) | 64 (8) | 79.10 (57.95) | 62.42 (24.01) | 45.51 (10.11) | 192.3 | 252.5 | 228.3 | 673.1 | 2.96 |

Phase 1 over the three lanes: 220.1 ms -> 173.5 ms (-46.5 ms); the GPU total drops by 47.0 ms, so nothing else moved
(scan and phase 3 within 1 ms). Individual rows: baseline lane A/M/B 213.0/270.1/223.1, 213.3/270.1/223.9,
213.2/269.5/225.5 ms (2.82, 2.82, 2.81 M/s); flatqs8b 192.0/252.3/214.8, 191.5/253.7/214.2, 192.0/253.5/214.7 ms (3.02
M/s each).

Thread-cycle shares (seeds / windows / stitching / select+encode; in the persistent kernel a read's seed interval
includes the iterations its warp spends on other lanes):

| lane | baseline | flatqs8b |
|---|---|---|
| A | 58.6 / 9.9 / 31.0 / 0.5 % | 62.4 / 9.3 / 27.9 / 0.4 % |
| M | 22.4 / 19.9 / 57.5 / 0.1 % | 24.0 / 18.6 / 57.3 / 0.1 % |
| B | 8.2 / 19.7 / 72.0 / 0.0 % | 7.1 / 18.2 / 74.6 / 0.0 % |

Registers (ptxas, 796040): flatqs8b `k_p1s` 64 registers, 128 B stack frame, 7,168 B shared memory, no spills (all
three lanes); windows kernels 64 registers, of which one of the two per lane (the compile summary labels `k_p1w` and
`k_p1wp` alike) has 8 B (lanes M, B) or 16 B (lane A) of spill stores and loads; `k_p3` 79 registers, unchanged. flatqb
`k_p1s` 72 registers, windows kernels 62-64, no spills.

Nsight Compute, lane A batch 2 (run 796044):

| kernel | ms | regs | occupancy achieved | active threads / warp instr. | warp cycles / issued instr. | issue slots busy | IPC | top stalls | L1 hit | L2 hit | DRAM | DRAM read |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| flatqs8b k_p1s | 15.46 | 64 | 47.8% | 3.38 | 21.2 | 36.6% | 1.44 | long_scoreboard 73.3%, wait 11.9% | 70.2% | 59.5% | 31.9% | 17.0 GB |
| flatqs8b k_p1w | 5.79 | 64 | 40.9% | 4.26 | 24.5 | 27.2% | 1.07 | long_scoreboard 78.3%, wait 10.5% | 77.8% | 68.9% | 24.0% | 3.8 GB |
| m5 k_p1 (baseline) | 26.65 | 95 | 27.1% | 3.77 | 18.2 | 24.0% | 0.95 | long_scoreboard 68.1%, wait 13.9%, no_instruction 7.9% | 80.0% | 62.6% | 20.7% | 18.1 GB |

Same batch: 15.46 + 5.79 = 21.25 ms against 26.65 ms (0.80x), with **fewer** active threads per instruction in the seed
kernel than in the baseline (3.38 vs 3.77; the baseline measured 3.56 on g051 in rounds 1-3). The gain is resident warps
(47.8% vs 27.1% achieved occupancy, 7.75 vs 4.37 active warps per scheduler) and issue rate (IPC 1.44 vs 0.95), not
convergence.

## What failed, and why

| attempt | result (lane A phase 1 vs baseline ~99-100 ms) | why (measured unless marked) |
|---|---|---|
| flat iterator, thread per read (`M5_SEEDFLAT` alone) | 127 ms, 1.28x | the switch-based iterator adds control flow; lanes still finish their reads at different times, so warps idle as before |
| 2-bit compare (`M5_G2`) | slower than the same variant without it in every pair but one (stepgsm6 115 vs stepsm6 123 ms), e.g. flatqsg8 100.2 vs flatqs8 78.9 ms; 188-192 registers inline, 1.1-1.5 KB spills when capped | the packed path needs many registers and is no faster at equal occupancy (stepgsm vs stepsm, flatqsg8 vs flatqs8); likely because most compares end within a few bases of where they start (the binary search starts each compare after the already-known common prefix), so 32-base words rarely save iterations (not measured) |
| stepped calls, one memory operation per iteration (`M5_SEEDSTEP`) | 1.8-6.5x (registers 164-204, or spills) | does not converge warps (2.2-3.9 active threads / instruction): lanes run different control flow between operations, and the warp pays for each path every iteration |
| stepped state in shared memory (`M5_STEPSMEM`) | 1.16-1.34x | removes the register problem, not the divergence; instruction count up (IPC 1.67 vs 0.96) |
| windows fused into the seed kernel (`M5_P1FUSE`) | 1.19x (call level), 2.2-3.7x (stepped) | the windows loop runs while 31 other lanes wait, and the fused kernel needs 96 registers |
| `compareSeqToGenome` out of line (`M5_CMPNI`) | +5% over flatp | 96 registers and 656 B of stack, not fewer |
| persistent windows for lanes A and M (`M5_P1WPERS`) | lane A +3 ms, lane M +23 ms | likely: the per-entry iterator's overhead exceeds the tail it removes for lanes A and M (not profiled per lane); only lane B (wide reads, long window loops) gains: 32 vs 45.5 ms |
| register cap with spills (`M5_P1MINB=8` without `M5_SEEDSMEM`) | 95 ms (vs 82) | 48-56 B of spills in the hot loop cost more than the 8th block gains |
| binary-search prefetch (`M5_PREF`, `M5_PREF2`) | +1-2% / +11-14% | likely: the SA words of the next midpoints mostly hit L1/L2 already (L1 hit 75-80%, L2 ~60% in every profile); the extra loads and registers (PREF2: 80 regs) cost more than they hide |
| `M5_PREF3` | built (796029), not run | dropped once `M5_PREF` showed the SA reads are not the critical latency |

## Verdict

- **Gate not met.** The best variant (seeds and windows split, persistent refilling seed kernel, seed iterator in shared
  memory, 64-register cap: flatqs8, plus persistent windows in lane B only: flatqs8b) runs lane A's phase 1 in
  **78.9 ms vs 100.1 ms = 0.79x** (gate 0.5x, target 0.3x), lane M's in 62.6 vs 79.5 ms (0.79x), lane B's in
  32.0 vs 40.5 ms (0.79x). M5 goes from **2.82 to 3.02 M distinct reads/s (+7.1%; GPU total 707.2 -> 660.2 ms)**.
- **Identity 1,991,926 / 1,991,926 (100%) in every row of every run** (55 rows in 4 runs, including the
  stepped and 2-bit ones): every transformation here is exact.
- **The hypothesis is false for this code.** Flattening the nest to one call site, and even to one memory operation per
  lane per iteration, does not raise active threads per instruction meaningfully: 3.3-3.5 in the call-level kernels
  against the baseline's 3.6-3.8, 2.2-3.9 stepped; the one exception (6.1, round 1's 192-register 2-bit kernel) ran at
  2 blocks per SM and was slower. The divergence is not an artefact of STAR's loop nest but of the data: every level has its own
  data-dependent trip count (calls per read, iDist passes, SAi lookups, binary-search steps, findMultRange steps, compare
  lengths), so lanes of a warp are never at the same point for long, however the loops are written. What the persistent
  kernel buys is occupancy and the end of the batch tail: more resident warps (42-48% achieved vs 27%) keep more
  independent memory chains in flight, and refilled lanes keep warps full until the batch runs out. The phase is latency
  bound (long_scoreboard 64-78% of stalls) at 20-36% of DRAM bandwidth.

## Surprises

- Converging the *call* sequence does nothing for SIMT efficiency (6.1 active threads in round 1's 192-register kernel at 2 blocks per SM, 3.3-3.5
  in the 7-8-block kernels, baseline 3.6-3.8); a warp-uniform outer loop is not a convergent warp when the inner loops are data dependent.
- Splitting phase 1 into a seeds kernel and a windows kernel costs ~21 ms in lane A and makes lane B's phase 1 slower
  (45.5 vs 40.5 ms): likely because the thread-per-read windows kernel re-reads each read's `Work` and is worst for lane B's long
  window loops; the persistent windows kernel fixes lane B (32 ms) and hurts lanes A and M.
- DRAM traffic is about the same for the baseline and the persistent kernels (18-21 GB read per lane-A batch in either: 20.9 GB for flatqs8b's k_p1s + k_p1w vs 18.1 GB for k_p1 in run 796044); only the latency
  hiding differs.
- Running phase 1 once with lane M's capacities would save 37 ms with the baseline kernels but only 14 ms with the
  faster persistent kernels, before the cost of lane A's phase 3 on lane M's larger state.
- Register count is the dominant variable: 72 -> 80 registers (an innocuous restructure for `M5_P1FUSE`) cost 9 ms in
  lane A; 72 -> 64 without spills (iterator in shared memory) saved 3.5 ms.

## Next steps

1. **Cooperative SA search** (the only route to a 0.5x phase 1 that these measurements leave open): a warp, or an
   8-lane group, per `maxMappableLength2strands` call, replacing the binary search by a k-ary search (k candidate
   midpoints compared in parallel per step: log_k instead of log_2 dependent steps, coalesced SA reads) and the per-byte
   compare by a group-parallel compare. This needs its own exactness argument: STAR's `maxMappableLength` returns the SA
   index its particular sequence of midpoints ends on, and `findMultRange` expands from there; the final range is path
   independent, but `L1`/`L2` and the ordering of equal-length hits must be shown to be, or reproduced.
2. Adopt flatqs8b (`-DM5_SEEDFLAT -DM5_P1SPLIT -DM5_SEEDSMEM -DM5_P1MINB=8 -DM5_P1WPB`) as m5's phase-1 default:
   exact, +7.1% on M5 (2.82 -> 3.02 M distinct reads/s).
3. Shorten the windows kernel (k_p1w: ~21 ms of lane A's ~79 ms phase 1): an 8-lane group per read over the anchors'
   SA entries, keeping STAR's entry order for `createExtendWindowsWithAlign` (the window-bin map is order sensitive).
4. Phase 1 once with lane M's capacities (the probe): at most 14 ms with the faster kernels; worth it only together
   with a layout that lets lane A's phase 3 keep its small state.
5. The 2-bit genome is not worth pursuing for phase 1 at thread-per-read granularity; it may pay off inside a
   cooperative search (item 1), where one 32-base word per lane per step is natural.

## Process note: Clipper bridge incident

Early in the session (about 23:18-23:35 EDT) one of my bridge commands (a loop whose `$f` expanded to nothing) started
a program that read the shared bridge's standard input, and it consumed the queued commands of other agents (about
eight) and my own first `sbatch` before it ended. I drained the queue, removed the stale lock, resubmitted my job, and
from then on appended `< /dev/null` to every bridge command that could read standard input. Other agents' commands in
that window may need to be re-run if they did not see a result.

## Jobs and commits

Branch `gpu/fable/seeds`. E7 commits: 8d0477f (flat phase 1, CMPU, P1SPLIT, P1MINB, jobs), 12d3340 (run job keeps DIFF
lines), 120f218 (G2), 2a404cb (P1WPERS), 3b33f7e (`--p1probe`), 5f9ace8 (SEEDSTEP, CMPNI), 7b2e031 (P1FUSE), f74869c
(round 2+3 run list), f753d75 (STEPSMEM), 09577be (k_p1s inline finish, SEEDSMEM), e1a3ecb (probe count fix), 4d49bcc
(PREF, PREF2), 0f2ce51 (PREF3), fc646e6 (P1WPB), 3040da0 (final run list), 641016c and later (this write-up).

| job | kind | commit | what |
|---|---|---|---|
| 795824 | compile | 2a404cb | round 1 variants |
| 795825 | run (g051) | 2a404cb | round 1 + ncu |
| 795839 | compile | | cancelled (folded into 795852) |
| 795840, 795853 | run | | cancelled before start (resubmitted with more variants) |
| 795852 | compile | 7b2e031 | round 2 variants |
| 795933 | compile | f753d75 | round 3 variants |
| 795952 | run (g051) | f74869c | rounds 2+3 + probe + ncu |
| 796016 | compile | e1a3ecb | round 4 variants |
| 796017 | run | | cancelled before start (resubmitted with the PREF variants) |
| 796022 | compile | 4d49bcc | PREF, PREF2 variants |
| 796023 | run (g052) | 4d49bcc | round 4 + probe + ncu |
| 796029 | compile | 0f2ce51 | PREF3 (not run) |
| 796040 | compile | fc646e6 | flatqs8b, flatqb (P1WPB) |
| 796044 | run | 3040da0 | final: baseline vs flatqs8b, flatqb, flatqs8 alternating + ncu |

Binaries: `/mnt/projects/debruinz_project/debruinz/singlet/jobs/fable/seeds/bin/`; logs
`/mnt/projects/debruinz_project/debruinz/singlet/logs/e7-*.out`; per-variant logs and ncu summaries under
`/mnt/projects/debruinz_project/debruinz/singlet/jobs/fable/seeds/results/`.
