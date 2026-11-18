# Device memory: a fixed per-slot budget with blockwise streaming

Analysis only (branch `rt/mem`, from `gpu/runtime-v138`; no code changed, nothing measured on a GPU for this note). Sources: the code at
this commit, `docs/gpu-engine/fable/E17-budget.md`, `docs/research-beta/{align-and-count,gpu-pileup}-design.md`,
`bench/gpu-runtime/results-2026-11-13.md`, and the campaign corpus table. Numbers marked (m) are measured elsewhere and quoted; numbers
marked (e) are my estimates from the formulas here. GB = 2^30.

## 0. Answer in brief

- The prefill (mapping) is already budgeted and blockwise (E17: read chunks, hash-partitioned dedup, lane parts, record chunks). It is
  not unbounded, but it keeps about 20 B per read on the device in the records phase, so it stops fitting a 31 GB slot near 1.2 G pairs
  (23 GB slot: near 0.9 G).
- Four phases are unbounded or sized from the wrong quantity today:
  1. **GPU pileup**: all records and all barcoded reads are device-resident (about 160 B per barcoded read (m)), only the per-read working
     set is cut into cell partitions. A 31 GB slot holds about 200 M pairs, a 23 GB slot about 140 M (e). Beyond that it widens (takes the
     whole card, evicts the index holder), then unified memory, then fails.
  2. **Non-host classify**: the 24.9 GB index plus about 1.2 KB per unmapped read (m, one point). It is outside the budget and exceeds a 31 GB
     slot above about 5 M unmapped reads, and a 23 GB slot always.
  3. **Hand-off pass** (`e2ePass`): all records resident, read chunk sized from the free memory of the card, not the slot. This is where
     two items OOMed in the 3-slot test (m).
  4. **Solo, all-site arena, pileup arena** use free memory in places (Solo: 97 % of free; all-site: 0.6 x free) instead of one slot
     arena. Two slots can both size from the same free memory.
- Samples above 2^31 reads cannot run on the device paths at all (32-bit indices; Solo goes to the host above 2^31 - 1 reads), and the
  corpus has at least 25 samples above 4.3 G pairs.
- Proposal: one hard-capped arena per slot, every phase asks it (never `cudaMemGetInfo`), a failed request halves the block instead of
  failing, and four phases get an out-of-core design (pileup, non-host, hand-off/records, prefill residue). Ranked list in section 5.

## 1. Fixed facts

| item | value | source |
|---|---|---|
| card | H100 NVL, 93.1 GB | E17 |
| slots per card | 2, `SINGLET_ENGINE_GPU_MEM_GB=31` set by `rt_node.sbatch` (default formula would give (93.1-2-16.4)/2 = 37 GB with the holder) | `bench/gpu-runtime/jobs/rt_node.sbatch:11,17`, `include/singlet/engine_gpu.h:395` |
| shared genome index (holder) | 16.4 GB, outside the budget (`sharedIndexReserve`) | `engine_gpu.cu:1256-1259` |
| non-host index + Bloom | 24.9 GB, outside the budget; 25.4 GB resident, 29.3 GB at classify on pbmc1k_v3 (m) | E17 section 1e |
| context + driver | 0.5-1 GB per process; lane M's warp stack reserves 3.96 GB device-wide (m) | `engine_gpu.cu:788-796` |
| card peaks | 77-82 GB with 2 slots | brief |
| 3 slots | 3 x 23 GB + 16.4 GB: OOM in hand-off pass (2 items) and in the non-host server; node rate 8.28 vs 9.19 at 2 slots (m) | `results-2026-11-13.md:76` |
| slot-seconds per item | map 54, all-site 22, pileup 15, hand-off 10, non-host 6.5, Solo 3.3 (~110 of ~126 available); each slot-second removed per item is about +0.9 % node rate | `results-2026-11-13.md:68` |

Budget is a planning tally (`g_bud`, `engine_gpu.cu:691`), enforced only by checks before allocations in the prefill; every other phase
allocates with raw `cudaMalloc`.

## 2. Phase inventory

Symbols: n = reads (= read pairs, R2 only), nB = barcoded reads (about 0.88 n), D = distinct sequences = rho n (rho = 0.475 on pbmc1k_v3:
31.7 M of 66.6 M (m)), nRec = records (about D), nAln about nRec, W = whitelist size (6.8 M for 3' v3), U = unmapped reads for the screen.

| # | phase (file:line) | device memory (formula) | blocked today by | when input exceeds the budget | access pattern |
|---|---|---|---|---|---|
| 1 | reads phase, normal / lean (`engine_gpu.cu:1273-1288`) | RD = S0 + 8n; peak about 275 n (clip term) normal, 219 n lean (e) (+ 16.4 GB index if private, + ctx + 24 MB + 1 GB slack) | nothing (whole sample) | normal -> lean (index after dedup) -> RC; hard stop `failBudget` if lean does not fit and RC is off | sequential uploads; per-read kernels; staged H2D (64 MB pinned) |
| 1b | reads by chunk, RCHUNK, P3-MEM-03 (`:1724-2060`) | chunk RCN = room/4/perR reads (min 1 M), perR about 330 B; dedup partition 96 B per read of the largest hash partition; NP = n x 96 / room | read chunks, then hash partition of h1 (host counting partition, `Ip`) | `failBudget` only if one chunk or partition does not fit | host: partition pass sequential per thread; device: radix sort per partition, sequential |
| 2 | lanes A / M / B (`:752-960`, `:2315-2360`) | result slots m x (647 / 2055 / 4103) B for the m sequences of a chunk + NB x per-read in flight (A 36 KB, M 68 KB thread / 2 KB+stack warp, B 550 KB before WSHRINK (m)) + 16 MB | chunk of up to 16 M distinct sequences (shrunk to the room), batch NB, `runLaneParts` | smaller NB, then parts; `failBudget` only if NBmin and its slots do not fit | random gathers into the 16.4 GB SA / SAindex / genome (inherent); atomics for counters; lane M sorts by window count |
| 3 | host lane (`:960-1000`, `:2370-2480`) | none (CPU, by design) | per batch overflow | reads over a lane capacity map on the CPU pool, results unchanged | n/a |
| 4 | E8 records (`:2808-3300`, `records_gpu.cu:121-160`) | `recNeed`: used + 28 nRec + blob + seq + 101 nRec + seq/2 + 24 nRec + 4n + 256 MB + (4n + 8 nRec [+ 30 n + 16 n lean]) (e), about 300-410 B per record whole-sample | whole sample (`e2eBuild`) or record chunks (`e2eBuildRC`, only when RC) | `recNeed > cap` -> "E8 records not built", hand-off file path, E10 off (level < 5); SL admission (`:3335`, off by default) may take several slots | blob / record parallel kernels, thrust scan and sort; sequential |
| 5 | E10 products, E11 keep (`hostloop_gpu.cu:302-400`, `solo_gpu.cu:1180-1270`) | products: 18 nRec + about 50 nE (junction entries x key, rec, val, sort double buffer); keep: 16 n (CR/UR, stays through records) + 4 n + 8 nRec | whole sample, or per record chunk in RC | raw `cudaMalloc` -> "E10 ran out of device memory", `oomFatal` fails the call, caller reruns; keep refused -> level 5 off | global stable radix sort on 64-bit junction keys; atomics `k_mult` (random into nRec counters) |
| 6 | hand-off pass `e2ePass` (`records_gpu.cu:404-470`) | records resident: 16 nRec + 18 nAln + 4 nOps; per read chunk 104 B (reads 24 + masks + keep + kept read + keys); 4 GB margin | read chunks m0 sized from **free** device memory | records `cudaMalloc` fails -> pass returns false (item failed; seen at 23 GB x 3 slots) | read -> record gather (random into nRec arrays), prefix scan, atomics for cell counters |
| 7 | GPU pileup (`pileup_gpu.cu:100-300`, `gpu_pileup/pileup/pileup.cu:1066-1260`, `m6/m6.cu`) | one arena = min(free - 3 GB, budget - 3 GB). Resident for the whole run: S 16 nS, Al 16 nA, ops, seq (rlen/2), R 16 nB, lowq 16 nB, annotation AA 60 nA, GRes 16 nS, lists, pid 1 nB, SNP panel 10 B x rows; about 160 B per barcoded read (m, derived: 100 GB at K=1 and 72 GB at K=2 for 223 M, `align-and-count-design.md:176`). Working set 290 B per barcoded read (64 + 196 gene + 110 sj + 150 snp + 34 mt + 24 ind per tuple, `pileup.cu:781`), divided by K | cell-hash partitions K <= 64 (`kMaxParts`, `pileup.cu:774`); K from `k_binsum` trials | attempt 0 arena; attempt 1 `pileupWiden`: SIGUSR1 to the index holders, takes all other slots of the card (waits up to 120 s); attempt 2 `cudaMallocManaged(2 x device)`; then `ArenaExhausted` -> pileup fails (exit 9 when the CPU steps were skipped) | CUB radix sorts and reduce-by-key (sequential); `k_gather64` by sort permutation and read -> signature gathers (random within resident arrays); atomics on per-cell counters (12 B x W) and union-find atomicMin; K select passes over all resident reads |
| 8 | all-site (`allsite_gpu.cu:468-860`, `allsite_device_glue.h:233`) | one arena min(0.6 x free - 0.5 GB, 24 GB, budget - 1 GiB), >= 1.5 GB; per chunk: gathered reads 60 B + 4 B x ops + seq + 8 B x lq words + ref; discovery nObs x 32 B + CUB temp (not blocked); counting 112 B per hit | chunks of contigs (<= 45 % of arena), oversize contigs resplit into position windows (`in.resplit`), counting blocks of sites cut by hit histogram | `out.why` -> host (CPU) all-site pass: "a contig's reads exceed the device budget", "arena: X of Y MiB" (discovery), "too many mismatch observations for one chunk" (>= 2^31), "one site's hits exceed the device budget", "discovery key past 64 bits" | radix sort of (site, barcode, UMI) keys, lock-free union-find, reference reads random inside one window |
| 9 | non-host classify (`nonhost_gpu.cu:307-400`, `:583-660`) | index + Bloom 25.4 GB resident; per sample n_unm x (seq 91 + 8 + 4 + 24 + 16 offsets) + Q x 32 (Q = non-host minimizers, about 12 per read) + H x (24 + 12 + 8) (H = index hits) + CUB temp; measured 3.9 GB for 3.36 M reads = 1.16 KB per read (m) | none: one call for all U reads (CPU subsample above 20 M) | any CUDA error -> up to 4 retries, then CPU screen with the full index load; H >= 2^31 or n >= 2^31 -> CPU | **random probes** into the 24.9 GB sorted index (directory read + short scan); radix sort of H keys |
| 10 | Solo (`solo_gpu.cu:1309-1432`) | one shot: 39 B x N + 12 nHost + 112 B x W + model + 256 MB; two pass: chunk C = 0.9 x free / 52 B per read | free-memory test (97 % of free) -> two-pass read chunks + key buckets | OOM -> chunk and bucket halving (3 attempts) -> host Solo (`toHostRun`, same result, slow); N > 2^31 - 1 -> host at once | per read gathers into whitelist-sized tables (W x 112 B), radix sort of keys |
| 11 | genome index / holder (`shared_index.cuh`) | 16.4 GB mapped by IPC into the prefill | one copy per card | holder evicted by pileup widen | read-only random |
| - | not in the campaign path | `include/singlet/gpu/*` (Python bindings, `SINGLET_BUILD_GPU`); ingest (E13) uses pinned host buffers only | | | |

Cells and donors: device memory depends on cells only through the whitelist size W (pileup per-cell counters 12 B x W, 64 B x W more with
reads-per-UMI; Solo 112 B x W (0.76 GB at 6.8 M); hand-off 4 W + 32 nCellsQc; key packing needs cell + gene + UMI bits <= 64). It does not
depend on the number of detected cells. Donors enter only through the SNP panel table (10 B per row, about 23 M rows with the editing panel =
0.23 GB, resident in the pileup); demux and genotyping are host code. Sites: all-site is bounded by contig windows and site blocks (row 8).

## 3. Campaign worst cases

Source: `$W/data/corpus-beta/out/samples.csv.gz` (eligible = True, 97,534 samples; read pairs = `read_pairs_gex` else `read_pairs`) and
`$W/campaign/analysis/calendar-2026-11-10/calendar.md` (human 10x 3', 55,980 samples, corrected depth). Read-only. **Cell counts per sample
were not obtained**: the bridge jammed (my recursive grep), and the corpus table has no cell column; see row "cells" above for why it matters little.

Largest samples by read pairs (eligible, all assays): SRS13118566 (118.3 G pairs, 10x 3' v3, 4,865 runs), SRS13258861 (78.7 G),
SRS28730911 (64.6 G), SRS13118565 (37.8 G), SRS26028727 (22.3 G, 3' v2), SRS21025050 (21.1 G), SRS21065646 (20.9 G), SRS13253830 (18.9 G),
ERS31097003 (17.0 G, Flex), ERS31097002 (16.9 G, Flex). The 25th is 6.0 G. Counts above a size: >= 1 G: 3,096; >= 0.5 G: 14,944; >= 0.3 G: 34,079;
>= 0.2 G: 46,630; >= 0.1 G: 62,400 (of 97,534). Human 10x 3' (calendar): 18,648 samples at 80-300 M, 12,127 at 300-500 M, 6,817 at 0.5-1 G, 1,189
at 1-2.3 G, 105 above 2.3 G (675 G pairs, mean 6.4 G); 75 % of the 15.3 T pairs sit above 300 M. Reference points with measured logs: pbmc1k_v3 (66.6 M,
the node benchmark), GSM7656149 (263 M; 223 M barcoded; pileup 100 GB at K=1, 72 GB at K=2 (m)), GSM7064234 (1.198 G; engine declined, CPU 3 h 19 min).
Whether the campaign splits the largest samples into shards before the engine is not visible in the engine code (one `Input` = one sample).

Peak device memory by phase, current code (e), GB. Rows use rho = 0.475, nB = 0.88 n, U = 5 % of n capped at 20 M (pbmc1k_v3: 3.36 M of 66.6 M).
The index is excluded for prefill (shared holder). "-" = bounded by construction.

| phase (formula) | 66.6 M | 263 M | 1.2 G | 2.3 G | 6.4 G (tier mean) | 118 G |
|---|---|---|---|---|---|---|
| prefill, RC: 2.1 + 20 n + chunk (>= 2) | 5.4 | 9.4 | 28 | 50 | 130 | n/a (2^32 index limit) |
| hand-off: 4 + 19 n (records, rho = 0.475) + chunk | 5.6 | 10 | 27 | 48 | 126 | n/a |
| pileup resident 141 n, K -> infinity | 9.4 | 37 | 169 | 324 | 900 | n/a |
| pileup total at K=1: 396 n (25 GB at 60 M (m)) | 26 | 104 | 475 | 910 | 2,500 | n/a |
| all-site arena | <= 24 | <= 24 | <= 24 | <= 24 | <= 24 | <= 24 |
| non-host: 25.4 + 1.16 KB x U | 29.3 (m) | 40 | 49 | 49 | 49 | 49 |
| Solo: 39 n + 1 | 3.6 | 11 | 48 (two-pass) | host path (> 2^31 - 1) | host | host |

Which exceed a 31 GB slot (arena budget 31, pileup arena 28, all-site arena 24) and a 23 GB slot (pileup arena 20, all-site 22):

| phase | breaks 31 GB at | breaks 23 GB at |
|---|---|---|
| prefill (RC) | about 1.3 G pairs (keep + readRecord linear term) | about 0.9 G |
| hand-off | about 1.3 G (records resident; sooner when another slot shrinks the free memory) | about 0.9 G |
| pileup resident | **about 200 M pairs** | **about 140 M** (K=1 total: 26 GB at 66 M already needs K >= 2 at 23 GB) |
| all-site | never by size (CPU fallback only by the discovery / single-block conditions in row 8) | never by size |
| non-host | about 5 M unmapped reads (about 100 M pairs at 5 %) | always (index 25.4 GB alone) |
| Solo | never (two-pass), CPU above 2^31 - 1 reads | never, same |

At 23 GB with three slots a node also needs 16.4 GB (holder) + 3 x 23 = 85.4 GB plus three contexts: the non-host phase (25-29 GB) cannot start
when two other slots are in prefill (m: server OOM).

## 4. Fallbacks and OOM handling

| where | trigger | consequence |
|---|---|---|
| `engine_gpu.cu:1211` `failBudget` (sites 1310, 1313, 1762, 2024, 2065, 2257, 2351, 2758) | reads phase, one RC chunk, largest dedup partition, index + smallest lanes, or a lane's smallest batch do not fit the budget | run stops with "EXITING ... E17"; `SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu` (`SeqCacheGpuPrefill.cpp:1800`) maps every read with STAR instead |
| `engine_gpu.cu:3367` | `recNeed() > capRec` or any records allocation | "E8 records not built", hand-off file path, `e10Off` (level 5 lost; `[engine_gpu] fallback:` line) |
| `engine_gpu.cu:1398-1410`, `:3442` | E10 or E11 keep out of device memory | E10 off, STAR's loop does the per-read work (CPU); with `oomFatal` the call fails and the caller reruns |
| `records_gpu.cu:404+` `e2ePass` | `cudaMalloc` of records (not budgeted) | pass fails (observed 2 items at 3 slots) |
| `pileup_gpu.cu:117-160` | arena too small | widen (all slots of the card, holders killed by SIGUSR1) -> unified memory (random access over PCIe) -> `po.log` "GPU memory: X MB needed" -> `pileupRun` false -> `[gpu_pileup] failed` (exit 9 once CPU steps were skipped) |
| `allsite_gpu.cu` returns, `allsite_device_glue.h:233` | see row 8 | `[allsite-gpu] host pass`: all-site runs on the CPU |
| `solo_gpu.cu:1354-1432` | need > 97 % of free memory | two-pass; on OOM halve and retry 3 times; then host Solo |
| `nonhost_gpu.cu:643-660` | any CUDA error (incl. OOM, "resource already mapped" at 2 slots, 12 of 40 runs (m)) | 4 retries, then the CPU screen (`singlet.cpp:1802`) |
| lane B overflow | read exceeds a capacity | host pool (by design, exact) |

Deep-sample conditions that trigger them today: pileup resident above about 200 M pairs (widen, unified memory, exit 9); hand-off records above about
1.3 G pairs; non-host above about 5 M unmapped reads at 31 GB (no fallback at 2 slots because it is outside the budget, but it OOMs when the other slot
is in prefill with a full budget: 16.4 + 31 + 25.4 + work > 93); Solo above 2^31 - 1 reads; `Ip` and `readRep` are `uint32` (4.29 G reads); `nRec >= 2^32`
refused (`engine_gpu.cu:3044`).

## 5. Design

Principles: (a) every device byte is requested from one arena of fixed size per slot; (b) every phase has a block loop whose block size is derived
from the arena, and a request that fails halves the block (floor: a block that fits the arena's fixed overhead, about 2-3 GB, or the config is
rejected at start); (c) blocks are formed on the host or by a device radix partition so that each block is processed contiguously, no random
gather over an array larger than the block; (d) double-buffered H2D / compute / D2H on two streams with pinned staging (the `Stager` / `D2H`
classes already exist).

### 5.0 Slot arena (cross-cutting)

- `SlotArena` (new header): one `cudaMalloc` of `budget - fixed overhead` at phase start (a 3/4 retry loop as in `allsite_gpu.cu:570`), 2 MB aligned
  bump allocation with `mark`/`reset` and a second allocator from the top for transients, `try_alloc` returning null. thrust goes through the existing
  `TAlloc` (`engine_gpu.cu:724`), CUB through the arena (`cub_run` already does).
- Replace raw `cudaMalloc` in `hostloop_gpu.cu:305`, `records_gpu.cu:130-205,398`, `solo_gpu.cu:75,116`, `nonhost_gpu.cu:206`, and the `cudaMemGetInfo`-driven sizing in
  `records_gpu.cu` (m0), `solo_gpu.cu:1346,718`, `pileup_gpu.cu:129`, `allsite_gpu.cu:563`, `engine_gpu.cu:1744` (`roomRC`).
- Fixed overhead per slot is explicit: context (0.5-1 GB) and lane M's warp stack (3.96 GB, only when warp lane is chosen). Budget = (card - 2 GB - shared index) / S - overhead.
  This is what makes a node's sum provable: index + S x (arena + overhead) <= card. The guarantee is by admission (flock slots), not by the driver, so a lint test
  (grep for `cudaMalloc` outside the arena header) belongs in ctest.
- Squeeze test: a helper process that holds `card - index - S x budget` so free memory equals the plan; the 8-sample identity gate must pass with zero fallback lines.
- Persistent context + arena per slot would also remove the repeated contexts (prefill process, pileup reset, all-site driver context, non-host new context; the last
  failed in 12 of 40 runs). The gpu-pool server and the IPC import code of the shared index (`shared_index.cuh`) are the starting point; not required for the cap.

### 5.1 Per phase

| phase | blocking key | host sort / partition | streaming | effect |
|---|---|---|---|---|
| reads / dedup | read chunk (RCN), h1 hash partition | counting partition on the host (exists) | RC already double buffers rc chunks (`rpipe`) | make RC the plan for any n above 25 % of budget, not only when lean fails |
| prefill residue | none (n-linear: keep 16 n, readRecord 4 n, dHN n) | keep CR/UR written per read chunk straight to pinned host (as `soloKeepToHost` does later); readRecord uploaded per record chunk; dHN per partition | D2H per chunk | removes 20-24 B per read from the device: prefill needs only index + chunk, any n |
| lanes | distinct-sequence chunk x batch (exists) | keep rep order; results D2H per batch (split mode) | decouple the chunk from device room: chunk is a host unit, device holds only in-flight batches | measure NB sweep at fixed budget first: at 24 GB lane B's 11 k batch cost +18 s (m); in-flight above about 2 x resident threads gives no parallelism, so the loss is probably per-batch tails, not occupancy |
| records / E10 products / hand-off | record block (sig range) | reads partitioned by record block on the host (stable, keeps read index), outputs written back by read index, keep-scan on the host | block: records block + its reads; products per block merged by key (the `e2eBuildRC` logic) | records no longer resident: n-independent device use |
| pileup | cell-hash block K (exists), plus signature block for the sig-level stage | (1) sig-level stage (annotate, resolve, lists) per signature block with a counts pass then fill pass, outputs kept as 58 B per record; (2) reads streamed in blocks, partitioned by cell hash on the device with a stable histogram scatter into K pinned host buckets (order preserved); (3) each bucket H2D, sorted by signature (sequential gathers), reduced as today | blocks double buffered; K up to 4096 (uint16 `pid`) instead of 64 | resident 160 -> about 28 B per pair, i.e. a 31 GB slot holds about 900 M pairs, a 23 GB slot about 650 M; beyond, emit tuples per signature block and merge buckets externally (design step 5.2 item 5) |
| all-site | contig window, site block (exists) | resplit into position windows (exists) | upload of chunk k+1 during chunk k (exists, `Prefetch`) | add a discovery block: split a chunk in halves when nObs x 48 B exceeds the arena left, instead of the arena exception |
| non-host | hash-prefix block of the sorted index x read block of U | reads: sort the Q minimizers (radix), partition by hash prefix; index: 2-4 GB blocks streamed from pinned host (the pool server already keeps it pinned) | index blocks double buffered, 25 GB over PCIe about 0.6 s per read block (the same bytes the sample uploads today) | device use about 8 GB + read block (2-3 GB per 4 M reads), independent of U; random probes become a merge join over a sorted block |
| Solo | read chunk, key bucket (exist) | chunks as today | budget-based C; counters 64-bit so N > 2^31 - 1 stays on the device | no host fallback below the host limit |

### 5.2 Ranked proposals

Value = effect on the campaign (robustness for deep samples first, node rate second). Effort in engineer-days (e). "3 slots" = needed or
enabling for 23 GB slots.

| rank | change | value | effort | 3 slots |
|---|---|---|---|---|
| 1 | Slot arena + budget-based sizing in every phase (5.0): no `cudaMemGetInfo` sizing, no raw `cudaMalloc`, shrink-and-retry; squeeze test and lint | high: removes the free-memory race (hand-off OOMs, non-host server OOM, Solo/all-site/pileup each sizing from the same free memory) and makes the fallbacks of section 4 deterministic | 4-6 | required |
| 2 | Prefill residue off the device (keep, readRecord, dHN) and RC as the plan above 25 % of budget | high for deep: prefill and hand-off limits from 1.3 G to unbounded pairs per slot | 3-4 | yes (0.9 G limit at 23 GB) |
| 3 | Pileup in two steps: (a) allocate inputs and alignment-level tables in a top (transient) region freed after the signature lists (resident 160 -> about 90 B per barcoded read; limit 200 M -> 255 M pairs at 31 GB; S-sized); (b) signature-block stage with counts and fill passes (resident about 65 B; 480 M pairs) | high: 75 % of the campaign's pairs are in samples above 300 M; today they widen the card or fail | (a) 2-3, (b) 8-10 | (b) yes: 20 GB arena holds 350 M pairs |
| 4 | Pileup streamed reads + external cell partition (K up to 4096, host buckets, stable) | high for >= 0.5 G pairs; also frees the K select passes over all reads (K x 16 B x nB read traffic) | 8-10 | yes |
| 5 | Non-host index streaming by hash block | high for 3 slots (25 GB outside the budget today); medium for deep (U above 5 M); removes the random probes | 6-8 | required |
| 6 | All-site discovery block + arena-exception shrink | medium: the only unguarded CPU-fallback condition in a bounded phase | 2 | no (arena already min(24, budget - 1)) |
| 7 | Record-block pass for hand-off + E10 products (reads partitioned by record block) | medium: matters above about 1.3 G pairs only | 8-12 | for 23 GB above 0.9 G |
| 8 | Solo 64-bit counters (N > 2^31 - 1) and budget-based chunk | low-medium: about 300 samples above 2.1 G pairs (human 10x 3', corrected) | 3-4 | no |
| 9 | Persistent per-slot context + arena in the pool (IPC import like the index) | medium: removes context churn and the "resource already mapped" CPU screens; allocation counts drop (about 15 `dmal` per lane run today); size of the gain not measured | 10+ | helps |
| 10 | Lane in-flight decoupled from chunk (ring of in-flight reads, persistent kernel) | speculative: only if the NB sweep shows the +18 s at 24 GB is per-batch tails | 10+ | would stop the hold lengthening |

Throughput expectations (honest): with 2 slots the slots are 83-91 % busy and >= 1 request waits 45 % of the time; 3 slots cut waiting to 7-19 % but at
23 GB every hold got longer (map hold 1 about 69 s vs about 50 s), so the node lost 10 % (8.28 vs 9.19 steady (m)). A third slot is worth at most
10-20 % (the results note says the same) and only if rank 1, 2, 5 and the NB-sweep fix keep the holds at their 31 GB length. The campaign value of ranks 1-4
is not speed but that the 75 % of pairs in samples above 300 M run on one slot without widening the card (which today stalls every other sample for up
to 120 s plus the pileup) and without a CPU fallback. Fewer allocations: not measured; expect under 1 s per item (cudaFree synchronises the device).

Memory floor after ranks 1-5: index (shared) + per slot about 8 GB of blocks + 1 GB context, so memory stops being the slot limit; four slots of 16 GB
would fit, whether the GPU has the compute for them is a separate measurement.

### 5.3 Risks and checks

- Identity: the stable partition order of reads must be kept (the pileup's SNP first-read and UMI collapse inputs; `pileup.cu:1179` is a stable
  `DeviceSelect`). RC records order differs in ties (documented in the code, content identical). Every step passes the 8-sample gate with zero fallback lines, plus a
  deep gate on GSM7656149 and a 1 G-pair sample under `SINGLET_ENGINE_GPU_MEM_GB=23`.
- The pileup's signature-block stage needs the global junction id table (`k_sjkey`, one sort over all list entries): do it per block and merge the unique keys
  (distinct junctions are small), not as one sort over n_sjl.
- 32-bit indices (`Ip`, `readRep`, `nRec`, pileup `int` counts, Solo 2^31): samples above 2^31 reads need a shard layer above the engine; the merge of a UMI collapse across shards is
  cell-keyed, so shards must be cell-hash shards of the hand-off output, not read ranges. That belongs to the pipeline, not this note. Host RAM
  (84 GB on nuclei2k_v2, OOM at 388 GB for 1.2 G pairs (m)) is a separate limit.
- Unverified assumptions: the 1.16 KB per unmapped read is one measurement; rho at depth is not measured (rho = 1 is the bound); 20 M subsample cap of the screen is from the
  run-time reference.

## 6. Host work inside the pileup hold (rt/deeppile, deeppile3)

What runs on the host while the slot lock is held (`pileupRun`, called by `gpu_pileup_apply` after `lockDevice`; the caller owns the lock, so nothing of
pileup.cu can run before it is taken without a new entry point in pileup_engine.h, which this branch does not touch):
1. Whole-table stream path: signature stage on the device (about 1 s on d300_3p), then `k_wsig` and a pageable D2H of the per-signature weights; **host sizing**
   (`host_bins`: one scan of all reads for K=1, then one scan per batch of 8 trial Ks, a hash and a 64-bit modulo per read and K); **host grouping**
   (`host_partition`: two scans). d300_3p job 815486: sizing 6.21 s + grouping 2.07 s of a 23.1 s `run_pipeline`.
2. Per partition: `host_gather` of the partition's reads (and masks) into pageable buffers, pageable `cudaMemcpy` H2D, the compute (default stream, scalar
   D2H syncs inside the reducers), then the fetch: `fetch_csr` and the other downloads into freshly allocated pageable arrays (page faults in the driver's
   staging copy, about 2 GB/s), their host halves deferred past the unlock (E24).
3. Sigblock path additionally: the blocked signature stage (pageable H2D per block, per-block host loops, pageable D2H of the lists into fresh arrays), the
   per-partition table rows gathered on a thread (`part_prepare`), `host_tabs` scans per trial K.
The `part.select` stage in the E23 stage line is an event interval that for partition 0 also spans everything since the signature stage (sizing, grouping): on
d1000_3p 37.5 of its 40.2 s were the sizing scan (job 815819 host profile: select proper 3.5 s).

Changes (each switch default on, `=0` restores the earlier code; outputs identical, `test_gpu_pileup_stream` 474/474):
- `SINGLET_PILEUP_SIGBLOCK` = auto (default): blocked only if max(signature-stage estimate, partition-stage estimate) exceeds the arena; the estimate is logged;
  an `ArenaExhausted` of the whole-table layout in auto mode repeats attempt 0 blocked before widening. `=1` forces, `=0` off.
- `SINGLET_PILEUP_CELLTAB`: one scan gives per-cell weight sums and read counts, every trial K's bounds are sums over cells; cell->partition table for the
  grouping; per-signature row weights made once and several trial Ks per bitmap scan for the sigblock bounds.
- `SINGLET_PILEUP_PIN`: partition buffers page-locked (cudaHostAlloc, malloc on failure). `SINGLET_PILEUP_PIPE`: the next partition's reads/masks/indices are
  gathered on a thread into the other of two pinned buffers while this one runs (whole-table path; the sigblock path already prepared on a thread).
- `SINGLET_PILEUP_D2HPIN`: result downloads go through a ring of eight 16 MB pinned chunks, each chunk copied into its destination by its own thread;
  the CSR downloads no longer zero-fill their destination.
- Log: a `host profile (s)` line (sizing scan with cells/tabs split, grouping, select with gather/upload split, compute, fetch, pinned allocations).
