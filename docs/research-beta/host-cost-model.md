# Host cost of `--engine gpu`: where the core-seconds go, and the fixes (2026-10-29)

Question: 10x 5' GSM6470208 costs 18.7 core-s per M reads on the GPU path (2,274 core-s, 121.8 M reads, 18 threads, 564 s wall). The break-even is
12.7, so 731 core-s have to go. Where are they, and what removes them without changing an output?

**Status: nothing here was compiled or run on Clipper.** Three changes are on the branch, each marked "not compiled" in section 5; W11 builds
and gates them. The 5' stage split is a model calibrated to the measured total (section 2), because no stage profile of a 5' run exists.

## 1. Summary

- **The cost is work, not idle threads.** On pbmc1k_v3 (the one hand-off-era run with a per-second CPU trace) 18 threads x 394 s wall is 7,092
  thread-s and 991 are busy (14%). About 290 s of the wall runs on one core (1.0-1.2). Idle threads cost no core-s; the serial stretches
  matter for wall, and the busy single core in them is real core-s (130 core-s in the three serial pileup stages + Solo counting).
- **Cost scales with distinct sequences and with alignments, not with reads.** The hand-off is built once per distinct sequence: restore,
  `alignBAM` per alignment, the summary, then parsed again by the pileup. 5' has far more alignments per read than 3':
  331.7 M records (222 M secondary) for 121.8 M reads in GSM6470208, against 73.9 M for 66.6 M reads in pbmc1k_v3.
- **5' pays for something no 3' sample does: STAR's adapter search, on the CPU, for every read.** 10x 5' passes `--clip3pAdapterSeq`
  (the 30-base TSO) to STAR. `ClipMate::clip` calls `localSearch` in `readLoad` for each read, and `localSearch` is a plain double loop with no
  early exit: about 2,300 compare steps per 91-base read, about 280-490 core-s for this sample. The GPU clips the same reads again for its own
  dedup.
- **Fixes implemented (central estimate for GSM6470208, per M reads):**

  | | fix | saves | output identical |
  |---|---|---|---|
  | F1 | `localSearch` with early exit | 2.5 (2.0-3.0) | yes: same `ixBest`, checked on 440,000 cases |
  | F2 | hand-off record built without `alignBAM` | 1.5 (0.7-2.0) | yes: same fields; `SINGLET_HANDOFF_LEAN=verify` compares |
  | F3 | no blob copy, `reserve` where vectors doubled, no per-read `std::string`, prefetch in the serial hand-off pass | 0.2 (0.1-0.3) | yes |

  That is about 4.2 per M reads (2.9-5.2): 18.7 -> **about 14.5 (13.5-15.8)**. **This does not reach 12.7 on its own.** Section 4 lists what has
  to follow; the 5' stage profile asked for in section 6 decides which.

## 2. Cost model

### 2.1 Measured: pbmc1k_v3, 66.6 M reads, 18 threads, `SINGLET_GPU_PILEUP=on` (job 791104, build f087be14)

W11's `stage_profile.sbatch` run: every log line stamped with the wall clock, the job cgroup's CPU counter every second
(`jobs/w11/engine/results/profile-791104/`). `bench/host_cost/stage_profile_handoff.py` attributes core-s between log markers:

| stage | wall s | core-s | cores | per M reads |
|---|---|---|---|---|
| decode `.1fq` to FASTQ text | 25.9 | 106 | 4.1 | 1.60 |
| references, STAR start, genome load | 14.4 | 18 | 1.2 | 0.27 |
| **GPU prefill** (reading R2, `mapDistinct`, host lane, hand-off build, inserts) | 84.4 | 379 | 4.5 | 5.70 |
| **STAR per-read pass** (`readLoad` + clip, barcode, Solo record, summary replay) | 33.5 | 212 | 6.3 | 3.19 |
| STAR Solo counting (serial) | 40.4 | 41 | 1.0 | 0.62 |
| pileup hand-off pass (serial) | 47.1 | 48 | 1.0 | 0.72 |
| `prepend_handoff` + GPU input (serial) | 9.0 | 9 | 1.0 | 0.14 |
| GPU pileup + apply + host pass (serial; GPU part 4.3 s) | 30.5 | 31 | 1.0 | 0.47 |
| pileup finish | 2.9 | 3 | 1.0 | 0.04 |
| non-host screen, EM, writers | 18.1 | 41 | 2.3 | 0.61 |
| export (matrices, QC, demux, writers) | 87.9 | 101 | 1.2 | 1.52 |
| **total** | **394** | **991** | **2.5** | **14.88** |

The prefill splits further from the per-second trace: about 154 core-s in the 11 s of reading R2 at 14-18 cores (the one-`push_back`-per-base
conversion, fixed in cfd66207, which this build predates), 65 core-s on one core for the 60 s of `mapDistinct` (upload, dedup, record assembly,
waiting), and about 160 core-s in the 10 s burst of the hand-off build at 18 cores (31.8 M records: 5 us per record). The run is 14.9 per M
reads; the fixed build measures 11.4-11.5 (job 791114).

### 2.2 Model: GSM6470208 (10x 5' v2), 121.8 M reads, `on`, 18 threads (jobs 791995/791997)

Measured: 2,274 core-s, 564 s wall; decode 30.2 s; `Pileup` stage 336 s (STAR start to pileup end); export 81.4 s; 108.8 M reads skip BAM,
87.9 M take alignments from the prefill, 88.5 M barcoded; 331.7 M alignment records, 222 M secondary; 12.3 M unmapped reads to non-host
(259 EM iterations); GPU 2.0 s of 9.7 s total.
Not measured: every other row. They use pbmc1k_v3's per-unit rates and sum to 2,274 by construction (the last row is the remainder).

| stage | model core-s | per M | basis |
|---|---|---|---|
| decode | 125 | 1.0 | 30.2 s wall at about 4 cores, as pbmc1k_v3 |
| references, STAR start | 20 | 0.2 | as pbmc1k_v3 |
| prefill: read R2 (fixed build) | 25 | 0.2 | parallel LUT conversion |
| prefill: `mapDistinct` host (serial) | 130 | 1.1 | 1.7 core-us per distinct sequence x about 75 M (guess: 5' reads are less duplicated) |
| prefill: hand-off build | 450 | 3.7 | about 75 M records x (restore + summary 2.5 us + 3.4 alignments x 1.2 us of `alignBAM` etc.) |
| STAR per-read pass, without the clip | 390 | 3.2 | pbmc1k_v3 rate: parse, barcode, Solo record, summary replay |
| **STAR per-read adapter clip (`localSearch`)** | **400** | **3.3** | 2,295 steps per 91-base read x 121.8 M at 1.2-1.7 ns (a branch per step, 75% mismatches); range 280-490 |
| STAR Solo counting (serial) | 80 | 0.7 | pbmc1k_v3 rate |
| pileup hand-off pass (serial) | 90 | 0.7 | pbmc1k_v3 rate x 108.8 M reads |
| `prepend_handoff`, GPU apply, host pass (serial) | 80 | 0.7 | pbmc1k_v3 rate per record and per barcoded read |
| non-host | 130 | 1.1 | 12.3 M unmapped reads against 2.9 M, EM 259 iterations |
| export | 150 | 1.2 | 81 s wall at about 1.8 cores (demux auto-K and VB are most of it) |
| other (stream-path BAM records of the 13 M reads without a record, system time, GPU waits) | 204 | 1.7 | remainder |
| **total** | **2,274** | **18.7** | measured |

The uncertain rows are the prefill's (the number of distinct sequences and alignments per record) and the clip (the compiler's code for the
inner loop). Section 6 says which counters settle them.

### 2.3 Why 5' costs 18.7 and 3' 10.7-12.9

| sample | reads | core-s | per M | what differs |
|---|---|---|---|---|
| GSM7036403 (3') | 111.6 M | 1,195 | 10.7 | |
| GSM4006843 (3') | 66.7 M | 857 | 12.9 | |
| GSM7656149 (3') | 263.4 M | 3,410 | 12.9 | |
| GSM6470208 (5') | 121.8 M | 2,274 | **18.7** | 3.4 alignments per record; adapter clip on the CPU; 10% unmapped reads |

## 3. Serial sections and idle threads

pbmc1k_v3 (section 2.1). Average cores per 5-second window, from the cgroup trace:

- 0-35 s: decode in bursts of 11 cores; references and genome load on one.
- Prefill: 11 s at 14-18 cores (reading R2), then 60 s at exactly 1.0 (`mapDistinct`: upload, dedup, assembling `out.seq` / `out.blob` record by record,
  waiting on kernels), a 2 s burst of 12-13 cores (host lane), then 10 s at 18 (hand-off build and inserts).
- STAR per-read pass: 7.0 cores for 30 s. The reader thread (`chunkRead` 30 thread-s in STAR's own counters, job 790043) feeds the workers; the
  pass looks reader-bound (inferred, not profiled with the hand-off), so most of the 18 threads wait.
- After STAR: 130 s at exactly 1.0: Solo counting 40 s, hand-off pass 47 s, `prepend_handoff` + input 9 s, GPU apply + host pass 31 s.
- Export 88 s at 1.2 cores: donor demux (auto-K 29 s, VB 43 s) is single-threaded in practice.

Parallelising the serial stages shortens the wall but leaves core-s unchanged; the metric here is core-s, so the fixes below remove work.
(Where the node's cores bind, shorter wall is also worth having: the three post-STAR pileup stages are 87 s of a 394 s run in pbmc1k_v3.)

## 4. Ranked fixes

Savings are for GSM6470208 per M reads; "3'" is the effect on pbmc1k_v3. Rank = estimated core-s saved.

| # | fix | where | saves 5' | saves 3' | identical | status |
|---|---|---|---|---|---|---|
| F1 | `localSearch` early exit (adapter clip in `readLoad`) | `src/star/SequenceFuns.cpp` | **2.5** (2.0-3.0) | 0 (no adapter) | yes | implemented, not compiled |
| F2 | hand-off alignment fields without `alignBAM`; one append per record | `src/star/ReadAlign_seqCache.cpp` | **1.5** (0.7-2.0) | 0.3-0.45 | yes (verify mode) | implemented, not compiled |
| F3 | blob copy removed; `reserve` for `part`/`sumPart` and `prepend_handoff`'s vectors; no `std::string` per read; prefetch in the hand-off pass | prefill, pileup headers | 0.2 (0.1-0.3) | 0.15 | yes | implemented, not compiled |
| N1 | GPU emits each read's clip lengths; STAR skips its CPU clip for hand-off reads and `handoffReadEntry` takes the lengths from the GPU | `engine_gpu.cu` + `ReadAlign_oneRead.cpp` | 0.4-0.6 after F1 | 0 | yes | patch described, 5.1 |
| N2 | `mapDistinct`: reserve + one copy in the record assembly, no `std::vector<uint32_t> repc` copies, `readRep`/`posOf` in one pass | `engine_gpu.cu` | 0.3-0.6 (serial 1.7 core-us per distinct sequence) | 0.4 | yes | patch described, 5.2 |
| N3 | `seqCacheHandoffSummary`: no `Stats` per record, one `put` per field group; `classifyAlign` only for nTr>0 already | `ReadAlign_seqCache.cpp` | 0.1-0.3 | 0.1 | yes | not done |
| N4 | FASTQ text round trip: `.1fq` decode writes R1/R2 text (26 GB for pbmc1k_v3), STAR parses it, the prefill parses R2 again | `singlet.cpp`, STAR input | 0.5-0.8 | 0.5 | yes | not done (large) |
| N5 | non-host EM stops at a tolerance instead of 259 iterations; unmapped reads (12.3 M) | non-host | 0.5-1.0 | 0.1 | **no** (EM iteration count, already a known difference D4b) | not done |
| N6 | export: skip auto-K and VB demux when one donor is evident | export | 0.5-1.0 | 0.5 | **no** (donor outputs) | not done |

F1-F3 plus N1 and N2 are about 5.2 (central) and bring 5' to about 13.5: above 12.7. N4 would bring it to about the break-even. The honest
reading: **F1 and F2 are worth doing now; whether the rest is needed depends on two numbers only a 5' profile gives** (section 6).

## 5. What changed on this branch

### Files

- `src/star/SequenceFuns.cpp`: **F1.** `localSearch` returns the same index as before. A candidate position is dropped as soon as it cannot qualify.
  With k mismatches among its m positions, nMM >= k and nMatch <= m - k, so nMM/nMatch >= k/(m-k), and a correctly rounded double division is
  monotone, so `kmax[m]` (the largest k with k/(m-k) <= pMM, built with the same double arithmetic, once per thread and pMM) is an exact bound.
  Also `nMatch <= m` and m only falls as ix grows, so the loop stops when m < nMatchBest.
- `src/star/ReadAlign_seqCache.cpp`, `ReadAlign.h`, `SeqCache.h`: **F2.** `seqCacheLeanAln` computes refID, pos, flag, MAPQ and the CIGAR from the
  transcript with the same expressions and integer widths as `alignBAM` for a single-end read, and `seqCacheAlignRecords` builds the record in
  a thread-local buffer, appended once. It no longer fills `Read0`/`Qual0` (only `alignBAM` reads them). A transcript with a mate gap (canonSJ -3)
  falls back to `alignBAM`. `SINGLET_HANDOFF_LEAN=0` uses `alignBAM` for everything; `=verify` computes both, uses `alignBAM`'s, and counts
  differences (`g_handoffLeanBad`, logged as `ENGINE_GPU: lean alignment fields: N alignments, M differ from alignBAM's`).
  Pointer overloads of `seqCacheRestore` and `seqCacheAlignRecords` remove the per-record `blob.assign`.
- `src/star/SeqCacheGpuPrefill.cpp`: **F3.** calls the pointer overload; `reserve` for `part`, `partOff`, `sumPart`, `sumPartOff`.
- `include/singlet/pileup/gpu_pileup_bridge.h`: **F3.** `prepend_handoff` reserves `al`, `op`, `sq` from the hand-off's byte size (upper bounds;
  untouched capacity costs no memory) so they never reallocate.
- `include/singlet/pileup/pileup_engine.h`: **F3.** in `gpu_handoff_pass`: prefetch of `rec_off[...]` and the record 16 and 8 reads ahead;
  the R4 sample key `"r<i>"` is built in a stack buffer and only when the read is long enough to be sampled (it was a `std::string` per read).
  The file has non-UTF-8 bytes: diff it as binary or with `git diff -a`.
- `bench/host_cost/`: `model_localsearch.py`, `model_lean_aln.py` (laptop checks), `stage_profile_handoff.py` (the stage table above).

### Checks done on the laptop (Python 3.11, numpy, numba)

- `python bench/host_cost/model_localsearch.py`: transcription of the old and new `localSearch`; 440,000 cases (random 20-91-base reads against the
  TSO with MMp 0.1, planted adapters with mismatches, Ns in the read and in the adapter, pMM = 0, 0.05, 0.1, 1/3, 0.5, 1, 2, 1e9, inf, -0.1, NaN,
  adapters of 1-650 bases): identical `ixBest` every time. Inner steps per read: 1,231 -> 178 (6.9x fewer) on 20-91-base reads, 4,246 -> 606
  for the polyA adapter.
- `python bench/host_cost/model_lean_aln.py`: transcription of the single-end branch of `alignBAM` against `seqCacheLeanAln`, with the C++
  integer widths: 300,000 random transcripts (junctions, annotated and unannotated, deletions, insertions, both strands, primary and not,
  clips, NH steps, MAPQ, flag AND/OR); identical fields. This checks the transcription, not the build.
- `python bench/host_cost/stage_profile_handoff.py <dir> <M reads>` reproduces section 2.1 from the job 791104 files.

### What W11 must build and gate

1. **Build** the CUDA binary (`build_clipper.sbatch`) from this branch. None of the C++ was compiled. Likely places to look if it does not: the
   `static thread_local std::vector<uint>` in `localSearch` (uses `DEF_readSeqLengthMax`, `std::max<uint>`); `g_handoffLeanAln/Bad` (defined in
   `ReadAlign_seqCache.cpp`, `extern` in `SeqCache.h`); the second `seqCacheAlignRecords` overload; `__builtin_prefetch` (GCC).
2. **F2 verify run:** `SINGLET_HANDOFF_LEAN=verify` on GSM6470208 and pbmc1k_v3 with `SINGLET_GPU_PILEUP=on`: the log must say
   `0 differ from alignBAM's`. Then `SINGLET_HANDOFF_LEAN=0` and the default must give the same outputs.
3. **Gate (a), `gpu_pileup_gate.sbatch`,** CPU pileup against `on`, on GSM6470208, GSM4006843, GSM7036403 and one 3' short sample. The CPU
   mode's BAM comes from STAR's own `alignBAM`, so any difference in F2 shows as a changed count. Only the 7 known differences should remain.
   F1 changes only how fast the clip is found: a wrong `ixBest` would change clip lengths, alignments and every count on GSM6470208.
4. **Cost:** GSM6470208 `on` at 18 threads, core-s against 2,274 (this branch should read about 1,750; 1,550 is the break-even).
5. **Stage profile of GSM6470208** (section 6), before and after, so the model rows become measurements.

### Patches for the two files I did not touch

**5.1 / N1: clip lengths from the GPU** (`engine_gpu.cu`, `k_clip` already computes them). `Output` gets `std::vector<uint16_t> clip5, clip3` (per
read: bases clipped from each end, from `dCs`/`dCl`/`hcs`), filled in `mapDistinct` after `k_clip` (`clip5[i] = hcs[i] - in.off[i]`, `clip3[i] =
(in.off[i+1]-in.off[i]) - rlen(i) - clip5[i]`). `ReadAlign_oneRead.cpp`: for a read with `g_handoffReadRecord[iReadAll-1] != 0xFFFFFFFF` and a
summary, call `readLoad` with the clip off (the `clipMates[im][..].clip` calls in `readLoad.cpp:75-76` skipped) and set
`clipMates[0][0].clippedN = clip5`, `clipMates[0][1].clippedN = clip3` from the new arrays, which `handoffReadEntry` reads. Check by
`SINGLET_HANDOFF_CLIPCHECK=1` running the CPU clip as well and counting differences. After F1 the CPU clip is about 0.6 us per read.

**5.2 / N2: record assembly in `processChunk`** (`engine_gpu.cu`, the "records" loop). Before it:

```cpp
u64 nOk = 0, bytesSeq = 0, bytesBlob = 0;
for (u64 k = 0; k < repc.size(); k++) bytesSeq += rlen(repc[k]);
bytesBlob = A.out.size() + M.out.size() + B.out.size();
for (const auto& h : hout) bytesBlob += h.size();
out.seq.reserve(out.seq.size() + bytesSeq); out.blob.reserve(out.blob.size() + bytesBlob);
out.seqOff.reserve(out.seqOff.size() + repc.size()); out.blobOff.reserve(out.blobOff.size() + repc.size());
```

(the vectors currently double, copying GBs at 16 M-sequence chunks). Same file, after the chunks: build `readRep` and `posOf` in the same
pass over `order`. Output unchanged byte for byte.

**Nothing is needed in `pileup.cu`.** The serial stages in `pileup_engine.h` (hand-off pass, `gpu_host_pass`, `gpu_put` accumulations) are host
code in a file I could change.

## 6. What to measure next (no job was run for this)

One job, `stage_profile.sbatch` with the hand-off setup of job 791104 (`SINGLET_GPU_PILEUP=on`, `SET=GSM6470208 T=18 ENGINES=gpu`) on a build
with `-DSTAR_PROFILE`, and `bench/host_cost/stage_profile_handoff.py` on the result. The numbers that turn the model into measurements:

1. STAR's `Log.out`: `ENGINE_GPU: engine_gpu: N reads, D distinct, ... host lane ...` (D = the distinct sequences), `hand-off ... R records, B bytes in S s`
   (B/R is the alignments per record) and `cache filled ... (reading reads, GPU, cache inserts)`.
2. `STAR_PROFILE readLoad=` thread-s: pbmc1k_v3 has 53 (0.8 us per read); the clip adds the difference. This is the direct measure of F1's target.
3. The cores per second in the prefill (the 1.0-core stretch gives `mapDistinct`'s serial seconds; the 18-core burst gives the hand-off build).
4. The same with F1-F3 in the build.

If `readLoad` is near 53 thread-s the clip is cheap and F1 saves little: then N4, N2 and the hand-off build are the rest of the gap. If it is near
250, F1 alone is about half of it.

## 7. Not done, and why

- **Parallelising the serial pileup stages** (hand-off pass, `gpu_host_pass`): shortens wall, not core-s.
- **`alignBAM` replaced elsewhere** (the non-hand-off path writes real BAM records): needed for those reads.
- **N4, N5, N6**: too large or not output-identical for a branch that cannot be compiled or run here.
- **Skipping STAR's Solo counting** (0.6-0.7 per M): `CellReads.stats` feeds `cell_qc_metrics.tsv`.
