# E19: the GPU path as the default of `--engine gpu` (release)

Experiment E19 ("release") of the GPU aligner program. Branch `gpu/fable/release`, from `gpu/fable-engine` at e0e1514a (the E16
report). It was merged with `gpu/fable-engine` at 503c07f7 (E17 "budget") before the final gate (a8c25a48). Goals:
1. level 5 + ingest + E16 become the default of `--engine gpu`, behind the existing fallbacks, and every fallback says so on stderr;
2. every sample of `pipeline-benchmark/data/` is byte-identical between the CPU engine and the GPU path (the all-sample gate);
3. a catalogue of the declines, and the operator's page `GPU-PATH.md`.

The CPU engine is unchanged. The operator's view (stages, switches, fallback lines, memory, slots, the gate) is `GPU-PATH.md` in this
directory. The draft PR description for `gpu/fable-engine` → `prod/campaign` is §9.

## 0. Answer in brief

- **Every sample is identical to the CPU engine on the merged head (gate 798027, binary ced7f553 = E19 + E17, T = 18, H100 NVL,
  against the CPU references of gate 797474; §2a).**
  - 10 runs: the 9 samples of `pipeline-benchmark/data/` and pbmc1k_cite_v3's antibody library.
  - In each, out_equiv finds only E11's run-dependent files: 4, once the read files are set aside.
  - The 10 STAR Solo files are byte-identical, and so are `SJ.out.tab` and the mapping statistics.
  - The unmapped mates have the same records, and `Gene/CellReads.stats` has the same rows.
  - The one exception is not the GPU path: nuclei2k_v2's `nonhost_reads.fq` has the same count (2,459,477) but other records. It is
    E15's known non-host writer bug (§4d), and the gate names it.
- **Every sample reaches level 5 with the ingest, with no fallback line.** That includes the barnyard (hgmm1k_v3) now that E17 is
  merged.
- **Over the 10 runs the GPU path takes 3.4x less wall time and 4.2x fewer core-s:** 971 s / 4,603 core-s against 3,303 s / 19,388
  core-s.
  - Per sample, wall time is 1.5x faster (pbmc2m, 2 M reads) to 4.0x faster (pbmc1k_v3, pbmc5p_v1, hgmm1k_v3).
  - Core-s fall by 1.3x to 5.2x (§3).
  - pbmc1k_v3 takes 101.5 s / 548 core-s, against 403.9 s / 2,507 core-s.
  - Host memory is higher on the large samples: nuclei2k_v2 peaks at 84.4 GB of RSS against 34.9 GB.
- **The defaults (§1a).** `--engine gpu` with `SINGLET_GPU_PILEUP=on` runs level 5, the ingest and E16's tail, with no other variable.
  - The opt-outs are `SINGLET_ENGINE_GPU_E2E=0` (the path before E8) and `SINGLET_ENGINE_GPU_INGEST=0`.
  - Every fallback prints one line on stderr, and every run prints the level it reached (§1b).
  - The level-4 hole now stops the run instead of writing incomplete outputs: STAR exits with 104 (§1e).
  - The CPU engine is untouched, and `--help` documents the switches.
- **New: a GPU probe before STAR runs (§1d).** On a node whose GPU CUDA cannot see (g050, §5.1), the run used to end with exit 9. It now
  keeps the CPU pileup and says so.
- **Declines (§4).** Neither the ingest nor any level declines on any sample.
  - Before the merge, hgmm1k_v3 fell back to STAR because the engine ran out of memory. The fallback was loud, and the outputs were
    still identical.
  - §4b lists each remaining decline condition and what a port would take. None of them is common in the benchmark set.
- **`SINGLET_ENGINE_GPU_E2E=0` matches E11's kept `gpuD` on pbmc1k_v3** apart from the whitelist path in `provenance.json` (§2c).
  - 60 identical files, and Solo 10/10.
  - `SJ.out.tab` and the mapping statistics are identical.
  - CellReads.stats has the same 323,542 rows.
- **pbmc1k_cite_v3's feature-barcode library (§4c).**
  - The benchmark `.1fq` is GEX only.
  - The pipeline counts antibody tags only from a CITE `.1fq` with `--feature-ref`, in a CPU pass after the export, the same with
    either engine.
  - Encoded as an ordinary `.1fq`, the antibody library maps as RNA (95 % too short), identically in both engines.
- **`prod/campaign` cannot take `gpu/fable-engine` as it is.** A trial merge conflicts in 9 files, 27 hunks (§6, §9).

## 1. What `--engine gpu` does now

### 1a. Defaults (`include/singlet/engine_gpu.h`, `applyDefaults`)

The pipeline calls `applyDefaults()` once after reading its options, when `--engine gpu` is set. It sets the defaults in the
environment, so that STAR's child, which the pipeline forks later, and every `getenv()` reader see the same values. A variable that is
already set is kept. With `SINGLET_GPU_PILEUP=on`:
- `SINGLET_ENGINE_GPU_E2E` unset → 5. E16's tail cuts are on at level 5 (`SINGLET_E16=0` turns them off).
- At level 5, `SINGLET_ENGINE_GPU_INGEST` unset → 1. Below level 5 the ingest stays opt-in, as before.
- `SINGLET_E16_TRACE` unset → 0: E16's `[e16]` timing marks are off. Before E19 they were on whenever E16 ran.

Without `SINGLET_GPU_PILEUP=on`, nothing is set: levels 1–5 and the ingest need the GPU pileup, and `--engine gpu` maps the distinct
sequences only, as before. The run's first line says which case it is:

```
[singlet] --engine gpu: level 5 (default), ingest (default), E16's tail; each falls back on its own with a line on stderr
```

The opt-outs are the variables' own values:
- `SINGLET_ENGINE_GPU_E2E=0` runs the path before E8 (gate mode `gpuO0`). 1–4 keep their meaning.
- `SINGLET_ENGINE_GPU_INGEST=0` runs level 5 on the FASTQ temp files (`gpuI0`).
- `SINGLET_GPU_PILEUP` keeps its meaning.

`singlet --help` documents them under `--engine` (the text is in `usage()`, `src/pipeline/singlet.cpp`).

### 1b. Every fallback is loud

Every condition that makes the GPU path do less than it was asked prints one line on the pipeline's stderr. GPU-PATH.md §3 lists them
all.
- **STAR's side** (`src/star/SeqCacheGpuPrefill.cpp`):
  - The 13 places that wrote "ENGINE_GPU: ... not used" only to `Log.out` now go through `gpuFallback()`, which writes the line to
    `Log.out` and as `[engine_gpu] fallback: <line>` to stderr. STAR's child shares the pipeline's stderr.
  - The engine's own log is echoed the same way (`gpuFallbackLines`): its lines with "not used", "not built", "not kept", "not
    computed", "out of memory" or "failed". E17's `engine_gpu: budget ...` line is the plan, not a fallback, and is left out (§5.3).
  - A failed GPU mapping now names the engine's reason. Before E19 it said only "GPU mapping failed". Example from hgmm1k_v3 before the
    merge: `ENGINE_GPU: not used (GPU mapping failed: engine_gpu: cudaDeviceSetLimit(cudaLimitStackSize, ...): out of memory)`.
  - One level line per run, on stderr and in `Log.out` (`ENGINE_GPU: E19: level N reached (requested M)`):

    ```
    [engine_gpu] level 5 (requested 5), reads from the .1fq (E13 ingest), Solo on the device (E11)
    ```

    It ends in `: FELL BACK` when the level reached is below the level requested.
- **The pipeline's side** (`src/pipeline/singlet.cpp`):
  - `[1fq-ingest] not used (<why>): the .1fq is decoded to FASTQ files`. It existed already. E19 adds the reasons that come from
    the GPU pileup (§1c).
  - `[singlet] SINGLET_GPU_PILEUP ignored: <why>; ...` existed. E19 adds the case with no usable device.

The gate collects these lines in `fallbacks.txt` per run, and counts them in `runs.tsv` (`fallback_lines`).

### 1c. The ingest declines up front when the GPU pileup will not run

E13 gated the ingest with E8's records only. When the GPU pileup or its hand-off would not run on STAR's stream, the ingest now
declines before the head scan, and the run keeps the FASTQ files. The reasons:
- no usable GPU;
- cDNA reads over 128 bases;
- intron rules outside 1|4|128|256;
- `SINGLET_GPU_PILEUP_FORCE_DECLINE`;
- the sorted-BAM path (`--sorted-bam`, `--bam`, a tagged BAM, `SINGLET_KEEP_STAR_BAM`);
- no GPU hand-off (a chrM BAM, a guide reference, or `SINGLET_GPU_HANDOFF=0`).

Debug gates 797473 (`gpuX1`, with `SINGLET_GPU_PILEUP_FORCE_DECLINE=1`) and 797469 (the g050 node) tested this path. In both, the run
was identical to the CPU engine (59/3/3) and printed the declines.

### 1d. The GPU probe (new; §5.1)

`singlet gpu-probe` is a new subcommand. It prints `ok` when CUDA sees a usable device, and the reason otherwise.

When `SINGLET_GPU_PILEUP` is set, the pipeline runs the probe as an exec'd child as soon as its options are read. The probe is a
child so that the pipeline itself never initialises CUDA before it forks STAR. Its answer is awaited, for at most 120 s, at the two
decisions that need it before STAR runs: the ingest and the GPU pileup. On a working node it answers in 0.8 s, before either decision
asks, so nothing waits.

Without a usable device:
- the pipeline keeps the CPU pileup and prints `[singlet] SINGLET_GPU_PILEUP ignored: no usable GPU (<reason>); the CPU pileup runs`;
- STAR's prefill falls back on its own;
- the run is the CPU engine's path.

Before E19 such a run consumed STAR's stream without the CPU's steps. The GPU pileup then could not get a device and ended the run
with exit 9 (gate 797469, §5.1).

### 1e. The level-4 hole (E11 §5.4) now stops the run

E11 found a hole. Level 4 or 5 tells STAR's loop to skip the per-read summaries. If the run then falls below level 4, Solo and the
QC files are written from incomplete inputs.

STAR now stops with `exitWithError` (exit 104, EXIT_CODE_INPUT_FILES) in the two places where that happens:
- level 4's read source cannot be built;
- level 3 is not reached after the summaries were skipped (`e10NoSumMade && g_e10Level < 4`).

The message is `EXITING because of FATAL ERROR (singlet --engine gpu, E10): ... its outputs would be incomplete (rerun with
SINGLET_ENGINE_GPU_E2E=3)`. No gated run reaches either place. Both are guards.

### 1f. The CPU engine

A run without `--engine gpu` reads none of these variables:
- `applyDefaults` runs only with `SINGLET_ENGINE_GPU=1`, which `--engine gpu` sets;
- the probe starts only with `SINGLET_GPU_PILEUP` set;
- the ingest's chain of reasons starts with the GPU pileup check.

The CPU references of the gate were made by this branch's binary (431f4ef9) without `--engine gpu` (gate 797474).

There is also an indirect check through E11's kept runs, on the files these comparisons cover:
- this branch's `gpuO0` is identical to E11's kept `gpuD` (§2c);
- E11's `gpuD` was identical to E11's CPU engine;
- this branch's `gpuO0` is identical to this branch's CPU engine.

## 2. The all-sample gate

The samples are every directory of `$W/pipeline-benchmark/data/`:
- pbmc2m, pbmc10m, pbmc1k_v3, neuron1k_v3, hgmm1k_v3, nuclei2k_v2 and pbmc5p_v1, from their own `.1fq`;
- pbmc1k_v2 and pbmc1k_cite_v3, which have only FASTQs there. They were encoded by `encode.sbatch` (job 797465) the way pbmc1k_v3's
  `.1fq` was, from their GEX FASTQs, under `$W/jobs/fable/release/data/`;
- pbmc1k_cite_v3's antibody library, encoded on its own as a tenth run (`pbmc1k_cite_v3_adt`, for §4c).

Every run: T = 18, an H100 NVL, `--pipeline --output-format 1pz`.

What the gate compares per pair (`bench/fable/release/jobs/gate.sbatch`):
- `out_equiv.py`, with the read files set aside. Its DIFF files must be E11's run-dependent set: `pileup_stats.json`,
  `read_stats.tsv`, `star_Log.std.out` and `summary.json`.
- The 10 STAR Solo files, by `cmp`.
- The mapping statistics of `star_Log.final.out`, and `star_SJ.out.tab`.
- `nonhost_reads.fq` and `star_Unmapped.out.mate1/2`, by the sha256 of their sorted 4-line records and the read count.
- `Gene/CellReads.stats`, by sorted rows (hard-linked the moment it appears, because the pipeline deletes it).

### 2a. The merged head (gate 798027, binary ced7f553, GPU path only, against the CPU references of gate 797474)

Every run: exit 0 and `level 5 (requested 5), reads from the .1fq (E13 ingest), Solo on the device (E11)`; `fallbacks.txt` is empty.

| sample | reads | chemistry, genome | out_equiv | Solo files | `SJ.out.tab` | CellReads.stats rows | unmapped mates; `nonhost_reads.fq` | verdict |
|---|---|---|---|---|---|---|---|---|
| pbmc2m | 2.0 M | 3' v3, GRCh38 | 59/2/4 | 10/10 | 18,316 | 64,310 | same records (116,369; 14,606) | IDENTICAL |
| pbmc10m | 10.0 M | 3' v3, GRCh38 | 59/2/4 | 10/10 | 47,173 | 173,482 | same (547,775; 70,595) | IDENTICAL |
| pbmc1k_v3 | 66.6 M | 3' v3, GRCh38 | 60/2/4 | 10/10 | 126,996 | 323,542 | same (3,355,559; 428,468) | IDENTICAL, and `gpuT`, `gpuO0` too |
| pbmc1k_v2 | 76.9 M | 3' v2, GRCh38 | 60/2/4 | 10/10 | 92,002 | 172,315 | same (6,048,253; 1,269,171) | IDENTICAL |
| pbmc1k_cite_v3 (GEX) | 45.4 M | 3' v3, GRCh38 | 60/2/4 | 10/10 | 109,722 | 224,209 | same (2,394,144; 513,976) | IDENTICAL |
| pbmc1k_cite_v3_adt | 12.6 M | antibody tags as 3' v3 RNA, GRCh38 | 57/2/4 | 10/10 | 1,294 | 60,082 | same (12,572,515; 397,478) | IDENTICAL |
| pbmc5p_v1 | 60.0 M | 5' v1, GRCh38 | 58/2/4 | 10/10 | 212,575 | 355,639 | same (4,818,388; 1,335,105) | IDENTICAL |
| neuron1k_v3 | 92.9 M | 3' v3, GRCm39 | 60/2/4 | 10/10 | 117,254 | 600,534 | same (3,667,707; 490,605) | IDENTICAL |
| hgmm1k_v3 | 63.1 M | 3' v3, GRCh38 + GRCm39 barnyard | 49/2/4 | 10/10 | 171,869 | 524,497 | same (1,812,610; 309,553) | IDENTICAL |
| nuclei2k_v2 | 118.7 M | 3' v2 nuclei, GRCm39 | 60/2/4 | 10/10 | 100,771 | 411,195 | mates same (22,811,732); `nonhost_reads.fq` other records, same count (2,459,477) | IDENTICAL except the known bug (§4d) |

- out_equiv is identical / differ only in timings or paths / differ. The 4 that differ are E11's run-dependent set in every row:
  - `pileup_stats.json`: `bytes_public`;
  - `read_stats.tsv` and `star_Log.std.out`: timings;
  - `summary.json`: `memory_tier` Medium against Large.
- The file counts differ between samples because the samples' outputs differ.
- The mapping statistics of `star_Log.final.out` are identical in every row.
- `Gene/CellReads.stats` is never byte-identical in file order. Its rows are the same when sorted.

### 2b. Before the merge (gate 797474, binary 431f4ef9, both engines)

The gate ran the CPU engine and the defaults on the same node. Its CPU runs are the references for §2a and §3.

| sample | GPU path: level, ingest | fallback lines | out_equiv | printed verdict |
|---|---|---|---|---|
| pbmc2m, pbmc10m | 5, yes | 0 | 59/2/4 | IDENTICAL |
| pbmc1k_v3 (`gpu`, `gpuT`, `gpuO0`) | 5, yes (`gpuO0`: 0, no) | 0 | 60/2/4 | IDENTICAL |
| pbmc1k_v2, pbmc1k_cite_v3, neuron1k_v3 | 5, yes | 0 | 60/2/4 | IDENTICAL |
| pbmc1k_cite_v3_adt | 5, yes | 0 | 57/2/4 | IDENTICAL |
| pbmc5p_v1 | 5, yes | 0 | 58/2/4 | IDENTICAL |
| hgmm1k_v3 | **0**: STAR maps every read, through the chunk source over the `.1fq` | 1: `[engine_gpu] fallback: ENGINE_GPU: not used (GPU mapping failed: engine_gpu: cudaDeviceSetLimit(cudaLimitStackSize, ...): out of memory): STAR maps every read` | 49/2/4 | IDENTICAL |
| nuclei2k_v2 | 5, yes | 0 | 60/2/4 | NOT IDENTICAL: `nonhost_reads.fq` alone (§4d). The rule naming it came after this gate |
| pbmc1k_v3 `e11gpuD:gpuO0` | | | 60/2/4 | DIFFERENT: `provenance.json`'s whitelist path alone (§2c). The rule came after this gate |

### 2c. `SINGLET_ENGINE_GPU_E2E=0` against E11's kept `gpuD` (pbmc1k_v3)

This compares E19's opt-out (`gpuO0`) with `gpuD` from E11's gate 796377 (`$W/jobs/fable/solowire/results/gate-796377/gpuD/out`, binary 8d9eaa85).
- The two are the same except for one string: the path of the whitelist in `provenance.json`. The whitelist is found next to the
  binary, and the two binaries are in different directories.
- out_equiv finds 60 identical files.
- The 10 Solo files are identical, `SJ.out.tab` is identical (126,996 junctions) and so are the mapping statistics.
- `CellReads.stats` has the same 323,542 rows.

The gate treats a `provenance.json` difference that is only `/references/whitelist/name` as a path (7477b615). Before that change,
gate 797474 printed this pair as DIFFERENT.

On the merged head (798027) the pair is IDENTICAL under the gate's rules (60/2/4). `gpuO0` takes 195.1 s / 965 core-s, against
level 5's 101.5 s / 548 core-s and the CPU engine's 403.9 s / 2,507 core-s. Its `pileup_stats.json` also records the 5' probe's
12,000 reads (the FASTQ-file path) instead of 12,194, one of E11's run-dependent fields.

### 2d. The other switches (debug gates, pbmc2m, T = 8)

| gate | binary | mode | exit | level, ingest | fallback lines | vs CPU |
|---|---|---|---|---|---|---|
| 797473 | 431f4ef9 | `gpu` | 0 | 5, yes | 0 | identical |
| | | `gpuO0` (E2E=0) | 0 | 0, no | 0 | identical |
| | | `gpuI0` (INGEST=0) | 0 | 5, no | 0 | identical |
| | | `gpuX1` (FORCE_DECLINE=1) | 0 | 0 of 5, declined | 5 | identical (59/3/3) |
| 797479 | a8c25a48 (merged) | `gpu` | 0 | 5, yes | 0 | identical (out_equiv's only extra DIFF is the whitelist path, §5.4) |
| | | `gpuO0` | 0 | 0, no | 1 (E17's budget line, §5.3) | identical (the same) |
| | | `gpuI0` | 0 | 5, no | 0 | identical (the same) |
| | | `gpuX1` (`MEM_GB=12`) | **3** | – | 2 | stops: `EXITING because of FATAL ERROR (singlet --engine gpu, E17): the sample does not fit the device memory budget: ...` |
| | | `gpuX2` (`MEM_GB=12`, `MEM_FALLBACK=cpu`) | 0 | 0 (STAR maps every read) | 2 | identical (59/2/4) |

## 3. Timings (T = 18, H100 NVL; wall s / core-s, peak RSS)

CPU engine: gate 797474, node g051. GPU path: gate 798027 (merged head), node g052, the same node type. "Device" is the 1 s
`nvidia-smi` peak; "tally" is the engine's own peak from E17's budget line.

| sample | reads | CPU engine: wall / core-s / RSS GB | GPU path: wall / core-s / RSS GB | device / tally GB | wall | core-s | GPU before the merge (797474) |
|---|---|---|---|---|---|---|---|
| pbmc2m | 2.0 M | 49.6 / 161 / 25.8 | 32.4 / 121 / 21.8 | 37.6 / 42.7 | 1.53x | 1.33x | 34.9 / 142 |
| pbmc10m | 10.0 M | 100.5 / 482 / 26.7 | 45.2 / 193 / 24.9 | 89.3 / 86.5 | 2.23x | 2.50x | 45.5 / 193 |
| pbmc1k_v3 | 66.6 M | 403.9 / 2,507 / 29.7 | **101.5 / 548** / 46.1 | 89.3 / 88.4 | 3.98x | 4.58x | 103.8 / 557 |
| (pbmc1k_v3 `gpuT`) | | | 101.4 / 548 / 46.1 | 89.3 / 88.4 | | | 105.3 / 564 |
| (pbmc1k_v3 `gpuO0`, E2E=0) | | | 195.1 / 965 / 43.5 | 89.8 / 89.7 | 2.07x | 2.60x | 198.1 / 971 |
| pbmc1k_v2 | 76.9 M | 380.8 / 2,798 / 29.4 | 104.4 / 536 / 47.5 | 89.3 / 88.4 | 3.65x | 5.22x | 107.4 / 540 |
| pbmc1k_cite_v3 | 45.4 M | 278.0 / 1,713 / 28.2 | 78.4 / 411 / 38.2 | 89.5 / 89.3 | 3.55x | 4.17x | 80.1 / 407 |
| pbmc1k_cite_v3_adt | 12.6 M | 129.9 / 1,255 / 32.9 | 65.0 / 274 / 29.6 | 89.3 / 89.2 | 2.00x | 4.58x | 66.7 / 281 |
| pbmc5p_v1 | 60.0 M | 495.4 / 2,123 / 32.3 | 123.6 / 509 / 39.6 | 89.3 / 88.2 | 4.01x | 4.17x | 124.1 / 514 |
| neuron1k_v3 | 92.9 M | 491.5 / 2,148 / 35.3 | 131.5 / 615 / 60.8 | 89.3 / 88.0 | 3.74x | 3.49x | 127.1 / 587 |
| hgmm1k_v3 | 63.1 M | 363.1 / 1,597 / 62.2 | **90.5 / 443** / 78.4 | 89.3 / 89.1 | 4.01x | 3.61x | 235.7 / 1,678 (level 0) |
| nuclei2k_v2 | 118.7 M | 609.9 / 4,604 / 34.9 | 198.7 / 954 / 84.4 | 89.3 / 88.5 | 3.07x | 4.83x | 205.8 / 989 |
| **10 runs** | 548 M | **3,303 / 19,388** | **971 / 4,603** | | **3.40x** | **4.21x** | |

- **Where the time goes:** GPU-PATH.md §1 has pbmc1k_v3's stages.
  - The non-host screen, the export and the writers are about 57 s of the 101 s, and they are the same work in both engines.
  - neuron1k_v3's GPU pileup holds the device for 28 s; on the other samples it takes 0.4–5.3 s (§5.6).
- **E17's merge changes nothing measurable except hgmm1k_v3.** The other samples are within 3.5 % of the pre-merge runs (−7.1 s on
  nuclei2k_v2 to +4.3 s on neuron1k_v3; different nodes, g051 against g052). hgmm1k_v3 drops from 235.7 s at level 0 to 90.5 s at level 5. E17's own gate measured 88.7 s.

## 4. Declines

### 4a. Per sample

No sample declines anything on the merged head. In every run:
- `fallbacks.txt` is empty;
- the ingest line reads `STAR's prefill decodes the reads (no FASTQ files)`;
- the level line reads `level 5 (requested 5), reads from the .1fq (E13 ingest), Solo on the device (E11)`.

The properties that the conditions look at:

| sample | `.1fq` protocol (confidence) | longest cDNA | 5' probe sample → clip | E17 plan: chunks of 16 M, index | ingest | level | fallback lines |
|---|---|---|---|---|---|---|---|
| pbmc2m | 1 (3) | 91 | 12,052 → 0 | 1 chunk, 16.4 GB, normal | yes | 5 | 0 |
| pbmc10m | 1 (3) | 91 | 12,020 → 0 | 1, 16.4 GB, normal | yes | 5 | 0 |
| pbmc1k_v3 | 1 (3) | 91 | 12,194 → 0 | 2, 16.4 GB, normal | yes | 5 | 0 |
| pbmc1k_v2 | 2 (3) | 91 | 12,166 → 0 | 2, 16.4 GB, normal | yes | 5 | 0 |
| pbmc1k_cite_v3 | 1 (3) | 91 | 12,103 → 0 | 2, 16.4 GB, normal | yes | 5 | 0 |
| pbmc1k_cite_v3_adt | 1 (2) | 91 | 12,039 → 0 | 1, 16.4 GB, normal | yes | 5 | 0 |
| pbmc5p_v1 | 5 (5) | 91 | 12,000 → 0 | 2, 16.4 GB, normal | yes | 5 | 0 |
| neuron1k_v3 | 1 (3) | 91 | 12,276 → 0 | 3, 25.1 GB, normal | yes | 5 | 0 |
| hgmm1k_v3 | 1 (3) | 91 | 12,192 → 0 | 2, 52.7 GB, normal | yes | 5 | 0 |
| nuclei2k_v2 | 2 (3) | 98 | 12,376 → 0 | 5, 25.1 GB, normal | yes | 5 | 0 |

Every `.1fq` has a barcode dictionary, 2-bit sequences, quality codes the decode takes, no duplicate counts, and a fixed barcode read
of at most 63 bases; otherwise the ingest would have declined. Before the merge the only decline was hgmm1k_v3's GPU mapping (§2b).

### 4b. The conditions, and what a port would take

None of the conditions below occurs in the benchmark set, so no decline here is both common and cheap. Each one falls back with a
line (GPU-PATH.md §3), and the outputs stay the CPU engine's.

| condition (line) | where | the port |
|---|---|---|
| per-base quality stored in full (`[1fq-ingest] not used (per-base quality stored in full)`) | ingest | The decode keeps quality as 2-bit codes, and STAR's quality-dependent steps (E11's RNA Q30, the CB posterior) read them through the code table. The port is a byte plane for full qualities: 4x the quality plane (6.0 GB instead of 1.5 GB on pbmc1k_v3's 6.0 G cDNA bases), read through the same accessors. Medium. |
| duplicate counts in the .1fq (`duplicate counts`) | ingest | A deduplicated .1fq stores a read once with a count. The decode would expand the counts into reads (simple, and the memory of the expanded reads), or the engine would carry a weight per read through E10 and E11 (large). |
| no barcode dictionary, byte-coded sequences, variable barcode read length, a staged `.1fg2` (`head_scan_supported`) | ingest | The head scan reads only the dictionary column. Without one it would decode the whole barcode column, which is slower but has no new logic. `.1fg2` has its own mapping (E12/E13 §1e) and nothing implemented. |
| barcode read over 63 bases | ingest | The barcode plane holds 3-bit codes, 21 bases per 64-bit word, and the decode takes at most 3 words per read. A fourth word is a small change, but no 10x chemistry needs it. |
| CB or UMI over 21 bases | ingest, E10 | The keys are 3-bit codes in one 64-bit word. 128-bit keys go through E10's sort and E11's tables. Medium. |
| a constant 5' prefix to clip | ingest | The prefill gives STAR the reads unclipped. The decode could drop the prefix (offset + n), and the CR4 clip and the records would follow. Small, but the 5' sample here (pbmc5p_v1) has none. |
| barcode in R2, CB_UMI_Complex, no whitelist, not a droplet scRNA assay | ingest, E11 | E9's device Solo takes CB_UMI_Simple with a whitelist and the barcode in R1. A barcode in R2 is a swap of the two streams in E13's decode, probably small. Complex barcodes and barcode discovery are each a port of STAR's Solo code to the device: large. ATAC, plate and bulk assays are other pipelines. |
| cDNA reads over 128 bases (`SINGLET_GPU_PILEUP ignored: reads over 128 bases ...`) | GPU pileup | The pileup's per-base-quality mask is 128 bits. Two words per read in `pileup.cu`'s mask, outside E19's files. Small to medium. |
| experimental intron rules (`... does not implement intron rules 0x...`) | GPU pileup | The GPU pileup mirrors rules 1, 4, 128 and 256. `prod/campaign` has since mirrored kFlank and the USA labels (03a5fac5, §9). The merge brings them, and `kGpuRules` must follow. |
| the sorted-BAM path, a chrM BAM, a guide reference, `SINGLET_GPU_HANDOFF=0` | GPU pileup hand-off | E8 does not write BAM records for the reads it keeps on the device. A requested BAM would need them formatted from the records (the hand-off file path does that today, at level 0). Medium. |
| the sample does not fit the device memory budget (E17) | prefill | It stops with the FATAL line, or STAR maps every read with `SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu`. With one slot no benchmark sample comes near it: hgmm1k_v3's normal plan (70.3 GB, with its 52.7 GB index) is the largest. |

### 4c. pbmc1k_cite_v3: what the pipeline does with the feature-barcode library

- **`pbmc1k_cite_v3`'s benchmark directory has two libraries:** GEX (`gex_fastqs.txt`, 4 files) and the antibody tags
  (`adt_fastqs.txt`, 6 files). pbmc5p_v1's directory also lists an antibody library.
- **The `.1fq` the gate runs is GEX only**, encoded as the release gate's `counts_singlet_gpu.sbatch` does. Encode detects it as
  `10x-3p-v3` (protocol 1). The pipeline counts it as RNA, and both engines give the same outputs: level 5 + ingest, 729 cells.
- **The pipeline's own feature-barcode path needs a CITE `.1fq` and a tag reference.**
  - The `.1fq` header must have the `CITE_SEQ_GEX` or `CITE_SEQ_ADT` assay, and the run needs `--feature-ref` (a name,sequence CSV).
  - The GEX part runs as usual. After the GEX export, a second pass over the `.1fq` matches R2 to the tags on the CPU (`AdtMatcher`,
    `AdtCounter`, `singlet.cpp` "T4: CITE-seq ADT/HTO processing"). It is the same pass with either engine.
  - A `CITE_SEQ_ADT` `.1fq` alone without `--exons` writes `summary.json` with status `feature_barcode_not_gex` and exits 0 without
    STAR.
- **The antibody library encoded alone is not classified as ADT.**
  - Encode's protocol detection takes `10x-3p-v3` first. The ADT rule only applies to undetected protocols with R2 ≤ 60 bases, and
    these R2s are 91 bases.
  - So the pipeline maps the tags as RNA: 95.42 % "too short", 0.23 % unique, and 315 "cells" with a median of 3 genes. The result is
    meaningless as biology, but it is identical in both engines, and the GPU path takes it at level 5 + ingest.
- **The benchmark has no tag reference CSV**, so the CITE path was not run.

### 4d. nuclei2k_v2: `nonhost_reads.fq`

nuclei2k_v2 has more than 20 M unmapped reads (22,811,732). Above 20 M the non-host screen subsamples by read-name hash, and the writer
then takes the classification by the record's index in the whole `star_Unmapped.out.mate1`. That index is wrong after subsampling
(E15 §2c, §5.7; `src/pipeline/singlet.cpp` near "Wrote classified reads").

So `nonhost_reads.fq` holds other reads in each run:
- 2,459,477 records in every run;
- the sorted sha256 is de1ed828… for the CPU engine (797474), ea1b184f… for the GPU path in 797474 and e34f8159… for the GPU path in
  798027. The written reads follow the unmapped file's order, which is run-dependent.

The unmapped mates themselves have the same records, and the non-host tables are identical.

This is a pipeline bug, not the GPU path. The gate names the file as known for this sample (`KNOWN_nuclei2k_v2`, e88fe91e), and only
when the read count matches. Gate 797474 printed nuclei2k_v2 as NOT IDENTICAL for this file alone, before that commit.

## 5. Surprises

1. **A node whose GPU is not usable ended GPU runs with exit 9 (gate 797469, g050).**
   - The job held the H100 NVL GRES, but `nvidia-smi` said "No devices were found". `/dev/nvidia-caps` was empty, and the device
     nodes dated Oct 31 16:00 (job 797470 looked).
   - STAR's prefill fell back as it should.
   - The GPU pileup's decision came before any device was opened. The CPU skipped its steps, and the pileup then failed: exit 9 on
     `gpu`, `gpuO0` and `gpuI0`, and only 15 files identical to the CPU engine's.
   - Fixed by the GPU probe (§1d, 431f4ef9). E19's later jobs exclude g050. The node is not ours to fix, and this report notes it.
2. **Small samples gain little.**
   - pbmc2m is 1.53x faster in wall time and 1.33x in core-s. pbmc10m is 2.2x and 2.5x.
   - The fixed costs dominate a 2 M-read run: the reference and index loads, the non-host screen's index, and the writers.
   - From 45 M reads up, the gain is 3.1–4.0x in wall time and 3.5–5.2x in core-s.
3. **E17's budget line was counted as a fallback.** At level 0 the engine log says "results not kept on the device" in its budget
   line. The echo of the engine's lines (`gpuFallbackLines`) printed it as `[engine_gpu] fallback:` (gate 797479, `gpuO0`).
   ced7f553 leaves out lines starting with `engine_gpu: budget`.
4. **`provenance.json` records the whitelist path that the binary resolved (`<bin>/../whitelists`).** Two binaries in different
   directories therefore differ in this one field. The gate now calls that a path (7477b615). The final gate runs the merged binary
   from the same `bin/` as the references.
5. **Every run fills the device budget, whatever the sample's size.**
   - E17's plan gives the lanes all the room that the index and the reads leave. So the engine's tally peaks at 86.5–89.3 GB on every
     sample but pbmc2m (42.7 GB), although its normal-mode estimates run from 18.8 GB (pbmc2m) to 70.3 GB (hgmm1k_v3).
   - That is fine with one sample per device. A second sample on the same device needs `SINGLET_ENGINE_GPU_SLOTS=2`, not a smaller
     sample (GPU-PATH.md §4, §5).
6. **neuron1k_v3's GPU pileup holds the device for 28 s**: `pileupRun` 28 s, against 0.4–5.3 s on the others, both before and after
   the merge.
   - Its tuple counts point to the mitochondrial pass: 141 M mt tuples against 69 M gene tuples.
   - The run is still 3.7x faster than the CPU engine, but those 28 s are a fifth of its wall time, with the device lock held
     throughout.
   - This is the GPU pileup (`pileup.cu`), outside E19's files.
7. **Of the gated samples, only the barnyard needed E17.** Before the merge hgmm1k_v3 fell back loudly at the engine's
   `cudaDeviceSetLimit` (out of memory), and STAR mapped every read: 235.7 s against 363.1 s for the CPU engine, identical. On the
   merged head it runs at level 5 (§2a).

## 6. What remains

- **Merging into `prod/campaign`** (§9).
  - `prod/campaign` has 291 commits that `gpu/fable-engine` lacks, from the merge base cf92cdb9. Several of them touch the GPU path:
    the device dedup pass, pinned staging, the GPU pileup's USA labels and kFlank, order-free multimapper gene tracking, and `.1fg2`
    table prefill.
  - A trial merge (`git merge-tree`) conflicts in 9 files, 27 hunks: `engine_gpu.cu` 10, `SeqCacheGpuPrefill.cpp` 5,
    `engine_gpu.h` 3, `pileup.cu` 2, `pileup_gpu.cu` 2, `singlet.cpp` 2, and `CMakeLists.txt`, `cb_correct.h`, `pileup_engine.h` 1
    each.
  - Merge `prod/campaign` into `gpu/fable-engine` (or a staging branch), then re-run this gate (`MODES="cpu gpu"`, `REFRESH_REF=1`,
    since the campaign changes the CPU engine's outputs too).
- **The non-host writer bug** (§4d, E15 §5.7) is still open. It is a pipeline fix: take the classification by the kept reads' names,
  not by the record's index. The fix changes the CPU engine's `nonhost_reads.fq` on samples with more than 20 M unmapped reads, so it
  needs its own gate.
- **The declines of §4b.** None is common in the benchmark set. The first ones worth porting are probably full qualities and
  duplicate counts, because other `.1fq` writers may produce them.
- **neuron1k_v3's GPU pileup** (§5.6): 28 s on the device, mostly the mitochondrial tuples. It is the largest single GPU stage left on
  a benchmark sample.
- **E18** (samples per device, the thread split) is still measuring. GPU-PATH.md §5 will take its numbers.
- **g050** (§5.1): its GPU is not visible to jobs. That is for the cluster's admins.

## 7. Files touched, and why

| file | change | why |
|---|---|---|
| `include/singlet/engine_gpu.h` | `kDefaultE2ELevel`, `applyDefaults()`; a comment on level 0 | the defaults (§1a) |
| `src/pipeline/singlet.cpp` | `--help` text under `--engine`; `e19probe` (start, usable) and the `gpu-probe` subcommand; the defaults line; the ingest's new reasons (§1c); the GPU pileup's no-device case | defaults, loud fallbacks, the probe |
| `src/star/SeqCacheGpuPrefill.cpp` | `gpuFallback()`, `gpuFallbackLines()` and the 13 call sites; the engine's reason on a failed mapping; the level line; the two `exitWithError` guards; E17's budget block merged ahead of E19's skip | loud fallbacks (§1b), the level-4 hole (§1e) |
| `bench/fable/release/jobs/build.sbatch` | the build (from E16's), branch `gpu/fable/release`, output `$W/jobs/fable/release/bin` | deliverable |
| `bench/fable/release/jobs/encode.sbatch` | the `.1fq` of pbmc1k_v2, pbmc1k_cite_v3 (GEX) and its antibody library | the samples with FASTQs only |
| `bench/fable/release/jobs/gate.sbatch` | the all-sample gate: CPU references kept per sample and T, the GPU modes, the fallback, level and ingest columns, the probe line, the device peaks (`nvidia-smi` and the engine's tally), the known-bug list | deliverable |
| `docs/gpu-engine/fable/GPU-PATH.md` (new) | the operator's page | deliverable |
| `docs/gpu-engine/fable/E19-release.md` (new) | this report | deliverable |

`engine_gpu.cu`, `hostloop_gpu.cu` and the CPU engine are untouched. `engine_gpu_ingest.h` and `SeqCacheGpuIngest.cpp` needed no
change.

## 8. Jobs, commits, how to run

**Branch `gpu/fable/release`** (pushed):

| commit | what |
|---|---|
| 266f7125 | the defaults, the loud fallbacks, the level-4 guards, `--help`, the build, encode and gate jobs |
| 9769d029 | `--help` names `SINGLET_ENGINE_GPU_SLOTS` and the device default; the gate collects the pipeline's fallback lines |
| 431f4ef9 | the GPU probe (§1d) |
| 9b0909d9, e7855212 | gate: the probe line; the prefill's device peak from the engine log |
| a8c25a48 | merge of `origin/gpu/fable-engine` at 503c07f7 (E17). One conflict in `SeqCacheGpuPrefill.cpp`: E17's budget block now comes ahead of E19's skip |
| 8e7e2ed0 | gate: the device peak from E17's budget line too |
| ced7f553 | E17's budget line is not a fallback (§5.3); the binary of the final gate |
| 7477b615 | gate: the whitelist path in `provenance.json` is a path (§5.4) |
| e88fe91e | gate: nuclei2k_v2's `nonhost_reads.fq` is the known bug (§4d) |
| this document's commit | GPU-PATH.md, this report; two comments now point to GPU-PATH.md §3 (comment only) |

**Jobs** (all `--nice=10000`; logs `$W/logs/e19-*-<job>.out`, results `$W/jobs/fable/release/results/gate-<job>/`):

| job | partition, node | what | result |
|---|---|---|---|
| 797464 | short-gpu (no GPU), g005 | build 266f7125 → `bin/` | ok |
| 797465 | short-gpu (no GPU) | encode, array 0–2 | pbmc1k_v2 76.9 M, pbmc1k_cite_v3 45.4 M, its antibody library 12.6 M reads |
| 797469 | debug-gpu, g050 | debug gate pbmc2m, T = 8 | GPU not visible: exit 9 on 3 modes (§5.1) |
| 797470 | debug-gpu, g050 | look at the node's devices | `/dev/nvidia-caps` empty |
| 797471 | short-gpu (no GPU) | build 431f4ef9 → `bin/` | ok |
| 797473 | debug-gpu, g051 | debug gate pbmc2m, T = 8 | all identical (§2d) |
| 797474 | short-gpu, g051 | all-sample gate, both engines, T = 18, 1 h 25 min | §2b; the CPU references |
| 797475 | short-gpu (no GPU) | build a8c25a48 (merge) → `bin-merged/` | ok |
| 797479 | debug-gpu, g052 | debug gate of the merge, pbmc2m, T = 8 | §2d |
| 797481 | short-gpu (no GPU) | build ced7f553 → `bin-merged/`, then copied to `bin/` (`bin-premerge/` keeps 431f4ef9) | ok |
| 798027 | short-gpu, g052 | all-sample gate of the merged head, GPU modes, T = 18 | §2a |

**How to run:** GPU-PATH.md §7. In short:
- build with `sbatch --export=ALL,BRANCH=gpu/fable/release bench/fable/release/jobs/build.sbatch`;
- gate with `sbatch --exclude=g050 bench/fable/release/jobs/gate.sbatch`. That runs the CPU engine and the defaults on every sample,
  and keeps the CPU references.
- For a new binary, add `MODES=gpu` to reuse the references.

## 9. Draft PR description (`gpu/fable-engine` → `prod/campaign`; not opened)

**Title:** GPU aligner: the full GPU path behind `--engine gpu`, byte-identical to the CPU engine (E1–E19)

```markdown
## Summary

`singlet ... --engine gpu` with `SINGLET_GPU_PILEUP=on` now runs the whole GPU path by default and writes the same outputs as the CPU
engine on every benchmark sample. The CPU engine (no `--engine gpu`) is unchanged.

- **Mapping on the device:** E1–E7 and E12 cover the engine, its lanes, warp-per-read stitching, the persistent seed kernel and the
  prefill. E14 adds the genome preload and the concurrent CR4 probes.
- **Level 5 (`SINGLET_ENGINE_GPU_E2E=5`, now the default):**
  - E8: records on the device;
  - E10: per-read products, `Log.final.out` / `SJ.out.tab` / `Unmapped` from the records;
  - E11: STARsolo on the device (E9's library), with STAR's loop mapping only the reads without a GPU record (0.24 %);
  - E15: the CB posterior and the kept Solo inputs;
  - E16: the tail.
- **E13 ingest (now the default at level 5):** the `.1fq` is decoded straight into the prefill's buffers, with no FASTQ temp files.
- **E17 device memory budget** (`SINGLET_ENGINE_GPU_MEM_GB`, default (device − 2 GB) / slots):
  - lean mode;
  - a FATAL stop, or a CPU fallback with `SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu`;
  - 64-bit junction keys, so the human + mouse barnyard runs at level 5.
- **E19:**
  - the defaults;
  - one stderr line per fallback, and the level reached per run;
  - a GPU probe before STAR, so a node without a usable GPU keeps the CPU pileup instead of exiting 9;
  - the level-4 hole stops the run (STAR exits with 104) instead of writing incomplete outputs;
  - `--help`;
  - `docs/gpu-engine/fable/GPU-PATH.md`.

Opt-outs: `SINGLET_ENGINE_GPU_E2E=0` (GPU mapping only) and `SINGLET_ENGINE_GPU_INGEST=0` (FASTQ temp files). `SINGLET_GPU_PILEUP`
unset gives GPU mapping only, as before.

## Results (H100 NVL, T = 18; CPU gate 797474, GPU gate 798027 on gpu/fable/release ced7f553)

| sample | reads | CPU wall s / core-s | GPU wall s / core-s | wall | core-s |
|---|---|---|---|---|---|
| pbmc2m | 2.0 M | 49.6 / 161 | 32.4 / 121 | 1.5x | 1.3x |
| pbmc10m | 10.0 M | 100.5 / 482 | 45.2 / 193 | 2.2x | 2.5x |
| pbmc1k_v3 | 66.6 M | 403.9 / 2,507 | 101.5 / 548 | 4.0x | 4.6x |
| pbmc1k_v2 | 76.9 M | 380.8 / 2,798 | 104.4 / 536 | 3.6x | 5.2x |
| pbmc1k_cite_v3 (GEX) | 45.4 M | 278.0 / 1,713 | 78.4 / 411 | 3.5x | 4.2x |
| pbmc1k_cite_v3 (antibody library as RNA) | 12.6 M | 129.9 / 1,255 | 65.0 / 274 | 2.0x | 4.6x |
| pbmc5p_v1 | 60.0 M | 495.4 / 2,123 | 123.6 / 509 | 4.0x | 4.2x |
| neuron1k_v3 (mouse) | 92.9 M | 491.5 / 2,148 | 131.5 / 615 | 3.7x | 3.5x |
| hgmm1k_v3 (barnyard) | 63.1 M | 363.1 / 1,597 | 90.5 / 443 | 4.0x | 3.6x |
| nuclei2k_v2 (mouse nuclei) | 118.7 M | 609.9 / 4,604 | 198.7 / 954 | 3.1x | 4.8x |

Every sample runs at level 5 with the ingest and prints no fallback line.

The outputs are byte-identical to the CPU engine's, except E11's run-dependent fields:
- timings in `read_stats.tsv` and `star_Log.std.out`;
- `pileup_stats.json`'s `bytes_public`;
- `summary.json`'s `memory_tier`;
- the record order of `Unmapped.out.mate1/2`, `nonhost_reads.fq` and `Gene/CellReads.stats` (same records and rows).

That covers every count matrix, `.1pz` member, the 10 STAR Solo files, `SJ.out.tab` and the mapping statistics. One known exception
is a pipeline bug in both engines (E15 §5.7): nuclei2k_v2's `nonhost_reads.fq` holds other reads (same count) when the non-host screen
subsamples more than 20 M unmapped reads.

The GPU path's host memory is higher on large samples: up to 84.4 GB RSS on nuclei2k_v2, against 34.9 GB. Device memory follows the
E17 budget, 91.1 GB on an H100 NVL with one sample per GPU.

## Before merging

- [ ] Merge `gpu/fable/release` (E19) into `gpu/fable-engine`. This PR assumes it.
- [ ] `prod/campaign` has 291 commits since the merge base cf92cdb9, several on the GPU path:
  - 3a064a19, the device dedup pass;
  - 653a75b2, pinned staging;
  - 03a5fac5, the GPU pileup's USA labels and kFlank;
  - 90b31e77, order-free multimapper gene tracking;
  - f5909477 / 33c637e3, the `.1fg2` table prefill.
- [ ] A trial merge conflicts in 9 files, 27 hunks:
  - `src/engine_gpu/engine_gpu.cu` 10;
  - `src/star/SeqCacheGpuPrefill.cpp` 5;
  - `include/singlet/engine_gpu.h` 3;
  - `src/engine_gpu/gpu_pileup/pileup/pileup.cu` 2;
  - `src/engine_gpu/pileup_gpu.cu` 2;
  - `src/pipeline/singlet.cpp` 2;
  - `CMakeLists.txt`, `include/singlet/pileup/cb_correct.h` and `include/singlet/pileup/pileup_engine.h` 1 each.

  Resolve them on a staging branch: `prod/campaign` merged into `gpu/fable-engine`.
- [ ] `kGpuRules` in `singlet.cpp` (the intron rules the GPU pileup mirrors) must follow the campaign's GPU pileup (kFlank).
- [ ] Re-run the all-sample gate on the merged result. The campaign changes the CPU engine's outputs too, so refresh the references:
  `sbatch --exclude=g050 --export=ALL,REFRESH_REF=1,BRANCH=<branch> bench/fable/release/jobs/gate.sbatch` (after `build.sbatch`).

## Test plan

- [x] All-sample gate on gpu/fable/release (E19 + E17), gate 798027: 10/10 identical to the CPU engine (with the known non-host bug
      named), every sample at level 5 + ingest, no fallback line.
- [x] Opt-outs and fallbacks on pbmc2m (debug gates 797473, 797479):
  - E2E=0, INGEST=0 and FORCE_DECLINE are identical;
  - a 12 GB budget stops with the FATAL line;
  - `MEM_FALLBACK=cpu` is identical.
- [x] `SINGLET_ENGINE_GPU_E2E=0` is identical to E11's kept gpuD (pbmc1k_v3).
- [ ] The gate on the campaign merge.
- [ ] E18: samples per GPU and the thread split.

Known: node g050's GPU is not visible to jobs ("No devices were found"). The pipeline now falls back to the CPU pileup there.

Docs: `docs/gpu-engine/fable/GPU-PATH.md` (the operator's page), E1–E19 reports in `docs/gpu-engine/fable/`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```
