# How `singlet --engine gpu` runs a sample

The GPU path as of E19 (`gpu/fable/release`, binary ced7f553; numbers from gate 798027, H100 NVL, T = 18). Design and measurements
are in the experiment reports E8–E19 in this directory; this page is the operator's view.

```
SINGLET_GPU_PILEUP=on singlet S.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
    --out-prefix out/ --threads 18 --pipeline --output-format 1pz --engine gpu
```

Needs a CUDA build (`-DSINGLET_ENGINE_GPU=ON`) and a device of compute capability 8.0 or more. Every run prints three lines that say
what it did: `[singlet] --engine gpu: level 5 (default), ingest (default), E16's tail; ...`, `[singlet] GPU probe: a usable device`,
and `[engine_gpu] level 5 (requested 5), reads from the .1fq (E13 ingest), Solo on the device (E11)`. Anything less than that is
announced by a fallback line (§3).

## 1. Stages (pbmc1k_v3, 66.6 M reads)

| stage (begins at the log mark) | what runs on the GPU path | GPU path s / core-s (device busy) | CPU engine s / core-s |
|---|---|---|---|
| head scan (`[1fq-ingest] Scanning`; CPU engine: the decode to FASTQ files) | E13's scan of the `.1fq`'s dictionary, UMI and length columns; R11's CB counts; the 5' probe sample | 3 / 29 | 6 / 58 |
| probe and reference load (`.1fq scanned`) | the GPU probe (0.8 s); the reference load; E14's genome preload; the CR4 probes' STAR runs; E13's decode of 66.6 M reads into the packed buffers (3.3 s) | 6 / 59 | 5 / 8 |
| STAR's start (`STAR pid=`) | STAR's parameters; E13's barcode pass; the CR4 clip once per distinct sequence; the prefill's uploads | 10 / 132 | 16 / 139 |
| STAR's stream (BAM header) | the prefill on the device, which holds it 16.2 s for seeds, stitching, records and E10/E11's products; STAR's loop over the 156,931 reads without a record (0.8 s); the BAM stream | 19 / 117 (65 %) | 309 / 2,122: STAR's loop over every read and the CPU pileup |
| hand-off and GPU pileup (`hand-off pass`) | E8's hand-off pass (2.0 s); the GPU pileup (4.1 s); Solo on the device (E11, 0.5 s); E16's tail beside them | 6 / 33 (65 %) | |
| non-host (`STAR completed`) | the non-host screen on the GPU | 18 / 45 | 27 / 51 |
| export, QC, callers | | 18 / 38 | 19 / 39 |
| demux, ancestry, writers | | 21 / 93 | 21 / 90 |
| **total** | | **101.5 / 548** | **403.9 / 2,507** |

The GPU path's numbers come from gate 798027 and the CPU engine's from gate 797474.

E16's marks come from the same run with `SINGLET_E16_TRACE=1` (`gpuT`, 101.4 s), counted from the moment the prefill releases the device:
- the E8 region is complete at +1.0 s;
- STAR's loop ends at +2.5 s;
- the hand-off pass returns at +3.1 s;
- Solo on the device is done and the GPU pileup starts at +3.6 s;
- STAR's after-mapping steps end at +5.2 s.

The non-host screen, the export and the writers (57 s) are the same work as in the CPU engine. The other samples' times are in
E19-release.md §3. Over the 10 benchmark runs, the GPU path takes 971 s / 4,603 core-s against the CPU engine's 3,303 s / 19,388 core-s.

## 2. Switches and defaults

| variable | default with `--engine gpu` | meaning |
|---|---|---|
| `SINGLET_GPU_PILEUP` | unset | `on`: the GPU pileup, and with it everything below. Unset: the GPU maps the distinct sequences only |
| `SINGLET_ENGINE_GPU_E2E` | **5** (with the pileup on) | 5: records (E8), per-read products (E10) and STARsolo (E11) on the device; STAR's loop maps only the reads without a GPU record. 0: the path before E8 (opt-out). 1–4: the intermediate levels (4 writes incomplete Solo files by design) |
| `SINGLET_ENGINE_GPU_INGEST` | **1** at level 5 | E13: the .1fq decoded straight into the prefill's buffers, no FASTQ temp files. 0: the temp files. Opt-in below level 5 |
| `SINGLET_E16` | on at level 5 | `0`: E16's tail changes off; `SINGLET_E16_OFF=a,b` single cuts off |
| `SINGLET_E16_TRACE` | **0** | `1`: the `[e16]` timing marks on stderr (E16's step tables) |
| `SINGLET_ENGINE_GPU_MEM_GB` | (device − 2 GB) / slots | E17: one sample's device memory budget; the prefill plans its chunk and lanes from it (lean mode, index + one chunk on the device, when the normal plan does not fit; the engine log's `engine_gpu: budget ...` line says which); the pileup's arena is budget − 3 GB. `0`: no budget |
| `SINGLET_ENGINE_GPU_MEM_FALLBACK` | unset | `cpu`: a sample that cannot fit the budget is mapped by STAR on the CPU instead of stopping |
| `SINGLET_ENGINE_GPU_SLOTS` | 1 | samples sharing one GPU (file locks per device, §5) |
| `SINGLET_ENGINE_GPU_DEVICE` | first free slot | pins the device |
| `SINGLET_E15_KEEP_ON_DEVICE`, `SINGLET_E15_PERREAD` | 0, 0 | E15: keep Solo's inputs on the device during STAR's loop; force the per-read CB posterior |
| `SINGLET_GENOME_PRELOAD` | on | E14: the parent reads the index once; `0`: every STAR loads its own |

The pipeline applies the defaults once, after reading its options, by setting the variables in its environment (so STAR's child and
every reader see the same values); a variable already set is kept. The CPU engine (no `--engine gpu`) reads none of them.

## 3. Fallbacks and their lines

Every part falls back on its own; the outputs stay those of the CPU engine. Each fallback prints one line on the run's stderr (STAR's
lines also go to `star_Log.out`), and the level line ends in `: FELL BACK` when the run is below the requested level.

| condition | line on stderr | what runs instead |
|---|---|---|
| no usable GPU (driver, MIG, no device: the probe `singlet gpu-probe`, run before STAR) | `[singlet] SINGLET_GPU_PILEUP ignored: no usable GPU (<reason>); the CPU pileup runs`, `[1fq-ingest] not used (no usable GPU ...)` and STAR's `ENGINE_GPU: not used (<reason>)` | the CPU engine's path |
| GPU pileup declined (cDNA reads > 128 bases, experimental intron rules, `SINGLET_GPU_PILEUP_FORCE_DECLINE`) | `[singlet] SINGLET_GPU_PILEUP ignored: <why>; the CPU pileup runs on the GPU-mapped reads` | GPU mapping, CPU pileup, level 0 |
| ingest not possible | `[1fq-ingest] not used (<why>): the .1fq is decoded to FASTQ files` | level 5 on FASTQ temp files |
| GPU mapping not possible (paired-end, `readFilesCommand`, parameters the engine does not port, E13 decode error, mapping failure such as out of memory) | `[engine_gpu] fallback: ENGINE_GPU: not used (<why>): STAR maps every read` | STAR maps every read (with the ingest: from a chunk source over the .1fq) |
| sample does not fit the budget (E17) | `EXITING because of FATAL ERROR (singlet --engine gpu, E17): the sample does not fit the device memory budget: ...` (exit) or, with `SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu`, `[engine_gpu] fallback: ENGINE_GPU: E17: ... STAR maps every read` | stop, or STAR on the CPU |
| E8's shared region (records on the device) | `[engine_gpu] fallback: ENGINE_GPU: E8 not used (...)` / `E8: the shared region is full; the hand-off file path runs`; `[gpu_pileup] E8: no shared memory object (...)` | the hand-off file (level 0) |
| E10 (per-read products) | `[engine_gpu] fallback: ENGINE_GPU: E10 (level N) not used: <why>`, `E10 decode: not used: <why>`, `E10 level 3 not used (<why>)` | STAR's loop does the per-read work |
| E11 (Solo on the device: CB_UMI_Simple, Gene, Unique, 1MM_multi_Nbase_pseudocounts, 1MM_All, CellRanger2.2 only) | `[engine_gpu] fallback: ENGINE_GPU: E11 (level 5) not used: <why>` | level 3: STAR's Solo counts every read |
| the engine's own refusals (lines of its log with "not used", "failed", "out of memory") | `[engine_gpu] fallback: engine_gpu: ...` | as the line says |
| R11's CB histogram not in the region | `[1fq-ingest] R11 exact CB histogram from the .1fq (STAR's prefill wrote none)` | the pipeline counts it itself |
| non-host screen on the GPU | `[nonhost] engine_gpu: <why>` / `no host Bloom filter; screening on the CPU` | the CPU screen |
| level 4/5 made no per-read summaries but the run fell below level 4 (E11 §5.4) | `EXITING because of FATAL ERROR (singlet --engine gpu, E10): ... its outputs would be incomplete (rerun with SINGLET_ENGINE_GPU_E2E=3)` | STAR stops (exit 104) and the run fails instead of writing incomplete outputs |
| the GPU pileup fails after the CPU skipped its steps | `[gpu_pileup] ERROR: ... rerun with SINGLET_GPU_PILEUP unset` | stops (exit 9) |

## 4. Memory

**Device** (H100 NVL: 93.1 GB visible; GB = 2^30 bytes):
- **The budget.** It is `SINGLET_ENGINE_GPU_MEM_GB`, by default (device − 2 GB) / `SINGLET_ENGINE_GPU_SLOTS`: 91.1 GB with one slot,
  45.5 GB with two.
  - The prefill plans its chunk (16 M reads) and its lane batches from the budget. The index and the reads take what they need, and
    the lanes take the rest.
  - So a run fills the budget whatever its size: the engine's tally peaks at 86.5–89.3 GB on the benchmark samples (pbmc2m: 42.7 GB).
    The normal plan's own estimates are 18.8 GB (pbmc2m) to 70.3 GB (hgmm1k_v3).
  - The engine log's `engine_gpu: budget ...` line has the plan and the tally peak.
- **Lean mode** keeps only the index and one chunk on the device. The prefill picks it when the normal plan does not fit. No
  benchmark sample needs it at 91.1 GB; pbmc1k_v3 needs it below about 35 GB (E17).
- **What the plan holds** (pbmc1k_v3):
  - the index, 16.4 GB (GRCm39 25.1 GB, the barnyard 52.7 GB);
  - the reads, 6.1 GB;
  - the CR4 clip, 7.3 GB;
  - E10's aux, 2.0 GB, and the entries and masks, 2.5 GB;
  - E11's keep, 1.0 GB;
  - the context, 0.5 GB;
  - the lanes, everything else.
- **The GPU pileup's arena** is budget − 3 GB (88.1 GB). It uses 1–33 GB of it: 26 GB on pbmc1k_v3, 33 GB on nuclei2k_v2.
- **The non-host screen** takes 25–29 GB, mostly its 24.9 GB index, outside the budget. It runs after STAR, once the prefill has
  released the device.
- **A sample that cannot fit** stops before allocating, with the E17 line of §3.

**Host** (peak RSS of the whole run at T = 18, GPU path against CPU engine):

| sample | reads | GPU path | CPU engine |
|---|---|---|---|
| pbmc2m / pbmc10m | 2.0 / 10.0 M | 21.8 / 24.9 GB | 25.8 / 26.7 GB |
| pbmc1k_cite_v3_adt | 12.6 M | 29.6 GB | 32.9 GB |
| pbmc1k_cite_v3 (GEX) | 45.4 M | 38.2 GB | 28.2 GB |
| pbmc5p_v1 | 60.0 M | 39.6 GB | 32.3 GB |
| pbmc1k_v3 | 66.6 M | 46.1 GB | 29.7 GB |
| pbmc1k_v2 | 76.9 M | 47.5 GB | 29.4 GB |
| neuron1k_v3 | 92.9 M | 60.8 GB | 35.3 GB |
| hgmm1k_v3 (barnyard index) | 63.1 M | 78.4 GB | 62.2 GB |
| nuclei2k_v2 | 118.7 M | 84.4 GB | 34.9 GB |

The GPU path holds more host memory on large samples, for three reasons:
- the `.1fq`'s reads decoded in memory (E13);
- the E8 region shared with STAR's process (5.3 GB on pbmc1k_v3);
- Solo's kept inputs.

Size a node by the largest sample. The gate runs with `--mem=200G`.

## 5. Device slots

One sample holds a device slot (an exclusive `flock` on `/tmp/singlet_engine_gpu_<PCI bus id>[.k].lock`) while it uses the GPU: the
prefill's device phase (`mapDistinct`; the host assembly of the records still runs under it), the hand-off pass and the pileup, the
non-host screen, and E11's short use after STAR's loop. `SINGLET_ENGINE_GPU_SLOTS=K` (1–8) lets K samples share a device; with E17 each
then keeps to (device − 2 GB) / K, and the non-host screen's 25–29 GB sit outside that budget. Without `SINGLET_ENGINE_GPU_DEVICE` a
sample takes the first free slot of any device, or waits for one. E18 measures what K and thread split a node should run.

## 6. What is byte-identical and what is run-dependent

CPU engine against the GPU path (E11 §2a; every sample of gate 798027):
- byte-identical: every count matrix and `.1pz` member, the 10 STAR Solo files (`Barcodes.stats`, `Gene/Features.stats`,
  `Gene/Summary.csv`, `Gene/UMIperCellSorted.txt`, raw and filtered `barcodes/features/matrix`), `cell_calls.tsv`, `cell_qc_metrics.tsv`,
  `read_qc_summary.json`, `star_SJ.out.tab`, the mapping statistics of `star_Log.final.out`;
- the same rows in another order: `Gene/CellReads.stats` (the pipeline deletes it after parsing; the gate hard-links it);
- the same records in another order: `nonhost_reads.fq`, `star_Unmapped.out.mate1/2`. The exception is nuclei2k_v2's
  `nonhost_reads.fq`, which holds other records (same count) in each engine. That is a pipeline bug when the non-host screen
  subsamples more than 20 M unmapped reads (E15 §5.7, E19-release.md §4d), and the gate names it (`KNOWN_<sample>`);
- run-dependent: timings in `read_stats.tsv` and `star_Log.std.out`; `pileup_stats.json` (`bytes_public`, and the 5' probe's read
  count on the FASTQ-file path); `summary.json`'s `memory_tier`; timings and paths only in `provenance.json` and
  `ambient_contamination.json`.

## 7. Running the gate

```
# build (short-gpu, no GPU): binary in $W/jobs/fable/release/bin/singlet-gpu
sbatch --export=ALL,BRANCH=gpu/fable/release bench/fable/release/jobs/build.sbatch
# every benchmark sample, CPU engine and the defaults; the CPU references are kept under $W/jobs/fable/release/ref/T18/<sample>/
# (g050's GPU is not visible to jobs: exclude it). SAMPLES adds pbmc1k_cite_v3_adt, the antibody library (encode.sbatch)
sbatch --exclude=g050 bench/fable/release/jobs/gate.sbatch
# the GPU path only, against the kept references (a new binary, e.g. after a merge)
sbatch --exclude=g050 --export=ALL,MODES=gpu,BINDIR=<dir> bench/fable/release/jobs/gate.sbatch
# a quick check on debug-gpu
sbatch -p debug-gpu -c 8 --mem=120G --time=00:29:00 --export=ALL,T=8,SAMPLES=pbmc2m,MODES="cpu gpu gpuO0 gpuI0" bench/fable/release/jobs/gate.sbatch
```

`results/gate-<job>/summary.txt` has per run the exit status, wall, core-s, peak RSS, device peaks, the level reached, the ingest and
every fallback line, and per pair a `VERDICT ... IDENTICAL` line; `runs.tsv` and `verdicts.tsv` tabulate them. The modes are listed in
the script's header (`gpuO0`: E2E=0, `gpuI0`: INGEST=0, `gpuT`: the E16 marks, `gpuX<N>`: any variables).
