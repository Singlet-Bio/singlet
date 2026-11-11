# E14: the genome read once per sample and the CR4 probe made cheap (probe)

Experiment E14 ("probe") of the GPU aligner program. Branch `gpu/fable/probe`, from `gpu/fable-engine` at 8e54be40 (the E11 head).
Target: the "STAR load, CR4 probe, start" stage of a sample run, 36 s on pbmc1k_v3 at T = 18 (E11's level-5 run), down to <= 12 s,
with `cr4_probe.json` and every output byte-identical on the CPU engine and on the GPU path.

## 0. Answer in brief

- **The stage is 3–5 s instead of 37–41 s on pbmc1k_v3 at T = 18** (gate 796618, node g050, the baseline binary in the same job):

  | run | stage, E11's table | stage, STAR pid → main STAR maps | wall | core-s |
  |---|---|---|---|---|
  | CPU engine, baseline | – | 41 s (probes 31, main start 10) | 436.2 s | 2,632 |
  | **CPU engine, E14** | – | **7 s** (probes 6, main start 1) | **405.1 s (−31.1 s, −7.1 %)** | **2,517 (−115)** |
  | level 5, baseline (= E11) | 37 s / 46.7 core-s | 39 s (probes 29, main start 10) | 211.9 s | 680 |
  | **level 5, E14** | **3 s / 7.9 core-s** | **5 s** (probes 4, main start 1) | **179.6 s (−32.2 s, −15.2 %)** | **660 (−20)** |

  On **neuron1k_v3** (mouse, GRCm39-2024-A, a 25.6 GB index; gate 796715, node g052) the stage goes from **52–54 s to 2–7 s**
  (each mouse genome load took 12–14 s on one core): CPU engine 519.1 → 477.2 s (**−41.9 s, −8.1 %**), level 5 274.2 → 228.5 s
  (**−45.7 s, −16.7 %**), core-s −53 and −20.
- **Two changes, both only in the probe and the genome load:**
  1. **The genome is read once per sample**, by the parent, in the background while it decodes the .1fq: Genome, SA and SAindex
     with 8 parallel `pread` readers into STAR's own layout on transparent hugepages (16,757 MB in 0.92–0.96 s). The STAR child,
     the two probes and the main STAR all take those pages through `fork()` (copy-on-write, never written), instead of each process
     reading 16.8 GB on one core (8 s, three times per sample). The STAR fork waited 0.0 s for it in every run.
  2. **The two CR4 probes run at the same time** on T/2 threads each, with 2 MB input chunks instead of 15 MB (200k reads were 4
     chunks, so only 4 threads mapped). Both probes now take 4.1–5.3 s together, against 29–31 s in sequence.
- **Gate (a) holds (§3).** `cr4_probe.json` is byte-identical in every comparison on both samples (pbmc1k_v3 88.21 % plain,
  87.56 % cr4; neuron1k_v3 82.24 % / 82.17 %; both ON), and so are the probes' SJ.out.tab and mapping statistics.
  - CPU engine (E14) against E11's kept CPU outputs (gate 796377): **60 identical**; the rest are the files the kept copy lacks,
    paths and timings.
  - CPU engine (E14) against level 5 (E14): **60 identical**, `Gene/CellReads.stats` the same 323,542 rows (sorted), and the same 7
    run-dependent files as E11 §2a.
  - Baseline against E14 in the same job: CPU engine 60 identical, level 5 63 identical; the remaining files differ only in paths,
    timings and record order.
  - neuron1k_v3: baseline vs E14 60 identical on the CPU engine and on level 5, CPU engine vs level 5 60 identical with E11's 7
    run-dependent files; `Gene/CellReads.stats` the same 600,534 rows everywhere.
- **Why the rates cannot move:** the preloaded bytes are the index files' bytes in STAR's layout, and a read maps the same way
  whatever thread or input chunk it lands in. Measured directly (job 796612): the probe's unique rate and SJ.out.tab are the same with
  8, 4 or 1 threads, 15 or 2 MB chunks, STAR's own load, the preload or a SysV segment.
- **Shared memory (§5).** SysV works under Slurm here (limits unlimited, segment charged to the job's cgroup), but it is the slower
  mechanism: STAR's shm load takes 16 s (single-threaded, 4 KB pages), and mapping from it costs **+18 % CPU** (79.9 vs 67.7 user-s
  for 2M pairs at T = 8) because `shmem_enabled=never` keeps shmem off hugepages. Across jobs it would leave unaccounted memory,
  needs cleanup, and a half-loaded segment makes the next run (picked up automatically by `--genome-shared auto`) wait 100 min.
  A file mapping of the index from the page cache (no SysV) sets up in 1.2 s but also gets 4 KB pages on this NFS (random reads
  1.9x slower than from the THP copy). For a persistent service the recommendation is this branch's mechanism one level up: a
  long-lived per-GPU process that preloads once and forks per sample.
- **Not needed:** stretch (c) (optimistic or GPU probe). No OFF decision is on record; both samples here are ON (neuron1k_v3 by
  0.07 points of drop against the 2-point limit, pbmc1k_v3 by 0.65).

## 1. What the stage was (step 1)

The stage runs from `[singlet] STAR pid=` (the parent forks the STAR child) to the main STAR's "Started mapping on" (its genome is
loaded; on the GPU path the prefill starts right after). In the STAR child, before the main run, the CR4 probe forks two more STAR
processes on the first 200,000 reads (`--outSAMtype None --readMapNumber 200000`): `plain`, and `cr4` with the Beta aligner profile
(`--clipAdapterType CellRanger4 --alignSJoverhangMin 8 --outFilterIntronMotifs RemoveNoncanonicalUnannotated
--outFilterMismatchNoverLmax 0.05`). They ran one after the other, then the main STAR started. Each of the three processes did
everything a STAR run does before mapping, on its own.

**Breakdown at T = 18** (E11's level-5 run, gate 796433, node g050, from the STAR logs' second stamps with
`bench/fable/probe/s8_probe_e14.py`; s after `STAR pid=`):

| process | starts | parameters + whitelist | genome load | map + Solo (200k reads) | ends |
|---|---|---|---|---|---|
| probe `plain` | 0 | 1 | **8** | 5 | 14 |
| probe `cr4` | 15 | 1 | **8** | 5 | 29 |
| main STAR | 29 | 1 | **8** | (starts mapping at 38) | |

So the 36–38 s were three single-threaded genome loads (24 s), two probe mappings that used 4 of 18 threads (10 s), and three
parameter/whitelist loads (3 s), strictly one after the other.

**What runs on one core:**
- **The genome load.** `Genome::genomeLoad` (NoSharedMemory) reads Genome (3.22 GB), SA (12.79 GB) and SAindex (1.57 GB) with
  `fstreamReadBig`, one thread, into fresh `new[]` arrays advised for transparent hugepages. The files are already in the page cache
  (the parent's prefetch reads them while the .1fq is decoded: 16,757 MB in 2.9–3.1 s), so the 8 s is a one-core copy plus the page
  faults of 16.8 GB: the probe run alone used 9.0 s user + 9.5 s system in 13.6 s (job 796612). Three times per sample.
- **Parameters and whitelist**: about 1 s per process (the 3M-february-2018 whitelist is read, sorted and deduplicated), three times.
- **The probe's mapping.** STAR cuts its input into chunks of `limitIObufferSize[0] / mates` = 15 MB per mate, about 60k read pairs:
  200k reads are 4 chunks, so 4 threads map and the others wait. Mapping 200k reads took 4 s at T = 8 and at T = 18 alike.

**Measured on debug-gpu at T = 8 (job 796612, node g051).** One probe STAR run alone (`singlet star`, the pipeline's probe arguments,
the first 200k read pairs of pbmc1k_v3's FASTQ):

| run | threads | genome | input chunk | wall | unique | SJ.out.tab md5 |
|---|---|---|---|---|---|---|
| plain, as before | 8 | STAR's load | 15 MB | 13.59 s | 87.81 % | aa68ef76c0a9 |
| plain | 8 | preload (0.96 s, 8 readers) | 15 MB | 6.49 s | 87.81 % | aa68ef76c0a9 |
| plain | 8 | preload | 2 MB | 5.16 s | 87.81 % | aa68ef76c0a9 |
| plain | 4 | preload | 2 MB | 5.56 s | 87.81 % | aa68ef76c0a9 |
| plain | 1 | preload | 2 MB | 10.27 s | 87.81 % | aa68ef76c0a9 |
| plain | 8 | SysV shm (attach) | 2 MB | 5.40 s | 87.81 % | aa68ef76c0a9 |
| cr4, as before | 8 | STAR's load | 15 MB | 13.62 s | 87.52 % | e2a27eb2d50b |
| cr4 | 8 | preload | 2 MB | 5.03 s | 87.52 % | e2a27eb2d50b |
| cr4 | 4 | preload | 2 MB | 5.37 s | 87.52 % | e2a27eb2d50b |
| cr4 | 8 | SysV shm (attach) | 2 MB | 5.30 s | 87.52 % | e2a27eb2d50b |

(These are the FASTQ's first 200k pairs; the pipeline's probe reads the .1fq's first 200k and gets 88.21 % / 87.56 %.)

- **The rates do not depend on the thread count, the chunk size or where the genome came from**, and neither do the probe's
  junctions (the same SJ.out.tab in every row). A read maps the same way whatever thread or chunk it lands in.
- With the genome already in memory, 4 threads are almost as fast as 8 (5.56 vs 5.16 s): what remains is ~1 s of parameters and
  whitelist, ~3 s of mapping and ~1 s of Solo output and exit. Two probes at the same time on half of the threads each therefore cost
  about what one costs.

## 2. What was done

### 2a. The genome is read once per sample (`src/star/GenomePreload.cpp`)

- **The parent reads the index while it decodes the .1fq.** A thread started where the old page-cache prefetch started (before the
  decode and the reference load) calls `star_genome_preload(genome_dir, max(2, min(8, T/2)))`. It:
  - opens Genome, SA and SAindex, reads the SAindex header, and allocates three `new[]` arrays in exactly the layout
    `Genome::genomeLoad` builds (G1 = 200 bytes of padding value 5, the genome, 200 bytes of padding; SA; the SAindex data after its
    header), advised `MADV_HUGEPAGE` before the first touch;
  - reads 64 MB pieces with `pread` on the reader threads (SA first);
  - records the directory's device and inode, the three sizes and the mtimes.

  16,757 MB in 0.96 s with 8 readers alone, 1.86 s with 4 readers while the decode runs (T = 8), 0.92 s and 0.94 s with 8
  readers in the T = 18 gate (mouse: 25,647 MB in 2.40–2.65 s). AnonHugePages 16.4 GB (mouse 25.0 GB): the whole index is on
  2 MB pages. It ends long before the STAR fork, which comes 10–36 s after the start (the decode and the reference load): the join
  before `fork()` waited 0.0 s in every run.
- **Every STAR process of the sample takes those pages.** The parent joins the thread before `fork()`, so the STAR child inherits the
  arrays; the parent drops its references right after the fork. `Genome::genomeLoad` calls `star_genome_preload_take` in its
  NoSharedMemory, no-insertion branch: when the directory, the sizes and the mtimes match, it adopts the arrays (`G1`, and
  `PackedArray::adoptArray` for SA and SAi) and skips the file read; otherwise it logs why and loads the files as before. The two
  probes are forked from the STAR child and take the same pages; the main STAR takes them after the probes. Nothing writes into G, SA
  or SAi, so the pages stay shared copy-on-write: one 16.8 GB copy for the parent, the STAR child and both probes. `star_Log.out`
  says "Genome preload: taken from singlet's preload (… s, shared copy-on-write with the processes forked after it)".
- **The bytes are the files' bytes** in STAR's own layout, so nothing downstream can change. STAR's own load is still used for
  everything the preload does not cover: the shared-memory modes (`--genome-shared`, LoadAndKeep), sjdb or sequence insertion at
  mapping time (those need STAR's larger arrays; the preload is released), SRA input, and a preload that failed or did not match.
- `SINGLET_GENOME_PRELOAD=0` turns it off (the old prefetch, and each STAR process loads its own copy).
- `singlet star <STAR arguments>` (new subcommand) runs the bundled STAR on its own; with `SINGLET_GENOME_PRELOAD_THREADS=N` it
  preloads first. The measurement job uses it.

### 2b. The two probes run at the same time, on half of the threads each, with small input chunks (`src/pipeline/singlet.cpp`)

- `plain` gets `T/2` threads (`--runThreadN`), `cr4` the other `T - T/2`; both are forked, then both are waited for. The decision,
  `cr4_probe.json` and the removal of the probes' Solo and STARgenome directories are unchanged.
- Each probe gets `--limitIObufferSize 4000000 50000000`: 2 MB input chunks per mate instead of 15 MB, so 200k reads are ~25 chunks
  instead of 4. The user's own `--limitIObufferSize` wins if given. The main run keeps STAR's default.
- Each probe's in-process read source uses `max(2, min(4, threads/4))` decode threads, out of its own half.
- If the parent did not preload, the STAR child preloads before forking the probes, so the probes and the main run still share one
  read.
- `[singlet] CR4 probe timing: plain X s, cr4 Y s, concurrent on a+b threads, total Z s; genome preload: …` reports it.
- `SINGLET_CR4_PROBE=serial` restores the old order (one after the other, all threads, STAR's default chunks).

## 3. Identity per output

### 3a. Gate (a), pbmc1k_v3, T = 18 (job 796618, node g050; `bench/star_accel/out_equiv.py` + the gate's own checks)

Binaries: E14 = 2fcb0203 (`bin-n1/`), baseline = 8e54be40, the E11 head (`bin-base/`). Modes: `cpu` (CPU engine) and `gpuE5`
(`--engine gpu`, `SINGLET_GPU_PILEUP=on`, `SINGLET_ENGINE_GPU_E2E=5`), suffix B for the baseline binary; `cpuK` is E11's kept CPU
output (gate 796377).

| comparison | out_equiv | `cr4_probe.json` | probes' SJ.out.tab + mapping stats | `star_SJ.out.tab`, `star_Log.final.out` | `Gene/CellReads.stats` | what differs |
|---|---|---|---|---|---|---|
| **cpuK vs cpu** (E11's kept CPU vs E14's CPU) | **60 identical**, 2 timing/path only, 7 differ | **byte-identical** | identical | identical (126,996 junctions); mapping stats identical | same 323,542 rows (sorted) | 3 files the kept copy lacks (`nonhost_reads.fq`, `star_Unmapped.out.mate1/2`); `provenance.json` (whitelist path `solowire/bin-s5/` vs `probe/bin-n1/`); `pileup_stats.json` `bytes_public`; timings in `read_stats.tsv`, `star_Log.std.out` |
| **cpu vs gpuE5** (both E14) | **60 identical**, 2 timing/path only, 7 differ | **byte-identical** | identical | identical; identical | same rows (sorted) | exactly E11 §2a's 7: record order in `nonhost_reads.fq` and `star_Unmapped.out.mate1/2` (same sorted records: 428,468 and 3,355,559 reads); `pileup_stats.json` `clip5p/probe/n_reads` 12,194 vs 12,000 and `bytes_public`; `summary.json` `memory_tier`; timings in `read_stats.tsv`, `star_Log.std.out` |
| **cpuB vs cpu** (baseline vs E14, CPU engine) | 60 identical, 2 timing/path only, 7 differ | **byte-identical** | identical | identical; identical | same rows (sorted) | record order of the 3 read files (same sorted records; two CPU runs differ in order too); `provenance.json` path; `bytes_public`; timings |
| **gpuE5B vs gpuE5** (baseline vs E14, level 5) | **63 identical**, 2 timing/path only, 4 differ | **byte-identical** | identical | identical; identical | **byte-identical in file order** | `provenance.json` path; `bytes_public`; timings in `read_stats.tsv`, `star_Log.std.out` |

`pileup_stats.json`'s `bytes_public` (the size of the run's outputs) is 1,637 bytes smaller in each E14 run than in the baseline
run of the same job: STAR's `Log.out` files are shorter (`star_Log.out` −338 bytes, each probe's `Log.out` −647 bytes: one
"Genome preload: taken from …" line instead of STAR's load lines), plus 2 bytes of path in `provenance.json` and 3 bytes of timing in
`star_Log.std.out`. Against E11's kept copy, the whitelist path adds its own difference.

So every output that depends on the alignments, the probe or Solo is identical, on the CPU engine and on level 5, to the baseline and
to E11's gate. The only files that change are logs, timings and paths.

### 3b. Second sample: neuron1k_v3 (mouse, GRCm39-2024-A; job 796715, node g052)

| comparison | out_equiv | `cr4_probe.json` | probes' SJ.out.tab + mapping stats | `star_SJ.out.tab`, `star_Log.final.out` | `Gene/CellReads.stats` | what differs |
|---|---|---|---|---|---|---|
| **cpuB vs cpu** | **60 identical**, 2 timing/path only, 7 differ | **byte-identical** | identical | identical (117,254 junctions); mapping stats identical | same 600,534 rows (sorted) | record order of `nonhost_reads.fq`, `star_Unmapped.out.mate1/2` (same sorted records: 490,605 and 3,667,707 reads); `provenance.json` path; `bytes_public`; timings |
| **gpuE5B vs gpuE5** | **60 identical**, 2 timing/path only, 7 differ | **byte-identical** | identical | identical; identical | **byte-identical in file order** | the same: record order of the 3 read files (on this sample two level-5 runs differ in order too), path, `bytes_public`, timings |
| **cpu vs gpuE5** (both E14) | **60 identical**, 2 timing/path only, 7 differ | **byte-identical** | identical | identical; identical | same rows (sorted) | E11 §2a's 7: record order (3 files), `pileup_stats.json` `clip5p/probe/n_reads` 12,276 vs 12,000 and `bytes_public`, `summary.json` `memory_tier`, timings (2) |

`cr4_probe.json` in all four runs: `{"profile": "beta-cr4-sj-mm", "enabled": true, "reason": "on", "probe_reads": 200000,
"unique_pct_plain": 82.240000, "unique_pct_cr4": 82.170000, "max_drop_pts": 2.000000}`. Level 5 ran on this sample too (CB
qualities 70..70; "E11: Solo on the device: 92902231 reads").

## 4. Timings

### 4a. pbmc1k_v3, T = 18, node g050 (gate 796618; one run each, baseline and E14 in the same job)

| run | wall s | core-s | wall s/M | core-s/M | maxRSS |
|---|---|---|---|---|---|
| cpuB (CPU engine, baseline) | 436.2 | 2,632.0 | 6.55 | 39.52 | 29.6 GB |
| **cpu (CPU engine, E14)** | **405.1** | **2,516.5** | 6.08 | 37.78 | 29.6 GB |
| gpuE5B (level 5, baseline = E11) | 211.9 | 679.9 | 3.18 | 10.21 | 61.0 GB |
| **gpuE5 (level 5, E14)** | **179.6** | **659.6** | **2.70** | **9.90** | 61.1 GB |

- CPU engine: **−31.1 s (−7.1 %), −115.5 core-s (−4.4 %)**.
- Level 5: **−32.2 s (−15.2 %), −20.2 core-s (−3.0 %)**. E11 measured level 5 at 210.3–210.8 s on the same node; E14's level 5 is
  2.36x faster than the CPU engine of the same job in wall time (E11: 2.0x).

**Phase breakdown of the level-5 runs** (`bench/fable/solowire/s8_stages_e11.py`; wall s / core-s):

| stage | gpuE5B (baseline) | **gpuE5 (E14)** |
|---|---|---|
| startup | 1 / 1.4 | 1 / 6.3 (the preload's readers) |
| decode .1fq → FASTQ + R11 histogram | 20 / 86.2 | 19 / 86.0 |
| probe/ref load | 4 / 6.9 | 5 / 8.6 |
| **STAR load, CR4 probe, start** | **37 / 46.7** | **3 / 7.9** |
| GPU prefill | 58 / 300.7 | 59 / 308.5 |
| STAR per-read host pass | 9 / 21.0 | 9 / 21.1 |
| STAR stream into pileup | 9 / 16.5 | 9 / 16.5 |
| GPU-pileup hand-off | 16 / 24.3 | 16 / 22.9 |
| GPU pileup + collapse | 1 / 9.9 | 1 / 10.0 |
| non-host | 20 / 42.0 | 20 / 43.7 |
| export, QC, callers | 17 / 35.1 | 17 / 35.6 |
| demux/ancestry/writers | 19 / 88.1 | 20 / 91.4 |
| total | 211 / 678.9 | 179 / 658.6 |

The E11 table ends the stage where the prefill starts (device memory above 4 GB minus the prefill's whole-second duration), so it
reads ~2 s shorter than the STAR-log definition below; both see the same −34 s.

**The stage from the logs** (`bench/fable/probe/s8_probe_e14.py`: `[singlet] STAR pid=` to the main STAR's "Started mapping on";
whole seconds; the probe timing line in ms):

| run | genome | probes (both) | main STAR start | stage |
|---|---|---|---|---|
| cpuB | prefetch 2.86 s (page cache only) | 31 s, one after the other | 10 s (params + whitelist 1, genome 8, 1) | 41 s |
| **cpu** | **preload 0.92 s, 8 readers, waited 0 s** | **5.27 s concurrent on 9 + 9 threads** (plain 5.21, cr4 5.27) | **1 s** (genome 0) | **7 s** |
| gpuE5B | prefetch 2.88 s | 29 s | 10 s | 39 s |
| **gpuE5** | **preload 0.94 s, waited 0 s** | **4.09 s** (plain 4.08, cr4 4.09) | **1 s** | **5 s** |

Per process (s after `STAR pid=`; in brackets parameters + whitelist / genome load / map + Solo, in s):

| run | probe `plain` | probe `cr4` | main STAR |
|---|---|---|---|
| cpuB | 0–15 (1 / 8 / 6) | 16–31 (0 / 8 / 6) | starts 31 (1 / 8), maps at 41 |
| **cpu** | **0–5 (1 / 0 / 3)** | **0–5 (1 / 0 / 3)** | starts 6 (1 / 0), maps at 7 |
| gpuE5B | 0–14 (1 / 8 / 5) | 15–29 (1 / 7 / 5) | starts 29 (1 / 8), maps at 39 |
| **gpuE5** | **0–4 (1 / 0 / 3)** | **0–4 (1 / 0 / 3)** | starts 4 (1 / 0), maps at 5 |

The CPU engine's probes took 5.3 s against level 5's 4.1 s; with one run each I did not look further.

**Memory.** The largest process is unchanged (maxRSS 29.6 GB CPU engine, 61.0 / 61.1 GB level 5). The job's peak anonymous memory
was sampled in the neuron1k_v3 gate (§4b): unchanged too.

### 4b. neuron1k_v3 (mouse, 92.9 M read pairs), T = 18 (job 796715, node g052)

| run | wall s | core-s | wall s/M | core-s/M | maxRSS | peak anon (job cgroup, 1 s samples) |
|---|---|---|---|---|---|---|
| cpuB (CPU engine, baseline) | 519.1 | 2,187.4 | 5.59 | 23.55 | 35.3 GB | 40.45 GB |
| **cpu (CPU engine, E14)** | **477.2** | **2,134.8** | 5.14 | 22.98 | 35.3 GB | 40.45 GB |
| gpuE5B (level 5, baseline) | 274.2 | 818.4 | 2.95 | 8.81 | 83.2 GB | 60.00 GB |
| **gpuE5 (level 5, E14)** | **228.5** | **798.0** | **2.46** | **8.59** | 83.3 GB | 59.82 GB |

- CPU engine: **−41.9 s (−8.1 %), −52.6 core-s (−2.4 %)**. Level 5: **−45.7 s (−16.7 %), −20.4 core-s (−2.5 %)**.
- The stage: E11's table 52 s / 59.2 core-s → **2 s / 4.1 core-s**; from the logs:

  | run | genome | probes (both) | main STAR start | stage |
  |---|---|---|---|---|
  | cpuB | prefetch 5.03 s | 39 s (plain 0–19, cr4 19–39; genome 12–13 s each) | 14 s (genome 13) | 53 s |
  | **cpu** | **preload 25,647 MB in 2.65 s, 8 readers, AnonHugePages 25.0 GB, waited 0 s** | **5.14 s** concurrent on 9 + 9 | 1 s | **7 s** |
  | gpuE5B | prefetch 4.81 s | 39 s (genome 13–14 s each) | 15 s (genome 13) | 54 s |
  | **gpuE5** | **preload 2.40 s, waited 0 s** | **4.11 s** (plain 4.02, cr4 4.11) | 2 s | **6 s** |

- The larger mouse index made the old stage longer (three loads of 25.6 GB at 12–14 s each) and the saving larger. The preload still
  ends long before the STAR fork (2.4–2.65 s; the fork comes 10–36 s after the start).
- One thing to watch: the level-5 decode stage was 32 s / 147.1 core-s in the E14 run against 28 s / 121.4 core-s in the baseline
  run (CPU engine: 8 s / 78.6 against 7 s / 65.6). The preload's 8 readers run during the decode. Copying 25.6 GB costs ~10–13
  core-s, against ~5 for the old single-threaded prefetch, which matches the CPU engine's +13 core-s. The rest of level 5's +26
  core-s and its +4 s are within what one run each can show (pbmc1k_v3's decode did not move: 19 s / 86.0 against 20 s / 86.2). If
  it repeats, fewer readers (the preload has ~25 s of slack on the GPU path) would trade preload time for decode time.

## 5. Shared memory, and the genome across samples (the persistent service)

**The node (g051, job 796612; g050 runs the same image):** kernel 6.8.0-136 (Ubuntu 24.04.4), 2 × Xeon Gold 6238R, 750 GB.
`kernel.shmmax` and `kernel.shmall` are unlimited (18446744073692774399), `shmmni` 4096. Transparent hugepages `enabled=madvise`,
`defrag=madvise`, **`shmem_enabled=never`**; no hugetlbfs pages (`nr_hugepages` 0). `/dev/shm` is a 376 GB tmpfs. Slurm puts each job
in a cgroup v2 with `memory.max` = `--mem`.

**SysV shared memory works under Slurm here** (the same job, STAR's own protocol through `singlet genome load|unload`):

| step | result |
|---|---|
| `singlet genome load` (STAR `LoadAndExit`) | 16.08 s: one thread copies the index into 4 KB shmem pages, twice the private load's 8 s |
| segments | `0x06cecaf0` 17,571,269,329 bytes (index + header) and `0x06cecaf1` 1 byte (STAR's use counter); key = the genome directory's inode |
| the job's cgroup | `memory.current` +16.40 GB, all of it `shmem`: the segment is charged to the job that touched it; ShmemHugePages 0 |
| a probe attached to it (`LoadAndKeep`) | 5.40 s plain / 5.30 s cr4, the same rates and SJ.out.tab as the private genome |
| mapping 2M read pairs, T = 8 | **shm 15.98 / 16.00 s, 79.9 / 79.6 user-s; private THP preload 14.02 / 13.99 s, 67.7 / 67.7 user-s** (STAR's own load: 21.72 s, 72.6 user-s) |
| `singlet genome unload` (STAR `Remove`) | 2.86 s; the 16.4 GB went back to the job; no segment left (the job's trap removes it on every exit path) |

So within one job, a SysV genome is worse than the fork-inherited private copy on every count:
- it takes 16 s to load instead of 1–2 s (STAR's loader is single-threaded and shmem faults 4 KB pages);
- **mapping from it costs 18 % more CPU** (13 s instead of 11 s for the mapping phase of 2M pairs): with `shmem_enabled=never` the
  segment is on 4 KB pages, and STAR's seed search is random reads into a 12.8 GB suffix array, where 4 KB pages miss the TLB.
  The private copy is on 2 MB pages;
- it must be removed on every exit path, or it outlives the job.

**Across jobs (a genome that stays on the node for the next sample)** SysV has problems beyond speed:
1. **Accounting.** The pages stay charged to the cgroup of the job that loaded them. When Slurm removes that cgroup the charge moves
   to the parent: 17.6 GB (human) or 26.9 GB (mouse) of node memory that no job owns and that Slurm's scheduler does not see, so it
   keeps placing jobs as if the memory were free.
2. **Cleanup.** Slurm itself does not remove SysV segments when a job ends; whether Clipper's node epilog (`epilog.sh`) does, I
   could not check (it is not readable from the login node). A job that crashes leaves its segment behind.
3. **A half-loaded segment blocks the next job for 100 minutes.** STAR marks a segment complete by writing the genome size into its
   header at the end of the load. A process that attaches a segment without it "sleeps 1 min" up to 100 times and then fails. A load
   that crashed half-way leaves exactly that. And `--genome-shared auto` (N21, the default) switches every singlet run on a Clipper
   node to `LoadAndKeep` as soon as `shmget(key)` finds *any* segment for the genome directory, complete or not.
4. **Integrity.** STAR creates the segment with mode 0666: any user on the node can attach it read-write. An index rebuilt in the same
   directory with the same sizes is not detected (the key is the directory's inode, the check is two sizes).
5. **Pages.** 4 KB unless the administrators enable THP for shmem (`shmem_enabled=advise` or `within_size`), a node setting; until
   then every sample mapped from it pays the 18 %.

**Recommendation for the persistent service:** keep this experiment's mechanism and move it one level up. A long-lived process per GPU
(one Slurm job that runs the samples of a batch one after another) reads the index once into private THP memory with parallel
readers and forks one child per sample; each sample's STAR takes the pages copy-on-write exactly as the probes do here. The memory is
charged to the service job and freed by the kernel when it ends or crashes; there is nothing to clean up and no stale state; the
pages are 2 MB. The change from this branch is small: the driver calls `star_genome_preload` once and does not call
`star_genome_preload_release` after each sample's fork. Note what it buys: from the page cache the preload takes ~1 s (8 readers)
and already runs hidden behind the decode (the STAR fork waited 0.0 s for it in every run), so a resident genome saves ~1 s of
background work and the 17.6 GB page-cache read per sample, not wall time. The page cache itself already keeps the index files on the
node between jobs (`cat` of the three files: 2.6–2.9 s on g050/g051).

`--genome-shared on` / N21 remain useful where several *processes* that do not share a parent must map one copy (e.g. many CPU-only
jobs packed on one node); on the GPU nodes as configured, they make mapping ~18 % more expensive than the default path now is.

**File-backed mmap (no SysV), measured** (job 796694, g051, GRCh38 index, `bench/fable/probe/mmap_e14.py`). `/mnt/projects` is
NFS 4 (`harbor-ib:/srv/projects`). Random 8-byte reads are 20 M positions of the SA on one thread (numpy gather, so loads overlap;
it shows the page-size cost, not STAR's):

| mapping | set-up | page size (smaps) | random 8-byte read |
|---|---|---|---|
| `mmap(MAP_SHARED \| MAP_POPULATE)` of Genome, SA, SAindex from the page cache | **1.16 s** for 16.8 GB (no copy) | 4 KB (`FilePmdMapped` 0) | 35.1 ns |
| SA copied into a `/dev/shm` file and mapped (what SysV gives) | 9.55 s for 12.8 GB (one thread) | 4 KB (`ShmemPmdMapped` 0) | 30.5 ns |
| private copy, THP (what the preload gives) | 5.06 s for 12.8 GB (one thread; the preload uses 8) | 2 MB (`AnonHugePages` 12.3 GB) | **18.4 ns** |

- A file mapping is as quick to set up as the parallel preload (1.2 s against 0.9 s) and needs no copy and no SysV: every process
  and every job on the node would share the page cache. But NFS gives the mapping 4 KB pages, and random reads cost 1.9x the THP
  copy's here (the shmem mapping 1.7x; STAR mapping from shmem costs +18 % CPU, so a file mapping should cost at least as much).
- The page cache is not pinned: under memory pressure it is evicted and the mapping faults back to NFS. Its pages are charged to
  whichever job read the files first.
- STAR also needs 200 padding bytes before and after the genome in the same array, so the Genome file cannot be used as `G1` as it
  is; it would take an anonymous page mapped in front of a page-aligned file mapping and a private write to the last page. Not done:
  the private THP copy is faster to map from and costs ~1 s of hidden background reading.

## 6. Surprises

- **The 8 s genome load was not the disk.** The files were already in the page cache (the parent's prefetch read them in 2.9 s while
  the .1fq was decoded). The 8 s was STAR copying 16.8 GB out of the page cache on one core (9.0 s user + 9.5 s system in the probe
  run alone), and it happened three times per sample. Eight `pread` threads do the same copy in 0.92–0.96 s.
- **The probe's mapping was limited by STAR's input chunk size, not by threads.** 200k reads fill 4 chunks of 15 MB, so at most 4 of
  the 18 threads mapped. With 2 MB chunks a probe on 4 threads is as fast as one on 8.
- **SysV shared memory is slower than a private copy on these nodes**, both to load (16 s, single-threaded, 4 KB shmem pages) and to
  map from (+18 % CPU), because `shmem_enabled=never` keeps shmem off hugepages. The node's sysctl limits are not the obstacle; they
  are unlimited.
- **`--genome-shared auto` picks up any segment** for the genome directory on a Clipper node, including a half-loaded one left by a
  crash, after which STAR sleeps up to 100 minutes waiting for a load that never ends.
- **The CPU engine gains as much wall time as the GPU path** (-31 s against -32 s): the stage was the same serial 34 s in both.
- The probe's rates depend on which 200k reads it sees: the .1fq's first 200k (the pipeline) give 88.21 % / 87.56 %, the FASTQ's
  first 200k (the measurement job) 87.81 % / 87.52 %. Both are "ON" by a wide margin; it is a reminder that the probe is a sample.

## 7. What remains

- **The stage now** (4–5 s at T = 18) is the two concurrent probes (4.1–5.3 s: ~1 s parameters and whitelist, ~2–3 s mapping, ~1 s
  Solo output and exit) and ~1 s for the main STAR's parameters and whitelist. Further steps, in order of size:
  1. **Run the probes during the decode.** They need only the genome (now ready ~1–2 s after the start) and the first 200k reads of
     the .1fq; the decode and the reference load take ~25 s before the STAR fork. Forking the probes from the parent as soon as the
     preload ends would take them off the critical path (−4 to −5 s) at the cost of sharing cores with the decode. The decision must be
     known before the main STAR's arguments are built, which it would be. Not done here because it moves the probe out of the STAR
     child (a restructuring of the start sequence, outside this experiment's scope).
  2. **Load the whitelist once** (in the parent, inherited like the genome): ~1 s off each STAR process's start, so ~1 s off the
     probes and ~1 s off the main start.
  3. Stretch (c) (an optimistic probe that starts the main run before the decision, or the probe on the GPU) is not needed for the
     12 s target and was not done.
- **How often the probe says OFF.** Every pbmc1k_v3 run says ON, and so does neuron1k_v3 (82.24 % / 82.17 %, a drop of 0.07
  points). `docs/research-beta/gpu-host-offload.md` notes "CR4 probe -> profile ON" in every log of its benchmark set; I found no recorded OFF. With the probe now ~5 s, caching its
  decision per sample is not worth the risk of a wrong cache key.
- **The persistent service** (§5): a long-lived per-GPU process that keeps the private preloaded genome and forks per sample, not a
  SysV segment across jobs.
- **Memory.** The parent now holds the 16.8 GB index (anonymous, THP) during the decode and the reference load, where before it was
  only page cache; after the STAR fork it is the same single copy as before (and the probes no longer hold copies of their own).
  The job's peak anonymous memory did not change on neuron1k_v3 (40.45 / 40.45 GB CPU engine, 60.00 / 59.82 GB level 5): the
  peak comes later, in the mapping and the pileup, when the main STAR holds the genome anyway.

## 8. Files touched, and why

| file | change | why |
|---|---|---|
| `src/star/GenomePreload.cpp`, `src/star/GenomePreload.h` (new) | `star_genome_preload` (parallel `pread` of Genome/SA/SAindex into STAR's layout, THP advised), `star_genome_preload_take` (hand-over with directory, size and mtime checks), `star_genome_preload_release` | the genome read once per sample |
| `include/singlet/star/star_api.h` | declares `star_genome_preload` and `star_genome_preload_release` | the pipeline calls them |
| `src/star/Genome_genomeLoad.cpp` | NoSharedMemory, no-insertion branch: take the preloaded arrays when they match, skip the file read and the THP advice for them; release the preload in the shared-memory and insertion branches; log "Genome preload: …" | the genome load (in scope) |
| `src/star/PackedArray.h` | `adoptArray(char*)` (take an allocated array, owned and freed as STAR's own) | SA and SAi take the preloaded arrays |
| `src/pipeline/singlet.cpp` | the parent's preload thread (replaces the page-cache prefetch when it runs; `SINGLET_GENOME_PRELOAD=0`), its join before the STAR fork and the release after it; the probe block: the two probes concurrently on T/2 + T−T/2 threads with 2 MB input chunks, the timing log line, `SINGLET_CR4_PROBE=serial`; the `singlet star` subcommand | the probe (in scope) |
| `bench/fable/probe/jobs/{build,measure,gate,mmap}.sbatch`, `bench/fable/probe/s8_probe_e14.py`, `bench/fable/probe/mmap_e14.py` | build (with `REV=`), the step-1 measurement job, the gate (new and baseline binaries in one job, probe comparisons, memory samples), the mapping measurement; the probe-stage script | deliverable |
| `docs/gpu-engine/fable/E14-probe.md` | this report | deliverable |

Not changed: the decode, the prefill, the engine, STAR's mapping and Solo code, the probe's decision rule and `cr4_probe.json`.

## 9. Jobs, commits, how to run

**Branch `gpu/fable/probe`** (from `gpu/fable-engine` 8e54be40; `gpu/fable-engine` had not moved at the end, so no merge):
- 806e975c: build job (`REV=` builds a given commit).
- 2fcb0203: the code (preload, concurrent probes, `singlet star`). The binaries measured are this commit.
- 251565f0: step-1 measurement job.
- c037010c, e8aff0d4, ee47368c, 3470f478: gate job and probe-stage script (probe logs, `PRESTAGE=`, memory samples).
- 8f3b5e5a: the mmap measurement.
- This document is the next commit.

**Jobs** (all `--nice=10000`; `$W = /mnt/projects/debruinz_project/debruinz/singlet`):

| job | partition, node | what | result |
|---|---|---|---|
| 796558 | short-gpu (no GPU), g050 | build 8e54be40 → `$W/jobs/fable/probe/bin-base/` | 271 s |
| 796585 | short-gpu (no GPU), g050 | build 2fcb0203 → `$W/jobs/fable/probe/bin-n1/` | 273 s |
| 796612 | debug-gpu, g051, T = 8 | step 1: node facts, probe STAR runs alone, SysV shm load/attach/map/unload, level-5 runs new and baseline | §1, §5 |
| 796618 | short-gpu, g050, T = 18 | gate (a) pbmc1k_v3: cpuB cpu gpuE5B gpuE5 + E11's kept CPU | §3a, §4a |
| 796715 | short-gpu, g052, T = 18 | neuron1k_v3 (mouse): cpuB cpu gpuE5B gpuE5 (and the per-probe rows of 796618) | §3b, §4b |
| 796694 | debug-gpu, g051 | mmap measurement | §5 |

No job left a shared-memory segment: 796612 removed its own (trap on every exit path) and the gates end with `ipcs -m` = 0 segments.

**Results:** `$W/jobs/fable/probe/results/{measure-796612,gate-<job>,mmap-<job>}/` (`summary.txt`, `equiv-*.txt`, per run
`stages.txt`, `probe_stage.txt`, `run.log.ts`, `star_Log.out`, `mem.tsv`, `out/` with `cr4_probe/`). Logs: `$W/logs/e14-*-<job>.out`.

**Run:**
- build: `sbatch --export=ALL,OUTD=<dir>[,REV=<commit>] bench/fable/probe/jobs/build.sbatch`;
- gate: `sbatch --export=ALL,BINDIR=<new>,BASEDIR=<baseline>,MODES="cpuB cpu gpuE5B gpuE5",PAIRS="cpuB:cpu gpuE5B:gpuE5 cpu:gpuE5" bench/fable/probe/jobs/gate.sbatch`
  (`SAMPLE=neuron1k_v3 GENOME=GRCm39-2024-A READSM=92.902231` for the mouse sample; a mode can carry `:ENV=V,…`);
- step 1: `sbatch bench/fable/probe/jobs/measure.sbatch` (debug-gpu); mmap: `sbatch bench/fable/probe/jobs/mmap.sbatch`.

**Switches** (both on by default):
- `SINGLET_GENOME_PRELOAD=0`: no preload (the old page-cache prefetch; every STAR process loads its own copy).
- `SINGLET_CR4_PROBE=serial`: the probes one after the other on all threads with STAR's default chunks.
- The run log says what happened: `[singlet] Genome preload: … waited X s for it at the STAR fork`, `[singlet] CR4 probe timing: …`,
  and each STAR `Log.out` "Genome preload: taken from singlet's preload (…)" or "Genome preload: not used (<why>), loading from the
  files".
