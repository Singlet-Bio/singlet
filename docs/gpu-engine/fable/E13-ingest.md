# E13: the .1fq straight into the prefill's packed buffers (ingest)

Experiment E13 ("ingest") of the GPU aligner program. Branch `gpu/fable/ingest`, from `gpu/fable-engine` at 8e54be40 (the E11 head),
with `gpu/fable-engine` merged in twice: at e2b0000e (E14, "probe") in 11a19707, and at d03d6c47 (E12, "prefill") in 1c25e24e. On the GPU
path, with `SINGLET_ENGINE_GPU_INGEST=1`, the .1fq goes from disk into the prefill's packed buffers:
no FASTQ temp files, no second text parse. The switch is off by default; with
it unset the binary runs the code it ran before (§2b).

## 0. Answer in brief

- **On pbmc1k_v3 at T = 18, level 5 with ingest takes 149.1 s / 564 core-s, against 175.9 s / 666 without it.** That is −26.8 s
  (−15.2 %) and −101 core-s (−15.2 %). The figures are from gate 796992 (node g051, final binary 12c54287, E14 merged), with both
  modes run twice in the same job:

  | run | wall s | core-s | maxRSS |
  |---|---|---|---|
  | level 5 (gpuE5, gpuE5b) | 175.7, 176.1 | 667.4, 663.9 | 61.0 GB |
  | **level 5 + ingest (gpuE5i, gpuE5ib)** | **148.3, 149.9** | **565.0, 563.9** | **45.8 GB** |
  | ingest with every phase inside the prefill (gpuE5s) | 153.2 | 565.9 | 45.8 GB |
  | ingest with the clip run on every read (gpuE5n) | 152.9 | 626.4 | 45.8 GB |
  | CPU engine, same job | 400.4 | 2,496.2 | 29.7 GB |

  - **Against E11's level 5** (210.6 s, 679 core-s), E14 and E13 together save −61.5 s (−29 %).
  - **The target was about −30 s.** It assumed the prefill spent 16 s reading reads; in these jobs that was 12 s (§4).
  - **neuron1k_v3** (T = 8, job 797012): 246.6 → 196.6 s (**−50.0 s, −20.3 %**), 721 → 566 core-s (−156), maxRSS 82.8 → 60.4 GB.
  - **pbmc1k_v3 at T = 8** (job 797150): 207.0 → 176.1 s (−31.0 s), 657 → 553 core-s (−104).
- **Where the time goes** (§3b):
  - **The parent's FASTQ decode is gone.** "decode .1fq → FASTQ + R11" (20 s, 90 core-s) is now a head scan of 2.7 s (19–29 core-s).
    The scan reads only the barcode-dictionary, UMI and length columns.
  - **The prefill no longer reads reads** (12 s → 0 s):
    - the pipeline decodes the .1fq into the packed buffers on its 18 threads in 3.4 s, beside its reference load and before it forks
      STAR;
    - STAR's process inherits the arrays through `fork()`;
    - STAR's process runs the CellRanger4 clip (7.5 s) beside E14's probes, and the barcode pass (1.5 s) after its parameters;
    - the prefill waited 0.3 s for them.

    The prefill is 42 s instead of 54 s.
  - **The clip runs once per distinct sequence** (31.9 M of 66.6 M reads): 88 thread-s instead of 168. That is worth 3.8 s and
    62 core-s.
  - **Running the phases before the prefill** is worth 4 s at T = 18 and nothing at T = 8, where the host work is CPU-bound (§4).
  - **maxRSS is 15 GB lower.** STAR's process no longer maps the FASTQ or holds per-thread sequence copies, and the quality is held
    as 2-bit codes.
- **Gate (a) holds (§2).**
  - **CPU engine vs level 5 + ingest:** 60 identical, 2 differ in timings or paths only, 7 differ. The 7 are E11 §2a's run-dependent
    files, the same as CPU vs level 5 in the same job: record order in `nonhost_reads.fq` and `star_Unmapped.out.mate1/2` (the same
    sorted records), `bytes_public`, `memory_tier`, and timings. The following are identical:
    - mapping statistics;
    - `SJ.out.tab` (126,996 junctions);
    - `Gene/CellReads.stats` (the same 323,542 rows);
    - R11.
  - **Level 5 without vs with ingest:** 63 identical. Every read file and `CellReads.stats` is byte-identical in file order, and so
    is the GPU pileup digest. Only timings and one value differ: the 5' probe's sample, 12,194 reads with ingest as on the CPU
    engine, against 12,000 (§4).
  - **Ingest's variants** (a second run, all phases in the prefill, no dedup): 63 identical with each other.
  - **With the switch unset** (default GPU path, level 5, CPU engine) against E11's kept outputs: 60 identical each.
  - **R11:** the pipeline's R11 is byte-identical to E11's reference in every run. In ingest runs it comes from STAR's region.
  - **neuron1k_v3:** level 5 vs level 5 + ingest gives 60 identical. The rest is record order, timings and the probe count (12,276).
    `CellReads.stats` is byte-identical in file order.
- **The handshake (§1d):**
  - STAR's prefill computes R11 from the decode and writes it into the E8 region behind the magic `SGR11H01` in `E2EHeader`.
  - The pipeline keeps a `dup` of the region's descriptor.
  - Its CB corrector fetches R11 lazily (`set_pending` / `settle`) the first time a barcode has two or more candidates, which is
    after STAR's prefill has returned.
  - Fallback: a head scan of the .1fq in R11 mode.
- **With E12 merged** (1c25e24e; gate 797219, T = 18, node g052; §2e, §3f):
  - level 5 takes 153.6 s / 656 core-s without ingest and **127.9 s / 561 with it**: **−25.6 s (−16.7 %), −95 core-s**. The saving
    is about the same as without E12, so the two experiments add up.
  - The prefill is 20 s, with no read-in.
  - Gate (a) gives the same verdicts as in §2a:
    - CPU vs level 5 + ingest: 60 identical, E11's 7;
    - level 5 without vs with ingest: 63 identical;
    - with the switch unset, against E11's kept outputs: 60 identical;
    - R11 identical.
  - Against E11's level 5, E12, E13 and E14 together save −82.6 s (−39 %). Level 5 with ingest is 3.09 times faster than the CPU
    engine of the same job.

## 1. Design

### 1a. The pipeline (`src/pipeline/singlet.cpp`)

With `SINGLET_ENGINE_GPU_INGEST=1` on the GPU path (`--engine gpu`, `SINGLET_GPU_PILEUP=on`, an E2E level set), the file-mode decode
of the .1fq into `R1.fastq` / `R2.fastq` on /dev/shm is replaced by a **head scan** (`RunSet::scan_head`,
`include/singlet/fq/read_source.h`). Per block it reads the barcode-dictionary column, the UMI column and the cDNA lengths only, and
gives what the pipeline needs before STAR:
- the barcode-dictionary counts for barcode discovery (file mode counted them during its decode);
- the 5' prefix probe's sample, taken as FIFO pass 1 takes it (12,194 reads on pbmc1k_v3, like the CPU engine; file mode takes 12,000);
- per block, the reads given to STAR, their bases and the longest cDNA read, so that every array of the decode is sized once.

The scan's result is the `Plan` (`include/singlet/engine_gpu_ingest.h`): the open `RunSet`, per block the first read, first base and
first read name, the longest cDNA read, the barcode read's length and the quality mode. STAR is forked with `/dev/null` as its read
files; the CR4 probe's two STARs read the .1fq through the existing in-process chunk source.

The conditions are checked at run time. When one fails, the file decode runs as before and the log says why
(`[1fq-ingest] not used (...)`):
- GPU pileup on and an E2E level set;
- a droplet scRNA assay, CB_UMI_Simple, the barcode in R1, a whitelist, CB and UMI of at most 21 bases;
- no duplicate counts in the .1fq, a barcode read of at most 63 bases, quality stored as codes (BINNED4, ILLUMINA4, BINNED2 or none);
- no constant 5' prefix to clip.

### 1b. The decode, in three phases

- **A, `decodeReads` (`src/star/SeqCacheGpuIngest.cpp`).** Every block goes through `RunSet::decode`, the decoder file mode used, on
  the job's threads, each block into its own ranges of the final arrays:
  - the cDNA in STAR's numeric code with its offsets (the engine's `Input::seq/off`, swapped in, never copied);
  - the cDNA quality as 2-bit codes;
  - the barcode read as 3-bit codes;
  - the bases per quality code.

  A needs nothing from STAR, so **the pipeline runs it right after its head scan** (`startParent`), beside its reference load and
  E14's genome preload. It joins A before forking STAR (`joinParent`). STAR's process inherits the arrays through `fork()`
  (copy-on-write: nothing is copied), and the pipeline frees its own mapping right after the fork (`releaseParent`), as E14 does for
  the genome.
- **B, `barcodes`.** This phase needs STAR's barcode layout and whitelist, known once STAR has read its parameters. It produces:
  - E10's per-read words: CR and UR 3-bit keys, and the low-quality bits from the quality plane;
  - E11's quality counts: RNA Q30 from the per-code base counts, CB counts, the CB quality range;
  - R11's exact CB histogram.
- **C, the CellRanger4 clip (`cr4Clip`, `src/star/SeqCacheGpuPrefill.cpp`).** It uses STAR's own `ClipMate` / `ClipCR4` code: the opal
  alignment of the TSO, 64 reads at a time, then `ClipMate::clip` for both ends. The result is the engine's `pre5`, `pre3` and
  `cr4info`, exactly as `readSequences` computed them from the FASTQ.

  It runs **once per distinct cDNA sequence**. A read's clip depends on its length and bases alone: opal's result for one sequence
  does not depend on the others in its batch (checked), and `ClipMate::clip` and `polyTail3p` read nothing else. So the reads are
  grouped by an exact-match hash table (a 64-bit hash of length and bases, then the bytes compared), the read that took the group's
  slot is clipped, and the others copy its result. `SINGLET_ENGINE_GPU_INGEST_CLIPDEDUP=0` clips every read (gate mode `gpuE5n`).

**Where each phase runs:**
1. **STAR's process, before its CR4 probe** (`startEarly`, called by the pipeline's STAR child). A thread takes the inherited reads, or
   decodes them if the pipeline did not. With the pipeline's Beta profile (`--clipAdapterType CellRanger4`), it then runs C right away
   with the ClipMates that `ParametersClip` builds for that profile: the 10x TSO at 5', polyA at 3', no other clipping.
2. **STAR.cpp, after its parameters and before the genome load** (`seqCacheGpuEarly`). A second thread joins the first and runs B.
   It runs C again only in two cases:
   - STAR's own ClipMates differ from the ones C ran with (compared by the fields the clip reads);
   - the CR4 probe dropped the profile, in which case the early clip is discarded.
3. **The prefill** (`ingestReads`) joins that thread and swaps the arrays into `Input`.

With `SINGLET_ENGINE_GPU_INGEST_EARLY=0`, none of this runs early: the prefill runs A, B and C itself (gate mode `gpuE5s`).

The threads' state is on the heap and never destroyed, because an `exit()` while a thread runs must not run `~thread`. It is also
per pid: the CR4 probe's forked STARs inherit the memory but not the threads, and never touch it.

### 1c. STAR's loop, Unmapped.out and the non-host reads

The reads the device did not cover (level 4/5) and every Unmapped.out record are written from the packed arrays. `ingRecord` rebuilds
a read's FASTQ record with the bytes file mode wrote:
- the name `@r<n>`;
- the cDNA from `seq`, with its quality from `qual` and `qualPhred`;
- the barcode read from `r1`, with quality 'F'.

Where the prefill does not run, STAR reads from a chunk source over the .1fq (`StarChunkSourceT`, the pipeline's existing in-process
source) with file mode's format options. No second copy of the reads exists:
- the engine's `seq` and `off` are the decode's arrays, swapped in;
- the host planes are freed when the prefill returns;
- the pipeline's copy is gone after the fork.

### 1d. R11's handshake

R11 is the stream CB corrector's exact CB histogram: for each whitelist entry, the reads whose CB is exactly that entry and whose UMI
has no N and is not a homopolymer. File mode computed it in the pipeline during its decode. With ingest it travels as follows:
1. **STAR's prefill writes it.** It computes the histogram in phase B (against STAR's sorted whitelist) and writes it into the E8 region
   right after the decode (`publishR11`): `nR11` pairs (whitelist index, reads) at `oR11`, the whitelist size and the total, and then
   the magic `SGR11H01` in `E2EHeader`.
2. **The pipeline keeps its own descriptor of the region** (`Plan::r11Fd`, a `dup`). It needs one because the region is unmapped and
   its descriptor closed when the prefill releases it.
3. **The pipeline's `CbCorrector` gets the histogram lazily.** `set_pending(fill)` registers the fetch, and `settle()` runs it once
   (`std::call_once`) the first time a correction needs exact counts, that is, a barcode with two or more candidates. That happens
   after STAR's first BAM record has reached the pipeline, so after the prefill has returned and the histogram is in the region. Exact
   hits and single candidates resolve as before without it.
4. **Fallback.** If the region holds no histogram at that point (no region, or the prefill did not run or stopped before its decode),
   `fill` computes R11 from the .1fq itself: `scan_head` in R11 mode, 3 s on 18 threads.

In every gate run the histogram the pipeline used (`SINGLET_R11_DUMP`) is byte-identical to E11's reference (4,835,008 bytes,
63,169,750 reads, 302,186 barcodes). The prefill also checks it against the device's own R11 (identical, 0 entries differ).

### 1e. The `.1fg2` mapping

`PackedReads` (`engine_gpu_ingest.h`) is a struct of arrays. Its header comment says field by field what a `.1fg2` reader would fill
instead of the `.1fq` decode:
- the cDNA plane unpacked to `seq` / `off`;
- E10's `aux` from its CB, UMI and quality planes;
- `qual` as is when it holds 2-bit codes;
- the chunk table as `first` / `nameAt`;
- E11's counts and R11 per chunk.

Nothing of `.1fg2` is implemented here.

## 2. Identity per output

The tools are `bench/star_accel/out_equiv.py` plus the gate's own checks (`bench/fable/ingest/jobs/gate.sbatch`):
- the mapping statistics of `star_Log.final.out`, without the timing lines;
- `star_SJ.out.tab`;
- the read files (`nonhost_reads.fq`, `star_Unmapped.out.mate1/2`) compared by their sorted 4-line records;
- `Gene/CellReads.stats` by sorted rows and in file order;
- the R11 histogram the pipeline used (`SINGLET_R11_DUMP`).

### 2a. Gate (a): pbmc1k_v3, T = 18 (job 796992, node g051, binary 12c54287 `bin5/`, E14 merged)

The modes are named in §7. `gpuE5` is E11's level 5 as E14 left it, the baseline. `gpuE5i` is the same with
`SINGLET_ENGINE_GPU_INGEST=1`.

| comparison | out_equiv | `star_SJ.out.tab`, `star_Log.final.out` | read files (sorted records) | `Gene/CellReads.stats` | what differs |
|---|---|---|---|---|---|
| **cpu vs gpuE5i** | **60 identical**, 2 timing/path only, 7 differ | identical (126,996 junctions); mapping stats identical | same records: `nonhost_reads.fq` 428,468 reads (sorted sha256 c115c73474e99993), `Unmapped.out.mate1/2` 3,355,559 reads (766e191fcc6181a5, 31fc9ef1d3d55a89); order differs | same 323,542 rows (sorted) | E11 §2a's 7 run-dependent files, as for cpu vs gpuE5 (E14 §3a): record order in the 3 read files; `pileup_stats.json` `bytes_public` only (`clip5p/probe/n_reads` is now 12,194 on both, §4); `summary.json` `memory_tier` (Medium vs Large); timings in `read_stats.tsv` (the GPU path's extra `encode` line too) and `star_Log.std.out` |
| **gpuE5 vs gpuE5i** (switch off vs on) | **63 identical**, 3 timing/path only, 3 differ | identical; identical | **byte-identical** (same order) | **byte-identical in file order** | `pileup_stats.json`: `clip5p/probe/n_reads` 12,000 → 12,194, and `bytes_public`; timings in `read_stats.tsv` (`encode` 20.17 → 2.76 s) and `star_Log.std.out` |
| **gpuE5i vs gpuE5ib** (second run) | **63 identical**, 3 timing/path only, 3 differ | identical; identical | byte-identical | byte-identical in file order | `bytes_public` (tens of bytes of logs); timings in `read_stats.tsv`, `star_Log.std.out` |
| **gpuE5i vs gpuE5s** (all three phases in the prefill) | **63 identical**, 3 timing/path only, 3 differ | identical; identical | byte-identical | byte-identical in file order | the same: `bytes_public`, timings |
| **gpuE5i vs gpuE5n** (the clip on every read) | **63 identical**, 3 timing/path only, 3 differ | identical; identical | byte-identical | byte-identical in file order | the same: `bytes_public`, timings |

Every level-5 run of the job also reports:
- the same GPU pileup digest (d2aa9309…, 15,324,458 molecules);
- "0 primary alignments differ from the stream's";
- the same Solo-on-device counts (903,105 distinct CRs, 36,678,568 counted reads, 1,136 cells).

So every output that depends on the reads, the alignments or the barcodes is byte-identical:
- with ingest against the baseline;
- across ingest's three variants;
- against the CPU engine, up to E11's run-dependent files.

The one value ingest changes is the 5' probe's sample size, which now matches the CPU engine's (§4).

### 2b. With the switch unset: against E11's kept outputs (gate 796377)

The same job compared its runs without the switch with E11's kept `out/` directories.

| comparison | out_equiv | `star_SJ.out.tab`, `star_Log.final.out` | `Gene/CellReads.stats` | what differs |
|---|---|---|---|---|
| **gpuD vs E11's gpuD** | **60 identical**, 2 timing/path only, 7 differ | identical; identical | same 323,542 rows (sorted) | the 3 read files the kept copy lacks; `provenance.json` (the whitelist path `ingest/bin5/` vs `solowire/…`); `bytes_public`; timings in `read_stats.tsv`, `star_Log.std.out` |
| **gpuE5 vs E11's gpuE5** | **60 identical**, 2 timing/path only, 7 differ | identical; identical | **byte-identical in file order** | the same |
| **cpu vs E11's cpu** | **60 identical**, 2 timing/path only, 7 differ | identical; identical | same rows (sorted) | the same |

With the switch unset, the binary gives E11's outputs, which are E14's too. Two T = 8 gates compared the ingest path directly with
E11's kept level 5 (`k11gpuE5` vs `gpuE5i`): 796822 (593a1ec9) and 796991 (12c54287). Each found 60 identical files and 7
different ones. The 7 are the 3 files the kept copy lacks, the whitelist path, `bytes_public` and `n_reads`, and timings.

### 2c. Second sample: neuron1k_v3 (mouse, GRCm39-2024-A), T = 8 (job 797012, node g051)

| comparison | out_equiv | `star_SJ.out.tab`, `star_Log.final.out` | read files (sorted records) | `Gene/CellReads.stats` | what differs |
|---|---|---|---|---|---|
| **gpuE5 vs gpuE5i** | **60 identical**, 3 timing/path only, 6 differ | identical (117,254 junctions); mapping stats identical | same records: `nonhost_reads.fq` 490,605 reads (b4ddddd14b523255), `Unmapped.out.mate1/2` 3,667,707 reads (37b1365ae014d9ff, 7340516c7041abe0); order differs | **same 600,534 rows, byte-identical in file order** | record order of the 3 read files (on this sample two level-5 runs without ingest differ in order too, E14 §3b); `clip5p/probe/n_reads` 12,000 → 12,276 and `bytes_public`; timings in `read_stats.tsv`, `star_Log.std.out` |

On this sample the R11 histogram (89,319,948 reads, 575,742 barcodes) is identical between the two runs. The prefill's own
histogram matches the device's: 0 entries differ.

### 2d. Other checks

- **R11.** In all 8 runs of gate 796992, the histogram the pipeline's CB corrector used is byte-identical to E11's reference
  (`jobs/fable/records/results/r11_exact_hist.bin`, 4,835,008 bytes; 63,169,750 reads, 302,186 barcodes):
  - in the ingest runs, it came through the region;
  - in the others, from the pipeline's decode.

  In every ingest run of every gate, the prefill's R11 equals the device's E10 R11 (0 entries differ).
- **E10's entries** (796733, level 2 + ingest + `SINGLET_ENGINE_GPU_E10_CHECK=1`): STAR's own hand-off entries against the
  device's, for 66,444,956 reads with a record. 0 entries and 0 low-quality masks differ.
- **E8's records** (796822, level 1 + ingest + `SINGLET_ENGINE_GPU_E2E_CHECK=1`): the device-gathered records equal the
  host-assembled records byte for byte.
- **T = 8 on pbmc1k_v3** (797150 and 796991, the final binary):
  - gpuE5 vs gpuE5i, gpuE5i vs gpuE5s and gpuE5i vs gpuE5n: 63 identical each, the other 3 differ in timings, `bytes_public` and
    `n_reads`;
  - gpuE5i vs E11's kept gpuE5: 60 identical, as in §2b.
- **Before E14** (796849, T = 18, 593a1ec9):
  - cpu vs gpuE5i and cpu vs gpuE5: 60 identical each;
  - gpuE5 vs gpuE5i and gpuE5i vs gpuE5ib: 63 identical each;
  - the three runs without the switch vs E11's kept outputs: 60 identical each.

### 2e. After merging E12 (gate 797219, T = 18, node g052, binary 1c25e24e `bin6/`)

`origin/gpu/fable-engine` moved during this experiment: E12 ("prefill", d03d6c47) was merged into it. This branch merged it at
1c25e24e (§7) and re-ran the gate.

| comparison | out_equiv | `star_SJ.out.tab`, `star_Log.final.out` | read files (sorted records) | `Gene/CellReads.stats` | what differs |
|---|---|---|---|---|---|
| **cpu vs gpuE5i** | **60 identical**, 2 timing/path only, 7 differ | identical (126,996 junctions); identical | same records (c115c73474e99993, 766e191fcc6181a5, 31fc9ef1d3d55a89); order differs | same 323,542 rows (sorted) | E11's 7, as in §2a |
| **gpuE5 vs gpuE5i** | **63 identical**, 3 timing/path only, 3 differ | identical; identical | byte-identical | **byte-identical in file order** | `n_reads` 12,000 → 12,194, `bytes_public`, timings |
| **gpuE5i vs gpuE5ib** | **63 identical**, 3 timing/path only, 3 differ | identical; identical | byte-identical | byte-identical in file order | `bytes_public`, timings |
| **gpuE5i vs gpuE5s** | **63 identical**, 3 timing/path only, 3 differ | identical; identical | byte-identical | byte-identical in file order | `bytes_public`, timings |
| **gpuD, gpuE5, cpu vs E11's kept outputs** | **60 identical** each, 2 timing/path only, 7 differ | identical; identical | – | same rows (gpuE5: byte-identical in file order) | the 3 files the kept copy lacks, the whitelist path (`ingest/bin6/`), `bytes_public`, timings |

R11 is byte-identical to E11's reference in all 7 runs. So E12's engine and E13's ingest together keep every output that §2a
checked.

## 3. Timings

### 3a. pbmc1k_v3, T = 18, node g051 (gate 796992; one job, binary 12c54287, E14 merged)

| run | switches over `--engine gpu`, `SINGLET_GPU_PILEUP=on` | wall s | core-s | wall s/M | core-s/M | maxRSS |
|---|---|---|---|---|---|---|
| gpuE5 | `SINGLET_ENGINE_GPU_E2E=5` (the baseline) | 175.7 | 667.4 | 2.64 | 10.02 | 61.0 GB |
| gpuE5b | the same, second run | 176.1 | 663.9 | 2.64 | 9.97 | 61.0 GB |
| **gpuE5i** | **+ `SINGLET_ENGINE_GPU_INGEST=1`** | **148.3** | **565.0** | **2.23** | **8.48** | **45.8 GB** |
| **gpuE5ib** | the same, second run | **149.9** | **563.9** | **2.25** | **8.47** | **45.8 GB** |
| gpuE5s | + `SINGLET_ENGINE_GPU_INGEST_EARLY=0` | 153.2 | 565.9 | 2.30 | 8.50 | 45.8 GB |
| gpuE5n | + `SINGLET_ENGINE_GPU_INGEST_CLIPDEDUP=0` | 152.9 | 626.4 | 2.30 | 9.40 | 45.8 GB |
| gpuD | (default GPU path) | 219.9 | 985.9 | 3.30 | 14.80 | 43.5 GB |
| cpu | CPU engine | 400.4 | 2,496.2 | 6.01 | 37.48 | 29.7 GB |

**Ingest against the baseline** (means of the two runs: 149.1 s / 564.4 core-s against 175.9 s / 665.6):
- **−26.8 s (−15.2 %) and −101.2 core-s (−15.2 %).** The two runs of each mode agree within 1.6 s and 3.5 core-s.
- Level 5 with ingest is 2.70 times faster than the CPU engine of the same job (2.28 times without ingest).

**The variants:**
- **Running A, B and C early**, before the prefill, is worth −4.1 s at the same core-s (gpuE5s against gpuE5i).
- **The dedup** (the clip once per distinct sequence) is worth −3.8 s and −62 core-s (gpuE5n against gpuE5i).

**maxRSS −15.2 GB.** The peak process is STAR's:
- **Without ingest,** its `readSequences` held three things at once: the mapped R2 FASTQ, per-thread sequence copies and the merged
  `seq`.
- **With ingest,** STAR's process holds:
  - `seq` once;
  - the quality as 2-bit codes (1.5 GB);
  - the barcode read as 3-bit codes (1.07 GB).

**Against E11's level 5** (gate 796433 on node g050: 210.3–210.8 s, 678–680 core-s), E14 and E13 together are −61.5 s (−29.2 %) and
−115 core-s.

### 3b. Where the time goes

`bench/fable/ingest/s8_stages_e13.py` splits the run into stages by the logs' second stamps. Each cell gives wall s / core-s; the
E13 rows come from `star_Log.out`.

| stage | gpuE5 | gpuE5b | **gpuE5i** | **gpuE5ib** | gpuE5s | gpuE5n |
|---|---|---|---|---|---|---|
| startup | 1 / 3.8 | 1 / 5.0 | 1 / 4.8 | 1 / 4.3 | 0 / 0.9 | 1 / 3.3 |
| decode .1fq → FASTQ + R11; with ingest the head scan | 20 / 90.6 | 20 / 89.7 | **2 / 18.6** | **3 / 29.4** | 3 / 28.4 | 3 / 29.5 |
| probe/ref load | 4 / 6.7 | 4 / 6.8 | 7 / 66.7 | 6 / 56.3 | 4 / 7.4 | 6 / 56.9 |
| STAR load, CR4 probe, start (before the prefill) | 3 / 7.5 | 4 / 16.6 | 5 / 75.7 | 6 / 90.4 | 4 / 9.6 | 8 / 128.8 |
| GPU prefill (read, clip, seed, stitch, cache) | 54 / 310.0 | 54 / 301.1 | **42 / 153.7** | **42 / 139.9** | 52 / 279.9 | 44 / 164.3 |
| … reading reads / GPU (whole s) | 12 / 38 | 12 / 38 | **0** / 38 | **0** / 38 | 9 / 39 | 2 / 38 |
| … E13 A: decode, s (where) | – | – | 3.36 (pipeline) | 3.34 (pipeline) | 3.65 (prefill) | 3.37 (pipeline) |
| … E13 B: barcodes, s | – | – | 1.52 | 1.58 | 1.12 | 1.17 |
| … E13 C: clip, s (thread-s) | – | – | 7.51 (88.2) | 7.77 (86.6) | 5.01 (69.3) | 11.54 (167.6) |
| … the prefill waited for them, s | – | – | 0.30 | 0.28 | 9.78 (all in it) | 2.40 |
| STAR per-read host pass | 9 / 20.2 | 8 / 19.1 | 10 / 24.3 | 10 / 23.5 | 10 / 22.3 | 9 / 22.1 |
| STAR stream into pileup | 9 / 17.2 | 9 / 17.2 | 8 / 15.7 | 8 / 15.5 | 8 / 15.7 | 9 / 17.2 |
| GPU-pileup hand-off (with GPU pileup + collapse) | 17 / 32.4 | 17 / 26.4 | 16 / 28.0 | 17 / 36.3 | 16 / 29.4 | 17 / 34.6 |
| non-host | 19 / 43.4 | 21 / 51.8 | 18 / 46.2 | 18 / 40.6 | 17 / 43.2 | 17 / 39.2 |
| export, QC, callers | 18 / 36.8 | 18 / 39.1 | 18 / 36.6 | 18 / 37.7 | 18 / 37.8 | 18 / 38.2 |
| demux/ancestry/writers | 21 / 97.9 | 19 / 89.9 | 21 / 94.3 | 21 / 89.3 | 21 / 91.2 | 21 / 91.5 |
| **total** | 175 / 666.6 | 175 / 662.7 | **148 / 564.5** | **149 / 563.3** | 153 / 565.8 | 152 / 625.6 |

**What moved, with ingest:**
- **The pipeline's decode stage is the head scan**: 20 s / 90 core-s become 2–3 s / 19–29 core-s, the scan itself 2.6–2.8 s on 18
  threads.
- **Phase A (the decode, 3.4 s, about 60 core-s) runs in the pipeline during the reference load.** The "probe/ref load" stage grows
  from 4 to 6–7 s, because the load shares the cores with it. The STAR fork waited 0 s for the decode.
- **Phase C (the clip, 87–88 thread-s) runs in STAR's process beside E14's two CR4 probes.** That stage grows from 3–4 to 5–6 s.
  Phase B (1.5 s) follows STAR's parameters.
- **The prefill waited 0.3 s for B and C.** Its "reading reads" is 0 s instead of 12, and the prefill is 42 s / 140–154 core-s
  instead of 54 s / 301–310. Its GPU part is unchanged (38 s).
- **Net, in wall s:** −17.5 (decode stage) +2.5 (reference load) +2 (STAR start) −12 (prefill) = −25 s; the rest is −2 s of noise.
- **Net, in core-s** (means): −66 (decode stage) +55 (reference load) +71 (STAR start) −159 (prefill) = −99.

**The variants:**
- **gpuE5s** (`_INGEST_EARLY=0`) pays all three phases inside the prefill: 9.8 s, with the clip at 69 thread-s, uncontended. The
  run is 4 s slower and uses the same CPU.
- **gpuE5n** (no dedup) clips all 66.6 M reads: 167.6 thread-s and 11.5 s. The prefill then waits 2.4 s.

### 3c. pbmc1k_v3, T = 8 (job 797150, node g052; debug-gpu, 8 CPUs)

| run | wall s | core-s | maxRSS | decode stage | ref load | STAR start | prefill | reading reads | the prefill waited |
|---|---|---|---|---|---|---|---|---|---|
| gpuE5 | 207.0 | 657.2 | 60.6 GB | 28 / 117.3 | 5 / 9.1 | 3 / 8.8 | 73 / 307.3 | 25 s | – |
| **gpuE5i** | **176.1** | **552.8** | 45.4 GB | 5 / 27.6 | 15 / 74.0 | 9 / 57.8 | 54 / 177.3 | 6 s | 5.2 s |
| gpuE5s | 173.5 | 537.3 | 45.3 GB | 4 / 24.5 | 4 / 7.5 | 5 / 17.7 | 68 / 271.8 | 21 s | 21.0 s (all in it) |

- **Ingest at T = 8: −31.0 s (−15.0 %) and −104 core-s (−15.9 %).** The sync variant gives −33.5 s and −120 core-s. One run
  each.
- **At T = 8 the early path buys nothing.** The host work before the prefill is CPU-bound:
  - the pipeline's decode takes 8.5 s on 8 threads;
  - E14's genome preload needs the same cores, so the reference load stage grows to 15 s;
  - in STAR's process the clip (15.6 s, 91 thread-s) shares the cores with the two probes.
- **The early path and the sync path end up within the noise** (29 + 54 s against 13 + 68 s).
- **Node variance is about 9 s.** On g051 (796991, the same binary), gpuE5i took 167.2 s / 524.8 core-s. gpuE5n took 172.6 s /
  587.7 there: without the dedup the clip is 169.5 thread-s against 90.8, +5.4 s and +63 core-s.

### 3d. neuron1k_v3 (mouse), T = 8 (job 797012, node g051)

| run | wall s | core-s | wall s/M | core-s/M | maxRSS | decode stage | ref load | STAR start | prefill (reading reads) |
|---|---|---|---|---|---|---|---|---|---|
| gpuE5 | 246.6 | 721.3 | 2.65 | 7.76 | 82.8 GB | 32 / 114.6 | 3 / 4.5 | 2 / 3.5 | 84 / 354.1 (35 s) |
| **gpuE5i** | **196.6** | **565.5** | **2.12** | **6.09** | **60.4 GB** | 5 / 32.4 | 10 / 75.5 | 6 / 40.8 | 56 / 176.7 (8 s) |

- **−50.0 s (−20.3 %), −155.8 core-s (−21.6 %), maxRSS −22.4 GB.** The sample has 92.9 M reads, 34,833,715 distinct sequences.
- **The decode** (9.1 s on 8 threads) was longer than the reference load it overlapped, so the STAR fork waited 4.9 s for it.
- **The prefill waited 8.6 s** for the barcode pass (3.6 s) and the clip.

### 3e. Before E14 (the first designs)

- **Gate 796849** (T = 18, g051, 593a1ec9; the decode and the clip in the prefill, no dedup):
  - gpuE5 took 210.2 / 210.3 s (689.4 / 688.5 core-s);
  - gpuE5i took 193.2 / 193.3 s (643.9 / 639.8 core-s);
  - that is **−17.0 s (−8.1 %), −47 core-s**. The decode inside the prefill took 12.5 s, about the same as the FASTQ read-in it
    replaced (§4).
- **796850** (neuron1k_v3, T = 8): 292.1 → 259.3 s (−32.8 s), 761.8 → 693.3 core-s.
- **796870** (T = 8, 3a5cad85; the decode in STAR's process during the serial CR4 probe): gpuE5i took 199.3 s against 209.7 s for
  the sync variant (−10.4 s), at the same core-s. E14 then removed most of that window (§4).

### 3f. After merging E12 (gate 797219, T = 18, node g052, binary 1c25e24e; one job)

| run | wall s | core-s | wall s/M | core-s/M | maxRSS |
|---|---|---|---|---|---|
| gpuE5 (level 5: E12 + E14) | 153.8 | 653.8 | 2.31 | 9.82 | 61.3 GB |
| gpuE5b | 153.3 | 657.4 | 2.30 | 9.87 | 61.3 GB |
| **gpuE5i (+ ingest)** | **127.9** | **559.8** | **1.92** | **8.41** | **46.1 GB** |
| **gpuE5ib** | **127.9** | **561.3** | **1.92** | **8.43** | **46.0 GB** |
| gpuE5s (ingest, all in the prefill) | 132.2 | 566.7 | 1.98 | 8.51 | 46.1 GB |
| gpuD | 198.5 | 967.1 | 2.98 | 14.52 | 43.5 GB |
| cpu | 395.0 | 2,482.4 | 5.93 | 37.27 | 29.6 GB |

**Ingest on top of E12** (means of two runs: 127.9 s / 560.5 core-s against 153.6 s / 655.6):
- **−25.6 s (−16.7 %) and −95.1 core-s (−14.5 %)**, about the same as without E12 (−26.8 s, −101 core-s). The two experiments
  add up.
- The early path is worth 4.3 s (gpuE5s).
- Level 5 with ingest is 3.09 times faster than the CPU engine of the same job.
- Against E11's level 5 (210.6 s on g050), E12, E13 and E14 together save −82.6 s (−39 %).

**Stages** (wall s / core-s, as in §3b):

| stage | gpuE5 | gpuE5b | **gpuE5i** | **gpuE5ib** | gpuE5s |
|---|---|---|---|---|---|
| decode .1fq → FASTQ + R11; with ingest the head scan | 21 / 91.6 | 19 / 86.7 | **3 / 23.0** | **3 / 22.4** | 3 / 24.0 |
| probe/ref load | 4 / 6.5 | 5 / 7.8 | 7 / 65.4 | 7 / 65.7 | 5 / 12.6 |
| STAR load, CR4 probe, start (before the prefill) | 6 / 20.7 | 4 / 19.1 | 7 / 103.2 | 7 / 104.0 | 4 / 18.5 |
| GPU prefill | 30 / 294.8 | 31 / 295.3 | **20 / 126.2** | **20 / 125.6** | 29 / 269.2 |
| … reading reads / GPU (whole s) | 12 / 16 | 12 / 17 | **0** / 17 | **0** / 17 | 10 / 16 |
| … E13 B, C (s); the prefill waited (s) | – | – | 1.6, 7.9; 0.3 | 1.7, 7.8; 0.3 | 1.1, 5.0; 9.8 (all in it) |
| STAR per-read host pass | 8 / 19.4 | 9 / 20.6 | 9 / 22.4 | 9 / 22.5 | 9 / 20.5 |
| STAR stream into pileup | 9 / 16.8 | 8 / 15.3 | 9 / 17.1 | 8 / 15.8 | 9 / 17.2 |
| GPU-pileup hand-off (with GPU pileup + collapse) | 17 / 30.3 | 18 / 35.3 | 17 / 33.3 | 17 / 27.4 | 17 / 35.0 |
| non-host | 20 / 43.0 | 20 / 42.6 | 17 / 40.5 | 18 / 47.6 | 17 / 39.2 |
| export, QC, callers | 18 / 36.7 | 18 / 37.9 | 18 / 36.7 | 17 / 34.9 | 18 / 38.6 |
| demux/ancestry/writers | 20 / 90.9 | 20 / 90.2 | 21 / 91.8 | 21 / 94.4 | 20 / 90.9 |
| **total** (with startup) | 154 / 653.5 | 153 / 656.6 | **128 / 559.6** | **127 / 560.3** | 131 / 565.7 |

**What changed with E12's engine:**
- **The prefill is 20 s with ingest**: 0 s of reading reads, 17 s of GPU and 3 s of bookkeeping. It was 30–31 s without ingest.
- **The CellRanger4 clip (7.8 s) now sets the length of the "STAR start" stage** (7 s, against 4–6 s without ingest). It runs in
  STAR's process from the fork, and the prefill waits 0.3 s for it. It is the next host step on the path to the prefill (§5).

## 4. Surprises

- **The CellRanger4 clip, not the decode, is the prefill's read-in.** At T = 18 the decode takes about 3.5 s and the barcode pass
  about 1.5 s. STAR's CellRanger4 clip (opal's alignment of the TSO against every read) costs 138 thread-s on pbmc1k_v3 and about
  195 on neuron1k_v3: 7.7 s at T = 18 and 18–20 s at T = 8. In the FASTQ path it was hidden inside "reading reads". Running it once
  per distinct sequence and before the prefill is most of this experiment's prefill saving.
- **A race only the gate could see.** The first version clipped each block from its decode callback and read `off[r0]`, the previous
  block's last offset, which another thread was still writing. Because the arrays are `NoInitVector` (no zero-fill to hide the
  race), 1–4 of 66.6 M reads got a wrong clip length in some runs, and out_equiv showed 13 differing files (796733, e206dc01). The fix
  passes the block's first base offset (593a1ec9); the current code clips after the decode.
- **The FASTQ path's "reading reads" was 12–13 s in these jobs, not 16 s.** E11's 16 s came from gate 796433. Run inside the
  prefill, the decode saves nothing over the FASTQ read-in (12.5 s against 12–13 s, gate 796849). The gain comes from:
  - removing the pipeline's FASTQ decode;
  - moving the decode and the clip out of the prefill;
  - clipping each distinct sequence once.
- **The 5' probe sample is 12,194 reads with ingest, as on the CPU engine.** The head scan takes the probe sample per block, as FIFO
  pass 1 does. So `pileup_stats.json`'s `/provenance/clip5p/probe/n_reads` is 12,194 with ingest and on the CPU engine, against
  12,000 on the file path (gpuD, gpuE5). The 5' clip decided from it is 0 bp either way. On neuron1k_v3 the figures are 12,276
  against 12,000.
- **E14 moved the window.** The first early design (3a5cad85) decoded in STAR's process during the CR4 probe, which was 31 s of
  mostly idle CPU at T = 18. E14 made the probe stage 3–5 s. The decode therefore moved into the pipeline's process, beside its
  reference load, and STAR's process now only clips and runs the barcode pass before its prefill.
- **At T = 8 the host work before the prefill is CPU-bound.**
  - The pipeline's decode (8.5–9 s) competes with E14's genome preload for the same 8 cores. On pbmc1k_v3 the reference load stage
    grew from 4–5 to 15 s.
  - On neuron1k_v3 the decode outlasted the load, and the fork waited 4.9 s for it.
  - The clip in STAR's process shares the cores with the two CR4 probes.
  - The prefill still waits 5–9 s.

  So at T = 8, running the phases early and running them in the prefill end within the noise of each other (§3c). At T = 18 the
  early path is worth 4 s.

## 5. What remains

- **Default on.** The switch is off by default, as the brief asked. The gates here are the case for turning it on with level 5.
  Everything it needs is checked at run time, and every failed condition falls back to the FASTQ files with the reason in the log.
- **Inputs the head scan does not take still go through the FASTQ files:**
  - per-base quality stored in full;
  - duplicate counts;
  - no barcode dictionary;
  - a barcode read over 63 bases;
  - a constant 5' prefix to clip (the decode could apply it);
  - CB_UMI_Complex, barcodes in R2, non-droplet assays.
- **The CellRanger4 clip is still the largest host cost before the prefill**: about 70–90 thread-s after the dedup on pbmc1k_v3. With E12 merged it sets the length of the stage before the prefill (§3f).
  The device could run it: it is a small local alignment of one 30-base TSO against each distinct 91-base read, on reads the device
  already holds. That would also bring the prefill's remaining wait to zero at any thread count.
- **The early phases at low thread counts.** At T = 8 they only move CPU work in front of the prefill (§3c). Two options:
  - start the decode after the reference load rather than beside it;
  - or fork STAR before the reference load, so the decode, the load and the probes are not all on the same cores at once.

  Either is a scheduling change in `singlet.cpp`. The switch `SINGLET_ENGINE_GPU_INGEST_EARLY=0` already gives the sync order.
- **The barcode pass (B) waits for STAR's parameters** only for the barcode layout and the whitelist, both of which the pipeline
  itself gives STAR. Computing B in the pipeline, with the whitelist it already loads for barcode discovery, would leave STAR's process
  nothing to do before its prefill except the clip.
- **The head scan (about 20 core-s) and the decode both decompress the barcode column.** With the decode now in the pipeline's process
  right after the scan, the two could be a single pass, at the price of barcode discovery waiting for the whole decode.
- **`.1fg2`** (E12): `PackedReads`'s header comment is the mapping; nothing is implemented.

## 6. Files touched outside `src/engine_gpu/`, and why

All of E13 is outside `src/engine_gpu/`. In the engine's interface (`include/singlet/engine_gpu.h`) only `Input::seq/off` changed (to
`NoInitVector`), plus the R11 fields in the region header `E2EHeader`; the engine itself is untouched.

| file | change | why |
|---|---|---|
| `include/singlet/engine_gpu_ingest.h` (new) | `PackedReads` (the struct of arrays, with the `.1fg2` mapping), `Plan`, the decode API (`decodeReads`, `barcodes`, `startParent`/`joinParent`/`takeParent`/`releaseParent`, `startEarly`), `publishR11`/`fetchR11` | the interface between the pipeline, the decode and the prefill; plain C++ (no CUDA, STAR or .1fq headers) |
| `src/star/SeqCacheGpuIngest.cpp` (new) | phases A and B; the pipeline-side decode thread | the .1fq decoder (`read_source.h`, zstd) cannot share a translation unit with STAR's `IncludeDefine.h` |
| `src/star/SeqCacheGpuPrefill.cpp` | `ingestReads` in place of `readSequences`/`e10Barcodes` when the plan is on; `cr4Clip` (with the dedup); `startEarly`, `seqCacheGpuEarly`; `ingRecord` (Unmapped.out and the read source from the packed arrays); `publishR11` and the decode-vs-device R11 check; E11's RNA Q30 from the decode's counts | in scope (the prefill) |
| `src/star/STAR.cpp` | `seqCacheGpuEarly(P)` after `Solo soloCellFilter`, before the genome load | phase B needs STAR's parameters and whitelist; one call, nothing unless the switch is on |
| `src/pipeline/singlet.cpp` | the head scan and the plan in place of the file decode; `startParent` after the scan, `joinParent` before and `releaseParent` after the STAR fork; `startEarly` in the STAR child; the CR4 probes and the fallback chunk source read the .1fq; R11 through `set_pending`; the merge with E14's concurrent probes | the pipeline's side of the switch |
| `include/singlet/fq/read_source.h` | `RunSet::scan_head`, `HeadScanOptions`, `BlockHead`, `head_scan_supported` | the head scan (barcode-dictionary, UMI and length columns only) |
| `include/singlet/pileup/cb_correct.h` | `set_pending` / `settle` (`std::call_once`) | R11's lazy fetch (the handshake) |
| `include/singlet/engine_gpu.h` | `E2EHeader` R11 fields (`r11Magic`, `oR11`, `nR11`, `r11NWL`, `r11Total`); `Input::seq/off` as `NoInitVector` | the handshake; the decode writes every element, so no serial zero-fill |
| `CMakeLists.txt` | zstd's include directory for `singlet_star` | the .1fq decoder in STAR's library |
| `bench/fable/ingest/jobs/{build,gate}.sbatch`, `bench/fable/ingest/s8_stages_e13.py` | build and gate jobs; the stage script with the head-scan markers and the E13 lines | deliverable |
| `docs/gpu-engine/fable/E13-ingest.md` | this report | deliverable |

## 7. Jobs, commits, how to run

**Branch `gpu/fable/ingest`** (pushed):

| commit | what |
|---|---|
| 8d3220e0 | the implementation: the head scan, the prefill's decode and R11's handshake (`SINGLET_ENGINE_GPU_INGEST=1`) |
| e206dc01 | the scan's log line; the build continues past errors; the gate picks the mouse genome for mouse samples |
| 593a1ec9 | the CR4 clip callback takes the block's first base offset (the race of §4); clip thread-seconds in the log |
| 3a5cad85 | the decode starts with STAR's process, beside the CR4 probe (the pre-E14 design); gate mode `gpuE5s` |
| ff18a1f9 | the CR4 clip right after the early decode, with STAR's fixed CellRanger4 ClipMates |
| 11a19707 | merge of `origin/gpu/fable-engine` (E14). One conflict, in the CR4 probe, resolved to E14's concurrent probes, which read the .1fq when ingest is on |
| 22761612 | the pipeline decodes before the STAR fork (`startParent`/`joinParent`/`takeParent`/`releaseParent`); the clip runs once per distinct sequence |
| 12c54287 | gate mode `gpuE5n` (the binary of gate 796992, `bin5/`) |
| aaa01ab9 | header comment |
| 183ebf4e | a comment in STAR.cpp |
| 1c25e24e | merge of `origin/gpu/fable-engine` (E12, d03d6c47). One conflict, at the end of `seqCacheGpuPrefill`: E12's timing marks, then E13's release of the packed reads (`bin6/`) |
| (next) | this report |

**Builds** (short-gpu without a GPU, 263–341 s each; the binary is `$W/jobs/fable/ingest/<dir>/singlet-gpu`):

| job | commit | dir |
|---|---|---|
| 796709 | 8d3220e0 | failed (a heredoc broke a string literal) |
| 796716 | e206dc01 | `bin/` |
| 796821 | 593a1ec9 | `bin/` |
| 796868 | 3a5cad85 | `bin2/` |
| 796911 | ff18a1f9 | `bin3/` |
| 796942 | 11a19707 | `bin4/` |
| 796971 | 12c54287 | `bin5/` |
| 797218 | 1c25e24e | `bin6/` (E12 merged) |

**Gates and checks** (H100 NVL; results in `$W/jobs/fable/ingest/results/gate-<job>/`):

| job | partition, T | binary | sample: modes | use |
|---|---|---|---|---|
| 796733 | debug-gpu, 8 | e206dc01 | pbmc1k_v3: gpuE5i gpuE2i | E10's entry check with ingest (0 entries, 0 masks differ); found the clip race |
| 796822 | debug-gpu, 8 | 593a1ec9 | pbmc1k_v3: gpuE5i gpuEci | the race fixed; E8's device-vs-host record check with ingest |
| 796849 | short-gpu, 18 | 593a1ec9 | pbmc1k_v3: gpuE5 gpuE5i gpuE5b gpuE5ib gpuD cpu | the first T = 18 gate (the decode in the prefill), before E14 |
| 796850 | debug-gpu, 8 | 593a1ec9 | neuron1k_v3: gpuE5 gpuE5i | the second sample, before E14 |
| 796870 | debug-gpu, 8 | 3a5cad85 | pbmc1k_v3: gpuE5i gpuE5s | the early decode in STAR's process, before E14 |
| 796991 | debug-gpu, 8 | 12c54287 | pbmc1k_v3: gpuE5i gpuE5n | the final binary against E11's kept gpuE5; the dedup against no dedup |
| **796992** | **short-gpu, 18** | **12c54287** | **pbmc1k_v3: gpuE5 gpuE5i gpuE5s gpuE5n gpuE5b gpuE5ib gpuD cpu** | **gate (a) and the timings of §3** |
| 797012 | debug-gpu, 8 | 12c54287 | neuron1k_v3: gpuE5 gpuE5i | the second sample, final binary |
| 797150 | debug-gpu, 8 | 12c54287 | pbmc1k_v3: gpuE5 gpuE5i gpuE5s | T = 8 in one job |
| 797219 | short-gpu, 18 | 1c25e24e | pbmc1k_v3: gpuE5 gpuE5i gpuE5s gpuE5b gpuE5ib gpuD cpu | the re-gate after merging E12 (§2e, §3f) |

Logs are in `$W/logs/e13-{build,gate,dbg*}-<job>.out`. Each result directory holds `summary.txt` and `equiv-*.txt`. Each run also
has `stages.txt`, `starlog_engine.txt`, `star_Log.out`, `run.log.ts`, `CellReads.stats`, `r11.bin`, `monitor.tsv`, `mem.txt` and
`out/`.

**Run:**
- Build: `sbatch --export=ALL,OUTD=<dir> bench/fable/ingest/jobs/build.sbatch`.
- Gate: `MODES="cpu gpuE5 gpuE5i" PAIRS="cpu:gpuE5i gpuE5:gpuE5i" BINDIR=<dir> sbatch bench/fable/ingest/jobs/gate.sbatch`.
  - For the second sample, add `SAMPLE=neuron1k_v3 READSM=92.902231`.
  - A pair member that is not run in the job is read from `OUT_<name>` (an `out/` dir) and `CRS_<name>` (a kept
    `CellReads.stats`). The comparisons with E11 used `OUT_k11gpuE5=$W/jobs/fable/solowire/results/gate-796377/gpuE5/out`,
    `CRS_k11gpuE5=…/gpuE5/CellReads.stats`, and the same for `gpuD` and `cpu`.

**Switches** (they apply with `--engine gpu`, `SINGLET_GPU_PILEUP=on` and `SINGLET_ENGINE_GPU_E2E` set):
- `SINGLET_ENGINE_GPU_INGEST=1` turns ingest on.
- `SINGLET_ENGINE_GPU_INGEST_EARLY=0` makes the prefill run all three phases.
- `SINGLET_ENGINE_GPU_INGEST_CLIPDEDUP=0` clips every read.

**The gate's modes:**

| mode | what it runs |
|---|---|
| `cpu` | the CPU engine |
| `gpuD` | `--engine gpu` + `SINGLET_GPU_PILEUP=on` (the default GPU path) |
| `gpuE5` | `gpuD` + `SINGLET_ENGINE_GPU_E2E=5` (the baseline) |
| `gpuE5i` | `gpuE5` + `SINGLET_ENGINE_GPU_INGEST=1` |
| `gpuE5s` | `gpuE5i` + `_INGEST_EARLY=0` |
| `gpuE5n` | `gpuE5i` + `_INGEST_CLIPDEDUP=0` |
| `gpuE2i` | level 2 + ingest + E10's entry check |
| `gpuEci` | level 1 + ingest + E8's record check |

A `b` suffix marks a second run of the same mode.

**What the logs say:**
- The pipeline logs one of:
  - `[1fq-ingest] Scanned …`;
  - `[1fq-ingest] N reads decoded into the packed buffers …`;
  - `[1fq-ingest] not used (<why>)`.
- `star_Log.out` logs `ENGINE_GPU: E13: …`, with:
  - the phase times;
  - the clip's distinct sequences;
  - the prefill's wait;
  - the decode-vs-device R11 check.
