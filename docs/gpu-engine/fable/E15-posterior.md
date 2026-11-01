# E15: the CB posterior per read at level 5 (posterior)

Experiment E15 ("posterior") of the GPU aligner program. Branch `gpu/fable/posterior`, from `gpu/fable-engine` at 8e54be40. Goal:
make level 5 (`SINGLET_ENGINE_GPU_E2E=5`, E11's STARsolo on the device) general enough for the atlas, so that it runs, byte-identical
to the CPU engine, on samples with binned NovaSeq barcode qualities and on a 5' sample instead of falling back to level 3.

With the variable unset or set to 1–4 the binary runs the code it ran before: the changes are reached only at level 5 (by
construction; the default paths were not gated separately here).

## 0. Answer in brief

- **Binned barcode qualities never blocked level 5 in the pipeline (§1b).**
  - `.1fq` stores the qualities of R2 only, and every decode writes 'F' for R1. STAR therefore sees CB qualities 70..70 on every
    sample, and E11's constant-quality guard never fires in a pipeline run.
  - With E11's own binary, level 5 already ran on pbmc2m, pbmc5p_v1 and neuron1k_v3 (probe 796560). The original R1 FASTQs are
    binned NovaSeq ('F', ':', ',', '#'), but those qualities do not reach STAR.
- **The per-read posterior is ported anyway, exact, and checked against STAR on real binned qualities (§1c–1f, §2a).**
  - A plain STAR from the vendored sources ran on pbmc1k_v3's original L001 FASTQs: 33,436,697 reads, 5.9 % of the CB bases not 'F',
    626,729 reads with several whitelist matches.
  - E9's library with the reads' CB qualities, on the host and on the device:
    - every read's cbMatch, first match, Solo flag and posterior CB are STAR's;
    - **10 of 10 Solo files are byte-identical to STAR's**, including CellReads.stats.
  - The control, one 'F' quality for every base (what the pipeline does), gives 20,492 other decisions and 5 of 10 files differ.
  - In the engine, level 5 takes the per-read path by itself when the decode's CB qualities vary. It reads the multi-match reads'
    qualities from the mapped barcode FASTQ after STAR's loop, and checks each line's CR against the read's.
- **Gates at T = 18, CPU engine vs level 5 (§2b, §2c), all with 10 of 10 STAR Solo files identical and the same CellReads.stats rows:**

  | sample | reads | out_equiv identical / timing-path / differ | CellReads.stats rows |
  |---|---|---|---|
  | neuron1k_v3 (3' v3, mouse) | 92.9 M | 60 / 2 / 7 | 600,534 |
  | **pbmc5p_v1 (5' v1)** | 60.0 M | 58 / 2 / 7 | 355,639 |
  | nuclei2k_v2 (v2, mouse nuclei) | 118.7 M | 60 / 2 / 7 | 411,195 |
  | pbmc1k_v3 (against E11's kept CPU outputs) | 66.6 M | 60 / 1 / 8 | 323,542 |

  - The differing files are E11's run-dependent set: record order (3 files, same records), timings (2), `pileup_stats.json`'s probe
    count and `summary.json`'s memory tier. For pbmc1k_v3 they are the 3 files the kept copy lacks, plus the whitelist path in
    `provenance.json`.
  - The exception is nuclei2k_v2's `nonhost_reads.fq`, which holds different records. That is a pipeline bug in the non-host writer
    when it subsamples (§2c), not level 5.
  - The per-read path forced on (`SINGLET_E15_PERREAD=1`) gives the same verdicts on neuron1k_v3, pbmc5p_v1 and pbmc1k_v3.
- **Timings at T = 18 (§3):**

  | sample | cpu | gpuD | **level 5** | vs gpuD |
  |---|---|---|---|---|
  | neuron1k_v3 | 520.4 s / 2180 core-s | 326.7 / 1206 | **271.3 / 801** | −17.0 % wall, −33.6 % core-s |
  | pbmc5p_v1 | 489.8 / 2164 | 229.5 / 846 | **199.2 / 579** | −13.2 %, −31.6 % |
  | nuclei2k_v2 | 611.2 / 4712 | 459.2 / 1751 | **391.0 / 1179** | −14.9 %, −32.6 % |
  | pbmc1k_v3 (cpu, gpuD: E11) | 424.5 / 2607 | 253.4 / 997 | **212.7 / 686** | −16.1 %, −31.2 % |

  Against the CPU engine level 5 is 1.6–2.5x faster in wall time and uses 2.7–4.0x fewer core-s.
- **What still falls back: hgmm1k_v3 (human + mouse barnyard), for two reasons outside Solo (§2c).**
  - GPU mapping runs out of device memory in lane A (`cudaMalloc(&dWork, …)`) at every GPU level, also with lane A at a half and
    a quarter of its size. The run is then level 0: STAR maps every read.
  - With smaller lanes (`NB` 32,768, `NBB` 16,384, chunk 8 M) the mapping fits (lanes peak 85.9 GB). Then E10 refuses: it packs
    junction starts into 32 bits, and the barnyard genome has about 5.8 G bases. Without E10's products, levels 3–5 are not made.
  - The ports it needs: a device memory budget in `engine_gpu.cu`'s lanes (§5.1) and 64-bit junction starts in E10 (§5.8).
  - Level 5 also needs about 11 GB more device memory than the default GPU path while it maps (nuclei2k_v2 peaks at 89.8 of
    94 GB).
- **E11 §5.3 is in:** the kept Solo inputs (1.35–3.0 GB) sit in page-locked host memory during STAR's loop. It costs about 0.4 s on the
  tail (copy back and `cudaFreeHost`), which is within run-to-run noise; `SINGLET_E15_KEEP_ON_DEVICE=1` restores E11's behaviour.
- **A side result for the atlas (§2d):** the pipeline's 'F' qualities change 3.3 % of the multi-match decisions against STARsolo run
  on the FASTQ (pbmc1k_v3 L001: 19,476 fewer reads with a CB, same 1,135 cells). This holds for the CPU engine and every GPU level.
- **`gpu/fable-engine` moved during E15** (E14, at e2b0000e). It merged into this branch without conflicts (13651944) and was
  re-gated (797017, §2e). pbmc1k_v3 at level 5 against E11's CPU outputs is unchanged: 60 / 1 / 8, Solo files identical, same
  CellReads.stats rows. It took 178.2 s / 654.4 core-s with E14's genome load.

## 1. Rule and design

### 1a. STAR's rule

For a read whose CR matches several whitelist entries with one mismatch (`cbMatch > 1`, `1MM_multi_Nbase_pseudocounts`), STAR decides
the CB in its Solo step, per read, from text it wrote during mapping:

- `matchCBtoWL` (`SoloReadBarcode_getCBandUMI.cpp:46-73`) appends, for each match in its search order, `' ' + index + ' ' + q` to
  `cbMatchString`. `q` is the read's CB quality character at the mismatch: `cbQual1.at(cbL - 1 - ii)` for the 2-bit symbol `ii` in
  which the whitelist entry differs from the CR, or `cbQual1[posN]` for a CR with one N.
- `SoloReadFeature::inputRecords` (`SoloReadFeature_inputRecords.cpp:77-111`) reads the pairs back with `>> cbin >> qin` (`qin` is a
  `char`), then `qin -= QSbase; qin = min(qin, QSmax)` (33 and 33), `pin = cbReadCountTotal[cbin] * std::pow(10.0, -qin/10.0)` stored
  in a `float`, `ptot += pin`, and keeps the first strict maximum. The read gets that CB when `ptot > 0 && pmax >= cbMinP * ptot`
  (float, 0.975); otherwise it counts as `noTooManyWLmatches`.
- `cbReadCountTotal` is the number of reads whose CR is exactly that entry, whatever their UMI (`getCBandUMI.cpp:94`, `:296`), plus one
  (the pseudocount, `Solo.cpp:66-70`).

E9's port took one `pq` for the whole run and decided per distinct CR. That is exact only when every CB quality character gives the
same `qin` (every character ≥ QSbase + QSmax = 66, or one character for the whole run). E11's `e11Posterior` checked this and otherwise
fell back to level 3.

### 1b. Which qualities STAR sees in the pipeline

**The pipeline's STAR never sees the barcode read's own qualities.** `.1fq` is the only production input (`singlet.cpp:10`) and it
stores the qualities of R2 only. Every path that turns it back into FASTQ for STAR writes `'F'` for every R1 base:
`singlet.cpp:1220`, and `include/singlet/fq/read_source.h:319`, `:325`. So the CB quality characters are 70..70 for every sample, E11's
decode reports "CB quality characters 70..70 (posterior pq 0.000501187)", and E11's guard never fires in a pipeline run.

The other entry points do not reach STARsolo's `CB_UMI_Simple` either: the legacy `--reads` FASTQ path (and SRA) runs `soloType
CB_samTagOut` (`detected_soloType` is set only from a `.1fq` header, `singlet.cpp:4088`, `:4224`), and multi-segment chemistries
(`CB_UMI_Complex`) are turned into `CB_samTagOut` too (`singlet.cpp:4461-4490`).

Consequence for the atlas: the pipeline's STARsolo, on the CPU engine as on the GPU engine, decides multi-match reads with quality
'F' (Q37) at every mismatch, not with the sequencer's binned qualities. §2d measures what that changes against STAR run on the
original FASTQs.

### 1c. The port (`src/engine_gpu/solo.cuh`, `solo_gpu.cu`)

The per-read posterior was ported anyway: it is what STAR does, it costs nothing when every quality is the same, and it makes level 5
exact on any input that does carry R1 qualities (a future `.1fq` with them, or a decode that keeps them).

- `cbPosteriorRead` replaces `cbPosterior`. Per candidate, the quality position is `cbL - 1 - ctz64(wl[cand] ^ cbB1) / 2` (the symbol
  in which the entry differs from the CR; the candidates are 1-mismatch variants by construction), or `posN` for a CR with one N. The
  factor is `pqTab[q]`, a 256-entry table of `std::pow(10.0, -qin/10.0)` computed on the host with STAR's own `char` arithmetic. The
  device does only what is IEEE-exact on both sides: the double product, its conversion to float, float sums and compares, in STAR's
  candidate order with the strict `>`.
- `soloRun` gets a new stage after the CB stage ("multi"). It flags the reads with `cbMatch > 1` after the UMI check, compacts their
  indices (CUB scan, read order) and runs the posterior per read. Stage (c) (Solo's record/inputRecords rules, CellReads.stats, the
  keys) looks up a read's choice by binary search in that list. Reads with one match keep the per-distinct-CR choice.
- Without per-read qualities every candidate takes `S.pq` (E11's value). With them (`SoloReads::cbQual`, a `CbQualFn`), `soloRun`
  downloads the multi-match reads' indices and CRs, calls `cbQual` once on the host for their `cbL` quality characters, and uploads
  them with the table. Only those reads move (16 B each).
- `keepPerRead` adds `SoloResult::readCb` (the CB each read gets, counted or not) for the per-read comparison with STAR.

### 1d. Where the qualities come from in the engine (`src/star/SeqCacheGpuPrefill.cpp`)

- `e11Posterior` no longer refuses varying qualities. It sets `e11.perRead` when the decode's CB quality characters do not all give
  one `qin`, or when `SINGLET_E15_PERREAD=1` (the test switch: the per-read path on a run where one factor would do).
- With `perRead`, E10's mapped barcode FASTQ stays mapped through STAR's loop (it was released right after the loop before), and
  `e11CbQual` is passed to `soloRunKept` as `cbQual`. For read r it finds the record by E10's name-line offset, packs the CB of its
  sequence line exactly as the decode did, and fails unless that equals the CR `soloRun` has for the read (so the qualities are
  provably that read's). Then it copies the CB span of the quality line, padded with 'H' past the line's end as `getCBandUMI` pads it.
  It runs on `runThreadN` threads.
- New guards (level 3, with the reason in `star_Log.out`): CB quality characters ≤ ' ' (STAR's `operator>>` would skip them as
  whitespace), and `outQSconversionAdd != 0` (`readLoad` would change the qualities STAR sees after the decode read them).

### 1e. The kept inputs during STAR's loop (E11 §5.3)

After `mapDistinct`, `soloKeepToHost` starts a thread. It allocates page-locked host memory for the four kept arrays (CR/UR per read,
read → record map, per-record annotation), copies them and frees the device copies; STAR's loop runs meanwhile. `soloRunKept` joins
the thread and copies them back before `soloRun`. A failed step leaves them on the device and logs why. `SINGLET_E15_KEEP_ON_DEVICE=1`
keeps E11's behaviour.

### 1f. The check against STAR on binned qualities (`bench/fable/posterior`)

The pipeline cannot give STAR binned barcode qualities (§1b), so the check runs outside it:
- `e15star` (`star_main.cpp`, `jobs/starbuild.sbatch`) is a plain STAR built from this branch's vendored sources (CPU only). It runs on
  the sample's original L001 FASTQs (gzipped, `--readFilesCommand zcat`) with the CPU engine's STAR Solo, clipping and filter options
  and E9's per-read dump (`SINGLET_SOLO_DUMP`).
- `post_test.cu` (`jobs/binned.sbatch`) reads the dump and the barcode FASTQ's quality lines (and checks every read's CR against the
  dump), transcribes STAR's text-stream posterior to count what one 'F' quality would change, then runs E9's library on STAR's per-read
  inputs, on the host and on the device:
  - `perread`: `cbQual` from the FASTQ;
  - `const`: one 'F' quality for every base (the control: what the pipeline's STAR does).

  It compares per read (cbMatch, first match, the Solo flag, the posterior's CB) and per file (the 10 Solo files, through `soloWrite`)
  with STAR's.

### 1g. What `star_Log.out` says

- `ENGINE_GPU: E11 decode: … CB quality characters 70..70 (posterior pq 0.000501187 for every read)`, or `(posterior per read: …)`;
- `ENGINE_GPU: E15: posterior per read: N reads with several whitelist matches, M given a CB, …`, or `… one quality factor pq …`;
- `soloRunKept: the kept inputs (x GB) were in page-locked host memory during STAR's loop: to the host … ms (thread), back … ms`.

## 2. Identity per output and per sample

### 2a. The per-read posterior against STAR on binned qualities (job 796790, `e15star` 28ee97d9, driver 904de6e2)

pbmc1k_v3, lane L001 of the original FASTQs: 33,436,697 reads. CB quality characters: 'F' 503,467,614, ':' 20,476,038, ','
11,019,470, '#' 24,030 (5.9 % of the CB bases are not 'F'). STAR (`e15star`, T = 8, 169 s, 87.45 % uniquely mapped) wrote the per-read
dump; the driver checked that every read's CR in the barcode FASTQ equals the dump's (0 differ).

| check | `perread` host | `perread` device | `const` host / device (control) |
|---|---|---|---|
| reads with several whitelist matches | 626,729 | 626,729 | 626,729 |
| given a CB by the posterior | 565,897 | 565,897 | 546,421 |
| per read: cbMatch / first match / Solo flag | 0 / 0 / 0 differ | 0 / 0 / 0 differ | 0 / 0 / 0 differ |
| per read: the posterior's CB vs STAR's rule (transcribed from `cbMatchString`) | **0 differ** | **0 differ** | 20,492 differ |
| `cbQual`'s CRs vs the reads' | 0 differ | 0 differ | (not called) |
| STAR's 10 Solo files (`Barcodes.stats`, `Features.stats`, raw and filtered `barcodes/features/matrix`, `UMIperCellSorted.txt`, `CellReads.stats`) | **10 of 10 identical** | **10 of 10 identical** | 5 of 10 identical |

`Summary.csv` is not among the 10: it needs the mapping statistics that the engine hands `soloWrite` at level 5, which the driver does
not have. The gates (§2b) compare it.

The first run of this job (796726, driver 166ba73d) gave the same 10 of 10 files and 3 per-read differences. They were in the driver's
transcription, not in the library: it counted the exact CRs of reads with a bad UMI. In STAR a bad UMI replaces `cbMatch` before
`addStats` counts the exact match (`SoloReadBarcode_getCBandUMI.cpp:285-302`, `MATCH_CellRanger` is not defined), and the library
(E9, `solo_gpu.cu` stage (a)) already did the same.

### 2b. Gate (a) on the new samples (job 796674, binary 166ba73d, T = 18, node g051)

CPU engine against level 5 (`gpuE5`) and against level 5 with the per-read path forced (`gpuE5p`, `SINGLET_E15_PERREAD=1`: the
qualities are 'F', so the decisions must not change, and every multi-match read's CR is checked against its barcode FASTQ line).
`bench/star_accel/out_equiv.py` plus the gate's own checks:

| sample | pair | out_equiv: identical / timing-path only / differ | 10 STAR Solo files (with `Summary.csv`) | `Gene/CellReads.stats` (sorted) | `star_SJ.out.tab`, `Log.final.out` |
|---|---|---|---|---|---|
| neuron1k_v3 (3' v3, mouse, 92,902,231 reads) | cpu vs gpuE5 | 60 / 2 / 7 | 10 of 10 identical | same 600,534 rows | identical (117,254 junctions) |
| | cpu vs gpuE5p | 60 / 2 / 7 | 10 of 10 identical | same 600,534 rows | identical |
| pbmc5p_v1 (5' v1, 737K whitelist, UMI 10; 60,000,000 reads) | cpu vs gpuE5 | 58 / 2 / 7 | 10 of 10 identical | same 355,639 rows | identical (212,575 junctions) |
| | cpu vs gpuE5p | 58 / 2 / 7 | 10 of 10 identical | same 355,639 rows | identical |
| pbmc1k_v3 (66,601,887 reads) | E11's kept cpu (796377) vs gpuE5 | 60 / 1 / 8 | 10 of 10 identical | same 323,542 rows | identical (126,996 junctions) |
| | vs gpuE5p | 60 / 1 / 8 | 10 of 10 identical | same 323,542 rows | identical |
| | vs gpuE5k (`SINGLET_E15_KEEP_ON_DEVICE=1`) | 60 / 1 / 8 | 10 of 10 identical | same 323,542 rows | identical |

The differing files are E11's run-dependent set:
- neuron1k_v3 and pbmc5p_v1 (7): record order in `nonhost_reads.fq`, `star_Unmapped.out.mate1/2` (same records: sorted sha256
  equal; 490,605 / 3,667,707 reads for neuron1k_v3, 1,335,105 / 4,818,388 for pbmc5p_v1); timings in `read_stats.tsv` and
  `star_Log.std.out`; `pileup_stats.json` (`clip5p/probe/n_reads` 12,276 vs 12,000 for neuron1k_v3, and `bytes_public`) and
  `summary.json` (`memory_tier` Medium vs Large). pbmc5p_v1 has 2 files fewer in all (a 5' run).
- pbmc1k_v3 (8): the 3 order-only files, which E11's kept copy lacks; `provenance.json` (the whitelist path under `bin-s5/` vs this
  job's `bin/`); `read_stats.tsv`, `star_Log.std.out`; `pileup_stats.json` (12,194 vs 12,000, `bytes_public`); `summary.json`.
  The same 8 as E11's own gate 796433 against 796377.

The device decided the same CBs as STAR for every multi-match read in every run (the 10 files and CellReads.stats, which counts
`cbMMmultiple` and `noTooManyWLmatches` per CB, are identical). Reads with several matches: 789,078 (neuron1k_v3, 614,647 given a CB),
484,278 (pbmc5p_v1, 298,259), 1,253,320 (pbmc1k_v3, 1,096,181).

### 2c. nuclei2k_v2 and hgmm1k_v3 (gates 796865, 797017; probe 797247)

Binary 166ba73d, T = 18, node g051.

**nuclei2k_v2** (v2 chemistry, 737K whitelist, UMI 10, mouse nuclei, 118,742,315 reads; unbinned HiSeq qualities in its original
FASTQ): level 5 ran. CPU engine vs level 5:
- out_equiv: 60 identical, 2 timing/path only, 7 differ;
- the 10 STAR Solo files identical; `Gene/CellReads.stats` same 411,195 rows; `star_SJ.out.tab` identical (100,771 junctions);
  mapping statistics identical; `star_Unmapped.out.mate1/2` same records (22,811,732 reads), different order;
- 1,269,077 reads with several matches, 755,357 given a CB; kept inputs 3.00 GB (to the host 1,943 ms on the thread, back 245 ms);
- **`nonhost_reads.fq`: different records** (529,887,225 vs 530,393,838 bytes). This is not level 5. The sample has more than
  20 M unmapped reads, so the non-host screen subsamples them by read-name hash (11,406,810 kept, in all three runs). The writer then
  walks the whole `star_Unmapped.out.mate1` and takes `multi_hits[idx_r]` for the file's idx_r-th record, but `multi_hits` is indexed
  by the subsampled list (`src/pipeline/singlet.cpp:1624-1660` and `:1905-1921`). With subsampling the written reads are not the
  classified ones, and which they are depends on the unmapped file's order, which differs between the CPU engine and every GPU path.
  The non-host tables (`nonhost_em_abundance.tsv`, the summary) are identical. A pipeline bug, outside E15's files (§5.7).

**hgmm1k_v3** (human + mouse barnyard index, 63,105,786 reads): GPU mapping failed in gpuD and gpuE5 even with
`SINGLET_ENGINE_GPU_NB=131072` (half of lane A's default): `engine_gpu: cudaMalloc(&dWork, NB * sizeof(Work<C>)): out of memory`,
then "ENGINE_GPU: not used (GPU mapping failed): STAR maps every read". Both runs are level 0 and slower than the CPU engine in core-s
(§3). Their outputs are still the CPU engine's (49 identical, 2 timing/path only, the same 7 run-dependent; Solo files identical;
CellReads.stats same 524,497 rows). 

Where the memory goes (1-s `nvidia-smi` samples in the gate's `monitor.tsv`): the barnyard index takes the device to 53 GB (GRCm39:
24.7 GB), the reads to about 60 GB, and lane A's allocations to 85.1 GB (gpuD, `NB` 131,072), 89.4 GB (gpuE5, `NB` 131,072) and
89.7 GB (gpuE5, `NB` 65,536, gate 797017) when `dWork` fails.

**Probe 797247** (merged binary 13651944, T = 8) with `SINGLET_ENGINE_GPU_NB=32768`, `SINGLET_ENGINE_GPU_NBB=16384` and
`SINGLET_ENGINE_GPU_CHUNK=8388608`: the GPU mapping fits (`engine_gpu: device memory`: index 53.2 GB, reads + clip 62.3 GB, lanes peak
85.9 GB; 24,130,301 distinct sequences mapped in 10.3 s). Then the second limit: "engine_gpu: E10 not used (products: junction start
beyond 32 bits)" and "ENGINE_GPU: E11 (level 5) not used: E10 products not made". The run goes on as the default GPU path does (E8's
records and the sequence cache; STAR's loop does the per-read work). E10 packs a junction's genome start into the upper 32 bits of a
64-bit sort key (`src/engine_gpu/hostloop_gpu.cu:223`, `:325`, `:351`), and the barnyard genome (GRCh38 + GRCm39, about 5.8 G bases) is longer than 2^32. So
levels 3 to 5 cannot run on this index at all, whatever the memory. Solo on the device was not reached; its own coordinates are untested
beyond 32 bits.

Device memory in the mapping lanes (`engine_gpu: device memory` line, nuclei2k_v2): gpuD index 25.6, reads + clip 36.6 GB, lanes peak
78.7 GB; level 5 reads + clip 42.8 GB, then 42.8 → 48.7 GB over the five chunks (E8 keeps each chunk's packed results on the device),
lanes peak 89.8 GB of 94. Level 5 needs about 11 GB more than the default GPU path while it maps.

### 2d. What the pipeline's 'F' qualities change (job 796790, the `const` control)

The `const` runs are what the pipeline's STAR does (§1b); `perread` is STAR on the original FASTQ. On the 33.4 M reads of
pbmc1k_v3's L001 lane:

| quantity | STAR on the FASTQ (binned qualities) | one 'F' quality (the pipeline) |
|---|---|---|
| multi-match reads given a CB | 565,897 | 546,421 (−19,476, −3.4 %) |
| reads with another decision | | 20,492 (3.3 % of the multi-match reads, 0.06 % of all reads) |
| `Features.stats` noTooManyWLmatches | 23,189 | 32,169 |
| `raw/matrix.mtx` non-zero entries | 2,872,081 | 2,866,447 |
| `filtered/barcodes.tsv` (cells) | 1,135 | 1,135 (identical) |
| `filtered/matrix.mtx` non-zero entries | 2,274,145 | 2,274,149 |
| `UMIperCellSorted.txt` lines | 199,289 | 197,767 |
| `CellReads.stats` lines | 274,844 | 271,329 |

Same cells, a small loss of multi-match reads to `noTooManyWLmatches`. This is a property of the pipeline (CPU engine and every GPU
level alike), not of level 5.

### 2e. After merging `gpu/fable-engine` (E14) into this branch (job 797017, binary 13651944, T = 18, node g050)

`origin/gpu/fable-engine` moved from 8e54be40 to e2b0000e (E14: the genome read once per sample, the CR4 probes concurrently). It
merged without conflicts (13651944; E14 touches `src/pipeline/singlet.cpp`, `src/star/Genome*`, `GenomePreload.*`, none of E15's
files).
- pbmc1k_v3, E11's kept CPU outputs vs level 5: 60 identical, 1 timing/path only, the same 8; the 10 Solo files identical;
  CellReads.stats same 323,542 rows; `star_SJ.out.tab` and the mapping statistics identical. Level 5 took 178.2 s / 654.4 core-s
  (212.7 s / 685.8 core-s before the merge, on g051).
- hgmm1k_v3 with `NB` 65,536: GPU mapping failed again (§2c), and the outputs again equal the CPU engine's (49 / 2 / 7, Solo files
  identical, CellReads.stats same 524,497 rows).

## 3. Timings

T = 18, short-gpu node g051 (H100 NVL), gate 796674 (binary 166ba73d). Wall time and core-s are for the whole `singlet --pipeline` run
(cgroup `cpu.stat`); maxRSS from `/usr/bin/time`. pbmc1k_v3's cpu and gpuD are E11's (gate 796377, node g050).

| sample (M reads) | run | wall s | core-s | wall s/M | core-s/M | maxRSS |
|---|---|---|---|---|---|---|
| neuron1k_v3 (92.90) | cpu | 520.4 | 2180.1 | 5.60 | 23.47 | 35.3 GB |
| | gpuD | 326.7 | 1205.6 | 3.52 | 12.98 | 62.6 GB |
| | **gpuE5** | **271.3** | **801.0** | **2.92** | **8.62** | 83.2 GB |
| | gpuE5p | 273.6 | 802.7 | 2.95 | 8.64 | 83.1 GB |
| pbmc5p_v1 (60.00) | cpu | 489.8 | 2163.9 | 8.16 | 36.07 | 32.3 GB |
| | gpuD | 229.5 | 845.5 | 3.83 | 14.09 | 41.0 GB |
| | **gpuE5** | **199.2** | **578.7** | **3.32** | **9.65** | 53.2 GB |
| | gpuE5p | 199.0 | 578.8 | 3.32 | 9.65 | 53.2 GB |
| pbmc1k_v3 (66.60) | cpu (E11, g050) | 424.5 | 2607.1 | 6.37 | 39.15 | 29.7 GB |
| | gpuD (E11, g050) | 253.4 | 996.6 | 3.80 | 14.96 | 43.6 GB |
| | gpuE5 (E11, g050) | 210.8 / 210.3 | 680.4 / 677.5 | 3.16 | 10.20 | 61.0 GB |
| | **gpuE5** | **212.7** | **685.8** | **3.19** | **10.30** | 61.0 GB |
| | gpuE5p | 211.4 | 679.2 | 3.17 | 10.20 | 61.0 GB |
| | gpuE5k (kept inputs on the device) | 215.1 | 685.4 | 3.23 | 10.29 | 61.0 GB |
| nuclei2k_v2 (118.74; gate 796865, g051) | cpu | 611.2 | 4712.4 | 5.15 | 39.69 | 35.0 GB |
| | gpuD | 459.2 | 1750.6 | 3.87 | 14.74 | 75.2 GB |
| | **gpuE5** | **391.0** | **1179.2** | **3.29** | **9.93** | 113.9 GB |
| hgmm1k_v3 (63.11; gate 796865, g051) | cpu | 431.7 | 1682.8 | 6.84 | 26.67 | 62.2 GB |
| | gpuD, `NB` 131,072 (GPU mapping failed: level 0) | 361.2 | 1729.9 | 5.72 | 27.41 | 78.1 GB |
| | gpuE5, `NB` 131,072 (GPU mapping failed: level 0) | 373.0 | 1783.1 | 5.91 | 28.26 | 83.3 GB |
| | gpuE5, `NB` 65,536 (level 0; gate 797017, g050, binary 13651944) | 289.9 | 1736.9 | 4.59 | 27.52 | 83.3 GB |
| | cpu (gate 797017, g050) | 345.6 | 1587.4 | 5.48 | 25.15 | 62.3 GB |
| pbmc1k_v3 after the merge of E14 (gate 797017, g050, binary 13651944) | **gpuE5** | **178.2** | **654.4** | **2.68** | **9.83** | 61.0 GB |

Level 5 compared with:
- the default GPU path: neuron1k_v3 −55.4 s (−17.0 %) and −404.6 core-s (−33.6 %); pbmc5p_v1 −30.3 s (−13.2 %) and −266.8 core-s
  (−31.6 %); nuclei2k_v2 −68.2 s (−14.9 %) and −571.4 core-s (−32.6 %) (E11 on pbmc1k_v3: −16.9 %, −31.9 %);
- the CPU engine: neuron1k_v3 1.92x wall, 2.72x core-s; pbmc5p_v1 2.46x, 3.74x; nuclei2k_v2 1.56x, 4.00x (pbmc1k_v3: 2.0x, 3.8x).

The per-read path (`gpuE5p`) costs nothing measurable: the "multi" stage goes from 1.9 to 36.6 ms on neuron1k_v3 (789,078 reads'
qualities fetched from the barcode FASTQ in 17.1 ms) and from 1.5 to 42.1 ms on pbmc1k_v3 (1,253,320 reads in 12.1 ms). The barcode
FASTQ's unmapping moves from right after STAR's loop to after `soloRunKept`, which is why the logged "write" interval grows (1.03 →
1.77 s, 0.65 → 1.22 s) while the run does not.

**Phase breakdown** (`bench/fable/solowire/s8_stages_e11.py`; wall s / core-s):

| stage | neuron gpuD | neuron **gpuE5** | neuron gpuE5p | pbmc5p gpuD | pbmc5p **gpuE5** | pbmc5p gpuE5p |
|---|---|---|---|---|---|---|
| decode .1fq → FASTQ + R11 histogram | 28 / 120.9 | 26 / 120.0 | 28 / 121.8 | 15 / 75.3 | 17 / 78.2 | 16 / 77.8 |
| probe/ref load | 3 / 4.8 | 3 / 4.6 | 3 / 5.3 | 5 / 9.0 | 4 / 6.6 | 4 / 6.4 |
| STAR load, CR4 probe, start | 31 / 34.2 | 47 / 53.3 | 48 / 54.2 | (*) | 6 / 6.5 | 6 / 5.8 |
| GPU prefill (read, clip, seed, stitch, records, E10/E11 keep) | 65 / 282.1 | 69 / 354.2 | 69 / 354.0 | 81 / 384.7 (*) | 42 / 148.1 | 43 / 149.3 |
| STAR per-read host pass | 66 / 404.1 | **10 / 24.4** | **9 / 20.3** | (*) | **8 / 29.9** | **7 / 28.7** |
| STAR stream into pileup (Solo, exit) | 21 / 72.4 | **13 / 28.6** | **14 / 33.0** | 8 / 35.8 | 9 / 21.3 | 9 / 21.4 |
| GPU-pileup hand-off + pileup | 53 / 137.7 | 43 / 64.3 | 44 / 67.2 | 32 / 88.6 | 26 / 36.6 | 26 / 37.0 |
| non-host | 23 / 46.7 | 23 / 49.7 | 22 / 45.8 | 30 / 57.1 | 30 / 57.2 | 30 / 56.3 |
| export, QC, callers | 25 / 53.1 | 26 / 54.4 | 25 / 51.6 | 40 / 116.1 | 39 / 113.3 | 40 / 118.3 |
| demux/ancestry/writers | 11 / 48.6 | 11 / 46.9 | 11 / 47.9 | 17 / 78.2 | 18 / 80.2 | 17 / 77.4 |
| total | 327 / 1205.4 | 271 / 800.9 | 274 / 802.5 | 229 / 844.8 | 198 / 578.0 | 199 / 578.4 |

The "startup" row (≤ 1 s) is left out; the totals include it. (*) For pbmc5p_v1's gpuD the script found no prefill-end marker
and put STAR's start, the prefill and STAR's loop in one 81 s row.

**STAR's Solo step** ("started/finished Solo counting"): neuron1k_v3 60 s (cpu), 17 s (gpuD); pbmc5p_v1 26 s (cpu), 3 s (gpuD); not run
at level 5. In its place, after the loop: `soloRunKept` 0.79 s and the 11 files 1.03 s (neuron1k_v3), 0.43 s and 0.21 s (pbmc5p_v1),
0.56 s and 0.65 s (pbmc1k_v3), 0.92 s and 0.50 s (nuclei2k_v2, whose Solo step took 63 s on the CPU engine and 11 s on gpuD).

**nuclei2k_v2** (stages, wall s / core-s): gpuD spends 249 / 1156 in STAR's load, prefill and loop together (the script found no
prefill-end marker); level 5 spends 35 / 43 in STAR's start, 143 / 558 in the prefill (reading reads 29 s, GPU 98 s, cache inserts
16 s) and 23 / 99 in STAR's loop on the 1,885,914 reads without a record (1.6 %). Non-host takes 78 s in both (11.4 M subsampled unmapped reads).

**The kept inputs in page-locked host memory during STAR's loop (§1e):**

| sample | kept | to the host (thread, overlaps the loop) | back to the device |
|---|---|---|---|
| neuron1k_v3 | 2.13 GB | 1,382 ms | 174 ms |
| pbmc5p_v1 | 1.35 GB | 877 ms | 111 ms |
| pbmc1k_v3 | 1.59 GB | 1,031 ms | 130 ms |
| nuclei2k_v2 (gate 796865) | 3.00 GB | 1,943 ms | 245 ms |

The device memory is free for another sample during STAR's loop (E11 §5.3). The price is on the tail of a single run: `soloRunKept`
takes 0.56 s against 0.16 s with the inputs left on the device (pbmc1k_v3, gpuE5 vs gpuE5k). That is the copy back (130 ms) and
`cudaFreeHost` of the page-locked copies in `soloKeepRelease` (about 0.27 s, by difference: soloRun's own stages are 0.14 s in both runs).
The three pbmc1k_v3 level-5 runs are within 3.7 s of
each other, so the 0.4 s is below the run-to-run noise.

## 4. Surprises

1. **The premise did not hold: binned barcode qualities never reach STAR in the pipeline** (§1b). E11 §5.1 expected level 5 to
   fall back on NovaSeq samples; it never did, because the `.1fq` has no R1 qualities and the decode writes 'F'. Every probed sample
   whose GPU mapping succeeds runs level 5 with E11's code already. The per-read port matters only for an input that carries R1
   qualities, and it is what makes STAR's rule exact there (§2a).
2. **The pipeline's CB decisions differ from STARsolo run on the FASTQ** (§2d): 20,492 of 626,729 multi-match reads (3.3 %) on
   pbmc1k_v3's L001 lane, 19,476 fewer reads with a CB; same 1,135 cells. Whether that matters for the atlas is a pipeline question
   (§5.2), not a level-5 one.
3. **The library was right where the test was wrong.** The first binned run showed 3 reads whose posterior differed from the driver's
   transcription of STAR, with all 10 files identical. The transcription counted exact CRs of reads with a bad UMI; STAR and E9's
   library do not (§2a).
4. **The pipeline cannot run STARsolo on FASTQ at all.** The legacy `--reads` path runs `CB_samTagOut` and always passes
   `--readFilesCommand zcat`; it also needs `--barcodes`. The binned check therefore needed a plain STAR (`e15star`) built from the
   vendored sources. Its STAR took 169 s on 33.4 M reads at T = 8 with Solo and no BAM.
5. **My first probe used the human genome for nuclei2k_v2 (a mouse sample)** and saw a device out-of-memory in the mapping lanes. With
   GRCm39 the sample maps on the device, but the lanes peak at 89.8 GB of the H100 NVL's 94 GB: the mapping lanes, not Solo, are the
   memory limit of the engine (§5.1).
6. **Moving the kept inputs off the device is not free on the tail:** `cudaFreeHost` of 1.6 GB of page-locked memory takes longer than
   the copy back (about 0.27 s by difference, against 130 ms; §3).
7. **A pipeline bug surfaced on nuclei2k_v2:** with more than 20 M unmapped reads, `nonhost_reads.fq` holds reads other than the
   classified ones, and different ones per engine (§2c, §5.7). The other gated samples have at most 4.8 M unmapped reads.
8. **The barnyard index hits two limits, neither of them Solo** (§2c): the lanes' device memory (lane A at a half and a quarter of
   its size still fails; it fits with lane A at 1/8, lane B at 1/4 and half-size chunks), then E10's 32-bit junction starts.

## 5. What remains before level 5 can be the default of `--engine gpu`

1. **A device memory budget for the mapping lanes** (`src/engine_gpu/engine_gpu.cu`, `gpuPart` / `runLane`; outside E15's files).
   - Lane A keeps `NB` = 262,144 reads in flight (`Work` + `ReadBuf` ≈ 150 KB each, ~40 GB), lane M 262,144 and lane B 65,536, and
     each lane also allocates its result slots for the whole chunk. Nothing compares this with the free memory.
   - A failed `cudaMalloc` abandons GPU mapping for the whole run: "ENGINE_GPU: not used (GPU mapping failed): STAR maps every read".
     That is level 0, not level 3: STAR maps all reads on the CPU, as in the CPU engine.
   - hgmm1k_v3 (the human + mouse barnyard index, 53 GB on the device against GRCm39's 25.6 GB) fails at lane A's `dWork` at
     every GPU level (probes 796560, 796691; gates 796865 and 797017, also with `NB` at 131,072 and 65,536). With smaller lanes
     (`NB` 32,768, `NBB` 16,384, chunk 8 M; probe 797247) it maps, with the lanes peaking at 85.9 GB, and hits the second limit:
     E10's 32-bit junction starts (§2c, item 8).
   - nuclei2k_v2 (118.7 M reads, 79.8 M distinct sequences) maps, with the lanes peaking at 89.8 of 94 GB at level 5 and 78.7 GB on
     the default GPU path. E8's and E10's device-resident results cost level 5 about 11 GB during mapping (§2c).
   - The port: size `NB` (and the chunk) from `cudaMemGetInfo` once the index and the reads are on the device, or halve `NB` and retry
     on `cudaErrorMemoryAllocation`. `SINGLET_ENGINE_GPU_NB`, `_NBB` and `_CHUNK` are the manual knobs today (§2c).
2. **R1 qualities, if the atlas wants STARsolo's decisions on the sequencer's qualities** (§2d). The `.1fq` would have to store the CB
   qualities (2 bits per base for the 4-level binning: 4 B per read for a 16-base CB) and the decode would write them. Level 5 then
   takes the per-read path by itself (§1d). The CPU engine's outputs change too, so this is a format and pipeline decision.
3. **Solo configurations the pipeline does not use** (multimappers, EmptyDrops_CR, GeneFull/Velocyto, CB matching without
   pseudocounts or Exact) still fall back to level 3. Multi-segment chemistries (`CB_UMI_Complex`) and the legacy FASTQ path run
   `CB_samTagOut`, where E11's guard keeps level 3 as well; that path was not studied here.
4. **E11 §5.4** (E10's level-4 fallback decides `e10NoSum` before the read source is built) is still open.
5. **The kept inputs on the host:** free the page-locked copies off the tail (about 0.27 s per sample, §3). With one sample per GPU,
   keeping them on the device (`SINGLET_E15_KEEP_ON_DEVICE=1`) is the faster choice. The gain (1.4–3 GB of device memory free during
   STAR's loop) is real only when another sample's prefill shares the device; that case was not measured.
6. **For the default switch**, beyond items 1, 4 and 5: the release-gate datasets gated at level 5 one by one (this report adds
   neuron1k_v3, pbmc5p_v1 and nuclei2k_v2 to E11's pbmc1k_v3; hgmm1k_v3 waits for items 1 and 8), and level 5 gated again on `gpu/fable-engine` once E12, E13
   and E16 are merged (they touch the prefill, the ingest and the tail that level 5 runs through).
7. **Not level 5, found by the gate:** `nonhost_reads.fq` is written from the wrong reads when the non-host screen subsamples (more
   than 20 M unmapped reads; §2c). The writer must take the classification by the kept reads' names (or keep their file positions),
   not by the record's index in the whole file. `src/pipeline/singlet.cpp:1905-1921`.
8. **E10's 32-bit junction starts** (`src/engine_gpu/hostloop_gpu.cu`, outside E15's files) keep levels 3–5 off any index longer
   than 2^32 bases (the barnyard, §2c). The port: a sort key with the full 64-bit start (start and gap as two keys, or a 128-bit
   key). Then hgmm1k_v3 can be gated at level 5 with item 1's knobs; Solo on the device has not run on such an index yet.

## 6. Files touched, and why

| file | change | why |
|---|---|---|
| `src/engine_gpu/solo.cuh` | `cbPosteriorRead` (per read, the quality at the mismatch or the N) replaces `cbPosterior`; `ctz64` | the rule |
| `src/engine_gpu/solo_gpu.h` | `CbQualFn`; `SoloReads::cbQual`; `SoloModel::pqTab`; `SoloResult::{nMulti, nMultiAccepted, perReadQual, readCb}`; `SoloKeep`'s host copies and thread; `soloKeepToHost`; `soloRunKept(…, cbQual)` | API of the above |
| `src/engine_gpu/solo_gpu.cu` | the "multi" stage (flags, compaction, the quality download/upload, the kernel); stage (c) and the per-read outputs take a multi-match read's own choice; `readCb`; the log line; `soloKeepToHost`, the copy back in `soloRunKept`; `soloKeepRelease` joins the thread and frees the page-locked copies | the rule; E11 §5.3 |
| `src/star/SeqCacheGpuPrefill.cpp` (E11 parts) | `E11State::{perRead, nQualFetched, sQualFetch}`; `e11Config` builds `pqTab` and guards `outQSconversionAdd`; `e11Posterior` without the constant-quality refusal (guard for characters ≤ ' ', `SINGLET_E15_PERREAD`); `e11CbQual`; the barcode file stays mapped until `soloRunKept` when per read; `soloKeepToHost` after `mapDistinct` (`SINGLET_E15_KEEP_ON_DEVICE`); the E15 log line; `pqStr` | the engine's side |
| `bench/fable/posterior/jobs/{build,probe,gate}.sbatch` | build; which samples reach level 5 and why not; the multi-sample gate (`MODES_<s>`, `PAIRS_<s>`, `NB_<s>`) | deliverable |
| `bench/fable/posterior/{post_test.cu, star_main.cpp}`, `jobs/{binned,starbuild}.sbatch` | the binned-quality check against STAR (§1f) | deliverable |
| `docs/gpu-engine/fable/E15-posterior.md` | this report | deliverable |

Not changed: `include/singlet/engine_gpu.h` (nothing needed), the decode, `engine_gpu.cu`, STAR's Solo code, `src/pipeline`.

## 7. Jobs, commits, how to run

**Branch `gpu/fable/posterior`** (from `gpu/fable-engine` 8e54be40):
- 46af66b7: build and probe jobs.
- 098408d8: the per-read posterior on the device; the kept inputs in page-locked host memory during STAR's loop.
- 166ba73d: `cbQual` gets the reads' CRs (`e11CbQual` checks its lines); the binned-quality driver. **Last E15 change to `src/`.**
- ad52e499, 6de3b25e, f5d91641: the gate job (`MODES_<s>`, `PAIRS_<s>`, `NB_<s>`).
- 1bda8f09, 61af5ebf, 5b882315: the binned job through the pipeline's legacy FASTQ path (abandoned: `CB_samTagOut`, §4.4).
- 2bb5c127: the probe takes GRCm39 for nuclei2k_v2.
- 28ee97d9: `e15star` and the binned job on it.
- 904de6e2: the driver's exact counts (§2a).
- 13651944: merge of `gpu/fable-engine` at e2b0000e (E14).
- 75c1d55e and later: this document.

**Builds** (short-gpu, no GPU):

| job | commit | output |
|---|---|---|
| 796559 | 46af66b7 (8e54be40's code) | `bin-base/singlet-gpu` |
| 796622 | 098408d8 | `bin/` (replaced) |
| 796664 | 166ba73d | `bin/singlet-gpu` (gates 796674, 796865; probe 796691) |
| 796718 | 28ee97d9 | `bin/e15star` (25 s) |
| 796866 | 13651944 (the merge) | `bin-m1/singlet-gpu` |

**Probes** (debug-gpu, T = 8, level 5 stopped after the prefill): 796560 (E11's binary 673cb317; pbmc2m, pbmc5p_v1, neuron1k_v3,
hgmm1k_v3, nuclei2k_v2 with the wrong genome), 796691 (166ba73d; nuclei2k_v2 on GRCm39, hgmm1k_v3), 797247 (13651944; hgmm1k_v3 with
`SINGLET_ENGINE_GPU_NB=32768`, `_NBB=16384`, `_CHUNK=8388608`).

**Binned-quality check** (debug-gpu, T = 8): 796790 (final), 796726 (the driver's transcription bug). 796663, 796680, 796689 tried the
pipeline's legacy path (§4.4).

**Gates** (short-gpu, H100 NVL, T = 18):

| job | binary | samples, modes | result |
|---|---|---|---|
| 796674 | 166ba73d | neuron1k_v3, pbmc5p_v1: cpu gpuD gpuE5 gpuE5p; pbmc1k_v3: gpuE5 gpuE5p gpuE5k against E11's kept cpu | identity (§2b), timings (§3) |
| 796865 | 166ba73d | nuclei2k_v2, hgmm1k_v3 (`NB_hgmm1k_v3=131072`): cpu gpuD gpuE5 | §2c |
| 797017 | 13651944 (merge with E14) | hgmm1k_v3 (`NB_hgmm1k_v3=65536`): gpuE5 cpu; pbmc1k_v3: gpuE5 against E11's kept cpu | §2e |

**Results:** `$W/jobs/fable/posterior/results/{gate,probe,binned}-<job>/` (`summary.txt`; per gate run `stages.txt`,
`starlog_engine.txt`, `star_Log.out`, `CellReads.stats`, `monitor.tsv`; per binned run STAR's `star_Solo.out/`, `post_test.log`).
Logs: `$W/logs/e15-{build,starbuild,probe,binned,gate}-<job>.out`. `$W` = `/mnt/projects/debruinz_project/debruinz/singlet`.

**Run:**
- build: `sbatch --export=ALL,OUTD=<dir> bench/fable/posterior/jobs/build.sbatch`;
- gate: `sbatch --export="ALL,BINDIR=<dir>,SAMPLES=neuron1k_v3 pbmc5p_v1,MODES=cpu gpuE5,PAIRS=cpu:gpuE5" bench/fable/posterior/jobs/gate.sbatch`
  (`MODES_<s>`, `PAIRS_<s>`, `NB_<s>` per sample; pbmc1k_v3's cpu defaults to E11's kept outputs);
- binned check: `sbatch bench/fable/posterior/jobs/starbuild.sbatch`, then `sbatch [--export=ALL,SAMPLE=<s>,NREADS=<n>]
  bench/fable/posterior/jobs/binned.sbatch` (debug-gpu; the driver is compiled from the branch in the job).

**Switches:** `SINGLET_ENGINE_GPU_E2E=5` (with `--engine gpu`, `SINGLET_GPU_PILEUP=on`) as in E11. New:
- `SINGLET_E15_PERREAD=1`: the per-read posterior path even when one quality factor would do (a test switch; the outputs do not change
  on `.1fq` input);
- `SINGLET_E15_KEEP_ON_DEVICE=1`: E11's behaviour, the kept inputs stay on the device during STAR's loop.
