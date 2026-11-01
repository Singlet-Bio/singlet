# E11: STARsolo on the device inside the engine (solowire)

Experiment E11 ("solowire") of the GPU aligner program. Branch `gpu/fable/solowire`, from `gpu/fable-engine` at 4439fc8b (the E10
merge). It wires E9's device Solo library (`src/engine_gpu/solo_gpu.{h,cu}`, `solo.cuh`) into the engine as a new level,
`SINGLET_ENGINE_GPU_E2E=5` (with `--engine gpu` and `SINGLET_GPU_PILEUP=on`). Level 4 keeps its meaning ("Solo ignored").

With the variable unset or set to 1–4, the binary runs the code it ran before. §2c checks the CPU engine, the default GPU path and
level 3 of this binary against E10's gates.

## 0. Answer in brief

- **Level 5 is a complete run with STAR's per-read loop skipped for every read that has a GPU record.**
  - STAR's loop sees only the 156,931 reads without a record (0.24 %: reads with N, and the host lane's failures).
  - STARsolo's Gene counting for all 66.6 M reads runs on the device after the loop, using E9's library on inputs kept on the device
    since the prefill.
  - `star_Solo.out/` is written from those counts, and STAR's Solo step does not run.
- **Gate (a), pbmc1k_v3 at T = 18, CPU engine vs level 5 (job 796377): identical (§2).**
  - 60 files are byte-identical (E8's gate also had 60). They include every count matrix, all 10 STAR Solo files (with `Gene/Summary.csv`,
    which E9 had not ported), `cell_calls.tsv`, `cell_qc_metrics.tsv` and `read_qc_summary.json`.
  - `Gene/CellReads.stats` has the same 323,542 rows (compared sorted). The pipeline deletes this file after parsing it.
  - 2 files differ only in timings or paths.
  - 7 files differ, and the same 7 differ between the CPU engine and the default GPU path (`gpuD`):
    - record order in 3 files (same sorted records);
    - stage timings in 2 files;
    - two GPU-vs-CPU fields in `pileup_stats.json` / `summary.json`.

  Against level 3 the only differences are run-dependent. Against level 4, level 5 restores exactly the 7 Solo files and the 2 QC files
  that level 4 lost.
- **Timing at T = 18 (§3, node g050):**

  | run | wall | core-s |
  |---|---|---|
  | CPU engine | 424.5 s | 2,607 |
  | gpuD | 253.4 s | 997 |
  | E8 | 251.8 s | 965 |
  | level 3 | 252.5 s | 942 |
  | level 4 (Solo incomplete) | 209.5 s | 672 |
  | **level 5** (two runs) | **210.8 s / 210.3 s** | **680 / 678** |

  Compared with the default GPU path, level 5 (mean of the two runs) saves **−42.8 s (−16.9 %) and −318 core-s (−31.9 %)**. Compared
  with the CPU engine it is 2.0x faster in wall time and uses 3.8x fewer core-s. Solo on the device costs about 1 s over level 4,
  which writes incomplete Solo files.
- **Solo on the device (§3a).**
  - Inputs kept during the prefill: 41 ms (3.4 ms for the reads' CR/UR, 37.8 ms for the records' Gene annotation and the
    read → record map), 1.59 GB held on the device through STAR's loop.
  - After the loop: `soloRunKept` takes 0.15 s, of which 89 ms is the download; the filtering and the 11 files take 0.65 s on the
    host.
  - It replaces STAR's Solo step:

    | run | STAR's Solo step |
    |---|---|
    | CPU engine | 43 s |
    | gpuD | 14 s |
    | E8 / level 3 | 13 s |

    It also replaces the per-read Solo work inside STAR's loop: barcode output 31 thread-s and Solo record 26 thread-s at E8
    (E10 profile).
- **Two bugs found by the gate and fixed.**
  - The multimapper guard: STAR never stores "Unique" in `MultiMappers::types`. Gate 796326 therefore ran level 5 as level 3, which
    at least exercised the fallback (identical outputs).
  - False sharing in the decode's new Q30 counters: the barcode pass took 38.6 s instead of 0.8 s (+620 core-s) in gate 796377's
    level-5 run. Gate 796377's level-5 outputs are valid; its level-5 timing is not, and §3 uses gate 796433's two runs instead.
- **Limits (§5).**
  - Level 5 runs only for the Solo configuration E9 ports: CB_UMI_Simple, Gene, Unique, 1MM_multi_Nbase_pseudocounts, 1MM_All,
    CellRanger2.2.
  - It also needs CB base qualities that give one posterior constant. pbmc1k_v3's R1 qualities are a constant 'F'; binned NovaSeq
    qualities (F, :, ,) are not.
  - Otherwise the run falls back to level 3 and says why in `star_Log.out`. A per-read posterior is the main step to make level 5
    general.

## 1. What level 5 does

Level 5 is level 4 plus two things:
- Level 4: every read with a GPU record skips STAR's loop, and STAR reads only the reads without a record.
- STARsolo's Gene counting runs on the device for every read.
- STAR's Solo post-map step is not run.

### 1a. Before STAR's loop (the prefill, `src/star/SeqCacheGpuPrefill.cpp`)

**`e11Config`** decides whether E9's port covers the run's Solo options. If it does, it fills E9's host model from STAR's own objects:
- the transcript model from `Transcriptome` (trS/trE/trEmax/trExI/trGene/trExN/trStr/exSE, gene ids and names);
- the whitelist from `ParametersSolo::cbWL`;
- the CB/UMI geometry and strand;
- the CellRanger2.2 parameters;
- `readQualityScoreBase`.

It covers the following options:
- `CB_UMI_Simple`;
- `soloFeatures Gene` only;
- whitelist < 2^22;
- `1MM_multi_Nbase_pseudocounts`;
- `soloUMIdedup 1MM_All`;
- `soloMultiMappers Unique`;
- no UMI filtering;
- `soloCellFilter CellRanger2.2`;
- `soloCellReadStats Standard`;
- no CB/UB SAM attributes and no readInfo;
- default Solo file names;
- CB ≤ 21, UMI ≤ 12, genes < 2^17, ≤ 4 mito contigs.

For anything else, the log says "E11 (level 5) not used: …" and the run is level 3: STAR's Solo sees every read and E10's other
products still come from the device.

**The E10 decode** already parses R1 and R2 for the device entries. It now also counts what `Gene/Summary.csv` needs and what STAR
would have counted per read in its loop:
- the Q30 bases of the CB+UMI spans (`SoloReadBarcode::qualHist`, padded with 'H' like STAR when the barcode read is short);
- the Q30 bases of whole R2 reads;
- the lowest and highest CB quality character.

The counters are thread-local; §0 describes the false-sharing version.

**`e11Posterior`** checks that E9's single-`pq` posterior is exact for this run. STAR computes the multi-match posterior per read from
the quality at the mismatch, `qin = min(q - QSbase, QSmax)`. That value is one constant when either:
- every CB quality character is ≥ QSbase + QSmax, or
- all CB quality characters are the same.

pbmc1k_v3: 70..70, pq = 10^-3.3. If neither holds, level 5 is not used.

**`mapDistinct`** (`src/engine_gpu/engine_gpu.cu`) keeps on the device, after E10's products:
- the CR/UR of every read (`soloKeepReads`: a strided copy out of E10's aux upload before it is freed);
- each record's Gene annotation, computed while the blobs are still on the device (Solo's stage (b), `classifyAlign` on the record's
  transcripts, `soloKeepRecords`);
- the read → record map.

For pbmc1k_v3 that is 66.6 M × 20 B + 31.6 M × 8 B = 1.59 GB. The blobs are freed as before.

**`e11RnaAdjust`** runs after E10's level block and turns the whole-read R2 Q30 counts into STAR's. STAR's `qualHist[0]` counts
`[clip5, Lorig - clip3)` of the reads it loads.
- For the 4.16 M clipped reads with a record, the counts outside their clip span (`g_handoffClip5/3`) are subtracted.
- The 156,931 reads without a record are subtracted entirely, because STAR's loop counts them itself.

This takes 0.21–0.24 s on 18 threads.

### 1b. STAR's loop

The loop handles only the reads without a record. For each of them, `ReadAlign::e11Annot` (`ReadAlign_outputAlignments.cpp`) stores
the read index and the packed Gene annotation that STAR's `ReadAnnotFeature` gives, in place of `soloRead->record`. The annotation
holds:
- the single gene, or none;
- the transcript count;
- the gene-set size, capped at 2;
- whether the read is exonic;
- whether it is mito.

These are the same fields as E9's debug dump. No Solo temp files are written.

### 1c. After STAR's loop (`seqCacheGpuAfterMapping`)

1. The annotations of the STAR-mapped reads are gathered from the chunks and sorted by read (5 ms).
2. `soloRunKept` takes a slot lock on the device that holds the kept inputs and runs `soloRun` there:
   - whitelist matching and the posterior per distinct CR;
   - per-read Solo rules and keys;
   - the (CB, gene, UMI) sort;
   - 1MM_All as Hamming-1 components;
   - Barcodes / Features / CellReads stats.
3. `soloWrite` runs the CellRanger2.2 filtering and writes `star_Solo.out/`:
   - `Barcodes.stats`;
   - `Gene/{Features.stats, Summary.csv, UMIperCellSorted.txt, CellReads.stats}`;
   - `Gene/raw/*`;
   - `Gene/filtered/*`.

   CellReads.stats rows are sorted by whitelist index. STAR's row order is run-dependent, and the pipeline reads the file by barcode.
4. `STAR.cpp` skips `soloMain.processAndOutput()`. Solo's constructor still runs.

### 1d. `Gene/Summary.csv` (new)

E9 had not ported this file. It is written from the device counts, STAR's mapping counters (`g_statsAll`) and the Q30 counts of §1a.
Its fields:
- reads;
- valid barcodes;
- saturation;
- Q30 fractions;
- mapped to genome and to gene;
- cells;
- reads, UMIs and genes per cell (means and medians);
- genes detected.

It follows STAR's conventions from `SoloFeature_statsOutput.cpp` and `cellFiltering.cpp`:
- `%g` formatting;
- unsigned integer means;
- medians at `[n/2]`;
- "Median UMI per Cell" from the descending sort over all CBs.

It came out byte-identical on the first run.

### 1e. Failures

Level 5 is decided in steps. Each step that cannot be taken falls back to level 3, with the reason in `star_Log.out`, before anything
is skipped. The steps are:
- the configuration;
- the posterior's qualities;
- E10's decode and products;
- the device keep.

Once STAR's loop has been given the reduced read source, a later failure exits with an error instead of writing incomplete Solo files.
Such failures are the read source, the RNA adjustment, `soloRunKept` and `soloWrite`.

## 2. Identity per output

### 2a. Gate (a): CPU engine vs level 5 (job 796377, binary 8d9eaa85; `bench/star_accel/out_equiv.py` + the gate's own checks)

| outputs | result |
|---|---|
| count matrices and pileup products: `exon_counts`, `intron_counts`, `ambiguous_counts`, `sj_counts`, `splice_psi`, `polya_counts`, `mt_heteroplasmy`, `ambient_corrected_counts`, `vdj_gene_usage` (`.1pz`); `private/snp_ad`, `snp_dp`, `mt_allele_counts(_rev)`; `mt/mt_events` | identical |
| tables: `gene_expression.tsv`, `cell_calls.tsv`, `auto_barcodes.tsv`, `doublet_scores.tsv`, `donor_assignments.tsv` (both), `cell_cycle_scores.tsv`, `ambient_profile.tsv`, `ambient_contamination.tsv`, `mt_variants.tsv`, `mt/mt_summary.tsv`, `splice_events.tsv`, `sj_features.tsv.gz`, `polya_sites.tsv.gz`, `saturation_curve.tsv`, `saturation_metrics.tsv`, `metrics_summary.csv`, `mt_reference.fa`, the `private/` VCFs, sites and consensus, non-host EM abundance and summary, `rrna_report.json`, `sex_call.json`, `ancestry_call.json`, `cr4_probe` files | identical |
| **STAR Solo**: `Barcodes.stats`, `Gene/Features.stats`, `Gene/Summary.csv`, `Gene/UMIperCellSorted.txt`, `Gene/raw/{barcodes.tsv, features.tsv, matrix.mtx}`, `Gene/filtered/{barcodes.tsv, features.tsv, matrix.mtx}` | **10 of 10 identical** |
| **`Gene/CellReads.stats`** (hard-linked before the pipeline deletes it) | **same 323,542 rows** (sorted; the file order differs, as between two CPU runs) |
| **Solo-derived**: `cell_qc_metrics.tsv` (reads CellReads.stats), `read_qc_summary.json` | **identical** (both differ at level 4) |
| `star_SJ.out.tab` | identical (126,996 junctions) |
| `star_Log.final.out` | mapping statistics identical (timing lines excluded) |
| `nonhost_reads.fq`, `star_Unmapped.out.mate1/2` | same records (sorted sha256; 428,468 and 3,355,559 reads); order differs |
| `provenance.json`, `ambient_contamination.json` | differ only in timings/paths |
| `read_stats.tsv`, `star_Log.std.out` | stage rows and timings |
| `pileup_stats.json` | `clip5p/probe/n_reads` 12,194 (CPU) vs 12,000 (GPU engine), and `bytes_public` |
| `summary.json` | `memory_tier` Medium (CPU) vs Large (GPU engine) |

`out_equiv`'s count: **60 identical, 2 timing/path only, 7 differ.** The 7 are exactly cpu vs gpuD's 7 in the same job: the GPU
engine's known run-dependent differences, also present without E11. E8's gate had the same 60.

### 2b. Level 5 against levels 3 and 4 (same job)

- **gpuE3 vs gpuE5:**
  - 60 identical, 3 timing-only, 6 differ;
  - the 6 are read order (3 files), timings (2), and `pileup_stats.json`'s `bytes_public`;
  - CellReads.stats has the same rows.
- **gpuE4 vs gpuE5:** 54 identical, 12 differ. On top of the run-dependent ones, these are:
  - the 7 Solo files (`Barcodes.stats`, `Features.stats`, `Summary.csv`, `UMIperCellSorted.txt`, raw and filtered `matrix.mtx`,
    filtered `barcodes.tsv`);
  - `cell_qc_metrics.tsv`;
  - `read_qc_summary.json`;
  - CellReads.stats (25,754 vs 323,542 rows).

  These are what level 4 loses and level 5 restores.
- **Gate 796433 (binary 673cb317, the timing runs).**
  - Against 796377's kept CPU outputs: 60 identical, CellReads.stats has the same rows, and the Solo files and the two QC files are
    identical. The other differences are:
    - the 3 order-only files, which the kept copy lacks;
    - the run-dependent ones above;
    - the whitelist path in `provenance.json` (`bin-s5/` vs `bin-s6/`).
  - gpuE5 vs gpuE5b: 63 identical, 3 timing-only, 3 differ (timings and `bytes_public`). CellReads.stats is byte-identical even in
    file order, because the rows are sorted by whitelist index and the file is therefore deterministic.

### 2c. Default paths and level 3 of the final code

`cpu`, `gpuD` and `gpuE3` of binary 8d9eaa85 were compared with E10's kept outputs (gates 796116, 796090). Each comparison gives
60 identical. The remaining differences are:
- 3 files absent from E10's kept copies (that gate did not keep `*.fq` / `*Unmapped*`);
- `read_stats.tsv` and `star_Log.std.out` (timings);
- `provenance.json` (the whitelist path is under `bin-s5/` instead of `bin/`);
- `pileup_stats.json`'s `bytes_public` (+42 bytes from that path).

So the default CPU path, the default GPU path and level 3 are unchanged. 673cb317 changes only code that runs when level 5 is
requested.

Gate 796326 (a621c749, the guard bug) ran a `SINGLET_ENGINE_GPU_E2E=5` run that fell back to level 3. It gave 60 identical against
E10's CPU outputs, so the fallback path keeps the outputs too.

## 3. Timings

T = 18, short-gpu node g050 (H100 NVL), pbmc1k_v3 (66,601,887 read pairs). Wall time and core-s are for the whole
`singlet --pipeline` run (cgroup `cpu.stat`).
- cpu, gpuD, gpuE and gpuE3 come from gate 796377 (8d9eaa85).
- gpuE4 and the two level-5 runs come from gate 796433 (673cb317).
- gpuE4 in 796377 gave 213.5 s / 680.5 core-s: the two gates differ by ~4 s for the same switch.

E10's runs on node g051, for comparison: cpu 432.8 / 2617, gpuD 255.2 / 991, E8 250.2 / 943, level 3 254.2 / 940.5, level 4
214.7 / 676.

| run | switch | wall s | core-s | wall s/M | core-s/M | maxRSS |
|---|---|---|---|---|---|---|
| cpu | CPU engine | 424.5 | 2607.1 | 6.37 | 39.15 | 29.7 GB |
| gpuD | `--engine gpu` + `SINGLET_GPU_PILEUP=on` | 253.4 | 996.6 | 3.80 | 14.96 | 43.6 GB |
| gpuE | gpuD + `SINGLET_ENGINE_GPU_E2E=1` (E8) | 251.8 | 964.9 | 3.78 | 14.49 | 43.6 GB |
| gpuE3 | `=3` (E10 level 3) | 252.5 | 941.8 | 3.79 | 14.14 | 61.0 GB |
| gpuE4 | `=4` (STAR's loop skipped, Solo incomplete) | 209.5 | 672.4 | 3.15 | 10.10 | 61.0 GB |
| **gpuE5** | **`=5` (E11)** | **210.8** | **680.4** | **3.17** | **10.22** | 61.0 GB |
| **gpuE5b** | `=5`, second run | **210.3** | **677.5** | **3.16** | **10.17** | 61.0 GB |
| gpuE5 (796377, 8d9eaa85) | `=5` with the false-sharing counters | 250.3 | 1302.6 | | | 61.0 GB |

Level 5, as the mean of the two runs (210.6 s / 679.0 core-s), compared with:
- the default GPU path: −42.8 s (−16.9 %), −317.6 core-s (−31.9 %);
- E8: −41.2 s (−16.4 %), −285.9 core-s (−29.6 %);
- the CPU engine: 2.02x wall, 3.84x core-s;
- level 4 in the same job: +1.0 s, +6.6 core-s.

**Phase breakdown** (`bench/fable/solowire/s8_stages_e11.py`; wall s / core-s):

| stage | gpuD | gpuE (E8) | gpuE3 | gpuE4 | **gpuE5** | **gpuE5b** |
|---|---|---|---|---|---|---|
| decode .1fq → FASTQ + R11 histogram | 19 / 85.1 | 18 / 84.0 | 20 / 86.9 | 19 / 84.8 | 20 / 86.7 | 19 / 84.8 |
| probe/ref load, startup | 5 / 9.4 | 6 / 10.2 | 5 / 7.8 | 6 / 10.1 | 5 / 8.3 | 5 / 8.6 |
| STAR load, CR4 probe, start | 21 / 26.3 | 34 / 43.6 | 33 / 42.9 | 36 / 45.9 | 36 / 46.0 | 37 / 47.1 |
| GPU prefill (read, clip, seed, stitch, records, E10/E11 keep) | 58 / 224.8 | 57 / 268.0 | 60 / 297.4 | 57 / 292.8 | 58 / 297.2 | 57 / 296.9 |
| STAR per-read host pass | 51 / 341.6 | 39 / 286.5 | 39 / 241.2 | **9 / 16.9** | **9 / 20.7** | **9 / 19.9** |
| STAR stream into pileup (Solo, exit) | 18 / 54.4 | 22 / 64.6 | 22 / 64.4 | **8 / 16.6** | **8 / 15.6** | **9 / 17.0** |
| GPU-pileup hand-off + pileup | 23 / 80.0 | 17 / 31.7 | 16 / 27.3 | 17 / 30.5 | 17 / 31.3 | 17 / 34.0 |
| non-host | 20 / 44.1 | 20 / 44.1 | 20 / 47.2 | 20 / 45.3 | 20 / 44.8 | 19 / 40.5 |
| export, QC, callers | 17 / 35.0 | 17 / 35.6 | 17 / 35.6 | 17 / 35.5 | 17 / 35.6 | 18 / 37.0 |
| demux/ancestry/writers | 21 / 95.6 | 21 / 96.2 | 20 / 90.8 | 20 / 93.2 | 21 / 93.9 | 20 / 91.4 |
| total | 253 / 996.4 | 251 / 964.5 | 252 / 941.6 | 209 / 671.7 | 211 / 680.0 | 210 / 677.4 |

gpuD, gpuE and gpuE3 are from 796377; gpuE4, gpuE5 and gpuE5b are from 796433. The level-5 prefill is 58 s: reading reads 16 s, GPU 38 s, cache bookkeeping 4 s. The device is
held for 37.7 s.

**STAR's Solo step** ("started/finished Solo counting" in `star_Log.out`, 796377):

| run | Solo step |
|---|---|
| cpu | 43 s |
| gpuD | 14 s |
| gpuE | 13 s |
| gpuE3 | 13 s |
| gpuE4 | 1 s (it counts only the 157 k reads) |
| gpuE5 | not run |

In its place, level 5 spends 0.81 s between "finished mapping" and STAR's exit: gather 5 ms, `soloRunKept` 0.15 s, write 0.65 s.

The step is not the only saving. Inside the loop at E8, STAR spent 31 thread-s on barcode output (`outBarcode`) and 26 thread-s on the
Solo record (`outSolo`) (E10's profile). Level 4 already removed those but lost the Solo outputs; level 5 keeps the saving and the
outputs. In the same job, level 5 is 0.8–1.3 s and 5–8 core-s above level 4. That is the decode's extra counts and the RNA
adjustment (about 4 core-s in the prefill), plus 0.8 s of Solo after the loop against level 4's 1 s Solo step on 157 k reads.

**Step 4a (no serial zero-fill of the decode's arrays: E10 aux 2.1 GB, name-line offsets 2 × 0.5 GB):** no measurable effect.
- "reading reads" is 16–17 s at levels 4–5, against E10's 17–18 s.
- The prefill stage is 58–59 s at both.

The zero-fill was a few hundred ms of one core. Its page faults now land in the parsing threads instead.

### 3a. Device times (CUDA events, H100 NVL)

| step | where | time |
|---|---|---|
| keep the reads' CR/UR (strided gather out of E10's aux, 66.6 M reads) | `mapDistinct`, `soloKeepReads` | 3.4–3.5 ms |
| keep the records' Gene annotation (31.6 M records) + read → record map | `mapDistinct`, `soloKeepRecords` | 37.7–37.8 ms |
| memory held through STAR's loop | | 1.59 GB (~8 s at level 5) |
| `soloRunKept`: upload (model, whitelist, 157 k host annotations) | after the loop | 12.9–13.1 ms |
| gene (records → reads) / cb (whitelist, posterior) / reads (rules, keys) | | 1.9 / 15.0–15.1 / 13.9 ms |
| sort / 1MM_All collapse / compaction | | 7.8–7.9 / 1.6 / 0.5 ms |
| download (counts, stats, matrices) | | 85–89 ms |
| `soloRunKept` total | | 148–151 ms |
| `soloWrite` (host: CellRanger2.2 filtering, 11 files) | | 0.65 s |
| RNA Q30 adjustment (host, 18 threads, 4.3 M quality lines) | prefill | 0.21–0.24 s |
| waiting for the device after the loop | | 30–37 µs |

Counts:
- 903,105 distinct CRs, with 707,240 whitelist candidates (exact and one-mismatch);
- 36,678,568 counted reads;
- 11,500,780 distinct (CB, gene, UMI);
- 3,686,799 (CB, gene);
- 333,210 CBs;
- 1,136 cells.

The download dominates `soloRunKept` (86–89 of 148–151 ms). The kernels take about 41 ms and the upload 13 ms.

## 4. The II.8 path at level 5 (hops of `GPU_ALIGNER_STUDY_2026-10-31.md` §II.8)

| hop | exists in the engine | measured (pbmc1k_v3, T = 18, level 5) |
|---|---|---|
| 1. `.1fq` blocks from disk | yes, but the parent decodes `.1fq` into FASTQ text files for STAR (and computes R11) | 20 s, 86.7 core-s |
| 2. decode + pack on the CPU | partly: STAR's prefill parses the FASTQ text into sequences, quality bits, CR/UR and Q30 counts; the pinned 2-bit ring is still only the M8 prototype | "reading reads" 16 s (incl. the barcode file 0.8 s) |
| 3. H2D | yes, pageable | index 3.1 s; reads + clip in "upload 5.8 s"; aux 0.39 s; Solo model 13 ms |
| 4. clip + dedup | yes | clip 1.15 s; dedup 3.7 s (66.6 M → 31.7 M distinct) |
| 5. mapping | yes | 15.2 s GPU + host lane 2.5 s (85,570 sequences) |
| 6. records | yes (E6/E8) | gather 178 ms, buildRecords 39 ms; region copy 0.83 s |
| 7. per-read expansion | yes (E10 entries 20 ms, products 54 ms; E8 hand-off pass in the parent 3.4 s incl. transfers; **E11: Solo's per-read CB/rule/key stages 29 ms**) | STAR's loop: 156,931 reads (0.24 %), 9 s / 20.7 core-s |
| 8. reducers | yes: GPU pileup (2.1 s GPU, lock held 8.8 s); **E11: STARsolo Gene counting (sort + collapse + compaction 10 ms)**; STAR's Solo step gone | pileup stage 17 s / 31 core-s |
| 9. D2H of results | yes | Solo 89 ms; pileup tallies in the 8.8 s |
| 10. callers and export | CPU (Solo's CellRanger2.2 filtering and writers: 0.65 s on the host) | non-host 20 s, export/QC/callers 17 s, demux/ancestry/writers 21 s |

What is still temporary or done twice at level 5:
- the FASTQ text files (hops 1–2);
- the shared region with the per-read entries, masks and records (5.3 GB): the records cross the host twice, device → region →
  parent → device;
- STAR's BAM stream, for the 157 k reads it maps.

The GPU is busy 21 s of the 211 s.

## 5. What remains, and next steps

1. **A per-read posterior, so that level 5 runs on most datasets.**
   - E9's posterior has one `pq`, which is exact only when every CB quality gives the same `qin`. With binned NovaSeq qualities
     (F, :, ,), `qin` is 33, 25 or 11 per base, and level 5 falls back to level 3.
   - Only reads whose CR matches several whitelist entries with one mismatch need the quality: for each candidate, the quality at the
     mismatch position.
   - Two ways to do it:
     - keep the CB quality string on the device (16 B per read, 1.07 GB for this sample, or 2 bits per base after binning);
     - after the device's CB stage, download the multi-match reads and compute their posterior on the host from R1's quality lines,
       which are still mapped then.

   Gate: a NovaSeq-binned 3' sample (for example one of the release-gate datasets) at level 5 against the CPU engine.
2. **Other Solo configurations** fall back to level 3 today: multimappers, EmptyDrops_CR, GeneFull/Velocyto, other CB matching
   types (`1MM_multi` without pseudocounts, `Exact`). Each needs its own port. 5' runs pass the guard (E9's model takes `soloStrand`)
   but are untested at level 5; a 5' sample (pbmc5p_v1, CR4 TSO clip) is the next gate.
3. **The device during STAR's loop.**
   - Level 5 leaves 1.59 GB allocated on the device for ~8 s while the lock is free. It then takes a slot of the same device for
     0.15 s (`lockDeviceSlot`).
   - With several samples per GPU, another sample's prefill sees 1.6 GB less memory. Copying the kept inputs to pinned host memory
     (0.13 s) would avoid that.
   - The parent's GPU pileup and e2ePass use the same lock files. They never wait for STAR, so there is no deadlock. In these runs
     the wait was 30 µs.
4. **E10's level-4 fallback (pre-existing).** At level 4, `e10NoSum` is decided before `e10BuildSource`. If the read-source build then
   fails, the run continues at level 3 without the summaries, and reads with a record would be treated as unmapped. At level 5 this
   case exits with an error instead; level 4 should do the same, or decide `e10NoSum` after the source is built.
5. **Not done from the optional step 4:**
   - **Device R11 histogram to the parent.** It needs the stream's CB correction to wait for STAR's prefill, which means a change in
     `src/pipeline` beyond one line (the histogram in E8's region header, read by the parent after the prefill).
   - **Releasing the device during the host lane** (2.5 s): the records are built after the host lane, so the release needs the host
     lane's results uploaded separately. It is a multi-sample throughput item, not a single-sample wall item.
   - **The first-touch change (4a)** is in, with no measurable effect (§3).
6. **Where the 211 s go now:**

   | piece | wall s | notes |
   |---|---|---|
   | `.1fq` decode | 20 | |
   | probe/ref load | 5 | |
   | CR4 probe and STAR start | 36 | about one core |
   | prefill | 58 | GPU 38 |
   | STAR's loop on 157 k reads | 9 | |
   | stream/exit | 8 | |
   | hand-off and pileup | 17 | |
   | non-host | 20 | |
   | export and callers | 38 | |

   These are the persistent-service and host-track items of II.1/II.11: the CR4 probe, index load and upload per sample, FASTQ text,
   and non-host's index load. They are not kernels.

## 6. Files touched outside `src/engine_gpu/`, and why

| file | change | why |
|---|---|---|
| `include/singlet/engine_gpu.h` | `e2eLevel()` accepts 5; `Input::E10::soloModel/soloParams`; `Output::E10::soloKept`; `lockDeviceSlot(device)`; `NoInitAlloc` / `NoInitVector` (and `Input::E10::aux` uses it) | in scope |
| `src/star/SeqCacheGpuPrefill.cpp` | `e11Config`, `e11Posterior`, `e11RnaAdjust`; the decode's Q30 and CB-quality counts; the level-5 decision and fallbacks; `seqCacheGpuAfterMapping`'s Solo block (gather, `soloRunKept`, `soloWrite`, Summary.csv inputs); step 4a | in scope |
| `src/star/ReadAlign.h` | `e11Ann`, `e11Annot` | the member the file below fills |
| `src/star/ReadAlign_outputAlignments.cpp` | `ReadAlign::e11Annot`; at level 5 the three `soloRead->record` calls (e10Skip branch, normal path, summary replay) record the annotation instead | in scope (`ReadAlign*.cpp`) |
| `src/star/STAR.cpp` | `soloMain.processAndOutput()` is skipped when `g_e11SoloDone` | STAR's Solo step must not overwrite `star_Solo.out/`; one line, does nothing unless level 5 ran |
| `CMakeLists.txt` | `src/engine_gpu/solo_gpu.cu` in `singlet_engine_gpu` (with `--extended-lambda`, OpenMP for its host path) | E9's library was built only by its bench driver |
| `bench/fable/solowire/jobs/{build,gate}.sbatch`, `bench/fable/solowire/s8_stages_e11.py` | build and gate jobs; the stage script with a 4 GB prefill-end marker (level 5 keeps 1.6 GB on the device through the loop) | deliverable |
| `docs/gpu-engine/fable/E11-solowire.md` | this report | deliverable |

Not changed:
- no `Solo*.cpp` (STAR's Solo code is untouched; its constructor still runs);
- nothing in `src/pipeline` or `include/singlet/pileup`.

Inside `src/engine_gpu/`:
- `engine_gpu.cu`: the two keep calls in `mapDistinct`, and their release on every failure path.
- `solo_gpu.{h,cu}`:
  - the keep API and `soloRunKept`;
  - device-resident inputs used in place;
  - sparse host annotations;
  - the per-record annotation input;
  - `Gene/Summary.csv` in `soloWrite`.

## 7. Jobs, commits, how to run

**Branch `gpu/fable/solowire`:**
- a621c749: wiring.
- cf309edd: gate job and stage script.
- b5fa681c: step 4a.
- 8d9eaa85: the multimapper guard fix.
- 673cb317: the thread-local counters.
- fc7bb650: gate mode `gpuE5b`.
- This document is the next commit.

**Builds** (short-gpu, no GPU, ~270 s each):

| job | commit | output |
|---|---|---|
| 796309 | a621c749 | `bin/` |
| 796333 | b5fa681c | `bin-s4/` |
| 796376 | 8d9eaa85 | `bin-s5/` |
| 796412 | 673cb317 | `bin-s6/` |

**Gates** (short-gpu, H100 NVL, T = 18, node g050):

| job | commit | modes | result |
|---|---|---|---|
| 796326 | a621c749 | `gpuE5` against E10's cpu | level 5 fell back to level 3 (guard bug) |
| 796358 | | | cancelled after 1 min (same bug) |
| 796377 | 8d9eaa85 | `gpuE5 gpuE4 gpuE3 gpuE gpuD cpu` plus 3 comparisons against E10's kept outputs | identity, default paths, baselines |
| 796433 | 673cb317 | `gpuE5 gpuE4 gpuE5b` | level-5 timing |

**Results:**
- `$W/jobs/fable/solowire/results/gate-<job>/`, containing `summary.txt`, `equiv-*.txt`, and per run `stages.txt`,
  `starlog_engine.txt`, `star_Log.out`, `CellReads.stats`, `monitor.tsv` and `out/`.
- Logs: `$W/logs/e11-{gate,build}-<job>.out`.

**Run:** build with `sbatch --export=ALL,OUTD=<dir> bench/fable/solowire/jobs/build.sbatch`, then run
`sbatch --export=ALL,BINDIR=<dir>,MODES="cpu gpuE5",PAIRS="cpu:gpuE5" bench/fable/solowire/jobs/gate.sbatch`.

**Switch:** `SINGLET_ENGINE_GPU_E2E=5` (with `--engine gpu`, `SINGLET_GPU_PILEUP=on`). `star_Log.out` reports the outcome:
- "ENGINE_GPU: E11 decode: …" and "ENGINE_GPU: E11: Solo on the device: …" when level 5 ran;
- "ENGINE_GPU: E11 (level 5) not used: <why>" when it fell back to level 3.
