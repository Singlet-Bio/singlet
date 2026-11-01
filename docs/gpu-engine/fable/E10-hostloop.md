# E10: STAR's per-read host loop on the device (hostloop)

Experiment E10 ("hostloop") of wave 2b of the GPU aligner program. Branch `gpu/fable/hostloop`, from `gpu/fable-engine` at 7ed4e9f0
(the E8 merge). Everything new runs only with `SINGLET_ENGINE_GPU_E2E` set to 2, 3 or 4 (with `--engine gpu` and
`SINGLET_GPU_PILEUP=on`). When the variable is unset or 1, the binary runs the code it ran before: gate 796116 (§3d) checks the CPU
engine and the default GPU path of the final binary against gate 796069.

## 0. Answer in brief

- **Profile (§1).** At T = 18 the brief's "167 s" (STAR from its start to the end of its loop) is 134 s:
  - CR4 probe and STAR start: 32 s at about one core;
  - index load: 9 s;
  - prefill: 58 s;
  - STAR's per-read loop: 33 s.

  Inside the loop, 28.5 of the 32.7 s are `chunkRead`, which parses FASTQ input one thread at a time under the input mutex. The per-read
  work (204 thread-s over 18 threads) is not on the critical path. So moving per-read work to the device saves core-s, but it does not
  save wall time unless STAR also stops reading those reads.

  The "58 s" (outputs) is still 58 s at T = 18:
  - Solo counting and STAR's exit: 16 s;
  - hand-off pass: 6 s;
  - GPU pileup: 16 s;
  - non-host: 20 s.
- **Step 2 (hand-off entries on the device): done and byte-identical.** The device computes `HandoffReadEntry {cr3, ur3, lowQual,
  qualNeqN, flags}` and the low-quality mask of every read. It uses `handoffReadEntry`'s rule on R2's quality bits and R1's CR/UR, which
  the prefill's decode packs. The R11 exact-CB histogram comes from the same keys. Gate 796090, run `gpuE2c`:
  - 66,444,956 reads with a record: entries differ 0, masks differ 0;
  - STAR wrote an entry for every one of them.

  STAR's threads wrote their own entries to a side buffer for this comparison.
  - The R11 histogram equals `bench/fable/records/r11_hist.cpp`'s byte for byte: 4,835,008 bytes, 63,169,750 reads, 302,186 barcodes.
  - Device time: entries 20–24 ms; R11 2.8 ms.
- **Step 3 (Log.final.out, SJ.out.tab, unmapped-read list on the device): done and identical.**
  - At level 3 and 4, `star_Log.final.out`'s statistics are identical to E8's.
  - `star_SJ.out.tab` is byte-identical (126,996 junctions).
  - `star_Unmapped.out.mate1/2` hold the same records (3,355,559 reads, sorted sha256). Their order is free.
  - So does `nonhost_reads.fq`.
  - Device time: 53–56 ms for the counters, 2,045,387 junction entries and the list of 3,301,393 unmapped reads.
- **Step 4 (the skip switch).**
  - **Level 3** (all but Solo skipped): every output is identical to E8. It saves 31 core-s in STAR's loop, but no wall time, because the
    loop is still bound by `chunkRead`.
  - **Level 4** (full skip): STAR's input holds only the 156,931 reads without a GPU record. The run takes 214.7 s instead of 250.2 s
    (−35.5 s, −14 %) and 676 core-s instead of 943 (−267 core-s, −28 %). The CPU engine takes 432.8 s and 2617 core-s.

    All count outputs are identical, as are Log.final.out, SJ.out.tab, Unmapped.out and non-host. The exceptions are STAR Solo's files
    and the two outputs that read Solo's stats (`cell_qc_metrics.tsv`, `read_qc_summary.json`): Solo does not see the skipped reads.
    E9's `soloRun` is the piece that removes that limit (§5).
- **Cost.** At T = 18, the E10 decode in the prefill adds 3–5 s and about 30 core-s:
  - 2.6 GB of zero-filled per-read words;
  - the quality and ACGTN pass;
  - R1's parse;
  - 1.2 s of transfers.

  So level 2 alone is slightly slower than E8: +3 to +8 s, +25 core-s. Peak RSS rises from 43.6 to 61.0 GB because R2 and R1 stay
  mapped until the loop ends.
- **Not done (§5).**
  - E9's Solo wired into level 4 (the coordinator's note; left to a separate experiment as offered).
  - Releasing the device during the host lane.
  - Feeding the device's R11 histogram to the parent.
  - Pinned read uploads: the profile shows the read upload is about 1 s, so they are not worth doing.
## 1. Profile: where the 167 s and the 58 s go

Gate 796069 (T = 18, short-gpu, g051, H100 NVL), run `prof`: the E8 path (`SINGLET_ENGINE_GPU_E2E=1`, `SINGLET_GPU_PILEUP=on`) built
with `-DSTAR_PROFILE` (55acf4e8: STAR's loop counters plus a counter for the hand-off entry and one per read path), with s8_stages.py
and the cgroup sampler. The profiled run takes 256.1 s and 1024.7 core-s; the unprofiled run of the same commit (`gpuE`) takes 255.3 s
and 950.9 core-s. At T = 18, the brief's "167 s" (STAR's index load, cache fill and loop at T = 8) is 134 s, and its "58 s" (outputs)
is 58 s.

### 1a. STAR, from its start to the end of its loop (134 s wall)

Wall times from the run log's timestamps (STAR started at +26 s, CR4 probe done at +58 s, index loaded at +68 s, loop ended at
+160 s); core-s from s8_stages.py.


| piece | wall s | core-s | what it is | what it feeds |
|---|---|---|---|---|
| CR4 probe and STAR start | 32 | 47.5 (both rows; stage "STAR load, CR4 probe, start") | two probe alignments of 200,000 reads, with and without the Beta profile; about one core | the aligner profile choice |
| index load | 9 | | Genome, SA and SAindex (17.6 GB) from the page cache | everything |
| prefill: read the reads | 15 | | R2 into memory (`readSequences`) | the device |
| prefill: GPU part | 37 | | index upload 3.2, read upload and clip ~2, dedup 3.6, GPU mapping 15.2, host lane 2.5, E8 records 1 | records |
| prefill: summaries and cache inserts | 6 | | per-record summaries (3 s), cache inserts of the 2.84 M unmapped records | the loop's replay |
| (prefill total) | 58 | 270.2 | | |
| STAR loop | 32.7 | 340.4 | 18 threads, 66.6 M reads: 63.1 M summary replays, 3.30 M cache hits, 157 K mapped (N reads) | see 1b |

### 1b. Inside the STAR loop (32.7 s wall, 18 threads = 588 thread-s)

| counter | thread-s | what it is | what it feeds |
|---|---|---|---|
| chunkRead | 28.5 | reading FASTQ chunks into the thread's buffer, **one thread at a time** under the input mutex | everything; this is the loop's critical path (28.5 of 32.7 s) |
| chunkMutexWait | 231.2 | threads waiting for the input mutex | (idle) |
| readLoad | 51.8 | parsing each read of the chunk (both mates) | everything |
| map | 24.5 | cache lookup and restore for the 3.30 M reads of unmapped records; STAR's mapping of the 157 K reads with N (seed 2.5, windows 2.1, stitch 2.3) | Log.final.out, Unmapped.out |
| outMapped | 11.4 | replay of the 63.1 M hand-off summaries | Log.final.out counters, SJ.out.tab, Solo's gene features |
| outBarcode | 30.9 | `getCBandUMI` (CB/UMI from R1) | Solo, the hand-off entries |
| outSolo | 25.9 | `soloRead->record` | Solo matrices |
| outEntry | 24.9 | the hand-off entry and low-quality mask of each read | the GPU pileup (E8 region) |
| outUnmapped | 3.1 | Unmapped.out.mate1/2 records | non-host |
| outSAM | 0.2 | BAM records of the 157 K mapped reads | the pileup's stream |
| other output | 31.8 | the rest of `outputAlignments` | |

Read paths: 63,143,563 summary replays, 3,301,393 cache hits, 156,931 mapped by STAR (no host-lane failures in this sample).

The wall time of the loop is set by chunkRead: 28.5 s of input parsing that runs one thread at a time. The per-read work
(readLoad + map + output = 204 thread-s) is spread over 18 threads, so moving any part of it to the device frees core-s but
does not shorten the loop unless the input is skipped too.

### 1c. After the loop (58 s wall)

| piece | wall s | what it is | what it feeds |
|---|---|---|---|
| Solo counting and STAR exit | 16 | STAR Solo's collapse and matrix output (14 s), drain of the BAM pipe, exit | Solo matrices (E9's area) |
| hand-off pass | 6 | 63.1 M entries from the region, rRNA detector on 0.93 M sampled reads | the GPU pileup |
| GPU pileup | 16 | build input 5.2 s, pileupRun 9.0 s, host pass 1.1 s | count outputs |
| non-host | 20 | screening the reads of Unmapped.out | nonhost_* |
## 2. What moved to the device, and the switch

All of it runs only with `--engine gpu`, `SINGLET_GPU_PILEUP=on` and `SINGLET_ENGINE_GPU_E2E` at 2 or more. At 0 or 1 the binary runs
the code it ran before: the new code is reached only through `e10Level >= 2` in the prefill, through `g_e10Level` (0 unless the prefill
set it to 3 or 4) in STAR's loop, and through `in.e10.on` in the engine.

| level | what the device does on top of E8 | what STAR's loop still does for a read with a GPU record |
|---|---|---|
| `=1` (E8) | records; the parent's hand-off pass | everything: readLoad, cache lookup or summary replay, stats, junctions, Unmapped.out, barcode, Solo record, hand-off entry |
| `=2` | **the hand-off entries and low-quality masks** of every read (E8's plan step 2), from R2's quality bits and R1's CR/UR packed by the host decode; the list of hook-sampled reads; **R11's exact CB histogram**; the loop products of step 3 (computed and logged, not used) | the same minus the hand-off entry (`outEntry`): STAR's threads no longer write the 66.4 M entries and masks |
| `=3` | also **Log.final.out's counters**, **SJ.out.tab's junctions** and **the list of Unmapped.out reads** for the reads with a record; the host writes their Unmapped.out records from the mapped R2/R1 files before the loop | readLoad, the barcode and the Solo record only: no cache lookup, no summary replay, no stats, no junctions, no Unmapped.out record |
| `=4` | the same | **nothing**: STAR's input holds only the 157 K reads without a GPU record (reads with N, host-lane failures). STAR Solo does not see the other 66.4 M reads, so its matrices are not usable (the brief's "Solo outputs ignored") |

Reads without a GPU record (157 K reads with N in this sample; host-lane failures in general) go through STAR's path at every level.

### 2a. Step 2: the hand-off entries on the device

- **Host decode (prefill).** `readSequences` already reads R2 into memory. Under E10 it also records each read's name-line offset and
  packs, per read, two quality words (one bit per base: `Q < minQ` after the Phred offset; read length up to 256), and checks that every
  cDNA base is uppercase ACGTN. `e10Barcodes` maps R1 and packs CR and UR as 3-bit codes into two words per read. Cost: 3 s on top of
  `readSequences`'s 15 s and 0.96 s for R1 at T = 18.
- **Device (`k_entries`, `hostloop_gpu.cu`).** One thread per read. It applies `handoffReadEntry`'s rule from the codes and quality bits:
  `lowQual` and `qualNeqN` over the original length; the mask over the clipped span (or flag 64); the clip flags from the device clip
  (`c5 = cs - off`, `c3 = L - cl - c5`, as `out.clip5/3`); the poly-A count; bit 2; and the hook bit 4 (FNV `hash_qname("r" + i) & 63 == 0`,
  clipped length ≥ 21). The host detector's bits 8 and 16 are set on the host for the 65,255 hook reads only (their sequence rebuilt from
  the device codes, through `g_handoffReadHook`).
- **Into the region.** The entries (24 B per read) and masks (16 B per read) are copied once from the device into E8's shared memory
  region (page-locked, 855 ms for 2.6 GB); the reads without a record are zero, as STAR leaves them.
- **R11 (`k_r11`).** From the same CR/UR codes: a UMI with no N that is not a homopolymer, an N-free CB that is an exact entry of STAR's
  sorted 2-bit whitelist (`ParametersSolo::cbWL`, which packs the same way as `CbCorrector`, so the indices are the same), counted by
  `atomicAdd`.

### 2b. Step 3: Log.final.out, SJ.out.tab and Unmapped.out

From the E8 record blobs, which are still on the device after `buildRecords`, and the read → record map:

- **Multiplicity (`k_mult`).** The reads per record.
- **Counters (`k_rec`).** One thread per record. It decodes the blob (nW, nTr, the best transcript and its exons) and applies
  `mappedFilter` as `recordCount` does. Each counter of `Stats` (readN, readBases, mappedReadsU/M, transcriptStats of transcript 0:
  mismatches, insertions, deletions and their lengths, splices by motif and annotation, mapped length, chimeric, too short, other) is
  multiplied by the multiplicity, reduced with warp shuffles and added into one device array.
- **Junctions (`k_sjfill` + thrust).** Every junction of every alignment that `recordSJ` keeps (`outSJfilterReads All`, or
  `nTr == 1`; `canonSJ >= 0`) becomes a key `start << 32 | gap` with its record, overhang (`min(L, Lnext)`), motif and annotation.
  A stable sort, a max-reduce per (junction, record) (STAR's in-read duplicate rule), then a reduce per junction gives the unique and
  multi counts (the multiplicity split by `nTr == 1`) and the max overhang. The host formats STAR's 27-byte junction records, and after
  the loop appends them to chunk 0's junction buffer and calls `collapseSJ`. STAR's own output step then sorts and filters as before.
- **Unmapped list (`copy_if`).** The reads whose record is unmapped. The host formats their Unmapped.out records, from R2/R1 kept mapped
  since the decode, as `ReadAlign::outputAlignments` does (name, `0:N` or `0:Y` filter flag, mate tag), and writes them before the loop.

Device times on the H100 NVL (CUDA events, gate 796090, run `gpuE2c`): entries 23.7 ms for 66.6 M reads; R11 histogram 2.8 ms;
products 53.9 ms (2,045,387 junction entries → 198,459 junctions, 3,301,393 unmapped reads, 65,255 hook reads).
Transfers: aux upload (CR/UR + quality words, 2.1 GB) 378 ms; entries and masks into the region 855 ms.

### 2c. Level 4: STAR's input without the reads that have a record

`StarReadSource` (the hook STAR's `chunkRead` uses for `.1fq` streams) is given a source that yields only the reads without a record,
formatted from the mapped R2/R1 files in STAR's internal chunk format, with each read's original `iReadAll`. So the BAM stream, the
Unmapped.out records and the counters of these reads are the same as at level 3.
## 3. Identity per output

Gate 796090 (T = 18, short-gpu, g051, H100 NVL, binaries at 8cddfa29). Each E10 run is compared with E8 (`gpuE`) in the same job. E8
was compared with the CPU engine in gate 796069: Log.final.out identical; SJ.out.tab identical; the same Unmapped.out and non-host
records; and the count outputs identical except for the known differences that E8-e2e.md §0 lists. Tools: `out_equiv.py` over every
output file; a byte comparison of `star_Log.final.out` from line 5 on (the timing lines excluded) and of `star_SJ.out.tab`; and a sorted
sha256 of the 4-line records of `nonhost_reads.fq` and `star_Unmapped.out.mate1/2`.

### 3a. Step 2: the hand-off entries and masks, and R11

| check | run | result |
|---|---|---|
| entries (24 B) of every read with a record, device vs STAR's threads | `gpuE2c` (`SINGLET_ENGINE_GPU_E10_CHECK=1`) | 66,444,956 reads: **0 differ**; STAR wrote an entry for all of them |
| low-quality masks (16 B) of the same reads | `gpuE2c` | **0 differ** |
| R11 exact-CB histogram vs `r11_hist.cpp` (`$W/jobs/fable/records/results/r11_exact_hist.bin`) | `gpuE2c` (`SINGLET_ENGINE_GPU_E10_R11`) | **byte-identical**, 4,835,008 bytes (63,169,750 reads, 302,186 barcodes) |
| GPU pileup input digest | all runs | `d2aa9309…` in every run (E8, E10 levels 2–4) |

In check mode, the region holds the device's entries and masks; STAR's threads write theirs into a side buffer, and
`seqCacheGpuAfterMapping` compares the two byte for byte. Reads without a record (the 156,931 reads with N) keep STAR's path:
their entries are STAR's at every level.

### 3b. Step 3 and the skip levels against E8 (`gpuE`)

| output | level 2 (`gpuE2`, `gpuE2c`) | level 3 (`gpuE3`) | level 4 (`gpuE4`) |
|---|---|---|---|
| `star_Log.final.out` statistics | identical | identical (device counters) | identical (device counters + STAR's for 156,931 reads) |
| `star_SJ.out.tab` | identical (126,996) | identical (198,459 device junctions merged before STAR's filters) | identical |
| `star_Unmapped.out.mate1/2` | same records, 3,355,559 reads | same records (3,301,393 written from the device's list, 54,166 by STAR) | same records |
| `nonhost_reads.fq` | same records, 428,468 | same records | same records |
| count outputs (`exon_counts`, `intron_counts`, `sj_counts`, `snp_*`, `splice_psi`, `cell_calls.tsv`, …) | identical | identical | identical |
| STAR Solo (`star_Solo.out/…`) | identical | identical | **differ**: Solo sees only the 156,931 reads STAR read (expected; "Solo outputs ignored") |
| `cell_qc_metrics.tsv`, `read_qc_summary.json` | identical | identical | **differ**: they read STAR Solo's stats (`barcodes_in_star_stats`, `cb_not_in_passlist_*`) |
| files compared, identical / timing-only / differ | 60 / 3 / 6 | 60 / 3 / 6 | 51 / 3 / 15 |

The 6 differences at levels 2 and 3 are the ones the E8 runs also show between themselves:
- read order in the three FASTQ outputs (same records);
- `read_stats.tsv` and `star_Log.std.out` timings;
- the output byte count in `pileup_stats.json`, because `star_Log.out` has longer E10 log lines.

At level 4 the 9 extra differences are:
- 7 Solo files (`Barcodes.stats`, `Gene/Features.stats`, `Gene/Summary.csv`, `Gene/UMIperCellSorted.txt`,
  `Gene/filtered/barcodes.tsv`, `Gene/filtered/matrix.mtx`, `Gene/raw/matrix.mtx`);
- `cell_qc_metrics.tsv` and `read_qc_summary.json`.

`Gene/raw/barcodes.tsv` and `features.tsv` stay identical (whitelist and gene list).

### 3d. The default paths of the final binary

Gate 796116 runs the CPU engine (`cpu`) and the default GPU path (`gpuD`) with the 8cddfa29 binary and compares them with 796069's
runs of 55acf4e8. 55acf4e8 holds only the profile counters, which are compiled out of the gated binary.

Both runs are unchanged, with 60 output files identical in each comparison. `star_Log.final.out`'s statistics and `star_SJ.out.tab`
(126,996 junctions) are identical, and no E10 line is logged. The differences are:
- the timing files (`read_stats.tsv`, `star_Log.std.out`);
- the output byte count in `pileup_stats.json` (3 bytes, the log sizes);
- the three FASTQ outputs, which 796069 did not keep.

The new runs took 436.4 s / 2635.9 core-s (cpu) and 254.0 s / 996.4 core-s (gpuD).
## 4. Timings

T = 18, short-gpu node g051 (H100 NVL), pbmc1k_v3 (66,601,887 read pairs), wall and core-s of the whole `singlet --pipeline` run
(cgroup `cpu.stat`). cpu, gpuD and the first gpuE come from gate 796069 (commit 55acf4e8). The others come from gate 796090
(8cddfa29). The two gpuE runs show the run-to-run spread: 250.2 vs 255.3 s.

| run | switch | wall s | core-s | wall s/M | core-s/M |
|---|---|---|---|---|---|
| cpu | CPU engine | 432.8 | 2616.6 | 6.50 | 39.29 |
| gpuD | `--engine gpu` + `SINGLET_GPU_PILEUP=on` | 255.2 | 990.6 | 3.83 | 14.87 |
| gpuE (796069) | gpuD + `SINGLET_ENGINE_GPU_E2E=1` (E8) | 255.3 | 950.9 | 3.83 | 14.28 |
| gpuE (796090) | same | 250.2 | 943.0 | 3.76 | 14.16 |
| gpuE2 | `=2`: entries, masks, R11 on the device | 257.9 | 968.6 | 3.87 | 14.54 |
| gpuE2c | `=2` + the entry check | 258.1 | 971.7 | 3.88 | 14.59 |
| gpuE3 | `=3`: + Log.final.out, SJ, Unmapped.out | 254.2 | 940.5 | 3.82 | 14.12 |
| **gpuE4** | `=4`: + reads with a record skip STAR's loop (Solo ignored) | **214.7** | **676.0** | **3.22** | **10.15** |
| prof / prof3 / prof4 | `=1` / `=3` / `=4`, `-DSTAR_PROFILE` binary | 256.1 / 257.2 / 216.0 | 1024.7 / 1016.4 / 676.8 | | |

Stages (s8_stages.py; wall s / core-s):

| stage | gpuE | gpuE2 | gpuE3 | gpuE4 |
|---|---|---|---|---|
| decode .1fq → FASTQ + R11 histogram | 19 / 85.5 | 18 / 83.7 | 18 / 83.4 | 21 / 87.3 |
| STAR load, CR4 probe, start | 34 / 43.9 | 36 / 46.0 | 35 / 44.2 | 38 / 47.5 |
| GPU prefill | 57 / 267.4 | 62 / 297.5 | 61 / 295.6 | 59 / 293.8 |
| STAR per-read host pass | 38 / 270.4 | 38 / 257.5 | 38 / 239.7 | **8 / 16.1** |
| STAR stream into pileup (Solo, exit) | 23 / 65.9 | 23 / 66.4 | 22 / 64.2 | **9 / 17.7** |
| GPU-pileup hand-off | 16 / 23.1 | 16 / 21.8 | 16 / 24.4 | 16 / 27.6 |
| non-host, export, demux | 56 / 168.7 | 57 / 175.0 | 58 / 177.8 | 58 / 176.0 |
| startup, probe/ref load, pileup + collapse | 6 / 17.5 | 7 / 19.7 | 6 / 10.6 | 5 / 9.3 |
| total | 250 / 942.4 | 257 / 967.6 | 254 / 940.0 | 214 / 675.4 |

STAR's loop counters (thread-s; `-DSTAR_PROFILE`):

| counter | prof (E8) | prof3 | prof4 |
|---|---|---|---|
| loop wall | 32.7 s | 33.4 s | 5.8 s |
| chunkRead | 28.5 | 29.9 | 0.01 |
| readLoad | 51.8 | 52.4 | 0.17 |
| map | 24.5 | 20.0 | 5.3 |
| outMapped (summary replay) | 11.4 | 12.7 | 0.14 |
| outBarcode | 30.9 | 28.4 | 0.13 |
| outSolo | 25.9 | 24.0 | 0.08 |
| outUnmapped | 3.1 | 0.08 | 0.05 |
| outEntry | 24.9 | 0.58 | 0.001 |
| reads by path: summary / cache hit / STAR-mapped | 63,143,563 / 3,301,393 / 156,931 | same | 0 / 0 / 156,931 |

At level 3 the summary replay stays (`outMapped`): STAR Solo's record needs the read's alignments, which the summary carries.

### 4a. Device times (CUDA events, H100 NVL, 66.6 M reads, 31.6 M records)

| step | kernel(s) | time |
|---|---|---|
| hand-off entries and masks | `k_entries` (one thread per read) | 20.2–23.7 ms |
| R11 exact-CB histogram | `k_r11` (binary search in the 3.69 M whitelist, `atomicAdd`) | 2.8 ms |
| products | `k_mult`, `k_zero`, `k_rec` (counters, warp-shuffle reduction), `k_sjfill`, thrust `stable_sort_by_key` + 2 × `reduce_by_key` + `k_sjcounts`, 2 × `count_if`/`copy_if` | 53.3–55.8 ms in all |
| aux upload (CR/UR + 2 quality words per read, 2.1 GB, pageable) | | 367–378 ms |
| entries and masks to the region (2.7 GB, page-locked) | | 768–855 ms |
| device held by the prefill | | 36.3–36.6 s (E8) → 37.7–38.3 s (E10) |

The prefill's host side under E10 (T = 18):
- `readSequences` + `e10Barcodes`: 17–18 s instead of 15 s, with R1's parse taking 0.93–0.97 s;
- the level 3/4 block (Stats, the 3.30 M Unmapped.out records written, junctions saved, level 4's read source): 0.88–1.14 s.
## 5. What remains on the host, and next steps

At level 4, for a read with a GPU record, the host still does the following:
- the parent's decode of the `.1fq` (19–21 s, 85 core-s), which also computes R11's histogram for the parent's CB correction;
- STAR's prefill reading R2 and R1 into memory (17–18 s at T = 18);
- the cache-free record bookkeeping;
- writing its Unmapped.out record, if it has one;
- the parent's pass over the region (hand-off pass, pileup input build, non-host, export and writers).

STAR's per-read loop no longer sees the read.

What remains of the 134 s and the 58 s at level 4 (T = 18):

| piece | wall s | note |
|---|---|---|
| CR4 probe and STAR start | ~32 | about one core; now the largest serial piece before the GPU starts |
| index load | 9 | |
| prefill: reading reads + E10 decode | 17–18 | 3 s more than E8 (below) |
| prefill: GPU part | 38 | device held; includes the 2.5 s host lane and 1.2 s of E10 transfers |
| prefill: records, Unmapped.out, read source | 4 | |
| STAR loop (156,931 reads with N) | 5.8 | STAR's mapping (seed/windows/stitch 5 thread-s) |
| Solo and STAR exit | ~8 | with E9 wired, Solo's counting moves to the device (~130 ms per E9) |
| hand-off pass, GPU pileup, non-host | 6 + 16 + 20 | unchanged |

Next steps, in the order I would take them:
1. **Wire E9's `soloRun` into level 4.** This is what makes level 4 a complete run instead of a measurement (the coordinator's note).
   Where it plugs in:
   - The records and the read → record map are on the device in `e2eBuild`.
   - CR/UR are in `dAux`, in a stride of `2 + nw` words; `SoloReads` wants contiguous `cr3`/`ur3`, so it needs either a strided gather
     or a stride field.
   - The annotation of the 156,931 STAR-mapped reads (`hostAnn`) exists only after STAR's loop. So `soloRun` runs in
     `seqCacheGpuAfterMapping`. Either the records stay on the device through the loop (5.8 s at level 4), or they are re-uploaded
     from the region.
   - STAR's Solo for reads with a record has to be switched off: `getCBandUMI`, `soloRead->record`, the Solo temp files and the
     post-map step. Then `soloWrite` writes `star_Solo.out/`.
   - The gate adds the 10 Solo files (`CellReads.stats` compared sorted) and the two Solo-derived outputs above.
2. **Make the E10 decode cheaper.** `in.e10.aux` (2.1 GB) and the two name-offset arrays (0.5 GB each) are zero-filled by a serial
   `std::vector::assign`. Allocating them uninitialized and first-touching them in the parsing threads should remove most of the
   3 s that the decode adds to `readSequences`. The quality and ACGTN loops could also run on 8 bytes at a time.
3. **The device's R11 histogram to the parent.** It is computed (2.8 ms) and verified, but the parent still computes its own in the
   `.1fq` decode, before STAR starts. Using the device's histogram needs the stream's CB correction to wait for the prefill, for
   example with the histogram in E8's region header.
4. **Release the device during the host lane.** E10 holds the device 1.3–1.7 s longer than E8, and E8 already holds it longer than the path
   before it (39.5 vs 24.8 s at T = 8, E8-e2e.md). The host lane (2.5 s) and the region copies could run after a release that keeps only the records. This
   matters when samples share a GPU, not for one sample's wall time.
5. **Pinned uploads: not worth it.** The engine's `upload` figure (5.4–5.8 s) includes the 3.1 s index upload and the 1.1 s clip.
   The read upload itself is about 1 s, and the index can already be pinned with `SINGLET_ENGINE_GPU_PIN=1`.

## 6. Files touched outside `src/engine_gpu/`, and why

| file | change | why |
|---|---|---|
| `include/singlet/engine_gpu.h` | `e2eLevel()` (0–4), `Input::E10`, `Output::E10` | in scope |
| `src/star/SeqCacheGpuPrefill.cpp` | the E10 decision, decode (R2 quality bits, R1 CR/UR, name offsets, R2/R1 kept mapped), the hook pass, levels 3–4 (Stats, Unmapped.out writer, junctions, read source), the R11 dump, `seqCacheGpuAfterMapping` (entry check, junction merge) | in scope |
| `src/star/ReadAlign_oneRead.cpp`, `ReadAlign_outputAlignments.cpp` | `e10Skip` (level 3): no stats, cache lookup, junctions or Unmapped.out for a read with a record; barcode + Solo record only | in scope |
| `src/star/ReadAlign.h` | the `e10Skip` member | the member the two files above share |
| `src/star/STAR.cpp` | declares and calls `seqCacheGpuAfterMapping(P, RAchunk)` after "RAM after mapping"; under `-DSTAR_PROFILE`, prints `pathSummary/pathCacheHit/pathMapped` and `outEntry` | the junction merge and the entry check must run after the loop and before STAR's output step; there is no other hook there. Does nothing unless E10 ran |
| `src/star/StarProfile.h`, `src/star/SeqCache.cpp` | the `outEntry` counter and `g_profPath[3]` (definition) | profile only: compiled in only with `-DSTAR_PROFILE` |
| `CMakeLists.txt` | adds `src/engine_gpu/hostloop_gpu.cu` to `singlet_engine_gpu` | new device code |
| `bench/fable/hostloop/jobs/{build,gate}.sbatch` | build (`VARIANT=gpu|prof`) and gate/profile jobs | deliverable |

No file named `src/engine_gpu/solo_*` was created; the new device code is `src/engine_gpu/hostloop_gpu.{h,cu}`.

## 7. Jobs, commits, how to run

- Branch `gpu/fable/hostloop`.
  - 55acf4e8: profile counters and build job.
  - a55e139b: gate job.
  - 05b042e4: steps 2–4.
  - 8cddfa29: R11 and type fixes; the gated binaries.
  - 3e2dcdbd: gate R11 check.
  - This document is the next commit.
- Builds (short-gpu, no GPU):
  - 796062 (gpu) and 796063 (prof) at 55acf4e8;
  - 796080 failed (uint64 vs uint64_t);
  - 796081 cancelled;
  - 796085 (gpu, 284 s) and 796086 (prof, 263 s) at 8cddfa29.
- Gates (short-gpu, H100 NVL, T = 18, node g051):
  - 796069 (`prof gpuE gpuD cpu`, 55acf4e8; profile and baselines);
  - 796090 (`gpuE2c gpuE3 gpuE4 gpuE gpuE2 prof3 prof4`, 8cddfa29);
  - 796116 (`cpu gpuD` at 8cddfa29 against 796069's outputs, §3d).
- Results: `$W/jobs/fable/hostloop/results/gate-<job>/` (`summary.txt`, per run `stages.txt`, `starlog_engine.txt`,
  `star_Log.out`, `monitor.tsv`; `equiv-*.txt`). Logs: `$W/logs/e10-gate-<job>.out`, `$W/logs/e10-build-<job>.out`.
- Switches:
  - `SINGLET_ENGINE_GPU_E2E=2|3|4` (with `--engine gpu`, `SINGLET_GPU_PILEUP=on`);
  - `SINGLET_ENGINE_GPU_E10_CHECK=1` (level 2: STAR's entries to a side buffer and compared);
  - `SINGLET_ENGINE_GPU_E10_R11=<file>` (writes the device's R11 histogram in `r11_hist.cpp`'s format).

  E10 turns itself off, and the log says why ("ENGINE_GPU: E10 (level N) not used: …"), when any of the following holds:
  - E8 does not run;
  - the input is not one cDNA FASTQ and one barcode FASTQ;
  - the barcode layout is not CB_UMI_Simple with `barcodeRead` 1;
  - `outQSconversionAdd`, `outFilterType BySJout` or a genome transform is set;
  - a cDNA read is longer than 256 bases or has a base other than ACGTN;
  - a barcode read has a length STAR would reject.
- Run: `sbatch --export=ALL,MODES="gpuE gpuE4",PAIRS="gpuE:gpuE4",T=18 bench/fable/hostloop/jobs/gate.sbatch` (binaries from
  `build.sbatch` in `$W/jobs/fable/hostloop/bin/`).
