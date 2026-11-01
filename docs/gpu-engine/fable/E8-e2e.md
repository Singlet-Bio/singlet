# E8: the GPU chain wired end to end inside the engine (records and per-read pass on the device)

Experiment E8 ("e2e") of wave 2 of the GPU aligner program. Branch `gpu/fable/e2e` (from `gpu/fable/records` = E6, 84a16500). Everything
new runs only under `SINGLET_ENGINE_GPU_E2E=1` (with `--engine gpu` and `SINGLET_GPU_PILEUP=on`); unset, the binary takes the paths it took
before, byte for byte (gate run `gpuD` below).

## 0. Answer in brief

- **Gate (a) passes.** On pbmc1k_v3 (gate 796010, T = 8, H100 NVL), the run with `--engine gpu`, `SINGLET_GPU_PILEUP=on` and
  `SINGLET_ENGINE_GPU_E2E=1` (`gpuE`) writes every count output byte for byte as the CPU engine does: 60 output files have the same
  sha256. They include `exon_counts`, `intron_counts`, `ambiguous_counts`, `ambient_corrected_counts`, `sj_counts`, `polya_counts`,
  `splice_psi`, `mt_heteroplasmy`, `mt_allele_counts`, `snp_ad` / `snp_dp`, `vdj_gene_usage`, STAR Solo's raw and filtered `matrix.mtx`,
  `star_SJ.out.tab`, `cell_calls.tsv` and `cell_qc_metrics.tsv`. The mapping statistics in `star_Log.final.out` are the same too.
  The remaining differences are the ones the default GPU path (`gpuD`) also has against the CPU engine; none comes from E8:
  - three files are written in thread-completion order but hold the same records: `nonhost_reads.fq` and `star_Unmapped.out.mate1` / `mate2`
    have the same sorted-record sha256 in all three runs;
  - `read_stats.tsv` and `star_Log.std.out` differ in timings;
  - in `pileup_stats.json`, two provenance counters differ: the sample size of the 5' prefix probe (12194 vs 12000, by engine) and the
    size of the log files at the time it is written;
  - `summary.json`'s `memory_tier` is `Large` for both GPU runs and `Medium` for the CPU engine. It is derived from the parent's peak
    virtual memory.
- **E8 against the default GPU path of the same binary.** `gpuE` and `gpuD` agree in every output except:
  - the read order of the same three files (same records);
  - two timing lines;
  - `pileup_stats.json`'s inventory byte count: `star_Log.out` is 110 bytes longer because of the E8 log lines.

  The GPU pileup's input digest is the same in both runs (`d2aa9309…`). Without the variable, the binary prints no E8 line and takes the
  hand-off file path (`gpuD`).
- **On the device now:** hops 1 to 3 (clip, dedup, mapping), 6 (records), 9 (the per-read hand-off pass) and 11 (the pileup reducers).
  Step 6 builds the records from the lanes' device-resident results with no hand-off file and no record parse. Step 9 runs with no
  per-read host loop: the host only resolves each of the 0.9 M distinct CRs once and runs the rRNA detector on 0.93 M sampled reads.
- **Still on the host:**
  - **Hop 8, the per-read entries.** CR/UR, quality counters and masks are still written by STAR's mapping threads, now into the shared
    memory object. Plan step 2 (computing them from R1 on the device) is **not done**.
  - **Hops 4, 5, 7, 10 and 12.**
- **Timing at T = 8.**
  - Wall time: 277.6 s for E8 vs 280.4 s on the default GPU path (−2.8 s, −1 %). The CPU engine takes 489.1 s.
  - CPU time: 905 vs 945 core-s (−40, −4 %). The CPU engine takes 2578 core-s.
  - Where it is saved: from the end of STAR's stream to the end of the pileup, 24.0 s and 41.5 core-s instead of 27.0 s and 78.9 core-s.
    STAR's prefill writes no 2.25 GB file (its summaries take 6 s instead of 9 s for the file).
  - What it costs: the device is held 39.5 s instead of 24.8 s in the prefill, because the host lane and the host record assembly now run
    before the device is released. This adds no wall time to a single sample, but it costs device occupancy when samples share a GPU (§6).
- **Why the wall-time gain is small.** The records cross the host twice: device → memfd in STAR's child (0.9 s), then memfd → device for
  the pass (1.3 s up, 1.6 s down) and → `PileupInput` → device for `pileupRun`. The STAR child's CUDA context ends with the child, and
  the pileup runs in the parent after STAR exits. The hops that dominate the wall time are still on the host: STAR's index load and
  stream, the cache fill, and the outputs.

## 1. What runs where

`singlet --engine gpu` forks STAR (fork, no exec). STAR's GPU prefill (`seqCacheGpuPrefill` -> `mapDistinct`) runs in the child and is
where the mapping results are on the device; the pileup runs in the parent and consumes STAR's BAM stream (only reads without a GPU record)
through a pipe. E8 keeps that process structure and replaces the hand-off between them:

| hop | before E8 (`--engine gpu`, `SINGLET_GPU_PILEUP=on`) | E8 (`SINGLET_ENGINE_GPU_E2E=1`) |
|---|---|---|
| 1 R2 (cDNA) to the device, 5'/3' clip | device (`mapDistinct`: upload, `k_clip`) | unchanged: device |
| 2 dedup of the clipped sequences | device | unchanged: device |
| 3 mapping, lanes A -> M -> B | device; packed results copied to the host, device copy freed | device; the packed results **stay on the device** (`runLane(..., keepDev)`) and are also copied to the host (STAR's sequence cache needs them) |
| 4 mapping, host lane (STAR's limits, 85,570 sequences, 0.27 %) | host, after the device is released | host, **before** the device is released; its results (41 MB) uploaded |
| 5 record order, host-side record assembly (STAR's cache) | host | unchanged: host (the cache inserts and STAR's per-read replay need them) |
| 6 records: blob -> SigRec / AlnRec / CIGAR ops / nt16 sequence (+ nM per alignment) | host: STAR's hand-off loop (`seqCacheAlignRecords`, 31.6 M records) -> 2.25 GB `.gpu_handoff.bin` -> parent's `prepend_handoff` parse (serial) | **device**: blobs and sequences gathered into record order on the device from the device-resident lane results (`k_rec_len`, scan, `k_rec_copy`), E6 `buildRecords` (`wantNm`), arrays copied once into a shared memory object (memfd); no file, no parse |
| 7 STAR's per-record summaries (Log.final.out, SJ.out.tab, Solo replay) | host, in the same loop as hop 6 | host: `seqCacheRestoreRecord` + `seqCacheHandoffSummary` only (the record bytes are no longer built) |
| 8 per-read CR / UR / quality counters / low-quality mask / clip flags / rRNA hook | host: STAR's mapping threads (`handoffReadEntry`) into `.reads` / `.reads.q` files | host: the same threads, into the shared region (step 2 of the plan, CR/UR from R1 on the device, **not done**) |
| 9 per-read hand-off pass (`gpu_handoff_pass_entries`: rule + every counter, per-cell reads and QC, WL-ambient counts, rRNA sample) | host, parallel (9 s at 8 CPUs, gate 796010) | **device** (`e2ePass`): one thread per read, block-reduced counters, per-cell atomics, scan + scatter of the kept reads in read order; the host resolves each distinct CR once (R11, barcode set, WL-ambient index) and runs the rRNA detector on the sampled unclipped reads |
| 10 PileupInput | host: the parsed hand-off vectors + stream vectors, copied | host: region arrays + stream arrays memcpy'd straight into `PileupInput` (stream offsets shifted as `prepend_handoff` shifts them) |
| 11 pileup reducers (exon/intron/ambiguous/SJ/SNP/chrM/molecules) | device (`pileupRun`) | unchanged: device |
| 12 `gpu_host_pass` (strand votes, poly(A), V(D)J), count matrices, outputs | host | unchanged: host |

The device-to-device chain hop 3 -> 6 is complete: the records are built from the lanes' device-resident results (only the host lane's
results are uploaded). Hop 6 -> 9 -> 11 crosses the host once each way because hop 6 runs in the STAR child (whose CUDA context ends with
it) and hops 9 and 11 run in the parent after STAR exits: the records travel device -> shared region (child) -> device (parent's pass) and
region -> `PileupInput` -> device (`pileupRun`). A device-pointer overload of `pileupRun` would need the arrays to outlive the child's
context (CUDA IPC with the parent holding a context while STAR runs) or the pileup to move into the child; see §6.

## 2. How it is wired

**Transport (no file).** When `SINGLET_ENGINE_GPU_E2E=1`, `--engine gpu` and `SINGLET_GPU_PILEUP=on` set up the hand-off, the parent
(`src/pipeline/singlet.cpp`) creates an anonymous shared memory object (`memfd_create`, raw syscall, `MFD_CLOEXEC`) before it forks STAR
and stores its descriptor in `singlet_engine_gpu::e2eRegion()`; nothing is mapped in the parent at that point. STAR's prefill sizes it
sparsely (`ftruncate` to `SINGLET_ENGINE_GPU_E2E_GB`, default 512 GB; tmpfs pages exist only once written) and maps it (`E2ERegion::attach`);
after STAR exits the parent reads the header with `pread`, checks the magic and maps only the bytes in use (`E2ERegion::mapUsed`).
A first version mapped 512 GB `MAP_SHARED | MAP_ANONYMOUS | MAP_NORESERVE` in the parent before the fork; the reservation counted in the
parent's VmPeak, which `summary.json` reports as `memory_tier` (gate 795833: `XLarge` instead of `Large`), hence the memfd.
Layout (`E2EHeader`, `include/singlet/engine_gpu.h`): header at offset 0, then 64-byte aligned
arrays reserved by a bump allocator (`E2ERegion::take`): SigRec, AlnRec, CIGAR ops, nM per alignment, nt16 sequence (written by
`mapDistinct`), readRecord, per-read entries, low-quality masks (reserved by the prefill; the entries and masks are written by STAR's mapping
threads exactly as they wrote the `.reads` / `.reads.q` files). The prefill writes the magic `SGE2E001` last; the parent's collector takes the
E2E path only if the magic is there, otherwise everything falls back to the file path (and the memfd is closed).

**Records on the device (`src/engine_gpu/engine_gpu.cu`, `mapDistinct`).** With `Input::e2e` set: the lanes keep their packed results on
the device; the host lane and the host-side record assembly (`hostPart`) run before the device is released, recording per record its source
(`chunk << 40 | lane << 32 | position in the lane's packed output`) and read; the index (G, SA, SAi) is freed; the host lane's results are
uploaded; `k_rec_len` + two scans + `k_rec_copy` gather every record's blob and (clipped) sequence into record order on the device; the
gathered sizes must equal the host-assembled ones (else fall back); `SINGLET_ENGINE_GPU_E2E_CHECK=1` also compares them byte for byte;
E6 `buildRecords(..., wantNm = true)` makes the pileup arrays (+ nM per alignment, which the per-cell QC needs); a record with a mate gap
(paired-end only) makes it fall back. The arrays are copied into the region (page-locked in place when the driver allows) and
`Output::e2eOk` is set.

**Prefill (`src/star/SeqCacheGpuPrefill.cpp`).** `Input::e2e` is set only where the "on" hand-off with summaries and per-read entries
would run (CB_UMI_Simple, Gene only, single end, `SINGLET_ENGINE_GPU_HANDOFF_MINQ` set: the conditions of the existing path). With
`e2eOk`: per record only `seqCacheRestoreRecord` (the restore half of `seqCacheAlignRecords`, split out in `ReadAlign_seqCache.cpp`) and
`seqCacheHandoffSummary`; per record its alignment count (`g_handoffRecAln`, which decides what goes into STAR's cache) is read from the
region's SigRec (`n_aln`, = the first byte of the record `seqCacheAlignRecords` writes: E6 identity); no `.gpu_handoff.bin`, no `.reads`
files; `g_handoffReads` / `g_handoffQMask` point into the region.

**Parent (`include/singlet/pileup/gpu_pileup_bridge.h`, `pileup_engine.h`).** `Collector::e2e()` finds the region (magic). A stream record
of a read that has a device record is counted (it would mean STAR's skip set differs; the pass then fails, as the file path does).
`gpu_handoff_pass` hands over to `gpu_handoff_pass_e2e`: under the device lock `e2ePass` uploads the region's arrays, lists the distinct
CRs on the device (E6 `distinctBarcodes`), calls back for the CR table (host, threaded: per distinct CR the barcode column, the WL-ambient
index and the `cb_status_` index, in the pass's order of checks), runs the per-read rule of `gpu_handoff_pass_entries` (`e6::readPass`,
`src/engine_gpu/records.cuh`) and every counter it keeps (`k_pass_eval`: block-reduced counters, per-cell reads / unique / multimapped /
mismatches / aligned bases and WL-ambient counts by atomics), scans the keep flags and scatters ReadRec, clip flags and masks in read order
(`k_pass_scatter`), returns the WL-ambient pending pairs and the sampled unclipped reads for the rRNA detector (both only feed sums, so
their order is free), and releases the context (`pileupRun` initialises the device afresh). The host runs the R4 detector on those reads
(threaded) and applies the counts; `Collector::add_e2e_reads` appends the kept reads as `add_handoff_read_u3` did one by one;
`Collector::finish` writes the region's arrays followed by the stream's (shifted) straight into `PileupInput` and unmaps and closes the region.

## 3. Gate (a): pbmc1k_v3, bit-identical counts

Job `bench/fable/e2e/jobs/gate.sbatch`; each pair was compared with `bench/star_accel/out_equiv.py`, which compares every output file by
sha256 and JSON files key by key with timing and path keys ignored. The files that STAR's threads write in completion order were also
compared by the sha256 of their sorted 4-line records.

Runs (one binary, built at b649ce93 by job 795961; debug-gpu, H100 NVL, 8 CPUs, `--threads 8`, node g051):

- `cpu` = the CPU engine;
- `gpuD` = `--engine gpu` + `SINGLET_GPU_PILEUP=on` (the default path);
- `gpuE` = `gpuD` + `SINGLET_ENGINE_GPU_E2E=1`;
- `gpuEc` = `gpuE` + `SINGLET_ENGINE_GPU_E2E_CHECK=1`, which compares the device-gathered record blobs and sequences with the host-assembled
  ones byte for byte.

| pair | identical files | differ only in timing / path keys | differences | what differs |
|---|---|---|---|---|
| cpu vs gpuE | 60 | 2 | 7 | read order of `nonhost_reads.fq`, `star_Unmapped.out.mate1/2` (same records); `read_stats.tsv` timings; `star_Log.std.out` timestamps; `pileup_stats.json` provenance (`clip5p/probe/n_reads` 12194 vs 12000, `outputs/bytes_public`); `summary.json` `memory_tier` Medium vs Large |
| cpu vs gpuD | 60 | 2 | 7 | the same seven, same values |
| gpuD vs gpuE | 60 | 3 | 6 | read order of the same three files (same records); `read_stats.tsv` and `star_Log.std.out` timings; `pileup_stats.json` `outputs/bytes_public` (+109 bytes: `star_Log.out` +110 bytes from the E8 log lines, `ambient_contamination.json` −1 byte in its `seconds` field) |
| cpu vs gpuEc | job 796036 (pending) | | | |

- `star_Log.final.out` is identical across all runs except its four timing lines. `out_equiv.py` skips this file, so it was checked by
  `diff`.
- The first gate (795833 `gpuE` / `gpuD`, 795834 `cpu`; commit 572b6e36; compared across jobs by 795937) found the same differences,
  with one exception: `summary.json`'s `memory_tier` was `XLarge` for `gpuE`, caused by the parent's 512 GB pre-fork mapping. The memfd
  transport (b649ce93) fixed it, and gate 796010 shows `Large` for both GPU runs.
- Only the read-order check is new in 796010.
- E8's own checks in the logs:
  - `[gpu_pileup] hand-off: 61294166 reads take their alignments from the GPU prefill, 0 primary alignments differ from the stream's`;
  - 63,143,563 reads without BAM records are handled on the device;
  - the stream count of hand-off reads is 0.

## 4. Timings

Gate 796010: debug-gpu, H100 NVL, 8 CPUs, `--threads 8`. CPU time is the job cgroup's `usage_usec`, read before and after each run and
sampled every second (`monitor.tsv`). A T = 18 run on short-gpu was not possible: the account's three short-gpu H100s were held by the
other track for the whole session.

| run | wall (s) | core-s | max RSS | first gate (795833/4): wall / core-s |
|---|---|---|---|---|
| CPU engine | 489.1 | 2577.7 | 29.0 GB | 494.7 / 2589.2 |
| `--engine gpu`, GPU pileup (gpuD) | 280.4 | 944.6 | 43.5 GB | 284.7 / 953.5 |
| + E8 (gpuE) | **277.6** | **905.0** | 43.5 GB | 282.8 / 919.3 |

Phases, measured from the run's log timestamps and the cgroup samples:

| phase | gpuD wall / core-s | gpuE wall / core-s |
|---|---|---|
| start → STAR streams (1fq decode, R11 histogram, STAR launch) | 27.3 / 85.9 | 28.7 / 88.4 |
| STAR: index load, GPU prefill + cache fill, stream | 168.0 / 632.0 | 167.0 / 622.7 |
| stream drained → GPU pileup done | 27.0 / 78.9 | **24.0 / 41.5** |
| outputs (`gpu_host_pass`, matrices, QC, donor, mt, ...) | 58.1 / 147.3 | 57.9 / 152.4 |

Device and E8 step times (gpuE; gpuD in parentheses where it differs):

| step | time |
|---|---|
| prefill: index upload / read upload / clip / dedup / GPU mapping | 3.13 s / 5.45 s / 1.14 s / 3.53 s / 15.29 s |
| prefill: host lane (85,570 sequences) | 4.78 s, while the device is held (in gpuD, after the device is released) |
| E8 gather (`k_rec_len`, scans, `k_rec_copy`; 41.4 MB host-lane upload) | 180 ms |
| E8 `buildRecords` (`wantNm`): kernels 15 / 4 / 17 ms | 44 ms |
| E8 copy to the shared memory object: 31.6 M SigRec, 32.0 M AlnRec, 42.0 M ops, 1.41 GB sequence, page-locked in place | 906 ms (1847 ms in the first gate, before the parallel prefault) |
| device held by the prefill | 39.5 s (24.8 s) |
| STAR's per-record pass in the prefill | summaries only, 6 s (records + summaries + 2.25 GB `.gpu_handoff.bin`, 9 s) |
| cache fill, total | 75 s (75 s) |
| E8 hand-off pass, `e2ePass`: upload 1330 ms (pinned) + distinct CRs 14 ms (897,422) + CR table on the host 319 ms + `k_pass_eval` 20 ms + `k_pass_scatter` 7 ms + download 1606 ms | 3.64 s |
| rRNA detector on the 926,859 sampled unclipped reads (host, threaded) | 0.66 s |
| drained → hand-off pass done | 6 s (9 s: entries loaded, host pass) |
| pileup: build input / `pileupRun` (GPU 2.16 s) / host pass | 5.27 / 9.05 / 2.24 s (5.55 / 8.98 / 2.28 s) |

## 5. Files touched outside `src/engine_gpu`

| file | lines (+/−) | why |
|---|---|---|
| `include/singlet/engine_gpu.h` | +110 | in scope: `E2EHeader`, `E2ERegion` (memfd create / attach / mapUsed / release), `Input::e2e` and the SAM flag / MAPQ options `buildRecords` needs, `Output::e2eOk`, the `e2ePass` API |
| `CMakeLists.txt` | +1 −1 | `src/engine_gpu/records_gpu.cu` (E6's record and pass kernels) added to the `singlet_engine_gpu` library; E6 built it only into its test |
| `src/pipeline/singlet.cpp` | +6 | the memfd must be created by the parent before it forks STAR. Only the parent can create something that the child inherits and the parent reads after the child exits. Runs only under `SINGLET_ENGINE_GPU_E2E=1` with the "on" hand-off |
| `src/star/SeqCacheGpuPrefill.cpp` | +53 −6 | sets `Input::e2e` under the existing "on" conditions, attaches the region, and adds the E8 branch of the hand-off loop: summaries only, entries and masks in the region, magic last |
| `src/star/ReadAlign.h`, `src/star/ReadAlign_seqCache.cpp` | +3; +6 −1 | `seqCacheRestoreRecord` split out of `seqCacheAlignRecords`, because the summaries need the restored read state without the record bytes. The old function now calls the new one and then builds the record exactly as before: same code, same order, so the default path is unchanged. Not in the listed scope, but it is the minimum: the alternative would duplicate STAR's restore logic in `SeqCacheGpuPrefill.cpp` |
| `include/singlet/pileup/gpu_pileup_bridge.h` | +55 −2 | in scope: `Collector::e2e()`, `add_e2e_reads`, `prepend_e2e` (region and stream arrays straight into `PileupInput`), the stream-overlap count, unmap and close |
| `include/singlet/pileup/pileup_engine.h` | +103 | `gpu_handoff_pass_e2e`. Not in the listed scope: `gpu_handoff_pass` is a private member of `PileupEngine`, and the pass updates its private state (`per_cell_reads_`, `per_cell_qc_`, `cb_status_`, `wl_umi_counts_`, `bc_index_`, `wl_amb_index_`, `rrna_detector_`, `chrm_tid_`), so it has to be a member. `gpu_handoff_pass` takes it only when `Collector::e2e()` finds the region's magic, which happens only under `SINGLET_ENGINE_GPU_E2E=1` |
| `bench/fable/e2e/jobs/*`, `docs/gpu-engine/fable/E8-e2e.md` | new | deliverables |

## 6. What remains for the host track

1. **Plan step 2: per-read entries from R1 on the device.** CR/UR packing, the counts of quality bases below threshold and not N, the
   low-quality masks and the R11 exact-CB histogram. Today:
   - STAR's mapping threads compute each entry (`handoffReadEntry`) during the stream, into the region;
   - the R11 histogram comes from the parent's 1fq decode before STAR starts.

   The prefill already has R2 on the device. R1 (28 bases plus qualities) would add about 3.7 GB of upload for this sample, and a
   kernel would write the 24-byte entries and 16-byte masks straight into the records' read order. The region's entry layout
   (`HandoffReadEntry`) is the contract.
2. **Device occupancy in the prefill (+14.7 s).** E8 runs the host lane (4.8 s) and the host record assembly before the device is
   released, so the lanes' results stay resident for the gather. The fix is to release the lock, or at least most of the memory, between
   the lanes and the gather. One option: free the index first (done), keep only the packed lane results, and let another
   process take the device while the host lane runs. This needs a device-sharing policy in `lockDevice` (memory-based admission instead
   of an exclusive lock).
3. **Records crossing the host.** The device → memfd → device round trip (0.9 + 1.3 + 1.6 s) and the copy into `PileupInput` would go
   away with a device-pointer `pileupRun(const DeviceInput&)` and arrays that outlive the STAR child. The parent cannot initialise CUDA
   before the fork. Two options:
   - the parent initialises CUDA after the fork, allocates the record buffers and passes CUDA IPC handles to the child through the
     existing pipe, sized by the prefill's request;
   - the pileup moves into the STAR child.
4. **STAR's per-record work.** `seqCacheRestoreRecord` + `seqCacheHandoffSummary` (6 s) feed `Log.final.out`, `SJ.out.tab` and Solo's
   replay; the cache inserts take 9 s; the host lane takes 4.8 s on 0.27 % of the sequences.
5. **PileupInput build (5.3 s) and the stream path.** The signature grouping and the parse of the ~0.1 M stream reads with BAM records.
6. **Outputs: about 58 s and 150 core-s.** `gpu_host_pass` (strand votes, poly(A), V(D)J), matrices, QC, donor and mitochondrial
   outputs. This is now the largest phase after STAR.
7. **Fallbacks that keep the hand-off file path:**
   - paired-end records with a mate gap;
   - barcode layouts other than CB_UMI_Simple, or CB/UMI longer than 21 bases;
   - reads over 128 bases;
   - `SINGLET_ENGINE_GPU_HANDOFF_MINQ` unset or summaries off.

## 7. Jobs, commits, reproduction

- Branch `gpu/fable/e2e`, from E6 `84a16500`. Commits:
  - dfe1e511: the chain;
  - 3dc3aba6: gate job;
  - 7c855d0f: context release;
  - d70e7ba8: build `-j`;
  - 572b6e36: compile fixes;
  - 36ef8734 and 76d4882d: kept outputs, compare job;
  - b649ce93: memfd, prefault, pinned copies, read-order check;
  - the commit that adds this document.
- Builds (`build_singlet.sbatch`, on short-gpu without a GPU):
  - 795823: cancelled while it waited on short-cpu;
  - 795830: failed (compile errors, fixed in 572b6e36);
  - 795832: ok at 572b6e36;
  - 795961: ok at b649ce93, 283 s.
- Gates (`gate.sbatch`):
  - 795833: `gpuE`, `gpuD` at 572b6e36;
  - 795834: `cpu` at 572b6e36;
  - 796010: `gpuE`, `gpuD`, `cpu` at b649ce93, the gate reported here;
  - 796036: `gpuEc` (record check) at b649ce93, compared against 796010's kept `cpu` output.
- Compare (`compare.sbatch`): 795937 compared `cpu` (795834) against `gpuE` and `gpuD` (795833).
- Results: `$W/jobs/fable/e2e/results/gate-<job>/` (`summary.txt`, `equiv-*.txt`, per mode `run.log.ts`, `monitor.tsv`, `time.txt`,
  `starlog_engine.txt`, and `out/` without sequence-bearing files). Logs: `$W/logs/e8-gate-<job>.out`.
- Reproduce:
  - `MODES="gpuE gpuD cpu" T=8 sbatch --partition=debug-gpu --gres=gpu:nvidia_h100_nvl:1 --cpus-per-task=8 --mem=100G --time=00:29:00 --export=ALL bench/fable/e2e/jobs/gate.sbatch`;
  - use the defaults (T = 18, 20 CPUs, short-gpu) for a whole-node run.
