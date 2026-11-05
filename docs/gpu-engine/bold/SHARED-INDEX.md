# Shared resident index on the device: design note (not built)

Branch `gpu/bold-admit`. Written 2026-11-05 for the decision after the advisor review. Numbers come from solo run 800349 (`star_Log.out` E17 line) and jobs 800347 and 800383. Statements marked **Inference** are not measured.

**Takeaway:** Every process holds its own 16.4 GB copy of the genome index. That copy is 47% of the normal-mode plan and 89% of the lean plan. One shared copy lets three samples run in normal mode where today only two fit.

## 1. What each process holds today

| Piece | GB | Same in every process |
|---|---|---|
| Genome index (GRCh38) and tables | 16.4 | Yes. Read only. |
| CUDA context | 0.5 | No. One per process. |
| Reads | 6.1 | No. |
| CR4 clip | 7.3 | No. |
| E10 aux, entries and masks | 4.5 | No. |
| E11 keep | 1.0 | No. |
| Lane buffers | 0.5 at the smallest batch, up to 63.6 at the largest | No. They fill the budget. |
| Pileup arena | 26.3 peak, of an 88 GB arena | No. |
| Non-host index and Bloom filter | 24.9, outside the budget | Yes. Read only. |

The E17 plan for one sample is 34.8 GB in normal mode and 18.5 GB in lean mode. At 3 slots the budget is 30.3 GB, so every sample runs lean. Jobs 800347 (3 and 4 slots) show 72 of 72 lean log lines. Lean mode re-uploads sequences per chunk and uses small batches.

## 2. Design

**One holder per GPU.** A small process, `singlet gpu-holder`, starts before the first sample. It creates a context, uploads the index once and exports each buffer with `cudaIpcGetMemHandle`. It serves the handles on a Unix socket. A client sends the genome key and receives the opaque handles.

**Clients import read only.** `mapDistinct` replaces its index allocation and upload with `cudaIpcOpenMemHandle`. The index is treated as const. Kernels already take device pointers as arguments, so no kernel changes.

**Evidence that IPC works here.** `bench/engine_gpu_gate/ipc_test.cu` imported a 43 GB buffer in 0.18 to 0.2 s with no mismatches and about 3 TB/s read. It worked from a process forked before CUDA init. A child forked after the parent initialized CUDA cannot use CUDA (`docs/research-beta/gpu-profile-deep.md` section 13). The engine already respects this, because the STAR child initializes CUDA itself.

**Saving per sample.** The index upload (5.4 s staged on a second thread at load) and 16.4 GB of device memory per process go away.

## 3. Lifetime and failure handling

- The holder owns the allocations and lives until the job ends. The job script starts it and stops it.
- A client closes its import when its phase ends. A client that dies needs no cleanup, because the driver releases its imports.
- If the holder dies, imported mappings become invalid. Kernels then fail. Clients check a heartbeat before each phase and run a small digest kernel after each import.
- On a failed check the client falls back to a private copy through the existing budget path and logs one line. A supervisor in the job script restarts the holder. The next sample imports again.
- **Inference:** a holder crash during a running kernel gives a CUDA error in the client. The engine's existing error paths then fall back to STAR on the CPU. That sample loses time and not correctness.

## 4. Identity risk

The index is read only, so results stay identical when the bytes are identical. Three risks remain.

1. **Wrong genome.** E21's pool failed on hgmm1k_v3 because the barnyard genome differs from the others (STAR exit 102, undiagnosed). The holder is keyed by genome directory, file sizes, mtimes and a digest. A client with a different key never imports. Mixed queues need one holder per genome. Human, mouse and barnyard indexes (16.4, 25.1 and 52.7 GB) do not all fit together. A mixed queue must be sorted by genome per GPU.
2. **Corrupt upload.** The holder computes a digest of the device buffers after upload and compares it to the one the CPU engine reference uses.
3. **Pointer assumptions.** The imported pointer differs per process. **Inference:** the kernels take the index by argument and never cache it. The full gate (65 identical files plus the run-dependent set) would catch a miss.

## 5. How many slots fit

Device 93.1 GB, less 2 GB margin, less the 16.4 GB shared index leaves 74.7 GB for processes. The per-process normal plan drops from 34.8 GB to 18.4 GB. The lean plan drops from 18.5 GB to 2.1 GB.

Today a 45.5 GB budget at 2 slots leaves 29.1 GB after the process's own index copy. With one shared copy the same card gives each of 2 processes 37.4 GB for everything else.

| Slots | Budget per process, GB | Fits normal mode (18.4 GB) | Spare for larger lane batches, GB |
|---|---|---|---|
| 2 | 37.4 | Yes | 19 |
| 3 | 24.9 | Yes | 6.5 |
| 4 | 18.7 | Barely | 0.3 |

Three slots is the comfortable limit. Four slots run with the smallest batches. Smaller batches mean small grids and low occupancy (ncu job 800386, lane B grid of 78 blocks).

**Non-host index.** Sharing it as well costs another 24.9 GB. That leaves 49.8 GB, enough for 2 slots in normal mode. A better first step shares the genome index only and keeps the non-host index per process, one at a time under admission (about 29 GB).

## 6. Limits of the idea

Memory is not the only limit. At 4 slots in lean mode the per-sample lock hold rose to 105 to 126 s (job 800347), against 39 to 55 s at 3 slots. **Inference:** time slicing between contexts costs more as contexts multiply, and GPM SM activity was only 31%. A shared index lifts the memory limit. It does not remove time slicing. Concurrency of kernels from several processes needs MPS. A single service that batches samples into one context avoids the issue and also fills the grids.

## 7. Effort

**Inference:** a holder of about 300 lines, a client change in `mapDistinct` where `dG1`, `dSA` and `dSAi` are allocated, and a job script change. Gate before any use: the full 10-sample identity gate, a mixed-genome queue, and a holder-kill test.
