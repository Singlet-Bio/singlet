# M8: host to device ingest pipeline with dummy kernels

Question (`docs/gpu-engine/exploration.md` section 9, M8): can one H100 NVL be fed >= 50 M read pairs/s from real `.1fq` files
with <= 3 host cores and < 5% GPU idle, and what does feeding both GPUs of a node cost?

Files: `m8_ingest.cu` (bench), `jobs/*.sbatch` + `jobs/m8_common.sh` (Clipper jobs), `make_m8_results.py` (aggregation), results in
`../results/m8.json` (85 pipeline runs, copies, bandwidth probes, node-load lines; per-batch arrays are in the raw JSONL on Clipper,
`$W/jobs/w11/m8/results_{smoke,single_789223,dual_789286}/m8_runs.jsonl`).

## What was built and measured

- **Decoder.** Own direct decoder over the library's block layout (`Reader` for header, block index and barcode dictionary): pread, zstd 1.5.7
  `ZSTD_decompressDCtx`, then one pass per column straight into the pinned ring buffer. Checked against `Reader::decode_payload` plus a
  re-pack on every block of GSM6470208 (244) and 300 blocks of GSM7656149: 0 mismatches (plus 100 blocks each at the start of the single job).
  CRC32 is off by default (the library verifies it by default; see ablations).
- **Payload per pair** (SoA over a batch): R2 2-bit, 24 B (reads <= 96 nt; longer reads would widen the slot to the next multiple of 8 B, up to
  127 nt, with the length field in the CB+UMI word); CB+UMI 8 B (CB 32 bit, UMI 24 bit, R2 length 7 bit, any-N flag 1 bit; N is stored as A like
  the file's packing); quality 24 B. Total **56 B/pair** (the design note's 53 B is unpadded). Unknown barcodes (sentinel index, 9.3% of reads in
  GSM7656149, 18.8% in GSM6470208) carry their raw barcode; known ones use a dictionary lookup table.
- **The atlas `.1fq` files store no quality** (`qual_mode` 0). The default runs therefore fill the Q slot from the R2 bits (a 24 B copy, a
  stand-in for the parse+copy of a binned-4 column) so that PCIe and kernel see the 56 B design payload. `--qual none` runs (32 B/pair) are the
  real content; they are within 3% of the 56 B runs. The Q column is synthetic; everything else is real data.
- **Pipeline.** N decode threads claim blocks in order (500 K reads per block in these files); ring of 4 pinned 256 MB buffers (4.5 M pairs per
  batch at 56 B); a submitter thread issues H2D on a copy stream into 3 device buffers; events gate the compute stream; dummy kernel reads every
  byte (XOR fold, verified equal to the host fold in `--verify` runs) and spins until `n * spin_ns` has passed. Kernel time matches: 44.9 ms
  at 10 ns/pair, 89.5 ms at 20 ns/pair for 4.5 M pairs. A releaser thread frees the pinned slot when the copy event fires.
- **Metrics.** Sustained pairs/s (whole run, and steady state from batch 3 on), GPU idle % = gaps between consecutive kernels on the compute
  stream / (gaps + kernel time) from cudaEvents over the steady window, host CPU from `getrusage` (all threads), per-phase thread times.
  **nsys is not in the CUDA image** (a host copy exists at `/usr/local/cuda-12.9/bin/nsys` but was not used), so idle % is event-based only.
  Idle % is only meaningful when the GPU is the slower side: with spin 0 the kernel is 0.3 ms and idle is ~99.8% by construction.
- **Environment.** Node g050 (single and smoke jobs; the dual job ran on an H100 node of the same type, not re-inspected): 2 x Xeon Gold 6238R (28 cores per socket, Cascade
  Lake), cgroup of 28 cores all on NUMA node 0 next to the GPU, CUDA 12.4.131 image, driver 610.43.02, H100 NVL 95.8 GB. `nvidia-smi` reports
  PCIe gen 4 x16, but **pinned H2D measures 12.3 GB/s and D2H 14.3 GB/s** for 256 MB transfers (median = min = max over 30 repeats), a
  Gen3-class figure, not the 53.8 GB/s the design note assumed. **With both GPUs copying at once each gets 6.2 GB/s (H2D), 7.2 GB/s (D2H):
  the two GPUs share one 12.3 GB/s host link.** Local scratch is `/tmp` on ext4 `/dev/sda1` (1.8 TB), the shared filesystem is NFSv4.2 over
  RDMA (rsize 1 MB).

## Results, one GPU, 56 B/pair, node-local `/tmp` with the page cache dropped before each run (M pairs/s, whole run; idle % in brackets)

GSM7656149 (3' v3, 263.4 M pairs, 22.5 B/pair in the file):

| threads | spin 0 | spin 10 ns | spin 20 ns | host cores (avg) | ingest core-s per M pairs |
|---|---|---|---|---|---|
| 1 | 11.6 | 11.6 (88%) | 11.5 (77%) | 0.87 | 0.075 |
| 2 | 21.5 | 21.0 (79%) | 20.2 (59%) | 1.68 | 0.078 |
| 3 | 29.9 | 30.1 (69%) | 30.1 (38%) | 2.42 | 0.081 |
| 4 | 38.3 | 39.2 (60%) | 37.5 (23%) | 3.13 | 0.082 |
| 6 | **51.3** | 50.2 (48%) | 48.3 (**0.02%**) | 4.39 | 0.086 |
| 8 | 62.1 | | | 5.54 | 0.089 |

GSM6470208 (5' v2, 121.8 M pairs, 14.4 B/pair; reads are 90 nt, fixed length):

| threads | spin 0 | spin 10 ns | spin 20 ns | host cores (avg) | core-s per M pairs |
|---|---|---|---|---|---|
| 1 | 13.2 | 13.2 (87%) | 13.1 (73%) | 0.90 | 0.068 |
| 2 | 23.5 | 23.2 (76%) | 23.0 (53%) | 1.68 | 0.072 |
| 3 | 34.9 | 35.2 (63%) | 34.4 (28%) | 2.53 | 0.072 |
| 4 | 45.5 | 45.1 (52%) | 43.9 (5.6%) | 3.35 | 0.074 |
| 6 | **62.5** | 59.0 (36%) | 47.1 (0%) | 4.75 | 0.076 |

Spin 20 ns/pair means the GPU consumes at most 50 M pairs/s, so the 6-thread rows at spin 20 are GPU-bound (idle 0%, rate = the 50 M/s kernel cap).
Spin 10 ns is a 100 M/s GPU: idle then shows how far the feed is below that.

Other placements and ablations (GSM7656149 unless noted, spin 0):

| variant | threads | M pairs/s | core-s per M pairs |
|---|---|---|---|
| local `/tmp`, **warm page cache** | 1 / 3 / 6 | 15.3 / 42.9 / 80.8 | 0.065 / 0.070 / 0.074 |
| local `/tmp`, cold, no WILLNEED prefetch | 3 | 30.1 (same as with prefetch, 29.9) | 0.081 |
| local `/tmp`, cold, **with CRC32 verify** (library default) | 3 | 15.4 | 0.175 |
| local `/tmp`, cold, no quality column (32 B/pair) | 3 / 6 | 30.7 / 52.1 | 0.079 / 0.083 |
| **shared FS read directly** (GSM7656149) | 1 / 2 / 3 / 4 / 6 | 13.1 / 24.7 / 36.8 / 47.9 / **70.1** | 0.075 / 0.080 / 0.081 / 0.082 / 0.084 |
| shared FS direct, GSM6470208 | 1 / 2 / 3 / 4 / 6 | 14.5 / 27.4 / 40.4 / 51.7 / 74.9 | 0.068-0.077 |
| shared FS direct, spin 20, GSM7656149 | 3 / 6 | 36.1 (idle 26%) / 48.5 (idle 0%) | 0.081 / 0.095 |
| shared FS direct, repeat of the 3-thread run | 3 | 35.7 (first: 36.8) | |

## Where the time goes (one thread, GSM7656149, per pair)

zstd decompress 43 ns, pack 20 ns, pread 4 ns from a warm cache, 12-23 ns wall when the file is cold (I/O wait, not CPU). That is
~0.065-0.075 core-s per M pairs for decode+pack alone: one thread sustains ~12-15 M pairs/s cold, ~15 M/s warm. The 6-thread cost per M pairs
rises to 0.086 (memory-bandwidth and cache contention). Pack is a third of the CPU; zstd alone would allow ~21 M pairs/s per thread, so even a
free pack step reaches only ~63 M pairs/s on 3 threads. The raw 22.5 B/pair file rate is 1.15 GB/s at 51 M pairs/s.

## Dual-GPU node (job 789286: two independent pipelines, GPU0 reads GSM7656149, GPU1 reads GSM7036403, first 100 M pairs each)

| threads per GPU | source | spin | pairs/s per GPU (M) | idle % | cores, both processes | node cores busy* |
|---|---|---|---|---|---|---|
| 3 (one GPU alone) | /tmp | 0 / 20 | 29.5 / 29.1 | 99.8 / 38.8 | 2.42 / 2.38 | 1.8 |
| 3 | /tmp | 0 | 29.9 and 28.7 | | 2.52 + 2.36 | 3.5 |
| 4 | /tmp | 0 | 36.8 and 35.0 | | 3.19 + 2.96 | 3.7 |
| 6 | /tmp | 0 | 49.1 and 44.5 | | 4.45 + 4.13 | 4.5 |
| 6 | /tmp | 20 | 45.9 and 39.6 | 0 and 15 | 4.14 + 3.76 | 4.2 |
| 3 | shared FS | 0 | 34.3 and 35.1 | | 2.93 + 2.95 | 3.8 |
| 4 | shared FS | 0 / 20 | 44.0 and 43.9 / 43.3 and 44.0 | 7.6 and 4.7 (spin 20) | 3.89 + 3.92 | 4.6 |
| 6 | shared FS | 0 | 64.5 and 64.5 | | 5.81 + 5.84 | 5.3 |

\* `node_cores_busy` is a `/proc/stat` delta over a window that includes ~3 s of container start and plan reading per run, so it understates the
load; the sum of the two processes' `getrusage` cores is the better node figure (e.g. 8.6 cores for 6 + 6 threads).

- Two pipelines run at the **same per-GPU rate as one** (29-30 M/s at 3 threads each, 6 + 6 threads 45-49 M/s from /tmp, 64.5 M/s each from the shared
  FS). Host cost per M pairs is unchanged to +10% (0.082 to 0.093). Two consumers did not slow the shared filesystem: 129 M pairs/s together =
  2.9 GB/s of file data, 5.8 + 5.8 cores.
- The shared host link is the constraint to watch: 2 GPUs x 50 M pairs/s x 56 B = 5.6 GB/s of the 12.3 GB/s total link (45%); at the 32 B real content
  it is 3.2 GB/s (26%). Both GPUs copying flat out get only 6.2 GB/s each.
- Staging (2 concurrent single-stream copies from the shared FS to /tmp): 0.87 GB/s each, 1.42 busy cores node-wide.

## Staging and the shared filesystem

| copy (client page cache dropped first) | streams | GB/s read+write | with fsync | CPU s |
|---|---|---|---|---|
| GSM7656149, 5.93 GB | 1 | 0.93 | 0.51 | 8.5 |
| GSM6470208, 1.76 GB | 4 | 1.53 | 0.64 | 5.2 |
| GSM7036403, 1.86 GB | 2 | 1.65 | 0.57 | 2.9 |
| two files at once, 1 stream each (dual job) | 1 + 1 | 0.88 + 0.87 | 0.49 + 0.49 | 11.9 total |

A single reader is limited by one core copying out of NFS (0.9 GB/s); several streams reach 1.5-1.7 GB/s. Writing to local `/tmp` (about
0.5 GB/s with fsync) is slower than reading the shared FS. The pipeline read the shared FS directly at up to 70 M pairs/s x 22.5 B = **1.6 GB/s per
GPU, 3.0 GB/s from two pipelines**, and did not run slower than from cold local `/tmp` (51 M/s at 6 threads); only a warm local page cache is faster
(81 M/s). **Staging is not required for rate**; it costs ~1 core-minute per 100 M pairs and a slower /tmp write.
Caching caveats: the client page cache was dropped with `posix_fadvise(DONTNEED)` before each run, and `mincore` residency after the drop was 0.0000
in every run (before the drop it was 0.26-1.0, since earlier jobs on the node had read the same files). **The NFS server's own cache cannot be
controlled or observed**; GSM7656149 was read many times that day, so these rates may include server-side cache hits. The rate of the first
shared read in the job (copy 0.93 GB/s) and the repeated 3-thread run (35.7 vs 36.8 M/s) did not differ, which suggests the server was already warm
for the whole run.

## Verdict against section 9

| criterion | result | |
|---|---|---|
| >= 50 M pairs/s | 3 threads: 30 M/s (local cold), 43 M/s (warm cache), 37-40 M/s (shared FS); 6 threads: 51-62 M/s (local cold), 70-75 M/s (shared FS), 81 M/s warm; 4 threads reach 45-52 M/s on the 5' sample | met only with 5-6 threads |
| <= 3 cores | 6 threads use 4.4-4.8 cores on average, 0.085 core-s per M pairs (5.9 cores from the shared FS); at 50 M pairs/s ingest alone costs ~4.3 cores | **FAIL** |
| < 5% idle | 0.02% idle with 6 threads at spin 20 (the GPU, not the feed, is the limit); with 3 threads and spin 20 the GPU idles 28-38% | met at 6 threads, not at <= 3 |
| overall, from local /tmp with <= 3 cores | **FAIL**: 3 threads feed ~30 M pairs/s (~43 with a warm cache) | |

Consequences for the host core budget:
- Ingest at the 50 M pairs/s target costs **~0.085 core-s per M pairs, 4.3-4.5 cores per GPU** (about 9 cores for a node with two GPUs), not 1-3 threads.
  Decode is ~50% zstd, ~25% pack; the CPU callers' 1.3-2.3 core-s per M pairs come on top. At 10-20 M pairs/s per GPU (the design's end-to-end
  target) ingest takes 1-2 cores per GPU and is not the limit.
- The design note's decode figure of 67 M reads/s per thread is not what a thread delivers in the full path: zstd plus packing into the GPU
  layout gives 12-15 M pairs/s per thread. Ways to get back to 3 threads at 50 M/s: a lighter codec or zstd level for the staged files, a GPU-layout
  on-disk format so decompression writes the pinned buffer directly (bound ~21 M/s per thread, about 63 M/s on 3 threads), or nvCOMP; none was measured here.
- The `.1fq` CRC32 check (library default, byte-wise table) more than doubles the cost (0.175 core-s per M pairs); keep it off or use a hardware CRC.
- PCIe is not the limit (12.3 GB/s, one pipeline at 50 M/s uses 2.8 GB/s) but the link is shared by both GPUs and is a Gen3-class figure, not 53.8 GB/s.

## Caveats and problems

- Quality column is synthetic (files have none); unpadded 53 B payload not tested. Reads of more than 96 nt were not in these files (max R2 91; no truncation
  occurred, 0 reads truncated).
- GSM7656149 has variable-length (trimmed) R2; the slot width was chosen from the first 8 blocks (max 91). Truncation counts are reported per run.
- Each configuration ran once (these runs are 1.6-23 s long; one-thread repeats in the smoke job agreed within 3%); the dual-GPU runs are 2-5 s each,
  so ramp-up and tail are visible (the steady-state column in `m8.json` excludes the first 3 batches).
- Idle % comes from cudaEvents, not nsys; the dummy kernel is a spin, not real work, so the effect of kernel-side L2/HBM contention with the copies
  is not captured.
- `debug-gpu` allows 8 CPUs per user, so all jobs ran on `short-gpu`. `clipper.py push` with a relative remote path writes to `$HOME`; use absolute paths.

## Jobs and GPU time

789190, 789198, 789207, 789222 (builds, CPU). 789204 (smoke, aborted after 13 s: files have no quality). 789208 (smoke, 2.6 min, 1 GPU).
789223 (single GPU, 28 cores, 10.6 min). 789286 (full node, 2 GPUs, 1.8 min). GPU time: 0.04 + 0.18 + 0.06 = **~0.3 GPU-hours** (budget 2).
