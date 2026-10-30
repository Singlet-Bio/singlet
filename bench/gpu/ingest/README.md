# Ingest format prototype (`.1fg`): CPU-node transform, GPU-node reader

Design note: `docs/research-beta/ingest-format.md`. Aggregated results: `bench/gpu/results/ingest.json` (built by `summarize.py` from `results_raw/`).
Written overnight 2026-10-28 by a W11 side session; nothing here is committed or pushed by that session.

## Files

| file | what |
|---|---|
| `ingest_fmt.h` | `.1fg` v1 structures, CRC32C |
| `enc_1fg.cpp` | CPU transform `.1fq` -> `.1fg` (threads, zstd levels, per-section raw, chunk size, exception list, `--selftest` against the library decoder) |
| `read_1fg.cu` | GPU-node reader: pinned ring, H2D, nvCOMP batched zstd, consume kernel; modes `e2e`, `--no-decode`, `--gpu-decode-bench`, `--verify` (every pair against `singlet decode` text) |
| `dec_1fg.cpp` | host decode of `.1fg` to the byte-numeric reads `lib1fq` hands the CPU engine, hash-compared with the `.1fq` library decode |
| `upload_dryrun.py` | dry-run upload planner (no network): object key, parts on chunk boundaries, CRC32C, manifest |
| `summarize.py`, `download_results.json` | aggregation; hand-collected download numbers |
| `jobs/*.sbatch` | the Clipper jobs (below). `jobs/*.txt` are earlier job scripts kept for the record (they hung in `wait` on a `tee` process substitution) |
| `results_raw/` | small result files copied back (JSONL, logs, dry-run manifests); no reads, no `.1fg` |

Build and run on Clipper: sources are put under `$W/jobs/w11/ingest/src`, binaries land in `$W/jobs/w11/ingest/bin2` (nvCOMP 6.0.0.12 libraries from the pip wheel
`nvidia-libnvcomp-cu12` in `$W/jobs/w11/ingest/nvcomp`); `jobs/build.sbatch` compiles in the CUDA 12.4.1 image. Reads and `.1fg` files exist only in job `/tmp`.

## Jobs (all `--nice=10000`, account debruinz; partition short-gpu; GPU jobs `--gres=gpu:nvidia_h100_nvl:1`, 16 CPUs; CPU-only jobs without a GPU gres)

| job | script | what | result |
|---|---|---|---|
| 789839 | `jobs/probe.sbatch` | ENA / NCBI SDL sizes, timed ranged downloads, nvCOMP wheel contents (CPU) | `results_raw/` (log in `$W/logs`) |
| 789842 | `jobs/get_nvcomp.sbatch` | pip-install nvCOMP wheel, keep headers and libs (CPU) | nvCOMP 6.0.0.12 |
| 789850-789855, 789873, 789877, 789902, 789919 | `jobs/build.sbatch` | builds (the first ones failed: `-msse4.2`, library name; 789902/789919 = final binaries) | `bin2/` |
| 789859 | `jobs/run_sample.sbatch` (first version) | PBMC 1k, first-generation format (256K-pair chunks, 8 per batch: GPU decode 28 M pairs/s); cancelled in the identity step: FIFO deadlock | `results_raw/pbmc1k_v3_789859_v0_cancelled_in_verify.jsonl` |
| 789881 | `jobs/run_sample.sbatch` | PBMC 1k: transform variants, reader, GPU decode, identity (3 files), M8 baseline. Format with N positions of unknown barcodes listed (changed afterwards, no effect on this sample's identity) | `results_raw/pbmc1k_v3_789881.*` |
| 789904 | `jobs/run_sample.sbatch` | GSM6470208 (121.8 M pairs), final format | `results_raw/GSM6470208_789904.*` |
| 789930 | `jobs/run_sample.sbatch` | PBMC 1k rerun with the final format | see `ingest.json` |
| 789905, 789914 | `jobs/cpu_extra.sbatch` | level ladder, CPU compatibility, dry run (superseded by the next two) | removed |
| 789931, 789932 | `jobs/cpu_extra.sbatch` | same, final format, PBMC 1k / GSM6470208 | `results_raw/cpu_*` |
| 789879, 789903 | `jobs/download*.txt` | download tests v1/v2: hung in `wait`; 789903 still gave gunzip cost and a 4 M-spot download | `results_raw/dl_789879_cancelled.log`, `dl2_*` |
| 789920 | `jobs/download.sbatch` | S3 range scaling, full `singlet download SRR21102187` (VDB streaming) | `results_raw/dl3_789920.*` |
| 789925 | `jobs/lite_probe.sbatch` | where SRA Lite lives (requester-pays on S3/GCS, anonymous on the NCBI host) | log |
| 789926, 789928, 789933 | `jobs/lite_compare.sbatch`, `jobs/lite_vs_full_cpu.sbatch` | Lite download speed; Lite vs full reads (md5 equal); local decode CPU | `results_raw/litecmp_*`, `litecpu_*` |
| 789941, 789944, 789945 | `jobs/build.sbatch` (bin3, adds `--shuffle`), `jobs/shuffle_test.sbatch` | does the barcode sort of the source matter? (pairs of each block in random order): +17-37% bytes | `results_raw/shuffle_*` |
| 789950, 789956, 789958, 789959 | `jobs/build.sbatch`, `jobs/meta_check.sbatch` | build with `--meta` in `dec_1fg`; metadata passthrough check: the PBMC 1k `.1fq` has an empty metadata block | `results_raw/meta_check_789959.log` |
| 789934 | `jobs/cpu_node_pipeline.sbatch` | fetch Lite -> `singlet download` local -> `enc_1fg` -> CPU identity -> dry run, SRR21102188 | `results_raw/cpupipe_*` |

Resource use (from `sacct`, allocation x elapsed, all 33 jobs of this study): **23.1 core-hours and 0.63 GPU-hours** (limits: 80 core-hours, 3 GPU-hours).
Four GPU jobs (789859 cancelled, 789881 superseded, 789904, 789930); the GPU queue was blocked for ~1.5 h by other sessions' jobs (`QOSMaxGRESPerUser`).

## Headline numbers

See the design note, section 0. Per M pairs: transform 0.24-0.42 core-s (zstd 1-3, including the `.1fq` read), reader host 0.011-0.013 core-s (M8 `.1fq`: 0.067-0.136), GPU decode 320-600 M pairs/s
(nvCOMP zstd, 128-256 frames per batch), identity 0 mismatches over 66.6 M and 121.8 M pairs.
