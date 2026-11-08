# Portable GPU timing bundle

This bundle times the GPU engine (`singlet --engine gpu`) on a rented cloud GPU with the same sample, settings and command line as
Clipper's throughput harness (`bench/gpu/pilot/tp_gc.sbatch`, config `base`). A point measured here can be set beside a Clipper H100 NVL
point of the same tag and commit.

| File | Purpose |
|---|---|
| `Dockerfile` | CUDA 12.4.1 devel Ubuntu 22.04 image with the binary, its whitelists and these scripts |
| `build.sh` | the build itself: Clipper's recipe (conda toolchain, `-march=x86-64-v3`, the `bam_cat.c` declaration, the cmake line of `scripts/engine_gpu/build_clipper.sbatch`) |
| `fetch.sh` | public inputs: 10x GRCh38-2024-A, Clipper's STAR 2.7.11b index and non-host indices (published tars, checksums checked), the SNP panel, pbmc1k_v3 as `.1fq` |
| `publish/` | the jobs that packed and published the reference tars, and the proof that a download reproduces Clipper's output |
| `run.sh` | one timing point: N copies of pbmc1k_v3, K in flight per GPU, with per-run wall time, core-seconds and a summary |

## What the machine needs

- **GPU.** An H100 (sm_90) or A100 (sm_80). The engine sizes its device budget from the card's memory. Clipper's H100 NVL has 94 GB,
  while most cloud H100s have 80 GB; E17 measured about 2 s more per single run at an 80 GB budget (`docs/gpu-engine/fable/E17-budget.md`).
- **Driver.** NVIDIA driver 550 or newer, which supports the CUDA 12.4 runtime linked into the binary. Clipper runs 610.43.02.
- **CPU and RAM per GPU in use.** The default point (8 in flight, 7 threads each) needs about 28 physical cores and 400 GB of RAM per
  GPU. Each sample peaks near 50 GB of host RAM. Clipper gives each GPU 28 cores (Xeon Gold 6238R) and 370 GB, and its 8-in-flight
  points peaked at 300 to 357 GB. A typical 8-GPU H100 node (about 100 cores and 2 TB) can therefore time one GPU at Clipper's density,
  or up to four GPUs at once.
- **Disk.** About 250 GB of local NVMe for `/data`. The STAR index tar is 22.9 GB, the non-host tar 26.5 GB (each deleted once unpacked),
  the `.1fq` 1.8 GB, and the 10x downloads are deleted after extraction. Each run in flight writes its outputs to `/data/scratch` and deletes them when it ends.
- **Software.** Docker with the NVIDIA Container Toolkit, which most GPU cloud images include. `docker run` needs `--gpus all` and
  `--ipc=host`, because the engine hands reads between processes through `/dev/shm`.

## Commands

On the VM, with the bundle on the branch you want to time:

```bash
git clone --depth 1 -b gpu/frontier https://github.com/Singlet-Bio/singlet && cd singlet/bench/gpu/portable
sudo mkdir -p /data && sudo chown "$USER" /data

# 1. Build the image and download the inputs at the same time. Pin SINGLET_REF to a commit for comparisons.
docker build -t singlet-gpu-bench --build-arg SINGLET_REF=gpu/frontier . > build.log 2>&1 &
DATA=/data bash fetch.sh download
wait

# 2. .1fq encoding, checksums against state/reference-manifest-v1.yaml (the STAR index came as a tar in step 1)
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v /data:/data singlet-gpu-bench fetch.sh prepare

# 3. The timing point (k8t7s2r5 on GPU 0). Results land in /data/results/tp-<time>-k8t7s2r5/.
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v /data:/data singlet-gpu-bench run.sh
```

`run.sh` takes its settings from the environment (`-e K=8 -e T=7 -e SLOTS=2 -e ROUNDS=5 -e GPUS="0 1"`). Each GPU in `GPUS` gets its
own queue, pinned with `CUDA_VISIBLE_DEVICES`. The last line printed is the result:

```
RESULT steady <x.xxx> M read pairs/s per GPU (<n> pairs/s); span <x.xxx> M/s per GPU; k8t7s2r5 on 1 x <GPU> (driver <d>); commit <c>; ...
```

On a container platform without Docker builds, start a container from `nvidia/cuda:12.4.1-devel-ubuntu22.04` with a large `/dev/shm`
and a volume at `/data`, clone the repository, run `bash build.sh` as root, then
`export SINGLET_BIN=/opt/singlet/bin/singlet PATH=/opt/singlet/env/bin:$PATH` and run `fetch.sh` and `run.sh` as above.

### Expected durations

| Step | Duration | Source |
|---|---|---|
| Image build | not measured | conda solve plus the CUDA and STAR compile |
| Downloads | not measured | the 10x reference tarball and the FASTQ tar, from 10x's CDN |
| Reference tars | 49 GB from data.singlet.bio; the STAR index takes about an hour to build instead (`BUILD_STAR=1`) | `fetch.sh download` |
| One pbmc1k_v3 run alone | 126 s at 18 threads on an H100 NVL | Clipper job 800981 |
| k8t7s2r5 (40 runs) | not yet measured on one pinned H100 NVL | the GPU pilot (earlier jobs ran on two GPUs) |

Clipper's `star_2.7.11b/` and its non-host indices are published at `https://data.singlet.bio/reference/v1/`, with
`manifest.json` listing every file's source, license, build command, size and sha256. `fetch.sh` downloads both tars and stops on a
checksum mismatch.

## Comparing with Clipper

Compare points that share the sample (pbmc1k_v3), the tag (k8t7s2r5: 40 runs, 8 in flight, 7 threads, 2 device slots) and the binary's
commit. The commit is in `meta.txt` and every `summary.tsv` row. For a fresh reference, queue the same commit on Clipper as a
`k8t7s2r5@base` item of the GPU pilot (`bench/gpu/pilot/README.md`).

| This bundle (`summary.tsv`) | Clipper harness (`results.tsv`, table) |
|---|---|
| `span_rate_Mps` | `span_rate`, "pairs/s (M)" |
| `steady_rate_Mps` | `steady_rate` (runs started after the first completion, as `bench/fable/throughput/e18_steady.py`) |
| `wall_mean_s` | `mean_wall_s` |
| `cgroup_core_s_per_M` (node row) | `core_s_per_M` (cgroup) |
| `host_peak_gb`, `dev_peak_mb` | `host_peak_gb`, `dev_peak_gb` |
| `fallback_runs` | `fallback_runs` |

Use only Clipper points whose log shows the pin check (`pin check ...: GPUs used by the copies [0]`), which starts with the GPU pilot.
Earlier Clipper throughput jobs (800984, 800985, 803326, 803458 and before) exported `CUDA_VISIBLE_DEVICES=0`, but it never reached the
runs: their samples ran on both GPUs of the node, so their rates are per node with two GPUs, not per GPU. The non-host screen also
can fall back to the CPU (`IDEAS.md` items 1 and 11). Compare `fallback_runs` before comparing rates.

The host CPU differs on every cloud machine, and most of a run's core-seconds are host work (STAR's host mapping, demultiplexing, the
writers). Read `core_s_per_M` beside the rates, and record the CPU model from `meta.txt` with every result.

`run.sh` writes `comparable_to_clipper_base=no` when the point differs from Clipper's base config: the non-host screen off (indices
missing), no SNP panel, or extra arguments. A point with the screen off compares with a Clipper `k8t7s2r5@nonh` point instead.

## Open items

- **SNP panel.** `fetch.sh` derives Clipper's `genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz` from the cellsnp-lite
  list by adding `chr` and sorting. The command that made Clipper's copy is not recorded, so the panel's column order may differ.
- **The `.1fq`.** `fetch.sh` encodes the two lanes with `singlet encode --quality none`. Clipper's file has the same 66,601,887 reads; its
  encoder version is not recorded, and `fetch.sh` reports when the size differs from 1,838,043,689 bytes.
- **Compiler.** `build.sh` takes the newest conda gcc, as Clipper's env did (gcc 16 there), with `-allow-unsupported-compiler`. If nvcc
  rejects it, rebuild with `--build-arg GCC_SPEC==13`.

## Owner checklist

1. Create an account with a GPU cloud provider and add a payment method.
2. Request H100 quota if the provider requires it.
3. Pick an instance with at least one H100, 28 physical cores and 400 GB of RAM per GPU to be timed, and 250 GB of local NVMe. Pick a
   GPU image with driver 550 or newer, Docker and the NVIDIA Container Toolkit, and add an SSH key.
4. Start the instance and run the three commands above.
5. Copy `/data/results/` back (it holds only logs and tables) and stop or delete the instance to end billing.
