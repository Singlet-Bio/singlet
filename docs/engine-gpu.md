# GPU engine (`singlet --engine gpu`)

The `singlet` binary can map reads on an NVIDIA GPU. `--engine gpu` gives the same outputs as `--engine cpu` (the default),
file by file, and runs anywhere the CPU build runs: without a GPU, or in a build without CUDA, it maps on the CPU and says so in
the STAR log.

Status (W11, 2026-10-28): the GPU maps every distinct cDNA sequence of a sample with a line-by-line port of STAR 2.7.11b's
mapping (seed search, windows, stitching, extension, scoring, multimapper selection, filters); STAR's mapping state for each
sequence is byte-identical to STAR's on every sequence tested, and a whole sample run through the engine had 0 differences in
STAR's own verification (66.0 M reads). Every other stage (barcodes, BAM, pileup, cell calling, export, non-host screen, demux)
is the CPU engine's code in this release. Evidence and measurements: `docs/gpu-engine/exploration.md` §12.12–12.17 on the
research branch (W11, PR Singlet-Bio/singlet#61) and the measurement section of this PR.

## How it works

STAR's sequence cache (`src/star/SeqCache.h`) stores, per distinct read sequence, the complete result of mapping it, and the
engine already uses it: a read whose sequence was mapped before takes the stored result (`seqCacheRestore`). With
`--engine gpu`, before STAR maps any read (`src/star/SeqCacheGpuPrefill.cpp`):

1. the cDNA reads are read from STAR's input file and copied to the GPU;
2. the GPU hashes every read, sorts, and keeps one read per distinct sequence (each group is compared byte for byte; a hash
   collision would turn the GPU path off for the sample, so results never depend on the hash);
3. the GPU maps each distinct sequence (`src/engine_gpu/stitch.cuh`, `engine_gpu.cu`) through a cascade of lanes with growing
   per-read capacities; the few sequences that exceed them run through the same code on the host with STAR's own limits;
4. the results go into STAR's cache; STAR then runs as usual, and almost every read is a cache hit.

The GPU path is used when the cache key is the raw read sequence: single-end cDNA from regular files, no read clipping
(`--clip3pAdapterSeq`, `--clip5pNbases`), the multimapper and filter options the campaign uses. Otherwise STAR maps every read
and the log says why (`ENGINE_GPU: not used (...)`). 3′ samples qualify; 5′ samples with adapter clipping do not yet.

## Build

Requirements: what the CPU build needs (see `docs/install.md`), plus the CUDA toolkit 12.x with a host compiler it supports.
GPUs of compute capability 8.0 or newer (A100, L40S, H100, H200).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON            # SINGLET_ENGINE_GPU=AUTO: on if nvcc is found
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_ENGINE_GPU=ON -DCMAKE_CUDA_ARCHITECTURES="80;90"
cmake --build build --target singlet -j
```

- `SINGLET_ENGINE_GPU=OFF` gives the CPU-only binary, unchanged; `ON` fails if no CUDA compiler is found.
- The CUDA runtime is linked statically; the binary needs only the NVIDIA driver at run time, and runs on machines without one.
- The port is compiled with `-fmad=false` (STAR's double arithmetic must round as on the host). Do not add `--use_fast_math`.
- Clipper recipe (conda toolchain inside the CUDA devel image): `scripts/engine_gpu/build_clipper.sbatch`.

## Use

```bash
singlet SAMPLE.1fq --pipeline --ref-base REF --genome-dir REF/GRCh38-2024-A/star_2.7.11b --exons REF/GRCh38-2024-A/genes/genes.gtf \
    --out-prefix out/ --threads 24 --engine gpu
```

- `SINGLET_ENGINE_GPU_DEVICE=N` selects the GPU (default 0). Two samples per GPU fit on a 94 GB H100 NVL (the genome index takes
  ~18 GB of GPU memory per process while the sample is mapped).
- `SINGLET_STAR_SEQCACHE_MB` sizes STAR's cache; with `--engine gpu` it defaults to ~0.6× the cDNA FASTQ size (4–96 GB) so that
  every distinct sequence fits.
- The STAR log (`star_Log.out`) reports `engine_gpu: ... reads, ... distinct, ...` and `ENGINE_GPU: cache filled with ...`, and
  the cache's hit rate at the end.

## Check identity yourself

```bash
scripts/engine_gpu/identity_test.sh build/src/pipeline/singlet SAMPLE.1fq /scratch/idtest
```

It runs the sample with `--engine cpu`, with `--engine gpu` plus `SINGLET_STAR_SEQCACHE_VERIFY=1` (STAR maps every read that took a
GPU result and counts differences: it must print `verify_mismatches=0`), and with `--engine gpu`, then compares every output file
(`bench/star_accel/out_equiv.py`; files that differ are compared again with their lines or FASTQ records sorted). Two CPU runs of
the engine already differ in the order of the unmapped-read files and in timings (`RUN_CPU2=1` shows it); the script fails only
on data files that differ beyond that.

## Reproduce the pipeline end to end

1. **Download** a public run and transform it to `.1fq` (barcodes, UMIs, 2-bit sequence; `docs/format-1fq.md`):
   `singlet download SRRxxxxxxx -o sample.1fq` (or `singlet encode` for FASTQ files you have).
2. **References**: the 10x GRCh38-2024-A / GRCm39-2024-A references with the STAR 2.7.11b index the campaign uses
   (`--genomeSAsparseD 2` for GRCh38: do not rebuild the index with other settings, it changes STAR's results), the whitelists
   (`SINGLET_WHITELIST_DIR`), and the reference base layout under `--ref-base` (SNP panel, non-host indices).
3. **Engine**: the command under *Use* (`--engine cpu` on a CPU node, `--engine gpu` on a GPU node: same outputs).
4. **Outputs**: `docs/pipeline-outputs.md` (matrices in `.1pz`, cell table, QC, junctions, SNP and mtDNA, non-host summary).
5. **Check**: `scripts/engine_gpu/identity_test.sh` as above.
