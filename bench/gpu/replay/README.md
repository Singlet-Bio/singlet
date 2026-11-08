# Kernel replay: the first tier for a kernel change

A kernel-level change to the GPU port of STAR's seed and stitch code is first A/B tested by replay: STAR 2.7.11b maps a fixed set of distinct read sequences once and dumps its results (the oracle), and the GPU harness maps the same sequences and compares every result byte for byte. GPU time per variant is under a second, so the job is dominated by preparing the dump. Only a change that is identical and faster here goes on to end-to-end timing (`bench/gpu_device/jobs/gate.sbatch`, `bench/fable/release/jobs/gate.sbatch`).

Commands run on Clipper from a checkout of this repo. `W=/mnt/projects/debruinz_project/debruinz/singlet`, written out in full where a command needs it. `clipper.py sbatch LOCAL jobs/NAME` submits the same scripts from the laptop.

## The replay set is pbmc1k_v3, not pbmc2m

Every existing replay job builds its set the same way: decode the sample's R2 reads, sort and count the distinct sequences, and keep every k-th so that about `MAXD` remain. The standard set is `MAXD=2000000` on **pbmc1k_v3** (`$W/pipeline-benchmark/data/pbmc1k_v3/pbmc1k_v3.1fq`, GRCh38-2024-A): 66.6 M reads, 31.9 M distinct, every 16th kept, giving 1,991,926 distinct sequences that stand for 4,118,999 reads (job 796113). pbmc2m (`$W/pipeline-benchmark/data/pbmc2m/pbmc2m.1fq`) is used by the end-to-end gates, M6 and M13, and by no replay. To replay on pbmc2m, `fable_gate.sbatch` needs its hard-coded `F=` made overridable; the genome is the same.

The dump holds read sequences, so it lives only in the job's `/tmp` and every gate job regenerates it. All variants of one A/B therefore run inside one job, on one dump.

## Harnesses

| Harness | Replays | Status |
|---|---|---|
| `src/gpu/stardump` | the oracle: STAR's seed calls and final mapping per distinct sequence | working; binary built 2026-10-27 |
| M5 via `bench/fable/engine/fable_cc.sbatch` + `fable_gate.sbatch` | everything after read load: seed search (`k_p1s`), windows (`k_p1w`, `k_p1wp`), SA scan (`k_p2`), stitching (`k_p3`, `k_p3a`, `k_p3w`), lanes A, M, B and the host lane | working; last built 2026-11-01 at `1d1bbac4`; current tree not yet compiled |
| M5 via `bench/gpu/m5/jobs/m5_compile.sbatch` + `m5_run.sbatch` | same code, older recipe | stale: builds a hand-uploaded 2026-10-28 copy |
| E2 census (`-DM5_CENSUS`, `bench/fable/census`) | M5 plus per-read and per-window cost records | stale as written: fetches a deleted branch |
| M3/M4 `bench/gpu/m3m4/seedsearch.cu` | a standalone port of STAR's seed search, plus U/V certificates | works, but it is not the production seed kernel |

### stardump (oracle)

The vendored STAR with two files swapped in a scratch copy (see `src/gpu/stardump/README.md`). `STARDUMP_M5=PREFIX` writes the M5 truth (`PREFIX.<tid>.bin`, `PREFIX.params.txt`, `PREFIX.genome.bin`); `STARDUMP_PREFIX=PREFIX` writes every seed call for M3. The sequence cache is off (`SINGLET_STAR_SEQCACHE_MB=0`) so every sequence is mapped. The gate jobs call it themselves; on the standard set it takes 23 s with 8 threads.

- Build: `sbatch bench/gpu/jobs/stardump_build.sbatch` (`short-cpu`, 16 CPUs, 1:30). Output `$W/jobs/w11/bin/stardump`, commit in `stardump.commit`.
- The deployed binary is from `prod/campaign` `c4a1be38`. The script copies its sources from `$W/jobs/w11/stardump` (an upload that matches git) and builds against the current `prod/campaign` tip (`a29b267b`). That rebuild has not been run; it is needed only if STAR's own source changes.

### M5 replay (the A/B tool)

`fable_cc.sbatch` (`debug-cpu`, 8 CPUs, 0:29, CUDA 13 image) fetches `BRANCH` from GitHub and builds `m5.cu` into `$OUT/bin` in five flavors: `m5` (default), `m5t` (`-DM5_TAIL`), `m5c` (`-DM5_CENSUS`), `m5b` (`-DM5_BASELINE`), `m5n` (`-DM5_P1NEST`). About 4 minutes per flavor. Its default branch `gpu/fable-engine` was deleted, so always pass `BRANCH`, and the change must be pushed first.

```bash
BRANCH=gpu/frontier OUT=/mnt/projects/debruinz_project/debruinz/singlet/jobs/replay/base sbatch bench/fable/engine/fable_cc.sbatch
```

`fable_gate.sbatch` (`debug-gpu`, 1 H100 NVL, 8 CPUs, 100 GB, 0:29) builds the standard set, runs stardump, then each variant of `M5RUNS` (`binary:args|binary:args`, binaries from `$OUT/bin`, `@STD` expands to the standard budgets and batches) with `--reps 3`:

```bash
OUT=/mnt/projects/debruinz_project/debruinz/singlet/jobs/replay/base TAG=ab M5RUNS='m5:@STD|m5:@STD --best-order' sbatch bench/fable/engine/fable_gate.sbatch
```

The engine's default configuration is the gate's last default variant: `m5:@STD --best-order --p3-maxreg-a 64 --batch-b 131072 --warp-m --wsort --wthreads 128` (`engine_gpu.cu`, "the m5 harness's best exact configuration"). Use it as the A side unless the change targets another configuration.

Per variant the log prints, and `$OUT/results/<TAG>_<job>_v<i>_<bin>.{log,json}` keeps:

- one line per lane and repeat: `lane A run 2: 177.7 ms (phase 1 79.0, scan 14.4, phase 3 84.3; seeds kernel 57.6)`, with register count, local memory and resident blocks per SM on the lane header;
- the identity line: `M5: 1991926 / 1991926 distinct reads identical (100.000000%), reads weighted 4118999 / 4118999 (100.000000%); differ 0`, split into STAR-mapped and STAR-unmapped;
- the total GPU rate (`= 3.59 M distinct reads/s on the GPU`). JSON keys: `gpu_ms_median`, `lane_{a,m,b}_ms`, `lane_a_phase_ms`, `differ_distinct`, `heavy_reads`, `heavy_failures`.

Measured run time (job 796113, five variants): index copy 13 s, decode and dedup 152 s, stardump 23 s, then about 33 s per variant, of which about 20 s is index load and 0.56 to 0.86 s per repeat is GPU time. Total 353 s.

`bench/fable/order/e4m_gate.sbatch` is the same gate with E4's variant list. It reads binaries from `e4m_cc.sbatch`, whose default branch `gpu/fable/order` was also deleted.

### Older M5 recipe (stale)

`m5_compile.sbatch` compiles `$W/jobs/w11/m5/m5.cu`, a hand upload from 2026-10-28 that predates E3 to E7, with the CUDA 12.4 image. `m5_run.sbatch` (`short-gpu`, 28 CPUs, 240 GB, 2:30) reads `SLURM_ARRAY_TASK_ID` (0 pbmc1k_v3, 1 pbmc5p_v1, 2 nuclei2k_v2) but declares no array, and defaults to `MAXD=10000000`. Use the fable gate instead.

### E2 census (stale scripts, live code)

`m5c` writes per-read and per-window records (`--census PREFIX`) that `census_agg.py` turns into cost tables (`docs/gpu-engine/fable/E2-census.md`). It is a profiler for choosing what to change; its identity check matches `m5`, and it costs about 4.5% GPU time. `census_compile.sbatch` and `census_run.sbatch` both fetch `gpu/fable/census`, which no longer exists, so they fail until those fetches point at a live branch. `M5_CENSUS`, `census.cuh` and `census_agg.py` are in the tree, and `fable_cc` builds `m5c`.

### M3/M4 seedsearch (validation tool)

`seedsearch.cu` holds its own line-by-line port of `maxMappableLength2strands` and checks every dumped seed call (maxL, Nrep, SA interval, branch). It has been unchanged since 2026-10-27. Production seed search is `k_p1s` in `stitch.cuh`, which M5 covers and times as "seeds kernel", so a seed-kernel change is tested in M5, not here. Build with `sbatch bench/gpu/jobs/m3m4_compile.sbatch` (`debug-cpu`; compiles the upload in `$W/jobs/w11/m3m4`); run one sample with `MAXD=2000000 sbatch --array=0 bench/gpu/jobs/m3m4_run.sbatch` (`short-gpu`, 28 CPUs, 240 GB, 2:30). Output `$W/jobs/w11/results/m3m4_<label>.json`. Recorded: 104,010,422 of 104,010,422 records identical on pbmc1k_v3 at `MAXD` 10 M, 101 to 105 ms per pass, index load 6.4 s; total job time was not recorded.

## What replay covers

The engine includes `bench/gpu/m5/stitch.cuh` itself, so any change to a device function there is replayed exactly. `m5.cu` has its own kernel wrappers and launch code, though. Kernels that only `engine_gpu.cu` defines (`k_p3e`, `k_p3c`, `k_wplan`, `k_wkey`, the record, clip and hash kernels), its chunking and streams, and the WSHRINK and BSTREAM paths of `21dbe9d0` have no replay. Port the change into `m5.cu`'s wrappers first, or accept end-to-end timing as the first measurement.

## A/B recipe

1. Put the change behind an `m5` flag, or push it to its own branch.
2. Flag: one build, `M5RUNS='m5:@STD <A args>|m5:@STD <A args> --new-flag|m5:@STD <A args>'`.
3. Branch: build the baseline (`BRANCH=gpu/frontier OUT=$W/jobs/replay/base`) and the candidate (`BRANCH=<yours> OUT=$W/jobs/replay/cand`), copy `base/bin/m5` to `cand/bin/m5base`, then gate with `OUT=$W/jobs/replay/cand M5RUNS='m5base:@STD <A args>|m5:@STD <A args>|m5base:@STD <A args>'`.
4. Pass: the identity line shows `differ 0` and 100% for every variant, and the candidate's median lane times beat both A runs by more than the A-to-A spread. Repeating A at the end exposes drift between runs on the node.
5. Record the job id, both commits (`$OUT/bin/commit`) and the lane medians; only then schedule end-to-end timing.

## Not confirmed

- That the current tree's `m5.cu` compiles: the last build is `1d1bbac4`, before `727f470b` (macro pruning, preprocessor-identical by its own check) and `21dbe9d0` (stitch.cuh changes).
- That `stardump_build.sbatch` still builds against the current `prod/campaign` tip.
- The distinct-sequence count of pbmc2m, and replay times on it.
