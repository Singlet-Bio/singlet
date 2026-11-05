# GPU batch 1 harness

Jobs for comparing GPU engine builds. The code under test is never part of this branch. Each job takes a binary directory and checks its recorded commit.

All results go under `$W/jobs/gpu-batch1/` with `W=/mnt/projects/debruinz_project/debruinz/singlet`.

## Rules shared by all jobs

- `BINDIR` has no default. `EXPECT_COMMIT` is required. The job stops unless `BINDIR/commit-gpu` starts with it.
- The first log line is `PROVENANCE job=... node=... bin=... sha256=... commit=... scripts=... T=...`. Every result table repeats these as columns.
- Scripts come from the `bench/gpu-batch1` branch. The job fetches the tip under `flock` and checks that it got the tip it asked for. `EXPECT_SCRIPTS` pins a commit prefix.
- Submit from the login node with the script copy under `$W/jobs/gpu-batch1/jobs/`. Put it there with `clipper.py put` and give an absolute remote path.

## gate.sbatch

Runs each sample in `MODES` and compares GPU to CPU. The CPU reference is that binary's own CPU mode. It is kept at `$W/jobs/gpu-batch1/ref/<commit8>/T<T>/<sample>`.

| Parameter | Meaning |
|---|---|
| `BINDIR`, `EXPECT_COMMIT` | Binary and its commit. Required. |
| `T` | Threads. Default 18. |
| `SAMPLES` | Default is the ten samples of HANDOFF section 1. The tenth is `pbmc1k_cite_v3_adt`, found under `$W/jobs/fable/release/data`. The long-read slice is `gsm7064234s`. |
| `MODES` | Default `cpu gpu`. Other modes are described in the script header. |
| `REFRESH_REF=1` | Runs the CPU mode again and replaces the reference. |

Memory is 200G. Use `--mem=370G` for `gsm7064234s`. The job stops with less.

Output lines to read:

- `VERDICT ...` is the old identity verdict. It is unchanged.
- `FIELDDIFF <file> <field> a b` and `FIELD CHECK: n unexplained` look into files that the verdict excuses by name. They never change the verdict.
- `GATE FALLBACK <sample>` marks a disallowed fallback line in a GPU run. It fails the gate.
- `MASK <sample> <mode>: ...` states whether the low-quality mask is active and its width.

Example, full gate of a binary with references at its own commit:

```
cd $W/jobs/gpu-batch1 && BINDIR=$W/jobs/gpu-next/bin/s3 EXPECT_COMMIT=e42c0636 sbatch jobs/gate.sbatch
```

Long-read slice:

```
cd $W/jobs/gpu-batch1 && BINDIR=$W/jobs/gpu-next/bin/s3 EXPECT_COMMIT=e42c0636 SAMPLES=gsm7064234s sbatch --mem=370G --time=03:59:00 jobs/gate.sbatch
```

## tp.sbatch

One or more points of node throughput. A point tag is `k<K>t<T>s<S>r<R>`. K runs are in flight, T threads each, S device slots, and K times R runs in the queue. `NRUNS` sets the queue length directly. `k8t7s2r5` is 40 runs.

Before a point the job runs each sample once alone at the same T and slots. Every run is then compared to that alone run and to the CPU reference. The outputs of a run are deleted after its comparison.

| Parameter | Meaning |
|---|---|
| `BINDIR`, `EXPECT_COMMIT` | Binary and its commit. Required. |
| `POINTS` | Tags in order. |
| `NRUNS` | Queue length of every point. Replaces K times R. |
| `SAMPLE`, `MIX` | Sample of the queue. |
| `CPU_REF` | Directory of the gate's references for this binary. Default `$W/jobs/gpu-batch1/ref/<commit8>`. The reference of a run is `<CPU_REF>/T<T>/<sample>`. |
| `IDENT=0` | Skips identity. |

Example, 40 runs at k=8 on a named node:

```
cd $W/jobs/gpu-batch1 && BINDIR=$W/jobs/gpu-next/bin/s3 EXPECT_COMMIT=e42c0636 SAMPLE=pbmc1k_v3 POINTS=k8t7s2r5 sbatch -w <node> jobs/tp.sbatch
```

A window needs more than 2k runs. With 2k runs or fewer `win_rate` is `NA`.

The reference for pbmc1k_v3 at T=7 must exist before the job. Run the gate with `T=7 SAMPLES=pbmc1k_v3 MODES=cpu`.

### results.tsv columns

One row per point. `RES0/results.tsv` holds all points of the job.

| Column | Meaning |
|---|---|
| `job`, `node`, `bin`, `bin_sha256`, `build_commit`, `scripts_commit`, `T` | Provenance. |
| `point`, `k`, `slots`, `runs`, `failed_runs` | Point tag, runs in flight, device slots, runs in the queue, runs with a non-zero exit. |
| `span_s`, `span_rate` | First start to last end, and M pairs per second over it. |
| `steady_rate` | The old steady number: runs started after the first completion, k times mean pairs over mean wall. |
| `win_rate`, `win_n`, `win_s` | Windowed rate in M pairs per second, completions inside the window, window length. The window starts when the k-th run completes and ends when the last run starts. `NA` when the window is empty. |
| `core_s_per_M` | Cgroup core-seconds over the span per million pairs. |
| `cores_span`, `cores_win` | Mean cores busy: cgroup core-seconds over the span, and inside the window. |
| `lock_wait_s`, `lock_wait_frac` | Mean seconds per run blocked on a device lock, and that as a fraction of the mean wall. |
| `lock_held_s` | Mean seconds per run holding a device lock. |
| `gpu_util` | Mean GPU utilization over the span, percent. |
| `host_peak_gb`, `dev_peak_gb` | Peak cgroup memory and peak device memory. |
| `fallback_runs` | Runs with level 5 or the GPU pileup missing, or a disallowed fallback line. The log shows `POINT FAILED: n fallback runs`. |
| `ident_alone`, `ident_cpu` | Runs identical to the alone run and to the CPU reference, over runs checked. |
| `other_jobs_max` | Most other jobs seen on the node in one minute. See `cotenants.log` in the point directory. |
| `mean_wall_s` | Mean wall time of a run. |

## gpu_ctest.sbatch

Builds a commit with `-DSINGLET_ENGINE_GPU=ON` in the CUDA image. Then it runs `test_gpu_pileup_labels` and the whole suite on the node itself. Inside the image the test binaries need a newer glibc than the image has. Parameters are `REV` (branch or full commit), `EXPECT_COMMIT` and `ONLY_GPU_TEST=1` for a quick check. The log ends with `GPU TEST test_gpu_pileup_labels: PASS|FAIL|SKIP` and `CTEST SUITE: ...`. Results are in `$W/jobs/gpu-batch1/ctest-<commit8>/`.

```
cd $W/jobs/gpu-batch1 && REV=<full commit> EXPECT_COMMIT=<commit8> sbatch jobs/gpu_ctest.sbatch
```

## Allow-lists

`fallback.py` and `fieldcheck.py` hold explicit lists with a reason for each entry. Add an entry only with a reason that can be stated.

`orig/` holds `tp-run.sbatch` and `pool-run.sbatch` as found on Clipper.
