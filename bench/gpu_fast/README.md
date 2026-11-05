# GPU fast lane

A short A/B test for ideas on the GPU engine. It answers "is this idea promising?" in minutes.
It promotes only promising ideas to the full benchmark in `bench/gpu_batch1/`.

**Only tier 3 can support a merge or a published number.** Tier 3 is the batch 1 harness: the gate on all samples and `tp.sbatch` with 40 runs and identity on every run. Tiers 0 to 2 only choose what to send to tier 3.

Everything here is provisional. The rule values are in one place each (`abcore.DEFAULT_TIER1`, `abcore.TIER2_RULE`) and are recalibrated on known pairs.

## Tiers

| Tier | Needs | Time | Measures | Cannot detect |
|---|---|---|---|---|
| 0 `model.py` | laptop | seconds | node rate at k and slots, with what-if edits | anything the template run did not show. See the validation notes. |
| 1 `ab.py --tier 1` | `debug-gpu`, 8 CPUs | about 25 min | one sample alone, arms ABBA: wall, core-s/M, lock held, device and host peak, stage core-s, identity, fallback | contention. Lock wait, device memory fallback under load, convoys. |
| 2 `ab.py --tier 2` | one H100, 28 CPUs | about 1 h | k=8 runs in flight, 2 slots, 24 runs per leg, ABBA on one node: progress rate, core-s/M, lock held and wait | rare failures. E20's out-of-memory fallback showed in 1 of 16 runs. Other samples. Identity of most runs. |
| 3 batch 1 | whole H100 node time | 4 to 6 h | everything, every run checked | nothing left. It is the reference. |

Tier 1 and 2 use the same sample (pbmc1k_v3), T=7 and 2 device slots by default. Identity is checked against a kept CPU reference.
The default reference is `$W/jobs/gpu-batch1/ref/e42c0636/T7/pbmc1k_v3`. It covers one sample at one commit.
An idea that changes output on purpose needs `--intended-output-change`. The table then reports the identity loss and the verdict ignores it.

## Rules

Tier 1 (provisional, options of `ab.py`).

- REJECT: B lost identity or fell back and A did not.
- PROMOTE: core-s/M or lock held improves by at least `--promote-gain` (5%), and the gain exceeds the run-to-run spread of the two runs per arm.
- INCONCLUSIVE: everything else.
- Validity first. A fallback or identity loss in A, or no identity check, gives INCONCLUSIVE. `--regress-guard` can block a promotion when the other measure gets worse. It is off.

Tier 2 (provisional, `abcore.TIER2_RULE`, noise from the A/A runs).

- REJECT: any fallback or identity loss in B, progress rate down by 9% or more, or core-s/M up by 2% or more with no rate gain.
- PROMOTE: core-s/M down by 2% or more, progress rate up by 9% or more, or lock held down by 10% or more.
- INCONCLUSIVE: everything else. REJECT is checked before PROMOTE.
- The rate is the progress-weighted rate. Each run adds its pairs times its overlap with the window over its wall. The window runs from the k-th completion to the last run start. The windowed completion rate and the span rate are shown. They never decide. Compare span only between arms with the same run count.
- Lock wait is shown and never decides. Its noise is too high.
- The job stops early at a fallback. It also stops after the third leg when B is clearly separated from A by the rate threshold.
- Core-s/M can miss a change that only shortens lock wait. The rate path covers that.

The verdict line names the rule that fired. `verdict.json` holds the numbers.

## Run

Push the scripts first. The jobs fetch them from `bench/gpu-fast` and check the commit. `ab.py` refuses to run with local changes or an unpushed head.

Tier 0, no cluster.

```
python bench/gpu_fast/model.py validate
python bench/gpu_fast/model.py predict bench/gpu_fast/model_data/s3-k1.compact.json --k 8 --slots 2 --runs 40 --gamma 0 --scale demux_core_s=0
```

Tier 1.

```
python bench/gpu_fast/ab.py --tier 1 --name e20-vs-s3 \
  --a bin=$W/jobs/gpu-next/bin/s3,commit=e42c0636 --b bin=$W/jobs/gpu-next/bin/e20w,env="SINGLET_X=1"
```

Tier 2.

```
python bench/gpu_fast/ab.py --tier 2 --name e20-vs-s3 \
  --a bin=$W/jobs/gpu-next/bin/s3 --b bin=$W/jobs/gpu-next/bin/e20w --runs 24 --k 8
```

Use `--dry-run` to print the spec and the sbatch lines. Each arm takes `env="A=1 B=2"` and `args="--flag"`. Build a binary at a pinned commit with `jobs/fast_build.sbatch`.
The binary check is the batch 1 one: the directory records a commit, the sha256 is stored, and the commit string must be inside the binary. Pass `commit=` to also require a given commit.

A dropped bridge is treated as unknown state. `ab.py` retries for `--bridge-grace` minutes, then exits with code 3 and prints the `--resume` command. It never starts the bridge.

## Files

| File | Use |
|---|---|
| `ab.py` | the command. Submits, waits, prints the table and verdict, writes `verdict.json`. |
| `abcore.py` | arm parsing, verdict rules, table. Also `decide`, which the tier 2 job calls. |
| `collect.py`, `watch.py` | job side. Build `result.json`. Watch for a fallback during a leg. |
| `jobs/t1.sbatch`, `jobs/t2.sbatch` | the two jobs. They fetch scripts from the branch and check that they equal the submitted copy. |
| `jobs/lib_ident.sh`, `jobs/lib_run.sh` | identity block copied verbatim from `tp.sbatch`, and run helpers modified from it. A test checks the verbatim part. |
| `jobs/fast_build.sbatch` | singlet target only, pinned commit, no ctest. |
| `model.py`, `model_data/` | tier 0 and the kept inputs. `model_data/validation.md` is the validation table. |
| `SWITCHES.md` | runtime environment switches, by stage. |
| `tests/` | `python -m pytest bench/gpu_fast/tests`. `smoke_lib_run.sh` runs in WSL or Linux. |

## Fast builds

The pbench toolchain has no ccache. `MODE=auto` uses ccache when one appears and otherwise a persistent source and build tree in `$W/cache/gf-build`. A new commit is copied over the tree file by file. Only files with changed content get a new time stamp, so only they and their dependents are recompiled. Builds are serialized by a lock on the tree. `MEASURE=1` times a clean build, a build with no change and a rebuild after a one-file patch that exists only in the job.

Measured on `short-bigmem` with 24 CPUs at e42c0636 with the patch in `src/engine_gpu/records_gpu.cu`. Three jobs gave the same numbers. A clean build of the singlet target took 247 to 249 s. A rebuild with no change took 22 s. A rebuild after the one-file patch took 23 s. The 162 STAR objects are compiled again at every rebuild, so the floor is about 20 s. The cause is not found. The CUDA objects and the final link are what the tree saves.

## Tier 0 notes

The model needs one k=1 run of the same binary with its sampler file. It cuts the run at its device holds and stage boundaries. Hold segments take a slot. All segments share the cores. `gamma` slows two holders on the device. Only `gamma` is fitted. The validation table lists errors for several choices of it and says which one was used.
The sustained rate caps are printed with each prediction: cores over core-s/M, and slots times pairs per run over lock held seconds.
