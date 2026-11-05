# GPU fast lane

A short A/B test for ideas on the GPU engine. It answers "is this idea promising?" in minutes.
It promotes only promising ideas to the full benchmark in `bench/gpu_batch1/`.

**Only tier 3 can support a merge or a published number.** Tier 3 is the batch 1 harness: the gate on all samples and `tp.sbatch` with 40 runs and identity on every run. Tiers 0 to 2 only choose what to send to tier 3.

Everything here is provisional. The rule values are in one place each (`abcore.DEFAULT_TIER1`, `abcore.TIER2_RULE`) and are recalibrated on known pairs.

## Tiers

| Tier | Needs | Time | Measures | Cannot detect |
|---|---|---|---|---|
| 0 `model.py` | laptop | seconds | node rate at k and slots, with what-if edits | anything the template run did not show. See the validation notes. |
| 1 `ab.py --tier 1` | `debug-gpu`, 8 CPUs | about 25 min | one sample alone, arms ABBA. The demand vector (hold by phase, core-s per run and per M, device and host peak, wall), stage core-s, identity, fallback, and the predicted node rate by limit | contention. Lock wait, device memory fallback under load, convoys. |
| 2 `ab.py --tier 2` | one H100, 28 CPUs | about 1 h | a batch check. k=8 runs in flight, 2 slots, 24 runs per leg, several arms in a palindrome order on one node: progress rate, core-s/M, lock held and wait, measured against the tier 1 prediction | rare failures. E20's out-of-memory fallback showed in 1 of 16 runs. Other samples. Identity of most runs. |
| 3 batch 1 | whole H100 node time | 4 to 6 h | everything, every run checked | nothing left. It is the reference. |

Tier 1 and 2 use the same sample (pbmc1k_v3), T=7 and 2 device slots by default. Identity is checked against a kept CPU reference.
The default reference is `$W/jobs/gpu-batch1/ref/e42c0636/T7/pbmc1k_v3`. It covers one sample at one commit.
An idea that changes output on purpose needs `--intended-output-change`. The table then reports the identity loss and the verdict ignores it.

## Demand first

The lane measures demand, not rate under load. The node rate is set by what one run asks of the node. Tier 1 measures those demands alone, with low noise.

Demand vector of an arm, from one run alone: hold seconds by phase (prefill, handoff, solo, pileup, nonhost) and in total, core-s per run and per M pairs, device peak GB, host peak GB and wall.

Limits on the sustained node rate in M pairs per second, from the demand vector (`demand.py`, `model.py limits`).

- Core limit: allocated cores over core-s per M.
- Lock limit: slots times pairs per run over hold seconds.
- Memory limit: runs that fit in (device minus 2 GB), times pairs per run over hold seconds. It binds only when fewer runs fit than there are slots.

The predicted node rate is the smallest limit. The table names the binding limit and the predicted change. It also prints the device headroom with all slots held. A small headroom flags the device out of memory risk (E20: 2.2 GB, s3: 3.4 GB).

Load factor and efficiency. One run alone costs less than a run under k=8 load. For s3 the core-s/M is 8.96 alone and 9.86 under load, and the hold is 38.2 s alone and 41.8 s under load. The limits are also upper bounds, because the measured sustained rate is below the binding limit. `python bench/gpu_fast/model.py loadfit --alone <tier 1 dirs> --loaded <tier 2 dirs> --write` fits the two load factors and the efficiency (measured progress rate over the binding limit) on legs of one binary. It leaves out legs with a fallback. `demand.py` reads `model_data/load_fit.json`. Every prediction prints both numbers. Predictions stay ratios between arms, so the factors cancel unless the binding limit changes.

These numbers are a ranking screen. Use the change between arms. The absolute rate is not a prediction. The measured sustained rate of s3 at k=8 is about 2.0, and the core limit says 3.1.

## Rules

Tier 1, the demand rule (default, `--rule demand`, values in `demand.RULE`).

- REJECT: B lost identity or fell back and A did not.
- PROMOTE: the predicted node rate rises by 5% or more (`--node-gain`) on the binding limit, with identity kept and no fallback. The rise must exceed the run-to-run spread of the binding input.
- STACK: the binding limit does not fall and a limit that does not bind rises by 10% or more (`--stack-gain`), with identity kept and no fallback. It is not promoted alone. It is listed for the batch check with the changes that lift the binding limit. The lock limit of e20w (+42% while cores bind) is the example.
- INCONCLUSIVE: everything else.
- Validity first. A fallback or identity loss in A, or no identity check, gives INCONCLUSIVE.
- `--rule gain` is the first rule: PROMOTE when core-s/M or lock held improves by `--promote-gain` (5%) and the gain exceeds the spread of the two runs per arm. `--regress-guard` can block a promotion when the other measure worsens. It is off.

Tier 2 (provisional, `abcore.TIER2_RULE`, noise from the A/A runs). Tier 2 is a batch check. See the next section.

Tier 2 (provisional, `abcore.TIER2_RULE`, noise from the A/A runs).

- REJECT: any fallback or identity loss in B, progress rate down by 9% or more, or core-s/M up by 2% or more with no rate gain.
- PROMOTE: core-s/M down by 2% or more, progress rate up by 9% or more, or lock held down by 10% or more.
- INCONCLUSIVE: everything else. REJECT is checked before PROMOTE.
- The rate is the progress-weighted rate. Each run adds its pairs times its overlap with the window over its wall. The window runs from the k-th completion to the last run start. The windowed completion rate and the span rate are shown. They never decide. Compare span only between arms with the same run count.
- Lock wait is shown and never decides. Its noise is too high.
- The job stops early at a fallback. It also stops after the third leg when B is clearly separated from A by the rate threshold.
- Core-s/M can miss a change that only shortens lock wait. The rate path covers that.

The verdict line names the rule that fired. `verdict.json` holds the numbers.

## Batch check (tier 2) and campaigns

Tier 2 runs several changes that tier 1 promoted in one load job. It reports measured against predicted.

```
python bench/gpu_fast/ab.py --tier 2 --name stack \
  --a bin=$W/jobs/gpu-next/bin/s3 --b bin=$W/jobs/gpu-next/bin/e20w \
  --arm C=bin=$W/jobs/gpu-next/bin/s3,env="SINGLET_Y=1" --arm D=bin=...,env="SINGLET_X=1 SINGLET_Y=1" \
  --predict B=out/<tier 1 dir of B> --predict C=out/<tier 1 dir of C> --predict D=out/<dir of B>,out/<dir of C>
```

- Arms A to H. A is the baseline. The default order is a palindrome over the arms (ABCCBA).
- Factorial design: give each change alone and the stack, as arms B, C and D above. Stacked arms have several `--predict` directories. Their demand ratios multiply.
- Every arm other than A gets its own verdict against A with the tier 2 rule. The table lists the predicted node rate change, then the measured progress rate, core-s/M, lock held and span rate changes, fallback runs, identity and the verdict. It also lists every leg.
- A batch check never stops early on the separation of two arms. It stops at the first fallback.

A campaign is a list of A/B pairs that runs by itself.

```
python bench/gpu_fast/ab.py --campaign campaign.json [--dry-run]
python bench/gpu_fast/ab.py --collect bench/gpu_fast/out/campaign-<name>-<time>
```

- The file format is in `campaign.py`. Each step is a tier 1 pair or a tier 2 batch check. A step can name tier 1 steps whose demand gives its predicted change.
- `--campaign` submits every step in one session as one Slurm chain. A step starts after the previous one ends, in any end state, or after the steps it names. The queue does not need the laptop.
- Each job writes its own `result.json` on Clipper and uses the scripts commit it was submitted with. Pushing later does not disturb queued steps.
- `--collect` reads the results later, writes each step's table and `verdict.json` and prints one summary. Run it again to see steps that had not finished. If the bridge is down it says so and exits.

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
| `demand.py` | demand vector, node rate limits, the demand rule, stacked predictions. |
| `campaign.py` | `--campaign` (one Slurm chain) and `--collect`. |
| `fetch_points.py` | copies the raw `point.json` of every leg of a finished run into its out directory. |
| `abcore.py` | arm parsing, verdict rules, table. Also `decide`, which the tier 2 job calls. |
| `collect.py`, `watch.py` | job side. Build `result.json`. Watch for a fallback during a leg. |
| `jobs/t1.sbatch`, `jobs/t2.sbatch` | the two jobs. They fetch scripts from the branch and check that they equal the submitted copy. |
| `jobs/lib_ident.sh`, `jobs/lib_run.sh` | identity block copied verbatim from `tp.sbatch`, and run helpers modified from it. A test checks the verbatim part. |
| `jobs/fast_build.sbatch` | singlet target only, pinned commit, no ctest. |
| `model.py`, `model_data/` | tier 0 (a ranking screen only, not an absolute prediction) and the kept inputs. `model_data/validation.md` is the validation table. |
| `SWITCHES.md` | runtime environment switches, by stage. |
| `tests/` | `python -m pytest bench/gpu_fast/tests`. `smoke_lib_run.sh` runs in WSL or Linux. |

## Fast builds

The pbench toolchain has no ccache. `MODE=auto` uses ccache when one appears and otherwise a persistent source and build tree in `$W/cache/gf-build`. A new commit is copied over the tree file by file. Only files with changed content get a new time stamp, so only they and their dependents are recompiled. Builds are serialized by a lock on the tree. `MEASURE=1` times a clean build, a build with no change and a rebuild after a one-file patch that exists only in the job.

Measured on `short-bigmem` with 24 CPUs at e42c0636 with the patch in `src/engine_gpu/records_gpu.cu`. Three jobs gave the same numbers. A clean build of the singlet target took 247 to 249 s. A rebuild with no change took 22 s. A rebuild after the one-file patch took 23 s. The 162 STAR objects are compiled again at every rebuild, so the floor is about 20 s. The cause is not found. The CUDA objects and the final link are what the tree saves.

## Tier 0 notes

The model needs one k=1 run of the same binary with its sampler file. It cuts the run at its device holds and stage boundaries. Hold segments take a slot. All segments share the cores. `gamma` slows two holders on the device. Only `gamma` is fitted. The validation table lists errors for several choices of it and says which one was used.
The sustained rate caps are printed with each prediction: cores over core-s/M, and slots times pairs per run over lock held seconds.
