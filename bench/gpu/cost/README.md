# W11 CPU baseline: cost of the campaign engine on the six real samples

Binary: `$W/pipeline-benchmark/bin/singlet-w3`, built by job 788956 from commit **bb51b063** with 3'/5' PGO (same `pgo_build.sh` recipe
as the campaign build). `git diff bb51b063 origin/prod/campaign` (= c4a1be38) touches only `bench/star_accel/w3_timing.sbatch`, so the sources
are identical to **prod/campaign c4a1be38** (FASTASSIGN #41, #62 serial-stage speed-ups, 3'/5' PGO). The binary's embedded commit string reads
bb51b063, not c4a1be38; no new binary was built. `singlet-campaign` on Clipper is the older 25 Oct build and was not used.

Command line (as `bench/gpu/host`): `<.1fq> --threads 4 --pipeline --output-format 1pz --bam-sort-mode disk --genome-shared off`, `--run-screen`
when a `screen.json` exists, non-host screen on. One sample per job on `short-cpu`, 8 cores allocated, `--nice=10000`. Reads, BAM and outputs stay in job `/tmp`.

## Files
- `run_single.sbatch` (one sample per array task), `run_pack.sbatch` (several samples concurrently on one exclusive node), `parse_cost.py`, `make_baseline.py` -> `cpu_baseline.json`.
- `runs/single/<sample>/` run.log, time.txt, node.txt, monitor.tsv (cgroup CPU and log size every 2 s), stages.json. `runs/hom4_t4/` packing run, `runs/hom8_t4_stalled/` the hung run.

## Method
- Wall, user+sys core-s and peak RSS: `/usr/bin/time -v` of the whole process tree (the cgroup CPU counter agrees within 0.5%). Core-s are the cost number: wall depends on the node.
- Stage windows: the monitor logs the size of run.log with the cgroup CPU every 2 s, so the time of a marker line is the first sample whose size covers it; wall and core-s
  between markers give STAR map+solo (incl. pass 1 and reference load), sort, index+CB refresh+pileup, non-host index load, non-host classify/EM/writes, export (incl. demux). Engine stage times are also in the JSON.
- Credited core-s: measured total minus the non-host index-load window (CPU-bound: core-s equals wall, 36-53 s), i.e. as if a persistent per-node service loaded the three indices once. The section 12.9 load numbers are listed beside them (other binary and CPU).

## Results (single samples, measured; CPU models differ per node, so wall is not comparable between rows)
| sample | M pairs | wall s | core-s | core-s per M | credited per M | node (CPU) | RSS GB |
|---|---|---|---|---|---|---|---|
| GSM4006843 | 66.7 | 972 | 2598 | 38.9 | 38.2 | c010 Gold 6230R | 54.5 |
| GSM5411066 (mouse) | 69.6 | 691 | 1639 | 23.6 | 22.9 | c003 Gold 6248 | 51.0 |
| GSM7901264 | 100.3 | 1173 | 3387 | 33.8 | 33.4 | c003 Gold 6248 | 51.8 |
| GSM7036403 | 111.6 | 1261 | 3205 | 28.7 | 28.3 | c003 Gold 6248 | 55.1 |
| GSM6470208 (5') | 121.8 | 2098 | 5282 | 43.4 | 43.1 | c003 Gold 6248 | 58.0 |
| GSM7656149 | 263.4 | 4129 | 11661 | 44.3 | 44.1 | c003 Gold 6248 | 70.3 |
| pooled six | 733.4 | | 27772 | 37.9 | 37.5 | | |
| pbmc1k_v3 (extra) | 66.6 | 911 | 2556 | 38.4 | 37.8 | c003 | 53.7 |
| nuclei2k_v2 (extra, mouse) | 118.7 | 1556 | 4805 | 40.5 | 40.1 | c003 | 52.5 |

- The non-host index load is only about 1% of a sample's core-s (0.3-1.3 per M), so crediting it barely moves the CPU baseline. STAR, sort, pileup and CB refresh are about 85% (JSON stage windows).
- Everything after the pileup (non-host + export) is 224, 147, 192, 298, 245, 757 core-s (stage windows, 2 s resolution), 0.9-1.6x the section 12.9 W (175, 92, 121, 244, 271, 695) measured on a Xeon 6730P with another binary; the effect of #62 is not separable here from the CPU difference.
- GSM4006843, GSM5411066 and GSM7901264 were measured before `$W/data/bam-stream` was deleted. The same three could not be used in packing. pbmc1k_v3 and nuclei2k_v2 were added from `pipeline-benchmark/data` (job 789777).
- The 789532 array shows FAILED for tasks 0-5: the engine exited 0 and wrote everything; the last lines of my script returned 1 (fixed with an EXIT trap). Data complete, not rerun.

## Node packing
Measured once: **4 x pbmc1k_v3, `--threads 4`, one exclusive Xeon 6730P node (c100, 64 cores, 512 GB), job 789778**: wall 667-676 s each, core-s 2011-2060 (30.2-30.9 per M), RSS 53.7 GB,
0.394 M pairs/s per node (21.3 samples/hour of this 66.6 M sample), 12 cores busy on average of 64. There is no same-CPU single run to compare with (the single pbmc1k_v3 ran on a Gold 6248: 38.4 core-s/M), so **co-tenancy slowdown is not measured**; the
0.80 ratio mixes CPU model and contention.
- Why 4 threads: it is the campaign setting; a thread sweep was not run. A 4-thread sample uses about 3 cores averaged over its life (STAR 4, pileup 3-4, then a serial export), and RAM (about 54 GB RSS and 213 GB cgroup peak for 4) bounds concurrency, not cores.
- **Six-sample mix: not measured.** The pack jobs (789778 then 789779) waited hours for an idle node; 789779 was cancelled as instructed. Projected instead (labelled projected in the JSON): 8 concurrent samples (90% of free RAM / 54 GB), per-slot rate unchanged -> 0.79 M pairs/s, six-sample mix in about 930 s, **about 23 samples/hour**. Untested above 4 concurrent.
- Hang: 8 x GSM4006843 on c010 (768 GB, job 789621) stopped making progress in the BAM sort at about zero CPU. I cancelled it after 30 min; `Kill task failed`, and Slurm drained c010 (still `drain`; needs an admin undrain). Cause unknown. It is the only 8-concurrent attempt, and the 4-concurrent run on c100 was fine.
- My stall watchdog in `run_pack.sbatch` fired after hom4 had finished and cancelled the job (CANCELLED); results complete, makespan from per-instance stamps.

## Core-hours (allocated cores x elapsed): about 70
Singles 789532 (23.0) and 789777 (5.5); hung 8x run 789621 (25.3); hom4 789778 (16.6). Cancelled without running: 789622, 789623, 789693, 789758, 789759, 789779.
