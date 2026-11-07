# GPU pilot

`pilot.sbatch` holds one whole H100 node on Clipper and runs job scripts from a queue directory. Timing runs then share one fixed node,
and nothing waits in the Slurm queue between runs. Clipper caps each user at three H100s at once, and an exclusive job counts as two
even when it uses one GPU. One pilot running two items side by side uses both GPUs it holds and leaves the third slot free for
`debug-gpu` checks.

## Queue

The queue root defaults to `$WORK/jobs/gpu-campaign/queue` (set `QUEUE` to change it).

- `new/` holds the items to run. Each item is a bash script named `<order>-<name>.g1.sh` (one GPU) or `<order>-<name>.g2.sh`
  (the whole node). Items start in filename order.
- `running/` holds the items in progress, and `done/` holds finished items with one log each (`<order>-<name>.log`).
- `pilot.tsv` gets one row per finished item: item, pilot job, node, GPU, start, end, wall seconds and exit code.
- `pilot-<job>.log` records the pilot's own decisions.

A g1 item takes the first free GPU. A g2 item waits until both GPUs are free, and no g1 item behind it starts first. Each item runs
as its own `srun` step, so its CPU and memory accounting is its own cgroup: half the cores and half the memory for a g1 item, all of
them for a g2 item. The pilot sets these variables:

| Variable | Value |
|---|---|
| `PILOT`, `PILOT_JOB`, `PILOT_NODE` | 1, the pilot's Slurm job, its node |
| `PILOT_GPU` | 0, 1 or `both` |
| `PILOT_SHARED` | 1 for a g1 item, since the other GPU may be busy |
| `RUNID` | `<job>-g<gpu>-<name>`, for result and scratch paths |
| `PINDEV`, `CUDA_VISIBLE_DEVICES` | the GPU of a g1 item |

An item may declare `# PILOT-MINUTES: <n>` (default 240). The pilot skips an item when less time than that is left in its allocation.

## Running it

```bash
sbatch bench/gpu/pilot/pilot.sbatch            # one whole H100 node for up to 24 h
bench/gpu/pilot/enqueue.sh 10-fallback.g1.sh   # copy an item into new/
touch $WORK/jobs/gpu-campaign/queue/STOP       # end after the running items finish
```

The pilot exits by itself after 30 minutes with nothing running and nothing queued (`IDLE_S`). Resubmit it when there is new work.

## Timing on a shared node

Two g1 items on one node share the host: memory bandwidth, the file system and the PCIe root. For an A/B comparison, run the legs at the
same time on both GPUs with the arms crossed, for example `base X X base` on GPU 0 and `X base base X` on GPU 1. Device and co-tenant
effects then fall equally on both arms. Arms that act on the whole node (the shared-index holder, CUDA MPS, memory-aware admission)
must run as g2 items; the throughput harness refuses them in a g1 item. Every result line records the pilot job, node and GPU.
