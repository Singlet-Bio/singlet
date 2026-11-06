# Co-located fetch test

Question: can a GPU job fetch and decode its own reads on the GPU node, using cores the engine leaves idle, without slowing the drain?
Background: the staging deck (2026-11-06) and `docs/research-beta/download-and-staging.md` §8. In batch 1 (k=8, T=7, 2 slots, one H100,
28 cores, 370 GB) the drain used 17.5–18.9 cores, waited on the device lock 31–39% of the time and peaked at 346–356 GB host memory.
One GPU drains about 1.8–2.0 M pairs/s; the fetch tier costs about 3.0 core-s per M pairs, so feeding one GPU takes about 6 cores.

## Design

`jobs/colo.sbatch` is `bench/gpu_batch1/jobs/tp.sbatch` with one addition: a point tag may carry `c<D>`, which starts `colo_fetch.py` with
D fetch slots inside the same job (same cgroup, same 28 cores) for the duration of the point. A tag may end in `_<name>` so a point can repeat.

Each fetch slot loops: ranged GET of a full `.sra` from public S3 into node `/tmp` (8 threads, `pwrite`, per-slot cap `COLO_SLOT_MBPS`,
default 100 MB/s), then `nice -n 19 singlet download <sra> --quality binned` (S9's `singlet-cpu`), then delete. Runs come from S9's
`runs.json`, 1.5–6 GB, in a fixed shuffled order. Every point waits `COLO_WARMUP_S` (60 s) before its drain starts, with or without a fetch.

Default campaign (A/B/A/B on one node in one job):

```
POINTS="k8t7s2_a k8t7s2c2 k8t7s2_b k8t7s2c4" NRUNS=24 SAMPLE=pbmc1k_v3 BINDIR=$W/jobs/gpu-next/bin/s3 EXPECT_COMMIT=e42c0636
```

## Reading the result

`colo_summary.py` writes `summary.md` / `summary.json` in the results directory. Per fetch point: `drain_vs_base_pct` (drain rate
against the mean of the points without a fetch), `fetch_covers_drain` (pairs decoded per pair drained, inside the drain span; 1 or more
means one GPU job could feed itself), engine-only core-s per M pairs, the fetch tier's cores, network rate, core-s per M pairs and peak RSS,
plus the harness's fallbacks and identity. Co-location is viable when the drain loses no more than the noise (a few percent within a job)
while `fetch_covers_drain` reaches 1, with no fallbacks, no identity change and memory inside the job.

Not covered: two GPUs of a node at once, the 0.6 GB/s global cap shared with a real fetch tier, `.1fg2` writing (`--stage-out`),
and the drain reading from node NVMe instead of the shared stage (the harness already copies the input to `/tmp`).

## Files

`colo_fetch.py` (fetch slots), `colo_summary.py` (table), `jobs/colo.sbatch` (GPU job), `jobs/smoke.sbatch` (CPU-only check of both
scripts with a fake drain point). On Clipper the scripts live in `$W/jobs/colo/scripts`, results in `$W/jobs/colo/results/`.
