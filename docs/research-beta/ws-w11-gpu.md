# W11 GPU engine: handoff note

Written by the W11 session; updated at every stop. Brief: [gpu-aligner-brief.md](gpu-aligner-brief.md). Design and every
measured number: [../gpu-engine/exploration.md](../gpu-engine/exploration.md) §12 (round 1: §12.5–12.11; round 2: §12.12–12.17).

## State at this stop (2026-10-28, round 2)

Branch `gpu/w11-prototypes`, stacked on `docs/beta-rollout-plan`. Draft PR
[Singlet-Bio/singlet#61](https://github.com/Singlet-Bio/singlet/pull/61): retarget it to `main` after PR #57 merges.
Research track only: nothing here touches the CPU engine or the Beta freeze. The CPU truth is `prod/campaign` c4a1be38.

| item | result | verdict | files |
|---|---|---|---|
| A. M5: STAR's mapping after the seeds, on the GPU | 1,991,926 / 1,991,926 distinct sequences byte-identical to STAR (its sequence-cache encoding: every transcript field, nW, nTr, best, unmapType); 2.40 M distinct reads/s per GPU (lanes A → M → B on the GPU, 0.27–0.37% on the host lane, same code) | **identity PASS; speed FAIL (4.2× short of 10 M)** | `bench/gpu/m5/`, `src/gpu/stardump/{ReadAlign_oneRead_dump.cpp,m5_dump.h}`, §12.12 |
| B. whole sample through the engine | pbmc1k_v3 (66.6 M pairs): 31.9 M distinct sequences mapped on the GPU in 17.1 s; engine fed through STAR's cache: **66,004,541 reads took the GPU's result, 0 STAR-verified mismatches**; every output equals a reference run up to the variation between two CPU-only runs | **PASS** | `bench/gpu/m5/inject/`, `results/e2e_*`, §12.14 |
| B. annotation + pileup on the GPU | 9 of 9 outputs identical to the engine on 3 samples; 40–63 ms per sample | **PASS** (Sonnet subagent) | `bench/gpu/pileup/`, `results/pileup*.json`, §12.13 |
| C. host: M13 non-host classify on the GPU | per-read results and all `nonhost/` files identical on 3 samples; index resident in HBM (8.9 s once vs 38–48 s per sample) | **PASS** (Sonnet subagent) | `bench/gpu/m13/`, `results/m13.json`, §12.16 |
| C. ranked bottlenecks | node GPU-bound by M5 at 8.7–11.2 M pairs/s; host bound 18 (today, incl. M5's host lane) / 24 (steps 1+2) | M5 is #1 | §12.15 |
| C. cross-sample reuse | 49.5% of pbmc1k_v3's reads (19% of distinct sequences) already occur in GSM7656149 (same chemistry); 0 across read lengths | a campaign-wide result store is exact and cuts M5 work | `results/m5_crosssample.json`, §12.16 |
| D. cost case | CPU baseline measured (c4a1be38 + PGO, 6 samples, 37.6 core-s per M pairs credited); a GPU node offsets **6.3–8.1 CPU nodes** (13 at today's host bound); cloud $ per sample 2.0–2.6× lower on GPU (1.0–1.3× if cloud CPU cores are 2× faster than Clipper's 2019 Xeons); ACCESS credits 2.5–3.1× fewer | labelled per number | `bench/gpu/cost/`, `results/cost_case.json`, §12.16 |

**Not done:** M5 at ≥ 10 M distinct reads/s; the M5 → pileup record glue (equivalence holds by composition); GPU read clipping
for 5′ samples; M7; nvCOMP ingest; demux (not needed until M5 is ~3× faster); CPU node-packing efficiency (single-sample
core-s used, labelled modelled); metered power.

## Overnight run (2026-10-28, 01:25–06:00): `singlet --engine gpu`

PR against `prod/campaign`: branch `gpu/engine-gpu` (see the PR link in the parent's report). Decisions made without asking, as
instructed:

- **Integration through STAR's sequence cache.** The GPU maps every distinct cDNA sequence (dedup on the GPU with a byte-level
  collision check, the M5 lane cascade, a host lane at STAR's limits) and fills STAR's cache before mapping; STAR and every other
  stage are unchanged. Outputs are the CPU engine's by construction, and `SINGLET_STAR_SEQCACHE_VERIFY=1` checks it in the run.
  Chosen over wiring the GPU pileup and non-host stages tonight: those need the M5 → record glue and engine refactoring that would
  not have been ready and verified by 05:00.
- **File-based decode with `--engine gpu`.** The default FIFO decode streams R2 to STAR, which cannot be read twice; the GPU path
  forces the engine's existing file decode. Effect on outputs: provenance only (5′-probe sample count, a decode line in
  `read_stats.tsv`).
- **Optional build:** `SINGLET_ENGINE_GPU=AUTO|ON|OFF`, CUDA runtime linked statically, sm_80/sm_90, `-fmad=false`.
- **Several samples per node:** an exclusive lock per GPU (named by PCI bus id), first free device or wait; any CUDA failure
  releases the device and falls back to CPU mapping for that sample.
- **Cache sizing** from the exact size of the GPU results (3×), replacing STAR's 4 GB default for `--engine gpu`.
- **Measured tonight:** identity PASS three times on pbmc1k_v3 (61.1 M, 64.8 M, 64.8 M reads STAR-verified, 0 mismatches); on a
  GPU node with 3 samples in flight, every data output identical to `--engine cpu` and **43% less CPU** (23.6 vs 41.1 core-s per M
  read pairs); GPU prefill for pbmc1k_v3 76 s (mapping 16 s). A single sample is not faster yet (468 vs 418 s): the prefill is
  serial before STAR. What binds the node is the CPU stages and per-sample memory (54–74 GB), not the GPU.
- **Ingest format** (Sonnet subagent): `docs/research-beta/ingest-format.md`, `.1fg` with nvCOMP-decodable zstd chunks, GPU
  reader 0.010–0.014 host core-s per M pairs (6–10× less than `.1fq`), decode identical on 188 M reads.
- **Profilers:** Nsight Compute could not collect counters on Clipper's GPU nodes; Nsight Systems is not in the CUDA 12.4 image.
  Evidence is in-kernel cycle counters and per-lane CUDA events.
- **Cluster incident (CPU-baseline subagent):** a hung 8-copy packing run left node c010 drained ("Kill task failed"); it needs an
  admin undrain.

## What changed in the design

- **Identity is solved for mapping.** M5 reproduces STAR's mapping state byte for byte through windows, stitching, extension,
  multimappers and filters; the truth is STAR's own sequence-cache encoding, and the same encoding is a deployable integration
  path (the GPU fills STAR's cache; the engine is unchanged).
- **The node is GPU-bound by M5**, not host-bound: the cost is window assignment and stitching for the 26% of sequences with more
  than 16 windows (multimappers). Lane A alone (all reads, ≤ 16 windows) runs at 6.4 M distinct/s.
- **Host steps 1 and 2 are shown exact** (M13), so the host bound is 24 M pairs/s per node with M5's host lane counted; host work is not the constraint now.
- **Cross-sample result reuse** is a real, exact lever for same-chemistry samples.
- **CPU-engine finding (outside W11):** the non-host EM depends on the order of its input reads, which multithreaded STAR leaves
  nondeterministic (iteration count 458 vs 462 between runs with identical inputs). Sorting the EM's reads would fix it.
- Aligner Brainstorming ideas used (re-implemented, cited in §12.12–12.13): pileup as sort + merge-join + segmented reduce with an
  arena; a separate heavy-read lane (here a lane cascade plus a per-read step budget). Not used: its seeding and scoring.

## Next steps, in order

1. **M5 speed:** warp-parallel windows in lanes M and B (two passes: each window's max finalized score is independent of
   recording, so a prefix max gives STAR's maxScoreMate per window); compact per-read state in lane A; overlap the lanes; a
   campaign-wide result store keyed by sequence (genome + parameters). Target ≥ 18–24 M pairs/s per node (the host bound), and a smaller host lane.
2. **M5 → pileup glue** with an order key for the multimapper ties; then one fused GPU pass per batch.
3. **GPU read clipping** (STAR's ClipMate) so 5′ samples can run.
4. CPU node packing (finish `bench/gpu/cost/run_pack.sbatch` on a free short-cpu node) and metered power; re-run `cost_model.py
   --pack-efficiency X`.

## Compute used

- Round 1: ~1 GPU-hour, ~75 core-hours. Round 2 (this session and its three Sonnet subagents): ~8 GPU-hours (M5 ~3.5,
  end to end ~2, pileup 1.1, M13 1.6) and ~250 core-hours (CPU baseline ~120 incl. packing attempts, pileup 31, M13 45, builds,
  end-to-end CPU runs on the GPU node). The CPU-baseline subagent's exact total is in `bench/gpu/cost/README.md`.

## How to resume (Clipper)

- Rules and paths: the shared notes file `w11-gpu-context.md`; the `clipper` skill. One shared bridge (never start it), never
  `sync.sh`, `--nice=10000`, no `short-bigmem`. short-cpu and debug-cpu were congested at this stop; GPU-node CPUs were used.
- Fetch `prod/campaign` by URL into the private ref `refs/w11/campaign` (fetching via the remote name tries to move the checked-out
  branch through the mirror refspec).
- The bam-stream staged samples (`$W/data/bam-stream/`) were deleted; use `$W/data/atlas-rerun/perf/<GSM>/sample.1fq` or
  `$W/pipeline-benchmark/data/<name>/<name>.1fq`.
- Binaries: `$W/jobs/w11/bin/{stardump,m5,m5m,m5w,singlet-m5inject}`. M5: `bench/gpu/m5/jobs/m5_compile.sbatch` (submit to
  short-gpu) then `m5_run.sbatch` (`M5RUNS` = variants, `MAXD` = distinct set size); best flags `--budget-a 1024 --budget-m 2048
  --budget-b 2048 --batch-a 262144 --batch-m 262144 --batch-b 65536`. Whole sample: `bench/gpu/m5/inject/inject_build.sbatch`,
  then `e2e.sbatch` (`SET`, `RUNS="ref ref2 verify gpu"`).
