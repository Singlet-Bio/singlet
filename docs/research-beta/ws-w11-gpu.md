# W11 GPU engine: handoff note

Written by the W11 session; updated at every stop. Brief: [gpu-aligner-brief.md](gpu-aligner-brief.md). Design and every
measured number: [../gpu-engine/exploration.md](../gpu-engine/exploration.md) §12 (§12.5–12.12 are this session's).

## State at this stop (2026-10-27)

Branch `gpu/w11-prototypes`, stacked on `docs/beta-rollout-plan`. Draft PR
[Singlet-Bio/singlet#61](https://github.com/Singlet-Bio/singlet/pull/61): retarget it to `main` after PR #57 merges.
Research track only: nothing here touches the CPU engine or the Beta freeze.

| step | result | verdict | files |
|---|---|---|---|
| 1. host bottleneck | timed dev build on 6 samples: W = 2.18 core-s per M pairs pooled; demux auto-K + non-host screen = 74% | node ≥ 20 M pairs/s **met** with a persistent non-host index (worst sample 20.7) | `bench/gpu/host/`, `results/m10b_*.json`, §12.9 |
| 2. STAR dump tool | vendored STAR with a dumping `maxMappableLength2strands` | built and used | `src/gpu/stardump/`, `jobs/stardump_build.sbatch` |
| 3. M3 exact seed search | 323,688,051 / 323,688,051 seed records identical to STAR (3 samples, sparse and dense SA); ~1.0 G records/s | **PASS** | `bench/gpu/m3m4/`, `results/m3m4_*.json`, §12.6 |
| 4. M4 certificates | 0 wrong, but 16–23% of calls resolved | exact, **dropped** (not worth it) | same, §12.7 |
| 5. M6 dedup + count + UMI | 1.6–2.0 G pairs/s; matrices identical to the CPU engine; identical across batch sizes and runs | **PASS** | `bench/gpu/m6/`, `results/m6*.json`, §12.8 |
| 6. M8 ingest | 12–15 M pairs/s per decode thread; 0.085 core-s/M (0.175 with CRC); PCIe 12.3 GB/s | fails "≤ 3 cores at 50 M/s", irrelevant at the rate the host allows | `bench/gpu/m8/`, `results/m8.json`, §12.10 |
| 7. M5 stitch port | not started | — | — |

**Done criteria of this session:** M3, M4, M6 and M8 measured; the host-side plan has a measured path to ≥ 20 M pairs/s per
GPU node (§12.9, step 1). Met. The GPU side of that 20 M is measured for every stage except M5 (§12.11).

## What changed in the design

- **The campaign's GRCh38 STAR index is sparse (`--genomeSAsparseD 2`):** 12.8 GB SA, two SA searches per human seed call.
  Mouse GRCm39 is dense. Never rebuild it dense: it would change STAR's results.
- **Seeding is not the bottleneck.** The exact STAR port runs ~1.0 G seed records/s (78–105 M distinct reads/s); certificates,
  LCP reuse, SBK and the locator are de-prioritised.
- **The node is host-bound** at 27–36 M pairs/s (two GPUs, 55 host cores) once the non-host index is loaded once per node
  (step 1) and non-host classify moves to the GPU (step 2). Each GPU then needs only 14–18 M pairs/s, which makes **M5's
  requirement ≥ 10 M distinct reads/s per GPU**.
- **Demux** is the largest host item; its fits run serially (`threads = 1` inside export), so a bit-exact GPU port is possible
  (one thread per accumulator in the CPU's cell order) if the host bound has to rise further (step 3: 73–83 M pairs/s per node).
- **The atlas `.1fq` files carry no base qualities:** the 53 B/pair payload becomes 32 B/pair.

## Next steps, in order

1. **M5 (stitch port), the equivalence risk.** Extend `stardump` to also dump windows, aligns and final transcripts (or run it
   with SAM output on the same distinct-read set) as truth; that gives A3 (PC/WC/WA sizes, overflow share) for free. Port
   `stitchPieces` → windows → `stitchWindowAligns` → `extendAlign` → scoring → `multMapSelect` / filters, feeding it from the
   M3 kernel's answers. Gate: 100% identical transcripts; ≥ 10 M distinct reads/s per GPU.
2. **Link M3 → M5 → per-sequence annotation → M6** into one GPU pass per batch (stage a and stages b–d of M6 are not linked yet).
3. **M13** non-host on the GPU (host step 2) and **M7** barcode correction (bit-exact).
4. Host: persistent service with the non-host index loaded once; measure memory per in-flight export (~45 per node).

## Compute used this session

- GPU: ~1 GPU-hour (M3/M4 ~0.4 incl. tests, M6 ~0.2, M8 ~0.3, diagnostics).
- CPU: ~75 core-hours (host timers ~50, of which a hung replay wasted ~24; M6 engine dumps ~16; builds).

## How to resume (Clipper)

- Rules and paths: the shared notes file `w11-gpu-context.md` (session notes folder of the parent session); the `clipper`
  skill. One shared bridge (never start it), never `sync.sh`, `--nice=10000`, no `short-bigmem`.
- Fetch `prod/campaign` into the bare repo only as a private ref (`refs/w11/campaign`); never move the checked-out branch.
- Binaries on Clipper: `$W/jobs/w11/bin/{stardump,seedsearch}`, `$W/jobs/w11/host/bin/singlet-timers`,
  `$W/jobs/w11/m6/bin/singlet-m6dump`, M8 under `$W/jobs/w11/m8/`.
- M3/M4: `bench/gpu/jobs/m3m4_compile.sbatch` then `m3m4_run.sbatch` (array over 3 samples; reads and dumps stay in job /tmp;
  ~8 min and ~0.15 GPU-h per sample). Upload sources with `clipper.py put` from inside the folder (absolute remote paths).
- Host timers: `bench/gpu/host/{build_timers,run_timers,replay_export}.sbatch`. M6: `bench/gpu/m6/jobs/`. M8: `bench/gpu/m8/jobs/`.
