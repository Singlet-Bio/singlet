# W11 GPU engine: handoff note

Written by the W11 session. Updated at every stop. Brief: [gpu-aligner-brief.md](gpu-aligner-brief.md). Design and
results: [../gpu-engine/exploration.md](../gpu-engine/exploration.md) (§12 has every measured number).

## State (2026-10-27, in progress)

Branch `gpu/w11-prototypes` (based on `docs/beta-rollout-plan`, PR #57). Draft PR: _pending_.

| step | status | result file |
|---|---|---|
| 1. host bottleneck (timers in a dev build) | running | `bench/gpu/results/m10b_timers.json` |
| 2. STAR dump tool | built (`src/gpu/stardump/`) | — |
| 3. M3 exact SA seed search on GPU | running | `bench/gpu/results/m3m4_<sample>.json` |
| 4. M4 unique-prefix certificates | running | same |
| 5. M6 dedup, count sort, UMI collapse | running | `bench/gpu/results/m6.json` |
| 6. M8 host↔device pipeline | running | `bench/gpu/results/m8.json` |
| 7. M5 stitch port | not started | — |

## Findings so far

- **The campaign's GRCh38 STAR index is sparse (`--genomeSAsparseD 2`)**: SA 12.8 GB, 3.10 G suffixes (every other
  position). Mouse GRCm39 is dense. So every human seed call runs two SA searches (iDist 0 and 1), and exploration §2.2's
  `gSAsparseD = 1` is wrong for human. Consequences in §12.
- Demux sums floats in per-thread partials (`donor_demux.h`, `s1_tl`, `e_part`), so its output depends on `--threads`.
  Any moved or parallel demux must pin the same chunking.

## How to resume

- Clipper rules and paths: `w11-gpu-context.md` in the session notes folder (copied below when final).
- Build `stardump`: `bench/gpu/jobs/stardump_build.sbatch` (scratch copy of prod/campaign, private ref `refs/w11/campaign`).
- M3/M4: `bench/gpu/jobs/m3m4_compile.sbatch`, then `bench/gpu/jobs/m3m4_run.sbatch` (array over 3 samples).
