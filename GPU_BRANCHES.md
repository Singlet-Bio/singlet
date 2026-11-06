# GPU branches and pull requests: one action each

This file maps every open GPU pull request and every `gpu/*` and GPU-related `perf/*` branch to one proposed action. It is a proposal. Nothing is deleted or closed by this file. Proposed deletion waits for a gate and for the owner.

**gpu/frontier** is the single branch for further GPU work. It starts at c2cfa889 on `perf/gpu-candidate-pr2`: `prod/campaign` 0a6de829, the staged Fable line E1 to E19 and E23, the strand-restart fix (#194), e20w (E20 lockclip, device CR4 clip), WSHRINK (off by default) and the kernel and run-time reference docs, with the gate result files in `bench/gpu_integ/results/`. The BSTREAM code is removed.

## What gpu/frontier does not hold yet

| Piece | Where it is | Needs |
|---|---|---|
| 512-base lift and the lean-path memory work (P3-MEM-03) | `perf/gpu-device` (#193), 8 commits ahead of gpu/frontier | rebase, ctest, the gates listed in its row below |
| VMM shared index | `gpu/bold-shidx-fix` | 3-slot MPS arm, identity at 2 and 3 slots, re-timing |
| Minimal transfer ledger (SINGLET_ENGINE_GPU_TRACE) | `perf/nvtx-host-ranges` | cherry-pick the ledger and NVTX commits only |
| k_resolve exon-multimapper fix | no source exists | write it |
| Drain-worker memory refit for ingest | partly in P3-MEM-03 of `perf/gpu-device` | refit the admission model on the ingest numbers |
| One device dedup | E12's and the one of #145 are both in the tree | keep one, delete the other |

Open blockers of gpu/frontier itself: the non-host GPU screen falls back to the CPU in many runs at 2 and 3 slots, in the b0 reference too (jobs 800983, 800984, 800985). The cause is not found. The corpus strand100 row is fixed on `bench/gpu-paths`. The four corpus inputs with reads over 128 bases fail by design until the 512-base lift is folded in.

## Pull requests

| PR | Branch | State | Proposed action | Why |
|---|---|---|---|---|
| #186 | `gpu/fable-engine` | open, draft, base prod/campaign | archive, then close | the study document `GPU_ALIGNER_STUDY.md` is its only content that perf/gpu-next lacks. Everything else is in gpu/frontier. Needs an owner decision on the study |
| #192 | `perf/gpu-next` | open, draft, base prod/campaign | folded, close when gpu/frontier merges into prod/campaign | gpu/frontier contains it. Its tip is an ancestor |
| #193 | `perf/gpu-device` | open, draft, base prod/campaign | fold next | the 512-base lift and memory work. Gates in its row below |
| #194 | `fix/gpu-strand-restart` | open, draft, base perf/gpu-next | folded, close when gpu/frontier merges | the fix is in gpu/frontier. The branch carries one extra result-file commit |
| #196 | `perf/gpu-candidate-pr2` | open, draft, base perf/gpu-next | folded (gpu/frontier starts from it), retarget or close when gpu/frontier has its own PR | the gates are recorded in its description |
| #180 | `gpu/fable/beta-notes` | open, base gpu/fable-engine | archive | notes only |
| #195 | `docs/gpu-handoff-2026-11-06` | open, base main | owner decision | handoff docs, one commit on main. Do not retarget |

## `gpu/*` branches (47)

| Branch | Tip | Last commit | Proposed action | PR | Archive tag | What it holds, or what it needs |
|---|---|---|---|---|---|---|
| `gpu/bold` | 857896c1 | 2026-11-05 | archive | - | yes | Prototype-switch identity job. Superseded by gpu/bold-integ. |
| `gpu/bold-admit` | 29815763 | 2026-11-05 | archive | - | yes | Admission control (SINGLET_GPU_ADMIT). A losing idea per the frontier plan. Its harness configs remain only in tp_integ.sbatch. |
| `gpu/bold-hostcut` | f04a64c7 | 2026-11-06 | archive | - | yes | Single-donor demux screen. Held: needs the tolerance check against truth (job 800850). |
| `gpu/bold-integ` | e80aeb65 | 2026-11-06 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/bold-kern` | bc6f6e53 | 2026-11-05 | archive | - | yes | MPS harness and job specs. MPS is an operations setting, not code. |
| `gpu/bold-kern2` | 2398b35f | 2026-11-05 | archive | - | yes | Campaign job scripts. The WSHRINK code of this branch is already in gpu/frontier. |
| `gpu/bold-shidx` | 398846ac | 2026-11-05 | archive | - | yes | Earlier VMM index attempt. Superseded by gpu/bold-shidx-fix. |
| `gpu/bold-shidx-fix` | 84be99f8 | 2026-11-06 | fold next | - | yes | VMM shared index (the holder shares driver virtual memory over SCM_RIGHTS). Identity PASS (job 800868). At 2 slots lock wait fell from 36.4 to 15.0 s, with fallback runs in both arms. Needs: the 3-slot MPS arm of job 800869 (stopped mid-arm), identity at 2 and 3 slots, re-timing at 3 and 4 slots with MPS, no fallback runs. |
| `gpu/bold-split` | 3c0ac3c2 | 2026-11-05 | archive | - | yes | Heavy-read split (STEPS_B auto). Held: the timing jobs 800769 to 800774 never produced a result. Revisit only if the timing lands. |
| `gpu/cr4-clip` | 697a679a | 2026-10-30 | folded | #98:MERGED->gpu/engine-gpu-order-free | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |
| `gpu/engine-gpu` | 9eda178b | 2026-10-29 | folded | #66:MERGED->prod/campaign | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |
| `gpu/engine-gpu-order-free` | d754edf8 | 2026-10-30 | archive | #95:MERGED->prod/campaign | yes | #95, merged through #98. Its one extra commit is a merge commit. |
| `gpu/engine-gpu-stream` | 9ca1ea60 | 2026-10-29 | folded | #69:CLOSED/draft->gpu/engine-gpu | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |
| `gpu/fable-engine` | feee2453 | 2026-11-06 | archive | #186:OPEN/draft->prod/campaign | yes | #186. The tip adds the 1,444-line study document GPU_ALIGNER_STUDY.md that perf/gpu-next lacks. Keep the tag. The document needs an owner decision before it is made public. |
| `gpu/fable/beta-notes` | 6974eed8 | 2026-11-01 | archive | #180:OPEN->gpu/fable-engine | yes | #180. One notes document, summarized in the train notes. |
| `gpu/fable/budget` | 503c07f7 | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/census` | 51eeb2c1 | 2026-10-31 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/compact` | ac61b3e5 | 2026-10-31 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/e2e` | dcb1a2b6 | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/hostloop` | e973e6e0 | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/ingest` | fe97c8b4 | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/lockclip` | 03f8bb20 | 2026-11-02 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/mtpileup` | f73938b2 | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/order` | 83f942ce | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/pool` | 719fd650 | 2026-11-01 | archive | - | yes | E21 resident pool (singlet gpu-pool). Held: hgmm1k_v3 in a mixed pool dies with STAR exit 102, no report, never run together with E20. |
| `gpu/fable/posterior` | 44a6e9fa | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/prefill` | d03d6c47 | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/probe` | e2b0000e | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/profile` | 26f0525f | 2026-10-31 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/records` | 5223bec4 | 2026-10-31 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/release` | dc45ef09 | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/seeds` | cce8514e | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/solo` | 5aab9e37 | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/solowire` | 8e54be40 | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/tail` | e0e1514a | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/throughput` | 283105e6 | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fable/warp-window` | d6230c5e | 2026-11-01 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `gpu/fg2-engine` | 8b1c3d19 | 2026-10-31 | archive | #113:CLOSED/draft->prod/campaign | yes | #113, closed. Its content is in prod/campaign (93dfa015) apart from lock-timing logging that E20 replaced. |
| `gpu/fg2-table-prefill` | afe3efb3 | 2026-10-31 | folded | #126:MERGED/draft->prod/campaign | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |
| `gpu/gpu-lock-release` | b216cf6d | 2026-10-31 | folded | #111:MERGED/draft->prod/campaign | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |
| `gpu/gpu-pileup-preflight` | 6f469719 | 2026-10-31 | folded | #120:MERGED/draft->prod/campaign | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |
| `gpu/gpu-pileup-quality` | 62e89ca9 | 2026-10-30 | folded | #109:MERGED/draft->prod/campaign | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |
| `gpu/gpu-slots` | 7cd8f289 | 2026-10-31 | archive | - | yes | Queue benchmark with MPS option. Operations tooling, superseded by the bold harness. |
| `gpu/gpu-slots2` | cf92cdb9 | 2026-10-31 | folded | #125:MERGED/draft->prod/campaign | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |
| `gpu/host-offload` | cbb07c3c | 2026-10-30 | folded | #107:MERGED/draft->prod/campaign | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |
| `gpu/intron-rules` | d70348c1 | 2026-10-30 | folded | - | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |
| `gpu/w11-prototypes` | 5836df84 | 2026-10-29 | folded | #61:MERGED->main | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |

## `perf/*` branches (17)

GPU-related branches first. Branches marked "not GPU work" are in the list only to show they need nothing from this train.

| Branch | Tip | Last commit | Proposed action | PR | Archive tag | What it holds, or what it needs |
|---|---|---|---|---|---|---|
| `perf/gpu-candidate-pr2` | 5243c98b | 2026-11-06 | folded | #196:OPEN/draft->perf/gpu-next | no | The PR-2 branch. gpu/frontier starts from c2cfa889 on it. |
| `perf/gpu-device` | ea736dc1 | 2026-11-06 | fold next | #193:OPEN/draft->prod/campaign | yes | #193. Holds the 512-base lift (5b8afcbd, variable-width masks), the lean-path memory work P3-MEM-03 steps A and B (the ingest memory refit), and build and gate jobs. Needs: rebase onto gpu/frontier, ctest, the 10-sample gate with unchanged outputs for reads up to 128 bases, corpus 16 of 16, a 150-base 5' sample, the GSM7064234 slice and then the full sample under the 370 GB cgroup, device memory budgets re-measured (masks up to 8 words per read, 16-bit index arrays near 512 bases). |
| `perf/gpu-instrumentation` | ddb5ff33 | 2026-10-31 | folded | #144:MERGED/draft->prod/campaign | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |
| `perf/gpu-next` | f15eebb6 | 2026-11-04 | folded | #192:OPEN/draft->prod/campaign | yes | Tip is an ancestor of gpu/frontier. |
| `perf/gpu-next-e20` | aeef7a5e | 2026-11-04 | folded | - | yes | Tip is an ancestor of gpu/frontier. |
| `perf/gpu-next-e20e21` | bcc551c9 | 2026-11-04 | archive | - | yes | E20 plus E21 binary. Never gated. Held with E21. |
| `perf/gpu-next-e21` | eb9ebf9f | 2026-11-04 | archive | - | yes | E21 merged with gpu-next. Held with gpu/fable/pool. |
| `perf/gpu-pinned-dedup` | 3a064a19 | 2026-10-31 | folded | #145:CLOSED/draft->perf/gpu-instrumentation | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |
| `perf/nvtx-host-ranges` | cacf6435 | 2026-10-31 | fold next | - | yes | Only the transfer ledger (SINGLET_ENGINE_GPU_TRACE=1, a210148e) and the NVTX ranges are wanted, as a minimal ledger of bytes and seconds per phase. Needs: cherry-pick those commits without the experiment and revert commits, then a gate. |
| `perf/pileup-lock-release` | 5daaa7a2 | 2026-10-31 | folded | #136:MERGED/draft->prod/campaign | not needed, reachable from prod/campaign | Already in prod/campaign, so in gpu/frontier. |

Not GPU work (merged into prod/campaign or closed, no action from this train): `perf/bam-stream`, `perf/deep-samples`, `perf/drain-memory-admission`, `perf/fq-read-source`, `perf/star-accel`, `perf/txome-first`, `perf/w3-speedups`.

## Counts

Folded: 45. Fold next: 3. Archive: 16. Delete: 0 (none proposed yet; folded branches with an archive tag or full reachability are safe to delete after the gold gate passes).

The tags `archive/gpu-2026-11-06/<branch>` pin 71 branches. The branches without a tag are merged into prod/campaign (reachable from it) or are `perf/gpu-candidate-pr2`.
