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

## Fold order and written TODOs

Each item enters gpu/frontier as its own commit or pull request with its gate.

1. The non-host GPU screen falls back to the CPU in many runs at 2 and 3 slots, the b0 reference included (jobs 800983 to 800985). Find the cause first, because the fallback contaminates every timing run. Diagnostic: job 802058.
2. `perf/gpu-device` (#193): the 512-base lift and the P3-MEM-03 memory work, with the gates in its row. This makes corpus 16 of 16 possible.
3. The minimal transfer ledger and NVTX ranges from `perf/nvtx-host-ranges`, by cherry-pick only.
4. The VMM shared index from `gpu/bold-shidx-fix`, with the 3-slot MPS arm and identity at 2 and 3 slots.
5. Keep one device dedup and delete the other. Remove any code the deleted branches leave dead.

**TODO, not started: the k_resolve exon-multimapper fix.** No source exists for it. It blocks the merge into prod/campaign. The resolver work owns it.

## Deletion log

Decision D-93 (owner, 2026-11-06) closes branches now. Each branch below was deleted from `origin` after its tip was verified: an ancestor of `origin/gpu/frontier`, `prod/campaign` or `main`, or pinned by the tag `archive/gpu-2026-11-06/<branch>` at the same tip. To restore a branch: `git branch <name> <sha>`, then `git push origin <name>`.

Kept: `gpu/frontier`, the three fold-next branches (`perf/gpu-device`, `gpu/bold-shidx-fix`, `perf/nvtx-host-ranges`), and every `prod/*`, `main` and `bench/*` branch. The non-GPU `perf/*` branches are left to the non-GPU consolidation.

Pull requests closed with a comment pointing to #198: #180, #186, #192, #194, #196. Open: #193 (fold next), #195 (docs into main), #198 (the single GPU pull request).

| Branch | Tip (restore from this SHA) | Why it was safe |
|---|---|---|
| `fix/gpu-strand-restart` | `91be1941a5da3e980027034b50b68c4e662d66f5` | reachable from gpu/frontier |
| `gpu/bold` | `857896c1ded331a5892b34e1877e2825b03809e7` | tag archive/gpu-2026-11-06/gpu/bold at the same tip |
| `gpu/bold-admit` | `29815763472bd2a8a23fcf0d656cde408019cc50` | tag archive/gpu-2026-11-06/gpu/bold-admit at the same tip |
| `gpu/bold-hostcut` | `f04a64c7b066c982400dbaca39589694b7255b4e` | tag archive/gpu-2026-11-06/gpu/bold-hostcut at the same tip |
| `gpu/bold-integ` | `e80aeb6580b31fbf42ad8749f71297b319885a19` | reachable from gpu/frontier |
| `gpu/bold-kern` | `bc6f6e532f0f4f0e59a618f0488279e948208e03` | tag archive/gpu-2026-11-06/gpu/bold-kern at the same tip |
| `gpu/bold-kern2` | `2398b35f6791e7357de743e3486c0f91900891af` | tag archive/gpu-2026-11-06/gpu/bold-kern2 at the same tip |
| `gpu/bold-shidx` | `398846ac2d2ad5ec0d67d2264a7da154ec2555c2` | tag archive/gpu-2026-11-06/gpu/bold-shidx at the same tip |
| `gpu/bold-split` | `3c0ac3c25cb2cdd174537c20ac5f16caf927428d` | tag archive/gpu-2026-11-06/gpu/bold-split at the same tip |
| `gpu/cr4-clip` | `697a679ab2558ae077a76d0f00b58454ca9a6a31` | reachable from gpu/frontier |
| `gpu/engine-gpu` | `9eda178beb198afde9070f877de98f844e135494` | reachable from gpu/frontier |
| `gpu/engine-gpu-order-free` | `d754edf87a37086a7b5666b266dc34b2a4924fd6` | tag archive/gpu-2026-11-06/gpu/engine-gpu-order-free at the same tip |
| `gpu/engine-gpu-stream` | `9ca1ea6027ac9392645a8c31bed3a130173e7eb6` | reachable from gpu/frontier |
| `gpu/fable-engine` | `feee2453dd94486aafa3b328f5f180fa2ff8cc7f` | tag archive/gpu-2026-11-06/gpu/fable-engine at the same tip |
| `gpu/fable/beta-notes` | `6974eed874dc5867d5b91d5e2445b0deeaef9029` | tag archive/gpu-2026-11-06/gpu/fable/beta-notes at the same tip |
| `gpu/fable/budget` | `503c07f7a3aaeeeeaebec731a9235b51141cbf0c` | reachable from gpu/frontier |
| `gpu/fable/census` | `51eeb2c1b7fda12aa5d8661756de94944f7b12a3` | reachable from gpu/frontier |
| `gpu/fable/compact` | `ac61b3e56bb830bc197e34188cc63c2c88f1659a` | reachable from gpu/frontier |
| `gpu/fable/e2e` | `dcb1a2b6717f3f655cc5acfa7f33ac9e3b039180` | reachable from gpu/frontier |
| `gpu/fable/hostloop` | `e973e6e0604e87f914a6f09b0e1514cd300b7be8` | reachable from gpu/frontier |
| `gpu/fable/ingest` | `fe97c8b4ebf2c9882a0b31778c271b9299c05de2` | reachable from gpu/frontier |
| `gpu/fable/lockclip` | `03f8bb20f65a36b3c3a7a03642a343535c36b65e` | reachable from gpu/frontier |
| `gpu/fable/mtpileup` | `f73938b2013580b8dcb6d57413a01fe532d8d732` | reachable from gpu/frontier |
| `gpu/fable/order` | `83f942ce075f83cbac219da174b607a569f1d60e` | reachable from gpu/frontier |
| `gpu/fable/pool` | `719fd650443b04dfdd495c10d3aad8df95162b9a` | tag archive/gpu-2026-11-06/gpu/fable/pool at the same tip |
| `gpu/fable/posterior` | `44a6e9fa19d3188550ff2f619b6e3bb1264cb788` | reachable from gpu/frontier |
| `gpu/fable/prefill` | `d03d6c47d34afa8d291ece7e3cfd156d45ddd973` | reachable from gpu/frontier |
| `gpu/fable/probe` | `e2b0000efe9dec67373bc9177870c38b224f3895` | reachable from gpu/frontier |
| `gpu/fable/profile` | `26f0525f936a0c9eb5c9f46bba7d9b821d1f1b4e` | reachable from gpu/frontier |
| `gpu/fable/records` | `5223bec4f91692b9ab730205a339a0dd842d97e4` | reachable from gpu/frontier |
| `gpu/fable/release` | `dc45ef099ab73c943e06000fde088ffc36764e15` | reachable from gpu/frontier |
| `gpu/fable/seeds` | `cce8514e65b89db4630dd7bd28443d815a824ea5` | reachable from gpu/frontier |
| `gpu/fable/solo` | `5aab9e379da81af23ab1bf71be8a4467325369a9` | reachable from gpu/frontier |
| `gpu/fable/solowire` | `8e54be4013631c69df9663a558c60aa1fef07070` | reachable from gpu/frontier |
| `gpu/fable/tail` | `e0e1514a30acc5cd53cb014a76ae394095384804` | reachable from gpu/frontier |
| `gpu/fable/throughput` | `283105e6cd28e8e3c603ceaea029811866555393` | reachable from gpu/frontier |
| `gpu/fable/warp-window` | `d6230c5e295dd621a97133bf144f66b593945a7a` | reachable from gpu/frontier |
| `gpu/fg2-engine` | `8b1c3d195af64315b13559000c1e1c51e8880a23` | tag archive/gpu-2026-11-06/gpu/fg2-engine at the same tip |
| `gpu/fg2-table-prefill` | `afe3efb3fb01a5cd86355e78522041f636a74633` | reachable from gpu/frontier |
| `gpu/gpu-lock-release` | `b216cf6df76c8c3906ae7a2d860e9d82aa5c67bb` | reachable from gpu/frontier |
| `gpu/gpu-pileup-preflight` | `6f469719eece41a54e01d32d90dd113ae852a96e` | reachable from gpu/frontier |
| `gpu/gpu-pileup-quality` | `62e89ca91013d4efabb7531a924439ee3c13c893` | reachable from gpu/frontier |
| `gpu/gpu-slots` | `7cd8f289676de0597112b486c7862dcfd0565d9d` | tag archive/gpu-2026-11-06/gpu/gpu-slots at the same tip |
| `gpu/gpu-slots2` | `cf92cdb98796a56c12b351248e7cc1a2bd8d8692` | reachable from gpu/frontier |
| `gpu/host-offload` | `cbb07c3c9f3e224ab0b1728a26a63b63816a175f` | reachable from gpu/frontier |
| `gpu/intron-rules` | `d70348c15ba06e06141bfdad920d59b2aa47b118` | reachable from gpu/frontier |
| `gpu/w11-prototypes` | `5836df84dc5985cdc4b72a804713df23449fc600` | reachable from gpu/frontier |
| `perf/gpu-candidate-pr2` | `ee421b06bee72f37236483a28423ed09637154f1` | reachable from gpu/frontier |
| `perf/gpu-instrumentation` | `ddb5ff334a9009e97431894c8bc245ee798a0477` | reachable from gpu/frontier |
| `perf/gpu-next` | `f15eebb63fa2804083a7acd258df4d73a29b3552` | reachable from gpu/frontier |
| `perf/gpu-next-e20` | `aeef7a5e48fd11fba63c27fcf19e97e8bfb3a2f5` | reachable from gpu/frontier |
| `perf/gpu-next-e20e21` | `bcc551c94dfcad281bec0e8c82f9150221d9f75d` | tag archive/gpu-2026-11-06/perf/gpu-next-e20e21 at the same tip |
| `perf/gpu-next-e21` | `eb9ebf9fac13fe14c38aacfc6b14d5df9cd27bea` | tag archive/gpu-2026-11-06/perf/gpu-next-e21 at the same tip |
| `perf/gpu-pinned-dedup` | `3a064a195bae0a9b14f5b2d6769f648b89d33fb2` | reachable from gpu/frontier |
| `perf/pileup-lock-release` | `5daaa7a2a172c2175f2e8246dabd9fef07fc2dc8` | reachable from gpu/frontier |

Total: 55 branches deleted.
