# Branches of Singlet-Bio/singlet, 2026-11-06

This file records the branch consolidation of 2026-11-06 for every branch outside the GPU line. Each remote branch got one action. A branch was deleted when its tip was reachable from `prod/campaign`, `main` or `gpu/frontier`, or after its tip was tagged `archive/2026-11-06/<branch>`. The GPU branches are listed in `GPU_BRANCHES.md` on `gpu/frontier`.

Remote branches before: 273. Deleted here: 122. Left to the GPU line: 60. Kept here: 91. Remote branches after this pass: 151 (fewer once the GPU deletions land).

Restore any deleted branch from its SHA with `git push origin <sha>:refs/heads/<branch>`, or from its tag with `git push origin archive/2026-11-06/<branch>:refs/heads/<branch>`.

## Kept

| Branch | Tip | Last commit | Open PR | Reachable from | Reason |
|---|---|---|---|---|---|
| `bench/cellcall-independent-truth` | 90c172e4 | 2026-10-26 | - | no | checked out in a local worktree |
| `bench/colo-fetch` | 649b8567 | 2026-11-06 | - | no | checked out in a local worktree |
| `bench/demux-truth` | 163e70a7 | 2026-11-05 | - | no | checked out in a local worktree |
| `bench/demux-truth2` | fe78ec01 | 2026-11-06 | - | no | checked out in a local worktree |
| `bench/exact-vs-pseudo` | 8460ed6f | 2026-10-26 | - | no | checked out in a local worktree |
| `bench/gpu-batch1` | 0fdec57f | 2026-11-05 | - | no | pinned pending jobs |
| `bench/gpu-fast` | 317c142d | 2026-11-05 | - | no | pinned pending jobs |
| `bench/gpu-paths` | f92d8136 | 2026-11-06 | - | no | checked out in a local worktree |
| `bench/hto-independent` | 53fd8116 | 2026-10-31 | - | no | checked out in a local worktree |
| `bench/nonhost-transcripts` | e53636db | 2026-10-31 | - | no | checked out in a local worktree |
| `bench/public-ambient-doublets` | 178cea48 | 2026-10-26 | - | no | checked out in a local worktree |
| `bench/public-counts` | 772eb0cd | 2026-10-25 | - | no | checked out in a local worktree |
| `bench/public-genotype` | 4eb87a24 | 2026-10-26 | - | no | checked out in a local worktree |
| `bench/public-perf` | 357a7989 | 2026-11-01 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `bench/read-collapse` | 4e496302 | 2026-11-05 | - | no | checked out in a local worktree |
| `bench/scbasecount` | b891bd71 | 2026-10-24 | - | prod/campaign, main, gpu/frontier | checked out in a local worktree |
| `bench/tx-first` | 85a7c4f0 | 2026-11-05 | - | no | checked out in a local worktree |
| `bench/vdj-heldout` | 90b93489 | 2026-10-31 | - | no | checked out in a local worktree |
| `bench/vdj-sota` | 3108b7f8 | 2026-11-01 | #143 (upd 2026-11-01) | no | checked out in a local worktree |
| `ci/macos-15` | 8eea308e | 2026-10-30 | - | main | checked out in a local worktree |
| `ci/python-matrix-runner` | c2f69787 | 2026-11-02 | #188 (upd 2026-11-02) | no | checked out in a local worktree |
| `claude/alpha-bench` | e003e124 | 2026-10-26 | - | prod/campaign, main, gpu/frontier | checked out in a local worktree |
| `claude/alpha-repack` | 7c7916b7 | 2026-10-26 | - | no | checked out in a local worktree |
| `claude/atac-orderfree-a1-a2` | 595d4e6a | 2026-10-29 | - | no | checked out in a local worktree |
| `claude/beta-pilot` | a97ed392 | 2026-11-06 | #191 (upd 2026-11-06) | no | open PR #191 updated in the last 7 days |
| `claude/cell-line-types` | 15d5d0ef | 2026-10-29 | - | no | checked out in a local worktree |
| `claude/cell-type-capability` | 0bf6dfe9 | 2026-10-25 | - | prod/campaign, main, gpu/frontier | checked out in a local worktree |
| `claude/celltype-nn` | 999e1954 | 2026-10-29 | - | prod/campaign, main, gpu/frontier | checked out in a local worktree |
| `claude/clients-retired-legacy` | dac25ef1 | 2026-10-27 | - | prod/campaign, main, gpu/frontier | checked out in a local worktree |
| `claude/docs-ancestry-call-legacy` | 60c0e202 | 2026-11-01 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `claude/expression-sota-31cf6d` | 7d7c139c | 2026-10-31 | - | no | checked out in a local worktree |
| `claude/genotype-sota-a9b44b` | d3d4aa7b | 2026-11-01 | #182 (upd 2026-11-01) | no | open PR #182 updated in the last 7 days |
| `claude/gpu-atac-reducers-270cf8` | cabdbfd5 | 2026-10-29 | - | no | checked out in a local worktree |
| `claude/gpu-barcode-layer-484c70` | 46484d80 | 2026-10-29 | - | no | checked out in a local worktree |
| `claude/loader-compare` | ae146dbe | 2026-10-28 | - | no | checked out in a local worktree |
| `claude/pypi-singlet` | 540a6de0 | 2026-10-28 | - | prod/campaign, main, gpu/frontier | checked out in a local worktree |
| `claude/r12-posterior-099` | de651cbb | 2026-10-31 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `claude/s1-client-labels` | 69ee7b42 | 2026-10-27 | - | prod/campaign, main, gpu/frontier | checked out in a local worktree |
| `claude/s1-rename` | 17b9befe | 2026-10-26 | - | prod/campaign, main, gpu/frontier | checked out in a local worktree |
| `claude/v2-cell-features` | efca0a81 | 2026-10-24 | - | prod/campaign, main, gpu/frontier | checked out in a local worktree |
| `contract/junctions-psi-reference` | 3230ced2 | 2026-10-31 | #138 (upd 2026-10-31) | no | open PR #138 updated in the last 7 days |
| `contract/su-longread` | 19f1ff82 | 2026-11-01 | #172 (upd 2026-11-01) | no | open PR #172 updated in the last 7 days |
| `dependabot/github_actions/actions/checkout-7.0.1` | d24288c6 | 2026-11-02 | #183 (upd 2026-11-02) | no | dependabot PR #183, CI green: listed for the owner |
| `dependabot/github_actions/actions/download-artifact-8.0.2` | 25a0520e | 2026-11-02 | #185 (upd 2026-11-02) | no | dependabot PR #185, CI green: listed for the owner |
| `dependabot/github_actions/pypa/cibuildwheel-4.3.5` | 31dc4470 | 2026-11-02 | #184 (upd 2026-11-02) | no | dependabot PR #184, CI green: listed for the owner |
| `docs/beta-rollout-plan` | 6aa280c7 | 2026-10-29 | - | prod/campaign, main, gpu/frontier | checked out in a local worktree |
| `docs/citation` | 4b9b2239 | 2026-10-30 | #103 (upd 2026-10-30) | no | checked out in a local worktree |
| `feat/atlas-parallel-ranges` | 34c56a3e | 2026-10-29 | #89 (upd 2026-10-29) | no | checked out in a local worktree |
| `feat/campaign-orchestrator` | ebd59e34 | 2026-10-25 | - | prod/campaign, main, gpu/frontier | checked out in a local worktree |
| `feat/chimeric-table` | aaebe141 | 2026-10-31 | - | no | checked out in a local worktree |
| `feat/doublet-score-v2` | b1f9b3bb | 2026-10-31 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `feat/genotype-outputs` | 5ce93861 | 2026-11-06 | #197 (upd 2026-11-06) | no | checked out in a local worktree |
| `feat/input-routes` | 75529abc | 2026-11-06 | - | no | checked out in a local worktree |
| `feat/r-parallel-ranges` | ec8f5062 | 2026-10-29 | #90 (upd 2026-10-29) | no | checked out in a local worktree |
| `feat/s1-write-and-stream` | 6dcaf08e | 2026-10-28 | - | no | checked out in a local worktree |
| `fix/ambient-calibration` | a57faa14 | 2026-10-26 | - | no | checked out in a local worktree |
| `fix/atlas-rerun-exit8` | 74662cc1 | 2026-10-29 | - | no | checked out in a local worktree |
| `fix/called-v2-golden-tolerance` | c8cf92aa | 2026-10-25 | - | prod/campaign, main, gpu/frontier | checked out in a local worktree |
| `fix/crispr-guide-orientation` | 06bfd70f | 2026-10-30 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `fix/editing-comment` | ec71ed8f | 2026-10-31 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `fix/intron-assignment` | 688aeb44 | 2026-10-26 | - | no | checked out in a local worktree |
| `fix/intron-rule-default` | 006af2ac | 2026-10-29 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `fix/is-cell-called-v2` | c844c044 | 2026-10-31 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `fix/nocall-rescue` | 878e50b2 | 2026-11-01 | #177 (upd 2026-11-01) | no | open PR #177 updated in the last 7 days |
| `fix/pilot-demux-cellcall` | 1b4f9fc7 | 2026-11-06 | #199 (upd 2026-11-06) | no | checked out in a local worktree |
| `fix/protocol-geometry` | dcea5ef9 | 2026-10-30 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `fix/s1-strip-mt-doublet` | bf71d8e1 | 2026-10-28 | - | no | checked out in a local worktree |
| `fix/s1cells-fork-inflight` | 714c92c9 | 2026-10-30 | - | main | checked out in a local worktree |
| `fix/s1stream-batch` | 94f8129a | 2026-10-30 | - | main | checked out in a local worktree |
| `fix/usa-labels` | 3ade35c4 | 2026-10-31 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `fix/usa-legacy-layers` | 80e2c23f | 2026-10-31 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `fix/write-s1-gene-pack` | 04d8dcc2 | 2026-11-02 | #187 (upd 2026-11-02) | no | checked out in a local worktree |
| `gate/release-gate-4` | 66841e15 | 2026-11-01 | #179 (upd 2026-11-01) | no | open PR #179 updated in the last 7 days |
| `main` | 7991abc9 | 2026-11-02 | - | main | main / prod/* |
| `paper/s1-benchmark-notes` | d285eb7c | 2026-11-02 | - | main | checked out in a local worktree |
| `perf/deep-samples` | 83ba3c5d | 2026-10-26 | - | no | checked out in a local worktree |
| `perf/star-accel` | fffca54e | 2026-10-25 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `perf/txome-first` | e1cc282b | 2026-10-27 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `perf/w3-speedups` | 9dbd42f2 | 2026-10-27 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `plan/atlas-rerun` | 1cc385e3 | 2026-10-26 | - | prod/campaign, main, gpu/frontier | checked out in a local worktree |
| `prod/atlas-rerun` | 8c3de82f | 2026-10-26 | - | no | main / prod/* |
| `prod/campaign` | 388bba32 | 2026-11-06 | #78 (upd 2026-11-06) | prod/campaign | main / prod/* |
| `prod/modality-fixes` | 4f110f28 | 2026-10-25 | - | prod/campaign, gpu/frontier | main / prod/* |
| `release/1.3.1` | e29a2d78 | 2026-10-30 | - | main | release branch; the v1.3.1 release session (#78) is running |
| `release/r-cran-1.2.0` | cc84eecf | 2026-10-28 | - | no | release branch; the v1.3.1 release session (#78) is running |
| `s11/gate3-harness` | 4530fcdf | 2026-11-01 | - | prod/campaign, gpu/frontier | checked out in a local worktree |
| `s16/companion-stage` | fceea2a9 | 2026-10-31 | #178 (upd 2026-11-01) | no | open PR #178 updated in the last 7 days |
| `s16/doublet-published` | f736a309 | 2026-11-01 | #162 (upd 2026-11-01) | no | open PR #162 updated in the last 7 days |
| `s16/policy-lift-doublets` | 5ddeb71b | 2026-11-01 | #163 (upd 2026-11-01) | no | open PR #163 updated in the last 7 days |
| `study/latent-qc-ambient-atlas` | 3abf26e8 | 2026-10-25 | - | no | checked out in a local worktree |
| `study/latent-qc-ambient-truth` | 67d4d522 | 2026-10-25 | - | no | checked out in a local worktree |

## Pull requests closed

These open pull requests had no activity for over 7 days. Each head was tagged before the pull request was closed with a comment naming the tag.

| PR | Branch | Tip | Tag |
|---|---|---|---|
| #84 | `claude/bench-public-doublets` | 34ba33e0 | `archive/2026-11-06/claude/bench-public-doublets` |
| #83 | `claude/doublet-score` | cfe5cfa2 | `archive/2026-11-06/claude/doublet-score` |
| #79 | `bench/release-gate` | 5577bd7f | `archive/2026-11-06/bench/release-gate` |
| #81 | `docs/beta-doublet-decision` | 3274c39f | `archive/2026-11-06/docs/beta-doublet-decision` |
| #51 | `claude/vdj-crispr-benchmark-721d3a` | 86f2cafd | `archive/2026-11-06/claude/vdj-crispr-benchmark-721d3a` |
| #43 | `study/latent-qc` | 8a577893 | `archive/2026-11-06/study/latent-qc` |

The dependabot pull requests #183, #184 and #185 pass CI and stay open.

## Deletion log

| Branch | Tip SHA | Last commit | Kept by |
|---|---|---|---|
| `bench/bam-stream` | 68646bb3d535c7839ddcb0e8a9bfb2f9bd985dd0 | 2026-10-30 | tag `archive/2026-11-06/bench/bam-stream` |
| `bench/mtdna-lineage` | d08c7c856d980050e368a83b1468dfde37da86a9 | 2026-10-26 | reachable from prod/campaign, main, gpu/frontier |
| `bench/release-gate` | 5577bd7f97d56caee0904266683e448bc3e35af6 | 2026-10-29 | tag `archive/2026-11-06/bench/release-gate` |
| `bench/s8-musc-kflank` | cf202b32d6f4a0b70819f596ccd31e6ced1e568b | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `bench/sex-harness-s1` | 817d893c4bb6dfabe8ce26339954a71627921b8e | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `ci/docs-only-build` | 1e863c32859f3cbb668682c962e3a27cc0610dc9 | 2026-10-24 | reachable from prod/campaign, main, gpu/frontier |
| `claude/adt-modality-0637a6` | d29cbbc5bb40f46281fafc31bfa8df54b9c7a646 | 2026-10-26 | tag `archive/2026-11-06/claude/adt-modality-0637a6` |
| `claude/alpha-qc` | bec55db6331dc5683b85b9d5acb5bac65d218dd4 | 2026-10-27 | reachable from prod/campaign, main, gpu/frontier |
| `claude/alpha-store-transcode` | 6db779a95e983da73b632af1293ea27a03012fc5 | 2026-10-27 | reachable from prod/campaign, main, gpu/frontier |
| `claude/ancestry-label-rule` | ef18983451ed61ae7726ab142118ccea784b02cd | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `claude/atac-spatial-benchmark-2ba7df` | 21477f1ac02d14c6b27696b3e3bd1373f4457ed9 | 2026-10-29 | reachable from prod/campaign, gpu/frontier |
| `claude/bench-public-cohort-485cf7` | aef9d59b7fe72b92dffe27e43a731d3c0fc40dde | 2026-10-29 | reachable from prod/campaign, main, gpu/frontier |
| `claude/bench-public-doublets` | 34ba33e0452ff46288499e7670b02469ee9ad407 | 2026-10-29 | tag `archive/2026-11-06/claude/bench-public-doublets` |
| `claude/called-v2-migration` | c48f8e86cfc089708949aa44df4eb2562ce97fd3 | 2026-10-23 | reachable from prod/campaign, main, gpu/frontier |
| `claude/client-fixes` | 7af03c2f70b3a77d59309be940ccc81a789c1529 | 2026-09-30 | tag `archive/2026-11-06/claude/client-fixes` |
| `claude/cpu-perf-c66d3f` | 181c1a8b96f89f77531456475954f8f1ba809f81 | 2026-10-31 | tag `archive/2026-11-06/claude/cpu-perf-c66d3f` |
| `claude/cpu-perf-flags` | a8753ee1abf01e2d5ea478de92d62a39bab5393b | 2026-10-31 | tag `archive/2026-11-06/claude/cpu-perf-flags` |
| `claude/cpu-perf-measure` | 77726b717f02c73316918e3774b8fc6661dbb38e | 2026-10-31 | tag `archive/2026-11-06/claude/cpu-perf-measure` |
| `claude/cpu-perf-mem` | d74f9fac6c14583e345e67b319d5ce04a9760faa | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `claude/cpu-perf-tail` | 181bba8df8aa3f856cc35a105250db0e9cc7677c | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `claude/cpu-perf-transcript` | 8e501ee6ca53c2846c0a15da0cece423e1043db3 | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `claude/determined-swartz-8fd810` | d08c7c856d980050e368a83b1468dfde37da86a9 | 2026-10-26 | reachable from prod/campaign, main, gpu/frontier |
| `claude/doublet-score` | cfe5cfa2960bf79caa09f9136c1a92aeaa3865fa | 2026-10-29 | tag `archive/2026-11-06/claude/doublet-score` |
| `claude/download-staging-cf698f` | 79d615595ffc0d92d6d5315e9a638c3c5b59b674 | 2026-10-30 | reachable from prod/campaign, gpu/frontier |
| `claude/drain-multisample-52667f` | b17909115743ff2cc2a2244ab6bcfdb5c974d1cb | 2026-11-01 | tag `archive/2026-11-06/claude/drain-multisample-52667f` |
| `claude/empty-droplet-check-624070` | 8b484b67f735430a3d4146c5c0c2f0987837fef9 | 2026-10-24 | reachable from prod/campaign, main, gpu/frontier |
| `claude/festive-vaughan-4a4a8f` | f25d7cabdede2a3bc06cdee835af9f5cad8f175e | 2026-10-23 | reachable from prod/campaign, main, gpu/frontier |
| `claude/fix-client-bugs` | a7bc6f8c8e5136a153eacef7a97dddc0dde3f1c5 | 2026-09-04 | reachable from prod/campaign, main, gpu/frontier |
| `claude/gpu-host-cost-3482d0` | 019f8949b630e02b4055ba78208622f4045d79d3 | 2026-10-29 | tag `archive/2026-11-06/claude/gpu-host-cost-3482d0` |
| `claude/gpu-paired-prep-f54059` | 1acfd71247ffd0798f032d54693e523885cff487 | 2026-10-29 | tag `archive/2026-11-06/claude/gpu-paired-prep-f54059` |
| `claude/hto-demux` | b53f4fec6493b4b141dada9498df07cc929ade2a | 2026-10-30 | reachable from prod/campaign, gpu/frontier |
| `claude/laughing-wing-cdaacf` | 6b6a1b45499ae02cb67b870ff324b7e5eb4b7081 | 2026-10-24 | reachable from prod/campaign, main, gpu/frontier |
| `claude/lint-release` | 633d113161614c0e158424d2fdff31f5a8c4ea61 | 2026-09-30 | tag `archive/2026-11-06/claude/lint-release` |
| `claude/ml-ready-loader` | 02b318ca5627ac10ef8324e2bfece8ec93f053d5 | 2026-10-28 | reachable from prod/campaign, main, gpu/frontier |
| `claude/nonhost-confirmed-reads` | a0b7da9bd6976ace282eb5ad67af40e38b705cff | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `claude/pz2-nnz-chunks` | 5eae6ac3570b501a4551546f3dfa21816db2a2e3 | 2026-10-24 | reachable from prod/campaign, main, gpu/frontier |
| `claude/release-gate-e1e139` | 4942348e260e7d5883164db74c2dbfd6328cbaea | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `claude/rename-to-1-0-447a9c` | 36ff33d236e3baaee324e155558db0052e159c81 | 2026-10-26 | reachable from prod/campaign, main, gpu/frontier |
| `claude/s12-ambient` | f9fba86a2c84f724602c8ef2593f08f0164fa503 | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `claude/s12-gate2` | 4d4b1a296e10764d234d235c1cf53554b9b8f676 | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `claude/s6-orchestrator` | a7b015c925731207599332d1b0bb2860f5812f79 | 2026-10-30 | reachable from prod/campaign, gpu/frontier |
| `claude/stage-1fg2` | f37ad86c0a88f5dc88e31b2f63af49eb0af06580 | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `claude/star-bamsort-budget` | 13b047200439199863c85c8c0bd97cf873fbc764 | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `claude/v2-cell-annotation-6e50dc` | a8893811c222e765ee3eaec7da27c17660a6e0b4 | 2026-10-24 | reachable from prod/campaign, main, gpu/frontier |
| `claude/v2-cell-index-5ba279` | c0118bea6779e89b472c21b20d6d213624e38148 | 2026-10-23 | reachable from prod/campaign, main, gpu/frontier |
| `claude/vdj-crispr-benchmark-721d3a` | 86f2cafd10753767b5041736677b968946278cf5 | 2026-10-26 | tag `archive/2026-11-06/claude/vdj-crispr-benchmark-721d3a` |
| `docs/beta-doublet-decision` | 3274c39ff99e40dcb78993226323047e5626ed1c | 2026-10-29 | tag `archive/2026-11-06/docs/beta-doublet-decision` |
| `exp/doublet-lite` | 0b99d7c40426c18caa7bb16e3bbe66697144bf04 | 2026-10-31 | tag `archive/2026-11-06/exp/doublet-lite` |
| `exp/umi-seed-on-counts` | 09977a3e70b5e45b13316f853cb4f442f73c548b | 2026-11-01 | tag `archive/2026-11-06/exp/umi-seed-on-counts` |
| `feat/client-sample-facts` | afeabcb8270be66ee7eccde452c8eced0855817f | 2026-10-28 | tag `archive/2026-11-06/feat/client-sample-facts` |
| `feat/reads-per-umi` | 1767d7a1efd18282dd3bf21b38097e83656ed54b | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `feat/restrict-genotype-members` | c7ee777dd1e2a1d0ff3ae951b3eb36306cb76e71 | 2026-10-22 | reachable from prod/campaign, main, gpu/frontier |
| `feat/s1-exclude-reads` | 2b7898f76a9b3d6084b8b870de2d379a204bc4bf | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `feat/s1-reads-members` | 537fb81ff63db47c817629857d04420a4db3a620 | 2026-10-31 | tag `archive/2026-11-06/feat/s1-reads-members` |
| `feat/s1-reads-values` | dd71187b854bc2cc8e21ce3e3ba65ae847c12d18 | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `feat/s1stream-concurrent-ranges` | 8153c461ae2d21c44592d9aca1fdfa543b68f344 | 2026-10-29 | reachable from main |
| `feat/singlet-v2` | 6fca1b8824081190c143f00799a094764570d296 | 2026-10-24 | reachable from prod/campaign, main, gpu/frontier |
| `feat/v2-junction-table` | aa4a82a6fc1a01a5b4328b0dffdb7f0ffe734757 | 2026-10-24 | reachable from prod/campaign, main, gpu/frontier |
| `feat/v2-migrate-junction-table` | edd6308f21f76df53f8359f7b088fd7bfb2bf4d2 | 2026-10-24 | reachable from prod/campaign, main, gpu/frontier |
| `feat/v2-modalities` | e6ee6e05eab91e336a0d65ac70a2628810c12b3b | 2026-10-23 | reachable from prod/campaign, main, gpu/frontier |
| `fix/ambient-headline` | aa3712ab5ab734bb8546898b183d63770491c1ec | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `fix/barcode-universe` | e1a4217b2b724f674c619a56f72a2951b2a777c1 | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `fix/beta-engine-blockers` | 8a140e0775435e1e436a2912e05339c561d0b043 | 2026-10-29 | reachable from prod/campaign, gpu/frontier |
| `fix/called-ed-paths` | faec1fe59797a9eef7f58016c484a85badaa693b | 2026-10-29 | reachable from prod/campaign, main, gpu/frontier |
| `fix/called-v2-knee-bounds` | 9814fb8a70faa76e459c292eb708a2a2509878ed | 2026-10-25 | reachable from prod/campaign, main, gpu/frontier |
| `fix/count-matrices` | 797e78f2df0db8e350fa9b1a42f93768c56528ab | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `fix/fifo-open-race` | ec9d80a31ba6d332e1f6f83a559ec0166f764908 | 2026-10-28 | reachable from prod/campaign, gpu/frontier |
| `fix/gate-log-final` | 836402c814bf874ec82d26019bb18e64e04cd468 | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `fix/gate2-s8` | 0bc98ffe24e759c58e3cdd954df7dfceaec269a2 | 2026-11-01 | tag `archive/2026-11-06/fix/gate2-s8` |
| `fix/gtf-next-to-genome-dir` | fd804b5486a791df043a86ae9b3735548d7f253f | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `fix/memory-tier-config` | 730674e89b4e8c9da59f748f18250e7d3025c8a9 | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `fix/molecule-labels` | 775e60dbf04f400b7ddc5a27efa1b0083595e940 | 2026-10-24 | reachable from prod/campaign, gpu/frontier |
| `fix/nonhost-subsample-alignment` | 5e559e84cf4118ee9735fb11f19cc97e19311f2d | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `fix/organism-reference` | 398142bbe29ae954a75f3619b330a46145c112f6 | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `fix/parked-companions` | d2198e4637e607426969f42f514cb1344f0f0347 | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `fix/pileup-thread-independence` | c96e64b57d298873795e2a2fbdd4188d69ea4a3c | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `fix/pinned-binaries` | ab404bc9638c6d5192e263b78f70a4a268f5d320 | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `fix/protocol-short-umi` | 83730c1e3b1c55c49d3ef9c710e34f6ec9e683dd | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `fix/protocol-vote` | e833ad7023e4648d38f879cf792b66a766f06d2a | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `fix/r1-untrimmed-protocol` | 8a0f5d3ca33c82021dd3ccbd52880bbed2e05962 | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `fix/s1stream-range-200` | 4891b4f07e3699df1e774071d152e8c2fce43037 | 2026-10-29 | reachable from prod/campaign, main, gpu/frontier |
| `fix/sj-unique-only` | 3eaa041c872fd763ab7e6dc6dce8b547bf8db87c | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `fix/snp-false-alt` | 0e8b51d668cb598473e4facd145a3ee9785305ba | 2026-10-30 | reachable from prod/campaign, gpu/frontier |
| `fix/species-bloom` | aa78380464ebf32a5767fe7b418d71a96a8da7ad | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `fix/stall-watchdog` | 22b5782f90ec5af907ac46c121f5c638a07613ba | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `fix/strand-detection` | af71c98ffd08860fe8a4aa3c10a223a6bf20e9a8 | 2026-10-24 | reachable from prod/campaign, gpu/frontier |
| `fix/stream-strand-fallback` | 6a6a6940903bd2259d4b049b7b223d015bb14382 | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `fix/umi-collapse-order-free` | e6262380ebdc21f9141ca5577d74af080beb10fd | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `fix/umi-ur-first` | d315f374ed24e7caa130ccff996badcaa5a94a44 | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `fix/usa-labels-base` | f15dad072fbd7ccc65749683f0036ae0ff5cfbe3 | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `fix/usa-labels-base2` | 1b7bf90d991e7082b04dc25d6712046d6a1bff39 | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `fix/usa-labels-gpu` | c38b7ef2a5ff86b2037f287f46463724763d27cb | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `fix/vote-main-family` | 2934e2fdb6ea58073e38e3e6701750ea60517c8e | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `gate/release-gate-2` | b1e0827d0cb80cf9e8f9ec3f8ae06c0b354a1343 | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `gate/release-gate-3` | a7d25e52b539ae85c25a43be40cb196309ff957a | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `integrate/atac-onto-campaign` | 21477f1ac02d14c6b27696b3e3bd1373f4457ed9 | 2026-10-29 | reachable from prod/campaign, gpu/frontier |
| `integrate/campaign-sync-main` | fc7e3523b3ef9f5ae869eede29d98fbe387b0df9 | 2026-10-29 | reachable from prod/campaign, gpu/frontier |
| `mc/p117` | 7ad431bb404d25b5fb29336e9fd8444dd56e903f | 2026-10-31 | tag `archive/2026-11-06/mc/p117` |
| `mc/p119` | 98bd78735c1823f35de0980753e487b477f94b27 | 2026-10-31 | tag `archive/2026-11-06/mc/p119` |
| `mc/p160` | 63743bb289f63e12f1290c25ad245ab8bf369e07 | 2026-11-01 | tag `archive/2026-11-06/mc/p160` |
| `mc/p163` | f790523d6098f54c3bc41c79120e4407c207a885 | 2026-11-01 | tag `archive/2026-11-06/mc/p163` |
| `mc/p165` | 630cc2b03a25cd3fede546fec30fdd7bdda38d68 | 2026-11-01 | tag `archive/2026-11-06/mc/p165` |
| `mc/p166` | 4a6d3fad78a555618ba036a25833ddd92e605827 | 2026-11-01 | tag `archive/2026-11-06/mc/p166` |
| `mc/p173` | dee6819582b5db17908af6f6b441a3eca0de1b82 | 2026-11-01 | tag `archive/2026-11-06/mc/p173` |
| `mc/p176` | 05064073319e70c345863b582d4a68237889c4d3 | 2026-11-01 | tag `archive/2026-11-06/mc/p176` |
| `mc/p17x` | d0094c41b99a7b66a9fe67a745949b18fdef3092 | 2026-11-01 | tag `archive/2026-11-06/mc/p17x` |
| `mc/p181` | b5881a5190abab3a414e2d5871ca1d05252ff039 | 2026-11-01 | tag `archive/2026-11-06/mc/p181` |
| `mc/pall` | 627f47f1be66ddd71a7214c383e0eb1162131798 | 2026-11-01 | tag `archive/2026-11-06/mc/pall` |
| `paper/s1-benchmark-plan` | b66f121e64b7bd08b0ef173d89c8778afd30156f | 2026-10-29 | reachable from main |
| `paper/s1-benchmark-results` | 08f43f0b88f6f03c154f7367f3fd33009acad3ef | 2026-11-02 | reachable from main |
| `paper/s1-tcbb-ai-use` | 36927e7966806c76d3b02d9c1c885de8b25d1388 | 2026-11-02 | reachable from main |
| `perf/bam-stream` | d1b7abb19019f65196c373eee8c09c3831e18418 | 2026-10-29 | reachable from prod/campaign, gpu/frontier |
| `perf/drain-memory-admission` | 9675e35bc9f15e336c801f142bdf89f8dbee5d2d | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `perf/fq-read-source` | 7cd5069f6f0f3351b1efca655509d3c66add54d2 | 2026-10-29 | reachable from prod/campaign, gpu/frontier |
| `prof/star-phase-marks` | f2c77334d25864794c661365e2d37d79a4dec16e | 2026-10-31 | tag `archive/2026-11-06/prof/star-phase-marks` |
| `rename-singlify-to-singlet` | 5ccea86bb367a6a4346e8a137ede644ec10b0c58 | 2026-05-18 | reachable from prod/campaign, main, gpu/frontier |
| `s16/companion-fetch` | 91b04694a69a7e88a1b45b6b9dbb9ee94b03a0b1 | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `s16/members-registry` | 9196c0e2c454b7ce6c8a8cf043110634e1a96979 | 2026-11-01 | tag `archive/2026-11-06/s16/members-registry` |
| `s16/packages-adt-vdj-crispr` | 393f7b2da62d8995d3befbe71c19b38ed5a4e784 | 2026-10-31 | reachable from prod/campaign, gpu/frontier |
| `s16/private-sidecar` | b02b0a7f348b43c27b7a5b0204b1450c0027c0e0 | 2026-11-01 | reachable from prod/campaign, gpu/frontier |
| `s7/beta-aligner-config` | d637b3ebf6dcb9664881a4c7133e0e421c90e548 | 2026-10-30 | reachable from prod/campaign, gpu/frontier |
| `study/latent-qc` | 8a5778938378c6ef6a8ade94b2a23e4467921247 | 2026-10-25 | tag `archive/2026-11-06/study/latent-qc` |

## Left to the GPU line

These branches are handled with `gpu/frontier` and listed in `GPU_BRANCHES.md` there: `docs/gpu-handoff-2026-11-06`, `fix/gpu-strand-restart`, `gpu/bold`, `gpu/bold-admit`, `gpu/bold-hostcut`, `gpu/bold-integ`, `gpu/bold-kern`, `gpu/bold-kern2`, `gpu/bold-shidx`, `gpu/bold-shidx-fix`, `gpu/bold-split`, `gpu/cr4-clip`, `gpu/engine-gpu`, `gpu/engine-gpu-order-free`, `gpu/engine-gpu-stream`, `gpu/fable-engine`, `gpu/fable/beta-notes`, `gpu/fable/budget`, `gpu/fable/census`, `gpu/fable/compact`, `gpu/fable/e2e`, `gpu/fable/hostloop`, `gpu/fable/ingest`, `gpu/fable/lockclip`, `gpu/fable/mtpileup`, `gpu/fable/order`, `gpu/fable/pool`, `gpu/fable/posterior`, `gpu/fable/prefill`, `gpu/fable/probe`, `gpu/fable/profile`, `gpu/fable/records`, `gpu/fable/release`, `gpu/fable/seeds`, `gpu/fable/solo`, `gpu/fable/solowire`, `gpu/fable/tail`, `gpu/fable/throughput`, `gpu/fable/warp-window`, `gpu/fg2-engine`, `gpu/fg2-table-prefill`, `gpu/frontier`, `gpu/gpu-lock-release`, `gpu/gpu-pileup-preflight`, `gpu/gpu-pileup-quality`, `gpu/gpu-slots`, `gpu/gpu-slots2`, `gpu/host-offload`, `gpu/intron-rules`, `gpu/w11-prototypes`, `perf/gpu-candidate-pr2`, `perf/gpu-device`, `perf/gpu-instrumentation`, `perf/gpu-next`, `perf/gpu-next-e20`, `perf/gpu-next-e20e21`, `perf/gpu-next-e21`, `perf/gpu-pinned-dedup`, `perf/nvtx-host-ranges`, `perf/pileup-lock-release`.
