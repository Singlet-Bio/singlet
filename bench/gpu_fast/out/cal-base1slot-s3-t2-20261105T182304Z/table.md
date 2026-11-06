**A** bin `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-next/bin/base` commit `0a6de829c75be8e216e2210a1622cc80b87df0de` sha256 `b6e0795d0c6cdc72` embedded yes env `SINGLET_ENGINE_GPU_SLOTS=1` args `-`
**B** no leg in the result
tier 2 job 800357 node g051 scripts `95b617e6` ref `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-batch1/ref/e42c0636/T7/pbmc1k_v3` sample pbmc1k_v3 T=7 slots=2 k=8 runs per leg=24
order A
note: STOP fallback in leg 1 (A) after leg 1 of 4
note: arm A: env sets SINGLET_ENGINE_GPU_SLOTS, which the harness also sets. The arm value wins.
note: tier 2 uses the provisional rule {"core_gain": 0.02, "rate_gain": 0.09, "lock_held_gain": 0.1}. Span rate is shown but never decides.

| measure | A | B | B vs A | A legs | B legs |
|---|---|---|---|---|---|
| progress rate M pairs/s | NA | NA | NA | NA |  |
| span rate M pairs/s (same run count only) | 0.883 | NA | NA | 0.883 |  |
| wall s | 461.2 | NA | NA | 461.2 |  |
| core-s/M | 16.50 | NA | NA | 16.50 |  |
| lock held s | 32.3 | NA | NA | 32.3 |  |
| lock wait s | 66.1 | NA | NA | 66.1 |  |
| device peak GB | 90.1 | NA | | | |
| host peak GB | 370.0 | NA | | | |
| identity vs reference | 1/1 | 0/0 | | | |
| fallback runs | 11 | 0 | | | |
| failed runs | 0 | 0 | | | |

Stage core-s (mean per run)

| stage | start | scan | refs | STAR | stream | handoff | pileup | pileupfin | nonhost | export | demux |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 94 | 9 | 610 | 53 | 84 | 6 | 0 | 36 | 37 | 137 |
| B | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

Fallback lines of A: c0: level 5 NO / c1: level 5 NO / c2: level 5 NO

Stopped early: fallback in run c9: level 5 NO

**VERDICT: INCONCLUSIVE** (rule V1) an arm has no leg in the result
Parameters: core_gain=0.02, rate_gain=0.09, lock_held_gain=0.1, intended_change=False
