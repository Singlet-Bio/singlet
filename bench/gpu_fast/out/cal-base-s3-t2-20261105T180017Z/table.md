**A** bin `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-next/bin/base` commit `0a6de829c75be8e216e2210a1622cc80b87df0de` sha256 `b6e0795d0c6cdc72` embedded yes env `-` args `-`
**B** no leg in the result
tier 2 job 800342 node g051 scripts `95b617e6` ref `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-batch1/ref/e42c0636/T7/pbmc1k_v3` sample pbmc1k_v3 T=7 slots=2 k=8 runs per leg=24
order A
note: STOP fallback in leg 1 (A) after leg 1 of 4
note: tier 2 uses the provisional rule {"core_gain": 0.02, "rate_gain": 0.09, "lock_held_gain": 0.1}. Span rate is shown but never decides.

| measure | A | B | B vs A | A legs | B legs |
|---|---|---|---|---|---|
| progress rate M pairs/s | NA | NA | NA | NA |  |
| span rate M pairs/s (same run count only) | 0.475 | NA | NA | 0.475 |  |
| wall s | 885.5 | NA | NA | 885.5 |  |
| core-s/M | 41.34 | NA | NA | 41.34 |  |
| lock held s | 29.7 | NA | NA | 29.7 |  |
| lock wait s | 21.2 | NA | NA | 21.2 |  |
| device peak GB | 90.1 | NA | | | |
| host peak GB | 370.0 | NA | | | |
| identity vs reference | 1/1 | 0/0 | | | |
| fallback runs | 9 | 0 | | | |
| failed runs | 0 | 0 | | | |

Stage core-s (mean per run)

| stage | start | scan | refs | STAR | stream | handoff | pileup | pileupfin | nonhost | export | demux |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 97 | 9 | 324 | 6 | 2041 | 3 | 0 | 40 | 37 | 141 |
| B | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

Fallback lines of A: c0: level 5 NO | star_Log.out: ENGINE_GPU: not used (GPU mapping failed): STAR maps every read / c1: level 5 NO | star_Log.out: ENGINE_GPU: not used (GPU mapping failed): STAR maps every read / c2: level 5 NO | star_Log.out: ENGINE_GPU: not used (GPU mapping failed): STAR maps every read

Stopped early: fallback in run c8: level 5 NO | star_Log.out: ENGINE_GPU: not used (GPU mapping failed): STAR maps every read

**VERDICT: INCONCLUSIVE** (rule V1) an arm has no leg in the result
Parameters: core_gain=0.02, rate_gain=0.09, lock_held_gain=0.1, intended_change=False
