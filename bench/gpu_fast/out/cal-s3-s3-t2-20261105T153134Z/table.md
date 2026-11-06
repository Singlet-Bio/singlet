**A** bin `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-next/bin/s3` commit `e42c06362b7584ef5b9f323bfa1eff95a590488c` sha256 `1772c91f36463b86` embedded yes env `-` args `-`
**B** bin `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-next/bin/s3` commit `e42c06362b7584ef5b9f323bfa1eff95a590488c` sha256 `1772c91f36463b86` embedded yes env `-` args `-`
tier 2 job 800309 node g051 scripts `95b617e6` ref `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-batch1/ref/e42c0636/T7/pbmc1k_v3` sample pbmc1k_v3 T=7 slots=2 k=8 runs per leg=24
order A B B A
note: tier 2 uses the provisional rule {"core_gain": 0.02, "rate_gain": 0.09, "lock_held_gain": 0.1}. Span rate is shown but never decides.

| measure | A | B | B vs A | A legs | B legs |
|---|---|---|---|---|---|
| progress rate M pairs/s | 1.924 | 1.837 | -4.5% | 1.832, 2.017 | 1.955, 1.720 |
| span rate M pairs/s (same run count only) | 1.751 | 1.709 | -2.3% | 1.749, 1.752 | 1.757, 1.662 |
| wall s | 288.0 | 297.5 | -3.3% | 291.2, 284.8 | 282.8, 312.2 |
| core-s/M | 9.84 | 9.88 | -0.4% | 9.81, 9.87 | 9.86, 9.89 |
| lock held s | 42.0 | 41.5 | +1.1% | 40.0, 44.0 | 43.0, 40.1 |
| lock wait s | 113.4 | 119.6 | -5.5% | 121.7, 105.1 | 103.6, 135.7 |
| device peak GB | 89.2 | 89.9 | | | |
| host peak GB | 345.0 | 332.0 | | | |
| identity vs reference | 4/4 | 4/4 | | | |
| fallback runs | 0 | 0 | | | |
| failed runs | 0 | 0 | | | |

Stage core-s (mean per run)

| stage | start | scan | refs | STAR | stream | handoff | pileup | pileupfin | nonhost | export | demux |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0 | 29 | 91 | 84 | 177 | 26 | 1 | 4 | 44 | 37 | 142 |
| B | 1 | 28 | 92 | 87 | 173 | 25 | 1 | 8 | 40 | 37 | 143 |

**VERDICT: INCONCLUSIVE** (rule T2-7) progress rate -4.5%, core-s/M +0.4%, lock held -1.1%: none passes a threshold
Parameters: core_gain=0.02, rate_gain=0.09, lock_held_gain=0.1, intended_change=False
