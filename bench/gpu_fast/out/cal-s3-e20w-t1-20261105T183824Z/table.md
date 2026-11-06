**A** bin `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-next/bin/s3` commit `e42c06362b7584ef5b9f323bfa1eff95a590488c` sha256 `1772c91f36463b86` embedded yes env `-` args `-`
**B** bin `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-next/bin/e20w` commit `aeef7a5e48fd11fba63c27fcf19e97e8bfb3a2f5` sha256 `f40ad295dc8a0e2b` embedded yes env `-` args `-`
tier 1 job 800360 node g051 scripts `95b617e6` ref `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-batch1/ref/e42c0636/T7/pbmc1k_v3` sample pbmc1k_v3 T=7 slots=2
order A B B A

| measure | A | B | B vs A | A legs | B legs |
|---|---|---|---|---|---|
| wall s | 152.9 | 144.2 | +5.7% | 151.8, 153.9 | 143.6, 144.8 |
| core-s/M | 8.94 | 7.79 | +12.8% | 8.93, 8.95 | 7.81, 7.78 |
| lock held s | 39.1 | 27.6 | +29.4% | 38.5, 39.7 | 27.0, 28.2 |
| lock wait s | 0.0 | 0.0 | NA | 0.0, 0.0 | 0.0, 0.0 |
| device peak GB | 44.1 | 44.7 | | | |
| host peak GB | 54.0 | 52.0 | | | |
| identity vs reference | 2/2 | 2/2 | | | |
| fallback runs | 0 | 0 | | | |
| failed runs | 0 | 0 | | | |

Stage core-s (mean per run)

| stage | start | scan | refs | STAR | stream | handoff | pileup | pileupfin | nonhost | export | demux |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 20 | 82 | 84 | 151 | 32 | 0 | 0 | 33 | 34 | 131 |
| B | 0 | 21 | 82 | 32 | 118 | 37 | 2 | 0 | 30 | 34 | 135 |

**VERDICT: PROMOTE** (rule T1-5) lock_held_s improves +29.4% (promotion gain 5.0%), identity kept, no fallback
Parameters: promote_gain=0.05, respect_spread=True, regress_guard=unset, intended_change=False
