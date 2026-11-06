**A** bin `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-next/bin/s3` commit `e42c06362b7584ef5b9f323bfa1eff95a590488c` sha256 `1772c91f36463b86` embedded yes env `-` args `-`
**B** bin `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-next/bin/s3` commit `e42c06362b7584ef5b9f323bfa1eff95a590488c` sha256 `1772c91f36463b86` embedded yes env `-` args `-`
tier 1 job 800305 node g051 scripts `95b617e6` ref `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-batch1/ref/e42c0636/T7/pbmc1k_v3` sample pbmc1k_v3 T=7 slots=2
order A B B A

| measure | A | B | B vs A | A legs | B legs |
|---|---|---|---|---|---|
| wall s | 152.9 | 152.5 | +0.3% | 152.6, 153.3 | 152.2, 152.8 |
| core-s/M | 8.99 | 8.96 | +0.4% | 9.00, 8.98 | 8.95, 8.96 |
| lock held s | 38.2 | 37.2 | +2.7% | 37.4, 39.1 | 36.5, 37.9 |
| lock wait s | 0.0 | 0.0 | NA | 0.0, 0.0 | 0.0, 0.0 |
| device peak GB | 44.1 | 44.1 | | | |
| host peak GB | 57.0 | 55.0 | | | |
| identity vs reference | 2/2 | 2/2 | | | |
| fallback runs | 0 | 0 | | | |
| failed runs | 0 | 0 | | | |

Stage core-s (mean per run)

| stage | start | scan | refs | STAR | stream | handoff | pileup | pileupfin | nonhost | export | demux |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0 | 22 | 81 | 89 | 141 | 35 | 2 | 0 | 32 | 34 | 135 |
| B | 1 | 20 | 82 | 88 | 143 | 35 | 0 | 0 | 33 | 35 | 133 |

**VERDICT: INCONCLUSIVE** (rule T1-2) the best gain is lock_held_s +2.7%, below the promotion gain 5.0%
Parameters: promote_gain=0.05, respect_spread=True, regress_guard=unset, intended_change=False
