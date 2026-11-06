**A** bin `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-next/bin/s3` commit `e42c06362b7584ef5b9f323bfa1eff95a590488c` sha256 `1772c91f36463b86` embedded yes env `-` args `-`
**B** bin `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-next/bin/s3` commit `e42c06362b7584ef5b9f323bfa1eff95a590488c` sha256 `1772c91f36463b86` embedded yes env `-` args `--n-donors 1`
tier 2 job 800381 node g051 scripts `b73b2435` ref `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-batch1/ref/e42c0636/T7/pbmc1k_v3` sample pbmc1k_v3 T=7 slots=2 k=8 runs per leg=24
order A B B
note: CO-TENANT. These legs shared the node with another job: leg 1 (A), leg 2 (B). Read their numbers with care.
note: STOP arms clearly separated (B better) after leg 3 of 4
note: arm B: extra command line arguments ['--n-donors', '1']. They can change the output.
note: an output change was declared. Identity loss is shown and does not reject.
note: tier 2 uses the provisional rule {"core_gain": 0.02, "rate_gain": 0.09, "lock_held_gain": 0.1}. Span rate is shown but never decides.

| measure | A | B | B vs A | A legs | B legs |
|---|---|---|---|---|---|
| progress rate M pairs/s | 1.826 | 2.154 | +18.0% | 1.826 | 2.063, 2.245 |
| span rate M pairs/s (same run count only) | 1.690 | 1.995 | +18.0% | 1.690 | 1.973, 2.017 |
| wall s | 309.3 | 255.3 | +17.5% | 309.3 | 260.1, 250.5 |
| core-s/M | 9.89 | 8.02 | +19.0% | 9.89 | 8.05, 7.98 |
| lock held s | 43.6 | 44.0 | -1.0% | 43.6 | 45.3, 42.8 |
| lock wait s | 139.3 | 131.8 | +5.4% | 139.3 | 133.2, 130.3 |
| device peak GB | 89.3 | 89.5 | | | |
| host peak GB | 346.0 | 343.0 | | | |
| identity vs reference | 2/2 | 0/4 | | | |
| fallback runs | 0 | 0 | | | |
| failed runs | 0 | 0 | | | |

Stage core-s (mean per run)

| stage | start | scan | refs | STAR | stream | handoff | pileup | pileupfin | nonhost | export | demux |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 2 | 29 | 90 | 92 | 172 | 29 | 1 | 2 | 42 | 37 | 143 |
| B | 1 | 30 | 90 | 93 | 169 | 24 | 1 | 7 | 44 | 0 | 55 |

**VERDICT: PROMOTE** (rule T2-4) core-s/M falls -19.0%, past 2%. Identity kept, no fallback
Parameters: core_gain=0.02, rate_gain=0.09, lock_held_gain=0.1, intended_change=True

Without the co-tenant legs: **INCONCLUSIVE** (rule V1) an arm has no leg in the result
