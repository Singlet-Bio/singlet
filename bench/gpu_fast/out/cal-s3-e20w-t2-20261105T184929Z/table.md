**A** bin `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-next/bin/s3` commit `e42c06362b7584ef5b9f323bfa1eff95a590488c` sha256 `1772c91f36463b86` embedded yes env `-` args `-`
**B** bin `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-next/bin/e20w` commit `aeef7a5e48fd11fba63c27fcf19e97e8bfb3a2f5` sha256 `f40ad295dc8a0e2b` embedded yes env `-` args `-`
tier 2 job 800365 node g051 scripts `95b617e6` ref `/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-batch1/ref/e42c0636/T7/pbmc1k_v3` sample pbmc1k_v3 T=7 slots=2 k=8 runs per leg=24
order A B
note: STOP fallback in leg 2 (B) after leg 2 of 4
note: tier 2 uses the provisional rule {"core_gain": 0.02, "rate_gain": 0.09, "lock_held_gain": 0.1}. Span rate is shown but never decides.

| measure | A | B | B vs A | A legs | B legs |
|---|---|---|---|---|---|
| progress rate M pairs/s | 1.809 | 2.098 | +16.0% | 1.809 | 2.098 |
| span rate M pairs/s (same run count only) | 1.689 | 1.939 | +14.8% | 1.689 | 1.939 |
| wall s | 298.8 | 263.9 | +11.7% | 298.8 | 263.9 |
| core-s/M | 9.89 | 9.92 | -0.3% | 9.89 | 9.92 |
| lock held s | 45.0 | 36.4 | +19.1% | 45.0 | 36.4 |
| lock wait s | 132.2 | 67.8 | +48.7% | 132.2 | 67.8 |
| device peak GB | 89.2 | 91.9 | | | |
| host peak GB | 349.0 | 298.0 | | | |
| identity vs reference | 2/2 | 2/2 | | | |
| fallback runs | 0 | 1 | | | |
| failed runs | 0 | 0 | | | |

Stage core-s (mean per run)

| stage | start | scan | refs | STAR | stream | handoff | pileup | pileupfin | nonhost | export | demux |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0 | 29 | 93 | 84 | 180 | 28 | 1 | 1 | 42 | 37 | 143 |
| B | 1 | 27 | 94 | 40 | 133 | 117 | 5 | 5 | 40 | 38 | 141 |

Fallback lines of B: c12: level 5 NO | run.log: [engine_gpu] fallback: ENGINE_GPU: not used (GPU mapping failed: engine_gpu: dmal(&dBits, bitsCap * 4): out of memory): STAR maps eve

Stopped early: fallback in run c12: level 5 NO | run.log: [engine_gpu] fallback: ENGINE_GPU: not used (GPU mapping failed: engine_gpu: dmal(&dBits, bitsCap * 4): out of memory): STAR maps every read | star_Log.out: ENGINE_GPU: not used (GPU mapping failed: engine_gpu: dmal(&dBits, bitsCap * 4): out of memory): STAR maps every read

**VERDICT: REJECT** (rule V2) B fell back in 1 run(s). First: c12: level 5 NO | run.log: [engine_gpu] fallback: ENGINE_GPU: not used (GPU mapping failed: engine_gpu: dmal(&dBits, bitsCap * 4): out of memory): STAR maps every read | star_Log.out: ENGINE_GPU: not used (GPU mapping failed: engine_gpu: dmal(&dBits, bitsCap * 4): out of memory): STAR maps every read
Parameters: core_gain=0.02, rate_gain=0.09, lock_held_gain=0.1, intended_change=False
