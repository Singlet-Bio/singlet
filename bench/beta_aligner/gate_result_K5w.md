# Gate result: imp_K5w

**Verdict: PASS**

- gate version 1.0.0, baseline `star_prod_campaign@1` (STAR production, campaign configuration), frozen code manifest sha256 `c826fb4372a7b80299ec87a54d47be297c632548c33adff215f6dd93c4288071`
- production path: ok: identity check, STAR 2.7.11b (/mnt/projects/debruinz_project/debruinz/singlet/envs/pbench/bin/STAR): production invocation vs bench invocation identical on 3835003 reads; the submitted BAMs are bench invocation

| dataset | produced by | status | note |
|---|---|---|---|
| PBMC-E pbmc1 (T1) | candidate | PASS |  |
| PBMC-E pbmc2 (T1-PB) | candidate | PASS |  |
| WTC11 | candidate | PASS |  |
| MIX-1k human cells | candidate | PASS |  |
| MIX-1k mouse cells | candidate | PASS |  |
| CHM13 | candidate | PASS |  |

## PBMC-E pbmc1 (T1) (pbmc1): PASS
Coverage guard: ok: unmapped reads declared omitted: 8608996 missing reads counted as unmapped; names match (probe hit rate 0.9534). Records 219010294 of 227619290 input reads; missing 8608996.
Baseline `star_prod_campaign@1` row `star_prod`; 503216 bootstrap units, 1000 replicates; score JSON identical to the baseline's: False.

| measure | candidate | baseline | difference | 95% CI | margin | verdict |
|---|---:|---:|---:|---|---:|---|
| M1 locus-correct share (default operating point) (higher better) | 0.9557 | 0.9469 | 0.00880 | [0.00866, 0.00893] | 0.001 | PASS |
| M1 wrong share (default operating point) (lower better) | 0.0392 | 0.0383 | 0.00085 | [0.00083, 0.00087] | 0.001 | PASS |
| M2 per-read junction recall (higher better) | 0.9394 | 0.9294 | 0.01006 | [0.00997, 0.01016] | 0.005 | PASS |
| M2 per-read junction precision (higher better) | 0.9522 | 0.9497 | 0.00246 | [0.00242, 0.00251] | 0.005 | PASS |
| M3 share of UMIs on the true gene (higher better) | 0.9959 | 0.9868 | 0.00915 | [0.00907, 0.00924] | 0.001 | PASS |
| M3 share of UMIs on a wrong gene (lower better) | 0.0004 | 0.0004 | -0.00000 | [-0.00001, 0.00000] | 0.001 | PASS |
| M4 spliced/unspliced/ambiguous accuracy (higher better) | 0.9847 | 0.9847 | 0.00001 | [-0.00000, 0.00002] | 0.005 | PASS |
| M5 expected calibration error (lower better) | 0.0150 | 0.0164 | -0.00138 | [-0.00140, -0.00136] | 0.01 | PASS |
| M1 registered headline: correct share at <= 0.5% wrong (STAR cannot reach it) (reported) (higher better) | 0.0071 | 0.0071 | 0.00000 | [-0.00000, 0.00000] | 0.001 | reported |
| M1 unmapped share (default operating point) (reported) (lower better) | 0.0051 | 0.0148 | -0.00965 | [-0.00978, -0.00951] | 0.001 | reported |
| M2 per-junction recall (bootstrap biased) (reported) (higher better) | 0.8774 | 0.8715 | 0.00588 | [0.00528, 0.00587] | 0.005 | reported |
| M2 per-junction precision (bootstrap biased) (reported) (higher better) | 0.8820 | 0.7760 | 0.10599 | [0.08774, 0.09037] | 0.005 | reported |
| M3 share of UMIs lost (reported) (lower better) | 0.0037 | 0.0128 | -0.00915 | [-0.00924, -0.00906] | 0.001 | reported |
| M3 per-cell count error (reported) (lower better) | 0.0045 | 0.0137 | -0.00920 | [-0.00929, -0.00910] | 0.001 | reported |
| M5 multimapper set recall (reported) (higher better) | 0.9759 | 0.9703 | 0.00556 | [0.00542, 0.00572] | 0.005 | reported |

## PBMC-E pbmc2 (T1-PB) (pbmc2): PASS
Coverage guard: ok: unmapped reads declared omitted: 7617774 missing reads counted as unmapped; names match (probe hit rate 0.9571). Records 229956098 of 237573872 input reads; missing 7617774.
Baseline `star_prod_campaign@1` row `star_prod`; 364509 bootstrap units, 1000 replicates; score JSON identical to the baseline's: False.

| measure | candidate | baseline | difference | 95% CI | margin | verdict |
|---|---:|---:|---:|---|---:|---|
| M1 locus-correct share (default operating point) (higher better) | 0.9463 | 0.9315 | 0.01478 | [0.01464, 0.01493] | 0.001 | PASS |
| M1 wrong share (default operating point) (lower better) | 0.0475 | 0.0466 | 0.00090 | [0.00086, 0.00094] | 0.001 | PASS |
| M2 per-read junction recall (higher better) | 0.9357 | 0.9194 | 0.01627 | [0.01611, 0.01643] | 0.005 | PASS |
| M2 per-read junction precision (higher better) | 0.9333 | 0.9313 | 0.00195 | [0.00188, 0.00203] | 0.005 | PASS |
| M3 share of UMIs on the true gene (higher better) | 0.9932 | 0.9810 | 0.01222 | [0.01211, 0.01234] | 0.001 | PASS |
| M3 share of UMIs on a wrong gene (lower better) | 0.0004 | 0.0004 | -0.00001 | [-0.00001, 0.00000] | 0.001 | PASS |
| M4 spliced/unspliced/ambiguous accuracy (higher better) | 0.9853 | 0.9854 | -0.00004 | [-0.00006, -0.00002] | 0.005 | PASS |
| M5 expected calibration error (lower better) | 0.0192 | 0.0206 | -0.00142 | [-0.00146, -0.00138] | 0.01 | PASS |
| M1 registered headline: correct share at <= 0.5% wrong (STAR cannot reach it) (reported) (higher better) | 0.0072 | 0.0072 | 0.00000 | [-0.00000, 0.00000] | 0.001 | reported |
| M1 unmapped share (default operating point) (reported) (lower better) | 0.0062 | 0.0219 | -0.01568 | [-0.01583, -0.01554] | 0.001 | reported |
| M2 per-junction recall (bootstrap biased) (reported) (higher better) | 0.9192 | 0.9127 | 0.00649 | [0.00599, 0.00683] | 0.005 | reported |
| M2 per-junction precision (bootstrap biased) (reported) (higher better) | 0.9107 | 0.8282 | 0.08247 | [0.06783, 0.07037] | 0.005 | reported |
| M3 share of UMIs lost (reported) (lower better) | 0.0064 | 0.0186 | -0.01222 | [-0.01233, -0.01210] | 0.001 | reported |
| M3 per-cell count error (reported) (lower better) | 0.0067 | 0.0182 | -0.01151 | [-0.01219, -0.01081] | 0.001 | reported |
| M5 multimapper set recall (reported) (higher better) | 0.9728 | 0.9620 | 0.01080 | [0.01050, 0.01107] | 0.005 | reported |

## WTC11 (wtc11): PASS
Coverage guard: ok: unmapped reads declared omitted: 12565692 missing reads counted as unmapped; names match (probe hit rate 0.7416). Records 33218460 of 45784152 input reads; missing 12565692.
Baseline `star_prod_campaign@1` row `star_prod`; 1000 bootstrap units, 1000 replicates; score JSON identical to the baseline's: False.

| measure | candidate | baseline | difference | 95% CI | margin | verdict |
|---|---:|---:|---:|---|---:|---|
| SIRV junction recall (>= 3 reads) (higher better) | 1.0000 | 1.0000 | 0.00000 | [0.00000, 0.00000] | 0.005 | PASS |
| SIRV junction precision (>= 3 reads) (higher better) | 1.0000 | 0.9496 | 0.05042 | [0.07014, 0.15419] | 0.005 | PASS |

## MIX-1k human cells (mix1k_human): PASS
Coverage guard: ok: unmapped reads declared omitted: 0 missing reads counted as unmapped; names match (probe hit rate 1.0000). Records 33844547 of 33844547 input reads; missing 0.
Baseline `star_prod_campaign@1` row `star_prod`; 421 bootstrap units, 1000 replicates; score JSON identical to the baseline's: False.

| measure | candidate | baseline | difference | 95% CI | margin | verdict |
|---|---:|---:|---:|---|---:|---|
| M7 foreign reads placed uniquely (raw share, D-53) (lower better) | 0.0609 | 0.1451 | -0.08412 | [-0.08560, -0.08252] | 0.0005 | PASS |
| M7 minus the empty-droplet fraction (not gating, D-53) (reported) (lower better) | -0.2530 | -0.1689 | -0.08412 | [-0.08560, -0.08252] | 0.0005 | reported |

## MIX-1k mouse cells (mix1k_mouse): PASS
Coverage guard: ok: unmapped reads declared omitted: 0 missing reads counted as unmapped; names match (probe hit rate 1.0000). Records 18739907 of 18739907 input reads; missing 0.
Baseline `star_prod_campaign@1` row `star_prod`; 458 bootstrap units, 1000 replicates; score JSON identical to the baseline's: False.

| measure | candidate | baseline | difference | 95% CI | margin | verdict |
|---|---:|---:|---:|---|---:|---|
| M7 foreign reads placed uniquely (raw share, D-53) (lower better) | 0.0750 | 0.1877 | -0.11273 | [-0.11430, -0.11108] | 0.0005 | PASS |
| M7 minus the empty-droplet fraction (not gating, D-53) (reported) (lower better) | -0.6111 | -0.4983 | -0.11273 | [-0.11430, -0.11108] | 0.0005 | reported |

## CHM13 (chm13): PASS
Coverage guard: ok: unmapped reads declared omitted: 6643606 missing reads counted as unmapped; names match (probe hit rate 0.9220). Records 77677525 of 84321131 input reads; missing 6643606.
Baseline `star_prod_campaign@1` row `star_prod`; 1000 bootstrap units, 1000 replicates; score JSON identical to the baseline's: False.

| measure | candidate | baseline | difference | 95% CI | margin | verdict |
|---|---:|---:|---:|---|---:|---|
| M6 locus-correct rate lost on reads over a CHM13-vs-GRCh38 difference (ref minus alt) (lower better) | 0.0208 | 0.0212 | -0.00039 | [-0.00043, -0.00035] | 0.01 | PASS |
| M1 locus-correct share (reported) (higher better) | 0.9958 | 0.9962 | -0.00035 | [-0.00035, -0.00034] | 0.001 | reported |
| M1 wrong share (reported) (lower better) | 0.0029 | 0.0029 | 0.00003 | [0.00003, 0.00004] | 0.001 | reported |
| locus-correct rate, reads over a difference (reported) (higher better) | 0.9765 | 0.9765 | 0.00002 | [-0.00002, 0.00006] | 0.01 | reported |
| locus-correct rate, reads over none (reported) (higher better) | 0.9973 | 0.9976 | -0.00037 | [-0.00038, -0.00037] | 0.01 | reported |
