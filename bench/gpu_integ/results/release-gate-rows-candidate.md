# Release-gate rows on the candidate (c8849bb2) against the gate-4 pin (18eb391c)

Requested by the Beta orchestrator: "no regressions against our gold standards". Inputs: the gate-4 result documents in `$W/release-gate4/results/` (pin) and the documents of the candidate in `$W/release-gate-candidate-cd6a630e/results/`. All candidate outputs are in that new directory (script set: `release-gate-candidate-cd6a630e/jobs/`, audit in `AUDIT.md`). Nothing shared was written. The candidate CPU run uses the candidate binary without `--engine gpu`. The pin column is the published gate-4 document. Its provenance may mix commits (the shared `pipeline-benchmark/runs/<ds>/singlet-public` holds d9834b8a outputs, and gate 4's actual 18eb391c runs sit under `release-gate4/gate2`). The candidate's runs were compared with those 18eb391c runs file by file (job 802128).

| Row | Pin pass/fail | Candidate pass/fail | Moved | Cause |
|---|---|---|---|---|
| gex-counts | 41/3 | 41/3 | no | all 44 metric entries identical, 0 differing fields |
| spliced-unspliced (agreement metrics) | 31/11, 7 na | 31/11, 7 na | no | 0 differing fields. The amended long-read metric (PBMC-E, MuSC) is not re-measured: no candidate long-read scores exist |
| junctions-psi | 42/0, 21 na | 42/0, 21 na | no | 0 differing fields |
| cell-calling | 21/4, 11 na | 21/4, 11 na | no | 0 differing fields |
| hto | 1/7 | 1/7 | no | 0 differing fields. HTO scoring runs Python code from the repo and does not use the GPU engine |
| non-host | 8/2, 1 na | 8/2, 1 na | no | 0 differing fields |
| **ambient-rna** | 8/2, 2 na | **9/1, 2 na** | **yes: bias_abs_log2 on hgmm1k_v3 fail to pass** | five ambient commits on `prod/campaign` after the pin (79cd1441, a30df331, b31706a7, 2330b901, 894c0b18, S12's fix, merged as mc/m176) change `ambient_correction.h`. The candidate CPU engine is IDENTICAL to the `prod/campaign` tip 0a6de829 on 10 of 10 samples tested (job 801026), so the cause is campaign code and not the GPU line |

Ambient metrics that moved (pin to candidate):

| Metric | Dataset | Pin | Candidate |
|---|---|---|---|
| bias_abs_log2 | hgmm1k_v3 | 0.4597 fail | 0.2083 pass |
| bias_low_depth_abs_log2 | hgmm1k_v3 | 0.8447 fail | 0.4451 fail (better, still failing) |
| rho_spearman | hgmm1k_v3 | 0.8538 pass | 0.8359 pass |
| own_removal_gap | hgmm1k_v3 | 0.1607 pass | 0.0617 pass |
| cross_species_residual | hgmm1k_v3 | 0.1490 pass | 0.1503 pass |
| bias_abs_log2 | hgmm1k_v2 | 0.0988 pass | 0.0149 pass |
| bias_low_depth_abs_log2 | hgmm1k_v2 | 0.3811 pass | 0.2237 pass |

The competitor values (SoupX, DecontX) are unchanged. The tolerances moved slightly because they scale with the Singlet values.

**File-by-file comparison of the candidate CPU runs with the pin runs (job 802128, 8 datasets):** every count matrix, barcode and feature file, cell call and the other outputs are the same (88 to 140 files per dataset). The files that differ are logs, timings, `provenance.json`, `summary.json`, `pileup_stats.json` (the run-dependent files of the gate) and the three ambient outputs (`ambient_contamination.json`, `.tsv`, `ambient_corrected_counts.mtx.gz`). That matches the ambient commits above.

**CPU engine unchanged against `prod/campaign` 0a6de829 (job 801026):** 10 of 10 samples identical so far, including the ambient outputs (the 11th sample, pbmc2tvdj, is still running).

Not covered: the other 15 rows of the 22-capability gate (carried forward in gate 4), the GPU-versus-CPU diff of the candidate release-gate runs (jobs 801885 and 801888 still running), and the spliced-unspliced amended contract.
