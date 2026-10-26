# ambient-rna and doublets: status at wind-down (2026-10-26)

**ambient-rna**: `results/ambient-rna.json`, status `draft`, validates. Libraries: hgmm1k_v3 (development), hgmm1k_v2 (held out). 8 of 10 gating metrics
pass. Failing, on held-out hgmm1k_v2: mean-removal bias (Singlet 0.20 vs CellBender 0.03) and own-species removal gap (0.22 vs DecontX-background 0.06).
Missing: hgmm10k_v3 (download submitted as Clipper job 786335, Singlet run never submitted), and the Kang real-tissue truth metrics (code is in
`harness/kang_ambient*.{py,sbatch}`; the tool runs 786367 and CellBender 786374/786375 were submitted, artifacts were not built).

**doublets**: no results file. The hgmm 1k libraries have fewer than 30 truth doublets, so nothing is gated and the contract rejects a
capability with no gating metric on real data. Singlet runs are complete for all Kang lanes (`$WORK/bench-public/runs/kang2018_{ctrl,b1_a,b1_b,b1_c}/singlet`);
Scrublet, scDblFinder, cellsnp-lite and vireo tools jobs were submitted for them (786074 done; 786368-786370 queued or running). Build artifacts with
`harness/artifacts.sbatch kang ID TRUTH LANE_FILTER`, then run `capabilities/doublets.py`.

Not included: the PR #43 doublet combiner (not in the 1.0.0 binary; its training set contains the Kang lanes) and hgmm5k_v3 (not in the cohort).

Facts: memory is about 59 GB for joint human+mouse runs regardless of read count and 27-31 GB for Kang lanes (not 600 GB). Auto-K picked 9 on kang2018_ctrl
(an 80-cell fragment of truth donor 1015; ARI 0.976), and 4, 4, 8 on the batch 1 lanes (ARI on lane A 0.995).
Artifacts: `$WORK/bench-public/artifacts/{ambient-rna,doublets}/`. Run dirs: `$WORK/bench-public/runs/<id>/singlet`, work dirs `$WORK/bench-public/work/<id>`.
