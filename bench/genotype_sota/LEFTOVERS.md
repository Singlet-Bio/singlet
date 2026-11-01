# Leftovers for Thursday (S10 genotype-sota, 2026-11-01)

Not pushed: Zach has not approved pushing this folder, the staged bench/public/genotype files or the snp_concordance.py edit.

## Done
- Single-binary freeze run on 2a3b0d61 (gate binary singlet-gpu, sha d95c8d30...0622), 7 pools; contract JSONs + artifacts in the shared notes
  folder `genotype-freeze-2a3b0d61/` (donor-demux 15/15, donor-snps 22/25, mtdna 11/18). `_gcommon.py` edit = SINGLET_COMMIT 2a3b0d61c247aa8cc8de5102b16114dbf4c802c8,
  SINGLET_BINARY singlet-gpu, SINGLET_BINARY_SHA256 d95c8d30e7c179d115514730a74b7fdfbaaa49bcae70c6055bc18c7151c40622 (file lives on origin/bench/public-genotype; copy in the notes folder).
- mtDNA ctrl/stim is not a regression (same numbers as earlier runs; Singlet leads recall/F1, trails precision).
- donor-snps 3 fails sit just outside the paired-bootstrap tolerance; stay fails. Add the Kang-truth-bias note to the contract.
## Running
- Clipper job 796940 (gsota-selfrep): selfreport ancestry on atlas donors with the tip's label rule -> /mnt/projects/debruinz_project/debruinz/singlet/release-gate3/bpub/artifacts/genetic-ancestry/selfreport_calls.tsv (+ work/ancestry/selfreport/anc.jsonl). Script: selfrep_gate.sbatch.
## Open
1. Tell S5 (local_02864135) when selfreport_calls.tsv exists. lcl_ancestry_calls.tsv still needs Singlet runs of osorio2019_lcl_mix and lcl_gse230239 on the gate binary (release-gate3/bpub/runs is empty), then lcl_score.py.
2. Ancestry blind check on S15's pilot samples (GEO-recorded ancestry).
3. Multi-donor fraction of campaign samples.
4. docs/research-beta/sota-genotype.md (not written).
5. Ask Zach to approve pushing bench/genotype_sota/ + _gcommon.py edit as a bench-only draft PR to prod/campaign (no genotype data or results).
6. Drafts with S1: #142 (nonhost_confirmed_reads), #154 (docs).
