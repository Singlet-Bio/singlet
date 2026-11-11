# Counting parity with STARsolo

Singlet aligns with an exact port of STAR 2.7.11b, so any accuracy gap to STARsolo on the same reads comes from the
counting rules in the pileup or from the alignment flags. This folder measures that gap on real truth, holds the
diagnosis tools, and keeps a standing gate so that every change to the counting code is scored against STARsolo.

## Truth and axes

| Axis | Truth | Singlet layer | STARsolo layer | Better |
|---|---|---|---|---|
| `barnyard.X` | HEK293T + NIH3T3 barnyard: a pure cell holds one species | raw X (exon + intron) | GeneFull | lower |
| `barnyard.exon` | same | exon layer (spliced + ambiguous) | Gene | lower |
| `barnyard.intron` | same | intron layer (unspliced) | Velocyto unspliced | lower |
| `chrY` | HEK293T is female: non-PAR chrY UMIs in pure human cells are false | raw X | GeneFull | lower |
| `cite` | CITE-seq protein measured on the same cells | raw X | GeneFull | higher |

Cells are the published Cell Ranger filtered barcodes, and purity comes from Cell Ranger's own species fraction, as in
`bench/exact_vs_pseudo/score.py`. Every interval is a paired bootstrap over cells (the same resampled cells for both
tools). An axis fails when the whole interval of Singlet minus STARsolo lies on the worse side of zero.

## What closed the gap (November 2026)

Three causes made Singlet's barnyard and chrY counts worse than STARsolo's. The first was counting: a multimapper whose other
alignment lay in another gene's intron was still counted, and an exon of a gene nested in another gene's intron took the read.
`intron_rule::kGeneFullSet` (512) adopts GeneFull's gene set and removes both. The second was alignment. Singlet's STAR ran
with `--outFilterScoreMin 10`, so after CellRanger4 clipping short cDNA reads kept 10 to 30 point alignments, and those carry
wrong-species reads. STARsolo run at 10 (`run_starsolo.sbatch PROFILE=beta10`) rose to Singlet's wrong-species fraction, with
the same number of wrong molecules. The Beta aligner profile now uses Cell Ranger's 30. The third showed only on hgmm5k_v3,
where Singlet counted fewer wrong-species molecules than STARsolo but 0.8% fewer molecules overall, which left its fraction just
above. `intron_rule::kUmiGeneSpliced` (2048) closes it: a UMI seen in two genes of a cell keeps its winner only when the winner
is spliced (`sim_rules.py` predicted the effect on a molecule dump before the change). The default rules were 2957 = 397 + 512 +
2048 at this commit; the release default is now 61837 (0xF18D) = 397 + 4096 + 8192 + 16384 + 32768, which dropped 512 and 2048. Dropping kUnique (908, or 1932) adds wrong intronic multimappers, so it stays on.

| hgmm1k_v3 | rules 397, score 10 | 909, score 10 | 909, score 30 | STARsolo 2.7.11b |
|---|---|---|---|---|
| X wrong-species | 0.367% | 0.347% | 0.322% | 0.328% |
| exon | 0.387% | 0.361% | 0.337% | 0.358% |
| intron | 0.293% | 0.294% | 0.267% | 0.271% |
| chrY per million | 8.2 | 2.2 | 2.0 | 5.2 |

Default 2957 with score 30 against STARsolo (commit d25045fa; Singlet / STARsolo, paired 95% CI of the difference in
`gate_results/` for the gated datasets):

| dataset | X | exon | intron | chrY per million | CITE Spearman |
|---|---|---|---|---|---|
| hgmm1k_v3 | 0.321% / 0.328% | 0.336% / 0.358% | 0.265% / 0.271% | 2.0 / 5.2 | |
| hgmm1k_v2 | 0.800% / 0.843% | 0.739% / 0.754% | 1.253% / 1.249% (within CI) | 3.1 / 6.3 | |
| hgmm5k_v3 | 0.9443% / 0.9474% | 1.066% / 1.091% | 0.508% / 0.516% | 2.5 / 5.5 | |
| pbmc1k_cite_v3 | | | | | 0.42755 / 0.42757 (within CI) |

## Files

- `run_singlet.sbatch`, `run_starsolo.sbatch`: one run per dataset on Clipper. STARsolo uses the flags of
  `bench/exact_vs_pseudo/run_dataset.sbatch`; `PROFILE=beta` adds Singlet's Beta aligner flags, which isolates counting.
- `diagnose.py`: classifies wrong-species and chrY molecules by read class from a `SINGLET_MOLECULE_DUMP` table.
- `sim_rules.py`: replays multi-gene UMI rules on a molecule dump. `deficit.py`: correct-species UMIs per gene that STARsolo
  counts and Singlet does not. `compare_runs.sbatch`: byte identity of two runs' matrices (CPU vs GPU pileup).
- `parity_score.py`: all axes, paired CIs, per-gene excess over STARsolo, gate verdict (exit 1 on a failed axis).
- `gate.sbatch`, `gate_record.py`, `gate_check.py`, `counting_sources.txt`, `gate_results/`: the standing gate.

## Standing gate

`counting_sources.txt` lists the files that decide the count matrices. `gate_check.py hash` turns their content into a
source hash. The workflow `.github/workflows/count-parity.yml` runs `gate_check.py check` on every pull request and push
that touches them, and fails unless `gate_results/<hash16>.json` exists and records PASS on every axis.

To produce the record after changing a counting source:

```bash
# on Clipper, from $W = /mnt/projects/debruinz_project/debruinz/singlet
sbatch --export=ALL,BR=<branch>,NAME=<commit8>,TESTS=0 jobs/gpu-pr2/gd-build2.sbatch
sbatch --export=ALL,BIN=$W/jobs/gpu-pr2/bin2/<commit8>/singlet-gpu,BR=<branch> jobs/count-parity/gate.sbatch
# then copy $W/count-parity/gate/<hash16>.json to bench/count_parity/gate_results/ and commit it
```

The gate runs hgmm1k_v3, hgmm1k_v2 (10x v2 chemistry) and pbmc1k_cite_v3 (about 25 minutes on 16 cores) against the frozen STARsolo outputs in
`$W/count-parity/starsolo-ref/`. `ENGINE=gpu` runs the GPU pileup instead; the CPU and GPU pileups are held
byte-identical by `tests/cpp/test_gpu_pileup_labels.cpp` and the PR-2 identity gate.
