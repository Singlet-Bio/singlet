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

## Files

- `run_singlet.sbatch`, `run_starsolo.sbatch`: one run per dataset on Clipper. STARsolo uses the flags of
  `bench/exact_vs_pseudo/run_dataset.sbatch`; `PROFILE=beta` adds Singlet's Beta aligner flags, which isolates counting.
- `diagnose.py`: classifies wrong-species and chrY molecules by read class from a `SINGLET_MOLECULE_DUMP` table.
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

The gate runs hgmm1k_v3 and pbmc1k_cite_v3 (about 15 minutes on 16 cores) against the frozen STARsolo outputs in
`$W/count-parity/starsolo-ref/`. `ENGINE=gpu` runs the GPU pileup instead; the CPU and GPU pileups are held
byte-identical by `tests/cpp/test_gpu_pileup_labels.cpp` and the PR-2 identity gate.
