# .s1 format paper

Manuscript and benchmark for the `.s1` count-matrix format. Every figure, table and number in the paper is produced
by one pipeline from public datasets; nothing in the `.tex` files is typed by hand.

The benchmark ran on Grand Valley State University's Clipper cluster in October and November 2026: 27 datasets and 29
matrices, plus two out-of-domain extensions, timed on 8 cores of shared bigmem nodes (Xeon Gold 6226, spinning RAID).
`generated/` holds the final numbers and tables, and `results/` the per-stage results behind them, with host records in
`results/env/` and the environment locks. `BENCHMARK_NOTES.md` explains how the comparisons are kept fair; the
manuscript's methods give the full description.

## Layout

| Path | What it is |
| --- | --- |
| `BENCHMARK_NOTES.md` | How the benchmark keeps comparisons fair: timing, loaders, network tests, records. Read first. |
| `PLAN.md` | The experiment plan written before the run; the methods describe what was run. |
| `datasets.yaml` | 29 public datasets (31 count matrices) plus 2 out-of-domain count matrices: source, accession, file, format, licence. |
| `bench/` | The pipeline, `python -m s1bench` (Python package `s1bench/`, R scripts `R/`, C++ decode benchmark `cpp/`). |
| `bench/config.yaml` | All settings: formats, threads, repeats, floor models, minibatch workload, network tiers. |
| `bench/environment.yml` | Conda environment (Python 3.11, R 4.4, all format libraries). |
| `bench/slurm/` | `array.sbatch` (one dataset per task) and `serial.sbatch` (minibatch, ecosystem, report). |
| `bench/tests/` | `smoke.py` (whole pipeline on two synthetic matrices) and `test_floor.py` (count models). |
| `results/` | Raw per-stage TSVs written by the pipeline (committed after the run). |
| `generated/` | Tables (`*.tsv`) and `numbers.tex` written by `report`; read by the manuscript. |
| `main.tex`, `sections/`, `figures/`, `refs.bib` | The manuscript. Figures are pgfplots drawing from `generated/*.tsv`. |

## Running

```bash
conda env create -f bench/environment.yml && conda activate s1bench
pip install ../../..                     # the singlet package from this commit (builds singlet._pz2)
R CMD INSTALL ../../../r                 # the R package
bash bench/cpp/build.sh                  # C++ decode benchmark

cd bench
python -m pytest tests/test_floor.py     # count-model unit tests
python tests/smoke.py                    # whole pipeline on synthetic data, in a temporary directory, about 5 min

python -m s1bench env
python -m s1bench all                    # or stage by stage: fetch canon write decode ladder floor second minibatch eco report
```

On a cluster, from `bench/`:

```bash
N=$(python -c "import yaml; print(len(yaml.safe_load(open('../datasets.yaml'))['datasets']))")
JOB=$(sbatch --parsable --array=0-$((N-1)) slurm/array.sbatch)
sbatch --dependency=afterok:$JOB slurm/serial.sbatch
```

Useful options: `--datasets pbmc3k,hlca_core` or `--datasets tier:S,M`, `--force` to redo finished work,
`--set key.sub=value` to override `config.yaml`, and `S1BENCH_DATA=/scratch/...` for the download directory
(0.3-0.5 TB). Every stage skips work whose results exist, so an interrupted run resumes where it stopped.

To use the in-tree package without installing it, set `PYTHONPATH=<repo>/python:<repo>/papers/manuscripts/s1-format/bench`
after `python setup.py build_ext --inplace` at the repository root.

## Ablation of the model

`python -m s1bench ablate` (C++ `bench/cpp/ablate_bench`, built by `bench/cpp/build.sh`) measures what each part of the codec-1 count
model contributes. Per matrix it prints the ideal static code length with one component added or removed (row order, cell-depth
context, position context, value contexts; `kind` `model`) and runs the shipped encoder with one option changed (context
configuration, class resolution, lane count, the generic zstd codec; `kind` `real`), with true payload bytes, the one-thread decode
rate and an exact round-trip check. The full model's estimate is compared with the real payload (`model_over_real` in the table).
`report` writes `generated/tab_ablation.tsv`; per-matrix results are in `results/ablate/`. Matrices above `ablate.max_nnz`
non-zeros (default `kernels.max_nnz`) are left out; the encoder takes at most 2^32 non-zeros. On Clipper, `slurm/clipper/ablate.sbatch`
runs one dataset per array task and `ablate_report.sbatch` summarizes.

## Building the PDF

```bash
python -m s1bench report    # from bench/; lists every \nm{key} the .tex files use that the results did not produce
cd .. && latexmk -pdf main.tex   # or `make s1-format` in papers/
```

`report` exits with status 1 while any key is missing.
