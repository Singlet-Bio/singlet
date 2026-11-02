# .s1 format paper

Manuscript and benchmark for the `.s1` count-matrix format. Every figure, table and number in the paper is produced
by one pipeline from public datasets; nothing in the `.tex` files is typed by hand.

**Status:** the benchmark has run on Clipper (GVSU), 2026-10-29/30: 27 datasets, 29 matrices, plus two out-of-domain
extensions, timed on 8 cores of shared bigmem nodes (Xeon Gold 6226, spinning RAID). `generated/` holds the final
numbers and tables, `results/` the per-stage results behind them (with host records in `results/env/` and the
environment locks), and the manuscript builds with no missing values. `PLAN.md` is the original work order; where the
run departed from it (CPU nodes instead of an H100 host, the trimmed timing protocol, the bandwidth proxy), the methods
section and `REVIEW_RESPONSE.md` describe what was done.

## Layout

| Path | What it is |
| --- | --- |
| `PLAN.md` | Questions, experiments, stages, hardware, compute estimate, codebase gaps (G1-G4), done criteria. Read first. |
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

## Building the PDF

```bash
python -m s1bench report    # from bench/; lists every \nm{key} the .tex files use that the results did not produce
cd .. && latexmk -pdf main.tex   # or `make s1-format` in papers/
```

`report` exits with status 1 while any key is missing. After the run, check each sentence of `sections/results.tex`
against the generated numbers (comments marked `CHECK` name the comparison a sentence depends on) and rewrite any the
data do not support.
