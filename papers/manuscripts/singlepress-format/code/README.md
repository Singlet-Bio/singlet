# SinglePress — Reproducible Figure Code

This directory contains all scripts and pre-computed data needed to reproduce
the figures in:

> **SinglePress: a high-performance sparse matrix format for single-cell genomics**
> Zach DeBruine. *Genome Biology* (2025).

## Quick start (figures only)

Regenerate all manuscript figures from pre-computed benchmark data:

```bash
# Requirements: R ≥ 4.4 with ggplot2, dplyr, tidyr, scales, patchwork
cd code
make figures
```

Output PDFs appear in `figures/`:

| PDF | Manuscript | Content |
|-----|-----------|---------|
| `fig1_compression.pdf` | Figure 1 | File format comparison (6 panels) |
| `fig2_performance.pdf` | Figure 2 | Performance benchmarks (6 panels) |
| `fig3_io.pdf` | Figure 3 | I/O throughput (4 panels) |
| `fig4_frontier.pdf` | Figure 4 | Compression frontier analysis (6 panels) |
| `figS1_structure.pdf` | Figure S1 | Statistical structure (6 panels) |

## Directory structure

```
code/
├── README.md                   ← this file
├── Makefile                    ← build all figures with `make figures`
├── data/                       ← pre-computed benchmark results (764 KB)
│   ├── all_datasets_survey.csv        3,253 datasets × format metrics
│   ├── benchmark_results_v3.json      detailed per-format benchmarks (19 datasets)
│   ├── bpcells_compression_bench.csv  BPCells compression benchmarks
│   ├── compression_frontier.csv       entropy, zstd sweep, alt codecs
│   ├── format_pytorch_bench.csv       PyTorch dataloader (fallback)
│   ├── gpu_benchmark_v2.csv           H100 GPU I/O vs compute
│   ├── io_benchmarks.csv              read/write throughput by format
│   ├── operations_benchmark.csv       column subsetting benchmarks
│   ├── r_format_benchmarks.csv        R ecosystem format comparison
│   ├── r_write_benchmarks.csv         R write throughput (RDS vs .1pz)
│   ├── read_throughput.csv            read throughput vs dataset size
│   ├── threading_benchmark_v2.csv     multi-threaded read scaling
│   ├── threading_benchmarks.csv       threading (fallback)
│   ├── value_distributions.csv        per-dataset value histograms
│   ├── write_benchmarks.csv           write throughput by format
│   └── zinb_data.csv                  entropy & ZINB fit statistics
├── figures/
│   ├── figures_main.R          ← Figures 1–3 and S1
│   └── figures_frontier.R      ← Figure 4
└── benchmarks/                 ← scripts to regenerate data/ from raw .1pz files
    ├── survey_all_datasets.py
    ├── benchmarks_v3.py
    ├── extract_zinb_data.py
    ├── extract_write_benchmarks.py
    ├── fast_bench.py
    ├── benchmark_threading_v2.py
    ├── benchmark_operations.py
    ├── benchmark_gpu_v2.py
    ├── benchmark_write_rds.py
    ├── benchmark_r_formats_v2.R
    ├── benchmark_bpcells_v3.R
    └── benchmark_compression_frontier.py
```

## R dependencies

```r
install.packages(c("ggplot2", "dplyr", "tidyr", "scales", "patchwork"))
```

R ≥ 4.4 is required. Figures were generated with R 4.5.2.

## Full reproduction (benchmarks from raw data)

The benchmark scripts in `benchmarks/` regenerate every CSV/JSON file in `data/`
from a corpus of 3,253 `.1pz` files. This requires:

- **Hardware**: HPC cluster with SLURM (CPU nodes: ≥64 GB RAM, ≥8 cores;
  GPU node: NVIDIA H100; threading benchmark: ≥32 cores)
- **Data**: the `.1pz` dataset corpus at a path like
  `/mnt/projects/.../pipeline/quant/<GSE_ID>/counts.1pz`
  (paths are hardcoded in each script — edit `BASE_DIR` / equivalent)
- **Python ≥ 3.10** with: `numpy`, `scipy`, `pandas`, `anndata`, `h5py`,
  `singlepress`, `scanpy`, `torch` (GPU only), `zstandard`, `lz4`, `brotli`
- **R ≥ 4.4** with: `Matrix`, `BPCells`, `HDF5Array`, `reticulate`

### Execution order

Benchmarks have a dependency graph. Run in this order:

```
Step  Script                            Output                    Notes
────  ─────────────────────────────────  ────────────────────────  ─────────────────────────────
 1    survey_all_datasets.py            all_datasets_survey.csv   ~2 h, CPU, 64 GB
                                        io_benchmarks.csv
                                        threading_benchmarks.csv
 2    benchmarks_v3.py                  benchmark_results_v3.json ~2 h, CPU, 64 GB, needs R
 3    extract_write_benchmarks.py       write_benchmarks.csv      seconds (extracts from step 2)
 4    extract_zinb_data.py              zinb_data.csv             minutes (needs step 2)
                                        value_distributions.csv
 5    fast_bench.py                     read_throughput.csv        ~30 min (needs step 1)
 6    benchmark_threading_v2.py         threading_benchmark_v2.csv ~1 h, 32 cores
 7    benchmark_operations.py           operations_benchmark.csv   ~1 h, CPU, 64 GB
 8    benchmark_write_rds.py            r_write_benchmarks.csv     ~30 min, needs R
 9    benchmark_r_formats_v2.R          r_format_benchmarks.csv    ~1 h, R + reticulate
10    benchmark_bpcells_v3.R            bpcells_compression_bench.csv  ~1 h, R + BPCells
11    benchmark_compression_frontier.py compression_frontier.csv   ~2 h, 64 GB
```

Steps 5–11 can run in parallel (only steps 3–5 depend on earlier outputs).

### Example SLURM invocations

```bash
# CPU benchmark (steps 1, 2, 5–8, 11)
srun --time=120 --mem=64G --cpus-per-task=8 python benchmarks/survey_all_datasets.py

# Threading benchmark (step 6)
srun --time=120 --mem=32G --cpus-per-task=32 python benchmarks/benchmark_threading_v2.py

# GPU benchmark (step 8)
sbatch --partition=gpu --gres=gpu:1 --constraint=h100 \
       --time=360 --mem=128G --cpus-per-task=8 \
       --wrap="python benchmarks/benchmark_gpu_v2.py"

# R benchmarks (steps 9–10)
srun --time=120 --mem=64G --cpus-per-task=8 Rscript benchmarks/benchmark_r_formats_v2.R
```

## Figure–panel data mapping

| Figure | Panel | Data file(s) | Key metric |
|--------|-------|-------------|------------|
| Fig 1 | a | io_benchmarks, r_format_benchmarks, bpcells_compression_bench | Compression ratio by format |
| Fig 1 | b | io_benchmarks, r_format_benchmarks, bpcells_compression_bench | Read throughput by format |
| Fig 1 | c | io_benchmarks | .1pz vs H5AD file size |
| Fig 1 | d | all_datasets_survey | Compression by species |
| Fig 1 | e | all_datasets_survey | Compression by protocol |
| Fig 1 | f | value_distributions | Count value distribution |
| Fig 2 | a | read_throughput | Read throughput vs dataset size |
| Fig 2 | b | threading_benchmark_v2 | Multi-threaded scaling |
| Fig 2 | c,d | operations_benchmark | Column subsetting |
| Fig 2 | e,f | gpu_benchmark_v2 | GPU compute fraction |
| Fig 3 | a,b | write_benchmarks, io_benchmarks | Python write/read |
| Fig 3 | c,d | r_write_benchmarks, r_format_benchmarks | R write/read |
| Fig 4 | a | compression_frontier | .1pz vs entropy bound |
| Fig 4 | b–d | compression_frontier | zstd level sweep |
| Fig 4 | e | compression_frontier | Alternative codec Pareto |
| Fig 4 | f | compression_frontier | VOCSC encoding advantage |
| Fig S1 | a | value_distributions | Per-dataset spaghetti |
| Fig S1 | b | zinb_data | Entropy vs fraction of ones |
| Fig S1 | c | all_datasets_survey | Bits/nz vs compression ratio |
| Fig S1 | d | all_datasets_survey | Size invariance |
| Fig S1 | e | io_benchmarks, r_format_benchmarks, bpcells | R ecosystem |
| Fig S1 | f | all_datasets_survey | Predicted vs observed size |

## Software versions

- **singlepress** 0.3.x (Python), 0.1.x (R via reticulate)
- **R** 4.5.2
- **Python** 3.12
- **zstd** 1.5.6 (via python-zstandard)
- **CUDA** 12.8.1 / PyTorch 2.6 (GPU benchmarks only)

## License

This code is released under the MIT License. See the repository root for details.
