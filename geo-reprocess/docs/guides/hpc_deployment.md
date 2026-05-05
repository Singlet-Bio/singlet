# HPC Deployment

## Overview

geo-reprocess is designed for HPC clusters with SLURM. This guide covers deployment on university and ACCESS-allocated systems.

## Cluster requirements

- **Scheduler**: SLURM
- **Memory**: 96 GB+ per node (recommended 192 GB+; P95 jobs need 373 GB)
- **CPUs**: 36+ cores per node
- **Storage**: 10+ TB scratch space for FASTQs and intermediate files
- **Network**: High-bandwidth for FASTQ downloads (ENA/SRA)

## Memory tiers

Based on analysis of 516 completed jobs:

| Tier | MaxRSS | Coverage | Recommended Node |
|------|--------|----------|-----------------|
| Standard | ≤192 GB | ~50% | 192 GB nodes |
| Large | ≤373 GB | ~95% | 384 GB nodes (e.g., TAMU Launch) |
| XL | >373 GB | 100% | 512+ GB nodes |

## ACCESS allocation strategy

For processing ~206K human RNA samples:

- **System**: TAMU Launch (371 GB/node, covers 95% of samples)
- **Credits**: ~88.7K ACCESS credits (Option B optimization)
- **Timeline**: ~14 days with 75 concurrent tasks
- **Cost**: University HPC is 60-100× cheaper than cloud

### Optimization strategies

- Use `kraken2 --memory-mapping` for smaller jobs (~80 GB savings)
- Batch by estimated memory requirements
- Retry failed high-memory jobs on larger nodes

## Environment setup

```bash
module load python/3.11
python -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install sc-geo
```

## Production submission pattern

```bash
# Never run on login node — submit via SLURM
sc-geo batch submit --catalog filtered.parquet \
    --partition general --cpus 36 --memory 192G --time 4:00:00
```
