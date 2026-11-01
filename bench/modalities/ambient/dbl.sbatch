#!/bin/bash
#SBATCH --job-name=amb-dbl
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Doublet detectors on the hgmm cells plus 44 synthetic human+mouse doublets; Singlet's detect_doublets() header compiled unchanged.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/ambient; S=$W/jobs/modalities/ambient; D=$W/modality-bench/ambient/work/dbl
export PATH=$E/bin:$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl NUMBA_CACHE_DIR=$W/cache/numba-edrop
python -I $S/dbl_synth.py
x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -I$S/inc $S/dbl_harness.cpp -o $D/dbl_harness
$D/dbl_harness $D/exon $D/singlet.tsv
python -I $S/dbl_tools.py scrublet
Rscript $S/dbl_scdbl.R
python -I $S/dbl_tools.py score
