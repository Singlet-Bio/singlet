#!/bin/bash
#SBATCH --job-name=lr-alevin
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:59:00
# simpleaf (piscem + alevin-fry, USA mode: index af_idx built with --usa, cr-like resolution) on full pbmc1; chemistry 16 nt CB + 11 nt UMI.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/expression-sota/longread_usa/code/common.sh
export ALEVIN_FRY_HOME=$PB/af_home PATH=$W/envs/pbench/bin:$PATH
IDX=$PB/refs/GRCh38-2024-A/af_idx
O=$X/alevinfry; rm -rf $O; mkdir -p $O
/usr/bin/time -v -o $O/time.txt simpleaf quant --reads1 $R1 --reads2 $R2 --index $IDX/index --t2g-map $IDX/index/t2g_3col.tsv   --chemistry '1{b[16]u[11]x:}2{r:}' --expected-ori fw --resolution cr-like --unfiltered-pl $WL --threads 16 --output $O/af > $O/run.log 2>&1
grep -E "Elapsed \(wall|Maximum resident" $O/time.txt; ls $O/af/af_quant/alevin
