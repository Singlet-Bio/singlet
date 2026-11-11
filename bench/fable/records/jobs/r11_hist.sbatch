#!/bin/bash
#SBATCH --job-name=e6-r11hist
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E6: STAR's exact CB histogram of the sample as singlet's decode pass builds it for R11 (bench/fable/records/r11_hist.cpp), so that the
# driver's R11 CR table is the engine's. Output: $W/jobs/fable/records/results/r11_exact_hist.bin (barcode counts only).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; E6=$W/jobs/fable/records; ENV=$W/envs/pbench; BARE=$W/repos/singlet-py.git
FQ=${FQ:-$PB/data/pbmc1k_v3/pbmc1k_v3.1fq}; WL=${WL:-$PB/whitelists/3M-february-2018.txt}; OUT=${OUT:-$E6/results/r11_exact_hist.bin}
SCR=/tmp/$USER-e6h-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT INT TERM
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/records:refs/fable/records
SHA=$(git -C $BARE rev-parse --short=8 refs/fable/records); echo "commit $SHA"
git -C $BARE archive refs/fable/records include/singlet bench/fable/records | tar -x -C $SCR/src
mkdir -p $SCR/inc; ln -s $ENV/include/zstd.h $SCR/inc/zstd.h; [ -e $ENV/include/zstd_errors.h ] && ln -s $ENV/include/zstd_errors.h $SCR/inc/zstd_errors.h
$ENV/bin/x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -I$SCR/src/include -I$SCR/inc -o $SCR/r11_hist $SCR/src/bench/fable/records/r11_hist.cpp \
  -L$ENV/lib -Wl,-rpath,$ENV/lib -lzstd 2>&1 | head -40
[ -x $SCR/r11_hist ] || { echo "compile FAILED"; exit 1; }
T0=$(date +%s)
$SCR/r11_hist $FQ $WL $OUT.tmp && mv $OUT.tmp $OUT
echo "exit=$? in $(( $(date +%s) - T0 )) s"; ls -la $OUT
