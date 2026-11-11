#!/bin/bash
#SBATCH --job-name=sexcalls-cellxy
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# cellXY (GPL-3.0, comparator only) on the named mtx dirs under work/mtx. Names come from the NAMES variable (default: dev pools).
W=/mnt/projects/debruinz_project/debruinz/singlet
B=$W/bench/calls-sota/sex
NAMES=${NAMES:-lcl_pool osorio}
T=$B/work/run_$SLURM_JOB_ID; mkdir -p $T/mtx $B/work/cellxy
for n in $NAMES; do ln -s $B/work/mtx/$n $T/mtx/$n; done
export R_LIBS=$W/envs/rlib-benchpublic
/usr/bin/time -v -o $B/work/cellxy.time.$SLURM_JOB_ID.txt $W/envs/ambient/bin/Rscript $B/cellxy.R $T
cp $T/cellxy/*.tsv $B/work/cellxy/
cat $T/cellxy_timing.jsonl >> $B/work/cellxy_timing.jsonl
