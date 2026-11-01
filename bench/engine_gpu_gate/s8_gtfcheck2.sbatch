#!/bin/bash
#SBATCH --job-name=s8-gtfcheck2
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: the missing-GTF error of the build in BINDIR: no --exons and (D1) no --genome-dir, (D2) a --genome-dir with no genes/genes.gtf next to it.
W=/mnt/projects/debruinz_project/debruinz/singlet; BIN=${BINDIR:?}/singlet-gpu; FQ=$W/jobs/s9/fixtures/SRR19421372.1fq
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/ref/star_x $SCR/o1 $SCR/o2; trap 'rm -rf $SCR' EXIT
$BIN $FQ --out-prefix $SCR/o1/ --threads 2 2>&1 | grep -a -m3 "ERROR"; echo "D1 rc=${PIPESTATUS[0]}"
$BIN $FQ --genome-dir $SCR/ref/star_x --out-prefix $SCR/o2/ --threads 2 2>&1 | grep -a -m3 "ERROR"; echo "D2 rc=${PIPESTATUS[0]}"
