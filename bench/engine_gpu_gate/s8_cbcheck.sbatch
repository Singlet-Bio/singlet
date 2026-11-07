#!/bin/bash
#SBATCH --job-name=s8-cbcheck
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# S8: stream vs sorted pileup. The sorted path with SINGLET_TEST_CB_CHECK=1 compares the stream's CB (CR corrected by the R11 corrector) with STAR's CB tag per primary record.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?}; SAMPLES=(SRR19421372 SRR19573735 SRR30477995); s=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_TEST_CB_CHECK=1
G=$REF/GRCh38-2024-A; IN=$W/jobs/s9/fixtures/$s.1fg2
SCR=/tmp/$USER-s8cb-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'rm -rf $SCR' EXIT
cd $SCR && $BINDIR/singlet-gpu $IN --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads 16 --pipeline --output-format 1pz --sorted-bam 2>&1 | grep -a -E "cb-check|cb-correct|Done:|stream CB" | cut -c1-300
