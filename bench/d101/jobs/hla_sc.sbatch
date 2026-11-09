#!/bin/bash
#SBATCH --job-name=d101-hla
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# HLA typing of one Oetjen sc library (row SLURM_ARRAY_TASK_ID of libs.tsv) with `singlet hla`, the same code path the pipeline runs
# (rows with a full pipeline run already have private/hla/ from run_sc.sbatch). Output: $W/jobs/d101/hla/<sc_run>/ (owner-only).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; ENV=$W/envs/pbench
LIBS=${LIBS:?}; BIN=${BIN:?}; T=${SLURM_CPUS_PER_TASK:-8}
read -r DONOR SC SCG BULK BULKG <<< "$(awk -v n=$SLURM_ARRAY_TASK_ID 'NR==n+1' $LIBS)"
FQ=$W/data/d101/sc/$SC.1fq; [ -s $FQ ] || { echo "no $FQ"; exit 3; }
O=$W/jobs/d101/hla/$SC; mkdir -p $O; chmod 700 $W/jobs/d101/hla $O
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
/usr/bin/time -v -o $O/time.txt $BIN hla --t1k $REF/hla/t1k --out $O --threads $T $FQ > $O/hla.log 2>&1
echo "donor $DONOR $SC exit $?"; tail -5 $O/hla.log; cat $O/private/hla/hla_genotype.tsv 2>/dev/null | cut -f1-3,6 | head -3
