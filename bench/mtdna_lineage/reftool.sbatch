#!/bin/bash
#SBATCH --job-name=mtl-reftool
#SBATCH --account=debruinz
#SBATCH --partition=cpu,bigmem
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch reftool.sbatch <name> <bam> <cells.txt> <mgatk|maegatk>
# The reference genotyper alone (mgatk bcall -ub UB for 3', maegatk bcall -mr 3 for MAESTER) on a cell subset; its
# per-cell snakemake steps make whole 15k-cell samples take > 10 h.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; P=$W/envs/pbench; E=$W/envs/mtl
export PATH=$E/bin:$PATH
NAME=$1; BAM=$2; CELLS=$3; TOOL=$4
O=$W/out/mtdna_lineage/pile/$NAME; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn)
if [ -n "${SUBSET_TRUTH:-}" ]; then   # env SUBSET_TRUTH=<truth cells> SUBSET_N=<n>: truth cells + random others
  bash $W/jobs/mtl/subset_cells.sh $CELLS $SUBSET_TRUTH ${SUBSET_N:-3000} $SCR/sub.txt; CELLS=$SCR/sub.txt
fi
sed 's/-1$//' $CELLS > $SCR/bc.txt
CB1=$(set +o pipefail; $P/bin/samtools view $BAM chrM:1-3000 | head -200 | grep -o 'CB:Z:[^[:space:]]*' | head -1)
case "$CB1" in *-1) sed 's/$/-1/' $SCR/bc.txt > $SCR/bc1.txt && mv $SCR/bc1.txt $SCR/bc.txt ;; esac
cp $SCR/bc.txt $O/${TOOL}_cells.txt
rm -rf $O/$TOOL
if [ "$TOOL" = mgatk ]; then
  /usr/bin/time -v -o $O/$TOOL.time.txt mgatk bcall -i $BAM -o $O/$TOOL -n $NAME -g hg38 -c $SLURM_CPUS_PER_TASK -bt CB -b $SCR/bc.txt \
    -ub UB -sr -so > $O/$TOOL.log 2>&1 || { echo "$TOOL failed"; tail -30 $O/$TOOL.log; }
else
  /usr/bin/time -v -o $O/$TOOL.time.txt maegatk bcall -i $BAM -o $O/$TOOL -n $NAME -g rCRS -c $SLURM_CPUS_PER_TASK -bt CB -b $SCR/bc.txt \
    -ub UB -mr 3 -sr -so > $O/$TOOL.log 2>&1 || { echo "$TOOL failed"; tail -30 $O/$TOOL.log; }
fi
grep -E "Elapsed|Maximum res" $O/$TOOL.time.txt || true; ls -la $O/$TOOL/final 2>/dev/null || true
