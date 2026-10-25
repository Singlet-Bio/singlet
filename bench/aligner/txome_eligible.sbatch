#!/bin/bash
#SBATCH --job-name=al-txelig
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch txome_eligible.sbatch <dataset>
# Run Singlet with the sorted BAM kept, then classify reads by what a transcriptome-first step could take.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
DS=$1; O=$W/aligner-bench/txelig/$DS; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
PYLIB=$W/envs/pylib-pysam
[ -d $PYLIB/pysam ] || $W/envs/pbench/bin/python -m pip install -q --target $PYLIB pysam
BAM=$W/aligner-bench/bam/$DS.bam; mkdir -p $(dirname $BAM)
if [ ! -s $BAM ]; then
  cp $PB/data/$DS/$DS.1fq $SCR/s.1fq
  $PB/bin/singlet-fixed-det $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
    --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $SCR/out/ --threads 16 --pipeline \
    --output-format mtx --raw-matrix --tagged-bam $BAM > $O/run.log 2>&1
fi
PYTHONPATH=$PYLIB $W/envs/pbench/bin/python $W/jobs/aligner/txome_eligible.py $BAM $REF/genes/genes.gtf $O/eligible.json 10
