#!/bin/bash
#SBATCH --job-name=pb-afidx
#SBATCH --account=debruinz
#SBATCH --partition=scavenger,short-cpu
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# alevin-fry (simpleaf/piscem) spliced+unspliced ("USA") index from the SAME genome FASTA
# and GTF that Singlet and STARsolo use.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__molecule-labels/bench/pipeline/common.sh
SP=(GRCh38-2024-A GRCm39-2024-A GRCh38-GRCm39-barnyard); S=${SP[$SLURM_ARRAY_TASK_ID]}
mkdir -p $ALEVIN_FRY_HOME $PB/refs/$S
FA=$REF/$S/fasta/genome.fa; [ -f $FA ] || FA=$(ls $REF/$S/fasta/*.fa* | head -1)
GTF=$REF/$S/genes/genes.gtf
simpleaf set-paths
[ -f $PB/refs/$S/af_idx/index/simpleaf_index.json ] || /usr/bin/time -v simpleaf index \
  --output $PB/refs/$S/af_idx --fasta $FA --gtf $GTF --ref-type spliced+unspliced \
  --rlen 91 --threads $SLURM_CPUS_PER_TASK --work-dir $PB/refs/$S/af_work 2> $PB/refs/$S/af_index.time
ls $PB/refs/$S/af_idx $PB/refs/$S/af_idx/index; tail -20 $PB/refs/$S/af_index.time
