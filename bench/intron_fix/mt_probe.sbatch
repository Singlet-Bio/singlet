#!/bin/bash
#SBATCH --job-name=if-mtprobe
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench
O=$W/pipeline-benchmark/runs/neuron1k_v3/singlet-if-keepbam
B=$O/star_Aligned.sortedByCoord.out.bam
ls -la $O | grep -i bam
$ENV/bin/python $W/jobs/intronfix/mt_probe.py $B /mnt/projects/debruinz_project/cellarium/reference/GRCm39-2024-A/genes/genes.gtf $ENV/bin/samtools
