#!/bin/bash
#SBATCH --job-name=gsota-frzclean
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Delete the re-fetched dev-lane reads (BAM and FASTQ) once the freeze run has finished; the tagged-BAM run outputs stay. Submit with
#   sbatch --dependency=afterany:<freeze_run job id> freeze_cleanup.sbatch
D=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/data/kang_b2ctrl
ls -la $D; rm -rf $D/lane.bam $D/parts $D/fastq; ls -la $D
