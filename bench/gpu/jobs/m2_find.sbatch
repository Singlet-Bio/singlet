#!/bin/bash
#SBATCH --job-name=w11-find
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:25:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Read-only discovery of staged .1fq inputs and CPU run logs (W11 M2 / M10).
W=/mnt/projects/debruinz_project/debruinz/singlet
echo "== 1fq files"; find $W -name '*.1fq' -printf '%s %p\n' 2>/dev/null | sort -k2 | head -400
echo "== run.log"; find $W/pipeline-benchmark $W/bam-stream $W/deep-perf $W/atlas-rerun -maxdepth 4 \( -name run.log -o -name 'stage*.json' -o -name 'timers*.json' \) 2>/dev/null | head -300
echo "== logs count"; ls $W/logs | wc -l
