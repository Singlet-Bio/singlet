#!/bin/bash
#SBATCH --job-name=allsite-pack
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=700G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# D-101 validation packed in one job (size it to the largest free slot; one node): every line of $LIBS ("name truth input", truth "-" for none) runs validate.sbatch in
# parallel with an equal share of the node's cores. Needs BINDIR and LIBS (a file). Logs per library: $W/logs/allsite-pack-$JOB-<name>.out
set -uo pipefail
: "${BINDIR:?}" "${LIBS:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; HERE=$W/jobs/allsite/val
N=$(grep -c . $LIBS); C=$(nproc); export THREADS=$(( C / N )) BINDIR
echo "node $(hostname) cores $C libraries $N threads each $THREADS"
while read -r name truth input; do
  [ -z "$name" ] && continue; [ "$truth" = - ] && truth=""
  NAME=$name TRUTH=$truth INPUT=$input bash $HERE/validate.sbatch > $W/logs/allsite-pack-$SLURM_JOB_ID-$name.out 2>&1 &
done < $LIBS
wait; echo "all done $(date -Is)"
