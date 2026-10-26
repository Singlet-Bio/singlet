#!/bin/bash
#SBATCH --job-name=bp-fetch10x
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch fetch_10x.sbatch <cohort-id> <FASTQ tar url>
# Download one public 10x FASTQ archive into $BP/data/<id>/ and list its gene-expression FASTQs.
set -euo pipefail
BP=/mnt/projects/debruinz_project/debruinz/singlet/bench-public
ID=$1; TAR=$2; D=$BP/data/$ID; mkdir -p $D; cd $D
echo "[fetch] $ID $TAR $(date -Is)"
if [ ! -f .extracted ]; then
  curl -sSL --retry 5 -C - -o fastqs.tar "$TAR"
  tar -xf fastqs.tar && rm -f fastqs.tar && touch .extracted
fi
find $D -name "*.fastq.gz" | sort > all_fastqs.txt
grep -viE "antibody|_adt_|_I[12]_" all_fastqs.txt | grep -E "_R[12]_" > gex_fastqs.txt || true
wc -l all_fastqs.txt gex_fastqs.txt; du -sh $D
