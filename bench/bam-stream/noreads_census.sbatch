#!/bin/bash
#SBATCH --job-name=bs-noreads
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Read-only census of pipeline runs that ended with "ERROR: No reads processed" (the 1fq FIFO open race
# fixed on fix/fifo-open-race leaves exactly this signature: STAR maps 0 reads).
# Searches text logs and JSON ledgers under our work tree and, read-only, the reprocessing bot's tree.
# Writes one line per hit: file, then the nearby sample/GSM/SRR ids and FIFO/STAR lines.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; O=$W/bam-stream/noreads; mkdir -p $O
ROOTS="$W/logs $W/state $W/atlas-rerun $W/alpha_bench $W/out $W/results /mnt/projects/debruinz_project/singlet_pipeline"
: > $O/hits.txt
for r in $ROOTS; do
  [ -d $r ] || continue
  echo "## $r" >> $O/hits.txt
  # -I skips binaries; -s quiet on unreadable files; exclude our own benchmark trees
  grep -rIls --exclude-dir=bam-stream --exclude-dir=.git -e "No reads processed" $r 2>/dev/null | while read -r f; do
    ids=$(grep -oE "\b(GSM[0-9]+|SRR[0-9]+|ERR[0-9]+|DRR[0-9]+|GSE[0-9]+)\b" "$f" | sort -u | head -8 | tr '\n' ' ')
    fifo=$(grep -m1 -E "1fq-fifo\] Pass 2" "$f" | tr -s ' ')
    star=$(grep -m1 -E "Very low mapping rate|STAR completed" "$f" | tr -s ' ')
    n=$(grep -c "No reads processed" "$f")
    printf "%s\tn=%s\tids=%s\t%s\t%s\n" "$f" "$n" "$ids" "$fifo" "$star" >> $O/hits.txt
  done
done
grep -vc "^##" $O/hits.txt | sed 's/^/files_with_hits=/'
