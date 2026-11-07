#!/bin/bash
#SBATCH --job-name=as-atac-rerun
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Re-run the ATAC post-alignment from a kept BAM (singlet atac), e.g. after a whitelist change, timed.
#   RUN=<run name under runs/atacsp> [PAIRED_GEX=<gex run dir>] [Q=<peak q cutoff>] [TAG=<output suffix>] [BIN=singlet-atacsp]
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/${BIN:-singlet-atacsp}
G=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
O=$PB/runs/atacsp/$RUN; N=$O/post${TAG:-}; rm -rf $N; mkdir -p $N
X=""; [ -n "${PAIRED_GEX:-}" ] && X="--paired-gex $PAIRED_GEX"; [ -n "${Q:-}" ] && X="$X --q-cutoff $Q"
/usr/bin/time -v -o $N/time_post.txt $BIN atac --bam $O/atac_sorted.bam --out $N --gtf $G/genes/genes.gtf \
  --whitelist $PB/whitelists/atacsp/737K-cratac-v1.txt --whitelist $PB/whitelists/737K-arc-v1-atac-gex.tsv.gz $X > $N/atac.log 2>&1
echo "exit=$? wall=$(grep 'Elapsed (wall' $N/time_post.txt | sed 's/.*: //') rss_kb=$(grep 'Maximum resident' $N/time_post.txt | sed 's/.*: //')"
cat $N/atac.log; cat $N/atac_summary.json
