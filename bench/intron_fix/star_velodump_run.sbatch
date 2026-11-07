#!/bin/bash
#SBATCH --job-name=if-starvd
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Re-run the public STARsolo command (runs/<ds>/starsolo-public-cr/command.txt) with STAR-velodump:
# same matrices + runs/<ds>/starsolo-velodump/velo_umis.tsv.gz (one line per CB x UMI Velocyto call).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
DS=(pbmc1k_v3 nuclei2k_v2 hgmm1k_v3); DS=${DS[$SLURM_ARRAY_TASK_ID]}
O=$PB/runs/$DS/starsolo-velodump; rm -rf $O; mkdir -p $O; cd $O
SCR=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
CMD=$(cat $PB/runs/$DS/starsolo-public-cr/command.txt)
CMD=${CMD/#STAR /$PB/bin/STAR-velodump }
CMD=$(echo "$CMD" | sed -E "s#--outTmpDir [^ ]+#--outTmpDir $SCR/startmp#")
echo "$CMD" > command.txt
export LD_LIBRARY_PATH=$ENV/lib STAR_VELO_DUMP=$SCR/velo_umis.tsv
eval "/usr/bin/time -v -o time.txt $CMD" > run.log 2>&1
gzip -1 -c $SCR/velo_umis.tsv > velo_umis.tsv.gz
for f in Velocyto/raw/spliced.mtx Velocyto/raw/unspliced.mtx Velocyto/raw/ambiguous.mtx GeneFull/raw/matrix.mtx; do
  if cmp -s <(grep -v "^%" Solo.out/$f) <(grep -v "^%" $PB/runs/$DS/starsolo-public-cr/Solo.out/$f); then echo "same $f"; else echo "DIFFERENT $f"; fi
done
zcat velo_umis.tsv.gz | cut -f4 | sort | uniq -c
