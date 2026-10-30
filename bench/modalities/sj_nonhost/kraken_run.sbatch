#!/bin/bash
#SBATCH --job-name=sjnh-kraken
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch kraken_run.sbatch LABEL UNMAPPED_MATE1 UNMAPPED_MATE2 CELLS_BARCODES.tsv
# Kraken2 (viral DB, k2_viral 2024-06-05) on the reads STAR could not map to the host, i.e. the same reads
# Singlet screens; per-cell attribution from the barcode mate. Time and RSS recorded with /usr/bin/time -v.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; B=$W/modality-bench/sj_nonhost
L=$1; M1=$2; M2=$3; CELLS=$4
O=$B/kraken/$L; rm -rf $O; mkdir -p $O
export PATH=$W/envs/kraken2/bin:$PATH
for CONF in 0.0 0.1; do
  /usr/bin/time -v -o $O/time_conf$CONF.txt kraken2 --db $B/db/k2_viral --threads 16 --confidence $CONF \
    --report $O/report_conf$CONF.txt --output $O/out_conf$CONF.txt $M1 > $O/kraken_conf$CONF.log 2>&1
  grep -E "Elapsed|Maximum resident" $O/time_conf$CONF.txt; cat $O/kraken_conf$CONF.log
done
$W/envs/pbench/bin/python -I $W/modality-bench/sj_nonhost/scripts/kraken_percell.py $O $M2 $CELLS $B/db/k2_viral/ktaxonomy.tsv
