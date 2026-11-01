#!/bin/bash
#SBATCH --job-name=sjnh-oracle
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch oracle_run.sbatch LABEL UNMAPPED_MATE1 UNMAPPED_MATE2 CELLS_TSV
# Independent per-read oracle: map the STAR-unmapped cDNA reads to the SARS-CoV-2 genome with minimap2 (-x sr).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; B=$W/modality-bench/sj_nonhost; L=$1
O=$B/oracle/$L; mkdir -p $O
/usr/bin/time -v -o $O/time.txt $W/envs/kraken2/bin/minimap2 -x sr -t 8 $B/db/NC_045512.2.fa $2 > $O/hits.paf 2> $O/minimap2.log
$W/envs/pbench/bin/python -I $B/scripts/oracle_percell.py $O/hits.paf $3 $4 $O/percell.tsv
