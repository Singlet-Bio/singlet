#!/bin/bash
#SBATCH --job-name=bp-nh-oracle
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch 30_oracle.sbatch LABEL
# Independent per-read oracle for SARS-CoV-2: minimap2 -x sr of the STAR-unmapped cDNA reads against NC_045512.2; a read
# counts when >= 60 nt align at >= 90 % identity; molecules = distinct (cell, UMI). This is the truth for the infected-vs-
# mock benchmark: a read-level alignment, not an independent cell annotation.
set -eu
source /mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet/bench/public/harness/common.sh
L=$1; OLD=$WORK/modality-bench/sj_nonhost
S=$BP/nonhost/$L/singlet; O=$BP/nonhost/$L/oracle; rm -rf $O; mkdir -p $O
echo "[bp] $L minimap2 $($WORK/envs/kraken2/bin/minimap2 --version) node=$(hostname)" | tee $O/node.txt
timed $O/run $WORK/envs/kraken2/bin/minimap2 -x sr -t 8 -o $O/hits.paf $OLD/db/NC_045512.2.fa $S/star_Unmapped.out.mate1 2> $O/minimap2.log
grep -hE "Elapsed|Maximum resident" $O/run/time.txt
$PY -I $SRC/bench/public/harness/non_host/oracle_percell.py $O/hits.paf $S/star_Unmapped.out.mate2 $BP/nonhost/cells/$L.txt $O/percell.tsv | tee $O/percell.log
gzip -f $O/hits.paf
