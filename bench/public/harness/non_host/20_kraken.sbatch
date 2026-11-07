#!/bin/bash
#SBATCH --job-name=bp-nh-kraken
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch 20_kraken.sbatch LABEL DB         DB = pluspf08 (standard, with human) | viral (viral-only, first-benchmark database)
# Kraken2 at its defaults (--confidence 0.0, k=35, l=31) on the reads STAR could not map to the host, i.e. the reads
# Singlet screens, from the Singlet run of 10_singlet.sbatch. Per-cell attribution from the barcode mate.
set -eu
source ${BP_SRC:-/mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet}/bench/public/harness/perf_common.sh
L=$1; D=$2
case $D in
  pluspf08) DB=$BP/db/k2_pluspf08 ;;
  viral) DB=$WORK/modality-bench/sj_nonhost/db/k2_viral ;;
  *) echo "unknown db $D"; exit 2 ;;
esac
S=$BP/nonhost/$L/singlet; O=$BP/nonhost/$L/kraken2_$D; rm -rf $O; mkdir -p $O
export PATH=$WORK/envs/kraken2/bin:$PATH
echo "[bp] $L kraken2 $(kraken2 --version | head -1) db=$DB $(cat $DB/VERSION 2>/dev/null) node=$(hostname) cpu=$(cpu_model)" | tee $O/node.txt
timed $O/run kraken2 --db $DB --threads 16 --report $O/report.txt --output $O/out.txt $S/star_Unmapped.out.mate1 > $O/kraken.log 2>&1
cat $O/kraken.log; grep -hE "Elapsed|Maximum resident" $O/run/time.txt; cat $O/run/mem.json; echo
$PY -I $SRC/bench/public/harness/non_host/kraken_percell.py $O/out.txt $O/report.txt $S/star_Unmapped.out.mate2 $BP/nonhost/cells/$L.txt $O | tee $O/percell.log
rm -f $O/out.txt.gz; gzip -f $O/out.txt
