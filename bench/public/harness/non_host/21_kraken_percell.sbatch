#!/bin/bash
#SBATCH --job-name=bp-nh-kpercell
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch 21_kraken_percell.sbatch
# Re-runs only the per-species / per-cell step (kraken_percell.py) on the saved Kraken2 output of every sample and database,
# without re-running Kraken2. Used when the roll-up rules change.
set -eu
source ${BP_SRC:-/mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet}/bench/public/harness/perf_common.sh
for L in gse166766_hbec_dpi2 gse166766_hbec_mock pbmc1k_v3; do
  for D in pluspf08 viral; do
    O=$BP/nonhost/$L/kraken2_$D; S=$BP/nonhost/$L/singlet
    echo "== $L $D"
    $PY -I $SRC/bench/public/harness/non_host/kraken_percell.py $O/out.txt.gz $O/report.txt $S/star_Unmapped.out.mate2 $BP/nonhost/cells/$L.txt $O | tee $O/percell.log
  done
done
