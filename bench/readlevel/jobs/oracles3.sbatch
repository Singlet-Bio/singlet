#!/bin/bash
#SBATCH --job-name=rl-or3
#SBATCH --account=debruinz
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=14:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Launch oracle B7, part 3 (after 804425): finish scTE, SoloTE and te_eval for A375 and mESC on one packed node with a limit that fits.
# Finished engine runs, STARsolo BAMs, scTE and SoloTE outputs are reused. 804425's mESC comparators used swapped mates (te_bench's
# length rule on 100/101 bp reads); they are cleared once and rebuilt with the poly-T rule.
set -uo pipefail
BR=${BR:-bench/readlevel-oracles}; NAME=${NAME:-rlo-13ee63ac}
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/readlevel; L=$W/logs/rl-or3-$SLURM_JOB_ID; mkdir -p $L $J/results
BIN=$J/bin/$NAME; TD=$W/data/readlevel/te; BARE=$W/repos/singlet-py.git; PY=$W/envs/pbench/bin/python
export PZ_SO=$W/venvs/forge/lib/python3.12/site-packages/singlet/_pz.cpython-312-x86_64-linux-gnu.so
export LD_LIBRARY_PATH=$W/envs/pbench/lib:${LD_LIBRARY_PATH:-}
SCR=/tmp/$USER-or3-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
( flock 9; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/rl/or3-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/rl/or3-$SLURM_JOB_ID bench/readlevel | tar -x -C $SCR; B=$SCR/bench/readlevel
echo "node=$(hostname) bench=$(git -C $BARE rev-parse refs/rl/or3-$SLURM_JOB_ID) bin=$(cat $BIN/commit-gpu) $(date -Is)"

O=$J/te/mesc_2c
if [ ! -f $TD/mesc_2c/.mates_fixed2 ]; then
  rm -rf $O/starsolo $O/scte* $O/solote* $O/comparators.log
  ln -sfn $TD/mesc_2c/ERR3981386_1.fastq.gz $TD/mesc_2c/10x/mesc_2c_S1_L001_R1_001.fastq.gz
  ln -sfn $TD/mesc_2c/ERR3981386_2.fastq.gz $TD/mesc_2c/10x/mesc_2c_S1_L001_R2_001.fastq.gz
  touch $TD/mesc_2c/.mates_fixed2; echo "mesc_2c: comparator outputs from swapped mates cleared"
fi
grep -h "Uniquely mapped reads %" $J/runs/te/mesc_2c/out/*Log.final.out $J/runs/te/mesc_2c/*.log 2>/dev/null | head -2

te() {   # NAME GENOME threads
  local n=$1 O=$J/te/$1 E=$J/runs/te/$1/out
  [ -d $E ] || { echo "no engine run for $n"; return 1; }
  [ -s $O/starsolo/Aligned.sortedByCoord.out.bam.bai ] || rm -rf $O/starsolo
  SLURM_JOB_ID=${SLURM_JOB_ID}t$n SLURM_CPUS_PER_TASK=$3 NAME=$n BIN=$BIN SKIP_ENGINE=1 bash $B/te/jobs/te_bench.sbatch > $L/te_bench-$n.log 2>&1
  if [ $n = a375_10x ]; then
    $PY -I $B/te/te_eval.py bulk --engine $E --scte $O/scte.h5ad --solote $O/solote --truth $O/longread_truth.tsv --out $J/results/b7_a375.json
  else
    $PY -I $B/te/te_eval.py twoc --engine $E --scte $O/scte.h5ad --solote $O/solote --solo-gene $O/starsolo/Solo.out/Gene/filtered --out $J/results/b7_mesc.json
  fi
}
( te a375_10x GRCh38-2024-A 12 ) > $L/te-a375.log 2>&1 & PA=$!
( te mesc_2c GRCm39-2024-A 20 ) > $L/te-mesc.log 2>&1 & PM=$!
for x in te-a375:$PA te-mesc:$PM; do wait ${x#*:}; echo "${x%%:*} exit $? $(date -Is)"; done
ls -l $J/results | grep b7
