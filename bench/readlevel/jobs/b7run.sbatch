#!/bin/bash
#SBATCH --job-name=rl-b7
#SBATCH --account=debruinz
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# B7 only: scTE again for A375 and mESC (804554's scTE failed at its last step: an absolute -o path breaks its temp paths), SoloTE and
# STARsolo outputs reused, then te_eval against the committed rule 6de5821a
set -uo pipefail
BR=${BR:-bench/readlevel-oracles}; NAME=${NAME:-rlo-13ee63ac}
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/readlevel; L=$W/logs/rl-b7-$SLURM_JOB_ID; mkdir -p $L $J/results
BIN=$J/bin/$NAME; BARE=$W/repos/singlet-py.git; PY=$W/envs/pbench/bin/python
SRC=$J/b7-src-$SLURM_JOB_ID; mkdir -p $SRC
( flock 9; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/rl/b7-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/rl/b7-$SLURM_JOB_ID bench/readlevel | tar -x -C $SRC; B=$SRC/bench/readlevel
echo "node=$(hostname) bench=$(git -C $BARE rev-parse refs/rl/b7-$SLURM_JOB_ID) $(date -Is)"
te() {   # NAME threads
  local n=$1 O=$J/te/$1 E=$J/runs/te/$1/out
  SLURM_JOB_ID=${SLURM_JOB_ID}t$n SLURM_CPUS_PER_TASK=$2 NAME=$n BIN=$BIN SKIP_ENGINE=1 bash $B/te/jobs/te_bench.sbatch > $L/te_bench-$n.log 2>&1
  if [ $n = a375_10x ]; then
    $PY -I $B/te/te_eval.py bulk --engine $E --scte $O/scte.h5ad --solote $O/solote --truth $O/longread_truth.tsv --out $J/results/b7_a375.json
  else
    $PY -I $B/te/te_eval.py twoc --engine $E --scte $O/scte.h5ad --solote $O/solote --solo-gene $O/starsolo/Solo.out/Gene/filtered --out $J/results/b7_mesc.json
  fi
}
( te a375_10x 14 ) > $L/te-a375.log 2>&1 & PA=$!
( te mesc_2c 16 ) > $L/te-mesc.log 2>&1 & PM=$!
for x in te-a375:$PA te-mesc:$PM; do wait ${x#*:}; echo "${x%%:*} exit $? $(date -Is)"; done
ls -l $J/results | grep b7
