#!/bin/bash
#SBATCH --job-name=rl-b7a
#SBATCH --account=debruinz
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# B7 A375: scTE and SoloTE on the STARsolo BAM made with the Multiome whitelist (step 804624.4 finished STARsolo and died when its batch
# exited), then te_eval against the committed rule 6de5821a
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/readlevel; O=$J/te/a375_10x; BARE=$W/repos/singlet-py.git; BR=${BR:-bench/readlevel-oracles}
export PZ_SO=$W/venvs/forge/lib/python3.12/site-packages/singlet/_pz.cpython-312-x86_64-linux-gnu.so LD_LIBRARY_PATH=$W/envs/pbench/lib:${LD_LIBRARY_PATH:-}
B=$J/b7a-src-$SLURM_JOB_ID; mkdir -p $B
( flock 9; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/rl/b7a-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/rl/b7a-$SLURM_JOB_ID bench/readlevel | tar -x -C $B; B=$B/bench/readlevel
echo "node=$(hostname) bench=$(git -C $BARE rev-parse refs/rl/b7a-$SLURM_JOB_ID) $(date -Is)"
[ -s $O/starsolo/Aligned.sortedByCoord.out.bam.bai ] || { echo "no STARsolo BAM"; exit 1; }
ls $O/solote/*/*_MATRIX/matrix.mtx* >/dev/null 2>&1 || rm -rf $O/solote $O/solote.log $O/solote_time.txt
SLURM_JOB_ID=${SLURM_JOB_ID}a SLURM_CPUS_PER_TASK=24 NAME=a375_10x BIN=$J/bin/rlo-13ee63ac SKIP_ENGINE=1 CBWL_OVERRIDE=$W/pipeline-benchmark/whitelists/gex_737K-arc-v1.txt \
  bash $B/te/jobs/te_bench.sbatch
$W/envs/pbench/bin/python -I $B/te/te_eval.py bulk --engine $J/runs/te/a375_10x/out --scte $O/scte.h5ad --solote $O/solote --truth $O/longread_truth.tsv --out $J/results/b7_a375.json
echo "te_eval exit $? $(date -Is)"
