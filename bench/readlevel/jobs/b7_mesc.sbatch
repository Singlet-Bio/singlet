#!/bin/bash
#SBATCH --job-name=rl-b7m
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# B7 mESC: scTE finished in 804624 (4 h 37 min, exit 0); only te_eval is left, against the committed rule 6de5821a
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/readlevel; O=$J/te/mesc_2c; BARE=$W/repos/singlet-py.git; BR=${BR:-bench/readlevel-oracles}
export PZ_SO=$W/venvs/forge/lib/python3.12/site-packages/singlet/_pz.cpython-312-x86_64-linux-gnu.so LD_LIBRARY_PATH=$W/envs/pbench/lib:${LD_LIBRARY_PATH:-}
B=$J/b7m-src-$SLURM_JOB_ID; mkdir -p $B
( flock 9; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/rl/b7m-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/rl/b7m-$SLURM_JOB_ID bench/readlevel | tar -x -C $B; B=$B/bench/readlevel
echo "node=$(hostname) bench=$(git -C $BARE rev-parse refs/rl/b7m-$SLURM_JOB_ID) $(date -Is)"
$W/envs/pbench/bin/python -I $B/te/te_eval.py twoc --engine $J/runs/te/mesc_2c/out --scte $O/scte.h5ad --solote $O/solote --solo-gene $O/starsolo/Solo.out/Gene/filtered --out $J/results/b7_mesc.json
echo "te_eval exit $? $(date -Is)"
