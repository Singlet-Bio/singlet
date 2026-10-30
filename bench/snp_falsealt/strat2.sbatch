#!/bin/bash
#SBATCH --job-name=sfa-strat2
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --nice=100
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
J=$W/jobs/snp-falsealt; RUNS=$W/bench-public/runs
for POOL in kang2018_stim kang2018_ctrl kang2018_b1_a kang2018_b1_b kang2018_b1_c; do
  python -I $J/strat1.py $RUNS $W/bench-public/data/kang_truth $RUNS $J/pub_$POOL.txt $POOL || echo fail $POOL
done
python -I $J/sweep.py $J/sweep_pub.txt $J/pub_*.txt.pkl
python -I $J/sweep.py $J/sweep_all.txt $J/pub_kang2018_stim.txt.pkl $J/pub_kang2018_ctrl.txt.pkl
