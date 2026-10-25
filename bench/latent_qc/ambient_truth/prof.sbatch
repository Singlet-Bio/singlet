#!/bin/bash
#SBATCH --job-name=at-prof
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
export STUDY_BRANCH=study/latent-qc-ambient-truth; W=/mnt/projects/debruinz_project/debruinz/singlet; source <(git --git-dir=$W/repos/singlet-py.git show refs/heads/$STUDY_BRANCH:bench/latent_qc/common.sh); lqc_sync
T=$SRC/bench/latent_qc/ambient_truth
rt python -I -c "
import sys,cProfile,pstats,time; sys.path.insert(0,'$T')
import atlib, methods
t=time.time(); ds=atlib.load('hgmm1k_v3_singlet'); print('load',time.time()-t,flush=True)
cProfile.run('methods.latent_knn(ds,k=15,tau=10)','/tmp/p.out')
pstats.Stats('/tmp/p.out').sort_stats('cumtime').print_stats(18)
"
