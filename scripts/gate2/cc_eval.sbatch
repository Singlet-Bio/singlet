#!/bin/bash
#SBATCH --job-name=cc-eval
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Cell-calling before/after on the gate datasets: builds bench/called_ed_truth/cv2_stored.cpp from BRANCH (default claude/s12-gate2), runs the
# pipeline's called_v2 (2.1 and 2.2 sets) on the stored counts of release-gate/gate2/pb/runs/<ds>/singlet-public, scores both against the truth
# columns of the gate-2 artifact tables. Output: $W/jobs/gate2/cc-eval/<COMMIT8>/. Env: DS (default the 8 gate datasets).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; BARE=$W/repos/singlet-py.git
BRANCH=${BRANCH:-claude/s12-gate2}; DS=${DS:-"nuclei2k_v2 kang2018_ctrl neuron1k_v3 pbmc1k_v2 pbmc1k_v3 pbmc1k_cite_v3 hgmm1k_v3 pbmc5p_v1"}
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BRANCH:refs/gate2/cc
COMMIT=$(git -C $BARE rev-parse refs/gate2/cc); O=$W/jobs/gate2/cc-eval/${COMMIT:0:8}; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $COMMIT | tar -x -C $SCR/src
apptainer exec -B /mnt/projects/debruinz_project -B $SCR $IMG $ENV/bin/x86_64-conda-linux-gnu-g++ -std=c++17 -O2 -pthread -I$SCR/src/include \
  $SCR/src/bench/called_ed_truth/cv2_stored.cpp -o $O/cv2_stored -Wl,-rpath,$ENV/lib -L$ENV/lib
for d in $DS; do
  R=$W/release-gate/gate2/pb/runs/$d/singlet-public
  LD_LIBRARY_PATH=$ENV/lib $O/cv2_stored $R $O/$d.tsv 8 > $O/$d.params.json
  echo "[cc-eval] $d done"
done
$ENV/bin/python $SCR/src/scripts/gate2/cc_eval.py $O $DS | tee $O/eval.txt
