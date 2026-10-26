#!/bin/bash
# Sourced by the ambient-calibration jobs on Clipper. Syncs this branch into $SRC, builds the harness, defines `py` (python 3.12
# image + venvs/singlet). Inputs: stored outputs of the bench-public Singlet runs (singlet-campaign, --pipeline: default SNP panel,
# --raw-matrix) and the barnyard inputs of bench/latent_qc/ambient_truth (PR #43). Results: $AF (never R2 / D1).
W=/mnt/projects/debruinz_project/debruinz/singlet
BR=${AMB_BRANCH:-fix/ambient-calibration}
REPO=$W/repos/singlet-py.git
SRC=$W/src/singlet-py/${BR//\//__}
LQC_SRC=$W/src/singlet-py/study__latent-qc
AF=$W/ambfix
RUNS=$W/bench-public/runs
TRUTH=$W/modality-bench/genotype_pool/truth
CXX=$W/envs/pbench/bin/x86_64-conda-linux-gnu-g++
SIF=$W/images/python-3.12-bookworm.sif
mkdir -p $AF/dump $AF/est $AF/res
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
af_sync() {
  (
    flock -w 3600 9
    git --git-dir=$REPO fetch -q --refmap= origin "+refs/heads/$BR:refs/heads/$BR"
    tip=$(git --git-dir=$REPO rev-parse refs/heads/$BR)
    [ -e $SRC/.git ] || git --git-dir=$REPO worktree add -q --detach $SRC $tip
    git -C $SRC checkout -q --detach $tip
  ) 9>$W/state/ambfix-sync.lock
  echo "ambfix @ $(git -C $SRC rev-parse --short HEAD) host $(hostname)"
}
af_build() {  # $1 = output binary
  $CXX -std=c++20 -O2 -pthread -I$SRC/include $SRC/bench/modalities/ambient_calibration/harness.cpp -o $1
}
py() {
  apptainer exec --bind $W,/tmp --env PATH=$W/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin \
    --env OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} --env OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} $SIF python -I "$@"
}
# lane -> (bench-public run id, truth table, lane code)
lane_run() { case $1 in ctrl) echo kang2018_ctrl;; A) echo kang2018_b1_a;; B) echo kang2018_b1_b;; C) echo kang2018_b1_c;; stim) echo kang2018_b2_stim;; esac; }
# counts: the default run (stim has only the --n-donors 8 / auto runs; same binary, identical ambient output)
lane_rundir() { case $1 in stim) echo $RUNS/kang2018_b2_stim/singlet_k8;; *) echo $RUNS/$(lane_run $1)/singlet;; esac; }
# Genotype truth needs the right donor clustering: on lane 2.1 (8 donors) the default run's auto-K found 9 donors (one spurious
# 108-cell donor), which deflates the truth to 1.47%; the --n-donors 8 rerun of the same binary has byte-identical counts and
# ambient output (md5 of ambient_contamination.tsv) and gives 2.40% (PR #43, singlet_modfix: 2.48%). b1 A/B (4 donors) and C (8)
# are called with the right K by default.
lane_truth_run() { case $1 in ctrl) echo $RUNS/kang2018_ctrl/singlet_k8;; *) lane_rundir $1;; esac; }
lane_truth() { case $1 in ctrl|stim) echo $TRUTH/GSE96583_batch2.total.tsne.df.tsv.gz;; *) echo $TRUTH/GSE96583_batch1.total.tsne.df.tsv.gz;; esac; }
BARNYARD="hgmm1k_v3_singlet hgmm1k_v3 hgmm5k_v3 hgmm10k_v3 hgmm1k_v2 hgmm6k_v2 hgmm12k_v2"
