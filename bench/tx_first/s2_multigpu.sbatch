#!/bin/bash
#SBATCH --job-name=txf-2gpu
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --time=03:55:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Study 2 part 1: the standard throughput point on both GPUs of one node (binary b0, commit c3a6096c, harness of gpu/bold).
# It runs the gpu/bold harness (jobs/tp.sbatch on Clipper, a copy of bench/gpu_bold/jobs/tp.sbatch) after three patches, made here with asserts:
#   cfg g1    : CUDA_VISIBLE_DEVICES=0, so the same k runs on one GPU of this node
#   cfg mps2  : one CUDA MPS daemon per GPU (separate pipe dirs), each process sees one GPU (worker w uses GPU w % 2)
#   RES0      : under jobs/tx-first/results/study2
# Points: k8t7s2r3 (2 GPUs, 2 slots per GPU), k8t7s2r3@g1 (1 GPU), k16t3s2r2 (2 GPUs), k12t4s3r2@mps2 (2 GPUs, 3 slots per GPU, MPS).
# The engine spreads processes over both devices by itself (first free slot of the first device, else wait; include/singlet/engine_gpu.h).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-56}
OUTD=$W/jobs/tx-first/results/study2; mkdir -p $OUTD
echo "node $(hostname -s) job $SLURM_JOB_ID gpus $(nvidia-smi -L | wc -l) $(date -Is)"
python3 - <<'PY'
import re
W = "/mnt/projects/debruinz_project/debruinz/singlet"
s = open(W + "/jobs/gpu-bold/jobs/tp.sbatch").read()
def sub(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)
sub('case $RES0 in $W/jobs/gpu-bold/*) ;; *) echo "RES0 must lie under $W/jobs/gpu-bold"; exit 2 ;; esac',
    'case $RES0 in $W/jobs/gpu-bold/*|$W/jobs/tx-first/*) ;; *) echo "RES0 under the wrong tree"; exit 2 ;; esac')
sub('cfg_set() {\n  CFG_ARGS=""; CFG_ENV=""',
    'start_mps2() {\n  [ -n "${MPS2_ON:-}" ] && return 0\n  for g in 0 1; do d=/tmp/gbmps-$SLURM_JOB_ID-$g; mkdir -p $d $RES0/mps-log-$g\n'
    '    CUDA_VISIBLE_DEVICES=$g CUDA_MPS_PIPE_DIRECTORY=$d CUDA_MPS_LOG_DIRECTORY=$RES0/mps-log-$g nvidia-cuda-mps-control -d; done\n'
    '  MPS2_ON=1; echo "MPS daemons started (one per GPU) $(date -Is)" | tee -a $S\n}\n'
    'stop_mps2() {\n  [ -z "${MPS2_ON:-}" ] && return 0\n  for g in 0 1; do echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/gbmps-$SLURM_JOB_ID-$g nvidia-cuda-mps-control; done\n  MPS2_ON=""; MPS2=""; sleep 3\n}\n'
    'cfg_set() {\n  CFG_ARGS=""; CFG_ENV=""; MPS2=""\n  case $1 in mps2) ;; *) stop_mps2 ;; esac')
sub('    base) ;;\n', '    base) ;;\n    g1) CFG_ENV="CUDA_VISIBLE_DEVICES=0" ;;\n    mps2) MPS2=1; start_mps2 ;;\n')
sub('  [ -n "$PIN" ] && XE+=(SINGLET_ENGINE_GPU_DEVICE=$((w % NGPU)))\n',
    '  [ -n "$PIN" ] && XE+=(SINGLET_ENGINE_GPU_DEVICE=$((w % NGPU)))\n'
    '  [ -n "${MPS2:-}" ] && XE+=(CUDA_VISIBLE_DEVICES=$((w % NGPU)) CUDA_MPS_PIPE_DIRECTORY=/tmp/gbmps-$SLURM_JOB_ID-$((w % NGPU)))\n')
sub('echo done $(date -Is) | tee -a $S', 'stop_mps2\necho done $(date -Is) | tee -a $S')
open("/tmp/txf-tp2-" + __import__("os").environ["SLURM_JOB_ID"] + ".sbatch", "w").write(s)
PY
[ -f /tmp/txf-tp2-$SLURM_JOB_ID.sbatch ] || { echo "patch failed"; exit 3; }
export BINDIR=$W/jobs/gpu-bold/bin/b0 EXPECT_COMMIT=c3a6096c RES0=$OUTD/tp-$SLURM_JOB_ID IDENT=0 SAMPLE=pbmc1k_v3
export POINTS="${POINTS:-k8t7s2r3 k8t7s2r3@g1 k16t3s2r2 k12t4s3r2@mps2}"
bash /tmp/txf-tp2-$SLURM_JOB_ID.sbatch
rc=$?; rm -f /tmp/txf-tp2-$SLURM_JOB_ID.sbatch
# host memory, lock wait and core use are columns of results.tsv (host_peak_gb, lock_wait_frac, cores_win); lane kernel times come from star_Log.out (summarizer)
echo "harness exit $rc"; ls $RES0 | head -20
exit $rc
