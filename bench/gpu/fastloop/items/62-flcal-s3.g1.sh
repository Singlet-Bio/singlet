#!/bin/bash
# PILOT-MINUTES: 103
# fastloop item (bench/gpu/fastloop, design.py): 3 ideas wshrink,shidx,stepsauto, tier S3: pbmc1k_v3, one run at a time, 2 runs per cell point (the first is warm-up), ends after STAR and the pileup.
# 8 cells x 2 blocks, resolution full. Effects, CIs and stages: fastloop.md in the results directory.
W=/mnt/projects/debruinz_project/debruinz/singlet; P2=$W/jobs/gpu-pr2; FL=$W/jobs/gpu-fastloop; N=9ca3c3f8
export BINDIR=$P2/bin2/9ca3c3f8-multi EXPECT_COMMIT=$N REFDIR=$P2/bin2/9560b8db REFCOMMIT=9560b8db ALLOW_NO_EMBED=1 IDENT=0 PINDEV=${PINDEV:-0}
export FL_IDEAS=$FL/ideas.tsv FL_SOLE=1 SAMPLE=pbmc1k_v3
if [ "${PILOT:-0}" = 1 ] && [ -n "${PILOT_NODE:-}" ] && [ -n "${PILOT_JOB:-}" ] && [ "${FL_NODE_OK:-0}" != 1 ]; then
  other=$(squeue -w $PILOT_NODE -h -o "%i %u %j" | grep -v "^$PILOT_JOB ")
  [ -n "$other" ] && { echo "FL_SOLE REFUSED: other jobs on $PILOT_NODE; the shared-index cells need the node to themselves:"; echo "$other"; exit 3; }
fi
export POINTS="k1t8s1r2@shidx+stepsauto+maponly#b1 k1t8s1r2@wshrink+shidx+maponly#b1 k1t8s1r2@stepsauto+maponly#b1 k1t8s1r2@wshrink+stepsauto+maponly#b1 k1t8s1r2@wshrink+shidx+stepsauto+maponly#b1 k1t8s1r2@maponly#b1 k1t8s1r2@wshrink+maponly#b1 k1t8s1r2@shidx+maponly#b1 k1t8s1r2@shidx+maponly#b2 k1t8s1r2@wshrink+maponly#b2 k1t8s1r2@maponly#b2 k1t8s1r2@wshrink+shidx+stepsauto+maponly#b2 k1t8s1r2@wshrink+stepsauto+maponly#b2 k1t8s1r2@stepsauto+maponly#b2 k1t8s1r2@wshrink+shidx+maponly#b2 k1t8s1r2@shidx+stepsauto+maponly#b2"
bash $FL/tp_gc.sbatch; rc=$?
RES=$W/jobs/gpu-integ/results/tp-${RUNID:-$SLURM_JOB_ID}
$W/envs/pbench/bin/python3 $FL/analyze.py --res $RES --ideas wshrink,shidx,stepsauto --drop-first 1 --metric held --theta 0.03 > $RES/fastloop.md 2>&1
cat $RES/fastloop.md
exit $rc
